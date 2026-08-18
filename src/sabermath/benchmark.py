from dataclasses import dataclass
from typing import Callable, Literal, get_args
from pathlib import Path
import json
import re

import numpy as np
from tqdm import tqdm
from datasets import Dataset

from sabermath.data import load_data
from sabermath.metrics import compute_ndcg_at_k
from sabermath.schemas import (
    Task,
    Report,
    TaskResult,
    Branch,
    BranchResult,
    DCGVariant,
)
from sabermath.processors import (
    ModelProcessor,
    EmbeddingProcessor,
    SentenceTransformersProcessor as STProcessor,
    VLLMProcessor,
    UnknownProcessor,
)


ALL_TASKS = get_args(Task)


def _ensure_dir(file_path: str | Path) -> None:
    path = Path(file_path)
    parent = path.parent
    if parent != Path():
        parent.mkdir(parents=True, exist_ok=True)


def _load_checkpoint(path: str | Path | None) -> dict[int, float | None]:
    """Load a per-query nDCG checkpoint written by a previous (possibly
    interrupted) evaluate_task() run. Returns {} if there is none yet, or if
    the file is missing/corrupt (e.g. killed mid-write before the atomic
    rename below could apply) - a bad checkpoint should never crash a run,
    only cost it a restart from scratch."""
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {int(k): v for k, v in raw.items()}


def _save_checkpoint(
    path: str | Path | None, all_ndcgs_by_idx: dict[int, float | None]
) -> None:
    """Persist per-query nDCGs computed so far. Writes to a temp file and
    renames it into place so a job killed mid-write can never leave a
    truncated/corrupt checkpoint behind."""
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps({str(k): v for k, v in all_ndcgs_by_idx.items()}))
    tmp.replace(p)


def _naive_type_check(obj, module: str, name: str) -> bool:
    t = type(obj)
    return t.__module__ == module and t.__name__ == name


def _make_processor(model, use_vllm: bool, init_kwargs: dict, no_init: bool):
    if isinstance(model, str):
        if no_init:
            init_kwargs["_xforcenini"] = 1
        if use_vllm:
            return VLLMProcessor.from_huggingface(model, **init_kwargs)
        return STProcessor.from_huggingface(model, **init_kwargs)

    if use_vllm:
        raise RuntimeError(
            "Flag 'use_vllm' is only valid when loading a "
            "model from HuggingFace path"
        )

    if isinstance(model, ModelProcessor):
        if init_kwargs:
            raise RuntimeError(
                "Processor has already been initialized, so "
                "'init_kwargs' is invalid."
            )
        return model

    if _naive_type_check(model, "vllm", "LLM"):
        return VLLMProcessor(model, **init_kwargs)

    if _naive_type_check(model, "sentence_transformers", "SentenceTransformer"):
        return STProcessor(model, **init_kwargs)

    return UnknownProcessor(model, **init_kwargs)


def _get_branch_row_idxs(
    targets: Dataset,
    branch: Branch,
) -> list[int]:
    branches_all = targets["domains"]
    return [i for i, branches in enumerate(branches_all) if branch in branches]


@dataclass(frozen=True)
class TaskSettings:
    queries_ds: Dataset
    documents_ds: Dataset
    processor: ModelProcessor
    k: int
    dcg_variant: Literal["linear", "exponent"] | None
    show_progress_bar: bool
    scores_kwargs: dict


def evaluate_task(
    name: Task,
    settings: TaskSettings,
    return_ndcgs: bool = False,
    checkpoint_path: str | Path | None = None,
    on_progress: Callable[[], None] | None = None,
) -> TaskResult:
    """Evaluate one task.

    If `checkpoint_path` is given, per-query nDCGs are written to it after
    every query and reloaded from it on entry, so a run interrupted partway
    (e.g. a SLURM job hitting its wall-clock limit) can be resumed by calling
    evaluate_task() again with the same checkpoint_path instead of losing all
    progress and restarting from query 0. This matters most for slow,
    generative rerankers (e.g. rank1-32b) evaluated over hundreds of queries.
    Do not reuse a checkpoint path across a differently-sized/ordered
    queries_ds (e.g. after changing a query subsample) - the checkpoint is
    keyed by query position, not by query content.

    If `on_progress` is given, it's called (no args) right after every
    freshly-computed (not resumed-and-skipped) query's checkpoint write -
    e.g. to periodically snapshot partial results to a separate location, or
    push them somewhere else. What it does is entirely up to the caller;
    this file has no opinion on it.
    """
    queries_ds = settings.queries_ds
    documents_ds = settings.documents_ds
    processor = settings.processor
    k = settings.k
    dcg_variant = settings.dcg_variant
    show_progress_bar = settings.show_progress_bar
    scores_kwargs = settings.scores_kwargs

    query_version = "full" if name == "full-full" else "statement"
    document_version = "statement" if name == "statement-statement" else "full"

    queries = transform(queries_ds, query_version)

    all_ndcgs_by_idx: dict[int, float | None] = _load_checkpoint(checkpoint_path)
    ndcg_by_query_idx: dict[int, float] = {
        i: v for i, v in all_ndcgs_by_idx.items() if v is not None
    }

    if all_ndcgs_by_idx:
        print(
            f"[~] Resuming task '{name}' from checkpoint: "
            f"{len(all_ndcgs_by_idx)}/{len(queries)} queries already scored."
        )

    for i, query in tqdm(
        enumerate(queries),
        total=len(queries),
        disable=not show_progress_bar,
    ):
        if i in all_ndcgs_by_idx:
            continue  # already scored - loaded from checkpoint

        doc_ids = list(queries_ds[i]["candidates"])
        relevance_scores = list(queries_ds[i]["relevance_scores"])

        documents = transform(documents_ds, document_version, doc_ids)
        model_scores = processor.get_scores(
            query,
            documents,
            show_progress_bar=False,
            **scores_kwargs,
        )

        if model_scores is None:
            all_ndcgs_by_idx[i] = None
        else:
            model_ranked_local_idxs = np.argsort(-np.asarray(model_scores))

            model_ranked_relevance_scores = np.asarray(relevance_scores)[
                model_ranked_local_idxs
            ]
            ndcg = compute_ndcg_at_k(
                model_ranked_relevance_scores, k=k, variant=dcg_variant
            )
            all_ndcgs_by_idx[i] = ndcg
            ndcg_by_query_idx[i] = ndcg

        # Written after every query so a killed job loses at most the query
        # currently in flight, never the whole task's progress.
        _save_checkpoint(checkpoint_path, all_ndcgs_by_idx)

        if on_progress is not None:
            on_progress()

    all_ndcgs = [all_ndcgs_by_idx.get(i) for i in range(len(queries))]
    ndcgs = np.asarray(list(ndcg_by_query_idx.values()))
    ndcg_at_k = float(ndcgs.mean()) if ndcgs.size > 0 else 0.0

    BRANCHES = get_args(Branch)
    branch_results: list[BranchResult] = []

    for branch in BRANCHES:
        idxs = _get_branch_row_idxs(queries_ds, branch)
        branch_ndcgs = [
            ndcg_by_query_idx[idx] for idx in idxs if idx in ndcg_by_query_idx
        ]
        branch_ndcg_at_k = (
            float(np.mean(branch_ndcgs)) if len(branch_ndcgs) > 0 else 0.0
        )
        br = BranchResult(branch, branch_ndcg_at_k)
        branch_results.append(br)

    report = TaskResult(
        name,
        ndcg_at_k,
        branch_results,
    )

    if return_ndcgs:
        return report, all_ndcgs
    else:
        return report


def transform(
    ds: Dataset,
    version: Literal["statement", "full"],
    idxs: list[int] | None = None,
) -> list[str]:
    if idxs is not None:
        ds = ds.select(idxs)

    if version == "full":
        statements = list(ds["problem"])
        solutions = list(ds["solution"])

        return [
            f"Problem: {statement}\n\nSolution: {solution}"
            for statement, solution in zip(statements, solutions)
        ]

    return list(ds["problem"])


def extract_npz_cache(query_ds, document_ds, npz_file) -> dict:
    q_st_vects = npz_file["target_statement_vectors"]
    q_full_vects = npz_file["target_full_vectors"]
    doc_st_vects = npz_file["candidate_statement_vectors"]
    doc_full_vects = npz_file["candidate_full_vectors"]

    cache = {}

    for statement, vect in zip(query_ds["problem"], q_st_vects):
        cache[statement] = vect
    for statement, solution, vect in zip(
        query_ds["problem"], query_ds["solution"], q_full_vects
    ):
        full = f"Problem: {statement}\n\nSolution: {solution}"
        cache[full] = vect

    for statement, vect in zip(document_ds["problem"], doc_st_vects):
        cache[statement] = vect
    for statement, solution, vect in zip(
        document_ds["problem"], document_ds["solution"], doc_full_vects
    ):
        full = f"Problem: {statement}\n\nSolution: {solution}"
        cache[full] = vect

    return cache


def evaluate(
    model,
    tasks: list[Task] | None = None,
    k: int = 10,
    *,
    # Init na Run Config
    dcg_variant: DCGVariant = "exponent",
    use_vllm: bool = False,
    init_kwargs: dict | None = None,
    scores_kwargs: dict | None = None,
    # Resume Config: if set, per-query nDCGs are checkpointed to
    # "{checkpoint_dir}/{task}.json" after every query and reloaded from
    # there on start, so an interrupted run (e.g. a SLURM job timeout) can be
    # resumed instead of restarting from query 0. Don't reuse a
    # checkpoint_dir across runs with a different queries_ds (e.g. after
    # changing --n / a query subsample) - checkpoints are keyed by query
    # position, not content.
    checkpoint_dir: str | Path | None = None,
    on_progress: Callable[[], None] | None = None,
    # Printing Config
    verbose: bool = True,
    show_progress_bars: bool = True,
    #       ! USE WITH CAUTION !
    # ARGUMENTS BELOW ARE FOR DEV PURPOSES
    # Cache Config
    cache_path: str | None = None,
    allow_export_cache: bool = True,
    allow_load_cache: bool = True,
    # Benchmark Data Setting (Queries & Documents)
    queries: Dataset | None = None,
    documents: Dataset | None = None,
    # Direct Input/Output
    return_ndcgs: bool = False,
    no_init: bool = False,
) -> Report:
    if tasks is None:
        tasks = ALL_TASKS

    for task in tasks:
        if task not in ALL_TASKS:
            raise ValueError(f"Invalid task: {task}")

    if dcg_variant not in get_args(DCGVariant):
        raise ValueError(f"Invalid DCG variant: {dcg_variant}")

    def vprint(text: str) -> None:
        if verbose:
            print(text)

    export_cache = allow_export_cache and cache_path is not None
    load_cache = allow_load_cache and cache_path is not None

    if load_cache or export_cache:
        _ensure_dir(cache_path)

    init_kwargs = init_kwargs or {}
    scores_kwargs = scores_kwargs or {}

    vprint("[~] Loading model...")

    processor = _make_processor(model, use_vllm, init_kwargs, no_init)

    vprint("[+] Model loaded.")

    task_results = []

    tasks = set(tasks) & set(ALL_TASKS)

    if queries is None or documents is None:
        vprint(
            f"[~] Loading data for task{'' if len(task) == 1 else 's'} "
            f"\"{','.join(tasks)}\"..."
        )

        queries, documents = load_data()

    vprint(f"[+] Loaded {len(queries)} queries.")

    if load_cache:
        try:
            vprint(f"[~] Loading cache from {cache_path}...")
            if cache_path.endswith(".npz"):
                npz_file = np.load(cache_path)
                cache_data = extract_npz_cache(queries, documents, npz_file)
                processor.import_cache(cache_data, is_path=False)
            else:
                processor.import_cache(cache_path)
            vprint("[+] Cache loaded.")
        except Exception as e:
            vprint(f"[-] Failed to load cache: {e}")

    settings = TaskSettings(
        queries_ds=queries,
        documents_ds=documents,
        processor=processor,
        k=k,
        dcg_variant=dcg_variant,
        show_progress_bar=show_progress_bars,
        scores_kwargs=scores_kwargs,
    )

    ndcgs = {}

    def _checkpoint_path(task_name: str) -> str | None:
        if checkpoint_dir is None:
            return None
        return str(Path(checkpoint_dir) / f"{task_name}.json")

    if "statement-statement" in tasks:
        vprint('[~] Evaluating on task "statement vs. statement"...')
        st_st_res, st_st_ndcgs = evaluate_task(
            "statement-statement",
            settings,
            True,
            checkpoint_path=_checkpoint_path("statement-statement"),
            on_progress=on_progress,
        )
        ndcgs["statement-statement"] = st_st_ndcgs
        ndcg_at_k = st_st_res.ndcg_at_k
        vprint(f"[+] Statement-statement nDCG@{k} ({dcg_variant}): {ndcg_at_k}")
        task_results.append(st_st_res)

    if "statement-full" in tasks:
        vprint('[~] Evaluating on task "statement vs. full statement + solution"...')
        st_fl_res, st_fl_ndcgs = evaluate_task(
            "statement-full",
            settings,
            True,
            checkpoint_path=_checkpoint_path("statement-full"),
            on_progress=on_progress,
        )
        ndcgs["statement-full"] = st_fl_ndcgs
        ndcg_at_k = st_fl_res.ndcg_at_k
        vprint(f"[+] Statement-full nDCG@{k} ({dcg_variant}): {ndcg_at_k}")
        task_results.append(st_fl_res)

    if "full-full" in tasks:
        vprint(
            '[~] Evaluating on task "full problem + solution vs. '
            'full problem + solution"...'
        )
        fl_fl_res, fl_fl_ndcgs = evaluate_task(
            "full-full",
            settings,
            True,
            checkpoint_path=_checkpoint_path("full-full"),
            on_progress=on_progress,
        )
        ndcgs["full-full"] = fl_fl_ndcgs
        ndcg_at_k = fl_fl_res.ndcg_at_k
        vprint(f"[+] Full-full nDCG@{k} ({dcg_variant}): {ndcg_at_k}")
        task_results.append(fl_fl_res)

    del queries
    del documents

    report = Report(
        model=processor.model or "unknown",
        processor=processor.processor or "unknown",
        dcg_variant=dcg_variant,
        k=k,
        tasks=task_results,
    )

    if return_ndcgs:
        return report, ndcgs
    else:
        return report
