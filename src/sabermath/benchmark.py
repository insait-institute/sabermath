from dataclasses import dataclass
from typing import Literal, get_args
from pathlib import Path
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
) -> TaskResult:
    queries_ds = settings.queries_ds
    documents_ds = settings.documents_ds
    processor = settings.processor
    k = settings.k
    dcg_variant = settings.dcg_variant
    show_progress_bar = settings.show_progress_bar
    scores_kwargs = settings.scores_kwargs

    ndcgs = []
    all_ndcgs = []
    ndcg_by_query_idx = {}

    query_version = "full" if name == "full-full" else "statement"
    document_version = "statement" if name == "statement-statement" else "full"

    queries = transform(queries_ds, query_version)

    for i, query in tqdm(
        enumerate(queries),
        total=len(queries),
        disable=not show_progress_bar,
    ):
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
            all_ndcgs.append(None)
            continue

        model_ranked_local_idxs = np.argsort(-np.asarray(model_scores))

        model_ranked_relevance_scores = np.asarray(relevance_scores)[
            model_ranked_local_idxs
        ]
        ndcg = compute_ndcg_at_k(
            model_ranked_relevance_scores, k=k, variant=dcg_variant
        )
        ndcgs.append(ndcg)
        all_ndcgs.append(ndcg)
        ndcg_by_query_idx[i] = ndcg

    ndcgs = np.asarray(ndcgs)
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

    if "statement-statement" in tasks:
        vprint('[~] Evaluating on task "statement vs. statement"...')
        st_st_res, st_st_ndcgs = evaluate_task("statement-statement", settings, True)
        ndcgs["statement-statement"] = st_st_ndcgs
        ndcg_at_k = st_st_res.ndcg_at_k
        vprint(f"[+] Statement-statement nDCG@{k} ({dcg_variant}): {ndcg_at_k}")
        task_results.append(st_st_res)

    if "statement-full" in tasks:
        vprint('[~] Evaluating on task "statement vs. full statement + solution"...')
        st_fl_res, st_fl_ndcgs = evaluate_task("statement-full", settings, True)
        ndcgs["statement-full"] = st_fl_ndcgs
        ndcg_at_k = st_fl_res.ndcg_at_k
        vprint(f"[+] Statement-full nDCG@{k} ({dcg_variant}): {ndcg_at_k}")
        task_results.append(st_fl_res)

    if "full-full" in tasks:
        vprint(
            '[~] Evaluating on task "full problem + solution vs. '
            'full problem + solution"...'
        )
        fl_fl_res, fl_fl_ndcgs = evaluate_task("full-full", settings, True)
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
