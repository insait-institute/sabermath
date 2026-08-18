"""Evaluate the SABER-Math rerankers that need custom scoring logic (rank1,
Qwen3-Reranker, GTE-ModernColBERT, Reason-ModernColBERT, ReasonIR, SPLADE),
plus Qwen3-Embedding-8B (which already runs through the generic HuggingFace
path), across all three tasks (statement-statement, statement-full,
full-full).

Each model/task run is checkpointed after every query to
"<save-to>/.checkpoints/<model>/<subset>/<task>.json" (see
sabermath.benchmark.evaluate's checkpoint_dir). If a job is killed (e.g. a
SLURM wall-clock timeout - the exact failure mode that killed 6 of 7 past
rank1-32b attempts in rag-math-test/running-rerankers/), simply re-running
the identical command resumes from the last completed query instead of
starting over from zero.

Each model's result is saved as its own JSON file with the same
{"domains": [...], "reports": {<model>: {..., "ndcgs_by_task": {...}}}}
shape that testing/test_models.py produces, so
testing/compute_confidence_intervals.py can be pointed at the output
directory (or individual files) to get per-task, per-domain 95% CIs (normal
+ bootstrap) for free - no separate stats code needed for these models.

IMPORTANT - dependency isolation: rank1 (vllm), Qwen3-Reranker
(transformers>=4.51.0), ReasonIR (transformers==4.47.1, per its model card
pin), GTE/Reason-ModernColBERT (pylate) and SPLADE (sentence-transformers'
SparseEncoder) have overlapping-but-conflicting package requirements (see
running-rerankers/*.yml in rag-math-test for the environments that were
already verified to work per model). Do not install all of them into one
shared venv/conda env and run the full --models list in a single process;
instead run this script once per model (or per compatible model family) from
that model's own environment, e.g.:

    python scripts/run_rerankers.py --models rank1-32b
    python scripts/run_rerankers.py --models reasonir-8b
    python scripts/run_rerankers.py --models gte-moderncolbert reason-moderncolbert

Each invocation still runs its one (or few) models in an isolated subprocess
so a crash/OOM doesn't corrupt already-written results.

Usage:
    # One model, all 3 tasks
    python scripts/run_rerankers.py --models splade-code-8b

    # One model, one task
    python scripts/run_rerankers.py --models rank1-32b --task statement-full

    # Smoke test on a random 20-query subset before committing to a full,
    # possibly multi-hour run (rank1-32b in particular - see
    # running-rerankers/ in rag-math-test for why that matters: every past
    # attempt at a full 1000-query run there was killed by the SLURM time
    # limit before finishing).
    python scripts/run_rerankers.py --models rank1-32b --n 20
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path
from typing import get_args

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sabermath import evaluate
from sabermath.data import load_data
from sabermath.processors import (
    ColBERTProcessor,
    Qwen3RerankerProcessor,
    Rank1Processor,
    ReasonIRProcessor,
    SentenceTransformersProcessor as STProcessor,
    SpladeProcessor,
)
from sabermath.schemas import Branch, Task

BRANCHES = list(get_args(Branch))

ALL_TASKS = list(get_args(Task))

# Models needing a custom ModelProcessor (not resolvable from a bare HF name).
# rank1-32b's Processor CAN take tensor_parallel_size > 1 (vLLM generation,
# actually GPU-hungry at 32B), but a direct diagnostic re-run confirmed that
# doing so (tensor_parallel_size=2, matched to a 2-GPU allocation) silently
# corrupts its relevance scores - see the note in Rank1Processor._init().
# Hardcoded to 1 here regardless of what --tensor-parallel-size/-gpus is
# passed, until/unless that's re-investigated (e.g. via data parallelism -
# two independent single-GPU engines splitting the query set - instead of
# tensor parallelism). The other models below are single-process HF/pylate
# calls that don't shard across GPUs at all, so --tensor-parallel-size is
# simply ignored for them too (see _run_one).
RADER_14B_MODEL_NAME = "Raderspace/RaDeR_Qwen25-14B_NuminaMath_MATH_allquerytypes"


def _build_rader_14b_processor():
    """Raderspace/RaDeR_Qwen25-14B_...: ships NO sentence-transformers module
    config at all (no modules.json / config_sentence_transformers.json /
    1_Pooling/config.json - confirmed by listing the actual repo). Loading
    it via the generic `SentenceTransformer(model_name)` path therefore
    falls back to sentence-transformers' own default (mean pooling) - but
    the model's own card explicitly says to serve it via vLLM with
    `pooling_type=LAST, normalize=true`. Confirmed the hard way: a full run
    under that wrong mean-pooling fallback scored statement-full
    nDCG@10=0.4126 against a reference value of 0.488, a far larger gap than
    every other model in this pipeline - and chunking (the earlier OOM fix)
    and precision were both ruled out as the cause (only 0.32% of documents
    even exceed the 2048-token chunk cutoff). Building the
    Transformer+Pooling stack explicitly instead of relying on
    auto-detection, so lasttoken pooling actually gets used.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer import modules

    transformer = modules.Transformer(
        RADER_14B_MODEL_NAME,
        model_kwargs={"trust_remote_code": True, "torch_dtype": "auto"},
        config_kwargs={"trust_remote_code": True},
    )
    pooling = modules.Pooling(
        transformer.get_embedding_dimension(), pooling_mode="lasttoken"
    )
    st = SentenceTransformer(modules=[transformer, pooling])
    return STProcessor(st, RADER_14B_MODEL_NAME)


CUSTOM_MODEL_BUILDERS = {
    "rank1-32b": lambda tp: Rank1Processor(tensor_parallel_size=1),
    "qwen3-reranker-8b": lambda tp: Qwen3RerankerProcessor(),
    "gte-moderncolbert": lambda tp: ColBERTProcessor("lightonai/GTE-ModernColBERT-v1"),
    "reason-moderncolbert": lambda tp: ColBERTProcessor("lightonai/Reason-ModernColBERT"),
    "reasonir-8b": lambda tp: ReasonIRProcessor(),
    "splade-code-8b": lambda tp: SpladeProcessor(),
    "rader-14b": lambda tp: _build_rader_14b_processor(),
}

# Per-model scores_kwargs for CUSTOM_MODEL_BUILDERS entries (forwarded to
# processor.get_scores(**scores_kwargs) - see _run_one). rader-14b keeps the
# chunk_to_context fix from the earlier OOM investigation (still a real,
# separate issue from the pooling bug above - only 0.32% of documents exceed
# 2048 tokens, but a 14B model can still spike on those few).
CUSTOM_MODEL_SCORES_KWARGS = {
    "rader-14b": {
        "chunk_to_context": True,
        "context_length": 2048,
        "batch_size": 4,
    },
}

# Already supported through evaluate()'s generic HuggingFace path - no custom
# processor needed, just the right (model name, use_vllm) pair. tensor_parallel_size
# is forwarded via init_kwargs since VLLMProcessor.from_huggingface passes
# **kwargs straight through to vllm.LLM(...).
#
# reason-embed-qwen3-8b / diver-retriever-4b are plain bi-encoder embedding
# models (confirmed via each model's own card/paper - neither is a
# cross-encoder), so - like qwen3-embedding-8b - they need no custom
# Processor: use_vllm defaults to False here, which routes them through
# SentenceTransformersProcessor's auto-detected module config (see
# _make_processor in benchmark.py), the same cosine-similarity bi-encoder
# path already used for reasonir-8b. (rader-14b is ALSO a bi-encoder, same
# family, but its HF repo ships no sentence-transformers module config to
# auto-detect at all, which silently produced the wrong pooling mode - see
# _build_rader_14b_processor for why it needs a CUSTOM_MODEL_BUILDERS entry
# instead of a place here.)
# trust_remote_code=True since these are community fine-tunes, not
# natively-registered architectures, mirroring every other custom-vendor
# model already in this pipeline (Rank1Processor, ReasonIRProcessor, etc.).
#
# model_kwargs={"torch_dtype": "auto"}: without it, sentence-transformers
# loads in fp32 by default. Confirmed the hard way for rader-14b (14B params
# -> ~56GiB in fp32 before any activations) - OOM'd on a 140GiB H200 partway
# through a run, mostly PyTorch allocator fragmentation from many
# variable-shaped encode() calls, not the weights themselves. "auto" mirrors
# the working convention already used for cached models in
# experiments/confidence_intervals/confidence.py (there hardcoded to bf16 -
# "auto" instead defers to each model's own declared config dtype, since
# these models' configs haven't been individually verified to all
# specifically want bf16).
GENERIC_MODELS = {
    "qwen3-embedding-8b": {"model": "Qwen/Qwen3-Embedding-8B", "use_vllm": True},
    "reason-embed-qwen3-8b": {
        "model": "hanhainebula/reason-embed-qwen3-8b-0928",
        "init_kwargs": {
            "trust_remote_code": True,
            "model_kwargs": {"torch_dtype": "auto"},
        },
    },
    "diver-retriever-4b": {
        "model": "AQ-MedAI/Diver-Retriever-4B",
        "init_kwargs": {
            "trust_remote_code": True,
            "model_kwargs": {"torch_dtype": "auto"},
        },
    },
}
# rader-14b moved out of GENERIC_MODELS and into CUSTOM_MODEL_BUILDERS below -
# see _build_rader_14b_processor for why.
GENERIC_MODELS_USE_TP = {"qwen3-embedding-8b"}  # vLLM-backed -> tp actually used

ALL_MODEL_KEYS = list(CUSTOM_MODEL_BUILDERS) + list(GENERIC_MODELS)


def _merge_results(filepath: Path, model_key: str, new_output: dict) -> dict:
    """Merge new_output into whatever's already saved at filepath (if
    anything), so writing one --task slice never erases another
    already-saved --task slice for the same model+subset (e.g. running
    statement-full and full-full for the same model as two separate
    submissions). Tasks are merged by name - re-running the same task
    overwrites just that task's entry with fresh data. A run that crashed
    entirely (no tasks computed) never overwrites previously-successful
    data for the same file."""
    if not filepath.exists():
        return new_output

    try:
        existing = json.loads(filepath.read_text())
    except (json.JSONDecodeError, OSError):
        return new_output

    existing_report = existing.get("reports", {}).get(model_key)
    new_report = new_output["reports"][model_key]

    if existing_report is None:
        return new_output
    if "error" in new_report:
        # This attempt failed - keep whatever was already saved rather than
        # replacing good data with an error.
        return existing if "tasks" in existing_report else new_output
    if "error" in existing_report:
        return new_output

    existing_tasks = {t["task"]: t for t in existing_report.get("tasks", [])}
    for t in new_report.get("tasks", []):
        existing_tasks[t["task"]] = t
    existing_report["tasks"] = list(existing_tasks.values())

    existing_ndcgs = existing_report.get("ndcgs_by_task", {})
    existing_ndcgs.update(new_report.get("ndcgs_by_task", {}))
    existing_report["ndcgs_by_task"] = existing_ndcgs

    for key in ("model", "processor", "dcg_variant", "k"):
        if key in new_report:
            existing_report[key] = new_report[key]

    domains = new_output.get("domains") or existing.get("domains")
    return {"domains": domains, "reports": {model_key: existing_report}}


def _partial_report_from_checkpoints(
    model_key: str, checkpoint_dir: Path, tasks: list[str], domains: list[list[str]]
) -> dict:
    """Rebuild an in-progress snapshot of this run's tasks straight from
    their checkpoint files (each {query_idx: ndcg_or_null}), without waiting
    for evaluate() to finish. Same {"task", "ndcg_at_k", "branches"} shape as
    a real TaskResult, plus "n_done"/"n_total" so a partial write is never
    mistaken for a finished one at a glance."""
    n_total = len(domains)
    tasks_out = []
    ndcgs_by_task = {}

    for task in tasks:
        ckpt_path = Path(checkpoint_dir) / f"{task}.json"
        if not ckpt_path.exists():
            continue
        try:
            raw = json.loads(ckpt_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        by_idx = {int(k): v for k, v in raw.items()}
        all_ndcgs = [by_idx.get(i) for i in range(n_total)]
        ndcgs_by_task[task] = all_ndcgs

        valid = [v for v in all_ndcgs if v is not None]
        ndcg_at_k = float(sum(valid) / len(valid)) if valid else 0.0

        branches_out = []
        for branch in BRANCHES:
            branch_vals = [
                by_idx[i]
                for i in range(n_total)
                if by_idx.get(i) is not None and branch in domains[i]
            ]
            branch_ndcg = float(sum(branch_vals) / len(branch_vals)) if branch_vals else 0.0
            branches_out.append({"branch": branch, "ndcg_at_k": branch_ndcg})

        tasks_out.append(
            {
                "task": task,
                "ndcg_at_k": ndcg_at_k,
                "branches": branches_out,
                "n_done": len(valid),
                "n_total": n_total,
            }
        )

    report_dict = {
        "model": model_key,
        "processor": None,
        "dcg_variant": "exponent",
        "k": 10,
        "tasks": tasks_out,
        "ndcgs_by_task": ndcgs_by_task,
    }
    return {"domains": domains, "reports": {model_key: report_dict}}


def _sync_back_if_needed(save_dir: str) -> None:
    """Push save_dir back to the canonical host over rsync, mirroring
    _common.sh's sync_back_to_canonical() - but callable from inside the
    query loop (via on_progress), not just once at process exit. A no-op
    unless NEEDS_REGION_SYNC=1 is set in the environment (i.e. this node's
    home storage isn't the canonical copy - see scripts/rerankers/*.slurm).
    Failures are logged, never raised - a flaky sync must not crash a run
    that's otherwise making progress."""
    if os.environ.get("NEEDS_REGION_SYNC") != "1":
        return
    host = os.environ.get("SABERMATH_CANONICAL_HOST", "hala")
    path = os.environ.get("SABERMATH_CANONICAL_PATH", "/home/maria_drencheva/sabermath")
    try:
        subprocess.run(
            ["rsync", "-auz", f"{save_dir}/", f"{host}:{path}/{save_dir}/"],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except Exception as e:
        print(f"[!!] Periodic sync-back to {host} failed (will retry next time): {e}")


def _run_one(
    model_key: str,
    save_dir: str,
    tasks: list[str],
    n: int | None,
    seed: int,
    tensor_parallel_size: int,
    progress_every: int,
) -> None:
    # Keyed by (model, query-subset) so a checkpoint - or a saved result -
    # from one --n/--seed combination is never mistaken for another subset:
    # evaluate_task() checkpoints by query *position*, not content, and a
    # 20-query smoke test must never land in the same file as a full run.
    subset_key = f"n{n}_seed{seed}" if n is not None else "full"
    suffix = "" if subset_key == "full" else f"__{subset_key}"
    filepath = Path(save_dir) / f"{model_key}{suffix}.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(save_dir) / ".checkpoints" / model_key / subset_key

    if tensor_parallel_size > 1 and model_key not in GENERIC_MODELS_USE_TP:
        reason = (
            "is pinned to tensor_parallel_size=1 (confirmed to corrupt scores "
            "otherwise - see Rank1Processor._init())"
            if model_key == "rank1-32b"
            else "doesn't shard across GPUs in this framework"
        )
        print(
            f"[~] {model_key} {reason} - "
            f"--tensor-parallel-size {tensor_parallel_size} will be ignored "
            f"and only 1 GPU actually used."
        )

    success = True

    try:
        queries, documents = load_data()
        domains = list(queries["domains"])

        if n is not None:
            rng = random.Random(seed)
            idxs = sorted(rng.sample(range(len(queries)), min(n, len(queries))))
            queries = queries.select(idxs)
            domains = [domains[i] for i in idxs]

        # Every `progress_every` freshly-scored queries (across all tasks in
        # this run, combined - not per-task), snapshot current progress
        # straight from the checkpoint files into the real results file (via
        # _merge_results, so it never clobbers an already-finished sibling
        # task) and push it to the canonical host if this node isn't it. So
        # a hard kill with no chance to run an exit handler loses at most
        # `progress_every` queries' worth of results, not the whole run.
        progress_counter = {"n": 0}

        def on_progress() -> None:
            if progress_every <= 0:
                return
            progress_counter["n"] += 1
            if progress_counter["n"] % progress_every != 0:
                return
            partial = _partial_report_from_checkpoints(
                model_key, checkpoint_dir, tasks, domains
            )
            merged = _merge_results(filepath, model_key, partial)
            filepath.write_text(json.dumps(merged, indent=2))
            _sync_back_if_needed(save_dir)

        if model_key in CUSTOM_MODEL_BUILDERS:
            model = CUSTOM_MODEL_BUILDERS[model_key](tensor_parallel_size)
            report, ndcgs = evaluate(
                model,
                tasks=tasks,
                queries=queries,
                documents=documents,
                return_ndcgs=True,
                checkpoint_dir=checkpoint_dir,
                on_progress=on_progress,
                scores_kwargs=CUSTOM_MODEL_SCORES_KWARGS.get(model_key, {}),
            )
        else:
            spec = dict(GENERIC_MODELS[model_key])
            model_name = spec.pop("model")
            # Start from this model's own fixed init kwargs (e.g.
            # trust_remote_code=True), then layer tensor_parallel_size on top
            # for the vLLM-backed ones - never the reverse, so a model-specific
            # kwarg can't accidentally get dropped by the tp branch.
            init_kwargs = dict(spec.pop("init_kwargs", {}))
            if model_key in GENERIC_MODELS_USE_TP and tensor_parallel_size > 1:
                init_kwargs["tensor_parallel_size"] = tensor_parallel_size
            # scores_kwargs flows to processor.get_scores(**scores_kwargs) -
            # e.g. rader-14b's chunk_to_context/context_length/batch_size, to
            # cap peak activation memory per encode() call (see the note on
            # its GENERIC_MODELS entry above).
            scores_kwargs = dict(spec.pop("scores_kwargs", {}))
            report, ndcgs = evaluate(
                model_name,
                tasks=tasks,
                queries=queries,
                documents=documents,
                return_ndcgs=True,
                checkpoint_dir=checkpoint_dir,
                on_progress=on_progress,
                init_kwargs=init_kwargs,
                scores_kwargs=scores_kwargs,
                **spec,
            )

        report_dict = report.to_dict()
        report_dict["ndcgs_by_task"] = ndcgs
        output = {"domains": domains, "reports": {model_key: report_dict}}

        print(f"\n=== {model_key} ===")
        for task in report_dict["tasks"]:
            print(f"  {task['task']:<20} nDCG@10 = {task['ndcg_at_k']:.4f}")
            for branch in task["branches"]:
                print(f"      {branch['branch']:<22} {branch['ndcg_at_k']:.4f}")

    except Exception as e:
        success = False
        tb = traceback.format_exc()
        output = {
            "domains": None,
            "reports": {
                model_key: {"model": model_key, "error": str(e), "traceback": tb}
            },
        }
        # Print the full traceback, not just str(e) - confirmed the hard way
        # that a bare message ("type 'array.array' is not subscriptable")
        # isn't enough to find the actual failure site without reproducing
        # it separately by hand. Goes to this job's .error log.
        print(f"\n[!!] {model_key} failed:")
        print(tb)

    output = _merge_results(filepath, model_key, output)
    filepath.write_text(json.dumps(output, indent=2))
    print(f"[+] Wrote {filepath}")
    _sync_back_if_needed(save_dir)

    if not success:
        # Propagate failure to main()'s exit-code check - previously this
        # process always exited 0 even when the model run itself failed
        # (only a hard crash before reaching this function's own try/except
        # would have been caught as a failure), so "N/N succeeded" was not
        # trustworthy. Comes after the write+sync above so the error record
        # (and traceback) still lands on disk either way.
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODEL_KEYS,
        default=None,
        help="Subset of models to run (default: all 7 - see the dependency "
        "isolation warning in this script's module docstring before doing "
        "that in a single shared environment).",
    )
    parser.add_argument(
        "--task",
        choices=ALL_TASKS,
        default=None,
        help="Run only this task (default: all 3, in one process per model)",
    )
    parser.add_argument("--save-to", type=str, default="results/rerankers")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Evaluate only a random n-query subset (for smoke-testing before "
        "committing to a full ~1000-query run)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="GPUs to shard across for vLLM-backed models (rank1-32b, "
        "qwen3-embedding-8b). Ignored (with a warning) for the other 5 "
        "models, which are single-GPU only in this framework.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Every N freshly-scored queries (combined across all tasks in "
        "this run), snapshot current progress into the real results file "
        "and push it to the canonical host if this node isn't it (default: "
        "10). Lower it for extra safety on very slow models (e.g. rank1-32b) "
        "at the cost of more frequent rsync overhead; 0 disables snapshotting "
        "entirely (falls back to writing only when the run finishes/fails, "
        "the old behavior).",
    )
    args = parser.parse_args()

    models = args.models or ALL_MODEL_KEYS
    tasks = [args.task] if args.task else ALL_TASKS

    ctx = mp.get_context("spawn")
    failures = []

    for i, model_key in enumerate(models, start=1):
        print("\n" + "#" * 60)
        print(f"# Running {model_key} ({i}/{len(models)}) | tasks={tasks}")
        print("#" * 60)

        p = ctx.Process(
            target=_run_one,
            args=(
                model_key,
                args.save_to,
                tasks,
                args.n,
                args.seed,
                args.tensor_parallel_size,
                args.progress_every,
            ),
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            print(f"[!!] Process for {model_key} exited with code {p.exitcode}")
            failures.append(model_key)

    print("\n" + "=" * 60)
    print(f"Done. {len(models) - len(failures)}/{len(models)} succeeded.")
    if failures:
        print(f"Failed: {', '.join(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
