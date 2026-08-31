"""Running one (model, prompt) cell of the benchmark, and keeping the result
on disk safe against interruption.

ONE CODE PATH. This module has a single run function, `run_cell`, and prompt
key "p0" means "no instruction". The previous split - a production path with
no prompt concept plus a parallel instruction-experiment path - was two
near-copies of the same logic that wrote to two different result directories
and disagreed about file naming; p0 collapses them, so a plain table run is
just the p0 arm of the same sweep and lands in the same folder as every other
arm.

CHECKPOINTING. Every task is checkpointed after each query into
"<save-to>/.checkpoints/<model>/<prompt>__<subset>/<task>.json". Re-running an
identical command resumes from the last completed query instead of restarting
- the failure mode that killed six of seven early rank1-32b attempts was a
wall-clock timeout, not a bug, so resumability is the difference between a 32B
model being evaluable and not.

CONCURRENT WRITERS. Result files are merged under an flock and replaced
atomically (write .tmp, rename), because several jobs legitimately target the
same file: one job per --task, one per query shard, and a periodic progress
snapshot racing them. A failed attempt never overwrites good data.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import random
import subprocess
import sys
import traceback
from pathlib import Path

from .benchmark import evaluate, transform
from .data import load_data
from .instructions import (
    DEFAULT_INSTRUCTION_TEMPLATE,
    INSTRUCTIONS,
    format_instructed_query,
)
from .processors.embedding_processor import AFFIX_KEYS
from .processors import EmbeddingProcessor
from . import registry as R

DEFAULT_SAVE_DIR = "results/evaluation"


# ---------------------------------------------------------------------------
# Result files
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def result_lock(filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    lock_path = filepath.with_name(filepath.name + ".lock")
    with open(lock_path, "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def merge_results(filepath: Path, model_key: str, new_output: dict) -> dict:
    """Merge new_output into whatever's already saved at filepath, so writing
    one --task slice never erases another already-saved slice for the same
    model+prompt+subset. Tasks are merged by name - re-running one task
    overwrites just that task's entry. A run that crashed entirely never
    overwrites previously-successful data."""
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
    merged = {"domains": domains, "reports": {model_key: existing_report}}
    for key in ("query_row_idxs", "part"):
        value = new_output.get(key)
        if value is None:
            value = existing.get(key)
        if value is not None:
            merged[key] = value
    return merged


def write_result(
    filepath: Path,
    model_key: str,
    new_output: dict,
    extra_report_fields: dict | None = None,
) -> dict:
    with result_lock(filepath):
        merged = merge_results(filepath, model_key, new_output)
        if extra_report_fields:
            merged["reports"][model_key].update(extra_report_fields)
        tmp = filepath.with_name(filepath.name + ".tmp")
        tmp.write_text(json.dumps(merged, indent=2))
        tmp.replace(filepath)
    return merged


def write_meta(meta_path: Path, meta: dict) -> None:
    with result_lock(meta_path):
        existing = {}
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                existing = {}
        merged = dict(existing)
        merged.update(meta)
        seen_tasks = list(existing.get("tasks", []))
        for task in meta.get("tasks", []):
            if task not in seen_tasks:
                seen_tasks.append(task)
        merged["tasks"] = seen_tasks
        tmp = meta_path.with_name(meta_path.name + ".tmp")
        tmp.write_text(json.dumps(merged, indent=2))
        tmp.replace(meta_path)


def partial_report_from_checkpoints(
    model_key: str, checkpoint_dir: Path, tasks: list[str], domains: list[list[str]]
) -> dict:
    """Rebuild an in-progress snapshot straight from the checkpoint files
    (each {query_idx: ndcg_or_null}), without waiting for evaluate() to
    finish. Same shape as a real TaskResult, plus "n_done"/"n_total" so a
    partial write is never mistaken for a finished one at a glance."""
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
        for branch in R.BRANCHES:
            branch_vals = [
                by_idx[i]
                for i in range(n_total)
                if by_idx.get(i) is not None and branch in domains[i]
            ]
            branch_ndcg = (
                float(sum(branch_vals) / len(branch_vals)) if branch_vals else 0.0
            )
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

    return {
        "domains": domains,
        "reports": {
            model_key: {
                "model": model_key,
                "processor": None,
                "dcg_variant": "exponent",
                "k": 10,
                "tasks": tasks_out,
                "ndcgs_by_task": ndcgs_by_task,
            }
        },
    }


def sync_back_if_needed(save_dir: str | Path) -> None:
    """Push save_dir back to the canonical host over rsync, callable from
    inside the query loop (via on_progress) rather than only at process exit.
    A no-op unless NEEDS_REGION_SYNC=1 is set (i.e. this node's home storage
    isn't the canonical copy). Failures are logged, never raised - a flaky
    sync must not crash a run that is otherwise making progress."""
    if os.environ.get("NEEDS_REGION_SYNC") != "1":
        return
    host = os.environ.get("SABERMATH_CANONICAL_HOST", "hala")
    path = os.environ.get("SABERMATH_CANONICAL_PATH", "/home/ivo_petrov/sabermath")
    try:
        # rsync exit 23/24 mean "some source files vanished or could not be
        # read" - guaranteed here, because this very loop renames *.json.tmp
        # into place while the transfer runs. Everything else did transfer, so
        # those are successes; check=True would turn every one of them into a
        # spurious failure and hide the real ones.
        completed = subprocess.run(
            [
                "rsync",
                "-auz",
                "--exclude=*.tmp",
                "--exclude=.*.??????",
                f"{save_dir}/",
                f"{host}:{path}/{save_dir}/",
            ],
            capture_output=True,
            timeout=300,
        )
        if completed.returncode not in (0, 23, 24):
            raise RuntimeError(
                f"rsync exit {completed.returncode}: "
                f"{completed.stderr.decode(errors='replace')[-400:]}"
            )
    except Exception as e:
        print(f"[!!] Periodic sync-back to {host} failed (will retry next time): {e}")


# ---------------------------------------------------------------------------
# Query subsets
# ---------------------------------------------------------------------------


def subset_key(
    n: int | None, seed: int, query_shard: int | None, query_shards: int | None
) -> str:
    """Keys a result and its checkpoints by which queries they cover, because
    evaluate() checkpoints by query POSITION, not content: a 20-query smoke
    test must never land in the same file as a full run."""
    parts = []
    if n is not None:
        parts.append(f"n{n}_seed{seed}")
    if query_shards:
        parts.append(f"shard{query_shard}of{query_shards}")
    return "__".join(parts) if parts else "full"


def select_shard(queries, domains, query_row_idxs, query_shard, query_shards):
    """Strided (i % shards == shard) query split, so every shard sees a mix of
    domains and difficulties rather than one contiguous block. Each shard
    writes its own result file and checkpoint dir, so shards are safe to run
    concurrently on separate machines; `run_experiments.py --merge-shards`
    (sabermath.shards) stitches them back into one result file."""
    idxs = [i for i in range(len(queries)) if i % query_shards == query_shard]
    if not idxs:
        raise ValueError(f"Shard {query_shard}/{query_shards} selected no queries.")
    global_idxs = (
        [query_row_idxs[i] for i in idxs] if query_row_idxs is not None else list(idxs)
    )
    return queries.select(idxs), [domains[i] for i in idxs], global_idxs


def instructed_query_texts(
    queries, tasks: list[str], query_instruction: str | None, template: str
) -> list[str]:
    """The query texts evaluate_task() will actually score for one prompt key,
    deduplicated across the run's tasks. Mirrors evaluate_task: transform() per
    task's query version, then the same format_instructed_query wrap when an
    instruction is in play."""
    texts, seen = [], set()
    for task in tasks:
        version = "full" if task == "full-full" else "statement"
        for text in transform(queries, version):
            if query_instruction is not None:
                text = format_instructed_query(query_instruction, text, template)
            if text not in seen:
                seen.add(text)
                texts.append(text)
    return texts


def assert_envelope_supported(model_key: str, processor, scores_kwargs: dict) -> None:
    """The vendor input envelope is applied by EmbeddingProcessor.get_scores.
    Every processor that gets one is a bi-encoder, but a future registry edit
    could point one at a cross-encoder whose get_scores(**kwargs) would
    silently swallow it - fail loudly instead."""
    envelope = [k for k in scores_kwargs if k in AFFIX_KEYS]
    if envelope and not isinstance(processor, EmbeddingProcessor):
        raise TypeError(
            f"{model_key}: input-envelope kwargs {envelope} are only applied "
            f"by EmbeddingProcessor.get_scores, but this model's processor is "
            f"{type(processor).__name__}."
        )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def result_filename(
    model_key: str,
    instruction_key: str,
    subset: str = "full",
    part_name: str | None = None,
) -> str:
    suffix = "" if subset == "full" else f"__{subset}"
    part = f"__part-{part_name}" if part_name else ""
    return f"{model_key}__{instruction_key}{suffix}{part}.json"


def run_model(
    model_key: str,
    save_dir: str | Path = DEFAULT_SAVE_DIR,
    tasks: list[str] | None = None,
    instruction_keys: list[str] | None = None,
    n: int | None = None,
    seed: int = 42,
    tensor_parallel_size: int = 1,
    progress_every: int = 10,
    save_scores: bool = True,
    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE,
    part_name: str | None = None,
    query_shard: int | None = None,
    query_shards: int | None = None,
) -> None:
    """Evaluate one model across one or more prompt arms.

    Runs in the calling process; the CLI wraps this in a fresh subprocess per
    model so an OOM or a CUDA crash cannot corrupt results already written.
    Processors are built once per `registry.processor_slot` and reused across
    prompt arms, so a multi-arm run loads an 8B model once rather than
    four times.
    """
    tasks = list(tasks or R.ALL_TASKS)
    instruction_keys = list(dict.fromkeys(instruction_keys or ["p0"]))
    save_dir = str(save_dir)

    if tensor_parallel_size > 1 and not R.uses_tensor_parallel(model_key):
        reason = (
            "is pinned to tensor_parallel_size=1 (confirmed to corrupt scores "
            "otherwise - see Rank1Processor._init())"
            if model_key == "rank1-32b"
            else "doesn't shard across GPUs in this framework"
        )
        print(
            f"[~] {model_key} {reason} - --tensor-parallel-size "
            f"{tensor_parallel_size} will be ignored and only 1 GPU used."
        )

    subset = subset_key(n, seed, query_shard, query_shards)

    queries, documents = load_data()
    domains = list(queries["domains"])
    query_row_idxs = None

    if n is not None:
        rng = random.Random(seed)
        query_row_idxs = sorted(rng.sample(range(len(queries)), min(n, len(queries))))
        queries = queries.select(query_row_idxs)
        domains = [domains[i] for i in query_row_idxs]

    if query_shards:
        queries, domains, query_row_idxs = select_shard(
            queries, domains, query_row_idxs, query_shard, query_shards
        )
        print(f"[~] Query shard {query_shard}/{query_shards}: {len(queries)} queries.")

    processors: dict[str, object] = {}
    task_scores_by_prompt: dict[str, dict[str, float]] = {}
    failures: list[str] = []

    for instruction_key in instruction_keys:
        instruction_text = INSTRUCTIONS[instruction_key]
        scores_kwargs, wrap_instruction = R.prompt_scores_kwargs(
            model_key, instruction_text
        )
        query_instruction = instruction_text
        if model_key in R.QWEN3_RERANKER_REPOS or not wrap_instruction:
            # The instruction reaches these models through their own slot, not
            # by wrapping the query text.
            query_instruction = None

        prompt_block = {
            "key": instruction_key,
            "text": instruction_text,
            "query_instruction_applied": query_instruction is not None,
            "protocol": "canonical",
            "instruction_template": instruction_template,
            "mechanism": (
                "control" if R.is_control_model(model_key) else "vendor-instruction"
            ),
            "control_reason": R.INSTRUCTION_CONTROL_REASONS.get(model_key),
            "input_envelope": {
                k: v for k, v in scores_kwargs.items() if k in AFFIX_KEYS
            },
        }

        filepath = Path(save_dir) / result_filename(
            model_key, instruction_key, subset, part_name
        )
        filepath.parent.mkdir(parents=True, exist_ok=True)
        part_suffix = f"__part-{part_name}" if part_name else ""
        checkpoint_dir = (
            Path(save_dir)
            / ".checkpoints"
            / model_key
            / f"{instruction_key}__{subset}{part_suffix}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        write_meta(
            checkpoint_dir / "meta.json",
            {
                "model_key": model_key,
                "instruction_key": instruction_key,
                "instruction_text": instruction_text,
                "query_instruction_applied": query_instruction is not None,
                "n": n,
                "seed": seed,
                "query_row_idxs": query_row_idxs,
                "k": 10,
                "dcg_variant": "exponent",
                "tasks": tasks,
                "part": part_name,
                "query_shard": query_shard,
                "query_shards": query_shards,
                "prompt": prompt_block,
            },
        )

        progress_counter = {"n": 0}

        def on_progress(
            _filepath=filepath,
            _checkpoint_dir=checkpoint_dir,
            _counter=progress_counter,
        ) -> None:
            # Snapshot progress from the checkpoints into the real result file
            # every `progress_every` freshly-scored queries, and push it to the
            # canonical host. A hard kill then loses at most that many queries'
            # work rather than the whole run.
            if progress_every <= 0:
                return
            _counter["n"] += 1
            if _counter["n"] % progress_every != 0:
                return
            partial = partial_report_from_checkpoints(
                model_key, _checkpoint_dir, tasks, domains
            )
            partial["query_row_idxs"] = query_row_idxs
            partial["part"] = part_name
            write_result(_filepath, model_key, partial)
            sync_back_if_needed(save_dir)

        print(f"\n[~] {model_key} / {instruction_key} ...")

        try:
            slot = R.processor_slot(model_key, instruction_key)
            if slot not in processors:
                processors[slot] = R.build_processor(
                    model_key, instruction_key, tensor_parallel_size, save_dir
                )
            model = processors[slot]

            assert_envelope_supported(model_key, model, scores_kwargs)

            # Batch any per-query preprocessing for THIS prompt key before
            # scoring: generating one query's rewrites per call leaves ~5
            # sequences in flight and runs ~10x slower than one batched call.
            # The texts handed over are byte-identical to what evaluate() will
            # score for this key.
            if hasattr(model, "prefetch_rewrites"):
                model.prefetch_rewrites(
                    instructed_query_texts(
                        queries, tasks, query_instruction, instruction_template
                    )
                )

            report, ndcgs = evaluate(
                model,
                tasks=tasks,
                queries=queries,
                documents=documents,
                return_ndcgs=True,
                checkpoint_dir=checkpoint_dir,
                on_progress=on_progress,
                instruction=query_instruction,
                instruction_template=instruction_template,
                save_scores=save_scores,
                scores_kwargs=scores_kwargs,
            )

            report_dict = report.to_dict()
            report_dict["ndcgs_by_task"] = ndcgs
            output = {
                "domains": domains,
                "query_row_idxs": query_row_idxs,
                "part": part_name,
                "reports": {model_key: report_dict},
            }

            task_scores_by_prompt[instruction_key] = {
                t["task"]: t["ndcg_at_k"] for t in report_dict["tasks"]
            }

            print(f"\n=== {model_key} / {instruction_key} ===")
            for task in report_dict["tasks"]:
                print(f"  {task['task']:<20} nDCG@10 = {task['ndcg_at_k']:.4f}")
                for branch in task.get("branches", []):
                    print(f"      {branch['branch']:<22} {branch['ndcg_at_k']:.4f}")

        except Exception as e:
            failures.append(instruction_key)
            tb = traceback.format_exc()
            output = {
                "domains": None,
                "reports": {
                    model_key: {"model": model_key, "error": str(e), "traceback": tb}
                },
            }
            # The full traceback, not just str(e): a bare message ("type
            # 'array.array' is not subscriptable") is not enough to find the
            # failure site without reproducing it by hand.
            print(f"\n[!!] {model_key} / {instruction_key} failed:")
            print(tb)

        write_result(
            filepath, model_key, output, extra_report_fields={"prompt": prompt_block}
        )
        print(f"[+] Wrote {filepath}")
        sync_back_if_needed(save_dir)

    if task_scores_by_prompt and len(task_scores_by_prompt) > 1:
        p0_scores = task_scores_by_prompt.get("p0")
        print(f"\n=== {model_key}: per-prompt nDCG@10 summary ===")
        for instruction_key, by_task in task_scores_by_prompt.items():
            parts = []
            for task_name, score in by_task.items():
                delta = ""
                if (
                    instruction_key != "p0"
                    and p0_scores is not None
                    and task_name in p0_scores
                ):
                    delta = f" ({score - p0_scores[task_name]:+.4f})"
                parts.append(f"{task_name}={score:.4f}{delta}")
            print(f"  {instruction_key}: " + " | ".join(parts))

    if failures:
        sys.exit(1)
