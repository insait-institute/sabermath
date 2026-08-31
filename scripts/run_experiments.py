#!/usr/bin/env python3
"""Evaluate SABER-Math models. This is THE endpoint for model evaluation.

    # every model, no instruction (the main tables)
    python scripts/run_experiments.py

    # a subset
    python scripts/run_experiments.py --models rank1-7b qwen3-reranker-4b

    # the instruction ablation: four prompt arms per model
    python scripts/run_experiments.py --prompts p0 p1 p2 p3

    # smoke test on 20 random queries before a multi-hour run
    python scripts/run_experiments.py --models rank1-32b --n 20

    # split a slow model across 4 concurrent jobs, then stitch them back
    python scripts/run_experiments.py --models rank1-32b --query-shards 4 --query-shard 0
    python scripts/run_experiments.py --merge-shards

    # one task only, as three concurrent jobs that will not race
    python scripts/run_experiments.py --models rank1-32b --task statement-full \
        --part-name statement-full

    # list what is available
    python scripts/run_experiments.py --list

`--models` defaults to EVERY model in the registry, and `--prompts` defaults
to p0 ("no instruction"), so a bare invocation reproduces the main tables. All
results land in ONE directory - results/evaluation/ - named
"<model>__<prompt>[__<subset>][__part-<name>].json". There is no separate
production-vs-instruction split: p0 IS the production arm.

DEPENDENCY ISOLATION. Most models run from one environment
(scripts/envs/env_vllm.yml: vllm==0.26.0 + peft), but four families need
their own and will fail loudly in the wrong one:

    gte/reason-moderncolbert   scripts/envs/env_colbert.yml   (pylate pins ST 5.3.0)
    splade-code-*              scripts/envs/env_splade.yml    (SparseEncoder)
    inf-retriever-v1-pro       scripts/envs/env_inf_retriever.yml
    inf-x-retriever            scripts/envs/env_inf_retriever.yml  (transformers 4.51.x)

So `--models` defaulting to all is right for a report over finished runs, but
in a single shared environment run one model, or one same-env family, per
invocation. Each model runs in its own spawned subprocess either way, so a
crash or OOM never corrupts results already written.

RESUMING. Every task checkpoints after each query. Re-running the identical
command resumes from the last completed query - the reason a 32B generative
reranker is evaluable here at all, given a bounded wall clock.
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sabermath import registry as R
from sabermath.instructions import INSTRUCTIONS
from sabermath.runner import DEFAULT_SAVE_DIR, run_model
from sabermath.shards import (
    add_merge_arguments,
    merge_evaluation_shards,
    run_merge,
)


def _print_registry() -> None:
    groups = [
        ("custom-processor pipeline models", list(R.CUSTOM_MODEL_BUILDERS)),
        ("generic HuggingFace models", list(R.GENERIC_MODELS)),
        ("table-only models", list(R.TABLE_MODELS)),
        ("closed/API models", list(R.API_MODELS)),
        ("lexical baselines", list(R.LEXICAL_MODEL_BUILDERS)),
    ]
    for title, keys in groups:
        print(f"\n{title} ({len(keys)}):")
        for key in keys:
            notes = []
            if key in R.INSTRUCTION_EXCLUDED:
                notes.append("p0 only")
            elif R.is_control_model(key):
                notes.append("instruction control")
            if R.uses_tensor_parallel(key):
                notes.append("multi-GPU")
            suffix = f"   [{', '.join(notes)}]" if notes else ""
            print(f"  {key}{suffix}")
    print(f"\ntotal: {len(R.ALL_MODEL_KEYS)} models")
    print(f"prompts: {', '.join(INSTRUCTIONS)}  (p0 = no instruction)")
    print(f"tasks:   {', '.join(R.ALL_TASKS)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="KEY",
        default=None,
        help="Models to run (default: all %d in the registry). Pass --list to "
        "see them." % len(R.ALL_MODEL_KEYS),
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        choices=list(INSTRUCTIONS),
        default=["p0"],
        help="Instruction arms to run (default: p0, i.e. no instruction). "
        "Several arms in one invocation share the loaded model.",
    )
    parser.add_argument(
        "--task",
        choices=R.ALL_TASKS,
        default=None,
        help="Run only this task (default: all 3). Combine with --part-name to "
        "run the three tasks as three concurrent jobs without racing.",
    )
    parser.add_argument("--save-to", type=str, default=DEFAULT_SAVE_DIR)
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Evaluate only a random n-query subset (smoke test). Subset runs "
        "get their own result file and checkpoints, so they can never "
        "overwrite a full run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--part-name",
        type=str,
        default=None,
        help="Write to <model>__<prompt>__part-<NAME>.json with its own "
        "checkpoints. Use whenever several jobs cover different slices of the "
        "same cell; stitch them back with --merge-shards.",
    )
    parser.add_argument(
        "--query-shards",
        type=int,
        default=None,
        help="Split the query set into this many strided shards so a slow "
        "model can run as N concurrent jobs. Stitch the parts back together "
        "with --merge-shards.",
    )
    parser.add_argument(
        "--query-shard", type=int, default=None, help="Shard index (0-based)."
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="GPUs to shard across, for the models where it is honored "
        "(ignored with a warning elsewhere - see --list).",
    )
    parser.add_argument(
        "--save-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist per-query candidate scores and the applied ranking. ON "
        "BY DEFAULT: a ranking is the one thing a finished run cannot "
        "reconstruct, and nDCG under a different gain or rescaling is a pure "
        "function of it, so keeping it turns an expensive re-run into a "
        "CPU-second replay.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Snapshot progress into the result file every N freshly-scored "
        "queries (0 disables). Lower it for very slow models.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List every model key and exit."
    )
    add_merge_arguments(parser, DEFAULT_SAVE_DIR)
    args = parser.parse_args()

    if args.list:
        _print_registry()
        return

    if args.merge_shards is not None:
        run_merge(args, merge_evaluation_shards, Path(args.save_to))
        return

    models = args.models or list(R.ALL_MODEL_KEYS)
    unknown = [m for m in models if m not in R.ALL_MODEL_KEYS]
    if unknown:
        parser.error(
            f"Unknown model key(s): {', '.join(unknown)}. "
            "Run --list to see the available keys."
        )

    tasks = [args.task] if args.task else list(R.ALL_TASKS)
    prompts = list(dict.fromkeys(args.prompts))

    if (args.query_shards is None) != (args.query_shard is None):
        parser.error("--query-shards and --query-shard must be used together.")
    if args.query_shards is not None:
        if args.query_shards < 2:
            parser.error("--query-shards must be >= 2.")
        if not 0 <= args.query_shard < args.query_shards:
            parser.error(f"--query-shard must be in [0, {args.query_shards - 1}].")

    instructed = [p for p in prompts if p != "p0"]
    if instructed:
        for model_key in models:
            if model_key in R.INSTRUCTION_EXCLUDED:
                parser.error(
                    f"{model_key} cannot run instructed prompts "
                    f"({', '.join(instructed)}): "
                    f"{R.INSTRUCTION_EXCLUDED[model_key]}. "
                    "Only --prompts p0 is valid for it."
                )

    ctx = mp.get_context("spawn")
    failures = []

    for i, model_key in enumerate(models, start=1):
        print("\n" + "#" * 60)
        print(
            f"# {model_key} ({i}/{len(models)}) | tasks={tasks} | prompts={prompts}"
            + (f" | part={args.part_name}" if args.part_name else "")
            + (
                f" | shard={args.query_shard}/{args.query_shards}"
                if args.query_shards
                else ""
            )
        )
        print("#" * 60)

        # One subprocess per model: a CUDA crash or OOM must not take down the
        # loop or corrupt results already on disk.
        p = ctx.Process(
            target=run_model,
            kwargs={
                "model_key": model_key,
                "save_dir": args.save_to,
                "tasks": tasks,
                "instruction_keys": prompts,
                "n": args.n,
                "seed": args.seed,
                "tensor_parallel_size": args.tensor_parallel_size,
                "progress_every": args.progress_every,
                "save_scores": args.save_scores,
                "part_name": args.part_name,
                "query_shard": args.query_shard,
                "query_shards": args.query_shards,
            },
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
