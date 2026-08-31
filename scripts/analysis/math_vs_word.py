#!/usr/bin/env python3
import argparse
import multiprocessing as mp
from pathlib import Path

from datasets import load_dataset
import yaml

from sabermath.math_vs_word.load_models import ALLOWED_MODELS

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

DEFAULT_CONFIG = CONFIG_DIR / "math_vs_word.yaml"

def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG))
    parser.add_argument("--method")
    parser.add_argument("--force-recalc", action="store_true")
    parser.add_argument(
        "--instruction",
        default=None,
        help="Instruction-ablation arm to run (p0/p1/p2/p3/pm). Omit for the "
        "default arm, which is p0 - no instruction TEXT, but still the "
        "model's full vendor input envelope, exactly as run_dedup.py runs it. "
        "An instructed arm writes similarities/<method>__<arm>.json so it "
        "never overwrites the default file. Embedding methods only.",
    )
    parser.add_argument(
        "--query-shards",
        type=int,
        default=None,
        help="Split the targets into N STRIDED shards (i %% N == shard), the "
        "same split run_experiments.py's --query-shards uses. Each shard writes "
        "similarities/<method>[__<arm>]__shard<i>of<N>.json and never touches "
        "another shard's file, so shards run concurrently - necessary here "
        "because sim_embeddings rewrites its whole dict after every target "
        "and two writers on one path clobber each other. Stitch them with "
        "merge_sim_parts.py. Embedding methods only.",
    )
    parser.add_argument(
        "--query-shard",
        type=int,
        default=None,
        help="Which shard to run, 0-based. Requires --query-shards.",
    )
    args = parser.parse_args(argv)

    if (args.query_shard is None) != (args.query_shards is None):
        parser.error("--query-shard and --query-shards must be given together")
    if args.query_shards is not None and not (0 <= args.query_shard < args.query_shards):
        parser.error(
            f"--query-shard must be in [0, {args.query_shards}), "
            f"got {args.query_shard}"
        )
    if args.query_shards is not None and args.force_recalc:
        parser.error(
            "--force_recalc with --query-shards would restart the shard from "
            "target 0; shards resume from their own file by design. Delete the "
            "shard file instead if you really mean to redo it."
        )

    config_file_path = args.config_file
    method = args.method

    if method not in ALLOWED_MODELS + ["jaccard", "approach0", "tf-idf", "bm25"]:
        raise ValueError(f"Unknown method {method}")

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    fixed_targets_dataset = config["hf_datasets"]["targets_maths_words_fixed"]
    fixed_candidates_dataset = config["hf_datasets"]["candidates_maths_words_fixed"]

    good_targets = load_dataset(fixed_targets_dataset)["train"]
    good_candidates = load_dataset(fixed_candidates_dataset)["train"]

    if method in ALLOWED_MODELS:
        from sabermath.math_vs_word.sim_embeddings import calc_embedding_sims

        calc_embedding_sims(
            method,
            good_targets,
            good_candidates,
            args.force_recalc,
            instruction_key=args.instruction,
            query_shard=args.query_shard,
            query_shards=args.query_shards,
        )
    elif args.instruction is not None or args.query_shards is not None:
        if args.query_shards is not None:
            raise SystemExit(
                f"--query-shards is not supported for {method}: the lexical "
                "methods build their own corpus state per run (tf-idf fits its "
                "vocabulary on all 150 candidates, approach0 indexes per "
                "target), so a strided subset is not the same computation."
            )
        flag = "--instruction"
        raise SystemExit(
            f"{flag} is not supported for {method}: jaccard/tf-idf/"
            "approach0/bm25 have no instruction mechanism and no vendor input "
            "envelope to strip. They are the CONTROL rows of the main "
            "ablation for exactly this reason "
            "(sabermath.registry.INSTRUCTION_CONTROL_REASONS)."
        )

    elif method == "jaccard":
        from sabermath.math_vs_word.sim_jaccard import calc_jaccard_sims

        calc_jaccard_sims(good_targets, good_candidates)

    elif method == "approach0":
        from sabermath.math_vs_word.sim_approach0 import calc_approach0_sims

        calc_approach0_sims(good_targets, good_candidates)

    elif method == "tf-idf":
        from sabermath.math_vs_word.sim_tfidf import calc_tfidf_sims

        calc_tfidf_sims(good_targets, good_candidates)

    elif method == "bm25":
        from sabermath.math_vs_word.sim_bm25 import calc_bm25_sims

        calc_bm25_sims(good_targets, good_candidates)

if __name__ == "__main__":
    # "spawn", because vLLM's EngineCore spawns subprocesses: under the
    # platform default (fork on Linux) that raises "An attempt has been made to
    # start a new process before the current process has finished its
    # bootstrapping phase".
    mp.set_start_method("spawn", force=True)
    main()
