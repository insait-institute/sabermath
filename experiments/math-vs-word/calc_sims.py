from datasets import load_dataset
import argparse
import multiprocessing as mp
import yaml

from load_models import ALLOWED_MODELS

# Deliberately NOT imported up front: sim_jaccard/sim_tfidf/sim_approach0
# all pull in pya0 (via sim_helpers.py), and sim_tfidf also needs
# scikit-learn - neither is needed by an embedding-model or bm25 run, and a
# missing optional dependency for ONE method must not take down every other
# method (confirmed the hard way: a bare `from sim_approach0 import ...` at
# module level here made even --method bm25 crash on ModuleNotFoundError:
# pya0, before ever reaching bm25's own code). Each is imported only inside
# the branch that actually needs it, below - same principle as
# sabermath/processors/__init__.py's own guarded imports for these same
# three legacy methods.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file")
    parser.add_argument("--method")
    parser.add_argument("--force_recalc", action="store_true")
    parser.add_argument(
        "--instruction",
        default=None,
        help="Instruction-ablation arm to run (p0/p1/p2/p3/pm). Omit for the "
        "prompt-free run every published similarities/ file holds. An "
        "instructed arm applies the model's full vendor input envelope AS "
        "WELL as the instruction - the same treatment run_rerankers.py gives "
        "the main ablation - and writes similarities/<method>__<arm>.json, "
        "so it never overwrites the prompt-free file. Embedding methods only.",
    )
    args = parser.parse_args()

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
        from sim_embeddings import calc_embedding_sims

        calc_embedding_sims(
            method,
            good_targets,
            good_candidates,
            args.force_recalc,
            instruction_key=args.instruction,
        )
    elif args.instruction is not None:
        raise SystemExit(
            f"--instruction is not supported for {method}: jaccard/tf-idf/"
            "approach0/bm25 have no instruction mechanism. They are the "
            "CONTROL rows of the main ablation for exactly this reason "
            "(run_rerankers.INSTRUCTION_CONTROL_REASONS)."
        )

    elif method == "jaccard":
        from sim_jaccard import calc_jaccard_sims

        calc_jaccard_sims(good_targets, good_candidates)

    elif method == "approach0":
        from sim_approach0 import calc_approach0_sims

        calc_approach0_sims(good_targets, good_candidates)

    elif method == "tf-idf":
        from sim_tfidf import calc_tfidf_sims

        calc_tfidf_sims(good_targets, good_candidates)

    elif method == "bm25":
        from sim_bm25 import calc_bm25_sims

        calc_bm25_sims(good_targets, good_candidates)


if __name__ == "__main__":
    # Required for RaDeRRerankerVLLMProcessor specifically (confirmed on
    # jobs 738673/738698, reproduced identically on two different nodes):
    # this script previously ran entirely as bare top-level module code
    # with no `if __name__ == "__main__":` guard at all, and relied on
    # Python's platform-default multiprocessing start method (fork on
    # Linux). That combination is exactly what
    # https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods
    # warns is unsafe once anything in-process later tries to spawn a
    # subprocess (here: vLLM's own EngineCore, right after this model's
    # one-off PEFT LoRA merge) - it raised "An attempt has been made to
    # start a new process before the current process has finished its
    # bootstrapping phase." Every OTHER vLLM-backed model in this sweep
    # happened not to trip it, but the underlying hazard was latent for
    # all of them too. scripts/run_rerankers.py's own main() sets this
    # exact start method for exactly this reason - mirroring it here
    # rather than special-casing just the one model that happened to
    # surface it.
    mp.set_start_method("spawn", force=True)
    main()
