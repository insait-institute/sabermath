"""DIAGNOSTIC: reproduce a similarities file on the OLD SentenceTransformers
backend, to test whether a "backend only" delta is real.

    python scripts/repro_st_backend.py --method nvidia/llama-embed-nemotron-8b

Writes similarities/<method>__stbackend.json and nothing else. Never touches
the canonical <method>.json.

WHY THIS EXISTS AND --legacy DOES NOT COVER IT
----------------------------------------------
calc_sims.py --legacy reverts the 2026-08-25 PROTOCOL change: vendor input
envelopes, the instruction template, ColBERT/GroupRank per-arm budgets, the
Qwen3-Reranker <Instruct> slot. It does NOT revert the 2026-08-20 rollout that
moved most models from SentenceTransformers to vLLM.

So for a model whose ONLY change was the backend - llama-embed-nemotron-8b,
bge-m3, Qwen3-Embedding-0.6B/4B, harrier-270m/0.6b/27b, Octen-4B/8B,
KaLM-Gemma3-12B - `--legacy` is a literal no-op (verified: identical builder
call AND identical scores_kwargs), and cannot reproduce the pre-delegation
numbers. This script can, because it rebuilds the model the way
load_models.get_model() did before 2026-08-26:

    SentenceTransformersProcessor.from_huggingface(name, trust_remote_code=True)

with scores_kwargs = {} (the old _SCORES_KWARGS carried entries for the three
RaDeR bi-encoders and nothing else) and no instruction.

READING THE RESULT
------------------
Compare __stbackend.json against similarities_baseline/<method>.json:
  * matches  -> the old numbers are reproducible, so the vLLM delta is a real
                backend effect, not a corrupted or stale baseline;
  * differs  -> the baseline itself cannot be reproduced from the old code
                path, and the delta attribution needs rethinking before any
                of it is trusted.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

# This script lives in scripts/, so Python puts scripts/ on sys.path[0] - NOT
# the experiment directory, where embed.py and load_models.py live. calc_sims.py
# never hits this because it sits in the experiment dir itself. Confirmed the
# hard way: job 752930 died in 10s with "ModuleNotFoundError: No module named
# 'embed'". run_sims.slurm already cd's into the experiment dir, so the cwd is
# right; only sys.path was wrong.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402

from embed import get_top5_candidates  # noqa: E402
from load_models import ALLOWED_MODELS, SentenceTransformersProcessor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--config_file", default="config.yaml")
    ap.add_argument("--force_recalc", action="store_true")
    ap.add_argument(
        "--attn",
        default=None,
        help="attn_implementation to force via model_kwargs (eager | "
        "flash_attention_2 | sdpa). REQUIRED for the bidirectional "
        "remote-code models under a transformers that still has the "
        "_update_causal_mask hook: nvidia/llama-embed-nemotron-8b's override "
        "asserts the impl is flash_attention_2 or eager, and "
        "SentenceTransformers defaults to sdpa, so it raises "
        "AssertionError rather than running. 'eager' is the reference "
        "implementation - slowest, but the right choice for a correctness "
        "check. Leaving this unset under transformers 5.x does NOT reproduce "
        "the model card: the hook is gone there, the assert never fires, and "
        "the model silently runs CAUSAL.",
    )
    ap.add_argument(
        "--tag",
        default="stbackend",
        help="Output suffix: similarities/<method>__<tag>.json. Use a tag that "
        "names what is being varied (e.g. stbackend_tf4513) so two ST runs "
        "under different transformers pins never collide.",
    )
    args = ap.parse_args()

    if args.method not in ALLOWED_MODELS:
        raise SystemExit(f"Unknown method {args.method}")

    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    targets = load_dataset(config["hf_datasets"]["targets_maths_words_fixed"])["train"]
    candidates = load_dataset(
        config["hf_datasets"]["candidates_maths_words_fixed"]
    )["train"]

    stem = args.method.replace("/", "_")
    out_path = f"similarities/{stem}__{args.tag}.json"
    tmp_path = f"similarities/.{stem}__{args.tag}.json.tmp"

    print(f"===== {args.method}: OLD SentenceTransformers path =====")
    # Report whether this transformers version still exposes the hooks a
    # bidirectional remote-code model needs. nvidia/llama-embed-nemotron-8b
    # and reasonir/ReasonIR-8B both subclass LlamaModel to (a) override
    # _update_causal_mask() and (b) set layer.self_attn.is_causal = False.
    # Under transformers 5.16.1 BOTH are inert: the method no longer exists
    # (LlamaModel.forward calls masking_utils.create_causal_mask directly) and
    # is_causal is never passed to the attention interface. A model whose card
    # says bidirectional then runs CAUSAL, silently. This block makes that
    # visible in the log instead of leaving it to be discovered by a
    # cross-backend diff.
    import transformers
    from transformers.models.llama import modeling_llama

    print(f"[~] transformers {transformers.__version__}")
    has_hook = hasattr(modeling_llama.LlamaModel, "_update_causal_mask")
    print(f"[~] LlamaModel._update_causal_mask present : {has_hook}")
    src = Path(modeling_llama.__file__).read_text()
    passes_is_causal = "is_causal=self.is_causal" in src or "self.is_causal," in src
    print(f"[~] is_causal forwarded to attention       : {passes_is_causal}")
    if not (has_hook or passes_is_causal):
        print("[!!] NEITHER hook is live - this run is CAUSAL, not bidirectional.")
    # EXACTLY the pre-2026-08-26 generic branch of load_models.get_model().
    # No pooler override, no dtype pin, no envelope - whatever the repo's own
    # modules.json specifies, which is the point of the comparison.
    st_kwargs = {"trust_remote_code": True}
    if args.attn:
        st_kwargs["model_kwargs"] = {"attn_implementation": args.attn}
        print(f"[~] forcing attn_implementation={args.attn}")
    processor = SentenceTransformersProcessor.from_huggingface(
        args.method, **st_kwargs
    )

    sims = {}
    if not args.force_recalc:
        os.makedirs("similarities", exist_ok=True)
        if os.path.exists(out_path):
            with open(out_path) as f:
                sims = json.load(f)
    print(f"===== starting from idx {len(sims)} =====")

    import tqdm

    for i in tqdm.tqdm(range(len(sims), len(targets))):
        t = targets[i]
        cands = [
            candidates[s]["problem_fixed"] + candidates[s]["solution_fixed"]
            for s in get_top5_candidates(t)
        ]
        # Same call shape sim_embeddings.py uses, with the old empty kwargs.
        vals = {}
        for field, text in (
            ("pr_full_vs_candidates", t["problem_fixed"]),
            ("pr_math_vs_candidates", t["problem_math_expr"]),
            ("pr_text_vs_candidates", t["problem_text_only"]),
        ):
            vals[field] = float(
                mean(processor.get_scores(text, cands, show_progress_bar=False))
            )
        sims[t["id"]] = vals
        with open(tmp_path, "w") as f:
            json.dump(sims, f)
        os.replace(tmp_path, out_path)

    print(f"Wrote {out_path} ({len(sims)} targets).")


if __name__ == "__main__":
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
