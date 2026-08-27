"""DIAGNOSTIC: score a model on the vLLM backend, to test whether a model
currently served by SentenceTransformers could move to vLLM safely.

    python scripts/repro_vllm_backend.py --method reasonir/ReasonIR-8B
    python scripts/repro_vllm_backend.py --method infly/inf-retriever-v1-pro

Writes similarities/<method>__vllm.json. Never touches the canonical file.
This is the mirror of scripts/repro_st_backend.py, and the pair is how the
nemotron question was settled: run both backends over the same 969 targets
and diff them, instead of arguing from source.

WHY THIS MATTERS FOR THESE TWO MODELS
-------------------------------------
Both are bidirectional encoders shipped as remote code, and what they declare
in config.json decides whether vLLM can possibly be right:

  * reasonir/ReasonIR-8B declares architectures: ["ReasonIRModel"], a name
    vLLM does not register. It CAN be mapped onto vLLM's own bidirectional
    Llama via hf_overrides - which is exactly what math-vs-word's pre-2026-08-26
    _CUSTOM_BUILDERS did - and vLLM's LlamaBidirectionalModel is now known
    correct (it reproduced a genuinely-bidirectional reference for
    llama-embed-nemotron-8b to median |delta| 6.8e-05, 0 verdict flips).
    So a match here is plausible. NOTE it would only license p0: per
    run_rerankers.py's reasonir-8b entry, vLLM cannot express this model's
    INSTRUCTION mechanism (its encode() zeroes the instruction positions in
    the POOLING mask), which is why it was reverted off vLLM on 2026-08-25.

  * infly/inf-retriever-v1-pro declares architectures: ["Qwen2Model"] - the
    STOCK name - while shipping a custom modeling_qwen.py whose
    Qwen2Model.forward defaults is_causal to False. vLLM will therefore load
    its own CAUSAL Qwen2 and has no way to detect the difference, and there is
    no hf_overrides escape because the declared name is already the stock one.
    run_rerankers.py's _build_inf_retriever_processor documents this as the
    reason the model is a deliberate vLLM exception. A LARGE divergence here
    is the expected result, and confirms that note rather than refuting it.

Read the outcome the same way as repro_st_backend.py: diff __vllm.json
against the canonical similarities/<method>.json (the ST run).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402

from embed import get_top5_candidates  # noqa: E402
from load_models import ALLOWED_MODELS  # noqa: E402

# Per-model vLLM recipe. Keys are the ONLY models this diagnostic will build,
# deliberately - a bare VLLMProcessor.from_huggingface() on an arbitrary
# bidirectional model is precisely the silent-causal failure being measured,
# so each entry states what it is asserting.
RECIPES = {
    # Verbatim from math-vs-word's pre-2026-08-26 _CUSTOM_BUILDERS entry.
    "reasonir/ReasonIR-8B": {
        "hf_overrides": {
            "architectures": ["LlamaBidirectionalModel"],
            "pooling": "avg",
        },
        "pooler_config": {"pooling_type": "MEAN", "normalize": True},
    },
    # NO overrides on purpose: this is what a naive TABLE_MODELS-style vLLM
    # entry would do, which is the thing under test. Expect causal attention.
    "infly/inf-retriever-v1-pro": {},
    # Octen-8B: dtype probe, NOT an architecture question. Its config.json
    # declares architectures ["Qwen3Model"] with auto_map None (no remote
    # code), and vLLM resolves pooling_type=LAST from the sentence_transformers
    # config, matching the hand-built _build_octen_processor stack it replaced
    # - so architecture and pooling are both already correct.
    #
    # What differs is precision. config.json has torch_dtype: None, so the old
    # ST builder's torch_dtype="auto" had nothing to read and fell back to
    # FP32, while vLLM's own default picked BFLOAT16. This model pools
    # LAST-token, which is far more precision-sensitive than MEAN: the
    # embedding is a single position's hidden state, with no averaging to
    # smooth numerical error. Candidate explanation for the +6.60pp / 136-flip
    # p0 shift.
    #
    # Pinning float32 here reproduces the ST side's precision. If this
    # converges on similarities_baseline/, dtype is the whole story and the
    # spec needs a dtype pin. If it does not, the cause is elsewhere - which
    # is exactly how the same test ruled dtype OUT for
    # llama-embed-nemotron-8b (median gap stayed 0.0884 vs 0.0889).
    "Octen/Octen-Embedding-8B": {"dtype": "float32"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--config_file", default="config.yaml")
    ap.add_argument("--force_recalc", action="store_true")
    ap.add_argument("--tag", default="vllm")
    args = ap.parse_args()

    if args.method not in ALLOWED_MODELS:
        raise SystemExit(f"Unknown method {args.method}")
    if args.method not in RECIPES:
        raise SystemExit(
            f"No vLLM recipe for {args.method}. Add one to RECIPES with a "
            "comment saying what it asserts - see this file's docstring."
        )

    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    targets = load_dataset(config["hf_datasets"]["targets_maths_words_fixed"])["train"]
    candidates = load_dataset(
        config["hf_datasets"]["candidates_maths_words_fixed"]
    )["train"]

    stem = args.method.replace("/", "_")
    out_path = f"similarities/{stem}__{args.tag}.json"
    tmp_path = f"similarities/.{stem}__{args.tag}.json.tmp"

    from sabermath.processors import VLLMProcessor

    recipe = RECIPES[args.method]
    print(f"===== {args.method}: vLLM backend =====")
    print(f"[~] recipe: {recipe if recipe else '(none - stock architecture)'}")
    processor = VLLMProcessor.from_huggingface(args.method, **recipe)

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
