"""DIAGNOSTIC: cross-encoder reranker, vLLM vs the legacy HF reference.

    python scripts/diag_reranker_backends.py --method rader-reranker-7b

rader-reranker-7b is a POINTWISE CROSS-ENCODER: it scores query-document
pairs directly and never produces an embedding. So scripts/diag_tokenization.py
does not apply - there are no vectors to compare, and no pooler. What can be
compared is the score vector itself, against this model's own legacy
reference implementation:

    RaDeRRerankerVLLMProcessor  (production since 2026-08-20, LoRA merged)
    RaDeRRerankerProcessor      (the HF-transformers reference it replaced)

run_rerankers.py claims the switch was "verified FEASIBLE, Spearman 0.9968".
That claim cites scripts/test_vllm_feasibility.py, which is NOT PRESENT in
this repo, so it has never been checkable here. This checks it directly.

WHY IT MATTERS BEYOND PROVENANCE: rader-reranker-7b reports a math-vs-word
statistic of 27.14%, far below every bi-encoder in the roster (next lowest is
gemini-embedding-001 at 40.35%). Two very different explanations:

  * a BACKEND problem, like the ones already found for
    llama-embed-nemotron-8b and Octen - in which case the number is wrong; or
  * the metric simply not transferring to a cross-encoder. The math-vs-word
    statistic compares cosine similarities of embeddings; a generative
    pointwise scorer emits logits on a different scale with a different
    response to short, equation-only text. Then 27.14% is "real" but not
    comparable to the bi-encoder rows.

This separates them: if the two backends AGREE, the backend is exonerated and
the low value is a property of the model/metric pairing, not a bug.
"""

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "src"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402

from embed import get_top5_candidates  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="rader-reranker-7b")
    ap.add_argument("--config_file", default="config.yaml")
    ap.add_argument("--n", type=int, default=12,
                    help="Targets to probe. The legacy HF path runs one "
                         "forward pass per pair and is slow, so this is "
                         "deliberately small - agreement or disagreement is "
                         "visible well before 969.")
    args = ap.parse_args()

    from sabermath.processors import (
        RaDeRRerankerProcessor,
        RaDeRRerankerVLLMProcessor,
    )

    with open(args.config_file) as f:
        cfg = yaml.safe_load(f)
    targets = load_dataset(cfg["hf_datasets"]["targets_maths_words_fixed"])["train"]
    cands = load_dataset(cfg["hf_datasets"]["candidates_maths_words_fixed"])["train"]

    # Same three query representations sim_embeddings.py scores, so the
    # math-vs-word verdict can be recomputed per backend rather than inferred.
    probes = []
    for i in range(args.n):
        t = targets[i]
        docs = [cands[s]["problem_fixed"] + cands[s]["solution_fixed"]
                for s in get_top5_candidates(t)]
        probes.append((t["id"], t["problem_fixed"], t["problem_math_expr"],
                       t["problem_text_only"], docs))
    print(f"{len(probes)} targets x 3 query variants x 5 candidates "
          f"= {len(probes)*15} pairs per backend\n")

    def run(proc, label):
        out = {}
        for tid, full, math, text, docs in probes:
            out[tid] = {
                "full": [float(x) for x in proc.get_scores(full, docs, show_progress_bar=False)],
                "math": [float(x) for x in proc.get_scores(math, docs, show_progress_bar=False)],
                "text": [float(x) for x in proc.get_scores(text, docs, show_progress_bar=False)],
            }
        print(f"[+] {label} done")
        return out

    print("=== vLLM (production) ===")
    v = run(RaDeRRerankerVLLMProcessor(), "vLLM")
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    print("=== legacy HF reference ===")
    h = run(RaDeRRerankerProcessor(), "legacy")

    flat_v = np.array([s for t in probes for k in ("full", "math", "text")
                       for s in v[t[0]][k]], dtype=np.float64)
    flat_h = np.array([s for t in probes for k in ("full", "math", "text")
                       for s in h[t[0]][k]], dtype=np.float64)
    pear = float(np.corrcoef(flat_v, flat_h)[0, 1])
    rv = np.argsort(np.argsort(flat_v)); rh = np.argsort(np.argsort(flat_h))
    spear = float(np.corrcoef(rv, rh)[0, 1])
    print(f"\n=== agreement over {len(flat_v)} pair scores ===")
    print(f"  Pearson  {pear:.6f}")
    print(f"  Spearman {spear:.6f}   (run_rerankers claims 0.9968)")
    print(f"  mean |delta| {np.mean(np.abs(flat_v-flat_h)):.6f}"
          f"   max {np.max(np.abs(flat_v-flat_h)):.6f}")

    print(f"\n=== math-vs-word verdict per backend (on these {len(probes)}) ===")
    for label, d in (("vLLM", v), ("legacy", h)):
        g = c = 0
        for tid, *_ in probes:
            m, t = np.mean(d[tid]["math"]), np.mean(d[tid]["text"])
            if m == t:
                continue
            c += 1; g += m > t
        print(f"  {label:<7} math>text {g}/{c}" + (f" = {100*g/c:.1f}%" if c else ""))
    agree = sum(1 for tid, *_ in probes
                if (np.mean(v[tid]["math"]) > np.mean(v[tid]["text"]))
                == (np.mean(h[tid]["math"]) > np.mean(h[tid]["text"])))
    print(f"  backends agree on {agree}/{len(probes)} verdicts")
    print("\n  -> if the two AGREE, the backend is exonerated and 27.14% is a "
          "property of\n     the model/metric pairing, not a bug.")


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
