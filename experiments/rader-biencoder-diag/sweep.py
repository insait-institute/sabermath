"""Input-envelope sweep for the RaDeR BI-ENCODERS (default: rader-14b).

Motivation (2026-08-28): the reranker investigation found that RaDeR's own
inference code does not reproduce the format its checkpoints were TRAINED on
(see experiments/rader-reranker-diag/sweep.py). The same suspicion applies to
the bi-encoders, from a different direction:

- RaDeR's README trains the retrievers with
      --query_prefix "Query: " --passage_prefix "Passage: "
  and Tevatron's retriever dataset applies those by plain concatenation
  (query_prefix + query_text), with NO title slot - so, unlike the reranker,
  there is no empty-title double space here.
- The training queries additionally carry a literal "Query: " prefix in the
  data itself (100% of rows in Raderspace/MATH_NuminaMath_allquerytypes), so
  the trained query string was plausibly the DOUBLED "Query: Query: {q}".
- run_rerankers.py currently sends "query: " / "document: ", taken from
  RaDeR's retrievers.py - which is internally inconsistent, using
  query:/Query:/document:/Document:/passage: across different functions and
  terminating several of them with "</s>", which is not even Qwen2.5's eos.

The eos is NOT varied: Tevatron's retriever collator appends it as a token id
to the already-tokenized text, so there is no space before it in training -
the same check that showed the reranker's pre-eos gain has no training-format
justification.

Everything else (pooler, dtype, chunking, engine flags) comes from
run_rerankers._build_rader_biencoder_vllm, i.e. the production path, so the
envelope is the only variable.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sabermath.data import load_data  # noqa: E402
from sabermath.benchmark import transform  # noqa: E402

# Affix arms: (query_prompt, document_prompt). The suffix stays at RaDeR's
# eos for every arm.
ARMS = {
    "R0": ("query: ", "document: "),
    "R1": ("Query: ", "Passage: "),
    "R2": ("Query: Query: ", "Passage: "),
    "R3": ("query: ", "Passage: "),
}
ARM_NOTES = {
    "R0": "production (run_rerankers.py, from RaDeR retrievers.py)",
    "R1": "the README's TRAINING prefixes",
    "R2": "R1 + the data-side 'Query: ' every training row carries",
    "R3": "document prefix only, to isolate it from the query prefix",
}


def ndcg_at_10(rel, order):
    d = 1.0 / np.log2(np.arange(2, 12))
    r = np.asarray(rel)[order][:10]
    ideal = np.sort(rel)[::-1][:10]
    denom = ((2**ideal - 1) * d[: len(ideal)]).sum()
    return 0.0 if denom == 0 else ((2**r - 1) * d[: len(r)]).sum() / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="rader-14b",
                    choices=["rader-3b", "rader-7b", "rader-14b"])
    ap.add_argument("--task", default="statement-full",
                    choices=["statement-statement", "statement-full", "full-full"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/rader_biencoder_diag")
    args = ap.parse_args()

    import run_rerankers as RR

    queries_ds, documents_ds = load_data()
    idxs = sorted(random.Random(args.seed).sample(range(len(queries_ds)), args.n))
    print(f"[~] {args.model_key} | {args.task} | {len(idxs)} queries (seed "
          f"{args.seed}) | arms {args.arms}", flush=True)

    proc = RR._build_rader_biencoder_vllm(RR.RADER_BIENCODER_MODELS[args.model_key])

    # Chunking knobs come from the production config; only the affixes vary.
    base_kwargs = {
        k: v for k, v in RR._RADER_BIENCODER_SCORES_KWARGS.items()
        if k in ("chunk_to_context", "context_length")
    }
    eos = RR.RADER_EXPECTED_EOS

    qv = "full" if args.task == "full-full" else "statement"
    dv = "statement" if args.task == "statement-statement" else "full"
    queries = transform(queries_ds, qv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for arm in args.arms:
        qp, dp = ARMS[arm]
        per_query, dump = [], {}
        for i in idxs:
            doc_ids = list(queries_ds[i]["candidates"])
            rel = np.asarray(queries_ds[i]["relevance_scores"], dtype=float)
            docs = transform(documents_ds, dv, doc_ids)
            s = np.asarray(proc.get_scores(
                queries[i], docs, show_progress_bar=False,
                query_prompt=qp, document_prompt=dp,
                query_suffix=eos, document_suffix=eos,
                **base_kwargs,
            ))
            per_query.append(ndcg_at_10(rel, np.argsort(-s)))
            dump[str(i)] = [float(x) for x in s]
        m = float(np.mean(per_query))
        summary[arm] = m
        print(f"[=] {args.task} {arm} nDCG@10 = {m:.4f}   "
              f"q={qp!r} d={dp!r}   # {ARM_NOTES[arm]}", flush=True)
        tag = f"{args.model_key}__{args.task}__{arm}"
        (out_dir / f"scores__{tag}.json").write_text(json.dumps(dump))
        (out_dir / f"summary__{args.model_key}__{args.task}.json").write_text(
            json.dumps({"model": args.model_key, "task": args.task,
                        "n": args.n, "seed": args.seed,
                        "query_idxs": idxs, "arms": ARMS,
                        "results": summary}, indent=1)
        )
    print("[+] done")


if __name__ == "__main__":
    main()
