"""Diagnostic sweep for rader-reranker-7b's low SaberMath score.

Motivation (2026-08-28): rader-reranker-7b scores ~0.15 nDCG@10 below the
RaDeR bi-encoders it shares training data with. The serving stack is already
cross-checked - the HF+peft path and vLLM 0.26.0 independently produce the
same full-benchmark numbers (max task delta 5.5e-4) - so this sweep targets
the INPUT FORMAT instead.

Why the format is suspect: the reranker was trained with Tevatron, whose
RerankerTrainDataset builds each pair as

    f'{query_prefix} {query} {passage_prefix} {title} {passage}'.strip()

and its training set (Raderspace/MATH_NuminaMath_allquerytypes) has
title == "" in 100% of passages and a query field that ALREADY carries a
literal "Query: " prefix in 100% of rows. So the trained-on string keeps the
empty title's spare space and a data-side query prefix - neither of which
RaDeR's own inference code (rerank.py: f"query: {q} document: {d}{eos}")
reproduces. T0 below is that reference format; T1-T4 are the Tevatron
reconstructions; T5 is the pair-encoding a sentence-transformers CrossEncoder
would produce; T6 drops the eos (Tevatron's append_eos_token=False default).

Run one backend per process (the HF model and a vLLM engine will not
co-exist on one GPU at usable memory fractions):

    python experiments/rader-reranker-diag/sweep.py --backend vllm
    python experiments/rader-reranker-diag/sweep.py --backend hf --arms T0
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sabermath.data import load_data  # noqa: E402
from sabermath.benchmark import transform  # noqa: E402

# All three benchmark tasks. The first template sweep (2026-08-28) ran only
# the two LENGTH EXTREMES - statement-statement (short query, short doc) and
# full-full (long, long) - to keep it cheap, which left the committed T10
# template's effect on statement-full unmeasured. Use --tasks to restrict.
TASKS = ["statement-statement", "statement-full", "full-full"]

# Each arm returns the plain STRING for a (query, document) pair; the eos is
# appended as a token id afterwards unless the arm is in NO_EOS.
ARMS = {
    "T0": lambda q, d: f"query: {q} document: {d}",
    "T1": lambda q, d: f"query: {q} document:  {d}",
    "T2": lambda q, d: f"query: Query: {q} document:  {d}",
    "T3": lambda q, d: f"Query: {q}   {d}",
    "T4": lambda q, d: f"Query: {q} {d}",
    "T5": None,  # pair encoding: tokenizer(q, d) - handled specially
    "T6": lambda q, d: f"query: {q} document: {d}",
    # T7-T9 (added 2026-08-28): the paper's Appendix G writes the input as
    #   input = query:{q} document:{d} <eos>
    # which differs from rerank.py's f"query: {q} document: {d}{eos}" in TWO
    # independent ways - no space after either colon, and a space before the
    # eos. Varying both at once would confound them, so this is the 2x2:
    #   post-colon space x pre-eos space
    #   T0 = (yes, no)   <- rerank.py, production
    #   T9 = (yes, yes)
    #   T7 = (no,  no)
    #   T8 = (no,  yes)  <- Appendix G read literally
    "T7": lambda q, d: f"query:{q} document:{d}",
    "T8": lambda q, d: f"query:{q} document:{d}",
    "T9": lambda q, d: f"query: {q} document: {d}",
    # T10 (added 2026-08-28): do the two independent gains stack? T2 (the
    # Tevatron format_pair reconstruction: empty-title double space + the
    # data-side "Query: " prefix) was worth +0.010 on statement-statement;
    # T9 (a space before the eos, from the paper's Appendix G notation) was
    # worth +0.017. T10 is T2 + T9's suffix.
    "T10": lambda q, d: f"query: Query: {q} document:  {d}",
}
NO_EOS = {"T6"}
# Appended AFTER doc-side truncation, so a trailing space can never be the
# thing that gets truncated away (it would be, if it rode inside the format
# string - the document is otherwise always last).
SUFFIX = {"T8": " ", "T9": " ", "T10": " "}
ARM_NOTES = {
    "T0": "rerank.py reference (current production)",
    "T1": "Tevatron format_pair, prefixes set, empty title -> double space",
    "T2": "T1 + the data-side 'Query: ' prefix the training rows carry",
    "T3": "Tevatron format_pair with DEFAULT empty prefixes + data-side prefix",
    "T4": "data-side prefix only, single space",
    "T5": "sentence-transformers CrossEncoder pair encoding: tokenizer(q, d)",
    "T6": "T0 with NO trailing eos (Tevatron append_eos_token=False default)",
    "T7": "paper App.G: NO space after either colon",
    "T8": "paper App.G read literally: no post-colon space + space before eos",
    "T9": "T0 + a space before the eos",
    "T10": "T2 + T9: Tevatron format_pair AND a space before the eos",
}


def build_ids(tokenizer, arm, query, document, max_length):
    """Tokenize one pair for `arm`, truncating the DOCUMENT side only so the
    sequence (plus any eos) fits max_length - same contract as the production
    processors, so a truncation difference can never be what separates arms."""
    eos = [] if arm in NO_EOS else [tokenizer.eos_token_id]

    if arm == "T5":
        ids = tokenizer(query, document, add_special_tokens=True).input_ids
        if len(ids) > max_length - len(eos):
            ids = ids[: max_length - len(eos)]
        return ids + eos

    # Every arm puts the document LAST inside the format string, so trimming
    # the tail truncates the document side only - the same contract the
    # production processors keep. Any arm suffix is added after the trim.
    suffix_ids = (
        tokenizer(SUFFIX[arm], add_special_tokens=False).input_ids
        if arm in SUFFIX
        else []
    )
    full_ids = tokenizer(ARMS[arm](query, document), add_special_tokens=False).input_ids
    budget = max_length - len(eos) - len(suffix_ids)
    if len(full_ids) > budget:
        full_ids = full_ids[:budget]
    return full_ids + suffix_ids + eos


def ndcg_at_10(relevances, order):
    d = 1.0 / np.log2(np.arange(2, 12))
    r = np.asarray(relevances)[order][:10]
    ideal = np.sort(relevances)[::-1][:10]
    denom = ((2**ideal - 1) * d[: len(ideal)]).sum()
    return 0.0 if denom == 0 else ((2**r - 1) * d[: len(r)]).sum() / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "vllm"], required=True)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--out", default="results/rader_reranker_diag")
    ap.add_argument(
        "--dtype",
        choices=["bfloat16", "float32", "float16"],
        default="bfloat16",
        help="hf backend only: precision for BOTH the LoRA merge and the "
        "forward pass. bfloat16 is production (and rerank.py's "
        "get_model_new); float32 matches rerank.py's OTHER loader, get_model, "
        "which passes no torch_dtype at all. This matters because the merged "
        "LoRA delta is small relative to bf16's resolution: mean|delta| is "
        "~3.8x the bf16 quantisation step near a typical Qwen2.5 weight, so "
        "storing W+delta in bf16 rounds away a few percent of the update. "
        "float32 removes that entirely; float16 keeps the same range as bf16 "
        "with a 10-bit rather than 8-bit mantissa.",
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS,
        choices=TASKS,
        help="Subset of tasks (default: both). fp32 is ~3x slower, so a "
        "precision arm is usually worth running on statement-statement first.",
    )
    args = ap.parse_args()

    queries_ds, documents_ds = load_data()
    idxs = sorted(random.Random(args.seed).sample(range(len(queries_ds)), args.n))
    print(f"[~] {len(idxs)} queries (seed {args.seed}); arms {args.arms}")

    if args.backend == "hf":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftConfig, PeftModel

        lora = "Raderspace/reranker_Qwen25_7B_NuminaMath_MATH_allquerytypes"
        base_name = PeftConfig.from_pretrained(lora).base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(base_name)
        dt = getattr(torch, args.dtype)
        print(f"[~] hf backend precision: {args.dtype} (merge AND forward)")
        model = AutoModelForSequenceClassification.from_pretrained(
            base_name, num_labels=1, torch_dtype=dt
        )
        model = PeftModel.from_pretrained(model, lora).merge_and_unload()
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        model.eval().cuda()

        def score(all_ids):
            out = []
            for s in range(0, len(all_ids), 8):
                batch = tokenizer.pad(
                    {"input_ids": all_ids[s : s + 8]}, padding=True, return_tensors="pt"
                )
                batch = {k: v.cuda() for k, v in batch.items()}
                with torch.no_grad():
                    out.extend(model(**batch).logits[:, 0].float().cpu().tolist())
            return out
    else:
        from sabermath.processors import RaDeRRerankerVLLMProcessor
        from vllm import TokensPrompt

        proc = RaDeRRerankerVLLMProcessor(max_length=args.max_length)
        proc._init()
        tokenizer = proc._tokenizer

        def score(all_ids):
            outs = proc._llm.classify(
                [TokensPrompt(prompt_token_ids=i) for i in all_ids], use_tqdm=False
            )
            return [float(o.outputs.probs[0]) for o in outs]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for task in args.tasks:
        qv = "full" if task == "full-full" else "statement"
        dv = "statement" if task == "statement-statement" else "full"
        queries = transform(queries_ds, qv)
        for arm in args.arms:
            per_query, dump = [], {}
            for i in idxs:
                doc_ids = list(queries_ds[i]["candidates"])
                rel = np.asarray(queries_ds[i]["relevance_scores"], dtype=float)
                docs = transform(documents_ds, dv, doc_ids)
                ids = [
                    build_ids(tokenizer, arm, queries[i], d, args.max_length)
                    for d in docs
                ]
                s = np.asarray(score(ids))
                order = np.argsort(-s)
                per_query.append(ndcg_at_10(rel, order))
                dump[str(i)] = [float(x) for x in s]
            m = float(np.mean(per_query))
            summary[f"{task}::{arm}"] = m
            print(f"[=] {task:22s} {arm} ({args.backend}) nDCG@10 = {m:.4f}"
                  f"   # {ARM_NOTES[arm]}", flush=True)
            dtag = "" if args.dtype == "bfloat16" else f"-{args.dtype}"
            tag = f"{args.backend}{dtag}__{task}__{arm}"
            (out_dir / f"scores__{tag}.json").write_text(json.dumps(dump))
            (out_dir / f"summary__{args.backend}{dtag}.json").write_text(
                json.dumps({"n": args.n, "seed": args.seed, "dtype": args.dtype,
                            "query_idxs": idxs, "results": summary}, indent=1)
            )
    print("[+] done")


if __name__ == "__main__":
    main()
