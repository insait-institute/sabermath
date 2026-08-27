"""Compare rank1-32b at the production float16 against its native bfloat16.

    python scripts/compare_rank1_dtype.py

rank1-32b's checkpoint is bfloat16. Rank1Processor pins vLLM to
dtype="float16" (inherited from the model card's example), so every
production run logs "Casting torch.bfloat16 to torch.float16". bf16 and fp16
both have 16 bits but spend them differently: bf16 keeps float32's 8-bit
exponent and drops mantissa, fp16 keeps 10 mantissa bits and narrows the
exponent to 5 (max ~65504). Downcasting bf16 weights to fp16 is therefore
lossy in RANGE, not just precision, and any activation that would have
exceeded fp16's max saturates to inf.

Whether that costs accuracy is an empirical question, and this answers it on
the main benchmark rather than by argument: two runs of the SAME model over
the SAME --n/--seed query subset, differing only in dtype (verified: the two
CUSTOM_MODEL_BUILDERS entries agree on model name, tensor_parallel_size,
max_model_len and max_thinking_tokens).

Reads the paired per-query nDCG lists, so it reports a PAIRED comparison -
mean deltas on a 50-query subset are noisy enough that the per-query win/loss
split and the count of queries that actually moved matter more than the
difference of the two means.
"""

import argparse
import json
import math
from pathlib import Path

TASKS = ("statement-statement", "statement-full", "full-full")


def load(path: Path, key: str | None = None):
    """Read one result file. The model key is taken FROM the file rather than
    assumed: these files carry exactly one report, and hardcoding the key
    breaks the moment a run is renamed or a variant key is added."""
    d = json.loads(path.read_text())
    if key is None:
        keys = list(d["reports"])
        if len(keys) != 1:
            raise ValueError(f"{path.name}: expected 1 report, got {keys}")
        key = keys[0]
    rep = d["reports"][key]
    by_task = {t["task"]: t["ndcg_at_k"] for t in rep["tasks"]}
    return d, rep, by_task, rep.get("ndcgs_by_task", {})


def paired(a, b):
    """Win/loss/tie split plus the mean paired delta and its standard error.

    ndcgs_by_task is a FIXED-LENGTH list padded with None for queries that
    have not been scored yet (these files are snapshotted mid-run), so the
    two arms are aligned on indices where BOTH sides have a value. Comparing
    a partial run against a complete one otherwise silently mixes scored and
    unscored positions."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "moved": 0, "wins": 0, "losses": 0, "mean_delta": 0.0,
                "se": float("nan"), "max_gain": 0.0, "min_delta": 0.0}
    deltas = [y - x for x, y in pairs]  # bf16 minus fp16
    moved = [d for d in deltas if abs(d) > 1e-12]
    wins = sum(1 for d in moved if d > 0)
    losses = sum(1 for d in moved if d < 0)
    mean = sum(deltas) / n
    if n > 1:
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = float("nan")
    return {"n": n, "moved": len(moved), "wins": wins, "losses": losses,
            "mean_delta": mean, "se": se,
            "max_gain": max(deltas) if deltas else 0.0,
            "min_delta": min(deltas) if deltas else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/rerankers")
    # Globs, not fixed names: run_rerankers bakes the subset key into the
    # filename (rank1-32b-bf16__p0__n50_seed42__part-n50bf16.json), so any
    # hardcoded name silently becomes "missing file" the moment --n or --seed
    # changes.
    ap.add_argument("--fp16", default="rank1-32b__p0__*part-n50fp16.json")
    ap.add_argument("--bf16", default="rank1-32b-bf16__p0__*part-n50bf16.json")
    args = ap.parse_args()

    root = Path(args.dir)

    def one(pattern, label):
        hits = sorted(root.glob(pattern))
        if not hits:
            print(f"[!!] no {label} result matching {root}/{pattern}")
            return None
        if len(hits) > 1:
            print(f"[!!] {len(hits)} {label} files match {pattern}; "
                  f"refusing to guess:")
            for h in hits:
                print(f"      {h.name}")
            return None
        return hits[0]

    fp, bp = one(args.fp16, "fp16"), one(args.bf16, "bf16")
    if fp is None or bp is None:
        return
    print(f"fp16 file: {fp.name}\nbf16 file: {bp.name}\n")

    df, rf, nf, qf = load(fp)
    db, rb, nb, qb = load(bp)

    print(f"fp16 (production): {rf['model']}  processor={rf['processor']}")
    print(f"bf16 (native)    : {rb['model']}  processor={rb['processor']}")
    print(f"dcg_variant={rf['dcg_variant']} k={rf['k']}  "
          f"(bf16: {rb['dcg_variant']} k={rb['k']})\n")

    # The comparison is only meaningful if both runs scored the SAME queries.
    # --n/--seed should guarantee it; checked rather than trusted, because a
    # silent subset mismatch would look exactly like a dtype effect.
    same = df.get("domains") == db.get("domains")
    print(f"identical query subset (domains list matches): {same}"
          f"   n={len(df.get('domains', []))}")
    if not same:
        print("  [!!] subsets DIFFER - the comparison below is not paired; "
              "re-run both arms with the same --n and --seed.")

    # Means are recomputed over the OVERLAPPING scored indices, not taken
    # from each file's own ndcg_at_k. While one arm is still running its
    # reported mean covers fewer queries than the other's, so differencing
    # the two published numbers compares different query sets and invents a
    # delta. Recomputing on the shared subset keeps every column paired.
    print(f"\n{'task':<22}{'fp16':>9}{'bf16':>9}{'delta':>10}   n   paired split")
    print("-" * 84)
    means = []
    for t in TASKS:
        if t not in qf or t not in qb:
            continue
        both = [(x, y) for x, y in zip(qf[t], qb[t])
                if x is not None and y is not None]
        if not both:
            print(f"{t:<22}{'-':>9}{'-':>9}{'-':>10}   0   no overlap yet")
            continue
        a = sum(x for x, _ in both) / len(both)
        b = sum(y for _, y in both) / len(both)
        means.append((a, b))
        st = paired(qf[t], qb[t])
        flag = "" if len(both) == len(qf[t]) else " *"
        print(f"{t:<22}{a:>9.4f}{b:>9.4f}{b-a:>+10.4f}{len(both):>4}   "
              f"{st['wins']}W/{st['losses']}L, {st['moved']} moved{flag}")
    print("-" * 84)
    if means:
        ma = sum(x for x, _ in means) / len(means)
        mb = sum(y for _, y in means) / len(means)
        label = f"{len(means)}-task mean"
        print(f"{label:<22}{ma:>9.4f}{mb:>9.4f}{mb-ma:>+10.4f}")

    partial = [t for t in TASKS if t in qf and t in qb
                and any(x is None or y is None for x, y in zip(qf[t], qb[t]))]
    if partial:
        print(f"\n  * PARTIAL - one arm is still running these tasks "
              f"({', '.join(partial)}); the rows above are paired on the "
              f"shared subset, so they are valid but will move as the rest "
              f"lands.")

    print("\n=== paired per-query detail (bf16 - fp16) ===")
    for t in TASKS:
        if t in qf and t in qb:
            s = paired(qf[t], qb[t])
            if not s["n"]:
                print(f"  {t:<22} no overlapping scored queries yet")
                continue
            print(f"  {t:<22} mean {s['mean_delta']:+.5f} +/- {s['se']:.5f} (SE) "
                  f"| compared {s['n']}/{len(qf[t])} | moved {s['moved']} "
                  f"| best {s['max_gain']:+.4f} worst {s['min_delta']:+.4f}")

    print("\n  A dtype that mattered would show a CONSISTENT sign across tasks "
          "and\n  many moved queries. Scattered wins and losses of similar size "
          "with the\n  mean delta inside its own standard error is decoding "
          "noise, not precision.")


if __name__ == "__main__":
    main()
