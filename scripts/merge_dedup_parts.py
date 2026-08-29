"""Stitch the shard files of a sharded run_dedup.py sweep into one result.

run_dedup.py's --query-shards/--query-shard splits the query set into strided
shards, each with its own checkpoint and its own
"<model>[__all-documents]__shard<i>of<n>.json". This merges them back: the
per-query records are unioned, and every summary statistic is RECOMPUTED from
the union rather than averaged across shards - avg_rank, the percentiles and
the top-k coverages are all non-linear in the query set, so averaging shard
summaries would be wrong even when the shards are equal-sized.

Usage:
    python scripts/merge_dedup_parts.py results/dedup/<model>__shard*of5.json
    python scripts/merge_dedup_parts.py results/dedup/<model>__shard*of5.json --out results/dedup/<model>.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402

TOP_K_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]
SHARD_RE = re.compile(r"__shard\d+of\d+(?=\.json$)")


def summarize(ranks):
    """Byte-for-byte the same statistics run_dedup.summarize computes (kept in
    sync deliberately; it is a dozen lines and importing run_dedup would drag
    in torch/vllm just to merge JSON)."""
    arr = np.asarray(ranks, dtype=float)
    out = {
        "avg_rank": float(arr.mean()),
        "median_rank": float(np.median(arr)),
        "trimmed_mean_rank_95": float(arr[arr <= np.percentile(arr, 95)].mean()),
        "p90_rank": float(np.percentile(arr, 90)),
        "p99_rank": float(np.percentile(arr, 99)),
        "max_rank": float(arr.max()),
    }
    for k in TOP_K_LEVELS:
        out[f"top_{k}"] = float(np.mean(arr <= k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write even if the shards do not cover a contiguous query set.",
    )
    args = ap.parse_args()

    paths = sorted(args.parts)
    parts = []
    for p in paths:
        try:
            parts.append((p, json.loads(p.read_text())))
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"Could not read {p}: {e}")

    keys = {(d["model"], d["regime"]) for _, d in parts}
    if len(keys) != 1:
        raise SystemExit(f"Parts differ in model/regime: {sorted(keys)}")
    model, regime = keys.pop()

    merged, owner = {}, {}
    for p, d in parts:
        for qid, rec in d["per_query"].items():
            if qid in merged:
                raise SystemExit(
                    f"Query {qid} appears in both {owner[qid].name} and "
                    f"{p.name} - the shards overlap."
                )
            merged[qid] = rec
            owner[qid] = p

    idxs = sorted(int(k) for k in merged)
    ranks = [merged[str(i)]["rank_corpus_only"] for i in idxs]
    with_all = [merged[str(i)]["rank_with_all_rephrased"] for i in idxs]

    base = parts[0][1]
    out = dict(base)
    out["n_queries"] = len(idxs)
    out["insert_per_query"] = summarize(ranks)
    out["insert_all_rephrased"] = (
        summarize(with_all) if all(v is not None for v in with_all) else None
    )
    out["per_query"] = {str(i): merged[str(i)] for i in idxs}
    # n_self_matches_excluded is a per-shard count of that shard's queries;
    # summing is correct because the shards partition the query set.
    out["n_self_matches_excluded"] = sum(
        d.get("n_self_matches_excluded", 0) for _, d in parts
    )
    out["merged_from"] = [
        {"file": str(p), "n_queries": len(d["per_query"])} for p, d in parts
    ]

    print(f"model: {model}  regime: {regime}")
    for p, d in parts:
        print(f"  {p.name}: {len(d['per_query'])} queries")
    s = out["insert_per_query"]
    print(
        f"  merged {len(idxs)} queries -> top1={s['top_1']:.3f} "
        f"top8={s['top_8']:.3f} median={s['median_rank']:.0f} mean={s['avg_rank']:.2f}"
    )

    gaps = [i for i in range(idxs[0], idxs[-1] + 1) if i not in set(idxs)]
    if gaps:
        print(f"  [!!] {len(gaps)} query indices missing from every shard")
        if not args.allow_incomplete and not args.dry_run:
            raise SystemExit("Refusing to write an incomplete merge (--allow-incomplete to override).")

    if args.dry_run:
        print("[~] --dry-run: nothing written.")
        return

    names = {SHARD_RE.sub("", p.name) for p in paths}
    if args.out is None and len(names) != 1:
        raise SystemExit(f"Parts do not share one base name: {sorted(names)}")
    out_path = args.out or paths[0].parent / names.pop()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(out_path)
    print(f"[+] Wrote {out_path}")


if __name__ == "__main__":
    main()
