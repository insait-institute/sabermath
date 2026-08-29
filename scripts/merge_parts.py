"""Stitch the shard files of one sharded run back into a single result file.

run_rerankers.py's --query-shards/--query-shard splits the query set into
strided shards, each written to its own
"<model>__<prompt>__shard<i>of<n>.json" with its own checkpoints, so a slow
model can be run as N concurrent jobs (see _select_shard). Three places in
that script tell you to merge the pieces "with scripts/merge_parts.py" - this
is that script, which did not previously exist.

What it does, per task found in the parts: place every shard's per-query
nDCG at its GLOBAL query index (each part carries "query_row_idxs" for
exactly this), then recompute the overall mean and the per-domain means from
the reassembled array. Recomputing rather than averaging the shards' own
numbers matters: a shard's mean is over its own ~1/N of the queries, and
averaging N such means is only equal to the true mean when the shards are
exactly equal-sized and every query is scored - neither is guaranteed (the
stride leaves shards differing by one, and an unfinished shard has holes).

The output has the same shape run_rerankers.py writes - {"domains": [...],
"reports": {<model>: {..., "tasks": [...], "ndcgs_by_task": {...}}}} - so
testing/compute_confidence_intervals.py reads it like any other result file.
It also records "merged_from" provenance: the part files, their shard ids,
and the coverage actually achieved.

Usage:
    # merge, writing <model>__<prompt>.json next to the parts
    python scripts/merge_parts.py results/rerankers/retro-star-32b__p0__shard*of5.json

    # merge to an explicit destination
    python scripts/merge_parts.py results/rerankers/<model>__p0__shard*of5.json \
        --out results/rerankers/<model>.json

    # check coverage without writing anything
    python scripts/merge_parts.py results/rerankers/<model>__p0__shard*of5.json --dry-run
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sabermath.schemas import Branch  # noqa: E402
from typing import get_args  # noqa: E402

BRANCHES = list(get_args(Branch))

SHARD_SUFFIX_RE = re.compile(r"__shard\d+of\d+(?=\.json$)")


def default_out(paths: list[Path]) -> Path:
    """The parts' own name with the __shard<i>of<n> component removed. Kept
    deliberately conservative: it never strips the prompt key, so merging p0
    shards yields <model>__p0.json and cannot silently overwrite a
    production <model>.json - pass --out for that."""
    names = {SHARD_SUFFIX_RE.sub("", p.name) for p in paths}
    if len(names) != 1:
        raise SystemExit(
            "Parts do not share one base name (differing model/prompt/subset?): "
            + ", ".join(sorted(names))
        )
    return paths[0].parent / names.pop()


def load_parts(paths: list[Path]) -> tuple[str, list[dict]]:
    parts, model_keys = [], set()
    for p in paths:
        try:
            data = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"Could not read {p}: {e}")
        reports = data.get("reports") or {}
        if len(reports) != 1:
            raise SystemExit(f"{p}: expected exactly one report, got {list(reports)}")
        key = next(iter(reports))
        if "error" in reports[key]:
            raise SystemExit(
                f"{p} holds an error, not results: {reports[key]['error'][:200]}"
            )
        if data.get("query_row_idxs") is None:
            raise SystemExit(
                f"{p} has no 'query_row_idxs' - it is not a shard file "
                "(only --query-shards runs record global indices)."
            )
        model_keys.add(key)
        parts.append({"path": p, "data": data, "key": key})
    if len(model_keys) != 1:
        raise SystemExit(f"Parts are from different models: {sorted(model_keys)}")
    return model_keys.pop(), parts


def merge(model_key: str, parts: list[dict]) -> tuple[dict, dict]:
    # Global index -> domains, assembled from the parts themselves so this
    # script never needs the dataset.
    domains_by_idx: dict[int, list[str]] = {}
    # task -> {global index: ndcg}
    ndcgs: dict[str, dict[int, float]] = {}
    owner: dict[tuple[str, int], Path] = {}

    for part in parts:
        data, path = part["data"], part["path"]
        idxs = list(data["query_row_idxs"])
        part_domains = list(data.get("domains") or [])
        if part_domains and len(part_domains) != len(idxs):
            raise SystemExit(
                f"{path}: {len(part_domains)} domain rows for {len(idxs)} queries."
            )
        for local, gidx in enumerate(idxs):
            if part_domains:
                prev = domains_by_idx.get(gidx)
                if prev is not None and prev != part_domains[local]:
                    raise SystemExit(
                        f"Query {gidx} has conflicting domains across parts."
                    )
                domains_by_idx[gidx] = part_domains[local]

        report = data["reports"][model_key]
        for task, values in (report.get("ndcgs_by_task") or {}).items():
            if len(values) != len(idxs):
                raise SystemExit(
                    f"{path}: task '{task}' has {len(values)} scores for "
                    f"{len(idxs)} queries."
                )
            slot = ndcgs.setdefault(task, {})
            for local, gidx in enumerate(idxs):
                value = values[local]
                if value is None:
                    continue
                if gidx in slot:
                    raise SystemExit(
                        f"Query {gidx} of task '{task}' is scored in both "
                        f"{owner[(task, gidx)].name} and {path.name} - the "
                        "parts overlap, so they are not a clean shard set."
                    )
                slot[gidx] = value
                owner[(task, gidx)] = path

    n_total = (max(domains_by_idx) + 1) if domains_by_idx else 0
    domains = [domains_by_idx.get(i, []) for i in range(n_total)]

    tasks_out, ndcgs_by_task, coverage = [], {}, {}
    for task in sorted(ndcgs):
        by_idx = ndcgs[task]
        all_ndcgs = [by_idx.get(i) for i in range(n_total)]
        ndcgs_by_task[task] = all_ndcgs
        valid = [v for v in all_ndcgs if v is not None]
        coverage[task] = {"n_done": len(valid), "n_total": n_total}

        branches = []
        for branch in BRANCHES:
            vals = [
                by_idx[i]
                for i in range(n_total)
                if i in by_idx and branch in domains[i]
            ]
            branches.append(
                {
                    "branch": branch,
                    "ndcg_at_k": float(sum(vals) / len(vals)) if vals else 0.0,
                }
            )

        entry = {
            "task": task,
            "ndcg_at_k": float(sum(valid) / len(valid)) if valid else 0.0,
            "branches": branches,
        }
        # Same convention as run_rerankers.py: n_done/n_total are present
        # ONLY on an incomplete result, so a partial merge can never be
        # mistaken for a finished one.
        if len(valid) != n_total:
            entry["n_done"] = len(valid)
            entry["n_total"] = n_total
        tasks_out.append(entry)

    first = parts[0]["data"]["reports"][model_key]
    report = {
        "model": first.get("model"),
        "processor": first.get("processor"),
        "dcg_variant": first.get("dcg_variant"),
        "k": first.get("k"),
        "tasks": tasks_out,
        "ndcgs_by_task": ndcgs_by_task,
    }
    if "prompt" in first:
        report["prompt"] = first["prompt"]
    report["merged_from"] = [
        {
            "file": str(p["path"]),
            "part": p["data"].get("part"),
            "n_queries": len(p["data"]["query_row_idxs"]),
        }
        for p in parts
    ]
    return {"domains": domains, "reports": {model_key: report}}, coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="+", type=Path, help="Shard result files")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination (default: the parts' name without __shard<i>of<n>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what the merge would produce; write nothing.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write even when some queries are missing from every part "
        "(default: refuse, so a half-finished sweep is never mistaken for a "
        "full run).",
    )
    args = parser.parse_args()

    paths = sorted(args.parts)
    if len(paths) < 2:
        print(f"[~] Merging a single part ({paths[0].name}) - nothing to stitch.")
    model_key, parts = load_parts(paths)
    merged, coverage = merge(model_key, parts)

    print(f"model: {model_key}")
    print(f"parts: {len(parts)}")
    for p in parts:
        print(f"  {p['path'].name}: {len(p['data']['query_row_idxs'])} queries")
    incomplete = False
    for task, cov in coverage.items():
        done, total = cov["n_done"], cov["n_total"]
        flag = "" if done == total else "   <- INCOMPLETE"
        ndcg = next(t["ndcg_at_k"] for t in merged["reports"][model_key]["tasks"] if t["task"] == task)
        print(f"  {task:<22} nDCG@10={ndcg:.4f}  {done}/{total}{flag}")
        incomplete |= done != total

    if args.dry_run:
        print("[~] --dry-run: nothing written.")
        return
    if incomplete and not args.allow_incomplete:
        raise SystemExit(
            "Refusing to write an incomplete merge (pass --allow-incomplete to "
            "override). Some queries are missing from every part - a shard is "
            "still running or died."
        )

    out = args.out or default_out(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2))
    tmp.replace(out)
    print(f"[+] Wrote {out}")


if __name__ == "__main__":
    main()
