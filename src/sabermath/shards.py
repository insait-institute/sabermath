from __future__ import annotations

import json
import re
from pathlib import Path
from typing import get_args

import numpy as np

from .schemas import Branch

BRANCHES = list(get_args(Branch))

SHARD_RE = re.compile(r"__shard\d+of\d+")

TOP_K_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]


def base_name(path: Path) -> str:
    return SHARD_RE.sub("", path.name)


def group_shards(directory: Path) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for path in sorted(directory.glob("*__shard*of*.json")):
        groups.setdefault(directory / base_name(path), []).append(path)
    return groups


def default_out(paths: list[Path]) -> Path:
    names = {base_name(p) for p in paths}
    if len(names) != 1:
        raise SystemExit(
            "Parts do not share one base name (differing model/prompt/subset?): "
            + ", ".join(sorted(names))
        )
    return paths[0].parent / names.pop()


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"Could not read {path}: {e}")


def _write(out: Path, payload: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out)


def load_evaluation_parts(paths: list[Path]) -> tuple[str, list[dict]]:
    parts, model_keys = [], set()
    for p in paths:
        data = _read(p)
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


def merge_evaluation(model_key: str, parts: list[dict]) -> tuple[dict, dict]:
    domains_by_idx: dict[int, list[str]] = {}
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


def merge_evaluation_shards(
    paths: list[Path],
    out: Path | None = None,
    dry_run: bool = False,
    allow_incomplete: bool = False,
) -> bool:
    paths = sorted(paths)
    if len(paths) < 2:
        print(f"[~] Merging a single part ({paths[0].name}) - nothing to stitch.")
    model_key, parts = load_evaluation_parts(paths)
    merged, coverage = merge_evaluation(model_key, parts)

    print(f"model: {model_key}")
    print(f"parts: {len(parts)}")
    for p in parts:
        print(f"  {p['path'].name}: {len(p['data']['query_row_idxs'])} queries")
    incomplete = False
    tasks = merged["reports"][model_key]["tasks"]
    for task, cov in coverage.items():
        done, total = cov["n_done"], cov["n_total"]
        flag = "" if done == total else "   <- INCOMPLETE"
        ndcg = next(t["ndcg_at_k"] for t in tasks if t["task"] == task)
        print(f"  {task:<22} nDCG@10={ndcg:.4f}  {done}/{total}{flag}")
        incomplete |= done != total

    if dry_run:
        print("[~] --dry-run: nothing written.")
        return False
    if incomplete and not allow_incomplete:
        raise SystemExit(
            "Refusing to write an incomplete merge (pass --allow-incomplete to "
            "override). Some queries are missing from every part - a shard is "
            "still running or died."
        )

    destination = out or default_out(paths)
    _write(destination, merged)
    print(f"[+] Wrote {destination}")
    return True


def summarize_ranks(ranks) -> dict:
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


def merge_dedup(parts: list[tuple[Path, dict]]) -> tuple[dict, list[int]]:
    keys = {(d["model"], d["regime"]) for _, d in parts}
    if len(keys) != 1:
        raise SystemExit(f"Parts differ in model/regime: {sorted(keys)}")

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

    out = dict(parts[0][1])
    out["n_queries"] = len(idxs)
    out["insert_per_query"] = summarize_ranks(ranks)
    out["insert_all_rephrased"] = (
        summarize_ranks(with_all) if all(v is not None for v in with_all) else None
    )
    out["per_query"] = {str(i): merged[str(i)] for i in idxs}
    out["n_self_matches_excluded"] = sum(
        d.get("n_self_matches_excluded", 0) for _, d in parts
    )
    out["merged_from"] = [
        {"file": str(p), "n_queries": len(d["per_query"])} for p, d in parts
    ]

    present = set(idxs)
    gaps = [i for i in range(idxs[0], idxs[-1] + 1) if i not in present]
    return out, gaps


def merge_dedup_shards(
    paths: list[Path],
    out: Path | None = None,
    dry_run: bool = False,
    allow_incomplete: bool = False,
) -> bool:
    paths = sorted(paths)
    parts = [(p, _read(p)) for p in paths]
    merged, gaps = merge_dedup(parts)

    print(f"model: {merged['model']}  regime: {merged['regime']}")
    for p, d in parts:
        print(f"  {p.name}: {len(d['per_query'])} queries")
    s = merged["insert_per_query"]
    print(
        f"  merged {merged['n_queries']} queries -> top1={s['top_1']:.3f} "
        f"top8={s['top_8']:.3f} median={s['median_rank']:.0f} "
        f"mean={s['avg_rank']:.2f}"
    )
    if gaps:
        print(f"  [!!] {len(gaps)} query indices missing from every shard")

    if dry_run:
        print("[~] --dry-run: nothing written.")
        return False
    if gaps and not allow_incomplete:
        raise SystemExit(
            "Refusing to write an incomplete merge (pass --allow-incomplete to "
            "override)."
        )

    destination = out or default_out(paths)
    _write(destination, merged)
    print(f"[+] Wrote {destination}")
    return True


def add_merge_arguments(parser, default_dir: str) -> None:
    group = parser.add_argument_group("merging a sharded sweep")
    group.add_argument(
        "--merge-shards",
        nargs="*",
        metavar="PART",
        default=None,
        help="Stitch --query-shards parts back together and exit. With no "
        f"arguments, merges every complete shard group in {default_dir}; "
        "otherwise merges exactly the part files given.",
    )
    group.add_argument(
        "--merge-out",
        type=Path,
        default=None,
        help="Destination for the merge (default: the parts' name without "
        "__shard<i>of<n>). Only valid when merging one explicit group.",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="With --merge-shards: report what the merge would produce, write "
        "nothing.",
    )
    group.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="With --merge-shards: write even when some queries are missing "
        "from every part (default: refuse, so a half-finished sweep is never "
        "mistaken for a full run).",
    )


def run_merge(args, merge_one, directory: Path) -> None:
    if args.merge_shards:
        merge_one(
            [Path(p) for p in args.merge_shards],
            out=args.merge_out,
            dry_run=args.dry_run,
            allow_incomplete=args.allow_incomplete,
        )
        return

    if args.merge_out is not None:
        raise SystemExit(
            "--merge-out needs an explicit list of part files; a scan can "
            "write several destinations."
        )
    groups = group_shards(directory)
    if not groups:
        print(f"[~] No __shard<i>of<n> files in {directory} - nothing to merge.")
        return
    print(f"Found {len(groups)} shard group(s) in {directory}.\n")
    written, skipped = 0, []
    for destination, paths in groups.items():
        print(f"--- {destination.name} ---")
        try:
            written += merge_one(
                paths,
                out=destination,
                dry_run=args.dry_run,
                allow_incomplete=args.allow_incomplete,
            )
        except SystemExit as e:
            print(f"[!!] skipped: {e}")
            skipped.append(destination.name)
        print()
    print(f"Merged {written}/{len(groups)} group(s).")
    if skipped:
        print(f"Skipped: {', '.join(skipped)}")
