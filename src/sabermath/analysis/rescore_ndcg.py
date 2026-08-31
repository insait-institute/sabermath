import argparse
import json
import sys
from pathlib import Path
from typing import get_args

import numpy as np

from ..data import load_data
from ..metrics import compute_ndcg_at_k
from ..schemas import Branch

BRANCHES = list(get_args(Branch))

DEFAULT_VARIANTS = ["linear:1.0", "exponent:1.0", "exponent:0.6"]


def parse_variant(token: str) -> tuple[str, float]:
    if token == "rank":
        return ("rank", 1.0)
    if ":" not in token:
        raise argparse.ArgumentTypeError(
            f"Variant '{token}' must look like 'exponent:1.0', 'linear:1.0' "
            "or be the bare token 'rank'."
        )
    variant, scale_str = token.split(":", 1)
    if variant == "exponential":
        raise argparse.ArgumentTypeError(
            "'exponential' is not a valid DCG variant name (this exact typo "
            "silently killed the original rescaling run) - use 'exponent'."
        )
    if variant not in ("linear", "exponent"):
        raise argparse.ArgumentTypeError(
            f"Unknown DCG variant '{variant}' - use 'linear', 'exponent' or "
            "'rank'."
        )
    try:
        scale = float(scale_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Bad scale in variant '{token}'."
        ) from e
    return (variant, scale)


def variant_label(variant: str, scale: float) -> str:
    if variant == "rank":
        return "rank"
    return f"{variant}:{scale:g}"


def rank_transform(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    for v in np.unique(values):
        mask = values == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def find_run_dirs(scan_root: Path) -> list[Path]:
    return sorted(
        meta.parent for meta in scan_root.glob(".checkpoints/*/*/meta.json")
    )


def load_scores_file(path: Path) -> dict[int, dict | None]:
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def load_ndcg_checkpoint(path: Path) -> dict[int, float | None]:
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def rescore_run(
    run_dir: Path,
    variants: list[tuple[str, float]],
    k: int,
    relevance_by_row: list[list[float]],
    domains_by_row: list[list[str]],
) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    row_idxs = meta.get("query_row_idxs")

    tasks_out: dict[str, dict] = {}
    per_query_out: dict[str, dict] = {}
    checks = {
        "exact_match_exponent_1": {"checked": 0, "max_abs_delta": 0.0},
        "linear_scale_invariance": {"checked": 0, "max_abs_delta": 0.0},
        "ranking_argsort_mismatches": 0,
    }

    for task in meta["tasks"]:
        scores_file = run_dir / f"{task}.scores.json"
        if not scores_file.exists():
            continue
        stored = load_scores_file(scores_file)

        ndcg_file = run_dir / f"{task}.json"
        stored_ndcgs = (
            load_ndcg_checkpoint(ndcg_file) if ndcg_file.exists() else {}
        )

        per_variant_query_ndcgs: dict[str, dict[int, float]] = {
            variant_label(v, s): {} for v, s in variants
        }
        linear_reference: dict[int, float] = {}

        for pos, entry in stored.items():
            if entry is None:
                continue
            row = row_idxs[pos] if row_idxs is not None else pos
            relevances = np.asarray(relevance_by_row[row], dtype=float)
            ranking = np.asarray(entry["ranking"], dtype=int)
            raw_scores = np.asarray(entry["scores"], dtype=float)

            re_argsort = np.argsort(-raw_scores)
            if not np.array_equal(re_argsort, ranking):
                checks["ranking_argsort_mismatches"] += 1

            ranked_relevances = relevances[ranking]

            for variant, scale in variants:
                label = variant_label(variant, scale)
                if variant == "rank":
                    transformed = rank_transform(relevances)[ranking]
                    ndcg = compute_ndcg_at_k(transformed, k=k, variant="linear")
                else:
                    ndcg = compute_ndcg_at_k(
                        scale * ranked_relevances, k=k, variant=variant
                    )
                per_variant_query_ndcgs[label][pos] = ndcg

                if variant == "exponent" and scale == 1.0:
                    reference = stored_ndcgs.get(pos)
                    if reference is not None:
                        delta = abs(ndcg - reference)
                        checks["exact_match_exponent_1"]["checked"] += 1
                        checks["exact_match_exponent_1"]["max_abs_delta"] = max(
                            checks["exact_match_exponent_1"]["max_abs_delta"],
                            delta,
                        )

            linear_check = compute_ndcg_at_k(
                ranked_relevances, k=k, variant="linear"
            )
            linear_scaled = compute_ndcg_at_k(
                0.6 * ranked_relevances, k=k, variant="linear"
            )
            checks["linear_scale_invariance"]["checked"] += 1
            checks["linear_scale_invariance"]["max_abs_delta"] = max(
                checks["linear_scale_invariance"]["max_abs_delta"],
                abs(linear_check - linear_scaled),
            )
            linear_reference[pos] = linear_check

        by_row = {
            label: {
                (row_idxs[pos] if row_idxs is not None else pos): ndcg
                for pos, ndcg in query_ndcgs.items()
            }
            for label, query_ndcgs in per_variant_query_ndcgs.items()
        }
        per_query_out[task] = by_row

        variant_results = {}
        for label, row_ndcgs in by_row.items():
            if not row_ndcgs:
                continue
            variant_results[label] = summarize_rows(row_ndcgs, domains_by_row)

        tasks_out[task] = variant_results

    return {
        "model_key": meta["model_key"],
        "instruction_key": meta.get("instruction_key"),
        "protocol_tag": meta.get("protocol_tag", ""),
        "part": meta.get("part"),
        "query_shard": meta.get("query_shard"),
        "query_shards": meta.get("query_shards"),
        "n": meta.get("n"),
        "seed": meta.get("seed"),
        "k": k,
        "run_dir": str(run_dir),
        "tasks": tasks_out,
        "per_query": per_query_out,
        "self_checks": checks,
    }


def summarize_rows(row_ndcgs: dict[int, float], domains_by_row) -> dict:
    values = list(row_ndcgs.values())
    branches = []
    for branch in BRANCHES:
        branch_values = [
            ndcg
            for row, ndcg in row_ndcgs.items()
            if branch in domains_by_row[row]
        ]
        branches.append(
            {
                "branch": branch,
                "ndcg_at_k": (
                    float(np.mean(branch_values)) if branch_values else 0.0
                ),
            }
        )
    return {
        "ndcg_at_k": float(np.mean(values)),
        "n_queries": len(values),
        "branches": branches,
    }


def pool_results(results: list[dict], domains_by_row) -> dict:
    """Pool several part/shard runs of the same (model, prompt, protocol)
    cell into one set of numbers. Per-query nDCGs are keyed by GLOBAL query
    row, so pooling is a dict union - never an average of averages, which
    would be wrong the moment two shards differ in size."""
    pooled: dict[str, dict[str, dict[int, float]]] = {}
    for result in results:
        for task, by_label in result["per_query"].items():
            task_slot = pooled.setdefault(task, {})
            for label, row_ndcgs in by_label.items():
                task_slot.setdefault(label, {}).update(row_ndcgs)

    tasks_out = {
        task: {
            label: summarize_rows(row_ndcgs, domains_by_row)
            for label, row_ndcgs in sorted(by_label.items())
            if row_ndcgs
        }
        for task, by_label in sorted(pooled.items())
    }

    head = results[0]
    return {
        "model_key": head["model_key"],
        "instruction_key": head["instruction_key"],
        "protocol_tag": head.get("protocol_tag", ""),
        "n": head.get("n"),
        "seed": head.get("seed"),
        "k": head["k"],
        "run_dirs": [r["run_dir"] for r in results],
        "tasks": tasks_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--runs", type=Path, nargs="+", default=None)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--variants",
        type=parse_variant,
        nargs="+",
        default=[parse_variant(v) for v in DEFAULT_VARIANTS],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/rescaled"))
    parser.add_argument("--export-table", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    run_dirs = args.runs if args.runs is not None else find_run_dirs(args.scan)
    if not run_dirs:
        raise SystemExit(f"No runs with meta.json found under {args.scan}")

    queries, _ = load_data()
    relevance_by_row = list(queries["relevance_scores"])
    domains_by_row = list(queries["domains"])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    failed_checks = []

    for run_dir in run_dirs:
        result = rescore_run(
            Path(run_dir), args.variants, args.k, relevance_by_row, domains_by_row
        )
        if not result["tasks"]:
            print(f"[~] {run_dir}: no scores files - skipped")
            continue
        all_results.append(result)

        subset = (
            "" if result["n"] is None else f"__n{result['n']}_seed{result['seed']}"
        )
        tag = f"__{result['protocol_tag']}" if result.get("protocol_tag") else ""
        disc = ""
        if result.get("query_shards"):
            disc += f"__shard{result['query_shard']}of{result['query_shards']}"
        if result.get("part"):
            disc += f"__part-{result['part']}"
        out_name = (
            f"{result['model_key']}__{result['instruction_key']}{tag}"
            f"{subset}{disc}.json"
        )
        out_path = args.out_dir / out_name
        writable = {k: v for k, v in result.items() if k != "per_query"}
        out_path.write_text(json.dumps(writable, indent=2))

        exact = result["self_checks"]["exact_match_exponent_1"]
        invariance = result["self_checks"]["linear_scale_invariance"]
        status = []
        if exact["checked"] > 0 and exact["max_abs_delta"] > args.tolerance:
            status.append(
                f"EXACT-MATCH FAILED (max|d|={exact['max_abs_delta']:.3e})"
            )
            failed_checks.append(str(run_dir))
        if invariance["max_abs_delta"] > args.tolerance:
            status.append(
                f"LINEAR-INVARIANCE FAILED (max|d|={invariance['max_abs_delta']:.3e})"
            )
            failed_checks.append(str(run_dir))
        mismatches = result["self_checks"]["ranking_argsort_mismatches"]
        if mismatches:
            status.append(f"{mismatches} tie-unstable rankings (informational)")

        print(
            f"[+] {result['model_key']} / {result['instruction_key']}"
            f"{subset} -> {out_path}"
            + (" | " + "; ".join(status) if status else " | checks OK")
        )

    groups: dict[tuple, list[dict]] = {}
    for result in all_results:
        groups.setdefault(
            (
                result["model_key"],
                result["instruction_key"],
                result.get("protocol_tag", ""),
                result.get("n"),
                result.get("seed"),
            ),
            [],
        ).append(result)

    pooled_results = [
        pool_results(members, domains_by_row) for members in groups.values()
    ]

    for pooled in pooled_results:
        if len(pooled["run_dirs"]) > 1:
            print(
                f"[~] Pooled {len(pooled['run_dirs'])} part/shard runs for "
                f"{pooled['model_key']} / {pooled['instruction_key']}"
                + (f" [{pooled['protocol_tag']}]" if pooled["protocol_tag"] else "")
            )

    if args.export_table is not None:
        table = {}
        for pooled in pooled_results:
            tag = f"__{pooled['protocol_tag']}" if pooled["protocol_tag"] else ""
            key = f"{pooled['model_key']}__{pooled['instruction_key']}{tag}"
            table[key] = {
                task: {
                    label: values["ndcg_at_k"]
                    for label, values in variants.items()
                }
                for task, variants in pooled["tasks"].items()
            }
        args.export_table.parent.mkdir(parents=True, exist_ok=True)
        args.export_table.write_text(json.dumps(table, indent=2))
        print(f"[+] Wrote summary table to {args.export_table}")

    if failed_checks:
        raise SystemExit(
            "Self-checks failed for: " + ", ".join(sorted(set(failed_checks)))
        )


if __name__ == "__main__":
    main()
