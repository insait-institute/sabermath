"""Compute Table-3-style bootstrap confidence intervals for reranker-sweep
results, straight from the per-query nDCGs that scripts/run_rerankers.py
already saves (the "ndcgs_by_task" field in results/rerankers/<model>.json).

This replaces the `testing/compute_confidence_intervals.py` step the README
used to point at (that script lives only on the canonical host's checkout,
which isn't readable from every account). Unlike
experiments/confidence_intervals/confidence.py, it does NOT re-run any model
or need any vector cache - the per-query nDCGs are already on disk.

The bootstrap protocol matches confidence.py exactly, so the new models'
intervals are directly comparable with the paper's existing Table 3 numbers:
  - 10,000 bootstrap samples, fixed seed 42411, fresh RNG per model+task;
  - 300 resampled queries per domain for the domain-level estimates;
  - a full-size resample (n = total query count) for the overall estimate;
  - 95% percentile intervals (2.5th / 97.5th percentiles);
  - queries whose nDCG is still null (unfinished checkpoint) are skipped,
    same as confidence.py's valid_idxs handling - a loud warning is printed
    when that happens so a partial run is never mistaken for a finished one.

Output: one JSON per model, in the exact shape confidence.py writes
({"k": 10, "tasks": [{"task", "mean", "confidence_interval", "branches"}]}),
into experiments/confidence_intervals/confresult/ by default - so
experiments/confidence_intervals/json_to_tex.py formats old and new models
into the same LaTeX tables with no extra glue.

Usage (from the sabermath repo root, after the sweep jobs finish):
    python scripts/rerankers/compute_confidence_intervals.py results/rerankers/*.json
    # then, to regenerate the LaTeX tables:
    #   cd experiments/confidence_intervals && python json_to_tex.py

Remember to exclude smoke-test files (results/rerankers/*__n20_seed42.json
etc.) from the glob - their checkpoints cover a 20-query subset, not the
benchmark.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

DOMAINS = [
    "Algebra",
    "Geometry",
    "Number Theory",
    "Combinatorics",
    "Calculus and Analysis",
]
DOMAIN_SAMPLE_COUNT = 300
X = 10000
SEED = 42411


def confidence_interval_95(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def get_task_confidence(task_ndcgs, domain_idxs, all_idxs, task, x=X):
    """Same estimator (and the same seeded draw sequence) as
    experiments/confidence_intervals/confidence.py's get_task_confidence."""
    rng = random.Random(SEED)

    valid = {
        domain: [i for i in idxs if task_ndcgs[i] is not None]
        for domain, idxs in domain_idxs.items()
    }
    valid_all = [i for i in all_idxs if task_ndcgs[i] is not None]

    for domain in DOMAINS:
        if not valid[domain]:
            raise ValueError(f"No valid nDCG values for task={task}, domain={domain}")
    if not valid_all:
        raise ValueError(f"No valid nDCG values for task={task}")

    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(x):
        # Draw order matches confidence.py: each domain (in DOMAINS order),
        # then the full-size overall resample.
        for domain in DOMAINS:
            idxs = [rng.choice(valid[domain]) for _ in range(DOMAIN_SAMPLE_COUNT)]
            samples[domain].append(float(np.mean([task_ndcgs[i] for i in idxs])))
        idxs = [rng.choice(valid_all) for _ in range(len(task_ndcgs))]
        samples["total"].append(float(np.mean([task_ndcgs[i] for i in idxs])))

    branch_dict = [
        {
            "branch": domain,
            "confidence_interval": confidence_interval_95(samples[domain]),
            "mean": float(np.mean(samples[domain])),
        }
        for domain in DOMAINS
    ]
    return {
        "task": task,
        "confidence_interval": confidence_interval_95(samples["total"]),
        "mean": float(np.mean(samples["total"])),
        "branches": branch_dict,
    }


def process_file(path: Path, out_dir: Path) -> None:
    data = json.loads(path.read_text())
    domains = data.get("domains")
    reports = data.get("reports", {})

    for model_key, report in reports.items():
        if "error" in report and "ndcgs_by_task" not in report:
            print(f"[!!] {path.name}: {model_key} only has an error record - skipped.")
            continue
        ndcgs_by_task = report.get("ndcgs_by_task")
        if not ndcgs_by_task or not domains:
            print(f"[!!] {path.name}: {model_key} has no ndcgs_by_task/domains - skipped.")
            continue

        all_idxs = list(range(len(domains)))
        domain_idxs = {
            domain: [i for i, ds in enumerate(domains) if domain in ds]
            for domain in DOMAINS
        }

        tasks_out = []
        for task, task_ndcgs in ndcgs_by_task.items():
            n_missing = sum(1 for v in task_ndcgs if v is None)
            if n_missing:
                print(
                    f"[!!] {model_key} / {task}: {n_missing}/{len(task_ndcgs)} "
                    f"queries have no nDCG yet (partial run?) - bootstrapping "
                    f"over the remaining {len(task_ndcgs) - n_missing} only."
                )
            tasks_out.append(
                get_task_confidence(task_ndcgs, domain_idxs, all_idxs, task)
            )

        result = {"k": report.get("k", 10), "tasks": tasks_out}
        out_path = out_dir / f"{model_key}.json"
        out_path.write_text(json.dumps(result, indent=2))

        print(f"\n=== {model_key} -> {out_path} ===")
        for t in tasks_out:
            lo, hi = t["confidence_interval"]
            print(
                f"  {t['task']:<20} mean nDCG@10 = {t['mean']:.4f} "
                f"[{lo:.4f}, {hi:.4f}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="results/rerankers/<model>.json files (exclude __n*_seed* smoke tests)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/confidence_intervals/confresult"),
        help="Where to write the per-model CI JSONs (default: the same "
        "confresult/ directory confidence.py writes to, so json_to_tex.py "
        "sees every model at once)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.results:
        if "__n" in path.stem:
            print(f"[~] Skipping smoke-test file {path.name}.")
            continue
        process_file(path, args.out_dir)


if __name__ == "__main__":
    main()
