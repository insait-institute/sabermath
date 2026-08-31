#!/usr/bin/env python3
"""Regenerate every SABER-Math table. THE endpoint for reporting.

    python scripts/report_experiments.py                 # everything
    python scripts/report_experiments.py main timing     # just these
    python scripts/report_experiments.py --list
    python scripts/report_experiments.py --out-dir /tmp/tables
    python scripts/report_experiments.py mteb --mteb-file leaderboard.csv

Safe to run at any point. Each generator reports how many rows it could fill
and where each row came from, so a partially-finished sweep produces a partial
table rather than a wrong one, and a generator whose inputs are missing is
skipped with a note instead of failing the run.

Every table reads from results/ and writes into results/tables/.

PROTOCOL PRECEDENCE is not a directory convention any more. All evaluation
runs live in one folder and each records the protocol that produced it, so
which run backs a row is decided by the run itself (see
sabermath.results.run_rank). Copying a result file somewhere cannot change a
published number.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS = Path("results")
DEFAULT_OUT = RESULTS / "tables"

# The MTEB leaderboard export is not redistributable, so it is not in the
# repo; report_mteb skips cleanly when it is absent. Overridden by --mteb-file.
DEFAULT_MTEB_CSV = RESULTS / "mteb" / "leaderboard.csv"
MTEB_CSV = DEFAULT_MTEB_CSV


def _run(label: str, argv: list[str], optional: bool = False) -> bool:
    """Run one generator as a subprocess so one failure cannot abort the rest."""
    print(f"\n{'=' * 70}\n== {label}\n{'=' * 70}")
    completed = subprocess.run([sys.executable, *argv], cwd=ROOT)
    if completed.returncode == 0:
        return True
    note = "inputs not ready yet - skipped" if optional else "FAILED"
    print(f"[~] {label}: {note} (exit {completed.returncode})")
    return False


def _module(name: str) -> list[str]:
    return ["-m", f"sabermath.{name}"]


def report_confidence(out_dir: Path) -> bool:
    """Recompute the bootstrap CIs the main tables read. Must run first."""
    p0 = sorted((RESULTS / "evaluation").glob("*__p0.json"))
    if not p0:
        print("[~] no p0 runs found - skipping CI recompute")
        return True
    ok = True
    for path in p0:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "sabermath.analysis.compute_confidence_intervals",
                str(path),
                "--out-dir",
                str(RESULTS / "confidence"),
            ],
            cwd=ROOT,
            capture_output=True,
        )
        ok &= completed.returncode == 0
    print(f"[+] recomputed CIs for {len(p0)} p0 runs" if ok else "[~] some CIs failed")
    return ok


def report_main(out_dir: Path) -> bool:
    return _run(
        "main tables (statement-full, statement-statement, full-full)",
        _module("reporting.main_tables") + ["--out-dir", str(out_dir)],
    )


def report_instructions(out_dir: Path) -> bool:
    return _run(
        "instruction ablation",
        _module("reporting.instruction_tables")
        + ["--out", str(out_dir / "RESULTS_instructions.md")],
        optional=True,
    )


def report_instructions_statement_full(out_dir: Path) -> bool:
    return _run(
        "instruction ablation (statement-full, with pooled shards)",
        _module("reporting.instruction_statement_full")
        + ["--out-dir", str(out_dir)],
        optional=True,
    )


def report_timing(out_dir: Path) -> bool:
    return _run(
        "per-query latency",
        _module("reporting.timing_tables")
        + ["--out", str(out_dir / "RESULTS_timing.md")],
    )


def report_dedup(out_dir: Path) -> bool:
    return _run(
        "deduplication",
        _module("reporting.dedup_tables")
        + ["--out", str(out_dir / "RESULTS_dedup.md")],
        optional=True,
    )


def report_rescaling(out_dir: Path) -> bool:
    return _run(
        "relevance-rescaling robustness",
        _module("analysis.rescaling")
        + [
            "--from-rankings",
            str(RESULTS / "rescaling" / "rankings.npz"),
            "--latex",
            str(out_dir / "rescaling_table.tex"),
        ],
        optional=True,
    )


def report_mteb(out_dir: Path) -> bool:
    """Correlate SABER-Math against MTEB Retrieval.

    Needs a leaderboard CSV (Model, Retrieval columns), which is not
    redistributable and so is not in this repo. Drop an export at
    DEFAULT_MTEB_CSV, or pass --mteb-file. The benchmark side is read live
    from results/, so this can never correlate against stale numbers.
    """
    if not MTEB_CSV.exists():
        print(f"[~] MTEB rank correlation: no leaderboard CSV at {MTEB_CSV} "
              "- skipped (see --mteb-file)")
        return True
    return _run(
        "MTEB rank correlation",
        _module("analysis.mteb") + ["--mteb_file", str(MTEB_CSV)],
        optional=True,
    )


REPORTS = {
    "confidence": report_confidence,
    "main": report_main,
    "instructions": report_instructions,
    "instructions-statement-full": report_instructions_statement_full,
    "timing": report_timing,
    "dedup": report_dedup,
    "rescaling": report_rescaling,
    "mteb": report_mteb,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "reports",
        nargs="*",
        metavar="REPORT",
        help=f"Which tables to build (default: all). One or more of: "
        f"{', '.join(REPORTS)}",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--mteb-file",
        type=Path,
        default=DEFAULT_MTEB_CSV,
        help="MTEB leaderboard CSV for the correlation report (default: "
        f"{DEFAULT_MTEB_CSV}, skipped if absent).",
    )
    parser.add_argument("--list", action="store_true", help="List reports and exit.")
    args = parser.parse_args()

    global MTEB_CSV
    MTEB_CSV = args.mteb_file

    if args.list:
        print("Available reports (in run order):")
        for name in REPORTS:
            print(f"  {name}")
        return

    unknown = [r for r in args.reports if r not in REPORTS]
    if unknown:
        parser.error(
            f"Unknown report(s): {', '.join(unknown)}. "
            f"Available: {', '.join(REPORTS)}"
        )

    # Preserve declaration order even when the user lists them out of order:
    # 'confidence' writes the intervals 'main' reads.
    selected = [r for r in REPORTS if not args.reports or r in args.reports]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in selected:
        results[name] = REPORTS[name](args.out_dir)

    print(f"\n{'=' * 70}")
    ok = [n for n, v in results.items() if v]
    bad = [n for n, v in results.items() if not v]
    print(f"Built {len(ok)}/{len(selected)} reports into {args.out_dir}")
    if bad:
        print(f"Not built: {', '.join(bad)}")


if __name__ == "__main__":
    main()
