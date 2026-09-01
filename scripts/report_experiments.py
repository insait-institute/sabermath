#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys

from sabermath.results import load_runs


USAGE = """\
  python scripts/report_experiments.py                 # everything
  python scripts/report_experiments.py main timing     # just these
  python scripts/report_experiments.py --list
  python scripts/report_experiments.py --out-dir /tmp/tables
  python scripts/report_experiments.py mteb --mteb-file leaderboard.csv
"""

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path("results")
DEFAULT_OUT = RESULTS / "tables"

DEFAULT_MTEB_CSV = RESULTS / "mteb" / "leaderboard.csv"
MTEB_CSV = DEFAULT_MTEB_CSV


def _run(label: str, script: str, argv: list[str], optional: bool = False) -> bool:
    print(f"\n{'=' * 70}\n== {label}\n{'=' * 70}")
    completed = subprocess.run(
        [sys.executable, str(ROOT / script), *argv],
        cwd=ROOT,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode == 0:
        return True
    reason = ""
    for line in reversed((completed.stderr or "").strip().split("\n")):
        if line.strip():
            reason = line.strip()
            break
    note = "skipped" if optional else "FAILED"
    print(f"[~] {label}: {note} (exit {completed.returncode})")
    if reason:
        print(f"    {reason}")
    return False


def report_confidence(out_dir: Path) -> bool:
    runs = load_runs(RESULTS / "evaluation", prompts=["p0"])
    p0 = sorted(
        {Path(payload["_path"]) for payload in runs.values() if "_path" in payload}
    )
    if not p0:
        print("[~] no p0 runs found - skipping CI recompute")
        return True
    ok = True
    for path in p0:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/analysis/compute_confidence_intervals.py"),
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
        "scripts/tables/main_tables.py",
        ["--out-dir", str(out_dir)],
    )


def report_instructions(out_dir: Path) -> bool:
    return _run(
        "instruction ablation",
        "scripts/tables/instruction_tables.py",
        ["--out", str(out_dir / "RESULTS_instructions.md")],
        optional=True,
    )


def report_instructions_statement_full(out_dir: Path) -> bool:
    return _run(
        "instruction ablation (statement-full, with pooled shards)",
        "scripts/tables/instruction_statement_full.py",
        ["--out-dir", str(out_dir)],
        optional=True,
    )


def report_timing(out_dir: Path) -> bool:
    return _run(
        "per-query latency",
        "scripts/tables/timing_tables.py",
        ["--out", str(out_dir / "RESULTS_timing.md")],
    )


def report_dedup(out_dir: Path) -> bool:
    return _run(
        "deduplication",
        "scripts/tables/dedup_tables.py",
        ["--out", str(out_dir / "RESULTS_dedup.md")],
        optional=True,
    )


def report_rescaling(out_dir: Path) -> bool:
    return _run(
        "relevance-rescaling robustness",
        "scripts/analysis/rescaling.py",
        [
            "--from-rankings",
            str(RESULTS / "rescaling" / "rankings.npz"),
            "--latex",
            str(out_dir / "rescaling_table.tex"),
        ],
        optional=True,
    )


def report_math_vs_word(out_dir: Path) -> bool:
    return _run(
        "math-vs-word table (equations vs prose, per instruction)",
        "scripts/tables/math_vs_word_table.py",
        ["--out-md", str(out_dir / "RESULTS_math_vs_word.md"),
         "--out-tex", str(out_dir / "math_vs_word_instructions.tex")],
        optional=True,
    )


def report_mteb(out_dir: Path) -> bool:
    if not MTEB_CSV.exists():
        print(f"[~] MTEB rank correlation: no leaderboard CSV at {MTEB_CSV} "
              "- skipped (see --mteb-file)")
        return True
    return _run(
        "MTEB rank correlation",
        "scripts/analysis/mteb_correlation.py",
        ["--mteb-file", str(MTEB_CSV)],
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
    "math-vs-word": report_math_vs_word,
    "mteb": report_mteb,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        epilog=USAGE, formatter_class=argparse.RawDescriptionHelpFormatter
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
