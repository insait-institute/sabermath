"""Math-vs-word: does a retriever key on the notation or on the prose?

Every module here reads and writes under one root, so a run started from the
repo root lands where `results/` says it should rather than next to the code.
"""

from pathlib import Path

RESULTS_DIR = Path("results/math_vs_word")
SIMILARITIES_DIR = RESULTS_DIR / "similarities"
# Runs made before the vendor input envelopes (pre-2026-08-26), kept for
# comparison only - see docs/experiment-math-vs-word.md.
BASELINE_DIR = RESULTS_DIR / "similarities_baseline"
PLOTS_DIR = RESULTS_DIR / "plots"

__all__ = ["RESULTS_DIR", "SIMILARITIES_DIR", "BASELINE_DIR", "PLOTS_DIR"]
