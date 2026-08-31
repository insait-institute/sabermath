from pathlib import Path

RESULTS_DIR = Path("results/math_vs_word")
SIMILARITIES_DIR = RESULTS_DIR / "similarities"
# Runs predating the vendor input envelopes, kept for comparison only.
BASELINE_DIR = RESULTS_DIR / "similarities_baseline"
PLOTS_DIR = RESULTS_DIR / "plots"

__all__ = ["RESULTS_DIR", "SIMILARITIES_DIR", "BASELINE_DIR", "PLOTS_DIR"]
