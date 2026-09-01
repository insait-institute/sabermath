from pathlib import Path

RESULTS_DIR = Path("results/math_vs_word")
SIMILARITIES_DIR = RESULTS_DIR / "similarities"
PLOTS_DIR = RESULTS_DIR / "plots"

__all__ = ["RESULTS_DIR", "SIMILARITIES_DIR", "PLOTS_DIR"]
