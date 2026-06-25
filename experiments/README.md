# SABER-Math experiments

This directory contains analysis scripts for the SABER-Math benchmark. The experiments support the paper's evaluation of mathematical information retrieval systems, including benchmark composition, retriever performance analysis, confidence intervals, retrieval-signal analysis, and comparisons against general-purpose retrieval benchmarks.

## Directory overview

| Path | What it does |
|---|---|
| `bechmark_analysis/` | Analyzes the composition of the benchmark and source corpus, including domain and subdomain distributions. |
| `additional_experiments/` | Contains extra analyses for the benchmark-construction pipeline, especially tournament-based relevance estimation and the effect of different candidate-selection signals. See the file-level notes below. |
| `math-vs-word/` | Compares how much retrievers rely on mathematical notation versus surrounding natural-language text. |
| `confidence_intervals/` | Computes bootstrap confidence intervals for retrieval results and formats them for inclusion in the paper. |
| `mteb_comparison/` | Compares SABER-Math performance with general-purpose retrieval benchmark performance using rank and score correlations. |

## Additional experiments

The `additional_experiments/` directory contains smaller analysis scripts tied to specific benchmark-construction checks:

| File | What it does |
|---|---|
| `avg_inversions.py` | Measures how close a reduced Swiss-style tournament ranking is to a ranking produced from many more pairwise comparisons. |
| `plot_inversions.py` | Plots the tournament-round ablation based on the inversion statistics. |
| `signal_effect.py` | Analyzes how candidates selected by topic similarity, solution-summary similarity, or both signals differ in their final relevance rankings. |
| `config.yaml` | Configuration for the additional tournament and signal-analysis experiments. |
| `environment.yml` | Conda environment for running these scripts. |

## Notes

These scripts assume that the required SABER-Math data artifacts, cached model outputs, or intermediate JSON files are available at the paths specified in the corresponding configuration files or script arguments.