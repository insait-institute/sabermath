# SABER-Math documentation index

## Prerequisites for everything

```bash
python -m pip install -e .
```


## Endpoints

| Script | Produces | Written to |
|---|---|---|
| `scripts/run_experiments.py` | nDCG evaluation: main tables and the instructions | `results/evaluation/` |
| `scripts/run_dedup.py` | Where a rephrased copy of a query's own problem ranks | `results/dedup/` |
| `scripts/run_timing.py` | Per-query latency on production backends | `results/timing/` |
| `scripts/report_experiments.py` | Every table | `results/tables/` |

## Per-experiment pages

| Document | Experiment |
|---|---|
| [experiment-evaluation.md](experiment-evaluation.md) | nDCG evaluation: environments, sharding, resuming, output naming |
| [experiment-instructions.md](experiment-instructions.md) | The four instruction prompts |
| [experiment-dedup.md](experiment-dedup.md) | Where a rephrased copy of a query's own problem ranks |
| [experiment-rescaling.md](experiment-rescaling.md) | Replaying rankings under a different gain or relevance scale |
| [experiment-confidence-intervals.md](experiment-confidence-intervals.md) | Bootstrap confidence intervals |
| [experiment-timing.md](experiment-timing.md) | Per-query latency |
| [experiment-latency.md](experiment-latency.md) | Latency vs. quality (figure 2) |
| [experiment-math-vs-word.md](experiment-math-vs-word.md) | Equations vs. prose: figure and table |
| [experiment-mteb.md](experiment-mteb.md) | Correlation with MTEB Retrieval |
| [experiment-benchmark-analysis.md](experiment-benchmark-analysis.md) | Dataset composition charts |
| [experiment-additional.md](experiment-additional.md) | Construction-pipeline checks |

## Results layout

| Path | Contents |
|---|---|
| `results/evaluation/` | Every nDCG run, `<model>__<prompt>.json` |
| `results/timing/` | Per-query latency |
| `results/dedup/` | Deduplication rankings |
| `results/confidence/` | Bootstrap confidence intervals |
| `results/rescaled/`, `results/rescaling/` | Relevance-rescaling robustness |
| `results/math_vs_word/` | Similarity dumps and figures |
| `results/latency/` | Latency-vs-quality figure and its data |
| `results/diagnostics/` | Archived backend-equivalence and protocol verdicts |
| `results/tables/` | Generated tables |
