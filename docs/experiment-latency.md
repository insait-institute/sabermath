# Latency vs. quality (figure 2)

Plots statement-full nDCG@10 against measured per-query latency, with a Pareto frontier per model class.

## Prerequisites

- `python -m pip install -e ".[analysis]"` — for `matplotlib` and `seaborn`.
- Runs in `results/evaluation/` for the nDCG axis, read through `sabermath.tables.collect`/`build_rows`.
- Runs in `results/timing/` for the latency axis — see [experiment-timing.md](experiment-timing.md).

A model with no timing run falls back to `results/latency/data.csv`, and every such use is printed when the figure is built.

## Usage

```bash
python scripts/plots/plot_latency.py
```

Names, markers and colours come from `src/sabermath/figures.py`, shared with the math-vs-word figure so both name a model identically.

## Output

```
results/latency/plots/figure2_latency.pdf
results/latency/plots/figure2_latency.svg
```
