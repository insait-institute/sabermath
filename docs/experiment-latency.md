# Latency vs. quality (figure 2)

Plots statement-full nDCG@10 against measured per-query latency, log x, with a
Pareto frontier per model class.

## Prerequisites

- `python -m pip install -e .` (needs `matplotlib` and `seaborn`).
- Runs in `results/evaluation/` for the nDCG axis, read through
  `sabermath.tables.collect`/`build_rows`.
- Runs in `results/timing/` for the latency axis — see
  [experiment-timing.md](experiment-timing.md).

A model with no timing run falls back to `results/latency/data.csv`, and every
such use is printed when the figure is built. Running `run_timing.py` for
those models retires the fallback.

## Usage

```bash
python scripts/plots/plot_latency.py
```

Names, markers and colours come from `src/sabermath/figures.py`, shared with
the math-vs-word figure so both name a model identically.

## Output

```
results/latency/plots/figure2_latency.pdf
results/latency/plots/figure2_latency.svg
```

Frontier membership is computed, not stored: a model is on its class's
frontier when nothing in that class is simultaneously at least as accurate and
at least as fast. `sabermath.tables.MODEL_INFO` sets the class — `RERANK` on
the cross-encoder / late-interaction frontier, everything else on the
bi-encoder one. Only frontier models get a legend entry; the rest are drawn as
one grey cloud.

A `*` on a legend entry marks a model run in its own framework rather than on
vLLM; `STARRED` in `scripts/plots/plot_latency.py` lists them.
