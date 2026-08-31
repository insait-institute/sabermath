# SABER-Math documentation index

Everything in this repo is run from four endpoints in `scripts/`. There is no
separate `experiments/` tree: analysis code lives in the package next to the
source it depends on, and every result lives under one `results/` root.

## The endpoints

| Endpoint | What it produces | Written to |
|---|---|---|
| `scripts/run_experiments.py` | nDCG evaluation — the main tables and the instruction arms | `results/evaluation/` |
| `scripts/run_dedup.py` | Where a rephrased copy of a query's own problem ranks | `results/dedup/` |
| `scripts/run_timing.py` | Per-query latency on production backends | `results/timing/` |
| `scripts/report_experiments.py` | Every table, regenerated from the above | `results/tables/` |

Each takes `--models`, defaulting to every model in the registry, and each
`--help` is the authoritative reference for its own flags. There are no
launchers and no job files: run an endpoint directly, in the right
environment.

## Per-experiment write-ups

| Document | Experiment |
|---|---|
| [experiment-evaluation.md](experiment-evaluation.md) | Running an nDCG sweep: environments, sharding, resuming, output naming |
| [experiment-instructions.md](experiment-instructions.md) | Does a task instruction help? The four prompt arms |
| [experiment-dedup.md](experiment-dedup.md) | Where a rephrased copy of a query's own problem ranks |
| [experiment-rescaling.md](experiment-rescaling.md) | Does the ranking survive a different gain or relevance scale? |
| [experiment-latency.md](experiment-latency.md) | Latency vs. quality (figure 2) |
| [experiment-confidence-intervals.md](experiment-confidence-intervals.md) | Bootstrap confidence intervals |
| [experiment-math-vs-word.md](experiment-math-vs-word.md) | Do retrievers key on notation or on prose? |
| [experiment-mteb.md](experiment-mteb.md) | Correlation with general-purpose MTEB retrieval |
| [experiment-benchmark-analysis.md](experiment-benchmark-analysis.md) | Benchmark and source-corpus composition |
| [experiment-additional.md](experiment-additional.md) | Construction-pipeline checks: tournament convergence, selection signals |

## How models are configured

| Document | What it settles |
|---|---|
| [protocol.md](protocol.md) | What each model actually receives: input envelopes, instruction templates and placement |
| [backend-provenance.md](backend-provenance.md) | Why each model is served on the backend it is, and what that was checked against |

## Where results live

Every result is under one `results/` root, one directory per experiment — the
table in the [repository README](../README.md#results) lists them.

Every evaluation run records the protocol that produced it, so which run backs
a given table row is decided by the run's own metadata rather than by which
directory it sits in.
