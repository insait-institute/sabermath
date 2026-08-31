<!--
Hand-generated on 2026-08-28 from the JSONs written by

    python3 python -m sabermath.analysis.compute_confidence_intervals \
        results/evaluation/{rank1-0.5b,rank1-7b,rank1-32b,reasonir-8b}__p0.json \
        --out-dir results/confidence

Not produced by scripts/report_experiments.py - it is a standalone view of four
models, and does not feed or supersede tables/RESULTS_*.md.

Inputs are the p0 arm of results/evaluation, i.e. the current
vendor-envelope input protocol with no instruction, which is the production
configuration. Bootstrap: 10,000 resamples, seed 42411, 300 resampled queries
per domain, full-size resample for the overall estimate, 95% percentile
intervals - identical to experiments/confidence_intervals/confidence.py, so
these are directly comparable with the paper's Table 3.
-->

# SABER-Math - Rank1 family and ReasonIR-8B, bootstrap confidence intervals

nDCG@10, 95% percentile bootstrap over 1000 queries. Cells are
bootstrap-mean [2.5th, 97.5th percentile].

## All three settings

| Model | Setting | Overall | Algebra | Geometry | Number Theory | Combinatorics | Calculus |
|---|---|---|---|---|---|---|---|
| Rank1-0.5B | statement-statement | **0.3713** [0.3642, 0.3785] | 0.377 [0.364, 0.390] | 0.384 [0.370, 0.397] | 0.342 [0.330, 0.356] | 0.375 [0.361, 0.390] | 0.376 [0.365, 0.387] |
| Rank1-0.5B | statement-full | **0.3651** [0.3582, 0.3719] | 0.376 [0.364, 0.388] | 0.390 [0.378, 0.403] | 0.340 [0.329, 0.352] | 0.353 [0.340, 0.367] | 0.374 [0.362, 0.386] |
| Rank1-0.5B | full-full | **0.3633** [0.3566, 0.3701] | 0.355 [0.344, 0.366] | 0.402 [0.388, 0.416] | 0.336 [0.324, 0.348] | 0.353 [0.341, 0.366] | 0.371 [0.360, 0.382] |
| Rank1-7B | statement-statement | **0.5064** [0.4972, 0.5157] | 0.483 [0.465, 0.500] | 0.507 [0.493, 0.522] | 0.491 [0.475, 0.509] | 0.569 [0.551, 0.586] | 0.483 [0.467, 0.499] |
| Rank1-7B | statement-full | **0.5514** [0.5413, 0.5614] | 0.517 [0.498, 0.536] | 0.575 [0.558, 0.591] | 0.548 [0.529, 0.567] | 0.615 [0.597, 0.633] | 0.512 [0.494, 0.531] |
| Rank1-7B | full-full [^partial] | **0.4668** [0.4563, 0.4772] | 0.444 [0.426, 0.462] | 0.499 [0.481, 0.517] | 0.456 [0.436, 0.475] | 0.531 [0.512, 0.551] | 0.410 [0.394, 0.427] |
| Rank1-32B | statement-statement | **0.5557** [0.5459, 0.5654] | 0.539 [0.521, 0.558] | 0.541 [0.525, 0.558] | 0.549 [0.530, 0.567] | 0.619 [0.601, 0.637] | 0.532 [0.514, 0.550] |
| Rank1-32B | statement-full | **0.6100** [0.5999, 0.6199] | 0.583 [0.564, 0.602] | 0.617 [0.599, 0.635] | 0.613 [0.595, 0.631] | 0.673 [0.655, 0.690] | 0.574 [0.556, 0.592] |
| Rank1-32B | full-full | **0.6288** [0.6185, 0.6390] | 0.594 [0.574, 0.614] | 0.645 [0.628, 0.662] | 0.624 [0.604, 0.643] | 0.682 [0.664, 0.700] | 0.605 [0.587, 0.623] |
| ReasonIR-8B | statement-statement | **0.5127** [0.5041, 0.5214] | 0.504 [0.488, 0.521] | 0.529 [0.514, 0.545] | 0.503 [0.488, 0.519] | 0.539 [0.522, 0.555] | 0.482 [0.468, 0.498] |
| ReasonIR-8B | statement-full | **0.5074** [0.4985, 0.5161] | 0.495 [0.479, 0.510] | 0.511 [0.495, 0.527] | 0.500 [0.484, 0.517] | 0.534 [0.517, 0.550] | 0.489 [0.475, 0.503] |
| ReasonIR-8B | full-full | **0.6169** [0.6085, 0.6253] | 0.584 [0.569, 0.599] | 0.635 [0.620, 0.649] | 0.618 [0.602, 0.634] | 0.629 [0.613, 0.644] | 0.613 [0.598, 0.628] |

[^partial]: Rank1-7B full-full is bootstrapped over 970 of 1000 queries - 30 are still unscored in `results/evaluation/rank1-7b__p0.json`. Every other cell in this file covers all 1000.

## Difference from the previous entries

### Rank1-32B and ReasonIR-8B - already present, exactly reproduced

These two already had `confresult_v2/` entries (written 2026-08-26 and 2026-08-27). Re-running the script over the same p0 inputs reproduced them:

| Model | Setting | Previous overall | New overall | Delta (mean) | Delta (CI bounds) |
|---|---|---|---|---|---|
| Rank1-32B | statement-statement | 0.5557 [0.5459, 0.5654] | 0.5557 [0.5459, 0.5654] | +0.0e+00 | +0.0e+00 / +0.0e+00 |
| Rank1-32B | statement-full | 0.6100 [0.5999, 0.6199] | 0.6100 [0.5999, 0.6199] | +0.0e+00 | +0.0e+00 / +0.0e+00 |
| Rank1-32B | full-full | 0.6288 [0.6185, 0.6390] | 0.6288 [0.6185, 0.6390] | +0.0e+00 | +0.0e+00 / +0.0e+00 |
| ReasonIR-8B | statement-statement | 0.5127 [0.5041, 0.5214] | 0.5127 [0.5041, 0.5214] | +0.0e+00 | +0.0e+00 / +0.0e+00 |
| ReasonIR-8B | statement-full | 0.5074 [0.4985, 0.5161] | 0.5074 [0.4985, 0.5161] | +0.0e+00 | +0.0e+00 / +0.0e+00 |
| ReasonIR-8B | full-full | 0.6169 [0.6085, 0.6253] | 0.6169 [0.6085, 0.6253] | +0.0e+00 | +0.0e+00 / +0.0e+00 |

Every interval bound is bit-identical; the means differ by at most 2.2e-16, which is float summation order, not a change in the estimate. Same for all 30 per-domain branch intervals.

### Rank1-0.5B and Rank1-7B - new under this protocol

Neither had a `confresult_v2/` entry. Their only previous interval was in `confresult/`, bootstrapped from `results/evaluation/<model>.json` - the older input protocol, before vendor envelopes. That is what `tables/RESULTS_*.md` still shows for these two rows (`Src = repo`).

| Model | Setting | Previous (legacy protocol) | New (v2 p0) | Delta | Max per-domain delta |
|---|---|---|---|---|---|
| Rank1-0.5B | statement-statement | 0.3704 [0.3634, 0.3775] | 0.3713 [0.3642, 0.3785] | +0.0009 | 0.0061 |
| Rank1-0.5B | statement-full | 0.3639 [0.3571, 0.3707] | 0.3651 [0.3582, 0.3719] | +0.0012 | 0.0052 |
| Rank1-0.5B | full-full | 0.3632 [0.3566, 0.3699] | 0.3633 [0.3566, 0.3701] | +0.0001 | 0.0033 |
| Rank1-7B | statement-statement | 0.5084 [0.4991, 0.5177] | 0.5064 [0.4972, 0.5157] | -0.0020 | 0.0044 |
| Rank1-7B | statement-full | 0.5521 [0.5421, 0.5621] | 0.5514 [0.5413, 0.5614] | -0.0007 | 0.0018 |
| Rank1-7B | full-full [^partial] | 0.4636 [0.4532, 0.4739] | 0.4668 [0.4563, 0.4772] | +0.0032 | 0.0110 |

These are genuinely different runs, not a re-bootstrap, so the deltas are real score movement. All six are small: the largest overall shift is +0.0032 (Rank1-7B full-full), against a CI half-width of ~0.010, and every new interval overlaps its predecessor heavily. The protocol change is effectively null for the Rank1 family - expected, since Rank1 has no instruction slot and the vendor route rewrites the query rather than wrapping it in an envelope.

Note that Rank1-7B's full-full delta is partly an artefact of coverage: the new run is missing 30 queries, so its 0.4668 is over a different query set than the legacy 0.4636 over all 1000.

### Against the paper

The paper reports Rank1-32B and ReasonIR-8B (Tables 1/3/4); it has no row for Rank1-0.5B or Rank1-7B.

| Model | Setting | Paper | New | Delta |
|---|---|---|---|---|
| Rank1-32B | statement-statement | 0.556 | 0.5557 [0.5459, 0.5654] | -0.0003 |
| Rank1-32B | statement-full | 0.610 [0.6000, 0.6200] | 0.6100 [0.5999, 0.6199] | +0.0000 |
| Rank1-32B | full-full | 0.629 | 0.6288 [0.6185, 0.6390] | -0.0002 |
| ReasonIR-8B | statement-statement | 0.513 | 0.5127 [0.5041, 0.5214] | -0.0003 |
| ReasonIR-8B | statement-full | 0.507 [0.4980, 0.5160] | 0.5074 [0.4985, 0.5161] | +0.0004 |
| ReasonIR-8B | full-full | 0.617 | 0.6169 [0.6085, 0.6253] | -0.0001 |

Both reproduce the paper to within 0.0005 on every setting, and every paper point estimate falls inside the recomputed interval. For Rank1-32B that is expected - its inputs are unchanged under the new protocol. ReasonIR-8B's are not: it is one of the eight models in `SUPERSEDED_BY_V2`, and its p0 run on the older protocol scored 0.5137 / 0.5092 / 0.6177 (statement-statement / statement-full / full-full) against 0.5127 / 0.5074 / 0.6169 here. The envelope costs it 0.0007-0.0019, well inside the interval, and the paper's rounded numbers are unchanged either way.
