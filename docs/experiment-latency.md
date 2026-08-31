# Latency vs. quality (figure 2)

A matplotlib rebuild of the repo-root `figure2-latency-mini.svg`: statement-full
nDCG@10 against measured per-query latency, log x, for all 42 models in the
paper's main results table.

```bash
python scripts/plots/plot_latency.py   # -> results/latency/figure2_latency.{pdf,svg}
```

It needs only matplotlib and seaborn on top of the package, so any of the
environments in `scripts/envs/` will do.

## What it shares with the math-vs-word figure

The two figures are meant to be read as a pair, so both take their **names,
markers and colours** from one place: `src/sabermath/figures.py`. Neither
figure carries its own copy, so a rename lands in both on the next run.

Two labels therefore differ from the mini SVG: `Diver-Retriever-4B` ->
`Diver-4B` and `Reason-ModernColBERT` -> `Reason-ColBERT`. Those are the paper
table's own shortenings.

A rewritten row suffixes its scorer's own label with `-Rewrite` rather than
prefixing the rewriter's name - matching the mini, and matching how
`results/rescaling/results.json` names them.

The axes are the size and weight of the **selected** (not `--all`)
math-vs-word figure: 15x8 at dpi 150, tick labels at 22/23, `s=120` markers
with no edges, the same 0.97 grey field and despined frame. Only the legend
departs from it - 15 entries in-plot would cover the frontier, so it sits
below the axes as on the `--all` figure: one box, three columns, each frontier
group headed by its own line in bold. The 18 entries divide 6/6/6, so a group
is allowed to continue into the next column; giving each group a column of its
own would set the box's height to the taller group's 10 rows, which is the
vertical space the three columns exist to save.

The frontier lines' own colours are not from the registry. They differ in hue,
dash and weight - slate solid against red long-dashed - so which line is which
reads without going to the legend. The red is deeper than the math-vs-word
figure's
`#e53935`, which is RaDeR-7B's marker colour: RaDeR-7B sits at (1.67 s, 0.690)
and the cross-encoder line runs along y=0.693 from 1.17 s out to 63.9 s, so the
two pass within about five pixels of each other and the marker needs to stay
readable there.

RaDeR-3B and Qwen3-Reranker-0.6B are 0.03 s and 0.002 nDCG apart, close enough
that whichever is drawn second hides the other. They sit on different
frontiers, so draw order would otherwise fall out of loop order; `MARKER_ZORDER`
in `scripts/plots/plot_latency.py` pins RaDeR-3B in front.

## Where the data comes from

`data.csv` has two columns of numbers, with different provenance:

- **nDCG** is read from `experiments/rescaling_robustness/results.json`, which
  already holds the authoritative statement-full nDCG@10 for exactly these 42
  models. Nothing is recovered from the picture.
- **Latency** exists nowhere else in this repo - `results/timing/` covers only
  7 of the 42 - so `figure2-latency-mini.svg` is its only source.
  `extract_from_svg.py` recovers it by inverting the figure's own axis
  transforms, rather than by re-typing numbers off a picture.

```
python extract_from_svg.py --verify     # rewrites data.csv, prints the check
```

`--verify` prints the pixel-derived nDCG beside the repository's own value for
every model. They agree to **0.004 or better across all 42**, which is what
establishes that the marker-to-model assignment is right.

That assignment is not made by nDCG alone: two pairs of models sit closer
together than the pixel error (Qwen3-Reranker-0.6B/SPLADE-Code-8B and
Jaccard/TF-IDF), and a nearest-value match swaps both. The 15 markers carrying
a legend entry are pinned first by their `(fill, shape)`, which the mini's
legend defines unambiguously; only the 27 anonymous grey circles are then
matched by rank against the rows that remain. `frontier` and `named` in the
CSV likewise come from the mini - the two step paths name their own members.

**If the real latency measurements turn up**, replace the `latency_s` column
and delete `scripts/analysis/extract_latency_data.py`; nothing else in
`latency_plot` reads the
SVG.

## The asterisk

`Jaccard*`, `BM25*` and `Reason-ColBERT*` carry a `*` in the mini's legend
whose meaning lives in the figure's caption, not in the SVG. It is reproduced
verbatim rather than dropped, since the caption refers to it.
