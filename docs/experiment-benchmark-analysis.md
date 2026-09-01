# Dataset composition charts

Sunburst charts of how the benchmark and its source corpus distribute across mathematical domains and subdomains.

## Prerequisites

- `python -m pip install -e .` (needs `matplotlib`).
- The Hugging Face dataset name, passed on the command line. The databank arm expects `numeric` and `proofs` splits and derives each problem's domains from its ontology tags; the benchmark arm reads the domains the dataset already carries.

## Usage

```bash
# the benchmark queries
python scripts/plots/plot_dataset.py benchmark <hf-dataset>

# the full source databank
python scripts/plots/plot_dataset.py databank <hf-dataset>

# the candidates of one target query
python scripts/plots/plot_candidates.py \
    --targets-dataset <hf-dataset> \
    --candidates-dataset <hf-dataset> \
    --targ-idx <index>
```

`plot_dataset.py` takes `--out-file` to override the destination; `plot_candidates.py` takes `--top-k` (default 10) for how many of the most relevant candidates to mark.

## Output

| Command | Default file |
|---|---|
| `plot_dataset.py benchmark` | `piechart_benchmark.pdf` |
| `plot_dataset.py databank` | `piechart_databank.pdf` |
| `plot_candidates.py` | `piechart_candidates.pdf`, for the given target index |
