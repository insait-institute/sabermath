# Dataset Domain Pie Charts

These generate sunburst/pie charts showing the distribution of mathematical
domains and subdomains. They live in `src/sabermath/`.

## Usage
Plot the benchmark queries dataset:

```bash
python scripts/plots/plot_dataset.py benchmark <benchmark targets dataset>
```

Plot the candidate documents for a specific target query:

```bash
python scripts/plots/plot_candidates.py \
    --targets-dataset <benchmark targets dataset> \
    --candidates-dataset <benchmark candidates dataset> \
    --targ-idx <target problem index>
```

Plot an entire databank dataset:

```bash
python scripts/plots/plot_dataset.py databank --databank-dataset <dataset_name>
```

## Output
Each script saves a PDF pie chart showing:

inner ring: main mathematical domains
outer ring: subdomains/topics
For candidate plots, the chart is generated only for the candidate documents associated with the target index passed with --targ-idx.
