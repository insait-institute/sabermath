# Dataset Domain Pie Charts

These scripts generate sunburst/pie charts showing the distribution of mathematical domains and subdomains.

## Usage
Plot the benchmark queries dataset:

```bash
python plot_main_dataset.py < bechmark targets dataset >
```

Plot the candidate documents for a specific target query:

```bash
python plot_candidates.py --targets_dataset < benchmark targets dataset > --candidates_dataset < benchmark candidates dataset > --targ_idx < target problem index >
```

Plot an entire databank dataset:

```bash
python plot_databank.py --databank_dataset < dataset_name >
```

## Output
Each script saves a PDF pie chart showing:

inner ring: main mathematical domains
outer ring: subdomains/topics
For candidate plots, the chart is generated only for the candidate documents associated with the target index passed with --targ_idx.
