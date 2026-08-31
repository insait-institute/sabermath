import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import load_dataset
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", required=True)

group = parser.add_mutually_exclusive_group(required=False)
group.add_argument("--algebra", action="store_true")
group.add_argument("--geometry", action="store_true")
group.add_argument("--combinatorics", action="store_true")
group.add_argument("--calculus", action="store_true")
group.add_argument("--number_theory", action="store_true")

args = parser.parse_args()

with open(args.config_file, "r") as f:
    config = yaml.safe_load(f)

targets = load_dataset(config["hf_datasets"]["original_targets_dataset"])["train"]

algebras = []
geos = []
combi = []
calc = []
nt = []
for i in range(1000):
    if "Algebra" in targets[i]["domains"]:
        algebras.append(i)
    if "Geometry" in targets[i]["domains"]:
        geos.append(i)
    if "Combinatorics" in targets[i]["domains"]:
        combi.append(i)
    if "Calculus and Analysis" in targets[i]["domains"]:
        calc.append(i)
    if "Number Theory" in targets[i]["domains"]:
        nt.append(i)

all_scores = []
all_categories = []

if args.algebra:
    desired = algebras
    tag = "algebra"
elif args.calculus:
    desired = calc
    tag = "calculus"
elif args.geometry:
    desired = geos
    tag = "geometry"
elif args.combinatorics:
    desired = combi
    tag = "combinatorics"
elif args.number_theory:
    desired = nt
    tag = "nt"
else:
    desired = range(0, 1000)
    tag = "all"

# Number of scores per query
n_per_query = 150

# Counts at each within-query rank
blue_rank_counts = np.zeros(n_per_query)
green_rank_counts = np.zeros(n_per_query)
red_rank_counts = np.zeros(n_per_query)

# Total counts per category
total_blue = 0
total_green = 0
total_red = 0

for t in desired:
    scores = targets[t]["relevance_scores"]
    idxs = range(n_per_query)

    # Sort only within this query
    sorted_scores = sorted(zip(scores, idxs), key=lambda x: x[0], reverse=True)

    # rank is now the position within this query's sorted list
    for rank, (score, original_idx) in enumerate(sorted_scores):
        if original_idx < 50:
            blue_rank_counts[rank] += 1
            total_blue += 1
        elif original_idx < 100:
            green_rank_counts[rank] += 1
            total_green += 1
        else:
            red_rank_counts[rank] += 1
            total_red += 1

blue_props = np.cumsum(blue_rank_counts) / total_blue
green_props = np.cumsum(green_rank_counts) / total_green
red_props = np.cumsum(red_rank_counts) / total_red

x_vals = np.arange(n_per_query)


sns.set_style("whitegrid")

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 17,
        "axes.edgecolor": "0.8",
        "axes.linewidth": 1.0,
    }
)

fig, ax = plt.subplots(figsize=(8.5, 5.3))

ax.set_facecolor((0.96, 0.96, 0.96))

ax.plot(x_vals, blue_props, color="blue", linewidth=1.3, label="Ontology")

ax.plot(x_vals, green_props, color="green", linewidth=1.3, label="Solution summary")

ax.plot(x_vals, red_props, color="red", linewidth=1.3, label="Both")

ax.set_xlabel("Sorted rank")
ax.set_ylabel("Cumulative Proportion")

ax.set_xlim(1, n_per_query + 1)
# ax.set_xticks(np.arange(1, n_per_query + 1, 20))
ticks = [1] + list(np.arange(20, n_per_query + 1, 20))
ax.set_xticks(ticks)

ax.set_ylim(0, 1)
ax.set_yticks(np.linspace(0, 1, 6))

ax.set_xlim(0, 150)

ax.grid(True, color="0.85", linewidth=0.8, alpha=0.45)

legend = ax.legend(
    loc="upper left", frameon=True, fancybox=True, framealpha=0.85, borderpad=0.5
)

legend.get_frame().set_facecolor("white")
legend.get_frame().set_edgecolor("0.85")

sns.despine(left=False, bottom=False)

plt.tight_layout()

plt.savefig(f"signal_{tag}_cumulative_proportions.pdf", dpi=300, bbox_inches="tight")
