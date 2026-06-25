import argparse
import json
import re
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--results_json", required=True)
args = parser.parse_args()

json_file = args.results_json

# Load JSON file
with open(json_file, "r") as f:
    data = json.load(f)

# Extract rounds and average inversion values
pattern = r"average_num_inversions_after_(\d+)_rounds"

points = []
for key, value in data.items():
    match = re.match(pattern, key)
    if match:
        round_num = int(match.group(1))
        points.append((round_num, value))

# Sort by round number
points.sort()

rounds = [p[0] for p in points]
avg_inversions = [p[1] for p in points]

# Plot
plt.figure(figsize=(5.67, 3.2))
ax = plt.gca()

ax.plot(
    rounds, avg_inversions, marker="o", linewidth=1.2, markersize=3, color="#1f77b4"
)

ax.set_xlabel("Round", fontsize=13)
ax.set_ylabel("Average No. Inversions", fontsize=13)

ax.set_xticks(rounds)
ax.set_ylim(bottom=0)

ax.grid(True, alpha=0.25)
ax.set_facecolor("#f7f7f7")

plt.tight_layout()
plt.savefig("average_inversions_plot.pdf", dpi=300)
plt.show()
