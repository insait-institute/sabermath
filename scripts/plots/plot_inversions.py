#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "construction"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", required=True)
    args = parser.parse_args(argv)

    json_file = args.results_json

    with open(json_file, "r") as f:
        data = json.load(f)

    pattern = r"average_num_inversions_after_(\d+)_rounds"

    points = []
    for key, value in data.items():
        match = re.match(pattern, key)
        if match:
            round_num = int(match.group(1))
            points.append((round_num, value))

    points.sort()

    rounds = [p[0] for p in points]
    avg_inversions = [p[1] for p in points]

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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "average_inversions_plot.pdf"
    plt.savefig(out_file, dpi=300)
    print(f"[+] wrote {out_file}")
    plt.show()


if __name__ == "__main__":
    main()
