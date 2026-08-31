# pip install datasets matplotlib numpy

from __future__ import annotations

import argparse
import re
import textwrap
from collections import defaultdict
from collections.abc import Mapping
from datasets import load_dataset, concatenate_datasets, Dataset
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# -----------------------------
# Domain/tag configuration
# -----------------------------

TAG_TO_DOMAIN = {
    "NumberTheory": "Number Theory",
    "Geometry": "Geometry",
    "Algebra": "Algebra",
    "CalculusandAnalysis": "Calculus and Analysis",
    "DiscreteMathematics": "Combinatorics",
    "ProbabilityandStatistics": "Combinatorics",
    "RecreationalMathematics": "Combinatorics",
}

DOMAIN_ORDER = [
    "Algebra",
    "Calculus and Analysis",
    "Combinatorics",
    "Geometry",
    "Number Theory",
]

DOMAIN_COLORS = {
    "Algebra": "#469c95",
    "Calculus and Analysis": "#7d83d5",
    "Combinatorics": "#f2ae35",
    "Geometry": "#3f51c5",
    "Number Theory": "#d63d5a",
}

DOMAIN_DISPLAY = {
    "Algebra": "Algebra",
    "Calculus and Analysis": "Calculus\n& Analysis",
    "Combinatorics": "Combinatorics",
    "Geometry": "Geometry",
    "Number Theory": "Number\nTheory",
}


# Generic tag components to ignore when choosing outer labels.
# You can add dataset-specific uninformative components here.
GENERIC_COMPONENTS = {
    "Mathematics",
    "DiscreteMathematics",
    "DiscreteMath",
    "FiniteMathematics",
    "GeneralDiscreteMathematics",
    "GeneralCombinatorics",
    "Combinatorics",
    "ProbabilityandStatistics",
    "RecreationalMathematics",
    "NumberTheory",
    "Geometry",
    "Algebra",
    "CalculusandAnalysis",
}


# -----------------------------
# Helper functions
# -----------------------------


def compact(s: str) -> str:
    """Normalize a string for comparisons."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


GENERIC_COMPONENTS_COMPACT = {compact(x) for x in GENERIC_COMPONENTS}


def pretty_label(component: str) -> str:
    """
    Convert tag component names like 'BinomialCoefficients' to 'Binomial Coefficients'.
    """
    s = component.strip().split("/")[-1]
    s = re.sub(r"[_\-]+", " ", s)

    # Split CamelCase.
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)

    s = re.sub(r"\s+", " ", s).strip()

    # Optional small cleanups.
    replacements = {
        "Eqs": "Equations",
        "Eq": "Equation",
        "Ineq": "Inequalities",
        "Num": "Number",
        "Geom": "Geometry",
        "Alg": "Algebra",
    }
    words = [replacements.get(w, w) for w in s.split()]
    return " ".join(words)


def wrap_label(label: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def split_tag(tag: str) -> tuple[str | None, list[str]]:
    """
    For a tag like
        /Mathematics/DiscreteMathematics/Combinatorics/BinomialCoefficients

    return
        top = 'DiscreteMathematics'
        rest = ['Combinatorics', 'BinomialCoefficients']
    """
    parts = [p for p in tag.split("/") if p]

    if not parts:
        return None, []

    if parts[0] == "Mathematics":
        if len(parts) < 2:
            return None, []
        return parts[1], parts[2:]

    return parts[0], parts[1:]


def domains_from_tags(tags: list[str]) -> list[str]:
    domains = set()

    for tag in tags:
        top, _ = split_tag(tag)
        if top in TAG_TO_DOMAIN:
            domains.add(TAG_TO_DOMAIN[top])

    return [d for d in DOMAIN_ORDER if d in domains]


def is_generic_component(component: str, domain: str) -> bool:
    c = compact(component)

    if c in GENERIC_COMPONENTS_COMPACT:
        return True

    # Skip the domain name itself.
    if c == compact(domain):
        return True

    # Skip things like GeneralAlgebra, GeneralGeometry, etc.
    if c.startswith("general"):
        return True

    return False


def extract_topic_from_tag(tag: str, domain: str) -> str | None:
    """
    Extract the first descriptive component after the top-level tag.

    Example:
        /Mathematics/DiscreteMathematics/Combinatorics/BinomialCoefficients

    top-level 'DiscreteMathematics' maps to domain 'Combinatorics'.
    We skip generic 'Combinatorics' and return 'Binomial Coefficients'.
    """
    top, rest = split_tag(tag)

    if top not in TAG_TO_DOMAIN:
        return None

    if TAG_TO_DOMAIN[top] != domain:
        return None

    for component in rest:
        if not is_generic_component(component, domain):
            return pretty_label(component)

    return None


def coerce_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def unique_preserving_order(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def mix_with_white(color: str, amount: float):
    """
    amount=0 gives original color, amount=1 gives white.
    """
    rgb = np.array(mcolors.to_rgb(color))
    white = np.ones(3)
    return tuple((1 - amount) * rgb + amount * white)


def spread_positions(targets, min_gap=0.075, low=-1.15, high=1.15):
    """
    Spread y-label positions to reduce label collisions.
    Assumes targets are already sorted bottom to top.
    """
    if not targets:
        return []

    ys = list(targets)

    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap

    if ys[-1] > high:
        shift = ys[-1] - high
        ys = [y - shift for y in ys]

    if ys[0] < low:
        shift = low - ys[0]
        ys = [y + shift for y in ys]

    # If there are too many labels to fit nicely, distribute uniformly.
    if len(ys) > 1 and ys[-1] > high:
        ys = np.linspace(low, high, len(ys)).tolist()

    return ys


# -----------------------------
# Aggregation
# -----------------------------


def aggregate_dataset_counts(ds):
    """
    Returns:
        domain_counts: dict domain -> fractional count
        topic_counts: dict (domain, topic) -> fractional count

    Each problem contributes total mass 1.

    If a problem has k domains, each domain gets 1/k.

    Within each domain, if the problem has m descriptive topics belonging to that
    domain, each topic gets 1/(k*m). If no descriptive topic is found, the mass
    goes to 'Other'.
    """
    domain_counts = defaultdict(float)
    topic_counts = defaultdict(float)

    n_examples_used = 0

    for ex in ds:
        tags = coerce_list(ex.get("tags", []))

        if "domains" in ex and ex["domains"] is not None:
            domains = coerce_list(ex["domains"])
        else:
            domains = domains_from_tags(tags)

        domains = unique_preserving_order(d for d in domains if d in DOMAIN_ORDER)

        if not domains:
            continue

        n_examples_used += 1

        domain_weight = 1.0 / len(domains)

        for domain in domains:
            domain_counts[domain] += domain_weight

        topics_by_domain = {domain: set() for domain in domains}

        for tag in tags:
            for domain in domains:
                topic = extract_topic_from_tag(tag, domain)
                if topic is not None:
                    topics_by_domain[domain].add(topic)

        for domain in domains:
            topics = sorted(topics_by_domain[domain])

            if not topics:
                topic_counts[(domain, "Other")] += domain_weight
            else:
                topic_weight = domain_weight / len(topics)
                for topic in topics:
                    topic_counts[(domain, topic)] += topic_weight

    if n_examples_used == 0:
        raise ValueError("No examples with recognized domains were found.")

    return dict(domain_counts), dict(topic_counts)


def get_domain_topic_limit(max_topics_per_domain, domain):
    if max_topics_per_domain is None:
        return None

    if isinstance(max_topics_per_domain, dict):
        return max_topics_per_domain.get(
            domain, max_topics_per_domain.get("__default__", None)
        )

    return max_topics_per_domain


def prepare_plot_data(
    domain_counts,
    topic_counts,
    max_topics_per_domain=10,
):
    """
    For each domain:

    If max_topics_per_domain[domain] = X, then the outer ring will show at most
    X subdomains for that domain.

    If there are more than X true subdomains, we show:
        - top X - 1 subdomains individually
        - one 'Other' wedge containing all remaining subdomains

    Therefore X counts the number of visible/captioned subdomains, including
    'Other'.
    """

    inner_labels = []
    inner_values = []
    inner_colors = []

    outer_labels = []
    outer_values = []
    outer_domains = []
    outer_colors = []

    for domain in DOMAIN_ORDER:
        domain_value = domain_counts.get(domain, 0.0)

        if domain_value <= 0:
            continue

        inner_labels.append(domain)
        inner_values.append(domain_value)
        inner_colors.append(DOMAIN_COLORS[domain])

        # All topics belonging to this domain.
        items = [
            (topic, value)
            for (d, topic), value in topic_counts.items()
            if d == domain and value > 0
        ]

        # Separate pre-existing Other, if it exists.
        existing_other_value = sum(v for t, v in items if t == "Other")
        normal_items = [(t, v) for t, v in items if t != "Other"]

        # Sort topics by frequency.
        normal_items.sort(key=lambda x: x[1], reverse=True)

        domain_max_topics = get_domain_topic_limit(
            max_topics_per_domain,
            domain,
        )

        if domain_max_topics is None:
            # Show everything.
            kept = normal_items
            other_value = existing_other_value

        else:
            domain_max_topics = int(domain_max_topics)

            if domain_max_topics <= 0:
                # Show only Other.
                kept = []
                other_value = existing_other_value + sum(v for _, v in normal_items)

            elif len(normal_items) + int(existing_other_value > 0) <= domain_max_topics:
                # Number of topics already fits within the limit.
                kept = normal_items
                other_value = existing_other_value

            else:
                # Need to group smaller topics into Other.
                #
                # Since Other itself uses one visible slot, keep only X - 1
                # named topics.
                n_named_to_keep = domain_max_topics - 1

                kept = normal_items[:n_named_to_keep]

                other_value = existing_other_value + sum(
                    v for _, v in normal_items[n_named_to_keep:]
                )

        # Add Other as one visible/captioned subdomain if it has mass.
        if other_value > 0:
            kept.append(("Other", other_value))

        # Now every item in kept will become exactly one outer wedge
        # and exactly one caption.
        n = max(len(kept), 1)
        lighten_amounts = np.linspace(0.25, 0.70, n)

        for i, (topic, value) in enumerate(kept):
            outer_labels.append(topic)
            outer_values.append(value)
            outer_domains.append(domain)
            outer_colors.append(
                mix_with_white(DOMAIN_COLORS[domain], lighten_amounts[i])
            )

    return {
        "inner_labels": inner_labels,
        "inner_values": inner_values,
        "inner_colors": inner_colors,
        "outer_labels": outer_labels,
        "outer_values": outer_values,
        "outer_domains": outer_domains,
        "outer_colors": outer_colors,
    }


# -----------------------------
# Plotting
# -----------------------------
def spread_positions_variable(targets, gaps, low=-1.25, high=1.25):
    """
    Spread label y-positions with possibly different gaps between labels.

    targets: desired y positions, sorted bottom to top
    gaps: minimum gaps between consecutive labels, length len(targets)-1
    """
    n = len(targets)

    if n == 0:
        return []

    if n == 1:
        return [min(max(targets[0], low), high)]

    available = high - low
    required = sum(gaps)

    # If there is not enough room, distribute uniformly.
    if required > available:
        return np.linspace(low, high, n).tolist()

    ys = np.array(targets, dtype=float)

    # Forward pass.
    ys[0] = max(ys[0], low)

    for i in range(1, n):
        ys[i] = max(ys[i], ys[i - 1] + gaps[i - 1])

    # Shift down if top exceeds high.
    if ys[-1] > high:
        ys -= ys[-1] - high

    # If that pushed the bottom too low, anchor at low and rebuild.
    if ys[0] < low:
        ys[0] = low
        for i in range(1, n):
            ys[i] = ys[i - 1] + gaps[i - 1]

    return ys.tolist()


def add_outer_labels(
    ax,
    wedges,
    labels,
    values,
    domains,
    total,
    min_label_pct: float = 0.0,
    label_radius: float = 1.55,
    elbow_radius: float = 1.13,
    fontsize: int = 9,
    hide_labels: set[str] | None = None,
    label_min_gap: float = 0.12,
    label_multiline_extra_gap: float = 0.055,
    label_y_low: float = -1.38,
    label_y_high: float = 1.38,
    wrap_width: int = 18,
):
    """
    Outer labels with controllable spacing.

    Important parameters
    --------------------
    label_min_gap:
        Base vertical distance between neighboring labels.

    label_multiline_extra_gap:
        Extra space added when labels have multiple lines.

    label_y_low, label_y_high:
        Vertical area where labels are allowed to live.

    fontsize:
        Font size of the outer labels.

    wrap_width:
        Maximum characters per line before wrapping.
    """

    if hide_labels is None:
        hide_labels = set()

    entries = []

    for wedge, label, value, domain in zip(wedges, labels, values, domains):
        pct = value / total

        if pct < min_label_pct:
            continue

        if label in hide_labels:
            continue

        theta = 0.5 * (wedge.theta1 + wedge.theta2)
        rad = np.deg2rad(theta)

        x = np.cos(rad)
        y = np.sin(rad)

        side = 1 if x >= 0 else -1

        wrapped = wrap_label(label, width=wrap_width)
        n_lines = wrapped.count("\n") + 1

        entries.append(
            {
                "side": side,
                "x": x,
                "y": y,
                "target_y": y * 1.08,
                "label": wrapped,
                "domain": domain,
                "n_lines": n_lines,
            }
        )

    for side in [-1, 1]:
        side_entries = [e for e in entries if e["side"] == side]
        side_entries.sort(key=lambda e: e["target_y"])

        if not side_entries:
            continue

        targets = [e["target_y"] for e in side_entries]

        gaps = []

        for e1, e2 in zip(side_entries[:-1], side_entries[1:]):
            multiline_penalty = max(e1["n_lines"], e2["n_lines"]) - 1

            gap = label_min_gap + multiline_penalty * label_multiline_extra_gap

            gaps.append(gap)

        adjusted_ys = spread_positions_variable(
            targets,
            gaps,
            low=label_y_low,
            high=label_y_high,
        )

        for e, y_text in zip(side_entries, adjusted_ys):
            x0 = e["x"] * 1.00
            y0 = e["y"] * 1.00

            x1 = side * elbow_radius
            y1 = y_text

            x2 = side * (label_radius - 0.05)

            line_color = DOMAIN_COLORS[e["domain"]]

            ax.plot(
                [x0, x1, x2],
                [y0, y1, y1],
                color=line_color,
                lw=0.75,
                alpha=0.60,
                solid_capstyle="round",
                zorder=3,
            )

            ha = "left" if side == 1 else "right"
            x_text = side * label_radius

            ax.text(
                x_text,
                y_text,
                e["label"],
                ha=ha,
                va="center",
                fontsize=fontsize,
                color=DOMAIN_COLORS[e["domain"]],
                zorder=4,
            )


def plot_math_dataset_sunburst(
    ds,
    split: str | None = None,
    out_file: str = "math_dataset_sunburst.pdf",
    title: str | None = None,
    max_topics_per_domain: int | None = 10,
    min_outer_label_pct: float = 0.0,
    startangle: float = 90,
    counterclock: bool = False,
    show: bool = True,
):
    """
    Main function.

    Parameters
    ----------
    ds:
        Hugging Face Dataset or DatasetDict.
    split:
        If ds is a DatasetDict, use this split. If None and ds is a DatasetDict,
        the first split is used.
    out_file:
        Output image path.
    max_topics_per_domain:
        Maximum number of outer topics shown per domain. Smaller values make the
        figure cleaner. Use None to show every extracted topic.
    min_outer_label_pct:
        Hide outer labels below this global fraction. For example 0.005 hides
        labels smaller than 0.5% of the whole dataset.
    """
    if isinstance(ds, Mapping):
        if split is None:
            split = next(iter(ds.keys()))
        ds = ds[split]

    domain_counts, topic_counts = aggregate_dataset_counts(ds)

    plot_data = prepare_plot_data(
        domain_counts,
        topic_counts,
        max_topics_per_domain=max_topics_per_domain,
    )

    inner_labels = plot_data["inner_labels"]
    inner_values = plot_data["inner_values"]
    inner_colors = plot_data["inner_colors"]

    outer_labels = plot_data["outer_labels"]
    outer_values = plot_data["outer_values"]
    outer_domains = plot_data["outer_domains"]
    outer_colors = plot_data["outer_colors"]

    total = sum(inner_values)

    fig, ax = plt.subplots(figsize=(13, 8), subplot_kw={"aspect": "equal"})

    outer_radius = 1.05
    outer_width = 0.30

    inner_radius = 0.75
    inner_width = 0.42

    outer_wedges, _ = ax.pie(
        outer_values,
        radius=outer_radius,
        startangle=startangle,
        counterclock=counterclock,
        colors=outer_colors,
        labels=None,
        wedgeprops={
            "width": outer_width,
            "edgecolor": "white",
            "linewidth": 0.9,
        },
    )

    inner_wedges, _ = ax.pie(
        inner_values,
        radius=inner_radius,
        startangle=startangle,
        counterclock=counterclock,
        colors=inner_colors,
        labels=None,
        wedgeprops={
            "width": inner_width,
            "edgecolor": "white",
            "linewidth": 1.2,
        },
    )

    # White center hole.
    hole_radius = inner_radius - inner_width
    center = plt.Circle((0, 0), hole_radius, fc="white", ec="white", zorder=2)
    ax.add_artist(center)

    ax.text(
        0,
        0,
        "283K\nproblems",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#333333",
        zorder=6,
    )

    # Inner domain labels.
    label_r = hole_radius + inner_width * 0.45

    for wedge, domain, value in zip(inner_wedges, inner_labels, inner_values):
        theta = 0.5 * (wedge.theta1 + wedge.theta2)
        rad = np.deg2rad(theta)

        x = label_r * np.cos(rad)
        y = label_r * np.sin(rad)

        pct = value / total * 100

        pct = value / total * 100

        ax.text(
            x,
            y,
            f"{DOMAIN_DISPLAY.get(domain, domain)}\n{pct:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
            zorder=5,
        )

    add_outer_labels(
        ax=ax,
        wedges=outer_wedges,
        labels=outer_labels,
        values=outer_values,
        domains=outer_domains,
        total=total,
        min_label_pct=0.0,
        fontsize=13,
        hide_labels=set(),
        label_min_gap=0.13,
        label_multiline_extra_gap=0.06,
        label_y_low=-1.45,
        label_y_high=1.45,
        label_radius=1.62,
        elbow_radius=1.15,
        wrap_width=18,
    )

    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=18)

    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.60, 1.60)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


TAG_TO_DOMAIN = {
    "NumberTheory": "Number Theory",
    "Geometry": "Geometry",
    "Algebra": "Algebra",
    "CalculusandAnalysis": "Calculus and Analysis",
    "DiscreteMathematics": "Combinatorics",
    "ProbabilityandStatistics": "Combinatorics",
    "RecreationalMathematics": "Combinatorics",
}


def get_domains(ds: Dataset, idx: int) -> list[str]:
    tags = [tag.split("/")[2] for tag in ds[idx]["tags"]]
    domains = set()
    for tag in tags:
        domain = TAG_TO_DOMAIN.get(tag)
        if domain:
            domains.add(domain)
    return list(domains)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--databank_dataset", required=True)
    args = parser.parse_args

    ds = load_dataset(args.databank_dataset)
    ds = concatenate_datasets([ds["numeric"], ds["proofs"]])

    print(ds[0].keys())

    for _ in tqdm.tqdm(range(len(ds))):
        ds[_]["domains"] = get_domains(ds, _)

    plot_math_dataset_sunburst(
        ds,
        out_file="piechart_databank.pdf",
        title=None,
        max_topics_per_domain={
            "Algebra": 8,
            "Geometry": 6,
            "Number Theory": 7,
            "Calculus and Analysis": 7,
            "Combinatorics": 6,
        },
        min_outer_label_pct=0.006,
    )
