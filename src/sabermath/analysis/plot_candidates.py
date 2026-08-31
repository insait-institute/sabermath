# pip install datasets matplotlib numpy

from __future__ import annotations

import argparse
import math
import re
import textwrap
from collections import defaultdict
from collections.abc import Mapping

from datasets import load_dataset

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


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
    "Calculus and Analysis": "Calculus & Analysis",
    "Combinatorics": "Combinatorics",
    "Geometry": "Geometry",
    "Number Theory": "Number Theory",
}

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
    return re.sub(r"[^a-z0-9]+", "", s.lower())


GENERIC_COMPONENTS_COMPACT = {compact(x) for x in GENERIC_COMPONENTS}


def normalize_domain_name(x: str) -> str | None:
    if x is None:
        return None

    x = str(x)

    if x in DOMAIN_ORDER:
        return x

    if x in TAG_TO_DOMAIN:
        return TAG_TO_DOMAIN[x]

    cx = compact(x)

    for domain in DOMAIN_ORDER:
        if cx == compact(domain):
            return domain

    for tag_domain, display_domain in TAG_TO_DOMAIN.items():
        if cx == compact(tag_domain):
            return display_domain

    return None


def pretty_label(component: str) -> str:
    """
    Convert tag component names like `BinomialCoefficients` to
    `Binomial Coefficients`.
    """
    s = component.strip().split("/")[-1]
    s = re.sub(r"[_\-]+", " ", s)

    # Split CamelCase.
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)

    s = re.sub(r"\s+", " ", s).strip()

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
    parts = [p for p in str(tag).split("/") if p]

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

    if c == compact(domain):
        return True

    if c.startswith("general"):
        return True

    return False


def extract_topic_from_tag(tag: str, domain: str) -> str | None:
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
    `amount=0` gives original color, `amount=1` gives white.
    """
    rgb = np.array(mcolors.to_rgb(color))
    white = np.ones(3)
    return tuple((1 - amount) * rgb + amount * white)


def get_example_domains(ex: dict) -> list[str]:
    """
    Get recognized domains for one example.
    """
    tags = coerce_list(ex.get("tags", []))

    if "domains" in ex and ex["domains"] is not None:
        raw_domains = coerce_list(ex["domains"])

        normalized_domains = [normalize_domain_name(domain) for domain in raw_domains]

        domains = [domain for domain in normalized_domains if domain in DOMAIN_ORDER]

        if not domains:
            domains = domains_from_tags(tags)
    else:
        domains = domains_from_tags(tags)

    domains = unique_preserving_order(d for d in domains if d in DOMAIN_ORDER)

    return domains


def get_example_topics_by_domain(ex: dict) -> dict[str, list[str]]:
    """
    Extract descriptive subdomains/topics for a single example.
    """
    tags = coerce_list(ex.get("tags", []))
    domains = get_example_domains(ex)

    topics_by_domain = {domain: set() for domain in domains}

    for tag in tags:
        for domain in domains:
            topic = extract_topic_from_tag(tag, domain)

            if topic is not None:
                topics_by_domain[domain].add(topic)

    out = {}

    for domain in domains:
        topics = sorted(topics_by_domain[domain])

        if topics:
            out[domain] = topics
        else:
            out[domain] = ["Other"]

    return out


def get_query_topic_keys(target: dict) -> set[tuple[str, str]]:
    """
    Return the query's own `(domain, topic)` pairs.

    These are used to shade corresponding outer labels.
    """
    topics_by_domain = get_example_topics_by_domain(target)

    query_topic_keys = set()

    for domain, topics in topics_by_domain.items():
        if domain not in DOMAIN_ORDER:
            continue

        for topic in topics:
            query_topic_keys.add((domain, topic))

    return query_topic_keys


def get_prevailing_domains(
    domain_counts: Mapping[str, float],
    total: float,
    *,
    threshold: float = 0.50,
    fallback_to_largest: bool = True,
) -> list[str]:
    """
    Return domains whose share is at least `threshold`.

    If none reach the threshold and `fallback_to_largest=True`, return the
    single largest non-empty domain.
    """
    if total <= 0:
        return []

    prevailing = [
        domain
        for domain in DOMAIN_ORDER
        if domain_counts.get(domain, 0.0) / total >= threshold
    ]

    if prevailing:
        return prevailing

    if not fallback_to_largest:
        return []

    non_empty = [
        (domain, domain_counts.get(domain, 0.0))
        for domain in DOMAIN_ORDER
        if domain_counts.get(domain, 0.0) > 0
    ]

    if not non_empty:
        return []

    largest_domain, _ = max(non_empty, key=lambda x: x[1])
    return [largest_domain]


# -----------------------------
# Relevance helpers
# -----------------------------


def get_top_k_candidate_ids(target: dict, k: int = 10) -> list[int]:
    """
    Return candidate ids for the top-k most relevant candidates.

    Assumes higher relevance score means more relevant.
    """
    scores = target["relevance_scores"]
    candidate_ids = target["candidates"]

    if len(scores) != len(candidate_ids):
        raise ValueError(
            "`relevance_scores` and `candidates` must have the same length."
        )

    sorted_pairs = sorted(
        zip(scores, candidate_ids),
        key=lambda pair: pair[0],
        reverse=True,
    )

    return [candidate_id for _, candidate_id in sorted_pairs[:k]]


def get_top_k_candidate_examples(
    target: dict,
    candidates_dataset,
    k: int = 10,
) -> list[dict]:
    candidate_ids = get_top_k_candidate_ids(target, k=k)
    return [candidates_dataset[candidate_id] for candidate_id in candidate_ids]


def get_top_relevant_topic_marker_counts(
    examples: list[dict],
) -> dict[tuple[str, str], int]:
    """
    Count which candidate subdomains/topics occur among the top relevant
    candidates.
    """
    marker_counts = defaultdict(int)

    for ex in examples:
        topics_by_domain = get_example_topics_by_domain(ex)

        labels_for_candidate = []

        for domain in DOMAIN_ORDER:
            topics = topics_by_domain.get(domain, [])

            for topic in topics:
                labels_for_candidate.append((domain, topic))

        if not labels_for_candidate:
            continue

        for key in sorted(set(labels_for_candidate)):
            marker_counts[key] += 1

    return dict(marker_counts)


# -----------------------------
# Aggregation
# -----------------------------


def aggregate_dataset_counts(ds):
    """
    Returns:

    - `domain_counts`: dict domain -> fractional count
    - `topic_counts`: dict `(domain, topic)` -> fractional count

    Each problem contributes total mass 1.
    """
    domain_counts = defaultdict(float)
    topic_counts = defaultdict(float)

    n_examples_used = 0

    for ex in ds:
        tags = coerce_list(ex.get("tags", []))
        domains = get_example_domains(ex)

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
            domain,
            max_topics_per_domain.get("__default__", None),
        )

    return max_topics_per_domain


def build_effective_max_topics_per_domain(
    max_topics_per_domain,
    prevailing_domains: set[str],
):
    """
    Prevailing domains always get all subdomains/topics in the outer ring.
    Smaller domains use the requested limit.
    """
    effective = {}

    for domain in DOMAIN_ORDER:
        if domain in prevailing_domains:
            effective[domain] = None
        else:
            effective[domain] = get_domain_topic_limit(
                max_topics_per_domain,
                domain,
            )

    return effective


def prepare_plot_data(
    domain_counts,
    topic_counts,
    max_topics_per_domain=10,
    force_keep_topics_by_domain: dict[str, set[str]] | None = None,
):
    """
    Prepare inner/outer ring data.

    `force_keep_topics_by_domain` keeps marked/query topics as individual wedges
    even if they would otherwise be grouped into `Other`.
    """
    if force_keep_topics_by_domain is None:
        force_keep_topics_by_domain = {}

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

        items = [
            (topic, value)
            for (d, topic), value in topic_counts.items()
            if d == domain and value > 0
        ]

        existing_other_value = sum(v for t, v in items if t == "Other")
        normal_items = [(t, v) for t, v in items if t != "Other"]

        normal_items.sort(key=lambda x: x[1], reverse=True)

        forced_topics = force_keep_topics_by_domain.get(domain, set())

        domain_max_topics = get_domain_topic_limit(
            max_topics_per_domain,
            domain,
        )

        if domain_max_topics is None:
            kept = normal_items
            other_value = existing_other_value

        else:
            domain_max_topics = int(domain_max_topics)

            if len(normal_items) + int(existing_other_value > 0) <= domain_max_topics:
                kept = normal_items
                other_value = existing_other_value

            else:
                forced_items = [
                    item for item in normal_items if item[0] in forced_topics
                ]

                nonforced_items = [
                    item for item in normal_items if item[0] not in forced_topics
                ]

                if domain_max_topics <= 0:
                    kept = forced_items
                    other_value = existing_other_value + sum(
                        v for _, v in nonforced_items
                    )
                else:
                    # Reserve one slot for Other, but never drop forced topics.
                    n_nonforced_to_keep = max(
                        domain_max_topics - 1 - len(forced_items),
                        0,
                    )

                    kept = forced_items + nonforced_items[:n_nonforced_to_keep]

                    kept_topics = {topic for topic, _ in kept}

                    other_value = existing_other_value + sum(
                        value
                        for topic, value in normal_items
                        if topic not in kept_topics
                    )

        if other_value > 0:
            kept.append(("Other", other_value))

        n = max(len(kept), 1)
        lighten_amounts = np.linspace(0.22, 0.68, n)

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
# Plotting helpers
# -----------------------------


def spread_positions_variable(targets, gaps, low=-1.10, high=1.10):
    """
    Spread label y-positions with possibly different gaps between labels.
    """
    n = len(targets)

    if n == 0:
        return []

    if n == 1:
        return [min(max(targets[0], low), high)]

    available = high - low
    required = sum(gaps)

    if required > available:
        return np.linspace(low, high, n).tolist()

    ys = np.array(targets, dtype=float)

    ys[0] = max(ys[0], low)

    for i in range(1, n):
        ys[i] = max(ys[i], ys[i - 1] + gaps[i - 1])

    if ys[-1] > high:
        ys -= ys[-1] - high

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
    label_radius: float = 1.48,
    elbow_radius: float = 1.17,
    fontsize: int = 10,
    hide_labels: set[str] | None = None,
    hide_domains: set[str] | None = None,
    show_domains: set[str] | None = None,
    force_label_keys: set[tuple[str, str]] | None = None,
    query_label_keys: set[tuple[str, str]] | None = None,
    max_labels_per_domain: int | None = None,
    label_min_gap: float = 0.115,
    label_multiline_extra_gap: float = 0.045,
    label_y_low: float = -1.36,
    label_y_high: float = 1.36,
    wrap_width: int = 16,
):
    """
    Add outside labels with leader lines for selected outer wedges.

    Query labels are shaded.
    Top-k relevant labels are forced.
    """
    if hide_labels is None:
        hide_labels = set()

    if hide_domains is None:
        hide_domains = set()

    if force_label_keys is None:
        force_label_keys = set()

    if query_label_keys is None:
        query_label_keys = set()

    candidate_items = []

    for wedge, label, value, domain in zip(wedges, labels, values, domains):
        key = (domain, label)

        is_forced = key in force_label_keys
        is_query = key in query_label_keys
        is_important = is_forced or is_query

        pct = value / total

        if not is_important:
            if pct < min_label_pct:
                continue

            if label in hide_labels:
                continue

            if domain in hide_domains:
                continue

            if show_domains is not None and domain not in show_domains:
                continue

        candidate_items.append(
            {
                "wedge": wedge,
                "label": label,
                "value": value,
                "domain": domain,
                "is_forced": is_forced,
                "is_query": is_query,
            }
        )

    if max_labels_per_domain is not None:
        kept_candidate_items = []

        for domain in DOMAIN_ORDER:
            domain_items = [
                item for item in candidate_items if item["domain"] == domain
            ]

            important_items = [
                item for item in domain_items if item["is_forced"] or item["is_query"]
            ]

            normal_items = [
                item
                for item in domain_items
                if not item["is_forced"] and not item["is_query"]
            ]

            normal_items.sort(
                key=lambda item: item["value"],
                reverse=True,
            )

            remaining_slots = max(
                max_labels_per_domain - len(important_items),
                0,
            )

            kept_candidate_items.extend(
                important_items + normal_items[:remaining_slots]
            )

        kept_ids = {id(item) for item in kept_candidate_items}

        candidate_items = [item for item in candidate_items if id(item) in kept_ids]

    entries = []

    for item in candidate_items:
        wedge = item["wedge"]
        label = item["label"]
        domain = item["domain"]
        is_forced = item["is_forced"]
        is_query = item["is_query"]

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
                "is_forced": is_forced,
                "is_query": is_query,
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

            x2 = side * (label_radius - 0.04)

            line_color = DOMAIN_COLORS[e["domain"]]

            is_query = e["is_query"]
            is_forced = e["is_forced"]

            if is_query:
                lw = 1.55
                alpha = 0.94
            elif is_forced:
                lw = 1.30
                alpha = 0.88
            else:
                lw = 0.85
                alpha = 0.64

            ax.plot(
                [x0, x1, x2],
                [y0, y1, y1],
                color=line_color,
                lw=lw,
                alpha=alpha,
                solid_capstyle="round",
                zorder=3,
                clip_on=False,
            )

            ha = "left" if side == 1 else "right"
            x_text = side * label_radius

            if is_query:
                text_bbox = {
                    "boxstyle": "round,pad=0.24",
                    "facecolor": mix_with_white(DOMAIN_COLORS[e["domain"]], 0.82),
                    "edgecolor": DOMAIN_COLORS[e["domain"]],
                    "linewidth": 1.0,
                    "alpha": 0.95,
                }
            else:
                text_bbox = None

            ax.text(
                x_text,
                y_text,
                e["label"],
                ha=ha,
                va="center",
                fontsize=fontsize,
                color=DOMAIN_COLORS[e["domain"]],
                fontweight="bold" if is_query or is_forced else "normal",
                bbox=text_bbox,
                zorder=5 if is_query else 4,
                clip_on=False,
            )


def add_top_relevant_black_marks(
    ax,
    wedges,
    labels,
    domains,
    marker_counts: dict[tuple[str, str], int],
    *,
    ring_inner_radius: float,
    ring_outer_radius: float,
    max_marks_per_wedge: int = 10,
):
    """
    Add black radial tick marks inside the outer ring.

    Important placement rule:
    For each marked subdomain wedge, if there are `n` black ticks, the angular
    gap from the wedge start boundary to the first tick, the gap between every
    pair of consecutive ticks, and the gap from the last tick to the wedge end
    boundary are all equal.

    Matplotlib wedge angles are in degrees, but equal spacing in degrees is
    exactly equivalent to equal spacing in radians because the conversion is
    linear.
    """
    if not marker_counts:
        return

    wedge_by_key = {
        (domain, label): wedge for wedge, label, domain in zip(wedges, labels, domains)
    }

    r0 = ring_inner_radius + 0.018
    r1 = ring_outer_radius - 0.018

    for key, count in marker_counts.items():
        wedge = wedge_by_key.get(key)

        if wedge is None:
            continue

        n_marks = max(1, min(int(count), max_marks_per_wedge))

        theta1 = wedge.theta1
        theta2 = wedge.theta2

        # Equal-gap rule:
        #
        # If there are n marks, divide the wedge angular span into n + 1 equal
        # gaps and place ticks after gap 1, gap 2, ..., gap n.
        #
        # This makes:
        #   boundary -> first tick
        #   tick i -> tick i+1
        #   last tick -> boundary
        # all equal in angular distance.
        span = theta2 - theta1
        gap = span / (n_marks + 1)

        thetas = [theta1 + (i + 1) * gap for i in range(n_marks)]

        for theta in thetas:
            rad = np.deg2rad(theta)

            x0 = r0 * np.cos(rad)
            y0 = r0 * np.sin(rad)

            x1 = r1 * np.cos(rad)
            y1 = r1 * np.sin(rad)

            ax.plot(
                [x0, x1],
                [y0, y1],
                color="black",
                lw=1.55,
                alpha=0.94,
                solid_capstyle="round",
                zorder=8,
                clip_on=False,
            )


def make_topic_handles_for_domain(
    domain,
    outer_labels,
    outer_values,
    outer_domains,
    outer_colors,
    total,
    marker_counts: dict[tuple[str, str], int] | None = None,
    max_items: int | None = None,
    legend_label_width: int = 28,
):
    if marker_counts is None:
        marker_counts = {}

    items = [
        (label, value, color)
        for label, value, d, color in zip(
            outer_labels,
            outer_values,
            outer_domains,
            outer_colors,
        )
        if d == domain
    ]

    items.sort(key=lambda x: x[1], reverse=True)

    forced_topics = {
        topic
        for (d, topic), count in marker_counts.items()
        if d == domain and count > 0
    }

    if max_items is not None and len(items) > max_items:
        forced_items = [item for item in items if item[0] in forced_topics]

        nonforced_items = [item for item in items if item[0] not in forced_topics]

        remaining_slots = max(max_items - 1 - len(forced_items), 0)
        kept = forced_items + nonforced_items[:remaining_slots]

        kept_topics = {label for label, _, _ in kept}

        remaining_value = sum(
            value for label, value, _ in items if label not in kept_topics
        )

        if remaining_value > 0:
            items = kept + [
                (
                    "Remaining topics",
                    remaining_value,
                    mix_with_white(DOMAIN_COLORS[domain], 0.78),
                )
            ]
        else:
            items = kept

    handles = []

    for label, value, color in items:
        pct = value / total * 100
        display_label = wrap_label(label, width=legend_label_width).replace("\n", " ")

        display_label = f"{display_label} — {pct:.1f}%"

        handles.append(
            Patch(
                facecolor=color,
                edgecolor="white",
                label=display_label,
            )
        )

    return handles


def add_topic_legend_axis(
    ax,
    domain,
    outer_labels,
    outer_values,
    outer_domains,
    outer_colors,
    total,
    marker_counts: dict[tuple[str, str], int] | None = None,
    max_items: int | None = None,
):
    ax.axis("off")

    handles = make_topic_handles_for_domain(
        domain=domain,
        outer_labels=outer_labels,
        outer_values=outer_values,
        outer_domains=outer_domains,
        outer_colors=outer_colors,
        total=total,
        marker_counts=marker_counts,
        max_items=max_items,
    )

    if not handles:
        return

    domain_total = sum(
        value for value, d in zip(outer_values, outer_domains) if d == domain
    )

    domain_pct = domain_total / total * 100

    legend = ax.legend(
        handles=handles,
        loc="center",
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        borderpad=0.75,
        labelspacing=0.55,
        handlelength=1.8,
        handletextpad=0.70,
        fontsize=14,
        title=f"{DOMAIN_DISPLAY.get(domain, domain)} — {domain_pct:.1f}%",
        title_fontsize=14.0,
    )

    legend.get_frame().set_edgecolor("#dddddd")
    legend.get_frame().set_linewidth(0.9)


def add_domain_legend_axis(
    ax,
    domain_labels,
    domain_values,
    total,
    *,
    top_k_relevant: int | None = None,
):
    ax.axis("off")

    handles = []

    for domain, value in zip(domain_labels, domain_values):
        pct = value / total * 100

        handles.append(
            Patch(
                facecolor=DOMAIN_COLORS[domain],
                edgecolor="white",
                label=f"{DOMAIN_DISPLAY.get(domain, domain)}: {pct:.1f}%",
            )
        )

    ax.legend(
        handles=handles,
        loc="center",
        ncol=min(len(handles), 3),
        frameon=False,
        fontsize=14.0,
        title="Candidate domains",
        title_fontsize=14.0,
        handlelength=1.9,
        columnspacing=1.4,
        handletextpad=0.65,
    )


# -----------------------------
# Main plot function
# -----------------------------


def plot_math_dataset_sunburst(
    ds,
    split: str | None = None,
    out_file: str = "math_dataset_sunburst.pdf",
    title: str | None = None,
    query_label_keys: set[tuple[str, str]] | None = None,
    top_relevant_marker_counts: dict[tuple[str, str], int] | None = None,
    top_k_relevant: int = 10,
    max_topics_per_domain: int | dict[str, int] | None = 10,
    min_outer_label_pct: float = 0.015,
    prevailing_domain_threshold: float = 0.50,
    prevailing_label_min_pct: float = 0.0,
    max_prevailing_labeled_topics: int | None = None,
    small_domain_legend_max_items: int | None = None,
    startangle: float = 90,
    counterclock: bool = False,
    show: bool = True,
    legend_domains: set[str] | None = None,
):
    """
    Draw a two-ring sunburst.

    This version:
    - Has no query title/annotation at the top.
    - Uses larger fonts.
    - Uses larger spacing between subdomain labels.
    - Places black ticks with equal angular gaps to wedge boundaries and
      consecutive ticks.
    """
    if top_relevant_marker_counts is None:
        top_relevant_marker_counts = {}

    if query_label_keys is None:
        query_label_keys = set()

    if isinstance(ds, Mapping):
        if split is None:
            split = next(iter(ds.keys()))
        ds = ds[split]

    domain_counts, topic_counts = aggregate_dataset_counts(ds)

    raw_total = sum(domain_counts.values())

    prevailing_domains = set(
        get_prevailing_domains(
            domain_counts,
            raw_total,
            threshold=prevailing_domain_threshold,
            fallback_to_largest=True,
        )
    )

    effective_max_topics_per_domain = build_effective_max_topics_per_domain(
        max_topics_per_domain=max_topics_per_domain,
        prevailing_domains=prevailing_domains,
    )

    force_keep_topics_by_domain = defaultdict(set)

    # Keep top-k relevant marked topics as individual wedges.
    for domain, topic in top_relevant_marker_counts.keys():
        if domain in DOMAIN_ORDER:
            force_keep_topics_by_domain[domain].add(topic)

    # Keep query subdomains as individual wedges, so they can be shaded.
    for domain, topic in query_label_keys:
        if domain in DOMAIN_ORDER:
            force_keep_topics_by_domain[domain].add(topic)

    plot_data = prepare_plot_data(
        domain_counts,
        topic_counts,
        max_topics_per_domain=effective_max_topics_per_domain,
        force_keep_topics_by_domain=dict(force_keep_topics_by_domain),
    )

    inner_labels = plot_data["inner_labels"]
    inner_values = plot_data["inner_values"]
    inner_colors = plot_data["inner_colors"]

    outer_labels = plot_data["outer_labels"]
    outer_values = plot_data["outer_values"]
    outer_domains = plot_data["outer_domains"]
    outer_colors = plot_data["outer_colors"]

    total = sum(inner_values)

    if not np.isclose(total, 150):
        print(f"Warning: expected total mass close to 150, got {total:.6f}.")

    present_domains = {
        domain for domain in DOMAIN_ORDER if domain_counts.get(domain, 0.0) > 0
    }

    if legend_domains is None:
        legend_domains_ordered = [
            domain
            for domain in DOMAIN_ORDER
            if domain in present_domains and domain not in prevailing_domains
        ]
    else:
        legend_domains_ordered = [
            domain
            for domain in DOMAIN_ORDER
            if domain in present_domains and domain in legend_domains
        ]

    legend_rows = math.ceil(len(legend_domains_ordered) / 2)

    # Dynamic layout:
    # - row 0: sunburst chart
    # - middle rows: topic legends
    # - final row: domain legend
    height_ratios = [7.2] + [1.90] * legend_rows + [1.10]
    nrows = len(height_ratios)

    fig_height = 8.5 + 1.75 * legend_rows

    fig = plt.figure(figsize=(10.2, fig_height), constrained_layout=False)

    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=2,
        height_ratios=height_ratios,
        width_ratios=[1.0, 1.0],
        left=0.055,
        right=0.945,
        top=0.965,
        bottom=0.055,
        wspace=0.08,
        hspace=0.06,
    )

    ax = fig.add_subplot(gs[0, :], aspect="equal")
    ax_bottom = fig.add_subplot(gs[-1, :])

    outer_radius = 1.00
    outer_width = 0.28

    inner_radius = 0.72
    inner_width = 0.40

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
            "linewidth": 0.75,
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
            "linewidth": 1.0,
        },
    )

    # White center hole.
    hole_radius = inner_radius - inner_width

    center = plt.Circle(
        (0, 0),
        hole_radius,
        fc="white",
        ec="white",
        zorder=2,
    )
    ax.add_artist(center)

    ax.text(
        0,
        0,
        f"{int(round(total))}\nproblems",
        ha="center",
        va="center",
        fontsize=17.0,
        fontweight="bold",
        color="#333333",
        zorder=6,
    )

    force_label_keys = {
        key for key, count in top_relevant_marker_counts.items() if count > 0
    }

    # Outside labels.
    add_outer_labels(
        ax=ax,
        wedges=outer_wedges,
        labels=outer_labels,
        values=outer_values,
        domains=outer_domains,
        total=total,
        min_label_pct=prevailing_label_min_pct,
        fontsize=14,
        hide_labels=set(),
        show_domains=prevailing_domains,
        force_label_keys=force_label_keys,
        query_label_keys=query_label_keys,
        max_labels_per_domain=max_prevailing_labeled_topics,
        label_min_gap=0.2,
        label_multiline_extra_gap=0.050,
        label_y_low=-1.35,
        label_y_high=1.38,
        label_radius=1.50,
        elbow_radius=1.17,
        wrap_width=16,
    )

    # Black marks inside the outer ring for top-k relevant candidate subdomains.
    add_top_relevant_black_marks(
        ax=ax,
        wedges=outer_wedges,
        labels=outer_labels,
        domains=outer_domains,
        marker_counts=top_relevant_marker_counts,
        ring_inner_radius=outer_radius - outer_width,
        ring_outer_radius=outer_radius,
        max_marks_per_wedge=top_k_relevant,
    )

    # if top_relevant_marker_counts:
    #     ax.text(
    #     0,
    #     -1.49,
    #     f"Black ticks mark subdomains appearing in the top- {top_k_relevant} most relevant candidates.",
    #     ha="center",
    #     va="top",
    #     fontsize=14.0,
    #     color="#333333",
    #     zorder=9,
    #     clip_on=False,
    #     )
    if title:
        prevailing_text = ", ".join(
            DOMAIN_DISPLAY.get(d, d) for d in DOMAIN_ORDER if d in prevailing_domains
        )
        ax.set_title(
            f"{title}\nLine labels: {prevailing_text}",
            fontsize=16,
            fontweight="bold",
            pad=10,
        )

    # Wider chart bounds to make room for larger labels and larger spacing.
    ax.set_xlim(-1.78, 1.78)
    ax.set_ylim(-1.55, 1.45)
    ax.axis("off")

    # Smaller-domain legends.
    for idx, domain in enumerate(legend_domains_ordered):
        row = 1 + idx // 2
        col = idx % 2

        ax_leg = fig.add_subplot(gs[row, col])

        add_topic_legend_axis(
            ax=ax_leg,
            domain=domain,
            outer_labels=outer_labels,
            outer_values=outer_values,
            outer_domains=outer_domains,
            outer_colors=outer_colors,
            total=total,
            marker_counts=top_relevant_marker_counts,
            max_items=small_domain_legend_max_items,
        )

    # Turn off unused legend cells.
    for idx in range(len(legend_domains_ordered), legend_rows * 2):
        row = 1 + idx // 2
        col = idx % 2

        ax_unused = fig.add_subplot(gs[row, col])
        ax_unused.axis("off")

    # Bottom domain legend.
    add_domain_legend_axis(
        ax=ax_bottom,
        domain_labels=inner_labels,
        domain_values=inner_values,
        total=total,
        top_k_relevant=top_k_relevant if top_relevant_marker_counts else None,
    )

    plt.savefig(
        out_file,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    if show:
        plt.show()

    return fig, ax


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets_dataset", required=True)
    parser.add_argument("--candidates_dataset", required=True)
    parser.add_argument("--targ_idx", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=10)

    args = parser.parse_args()

    targets_dataset = args.targets_dataset
    candidates_dataset = args.candidates_dataset
    targ_idx = args.targ_idx
    top_k = args.top_k

    targets = load_dataset(targets_dataset, split="train")
    candidates = load_dataset(candidates_dataset, split="train")

    target = targets[targ_idx]

    list_of_candidates = [candidates[x] for x in target["candidates"]]

    assert len(list_of_candidates) == 150

    query_label_keys = get_query_topic_keys(target)

    top_relevant_examples = get_top_k_candidate_examples(
        target=target,
        candidates_dataset=candidates,
        k=top_k,
    )

    top_relevant_marker_counts = get_top_relevant_topic_marker_counts(
        top_relevant_examples,
    )

    plot_math_dataset_sunburst(
        list_of_candidates,
        out_file=f"piechart_candidates_NT.pdf",
        title=None,
        query_label_keys=query_label_keys,
        top_relevant_marker_counts=top_relevant_marker_counts,
        top_k_relevant=top_k,
        max_topics_per_domain={
            "Algebra": 5,
            "Geometry": 5,
            "Number Theory": 5,
            "Calculus and Analysis": 5,
            "Combinatorics": 5,
        },
        prevailing_domain_threshold=0.50,
        prevailing_label_min_pct=0.0,
        max_prevailing_labeled_topics=10,
        small_domain_legend_max_items=None,
        startangle=90,
        counterclock=False,
        show=True,
    )
