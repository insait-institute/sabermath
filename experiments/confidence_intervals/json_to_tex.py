#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


TASKS = {
    "statement-statement": {
        "filename": "statement-statement.tex",
        "caption": "Statement vs. statement",
        "label": "tab:statement-statement",
    },
    "statement-full": {
        "filename": "statement-full.tex",
        "caption": "Statement vs. full solution",
        "label": "tab:statement-full",
    },
    "full-full": {
        "filename": "full-full.tex",
        "caption": "Full solution vs. full solution",
        "label": "tab:full-full",
    },
}

BRANCHES = [
    "Algebra",
    "Geometry",
    "Number Theory",
    "Combinatorics",
    "Calculus and Analysis",
]

HEADER = r"""
\begin{tabular}{@{}l RRRRRR@{}}
\toprule
        & \multicolumn{1}{c}{\makecell{Overall}}
        & \multicolumn{1}{c}{\makecell{Algebra}}
        & \multicolumn{1}{c}{\makecell{Geometry}}
        & \multicolumn{1}{c}{\makecell{Number\\Theory}}
        & \multicolumn{1}{c}{\makecell{Comb.}}
        & \multicolumn{1}{c}{\makecell[c]{Calc./\\Analysis}}\\
\midrule
""".strip()


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def parse_mean_and_interval(obj):
    """
    Returns:
        mean, lower_error, upper_error

    Cell format later becomes:
        mean^{+upper_error}_{-lower_error}
    """
    if "mean" not in obj:
        raise ValueError(f"Missing required field 'mean' in object: {obj}")

    if "confidence_interval" not in obj:
        raise ValueError(
            f"Missing required field 'confidence_interval' in object: {obj}"
        )

    interval = obj["confidence_interval"]

    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"Invalid confidence interval: {interval}")

    lower, upper = interval
    mean = obj["mean"]

    lower_error = mean - lower
    upper_error = upper - mean

    return mean, lower_error, upper_error


def find_row_name(data, json_path: Path) -> str:
    """
    Prefer an explicit model/method/name/system field if present.
    Otherwise use the JSON filename as the row name.
    """
    for key in ("model", "method", "name", "system"):
        if key in data and isinstance(data[key], str):
            return data[key]

    return json_path.stem


def load_result(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    row_name = find_row_name(data, json_path)
    k = data.get("k")

    task_map = {}

    for task_obj in data.get("tasks", []):
        task_name = task_obj["task"]

        # Ignore unknown tasks unless you add them to TASKS above.
        if task_name not in TASKS:
            continue

        overall = parse_mean_and_interval(task_obj)

        branch_map = {
            branch_obj["branch"]: parse_mean_and_interval(branch_obj)
            for branch_obj in task_obj.get("branches", [])
        }

        task_map[task_name] = {
            "overall": overall,
            "branches": branch_map,
        }

    return {
        "name": row_name,
        "k": k,
        "path": json_path,
        "tasks": task_map,
    }


def format_cell(mean, lower_error, upper_error, is_best=False, digits=3):
    mean_s = f"{mean:.{digits}f}"
    lower_s = f"{lower_error:.{digits}f}"
    upper_s = f"{upper_error:.{digits}f}"

    if is_best:
        return (
            rf"\mathbf{{{mean_s}}}^{{+\mathbf{{{upper_s}}}}}_{{-\mathbf{{{lower_s}}}}}"
        )

    return rf"{mean_s}^{{+{upper_s}}}_{{-{lower_s}}}"


def validate_task_rows(task_name, rows):
    """
    Validate only the rows that actually contain task_name.
    """
    for row in rows:
        if task_name not in row["tasks"]:
            continue

        task_data = row["tasks"][task_name]

        if "overall" not in task_data:
            raise ValueError(f"{row['name']} / {task_name} is missing overall result")

        for branch in BRANCHES:
            if branch not in task_data["branches"]:
                raise ValueError(
                    f"{row['name']} / {task_name} is missing branch {branch}"
                )


def build_table(task_name, rows, k, digits=3, bold_best=True):
    task_info = TASKS[task_name]
    columns = ["overall"] + BRANCHES

    # Keep only rows that contain this task.
    rows = [row for row in rows if task_name in row["tasks"]]

    if not rows:
        raise ValueError(f"No rows found for task {task_name}")

    validate_task_rows(task_name, rows)

    best_by_col = {}

    if bold_best:
        for col in columns:
            values = []

            for row in rows:
                task_data = row["tasks"][task_name]

                if col == "overall":
                    mean, _, _ = task_data["overall"]
                else:
                    mean, _, _ = task_data["branches"][col]

                values.append(mean)

            best_by_col[col] = max(values)

    caption = f"{task_info['caption']} (nDCG@{k} (exponent))"

    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\footnotesize",
        "",
        rf"\caption{{{caption}}}",
        rf"\label{{{task_info['label']}}}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\setlength{\tabcolsep}{3pt}",
        "",
        r"\newcolumntype{R}{>{$}r<{$}}",
        "",
        HEADER,
    ]

    for row in rows:
        task_data = row["tasks"][task_name]
        cells = []

        mean, lower_error, upper_error = task_data["overall"]
        is_best = bold_best and abs(mean - best_by_col["overall"]) < 1e-12

        cells.append(
            format_cell(
                mean,
                lower_error,
                upper_error,
                is_best=is_best,
                digits=digits,
            )
        )

        for branch in BRANCHES:
            mean, lower_error, upper_error = task_data["branches"][branch]
            is_best = bold_best and abs(mean - best_by_col[branch]) < 1e-12

            cells.append(
                format_cell(
                    mean,
                    lower_error,
                    upper_error,
                    is_best=is_best,
                    digits=digits,
                )
            )

        line = f"        {latex_escape(row['name'])} & " + " & ".join(cells) + r" \\"
        lines.append(line)

    lines.extend(
        [
            "",
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\end{table}",
            "",
        ]
    )

    return "\n".join(lines)


def get_present_tasks(rows):
    """
    Return known tasks that appear in at least one JSON file,
    preserving the order from TASKS.
    """
    present = set()

    for row in rows:
        present.update(row["tasks"].keys())

    return [task_name for task_name in TASKS if task_name in present]


def get_k_for_task(task_name, rows):
    """
    Use the first k value among rows that contain this task.
    Warn if the task has inconsistent k values.
    """
    task_rows = [row for row in rows if task_name in row["tasks"]]
    k_values = {row["k"] for row in task_rows}
    k = task_rows[0]["k"]

    if len(k_values) > 1:
        print(
            f"Warning: inconsistent k values for task {task_name}: "
            f"{sorted(k_values)}. Using k={k} in caption.",
            file=sys.stderr,
        )

    return k


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON benchmark files into LaTeX tables."
    )

    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing JSON files.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for .tex files. Defaults to the input folder.",
    )

    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Number of decimal places to print.",
    )

    parser.add_argument(
        "--no-bold",
        action="store_true",
        help="Do not bold the best mean in each column.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Read JSON files recursively.",
    )

    args = parser.parse_args()

    input_dir = args.folder
    output_dir = args.output_dir or input_dir

    if not input_dir.is_dir():
        print(f"Error: not a folder: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.recursive:
        json_files = sorted(input_dir.rglob("*.json"))
    else:
        json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print(f"Error: no JSON files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []

    for json_path in json_files:
        rows.append(load_result(json_path))

    present_tasks = get_present_tasks(rows)

    if not present_tasks:
        print(
            "Error: no known tasks found in the JSON files. "
            f"Known tasks are: {', '.join(TASKS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    for task_name in present_tasks:
        task_rows = [row for row in rows if task_name in row["tasks"]]

        # Sort independently for each table by overall mean, descending.
        sorted_rows = sorted(
            task_rows,
            key=lambda row: row["tasks"][task_name]["overall"][0],
            reverse=True,
        )

        k = get_k_for_task(task_name, sorted_rows)

        tex = build_table(
            task_name=task_name,
            rows=sorted_rows,
            k=k,
            digits=args.digits,
            bold_best=not args.no_bold,
        )

        output_path = output_dir / TASKS[task_name]["filename"]
        output_path.write_text(tex, encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
