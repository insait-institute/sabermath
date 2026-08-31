import argparse
import json
from pathlib import Path

TASK_ORDER = [
    "spoken-in-context-expression",
    "expression-spoken-in-context",
    "expression-in-context-spoken",
    "expression-spoken",
    "spoken-expression",
    "spoken-expression-in-context",
]

TASK_LABELS = {
    "spoken-in-context-expression": r"Spoken ctx $\to$ Expr",
    "expression-spoken-in-context": r"Expr $\to$ Spoken ctx",
    "expression-in-context-spoken": r"Expr ctx $\to$ Spoken",
    "expression-spoken": r"Expr $\to$ Spoken",
    "spoken-expression": r"Spoken $\to$ Expr",
    "spoken-expression-in-context": r"Spoken $\to$ Expr ctx",
}


def short_model_name(model: str) -> str:
    return str(model).split("/")[-1]


def escape_latex(text: str) -> str:
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def avg(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value, best=None):
    if value is None:
        return "--"

    text = f"{value:.3f}"

    if best is not None and abs(value - best) < 1e-12:
        return rf"\textbf{{{text}}}"

    return text


def load_rows(directory: Path):
    rows = []
    k_values = set()

    for file in sorted(directory.glob("*.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        k_values.add(data.get("k"))

        task_map = {task["task"]: task for task in data.get("tasks", [])}

        row = {
            "model": short_model_name(data.get("model", file.stem)),
            "source_file": file.name,
        }

        for task_name in TASK_ORDER:
            task = task_map.get(task_name, {})
            row[f"{task_name}:mrr"] = task.get("mrr")
            row[f"{task_name}:recall"] = task.get("recall_at_k")

        row["avg_mrr"] = avg(row[f"{t}:mrr"] for t in TASK_ORDER)
        row["avg_recall"] = avg(row[f"{t}:recall"] for t in TASK_ORDER)

        rows.append(row)

    if not rows:
        raise SystemExit(f"No .json files found in {directory}")

    return rows, k_values


def make_table(rows, k_values):
    recall_label = f"R@{next(iter(k_values))}" if len(k_values) == 1 else "R@k"

    numeric_cols = ["avg_mrr", "avg_recall"]

    for task_name in TASK_ORDER:
        numeric_cols += [f"{task_name}:mrr", f"{task_name}:recall"]

    best = {}

    for col in numeric_cols:
        vals = [row[col] for row in rows if row[col] is not None]
        best[col] = max(vals) if vals else None

    rows = sorted(
        rows,
        key=lambda row: (
            row["avg_mrr"] if row["avg_mrr"] is not None else -1,
            row["avg_recall"] if row["avg_recall"] is not None else -1,
        ),
        reverse=True,
    )

    groups = ["Avg"] + [TASK_LABELS[t] for t in TASK_ORDER]

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(
        rf"\caption{{MRR and Recall@{next(iter(k_values)) if len(k_values) == 1 else 'k'} across retrieval tasks}}"
    )
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "r" * (2 * len(groups)) + r"}")
    lines.append(r"\toprule")

    header_1 = ["Model"] + [rf"\multicolumn{{2}}{{c}}{{{g}}}" for g in groups]
    lines.append(" & ".join(header_1) + r" \\")

    cmidrules = []
    start = 2

    for _ in groups:
        cmidrules.append(rf"\cmidrule(lr){{{start}-{start + 1}}}")
        start += 2

    lines.append(" ".join(cmidrules))

    header_2 = [""] + sum(([r"MRR", recall_label] for _ in groups), [])
    lines.append(" & ".join(header_2) + r" \\")
    lines.append(r"\midrule")

    for row in rows:
        values = [
            fmt(row["avg_mrr"], best["avg_mrr"]),
            fmt(row["avg_recall"], best["avg_recall"]),
        ]

        for task_name in TASK_ORDER:
            values.append(fmt(row[f"{task_name}:mrr"], best[f"{task_name}:mrr"]))
            values.append(fmt(row[f"{task_name}:recall"], best[f"{task_name}:recall"]))

        lines.append(escape_latex(row["model"]) + " & " + " & ".join(values) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def build_document(table_tex):
    return "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage{booktabs}",
            r"\usepackage{graphicx}",
            r"\usepackage[margin=1in]{geometry}",
            r"\begin{document}",
            r"\small",
            table_tex,
            r"\end{document}",
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory containing the JSON result files",
    )

    parser.add_argument(
        "--fragment",
        action="store_true",
        help="Write only the table environment, not a full LaTeX document",
    )

    args = parser.parse_args()

    directory = Path(args.directory)

    rows, k_values = load_rows(directory)
    table_tex = make_table(rows, k_values)

    output = table_tex if args.fragment else build_document(table_tex)

    out_file = directory / "results_table.tex"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"✓ Generated {out_file}")


if __name__ == "__main__":
    main()
