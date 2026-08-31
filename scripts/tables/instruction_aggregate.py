#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json
from pathlib import Path

from sabermath import registry as rr
from sabermath.results import parse_result_name

TASK_ORDER = ["statement-statement", "statement-full", "full-full"]


def load_cells(scan: Path, protocol_tag: str, subset: str) -> dict:
    cells: dict[tuple[str, str], dict] = {}
    for path in sorted(scan.glob("*.json")):
        parsed = parse_result_name(path.stem)
        if parsed is None:
            continue
        if parsed["protocol_tag"] != protocol_tag or parsed["subset"] != subset:
            continue
        if parsed["part"] is not None or parsed["shard"] is not None:
            continue

        payload = json.loads(path.read_text())
        reports = payload.get("reports", {})
        if not reports:
            continue
        model_key = next(iter(reports))
        report = reports[model_key]
        if "error" in report and "tasks" not in report:
            continue

        by_task = {}
        for task in report.get("tasks", []):
            ndcgs = report.get("ndcgs_by_task", {}).get(task["task"], [])
            done = len([v for v in ndcgs if v is not None])
            by_task[task["task"]] = {
                "ndcg_at_k": task["ndcg_at_k"],
                "n_done": done,
                "n_total": len(ndcgs),
            }

        cells[(model_key, parsed["instruction_key"])] = {
            "tasks": by_task,
            "prompt": report.get("prompt", {}),
            "file": path.name,
        }
    return cells


def format_cell(cell: dict | None, task: str) -> str:
    if cell is None:
        return "-"
    entry = cell["tasks"].get(task)
    if entry is None:
        return "-"
    text = f"{entry['ndcg_at_k']:.4f}"
    if entry["n_total"] and entry["n_done"] < entry["n_total"]:
        text += f" ({entry['n_done']}/{entry['n_total']})"
    return text


def delta_cell(cell: dict | None, base: dict | None, task: str) -> str:
    if cell is None or base is None:
        return ""
    entry = cell["tasks"].get(task)
    base_entry = base["tasks"].get(task)
    if entry is None or base_entry is None:
        return ""
    if entry["n_done"] != base_entry["n_done"]:
        return " (n differs)"
    return f" ({entry['ndcg_at_k'] - base_entry['ndcg_at_k']:+.4f})"


def build_block(models, cells, prompt_keys, tasks, with_deltas) -> list[list[str]]:
    header = ["model"]
    for task in tasks:
        for key in prompt_keys:
            header.append(f"{task}/{key}")
    rows = [header]

    for model_key in models:
        row = [model_key]
        base = cells.get((model_key, "p0"))
        for task in tasks:
            for key in prompt_keys:
                cell = cells.get((model_key, key))
                text = format_cell(cell, task)
                if with_deltas and key != "p0" and text != "-":
                    text += delta_cell(cell, base, task)
                row.append(text)
        rows.append(row)
    return rows


def render_markdown(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for i, row in enumerate(rows):
        lines.append(
            "| " + " | ".join(c.ljust(widths[j]) for j, c in enumerate(row)) + " |"
        )
        if i == 0:
            lines.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, default=Path("results/evaluation"))
    parser.add_argument(
        "--protocol-tag",
        type=str,
        default="",
        help="Which protocol's files to aggregate: '' (current protocol, the "
        "default), 'legacy', 'nl1', 'nl2', 'legacy-nl1'.",
    )
    parser.add_argument("--subset", type=str, default="")
    parser.add_argument(
        "--prompts", nargs="+", default=["p0", "p1", "p2", "p3"]
    )
    parser.add_argument("--tasks", nargs="+", default=TASK_ORDER)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    cells = load_cells(args.scan, args.protocol_tag, args.subset)
    if not cells:
        raise SystemExit(
            f"No result files matching protocol_tag={args.protocol_tag!r} "
            f"subset={args.subset!r} under {args.scan}"
        )

    models = sorted({model for model, _ in cells})
    instructable = [m for m in models if m not in rr.INSTRUCTION_CONTROL_MODELS]
    controls = [m for m in models if m in rr.INSTRUCTION_CONTROL_MODELS]

    blocks = {
        "instructable": build_block(
            instructable, cells, args.prompts, args.tasks, True
        ),
        "control": build_block(controls, cells, args.prompts, args.tasks, True),
    }

    out = []
    protocol_label = args.protocol_tag or "canonical"
    out.append(f"# Instruction ablation ({protocol_label})\n")
    out.append(f"Source: `{args.scan}`\n")
    out.append("\n## Instructable models\n")
    out.append(render_markdown(blocks["instructable"]))
    out.append("\n\n## Controls (no vendor instruction mechanism)\n")
    out.append(render_markdown(blocks["control"]))
    out.append("\n\nControl reasons:\n")
    for model_key in controls:
        reason = rr.INSTRUCTION_CONTROL_REASONS.get(model_key, "")
        out.append(f"- `{model_key}` - {reason}\n")
    text = "".join(out)

    print(text)

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text)
        print(f"[+] Wrote {args.markdown}")

    if args.json is not None:
        payload = {
            "protocol_tag": args.protocol_tag,
            "subset": args.subset,
            "instructable": instructable,
            "control": controls,
            "cells": {
                f"{model}__{key}": cell["tasks"]
                for (model, key), cell in sorted(cells.items())
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"[+] Wrote {args.json}")


if __name__ == "__main__":
    main()
