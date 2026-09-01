#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from sabermath import registry as rr
from sabermath.results import parse_result_name
from sabermath.tables import BRANCH_TO_DOMAIN, DOMAINS, MODEL_INFO


TASKS = ["statement-statement", "statement-full", "full-full"]
PROMPT_LABELS = {
    "p0": "p0 (none)",
    "p1": "p1",
    "p2": "p2",
    "p3": "p3",
    "pm": "pm (repo math)",
}

def load(scan: Path) -> dict:
    cells = {}
    for path in sorted(scan.glob("*.json")):
        parsed = parse_result_name(path.stem)
        if parsed is None or parsed["part"] or parsed["shard"] or parsed["subset"]:
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        reports = payload.get("reports", {})
        if not reports:
            continue
        model_key = next(iter(reports))
        report = reports[model_key]
        if "tasks" not in report:
            continue
        by_task = {}
        for task in report["tasks"]:
            scored = report.get("ndcgs_by_task", {}).get(task["task"], [])
            done = len([v for v in scored if v is not None])
            by_task[task["task"]] = (task["ndcg_at_k"], done, len(scored))
            for branch in task.get("branches", []):
                domain = BRANCH_TO_DOMAIN.get(branch["branch"].lower())
                if domain is None:
                    continue
                by_task[(task["task"], domain)] = (
                    branch["ndcg_at_k"],
                    done,
                    len(scored),
                )
        cells[(model_key, parsed["instruction_key"])] = by_task
    return cells

def mean_delta(cells, model_keys, prompt, slot, base_prompt="p0"):
    deltas = []
    for key in model_keys:
        entry = cells.get((key, prompt), {}).get(slot)
        base = cells.get((key, base_prompt), {}).get(slot)
        if entry is None or base is None or entry[1] != base[1]:
            continue
        deltas.append(entry[0] - base[0])
    if not deltas:
        return None, 0, 0
    return sum(deltas) / len(deltas), len(deltas), sum(1 for d in deltas if d > 0)

def summary_block(cells, groups, prompts, slot, label) -> list[str]:
    header = [label] + [PROMPT_LABELS.get(p, p) for p in prompts if p != "p0"] + ["n"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for name, keys in groups:
        row, count = [name], 0
        for prompt in prompts:
            if prompt == "p0":
                continue
            delta, n, improved = mean_delta(cells, keys, prompt, slot)
            count = max(count, n)
            row.append("-" if delta is None else f"{delta:+.4f} ({improved}/{n})")
        row.append(str(count))
        lines.append("| " + " | ".join(row) + " |")
    return lines

def cell_text(cells, model_key, prompt, task, base_prompt="p0") -> str:
    entry = cells.get((model_key, prompt), {}).get(task)
    if entry is None:
        return "-"
    score, done, total = entry
    text = f"{score:.4f}"
    if total and done < total:
        return f"{text} ({done}/{total})"
    if prompt != base_prompt:
        base = cells.get((model_key, base_prompt), {}).get(task)
        if base is not None and base[1] == done:
            text += f" ({score - base[0]:+.4f})"
    return text

def block(cells, model_keys, prompts, task) -> list[str]:
    header = ["Model"] + [PROMPT_LABELS.get(p, p) for p in prompts] + ["best"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for key in model_keys:
        name = MODEL_INFO.get(key, (key, ""))[0]
        row = [name] + [cell_text(cells, key, p, task) for p in prompts]
        scored = {
            p: cells.get((key, p), {}).get(task)
            for p in prompts
            if cells.get((key, p), {}).get(task) is not None
        }
        if scored:
            best = max(scored, key=lambda p: scored[p][0])
            row.append(PROMPT_LABELS.get(best, best).split(" ")[0])
        else:
            row.append("-")
        lines.append("| " + " | ".join(row) + " |")
    return lines

def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--prompts", nargs="+", default=["p0", "p1", "p2", "p3"])
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument(
        "--out", type=Path, default=Path("results/tables/RESULTS_instructions.md")
    )
    args = parser.parse_args(argv)

    cells = load(args.scan)
    if not cells:
        raise SystemExit(f"No completed runs under {args.scan}")

    models = sorted({key for key, _ in cells})
    instructable = models

    def order(keys, task):
        return sorted(
            keys,
            key=lambda k: -(cells.get((k, "p0"), {}).get(task, (0,))[0]),
        )

    body = [
        "<!-- Generated by scripts/report_experiments.py. Deltas are against each",
        "     model's own p0. The prompt texts are in src/sabermath/instructions.py;",
        "     how each family receives them is in src/sabermath/registry.py. -->",
        "",
        "# SABER-Math — Instruction-prompt ablation",
        "",
        f"Source: `{args.scan}`",
        "",
        "All numbers are nDCG@10. `p0` is the no-instruction baseline; the",
        "parenthesised value is the change from it.",
        "",
    ]

    for task in args.tasks:
        body += [f"## {task}", ""]
        body += block(cells, order(instructable, task), args.prompts, task)
        body += [""]

    main_task = "statement-full"

    # ---- summary: how much each prompt moves each model category ----
    cats = {}
    for key in instructable:
        cats.setdefault(MODEL_INFO.get(key, (key, "?"))[1], []).append(key)
    cat_groups = [(c, cats[c]) for c in sorted(cats)]
    cat_groups.append(("ALL instructable", instructable))

    body += [
        f"## Summary by model category ({main_task})",
        "",
        "Mean nDCG@10 change against each model's own p0, over the models in",
        "that category. `(k/n)` is how many of the n compared models improved.",
        "Instructable models only — the control block is summarised separately",
        "below, and a flat control summary is the evidence that movement here",
        "is instruction following rather than noise.",
        "",
    ]
    body += summary_block(cells, cat_groups, args.prompts, main_task, "Category")
    body += [""]


    # ---- summary: per domain ----
    body += [
        f"## Summary by domain ({main_task})",
        "",
        "Same statistic, grouped by problem domain instead of model category.",
        "A prompt that helps everywhere shows a uniform column; one that only",
        "helps a subject shows up here and nowhere else.",
        "",
    ]
    domain_groups = [
        (domain.title(), instructable) for domain in DOMAINS
    ]
    rows = ["| Domain | " + " | ".join(
        PROMPT_LABELS.get(p, p) for p in args.prompts if p != "p0"
    ) + " | n |", "|" + "|".join("---" for _ in range(len(
        [p for p in args.prompts if p != "p0"]) + 2)) + "|"]
    for domain in DOMAINS:
        row, count = [domain.title()], 0
        for prompt in args.prompts:
            if prompt == "p0":
                continue
            delta, n, improved = mean_delta(
                cells, instructable, prompt, (main_task, domain)
            )
            count = max(count, n)
            row.append("-" if delta is None else f"{delta:+.4f} ({improved}/{n})")
        row.append(str(count))
        rows.append("| " + " | ".join(row) + " |")
    body += rows + [""]

    # ---- per-domain detail, main task only ----
    body += [
        f"## Per-domain detail ({main_task})",
        "",
        "One table per domain, nDCG@10 with the change against that model's",
        "own p0 in the same domain. Restricted to the main setting: the same",
        "breakdown for all three tasks would be 15 tables of 49 rows, and the",
        "other two are reconstructible from the result JSONs the same way.",
        "",
    ]
    for domain in DOMAINS:
        slot = (main_task, domain)
        body += [f"### {domain.title()}", ""]
        body += block(cells, order(instructable, main_task), args.prompts, slot)
        body += [""]

    # ---- pm instruction, where it exists ----
    pm_models = sorted({k for (k, prompt) in cells if prompt == "pm"})
    if pm_models:
        body += [
            "## The `pm` instruction (Qwen3-Reranker only)",
            "",
            "`pm` is this repo's own math-specific instruction, the one the",
            "PUBLISHED Qwen3-Reranker rows were produced with. It is reported",
            "separately because p0 for this family is the vendor's model-card",
            "default rather than an empty slot — that family's `<Instruct>`",
            "field cannot be left empty — so `pm` is an ablation instruction, not a",
            "baseline. See docs/experiment-instructions.md.",
            "",
        ]
        for task in args.tasks:
            body += [f"**{task}**", ""]
            body += block(cells, pm_models, ["p0", "pm"], task)
            body += [""]

    missing = sorted(set(MODEL_INFO) - set(models))
    if missing:
        body += [
            "## Not yet covered",
            "",
            f"{len(missing)} model(s) in the paper's table have no completed "
            "cell in this sweep, so they appear in no table above:",
            "",
        ]
        for key in missing:
            body.append(f"- `{key}`")
        body.append("")


    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(body))
    print(
        f"[+] {len(models)} models -> {args.out}"
    )

if __name__ == "__main__":
    main()
