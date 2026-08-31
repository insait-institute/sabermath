#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json
from pathlib import Path

from sabermath.tables import MODEL_INFO

TOP_K = [1, 2, 4, 8, 16, 32, 64, 128]

# saberniki2/dedup_results.txt, verbatim - the rows behind the rebuttal table.
# (avg_rank, median_rank, top-1 .. top-128 as percentages)
PUBLISHED = {
    "reason-embed-qwen3-8b": (3.095, 1.0, [67.5, 80.0, 89.5, 93.8, 96.8, 98.8, 99.5, 100.0]),
    "gemini-embedding-2": (5.365, 2.0, [46.4, 62.5, 73.7, 82.9, 91.9, 97.2, 99.7, 99.9]),
    "octen-embedding-8b": (6.848, 2.0, [44.1, 57.6, 68.7, 79.9, 88.4, 95.9, 99.0, 100.0]),
    "qwen3-embedding-4b": (9.529, 3.0, [32.6, 45.1, 57.6, 71.3, 83.3, 92.4, 98.4, 99.9]),
    "qwen3-embedding-8b": (15.814, 6.0, [26.4, 34.9, 45.4, 58.5, 70.5, 83.7, 94.1, 99.8]),
    "bm25": (24.086, 11.0, [20.1, 26.8, 36.9, 46.1, 58.5, 73.1, 87.6, 99.9]),
    "tf-idf": (25.918, 10.5, [20.4, 27.6, 36.5, 46.6, 58.8, 70.1, 85.6, 99.2]),
    "embeddinggemma-300m": (26.516, 11.0, [17.8, 26.4, 34.5, 44.9, 58.4, 69.5, 85.2, 99.4]),
    "llama-embed-nemotron-8b": (26.920, 12.0, [17.8, 25.3, 33.8, 42.9, 56.2, 69.1, 84.5, 99.5]),
}

# Why a published row is not directly comparable to ours. Absent = comparable.
PUBLISHED_CONFIG_DIFF = {
    "bm25": "theirs tokenizes `text.lower().split()` (our `bm25-no-tok`); "
            "ours uses the Approach Zero tokenizer",
    "tf-idf": "theirs uses word 1-2 grams + sublinear TF; ours uses "
              "Approach Zero unigrams with raw TF",
}
# Models whose published row was encoded WITHOUT the vendor input envelope,
# because saberniki2/embed_duplicates.py has no prefix parameter at all. Our
# canonical runs apply the envelope, so the rows differ by that alone; the
# __legacy file, where present, reproduces the published encoding.
ENVELOPED = {"embeddinggemma-300m"}

# Models that will never appear here, so they are never reported as pending.
DEDUP_EXCLUDED: dict[str, str] = {}


def load(directory: Path) -> dict:
    out = {}
    for path in sorted(directory.glob("*.json")):
        stem = path.stem
        if "__n" in stem or "smoke" in stem or "selfmatch" in stem:
            continue
        if "candidates-union" in stem:
            continue
        payload = json.loads(path.read_text())
        if "insert_per_query" not in payload:
            continue
        key = payload.get("model") or stem.split("__")[0]
        out[(key, payload.get("regime"), payload.get("protocol", "canonical"))] = payload
    return out


def ndcg_lookup(model_key: str, task: str) -> float | None:
    for directory in ("results/evaluation", "results/evaluation"):
        path = Path(directory) / f"{model_key}__p0.json"
        if not path.exists():
            continue
        reports = json.loads(path.read_text()).get("reports", {})
        if not reports:
            continue
        report = reports[next(iter(reports))]
        for entry in report.get("tasks", []):
            if entry["task"] == task:
                return entry["ndcg_at_k"]
    return None


def render(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(c.ljust(widths[j]) for j, c in enumerate(row)) + " |")
        if i == 0:
            lines.append("|" + "|".join("-" * (w + 2) for w in widths))
            lines[-1] += "|"
    return "\n".join(lines)


def regime_table(cells: dict, regime: str) -> tuple[str, list[str]]:
    header = ["Model", "SABER-Math", "Avg. Rank", "Med. Rank"]
    header += [f"Top-{k}" for k in TOP_K]
    rows = [header]

    entries = []
    for (model_key, cell_regime, protocol), payload in cells.items():
        if cell_regime != regime or protocol != "canonical":
            continue
        entries.append((model_key, payload))
    # Median ties heavily at 1.0, so break ties on the mean - the ordering
    # is then total and stable across regenerations.
    entries.sort(
        key=lambda e: (
            e[1]["insert_per_query"]["median_rank"],
            e[1]["insert_per_query"]["avg_rank"],
        )
    )

    for model_key, payload in entries:
        stats = payload["insert_per_query"]
        name = MODEL_INFO.get(model_key, (model_key, ""))[0]
        ndcg = ndcg_lookup(model_key, "statement-statement")
        row = [
            name,
            f"{ndcg:.3f}" if ndcg is not None else "-",
            f"{stats['avg_rank']:.2f}",
            f"{stats['median_rank']:.1f}",
        ]
        row += [f"{stats[f'top_{k}'] * 100:.1f}%" for k in TOP_K]
        rows.append(row)

    covered = {model_key for model_key, _ in entries}
    return render(rows), sorted(covered)


def reproduction_table(cells: dict) -> str:
    header = ["Model", "Published", "Ours", "Protocol", "Delta (med.)", "Note"]
    rows = [header]
    for model_key, (avg, median, tops) in sorted(
        PUBLISHED.items(), key=lambda kv: kv[1][0]
    ):
        legacy = cells.get((model_key, "per-query-candidates", "legacy"))
        canonical = cells.get((model_key, "per-query-candidates", "canonical"))
        payload = legacy or canonical
        if payload is None:
            rows.append([
                MODEL_INFO.get(model_key, (model_key, ""))[0],
                f"{avg:.2f} / {median:.1f}", "-", "-", "-", "not run yet",
            ])
            continue
        stats = payload["insert_per_query"]
        note = PUBLISHED_CONFIG_DIFF.get(model_key, "")
        if not note and model_key in ENVELOPED:
            note = (
                "published row is prompt-free; our canonical row applies the "
                "vendor envelope"
                if legacy is not None
                else "our row applies a vendor envelope the published one could not"
            )
        rows.append([
            MODEL_INFO.get(model_key, (model_key, ""))[0],
            f"{avg:.2f} / {median:.1f}",
            f"{stats['avg_rank']:.2f} / {stats['median_rank']:.1f}",
            "legacy" if legacy is not None else "canonical",
            f"{stats['median_rank'] - median:+.1f}",
            note or "comparable",
        ])
    return render(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=Path, default=Path("results/dedup"))
    parser.add_argument("--out", type=Path, default=Path("results/tables/RESULTS_dedup.md"))
    args = parser.parse_args(argv)

    cells = load(args.scan)
    if not cells:
        raise SystemExit(f"No dedup results under {args.scan}")

    per_query, per_query_models = regime_table(cells, "per-query-candidates")
    all_docs, all_docs_models = regime_table(cells, "all-documents")

    out = [
        "# Deduplication: where a rephrased copy of the query ranks\n\n",
        f"Source: `{args.scan}`. Generated by `scripts/report_experiments.py`.\n\n",
        "For each of the 1000 queries, an LLM-rephrased copy of that query's\n",
        "own problem is inserted into a corpus and the corpus is ranked by\n",
        "similarity to the original query statement. The reported rank is the\n",
        "position of the inserted copy. **Lower is better.**\n\n",
        "The `SABER-Math` column is **statement-statement** nDCG@10, matching\n",
        "the task setting the dedup vectors are built in. The rebuttal table\n",
        "quoted statement-full beside these numbers, which compares two\n",
        "different settings - see docs/experiment-dedup.md.\n\n",
        "---\n\n## 1. Per-query candidates (the published protocol)\n\n",
        "The copy competes against that query's own 150 candidates, ranked\n",
        "among 151. Every model can run this regime.\n\n",
        per_query,
        "\n\n---\n\n## 2. All documents\n\n",
        "The copy competes against all 71,117 documents. A much harder\n",
        "question with no published reference. Bi-encoders and lexical models\n",
        "only: a pair scorer would need 71,117 forward passes per query.\n\n",
        all_docs,
        "\n\n---\n\n## 3. Reproduction of the published rows\n\n",
        "The nine rows in `saberniki2/dedup_results.txt`, against ours.\n",
        "`avg / median`. A row is only comparable where the configuration\n",
        "matches; the Note column says when it does not.\n\n",
        reproduction_table(cells),
        "\n\n---\n\n## Notes\n\n",
        "- **Self-match exclusion is a no-op in regime 1.** 92.8% of query\n",
        "  problems appear verbatim in the 71,117-document corpus, so excluding\n",
        "  each query's own original matters a great deal in regime 2. Measured,\n",
        "  **0 of those 928 self-documents fall inside their own query's\n",
        "  150-candidate list**, so regime 1 is unaffected either way.\n",
        "- **The published open-weights rows are prompt-free.**\n",
        "  `saberniki2/embed_duplicates.py` embeds raw text and has no prefix,\n",
        "  prompt or task-type parameter at all; `make_cache.py`'s prefixes\n",
        "  default to empty. Our canonical runs apply each model's vendor input\n",
        "  envelope. On `embeddinggemma-300m`, isolated end-to-end, that is the\n",
        "  entire difference: prompt-free gives avg 26.51 / median 11.0 against\n",
        "  the published 26.52 / 11.0, while the envelope gives 27.86 / 13.0.\n",
        "- **What this experiment measures.** The rephrasings preserve the\n",
        "  mathematics and replace the prose: over all 1000 pairs, Jaccard\n",
        "  between a target and its own rephrase is **0.735 on Approach Zero\n",
        "  math tokens but 0.191 on prose words**. Deduplication here is largely\n",
        "  a notation-matching task, which is why BM25 places far above its\n",
        "  retrieval score on it.\n",
        "- `avg_rank` is reported for continuity with the published rows but is\n",
        "  not the statistic to read: the distribution has a long tail driven by\n",
        "  degenerate query texts, so a handful of queries move the mean by an\n",
        "  order of magnitude. Median and the top-k coverage are robust.\n",
    ]

    pending = sorted(set(MODEL_INFO) - set(per_query_models) - set(DEDUP_EXCLUDED))
    if pending:
        out.append(
            f"\n### Still running ({len(pending)} of "
            f"{len(MODEL_INFO) - len(DEDUP_EXCLUDED)})\n\n"
        )
        out.append(", ".join(f"`{m}`" for m in pending) + "\n")
    if DEDUP_EXCLUDED:
        out.append("\n### Excluded by design\n\n")
        for model_key, reason in sorted(DEDUP_EXCLUDED.items()):
            out.append(f"- `{model_key}` - {reason}\n")

    text = "".join(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(
        f"[+] {args.out}: {len(per_query_models)} models in per-query-candidates, "
        f"{len(all_docs_models)} in all-documents"
    )
    if pending:
        print(f"[~] {len(pending)} model(s) still running: {', '.join(pending)}")


if __name__ == "__main__":
    main()
