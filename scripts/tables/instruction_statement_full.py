#!/usr/bin/env python3
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = ROOT / "results" / "evaluation"
DEFAULT_OUT = ROOT / "results" / "tables"
TASK = "statement-full"
ARMS = ["p0", "p1", "p2", "p3"]
NAME = re.compile(r"^([a-z0-9._-]+?)(?:__(p[0-3]|pm))?(?:__shard(\d+)of(\d+))?$")

TABLE = [
    ("ReasonReranker-Qwen3-32B-Rewrite", "retro-star-32b-rewritten", "RTWC"),
    ("ReasonEmbed-Qwen3-8B-Rewrite", "reason-rewriter-reason-embed-8b", "RTWB"),
    ("ReasonReranker-Qwen3-32B", "retro-star-32b", "RTC"),
    ("ReasonEmbed-Qwen3-8B", "reason-embed-qwen3-8b", "RB"),
    ("Diver-GroupRank-32B", "diver-grouprank-32b", "RTC"),
    ("RaDeR-14B", "rader-14b", "RB"),
    ("RaDeR-7B", "rader-7b", "RB"),
    ("Qwen3-Reranker-4B", "qwen3-reranker-4b", "C"),
    ("Diver-Retriever-4B", "diver-retriever-4b", "RB"),
    ("Qwen3-Reranker-8B", "qwen3-reranker-8b", "C"),
    ("ReasonEmbed-Llama-3.1-8B", "reason-embed-llama-3.1-8b", "RB"),
    ("RaDeR-3B", "rader-3b", "RB"),
    ("Qwen3-Reranker-0.6B", "qwen3-reranker-0.6b", "C"),
    ("SPLADE-Code-8B", "splade-code-8b", "SB"),
    ("Octen-Embedding-8B", "octen-embedding-8b", "B"),
    ("Diver-Retriever-0.6B", "diver-retriever-0.6b", "RB"),
    ("Octen-Embedding-4B", "octen-embedding-4b", "B"),
    ("Gemini-Embedding-2", "gemini-embedding-2", "AB"),
    ("Gemini-Embedding-001", "gemini-embedding-001", "AB"),
    ("Qwen3-Embedding-4B", "qwen3-embedding-4b", "B"),
    ("Qwen3-Embedding-8B", "qwen3-embedding-8b", "B"),
    ("Rank1-32B", "rank1-32b", "RTC"),
    ("Harrier-OSS-v1-27b", "harrier-oss-v1-27b", "B"),
    ("SPLADE-Code-0.6B", "splade-code-0.6b", "SB"),
    ("INF-Retriever-v1-Pro", "inf-retriever-v1-pro", "RB"),
    ("KaLM-Embedding-Gemma3-12B-2511", "kalm-embedding-gemma3-12b-2511", "B"),
    ("LLaMa-Embed-Nemotron-8b", "llama-embed-nemotron-8b", "B"),
    ("Qwen3-Embedding-0.6B", "qwen3-embedding-0.6b", "B"),
    ("Harrier-OSS-v1-0.6b", "harrier-oss-v1-0.6b", "B"),
    ("Jina-Embeddings-v5-Text-Small", "jina-embeddings-v5-text-small", "B"),
    ("Reason-ModernColBERT", "reason-moderncolbert", "RL"),
    ("Text-Embedding-3-Large", "text-embedding-3-large", "AB"),
    ("Rank1-7B", "rank1-7b", "RTC"),
    ("Jina-Embeddings-v5-Text-Nano", "jina-embeddings-v5-text-nano", "B"),
    ("GTE-ModernColBERT", "gte-moderncolbert", "L"),
    ("EmbeddingGemma-300m", "embeddinggemma-300m", "B"),
    ("Text-Embedding-3-Small", "text-embedding-3-small", "AB"),
    ("BGE-m3", "bge-m3", "B"),
    ("ReasonIR-8B", "reasonir-8b", "RB"),
    ("Harrier-OSS-v1-270m", "harrier-oss-v1-270m", "B"),
    ("INF-X-Retriever", "inf-x-retriever", "RWB"),
    ("Multilingual-E5-Large", "multilingual-e5-large", "B"),
    ("Approach Zero", "approach0", "SL"),
    ("BM25", "bm25", "SB"),
    ("Jaccard", "jaccard", "SB"),
    ("TF-IDF", "tf-idf", "SB"),
    ("Rank1-0.5B", "rank1-0.5b", "RTC"),
    ("BERT", "bert-base-uncased", "B"),
    ("RoBERTa", "roberta-base", "B"),
]

EXCLUDED = {
    "approach0": "its segfault skip-list matches an MD5 of the raw query text, which any rewrite defeats",
    "bm25": "no instruction mechanism; the arms it did have were dropped so the lexical rows are treated alike",
    "jaccard": "instruction tokens inflate the query token-set union, moving every score monotonically",
    "tf-idf": "its vocabulary is fitted on documents only and cosine dilutes real query terms",
}


def collect(runs_dir: Path):
    whole, shards = {}, defaultdict(dict)
    for path in sorted(runs_dir.glob("*.json")):
        m = NAME.match(path.stem)
        if not m:
            continue
        arm, shard, nshards = m.group(2) or "p0", m.group(3), m.group(4)
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for key, report in (payload.get("reports") or {}).items():
            values = (report.get("ndcgs_by_task") or {}).get(TASK)
            if not values:
                continue
            scored = [v for v in values if v is not None]
            if not scored:
                continue
            entry = (scored, report, path.name)
            rank = 0 if m.group(2) else 1
            if shard is None:
                prev = whole.get((key, arm))
                if prev is None or rank < prev[0]:
                    whole[(key, arm)] = (rank, entry)
            else:
                shards[(key, arm, int(nshards))].setdefault(int(shard), entry)

    cells, rejected = {}, []

    def verify(key, arm, report, n, source):
        prompt = report.get("prompt") or {}
        if n != 1000:
            rejected.append((key, arm, f"{n}/1000 queries scored", source))
            return False
        if report.get("k") != 10 or report.get("dcg_variant") != "exponent":
            rejected.append((key, arm, "not nDCG@10 with exponential gain", source))
            return False
        if arm != "p0" and prompt.get("key") != arm:
            rejected.append((key, arm, f"file records prompt {prompt.get('key')!r}", source))
            return False
        return True

    for (key, arm), (_rank, (scored, report, source)) in whole.items():
        if verify(key, arm, report, len(scored), source):
            task = next(
                (t for t in report.get("tasks", []) if t["task"] == TASK), None
            )
            score = task["ndcg_at_k"] if task else statistics.fmean(scored)
            cells[(key, arm)] = (score, source)

    for (key, arm, nshards), parts in sorted(shards.items()):
        if (key, arm) in cells or len(parts) != nshards:
            continue
        pooled = [v for i in sorted(parts) for v in parts[i][0]]
        report = parts[min(parts)][1]
        source = f"{key}__{arm}__shard*of{nshards} ({nshards} shards)"
        if verify(key, arm, report, len(pooled), source):
            cells[(key, arm)] = (statistics.fmean(pooled), source)

    return cells, rejected


def rows(cells):
    out = []
    for name, key, code in TABLE:
        base = cells.get((key, "p0"))
        arms = [cells.get((key, a)) for a in ARMS[1:]]
        out.append((name, key, code, base, arms))
    return sorted(out, key=lambda r: -(r[3][0] if r[3] else 0))


def markdown(cells, rejected, ordered):
    md = [
        "<!-- Generated by python scripts/tables/instruction_statement_full.py - do not edit by hand. -->",
        "",
        "# Instruction prompts on SABER-Math (statement-full)",
        "",
        "nDCG@10 over all 1000 queries, canonical protocol. `p0` is the",
        "no-instruction baseline; the parenthesised value is the change from it.",
        "`--` marks a model with no run for that arm. Rows are ordered by `p0`.",
        "",
        "| Model | Type | p0 (none) | p1 | p2 | p3 | best |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, key, code, base, arms in ordered:
        if base is None:
            continue
        cells_txt = [f"{base[0]:.4f}"]
        best, best_name = base[0], "p0"
        for arm, got in zip(ARMS[1:], arms):
            if got is None:
                cells_txt.append("--")
                continue
            cells_txt.append(f"{got[0]:.4f} ({got[0] - base[0]:+.4f})")
            if got[0] > best:
                best, best_name = got[0], arm
        md.append(f"| {name} | {code} | " + " | ".join(cells_txt) + f" | {best_name} |")

    md += ["", "## Rows with no instruction arms", ""]
    for key, why in sorted(EXCLUDED.items()):
        name = next(n for n, k, _ in TABLE if k == key)
        md.append(f"- **{name}** - {why}.")

    if rejected:
        md += ["", "## Incomplete or unverifiable runs (excluded from the table)", ""]
        for key, arm, why, source in sorted(rejected):
            md.append(f"- `{key}` {arm}: {why} (`{source}`)")

    extra = sorted({k for k, _ in cells} - {k for _, k, _ in TABLE})
    if extra:
        md += [
            "",
            "## Also in runs/, not in the paper's table",
            "",
            "Ablation arms and size siblings that were run but are not rows of the",
            "results table: " + ", ".join(f"`{k}`" for k in extra) + ".",
        ]
    md.append("")
    return "\n".join(md)


def latex(ordered):
    tex = [
        "% !TEX root = ../main.tex",
        "% Generated by python scripts/tables/instruction_statement_full.py - do not edit by hand.",
        "\\begin{table*}[!thbp]",
        "    \\centering",
        "    \\footnotesize",
        "    \\caption{Effect of a task instruction on \\bench{} (statement-full), "
        "nDCG@10 over all 1000 queries. \\texttt{p0} is the no-instruction "
        "baseline and \\texttt{p1}--\\texttt{p3} prepend one prompt to the query; "
        "``--'' marks a model with no run for that arm. The four lexical and "
        "symbolic rows have no instruction mechanism and are reported without arms.}",
        "    \\label{tab:instructions-statement-full}",
        "    \\renewcommand{\\arraystretch}{1.05}",
        "    \\setlength{\\tabcolsep}{4pt}",
        "    \\newcolumntype{R}{>{$}r<{$}}",
        "    \\resizebox{0.72\\linewidth}{!}{",
        "        \\begin{tabular}{@{}l c RRRR@{}}",
        "            \\toprule",
        "            & \\multicolumn{1}{c}{Type} & \\multicolumn{1}{c}{No instr.}"
        " & \\multicolumn{1}{c}{p1} & \\multicolumn{1}{c}{p2}"
        " & \\multicolumn{1}{c}{p3} \\\\",
        "            \\midrule",
    ]
    for name, key, code, base, arms in ordered:
        if base is None:
            continue
        best = max([base[0]] + [g[0] for g in arms if g])
        cells = [f"{base[0]:.3f}" if base[0] < best else f"\\mathbf{{{base[0]:.3f}}}"]
        for got in arms:
            if got is None:
                cells.append("\\text{---}")
            elif got[0] == best:
                cells.append(f"\\mathbf{{{got[0]:.3f}}}")
            else:
                cells.append(f"{got[0]:.3f}")
        safe = name.replace("&", "\\&")
        tex.append(f"            {safe} & {code} & " + " & ".join(cells) + " \\\\")
    tex += [
        "            \\bottomrule",
        "        \\end{tabular}",
        "    }",
        "\\end{table*}",
        "",
    ]
    return "\n".join(tex)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    cells, rejected = collect(args.results_dir)
    ordered = rows(cells)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "RESULTS_instructions_statement_full.md").write_text(
        markdown(cells, rejected, ordered)
    )
    (args.out_dir / "statement_full_instructions.tex").write_text(latex(ordered))
    shown = sum(1 for _, _, _, base, arms in ordered if base for g in arms if g)
    print(f"[+] {len(cells)} verified cells; {len(ordered)} table rows, {shown} arm cells")
    if rejected:
        print(f"[!] {len(rejected)} run(s) excluded:")
        for key, arm, why, source in sorted(rejected):
            print(f"      {key} {arm}: {why}")
    print(f"[+] wrote 2 files into {args.out_dir}")


if __name__ == "__main__":
    main()
