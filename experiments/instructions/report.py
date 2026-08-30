#!/usr/bin/env python3
"""Instruction-ablation tables for the statement-full task.

    python experiments/instructions/report.py

Reads the raw run files in runs/ and writes RESULTS.md and
statement_full_instructions.tex beside them. Everything the tables show is
recomputed from those files on every run, so the committed tables can never
drift from the committed data.

WHAT A CELL IS. One (model, prompt) run of the whole 1000-query benchmark
under the canonical protocol, reported as nDCG@10 on statement-full. p0 is
the no-instruction baseline and p1-p3 prepend one prompt to the query (the
texts are in src/sabermath/instructions.py). Deltas are against each model's
own p0, never across models.

THREE RULES THIS FILE ENFORCES, each of which was a real bug first:

  1. SHARDED RUNS ARE POOLED, NOT SKIPPED. The two ReasonReranker rows and
     Diver-GroupRank-32B were executed as disjoint query shards
     (__shardIofN), because a 32B generative reranker does not finish 1000
     queries inside a wall-clock limit. A scan that skips any filename
     containing "shard" silently reports those rows as having no instruction
     arms at all - they have complete ones.

  2. A CELL IS SHOWN ONLY IF ALL 1000 QUERIES SCORED. Interrupted runs leave
     a result file whose ndcgs_by_task list is padded with nulls. Averaging
     over the non-null entries produces a number that is not comparable to
     the full runs on its own row, so partial cells are dropped and listed
     under "Incomplete" instead.

  3. PROVENANCE IS CHECKED, NOT ASSUMED. Every cell must carry the prompt key
     its filename claims, k=10 and the exponential gain. A file that does not
     is excluded and reported.

WHICH RUNS ARE HERE. runs/ holds the canonical instruction sweep
(results/instructions_v2 on the cluster) plus the files that back cells that
sweep does not contain: the sharded ReasonReranker arms and Diver-GroupRank's
p0, which live with the main experiment. runs/.provenance.json records where
each file came from. The files carry all three tasks, so
statement-statement and full-full tables can be added here without re-running
anything.
"""

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TASK = "statement-full"
ARMS = ["p0", "p1", "p2", "p3"]
NAME = re.compile(r"^([a-z0-9._-]+?)(?:__(p[0-3]|pm))?(?:__shard(\d+)of(\d+))?$")

# The 49 rows of the paper's confidence-interval table, minus its three
# -no-tok lexical variants: (display name, run key, Type code).
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

# Models the harness refuses to run instructed, with its own stated reason
# (scripts/run_rerankers.py, INSTRUCTION_EXCLUDED). None of the four has an
# instruction mechanism: a prompt could only reach them as extra query terms.
EXCLUDED = {
    "approach0": "its segfault skip-list matches an MD5 of the raw query text, which any rewrite defeats",
    "bm25": "no instruction mechanism; the arms it did have were dropped so the lexical rows are treated alike",
    "jaccard": "instruction tokens inflate the query token-set union, moving every score monotonically",
    "tf-idf": "its vocabulary is fitted on documents only and cosine dilutes real query terms",
}


def collect():
    """(key, arm) -> (score, provenance) for complete, verified cells."""
    whole, shards = {}, defaultdict(dict)
    for path in sorted(RUNS.glob("*.json")):
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
            # An explicit __pN filename is the ablation's own run for that
            # arm; a bare "<model>.json" is the main experiment, which for the
            # qwen3-reranker family fills the model's <Instruct> slot with the
            # vendor default rather than leaving it empty. Both parse as p0, and
            # the bare name sorts FIRST, so a plain setdefault silently measures
            # p1-p3 against a different baseline condition than the arms
            # themselves. Explicit always wins; the bare file is only a fallback
            # for a model whose p0 the ablation tree does not contain.
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
        "<!-- Generated by experiments/instructions/report.py - do not edit by hand. -->",
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
        "% Generated by experiments/instructions/report.py - do not edit by hand.",
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


def main():
    cells, rejected = collect()
    ordered = rows(cells)
    (HERE / "RESULTS.md").write_text(markdown(cells, rejected, ordered))
    (HERE / "statement_full_instructions.tex").write_text(latex(ordered))
    shown = sum(1 for _, _, _, base, arms in ordered if base for g in arms if g)
    print(f"[+] {len(cells)} verified cells; {len(ordered)} table rows, {shown} arm cells")
    if rejected:
        print(f"[!] {len(rejected)} run(s) excluded:")
        for key, arm, why, source in sorted(rejected):
            print(f"      {key} {arm}: {why}")
    print("[+] wrote RESULTS.md and statement_full_instructions.tex")


if __name__ == "__main__":
    main()
