import json
from pathlib import Path

from .results import DEFAULT_RESULTS_DIR, load_runs, run_rank

TASKS = ["statement-statement", "statement-full", "full-full"]
DOMAINS = ["algebra", "combinatorics", "geometry", "number theory", "calculus"]
CURRENT_PROTOCOL_RANK = 2

BRANCH_TO_DOMAIN = {
    "algebra": "algebra",
    "combinatorics": "combinatorics",
    "geometry": "geometry",
    "number theory": "number theory",
    "calculus and analysis": "calculus",
}

# Every model key the tables can report, with its display name and the
# architecture class its scorer belongs to. `collect()` drops any run whose key
# is missing here, so an omission silently deletes a row: the four
# ReasonEmbed/ReasonReranker rows below - the top of the paper's main table -
# were absent from this map and therefore from every generated main table,
# despite complete runs sitting in results/evaluation.
#
# Names and classes follow the paper-row table in
# scripts/tables/instruction_statement_full.py, whose Type codes end in the
# architecture letter: C (cross-encoder) -> RERANK, B (bi-encoder) -> EMBED.
# The two rewrite rows are composed pipelines (rewriter + scorer) and are
# classed by their scorer, which is what every consumer of this map needs.
MODEL_INFO = {
    "retro-star-32b-rewritten": ("ReasonReranker-Qwen3-32B-Rewrite", "RERANK"),
    "reason-rewriter-reason-embed-8b": ("ReasonEmbed-Qwen3-8B-Rewrite", "EMBED"),
    "retro-star-32b": ("ReasonReranker-Qwen3-32B", "RERANK"),
    "reason-embed-llama-3.1-8b": ("ReasonEmbed-Llama-3.1-8B", "EMBED"),
    "reason-embed-qwen3-8b": ("Reason-Embed-Qwen3-8B", "EMBED"),
    "diver-retriever-4b": ("Diver-Retriever-4B", "EMBED"),
    "diver-retriever-0.6b": ("Diver-Retriever-0.6B", "EMBED"),
    "diver-grouprank-32b": ("Diver-GroupRank-32B", "RERANK"),
    "qwen3-reranker-8b": ("Qwen3-Reranker-8B", "RERANK"),
    "qwen3-reranker-4b": ("Qwen3-Reranker-4B", "RERANK"),
    "qwen3-reranker-0.6b": ("Qwen3-Reranker-0.6B", "RERANK"),
    "splade-code-8b": ("SPLADE-Code-8B", "RERANK"),
    "splade-code-0.6b": ("SPLADE-Code-0.6B", "RERANK"),
    "octen-embedding-8b": ("Octen-Embedding-8B", "EMBED"),
    "octen-embedding-4b": ("Octen-Embedding-4B", "EMBED"),
    "rader-14b": ("RaDeR-14B", "EMBED"),
    "rader-7b": ("RaDeR-7B", "EMBED"),
    "rader-3b": ("RaDeR-3B", "EMBED"),
    "rader-reranker-7b": ("RaDeR-Reranker-7B", "RERANK"),
    "gemini-embedding-2": ("Gemini-Embedding-2", "EMBED"),
    "gemini-embedding-001": ("Gemini-Embedding-001", "EMBED"),
    "qwen3-embedding-8b": ("Qwen3-Embedding-8B", "EMBED"),
    "qwen3-embedding-4b": ("Qwen3-Embedding-4B", "EMBED"),
    "qwen3-embedding-0.6b": ("Qwen3-Embedding-0.6B", "EMBED"),
    "rank1-32b": ("Rank1-32B", "RERANK"),
    "rank1-7b": ("Rank1-7B", "RERANK"),
    "rank1-0.5b": ("Rank1-0.5B", "RERANK"),
    "harrier-oss-v1-27b": ("Harrier-OSS-v1-27b", "EMBED"),
    "harrier-oss-v1-0.6b": ("Harrier-OSS-v1-0.6b", "EMBED"),
    "harrier-oss-v1-270m": ("Harrier-OSS-v1-270m", "EMBED"),
    "kalm-embedding-gemma3-12b-2511": ("KaLM-Embedding-Gemma3-12B-2511", "EMBED"),
    "llama-embed-nemotron-8b": ("LLaMa-Embed-Nemotron-8b", "EMBED"),
    "jina-embeddings-v5-text-small": ("Jina-Embeddings-v5-Text-Small", "EMBED"),
    "jina-embeddings-v5-text-nano": ("Jina-Embeddings-v5-Text-Nano", "EMBED"),
    "text-embedding-3-large": ("Text-Embedding-3-Large", "EMBED"),
    "text-embedding-3-small": ("Text-Embedding-3-Small", "EMBED"),
    "reason-moderncolbert": ("Reason-ModernColBERT", "RERANK"),
    "gte-moderncolbert": ("GTE-ModernColBERT", "RERANK"),
    "embeddinggemma-300m": ("EmbeddingGemma-300m", "EMBED"),
    "bge-m3": ("BGE-m3", "EMBED"),
    "reasonir-8b": ("ReasonIR-8B", "EMBED"),
    "multilingual-e5-large": ("Multilingual-E5-Large", "EMBED"),
    "inf-retriever-v1-pro": ("INF-Retriever-v1-Pro", "EMBED"),
    "inf-x-retriever": ("INF-X-Retriever", "SYSTEM"),
    "approach0": ("Approach Zero", "CLASSICAL"),
    "bm25": ("BM25", "CLASSICAL"),
    "tf-idf": ("TF-IDF", "CLASSICAL"),
    "jaccard": ("Jaccard", "CLASSICAL"),
    "bert-base-uncased": ("BERT", "EMBED"),
    "roberta-base": ("RoBERTa", "EMBED"),
    "bm25-no-tok": ("BM25-no-tok", "CLASSICAL"),
    "tf-idf-no-tok": ("TF-IDF-no-tok", "CLASSICAL"),
    "jaccard-no-tok": ("Jaccard-no-tok", "CLASSICAL"),
}
def extract_tasks(payload: dict) -> dict | None:
    reports = payload.get("reports")
    if not reports:
        return None
    report = next(iter(reports.values()))
    if "tasks" not in report:
        return None

    out = {}
    for task in report["tasks"]:
        name = task["task"]
        scored = report.get("ndcgs_by_task", {}).get(name, [])
        n_done = len([v for v in scored if v is not None])
        if scored and n_done < len(scored):
            continue
        by_branch = {b["branch"].lower(): b["ndcg_at_k"] for b in task["branches"]}
        domains = [
            by_branch.get(branch, 0.0)
            for branch in BRANCH_TO_DOMAIN
        ]
        out[name] = (task["ndcg_at_k"], domains, n_done)
    return out or None


def load_confidence(directory: Path) -> dict:
    out = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        by_task = {}
        for task in payload.get("tasks", []):
            by_task[task["task"]] = {
                "interval": task["confidence_interval"],
                "branches": {
                    b["branch"].lower(): b["confidence_interval"]
                    for b in task.get("branches", [])
                },
            }
        if by_task:
            out[path.stem] = by_task
    return out


def collect(results_dir=DEFAULT_RESULTS_DIR) -> tuple[dict, dict]:
    runs, ranks = {}, {}
    for (model_key, _prompt), payload in load_runs(results_dir, prompts=["p0"]).items():
        if model_key not in MODEL_INFO:
            continue
        run = extract_tasks(payload)
        if run is not None:
            runs[model_key] = run
            ranks[model_key] = run_rank(payload)
    return runs, ranks


def build_rows(runs: dict, ranks: dict, ci=None) -> list[dict]:
    ci = ci or {}
    rows, pending = [], []
    for key, (name, category) in MODEL_INFO.items():
        run = runs.get(key)
        if run is None or "statement-full" not in run:
            pending.append((key, name, "no run"))
            continue
        if ranks.get(key, 0) < CURRENT_PROTOCOL_RANK:
            pending.append((key, name, "predates the current input protocol"))
            continue
        entry = {
            "key": key,
            "name": name,
            "category": category,
            "ci": ci.get(key, {}),
        }
        for task in TASKS:
            entry[task] = run.get(task)
        entry["sort_key"] = entry["statement-full"][0]
        rows.append(entry)

    rows.sort(key=lambda r: -r["sort_key"])
    return rows, pending


def load_timing(directory=Path("results/timing")) -> dict:
    timing = {}
    for path in sorted(Path(directory).glob("*.json")):
        if path.stem == "query_sample":
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "mean_seconds" in payload:
            timing[path.stem] = payload
    return timing
