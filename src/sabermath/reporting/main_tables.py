"""Generate the SABER-Math result tables as markdown, one file per table.

    python scripts/report_experiments.py --out-dir tables

Each row is drawn from the best available source, in this order:

1. `results/evaluation/` - runs on the current input protocol. These
   SUPERSEDE everything below for the models whose inputs changed
   (see docs/protocol.md).
2. `results/evaluation/` - full 1000-query runs on this repo's pipeline.
3. The paper's Tables 1 and 4, transcribed below, for models this repo has
   no run of.

Every row records which source it came from, so a table is never a silent
mix of protocols. Timing comes from `results/timing/`.
"""

import argparse
import json
from pathlib import Path

from ..results import DEFAULT_RESULTS_DIR, load_runs, run_rank

TASKS = ["statement-statement", "statement-full", "full-full"]
DOMAINS = ["algebra", "combinatorics", "geometry", "number theory", "calculus"]
BRANCH_TO_DOMAIN = {
    "algebra": "algebra",
    "combinatorics": "combinatorics",
    "geometry": "geometry",
    "number theory": "number theory",
    "calculus and analysis": "calculus",
}

# Display name and category, keyed by this repo's model key.
MODEL_INFO = {
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
    # Each lexical method in its own default tokenization, with no
    # mathematics-aware preprocessing. The paper carries these beside the
    # math-aware rows in Table 3.
    "bm25-no-tok": ("BM25-no-tok", "CLASSICAL"),
    "tf-idf-no-tok": ("TF-IDF-no-tok", "CLASSICAL"),
    "jaccard-no-tok": ("Jaccard-no-tok", "CLASSICAL"),
}

# Rows the paper reports ONLY in Table 3 (statement-full, with a CI) and not
# in Tables 1 or 4, so there is no statement-statement or full-full reference
# value to compare against - our runs supply those columns outright.
PAPER_STATEMENT_FULL_ONLY = {
    "bm25-no-tok": 0.398,
    "tf-idf-no-tok": 0.430,
    "jaccard-no-tok": 0.403,
}

# Paper Tables 1 and 4, verbatim. Order: statement-full overall, then the five
# domains for statement-full, then statement-statement and full-full overall,
# then the five domains for each of those two.
PAPER = {
    "reason-embed-qwen3-8b": (0.710, [0.689, 0.738, 0.744, 0.697, 0.682], 0.685, [0.668, 0.724, 0.706, 0.675, 0.655], 0.795, [0.768, 0.823, 0.819, 0.793, 0.775]),
    "diver-retriever-4b": (0.681, [0.665, 0.708, 0.711, 0.672, 0.645], 0.659, [0.641, 0.691, 0.696, 0.647, 0.619], 0.737, [0.713, 0.759, 0.771, 0.737, 0.711]),
    "qwen3-reranker-8b": (0.671, [0.638, 0.703, 0.697, 0.665, 0.645], 0.635, [0.607, 0.668, 0.655, 0.629, 0.611], 0.723, [0.699, 0.752, 0.732, 0.730, 0.704]),
    "splade-code-8b": (0.650, [0.610, 0.694, 0.689, 0.644, 0.618], 0.634, [0.592, 0.675, 0.665, 0.633, 0.600], 0.693, [0.649, 0.729, 0.725, 0.691, 0.673]),
    "octen-embedding-8b": (0.636, [0.594, 0.665, 0.664, 0.629, 0.630], 0.623, [0.586, 0.654, 0.648, 0.624, 0.606], 0.672, [0.631, 0.700, 0.685, 0.670, 0.673]),
    "rader-14b": (0.635, [0.623, 0.655, 0.671, 0.615, 0.612], 0.644, [0.625, 0.675, 0.678, 0.626, 0.609], 0.708, [0.689, 0.737, 0.738, 0.700, 0.685]),
    "octen-embedding-4b": (0.632, [0.586, 0.667, 0.673, 0.627, 0.609], 0.619, [0.581, 0.651, 0.659, 0.615, 0.591], 0.663, [0.616, 0.695, 0.695, 0.658, 0.653]),
    "gemini-embedding-2": (0.628, [0.599, 0.656, 0.647, 0.622, 0.614], 0.603, [0.569, 0.630, 0.623, 0.602, 0.587], 0.658, [0.630, 0.675, 0.665, 0.659, 0.652]),
    "qwen3-embedding-4b": (0.615, [0.576, 0.642, 0.652, 0.610, 0.597], 0.597, [0.554, 0.626, 0.638, 0.591, 0.573], 0.667, [0.624, 0.693, 0.693, 0.670, 0.656]),
    "qwen3-embedding-8b": (0.611, [0.569, 0.633, 0.647, 0.606, 0.598], 0.600, [0.562, 0.622, 0.643, 0.600, 0.569], 0.662, [0.621, 0.679, 0.693, 0.660, 0.657]),
    "rank1-32b": (0.610, [0.581, 0.673, 0.618, 0.614, 0.575], 0.556, [0.538, 0.620, 0.545, 0.546, 0.531], 0.629, [0.594, 0.683, 0.645, 0.623, 0.604]),
    "harrier-oss-v1-27b": (0.608, [0.569, 0.620, 0.651, 0.601, 0.596], 0.585, [0.554, 0.609, 0.612, 0.586, 0.562], 0.659, [0.615, 0.678, 0.700, 0.664, 0.640]),
    "gemini-embedding-001": (0.605, [0.573, 0.626, 0.650, 0.604, 0.577], 0.591, [0.553, 0.619, 0.629, 0.598, 0.560], 0.667, [0.630, 0.685, 0.690, 0.679, 0.651]),
    "kalm-embedding-gemma3-12b-2511": (0.585, [0.548, 0.606, 0.617, 0.583, 0.569], 0.579, [0.546, 0.597, 0.605, 0.578, 0.565], 0.599, [0.557, 0.618, 0.628, 0.607, 0.585]),
    "llama-embed-nemotron-8b": (0.579, [0.542, 0.600, 0.610, 0.580, 0.562], 0.565, [0.535, 0.588, 0.584, 0.578, 0.540], 0.616, [0.569, 0.638, 0.641, 0.629, 0.602]),
    "qwen3-embedding-0.6b": (0.575, [0.545, 0.589, 0.629, 0.564, 0.546], 0.566, [0.529, 0.586, 0.611, 0.564, 0.538], 0.625, [0.586, 0.634, 0.676, 0.624, 0.603]),
    "harrier-oss-v1-0.6b": (0.572, [0.538, 0.581, 0.613, 0.566, 0.557], 0.549, [0.521, 0.568, 0.589, 0.538, 0.523], 0.632, [0.588, 0.638, 0.676, 0.637, 0.613]),
    "jina-embeddings-v5-text-small": (0.570, [0.525, 0.593, 0.620, 0.561, 0.549], 0.554, [0.513, 0.577, 0.603, 0.549, 0.528], 0.610, [0.556, 0.631, 0.664, 0.608, 0.591]),
    "text-embedding-3-large": (0.558, [0.535, 0.574, 0.571, 0.560, 0.552], 0.539, [0.522, 0.560, 0.554, 0.541, 0.526], 0.593, [0.564, 0.611, 0.609, 0.598, 0.588]),
    "reason-moderncolbert": (0.557, [0.533, 0.567, 0.592, 0.555, 0.542], 0.540, [0.521, 0.554, 0.568, 0.538, 0.518], 0.601, [0.561, 0.607, 0.638, 0.613, 0.587]),
    "jina-embeddings-v5-text-nano": (0.532, [0.492, 0.552, 0.573, 0.538, 0.506], 0.522, [0.484, 0.547, 0.563, 0.529, 0.487], 0.569, [0.522, 0.585, 0.604, 0.576, 0.560]),
    "embeddinggemma-300m": (0.519, [0.496, 0.540, 0.550, 0.518, 0.485], 0.511, [0.488, 0.545, 0.547, 0.502, 0.472], 0.588, [0.548, 0.604, 0.629, 0.594, 0.559]),
    "gte-moderncolbert": (0.519, [0.506, 0.527, 0.540, 0.514, 0.509], 0.493, [0.480, 0.504, 0.507, 0.485, 0.486], 0.515, [0.500, 0.522, 0.537, 0.513, 0.504]),
    "text-embedding-3-small": (0.512, [0.487, 0.526, 0.557, 0.504, 0.491], 0.497, [0.470, 0.519, 0.540, 0.483, 0.473], 0.554, [0.520, 0.565, 0.597, 0.552, 0.536]),
    "bge-m3": (0.511, [0.484, 0.518, 0.549, 0.502, 0.500], 0.505, [0.483, 0.518, 0.539, 0.498, 0.485], 0.563, [0.513, 0.567, 0.616, 0.575, 0.543]),
    "reasonir-8b": (0.507, [0.495, 0.534, 0.511, 0.500, 0.489], 0.513, [0.504, 0.538, 0.529, 0.503, 0.482], 0.617, [0.584, 0.629, 0.635, 0.618, 0.613]),
    "harrier-oss-v1-270m": (0.498, [0.470, 0.512, 0.545, 0.491, 0.469], 0.497, [0.466, 0.517, 0.537, 0.489, 0.473], 0.557, [0.525, 0.565, 0.590, 0.565, 0.539]),
    "multilingual-e5-large": (0.488, [0.455, 0.500, 0.529, 0.471, 0.477], 0.508, [0.485, 0.523, 0.549, 0.501, 0.475], 0.545, [0.505, 0.544, 0.580, 0.549, 0.536]),
    "approach0": (0.468, [0.485, 0.480, 0.443, 0.454, 0.490], 0.469, [0.489, 0.483, 0.446, 0.456, 0.484], 0.481, [0.493, 0.505, 0.471, 0.461, 0.487]),
    "bm25": (0.416, [0.405, 0.429, 0.448, 0.392, 0.393], 0.426, [0.421, 0.446, 0.450, 0.409, 0.397], 0.437, [0.420, 0.455, 0.454, 0.425, 0.417]),
    "tf-idf": (0.412, [0.414, 0.424, 0.402, 0.383, 0.414], 0.434, [0.445, 0.453, 0.435, 0.427, 0.410], 0.447, [0.447, 0.447, 0.457, 0.442, 0.433]),
    "jaccard": (0.412, [0.381, 0.442, 0.448, 0.397, 0.383], 0.448, [0.425, 0.477, 0.476, 0.445, 0.406], 0.469, [0.438, 0.496, 0.500, 0.472, 0.419]),
    "bert-base-uncased": (0.357, [0.369, 0.389, 0.342, 0.344, 0.335], 0.417, [0.432, 0.440, 0.399, 0.394, 0.416], 0.429, [0.435, 0.453, 0.428, 0.425, 0.406]),
    "roberta-base": (0.311, [0.306, 0.342, 0.293, 0.314, 0.287], 0.406, [0.415, 0.423, 0.394, 0.399, 0.392], 0.397, [0.409, 0.416, 0.390, 0.382, 0.381]),
}

# 95% CI half-widths on the OVERALL statement-full score, transcribed from
# the paper's Table 3 as (plus, minus). Only the overall column is carried
# over: the per-domain intervals are in that table, and for the rows this repo
# recomputes they are written out in full to RESULTS_confidence_intervals.md.
# The paper itself notes its Table 1 and Table 3 point estimates differ by up
# to 0.001 from bootstrap resampling noise; Table 1 is used for the value.
PAPER_CI_STATEMENT_FULL = {
    "reason-embed-qwen3-8b": (0.010, 0.010),
    "diver-retriever-4b": (0.010, 0.010),
    "qwen3-reranker-8b": (0.010, 0.010),
    "splade-code-8b": (0.009, 0.009),
    "octen-embedding-8b": (0.009, 0.010),
    "rader-14b": (0.010, 0.010),
    "octen-embedding-4b": (0.009, 0.010),
    "gemini-embedding-2": (0.009, 0.009),
    "qwen3-embedding-4b": (0.009, 0.009),
    "qwen3-embedding-8b": (0.010, 0.010),
    "rank1-32b": (0.010, 0.010),
    "harrier-oss-v1-27b": (0.009, 0.009),
    "gemini-embedding-001": (0.009, 0.009),
    "kalm-embedding-gemma3-12b-2511": (0.009, 0.009),
    "llama-embed-nemotron-8b": (0.009, 0.009),
    "qwen3-embedding-0.6b": (0.009, 0.009),
    "harrier-oss-v1-0.6b": (0.009, 0.009),
    "jina-embeddings-v5-text-small": (0.009, 0.009),
    "text-embedding-3-large": (0.009, 0.009),
    "reason-moderncolbert": (0.009, 0.009),
    "jina-embeddings-v5-text-nano": (0.009, 0.009),
    "embeddinggemma-300m": (0.009, 0.009),
    "gte-moderncolbert": (0.009, 0.009),
    "text-embedding-3-small": (0.009, 0.009),
    "bge-m3": (0.009, 0.009),
    "reasonir-8b": (0.009, 0.009),
    "harrier-oss-v1-270m": (0.009, 0.009),
    "multilingual-e5-large": (0.009, 0.009),
    "approach0": (0.009, 0.009),
    "bm25": (0.008, 0.008),
    "tf-idf": (0.009, 0.009),
    "jaccard": (0.008, 0.008),
    "bert-base-uncased": (0.008, 0.007),
    "roberta-base": (0.007, 0.007),
    "bm25-no-tok": (0.008, 0.008),
    "tf-idf-no-tok": (0.008, 0.008),
    "jaccard-no-tok": (0.008, 0.008),
}

# Models whose production inputs changed on 2026-08-25 and whose
# results/evaluation run therefore supersedes every earlier source.
SUPERSEDED_BY_V2 = {
    "rader-3b",
    "rader-7b",
    "rader-14b",
    "reasonir-8b",
    "embeddinggemma-300m",
    "jina-embeddings-v5-text-nano",
    "jina-embeddings-v5-text-small",
    "gemini-embedding-001",
}


def extract_tasks(payload: dict) -> dict | None:
    """One result payload as {task: (overall, [domain scores], n_scored)}, or
    None if it holds no usable data. A task scored on fewer queries than the
    run covers is DROPPED, never averaged into a table as if complete."""
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
    """Read the bootstrap CIs written by
    python -m sabermath.analysis.compute_confidence_intervals: 10,000 resamples,
    seed 42411, 95% percentile intervals - the same estimator that produced
    the paper's Table 3, so a recomputed row is directly comparable to a
    transcribed one."""
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
    """({model_key: run}, {model_key: protocol_rank}) for the p0 arm.

    Reads ONE directory. Which of several runs of a cell wins is decided by
    the payload's own recorded protocol (see results.run_rank), not by which
    directory it sits in - so a row cannot change because a file was copied
    somewhere, and no caller has to pass directories in the right order.
    """
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
    # One CI directory, matching the one results directory: the intervals are
    # recomputed from whichever run backs the row, so there is no longer a
    # per-protocol CI dir to choose between.
    ci = ci or {}
    rows = []
    for key, (name, category) in MODEL_INFO.items():
        run = runs.get(key)
        # "canonical" == this run applied the vendor input envelopes; anything
        # lower predates them.
        source = None
        if run is not None:
            source = "v2" if ranks.get(key, 0) >= 2 else "repo"

        # A model whose inputs changed but whose rerun has not landed yet is
        # NOT shown with its old number - the old number was produced with a
        # configuration we no longer run, so it would be wrong in a way a
        # reader could not see. It gets an explicit TODO instead.
        todo = key in SUPERSEDED_BY_V2 and source != "v2"

        paper = PAPER.get(key)

        if todo:
            entry = {
                "key": key,
                "name": name,
                "category": category,
                "source": "TODO",
                "todo": True,
                "ci": {},
            }
            for task in TASKS:
                entry[task] = None
            if paper is not None:
                entry["paper_statement_full"] = paper[0]
                entry["sort_key"] = paper[0]
            rows.append(entry)
            continue

        if run is not None and "statement-full" in run:
            entry = {
                "key": key,
                "name": name,
                "category": category,
                "source": source,
                "todo": False,
                "ci": ci.get(key, {}),
            }
            for task in TASKS:
                entry[task] = run.get(task)
            if paper is not None:
                entry["paper_statement_full"] = paper[0]
            elif key in PAPER_STATEMENT_FULL_ONLY:
                entry["paper_statement_full"] = PAPER_STATEMENT_FULL_ONLY[key]
            entry["sort_key"] = entry["statement-full"][0]
            rows.append(entry)
            continue

        if paper is None:
            continue
        sf, sf_d, ss, ss_d, ff, ff_d = paper
        plus, minus = PAPER_CI_STATEMENT_FULL.get(key, (None, None))
        # A transcribed row's interval comes from the paper, not from a
        # bootstrap over a run we have - name it apart from the `ci` argument
        # so the two can never be confused.
        paper_ci = {}
        if plus is not None:
            paper_ci["statement-full"] = {
                "interval": [sf - minus, sf + plus],
                "branches": {},
                "from_paper": True,
            }
        rows.append(
            {
                "key": key,
                "name": name,
                "category": category,
                "source": "paper",
                "todo": False,
                "ci": paper_ci,
                "statement-full": (sf, sf_d, 1000),
                "statement-statement": (ss, ss_d, 1000),
                "full-full": (ff, ff_d, 1000),
                "paper_statement_full": sf,
                "sort_key": sf,
            }
        )

    rows.sort(key=lambda r: -r.get("sort_key", 0))
    return rows


def load_timing(directory=Path("results/timing")) -> dict:
    """Every model's per-query latency. One directory: the retimed runs were
    merged over the originals when the results tree was consolidated, so
    "newer wins" is already resolved on disk rather than by read order."""
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

def fmt(value) -> str:
    return "-" if value is None else f"{value:.3f}"


def delta_note(row) -> str:
    paper = row.get("paper_statement_full")
    if paper is None or row["source"] == "paper" or row["statement-full"] is None:
        return ""
    delta = row["statement-full"][0] - paper
    if abs(delta) < 0.0005:
        return ""
    return f"{delta:+.3f}"


def overall_cell(row, task) -> str:
    entry = row.get(task)
    if entry is None:
        return "**TODO**"
    overall = entry[0]
    ci = row.get("ci", {}).get(task)
    if not ci:
        return f"{overall:.3f}"
    lo, hi = ci["interval"]
    return f"{overall:.3f} [{lo:.3f}, {hi:.3f}]"


def task_sort_key(row, task) -> float:
    """Sort each table by its OWN task, not by the main setting - these are
    standalone tables, and a table ordered by a column it does not show reads
    as unsorted. A TODO row sorts at the paper's value for that task so it
    still lands in a sensible place."""
    entry = row.get(task)
    if entry is not None:
        return entry[0]
    paper = PAPER.get(row["key"])
    if paper is None:
        return 0.0
    return {"statement-full": paper[0], "statement-statement": paper[2],
            "full-full": paper[4]}.get(task, 0.0)


def table_for_task(rows, task, timing=None) -> str:
    rows = sorted(rows, key=lambda r: -task_sort_key(r, task))
    header = ["Model", "Cat", "Src", f"Overall (95% CI)"] + [
        d.title() for d in DOMAINS
    ]
    if task == "statement-full":
        header += ["vs paper"]
    if timing is not None:
        header += ["Median (s)"]

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]

    for row in rows:
        entry = row.get(task)
        cells = [row["name"], row["category"], row["source"], overall_cell(row, task)]
        if entry is None:
            cells += ["TODO"] * len(DOMAINS)
        else:
            cells += [fmt(v) for v in entry[1]]
        if task == "statement-full":
            cells.append("TODO" if entry is None else (delta_note(row) or ""))
        if timing is not None:
            t = timing.get(row["key"])
            cells.append(f"{t['median_seconds']:.3f}" if t else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def confidence_detail(rows) -> list[str]:
    """Per-domain intervals for every row this repo recomputed. The paper's
    own per-domain intervals stay in its Table 3 - they are not restated
    here, so nothing in this file is a transcription."""
    body = []
    for task in TASKS:
        # r[task] must exist too, not just the interval: a model can have a
        # bootstrapped CI for a task whose point estimate this table has no
        # source for (a run that finished in one results dir but not the one
        # this table reads), and the row prints the two side by side.
        have = [
            r for r in rows
            if r.get("ci", {}).get(task)
            and r.get(task)
            and not r["ci"][task].get("from_paper")
            and r["ci"][task].get("branches")
        ]
        if not have:
            continue
        body += [f"## {task}", ""]
        header = ["Model", "Src", "Overall"] + [d.title() for d in DOMAINS]
        body += [
            "| " + " | ".join(header) + " |",
            "|" + "|".join("---" for _ in header) + "|",
        ]
        for row in have:
            ci = row["ci"][task]
            lo, hi = ci["interval"]
            cells = [
                row["name"],
                row["source"],
                f"{row[task][0]:.4f} [{lo:.4f}, {hi:.4f}]",
            ]
            for domain in DOMAINS:
                branch = "calculus and analysis" if domain == "calculus" else domain
                iv = ci["branches"].get(branch)
                cells.append(f"[{iv[0]:.4f}, {iv[1]:.4f}]" if iv else "-")
            body.append("| " + " | ".join(cells) + " |")
        body.append("")
    return body


HEADER_NOTE = """<!--
Generated by scripts/report_experiments.py.

Src column:
  v2    - run on the current input protocol (results/evaluation). These rows
          supersede the paper's, and the "vs paper" column gives the change.
  repo  - full 1000-query run on this repo's pipeline (results/evaluation).
  paper - transcribed from the paper's Tables 1 and 4.
  TODO  - the inputs for this model changed and the rerun has NOT landed yet.
          The old number is deliberately not shown: it was produced with a
          configuration we no longer run.

Confidence intervals are 95% percentile bootstrap, 10,000 resamples, seed
42411 - the same estimator that produced the paper's Table 3, so a recomputed
interval is directly comparable to a transcribed one. Only the overall column
carries one here; per-domain intervals for every recomputed row are in
RESULTS_confidence_intervals.md. The paper reports no intervals at all for
statement-statement or full-full, so those tables show intervals only for rows
this repo recomputed.

Protocol changes behind the v2 rows are documented in
docs/protocol.md. Notably:
multilingual-e5-large deliberately keeps our prompt-free configuration
against its model card, and ReasonIR-8B runs on the official remote-code
path rather than vLLM.
-->
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Evaluation results. One directory: protocol precedence comes "
        "from each run's own recorded protocol, not from its location.",
    )
    parser.add_argument("--ci-dir", type=Path, default=Path("results/confidence"))
    parser.add_argument("--timing-dir", type=Path, default=Path("results/timing"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()

    runs, ranks = collect(args.results_dir)
    rows = build_rows(runs, ranks, load_confidence(args.ci_dir))
    timing = load_timing(args.timing_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    files = {
        "RESULTS_main_statement-full.md": (
            "SABER-Math — Statement–Full (main setting)",
            "statement-full",
            timing,
            "The query is a problem statement; each document is a problem "
            "statement plus its solution. This is the setting the paper "
            "reports by default.",
        ),
        "RESULTS_statement-statement.md": (
            "SABER-Math — Statement–Statement",
            "statement-statement",
            None,
            "Queries and documents are both problem statements only — search "
            "over corpora without human-readable solutions.",
        ),
        "RESULTS_full-full.md": (
            "SABER-Math — Full–Full",
            "full-full",
            None,
            "Queries and documents both carry statement plus solution — "
            "retrieval over solved material.",
        ),
    }

    for filename, (title, task, timing_for_table, blurb) in files.items():
        body = [HEADER_NOTE, f"# {title}\n", blurb, "", "All numbers are nDCG@10.\n"]
        body.append(table_for_task(rows, task, timing_for_table))
        body.append("")

        todo = [r for r in rows if r.get("todo")]
        if todo:
            body += [
                "\n## Outstanding",
                "",
                "These models' inputs changed and the rerun has not landed yet, "
                "so no number is shown for them:",
                "",
            ]
            for row in todo:
                paper = row.get("paper_statement_full")
                was = f" (paper had {paper:.3f} on statement-full)" if paper else ""
                body.append(f"- **TODO {row['name']}**{was}")
            body.append("")

        changed = [r for r in rows if r["source"] == "v2"]
        if changed and task == "statement-full":
            body += ["\n## Rows that changed", "",
                     "| Model | Paper | Current (95% CI) | Delta |", "|---|---|---|---|"]
            for row in sorted(changed, key=lambda r: r["name"]):
                paper = row.get("paper_statement_full")
                if paper is None:
                    delta = "new row"
                else:
                    delta = delta_note(row) or "0.000"
                body.append(
                    f"| {row['name']} | {fmt(paper)} | {overall_cell(row, task)} | "
                    f"{delta} |"
                )
            body.append("")

        path = args.out_dir / filename
        path.write_text("\n".join(body))
        written.append(path)

    detail = confidence_detail(rows)
    if detail:
        body = [
            HEADER_NOTE,
            "# SABER-Math — Confidence intervals (recomputed rows)\n",
            "95% percentile bootstrap, 10,000 resamples, seed 42411; domains "
            "resampled at 300 draws, overall at full query count. Produced by "
            "`python -m sabermath.analysis.compute_confidence_intervals`.\n",
        ] + detail
        path = args.out_dir / "RESULTS_confidence_intervals.md"
        path.write_text("\n".join(body))
        written.append(path)

    counts = {}
    for row in rows:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    print(
        f"[+] {len(rows)} models: "
        + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    )
    for path in written:
        print(f"[+] Wrote {path}")


if __name__ == "__main__":
    main()
