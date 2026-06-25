from contextlib import contextmanager
from dataclasses import dataclass
from datasets import Dataset
from statistics import mean
import ctypes
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import tqdm

import pya0

from embed import get_top5_candidates

_BROKEN_QUERIES = [
    # MD5 hashes of queries that cause Approach0's
    # engine to crash with a segfault. Example:
    "bdfa1a8b8dfaecd02f12e2d7c236fbd0",
]


def approach0_scores(
    query: str,
    documents: list[str],
    *,
    keep_index_dir: str | None = None,
    show_progress_bar: bool = True,
) -> list[float] | None:
    """
    This function uses the engine Approach0 to compute relevance score for each
    document in a list given a query. The higher the score, the more relevant the
    document is to the query. The function returns a list of scores corresponding
    to the input documents. Internally, it constructs a temporary index for the
    documents, performs a search with the query, and extracts the scores from
    the search results.

    Args:
        query: A string representing the search query.
        documents: A list of strings, where each string is a document to be scored.

    Optional Args:
        keep_index_dir: If provided, this directory will be used to store the temporary index.
                        If None, a temporary directory will be created and deleted after use.
        show_progress_bar: If True, a progress bar will be displayed during the search process.

    Returns:
        A list of floats representing the relevance scores for each document (in the same order
        as the documents).
        A bug in Approach0's engine might cause Segmentation fault. If a query is known to lead
        to such and it is reported in the _BROKEN_QUERIES list above, the function returns None.
    """
    if len(documents) == 0:
        return []

    if keep_index_dir is None:
        index_dir = tempfile.mkdtemp(prefix="pya0_problem_index_")
        cleanup = True
    else:
        index_dir = keep_index_dir
        os.makedirs(index_dir, exist_ok=True)
        cleanup = False

    try:
        build_pya0_index(documents, index_dir)

        hits = search_pya0_index(
            problems=documents,
            target=query,
            index_dir=index_dir,
            topk=len(documents),
        )

        if hits is None:
            return None

        by_index = {h["index"]: h for h in hits if 0 <= h["index"] < len(documents)}

        ranked = []
        for i, document in enumerate(documents):
            h = by_index.get(i)
            if h is None:
                ranked.append(0.0)
            else:
                ranked.append(h["score"])

        return ranked

    finally:
        if cleanup:
            shutil.rmtree(index_dir, ignore_errors=True)


"""
====== HELPERS ======
"""


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


@contextmanager
def silence_native_output():
    """
    Silence stdout + stderr at the OS file-descriptor level.

    This catches both Python output and native extension output from pya0,
    but only inside the scoped `with` block.
    """
    stdout_fd = 1
    stderr_fd = 2

    # Flush Python-level buffers before swapping file descriptors.
    sys.stdout.flush()
    sys.stderr.flush()

    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        # Flush again before restoring.
        sys.stdout.flush()
        sys.stderr.flush()

        # Flush native C stdio while stdout/stderr still point to /dev/null.
        ctypes.CDLL(None).fflush(None)

        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)

        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


@dataclass
class RankedProblem:
    index: int
    problem: str
    score: float
    rank: int | None
    raw_result: dict | None


LATEX_PATTERNS = [
    r"\$\$(.+?)\$\$",
    r"\$(.+?)\$",
    r"\\\[(.+?)\\\]",
    r"\\\((.+?)\\\)",
    r"\\begin\{equation\*?\}(.+?)\\end\{equation\*?\}",
    r"\\begin\{align\*?\}(.+?)\\end\{align\*?\}",
]


def clean_for_pya0(value) -> str:
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    # PyA0's C wrapper uses PyArg_ParseTupleAndKeywords with "s",
    # so embedded NUL characters will crash argument parsing.
    value = value.replace("\x00", " ")

    # Replace invalid Unicode/surrogate characters so UTF-8 conversion succeeds.
    value = value.encode("utf-8", errors="replace").decode("utf-8")

    return value


def split_text_and_latex(s: str):
    """
    Returns a list of chunks:
      ("term", normal_text)
      ("tex", latex)
    """

    s = clean_for_pya0(s)

    combined = "|".join(f"({p})" for p in LATEX_PATTERNS)
    regex = re.compile(combined, flags=re.DOTALL)

    chunks = []
    last = 0

    for m in regex.finditer(s):
        if m.start() > last:
            text = s[last : m.start()].strip()
            if text:
                chunks.append(("term", text))

        latex = next(g for g in m.groups() if g is not None).strip()
        if latex:
            chunks.append(("tex", latex))

        last = m.end()

    if last < len(s):
        text = s[last:].strip()
        if text:
            chunks.append(("term", text))

    return chunks


def to_pya0_document(problem: str) -> str:
    """
    PyA0 indexing expects math in [imath]...[/imath] blocks.
    """
    parts = []
    for kind, value in split_text_and_latex(problem):
        if kind == "term":
            parts.append(value)
        else:
            parts.append(f"[imath]{value}[/imath]")
    return " ".join(parts)


def to_pya0_query(problem: str):
    """
    PyA0 search keywords.

    Recent source code expects key 'str'.
    Some old examples online used 'keyword'; if your installed version errors,
    replace 'str' with 'keyword'.
    """
    query = []
    for kind, value in split_text_and_latex(problem):
        if value:
            query.append({"str": value, "type": kind})
    return query


def build_pya0_index(problems: list[str], index_dir: str):
    with silence_native_output():
        idx = pya0.index_open(index_dir, option="w")
        writer = pya0.index_writer(idx)

        for i, problem in enumerate(problems):
            content = to_pya0_document(problem)

            # Store the original Python index in the URL field for easy recovery.
            pya0.writer_add_doc(
                writer,
                content=content,
                url=f"problem:{i}",
            )

        pya0.writer_flush(writer)
        pya0.writer_maintain(writer, force=True)
        pya0.writer_close(writer)
        pya0.index_close(idx)


def search_pya0_index(
    problems: list[str],
    target: str,
    index_dir: str,
    topk: int,
):
    # Skip known-broken queries that cause
    # Approach0's engine to crash with a segfault.
    if md5(target) in _BROKEN_QUERIES:
        return None

    idx = pya0.index_open(index_dir, option="r")
    query = to_pya0_query(target)

    raw = pya0.search(
        idx,
        query,
        topk=topk,
    )

    pya0.index_close(idx)

    results = json.loads(raw)

    if isinstance(results, dict):
        candidates = (
            results.get("results")
            or results.get("hits")
            or results.get("documents")
            or []
        )
    else:
        candidates = results

    normalized = []

    for rank, r in enumerate(candidates, start=1):
        url = r.get("url", "")
        idx_match = re.search(r"problem:(\d+)", url)

        if idx_match:
            problem_index = int(idx_match.group(1))
        else:
            doc_id = r.get("docid") or r.get("docID") or r.get("id")
            problem_index = int(doc_id) - 1

        score = r.get("score")

        normalized.append(
            {
                "index": problem_index,
                "score": float(score) if score is not None else None,
                "rank": rank,
                "raw_result": r,
            }
        )

    return normalized


def calc_approach0_sims(good_targets: Dataset, good_candidates: Dataset):

    output_path = "similarities/approach0.json"

    similarities_dict = {}

    for target in tqdm.tqdm(good_targets):

        target_id = target["id"]
        cands_idxs = get_top5_candidates(target)

        query_math = target["problem_math_expr"]
        query_text = target["problem_text_only"]
        query_full = target["problem_fixed"]

        candidates = [
            good_candidates[i]["problem_fixed"] + good_candidates[i]["solution_fixed"]
            for i in cands_idxs
        ]

        math_scores = approach0_scores(query_math, candidates)
        text_scores = approach0_scores(query_text, candidates)
        full_scores = approach0_scores(query_full, candidates)

        similarities_dict[target_id] = {
            "pr_full_vs_candidates": float(mean(full_scores)),
            "pr_math_vs_candidates": float(mean(math_scores)),
            "pr_text_vs_candidates": float(mean(text_scores)),
        }

        with open(output_path, "w") as f:
            json.dump(similarities_dict, f)
