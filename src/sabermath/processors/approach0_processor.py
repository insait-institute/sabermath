from contextlib import contextmanager
from dataclasses import dataclass
import ctypes
import hashlib
import json
import os
import re
from typing import ClassVar
import shutil
import sys
import tempfile

from .base import ModelProcessor


pya0 = None

_BROKEN_QUERIES = [
    "199b11579b8b7298a613b1ca2e041661",
    "6892f32a9b3b1c33423b72ca99032e3b",
    "f5b65a45f9fcc050d3bc09e7251c6e6f",
    "175324faed909ccf469ebb699449f236",
    "3c20d12c7f956a95716b54c3a5a18657",
    "cd11b4b3032267c0c2ee4c158946a900",
    "77a07ca2724aef9be026aedd856b7015",
    "c0805a4c75470d9217c8f47495793c31",
    "467db1c46a5b32833b051f02ef77d933",
    "0131eb5a7384ba6c2308a8af1c5d7d74",
    "bdfa1a8b8dfaecd02f12e2d7c236fbd0",
    "9a1fefdab9c1e5182eb8a9a3253eece7",
    "010d2c257c9421cb447816bfcda0ecdf",
    "15970d80da7b9d40f50e844474aa5d85",
    "15970d80da7b9d40f50e844474aa5d85",
    "9184436f35cd4f5ec0afb21e06f31d76",
    "6aa8861d549689f41ad0a22453391648",
    "c14db7d65c26c4b04355187d1f7dee40",
    "5d56202d067cc417523ce51ddd72199b",
    "12b526d94e6c5bba5e3acc09f8fbe267",
    "20428a09daeec87bc7868216339cd566",
    "5c6e53511f82fa690239eb0fbf821751",
    "aadbc12ff59fbc72e08bb03100033732",
    "2ef84af93672344b9150d799c4a755c5",
    "1fbbcb4c4b7654f1eca51d4b8b90e45d",
    "ac7cb7dca2130d0ab7bfff93cbd8f1ce",
    "65c16346697a42d035567ef5c812ac74",
    "16f5db20f98541810ff3f63ecdde7ecf",
    "97ddf65b322494fe26d665dd7155ffe3",
    "ecc600c795c74028decbb11e610c3155",
]


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


@contextmanager
def silence_native_output():
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


"""
def split_text_and_latex(s: str):
    s = clean_for_pya0(s)

    combined = "|".join(f"({p})" for p in LATEX_PATTERNS)
    regex = re.compile(combined, flags=re.DOTALL)

    chunks = []
    last = 0

    for m in regex.finditer(s):
        if m.start() > last:
            text = s[last:m.start()].strip()
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
"""


def split_text_and_latex(s: str):
    s = clean_for_pya0(s)

    combined = "|".join(f"(?:{p})" for p in LATEX_PATTERNS)
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
    query = []
    for kind, value in split_text_and_latex(problem):
        if value:
            query.append({"str": value, "type": kind})
    return query


def build_pya0_index(problems: list[str], index_dir: str, q):
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


class Approach0Processor(ModelProcessor):
    processor: ClassVar[str | None] = "approach0"

    def __init__(self):
        global pya0

        if pya0 is None:
            try:
                from pya0 import _pya0

                pya0 = _pya0
            except ImportError as e:
                raise ImportError(
                    "Please install pya0 to use Approach0 as a processor"
                ) from e

    @property
    def model(self) -> str:
        return "approach0"

    def get_scores(
        self,
        query: str,
        documents: list[str],
        *,
        keep_index_dir: str | None = None,
        show_progress_bar: bool = True,
        **kwargs,
    ) -> list[float] | None:

        # Skip known-broken queries that cause
        # Approach0's engine to crash with a segfault.
        if md5(query) in _BROKEN_QUERIES:
            return None

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
            build_pya0_index(documents, index_dir, q=query)

            hits = search_pya0_index(
                problems=documents,
                target=query,
                index_dir=index_dir,
                topk=len(documents),
            )

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
