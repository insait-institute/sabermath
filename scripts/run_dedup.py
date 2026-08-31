#!/usr/bin/env python3
"""Deduplication: where does an LLM-rephrased copy of a query's own problem
rank when inserted into the corpus? THE endpoint for the dedup experiment.

    # every eligible model
    python scripts/run_dedup.py

    # a subset
    python scripts/run_dedup.py --models bge-m3 qwen3-embedding-8b

    # smoke test
    python scripts/run_dedup.py --models bge-m3 --n 20

TWO REGIMES, computed from the same embeddings in one pass, answering
different questions - never merge them into one table:

  per-query-candidates  the PUBLISHED protocol. The copy competes against
                        that query's own 150 candidates, ranked among 151.
                        Every model can run this.
  all-documents         the copy competes against all 71,117 documents. Much
                        harder, no published reference. Bi-encoders and
                        lexical models only - a pair scorer would need 71,117
                        forward passes per query.

Results land in results/dedup/. Read them with scripts/report_experiments.py.
A sharded sweep is stitched back together with --merge-shards.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sabermath import registry as rr

from sabermath.benchmark import transform
from sabermath.data import load_data
from sabermath.processors import EmbeddingProcessor
from sabermath.processors.embedding_processor import (
    apply_affixes,
    split_affix_kwargs,
)
from sabermath.shards import add_merge_arguments, merge_dedup_shards, run_merge

DEFAULT_REPHRASED_DATASET = "RAG4Math/targets-with-rephrased"
TOP_K_LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]

# Models that score a (query, document) pair rather than producing an
# independent document vector. They can only run the per-query regime - see
# run_pairwise. approach0 is deliberately absent: its _BROKEN_QUERIES md5
# skip-list is keyed to raw benchmark query text.
PAIRWISE_MODELS = frozenset(
    {
        "qwen3-reranker-0.6b",
        "qwen3-reranker-4b",
        "qwen3-reranker-8b",
        "rank1-0.5b",
        "rank1-7b",
        "rank1-32b",
        "rader-reranker-7b",
        "diver-grouprank-32b",
        "gte-moderncolbert",
        "reason-moderncolbert",
        # Retro* and its rewritten-query variants: generative pointwise
        # rerankers. They emit a score per (query, document) pair from a
        # <score> tag and have no encode() at all, so the vector path cannot
        # serve them - the per-query regime through get_scores() is the only
        # one that exists for this family.
        "retro-star-8b",
        "retro-star-32b",
        "retro-star-8b-rewritten",
        "retro-star-32b-rewritten",
        "splade-code-0.6b",
        "splade-code-8b",
    }
)
# NOT in the set above, deliberately: reason-rewriter-reason-embed-8b was
# routed through run_pairwise for a while because its QUERY vector is not
# encode(query) - it is the mean of five rewrites' embeddings - and the vector
# path would otherwise have measured the bare retriever under the composed
# model's name. That cost it the all-documents regime, which needs a
# standalone query vector. It now implements encode_queries(), which the
# vector path prefers when present, so BOTH regimes are correct for it and
# nothing has to be special-cased here.


DEDUP_EXCLUDED = {
    "approach0": (
        "cannot run the dedup experiment - its _BROKEN_QUERIES md5 skip-list "
        "is keyed to raw benchmark query text, so a rephrased insertion "
        "reintroduces known segfaults"
    ),
}


def align_rephrased(queries, rephrased):
    problems = list(queries["problem"])
    reph_problems = list(rephrased["problem"])
    if problems == reph_problems:
        return list(rephrased["rephrased_problem"]), list(rephrased["solution"])
    by_problem = {
        p: i for i, p in enumerate(reph_problems)
    }
    if len(by_problem) == len(reph_problems) and all(
        p in by_problem for p in problems
    ):
        order = [by_problem[p] for p in problems]
        reph = list(rephrased["rephrased_problem"])
        sol = list(rephrased["solution"])
        return [reph[i] for i in order], [sol[i] for i in order]
    raise SystemExit(
        "Could not align the rephrased dataset with the queries dataset - "
        "problems differ and are not a permutation."
    )


def self_document_positions(queries, documents, doc_ids) -> dict[int, int]:
    """Position in the scored corpus of each query's OWN original document.

    92.8% of SABER-Math query problems also appear verbatim in the document
    corpus. Left in, that identical original outranks the rephrased copy for
    almost every query - which is why a first run of this experiment put only
    1.9% of rephrased copies at rank 1 but 24% at rank <= 2. That measures
    "can the model find an exact duplicate", not "can it find a rephrase",
    so the self-match is excluded by default (--keep-self-match restores it).
    """
    position_of_doc = {doc_id: pos for pos, doc_id in enumerate(doc_ids)}
    doc_index_by_problem = {}
    for i, problem in enumerate(documents["problem"]):
        doc_index_by_problem.setdefault(problem, i)

    out = {}
    for row, problem in enumerate(queries["problem"]):
        doc_index = doc_index_by_problem.get(problem)
        if doc_index is not None and doc_index in position_of_doc:
            out[row] = position_of_doc[doc_index]
    return out


def build_texts(queries, documents, rephrased, doc_version, query_version, corpus):
    rephrased_problems, solutions = align_rephrased(queries, rephrased)

    # Every document that appears in any candidate list - which for this
    # benchmark is the whole document set. Both regimes score exactly these.
    doc_ids = sorted({int(i) for row in queries["candidates"] for i in row})

    corpus_texts = transform(documents, doc_version, doc_ids)

    if doc_version == "full":
        rephrased_texts = [
            f"Problem: {p}\n\nSolution: {s}"
            for p, s in zip(rephrased_problems, solutions)
        ]
    else:
        rephrased_texts = rephrased_problems

    query_texts = transform(queries, query_version)

    position_of_doc = {doc_id: pos for pos, doc_id in enumerate(doc_ids)}
    candidate_positions = [
        [position_of_doc[int(c)] for c in row if int(c) in position_of_doc]
        for row in queries["candidates"]
    ]

    return corpus_texts, rephrased_texts, query_texts, doc_ids, candidate_positions


def ranks_both_regimes(corpus_scores, rephrased_scores, own_idx, self_pos,
                       restrict_row):
    """Both regimes from ONE set of scores.

    They differ only in what the rephrased copy is ranked against, never in
    the vectors themselves - so encoding 71,117 documents twice, or paying an
    embedding API twice for byte-identical vectors, would be pure waste.
    """
    out = {
        "all-documents": ranks_from_scores(
            corpus_scores, rephrased_scores, own_idx, self_pos, None
        )
    }
    if restrict_row is not None:
        out["per-query-candidates"] = ranks_from_scores(
            corpus_scores, rephrased_scores, own_idx, self_pos, restrict_row
        )
    return out


def ranks_from_scores(corpus_scores, rephrased_scores, own_idx, self_pos=None,
                      restrict=None):
    """Rank of the query's rephrased copy. Matches the published definition
    verbatim: "1 + number of original candidate vectors with strictly greater
    cosine similarity; exact ties receive the best tied rank".

    restrict, when given, is the set of corpus positions the copy competes
    against - the query's own candidate list under the published protocol.
    """
    own = rephrased_scores[own_idx]
    if restrict is not None:
        competitors = corpus_scores[restrict]
        beaten = competitors > own
        if self_pos is not None:
            beaten = beaten & (np.asarray(restrict) != self_pos)
    else:
        beaten = corpus_scores > own
        if self_pos is not None:
            beaten = beaten.copy()
            beaten[self_pos] = False
    rank_corpus_only = 1 + int(np.sum(beaten))
    others = np.delete(rephrased_scores, own_idx)
    rank_with_all = rank_corpus_only + int(np.sum(others > own))
    return rank_corpus_only, rank_with_all, float(own)


def summarize(ranks: list[int]) -> dict:
    """avg_rank is reported for continuity with the published rows, but it is
    NOT the statistic to read: the rank distribution has a long tail driven by
    degenerate query texts (one query is the 19-character string "Solve the
    equation:" with no equation in it), so a handful of queries move the mean
    by an order of magnitude. median_rank and the top_k coverage are the
    robust summaries."""
    arr = np.asarray(ranks, dtype=float)
    out = {
        "avg_rank": float(arr.mean()),
        "median_rank": float(np.median(arr)),
        "trimmed_mean_rank_95": float(arr[arr <= np.percentile(arr, 95)].mean()),
        "p90_rank": float(np.percentile(arr, 90)),
        "p99_rank": float(np.percentile(arr, 99)),
        "max_rank": float(arr.max()),
    }
    for k in TOP_K_LEVELS:
        out[f"top_{k}"] = float(np.mean(arr <= k))
    return out


def run_embedding(model_key, args, corpus_texts, rephrased_texts, query_texts, query_idxs, self_pos, restrict):
    processor = rr.build_processor(
        model_key, "p0", args.tensor_parallel_size, args.save_to
    )
    # Duck-typed, not isinstance(EmbeddingProcessor): what this path actually
    # requires is "one embedding per text", i.e. an encode(). A COMPOSED
    # processor satisfies that without inheriting from EmbeddingProcessor -
    # reason-rewriter-reason-embed-8b wraps a vLLM bi-encoder and exposes both
    # encode() (documents) and encode_queries() (queries, with the rewrite),
    # and INFXRetrieverProcessor is the same shape. An isinstance check
    # rejected those for the wrong reason: they are bi-encoders in every sense
    # this script cares about, they just are not subclasses. A genuine
    # cross-encoder still has no encode() and is still rejected here, and the
    # pairwise families never reach this function at all (PAIRWISE_MODELS).
    if not (
        isinstance(processor, EmbeddingProcessor)
        or callable(getattr(processor, "encode", None))
    ):
        raise SystemExit(
            f"{model_key} is not a bi-encoder: corpus-scale dedup ranking "
            "needs one embedding per text, not one forward pass per "
            "query-document pair. Only models exposing encode() (and lexical "
            "models) are supported."
        )

    scores_kwargs, _ = rr.prompt_scores_kwargs(model_key, None)
    # This script encodes whole corpora up front rather than going through
    # get_scores per query, so the per-side envelope has to be applied here:
    # the affix keys are get_scores parameters and would raise as unknown
    # kwargs inside llm.embed / SentenceTransformer.encode.
    affixes, encode_kwargs = split_affix_kwargs(scores_kwargs)
    encode_kwargs.pop("batch_size", None)

    query_side = dict(affixes.get("query_encode_kwargs") or {})
    document_side = dict(affixes.get("document_encode_kwargs") or {})
    if affixes:
        print(f"[~] Input envelope: {affixes}")

    corpus_texts = apply_affixes(
        corpus_texts, affixes.get("document_prompt"), affixes.get("document_suffix")
    )
    rephrased_texts = apply_affixes(
        rephrased_texts, affixes.get("document_prompt"), affixes.get("document_suffix")
    )
    query_inputs = apply_affixes(
        [query_texts[i] for i in query_idxs],
        affixes.get("query_prompt"),
        affixes.get("query_suffix"),
    )

    print(f"[~] Encoding {len(corpus_texts)} corpus documents...")
    corpus_emb = np.asarray(
        processor.encode(
            corpus_texts, show_progress_bar=True, **encode_kwargs, **document_side
        ),
        dtype=np.float32,
    )
    print(f"[~] Encoding {len(rephrased_texts)} rephrased insertions...")
    reph_emb = np.asarray(
        processor.encode(
            rephrased_texts, show_progress_bar=True, **encode_kwargs, **document_side
        ),
        dtype=np.float32,
    )
    # A processor whose query vector is NOT encode(query) - e.g. the
    # reason-rewriter composition, whose query vector is the mean of its
    # rewrites' embeddings - exposes encode_queries(). Calling plain encode()
    # on such a model returns bare-retriever vectors and would report them
    # under the composed model's name, so prefer the query-aware method
    # whenever it exists. Models without it are unaffected.
    encode_queries = getattr(processor, "encode_queries", None)
    if encode_queries is not None:
        print(
            f"[~] Encoding {len(query_inputs)} queries via "
            f"{type(processor).__name__}.encode_queries (query-side transform applied)..."
        )
        query_emb = np.asarray(
            encode_queries(
                query_inputs, show_progress_bar=True, **encode_kwargs, **query_side
            ),
            dtype=np.float32,
        )
    else:
        print(f"[~] Encoding {len(query_inputs)} queries...")
        query_emb = np.asarray(
            processor.encode(
                query_inputs,
                show_progress_bar=True,
                **encode_kwargs,
                **query_side,
            ),
            dtype=np.float32,
        )

    def normalize(m):
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise SystemExit("Zero-norm embedding encountered.")
        return m / norms

    corpus_emb = normalize(corpus_emb)
    reph_emb = normalize(reph_emb)
    query_emb = normalize(query_emb)

    per_query = {}
    for j, row in enumerate(query_idxs):
        corpus_scores = corpus_emb @ query_emb[j]
        rephrased_scores = reph_emb @ query_emb[j]
        per_query[row] = ranks_both_regimes(
            corpus_scores, rephrased_scores, row, self_pos.get(row),
            restrict[row] if restrict else None,
        )
    return per_query


def run_pairwise(
    model_key, args, corpus_texts, rephrased_texts, query_texts, query_idxs,
    self_pos, restrict,
):
    """Cross-encoders, late-interaction and sparse models score a (query,
    document) PAIR - there is no document vector to precompute and reuse, so
    the rephrased copy can only be ranked by scoring it alongside the query's
    own candidates. That is exactly the published per-query protocol, and it
    is the ONLY regime these models can run in: ranking against the whole
    corpus would mean 71,117 forward passes per query.

    Checkpointed after every query, because the generative rerankers take
    hours and this cluster reaps idle-GPU jobs.
    """
    if restrict is None:
        raise SystemExit(
            f"{model_key} scores query-document pairs, so it can only run "
            "with --corpus per-query-candidates. Ranking it against the full "
            "corpus would need one forward pass per document per query."
        )

    processor = rr.build_processor(
        model_key, "p0", args.tensor_parallel_size, args.save_to
    )
    scores_kwargs, _ = rr.prompt_scores_kwargs(model_key, None)

    checkpoint = (
        Path(args.save_to) / ".checkpoints"
        / f"{model_key}__pairwise{getattr(args, 'shard_tag', '')}.json"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    per_query = {}
    if checkpoint.exists():
        try:
            per_query = {
                int(k): v for k, v in json.loads(checkpoint.read_text()).items()
            }
            print(f"[~] Resuming from {checkpoint}: {len(per_query)} queries done.")
        except (json.JSONDecodeError, OSError):
            per_query = {}

    todo = [r for r in query_idxs if r not in per_query]
    print(f"[~] Scoring {len(todo)} queries x ~{len(restrict[query_idxs[0]]) + 1} pairs...")

    for n, row in enumerate(todo, start=1):
        positions = restrict[row]
        documents = [corpus_texts[p] for p in positions] + [rephrased_texts[row]]
        scores = np.asarray(
            processor.get_scores(
                query_texts[row], documents, show_progress_bar=False, **scores_kwargs
            ),
            dtype=np.float64,
        )
        own = float(scores[-1])
        beaten = scores[:-1] > own
        own_doc = self_pos.get(row)
        if own_doc is not None:
            beaten = beaten & (np.asarray(positions) != own_doc)
        # Inserting all 1000 rephrased copies would need 1000 extra forward
        # passes per query, so that reading is not computed for pair scorers.
        per_query[row] = {
            "per-query-candidates": (1 + int(np.sum(beaten)), None, own)
        }
        if n % 10 == 0 or n == len(todo):
            tmp = checkpoint.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({str(k): v for k, v in per_query.items()}))
            tmp.replace(checkpoint)
            print(f"    {n}/{len(todo)} queries", flush=True)

    return per_query


def run_lexical(model_key, args, corpus_texts, rephrased_texts, query_texts, query_idxs, self_pos, restrict):
    """Lexical dedup.

    The three lexical families each come in two tokenizer variants, exactly
    as in sabermath.registry.LEXICAL_MODEL_BUILDERS: the default keys tokenize
    with Approach Zero's operator-tree tokenizer over the LaTeX blocks plus
    lowercased prose words, and the "-no-tok" keys use the plain-text
    fallback (lowercase + whitespace split for BM25/Jaccard, sklearn's own
    regex tokenizer for TF-IDF). The variant has to be threaded through here
    rather than read off a processor because this script indexes the whole
    corpus once instead of rebuilding an index per query.
    """
    from sabermath.processors.tokenization_helper import math_word_tokens

    approach0 = not model_key.endswith("-no-tok")
    family = model_key if approach0 else model_key[: -len("-no-tok")]

    def tokenize(text: str) -> list[str]:
        if approach0:
            return math_word_tokens(text, lowercase=True)
        return text.lower().split()

    if family == "bm25":
        from rank_bm25 import BM25Okapi

        print(f"[~] Tokenizing {len(corpus_texts) + len(rephrased_texts)} documents...")
        corpus_tokens = [tokenize(t) for t in corpus_texts]
        reph_tokens = [tokenize(t) for t in rephrased_texts]
        index = BM25Okapi(corpus_tokens + reph_tokens)
        n_corpus = len(corpus_texts)

        per_query = {}
        for row in query_idxs:
            scores = np.asarray(index.get_scores(tokenize(query_texts[row])))
            per_query[row] = ranks_both_regimes(
                scores[:n_corpus], scores[n_corpus:], row, self_pos.get(row),
                restrict[row] if restrict else None,
            )
        return per_query

    if family == "tf-idf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        if approach0:
            vectorizer = TfidfVectorizer(
                tokenizer=math_word_tokens, token_pattern=None, lowercase=False
            )
        else:
            # No custom tokenizer at all, so sklearn's defaults apply: its
            # r"(?u)\b\w\w+\b" regex plus lowercasing. Matches
            # TfidfProcessor(tokenize_approach0=False).
            vectorizer = TfidfVectorizer()
        doc_matrix = vectorizer.fit_transform(corpus_texts + rephrased_texts)
        n_corpus = len(corpus_texts)

        per_query = {}
        for row in query_idxs:
            query_vec = vectorizer.transform([query_texts[row]])
            scores = cosine_similarity(query_vec, doc_matrix).ravel()
            per_query[row] = ranks_both_regimes(
                scores[:n_corpus], scores[n_corpus:], row, self_pos.get(row),
                restrict[row] if restrict else None,
            )
        return per_query

    if family == "jaccard":
        corpus_sets = [set(tokenize(t)) for t in corpus_texts]
        reph_sets = [set(tokenize(t)) for t in rephrased_texts]

        def jaccard(a, b):
            if not a and not b:
                return 0.0
            return len(a & b) / len(a | b)

        per_query = {}
        for row in query_idxs:
            q = set(tokenize(query_texts[row]))
            corpus_scores = np.asarray([jaccard(q, d) for d in corpus_sets])
            reph_scores = np.asarray([jaccard(q, d) for d in reph_sets])
            per_query[row] = ranks_both_regimes(
                corpus_scores, reph_scores, row, self_pos.get(row),
                restrict[row] if restrict else None,
            )
        return per_query

    raise SystemExit(
        f"{model_key} has no dedup path (approach0 rebuilds a native index "
        "per candidate list and its crash skip-list is keyed to benchmark "
        "queries)."
    )


def run_one_model(
    model_key: str,
    args,
    corpus_texts,
    rephrased_texts,
    query_texts,
    query_idxs,
    self_pos,
    restrict,
) -> None:
    """One model's full dedup measurement, written out per regime."""
    if model_key in PAIRWISE_MODELS:
        per_query = run_pairwise(
            model_key, args, corpus_texts, rephrased_texts, query_texts,
            query_idxs, self_pos, restrict,
        )
    elif model_key in rr.LEXICAL_MODEL_BUILDERS:
        per_query = run_lexical(
            model_key, args, corpus_texts, rephrased_texts, query_texts,
            query_idxs, self_pos, restrict,
        )
    else:
        per_query = run_embedding(
            model_key, args, corpus_texts, rephrased_texts, query_texts,
            query_idxs, self_pos, restrict,
        )

    subset = "" if args.n is None else f"__n{args.n}_seed{args.seed}"
    subset = f"{subset}{getattr(args, 'shard_tag', '')}"
    if args.keep_self_match:
        subset = f"{subset}__selfmatch"

    regimes = (
        ["per-query-candidates", "all-documents"]
        if args.corpus == "both"
        else [args.corpus]
    )
    available = set().union(*(set(v) for v in per_query.values()))

    for regime in regimes:
        if regime not in available:
            print(f"[~] {regime}: not available for {model_key} - skipped.")
            continue

        # len(self_pos) counts queries whose own document exists ANYWHERE in
        # the 71,117-document corpus, which is the right number only for the
        # all-documents regime. In the per-query regime a query competes only
        # against its own 150 candidates, and a query's own document is never
        # one of them (measured: 0 of 928), so the flag changes nothing there.
        # Reporting the global count per regime read as if 928 competitors had
        # been removed from a 151-vector pool, so report what was actually
        # excluded under this regime.
        if regime == "all-documents" or restrict is None:
            n_excluded = len(self_pos)
        else:
            n_excluded = sum(
                1 for row, pos in self_pos.items() if pos in set(restrict[row])
            )

        ranks = [per_query[i][regime][0] for i in query_idxs]
        with_all = [per_query[i][regime][1] for i in query_idxs]

        out = {
            "model": model_key,
            "regime": regime,
            "protocol": "canonical",
            "exclude_self_match": not args.keep_self_match,
            "n_self_matches_excluded": n_excluded,
            "rephrased_dataset": args.rephrased,
            "doc_version": args.doc_version,
            "query_version": args.query_version,
            "corpus_size": (
                len(corpus_texts)
                if regime == "all-documents"
                else int(np.median([len(r) for r in restrict]))
            ),
            "ranked_vectors_per_query": (
                len(corpus_texts) + 1
                if regime == "all-documents"
                else int(np.median([len(r) for r in restrict])) + 1
            ),
            "n_queries": len(query_idxs),
            "insert_per_query": summarize(ranks),
            "insert_all_rephrased": (
                summarize(with_all) if all(v is not None for v in with_all) else None
            ),
            "per_query": {
                str(i): {
                    "rank_corpus_only": per_query[i][regime][0],
                    "rank_with_all_rephrased": per_query[i][regime][1],
                    "own_score": per_query[i][regime][2],
                }
                for i in query_idxs
            },
        }

        tag = "" if regime == "per-query-candidates" else "__all-documents"
        out_path = Path(args.save_to) / f"{model_key}{tag}{subset}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))

        print(f"\n=== {model_key} dedup ranking [{regime}] ===")
        for mode in ("insert_per_query", "insert_all_rephrased"):
            m = out[mode]
            if m is None:
                print(f"  {mode}: not computed for pair scorers")
                continue
            tops = " ".join(f"top{k}={m[f'top_{k}']:.1%}" for k in TOP_K_LEVELS[:5])
            print(
                f"  {mode}: median={m['median_rank']:.1f} "
                f"trimmed-mean(95%)={m['trimmed_mean_rank_95']:.2f} "
                f"avg={m['avg_rank']:.2f} (p90={m['p90_rank']:.0f} "
                f"max={m['max_rank']:.0f})\n    {tops}"
            )
        print(f"[+] Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="KEY",
        default=None,
        help="Models to run (default: every model that can do dedup - all "
        "bi-encoders, lexical baselines and pair scorers in the registry).",
    )
    parser.add_argument("--rephrased", type=str, default=DEFAULT_REPHRASED_DATASET)
    parser.add_argument(
        "--doc-version", choices=["statement", "full"], default="statement"
    )
    parser.add_argument(
        "--query-version", choices=["statement", "full"], default="statement"
    )
    parser.add_argument(
        "--corpus",
        choices=["both", "per-query-candidates", "all-documents"],
        default="both",
        help="Which regime(s) to write out. per-query-candidates is the "
        "PUBLISHED protocol: the copy is inserted into that query's own 150 "
        "candidates and ranked among 151. all-documents ranks it against all "
        "71,117 documents - a far harder question with no published "
        "reference. Both are computed from the SAME embeddings in one pass, "
        "so 'both' (the default) costs no more than either alone.",
    )
    parser.add_argument("--save-to", type=str, default="results/dedup")
    parser.add_argument(
        "--keep-self-match",
        action="store_true",
        help="Do NOT exclude each query's own original document from the "
        "corpus. 92.8%% of query problems appear verbatim among the "
        "documents, so leaving them in measures exact-duplicate retrieval "
        "instead of rephrase retrieval - see self_document_positions().",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--query-shards",
        type=int,
        default=None,
        help="Split the queries into this many strided shards so a slow model "
        "(the generative rerankers take hours per 1000 queries) can run as N "
        "concurrent jobs. Each shard keeps its OWN checkpoint and output file; "
        "stitch them back with --merge-shards.",
    )
    parser.add_argument(
        "--query-shard", type=int, default=None, help="Which shard (0-based)."
    )
    add_merge_arguments(parser, "results/dedup")
    args = parser.parse_args()

    # Before load_data(): merging is pure JSON arithmetic over finished runs.
    if args.merge_shards is not None:
        run_merge(args, merge_dedup_shards, Path(args.save_to))
        return

    queries, documents = load_data()
    rephrased = load_dataset(args.rephrased, split="train")

    corpus_texts, rephrased_texts, query_texts, doc_ids, restrict = build_texts(
        queries, documents, rephrased, args.doc_version, args.query_version, args.corpus
    )
    sizes = {len(r) for r in restrict}
    print(
        f"[+] {len(corpus_texts)} documents encoded ({args.doc_version} "
        f"version) + {len(rephrased_texts)} rephrased insertions. Ranked both "
        f"among each query's own candidates ({min(sizes)}-{max(sizes)} of "
        f"them, the published protocol) and against the full corpus."
    )

    if (args.query_shards is None) != (args.query_shard is None):
        raise SystemExit("--query-shards and --query-shard must be used together.")
    if args.query_shards is not None and not 0 <= args.query_shard < args.query_shards:
        raise SystemExit(f"--query-shard must be in [0, {args.query_shards - 1}].")

    query_idxs = list(range(len(query_texts)))
    if args.n is not None:
        rng = random.Random(args.seed)
        query_idxs = sorted(rng.sample(query_idxs, min(args.n, len(query_idxs))))
    # Strided, not contiguous - the same split run_experiments uses (see
    # sabermath.runner.select_shard), so every shard sees a mix of domains
    # and difficulties rather than one block of the query list.
    args.shard_tag = ""
    if args.query_shards is not None:
        query_idxs = [
            r for k, r in enumerate(query_idxs) if k % args.query_shards == args.query_shard
        ]
        if not query_idxs:
            raise SystemExit("That shard selected no queries.")
        args.shard_tag = f"__shard{args.query_shard}of{args.query_shards}"
        print(f"[~] Query shard {args.query_shard}/{args.query_shards}: {len(query_idxs)} queries.")

    self_pos = {} if args.keep_self_match else self_document_positions(
        queries, documents, doc_ids
    )
    print(
        f"[+] Self-match exclusion: "
        + (
            "OFF (--keep-self-match)"
            if args.keep_self_match
            else f"{len(self_pos)}/{len(query_texts)} queries have their own "
            "original document in the corpus; it is excluded as a competitor"
        )
    )

    # Every model can do dedup EXCEPT approach0: its _BROKEN_QUERIES md5
    # skip-list is keyed to raw benchmark query text, and a rephrased
    # insertion is not in it, so it would segfault on the very rows this
    # experiment is about.
    eligible = [k for k in rr.ALL_MODEL_KEYS if k not in DEDUP_EXCLUDED]
    models = args.models or eligible
    excluded = [m for m in models if m in DEDUP_EXCLUDED]
    if excluded:
        raise SystemExit(
            "; ".join(f"{m}: {DEDUP_EXCLUDED[m]}" for m in excluded)
        )
    unknown = [m for m in models if m not in rr.ALL_MODEL_KEYS]
    if unknown:
        raise SystemExit(
            f"Unknown model key(s): {', '.join(unknown)}. "
            "See scripts/run_experiments.py --list for the available keys."
        )

    failures = []
    for i, model_key in enumerate(models, start=1):
        print("\n" + "#" * 60)
        print(f"# dedup: {model_key} ({i}/{len(models)})")
        print("#" * 60)
        try:
            run_one_model(
                model_key,
                args,
                corpus_texts,
                rephrased_texts,
                query_texts,
                query_idxs,
                self_pos,
                restrict,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[!!] {model_key} failed: {e}")
            failures.append(model_key)

    print("\n" + "=" * 60)
    print(f"Done. {len(models) - len(failures)}/{len(models)} succeeded.")
    if failures:
        print(f"Failed: {', '.join(failures)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
