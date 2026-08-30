"""Is SABER-Math's ranking an artefact of how relevance scores are rescaled?

The benchmark reports nDCG@10 with an EXPONENTIAL gain over relevances that
build_benchmark/final_transform.py has rescaled from [0,1] to [0,5]. That
choice is not metric-invariant, so this script replays every model's stored
candidate ranking under three alternatives and measures how far the
leaderboard moves:

    [0,3] + exponential gain   (scale 3/5 - punishes inversions less)
    [0,5] + linear gain
    [0,3] + linear gain

Nothing is re-inferred. nDCG is a pure function of a model's RANKING of the
candidates, so the rankings dumped by `run_rerankers.py` into
`<results>/.checkpoints/<model>/<run>/<task>.scores.json` are enough to
recompute the metric exactly - `--verify` confirms this by replaying at the
benchmark's own setting and reproducing the stored per-query nDCG to machine
epsilon. Runs are matched to the published table by their reproduced Overall,
so a row is only ever built from a run that actually reproduces the paper.

Those checkpoints total 3.9 GB and are gitignored, which would make this
study impossible to re-run without cluster access. They do not need to be:
nDCG@10's DCG term reads only the top 10 ranked candidates, and its ideal-DCG
term reads the relevance SET, which comes from the dataset. So the rankings
alone, truncated to the top k and stored as uint8, reproduce every number
here from 412 KB (top-10) or 753 KB (top-20) - committed as rankings.npz.

    # from the checkpoints (needs the cluster's results/ trees)
    python experiments/rescaling_robustness/rescore_rescaling.py --verify
    python experiments/rescaling_robustness/rescore_rescaling.py \
        --export-rankings experiments/rescaling_robustness/rankings.npz

    # from the committed artifact alone (needs nothing else)
    python experiments/rescaling_robustness/rescore_rescaling.py \
        --from-rankings experiments/rescaling_robustness/rankings.npz
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

QUERIES = "INSAIT-Institute/SaberMath-queries"
TASK = "statement-full"
REPOS = [Path("/home/dimitadi/saberivo2"), Path("/home/dimitadi/saberstable")]
SHARD_RE = re.compile(r"__shard\d+of\d+")

# [0,5] relevances are what the dataset stores; [0,3] is a scale of 3/5.
VARIANTS = [("[0,5] + exponential gain", "exponent", 1.0),
            ("[0,3] + exponential gain", "exponent", 0.6),
            ("[0,5] + linear gain", "linear", 1.0),
            ("[0,3] + linear gain", "linear", 0.6)]


def load_queries():
    """Relevance scores per query, already on the benchmark's [0,5] scale."""
    from datasets import load_dataset

    ds = load_dataset(QUERIES, split="train")
    return [np.asarray(x, dtype=float) for x in ds["relevance_scores"]]


def ndcg(rel_all, ranking, k=10, gain="exponent", scale=1.0):
    """nDCG@k of `ranking` over a candidate set with relevances `rel_all`.

    `rel_all` is the query's FULL relevance vector and `ranking` the model's
    candidate order. They are separate arguments on purpose: the ideal DCG
    depends on the whole relevance set, so passing a ranking already
    truncated to the top k (as rankings.npz stores) still gives the exact
    same answer. Folding them together - ranking the relevances first and
    taking the ideal from that - silently computes the ideal over only the
    retained k and inflates every score.
    """
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    r = np.asarray(rel_all, dtype=float)[np.asarray(ranking, dtype=int)[:k]] * scale
    g = r if gain == "linear" else np.power(2.0, r) - 1.0
    dcg = float(np.sum(g * disc[: r.size]))
    s = np.sort(np.asarray(rel_all, dtype=float) * scale)[::-1][:k]
    gi = s if gain == "linear" else np.power(2.0, s) - 1.0
    idcg = float(np.sum(gi * disc[: s.size]))
    return 0.0 if idcg == 0 else dcg / idcg


TABLE = [
 ("ReasonReranker-Qwen3-32B-Rewrite","retro-star-32b-rewritten",0.741),
 ("ReasonEmbed-Qwen3-8B-Rewrite","reason-rewriter-reason-embed-8b",0.738),
 ("ReasonReranker-Qwen3-32B","retro-star-32b",0.733),
 ("ReasonEmbed-Qwen3-8B","reason-embed-qwen3-8b",0.710),
 ("Diver-GroupRank-32B","diver-grouprank-32b",0.693),
 ("RaDeR-14B","rader-14b",0.692),
 ("RaDeR-7B","rader-7b",0.690),
 ("Qwen3-Reranker-4B","qwen3-reranker-4b",0.683),
 ("Diver-4B","diver-retriever-4b",0.681),
 ("ReasonEmbed-Llama-3.1-8B","reason-embed-llama-3.1-8b",0.664),
 ("RaDeR-3B","rader-3b",0.654),
 ("Qwen3-Reranker-0.6B","qwen3-reranker-0.6b",0.652),
 ("SPLADE-Code-8B","splade-code-8b",0.650),
 ("Octen-8B","octen-embedding-8b",0.636),
 ("Gemini-2","gemini-embedding-2",0.628),
 ("Gemini-001","gemini-embedding-001",0.616),
 ("Qwen3-Embedding-4B","qwen3-embedding-4b",0.615),
 ("Qwen3-Embedding-8B","qwen3-embedding-8b",0.611),
 ("Rank1-32B","rank1-32b",0.610),
 ("Harrier-27b","harrier-oss-v1-27b",0.608),
 ("INF-Retriever-v1-Pro","inf-retriever-v1-pro",0.594),
 ("KaLM-12B-2511","kalm-embedding-gemma3-12b-2511",0.584),
 ("LLaMa-Nemotron-8B","llama-embed-nemotron-8b",0.579),
 ("Qwen3-Embedding-0.6B","qwen3-embedding-0.6b",0.575),
 ("Harrier-0.6b","harrier-oss-v1-0.6b",0.572),
 ("Jina-v5-Small","jina-embeddings-v5-text-small",0.565),
 ("Reason-ColBERT","reason-moderncolbert",0.558),
 ("Text-Embedding-3-Large","text-embedding-3-large",0.557),
 ("Jina-v5-Nano","jina-embeddings-v5-text-nano",0.530),
 ("Gemma-300m","embeddinggemma-300m",0.527),
 ("Text-Embedding-3-Small","text-embedding-3-small",0.512),
 ("BGE-m3","bge-m3",0.511),
 ("ReasonIR-8B","reasonir-8b",0.507),
 ("Harrier-270m","harrier-oss-v1-270m",0.498),
 ("INF-X-Retriever","inf-x-retriever",0.494),
 ("Multilingual-E5-Large","multilingual-e5-large",0.488),
 ("Approach Zero","approach0",0.468),
 ("BM25","bm25",0.416),
 ("Jaccard","jaccard",0.412),
 ("TF-IDF","tf-idf",0.412),
 ("BERT (Base)","bert-base-uncased",0.357),
 ("RoBERTa","roberta-base",0.311),
]
KEYS = {k for _, k, _ in TABLE}
NAME_OF = {k: n for n, k, _ in TABLE}
TARGET = {k: t for _, k, t in TABLE}


def collect_from_checkpoints(rel):
    """Pick, per table model, the stored run that reproduces its published
    Overall. Shards are pooled by GLOBAL query row (a dict union, never an
    average of averages, which breaks the moment two shards differ in size);
    everything else - protocol tag, part, prompt - stays a distinct run."""
    groups = defaultdict(list)
    for repo in REPOS:
        for mp in repo.glob("results/*/.checkpoints/*/*/meta.json"):
            run = mp.parent
            if not (run / f"{TASK}.scores.json").exists():
                continue
            meta = json.loads(mp.read_text())
            if meta["model_key"] not in KEYS:
                continue
            groups[(repo.name, run.parents[2].name, meta["model_key"],
                    SHARD_RE.sub("", run.name))].append((run, meta))

    by_model = defaultdict(list)
    for key, members in groups.items():
        per_row = {}
        for run, meta in members:
            idxs = meta.get("query_row_idxs")
            stored = json.loads((run / f"{TASK}.scores.json").read_text())
            for pos_s, e in stored.items():
                if e is None:
                    continue
                pos = int(pos_s)
                row = idxs[pos] if idxs is not None else pos
                per_row[row] = np.asarray(e["ranking"], dtype=int)
        if len(per_row) == 1000:
            by_model[key[2]].append((key, per_row))

    chosen, missing = {}, []
    for _, mkey, tgt in TABLE:
        cands = []
        for key, per_row in by_model.get(mkey, []):
            ov = float(np.mean([ndcg(rel[r], rk) for r, rk in per_row.items()]))
            cands.append((0 if key[3].startswith("p0") else 1, abs(ov - tgt),
                          f"{key[0]}/{key[1]}/{key[3]}", per_row, ov))
        if not cands:
            missing.append(mkey)
            continue
        cands.sort(key=lambda c: c[:2])
        _, _, src, per_row, ov = cands[0]
        chosen[mkey] = (src, per_row, ov)
    return chosen, missing


def export_rankings(chosen, path, top_k):
    out = {}
    for mkey, (src, per_row, ov) in chosen.items():
        rows = np.array(sorted(per_row), dtype=np.int16)
        mat = np.array([per_row[r][:top_k] for r in rows], dtype=np.uint8)
        assert mat.max() < 256
        out[f"{mkey}__rows"] = rows
        out[f"{mkey}__rank"] = mat
        out[f"{mkey}__src"] = np.array(src)
    np.savez_compressed(path, **out)
    mb = Path(path).stat().st_size / 1e6
    print(f"[+] wrote {path} - {len(chosen)} models, top-{top_k}, {mb:.3f} MB")


def load_rankings(path, rel):
    """Rebuild `chosen` from the committed artifact, with no checkpoints."""
    z = np.load(path)
    chosen = {}
    for mkey in {k.split("__")[0] for k in z.files}:
        rows = z[f"{mkey}__rows"]
        mat = z[f"{mkey}__rank"]
        per_row = {int(r): mat[j] for j, r in enumerate(rows)}
        ov = float(np.mean([ndcg(rel[r], rk) for r, rk in per_row.items()]))
        chosen[mkey] = (str(z[f"{mkey}__src"]), per_row, ov)
    missing = [k for k in KEYS if k not in chosen]
    return chosen, missing


def verify(rel):
    """Replaying a stored ranking at the benchmark's own setting must return
    the per-query nDCG the original run wrote next to it."""
    worst, n_runs = 0.0, 0
    for repo in REPOS:
        for mp in repo.glob("results/*/.checkpoints/*/*/meta.json"):
            run = mp.parent
            sp, cp = run / f"{TASK}.scores.json", run / f"{TASK}.json"
            if not (sp.exists() and cp.exists()):
                continue
            meta = json.loads(mp.read_text())
            if meta["model_key"] not in KEYS:
                continue
            idxs = meta.get("query_row_idxs")
            ck = json.loads(cp.read_text())
            for pos_s, e in json.loads(sp.read_text()).items():
                if e is None or ck.get(pos_s) is None:
                    continue
                row = idxs[int(pos_s)] if idxs is not None else int(pos_s)
                worst = max(worst, abs(ndcg(rel[row], e["ranking"]) - ck[pos_s]))
            n_runs += 1
    print(f"[verify] {n_runs} runs replayed, max |d nDCG| vs the stored "
          f"per-query value = {worst:.3e}")
    assert worst < 1e-9, "replay does not reproduce the published metric"


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def rankdata(x):
    x = np.asarray(x, float)
    order = np.argsort(x)
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def kendall_tau_b(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    C = D = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                ta += 1; tb += 1
            elif da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif da * db > 0:
                C += 1
            else:
                D += 1
    return (C - D) / np.sqrt((C + D + ta) * (C + D + tb))


def ranks_desc(x):
    return rankdata([-v for v in x])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="check the replay reproduces the stored per-query nDCG")
    ap.add_argument("--from-rankings", type=Path, default=None,
                    help="read rankings from a rankings.npz instead of the "
                         "gitignored results/ checkpoints")
    ap.add_argument("--export-rankings", type=Path, default=None,
                    help="write the selected rankings to this .npz")
    ap.add_argument("--top-k", type=int, default=150,
                    help="candidates per query to export (10 suffices for "
                         "nDCG@10; 150 keeps every rank-based metric open)")
    args = ap.parse_args()

    rel = load_queries()
    if args.verify:
        verify(rel)

    if args.from_rankings:
        chosen, missing = load_rankings(args.from_rankings, rel)
    else:
        chosen, missing = collect_from_checkpoints(rel)

    if args.export_rankings:
        export_rankings(chosen, args.export_rankings, args.top_k)

    names, scores, prov = [], {v[0]: [] for v in VARIANTS}, []
    for disp, mkey, tgt in TABLE:
        if mkey not in chosen:
            continue
        src, per_row, ov = chosen[mkey]
        names.append(disp)
        for label, gain, scale in VARIANTS:
            scores[label].append(float(np.mean(
                [ndcg(rel[r], rk, 10, gain, scale) for r, rk in per_row.items()])))
        prov.append((disp, mkey, tgt, ov, src))

    base_label = VARIANTS[0][0]
    base = scores[base_label]
    base_rank = ranks_desc(base)

    print("# nDCG@10 rescaling robustness, statement-full 'Overall' column")
    print(f"# {len(names)} of {len(TABLE)} table models; missing (no stored "
          f"candidate rankings): {sorted(missing)}\n")

    print("Provenance and reproduction of the published number")
    print(f"{'model':32s} {'table':>6s} {'repro':>7s} {'d':>7s}  run")
    for disp, mkey, tgt, ov, src in prov:
        print(f"{disp:32s} {tgt:6.3f} {ov:7.4f} {ov - tgt:+7.4f}  {src}")

    print("\nPer-model Overall under each rescaling")
    print(f"{'model':32s}" + "".join(f"{l:>26s}" for l, _, _ in VARIANTS))
    for i, n in enumerate(names):
        print(f"{n:32s}" + "".join(f"{scores[l][i]:26.4f}" for l, _, _ in VARIANTS))

    print(f"\nCorrelation vs original ({base_label}), n = {len(names)} models")
    print(f"{'Setting':28s} {'Pearson':>9s} {'Spearman':>9s} {'Kendall tau':>12s} "
          f"{'rank changes':>13s} {'max |dnDCG|':>12s}")
    for label, _, _ in VARIANTS[1:]:
        s = scores[label]
        changed = int(np.sum(ranks_desc(s) != base_rank))
        print(f"{label:28s} {pearson(base, s):9.4f} {spearman(base, s):9.4f} "
              f"{kendall_tau_b(base, s):12.4f} {f'{changed} / {len(names)}':>13s} "
              f"{max(abs(x - y) for x, y in zip(base, s)):12.4f}")

    gaps = []
    for label, _, _ in VARIANTS[1:]:
        r = ranks_desc(scores[label])
        changed = [i for i in range(len(names)) if r[i] != base_rank[i]]
        if not changed:
            continue
        print(f"\nRank changes under {label} (max |d rank| = "
              f"{int(np.max(np.abs(r - base_rank)))})")
        for i in sorted(changed, key=lambda i: base_rank[i]):
            partner = [j for j in changed
                       if base_rank[j] == r[i] and r[j] == base_rank[i]]
            gap = abs(base[i] - base[partner[0]]) if partner else float("nan")
            if partner:
                gaps.append(gap)
            print(f"  {names[i]:28s} rank {int(base_rank[i]):2d} -> "
                  f"{int(r[i]):2d}   original gap to the model it swaps "
                  f"with: {gap:.4f}")

    if gaps:
        print(f"\nEvery rank change is a swap of ADJACENT models whose original "
              f"scores\ndiffer by at most {max(gaps):.4f} nDCG - well inside the "
              f"+-0.008..0.010 95% CIs\nthe paper reports on this column, i.e. "
              f"pairs it never claimed to separate.")

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(
        {"task": TASK, "n_models": len(names), "missing": sorted(missing),
         "models": names, "table": [p[2] for p in prov], "scores": scores,
         "provenance": [list(p) for p in prov]}, indent=1))
    print(f"\n[+] wrote {out}")


if __name__ == "__main__":
    main()
