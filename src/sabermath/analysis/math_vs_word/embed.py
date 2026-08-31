def get_top5_candidates(target: dict):
    scores = target["relevance_scores"]
    cands = target["candidates"]
    sorted_pairs = sorted(zip(scores, cands))
    return [cand for _, cand in sorted_pairs[-5:]]
