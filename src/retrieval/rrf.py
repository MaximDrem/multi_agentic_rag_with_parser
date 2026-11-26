from collections.abc import Mapping, Sequence

from retrieval.contracts import MAX_FUSION_CANDIDATES, RRF_K


def rrf_merge(
    rankings: Mapping[str, Sequence[str]],
    weights: Mapping[str, float] | None = None,
    k: int = RRF_K,
    limit: int = MAX_FUSION_CANDIDATES,
) -> list[dict]:
    weights = dict(weights or {name: 1.0 for name in rankings})
    scores: dict[str, float] = {}
    source_ranks: dict[str, dict[str, int]] = {}
    for source, ranking in rankings.items():
        seen = set()
        rank = 0
        for chunk_id in ranking:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + float(weights.get(source, 0.0)) / (k + rank)
            source_ranks.setdefault(chunk_id, {})[source] = rank
    ordered = sorted(scores, key=lambda chunk_id: -scores[chunk_id])[:limit]
    return [
        {
            "chunk_id": chunk_id,
            "score": scores[chunk_id],
            "rank": rank,
            "source_ranks": source_ranks[chunk_id],
        }
        for rank, chunk_id in enumerate(ordered, 1)
    ]


def weighted_rrf(
    dense_ranking: Sequence[str],
    bm25_ranking: Sequence[str],
    alpha: float = 0.5,
    k: int = RRF_K,
    limit: int = MAX_FUSION_CANDIDATES,
) -> list[dict]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return rrf_merge(
        {"dense": dense_ranking, "bm25": bm25_ranking},
        {"dense": alpha, "bm25": 1.0 - alpha},
        k,
        limit,
    )
