
import math


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    results = retrieved_ids[:k]

    if not results:
        return 0.0

    hits = sum(chunk_id in relevant for chunk_id in results)

    # Standard Precision@K uses K as the denominator.
    return hits / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    results = retrieved_ids[:k]
    hits = sum(chunk_id in relevant for chunk_id in results)

    return hits / len(relevant)


def reciprocal_rank_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    # Binary relevance: 1 if relevant, otherwise 0.
    actual_relevance = [
        1 if chunk_id in relevant else 0
        for chunk_id in retrieved_ids[:k]
    ]

    def dcg(relevance_scores: list[int]) -> float:
        return sum(
            score / math.log2(rank + 1)
            for rank, score in enumerate(relevance_scores, start=1)
        )

    actual_dcg = dcg(actual_relevance)

    ideal_relevance = [1] * min(len(relevant), k)
    ideal_dcg = dcg(ideal_relevance)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg
