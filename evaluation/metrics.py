def precision_at_k(recommended_ids, relevant_ids, k: int) -> float:
    """
    Calculates Precision@K.

    Precision@K = relevant recommended items / K
    """
    if k <= 0:
        return 0.0

    recommended_at_k = list(recommended_ids)[:k]

    if not recommended_at_k:
        return 0.0

    relevant_set = set(relevant_ids)

    hits = sum(1 for movie_id in recommended_at_k if movie_id in relevant_set)

    return hits / k


def recall_at_k(recommended_ids, relevant_ids, k: int) -> float:
    """
    Calculates Recall@K.

    Recall@K = relevant recommended items / all relevant items
    """
    relevant_set = set(relevant_ids)

    if not relevant_set:
        return 0.0

    recommended_at_k = list(recommended_ids)[:k]

    hits = sum(1 for movie_id in recommended_at_k if movie_id in relevant_set)

    return hits / len(relevant_set)