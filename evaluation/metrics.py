import pandas as pd


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


def mean_absolute_error(actual_values, predicted_values):
    """
    Calculates MAE.

    MAE = mean absolute difference between actual and predicted ratings.
    """
    if len(actual_values) == 0 or len(predicted_values) == 0:
        return None

    actual_series = pd.Series(actual_values, dtype="float")
    predicted_series = pd.Series(predicted_values, dtype="float")

    if len(actual_series) != len(predicted_series):
        return None

    return float((actual_series - predicted_series).abs().mean())