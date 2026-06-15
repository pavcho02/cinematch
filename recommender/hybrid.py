import pandas as pd


def normalize_score(series: pd.Series) -> pd.Series:
    """
    Normalizes a score column to the range [0, 1].

    If all values are the same, returns 1.0 for non-empty values.
    """
    if series.empty:
        return series

    min_value = series.min()
    max_value = series.max()

    if pd.isna(min_value) or pd.isna(max_value):
        return series.fillna(0)

    if max_value == min_value:
        return pd.Series([1.0] * len(series), index=series.index)

    return (series - min_value) / (max_value - min_value)


def prepare_recommendation_source(
    recommendations: pd.DataFrame,
    score_column: str,
    normalized_score_column: str,
    source_name: str
) -> pd.DataFrame:
    """
    Prepares recommendations from a single algorithm for hybrid merging.

    Each algorithm can have a different score column:
        - content_score
        - genre_score
        - predicted_rating
        - item_cf_score

    This function normalizes that score and keeps the common movie columns.
    """
    if recommendations is None or recommendations.empty:
        return pd.DataFrame()

    prepared = recommendations.copy()

    if score_column not in prepared.columns:
        return pd.DataFrame()

    prepared[normalized_score_column] = normalize_score(
        prepared[score_column].astype(float)
    )

    prepared["source"] = source_name

    columns_to_keep = [
        "movieId",
        "title",
        "clean_title",
        "year",
        "genres",
        normalized_score_column,
        "source"
    ]

    optional_columns = [
        "avg_rating",
        "rating_count",
        "matched_genres_text",
        "reason_signals",
        "support_count",
        "predicted_rating",
        "item_cf_score",
        "content_score",
        "genre_score"
    ]

    for column in optional_columns:
        if column in prepared.columns and column not in columns_to_keep:
            columns_to_keep.append(column)

    return prepared[columns_to_keep]


def combine_hybrid_recommendations(
    content_recommendations: pd.DataFrame,
    genre_recommendations: pd.DataFrame,
    user_based_recommendations: pd.DataFrame,
    item_based_recommendations: pd.DataFrame,
    top_n: int = 10,
    content_weight: float = 0.35,
    genre_weight: float = 0.20,
    user_based_weight: float = 0.25,
    item_based_weight: float = 0.20
) -> pd.DataFrame:
    """
    Combines recommendations from multiple algorithms into one hybrid ranking.

    The final score is a weighted sum of normalized scores.
    """
    prepared_sources = []

    content_prepared = prepare_recommendation_source(
        recommendations=content_recommendations,
        score_column="content_score",
        normalized_score_column="content_norm",
        source_name="content_based"
    )

    genre_prepared = prepare_recommendation_source(
        recommendations=genre_recommendations,
        score_column="genre_score",
        normalized_score_column="genre_norm",
        source_name="genre_based"
    )

    user_based_prepared = prepare_recommendation_source(
        recommendations=user_based_recommendations,
        score_column="predicted_rating",
        normalized_score_column="user_based_norm",
        source_name="user_based_cf"
    )

    item_based_prepared = prepare_recommendation_source(
        recommendations=item_based_recommendations,
        score_column="item_cf_score",
        normalized_score_column="item_based_norm",
        source_name="item_based_cf"
    )

    for prepared in [
        content_prepared,
        genre_prepared,
        user_based_prepared,
        item_based_prepared
    ]:
        if not prepared.empty:
            prepared_sources.append(prepared)

    if not prepared_sources:
        return pd.DataFrame()

    all_recommendations = pd.concat(prepared_sources, ignore_index=True)

    score_columns = [
        "content_norm",
        "genre_norm",
        "user_based_norm",
        "item_based_norm"
    ]

    for column in score_columns:
        if column not in all_recommendations.columns:
            all_recommendations[column] = 0.0

    all_recommendations[score_columns] = all_recommendations[score_columns].fillna(0.0)

    grouped = (
        all_recommendations
        .groupby("movieId", as_index=False)
        .agg(
            title=("title", "first"),
            clean_title=("clean_title", "first"),
            year=("year", "first"),
            genres=("genres", "first"),
            content_norm=("content_norm", "max"),
            genre_norm=("genre_norm", "max"),
            user_based_norm=("user_based_norm", "max"),
            item_based_norm=("item_based_norm", "max"),
            sources=("source", lambda values: ", ".join(sorted(set(values))))
        )
    )

    grouped["hybrid_score"] = (
        content_weight * grouped["content_norm"] +
        genre_weight * grouped["genre_norm"] +
        user_based_weight * grouped["user_based_norm"] +
        item_based_weight * grouped["item_based_norm"]
    )

    grouped["algorithm_count"] = grouped[
        [
            "content_norm",
            "genre_norm",
            "user_based_norm",
            "item_based_norm"
        ]
    ].gt(0).sum(axis=1)

    grouped = grouped.sort_values(
        by=["hybrid_score", "algorithm_count"],
        ascending=[False, False]
    )

    return grouped.head(top_n)