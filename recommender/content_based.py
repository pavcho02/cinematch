import math
import time
from typing import Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_content_dataframe(
    movies: pd.DataFrame,
    tags: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Builds a movie dataframe with enriched content text.

    The content text is used as input for TF-IDF.
    It combines:
        - clean title
        - genres
        - MovieLens tags, if available
    """
    content_movies = movies.copy()

    if "content_features" not in content_movies.columns:
        content_movies["content_features"] = (
            content_movies["clean_title"].fillna("") + " " +
            content_movies["genres"].fillna("").str.replace("|", " ", regex=False)
        )

    if tags is not None and not tags.empty:
        tags_aggregated = (
            tags
            .dropna(subset=["tag"])
            .groupby("movieId")["tag"]
            .apply(lambda values: " ".join(values.astype(str)))
            .reset_index()
            .rename(columns={"tag": "tag_features"})
        )

        content_movies = content_movies.merge(
            tags_aggregated,
            on="movieId",
            how="left"
        )

        content_movies["tag_features"] = content_movies["tag_features"].fillna("")

        content_movies["content_text"] = (
            content_movies["content_features"].fillna("") + " " +
            content_movies["tag_features"].fillna("")
        )
    else:
        content_movies["content_text"] = content_movies["content_features"].fillna("")

    return content_movies


def create_tfidf_matrix(content_movies: pd.DataFrame):
    """
    Creates a TF-IDF matrix from movie content text.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(content_movies["content_text"])

    return vectorizer, tfidf_matrix


def calculate_cosine_similarity(tfidf_matrix):
    """
    Calculates cosine similarity between all movies.
    """
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_similar_movies(
    movie_id: int,
    content_movies: pd.DataFrame,
    cosine_sim_matrix,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Returns movies similar to the selected movie based on cosine similarity.
    """
    movie_indices = pd.Series(
        content_movies.index,
        index=content_movies["movieId"]
    )

    if movie_id not in movie_indices:
        return pd.DataFrame()

    movie_index = movie_indices[movie_id]

    similarity_scores = list(enumerate(cosine_sim_matrix[movie_index]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda item: item[1],
        reverse=True
    )

    # skip first result because it is the same movie
    similarity_scores = similarity_scores[1:top_n + 1]

    similar_movie_indices = [item[0] for item in similarity_scores]
    similar_scores = [item[1] for item in similarity_scores]

    recommendations = content_movies.iloc[similar_movie_indices].copy()
    recommendations["content_score"] = similar_scores

    return recommendations


def calculate_recency_weight(timestamp_value, half_life_days: int = 90) -> float:
    """
    Calculates a recency weight using exponential decay.

    More recent actions receive higher weight.
    Older actions still matter, but less.
    """
    if timestamp_value is None or pd.isna(timestamp_value):
        return 1.0

    current_timestamp = int(time.time())
    seconds_since_action = current_timestamp - int(timestamp_value)
    days_since_action = max(seconds_since_action / 86400, 0)

    decay_lambda = math.log(2) / half_life_days

    return math.exp(-decay_lambda * days_since_action)


def build_user_seed_movies(
    user_id: int,
    ratings: pd.DataFrame,
    favorite_movie_ids: set[int],
    watched_movie_ids: set[int],
    min_positive_rating: float = 4.0
) -> pd.DataFrame:
    """
    Builds a set of seed movies for the current user.

    Seed movies are movies that represent the user's taste.
    We use:
        - highly rated movies
        - favorite movies
        - watched movies, with smaller weight
    """
    seed_rows = []

    user_ratings = ratings[
        (ratings["userId"] == user_id) &
        (ratings["source"] == "cinematch")
    ].copy()

    for _, rating_row in user_ratings.iterrows():
        movie_id = int(rating_row["movieId"])
        rating = float(rating_row["rating"])

        if rating >= min_positive_rating:
            recency_weight = calculate_recency_weight(rating_row.get("timestamp"))

            seed_rows.append(
                {
                    "movieId": movie_id,
                    "signal": "high_rating",
                    "weight": rating * recency_weight
                }
            )

    for movie_id in favorite_movie_ids:
        seed_rows.append(
            {
                "movieId": int(movie_id),
                "signal": "favorite",
                "weight": 5.0
            }
        )

    for movie_id in watched_movie_ids:
        seed_rows.append(
            {
                "movieId": int(movie_id),
                "signal": "watched",
                "weight": 1.0
            }
        )

    if not seed_rows:
        return pd.DataFrame(columns=["movieId", "signal", "weight"])

    seed_movies = pd.DataFrame(seed_rows)

    seed_movies = (
        seed_movies
        .groupby("movieId", as_index=False)
        .agg(
            weight=("weight", "sum"),
            signals=("signal", lambda values: ", ".join(sorted(set(values))))
        )
    )

    return seed_movies


def get_content_based_recommendations_for_user(
    user_id: int,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    tags: pd.DataFrame,
    favorite_movie_ids: set[int],
    watched_movie_ids: set[int],
    top_n: int = 10
) -> pd.DataFrame:
    """
    Generates personalized content-based recommendations for a user.
    """
    content_movies = build_content_dataframe(movies, tags)
    _, tfidf_matrix = create_tfidf_matrix(content_movies)
    cosine_sim_matrix = calculate_cosine_similarity(tfidf_matrix)

    seed_movies = build_user_seed_movies(
        user_id=user_id,
        ratings=ratings,
        favorite_movie_ids=favorite_movie_ids,
        watched_movie_ids=watched_movie_ids
    )

    if seed_movies.empty:
        return pd.DataFrame()

    already_seen_movie_ids = set(seed_movies["movieId"].astype(int))

    all_recommendations = []

    for _, seed_movie in seed_movies.iterrows():
        seed_movie_id = int(seed_movie["movieId"])
        seed_weight = float(seed_movie["weight"])
        seed_signals = seed_movie["signals"]

        similar_movies = get_similar_movies(
            movie_id=seed_movie_id,
            content_movies=content_movies,
            cosine_sim_matrix=cosine_sim_matrix,
            top_n=30
        )

        if similar_movies.empty:
            continue

        similar_movies = similar_movies[
            ~similar_movies["movieId"].isin(already_seen_movie_ids)
        ].copy()

        similar_movies["seed_movie_id"] = seed_movie_id
        similar_movies["seed_signals"] = seed_signals
        similar_movies["weighted_content_score"] = (
            similar_movies["content_score"] * seed_weight
        )

        all_recommendations.append(similar_movies)

    if not all_recommendations:
        return pd.DataFrame()

    recommendations = pd.concat(all_recommendations, ignore_index=True)

    final_recommendations = (
        recommendations
        .groupby("movieId", as_index=False)
        .agg(
            title=("title", "first"),
            clean_title=("clean_title", "first"),
            year=("year", "first"),
            genres=("genres", "first"),
            content_features=("content_features", "first"),
            content_score=("weighted_content_score", "sum"),
            matched_seed_movies=("seed_movie_id", lambda values: list(set(values))),
            reason_signals=("seed_signals", lambda values: ", ".join(sorted(set(values))))
        )
        .sort_values("content_score", ascending=False)
        .head(top_n)
    )

    return final_recommendations