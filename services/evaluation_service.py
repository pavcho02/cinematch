import numpy as np
import pandas as pd

from evaluation.metrics import (
    mean_absolute_error,
    precision_at_k,
    recall_at_k,
)
from recommender.content_based import (
    build_content_dataframe,
    calculate_cosine_similarity,
    create_tfidf_matrix,
    get_similar_movies,
)
from recommender.hybrid import combine_hybrid_recommendations
from recommender.item_based_cf import (
    build_user_movie_matrix as build_item_user_movie_matrix,
    find_similar_items,
    get_item_based_recommendations,
)
from recommender.user_based_cf import (
    build_user_movie_matrix,
    find_similar_users,
    get_user_based_recommendations,
)
from services.database_service import (
    load_movies_from_db,
    load_ratings_from_db,
    load_tags_from_db,
)
from utils.preprocessing import split_genres


def load_evaluation_data():
    """
    Loads MovieLens data from SQLite for evaluation.

    We evaluate using MovieLens historical users only.
    CineMatch users are excluded from the offline evaluation.
    """
    movies = load_movies_from_db()
    ratings = load_ratings_from_db()
    tags = load_tags_from_db()

    ratings = ratings[ratings["source"] == "movielens"].copy()

    movies["genres_list"] = movies["genres"].apply(split_genres)

    return movies, ratings, tags


def create_train_test_split(
    ratings: pd.DataFrame,
    max_users: int = 20,
    min_user_ratings: int = 20,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Creates a simple per-user train/test split.

    For each sampled user:
        - some ratings go to train
        - some ratings go to test
    """
    user_rating_counts = ratings.groupby("userId").size()

    eligible_users = user_rating_counts[
        user_rating_counts >= min_user_ratings
    ].index.tolist()

    if not eligible_users:
        return pd.DataFrame(), pd.DataFrame(), []

    rng = np.random.default_rng(random_state)

    sampled_users = rng.choice(
        eligible_users,
        size=min(max_users, len(eligible_users)),
        replace=False
    )

    train_parts = []
    test_parts = []

    for user_id in sampled_users:
        user_ratings = ratings[ratings["userId"] == user_id].copy()

        test_count = max(1, int(len(user_ratings) * test_size))

        test_ratings = user_ratings.sample(
            n=test_count,
            random_state=random_state + int(user_id)
        )

        train_ratings = user_ratings.drop(test_ratings.index)

        train_parts.append(train_ratings)
        test_parts.append(test_ratings)

    selected_user_ids = [int(user_id) for user_id in sampled_users]

    selected_user_ratings = ratings[
        ratings["userId"].isin(selected_user_ids)
    ]

    other_ratings = ratings[
        ~ratings["userId"].isin(selected_user_ids)
    ]

    train_ratings = pd.concat([other_ratings] + train_parts, ignore_index=True)
    test_ratings = pd.concat(test_parts, ignore_index=True)

    return train_ratings, test_ratings, selected_user_ids


def get_content_recommendations_for_evaluation(
    user_id: int,
    train_ratings: pd.DataFrame,
    content_movies: pd.DataFrame,
    cosine_sim_matrix,
    top_n: int = 10,
    relevance_threshold: float = 4.0
) -> pd.DataFrame:
    """
    Generates content-based recommendations for evaluation.

    Seed movies are the user's highly rated movies from the train set.
    """
    user_train_ratings = train_ratings[train_ratings["userId"] == user_id].copy()

    if user_train_ratings.empty:
        return pd.DataFrame()

    positive_ratings = user_train_ratings[
        user_train_ratings["rating"] >= relevance_threshold
    ]

    if positive_ratings.empty:
        return pd.DataFrame()

    already_rated_ids = set(user_train_ratings["movieId"].astype(int).tolist())

    all_recommendations = []

    for _, rating_row in positive_ratings.iterrows():
        seed_movie_id = int(rating_row["movieId"])
        seed_weight = float(rating_row["rating"]) / 5.0

        similar_movies = get_similar_movies(
            movie_id=seed_movie_id,
            content_movies=content_movies,
            cosine_sim_matrix=cosine_sim_matrix,
            top_n=30
        )

        if similar_movies.empty:
            continue

        similar_movies = similar_movies[
            ~similar_movies["movieId"].isin(already_rated_ids)
        ].copy()

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
            content_score=("weighted_content_score", "sum")
        )
        .sort_values("content_score", ascending=False)
        .head(top_n)
    )

    return final_recommendations


def get_genre_recommendations_for_evaluation(
    user_id: int,
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
    min_ratings: int = 50,
    relevance_threshold: float = 4.0
) -> pd.DataFrame:
    """
    Generates genre-based recommendations for evaluation.

    Favorite genres are inferred from highly rated movies in the train set.
    """
    user_train_ratings = train_ratings[train_ratings["userId"] == user_id].copy()

    if user_train_ratings.empty:
        return pd.DataFrame()

    positive_ratings = user_train_ratings[
        user_train_ratings["rating"] >= relevance_threshold
    ]

    if positive_ratings.empty:
        return pd.DataFrame()

    positive_movies = positive_ratings.merge(
        movies[["movieId", "genres_list"]],
        on="movieId",
        how="left"
    )

    genre_weights = {}

    for _, row in positive_movies.iterrows():
        genres = row["genres_list"]

        if not isinstance(genres, list):
            continue

        for genre in genres:
            genre_weights[genre] = genre_weights.get(genre, 0.0) + float(row["rating"])

    if not genre_weights:
        return pd.DataFrame()

    favorite_genres = set(genre_weights.keys())

    rating_stats = (
        train_ratings
        .groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    movies_with_stats = movies.merge(
        rating_stats,
        on="movieId",
        how="left"
    )

    movies_with_stats["avg_rating"] = movies_with_stats["avg_rating"].fillna(0)
    movies_with_stats["rating_count"] = movies_with_stats["rating_count"].fillna(0)

    movies_with_stats["matched_genres"] = movies_with_stats["genres_list"].apply(
        lambda genres: sorted(set(genres).intersection(favorite_genres))
    )

    movies_with_stats["genre_match_count"] = movies_with_stats["matched_genres"].apply(len)

    already_rated_ids = set(user_train_ratings["movieId"].astype(int).tolist())

    recommendations = movies_with_stats[
        (movies_with_stats["genre_match_count"] > 0) &
        (movies_with_stats["rating_count"] >= min_ratings) &
        (~movies_with_stats["movieId"].isin(already_rated_ids))
    ].copy()

    if recommendations.empty:
        return pd.DataFrame()

    recommendations["genre_score"] = (
        recommendations["genre_match_count"] * 2.0 +
        recommendations["avg_rating"] +
        (recommendations["rating_count"].clip(upper=500) / 500)
    )

    recommendations["matched_genres_text"] = recommendations["matched_genres"].apply(
        lambda genres: ", ".join(genres)
    )

    recommendations = recommendations.sort_values(
        by=["genre_score", "genre_match_count", "avg_rating", "rating_count"],
        ascending=[False, False, False, False]
    )

    return recommendations.head(top_n)


def predict_rating_user_based(
    user_id: int,
    movie_id: int,
    rating_matrix: pd.DataFrame,
    similar_users: pd.DataFrame
):
    """
    Predicts a rating using similar users.

    Prediction = weighted average of ratings from similar users.
    """
    if movie_id not in rating_matrix.columns:
        return None

    if similar_users.empty:
        return None

    numerator = 0.0
    denominator = 0.0

    for _, similar_user in similar_users.iterrows():
        similar_user_id = int(similar_user["similar_user_id"])
        similarity = float(similar_user["similarity"])

        if similar_user_id not in rating_matrix.index:
            continue

        neighbor_rating = rating_matrix.loc[similar_user_id, movie_id]

        if pd.isna(neighbor_rating):
            continue

        numerator += similarity * float(neighbor_rating)
        denominator += abs(similarity)

    if denominator == 0:
        return None

    prediction = numerator / denominator

    return max(0.5, min(5.0, prediction))


def predict_rating_item_based(
    user_id: int,
    movie_id: int,
    rating_matrix: pd.DataFrame,
    min_common_users: int = 2
):
    """
    Predicts a rating using similar items.

    Prediction = weighted average of the user's ratings for similar movies.
    """
    if user_id not in rating_matrix.index:
        return None

    if movie_id not in rating_matrix.columns:
        return None

    user_ratings = rating_matrix.loc[user_id].dropna()

    if user_ratings.empty:
        return None

    similar_items = find_similar_items(
        movie_id=movie_id,
        rating_matrix=rating_matrix,
        top_n=50,
        min_common_users=min_common_users
    )

    if similar_items.empty:
        return None

    numerator = 0.0
    denominator = 0.0

    for _, similar_item in similar_items.iterrows():
        similar_movie_id = int(similar_item["movieId"])
        similarity = float(similar_item["item_similarity"])

        if similar_movie_id not in user_ratings.index:
            continue

        user_rating = float(user_ratings[similar_movie_id])

        numerator += similarity * user_rating
        denominator += abs(similarity)

    if denominator == 0:
        return None

    prediction = numerator / denominator

    return max(0.5, min(5.0, prediction))


def evaluate_recommenders(
    k: int = 10,
    max_users: int = 20,
    min_user_ratings: int = 20,
    test_size: float = 0.2,
    relevance_threshold: float = 4.0,
    random_state: int = 42
):
    """
    Evaluates CineMatch recommendation algorithms.

    Returns:
        summary_df, details
    """
    movies, ratings, tags = load_evaluation_data()

    train_ratings, test_ratings, selected_user_ids = create_train_test_split(
        ratings=ratings,
        max_users=max_users,
        min_user_ratings=min_user_ratings,
        test_size=test_size,
        random_state=random_state
    )

    if train_ratings.empty or test_ratings.empty:
        return pd.DataFrame(), {
            "evaluated_users": 0,
            "train_ratings": 0,
            "test_ratings": 0
        }

    content_movies = build_content_dataframe(movies, tags)
    _, tfidf_matrix = create_tfidf_matrix(content_movies)
    cosine_sim_matrix = calculate_cosine_similarity(tfidf_matrix)

    user_rating_matrix = build_user_movie_matrix(train_ratings)
    item_rating_matrix = build_item_user_movie_matrix(train_ratings)

    metric_rows = {
        "Content-Based": {
            "precision": [],
            "recall": [],
            "mae_actual": [],
            "mae_predicted": []
        },
        "Genre-Based Cold Start": {
            "precision": [],
            "recall": [],
            "mae_actual": [],
            "mae_predicted": []
        },
        "User-Based CF": {
            "precision": [],
            "recall": [],
            "mae_actual": [],
            "mae_predicted": []
        },
        "Item-Based CF": {
            "precision": [],
            "recall": [],
            "mae_actual": [],
            "mae_predicted": []
        },
        "Hybrid": {
            "precision": [],
            "recall": [],
            "mae_actual": [],
            "mae_predicted": []
        }
    }

    evaluated_users = 0

    for user_id in selected_user_ids:
        user_test_ratings = test_ratings[test_ratings["userId"] == user_id]
        user_train_ratings = train_ratings[train_ratings["userId"] == user_id]

        if user_test_ratings.empty or user_train_ratings.empty:
            continue

        relevant_test_ids = set(
            user_test_ratings[
                user_test_ratings["rating"] >= relevance_threshold
            ]["movieId"].astype(int).tolist()
        )

        if not relevant_test_ids:
            continue

        evaluated_users += 1

        already_rated_ids = set(user_train_ratings["movieId"].astype(int).tolist())

        content_recommendations = get_content_recommendations_for_evaluation(
            user_id=user_id,
            train_ratings=train_ratings,
            content_movies=content_movies,
            cosine_sim_matrix=cosine_sim_matrix,
            top_n=k,
            relevance_threshold=relevance_threshold
        )

        genre_recommendations = get_genre_recommendations_for_evaluation(
            user_id=user_id,
            train_ratings=train_ratings,
            movies=movies,
            top_n=k,
            min_ratings=50,
            relevance_threshold=relevance_threshold
        )

        user_based_recommendations = get_user_based_recommendations(
            user_id=user_id,
            ratings=train_ratings,
            movies=movies,
            favorite_movie_ids=set(),
            watched_movie_ids=set(),
            top_n=k,
            similar_users_count=20,
            min_common_items=2,
            min_neighbor_rating=relevance_threshold
        )

        positive_train_ids = set(
            user_train_ratings[
                user_train_ratings["rating"] >= 3.5
            ]["movieId"].astype(int).tolist()
        )

        item_based_recommendations = get_item_based_recommendations(
            user_id=user_id,
            ratings=train_ratings,
            movies=movies,
            seed_movie_ids=positive_train_ids,
            favorite_movie_ids=set(),
            watched_movie_ids=set(),
            top_n=k,
            similar_items_count=20,
            min_common_users=2
        )

        hybrid_recommendations = combine_hybrid_recommendations(
            content_recommendations=content_recommendations,
            genre_recommendations=genre_recommendations,
            user_based_recommendations=user_based_recommendations,
            item_based_recommendations=item_based_recommendations,
            top_n=k
        )

        algorithm_outputs = {
            "Content-Based": content_recommendations,
            "Genre-Based Cold Start": genre_recommendations,
            "User-Based CF": user_based_recommendations,
            "Item-Based CF": item_based_recommendations,
            "Hybrid": hybrid_recommendations
        }

        for algorithm_name, recommendations in algorithm_outputs.items():
            if recommendations.empty:
                recommended_ids = []
            else:
                recommended_ids = recommendations["movieId"].astype(int).tolist()

            metric_rows[algorithm_name]["precision"].append(
                precision_at_k(recommended_ids, relevant_test_ids, k)
            )

            metric_rows[algorithm_name]["recall"].append(
                recall_at_k(recommended_ids, relevant_test_ids, k)
            )

        similar_users = find_similar_users(
            user_id=user_id,
            rating_matrix=user_rating_matrix,
            top_n=30,
            min_common_items=2
        )

        for _, test_row in user_test_ratings.iterrows():
            movie_id = int(test_row["movieId"])
            actual_rating = float(test_row["rating"])

            user_based_prediction = predict_rating_user_based(
                user_id=user_id,
                movie_id=movie_id,
                rating_matrix=user_rating_matrix,
                similar_users=similar_users
            )

            if user_based_prediction is not None:
                metric_rows["User-Based CF"]["mae_actual"].append(actual_rating)
                metric_rows["User-Based CF"]["mae_predicted"].append(user_based_prediction)

            item_based_prediction = predict_rating_item_based(
                user_id=user_id,
                movie_id=movie_id,
                rating_matrix=item_rating_matrix,
                min_common_users=2
            )

            if item_based_prediction is not None:
                metric_rows["Item-Based CF"]["mae_actual"].append(actual_rating)
                metric_rows["Item-Based CF"]["mae_predicted"].append(item_based_prediction)

    summary_rows = []

    for algorithm_name, values in metric_rows.items():
        precision_values = values["precision"]
        recall_values = values["recall"]

        mae_value = mean_absolute_error(
            values["mae_actual"],
            values["mae_predicted"]
        )

        summary_rows.append(
            {
                "algorithm": algorithm_name,
                f"precision@{k}": float(np.mean(precision_values)) if precision_values else 0.0,
                f"recall@{k}": float(np.mean(recall_values)) if recall_values else 0.0,
                "mae": mae_value,
                "mae_predictions_count": len(values["mae_predicted"])
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    details = {
        "evaluated_users": evaluated_users,
        "train_ratings": len(train_ratings),
        "test_ratings": len(test_ratings),
        "selected_users": len(selected_user_ids),
        "k": k,
        "relevance_threshold": relevance_threshold
    }

    return summary_df, details