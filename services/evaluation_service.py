import warnings

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
)
from recommender.content_based import (
    build_content_dataframe,
    create_tfidf_matrix,
)
from recommender.hybrid import combine_hybrid_recommendations
from recommender.user_based_cf import (
    build_user_movie_matrix,
    find_similar_users,
)
from services.database_service import (
    load_movies_from_db,
    load_ratings_from_db,
    load_tags_from_db,
)
from utils.preprocessing import split_genres


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in divide"
)

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="Degrees of freedom <= 0 for slice"
)

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="divide by zero encountered in divide"
)

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in multiply"
)


def load_evaluation_data():
    """
    Loads MovieLens data from SQLite for evaluation.

    CineMatch users are excluded from offline evaluation.
    Only MovieLens historical users are used.
    """
    movies = load_movies_from_db()
    ratings = load_ratings_from_db()
    tags = load_tags_from_db()

    ratings = ratings[ratings["source"] == "movielens"].copy()

    movies["genres_list"] = movies["genres"].apply(split_genres)

    return movies, ratings, tags


def create_train_test_split(
    ratings: pd.DataFrame,
    max_users: int = 5,
    min_user_ratings: int = 20,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Creates a per-user train/test split.

    For each selected MovieLens user:
        - part of the user's ratings go to train;
        - part of the user's ratings go to test.
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

    other_ratings = ratings[
        ~ratings["userId"].isin(selected_user_ids)
    ]

    train_ratings = pd.concat(
        [other_ratings] + train_parts,
        ignore_index=True
    )

    test_ratings = pd.concat(
        test_parts,
        ignore_index=True
    )

    return train_ratings, test_ratings, selected_user_ids


def get_popular_movie_ids(
    train_ratings: pd.DataFrame,
    max_candidates: int = 1500
):
    """
    Returns the most rated movie IDs.

    This is used to limit item-based evaluation to a reasonable candidate set.
    """
    popular_movie_ids = (
        train_ratings
        .groupby("movieId")
        .size()
        .sort_values(ascending=False)
        .head(max_candidates)
        .index
        .astype(int)
        .tolist()
    )

    return popular_movie_ids


def get_popular_recommendations_for_evaluation(
    user_id: int,
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    top_n: int = 10,
    min_ratings: int = 50
):
    """
    Generates a simple popularity-based baseline.

    This is useful as a comparison point for the other algorithms.
    """
    user_train_ratings = train_ratings[train_ratings["userId"] == user_id]
    already_rated_ids = set(user_train_ratings["movieId"].astype(int).tolist())

    rating_stats = (
        train_ratings
        .groupby("movieId")
        .agg(
            avg_rating=("rating", "mean"),
            rating_count=("rating", "count")
        )
        .reset_index()
    )

    recommendations = movies.merge(
        rating_stats,
        on="movieId",
        how="left"
    )

    recommendations["avg_rating"] = recommendations["avg_rating"].fillna(0)
    recommendations["rating_count"] = recommendations["rating_count"].fillna(0)

    recommendations = recommendations[
        (recommendations["rating_count"] >= min_ratings) &
        (~recommendations["movieId"].isin(already_rated_ids))
    ].copy()

    if recommendations.empty:
        return pd.DataFrame()

    recommendations["popularity_score"] = (
        recommendations["avg_rating"] +
        recommendations["rating_count"].clip(upper=500) / 500
    )

    recommendations = recommendations.sort_values(
        by=["popularity_score", "avg_rating", "rating_count"],
        ascending=[False, False, False]
    )

    return recommendations.head(top_n)


def get_similar_content_movies_fast(
    seed_movie_id: int,
    content_movies: pd.DataFrame,
    tfidf_matrix,
    movie_index_map: dict,
    top_n: int = 20
):
    """
    Finds content-similar movies for one seed movie.

    This avoids building the full all-vs-all cosine similarity matrix.
    It calculates similarity only for the selected seed movie.
    """
    if seed_movie_id not in movie_index_map:
        return pd.DataFrame()

    seed_index = movie_index_map[seed_movie_id]

    similarity_scores = cosine_similarity(
        tfidf_matrix[seed_index],
        tfidf_matrix
    ).flatten()

    sorted_indices = np.argsort(similarity_scores)[::-1]

    result_rows = []

    for movie_index in sorted_indices:
        if movie_index == seed_index:
            continue

        score = float(similarity_scores[movie_index])

        if score <= 0:
            continue

        movie_row = content_movies.iloc[movie_index].copy()
        movie_row["content_score"] = score

        result_rows.append(movie_row)

        if len(result_rows) >= top_n:
            break

    if not result_rows:
        return pd.DataFrame()

    return pd.DataFrame(result_rows)


def get_content_recommendations_for_evaluation(
    user_id: int,
    train_ratings: pd.DataFrame,
    content_movies: pd.DataFrame,
    tfidf_matrix,
    movie_index_map: dict,
    top_n: int = 10,
    relevance_threshold: float = 4.0,
    max_seed_movies: int = 3
) -> pd.DataFrame:
    """
    Generates lightweight content-based recommendations for evaluation.

    Only a limited number of highly rated seed movies is used.
    This keeps the evaluation fast.
    """
    user_train_ratings = train_ratings[train_ratings["userId"] == user_id].copy()

    if user_train_ratings.empty:
        return pd.DataFrame()

    positive_ratings = (
        user_train_ratings[
            user_train_ratings["rating"] >= relevance_threshold
        ]
        .sort_values(by="rating", ascending=False)
        .head(max_seed_movies)
    )

    if positive_ratings.empty:
        return pd.DataFrame()

    already_rated_ids = set(user_train_ratings["movieId"].astype(int).tolist())

    all_recommendations = []

    for _, rating_row in positive_ratings.iterrows():
        seed_movie_id = int(rating_row["movieId"])
        seed_weight = float(rating_row["rating"]) / 5.0

        similar_movies = get_similar_content_movies_fast(
            seed_movie_id=seed_movie_id,
            content_movies=content_movies,
            tfidf_matrix=tfidf_matrix,
            movie_index_map=movie_index_map,
            top_n=20
        )

        if similar_movies.empty:
            continue

        similar_movies = similar_movies[
            ~similar_movies["movieId"].isin(already_rated_ids)
        ].copy()

        if similar_movies.empty:
            continue

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


def get_user_based_recommendations_for_evaluation(
    user_id: int,
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    rating_matrix: pd.DataFrame,
    top_n: int = 10,
    similar_users_count: int = 20,
    min_common_items: int = 2,
    min_neighbor_rating: float = 4.0
) -> pd.DataFrame:
    """
    Generates User-Based CF recommendations using a precomputed rating matrix.

    This is faster than rebuilding the matrix for every evaluated user.
    """
    user_train_ratings = train_ratings[train_ratings["userId"] == user_id]

    if user_train_ratings.empty:
        return pd.DataFrame()

    similar_users = find_similar_users(
        user_id=user_id,
        rating_matrix=rating_matrix,
        top_n=similar_users_count,
        min_common_items=min_common_items
    )

    if similar_users.empty:
        return pd.DataFrame()

    already_rated_ids = set(user_train_ratings["movieId"].astype(int).tolist())

    candidate_rows = []

    for _, similar_user in similar_users.iterrows():
        similar_user_id = int(similar_user["similar_user_id"])
        similarity = float(similar_user["similarity"])

        neighbor_ratings = train_ratings[
            (train_ratings["userId"] == similar_user_id) &
            (train_ratings["rating"] >= min_neighbor_rating)
        ]

        for _, rating_row in neighbor_ratings.iterrows():
            movie_id = int(rating_row["movieId"])

            if movie_id in already_rated_ids:
                continue

            neighbor_rating = float(rating_row["rating"])

            candidate_rows.append(
                {
                    "movieId": movie_id,
                    "similar_user_id": similar_user_id,
                    "similarity": similarity,
                    "neighbor_rating": neighbor_rating,
                    "weighted_rating": similarity * neighbor_rating
                }
            )

    if not candidate_rows:
        return pd.DataFrame()

    candidates = pd.DataFrame(candidate_rows)

    scored_candidates = (
        candidates
        .groupby("movieId", as_index=False)
        .agg(
            weighted_rating_sum=("weighted_rating", "sum"),
            similarity_sum=("similarity", "sum"),
            support_count=("similar_user_id", "nunique"),
            avg_neighbor_rating=("neighbor_rating", "mean")
        )
    )

    scored_candidates["predicted_rating"] = (
        scored_candidates["weighted_rating_sum"] /
        scored_candidates["similarity_sum"]
    )

    recommendations = scored_candidates.merge(
        movies,
        on="movieId",
        how="left"
    )

    recommendations = recommendations.dropna(subset=["clean_title"])

    recommendations = recommendations.sort_values(
        by=["predicted_rating", "support_count", "avg_neighbor_rating"],
        ascending=[False, False, False]
    )

    return recommendations.head(top_n)


def find_similar_items_fast(
    movie_id: int,
    rating_matrix: pd.DataFrame,
    candidate_movie_ids: list,
    top_n: int = 20,
    min_common_users: int = 2
) -> pd.DataFrame:
    """
    Finds similar movies using Pearson correlation in a faster way.

    The comparison is limited to popular candidate movies.
    Runtime warnings from undefined Pearson cases are suppressed and invalid
    similarities are removed.
    """
    if movie_id not in rating_matrix.columns:
        return pd.DataFrame(
            columns=["movieId", "item_similarity", "common_users"]
        )

    candidate_columns = [
        movie_id_candidate
        for movie_id_candidate in candidate_movie_ids
        if movie_id_candidate in rating_matrix.columns and movie_id_candidate != movie_id
    ]

    if not candidate_columns:
        return pd.DataFrame(
            columns=["movieId", "item_similarity", "common_users"]
        )

    target_ratings = rating_matrix[movie_id]
    candidate_ratings = rating_matrix[candidate_columns]

    common_counts = (
        candidate_ratings
        .notna()
        .multiply(target_ratings.notna(), axis=0)
        .sum(axis=0)
    )

    valid_columns = common_counts[
        common_counts >= min_common_users
    ].index.tolist()

    if not valid_columns:
        return pd.DataFrame(
            columns=["movieId", "item_similarity", "common_users"]
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        similarities = candidate_ratings[valid_columns].corrwith(
            target_ratings,
            axis=0,
            method="pearson"
        )

    similar_items = pd.DataFrame(
        {
            "movieId": similarities.index.astype(int),
            "item_similarity": similarities.values,
            "common_users": common_counts[similarities.index].values
        }
    )

    similar_items = similar_items.dropna(subset=["item_similarity"])

    similar_items = similar_items[
        similar_items["item_similarity"] > 0
    ].copy()

    if similar_items.empty:
        return pd.DataFrame(
            columns=["movieId", "item_similarity", "common_users"]
        )

    similar_items = similar_items.sort_values(
        by=["item_similarity", "common_users"],
        ascending=[False, False]
    )

    return similar_items.head(top_n)


def get_item_recommendations_for_evaluation(
    user_id: int,
    train_ratings: pd.DataFrame,
    movies: pd.DataFrame,
    rating_matrix: pd.DataFrame,
    candidate_movie_ids: list,
    top_n: int = 10,
    relevance_threshold: float = 4.0,
    min_common_users: int = 2,
    max_seed_movies: int = 2
) -> pd.DataFrame:
    """
    Generates lightweight Item-Based CF recommendations for evaluation.

    Only a small number of highly rated seed movies is used.
    Candidate comparisons are limited to popular movies.
    """
    user_train_ratings = train_ratings[train_ratings["userId"] == user_id].copy()

    if user_train_ratings.empty:
        return pd.DataFrame()

    seed_ratings = (
        user_train_ratings[
            user_train_ratings["rating"] >= relevance_threshold
        ]
        .sort_values(by="rating", ascending=False)
        .head(max_seed_movies)
    )

    if seed_ratings.empty:
        return pd.DataFrame()

    already_rated_ids = set(user_train_ratings["movieId"].astype(int).tolist())

    all_candidates = []

    for _, seed_row in seed_ratings.iterrows():
        seed_movie_id = int(seed_row["movieId"])
        seed_weight = float(seed_row["rating"]) / 5.0

        similar_items = find_similar_items_fast(
            movie_id=seed_movie_id,
            rating_matrix=rating_matrix,
            candidate_movie_ids=candidate_movie_ids,
            top_n=20,
            min_common_users=min_common_users
        )

        if similar_items.empty:
            continue

        similar_items = similar_items[
            ~similar_items["movieId"].isin(already_rated_ids)
        ].copy()

        if similar_items.empty:
            continue

        similar_items["weighted_item_score"] = (
            similar_items["item_similarity"] * seed_weight
        )

        all_candidates.append(similar_items)

    if not all_candidates:
        return pd.DataFrame()

    candidates = pd.concat(all_candidates, ignore_index=True)

    scored_candidates = (
        candidates
        .groupby("movieId", as_index=False)
        .agg(
            item_cf_score=("weighted_item_score", "sum"),
            avg_similarity=("item_similarity", "mean"),
            avg_common_users=("common_users", "mean")
        )
    )

    recommendations = scored_candidates.merge(
        movies,
        on="movieId",
        how="left"
    )

    recommendations = recommendations.dropna(subset=["clean_title"])

    recommendations = recommendations.sort_values(
        by=["item_cf_score", "avg_similarity", "avg_common_users"],
        ascending=[False, False, False]
    )

    return recommendations.head(top_n)


def evaluate_recommenders(
    k: int = 10,
    max_users: int = 5,
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
            "test_ratings": 0,
            "selected_users": 0,
            "k": k,
            "relevance_threshold": relevance_threshold
        }

    content_movies = build_content_dataframe(movies, tags)
    _, tfidf_matrix = create_tfidf_matrix(content_movies)

    movie_index_map = {
        int(movie_id): index
        for index, movie_id in enumerate(content_movies["movieId"].astype(int))
    }

    rating_matrix = build_user_movie_matrix(train_ratings)

    popular_candidate_movie_ids = get_popular_movie_ids(
        train_ratings=train_ratings,
        max_candidates=1500
    )

    metric_rows = {
        "Popularity Baseline": {
            "precision": [],
            "recall": []
        },
        "Content-Based": {
            "precision": [],
            "recall": []
        },
        "Genre-Based Cold Start": {
            "precision": [],
            "recall": []
        },
        "User-Based CF": {
            "precision": [],
            "recall": []
        },
        "Item-Based CF": {
            "precision": [],
            "recall": []
        },
        "Hybrid": {
            "precision": [],
            "recall": []
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

        popularity_recommendations = get_popular_recommendations_for_evaluation(
            user_id=user_id,
            train_ratings=train_ratings,
            movies=movies,
            top_n=k,
            min_ratings=50
        )

        content_recommendations = get_content_recommendations_for_evaluation(
            user_id=user_id,
            train_ratings=train_ratings,
            content_movies=content_movies,
            tfidf_matrix=tfidf_matrix,
            movie_index_map=movie_index_map,
            top_n=k,
            relevance_threshold=relevance_threshold,
            max_seed_movies=3
        )

        genre_recommendations = get_genre_recommendations_for_evaluation(
            user_id=user_id,
            train_ratings=train_ratings,
            movies=movies,
            top_n=k,
            min_ratings=50,
            relevance_threshold=relevance_threshold
        )

        user_based_recommendations = get_user_based_recommendations_for_evaluation(
            user_id=user_id,
            train_ratings=train_ratings,
            movies=movies,
            rating_matrix=rating_matrix,
            top_n=k,
            similar_users_count=20,
            min_common_items=2,
            min_neighbor_rating=relevance_threshold
        )

        item_based_recommendations = get_item_recommendations_for_evaluation(
            user_id=user_id,
            train_ratings=train_ratings,
            movies=movies,
            rating_matrix=rating_matrix,
            candidate_movie_ids=popular_candidate_movie_ids,
            top_n=k,
            relevance_threshold=relevance_threshold,
            min_common_users=2,
            max_seed_movies=2
        )

        hybrid_recommendations = combine_hybrid_recommendations(
            content_recommendations=content_recommendations,
            genre_recommendations=genre_recommendations,
            user_based_recommendations=user_based_recommendations,
            item_based_recommendations=item_based_recommendations,
            top_n=k
        )

        algorithm_outputs = {
            "Popularity Baseline": popularity_recommendations,
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

    summary_rows = []

    for algorithm_name, values in metric_rows.items():
        precision_values = values["precision"]
        recall_values = values["recall"]

        summary_rows.append(
            {
                "algorithm": algorithm_name,
                f"precision@{k}": float(np.mean(precision_values)) if precision_values else 0.0,
                f"recall@{k}": float(np.mean(recall_values)) if recall_values else 0.0
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