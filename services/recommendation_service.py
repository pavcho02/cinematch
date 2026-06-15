import pandas as pd

from recommender.content_based import (
    build_content_dataframe,
    calculate_cosine_similarity,
    create_tfidf_matrix,
    get_content_based_recommendations_for_user,
    get_similar_movies,
)
from recommender.hybrid import combine_hybrid_recommendations
from recommender.item_based_cf import (
    get_item_based_recommendations,
    get_similar_items_for_movie,
)
from recommender.user_based_cf import get_user_based_recommendations
from services.database_service import (
    load_movies_from_db,
    load_ratings_from_db,
    load_tags_from_db,
)
from services.preference_service import (
    get_favorite_genres,
    get_favorite_movie_ids,
    get_favorite_movies,
)
from services.rating_service import (
    get_user_ratings,
    get_watched_movie_ids,
    get_watched_movies,
)
from utils.preprocessing import split_genres


def load_recommendation_data():
    """
    Loads all data needed for recommendation algorithms.
    """
    movies = load_movies_from_db()
    ratings = load_ratings_from_db()
    tags = load_tags_from_db()

    movies["genres_list"] = movies["genres"].apply(split_genres)

    return movies, ratings, tags


def get_content_recommendations(user_id: int, top_n: int = 10) -> pd.DataFrame:
    """
    Returns personalized content-based recommendations for the given user.
    """
    movies, ratings, tags = load_recommendation_data()

    favorite_movie_ids = get_favorite_movie_ids(user_id)
    watched_movie_ids = get_watched_movie_ids(user_id)

    recommendations = get_content_based_recommendations_for_user(
        user_id=user_id,
        movies=movies,
        ratings=ratings,
        tags=tags,
        favorite_movie_ids=favorite_movie_ids,
        watched_movie_ids=watched_movie_ids,
        top_n=top_n
    )

    return recommendations


def get_similar_movies_for_movie(
    movie_id: int,
    top_n: int = 10,
    exclude_movie_ids=None
) -> pd.DataFrame:
    """
    Returns movies similar to a given movie using Content-Based Filtering.

    Used for the "Because you watched..." section.
    """
    movies, ratings, tags = load_recommendation_data()

    content_movies = build_content_dataframe(movies, tags)
    _, tfidf_matrix = create_tfidf_matrix(content_movies)
    cosine_sim_matrix = calculate_cosine_similarity(tfidf_matrix)

    similar_movies = get_similar_movies(
        movie_id=movie_id,
        content_movies=content_movies,
        cosine_sim_matrix=cosine_sim_matrix,
        top_n=top_n + 20
    )

    if similar_movies.empty:
        return pd.DataFrame()

    if exclude_movie_ids:
        similar_movies = similar_movies[
            ~similar_movies["movieId"].isin(exclude_movie_ids)
        ]

    return similar_movies.head(top_n)


def get_user_seed_movies_for_because_you_watched(user_id: int) -> pd.DataFrame:
    """
    Returns movies that can be used as seed movies for the
    'Because you watched...' and item-based sections.
    """
    watched_movies = get_watched_movies(user_id)
    favorite_movies = get_favorite_movies(user_id)
    rated_movies = get_user_ratings(user_id)

    seed_dataframes = []

    if not watched_movies.empty:
        watched = watched_movies.copy()
        watched["source_signal"] = "watched"
        watched["signal_priority"] = 3
        watched["signal_time"] = watched["watched_at"]

        seed_dataframes.append(
            watched[
                [
                    "movieId",
                    "clean_title",
                    "year",
                    "genres",
                    "source_signal",
                    "signal_priority",
                    "signal_time"
                ]
            ]
        )

    if not favorite_movies.empty:
        favorites = favorite_movies.copy()
        favorites["source_signal"] = "favorite"
        favorites["signal_priority"] = 2
        favorites["signal_time"] = favorites["created_at"]

        seed_dataframes.append(
            favorites[
                [
                    "movieId",
                    "clean_title",
                    "year",
                    "genres",
                    "source_signal",
                    "signal_priority",
                    "signal_time"
                ]
            ]
        )

    if not rated_movies.empty:
        rated = rated_movies.copy()
        rated["source_signal"] = "rated"
        rated["signal_priority"] = 1
        rated["signal_time"] = rated["created_at"]

        seed_dataframes.append(
            rated[
                [
                    "movieId",
                    "clean_title",
                    "year",
                    "genres",
                    "source_signal",
                    "signal_priority",
                    "signal_time"
                ]
            ]
        )

    if not seed_dataframes:
        return pd.DataFrame()

    seed_movies = pd.concat(seed_dataframes, ignore_index=True)

    seed_movies = seed_movies.sort_values(
        by=["signal_time", "signal_priority"],
        ascending=[False, False]
    )

    seed_movies = seed_movies.drop_duplicates(
        subset=["movieId"],
        keep="first"
    )

    return seed_movies


def get_genre_based_recommendations(
    user_id: int,
    top_n: int = 10,
    min_ratings: int = 50
) -> pd.DataFrame:
    """
    Returns cold start recommendations based on the user's favorite genres.
    """
    favorite_genres = get_favorite_genres(user_id)

    if not favorite_genres:
        return pd.DataFrame()

    movies, ratings, tags = load_recommendation_data()

    rating_stats = (
        ratings
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

    favorite_genres_set = set(favorite_genres)

    movies_with_stats["matched_genres"] = movies_with_stats["genres_list"].apply(
        lambda genres: sorted(set(genres).intersection(favorite_genres_set))
    )

    movies_with_stats["genre_match_count"] = movies_with_stats["matched_genres"].apply(len)

    recommendations = movies_with_stats[
        movies_with_stats["genre_match_count"] > 0
    ].copy()

    recommendations = recommendations[
        recommendations["rating_count"] >= min_ratings
    ].copy()

    watched_ids = get_watched_movie_ids(user_id)
    favorite_movie_ids = get_favorite_movie_ids(user_id)
    rated_movie_ids = set(
        get_user_ratings(user_id)["movieId"].astype(int).tolist()
    )

    excluded_movie_ids = watched_ids.union(favorite_movie_ids).union(rated_movie_ids)

    if excluded_movie_ids:
        recommendations = recommendations[
            ~recommendations["movieId"].isin(excluded_movie_ids)
        ]

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
        by=[
            "genre_score",
            "genre_match_count",
            "avg_rating",
            "rating_count"
        ],
        ascending=[False, False, False, False]
    )

    return recommendations.head(top_n)


def get_user_based_cf_recommendations(
    user_id: int,
    top_n: int = 10,
    similar_users_count: int = 20,
    min_common_items: int = 2,
    min_neighbor_rating: float = 4.0
) -> pd.DataFrame:
    """
    Returns User-Based Collaborative Filtering recommendations.

    These recommendations are based on movies liked by users
    with similar rating behavior.
    """
    movies, ratings, tags = load_recommendation_data()

    favorite_movie_ids = get_favorite_movie_ids(user_id)
    watched_movie_ids = get_watched_movie_ids(user_id)

    recommendations = get_user_based_recommendations(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        favorite_movie_ids=favorite_movie_ids,
        watched_movie_ids=watched_movie_ids,
        top_n=top_n,
        similar_users_count=similar_users_count,
        min_common_items=min_common_items,
        min_neighbor_rating=min_neighbor_rating
    )

    return recommendations


def get_item_based_cf_recommendations(
    user_id: int,
    top_n: int = 10,
    similar_items_count: int = 20,
    min_common_users: int = 2
) -> pd.DataFrame:
    """
    Returns Item-Based Collaborative Filtering recommendations.

    These recommendations are based on movies similar to the user's
    watched, favorite and highly rated movies.
    """
    movies, ratings, tags = load_recommendation_data()

    favorite_movie_ids = get_favorite_movie_ids(user_id)
    watched_movie_ids = get_watched_movie_ids(user_id)
    user_ratings = get_user_ratings(user_id)

    rated_movie_ids = set(user_ratings["movieId"].astype(int).tolist())

    seed_movie_ids = (
        favorite_movie_ids
        .union(watched_movie_ids)
        .union(rated_movie_ids)
    )

    recommendations = get_item_based_recommendations(
        user_id=user_id,
        ratings=ratings,
        movies=movies,
        seed_movie_ids=seed_movie_ids,
        favorite_movie_ids=favorite_movie_ids,
        watched_movie_ids=watched_movie_ids,
        top_n=top_n,
        similar_items_count=similar_items_count,
        min_common_users=min_common_users
    )

    return recommendations


def get_item_based_cf_for_movie(
    user_id: int,
    movie_id: int,
    top_n: int = 10,
    min_common_users: int = 2
) -> pd.DataFrame:
    """
    Returns item-based recommendations for a selected movie.

    This powers:
        "Users who watched this also watched..."
    """
    movies, ratings, tags = load_recommendation_data()

    favorite_movie_ids = get_favorite_movie_ids(user_id)
    watched_movie_ids = get_watched_movie_ids(user_id)
    user_ratings = get_user_ratings(user_id)

    rated_movie_ids = set(user_ratings["movieId"].astype(int).tolist())

    excluded_movie_ids = (
        favorite_movie_ids
        .union(watched_movie_ids)
        .union(rated_movie_ids)
    )

    excluded_movie_ids.add(movie_id)

    recommendations = get_similar_items_for_movie(
        movie_id=movie_id,
        ratings=ratings,
        movies=movies,
        exclude_movie_ids=excluded_movie_ids,
        top_n=top_n,
        min_common_users=min_common_users
    )

    return recommendations


def get_hybrid_recommendations(
    user_id: int,
    top_n: int = 10,
    content_weight: float = 0.35,
    genre_weight: float = 0.20,
    user_based_weight: float = 0.25,
    item_based_weight: float = 0.20
) -> pd.DataFrame:
    """
    Returns final hybrid recommendations for the user.

    This combines:
        - Content-Based Filtering
        - Genre-Based Cold Start
        - User-Based Collaborative Filtering
        - Item-Based Collaborative Filtering
    """
    content_recommendations = get_content_recommendations(
        user_id=user_id,
        top_n=30
    )

    genre_recommendations = get_genre_based_recommendations(
        user_id=user_id,
        top_n=30,
        min_ratings=50
    )

    user_based_recommendations = get_user_based_cf_recommendations(
        user_id=user_id,
        top_n=30,
        similar_users_count=20,
        min_common_items=1,
        min_neighbor_rating=4.0
    )

    item_based_recommendations = get_item_based_cf_recommendations(
        user_id=user_id,
        top_n=30,
        similar_items_count=20,
        min_common_users=1
    )

    hybrid_recommendations = combine_hybrid_recommendations(
        content_recommendations=content_recommendations,
        genre_recommendations=genre_recommendations,
        user_based_recommendations=user_based_recommendations,
        item_based_recommendations=item_based_recommendations,
        top_n=top_n,
        content_weight=content_weight,
        genre_weight=genre_weight,
        user_based_weight=user_based_weight,
        item_based_weight=item_based_weight
    )

    return hybrid_recommendations