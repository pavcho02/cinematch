import pandas as pd


def build_user_movie_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a user-movie rating matrix.

    Rows:
        users

    Columns:
        movies

    Values:
        ratings
    """
    rating_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    return rating_matrix


def find_similar_users(
    user_id: int,
    rating_matrix: pd.DataFrame,
    top_n: int = 20,
    min_common_items: int = 2
) -> pd.DataFrame:
    """
    Finds users with similar taste using Pearson correlation.

    Only users with at least min_common_items commonly rated movies
    are considered.
    """
    if user_id not in rating_matrix.index:
        return pd.DataFrame(
            columns=[
                "similar_user_id",
                "similarity",
                "common_items"
            ]
        )

    target_user_ratings = rating_matrix.loc[user_id]

    similar_users = []

    for other_user_id in rating_matrix.index:
        if other_user_id == user_id:
            continue

        other_user_ratings = rating_matrix.loc[other_user_id]

        common_ratings = target_user_ratings.notna() & other_user_ratings.notna()
        common_items_count = int(common_ratings.sum())

        if common_items_count < min_common_items:
            continue

        similarity = target_user_ratings[common_ratings].corr(
            other_user_ratings[common_ratings],
            method="pearson"
        )

        if pd.isna(similarity):
            continue

        if similarity <= 0:
            continue

        similar_users.append(
            {
                "similar_user_id": int(other_user_id),
                "similarity": float(similarity),
                "common_items": common_items_count
            }
        )

    if not similar_users:
        return pd.DataFrame(
            columns=[
                "similar_user_id",
                "similarity",
                "common_items"
            ]
        )

    similar_users_df = pd.DataFrame(similar_users)

    similar_users_df = similar_users_df.sort_values(
        by=["similarity", "common_items"],
        ascending=[False, False]
    )

    return similar_users_df.head(top_n)


def get_user_based_recommendations(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    favorite_movie_ids: set[int],
    watched_movie_ids: set[int],
    top_n: int = 10,
    similar_users_count: int = 20,
    min_common_items: int = 2,
    min_neighbor_rating: float = 4.0
) -> pd.DataFrame:
    """
    Generates User-Based Collaborative Filtering recommendations.

    The algorithm:
        1. Builds a user-movie rating matrix.
        2. Finds similar users using Pearson correlation.
        3. Takes movies highly rated by similar users.
        4. Excludes movies already rated, watched or favorited by current user.
        5. Scores candidates using weighted average rating.
    """
    if ratings.empty:
        return pd.DataFrame()

    user_ratings = ratings[ratings["userId"] == user_id]

    if user_ratings.empty:
        return pd.DataFrame()

    rating_matrix = build_user_movie_matrix(ratings)

    similar_users = find_similar_users(
        user_id=user_id,
        rating_matrix=rating_matrix,
        top_n=similar_users_count,
        min_common_items=min_common_items
    )

    if similar_users.empty:
        return pd.DataFrame()

    rated_movie_ids = set(user_ratings["movieId"].astype(int).tolist())

    excluded_movie_ids = (
        rated_movie_ids
        .union(favorite_movie_ids)
        .union(watched_movie_ids)
    )

    candidate_rows = []

    for _, similar_user in similar_users.iterrows():
        similar_user_id = int(similar_user["similar_user_id"])
        similarity = float(similar_user["similarity"])

        neighbor_ratings = ratings[
            (ratings["userId"] == similar_user_id) &
            (ratings["rating"] >= min_neighbor_rating)
        ]

        for _, rating_row in neighbor_ratings.iterrows():
            movie_id = int(rating_row["movieId"])

            if movie_id in excluded_movie_ids:
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
            avg_neighbor_rating=("neighbor_rating", "mean"),
            max_similarity=("similarity", "max")
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
        by=[
            "predicted_rating",
            "support_count",
            "avg_neighbor_rating",
            "max_similarity"
        ],
        ascending=[False, False, False, False]
    )

    return recommendations.head(top_n)