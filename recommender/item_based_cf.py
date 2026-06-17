import pandas as pd


def build_user_movie_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a user-movie rating matrix.

    Rows are users.
    Columns are movies.
    Values are ratings.
    """
    rating_matrix = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    return rating_matrix


def safe_pearson_correlation(first_ratings: pd.Series, second_ratings: pd.Series):
    """
    Safely calculates Pearson correlation between two rating vectors.

    Pearson correlation is undefined when:
        - there are fewer than 2 common values;
        - one vector has no variance;
        - the result is NaN.

    This prevents RuntimeWarning messages from NumPy/Pandas.
    """
    first_ratings = first_ratings.astype(float)
    second_ratings = second_ratings.astype(float)

    if len(first_ratings) < 2 or len(second_ratings) < 2:
        return None

    if first_ratings.nunique() < 2:
        return None

    if second_ratings.nunique() < 2:
        return None

    similarity = first_ratings.corr(
        second_ratings,
        method="pearson"
    )

    if pd.isna(similarity):
        return None

    return float(similarity)


def find_similar_items(
    movie_id: int,
    rating_matrix: pd.DataFrame,
    top_n: int = 20,
    min_common_users: int = 2
) -> pd.DataFrame:
    """
    Finds movies similar to a given movie using Pearson correlation.

    Similarity is calculated based on users who rated both movies.
    """
    if movie_id not in rating_matrix.columns:
        return pd.DataFrame(
            columns=[
                "movieId",
                "item_similarity",
                "common_users"
            ]
        )

    target_movie_ratings = rating_matrix[movie_id]

    similar_items = []

    for other_movie_id in rating_matrix.columns:
        if other_movie_id == movie_id:
            continue

        other_movie_ratings = rating_matrix[other_movie_id]

        common_ratings = target_movie_ratings.notna() & other_movie_ratings.notna()
        common_users_count = int(common_ratings.sum())

        if common_users_count < min_common_users:
            continue

        target_common_ratings = target_movie_ratings[common_ratings]
        other_common_ratings = other_movie_ratings[common_ratings]

        similarity = safe_pearson_correlation(
            target_common_ratings,
            other_common_ratings
        )

        if similarity is None:
            continue

        if similarity <= 0:
            continue

        similar_items.append(
            {
                "movieId": int(other_movie_id),
                "item_similarity": similarity,
                "common_users": common_users_count
            }
        )

    if not similar_items:
        return pd.DataFrame(
            columns=[
                "movieId",
                "item_similarity",
                "common_users"
            ]
        )

    similar_items_df = pd.DataFrame(similar_items)

    similar_items_df = similar_items_df.sort_values(
        by=["item_similarity", "common_users"],
        ascending=[False, False]
    )

    return similar_items_df.head(top_n)


def get_similar_items_for_movie(
    movie_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    exclude_movie_ids=None,
    top_n: int = 10,
    min_common_users: int = 2
) -> pd.DataFrame:
    """
    Returns item-based similar movies for one selected movie.

    This powers the section:
        "Users who watched this also watched..."
    """
    if ratings.empty:
        return pd.DataFrame()

    if exclude_movie_ids is None:
        exclude_movie_ids = set()

    rating_matrix = build_user_movie_matrix(ratings)

    similar_items = find_similar_items(
        movie_id=movie_id,
        rating_matrix=rating_matrix,
        top_n=top_n + 30,
        min_common_users=min_common_users
    )

    if similar_items.empty:
        return pd.DataFrame()

    similar_items = similar_items[
        ~similar_items["movieId"].isin(exclude_movie_ids)
    ].copy()

    recommendations = similar_items.merge(
        movies,
        on="movieId",
        how="left"
    )

    recommendations = recommendations.dropna(subset=["clean_title"])

    recommendations = recommendations.sort_values(
        by=["item_similarity", "common_users"],
        ascending=[False, False]
    )

    return recommendations.head(top_n)


def calculate_seed_weight(
    seed_movie_id: int,
    user_rating_map: dict,
    favorite_movie_ids: set,
    watched_movie_ids: set
) -> float:
    """
    Calculates how important a seed movie is for item-based recommendations.

    A movie can be important because:
        - the user rated it highly;
        - the user added it to favorites;
        - the user watched it.
    """
    weight = 0.0

    if seed_movie_id in user_rating_map:
        rating = float(user_rating_map[seed_movie_id])

        if rating < 3.5:
            return 0.0

        weight += rating / 5.0

    if seed_movie_id in favorite_movie_ids:
        weight += 1.0

    if seed_movie_id in watched_movie_ids:
        weight += 0.6

    return weight


def get_item_based_recommendations(
    user_id: int,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    seed_movie_ids: set,
    favorite_movie_ids: set,
    watched_movie_ids: set,
    top_n: int = 10,
    similar_items_count: int = 20,
    min_common_users: int = 2
) -> pd.DataFrame:
    """
    Generates Item-Based Collaborative Filtering recommendations.

    The algorithm:
        1. Uses the user's watched, favorite and highly rated movies as seed movies.
        2. Finds movies similar to each seed movie using Pearson correlation.
        3. Excludes movies already rated, watched or favorited by the current user.
        4. Combines scores from all seed movies.
    """
    if ratings.empty:
        return pd.DataFrame()

    if not seed_movie_ids:
        return pd.DataFrame()

    user_ratings = ratings[ratings["userId"] == user_id].copy()

    user_rating_map = dict(
        zip(
            user_ratings["movieId"].astype(int),
            user_ratings["rating"].astype(float)
        )
    )

    rated_movie_ids = set(user_ratings["movieId"].astype(int).tolist())

    excluded_movie_ids = (
        rated_movie_ids
        .union(favorite_movie_ids)
        .union(watched_movie_ids)
    )

    rating_matrix = build_user_movie_matrix(ratings)

    candidate_rows = []

    for seed_movie_id in seed_movie_ids:
        seed_movie_id = int(seed_movie_id)

        seed_weight = calculate_seed_weight(
            seed_movie_id=seed_movie_id,
            user_rating_map=user_rating_map,
            favorite_movie_ids=favorite_movie_ids,
            watched_movie_ids=watched_movie_ids
        )

        if seed_weight <= 0:
            continue

        similar_items = find_similar_items(
            movie_id=seed_movie_id,
            rating_matrix=rating_matrix,
            top_n=similar_items_count,
            min_common_users=min_common_users
        )

        if similar_items.empty:
            continue

        similar_items = similar_items[
            ~similar_items["movieId"].isin(excluded_movie_ids)
        ].copy()

        for _, row in similar_items.iterrows():
            candidate_rows.append(
                {
                    "movieId": int(row["movieId"]),
                    "seed_movie_id": seed_movie_id,
                    "item_similarity": float(row["item_similarity"]),
                    "common_users": int(row["common_users"]),
                    "seed_weight": seed_weight,
                    "weighted_item_score": float(row["item_similarity"]) * seed_weight
                }
            )

    if not candidate_rows:
        return pd.DataFrame()

    candidates = pd.DataFrame(candidate_rows)

    scored_candidates = (
        candidates
        .groupby("movieId", as_index=False)
        .agg(
            item_cf_score=("weighted_item_score", "sum"),
            support_count=("seed_movie_id", "nunique"),
            avg_similarity=("item_similarity", "mean"),
            max_similarity=("item_similarity", "max"),
            avg_common_users=("common_users", "mean"),
            matched_seed_movies=("seed_movie_id", lambda values: list(set(values)))
        )
    )

    recommendations = scored_candidates.merge(
        movies,
        on="movieId",
        how="left"
    )

    recommendations = recommendations.dropna(subset=["clean_title"])

    recommendations = recommendations.sort_values(
        by=[
            "item_cf_score",
            "support_count",
            "avg_similarity",
            "avg_common_users"
        ],
        ascending=[False, False, False, False]
    )

    return recommendations.head(top_n)