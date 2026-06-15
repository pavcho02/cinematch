import pandas as pd


def get_all_genres(movies: pd.DataFrame) -> list[str]:
    """
    Returns all unique genres from the movies dataset.
    """
    genres = sorted(
        {
            genre
            for genres_list in movies["genres_list"]
            for genre in genres_list
        }
    )

    return genres


def search_movies(
    movies: pd.DataFrame,
    search_query: str = "",
    selected_genre: str = "All",
    min_year: int | None = None,
    max_year: int | None = None,
) -> pd.DataFrame:
    """
    Searches and filters movies by title, genre and year range.
    """
    filtered_movies = movies.copy()

    if search_query:
        filtered_movies = filtered_movies[
            filtered_movies["clean_title"]
            .str.contains(search_query, case=False, na=False)
        ]

    if selected_genre != "All":
        filtered_movies = filtered_movies[
            filtered_movies["genres_list"].apply(
                lambda genres: selected_genre in genres
            )
        ]

    if min_year is not None:
        filtered_movies = filtered_movies[
            filtered_movies["year"].fillna(0) >= min_year
        ]

    if max_year is not None:
        filtered_movies = filtered_movies[
            filtered_movies["year"].fillna(0) <= max_year
        ]

    return filtered_movies.sort_values(
        by=["year", "clean_title"],
        ascending=[False, True]
    )


def get_movie_by_id(movies: pd.DataFrame, movie_id: int) -> pd.Series | None:
    """
    Returns a single movie by movieId.
    """
    result = movies[movies["movieId"] == movie_id]

    if result.empty:
        return None

    return result.iloc[0]


def get_movies_with_rating_stats(
    movies: pd.DataFrame,
    ratings: pd.DataFrame
) -> pd.DataFrame:
    """
    Adds average rating and rating count to the movies dataframe.
    """
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

    return movies_with_stats


def get_popular_movies(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    min_ratings: int = 50,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Returns popular movies based on average rating and number of ratings.
    """
    movies_with_stats = get_movies_with_rating_stats(movies, ratings)

    popular_movies = movies_with_stats[
        movies_with_stats["rating_count"] >= min_ratings
    ].sort_values(
        by=["avg_rating", "rating_count"],
        ascending=[False, False]
    )

    return popular_movies.head(top_n)