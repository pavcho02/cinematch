from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/ml-latest-small")


def load_movies() -> pd.DataFrame:
    """
    Loads the MovieLens movies dataset.

    Returns:
        DataFrame with columns: movieId, title, genres
    """
    movies_path = DATA_DIR / "movies.csv"

    if not movies_path.exists():
        raise FileNotFoundError(
            f"Movies file not found: {movies_path}. "
            "Please download MovieLens ml-latest-small and place movies.csv there."
        )

    return pd.read_csv(movies_path)


def load_ratings() -> pd.DataFrame:
    """
    Loads the MovieLens ratings dataset.

    Returns:
        DataFrame with columns: userId, movieId, rating, timestamp
    """
    ratings_path = DATA_DIR / "ratings.csv"

    if not ratings_path.exists():
        raise FileNotFoundError(
            f"Ratings file not found: {ratings_path}. "
            "Please download MovieLens ml-latest-small and place ratings.csv there."
        )

    return pd.read_csv(ratings_path)


def load_tags() -> pd.DataFrame:
    """
    Loads the MovieLens tags dataset.

    Returns:
        DataFrame with columns: userId, movieId, tag, timestamp
    """
    tags_path = DATA_DIR / "tags.csv"

    if not tags_path.exists():
        raise FileNotFoundError(
            f"Tags file not found: {tags_path}. "
            "Please download MovieLens ml-latest-small and place tags.csv there."
        )

    return pd.read_csv(tags_path)


def load_links() -> pd.DataFrame:
    """
    Loads the MovieLens links dataset.

    Returns:
        DataFrame with columns: movieId, imdbId, tmdbId
    """
    links_path = DATA_DIR / "links.csv"

    if not links_path.exists():
        raise FileNotFoundError(
            f"Links file not found: {links_path}. "
            "Please download MovieLens ml-latest-small and place links.csv there."
        )

    return pd.read_csv(links_path)


def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads all MovieLens CSV files.

    Returns:
        movies, ratings, tags, links
    """
    movies = load_movies()
    ratings = load_ratings()
    tags = load_tags()
    links = load_links()

    return movies, ratings, tags, links