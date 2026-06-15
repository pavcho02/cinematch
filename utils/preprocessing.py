import re

import pandas as pd


def extract_year(title: str):
    """
    Extracts the release year from a movie title.

    Example:
        "Toy Story (1995)" -> 1995
    """
    match = re.search(r"\((\d{4})\)", title)

    if match:
        return int(match.group(1))

    return None


def clean_movie_title(title: str) -> str:
    """
    Removes the year from a movie title.

    Example:
        "Toy Story (1995)" -> "Toy Story"
    """
    return re.sub(r"\s*\(\d{4}\)", "", title).strip()


def split_genres(genres: str) -> list[str]:
    """
    Splits MovieLens genres into a Python list.

    Example:
        "Adventure|Comedy|Drama" -> ["Adventure", "Comedy", "Drama"]
    """
    if pd.isna(genres) or genres == "(no genres listed)":
        return []

    return genres.split("|")


def create_content_features(row: pd.Series) -> str:
    """
    Creates a text representation of a movie.

    This text will later be used for Content-Based Filtering with TF-IDF.
    """
    clean_title = row.get("clean_title", "")
    genres_list = row.get("genres_list", [])

    genres_text = " ".join(genres_list)

    return f"{clean_title} {genres_text}".strip()


def preprocess_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the movies dataset.

    Adds:
        - clean_title
        - year
        - genres_list
        - content_features
    """
    processed_movies = movies.copy()

    processed_movies["year"] = processed_movies["title"].apply(extract_year)
    processed_movies["clean_title"] = processed_movies["title"].apply(clean_movie_title)
    processed_movies["genres_list"] = processed_movies["genres"].apply(split_genres)
    processed_movies["content_features"] = processed_movies.apply(
        create_content_features,
        axis=1
    )

    return processed_movies