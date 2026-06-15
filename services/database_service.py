import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "database" / "cinematch.db"


def get_connection():
    """
    Returns a SQLite connection to the CineMatch database.
    """
    return sqlite3.connect(DATABASE_PATH)


def database_exists() -> bool:
    """
    Checks if the SQLite database file exists.
    """
    return DATABASE_PATH.exists()


def get_table_count(table_name: str) -> int:
    """
    Returns the number of rows in a table.
    """
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]


def get_database_summary() -> dict:
    """
    Returns basic row counts from the main database tables.
    """
    if not database_exists():
        return {}

    tables = [
        "movies",
        "ratings",
        "tags",
        "users",
        "favorite_movies",
        "favorite_genres",
        "watched_movies",
    ]

    summary = {}

    for table in tables:
        summary[table] = get_table_count(table)

    return summary


def load_movies_from_db() -> pd.DataFrame:
    """
    Loads movies from SQLite.

    The SQL table uses snake_case column names,
    but the returned DataFrame uses MovieLens-style names
    so the rest of the application can work consistently.
    """
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                movie_id AS movieId,
                title,
                clean_title,
                year,
                genres,
                content_features,
                imdb_id AS imdbId,
                tmdb_id AS tmdbId
            FROM movies
            """,
            connection
        )


def load_ratings_from_db() -> pd.DataFrame:
    """
    Loads ratings from SQLite.

    Returns columns compatible with MovieLens CSV format:
    userId, movieId, rating, timestamp, source
    """
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                user_id AS userId,
                movie_id AS movieId,
                rating,
                timestamp,
                source
            FROM ratings
            """,
            connection
        )


def load_tags_from_db() -> pd.DataFrame:
    """
    Loads tags from SQLite.

    Returns columns compatible with MovieLens CSV format:
    userId, movieId, tag, timestamp, source
    """
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                user_id AS userId,
                movie_id AS movieId,
                tag,
                timestamp,
                source
            FROM tags
            """,
            connection
        )