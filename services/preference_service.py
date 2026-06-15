import sqlite3

import pandas as pd

from services.database_service import get_connection


def get_favorite_genres(user_id: int) -> list[str]:
    """
    Returns the user's favorite genres.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT genre
            FROM favorite_genres
            WHERE user_id = ?
            ORDER BY genre
            """,
            (user_id,)
        )

        rows = cursor.fetchall()

    return [row[0] for row in rows]


def set_favorite_genres(user_id: int, genres: list[str]):
    """
    Replaces the user's favorite genres with the selected genres.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM favorite_genres
            WHERE user_id = ?
            """,
            (user_id,)
        )

        for genre in genres:
            cursor.execute(
                """
                INSERT OR IGNORE INTO favorite_genres (user_id, genre)
                VALUES (?, ?)
                """,
                (user_id, genre)
            )

        connection.commit()


def add_favorite_movie(user_id: int, movie_id: int) -> bool:
    """
    Adds a movie to the user's favorite movies.

    Returns True if the movie was added, False if it already existed.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO favorite_movies (user_id, movie_id)
                VALUES (?, ?)
                """,
                (user_id, movie_id)
            )

            connection.commit()
            return True

        except sqlite3.IntegrityError:
            return False


def remove_favorite_movie(user_id: int, movie_id: int):
    """
    Removes a movie from the user's favorite movies.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM favorite_movies
            WHERE user_id = ? AND movie_id = ?
            """,
            (user_id, movie_id)
        )

        connection.commit()


def get_favorite_movie_ids(user_id: int) -> set[int]:
    """
    Returns the user's favorite movie IDs.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT movie_id
            FROM favorite_movies
            WHERE user_id = ?
            """,
            (user_id,)
        )

        rows = cursor.fetchall()

    return {row[0] for row in rows}


def is_favorite_movie(user_id: int, movie_id: int) -> bool:
    """
    Checks whether a movie is in the user's favorite movies.
    """
    return movie_id in get_favorite_movie_ids(user_id)


def get_favorite_movies(user_id: int) -> pd.DataFrame:
    """
    Returns the user's favorite movies with movie details.
    """
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                m.movie_id AS movieId,
                m.title,
                m.clean_title,
                m.year,
                m.genres,
                m.content_features,
                fm.created_at
            FROM favorite_movies fm
            JOIN movies m ON fm.movie_id = m.movie_id
            WHERE fm.user_id = ?
            ORDER BY fm.created_at DESC
            """,
            connection,
            params=(user_id,)
        )