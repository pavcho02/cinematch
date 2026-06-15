import sqlite3
import time

import pandas as pd

from services.database_service import get_connection


def get_user_rating(user_id: int, movie_id: int):
    """
    Returns the user's rating for a movie, or None if not rated.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT rating
            FROM ratings
            WHERE user_id = ?
              AND movie_id = ?
              AND source = 'cinematch'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id, movie_id)
        )

        row = cursor.fetchone()

    if row is None:
        return None

    return float(row[0])


def rate_movie(user_id: int, movie_id: int, rating: float):
    """
    Adds or updates a CineMatch user's rating for a movie.

    Also marks the movie as watched, because rating a movie means
    the user has watched it.
    """
    if rating < 0.5 or rating > 5.0:
        raise ValueError("Rating must be between 0.5 and 5.0.")

    current_timestamp = int(time.time())

    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id
            FROM ratings
            WHERE user_id = ?
              AND movie_id = ?
              AND source = 'cinematch'
            LIMIT 1
            """,
            (user_id, movie_id)
        )

        existing_rating = cursor.fetchone()

        if existing_rating:
            cursor.execute(
                """
                UPDATE ratings
                SET rating = ?,
                    timestamp = ?,
                    created_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (rating, current_timestamp, existing_rating[0])
            )
        else:
            cursor.execute(
                """
                INSERT INTO ratings (
                    user_id,
                    movie_id,
                    rating,
                    timestamp,
                    source
                )
                VALUES (?, ?, ?, ?, 'cinematch')
                """,
                (
                    user_id,
                    movie_id,
                    rating,
                    current_timestamp
                )
            )

        cursor.execute(
            """
            INSERT OR IGNORE INTO watched_movies (
                user_id,
                movie_id
            )
            VALUES (?, ?)
            """,
            (user_id, movie_id)
        )

        connection.commit()


def remove_rating(user_id: int, movie_id: int):
    """
    Removes the user's CineMatch rating for a movie.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM ratings
            WHERE user_id = ?
              AND movie_id = ?
              AND source = 'cinematch'
            """,
            (user_id, movie_id)
        )

        connection.commit()


def mark_movie_as_watched(user_id: int, movie_id: int) -> bool:
    """
    Marks a movie as watched.

    Returns True if added, False if it was already marked as watched.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO watched_movies (
                    user_id,
                    movie_id
                )
                VALUES (?, ?)
                """,
                (user_id, movie_id)
            )

            connection.commit()
            return True

        except sqlite3.IntegrityError:
            return False


def remove_watched_movie(user_id: int, movie_id: int):
    """
    Removes a movie from watched movies.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM watched_movies
            WHERE user_id = ?
              AND movie_id = ?
            """,
            (user_id, movie_id)
        )

        connection.commit()


def get_watched_movie_ids(user_id: int) -> set[int]:
    """
    Returns IDs of movies watched by the user.
    """
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT movie_id
            FROM watched_movies
            WHERE user_id = ?
            """,
            (user_id,)
        )

        rows = cursor.fetchall()

    return {row[0] for row in rows}


def get_user_ratings(user_id: int) -> pd.DataFrame:
    """
    Returns CineMatch ratings made by the current user with movie details.
    """
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                r.user_id AS userId,
                r.movie_id AS movieId,
                r.rating,
                r.timestamp,
                r.created_at,
                m.title,
                m.clean_title,
                m.year,
                m.genres
            FROM ratings r
            JOIN movies m ON r.movie_id = m.movie_id
            WHERE r.user_id = ?
              AND r.source = 'cinematch'
            ORDER BY r.created_at DESC
            """,
            connection,
            params=(user_id,)
        )


def get_watched_movies(user_id: int) -> pd.DataFrame:
    """
    Returns watched movies for the current user with movie details.
    """
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                wm.user_id AS userId,
                wm.movie_id AS movieId,
                wm.watched_at,
                m.title,
                m.clean_title,
                m.year,
                m.genres
            FROM watched_movies wm
            JOIN movies m ON wm.movie_id = m.movie_id
            WHERE wm.user_id = ?
            ORDER BY wm.watched_at DESC
            """,
            connection,
            params=(user_id,)
        )