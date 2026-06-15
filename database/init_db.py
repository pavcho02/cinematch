import sqlite3
from pathlib import Path

from utils.data_loader import load_all_data
from utils.preprocessing import preprocess_movies


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "cinematch.db"
SCHEMA_PATH = DATABASE_DIR / "schema.sql"


def get_connection():
    """
    Creates and returns a SQLite database connection.
    """
    return sqlite3.connect(DATABASE_PATH)


def create_database_schema(connection: sqlite3.Connection):
    """
    Creates all database tables from schema.sql.
    """
    with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
        schema_sql = schema_file.read()

    connection.executescript(schema_sql)
    connection.commit()


def import_movies(connection: sqlite3.Connection):
    """
    Imports MovieLens movies into the movies table.
    """
    movies, ratings, tags, links = load_all_data()
    processed_movies = preprocess_movies(movies)

    movies_with_links = processed_movies.merge(
        links,
        on="movieId",
        how="left"
    )

    movies_to_insert = movies_with_links[
        [
            "movieId",
            "title",
            "clean_title",
            "year",
            "genres",
            "content_features",
            "imdbId",
            "tmdbId",
        ]
    ].copy()

    movies_to_insert = movies_to_insert.rename(
        columns={
            "movieId": "movie_id",
            "imdbId": "imdb_id",
            "tmdbId": "tmdb_id",
        }
    )

    movies_to_insert.to_sql(
        "movies",
        connection,
        if_exists="append",
        index=False
    )


def import_ratings(connection: sqlite3.Connection):
    """
    Imports MovieLens ratings into the ratings table.
    """
    movies, ratings, tags, links = load_all_data()

    ratings_to_insert = ratings.rename(
        columns={
            "userId": "user_id",
            "movieId": "movie_id",
        }
    ).copy()

    ratings_to_insert["source"] = "movielens"

    ratings_to_insert = ratings_to_insert[
        [
            "user_id",
            "movie_id",
            "rating",
            "timestamp",
            "source",
        ]
    ]

    ratings_to_insert.to_sql(
        "ratings",
        connection,
        if_exists="append",
        index=False
    )


def import_tags(connection: sqlite3.Connection):
    """
    Imports MovieLens tags into the tags table.
    """
    movies, ratings, tags, links = load_all_data()

    tags_to_insert = tags.rename(
        columns={
            "userId": "user_id",
            "movieId": "movie_id",
        }
    ).copy()

    tags_to_insert["source"] = "movielens"

    tags_to_insert = tags_to_insert[
        [
            "user_id",
            "movie_id",
            "tag",
            "timestamp",
            "source",
        ]
    ]

    tags_to_insert.to_sql(
        "tags",
        connection,
        if_exists="append",
        index=False
    )


def print_database_summary(connection: sqlite3.Connection):
    """
    Prints a short summary after importing the data.
    """
    cursor = connection.cursor()

    tables = [
        "movies",
        "ratings",
        "tags",
        "users",
        "favorite_movies",
        "favorite_genres",
        "watched_movies",
    ]

    print("\nCineMatch database summary:")
    print("-" * 35)

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table}: {count}")

    print("-" * 35)
    print(f"Database created at: {DATABASE_PATH}")


def initialize_database():
    """
    Creates the SQLite database and imports MovieLens data.
    """
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = get_connection()

    try:
        print("Creating database schema...")
        create_database_schema(connection)

        print("Importing movies...")
        import_movies(connection)

        print("Importing ratings...")
        import_ratings(connection)

        print("Importing tags...")
        import_tags(connection)

        print_database_summary(connection)

    finally:
        connection.close()


if __name__ == "__main__":
    initialize_database()