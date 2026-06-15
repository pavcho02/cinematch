PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS recommendation_logs;
DROP TABLE IF EXISTS watched_movies;
DROP TABLE IF EXISTS favorite_movies;
DROP TABLE IF EXISTS favorite_genres;
DROP TABLE IF EXISTS tags;
DROP TABLE IF EXISTS ratings;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS movies;

CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    clean_title TEXT NOT NULL,
    year INTEGER,
    genres TEXT,
    content_features TEXT,
    imdb_id INTEGER,
    tmdb_id INTEGER
);

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recommender_user_id INTEGER UNIQUE,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    rating REAL NOT NULL,
    timestamp INTEGER,
    source TEXT NOT NULL DEFAULT 'cinematch',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);

CREATE TABLE tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    timestamp INTEGER,
    source TEXT NOT NULL DEFAULT 'movielens',

    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);

CREATE TABLE favorite_genres (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    genre TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(recommender_user_id),
    UNIQUE(user_id, genre)
);

CREATE TABLE favorite_movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(recommender_user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id),
    UNIQUE(user_id, movie_id)
);

CREATE TABLE watched_movies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    watched_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(recommender_user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id),
    UNIQUE(user_id, movie_id)
);

CREATE TABLE recommendation_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    algorithm TEXT NOT NULL,
    score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id) REFERENCES users(recommender_user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);