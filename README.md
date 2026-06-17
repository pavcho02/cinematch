# 🎬 CineMatch

**CineMatch** is a hybrid movie recommendation system built as a university project for the course **Recommender Systems**.

The application allows users to create profiles, select favorite genres, add favorite movies, rate movies, mark movies as watched, and receive personalized movie recommendations using multiple recommendation techniques.

CineMatch combines:

* Content-Based Filtering
* User-Based Collaborative Filtering
* Item-Based Collaborative Filtering
* Genre-Based Cold Start Recommendations
* Hybrid Recommendation Ranking
* Time-Aware User Profiling
* Offline Evaluation Metrics

---

## 📌 Project Overview

CineMatch is designed as a complete recommendation system with a working user interface, persistent storage, user profiles, recommendation algorithms, and evaluation metrics.

The system uses the **MovieLens ml-latest-small** dataset as the main data source. MovieLens provides movie metadata, ratings, tags, and external identifiers. CineMatch imports this data into a local SQLite database and extends it with its own user-generated data.

Users can interact with the system by:

* registering and logging in;
* selecting favorite genres;
* adding movies to favorites;
* marking movies as watched;
* rating movies from 0.5 to 5.0;
* receiving personalized recommendations;
* viewing evaluation results for the recommendation algorithms.

---

## 🎯 Main Goal

The main goal of CineMatch is to demonstrate how different recommender system approaches can be combined in a practical application.

Instead of using only one algorithm, CineMatch uses a **hybrid approach** that combines content information, user behavior, rating patterns, favorite genres, and recently performed actions.

This makes the system more realistic and allows it to handle different user scenarios:

* new users with no ratings;
* users with favorite genres only;
* users with several rated movies;
* users with watched and favorite movies;
* users with enough activity for collaborative filtering.

---

## ✨ Features

### 👤 User Authentication

CineMatch supports basic user authentication:

* user registration;
* login;
* logout;
* password hashing;
* session state management with Streamlit.

MovieLens users are not stored as application accounts. They are used only as anonymous historical rating profiles for collaborative filtering.

CineMatch users receive separate recommender IDs starting from:

```text
100001
```

This avoids conflicts with MovieLens user IDs.

---

### 🎭 User Preferences

Each CineMatch user can manage personal preferences:

* favorite genres;
* favorite movies;
* watched movies;
* movie ratings.

These user signals are later used by the recommendation algorithms.

---

### ⭐ Movie Ratings

Users can rate movies from:

```text
0.5 to 5.0
```

When a user rates a movie, CineMatch automatically marks the movie as watched.

Ratings are stored in SQLite with:

```text
source = 'cinematch'
```

MovieLens ratings are stored with:

```text
source = 'movielens'
```

This allows the system to distinguish between original dataset ratings and ratings created inside the application.

---

### 🎬 Movie Catalog

The application includes a searchable movie catalog.

Users can:

* search movies by title;
* filter movies by genre;
* filter movies by release year;
* sort movies by rating, popularity, year, or title;
* add movies to favorites;
* mark movies as watched;
* rate movies.

---

## 🧠 Recommendation Algorithms

CineMatch implements several recommendation approaches.

---

## 1. Content-Based Filtering

Content-Based Filtering recommends movies based on the content of the movies themselves.

The system builds a textual representation of each movie using:

* title;
* genres;
* MovieLens tags.

Example content representation:

```text
Matrix Action Sci-Fi Thriller artificial intelligence cyberpunk
```

Then CineMatch applies:

```text
TF-IDF Vectorization
Cosine Similarity
```

This allows the system to find movies that are similar in content.

Used in sections such as:

```text
Personalized Content-Based Recommendations
Because you watched...
```

---

## 2. Genre-Based Cold Start Recommendations

Cold start is a common problem in recommender systems.

A new user may not have:

* ratings;
* watched movies;
* favorite movies.

To solve this, CineMatch asks users to choose favorite genres.

The system then recommends popular and highly rated movies from those genres.

This allows the system to generate recommendations even for new users.

Used in section:

```text
Based on your favorite genres
```

---

## 3. User-Based Collaborative Filtering

User-Based Collaborative Filtering recommends movies by finding users with similar taste.

The system creates a user-movie rating matrix:

```text
rows = users
columns = movies
values = ratings
```

Then it compares users using:

```text
Pearson correlation
```

If another user has rated movies similarly to the current user, CineMatch recommends movies that the similar user liked but the current user has not watched or rated yet.

Used in section:

```text
Users with similar taste also liked...
```

---

## 4. Item-Based Collaborative Filtering

Item-Based Collaborative Filtering finds movies that are similar based on rating patterns.

Instead of comparing users, it compares movies.

Two movies are considered similar if many users rated them in a similar way.

The system uses:

```text
Pearson correlation between movie rating vectors
```

Used in section:

```text
Users who watched this also watched...
```

This is useful when the system wants to recommend movies related to a specific watched or rated movie.

---

## 5. Hybrid Recommendation System

The Hybrid Recommendation System combines results from multiple algorithms:

* Content-Based Filtering;
* Genre-Based Cold Start;
* User-Based Collaborative Filtering;
* Item-Based Collaborative Filtering.

Each algorithm produces its own score. Since the scores have different scales, CineMatch normalizes them and calculates a final hybrid score.

Example formula:

```text
hybrid_score =
0.35 * content_score +
0.20 * genre_score +
0.25 * user_based_score +
0.20 * item_based_score
```

The weights can be adjusted from the UI.

Used in section:

```text
Top recommendations for you
```

---

## ⏱️ Time-Aware User Profiling

CineMatch also considers when a user performed an action.

Recent ratings and watched movies can be more important than old ones.

This is useful because user interests may change over time.

For example:

* a user may have watched many horror movies in the past;
* but recently they may be watching mostly sci-fi and thrillers.

CineMatch can give more weight to recent actions when building the user profile.

---

## 📊 Evaluation Metrics

CineMatch includes an Evaluation page that compares recommendation algorithms using offline evaluation.

The implemented metrics are:

### Precision@K

Precision@K measures how many of the top K recommended movies are relevant.

```text
Precision@K = relevant recommended movies / K
```

### Recall@K

Recall@K measures how many relevant movies were successfully recommended.

```text
Recall@K = relevant recommended movies / all relevant movies
```

### MAE

MAE stands for Mean Absolute Error.

It measures the average difference between predicted ratings and real ratings.

```text
MAE = average absolute error between actual and predicted ratings
```

MAE is mainly used for algorithms that predict explicit ratings, such as:

* User-Based Collaborative Filtering;
* Item-Based Collaborative Filtering.

---

## 🛠️ Technologies Used

The project uses the following technologies:

| Technology        | Purpose                      |
| ----------------- | ---------------------------- |
| Python            | Main programming language    |
| Streamlit         | Web user interface           |
| SQLite            | Local database               |
| Pandas            | Data processing              |
| NumPy             | Numerical operations         |
| Scikit-learn      | TF-IDF and cosine similarity |
| MovieLens Dataset | Movie, rating and tag data   |

---

## 🗂️ Project Structure

```text
CineMatch/
│
├── Home.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── ml-latest-small/
│       ├── movies.csv
│       ├── ratings.csv
│       ├── tags.csv
│       └── links.csv
│
├── database/
│   ├── __init__.py
│   ├── schema.sql
│   ├── init_db.py
│   └── cinematch.db
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
│
├── pages/
│   ├── 1_Movies.py
│   ├── 2_Profile.py
│   ├── 3_Recommendations.py
│   └── 4_Evaluation.py
│
├── recommender/
│   ├── __init__.py
│   ├── content_based.py
│   ├── user_based_cf.py
│   ├── item_based_cf.py
│   └── hybrid.py
│
├── services/
│   ├── __init__.py
│   ├── auth_service.py
│   ├── database_service.py
│   ├── evaluation_service.py
│   ├── movie_service.py
│   ├── preference_service.py
│   ├── rating_service.py
│   └── recommendation_service.py
│
└── utils/
    ├── __init__.py
    ├── auth_ui.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── session.py
    └── ui.py
```

---

## 🧩 Architecture

CineMatch follows a modular monolithic architecture.

The project is divided into several logical layers:

```text
Streamlit UI
    ↓
Services Layer
    ↓
SQLite Database
    ↓
Recommendation Modules
    ↓
Evaluation Modules
```

### UI Layer

The UI layer is built with Streamlit.

It contains pages for:

* home dashboard;
* movie catalog;
* user profile;
* recommendations;
* evaluation metrics.

### Services Layer

The services layer contains business logic and database access.

Examples:

* authentication service;
* movie service;
* rating service;
* preference service;
* recommendation service;
* evaluation service.

### Database Layer

SQLite is used as the local database.

The database stores:

* movies;
* ratings;
* tags;
* users;
* favorite genres;
* favorite movies;
* watched movies;
* recommendation logs.

### Recommender Layer

This layer contains the actual recommendation algorithms:

* content-based recommender;
* user-based collaborative filtering;
* item-based collaborative filtering;
* hybrid recommender.

### Evaluation Layer

This layer contains metric functions and evaluation logic.

---

## 💾 Database

The database is stored locally as:

```text
database/cinematch.db
```

It is generated from:

```text
database/schema.sql
database/init_db.py
```

The SQLite database contains both:

* imported MovieLens data;
* CineMatch application data.

Main tables:

```text
movies
ratings
tags
users
favorite_genres
favorite_movies
watched_movies
recommendation_logs
```

---

## 📥 Dataset

CineMatch uses the MovieLens `ml-latest-small` dataset.

Required files:

```text
movies.csv
ratings.csv
tags.csv
links.csv
```

They should be placed in:

```text
data/ml-latest-small/
```

Expected structure:

```text
data/
└── ml-latest-small/
    ├── movies.csv
    ├── ratings.csv
    ├── tags.csv
    └── links.csv
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <repository-url>
cd CineMatch
```

---

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Add MovieLens dataset

Place the MovieLens files inside:

```text
data/ml-latest-small/
```

Required files:

```text
movies.csv
ratings.csv
tags.csv
links.csv
```

---

### 5. Initialize the database

Run:

```bash
python -m database.init_db
```

This creates:

```text
database/cinematch.db
```

and imports MovieLens data into SQLite.

---

### 6. Start the Streamlit app

```bash
streamlit run Home.py
```

The application will start at:

```text
http://localhost:8501
```

---

## 🧪 Recommended Testing Flow

After starting the app:

1. Register a new user.
2. Go to the Profile page.
3. Select favorite genres.
4. Go to the Movies page.
5. Search for movies such as:

```text
Matrix
Star Wars
Toy Story
Jurassic Park
Pulp Fiction
Forrest Gump
Terminator
```

6. Add some movies to favorites.
7. Mark some movies as watched.
8. Rate at least 3–5 movies.
9. Open the Recommendations page.
10. Test all recommendation tabs.
11. Open the Evaluation page and run the metrics.

---

## 📄 Application Pages

### Home

The home page contains:

* project overview;
* dataset statistics;
* user onboarding checklist;
* popular highly rated movies.

### Movies

The Movies page allows users to:

* browse movies;
* search by title;
* filter by genre;
* filter by year;
* sort movies;
* add favorites;
* mark watched movies;
* rate movies.

### Profile

The Profile page allows users to manage:

* favorite genres;
* favorite movies;
* rated movies;
* watched movies.

### Recommendations

The Recommendations page contains:

* Top recommendations;
* Personalized Content-Based recommendations;
* Because you watched;
* Based on favorite genres;
* Users with similar taste;
* Users who watched this also watched.

### Evaluation

The Evaluation page calculates:

* Precision@K;
* Recall@K;
* MAE.

---

## 🧠 Recommendation Sections

| Section                             | Algorithm                          |
| ----------------------------------- | ---------------------------------- |
| Top recommendations for you         | Hybrid Recommendation System       |
| Personalized Content-Based          | TF-IDF + Cosine Similarity         |
| Because you watched                 | Content-Based Filtering            |
| Based on favorite genres            | Genre-Based Cold Start             |
| Users with similar taste            | User-Based Collaborative Filtering |
| Users who watched this also watched | Item-Based Collaborative Filtering |

---

## 🔐 Authentication Notes

Passwords are not stored as plain text.

CineMatch uses password hashing with:

```text
PBKDF2-HMAC-SHA256
```

The stored password format is:

```text
salt$hash
```

---

## 📌 Current Limitations

The current version is built as an educational project.

Some limitations:

* no real deployment configuration;
* no external movie posters yet;
* no TMDB enrichment yet;
* no advanced model training pipeline;
* recommendation calculations may be slower for larger datasets;
* evaluation is simplified for educational purposes.

---

## 🚀 Possible Future Improvements

Possible improvements include:

* adding TMDB API integration for posters and movie descriptions;
* adding movie details pages;
* caching similarity matrices;
* adding advanced hybrid weighting strategies;
* adding matrix factorization;
* adding Surprise or implicit recommendation models;
* improving UI with posters and richer cards;
* deploying the app online;
* adding admin dashboard;
* adding recommendation explanation logs.

---

## 📚 Educational Value

CineMatch demonstrates the main concepts of recommender systems:

* cold start problem;
* content-based filtering;
* collaborative filtering;
* user-based similarity;
* item-based similarity;
* hybrid recommendations;
* time-aware user profiling;
* evaluation metrics.

The project shows not only the algorithms, but also how they can be integrated into a complete software application with UI, database, user profiles, and evaluation.

---

## 👨‍💻 Author

**Pavlin Georgiev**

Project for the course:

```text
Recommendation systems, summer semester, 2025/2026
```

---

## ✅ Project Status

CineMatch currently includes:

```text
✔ User authentication
✔ SQLite database
✔ MovieLens data import
✔ Movie catalog
✔ User profiles
✔ Favorite genres
✔ Favorite movies
✔ Ratings
✔ Watched movies
✔ Content-Based Filtering
✔ User-Based Collaborative Filtering
✔ Item-Based Collaborative Filtering
✔ Hybrid Recommendations
✔ Evaluation metrics
✔ Streamlit UI
```

---

## 🏁 Final Summary

CineMatch is a complete hybrid movie recommendation system.

It combines user preferences, movie metadata, historical ratings, collaborative filtering, and evaluation metrics into one working Streamlit application.

The system is designed to demonstrate how recommender systems work in practice and how multiple recommendation strategies can be combined to provide better personalized results.
