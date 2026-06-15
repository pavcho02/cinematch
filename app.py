import streamlit as st

from services.database_service import (
    database_exists,
    get_database_summary,
    load_movies_from_db,
    load_ratings_from_db,
)
from services.movie_service import get_popular_movies
from utils.auth_ui import render_auth_sidebar
from utils.preprocessing import split_genres
from utils.session import (
    get_current_username,
    initialize_session_state,
    is_logged_in,
)


st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def get_data_from_database():
    """
    Loads movies and ratings from SQLite database.
    """
    movies = load_movies_from_db()
    ratings = load_ratings_from_db()

    movies["genres_list"] = movies["genres"].apply(split_genres)

    return movies, ratings


def main():
    initialize_session_state()
    render_auth_sidebar("home")

    st.title("🎬 CineMatch")
    st.subheader("Hybrid Movie Recommendation System")

    if is_logged_in():
        st.success(f"Welcome back, {get_current_username()}!")
    else:
        st.info("Login or create an account from the sidebar to start using CineMatch.")

    st.write(
        """
        CineMatch is a movie recommendation system that combines:
        
        - Content-Based Filtering
        - User-Based Collaborative Filtering
        - Item-Based Collaborative Filtering
        - Hybrid Recommendations
        - Time-Aware User Profiling
        """
    )

    st.divider()

    if not database_exists():
        st.error("CineMatch database was not found.")
        st.info(
            """
            Please initialize the database first:
            
            ```bash
            python -m database.init_db
            ```
            """
        )
        return

    summary = get_database_summary()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Movies", f"{summary.get('movies', 0):,}")

    with col2:
        st.metric("Ratings", f"{summary.get('ratings', 0):,}")

    with col3:
        st.metric("Tags", f"{summary.get('tags', 0):,}")

    with col4:
        st.metric("CineMatch Users", f"{summary.get('users', 0):,}")

    st.success("SQLite database loaded successfully.")

    st.divider()

    movies, ratings = get_data_from_database()

    st.subheader("🔥 Popular highly rated movies")

    popular_movies = get_popular_movies(
        movies=movies,
        ratings=ratings,
        min_ratings=100,
        top_n=10
    )

    st.dataframe(
        popular_movies[
            [
                "movieId",
                "clean_title",
                "year",
                "genres",
                "avg_rating",
                "rating_count"
            ]
        ],
        use_container_width=True
    )

    st.info("Open the Movies page from the sidebar to browse the full movie catalog.")


if __name__ == "__main__":
    main()