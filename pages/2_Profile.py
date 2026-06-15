import streamlit as st

from services.database_service import load_movies_from_db
from services.movie_service import get_all_genres
from services.preference_service import (
    get_favorite_genres,
    get_favorite_movies,
    remove_favorite_movie,
    set_favorite_genres,
)
from services.rating_service import (
    get_user_ratings,
    get_watched_movies,
    remove_rating,
    remove_watched_movie,
)
from utils.auth_ui import render_auth_sidebar
from utils.preprocessing import split_genres
from utils.session import (
    get_current_user,
    get_current_user_id,
    initialize_session_state,
    is_logged_in,
)


st.set_page_config(
    page_title="Profile - CineMatch",
    page_icon="👤",
    layout="wide"
)


@st.cache_data
def get_movies():
    """
    Loads movies from SQLite and prepares genres_list.
    """
    movies = load_movies_from_db()
    movies["genres_list"] = movies["genres"].apply(split_genres)

    return movies


def render_movie_list_item(movie, remove_button_label: str, remove_callback, key_prefix: str):
    """
    Renders a simple movie list item with a remove button.
    """
    with st.container(border=True):
        col1, col2 = st.columns([4, 1])

        with col1:
            st.subheader(movie["clean_title"])

            if movie["year"]:
                st.caption(f"Year: {int(movie['year'])}")
            else:
                st.caption("Year: Unknown")

            st.write(f"**Genres:** {movie['genres']}")

        with col2:
            if st.button(
                remove_button_label,
                key=f"{key_prefix}_{movie['movieId']}"
            ):
                remove_callback(int(movie["movieId"]))
                st.success("Removed successfully.")
                st.rerun()


def main():
    initialize_session_state()
    render_auth_sidebar("profile")

    st.title("👤 User Profile")

    if not is_logged_in():
        st.warning("Please login or create an account to manage your profile.")
        return

    current_user = get_current_user()
    user_id = get_current_user_id()

    st.success(f"Logged in as {current_user['username']}")

    st.write(
        """
        Manage your preferences and activity.  
        CineMatch will use these signals for cold start, content-based recommendations,
        collaborative filtering and time-aware user profiling.
        """
    )

    st.divider()

    movies = get_movies()
    all_genres = get_all_genres(movies)
    current_favorite_genres = get_favorite_genres(user_id)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Favorite genres",
            "Favorite movies",
            "Rated movies",
            "Watched movies"
        ]
    )

    with tab1:
        st.subheader("🎭 Favorite genres")

        selected_genres = st.multiselect(
            "Choose your favorite genres",
            all_genres,
            default=current_favorite_genres
        )

        if st.button("Save favorite genres"):
            set_favorite_genres(user_id, selected_genres)
            st.success("Favorite genres updated successfully.")
            st.rerun()

        if selected_genres:
            st.info(f"Current selected genres: {', '.join(selected_genres)}")
        else:
            st.info("No favorite genres selected yet.")

    with tab2:
        st.subheader("⭐ Favorite movies")

        favorite_movies = get_favorite_movies(user_id)

        if favorite_movies.empty:
            st.info("You have not added any favorite movies yet. Open the Movies page and add some favorites.")
        else:
            st.write(f"You have **{len(favorite_movies)}** favorite movies.")

            for _, movie in favorite_movies.iterrows():
                render_movie_list_item(
                    movie=movie,
                    remove_button_label="Remove",
                    remove_callback=lambda movie_id: remove_favorite_movie(user_id, movie_id),
                    key_prefix="remove_favorite_profile"
                )

    with tab3:
        st.subheader("⭐ Rated movies")

        user_ratings = get_user_ratings(user_id)

        if user_ratings.empty:
            st.info("You have not rated any movies yet.")
        else:
            st.write(f"You have rated **{len(user_ratings)}** movies.")

            for _, movie in user_ratings.iterrows():
                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.subheader(movie["clean_title"])

                        if movie["year"]:
                            st.caption(f"Year: {int(movie['year'])}")
                        else:
                            st.caption("Year: Unknown")

                        st.write(f"**Genres:** {movie['genres']}")
                        st.write(f"⭐ **Your rating:** {movie['rating']:.1f} / 5")
                        st.caption(f"Rated at: {movie['created_at']}")

                    with col2:
                        if st.button(
                            "Remove rating",
                            key=f"remove_rating_profile_{movie['movieId']}"
                        ):
                            remove_rating(user_id, int(movie["movieId"]))
                            st.success("Rating removed.")
                            st.rerun()

    with tab4:
        st.subheader("✅ Watched movies")

        watched_movies = get_watched_movies(user_id)

        if watched_movies.empty:
            st.info("You have not marked any movies as watched yet.")
        else:
            st.write(f"You have watched **{len(watched_movies)}** movies.")

            for _, movie in watched_movies.iterrows():
                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        st.subheader(movie["clean_title"])

                        if movie["year"]:
                            st.caption(f"Year: {int(movie['year'])}")
                        else:
                            st.caption("Year: Unknown")

                        st.write(f"**Genres:** {movie['genres']}")
                        st.caption(f"Watched at: {movie['watched_at']}")

                    with col2:
                        if st.button(
                            "Remove watched",
                            key=f"remove_watched_profile_{movie['movieId']}"
                        ):
                            remove_watched_movie(user_id, int(movie["movieId"]))
                            st.success("Movie removed from watched list.")
                            st.rerun()


if __name__ == "__main__":
    main()