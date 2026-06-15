import pandas as pd
import streamlit as st

from services.database_service import load_movies_from_db, load_ratings_from_db
from services.movie_service import (
    get_all_genres,
    get_movies_with_rating_stats,
    search_movies,
)
from services.preference_service import (
    add_favorite_movie,
    get_favorite_movie_ids,
    remove_favorite_movie,
)
from services.rating_service import (
    get_user_rating,
    get_watched_movie_ids,
    mark_movie_as_watched,
    rate_movie,
    remove_rating,
    remove_watched_movie,
)
from utils.auth_ui import render_auth_sidebar
from utils.preprocessing import split_genres
from utils.session import (
    get_current_user_id,
    get_current_username,
    initialize_session_state,
    is_logged_in,
)
from utils.ui import (
    apply_global_styles,
    format_genres,
    format_year,
    render_page_header,
    render_section_intro,
)


st.set_page_config(
    page_title="Movies - CineMatch",
    page_icon="🎬",
    layout="wide"
)


@st.cache_data
def get_data():
    """
    Loads movies and ratings from SQLite.
    """
    movies = load_movies_from_db()
    ratings = load_ratings_from_db()

    movies["genres_list"] = movies["genres"].apply(split_genres)

    movies_with_stats = get_movies_with_rating_stats(movies, ratings)

    return movies_with_stats, ratings


def render_movie_card(
    movie,
    favorite_movie_ids: set,
    watched_movie_ids: set,
):
    """
    Renders a movie card with favorite, watched and rating actions.
    """
    title = movie["clean_title"]
    year = format_year(movie["year"])
    genres = format_genres(movie["genres_list"])
    avg_rating = movie["avg_rating"]
    rating_count = int(movie["rating_count"])
    movie_id = int(movie["movieId"])

    with st.container(border=True):
        st.subheader(f"🎬 {title}")
        st.caption(f"Year: {year}")
        st.write(f"**Genres:** {genres}")

        if rating_count > 0:
            st.write(f"⭐ **Average rating:** {avg_rating:.2f} / 5")
            st.write(f"👥 **Ratings:** {rating_count}")
        else:
            st.write("⭐ **Average rating:** No ratings yet")

        st.caption(f"MovieLens ID: {movie_id}")

        if not is_logged_in():
            st.info("Login to rate, mark as watched or save this movie.")
            return

        user_id = get_current_user_id()

        st.divider()

        favorite_col, watched_col = st.columns(2)

        with favorite_col:
            if movie_id in favorite_movie_ids:
                st.success("In favorites")

                if st.button(
                    "Remove favorite",
                    key=f"remove_favorite_{movie_id}"
                ):
                    remove_favorite_movie(user_id, movie_id)
                    st.success("Movie removed from favorites.")
                    st.rerun()
            else:
                if st.button(
                    "Add favorite",
                    key=f"add_favorite_{movie_id}"
                ):
                    was_added = add_favorite_movie(user_id, movie_id)

                    if was_added:
                        st.success("Movie added to favorites.")
                    else:
                        st.info("Movie is already in favorites.")

                    st.rerun()

        with watched_col:
            if movie_id in watched_movie_ids:
                st.success("Watched")

                if st.button(
                    "Remove watched",
                    key=f"remove_watched_{movie_id}"
                ):
                    remove_watched_movie(user_id, movie_id)
                    st.success("Movie removed from watched list.")
                    st.rerun()
            else:
                if st.button(
                    "Mark watched",
                    key=f"mark_watched_{movie_id}"
                ):
                    was_added = mark_movie_as_watched(user_id, movie_id)

                    if was_added:
                        st.success("Movie marked as watched.")
                    else:
                        st.info("Movie is already marked as watched.")

                    st.rerun()

        st.divider()

        current_rating = get_user_rating(user_id, movie_id)

        if current_rating is not None:
            st.info(f"Your rating: {current_rating:.1f} / 5")

        selected_rating = st.slider(
            "Your rating",
            min_value=0.5,
            max_value=5.0,
            value=float(current_rating) if current_rating is not None else 3.0,
            step=0.5,
            key=f"rating_slider_{movie_id}"
        )

        rating_col1, rating_col2 = st.columns(2)

        with rating_col1:
            if st.button(
                "Save rating",
                key=f"save_rating_{movie_id}"
            ):
                rate_movie(user_id, movie_id, selected_rating)
                st.success("Rating saved.")
                st.cache_data.clear()
                st.rerun()

        with rating_col2:
            if current_rating is not None:
                if st.button(
                    "Remove rating",
                    key=f"remove_rating_{movie_id}"
                ):
                    remove_rating(user_id, movie_id)
                    st.success("Rating removed.")
                    st.cache_data.clear()
                    st.rerun()


def main():
    initialize_session_state()
    apply_global_styles()
    render_auth_sidebar("movies")

    render_page_header(
        title="🎬 Movie Catalog",
        subtitle="Browse, search, rate and save movies. Your actions help CineMatch learn your taste.",
        badge="MovieLens Dataset"
    )

    if not is_logged_in():
        st.warning("You can browse movies, but login is required to rate or save favorites.")
        favorite_movie_ids = set()
        watched_movie_ids = set()
    else:
        st.success(f"You are logged in as {get_current_username()}.")
        user_id = get_current_user_id()
        favorite_movie_ids = get_favorite_movie_ids(user_id)
        watched_movie_ids = get_watched_movie_ids(user_id)

    movies, ratings = get_data()

    st.divider()

    render_section_intro(
        "Search and filters",
        "Use the filters below to find movies by title, genre, year and rating popularity."
    )

    all_genres = get_all_genres(movies)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_query = st.text_input(
            "Search by movie title",
            placeholder="Example: Toy Story, Matrix, Star Wars..."
        )

    with col2:
        selected_genre = st.selectbox(
            "Filter by genre",
            ["All"] + all_genres
        )

    with col3:
        sort_option = st.selectbox(
            "Sort by",
            [
                "Newest first",
                "Oldest first",
                "Highest rated",
                "Most rated",
                "Title A-Z"
            ]
        )

    years = movies["year"].dropna()

    min_available_year = int(years.min())
    max_available_year = int(years.max())

    selected_year_range = st.slider(
        "Release year range",
        min_value=min_available_year,
        max_value=max_available_year,
        value=(min_available_year, max_available_year)
    )

    filtered_movies = search_movies(
        movies=movies,
        search_query=search_query,
        selected_genre=selected_genre,
        min_year=selected_year_range[0],
        max_year=selected_year_range[1],
    )

    if sort_option == "Newest first":
        filtered_movies = filtered_movies.sort_values(
            by=["year", "clean_title"],
            ascending=[False, True]
        )
    elif sort_option == "Oldest first":
        filtered_movies = filtered_movies.sort_values(
            by=["year", "clean_title"],
            ascending=[True, True]
        )
    elif sort_option == "Highest rated":
        filtered_movies = filtered_movies.sort_values(
            by=["avg_rating", "rating_count"],
            ascending=[False, False]
        )
    elif sort_option == "Most rated":
        filtered_movies = filtered_movies.sort_values(
            by=["rating_count", "avg_rating"],
            ascending=[False, False]
        )
    elif sort_option == "Title A-Z":
        filtered_movies = filtered_movies.sort_values(
            by="clean_title",
            ascending=True
        )

    st.divider()

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Found movies", f"{len(filtered_movies):,}")

    with result_col2:
        if selected_genre == "All":
            st.metric("Genre filter", "All")
        else:
            st.metric("Genre filter", selected_genre)

    with result_col3:
        st.metric("Sort", sort_option)

    movies_per_page = st.selectbox(
        "Movies per page",
        [6, 12, 24, 48],
        index=0
    )

    total_pages = max(1, (len(filtered_movies) - 1) // movies_per_page + 1)

    current_page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1
    )

    start_index = (current_page - 1) * movies_per_page
    end_index = start_index + movies_per_page

    page_movies = filtered_movies.iloc[start_index:end_index]

    st.caption(f"Page {current_page} of {total_pages}")

    if page_movies.empty:
        st.info("No movies found with the selected filters.")
        return

    columns = st.columns(3)

    for index, (_, movie) in enumerate(page_movies.iterrows()):
        with columns[index % 3]:
            render_movie_card(
                movie=movie,
                favorite_movie_ids=favorite_movie_ids,
                watched_movie_ids=watched_movie_ids
            )


if __name__ == "__main__":
    main()