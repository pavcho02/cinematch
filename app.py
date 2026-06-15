import streamlit as st

from services.database_service import (
    database_exists,
    get_database_summary,
    load_movies_from_db,
    load_ratings_from_db,
)
from services.movie_service import get_popular_movies
from services.preference_service import (
    get_favorite_genres,
    get_favorite_movies,
)
from services.rating_service import (
    get_user_ratings,
    get_watched_movies,
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
    render_feature_card,
    render_movie_preview_card,
    render_page_header,
    render_section_intro,
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


def render_project_overview():
    """
    Renders the main project feature overview.
    """
    render_section_intro(
        "Какво представлява CineMatch?",
        "CineMatch е хибридна препоръчваща система за филми, която комбинира няколко подхода за персонализирани препоръки."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        render_feature_card(
            "🎯",
            "Content-Based Filtering",
            "Препоръчва филми според жанрове, тагове и сходство между съдържанието на филмите."
        )

    with col2:
        render_feature_card(
            "🤝",
            "Collaborative Filtering",
            "Използва оценки от потребители и открива сходни потребители или сходни филми."
        )

    with col3:
        render_feature_card(
            "🏆",
            "Hybrid Recommendations",
            "Комбинира Content-Based, User-Based, Item-Based и Genre-Based подходи."
        )


def render_user_onboarding():
    """
    Renders onboarding guidance for the logged-in user.
    """
    if not is_logged_in():
        st.info("Влез в профила си или създай акаунт, за да получаваш персонализирани препоръки.")
        return

    user_id = get_current_user_id()

    favorite_genres = get_favorite_genres(user_id)
    favorite_movies = get_favorite_movies(user_id)
    rated_movies = get_user_ratings(user_id)
    watched_movies = get_watched_movies(user_id)

    st.subheader(f"👋 Здравей, {get_current_username()}!")

    st.write(
        """
        За да получиш по-добри препоръки, CineMatch използва твоите жанрове,
        любими филми, оценки и гледани филми.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Любими жанрове", len(favorite_genres))

    with col2:
        st.metric("Любими филми", len(favorite_movies))

    with col3:
        st.metric("Оценени филми", len(rated_movies))

    with col4:
        st.metric("Гледани филми", len(watched_movies))

    st.divider()

    checklist_items = [
        {
            "done": len(favorite_genres) > 0,
            "text": "Избери любими жанрове в Profile страницата."
        },
        {
            "done": len(favorite_movies) > 0,
            "text": "Добави поне няколко любими филма от Movies страницата."
        },
        {
            "done": len(rated_movies) >= 3,
            "text": "Оцени поне 3–5 филма, за да работи по-добре Collaborative Filtering."
        },
        {
            "done": len(watched_movies) > 0,
            "text": "Маркирай филми като гледани, за да се активират секциите 'Because you watched'."
        },
    ]

    st.subheader("✅ Препоръчани стъпки")

    for item in checklist_items:
        if item["done"]:
            st.success(item["text"])
        else:
            st.warning(item["text"])


def render_popular_movies(movies, ratings):
    """
    Renders popular movies on the home page.
    """
    render_section_intro(
        "🔥 Популярни високо оценени филми",
        "Това са филми с много оценки и висока средна оценка в MovieLens dataset-а."
    )

    popular_movies = get_popular_movies(
        movies=movies,
        ratings=ratings,
        min_ratings=100,
        top_n=6
    )

    columns = st.columns(3)

    for index, (_, movie) in enumerate(popular_movies.iterrows()):
        with columns[index % 3]:
            render_movie_preview_card(movie)


def main():
    initialize_session_state()
    apply_global_styles()
    render_auth_sidebar("home")

    render_page_header(
        title="🎬 CineMatch",
        subtitle="Hybrid Movie Recommendation System with Content-Based, Collaborative and Hybrid recommendations.",
        badge="Recommendation Systems Project"
    )

    if not database_exists():
        st.error("CineMatch database was not found.")
        st.info(
            """
            Initialize the database first:

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

    st.divider()

    render_user_onboarding()

    st.divider()

    render_project_overview()

    st.divider()

    movies, ratings = get_data_from_database()

    render_popular_movies(movies, ratings)

    st.divider()

    st.info(
        """
        Използвай страниците от sidebar менюто:

        - **Movies** — търсене, оценяване, любими и гледани филми;
        - **Profile** — любими жанрове, любими филми, оценки и гледани филми;
        - **Recommendations** — всички препоръчващи алгоритми;
        - **Evaluation** — Precision@K, Recall@K и MAE.
        """
    )


if __name__ == "__main__":
    main()