import streamlit as st

from services.preference_service import get_favorite_movie_ids
from services.rating_service import get_watched_movie_ids
from services.recommendation_service import (
    get_content_recommendations,
    get_genre_based_recommendations,
    get_hybrid_recommendations,
    get_item_based_cf_for_movie,
    get_item_based_cf_recommendations,
    get_similar_movies_for_movie,
    get_user_based_cf_recommendations,
    get_user_seed_movies_for_because_you_watched,
)
from utils.auth_ui import render_auth_sidebar
from utils.session import (
    get_current_user_id,
    get_current_username,
    initialize_session_state,
    is_logged_in,
)
from utils.ui import format_genres


st.set_page_config(
    page_title="Recommendations - CineMatch",
    page_icon="✨",
    layout="wide"
)


@st.cache_data
def get_cached_hybrid_recommendations(
    user_id: int,
    top_n: int,
    content_weight: float,
    genre_weight: float,
    user_based_weight: float,
    item_based_weight: float
):
    return get_hybrid_recommendations(
        user_id=user_id,
        top_n=top_n,
        content_weight=content_weight,
        genre_weight=genre_weight,
        user_based_weight=user_based_weight,
        item_based_weight=item_based_weight
    )


@st.cache_data
def get_cached_content_recommendations(user_id: int, top_n: int):
    return get_content_recommendations(user_id, top_n)


@st.cache_data
def get_cached_similar_movies(movie_id: int, top_n: int, exclude_ids_tuple):
    exclude_movie_ids = set(exclude_ids_tuple)

    return get_similar_movies_for_movie(
        movie_id=movie_id,
        top_n=top_n,
        exclude_movie_ids=exclude_movie_ids
    )


@st.cache_data
def get_cached_genre_recommendations(user_id: int, top_n: int, min_ratings: int):
    return get_genre_based_recommendations(
        user_id=user_id,
        top_n=top_n,
        min_ratings=min_ratings
    )


@st.cache_data
def get_cached_user_based_recommendations(
    user_id: int,
    top_n: int,
    similar_users_count: int,
    min_common_items: int,
    min_neighbor_rating: float
):
    return get_user_based_cf_recommendations(
        user_id=user_id,
        top_n=top_n,
        similar_users_count=similar_users_count,
        min_common_items=min_common_items,
        min_neighbor_rating=min_neighbor_rating
    )


@st.cache_data
def get_cached_item_based_recommendations(
    user_id: int,
    top_n: int,
    similar_items_count: int,
    min_common_users: int
):
    return get_item_based_cf_recommendations(
        user_id=user_id,
        top_n=top_n,
        similar_items_count=similar_items_count,
        min_common_users=min_common_users
    )


@st.cache_data
def get_cached_item_based_for_movie(
    user_id: int,
    movie_id: int,
    top_n: int,
    min_common_users: int
):
    return get_item_based_cf_for_movie(
        user_id=user_id,
        movie_id=movie_id,
        top_n=top_n,
        min_common_users=min_common_users
    )


def render_recommendation_card(movie, score_column=None):
    """
    Renders a recommendation card.
    """
    with st.container(border=True):
        st.subheader(movie["clean_title"])

        if movie["year"]:
            st.caption(f"Year: {int(movie['year'])}")
        else:
            st.caption("Year: Unknown")

        st.write(f"**Genres:** {format_genres(movie['genres'])}")

        if score_column is not None and score_column in movie:
            st.write(f"🎯 **Score:** {movie[score_column]:.4f}")

        if "hybrid_score" in movie:
            st.write(f"🏆 **Hybrid score:** {movie['hybrid_score']:.4f}")

        if "sources" in movie:
            st.caption(f"Used algorithms: {movie['sources']}")

        if "algorithm_count" in movie:
            st.caption(f"Matched by {int(movie['algorithm_count'])} algorithm(s)")

        if "reason_signals" in movie:
            st.caption(f"Recommendation signals: {movie['reason_signals']}")

        if "matched_genres_text" in movie:
            st.caption(f"Matched genres: {movie['matched_genres_text']}")

        if "predicted_rating" in movie:
            st.write(f"🤝 **Predicted rating:** {movie['predicted_rating']:.2f} / 5")

        if "item_similarity" in movie:
            st.write(f"🔗 **Item similarity:** {movie['item_similarity']:.4f}")

        if "item_cf_score" in movie:
            st.write(f"🔗 **Item-based score:** {movie['item_cf_score']:.4f}")

        if "support_count" in movie:
            st.write(f"👥 **Support count:** {int(movie['support_count'])}")

        if "common_users" in movie:
            st.write(f"👥 **Common users:** {int(movie['common_users'])}")

        if "avg_common_users" in movie:
            st.caption(f"Average common users: {movie['avg_common_users']:.2f}")

        if "avg_neighbor_rating" in movie:
            st.caption(
                f"Average rating from similar users: {movie['avg_neighbor_rating']:.2f}"
            )

        if "avg_similarity" in movie:
            st.caption(f"Average item similarity: {movie['avg_similarity']:.4f}")

        if "avg_rating" in movie and "rating_count" in movie:
            st.write(f"⭐ **Average rating:** {movie['avg_rating']:.2f} / 5")
            st.write(f"👥 **Ratings:** {int(movie['rating_count'])}")

        st.caption(f"MovieLens ID: {int(movie['movieId'])}")


def render_hybrid_tab(user_id: int):
    st.subheader("🏆 Top recommendations for you")

    st.write(
        """
        This is the final hybrid recommendation section.
        CineMatch combines multiple recommendation strategies into one ranking:

        - Content-Based Filtering
        - Genre-Based Cold Start
        - User-Based Collaborative Filtering
        - Item-Based Collaborative Filtering
        """
    )

    top_n = st.slider(
        "Number of hybrid recommendations",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        key="hybrid_top_n"
    )

    with st.expander("Hybrid weights"):
        st.write(
            """
            These weights control how much each algorithm contributes to the final score.
            The default values are balanced for the current project stage.
            """
        )

        content_weight = st.slider(
            "Content-Based weight",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.05,
            key="content_weight"
        )

        genre_weight = st.slider(
            "Genre-Based weight",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.05,
            key="genre_weight"
        )

        user_based_weight = st.slider(
            "User-Based CF weight",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key="user_based_weight"
        )

        item_based_weight = st.slider(
            "Item-Based CF weight",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.05,
            key="item_based_weight"
        )

    total_weight = (
        content_weight +
        genre_weight +
        user_based_weight +
        item_based_weight
    )

    if total_weight == 0:
        st.error("At least one weight must be greater than 0.")
        return

    content_weight = content_weight / total_weight
    genre_weight = genre_weight / total_weight
    user_based_weight = user_based_weight / total_weight
    item_based_weight = item_based_weight / total_weight

    st.caption(
        f"Normalized weights → Content: {content_weight:.2f}, "
        f"Genre: {genre_weight:.2f}, "
        f"User-Based: {user_based_weight:.2f}, "
        f"Item-Based: {item_based_weight:.2f}"
    )

    recommendations = get_cached_hybrid_recommendations(
        user_id=user_id,
        top_n=top_n,
        content_weight=content_weight,
        genre_weight=genre_weight,
        user_based_weight=user_based_weight,
        item_based_weight=item_based_weight
    )

    if recommendations.empty:
        st.warning(
            """
            No hybrid recommendations found yet.

            Add favorite genres, favorite movies, watched movies or ratings first.
            """
        )
        return

    columns = st.columns(3)

    for index, (_, movie) in enumerate(recommendations.iterrows()):
        with columns[index % 3]:
            render_recommendation_card(movie, score_column="hybrid_score")


def render_personalized_content_tab(user_id: int):
    st.subheader("🎬 Personalized content-based recommendations")

    st.write(
        """
        These recommendations are generated from your favorite movies,
        watched movies and highly rated movies.
        """
    )

    top_n = st.slider(
        "Number of personalized recommendations",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        key="personalized_top_n"
    )

    recommendations = get_cached_content_recommendations(user_id, top_n)

    if recommendations.empty:
        st.warning(
            """
            Not enough information to generate recommendations yet.

            Add some favorite movies, mark movies as watched, or rate a few movies with 4.0+.
            """
        )
        return

    columns = st.columns(3)

    for index, (_, movie) in enumerate(recommendations.iterrows()):
        with columns[index % 3]:
            render_recommendation_card(movie, score_column="content_score")


def render_because_you_watched_tab(user_id: int):
    st.subheader("👀 Because you watched...")

    st.write(
        """
        Choose a movie from your watched, favorite or rated movies.
        CineMatch will find similar movies using TF-IDF and cosine similarity.
        """
    )

    seed_movies = get_user_seed_movies_for_because_you_watched(user_id)

    if seed_movies.empty:
        st.warning(
            """
            You do not have enough movie activity yet.

            Mark a movie as watched, add a favorite movie, or rate a movie first.
            """
        )
        return

    seed_movie_options = {
        f"{row['clean_title']} ({int(row['year']) if row['year'] else 'Unknown'}) — {row['source_signal']}": int(row["movieId"])
        for _, row in seed_movies.iterrows()
    }

    selected_label = st.selectbox(
        "Select a movie",
        list(seed_movie_options.keys()),
        key="content_seed_movie"
    )

    selected_movie_id = seed_movie_options[selected_label]

    top_n = st.slider(
        "Number of similar movies",
        min_value=5,
        max_value=30,
        value=10,
        step=5,
        key="because_watched_top_n"
    )

    watched_ids = get_watched_movie_ids(user_id)
    favorite_ids = get_favorite_movie_ids(user_id)

    exclude_movie_ids = watched_ids.union(favorite_ids)
    exclude_movie_ids.add(selected_movie_id)

    similar_movies = get_cached_similar_movies(
        movie_id=selected_movie_id,
        top_n=top_n,
        exclude_ids_tuple=tuple(sorted(exclude_movie_ids))
    )

    st.info(f"Showing recommendations because of: **{selected_label}**")

    if similar_movies.empty:
        st.warning("No similar movies found for the selected movie.")
        return

    columns = st.columns(3)

    for index, (_, movie) in enumerate(similar_movies.iterrows()):
        with columns[index % 3]:
            render_recommendation_card(movie, score_column="content_score")


def render_genre_based_tab(user_id: int):
    st.subheader("🎭 Based on your favorite genres")

    st.write(
        """
        These recommendations are useful for new users.
        CineMatch uses your selected favorite genres and recommends popular,
        highly rated movies from those genres.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        top_n = st.slider(
            "Number of genre-based recommendations",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            key="genre_top_n"
        )

    with col2:
        min_ratings = st.slider(
            "Minimum number of ratings",
            min_value=10,
            max_value=300,
            value=50,
            step=10,
            key="genre_min_ratings"
        )

    recommendations = get_cached_genre_recommendations(
        user_id=user_id,
        top_n=top_n,
        min_ratings=min_ratings
    )

    if recommendations.empty:
        st.warning(
            """
            No genre-based recommendations found.

            Go to your Profile page and choose at least one favorite genre.
            If you already selected genres, try lowering the minimum number of ratings.
            """
        )
        return

    columns = st.columns(3)

    for index, (_, movie) in enumerate(recommendations.iterrows()):
        with columns[index % 3]:
            render_recommendation_card(movie, score_column="genre_score")


def render_user_based_cf_tab(user_id: int):
    st.subheader("🤝 Users with similar taste also liked...")

    st.write(
        """
        These recommendations are based on users who rated movies similarly to you.
        CineMatch uses Pearson correlation to find similar users.
        """
    )

    st.info(
        """
        For this method to work well, rate at least 3–5 movies.
        The more ratings you provide, the better the similar-user recommendations become.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        top_n = st.slider(
            "Number of user-based recommendations",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            key="user_based_top_n"
        )

        similar_users_count = st.slider(
            "Number of similar users",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            key="similar_users_count"
        )

    with col2:
        min_common_items = st.slider(
            "Minimum commonly rated movies",
            min_value=2,
            max_value=10,
            value=2,
            step=1,
            key="min_common_items"
        )

        min_neighbor_rating = st.slider(
            "Minimum rating from similar users",
            min_value=3.0,
            max_value=5.0,
            value=4.0,
            step=0.5,
            key="min_neighbor_rating"
        )

    recommendations = get_cached_user_based_recommendations(
        user_id=user_id,
        top_n=top_n,
        similar_users_count=similar_users_count,
        min_common_items=min_common_items,
        min_neighbor_rating=min_neighbor_rating
    )

    if recommendations.empty:
        st.warning(
            """
            No user-based collaborative recommendations found yet.

            Try rating more movies first. For best results, rate several popular movies
            so CineMatch can compare your taste with MovieLens users.
            """
        )
        return

    columns = st.columns(3)

    for index, (_, movie) in enumerate(recommendations.iterrows()):
        with columns[index % 3]:
            render_recommendation_card(movie)


def render_item_based_cf_tab(user_id: int):
    st.subheader("🔗 Users who watched this also watched...")

    st.write(
        """
        These recommendations are based on item-item similarity.
        CineMatch compares movies by their rating patterns and finds movies
        that are often liked by the same users.
        """
    )

    seed_movies = get_user_seed_movies_for_because_you_watched(user_id)

    if seed_movies.empty:
        st.warning(
            """
            You need at least one watched, favorite or rated movie before item-based
            collaborative filtering can generate recommendations.
            """
        )
        return

    mode = st.radio(
        "Recommendation mode",
        [
            "Based on selected movie",
            "Based on all my activity"
        ],
        key="item_based_mode"
    )

    col1, col2 = st.columns(2)

    with col1:
        top_n = st.slider(
            "Number of item-based recommendations",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
            key="item_based_top_n"
        )

    with col2:
        min_common_users = st.slider(
            "Minimum common users between movies",
            min_value=2,
            max_value=20,
            value=2,
            step=1,
            key="item_min_common_users"
        )

    if mode == "Based on selected movie":
        seed_movie_options = {
            f"{row['clean_title']} ({int(row['year']) if row['year'] else 'Unknown'}) — {row['source_signal']}": int(row["movieId"])
            for _, row in seed_movies.iterrows()
        }

        selected_label = st.selectbox(
            "Select a movie",
            list(seed_movie_options.keys()),
            key="item_seed_movie"
        )

        selected_movie_id = seed_movie_options[selected_label]

        recommendations = get_cached_item_based_for_movie(
            user_id=user_id,
            movie_id=selected_movie_id,
            top_n=top_n,
            min_common_users=min_common_users
        )

        st.info(f"Showing item-based recommendations for: **{selected_label}**")

        score_column = "item_similarity"

    else:
        similar_items_count = st.slider(
            "Similar movies checked per seed movie",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            key="similar_items_count"
        )

        recommendations = get_cached_item_based_recommendations(
            user_id=user_id,
            top_n=top_n,
            similar_items_count=similar_items_count,
            min_common_users=min_common_users
        )

        st.info("Showing item-based recommendations using all your movie activity.")

        score_column = "item_cf_score"

    if recommendations.empty:
        st.warning(
            """
            No item-based collaborative recommendations found yet.

            Try rating or marking more popular movies as watched.
            """
        )
        return

    columns = st.columns(3)

    for index, (_, movie) in enumerate(recommendations.iterrows()):
        with columns[index % 3]:
            render_recommendation_card(movie, score_column=score_column)


def main():
    initialize_session_state()
    render_auth_sidebar("recommendations")

    st.title("✨ Recommendations")

    st.write(
        """
        CineMatch currently supports:

        - Hybrid Recommendation System
        - Content-Based Filtering with TF-IDF and cosine similarity
        - Genre-based cold start recommendations
        - User-Based Collaborative Filtering with Pearson correlation
        - Item-Based Collaborative Filtering with Pearson correlation
        """
    )

    if not is_logged_in():
        st.warning("Please login to see personalized recommendations.")
        return

    user_id = get_current_user_id()

    st.success(f"Recommendations for {get_current_username()}")

    if st.button("Refresh recommendations"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Top recommendations",
            "Personalized Content-Based",
            "Because you watched",
            "Based on favorite genres",
            "Users with similar taste",
            "Users who watched this also watched"
        ]
    )

    with tab1:
        render_hybrid_tab(user_id)

    with tab2:
        render_personalized_content_tab(user_id)

    with tab3:
        render_because_you_watched_tab(user_id)

    with tab4:
        render_genre_based_tab(user_id)

    with tab5:
        render_user_based_cf_tab(user_id)

    with tab6:
        render_item_based_cf_tab(user_id)


if __name__ == "__main__":
    main()