import streamlit as st

from services.evaluation_service import evaluate_recommenders
from utils.auth_ui import render_auth_sidebar
from utils.session import initialize_session_state


st.set_page_config(
    page_title="Evaluation - CineMatch",
    page_icon="📊",
    layout="wide"
)


@st.cache_data
def get_cached_evaluation(
    k: int,
    max_users: int,
    min_user_ratings: int,
    test_size: float,
    relevance_threshold: float,
    random_state: int
):
    """
    Caches evaluation results because evaluation can take some time.
    """
    return evaluate_recommenders(
        k=k,
        max_users=max_users,
        min_user_ratings=min_user_ratings,
        test_size=test_size,
        relevance_threshold=relevance_threshold,
        random_state=random_state
    )


def main():
    initialize_session_state()
    render_auth_sidebar("evaluation")

    st.title("📊 Evaluation Metrics")
    st.write(
        """
        This page evaluates CineMatch recommendation algorithms using MovieLens ratings.

        The evaluation uses a train/test split:
        - train ratings are used to generate recommendations;
        - test ratings are used to check whether the recommendations were relevant.
        """
    )

    st.divider()

    st.subheader("Evaluation settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        k = st.slider(
            "K for Precision@K and Recall@K",
            min_value=5,
            max_value=30,
            value=10,
            step=5
        )

        max_users = st.slider(
            "Number of MovieLens users to evaluate",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )

    with col2:
        min_user_ratings = st.slider(
            "Minimum ratings per user",
            min_value=10,
            max_value=100,
            value=20,
            step=5
        )

        test_size = st.slider(
            "Test set size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05
        )

    with col3:
        relevance_threshold = st.slider(
            "Relevant rating threshold",
            min_value=3.0,
            max_value=5.0,
            value=4.0,
            step=0.5
        )

        random_state = st.number_input(
            "Random state",
            min_value=1,
            max_value=9999,
            value=42,
            step=1
        )

    st.info(
        """
        Evaluation can take some time because several algorithms are executed.
        For quick testing, use 5–20 users. For more stable results, increase the number of users.
        """
    )

    run_evaluation = st.button("Run evaluation")

    if not run_evaluation:
        st.warning("Click **Run evaluation** to calculate the metrics.")
        return

    with st.spinner("Running evaluation... This may take a little while."):
        summary_df, details = get_cached_evaluation(
            k=k,
            max_users=max_users,
            min_user_ratings=min_user_ratings,
            test_size=test_size,
            relevance_threshold=relevance_threshold,
            random_state=random_state
        )

    if summary_df.empty:
        st.error("Evaluation could not be completed. Try lowering the minimum ratings per user.")
        return

    st.success("Evaluation completed.")

    st.divider()

    st.subheader("Evaluation summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Evaluated users", details["evaluated_users"])

    with col2:
        st.metric("Train ratings", f"{details['train_ratings']:,}")

    with col3:
        st.metric("Test ratings", f"{details['test_ratings']:,}")

    with col4:
        st.metric("K", details["k"])

    display_df = summary_df.copy()

    precision_column = f"precision@{k}"
    recall_column = f"recall@{k}"

    display_df[precision_column] = display_df[precision_column].round(4)
    display_df[recall_column] = display_df[recall_column].round(4)

    display_df["mae"] = display_df["mae"].apply(
        lambda value: round(value, 4) if value is not None else None
    )

    st.dataframe(
        display_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("How to interpret the results")

    st.write(
        f"""
        **Precision@{k}** shows what part of the top {k} recommendations were actually relevant.

        **Recall@{k}** shows what part of the user's relevant test movies were successfully recommended.

        **MAE** shows the average rating prediction error.
        MAE is mainly meaningful for algorithms that predict explicit ratings,
        such as User-Based CF and Item-Based CF.
        """
    )

    st.warning(
        """
        The results may vary depending on the selected users and train/test split.
        This evaluation is intended for educational comparison between algorithms,
        not as a production-grade benchmark.
        """
    )


if __name__ == "__main__":
    main()