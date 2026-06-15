import streamlit as st


def initialize_session_state():
    """
    Initializes Streamlit session state variables used for authentication.
    """
    if "is_logged_in" not in st.session_state:
        st.session_state.is_logged_in = False

    if "current_user" not in st.session_state:
        st.session_state.current_user = None


def login_user(user: dict):
    """
    Stores the logged-in user in Streamlit session state.
    """
    st.session_state.is_logged_in = True
    st.session_state.current_user = user


def logout_user():
    """
    Clears the current user from Streamlit session state.
    """
    st.session_state.is_logged_in = False
    st.session_state.current_user = None


def is_logged_in() -> bool:
    """
    Returns whether a user is currently logged in.
    """
    initialize_session_state()
    return st.session_state.is_logged_in


def get_current_user():
    """
    Returns the current logged-in user.
    """
    initialize_session_state()
    return st.session_state.current_user


def get_current_user_id():
    """
    Returns the current user's recommender_user_id.
    """
    user = get_current_user()

    if user is None:
        return None

    return user.get("recommender_user_id")


def get_current_username():
    """
    Returns the current username.
    """
    user = get_current_user()

    if user is None:
        return None

    return user.get("username")