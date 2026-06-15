import streamlit as st

from services.auth_service import authenticate_user, create_user
from utils.session import (
    get_current_username,
    initialize_session_state,
    is_logged_in,
    login_user,
    logout_user,
)


def render_auth_sidebar(key_prefix: str = "auth"):
    """
    Renders login/register/logout UI in the sidebar.

    key_prefix is used to avoid duplicate Streamlit widget keys
    across different pages.
    """
    st.sidebar.title("🎬 CineMatch")

    initialize_session_state()

    if is_logged_in():
        st.sidebar.success(f"Logged in as {get_current_username()}")

        if st.sidebar.button("Logout", key=f"{key_prefix}_logout"):
            logout_user()
            st.rerun()

        return

    auth_mode = st.sidebar.radio(
        "Account",
        ["Login", "Register"],
        key=f"{key_prefix}_auth_mode"
    )

    if auth_mode == "Login":
        st.sidebar.subheader("Login")

        with st.sidebar.form(f"{key_prefix}_login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            submitted = st.form_submit_button("Login")

            if submitted:
                result = authenticate_user(username, password)

                if result["success"]:
                    login_user(result["user"])
                    st.sidebar.success(result["message"])
                    st.rerun()
                else:
                    st.sidebar.error(result["message"])

    else:
        st.sidebar.subheader("Register")

        with st.sidebar.form(f"{key_prefix}_register_form"):
            username = st.text_input("Choose username")
            password = st.text_input("Choose password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")

            submitted = st.form_submit_button("Create account")

            if submitted:
                if password != confirm_password:
                    st.sidebar.error("Passwords do not match.")
                else:
                    result = create_user(username, password)

                    if result["success"]:
                        login_user(result["user"])
                        st.sidebar.success("Account created successfully.")
                        st.rerun()
                    else:
                        st.sidebar.error(result["message"])