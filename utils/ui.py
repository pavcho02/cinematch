import html

import pandas as pd
import streamlit as st


def apply_global_styles():
    """
    Applies global CSS styling for the CineMatch Streamlit application.
    """
    st.markdown(
        """
        <style>
            .main {
                background-color: #0e1117;
            }

            .cinematch-hero {
                padding: 28px 32px;
                border-radius: 18px;
                background: linear-gradient(135deg, #1f2937 0%, #111827 45%, #7f1d1d 100%);
                border: 1px solid rgba(255, 255, 255, 0.10);
                margin-bottom: 24px;
            }

            .cinematch-hero h1 {
                color: #ffffff;
                font-size: 42px;
                margin-bottom: 8px;
            }

            .cinematch-hero p {
                color: #d1d5db;
                font-size: 18px;
                margin-bottom: 0;
            }

            .cinematch-card {
                padding: 18px 20px;
                border-radius: 16px;
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.10);
                margin-bottom: 14px;
            }

            .cinematch-card h3 {
                margin-top: 0;
                margin-bottom: 8px;
                color: #f9fafb;
            }

            .cinematch-card p {
                color: #d1d5db;
                margin-bottom: 0;
            }

            .cinematch-badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 13px;
                font-weight: 600;
                background-color: rgba(239, 68, 68, 0.20);
                color: #fecaca;
                border: 1px solid rgba(239, 68, 68, 0.35);
                margin-bottom: 10px;
            }

            .cinematch-muted {
                color: #9ca3af;
                font-size: 14px;
            }

            .cinematch-section-title {
                font-size: 24px;
                font-weight: 700;
                margin-top: 8px;
                margin-bottom: 4px;
            }

            .cinematch-section-description {
                color: #9ca3af;
                font-size: 15px;
                margin-bottom: 18px;
            }

            div[data-testid="stMetric"] {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                padding: 14px;
                border-radius: 14px;
            }

            div[data-testid="stDataFrame"] {
                border-radius: 14px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_page_header(title: str, subtitle: str, badge: str | None = None):
    """
    Renders a styled page header.
    """
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle)

    badge_html = ""

    if badge:
        safe_badge = html.escape(badge)
        badge_html = f'<div class="cinematch-badge">{safe_badge}</div>'

    st.markdown(
        f"""
        <div class="cinematch-hero">
            {badge_html}
            <h1>{safe_title}</h1>
            <p>{safe_subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_feature_card(icon: str, title: str, description: str):
    """
    Renders a small feature card.
    """
    safe_icon = html.escape(icon)
    safe_title = html.escape(title)
    safe_description = html.escape(description)

    st.markdown(
        f"""
        <div class="cinematch-card">
            <h3>{safe_icon} {safe_title}</h3>
            <p>{safe_description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_section_intro(title: str, description: str):
    """
    Renders a section title with a short description.
    """
    safe_title = html.escape(title)
    safe_description = html.escape(description)

    st.markdown(
        f"""
        <div class="cinematch-section-title">{safe_title}</div>
        <div class="cinematch-section-description">{safe_description}</div>
        """,
        unsafe_allow_html=True
    )


def render_empty_state(title: str, description: str):
    """
    Renders a reusable empty state message.
    """
    safe_title = html.escape(title)
    safe_description = html.escape(description)

    st.info(f"**{safe_title}**\n\n{safe_description}")


def format_year(value):
    """
    Safely formats a year value from a dataframe.
    """
    if value is None:
        return "Unknown"

    if pd.isna(value):
        return "Unknown"

    try:
        return str(int(value))
    except (ValueError, TypeError):
        return "Unknown"


def format_genres(genres_value):
    """
    Formats genres consistently across the whole application.

    MovieLens stores genres as:
        Action|Adventure|Sci-Fi

    The UI displays them as:
        Action, Adventure, Sci-Fi
    """
    if genres_value is None:
        return "No genres listed"

    if isinstance(genres_value, float) and pd.isna(genres_value):
        return "No genres listed"

    if isinstance(genres_value, list):
        cleaned_genres = [
            str(genre).strip()
            for genre in genres_value
            if str(genre).strip()
        ]

        if not cleaned_genres:
            return "No genres listed"

        return ", ".join(cleaned_genres)

    genres_text = str(genres_value).strip()

    if not genres_text or genres_text == "(no genres listed)":
        return "No genres listed"

    return genres_text.replace("|", ", ")


def render_movie_preview_card(movie):
    """
    Renders a simple movie preview card for dashboard sections.
    """
    title = html.escape(str(movie["clean_title"]))
    year = format_year(movie.get("year"))

    genres_value = movie.get(
        "genres_list",
        movie.get("genres", "No genres listed")
    )

    genres = html.escape(format_genres(genres_value))

    avg_rating = movie.get("avg_rating", None)
    rating_count = movie.get("rating_count", None)

    rating_html = ""

    if avg_rating is not None and rating_count is not None:
        rating_html = (
            f"<p>⭐ <strong>{float(avg_rating):.2f}</strong> / 5 "
            f"from {int(rating_count)} ratings</p>"
        )

    st.markdown(
        f"""
        <div class="cinematch-card">
            <h3>🎬 {title}</h3>
            <p class="cinematch-muted">{year}</p>
            <p><strong>Genres:</strong> {genres}</p>
            {rating_html}
        </div>
        """,
        unsafe_allow_html=True
    )