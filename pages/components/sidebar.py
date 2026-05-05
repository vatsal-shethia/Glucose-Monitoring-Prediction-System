# pages/components/sidebar.py
# Reusable sidebar component for the Lingo dashboard.
# Call render_sidebar() at the top of any page to get the current
# user selection and time window without duplicating UI code.

from __future__ import annotations   # enables str | int syntax on Python 3.7+

import streamlit as st
from pages.utils.db import run_query


def render_sidebar() -> tuple[str | int, str]:
    """
    Renders the Lingo sidebar and returns the current user/window selections.

    Sidebar contents:
        - App title "Lingo CGM"
        - Selectbox: "All Users" or a specific user_id integer
        - Radio: time window — "Day", "Week", or "Month"

    Returns:
        selected_user  (str | int): "All Users" or the chosen user_id integer.
        selected_window (str):      "Day", "Week", or "Month".
    """
    st.sidebar.title("🧬 Lingo CGM")

    # ── User selector ──────────────────────────────────────────────────────────
    users_df = run_query("SELECT DISTINCT user_id FROM users ORDER BY user_id")
    user_options = ["All Users"] + users_df["user_id"].tolist()

    selected_user = st.sidebar.selectbox(
        label="Select User",
        options=user_options,
        key="sidebar_user_select",
    )

    # Return the integer id directly so callers can use it in SQL params
    # without extra casting.
    if selected_user != "All Users":
        selected_user = int(selected_user)

    # ── Time window ────────────────────────────────────────────────────────────
    selected_window = st.sidebar.radio(
        label="Time Window",
        options=["Day", "Week", "Month"],
        horizontal=False,
        key="sidebar_window_radio",
    )

    return selected_user, selected_window
