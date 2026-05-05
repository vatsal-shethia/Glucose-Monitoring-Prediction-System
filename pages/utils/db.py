# pages/utils/db.py
# Shared database utilities for the Lingo dashboard.
# Import get_connection and run_query from here instead of defining them per-page.

import streamlit as st
import psycopg2
import pandas as pd


@st.cache_resource
def get_connection():
    """
    Returns a persistent, cached psycopg2 connection.
    Uses @st.cache_resource so a single connection object is reused
    across all reruns and users for the lifetime of the app process.
    Credentials are read from .streamlit/secrets.toml under [postgres].
    """
    creds = st.secrets["postgres"]
    return psycopg2.connect(
        host=creds["host"],
        database=creds["database"],
        user=creds["user"],
        password=creds["password"]
    )


@st.cache_data(ttl=300)
def run_query(query: str, params=None) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.
    Uses @st.cache_data(ttl=300) so identical query+params combinations
    are served from cache for up to 5 minutes before hitting the DB again.

    Args:
        query:  SQL query string (use %s placeholders for parameters).
        params: Optional tuple or list of parameters to pass to the query.

    Returns:
        pd.DataFrame with query results.
    """
    conn = get_connection()
    return pd.read_sql(query, conn, params=params)
