# utils/time_utils.py

import pandas as pd


def get_latest_timestamp(df, column="timestamp"):
    """
    Returns the latest timestamp in a dataframe
    """
    return df[column].max()


def get_time_window(df, user_id, timestamp, window_hours):
    """
    Filter dataframe for a user within a time window
    """
    return df[
        (df["user_id"] == user_id) &
        (df["timestamp"] <= timestamp) &
        (df["timestamp"] >= timestamp - pd.Timedelta(hours=window_hours))
    ]


def get_latest_row(df, user_id, timestamp):
    """
    Get latest row before a timestamp
    """
    subset = df[
        (df["user_id"] == user_id) &
        (df["timestamp"] <= timestamp)
    ]

    if not subset.empty:
        return subset.iloc[-1]

    return None