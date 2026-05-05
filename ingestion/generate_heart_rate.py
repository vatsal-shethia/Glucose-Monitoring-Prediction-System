# ingestion/generate_heart_rate.py

import pandas as pd
import numpy as np

from utils.simulation_utils import (
    BASE_HEART_RATE,
    HR_WALKING_BOOST,
    HR_RUNNING_BOOST,
    HEART_RATE_NOISE_STD,
)

np.random.seed(42)


def generate_heart_rate(glucose_df, activity_df):
    heart_data = []

    for _, row in glucose_df.iterrows():
        user_id = row["user_id"]
        timestamp = row["timestamp"]
        glucose = row["glucose_level"]

        # Base heart rate
        base_hr = BASE_HEART_RATE

        # Activity effect
        activity_window = activity_df[
            (activity_df["user_id"] == user_id) &
            (activity_df["timestamp"] <= timestamp) &
            (activity_df["timestamp"] >= timestamp - pd.Timedelta(minutes=30))
        ]

        activity_effect = 0

        if not activity_window.empty:
            activity_type = activity_window.iloc[-1]["activity_type"]

            if activity_type == "walking":
                activity_effect = HR_WALKING_BOOST
            elif activity_type == "running":
                activity_effect = HR_RUNNING_BOOST

        # Glucose effect (small)
        glucose_effect = (glucose - 90) * 0.2

        noise = np.random.normal(0, HEART_RATE_NOISE_STD)

        heart_rate = base_hr + activity_effect + glucose_effect + noise
        heart_rate = max(50, min(140, heart_rate))

        heart_data.append({
            "user_id": user_id,
            "timestamp": timestamp,
            "heart_rate": round(heart_rate, 1)
        })

    hr_df = pd.DataFrame(heart_data)
    hr_df = hr_df.sort_values(by=["user_id", "timestamp"])

    return hr_df


def save_heart_rate(hr_df):
    hr_df.to_csv("data/heart_rate.csv", index=False)
    print("✅ heart_rate.csv created")


if __name__ == "__main__":
    glucose_df = pd.read_csv("data/glucose.csv", parse_dates=["timestamp"])
    activity_df = pd.read_csv("data/activity.csv", parse_dates=["timestamp"])

    hr_df = generate_heart_rate(glucose_df, activity_df)
    save_heart_rate(hr_df)