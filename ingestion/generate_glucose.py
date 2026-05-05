# ingestion/generate_glucose.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.simulation_utils import (
    BASELINE_GLUCOSE,
    MEAL_COEFF,
    ACTIVITY_COEFF,
    GLUCOSE_NOISE_STD,
    GLUCOSE_MIN,
    GLUCOSE_MAX,
)

np.random.seed(42)


def generate_glucose(users_df, meals_df, activity_df, num_days=21):
    glucose_data = []

    for _, user in users_df.iterrows():
        user_id = user["user_id"]

        user_meals = meals_df[meals_df["user_id"] == user_id]
        user_activity = activity_df[activity_df["user_id"] == user_id]

        start_date = datetime.now() - timedelta(days=num_days)

        current_time = start_date

        while current_time < datetime.now():
            baseline = BASELINE_GLUCOSE

            # --- Meal effect (last 2 hours) ---
            recent_meals = user_meals[
                (user_meals["timestamp"] <= current_time) &
                (user_meals["timestamp"] >= current_time - timedelta(hours=2))
            ]

            meal_effect = recent_meals["carbs"].sum() * MEAL_COEFF if not recent_meals.empty else 0

            # --- Activity effect (last 1 hour) ---
            recent_activity = user_activity[
                (user_activity["timestamp"] <= current_time) &
                (user_activity["timestamp"] >= current_time - timedelta(hours=1))
            ]

            activity_effect = recent_activity["steps"].sum() * ACTIVITY_COEFF if not recent_activity.empty else 0

            # --- Noise ---
            noise = np.random.normal(0, GLUCOSE_NOISE_STD)

            glucose = baseline + meal_effect - activity_effect + noise
            glucose = max(GLUCOSE_MIN, min(GLUCOSE_MAX, glucose))  # clamp

            glucose_data.append({
                "user_id": user_id,
                "timestamp": current_time,
                "glucose_level": round(glucose, 1)
            })

            current_time += timedelta(minutes=10)

    glucose_df = pd.DataFrame(glucose_data)
    glucose_df = glucose_df.sort_values(by=["user_id", "timestamp"])

    return glucose_df


def save_glucose(glucose_df):
    glucose_df.to_csv("data/glucose.csv", index=False)
    print("✅ glucose.csv created")


if __name__ == "__main__":
    users_df = pd.read_csv("data/users.csv")
    meals_df = pd.read_csv("data/meals.csv", parse_dates=["timestamp"])
    activity_df = pd.read_csv("data/activity.csv", parse_dates=["timestamp"])

    glucose_df = generate_glucose(users_df, meals_df, activity_df)
    save_glucose(glucose_df)