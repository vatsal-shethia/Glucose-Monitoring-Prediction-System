# ingestion/generate_meals.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)


def generate_meals(users_df, num_days=21):
    meals = []

    meal_types = ["breakfast", "lunch", "dinner", "snack"]

    for _, user in users_df.iterrows():
        user_id = user["user_id"]

        start_date = datetime.now() - timedelta(days=num_days)

        for day in range(num_days):
            current_date = start_date + timedelta(days=day)

            # --- Breakfast ---
            meals.append({
                "user_id": user_id,
                "timestamp": current_date.replace(hour=np.random.randint(7, 10), minute=0),
                "meal_type": "breakfast",
                "carbs": np.random.randint(30, 60)
            })

            # --- Lunch ---
            meals.append({
                "user_id": user_id,
                "timestamp": current_date.replace(hour=np.random.randint(12, 15), minute=0),
                "meal_type": "lunch",
                "carbs": np.random.randint(50, 100)
            })

            # --- Dinner ---
            meals.append({
                "user_id": user_id,
                "timestamp": current_date.replace(hour=np.random.randint(19, 22), minute=0),
                "meal_type": "dinner",
                "carbs": np.random.randint(40, 90)
            })

            # --- Optional Snack ---
            if np.random.rand() > 0.5:
                meals.append({
                    "user_id": user_id,
                    "timestamp": current_date.replace(hour=np.random.randint(15, 18), minute=0),
                    "meal_type": "snack",
                    "carbs": np.random.randint(10, 40)
                })

    meals_df = pd.DataFrame(meals)
    meals_df = meals_df.sort_values(by=["user_id", "timestamp"])

    return meals_df


def save_meals(meals_df):
    meals_df.to_csv("data/meals.csv", index=False)
    print("✅ meals.csv created")


if __name__ == "__main__":
    users_df = pd.read_csv("data/users.csv")

    meals_df = generate_meals(users_df, num_days=21)
    save_meals(meals_df)