# ingestion/generate_sleep.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)


def generate_sleep(users_df, num_days=21):
    sleep_data = []

    for _, user in users_df.iterrows():
        user_id = user["user_id"]

        start_date = datetime.now() - timedelta(days=num_days)

        for day in range(num_days):
            current_date = start_date + timedelta(days=day)

            sleep_duration = np.round(np.random.uniform(5, 9), 1)  # hours
            sleep_quality = np.random.randint(1, 6)  # 1–5

            sleep_data.append({
                "user_id": user_id,
                "date": current_date.date(),
                "sleep_duration": sleep_duration,
                "sleep_quality": sleep_quality
            })

    sleep_df = pd.DataFrame(sleep_data)
    sleep_df = sleep_df.sort_values(by=["user_id", "date"])

    return sleep_df


def save_sleep(sleep_df):
    sleep_df.to_csv("data/sleep.csv", index=False)
    print("✅ sleep.csv created")


if __name__ == "__main__":
    users_df = pd.read_csv("data/users.csv")

    sleep_df = generate_sleep(users_df, num_days=21)
    save_sleep(sleep_df)