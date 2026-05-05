# ingestion/generate_activity.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.simulation_utils import ACTIVITY_TYPES, ACTIVITY_PROBS

np.random.seed(42)


def generate_activity(users_df, num_days=21):
    activities = []


    for _, user in users_df.iterrows():
        user_id = user["user_id"]

        start_date = datetime.now() - timedelta(days=num_days)

        for day in range(num_days):
            current_date = start_date + timedelta(days=day)

            # Simulate 3 activity slots per day
            time_slots = [8, 14, 19]

            for hour in time_slots:
                activity_type = np.random.choice(
                    ACTIVITY_TYPES,
                    p=ACTIVITY_PROBS
                )

                if activity_type == "sedentary":
                    steps = np.random.randint(0, 500)
                elif activity_type == "walking":
                    steps = np.random.randint(1000, 4000)
                else:  # running
                    steps = np.random.randint(3000, 8000)

                activities.append({
                    "user_id": user_id,
                    "timestamp": current_date.replace(hour=hour, minute=0),
                    "activity_type": activity_type,
                    "steps": steps
                })

    activity_df = pd.DataFrame(activities)
    activity_df = activity_df.sort_values(by=["user_id", "timestamp"])

    return activity_df


def save_activity(activity_df):
    activity_df.to_csv("data/activity.csv", index=False)
    print("✅ activity.csv created")


if __name__ == "__main__":
    users_df = pd.read_csv("data/users.csv")

    activity_df = generate_activity(users_df, num_days=21)
    save_activity(activity_df)