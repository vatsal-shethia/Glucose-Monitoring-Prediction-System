import pandas as pd
import numpy as np

def load_data():
    """Loads the raw CSV files from the data directory."""
    glucose = pd.read_csv("data/glucose.csv", parse_dates=["timestamp"])
    meals = pd.read_csv("data/meals.csv", parse_dates=["timestamp"])
    activity = pd.read_csv("data/activity.csv", parse_dates=["timestamp"])
    sleep = pd.read_csv("data/sleep.csv")
    heart = pd.read_csv("data/heart_rate.csv", parse_dates=["timestamp"])
    
    # Convert sleep date to datetime objects for merging
    sleep['date'] = pd.to_datetime(sleep['date']).dt.date
    return glucose, meals, activity, sleep, heart

def save_features(df, filepath="data/features.csv"):
    """Saves the processed features to a CSV file."""
    df.to_csv(filepath, index=False)
    print(f"Features saved successfully to {filepath}")

def create_features(glucose, meals, activity, sleep, heart):
    # 1. CLEANING: Remove duplicates and sort for merge_asof
    glucose = glucose.drop_duplicates(subset=['user_id', 'timestamp']).sort_values("timestamp")
    meals = meals.drop_duplicates(subset=['user_id', 'timestamp']).sort_values("timestamp")
    activity = activity.sort_values("timestamp")
    heart = heart.sort_values("timestamp")

    # 2. MEAL FEATURES
    df = pd.merge_asof(
        glucose, 
        meals, 
        on="timestamp", 
        by="user_id", 
        direction="backward"
    )
    df['carbs_last_meal'] = df['carbs'].fillna(0)
    df['meal_type'] = df['meal_type'].fillna('unknown')

    # 3. ACTIVITY (Rolling 1h sum)
    activity = activity.set_index("timestamp")
    # Use '1h' (lowercase) to avoid deprecation warnings
    activity['steps_last_1hr'] = activity.groupby('user_id')['steps'].rolling('1h', closed='left').sum().reset_index(0, drop=True)
    activity = activity.reset_index()

    df = pd.merge_asof(
        df, 
        activity[['user_id', 'timestamp', 'steps_last_1hr']], 
        on="timestamp", 
        by="user_id", 
        direction="backward"
    ).fillna({'steps_last_1hr': 0})

    # 4. HEART RATE (Rolling 30min mean)
    heart = heart.set_index("timestamp")
    heart['hr_avg_30min'] = heart.groupby('user_id')['heart_rate'].rolling('30min', closed='left').mean().reset_index(0, drop=True)
    heart = heart.reset_index()

    df = pd.merge_asof(
        df, 
        heart[['user_id', 'timestamp', 'hr_avg_30min']], 
        on="timestamp", 
        by="user_id", 
        direction="backward"
    )

    # 5. SLEEP & TIME FEATURES
    df['date'] = df['timestamp'].dt.date
    df = df.merge(sleep, on=['user_id', 'date'], how='left')

    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = (df['timestamp'].dt.weekday >= 5).astype(int)
    
    for m_type in ['breakfast', 'lunch', 'dinner', 'snack']:
        df[f'is_{m_type}'] = (df['meal_type'] == m_type).astype(int)

    # 6. TARGET & PREV GLUCOSE
    df['prev_glucose'] = df.groupby('user_id')['glucose_level'].shift(1).fillna(df['glucose_level'])
    df['spike'] = (df['glucose_level'] > 140).astype(int)

    # 7. IMPUTATION
    df["sleep_duration"] = df.groupby("user_id")["sleep_duration"].transform(lambda x: x.fillna(x.mean() if not x.isnull().all() else 8.0))
    df["sleep_quality"] = df.groupby("user_id")["sleep_quality"].transform(lambda x: x.fillna(x.mean() if not x.isnull().all() else 3.0))
    df["hr_avg_30min"] = df["hr_avg_30min"].fillna(df["hr_avg_30min"].mean() if not df["hr_avg_30min"].isnull().all() else 70.0)

    final_cols = [
        "user_id", "timestamp", "carbs_last_meal", "steps_last_1hr", 
        "sleep_duration", "sleep_quality", "prev_glucose", "hr_avg_30min", 
        "hour", "is_breakfast", "is_lunch", "is_dinner", "is_snack", "is_weekend", "spike"
    ]
    return df[final_cols]