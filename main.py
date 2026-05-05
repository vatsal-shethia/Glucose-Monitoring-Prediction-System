# main.py

import argparse

from ingestion.generate_users import generate_users, save_users
from ingestion.generate_meals import generate_meals, save_meals
from ingestion.generate_activity import generate_activity, save_activity
from ingestion.generate_sleep import generate_sleep, save_sleep
from ingestion.generate_glucose import generate_glucose, save_glucose
from ingestion.generate_heart_rate import generate_heart_rate, save_heart_rate

from ml.feature_engineering import create_features, load_data, save_features
from ml.train_model import load_data as load_train, prepare_data, train_model, evaluate_model, save_model


def run_pipeline(skip_ingestion=False):

    if not skip_ingestion:
        print("[*] Generating data...")

        users = generate_users()
        save_users(users)

        meals = generate_meals(users)
        save_meals(meals)

        activity = generate_activity(users)
        save_activity(activity)

        sleep = generate_sleep(users)
        save_sleep(sleep)

        glucose = generate_glucose(users, meals, activity)
        save_glucose(glucose)

        heart = generate_heart_rate(glucose, activity)
        save_heart_rate(heart)

    print("[*] Feature engineering...")
    glucose, meals, activity, sleep, heart = load_data()
    features = create_features(glucose, meals, activity, sleep, heart)
    save_features(features)

    print("[*] Training model...")
    df = load_train()
    X, y = prepare_data(df)
    model, X_test, y_test, feature_names = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    save_model(model)

    print("[OK] Pipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingestion", action="store_true")

    args = parser.parse_args()

    run_pipeline(skip_ingestion=args.skip_ingestion)