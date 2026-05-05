# ml/monitor.py

import pandas as pd
import joblib


# ---------------- LOAD ----------------
model = joblib.load("ml/model.pkl")
df = pd.read_csv("data/features.csv")

X = df.drop(columns=["user_id", "timestamp", "spike"])
y = df["spike"]

# Predictions
df["pred_prob"] = model.predict_proba(X)[:, 1]
df["pred_spike"] = (df["pred_prob"] > 0.6).astype(int)


print("\n📊 MONITORING REPORT")
print("="*50)

# ---------------- 1. PREDICTION RATE ----------------
actual_rate = y.mean()
pred_rate = df["pred_spike"].mean()

print(f"\nActual Spike Rate: {actual_rate:.4f}")
print(f"Predicted Spike Rate: {pred_rate:.4f}")

if pred_rate > actual_rate * 2:
    print("⚠️ Model is over-predicting spikes")

# ---------------- 2. FEATURE DRIFT ----------------
print("\nFeature Drift Check (mean values):")

for col in X.columns:
    print(f"{col}: {X[col].mean():.2f}")

# ---------------- 3. CONFIDENCE ----------------
print("\nPrediction Confidence:")

print(df["pred_prob"].describe())

low_conf = df[df["pred_prob"].between(0.4, 0.6)]

print(f"\nLow-confidence predictions: {len(low_conf)} rows")

# ---------------- 4. HIGH RISK USERS ----------------
high_risk = df[df["pred_spike"] == 1]

print(f"\nHigh-risk predictions: {len(high_risk)} rows")