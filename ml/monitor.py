# ml/monitor.py

import pandas as pd
import joblib


# ---------------- LOAD ----------------
model = joblib.load("ml/model.pkl")
df    = pd.read_csv("data/features.csv")

X = df.drop(columns=["user_id", "timestamp", "spike"])
y = df["spike"]


# ---------------- HELPER ----------------
def run_report(mdl, X, y, label, threshold=0.6):
    """Print monitoring stats for a single model."""
    print(f"\n{'='*50}")
    print(f"=== {label} ===")
    print(f"{'='*50}")

    probs  = mdl.predict_proba(X)[:, 1]
    preds  = (probs > threshold).astype(int)

    actual_rate = y.mean()
    pred_rate   = preds.mean()

    # 1. Spike rates
    print(f"\nActual Spike Rate   : {actual_rate:.4f}")
    print(f"Predicted Spike Rate: {pred_rate:.4f}")
    if pred_rate > actual_rate * 2:
        print("⚠️  Model is over-predicting spikes")

    # 2. Confidence
    print("\nPrediction Confidence:")
    prob_series = pd.Series(probs, name="pred_prob")
    print(prob_series.describe())

    low_conf  = ((probs >= 0.4) & (probs <= 0.6)).sum()
    high_risk = (preds == 1).sum()

    # 3. Low-confidence & high-risk counts
    print(f"\nLow-confidence predictions (0.4–0.6): {low_conf} rows")
    print(f"High-risk predictions               : {high_risk} rows")


# =============================================
print("\n📊 MONITORING REPORT")

# --- Logistic Regression (existing logic unchanged) ---
run_report(model, X, y, "Logistic Regression")

# Feature drift printed once — same features feed both models
print(f"\n{'='*50}")
print("Feature Drift Check (mean values):")
for col in X.columns:
    print(f"  {col}: {X[col].mean():.2f}")

# --- XGBoost ---
try:
    xgb_model = joblib.load("ml/xgb_model.pkl")
    run_report(xgb_model, X, y, "XGBoost")
except FileNotFoundError:
    print("\n⚠️  ml/xgb_model.pkl not found — XGBoost monitoring skipped.")

print(f"\n{'='*50}")