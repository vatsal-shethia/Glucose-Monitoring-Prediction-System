# ml/predict.py

import joblib
import pandas as pd


# ---------------- LOAD MODELS ----------------
model = joblib.load("ml/model.pkl")

xgb_model = None
try:
    xgb_model = joblib.load("ml/xgb_model.pkl")
except FileNotFoundError:
    print("⚠️  Warning: ml/xgb_model.pkl not found — XGBoost predictions skipped.")


# ---------------- SAMPLE INPUT ----------------
# (You can later replace this with user input / Streamlit form)

sample_input = {
    "carbs_last_meal": 80,
    "steps_last_1hr": 200,
    "sleep_duration": 6.5,
    "sleep_quality": 3,        # middle of the 1–5 scale
    "prev_glucose": 130,
    "hr_avg_30min": 75.0,      # resting average heart rate (BPM)
    "hour": 14,

    "is_breakfast": 0,
    "is_lunch": 1,
    "is_dinner": 0,
    "is_snack": 0,

    "is_weekend": 0
}


# ---------------- PREDICTION ----------------
df = pd.DataFrame([sample_input])

threshold = 0.6


def _predict_and_print(mdl, label):
    prob = mdl.predict_proba(df)[0][1]
    prediction = 1 if prob > threshold else 0
    print(f"\n  [{label}]")
    if prediction == 1:
        print(f"  ⚠️  Warning: {prob*100:.1f}% chance of spike.")
        print("  💡 Suggestion: Take a 15-minute walk or avoid high-carb intake.")
    else:
        print(f"  ✅ Low Risk: {prob*100:.1f}% chance of spike.")
        print("  👍 Glucose levels likely stable.")


# ---------------- OUTPUT ----------------
print("\n🔍 PREDICTION RESULT")
print("=" * 50)

_predict_and_print(model, "Logistic Regression")

if xgb_model is not None:
    _predict_and_print(xgb_model, "XGBoost")

print("=" * 50)