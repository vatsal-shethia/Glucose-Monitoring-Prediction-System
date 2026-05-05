# ml/predict.py

import joblib
import pandas as pd


# ---------------- LOAD MODEL ----------------
model = joblib.load("ml/model.pkl")


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

prob = model.predict_proba(df)[0][1]

threshold = 0.6
prediction = 1 if prob > threshold else 0


# ---------------- OUTPUT ----------------
print("\n🔍 PREDICTION RESULT")
print("="*40)

if prediction == 1:
    print(f"⚠️ Warning: {prob*100:.1f}% chance of spike.")
    print("💡 Suggestion: Take a 15-minute walk or avoid high-carb intake.")
else:
    print(f"✅ Low Risk: {prob*100:.1f}% chance of spike.")
    print("👍 Glucose levels likely stable.")