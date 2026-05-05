# pages/ML_Insights.py
# ML insights page: feature importance, confusion matrix, what-if predictor, high-risk windows.
# This page reads from CSV files and model.pkl — no database connection required.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from pages.components.sidebar import render_sidebar

st.set_page_config(page_title="ML Insights | Lingo", layout="wide")

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# Must match the column order used in feature_engineering.py exactly.
# ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "carbs_last_meal", "steps_last_1hr", "sleep_duration", "sleep_quality",
    "prev_glucose", "hr_avg_30min", "hour",
    "is_breakfast", "is_lunch", "is_dinner", "is_snack", "is_weekend",
]
THRESHOLD = 0.6   # consistent with rest of codebase

# ─────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_features():
    return pd.read_csv("data/features.csv")

@st.cache_resource
def load_model():
    return joblib.load("ml/model.pkl")

try:
    features_df = load_features()
    model       = load_model()
except FileNotFoundError as e:
    st.error(
        f"Required file not found: `{e.filename}`. "
        "Run `python main.py` to generate data and train the model first."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — user filter only; window is irrelevant for CSV-based page
# ─────────────────────────────────────────────────────────────────
selected_user, _ = render_sidebar()

if selected_user != "All Users":
    features_df = features_df[features_df["user_id"] == selected_user].copy()

if features_df.empty:
    st.warning("No feature data available for the selected user.")
    st.stop()

st.title("🧠 ML Insights")
st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════
st.subheader("📊 Feature Importance (Logistic Regression Coefficients)")
st.caption("Positive coefficient → increases spike risk   |   Negative coefficient → reduces spike risk")

# Pipeline is: scaler → logreg  (see ml/train_model.py)
logreg      = model.named_steps["logreg"]
coefs       = logreg.coef_[0]

importance_df = pd.DataFrame({
    "feature":     FEATURE_COLS,
    "coefficient": coefs,
}).assign(abs_coef=lambda d: d["coefficient"].abs())
importance_df = importance_df.sort_values("abs_coef", ascending=True)

importance_df["direction"] = importance_df["coefficient"].apply(
    lambda c: "Increases Risk" if c > 0 else "Reduces Risk"
)

fig_importance = px.bar(
    importance_df,
    x="coefficient",
    y="feature",
    orientation="h",
    color="direction",
    color_discrete_map={
        "Increases Risk": "#E74C3C",
        "Reduces Risk":   "#2ECC71",
    },
    labels={"coefficient": "Coefficient", "feature": "Feature"},
    title="Feature Importance (Logistic Regression Coefficients)",
)
fig_importance.update_layout(
    title_font_size=16,
    legend_title_text="",
    xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="white"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_importance, use_container_width=True)
st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — CONFUSION MATRIX & METRICS
# ══════════════════════════════════════════════════════════════════
st.subheader("🎯 Model Evaluation")

X_all = features_df[FEATURE_COLS]
y_all = features_df["spike"]

# Use the same 0.6 threshold applied throughout the codebase
probs_all = model.predict_proba(X_all)[:, 1]
preds_all = (probs_all >= THRESHOLD).astype(int)

cm = confusion_matrix(y_all, preds_all)
tn, fp, fn, tp = cm.ravel()

# Heatmap — annotated 2×2
cm_text = np.array([
    [f"TN\n{tn:,}", f"FP\n{fp:,}"],
    [f"FN\n{fn:,}", f"TP\n{tp:,}"],
])

fig_cm = go.Figure(go.Heatmap(
    z=[[tn, fp], [fn, tp]],
    x=["Predicted: No Spike", "Predicted: Spike"],
    y=["Actual: No Spike",    "Actual: Spike"],
    text=cm_text,
    texttemplate="%{text}",
    textfont=dict(size=16),
    colorscale="Blues",
    showscale=False,
))
fig_cm.update_layout(
    title="Confusion Matrix",
    title_font_size=16,
    xaxis_title="Predicted Label",
    yaxis_title="Actual Label",
    width=500,
    height=400,
)

col_cm, col_metrics = st.columns([1, 1])

with col_cm:
    st.plotly_chart(fig_cm, use_container_width=True)

with col_metrics:
    st.markdown("#### Performance Metrics")
    st.caption(f"Threshold: {THRESHOLD}")

    precision = precision_score(y_all, preds_all, zero_division=0)
    recall    = recall_score(y_all, preds_all, zero_division=0)
    f1        = f1_score(y_all, preds_all, zero_division=0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{precision:.2%}")
    m2.metric("Recall",    f"{recall:.2%}")
    m3.metric("F1 Score",  f"{f1:.2%}")

    st.divider()
    st.markdown(f"""
    | | Count |
    |---|---|
    | True Negatives  | {tn:,} |
    | False Positives | {fp:,} |
    | False Negatives | {fn:,} |
    | True Positives  | {tp:,} |
    | **Total rows**  | **{len(y_all):,}** |
    """)

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — WHAT-IF SPIKE PREDICTOR
# ══════════════════════════════════════════════════════════════════
st.subheader("🔮 What-If Spike Predictor")
st.caption("Adjust the sliders to simulate a scenario and see the predicted spike risk in real time.")

col_inputs, col_output = st.columns([1.2, 1])

with col_inputs:
    carbs_last_meal = st.slider("Carbs from Last Meal (g)",   0,    150,  60)
    steps_last_1hr  = st.slider("Steps in Last Hour",         0,   5000, 500)
    prev_glucose    = st.slider("Previous Glucose (mg/dL)",  70,    180, 100)
    hr_avg_30min    = st.slider("Avg Heart Rate - 30 min",   50,    120,  75)
    sleep_duration  = st.slider("Sleep Duration (hrs)",       4.0, 10.0,  7.0, step=0.5)
    sleep_quality   = st.slider("Sleep Quality (1–5)",        1,      5,   3)
    hour            = st.slider("Hour of Day",                0,     23,  12)

    meal_choice = st.radio(
        "Current Meal",
        options=["None", "Breakfast", "Lunch", "Dinner", "Snack"],
        horizontal=True,
    )
    is_weekend = st.checkbox("Is Weekend?")

# Map meal radio → one-hot flags
meal_flags = {
    "is_breakfast": int(meal_choice == "Breakfast"),
    "is_lunch":     int(meal_choice == "Lunch"),
    "is_dinner":    int(meal_choice == "Dinner"),
    "is_snack":     int(meal_choice == "Snack"),
}

input_dict = {
    "carbs_last_meal": carbs_last_meal,
    "steps_last_1hr":  steps_last_1hr,
    "sleep_duration":  sleep_duration,
    "sleep_quality":   sleep_quality,
    "prev_glucose":    prev_glucose,
    "hr_avg_30min":    hr_avg_30min,
    "hour":            hour,
    **meal_flags,
    "is_weekend":      int(is_weekend),
}

input_df   = pd.DataFrame([input_dict])[FEATURE_COLS]
spike_prob = model.predict_proba(input_df)[0][1]
spike_pct  = spike_prob * 100

# Risk tier
if spike_prob < 0.4:
    risk_label = "🟢 Low Risk"
    bar_color  = "#2ECC71"
elif spike_prob < 0.6:
    risk_label = "🟡 Moderate Risk"
    bar_color  = "#F39C12"
else:
    risk_label = "🔴 High Risk"
    bar_color  = "#E74C3C"

with col_output:
    st.markdown("#### Prediction")
    st.metric(label="Spike Risk", value=f"{spike_pct:.1f}%", delta=risk_label)

    # Plotly gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=spike_pct,
        number={"suffix": "%", "font": {"size": 36}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": bar_color, "thickness": 0.3},
            "steps": [
                {"range": [0,  40], "color": "#d4efdf"},   # light green
                {"range": [40, 60], "color": "#fef9e7"},   # light yellow
                {"range": [60,100], "color": "#fadbd8"},   # light red
            ],
            "threshold": {
                "line":  {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": THRESHOLD * 100,
            },
        },
    ))
    fig_gauge.update_layout(
        height=280,
        margin=dict(t=20, b=0, l=20, r=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    if spike_prob >= THRESHOLD:
        st.warning("⚠️ Above the 60% action threshold — consider a short walk or avoiding high-carb intake.")
    else:
        st.success("✅ Below the action threshold — glucose levels likely stable.")

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — HIGH-RISK PREDICTION WINDOWS
# ══════════════════════════════════════════════════════════════════
st.subheader("⚠️ High-Risk Prediction Windows")
st.caption(f"Rows where predicted spike probability > {THRESHOLD:.0%}")

# Re-run on the full (user-filtered) feature set
X_display     = features_df[FEATURE_COLS]
spike_probs   = model.predict_proba(X_display)[:, 1]
features_df   = features_df.copy()
features_df["spike_probability"] = spike_probs.round(3)

high_risk_df = (
    features_df[features_df["spike_probability"] > THRESHOLD]
    [[
        "user_id", "timestamp", "prev_glucose",
        "carbs_last_meal", "spike_probability",
    ]]
    .sort_values("spike_probability", ascending=False)
    .reset_index(drop=True)
)

if high_risk_df.empty:
    st.info("No high-risk windows found for the selected user and threshold.")
else:
    st.markdown(f"**{len(high_risk_df):,} high-risk window(s) detected**")
    st.dataframe(
        high_risk_df.style.background_gradient(
            subset=["spike_probability"],
            cmap="Reds",
        ),
        use_container_width=True,
        hide_index=True,
    )
