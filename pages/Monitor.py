# pages/Monitor.py
# Model monitoring page: model info, global metrics, confidence analysis, per-user risk counts.
# Reads from CSV files and model.pkl only — no database connection required.

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

st.set_page_config(page_title="Monitor | Lingo", layout="wide")

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "carbs_last_meal", "steps_last_1hr", "sleep_duration", "sleep_quality",
    "prev_glucose", "hr_avg_30min", "hour",
    "is_breakfast", "is_lunch", "is_dinner", "is_snack", "is_weekend",
]
THRESHOLD = 0.6

# ─────────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_features():
    return pd.read_csv("data/features.csv")

@st.cache_resource
def load_model():
    return joblib.load("ml/model.pkl")

@st.cache_resource
def load_xgb_model():
    return joblib.load("ml/xgb_model.pkl")

try:
    features_df = load_features()
    model       = load_model()
except FileNotFoundError as e:
    st.error(
        f"Required file not found: `{e.filename}`. "
        "Run `python main.py` to generate data and train the model first."
    )
    st.stop()

xgb_model = None
try:
    xgb_model = load_xgb_model()
except FileNotFoundError:
    pass  # handled gracefully when XGBoost is selected

# ─────────────────────────────────────────────────────────────────
# PAGE TITLE + MODEL SELECTOR
# ─────────────────────────────────────────────────────────────────
st.title("⚙️ Model Monitor")

model_choice = st.selectbox("Select Model", ["Logistic Regression", "XGBoost"])

if model_choice == "XGBoost":
    if xgb_model is None:
        st.warning("⚠️ `ml/xgb_model.pkl` not found. Run `python ml/train_model.py` first.")
        st.stop()
    active_model = xgb_model
else:
    active_model = model

st.divider()

# ─────────────────────────────────────────────────────────────────
# PRE-COMPUTE PREDICTIONS (reused by all sections)
# ─────────────────────────────────────────────────────────────────
X        = features_df[FEATURE_COLS]
y        = features_df["spike"]
probs    = active_model.predict_proba(X)[:, 1]
preds    = (probs >= THRESHOLD).astype(int)

predictions_df = features_df[["user_id", "timestamp", "prev_glucose", "carbs_last_meal", "spike"]].copy()
predictions_df["pred_probability"] = probs.round(4)
predictions_df["pred_spike"]       = preds

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — MODEL INFO
# ══════════════════════════════════════════════════════════════════
st.subheader("🗂️ Model Info")

if model_choice == "Logistic Regression":
    logreg = active_model.named_steps["logreg"]
    model_info = {
        "model_type":               type(logreg).__name__,
        "pipeline_steps":           list(active_model.named_steps.keys()),
        "num_features":             len(FEATURE_COLS),
        "feature_names":            FEATURE_COLS,
        "classification_threshold": THRESHOLD,
        "training_target":          "spike  (glucose_level > 140 mg/dL)",
        "class_weight":             str(logreg.class_weight),
        "max_iter":                 logreg.max_iter,
        "solver":                   logreg.solver,
    }
else:  # XGBoost
    xgb_step = active_model.named_steps["xgb"]
    model_info = {
        "model_type":               type(xgb_step).__name__,
        "pipeline_steps":           list(active_model.named_steps.keys()),
        "num_features":             len(FEATURE_COLS),
        "feature_names":            FEATURE_COLS,
        "classification_threshold": THRESHOLD,
        "training_target":          "spike  (glucose_level > 140 mg/dL)",
        "n_estimators":             xgb_step.n_estimators,
        "max_depth":                xgb_step.max_depth,
        "eval_metric":              xgb_step.eval_metric,
        "random_state":             xgb_step.random_state,
    }

st.json(model_info)
st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — FULL MODEL METRICS
# ══════════════════════════════════════════════════════════════════
st.subheader("📊 Full Dataset Metrics")
st.caption(f"Evaluated on all {len(features_df):,} rows in features.csv  |  Threshold: {THRESHOLD}")

total_records = len(features_df)
spike_rate    = y.mean() * 100
accuracy      = accuracy_score(y, preds) * 100
precision     = precision_score(y, preds, zero_division=0) * 100
recall_val    = recall_score(y, preds, zero_division=0) * 100
f1            = f1_score(y, preds, zero_division=0) * 100

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total Records",  f"{total_records:,}")
m2.metric("Actual Spike Rate",  f"{spike_rate:.2f}%")
m3.metric("Accuracy",       f"{accuracy:.2f}%")
m4.metric("Precision",      f"{precision:.2f}%")
m5.metric("Recall",         f"{recall_val:.2f}%")
m6.metric("F1 Score",       f"{f1:.2f}%")

# Prediction rate drift check
pred_rate = preds.mean() * 100
col_rate, col_warn = st.columns([1, 2])
col_rate.metric(
    "Predicted Spike Rate",
    f"{pred_rate:.2f}%",
    delta=f"{pred_rate - spike_rate:+.2f}% vs actual",
    delta_color="inverse",
)
if pred_rate > spike_rate * 2:
    col_warn.warning(
        "⚠️ Model is **over-predicting** spikes — predicted rate is more than 2× the actual rate. "
        "Consider retraining or adjusting the threshold."
    )
else:
    col_warn.success("✅ Prediction rate is within acceptable range of the actual spike rate.")

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — LOW-CONFIDENCE PREDICTIONS
# ══════════════════════════════════════════════════════════════════
st.subheader("🟡 Low-Confidence Predictions (0.4–0.6 probability)")

low_conf_df = (
    predictions_df[predictions_df["pred_probability"].between(0.4, 0.6, inclusive="neither")]
    .sort_values("pred_probability", ascending=False)
    .reset_index(drop=True)
)

col_count, col_pct = st.columns(2)
col_count.metric("Low-Confidence Row Count",  f"{len(low_conf_df):,}")
col_pct.metric(
    "As % of Total",
    f"{len(low_conf_df) / total_records * 100:.2f}%",
)

if low_conf_df.empty:
    st.info("No low-confidence predictions in the current dataset.")
else:
    st.dataframe(
        low_conf_df.style.background_gradient(
            subset=["pred_probability"],
            cmap="YlOrBr",
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — HIGH-RISK COUNT BY USER
# ══════════════════════════════════════════════════════════════════
st.subheader("🔴 High-Risk Predictions by User")
st.caption(f"Rows where predicted probability > {THRESHOLD}")

high_risk_df  = predictions_df[predictions_df["pred_probability"] > THRESHOLD]
high_risk_by_user = (
    high_risk_df
    .groupby("user_id", as_index=False)
    .agg(high_risk_count=("pred_spike", "sum"))
    .sort_values("high_risk_count", ascending=False)
)

if high_risk_by_user.empty:
    st.info("No high-risk predictions found.")
else:
    fig_risk = px.bar(
        high_risk_by_user,
        x="user_id",
        y="high_risk_count",
        labels={"user_id": "User ID", "high_risk_count": "High-Risk Predictions"},
        title="High-Risk Predictions by User",
        color="high_risk_count",
        color_continuous_scale="Reds",
        text="high_risk_count",
    )
    fig_risk.update_traces(textposition="outside")
    fig_risk.update_layout(
        title_font_size=16,
        showlegend=False,
        coloraxis_showscale=False,
        bargap=0.3,
        xaxis=dict(tickmode="linear"),
        yaxis=dict(range=[0, high_risk_by_user["high_risk_count"].max() * 1.2]),
    )
    st.plotly_chart(fig_risk, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# DOWNLOAD — FULL PREDICTIONS
# ══════════════════════════════════════════════════════════════════
st.download_button(
    label="⬇️ Download Full Predictions as CSV",
    data=predictions_df.to_csv(index=False),
    file_name="lingo_predictions_export.csv",
    mime="text/csv",
)
