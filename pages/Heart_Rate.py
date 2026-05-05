# pages/Heart_Rate.py
# Heart rate analysis page: HR trend, HR-glucose scatter, and HR by activity type.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pages.utils.db import run_query
from pages.components.sidebar import render_sidebar

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Heart Rate | Lingo", layout="wide")

# ---------------------- SIDEBAR ----------------------
selected_user, selected_window = render_sidebar()

st.title("❤️ Heart Rate")
st.divider()

# ---------------------- TIME WINDOW FILTER ----------------------
latest_row = run_query("SELECT MAX(timestamp) as latest FROM glucose")
LATEST_DATE = latest_row.iloc[0]["latest"]

if selected_window == "Day":
    time_filter = "DATE(timestamp) = DATE(%s)"
elif selected_window == "Week":
    time_filter = "timestamp >= %s - INTERVAL '7 days'"
else:  # Month
    time_filter = "timestamp >= %s - INTERVAL '30 days'"

# Qualified version for JOIN queries (h.timestamp avoids ambiguity)
if selected_window == "Day":
    hr_time_filter = "DATE(h.timestamp) = DATE(%s)"
elif selected_window == "Week":
    hr_time_filter = "h.timestamp >= %s - INTERVAL '7 days'"
else:
    hr_time_filter = "h.timestamp >= %s - INTERVAL '30 days'"

# ---------------------- USER FILTER ----------------------
if selected_user == "All Users":
    user_filter    = ""
    hr_user_filter = ""
    base_params    = [LATEST_DATE]
else:
    user_filter    = "AND user_id = %s"
    hr_user_filter = "AND h.user_id = %s"
    base_params    = [LATEST_DATE, selected_user]

# ─────────────────────────────────────────────────────────────────
# Shared raw heart rate query — reused for Chart 1 and download
# ─────────────────────────────────────────────────────────────────
q_hr_raw = f"""
SELECT user_id, timestamp, heart_rate
FROM heart_rate
WHERE {time_filter}
{user_filter}
ORDER BY timestamp
"""
df_hr_raw = run_query(q_hr_raw, base_params)
if not df_hr_raw.empty:
    df_hr_raw["timestamp"] = pd.to_datetime(df_hr_raw["timestamp"])

# ══════════════════════════════════════════════════════════════════
# CHART 1 — HEART RATE TREND
# ══════════════════════════════════════════════════════════════════
st.subheader("📈 Heart Rate Over Time")

if df_hr_raw.empty:
    st.warning("No heart rate data available for the selected filters.")
else:
    # Average per timestamp when multiple users are selected
    df_chart1 = (
        df_hr_raw
        .groupby("timestamp", as_index=False)
        .agg(heart_rate=("heart_rate", "mean"))
    )

    fig_hr = go.Figure()

    fig_hr.add_trace(go.Scatter(
        x=df_chart1["timestamp"],
        y=df_chart1["heart_rate"].round(1),
        mode="lines",
        name="Heart Rate",
        line=dict(color="#E74C3C", width=1.5),
        hovertemplate="<b>%{x}</b><br>HR: %{y:.1f} BPM<extra></extra>",
    ))

    # Elevated threshold
    fig_hr.add_hline(
        y=100,
        line_dash="dot",
        line_color="orange",
        line_width=1.5,
        annotation_text="Elevated (100)",
        annotation_position="top left",
    )

    # High threshold
    fig_hr.add_hline(
        y=120,
        line_dash="dot",
        line_color="red",
        line_width=1.5,
        annotation_text="High (120)",
        annotation_position="top left",
    )

    fig_hr.update_layout(
        title="Heart Rate Over Time",
        title_font_size=16,
        xaxis_title="Time",
        yaxis_title="Heart Rate (BPM)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_hr, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# CHART 2 — HEART RATE VS GLUCOSE SCATTER
# ══════════════════════════════════════════════════════════════════
st.subheader("🔬 Heart Rate vs Glucose Level")

# heart_rate and glucose share exact timestamps (generated from the same events),
# so an exact timestamp match join is correct and efficient here.
q_hr_glucose = f"""
SELECT
    h.heart_rate,
    g.glucose_level
FROM heart_rate h
JOIN glucose g
  ON  h.user_id   = g.user_id
  AND h.timestamp = g.timestamp
WHERE {hr_time_filter}
{hr_user_filter}
"""
df_hg = run_query(q_hr_glucose, base_params)

if df_hg.empty:
    st.warning("No heart rate–glucose data available for the selected filters.")
else:
    # Color points by whether glucose is in the spike range
    df_hg["glucose_category"] = df_hg["glucose_level"].apply(
        lambda x: "Spike (>140 mg/dL)" if x > 140 else "Normal (≤140 mg/dL)"
    )

    fig_scatter = px.scatter(
        df_hg,
        x="glucose_level",
        y="heart_rate",
        color="glucose_category",
        color_discrete_map={
            "Normal (≤140 mg/dL)":  "#4C9BE8",
            "Spike (>140 mg/dL)":   "#E74C3C",
        },
        labels={
            "glucose_level":    "Glucose (mg/dL)",
            "heart_rate":       "Heart Rate (BPM)",
            "glucose_category": "Glucose Status",
        },
        title="Heart Rate vs Glucose Level",
        opacity=0.6,
    )
    fig_scatter.add_vline(
        x=140,
        line_dash="dash",
        line_color="red",
        opacity=0.4,
        annotation_text="Spike threshold (140)",
        annotation_position="top right",
    )
    fig_scatter.update_layout(
        title_font_size=16,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# CHART 3 — AVERAGE HR BY ACTIVITY TYPE
# ══════════════════════════════════════════════════════════════════
st.subheader("🏃 Average Heart Rate by Activity Type")

q_hr_activity = f"""
SELECT
    a.activity_type,
    ROUND(AVG(h.heart_rate)::numeric, 1) AS avg_heart_rate
FROM heart_rate h
JOIN activity a
  ON  h.user_id = a.user_id
  AND ABS(EXTRACT(EPOCH FROM (h.timestamp - a.timestamp))) <= 1800
WHERE {hr_time_filter}
{hr_user_filter}
GROUP BY a.activity_type
ORDER BY avg_heart_rate DESC
"""
df_hr_activity = run_query(q_hr_activity, base_params)

if df_hr_activity.empty:
    st.warning("No heart rate–activity data available for the selected filters.")
else:
    fig_act = px.bar(
        df_hr_activity,
        x="activity_type",
        y="avg_heart_rate",
        labels={
            "activity_type":  "Activity Type",
            "avg_heart_rate": "Avg Heart Rate (BPM)",
        },
        title="Average Heart Rate by Activity Type",
        color="activity_type",
        color_discrete_map={
            "walking":   "#4C9BE8",
            "running":   "#2ECC71",
            "sedentary": "#E74C3C",
        },
        text="avg_heart_rate",
    )
    fig_act.update_traces(textposition="outside")

    # Reference lines for context
    fig_act.add_hline(
        y=100,
        line_dash="dot",
        line_color="orange",
        opacity=0.5,
        annotation_text="Elevated (100)",
        annotation_position="top right",
    )
    fig_act.add_hline(
        y=120,
        line_dash="dot",
        line_color="red",
        opacity=0.5,
        annotation_text="High (120)",
        annotation_position="top right",
    )

    fig_act.update_layout(
        title_font_size=16,
        showlegend=False,
        bargap=0.4,
        yaxis=dict(range=[0, df_hr_activity["avg_heart_rate"].max() * 1.25]),
    )
    st.plotly_chart(fig_act, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# DOWNLOAD BUTTON
# ══════════════════════════════════════════════════════════════════
st.download_button(
    label="⬇️ Download Heart Rate Data as CSV",
    data=df_hr_raw.to_csv(index=False),
    file_name="lingo_heart_rate_export.csv",
    mime="text/csv",
)
