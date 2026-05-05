# pages/Activity.py
# Activity analysis page: daily steps, activity breakdown, and activity-vs-glucose.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pages.utils.db import run_query
from pages.components.sidebar import render_sidebar

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Activity | Lingo", layout="wide")

# ---------------------- SIDEBAR ----------------------
selected_user, selected_window = render_sidebar()

st.title("🏃 Activity")
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

# ---------------------- USER FILTER ----------------------
if selected_user == "All Users":
    user_filter = ""
    base_params = [LATEST_DATE]
else:
    user_filter = "AND user_id = %s"
    base_params = [LATEST_DATE, selected_user]

# ─────────────────────────────────────────────────────────────────
# Shared raw activity query — reused for download
# ─────────────────────────────────────────────────────────────────
q_activity_raw = f"""
SELECT user_id, timestamp, activity_type, steps
FROM activity
WHERE {time_filter}
{user_filter}
ORDER BY timestamp
"""
df_activity_raw = run_query(q_activity_raw, base_params)
df_activity_raw["timestamp"] = pd.to_datetime(df_activity_raw["timestamp"])

# ══════════════════════════════════════════════════════════════════
# CHART 1 — DAILY STEPS OVER TIME
# ══════════════════════════════════════════════════════════════════
st.subheader("👟 Daily Steps")

q_steps_daily = f"""
SELECT
    DATE(timestamp) AS date,
    SUM(steps)      AS total_steps
FROM activity
WHERE {time_filter}
{user_filter}
GROUP BY DATE(timestamp)
ORDER BY date
"""
df_steps_daily = run_query(q_steps_daily, base_params)

STEP_GOAL = 6000

if df_steps_daily.empty:
    st.warning("No activity data available for the selected filters.")
else:
    fig_steps = go.Figure()

    fig_steps.add_trace(go.Bar(
        x=df_steps_daily["date"],
        y=df_steps_daily["total_steps"],
        name="Steps",
        marker_color=[
            "#2ECC71" if s >= STEP_GOAL else "#E74C3C"
            for s in df_steps_daily["total_steps"]
        ],
        hovertemplate="<b>%{x}</b><br>Steps: %{y:,}<extra></extra>",
    ))

    # Goal reference line
    fig_steps.add_hline(
        y=STEP_GOAL,
        line_dash="dot",
        line_color="red",
        line_width=2,
        annotation_text="Goal (6,000)",
        annotation_position="top right",
    )

    fig_steps.update_layout(
        title="Daily Steps",
        title_font_size=16,
        xaxis_title="Date",
        yaxis_title="Total Steps",
        bargap=0.3,
        showlegend=False,
    )
    st.plotly_chart(fig_steps, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# CHART 2 — ACTIVITY TYPE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════
st.subheader("📊 Activity Type Breakdown")

q_act_types = f"""
SELECT
    activity_type,
    COUNT(*) AS count
FROM activity
WHERE {time_filter}
{user_filter}
GROUP BY activity_type
ORDER BY count DESC
"""
df_act_types = run_query(q_act_types, base_params)

if df_act_types.empty:
    st.warning("No activity data available for the selected filters.")
else:
    fig_types = px.bar(
        df_act_types,
        x="activity_type",
        y="count",
        labels={"activity_type": "Activity Type", "count": "Count"},
        title="Activity Type Breakdown",
        color="activity_type",
        color_discrete_map={
            "walking":   "#4C9BE8",
            "running":   "#2ECC71",
            "sedentary": "#E74C3C",
        },
    )
    fig_types.update_layout(
        title_font_size=16,
        showlegend=False,
        bargap=0.4,
    )
    st.plotly_chart(fig_types, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# CHART 3 — ACTIVITY VS GLUCOSE (box plot)
# ══════════════════════════════════════════════════════════════════
st.subheader("📉 Glucose Level by Activity Type")

# Build qualified filters for the JOIN query to avoid ambiguous column refs
if selected_window == "Day":
    act_time_filter = "DATE(a.timestamp) = DATE(%s)"
elif selected_window == "Week":
    act_time_filter = "a.timestamp >= %s - INTERVAL '7 days'"
else:
    act_time_filter = "a.timestamp >= %s - INTERVAL '30 days'"

act_user_filter = "" if selected_user == "All Users" else "AND a.user_id = %s"

q_act_glucose = f"""
SELECT
    a.activity_type,
    g.glucose_level
FROM activity a
JOIN glucose g
  ON  a.user_id   = g.user_id
  AND ABS(EXTRACT(EPOCH FROM (g.timestamp - a.timestamp))) <= 1800
WHERE {act_time_filter}
{act_user_filter}
"""
df_act_glucose = run_query(q_act_glucose, base_params)

if df_act_glucose.empty:
    st.warning("No paired activity-glucose data available for the selected filters.")
else:
    fig_box = px.box(
        df_act_glucose,
        x="activity_type",
        y="glucose_level",
        labels={
            "activity_type": "Activity Type",
            "glucose_level": "Glucose (mg/dL)",
        },
        title="Glucose Level by Activity Type",
        color="activity_type",
        color_discrete_map={
            "walking":   "#4C9BE8",
            "running":   "#2ECC71",
            "sedentary": "#E74C3C",
        },
        points="outliers",
    )
    fig_box.add_hline(
        y=140,
        line_dash="dash",
        line_color="orange",
        opacity=0.5,
        annotation_text="Spike threshold (140)",
        annotation_position="top right",
    )
    fig_box.update_layout(
        title_font_size=16,
        showlegend=False,
    )
    st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# DOWNLOAD BUTTON
# ══════════════════════════════════════════════════════════════════
st.download_button(
    label="⬇️ Download Activity Data as CSV",
    data=df_activity_raw.to_csv(index=False),
    file_name="lingo_activity_export.csv",
    mime="text/csv",
)
