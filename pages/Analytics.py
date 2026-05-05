# pages/Analytics.py
# Deep-dive analytics page: glucose heatmap, spike timeline, and user comparison.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pages.utils.db import run_query
from pages.components.sidebar import render_sidebar

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Analytics | Lingo", layout="wide")

# ---------------------- SIDEBAR ----------------------
selected_user, selected_window = render_sidebar()

st.title("📊 Analytics")
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

# ══════════════════════════════════════════════════════════════════
# CHART 1 — GLUCOSE HEATMAP
# ══════════════════════════════════════════════════════════════════
st.subheader("🌡️ Average Glucose by Hour and Day of Week")

q_heat = f"""
SELECT
    EXTRACT(hour FROM timestamp)::int    AS hour,
    EXTRACT(dow  FROM timestamp)::int    AS day_of_week,
    AVG(glucose_level)                   AS avg_glucose
FROM glucose
WHERE {time_filter}
{user_filter}
GROUP BY hour, day_of_week
ORDER BY day_of_week, hour
"""
df_heat = run_query(q_heat, base_params)

if df_heat.empty:
    st.warning("No glucose data available for the selected filters.")
else:
    # Pivot so rows = day of week, columns = hour of day
    pivot = df_heat.pivot_table(
        index="day_of_week",
        columns="hour",
        values="avg_glucose"
    )

    # Label axes with readable names (PostgreSQL DOW: 0 = Sunday)
    day_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    pivot.index = [day_labels[int(d)] for d in pivot.index]

    fig_heat = px.imshow(
        pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg Glucose (mg/dL)"),
        color_continuous_scale="RdYlGn_r",   # red = high, green = low
        aspect="auto",
        title="Average Glucose by Hour and Day of Week",
    )
    fig_heat.update_layout(
        title_font_size=16,
        coloraxis_colorbar=dict(title="mg/dL"),
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# CHART 2 — SPIKE TIMELINE
# ══════════════════════════════════════════════════════════════════
st.subheader("📈 Glucose Trend with Spike Highlights")

q_trend = f"""
SELECT timestamp, glucose_level
FROM glucose
WHERE {time_filter}
{user_filter}
ORDER BY timestamp
"""
df_trend = run_query(q_trend, base_params)

if df_trend.empty:
    st.warning("No glucose readings available for the selected filters.")
else:
    df_trend["timestamp"] = pd.to_datetime(df_trend["timestamp"])
    spikes = df_trend[df_trend["glucose_level"] > 140]

    fig_trend = go.Figure()

    # Base glucose line
    fig_trend.add_trace(go.Scatter(
        x=df_trend["timestamp"],
        y=df_trend["glucose_level"],
        mode="lines",
        name="Glucose",
        line=dict(color="#4C9BE8", width=1.5),
    ))

    # Red spike markers overlaid on the line
    fig_trend.add_trace(go.Scatter(
        x=spikes["timestamp"],
        y=spikes["glucose_level"],
        mode="markers",
        name="Spike (>140)",
        marker=dict(color="red", size=6, symbol="circle"),
    ))

    # Reference line at 140 mg/dL
    fig_trend.add_hline(
        y=140,
        line_dash="dash",
        line_color="orange",
        annotation_text="Spike threshold (140)",
        annotation_position="top left",
    )

    fig_trend.update_layout(
        title="Glucose Trend with Spike Highlights",
        title_font_size=16,
        xaxis_title="Time",
        yaxis_title="Glucose (mg/dL)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# CHART 3 — USER COMPARISON TABLE (all users only)
# ══════════════════════════════════════════════════════════════════
if selected_user == "All Users":
    st.subheader("👥 User Comparison Summary")

    q_compare = """
    SELECT
        g.user_id,
        ROUND(AVG(g.glucose_level)::numeric, 1)                                          AS avg_glucose,
        ROUND(
            (COUNT(CASE WHEN g.glucose_level > 140 THEN 1 END) * 100.0 / COUNT(*))::numeric,
        1)                                                                                AS spike_rate_pct,
        ROUND(AVG(a.steps)::numeric, 0)                                                   AS avg_steps,
        ROUND(AVG(s.sleep_duration)::numeric, 1)                                          AS avg_sleep_hrs
    FROM glucose g
    LEFT JOIN activity a ON g.user_id = a.user_id
    LEFT JOIN sleep   s ON g.user_id = s.user_id
    GROUP BY g.user_id
    ORDER BY g.user_id
    """
    df_compare = run_query(q_compare)

    if df_compare.empty:
        st.warning("No data available for user comparison.")
    else:
        df_compare.columns = [
            "User ID",
            "Avg Glucose (mg/dL)",
            "Spike Rate (%)",
            "Avg Steps",
            "Avg Sleep (hrs)",
        ]
        st.dataframe(df_compare, use_container_width=True, hide_index=True)
