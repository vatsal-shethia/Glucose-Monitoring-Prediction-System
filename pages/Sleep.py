# pages/Sleep.py
# Sleep analysis page: duration/quality trends, sleep-glucose correlation, duration buckets.
# Note: trendline="ols" in Chart 2 requires the `statsmodels` package (listed in requirements.txt).

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pages.utils.db import run_query
from pages.components.sidebar import render_sidebar

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Sleep | Lingo", layout="wide")

# ---------------------- SIDEBAR ----------------------
selected_user, selected_window = render_sidebar()

st.title("😴 Sleep")
st.divider()

# ---------------------- ANCHOR DATE ----------------------
# Use the latest glucose timestamp as a consistent anchor across all pages.
latest_row = run_query("SELECT MAX(timestamp) as latest FROM glucose")
LATEST_DATE = latest_row.iloc[0]["latest"]

# ---------------------- TIME WINDOW FILTER (sleep uses DATE column) ----------------------
# The sleep table stores a `date` column (DATE type), not a `timestamp`,
# so we build a separate date-based filter rather than reusing the glucose time_filter.
if selected_window == "Day":
    sleep_time_filter = "s.date = DATE(%s)"
elif selected_window == "Week":
    sleep_time_filter = "s.date >= DATE(%s) - INTERVAL '7 days'"
else:  # Month
    sleep_time_filter = "s.date >= DATE(%s) - INTERVAL '30 days'"

# Unqualified version for queries that only touch the sleep table
if selected_window == "Day":
    sleep_time_filter_plain = "date = DATE(%s)"
elif selected_window == "Week":
    sleep_time_filter_plain = "date >= DATE(%s) - INTERVAL '7 days'"
else:
    sleep_time_filter_plain = "date >= DATE(%s) - INTERVAL '30 days'"

# ---------------------- USER FILTER ----------------------
if selected_user == "All Users":
    user_filter_plain     = ""              # for single-table queries
    user_filter_qualified = ""              # for JOIN queries (s.user_id)
    base_params           = [LATEST_DATE]
else:
    user_filter_plain     = "AND user_id = %s"
    user_filter_qualified = "AND s.user_id = %s"
    base_params           = [LATEST_DATE, selected_user]

# ─────────────────────────────────────────────────────────────────
# Shared raw sleep query — reused for Chart 1 and download
# ─────────────────────────────────────────────────────────────────
q_sleep_raw = f"""
SELECT user_id, date, sleep_duration, sleep_quality
FROM sleep s
WHERE {sleep_time_filter_plain}
{user_filter_plain}
ORDER BY date
"""
df_sleep_raw = run_query(q_sleep_raw, base_params)
df_sleep_raw["date"] = pd.to_datetime(df_sleep_raw["date"])

# ══════════════════════════════════════════════════════════════════
# CHART 1 — SLEEP DURATION AND QUALITY OVER TIME (dual y-axis)
# ══════════════════════════════════════════════════════════════════
st.subheader("🛏️ Sleep Duration and Quality Over Time")

if df_sleep_raw.empty:
    st.warning("No sleep data available for the selected filters.")
else:
    # Aggregate per date if multiple users are selected
    df_chart1 = (
        df_sleep_raw
        .groupby("date", as_index=False)
        .agg(sleep_duration=("sleep_duration", "mean"),
             sleep_quality=("sleep_quality", "mean"))
    )

    fig_sleep = make_subplots(specs=[[{"secondary_y": True}]])

    # Left axis — sleep duration (hours)
    fig_sleep.add_trace(
        go.Scatter(
            x=df_chart1["date"],
            y=df_chart1["sleep_duration"].round(1),
            name="Duration (hrs)",
            mode="lines+markers",
            line=dict(color="#4C9BE8", width=2),
            marker=dict(size=5),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Duration: %{y:.1f} hrs<extra></extra>",
        ),
        secondary_y=False,
    )

    # Right axis — sleep quality (1–5)
    fig_sleep.add_trace(
        go.Scatter(
            x=df_chart1["date"],
            y=df_chart1["sleep_quality"].round(2),
            name="Quality (1–5)",
            mode="lines+markers",
            line=dict(color="#9B59B6", width=2, dash="dot"),
            marker=dict(size=5),
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Quality: %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig_sleep.update_layout(
        title="Sleep Duration and Quality Over Time",
        title_font_size=16,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_sleep.update_yaxes(title_text="Sleep Duration (hrs)", secondary_y=False)
    fig_sleep.update_yaxes(
        title_text="Sleep Quality (1–5)",
        secondary_y=True,
        range=[0, 6],           # fix scale so 1–5 range is readable
    )
    fig_sleep.update_xaxes(title_text="Date")

    st.plotly_chart(fig_sleep, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# CHART 2 — SLEEP QUALITY VS NEXT-DAY GLUCOSE
# ══════════════════════════════════════════════════════════════════
st.subheader("📊 Sleep Quality vs Next-Day Average Glucose")

q_quality_glucose = f"""
SELECT
    s.sleep_quality,
    AVG(g.glucose_level) AS avg_next_day_glucose
FROM sleep s
JOIN glucose g
  ON  s.user_id       = g.user_id
  AND DATE(g.timestamp) = s.date + INTERVAL '1 day'
WHERE {sleep_time_filter}
{user_filter_qualified}
GROUP BY s.sleep_quality
ORDER BY s.sleep_quality
"""
df_quality_glucose = run_query(q_quality_glucose, base_params)

if df_quality_glucose.empty:
    st.warning("No sleep-glucose pairing available for the selected filters.")
else:
    df_quality_glucose["avg_next_day_glucose"] = (
        df_quality_glucose["avg_next_day_glucose"].round(1)
    )
    fig_scatter = px.scatter(
        df_quality_glucose,
        x="sleep_quality",
        y="avg_next_day_glucose",
        trendline="ols",
        labels={
            "sleep_quality":        "Sleep Quality (1–5)",
            "avg_next_day_glucose": "Avg Next-Day Glucose (mg/dL)",
        },
        title="Sleep Quality vs Next-Day Average Glucose",
        color_discrete_sequence=["#9B59B6"],
    )
    fig_scatter.update_traces(
        marker=dict(size=12),
        selector=dict(mode="markers"),
    )
    fig_scatter.update_layout(
        title_font_size=16,
        xaxis=dict(tickmode="linear", tick0=1, dtick=1, range=[0, 6]),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# CHART 3 — SLEEP DURATION GROUPS VS GLUCOSE
# ══════════════════════════════════════════════════════════════════
st.subheader("🔢 Glucose by Sleep Duration Group")

q_sleep_groups = f"""
SELECT
    CASE
        WHEN s.sleep_duration <  6 THEN '<6h'
        WHEN s.sleep_duration <  7 THEN '6-7h'
        WHEN s.sleep_duration <  8 THEN '7-8h'
        ELSE '8h+'
    END                      AS sleep_group,
    AVG(g.glucose_level)     AS avg_glucose
FROM sleep s
JOIN glucose g
  ON  s.user_id       = g.user_id
  AND DATE(g.timestamp) = s.date
WHERE {sleep_time_filter}
{user_filter_qualified}
GROUP BY sleep_group
"""
df_sleep_groups = run_query(q_sleep_groups, base_params)

if df_sleep_groups.empty:
    st.warning("No sleep-group data available for the selected filters.")
else:
    # Enforce logical ordering of the buckets (alphabetical would be wrong)
    group_order = ["<6h", "6-7h", "7-8h", "8h+"]
    df_sleep_groups["sleep_group"] = pd.Categorical(
        df_sleep_groups["sleep_group"],
        categories=group_order,
        ordered=True,
    )
    df_sleep_groups = df_sleep_groups.sort_values("sleep_group")
    df_sleep_groups["avg_glucose"] = df_sleep_groups["avg_glucose"].round(1)

    fig_groups = px.bar(
        df_sleep_groups,
        x="sleep_group",
        y="avg_glucose",
        labels={
            "sleep_group": "Sleep Duration Group",
            "avg_glucose": "Avg Glucose (mg/dL)",
        },
        title="Glucose by Sleep Duration Group",
        color="sleep_group",
        color_discrete_map={
            "<6h":  "#E74C3C",   # red   — insufficient sleep
            "6-7h": "#E67E22",   # orange
            "7-8h": "#2ECC71",   # green — optimal range
            "8h+":  "#4C9BE8",   # blue
        },
        text="avg_glucose",
    )
    fig_groups.update_traces(textposition="outside")
    fig_groups.update_layout(
        title_font_size=16,
        showlegend=False,
        bargap=0.4,
        yaxis=dict(range=[0, df_sleep_groups["avg_glucose"].max() * 1.2]),
    )
    # Mark the 140 spike threshold for context
    fig_groups.add_hline(
        y=140,
        line_dash="dash",
        line_color="red",
        opacity=0.4,
        annotation_text="Spike threshold (140)",
        annotation_position="top right",
    )
    st.plotly_chart(fig_groups, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# DOWNLOAD BUTTON
# ══════════════════════════════════════════════════════════════════
st.download_button(
    label="⬇️ Download Sleep Data as CSV",
    data=df_sleep_raw.to_csv(index=False),
    file_name="lingo_sleep_export.csv",
    mime="text/csv",
)
