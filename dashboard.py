import streamlit as st
import pandas as pd

from pages.utils.db import get_connection, run_query
from pages.components.sidebar import render_sidebar

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Lingo Dashboard", layout="wide")

# ---------------------- GET LATEST DATE ----------------------
def get_latest_date():
    df = run_query("SELECT MAX(timestamp) as latest FROM glucose")
    return df.iloc[0]["latest"]

LATEST_DATE = get_latest_date()

# ---------------------- SIDEBAR ----------------------
selected_user, view = render_sidebar()

st.title("Lingo Health Dashboard")

if view == "Day":
    time_filter = "DATE(timestamp) = DATE(%s)"
elif view == "Week":
    time_filter = "timestamp >= %s - INTERVAL '7 days'"
else:
    time_filter = "timestamp >= %s - INTERVAL '30 days'"

# ---------------------- USER FILTER ----------------------
if selected_user == "All Users":
    user_filter = ""
    base_params = [LATEST_DATE]
else:
    user_filter = "AND user_id = %s"
    base_params = [LATEST_DATE, selected_user]

# ---------------------- GLUCOSE ----------------------
query_glucose = f"""
SELECT timestamp, AVG(glucose_level) as glucose_level
FROM glucose
WHERE {time_filter}
{user_filter}
GROUP BY timestamp
ORDER BY timestamp
"""

df_glucose = run_query(query_glucose, base_params)

# ---------------------- METRICS ----------------------
if not df_glucose.empty:
    current_glucose = df_glucose.iloc[-1]["glucose_level"]
    avg_glucose = df_glucose["glucose_level"].mean()
    peak_glucose = df_glucose["glucose_level"].max()
else:
    current_glucose, avg_glucose, peak_glucose = 0, 0, 0

# ---------------------- TIME IN RANGE ----------------------
if not df_glucose.empty:
    in_range = df_glucose[
        (df_glucose["glucose_level"] >= 70) &
        (df_glucose["glucose_level"] <= 140)
    ]
    tir = (len(in_range) / len(df_glucose)) * 100
else:
    tir = 0

# ---------------------- SMART INSIGHTS ----------------------
st.subheader("💡 Smart Insights")

insights = []

if current_glucose > 140:
    insights.append("⚠️ High glucose — consider activity")
elif current_glucose < 70:
    insights.append("⚠️ Low glucose — consider intake")
else:
    insights.append("✅ Glucose stable")

if avg_glucose > 120:
    insights.append("📈 Elevated average glucose")

if tir > 70:
    insights.append("🎯 Good glucose control")

for i in insights:
    st.info(i)

st.divider()

# ---------------------- METRIC CARDS ----------------------
c1, c2, c3 = st.columns(3)
c1.metric("Current Glucose", f"{current_glucose:.0f}")
c2.metric("Average Glucose", f"{avg_glucose:.0f}")
c3.metric("Peak Glucose", f"{peak_glucose:.0f}")

st.divider()

# ---------------------- BEHAVIORAL INSIGHTS ----------------------
st.subheader("🧠 Behavioral Insights")

behavior_insights = []

# ---- Meal ----
if selected_user == "All Users":
    q = """
    SELECT m.meal_type, AVG(g.glucose_level) as avg_glucose
    FROM meals m
    JOIN glucose g
    ON m.user_id = g.user_id
    AND g.timestamp BETWEEN m.timestamp AND m.timestamp + INTERVAL '2 hours'
    GROUP BY m.meal_type
    """
    meal_df = run_query(q)
else:
    q = """
    SELECT m.meal_type, AVG(g.glucose_level) as avg_glucose
    FROM meals m
    JOIN glucose g
    ON m.user_id = g.user_id
    AND g.timestamp BETWEEN m.timestamp AND m.timestamp + INTERVAL '2 hours'
    WHERE m.user_id = %s
    GROUP BY m.meal_type
    """
    meal_df = run_query(q, (selected_user,))

if not meal_df.empty:
    row = meal_df.sort_values("avg_glucose", ascending=False).iloc[0]
    behavior_insights.append(
        f"🍽️ {row['meal_type'].capitalize()} causes highest spikes (~{row['avg_glucose']:.0f})"
    )

# ---- Activity ----
if selected_user == "All Users":
    q = """
    SELECT 
        CASE 
            WHEN steps < 500 THEN 'low'
            WHEN steps < 3000 THEN 'moderate'
            ELSE 'high'
        END as activity_level,
        AVG(g.glucose_level) as avg_glucose
    FROM activity a
    JOIN glucose g
    ON a.user_id = g.user_id
    AND g.timestamp BETWEEN a.timestamp AND a.timestamp + INTERVAL '1 hour'
    GROUP BY activity_level
    """
    act_df = run_query(q)
else:
    q = """
    SELECT 
        CASE 
            WHEN steps < 500 THEN 'low'
            WHEN steps < 3000 THEN 'moderate'
            ELSE 'high'
        END as activity_level,
        AVG(g.glucose_level) as avg_glucose
    FROM activity a
    JOIN glucose g
    ON a.user_id = g.user_id
    AND g.timestamp BETWEEN a.timestamp AND a.timestamp + INTERVAL '1 hour'
    WHERE a.user_id = %s
    GROUP BY activity_level
    """
    act_df = run_query(q, (selected_user,))

if not act_df.empty:
    low = act_df[act_df["activity_level"] == "low"]
    if not low.empty:
        behavior_insights.append(
            f"🚶 Low activity → higher glucose (~{low.iloc[0]['avg_glucose']:.0f})"
        )

# ---- Sleep ----
if selected_user == "All Users":
    q = """
    SELECT 
        CASE 
            WHEN sleep_duration < 6 THEN 'low'
            WHEN sleep_duration < 8 THEN 'normal'
            ELSE 'high'
        END as sleep_group,
        AVG(g.glucose_level) as avg_glucose
    FROM sleep s
    JOIN glucose g
    ON s.user_id = g.user_id
    GROUP BY sleep_group
    """
    sleep_df2 = run_query(q)
else:
    q = """
    SELECT 
        CASE 
            WHEN sleep_duration < 6 THEN 'low'
            WHEN sleep_duration < 8 THEN 'normal'
            ELSE 'high'
        END as sleep_group,
        AVG(g.glucose_level) as avg_glucose
    FROM sleep s
    JOIN glucose g
    ON s.user_id = g.user_id
    WHERE s.user_id = %s
    GROUP BY sleep_group
    """
    sleep_df2 = run_query(q, (selected_user,))

if not sleep_df2.empty:
    row = sleep_df2.sort_values("avg_glucose", ascending=False).iloc[0]
    behavior_insights.append(
        f"😴 {row['sleep_group']} sleep linked to higher glucose (~{row['avg_glucose']:.0f})"
    )

# ---- Heart Rate ----
if selected_user == "All Users":
    hr_df = run_query("SELECT AVG(heart_rate) as avg_hr FROM heart_rate")
else:
    hr_df = run_query(
        "SELECT AVG(heart_rate) as avg_hr FROM heart_rate WHERE user_id = %s",
        (selected_user,)
    )

if not hr_df.empty:
    avg_hr = hr_df.iloc[0]["avg_hr"]
    if avg_hr and avg_hr > 85:
        behavior_insights.append("❤️ Elevated heart rate → possible stress/activity")
    else:
        behavior_insights.append("❤️ Heart rate normal")

for i in behavior_insights:
    st.info(i)

st.divider()

# ---------------------- CHART ----------------------
st.subheader("📈 Glucose Trend")
if not df_glucose.empty:
    st.line_chart(df_glucose.set_index("timestamp"))
else:
    st.warning("No data")

st.divider()

# ---------------------- STEPS ----------------------
q = f"""
SELECT SUM(steps) as val
FROM activity
WHERE {time_filter}
{user_filter}
"""
steps_df = run_query(q, base_params)

steps_val = 0
if not steps_df.empty and steps_df.iloc[0]["val"] is not None:
    steps_val = steps_df.iloc[0]["val"]

# ---------------------- SLEEP ----------------------
q = f"""
SELECT AVG(sleep_duration) as val
FROM sleep
WHERE date >= DATE(%s)
{user_filter}
"""
sleep_df = run_query(q, base_params)

sleep_val = 0
if not sleep_df.empty and sleep_df.iloc[0]["val"] is not None:
    sleep_val = sleep_df.iloc[0]["val"]

# ---------------------- FINAL CARDS ----------------------
c1, c2, c3 = st.columns(3)
c1.metric("Steps", int(steps_val))
c2.metric("Sleep", f"{sleep_val:.1f} hrs")
c3.metric("Time in Range", f"{tir:.1f}%")

st.divider()

# ---------------------- ACTIONS ----------------------
st.subheader("🚀 Recommended Actions")

actions = []

if current_glucose > 140:
    actions.append("Take a 10–15 min walk")

if avg_glucose > 120:
    actions.append("Reduce high-carb meals")

if steps_val < 3000:
    actions.append("Increase daily steps (6k–8k)")

if sleep_val < 6:
    actions.append("Improve sleep duration")

if tir < 70:
    actions.append("Stabilize glucose via routine")

if not actions:
    actions.append("Maintain current lifestyle")

for a in actions:
    st.success(a)