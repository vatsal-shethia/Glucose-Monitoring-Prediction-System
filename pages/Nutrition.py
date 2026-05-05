# pages/Nutrition.py
# Nutrition analysis page: carb intake, meal distribution, and meal-glucose overlay.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pages.utils.db import run_query
from pages.components.sidebar import render_sidebar

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Nutrition | Lingo", layout="wide")

# ---------------------- SIDEBAR ----------------------
selected_user, selected_window = render_sidebar()

st.title("🍽️ Nutrition")
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
# Shared meals query — raw rows reused for Chart 3 and download
# ─────────────────────────────────────────────────────────────────
q_meals_raw = f"""
SELECT user_id, timestamp, meal_type, carbs
FROM meals
WHERE {time_filter}
{user_filter}
ORDER BY timestamp
"""
df_meals_raw = run_query(q_meals_raw, base_params)
if not df_meals_raw.empty:
    df_meals_raw["timestamp"] = pd.to_datetime(df_meals_raw["timestamp"])

# ══════════════════════════════════════════════════════════════════
# CHART 1 — CARB INTAKE OVER TIME
# ══════════════════════════════════════════════════════════════════
st.subheader("🍞 Daily Carb Intake")

q_carbs_daily = f"""
SELECT
    DATE(timestamp) AS date,
    SUM(carbs)      AS total_carbs
FROM meals
WHERE {time_filter}
{user_filter}
GROUP BY DATE(timestamp)
ORDER BY date
"""
df_carbs_daily = run_query(q_carbs_daily, base_params)

if df_carbs_daily.empty:
    st.warning("No meal data available for the selected filters.")
else:
    fig_carbs = px.bar(
        df_carbs_daily,
        x="date",
        y="total_carbs",
        labels={"date": "Date", "total_carbs": "Total Carbs (g)"},
        title="Daily Carb Intake",
        color_discrete_sequence=["#6C63FF"],
    )
    fig_carbs.update_layout(
        title_font_size=16,
        xaxis_title="Date",
        yaxis_title="Carbs (g)",
        bargap=0.3,
    )
    st.plotly_chart(fig_carbs, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# CHART 2 — MEAL TYPE BREAKDOWN
# ══════════════════════════════════════════════════════════════════
st.subheader("🥗 Meal Type Distribution")

if df_meals_raw.empty:
    st.warning("No meal data available for the selected filters.")
else:
    df_meal_types = (
        df_meals_raw
        .groupby("meal_type", as_index=False)
        .agg(count=("carbs", "count"), avg_carbs=("carbs", "mean"))
    )
    df_meal_types["avg_carbs"] = df_meal_types["avg_carbs"].round(1)

    fig_pie = px.pie(
        df_meal_types,
        values="count",
        names="meal_type",
        title="Meal Type Distribution",
        hole=0.35,                          # donut style
        color="meal_type",
        color_discrete_map={
            "breakfast": "#4C9BE8",
            "lunch":     "#2ECC71",
            "dinner":    "#E67E22",
            "snack":     "#9B59B6",
        },
    )
    fig_pie.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Avg Carbs: %{customdata} g",
        customdata=df_meal_types["avg_carbs"],
    )
    fig_pie.update_layout(title_font_size=16, showlegend=True)
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# CHART 3 — MEAL OVERLAY ON GLUCOSE
# ══════════════════════════════════════════════════════════════════
st.subheader("📉 Glucose with Meal Timestamps")

q_glucose = f"""
SELECT timestamp, glucose_level
FROM glucose
WHERE {time_filter}
{user_filter}
ORDER BY timestamp
"""
df_glucose = run_query(q_glucose, base_params)
df_glucose["timestamp"] = pd.to_datetime(df_glucose["timestamp"])

MEAL_COLORS = {
    "breakfast": "#4C9BE8",   # blue
    "lunch":     "#2ECC71",   # green
    "dinner":    "#E67E22",   # orange
    "snack":     "#9B59B6",   # purple
}

if df_glucose.empty:
    st.warning("No glucose data available for the selected filters.")
else:
    fig_overlay = go.Figure()

    # Glucose line
    fig_overlay.add_trace(go.Scatter(
        x=df_glucose["timestamp"],
        y=df_glucose["glucose_level"],
        mode="lines",
        name="Glucose",
        line=dict(color="#4C9BE8", width=1.5),
    ))

    # Vertical dotted lines for each meal, coloured by meal_type
    if not df_meals_raw.empty:
        for meal_type, color in MEAL_COLORS.items():
            subset = df_meals_raw[df_meals_raw["meal_type"] == meal_type]
            # Add a single invisible scatter trace per meal_type for the legend
            fig_overlay.add_trace(go.Scatter(
                x=subset["timestamp"],
                y=[df_glucose["glucose_level"].max()] * len(subset),
                mode="markers",
                marker=dict(color=color, symbol="triangle-down", size=10),
                name=meal_type.capitalize(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Time: %{x}<br>"
                    "Carbs: %{customdata[1]} g"
                ),
                customdata=list(zip(
                    subset["meal_type"].str.capitalize(),
                    subset["carbs"],
                )),
            ))
            # Add a dotted vertical line for each meal event
            for ts in subset["timestamp"]:
                fig_overlay.add_vline(
                    x=ts,
                    line_dash="dot",
                    line_color=color,
                    line_width=1,
                    opacity=0.5,
                )

    # Spike reference line
    fig_overlay.add_hline(
        y=140,
        line_dash="dash",
        line_color="red",
        opacity=0.4,
        annotation_text="Spike threshold (140)",
        annotation_position="top right",
    )

    fig_overlay.update_layout(
        title="Glucose with Meal Timestamps",
        title_font_size=16,
        xaxis_title="Time",
        yaxis_title="Glucose (mg/dL)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_overlay, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════
# DOWNLOAD BUTTON
# ══════════════════════════════════════════════════════════════════
st.download_button(
    label="⬇️ Download Nutrition Data as CSV",
    data=df_meals_raw.to_csv(index=False),
    file_name="lingo_nutrition_export.csv",
    mime="text/csv",
)
