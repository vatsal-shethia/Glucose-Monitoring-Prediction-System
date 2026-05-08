# pages/Live_Monitor.py
# Real-time streaming predictions dashboard — reads from stream_predictions table
# populated by the Kafka consumer (streaming/consumer.py).

import streamlit as st
import pandas as pd

from pages.utils.db import run_query

st.set_page_config(page_title="Live Monitor | Lingo", layout="wide")

# ─────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────
st.title("📡 Live Glucose Spike Monitor")
st.caption(
    "Real-time predictions from the Kafka consumer pipeline. "
    "Data is sourced from the `stream_predictions` table."
)

if st.button("🔄 Refresh"):
    st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — LATEST PREDICTIONS
# ══════════════════════════════════════════════════════════════════
st.subheader("📡 Latest Predictions")
st.caption("Most recent 50 inference results ordered by event time.")

try:
    latest_df = run_query(
        """
        SELECT id, user_id, timestamp, event_type, pred_probability, pred_label
        FROM   stream_predictions
        ORDER  BY timestamp DESC
        LIMIT  50
        """
    )

    if latest_df.empty:
        st.info("No streaming data yet. Start the Kafka consumer first.")
    else:
        st.dataframe(
            latest_df.style.format({"pred_probability": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
except Exception:
    st.info("No streaming data yet. Start the Kafka consumer first.")

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — SPIKE ALERTS
# ══════════════════════════════════════════════════════════════════
st.subheader("🔴 Spike Alerts")
st.caption("Rows where `pred_label = 1` within the last 100 events.")

try:
    spike_df = run_query(
        """
        SELECT id, user_id, timestamp, pred_probability, pred_label
        FROM (
            SELECT *
            FROM   stream_predictions
            ORDER  BY timestamp DESC
            LIMIT  100
        ) sub
        WHERE pred_label = 1
        ORDER BY timestamp DESC
        """
    )

    spike_count = len(spike_df)
    st.metric("🚨 Spike Events Detected", spike_count)

    if spike_df.empty:
        st.info("No spike alerts in the last 100 events.")
    else:
        st.dataframe(
            spike_df.style
                .format({"pred_probability": "{:.4f}"})
                .highlight_between(
                    subset=["pred_probability"],
                    left=0.6,
                    right=1.0,
                    color="#ffe0e0",
                ),
            use_container_width=True,
            hide_index=True,
        )
except Exception:
    st.info("No streaming data yet. Start the Kafka consumer first.")

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — SPIKE RATE BY USER
# ══════════════════════════════════════════════════════════════════
st.subheader("📊 Spike Rate by User")
st.caption("Total spike predictions (`pred_label = 1`) grouped by user.")

try:
    user_spikes_df = run_query(
        """
        SELECT user_id, COUNT(*) AS spike_count
        FROM   stream_predictions
        WHERE  pred_label = 1
        GROUP  BY user_id
        ORDER  BY spike_count DESC
        """
    )

    if user_spikes_df.empty:
        st.info("No streaming data yet. Start the Kafka consumer first.")
    else:
        user_spikes_df["user_id"] = user_spikes_df["user_id"].astype(str)
        st.bar_chart(
            user_spikes_df.set_index("user_id")["spike_count"],
            use_container_width=True,
        )
except Exception:
    st.info("No streaming data yet. Start the Kafka consumer first.")

st.divider()

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — LIVE RISK TREND
# ══════════════════════════════════════════════════════════════════
st.subheader("📈 Live Risk Trend")
st.caption("Predicted spike probability over time (last 100 events, chronological).")

try:
    trend_df = run_query(
        """
        SELECT timestamp, pred_probability
        FROM (
            SELECT timestamp, pred_probability
            FROM   stream_predictions
            ORDER  BY timestamp DESC
            LIMIT  100
        ) sub
        ORDER BY timestamp ASC
        """
    )

    if trend_df.empty:
        st.info("No streaming data yet. Start the Kafka consumer first.")
    else:
        trend_df = trend_df.set_index("timestamp")
        st.line_chart(
            trend_df["pred_probability"],
            use_container_width=True,
        )
except Exception:
    st.info("No streaming data yet. Start the Kafka consumer first.")
