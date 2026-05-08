# Kafka decouples the data producers (CGM sensors) from inference.
# This consumer processes events as they arrive, maintaining state
# per user to reconstruct feature windows — matching the batch pipeline logic.

import json
import os
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
import psycopg2

try:
    from kafka import KafkaConsumer
    from kafka.errors import NoBrokersAvailable
except ImportError:
    raise ImportError("kafka-python is required. Install with: pip install kafka-python")

# Allow running from the project root as well as from within streaming/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from streaming.state_manager import UserStateManager, FEATURE_COLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KAFKA_BOOTSTRAP  = "localhost:9092"
TOPIC            = "health_events"
GROUP_ID         = "lingo_inference"
MODEL_PATH       = os.path.join(os.path.dirname(__file__), "..", "ml", "model.pkl")
DATABASE_URL     = os.environ.get(
    "DATABASE_URL", "postgresql://lingo:lingo@localhost:5432/lingo"
)
SPIKE_THRESHOLD  = 0.6

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS stream_predictions (
  id               SERIAL PRIMARY KEY,
  user_id          INT,
  timestamp        TIMESTAMPTZ,
  event_type       TEXT,
  pred_probability FLOAT,
  pred_label       INT
);
"""

INSERT_SQL = """
INSERT INTO stream_predictions
    (user_id, timestamp, event_type, pred_probability, pred_label)
VALUES (%s, %s, %s, %s, %s);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_db(conn_str: str) -> psycopg2.extensions.connection:
    """Connect to PostgreSQL and ensure the predictions table exists."""
    conn = psycopg2.connect(conn_str)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLE_SQL)
    print("[consumer] PostgreSQL connected — stream_predictions table ready.")
    return conn


def load_model(path: str):
    """Load the sklearn pipeline/model from disk."""
    model = joblib.load(path)
    print(f"[consumer] model loaded from {path}")
    return model


def build_dataframe(features: dict) -> pd.DataFrame:
    """Return a single-row DataFrame with columns in FEATURE_COLS order."""
    return pd.DataFrame([[features[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)


def run_inference(model, features: dict) -> tuple[float, int]:
    """Return (probability, label) for a single feature vector."""
    df = build_dataframe(features)
    prob = float(model.predict_proba(df)[0][1])
    label = int(prob >= SPIKE_THRESHOLD)
    return prob, label


def persist(conn, user_id: str, ts: datetime, event_type: str, prob: float, label: int) -> None:
    """Insert one prediction row into PostgreSQL."""
    with conn.cursor() as cur:
        cur.execute(INSERT_SQL, (int(user_id), ts, event_type, prob, label))


# ---------------------------------------------------------------------------
# Main consumer loop
# ---------------------------------------------------------------------------

def run_consumer() -> None:
    # --- Model ---
    try:
        model = load_model(MODEL_PATH)
    except FileNotFoundError:
        print(f"[consumer] ERROR: model not found at {MODEL_PATH}. Train it first.")
        return

    # --- Database ---
    try:
        conn = init_db(DATABASE_URL)
    except Exception as exc:
        print(f"[consumer] ERROR: could not connect to PostgreSQL — {exc}")
        return

    # --- Kafka ---
    print(f"[consumer] connecting to Kafka at {KAFKA_BOOTSTRAP} …")
    try:
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP,
            group_id=GROUP_ID,
            auto_offset_reset="earliest",
            value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
        )
    except NoBrokersAvailable:
        print(
            f"[consumer] ERROR: No Kafka brokers available at {KAFKA_BOOTSTRAP}. "
            "Start with: docker-compose up -d kafka"
        )
        conn.close()
        return
    except Exception as exc:
        print(f"[consumer] ERROR: Kafka connection failed — {exc}")
        conn.close()
        return

    state_manager = UserStateManager()
    processed = 0

    print(f"[consumer] listening on '{TOPIC}' (group={GROUP_ID}) — Ctrl+C to stop.\n")
    try:
        for message in consumer:
            try:
                event = message.value
                user_id   = str(event.get("user_id", "unknown"))
                event_type = event.get("event_type", "")

                # Update rolling state regardless of whether we can infer yet
                state_manager.update(event)

                # Only run inference on glucose events (target variable context)
                if event_type != "glucose":
                    continue

                # Parse the event timestamp
                raw_ts = event.get("timestamp", "")
                try:
                    current_ts = datetime.fromisoformat(raw_ts.strip())
                    # Ensure timezone-aware for TIMESTAMPTZ column
                    if current_ts.tzinfo is None:
                        current_ts = current_ts.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue  # skip malformed timestamps

                features = state_manager.get_features(user_id, current_ts)
                if features is None:
                    # Not enough history yet (no prior glucose reading)
                    continue

                prob, label = run_inference(model, features)
                persist(conn, user_id, current_ts, event_type, prob, label)

                processed += 1
                if label == 1:
                    print(f"[SPIKE ALERT] user_id={user_id} prob={prob:.2f}")

            except Exception as exc:
                # Log per-message errors without crashing the consumer loop
                print(f"[consumer] WARNING: failed to process message — {exc}")

    except KeyboardInterrupt:
        print(f"\n[consumer] shutting down — {processed} predictions written.")
    finally:
        consumer.close()
        conn.close()
        print("[consumer] connections closed.")


if __name__ == "__main__":
    run_consumer()
