# Single topic design mirrors how a real CGM wearable emits all sensor types
# as one chronologically ordered stream. Separate topics would lose time ordering.

import csv
import json
import time
import os
from datetime import datetime

try:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable
except ImportError:
    raise ImportError("kafka-python is required. Install with: pip install kafka-python")

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "health_events"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CSV_FILES = {
    "glucose":    "glucose.csv",
    "meals":      "meals.csv",
    "activity":   "activity.csv",
    "heart_rate": "heart_rate.csv",
    "sleep":      "sleep.csv",
}

# sleep.csv uses "date" instead of "timestamp"
TIMESTAMP_FIELD = {
    "glucose":    "timestamp",
    "meals":      "timestamp",
    "activity":   "timestamp",
    "heart_rate": "timestamp",
    "sleep":      "date",
}


def load_events(event_type: str, filename: str) -> list[dict]:
    """Read a CSV and return a list of event dicts with event_type injected."""
    path = os.path.join(DATA_DIR, filename)
    events = []
    ts_field = TIMESTAMP_FIELD[event_type]

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            event = {"event_type": event_type}
            event.update(row)

            # Normalise to ISO string under the key "timestamp"
            raw_ts = row.get(ts_field, "")
            try:
                if ts_field == "date":
                    # date-only string → parse as date, convert to ISO datetime
                    dt = datetime.strptime(raw_ts.strip(), "%Y-%m-%d")
                else:
                    dt = datetime.fromisoformat(raw_ts.strip())
                event["timestamp"] = dt.isoformat()
            except ValueError:
                # Unparseable row: keep raw value so we don't silently drop data
                event["timestamp"] = raw_ts

            events.append(event)

    return events


def build_master_stream() -> list[dict]:
    """Merge all event lists, sorted chronologically."""
    master = []
    for event_type, filename in CSV_FILES.items():
        master.extend(load_events(event_type, filename))

    master.sort(key=lambda e: e["timestamp"])
    return master


def run_producer():
    print(f"[producer] connecting to Kafka at {KAFKA_BOOTSTRAP} …")
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
    except NoBrokersAvailable:
        print(
            "[producer] ERROR: No Kafka brokers available at "
            f"{KAFKA_BOOTSTRAP}. Is the broker running? "
            "Start it with: docker-compose up -d kafka"
        )
        return
    except Exception as exc:
        print(f"[producer] ERROR: Could not connect to Kafka — {exc}")
        return

    print("[producer] loading and merging CSV data …")
    events = build_master_stream()
    total = len(events)
    print(f"[producer] {total} events loaded across all data types. Publishing to '{TOPIC}' …")

    for i, event in enumerate(events, start=1):
        producer.send(TOPIC, value=event)
        if i % 200 == 0:
            print(f"[producer] sent {i} events…")
        time.sleep(0.05)

    producer.flush()
    print(f"[producer] done — {total} events published to '{TOPIC}'.")


if __name__ == "__main__":
    run_producer()
