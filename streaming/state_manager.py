# Stateless inference on a single glucose event would be meaningless.
# This class reconstructs the same temporal context the batch pipeline
# builds via rolling windows and merge_asof — but incrementally.

from datetime import datetime, timedelta
from typing import Optional


# Canonical feature order — must match the batch pipeline's FEATURE_COLS exactly.
FEATURE_COLS = [
    "carbs_last_meal",
    "steps_last_1hr",
    "sleep_duration",
    "sleep_quality",
    "prev_glucose",
    "hr_avg_30min",
    "hour",
    "is_breakfast",
    "is_lunch",
    "is_dinner",
    "is_snack",
    "is_weekend",
]


def _empty_user_state() -> dict:
    """Return a fresh per-user state bucket."""
    return {
        "glucose_history":  [],   # list of (ts: datetime, glucose_level: float)
        "meal_history":     [],   # list of (ts: datetime, carbs: float, meal_type: str)
        "activity_history": [],   # list of (ts: datetime, steps: float)
        "hr_history":       [],   # list of (ts: datetime, heart_rate: float)
        "sleep": {
            "sleep_duration": 8.0,   # sensible defaults until first sleep event
            "sleep_quality":  3.0,
        },
    }


class UserStateManager:
    """Maintain per-user rolling state for incremental real-time feature engineering."""

    def __init__(self):
        # Keyed by user_id (str); each value is a dict produced by _empty_user_state().
        self.state: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, event: dict) -> None:
        """Route an event to the appropriate private updater."""
        user_id = str(event.get("user_id", "unknown"))
        if user_id not in self.state:
            self.state[user_id] = _empty_user_state()

        etype = event.get("event_type", "")

        if etype == "glucose":
            ts = self._parse_ts(event.get("timestamp", ""))
            if ts:
                self._update_glucose(user_id, ts, float(event.get("glucose_level", 0.0)))

        elif etype == "meals":
            ts = self._parse_ts(event.get("timestamp", ""))
            if ts:
                self._update_meal(
                    user_id, ts,
                    float(event.get("carbs", 0.0)),
                    str(event.get("meal_type", "")),
                )

        elif etype == "activity":
            ts = self._parse_ts(event.get("timestamp", ""))
            if ts:
                self._update_activity(user_id, ts, float(event.get("steps", 0.0)))

        elif etype == "heart_rate":
            ts = self._parse_ts(event.get("timestamp", ""))
            if ts:
                self._update_heart_rate(user_id, ts, float(event.get("heart_rate", 0.0)))

        elif etype == "sleep":
            self._update_sleep(
                user_id,
                float(event.get("sleep_duration", 8.0)),
                float(event.get("sleep_quality", 3.0)),
            )

    def get_features(self, user_id: str, current_ts: datetime) -> Optional[dict]:
        """
        Compute the feature vector for *user_id* at *current_ts*.

        Returns None if there is insufficient state to make a meaningful
        prediction (i.e. no previous glucose reading exists yet).

        Feature keys are returned in the exact order of FEATURE_COLS.
        """
        user_id = str(user_id)
        if user_id not in self.state:
            return None

        s = self.state[user_id]

        # prev_glucose — must have at least one reading
        if not s["glucose_history"]:
            return None
        prev_glucose = s["glucose_history"][-1][1]

        # Rolling aggregates
        steps_last_1hr = self._rolling_sum(s["activity_history"], current_ts, 60)
        hr_avg_30min   = self._rolling_mean(s["hr_history"],      current_ts, 30)

        # Most recent meal features
        carbs_last_meal = 0.0
        is_breakfast = is_lunch = is_dinner = is_snack = 0
        if s["meal_history"]:
            _, carbs_last_meal, meal_type = s["meal_history"][-1]
            meal_type = meal_type.lower().strip()
            is_breakfast = int(meal_type == "breakfast")
            is_lunch     = int(meal_type == "lunch")
            is_dinner    = int(meal_type == "dinner")
            is_snack     = int(meal_type == "snack")

        # Sleep (defaults already set in _empty_user_state)
        sleep_duration = s["sleep"]["sleep_duration"]
        sleep_quality  = s["sleep"]["sleep_quality"]

        # Time features
        hour       = current_ts.hour
        is_weekend = int(current_ts.weekday() >= 5)

        # Build ordered dict matching FEATURE_COLS exactly
        return {
            "carbs_last_meal": carbs_last_meal,
            "steps_last_1hr":  steps_last_1hr,
            "sleep_duration":  sleep_duration,
            "sleep_quality":   sleep_quality,
            "prev_glucose":    prev_glucose,
            "hr_avg_30min":    hr_avg_30min,
            "hour":            hour,
            "is_breakfast":    is_breakfast,
            "is_lunch":        is_lunch,
            "is_dinner":       is_dinner,
            "is_snack":        is_snack,
            "is_weekend":      is_weekend,
        }

    # ------------------------------------------------------------------
    # Private updaters
    # ------------------------------------------------------------------

    def _update_glucose(self, user_id: str, ts: datetime, glucose_level: float) -> None:
        self.state[user_id]["glucose_history"].append((ts, glucose_level))

    def _update_meal(
        self, user_id: str, ts: datetime, carbs: float, meal_type: str
    ) -> None:
        self.state[user_id]["meal_history"].append((ts, carbs, meal_type))

    def _update_activity(self, user_id: str, ts: datetime, steps: float) -> None:
        self.state[user_id]["activity_history"].append((ts, steps))

    def _update_heart_rate(self, user_id: str, ts: datetime, heart_rate: float) -> None:
        self.state[user_id]["hr_history"].append((ts, heart_rate))

    def _update_sleep(
        self, user_id: str, sleep_duration: float, sleep_quality: float
    ) -> None:
        """Overwrite latest sleep values — only the most recent night matters."""
        self.state[user_id]["sleep"]["sleep_duration"] = sleep_duration
        self.state[user_id]["sleep"]["sleep_quality"]  = sleep_quality

    # ------------------------------------------------------------------
    # Rolling window helpers
    # ------------------------------------------------------------------

    def _rolling_sum(
        self, history: list, current_ts: datetime, window_minutes: int
    ) -> float:
        """Sum the second element of tuples whose timestamp falls within [current_ts - window, current_ts]."""
        cutoff = current_ts - timedelta(minutes=window_minutes)
        return sum(val for ts, val in history if cutoff <= ts <= current_ts)

    def _rolling_mean(
        self, history: list, current_ts: datetime, window_minutes: int
    ) -> float:
        """Average the second element of tuples within the rolling window. Returns 0.0 if empty."""
        cutoff = current_ts - timedelta(minutes=window_minutes)
        values = [val for ts, val in history if cutoff <= ts <= current_ts]
        return sum(values) / len(values) if values else 0.0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ts(raw: str) -> Optional[datetime]:
        """Parse an ISO timestamp string into a datetime. Returns None on failure."""
        try:
            return datetime.fromisoformat(raw.strip())
        except (ValueError, AttributeError):
            return None
