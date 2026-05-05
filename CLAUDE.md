# CLAUDE.md — Lingo Glucose Intelligence System

> Full project context for AI assistants, new contributors, and code review.
> Generated: 2026-04-21

---

## 🧬 Project Overview

**Lingo** is an end-to-end machine learning system inspired by Abbott's Lingo CGM platform. It simulates continuous glucose monitoring (CGM) health data for synthetic users, engineers time-aware features, trains a spike-prediction model, and presents behavioral insights through a Streamlit dashboard backed by a PostgreSQL database.

**Core value proposition:** Transform raw physiological data (glucose, meals, activity, sleep, heart rate) into personalized actionable health insights.

---

## 🗂️ Repository Structure

```
lingo-project/
├── CLAUDE.md                  ← This file
├── readme.md                  ← High-level project documentation
├── main.py                    ← CLI pipeline entry point
├── dashboard.py               ← Streamlit dashboard (DB-connected)
│
├── ingestion/                 ← Synthetic data generators
│   ├── generate_users.py
│   ├── generate_meals.py
│   ├── generate_activity.py
│   ├── generate_sleep.py
│   ├── generate_glucose.py
│   └── generate_heart_rate.py
│
├── ml/                        ← Machine learning pipeline
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py             ← Standalone spike predictor script
│   ├── monitor.py             ← Model monitoring report script
│   └── model.pkl              ← Serialized trained model (joblib)
│
├── utils/                     ← Shared utilities
│   ├── simulation_utils.py    ← Simulation constants
│   └── time_utils.py          ← Timestamp/window helper functions
│
├── data/                      ← Generated CSV data files
│   ├── users.csv
│   ├── meals.csv
│   ├── activity.csv
│   ├── sleep.csv
│   ├── glucose.csv
│   ├── heart_rate.csv
│   └── features.csv           ← Engineered feature matrix
│
└── .streamlit/
    └── secrets.toml           ← Database credentials (NOT committed)
```

---

## 🚀 How to Run

```bash
# 1. Run the full pipeline (data generation → feature engineering → model training)
python main.py

# 2. Run pipeline but skip data re-generation (reuse existing CSVs)
python main.py --skip-ingestion

# 3. Launch the Streamlit dashboard
streamlit run dashboard.py

# 4. Run standalone prediction demo
python ml/predict.py

# 5. Run model monitoring report
python ml/monitor.py
```

**Working directory:** All scripts must be run from the project root (`lingo-project/`).  
All file paths in the code are relative to root (e.g., `"data/glucose.csv"`, `"ml/model.pkl"`).

---

## ⚙️ System Architecture & Pipeline Flow

```
generate_users()
    ↓
generate_meals(users) → generate_activity(users) → generate_sleep(users)
    ↓                          ↓
generate_glucose(users, meals, activity)
    ↓
generate_heart_rate(glucose, activity)
    ↓
create_features(glucose, meals, activity, sleep, heart)  → data/features.csv
    ↓
train_model(X, y)  → ml/model.pkl
    ↓
dashboard.py  (reads from PostgreSQL)   |   monitor.py / predict.py (reads CSVs)
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical simulation |
| `scikit-learn` | Logistic Regression, Pipeline, StandardScaler, metrics |
| `imbalanced-learn` | SMOTE for class imbalance handling |
| `joblib` | Model serialization (`.pkl`) |
| `streamlit` | Dashboard UI |
| `psycopg2` | PostgreSQL connection from dashboard |

> ⚠️ **Note:** `ingestion/` and `ml/` scripts work entirely from CSV files. The dashboard (`dashboard.py`) is the **only** component that requires a live PostgreSQL database.

---

## 🗄️ Database Configuration

The dashboard connects to a local PostgreSQL instance via Streamlit secrets.

**`.streamlit/secrets.toml`:**
```toml
[postgres]
host     = "localhost"
database = "lingo_db"
user     = "postgres"
password = "<password>"
```

**Expected PostgreSQL tables** (must mirror the CSV schemas):

| Table | Columns |
|---|---|
| `users` | `user_id`, `age`, `gender` |
| `meals` | `user_id`, `timestamp`, `meal_type`, `carbs` |
| `activity` | `user_id`, `timestamp`, `activity_type`, `steps` |
| `sleep` | `user_id`, `date`, `sleep_duration`, `sleep_quality` |
| `glucose` | `user_id`, `timestamp`, `glucose_level` |
| `heart_rate` | `user_id`, `timestamp`, `heart_rate` |

> The dashboard uses `@st.cache_resource` for the DB connection; call `get_connection()` to retrieve it.

---

## 🔬 Data Simulation Details (`ingestion/`)

All generators use `np.random.seed(42)` for reproducibility.  
Default simulation window: **21 days**, **8 users**.

### `generate_users.py`
- Generates `num_users` (default: 8) users with random `age` (20–60) and `gender`.
- Output: `data/users.csv`

### `generate_meals.py`
- Per user per day: always generates **breakfast**, **lunch**, **dinner**; optionally generates a **snack** (50% probability).
- Meal timing: Breakfast 7–9h, Lunch 12–14h, Snack 15–17h, Dinner 19–21h.
- `carbs` ranges: Breakfast 30–60g, Lunch 50–100g, Dinner 40–90g, Snack 10–40g.
- Output: `data/meals.csv`

### `generate_activity.py`
- Per user per day: 3 time slots (08:00, 14:00, 19:00).
- Activity type sampled with probabilities: walking 50%, sedentary 30%, running 20%.
- Steps: sedentary 0–500, walking 1000–4000, running 3000–8000.
- Output: `data/activity.csv`

### `generate_sleep.py`
- Per user per day: `sleep_duration` uniform 5–9h, `sleep_quality` int 1–5.
- Output: `data/sleep.csv`

### `generate_glucose.py`
- Generates readings every **10 minutes** per user for 21 days.
- **Physiological model:**
  ```
  glucose = baseline(90) + meal_effect - activity_effect + noise(N(0,5))
  ```
  - `meal_effect` = sum of `carbs` from meals in last 2 hours × 0.5
  - `activity_effect` = sum of `steps` from activity in last 1 hour × 0.005
  - Clamped to [70, 180] mg/dL
- Output: `data/glucose.csv`

### `generate_heart_rate.py`
- Generated at same timestamps as glucose readings.
- **Model:**
  ```
  heart_rate = base(70) + activity_effect + glucose_effect + noise(N(0,3))
  ```
  - `activity_effect`: walking +10 BPM, running +25 BPM (based on last 30-min activity window)
  - `glucose_effect` = (glucose − 90) × 0.2
  - Clamped to [50, 140] BPM
- Output: `data/heart_rate.csv`

---

## 🧠 Feature Engineering (`ml/feature_engineering.py`)

**Input:** 5 raw DataFrames (glucose, meals, activity, sleep, heart)  
**Output:** `data/features.csv` — one row per glucose reading per user

### Feature List

| Feature | Description |
|---|---|
| `prev_glucose` | Previous glucose reading (1-step lag, per user) |
| `carbs_last_meal` | Carb count from the most recent meal (merge_asof backward) |
| `steps_last_1hr` | Rolling 1-hour sum of steps (per user) |
| `hr_avg_30min` | Rolling 30-minute average heart rate (per user) |
| `sleep_duration` | Duration from the same day's sleep record (left merge on date) |
| `sleep_quality` | Quality score (1–5) from same day's sleep record |
| `hour` | Hour of day (0–23) |
| `is_weekend` | Binary flag: 1 if Saturday or Sunday |
| `is_breakfast` | Binary: current row meal_type == 'breakfast' |
| `is_lunch` | Binary: current row meal_type == 'lunch' |
| `is_dinner` | Binary: current row meal_type == 'dinner' |
| `is_snack` | Binary: current row meal_type == 'snack' |
| `spike` | **Target**: 1 if `glucose_level > 140`, else 0 |

### Key Engineering Notes
- Uses `pd.merge_asof` with `direction="backward"` to join meals and activity without lookahead.
- Rolling windows use `closed='left'` to prevent data leakage.
- Missing `sleep_*` values are imputed with per-user mean (fallback global: 8.0 / 3.0).
- Missing `hr_avg_30min` imputed with global mean (fallback: 70.0).
- Final column order strictly defined in `final_cols` list.

---

## 🤖 Model (`ml/train_model.py`)

### Architecture
- **Algorithm:** Logistic Regression (interpretable; suited for healthcare)
- **Pipeline:** `StandardScaler` → `LogisticRegression(max_iter=2000, class_weight="balanced")`
- **Imbalance handling:** SMOTE applied to training set only (after train/test split)
- **Train/Test split:** 80/20, stratified on `spike`

### Thresholding
- Default prediction threshold: **0.6** (recall-prioritized)
- Reasoning: Missing a spike (false negative) is clinically worse than a false alarm.

### Performance (on synthetic data)
- ROC-AUC ≈ **0.97**
- Spike class prevalence: ~0.6% of all readings (extreme imbalance)

### Persistence
- Model saved via `joblib.dump()` to `ml/model.pkl`
- Loaded via `joblib.load("ml/model.pkl")` in `predict.py` and `monitor.py`

### ⚠️ Signature mismatch warning
`main.py` calls `train_model(X, y)` and expects **3 return values** `(model, X_test, y_test)`, but `train_model.py` actually returns **4 values** `(model, X_test, y_test, X.columns)`. The `evaluate_model` call in `main.py` does not use `feature_names`. This works in practice because Python allows ignoring extra tuple values when not unpacking all, but the call in `main.py` should be updated for clarity.

---

## 📡 Model Monitoring (`ml/monitor.py`)

Run as a standalone script from root. Generates a console monitoring report covering:

1. **Prediction Rate** — Actual vs predicted spike rates; warns if model over-predicts (>2× actual rate).
2. **Feature Drift** — Prints mean value of every feature column.
3. **Confidence Distribution** — Describes `pred_prob` column statistics.
4. **Low-Confidence Predictions** — Rows where `pred_prob` is between 0.4 and 0.6.
5. **High-Risk Predictions** — Count of rows predicted as spike.

---

## 🔮 Standalone Prediction (`ml/predict.py`)

Demonstrates single-row inference with a hardcoded sample input. Replace `sample_input` dict to test other scenarios.

**Sample input fields:**
```python
{
    "carbs_last_meal": 80,
    "steps_last_1hr": 200,
    "sleep_duration": 6.5,
    "prev_glucose": 130,
    "hour": 14,
    "is_breakfast": 0,
    "is_lunch": 1,
    "is_dinner": 0,
    "is_snack": 0,
    "is_weekend": 0
}
```

> ⚠️ `sleep_quality` and `hr_avg_30min` features are missing from the sample input. The model's feature set includes these; add them when updating the script.

---

## 📊 Dashboard (`dashboard.py`)

A Streamlit single-page dashboard. Runs `streamlit run dashboard.py` from project root.

### UI Structure

1. **Sidebar** — Logo, user selector (All Users + individual user IDs)
2. **View Toggle** — Day / Week / Month radio buttons (filters all queries)
3. **Smart Insights** — Auto-generated alerts (glucose level status, average alert, TIR)
4. **Metric Cards** — Current Glucose, Average Glucose, Peak Glucose
5. **Behavioral Insights** — 4 sub-sections:
   - 🍽️ Meal type → highest glucose spike correlation
   - 🚶 Activity level → glucose correlation
   - 😴 Sleep duration group → glucose correlation
   - ❤️ Average heart rate alert
6. **Glucose Trend Chart** — `st.line_chart` of glucose over time
7. **Steps / Sleep / TIR Cards** — Aggregated metrics for period
8. **Recommended Actions** — Auto-generated action list based on thresholds

### Dashboard Query Patterns
- All queries are parameterized with `[LATEST_DATE]` to anchor the time window.
- User filter added with `AND user_id = %s` when a specific user is selected.
- `@st.cache_resource` caches the DB connection across reruns.
- `run_query(query, params)` is the central query helper.

### Time-in-Range (TIR) Definition
```
TIR = % of glucose readings in [70, 140] mg/dL
```

### Action Thresholds
| Condition | Action |
|---|---|
| `current_glucose > 140` | "Take a 10–15 min walk" |
| `avg_glucose > 120` | "Reduce high-carb meals" |
| `steps_val < 3000` | "Increase daily steps (6k–8k)" |
| `sleep_val < 6` | "Improve sleep duration" |
| `tir < 70%` | "Stabilize glucose via routine" |

---

## 🛠️ Shared Utilities

### `utils/simulation_utils.py` — Constants

```python
BASELINE_GLUCOSE  = 90
MEAL_COEFF        = 0.5      # carbs → glucose multiplier
ACTIVITY_COEFF    = 0.005    # steps → glucose reduction multiplier
GLUCOSE_MIN       = 70
GLUCOSE_MAX       = 180
BASE_HEART_RATE   = 70
HR_WALKING_BOOST  = 10
HR_RUNNING_BOOST  = 25
GLUCOSE_NOISE_STD = 5
HEART_RATE_NOISE_STD = 3
```

> ⚠️ **These constants are defined here but are NOT currently imported by the ingestion scripts**. The actual simulation uses hardcoded values that match these. When refactoring, import from this module to avoid drift.

### `utils/time_utils.py` — Helper Functions

```python
get_latest_timestamp(df, column="timestamp")  → max timestamp value
get_time_window(df, user_id, timestamp, window_hours)  → filtered DataFrame
get_latest_row(df, user_id, timestamp)  → last row before timestamp, or None
```

> ⚠️ `time_utils.py` functions are also not currently imported anywhere in the codebase. They are available for use but unused.

---

## 🐛 Known Issues & Technical Debt

| Issue | Location | Severity |
|---|---|---|
| `train_model()` returns 4 values but `main.py` only unpacks 3 | `main.py:47`, `train_model.py:51` | Low (works but misleading) |
| `predict.py` sample input missing `sleep_quality` and `hr_avg_30min` | `ml/predict.py:14-27` | Medium (will fail if model was trained with those features) |
| `simulation_utils.py` constants not imported by ingestion scripts | `ingestion/`, `utils/simulation_utils.py` | Low (duplication risk) |
| `time_utils.py` helper functions are defined but never called | `utils/time_utils.py` | Low (dead code) |
| Dashboard and ML pipeline use separate data sources (DB vs CSV) | `dashboard.py` vs `ml/` | Medium (data could diverge) |
| `np.random.seed(42)` set at module-level in every ingestion file | All `ingestion/*.py` | Low (may cause subtle test issues) |
| No `requirements.txt` or `pyproject.toml` in repo | root | Medium (hard for others to install) |
| `.streamlit/secrets.toml` contains plaintext credentials | `.streamlit/secrets.toml` | High (add to `.gitignore`) |

---

## 📐 Data Schemas (CSV)

### `data/users.csv`
| Column | Type | Description |
|---|---|---|
| `user_id` | int | 1-indexed user identifier |
| `age` | int | 20–60 |
| `gender` | str | "Male" or "Female" |

### `data/meals.csv`
| Column | Type | Description |
|---|---|---|
| `user_id` | int | Foreign key to users |
| `timestamp` | datetime | Meal timestamp |
| `meal_type` | str | "breakfast", "lunch", "dinner", "snack" |
| `carbs` | int | Carbohydrates in grams |

### `data/activity.csv`
| Column | Type | Description |
|---|---|---|
| `user_id` | int | Foreign key to users |
| `timestamp` | datetime | Activity timestamp |
| `activity_type` | str | "walking", "running", "sedentary" |
| `steps` | int | Step count |

### `data/sleep.csv`
| Column | Type | Description |
|---|---|---|
| `user_id` | int | Foreign key to users |
| `date` | date | Date of sleep record |
| `sleep_duration` | float | Hours of sleep (5–9) |
| `sleep_quality` | int | Quality score 1–5 |

### `data/glucose.csv`
| Column | Type | Description |
|---|---|---|
| `user_id` | int | Foreign key to users |
| `timestamp` | datetime | Reading timestamp (every 10 min) |
| `glucose_level` | float | mg/dL, clamped to [70, 180] |

### `data/heart_rate.csv`
| Column | Type | Description |
|---|---|---|
| `user_id` | int | Foreign key to users |
| `timestamp` | datetime | Same as glucose timestamps |
| `heart_rate` | float | BPM, clamped to [50, 140] |

### `data/features.csv`
| Column | Type | Description |
|---|---|---|
| `user_id` | int | User identifier |
| `timestamp` | datetime | Reading timestamp |
| `carbs_last_meal` | float | Carbs from last meal (g) |
| `steps_last_1hr` | float | Steps in last 1 hour |
| `sleep_duration` | float | Hours slept previous night |
| `sleep_quality` | float | Sleep quality score |
| `prev_glucose` | float | Previous glucose reading |
| `hr_avg_30min` | float | 30-min rolling avg heart rate |
| `hour` | int | Hour of day |
| `is_breakfast` | int | Binary meal type flag |
| `is_lunch` | int | Binary meal type flag |
| `is_dinner` | int | Binary meal type flag |
| `is_snack` | int | Binary meal type flag |
| `is_weekend` | int | Binary day flag |
| `spike` | int | Target: 1 if glucose > 140 |

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| Logistic Regression over tree models | Interpretable coefficients; suitable for healthcare |
| SMOTE on training set only | Prevents data leakage; handles ~0.6% spike prevalence |
| Threshold = 0.6 (not 0.5) | Prioritizes recall — false negatives (missed spikes) are worse clinically |
| `merge_asof` for feature joins | Correct time-series join without lookahead leakage |
| `closed='left'` on rolling windows | Excludes current observation from rolling features |
| PostgreSQL for dashboard | Enables multi-user concurrent reads, SQL aggregations |
| CSV for ML pipeline | Self-contained, portable, no DB dependency for training |

---

## 📋 Limitations

- **Synthetic data only** — physiological model is simplified; real-world variability is much higher
- **No real-time streaming** — pipeline is batch; data must be regenerated manually
- **Low precision** — SMOTE improves recall but precision remains low due to severe class imbalance
- **Single model** — Logistic Regression; no ensemble or deep learning components
- **No authentication** in dashboard — user filter is UI-only, not access-controlled

---

## 🧩 Extension Points

| What to add | Where |
|---|---|
| New data signals (e.g., stress, HRV) | New `ingestion/generate_*.py` + new features in `feature_engineering.py` |
| Real-time data ingestion | Replace CSV writes with DB inserts in ingestion scripts |
| Better model (XGBoost, etc.) | `ml/train_model.py` — swap `LogisticRegression` in Pipeline |
| Multi-page dashboard | Convert `dashboard.py` to a `pages/` structure in Streamlit |
| User authentication | Add Streamlit `authenticator` component to `dashboard.py` |
| Model retraining trigger | Add a monitoring threshold in `monitor.py` and trigger `train_model.py` |
| API endpoint for predictions | Wrap `predict.py` logic in a FastAPI app |
