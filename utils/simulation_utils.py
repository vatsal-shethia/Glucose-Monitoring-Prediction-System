# utils/simulation_utils.py

"""
Central place for all simulation constants
"""

# --- Glucose Model ---
BASELINE_GLUCOSE = 90
MEAL_COEFF = 0.5
ACTIVITY_COEFF = 0.005
GLUCOSE_MIN = 70
GLUCOSE_MAX = 180

# --- Activity ---
ACTIVITY_TYPES = ["walking", "running", "sedentary"]
ACTIVITY_PROBS = [0.5, 0.2, 0.3]

# --- Heart Rate ---
BASE_HEART_RATE = 70
HR_WALKING_BOOST = 10
HR_RUNNING_BOOST = 25

# --- Noise ---
GLUCOSE_NOISE_STD = 5
HEART_RATE_NOISE_STD = 3