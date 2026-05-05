# 🧬 Lingo-Style Glucose Intelligence System

An end-to-end machine learning system that simulates user health data, predicts glucose spikes, and provides personalized behavioral insights and actionable recommendations.

---

## 🚀 Overview

This project replicates the core idea behind Abbott’s Lingo platform — transforming continuous health data into meaningful insights.

It combines:

- Synthetic health data generation
- Time-aware feature engineering
- Machine learning for spike prediction
- A Streamlit dashboard for insights & recommendations

---

## ⚙️ System Architecture

Data Simulation → Feature Engineering → Model Training → Monitoring → Dashboard

### Pipeline Flow:

1. Generate users, meals, activity, sleep, glucose, heart rate
2. Create time-based features (meals, activity, sleep, HR)
3. Train ML model with imbalance handling (SMOTE)
4. Evaluate and monitor predictions
5. Visualize insights via dashboard

---

## 📊 Features

### 🔬 Data Simulation

- Physiologically-inspired glucose model
- Meal → increases glucose
- Activity → reduces glucose
- Heart rate correlated with activity

---

### 🧠 Feature Engineering

- Previous glucose (time dependency)
- Carbs from last meal
- Steps in last 1 hour
- Heart rate (30-min rolling average)
- Sleep duration & quality
- Time features (hour, weekend)

---

### 🤖 Machine Learning

- Logistic Regression (interpretable)
- SMOTE for class imbalance
- Custom threshold (0.6) for recall prioritization
- ROC-AUC ≈ 0.97

---

### 📊 Dashboard (Streamlit)

- Glucose trends
- Time in range (TIR)
- Behavioral insights:
  - Meal impact
  - Activity patterns
  - Sleep effects
  - Heart rate context
- Personalized recommendations

---

### 📡 Monitoring

- Prediction rate tracking
- Feature drift checks
- Confidence distribution analysis

---

## 🧠 Key Technical Decisions

### Why Logistic Regression?

- Interpretable coefficients
- Suitable for healthcare context
- Fast inference

---

### Why SMOTE?

- Dataset has extreme imbalance (~0.6% spikes)
- Improves learning of minority class

---

### Why Threshold = 0.6?

- Prioritizes recall over precision
- Missing a spike is worse than a false alarm

---

## 📉 Limitations

- Synthetic dataset (not real-world variability)
- Precision is low due to class imbalance
- No real-time streaming

---

## 🚀 How to Run

```bash
# Run full pipeline
python main.py

# Skip data generation
python main.py --skip-ingestion

# Launch dashboard
streamlit run dashboard.py
```
