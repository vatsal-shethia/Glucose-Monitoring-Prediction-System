# ml/train_model.py

import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


# ---------------------- LOAD DATA ----------------------
def load_data():
    df = pd.read_csv("data/features.csv")
    return df


# ---------------------- PREPARE ----------------------
def prepare_data(df):
    X = df.drop(columns=["user_id", "timestamp", "spike"])
    y = df["spike"]
    return X, y


# ---------------------- TRAIN ----------------------
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"Before SMOTE:\n{y_train.value_counts()}")
    print(f"After SMOTE:\n{pd.Series(y_train_res).value_counts()}")

    # Pipeline (Scaling + Logistic Regression)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])

    model.fit(X_train_res, y_train_res)

    return model, X_test, y_test, X.columns


# ---------------------- EVALUATE ----------------------
def evaluate_model(model, X_test, y_test, threshold=0.6):

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    print("\n" + "="*40)
    print(f"MODEL PERFORMANCE (Threshold: {threshold})")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, preds)
    print(cm)

    print(f"\nInterpretation: TN={cm[0][0]}, TP={cm[1][1]}, FP={cm[0][1]}")


# ---------------------- FEATURE IMPORTANCE ----------------------
def show_feature_importance(model, feature_names):

    logreg = model.named_steps["logreg"]
    coefs = logreg.coef_[0]

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": coefs
    })

    importance["abs_importance"] = importance["importance"].abs()
    importance = importance.sort_values(by="abs_importance", ascending=False)

    print("\n🔍 FEATURE IMPORTANCE (Top Drivers):")
    print(importance[["feature", "importance"]].head(10))


# ---------------------- SAVE ----------------------
def save_model(model):
    joblib.dump(model, "ml/model.pkl")
    print("\nModel saved as ml/model.pkl")


# ---------------------- TRAIN XGBOOST ----------------------
def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier pipeline (no scaling needed)."""
    xgb_pipeline = Pipeline([
        ("xgb", XGBClassifier(
            n_estimators=100,
            max_depth=4,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ))
    ])
    xgb_pipeline.fit(X_train, y_train)
    return xgb_pipeline


# ---------------------- COMPARE MODELS ----------------------
def compare_models(logreg_model, xgb_model, X_test, y_test, threshold=0.6):
    """Print ROC-AUC, Precision, Recall and F1 for both models side by side."""

    def _metrics(model, label):
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > threshold).astype(int)
        return {
            "Model":     label,
            "ROC-AUC":   round(roc_auc_score(y_test, probs), 4),
            "Precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, preds, zero_division=0), 4),
            "F1":        round(f1_score(y_test, preds, zero_division=0), 4),
        }

    lr_metrics  = _metrics(logreg_model, "Logistic Regression")
    xgb_metrics = _metrics(xgb_model,    "XGBoost")

    print("\n" + "="*60)
    print(f"MODEL COMPARISON  (Threshold: {threshold})")
    print("="*60)
    header = f"{'Metric':<14}{'Logistic Regression':>22}{'XGBoost':>16}"
    print(header)
    print("-"*60)
    for key in ["ROC-AUC", "Precision", "Recall", "F1"]:
        print(f"{key:<14}{str(lr_metrics[key]):>22}{str(xgb_metrics[key]):>16}")
    print("="*60)


# ---------------------- MAIN ----------------------
if __name__ == "__main__":

    df = load_data()
    X, y = prepare_data(df)

    # --- Logistic Regression (existing) ---
    logreg_model, X_test, y_test, feature_names = train_model(X, y)
    evaluate_model(logreg_model, X_test, y_test)
    show_feature_importance(logreg_model, feature_names)
    save_model(logreg_model)  # saves ml/model.pkl

    # --- XGBoost (new) ---
    # Reconstruct the same train split + SMOTE so feature order is identical
    from sklearn.model_selection import train_test_split
    X_train_raw, _, y_train_raw, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_raw, y_train_raw)

    xgb_model = train_xgboost(X_train_res, y_train_res)
    print("\n[XGBoost] Individual evaluation:")
    evaluate_model(xgb_model, X_test, y_test)

    # --- Side-by-side comparison ---
    compare_models(logreg_model, xgb_model, X_test, y_test, threshold=0.6)

    # --- Save XGBoost ---
    joblib.dump(xgb_model, "ml/xgb_model.pkl")
    print("\nXGBoost model saved as ml/xgb_model.pkl")