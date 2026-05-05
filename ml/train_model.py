# ml/train_model.py

import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
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


# ---------------------- MAIN ----------------------
if __name__ == "__main__":

    df = load_data()
    X, y = prepare_data(df)

    model, X_test, y_test, feature_names = train_model(X, y)

    evaluate_model(model, X_test, y_test)
    show_feature_importance(model, feature_names)

    save_model(model)