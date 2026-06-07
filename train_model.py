"""
Simple training script for Credit Card Fraud Detection System.

- Loads dataset from dataset/creditcard.csv
- Performs basic EDA (shape, null checks, class distribution)
- Splits data into train/test sets
- Trains a RandomForestClassifier
- Prints accuracy, precision, recall and confusion matrix
- Saves trained model to models/fraud_model.pkl
- Generates and saves a matplotlib chart of fraud vs non-fraud counts

This script is intentionally kept simple and beginner friendly.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


DATA_PATH = os.path.join("dataset", "creditcard.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
CHART_PATH = os.path.join(MODEL_DIR, "fraud_vs_nonfraud.png")
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Dataset not found at '{path}'. Please put creditcard.csv in the dataset/ folder.")
        sys.exit(1)
    return pd.read_csv(path)


def exploratory_data_analysis(df: pd.DataFrame) -> None:
    print("\n=== Exploratory Data Analysis ===")
    print(f"Shape: {df.shape}")
    print("\nNull values per column:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    counts = df["Class"].value_counts()
    print(counts)

    # Save a simple chart of class distribution
    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", color=["tab:green", "tab:red"])
    plt.title("Fraud (1) vs Non-Fraud (0) Transactions")
    plt.xlabel("Class")
    plt.ylabel("Number of Transactions")
    plt.xticks([0, 1], ["Non-Fraud (0)", "Fraud (1)"])
    plt.tight_layout()

    # Ensure model directory exists for saving chart
    os.makedirs(MODEL_DIR, exist_ok=True)
    plt.savefig(CHART_PATH)
    plt.close()
    print(f"Saved class distribution chart to: {CHART_PATH}")


def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return X, y


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    print("\n=== Train / Test Split ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    print("\n=== Training RandomForestClassifier ===")
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("\n=== Evaluation on Test Set ===")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save the trained model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved trained model to: {MODEL_PATH}")

    return clf


def main():
    df = load_data(DATA_PATH)
    exploratory_data_analysis(df)
    X, y = prepare_features(df)
    train_and_evaluate(X, y)


if __name__ == "__main__":
    main()