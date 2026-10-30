# -*- coding: utf-8 -*-
"""Paisa Bazar.ipynb

## 1. Summary and Technical Documentation

This notebook details the process of building and comparing five machine learning models to predict loan credit scores. The best model will be selected and saved for deployment.

The project follows a structured approach:
1.  **Data Exploration & Cleaning:** Understanding and preparing the dataset.
2.  **Exploratory Data Analysis (EDA):** Using visualizations to find insights.
3.  **Feature Engineering:** Preparing data for modeling.
4.  **Model Building & Comparison:** Training and evaluating five different classification models.
5.  **Model Selection & Saving:** Choosing the best model and saving it for deployment.

**Technical Stack:**
* **Languages:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly

## 2. Exploration: Head, Tail, Summary, Data Dictionary

Let's start by loading and getting a first look at the data.
"""

"""
Fixed, runnable PaisaBazar script.
Changes:
 - corrected function signatures / return types
 - robust file path (absolute), stratify fallback
 - added simple EDA prints (head, tail, describe, missing, duplicates)
 - safe encoding and scaling
 - robust training/evaluation and saving best model
 - minimal plotting disabled if headless
"""
import os
import sys
import pickle
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


def load_data(path: str) -> pd.DataFrame:
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    try:
        df = pd.read_csv(path)
        print(f"Loaded dataset: {path} -> shape {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: file not found at '{path}'. Update path or place CSV there.")
        sys.exit(1)


def quick_eda(df: pd.DataFrame) -> None:
    print("\n=== Quick EDA ===")
    print("Head:")
    print(df.head(3).to_string(index=False))
    print("\nTail:")
    print(df.tail(3).to_string(index=False))
    print("\nShape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes)
    print("\nDescribe (numeric):")
    print(df.describe().T)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("=== End EDA ===\n")


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    df = df.copy()

    # Basic EDA print
    quick_eda(df)

    # Drop non-predictive/sensitive columns if present
    drop_cols = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Type_of_Loan']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    if 'Credit_Score' not in df.columns:
        print("Error: 'Credit_Score' column not found.")
        sys.exit(1)

    df['Credit_Score'] = df['Credit_Score'].astype(str).str.strip()

    # Map textual credit score -> numeric
    score_map = {'Good': 2, 'Standard': 1, 'Poor': 0}
    df['Credit_Score_MAPPED'] = df['Credit_Score'].map(score_map)

    # Drop unmapped targets
    n_unmapped = int(df['Credit_Score_MAPPED'].isna().sum())
    if n_unmapped:
        print(f"Dropping {n_unmapped} rows with unknown Credit_Score.")
        df = df[~df['Credit_Score_MAPPED'].isna()]

    # Simple missing value handling: drop any remaining NA
    if df.isnull().values.any():
        print("Dropping rows with missing values (simple strategy).")
        df = df.dropna()

    # One-hot encode common categorical cols if present
    categorical_cols = [c for c in ['Occupation', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour'] if c in df.columns]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Features and target
    X = df.drop(columns=['Credit_Score', 'Credit_Score_MAPPED'], errors='ignore')
    y = df['Credit_Score_MAPPED'].astype(int)

    # Train-test split with stratify fallback
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numeric features
    scaler = StandardScaler()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    if num_cols:
        X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    else:
        # nothing numeric to scale; keep copies
        scaler = None

    print("Preprocessing complete. Train shape:", X_train.shape, "Test shape:", X_test.shape)
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_evaluate(X_train: pd.DataFrame, X_test: pd.DataFrame,
                       X_train_scaled: pd.DataFrame, X_test_scaled: pd.DataFrame,
                       y_train: pd.Series, y_test: pd.Series) -> Dict[str, dict]:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        # "Support Vector Machine": SVC(probability=False, random_state=42),
        # Use LinearSVC for much faster training on large datasets.
        # It solves the linear SVM optimization using an efficient liblinear-style solver.
        "Support Vector Machine": LinearSVC(max_iter=5000, tol=1e-4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results: Dict[str, dict] = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            # Use scaled for linear/distance models if scaler available
            if name in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"] and X_train_scaled is not None:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            present_labels = sorted(np.unique(np.concatenate([y_test.values, np.array(y_pred)])))
            inv_map = {2: 'Good', 1: 'Standard', 0: 'Poor'}
            target_names = [inv_map.get(int(l), str(int(l))) for l in present_labels]
            report = classification_report(y_test, y_pred, labels=present_labels, target_names=target_names, zero_division=0)

            results[name] = {'model': model, 'accuracy': acc, 'report': report}
            print(f"Accuracy: {acc:.4f}")
            print(report)
        except Exception as ex:
            print(f"Failed training {name}: {ex}")
            results[name] = {'model': None, 'accuracy': 0.0, 'report': str(ex)}

    return results


def save_best_model(results: Dict[str, dict], X_columns: List[str], scaler: StandardScaler = None) -> None:
    valid = {k: v for k, v in results.items() if v.get('model') is not None}
    if not valid:
        print("No valid models to save.")
        return
    best_name = max(valid.keys(), key=lambda k: valid[k]['accuracy'])
    best = valid[best_name]['model']
    best_acc = valid[best_name]['accuracy']
    print(f"Best model: {best_name} (acc={best_acc:.4f})")

    with open('loan_model.pkl', 'wb') as f:
        pickle.dump(best, f)
    with open('model_columns.pkl', 'wb') as f:
        pickle.dump(list(X_columns), f)

    if scaler is not None and best_name in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"]:
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        print("Scaler saved to 'scaler.pkl'")

    print("Saved artifacts: loan_model.pkl, model_columns.pkl")


def plot_accuracy(results: Dict[str, dict]) -> None:
    try:
        summary = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [v['accuracy'] for v in results.values()]
        }).sort_values('Accuracy', ascending=False)
        plt.figure(figsize=(8, 4))
        sns.barplot(x='Accuracy', y='Model', data=summary, palette='rocket')
        plt.xlim(0, 1)
        plt.title('Model Accuracy Comparison')
        plt.tight_layout()
        plt.show()
    except Exception:
        # plotting may fail in headless environments; ignore
        pass


def main():
    csv_path = os.path.join('d:\\', 'Internships', 'labmentix', 'paisa_bazar_loan_dataset.csv')
    df = load_data(csv_path)

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess(df)

    results = train_and_evaluate(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)

    print("\n=== Accuracy summary ===")
    for name, v in sorted(results.items(), key=lambda kv: kv[1]['accuracy'], reverse=True):
        print(f"{name}: {v['accuracy']:.4f}")

    plot_accuracy(results)

    save_best_model(results, list(X_train.columns), scaler=scaler)


if __name__ == "__main__":
    main()