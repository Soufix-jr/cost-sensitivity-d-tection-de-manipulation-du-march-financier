import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("reports/results")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"


TARGET = "label"

DROP_COLUMNS = [
    "Date",
    "ticker",
    "label",
    "suspicion_score"
]


def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    X_train = train_df.drop(columns=DROP_COLUMNS)
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=DROP_COLUMNS)
    y_test = test_df[TARGET]

    return X_train, y_train, X_test, y_test


def get_models():
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42
                ))
            ]
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=35,
            random_state=42,
            n_jobs=-1
        )
    }

    return models


def evaluate_model(name, model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results = {
        "model": name,
        "threshold": 0.5,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    print("\n" + "=" * 60)
    print(f"MODÈLE : {name}")
    print("=" * 60)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport détaillé :")
    print(classification_report(y_test, y_pred, zero_division=0))

    return results


def main():
    X_train, y_train, X_test, y_test = load_data()

    print("Train shape :", X_train.shape)
    print("Test shape  :", X_test.shape)
    print("Taux positif train :", y_train.mean())
    print("Taux positif test  :", y_test.mean())

    models = get_models()

    all_results = []

    for name, model in models.items():
        print(f"\nEntraînement du modèle : {name}")

        model.fit(X_train, y_train)

        model_path = MODELS_DIR / f"{name}.pkl"
        joblib.dump(model, model_path)

        print(f"Modèle sauvegardé : {model_path}")

        results = evaluate_model(name, model, X_test, y_test)
        all_results.append(results)

    results_df = pd.DataFrame(all_results)

    output_path = RESULTS_DIR / "baseline_models_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nRésultats sauvegardés :", output_path)
    print("\nRésumé final :")
    print(results_df)


if __name__ == "__main__":
    main()