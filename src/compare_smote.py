import pandas as pd
from pathlib import Path
import joblib

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("reports/results")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

OUTPUT_FILE = RESULTS_DIR / "smote_comparison_results.csv"

TARGET = "label"

DROP_COLUMNS = [
    "Date",
    "ticker",
    "label",
    "suspicion_score"
]

C_FN = 15_000_000
C_FP = 7_000


def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    X_train = train_df.drop(columns=DROP_COLUMNS)
    y_train = train_df[TARGET]

    X_test = test_df.drop(columns=DROP_COLUMNS)
    y_test = test_df[TARGET]

    return X_train, y_train, X_test, y_test


def compute_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = C_FN * fn + C_FP * fp
    return cost, tn, fp, fn, tp


def main():
    X_train, y_train, X_test, y_test = load_data()

    print("Train shape :", X_train.shape)
    print("Test shape  :", X_test.shape)
    print("Taux positif train avant SMOTE :", y_train.mean())

    smote_rf = ImbPipeline(
        steps=[
            ("smote", SMOTE(
                sampling_strategy=0.30,
                random_state=42,
                k_neighbors=5
            )),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ))
        ]
    )

    print("\nEntraînement Random Forest + SMOTE...")
    smote_rf.fit(X_train, y_train)

    model_path = MODELS_DIR / "random_forest_smote.pkl"
    joblib.dump(smote_rf, model_path)

    print(f"Modèle sauvegardé : {model_path}")

    y_proba = smote_rf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    cost, tn, fp, fn, tp = compute_cost(y_test, y_pred)

    result = {
        "model": "random_forest_smote",
        "threshold": 0.5,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cost": cost,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    results_df = pd.DataFrame([result])
    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nRésultat Random Forest + SMOTE :")
    print(results_df)

    print(f"\nRésultats sauvegardés : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()