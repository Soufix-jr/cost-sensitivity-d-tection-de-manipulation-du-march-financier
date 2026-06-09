import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("reports/results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VAL_THRESHOLD_FILE = PROCESSED_DATA_DIR / "val_threshold.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

OUTPUT_FILE = RESULTS_DIR / "recall_constrained_threshold_results.csv"

TARGET = "label"

DROP_COLUMNS = [
    "Date",
    "ticker",
    "label",
    "suspicion_score"
]

C_FN = 15_000_000
C_FP = 7_000


MODELS = {
    "logistic_regression_calibrated_sigmoid": MODELS_DIR / "logistic_regression_calibrated_sigmoid.pkl",
    "random_forest_calibrated_sigmoid": MODELS_DIR / "random_forest_calibrated_sigmoid.pkl",
    "xgboost_calibrated_isotonic": MODELS_DIR / "xgboost_calibrated_isotonic.pkl"
}


def load_split(path):
    df = pd.read_csv(path)

    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET].astype(int)

    return X, y


def compute_metrics(y_true, y_proba, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float).ravel()

    y_pred = (y_proba >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    cost = C_FN * fn + C_FP * fp

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "cost": cost,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }


def find_threshold_with_zero_fn(y_true, y_proba):
    """
    Version rapide.

    Pour avoir FN = 0 sur validation :
    le seuil doit être inférieur ou égal à la plus petite probabilité
    prédite parmi les vraies manipulations.

    On prend donc :
    threshold = min(P(manipulation) pour y=1)

    Cela donne le seuil le plus haut possible qui garde tous les positifs.
    """

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float).ravel()

    positive_probs = y_proba[y_true == 1]

    if len(positive_probs) == 0:
        return None

    threshold = positive_probs.min()

    return compute_metrics(y_true, y_proba, threshold)


def main():
    X_val, y_val = load_split(VAL_THRESHOLD_FILE)
    X_test, y_test = load_split(TEST_FILE)

    results = []

    for model_name, model_path in MODELS.items():
        print("\n" + "=" * 70)
        print(f"Modèle : {model_name}")
        print("=" * 70)

        model = joblib.load(model_path)

        y_proba_val = model.predict_proba(X_val)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]

        best_val = find_threshold_with_zero_fn(y_val, y_proba_val)

        if best_val is None:
            print("Aucun label positif dans validation threshold.")
            continue

        best_threshold = best_val["threshold"]

        test_metrics = compute_metrics(
            y_true=y_test,
            y_proba=y_proba_test,
            threshold=best_threshold
        )

        result = {
            "model": model_name,
            "strategy": "Recall constrained threshold",
            "threshold": best_threshold,

            "val_accuracy": best_val["accuracy"],
            "val_precision": best_val["precision"],
            "val_recall": best_val["recall"],
            "val_f1_score": best_val["f1_score"],
            "val_roc_auc": best_val["roc_auc"],
            "val_cost": best_val["cost"],
            "val_TN": best_val["TN"],
            "val_FP": best_val["FP"],
            "val_FN": best_val["FN"],
            "val_TP": best_val["TP"],

            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1_score": test_metrics["f1_score"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_cost": test_metrics["cost"],
            "test_TN": test_metrics["TN"],
            "test_FP": test_metrics["FP"],
            "test_FN": test_metrics["FN"],
            "test_TP": test_metrics["TP"]
        }

        results.append(result)

        print(f"Seuil choisi validation : {best_threshold:.6f}")
        print(f"Validation FP/FN       : {best_val['FP']}/{best_val['FN']}")
        print(f"Validation Recall      : {best_val['recall']:.4f}")
        print(f"Test FP/FN             : {test_metrics['FP']}/{test_metrics['FN']}")
        print(f"Test coût              : {test_metrics['cost']:,} €")
        print(f"Test precision         : {test_metrics['precision']:.4f}")
        print(f"Test recall            : {test_metrics['recall']:.4f}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(OUTPUT_FILE, index=False)

    print("\nRésumé Recall-Constrained Threshold :")
    print(results_df)

    print(f"\nFichier sauvegardé : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()