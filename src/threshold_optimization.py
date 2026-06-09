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

TARGET = "label"

DROP_COLUMNS = [
    "Date",
    "ticker",
    "label",
    "suspicion_score"
]

C_FN = 15_000_000
C_FP = 7_000


def load_split(file_path):
    df = pd.read_csv(file_path)

    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]

    return X, y


def compute_cost(y_true, y_pred, c_fn=C_FN, c_fp=C_FP):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total_cost = c_fn * fn + c_fp * fp
    normalized_cost = total_cost / len(y_true)

    return total_cost, normalized_cost, tn, fp, fn, tp


def find_best_threshold(y_true, y_proba):
    thresholds = np.linspace(0.0001, 0.9999, 10000)

    best_threshold = None
    best_cost = float("inf")
    best_result = None

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        total_cost, normalized_cost, tn, fp, fn, tp = compute_cost(y_true, y_pred)

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = threshold
            best_result = {
                "threshold": threshold,
                "total_cost": total_cost,
                "normalized_cost": normalized_cost,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp
            }

    return best_result


def evaluate_on_test(model_name, model, best_threshold):
    X_test, y_test = load_split(TEST_FILE)

    y_proba = model.predict_proba(X_test)[:, 1]

    y_pred_05 = (y_proba >= 0.5).astype(int)
    y_pred_opt = (y_proba >= best_threshold).astype(int)

    cost_05, norm_cost_05, tn_05, fp_05, fn_05, tp_05 = compute_cost(y_test, y_pred_05)
    cost_opt, norm_cost_opt, tn_opt, fp_opt, fn_opt, tp_opt = compute_cost(y_test, y_pred_opt)

    savings = (cost_05 - cost_opt) / cost_05 if cost_05 > 0 else 0

    result = {
        "model": model_name,
        "best_threshold": best_threshold,

        "accuracy_05": accuracy_score(y_test, y_pred_05),
        "precision_05": precision_score(y_test, y_pred_05, zero_division=0),
        "recall_05": recall_score(y_test, y_pred_05, zero_division=0),
        "f1_05": f1_score(y_test, y_pred_05, zero_division=0),
        "cost_05": cost_05,
        "TN_05": tn_05,
        "FP_05": fp_05,
        "FN_05": fn_05,
        "TP_05": tp_05,

        "accuracy_opt": accuracy_score(y_test, y_pred_opt),
        "precision_opt": precision_score(y_test, y_pred_opt, zero_division=0),
        "recall_opt": recall_score(y_test, y_pred_opt, zero_division=0),
        "f1_opt": f1_score(y_test, y_pred_opt, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "cost_opt": cost_opt,
        "normalized_cost_opt": norm_cost_opt,
        "savings": savings,
        "TN_opt": tn_opt,
        "FP_opt": fp_opt,
        "FN_opt": fn_opt,
        "TP_opt": tp_opt
    }

    return result


def optimize_model_threshold(model_name, model_path):
    model = joblib.load(model_path)

    X_val, y_val = load_split(VAL_THRESHOLD_FILE)
    y_proba_val = model.predict_proba(X_val)[:, 1]

    best = find_best_threshold(y_val, y_proba_val)

    print("\n" + "=" * 70)
    print(f"MODÈLE : {model_name}")
    print("=" * 70)
    print(f"Meilleur seuil validation : {best['threshold']:.6f}")
    print(f"Coût validation           : {best['total_cost']:,} €")
    print(f"FP validation             : {best['FP']}")
    print(f"FN validation             : {best['FN']}")
    print(f"TP validation             : {best['TP']}")
    print(f"TN validation             : {best['TN']}")

    test_result = evaluate_on_test(
        model_name=model_name,
        model=model,
        best_threshold=best["threshold"]
    )

    print("\nÉvaluation test avec seuil 0.5 vs seuil optimal")
    print(f"Coût seuil 0.5      : {test_result['cost_05']:,} €")
    print(f"Coût seuil optimal  : {test_result['cost_opt']:,} €")
    print(f"Savings             : {test_result['savings']:.2%}")
    print(f"Recall optimal      : {test_result['recall_opt']:.4f}")
    print(f"Precision optimal   : {test_result['precision_opt']:.4f}")
    print(f"FN optimal          : {test_result['FN_opt']}")
    print(f"FP optimal          : {test_result['FP_opt']}")

    return test_result


def main():
    bayes_threshold = C_FP / (C_FN + C_FP)

    print("\n==============================")
    print("OPTIMISATION DU SEUIL")
    print("==============================")
    print(f"c_FN = {C_FN:,} €")
    print(f"c_FP = {C_FP:,} €")
    print(f"Seuil théorique de Bayes = {bayes_threshold:.6f}")

    models = {
        "logistic_regression_calibrated_sigmoid": MODELS_DIR / "logistic_regression_calibrated_sigmoid.pkl",
        "random_forest_calibrated_sigmoid": MODELS_DIR / "random_forest_calibrated_sigmoid.pkl",
        "xgboost_calibrated_isotonic": MODELS_DIR / "xgboost_calibrated_isotonic.pkl"
    }

    all_results = []

    for model_name, model_path in models.items():
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        result = optimize_model_threshold(model_name, model_path)
        all_results.append(result)

    results_df = pd.DataFrame(all_results)

    output_path = RESULTS_DIR / "threshold_optimization_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nRésumé final threshold moving :")
    print(results_df)

    print("\nRésultats sauvegardés :", output_path)


if __name__ == "__main__":
    main()