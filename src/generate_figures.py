import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)


PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
RESULTS_FILE = RESULTS_DIR / "threshold_optimization_results.csv"

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
    "Logistic Regression": "logistic_regression_calibrated_sigmoid.pkl",
    "Random Forest": "random_forest_calibrated_sigmoid.pkl",
    "XGBoost": "xgboost_calibrated_isotonic.pkl"
}


def load_test_data():
    df = pd.read_csv(TEST_FILE)

    X_test = df.drop(columns=DROP_COLUMNS)
    y_test = df[TARGET]

    return X_test, y_test


def compute_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return C_FN * fn + C_FP * fp


def plot_roc_curves(X_test, y_test):
    plt.figure(figsize=(8, 6))

    for model_label, model_file in MODELS.items():
        model = joblib.load(MODELS_DIR / model_file)
        y_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)

        plt.plot(fpr, tpr, label=model_label)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Calibrated Models")
    plt.legend()
    plt.grid(True)

    output_path = FIGURES_DIR / "roc_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure sauvegardée : {output_path}")


def plot_precision_recall_curves(X_test, y_test):
    plt.figure(figsize=(8, 6))

    for model_label, model_file in MODELS.items():
        model = joblib.load(MODELS_DIR / model_file)
        y_proba = model.predict_proba(X_test)[:, 1]

        precision, recall, _ = precision_recall_curve(y_test, y_proba)

        plt.plot(recall, precision, label=model_label)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Calibrated Models")
    plt.legend()
    plt.grid(True)

    output_path = FIGURES_DIR / "precision_recall_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure sauvegardée : {output_path}")


def plot_cost_vs_threshold(X_test, y_test):
    thresholds = np.linspace(0.0001, 0.9999, 1000)

    plt.figure(figsize=(8, 6))

    for model_label, model_file in MODELS.items():
        model = joblib.load(MODELS_DIR / model_file)
        y_proba = model.predict_proba(X_test)[:, 1]

        costs = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            cost = compute_cost(y_test, y_pred)
            costs.append(cost)

        plt.plot(thresholds, costs, label=model_label)

    plt.xlabel("Threshold")
    plt.ylabel("Total cost (€)")
    plt.title("Cost vs Threshold")
    plt.legend()
    plt.grid(True)

    output_path = FIGURES_DIR / "cost_vs_threshold.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure sauvegardée : {output_path}")


def plot_confusion_matrices_best_model(X_test, y_test):
    results_df = pd.read_csv(RESULTS_FILE)

    best_row = results_df.sort_values("cost_opt").iloc[0]
    best_model_name = best_row["model"]
    best_threshold = best_row["best_threshold"]

    if "random_forest" in best_model_name:
        model_file = "random_forest_calibrated_sigmoid.pkl"
        display_name = "Random Forest calibrated"
    elif "xgboost" in best_model_name:
        model_file = "xgboost_calibrated_isotonic.pkl"
        display_name = "XGBoost calibrated"
    else:
        model_file = "logistic_regression_calibrated_sigmoid.pkl"
        display_name = "Logistic Regression calibrated"

    model = joblib.load(MODELS_DIR / model_file)
    y_proba = model.predict_proba(X_test)[:, 1]

    y_pred_05 = (y_proba >= 0.5).astype(int)
    y_pred_opt = (y_proba >= best_threshold).astype(int)

    cm_05 = confusion_matrix(y_test, y_pred_05)
    cm_opt = confusion_matrix(y_test, y_pred_opt)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_05)
    disp.plot()
    plt.title(f"{display_name} - Threshold 0.5")
    output_path = FIGURES_DIR / "confusion_matrix_threshold_05.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure sauvegardée : {output_path}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_opt)
    disp.plot()
    plt.title(f"{display_name} - Optimal Threshold = {best_threshold:.4f}")
    output_path = FIGURES_DIR / "confusion_matrix_threshold_optimal.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure sauvegardée : {output_path}")


def plot_cost_comparison():
    results_df = pd.read_csv(RESULTS_FILE)

    labels = results_df["model"]
    cost_05 = results_df["cost_05"]
    cost_opt = results_df["cost_opt"]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, cost_05, width, label="Threshold 0.5")
    plt.bar(x + width / 2, cost_opt, width, label="Optimal threshold")

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Total cost (€)")
    plt.title("Cost comparison: threshold 0.5 vs optimal threshold")
    plt.legend()
    plt.grid(axis="y")

    output_path = FIGURES_DIR / "cost_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure sauvegardée : {output_path}")


def plot_savings_comparison():
    results_df = pd.read_csv(RESULTS_FILE)

    labels = results_df["model"]
    savings = results_df["savings"] * 100

    plt.figure(figsize=(10, 6))
    plt.bar(labels, savings)

    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Savings (%)")
    plt.title("Cost savings after Threshold Moving")
    plt.grid(axis="y")

    output_path = FIGURES_DIR / "savings_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure sauvegardée : {output_path}")


def main():
    X_test, y_test = load_test_data()

    plot_roc_curves(X_test, y_test)
    plot_precision_recall_curves(X_test, y_test)
    plot_cost_vs_threshold(X_test, y_test)
    plot_confusion_matrices_best_model(X_test, y_test)
    plot_cost_comparison()
    plot_savings_comparison()

    print("\nToutes les figures ont été générées dans reports/figures.")


if __name__ == "__main__":
    main()