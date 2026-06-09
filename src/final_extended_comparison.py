import pandas as pd
from pathlib import Path


RESULTS_DIR = Path("reports/results")

THRESHOLD_FILE = RESULTS_DIR / "threshold_optimization_results.csv"
SMOTE_FILE = RESULTS_DIR / "smote_comparison_results.csv"
RECALL_CONSTRAINED_FILE = RESULTS_DIR / "recall_constrained_threshold_results.csv"

OUTPUT_FILE = RESULTS_DIR / "final_extended_model_comparison.csv"


def main():
    rows = []

    threshold_df = pd.read_csv(THRESHOLD_FILE)
    smote_df = pd.read_csv(SMOTE_FILE)
    recall_df = pd.read_csv(RECALL_CONSTRAINED_FILE)

    for _, row in threshold_df.iterrows():
        rows.append({
            "model": row["model"],
            "method": "Calibration + Threshold Moving",
            "threshold": row["best_threshold"],
            "accuracy": row["accuracy_opt"],
            "precision": row["precision_opt"],
            "recall": row["recall_opt"],
            "f1_score": row["f1_opt"],
            "roc_auc": row["roc_auc"],
            "cost": row["cost_opt"],
            "TN": row["TN_opt"],
            "FP": row["FP_opt"],
            "FN": row["FN_opt"],
            "TP": row["TP_opt"]
        })

    for _, row in smote_df.iterrows():
        rows.append({
            "model": row["model"],
            "method": "SMOTE + Threshold 0.5",
            "threshold": row["threshold"],
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1_score"],
            "roc_auc": row["roc_auc"],
            "cost": row["cost"],
            "TN": row["TN"],
            "FP": row["FP"],
            "FN": row["FN"],
            "TP": row["TP"]
        })

    for _, row in recall_df.iterrows():
        rows.append({
            "model": row["model"],
            "method": "Recall-Constrained Threshold",
            "threshold": row["threshold"],
            "accuracy": row["test_accuracy"],
            "precision": row["test_precision"],
            "recall": row["test_recall"],
            "f1_score": row["test_f1_score"],
            "roc_auc": row["test_roc_auc"],
            "cost": row["test_cost"],
            "TN": row["test_TN"],
            "FP": row["test_FP"],
            "FN": row["test_FN"],
            "TP": row["test_TP"]
        })

    final_df = pd.DataFrame(rows)
    final_df = final_df.sort_values("cost").reset_index(drop=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nComparaison finale étendue :")
    print(final_df)

    print(f"\nFichier sauvegardé : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()