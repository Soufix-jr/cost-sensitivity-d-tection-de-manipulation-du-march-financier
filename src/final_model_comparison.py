import pandas as pd
from pathlib import Path


RESULTS_DIR = Path("reports/results")

THRESHOLD_FILE = RESULTS_DIR / "threshold_optimization_results.csv"
SMOTE_FILE = RESULTS_DIR / "smote_comparison_results.csv"
OUTPUT_FILE = RESULTS_DIR / "final_model_comparison.csv"


def main():
    threshold_df = pd.read_csv(THRESHOLD_FILE)
    smote_df = pd.read_csv(SMOTE_FILE)

    rows = []

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
            "savings": row["savings"],
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
            "savings": None,
            "TN": row["TN"],
            "FP": row["FP"],
            "FN": row["FN"],
            "TP": row["TP"]
        })

    final_df = pd.DataFrame(rows)
    final_df = final_df.sort_values("cost").reset_index(drop=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\nComparaison finale des modèles :")
    print(final_df)

    print(f"\nFichier sauvegardé : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()