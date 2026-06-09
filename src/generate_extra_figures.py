import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix


PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
RESULTS_DIR = Path("reports/results")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LABELED_FILE = PROCESSED_DATA_DIR / "market_labeled.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
FINAL_COMPARISON_FILE = RESULTS_DIR / "final_extended_model_comparison.csv"

DROP_COLUMNS = [
    "Date",
    "ticker",
    "label",
    "suspicion_score"
]

C_FN = 15_000_000
C_FP = 7_000


def save_fig(name):
    output_path = FIGURES_DIR / name
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure sauvegardée : {output_path}")


def load_labeled_data():
    df = pd.read_csv(LABELED_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_test_data():
    df = pd.read_csv(TEST_FILE)
    X = df.drop(columns=DROP_COLUMNS)
    y = df["label"].astype(int)
    return X, y, df


def plot_label_distribution(df):
    counts = df["label"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    plt.bar(["Normal", "Manipulation suspecte"], counts.values)
    plt.title("Distribution globale des classes")
    plt.ylabel("Nombre d'observations")
    plt.grid(axis="y", alpha=0.3)

    for i, value in enumerate(counts.values):
        plt.text(i, value, str(value), ha="center", va="bottom")

    save_fig("label_distribution.png")


def plot_label_distribution_by_ticker(df):
    table = pd.crosstab(df["ticker"], df["label"])
    table.columns = ["Normal", "Manipulation suspecte"]

    table.plot(kind="bar", figsize=(9, 5))
    plt.title("Distribution des classes par actif")
    plt.xlabel("Actif")
    plt.ylabel("Nombre d'observations")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    save_fig("label_distribution_by_ticker.png")


def plot_temporal_manipulations(df):
    monthly = (
    df.set_index("Date")
    .groupby("ticker")["label"]
    .resample("ME")
    .sum()
    .reset_index()
)

    plt.figure(figsize=(11, 5))

    for ticker in monthly["ticker"].unique():
        temp = monthly[monthly["ticker"] == ticker]
        plt.plot(temp["Date"], temp["label"], marker="o", linewidth=1.5, label=ticker)

    plt.title("Évolution temporelle des manipulations suspectes")
    plt.xlabel("Date")
    plt.ylabel("Nombre de signaux suspects")
    plt.grid(alpha=0.3)
    plt.legend()

    save_fig("temporal_suspicious_events.png")


def plot_suspicion_score_distribution(df):
    counts = df["suspicion_score"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Distribution du score de suspicion")
    plt.xlabel("Suspicion score")
    plt.ylabel("Nombre d'observations")
    plt.grid(axis="y", alpha=0.3)

    save_fig("suspicion_score_distribution.png")


def plot_feature_correlation(df):
    numeric_df = df.drop(columns=["Date", "ticker"], errors="ignore")
    corr = numeric_df.corr(numeric_only=True)

    selected_features = [
        "return",
        "abnormal_return",
        "price_zscore",
        "high_low_range",
        "realized_volatility_5",
        "realized_volatility_20",
        "volatility_ratio",
        "abnormal_volume",
        "volume_zscore",
        "return_volume_interaction",
        "volatility_volume_interaction",
        "momentum_5",
        "momentum_10",
        "price_reversal",
        "label"
    ]

    selected_features = [col for col in selected_features if col in corr.columns]
    corr_selected = corr.loc[selected_features, selected_features]

    plt.figure(figsize=(11, 9))
    plt.imshow(corr_selected, aspect="auto")
    plt.colorbar(label="Corrélation")
    plt.xticks(range(len(selected_features)), selected_features, rotation=90)
    plt.yticks(range(len(selected_features)), selected_features)
    plt.title("Matrice de corrélation des variables financières")

    save_fig("feature_correlation_heatmap.png")


def plot_random_forest_feature_importance():
    train_df = pd.read_csv(TRAIN_FILE)
    X_train = train_df.drop(columns=DROP_COLUMNS)

    model = joblib.load(MODELS_DIR / "random_forest.pkl")

    importances = model.feature_importances_
    features = X_train.columns

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 15 des variables importantes — Random Forest")
    plt.xlabel("Importance")
    plt.grid(axis="x", alpha=0.3)

    save_fig("random_forest_feature_importance.png")


def plot_xgboost_feature_importance():
    train_df = pd.read_csv(TRAIN_FILE)
    X_train = train_df.drop(columns=DROP_COLUMNS)

    model = joblib.load(MODELS_DIR / "xgboost.pkl")

    importances = model.feature_importances_
    features = X_train.columns

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": importances
    }).sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(9, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 15 des variables importantes — XGBoost")
    plt.xlabel("Importance")
    plt.grid(axis="x", alpha=0.3)

    save_fig("xgboost_feature_importance.png")


def plot_predicted_probability_distribution():
    X_test, y_test, _ = load_test_data()

    models = {
        "Random Forest calibrée": MODELS_DIR / "random_forest_calibrated_sigmoid.pkl",
        "XGBoost calibré": MODELS_DIR / "xgboost_calibrated_isotonic.pkl"
    }

    for model_name, model_path in models.items():
        model = joblib.load(model_path)
        y_proba = model.predict_proba(X_test)[:, 1]

        plt.figure(figsize=(8, 5))
        plt.hist(y_proba[y_test == 0], bins=30, alpha=0.7, label="Classe normale")
        plt.hist(y_proba[y_test == 1], bins=30, alpha=0.7, label="Manipulation suspecte")
        plt.title(f"Distribution des probabilités prédites — {model_name}")
        plt.xlabel("Probabilité prédite de manipulation")
        plt.ylabel("Nombre d'observations")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        clean_name = model_name.lower().replace(" ", "_").replace("é", "e")
        save_fig(f"predicted_probability_distribution_{clean_name}.png")


def plot_calibration_curves():
    X_test, y_test, _ = load_test_data()

    models = {
        "Logistic Regression": MODELS_DIR / "logistic_regression_calibrated_sigmoid.pkl",
        "Random Forest": MODELS_DIR / "random_forest_calibrated_sigmoid.pkl",
        "XGBoost": MODELS_DIR / "xgboost_calibrated_isotonic.pkl"
    }

    plt.figure(figsize=(8, 6))

    for model_name, model_path in models.items():
        model = joblib.load(model_path)
        y_proba = model.predict_proba(X_test)[:, 1]

        prob_true, prob_pred = calibration_curve(
            y_test,
            y_proba,
            n_bins=8,
            strategy="quantile"
        )

        plt.plot(prob_pred, prob_true, marker="o", label=model_name)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Calibration parfaite")
    plt.title("Courbes de calibration des modèles")
    plt.xlabel("Probabilité moyenne prédite")
    plt.ylabel("Fréquence observée")
    plt.legend()
    plt.grid(alpha=0.3)

    save_fig("calibration_curves.png")


def plot_cost_sensitivity_analysis():
    X_test, y_test, _ = load_test_data()

    model = joblib.load(MODELS_DIR / "random_forest_calibrated_sigmoid.pkl")
    y_proba = model.predict_proba(X_test)[:, 1]

    c_fn_values = [5_000_000, 10_000_000, 15_000_000, 30_000_000, 50_000_000]
    c_fp_values = [2_000, 5_000, 7_000, 10_000, 20_000]

    cost_matrix = np.zeros((len(c_fn_values), len(c_fp_values)))

    for i, c_fn in enumerate(c_fn_values):
        for j, c_fp in enumerate(c_fp_values):
            threshold = c_fp / (c_fn + c_fp)
            y_pred = (y_proba >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            cost = c_fn * fn + c_fp * fp

            cost_matrix[i, j] = cost / 1_000_000

    plt.figure(figsize=(8, 6))
    plt.imshow(cost_matrix, aspect="auto")
    plt.colorbar(label="Coût total en millions d'euros")
    plt.xticks(range(len(c_fp_values)), [f"{v//1000}k" for v in c_fp_values])
    plt.yticks(range(len(c_fn_values)), [f"{v//1_000_000}M" for v in c_fn_values])
    plt.xlabel("Coût faux positif c_FP")
    plt.ylabel("Coût faux négatif c_FN")
    plt.title("Analyse de sensibilité du coût total")

    for i in range(len(c_fn_values)):
        for j in range(len(c_fp_values)):
            plt.text(j, i, f"{cost_matrix[i, j]:.1f}", ha="center", va="center")

    save_fig("cost_sensitivity_analysis.png")


def main():
    df = load_labeled_data()

    plot_label_distribution(df)
    plot_label_distribution_by_ticker(df)
    plot_temporal_manipulations(df)
    plot_suspicion_score_distribution(df)
    plot_feature_correlation(df)
    plot_random_forest_feature_importance()
    plot_xgboost_feature_importance()
    plot_predicted_probability_distribution()
    plot_calibration_curves()
    plot_cost_sensitivity_analysis()

    print("\nToutes les figures supplémentaires ont été générées.")


if __name__ == "__main__":
    main()