import pandas as pd
from pathlib import Path
import joblib

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.frozen import FrozenEstimator


PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("reports/results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VAL_CALIB_FILE = PROCESSED_DATA_DIR / "val_calib.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

TARGET = "label"

DROP_COLUMNS = [
    "Date",
    "ticker",
    "label",
    "suspicion_score"
]


def load_split(file_path):
    df = pd.read_csv(file_path)

    X = df.drop(columns=DROP_COLUMNS)
    y = df[TARGET]

    return X, y


def calibrate_model(model_name, calibration_method):
    model_path = MODELS_DIR / f"{model_name}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    base_model = joblib.load(model_path)

    X_calib, y_calib = load_split(VAL_CALIB_FILE)
    X_test, y_test = load_split(TEST_FILE)

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_model),
        method=calibration_method
    )

    calibrated_model.fit(X_calib, y_calib)

    calibrated_path = MODELS_DIR / f"{model_name}_calibrated_{calibration_method}.pkl"
    joblib.dump(calibrated_model, calibrated_path)

    y_proba_test = calibrated_model.predict_proba(X_test)[:, 1]

    brier = brier_score_loss(y_test, y_proba_test)
    auc = roc_auc_score(y_test, y_proba_test)
    loss = log_loss(y_test, y_proba_test)

    result = {
        "model": model_name,
        "calibration": calibration_method,
        "brier_score": brier,
        "roc_auc": auc,
        "log_loss": loss,
        "model_path": str(calibrated_path)
    }

    print("\n" + "=" * 60)
    print(f"MODÈLE CALIBRÉ : {model_name} | {calibration_method}")
    print("=" * 60)
    print(f"Brier score : {brier:.6f}")
    print(f"ROC-AUC     : {auc:.6f}")
    print(f"Log loss    : {loss:.6f}")
    print(f"Sauvegardé  : {calibrated_path}")

    return result


def main():
    configs = [
        ("logistic_regression", "sigmoid"),
        ("random_forest", "sigmoid"),
        ("xgboost", "isotonic")
    ]

    results = []

    for model_name, method in configs:
        result = calibrate_model(model_name, method)
        results.append(result)

    results_df = pd.DataFrame(results)

    output_path = RESULTS_DIR / "calibrated_models_results.csv"
    results_df.to_csv(output_path, index=False)

    print("\nRésumé calibration :")
    print(results_df)

    print("\nRésultats sauvegardés :", output_path)


if __name__ == "__main__":
    main()