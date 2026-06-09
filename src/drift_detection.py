import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp
from sklearn.metrics import confusion_matrix
import joblib
from datetime import datetime


# =====================================================
# CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "reports" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.csv"
RECENT_FILE = DATA_DIR / "test.csv"

THRESHOLD_RESULTS_FILE = RESULTS_DIR / "threshold_optimization_results.csv"

FEATURE_DRIFT_REPORT = RESULTS_DIR / "drift_feature_report.csv"
LABEL_DRIFT_REPORT = RESULTS_DIR / "drift_label_report.csv"
DRIFT_EVENTS_LOG = RESULTS_DIR / "drift_events_log.csv"
THRESHOLD_RECALIBRATION_REPORT = RESULTS_DIR / "drift_threshold_recalibration.csv"

C_FN = 15_000_000
C_FP = 7_000

PSI_WARNING_THRESHOLD = 0.1
PSI_DRIFT_THRESHOLD = 0.2
KS_PVALUE_THRESHOLD = 0.05

COST_DEGRADATION_THRESHOLD = 0.20

DROP_COLUMNS = [
    "Date",
    "ticker",
    "label",
    "suspicion_score"
]


# =====================================================
# BASIC HELPERS
# =====================================================

def load_data():
    train_df = pd.read_csv(TRAIN_FILE)
    recent_df = pd.read_csv(RECENT_FILE)

    train_df["Date"] = pd.to_datetime(train_df["Date"])
    recent_df["Date"] = pd.to_datetime(recent_df["Date"])

    return train_df, recent_df


def get_numeric_features(df):
    excluded = set(DROP_COLUMNS)
    numeric_cols = []

    for col in df.columns:
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)

    return numeric_cols


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def cost_score(y_true, y_pred, c_fn=C_FN, c_fp=C_FP):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = c_fn * fn + c_fp * fp

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "cost": float(cost)
    }


def append_log(event_type, severity, variable, message):
    log_row = pd.DataFrame([{
        "detection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_type": event_type,
        "severity": severity,
        "variable": variable,
        "message": message
    }])

    if DRIFT_EVENTS_LOG.exists():
        old_log = pd.read_csv(DRIFT_EVENTS_LOG)
        new_log = pd.concat([old_log, log_row], ignore_index=True)
    else:
        new_log = log_row

    new_log.to_csv(DRIFT_EVENTS_LOG, index=False)


# =====================================================
# PSI
# =====================================================

def calculate_psi(expected, actual, buckets=10):
    expected = safe_numeric(expected)
    actual = safe_numeric(actual)

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.quantile(expected, quantiles)
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) <= 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_percents = expected_counts / max(expected_counts.sum(), 1)
    actual_percents = actual_counts / max(actual_counts.sum(), 1)

    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)

    return float(np.sum(psi_values))


def interpret_psi(psi):
    if pd.isna(psi):
        return "not_available"

    if psi < PSI_WARNING_THRESHOLD:
        return "stable"

    if psi < PSI_DRIFT_THRESHOLD:
        return "moderate_drift"

    return "significant_drift"


# =====================================================
# FEATURE DRIFT
# =====================================================

def detect_feature_drift(train_df, recent_df):
    numeric_features = get_numeric_features(train_df)

    rows = []

    for feature in numeric_features:
        train_values = safe_numeric(train_df[feature])
        recent_values = safe_numeric(recent_df[feature])

        if len(train_values) < 20 or len(recent_values) < 20:
            continue

        ks_stat, ks_pvalue = ks_2samp(train_values, recent_values)
        psi = calculate_psi(train_values, recent_values)

        psi_status = interpret_psi(psi)

        drift_detected = bool(
            (psi_status == "significant_drift") or
            (ks_pvalue < KS_PVALUE_THRESHOLD and psi_status != "stable")
        )

        severity = "none"

        if psi_status == "moderate_drift":
            severity = "warning"

        if psi_status == "significant_drift":
            severity = "critical"

        rows.append({
            "feature": feature,
            "train_mean": train_values.mean(),
            "recent_mean": recent_values.mean(),
            "train_std": train_values.std(),
            "recent_std": recent_values.std(),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            "psi": psi,
            "psi_status": psi_status,
            "drift_detected": drift_detected,
            "severity": severity
        })

        if drift_detected:
            append_log(
                event_type="feature_drift",
                severity=severity,
                variable=feature,
                message=f"Feature drift detected on {feature}: PSI={psi:.4f}, KS p-value={ks_pvalue:.6f}"
            )

    report = pd.DataFrame(rows)
    report = report.sort_values(["drift_detected", "psi"], ascending=[False, False])
    report.to_csv(FEATURE_DRIFT_REPORT, index=False)

    return report


# =====================================================
# LABEL DRIFT
# =====================================================

def detect_label_drift(full_df, window_size=60):
    df = full_df.copy()
    df = df.sort_values("Date")

    rows = []

    unique_dates = sorted(df["Date"].dropna().unique())

    for end_date in unique_dates:
        start_date = pd.Timestamp(end_date) - pd.Timedelta(days=window_size)

        window_df = df[
            (df["Date"] >= start_date) &
            (df["Date"] <= end_date)
        ]

        if len(window_df) < 30:
            continue

        positive_rate = window_df["label"].mean()
        positive_count = int(window_df["label"].sum())
        total_count = len(window_df)

        rows.append({
            "window_start": start_date,
            "window_end": end_date,
            "total_count": total_count,
            "positive_count": positive_count,
            "positive_rate": positive_rate
        })

    label_report = pd.DataFrame(rows)

    if len(label_report) == 0:
        label_report.to_csv(LABEL_DRIFT_REPORT, index=False)
        return label_report

    baseline_rate = full_df["label"].mean()

    label_report["baseline_positive_rate"] = baseline_rate
    label_report["absolute_change"] = (
        label_report["positive_rate"] - baseline_rate
    )
    label_report["relative_change"] = (
        label_report["absolute_change"] / max(baseline_rate, 1e-8)
    )

    label_report["label_drift_detected"] = (
        label_report["relative_change"].abs() >= 0.5
    )

    label_report.to_csv(LABEL_DRIFT_REPORT, index=False)

    detected_windows = label_report[label_report["label_drift_detected"] == True]

    for _, row in detected_windows.iterrows():
        append_log(
            event_type="label_drift",
            severity="warning",
            variable="label_positive_rate",
            message=(
                f"Label drift detected between {row['window_start']} and {row['window_end']}: "
                f"positive rate={row['positive_rate']:.4f}, baseline={baseline_rate:.4f}"
            )
        )

    return label_report


# =====================================================
# COST DEGRADATION
# =====================================================

def evaluate_cost_degradation(recent_df):
    if not THRESHOLD_RESULTS_FILE.exists():
        return None

    threshold_df = pd.read_csv(THRESHOLD_RESULTS_FILE)

    rows = []

    X_recent = recent_df.drop(columns=DROP_COLUMNS)
    y_recent = recent_df["label"].astype(int)

    for _, row in threshold_df.iterrows():
        model_name = row["model"]
        threshold = row["best_threshold"]
        reference_cost = row["cost_opt"]

        model_path = MODELS_DIR / f"{model_name}.pkl"

        if not model_path.exists():
            continue

        model = joblib.load(model_path)
        y_proba = model.predict_proba(X_recent)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        current_cost_info = cost_score(y_recent, y_pred)
        current_cost = current_cost_info["cost"]

        degradation = (current_cost - reference_cost) / max(reference_cost, 1)

        degraded = degradation > COST_DEGRADATION_THRESHOLD

        rows.append({
            "model": model_name,
            "threshold": threshold,
            "reference_cost": reference_cost,
            "current_cost": current_cost,
            "cost_degradation_ratio": degradation,
            "cost_degradation_detected": degraded,
            "TN": current_cost_info["TN"],
            "FP": current_cost_info["FP"],
            "FN": current_cost_info["FN"],
            "TP": current_cost_info["TP"]
        })

        if degraded:
            append_log(
                event_type="cost_degradation",
                severity="critical",
                variable=model_name,
                message=(
                    f"Cost degradation detected for {model_name}: "
                    f"reference={reference_cost:.0f}, current={current_cost:.0f}, "
                    f"degradation={degradation:.2%}"
                )
            )

    if len(rows) == 0:
        return None

    return pd.DataFrame(rows)


# =====================================================
# THRESHOLD RECALIBRATION
# =====================================================

def find_optimal_threshold(y_true, y_proba, c_fn=C_FN, c_fp=C_FP):
    thresholds = np.unique(np.quantile(y_proba, np.linspace(0, 1, 1000)))
    thresholds = np.concatenate(([0.000001], thresholds, [0.999999]))
    thresholds = np.unique(thresholds)

    best_threshold = None
    best_cost = np.inf
    best_info = None

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        info = cost_score(y_true, y_pred, c_fn=c_fn, c_fp=c_fp)

        if info["cost"] < best_cost:
            best_cost = info["cost"]
            best_threshold = threshold
            best_info = info

    return best_threshold, best_cost, best_info


def recalibrate_thresholds_if_needed(recent_df, feature_report, cost_degradation_report):
    significant_feature_drift = False

    if feature_report is not None and len(feature_report) > 0:
        significant_feature_drift = bool(
            (feature_report["psi"] > PSI_DRIFT_THRESHOLD).any()
        )

    cost_degradation_detected = False

    if cost_degradation_report is not None and len(cost_degradation_report) > 0:
        cost_degradation_detected = bool(
            cost_degradation_report["cost_degradation_detected"].any()
        )

    should_recalibrate = significant_feature_drift or cost_degradation_detected

    rows = []

    if not should_recalibrate:
        pd.DataFrame(rows).to_csv(THRESHOLD_RECALIBRATION_REPORT, index=False)
        return pd.DataFrame(rows)

    if not THRESHOLD_RESULTS_FILE.exists():
        return pd.DataFrame(rows)

    threshold_df = pd.read_csv(THRESHOLD_RESULTS_FILE)

    X_recent = recent_df.drop(columns=DROP_COLUMNS)
    y_recent = recent_df["label"].astype(int)

    for _, row in threshold_df.iterrows():
        model_name = row["model"]
        old_threshold = row["best_threshold"]
        old_cost = row["cost_opt"]

        model_path = MODELS_DIR / f"{model_name}.pkl"

        if not model_path.exists():
            continue

        model = joblib.load(model_path)
        y_proba = model.predict_proba(X_recent)[:, 1]

        new_threshold, new_cost, new_info = find_optimal_threshold(
            y_recent,
            y_proba,
            c_fn=C_FN,
            c_fp=C_FP
        )

        improvement = (old_cost - new_cost) / max(old_cost, 1)

        rows.append({
            "recalibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "old_threshold": old_threshold,
            "new_threshold": new_threshold,
            "old_cost": old_cost,
            "new_cost": new_cost,
            "relative_improvement": improvement,
            "TN": new_info["TN"],
            "FP": new_info["FP"],
            "FN": new_info["FN"],
            "TP": new_info["TP"],
            "trigger_feature_drift": significant_feature_drift,
            "trigger_cost_degradation": cost_degradation_detected
        })

        append_log(
            event_type="threshold_recalibration",
            severity="action",
            variable=model_name,
            message=(
                f"Threshold recalibrated for {model_name}: "
                f"old={old_threshold:.6f}, new={new_threshold:.6f}, "
                f"old_cost={old_cost:.0f}, new_cost={new_cost:.0f}"
            )
        )

    report = pd.DataFrame(rows)
    report.to_csv(THRESHOLD_RECALIBRATION_REPORT, index=False)

    return report


# =====================================================
# GLOBAL MONITORING PIPELINE
# =====================================================

def run_drift_monitoring():
    train_df, recent_df = load_data()

    full_df = pd.concat([train_df, recent_df], ignore_index=True)

    print("\n[1/4] Feature drift detection...")
    feature_report = detect_feature_drift(train_df, recent_df)
    print(f"Feature drift report saved: {FEATURE_DRIFT_REPORT}")

    print("\n[2/4] Label drift detection...")
    label_report = detect_label_drift(full_df)
    print(f"Label drift report saved: {LABEL_DRIFT_REPORT}")

    print("\n[3/4] Cost degradation monitoring...")
    cost_degradation_report = evaluate_cost_degradation(recent_df)

    if cost_degradation_report is not None:
        cost_degradation_file = RESULTS_DIR / "drift_cost_degradation_report.csv"
        cost_degradation_report.to_csv(cost_degradation_file, index=False)
        print(f"Cost degradation report saved: {cost_degradation_file}")
    else:
        print("Cost degradation report skipped.")

    print("\n[4/4] Threshold recalibration if drift detected...")
    recalibration_report = recalibrate_thresholds_if_needed(
        recent_df=recent_df,
        feature_report=feature_report,
        cost_degradation_report=cost_degradation_report
    )

    print(f"Threshold recalibration report saved: {THRESHOLD_RECALIBRATION_REPORT}")
    print(f"Drift event log saved: {DRIFT_EVENTS_LOG}")

    print("\nDrift monitoring completed.")

    return {
        "feature_report": feature_report,
        "label_report": label_report,
        "cost_degradation_report": cost_degradation_report,
        "recalibration_report": recalibration_report
    }


if __name__ == "__main__":
    run_drift_monitoring()