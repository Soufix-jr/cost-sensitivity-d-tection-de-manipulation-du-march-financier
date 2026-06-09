import pandas as pd
import numpy as np
from pathlib import Path


PROCESSED_DATA_DIR = Path("data/processed")


INPUT_FILE = PROCESSED_DATA_DIR / "market_features_all.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "market_labeled.csv"


def generate_statistical_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Génère des labels de manipulation à partir de règles statistiques.

    Label = 1 si plusieurs signaux anormaux apparaissent ensemble :
    - rendement anormal élevé
    - volume anormal élevé
    - volatilité élevée
    - interaction prix-volume élevée
    """

    df = df.copy()

    # =========================
    # 1. Conditions principales
    # =========================

    condition_abnormal_return = df["abnormal_return"].abs() > 2.5

    condition_abnormal_volume = df["abnormal_volume"] > 2.5

    condition_price_zscore = df["price_zscore"].abs() > 2.0

    condition_volatility = df["volatility_ratio"] > 1.5

    condition_interaction = (
        df["return_volume_interaction"]
        > df["return_volume_interaction"].quantile(0.97)
    )

    # =========================
    # 2. Score de suspicion
    # =========================

    df["suspicion_score"] = (
        condition_abnormal_return.astype(int)
        + condition_abnormal_volume.astype(int)
        + condition_price_zscore.astype(int)
        + condition_volatility.astype(int)
        + condition_interaction.astype(int)
    )

    # =========================
    # 3. Label final
    # =========================
    # Manipulation suspectée si au moins 3 signaux sont actifs

    df["label"] = (df["suspicion_score"] >= 3).astype(int)

    return df


def inject_known_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des labels sur certaines périodes connues de tension extrême.
    Exemple : épisode GameStop / AMC en janvier-février 2021.

    Ce n'est pas une vérité judiciaire.
    C'est un proxy académique pour enrichir les cas positifs.
    """

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    known_events = {
        "GME": [
            ("2021-01-25", "2021-02-05"),
        ],
        "AMC": [
            ("2021-01-25", "2021-02-05"),
        ],
    }

    for ticker, periods in known_events.items():
        for start_date, end_date in periods:
            mask = (
                (df["ticker"] == ticker)
                & (df["Date"] >= pd.to_datetime(start_date))
                & (df["Date"] <= pd.to_datetime(end_date))
            )

            df.loc[mask, "label"] = 1
            df.loc[mask, "suspicion_score"] = np.maximum(
                df.loc[mask, "suspicion_score"],
                4
            )

    return df


def print_label_report(df: pd.DataFrame) -> None:
    """
    Affiche un rapport simple sur la distribution des labels.
    """

    total = len(df)
    positives = df["label"].sum()
    negatives = total - positives
    positive_rate = positives / total

    print("\n==============================")
    print("RAPPORT DE LABELLISATION")
    print("==============================")
    print(f"Nombre total d'observations : {total}")
    print(f"Classe 0 - normal           : {negatives}")
    print(f"Classe 1 - manipulation     : {positives}")
    print(f"Taux de manipulation        : {positive_rate:.4%}")

    print("\nDistribution par ticker :")
    print(pd.crosstab(df["ticker"], df["label"], normalize="index") * 100)

    print("\nNombre de labels positifs par ticker :")
    print(pd.crosstab(df["ticker"], df["label"]))


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}. "
            "Lance d'abord src/feature_engineering.py"
        )

    df = pd.read_csv(INPUT_FILE)

    df = generate_statistical_labels(df)

    df = inject_known_event_labels(df)

    df.to_csv(OUTPUT_FILE, index=False)

    print_label_report(df)

    print(f"\nDataset labellisé sauvegardé : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()