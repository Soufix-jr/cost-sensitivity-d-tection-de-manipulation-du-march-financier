import pandas as pd
import numpy as np
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_yfinance_file(file_path: Path) -> pd.DataFrame:
    """
    Lit un fichier yfinance et corrige les problèmes fréquents :
    - colonnes multi-index exportées en CSV
    - lignes parasites comme Ticker / Price
    - colonnes numériques lues comme texte
    """

    df = pd.read_csv(file_path)

    # Cas où yfinance a sauvegardé une ligne parasite dans le CSV
    # Exemple : une ligne contenant le nom du ticker au lieu de valeurs numériques
    if "Date" not in df.columns:
        df = pd.read_csv(file_path, skiprows=2)

    # Nettoyage noms colonnes
    df.columns = [str(col).strip() for col in df.columns]

    # Supprimer les lignes où Date n'est pas une vraie date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    required_columns = ["Open", "High", "Low", "Close", "Volume"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans {file_path.name} : {col}")

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns)

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def build_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Construit les features financières.
    """

    # =========================
    # 1. Features de prix
    # =========================

    df["return"] = df["Close"].pct_change()

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    df["rolling_return_mean_20"] = df["return"].rolling(window=20).mean()

    df["rolling_return_std_20"] = df["return"].rolling(window=20).std()

    df["abnormal_return"] = (
        df["return"] - df["rolling_return_mean_20"]
    ) / df["rolling_return_std_20"]

    df["price_moving_average_20"] = df["Close"].rolling(window=20).mean()

    df["price_moving_std_20"] = df["Close"].rolling(window=20).std()

    df["price_zscore"] = (
        df["Close"] - df["price_moving_average_20"]
    ) / df["price_moving_std_20"]

    df["high_low_range"] = (df["High"] - df["Low"]) / df["Close"]

    df["open_close_body"] = (df["Close"] - df["Open"]) / df["Open"]

    # =========================
    # 2. Features de volatilité
    # =========================

    df["realized_volatility_5"] = df["return"].rolling(window=5).std()

    df["realized_volatility_20"] = df["return"].rolling(window=20).std()

    df["volatility_ratio"] = (
        df["realized_volatility_5"] / df["realized_volatility_20"]
    )

    # =========================
    # 3. Features de volume
    # =========================

    df["volume_moving_average_20"] = df["Volume"].rolling(window=20).mean()

    df["volume_moving_std_20"] = df["Volume"].rolling(window=20).std()

    df["abnormal_volume"] = df["Volume"] / df["volume_moving_average_20"]

    df["volume_zscore"] = (
        df["Volume"] - df["volume_moving_average_20"]
    ) / df["volume_moving_std_20"]

    df["volume_change"] = df["Volume"].pct_change()

    # =========================
    # 4. Features d'interaction
    # =========================

    df["return_volume_interaction"] = (
        df["abnormal_return"].abs() * df["abnormal_volume"]
    )

    df["volatility_volume_interaction"] = (
        df["realized_volatility_20"] * df["abnormal_volume"]
    )

    # =========================
    # 5. Momentum / reversal
    # =========================

    df["momentum_5"] = df["Close"] / df["Close"].shift(5) - 1

    df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1

    df["price_reversal"] = -df["return"].shift(1) * df["return"]

    # =========================
    # 6. Identification actif
    # =========================

    df["ticker"] = ticker

    # Nettoyage final
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df


def process_all_files():
    all_data = []

    csv_files = list(RAW_DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("Aucun fichier trouvé dans data/raw.")

    for file_path in csv_files:
        ticker = file_path.stem

        print(f"\nTraitement de {ticker}...")

        df = clean_yfinance_file(file_path)

        print("Types après nettoyage :")
        print(df[["Open", "High", "Low", "Close", "Volume"]].dtypes)

        featured_df = build_features(df, ticker)

        output_path = PROCESSED_DATA_DIR / f"{ticker}_features.csv"
        featured_df.to_csv(output_path, index=False)

        all_data.append(featured_df)

        print(f"Features sauvegardées : {output_path}")
        print(f"Shape : {featured_df.shape}")

    final_df = pd.concat(all_data, axis=0, ignore_index=True)

    final_output_path = PROCESSED_DATA_DIR / "market_features_all.csv"
    final_df.to_csv(final_output_path, index=False)

    print("\nDataset final sauvegardé :", final_output_path)
    print("Shape finale :", final_df.shape)
    print("\nColonnes finales :")
    print(final_df.columns.tolist())


if __name__ == "__main__":
    process_all_files()