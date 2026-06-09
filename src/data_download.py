import yfinance as yf
import pandas as pd
from pathlib import Path


RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_market_data(
    tickers,
    start_date="2019-01-01",
    end_date="2023-12-31",
    interval="1d"
):
    """
    Télécharge les données boursières depuis Yahoo Finance.

    Paramètres :
    - tickers : liste des actifs financiers
    - start_date : date de début
    - end_date : date de fin
    - interval : fréquence des données

    Retour :
    - sauvegarde un fichier CSV par ticker
    """

    for ticker in tickers:
        print(f"Téléchargement de {ticker}...")

        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            print(f"Aucune donnée trouvée pour {ticker}")
            continue

        df.reset_index(inplace=True)

        file_name = ticker.replace("^", "index_").replace("/", "_")
        output_path = RAW_DATA_DIR / f"{file_name}.csv"

        df.to_csv(output_path, index=False)

        print(f"Données sauvegardées : {output_path}")


if __name__ == "__main__":
    tickers = ["GME", "AMC", "AAPL", "MSFT", "^FCHI"]

    download_market_data(
        tickers=tickers,
        start_date="2019-01-01",
        end_date="2023-12-31",
        interval="1d"
    )