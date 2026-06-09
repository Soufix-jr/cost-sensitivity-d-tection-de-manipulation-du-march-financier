import pandas as pd
from pathlib import Path


PROCESSED_DATA_DIR = Path("data/processed")

INPUT_FILE = PROCESSED_DATA_DIR / "market_labeled.csv"

TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_CALIB_FILE = PROCESSED_DATA_DIR / "val_calib.csv"
VAL_THRESHOLD_FILE = PROCESSED_DATA_DIR / "val_threshold.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"


def temporal_split(df: pd.DataFrame):
    """
    Split temporel strict :
    - 60% train
    - 10% validation calibration
    - 10% validation threshold
    - 20% test
    """

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Date", "ticker"]).reset_index(drop=True)

    n = len(df)

    train_end = int(0.60 * n)
    val_calib_end = int(0.70 * n)
    val_threshold_end = int(0.80 * n)

    train_df = df.iloc[:train_end].copy()
    val_calib_df = df.iloc[train_end:val_calib_end].copy()
    val_threshold_df = df.iloc[val_calib_end:val_threshold_end].copy()
    test_df = df.iloc[val_threshold_end:].copy()

    return train_df, val_calib_df, val_threshold_df, test_df


def print_split_report(name: str, df: pd.DataFrame):
    total = len(df)
    positives = int(df["label"].sum())
    negatives = total - positives
    rate = positives / total if total > 0 else 0

    print(f"\n{name}")
    print("-" * 40)
    print(f"Nombre d'observations : {total}")
    print(f"Classe 0              : {negatives}")
    print(f"Classe 1              : {positives}")
    print(f"Taux positif          : {rate:.4%}")
    print(f"Date min              : {df['Date'].min()}")
    print(f"Date max              : {df['Date'].max()}")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_FILE}. "
            "Lance d'abord src/label_generation.py"
        )

    df = pd.read_csv(INPUT_FILE)

    train_df, val_calib_df, val_threshold_df, test_df = temporal_split(df)

    train_df.to_csv(TRAIN_FILE, index=False)
    val_calib_df.to_csv(VAL_CALIB_FILE, index=False)
    val_threshold_df.to_csv(VAL_THRESHOLD_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("\n==============================")
    print("RAPPORT DE SPLIT TEMPOREL")
    print("==============================")

    print_split_report("TRAIN", train_df)
    print_split_report("VALIDATION CALIBRATION", val_calib_df)
    print_split_report("VALIDATION THRESHOLD", val_threshold_df)
    print_split_report("TEST", test_df)

    print("\nFichiers sauvegardés :")
    print(TRAIN_FILE)
    print(VAL_CALIB_FILE)
    print(VAL_THRESHOLD_FILE)
    print(TEST_FILE)


if __name__ == "__main__":
    main()