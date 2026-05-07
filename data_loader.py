import pandas as pd
import os

DATASET_PATH = r"E:\H.gibberish\Phishhook1\phishing_dataset_total.csv"


def generate_dataset():
    print("[data_loader] Using real dataset instead of synthetic generator.")
    return load_dataset()


def load_dataset(path=DATASET_PATH):

    print(f"[data_loader] Loading dataset from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    print("\nOriginal Label distribution:")
    print(df["label"].value_counts())

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["label"] = df["label"].astype(int)

    # Fix missing timestamps instead of dropping phishing rows
    missing_ts = df["timestamp"].isna().sum()
    if missing_ts > 0:
        print(f"[data_loader] Filling {missing_ts} missing timestamps...")
        df["timestamp"] = df["timestamp"].fillna(pd.Timestamp("2022-01-01"))

    # Drop rows only if URL is missing
    df = df.dropna(subset=["url"])

    df = df.reset_index(drop=True)

    # Optional: balance dataset (recommended for ML)
    legit = df[df["label"] == 0]
    phishing = df[df["label"] == 1]

    n = min(len(legit), len(phishing))

    legit_sample = legit.sample(n, random_state=42)
    phishing_sample = phishing.sample(n, random_state=42)

    df = pd.concat([legit_sample, phishing_sample]).reset_index(drop=True)

    # Final stats
    phishing_count = int((df["label"] == 1).sum())
    legit_count = int((df["label"] == 0).sum())

    print(
        f"[data_loader] Loaded {len(df)} balanced samples | "
        f"phishing={phishing_count} legit={legit_count}"
    )

    return df
