# Step 1: Load + clean UCI dataset + resample hourly + features + time split
# Runs with: python main_step1.py

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


# ===== Project Configuration =====
RAW_PATH_CANDIDATES = [
    "data/raw/household_power_consumption.txt",
    "data/raw/household_power_consumption",
]
TARGET_COL = "Global_active_power"
RESAMPLE_RULE = "h"  # hourly

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

LAGS = [1, 2, 24]
ROLLING_WINDOW = 24
# ================================


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def find_raw_path() -> str:
    for p in RAW_PATH_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find the UCI dataset file. Expected one of:\n"
        + "\n".join(RAW_PATH_CANDIDATES)
        + "\n\nPut the file into data/raw/ and keep the name as household_power_consumption(.txt)."
    )


def load_uci(path: str) -> pd.DataFrame:
    """
    Loads UCI Individual Household Electric Power Consumption dataset.
    Expected separator ';' and missing values marked as '?'.
    """
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?", "NA", ""],
        low_memory=False,
    )

    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError(
            "Expected columns 'Date' and 'Time' were not found in the raw dataset."
        )

    dt = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )
    df = df.drop(columns=["Date", "Time"])
    df.insert(0, "datetime", dt)

    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def resample_hourly(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found. Available columns: {list(df.columns)}")

    hourly = df[[target_col]].resample(RESAMPLE_RULE).mean()
    return hourly


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = df.isna().sum().to_dict()

    out = df.copy()
    out = out.interpolate(method="time")
    out = out.ffill().bfill()

    after = out.isna().sum().to_dict()

    print("\nMissing values (before):", before)
    print("Missing values (after): ", after)

    return out


def add_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()

    for lag in LAGS:
        out[f"lag_{lag}"] = out[target_col].shift(lag)

    out[f"roll_mean_{ROLLING_WINDOW}"] = (
        out[target_col].shift(1).rolling(window=ROLLING_WINDOW).mean()
    )

    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month

    out = out.dropna()

    return out


def time_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> SplitData:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return SplitData(train=train, val=val, test=test)


def ensure_dirs() -> None:
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)


def save_splits(split: SplitData) -> None:
    split.train.to_csv("data/processed/train.csv")
    split.val.to_csv("data/processed/val.csv")
    split.test.to_csv("data/processed/test.csv")


def plot_target_with_splits(full_df: pd.DataFrame, split: SplitData, target_col: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(full_df.index, full_df[target_col])

    if len(split.train) > 0:
        plt.axvline(split.train.index[-1])
    if len(split.val) > 0:
        plt.axvline(split.val.index[-1])

    plt.title("Hourly Global_active_power with Train/Val/Test Split")
    plt.xlabel("Time")
    plt.ylabel(target_col)
    plt.tight_layout()
    plt.savefig("outputs/figures/target_split.png", dpi=200)
    plt.close()


def describe_part(name: str, part: pd.DataFrame) -> None:
    print(f"\n{name}:")
    print(" - rows:", len(part))
    print(" - date range:", part.index.min(), "to", part.index.max())


def main() -> None:
    ensure_dirs()

    raw_path = find_raw_path()
    print("Using raw dataset file:", raw_path)

    print("\nLoading raw dataset...")
    raw = load_uci(raw_path)

    print("\nRaw dataset evidence:")
    print(" - rows:", len(raw))
    print(" - date range:", raw.index.min(), "to", raw.index.max())
    print(" - columns:", list(raw.columns))

    print("\nResampling to hourly...")
    hourly = resample_hourly(raw, TARGET_COL)

    print("\nHourly dataset (pre-clean):")
    print(" - rows:", len(hourly))
    print(" - date range:", hourly.index.min(), "to", hourly.index.max())

    hourly = clean_missing(hourly)

    print("\nAdding supervised features...")
    feat = add_features(hourly, TARGET_COL)

    print("\nFeature dataset evidence:")
    print(" - rows:", len(feat))
    print(" - columns:", list(feat.columns))
    print(" - date range:", feat.index.min(), "to", feat.index.max())

    print("\nSplitting chronologically (70/15/15, no shuffling)...")
    split = time_split(feat, TRAIN_RATIO, VAL_RATIO)

    describe_part("TRAIN", split.train)
    describe_part("VAL", split.val)
    describe_part("TEST", split.test)

    feat.to_csv("data/processed/ucihourly_features.csv")
    save_splits(split)

    plot_target_with_splits(feat, split, TARGET_COL)

    print("\nSaved:")
    print(" - data/processed/ucihourly_features.csv")
    print(" - data/processed/train.csv, val.csv, test.csv")
    print(" - outputs/figures/target_split.png")


if __name__ == "__main__":
    main()
