# Step 1: Load + clean UCI dataset + resample hourly + features + time split
# Save as: main_step1_data_pipeline.py

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_uci_household_power(path: str) -> pd.DataFrame:
    """
    Loads the UCI Individual Household Electric Power Consumption dataset.
    Expected raw format: columns include Date, Time, and numeric variables.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?", "NA", ""],
        low_memory=False,
    )

    if "Date" not in df.columns or "Time" not in df.columns:
        raise ValueError("Expected columns 'Date' and 'Time' not found in dataset.")

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


def resample_hourly(df: pd.DataFrame, target_col: str = "Global_active_power") -> pd.DataFrame:
    """Resample to hourly frequency. Uses mean aggregation."""
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not in dataset columns: {list(df.columns)}"
        )

    keep_cols = [c for c in df.columns if c == target_col]
    hourly = df[keep_cols].resample("h").mean()

    return hourly


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in a transparent, defensible way.
    For hourly series: time interpolation is common and reasonable.
    """
    before = df.isna().sum().to_dict()

    df_clean = df.copy()
    df_clean = df_clean.interpolate(method="time")
    df_clean = df_clean.ffill().bfill()

    after = df_clean.isna().sum().to_dict()

    print("\nMissing values (before):", before)
    print("Missing values (after): ", after)

    return df_clean


def add_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Minimal, defensible features:
    - lags: 1, 2, 24 hours
    - rolling mean: 24 hours
    - calendar: hour, dayofweek, month
    """
    out = df.copy()

    out["lag_1"] = out[target_col].shift(1)
    out["lag_2"] = out[target_col].shift(2)
    out["lag_24"] = out[target_col].shift(24)

    out["roll_mean_24"] = out[target_col].shift(1).rolling(window=24).mean()

    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["month"] = out.index.month

    out = out.dropna()

    return out


def time_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> SplitData:
    """Chronological split: train -> val -> test (no shuffling)."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return SplitData(train=train, val=val, test=test)


def main():
    raw_path = os.path.join("data", "raw", "household_power_consumption.txt")
    target = "Global_active_power"

    print("Loading raw dataset...")
    raw = load_uci_household_power(raw_path)

    print("\nRaw dataset:")
    print(" - rows:", len(raw))
    print(" - date range:", raw.index.min(), "to", raw.index.max())
    print(" - columns:", list(raw.columns))

    print("\nResampling to hourly...")
    hourly = resample_hourly(raw, target_col=target)

    print("\nHourly dataset (pre-clean):")
    print(" - rows:", len(hourly))
    print(" - date range:", hourly.index.min(), "to", hourly.index.max())

    hourly = clean_missing(hourly)

    print("\nAdding features...")
    feat = add_features(hourly, target_col=target)

    print("\nFeature dataset:")
    print(" - rows:", len(feat))
    print(" - columns:", list(feat.columns))
    print(" - date range:", feat.index.min(), "to", feat.index.max())

    print("\nSplitting train/val/test (70/15/15)...")
    split = time_split(feat, train_ratio=0.70, val_ratio=0.15)

    def describe_part(name: str, part: pd.DataFrame):
        print(f"\n{name}:")
        print(" - rows:", len(part))
        print(" - date range:", part.index.min(), "to", part.index.max())

    describe_part("TRAIN", split.train)
    describe_part("VAL", split.val)
    describe_part("TEST", split.test)

    os.makedirs(os.path.join("data", "processed"), exist_ok=True)
    processed_path = os.path.join("data", "processed", "ucihourly_features.csv")
    feat.to_csv(processed_path)
    print(f"\nSaved processed dataset to: {processed_path}")

    try:
        import matplotlib.pyplot as plt

        plt.figure()
        feat[target].plot()
        plt.title("Hourly Global_active_power (processed)")
        plt.xlabel("Time")
        plt.ylabel(target)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("\nPlot skipped (matplotlib not available or display issue):", e)


if __name__ == "__main__":
    main()
