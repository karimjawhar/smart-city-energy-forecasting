from __future__ import annotations

# IMPORTANT: Force non-GUI backend to avoid Tkinter/thread crashes in debugger
import matplotlib
matplotlib.use("Agg")

import os
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ================= CONFIG =================
PROCESSED_PATH = "data/processed/ucihourly_features.csv"
TARGET = "Global_active_power"

# Must match your Step 1 boundaries
TRAIN_END = "2009-09-20 19:00:00"
VAL_END   = "2010-04-24 20:00:00"

# Your pandas build expects lowercase frequency
RESAMPLE_FREQ = "h"

# Output folders
TABLES_DIR = "outputs/tables"
FIG_DIR = "outputs/figures"
# =========================================


def ensure_dirs() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def eval_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


def split_by_dates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split:
      Train: <= TRAIN_END
      Val:   TRAIN_END+1h .. VAL_END
      Test:  VAL_END+1h .. end
    """
    train = df.loc[:TRAIN_END].copy()
    val = df.loc[pd.to_datetime(TRAIN_END) + pd.Timedelta(hours=1):VAL_END].copy()
    test = df.loc[pd.to_datetime(VAL_END) + pd.Timedelta(hours=1):].copy()
    return train, val, test


def make_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)
    return X, y


def plot_actual_vs_pred(test_index, y_true, y_pred, title: str, out_path: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(test_index, y_true, label="Actual")
    plt.plot(test_index, y_pred, label="Predicted")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(TARGET)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def load_processed() -> pd.DataFrame:
    """
    Loads processed dataset saved from Step 1.
    Ensures sorted datetime index and explicit hourly frequency.
    """
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_PATH}\n"
            "Run main_step1.py first to generate it."
        )

    df = pd.read_csv(PROCESSED_PATH, index_col=0, parse_dates=True)
    df.index.name = "datetime"
    df = df.sort_index()

    # Force explicit hourly frequency; fill any introduced gaps
    df = df.asfreq(RESAMPLE_FREQ)
    if df.isna().any().any():
        df = df.interpolate("time").ffill().bfill()

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in processed dataset.")

    return df


def safe_save_csv(df: pd.DataFrame, base_name: str) -> str:
    """
    Saves to outputs/tables with a timestamped filename so Windows/Excel file locks
    never block writing.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(TABLES_DIR, f"{base_name}_{ts}.csv")
    df.to_csv(path, index=False)
    return path


def safe_save_csv_indexed(df: pd.DataFrame, base_name: str) -> str:
    """
    Same as safe_save_csv but keeps the index (useful for predictions with datetime index).
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(TABLES_DIR, f"{base_name}_{ts}.csv")
    df.to_csv(path)  # keep datetime index
    return path


def main():
    ensure_dirs()

    df = load_processed()
    train, val, test = split_by_dates(df)

    print("Train:", train.shape, train.index.min(), "to", train.index.max())
    print("Val:  ", val.shape, val.index.min(), "to", val.index.max())
    print("Test: ", test.shape, test.index.min(), "to", test.index.max())

    X_train, y_train = make_xy(train)
    X_test, y_test = make_xy(test)

    results: list[dict] = []
    preds_out = pd.DataFrame({"y_true": y_test}, index=test.index)

    # ===== Ridge =====
    ridge = Ridge(alpha=1.0)
    t0 = time.time()
    ridge.fit(X_train, y_train)
    ridge_train_time = time.time() - t0

    t1 = time.time()
    ridge_pred = ridge.predict(X_test)
    ridge_pred_time = time.time() - t1

    ridge_metrics = eval_metrics(y_test, ridge_pred)
    print("\nRidge metrics:", ridge_metrics)

    results.append({
        "Model": "Ridge(alpha=1.0)",
        **ridge_metrics,
        "TrainTimeSec": ridge_train_time,
        "PredictTimeSec": ridge_pred_time,
    })
    preds_out["y_ridge"] = ridge_pred

    plot_actual_vs_pred(
        test.index, y_test.values, ridge_pred,
        "Ridge: Actual vs Predicted (Test)",
        os.path.join(FIG_DIR, "test_plot_ridge.png"),
    )

    # ===== Random Forest =====
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    t1 = time.time()
    rf_pred = rf.predict(X_test)
    rf_pred_time = time.time() - t1

    rf_metrics = eval_metrics(y_test, rf_pred)
    print("\nRandom Forest metrics:", rf_metrics)

    results.append({
        "Model": "RandomForest(n_estimators=300)",
        **rf_metrics,
        "TrainTimeSec": rf_train_time,
        "PredictTimeSec": rf_pred_time,
    })
    preds_out["y_rf"] = rf_pred

    plot_actual_vs_pred(
        test.index, y_test.values, rf_pred,
        "Random Forest: Actual vs Predicted (Test)",
        os.path.join(FIG_DIR, "test_plot_rf.png"),
    )

    # ===== Save outputs safely (timestamped) =====
    res_df = pd.DataFrame(results)

    metrics_path = safe_save_csv(res_df, "ml_metrics")
    preds_path = safe_save_csv_indexed(preds_out, "ml_predictions")

    print("\nSaved:")
    print(" -", metrics_path)
    print(" -", preds_path)
    print(" -", os.path.join(FIG_DIR, "test_plot_ridge.png"))
    print(" -", os.path.join(FIG_DIR, "test_plot_rf.png"))


if __name__ == "__main__":
    main()
