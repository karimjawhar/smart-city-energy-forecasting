from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


TARGET = "Global_active_power"

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"
TEST_PATH = "data/processed/test.csv"


@dataclass
class Metrics:
    mae: float
    rmse: float
    mape: float
    r2: float


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def eval_metrics(y_true: pd.Series, y_pred: pd.Series) -> Metrics:
    yt, yp = y_true.align(y_pred, join="inner")
    y_true_np = yt.to_numpy(dtype=float)
    y_pred_np = yp.to_numpy(dtype=float)

    err = y_pred_np - y_true_np
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    return Metrics(
        mae=mae,
        rmse=rmse,
        mape=mape(y_true_np, y_pred_np),
        r2=r2_score_np(y_true_np, y_pred_np),
    )


def read_split_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing split file: {path}. Run Step 1 first (python main_step1.py)."
        )

    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df = df.sort_index()
    df = df.asfreq("h")
    if df.isna().any().any():
        raise ValueError(
            f"Split file {path} is not a regular hourly time series (asfreq('h') introduced NaNs)."
        )
    return df


def naive_forecast(full_series: pd.Series) -> pd.Series:
    return full_series.shift(1)


def fit_arima_and_forecast(
    train_y: pd.Series,
    horizon_index: pd.DatetimeIndex,
    order: tuple[int, int, int],
) -> pd.Series:
    from statsmodels.tsa.arima.model import ARIMA

    model = ARIMA(train_y.astype(float), order=order)
    fitted = model.fit()

    fc = fitted.forecast(steps=len(horizon_index))
    fc.index = horizon_index
    return fc


def try_arima_baseline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
) -> Optional[dict]:
    try:
        import statsmodels  # noqa: F401
    except Exception:
        print("statsmodels not available; skipping ARIMA baseline")
        return None

    candidate_orders = [(1, 0, 1), (2, 0, 2), (3, 0, 3), (2, 0, 1), (1, 0, 2)]
    best_order: Optional[tuple[int, int, int]] = None
    best_rmse = float("inf")

    for order in candidate_orders:
        try:
            val_pred = fit_arima_and_forecast(train[target], val.index, order=order)
            met = eval_metrics(val[target], val_pred)
            if met.rmse < best_rmse:
                best_rmse = met.rmse
                best_order = order
        except Exception:
            continue

    if best_order is None:
        print("ARIMA tuning failed; skipping ARIMA baseline")
        return None

    print(f"Best ARIMA order from VAL: {best_order} (VAL RMSE={best_rmse:.4f})")

    t0 = time.time()
    trainval_y = pd.concat([train[target], val[target]])
    test_pred = fit_arima_and_forecast(trainval_y, test.index, order=best_order)
    train_time = time.time() - t0

    met = eval_metrics(test[target], test_pred)

    out_pred = pd.DataFrame({"y_true": test[target], "y_pred": test_pred}, index=test.index)
    out_pred.to_csv("outputs/tables/arima_test_predictions.csv")

    return {
        "Model": f"ARIMA{best_order}",
        "MAE": met.mae,
        "RMSE": met.rmse,
        "MAPE": met.mape,
        "R2": met.r2,
        "TrainTimeSec": train_time,
        "PredictTimeSec": 0.0,
    }


def main() -> None:
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    train = read_split_csv(TRAIN_PATH)
    val = read_split_csv(VAL_PATH)
    test = read_split_csv(TEST_PATH)

    print("Loaded splits:")
    print("Train:", train.shape, train.index.min(), "to", train.index.max())
    print("Val:  ", val.shape, val.index.min(), "to", val.index.max())
    print("Test: ", test.shape, test.index.min(), "to", test.index.max())

    if TARGET not in train.columns:
        raise KeyError(f"Target column '{TARGET}' not in train.csv columns: {list(train.columns)}")

    results = []

    # ===== Naive baseline (t-1) =====
    full = pd.concat([train, val, test]).sort_index()
    t0 = time.time()
    naive_pred_full = naive_forecast(full[TARGET])
    naive_test_pred = naive_pred_full.reindex(test.index)
    naive_test_true = test[TARGET]

    met = eval_metrics(naive_test_true, naive_test_pred)
    results.append(
        {
            "Model": "Naive(t-1)",
            "MAE": met.mae,
            "RMSE": met.rmse,
            "MAPE": met.mape,
            "R2": met.r2,
            "TrainTimeSec": 0.0,
            "PredictTimeSec": time.time() - t0,
        }
    )

    out_pred = pd.DataFrame(
        {"y_true": naive_test_true, "y_pred": naive_test_pred}, index=test.index
    )
    out_pred.to_csv("outputs/tables/naive_test_predictions.csv")

    # ===== ARIMA baseline (optional) =====
    arima_row = try_arima_baseline(train, val, test, target=TARGET)
    if arima_row is not None:
        results.append(arima_row)

    res_df = pd.DataFrame(results)
    res_df.to_csv("outputs/tables/baseline_metrics.csv", index=False)
    print("Saved: outputs/tables/baseline_metrics.csv")


if __name__ == "__main__":
    main()
