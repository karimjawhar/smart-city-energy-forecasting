from __future__ import annotations

import os
import time
from dataclasses import dataclass

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

    return Metrics(mae=mae, rmse=rmse, mape=mape(y_true_np, y_pred_np), r2=r2_score_np(y_true_np, y_pred_np))


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


def make_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not found. Columns: {list(df.columns)}")

    X_df = df.drop(columns=[TARGET]).copy()

    for c in X_df.columns:
        if X_df[c].dtype.kind not in "if":
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    X_df = X_df.astype(float)

    y = df[TARGET].astype(float).to_numpy()
    return X_df.to_numpy(), y, list(X_df.columns)


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    Xs = (X - mu) / sigma
    return Xs, mu, sigma


def standardize_apply(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    n, p = X.shape

    ones = np.ones((n, 1))
    Xb = np.hstack([ones, X])

    I = np.eye(p + 1)
    I[0, 0] = 0.0

    A = Xb.T @ Xb + alpha * I
    b = Xb.T @ y

    w = np.linalg.solve(A, b)
    intercept = float(w[0])
    coef = w[1:]

    return coef, intercept


def predict_linear(X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    return X @ coef + intercept


def plot_actual_vs_pred(index: pd.DatetimeIndex, y_true: pd.Series, y_pred: pd.Series, out_path: str, title: str) -> None:
    import matplotlib.pyplot as plt

    tail_n = min(7 * 24, len(index))
    y_true_tail = y_true.tail(tail_n)
    y_pred_tail = y_pred.tail(tail_n)

    plt.figure(figsize=(12, 4))
    plt.plot(y_true_tail.index, y_true_tail.values, label="Actual", linewidth=1)
    plt.plot(y_pred_tail.index, y_pred_tail.values, label="Predicted", linewidth=1)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(TARGET)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    train_df = read_split_csv(TRAIN_PATH)
    val_df = read_split_csv(VAL_PATH)
    test_df = read_split_csv(TEST_PATH)

    print("Loaded splits:")
    print("Train:", train_df.shape, train_df.index.min(), "to", train_df.index.max())
    print("Val:  ", val_df.shape, val_df.index.min(), "to", val_df.index.max())
    print("Test: ", test_df.shape, test_df.index.min(), "to", test_df.index.max())

    X_train, y_train, feature_names = make_xy(train_df)
    X_val, y_val, _ = make_xy(val_df)
    X_test, y_test, _ = make_xy(test_df)

    X_train_s, mu, sigma = standardize_fit(X_train)
    X_val_s = standardize_apply(X_val, mu, sigma)
    X_test_s = standardize_apply(X_test, mu, sigma)

    alphas = [0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]

    best_alpha = None
    best_val_rmse = float("inf")

    for a in alphas:
        coef, intercept = fit_ridge_closed_form(X_train_s, y_train, alpha=a)
        val_pred = predict_linear(X_val_s, coef, intercept)
        rmse = float(np.sqrt(np.mean((val_pred - y_val) ** 2)))
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_alpha = a

    print(f"Best ridge alpha from VAL: {best_alpha} (VAL RMSE={best_val_rmse:.4f})")

    trainval_df = pd.concat([train_df, val_df]).sort_index()
    X_trainval, y_trainval, _ = make_xy(trainval_df)
    X_trainval_s, mu2, sigma2 = standardize_fit(X_trainval)
    X_test_s2 = standardize_apply(X_test, mu2, sigma2)

    t0 = time.time()
    coef, intercept = fit_ridge_closed_form(X_trainval_s, y_trainval, alpha=float(best_alpha))
    train_time = time.time() - t0

    t1 = time.time()
    test_pred_np = predict_linear(X_test_s2, coef, intercept)
    pred_time = time.time() - t1

    test_pred = pd.Series(test_pred_np, index=test_df.index, name="y_pred")

    met = eval_metrics(test_df[TARGET], test_pred)
    print("Test metrics:", {"MAE": met.mae, "RMSE": met.rmse, "MAPE": met.mape, "R2": met.r2})

    pd.DataFrame(
        [
            {
                "Model": f"Ridge(alpha={best_alpha})",
                "MAE": met.mae,
                "RMSE": met.rmse,
                "MAPE": met.mape,
                "R2": met.r2,
                "TrainTimeSec": train_time,
                "PredictTimeSec": pred_time,
            }
        ]
    ).to_csv("outputs/tables/ml_metrics.csv", index=False)

    out_pred = pd.DataFrame({"y_true": test_df[TARGET], "y_pred": test_pred}, index=test_df.index)
    out_pred.to_csv("outputs/tables/ml_predictions.csv")

    plot_actual_vs_pred(
        test_df.index,
        test_df[TARGET],
        test_pred,
        out_path="outputs/figures/test_plot_ridge.png",
        title="Ridge Regression: Actual vs Predicted (Test, last 7 days)",
    )

    coef_series = pd.Series(coef, index=feature_names).sort_values(key=np.abs, ascending=False)
    coef_series.to_csv("outputs/tables/ridge_coefficients.csv", header=["coef"])

    print("Saved:")
    print(" - outputs/tables/ml_metrics.csv")
    print(" - outputs/tables/ml_predictions.csv")
    print(" - outputs/tables/ridge_coefficients.csv")
    print(" - outputs/figures/test_plot_ridge.png")


if __name__ == "__main__":
    main()
