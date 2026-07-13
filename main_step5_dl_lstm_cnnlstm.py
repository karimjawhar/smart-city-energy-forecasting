from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

# Force non-GUI backend to avoid Tkinter/thread crashes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ================= CONFIG =================
SEQ_LEN = 24
TARGET = "Global_active_power"

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH   = "data/processed/val.csv"
TEST_PATH  = "data/processed/test.csv"

BATCH_SIZE = 256
EPOCHS = 20
PATIENCE = 3
LR = 1e-3
SEED = 42

OUT_TABLES = "outputs/tables"
OUT_FIGS = "outputs/figures"
# =========================================


def ensure_dirs() -> None:
    os.makedirs(OUT_TABLES, exist_ok=True)
    os.makedirs(OUT_FIGS, exist_ok=True)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape(y_true, y_pred),
        "R2": r2_score_np(y_true, y_pred),
    }


def read_split_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df = df.sort_index()
    df = df.asfreq("h")
    if df.isna().any().any():
        raise ValueError(
            f"Split file {path} is not a regular hourly time series (asfreq('h') introduced NaNs)."
        )
    return df


def make_sequences(df: pd.DataFrame, target_col: str, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequences using ALL features (everything except target).
    Input:  X[t-seq_len : t, :]
    Output: y[t]   (next-hour target at time t)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    X_df = df.drop(columns=[target_col]).astype(float)
    y_s  = df[target_col].astype(float)

    X = X_df.to_numpy()
    y = y_s.to_numpy()

    X_seq = []
    y_seq = []

    for t in range(seq_len, len(df)):
        X_seq.append(X[t - seq_len : t, :])
        y_seq.append(y[t])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def standardize_on_train(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize using training statistics ONLY (no leakage).
    X shapes: (samples, seq_len, n_features)
    """
    # compute mu/sigma over (samples, time) for each feature
    mu = np.mean(X_train, axis=(0, 1), keepdims=True)      # shape (1,1,n_features)
    sigma = np.std(X_train, axis=(0, 1), keepdims=True)    # shape (1,1,n_features)
    sigma = np.where(sigma == 0.0, 1.0, sigma)

    X_train_s = (X_train - mu) / sigma
    X_val_s   = (X_val   - mu) / sigma
    X_test_s  = (X_test  - mu) / sigma
    return X_train_s, X_val_s, X_test_s, mu, sigma


def build_lstm(input_shape: Tuple[int, ...]) -> models.Sequential:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=LR), loss="mse")
    return model


def build_cnn_lstm(input_shape: Tuple[int, ...]) -> models.Sequential:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=3, activation="relu", padding="causal"),
        layers.MaxPooling1D(2),
        layers.LSTM(64),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=LR), loss="mse")
    return model


def plot_history(history, out_path: str, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_predictions(
    index: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    tail_n = min(7 * 24, len(index))
    plt.figure(figsize=(12, 4))
    plt.plot(index[-tail_n:], y_true[-tail_n:], label="Actual", linewidth=1)
    plt.plot(index[-tail_n:], y_pred[-tail_n:], label="Predicted", linewidth=1)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(TARGET)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ensure_dirs()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    train = read_split_csv(TRAIN_PATH)
    val   = read_split_csv(VAL_PATH)
    test  = read_split_csv(TEST_PATH)

    print("Loaded splits:")
    print("Train:", train.shape, train.index.min(), "to", train.index.max())
    print("Val:  ", val.shape, val.index.min(), "to", val.index.max())
    print("Test: ", test.shape, test.index.min(), "to", test.index.max())

    # Build sequences (no leakage; uses all features)
    X_train_seq, y_train_seq = make_sequences(train, TARGET, SEQ_LEN)
    X_val_seq,   y_val_seq   = make_sequences(val, TARGET, SEQ_LEN)
    X_test_seq,  y_test_seq  = make_sequences(test, TARGET, SEQ_LEN)

    print("\nSequence shapes:")
    print(" - X_train:", X_train_seq.shape, "y_train:", y_train_seq.shape)
    print(" - X_val:  ", X_val_seq.shape,   "y_val:  ", y_val_seq.shape)
    print(" - X_test: ", X_test_seq.shape,  "y_test: ", y_test_seq.shape)

    # Standardize using train only (prevents leakage)
    X_train_s, X_val_s, X_test_s, mu, sigma = standardize_on_train(X_train_seq, X_val_seq, X_test_seq)

    input_shape = (SEQ_LEN, X_train_s.shape[2])

    es = callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    test_index_seq = test.index[SEQ_LEN:]  # align with y_test_seq
    preds_df = pd.DataFrame({"y_true": y_test_seq}, index=test_index_seq)

    # ===== LSTM =====
    lstm = build_lstm(input_shape)
    t0 = time.time()
    hist_lstm = lstm.fit(
        X_train_s, y_train_seq,
        validation_data=(X_val_s, y_val_seq),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=1,
    )
    train_time = time.time() - t0

    t1 = time.time()
    lstm_pred = lstm.predict(X_test_s, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    pred_time = time.time() - t1

    lstm_metrics = eval_metrics(y_test_seq, lstm_pred)
    print("\nLSTM metrics:", lstm_metrics)

    results.append({
        "Model": f"LSTM(seq_len={SEQ_LEN})",
        **lstm_metrics,
        "TrainTimeSec": train_time,
        "PredictTimeSec": pred_time,
    })
    preds_df["y_lstm"] = lstm_pred

    plot_history(hist_lstm, f"{OUT_FIGS}/train_history_lstm_{ts}.png", "LSTM Training Loss")
    plot_predictions(test_index_seq, y_test_seq, lstm_pred, f"{OUT_FIGS}/test_plot_lstm_{ts}.png",
                     "LSTM: Actual vs Predicted (Test, last 7 days)")

    # ===== CNN-LSTM =====
    cnn_lstm = build_cnn_lstm(input_shape)
    t2 = time.time()
    hist_cnn = cnn_lstm.fit(
        X_train_s, y_train_seq,
        validation_data=(X_val_s, y_val_seq),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=1,
    )
    train_time_cnn = time.time() - t2

    t3 = time.time()
    cnn_pred = cnn_lstm.predict(X_test_s, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    pred_time_cnn = time.time() - t3

    cnn_metrics = eval_metrics(y_test_seq, cnn_pred)
    print("\nCNN-LSTM metrics:", cnn_metrics)

    results.append({
        "Model": f"CNN-LSTM(seq_len={SEQ_LEN})",
        **cnn_metrics,
        "TrainTimeSec": train_time_cnn,
        "PredictTimeSec": pred_time_cnn,
    })
    preds_df["y_cnnlstm"] = cnn_pred

    plot_history(hist_cnn, f"{OUT_FIGS}/train_history_cnnlstm_{ts}.png", "CNN-LSTM Training Loss")
    plot_predictions(test_index_seq, y_test_seq, cnn_pred, f"{OUT_FIGS}/test_plot_cnnlstm_{ts}.png",
                     "CNN-LSTM: Actual vs Predicted (Test, last 7 days)")

    # ===== Save outputs =====
    metrics_df = pd.DataFrame(results)
    metrics_path = f"{OUT_TABLES}/dl_metrics_{ts}.csv"
    preds_path   = f"{OUT_TABLES}/dl_predictions_{ts}.csv"

    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path)

    print("\nSaved:")
    print(" -", metrics_path)
    print(" -", preds_path)
    print(" -", f"{OUT_FIGS}/train_history_lstm_{ts}.png")
    print(" -", f"{OUT_FIGS}/train_history_cnnlstm_{ts}.png")
    print(" -", f"{OUT_FIGS}/test_plot_lstm_{ts}.png")
    print(" -", f"{OUT_FIGS}/test_plot_cnnlstm_{ts}.png")


if __name__ == "__main__":
    main()