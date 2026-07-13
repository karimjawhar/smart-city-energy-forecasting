from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ================= CONFIG =================
RAW_PATH      = "data/raw/household_power_consumption.txt"
TARGET        = "Global_active_power"
TRAIN_END     = "2009-09-20 19:00:00"
VAL_END       = "2010-04-24 20:00:00"
RESAMPLE_FREQ = "h"
CORR_THRESHOLD = 0.97

SEQ_LEN    = 48       # 2-day context window
BATCH_SIZE = 512
EPOCHS     = 60
PATIENCE   = 10
LR         = 1e-3
SEED       = 42

OUT_TABLES = "outputs/tables"
OUT_FIGS   = "outputs/figures"
# =========================================


def ensure_dirs() -> None:
    os.makedirs(OUT_TABLES, exist_ok=True)
    os.makedirs(OUT_FIGS, exist_ok=True)


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(np.asarray(y_true, float)), eps)
    return float(np.mean(np.abs((np.asarray(y_true, float) - np.asarray(y_pred, float)) / denom)) * 100.0)


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    err    = yp - yt
    return {
        "MAE":  float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": mape(yt, yp),
        "R2":   r2_score_np(yt, yp),
    }


# ── Feature engineering (mirrors main_step3_xgb_optimized.py) ────
def load_raw_hourly() -> pd.DataFrame:
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw data not found: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, sep=";", na_values=["?", "NA", ""], low_memory=False)
    dt = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True, errors="coerce",
    )
    df = df.drop(columns=["Date", "Time"])
    df.insert(0, "datetime", dt)
    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    hourly = df.resample(RESAMPLE_FREQ).mean()
    hourly = hourly.interpolate("time").ffill().bfill()
    if all(c in hourly.columns for c in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]):
        sub_kwh = (hourly["Sub_metering_1"] + hourly["Sub_metering_2"] + hourly["Sub_metering_3"]) / 1000.0
        hourly["Sub_other"] = np.maximum(0.0, hourly[TARGET] - sub_kwh)
    return hourly


def build_rich_features(df: pd.DataFrame, target_col: str):
    y    = df[target_col].copy()
    y_s1 = y.shift(1)
    feats = pd.DataFrame(index=df.index)

    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]:
        feats[f"lag_{lag}"] = y.shift(lag)
    feats["diff_1"]   = y.shift(1) - y.shift(2)
    feats["diff_24"]  = y.shift(24) - y.shift(48)
    feats["diff_168"] = y.shift(168) - y.shift(336)
    for w in [3, 6, 12, 24, 168]:
        feats[f"roll_mean_{w}"] = y_s1.rolling(w).mean()
    for w in [6, 12, 24]:
        feats[f"roll_std_{w}"] = y_s1.rolling(w).std()
    feats["ema_12"] = y_s1.ewm(span=12, adjust=False).mean()
    feats["ema_24"] = y_s1.ewm(span=24, adjust=False).mean()
    lag24_safe  = y.shift(24).replace(0, np.nan)
    lag168_safe = y.shift(168).replace(0, np.nan)
    feats["lag1_over_lag24"]  = y.shift(1) / lag24_safe
    feats["lag1_over_lag168"] = y.shift(1) / lag168_safe

    for col in [c for c in df.columns if c != target_col]:
        s = df[col]
        for lag in [1, 24, 48, 168]:
            feats[f"{col}_lag{lag}"] = s.shift(lag)

    if "Sub_metering_3" in df.columns:
        sm3 = df["Sub_metering_3"].shift(1)
        feats["sm3_roll_mean_24"]  = sm3.rolling(24).mean()
        feats["sm3_roll_mean_168"] = sm3.rolling(168).mean()
        feats["sm3_roll_std_24"]   = sm3.rolling(24).std()
    if "Sub_metering_1" in df.columns:
        feats["sm1_roll_mean_24"] = df["Sub_metering_1"].shift(1).rolling(24).mean()
    if "Sub_metering_2" in df.columns:
        feats["sm2_roll_mean_24"] = df["Sub_metering_2"].shift(1).rolling(24).mean()
    if "Voltage" in df.columns:
        volt = df["Voltage"].shift(1)
        feats["volt_roll_mean_24"]  = volt.rolling(24).mean()
        feats["volt_roll_mean_168"] = volt.rolling(168).mean()
    if "Global_reactive_power" in df.columns:
        reac = df["Global_reactive_power"].shift(1)
        feats["reac_roll_mean_24"]  = reac.rolling(24).mean()
        feats["reac_roll_mean_168"] = reac.rolling(168).mean()

    idx = feats.index
    h_sin = np.sin(2 * np.pi * idx.hour  / 24)
    h_cos = np.cos(2 * np.pi * idx.hour  / 24)
    m_sin = np.sin(2 * np.pi * idx.month / 12)
    m_cos = np.cos(2 * np.pi * idx.month / 12)
    feats["hour_sin"]         = h_sin
    feats["hour_cos"]         = h_cos
    feats["dow_sin"]          = np.sin(2 * np.pi * idx.dayofweek / 7)
    feats["dow_cos"]          = np.cos(2 * np.pi * idx.dayofweek / 7)
    feats["month_sin"]        = m_sin
    feats["month_cos"]        = m_cos
    woy = idx.isocalendar().week.astype(int).to_numpy()
    feats["woy_sin"]          = np.sin(2 * np.pi * woy / 52)
    feats["woy_cos"]          = np.cos(2 * np.pi * woy / 52)
    feats["month_x_hour_sin"] = m_sin * h_sin
    feats["month_x_hour_cos"] = m_cos * h_cos

    out = feats.join(y.to_frame("y"), how="inner").dropna()
    return out.drop(columns=["y"]), out["y"]


_PROTECTED = {"lag_1", "lag_24", "lag_168", "lag_336", "hist_mean_hod",
              "dev_from_hod", "rel_dev_from_hod"}


def remove_correlated_features(X_train, X_val, X_test, threshold=CORR_THRESHOLD):
    corr  = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        for other in upper.index[upper[col] > threshold]:
            col_p, other_p = col in _PROTECTED, other in _PROTECTED
            if col_p and other_p:
                continue
            elif other_p:
                to_drop.add(col)
            elif col_p:
                to_drop.add(other)
            elif X_train[col].var() <= X_train[other].var():
                to_drop.add(col)
            else:
                to_drop.add(other)
    kept = [c for c in X_train.columns if c not in to_drop]
    print(f"  Feature selection: {len(X_train.columns)} -> {len(kept)} (dropped {len(to_drop)})")
    return X_train[kept], X_val[kept], X_test[kept], kept


def split_by_dates(df):
    train = df.loc[:TRAIN_END].copy()
    val   = df.loc[pd.to_datetime(TRAIN_END) + pd.Timedelta(hours=1):VAL_END].copy()
    test  = df.loc[pd.to_datetime(VAL_END)   + pd.Timedelta(hours=1):].copy()
    return train, val, test


def make_xy(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)
    return X, y


def make_sequences(X_arr: np.ndarray, y_arr: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for t in range(seq_len, len(y_arr)):
        Xs.append(X_arr[t - seq_len: t])
        ys.append(y_arr[t])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def standardize_sequences(X_tr, X_va, X_te):
    mu    = np.mean(X_tr, axis=(0, 1), keepdims=True)
    sigma = np.std(X_tr,  axis=(0, 1), keepdims=True)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    return (X_tr - mu) / sigma, (X_va - mu) / sigma, (X_te - mu) / sigma


# ── Model architectures ───────────────────────────────────────────
def build_bilstm(input_shape):
    """Bidirectional stacked LSTM with dropout."""
    inp = layers.Input(shape=input_shape)
    x   = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.LSTM(64)(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    mdl = models.Model(inp, out)
    mdl.compile(optimizer=optimizers.Adam(learning_rate=LR), loss="mse")
    return mdl


def build_cnn_lstm(input_shape):
    """Dual-Conv + LSTM with dropout."""
    inp = layers.Input(shape=input_shape)
    x   = layers.Conv1D(128, kernel_size=3, activation="relu", padding="causal")(inp)
    x   = layers.Conv1D(64,  kernel_size=3, activation="relu", padding="causal")(x)
    x   = layers.MaxPooling1D(2)(x)
    x   = layers.LSTM(64)(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    mdl = models.Model(inp, out)
    mdl.compile(optimizer=optimizers.Adam(learning_rate=LR), loss="mse")
    return mdl


def train_model(model, X_tr, y_tr, X_va, y_va, tag: str):
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE,
        restore_best_weights=True, verbose=1,
    )
    rlr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5,
        min_lr=1e-5, verbose=1,
    )
    t0  = time.time()
    hist = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es, rlr],
        verbose=1,
    )
    train_time = time.time() - t0

    # Training loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.title(f"{tag} Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_FIGS}/train_history_{tag.lower()}_{ts_str}.png", dpi=200)
    plt.close()

    return hist, train_time


def save_pred_plot(index, y_true, y_pred, tag: str, ts_str: str):
    tail_n = min(7 * 24, len(index))
    plt.figure(figsize=(12, 4))
    plt.plot(index[-tail_n:], y_true[-tail_n:], label="Actual", linewidth=1)
    plt.plot(index[-tail_n:], y_pred[-tail_n:], label="Predicted", linewidth=1, alpha=0.85)
    plt.title(f"{tag}: Actual vs Predicted (last 7 days of test)")
    plt.xlabel("Time")
    plt.ylabel(TARGET)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_FIGS}/test_plot_{tag.lower()}_{ts_str}.png", dpi=200)
    plt.close()


def main() -> None:
    print("=" * 62)
    print("Optimized LSTM + CNN-LSTM — Electricity Forecasting")
    print("=" * 62)

    ensure_dirs()
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Load & feature engineering ────────────────────────────
    print("\n1. Loading raw data & building features...")
    df = load_raw_hourly()
    X, y = build_rich_features(df, TARGET)
    df_feat = X.copy()
    df_feat[TARGET] = y

    # ── 2. Chronological split ────────────────────────────────────
    train, val, test = split_by_dates(df_feat)
    X_train, y_train = make_xy(train)
    X_val,   y_val   = make_xy(val)
    X_test,  y_test  = make_xy(test)
    print(f"  Train : {train.shape}  {train.index.min()} -> {train.index.max()}")
    print(f"  Val   : {val.shape}   {val.index.min()} -> {val.index.max()}")
    print(f"  Test  : {test.shape}  {test.index.min()} -> {test.index.max()}")

    # ── 3. Hist-mean hod + deviation features ────────────────────
    hod_mean = y_train.groupby([X_train.index.hour, X_train.index.dayofweek]).mean()
    hod_mean.index.names = ["hour", "dow"]

    def _add_hod(X):
        keys = pd.MultiIndex.from_arrays([X.index.hour, X.index.dayofweek],
                                         names=["hour", "dow"])
        vals = keys.map(hod_mean)
        return X.assign(hist_mean_hod=vals.astype(float).values)

    X_train = _add_hod(X_train)
    X_val   = _add_hod(X_val)
    X_test  = _add_hod(X_test)
    eps = 1e-4
    for Xs in [X_train, X_val, X_test]:
        Xs["dev_from_hod"]     = Xs["lag_1"] - Xs["hist_mean_hod"]
        Xs["rel_dev_from_hod"] = Xs["lag_1"] / (Xs["hist_mean_hod"] + eps)
    print(f"  Features after hod-mean: {len(X_train.columns)}")

    # ── 4. Correlation feature selection ─────────────────────────
    from sklearn.preprocessing import StandardScaler
    print(f"\n2. Feature selection (corr threshold = {CORR_THRESHOLD})...")
    # Compute corr on raw (not scaled yet) to decide columns
    X_train_sc_tmp = pd.DataFrame(
        StandardScaler().fit_transform(X_train),
        index=X_train.index, columns=X_train.columns
    )
    X_val_tmp  = pd.DataFrame(
        StandardScaler().fit(X_train).transform(X_val),
        index=X_val.index, columns=X_val.columns
    )
    X_test_tmp = pd.DataFrame(
        StandardScaler().fit(X_train).transform(X_test),
        index=X_test.index, columns=X_test.columns
    )
    _, _, _, kept = remove_correlated_features(X_train_sc_tmp, X_val_tmp, X_test_tmp)

    X_train = X_train[kept]
    X_val   = X_val[kept]
    X_test  = X_test[kept]

    # ── 5. Scale (fit on train only) ──────────────────────────────
    print("\n3. Scaling (fit on train only)...")
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_val_sc   = scaler.transform(X_val).astype(np.float32)
    X_test_sc  = scaler.transform(X_test).astype(np.float32)

    # ── 6. Log1p target transform ─────────────────────────────────
    y_train_np = np.log1p(y_train.to_numpy(dtype=np.float32))
    y_val_np   = np.log1p(y_val.to_numpy(dtype=np.float32))
    y_test_np  = y_test.to_numpy(dtype=np.float32)   # original scale for evaluation

    # ── 7. Build sequences ────────────────────────────────────────
    print(f"\n4. Building sequences (seq_len={SEQ_LEN})...")
    X_tr_seq, y_tr_seq = make_sequences(X_train_sc, y_train_np, SEQ_LEN)
    X_va_seq, y_va_seq = make_sequences(X_val_sc,   y_val_np,   SEQ_LEN)
    X_te_seq, y_te_seq = make_sequences(X_test_sc,  y_test_np,  SEQ_LEN)
    # y_test_seq is still original scale for eval (no log on test y)
    y_te_true = y_test_np[SEQ_LEN:]   # original scale, aligned with sequences

    # Standardize sequences using train stats
    X_tr_seq, X_va_seq, X_te_seq = standardize_sequences(X_tr_seq, X_va_seq, X_te_seq)

    print(f"  X_train seq: {X_tr_seq.shape}  y: {y_tr_seq.shape}")
    print(f"  X_val   seq: {X_va_seq.shape}  y: {y_va_seq.shape}")
    print(f"  X_test  seq: {X_te_seq.shape}  y: {y_te_seq.shape}")

    input_shape = (SEQ_LEN, X_tr_seq.shape[2])
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    test_index_seq = test.index[SEQ_LEN:]
    results = []
    preds_df = pd.DataFrame({"y_true": y_te_true}, index=test_index_seq)

    # ═══════════════════════════════════════════════════════════════
    # Bidirectional LSTM
    # ═══════════════════════════════════════════════════════════════
    print("\n5. Training Bidirectional LSTM...")
    bilstm = build_bilstm(input_shape)
    bilstm.summary()
    hist_bilstm, bilstm_train_time = train_model(
        bilstm, X_tr_seq, y_tr_seq, X_va_seq, y_va_seq, "BiLSTM"
    )

    t0 = time.time()
    bilstm_pred_log = bilstm.predict(X_te_seq, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    bilstm_pred = np.expm1(bilstm_pred_log)
    bilstm_pred_time = time.time() - t0

    bilstm_metrics = eval_metrics(y_te_true, bilstm_pred)
    print(f"\n  BiLSTM: MAE={bilstm_metrics['MAE']:.4f}  RMSE={bilstm_metrics['RMSE']:.4f}  "
          f"MAPE={bilstm_metrics['MAPE']:.2f}%  R²={bilstm_metrics['R2']:.4f}")

    results.append({
        "Model": f"BiLSTM(seq={SEQ_LEN})",
        **bilstm_metrics,
        "TrainTimeSec": bilstm_train_time,
        "PredictTimeSec": bilstm_pred_time,
    })
    preds_df["y_bilstm"] = bilstm_pred
    save_pred_plot(test_index_seq, y_te_true, bilstm_pred, "BiLSTM", ts_str)

    # ═══════════════════════════════════════════════════════════════
    # CNN-LSTM
    # ═══════════════════════════════════════════════════════════════
    print("\n6. Training CNN-LSTM...")
    cnnlstm = build_cnn_lstm(input_shape)
    cnnlstm.summary()
    hist_cnn, cnn_train_time = train_model(
        cnnlstm, X_tr_seq, y_tr_seq, X_va_seq, y_va_seq, "CNN-LSTM"
    )

    t0 = time.time()
    cnn_pred_log = cnnlstm.predict(X_te_seq, batch_size=BATCH_SIZE, verbose=0).reshape(-1)
    cnn_pred = np.expm1(cnn_pred_log)
    cnn_pred_time = time.time() - t0

    cnn_metrics = eval_metrics(y_te_true, cnn_pred)
    print(f"\n  CNN-LSTM: MAE={cnn_metrics['MAE']:.4f}  RMSE={cnn_metrics['RMSE']:.4f}  "
          f"MAPE={cnn_metrics['MAPE']:.2f}%  R²={cnn_metrics['R2']:.4f}")

    results.append({
        "Model": f"CNN-LSTM(seq={SEQ_LEN})",
        **cnn_metrics,
        "TrainTimeSec": cnn_train_time,
        "PredictTimeSec": cnn_pred_time,
    })
    preds_df["y_cnnlstm"] = cnn_pred
    save_pred_plot(test_index_seq, y_te_true, cnn_pred, "CNN-LSTM", ts_str)

    # ── Save ─────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(results)
    metrics_path = f"{OUT_TABLES}/dl_opt_metrics_{ts_str}.csv"
    preds_path   = f"{OUT_TABLES}/dl_opt_predictions_{ts_str}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    preds_df.to_csv(preds_path)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("FINAL RESULTS")
    print("=" * 62)
    print(f"  Features kept : {len(kept)}  |  Seq len: {SEQ_LEN}")
    print()
    for row in results:
        print(f"  {row['Model']}")
        print(f"    MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}  "
              f"MAPE={row['MAPE']:.2f}%  R²={row['R2']:.4f}")
        print(f"    Train: {row['TrainTimeSec']:.1f}s")
    print()
    print("Saved:")
    print(f"  - {metrics_path}")
    print(f"  - {preds_path}")


if __name__ == "__main__":
    main()
