from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import os
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================= CONFIG =================
RAW_PATH    = "data/raw/household_power_consumption.txt"
TARGET      = "Global_active_power"
TRAIN_END   = "2009-09-20 19:00:00"
VAL_END     = "2010-04-24 20:00:00"
RESAMPLE_FREQ  = "h"
CORR_THRESHOLD = 0.97
TABLES_DIR  = "outputs/tables"
FIG_DIR     = "outputs/figures"
PREV_R2     = {"ridge": 0.0, "rf": 0.0}
# =========================================


def ensure_dirs() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom  = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def eval_metrics(y_true, y_pred) -> dict:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return {
        "MAE":  float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAPE": mape(yt, yp),
        "R2":   float(r2_score(yt, yp)),
    }


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
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

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

    exog_cols = [c for c in df.columns if c != target_col]
    for col in exog_cols:
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
    h_sin = np.sin(2 * np.pi * idx.hour      / 24)
    h_cos = np.cos(2 * np.pi * idx.hour      / 24)
    m_sin = np.sin(2 * np.pi * idx.month     / 12)
    m_cos = np.cos(2 * np.pi * idx.month     / 12)
    feats["hour_sin"]  = h_sin
    feats["hour_cos"]  = h_cos
    feats["dow_sin"]   = np.sin(2 * np.pi * idx.dayofweek / 7)
    feats["dow_cos"]   = np.cos(2 * np.pi * idx.dayofweek / 7)
    feats["month_sin"] = m_sin
    feats["month_cos"] = m_cos
    woy = idx.isocalendar().week.astype(int).to_numpy()
    feats["woy_sin"]   = np.sin(2 * np.pi * woy / 52)
    feats["woy_cos"]   = np.cos(2 * np.pi * woy / 52)
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


def safe_save_csv(df, base_name):
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(TABLES_DIR, f"{base_name}_{ts}.csv")
    df.to_csv(path, index=False)
    return path


def safe_save_csv_indexed(df, base_name):
    ts   = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(TABLES_DIR, f"{base_name}_{ts}.csv")
    df.to_csv(path)
    return path


def plot_actual_vs_pred(index, y_true, y_pred, title, out_path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(index, y_true, label="Actual", linewidth=0.8)
    plt.plot(index, y_pred, label="Predicted", linewidth=0.8, alpha=0.8)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(TARGET)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    print("=" * 62)
    print("Optimized Ridge + Random Forest — Electricity Forecasting")
    print("=" * 62)

    ensure_dirs()

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
    print(f"  Raw features: {len(X_train.columns)}")

    # ── 3a. Historical (hour × dayofweek) mean — train only ───────
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
    for X in [X_train, X_val, X_test]:
        X["dev_from_hod"]     = X["lag_1"] - X["hist_mean_hod"]
        X["rel_dev_from_hod"] = X["lag_1"] / (X["hist_mean_hod"] + eps)
    print(f"  Features after hod-mean + deviations: {len(X_train.columns)}")

    # ── 4. Scale — fit on train only ─────────────────────────────
    print("\n2. Scaling (fit on train only)...")
    scaler     = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train),
                               index=X_train.index, columns=X_train.columns)
    X_val_sc   = pd.DataFrame(scaler.transform(X_val),
                               index=X_val.index,   columns=X_val.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),
                               index=X_test.index,  columns=X_test.columns)

    # ── 5. Correlation-based feature selection ────────────────────
    print(f"\n3. Feature selection (corr threshold = {CORR_THRESHOLD})...")
    X_train_sc, X_val_sc, X_test_sc, kept = remove_correlated_features(
        X_train_sc, X_val_sc, X_test_sc
    )

    # ── 6. Log1p target transform ─────────────────────────────────
    y_train_t = np.log1p(y_train)
    y_val_t   = np.log1p(y_val)

    # Combine train+val for final model fitting
    X_tv = pd.concat([X_train_sc, X_val_sc])
    y_tv = np.log1p(pd.concat([y_train, y_val]))

    results = []
    preds_df = pd.DataFrame({"y_true": y_test}, index=test.index)

    # ═══════════════════════════════════════════════════════════════
    # RIDGE — tune alpha with TimeSeriesSplit
    # ═══════════════════════════════════════════════════════════════
    print("\n4. Tuning Ridge (GridSearchCV, alpha)...")
    tscv = TimeSeriesSplit(n_splits=3)
    ridge_grid = GridSearchCV(
        Ridge(),
        param_grid={"alpha": [0.01, 0.1, 1, 10, 100, 1000, 10000]},
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    ridge_grid.fit(X_train_sc, y_train_t)
    best_alpha = ridge_grid.best_params_["alpha"]
    print(f"  Best alpha: {best_alpha}")

    # Final Ridge on train+val
    ridge_final = Ridge(alpha=best_alpha)
    t0 = time.time()
    ridge_final.fit(X_tv, y_tv)
    ridge_train_time = time.time() - t0

    t1 = time.time()
    ridge_pred = np.expm1(ridge_final.predict(X_test_sc))
    ridge_pred_time = time.time() - t1

    ridge_metrics = eval_metrics(y_test, ridge_pred)
    print(f"  Ridge: MAE={ridge_metrics['MAE']:.4f}  RMSE={ridge_metrics['RMSE']:.4f}  "
          f"MAPE={ridge_metrics['MAPE']:.2f}%  R²={ridge_metrics['R2']:.4f}")

    results.append({
        "Model": f"Ridge(alpha={best_alpha})",
        **ridge_metrics,
        "TrainTimeSec": ridge_train_time,
        "PredictTimeSec": ridge_pred_time,
    })
    preds_df["y_ridge"] = ridge_pred

    plot_actual_vs_pred(
        test.index, y_test.values, ridge_pred,
        "Optimized Ridge: Actual vs Predicted (Test)",
        os.path.join(FIG_DIR, "test_plot_ridge_opt.png"),
    )

    # ═══════════════════════════════════════════════════════════════
    # RANDOM FOREST — fixed proven hyperparameters (no CV overhead)
    # ═══════════════════════════════════════════════════════════════
    print("\n5. Training Random Forest (fixed hyperparameters)...")
    best_rf_params = {
        "n_estimators":     800,
        "max_depth":        None,
        "max_features":     0.4,
        "min_samples_leaf": 2,
        "bootstrap":        True,
    }
    print(f"  RF params: {best_rf_params}")

    # Final RF on train+val (log1p helps focus on low-consumption hours)
    rf_final = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
    t1 = time.time()
    rf_final.fit(X_tv, y_tv)   # y_tv is already log1p-transformed
    rf_train_time = time.time() - t1

    t2 = time.time()
    rf_pred = np.expm1(rf_final.predict(X_test_sc))
    rf_pred_time = time.time() - t2

    rf_metrics = eval_metrics(y_test, rf_pred)
    print(f"  RF:    MAE={rf_metrics['MAE']:.4f}  RMSE={rf_metrics['RMSE']:.4f}  "
          f"MAPE={rf_metrics['MAPE']:.2f}%  R²={rf_metrics['R2']:.4f}")

    results.append({
        "Model": f"RandomForest(opt)",
        **rf_metrics,
        "TrainTimeSec": rf_train_time,
        "PredictTimeSec": rf_pred_time,
    })
    preds_df["y_rf"] = rf_pred

    plot_actual_vs_pred(
        test.index, y_test.values, rf_pred,
        "Optimized RF: Actual vs Predicted (Test)",
        os.path.join(FIG_DIR, "test_plot_rf_opt.png"),
    )

    # ── Feature importance (RF) ───────────────────────────────────
    import matplotlib.pyplot as plt
    importance = pd.DataFrame({
        "feature":    kept,
        "importance": rf_final.feature_importances_,
    }).sort_values("importance", ascending=False)
    fig_h = max(6, len(kept) * 0.30)
    plt.figure(figsize=(10, fig_h))
    plt.barh(range(len(importance)), importance["importance"])
    plt.yticks(range(len(importance)), importance["feature"])
    plt.xlabel("Feature Importance (impurity)")
    plt.title("Feature Importance — Optimized Random Forest")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance_rf_opt.png"), dpi=200, bbox_inches="tight")
    plt.close()
    importance.to_csv(os.path.join(TABLES_DIR, "feature_importance_rf_opt.csv"), index=False)

    # ── Save ─────────────────────────────────────────────────────
    res_df = pd.DataFrame(results)
    metrics_path = safe_save_csv(res_df, "ml_opt_metrics")
    preds_path   = safe_save_csv_indexed(preds_df, "ml_opt_predictions")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("FINAL RESULTS")
    print("=" * 62)
    print(f"  Features kept : {len(kept)}")
    print()
    for row in results:
        print(f"  {row['Model']}")
        print(f"    MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}  "
              f"MAPE={row['MAPE']:.2f}%  R²={row['R2']:.4f}")
    print()
    print("Saved:")
    print(f"  - {metrics_path}")
    print(f"  - {preds_path}")
    print(f"  - {os.path.join(FIG_DIR, 'test_plot_ridge_opt.png')}")
    print(f"  - {os.path.join(FIG_DIR, 'test_plot_rf_opt.png')}")
    print(f"  - {os.path.join(FIG_DIR, 'feature_importance_rf_opt.png')}")


if __name__ == "__main__":
    main()
