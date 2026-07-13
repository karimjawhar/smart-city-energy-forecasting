from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit  # kept for potential future use

try:
    from xgboost import XGBRegressor
except Exception as e:
    raise ImportError("XGBoost not installed. Run: python -m pip install xgboost") from e

# ================= CONFIG =================
RAW_PATH      = "data/raw/household_power_consumption.txt"
TARGET = "Global_active_power"

TRAIN_END = "2009-09-20 19:00:00"
VAL_END   = "2010-04-24 20:00:00"

RESAMPLE_FREQ = "1h"

TABLES_DIR = "outputs/tables"
FIG_DIR    = "outputs/figures"

PREV_R2          = 0.5771   # previous best result for comparison
CORR_THRESHOLD   = 0.97     # drop one feature from each pair above this
# =========================================


def ensure_dirs():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100.0)


def eval_metrics(y_true, y_pred):
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape(y_true, y_pred),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def split_by_dates(df):
    """Strict chronological train / val / test split — no shuffling."""
    train = df.loc[:TRAIN_END].copy()
    val   = df.loc[pd.to_datetime(TRAIN_END) + pd.Timedelta(hours=1):VAL_END].copy()
    test  = df.loc[pd.to_datetime(VAL_END)   + pd.Timedelta(hours=1):].copy()
    return train, val, test


def make_xy(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)
    return X, y


def build_rich_features(df: pd.DataFrame, target_col: str):
    """
    Build a rich feature set for electricity consumption forecasting.

    All window operations use .shift(1) before the rolling/ewm call so that
    the value at time t is computed ONLY from data at t-1 and earlier.
    Exogenous columns (sub-meters, voltage, etc.) are also lagged.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    y    = df[target_col].copy()
    y_s1 = y.shift(1)
    feats = pd.DataFrame(index=df.index)

    # ── Target lags (daily + weekly seasonality) ──────────────────
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]:
        feats[f"lag_{lag}"] = y.shift(lag)

    # ── Difference features ───────────────────────────────────────
    feats["diff_1"]   = y.shift(1)   - y.shift(2)
    feats["diff_24"]  = y.shift(24)  - y.shift(48)
    feats["diff_168"] = y.shift(1)   - y.shift(169)   # vs same time last week

    # ── Rolling mean (including weekly scale) ─────────────────────
    for w in [3, 6, 12, 24, 168]:
        feats[f"roll_mean_{w}"] = y_s1.rolling(w).mean()

    # ── Rolling std ───────────────────────────────────────────────
    for w in [6, 12, 24, 168]:
        feats[f"roll_std_{w}"] = y_s1.rolling(w).std()

    # ── Exponential moving averages ───────────────────────────────
    feats["ema_12"] = y_s1.ewm(span=12, adjust=False).mean()
    feats["ema_24"] = y_s1.ewm(span=24, adjust=False).mean()

    # ── Interaction features ──────────────────────────────────────
    lag24_safe  = y.shift(24).replace(0, np.nan)
    lag168_safe = y.shift(168).replace(0, np.nan)
    feats["lag1_over_lag24"]  = y.shift(1) / lag24_safe
    feats["lag1_over_lag168"] = y.shift(1) / lag168_safe

    # ── Exogenous lags (sub-meter, voltage, reactive, intensity) ──
    exog_cols = [c for c in df.columns if c != target_col]
    for col in exog_cols:
        s = df[col]
        for lag in [1, 24, 48, 168]:
            feats[f"{col}_lag{lag}"] = s.shift(lag)

    # ── Rolling stats on most important exogenous (Sub_metering_3) ──
    if "Sub_metering_3" in df.columns:
        sm3 = df["Sub_metering_3"].shift(1)
        feats["sm3_roll_mean_24"]  = sm3.rolling(24).mean()
        feats["sm3_roll_mean_168"] = sm3.rolling(168).mean()
        feats["sm3_roll_std_24"]   = sm3.rolling(24).std()
    if "Sub_metering_1" in df.columns:
        sm1 = df["Sub_metering_1"].shift(1)
        feats["sm1_roll_mean_24"] = sm1.rolling(24).mean()
    if "Sub_metering_2" in df.columns:
        sm2 = df["Sub_metering_2"].shift(1)
        feats["sm2_roll_mean_24"] = sm2.rolling(24).mean()
    if "Voltage" in df.columns:
        volt = df["Voltage"].shift(1)
        feats["volt_roll_mean_24"]  = volt.rolling(24).mean()
        feats["volt_roll_mean_168"] = volt.rolling(168).mean()
    if "Global_reactive_power" in df.columns:
        reac = df["Global_reactive_power"].shift(1)
        feats["reac_roll_mean_24"]  = reac.rolling(24).mean()
        feats["reac_roll_mean_168"] = reac.rolling(168).mean()

    # ── Cyclical time encoding ────────────────────────────────────
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
    # Cross-terms: seasonal × daily pattern
    feats["month_x_hour_sin"] = m_sin * h_sin
    feats["month_x_hour_cos"] = m_cos * h_cos

    out = feats.join(y.to_frame("y"), how="inner").dropna()
    return out.drop(columns=["y"]), out["y"]


# These features are never dropped regardless of correlation
_PROTECTED = {"lag_1", "lag_24", "lag_168", "lag_336", "hist_mean_hod",
              "dev_from_hod", "rel_dev_from_hod"}


def remove_correlated_features(X_train, X_val, X_test, threshold=CORR_THRESHOLD):
    """
    Drop one feature from each highly-correlated pair.
    Correlation matrix computed on TRAINING data only (no leakage).
    Protected features are never dropped.
    """
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        for other in upper.index[upper[col] > threshold]:
            col_p  = col   in _PROTECTED
            other_p = other in _PROTECTED
            if col_p and other_p:
                continue               # both protected — keep both
            elif other_p:
                to_drop.add(col)       # protect other, drop col
            elif col_p:
                to_drop.add(other)     # protect col, drop other
            elif X_train[col].var() <= X_train[other].var():
                to_drop.add(col)
            else:
                to_drop.add(other)

    kept = [c for c in X_train.columns if c not in to_drop]
    print(f"  Feature selection: {len(X_train.columns)} -> {len(kept)} "
          f"(dropped {len(to_drop)}: {sorted(to_drop)})")
    return X_train[kept], X_val[kept], X_test[kept], kept


def load_raw_hourly():
    """Load raw UCI data, keep ALL columns, resample to hourly mean."""
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw data not found: {RAW_PATH}")
    df = pd.read_csv(
        RAW_PATH,
        sep=";",
        na_values=["?", "NA", ""],
        low_memory=False,
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
    hourly = df.resample(RESAMPLE_FREQ).mean()
    hourly = hourly.interpolate("time").ffill().bfill()
    # Derived: unmetered load = total active power - sum of sub-meters
    if all(c in hourly.columns for c in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]):
        sub_kwh = (hourly["Sub_metering_1"] + hourly["Sub_metering_2"] + hourly["Sub_metering_3"]) / 1000.0
        hourly["Sub_other"] = np.maximum(0.0, hourly[TARGET] - sub_kwh)
    return hourly


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
    plt.figure(figsize=(14, 6))
    plt.plot(index, y_true, label="Actual",    alpha=0.8, linewidth=1)
    plt.plot(index, y_pred, label="Predicted", alpha=0.8, linewidth=1)
    plt.title(title, fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel(TARGET, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 62)
    print("Enhanced XGBoost v2 — Electricity Consumption Forecasting")
    print("=" * 62)

    ensure_dirs()

    # ── 1. Load data (all columns from raw file) ─────────────────
    print("\n1. Loading data...")
    df = load_raw_hourly()

    # ── 2. Feature engineering ────────────────────────────────────
    print("2. Building rich features...")
    X, y = build_rich_features(df, TARGET)
    df_features = X.copy()
    df_features[TARGET] = y

    # ── 3. Chronological split ────────────────────────────────────
    train, val, test = split_by_dates(df_features)
    X_train, y_train = make_xy(train)
    X_val,   y_val   = make_xy(val)
    X_test,  y_test  = make_xy(test)

    print(f"  Train : {train.shape}  {train.index.min()} -> {train.index.max()}")
    print(f"  Val   : {val.shape}   {val.index.min()} -> {val.index.max()}")
    print(f"  Test  : {test.shape}   {test.index.min()} -> {test.index.max()}")
    print(f"  Raw features: {len(X_train.columns)}")

    # ── 3b. Historical (hour × dayofweek) mean ────────────────────
    # Computed ONLY on training data to prevent leakage.
    # Captures deterministic daily/weekly consumption patterns.
    hod_mean = (
        y_train.groupby([X_train.index.hour, X_train.index.dayofweek]).mean()
    )
    hod_mean.index.names = ["hour", "dow"]

    def _add_hod(X):
        keys = pd.MultiIndex.from_arrays([X.index.hour, X.index.dayofweek],
                                         names=["hour", "dow"])
        vals = keys.map(hod_mean)
        return X.assign(hist_mean_hod=vals.astype(float).values)

    X_train = _add_hod(X_train)
    X_val   = _add_hod(X_val)
    X_test  = _add_hod(X_test)

    # Deviation features — helps model learn residuals
    eps = 1e-4
    for X in [X_train, X_val, X_test]:
        X["dev_from_hod"]     = X["lag_1"] - X["hist_mean_hod"]
        X["rel_dev_from_hod"] = X["lag_1"] / (X["hist_mean_hod"] + eps)
    print(f"  Features after hod-mean + deviations: {len(X_train.columns)}")

    # ── 4. Scale — fit ONLY on training data ─────────────────────
    print("\n3. Scaling (fit on train only)...")
    scaler     = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train),
                               index=X_train.index, columns=X_train.columns)
    X_val_sc   = pd.DataFrame(scaler.transform(X_val),
                               index=X_val.index,   columns=X_val.columns)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),
                               index=X_test.index,  columns=X_test.columns)

    # ── 5. Correlation-based feature selection ────────────────────
    print(f"\n4. Feature selection (corr threshold = {CORR_THRESHOLD})...")
    X_train_sc, X_val_sc, X_test_sc, kept = remove_correlated_features(
        X_train_sc, X_val_sc, X_test_sc
    )

    # ── 6. Fixed hyperparameters + Early stopping ───────────────────
    # Using proven params from prior tuning run (CV MAE=0.3789).
    # Early stopping on (train, val) finds optimal n_estimators
    # without touching the test set.  Skipping RandomizedSearchCV
    # removes ~40 min of overhead; the richer features drive the R² gain.
    print("\n5. Early-stopping pass (train → val)...")

    best_params = {
        "learning_rate":    0.005,
        "max_depth":        6,
        "min_child_weight": 1,
        "subsample":        0.9,
        "colsample_bytree": 0.8,
        "reg_lambda":       3.0,
        "reg_alpha":        0.1,
    }

    # Log1p-transform: electricity consumption is right-skewed (MAPE=38%).
    # Fitting on log scale makes low-consumption hours equally important.
    # Predictions are inverse-transformed with expm1 before evaluation.
    y_train_t = np.log1p(y_train)
    y_val_t   = np.log1p(y_val)

    t0 = time.time()
    es_model = XGBRegressor(
        **best_params,
        n_estimators=8000,
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        max_bin=512,
        colsample_bylevel=0.8,
    )
    es_model.fit(
        X_train_sc, y_train_t,
        eval_set=[(X_val_sc, y_val_t)],
        verbose=100,
    )
    optimal_trees = es_model.best_iteration + 1
    tuning_time   = time.time() - t0
    print(f"  Optimal n_estimators : {optimal_trees}")
    print(f"  Best val MAE         : {es_model.best_score:.4f}")
    print(f"  Time                 : {tuning_time:.1f}s")

    # ── 8. Final model — retrain on train + val ───────────────────
    print("\n7. Retraining final model on train + val...")
    X_tv = pd.concat([X_train_sc, X_val_sc])
    y_tv = np.log1p(pd.concat([y_train, y_val]))

    final_model = XGBRegressor(
        **best_params,
        n_estimators=optimal_trees,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        max_bin=512,
        colsample_bylevel=0.8,
    )

    t1 = time.time()
    final_model.fit(X_tv, y_tv, verbose=False)
    train_time = time.time() - t1

    # ── 9. Evaluate on test set ONLY ──────────────────────────────
    print("\n8. Evaluating on test set...")
    t2     = time.time()
    y_pred = np.expm1(final_model.predict(X_test_sc))
    pred_time = time.time() - t2

    metrics = eval_metrics(y_test, y_pred)

    # ── 10. Save results ──────────────────────────────────────────
    final_params_str = str({**best_params, "n_estimators": optimal_trees})
    res_df = pd.DataFrame([{
        "Model":            "XGBoost_v2",
        **metrics,
        "TrainTimeSec":     train_time,
        "PredictTimeSec":   pred_time,
        "TuningTimeSec":    tuning_time,
        "NFeatures":        len(kept),
        "OptimalEstimators":optimal_trees,
        "BestParams":       final_params_str,
    }])

    preds_df = pd.DataFrame(
        {"y_true": y_test, "y_xgb_v2": y_pred}, index=test.index
    )

    metrics_path = safe_save_csv(res_df, "xgb_v2_metrics")
    preds_path   = safe_save_csv_indexed(preds_df, "xgb_v2_predictions")

    plot_actual_vs_pred(
        test.index, y_test.values, y_pred,
        "XGBoost v2: Actual vs Predicted (Test Set)",
        os.path.join(FIG_DIR, "test_plot_xgb_v2.png"),
    )

    # Feature importance plot
    import matplotlib.pyplot as plt
    importance = pd.DataFrame({
        "feature":    kept,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    fig_h = max(6, len(kept) * 0.35)
    plt.figure(figsize=(10, fig_h))
    plt.barh(range(len(importance)), importance["importance"])
    plt.yticks(range(len(importance)), importance["feature"])
    plt.xlabel("Feature Importance (gain)")
    plt.title("Feature Importance — Enhanced XGBoost v2")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "feature_importance_xgb_v2.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    importance_path = os.path.join(TABLES_DIR, "feature_importance_xgb_v2.csv")
    importance.to_csv(importance_path, index=False)

    # ── 11. Summary ───────────────────────────────────────────────
    r2_delta = metrics["R2"] - PREV_R2
    print("\n" + "=" * 62)
    print("FINAL RESULTS")
    print("=" * 62)
    print(f"  Features kept       : {len(kept)}")
    print(f"  Optimal trees       : {optimal_trees}")
    print(f"  Best hyperparameters: {final_params_str}")
    print()
    print(f"  MAE  : {metrics['MAE']:.4f}")
    print(f"  RMSE : {metrics['RMSE']:.4f}")
    print(f"  MAPE : {metrics['MAPE']:.2f}%")
    print(f"  R²   : {metrics['R2']:.4f}  (previous best: {PREV_R2:.4f})")
    print(f"  ΔR²  : {r2_delta:+.4f}  ({'improved' if r2_delta > 0 else 'regressed'})")
    print()
    print(f"  Tuning time : {tuning_time:.1f}s")
    print(f"  Train time  : {train_time:.1f}s")
    print()
    print("Saved:")
    print(f"  - {metrics_path}")
    print(f"  - {preds_path}")
    print(f"  - {importance_path}")
    print(f"  - {os.path.join(FIG_DIR, 'test_plot_xgb_v2.png')}")
    print(f"  - {os.path.join(FIG_DIR, 'feature_importance_xgb_v2.png')}")


if __name__ == "__main__":
    main()
