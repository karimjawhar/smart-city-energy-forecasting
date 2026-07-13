import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_uci_household_power_consumption
from src.features import build_supervised_features
from src.metrics import compute_regression_metrics
from src.models.arima import fit_predict_arima
from src.models.rf import fit_predict_random_forest
from src.models.xgboost import fit_predict_xgboost
from src.split import chronological_split


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    raw_path = project_root / "data" / "raw" / "household_power_consumption.txt"

    outputs_dir = project_root / "outputs"
    figures_dir = outputs_dir / "figures"
    tables_dir = outputs_dir / "tables"
    ensure_dirs(figures_dir, tables_dir, project_root / "data" / "processed")

    target_col = "Global_active_power"
    df = load_uci_household_power_consumption(
        raw_path,
        resample_rule="H",
        target_col=target_col,
    )

    df.to_parquet(project_root / "data" / "processed" / "uci_hourly.parquet")

    X, y = build_supervised_features(df, target_col=target_col)

    splits = chronological_split(X, y, train_frac=0.7, val_frac=0.15)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    results = []

    def eval_model(name: str, yhat_val: pd.Series, yhat_test: pd.Series) -> None:
        m_val = compute_regression_metrics(y_val, yhat_val)
        m_test = compute_regression_metrics(y_test, yhat_test)
        results.append(
            {
                "model": name,
                **{f"val_{k}": v for k, v in m_val.items()},
                **{f"test_{k}": v for k, v in m_test.items()},
            }
        )

        pred_df = pd.DataFrame(
            {
                "y_true": y_test,
                "y_pred": yhat_test,
            }
        )
        pred_df.to_csv(tables_dir / f"predictions_{name}.csv", index=True)

        tail_n = min(7 * 24, len(pred_df))
        fig, ax = plt.subplots(figsize=(12, 4))
        pred_df.tail(tail_n).plot(ax=ax)
        ax.set_title(f"Test: {name} (last {tail_n} hours)")
        ax.set_xlabel("time")
        ax.set_ylabel(target_col)
        fig.tight_layout()
        fig.savefig(figures_dir / f"test_plot_{name}.png", dpi=150)
        plt.close(fig)

    naive_val = y_train.iloc[-1:].reindex(y_val.index, method="ffill")
    naive_test = y_val.iloc[-1:].reindex(y_test.index, method="ffill")
    eval_model("naive_last", naive_val, naive_test)

    yhat_val, yhat_test = fit_predict_random_forest(X_train, y_train, X_val, X_test)
    eval_model("rf", yhat_val, yhat_test)

    yhat_val, yhat_test = fit_predict_xgboost(X_train, y_train, X_val, X_test)
    if yhat_val is not None and yhat_test is not None:
        eval_model("xgboost", yhat_val, yhat_test)

    yhat_val, yhat_test = fit_predict_arima(y_train, y_val.index, y_test.index)
    if yhat_val is not None and yhat_test is not None:
        eval_model("arima", yhat_val, yhat_test)

    metrics_df = pd.DataFrame(results).sort_values(by="test_rmse")
    metrics_df.to_csv(tables_dir / "metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 4))
    y.plot(ax=ax, label="target")
    ax.axvline(y_train.index[-1], color="k", linestyle="--", linewidth=1)
    ax.axvline(y_val.index[-1], color="k", linestyle="--", linewidth=1)
    ax.set_title("Target with split boundaries")
    ax.set_xlabel("time")
    ax.set_ylabel(target_col)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "target_splits.png", dpi=150)
    plt.close(fig)

    print("Done")
    print(f"Saved: {tables_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
