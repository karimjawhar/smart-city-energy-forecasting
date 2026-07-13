# Smart City Energy Consumption Forecasting

Forecasting hourly household electricity demand using classical time-series baselines, tree-based ML models, and deep learning (LSTM / CNN-LSTM), built on the UCI Individual Household Electric Power Consumption dataset.

This project was my final year dissertation (BSc Computer Science - AI, Heriot-Watt University Dubai, submitted November 2025), focused on short-term electricity demand forecasting for smart city applications — where model accuracy *and* interpretability both matter, since stakeholders need to trust and act on the predictions.

## Problem

Smart grids and smart city infrastructure need reliable short-term load forecasts to manage demand, plan capacity, and reduce waste. This project builds and rigorously compares seven forecasting approaches — from a naive baseline through classical statistics, tree-based ensembles, and deep learning — on real household electricity consumption data, then integrates SHAP explainability so the results are interpretable, not just accurate.

## Dataset

- **Source:** [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- Minute-level electricity measurements from a single residential household, December 2006 – November 2010 (2M+ raw observations)
- **Target variable:** `Global_active_power` (kW)
- Resampled to hourly resolution → **34,589 hourly observations**, with 421 missing values resolved via forward-fill interpolation
- Modelling deliberately relies on autoregressive/temporal features (lags, rolling means, calendar features) derived from the target itself, rather than the other electrical sub-metering columns, to keep model comparison fair and avoid feature bias
- Not included in this repo due to size — download it from the UCI link above and place it at `data/raw/household_power_consumption.txt`

## Approach

The pipeline is organized into sequential stages:

| Stage | Script | Description |
|---|---|---|
| 0 | `main_step0_eda.py` | Exploratory data analysis — time series trends, ACF/PACF, hourly/daily/monthly load distributions, average daily profile |
| 1 | `main_step1_data_pipeline.py` | Load, clean, resample to hourly, engineer lag/rolling/calendar features, chronological train/val/test split |
| 2 | `main_step2_baselines.py` | Naive (t-1) and ARIMA baselines |
| 3 | `main_step3_ml.py` / `main_step3_ml_rf.py` / `main_step3_xgb.py` | Ridge regression, Random Forest, and XGBoost models (with hyperparameter-optimized variants) |
| 4 | `main_step4_shap.py` | SHAP-based feature importance for model interpretability |
| 5 | `main_step5_dl_lstm_cnnlstm.py` | LSTM and CNN-LSTM sequence models for capturing temporal dependencies |

Models are evaluated consistently using **MAE, RMSE, MAPE, and R²** on a held-out, chronologically-separated test set (no shuffling, no leakage — feature scaling and lag/rolling statistics are fit on training data only).

`main.py` also provides a consolidated pipeline that runs the core comparison (naive, Random Forest, XGBoost, ARIMA) end-to-end using shared `src/` modules.

## Models compared

- Naive (persistence) baseline
- ARIMA
- Ridge Regression
- Random Forest
- XGBoost
- LSTM
- CNN-LSTM

## Results

Test set performance (chronological 70/15/15 split, no shuffling):

| Model | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|
| Naive (t−1) | 0.3728 | 0.5752 | 44.75 | 0.3291 |
| ARIMA(3,0,3) | 0.6209 | 0.7245 | 116.96 | −0.0645 |
| Ridge Regression | 0.3382 | 0.4828 | 42.38 | 0.5273 |
| Random Forest | 0.2919 | 0.4389 | 36.24 | 0.6092 |
| **XGBoost** | **0.2853** | **0.4301** | **34.78** | **0.6249** |
| LSTM | 0.4050 | 0.5569 | 56.37 | 0.3683 |
| CNN-LSTM | 0.3934 | 0.5507 | 52.38 | 0.3822 |

**Key findings:**
- **XGBoost was the strongest model** across every metric, followed closely by Random Forest — tree-based ensembles handled the structured lag/temporal features more effectively than either the linear or sequential approaches.
- **ARIMA underperformed even the naive baseline** (negative R²), suggesting its linear autoregressive assumptions can't capture the irregular, behaviour-driven noise in single-household consumption data.
- **LSTM and CNN-LSTM underperformed the ensemble models**, most likely due to the modest dataset size after sequence construction (~24k training samples) — not enough for recurrent architectures to learn what the engineered lag/rolling features already encode explicitly for the tree-based models.
- **SHAP analysis** confirmed Lag 1 and Lag 24 as the dominant predictive features, with hour-of-day also contributing meaningfully — consistent with the two clear daily consumption peaks found in EDA (~07:00–08:00 and ~19:00–22:00).

Key plots (saved automatically to `outputs/figures/`):
- Full time series with 30-day rolling mean
- ACF/PACF for autocorrelation structure
- Hourly / day-of-week / monthly load distributions
- Actual vs. predicted plots for each model (last 7 days of test set)
- Training loss curves for LSTM and CNN-LSTM
- SHAP global importance (beeswarm) and local waterfall plots

## Project structure

```
smart-city-energy-forecasting/
├── data/
│   ├── raw/               # place household_power_consumption.txt here (not included)
│   └── processed/         # generated features and train/val/test splits
├── src/                   # shared modules used by main.py
├── outputs/
│   ├── figures/           # generated plots
│   └── tables/            # generated metrics and predictions
├── main.py                # consolidated pipeline
├── main_step0_eda.py
├── main_step1_data_pipeline.py
├── main_step2_baselines.py
├── main_step3_ml.py
├── main_step3_ml_rf.py
├── main_step3_xgb.py
├── main_step4_shap.py
├── main_step5_dl_lstm_cnnlstm.py
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/YOUR-USERNAME/smart-city-energy-forecasting.git
cd smart-city-energy-forecasting

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Download the dataset from the UCI link above and place it at:
```
data/raw/household_power_consumption.txt
```

## Usage

Run stages in order:

```bash
python main_step0_eda.py              # exploratory analysis
python main_step1_data_pipeline.py    # data cleaning, features, splits
python main_step2_baselines.py        # naive + ARIMA
python main_step3_ml_rf.py            # Ridge + Random Forest
python main_step3_xgb_optimized.py    # XGBoost (tuned)
python main_step4_shap.py             # feature importance
python main_step5_dl_lstm_cnnlstm.py  # LSTM + CNN-LSTM
```

Or run the consolidated pipeline:
```bash
python main.py
```

Outputs (metrics tables and figures) are saved to `outputs/tables/` and `outputs/figures/`.

### Interactive dashboard

An interactive Streamlit dashboard (`app.py`) lets you explore forecasts and compare models directly:

```bash
streamlit run app.py
```

It includes:
- A forecast viewer with dynamic model and time-range selection (Ridge, Random Forest, XGBoost, LSTM, CNN-LSTM)
- A model comparison panel showing MAE/RMSE/MAPE/R² side by side
- Feature importance and error analysis views

*Run the pipeline stages above first so the dashboard has predictions/metrics to load.*

## Tech stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · TensorFlow/Keras · Statsmodels · SHAP · Streamlit · Matplotlib

## Future work

- Extend to multi-step (multi-horizon) forecasting rather than next-hour only
- Incorporate exogenous data (weather, occupancy) — the current models deliberately use only autoregressive/temporal features for fair comparison, but contextual IoT signals could improve accuracy further
- Deploy the trained XGBoost model behind a live inference API
