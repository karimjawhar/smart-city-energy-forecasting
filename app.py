"""
Electricity Consumption Forecasting Dashboard
UCI Household Electric Power Consumption — Dissertation Project
Run: streamlit run app.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Electricity Forecasting Dashboard",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
.main .block-container{padding-top:1.4rem;max-width:1380px}
[data-testid="stSidebar"]{background-color:#0f172a}
[data-testid="stSidebar"] *{color:#e2e8f0 !important}
div[data-testid="metric-container"]{
  background:linear-gradient(135deg,#1e293b,#0f172a);
  border:1px solid #334155;border-radius:12px;padding:.9rem}
div[data-testid="metric-container"] label{color:#94a3b8 !important;font-size:.78rem}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{
  color:#f1f5f9 !important;font-size:1.55rem;font-weight:700}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:#f1f5f9;border-radius:10px;padding:4px}
.stTabs [data-baseweb="tab"]{border-radius:8px;color:#64748b;font-weight:500}
.stTabs [aria-selected="true"]{background:#0f172a !important;color:#f1f5f9 !important}
.sec-hdr{font-size:1rem;font-weight:600;color:#1e293b;
  border-left:4px solid #3b82f6;padding-left:10px;margin:1.1rem 0 .5rem}
.insight{background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid #3b82f6;
  border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.7rem;
  font-size:.88rem;color:#334155}
[data-testid="stAppDeployButton"]{display:none !important}
[data-testid="stStatusWidget"]{display:none !important}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────
TABLES = Path("outputs/tables")

COLORS = {
    "Actual":   "#1e293b",
    "XGBoost":  "#ef4444",
    "RF":       "#3b82f6",
    "Ridge":    "#10b981",
    "LSTM":     "#8b5cf6",
    "CNN-LSTM": "#f59e0b",
    "Naive":    "#94a3b8",
}

DISPLAY = {
    "XGBoost":"XGBoost","RF":"Random Forest","Ridge":"Ridge Regression",
    "LSTM":"LSTM","CNN-LSTM":"CNN–LSTM","Naive":"Naïve (t−1)",
}

METRICS_DF = pd.DataFrame({
    "Model":  ["Naïve (t−1)","Ridge Regression","Random Forest","XGBoost","LSTM","CNN–LSTM"],
    "MAE":    [0.3728, 0.3382, 0.2919, 0.2853, 0.4050, 0.3934],
    "RMSE":   [0.5752, 0.4828, 0.4389, 0.4301, 0.5569, 0.5507],
    "MAPE %": [44.75,  42.38,  36.24,  34.78,  56.37,  52.38],
    "R²":     [0.3291, 0.5273, 0.6092, 0.6249, 0.3683, 0.3822],
})

FEAT_LABELS = {
    "lag_1":"Lag 1 (t−1h)","lag_2":"Lag 2 (t−2h)",
    "lag_24":"Lag 24 (t−24h)","lag_168":"Lag 168 (t−1 week)",
    "lag_336":"Lag 336 (t−2 weeks)",
    "hist_mean_hod":"Historical Mean (Hour×DoW)",
    "dev_from_hod":"Deviation from Hist. Mean",
    "rel_dev_from_hod":"Relative Dev. from Hist. Mean",
    "hour_cos":"Hour of Day (cos)","hour_sin":"Hour of Day (sin)",
    "month_cos":"Month (cos)","month_sin":"Month (sin)",
    "roll_mean_3":"Rolling Mean 3h","roll_mean_24":"Rolling Mean 24h",
    "roll_mean_168":"Rolling Mean 1 week",
    "diff_1":"1-Hour Difference","ema_24":"EMA 24h",
    "Sub_metering_3_lag1":"Sub-meter 3 Lag 1h",
}

INSIGHTS = {
    "XGBoost": "XGBoost achieves the best R²=0.625. It captures sharp fluctuations via lag features and the historical hour×DoW mean (>40% combined importance). Log₁p target transformation reduces outlier bias.",
    "RF": "Random Forest (R²=0.609) closely matches XGBoost using 800 unconstrained trees. It slightly underestimates consumption peaks due to averaging behaviour.",
    "Ridge": "Ridge Regression (R²=0.527) captures broad temporal patterns but is bounded by its linear decision boundary; it cannot model non-linear lag relationships.",
    "LSTM": "LSTM (R²=0.368) produces smoother predictions than tree models but consistently underestimates peaks. A compact 8-feature input limits representational power.",
    "CNN-LSTM": "CNN–LSTM (R²=0.382) adds a causal convolution layer before the LSTM, extracting local patterns and marginally improving over vanilla LSTM.",
    "Naive": "Naïve (t−1) (R²=0.329) repeats the last known value. It is the performance floor — all trained models must exceed this to demonstrate value.",
}

# ── Data loading ──────────────────────────────────────────────────
def _latest(pattern: str) -> Path | None:
    files = sorted(TABLES.glob(pattern))
    return files[-1] if files else None

@st.cache_data(show_spinner="Loading predictions…")
def load_predictions() -> pd.DataFrame:
    frames = []

    # Naive
    p = TABLES / "naive_test_predictions.csv"
    if p.exists():
        df = pd.read_csv(p, parse_dates=["datetime"], index_col="datetime")
        frames.append(df.rename(columns={"y_pred": "Naive"})[["y_true","Naive"]])

    # ML (Ridge + RF)
    p = _latest("ml_opt_predictions_*.csv")
    if p:
        df = pd.read_csv(p, parse_dates=["datetime"], index_col="datetime")
        frames.append(df[["y_ridge","y_rf"]].rename(columns={"y_ridge":"Ridge","y_rf":"RF"}))

    # XGBoost
    p = _latest("xgb_v2_predictions_*.csv")
    if p:
        df = pd.read_csv(p, parse_dates=["datetime"], index_col="datetime")
        frames.append(df[["y_xgb_v2"]].rename(columns={"y_xgb_v2":"XGBoost"}))

    # DL
    p = _latest("dl_predictions_*.csv")
    if p:
        df = pd.read_csv(p, parse_dates=["datetime"], index_col="datetime")
        frames.append(df[["y_lstm","y_cnnlstm"]].rename(columns={"y_lstm":"LSTM","y_cnnlstm":"CNN-LSTM"}))

    if not frames:
        st.error("No prediction files found in outputs/tables/")
        st.stop()

    base = frames[0][["y_true"]]
    for f in frames:
        base = base.join(f.drop(columns=["y_true"], errors="ignore"), how="left")
    return base.sort_index()

@st.cache_data(show_spinner="Loading feature importance…")
def load_importance(model: str) -> pd.DataFrame | None:
    fname = {"XGBoost": "feature_importance_xgb_v2.csv",
             "RF":      "feature_importance_rf_opt.csv"}.get(model)
    if fname is None:
        return None
    p = TABLES / fname
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["label"] = df["feature"].map(FEAT_LABELS).fillna(df["feature"])
    return df.sort_values("importance", ascending=False).head(15)

# ── Plot helpers ──────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="white", plot_bgcolor="#f8fafc",
    font=dict(family="Inter, sans-serif", size=12, color="#1e293b"),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#e2e8f0",
                borderwidth=1, orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1),
    xaxis=dict(gridcolor="#e2e8f0", showgrid=True),
    yaxis=dict(gridcolor="#e2e8f0", showgrid=True),
)

def forecast_chart(df: pd.DataFrame, models: list[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["y_true"], name="Actual",
        line=dict(color=COLORS["Actual"], width=1.5),
        hovertemplate="<b>Actual</b>: %{y:.3f} kW<br>%{x}<extra></extra>",
    ))
    for m in models:
        if m not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df[m], name=DISPLAY.get(m, m),
            line=dict(color=COLORS.get(m, "#666"), width=1.5, dash="dot" if m in ("LSTM","CNN-LSTM") else "solid"),
            hovertemplate=f"<b>{DISPLAY.get(m,m)}</b>: %{{y:.3f}} kW<br>%{{x}}<extra></extra>",
        ))
    if len(models) == 1:
        m = models[0]
        if m in df.columns:
            err = (df["y_true"] - df[m]).abs()
            fig.add_trace(go.Scatter(
                x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
                y=pd.concat([df["y_true"], df[m][::-1]]),
                fill="toself", fillcolor="rgba(239,68,68,0.08)",
                line=dict(color="rgba(0,0,0,0)"), showlegend=True,
                name="Error band", hoverinfo="skip",
            ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=400, title="Actual vs Predicted — Test Set",
        yaxis_title="Global Active Power (kW)",
        xaxis_title="Date/Time",
    )
    return fig

def metrics_bar_chart(metric: str) -> go.Figure:
    df = METRICS_DF.sort_values(metric, ascending=(metric in ("MAE","RMSE","MAPE %")))
    best_row = df.iloc[-1 if metric not in ("MAE","RMSE","MAPE %") else 0]
    colors = ["#ef4444" if r["Model"] == "XGBoost" else "#3b82f6"
              for _, r in df.iterrows()]
    fig = go.Figure(go.Bar(
        x=df["Model"], y=df[metric],
        marker_color=colors,
        text=df[metric].round(4).astype(str), textposition="outside",
        hovertemplate="<b>%{x}</b><br>" + metric + ": %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=360, title=f"Model Comparison — {metric}",
        yaxis_title=metric, xaxis_title="",
        showlegend=False,
    )
    return fig

def importance_chart(df: pd.DataFrame, model: str) -> go.Figure:
    d = df.sort_values("importance")
    bar_colors = ["#ef4444" if d["importance"].iloc[-1] == v else "#3b82f6"
                  for v in d["importance"]]
    fig = go.Figure(go.Bar(
        x=d["importance"], y=d["label"],
        orientation="h", marker_color=bar_colors,
        text=d["importance"].round(4).astype(str), textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    imp_layout = {**PLOTLY_LAYOUT, "margin": dict(l=200, r=60, t=40, b=40)}
    fig.update_layout(
        **imp_layout,
        height=max(350, len(d) * 28),
        title=f"Top Feature Importances — {DISPLAY.get(model, model)}",
        xaxis_title="Importance Score", yaxis_title="",
    )
    return fig

def error_time_chart(df: pd.DataFrame, model: str) -> go.Figure:
    if model not in df.columns:
        return go.Figure()
    err = df["y_true"] - df[model]
    roll = err.rolling(24, center=True).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=err, name="Hourly Error",
        line=dict(color="#94a3b8", width=0.8),
        hovertemplate="<b>Error</b>: %{y:.3f} kW<br>%{x}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=roll, name="24h Rolling Mean",
        line=dict(color=COLORS.get(model, "#ef4444"), width=2.5),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#1e293b", line_width=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=350, title=f"Prediction Error Over Time — {DISPLAY.get(model, model)}",
        yaxis_title="Error (Actual − Predicted, kW)", xaxis_title="Date/Time",
    )
    return fig

def error_histogram(df: pd.DataFrame, models: list[str]) -> go.Figure:
    fig = go.Figure()
    for m in models:
        if m not in df.columns:
            continue
        err = (df["y_true"] - df[m]).dropna()
        fig.add_trace(go.Histogram(
            x=err, name=DISPLAY.get(m, m),
            opacity=0.7, nbinsx=60,
            marker_color=COLORS.get(m, "#666"),
            hovertemplate="Error: %{x:.3f}<br>Count: %{y}<extra></extra>",
        ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=340, barmode="overlay",
        title="Error Distribution",
        xaxis_title="Prediction Error (kW)", yaxis_title="Count",
    )
    return fig

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Forecasting Dashboard")
    st.markdown("---")
    st.markdown("### Model Selection")
    all_models = list(DISPLAY.keys())
    selected_models = st.multiselect(
        "Select models to display",
        options=all_models,
        default=["XGBoost", "RF"],
        format_func=lambda k: DISPLAY[k],
    )

    st.markdown("### Time Range")
    time_range = st.radio(
        "Select range",
        ["Last 24 hours", "Last 7 days", "Last 30 days", "Full test set"],
        index=1,
    )

    st.markdown("---")
    st.markdown("#### Dataset")
    st.markdown("**UCI Household Power Consumption**")
    st.markdown("Hourly aggregated · 2006–2010")
    st.markdown("#### Test Period")
    st.markdown("25 Apr 2010 → 26 Nov 2010")
    st.markdown("*(5,185 hourly observations)*")
    st.markdown("---")
    st.markdown("<small>Dissertation Project · Short-Term Residential Electricity Forecasting</small>",
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════
preds = load_predictions()

# Apply time range filter
hours_map = {
    "Last 24 hours": 24,
    "Last 7 days":   7 * 24,
    "Last 30 days":  30 * 24,
    "Full test set": len(preds),
}
n = hours_map[time_range]
preds_view = preds.iloc[-n:]

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("# ⚡ Electricity Consumption Forecasting")
st.markdown("**Short-term residential demand forecasting · UCI Smart Meter Dataset**")
st.markdown("---")

# Top KPI cards (best model — XGBoost)
kc1, kc2, kc3, kc4, kc5 = st.columns(5)
kc1.metric("Best Model", "XGBoost")
kc2.metric("R² (XGBoost)", "0.6249", delta="vs Naïve +0.2958")
kc3.metric("MAE (XGBoost)", "0.285 kW")
kc4.metric("RMSE (XGBoost)", "0.430 kW")
kc5.metric("MAPE (XGBoost)", "34.78 %")

st.markdown("")

# ═══════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Forecast Viewer",
    "📊  Model Comparison",
    "🔍  Interpretability",
    "⚠️  Error Analysis",
])

# ──────────────────────────────────────────────────────────────────
# TAB 1 — FORECAST VIEWER
# ──────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-hdr">Actual vs Predicted</div>', unsafe_allow_html=True)

    if not selected_models:
        st.info("Select at least one model in the sidebar.")
    else:
        st.plotly_chart(forecast_chart(preds_view, selected_models),
                        use_container_width=True)

        # Zoom / custom date range
        with st.expander("🔎 Custom date range"):
            min_d = preds.index.min().date()
            max_d = preds.index.max().date()
            c1, c2 = st.columns(2)
            date_from = c1.date_input("From", value=max_d - pd.Timedelta(days=7),
                                       min_value=min_d, max_value=max_d)
            date_to   = c2.date_input("To",   value=max_d,
                                       min_value=min_d, max_value=max_d)
            custom_df = preds.loc[str(date_from):str(date_to)]
            if not custom_df.empty:
                st.plotly_chart(forecast_chart(custom_df, selected_models),
                                use_container_width=True)

        # Download predictions
        st.markdown("#### Download Predictions")
        dl_cols = ["y_true"] + [m for m in selected_models if m in preds_view.columns]
        dl_df = preds_view[dl_cols].rename(columns={"y_true": "Actual"})
        dl_df.columns = ["Actual"] + [DISPLAY.get(m, m) for m in dl_cols[1:]]
        st.download_button(
            "⬇  Download visible predictions (CSV)",
            data=dl_df.to_csv().encode(),
            file_name="forecast_predictions.csv",
            mime="text/csv",
        )

# ──────────────────────────────────────────────────────────────────
# TAB 2 — MODEL COMPARISON
# ──────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-hdr">Performance Metrics — All Models</div>',
                unsafe_allow_html=True)

    # Styled metrics table
    def style_metrics(df: pd.DataFrame):
        def highlight(col):
            if col.name == "R²":
                best = col.max()
                return ["background:#fef2f2;font-weight:700;color:#dc2626"
                        if v == best else "" for v in col]
            elif col.name in ("MAE", "RMSE", "MAPE %"):
                best = col.min()
                return ["background:#f0fdf4;font-weight:700;color:#16a34a"
                        if v == best else "" for v in col]
            return [""] * len(col)

        return (df.style
                .apply(highlight, axis=0)
                .format({"MAE": "{:.4f}", "RMSE": "{:.4f}",
                         "MAPE %": "{:.2f}", "R²": "{:.4f}"}))

    st.dataframe(style_metrics(METRICS_DF), use_container_width=True, hide_index=True)
    st.caption("🟩 Green = best value for that metric  |  🟥 Red = best R² (XGBoost)")

    st.markdown('<div class="sec-hdr">Visual Comparison</div>', unsafe_allow_html=True)
    m_col1, m_col2 = st.columns(2)
    sel_metric = m_col1.selectbox("Metric", ["R²", "RMSE", "MAE", "MAPE %"], index=0)
    st.plotly_chart(metrics_bar_chart(sel_metric), use_container_width=True)

    # Radar chart
    st.markdown('<div class="sec-hdr">Model Radar — Normalised Metrics</div>',
                unsafe_allow_html=True)
    cats = ["R²","MAE","RMSE","MAPE %"]
    radar_df = METRICS_DF.copy()
    for c in ["MAE", "RMSE", "MAPE %"]:
        radar_df[c] = 1 - (radar_df[c] - radar_df[c].min()) / (radar_df[c].max() - radar_df[c].min())
    radar_df["R²"] = (radar_df["R²"] - radar_df["R²"].min()) / (radar_df["R²"].max() - radar_df["R²"].min())
    radar_fig = go.Figure()
    for _, row in radar_df.iterrows():
        vals = [row[c] for c in cats] + [row[cats[0]]]
        model_key = {v: k for k, v in DISPLAY.items()}.get(row["Model"], row["Model"])
        radar_fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats + [cats[0]], fill="toself",
            name=row["Model"],
            line_color=COLORS.get(model_key, "#666"),
            opacity=0.65,
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        paper_bgcolor="white", height=400,
        legend=dict(orientation="h", y=-0.15),
        title="Normalised Performance Radar (higher = better for all axes)",
    )
    st.plotly_chart(radar_fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# TAB 3 — INTERPRETABILITY
# ──────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-hdr">Feature Importance</div>', unsafe_allow_html=True)

    imp_model = st.selectbox(
        "Select model", ["XGBoost", "RF"],
        format_func=lambda k: DISPLAY[k],
    )
    imp_df = load_importance(imp_model)
    if imp_df is not None:
        st.plotly_chart(importance_chart(imp_df, imp_model), use_container_width=True)
        with st.expander("View raw importance table"):
            st.dataframe(
                imp_df[["label", "importance"]].rename(
                    columns={"label": "Feature", "importance": "Importance"}
                ).style.format({"Importance": "{:.5f}"}),
                hide_index=True, use_container_width=True,
            )
    else:
        st.warning("Feature importance file not found.")

    st.markdown('<div class="sec-hdr">Model Behaviour Insights</div>',
                unsafe_allow_html=True)
    for key, insight in INSIGHTS.items():
        st.markdown(f'<div class="insight"><b>{DISPLAY.get(key, key)}</b> — {insight}</div>',
                    unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# TAB 4 — ERROR ANALYSIS
# ──────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-hdr">Prediction Error Over Time</div>',
                unsafe_allow_html=True)

    err_model = st.selectbox(
        "Select model for error analysis",
        options=[m for m in all_models if m in preds.columns],
        format_func=lambda k: DISPLAY[k],
    )
    st.plotly_chart(error_time_chart(preds_view, err_model), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec-hdr">Error Distribution (selected models)</div>',
                    unsafe_allow_html=True)
        hist_models = st.multiselect(
            "Models for histogram",
            options=[m for m in selected_models if m in preds.columns],
            default=[m for m in selected_models if m in preds.columns][:3],
            format_func=lambda k: DISPLAY[k],
        )
        if hist_models:
            st.plotly_chart(error_histogram(preds_view, hist_models),
                            use_container_width=True)

    with c2:
        st.markdown('<div class="sec-hdr">Error Summary Statistics</div>',
                    unsafe_allow_html=True)
        rows = []
        for m in [m for m in all_models if m in preds.columns]:
            e = (preds_view["y_true"] - preds_view[m]).dropna()
            rows.append({
                "Model":       DISPLAY.get(m, m),
                "Mean Error":  round(e.mean(), 4),
                "Std Error":   round(e.std(), 4),
                "Max Overest.": round(e.min(), 4),
                "Max Underest.": round(e.max(), 4),
                "% Within ±0.5kW": round((e.abs() <= 0.5).mean() * 100, 1),
            })
        err_stats = pd.DataFrame(rows)
        st.dataframe(err_stats.style.format({
            "Mean Error": "{:.4f}", "Std Error": "{:.4f}",
            "Max Overest.": "{:.4f}", "Max Underest.": "{:.4f}",
            "% Within ±0.5kW": "{:.1f}",
        }), hide_index=True, use_container_width=True)

        st.markdown('<div class="sec-hdr">Key Observations</div>',
                    unsafe_allow_html=True)
        st.markdown("""
<div class="insight">
📌 <b>Tree-based models</b> (XGBoost, RF) exhibit larger but more localised error
spikes that align with sudden consumption changes — they track trends well but
struggle with abrupt jumps.
</div>
<div class="insight">
📌 <b>DL models</b> (LSTM, CNN–LSTM) show systematic underestimation of consumption
peaks. Their smooth output reduces worst-case errors but inflates mean absolute error
across high-demand periods.
</div>
<div class="insight">
📌 <b>Lag 1 dominance</b> — over 60% of test hours fall within ±0.5 kW for
XGBoost, indicating strong short-term autocorrelation in the dataset that the model
exploits effectively.
</div>
""", unsafe_allow_html=True)
