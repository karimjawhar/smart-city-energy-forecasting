"""
EDA — Exploratory Data Analysis
UCI Household Electric Power Consumption
Generates publication-quality plots saved to outputs/figures/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ── Config ────────────────────────────────────────────────────────
RAW_PATH   = "data/raw/household_power_consumption.txt"
FIG_DIR    = "outputs/figures"
TARGET     = "Global_active_power"
os.makedirs(FIG_DIR, exist_ok=True)

STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8fafc",
    "axes.edgecolor":    "#cbd5e1",
    "axes.grid":         True,
    "grid.color":        "#e2e8f0",
    "grid.linewidth":    0.7,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
}
plt.rcParams.update(STYLE)

BLUE   = "#3b82f6"
RED    = "#ef4444"
ORANGE = "#f59e0b"

# ── Load & resample ───────────────────────────────────────────────
print("Loading raw data...")
df = pd.read_csv(RAW_PATH, sep=";", na_values=["?","NA",""], low_memory=False)
dt = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str),
                    dayfirst=True, errors="coerce")
df = df.drop(columns=["Date","Time"])
df.insert(0, "datetime", dt)
df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

hourly = df[TARGET].resample("1h").mean().interpolate("time").ffill().bfill()
hourly.name = TARGET
print(f"  Hourly series: {len(hourly):,} observations  "
      f"({hourly.index.min().date()} → {hourly.index.max().date()})")

# ════════════════════════════════════════════════════════════════════
# 1. Full time series + 30-day rolling mean
# ════════════════════════════════════════════════════════════════════
print("Plotting 1/6 — full time series...")
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(hourly.index, hourly.values, color=BLUE, lw=0.4, alpha=0.6, label="Hourly")
roll = hourly.rolling(24*30).mean()
ax.plot(roll.index, roll.values, color=RED, lw=2, label="30-day rolling mean")
ax.set_title("Global Active Power — Full Time Series (2006–2010)")
ax.set_ylabel("Power (kW)")
ax.set_xlabel("Date")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
plt.xticks(rotation=30)
ax.legend(framealpha=0.9)
plt.tight_layout()
out = os.path.join(FIG_DIR, "eda_full_time_series.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ════════════════════════════════════════════════════════════════════
# 2. ACF + PACF
# ════════════════════════════════════════════════════════════════════
print("Plotting 2/6 — ACF / PACF...")
sample = hourly.dropna().iloc[:5000]   # use first 5000 pts for speed
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(sample,  ax=axes[0], lags=72, color=BLUE, title="Autocorrelation (ACF) — lags 0–72h")
plot_pacf(sample, ax=axes[1], lags=72, color=RED,  title="Partial Autocorrelation (PACF) — lags 0–72h",
          method="ywm")
for ax in axes:
    ax.axvline(x=24, color=ORANGE, ls="--", lw=1.2, label="lag 24 (1 day)")
    ax.axvline(x=48, color=ORANGE, ls=":",  lw=1.0, label="lag 48 (2 days)")
    ax.legend(fontsize=9)
    ax.set_xlabel("Lag (hours)")
plt.tight_layout()
out = os.path.join(FIG_DIR, "eda_acf_pacf.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ════════════════════════════════════════════════════════════════════
# 3. Hourly distribution (hour of day)
# ════════════════════════════════════════════════════════════════════
print("Plotting 3/6 — hourly distribution...")
hdf = hourly.to_frame()
hdf["hour"] = hdf.index.hour
groups = [hdf.loc[hdf["hour"]==h, TARGET].dropna().values for h in range(24)]

fig, ax = plt.subplots(figsize=(14, 5))
bp = ax.boxplot(groups, patch_artist=True, showfliers=False,
                medianprops=dict(color=RED, lw=2),
                whiskerprops=dict(color="#64748b"),
                capprops=dict(color="#64748b"),
                boxprops=dict(facecolor="#dbeafe", color=BLUE))
ax.set_xticks(range(1, 25))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right")
ax.set_title("Distribution of Global Active Power by Hour of Day")
ax.set_ylabel("Power (kW)")
ax.set_xlabel("Hour of Day")
plt.tight_layout()
out = os.path.join(FIG_DIR, "eda_hourly_distribution.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ════════════════════════════════════════════════════════════════════
# 4. Day-of-week distribution
# ════════════════════════════════════════════════════════════════════
print("Plotting 4/6 — day-of-week distribution...")
hdf["dow"] = hdf.index.dayofweek
dow_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
groups_dow = [hdf.loc[hdf["dow"]==d, TARGET].dropna().values for d in range(7)]

fig, ax = plt.subplots(figsize=(9, 5))
bp = ax.boxplot(groups_dow, patch_artist=True, showfliers=False,
                medianprops=dict(color=RED, lw=2),
                whiskerprops=dict(color="#64748b"),
                capprops=dict(color="#64748b"),
                boxprops=dict(facecolor="#dbeafe", color=BLUE))
ax.set_xticklabels(dow_labels)
ax.set_title("Distribution of Global Active Power by Day of Week")
ax.set_ylabel("Power (kW)")
ax.set_xlabel("Day of Week")
plt.tight_layout()
out = os.path.join(FIG_DIR, "eda_dow_distribution.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ════════════════════════════════════════════════════════════════════
# 5. Monthly distribution
# ════════════════════════════════════════════════════════════════════
print("Plotting 5/6 — monthly distribution...")
hdf["month"] = hdf.index.month
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
groups_mon = [hdf.loc[hdf["month"]==m, TARGET].dropna().values for m in range(1,13)]

fig, ax = plt.subplots(figsize=(12, 5))
bp = ax.boxplot(groups_mon, patch_artist=True, showfliers=False,
                medianprops=dict(color=RED, lw=2),
                whiskerprops=dict(color="#64748b"),
                capprops=dict(color="#64748b"),
                boxprops=dict(facecolor="#dbeafe", color=BLUE))
ax.set_xticklabels(month_labels)
ax.set_title("Distribution of Global Active Power by Month")
ax.set_ylabel("Power (kW)")
ax.set_xlabel("Month")
plt.tight_layout()
out = os.path.join(FIG_DIR, "eda_monthly_distribution.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ════════════════════════════════════════════════════════════════════
# 6. Average daily profile (mean + 1-std band)
# ════════════════════════════════════════════════════════════════════
print("Plotting 6/6 — average daily load profile...")
profile = hdf.groupby("hour")[TARGET].agg(["mean","std"])

fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(profile.index,
                profile["mean"] - profile["std"],
                profile["mean"] + profile["std"],
                color=BLUE, alpha=0.15, label="±1 std")
ax.plot(profile.index, profile["mean"], color=BLUE, lw=2.5, marker="o",
        markersize=4, label="Mean")
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right")
ax.set_title("Average Daily Load Profile (Mean ± 1 SD)")
ax.set_ylabel("Power (kW)")
ax.set_xlabel("Hour of Day")
ax.legend()
plt.tight_layout()
out = os.path.join(FIG_DIR, "eda_daily_profile.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

print("\nAll EDA plots saved to outputs/figures/")
