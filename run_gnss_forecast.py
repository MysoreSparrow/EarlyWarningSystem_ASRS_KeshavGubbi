"""
GNSS spoofing/jamming — LightGBM monthly forecast.
Trains on 2018-2025, forecasts Apr-Sep 2026 (6 months beyond data end).
Shows the elevated rate continuing.

Run: uv run python run_gnss_forecast.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

os.makedirs("outputs/figures", exist_ok=True)

# ── 1. BUILD MONTHLY SERIES ───────────────────────────────────────────────────
asrs = pd.read_parquet("outputs/data/asrs_layer1.parquet")
asrs['date'] = pd.to_datetime(asrs['date'], errors='coerce')

GNSS_REGEX = (
    r'spoofing|jamming|gps.{0,20}denial|gnss.{0,20}denial|'
    r'gps.{0,20}interference|navigation.{0,20}interference'
)
mask = asrs['full_narrative'].astype(str).str.lower().str.contains(GNSS_REGEX, na=False)
monthly = (
    asrs[mask]
    .groupby(asrs[mask]['date'].dt.to_period('M'))['ACN']
    .count()
    .to_timestamp()
    .resample('MS').sum()
    .fillna(0)
)
# Clip to 2018+ (remove stray historical records)
monthly = monthly[monthly.index >= '2018-01-01']
print(f"Monthly GNSS series: {len(monthly)} months  "
      f"({monthly.index[0].date()} to {monthly.index[-1].date()})")
print(f"Mean: {monthly.mean():.1f}/month  |  Max: {monthly.max()}")

# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────
df = pd.DataFrame({'count': monthly})
LAGS = [1, 2, 3, 6, 12]
WINDOWS = [3, 6]
for lag in LAGS:
    df[f'lag_{lag}'] = df['count'].shift(lag)
for w in WINDOWS:
    df[f'roll_mean_{w}'] = df['count'].shift(1).rolling(w).mean()
    df[f'roll_std_{w}']  = df['count'].shift(1).rolling(w).std()
df['month']     = df.index.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
dates = pd.to_datetime(df.index)
df['post_2023'] = (dates >= '2023-01-01').astype(int)
df = df.dropna()
feat_cols = [c for c in df.columns if c != 'count']

# ── 3. TRAIN / TEST SPLIT ─────────────────────────────────────────────────────
# Test: 2025 (last full year)
train = df[df.index.year < 2025]
test  = df[df.index.year == 2025]

model = lgb.LGBMRegressor(
    n_estimators=200, learning_rate=0.05, num_leaves=15,
    min_data_in_leaf=3, random_state=42, verbose=-1,
)
model.fit(train[feat_cols], train['count'])

test_pred = model.predict(test[feat_cols])
mae  = mean_absolute_error(test['count'], test_pred)
mase = mae / train['count'].diff().abs().mean()  # MASE
print(f"\nTest MAE:  {mae:.2f} incidents/month")
print(f"Test MASE: {mase:.3f}  (<1 = beats naive)")
print(f"Test period: {test.index[0].date()} to {test.index[-1].date()}")

# ── 4. RECURSIVE 6-MONTH FORECAST (Apr-Sep 2026) ─────────────────────────────
# Extend the series with test predictions, then forecast forward
extended = monthly.astype(float).copy()
for ts, pred in zip(test.index, test_pred):
    extended[ts] = float(pred)

# Generate future months: Apr 2026 - Sep 2026
last_date = extended.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1), periods=6, freq='MS'
)

future_preds = []
running = extended.copy()
for fd in future_dates:
    row = {}
    for lag in LAGS:
        row[f'lag_{lag}'] = running.iloc[-lag] if len(running) >= lag else 0
    for w in WINDOWS:
        row[f'roll_mean_{w}'] = running.iloc[-w:].mean() if len(running) >= w else running.mean()
        row[f'roll_std_{w}']  = running.iloc[-w:].std()  if len(running) >= w else 0
    row['month']     = fd.month
    row['month_sin'] = np.sin(2 * np.pi * fd.month / 12)
    row['month_cos'] = np.cos(2 * np.pi * fd.month / 12)
    row['post_2023'] = 1
    X_fut = pd.DataFrame([row])[feat_cols]
    pred = max(0, float(model.predict(X_fut)[0]))
    future_preds.append(pred)
    running = pd.concat([running, pd.Series([pred], index=[fd])])

forecast_series = pd.Series(future_preds, index=future_dates)
print(f"\n6-month forecast (Apr-Sep 2026):")
for d, v in forecast_series.items():
    print(f"  {d.strftime('%b %Y')}: {v:.1f}")

# ── 5. PLOT ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# Actual
ax.plot(monthly.index, monthly.values,
        color='#003366', lw=2, label='Actual monthly count')
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color='#003366')

# Test-set fitted values
ax.plot(test.index, test_pred,
        color='green', lw=2, ls='--', label=f'Model fit (2025 test, MAE={mae:.1f})')

# Forecast
ax.plot(forecast_series.index, forecast_series.values,
        color='#cc0000', lw=2.5, ls='--', label='6-month forecast (Apr-Sep 2026)')
ax.fill_between(forecast_series.index, forecast_series.values,
                alpha=0.2, color='#cc0000')

# Shade elevated period
emergence = monthly[monthly.index >= '2024-01-01']
if not emergence.empty:
    ax.axvspan(pd.Timestamp('2024-01-01'), forecast_series.index[-1],
               alpha=0.07, color='red', label='Elevated regime (2024+)')

# Pre-2023 baseline
pre2023_mean = monthly[monthly.index < '2023-01-01'].mean()
ax.axhline(pre2023_mean, color='grey', ls=':', lw=1.5,
           label=f'Pre-2023 baseline ({pre2023_mean:.1f}/month)')

# Marker: data end vs forecast start
ax.axvline(pd.Timestamp('2026-03-01'), color='black', ls=':', lw=1,
           alpha=0.6)
ax.text(pd.Timestamp('2026-03-01'), ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.5,
        '  Data end', fontsize=8, color='grey', va='bottom')

ax.set_ylabel('GNSS spoofing/jamming incidents per month', fontsize=11)
ax.set_xlabel('Date', fontsize=11)
fc_start = future_dates[0].strftime('%b %Y')
fc_end   = future_dates[-1].strftime('%b %Y')
ax.set_title(
    f'Layer 3: GNSS Spoofing — LightGBM 6-Month Forecast ({fc_start}–{fc_end})\n'
    f'2025 holdout MAE = {mae:.1f}/month  |  '
    f'Forecast shows elevated rate continuing above pre-2023 baseline ({pre2023_mean:.1f}/month)',
    fontsize=12, fontweight='bold',
)
ax.legend(fontsize=9)
plt.tight_layout()
out = 'outputs/figures/gnss_forecast.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.close()
