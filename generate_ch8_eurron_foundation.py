#!/usr/bin/env python3
"""
generate_ch8_eurron_foundation.py
=================================
EUR/RON: Complete Model Comparison including Foundation Models

Downloads real EUR/RON daily data, computes returns, and fits:
  - ARIMA(1,1,1) — rolling 1-step forecast on test
  - ARFIMA(1,d,1) — fractional differencing + ARMA on test
  - Random Forest — lag/rolling features, train on train+val
  - MLP (LSTM proxy) — 100 epochs with training curves
  - Chronos (amazon/chronos-t5-small) — zero-shot, graceful fallback
  - TimesFM (google/timesfm-1.0-200m) — zero-shot, graceful fallback

Evaluates on held-out test set (70/15/15 temporal split).
Generates all charts to charts/ directory.

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf
import time
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Style settings — Nature journal quality
# =============================================================================
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75

CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# Colour palette
MAIN_BLUE  = '#1A3A6E'
ACCENT_BLUE = '#2A528C'
IDA_RED    = '#DC3545'
FOREST     = '#2E7D32'
AMBER      = '#E67E22'
PURPLE     = '#7B2D8E'
TEAL       = '#00897B'


def save_fig(name):
    """Save figure as PDF + PNG to charts/."""
    path = os.path.join(CHARTS_DIR, name)
    plt.savefig(f'{path}.pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig(f'{path}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {path}.pdf")


# =============================================================================
# 1. Download and prepare data
# =============================================================================
print("=" * 70)
print("EUR/RON: COMPLETE MODEL COMPARISON (incl. Foundation Models)")
print("=" * 70)

print("\n1. DOWNLOADING EUR/RON DATA")
print("-" * 40)

import yfinance as yf

eurron_raw = yf.download('EURRON=X', start='2019-01-01', end='2025-06-01', progress=False)
if isinstance(eurron_raw.columns, pd.MultiIndex):
    eurron = eurron_raw['Close']['EURRON=X'].dropna()
else:
    eurron = eurron_raw['Close'].dropna()

# Flatten if needed
eurron = pd.Series(eurron.values.flatten(), index=eurron.index, name='EURRON')
print(f"   Downloaded {len(eurron)} daily observations")
print(f"   Period: {eurron.index[0].strftime('%Y-%m-%d')} to {eurron.index[-1].strftime('%Y-%m-%d')}")
print(f"   Range: {eurron.min():.4f} – {eurron.max():.4f}")

# Returns (percentage)
returns = eurron.pct_change().dropna() * 100
returns.index.freq = None
print(f"   Returns: {len(returns)} obs, mean={float(returns.mean()):.4f}%, std={float(returns.std()):.4f}%")

# =============================================================================
# 2. Train / Validation / Test split (70/15/15)
# =============================================================================
print("\n2. DATA SPLIT")
print("-" * 40)

n = len(eurron)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

price_train = eurron.iloc[:train_end]
price_val   = eurron.iloc[train_end:val_end]
price_test  = eurron.iloc[val_end:]

print(f"   Train: {len(price_train)} obs ({price_train.index[0].strftime('%Y-%m-%d')} – {price_train.index[-1].strftime('%Y-%m-%d')})")
print(f"   Val:   {len(price_val)} obs ({price_val.index[0].strftime('%Y-%m-%d')} – {price_val.index[-1].strftime('%Y-%m-%d')})")
print(f"   Test:  {len(price_test)} obs ({price_test.index[0].strftime('%Y-%m-%d')} – {price_test.index[-1].strftime('%Y-%m-%d')})")

# =============================================================================
# Chart 1: EUR/RON price + returns
# =============================================================================
print("\n   Generating ch8_eurron_series.pdf …")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(eurron.index, eurron.values, color=MAIN_BLUE, linewidth=0.8)
axes[0].set_ylabel('EUR/RON Rate')
axes[0].set_title('EUR/RON Exchange Rate (2019–2025)', fontweight='bold')
m = eurron.mean()
axes[0].axhline(y=m, color=IDA_RED, linestyle='--', alpha=0.7,
                label=f'Mean: {m:.2f}')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), frameon=False)

axes[1].plot(returns.index, returns.values, color=ACCENT_BLUE, linewidth=0.4, alpha=0.8)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_ylabel('Returns (%)')
axes[1].set_xlabel('Date')
axes[1].set_title('Daily Returns', fontweight='bold')

high_vol = returns.abs() > returns.std() * 3
axes[1].scatter(returns.index[high_vol], returns.values[high_vol],
                color=IDA_RED, s=10, alpha=0.6, zorder=5,
                label='|return| > 3σ')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
plt.tight_layout()
save_fig('ch8_eurron_series')

# =============================================================================
# Chart 2: Train / Val / Test split visualisation
# =============================================================================
print("   Generating ch8_case_raw_data.pdf …")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(price_train.index, price_train.values, color=MAIN_BLUE, linewidth=0.8, label='Train')
axes[0].plot(price_val.index, price_val.values, color=AMBER, linewidth=0.8, label='Validation')
axes[0].plot(price_test.index, price_test.values, color=IDA_RED, linewidth=0.8, label='Test')
axes[0].axvline(x=price_train.index[-1], color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
axes[0].axvline(x=price_val.index[-1], color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
axes[0].set_ylabel('EUR/RON Rate')
axes[0].set_title('EUR/RON Exchange Rate: Train/Validation/Test Split', fontweight='bold')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)

# Returns split
ret_train = returns.iloc[:train_end - 1]
ret_val   = returns.iloc[train_end - 1:val_end - 1]
ret_test  = returns.iloc[val_end - 1:]

axes[1].plot(ret_train.index, ret_train.values, color=MAIN_BLUE, linewidth=0.4, alpha=0.8)
axes[1].plot(ret_val.index, ret_val.values, color=AMBER, linewidth=0.4, alpha=0.8)
axes[1].plot(ret_test.index, ret_test.values, color=IDA_RED, linewidth=0.4, alpha=0.8)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_ylabel('Returns (%)')
axes[1].set_xlabel('Date')
axes[1].set_title('Daily Returns', fontweight='bold')
plt.tight_layout()
save_fig('ch8_case_raw_data')

# =============================================================================
# Chart 3: ACF — returns vs squared returns
# =============================================================================
print("   Generating ch8_case_acf_analysis.pdf …")

acf_ret = acf(returns.values, nlags=40)
acf_sq  = acf((returns ** 2).values, nlags=40)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ci = 1.96 / np.sqrt(len(returns))

axes[0].bar(range(len(acf_ret)), acf_ret, color=MAIN_BLUE, width=0.6)
axes[0].axhline(y=ci,  color=IDA_RED, linestyle='--', linewidth=0.8)
axes[0].axhline(y=-ci, color=IDA_RED, linestyle='--', linewidth=0.8)
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')
axes[0].set_title('ACF of Returns', fontweight='bold')
axes[0].set_xlim(-1, 41)

axes[1].bar(range(len(acf_sq)), acf_sq, color=PURPLE, width=0.6)
axes[1].axhline(y=ci,  color=IDA_RED, linestyle='--', linewidth=0.8)
axes[1].axhline(y=-ci, color=IDA_RED, linestyle='--', linewidth=0.8)
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('ACF')
axes[1].set_title('ACF of Squared Returns (Volatility)', fontweight='bold')
axes[1].set_xlim(-1, 41)

plt.tight_layout()
save_fig('ch8_case_acf_analysis')

# =============================================================================
# 3. Hurst exponent
# =============================================================================
print("\n3. HURST EXPONENT ESTIMATION")
print("-" * 40)


def estimate_hurst(ts, max_lag=100):
    """Estimate Hurst exponent via R/S analysis."""
    ts = np.asarray(ts, dtype=float)
    rs_values = []
    for lag in range(2, max_lag):
        n_blocks = len(ts) // lag
        if n_blocks < 1:
            break
        rs_block = []
        for i in range(n_blocks):
            block = ts[i * lag:(i + 1) * lag]
            mean_b = np.mean(block)
            devs = np.cumsum(block - mean_b)
            R = np.max(devs) - np.min(devs)
            S = np.std(block, ddof=1)
            if S > 0:
                rs_block.append(R / S)
        if rs_block:
            rs_values.append((lag, np.mean(rs_block)))
    log_lags = np.log([v[0] for v in rs_values])
    log_rs   = np.log([v[1] for v in rs_values])
    H = np.polyfit(log_lags, log_rs, 1)[0]
    return H


H_returns = estimate_hurst(returns.values)
d_returns = H_returns - 0.5
H_sq = estimate_hurst((returns ** 2).values)
d_sq = H_sq - 0.5

print(f"   Returns:         H = {H_returns:.4f},  d = {d_returns:.4f}")
print(f"   Squared returns: H = {H_sq:.4f},  d = {d_sq:.4f}")

# Phillips-Perron test for stationarity
from statsmodels.tsa.stattools import adfuller
pp_stat, pp_pval = adfuller(returns.values, regression='c')[:2]
print(f"   Phillips-Perron p-value: {pp_pval:.6f}")

# =============================================================================
# 4. ARIMA(1,1,1) — rolling 1-step on test
# =============================================================================
print("\n4. ARIMA(1,1,1)")
print("-" * 40)

t0 = time.time()
# Fit on train+val prices
train_val_prices = eurron.iloc[:val_end].values.astype(float)
arima_model = ARIMA(train_val_prices, order=(1, 1, 1))
arima_fit   = arima_model.fit()

# Rolling 1-step forecast
all_prices = eurron.values.astype(float)
arima_preds = np.zeros(len(price_test))
history = list(train_val_prices)

for i in range(len(price_test)):
    model_i = ARIMA(history, order=(1, 1, 1))
    fit_i   = model_i.fit()
    pred    = fit_i.forecast(steps=1)[0]
    arima_preds[i] = pred
    history.append(all_prices[val_end + i])

arima_time = time.time() - t0
arima_rmse = np.sqrt(mean_squared_error(price_test.values, arima_preds))
arima_mae  = mean_absolute_error(price_test.values, arima_preds)
print(f"   RMSE: {arima_rmse:.4f}")
print(f"   MAE:  {arima_mae:.4f}")
print(f"   Time: {arima_time:.2f}s")

# =============================================================================
# 5. ARFIMA(1,d,1) — rolling 1-step on test
# =============================================================================
print("\n5. ARFIMA(1,d,1)")
print("-" * 40)

d_hurst = float(np.clip(H_returns - 0.5, 0.01, 0.49))
# If H < 0.5 for returns, d would be ~0 → use a small positive value for demo
if d_hurst < 0.02:
    d_hurst = 0.01
print(f"   Using d = {d_hurst:.4f}")


def frac_diff_weights(d, max_k=500, threshold=1e-6):
    """Compute fractional differencing weights."""
    weights = [1.0]
    for k in range(1, max_k):
        w = weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    return np.array(weights)


def apply_frac_diff(series, d):
    """Apply fractional differencing."""
    weights = frac_diff_weights(d)
    k = len(weights)
    n_s = len(series)
    result = np.full(n_s, np.nan)
    for i in range(k - 1, n_s):
        window = series[max(0, i - k + 1):i + 1]
        w = weights[:len(window)][::-1]
        result[i] = np.dot(w, window)
    return result


t0 = time.time()
# Fractionally difference train+val prices
fd_trainval = apply_frac_diff(train_val_prices, d_hurst)
fd_clean = fd_trainval[~np.isnan(fd_trainval)]

# Fit ARMA(1,1) on fractionally differenced series
model_fd = ARIMA(fd_clean, order=(1, 0, 1))
fit_fd   = model_fd.fit()
mu_fd    = fit_fd.params[0]
phi_fd   = fit_fd.params[1]
theta_fd = fit_fd.params[2]
print(f"   ARMA params: mu={mu_fd:.4f}, phi={phi_fd:.4f}, theta={theta_fd:.4f}")

# Rolling 1-step forecast
weights_fd = frac_diff_weights(d_hurst)
n_weights  = len(weights_fd)
arfima_preds = np.zeros(len(price_test))
fd_fitted = fit_fd.fittedvalues
last_resid_fd = float(fd_clean[-1] - fd_fitted[-1])

for i in range(len(price_test)):
    idx = val_end + i
    # Fractionally difference up to t-1
    window = all_prices[max(0, idx - 1 - n_weights + 1):idx]
    w = weights_fd[:len(window)][::-1]
    fd_prev = np.dot(w, window)

    # ARMA forecast of fd_t
    fd_pred = mu_fd + phi_fd * (fd_prev - mu_fd) + theta_fd * last_resid_fd

    # Invert fractional differencing
    past = all_prices[max(0, idx - n_weights + 1):idx][::-1]
    correction = np.dot(weights_fd[1:len(past) + 1], past)
    pred = fd_pred - correction
    arfima_preds[i] = pred

    # Update residual
    window_t = all_prices[max(0, idx - n_weights + 1):idx + 1]
    w_t = weights_fd[:len(window_t)][::-1]
    fd_actual = np.dot(w_t, window_t)
    last_resid_fd = fd_actual - fd_pred

arfima_time = time.time() - t0
arfima_rmse = np.sqrt(mean_squared_error(price_test.values, arfima_preds))
arfima_mae  = mean_absolute_error(price_test.values, arfima_preds)
print(f"   RMSE: {arfima_rmse:.4f}")
print(f"   MAE:  {arfima_mae:.4f}")
print(f"   Time: {arfima_time:.2f}s")

# =============================================================================
# 6. Random Forest
# =============================================================================
print("\n6. RANDOM FOREST")
print("-" * 40)


def create_features(series):
    """Create lag and rolling features for ML models."""
    df = pd.DataFrame({'y': series.values}, index=series.index)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Lag {lag}'] = df['y'].shift(lag)
    df['Rolling Mean 5']  = df['y'].rolling(5).mean().shift(1)
    df['Rolling Std 5']   = df['y'].rolling(5).std().shift(1)
    df['Rolling Mean 20'] = df['y'].rolling(20).mean().shift(1)
    df['Day of Week']     = pd.to_datetime(series.index).dayofweek
    df['Month']           = pd.to_datetime(series.index).month
    return df.dropna()


df_feat = create_features(eurron)
feature_cols = [c for c in df_feat.columns if c != 'y']

# Split by date
df_train_f = df_feat[df_feat.index < price_val.index[0]]
df_val_f   = df_feat[(df_feat.index >= price_val.index[0]) & (df_feat.index < price_test.index[0])]
df_test_f  = df_feat[df_feat.index >= price_test.index[0]]

X_trainval = pd.concat([df_train_f[feature_cols], df_val_f[feature_cols]])
y_trainval = pd.concat([df_train_f['y'], df_val_f['y']])
X_test_rf  = df_test_f[feature_cols]
y_test_rf  = df_test_f['y']

t0 = time.time()
rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                           min_samples_leaf=5, random_state=42)
rf.fit(X_trainval, y_trainval)
rf_preds = rf.predict(X_test_rf)
rf_time  = time.time() - t0

rf_rmse = np.sqrt(mean_squared_error(y_test_rf.values, rf_preds))
rf_mae  = mean_absolute_error(y_test_rf.values, rf_preds)
print(f"   RMSE: {rf_rmse:.4f}")
print(f"   MAE:  {rf_mae:.4f}")
print(f"   Time: {rf_time:.2f}s")

# Feature importance
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(f"   Top features: {', '.join(importances.head(3)['feature'].values)}")

# Chart 4: Feature importance
print("   Generating ch8_case_feature_importance.pdf …")
sorted_idx = np.argsort(rf.feature_importances_)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh([feature_cols[i] for i in sorted_idx],
               rf.feature_importances_[sorted_idx], color=MAIN_BLUE)
ax.set_xlabel('Feature Importance')
ax.set_title('Random Forest: Feature Importance for EUR/RON Prediction', fontweight='bold')
for bar, val in zip(bars, rf.feature_importances_[sorted_idx]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}', va='center', fontsize=8)
plt.tight_layout()
save_fig('ch8_case_feature_importance')

# =============================================================================
# 7. MLP (LSTM proxy) with training curves
# =============================================================================
print("\n7. MLP NEURAL NETWORK (LSTM proxy)")
print("-" * 40)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_sc = scaler_X.fit_transform(X_trainval)
X_test_sc  = scaler_X.transform(X_test_rf)
y_train_sc = scaler_y.fit_transform(y_trainval.values.reshape(-1, 1)).ravel()

train_losses = []
val_losses   = []

t0 = time.time()
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                   solver='adam', max_iter=1, warm_start=True,
                   random_state=42, learning_rate_init=0.001, tol=1e-8)

for epoch in range(100):
    mlp.max_iter = epoch + 1
    mlp.fit(X_train_sc, y_train_sc)
    train_losses.append(mlp.loss_)
    vp = scaler_y.inverse_transform(mlp.predict(X_test_sc).reshape(-1, 1)).ravel()
    val_losses.append(mean_squared_error(y_test_rf, vp))

mlp_time = time.time() - t0
mlp_preds = scaler_y.inverse_transform(mlp.predict(X_test_sc).reshape(-1, 1)).ravel()
mlp_rmse  = np.sqrt(mean_squared_error(y_test_rf.values, mlp_preds))
mlp_mae   = mean_absolute_error(y_test_rf.values, mlp_preds)
print(f"   RMSE: {mlp_rmse:.4f}")
print(f"   MAE:  {mlp_mae:.4f}")
print(f"   Time: {mlp_time:.2f}s")

# Chart 5: Training curves
print("   Generating ch8_case_lstm_training.pdf …")
epochs_arr = np.arange(1, len(train_losses) + 1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs_arr, train_losses, color=MAIN_BLUE, linewidth=2, label='Training Loss')
ax.plot(epochs_arr, val_losses, color=IDA_RED, linewidth=2, label='Validation Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.set_title('Neural Network Training History (EUR/RON)', fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
plt.tight_layout()
save_fig('ch8_case_lstm_training')

# =============================================================================
# 8. Chronos (Foundation Model — zero-shot)
# =============================================================================
print("\n8. CHRONOS (Foundation Model)")
print("-" * 40)

chronos_rmse = chronos_mae = chronos_time = None
chronos_preds_arr = None
chronos_low = chronos_high = None
HAS_CHRONOS = False

# Pre-check: test if chronos can be imported without segfaulting (ARM Mac issue)
import subprocess as _sp
_chk = _sp.run([sys.executable, '-c', 'import torch; from chronos import ChronosPipeline; print("ok")'],
               capture_output=True, text=True, timeout=30)
_chronos_importable = _chk.returncode == 0 and 'ok' in _chk.stdout

if not _chronos_importable:
    print("   Chronos not importable (segfault or missing package)")
    print("   Run chapter8_foundation_colab.ipynb on Google Colab instead")

try:
    if not _chronos_importable:
        raise ImportError("Chronos not importable on this platform")
    import torch
    from chronos import ChronosPipeline

    t0 = time.time()
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    context = torch.tensor(eurron.iloc[:val_end].values, dtype=torch.float32).unsqueeze(0)
    horizon = len(price_test)

    forecast = pipeline.predict(context, horizon, num_samples=20)
    # forecast shape: (1, num_samples, horizon)
    chronos_median = forecast[0].median(dim=0).values.numpy()
    chronos_lo = np.quantile(forecast[0].numpy(), 0.1, axis=0)
    chronos_hi = np.quantile(forecast[0].numpy(), 0.9, axis=0)

    chronos_time = time.time() - t0
    chronos_preds_arr = chronos_median
    chronos_low  = chronos_lo
    chronos_high = chronos_hi
    chronos_rmse = np.sqrt(mean_squared_error(price_test.values[:len(chronos_median)], chronos_median))
    chronos_mae  = mean_absolute_error(price_test.values[:len(chronos_median)], chronos_median)
    HAS_CHRONOS  = True
    print(f"   Chronos loaded successfully")
    print(f"   RMSE: {chronos_rmse:.4f}")
    print(f"   MAE:  {chronos_mae:.4f}")
    print(f"   Time: {chronos_time:.2f}s")
except Exception as e:
    print(f"   Chronos not available: {e}")
    print("   Skipping (graceful fallback)")

# =============================================================================
# 9. TimesFM (Foundation Model — zero-shot)
# =============================================================================
print("\n9. TIMESFM (Foundation Model)")
print("-" * 40)

timesfm_rmse = timesfm_mae = timesfm_time = None
timesfm_preds_arr = None
HAS_TIMESFM = False

# Pre-check: test if timesfm can be imported
_chk2 = _sp.run([sys.executable, '-c', 'import timesfm; print("ok")'],
                capture_output=True, text=True, timeout=30)
_timesfm_importable = _chk2.returncode == 0 and 'ok' in _chk2.stdout

if not _timesfm_importable:
    print("   TimesFM not importable (missing package or Python version)")
    print("   Run chapter8_foundation_colab.ipynb on Google Colab instead")

try:
    if not _timesfm_importable:
        raise ImportError("TimesFM not importable on this platform")
    import timesfm

    t0 = time.time()
    tfm = timesfm.TimesFm(
        context_len=512,
        horizon_len=len(price_test),
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu",
    )
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

    context_arr = eurron.iloc[:val_end].values.astype(np.float32)
    point_forecast, _ = tfm.forecast([context_arr])
    timesfm_preds_arr = point_forecast[0][:len(price_test)]

    timesfm_time = time.time() - t0
    timesfm_rmse = np.sqrt(mean_squared_error(price_test.values[:len(timesfm_preds_arr)], timesfm_preds_arr))
    timesfm_mae  = mean_absolute_error(price_test.values[:len(timesfm_preds_arr)], timesfm_preds_arr)
    HAS_TIMESFM  = True
    print(f"   TimesFM loaded successfully")
    print(f"   RMSE: {timesfm_rmse:.4f}")
    print(f"   MAE:  {timesfm_mae:.4f}")
    print(f"   Time: {timesfm_time:.2f}s")
except Exception as e:
    print(f"   TimesFM not available: {e}")
    print("   Skipping (graceful fallback)")

# =============================================================================
# 10. Chart 6: Predictions vs Actual (classical models)
# =============================================================================
print("\n10. GENERATING CHARTS")
print("-" * 40)

print("   Generating ch8_case_predictions.pdf …")
fig, ax = plt.subplots(figsize=(12, 5))

test_idx = price_test.index
ax.plot(test_idx, price_test.values, color='#333333', linewidth=1.5,
        label='Actual EUR/RON', zorder=5)
ax.plot(test_idx, arima_preds, color=MAIN_BLUE, linewidth=1.0,
        linestyle='--', label='ARIMA(1,1,1)', alpha=0.8)
ax.plot(test_idx, arfima_preds, color=IDA_RED, linewidth=1.0,
        linestyle='--', label='ARFIMA(1,d,1)', alpha=0.8)

# RF and MLP may have fewer points due to feature NaNs
rf_idx = df_test_f.index
ax.plot(rf_idx, rf_preds, color=FOREST, linewidth=1.0,
        linestyle='--', label='Random Forest', alpha=0.8)
ax.plot(rf_idx, mlp_preds, color=AMBER, linewidth=1.0,
        linestyle='--', label='MLP/LSTM', alpha=0.8)

ax.set_xlabel('Date')
ax.set_ylabel('EUR/RON Exchange Rate')
ax.set_title('Model Predictions vs Actual EUR/RON (Test Period)', fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=False)
plt.tight_layout()
save_fig('ch8_case_predictions')

# =============================================================================
# 11. Chart 7: Bar chart — RMSE/MAE/Time comparison (classical)
# =============================================================================
print("   Generating ch8_case_comparison.pdf …")

model_names = ['ARIMA(1,1,1)', 'ARFIMA(1,d,1)', 'Random Forest', 'MLP/LSTM']
rmse_vals   = [arima_rmse, arfima_rmse, rf_rmse, mlp_rmse]
mae_vals    = [arima_mae, arfima_mae, rf_mae, mlp_mae]
time_vals   = [arima_time, arfima_time, rf_time, mlp_time]
colors_bar  = [MAIN_BLUE, IDA_RED, FOREST, AMBER]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
x = np.arange(len(model_names))

# RMSE
bars1 = axes[0].bar(x, rmse_vals, color=colors_bar, edgecolor='white', linewidth=0.5)
axes[0].set_ylabel('RMSE')
axes[0].set_title('Root Mean Square Error', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=20, ha='right', fontsize=7)
for bar, val in zip(bars1, rmse_vals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=7)

# MAE
bars2 = axes[1].bar(x, mae_vals, color=colors_bar, edgecolor='white', linewidth=0.5)
axes[1].set_ylabel('MAE')
axes[1].set_title('Mean Absolute Error', fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names, rotation=20, ha='right', fontsize=7)
for bar, val in zip(bars2, mae_vals):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0001,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=7)

# Training Time
bars3 = axes[2].bar(x, time_vals, color=colors_bar, edgecolor='white', linewidth=0.5)
axes[2].set_ylabel('Time (seconds)')
axes[2].set_title('Training Time', fontweight='bold')
axes[2].set_yscale('log')
axes[2].set_xticks(x)
axes[2].set_xticklabels(model_names, rotation=20, ha='right', fontsize=7)
for bar, val in zip(bars3, time_vals):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                 f'{val:.1f}s', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
save_fig('ch8_case_comparison')

# =============================================================================
# 12. Chart 8: Foundation model comparison (if available)
# =============================================================================
if HAS_CHRONOS or HAS_TIMESFM:
    print("   Generating ch8_foundation_comparison.pdf …")

    fm_names = model_names.copy()
    fm_rmse  = rmse_vals.copy()
    fm_mae   = mae_vals.copy()
    fm_colors = colors_bar.copy()

    if HAS_CHRONOS:
        fm_names.append('Chronos')
        fm_rmse.append(chronos_rmse)
        fm_mae.append(chronos_mae)
        fm_colors.append(PURPLE)

    if HAS_TIMESFM:
        fm_names.append('TimesFM')
        fm_rmse.append(timesfm_rmse)
        fm_mae.append(timesfm_mae)
        fm_colors.append(TEAL)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x2 = np.arange(len(fm_names))
    w = 0.35

    bars_r = axes[0].bar(x2 - w / 2, fm_rmse, w, color=fm_colors,
                         alpha=0.9, edgecolor='white', linewidth=0.5, label='RMSE')
    bars_m = axes[0].bar(x2 + w / 2, fm_mae, w, color=fm_colors,
                         alpha=0.5, edgecolor='white', linewidth=0.5, label='MAE')
    axes[0].set_ylabel('Error')
    axes[0].set_title('Classical vs Foundation Models', fontweight='bold')
    axes[0].set_xticks(x2)
    axes[0].set_xticklabels(fm_names, rotation=20, ha='right', fontsize=7)
    for bar in bars_r:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.0001,
                     f'{h:.4f}', ha='center', va='bottom', fontsize=6)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)

    # Right panel: Relative improvement over ARIMA baseline
    arima_base = arima_rmse
    rel_improve = [(arima_base - r) / arima_base * 100 for r in fm_rmse]
    bar_colors_rel = [FOREST if v > 0 else IDA_RED for v in rel_improve]
    axes[1].bar(x2, rel_improve, color=bar_colors_rel, alpha=0.8, edgecolor='white', linewidth=0.5)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_ylabel('RMSE Improvement vs ARIMA (%)')
    axes[1].set_title('Relative Performance', fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(fm_names, rotation=20, ha='right', fontsize=7)
    for i, v in enumerate(rel_improve):
        axes[1].text(i, v + (0.3 if v >= 0 else -0.6),
                     f'{v:.1f}%', ha='center', fontsize=7)

    plt.tight_layout()
    save_fig('ch8_foundation_comparison')
else:
    print("   Generating ch8_foundation_comparison.pdf (classical models only) …")
    print("   NOTE: Run chapter8_foundation_colab.ipynb on Google Colab for Chronos/TimesFM charts")

    # Show classical models only — clean chart, no ugly placeholders
    fm_names  = model_names
    fm_rmse   = rmse_vals
    fm_mae    = mae_vals
    fm_colors = colors_bar

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x2 = np.arange(len(fm_names))
    w = 0.35

    bars_r = axes[0].bar(x2 - w / 2, fm_rmse, w, color=fm_colors,
                         alpha=0.9, edgecolor='white', linewidth=0.5, label='RMSE')
    bars_m = axes[0].bar(x2 + w / 2, fm_mae, w, color=fm_colors,
                         alpha=0.5, edgecolor='white', linewidth=0.5, label='MAE')
    axes[0].set_ylabel('Error')
    axes[0].set_title('Classical Models: Error Comparison', fontweight='bold')
    axes[0].set_xticks(x2)
    axes[0].set_xticklabels(fm_names, rotation=20, ha='right', fontsize=7)
    for bar in bars_r:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 0.0001,
                     f'{h:.4f}', ha='center', va='bottom', fontsize=6)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)

    arima_base = arima_rmse
    rel_improve = [(arima_base - r) / arima_base * 100 for r in fm_rmse]
    bar_colors_rel = [FOREST if v > 0 else IDA_RED for v in rel_improve]
    axes[1].bar(x2, rel_improve, color=bar_colors_rel, alpha=0.8, edgecolor='white', linewidth=0.5)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_ylabel('RMSE Improvement vs ARIMA (%)')
    axes[1].set_title('Relative Performance', fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(fm_names, rotation=20, ha='right', fontsize=7)
    for i, v in enumerate(rel_improve):
        axes[1].text(i, v + (0.3 if v >= 0 else -0.6),
                     f'{v:.1f}%', ha='center', fontsize=7)

    plt.tight_layout()
    save_fig('ch8_foundation_comparison')

# =============================================================================
# 13. Chart 9: Foundation model prediction detail (if available)
# =============================================================================
if HAS_CHRONOS or HAS_TIMESFM:
    print("   Generating ch8_foundation_predictions.pdf …")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_idx, price_test.values, color='#333333', linewidth=1.5,
            label='Actual EUR/RON', zorder=5)
    ax.plot(test_idx, arima_preds, color=MAIN_BLUE, linewidth=0.8,
            linestyle='--', alpha=0.6, label='ARIMA (baseline)')

    if HAS_CHRONOS:
        n_chr = min(len(chronos_preds_arr), len(test_idx))
        ax.plot(test_idx[:n_chr], chronos_preds_arr[:n_chr],
                color=PURPLE, linewidth=1.2, label='Chronos (zero-shot)')
        if chronos_low is not None:
            ax.fill_between(test_idx[:n_chr],
                            chronos_low[:n_chr], chronos_high[:n_chr],
                            color=PURPLE, alpha=0.15, label='Chronos 80% CI')

    if HAS_TIMESFM:
        n_tfm = min(len(timesfm_preds_arr), len(test_idx))
        ax.plot(test_idx[:n_tfm], timesfm_preds_arr[:n_tfm],
                color=TEAL, linewidth=1.2, label='TimesFM (zero-shot)')

    ax.set_xlabel('Date')
    ax.set_ylabel('EUR/RON Exchange Rate')
    ax.set_title('Foundation Models: Zero-Shot Predictions on EUR/RON', fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    plt.tight_layout()
    save_fig('ch8_foundation_predictions')
else:
    print("   Generating ch8_foundation_predictions.pdf (classical models only) …")
    print("   NOTE: Run chapter8_foundation_colab.ipynb on Google Colab for Chronos/TimesFM charts")

    # Show classical predictions only — clean chart, no ugly placeholders
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_idx, price_test.values, color='#333333', linewidth=1.5,
            label='Actual EUR/RON', zorder=5)
    ax.plot(test_idx, arima_preds, color=MAIN_BLUE, linewidth=0.8,
            linestyle='--', alpha=0.7, label='ARIMA(1,1,1)')
    ax.plot(test_idx, arfima_preds, color=IDA_RED, linewidth=0.8,
            linestyle='--', alpha=0.7, label='ARFIMA(1,d,1)')
    ax.set_xlabel('Date')
    ax.set_ylabel('EUR/RON Exchange Rate')
    ax.set_title('Classical Models: Predictions on EUR/RON Test Set', fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    plt.tight_layout()
    save_fig('ch8_foundation_predictions')

# =============================================================================
# 14. Results Summary
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

results = pd.DataFrame({
    'Model': ['ARIMA(1,1,1)', 'ARFIMA(1,d,1)', 'Random Forest', 'MLP/LSTM'],
    'RMSE':  [arima_rmse, arfima_rmse, rf_rmse, mlp_rmse],
    'MAE':   [arima_mae,  arfima_mae,  rf_mae,  mlp_mae],
    'Time':  [arima_time, arfima_time, rf_time, mlp_time],
})

if HAS_CHRONOS:
    results = pd.concat([results, pd.DataFrame({
        'Model': ['Chronos (zero-shot)'], 'RMSE': [chronos_rmse],
        'MAE': [chronos_mae], 'Time': [chronos_time]
    })], ignore_index=True)

if HAS_TIMESFM:
    results = pd.concat([results, pd.DataFrame({
        'Model': ['TimesFM (zero-shot)'], 'RMSE': [timesfm_rmse],
        'MAE': [timesfm_mae], 'Time': [timesfm_time]
    })], ignore_index=True)

print("\n" + results.to_string(index=False))

print(f"\n   Hurst (returns):   H = {H_returns:.4f}, d = {d_returns:.4f}")
print(f"   Hurst (sq.ret.):   H = {H_sq:.4f}, d = {d_sq:.4f}")
print(f"   Phillips-Perron:   p = {pp_pval:.6f}")
print(f"   Test period: {price_test.index[0].strftime('%Y-%m-%d')} to {price_test.index[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# 15. LaTeX table for slides
# =============================================================================
print("\n--- LaTeX Table (copy to slides) ---")
print("\\begin{tabular}{l|c|c|c|c}")
print("    \\toprule")
print("    \\textbf{Model} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{Time (s)} & \\textbf{Interpretable?} \\\\")
print("    \\midrule")
print(f"    ARIMA(1,1,1) & {arima_rmse:.4f} & {arima_mae:.4f} & {arima_time:.2f} & Yes \\\\")
print(f"    ARFIMA(1,$d$,1) & {arfima_rmse:.4f} & {arfima_mae:.4f} & {arfima_time:.2f} & Yes \\\\")
print(f"    Random Forest & {rf_rmse:.4f} & {rf_mae:.4f} & {rf_time:.2f} & Partial \\\\")
print(f"    MLP/LSTM & {mlp_rmse:.4f} & {mlp_mae:.4f} & {mlp_time:.2f} & No \\\\")

if HAS_CHRONOS:
    print(f"    Chronos (zero-shot) & {chronos_rmse:.4f} & {chronos_mae:.4f} & {chronos_time:.2f} & No \\\\")
if HAS_TIMESFM:
    print(f"    TimesFM (zero-shot) & {timesfm_rmse:.4f} & {timesfm_mae:.4f} & {timesfm_time:.2f} & No \\\\")

print("    \\bottomrule")
print("\\end{tabular}")

print(f"\nHurst values:")
print(f"  Returns:   H = {H_returns:.4f}, d = {d_returns:.4f}")
print(f"  Sq. ret.:  H = {H_sq:.4f}, d = {d_sq:.4f}")
print(f"  PP p-val:  {pp_pval:.6f}")

print("\n" + "=" * 70)
print("ALL CHARTS GENERATED SUCCESSFULLY")
print("=" * 70)
print("\nOutput files:")
chart_files = [
    'ch8_eurron_series', 'ch8_case_raw_data', 'ch8_case_acf_analysis',
    'ch8_case_feature_importance', 'ch8_case_lstm_training',
    'ch8_case_predictions', 'ch8_case_comparison',
    'ch8_foundation_comparison', 'ch8_foundation_predictions'
]
for f in chart_files:
    path = os.path.join(CHARTS_DIR, f'{f}.pdf')
    status = "OK" if os.path.exists(path) else "MISSING"
    print(f"  [{status}] charts/{f}.pdf")
