"""
TSA_ch8_btc_model_comparison
=============================
Bitcoin: ARFIMA Estimation and Model Comparison (Real Data)

Downloads BTC-USD daily data, computes returns, and fits:
  - ARIMA(1,0,1)
  - ARFIMA(1,d,1) with Hurst-estimated d
  - Random Forest with lag/rolling features
  - LSTM neural network
Evaluates on held-out test set (70/15/15 temporal split).

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Chart style settings - Nature journal quality
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..', 'charts'))

def save_fig(name):
    """Save figure with transparent background."""
    # If name has no path separator, save to charts directory
    if os.sep not in name and '/' not in name:
        path = os.path.join(CHARTS_DIR, name)
    else:
        path = name
    plt.savefig(f'{path}.pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig(f'{path}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig(f'{path}.pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"   Saved: {path}.pdf")

print("=" * 70)
print("BITCOIN: ARFIMA ESTIMATION AND MODEL COMPARISON")
print("=" * 70)

# =============================================================================
# 1. Download and prepare data
# =============================================================================
print("\n1. DOWNLOADING BTC-USD DATA")
print("-" * 40)

import yfinance as yf

btc = yf.download('BTC-USD', start='2019-01-01', end='2025-01-01', progress=False)
# Handle multi-level columns from newer yfinance
if isinstance(btc.columns, pd.MultiIndex):
    btc = btc['Close']['BTC-USD'].dropna()
else:
    btc = btc['Close'].dropna()
print(f"   Downloaded {len(btc)} daily observations")
print(f"   Period: {btc.index[0].strftime('%Y-%m-%d')} to {btc.index[-1].strftime('%Y-%m-%d')}")

# Compute daily percentage returns
returns = btc.pct_change().dropna() * 100
returns.index.freq = None  # avoid frequency inference issues
print(f"   Returns: {len(returns)} observations")
print(f"   Mean: {float(returns.mean()):.4f}%, Std: {float(returns.std()):.4f}%")

# =============================================================================
# 2. Train / Validation / Test split (70/15/15)
# =============================================================================
print("\n2. DATA SPLIT")
print("-" * 40)

n = len(returns)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = returns.iloc[:train_end]
val = returns.iloc[train_end:val_end]
test = returns.iloc[val_end:]

print(f"   Train: {len(train)} obs ({train.index[0].strftime('%Y-%m-%d')} to {train.index[-1].strftime('%Y-%m-%d')})")
print(f"   Val:   {len(val)} obs ({val.index[0].strftime('%Y-%m-%d')} to {val.index[-1].strftime('%Y-%m-%d')})")
print(f"   Test:  {len(test)} obs ({test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')})")

# =============================================================================
# 3. Estimate Hurst exponent for fractional d
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
            block = ts[i*lag:(i+1)*lag]
            mean_block = np.mean(block)
            devs = np.cumsum(block - mean_block)
            R = np.max(devs) - np.min(devs)
            S = np.std(block, ddof=1)
            if S > 0:
                rs_block.append(R / S)
        if rs_block:
            rs_values.append((lag, np.mean(rs_block)))

    log_lags = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])
    H = np.polyfit(log_lags, log_rs, 1)[0]
    return H

H = estimate_hurst(train.values)
d_hurst = H - 0.5
d_hurst = float(np.clip(d_hurst, 0.01, 0.49))
print(f"   Hurst exponent: {H:.4f}")
print(f"   Fractional d = H - 0.5 = {d_hurst:.4f}")

# =============================================================================
# 4. Model 1: ARIMA(1,0,1)
# =============================================================================
print("\n4. ARIMA(1,0,1)")
print("-" * 40)

from statsmodels.tsa.arima.model import ARIMA

# Fit once on train+val, then apply fitted ARMA equation for 1-step forecasts
train_val = returns.iloc[:val_end].values.astype(float)

model_arima = ARIMA(train_val, order=(1, 0, 1))
fit_arima = model_arima.fit()
mu = fit_arima.params[0]  # const
phi = fit_arima.params[1]  # AR(1)
theta = fit_arima.params[2]  # MA(1)
print(f"   Params: mu={mu:.4f}, phi={phi:.4f}, theta={theta:.4f}")

# Rolling 1-step: use fitted params, update with each new observation
all_returns = returns.values.astype(float)
arima_preds = np.zeros(len(test))

# Compute residuals on train+val to initialize
fitted_vals = fit_arima.fittedvalues
last_resid = float(all_returns[val_end - 1] - fitted_vals[-1])

for i in range(len(test)):
    idx = val_end + i
    y_prev = all_returns[idx - 1]
    # ARMA(1,1): y_t = mu + phi*(y_{t-1} - mu) + theta*e_{t-1}
    pred = mu + phi * (y_prev - mu) + theta * last_resid
    arima_preds[i] = pred
    # Update residual
    last_resid = all_returns[idx] - pred

arima_rmse = np.sqrt(mean_squared_error(test.values, arima_preds))
arima_mae = mean_absolute_error(test.values, arima_preds)
print(f"   RMSE: {arima_rmse:.4f}")
print(f"   MAE:  {arima_mae:.4f}")

# =============================================================================
# 5. Model 2: ARFIMA(1,d,1)
# =============================================================================
print("\n5. ARFIMA(1,d,1)")
print("-" * 40)

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
    """Apply fractional differencing to a numpy array."""
    weights = frac_diff_weights(d)
    k = len(weights)
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(k - 1, n):
        window = series[max(0, i - k + 1):i + 1]
        w = weights[:len(window)][::-1]
        result[i] = np.dot(w, window)
    return result

# Fractionally difference the train+val data
fd_trainval = apply_frac_diff(train_val, d_hurst)
fd_clean = fd_trainval[~np.isnan(fd_trainval)]

# Fit ARMA(1,1) on fractionally differenced series
model_fd = ARIMA(fd_clean, order=(1, 0, 1))
fit_fd = model_fd.fit()
mu_fd = fit_fd.params[0]
phi_fd = fit_fd.params[1]
theta_fd = fit_fd.params[2]
print(f"   d = {d_hurst:.4f}")
print(f"   ARMA params: mu={mu_fd:.4f}, phi={phi_fd:.4f}, theta={theta_fd:.4f}")

# Rolling 1-step forecast: frac diff -> ARMA predict -> invert frac diff
weights_fd = frac_diff_weights(d_hurst)
n_weights = len(weights_fd)

arfima_preds = np.zeros(len(test))
fd_fitted = fit_fd.fittedvalues
last_resid_fd = float(fd_clean[-1] - fd_fitted[-1])

for i in range(len(test)):
    idx = val_end + i
    # Fractionally difference up to t-1 to get fd_{t-1}
    window = all_returns[max(0, idx - 1 - n_weights + 1):idx]
    w = weights_fd[:len(window)][::-1]
    fd_prev = np.dot(w, window)

    # ARMA forecast of fd_t
    fd_pred = mu_fd + phi_fd * (fd_prev - mu_fd) + theta_fd * last_resid_fd

    # Invert fractional differencing:
    # fd_t = sum_{k=0}^{K} w_k * y_{t-k}  =>  y_t = fd_t - sum_{k=1}^{K} w_k * y_{t-k}
    past = all_returns[max(0, idx - n_weights + 1):idx][::-1]
    correction = np.dot(weights_fd[1:len(past) + 1], past)
    pred = fd_pred - correction

    arfima_preds[i] = pred

    # Update residual: compute actual fd_t
    window_t = all_returns[max(0, idx - n_weights + 1):idx + 1]
    w_t = weights_fd[:len(window_t)][::-1]
    fd_actual = np.dot(w_t, window_t)
    last_resid_fd = fd_actual - fd_pred

arfima_rmse = np.sqrt(mean_squared_error(test.values, arfima_preds))
arfima_mae = mean_absolute_error(test.values, arfima_preds)
print(f"   RMSE: {arfima_rmse:.4f}")
print(f"   MAE:  {arfima_mae:.4f}")

# =============================================================================
# 6. Model 3: Random Forest
# =============================================================================
print("\n6. RANDOM FOREST")
print("-" * 40)

def create_features(series):
    """Create lag and rolling features for ML models."""
    df = pd.DataFrame({'y': series.values}, index=series.index)

    # Lag features
    for lag in [1, 2, 3, 7, 14]:
        df[f'lag_{lag}'] = df['y'].shift(lag)

    # Rolling features
    df['rolling_mean_7'] = df['y'].rolling(window=7).mean().shift(1)
    df['rolling_std_7'] = df['y'].rolling(window=7).std().shift(1)

    # Day of week
    df['day_of_week'] = pd.to_datetime(series.index).dayofweek

    return df.dropna()

df_all = create_features(returns)

# Split features
feature_cols = [c for c in df_all.columns if c != 'y']
df_train = df_all[df_all.index < val.index[0]]
df_val = df_all[(df_all.index >= val.index[0]) & (df_all.index < test.index[0])]
df_test = df_all[df_all.index >= test.index[0]]

X_train_rf = df_train[feature_cols]
y_train_rf = df_train['y']
X_val_rf = df_val[feature_cols]
y_val_rf = df_val['y']
X_test_rf = df_test[feature_cols]
y_test_rf = df_test['y']

# Train on train+val
X_trainval = pd.concat([X_train_rf, X_val_rf])
y_trainval = pd.concat([y_train_rf, y_val_rf])

rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42)
rf.fit(X_trainval, y_trainval)

rf_preds = rf.predict(X_test_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test_rf.values, rf_preds))
rf_mae = mean_absolute_error(y_test_rf.values, rf_preds)
print(f"   RMSE: {rf_rmse:.4f}")
print(f"   MAE:  {rf_mae:.4f}")

# Feature importance
importances = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(f"   Top features: {', '.join(importances.head(3)['feature'].values)}")

# =============================================================================
# 7. Model 4: LSTM (Neural Network)
# =============================================================================
print("\n7. LSTM (Neural Network)")
print("-" * 40)

from sklearn.neural_network import MLPRegressor
import subprocess, sys

seq_len = 14

def create_sequences(data, seq_length):
    """Create input sequences for LSTM."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Try TensorFlow LSTM via subprocess (avoids segfault in main process)
HAS_TF = False
try:
    ret = subprocess.run(
        [sys.executable, '-c', 'import tensorflow; print(tensorflow.__version__)'],
        capture_output=True, text=True, timeout=15)
    if ret.returncode == 0:
        HAS_TF = True
        print(f"   TensorFlow {ret.stdout.strip()} available")
except Exception:
    pass

if HAS_TF:
    # Run LSTM in subprocess to isolate TF
    lstm_script = f"""
import numpy as np, json, sys
np.random.seed(42)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = np.load('{os.path.abspath("_lstm_data.npz")}')
returns_vals = data['returns']
val_end = int(data['val_end'])
seq_len = 14

scaler = StandardScaler()
train_val_scaled = scaler.fit_transform(returns_vals[:val_end].reshape(-1, 1)).flatten()
test_data = returns_vals[val_end - seq_len:]
test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

def create_sequences(d, sl):
    X, y = [], []
    for i in range(sl, len(d)):
        X.append(d[i-sl:i])
        y.append(d[i])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_val_scaled, seq_len)
X_test, y_test = create_sequences(test_scaled, seq_len)
X_train = X_train.reshape(-1, seq_len, 1)
X_test = X_test.reshape(-1, seq_len, 1)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0, callbacks=[es])

preds_scaled = model.predict(X_test, verbose=0).flatten()
preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

rmse = float(np.sqrt(mean_squared_error(actual, preds)))
mae = float(mean_absolute_error(actual, preds))
np.savez('{os.path.abspath("_lstm_results.npz")}', preds=preds, actual=actual, rmse=rmse, mae=mae)
print(json.dumps({{'rmse': rmse, 'mae': mae}}))
"""
    # Save data for subprocess
    np.savez('_lstm_data.npz', returns=returns.values.astype(float),
             val_end=np.array(val_end))

    try:
        ret = subprocess.run([sys.executable, '-c', lstm_script],
                             capture_output=True, text=True, timeout=300)
        if ret.returncode == 0 and os.path.exists('_lstm_results.npz'):
            import json
            res = np.load('_lstm_results.npz')
            lstm_preds = res['preds']
            lstm_actual = res['actual']
            lstm_rmse = float(res['rmse'])
            lstm_mae = float(res['mae'])
            print(f"   TensorFlow LSTM completed")
        else:
            print(f"   TF subprocess failed, using MLP fallback")
            HAS_TF = False
    except Exception as e:
        print(f"   TF subprocess error: {e}, using MLP fallback")
        HAS_TF = False
    finally:
        for f in ['_lstm_data.npz', '_lstm_results.npz']:
            if os.path.exists(f):
                os.remove(f)

if not HAS_TF:
    print("   Using MLP neural network (LSTM proxy)")
    scaler = StandardScaler()
    ret_scaled = scaler.fit_transform(
        returns.values.reshape(-1, 1)).flatten()

    X_all, y_all = create_sequences(ret_scaled, seq_len)
    X_trainval_nn = X_all[:val_end - seq_len]
    y_trainval_nn = y_all[:val_end - seq_len]
    X_test_nn = X_all[val_end - seq_len:]
    y_test_nn = y_all[val_end - seq_len:]

    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200,
                       random_state=42, early_stopping=True)
    mlp.fit(X_trainval_nn, y_trainval_nn)
    lstm_preds_scaled = mlp.predict(X_test_nn)
    lstm_preds = scaler.inverse_transform(
        lstm_preds_scaled.reshape(-1, 1)).flatten()
    lstm_actual = scaler.inverse_transform(
        y_test_nn.reshape(-1, 1)).flatten()

    lstm_rmse = np.sqrt(mean_squared_error(lstm_actual, lstm_preds))
    lstm_mae = mean_absolute_error(lstm_actual, lstm_preds)
print(f"   RMSE: {lstm_rmse:.4f}")
print(f"   MAE:  {lstm_mae:.4f}")

# =============================================================================
# 8. Results summary
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

results = pd.DataFrame({
    'Model': ['ARIMA(1,0,1)', 'ARFIMA(1,d,1)', 'Random Forest', 'LSTM'],
    'RMSE': [arima_rmse, arfima_rmse, rf_rmse, lstm_rmse],
    'MAE': [arima_mae, arfima_mae, rf_mae, lstm_mae]
})
results['RMSE'] = results['RMSE'].round(2)
results['MAE'] = results['MAE'].round(2)

print("\n" + results.to_string(index=False))
print(f"\n   Hurst d = {d_hurst:.4f}")
print(f"   Test period: {test.index[0].strftime('%Y-%m-%d')} to {test.index[-1].strftime('%Y-%m-%d')}")

# =============================================================================
# 9. Comparison chart
# =============================================================================
print("\n9. GENERATING COMPARISON CHART")
print("-" * 40)

colors = ['#1A3A6E', '#DC3545', '#2E7D32', '#E67E22']
model_names = ['ARIMA(1,0,1)', 'ARFIMA(1,d,1)', 'Random Forest', 'LSTM']
rmse_vals = [arima_rmse, arfima_rmse, rf_rmse, lstm_rmse]
mae_vals = [arima_mae, arfima_mae, rf_mae, lstm_mae]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left panel: bar chart of RMSE and MAE
x = np.arange(len(model_names))
width = 0.35

bars1 = axes[0].bar(x - width/2, rmse_vals, width, label='RMSE', color=colors,
                     alpha=0.9, edgecolor='white', linewidth=0.5)
bars2 = axes[0].bar(x + width/2, mae_vals, width, label='MAE', color=colors,
                     alpha=0.5, edgecolor='white', linewidth=0.5)

axes[0].set_ylabel('Error')
axes[0].set_title('Model Comparison: RMSE and MAE', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=15, ha='right')

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., h + 0.02,
                 f'{h:.2f}', ha='center', va='bottom', fontsize=7)
for bar in bars2:
    h = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., h + 0.02,
                 f'{h:.2f}', ha='center', va='bottom', fontsize=7)

axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# Right panel: predictions vs actual (last 60 days of test)
n_show = min(60, len(test))
test_tail = test.iloc[-n_show:]

arima_tail = arima_preds[-n_show:]
arfima_tail = arfima_preds[-n_show:]
n_rf_show = min(n_show, len(rf_preds))
rf_tail = rf_preds[-n_rf_show:]
n_lstm_show = min(n_show, len(lstm_preds))

axes[1].plot(range(n_show), test_tail.values, color='black', linewidth=1.5,
             label='Actual', zorder=5)
axes[1].plot(range(n_show), arima_tail, color=colors[0], linewidth=0.8,
             alpha=0.7, label='ARIMA')
axes[1].plot(range(n_show), arfima_tail, color=colors[1], linewidth=0.8,
             alpha=0.7, label='ARFIMA')
axes[1].plot(range(n_show - n_rf_show, n_show), rf_tail, color=colors[2],
             linewidth=0.8, alpha=0.7, label='RF')
axes[1].plot(range(n_show - n_lstm_show, n_show), lstm_preds[-n_lstm_show:],
             color=colors[3], linewidth=0.8, alpha=0.7, label='LSTM')

axes[1].set_xlabel('Days (last 60 of test set)')
axes[1].set_ylabel('Return (%)')
axes[1].set_title('Test Set Predictions (Last 60 Days)', fontweight='bold')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=5, frameon=False)

plt.tight_layout()
save_fig('ch8_btc_model_comparison')

print("\n" + "=" * 70)
print("BITCOIN MODEL COMPARISON COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - charts/ch8_btc_model_comparison.pdf")
print("  - charts/ch8_btc_model_comparison.png")

# Print LaTeX table for slides
print("\n--- LaTeX Table (copy to slides) ---")
print("\\begin{tabular}{lccc}")
print("    \\toprule")
print("    \\textbf{Model} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{Interpretable?} \\\\")
print("    \\midrule")
print(f"    ARIMA(1,0,1) & {arima_rmse:.2f} & {arima_mae:.2f} & Yes \\\\")
print(f"    ARFIMA(1,$d$,1) & {arfima_rmse:.2f} & {arfima_mae:.2f} & Yes \\\\")
print(f"    Random Forest & {rf_rmse:.2f} & {rf_mae:.2f} & Partial \\\\")
print(f"    LSTM & {lstm_rmse:.2f} & {lstm_mae:.2f} & No \\\\")
print("    \\bottomrule")
print("\\end{tabular}")
print(f"\nNote: d = {d_hurst:.4f}, Test: {test.index[0].strftime('%Y-%m-%d')} -- {test.index[-1].strftime('%Y-%m-%d')}")
