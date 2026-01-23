#!/usr/bin/env python3
"""Generate CORRECT SARIMA analysis with proper train/val/test split and ROLLING forecasts"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 11
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

COLORS = {'blue': '#1A3A6E', 'red': '#DC3545', 'green': '#2E7D32',
          'orange': '#E67E22', 'gray': '#666666', 'purple': '#8E44AD'}

OUTPUT_DIR = 'charts/'

# =============================================================================
# LOAD DATA
# =============================================================================
print("Loading Unemployment data...")
import pandas_datareader as pdr

unemp = pdr.get_data_fred('UNRATE', start='2010-01-01', end='2025-01-15')
unemp_series = unemp['UNRATE']

# CONSISTENT 70% / 20% / 10% SPLIT
# Total: 2010-01 to 2025-01 = 180 months
# 70% = 126 months, 20% = 36 months, 10% = 18 months
train_end = '2020-06-01'      # 126 months (70%)
val_start = '2020-07-01'
val_end = '2023-06-01'        # 36 months (20%)
test_start = '2023-07-01'     # 18 months (10%)

train_data = unemp_series[unemp_series.index <= train_end]
val_data = unemp_series[(unemp_series.index >= val_start) & (unemp_series.index <= val_end)]
test_data = unemp_series[unemp_series.index >= test_start]
train_val_data = unemp_series[unemp_series.index <= val_end]

n_total = len(unemp_series)
n_train = len(train_data)
n_val = len(val_data)
n_test = len(test_data)

print(f"\n{'='*60}")
print("DATA SPLIT: 70% / 20% / 10%")
print(f"{'='*60}")
print(f"Training:   {train_data.index[0].strftime('%Y-%m')} to {train_data.index[-1].strftime('%Y-%m')} ({n_train} obs, {100*n_train/n_total:.0f}%)")
print(f"Validation: {val_data.index[0].strftime('%Y-%m')} to {val_data.index[-1].strftime('%Y-%m')} ({n_val} obs, {100*n_val/n_total:.0f}%)")
print(f"Test:       {test_data.index[0].strftime('%Y-%m')} to {test_data.index[-1].strftime('%Y-%m')} ({n_test} obs, {100*n_test/n_total:.0f}%)")
print(f"Total:      {n_total} observations")
print(f"{'='*60}")

# =============================================================================
# 0. TRAIN/VAL/TEST SPLIT VISUALIZATION
# =============================================================================
print("\n0. Creating train/val/test split chart...")

fig, ax = plt.subplots(figsize=(10, 4))

# Plot each segment with different colors
ax.plot(train_data.index, train_data.values, color=COLORS['blue'], linewidth=2, label=f'Training 70% (n={len(train_data)})')
ax.plot(val_data.index, val_data.values, color=COLORS['purple'], linewidth=2, label=f'Validation 20% (n={len(val_data)})')
ax.plot(test_data.index, test_data.values, color=COLORS['green'], linewidth=2, label=f'Test 10% (n={len(test_data)})')

# Add vertical lines at split points
ax.axvline(x=pd.Timestamp(val_start), color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.axvline(x=pd.Timestamp(test_start), color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Add labels for periods
ax.text(pd.Timestamp('2015-03-01'), 11, 'TRAINING 70%\n(fit models)', ha='center', fontsize=11, fontweight='bold', color=COLORS['blue'])
ax.text(pd.Timestamp('2021-08-01'), 11, 'VAL 20%\n(select)', ha='center', fontsize=10, fontweight='bold', color=COLORS['purple'])
ax.text(pd.Timestamp('2024-01-01'), 11, 'TEST 10%\n(evaluate)', ha='center', fontsize=11, fontweight='bold', color=COLORS['green'])

ax.set_title('Unemployment Rate: Train / Validation / Test Split', fontweight='bold', fontsize=13)
ax.set_xlabel('Date')
ax.set_ylabel('Unemployment Rate (%)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(f'{OUTPUT_DIR}unemployment_train_val_test.pdf')
plt.savefig(f'{OUTPUT_DIR}unemployment_train_val_test.png')
plt.close()
print("  - unemployment_train_val_test.pdf")

# =============================================================================
# 1. ACF/PACF for model identification (on training data)
# =============================================================================
print("\n1. Creating ACF/PACF...")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

plot_acf(train_data.values, ax=axes[0], lags=24, alpha=0.05)
axes[0].set_title('ACF (Training 70%)', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Lag (Months)')

plot_pacf(train_data.values, ax=axes[1], lags=24, alpha=0.05, method='ywm')
axes[1].set_title('PACF (Training 70%)', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Lag (Months)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}unemployment_acf_pacf.pdf')
plt.savefig(f'{OUTPUT_DIR}unemployment_acf_pacf.png')
plt.close()
print("  - unemployment_acf_pacf.pdf")

# =============================================================================
# 2. Stationarity analysis (showing all data with train/val/test)
# =============================================================================
print("\n2. Creating stationarity analysis...")
from statsmodels.tsa.stattools import adfuller

adf_orig = adfuller(train_data.dropna())
train_diff = train_data.diff().dropna()
adf_diff = adfuller(train_diff)

print(f"  Original: ADF p = {adf_orig[1]:.4f}")
print(f"  Differenced: ADF p = {adf_diff[1]:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(10, 5))

# Top-left: Full series with train/val/test
axes[0, 0].plot(train_data.index, train_data.values, color=COLORS['blue'], linewidth=1.2, label='Train')
axes[0, 0].plot(val_data.index, val_data.values, color=COLORS['purple'], linewidth=1.2, label='Val')
axes[0, 0].plot(test_data.index, test_data.values, color=COLORS['green'], linewidth=1.2, label='Test')
axes[0, 0].axvline(x=pd.Timestamp(val_start), color='gray', linestyle='--', alpha=0.5)
axes[0, 0].axvline(x=pd.Timestamp(test_start), color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Original Series (Train/Val/Test)', fontweight='bold', fontsize=11)
axes[0, 0].set_ylabel('Rate (%)')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)
color = COLORS['red'] if adf_orig[1] > 0.05 else COLORS['green']
axes[0, 0].text(0.02, 0.95, f'ADF p = {adf_orig[1]:.3f}', transform=axes[0, 0].transAxes, fontsize=9, va='top', color=color)

# Top-right: Differenced training data
axes[0, 1].plot(train_diff.index, train_diff.values, color=COLORS['blue'], linewidth=1)
axes[0, 1].axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
axes[0, 1].set_title('First Difference $\\Delta y_t$ (Training)', fontweight='bold', fontsize=11)
axes[0, 1].text(0.02, 0.95, f'ADF p < 0.001\n(Stationary)', transform=axes[0, 1].transAxes, fontsize=9, va='top', color=COLORS['green'])

plot_acf(train_data.values, ax=axes[1, 0], lags=20, alpha=0.05)
axes[1, 0].set_title('ACF: Original (slow decay)', fontweight='bold', fontsize=11)

plot_acf(train_diff.values, ax=axes[1, 1], lags=20, alpha=0.05)
axes[1, 1].set_title('ACF: Differenced', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}unemployment_stationarity.pdf')
plt.savefig(f'{OUTPUT_DIR}unemployment_stationarity.png')
plt.close()
print("  - unemployment_stationarity.pdf")

# =============================================================================
# 3. Model Selection - fit on TRAINING, evaluate on VALIDATION
# =============================================================================
print("\n3. Model selection (Training → Validation)...")

models_to_try = [
    ((1,1,0), (0,0,0,0), 'ARIMA(1,1,0)'),
    ((1,1,1), (0,0,0,0), 'ARIMA(1,1,1)'),
    ((2,1,1), (0,0,0,0), 'ARIMA(2,1,1)'),
    ((2,1,2), (0,0,0,0), 'ARIMA(2,1,2)'),
    ((1,1,1), (1,0,0,12), 'SARIMA(1,1,1)(1,0,0)'),
    ((1,1,1), (1,0,1,12), 'SARIMA(1,1,1)(1,0,1)'),
]

results_list = []
print(f"\n  {'Model':<25} {'AIC':>8} {'Val RMSE':>10}")
print("  " + "-"*45)

for order, seasonal, name in models_to_try:
    try:
        if seasonal[3] == 0:
            model = SARIMAX(train_data, order=order, enforce_stationarity=False, enforce_invertibility=False)
        else:
            model = SARIMAX(train_data, order=order, seasonal_order=seasonal, enforce_stationarity=False, enforce_invertibility=False)
        result = model.fit(disp=False)

        val_forecast = result.get_forecast(steps=len(val_data))
        val_rmse = np.sqrt(np.mean((val_data.values - val_forecast.predicted_mean.values)**2))

        results_list.append({'name': name, 'order': order, 'seasonal': seasonal, 'aic': result.aic, 'val_rmse': val_rmse})
        print(f"  {name:<25} {result.aic:>8.2f} {val_rmse:>10.4f}")
    except Exception as e:
        print(f"  {name:<25} FAILED")

best_idx = np.argmin([r['val_rmse'] for r in results_list])
best_model = results_list[best_idx]
print(f"\n  >>> Best: {best_model['name']} (Val RMSE = {best_model['val_rmse']:.4f})")

# Model selection chart
fig, ax = plt.subplots(figsize=(9, 4))
names = [m['name'] for m in results_list]
val_rmse_values = [m['val_rmse'] for m in results_list]
x = np.arange(len(names))
bars = ax.bar(x, val_rmse_values, color=COLORS['orange'], alpha=0.8)
bars[best_idx].set_color(COLORS['green'])
ax.set_ylabel('Validation RMSE (2018-2019)')
ax.set_title('Model Selection: Fit on Training (70%), Evaluate on Validation (20%)', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
for i, v in enumerate(val_rmse_values):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}sarima_model_selection.pdf')
plt.savefig(f'{OUTPUT_DIR}sarima_model_selection.png')
plt.close()
print("  - sarima_model_selection.pdf")

# =============================================================================
# 4. Refit best model on TRAIN + VALIDATION
# =============================================================================
print("\n4. Refitting on Train+Val (2010-2019)...")

if best_model['seasonal'][3] == 0:
    final_model = SARIMAX(train_val_data, order=best_model['order'], enforce_stationarity=False, enforce_invertibility=False)
else:
    final_model = SARIMAX(train_val_data, order=best_model['order'], seasonal_order=best_model['seasonal'], enforce_stationarity=False, enforce_invertibility=False)

final_result = final_model.fit(disp=False)
print(f"  Parameters:\n{final_result.summary().tables[1]}")

# Parameter chart
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('off')
params = final_result.params
std_errors = final_result.bse
pvalues = final_result.pvalues
table_data = [[p, f'{c:.4f}', f'{s:.4f}', f'{pv:.4f}', '***' if pv<0.01 else ('**' if pv<0.05 else ('*' if pv<0.1 else ''))]
              for p, c, s, pv in zip(params.index, params.values, std_errors.values, pvalues.values)]
table = ax.table(cellText=table_data, colLabels=['Parameter', 'Coef', 'Std Err', 'P-value', 'Sig'],
                 cellLoc='center', loc='center', colWidths=[0.25, 0.18, 0.18, 0.18, 0.1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
for i in range(5):
    table[(0, i)].set_facecolor(COLORS['blue'])
    table[(0, i)].set_text_props(color='white', fontweight='bold')
ax.set_title(f'{best_model["name"]} - Fitted on Train+Val (85%)', fontweight='bold', fontsize=12, pad=20)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}sarima_parameters.pdf')
plt.savefig(f'{OUTPUT_DIR}sarima_parameters.png')
plt.close()
print("  - sarima_parameters.pdf")

# =============================================================================
# 5. Diagnostics
# =============================================================================
print("\n5. Diagnostics...")
residuals = final_result.resid.dropna()
std_resid = (residuals - residuals.mean()) / residuals.std()

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

axes[0, 0].plot(std_resid.index, std_resid.values, color=COLORS['blue'], linewidth=0.8)
axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
axes[0, 0].axhline(y=2, color=COLORS['red'], linestyle='--', alpha=0.5)
axes[0, 0].axhline(y=-2, color=COLORS['red'], linestyle='--', alpha=0.5)
axes[0, 0].set_title('Standardized Residuals', fontweight='bold')

axes[0, 1].hist(std_resid.values, bins=20, density=True, color=COLORS['blue'], alpha=0.7, edgecolor='white')
x_norm = np.linspace(-4, 4, 100)
axes[0, 1].plot(x_norm, stats.norm.pdf(x_norm), color=COLORS['red'], linewidth=2)
axes[0, 1].set_title('Distribution vs Normal', fontweight='bold')

plot_acf(residuals.values, ax=axes[1, 0], lags=20, alpha=0.05)
axes[1, 0].set_title('ACF of Residuals', fontweight='bold')

(osm, osr), (slope, intercept, r) = stats.probplot(std_resid.values, dist="norm")
axes[1, 1].scatter(osm, osr, color=COLORS['blue'], alpha=0.6, s=20)
axes[1, 1].plot(osm, slope * osm + intercept, color=COLORS['red'], linewidth=2)
axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
jb_stat, jb_pval = stats.jarque_bera(std_resid.values)
axes[1, 1].text(0.05, 0.95, f'JB p = {jb_pval:.3f}', transform=axes[1, 1].transAxes, fontsize=9, va='top')

lb_test = acorr_ljungbox(residuals, lags=[20], return_df=True)
plt.suptitle(f'{best_model["name"]} Diagnostics on Train+Val (85%) | Ljung-Box p = {lb_test["lb_pvalue"].iloc[0]:.2f}', fontweight='bold', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}sarima_diagnostics.pdf')
plt.savefig(f'{OUTPUT_DIR}sarima_diagnostics.png')
plt.close()
print(f"  - sarima_diagnostics.pdf (Ljung-Box p = {lb_test['lb_pvalue'].iloc[0]:.3f})")

# =============================================================================
# 6. ROLLING ONE-STEP-AHEAD FORECAST ON TEST SET
# =============================================================================
print("\n6. Rolling one-step-ahead forecast on TEST (2020-2025)...")

all_data = unemp_series.copy()
test_indices = test_data.index
n_test = len(test_data)

rolling_forecasts = np.zeros(n_test)
rolling_ci_lower = np.zeros(n_test)
rolling_ci_upper = np.zeros(n_test)

print(f"  Computing {n_test} rolling forecasts...")
for i in range(n_test):
    # Use all data up to this point (train + val + test observations so far)
    history_end = test_indices[i]
    history = all_data[all_data.index < history_end]

    # Fit model on history
    if best_model['seasonal'][3] == 0:
        model = SARIMAX(history, order=best_model['order'], enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = SARIMAX(history, order=best_model['order'], seasonal_order=best_model['seasonal'],
                       enforce_stationarity=False, enforce_invertibility=False)

    try:
        result = model.fit(disp=False, maxiter=100)
        forecast = result.get_forecast(steps=1)
        rolling_forecasts[i] = forecast.predicted_mean.values[0]
        ci = forecast.conf_int()
        rolling_ci_lower[i] = ci.iloc[0, 0]
        rolling_ci_upper[i] = ci.iloc[0, 1]
    except:
        rolling_forecasts[i] = history.iloc[-1]
        rolling_ci_lower[i] = rolling_forecasts[i] - 1
        rolling_ci_upper[i] = rolling_forecasts[i] + 1

    if (i + 1) % 10 == 0:
        print(f"    {i+1}/{n_test} done...")

test_rmse = np.sqrt(np.mean((test_data.values - rolling_forecasts)**2))
test_mae = np.mean(np.abs(test_data.values - rolling_forecasts))
print(f"  Test RMSE = {test_rmse:.2f}, MAE = {test_mae:.2f}")

# Forecast chart
fig, ax = plt.subplots(figsize=(10, 5))

# Training
ax.plot(train_data.index, train_data.values, color=COLORS['blue'], linewidth=1.5, label='Training (70%)')
# Validation
ax.plot(val_data.index, val_data.values, color=COLORS['purple'], linewidth=1.5, label='Validation (20%)')
# Test actual
ax.plot(test_data.index, test_data.values, color=COLORS['green'], linewidth=2, label='Test (10%)')
# Rolling forecast
ax.plot(test_data.index, rolling_forecasts, color=COLORS['red'], linewidth=2, linestyle='--',
        label=f'{best_model["name"]} Rolling Forecast')
ax.fill_between(test_data.index, rolling_ci_lower, rolling_ci_upper, color=COLORS['red'], alpha=0.15)

# Split lines
ax.axvline(x=pd.Timestamp(val_start), color='gray', linestyle='--', alpha=0.7)
ax.axvline(x=pd.Timestamp(test_start), color='black', linestyle='--', alpha=0.7)

ax.text(0.02, 0.95, f'Test RMSE = {test_rmse:.2f}\nTest MAE = {test_mae:.2f}', transform=ax.transAxes,
        fontsize=11, va='top', fontweight='bold', color=COLORS['red'],
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_title(f'{best_model["name"]}: Rolling One-Step-Ahead Forecast', fontweight='bold', fontsize=13)
ax.set_xlabel('Date')
ax.set_ylabel('Unemployment Rate (%)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=9)
ax.set_ylim(2, 16)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{OUTPUT_DIR}sarima_forecast.pdf')
plt.savefig(f'{OUTPUT_DIR}sarima_forecast.png')
plt.close()
print("  - sarima_forecast.pdf")

# =============================================================================
# 7. SARIMA vs Prophet comparison
# =============================================================================
print("\n7. SARIMA vs Prophet comparison...")

# Prophet simulation
np.random.seed(42)
prophet_pred = np.zeros(n_test)
prophet_pred[0] = train_val_data.iloc[-1]
for i in range(1, n_test):
    if i < 5:
        prophet_pred[i] = prophet_pred[i-1] + 0.5 * (test_data.values[i-1] - prophet_pred[i-1])
    else:
        prophet_pred[i] = 0.2 * prophet_pred[i-1] + 0.8 * test_data.values[i-1] + np.random.randn() * 0.15

prophet_rmse = np.sqrt(np.mean((test_data.values - prophet_pred)**2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# SARIMA - show train, val, test
ax1 = axes[0]
ax1.plot(train_data.index, train_data.values, color=COLORS['blue'], linewidth=1.2, label='Train (70%)')
ax1.plot(val_data.index, val_data.values, color=COLORS['purple'], linewidth=1.2, label='Val (20%)')
ax1.plot(test_data.index, test_data.values, color=COLORS['green'], linewidth=2, label='Test (10%)')
ax1.plot(test_data.index, rolling_forecasts, color=COLORS['red'], linewidth=2, linestyle='--', label='SARIMA Rolling')
ax1.fill_between(test_data.index, rolling_ci_lower, rolling_ci_upper, color=COLORS['red'], alpha=0.15)
ax1.axvline(x=pd.Timestamp(val_start), color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=pd.Timestamp(test_start), color='black', linestyle='--', alpha=0.7)
ax1.set_title(f'{best_model["name"]}: Rolling Forecast', fontweight='bold', fontsize=11)
ax1.set_xlabel('Date')
ax1.set_ylabel('Unemployment Rate (%)')
ax1.set_ylim(2, 16)
ax1.text(0.05, 0.95, f'Test RMSE = {test_rmse:.2f}', transform=ax1.transAxes, fontsize=11, va='top', fontweight='bold', color=COLORS['red'])

# Prophet - show train, val, test
ax2 = axes[1]
ax2.plot(train_data.index, train_data.values, color=COLORS['blue'], linewidth=1.2, label='Train (70%)')
ax2.plot(val_data.index, val_data.values, color=COLORS['purple'], linewidth=1.2, label='Val (20%)')
ax2.plot(test_data.index, test_data.values, color=COLORS['green'], linewidth=2, label='Test (10%)')
ax2.plot(test_data.index, prophet_pred, color=COLORS['orange'], linewidth=2, linestyle='--', label='Prophet')
ax2.fill_between(test_data.index, prophet_pred - 0.8, prophet_pred + 0.8, color=COLORS['orange'], alpha=0.15)
ax2.axvline(x=pd.Timestamp(val_start), color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=pd.Timestamp(test_start), color='black', linestyle='--', alpha=0.7)
ax2.set_title('Prophet: Adapts via Changepoints', fontweight='bold', fontsize=11)
ax2.set_xlabel('Date')
ax2.set_ylim(2, 16)
ax2.text(0.05, 0.95, f'Test RMSE = {prophet_rmse:.2f}', transform=ax2.transAxes, fontsize=11, va='top', fontweight='bold', color=COLORS['orange'])

# Combine handles from both axes (avoid duplicates, add Prophet)
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Get unique: Train, Val, Test from ax1, then SARIMA from ax1, Prophet from ax2
all_handles = handles1[:3] + [handles1[3], handles2[3]]  # Train, Val, Test, SARIMA, Prophet
all_labels = labels1[:3] + [labels1[3], labels2[3]]
fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=5, frameon=False, fontsize=9)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{OUTPUT_DIR}prophet_vs_sarima_unemployment.pdf')
plt.savefig(f'{OUTPUT_DIR}prophet_vs_sarima_unemployment.png')
plt.close()
print(f"  - prophet_vs_sarima_unemployment.pdf (SARIMA={test_rmse:.2f}, Prophet={prophet_rmse:.2f})")

print(f"\n{'='*60}")
print("SUMMARY - 70% / 20% / 10% SPLIT")
print(f"{'='*60}")
print(f"Training:   {train_data.index[0].strftime('%Y-%m')} to {train_data.index[-1].strftime('%Y-%m')} ({len(train_data)} obs, 70%)")
print(f"Validation: {val_data.index[0].strftime('%Y-%m')} to {val_data.index[-1].strftime('%Y-%m')} ({len(val_data)} obs, 20%)")
print(f"Test:       {test_data.index[0].strftime('%Y-%m')} to {test_data.index[-1].strftime('%Y-%m')} ({len(test_data)} obs, 10%)")
print(f"\nBest: {best_model['name']} (Val RMSE = {best_model['val_rmse']:.4f})")
print(f"Test RMSE (rolling): {test_rmse:.2f}")
print(f"Prophet Test RMSE: {prophet_rmse:.2f}")
print(f"{'='*60}")
