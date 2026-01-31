"""
TSA_ch10_comprehensive
======================
Comprehensive Time Series Analysis Review

This script demonstrates a complete workflow:
1. Data exploration and visualization
2. Stationarity testing
3. Model identification (ACF/PACF)
4. Model estimation and selection
5. Diagnostics
6. Forecasting
7. Model comparison

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

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

def save_fig(name):
    """Save figure with transparent background."""
    plt.savefig(f'{name}.pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig(f'{name}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf")

print("=" * 70)
print("COMPREHENSIVE TIME SERIES ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. Data Generation and Exploration
# =============================================================================
np.random.seed(42)
n = 300

print("\n" + "=" * 70)
print("STEP 1: DATA EXPLORATION")
print("=" * 70)

# Generate ARIMA(1,1,1) process
y = np.zeros(n)
eps = np.random.normal(0, 1, n)
phi, theta = 0.7, -0.4

# Start with integrated process
for t in range(2, n):
    y[t] = y[t-1] + phi * (y[t-1] - y[t-2]) + eps[t] + theta * eps[t-1]

# Add trend
y = y + 0.1 * np.arange(n) + 50

dates = pd.date_range('2000-01-01', periods=n, freq='M')
ts = pd.Series(y, index=dates, name='Value')

print(f"\n   Data: {n} monthly observations")
print(f"   Period: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
print(f"\n   Descriptive Statistics:")
print(f"     Mean: {ts.mean():.2f}")
print(f"     Std Dev: {ts.std():.2f}")
print(f"     Min: {ts.min():.2f}")
print(f"     Max: {ts.max():.2f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time series plot
axes[0, 0].plot(ts.index, ts.values, color='#1A3A6E', linewidth=1)
axes[0, 0].set_title('Original Time Series', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Value')

# Histogram
axes[0, 1].hist(ts.values, bins=30, color='#1A3A6E', alpha=0.7, edgecolor='white', density=True)
x = np.linspace(ts.min(), ts.max(), 100)
axes[0, 1].plot(x, stats.norm.pdf(x, ts.mean(), ts.std()), 'r-', lw=2)
axes[0, 1].set_title('Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Value')

# ACF
plot_acf(ts, lags=30, ax=axes[1, 0], color='#1A3A6E')
axes[1, 0].set_title('Autocorrelation Function (ACF)', fontweight='bold')

# PACF
plot_pacf(ts, lags=30, ax=axes[1, 1], color='#1A3A6E', method='ywm')
axes[1, 1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold')

plt.tight_layout()
plt.savefig('ch10_exploration.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch10_exploration.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n   Saved: ch10_exploration.pdf")

# =============================================================================
# 2. Stationarity Testing
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: STATIONARITY TESTING")
print("=" * 70)

# ADF test on original series
adf_orig = adfuller(ts)
print(f"\n   Original Series:")
print(f"     ADF Statistic: {adf_orig[0]:.4f}")
print(f"     p-value: {adf_orig[1]:.4f}")
print(f"     Critical Values: 1%: {adf_orig[4]['1%']:.3f}, 5%: {adf_orig[4]['5%']:.3f}")
print(f"     Conclusion: {'Stationary' if adf_orig[1] < 0.05 else 'Non-stationary'}")

# First difference
ts_diff = ts.diff().dropna()
adf_diff = adfuller(ts_diff)
print(f"\n   First Difference:")
print(f"     ADF Statistic: {adf_diff[0]:.4f}")
print(f"     p-value: {adf_diff[1]:.4f}")
print(f"     Conclusion: {'Stationary' if adf_diff[1] < 0.05 else 'Non-stationary'}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].plot(ts.index, ts.values, color='#1A3A6E', linewidth=1)
axes[0, 0].set_title(f'Original Series (ADF p={adf_orig[1]:.4f})', fontweight='bold')
axes[0, 0].set_ylabel('Value')

plot_acf(ts, lags=30, ax=axes[0, 1], color='#1A3A6E')
axes[0, 1].set_title('ACF: Original (Slow Decay = Non-stationary)', fontweight='bold')

axes[1, 0].plot(ts_diff.index, ts_diff.values, color='#DC3545', linewidth=1)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[1, 0].set_title(f'First Difference (ADF p={adf_diff[1]:.4f})', fontweight='bold')
axes[1, 0].set_ylabel('ΔValue')

plot_acf(ts_diff, lags=30, ax=axes[1, 1], color='#DC3545')
axes[1, 1].set_title('ACF: Differenced (Fast Decay = Stationary)', fontweight='bold')

plt.tight_layout()
plt.savefig('ch10_stationarity.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch10_stationarity.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n   Saved: ch10_stationarity.pdf")

# =============================================================================
# 3. Model Identification
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: MODEL IDENTIFICATION")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(ts_diff, lags=20, ax=axes[0], color='#1A3A6E')
axes[0].set_title('ACF of Differenced Series', fontweight='bold')

plot_pacf(ts_diff, lags=20, ax=axes[1], color='#1A3A6E', method='ywm')
axes[1].set_title('PACF of Differenced Series', fontweight='bold')

plt.tight_layout()
plt.savefig('ch10_identification.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch10_identification.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n   ACF/PACF Pattern Analysis:")
print("     - ACF: Significant spike at lag 1, then cuts off → MA(1)")
print("     - PACF: Decay pattern → AR component")
print("     - Suggested model: ARIMA(1,1,1)")
print("\n   Saved: ch10_identification.pdf")

# =============================================================================
# 4. Model Estimation and Selection
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: MODEL ESTIMATION AND SELECTION")
print("=" * 70)

# Fit multiple models
models_to_try = [
    (0, 1, 1), (1, 1, 0), (1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2)
]

results_list = []
for order in models_to_try:
    try:
        model = ARIMA(ts, order=order)
        result = model.fit()
        results_list.append({
            'Model': f'ARIMA{order}',
            'AIC': result.aic,
            'BIC': result.bic,
            'LogL': result.llf
        })
    except:
        pass

results_df = pd.DataFrame(results_list)
print("\n   Model Comparison:")
print(results_df.to_string(index=False))

best_idx = results_df['AIC'].idxmin()
best_model_name = results_df.loc[best_idx, 'Model']
print(f"\n   Best Model (AIC): {best_model_name}")

# Fit best model
best_order = (1, 1, 1)
best_model = ARIMA(ts, order=best_order)
best_result = best_model.fit()

print(f"\n   {best_model_name} Parameters:")
for i, (name, val) in enumerate(zip(best_result.param_names, best_result.params)):
    print(f"     {name}: {val:.4f}")

# =============================================================================
# 5. Model Diagnostics
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: MODEL DIAGNOSTICS")
print("=" * 70)

residuals = best_result.resid

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals time plot
axes[0, 0].plot(residuals.index, residuals.values, color='#1A3A6E', linewidth=0.8)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 0].axhline(y=2*residuals.std(), color='gray', linestyle=':', alpha=0.5)
axes[0, 0].axhline(y=-2*residuals.std(), color='gray', linestyle=':', alpha=0.5)
axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residual')

# Histogram
axes[0, 1].hist(residuals.values, bins=30, color='#1A3A6E', alpha=0.7,
                edgecolor='white', density=True)
x = np.linspace(residuals.min(), residuals.max(), 100)
axes[0, 1].plot(x, stats.norm.pdf(x, 0, residuals.std()), 'r-', lw=2, label='Normal')
axes[0, 1].set_title('Residual Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# ACF of residuals
plot_acf(residuals, lags=20, ax=axes[1, 0], color='#1A3A6E')
axes[1, 0].set_title('ACF of Residuals', fontweight='bold')

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontweight='bold')

plt.tight_layout()
plt.savefig('ch10_diagnostics.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch10_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n   Saved: ch10_diagnostics.pdf")

# Ljung-Box test
lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print("\n   Ljung-Box Test for Residual Autocorrelation:")
print(f"     Lag 10: Q = {lb_test['lb_stat'].iloc[0]:.2f}, p-value = {lb_test['lb_pvalue'].iloc[0]:.4f}")
print(f"     Lag 20: Q = {lb_test['lb_stat'].iloc[1]:.2f}, p-value = {lb_test['lb_pvalue'].iloc[1]:.4f}")

if lb_test['lb_pvalue'].iloc[0] > 0.05:
    print("     Conclusion: No significant autocorrelation (model adequate)")
else:
    print("     Conclusion: Significant autocorrelation (model may be inadequate)")

# Normality test
jb_stat, jb_pval = stats.jarque_bera(residuals)
print(f"\n   Jarque-Bera Normality Test:")
print(f"     Statistic: {jb_stat:.2f}, p-value: {jb_pval:.4f}")

# =============================================================================
# 6. Forecasting
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: FORECASTING")
print("=" * 70)

# Split data for validation
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

# Fit on training data
train_model = ARIMA(train, order=best_order)
train_result = train_model.fit()

# Forecast
forecast_steps = len(test)
forecast = train_result.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Calculate accuracy metrics
actual = test.values
predicted = forecast_mean.values

mae = np.mean(np.abs(actual - predicted))
rmse = np.sqrt(np.mean((actual - predicted)**2))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

print(f"\n   Forecast Accuracy (Out-of-sample):")
print(f"     MAE: {mae:.4f}")
print(f"     RMSE: {rmse:.4f}")
print(f"     MAPE: {mape:.2f}%")

# Visualization
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(train.index, train.values, color='#1A3A6E', linewidth=1, label='Training')
ax.plot(test.index, test.values, color='#2E7D32', linewidth=1.5, label='Actual')
ax.plot(test.index, forecast_mean, color='#DC3545', linewidth=2,
        linestyle='--', label='Forecast')
ax.fill_between(test.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                color='#DC3545', alpha=0.2, label='95% CI')

# Visual separator between training and test/forecast
split_point = train.index[-1]
ax.axvline(x=split_point, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
y_pos = ax.get_ylim()[1] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
ax.text(split_point, y_pos, '  Forecast ', fontsize=9, ha='left', va='top',
        color='black', fontweight='bold', alpha=0.8)

ax.set_title(f'{best_model_name} Forecast (RMSE={rmse:.2f})', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)

plt.tight_layout()
plt.savefig('ch10_forecast.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch10_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n   Saved: ch10_forecast.pdf")

# =============================================================================
# 7. Summary Flowchart
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: ANALYSIS WORKFLOW SUMMARY")
print("=" * 70)

print("""
   ┌─────────────────────────────────────┐
   │         1. DATA EXPLORATION         │
   │    Visualize, check for patterns    │
   └──────────────────┬──────────────────┘
                      │
   ┌──────────────────▼──────────────────┐
   │       2. STATIONARITY TESTING       │
   │         ADF, KPSS tests             │
   └──────────────────┬──────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
   ┌──────▼──────┐         ┌──────▼──────┐
   │ Stationary  │         │Non-stationary│
   └──────┬──────┘         └──────┬──────┘
          │                       │
          │              ┌────────▼────────┐
          │              │   Difference    │
          │              │   (d times)     │
          │              └────────┬────────┘
          │                       │
          └───────────┬───────────┘
                      │
   ┌──────────────────▼──────────────────┐
   │       3. MODEL IDENTIFICATION       │
   │         ACF/PACF analysis           │
   └──────────────────┬──────────────────┘
                      │
   ┌──────────────────▼──────────────────┐
   │        4. MODEL ESTIMATION          │
   │    Compare AIC/BIC, select best     │
   └──────────────────┬──────────────────┘
                      │
   ┌──────────────────▼──────────────────┐
   │         5. DIAGNOSTICS              │
   │   Ljung-Box, normality, residuals   │
   └──────────────────┬──────────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
   ┌──────▼──────┐         ┌──────▼──────┐
   │   Adequate  │         │ Inadequate  │
   └──────┬──────┘         └──────┬──────┘
          │                       │
          │              ┌────────▼────────┐
          │              │ Revise model    │
          │              │ (go to step 3)  │
          │              └─────────────────┘
          │
   ┌──────▼───────────────────────────────┐
   │           6. FORECASTING             │
   │    Generate predictions with CI      │
   └──────────────────────────────────────┘
""")

print("\n" + "=" * 70)
print("COMPREHENSIVE ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch10_exploration.pdf: Data exploration")
print("  - ch10_stationarity.pdf: Stationarity analysis")
print("  - ch10_identification.pdf: Model identification")
print("  - ch10_diagnostics.pdf: Residual diagnostics")
print("  - ch10_forecast.pdf: Forecasting results")
