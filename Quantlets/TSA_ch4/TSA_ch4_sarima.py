"""
TSA_ch4_sarima
==============
SARIMA Models: Seasonal ARIMA Analysis

This script demonstrates:
- Seasonal decomposition
- SARIMA model specification: ARIMA(p,d,q)(P,D,Q)[s]
- Seasonal differencing
- Model estimation and diagnostics
- Forecasting with seasonality

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
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
print("SARIMA MODELS: ARIMA(p,d,q)(P,D,Q)[s]")
print("=" * 70)

# =============================================================================
# 1. Generate Seasonal Data (Airline-like pattern)
# =============================================================================
np.random.seed(42)
n = 144  # 12 years of monthly data
t = np.arange(n)

# Components
trend = 100 + 0.5 * t
seasonal = 20 * np.sin(2 * np.pi * t / 12) + 10 * np.cos(2 * np.pi * t / 12)
noise = np.random.normal(0, 5, n)
y = trend + seasonal + noise

dates = pd.date_range('2010-01-01', periods=n, freq='M')
ts = pd.Series(y, index=dates, name='Passengers')

print("\n1. SEASONAL TIME SERIES")
print("-" * 40)
print(f"   Data: {n} monthly observations")
print(f"   Period: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
print(f"   Seasonal period: s = 12 (monthly)")

# =============================================================================
# 2. Seasonal Decomposition
# =============================================================================
print("\n2. SEASONAL DECOMPOSITION")
print("-" * 40)

decomposition = seasonal_decompose(ts, model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
decomposition.observed.plot(ax=axes[0], color='#1A3A6E')
axes[0].set_ylabel('Observed')
axes[0].set_title('Seasonal Decomposition (Additive)', fontweight='bold')

decomposition.trend.plot(ax=axes[1], color='#2E7D32')
axes[1].set_ylabel('Trend')

decomposition.seasonal.plot(ax=axes[2], color='#E67E22')
axes[2].set_ylabel('Seasonal')

decomposition.resid.plot(ax=axes[3], color='#666666')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Date')

plt.tight_layout()
plt.savefig('ch4_decomposition.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch4_decomposition.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch4_decomposition.pdf")

# =============================================================================
# 3. Stationarity Testing
# =============================================================================
print("\n3. STATIONARITY ANALYSIS")
print("-" * 40)

# Original series
adf_orig = adfuller(ts)
print(f"   Original series:")
print(f"     ADF statistic: {adf_orig[0]:.4f}")
print(f"     p-value: {adf_orig[1]:.4f}")
print(f"     Stationary: {'Yes' if adf_orig[1] < 0.05 else 'No'}")

# First difference
ts_diff = ts.diff().dropna()
adf_diff = adfuller(ts_diff)
print(f"\n   After first difference (d=1):")
print(f"     ADF statistic: {adf_diff[0]:.4f}")
print(f"     p-value: {adf_diff[1]:.4f}")

# Seasonal difference
ts_seasonal_diff = ts.diff(12).dropna()
adf_seasonal = adfuller(ts_seasonal_diff)
print(f"\n   After seasonal difference (D=1, s=12):")
print(f"     ADF statistic: {adf_seasonal[0]:.4f}")
print(f"     p-value: {adf_seasonal[1]:.4f}")

# =============================================================================
# 4. ACF/PACF Analysis
# =============================================================================
print("\n4. ACF/PACF PATTERNS")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Original series ACF/PACF
plot_acf(ts, lags=36, ax=axes[0, 0], color='#1A3A6E')
axes[0, 0].set_title('ACF: Original Series', fontweight='bold')

plot_pacf(ts, lags=36, ax=axes[0, 1], color='#1A3A6E', method='ywm')
axes[0, 1].set_title('PACF: Original Series', fontweight='bold')

# After differencing
ts_both_diff = ts.diff().diff(12).dropna()
plot_acf(ts_both_diff, lags=36, ax=axes[1, 0], color='#DC3545')
axes[1, 0].set_title('ACF: After d=1, D=1', fontweight='bold')

plot_pacf(ts_both_diff, lags=36, ax=axes[1, 1], color='#DC3545', method='ywm')
axes[1, 1].set_title('PACF: After d=1, D=1', fontweight='bold')

plt.tight_layout()
plt.savefig('ch4_acf_pacf.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch4_acf_pacf.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch4_acf_pacf.pdf")
print("   Seasonal spikes at lags 12, 24, 36 indicate seasonality")

# =============================================================================
# 5. SARIMA Model Estimation
# =============================================================================
print("\n5. SARIMA MODEL ESTIMATION")
print("-" * 40)

# Fit SARIMA(1,1,1)(1,1,1)[12]
model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)

print(f"\n   Model: SARIMA(1,1,1)(1,1,1)[12]")
print(f"\n   Parameters:")
print(f"     AR(1) coefficient (φ): {results.params['ar.L1']:.4f}")
print(f"     MA(1) coefficient (θ): {results.params['ma.L1']:.4f}")
print(f"     Seasonal AR(1) (Φ): {results.params['ar.S.L12']:.4f}")
print(f"     Seasonal MA(1) (Θ): {results.params['ma.S.L12']:.4f}")
print(f"\n   Model Fit:")
print(f"     AIC: {results.aic:.2f}")
print(f"     BIC: {results.bic:.2f}")
print(f"     Log-likelihood: {results.llf:.2f}")

# =============================================================================
# 6. Model Diagnostics
# =============================================================================
print("\n6. MODEL DIAGNOSTICS")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Residuals plot
residuals = results.resid
axes[0, 0].plot(residuals, color='#1A3A6E', linewidth=0.8)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('Residuals', fontweight='bold')
axes[0, 0].set_xlabel('Date')

# Histogram
axes[0, 1].hist(residuals, bins=30, color='#1A3A6E', alpha=0.7, edgecolor='white', density=True)
x = np.linspace(residuals.min(), residuals.max(), 100)
from scipy import stats
axes[0, 1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r-', lw=2)
axes[0, 1].set_title('Residual Distribution', fontweight='bold')

# ACF of residuals
plot_acf(residuals, lags=24, ax=axes[1, 0], color='#1A3A6E')
axes[1, 0].set_title('ACF of Residuals', fontweight='bold')

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontweight='bold')

plt.tight_layout()
plt.savefig('ch4_diagnostics.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch4_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch4_diagnostics.pdf")

# Ljung-Box test
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[12, 24], return_df=True)
print(f"\n   Ljung-Box Test:")
print(f"     Lag 12: Q = {lb_test['lb_stat'].iloc[0]:.2f}, p-value = {lb_test['lb_pvalue'].iloc[0]:.4f}")
print(f"     Lag 24: Q = {lb_test['lb_stat'].iloc[1]:.2f}, p-value = {lb_test['lb_pvalue'].iloc[1]:.4f}")

# =============================================================================
# 7. Forecasting
# =============================================================================
print("\n7. FORECASTING")
print("-" * 40)

# Forecast 24 months ahead
forecast_steps = 24
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Create forecast dates
forecast_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(months=1),
                                periods=forecast_steps, freq='M')

fig, ax = plt.subplots(figsize=(14, 6))

# Historical data
ax.plot(ts.index, ts, color='#1A3A6E', linewidth=1.5, label='Historical')

# Forecast
ax.plot(forecast_dates, forecast_mean, color='#DC3545', linewidth=2,
        linestyle='--', label='Forecast')
ax.fill_between(forecast_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                color='#DC3545', alpha=0.2, label='95% CI')

# Visual separator between historical and forecast
split_point = ts.index[-1]
ax.axvline(x=split_point, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
y_pos = ax.get_ylim()[1] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
ax.text(split_point, y_pos, '  Forecast ', fontsize=9, ha='left', va='top',
        color='black', fontweight='bold', alpha=0.8)

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('SARIMA(1,1,1)(1,1,1)[12] Forecast', fontweight='bold', fontsize=14)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch4_forecast.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch4_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch4_forecast.pdf")
print(f"\n   Forecast horizon: {forecast_steps} months")
print(f"   Next month forecast: {forecast_mean.iloc[0]:.2f}")
print(f"   95% CI: [{forecast_ci.iloc[0, 0]:.2f}, {forecast_ci.iloc[0, 1]:.2f}]")

# =============================================================================
# 8. Model Comparison
# =============================================================================
print("\n8. MODEL COMPARISON")
print("-" * 40)

models_to_try = [
    ((1, 1, 0), (1, 1, 0, 12), "SARIMA(1,1,0)(1,1,0)[12]"),
    ((0, 1, 1), (0, 1, 1, 12), "SARIMA(0,1,1)(0,1,1)[12]"),
    ((1, 1, 1), (1, 1, 1, 12), "SARIMA(1,1,1)(1,1,1)[12]"),
    ((2, 1, 1), (1, 1, 1, 12), "SARIMA(2,1,1)(1,1,1)[12]"),
]

results_list = []
for order, seasonal_order, name in models_to_try:
    try:
        model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
        res = model.fit(disp=False)
        results_list.append({
            'Model': name,
            'AIC': res.aic,
            'BIC': res.bic,
            'LogL': res.llf
        })
    except:
        pass

comparison_df = pd.DataFrame(results_list)
print(comparison_df.to_string(index=False))
print(f"\n   Best model (lowest AIC): {comparison_df.loc[comparison_df['AIC'].idxmin(), 'Model']}")

print("\n" + "=" * 70)
print("SARIMA ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch4_decomposition.pdf: Seasonal decomposition")
print("  - ch4_acf_pacf.pdf: ACF/PACF analysis")
print("  - ch4_diagnostics.pdf: Model diagnostics")
print("  - ch4_forecast.pdf: Forecast with confidence intervals")
