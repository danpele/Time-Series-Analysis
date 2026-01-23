#!/usr/bin/env python3
"""Generate charts for Chapter 3: ARIMA Models"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 12

# Colors
BLUE = '#1A3A6E'
RED = '#DC3545'
GREEN = '#2E7D32'

np.random.seed(42)

# Try to get real data, fallback to synthetic
try:
    import pandas_datareader as pdr
    # US Real GDP (quarterly)
    gdp = pdr.get_data_fred('GDPC1', start='1990-01-01', end='2024-06-30')
    gdp_data = gdp['GDPC1'].dropna()
    data_source = "US Real GDP (Billions of Chained 2017 Dollars)"
except:
    # Synthetic GDP-like data
    n = 140
    dates = pd.date_range(start='1990-01-01', periods=n, freq='Q')
    trend = np.linspace(8000, 22000, n)
    noise = np.cumsum(np.random.normal(0, 100, n))
    gdp_data = pd.Series(trend + noise, index=dates)
    data_source = "US Real GDP (Simulated)"

print(f"Using: {data_source}")

# Chart 1: Non-stationary time series (GDP in levels)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(gdp_data.index, gdp_data.values, color=BLUE, linewidth=1.5)
ax.set_title('US Real GDP (Quarterly, 1990-2024): A Non-Stationary Time Series', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Billions of 2017 Dollars')
ax.text(0.02, 0.98, 'Source: FRED (GDPC1)', transform=ax.transAxes, fontsize=9,
        verticalalignment='top', style='italic', color='gray')
plt.tight_layout()
plt.savefig('charts/ch3_gdp_levels.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_gdp_levels.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_gdp_levels.pdf")

# Chart 2: GDP growth rate (first difference of log)
gdp_growth = np.log(gdp_data).diff().dropna() * 100  # Percentage growth

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Levels
axes[0].plot(gdp_data.index, gdp_data.values, color=BLUE, linewidth=1.5)
axes[0].set_title('GDP in Levels (Non-Stationary)', fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Billions $')

# Growth rate
axes[1].plot(gdp_growth.index, gdp_growth.values, color=GREEN, linewidth=1)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].axhline(y=gdp_growth.mean(), color=RED, linestyle='--', alpha=0.7, label=f'Mean = {gdp_growth.mean():.2f}%')
axes[1].set_title('GDP Growth Rate (Stationary)', fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Growth Rate (%)')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch3_differencing.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_differencing.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_differencing.pdf")

# Chart 3: ACF/PACF comparison (levels vs differenced)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ACF of levels
acf_levels = acf(gdp_data.dropna(), nlags=20)
axes[0, 0].bar(range(len(acf_levels)), acf_levels, color=BLUE, width=0.3)
axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
axes[0, 0].axhline(y=1.96/np.sqrt(len(gdp_data)), color=RED, linestyle='--', alpha=0.7)
axes[0, 0].axhline(y=-1.96/np.sqrt(len(gdp_data)), color=RED, linestyle='--', alpha=0.7)
axes[0, 0].set_title('ACF of GDP Levels', fontweight='bold')
axes[0, 0].set_xlabel('Lag')
axes[0, 0].set_ylabel('ACF')

# PACF of levels
pacf_levels = pacf(gdp_data.dropna(), nlags=20)
axes[0, 1].bar(range(len(pacf_levels)), pacf_levels, color=BLUE, width=0.3)
axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
axes[0, 1].axhline(y=1.96/np.sqrt(len(gdp_data)), color=RED, linestyle='--', alpha=0.7)
axes[0, 1].axhline(y=-1.96/np.sqrt(len(gdp_data)), color=RED, linestyle='--', alpha=0.7)
axes[0, 1].set_title('PACF of GDP Levels', fontweight='bold')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('PACF')

# ACF of growth
acf_diff = acf(gdp_growth.dropna(), nlags=20)
axes[1, 0].bar(range(len(acf_diff)), acf_diff, color=GREEN, width=0.3)
axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
axes[1, 0].axhline(y=1.96/np.sqrt(len(gdp_growth)), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=-1.96/np.sqrt(len(gdp_growth)), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].set_title('ACF of GDP Growth (Differenced)', fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')

# PACF of growth
pacf_diff = pacf(gdp_growth.dropna(), nlags=20)
axes[1, 1].bar(range(len(pacf_diff)), pacf_diff, color=GREEN, width=0.3)
axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
axes[1, 1].axhline(y=1.96/np.sqrt(len(gdp_growth)), color=RED, linestyle='--', alpha=0.7)
axes[1, 1].axhline(y=-1.96/np.sqrt(len(gdp_growth)), color=RED, linestyle='--', alpha=0.7)
axes[1, 1].set_title('PACF of GDP Growth (Differenced)', fontweight='bold')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('PACF')

plt.tight_layout()
plt.savefig('charts/ch3_acf_pacf.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_acf_pacf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_acf_pacf.pdf")

# Chart 4: ARIMA forecasting with train/test
train = gdp_data[:-12]
test = gdp_data[-12:]

# Fit ARIMA(1,1,1) on log GDP
log_train = np.log(train)
log_test = np.log(test)

model = ARIMA(log_train, order=(1, 1, 1))
fitted = model.fit()

# Forecast
forecast = fitted.get_forecast(steps=len(test))
forecast_mean = np.exp(forecast.predicted_mean)
conf_int = np.exp(forecast.conf_int())

fig, ax = plt.subplots(figsize=(14, 6))

# Training data (last 40 obs)
ax.plot(train.index[-40:], train.values[-40:], color=BLUE, linewidth=1.5, label='Training Data')

# Actual test data
ax.plot(test.index, test.values, color=GREEN, linewidth=2, label='Actual', marker='o', markersize=5)

# Forecast
ax.plot(test.index, forecast_mean.values, color=RED, linewidth=2, linestyle='--', label='ARIMA(1,1,1) Forecast')

# Confidence interval
ax.fill_between(test.index, conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values,
                color=RED, alpha=0.2, label='95% CI')

ax.axvline(x=train.index[-1], color='gray', linestyle=':', alpha=0.7, linewidth=2)
ax.set_title('US GDP: ARIMA(1,1,1) Forecast vs Actual', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Billions of 2017 Dollars')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.16)
plt.savefig('charts/ch3_arima_forecast.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_arima_forecast.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_arima_forecast.pdf")

# Chart 5: Unit root test visualization (ADF results)
adf_levels = adfuller(gdp_data.dropna(), autolag='AIC')
adf_diff = adfuller(gdp_growth.dropna(), autolag='AIC')

fig, ax = plt.subplots(figsize=(10, 6))

# Create bar chart for ADF statistics vs critical values
categories = ['GDP Levels', 'GDP Growth\n(Differenced)', '1% Critical', '5% Critical', '10% Critical']
values = [adf_levels[0], adf_diff[0], adf_levels[4]['1%'], adf_levels[4]['5%'], adf_levels[4]['10%']]
colors = [BLUE, GREEN, RED, RED, RED]

bars = ax.bar(categories, values, color=colors, width=0.6, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.5)

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}',
            ha='center', va='bottom' if height > 0 else 'top', fontsize=11)

ax.set_title('Augmented Dickey-Fuller Test Results', fontweight='bold', fontsize=14)
ax.set_ylabel('Test Statistic / Critical Value')
ax.set_ylim(min(values) - 1, max(values) + 1)

# Add interpretation
ax.text(0.02, 0.98, f'GDP Levels: p-value = {adf_levels[1]:.4f} (Non-stationary)',
        transform=ax.transAxes, fontsize=10, verticalalignment='top', color=BLUE)
ax.text(0.02, 0.92, f'GDP Growth: p-value = {adf_diff[1]:.4f} (Stationary)',
        transform=ax.transAxes, fontsize=10, verticalalignment='top', color=GREEN)

plt.tight_layout()
plt.savefig('charts/ch3_adf_test.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_adf_test.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_adf_test.pdf")

# Chart 6: Residual diagnostics
resid = fitted.resid

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals over time
axes[0, 0].plot(resid.index, resid.values, color=BLUE, linewidth=0.8)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_title('ARIMA Residuals', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residual')

# Histogram
axes[0, 1].hist(resid, bins=25, density=True, color=BLUE, alpha=0.7, edgecolor='white')
from scipy import stats
x_range = np.linspace(resid.min(), resid.max(), 100)
axes[0, 1].plot(x_range, stats.norm.pdf(x_range, resid.mean(), resid.std()),
               color=RED, linewidth=2, label='Normal')
axes[0, 1].set_title('Residual Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend(loc='upper right', fontsize=9, frameon=False)

# ACF of residuals
acf_resid = acf(resid.dropna(), nlags=20)
axes[1, 0].bar(range(len(acf_resid)), acf_resid, color=BLUE, width=0.3)
axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
axes[1, 0].axhline(y=1.96/np.sqrt(len(resid)), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=-1.96/np.sqrt(len(resid)), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].set_title('ACF of Residuals', fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')

# Q-Q plot (45-degree reference line)
(osm, osr), _ = stats.probplot(resid.dropna(), dist='norm', fit=True)
axes[1, 1].scatter(osm, osr, color=BLUE, alpha=0.6, s=30)
axes[1, 1].plot([-3, 3], [-3, 3], color=RED, linewidth=2, linestyle='--', label='45° line')
axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
axes[1, 1].set_xlabel('Theoretical Quantiles')
axes[1, 1].set_ylabel('Sample Quantiles')

plt.tight_layout()
plt.savefig('charts/ch3_diagnostics.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_diagnostics.pdf")

# Chart 7: Random Walk Simulation
np.random.seed(123)
n_sim = 200
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Simulate multiple random walks
for i in range(5):
    rw = np.cumsum(np.random.normal(0, 1, n_sim))
    axes[0].plot(rw, linewidth=1, alpha=0.8)
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title('Random Walk: $Y_t = Y_{t-1} + \\varepsilon_t$', fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('$Y_t$')

# Random walk with drift
for i in range(5):
    rw_drift = 0.1 * np.arange(n_sim) + np.cumsum(np.random.normal(0, 1, n_sim))
    axes[1].plot(rw_drift, linewidth=1, alpha=0.8)
axes[1].set_title('Random Walk with Drift: $Y_t = \\mu + Y_{t-1} + \\varepsilon_t$', fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('$Y_t$')

plt.tight_layout()
plt.savefig('charts/ch3_random_walk.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_random_walk.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_random_walk.pdf")

# Chart 8: Deterministic Trend vs Stochastic Trend
np.random.seed(42)
n_sim = 150
t = np.arange(n_sim)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Deterministic trend
det_trend = 2 + 0.1 * t + np.random.normal(0, 2, n_sim)
axes[0].plot(det_trend, color=BLUE, linewidth=1.5, label='$Y_t = \\alpha + \\beta t + \\varepsilon_t$')
axes[0].plot(2 + 0.1 * t, color=RED, linewidth=2, linestyle='--', label='Trend line')
axes[0].set_title('Deterministic Trend (Trend-Stationary)', fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('$Y_t$')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

# Stochastic trend (random walk with drift)
stoch_trend = np.cumsum(0.1 + np.random.normal(0, 2, n_sim))
axes[1].plot(stoch_trend, color=GREEN, linewidth=1.5, label='$Y_t = Y_{t-1} + \\mu + \\varepsilon_t$')
axes[1].set_title('Stochastic Trend (Unit Root)', fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('$Y_t$')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.20)
plt.savefig('charts/ch3_trend_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_trend_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_trend_comparison.pdf")

# Chart 9: Variance Growth of Random Walk
np.random.seed(42)
n_paths = 1000
n_time = 100

# Simulate many random walks
all_paths = np.cumsum(np.random.normal(0, 1, (n_paths, n_time)), axis=1)

# Calculate variance at each time point
variances = np.var(all_paths, axis=0)
theoretical_var = np.arange(1, n_time + 1)  # Var(Y_t) = t * sigma^2

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Show fan chart of paths
percentiles = [5, 25, 50, 75, 95]
for p in percentiles:
    axes[0].plot(np.percentile(all_paths, p, axis=0), linewidth=1, alpha=0.7)
axes[0].fill_between(range(n_time),
                      np.percentile(all_paths, 5, axis=0),
                      np.percentile(all_paths, 95, axis=0),
                      alpha=0.2, color=BLUE)
axes[0].set_title('Random Walk Paths: Variance Grows Over Time', fontweight='bold')
axes[0].set_xlabel('Time $t$')
axes[0].set_ylabel('$Y_t$')
axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Variance plot
axes[1].plot(variances, color=BLUE, linewidth=2, label='Sample Variance')
axes[1].plot(theoretical_var, color=RED, linewidth=2, linestyle='--', label='Theoretical: $Var(Y_t) = t\\sigma^2$')
axes[1].set_title('Variance of Random Walk Grows Linearly', fontweight='bold')
axes[1].set_xlabel('Time $t$')
axes[1].set_ylabel('$Var(Y_t)$')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.20)
plt.savefig('charts/ch3_variance_growth.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_variance_growth.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_variance_growth.pdf")

# Chart 10: ACF of Non-Stationary vs Stationary
np.random.seed(42)
n_sim = 200

# Random walk
rw = np.cumsum(np.random.normal(0, 1, n_sim))
# White noise
wn = np.random.normal(0, 1, n_sim)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Random walk series
axes[0, 0].plot(rw, color=BLUE, linewidth=1)
axes[0, 0].set_title('Random Walk (Non-Stationary)', fontweight='bold')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('$Y_t$')

# ACF of random walk - slow decay
acf_rw = acf(rw, nlags=30)
axes[0, 1].bar(range(len(acf_rw)), acf_rw, color=BLUE, width=0.4)
axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
axes[0, 1].axhline(y=1.96/np.sqrt(n_sim), color=RED, linestyle='--', alpha=0.7)
axes[0, 1].axhline(y=-1.96/np.sqrt(n_sim), color=RED, linestyle='--', alpha=0.7)
axes[0, 1].set_title('ACF: Slow Decay → Unit Root', fontweight='bold')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('ACF')

# Differenced (white noise)
axes[1, 0].plot(np.diff(rw), color=GREEN, linewidth=1)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].set_title('First Difference $\\Delta Y_t$ (Stationary)', fontweight='bold')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('$\\Delta Y_t$')

# ACF of differenced - cuts off
acf_diff = acf(np.diff(rw), nlags=30)
axes[1, 1].bar(range(len(acf_diff)), acf_diff, color=GREEN, width=0.4)
axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
axes[1, 1].axhline(y=1.96/np.sqrt(n_sim), color=RED, linestyle='--', alpha=0.7)
axes[1, 1].axhline(y=-1.96/np.sqrt(n_sim), color=RED, linestyle='--', alpha=0.7)
axes[1, 1].set_title('ACF: Cuts Off → Stationary', fontweight='bold')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('ACF')

plt.tight_layout()
plt.savefig('charts/ch3_acf_nonstationary.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_acf_nonstationary.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_acf_nonstationary.pdf")

# Chart 11: Rolling Forecast Illustration
np.random.seed(42)
n_total = 150
window_size = 80

# Generate ARIMA(1,1,0) data
phi = 0.6
eps = np.random.normal(0, 2, n_total)
diff_y = np.zeros(n_total)
for t in range(1, n_total):
    diff_y[t] = phi * diff_y[t-1] + eps[t]
y = 100 + np.cumsum(diff_y)

# Rolling forecasts
forecasts = []
forecast_times = []
actuals = []

for t in range(window_size, n_total - 1):
    # Fit on window
    train = y[:t]
    # Simple 1-step forecast using last difference
    last_diff = train[-1] - train[-2]
    forecast = train[-1] + phi * last_diff
    forecasts.append(forecast)
    forecast_times.append(t)
    actuals.append(y[t])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Show rolling window concept
t_show = list(range(n_total))
axes[0].plot(t_show, y, color=BLUE, linewidth=1.5, label='Actual Data')

# Highlight different windows
windows_to_show = [80, 100, 120]
colors_w = ['#FFD700', '#FFA500', '#FF6347']
for i, w_start in enumerate(windows_to_show):
    w_end = w_start
    axes[0].axvspan(w_start - window_size, w_start, alpha=0.2, color=colors_w[i])
    axes[0].scatter([w_start], [y[w_start]], color=colors_w[i], s=100, zorder=5,
                    edgecolor='black', linewidth=1.5)

axes[0].axvline(x=window_size, color='gray', linestyle='--', alpha=0.7)
axes[0].set_title('Rolling Window Concept', fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('$Y_t$')
axes[0].text(40, max(y)-10, f'Window\nsize = {window_size}', fontsize=10, ha='center')
axes[0].legend(loc='upper left', fontsize=9, frameon=False)

# Right: Forecasts vs Actuals
axes[1].plot(forecast_times, actuals, color=BLUE, linewidth=1.5, label='Actual')
axes[1].plot(forecast_times, forecasts, color=RED, linewidth=1.5, linestyle='--',
             label='1-Step Forecast', alpha=0.8)

# Calculate RMSE
rmse = np.sqrt(np.mean((np.array(forecasts) - np.array(actuals))**2))
axes[1].set_title(f'Rolling 1-Step Forecasts (RMSE = {rmse:.2f})', fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch3_rolling_forecast.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_rolling_forecast.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_rolling_forecast.pdf")

print("\nAll Chapter 3 charts generated successfully!")
