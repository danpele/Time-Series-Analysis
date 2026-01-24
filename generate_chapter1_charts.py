#!/usr/bin/env python3
"""Generate charts for Chapter 1: Introduction to Time Series"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set Harvard-style professional settings
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['axes.linewidth'] = 0.8

# Professional color palette (Harvard/academic style)
BLUE = '#1A3A6E'      # Dark navy blue
RED = '#DC3545'       # Clear red
GREEN = '#2E7D32'     # Forest green
ORANGE = '#E67E22'    # Muted orange
PURPLE = '#8E44AD'    # Academic purple
GRAY = '#666666'      # Neutral gray

np.random.seed(42)

# Chart 1: Time Series Examples
n = 120
dates = pd.date_range(start='2015-01-01', periods=n, freq='M')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Trend
trend_data = 100 + 0.5 * np.arange(n) + np.random.normal(0, 5, n)
axes[0, 0].plot(dates, trend_data, color=BLUE, linewidth=1.5)
axes[0, 0].set_title('Trend Pattern', fontweight='bold')
axes[0, 0].set_ylabel('Value')

# Seasonality
seasonal = 100 + 20 * np.sin(2 * np.pi * np.arange(n) / 12) + np.random.normal(0, 3, n)
axes[0, 1].plot(dates, seasonal, color=GREEN, linewidth=1.5)
axes[0, 1].set_title('Seasonal Pattern', fontweight='bold')
axes[0, 1].set_ylabel('Value')

# Trend + Seasonality
trend_seasonal = 100 + 0.3 * np.arange(n) + 15 * np.sin(2 * np.pi * np.arange(n) / 12) + np.random.normal(0, 3, n)
axes[1, 0].plot(dates, trend_seasonal, color=ORANGE, linewidth=1.5)
axes[1, 0].set_title('Trend + Seasonality', fontweight='bold')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Value')

# Random/Irregular
random_data = np.random.normal(100, 10, n)
axes[1, 1].plot(dates, random_data, color=RED, linewidth=1.5)
axes[1, 1].set_title('Random (No Pattern)', fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Value')

plt.tight_layout()
plt.savefig('charts/ch1_ts_patterns.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch1_ts_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch1_ts_patterns.pdf")

# Chart 2: Decomposition Example
n = 144
t = np.arange(n)
trend = 100 + 0.5 * t
seasonal = 20 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 5, n)
y = trend + seasonal + noise

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
dates = pd.date_range(start='2012-01-01', periods=n, freq='M')

axes[0].plot(dates, y, color=BLUE, linewidth=1)
axes[0].set_title('Original Series: $Y_t = T_t + S_t + \\varepsilon_t$', fontweight='bold')
axes[0].set_ylabel('$Y_t$')

axes[1].plot(dates, trend, color=GREEN, linewidth=2)
axes[1].set_title('Trend Component $T_t$', fontweight='bold')
axes[1].set_ylabel('$T_t$')

axes[2].plot(dates, seasonal, color=ORANGE, linewidth=1.5)
axes[2].set_title('Seasonal Component $S_t$', fontweight='bold')
axes[2].set_ylabel('$S_t$')

axes[3].plot(dates, noise, color=RED, linewidth=0.8)
axes[3].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[3].set_title('Irregular Component $\\varepsilon_t$', fontweight='bold')
axes[3].set_xlabel('Date')
axes[3].set_ylabel('$\\varepsilon_t$')

plt.tight_layout()
plt.savefig('charts/ch1_decomposition.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch1_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch1_decomposition.pdf")

# Chart 3: Moving Average Smoothing
n = 100
t = np.arange(n)
y = 50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 5, n)

def moving_average(data, window):
    return pd.Series(data).rolling(window=window, center=True).mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t, y, color=BLUE, linewidth=1, alpha=0.5, label='Original')
ax.plot(t, moving_average(y, 3), color=GREEN, linewidth=2, label='MA(3)')
ax.plot(t, moving_average(y, 12), color=RED, linewidth=2, label='MA(12)')
ax.set_title('Moving Average Smoothing', fontweight='bold')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

plt.tight_layout()
plt.savefig('charts/ch1_moving_average.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch1_moving_average.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch1_moving_average.pdf")

# Chart 4: Exponential Smoothing
n = 50
y = np.cumsum(np.random.normal(0, 1, n)) + 50

def exponential_smoothing(data, alpha):
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1 - alpha) * result[-1])
    return np.array(result)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(n), y, 'o-', color=BLUE, linewidth=1, markersize=4, alpha=0.6, label='Original')
ax.plot(range(n), exponential_smoothing(y, 0.1), color=GREEN, linewidth=2, label='$\\alpha = 0.1$ (smooth)')
ax.plot(range(n), exponential_smoothing(y, 0.5), color=ORANGE, linewidth=2, label='$\\alpha = 0.5$')
ax.plot(range(n), exponential_smoothing(y, 0.9), color=RED, linewidth=2, label='$\\alpha = 0.9$ (responsive)')
ax.set_title('Simple Exponential Smoothing: Effect of $\\alpha$', fontweight='bold')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)

plt.tight_layout()
plt.savefig('charts/ch1_exponential_smoothing.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch1_exponential_smoothing.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch1_exponential_smoothing.pdf")

# Chart 5: ACF Examples
n = 200

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# White noise
wn = np.random.normal(0, 1, n)
acf_wn = [np.corrcoef(wn[:-k], wn[k:])[0, 1] if k > 0 else 1 for k in range(21)]
axes[0, 0].bar(range(21), acf_wn, color=BLUE, width=0.4)
axes[0, 0].axhline(y=1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
axes[0, 0].axhline(y=-1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
axes[0, 0].set_title('ACF of White Noise', fontweight='bold')
axes[0, 0].set_xlabel('Lag')
axes[0, 0].set_ylabel('ACF')

# AR(1) positive
ar1_pos = [0]
for i in range(1, n):
    ar1_pos.append(0.8 * ar1_pos[-1] + np.random.normal(0, 1))
ar1_pos = np.array(ar1_pos)
acf_ar1 = [np.corrcoef(ar1_pos[:-k], ar1_pos[k:])[0, 1] if k > 0 else 1 for k in range(21)]
axes[0, 1].bar(range(21), acf_ar1, color=GREEN, width=0.4)
axes[0, 1].axhline(y=1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
axes[0, 1].axhline(y=-1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
axes[0, 1].set_title('ACF of AR(1) with $\\phi = 0.8$', fontweight='bold')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('ACF')

# Seasonal pattern
seasonal_data = np.sin(2 * np.pi * np.arange(n) / 12) + np.random.normal(0, 0.3, n)
acf_seasonal = [np.corrcoef(seasonal_data[:-k], seasonal_data[k:])[0, 1] if k > 0 else 1 for k in range(37)]
axes[1, 0].bar(range(37), acf_seasonal, color=ORANGE, width=0.4)
axes[1, 0].axhline(y=1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=-1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
for lag in [12, 24, 36]:
    axes[1, 0].axvline(x=lag, color=PURPLE, linestyle=':', alpha=0.5)
axes[1, 0].set_title('ACF of Seasonal Data (s=12)', fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')

# Random walk (unit root)
rw = np.cumsum(np.random.normal(0, 1, n))
acf_rw = [np.corrcoef(rw[:-k], rw[k:])[0, 1] if k > 0 else 1 for k in range(21)]
axes[1, 1].bar(range(21), acf_rw, color=PURPLE, width=0.4)
axes[1, 1].axhline(y=1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
axes[1, 1].axhline(y=-1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
axes[1, 1].set_title('ACF of Random Walk (Slow Decay)', fontweight='bold')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('ACF')

plt.tight_layout()
plt.savefig('charts/ch1_acf_examples.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch1_acf_examples.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch1_acf_examples.pdf")

# Chart 6: Stationarity vs Non-Stationarity
n = 150
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Stationary (constant mean and variance)
stationary = np.random.normal(50, 5, n)
axes[0, 0].plot(stationary, color=BLUE, linewidth=1)
axes[0, 0].axhline(y=50, color=RED, linestyle='--', linewidth=2, label='Mean')
axes[0, 0].fill_between(range(n), 40, 60, alpha=0.1, color=BLUE)
axes[0, 0].set_title('Stationary: Constant Mean & Variance', fontweight='bold')
axes[0, 0].set_ylabel('$Y_t$')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

# Non-stationary: Changing mean (trend)
trend = 20 + 0.3 * np.arange(n) + np.random.normal(0, 3, n)
axes[0, 1].plot(trend, color=GREEN, linewidth=1)
axes[0, 1].plot(20 + 0.3 * np.arange(n), color=RED, linestyle='--', linewidth=2, label='Changing Mean')
axes[0, 1].set_title('Non-Stationary: Trend (Changing Mean)', fontweight='bold')
axes[0, 1].set_ylabel('$Y_t$')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

# Non-stationary: Changing variance (heteroscedasticity)
hetero = np.array([np.random.normal(50, 2 + 0.1*i) for i in range(n)])
axes[1, 0].plot(hetero, color=ORANGE, linewidth=1)
axes[1, 0].axhline(y=50, color=RED, linestyle='--', linewidth=2)
axes[1, 0].fill_between(range(n), 50 - (2 + 0.1*np.arange(n))*2, 50 + (2 + 0.1*np.arange(n))*2,
                         alpha=0.2, color=ORANGE, label='Growing Variance')
axes[1, 0].set_title('Non-Stationary: Growing Variance', fontweight='bold')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('$Y_t$')

# Random walk
rw = np.cumsum(np.random.normal(0, 1, n))
axes[1, 1].plot(rw, color=PURPLE, linewidth=1)
axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Non-Stationary: Random Walk', fontweight='bold')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('$Y_t$')

plt.tight_layout()
plt.savefig('charts/ch1_stationarity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch1_stationarity.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch1_stationarity.pdf")

# Chart 7: Forecast Evaluation
n = 100
np.random.seed(123)
actual = 50 + 0.2 * np.arange(n) + np.random.normal(0, 5, n)
forecast = 50 + 0.2 * np.arange(n) + np.random.normal(0, 2, n)  # Slightly biased

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time series plot
axes[0].plot(actual, color=BLUE, linewidth=1.5, label='Actual')
axes[0].plot(forecast, color=RED, linewidth=1.5, linestyle='--', label='Forecast')
axes[0].set_title('Actual vs Forecast', fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

# Residuals
residuals = actual - forecast
axes[1].bar(range(n), residuals, color=BLUE, width=0.8, alpha=0.7)
axes[1].axhline(y=0, color='black', linewidth=1)
axes[1].axhline(y=np.mean(residuals), color=RED, linestyle='--', linewidth=2,
                label=f'Mean Error = {np.mean(residuals):.2f}')
axes[1].set_title('Forecast Errors (Residuals)', fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Error')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

plt.tight_layout()
plt.savefig('charts/ch1_forecast_eval.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch1_forecast_eval.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch1_forecast_eval.pdf")

# Chart 8: White Noise vs Random Walk
n = 200
np.random.seed(42)
wn = np.random.normal(0, 1, n)
rw = np.cumsum(wn)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# White noise series
axes[0, 0].plot(wn, color=BLUE, linewidth=0.8)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_title('White Noise: $\\varepsilon_t \\sim N(0, 1)$', fontweight='bold')
axes[0, 0].set_ylabel('$\\varepsilon_t$')

# White noise histogram
axes[0, 1].hist(wn, bins=30, density=True, color=BLUE, alpha=0.7, edgecolor='white')
x = np.linspace(-4, 4, 100)
axes[0, 1].plot(x, stats.norm.pdf(x), color=RED, linewidth=2, label='N(0,1)')
axes[0, 1].set_title('White Noise Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

# Random walk series
axes[1, 0].plot(rw, color=GREEN, linewidth=1)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1, 0].set_title('Random Walk: $Y_t = Y_{t-1} + \\varepsilon_t$', fontweight='bold')
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('$Y_t$')

# Random walk - multiple paths
axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
for i in range(10):
    path = np.cumsum(np.random.normal(0, 1, n))
    axes[1, 1].plot(path, linewidth=0.8, alpha=0.7)
axes[1, 1].set_title('Random Walk: Multiple Realizations', fontweight='bold')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('$Y_t$')

plt.tight_layout()
plt.savefig('charts/ch1_wn_rw.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch1_wn_rw.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch1_wn_rw.pdf")

# Chart 9: HP Filter - Effect of Lambda
def hp_filter(y, lamb):
    """Apply Hodrick-Prescott filter"""
    T = len(y)
    I = np.eye(T)
    D = np.zeros((T-2, T))
    for i in range(T-2):
        D[i, i] = 1
        D[i, i+1] = -2
        D[i, i+2] = 1
    trend = np.linalg.solve(I + lamb * D.T @ D, y)
    return trend

# Generate sample data with trend and cycle
np.random.seed(42)
T = 120
t = np.arange(T)
trend_true = 100 + 0.3 * t + 0.005 * t**2
cycle = 8 * np.sin(2 * np.pi * t / 20)
noise = np.random.normal(0, 2, T)
y = trend_true + cycle + noise

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Effect of different lambdas
lambdas = [10, 100, 1600, 50000]
colors = [BLUE, GREEN, ORANGE, RED]
labels = ['$\\lambda = 10$ (flexible)', '$\\lambda = 100$', '$\\lambda = 1600$ (standard)', '$\\lambda = 50000$ (smooth)']

axes[0].plot(t, y, 'k-', linewidth=0.8, alpha=0.5, label='Original data')
for lamb, color, label in zip(lambdas, colors, labels):
    trend_hp = hp_filter(y, lamb)
    axes[0].plot(t, trend_hp, color=color, linewidth=2, label=label)

axes[0].set_title('HP Filter: Effect of $\\lambda$ on Trend', fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

# Corresponding cycles
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
for lamb, color in zip(lambdas, colors):
    trend_hp = hp_filter(y, lamb)
    cycle_hp = y - trend_hp
    axes[1].plot(t, cycle_hp, color=color, linewidth=1.5, alpha=0.8)

axes[1].set_title('HP Filter: Extracted Cycles', fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Cycle Component')

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig('charts/ch1_hp_filter_lambda.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch1_hp_filter_lambda.pdf")

# Chart 10: HP Filter - Business Cycle Extraction
# Use real-like GDP data
np.random.seed(123)
T = 136  # Quarterly data 1990-2023
t = np.arange(T)

# Simulate realistic GDP growth pattern
trend_gdp = 100 * np.exp(0.006 * t)  # ~2.5% annual growth
cycle_gdp = 3 * np.sin(2 * np.pi * t / 32) + 2 * np.sin(2 * np.pi * t / 16)
# Add recession shocks
cycle_gdp[32:36] -= 5  # Early 1990s
cycle_gdp[44:48] -= 3  # 2001 recession
cycle_gdp[72:80] -= 10  # Great recession
cycle_gdp[120:122] -= 15  # COVID
noise_gdp = np.random.normal(0, 0.8, T)
gdp = trend_gdp * (1 + (cycle_gdp + noise_gdp) / 100)

# Apply HP filter with lambda = 1600 (quarterly)
trend_hp = hp_filter(gdp, 1600)
cycle_hp = (gdp - trend_hp) / trend_hp * 100  # Percentage deviation

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
dates = pd.date_range('1990-01-01', periods=T, freq='QS')

# GDP and trend
axes[0].plot(dates, gdp, 'b-', linewidth=1, label='GDP (Index)')
axes[0].plot(dates, trend_hp, 'r--', linewidth=2, label='HP Trend ($\\lambda=1600$)')
axes[0].set_title('US GDP: Level and HP Trend', fontweight='bold')
axes[0].set_ylabel('Index (1990=100)')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

# Add recession shading
recessions = [('1990-07-01', '1991-03-01'), ('2001-03-01', '2001-11-01'),
              ('2007-12-01', '2009-06-01'), ('2020-02-01', '2020-04-01')]
for start, end in recessions:
    for ax in axes:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2, color='gray')

# Cyclical component
axes[1].fill_between(dates, 0, cycle_hp, where=(cycle_hp >= 0), color=GREEN, alpha=0.4, label='Expansion')
axes[1].fill_between(dates, 0, cycle_hp, where=(cycle_hp < 0), color=RED, alpha=0.4, label='Contraction')
axes[1].plot(dates, cycle_hp, 'k-', linewidth=1)
axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
axes[1].set_title('Business Cycle: HP Cyclical Component', fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('% Deviation from Trend')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig('charts/ch1_hp_filter_cycle.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch1_hp_filter_cycle.pdf")

print("\nAll Chapter 1 charts generated successfully!")
