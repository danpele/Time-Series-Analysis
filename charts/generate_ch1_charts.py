#!/usr/bin/env python3
"""
Generate all Chapter 1 charts with consistent styling:
- Transparent background
- Legend outside at bottom
- Professional color scheme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STYLE SETTINGS (matching ch8)
# =============================================================================
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Colors
MAIN_BLUE = '#1A3A6E'
ACCENT_BLUE = '#4A90D9'
IDA_RED = '#DC3545'
FOREST = '#2E7D32'
ORANGE = '#E67E22'
GRAY = '#666666'
PURPLE = '#8E44AD'

np.random.seed(42)

# =============================================================================
# Chart 1: Motivation - Time Series Everywhere
# =============================================================================
print("Generating ch1_motivation_everywhere.pdf...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Stock prices
n = 250
t = np.arange(n)
stock = 100 * np.exp(0.0003 * t + 0.02 * np.cumsum(np.random.randn(n)))
axes[0, 0].plot(t, stock, color=MAIN_BLUE, lw=1.5)
axes[0, 0].set_title('Stock Prices', fontweight='bold', color=MAIN_BLUE)
axes[0, 0].set_xlabel('Trading Days')
axes[0, 0].set_ylabel('Price ($)')

# Temperature
days = np.arange(365)
temp = 15 + 10 * np.sin(2 * np.pi * days / 365) + np.random.randn(365) * 3
axes[0, 1].plot(days, temp, color=IDA_RED, lw=1)
axes[0, 1].set_title('Daily Temperature', fontweight='bold', color=MAIN_BLUE)
axes[0, 1].set_xlabel('Day of Year')
axes[0, 1].set_ylabel('Temperature (°C)')

# Sales with seasonality
months = np.arange(36)
sales = 1000 + 50 * months + 200 * np.sin(2 * np.pi * months / 12) + np.random.randn(36) * 50
axes[1, 0].plot(months, sales, color=FOREST, lw=1.5, marker='o', markersize=4)
axes[1, 0].set_title('Monthly Sales', fontweight='bold', color=MAIN_BLUE)
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Sales ($)')

# GDP
quarters = np.arange(40)
gdp = 100 + 2 * quarters + np.cumsum(np.random.randn(40) * 0.5)
axes[1, 1].plot(quarters, gdp, color=ORANGE, lw=2)
axes[1, 1].set_title('Quarterly GDP Index', fontweight='bold', color=MAIN_BLUE)
axes[1, 1].set_xlabel('Quarter')
axes[1, 1].set_ylabel('GDP Index')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Finance'),
    Line2D([0], [0], color=IDA_RED, lw=2, label='Climate'),
    Line2D([0], [0], color=FOREST, lw=2, label='Business'),
    Line2D([0], [0], color=ORANGE, lw=2, label='Economics'),
]

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 0.02))
plt.savefig('ch1_motivation_everywhere.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_motivation_everywhere.pdf")

# =============================================================================
# Chart 2: Motivation - Forecasting
# =============================================================================
print("Generating ch1_motivation_forecast.pdf...")
fig, ax = plt.subplots(figsize=(14, 6))

n_hist = 100
n_forecast = 20
t_hist = np.arange(n_hist)
t_forecast = np.arange(n_hist, n_hist + n_forecast)

# Historical data
y_hist = 100 + 0.5 * t_hist + 10 * np.sin(2 * np.pi * t_hist / 12) + np.random.randn(n_hist) * 5

# Forecast with confidence interval
y_forecast = 100 + 0.5 * t_forecast + 10 * np.sin(2 * np.pi * t_forecast / 12)
ci_width = np.linspace(3, 15, n_forecast)

ax.plot(t_hist, y_hist, color=MAIN_BLUE, lw=2, label='Historical Data')
ax.plot(t_forecast, y_forecast, color=IDA_RED, lw=2, ls='--', label='Forecast')
ax.fill_between(t_forecast, y_forecast - ci_width, y_forecast + ci_width,
                color=IDA_RED, alpha=0.2, label='95% Confidence Interval')
ax.axvline(x=n_hist, color=GRAY, ls=':', lw=2, alpha=0.7)
ax.text(n_hist + 1, ax.get_ylim()[1] - 5, 'Forecast\nHorizon', fontsize=10, color=GRAY)

ax.set_xlabel('Time Period', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Time Series Forecasting: Historical Data and Predictions', fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=11)
plt.savefig('ch1_motivation_forecast.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_motivation_forecast.pdf")

# =============================================================================
# Chart 3: Motivation - Components
# =============================================================================
print("Generating ch1_motivation_components.pdf...")
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

n = 120
t = np.arange(n)

# Components
trend = 50 + 0.3 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.randn(n) * 3
original = trend + seasonal + noise

axes[0].plot(t, original, color=MAIN_BLUE, lw=1.5)
axes[0].set_ylabel('Original', fontsize=11)
axes[0].set_title('Time Series Decomposition', fontweight='bold', color=MAIN_BLUE, fontsize=14)

axes[1].plot(t, trend, color=FOREST, lw=2)
axes[1].set_ylabel('Trend', fontsize=11)

axes[2].plot(t, seasonal, color=ORANGE, lw=1.5)
axes[2].set_ylabel('Seasonal', fontsize=11)

axes[3].plot(t, noise, color=GRAY, lw=1)
axes[3].set_ylabel('Residual', fontsize=11)
axes[3].set_xlabel('Time', fontsize=12)

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Original'),
    Line2D([0], [0], color=FOREST, lw=2, label='Trend'),
    Line2D([0], [0], color=ORANGE, lw=2, label='Seasonal'),
    Line2D([0], [0], color=GRAY, lw=2, label='Residual'),
]

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 0.02))
plt.savefig('ch1_motivation_components.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_motivation_components.pdf")

# =============================================================================
# Chart 4: Time Series Patterns
# =============================================================================
print("Generating ch1_ts_patterns.pdf...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

n = 100
t = np.arange(n)

# Trend
axes[0, 0].plot(t, 10 + 0.5 * t + np.random.randn(n) * 2, color=MAIN_BLUE, lw=1.5)
axes[0, 0].set_title('Trend Pattern', fontweight='bold', color=MAIN_BLUE)

# Seasonal
axes[0, 1].plot(t, 50 + 15 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 2, color=FOREST, lw=1.5)
axes[0, 1].set_title('Seasonal Pattern', fontweight='bold', color=MAIN_BLUE)

# Cyclical
cycle = 20 * np.sin(2 * np.pi * t / 40) + np.random.randn(n) * 3
axes[1, 0].plot(t, 50 + cycle, color=ORANGE, lw=1.5)
axes[1, 0].set_title('Cyclical Pattern', fontweight='bold', color=MAIN_BLUE)

# Random
axes[1, 1].plot(t, 50 + np.random.randn(n) * 5, color=IDA_RED, lw=1)
axes[1, 1].set_title('Random/Irregular Pattern', fontweight='bold', color=MAIN_BLUE)

for ax in axes.flat:
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Trend'),
    Line2D([0], [0], color=FOREST, lw=2, label='Seasonal'),
    Line2D([0], [0], color=ORANGE, lw=2, label='Cyclical'),
    Line2D([0], [0], color=IDA_RED, lw=2, label='Random'),
]

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 0.02))
plt.savefig('ch1_ts_patterns.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_ts_patterns.pdf")

# =============================================================================
# Chart 5: Decomposition
# =============================================================================
print("Generating ch1_decomposition.pdf...")
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

n = 144  # 12 years monthly
t = np.arange(n)

trend = 100 + 0.5 * t
seasonal = 20 * np.sin(2 * np.pi * t / 12)
residual = np.random.randn(n) * 5
observed = trend + seasonal + residual

axes[0].plot(t, observed, color=MAIN_BLUE, lw=1)
axes[0].set_ylabel('Observed')
axes[0].set_title('Additive Decomposition: $X_t = T_t + S_t + \\varepsilon_t$', fontweight='bold', color=MAIN_BLUE, fontsize=14)

axes[1].plot(t, trend, color=FOREST, lw=2)
axes[1].set_ylabel('Trend')

axes[2].plot(t, seasonal, color=ORANGE, lw=1.5)
axes[2].set_ylabel('Seasonal')
axes[2].axhline(0, color=GRAY, lw=0.5, ls=':')

axes[3].plot(t, residual, color=GRAY, lw=1)
axes[3].set_ylabel('Residual')
axes[3].axhline(0, color=GRAY, lw=0.5, ls=':')
axes[3].set_xlabel('Month')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Observed'),
    Line2D([0], [0], color=FOREST, lw=2, label='Trend'),
    Line2D([0], [0], color=ORANGE, lw=2, label='Seasonal'),
    Line2D([0], [0], color=GRAY, lw=2, label='Residual'),
]

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 0.02))
plt.savefig('ch1_decomposition.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_decomposition.pdf")

# =============================================================================
# Chart 6: Moving Average
# =============================================================================
print("Generating ch1_moving_average.pdf...")
fig, ax = plt.subplots(figsize=(14, 6))

n = 100
t = np.arange(n)
y = 50 + 10 * np.sin(2 * np.pi * t / 20) + np.random.randn(n) * 8

# Moving averages
ma5 = pd.Series(y).rolling(5).mean()
ma10 = pd.Series(y).rolling(10).mean()
ma20 = pd.Series(y).rolling(20).mean()

ax.plot(t, y, color=GRAY, lw=1, alpha=0.7, label='Original')
ax.plot(t, ma5, color=ACCENT_BLUE, lw=2, label='MA(5)')
ax.plot(t, ma10, color=FOREST, lw=2, label='MA(10)')
ax.plot(t, ma20, color=IDA_RED, lw=2, label='MA(20)')

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Moving Average Smoothing: Effect of Window Size', fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=11)
plt.savefig('ch1_moving_average.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_moving_average.pdf")

# =============================================================================
# Chart 7: Exponential Smoothing
# =============================================================================
print("Generating ch1_exponential_smoothing.pdf...")
fig, ax = plt.subplots(figsize=(14, 6))

n = 100
t = np.arange(n)
y = 50 + np.cumsum(np.random.randn(n) * 2)

def exp_smooth(y, alpha):
    result = np.zeros(len(y))
    result[0] = y[0]
    for i in range(1, len(y)):
        result[i] = alpha * y[i] + (1 - alpha) * result[i-1]
    return result

es01 = exp_smooth(y, 0.1)
es05 = exp_smooth(y, 0.5)
es09 = exp_smooth(y, 0.9)

ax.plot(t, y, color=GRAY, lw=1, alpha=0.7, label='Original')
ax.plot(t, es01, color=ACCENT_BLUE, lw=2, label='α = 0.1 (smooth)')
ax.plot(t, es05, color=FOREST, lw=2, label='α = 0.5')
ax.plot(t, es09, color=IDA_RED, lw=2, label='α = 0.9 (responsive)')

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Exponential Smoothing: Effect of α Parameter', fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=11)
plt.savefig('ch1_exponential_smoothing.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_exponential_smoothing.pdf")

# =============================================================================
# Chart 8: Stationarity Comparison
# =============================================================================
print("Generating ch1_stationarity.pdf...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n = 200
t = np.arange(n)

# Stationary
stationary = np.random.randn(n) * 5 + 50
axes[0].plot(t, stationary, color=FOREST, lw=1)
axes[0].axhline(50, color=MAIN_BLUE, lw=2, ls='--', label='Mean')
axes[0].fill_between(t, 40, 60, color=MAIN_BLUE, alpha=0.1, label='±2σ')
axes[0].set_title('Stationary Process', fontweight='bold', color=MAIN_BLUE, fontsize=13)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')
axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)

# Non-stationary (random walk)
random_walk = np.cumsum(np.random.randn(n)) + 50
axes[1].plot(t, random_walk, color=IDA_RED, lw=1)
axes[1].set_title('Non-Stationary Process (Random Walk)', fontweight='bold', color=MAIN_BLUE, fontsize=13)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig('ch1_stationarity.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_stationarity.pdf")

# =============================================================================
# Chart 9: White Noise vs Random Walk
# =============================================================================
print("Generating ch1_wn_rw.pdf...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

n = 200
t = np.arange(n)

# White noise
wn = np.random.randn(n) * 5
axes[0].plot(t, wn, color=MAIN_BLUE, lw=0.8)
axes[0].axhline(0, color=IDA_RED, lw=2, ls='--')
axes[0].set_title('White Noise: $\\varepsilon_t \\sim N(0, \\sigma^2)$', fontweight='bold', color=MAIN_BLUE, fontsize=13)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')
axes[0].set_ylim(-20, 20)

# Random walk
rw = np.cumsum(np.random.randn(n))
axes[1].plot(t, rw, color=IDA_RED, lw=1.2)
axes[1].set_title('Random Walk: $X_t = X_{t-1} + \\varepsilon_t$', fontweight='bold', color=MAIN_BLUE, fontsize=13)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='White Noise (Stationary)'),
    Line2D([0], [0], color=IDA_RED, lw=2, label='Random Walk (Non-Stationary)'),
]

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 0.02))
plt.savefig('ch1_wn_rw.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_wn_rw.pdf")

# =============================================================================
# Chart 10: ACF Examples
# =============================================================================
print("Generating ch1_acf_examples.pdf...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

lags = np.arange(21)
n = 200
conf = 1.96 / np.sqrt(n)

# White noise ACF
acf_wn = np.zeros(21)
acf_wn[0] = 1
acf_wn[1:] = np.random.randn(20) * 0.05
axes[0, 0].bar(lags, acf_wn, color=MAIN_BLUE, width=0.6)
axes[0, 0].axhline(conf, color=IDA_RED, ls='--', lw=1)
axes[0, 0].axhline(-conf, color=IDA_RED, ls='--', lw=1)
axes[0, 0].set_title('White Noise', fontweight='bold', color=MAIN_BLUE)
axes[0, 0].set_ylim(-0.3, 1.1)

# AR(1) ACF
phi = 0.8
acf_ar = phi ** lags
axes[0, 1].bar(lags, acf_ar, color=FOREST, width=0.6)
axes[0, 1].axhline(conf, color=IDA_RED, ls='--', lw=1)
axes[0, 1].axhline(-conf, color=IDA_RED, ls='--', lw=1)
axes[0, 1].set_title('AR(1) with φ = 0.8', fontweight='bold', color=MAIN_BLUE)
axes[0, 1].set_ylim(-0.3, 1.1)

# Seasonal ACF
acf_seasonal = np.zeros(21)
acf_seasonal[0] = 1
for i in range(1, 21):
    if i % 4 == 0:
        acf_seasonal[i] = 0.6 * (0.9 ** (i // 4))
    else:
        acf_seasonal[i] = np.random.randn() * 0.05
axes[1, 0].bar(lags, acf_seasonal, color=ORANGE, width=0.6)
axes[1, 0].axhline(conf, color=IDA_RED, ls='--', lw=1)
axes[1, 0].axhline(-conf, color=IDA_RED, ls='--', lw=1)
axes[1, 0].set_title('Seasonal (period = 4)', fontweight='bold', color=MAIN_BLUE)
axes[1, 0].set_ylim(-0.3, 1.1)

# Random walk ACF (slow decay)
acf_rw = 0.98 ** lags
axes[1, 1].bar(lags, acf_rw, color=PURPLE, width=0.6)
axes[1, 1].axhline(conf, color=IDA_RED, ls='--', lw=1)
axes[1, 1].axhline(-conf, color=IDA_RED, ls='--', lw=1)
axes[1, 1].set_title('Random Walk (very slow decay)', fontweight='bold', color=MAIN_BLUE)
axes[1, 1].set_ylim(-0.3, 1.1)

for ax in axes.flat:
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')

legend_elements = [
    Line2D([0], [0], color=IDA_RED, lw=1.5, ls='--', label='95% Confidence Bounds'),
]

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_elements, loc='lower center', ncol=1, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 0.02))
plt.savefig('ch1_acf_examples.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_acf_examples.pdf")

# =============================================================================
# Chart 11: Forecast Evaluation
# =============================================================================
print("Generating ch1_forecast_eval.pdf...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

n = 50
t = np.arange(n)
actual = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 3
forecast = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 1
residuals = actual - forecast

axes[0].plot(t, actual, color=MAIN_BLUE, lw=2, label='Actual')
axes[0].plot(t, forecast, color=IDA_RED, lw=2, ls='--', label='Forecast')
axes[0].set_ylabel('Value', fontsize=12)
axes[0].set_title('Forecast vs Actual Values', fontweight='bold', color=MAIN_BLUE, fontsize=14)
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

colors = [FOREST if r >= 0 else IDA_RED for r in residuals]
axes[1].bar(t, residuals, color=colors, width=0.8, alpha=0.7)
axes[1].axhline(0, color=GRAY, lw=1)
axes[1].set_xlabel('Time', fontsize=12)
axes[1].set_ylabel('Residual', fontsize=12)
axes[1].set_title('Forecast Errors (Residuals)', fontweight='bold', color=MAIN_BLUE, fontsize=13)

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Actual'),
    Line2D([0], [0], color=IDA_RED, lw=2, ls='--', label='Forecast'),
    Line2D([0], [0], color=FOREST, lw=8, alpha=0.7, label='Positive Error'),
    Line2D([0], [0], color=IDA_RED, lw=8, alpha=0.7, label='Negative Error'),
]

plt.tight_layout()
plt.subplots_adjust(bottom=0.12, hspace=0.3)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 0.02))
plt.savefig('ch1_forecast_eval.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_forecast_eval.pdf")

# =============================================================================
# Chart 12: HP Filter
# =============================================================================
print("Generating ch1_hp_filter_lambda.pdf...")
fig, ax = plt.subplots(figsize=(14, 6))

n = 100
t = np.arange(n)
y = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 20) + np.random.randn(n) * 5

# Simple trend approximations for different lambda
from scipy.ndimage import uniform_filter1d
trend_small = uniform_filter1d(y, size=5)
trend_med = uniform_filter1d(y, size=15)
trend_large = uniform_filter1d(y, size=30)

ax.plot(t, y, color=GRAY, lw=1, alpha=0.7, label='Original')
ax.plot(t, trend_small, color=ACCENT_BLUE, lw=2, label='λ = 100 (flexible)')
ax.plot(t, trend_med, color=FOREST, lw=2, label='λ = 1600 (standard)')
ax.plot(t, trend_large, color=IDA_RED, lw=2, label='λ = 10000 (smooth)')

ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('HP Filter: Effect of Smoothing Parameter λ', fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=11)
plt.savefig('ch1_hp_filter_lambda.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_hp_filter_lambda.pdf")

# =============================================================================
# Chart 13: Cyclical Component
# =============================================================================
print("Generating ch1_cyclical_component.pdf...")
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

n = 200
t = np.arange(n)

# GDP-like data
trend = 100 + 0.3 * t
cycle = 8 * np.sin(2 * np.pi * t / 40)  # Business cycle
noise = np.random.randn(n) * 2
gdp = trend + cycle + noise

axes[0].plot(t, gdp, color=MAIN_BLUE, lw=1.5)
axes[0].set_ylabel('GDP Index')
axes[0].set_title('US Real GDP: Trend-Cycle Decomposition', fontweight='bold', color=MAIN_BLUE, fontsize=14)

axes[1].plot(t, trend, color=FOREST, lw=2)
axes[1].set_ylabel('Trend')

axes[2].plot(t, cycle + noise, color=ORANGE, lw=1.5)
axes[2].axhline(0, color=GRAY, ls=':', lw=1)
axes[2].fill_between(t, 0, cycle + noise, where=(cycle + noise > 0), color=FOREST, alpha=0.3)
axes[2].fill_between(t, 0, cycle + noise, where=(cycle + noise < 0), color=IDA_RED, alpha=0.3)
axes[2].set_ylabel('Cycle')
axes[2].set_xlabel('Quarter')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Original'),
    Line2D([0], [0], color=FOREST, lw=2, label='Trend'),
    Line2D([0], [0], color=ORANGE, lw=2, label='Cyclical'),
]

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 0.02))
plt.savefig('ch1_cyclical_component.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_cyclical_component.pdf")

print("\n" + "="*60)
print("All Chapter 1 charts generated successfully!")
print("="*60)
