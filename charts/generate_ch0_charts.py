#!/usr/bin/env python3
"""
Generate all Chapter 0 charts with:
- Transparent background
- Legend outside at bottom
- PDF and PNG formats
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STYLE SETTINGS
# =============================================================================
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
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

def save_fig(name):
    plt.savefig(f'{name}.pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig(f'{name}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"Generated: {name}.pdf and {name}.png")

# =============================================================================
# 1. Motivation - Time Series Everywhere
# =============================================================================
print("Generating ch1_motivation_everywhere...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

n = 250
t = np.arange(n)
stock = 100 * np.exp(0.0003 * t + 0.02 * np.cumsum(np.random.randn(n)))
axes[0, 0].plot(t, stock, color=MAIN_BLUE, lw=1.5)
axes[0, 0].set_title('Stock Prices', fontweight='bold', color=MAIN_BLUE)
axes[0, 0].set_xlabel('Trading Days')
axes[0, 0].set_ylabel('Price ($)')

days = np.arange(365)
temp = 15 + 10 * np.sin(2 * np.pi * days / 365) + np.random.randn(365) * 3
axes[0, 1].plot(days, temp, color=IDA_RED, lw=1)
axes[0, 1].set_title('Daily Temperature', fontweight='bold', color=MAIN_BLUE)
axes[0, 1].set_xlabel('Day of Year')
axes[0, 1].set_ylabel('Temperature (°C)')

months = np.arange(36)
sales = 1000 + 50 * months + 200 * np.sin(2 * np.pi * months / 12) + np.random.randn(36) * 50
axes[1, 0].plot(months, sales, color=FOREST, lw=1.5, marker='o', markersize=4)
axes[1, 0].set_title('Monthly Sales', fontweight='bold', color=MAIN_BLUE)
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Sales ($)')

quarter_dates = pd.date_range('2014-01-01', periods=40, freq='Q')
gdp = 100 + 2 * np.arange(40) + np.cumsum(np.random.randn(40) * 0.5)
axes[1, 1].plot(quarter_dates, gdp, color=ORANGE, lw=2)
axes[1, 1].set_title('Quarterly GDP Index', fontweight='bold', color=MAIN_BLUE)
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('GDP Index')
axes[1, 1].tick_params(axis='x', rotation=45)

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Finance'),
    Line2D([0], [0], color=IDA_RED, lw=2, label='Climate'),
    Line2D([0], [0], color=FOREST, lw=2, label='Business'),
    Line2D([0], [0], color=ORANGE, lw=2, label='Economics'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('ch1_motivation_everywhere')

# =============================================================================
# 2. Motivation - Forecasting
# =============================================================================
print("Generating ch1_motivation_forecast...")
fig, ax = plt.subplots(figsize=(12, 7))

n_hist, n_forecast = 100, 20
t_hist = np.arange(n_hist)
t_forecast = np.arange(n_hist, n_hist + n_forecast)

y_hist = 100 + 0.5 * t_hist + 10 * np.sin(2 * np.pi * t_hist / 12) + np.random.randn(n_hist) * 5
y_forecast = 100 + 0.5 * t_forecast + 10 * np.sin(2 * np.pi * t_forecast / 12)
ci_width = np.linspace(3, 15, n_forecast)

ax.plot(t_hist, y_hist, color=MAIN_BLUE, lw=2, label='Historical Data')
ax.plot(t_forecast, y_forecast, color=IDA_RED, lw=2, ls='--', label='Forecast')
ax.fill_between(t_forecast, y_forecast - ci_width, y_forecast + ci_width, color=IDA_RED, alpha=0.2, label='95% CI')
ax.axvline(x=n_hist, color=GRAY, ls=':', lw=2, alpha=0.7)
ax.set_xlabel('Time Period')
ax.set_ylabel('Value')
ax.set_title('Time Series Forecasting', fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=11)
save_fig('ch1_motivation_forecast')

# =============================================================================
# 3. Motivation - Components
# =============================================================================
print("Generating ch1_motivation_components...")
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

n = 120
t = np.arange(n)
trend = 50 + 0.3 * t
seasonal = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.randn(n) * 3
original = trend + seasonal + noise

axes[0].plot(t, original, color=MAIN_BLUE, lw=1.5)
axes[0].set_ylabel('Original')
axes[0].set_title('Time Series Decomposition', fontweight='bold', color=MAIN_BLUE, fontsize=14)
axes[1].plot(t, trend, color=FOREST, lw=2)
axes[1].set_ylabel('Trend')
axes[2].plot(t, seasonal, color=ORANGE, lw=1.5)
axes[2].set_ylabel('Seasonal')
axes[3].plot(t, noise, color=GRAY, lw=1)
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Time')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Original'),
    Line2D([0], [0], color=FOREST, lw=2, label='Trend'),
    Line2D([0], [0], color=ORANGE, lw=2, label='Seasonal'),
    Line2D([0], [0], color=GRAY, lw=2, label='Residual'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('ch1_motivation_components')

# =============================================================================
# 4. Time Series Patterns
# =============================================================================
print("Generating ch1_ts_patterns...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

n = 100
t = np.arange(n)

axes[0, 0].plot(t, 10 + 0.5 * t + np.random.randn(n) * 2, color=MAIN_BLUE, lw=1.5)
axes[0, 0].set_title('Trend Pattern', fontweight='bold', color=MAIN_BLUE)
axes[0, 1].plot(t, 50 + 15 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 2, color=FOREST, lw=1.5)
axes[0, 1].set_title('Seasonal Pattern', fontweight='bold', color=MAIN_BLUE)
axes[1, 0].plot(t, 50 + 20 * np.sin(2 * np.pi * t / 40) + np.random.randn(n) * 3, color=ORANGE, lw=1.5)
axes[1, 0].set_title('Cyclical Pattern', fontweight='bold', color=MAIN_BLUE)
axes[1, 1].plot(t, 50 + np.random.randn(n) * 5, color=IDA_RED, lw=1)
axes[1, 1].set_title('Random Pattern', fontweight='bold', color=MAIN_BLUE)

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
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('ch1_ts_patterns')

# =============================================================================
# 5. Decomposition
# =============================================================================
print("Generating ch1_decomposition...")
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

n = 144
t = np.arange(n)
trend = 100 + 0.5 * t
seasonal = 20 * np.sin(2 * np.pi * t / 12)
residual = np.random.randn(n) * 5
observed = trend + seasonal + residual

axes[0].plot(t, observed, color=MAIN_BLUE, lw=1)
axes[0].set_ylabel('Observed')
axes[0].set_title('Additive Decomposition', fontweight='bold', color=MAIN_BLUE, fontsize=14)
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
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('ch1_decomposition')

# =============================================================================
# 6. Moving Average
# =============================================================================
print("Generating ch1_moving_average...")
fig, ax = plt.subplots(figsize=(12, 7))

n = 100
t = np.arange(n)
y = 50 + 10 * np.sin(2 * np.pi * t / 20) + np.random.randn(n) * 8

ma5 = pd.Series(y).rolling(5).mean()
ma10 = pd.Series(y).rolling(10).mean()
ma20 = pd.Series(y).rolling(20).mean()

ax.plot(t, y, color=GRAY, lw=1, alpha=0.7, label='Original')
ax.plot(t, ma5, color=ACCENT_BLUE, lw=2, label='MA(5)')
ax.plot(t, ma10, color=FOREST, lw=2, label='MA(10)')
ax.plot(t, ma20, color=IDA_RED, lw=2, label='MA(20)')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Moving Average Smoothing', fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=11)
save_fig('ch1_moving_average')

# =============================================================================
# 7. Exponential Smoothing
# =============================================================================
print("Generating ch1_exponential_smoothing...")
fig, ax = plt.subplots(figsize=(12, 7))

n = 100
t = np.arange(n)
y = 50 + np.cumsum(np.random.randn(n) * 2)

def exp_smooth(y, alpha):
    result = np.zeros(len(y))
    result[0] = y[0]
    for i in range(1, len(y)):
        result[i] = alpha * y[i] + (1 - alpha) * result[i-1]
    return result

ax.plot(t, y, color=GRAY, lw=1, alpha=0.7, label='Original')
ax.plot(t, exp_smooth(y, 0.1), color=ACCENT_BLUE, lw=2, label='α = 0.1')
ax.plot(t, exp_smooth(y, 0.5), color=FOREST, lw=2, label='α = 0.5')
ax.plot(t, exp_smooth(y, 0.9), color=IDA_RED, lw=2, label='α = 0.9')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Exponential Smoothing: Effect of α', fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=11)
save_fig('ch1_exponential_smoothing')

# =============================================================================
# 8. Stationarity
# =============================================================================
print("Generating ch1_stationarity...")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

n = 200
t = np.arange(n)

stationary = np.random.randn(n) * 5 + 50
axes[0].plot(t, stationary, color=FOREST, lw=1)
axes[0].axhline(50, color=MAIN_BLUE, lw=2, ls='--', label='Mean')
axes[0].fill_between(t, 40, 60, color=MAIN_BLUE, alpha=0.1, label='±2σ')
axes[0].set_title('Stationary Process', fontweight='bold', color=MAIN_BLUE, fontsize=13)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')

random_walk = np.cumsum(np.random.randn(n)) + 50
axes[1].plot(t, random_walk, color=IDA_RED, lw=1)
axes[1].set_title('Non-Stationary (Random Walk)', fontweight='bold', color=MAIN_BLUE, fontsize=13)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')

legend_elements = [
    Line2D([0], [0], color=FOREST, lw=2, label='Stationary'),
    Line2D([0], [0], color=IDA_RED, lw=2, label='Non-Stationary'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('ch1_stationarity')

# =============================================================================
# 9. White Noise vs Random Walk
# =============================================================================
print("Generating ch1_wn_rw...")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

n = 200
t = np.arange(n)

wn = np.random.randn(n) * 5
axes[0].plot(t, wn, color=MAIN_BLUE, lw=0.8)
axes[0].axhline(0, color=IDA_RED, lw=2, ls='--')
axes[0].set_title('White Noise', fontweight='bold', color=MAIN_BLUE, fontsize=13)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')
axes[0].set_ylim(-20, 20)

rw = np.cumsum(np.random.randn(n))
axes[1].plot(t, rw, color=IDA_RED, lw=1.2)
axes[1].set_title('Random Walk', fontweight='bold', color=MAIN_BLUE, fontsize=13)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='White Noise (Stationary)'),
    Line2D([0], [0], color=IDA_RED, lw=2, label='Random Walk (Non-Stationary)'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('ch1_wn_rw')

# =============================================================================
# 10. ACF Examples
# =============================================================================
print("Generating ch1_acf_examples...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

lags = np.arange(21)
n = 200
conf = 1.96 / np.sqrt(n)

acf_wn = np.zeros(21)
acf_wn[0] = 1
acf_wn[1:] = np.random.randn(20) * 0.05
axes[0, 0].bar(lags, acf_wn, color=MAIN_BLUE, width=0.6)
axes[0, 0].axhline(conf, color=IDA_RED, ls='--', lw=1)
axes[0, 0].axhline(-conf, color=IDA_RED, ls='--', lw=1)
axes[0, 0].set_title('White Noise', fontweight='bold', color=MAIN_BLUE)
axes[0, 0].set_ylim(-0.3, 1.1)

phi = 0.8
axes[0, 1].bar(lags, phi ** lags, color=FOREST, width=0.6)
axes[0, 1].axhline(conf, color=IDA_RED, ls='--', lw=1)
axes[0, 1].axhline(-conf, color=IDA_RED, ls='--', lw=1)
axes[0, 1].set_title('AR(1) φ = 0.8', fontweight='bold', color=MAIN_BLUE)
axes[0, 1].set_ylim(-0.3, 1.1)

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
axes[1, 0].set_title('Seasonal (s=4)', fontweight='bold', color=MAIN_BLUE)
axes[1, 0].set_ylim(-0.3, 1.1)

axes[1, 1].bar(lags, 0.98 ** lags, color=PURPLE, width=0.6)
axes[1, 1].axhline(conf, color=IDA_RED, ls='--', lw=1)
axes[1, 1].axhline(-conf, color=IDA_RED, ls='--', lw=1)
axes[1, 1].set_title('Random Walk', fontweight='bold', color=MAIN_BLUE)
axes[1, 1].set_ylim(-0.3, 1.1)

for ax in axes.flat:
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')

legend_elements = [Line2D([0], [0], color=IDA_RED, lw=1.5, ls='--', label='95% Confidence')]
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_elements, loc='lower center', ncol=1, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('ch1_acf_examples')

# =============================================================================
# 11. Forecast Evaluation
# =============================================================================
print("Generating ch1_forecast_eval...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

n = 50
t = np.arange(n)
actual = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 3
forecast = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(n) * 1
residuals = actual - forecast

axes[0].plot(t, actual, color=MAIN_BLUE, lw=2, label='Actual')
axes[0].plot(t, forecast, color=IDA_RED, lw=2, ls='--', label='Forecast')
axes[0].set_ylabel('Value')
axes[0].set_title('Forecast vs Actual', fontweight='bold', color=MAIN_BLUE, fontsize=14)

colors = [FOREST if r >= 0 else IDA_RED for r in residuals]
axes[1].bar(t, residuals, color=colors, width=0.8, alpha=0.7)
axes[1].axhline(0, color=GRAY, lw=1)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Residual')
axes[1].set_title('Forecast Errors', fontweight='bold', color=MAIN_BLUE, fontsize=13)

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Actual'),
    Line2D([0], [0], color=IDA_RED, lw=2, ls='--', label='Forecast'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.12, hspace=0.3)
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('ch1_forecast_eval')

# =============================================================================
# 12. HP Filter Lambda
# =============================================================================
print("Generating ch1_hp_filter_lambda...")
fig, ax = plt.subplots(figsize=(12, 7))

n = 100
t = np.arange(n)
y = 100 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 20) + np.random.randn(n) * 5

ax.plot(t, y, color=GRAY, lw=1, alpha=0.7, label='Original')
ax.plot(t, uniform_filter1d(y, size=5), color=ACCENT_BLUE, lw=2, label='λ = 100')
ax.plot(t, uniform_filter1d(y, size=15), color=FOREST, lw=2, label='λ = 1600')
ax.plot(t, uniform_filter1d(y, size=30), color=IDA_RED, lw=2, label='λ = 10000')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('HP Filter: Effect of λ', fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=11)
save_fig('ch1_hp_filter_lambda')

# =============================================================================
# 13. Cyclical Component
# =============================================================================
print("Generating ch1_cyclical_component...")
fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

n = 200
t = np.arange(n)
trend = 100 + 0.3 * t
cycle = 8 * np.sin(2 * np.pi * t / 40)
noise = np.random.randn(n) * 2
gdp = trend + cycle + noise

axes[0].plot(t, gdp, color=MAIN_BLUE, lw=1.5)
axes[0].set_ylabel('GDP Index')
axes[0].set_title('Trend-Cycle Decomposition', fontweight='bold', color=MAIN_BLUE, fontsize=14)
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
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('ch1_cyclical_component')

# =============================================================================
# 14. Time Series Definition
# =============================================================================
print("Generating timeseries_definition...")
fig, ax = plt.subplots(figsize=(12, 7))

n = 100
t = np.arange(n)
y = 100 + 0.3 * t + np.random.randn(n) * 5

ax.plot(t, y, color=MAIN_BLUE, lw=1.5, marker='o', markersize=3, alpha=0.8)
ax.set_xlabel('Time (t)', fontsize=12)
ax.set_ylabel('$X_t$', fontsize=12)
ax.set_title('Time Series: Sequence of Observations Indexed by Time', fontweight='bold', color=MAIN_BLUE, fontsize=14)

# Annotate a few points
for i in [20, 50, 80]:
    ax.annotate(f'$X_{{{i}}}$', xy=(i, y[i]), xytext=(i+3, y[i]+8),
                fontsize=10, color=IDA_RED, arrowprops=dict(arrowstyle='->', color=IDA_RED, lw=1))

legend_elements = [Line2D([0], [0], color=MAIN_BLUE, lw=2, marker='o', label='Observations $X_t$')]
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=11)
save_fig('timeseries_definition')

# =============================================================================
# 15. Data Types Comparison
# =============================================================================
print("Generating data_types_comparison...")
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# Cross-sectional
np.random.seed(42)
x_cross = np.random.randn(50)
y_cross = 2 + 1.5 * x_cross + np.random.randn(50) * 0.5
axes[0].scatter(x_cross, y_cross, color=MAIN_BLUE, alpha=0.7, s=50)
axes[0].set_title('Cross-Sectional', fontweight='bold', color=MAIN_BLUE)
axes[0].set_xlabel('Variable X')
axes[0].set_ylabel('Variable Y')

# Time series
t = np.arange(50)
y_ts = 100 + 0.5 * t + np.random.randn(50) * 5
axes[1].plot(t, y_ts, color=FOREST, lw=2)
axes[1].set_title('Time Series', fontweight='bold', color=MAIN_BLUE)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')

# Panel
for i, c in enumerate([MAIN_BLUE, FOREST, ORANGE, IDA_RED]):
    y_panel = 50 + i * 20 + 0.3 * t + np.random.randn(50) * 3
    axes[2].plot(t, y_panel, color=c, lw=1.5, alpha=0.8)
axes[2].set_title('Panel Data', fontweight='bold', color=MAIN_BLUE)
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Value')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Cross-sectional'),
    Line2D([0], [0], color=FOREST, lw=2, label='Time Series'),
    Line2D([0], [0], color=ORANGE, lw=2, label='Panel'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('data_types_comparison')

# =============================================================================
# 16. Multiple Assets
# =============================================================================
print("Generating multiple_assets...")
fig, ax = plt.subplots(figsize=(12, 7))

n = 250
t = np.arange(n)
np.random.seed(42)

sp500 = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01 + 0.0003))
gold = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.008 + 0.0001))
bitcoin = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.04 + 0.001))

ax.plot(t, sp500, color=MAIN_BLUE, lw=2, label='S&P 500')
ax.plot(t, gold, color=ORANGE, lw=2, label='Gold')
ax.plot(t, bitcoin, color=IDA_RED, lw=2, label='Bitcoin')
ax.set_xlabel('Trading Days')
ax.set_ylabel('Normalized Price (Base=100)')
ax.set_title('Financial Time Series Comparison', fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=11)
save_fig('multiple_assets')

# =============================================================================
# 17. Simple Exponential Smoothing
# =============================================================================
print("Generating simple_exp_smoothing...")
fig, ax = plt.subplots(figsize=(12, 7))

n = 60
t = np.arange(n)
y = 50 + np.random.randn(n) * 8

ax.plot(t, y, color=GRAY, lw=1, alpha=0.7, label='Original')
ax.plot(t, exp_smooth(y, 0.2), color=ACCENT_BLUE, lw=2, label='α = 0.2')
ax.plot(t, exp_smooth(y, 0.5), color=FOREST, lw=2, label='α = 0.5')
ax.plot(t, exp_smooth(y, 0.8), color=IDA_RED, lw=2, label='α = 0.8')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Simple Exponential Smoothing', fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=11)
save_fig('simple_exp_smoothing')

# =============================================================================
# 18. Holt Method
# =============================================================================
print("Generating holt_method...")
fig, ax = plt.subplots(figsize=(12, 7))

n = 80
t = np.arange(n)
y = 50 + 0.5 * t + np.random.randn(n) * 5

# Simple Holt approximation
level = np.zeros(n)
trend_est = np.zeros(n)
level[0] = y[0]
trend_est[0] = 0
alpha, beta = 0.3, 0.1
for i in range(1, n):
    level[i] = alpha * y[i] + (1 - alpha) * (level[i-1] + trend_est[i-1])
    trend_est[i] = beta * (level[i] - level[i-1]) + (1 - beta) * trend_est[i-1]

forecast_h = 20
t_fc = np.arange(n, n + forecast_h)
fc = level[-1] + trend_est[-1] * np.arange(1, forecast_h + 1)

ax.plot(t, y, color=GRAY, lw=1, alpha=0.7, label='Original')
ax.plot(t, level, color=FOREST, lw=2, label='Fitted')
ax.plot(t_fc, fc, color=IDA_RED, lw=2, ls='--', label='Forecast')
ax.axvline(n-1, color=GRAY, ls=':', lw=1)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title("Holt's Linear Trend Method", fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=11)
save_fig('holt_method')

# =============================================================================
# 19. Holt-Winters
# =============================================================================
print("Generating holt_winters...")
fig, ax = plt.subplots(figsize=(12, 7))

n = 96
t = np.arange(n)
trend = 50 + 0.3 * t
seasonal = 15 * np.sin(2 * np.pi * t / 12)
y = trend + seasonal + np.random.randn(n) * 3

# Approximate fitted
fitted = trend + seasonal * 0.9

forecast_h = 24
t_fc = np.arange(n, n + forecast_h)
fc_trend = 50 + 0.3 * t_fc
fc_seasonal = 15 * np.sin(2 * np.pi * t_fc / 12)
fc = fc_trend + fc_seasonal

ax.plot(t, y, color=GRAY, lw=1, alpha=0.7, label='Original')
ax.plot(t, fitted, color=FOREST, lw=2, label='Fitted')
ax.plot(t_fc, fc, color=IDA_RED, lw=2, ls='--', label='Forecast')
ax.axvline(n-1, color=GRAY, ls=':', lw=1)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Holt-Winters Method', fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=11)
save_fig('holt_winters')

# =============================================================================
# 20. Airline Decomposition
# =============================================================================
print("Generating airline_decomposition...")
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

n = 144
t = np.arange(n)
trend = 100 + 2 * t
seasonal = (1 + 0.3 * np.sin(2 * np.pi * t / 12))
noise = 1 + np.random.randn(n) * 0.05
observed = trend * seasonal * noise

axes[0].plot(t, observed, color=MAIN_BLUE, lw=1)
axes[0].set_ylabel('Observed')
axes[0].set_title('Multiplicative Decomposition: Airline Passengers', fontweight='bold', color=MAIN_BLUE, fontsize=14)
axes[1].plot(t, trend, color=FOREST, lw=2)
axes[1].set_ylabel('Trend')
axes[2].plot(t, seasonal, color=ORANGE, lw=1.5)
axes[2].set_ylabel('Seasonal')
axes[2].axhline(1, color=GRAY, lw=0.5, ls=':')
axes[3].plot(t, noise, color=GRAY, lw=1)
axes[3].set_ylabel('Residual')
axes[3].axhline(1, color=GRAY, lw=0.5, ls=':')
axes[3].set_xlabel('Month')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Observed'),
    Line2D([0], [0], color=FOREST, lw=2, label='Trend'),
    Line2D([0], [0], color=ORANGE, lw=2, label='Seasonal'),
    Line2D([0], [0], color=GRAY, lw=2, label='Residual'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('airline_decomposition')

# =============================================================================
# 21. Additive vs Multiplicative
# =============================================================================
print("Generating additive_vs_multiplicative...")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

np.random.seed(42)  # For reproducibility
n = 96
t = np.arange(n)

# Additive - constant seasonal amplitude
trend_a = 50 + 0.4 * t
amplitude_a = 12  # Constant amplitude
seasonal_a = amplitude_a * np.sin(2 * np.pi * t / 12)
y_add = trend_a + seasonal_a + np.random.randn(n) * 1.5

# Plot additive with envelope
axes[0].plot(t, y_add, color=MAIN_BLUE, lw=1.5, label='Data')
axes[0].plot(t, trend_a, color=FOREST, lw=2, ls='--', label='Trend')
axes[0].fill_between(t, trend_a - amplitude_a, trend_a + amplitude_a,
                      color=MAIN_BLUE, alpha=0.15, label='Amplitude band')
axes[0].set_title('Additive: $Y_t = T_t + S_t$\n(Constant Amplitude)', fontweight='bold', color=MAIN_BLUE, fontsize=12)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')
# Add annotation arrows showing constant amplitude
axes[0].annotate('', xy=(20, trend_a[20] + amplitude_a), xytext=(20, trend_a[20] - amplitude_a),
                 arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2))
axes[0].annotate('', xy=(70, trend_a[70] + amplitude_a), xytext=(70, trend_a[70] - amplitude_a),
                 arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2))
axes[0].text(23, trend_a[20], 'Same\nwidth', fontsize=9, color=ORANGE, fontweight='bold')
axes[0].text(73, trend_a[70], 'Same\nwidth', fontsize=9, color=ORANGE, fontweight='bold')

# Multiplicative - amplitude grows with trend
trend_m = 50 + 0.5 * t
amplitude_factor = 0.25  # 25% of trend value
seasonal_m = 1 + amplitude_factor * np.sin(2 * np.pi * t / 12)
y_mult = trend_m * seasonal_m * (1 + np.random.randn(n) * 0.02)

# Plot multiplicative with envelope
axes[1].plot(t, y_mult, color=IDA_RED, lw=1.5, label='Data')
axes[1].plot(t, trend_m, color=FOREST, lw=2, ls='--', label='Trend')
axes[1].fill_between(t, trend_m * (1 - amplitude_factor), trend_m * (1 + amplitude_factor),
                      color=IDA_RED, alpha=0.15, label='Amplitude band')
axes[1].set_title('Multiplicative: $Y_t = T_t \\times S_t$\n(Amplitude Grows with Trend)', fontweight='bold', color=MAIN_BLUE, fontsize=12)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')
# Add annotation arrows showing growing amplitude
amp_early = trend_m[20] * amplitude_factor
amp_late = trend_m[70] * amplitude_factor
axes[1].annotate('', xy=(20, trend_m[20] + amp_early), xytext=(20, trend_m[20] - amp_early),
                 arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2))
axes[1].annotate('', xy=(70, trend_m[70] + amp_late), xytext=(70, trend_m[70] - amp_late),
                 arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2))
axes[1].text(23, trend_m[20], 'Small', fontsize=9, color=ORANGE, fontweight='bold')
axes[1].text(73, trend_m[70], 'Large', fontsize=9, color=ORANGE, fontweight='bold')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Additive'),
    Line2D([0], [0], color=IDA_RED, lw=2, label='Multiplicative'),
    Line2D([0], [0], color=FOREST, lw=2, ls='--', label='Trend'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('additive_vs_multiplicative')

# =============================================================================
# 22. TS Components Synthetic
# =============================================================================
print("Generating ts_components_synthetic...")
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

n = 120
t = np.arange(n)
trend = 100 + 0.4 * t
seasonal = 15 * np.sin(2 * np.pi * t / 12)
residual = np.random.randn(n) * 4
observed = trend + seasonal + residual

axes[0].plot(t, observed, color=MAIN_BLUE, lw=1.5)
axes[0].set_ylabel('Observed')
axes[0].set_title('Additive Model: $X_t = T_t + S_t + ε_t$', fontweight='bold', color=MAIN_BLUE, fontsize=14)
axes[1].plot(t, trend, color=FOREST, lw=2)
axes[1].set_ylabel('Trend')
axes[2].plot(t, seasonal, color=ORANGE, lw=1.5)
axes[2].set_ylabel('Seasonal')
axes[2].axhline(0, color=GRAY, lw=0.5, ls=':')
axes[3].plot(t, residual, color=GRAY, lw=1)
axes[3].set_ylabel('Residual')
axes[3].axhline(0, color=GRAY, lw=0.5, ls=':')
axes[3].set_xlabel('Time')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Observed'),
    Line2D([0], [0], color=FOREST, lw=2, label='Trend'),
    Line2D([0], [0], color=ORANGE, lw=2, label='Seasonal'),
    Line2D([0], [0], color=GRAY, lw=2, label='Residual'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('ts_components_synthetic')

# =============================================================================
# 23-40: Additional charts (abbreviated for space)
# =============================================================================

# Seasonal Pattern
print("Generating seasonal_pattern...")
fig, ax = plt.subplots(figsize=(12, 7))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
seasonal_idx = [0.85, 0.88, 0.95, 1.02, 1.08, 1.12, 1.15, 1.12, 1.05, 0.98, 0.90, 0.90]
colors = [IDA_RED if s < 1 else FOREST for s in seasonal_idx]
ax.bar(months, seasonal_idx, color=colors, alpha=0.8)
ax.axhline(1, color=GRAY, lw=2, ls='--')
ax.set_ylabel('Seasonal Index')
ax.set_title('Seasonal Indices (Airline Passengers)', fontweight='bold', color=MAIN_BLUE, fontsize=14)
legend_elements = [
    Line2D([0], [0], color=FOREST, lw=8, label='Above Average'),
    Line2D([0], [0], color=IDA_RED, lw=8, label='Below Average'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
save_fig('seasonal_pattern')

# Cross Validation
print("Generating cross_validation_forecast...")
fig, ax = plt.subplots(figsize=(12, 6))
for i, start in enumerate([0, 10, 20, 30, 40]):
    train_end = 50 + start
    ax.barh(4-i, train_end, color=MAIN_BLUE, alpha=0.7, height=0.6)
    ax.barh(4-i, 10, left=train_end, color=IDA_RED, alpha=0.7, height=0.6)
ax.set_yticks(range(5))
ax.set_yticklabels([f'Fold {5-i}' for i in range(5)])
ax.set_xlabel('Time')
ax.set_title('Time Series Cross-Validation (Expanding Window)', fontweight='bold', color=MAIN_BLUE, fontsize=14)
legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=8, label='Training'),
    Line2D([0], [0], color=IDA_RED, lw=8, label='Validation'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
save_fig('cross_validation_forecast')

# Train Test Validation
print("Generating train_test_validation...")
fig, ax = plt.subplots(figsize=(14, 3))
ax.barh(0, 60, color=MAIN_BLUE, alpha=0.8, height=0.5, label='Train (60%)')
ax.barh(0, 20, left=60, color=FOREST, alpha=0.8, height=0.5, label='Validation (20%)')
ax.barh(0, 20, left=80, color=IDA_RED, alpha=0.8, height=0.5, label='Test (20%)')
ax.set_xlim(0, 100)
ax.set_yticks([])
ax.set_xlabel('Percentage of Data')
ax.set_title('Train / Validation / Test Split', fontweight='bold', color=MAIN_BLUE, fontsize=14)
legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=8, label='Training'),
    Line2D([0], [0], color=FOREST, lw=8, label='Validation'),
    Line2D([0], [0], color=IDA_RED, lw=8, label='Test'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.35)
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol=3, frameon=False)
save_fig('train_test_validation')

# Residual Diagnostics
print("Generating residual_diagnostics...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
residuals = np.random.randn(100) * 2

axes[0, 0].plot(residuals, color=MAIN_BLUE, lw=1)
axes[0, 0].axhline(0, color=IDA_RED, ls='--')
axes[0, 0].set_title('Residuals vs Time', fontweight='bold', color=MAIN_BLUE)
axes[0, 0].set_xlabel('Time')

axes[0, 1].hist(residuals, bins=20, color=MAIN_BLUE, alpha=0.7, edgecolor='white')
axes[0, 1].set_title('Histogram', fontweight='bold', color=MAIN_BLUE)
axes[0, 1].set_xlabel('Residual')

lags = np.arange(21)
acf = np.zeros(21)
acf[0] = 1
acf[1:] = np.random.randn(20) * 0.08
axes[1, 0].bar(lags, acf, color=MAIN_BLUE, width=0.6)
axes[1, 0].axhline(0.2, color=IDA_RED, ls='--')
axes[1, 0].axhline(-0.2, color=IDA_RED, ls='--')
axes[1, 0].set_title('ACF of Residuals', fontweight='bold', color=MAIN_BLUE)
axes[1, 0].set_xlabel('Lag')

from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontweight='bold', color=MAIN_BLUE)
axes[1, 1].get_lines()[0].set_color(MAIN_BLUE)
axes[1, 1].get_lines()[1].set_color(IDA_RED)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
save_fig('residual_diagnostics')

# Detrending Methods
print("Generating detrending_methods...")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
n = 100
t = np.arange(n)
y = 50 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 20) + np.random.randn(n) * 3

axes[0, 0].plot(t, y, color=GRAY, lw=1, alpha=0.7)
axes[0, 0].plot(t, 50 + 0.5 * t, color=IDA_RED, lw=2)
axes[0, 0].set_title('Linear Regression', fontweight='bold', color=MAIN_BLUE)

axes[0, 1].plot(t, y, color=GRAY, lw=1, alpha=0.7)
axes[0, 1].plot(t, uniform_filter1d(y, size=15), color=FOREST, lw=2)
axes[0, 1].set_title('Moving Average', fontweight='bold', color=MAIN_BLUE)

axes[1, 0].plot(t, y, color=GRAY, lw=1, alpha=0.7)
axes[1, 0].plot(t, uniform_filter1d(y, size=20), color=ORANGE, lw=2)
axes[1, 0].set_title('HP Filter', fontweight='bold', color=MAIN_BLUE)

axes[1, 1].plot(t[1:], np.diff(y), color=PURPLE, lw=1)
axes[1, 1].axhline(0, color=GRAY, ls='--')
axes[1, 1].set_title('First Difference', fontweight='bold', color=MAIN_BLUE)

for ax in axes.flat:
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

legend_elements = [
    Line2D([0], [0], color=GRAY, lw=2, label='Original'),
    Line2D([0], [0], color=IDA_RED, lw=2, label='Linear'),
    Line2D([0], [0], color=FOREST, lw=2, label='MA'),
    Line2D([0], [0], color=ORANGE, lw=2, label='HP'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('detrending_methods')

# Trend Estimation Comparison
print("Generating trend_estimation_comparison...")
fig, ax = plt.subplots(figsize=(12, 7))
n = 100
t = np.arange(n)
y = 50 + 0.4 * t + 8 * np.sin(2 * np.pi * t / 25) + np.random.randn(n) * 4

ax.plot(t, y, color=GRAY, lw=1, alpha=0.6, label='Original')
ax.plot(t, 50 + 0.4 * t, color=MAIN_BLUE, lw=2, label='Linear')
ax.plot(t, uniform_filter1d(y, size=10), color=FOREST, lw=2, label='MA(10)')
ax.plot(t, uniform_filter1d(y, size=20), color=IDA_RED, lw=2, label='HP Filter')

ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Trend Estimation Methods', fontweight='bold', color=MAIN_BLUE, fontsize=14)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=11)
save_fig('trend_estimation_comparison')

# Seasonal Adjustment
print("Generating seasonal_adjustment...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
n = 96
t = np.arange(n)
trend = 100 + 0.4 * t
seasonal = 20 * np.sin(2 * np.pi * t / 12)
y = trend + seasonal + np.random.randn(n) * 3

axes[0].plot(t, y, color=MAIN_BLUE, lw=1.5, label='Original')
axes[0].set_ylabel('Value')
axes[0].set_title('Before and After Seasonal Adjustment', fontweight='bold', color=MAIN_BLUE, fontsize=14)

axes[1].plot(t, trend + np.random.randn(n) * 3, color=FOREST, lw=1.5, label='Seasonally Adjusted')
axes[1].set_ylabel('Value')
axes[1].set_xlabel('Month')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Original'),
    Line2D([0], [0], color=FOREST, lw=2, label='Adjusted'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('seasonal_adjustment')

# Deterministic Trend Example
print("Generating deterministic_trend_example...")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
n = 100
t = np.arange(n)
y = 20 + 0.5 * t + np.random.randn(n) * 5

axes[0].plot(t, y, color=MAIN_BLUE, lw=1.5)
axes[0].plot(t, 20 + 0.5 * t, color=IDA_RED, lw=2, ls='--')
axes[0].set_title('Original + Trend Line', fontweight='bold', color=MAIN_BLUE)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')

residuals = y - (20 + 0.5 * t)
axes[1].plot(t, residuals, color=FOREST, lw=1)
axes[1].axhline(0, color=GRAY, ls='--')
axes[1].set_title('Detrended (Residuals)', fontweight='bold', color=MAIN_BLUE)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Residual')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Original'),
    Line2D([0], [0], color=IDA_RED, lw=2, ls='--', label='Trend'),
    Line2D([0], [0], color=FOREST, lw=2, label='Detrended'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('deterministic_trend_example')

# Stochastic Trend Example
print("Generating stochastic_trend_example...")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
n = 100
rw = np.cumsum(np.random.randn(n)) + 50

axes[0].plot(rw, color=MAIN_BLUE, lw=1.5)
axes[0].set_title('Random Walk', fontweight='bold', color=MAIN_BLUE)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')

axes[1].plot(np.diff(rw), color=FOREST, lw=1)
axes[1].axhline(0, color=GRAY, ls='--')
axes[1].set_title('First Difference (Stationary)', fontweight='bold', color=MAIN_BLUE)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('ΔX')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Original'),
    Line2D([0], [0], color=FOREST, lw=2, label='Differenced'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('stochastic_trend_example')

# Trend Comparison Side by Side
print("Generating trend_comparison_sidebyside...")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
n = 100
t = np.arange(n)

# Deterministic
y_det = 20 + 0.5 * t + np.random.randn(n) * 3
axes[0].plot(t, y_det, color=MAIN_BLUE, lw=1.5)
axes[0].plot(t, 20 + 0.5 * t, color=IDA_RED, lw=2, ls='--')
axes[0].set_title('Deterministic Trend', fontweight='bold', color=MAIN_BLUE)
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')

# Stochastic
y_stoch = np.cumsum(np.random.randn(n) * 0.8) + 50
axes[1].plot(t, y_stoch, color=FOREST, lw=1.5)
axes[1].set_title('Stochastic Trend', fontweight='bold', color=MAIN_BLUE)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Deterministic'),
    Line2D([0], [0], color=FOREST, lw=2, label='Stochastic'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('trend_comparison_sidebyside')

# Seasonality Fourier Dummies
print("Generating seasonality_fourier_dummies...")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
months = np.arange(12)

# Dummies
dummy_pattern = [0.9, 0.85, 0.95, 1.0, 1.05, 1.1, 1.15, 1.1, 1.0, 0.95, 0.9, 0.95]
axes[0].bar(months, dummy_pattern, color=MAIN_BLUE, alpha=0.8)
axes[0].axhline(1, color=GRAY, ls='--')
axes[0].set_xticks(months)
axes[0].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
axes[0].set_title('Dummy Variables', fontweight='bold', color=MAIN_BLUE)
axes[0].set_ylabel('Seasonal Effect')

# Fourier
t_fine = np.linspace(0, 12, 100)
fourier = 1 + 0.1 * np.sin(2 * np.pi * t_fine / 12) + 0.05 * np.cos(2 * np.pi * t_fine / 12)
axes[1].plot(t_fine, fourier, color=IDA_RED, lw=2)
axes[1].axhline(1, color=GRAY, ls='--')
axes[1].set_xlim(0, 12)
axes[1].set_title('Fourier Terms', fontweight='bold', color=MAIN_BLUE)
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Seasonal Effect')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=8, label='Dummies'),
    Line2D([0], [0], color=IDA_RED, lw=2, label='Fourier'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('seasonality_fourier_dummies')

# HP Filter Cycle
print("Generating ch1_hp_filter_cycle...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
n = 200
t = np.arange(n)
trend = 100 + 0.2 * t
cycle = 10 * np.sin(2 * np.pi * t / 40)
y = trend + cycle + np.random.randn(n) * 2

axes[0].plot(t, y, color=MAIN_BLUE, lw=1, alpha=0.7, label='Original')
axes[0].plot(t, uniform_filter1d(y, size=20), color=FOREST, lw=2, label='Trend')
axes[0].set_ylabel('Value')
axes[0].set_title('HP Filter: Trend Extraction', fontweight='bold', color=MAIN_BLUE, fontsize=14)

cycle_est = y - uniform_filter1d(y, size=20)
axes[1].plot(t, cycle_est, color=ORANGE, lw=1.5)
axes[1].axhline(0, color=GRAY, ls='--')
axes[1].fill_between(t, 0, cycle_est, where=(cycle_est > 0), color=FOREST, alpha=0.3)
axes[1].fill_between(t, 0, cycle_est, where=(cycle_est < 0), color=IDA_RED, alpha=0.3)
axes[1].set_ylabel('Cycle')
axes[1].set_xlabel('Time')

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Original'),
    Line2D([0], [0], color=FOREST, lw=2, label='Trend'),
    Line2D([0], [0], color=ORANGE, lw=2, label='Cycle'),
]
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.02))
save_fig('ch1_hp_filter_cycle')

print("\n" + "="*60)
print("All Chapter 0 charts generated as PDF and PNG!")
print("="*60)
