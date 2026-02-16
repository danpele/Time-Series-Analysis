"""
TSA_ch1_intro
=============
Introduction to Time Series Analysis

This script demonstrates:
- Time series patterns (trend, seasonality, cycles)
- Stationarity concepts
- White noise and random walk
- ACF/PACF basics
- Decomposition methods

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
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
    plt.savefig(f'{name}.pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"   Saved: {name}.pdf")

print("=" * 70)
print("INTRODUCTION TO TIME SERIES ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. Time Series Components
# =============================================================================
np.random.seed(42)
n = 144  # 12 years of monthly data

print("\n1. TIME SERIES COMPONENTS")
print("-" * 40)

t = np.arange(n)
dates = pd.date_range('2012-01-01', periods=n, freq='M')

# Components
trend = 50 + 0.3 * t
seasonality = 10 * np.sin(2 * np.pi * t / 12)
cycle = 5 * np.sin(2 * np.pi * t / 36)
noise = np.random.normal(0, 3, n)

# Combined series
y = trend + seasonality + cycle + noise

fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

axes[0].plot(dates, y, color='#1A3A6E', linewidth=1.5, label='Observed')
axes[0].set_title('Observed Time Series', fontweight='bold')
axes[0].set_ylabel('Value')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

axes[1].plot(dates, trend, color='#DC3545', linewidth=2, label='Trend')
axes[1].set_title('Trend Component', fontweight='bold')
axes[1].set_ylabel('Value')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

axes[2].plot(dates, seasonality, color='#2E7D32', linewidth=2, label='Seasonality')
axes[2].set_title('Seasonal Component (Period = 12)', fontweight='bold')
axes[2].set_ylabel('Value')
axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

axes[3].plot(dates, cycle, color='#E67E22', linewidth=2, label='Cycle')
axes[3].set_title('Cyclical Component (Period = 36)', fontweight='bold')
axes[3].set_ylabel('Value')
axes[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

axes[4].plot(dates, noise, color='#666666', linewidth=1, label='Noise')
axes[4].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[4].set_title('Random Component (Noise)', fontweight='bold')
axes[4].set_ylabel('Value')
axes[4].set_xlabel('Date')
axes[4].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plt.tight_layout()
save_fig('ch1_components')

# =============================================================================
# 2. Stationarity Comparison
# =============================================================================
print("\n2. STATIONARITY")
print("-" * 40)

n_stat = 200

# Stationary series
stationary = np.random.normal(0, 1, n_stat)

# Non-stationary: random walk
random_walk = np.cumsum(np.random.normal(0, 1, n_stat))

# Non-stationary: trend
trend_series = 0.1 * np.arange(n_stat) + np.random.normal(0, 1, n_stat)

# Non-stationary: changing variance
changing_var = np.concatenate([
    np.random.normal(0, 1, n_stat//2),
    np.random.normal(0, 3, n_stat//2)
])

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(stationary, color='#2E7D32', linewidth=1, label='Stationary')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 0].set_title('Stationary Series', fontweight='bold')
axes[0, 0].set_xlabel('Time')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

axes[0, 1].plot(random_walk, color='#DC3545', linewidth=1, label='Random Walk')
axes[0, 1].set_title('Non-Stationary: Random Walk', fontweight='bold')
axes[0, 1].set_xlabel('Time')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

axes[1, 0].plot(trend_series, color='#DC3545', linewidth=1, label='Trend')
axes[1, 0].set_title('Non-Stationary: Trend', fontweight='bold')
axes[1, 0].set_xlabel('Time')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

axes[1, 1].plot(changing_var, color='#DC3545', linewidth=1, label='Changing Variance')
axes[1, 1].axvline(x=n_stat//2, color='gray', linestyle=':', linewidth=2)
axes[1, 1].set_title('Non-Stationary: Changing Variance', fontweight='bold')
axes[1, 1].set_xlabel('Time')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plt.tight_layout()
save_fig('ch1_stationarity')

# ADF tests
print("\n   ADF Test Results:")
for name, series in [('Stationary', stationary), ('Random Walk', random_walk),
                     ('Trend', trend_series), ('Changing Var', changing_var)]:
    adf_stat, p_value = adfuller(series)[:2]
    status = "Stationary" if p_value < 0.05 else "Non-stationary"
    print(f"   {name:15s}: ADF = {adf_stat:7.3f}, p = {p_value:.4f} → {status}")

# =============================================================================
# 3. White Noise vs Random Walk
# =============================================================================
print("\n3. WHITE NOISE VS RANDOM WALK")
print("-" * 40)

n_wn = 500
white_noise = np.random.normal(0, 1, n_wn)
random_walk = np.cumsum(white_noise)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# White noise
axes[0, 0].plot(white_noise, color='#1A3A6E', linewidth=0.8, label='εₜ ~ N(0,1)')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 0].set_title('White Noise: εₜ ~ iid N(0,1)', fontweight='bold')
axes[0, 0].set_xlabel('Time')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# Random walk
axes[0, 1].plot(random_walk, color='#DC3545', linewidth=1, label='Yₜ = Yₜ₋₁ + εₜ')
axes[0, 1].set_title('Random Walk: Yₜ = Yₜ₋₁ + εₜ', fontweight='bold')
axes[0, 1].set_xlabel('Time')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# White noise histogram
axes[1, 0].hist(white_noise, bins=40, density=True, color='#1A3A6E', alpha=0.7, edgecolor='white', label='White Noise')
x = np.linspace(-4, 4, 100)
axes[1, 0].plot(x, 1/np.sqrt(2*np.pi) * np.exp(-x**2/2), 'r-', linewidth=2, label='N(0,1)')
axes[1, 0].set_title('White Noise Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Value')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Multiple random walks
for i in range(10):
    rw = np.cumsum(np.random.normal(0, 1, n_wn))
    axes[1, 1].plot(rw, linewidth=0.8, alpha=0.7)
axes[1, 1].set_title('Multiple Random Walk Realizations', fontweight='bold')
axes[1, 1].set_xlabel('Time')

plt.tight_layout()
save_fig('ch1_whitenoise_randomwalk')

# =============================================================================
# 4. ACF and PACF
# =============================================================================
print("\n4. ACF AND PACF")
print("-" * 40)

# Generate different processes
ar1 = np.zeros(200)
for t in range(1, 200):
    ar1[t] = 0.8 * ar1[t-1] + np.random.normal(0, 1)

ma1 = np.zeros(200)
eps = np.random.normal(0, 1, 200)
for t in range(1, 200):
    ma1[t] = eps[t] + 0.8 * eps[t-1]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# AR(1) series
axes[0, 0].plot(ar1, color='#1A3A6E', linewidth=1, label='AR(1): φ=0.8')
axes[0, 0].set_title('AR(1) Process: φ = 0.8', fontweight='bold')
axes[0, 0].set_xlabel('Time')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# AR(1) ACF
plot_acf(ar1, lags=20, ax=axes[0, 1], color='#1A3A6E', title='')
axes[0, 1].set_title('ACF of AR(1): Geometric Decay', fontweight='bold')

# AR(1) PACF
plot_pacf(ar1, lags=20, ax=axes[0, 2], color='#1A3A6E', title='')
axes[0, 2].set_title('PACF of AR(1): Cuts off at lag 1', fontweight='bold')

# MA(1) series
axes[1, 0].plot(ma1, color='#DC3545', linewidth=1, label='MA(1): θ=0.8')
axes[1, 0].set_title('MA(1) Process: θ = 0.8', fontweight='bold')
axes[1, 0].set_xlabel('Time')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# MA(1) ACF
plot_acf(ma1, lags=20, ax=axes[1, 1], color='#DC3545', title='')
axes[1, 1].set_title('ACF of MA(1): Cuts off at lag 1', fontweight='bold')

# MA(1) PACF
plot_pacf(ma1, lags=20, ax=axes[1, 2], color='#DC3545', title='')
axes[1, 2].set_title('PACF of MA(1): Geometric Decay', fontweight='bold')

plt.tight_layout()
save_fig('ch1_acf_pacf')

# =============================================================================
# 5. Seasonal Decomposition
# =============================================================================
print("\n5. SEASONAL DECOMPOSITION")
print("-" * 40)

# Create seasonal time series
ts = pd.Series(y, index=dates)

# Additive decomposition
decomposition = seasonal_decompose(ts, model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(decomposition.observed, color='#1A3A6E', linewidth=1.5, label='Observed')
axes[0].set_title('Observed', fontweight='bold')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

axes[1].plot(decomposition.trend, color='#DC3545', linewidth=2, label='Trend')
axes[1].set_title('Trend', fontweight='bold')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

axes[2].plot(decomposition.seasonal, color='#2E7D32', linewidth=1.5, label='Seasonal')
axes[2].set_title('Seasonal', fontweight='bold')
axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

axes[3].plot(decomposition.resid, color='#666666', linewidth=1, label='Residual')
axes[3].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[3].set_title('Residual', fontweight='bold')
axes[3].set_xlabel('Date')
axes[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plt.suptitle('Additive Decomposition: Y = T + S + R', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
save_fig('ch1_decomposition')

# =============================================================================
# 6. Moving Average Smoothing
# =============================================================================
print("\n6. MOVING AVERAGE SMOOTHING")
print("-" * 40)

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(dates, y, color='#CCCCCC', linewidth=1, label='Original')

windows = [3, 6, 12]
colors = ['#1A3A6E', '#DC3545', '#2E7D32']

for window, color in zip(windows, colors):
    ma = pd.Series(y).rolling(window=window, center=True).mean()
    ax.plot(dates, ma, color=color, linewidth=2, label=f'MA({window})')

ax.set_title('Moving Average Smoothing', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)

plt.tight_layout()
save_fig('ch1_moving_average')

print("\n" + "=" * 70)
print("INTRODUCTION ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch1_components.pdf: Time series components")
print("  - ch1_stationarity.pdf: Stationarity comparison")
print("  - ch1_whitenoise_randomwalk.pdf: White noise vs random walk")
print("  - ch1_acf_pacf.pdf: ACF/PACF patterns")
print("  - ch1_decomposition.pdf: Seasonal decomposition")
print("  - ch1_moving_average.pdf: Moving average smoothing")
