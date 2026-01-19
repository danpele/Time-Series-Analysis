#!/usr/bin/env python3
"""
Generate motivation charts for Chapters 1 and 2
Time Series Analysis Course
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import os

# Set style for transparent backgrounds
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Output directory
output_dir = 'charts'
os.makedirs(output_dir, exist_ok=True)

def save_fig(name):
    plt.savefig(f'{output_dir}/{name}.pdf', format='pdf', bbox_inches='tight',
                transparent=True, dpi=150)
    plt.close()
    print(f"  Created {name}.pdf")

print("Creating motivation charts...")

# =============================================================================
# CHAPTER 1: Introduction Motivation
# =============================================================================
print("\nChapter 1 Motivation Charts:")

# Chart 1: Time series are everywhere
def ch1_motivation_everywhere():
    np.random.seed(42)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Stock prices
    t = np.arange(250)
    stock = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, 250)))
    axes[0, 0].plot(t, stock, 'b-', linewidth=1.5)
    axes[0, 0].set_title('Stock Prices', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_xlabel('Trading Days')

    # Temperature
    t = np.arange(365)
    temp = 15 + 10*np.sin(2*np.pi*t/365) + np.random.normal(0, 2, 365)
    axes[0, 1].plot(t, temp, 'r-', linewidth=1)
    axes[0, 1].set_title('Daily Temperature', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('°C')
    axes[0, 1].set_xlabel('Day of Year')

    # Sales
    t = np.arange(48)
    trend = 100 + 2*t
    seasonal = 20*np.sin(2*np.pi*t/12)
    sales = trend + seasonal + np.random.normal(0, 5, 48)
    axes[1, 0].plot(t, sales, 'g-o', linewidth=1.5, markersize=3)
    axes[1, 0].set_title('Monthly Retail Sales', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Sales ($K)')
    axes[1, 0].set_xlabel('Month')

    # Website traffic
    t = np.arange(30)
    traffic = 5000 + 1000*np.sin(2*np.pi*t/7) + np.random.poisson(500, 30)
    axes[1, 1].bar(t, traffic, color='purple', alpha=0.7)
    axes[1, 1].set_title('Daily Website Visitors', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Visitors')
    axes[1, 1].set_xlabel('Day')

    plt.tight_layout()
    save_fig('ch1_motivation_everywhere')

ch1_motivation_everywhere()

# Chart 2: Why time series analysis matters
def ch1_motivation_forecast():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Historical data + forecast
    n_hist = 80
    n_fore = 20
    t_hist = np.arange(n_hist)
    t_fore = np.arange(n_hist, n_hist + n_fore)

    # Generate data with trend and seasonality
    trend = 50 + 0.5*t_hist
    seasonal = 10*np.sin(2*np.pi*t_hist/12)
    noise = np.random.normal(0, 3, n_hist)
    y = trend + seasonal + noise

    # Forecast
    trend_fore = 50 + 0.5*t_fore
    seasonal_fore = 10*np.sin(2*np.pi*t_fore/12)
    y_fore = trend_fore + seasonal_fore

    # Confidence intervals
    ci = np.sqrt(np.arange(1, n_fore+1)) * 3

    axes[0].plot(t_hist, y, 'b-', linewidth=1.5, label='Historical')
    axes[0].plot(t_fore, y_fore, 'r--', linewidth=2, label='Forecast')
    axes[0].fill_between(t_fore, y_fore - 1.96*ci, y_fore + 1.96*ci,
                         color='red', alpha=0.2, label='95% CI')
    axes[0].axvline(x=n_hist, color='gray', linestyle='-', alpha=0.5)
    axes[0].set_title('Forecasting Future Values', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)

    # Business decision making
    categories = ['Inventory\nPlanning', 'Budget\nAllocation', 'Risk\nManagement', 'Resource\nScheduling']
    values = [85, 78, 92, 88]
    colors = ['steelblue', 'coral', 'green', 'purple']

    axes[1].barh(categories, values, color=colors, alpha=0.7)
    axes[1].set_xlim(0, 100)
    axes[1].set_title('Business Applications of Time Series', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Importance Score (%)')

    for i, v in enumerate(values):
        axes[1].text(v + 2, i, f'{v}%', va='center', fontsize=9)

    plt.tight_layout()
    save_fig('ch1_motivation_forecast')

ch1_motivation_forecast()

# Chart 3: Components of time series
def ch1_motivation_components():
    np.random.seed(42)
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)

    t = np.arange(120)

    # Original series
    trend = 50 + 0.3*t
    seasonal = 15*np.sin(2*np.pi*t/12)
    noise = np.random.normal(0, 3, 120)
    y = trend + seasonal + noise

    axes[0].plot(t, y, 'b-', linewidth=1.5)
    axes[0].set_title('Original Time Series = Trend + Seasonal + Noise', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('$Y_t$')

    axes[1].plot(t, trend, 'r-', linewidth=2)
    axes[1].set_title('Trend Component: Long-term direction', fontsize=10)
    axes[1].set_ylabel('Trend')

    axes[2].plot(t, seasonal, 'g-', linewidth=2)
    axes[2].set_title('Seasonal Component: Repeating patterns', fontsize=10)
    axes[2].set_ylabel('Seasonal')

    axes[3].plot(t, noise, 'gray', linewidth=1, alpha=0.7)
    axes[3].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[3].set_title('Residual/Noise: Random fluctuations', fontsize=10)
    axes[3].set_ylabel('Noise')
    axes[3].set_xlabel('Time')

    plt.tight_layout()
    save_fig('ch1_motivation_components')

ch1_motivation_components()

# =============================================================================
# CHAPTER 2: ARMA Motivation
# =============================================================================
print("\nChapter 2 Motivation Charts:")

# Chart 1: Stationary series patterns
def ch2_motivation_stationary():
    np.random.seed(42)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    n = 200

    # AR(1) - mean reverting
    ar1 = np.zeros(n)
    eps = np.random.normal(0, 1, n)
    for t in range(1, n):
        ar1[t] = 0.8 * ar1[t-1] + eps[t]

    axes[0, 0].plot(ar1, 'b-', linewidth=1)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].set_title('AR(1): Mean-Reverting Behavior', fontsize=10)
    axes[0, 0].set_ylabel('$Y_t$')

    # MA(1)
    ma1 = eps[1:] + 0.7 * eps[:-1]

    axes[0, 1].plot(ma1, 'g-', linewidth=1)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('MA(1): Moving Average Process', fontsize=10)

    # ARMA(1,1)
    arma = np.zeros(n)
    for t in range(1, n):
        arma[t] = 0.6 * arma[t-1] + eps[t] + 0.4 * eps[t-1]

    axes[1, 0].plot(arma, 'purple', linewidth=1)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('ARMA(1,1): Combined Model', fontsize=10)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('$Y_t$')

    # White noise
    axes[1, 1].plot(eps, 'gray', linewidth=1, alpha=0.7)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('White Noise: No Pattern', fontsize=10)
    axes[1, 1].set_xlabel('Time')

    plt.tight_layout()
    save_fig('ch2_motivation_stationary')

ch2_motivation_stationary()

# Chart 2: Real-world stationary examples
def ch2_motivation_realworld():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    n = 150

    # Stock returns (approximately stationary)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, n)))
    returns = np.diff(np.log(prices)) * 100

    axes[0].plot(returns, 'b-', linewidth=1)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title('Stock Returns (%)', fontsize=10, fontweight='bold')
    axes[0].set_xlabel('Day')
    axes[0].set_ylabel('Return (%)')

    # Interest rate changes
    rate_changes = np.random.normal(0, 0.1, n) + 0.3 * np.random.normal(0, 0.1, n)
    for t in range(1, n):
        rate_changes[t] += 0.5 * rate_changes[t-1]

    axes[1].plot(rate_changes, 'g-', linewidth=1)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_title('Interest Rate Changes', fontsize=10, fontweight='bold')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Change (%)')

    # Inflation deviations from mean
    inflation = np.zeros(n)
    eps = np.random.normal(0, 0.3, n)
    for t in range(1, n):
        inflation[t] = 0.7 * inflation[t-1] + eps[t]

    axes[2].plot(inflation, 'r-', linewidth=1)
    axes[2].axhline(y=0, color='blue', linestyle='--', alpha=0.7)
    axes[2].set_title('Inflation Deviations from Target', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('Quarter')
    axes[2].set_ylabel('Deviation (%)')

    plt.tight_layout()
    save_fig('ch2_motivation_realworld')

ch2_motivation_realworld()

# Chart 3: Why model structure matters
def ch2_motivation_acf():
    fig, axes = plt.subplots(2, 3, figsize=(11, 5))

    lags = np.arange(0, 16)
    ci = 1.96 / np.sqrt(100)

    # AR(1) series and ACF
    np.random.seed(42)
    n = 100
    ar1 = np.zeros(n)
    eps = np.random.normal(0, 1, n)
    for t in range(1, n):
        ar1[t] = 0.8 * ar1[t-1] + eps[t]

    axes[0, 0].plot(ar1[:50], 'b-', linewidth=1)
    axes[0, 0].set_title('AR(1) Process', fontsize=9)
    axes[0, 0].set_ylabel('$Y_t$')

    acf_ar1 = 0.8 ** lags
    axes[1, 0].bar(lags, acf_ar1, color='steelblue', alpha=0.7)
    axes[1, 0].axhline(y=ci, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=-ci, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('ACF: Exponential Decay', fontsize=9)
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylim(-0.3, 1.1)

    # MA(1) series and ACF
    ma1 = eps[1:51] + 0.7 * eps[:50]

    axes[0, 1].plot(ma1, 'g-', linewidth=1)
    axes[0, 1].set_title('MA(1) Process', fontsize=9)

    acf_ma1 = np.zeros(16)
    acf_ma1[0] = 1
    acf_ma1[1] = 0.7 / (1 + 0.7**2)
    axes[1, 1].bar(lags, acf_ma1, color='green', alpha=0.7)
    axes[1, 1].axhline(y=ci, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=-ci, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('ACF: Cuts Off at Lag 1', fontsize=9)
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylim(-0.3, 1.1)

    # ARMA(1,1) series and ACF
    arma = np.zeros(n)
    for t in range(1, n):
        arma[t] = 0.6 * arma[t-1] + eps[t] + 0.4 * eps[t-1]

    axes[0, 2].plot(arma[:50], 'purple', linewidth=1)
    axes[0, 2].set_title('ARMA(1,1) Process', fontsize=9)

    acf_arma = np.zeros(16)
    acf_arma[0] = 1
    for k in range(1, 16):
        acf_arma[k] = 0.7 * 0.6**(k-1)
    axes[1, 2].bar(lags, acf_arma, color='purple', alpha=0.7)
    axes[1, 2].axhline(y=ci, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].axhline(y=-ci, color='red', linestyle='--', alpha=0.5)
    axes[1, 2].set_title('ACF: Decay After Lag 1', fontsize=9)
    axes[1, 2].set_xlabel('Lag')
    axes[1, 2].set_ylim(-0.3, 1.1)

    plt.tight_layout()
    save_fig('ch2_motivation_acf')

ch2_motivation_acf()

print("\nAll motivation charts created successfully!")
