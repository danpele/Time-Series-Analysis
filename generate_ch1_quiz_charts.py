#!/usr/bin/env python3
"""
Generate charts for Chapter 1 lecture quiz answers
Time Series Analysis Course
"""

import numpy as np
import matplotlib.pyplot as plt
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

print("Chapter 1: Introduction Quiz Charts")

# Quiz 1: Time Series Components
def ch1_quiz1_components():
    np.random.seed(42)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    t = np.arange(120)

    # Trend
    trend = 0.1 * t
    axes[0, 0].plot(t, trend, 'b-', linewidth=2)
    axes[0, 0].set_title('Trend Component', fontsize=10)
    axes[0, 0].set_xlabel('Time')

    # Seasonality
    seasonal = 3 * np.sin(2 * np.pi * t / 12)
    axes[0, 1].plot(t, seasonal, 'g-', linewidth=2)
    axes[0, 1].set_title('Seasonal Component (s=12)', fontsize=10)
    axes[0, 1].set_xlabel('Time')

    # Noise
    noise = np.random.normal(0, 1, 120)
    axes[1, 0].plot(t, noise, 'r-', linewidth=1, alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Random/Irregular Component', fontsize=10)
    axes[1, 0].set_xlabel('Time')

    # Combined
    combined = trend + seasonal + noise
    axes[1, 1].plot(t, combined, 'purple', linewidth=1.5)
    axes[1, 1].set_title('Combined: Trend + Seasonal + Noise', fontsize=10)
    axes[1, 1].set_xlabel('Time')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch1_quiz1_components')

ch1_quiz1_components()

# Quiz 2: Stationary vs Non-Stationary
def ch1_quiz2_stationarity():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n = 150

    # Stationary process
    stationary = np.zeros(n)
    eps = np.random.normal(0, 1, n)
    for t in range(1, n):
        stationary[t] = 0.7 * stationary[t-1] + eps[t]

    axes[0].plot(stationary, 'g-', linewidth=1.5)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].axhline(y=2, color='orange', linestyle=':', alpha=0.7)
    axes[0].axhline(y=-2, color='orange', linestyle=':', alpha=0.7)
    axes[0].fill_between(range(n), -2, 2, alpha=0.1, color='green')
    axes[0].set_title('Stationary: Constant Mean & Variance', fontsize=10, color='green')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')

    # Non-stationary (random walk)
    random_walk = np.cumsum(np.random.normal(0, 1, n))

    axes[1].plot(random_walk, 'r-', linewidth=1.5)
    axes[1].axhline(y=0, color='blue', linestyle='--', alpha=0.5)
    axes[1].set_title('Non-Stationary: Wandering Mean', fontsize=10, color='red')
    axes[1].set_xlabel('Time')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch1_quiz2_stationarity')

ch1_quiz2_stationarity()

# Quiz 3: ACF Patterns
def ch1_quiz3_acf():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    lags = np.arange(0, 16)
    ci = 1.96 / np.sqrt(100)

    # White noise ACF
    np.random.seed(42)
    acf_wn = np.zeros(16)
    acf_wn[0] = 1
    acf_wn[1:] = np.random.normal(0, 0.08, 15)

    axes[0, 0].bar(lags, acf_wn, color='steelblue', alpha=0.7)
    axes[0, 0].axhline(y=ci, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].axhline(y=-ci, color='red', linestyle='--', alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 0].set_title('White Noise: No Autocorrelation', fontsize=9)
    axes[0, 0].set_ylim(-0.4, 1.1)

    # AR(1) ACF - exponential decay
    phi = 0.8
    acf_ar = phi ** lags

    axes[0, 1].bar(lags, acf_ar, color='green', alpha=0.7)
    axes[0, 1].axhline(y=ci, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axhline(y=-ci, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].set_title('AR(1): Exponential Decay', fontsize=9)
    axes[0, 1].set_ylim(-0.4, 1.1)

    # MA(1) ACF - cuts off after lag 1
    acf_ma = np.zeros(16)
    acf_ma[0] = 1
    acf_ma[1] = 0.6

    axes[1, 0].bar(lags, acf_ma, color='orange', alpha=0.7)
    axes[1, 0].axhline(y=ci, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].axhline(y=-ci, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].set_title('MA(1): Cuts Off After Lag 1', fontsize=9)
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylim(-0.4, 1.1)

    # Random walk ACF - slow decay
    acf_rw = np.array([1.0, 0.95, 0.90, 0.85, 0.80, 0.76, 0.72, 0.68,
                       0.64, 0.61, 0.58, 0.55, 0.52, 0.49, 0.46, 0.44])

    axes[1, 1].bar(lags, acf_rw, color='red', alpha=0.7)
    axes[1, 1].axhline(y=ci, color='blue', linestyle='--', alpha=0.7)
    axes[1, 1].axhline(y=-ci, color='blue', linestyle='--', alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].set_title('Random Walk: Very Slow Decay', fontsize=9)
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylim(-0.4, 1.1)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch1_quiz3_acf')

ch1_quiz3_acf()

# Quiz 4: White Noise vs Random Walk
def ch1_quiz4_wn_rw():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n = 100

    # White noise
    wn = np.random.normal(0, 1, n)

    axes[0].plot(wn, 'b-', linewidth=1, alpha=0.8)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2, label='Mean = 0')
    axes[0].axhline(y=2, color='orange', linestyle=':', alpha=0.7)
    axes[0].axhline(y=-2, color='orange', linestyle=':', alpha=0.7)
    axes[0].set_title('White Noise: $\\varepsilon_t \\sim N(0, \\sigma^2)$\nConstant Mean & Variance', fontsize=10)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')

    # Random walk
    rw = np.cumsum(wn)

    axes[1].plot(rw, 'r-', linewidth=1.5)
    axes[1].axhline(y=0, color='blue', linestyle='--', alpha=0.5)
    axes[1].set_title('Random Walk: $Y_t = Y_{t-1} + \\varepsilon_t$\nVariance Grows with Time', fontsize=10)
    axes[1].set_xlabel('Time')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch1_quiz4_wn_rw')

ch1_quiz4_wn_rw()

# Quiz 5: Forecast Error Metrics
def ch1_quiz5_forecast_errors():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n = 20
    t = np.arange(n)
    actual = 100 + 2*t + 5*np.sin(2*np.pi*t/6) + np.random.normal(0, 3, n)

    # Good forecast
    forecast_good = actual + np.random.normal(0, 2, n)
    errors_good = actual - forecast_good

    axes[0].plot(t, actual, 'b-o', label='Actual', markersize=5)
    axes[0].plot(t, forecast_good, 'g--s', label='Forecast', markersize=4)
    axes[0].fill_between(t, actual, forecast_good, alpha=0.3, color='green')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')

    mae_good = np.mean(np.abs(errors_good))
    rmse_good = np.sqrt(np.mean(errors_good**2))
    axes[0].set_title(f'Good Forecast\nMAE={mae_good:.2f}, RMSE={rmse_good:.2f}', fontsize=10, color='green')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)

    # Bad forecast
    forecast_bad = actual + np.random.normal(5, 8, n)
    errors_bad = actual - forecast_bad

    axes[1].plot(t, actual, 'b-o', label='Actual', markersize=5)
    axes[1].plot(t, forecast_bad, 'r--s', label='Forecast', markersize=4)
    axes[1].fill_between(t, actual, forecast_bad, alpha=0.3, color='red')
    axes[1].set_xlabel('Time')

    mae_bad = np.mean(np.abs(errors_bad))
    rmse_bad = np.sqrt(np.mean(errors_bad**2))
    axes[1].set_title(f'Poor Forecast\nMAE={mae_bad:.2f}, RMSE={rmse_bad:.2f}', fontsize=10, color='red')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch1_quiz5_forecast_errors')

ch1_quiz5_forecast_errors()

# Quiz 6: Decomposition Types
def ch1_quiz6_decomposition():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    t = np.arange(48)
    trend = 10 + 0.2 * t
    seasonal = 3 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 0.5, 48)

    # Additive: Y = T + S + e
    y_add = trend + seasonal + noise

    axes[0].plot(t, y_add, 'b-', linewidth=1.5, label='Y = T + S + ε')
    axes[0].plot(t, trend, 'r--', linewidth=2, label='Trend', alpha=0.7)
    axes[0].set_title('Additive: Y = T + S + ε\nConstant Seasonal Amplitude', fontsize=10)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)

    # Multiplicative: Y = T * S * e
    seasonal_mult = 1 + 0.2 * np.sin(2 * np.pi * t / 12)
    noise_mult = 1 + np.random.normal(0, 0.05, 48)
    y_mult = trend * seasonal_mult * noise_mult

    axes[1].plot(t, y_mult, 'g-', linewidth=1.5, label='Y = T × S × ε')
    axes[1].plot(t, trend, 'r--', linewidth=2, label='Trend', alpha=0.7)
    axes[1].set_title('Multiplicative: Y = T × S × ε\nSeasonal Amplitude Grows', fontsize=10)
    axes[1].set_xlabel('Time')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_fig('ch1_quiz6_decomposition')

ch1_quiz6_decomposition()

print("\nAll Chapter 1 quiz charts created successfully!")
