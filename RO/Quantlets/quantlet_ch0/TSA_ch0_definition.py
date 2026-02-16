"""
TSA_ch0_definition
==================
Time Series Definition and Basic Concepts

This script demonstrates the fundamental characteristics of time series data:
- Temporal ordering
- Autocorrelation (dependence between observations)
- Trend, seasonality, and noise components
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample time series with different characteristics
n = 200
t = np.arange(n)

# 1. White noise (independent observations)
white_noise = np.random.normal(0, 1, n)

# 2. Time series with autocorrelation (AR(1) process)
ar1 = np.zeros(n)
phi = 0.8
for i in range(1, n):
    ar1[i] = phi * ar1[i-1] + np.random.normal(0, 1)

# 3. Time series with trend and seasonality
trend = 0.05 * t
seasonality = 2 * np.sin(2 * np.pi * t / 12)
ts_complex = trend + seasonality + 0.5 * np.random.normal(0, 1, n)

# Create figure
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Plot time series
axes[0, 0].plot(white_noise, 'b-', alpha=0.7)
axes[0, 0].set_title('White Noise (Independent)', fontsize=12)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Value')
axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

axes[1, 0].plot(ar1, 'g-', alpha=0.7)
axes[1, 0].set_title('AR(1) Process (Autocorrelated)', fontsize=12)
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Value')

axes[2, 0].plot(ts_complex, 'purple', alpha=0.7)
axes[2, 0].set_title('Trend + Seasonality + Noise', fontsize=12)
axes[2, 0].set_xlabel('Time')
axes[2, 0].set_ylabel('Value')

# Plot ACF
plot_acf(white_noise, ax=axes[0, 1], lags=30, title='ACF: White Noise')
plot_acf(ar1, ax=axes[1, 1], lags=30, title='ACF: AR(1)')
plot_acf(ts_complex, ax=axes[2, 1], lags=30, title='ACF: Complex Series')

plt.tight_layout()
plt.savefig('../../charts/timeseries_definition.png', dpi=150, bbox_inches='tight')
plt.show()

print("Key Characteristics of Time Series:")
print("1. Observations are ordered in time")
print("2. Consecutive observations are usually correlated (autocorrelated)")
print("3. NOT independent and identically distributed (unlike cross-sectional data)")
print(f"\nAR(1) autocorrelation at lag 1: {np.corrcoef(ar1[:-1], ar1[1:])[0,1]:.3f}")
