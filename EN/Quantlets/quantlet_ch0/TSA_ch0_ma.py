"""
TSA_ch0_ma
==========
Moving Averages for Trend Extraction

This script demonstrates:
- Centered moving averages
- Effect of window size
- MA-k uses (k-1)/2 observations on each side
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Generate sample data with trend and seasonality
n = 120
t = np.arange(n)
trend = 50 + 0.3 * t
seasonality = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 3, n)
data = trend + seasonality + noise

def centered_moving_average(data, k):
    """Centered Moving Average of order k"""
    if k % 2 == 0:
        # For even k, use 2xMA
        ma1 = pd.Series(data).rolling(window=k, center=True).mean()
        ma2 = ma1.rolling(window=2, center=True).mean()
        return ma2.values
    else:
        return pd.Series(data).rolling(window=k, center=True).mean().values

# Apply different window sizes
windows = [3, 5, 7, 12]
colors = ['blue', 'green', 'orange', 'red']

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, k in enumerate(windows):
    ma = centered_moving_average(data, k)

    axes[i].plot(data, 'gray', alpha=0.4, linewidth=1, label='Observed')
    axes[i].plot(trend, 'black', linestyle='--', linewidth=1, alpha=0.5, label='True Trend')
    axes[i].plot(ma, colors[i], linewidth=2, label=f'MA-{k}')
    axes[i].set_title(f'Centered MA-{k}: uses {(k-1)//2} obs. each side of t', fontsize=12)
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Value')
    axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../charts/ch1_moving_average.png', dpi=150, bbox_inches='tight')
plt.show()

# Demonstrate centered MA calculation
print("Centered Moving Average Formula:")
print("MA-k centered at t uses observations from t-(k-1)/2 to t+(k-1)/2")
print("\nExample: MA-5 centered at t uses:")
print("X_{t-2}, X_{t-1}, X_t, X_{t+1}, X_{t+2}")
print("\nMA-5 = (X_{t-2} + X_{t-1} + X_t + X_{t+1} + X_{t+2}) / 5")

print("\nKey Points:")
print("- Larger window = smoother curve but more lag")
print("- Loses (k-1)/2 observations at each end")
print("- MA-12 removes annual seasonality from monthly data")
