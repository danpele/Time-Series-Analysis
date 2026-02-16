"""
TSA_ch1_acf_patterns
====================
ACF Patterns for Different Time Series

This script demonstrates:
- ACF of white noise (no significant lags)
- ACF of AR(1) process (exponential decay)
- ACF of random walk (slow decay)
- ACF of seasonal data (periodic peaks)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Set random seed
np.random.seed(42)

n = 500
max_lag = 30

def compute_acf(series, max_lag):
    """Compute ACF values"""
    return acf(series, nlags=max_lag, fft=True)

# Generate different series
# 1. White noise
white_noise = np.random.normal(0, 1, n)

# 2. AR(1) with positive phi
ar1_pos = np.zeros(n)
phi = 0.8
for t in range(1, n):
    ar1_pos[t] = phi * ar1_pos[t-1] + np.random.normal(0, 1)

# 3. AR(1) with negative phi
ar1_neg = np.zeros(n)
phi_neg = -0.7
for t in range(1, n):
    ar1_neg[t] = phi_neg * ar1_neg[t-1] + np.random.normal(0, 1)

# 4. Random walk
random_walk = np.cumsum(np.random.normal(0, 1, n))

# 5. Seasonal (period 12)
t = np.arange(n)
seasonal = 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1, n)

# 6. MA(1)
ma1 = np.zeros(n)
theta = 0.8
eps = np.random.normal(0, 1, n+1)
for t in range(n):
    ma1[t] = eps[t+1] + theta * eps[t]

# Compute ACFs
acf_wn = compute_acf(white_noise, max_lag)
acf_ar1_pos = compute_acf(ar1_pos, max_lag)
acf_ar1_neg = compute_acf(ar1_neg, max_lag)
acf_rw = compute_acf(random_walk, max_lag)
acf_seasonal = compute_acf(seasonal, max_lag)
acf_ma1 = compute_acf(ma1, max_lag)

# Confidence interval
ci = 1.96 / np.sqrt(n)
lags = np.arange(max_lag + 1)

# Plot function
def plot_acf(ax, acf_values, title, color, pattern_desc):
    ax.bar(lags, acf_values, color=color, alpha=0.7, edgecolor='black', width=0.8)
    ax.axhline(y=ci, color='red', linestyle='--', linewidth=1)
    ax.axhline(y=-ci, color='red', linestyle='--', linewidth=1)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title(f'{title}\n{pattern_desc}', fontsize=11)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_xlim(-0.5, max_lag + 0.5)

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 11))

# 1. White noise
plot_acf(axes[0, 0], acf_wn, 'White Noise', 'blue', 'No significant lags')

# 2. AR(1) positive
plot_acf(axes[0, 1], acf_ar1_pos, f'AR(1), phi = {phi}', 'green', 'Exponential decay (positive)')

# 3. AR(1) negative
plot_acf(axes[0, 2], acf_ar1_neg, f'AR(1), phi = {phi_neg}', 'purple', 'Alternating decay')

# 4. Random walk
plot_acf(axes[1, 0], acf_rw, 'Random Walk', 'red', 'Very slow decay - Non-stationary!')

# 5. Seasonal
plot_acf(axes[1, 1], acf_seasonal, 'Seasonal (period=12)', 'orange', 'Peaks at seasonal lags')

# 6. MA(1)
plot_acf(axes[1, 2], acf_ma1, f'MA(1), theta = {theta}', 'brown', 'Cuts off after lag 1')

plt.tight_layout()

# Add legend outside bottom
fig.legend(['ACF', '95% CI'], loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
plt.subplots_adjust(bottom=0.1)

# Set transparent background
fig.patch.set_facecolor('none')
for ax in fig.axes:
    ax.patch.set_facecolor('none')

# Save both PDF and PNG with transparent background
plt.savefig('../../charts/ch1_acf_examples.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_acf_examples.png', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_acf_examples.pdf', transparent=True, bbox_inches='tight')
plt.show()

# Print ACF interpretation guide
print("=" * 70)
print("ACF PATTERN INTERPRETATION GUIDE")
print("=" * 70)
print("""
Pattern                  | Likely Process        | Action
-------------------------|----------------------|------------------------
No significant lags      | White noise          | No modeling needed
Exponential decay        | AR(p) process        | Fit AR model
Alternating decay        | AR with negative phi | Fit AR model
Cuts off after lag q     | MA(q) process        | Fit MA model
Very slow decay          | Non-stationary       | Difference first!
Peaks at seasonal lags   | Seasonal component   | Seasonal differencing

Key Points:
1. ACF at lag 0 is always 1 (correlation with itself)
2. Values outside +/-1.96/sqrt(n) are significant at 5% level
3. Slow decay suggests unit root - use differencing
4. Seasonal peaks at lags s, 2s, 3s... indicate seasonality
""")
