"""
TSA_ch0_additive_mult
=====================
Additive vs Multiplicative Decomposition

This script demonstrates when to use additive vs multiplicative decomposition:
- Additive: X_t = T_t + S_t + e_t (constant seasonal amplitude)
- Multiplicative: X_t = T_t * S_t * e_t (seasonal amplitude grows with level)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Generate time index (5 years of monthly data)
n = 60
t = np.arange(n)

# Trend component
trend = 100 + 2 * t

# Seasonal pattern (12-month cycle)
seasonal_pattern = np.array([0.9, 0.85, 0.95, 1.0, 1.05, 1.1, 1.15, 1.1, 1.05, 1.0, 0.95, 0.9])
seasonal = np.tile(seasonal_pattern, n // 12 + 1)[:n]

# Noise
noise_add = np.random.normal(0, 5, n)
noise_mult = np.random.normal(0, 0.03, n)

# Additive model: constant seasonal amplitude
additive = trend + 20 * (seasonal - 1) + noise_add

# Multiplicative model: seasonal amplitude grows with trend
multiplicative = trend * seasonal * (1 + noise_mult)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Additive time series
axes[0, 0].plot(additive, 'b-', linewidth=1.5, label='Observed')
axes[0, 0].plot(trend, 'r--', linewidth=2, label='Trend')
axes[0, 0].set_title('Additive: $X_t = T_t + S_t + \\varepsilon_t$', fontsize=12)
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Value')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
axes[0, 0].grid(True, alpha=0.3)

# Multiplicative time series
axes[0, 1].plot(multiplicative, 'g-', linewidth=1.5, label='Observed')
axes[0, 1].plot(trend, 'r--', linewidth=2, label='Trend')
axes[0, 1].set_title('Multiplicative: $X_t = T_t \\times S_t \\times \\varepsilon_t$', fontsize=12)
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Value')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
axes[0, 1].grid(True, alpha=0.3)

# Seasonal deviations - Additive
add_deviations = additive - trend
axes[1, 0].plot(add_deviations, 'b-', alpha=0.7)
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_title('Additive: Seasonal Deviations (Constant Amplitude)', fontsize=12)
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Deviation from Trend')
axes[1, 0].set_ylim(-40, 40)
axes[1, 0].grid(True, alpha=0.3)

# Seasonal deviations - Multiplicative (as ratio)
mult_ratios = multiplicative / trend
axes[1, 1].plot(mult_ratios, 'g-', alpha=0.7)
axes[1, 1].axhline(y=1, color='r', linestyle='--')
axes[1, 1].set_title('Multiplicative: Seasonal Ratios (Constant Ratio)', fontsize=12)
axes[1, 1].set_xlabel('Month')
axes[1, 1].set_ylabel('Ratio to Trend')
axes[1, 1].set_ylim(0.7, 1.3)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../charts/additive_vs_multiplicative.png', dpi=150, bbox_inches='tight')
plt.show()

print("When to use each decomposition:")
print("\nADDITIVE:")
print("- Seasonal fluctuations have constant amplitude")
print("- Variance is stable over time")
print("- Example: Temperature data")

print("\nMULTIPLICATIVE:")
print("- Seasonal fluctuations grow with the level")
print("- 'Fan' pattern in the data")
print("- Example: Retail sales, airline passengers")
