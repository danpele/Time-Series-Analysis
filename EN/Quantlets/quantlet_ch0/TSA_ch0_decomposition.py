"""
TSA_ch0_decomposition
=====================
Classical Time Series Decomposition

This script demonstrates:
- Additive decomposition: X_t = T_t + S_t + e_t
- Multiplicative decomposition: X_t = T_t * S_t * e_t
- Using statsmodels seasonal_decompose
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Set random seed
np.random.seed(42)

# Load classic airline passengers dataset (or simulate similar)
# Monthly data from 1949-1960
n = 144  # 12 years of monthly data
t = np.arange(n)

# Create realistic airline passenger-like data
trend = 100 + 2.5 * t + 0.01 * t**2  # Accelerating growth
seasonal_pattern = np.array([0.85, 0.82, 0.95, 0.98, 1.02, 1.15,
                             1.25, 1.22, 1.08, 0.98, 0.88, 0.83])
seasonal = np.tile(seasonal_pattern, n // 12)
noise = np.random.normal(1, 0.02, n)

# Multiplicative model (typical for economic data)
data = trend * seasonal * noise

# Create date index
dates = pd.date_range(start='1949-01', periods=n, freq='ME')
ts = pd.Series(data, index=dates, name='Passengers')

# Perform decomposition
decomp_mult = seasonal_decompose(ts, model='multiplicative', period=12)
decomp_add = seasonal_decompose(ts, model='additive', period=12)

# Create figure comparing both methods
fig, axes = plt.subplots(4, 2, figsize=(14, 12))

# Multiplicative decomposition
axes[0, 0].plot(ts, 'b-', linewidth=1)
axes[0, 0].set_title('Observed (Multiplicative)', fontsize=11)
axes[0, 0].set_ylabel('Value')

axes[1, 0].plot(decomp_mult.trend, 'r-', linewidth=1.5)
axes[1, 0].set_title('Trend', fontsize=11)
axes[1, 0].set_ylabel('Value')

axes[2, 0].plot(decomp_mult.seasonal, 'g-', linewidth=1)
axes[2, 0].set_title('Seasonal (Ratios around 1)', fontsize=11)
axes[2, 0].set_ylabel('Factor')
axes[2, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)

axes[3, 0].plot(decomp_mult.resid, 'purple', linewidth=0.5, alpha=0.7)
axes[3, 0].set_title('Residual', fontsize=11)
axes[3, 0].set_ylabel('Factor')
axes[3, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
axes[3, 0].set_xlabel('Year')

# Additive decomposition (for comparison)
axes[0, 1].plot(ts, 'b-', linewidth=1)
axes[0, 1].set_title('Observed (Additive)', fontsize=11)

axes[1, 1].plot(decomp_add.trend, 'r-', linewidth=1.5)
axes[1, 1].set_title('Trend', fontsize=11)

axes[2, 1].plot(decomp_add.seasonal, 'g-', linewidth=1)
axes[2, 1].set_title('Seasonal (Deviations around 0)', fontsize=11)
axes[2, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

axes[3, 1].plot(decomp_add.resid, 'purple', linewidth=0.5, alpha=0.7)
axes[3, 1].set_title('Residual (Note: Pattern remains!)', fontsize=11, color='red')
axes[3, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[3, 1].set_xlabel('Year')

fig.suptitle('Decomposition Comparison: Multiplicative vs Additive', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('../../charts/ch1_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary
print("=" * 60)
print("Time Series Decomposition Summary")
print("=" * 60)
print("\nMultiplicative Model: X_t = T_t × S_t × e_t")
print("- Seasonal component: ratios centered around 1")
print("- Use when seasonal amplitude grows with level")
print(f"- Seasonal range: {decomp_mult.seasonal.min():.3f} to {decomp_mult.seasonal.max():.3f}")

print("\nAdditive Model: X_t = T_t + S_t + e_t")
print("- Seasonal component: deviations from trend")
print("- Use when seasonal amplitude is constant")
print(f"- Seasonal range: {decomp_add.seasonal.min():.1f} to {decomp_add.seasonal.max():.1f}")

print("\nDiagnostic: Check residuals for remaining patterns")
print("If residuals show pattern → wrong model choice!")
