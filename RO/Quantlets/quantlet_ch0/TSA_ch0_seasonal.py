"""
TSA_ch0_seasonal
================
Seasonal Adjustment and Deseasonalization

This script demonstrates:
- Extracting seasonal indices
- Deseasonalizing data
- Seasonal adjustment methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Set random seed
np.random.seed(42)

# Generate seasonal data
n = 48  # 4 years of monthly data
t = np.arange(n)

# Create components
trend = 100 + 0.8 * t
seasonal_pattern = np.array([0.85, 0.80, 0.90, 1.00, 1.05, 1.15,
                             1.20, 1.18, 1.05, 0.95, 0.90, 0.88])
seasonal = np.tile(seasonal_pattern, n // 12)
noise = np.random.normal(1, 0.02, n)

# Multiplicative model
data = trend * seasonal * noise

# Create pandas series
dates = pd.date_range(start='2020-01', periods=n, freq='ME')
ts = pd.Series(data, index=dates)

# Decompose
decomp = seasonal_decompose(ts, model='multiplicative', period=12)

# Get seasonal indices (average for each month)
seasonal_indices = decomp.seasonal[:12].values
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Deseasonalized series
deseasonalized = ts / decomp.seasonal

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Original data
ax1 = axes[0, 0]
ax1.plot(ts.index, ts, 'b-', linewidth=1.5, label='Original')
ax1.plot(decomp.trend.index, decomp.trend, 'r--', linewidth=2, label='Trend')
ax1.set_title('Original Data with Clear Seasonality', fontsize=12)
ax1.set_xlabel('Date')
ax1.set_ylabel('Value')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax1.grid(True, alpha=0.3)

# Plot 2: Seasonal indices
ax2 = axes[0, 1]
colors = ['blue' if s < 1 else 'green' for s in seasonal_indices]
bars = ax2.bar(months, seasonal_indices, color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='No Effect (1.0)')
ax2.set_title('Seasonal Indices by Month', fontsize=12)
ax2.set_xlabel('Month')
ax2.set_ylabel('Seasonal Index')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, seasonal_indices):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.2f}', ha='center', va='bottom', fontsize=8)

# Plot 3: Original vs Deseasonalized
ax3 = axes[1, 0]
ax3.plot(ts.index, ts, 'b-', linewidth=1, alpha=0.5, label='Original')
ax3.plot(deseasonalized.index, deseasonalized, 'g-', linewidth=1.5, label='Deseasonalized')
ax3.set_title('Seasonal Adjustment: Before vs After', fontsize=12)
ax3.set_xlabel('Date')
ax3.set_ylabel('Value')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax3.grid(True, alpha=0.3)

# Plot 4: Explanation
ax4 = axes[1, 1]
ax4.axis('off')
explanation = """
Seasonal Adjustment Process:

MULTIPLICATIVE MODEL:
  Deseasonalized = Original / Seasonal Index

  Example (July, Index = 1.20):
  Original: 150
  Deseasonalized: 150 / 1.20 = 125

  → Removes the 20% July boost

ADDITIVE MODEL:
  Deseasonalized = Original - Seasonal Index

  Example (July, Index = +20):
  Original: 150
  Deseasonalized: 150 - 20 = 130

  → Removes the +20 July effect

WHY DESEASONALIZE?
• Compare months fairly
• Identify underlying trends
• Detect unusual movements
• Economic indicators (GDP, unemployment)
"""
ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('../../charts/seasonal_adjustment.png', dpi=150, bbox_inches='tight')
plt.show()

# Print seasonal indices
print("=" * 50)
print("Seasonal Indices (Multiplicative)")
print("=" * 50)
print("\nMonth     | Index  | Interpretation")
print("-" * 50)
for month, idx in zip(months, seasonal_indices):
    if idx > 1:
        interp = f"+{(idx-1)*100:.1f}% above normal"
    else:
        interp = f"{(idx-1)*100:.1f}% below normal"
    print(f"{month:9} | {idx:.3f} | {interp}")

print("\nSum of indices:", sum(seasonal_indices))
print("(Should equal 12 for multiplicative model)")
