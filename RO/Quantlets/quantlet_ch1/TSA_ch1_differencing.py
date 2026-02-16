"""
TSA_ch1_differencing
====================
Differencing for Stationarity

This script demonstrates:
- First differencing: DX_t = X_t - X_{t-1}
- Seasonal differencing: D_s X_t = X_t - X_{t-s}
- Over-differencing problems
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

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

n = 200

# Generate non-stationary series
# 1. Random walk
random_walk = np.cumsum(np.random.normal(0, 1, n))

# 2. Random walk with seasonal component
t = np.arange(n)
seasonal_rw = np.cumsum(np.random.normal(0.1, 0.5, n)) + 15 * np.sin(2 * np.pi * t / 12)

# Create figure
fig, axes = plt.subplots(3, 2, figsize=(14, 13))

# === LEFT COLUMN: REGULAR DIFFERENCING ===

# Original random walk
ax1 = axes[0, 0]
ax1.plot(random_walk, 'b-', linewidth=1, label='Random Walk')
ax1.set_title('Original: Random Walk (Non-Stationary)', fontsize=11)
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.grid(True, alpha=0.3)
adf_orig = adfuller(random_walk)[1]
ax1.text(0.02, 0.98, f'ADF p-value: {adf_orig:.4f}\n(Non-stationary)',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# First difference
diff1 = np.diff(random_walk)
ax2 = axes[1, 0]
ax2.plot(diff1, 'g-', linewidth=0.8, label='First Difference')
ax2.axhline(y=0, color='red', linestyle='--', label='Zero line')
ax2.set_title('First Difference: $\\Delta X_t = X_t - X_{t-1}$ (Stationary)', fontsize=11)
ax2.set_xlabel('Time')
ax2.set_ylabel('DX')
ax2.grid(True, alpha=0.3)
adf_diff1 = adfuller(diff1)[1]
ax2.text(0.02, 0.98, f'ADF p-value: {adf_diff1:.4f}\n(Stationary!)',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Second difference (over-differencing)
diff2 = np.diff(diff1)
ax3 = axes[2, 0]
ax3.plot(diff2, 'r-', linewidth=0.8, label='Second Difference')
ax3.axhline(y=0, color='black', linestyle='--', label='Zero line')
ax3.set_title('Second Difference: $\\Delta^2 X_t$ (OVER-DIFFERENCED!)', fontsize=11, color='red')
ax3.set_xlabel('Time')
ax3.set_ylabel('D2X')
ax3.grid(True, alpha=0.3)
ax3.text(0.02, 0.98, 'Warning: Over-differencing\nintroduces MA component!',
         transform=ax3.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# === RIGHT COLUMN: SEASONAL DIFFERENCING ===

# Original seasonal series
ax4 = axes[0, 1]
ax4.plot(seasonal_rw, 'b-', linewidth=1, label='Trend + Seasonality')
ax4.set_title('Original: Trend + Seasonality', fontsize=11)
ax4.set_xlabel('Time')
ax4.set_ylabel('Value')
ax4.grid(True, alpha=0.3)

# Seasonal difference (lag 12)
seasonal_diff = seasonal_rw[12:] - seasonal_rw[:-12]
ax5 = axes[1, 1]
ax5.plot(seasonal_diff, 'orange', linewidth=0.8, label='Seasonal Difference')
ax5.axhline(y=np.mean(seasonal_diff), color='red', linestyle='--', label='Mean')
ax5.set_title('Seasonal Difference: $\\Delta_{12} X_t = X_t - X_{t-12}$', fontsize=11)
ax5.set_xlabel('Time')
ax5.set_ylabel('D12X')
ax5.grid(True, alpha=0.3)
ax5.text(0.02, 0.98, 'Removes seasonality\nbut trend remains',
         transform=ax5.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

# Both differences
both_diff = np.diff(seasonal_diff)
ax6 = axes[2, 1]
ax6.plot(both_diff, 'purple', linewidth=0.8, label='Both Differences')
ax6.axhline(y=0, color='red', linestyle='--', label='Zero line')
ax6.set_title('Both: $\\Delta\\Delta_{12} X_t$ (Stationary)', fontsize=11)
ax6.set_xlabel('Time')
ax6.set_ylabel('DD12X')
ax6.grid(True, alpha=0.3)
adf_both = adfuller(both_diff)[1]
ax6.text(0.02, 0.98, f'ADF p-value: {adf_both:.4f}\n(Stationary!)',
         transform=ax6.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()

# Add legend outside bottom
fig.legend(['Series', 'Reference Line'], loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
plt.subplots_adjust(bottom=0.08)

# Set transparent background
fig.patch.set_facecolor('none')
for ax in fig.axes:
    ax.patch.set_facecolor('none')

# Save both PDF and PNG with transparent background
plt.savefig('../../charts/ch1_differencing.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_differencing.png', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_differencing.pdf', transparent=True, bbox_inches='tight')
plt.show()

# Print differencing guide
print("=" * 70)
print("DIFFERENCING GUIDE")
print("=" * 70)
print("""
Type                | Formula                   | Removes
--------------------|---------------------------|-------------------
First Difference    | DX_t = X_t - X_{t-1}      | Linear trend
Second Difference   | D2X_t = D(DX_t)           | Quadratic trend
Seasonal (lag s)    | D_s X_t = X_t - X_{t-s}   | Seasonality
Both                | DD_s X_t                  | Trend + Seasonality

ARIMA NOTATION:
- ARIMA(p,d,q): d = number of regular differences
- SARIMA(p,d,q)(P,D,Q)_s: D = number of seasonal differences

RULES OF THUMB:
1. Most economic series need d=1 (one difference)
2. Rarely need d=2 (check for over-differencing)
3. Seasonal data often needs D=1

WARNING SIGNS OF OVER-DIFFERENCING:
- ACF shows large negative spike at lag 1
- Variance increases after differencing
- Series looks "too erratic"

PROCEDURE:
1. Test original series (ADF/KPSS)
2. If non-stationary, take first difference
3. Test again
4. If seasonal pattern remains, take seasonal difference
5. Never difference more than necessary!
""")
