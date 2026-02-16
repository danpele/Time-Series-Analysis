"""
TSA_ch1_trend_types
===================
Deterministic vs Stochastic Trends

This script demonstrates:
- Deterministic trend: X_t = a + bt + e_t
- Stochastic trend: X_t = X_{t-1} + e_t (random walk)
- How to distinguish and handle each type
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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
t = np.arange(n)

# Generate multiple realizations to show the difference
n_realizations = 5

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: Deterministic trend - multiple realizations
ax1 = axes[0, 0]
for i in range(n_realizations):
    det_trend = 10 + 0.5 * t + np.random.normal(0, 3, n)
    ax1.plot(det_trend, alpha=0.7, linewidth=1)
ax1.plot(10 + 0.5 * t, 'k--', linewidth=2, label='True Trend: 10 + 0.5t')
ax1.set_title('Deterministic Trend\n$X_t = a + bt + \\varepsilon_t$', fontsize=12)
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.grid(True, alpha=0.3)

# Panel 2: Stochastic trend - multiple realizations
ax2 = axes[0, 1]
for i in range(n_realizations):
    stoch_trend = np.cumsum(np.random.normal(0.5, 3, n))
    ax2.plot(stoch_trend, alpha=0.7, linewidth=1)
ax2.set_title('Stochastic Trend (Random Walk)\n$X_t = X_{t-1} + \\varepsilon_t$', fontsize=12)
ax2.set_xlabel('Time')
ax2.set_ylabel('Value')
ax2.grid(True, alpha=0.3)
ax2.text(0.02, 0.98, 'No fixed path!\nEach realization differs', transform=ax2.transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 3: After detrending (deterministic)
ax3 = axes[1, 0]
det_trend = 10 + 0.5 * t + np.random.normal(0, 3, n)
# Detrend by regression
slope, intercept, _, _, _ = stats.linregress(t, det_trend)
detrended = det_trend - (intercept + slope * t)
ax3.plot(detrended, 'b-', linewidth=1, alpha=0.8, label='Detrended')
ax3.axhline(y=0, color='red', linestyle='--', label='Zero line')
ax3.axhline(y=2*np.std(detrended), color='gray', linestyle=':', alpha=0.5, label='+/- 2 Std')
ax3.axhline(y=-2*np.std(detrended), color='gray', linestyle=':', alpha=0.5)
ax3.set_title('After Detrending (Regression)\n-> STATIONARY', fontsize=12, color='green')
ax3.set_xlabel('Time')
ax3.set_ylabel('Residual')
ax3.grid(True, alpha=0.3)

# Panel 4: After differencing (stochastic)
ax4 = axes[1, 1]
stoch_trend = np.cumsum(np.random.normal(0.5, 3, n))
differenced = np.diff(stoch_trend)
ax4.plot(differenced, 'r-', linewidth=1, alpha=0.8, label='Differenced')
ax4.axhline(y=np.mean(differenced), color='red', linestyle='--', label='Mean')
ax4.axhline(y=np.mean(differenced) + 2*np.std(differenced), color='gray', linestyle=':', alpha=0.5, label='+/- 2 Std')
ax4.axhline(y=np.mean(differenced) - 2*np.std(differenced), color='gray', linestyle=':', alpha=0.5)
ax4.set_title('After Differencing\n-> STATIONARY', fontsize=12, color='green')
ax4.set_xlabel('Time')
ax4.set_ylabel('First Difference')
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Add legend outside bottom
fig.legend(['Series/Realizations', 'True Trend/Mean', '+/- 2 Std'],
           loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
plt.subplots_adjust(bottom=0.1)

# Set transparent background
fig.patch.set_facecolor('none')
for ax in fig.axes:
    ax.patch.set_facecolor('none')

# Save both PDF and PNG with transparent background
plt.savefig('../../charts/ch1_trend_types.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_trend_types.png', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_trend_types.pdf', transparent=True, bbox_inches='tight')
plt.show()

# Print comparison
print("=" * 70)
print("DETERMINISTIC vs STOCHASTIC TRENDS")
print("=" * 70)
print("""
                    | Deterministic Trend    | Stochastic Trend
--------------------|------------------------|------------------------
Model               | X_t = a + bt + e_t     | X_t = X_{t-1} + e_t
Trend               | Fixed, predictable     | Random, unpredictable
Shocks              | Temporary effect       | Permanent effect
Variance            | Constant               | Grows with time
Future path         | Returns to trend       | Wanders indefinitely
Remove trend by     | REGRESSION             | DIFFERENCING
Test                | Unit root tests        | Unit root tests

How to Choose?
1. Run ADF/KPSS tests:
   - Reject unit root -> Deterministic (detrend)
   - Fail to reject -> Stochastic (difference)

2. Visual inspection:
   - Reverts to trend line -> Deterministic
   - No mean reversion -> Stochastic

3. Economic intuition:
   - Most economic series have stochastic trends
   - Physical processes often have deterministic trends

WRONG TREATMENT CONSEQUENCES:
- Detrending a random walk -> Spurious patterns
- Differencing deterministic trend -> Over-differencing, introduces MA
""")
