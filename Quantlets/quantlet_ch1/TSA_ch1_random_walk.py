"""
TSA_ch1_random_walk
===================
Random Walk vs White Noise

This script demonstrates:
- White noise: X_t = e_t (stationary)
- Random walk: X_t = X_{t-1} + e_t (non-stationary)
- Key differences in behavior and properties
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Set random seed
np.random.seed(42)

n = 500
n_simulations = 5

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 11))

# === TOP ROW: WHITE NOISE ===

# Panel 1: Multiple white noise realizations
ax1 = axes[0, 0]
for i in range(n_simulations):
    wn = np.random.normal(0, 1, n)
    ax1.plot(wn, alpha=0.7, linewidth=0.5)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Mean = 0')
ax1.set_title('White Noise: Multiple Realizations', fontsize=11)
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.set_ylim(-4, 4)
ax1.text(0.02, 0.98, 'E[X_t] = 0\nVar(X_t) = sigma^2', transform=ax1.transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 2: White noise ACF
ax2 = axes[0, 1]
wn = np.random.normal(0, 1, n)
lags = np.arange(1, 31)
acf_values = [np.corrcoef(wn[:-lag], wn[lag:])[0, 1] for lag in lags]
ax2.bar(lags, acf_values, color='blue', alpha=0.7, label='ACF')
ax2.axhline(y=1.96/np.sqrt(n), color='red', linestyle='--', label='95% CI')
ax2.axhline(y=-1.96/np.sqrt(n), color='red', linestyle='--')
ax2.set_title('White Noise: ACF', fontsize=11)
ax2.set_xlabel('Lag')
ax2.set_ylabel('Autocorrelation')
ax2.text(0.5, 0.95, 'No significant correlations', transform=ax2.transAxes,
         ha='center', fontsize=10, color='green')

# Panel 3: White noise histogram
ax3 = axes[0, 2]
ax3.hist(wn, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black', label='Histogram')
x = np.linspace(-4, 4, 100)
ax3.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')
ax3.set_title('White Noise: Distribution', fontsize=11)
ax3.set_xlabel('Value')
ax3.set_ylabel('Density')

# === BOTTOM ROW: RANDOM WALK ===

# Panel 4: Multiple random walk realizations
ax4 = axes[1, 0]
for i in range(n_simulations):
    rw = np.cumsum(np.random.normal(0, 1, n))
    ax4.plot(rw, alpha=0.7, linewidth=0.8)
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Initial value')
ax4.set_title('Random Walk: Multiple Realizations', fontsize=11)
ax4.set_xlabel('Time')
ax4.set_ylabel('Value')
ax4.text(0.02, 0.98, 'E[X_t] = X_0\nVar(X_t) = t*sigma^2', transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Panel 5: Random walk ACF
ax5 = axes[1, 1]
rw = np.cumsum(np.random.normal(0, 1, n))
acf_rw = [np.corrcoef(rw[:-lag], rw[lag:])[0, 1] for lag in lags]
ax5.bar(lags, acf_rw, color='red', alpha=0.7, label='ACF')
ax5.axhline(y=1.96/np.sqrt(n), color='gray', linestyle='--', label='95% CI')
ax5.axhline(y=-1.96/np.sqrt(n), color='gray', linestyle='--')
ax5.set_title('Random Walk: ACF', fontsize=11)
ax5.set_xlabel('Lag')
ax5.set_ylabel('Autocorrelation')
ax5.text(0.5, 0.95, 'Slow decay - Non-stationary!', transform=ax5.transAxes,
         ha='center', fontsize=10, color='red')

# Panel 6: Variance over time comparison
ax6 = axes[1, 2]
# Calculate rolling variance
window = 50
wn_var = pd.Series(np.random.normal(0, 1, n)).rolling(window).var()
rw_var = pd.Series(np.cumsum(np.random.normal(0, 1, n))).rolling(window).var()
ax6.plot(wn_var, 'b-', linewidth=1.5, label='White Noise Variance', alpha=0.7)
ax6.plot(rw_var, 'r-', linewidth=1.5, label='Random Walk Variance', alpha=0.7)
ax6.set_title(f'Rolling Variance (window={window})', fontsize=11)
ax6.set_xlabel('Time')
ax6.set_ylabel('Variance')
ax6.set_yscale('log')

plt.tight_layout()

# Add legend outside bottom
fig.legend(['White Noise', 'Random Walk', '95% CI', 'Reference Line'],
           loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
plt.subplots_adjust(bottom=0.1)

# Set transparent background
fig.patch.set_facecolor('none')
for ax in fig.axes:
    ax.patch.set_facecolor('none')

# Save both PDF and PNG with transparent background
plt.savefig('../../charts/ch1_wn_rw.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_wn_rw.png', transparent=True, bbox_inches='tight', dpi=300)
plt.show()

# Print comparison
print("=" * 70)
print("WHITE NOISE vs RANDOM WALK COMPARISON")
print("=" * 70)
print("""
Property          | White Noise           | Random Walk
------------------|----------------------|------------------------
Definition        | X_t = e_t            | X_t = X_{t-1} + e_t
Mean              | Constant (0)         | Depends on X_0
Variance          | Constant (sigma^2)   | Grows with time (t*sigma^2)
ACF               | Zero at all lags     | Slow decay
Stationarity      | STATIONARY           | NON-STATIONARY
Predictability    | Unpredictable        | Best guess = current value
Shocks            | Temporary effect     | Permanent effect

Key Insight:
- White noise: Each observation is independent, returns to mean
- Random walk: Shocks accumulate, no mean reversion
""")
