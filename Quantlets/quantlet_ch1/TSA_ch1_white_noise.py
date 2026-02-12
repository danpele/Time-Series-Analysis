"""
TSA_ch1_white_noise
===================
White Noise Properties and Testing

This script demonstrates:
- Properties of white noise: E[X_t]=0, Var(X_t)=sigma^2, Cov(X_t,X_s)=0
- Generating and identifying white noise
- Ljung-Box test for white noise
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

# Set random seed
np.random.seed(42)

n = 500

# Generate white noise
white_noise = np.random.normal(0, 1, n)

# Generate non-white noise for comparison (AR(1) process)
ar1 = np.zeros(n)
phi = 0.8
for t in range(1, n):
    ar1[t] = phi * ar1[t-1] + np.random.normal(0, 1)

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 11))

# === WHITE NOISE (TOP ROW) ===
ax1 = axes[0, 0]
ax1.plot(white_noise, 'b-', linewidth=0.5, alpha=0.8, label='White Noise')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Mean = 0')
ax1.axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='+/- 2 Std')
ax1.axhline(y=-2, color='gray', linestyle=':', alpha=0.5)
ax1.set_title('White Noise: Time Series', fontsize=11)
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.fill_between(range(n), -2, 2, alpha=0.1, color='blue')

# ACF of white noise
ax2 = axes[0, 1]
lags = np.arange(1, 21)
acf_wn = [np.corrcoef(white_noise[:-lag], white_noise[lag:])[0, 1] for lag in lags]
ci = 1.96 / np.sqrt(n)
ax2.bar(lags, acf_wn, color='blue', alpha=0.7, edgecolor='black', label='ACF')
ax2.axhline(y=ci, color='red', linestyle='--', label=f'95% CI (+/-{ci:.3f})')
ax2.axhline(y=-ci, color='red', linestyle='--')
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_title('White Noise: ACF', fontsize=11)
ax2.set_xlabel('Lag')
ax2.set_ylabel('Autocorrelation')

# Distribution
ax3 = axes[0, 2]
ax3.hist(white_noise, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black', label='Histogram')
x = np.linspace(-4, 4, 100)
ax3.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')
ax3.set_title('White Noise: Distribution', fontsize=11)
ax3.set_xlabel('Value')
ax3.set_ylabel('Density')

# === NOT WHITE NOISE - AR(1) (BOTTOM ROW) ===
ax4 = axes[1, 0]
ax4.plot(ar1, 'r-', linewidth=0.5, alpha=0.8, label='AR(1)')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, label='Zero')
ax4.set_title(f'AR(1) Process (phi={phi}): NOT White Noise', fontsize=11)
ax4.set_xlabel('Time')
ax4.set_ylabel('Value')

# ACF of AR(1)
ax5 = axes[1, 1]
acf_ar1 = [np.corrcoef(ar1[:-lag], ar1[lag:])[0, 1] for lag in lags]
ax5.bar(lags, acf_ar1, color='red', alpha=0.7, edgecolor='black', label='ACF')
ax5.axhline(y=ci, color='gray', linestyle='--', label='95% CI')
ax5.axhline(y=-ci, color='gray', linestyle='--')
ax5.axhline(y=0, color='black', linewidth=0.5)
ax5.set_title('AR(1): ACF (Significant correlations!)', fontsize=11)
ax5.set_xlabel('Lag')
ax5.set_ylabel('Autocorrelation')

# Ljung-Box test results
ax6 = axes[1, 2]
ax6.axis('off')

# Perform Ljung-Box test
lb_wn = acorr_ljungbox(white_noise, lags=[10, 20], return_df=True)
lb_ar1 = acorr_ljungbox(ar1, lags=[10, 20], return_df=True)

test_results = f"""
Ljung-Box Test for White Noise
H0: Data is white noise (no autocorrelation)

WHITE NOISE SERIES:
  Lag 10: Q = {lb_wn.loc[10, 'lb_stat']:.2f}, p-value = {lb_wn.loc[10, 'lb_pvalue']:.4f}
  Lag 20: Q = {lb_wn.loc[20, 'lb_stat']:.2f}, p-value = {lb_wn.loc[20, 'lb_pvalue']:.4f}
  -> p > 0.05: FAIL TO REJECT H0
  -> Consistent with white noise

AR(1) SERIES:
  Lag 10: Q = {lb_ar1.loc[10, 'lb_stat']:.2f}, p-value = {lb_ar1.loc[10, 'lb_pvalue']:.6f}
  Lag 20: Q = {lb_ar1.loc[20, 'lb_stat']:.2f}, p-value = {lb_ar1.loc[20, 'lb_pvalue']:.6f}
  -> p < 0.05: REJECT H0
  -> NOT white noise (autocorrelation present)

White Noise Properties:
  1. E[X_t] = mu (constant mean)
  2. Var(X_t) = sigma^2 (constant variance)
  3. Cov(X_t, X_s) = 0 for t != s
"""

ax6.text(0.05, 0.95, test_results, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()

# Add legend outside bottom
fig.legend(['White Noise', 'AR(1)', '95% CI', 'N(0,1) Distribution'],
           loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
plt.subplots_adjust(bottom=0.08)

# Set transparent background
fig.patch.set_facecolor('none')
for ax in fig.axes:
    ax.patch.set_facecolor('none')

# Save both PDF and PNG with transparent background
plt.savefig('../../charts/ch1_white_noise_test.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_white_noise_test.png', transparent=True, bbox_inches='tight', dpi=300)
plt.show()

# Print results
print("=" * 60)
print("WHITE NOISE TESTING SUMMARY")
print("=" * 60)
print(f"\nWhite noise series mean: {np.mean(white_noise):.4f} (should be ~ 0)")
print(f"White noise series std:  {np.std(white_noise):.4f} (should be ~ 1)")
print(f"\nLjung-Box test (H0: white noise):")
print(f"  White noise p-value: {lb_wn.loc[20, 'lb_pvalue']:.4f} -> Fail to reject")
print(f"  AR(1) p-value:       {lb_ar1.loc[20, 'lb_pvalue']:.6f} -> Reject")
