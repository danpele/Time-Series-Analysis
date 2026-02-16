"""
TSA_ch1_stationarity
====================
Stationarity: Strict vs Weak (Covariance) Stationarity

This script demonstrates:
- Strict stationarity: Full distribution invariant to time shifts
- Weak stationarity: First two moments invariant to time shifts
- Visual examples and tests
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

n = 500

# Generate different types of series
# 1. Stationary: White noise (both strict and weak stationary)
stationary = np.random.normal(0, 1, n)

# 2. Non-stationary mean: Random walk
random_walk = np.cumsum(np.random.normal(0, 0.5, n))

# 3. Non-stationary variance: GARCH-like (variance changes over time)
garch_like = np.zeros(n)
sigma = np.ones(n)
for t in range(1, n):
    sigma[t] = 0.1 + 0.3 * garch_like[t-1]**2 + 0.6 * sigma[t-1]
    garch_like[t] = np.sqrt(sigma[t]) * np.random.normal(0, 1)

# 4. Deterministic trend (non-stationary mean)
trend = 0.02 * np.arange(n) + np.random.normal(0, 1, n)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Panel 1: Stationary series
ax1 = axes[0, 0]
ax1.plot(stationary, 'b-', linewidth=0.5, alpha=0.8, label='Series')
ax1.axhline(y=np.mean(stationary), color='red', linestyle='--', linewidth=2, label='Mean')
ax1.axhline(y=np.mean(stationary) + 2*np.std(stationary), color='orange', linestyle=':', alpha=0.7, label='+/- 2 Std')
ax1.axhline(y=np.mean(stationary) - 2*np.std(stationary), color='orange', linestyle=':', alpha=0.7)
ax1.set_title('STATIONARY: White Noise\n(Constant mean & variance)', fontsize=11, color='green')
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')

# Panel 2: Non-stationary mean (random walk)
ax2 = axes[0, 1]
ax2.plot(random_walk, 'r-', linewidth=0.8, label='Series')
# Show changing mean
window = 50
rolling_mean = pd.Series(random_walk).rolling(window).mean()
ax2.plot(rolling_mean, 'k--', linewidth=2, label='Rolling Mean')
ax2.set_title('NON-STATIONARY: Random Walk\n(Mean drifts over time)', fontsize=11, color='red')
ax2.set_xlabel('Time')
ax2.set_ylabel('Value')

# Panel 3: Non-stationary variance
ax3 = axes[1, 0]
ax3.plot(garch_like, 'purple', linewidth=0.5, alpha=0.8, label='Series')
rolling_var = pd.Series(garch_like).rolling(window).var()
ax3_twin = ax3.twinx()
ax3_twin.plot(rolling_var, 'orange', linewidth=2, label='Rolling Variance')
ax3_twin.set_ylabel('Variance', color='orange')
ax3.set_title('NON-STATIONARY: Changing Variance\n(Volatility clustering)', fontsize=11, color='red')
ax3.set_xlabel('Time')
ax3.set_ylabel('Value')

# Panel 4: Summary table
ax4 = axes[1, 1]
ax4.axis('off')

summary = """
STATIONARITY CONDITIONS

WEAK (COVARIANCE) STATIONARITY:
  1. E[X_t] = mu         (constant mean)
  2. Var(X_t) = sigma^2  (constant variance)
  3. Cov(X_t, X_{t+k}) = gamma(k)  (depends only on lag k)

STRICT STATIONARITY:
  Joint distribution of (X_{t1}, ..., X_{tk}) equals
  joint distribution of (X_{t1+h}, ..., X_{tk+h})
  for all t1,...,tk and all h

RELATIONSHIP:
  - Strict stationarity -> Weak stationarity (if moments exist)
  - Weak stationarity -/> Strict stationarity

EXAMPLES:
  - White noise: Both strict & weak stationary
  - Gaussian process with constant mean/var: Weak -> Strict
  - Heavy-tailed process: Can be strict but not weak
    (if variance doesn't exist)

WHY IT MATTERS:
  - ARMA models require (weak) stationarity
  - Non-stationary series need transformation first
  - Forecasting assumes future ~ past
"""

ax4.text(0.02, 0.98, summary, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()

# Add legend outside bottom
fig.legend(['Series', 'Mean/Reference', 'Rolling Statistics'],
           loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
plt.subplots_adjust(bottom=0.1)

# Set transparent background
fig.patch.set_facecolor('none')
for ax in fig.axes:
    ax.patch.set_facecolor('none')

# Save both PDF and PNG with transparent background
plt.savefig('../../charts/ch1_stationarity.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_stationarity.png', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_stationarity.pdf', transparent=True, bbox_inches='tight')
plt.show()

# Print diagnostic summary
print("=" * 60)
print("STATIONARITY DIAGNOSTICS")
print("=" * 60)

print("\n1. STATIONARY SERIES (White Noise):")
print(f"   Mean: {np.mean(stationary):.4f}")
print(f"   Std:  {np.std(stationary):.4f}")
print(f"   First half mean: {np.mean(stationary[:n//2]):.4f}")
print(f"   Second half mean: {np.mean(stationary[n//2:]):.4f}")

print("\n2. NON-STATIONARY (Random Walk):")
print(f"   First half mean: {np.mean(random_walk[:n//2]):.4f}")
print(f"   Second half mean: {np.mean(random_walk[n//2:]):.4f}")
print(f"   -> Means are very different!")

print("\n3. NON-STATIONARY VARIANCE:")
print(f"   First half var: {np.var(garch_like[:n//2]):.4f}")
print(f"   Second half var: {np.var(garch_like[n//2:]):.4f}")
print(f"   -> Variance changes over time!")
