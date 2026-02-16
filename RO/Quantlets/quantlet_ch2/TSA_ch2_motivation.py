"""
TSA_ch2_motivation
==================
Why ARMA Models? Motivation and Overview

This script demonstrates:
- Stationary time series patterns that motivate AR and MA models
- ACF/PACF patterns as identification tools
- Comparison of white noise, AR, MA, and ARMA processes
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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

n = 300

print("=" * 60)
print("WHY ARMA MODELS? MOTIVATION")
print("=" * 60)

print("""
After establishing stationarity (Chapter 1), we need models
that capture temporal dependence in the data.

Key patterns in stationary series:
  1. White Noise: No dependence (baseline)
  2. AR(p): Current value depends on p past values
  3. MA(q): Current value depends on q past shocks
  4. ARMA(p,q): Combination of both
""")

# Simulate different processes
eps = np.random.normal(0, 1, n)

# White Noise
wn = eps.copy()

# AR(1) with phi = 0.8
ar1 = np.zeros(n)
for t in range(1, n):
    ar1[t] = 0.8 * ar1[t-1] + eps[t]

# MA(1) with theta = 0.6
ma1 = np.zeros(n)
for t in range(1, n):
    ma1[t] = eps[t] + 0.6 * eps[t-1]

# ARMA(1,1) with phi=0.7, theta=0.4
arma11 = np.zeros(n)
for t in range(1, n):
    arma11[t] = 0.7 * arma11[t-1] + eps[t] + 0.4 * eps[t-1]

# Plot comparison
fig, axes = plt.subplots(4, 2, figsize=(14, 12))

series = [wn, ar1, ma1, arma11]
titles = ['White Noise', 'AR(1): φ = 0.8', 'MA(1): θ = 0.6', 'ARMA(1,1): φ=0.7, θ=0.4']

for i, (s, title) in enumerate(zip(series, titles)):
    axes[i, 0].plot(s, color='steelblue', linewidth=0.8)
    axes[i, 0].set_title(f'{title} — Time Series', fontsize=10)
    axes[i, 0].set_xlabel('t')

    plot_acf(s, ax=axes[i, 1], lags=20, alpha=0.05)
    axes[i, 1].set_title(f'{title} — ACF', fontsize=10)

plt.tight_layout()
plt.savefig('../../charts/ch2_motivation_stationary.pdf', bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch2_motivation_stationary.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch2_motivation_stationary.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("Observation:")
print("  - White Noise: ACF = 0 for all lags")
print("  - AR(1): ACF decays exponentially")
print("  - MA(1): ACF cuts off after lag 1")
print("  - ARMA(1,1): ACF decays (mixed pattern)")
print("\nThese distinct patterns allow us to IDENTIFY the model type!")
