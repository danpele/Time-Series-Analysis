"""
TSA_ch2_ar1
===========
AR(1) Process: Properties and Simulation

This script demonstrates:
- AR(1) model: X_t = c + φX_{t-1} + ε_t
- Stationarity condition: |φ| < 1
- Mean, variance, and ACF
"""

import numpy as np
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

def simulate_ar1(n, phi, c=0, sigma=1):
    """Simulate AR(1) process"""
    x = np.zeros(n)
    eps = np.random.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = c + phi * x[t-1] + eps[t]
    return x

n = 500

print("=" * 60)
print("AR(1) PROCESS: X_t = c + φX_{t-1} + ε_t")
print("=" * 60)

print("""
Stationarity Condition: |φ| < 1

If stationary:
  Mean:     μ = c / (1 - φ)
  Variance: γ(0) = σ² / (1 - φ²)
  ACF:      ρ(h) = φ^h  (exponential decay)
""")

# Different phi values
phi_values = [0.9, 0.5, -0.7, 0.99]
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, phi in enumerate(phi_values):
    # Simulate
    x = simulate_ar1(n, phi, c=0, sigma=1)

    # Time series plot
    axes[0, i].plot(x, 'b-', linewidth=0.5, alpha=0.8)
    axes[0, i].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, i].set_title(f'AR(1): φ = {phi}', fontsize=11)
    axes[0, i].set_xlabel('Time')
    axes[0, i].set_ylabel('X_t')
    axes[0, i].grid(True, alpha=0.3)

    # Theoretical variance
    if abs(phi) < 1:
        var_theory = 1 / (1 - phi**2)
        axes[0, i].axhline(y=2*np.sqrt(var_theory), color='gray', linestyle=':', alpha=0.5)
        axes[0, i].axhline(y=-2*np.sqrt(var_theory), color='gray', linestyle=':', alpha=0.5)

    # ACF plot
    acf_values = acf(x, nlags=20)
    theoretical_acf = phi ** np.arange(21)

    axes[1, i].bar(range(21), acf_values, color='blue', alpha=0.5, label='Sample ACF')
    axes[1, i].plot(range(21), theoretical_acf, 'r-o', markersize=4, linewidth=2, label='Theoretical')
    axes[1, i].axhline(y=1.96/np.sqrt(n), color='gray', linestyle='--')
    axes[1, i].axhline(y=-1.96/np.sqrt(n), color='gray', linestyle='--')
    axes[1, i].set_title(f'ACF: ρ(h) = φ^h = {phi}^h', fontsize=10)
    axes[1, i].set_xlabel('Lag')
    axes[1, i].set_ylabel('ACF')
    axes[1, i].legend(fontsize=8)
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../charts/ch2_ar1_properties.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch2_ar1_properties.pdf', bbox_inches='tight', transparent=True)
plt.show()

# Detailed calculations for one example
print("\n" + "=" * 60)
print("NUMERICAL EXAMPLE: AR(1) with φ = 0.7, c = 2, σ² = 9")
print("=" * 60)

phi = 0.7
c = 2
sigma_sq = 9

mu = c / (1 - phi)
gamma_0 = sigma_sq / (1 - phi**2)
gamma_1 = phi * gamma_0
rho_1 = phi

print(f"""
Given: c = {c}, φ = {phi}, σ² = {sigma_sq}

Mean:
  μ = c / (1-φ) = {c} / (1-{phi}) = {c} / {1-phi} = {mu:.2f}

Variance:
  γ(0) = σ² / (1-φ²) = {sigma_sq} / (1-{phi**2}) = {sigma_sq} / {1-phi**2:.2f} = {gamma_0:.2f}

Autocovariance at lag 1:
  γ(1) = φ × γ(0) = {phi} × {gamma_0:.2f} = {gamma_1:.2f}

Autocorrelation:
  ρ(1) = φ = {rho_1}
  ρ(2) = φ² = {phi**2}
  ρ(3) = φ³ = {phi**3:.3f}
""")

print("\n" + "=" * 60)
print("STATIONARITY REGIONS")
print("=" * 60)
print("""
φ = 1:    Unit root (random walk) - NON-STATIONARY
|φ| < 1:  Stationary (mean-reverting)
|φ| > 1:  Explosive - NON-STATIONARY

φ > 0:    Positive autocorrelation, smooth patterns
φ < 0:    Negative autocorrelation, oscillating patterns
φ ≈ 1:    Highly persistent, slow mean reversion
φ ≈ 0:    Weak dependence, close to white noise
""")
