"""
TSA_ch2_ma1
===========
MA(1) Process: Properties and Invertibility

This script demonstrates:
- MA(1) model: X_t = ε_t + θε_{t-1}
- Always stationary (finite variance)
- Invertibility condition: |θ| < 1
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

# Set random seed
np.random.seed(42)

def simulate_ma1(n, theta, sigma=1):
    """Simulate MA(1) process"""
    eps = np.random.normal(0, sigma, n+1)
    x = np.zeros(n)
    for t in range(n):
        x[t] = eps[t+1] + theta * eps[t]
    return x

n = 500

print("=" * 60)
print("MA(1) PROCESS: X_t = ε_t + θε_{t-1}")
print("=" * 60)

print("""
Key Properties:
  Mean:     E[X_t] = 0
  Variance: γ(0) = σ²(1 + θ²)
  ACF:      ρ(1) = θ/(1+θ²), ρ(h) = 0 for h > 1

ALWAYS STATIONARY (no condition on θ for stationarity)

Invertibility: |θ| < 1
  - Needed for unique representation
  - Allows AR(∞) representation
""")

# Different theta values
theta_values = [0.8, -0.8, 0.4, 1.5]
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i, theta in enumerate(theta_values):
    # Simulate
    x = simulate_ma1(n, theta)

    # Time series plot
    axes[0, i].plot(x[:200], 'b-', linewidth=0.8, alpha=0.8)
    axes[0, i].axhline(y=0, color='red', linestyle='--', linewidth=1)
    axes[0, i].set_title(f'MA(1): θ = {theta}', fontsize=11)
    axes[0, i].set_xlabel('Time')
    axes[0, i].set_ylabel('X_t')
    axes[0, i].grid(True, alpha=0.3)

    # Invertibility check
    if abs(theta) < 1:
        axes[0, i].text(0.02, 0.98, 'Invertible', transform=axes[0, i].transAxes,
                       fontsize=9, verticalalignment='top', color='green',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    else:
        axes[0, i].text(0.02, 0.98, 'NOT Invertible!', transform=axes[0, i].transAxes,
                       fontsize=9, verticalalignment='top', color='red',
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    # ACF plot
    acf_values = acf(x, nlags=15)
    theoretical_acf = np.zeros(16)
    theoretical_acf[0] = 1
    theoretical_acf[1] = theta / (1 + theta**2)

    axes[1, i].bar(range(16), acf_values, color='blue', alpha=0.5, label='Sample ACF')
    axes[1, i].plot(range(16), theoretical_acf, 'r-o', markersize=4, linewidth=2, label='Theoretical')
    axes[1, i].axhline(y=1.96/np.sqrt(n), color='gray', linestyle='--')
    axes[1, i].axhline(y=-1.96/np.sqrt(n), color='gray', linestyle='--')
    axes[1, i].set_title(f'ACF: cuts off after lag 1', fontsize=10)
    axes[1, i].set_xlabel('Lag')
    axes[1, i].set_ylabel('ACF')
    axes[1, i].legend(fontsize=7)
    axes[1, i].grid(True, alpha=0.3)

    # PACF plot
    pacf_values = pacf(x, nlags=15)
    axes[2, i].bar(range(16), pacf_values, color='green', alpha=0.5)
    axes[2, i].axhline(y=1.96/np.sqrt(n), color='gray', linestyle='--')
    axes[2, i].axhline(y=-1.96/np.sqrt(n), color='gray', linestyle='--')
    axes[2, i].set_title(f'PACF: decays (doesn\'t cut off)', fontsize=10)
    axes[2, i].set_xlabel('Lag')
    axes[2, i].set_ylabel('PACF')
    axes[2, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../charts/ch2_ma1_properties.png', dpi=150, bbox_inches='tight')
plt.show()

# Numerical example
print("\n" + "=" * 60)
print("NUMERICAL EXAMPLE: MA(1) with θ = 0.6, σ² = 4")
print("=" * 60)

theta = 0.6
sigma_sq = 4

gamma_0 = sigma_sq * (1 + theta**2)
gamma_1 = theta * sigma_sq
rho_1 = theta / (1 + theta**2)

print(f"""
Given: θ = {theta}, σ² = {sigma_sq}

Mean:
  E[X_t] = 0

Variance:
  γ(0) = σ²(1 + θ²) = {sigma_sq} × (1 + {theta**2}) = {sigma_sq} × {1+theta**2} = {gamma_0}

Autocovariance:
  γ(1) = θσ² = {theta} × {sigma_sq} = {gamma_1}
  γ(h) = 0 for h > 1

Autocorrelation:
  ρ(1) = θ/(1+θ²) = {theta}/(1+{theta**2}) = {theta}/{1+theta**2:.2f} = {rho_1:.4f}
  ρ(h) = 0 for h > 1

Invertibility:
  |θ| = {abs(theta)} {'<' if abs(theta) < 1 else '≥'} 1
  → {'INVERTIBLE ✓' if abs(theta) < 1 else 'NOT INVERTIBLE ✗'}
""")

print("\n" + "=" * 60)
print("KEY IDENTIFICATION FEATURE")
print("=" * 60)
print("""
MA(q) Signature:
  - ACF CUTS OFF after lag q
  - PACF DECAYS (doesn't cut off)

Compare with AR(p):
  - ACF DECAYS
  - PACF CUTS OFF after lag p

This is how we identify model order from data!
""")
