"""
TSA_ch2_ma1
===========
MA(1) Process Properties

This script demonstrates:
- MA(1) model: X_t = e_t + θe_{t-1}
- ACF cuts off after lag 1
- PACF shows exponential decay
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

# Set random seed
np.random.seed(42)

n = 500

def generate_ma1(n, theta, sigma=1):
    """Generate MA(1) process: X_t = e_t + theta * e_{t-1}"""
    eps = np.random.normal(0, sigma, n+1)
    x = np.zeros(n)
    for t in range(n):
        x[t] = eps[t+1] + theta * eps[t]
    return x

# Generate MA(1) with different theta values
thetas = [0.8, -0.8, 0.5, -0.5]
colors = ['blue', 'red', 'green', 'purple']

# Create figure
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

for i, (theta, color) in enumerate(zip(thetas, colors)):
    ma1 = generate_ma1(n, theta)

    # Time series plot
    axes[0, i].plot(ma1[:100], color=color, linewidth=0.8)
    axes[0, i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, i].set_title(f'MA(1): θ = {theta}', fontsize=11)
    axes[0, i].set_xlabel('Time')
    axes[0, i].set_ylabel('Value')
    axes[0, i].grid(True, alpha=0.3)

    # ACF plot
    acf_values = acf(ma1, nlags=15)
    axes[1, i].bar(range(16), acf_values, color=color, alpha=0.7, edgecolor='black')
    axes[1, i].axhline(y=1.96/np.sqrt(n), color='red', linestyle='--')
    axes[1, i].axhline(y=-1.96/np.sqrt(n), color='red', linestyle='--')
    axes[1, i].axhline(y=0, color='black', linewidth=0.5)
    axes[1, i].set_title(f'ACF: Cuts off after lag 1', fontsize=10)
    axes[1, i].set_xlabel('Lag')
    axes[1, i].set_ylabel('ACF')

    # Theoretical ACF(1)
    theoretical_acf1 = theta / (1 + theta**2)
    axes[1, i].axhline(y=theoretical_acf1, color='orange', linestyle=':', linewidth=2,
                       label=f'ρ(1) = {theoretical_acf1:.3f}')
    axes[1, i].legend(fontsize=8)

    # PACF plot
    pacf_values = pacf(ma1, nlags=15)
    axes[2, i].bar(range(16), pacf_values, color=color, alpha=0.7, edgecolor='black')
    axes[2, i].axhline(y=1.96/np.sqrt(n), color='red', linestyle='--')
    axes[2, i].axhline(y=-1.96/np.sqrt(n), color='red', linestyle='--')
    axes[2, i].axhline(y=0, color='black', linewidth=0.5)
    axes[2, i].set_title(f'PACF: Gradual decay', fontsize=10)
    axes[2, i].set_xlabel('Lag')
    axes[2, i].set_ylabel('PACF')

plt.tight_layout()
plt.savefig('../../charts/ch2_ma1_acf_pacf.png', dpi=150, bbox_inches='tight')
plt.show()

# Print MA(1) properties
print("=" * 70)
print("MA(1) PROCESS PROPERTIES")
print("=" * 70)
print("""
Model: X_t = e_t + θe_{t-1}  where e_t ~ WN(0, σ²)

PROPERTIES:
  E[X_t] = 0  (always stationary!)
  Var(X_t) = σ²(1 + θ²)

AUTOCORRELATION:
  ρ(1) = θ / (1 + θ²)
  ρ(k) = 0  for k > 1

  → ACF CUTS OFF after lag 1!

PARTIAL AUTOCORRELATION:
  φ_kk shows exponential/oscillating decay

  → PACF DECAYS gradually

IDENTIFICATION:
  If ACF cuts off after lag q → MA(q) process

INVERTIBILITY:
  MA(1) is invertible if |θ| < 1
  Invertibility needed for unique representation
""")

# Show theoretical vs empirical
print("\nTHEORETICAL vs EMPIRICAL ACF(1):")
print("-" * 40)
for theta in thetas:
    ma1 = generate_ma1(n, theta)
    theoretical = theta / (1 + theta**2)
    empirical = acf(ma1, nlags=1)[1]
    print(f"θ = {theta:5.1f}: Theoretical = {theoretical:7.4f}, Empirical = {empirical:7.4f}")
