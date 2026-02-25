"""
TSA_ch2_ar1_simulation
======================
Simulate AR(1) processes with different phi values.

Description:
- Simulate AR(1) for various phi values
- Compare persistence and mean reversion
- Plot ACF patterns for different phi
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Set seed
np.random.seed(42)

def simulate_ar1(phi, c, sigma, n):
    """Simulate AR(1): X_t = c + phi*X_{t-1} + epsilon_t"""
    epsilon = np.random.normal(0, sigma, n)
    x = np.zeros(n)
    mu = c / (1 - phi) if abs(phi) < 1 else 0
    x[0] = mu
    for t in range(1, n):
        x[t] = c + phi * x[t-1] + epsilon[t]
    return x

# Parameters
n = 200
c = 0
sigma = 1.0
phi_values = [0.9, 0.5, -0.5, -0.9]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, phi in zip(axes.flatten(), phi_values):
    x = simulate_ar1(phi, c, sigma, n)
    ax.plot(x, 'b-', linewidth=0.8, alpha=0.8)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_title(f'AR(1) with $\\phi$ = {phi}', fontsize=11)
    ax.set_xlabel('Time')
    ax.set_ylabel('$X_t$')
    ax.grid(False)

plt.tight_layout()
plt.savefig('../../charts/ch2_ar1_simulations.pdf', bbox_inches='tight', transparent=True)
plt.show()

# Print theoretical properties
print("AR(1) Theoretical Properties:")
for phi in phi_values:
    if abs(phi) < 1:
        var = sigma**2 / (1 - phi**2)
        print(f"  phi={phi:5.1f}: Variance = {var:.4f}, ACF(1) = {phi:.2f}")
