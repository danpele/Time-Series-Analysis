"""
TSA_ch2_python_simulate
=======================
Python Exercise 1: Simulate and Fit AR(1)

Tasks:
1. Simulate AR(1) with φ = 0.6, n = 300
2. Fit AR(1) model and compare estimated vs true parameters
3. Plot ACF and compare with theoretical
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("PYTHON EXERCISE: Simulate and Fit AR(1)")
print("=" * 60)

# Task 1: Simulate AR(1)
print("\n" + "-" * 60)
print("Task 1: Simulate AR(1) Process")
print("-" * 60)

# Parameters
phi_true = 0.6
c_true = 2.0  # constant term
sigma = 1.0
n = 300

print(f"\nTrue parameters:")
print(f"  φ = {phi_true}")
print(f"  c = {c_true}")
print(f"  σ = {sigma}")
print(f"  n = {n}")

# Calculate theoretical mean
mu_true = c_true / (1 - phi_true)
print(f"\nTheoretical mean: μ = c/(1-φ) = {c_true}/(1-{phi_true}) = {mu_true}")

# Simulate
eps = np.random.normal(0, sigma, n)
x = np.zeros(n)
x[0] = mu_true  # Start at mean

for t in range(1, n):
    x[t] = c_true + phi_true * x[t-1] + eps[t]

print(f"\nSimulated series:")
print(f"  Sample mean: {np.mean(x):.4f}")
print(f"  Sample std:  {np.std(x):.4f}")

# Task 2: Fit AR(1) model
print("\n" + "-" * 60)
print("Task 2: Fit AR(1) Model")
print("-" * 60)

model = ARIMA(x, order=(1, 0, 0)).fit()

print("\nEstimated parameters:")
print(f"  φ̂ = {model.params[1]:.4f}  (true: {phi_true})")
print(f"  ĉ = {model.params[0]:.4f}  (true: {c_true})")
print(f"  σ̂ = {np.sqrt(model.params[2]):.4f}  (true: {sigma})")

# Estimated mean
mu_hat = model.params[0] / (1 - model.params[1])
print(f"\nEstimated mean: μ̂ = {mu_hat:.4f}  (true: {mu_true})")

print("\nModel summary:")
print(model.summary().tables[1])

# Task 3: Compare ACF
print("\n" + "-" * 60)
print("Task 3: Compare Theoretical vs Sample ACF")
print("-" * 60)

lags = np.arange(0, 16)
acf_theoretical = phi_true ** lags
acf_sample = acf(x, nlags=15)

print("\nACF Comparison:")
print(f"{'Lag':<6} {'Theoretical':<15} {'Sample':<15} {'Difference':<15}")
print("-" * 50)
for h in range(6):
    diff = acf_sample[h] - acf_theoretical[h]
    print(f"{h:<6} {acf_theoretical[h]:<15.4f} {acf_sample[h]:<15.4f} {diff:<15.4f}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Simulated series
ax1 = axes[0, 0]
ax1.plot(x, 'b-', linewidth=0.8, alpha=0.8)
ax1.axhline(y=mu_true, color='red', linestyle='--', linewidth=2, label=f'True μ = {mu_true}')
ax1.axhline(y=np.mean(x), color='green', linestyle=':', linewidth=2, label=f'Sample mean = {np.mean(x):.2f}')
ax1.set_title(f'Simulated AR(1): X_t = {c_true} + {phi_true}X_{{t-1}} + ε_t', fontsize=12)
ax1.set_xlabel('Time')
ax1.set_ylabel('X_t')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: ACF comparison
ax2 = axes[0, 1]
width = 0.35
ax2.bar(lags - width/2, acf_theoretical, width, label='Theoretical', color='blue', alpha=0.7)
ax2.bar(lags + width/2, acf_sample, width, label='Sample', color='red', alpha=0.7)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.axhline(y=1.96/np.sqrt(n), color='gray', linestyle='--', alpha=0.5, label='95% bounds')
ax2.axhline(y=-1.96/np.sqrt(n), color='gray', linestyle='--', alpha=0.5)
ax2.set_title('ACF: Theoretical vs Sample', fontsize=12)
ax2.set_xlabel('Lag')
ax2.set_ylabel('ACF')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: PACF
ax3 = axes[1, 0]
plot_pacf(x, lags=15, ax=ax3, method='ywm')
ax3.axhline(y=phi_true, color='red', linestyle='--', linewidth=2, label=f'True φ = {phi_true}')
ax3.set_title('Sample PACF (should cut off after lag 1)', fontsize=12)
ax3.legend()

# Plot 4: Parameter comparison
ax4 = axes[1, 1]
ax4.axis('off')
summary = f"""
Parameter Estimation Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

True Model: X_t = {c_true} + {phi_true}X_{{t-1}} + ε_t
Sample size: n = {n}

Parameter Comparison:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Parameter    True      Estimated   Diff
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
φ            {phi_true:<10.4f}{model.params[1]:<12.4f}{model.params[1]-phi_true:+.4f}
c            {c_true:<10.4f}{model.params[0]:<12.4f}{model.params[0]-c_true:+.4f}
σ            {sigma:<10.4f}{np.sqrt(model.params[2]):<12.4f}{np.sqrt(model.params[2])-sigma:+.4f}
μ            {mu_true:<10.4f}{mu_hat:<12.4f}{mu_hat-mu_true:+.4f}

Model Fit:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AIC: {model.aic:.2f}
BIC: {model.bic:.2f}
Log-likelihood: {model.llf:.2f}

Conclusion:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Estimates are close to true values.
Larger n → more precise estimates.
"""
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('../../charts/ch2_python_simulate.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("EXERCISE COMPLETE")
print("=" * 60)
print(f"\nKey findings:")
print(f"1. Estimated φ̂ = {model.params[1]:.4f} (true: {phi_true})")
print(f"2. Sample ACF decays geometrically as expected")
print(f"3. Sample PACF cuts off after lag 1 (AR(1) signature)")
