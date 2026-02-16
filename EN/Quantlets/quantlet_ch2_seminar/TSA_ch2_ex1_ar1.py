"""
TSA_ch2_ex1_ar1
===============
Seminar Exercise 1: AR(1) Properties

Problem: X_t = 2 + 0.7X_{t-1} + ε_t, ε_t ~ WN(0, 9)
Calculate mean, variance, autocovariance, and autocorrelation
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Given parameters
c = 2           # constant
phi = 0.7       # AR coefficient
sigma_sq = 9    # variance of white noise

print("=" * 60)
print("AR(1) PROPERTIES - Step by Step Solution")
print("=" * 60)
print(f"\nModel: X_t = {c} + {phi}X_{{t-1}} + ε_t")
print(f"where ε_t ~ WN(0, {sigma_sq})")

# 1. Mean
print("\n" + "-" * 60)
print("1. Calculate the Mean μ")
print("-" * 60)
print("\nFor AR(1): X_t = c + φX_{t-1} + ε_t")
print("Taking expectations: E[X_t] = c + φE[X_{t-1}] + E[ε_t]")
print("Since stationary: μ = c + φμ + 0")
print("Solving: μ(1 - φ) = c")
print("        μ = c / (1 - φ)")
print()
print(f"μ = {c} / (1 - {phi})")
print(f"μ = {c} / {1 - phi}")

mu = c / (1 - phi)
print(f"μ = {mu:.4f}")

# 2. Variance
print("\n" + "-" * 60)
print("2. Calculate the Variance γ(0)")
print("-" * 60)
print("\nFor stationary AR(1):")
print("  γ(0) = Var(X_t) = σ² / (1 - φ²)")
print()
print(f"γ(0) = {sigma_sq} / (1 - {phi}²)")
print(f"γ(0) = {sigma_sq} / (1 - {phi**2})")
print(f"γ(0) = {sigma_sq} / {1 - phi**2:.4f}")

gamma_0 = sigma_sq / (1 - phi**2)
print(f"γ(0) = {gamma_0:.4f}")

# 3. Autocovariance
print("\n" + "-" * 60)
print("3. Calculate Autocovariance γ(1) and γ(2)")
print("-" * 60)
print("\nFor AR(1): γ(h) = φ^h × γ(0)")
print()
print("γ(1) = φ × γ(0)")
print(f"γ(1) = {phi} × {gamma_0:.4f}")

gamma_1 = phi * gamma_0
print(f"γ(1) = {gamma_1:.4f}")

print()
print("γ(2) = φ² × γ(0)")
print(f"γ(2) = {phi}² × {gamma_0:.4f}")
print(f"γ(2) = {phi**2} × {gamma_0:.4f}")

gamma_2 = phi**2 * gamma_0
print(f"γ(2) = {gamma_2:.4f}")

# 4. Autocorrelation
print("\n" + "-" * 60)
print("4. Calculate Autocorrelation ρ(1) and ρ(2)")
print("-" * 60)
print("\nFor AR(1): ρ(h) = φ^h (simple geometric decay)")
print()
print(f"ρ(1) = φ¹ = {phi}^1 = {phi}")
print(f"ρ(2) = φ² = {phi}^2 = {phi**2}")

rho_1 = phi
rho_2 = phi**2

print()
print("Verification: ρ(h) = γ(h) / γ(0)")
print(f"  ρ(1) = {gamma_1:.4f} / {gamma_0:.4f} = {gamma_1/gamma_0:.4f} ✓")
print(f"  ρ(2) = {gamma_2:.4f} / {gamma_0:.4f} = {gamma_2/gamma_0:.4f} ✓")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Simulated AR(1)
np.random.seed(42)
n = 200
x = np.zeros(n)
eps = np.random.normal(0, np.sqrt(sigma_sq), n)
x[0] = mu
for t in range(1, n):
    x[t] = c + phi * x[t-1] + eps[t]

ax1 = axes[0]
ax1.plot(x, 'b-', linewidth=0.8, alpha=0.8)
ax1.axhline(y=mu, color='red', linestyle='--', linewidth=2, label=f'Mean μ = {mu:.2f}')
ax1.axhline(y=mu + 2*np.sqrt(gamma_0), color='gray', linestyle=':', alpha=0.7)
ax1.axhline(y=mu - 2*np.sqrt(gamma_0), color='gray', linestyle=':', alpha=0.7, label=f'±2σ')
ax1.set_title(f'Simulated AR(1): X_t = {c} + {phi}X_{{t-1}} + ε_t', fontsize=11)
ax1.set_xlabel('Time')
ax1.set_ylabel('X_t')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax1.grid(True, alpha=0.3)

# ACF
ax2 = axes[1]
lags = np.arange(0, 11)
acf_theoretical = phi ** lags
ax2.bar(lags, acf_theoretical, color='blue', alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_title(f'Theoretical ACF: ρ(h) = φ^h = {phi}^h', fontsize=11)
ax2.set_xlabel('Lag h')
ax2.set_ylabel('ρ(h)')
ax2.grid(True, alpha=0.3)

# Add values
for i, v in enumerate(acf_theoretical[:5]):
    ax2.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

# Summary
ax3 = axes[2]
ax3.axis('off')
summary = f"""
AR(1) Solution Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: X_t = {c} + {phi}X_{{t-1}} + ε_t
       ε_t ~ WN(0, {sigma_sq})

Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Mean:
   μ = c/(1-φ) = {c}/(1-{phi}) = {mu:.4f}

2. Variance:
   γ(0) = σ²/(1-φ²) = {sigma_sq}/{1-phi**2:.4f}
        = {gamma_0:.4f}

3. Autocovariance:
   γ(1) = φ × γ(0) = {phi} × {gamma_0:.2f}
        = {gamma_1:.4f}
   γ(2) = φ² × γ(0) = {phi**2} × {gamma_0:.2f}
        = {gamma_2:.4f}

4. Autocorrelation:
   ρ(1) = φ = {rho_1}
   ρ(2) = φ² = {rho_2}
"""
ax3.text(0.05, 0.95, summary, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('../../charts/ch2_seminar_ex1_ar1.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch2_seminar_ex1_ar1.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("\n" + "=" * 60)
print("FINAL ANSWERS:")
print("=" * 60)
print(f"1. Mean: μ = {mu:.2f}")
print(f"2. Variance: γ(0) = {gamma_0:.2f}")
print(f"3. Autocovariance: γ(1) = {gamma_1:.2f}, γ(2) = {gamma_2:.2f}")
print(f"4. Autocorrelation: ρ(1) = {rho_1}, ρ(2) = {rho_2}")
