"""
TSA_ch2_ex2_ma1
===============
Seminar Exercise 2: MA(1) Properties

Problem: X_t = 5 + ε_t - 0.4ε_{t-1}, ε_t ~ WN(0, 4)
Calculate mean, variance, autocovariance, autocorrelation, check invertibility
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
mu_given = 5        # constant (mean)
theta = -0.4        # MA coefficient (note the sign!)
sigma_sq = 4        # variance of white noise

print("=" * 60)
print("MA(1) PROPERTIES - Step by Step Solution")
print("=" * 60)
print(f"\nModel: X_t = {mu_given} + ε_t - {abs(theta)}ε_{{t-1}}")
print(f"       X_t = {mu_given} + ε_t + ({theta})ε_{{t-1}}")
print(f"where ε_t ~ WN(0, {sigma_sq})")
print(f"\nNote: θ = {theta} (negative coefficient)")

# 1. Mean
print("\n" + "-" * 60)
print("1. Calculate E[X_t]")
print("-" * 60)
print("\nE[X_t] = E[μ + ε_t + θε_{t-1}]")
print("       = μ + E[ε_t] + θ E[ε_{t-1}]")
print("       = μ + 0 + θ × 0")

E_X = mu_given
print(f"       = {E_X}")
print("\nNote: MA processes with constant have mean equal to the constant.")

# 2. Variance
print("\n" + "-" * 60)
print("2. Calculate γ(0) = Var(X_t)")
print("-" * 60)
print("\nVar(X_t) = Var(ε_t + θε_{t-1})")
print("         = Var(ε_t) + θ² Var(ε_{t-1})  [ε's independent]")
print("         = σ² + θ² σ²")
print("         = σ²(1 + θ²)")
print()
print(f"γ(0) = {sigma_sq} × (1 + ({theta})²)")
print(f"γ(0) = {sigma_sq} × (1 + {theta**2})")
print(f"γ(0) = {sigma_sq} × {1 + theta**2}")

gamma_0 = sigma_sq * (1 + theta**2)
print(f"γ(0) = {gamma_0:.2f}")

# 3. Autocovariance at lag 1
print("\n" + "-" * 60)
print("3. Calculate γ(1) = Cov(X_t, X_{t-1})")
print("-" * 60)
print("\nX_t = ε_t + θε_{t-1}")
print("X_{t-1} = ε_{t-1} + θε_{t-2}")
print()
print("Cov(X_t, X_{t-1}) = Cov(ε_t + θε_{t-1}, ε_{t-1} + θε_{t-2})")
print("                 = Cov(ε_t, ε_{t-1}) + Cov(ε_t, θε_{t-2})")
print("                   + Cov(θε_{t-1}, ε_{t-1}) + Cov(θε_{t-1}, θε_{t-2})")
print("                 = 0 + 0 + θ × Var(ε_{t-1}) + 0")
print("                 = θσ²")
print()
print(f"γ(1) = {theta} × {sigma_sq}")

gamma_1 = theta * sigma_sq
print(f"γ(1) = {gamma_1}")

# 4. Autocorrelation
print("\n" + "-" * 60)
print("4. Calculate ρ(1)")
print("-" * 60)
print("\nρ(1) = γ(1) / γ(0)")
print(f"     = {gamma_1} / {gamma_0}")

rho_1 = gamma_1 / gamma_0
print(f"     = {rho_1:.4f}")

print("\nAlternative formula: ρ(1) = θ / (1 + θ²)")
print(f"                          = {theta} / (1 + {theta**2})")
print(f"                          = {theta} / {1 + theta**2}")
print(f"                          = {theta / (1 + theta**2):.4f} ✓")

# Note about ρ(2) and beyond
print("\n" + "-" * 60)
print("Note: ρ(h) for h > 1")
print("-" * 60)
print("\nFor MA(1): ρ(h) = 0 for all h > 1")
print("This is because X_t and X_{t-h} share no common ε terms when h > 1.")

# 5. Invertibility
print("\n" + "-" * 60)
print("5. Check Invertibility")
print("-" * 60)
print("\nMA(1) is invertible if |θ| < 1")
print()
print(f"|θ| = |{theta}| = {abs(theta)}")
print(f"{abs(theta)} < 1 ? {'YES ✓' if abs(theta) < 1 else 'NO ✗'}")

if abs(theta) < 1:
    print("\n→ The process IS INVERTIBLE")
    print("  This means it can be written as an AR(∞) process.")
else:
    print("\n→ The process is NOT INVERTIBLE")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Simulated MA(1)
np.random.seed(42)
n = 200
eps = np.random.normal(0, np.sqrt(sigma_sq), n+1)
x = np.zeros(n)
for t in range(n):
    x[t] = mu_given + eps[t+1] + theta * eps[t]

ax1 = axes[0]
ax1.plot(x, 'b-', linewidth=0.8, alpha=0.8)
ax1.axhline(y=mu_given, color='red', linestyle='--', linewidth=2, label=f'Mean = {mu_given}')
ax1.set_title(f'Simulated MA(1): X_t = {mu_given} + ε_t + ({theta})ε_{{t-1}}', fontsize=11)
ax1.set_xlabel('Time')
ax1.set_ylabel('X_t')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax1.grid(True, alpha=0.3)

# ACF
ax2 = axes[1]
lags = np.arange(0, 11)
acf_theoretical = np.zeros(11)
acf_theoretical[0] = 1
acf_theoretical[1] = rho_1
ax2.bar(lags, acf_theoretical, color='blue', alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_title('Theoretical ACF: Cuts off after lag 1', fontsize=11)
ax2.set_xlabel('Lag h')
ax2.set_ylabel('ρ(h)')
ax2.grid(True, alpha=0.3)
ax2.text(1, rho_1 - 0.1 if rho_1 < 0 else rho_1 + 0.05, f'{rho_1:.3f}', ha='center', fontsize=10)

# Summary
ax3 = axes[2]
ax3.axis('off')
summary = f"""
MA(1) Solution Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: X_t = {mu_given} + ε_t + ({theta})ε_{{t-1}}
       ε_t ~ WN(0, {sigma_sq})

Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Mean:
   E[X_t] = {E_X}

2. Variance:
   γ(0) = σ²(1+θ²) = {sigma_sq}×{1+theta**2:.2f}
        = {gamma_0:.2f}

3. Autocovariance at lag 1:
   γ(1) = θσ² = {theta}×{sigma_sq}
        = {gamma_1}

4. Autocorrelation:
   ρ(1) = θ/(1+θ²) = {rho_1:.4f}
   ρ(h) = 0 for h > 1

5. Invertibility:
   |θ| = {abs(theta)} < 1
   → {'INVERTIBLE ✓' if abs(theta) < 1 else 'NOT INVERTIBLE ✗'}
"""
ax3.text(0.05, 0.95, summary, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('../../charts/ch2_seminar_ex2_ma1.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch2_seminar_ex2_ma1.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("\n" + "=" * 60)
print("FINAL ANSWERS:")
print("=" * 60)
print(f"1. Mean: E[X_t] = {E_X}")
print(f"2. Variance: γ(0) = {gamma_0:.2f}")
print(f"3. Autocovariance: γ(1) = {gamma_1}")
print(f"4. Autocorrelation: ρ(1) = {rho_1:.3f}")
print(f"5. Invertibility: |θ| = {abs(theta)} < 1 → YES, invertible")
