"""
TSA_ch1_ex4_ma1
===============
Seminar Exercise 4: MA(1) Process

Problem: X_t = ε_t + 0.6ε_{t-1} where ε_t ~ WN(0, 4)
Calculate E[X_t], Var(X_t), γ(1), ρ(1), ρ(2)
"""

import numpy as np
import matplotlib.pyplot as plt

# Given parameters
theta = 0.6         # MA coefficient
sigma_sq = 4        # Variance of white noise
sigma = np.sqrt(sigma_sq)

print("=" * 60)
print("MA(1) PROCESS - Step by Step Solution")
print("=" * 60)
print(f"\nModel: X_t = ε_t + {theta}ε_{{t-1}}")
print(f"where ε_t ~ WN(0, {sigma_sq})")

# Part a: E[X_t]
print("\n" + "-" * 60)
print("a) Calculate E[X_t]")
print("-" * 60)
print("\nE[X_t] = E[ε_t + θε_{t-1}]")
print("      = E[ε_t] + θ E[ε_{t-1}]")
print("      = 0 + θ × 0")

E_X = 0
print(f"      = {E_X}")
print()
print("Note: MA processes ALWAYS have zero mean when built from WN(0,σ²)")

# Part b: Var(X_t)
print("\n" + "-" * 60)
print("b) Calculate Var(X_t)")
print("-" * 60)
print("\nVar(X_t) = Var(ε_t + θε_{t-1})")
print("        = Var(ε_t) + θ² Var(ε_{t-1})  [since ε's are independent]")
print("        = σ² + θ² σ²")
print("        = σ²(1 + θ²)")
print()
print(f"        = {sigma_sq} × (1 + {theta}²)")
print(f"        = {sigma_sq} × (1 + {theta**2})")
print(f"        = {sigma_sq} × {1 + theta**2}")

Var_X = sigma_sq * (1 + theta**2)
print(f"        = {Var_X:.2f}")

# Part c: γ(1)
print("\n" + "-" * 60)
print("c) Calculate γ(1) = Cov(X_t, X_{t-1})")
print("-" * 60)
print("\nX_t = ε_t + θε_{t-1}")
print("X_{t-1} = ε_{t-1} + θε_{t-2}")
print()
print("Cov(X_t, X_{t-1}) = Cov(ε_t + θε_{t-1}, ε_{t-1} + θε_{t-2})")
print("                 = Cov(ε_t, ε_{t-1}) + Cov(ε_t, θε_{t-2})")
print("                   + Cov(θε_{t-1}, ε_{t-1}) + Cov(θε_{t-1}, θε_{t-2})")
print("                 = 0 + 0 + θ Var(ε_{t-1}) + 0")
print("                 = θ × σ²")
print()
print(f"                 = {theta} × {sigma_sq}")

gamma_1 = theta * sigma_sq
print(f"                 = {gamma_1}")

# Part d: ρ(1) and ρ(2)
print("\n" + "-" * 60)
print("d) Calculate ρ(1) and ρ(2)")
print("-" * 60)
print("\nρ(h) = γ(h) / γ(0) = γ(h) / Var(X_t)")
print()
print("For ρ(1):")
print(f"  ρ(1) = γ(1) / Var(X_t)")
print(f"       = {gamma_1} / {Var_X}")
print(f"       = θσ² / σ²(1+θ²)")
print(f"       = θ / (1+θ²)")
print(f"       = {theta} / (1 + {theta**2})")
print(f"       = {theta} / {1 + theta**2}")

rho_1 = theta / (1 + theta**2)
print(f"       = {rho_1:.4f}")

print("\nFor ρ(2):")
print("  γ(2) = Cov(X_t, X_{t-2})")
print("       = Cov(ε_t + θε_{t-1}, ε_{t-2} + θε_{t-3})")
print("       = 0 + 0 + 0 + 0 = 0")
print()
print("  No shared ε terms between X_t and X_{t-2}!")
print()

rho_2 = 0
print(f"  ρ(2) = γ(2) / Var(X_t) = 0 / {Var_X} = {rho_2}")

print("\n" + "=" * 60)
print("KEY INSIGHT: MA(1) ACF CUTS OFF after lag 1!")
print("=" * 60)
print("  ρ(h) = 0 for all h > 1")
print("  This is the SIGNATURE of an MA(1) process!")
print("  (Compare with AR(1) which decays gradually)")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: ACF comparison
ax1 = axes[0, 0]
lags = np.arange(0, 11)
ma1_acf = np.zeros(11)
ma1_acf[0] = 1.0
ma1_acf[1] = rho_1

ax1.bar(lags - 0.15, ma1_acf, 0.3, color='blue', alpha=0.7, edgecolor='black', label='MA(1)')

# Compare with AR(1) with same ρ(1)
ar1_phi = rho_1
ar1_acf = ar1_phi ** lags
ax1.bar(lags + 0.15, ar1_acf, 0.3, color='red', alpha=0.7, edgecolor='black', label='AR(1) same ρ(1)')

ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.set_xlabel('Lag (h)')
ax1.set_ylabel('ρ(h)')
ax1.set_title('ACF: MA(1) cuts off vs AR(1) decays')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticks(lags)

# Top-right: Simulated MA(1)
ax2 = axes[0, 1]
np.random.seed(42)
n = 200
eps = np.random.normal(0, sigma, n+1)
x = np.zeros(n)
for t in range(n):
    x[t] = eps[t+1] + theta * eps[t]

ax2.plot(x, 'b-', linewidth=0.8, alpha=0.8)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Mean = 0')
ax2.axhline(y=2*np.sqrt(Var_X), color='gray', linestyle=':', alpha=0.7)
ax2.axhline(y=-2*np.sqrt(Var_X), color='gray', linestyle=':', alpha=0.7, label='±2 SD')
ax2.set_xlabel('Time')
ax2.set_ylabel('X_t')
ax2.set_title(f'Simulated MA(1): X_t = ε_t + {theta}ε_{{t-1}}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Bottom-left: Impact of theta on ρ(1)
ax3 = axes[1, 0]
theta_vals = np.linspace(-0.99, 0.99, 100)
rho1_vals = theta_vals / (1 + theta_vals**2)
ax3.plot(theta_vals, rho1_vals, 'b-', linewidth=2)
ax3.scatter([theta], [rho_1], color='red', s=100, zorder=5, label=f'θ = {theta}, ρ(1) = {rho_1:.4f}')
ax3.axhline(y=0, color='gray', linestyle='--')
ax3.axvline(x=0, color='gray', linestyle='--')
ax3.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Maximum ρ(1) = 0.5')
ax3.axhline(y=-0.5, color='green', linestyle=':', alpha=0.5)
ax3.set_xlabel('θ')
ax3.set_ylabel('ρ(1)')
ax3.set_title('MA(1): ρ(1) = θ/(1+θ²)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-1.1, 1.1)

# Bottom-right: Summary table
ax4 = axes[1, 1]
ax4.axis('off')
summary = f"""
MA(1) Process Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model:     X_t = ε_t + θε_{{t-1}}

Given:     θ = {theta}
           σ² = {sigma_sq}

Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
E[X_t]     = 0
Var(X_t)   = σ²(1 + θ²) = {Var_X:.2f}
γ(1)       = θσ² = {gamma_1}
ρ(1)       = θ/(1+θ²) = {rho_1:.4f}
ρ(2)       = 0 (and ρ(h) = 0 for h > 1)

Key Feature:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACF CUTS OFF at lag 1!
This identifies MA(1) vs AR processes.

Invertibility: |θ| < 1 required
"""
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('../../charts/ch1_seminar_ex4_ma1.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("FINAL ANSWERS:")
print("=" * 60)
print(f"a) E[X_t] = {E_X}")
print(f"b) Var(X_t) = {Var_X:.2f}")
print(f"c) γ(1) = {gamma_1}")
print(f"d) ρ(1) = {rho_1:.4f}, ρ(2) = {rho_2}")
