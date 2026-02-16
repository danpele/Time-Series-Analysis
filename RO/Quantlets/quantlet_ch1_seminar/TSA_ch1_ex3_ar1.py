"""
TSA_ch1_ex3_ar1
===============
Seminar Exercise 3: AR(1) Process

Problem: X_t = 0.8 X_{t-1} + ε_t where ε_t ~ WN(0, 9)
Check stationarity, calculate variance and autocorrelations
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
phi = 0.8           # AR coefficient
sigma_sq = 9        # Variance of white noise
sigma = np.sqrt(sigma_sq)

print("=" * 60)
print("AR(1) PROCESS - Step by Step Solution")
print("=" * 60)
print(f"\nModel: X_t = {phi} X_{{t-1}} + ε_t")
print(f"where ε_t ~ WN(0, {sigma_sq})")

# Part a: Stationarity check
print("\n" + "-" * 60)
print("a) Is this process stationary? Why?")
print("-" * 60)
print("\nAR(1) stationarity condition: |φ| < 1")
print()
print(f"Given: φ = {phi}")
print(f"Check: |{phi}| = {abs(phi)} < 1 ?")
print(f"       {abs(phi)} < 1 ✓ TRUE")
print()
print("Therefore: The process IS stationary.")
print()
print("Alternative view (characteristic equation):")
print("  1 - φL = 0  →  L = 1/φ = 1/0.8 = 1.25")
print("  Root is OUTSIDE the unit circle (|1.25| > 1)")
print("  → Stationary")

# Part b: Unconditional variance
print("\n" + "-" * 60)
print("b) Calculate the unconditional variance Var(X_t)")
print("-" * 60)
print("\nFor stationary AR(1):")
print("  Var(X_t) = σ² / (1 - φ²)")
print()
print(f"  Var(X_t) = {sigma_sq} / (1 - {phi}²)")
print(f"          = {sigma_sq} / (1 - {phi**2})")
print(f"          = {sigma_sq} / {1 - phi**2}")

Var_X = sigma_sq / (1 - phi**2)
print(f"          = {Var_X:.2f}")
print()
print(f"Standard deviation: SD(X_t) = √{Var_X:.2f} = {np.sqrt(Var_X):.2f}")

# Part c: Autocorrelations
print("\n" + "-" * 60)
print("c) Calculate ρ(1), ρ(2), ρ(3)")
print("-" * 60)
print("\nFor AR(1): ρ(h) = φ^h (geometric decay)")
print()

rho_1 = phi**1
rho_2 = phi**2
rho_3 = phi**3

print(f"  ρ(1) = φ¹ = {phi}^1 = {rho_1}")
print(f"  ρ(2) = φ² = {phi}^2 = {rho_2}")
print(f"  ρ(3) = φ³ = {phi}^3 = {rho_3}")
print()
print("Note: ACF decays EXPONENTIALLY (geometric decay)")
print(f"Since φ = {phi} > 0, ACF is always positive")
print("(If φ < 0, ACF would alternate in sign)")

# Part d: Conditional expectations
print("\n" + "-" * 60)
print("d) Calculate E[X_{t+1}|X_t=10] and E[X_{t+2}|X_t=10]")
print("-" * 60)
print("\nFor AR(1) with zero mean:")
print("  E[X_{t+h}|X_t] = φ^h × X_t")
print()
X_t = 10

E_Xt1 = phi**1 * X_t
E_Xt2 = phi**2 * X_t

print(f"Given: X_t = {X_t}")
print()
print(f"  E[X_{{t+1}}|X_t={X_t}] = φ¹ × X_t = {phi} × {X_t} = {E_Xt1}")
print(f"  E[X_{{t+2}}|X_t={X_t}] = φ² × X_t = {phi**2} × {X_t} = {E_Xt2}")
print()
print("Interpretation:")
print(f"  Starting from X_t = {X_t}:")
print(f"  - After 1 period, expect {E_Xt1}")
print(f"  - After 2 periods, expect {E_Xt2}")
print(f"  - Long-run: E[X_∞] → 0 (unconditional mean)")
print(f"  This shows MEAN REVERSION: values drift toward the mean")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top-left: ACF
ax1 = axes[0, 0]
lags = np.arange(0, 16)
acf_values = phi ** lags
ax1.bar(lags, acf_values, color='blue', alpha=0.7, edgecolor='black')
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Approx. significance')
ax1.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
ax1.set_xlabel('Lag (h)')
ax1.set_ylabel('ρ(h)')
ax1.set_title(f'ACF of AR(1) with φ = {phi}')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax1.grid(True, alpha=0.3, axis='y')

# Top-right: Simulated AR(1) path
ax2 = axes[0, 1]
np.random.seed(42)
n = 200
x = np.zeros(n)
eps = np.random.normal(0, sigma, n)
for t in range(1, n):
    x[t] = phi * x[t-1] + eps[t]

ax2.plot(x, 'b-', linewidth=0.8, alpha=0.8)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Mean = 0')
ax2.axhline(y=2*np.sqrt(Var_X), color='gray', linestyle=':', alpha=0.7)
ax2.axhline(y=-2*np.sqrt(Var_X), color='gray', linestyle=':', alpha=0.7, label='±2 SD')
ax2.set_xlabel('Time')
ax2.set_ylabel('X_t')
ax2.set_title(f'Simulated AR(1): X_t = {phi}X_{{t-1}} + ε_t')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax2.grid(True, alpha=0.3)

# Bottom-left: Forecast decay
ax3 = axes[1, 0]
h_vals = np.arange(0, 11)
forecast_from_10 = phi ** h_vals * X_t
ax3.plot(h_vals, forecast_from_10, 'go-', markersize=8, linewidth=2, label=f'E[X_{{t+h}}|X_t={X_t}]')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Long-run mean')
ax3.set_xlabel('Horizon (h)')
ax3.set_ylabel('E[X_{t+h}|X_t]')
ax3.set_title('Conditional Expectation: Mean Reversion')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(h_vals)

# Bottom-right: Stationarity region
ax4 = axes[1, 1]
phi_vals = np.linspace(-1.2, 1.2, 100)
ax4.axvspan(-1, 1, alpha=0.2, color='green', label='Stationary region')
ax4.axvline(x=1, color='red', linestyle='--', linewidth=2)
ax4.axvline(x=-1, color='red', linestyle='--', linewidth=2, label='Unit root boundary')
ax4.axvline(x=phi, color='blue', linewidth=3, label=f'φ = {phi}')
ax4.scatter([phi], [0], color='blue', s=200, zorder=5)
ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(-0.5, 0.5)
ax4.set_xlabel('φ')
ax4.set_title('AR(1) Stationarity: |φ| < 1')
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax4.set_yticks([])
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../../charts/ch1_seminar_ex3_ar1.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch1_seminar_ex3_ar1.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("\n" + "=" * 60)
print("FINAL ANSWERS:")
print("=" * 60)
print(f"a) Stationary? YES, because |φ| = |{phi}| = {abs(phi)} < 1")
print(f"b) Var(X_t) = {Var_X:.2f}")
print(f"c) ρ(1) = {rho_1}, ρ(2) = {rho_2}, ρ(3) = {rho_3}")
print(f"d) E[X_{{t+1}}|X_t=10] = {E_Xt1}, E[X_{{t+2}}|X_t=10] = {E_Xt2}")
