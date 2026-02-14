"""
TSA_ch1_ex1_autocovariance
==========================
Seminar Exercise 1: Autocovariance and Autocorrelation

Problem: Given E[X_t]=5, γ(0)=4, γ(1)=2, γ(2)=1
Calculate autocorrelations, covariances, and conditional expectation
"""

import numpy as np
import matplotlib.pyplot as plt

# Given values
mu = 5           # E[X_t] = mean
gamma_0 = 4      # γ(0) = Var(X_t)
gamma_1 = 2      # γ(1) = Cov(X_t, X_{t-1})
gamma_2 = 1      # γ(2) = Cov(X_t, X_{t-2})

print("=" * 60)
print("AUTOCOVARIANCE & AUTOCORRELATION - Step by Step Solution")
print("=" * 60)
print("\nGiven:")
print(f"  E[X_t] = μ = {mu}")
print(f"  γ(0) = Var(X_t) = {gamma_0}")
print(f"  γ(1) = Cov(X_t, X_{t-1}) = {gamma_1}")
print(f"  γ(2) = Cov(X_t, X_{t-2}) = {gamma_2}")

# Part a: Autocorrelation function
print("\n" + "-" * 60)
print("a) Calculate Autocorrelation Function ρ(0), ρ(1), ρ(2)")
print("-" * 60)
print("\nFormula: ρ(h) = γ(h) / γ(0)")
print("\nCalculations:")

rho_0 = gamma_0 / gamma_0
rho_1 = gamma_1 / gamma_0
rho_2 = gamma_2 / gamma_0

print(f"  ρ(0) = γ(0) / γ(0) = {gamma_0} / {gamma_0} = {rho_0}")
print(f"  ρ(1) = γ(1) / γ(0) = {gamma_1} / {gamma_0} = {rho_1}")
print(f"  ρ(2) = γ(2) / γ(0) = {gamma_2} / {gamma_0} = {rho_2}")

print("\nNote: ρ(0) is ALWAYS 1 (correlation with itself)")

# Part b: Cov(X_t, X_{t-1})
print("\n" + "-" * 60)
print("b) Calculate Cov(X_t, X_{t-1})")
print("-" * 60)
print("\nFor stationary process, Cov(X_t, X_{t-h}) = γ(h)")
print(f"\nCov(X_t, X_{t-1}) = γ(1) = {gamma_1}")

# Part c: Corr(X_5, X_7)
print("\n" + "-" * 60)
print("c) Calculate Corr(X_5, X_7)")
print("-" * 60)
print("\nFor stationary process, Corr(X_t, X_{t+h}) = ρ(h) regardless of t")
print("\nCorr(X_5, X_7) = Corr(X_5, X_{5+2}) = ρ(2)")
print(f"Corr(X_5, X_7) = {rho_2}")

# Part d: Conditional expectation (assuming AR(1))
print("\n" + "-" * 60)
print("d) E[X_{t+1} | X_t = 6] assuming AR(1) process")
print("-" * 60)
print("\nFor AR(1): X_t = μ + φ(X_{t-1} - μ) + ε_t")
print("where φ = ρ(1) for AR(1)")
print()
print(f"Given: φ = ρ(1) = {rho_1}")
print(f"       μ = {mu}")
print(f"       X_t = 6")
print()
print("Conditional expectation for AR(1):")
print("  E[X_{t+1} | X_t] = μ + φ(X_t - μ)")
print(f"  E[X_{t+1} | X_t = 6] = {mu} + {rho_1} × (6 - {mu})")
print(f"  E[X_{t+1} | X_t = 6] = {mu} + {rho_1} × {6 - mu}")
print(f"  E[X_{t+1} | X_t = 6] = {mu} + {rho_1 * (6 - mu)}")

conditional_expectation = mu + rho_1 * (6 - mu)
print(f"  E[X_{t+1} | X_t = 6] = {conditional_expectation}")

print("\nInterpretation:")
print(f"  X_t = 6 is above the mean ({mu})")
print(f"  AR(1) with φ = {rho_1} shows mean reversion")
print(f"  So E[X_{t+1}] = {conditional_expectation} is between X_t and μ")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: ACF
ax1 = axes[0]
lags = [0, 1, 2]
acf_values = [rho_0, rho_1, rho_2]
ax1.bar(lags, acf_values, color='blue', alpha=0.7, edgecolor='black')
ax1.axhline(y=0, color='black', linewidth=0.5)
ax1.set_xlabel('Lag (h)')
ax1.set_ylabel('Autocorrelation ρ(h)')
ax1.set_title('Autocorrelation Function (ACF)')
ax1.set_xticks(lags)
ax1.set_ylim(0, 1.2)
ax1.grid(True, alpha=0.3, axis='y')

for lag, rho in zip(lags, acf_values):
    ax1.text(lag, rho + 0.02, f'{rho:.2f}', ha='center', va='bottom', fontsize=11)

# Right: Conditional expectation visualization
ax2 = axes[1]
x_range = np.linspace(2, 8, 50)
x_current = 6

# Mean and variance
ax2.axhline(y=mu, color='red', linestyle='--', linewidth=2, label=f'Mean μ = {mu}')
ax2.axvline(x=x_current, color='green', linestyle='--', linewidth=2, label=f'$X_t$ = {x_current}')
ax2.plot([x_current], [conditional_expectation], 'b*', markersize=20, label=f'E[$X_{{t+1}}$|$X_t$] = {conditional_expectation}')

# Draw arrow showing mean reversion
ax2.annotate('', xy=(x_current + 0.5, conditional_expectation), xytext=(x_current + 0.5, x_current),
             arrowprops=dict(arrowstyle='->', color='purple', lw=2))
ax2.text(x_current + 0.7, (x_current + conditional_expectation)/2, 'Mean\nReversion',
         fontsize=9, color='purple')

ax2.set_xlabel('Time')
ax2.set_ylabel('Value')
ax2.set_title(f'AR(1) Mean Reversion: φ = {rho_1}')
ax2.set_xlim(4, 8)
ax2.set_ylim(4, 7)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../charts/ch1_seminar_ex1_autocovariance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("FINAL ANSWERS:")
print("=" * 60)
print(f"a) ρ(0) = {rho_0}, ρ(1) = {rho_1}, ρ(2) = {rho_2}")
print(f"b) Cov(X_t, X_{{t-1}}) = γ(1) = {gamma_1}")
print(f"c) Corr(X_5, X_7) = ρ(2) = {rho_2}")
print(f"d) E[X_{{t+1}} | X_t = 6] = {conditional_expectation}")
