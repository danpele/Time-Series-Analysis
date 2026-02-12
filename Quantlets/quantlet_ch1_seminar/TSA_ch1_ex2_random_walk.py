"""
TSA_ch1_ex2_random_walk
=======================
Seminar Exercise 2: Random Walk Properties

Problem: X_t = X_{t-1} + ε_t where ε_t ~ WN(0, 4), X_0 = 100
Calculate E[X_10], Var(X_10), Cov(X_5, X_10), 95% CI for X_100
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Given parameters
X_0 = 100          # Initial value
sigma_sq = 4       # Variance of white noise ε_t
sigma = np.sqrt(sigma_sq)

print("=" * 60)
print("RANDOM WALK PROPERTIES - Step by Step Solution")
print("=" * 60)
print("\nModel: X_t = X_{t-1} + ε_t")
print(f"where ε_t ~ WN(0, {sigma_sq}) and X_0 = {X_0}")
print()
print("Key formulas for random walk:")
print("  X_t = X_0 + ε_1 + ε_2 + ... + ε_t")
print("  E[X_t] = X_0")
print("  Var(X_t) = t × σ²")
print("  Cov(X_s, X_t) = min(s,t) × σ²  for s ≤ t")

# Part a: E[X_10]
print("\n" + "-" * 60)
print("a) Calculate E[X_10]")
print("-" * 60)
print("\nFor random walk (no drift):")
print("  E[X_t] = X_0  (constant for all t)")
print()
E_X_10 = X_0
print(f"  E[X_10] = {E_X_10}")
print()
print("Intuition: Best forecast is the starting value since")
print("ε_t has zero mean and shocks are unpredictable.")

# Part b: Var(X_10)
print("\n" + "-" * 60)
print("b) Calculate Var(X_10)")
print("-" * 60)
print("\nFor random walk:")
print("  X_10 = X_0 + ε_1 + ε_2 + ... + ε_10")
print()
print("Since ε_t are independent:")
print("  Var(X_10) = Var(ε_1) + Var(ε_2) + ... + Var(ε_10)")
print("           = 10 × σ²")
print(f"           = 10 × {sigma_sq}")

Var_X_10 = 10 * sigma_sq
print(f"           = {Var_X_10}")
print()
print(f"Standard deviation: SD(X_10) = √{Var_X_10} = {np.sqrt(Var_X_10):.2f}")

# Part c: Cov(X_5, X_10)
print("\n" + "-" * 60)
print("c) Calculate Cov(X_5, X_10)")
print("-" * 60)
print("\nNote: X_5 = X_0 + ε_1 + ... + ε_5")
print("      X_10 = X_0 + ε_1 + ... + ε_5 + ε_6 + ... + ε_10")
print()
print("Cov(X_5, X_10) = Cov(ε_1+...+ε_5, ε_1+...+ε_10)")
print("              = Var(ε_1) + ... + Var(ε_5)  (shared terms)")
print("              = 5 × σ²")
print(f"              = 5 × {sigma_sq}")

Cov_X5_X10 = 5 * sigma_sq
print(f"              = {Cov_X5_X10}")
print()
print("General formula: Cov(X_s, X_t) = min(s,t) × σ²")

# Part d: 95% CI for X_100
print("\n" + "-" * 60)
print("d) 95% Confidence Interval for X_100")
print("-" * 60)
print("\nFor random walk:")
print(f"  E[X_100] = {X_0}")
print(f"  Var(X_100) = 100 × {sigma_sq} = {100 * sigma_sq}")
print(f"  SD(X_100) = √{100 * sigma_sq} = {np.sqrt(100 * sigma_sq):.2f}")
print()
print("Assuming normal distribution (CLT applies):")
print("  95% CI = E[X_100] ± 1.96 × SD(X_100)")

E_X_100 = X_0
SD_X_100 = np.sqrt(100 * sigma_sq)
z_95 = 1.96

CI_lower = E_X_100 - z_95 * SD_X_100
CI_upper = E_X_100 + z_95 * SD_X_100

print(f"         = {E_X_100} ± 1.96 × {SD_X_100:.2f}")
print(f"         = {E_X_100} ± {z_95 * SD_X_100:.2f}")
print(f"         = [{CI_lower:.1f}, {CI_upper:.1f}]")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Variance growth
ax1 = axes[0]
t_vals = np.arange(0, 101)
var_vals = t_vals * sigma_sq
sd_vals = np.sqrt(var_vals)

ax1.fill_between(t_vals, X_0 - 1.96*sd_vals, X_0 + 1.96*sd_vals,
                 alpha=0.2, color='blue', label='95% CI')
ax1.fill_between(t_vals, X_0 - sd_vals, X_0 + sd_vals,
                 alpha=0.3, color='blue', label='±1 SD')
ax1.axhline(y=X_0, color='red', linestyle='--', linewidth=2, label=f'E[X_t] = {X_0}')
ax1.plot(t_vals, X_0 + 1.96*sd_vals, 'b-', alpha=0.7)
ax1.plot(t_vals, X_0 - 1.96*sd_vals, 'b-', alpha=0.7)

# Mark specific points
ax1.scatter([10], [X_0], color='green', s=100, zorder=5, label=f'E[X_10] = {X_0}')
ax1.scatter([100], [X_0], color='purple', s=100, zorder=5, label=f'E[X_100] = {X_0}')

ax1.set_xlabel('Time (t)')
ax1.set_ylabel('X_t')
ax1.set_title('Random Walk: Variance Grows with Time')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 100)

# Right: Simulated paths
ax2 = axes[1]
np.random.seed(42)
n_paths = 20
for i in range(n_paths):
    path = X_0 + np.cumsum(np.random.normal(0, sigma, 100))
    path = np.insert(path, 0, X_0)
    ax2.plot(t_vals, path, alpha=0.3, linewidth=0.8)

ax2.axhline(y=X_0, color='red', linestyle='--', linewidth=2, label=f'Start = {X_0}')
ax2.axhline(y=CI_lower, color='blue', linestyle=':', linewidth=2)
ax2.axhline(y=CI_upper, color='blue', linestyle=':', linewidth=2, label=f'95% CI at t=100')

ax2.set_xlabel('Time (t)')
ax2.set_ylabel('X_t')
ax2.set_title('Random Walk: 20 Simulated Paths')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 100)

plt.tight_layout()
plt.savefig('../../charts/ch1_seminar_ex2_random_walk.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("FINAL ANSWERS:")
print("=" * 60)
print(f"a) E[X_10] = {E_X_10}")
print(f"b) Var(X_10) = {Var_X_10}")
print(f"c) Cov(X_5, X_10) = {Cov_X5_X10}")
print(f"d) 95% CI for X_100: [{CI_lower:.1f}, {CI_upper:.1f}]")

print("\n" + "=" * 60)
print("KEY INSIGHT:")
print("=" * 60)
print("""
Random walk uncertainty grows with time!
  - At t=10: 95% CI width = 2 × 1.96 × √40 ≈ 24.8
  - At t=100: 95% CI width = 2 × 1.96 × √400 ≈ 78.4

This is why random walks are non-stationary:
variance is NOT constant but depends on t.
""")
