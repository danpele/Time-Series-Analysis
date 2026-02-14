"""
TSA_ch2_ex4_forecast
====================
Seminar Exercise 4: AR(1) Forecasting

Problem: X_t = 3 + 0.8X_{t-1} + ε_t, σ² = 4, X_100 = 20
Calculate 1-step, 2-step, long-run forecasts and 95% CI
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Given parameters
c = 3           # constant
phi = 0.8       # AR coefficient
sigma_sq = 4    # variance of white noise
X_100 = 20      # current observation

print("=" * 60)
print("AR(1) FORECASTING - Step by Step Solution")
print("=" * 60)
print(f"\nModel: X_t = {c} + {phi}X_{{t-1}} + ε_t")
print(f"where σ² = {sigma_sq}")
print(f"Given: X_100 = {X_100}")

# First, calculate the unconditional mean
print("\n" + "-" * 60)
print("Preliminary: Calculate Unconditional Mean μ")
print("-" * 60)
print("\nμ = c / (1 - φ)")
print(f"μ = {c} / (1 - {phi})")
print(f"μ = {c} / {1 - phi}")

mu = c / (1 - phi)
print(f"μ = {mu}")

# 1. One-step forecast
print("\n" + "-" * 60)
print("1. Calculate 1-Step Ahead Forecast X̂_{101|100}")
print("-" * 60)
print("\nForecast formula:")
print("  X̂_{n+1|n} = c + φX_n")
print()
print(f"  X̂_{{101|100}} = {c} + {phi} × X_100")
print(f"             = {c} + {phi} × {X_100}")
print(f"             = {c} + {phi * X_100}")

X_hat_101 = c + phi * X_100
print(f"             = {X_hat_101}")

# 2. Two-step forecast
print("\n" + "-" * 60)
print("2. Calculate 2-Step Ahead Forecast X̂_{102|100}")
print("-" * 60)
print("\nForecast formula (iterating):")
print("  X̂_{n+2|n} = c + φX̂_{n+1|n}")
print()
print(f"  X̂_{{102|100}} = {c} + {phi} × X̂_{{101|100}}")
print(f"             = {c} + {phi} × {X_hat_101}")
print(f"             = {c} + {phi * X_hat_101}")

X_hat_102 = c + phi * X_hat_101
print(f"             = {X_hat_102}")

print("\nAlternative formula:")
print("  X̂_{n+h|n} = μ + φ^h(X_n - μ)")
print(f"  X̂_{{102|100}} = {mu} + {phi}²×({X_100} - {mu})")
print(f"             = {mu} + {phi**2}×{X_100 - mu}")
print(f"             = {mu} + {phi**2 * (X_100 - mu)}")
print(f"             = {mu + phi**2 * (X_100 - mu)} ✓")

# 3. Long-run forecast
print("\n" + "-" * 60)
print("3. Calculate Long-Run Forecast (h → ∞)")
print("-" * 60)
print("\nAs h → ∞:")
print("  X̂_{n+h|n} = μ + φ^h(X_n - μ)")
print()
print(f"Since |φ| = {abs(phi)} < 1:")
print(f"  lim_{{h→∞}} φ^h = lim_{{h→∞}} {phi}^h = 0")
print()
print("Therefore:")
print(f"  lim_{{h→∞}} X̂_{{100+h|100}} = μ + 0×(X_100 - μ) = μ = {mu}")

X_hat_inf = mu

# 4. 95% Confidence interval for 1-step forecast
print("\n" + "-" * 60)
print("4. Calculate 95% CI for X̂_{101|100}")
print("-" * 60)
print("\nMean Squared Forecast Error for 1-step ahead:")
print("  MSFE(1) = Var(ε_{n+1}) = σ²")
print(f"  MSFE(1) = {sigma_sq}")
print()
print("Standard error:")
print(f"  SE(1) = √MSFE(1) = √{sigma_sq} = {np.sqrt(sigma_sq)}")
print()
print("95% Confidence Interval:")
print("  X̂_{101|100} ± 1.96 × SE(1)")
print(f"  {X_hat_101} ± 1.96 × {np.sqrt(sigma_sq)}")
print(f"  {X_hat_101} ± {1.96 * np.sqrt(sigma_sq):.2f}")

CI_lower = X_hat_101 - 1.96 * np.sqrt(sigma_sq)
CI_upper = X_hat_101 + 1.96 * np.sqrt(sigma_sq)
print(f"\n95% CI: [{CI_lower:.2f}, {CI_upper:.2f}]")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Forecast path
ax1 = axes[0]
h_vals = np.arange(0, 21)
forecasts = mu + phi**h_vals * (X_100 - mu)

ax1.plot(h_vals, forecasts, 'bo-', markersize=8, linewidth=2, label='Point Forecast')
ax1.axhline(y=mu, color='red', linestyle='--', linewidth=2, label=f'Long-run mean μ = {mu}')
ax1.scatter([0], [X_100], color='green', s=150, zorder=5, label=f'X_100 = {X_100}')
ax1.set_title('Forecast Path: Mean Reversion', fontsize=12)
ax1.set_xlabel('Horizon h')
ax1.set_ylabel('E[X_{100+h}|X_100]')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax1.grid(True, alpha=0.3)

# Forecast with confidence intervals
ax2 = axes[1]
# Calculate MSFE for each horizon
msfe = np.array([sigma_sq * sum(phi**(2*j) for j in range(i+1)) for i in range(20)])
se = np.sqrt(msfe)

ax2.plot(range(1, 21), forecasts[1:], 'bo-', markersize=6, linewidth=2, label='Point Forecast')
ax2.fill_between(range(1, 21),
                 forecasts[1:] - 1.96*se,
                 forecasts[1:] + 1.96*se,
                 alpha=0.2, color='blue', label='95% CI')
ax2.axhline(y=mu, color='red', linestyle='--', linewidth=2, label=f'μ = {mu}')
ax2.set_title('Forecasts with Expanding Confidence Intervals', fontsize=12)
ax2.set_xlabel('Horizon h')
ax2.set_ylabel('Forecast')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax2.grid(True, alpha=0.3)

# Summary
ax3 = axes[2]
ax3.axis('off')
summary = f"""
AR(1) Forecasting Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: X_t = {c} + {phi}X_{{t-1}} + ε_t
       σ² = {sigma_sq}, X_100 = {X_100}

Unconditional Mean:
  μ = c/(1-φ) = {c}/{1-phi} = {mu}

Forecasts:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. One-step ahead:
   X̂_{{101|100}} = c + φX_100
                = {c} + {phi}×{X_100}
                = {X_hat_101}

2. Two-step ahead:
   X̂_{{102|100}} = c + φX̂_{{101|100}}
                = {c} + {phi}×{X_hat_101}
                = {X_hat_102}

3. Long-run (h→∞):
   lim X̂_{{100+h|100}} = μ = {mu}

4. 95% CI for X̂_{{101|100}}:
   {X_hat_101} ± 1.96×√{sigma_sq}
   = [{CI_lower:.2f}, {CI_upper:.2f}]
"""
ax3.text(0.02, 0.98, summary, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('../../charts/ch2_seminar_ex4_forecast.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("FINAL ANSWERS:")
print("=" * 60)
print(f"Mean: μ = {mu}")
print(f"1. X̂_{{101|100}} = {X_hat_101}")
print(f"2. X̂_{{102|100}} = {X_hat_102}")
print(f"3. Long-run forecast: {X_hat_inf}")
print(f"4. 95% CI for X̂_{{101|100}}: [{CI_lower:.2f}, {CI_upper:.2f}]")
