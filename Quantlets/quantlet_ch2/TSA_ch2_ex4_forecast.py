"""
TSA_ch2_ex4_forecast
====================
Seminar Exercise 4: AR(1) Forecasting

Problem: X_t = 3 + 0.8*X_{t-1} + ε_t, σ² = 4, X_100 = 20
Calculate: 1-step, 2-step, long-run forecast, and 95% CI
"""

import numpy as np

print("=" * 60)
print("EXERCISE 4: AR(1) FORECASTING")
print("=" * 60)

# Given parameters
c = 3
phi = 0.8
sigma2 = 4
X_100 = 20

mu = c / (1 - phi)
print(f"\nGiven: X_t = {c} + {phi}*X_{{t-1}} + ε_t, σ² = {sigma2}, X_100 = {X_100}")
print(f"Mean: μ = c/(1-φ) = {c}/{1-phi:.1f} = {mu:.1f}")

# 1. One-step-ahead forecast
X_101 = c + phi * X_100
print(f"\n1. One-step forecast:")
print(f"   X̂_{{101|100}} = {c} + {phi} × {X_100} = {X_101:.1f}")

# 2. Two-step-ahead forecast
X_102 = c + phi * X_101
print(f"2. Two-step forecast:")
print(f"   X̂_{{102|100}} = {c} + {phi} × {X_101:.1f} = {X_102:.1f}")

# 3. Long-run forecast
print(f"3. Long-run forecast:")
print(f"   lim(h→∞) X̂_{{100+h|100}} = μ = {mu:.1f}")
print(f"   (Mean reversion: φ^h → 0 as h → ∞)")

# 4. 95% Confidence Interval
se = np.sqrt(sigma2)
z = 1.96
ci_lower = X_101 - z * se
ci_upper = X_101 + z * se
print(f"\n4. 95% Confidence Interval for X̂_{{101|100}}:")
print(f"   X̂ ± z_{{0.025}} × σ = {X_101:.1f} ± {z} × {se:.1f}")
print(f"   = [{ci_lower:.2f}, {ci_upper:.2f}]")
