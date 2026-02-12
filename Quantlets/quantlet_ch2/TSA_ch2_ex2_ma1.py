"""
TSA_ch2_ex2_ma1
===============
Seminar Exercise 2: MA(1) Properties

Problem: X_t = 5 + ε_t - 0.4*ε_{t-1}, ε_t ~ WN(0, 4)
Calculate: mean, variance, autocovariance, autocorrelation, invertibility
"""

import numpy as np

print("=" * 60)
print("EXERCISE 2: MA(1) PROPERTIES")
print("=" * 60)

# Given parameters
mu = 5
theta = -0.4
sigma2 = 4

print(f"\nGiven: X_t = {mu} + ε_t + ({theta})*ε_{{t-1}}, σ² = {sigma2}")

# 1. Mean
print(f"\n1. Mean: E[X_t] = μ = {mu}")

# 2. Variance
gamma0 = sigma2 * (1 + theta**2)
print(f"2. Variance: γ(0) = σ²(1+θ²) = {sigma2}×(1+{theta**2:.2f}) = {gamma0:.4f}")

# 3. Autocovariance
gamma1 = theta * sigma2
print(f"3. Autocovariance: γ(1) = θσ² = {theta} × {sigma2} = {gamma1:.4f}")
print(f"   γ(h) = 0 for h > 1 (MA(1) property)")

# 4. Autocorrelation
rho1 = gamma1 / gamma0
print(f"4. Autocorrelation: ρ(1) = γ(1)/γ(0) = {gamma1}/{gamma0:.2f} = {rho1:.4f}")
print(f"   ρ(h) = 0 for h > 1")

# 5. Invertibility
print(f"\n5. Invertibility: |θ| = |{theta}| = {abs(theta)} < 1")
print(f"   → YES, the process is invertible")
