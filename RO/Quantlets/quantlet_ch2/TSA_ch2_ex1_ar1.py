"""
TSA_ch2_ex1_ar1
===============
Seminar Exercise 1: AR(1) Properties

Problem: X_t = 2 + 0.7*X_{t-1} + ε_t, ε_t ~ WN(0, 9)
Calculate: mean, variance, autocovariance, autocorrelation
"""

import numpy as np

print("=" * 60)
print("EXERCISE 1: AR(1) PROPERTIES")
print("=" * 60)

# Given parameters
c = 2
phi = 0.7
sigma2 = 9

print(f"\nGiven: X_t = {c} + {phi}*X_{{t-1}} + ε_t, σ² = {sigma2}")
print(f"\nStationarity check: |φ| = |{phi}| = {abs(phi)} < 1 ✓")

# 1. Mean
mu = c / (1 - phi)
print(f"\n1. Mean: μ = c/(1-φ) = {c}/{1-phi:.1f} = {mu:.4f}")

# 2. Variance
gamma0 = sigma2 / (1 - phi**2)
print(f"2. Variance: γ(0) = σ²/(1-φ²) = {sigma2}/{1-phi**2:.2f} = {gamma0:.4f}")

# 3. Autocovariance
gamma1 = phi * gamma0
gamma2 = phi**2 * gamma0
print(f"3. Autocovariance:")
print(f"   γ(1) = φ·γ(0) = {phi} × {gamma0:.2f} = {gamma1:.4f}")
print(f"   γ(2) = φ²·γ(0) = {phi**2:.2f} × {gamma0:.2f} = {gamma2:.4f}")

# 4. Autocorrelation
rho1 = phi
rho2 = phi**2
print(f"4. Autocorrelation:")
print(f"   ρ(1) = φ = {rho1}")
print(f"   ρ(2) = φ² = {rho2:.2f}")

print(f"\nGeneral formula: ρ(h) = φ^h = {phi}^h (exponential decay)")
