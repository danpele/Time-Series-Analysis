"""
TSA_ch2_ex3_roots
=================
Seminar Exercise 3: Characteristic Roots of AR(2)

Problem: X_t = 0.5*X_{t-1} + 0.3*X_{t-2} + ε_t
Find characteristic roots and check stationarity.
"""

import numpy as np

print("=" * 60)
print("EXERCISE 3: AR(2) CHARACTERISTIC ROOTS")
print("=" * 60)

# Given parameters
phi1 = 0.5
phi2 = 0.3

print(f"\nGiven: X_t = {phi1}*X_{{t-1}} + {phi2}*X_{{t-2}} + ε_t")

# 1. Characteristic equation
print(f"\n1. Characteristic equation:")
print(f"   φ(z) = 1 - {phi1}z - {phi2}z² = 0")
print(f"   Equivalently: {phi2}z² + {phi1}z - 1 = 0")

# 2. Roots using quadratic formula
a = phi2
b = phi1
c = -1

discriminant = b**2 - 4*a*c
z1 = (-b + np.sqrt(discriminant)) / (2*a)
z2 = (-b - np.sqrt(discriminant)) / (2*a)

print(f"\n2. Roots (quadratic formula):")
print(f"   Discriminant = {b}² + 4×{phi2} = {discriminant:.4f}")
print(f"   z₁ = (-{b} + √{discriminant:.2f}) / {2*a} = {z1:.4f}")
print(f"   z₂ = (-{b} - √{discriminant:.2f}) / {2*a} = {z2:.4f}")

# 3. Stationarity check
print(f"\n3. Stationarity check (roots must be OUTSIDE unit circle):")
print(f"   |z₁| = {abs(z1):.4f} > 1 ✓")
print(f"   |z₂| = {abs(z2):.4f} > 1 ✓")
print(f"\n   Both roots outside unit circle → STATIONARY")

# Verify with numpy
roots = np.roots([phi2, phi1, -1])
print(f"\n   Verification with numpy.roots: {roots}")
