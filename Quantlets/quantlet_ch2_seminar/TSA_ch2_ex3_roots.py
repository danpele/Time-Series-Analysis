"""
TSA_ch2_ex3_roots
=================
Seminar Exercise 3: AR(2) Characteristic Roots

Problem: X_t = 0.5X_{t-1} + 0.3X_{t-2} + ε_t
Find characteristic equation, roots, and check stationarity
"""

import numpy as np
import matplotlib.pyplot as plt

# Given parameters
phi1 = 0.5
phi2 = 0.3

print("=" * 60)
print("AR(2) CHARACTERISTIC ROOTS - Step by Step Solution")
print("=" * 60)
print(f"\nModel: X_t = {phi1}X_{{t-1}} + {phi2}X_{{t-2}} + ε_t")

# 1. Characteristic equation
print("\n" + "-" * 60)
print("1. Write the Characteristic Equation")
print("-" * 60)
print("\nThe AR(2) model in lag notation:")
print("  (1 - φ₁L - φ₂L²)X_t = ε_t")
print(f"  (1 - {phi1}L - {phi2}L²)X_t = ε_t")
print()
print("Characteristic polynomial (setting z = L):")
print("  φ(z) = 1 - φ₁z - φ₂z² = 0")
print(f"  φ(z) = 1 - {phi1}z - {phi2}z² = 0")
print()
print("Rearranging to standard quadratic form:")
print(f"  {phi2}z² + {phi1}z - 1 = 0")

# 2. Find roots using quadratic formula
print("\n" + "-" * 60)
print("2. Find the Characteristic Roots")
print("-" * 60)
print("\nUsing quadratic formula: az² + bz + c = 0")
print(f"  a = {phi2}, b = {phi1}, c = -1")
print()
print("  z = (-b ± √(b² - 4ac)) / (2a)")
print(f"  z = (-{phi1} ± √({phi1}² - 4×{phi2}×(-1))) / (2×{phi2})")
print(f"  z = (-{phi1} ± √({phi1**2} + {4*phi2})) / {2*phi2}")

discriminant = phi1**2 + 4*phi2
print(f"  z = (-{phi1} ± √{discriminant:.3f}) / {2*phi2}")

sqrt_disc = np.sqrt(discriminant)
print(f"  z = (-{phi1} ± {sqrt_disc:.4f}) / {2*phi2}")

z1 = (-phi1 + sqrt_disc) / (2*phi2)
z2 = (-phi1 - sqrt_disc) / (2*phi2)

print()
print(f"  z₁ = (-{phi1} + {sqrt_disc:.4f}) / {2*phi2}")
print(f"     = {-phi1 + sqrt_disc:.4f} / {2*phi2}")
print(f"     = {z1:.4f}")
print()
print(f"  z₂ = (-{phi1} - {sqrt_disc:.4f}) / {2*phi2}")
print(f"     = {-phi1 - sqrt_disc:.4f} / {2*phi2}")
print(f"     = {z2:.4f}")

# 3. Check stationarity
print("\n" + "-" * 60)
print("3. Check Stationarity")
print("-" * 60)
print("\nStationarity condition: ALL roots must be OUTSIDE unit circle")
print("  i.e., |z| > 1 for all roots")
print()
print(f"  |z₁| = |{z1:.4f}| = {abs(z1):.4f}")
print(f"  |z₂| = |{z2:.4f}| = {abs(z2):.4f}")
print()

if abs(z1) > 1 and abs(z2) > 1:
    print(f"  Both |z₁| = {abs(z1):.4f} > 1 ✓")
    print(f"       |z₂| = {abs(z2):.4f} > 1 ✓")
    print("\n→ STATIONARY (all roots outside unit circle)")
    stationary = True
else:
    print(f"  Some root has |z| ≤ 1 ✗")
    print("\n→ NOT STATIONARY")
    stationary = False

# Alternative stationarity conditions
print("\n" + "-" * 60)
print("Alternative: Stationarity Triangle Conditions")
print("-" * 60)
print("\nFor AR(2), stationarity requires:")
print("  1. |φ₂| < 1")
print("  2. φ₁ + φ₂ < 1")
print("  3. φ₂ - φ₁ < 1")
print()
cond1 = abs(phi2) < 1
cond2 = phi1 + phi2 < 1
cond3 = phi2 - phi1 < 1
print(f"  1. |{phi2}| < 1 → {abs(phi2)} < 1 → {'✓' if cond1 else '✗'}")
print(f"  2. {phi1} + {phi2} < 1 → {phi1 + phi2} < 1 → {'✓' if cond2 else '✗'}")
print(f"  3. {phi2} - {phi1} < 1 → {phi2 - phi1} < 1 → {'✓' if cond3 else '✗'}")
print()
if cond1 and cond2 and cond3:
    print("All conditions satisfied → STATIONARY ✓")
else:
    print("Some condition violated → NOT STATIONARY")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Unit circle with roots
ax1 = axes[0]
theta_circle = np.linspace(0, 2*np.pi, 100)
ax1.plot(np.cos(theta_circle), np.sin(theta_circle), 'b-', linewidth=2, label='Unit Circle')
ax1.axhline(y=0, color='gray', linewidth=0.5)
ax1.axvline(x=0, color='gray', linewidth=0.5)

# Plot roots
ax1.scatter([z1, z2], [0, 0], s=200, c='red', marker='x', linewidths=3, label=f'Roots: z₁={z1:.2f}, z₂={z2:.2f}')
ax1.scatter([1/z1, 1/z2], [0, 0], s=100, c='green', marker='o', label='Inverse roots')

ax1.set_xlim(-3, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')
ax1.set_title('Characteristic Roots\n(Must be OUTSIDE unit circle)', fontsize=11)
ax1.set_xlabel('Real')
ax1.set_ylabel('Imaginary')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=9)
ax1.grid(True, alpha=0.3)

# Stationarity triangle
ax2 = axes[1]
from matplotlib.patches import Polygon
triangle = Polygon([(-2, 1), (2, 1), (0, -1)], closed=True,
                   fill=True, facecolor='lightgreen', edgecolor='black', alpha=0.3)
ax2.add_patch(triangle)
ax2.scatter([phi1], [phi2], s=200, c='red', marker='*', zorder=5, label=f'(φ₁,φ₂) = ({phi1},{phi2})')
ax2.set_xlim(-2.5, 2.5)
ax2.set_ylim(-1.5, 1.5)
ax2.axhline(y=0, color='gray', linewidth=0.5)
ax2.axvline(x=0, color='gray', linewidth=0.5)
ax2.set_xlabel('$\\phi_1$', fontsize=12)
ax2.set_ylabel('$\\phi_2$', fontsize=12)
ax2.set_title('Stationarity Triangle\n(Point must be INSIDE)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.text(0, 0.5, 'STATIONARY\nREGION', ha='center', fontsize=10, color='green')

# Summary
ax3 = axes[2]
ax3.axis('off')
summary = f"""
AR(2) Characteristic Roots Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: X_t = {phi1}X_{{t-1}} + {phi2}X_{{t-2}} + ε_t

1. Characteristic Equation:
   1 - {phi1}z - {phi2}z² = 0
   Or: {phi2}z² + {phi1}z - 1 = 0

2. Roots:
   z₁ = {z1:.4f}
   z₂ = {z2:.4f}

3. Moduli:
   |z₁| = {abs(z1):.4f} {'> 1 ✓' if abs(z1) > 1 else '≤ 1 ✗'}
   |z₂| = {abs(z2):.4f} {'> 1 ✓' if abs(z2) > 1 else '≤ 1 ✗'}

4. Conclusion:
   {'STATIONARY' if stationary else 'NOT STATIONARY'}
   (Roots {'outside' if stationary else 'inside'} unit circle)
"""
ax3.text(0.05, 0.95, summary, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('../../charts/ch2_seminar_ex3_roots.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("FINAL ANSWERS:")
print("=" * 60)
print(f"1. Characteristic equation: 1 - {phi1}z - {phi2}z² = 0")
print(f"   Or: {phi2}z² + {phi1}z - 1 = 0")
print(f"2. Roots: z₁ = {z1:.4f}, z₂ = {z2:.4f}")
print(f"3. Stationarity: {'YES' if stationary else 'NO'} (|z₁|={abs(z1):.2f}>1, |z₂|={abs(z2):.2f}>1)")
