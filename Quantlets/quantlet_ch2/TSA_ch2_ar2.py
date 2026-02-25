"""
TSA_ch2_ar2
===========
AR(2) Process: Stationarity and Characteristic Roots

This script demonstrates:
- AR(2) model: X_t = φ₁X_{t-1} + φ₂X_{t-2} + ε_t
- Characteristic equation and roots
- Stationarity triangle
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Set random seed
np.random.seed(42)

def simulate_ar2(n, phi1, phi2, sigma=1):
    """Simulate AR(2) process"""
    x = np.zeros(n)
    eps = np.random.normal(0, sigma, n)
    for t in range(2, n):
        x[t] = phi1 * x[t-1] + phi2 * x[t-2] + eps[t]
    return x

def check_ar2_stationarity(phi1, phi2):
    """Check AR(2) stationarity conditions"""
    # Conditions: |φ₂| < 1, φ₁ + φ₂ < 1, φ₂ - φ₁ < 1
    cond1 = abs(phi2) < 1
    cond2 = phi1 + phi2 < 1
    cond3 = phi2 - phi1 < 1
    return cond1 and cond2 and cond3

def get_ar2_roots(phi1, phi2):
    """Get characteristic roots of AR(2)"""
    # Characteristic equation: 1 - φ₁z - φ₂z² = 0
    # Rearranged: φ₂z² + φ₁z - 1 = 0
    # Multiply by -1: -φ₂z² - φ₁z + 1 = 0
    # Or solve z² - (φ₁/φ₂)z - 1/φ₂ = 0 when φ₂ ≠ 0
    if abs(phi2) < 1e-10:
        return np.array([1/phi1]) if abs(phi1) > 1e-10 else np.array([np.inf])
    coeffs = [phi2, phi1, -1]  # φ₂z² + φ₁z - 1 = 0
    roots = np.roots(coeffs)
    return roots

print("=" * 60)
print("AR(2) PROCESS: X_t = φ₁X_{t-1} + φ₂X_{t-2} + ε_t")
print("=" * 60)

print("""
Characteristic Equation:
  φ(z) = 1 - φ₁z - φ₂z² = 0

Stationarity: All roots must be OUTSIDE unit circle (|z| > 1)

Equivalent Conditions (Stationarity Triangle):
  1. |φ₂| < 1
  2. φ₁ + φ₂ < 1
  3. φ₂ - φ₁ < 1
""")

# Create figure
fig = plt.figure(figsize=(16, 10))

# Plot 1: Stationarity triangle
ax1 = fig.add_subplot(2, 2, 1)
# Triangle vertices
triangle = Polygon([(-2, 1), (2, 1), (0, -1)], closed=True,
                   fill=True, facecolor='lightgreen', edgecolor='black', alpha=0.3)
ax1.add_patch(triangle)
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-1.5, 1.5)
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax1.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

# Mark some points
points = [(0.5, 0.3, 'A'), (-0.5, 0.3, 'B'), (0.8, -0.5, 'C'), (1.5, 0.2, 'D')]
for phi1, phi2, label in points:
    is_stat = check_ar2_stationarity(phi1, phi2)
    color = 'green' if is_stat else 'red'
    ax1.scatter([phi1], [phi2], c=color, s=100, zorder=5)
    ax1.annotate(f'{label}({phi1},{phi2})', (phi1, phi2), xytext=(5, 5),
                 textcoords='offset points', fontsize=9)

ax1.set_xlabel('$\\phi_1$', fontsize=12)
ax1.set_ylabel('$\\phi_2$', fontsize=12)
ax1.set_title('AR(2) Stationarity Triangle', fontsize=12)
ax1.text(0, 0.5, 'STATIONARY', ha='center', fontsize=11, color='green', fontweight='bold')
ax1.grid(False)

# Plots 2-4: AR(2) simulations with different parameters
params = [
    (0.5, 0.3, 'Stationary (real roots)'),
    (1.0, -0.5, 'Stationary (complex roots, oscillating)'),
    (-0.5, 0.3, 'Stationary (negative φ₁)')
]

for idx, (phi1, phi2, title) in enumerate(params):
    ax = fig.add_subplot(2, 2, idx + 2)

    # Simulate
    x = simulate_ar2(500, phi1, phi2)

    # Get roots
    roots = get_ar2_roots(phi1, phi2)

    ax.plot(x[:200], 'b-', linewidth=0.8, alpha=0.8)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_title(f'AR(2): φ₁={phi1}, φ₂={phi2}\nRoots: {np.abs(roots).round(2)}', fontsize=10)
    ax.set_xlabel('Time')
    ax.set_ylabel('X_t')
    ax.grid(False)

    # Check if complex roots (pseudo-cyclical behavior)
    if np.any(np.iscomplex(roots)):
        ax.text(0.02, 0.98, 'Complex roots\n→ Pseudo-cycles', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('../../charts/ch2_ar2_stationarity.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch2_ar2_stationarity.pdf', bbox_inches='tight', transparent=True)
plt.show()

# Numerical example
print("\n" + "=" * 60)
print("NUMERICAL EXAMPLE: Finding Characteristic Roots")
print("=" * 60)

phi1, phi2 = 0.5, 0.3
print(f"\nGiven: φ₁ = {phi1}, φ₂ = {phi2}")
print(f"\nCharacteristic equation: 1 - {phi1}z - {phi2}z² = 0")
print(f"Rearranged: {phi2}z² + {phi1}z - 1 = 0")
print(f"\nUsing quadratic formula:")
print(f"z = (-{phi1} ± √({phi1}² + 4×{phi2})) / (2×{phi2})")

discriminant = phi1**2 + 4*phi2
print(f"  = (-{phi1} ± √{discriminant:.2f}) / {2*phi2}")

roots = get_ar2_roots(phi1, phi2)
print(f"\nRoots: z₁ = {roots[0]:.4f}, z₂ = {roots[1]:.4f}")
print(f"Moduli: |z₁| = {abs(roots[0]):.4f}, |z₂| = {abs(roots[1]):.4f}")

if all(abs(r) > 1 for r in roots):
    print("\nBoth |z| > 1 → STATIONARY ✓")
else:
    print("\nSome |z| ≤ 1 → NOT STATIONARY ✗")
