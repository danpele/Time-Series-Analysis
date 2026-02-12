"""
TSA_ch2_stationarity
====================
Visualize stationarity conditions for AR(2) processes.

Description:
- Plot AR(2) stationarity triangle
- Visualize unit circle condition
- Show characteristic roots
"""

import numpy as np
import matplotlib.pyplot as plt

# Create stationarity triangle
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Stationarity triangle
ax1 = axes[0]
phi1 = np.linspace(-2, 2, 400)

# Boundary conditions for AR(2) stationarity
# 1) phi1 + phi2 < 1  => phi2 < 1 - phi1
# 2) phi2 - phi1 < 1  => phi2 < 1 + phi1
# 3) |phi2| < 1       => -1 < phi2 < 1

# Fill stationary region
vertices = np.array([[-2, 1], [0, -1], [2, 1], [-2, 1]])
ax1.fill(vertices[:, 0], vertices[:, 1], alpha=0.3, color='green', label='Stationary Region')
ax1.plot([-2, 0], [1, -1], 'b-', linewidth=2, label=r'$\phi_2 - \phi_1 = 1$')
ax1.plot([0, 2], [-1, 1], 'r-', linewidth=2, label=r'$\phi_1 + \phi_2 = 1$')
ax1.axhline(y=1, color='purple', linestyle='--', linewidth=2, label=r'$\phi_2 = 1$')
ax1.axhline(y=-1, color='purple', linestyle='--', linewidth=2, label=r'$\phi_2 = -1$')
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlabel(r'$\phi_1$', fontsize=12)
ax1.set_ylabel(r'$\phi_2$', fontsize=12)
ax1.set_title('AR(2) Stationarity Triangle', fontsize=12)
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot 2: Unit circle
ax2 = axes[1]
theta = np.linspace(0, 2*np.pi, 100)
ax2.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Unit Circle')
ax2.fill(np.cos(theta), np.sin(theta), alpha=0.1, color='red')

# Example roots
roots_stationary = [0.4 + 0.5j, 0.4 - 0.5j]
roots_nonstationary = [0.8 + 0.4j, 0.8 - 0.4j]

for r in roots_stationary:
    ax2.plot(r.real, r.imag, 'go', markersize=10, label='Stationary (|z|<1)')
for r in roots_nonstationary:
    inv_r = 1/r
    ax2.plot(inv_r.real, inv_r.imag, 'b^', markersize=10, label='Char. root outside')

ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_xlabel('Real', fontsize=12)
ax2.set_ylabel('Imaginary', fontsize=12)
ax2.set_title('Unit Circle: Roots Must Be Outside', fontsize=12)
ax2.axhline(y=0, color='gray', linewidth=0.5)
ax2.axvline(x=0, color='gray', linewidth=0.5)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

plt.tight_layout()
plt.savefig('../../charts/ch2_ar2_stationarity.pdf', bbox_inches='tight')
plt.show()

print("AR(2) Stationarity Conditions:")
print("  1. phi1 + phi2 < 1")
print("  2. phi2 - phi1 < 1")
print("  3. |phi2| < 1")
