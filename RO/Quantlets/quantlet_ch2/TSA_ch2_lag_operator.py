"""
TSA_ch2_lag_operator
====================
The Lag Operator (Backshift Operator)

This script demonstrates:
- Lag operator L: LX_t = X_{t-1}
- Difference operator: (1-L)X_t = X_t - X_{t-1}
- Higher order differences: (1-L)^d
"""

import numpy as np
import matplotlib.pyplot as plt

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

n = 100
t = np.arange(n)

# Generate a sample time series (random walk with trend)
trend = 0.5 * t
noise = np.cumsum(np.random.normal(0, 1, n))
X = 100 + trend + noise

print("=" * 60)
print("THE LAG OPERATOR (BACKSHIFT OPERATOR)")
print("=" * 60)

print("""
Definition: L (or B) is the lag operator
  L X_t = X_{t-1}
  L^k X_t = X_{t-k}

Key Properties:
  L^0 = 1 (identity)
  L^a × L^b = L^{a+b}
  (L + L^2) X_t = X_{t-1} + X_{t-2}

Difference Operator:
  Δ = 1 - L
  ΔX_t = (1-L)X_t = X_t - X_{t-1}

Second Difference:
  Δ² = (1-L)² = 1 - 2L + L²
  Δ²X_t = X_t - 2X_{t-1} + X_{t-2}
""")

# Demonstrate operations
print("\n" + "=" * 60)
print("NUMERICAL EXAMPLE")
print("=" * 60)
print(f"\nOriginal series X: {X[:5].round(2)}")
print(f"                   X_1={X[0]:.2f}, X_2={X[1]:.2f}, X_3={X[2]:.2f}, ...")

# First difference
diff1 = np.diff(X)
print(f"\nFirst difference ΔX = (1-L)X:")
print(f"  ΔX_2 = X_2 - X_1 = {X[1]:.2f} - {X[0]:.2f} = {diff1[0]:.2f}")
print(f"  ΔX_3 = X_3 - X_2 = {X[2]:.2f} - {X[1]:.2f} = {diff1[1]:.2f}")

# Second difference
diff2 = np.diff(diff1)
print(f"\nSecond difference Δ²X = (1-L)²X:")
print(f"  Δ²X_3 = X_3 - 2X_2 + X_1")
print(f"        = {X[2]:.2f} - 2×{X[1]:.2f} + {X[0]:.2f}")
print(f"        = {X[2] - 2*X[1] + X[0]:.2f}")
print(f"  Verify: {diff2[0]:.2f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Original series
ax1 = axes[0, 0]
ax1.plot(X, 'b-', linewidth=1)
ax1.set_title('Original Series $X_t$', fontsize=12)
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.grid(True, alpha=0.3)

# First difference
ax2 = axes[0, 1]
ax2.plot(diff1, 'g-', linewidth=1)
ax2.axhline(y=0, color='red', linestyle='--')
ax2.axhline(y=np.mean(diff1), color='orange', linestyle='-', linewidth=2, label=f'Mean = {np.mean(diff1):.2f}')
ax2.set_title('First Difference $\\Delta X_t = (1-L)X_t$', fontsize=12)
ax2.set_xlabel('Time')
ax2.set_ylabel('Value')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax2.grid(True, alpha=0.3)

# Second difference
ax3 = axes[1, 0]
ax3.plot(diff2, 'r-', linewidth=1)
ax3.axhline(y=0, color='black', linestyle='--')
ax3.set_title('Second Difference $\\Delta^2 X_t = (1-L)^2 X_t$', fontsize=12)
ax3.set_xlabel('Time')
ax3.set_ylabel('Value')
ax3.grid(True, alpha=0.3)

# Summary
ax4 = axes[1, 1]
ax4.axis('off')
summary = """
Lag Operator Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Notation:
  L X_t = X_{t-1}     (one lag)
  L^k X_t = X_{t-k}   (k lags)

Difference Operator:
  Δ = (1 - L)
  ΔX_t = X_t - X_{t-1}

Second Difference:
  Δ² = (1-L)² = 1 - 2L + L²
  Δ²X_t = X_t - 2X_{t-1} + X_{t-2}

AR(p) in Lag Notation:
  φ(L)X_t = ε_t
  where φ(L) = 1 - φ₁L - φ₂L² - ... - φₚLᵖ

MA(q) in Lag Notation:
  X_t = θ(L)ε_t
  where θ(L) = 1 + θ₁L + θ₂L² + ... + θqLq

ARMA(p,q):
  φ(L)X_t = θ(L)ε_t
"""
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('../../charts/ch2_lag_operator.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch2_lag_operator.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("\n" + "=" * 60)
print("WHY LAG NOTATION IS USEFUL")
print("=" * 60)
print("""
1. Compact representation of complex models
2. Easy manipulation using algebra
3. Roots of lag polynomials determine stationarity/invertibility
4. Facilitates forecasting formulas
""")
