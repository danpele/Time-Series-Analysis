"""
TSA_ch0_fundamentals
====================
Mathematical Fundamentals for Time Series

This script demonstrates:
- Basic statistical concepts
- Probability distributions
- Linear algebra essentials
- Lag operator basics

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Chart style settings - Nature journal quality
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75

def save_fig(name):
    """Save figure with transparent background."""
    plt.savefig(f'{name}.pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig(f'{name}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf")

print("=" * 70)
print("MATHEMATICAL FUNDAMENTALS FOR TIME SERIES")
print("=" * 70)

# =============================================================================
# 1. Probability Distributions
# =============================================================================
np.random.seed(42)

print("\n1. PROBABILITY DISTRIBUTIONS")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Normal distribution
x = np.linspace(-4, 4, 200)
axes[0, 0].plot(x, stats.norm.pdf(x), color='#1A3A6E', linewidth=2.5, label='N(0,1)')
axes[0, 0].fill_between(x, stats.norm.pdf(x), alpha=0.3, color='#1A3A6E')
axes[0, 0].set_title('Normal Distribution', fontweight='bold')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# Student-t distribution
for df in [3, 5, 30]:
    axes[0, 1].plot(x, stats.t.pdf(x, df), linewidth=2, label=f't(df={df})')
axes[0, 1].plot(x, stats.norm.pdf(x), 'k--', linewidth=1.5, label='Normal')
axes[0, 1].set_title('Student-t Distribution (Fat Tails)', fontweight='bold')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)

# Chi-square distribution
x_chi = np.linspace(0.01, 20, 200)
for df in [2, 4, 8]:
    axes[1, 0].plot(x_chi, stats.chi2.pdf(x_chi, df), linewidth=2, label=f'χ²({df})')
axes[1, 0].set_title('Chi-Square Distribution', fontweight='bold')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_xlim(0, 20)
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

# F distribution
x_f = np.linspace(0.01, 5, 200)
for d1, d2 in [(5, 10), (10, 20), (20, 30)]:
    axes[1, 1].plot(x_f, stats.f.pdf(x_f, d1, d2), linewidth=2, label=f'F({d1},{d2})')
axes[1, 1].set_title('F Distribution', fontweight='bold')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

plt.tight_layout()
save_fig('ch0_distributions')

# =============================================================================
# 2. Central Limit Theorem
# =============================================================================
print("\n2. CENTRAL LIMIT THEOREM")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Different sample sizes
sample_sizes = [1, 5, 30, 100]
n_simulations = 5000

for idx, n in enumerate(sample_sizes):
    ax = axes[idx // 2, idx % 2]

    # Sample means from uniform distribution
    means = [np.mean(np.random.uniform(0, 1, n)) for _ in range(n_simulations)]

    ax.hist(means, bins=50, density=True, color='#1A3A6E', alpha=0.7, edgecolor='white')

    # Overlay normal
    x = np.linspace(min(means), max(means), 100)
    theoretical_mean = 0.5
    theoretical_std = np.sqrt(1/12) / np.sqrt(n)
    ax.plot(x, stats.norm.pdf(x, theoretical_mean, theoretical_std),
            'r-', linewidth=2, label='Normal approx.')

    ax.set_title(f'Sample Size n = {n}', fontweight='bold')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Density')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plt.suptitle('Central Limit Theorem: Uniform → Normal', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
save_fig('ch0_clt')

# =============================================================================
# 3. Covariance and Correlation
# =============================================================================
print("\n3. COVARIANCE AND CORRELATION")
print("-" * 40)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

correlations = [0.9, 0.0, -0.8]
titles = ['Strong Positive (ρ=0.9)', 'No Correlation (ρ=0)', 'Strong Negative (ρ=-0.8)']

for idx, (rho, title) in enumerate(zip(correlations, titles)):
    # Generate correlated data
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    x, y = np.random.multivariate_normal(mean, cov, 200).T

    axes[idx].scatter(x, y, color='#1A3A6E', alpha=0.6, s=30)
    axes[idx].set_title(title, fontweight='bold')
    axes[idx].set_xlabel('X')
    axes[idx].set_ylabel('Y')

    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    axes[idx].plot(x_line, p(x_line), 'r-', linewidth=2, label='Regression line')
    axes[idx].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plt.tight_layout()
save_fig('ch0_correlation')

# =============================================================================
# 4. Expectation and Variance
# =============================================================================
print("\n4. EXPECTATION AND VARIANCE")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Different variances
x = np.linspace(-6, 6, 200)
variances = [0.5, 1, 2, 4]
colors = ['#1A3A6E', '#DC3545', '#2E7D32', '#E67E22']

for var, color in zip(variances, colors):
    axes[0].plot(x, stats.norm.pdf(x, 0, np.sqrt(var)),
                 linewidth=2, color=color, label=f'σ² = {var}')
axes[0].set_title('Effect of Variance on Normal Distribution', fontweight='bold')
axes[0].set_xlabel('x')
axes[0].set_ylabel('Density')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)

# Different means
means = [-2, 0, 2]
for mu, color in zip(means, colors[:3]):
    axes[1].plot(x, stats.norm.pdf(x, mu, 1), linewidth=2, color=color, label=f'μ = {mu}')
axes[1].set_title('Effect of Mean on Normal Distribution', fontweight='bold')
axes[1].set_xlabel('x')
axes[1].set_ylabel('Density')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

plt.tight_layout()
save_fig('ch0_mean_variance')

# =============================================================================
# 5. Lag Operator Visualization
# =============================================================================
print("\n5. LAG OPERATOR")
print("-" * 40)

n = 50
t = np.arange(n)
y = np.sin(2 * np.pi * t / 12) + 0.5 * np.random.randn(n)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Original series
axes[0, 0].plot(t, y, color='#1A3A6E', linewidth=2, marker='o', markersize=4, label='Yₜ')
axes[0, 0].set_title('Original Series Yₜ', fontweight='bold')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('Y')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# Lag 1
axes[0, 1].plot(t[1:], y[:-1], color='#DC3545', linewidth=2, marker='o', markersize=4, label='LYₜ = Yₜ₋₁')
axes[0, 1].plot(t, y, color='#1A3A6E', linewidth=1, alpha=0.3, label='Yₜ (reference)')
axes[0, 1].set_title('Lag Operator: LYₜ = Yₜ₋₁', fontweight='bold')
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('Y')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# First difference
diff_y = np.diff(y)
axes[1, 0].plot(t[1:], diff_y, color='#2E7D32', linewidth=2, marker='o', markersize=4, label='ΔYₜ = Yₜ - Yₜ₋₁')
axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=1)
axes[1, 0].set_title('Difference Operator: ΔYₜ = (1-L)Yₜ', fontweight='bold')
axes[1, 0].set_xlabel('t')
axes[1, 0].set_ylabel('ΔY')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# Second difference
diff2_y = np.diff(diff_y)
axes[1, 1].plot(t[2:], diff2_y, color='#E67E22', linewidth=2, marker='o', markersize=4, label='Δ²Yₜ')
axes[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
axes[1, 1].set_title('Second Difference: Δ²Yₜ = (1-L)²Yₜ', fontweight='bold')
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('Δ²Y')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plt.tight_layout()
save_fig('ch0_lag_operator')

# =============================================================================
# 6. Matrix Operations for Time Series
# =============================================================================
print("\n6. MATRIX OPERATIONS")
print("-" * 40)

# Companion matrix for AR(2)
phi1, phi2 = 0.6, -0.2

print(f"   AR(2) model: Yₜ = {phi1}Yₜ₋₁ + {phi2}Yₜ₋₂ + εₜ")
print(f"\n   Companion matrix:")
print(f"   F = | {phi1:5.2f}  {phi2:5.2f} |")
print(f"       | 1.00   0.00 |")

# Eigenvalues
F = np.array([[phi1, phi2], [1, 0]])
eigenvalues = np.linalg.eigvals(F)
print(f"\n   Eigenvalues: {eigenvalues}")
print(f"   |λ₁| = {abs(eigenvalues[0]):.4f}, |λ₂| = {abs(eigenvalues[1]):.4f}")

if all(abs(e) < 1 for e in eigenvalues):
    print("   → All eigenvalues inside unit circle: STATIONARY")
else:
    print("   → Eigenvalue(s) outside unit circle: NON-STATIONARY")

# Visualize unit circle
fig, ax = plt.subplots(figsize=(8, 8))

theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Unit circle')
ax.scatter(eigenvalues.real, eigenvalues.imag, color='#DC3545', s=150,
           zorder=5, label='Eigenvalues')

for i, ev in enumerate(eigenvalues):
    ax.annotate(f'λ{i+1}', (ev.real + 0.1, ev.imag + 0.1), fontsize=12)

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.set_title('Eigenvalues and Unit Circle (Stationarity Check)', fontweight='bold')
ax.set_xlabel('Real Part')
ax.set_ylabel('Imaginary Part')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
save_fig('ch0_eigenvalues')

print("\n" + "=" * 70)
print("FUNDAMENTALS ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch0_distributions.pdf: Probability distributions")
print("  - ch0_clt.pdf: Central limit theorem")
print("  - ch0_correlation.pdf: Correlation patterns")
print("  - ch0_mean_variance.pdf: Mean and variance effects")
print("  - ch0_lag_operator.pdf: Lag operator demonstration")
print("  - ch0_eigenvalues.pdf: Eigenvalues and stationarity")
