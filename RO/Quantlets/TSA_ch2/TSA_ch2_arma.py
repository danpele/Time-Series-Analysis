"""
TSA_ch2_arma
============
ARMA Models

This script demonstrates:
- AR(p) model simulation and properties
- MA(q) model simulation and properties
- ARMA(p,q) combined models
- Model identification via ACF/PACF
- Model estimation and diagnostics

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
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
    plt.savefig(f'{name}.pdf', bbox_inches='tight', transparent=True)
    plt.close()
    print(f"   Saved: {name}.pdf")

print("=" * 70)
print("ARMA MODELS")
print("=" * 70)

np.random.seed(42)
n = 300

# =============================================================================
# 1. AR(1) Model
# =============================================================================
print("\n1. AR(1) MODEL")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

phi_values = [0.9, 0.5, -0.7]
colors = ['#1A3A6E', '#2E7D32', '#DC3545']

for idx, (phi, color) in enumerate(zip(phi_values, colors)):
    # Simulate AR(1)
    ar = np.array([1, -phi])  # 1 - phi*L
    ma = np.array([1])
    ar_process = ArmaProcess(ar, ma)
    y = ar_process.generate_sample(nsample=n)

    # Time series plot
    axes[0, idx].plot(y, color=color, linewidth=0.8, label=f'φ = {phi}')
    axes[0, idx].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    axes[0, idx].set_title(f'AR(1): φ = {phi}', fontweight='bold')
    axes[0, idx].set_xlabel('Time')
    axes[0, idx].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

    # ACF
    plot_acf(y, lags=20, ax=axes[1, idx], color=color, title='')
    axes[1, idx].set_title(f'ACF: φ = {phi}', fontweight='bold')

plt.suptitle('AR(1) Process: Yₜ = φYₜ₋₁ + εₜ', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
save_fig('ch2_ar1')

print(f"   AR(1) properties:")
print(f"   - Mean: E[Yₜ] = 0 (when μ = 0)")
print(f"   - Variance: Var(Yₜ) = σ²/(1-φ²)")
print(f"   - ACF: ρₖ = φᵏ (geometric decay)")
print(f"   - Stationarity: |φ| < 1")

# =============================================================================
# 2. AR(2) Model and Stationarity Region
# =============================================================================
print("\n2. AR(2) STATIONARITY REGION")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Stationarity triangle
phi1_range = np.linspace(-2.5, 2.5, 500)
phi2_range = np.linspace(-1.5, 1.5, 500)
PHI1, PHI2 = np.meshgrid(phi1_range, phi2_range)

# Stationarity conditions for AR(2)
cond1 = PHI2 < 1 - PHI1
cond2 = PHI2 < 1 + PHI1
cond3 = PHI2 > -1
stationary_region = cond1 & cond2 & cond3

axes[0].contourf(PHI1, PHI2, stationary_region.astype(int), levels=[0.5, 1.5],
                  colors=['#1A3A6E'], alpha=0.3)
axes[0].contour(PHI1, PHI2, stationary_region.astype(int), levels=[0.5],
                colors=['#1A3A6E'], linewidths=2)

# Mark example points
points = [(0.5, 0.3, 'A', 'Stationary'), (1.5, 0.2, 'B', 'Non-stationary'),
          (0.3, -0.8, 'C', 'Stationary')]
for phi1, phi2, label, status in points:
    color = '#2E7D32' if 'Stationary' == status else '#DC3545'
    axes[0].scatter(phi1, phi2, color=color, s=100, zorder=5)
    axes[0].annotate(f'{label}: ({phi1}, {phi2})', (phi1+0.1, phi2+0.1), fontsize=10)

axes[0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
axes[0].axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
axes[0].set_xlabel('φ₁')
axes[0].set_ylabel('φ₂')
axes[0].set_title('AR(2) Stationarity Region', fontweight='bold')
axes[0].set_xlim(-2.5, 2.5)
axes[0].set_ylim(-1.5, 1.5)

# Simulate AR(2) examples
phi1, phi2 = 0.5, 0.3
ar = np.array([1, -phi1, -phi2])
ma = np.array([1])
ar2_process = ArmaProcess(ar, ma)
y_ar2 = ar2_process.generate_sample(nsample=n)

axes[1].plot(y_ar2, color='#1A3A6E', linewidth=0.8, label=f'AR(2): φ₁={phi1}, φ₂={phi2}')
axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[1].set_title(f'AR(2) Process: Yₜ = {phi1}Yₜ₋₁ + {phi2}Yₜ₋₂ + εₜ', fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plt.tight_layout()
save_fig('ch2_ar2_stationarity')

# =============================================================================
# 3. MA(1) Model
# =============================================================================
print("\n3. MA(1) MODEL")
print("-" * 40)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

theta_values = [0.9, 0.5, -0.7]

for idx, (theta, color) in enumerate(zip(theta_values, colors)):
    # Simulate MA(1)
    ar = np.array([1])
    ma = np.array([1, theta])  # 1 + theta*L
    ma_process = ArmaProcess(ar, ma)
    y = ma_process.generate_sample(nsample=n)

    # Time series plot
    axes[0, idx].plot(y, color=color, linewidth=0.8, label=f'θ = {theta}')
    axes[0, idx].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    axes[0, idx].set_title(f'MA(1): θ = {theta}', fontweight='bold')
    axes[0, idx].set_xlabel('Time')
    axes[0, idx].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

    # ACF
    plot_acf(y, lags=20, ax=axes[1, idx], color=color, title='')
    axes[1, idx].set_title(f'ACF: θ = {theta}', fontweight='bold')

plt.suptitle('MA(1) Process: Yₜ = εₜ + θεₜ₋₁', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
save_fig('ch2_ma1')

print(f"   MA(1) properties:")
print(f"   - Always stationary")
print(f"   - Mean: E[Yₜ] = 0")
print(f"   - Variance: Var(Yₜ) = σ²(1+θ²)")
print(f"   - ACF: ρ₁ = θ/(1+θ²), ρₖ = 0 for k > 1")
print(f"   - Invertibility: |θ| < 1")

# =============================================================================
# 4. ACF/PACF Patterns for Model Identification
# =============================================================================
print("\n4. ACF/PACF PATTERNS")
print("-" * 40)

fig, axes = plt.subplots(3, 3, figsize=(14, 10))

# AR(1) φ=0.8
ar = np.array([1, -0.8])
ma = np.array([1])
y_ar1 = ArmaProcess(ar, ma).generate_sample(nsample=500)

axes[0, 0].plot(y_ar1[:100], color='#1A3A6E', linewidth=1, label='AR(1)')
axes[0, 0].set_title('AR(1): φ = 0.8', fontweight='bold')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plot_acf(y_ar1, lags=15, ax=axes[0, 1], color='#1A3A6E', title='')
axes[0, 1].set_title('ACF: Decays', fontweight='bold')

plot_pacf(y_ar1, lags=15, ax=axes[0, 2], color='#1A3A6E', title='')
axes[0, 2].set_title('PACF: Cuts off at 1', fontweight='bold')

# MA(1) θ=0.8
ar = np.array([1])
ma = np.array([1, 0.8])
y_ma1 = ArmaProcess(ar, ma).generate_sample(nsample=500)

axes[1, 0].plot(y_ma1[:100], color='#DC3545', linewidth=1, label='MA(1)')
axes[1, 0].set_title('MA(1): θ = 0.8', fontweight='bold')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plot_acf(y_ma1, lags=15, ax=axes[1, 1], color='#DC3545', title='')
axes[1, 1].set_title('ACF: Cuts off at 1', fontweight='bold')

plot_pacf(y_ma1, lags=15, ax=axes[1, 2], color='#DC3545', title='')
axes[1, 2].set_title('PACF: Decays', fontweight='bold')

# ARMA(1,1)
ar = np.array([1, -0.7])
ma = np.array([1, 0.5])
y_arma = ArmaProcess(ar, ma).generate_sample(nsample=500)

axes[2, 0].plot(y_arma[:100], color='#2E7D32', linewidth=1, label='ARMA(1,1)')
axes[2, 0].set_title('ARMA(1,1): φ=0.7, θ=0.5', fontweight='bold')
axes[2, 0].set_xlabel('Time')
axes[2, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plot_acf(y_arma, lags=15, ax=axes[2, 1], color='#2E7D32', title='')
axes[2, 1].set_title('ACF: Both decay', fontweight='bold')

plot_pacf(y_arma, lags=15, ax=axes[2, 2], color='#2E7D32', title='')
axes[2, 2].set_title('PACF: Both decay', fontweight='bold')

plt.tight_layout()
save_fig('ch2_identification')

print("\n   Model Identification Rules:")
print("   ┌─────────────┬───────────────┬───────────────┐")
print("   │   Model     │     ACF       │     PACF      │")
print("   ├─────────────┼───────────────┼───────────────┤")
print("   │   AR(p)     │   Decays      │ Cuts off at p │")
print("   │   MA(q)     │ Cuts off at q │   Decays      │")
print("   │  ARMA(p,q)  │   Decays      │   Decays      │")
print("   └─────────────┴───────────────┴───────────────┘")

# =============================================================================
# 5. Model Estimation
# =============================================================================
print("\n5. MODEL ESTIMATION")
print("-" * 40)

# Generate ARMA(1,1) data
ar = np.array([1, -0.7])
ma = np.array([1, 0.4])
true_process = ArmaProcess(ar, ma)
y = true_process.generate_sample(nsample=300)

print("   True model: ARMA(1,1) with φ=0.7, θ=0.4")

# Fit different models
models = {
    'AR(1)': (1, 0, 0),
    'AR(2)': (2, 0, 0),
    'MA(1)': (0, 0, 1),
    'MA(2)': (0, 0, 2),
    'ARMA(1,1)': (1, 0, 1),
    'ARMA(2,1)': (2, 0, 1),
}

results = []
for name, order in models.items():
    try:
        model = ARIMA(y, order=order)
        fit = model.fit()
        results.append({
            'Model': name,
            'AIC': fit.aic,
            'BIC': fit.bic,
            'LogL': fit.llf
        })
    except:
        pass

results_df = pd.DataFrame(results)
print(f"\n   Model Comparison:")
print(results_df.to_string(index=False))

best_model = results_df.loc[results_df['AIC'].idxmin(), 'Model']
print(f"\n   Best model (AIC): {best_model}")

# Fit best model and show parameters
best_fit = ARIMA(y, order=(1, 0, 1)).fit()
print(f"\n   ARMA(1,1) Estimated Parameters:")
print(f"   φ (AR): {best_fit.params[1]:.4f} (true: 0.7)")
print(f"   θ (MA): {best_fit.params[2]:.4f} (true: 0.4)")

# =============================================================================
# 6. Model Diagnostics
# =============================================================================
print("\n6. MODEL DIAGNOSTICS")
print("-" * 40)

residuals = best_fit.resid

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Residuals time plot
axes[0, 0].plot(residuals, color='#1A3A6E', linewidth=0.8, label='Residuals')
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0, 0].axhline(y=2*residuals.std(), color='gray', linestyle=':', alpha=0.5)
axes[0, 0].axhline(y=-2*residuals.std(), color='gray', linestyle=':', alpha=0.5)
axes[0, 0].set_title('Residuals', fontweight='bold')
axes[0, 0].set_xlabel('Time')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# ACF of residuals
plot_acf(residuals, lags=20, ax=axes[0, 1], color='#1A3A6E', title='')
axes[0, 1].set_title('ACF of Residuals', fontweight='bold')

# Histogram
axes[1, 0].hist(residuals, bins=30, density=True, color='#1A3A6E', alpha=0.7, edgecolor='white', label='Residuals')
x = np.linspace(residuals.min(), residuals.max(), 100)
from scipy import stats
axes[1, 0].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r-', linewidth=2, label='Normal')
axes[1, 0].set_title('Residual Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Residual')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
axes[1, 1].get_lines()[0].set_color('#1A3A6E')
axes[1, 1].get_lines()[1].set_color('#DC3545')

plt.tight_layout()
save_fig('ch2_diagnostics')

# Ljung-Box test
lb_test = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print("\n   Ljung-Box Test:")
print(f"   Lag 10: Q = {lb_test['lb_stat'].iloc[0]:.2f}, p-value = {lb_test['lb_pvalue'].iloc[0]:.4f}")
print(f"   Lag 20: Q = {lb_test['lb_stat'].iloc[1]:.2f}, p-value = {lb_test['lb_pvalue'].iloc[1]:.4f}")

if lb_test['lb_pvalue'].iloc[0] > 0.05:
    print("   → No significant autocorrelation in residuals (model adequate)")

print("\n" + "=" * 70)
print("ARMA ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch2_ar1.pdf: AR(1) process examples")
print("  - ch2_ar2_stationarity.pdf: AR(2) stationarity region")
print("  - ch2_ma1.pdf: MA(1) process examples")
print("  - ch2_identification.pdf: ACF/PACF patterns")
print("  - ch2_diagnostics.pdf: Model diagnostics")
