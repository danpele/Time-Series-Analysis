"""
TSA_ch7_cointegration
=====================
Cointegration and VECM Analysis

This script demonstrates:
- Spurious regression problem
- Cointegration concept
- Engle-Granger two-step method
- Johansen cointegration test
- Vector Error Correction Model (VECM)

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
from statsmodels.regression.linear_model import OLS
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
print("COINTEGRATION AND VECM ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. Spurious Regression Problem
# =============================================================================
np.random.seed(42)
n = 500

print("\n1. SPURIOUS REGRESSION PROBLEM")
print("-" * 40)

# Two independent random walks
rw1 = np.cumsum(np.random.normal(0, 1, n))
rw2 = np.cumsum(np.random.normal(0, 1, n))

# Regress rw1 on rw2 (should find no relationship)
X = np.column_stack([np.ones(n), rw2])
model_spurious = OLS(rw1, X).fit()

print("   Regressing two independent random walks:")
print(f"   R² = {model_spurious.rsquared:.4f} (spuriously high!)")
print(f"   t-statistic for slope: {model_spurious.tvalues[1]:.2f}")
print(f"   p-value: {model_spurious.pvalues[1]:.4f}")
print("\n   WARNING: High R² and significant t-stat are misleading!")
print("   These series are completely unrelated!")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(rw1, color='#1A3A6E', label='Series 1', linewidth=1)
axes[0].plot(rw2, color='#DC3545', label='Series 2', linewidth=1)
axes[0].set_title('Two Independent Random Walks', fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

axes[1].scatter(rw2, rw1, alpha=0.5, color='#1A3A6E', s=10)
axes[1].plot(rw2, model_spurious.fittedvalues, color='#DC3545', linewidth=2)
axes[1].set_xlabel('Series 2')
axes[1].set_ylabel('Series 1')
axes[1].set_title(f'Spurious Regression (R² = {model_spurious.rsquared:.3f})', fontweight='bold')

plt.tight_layout()
plt.savefig('ch7_spurious.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch7_spurious.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch7_spurious.pdf")

# =============================================================================
# 2. Cointegrated Series
# =============================================================================
print("\n2. COINTEGRATED SERIES")
print("-" * 40)

# Generate cointegrated series
# Y1 and Y2 share a common stochastic trend
common_trend = np.cumsum(np.random.normal(0, 1, n))
y1 = common_trend + np.random.normal(0, 0.5, n)
y2 = 0.5 * common_trend + np.random.normal(0, 0.5, n)

# The equilibrium relationship: Y1 - 2*Y2 should be stationary
equilibrium_error = y1 - 2 * y2

print("   Cointegrating relationship: Y1 = 2*Y2 + stationary error")
print(f"\n   Y1: ADF p-value = {adfuller(y1)[1]:.4f} (Non-stationary)")
print(f"   Y2: ADF p-value = {adfuller(y2)[1]:.4f} (Non-stationary)")
print(f"   Y1 - 2*Y2: ADF p-value = {adfuller(equilibrium_error)[1]:.4f} (Stationary!)")

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(y1, color='#1A3A6E', label='Y1', linewidth=1)
axes[0].plot(y2 * 2, color='#DC3545', label='2×Y2', linewidth=1, alpha=0.7)
axes[0].set_title('Cointegrated Series (Share Common Trend)', fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

axes[1].plot(equilibrium_error, color='#2E7D32', linewidth=1)
axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
axes[1].axhline(y=equilibrium_error.mean() + 2*equilibrium_error.std(), color='red', linestyle=':', alpha=0.5)
axes[1].axhline(y=equilibrium_error.mean() - 2*equilibrium_error.std(), color='red', linestyle=':', alpha=0.5)
axes[1].set_title('Equilibrium Error: Y1 - 2×Y2 (Stationary)', fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Error')

plt.tight_layout()
plt.savefig('ch7_cointegration.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch7_cointegration.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch7_cointegration.pdf")

# =============================================================================
# 3. Engle-Granger Two-Step Method
# =============================================================================
print("\n3. ENGLE-GRANGER TWO-STEP METHOD")
print("-" * 40)

# Step 1: Estimate cointegrating regression
X_coint = np.column_stack([np.ones(n), y2])
model_coint = OLS(y1, X_coint).fit()

print("   Step 1: Estimate Y1 = α + β×Y2 + u")
print(f"     Intercept (α): {model_coint.params[0]:.4f}")
print(f"     Coefficient (β): {model_coint.params[1]:.4f}")

# Step 2: Test residuals for stationarity
residuals = model_coint.resid
adf_resid = adfuller(residuals)

print("\n   Step 2: Test residuals for stationarity")
print(f"     ADF statistic: {adf_resid[0]:.4f}")
print(f"     Critical values: 1%: {adf_resid[4]['1%']:.4f}, 5%: {adf_resid[4]['5%']:.4f}")
print(f"     p-value: {adf_resid[1]:.4f}")

if adf_resid[1] < 0.05:
    print("\n   CONCLUSION: Series are cointegrated!")
else:
    print("\n   CONCLUSION: No cointegration found")

# Engle-Granger test (using statsmodels)
coint_stat, pvalue, crit_values = coint(y1, y2)
print(f"\n   Engle-Granger test:")
print(f"     Test statistic: {coint_stat:.4f}")
print(f"     p-value: {pvalue:.4f}")

# =============================================================================
# 4. Johansen Cointegration Test
# =============================================================================
print("\n4. JOHANSEN COINTEGRATION TEST")
print("-" * 40)

# Create DataFrame for Johansen test
df_coint = pd.DataFrame({'Y1': y1, 'Y2': y2})

# Johansen test
johansen_result = coint_johansen(df_coint, det_order=0, k_ar_diff=2)

print("\n   Trace Test:")
print(f"   {'r':>4} {'Trace Stat':>12} {'95% CV':>10} {'Conclusion':>15}")
print("   " + "-" * 45)
for i in range(2):
    trace_stat = johansen_result.lr1[i]
    cv_95 = johansen_result.cvt[i, 1]
    conclusion = "Reject H0" if trace_stat > cv_95 else "Fail to reject"
    print(f"   {i:>4} {trace_stat:>12.4f} {cv_95:>10.4f} {conclusion:>15}")

print("\n   Maximum Eigenvalue Test:")
print(f"   {'r':>4} {'Max Eig':>12} {'95% CV':>10} {'Conclusion':>15}")
print("   " + "-" * 45)
for i in range(2):
    max_eig = johansen_result.lr2[i]
    cv_95 = johansen_result.cvm[i, 1]
    conclusion = "Reject H0" if max_eig > cv_95 else "Fail to reject"
    print(f"   {i:>4} {max_eig:>12.4f} {cv_95:>10.4f} {conclusion:>15}")

print(f"\n   Cointegrating vector (normalized):")
print(f"     β = [{johansen_result.evec[0, 0]:.4f}, {johansen_result.evec[1, 0]:.4f}]")

# =============================================================================
# 5. Vector Error Correction Model (VECM)
# =============================================================================
print("\n5. VECTOR ERROR CORRECTION MODEL (VECM)")
print("-" * 40)

# Fit VECM
vecm = VECM(df_coint, k_ar_diff=2, coint_rank=1, deterministic='ci')
vecm_results = vecm.fit()

print("\n   VECM with 1 cointegrating relationship:")
print(f"\n   Error Correction Coefficients (α):")
print(f"     Y1 equation: α1 = {vecm_results.alpha[0, 0]:.4f}")
print(f"     Y2 equation: α2 = {vecm_results.alpha[1, 0]:.4f}")

print(f"\n   Interpretation:")
if vecm_results.alpha[0, 0] < 0:
    print(f"     Y1 adjusts {abs(vecm_results.alpha[0, 0])*100:.1f}% of disequilibrium per period")
if vecm_results.alpha[1, 0] > 0:
    print(f"     Y2 adjusts {abs(vecm_results.alpha[1, 0])*100:.1f}% of disequilibrium per period")

# Half-life of adjustment
if vecm_results.alpha[0, 0] < 0:
    half_life = np.log(0.5) / np.log(1 + vecm_results.alpha[0, 0])
    print(f"\n   Half-life of adjustment: {half_life:.1f} periods")

# =============================================================================
# 6. VECM Impulse Response Functions
# =============================================================================
print("\n6. VECM IMPULSE RESPONSE FUNCTIONS")
print("-" * 40)

irf = vecm_results.irf(periods=30)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# IRF plots
axes[0, 0].plot(irf.irfs[:, 0, 0], color='#1A3A6E', linewidth=2)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[0, 0].set_title('Y1 → Y1', fontweight='bold')
axes[0, 0].set_xlabel('Periods')

axes[0, 1].plot(irf.irfs[:, 0, 1], color='#DC3545', linewidth=2)
axes[0, 1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[0, 1].set_title('Y2 → Y1', fontweight='bold')
axes[0, 1].set_xlabel('Periods')

axes[1, 0].plot(irf.irfs[:, 1, 0], color='#1A3A6E', linewidth=2)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[1, 0].set_title('Y1 → Y2', fontweight='bold')
axes[1, 0].set_xlabel('Periods')

axes[1, 1].plot(irf.irfs[:, 1, 1], color='#DC3545', linewidth=2)
axes[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[1, 1].set_title('Y2 → Y2', fontweight='bold')
axes[1, 1].set_xlabel('Periods')

plt.suptitle('VECM Impulse Response Functions', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('ch7_vecm_irf.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch7_vecm_irf.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch7_vecm_irf.pdf")

# =============================================================================
# 7. Error Correction Visualization
# =============================================================================
print("\n7. ERROR CORRECTION MECHANISM")
print("-" * 40)

# Get the error correction term
ect = np.dot(df_coint.values, vecm_results.beta[:, 0])

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Error correction term
axes[0].plot(ect, color='#2E7D32', linewidth=1)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes[0].fill_between(range(len(ect)), 0, ect, where=ect > 0, alpha=0.3, color='green')
axes[0].fill_between(range(len(ect)), 0, ect, where=ect < 0, alpha=0.3, color='red')
axes[0].set_title('Error Correction Term (Deviation from Equilibrium)', fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('ECT')

# Adjustment process
axes[1].plot(np.diff(y1), color='#1A3A6E', alpha=0.7, linewidth=0.8, label='ΔY1')
axes[1].plot(np.diff(y2), color='#DC3545', alpha=0.7, linewidth=0.8, label='ΔY2')
axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[1].set_title('Changes in Series (Error Correction Adjustments)', fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('ΔY')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig('ch7_error_correction.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch7_error_correction.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch7_error_correction.pdf")

print("\n" + "=" * 70)
print("COINTEGRATION ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch7_spurious.pdf: Spurious regression example")
print("  - ch7_cointegration.pdf: Cointegrated series")
print("  - ch7_vecm_irf.pdf: VECM impulse responses")
print("  - ch7_error_correction.pdf: Error correction mechanism")
