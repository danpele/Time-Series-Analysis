"""
TSA_ch2_python_diagnostics
==========================
Python Exercise 3: Diagnostic Checking

Tasks:
1. Fit an ARMA model
2. Perform comprehensive residual diagnostics
3. Ljung-Box test at multiple lags
4. Check normality of residuals
5. Interpret diagnostic results
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("=" * 60)
print("PYTHON EXERCISE: Diagnostic Checking")
print("=" * 60)

# Generate data from AR(2)
print("\n" + "-" * 60)
print("Step 1: Generate Data and Fit Model")
print("-" * 60)

n = 300
phi1 = 0.6
phi2 = -0.3
sigma = 1.5

# Generate AR(2)
eps = np.random.normal(0, sigma, n)
x = np.zeros(n)
for t in range(2, n):
    x[t] = phi1 * x[t-1] + phi2 * x[t-2] + eps[t]

# Add mean
x = x + 5

print(f"True model: AR(2) with φ₁={phi1}, φ₂={phi2}")
print(f"Sample size: n = {n}")

# Fit AR(2)
model = ARIMA(x, order=(2, 0, 0)).fit()
print(f"\nFitted model: AR(2)")
print(f"  φ̂₁ = {model.params[1]:.4f} (true: {phi1})")
print(f"  φ̂₂ = {model.params[2]:.4f} (true: {phi2})")

residuals = model.resid

# Step 2: Residual statistics
print("\n" + "-" * 60)
print("Step 2: Residual Statistics")
print("-" * 60)

print(f"\nResidual summary:")
print(f"  Mean:     {np.mean(residuals):.6f} (should be ≈ 0)")
print(f"  Std Dev:  {np.std(residuals):.4f} (true σ = {sigma})")
print(f"  Skewness: {stats.skew(residuals):.4f} (should be ≈ 0)")
print(f"  Kurtosis: {stats.kurtosis(residuals):.4f} (should be ≈ 0 for normal)")

# Step 3: Ljung-Box test
print("\n" + "-" * 60)
print("Step 3: Ljung-Box Test for Autocorrelation")
print("-" * 60)

lags_to_test = [5, 10, 15, 20]
lb_results = acorr_ljungbox(residuals, lags=lags_to_test, return_df=True)

print("\nH₀: Residuals are white noise (no autocorrelation)")
print("H₁: Residuals have autocorrelation")
print(f"\n{'Lag':<6} {'Q-Statistic':<15} {'p-value':<12} {'Decision':<15}")
print("-" * 50)

for lag in lags_to_test:
    q_stat = lb_results.loc[lag, 'lb_stat']
    p_val = lb_results.loc[lag, 'lb_pvalue']
    decision = "Fail to reject H₀" if p_val > 0.05 else "Reject H₀"
    print(f"{lag:<6} {q_stat:<15.4f} {p_val:<12.4f} {decision}")

if all(lb_results['lb_pvalue'] > 0.05):
    lb_conclusion = "✓ Residuals appear to be white noise"
else:
    lb_conclusion = "✗ Evidence of autocorrelation in residuals"
print(f"\nConclusion: {lb_conclusion}")

# Step 4: Normality tests
print("\n" + "-" * 60)
print("Step 4: Normality Tests")
print("-" * 60)

# Jarque-Bera test
jb_stat, jb_pval = stats.jarque_bera(residuals)
print(f"\nJarque-Bera Test:")
print(f"  H₀: Residuals are normally distributed")
print(f"  Test statistic: {jb_stat:.4f}")
print(f"  p-value: {jb_pval:.4f}")
print(f"  Decision: {'Fail to reject H₀ (Normal)' if jb_pval > 0.05 else 'Reject H₀ (Not normal)'}")

# Shapiro-Wilk test (for smaller samples)
sw_stat, sw_pval = stats.shapiro(residuals[:100])  # Use first 100 for Shapiro
print(f"\nShapiro-Wilk Test (first 100 obs):")
print(f"  Test statistic: {sw_stat:.4f}")
print(f"  p-value: {sw_pval:.4f}")
print(f"  Decision: {'Fail to reject H₀ (Normal)' if sw_pval > 0.05 else 'Reject H₀ (Not normal)'}")

# Step 5: Heteroskedasticity check (simple)
print("\n" + "-" * 60)
print("Step 5: Heteroskedasticity Check")
print("-" * 60)

# Split residuals into halves and compare variance
half = len(residuals) // 2
var_first = np.var(residuals[:half])
var_second = np.var(residuals[half:])
f_stat = var_first / var_second if var_first > var_second else var_second / var_first
f_pval = 2 * (1 - stats.f.cdf(f_stat, half-1, half-1))

print(f"\nVariance comparison (first vs second half):")
print(f"  Var(first half):  {var_first:.4f}")
print(f"  Var(second half): {var_second:.4f}")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  p-value: {f_pval:.4f}")
print(f"  Decision: {'No evidence of heteroskedasticity' if f_pval > 0.05 else 'Evidence of heteroskedasticity'}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Residual time series
ax1 = axes[0, 0]
ax1.plot(residuals, 'b-', linewidth=0.8, alpha=0.8)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax1.axhline(y=2*np.std(residuals), color='gray', linestyle=':', alpha=0.7)
ax1.axhline(y=-2*np.std(residuals), color='gray', linestyle=':', alpha=0.7)
ax1.set_title('Residuals Over Time', fontsize=12)
ax1.set_xlabel('Time')
ax1.set_ylabel('Residual')
ax1.grid(True, alpha=0.3)

# Plot 2: Residual ACF
ax2 = axes[0, 1]
plot_acf(residuals, lags=20, ax=ax2)
ax2.set_title('Residual ACF (should be insignificant)', fontsize=12)

# Plot 3: Residual PACF
ax3 = axes[0, 2]
plot_pacf(residuals, lags=20, ax=ax3, method='ywm')
ax3.set_title('Residual PACF (should be insignificant)', fontsize=12)

# Plot 4: Histogram with normal overlay
ax4 = axes[1, 0]
ax4.hist(residuals, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
x_range = np.linspace(residuals.min(), residuals.max(), 100)
ax4.plot(x_range, stats.norm.pdf(x_range, np.mean(residuals), np.std(residuals)),
         'r-', linewidth=2, label='Normal fit')
ax4.set_title('Residual Distribution', fontsize=12)
ax4.set_xlabel('Residual')
ax4.set_ylabel('Density')
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax4.grid(True, alpha=0.3)

# Plot 5: Q-Q plot
ax5 = axes[1, 1]
stats.probplot(residuals, dist="norm", plot=ax5)
ax5.set_title('Q-Q Plot (should follow diagonal)', fontsize=12)
ax5.grid(True, alpha=0.3)

# Plot 6: Summary
ax6 = axes[1, 2]
ax6.axis('off')

# Overall diagnostic assessment
all_checks_pass = (
    all(lb_results['lb_pvalue'] > 0.05) and
    jb_pval > 0.05 and
    f_pval > 0.05
)

summary = f"""
Diagnostic Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Model: AR(2) fitted to n = {n} obs

1. Autocorrelation (Ljung-Box):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Lag 10: Q = {lb_results.loc[10, 'lb_stat']:.2f}, p = {lb_results.loc[10, 'lb_pvalue']:.4f}
   {'✓ Pass' if lb_results.loc[10, 'lb_pvalue'] > 0.05 else '✗ Fail'}

2. Normality (Jarque-Bera):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   JB = {jb_stat:.2f}, p = {jb_pval:.4f}
   Skewness: {stats.skew(residuals):.3f}
   Kurtosis: {stats.kurtosis(residuals):.3f}
   {'✓ Pass' if jb_pval > 0.05 else '✗ Fail'}

3. Homoskedasticity:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   F = {f_stat:.2f}, p = {f_pval:.4f}
   {'✓ Pass' if f_pval > 0.05 else '✗ Fail'}

Overall Assessment:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{'✓ MODEL ADEQUATE' if all_checks_pass else '⚠ ISSUES DETECTED'}

{'Residuals behave as white noise.' if all_checks_pass else 'Consider model refinement.'}
"""
ax6.text(0.02, 0.98, summary, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen' if all_checks_pass else 'lightyellow', alpha=0.5))

plt.tight_layout()
plt.savefig('../../charts/ch2_python_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("DIAGNOSTIC CONCLUSIONS")
print("=" * 60)
print(f"""
1. Autocorrelation: {'✓ None detected' if all(lb_results['lb_pvalue'] > 0.05) else '✗ Present'}
2. Normality: {'✓ Normal' if jb_pval > 0.05 else '✗ Non-normal'}
3. Homoskedasticity: {'✓ Constant variance' if f_pval > 0.05 else '✗ Varying variance'}

Overall: {'Model is adequate' if all_checks_pass else 'Consider alternative models'}
""")

print("\nInterpretation Guide:")
print("-" * 40)
print("• If Ljung-Box fails: Add more AR/MA terms")
print("• If normality fails: Consider robust methods")
print("• If heteroskedasticity: Consider GARCH models")
