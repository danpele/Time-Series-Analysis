"""
TSA_ch2_diagnostics
===================
Residual Diagnostics and Ljung-Box Test

This script demonstrates:
- Residual analysis for ARMA models
- Ljung-Box test for autocorrelation
- Q-Q plots for normality
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

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

n = 300

print("=" * 60)
print("RESIDUAL DIAGNOSTICS FOR ARMA MODELS")
print("=" * 60)

print("""
After fitting an ARMA model, check residuals:

1. No Autocorrelation (Ljung-Box test)
   H₀: Residuals are white noise
   If p-value < 0.05: model inadequate

2. Zero Mean
   Residuals should average around zero

3. Constant Variance (Homoscedasticity)
   No patterns in residual magnitude over time

4. Normality (for inference)
   Q-Q plot should be approximately linear
""")

# Generate ARMA(1,1) data
phi_true, theta_true = 0.7, 0.4
ar = np.array([1, -phi_true])
ma = np.array([1, theta_true])
arma_process = ArmaProcess(ar, ma)
data = arma_process.generate_sample(nsample=n)

# Fit correct model (ARMA(1,1)) and wrong model (AR(1))
model_correct = ARIMA(data, order=(1, 0, 1)).fit()
model_wrong = ARIMA(data, order=(1, 0, 0)).fit()

resid_correct = model_correct.resid
resid_wrong = model_wrong.resid

# Ljung-Box tests
lb_correct = acorr_ljungbox(resid_correct, lags=[10, 20], return_df=True)
lb_wrong = acorr_ljungbox(resid_wrong, lags=[10, 20], return_df=True)

print("\n" + "=" * 60)
print("LJUNG-BOX TEST RESULTS")
print("=" * 60)
print("\nCorrect Model: ARMA(1,1)")
print(f"  Lag 10: Q = {lb_correct.loc[10, 'lb_stat']:.2f}, p-value = {lb_correct.loc[10, 'lb_pvalue']:.4f}")
print(f"  Lag 20: Q = {lb_correct.loc[20, 'lb_stat']:.2f}, p-value = {lb_correct.loc[20, 'lb_pvalue']:.4f}")
if lb_correct.loc[20, 'lb_pvalue'] > 0.05:
    print("  → Fail to reject H₀: Residuals consistent with white noise ✓")
else:
    print("  → Reject H₀: Significant autocorrelation in residuals ✗")

print("\nWrong Model: AR(1)")
print(f"  Lag 10: Q = {lb_wrong.loc[10, 'lb_stat']:.2f}, p-value = {lb_wrong.loc[10, 'lb_pvalue']:.4f}")
print(f"  Lag 20: Q = {lb_wrong.loc[20, 'lb_stat']:.2f}, p-value = {lb_wrong.loc[20, 'lb_pvalue']:.4f}")
if lb_wrong.loc[20, 'lb_pvalue'] > 0.05:
    print("  → Fail to reject H₀: Residuals consistent with white noise ✓")
else:
    print("  → Reject H₀: Significant autocorrelation in residuals ✗")

# Create diagnostic plots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Row 1: Correct model (ARMA(1,1))
axes[0, 0].plot(resid_correct, 'b-', linewidth=0.5, alpha=0.8)
axes[0, 0].axhline(y=0, color='red', linestyle='--')
axes[0, 0].set_title('ARMA(1,1) Residuals', fontsize=11)
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Residual')
axes[0, 0].grid(True, alpha=0.3)

plot_acf(resid_correct, ax=axes[0, 1], lags=20, alpha=0.05)
axes[0, 1].set_title('ARMA(1,1) Residual ACF', fontsize=11)

stats.probplot(resid_correct, dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('ARMA(1,1) Q-Q Plot', fontsize=11)
axes[0, 2].grid(True, alpha=0.3)

# Residual histogram
axes[0, 3].hist(resid_correct, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
x = np.linspace(resid_correct.min(), resid_correct.max(), 100)
axes[0, 3].plot(x, stats.norm.pdf(x, resid_correct.mean(), resid_correct.std()), 'r-', linewidth=2)
axes[0, 3].set_title('ARMA(1,1) Residual Distribution', fontsize=11)
axes[0, 3].set_xlabel('Residual')
axes[0, 3].set_ylabel('Density')

# Row 2: Wrong model (AR(1))
axes[1, 0].plot(resid_wrong, 'r-', linewidth=0.5, alpha=0.8)
axes[1, 0].axhline(y=0, color='black', linestyle='--')
axes[1, 0].set_title('AR(1) Residuals (Wrong Model)', fontsize=11)
axes[1, 0].set_xlabel('Time')
axes[1, 0].set_ylabel('Residual')
axes[1, 0].grid(True, alpha=0.3)

plot_acf(resid_wrong, ax=axes[1, 1], lags=20, alpha=0.05)
axes[1, 1].set_title('AR(1) Residual ACF', fontsize=11)

stats.probplot(resid_wrong, dist="norm", plot=axes[1, 2])
axes[1, 2].set_title('AR(1) Q-Q Plot', fontsize=11)
axes[1, 2].grid(True, alpha=0.3)

axes[1, 3].hist(resid_wrong, bins=30, density=True, alpha=0.7, color='red', edgecolor='black')
x = np.linspace(resid_wrong.min(), resid_wrong.max(), 100)
axes[1, 3].plot(x, stats.norm.pdf(x, resid_wrong.mean(), resid_wrong.std()), 'k-', linewidth=2)
axes[1, 3].set_title('AR(1) Residual Distribution', fontsize=11)
axes[1, 3].set_xlabel('Residual')
axes[1, 3].set_ylabel('Density')

plt.tight_layout()
plt.savefig('../../charts/ch2_diagnostics.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch2_diagnostics.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("\n" + "=" * 60)
print("DIAGNOSTIC CHECKLIST")
print("=" * 60)
print("""
□ Ljung-Box test p-value > 0.05 (no autocorrelation)
□ Residual ACF within confidence bands
□ No patterns in residual time plot
□ Q-Q plot approximately linear (normality)
□ Histogram roughly bell-shaped

If diagnostics FAIL:
  1. Try different model order (increase p or q)
  2. Check for outliers
  3. Consider non-linear models (GARCH for volatility)
  4. Check for structural breaks
""")
