"""
TSA_ch2_python_diagnostics
==========================
Python Exercise 3: Residual Diagnostics

Tasks:
1. Plot residuals over time
2. Plot ACF of residuals
3. Create Q-Q plot for normality
4. Run Ljung-Box test
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


np.random.seed(42)

print("=" * 60)
print("PYTHON EXERCISE 3: RESIDUAL DIAGNOSTICS")
print("=" * 60)

# Generate ARMA(1,1) data and fit model
ar_params = np.array([1, -0.7])
ma_params = np.array([1, 0.4])
arma_process = ArmaProcess(ar_params, ma_params)
y = arma_process.generate_sample(nsample=300)

model = ARIMA(y, order=(1, 0, 1)).fit()
resid = model.resid

print(f"\nFitted ARMA(1,1) model")
print(f"Residual mean: {resid.mean():.6f}")
print(f"Residual std:  {resid.std():.6f}")

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Residuals over time
axes[0, 0].plot(resid, color='steelblue', linewidth=0.8)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=0.8)
axes[0, 0].set_title('Residuals Over Time')
axes[0, 0].set_xlabel('t')

# 2. ACF of residuals
plot_acf(resid, ax=axes[0, 1], lags=20, alpha=0.05)
axes[0, 1].set_title('ACF of Residuals')

# 3. Q-Q plot
stats.probplot(resid, dist='norm', plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normality)')

# 4. Histogram of residuals
axes[1, 1].hist(resid, bins=30, density=True, color='steelblue', alpha=0.7, edgecolor='white')
x_range = np.linspace(resid.min(), resid.max(), 100)
axes[1, 1].plot(x_range, stats.norm.pdf(x_range, resid.mean(), resid.std()),
                'r-', linewidth=2, label='Normal')
axes[1, 1].set_title('Residual Distribution')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

plt.tight_layout()
plt.savefig('../../charts/sem2_diagnostics.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/sem2_diagnostics.pdf', bbox_inches='tight', transparent=True)
plt.show()

# Ljung-Box test
lb_test = acorr_ljungbox(resid, lags=[5, 10, 15, 20], return_df=True)
print(f"\nLjung-Box Test Results:")
print(f"{'Lag':>5} {'Q-stat':>10} {'p-value':>10} {'Decision':>15}")
print("-" * 42)
for lag, row in lb_test.iterrows():
    decision = "White noise ✓" if row['lb_pvalue'] > 0.05 else "Autocorrelation ✗"
    print(f"{lag:>5} {row['lb_stat']:>10.4f} {row['lb_pvalue']:>10.4f} {decision:>15}")

# Jarque-Bera normality test
jb_stat, jb_pval = stats.jarque_bera(resid)
print(f"\nJarque-Bera Test: statistic = {jb_stat:.4f}, p-value = {jb_pval:.4f}")
if jb_pval > 0.05:
    print("  → Fail to reject H₀: residuals are normally distributed ✓")
else:
    print("  → Reject H₀: residuals are NOT normally distributed ✗")
