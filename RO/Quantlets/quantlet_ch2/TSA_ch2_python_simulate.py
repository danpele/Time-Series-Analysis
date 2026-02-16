"""
TSA_ch2_python_simulate
=======================
Python Exercise 1: Simulate and Fit AR(1)

Tasks:
1. Simulate 300 observations from AR(1) with φ = 0.6
2. Plot the series and ACF/PACF
3. Fit AR(1) and compare estimated vs true φ
4. Examine residual diagnostics
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


np.random.seed(42)

# Parameters
n = 300
phi_true = 0.6
sigma = 1.0

print("=" * 60)
print("PYTHON EXERCISE 1: SIMULATE AND FIT AR(1)")
print("=" * 60)

# Step 1: Simulate AR(1)
eps = np.random.normal(0, sigma, n)
x = np.zeros(n)
for t in range(1, n):
    x[t] = phi_true * x[t-1] + eps[t]

print(f"\nSimulated AR(1) with φ = {phi_true}, n = {n}")

# Step 2: Plot series and ACF/PACF
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(x, color='steelblue', linewidth=0.8)
axes[0].set_title('Simulated AR(1) Series')
axes[0].set_xlabel('t')

plot_acf(x, ax=axes[1], lags=20, alpha=0.05)
axes[1].set_title('ACF')

plot_pacf(x, ax=axes[2], lags=20, alpha=0.05)
axes[2].set_title('PACF')

plt.tight_layout()
plt.savefig('../../charts/sem2_ar1_simulation.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/sem2_ar1_simulation.pdf', bbox_inches='tight', transparent=True)
plt.show()

# Step 3: Fit AR(1)
model = ARIMA(x, order=(1, 0, 0)).fit()
print(model.summary())

phi_hat = model.params[1]
print(f"\nTrue φ = {phi_true}")
print(f"Estimated φ̂ = {phi_hat:.4f}")
print(f"Estimation error = {abs(phi_hat - phi_true):.4f}")

# Step 4: Residual diagnostics
resid = model.resid
lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
print(f"\nLjung-Box test (lag 10):")
print(f"  Q-statistic = {lb_test['lb_stat'].values[0]:.4f}")
print(f"  p-value = {lb_test['lb_pvalue'].values[0]:.4f}")

if lb_test['lb_pvalue'].values[0] > 0.05:
    print("  → Fail to reject H₀: residuals are white noise ✓")
else:
    print("  → Reject H₀: residuals have autocorrelation ✗")
