"""
TSA_ch2_estimation
==================
Demonstrate parameter estimation methods for ARMA models.

Description:
- Yule-Walker equations for AR
- Maximum Likelihood Estimation
- Compare estimation methods
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Set seed
np.random.seed(42)

# Simulate AR(2)
n = 500
phi1_true, phi2_true = 0.6, -0.3
sigma_true = 1.0

epsilon = np.random.normal(0, sigma_true, n)
x = np.zeros(n)
for t in range(2, n):
    x[t] = phi1_true * x[t-1] + phi2_true * x[t-2] + epsilon[t]

# Estimation methods
# 1. Yule-Walker
model_yw = AutoReg(x, lags=2, old_names=False).fit()
phi1_yw = model_yw.params[1]
phi2_yw = model_yw.params[2]

# 2. MLE
model_mle = ARIMA(x, order=(2, 0, 0)).fit()
phi1_mle = model_mle.params[1]
phi2_mle = model_mle.params[2]

# Create comparison figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Series with fitted values
ax1 = axes[0]
ax1.plot(x[:100], 'b-', linewidth=0.8, alpha=0.8, label='Observed')
ax1.plot(model_mle.fittedvalues[:100], 'r--', linewidth=1.2, label='Fitted (MLE)')
ax1.set_title('AR(2) Series with Fitted Values', fontsize=12)
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax1.grid(False)

# Plot 2: Parameter comparison
ax2 = axes[1]
methods = ['True', 'Yule-Walker', 'MLE']
phi1_vals = [phi1_true, phi1_yw, phi1_mle]
phi2_vals = [phi2_true, phi2_yw, phi2_mle]

x_pos = np.arange(len(methods))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, phi1_vals, width, label=r'$\phi_1$', color='blue', alpha=0.7)
bars2 = ax2.bar(x_pos + width/2, phi2_vals, width, label=r'$\phi_2$', color='red', alpha=0.7)

ax2.set_ylabel('Parameter Value')
ax2.set_title('Parameter Estimates Comparison', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(methods)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax2.grid(False)

plt.tight_layout()
plt.savefig('../../charts/estimation_comparison.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("Estimation Results:")
print(f"  True:        phi1={phi1_true:.4f}, phi2={phi2_true:.4f}")
print(f"  Yule-Walker: phi1={phi1_yw:.4f}, phi2={phi2_yw:.4f}")
print(f"  MLE:         phi1={phi1_mle:.4f}, phi2={phi2_mle:.4f}")
