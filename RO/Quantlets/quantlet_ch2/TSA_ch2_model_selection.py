"""
TSA_ch2_model_selection
=======================
Model Selection: AIC, BIC, and Cross-Validation

This script demonstrates:
- Akaike Information Criterion (AIC)
- Bayesian Information Criterion (BIC)
- Model comparison and selection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess

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
print("MODEL SELECTION: AIC AND BIC")
print("=" * 60)

print("""
Information Criteria:

AIC = -2 ln(L̂) + 2k
BIC = -2 ln(L̂) + k ln(n)

Where:
  L̂ = maximized likelihood
  k = number of parameters
  n = sample size

Key Differences:
  - AIC: penalizes complexity less → larger models
  - BIC: penalizes complexity more → more parsimonious models
  - BIC penalty grows with n → increasingly favors simpler models

Rule: LOWER is BETTER
""")

# Generate true ARMA(1,1) data
phi_true = 0.7
theta_true = 0.4
ar = np.array([1, -phi_true])
ma = np.array([1, theta_true])
arma_process = ArmaProcess(ar, ma)
data = arma_process.generate_sample(nsample=n)

# Fit various models and compare
models = [
    (1, 0, 'AR(1)'),
    (2, 0, 'AR(2)'),
    (3, 0, 'AR(3)'),
    (0, 1, 'MA(1)'),
    (0, 2, 'MA(2)'),
    (1, 1, 'ARMA(1,1)'),
    (2, 1, 'ARMA(2,1)'),
    (1, 2, 'ARMA(1,2)'),
    (2, 2, 'ARMA(2,2)')
]

results = []

print("\n" + "=" * 60)
print("FITTING MULTIPLE MODELS TO ARMA(1,1) DATA")
print("=" * 60)
print(f"True model: ARMA(1,1) with φ={phi_true}, θ={theta_true}")
print(f"Sample size: n = {n}")
print("-" * 60)
print(f"{'Model':<12} {'k':>5} {'Log-Like':>12} {'AIC':>10} {'BIC':>10}")
print("-" * 60)

for p, q, name in models:
    try:
        model = ARIMA(data, order=(p, 0, q))
        fit = model.fit()
        k = p + q + 1  # AR + MA + variance
        results.append({
            'Model': name,
            'p': p,
            'q': q,
            'k': k,
            'LogLike': fit.llf,
            'AIC': fit.aic,
            'BIC': fit.bic
        })
        print(f"{name:<12} {k:>5} {fit.llf:>12.2f} {fit.aic:>10.2f} {fit.bic:>10.2f}")
    except Exception as e:
        print(f"{name:<12} -- fitting failed")

df = pd.DataFrame(results)

# Find best models
best_aic = df.loc[df['AIC'].idxmin()]
best_bic = df.loc[df['BIC'].idxmin()]

print("-" * 60)
print(f"\nBest by AIC: {best_aic['Model']} (AIC = {best_aic['AIC']:.2f})")
print(f"Best by BIC: {best_bic['Model']} (BIC = {best_bic['BIC']:.2f})")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# AIC comparison
ax1 = axes[0]
colors = ['green' if m == best_aic['Model'] else 'blue' for m in df['Model']]
ax1.barh(df['Model'], df['AIC'], color=colors, alpha=0.7, edgecolor='black')
ax1.axvline(x=best_aic['AIC'], color='green', linestyle='--', linewidth=2)
ax1.set_xlabel('AIC (lower is better)')
ax1.set_title('AIC Comparison')
ax1.grid(True, alpha=0.3, axis='x')

# BIC comparison
ax2 = axes[1]
colors = ['green' if m == best_bic['Model'] else 'blue' for m in df['Model']]
ax2.barh(df['Model'], df['BIC'], color=colors, alpha=0.7, edgecolor='black')
ax2.axvline(x=best_bic['BIC'], color='green', linestyle='--', linewidth=2)
ax2.set_xlabel('BIC (lower is better)')
ax2.set_title('BIC Comparison')
ax2.grid(True, alpha=0.3, axis='x')

# Parameter count vs fit
ax3 = axes[2]
ax3.scatter(df['k'], df['AIC'], s=100, label='AIC', alpha=0.7)
ax3.scatter(df['k'], df['BIC'], s=100, marker='s', label='BIC', alpha=0.7)
for _, row in df.iterrows():
    ax3.annotate(row['Model'], (row['k'], row['AIC']), xytext=(5, 5),
                 textcoords='offset points', fontsize=8)
ax3.set_xlabel('Number of Parameters (k)')
ax3.set_ylabel('Information Criterion')
ax3.set_title('Complexity vs Fit Trade-off')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../charts/ch2_model_selection.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch2_model_selection.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("\n" + "=" * 60)
print("MODEL SELECTION GUIDELINES")
print("=" * 60)
print("""
1. Start with ACF/PACF to get initial p, q estimates
2. Fit candidate models (vary p and q around initial guess)
3. Compare using AIC and BIC
4. Check residual diagnostics (Ljung-Box test)
5. Use out-of-sample validation if possible

When AIC and BIC Disagree:
  - AIC tends to select larger models
  - BIC tends to select smaller models
  - BIC is consistent (selects true model as n → ∞)
  - AIC is asymptotically efficient (best predictions)

Practical Advice:
  - Use BIC for model interpretation
  - Use AIC for forecasting
  - Consider both and check residuals
""")
