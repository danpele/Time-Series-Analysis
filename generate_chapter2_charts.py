#!/usr/bin/env python3
"""Generate charts for Chapter 2: ARMA Models"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 12

# Colors
BLUE = '#1A3A6E'
RED = '#DC3545'
GREEN = '#2E7D32'
ORANGE = '#FF8C00'
PURPLE = '#6A1B9A'

np.random.seed(42)

def simulate_ar1(phi, n, sigma=1):
    """Simulate AR(1) process"""
    y = [0]
    for i in range(1, n):
        y.append(phi * y[-1] + np.random.normal(0, sigma))
    return np.array(y)

def simulate_ma1(theta, n, sigma=1):
    """Simulate MA(1) process"""
    eps = np.random.normal(0, sigma, n+1)
    y = eps[1:] + theta * eps[:-1]
    return y

def compute_acf(y, nlags=20):
    """Compute sample ACF"""
    n = len(y)
    y_centered = y - np.mean(y)
    acf = [1.0]
    for k in range(1, nlags + 1):
        acf.append(np.sum(y_centered[k:] * y_centered[:-k]) / np.sum(y_centered**2))
    return np.array(acf)

def compute_pacf(y, nlags=20):
    """Compute sample PACF using Yule-Walker"""
    acf_vals = compute_acf(y, nlags)
    pacf = [1.0, acf_vals[1]]
    for k in range(2, nlags + 1):
        # Durbin-Levinson algorithm
        phi = np.zeros(k)
        phi[0] = acf_vals[1]
        for j in range(1, k):
            num = acf_vals[j+1] - sum(phi[i] * acf_vals[j-i] for i in range(j))
            den = 1 - sum(phi[i] * acf_vals[i+1] for i in range(j))
            phi[j] = num / den if den != 0 else 0
            for i in range(j):
                phi[i] = phi[i] - phi[j] * phi[j-1-i]
        pacf.append(phi[-1])
    return np.array(pacf)

# Chart 1: AR(1) with different phi values
n = 150
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

phis = [0.9, 0.5, -0.5, -0.9]
colors = [BLUE, GREEN, ORANGE, RED]
titles = ['$\\phi = 0.9$ (High Persistence)', '$\\phi = 0.5$ (Moderate)',
          '$\\phi = -0.5$ (Oscillating)', '$\\phi = -0.9$ (Strong Oscillation)']

for ax, phi, color, title in zip(axes.flat, phis, colors, titles):
    y = simulate_ar1(phi, n)
    ax.plot(y, color=color, linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'AR(1): {title}', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('$Y_t$')

plt.tight_layout()
plt.savefig('charts/ch2_ar1_simulations.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch2_ar1_simulations.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch2_ar1_simulations.pdf")

# Chart 2: ACF and PACF of AR(1)
n = 500
y_ar1 = simulate_ar1(0.8, n)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Theoretical ACF for AR(1): rho_k = phi^k
nlags = 20
theoretical_acf = 0.8 ** np.arange(nlags + 1)
sample_acf = compute_acf(y_ar1, nlags)

axes[0].bar(range(nlags + 1), sample_acf, color=BLUE, width=0.4, alpha=0.7, label='Sample ACF')
axes[0].plot(range(nlags + 1), theoretical_acf, 'o-', color=RED, linewidth=2, markersize=6, label='Theoretical')
axes[0].axhline(y=1.96/np.sqrt(n), color='gray', linestyle='--', alpha=0.7)
axes[0].axhline(y=-1.96/np.sqrt(n), color='gray', linestyle='--', alpha=0.7)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_title('AR(1) ACF: Exponential Decay', fontweight='bold')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

# PACF - should cut off after lag 1
sample_pacf = compute_pacf(y_ar1, nlags)
axes[1].bar(range(nlags + 1), sample_pacf, color=GREEN, width=0.4)
axes[1].axhline(y=1.96/np.sqrt(n), color='gray', linestyle='--', alpha=0.7)
axes[1].axhline(y=-1.96/np.sqrt(n), color='gray', linestyle='--', alpha=0.7)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_title('AR(1) PACF: Cuts Off After Lag 1', fontweight='bold')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('PACF')

plt.tight_layout()
plt.savefig('charts/ch2_ar1_acf_pacf.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch2_ar1_acf_pacf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch2_ar1_acf_pacf.pdf")

# Chart 3: MA(1) with different theta values
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

thetas = [0.9, 0.5, -0.5, -0.9]
titles = ['$\\theta = 0.9$', '$\\theta = 0.5$',
          '$\\theta = -0.5$', '$\\theta = -0.9$']

for ax, theta, color, title in zip(axes.flat, thetas, colors, titles):
    y = simulate_ma1(theta, n)
    ax.plot(y, color=color, linewidth=1)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'MA(1): {title}', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('$Y_t$')

plt.tight_layout()
plt.savefig('charts/ch2_ma1_simulations.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch2_ma1_simulations.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch2_ma1_simulations.pdf")

# Chart 4: ACF and PACF of MA(1)
n = 500
y_ma1 = simulate_ma1(0.7, n)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ACF - should cut off after lag 1
sample_acf = compute_acf(y_ma1, nlags)
# Theoretical ACF for MA(1): rho_1 = theta/(1+theta^2), rho_k = 0 for k > 1
theoretical_acf = np.zeros(nlags + 1)
theoretical_acf[0] = 1
theoretical_acf[1] = 0.7 / (1 + 0.7**2)

axes[0].bar(range(nlags + 1), sample_acf, color=BLUE, width=0.4, alpha=0.7, label='Sample ACF')
axes[0].scatter(range(nlags + 1), theoretical_acf, color=RED, s=80, zorder=5, label='Theoretical')
axes[0].axhline(y=1.96/np.sqrt(n), color='gray', linestyle='--', alpha=0.7)
axes[0].axhline(y=-1.96/np.sqrt(n), color='gray', linestyle='--', alpha=0.7)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_title('MA(1) ACF: Cuts Off After Lag 1', fontweight='bold')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

# PACF - should decay
sample_pacf = compute_pacf(y_ma1, nlags)
axes[1].bar(range(nlags + 1), sample_pacf, color=GREEN, width=0.4)
axes[1].axhline(y=1.96/np.sqrt(n), color='gray', linestyle='--', alpha=0.7)
axes[1].axhline(y=-1.96/np.sqrt(n), color='gray', linestyle='--', alpha=0.7)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_title('MA(1) PACF: Exponential Decay', fontweight='bold')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('PACF')

plt.tight_layout()
plt.savefig('charts/ch2_ma1_acf_pacf.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch2_ma1_acf_pacf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch2_ma1_acf_pacf.pdf")

# Chart 5: ACF/PACF Pattern Summary
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# AR(1) example
y_ar = simulate_ar1(0.7, 300)
acf_ar = compute_acf(y_ar, 15)
pacf_ar = compute_pacf(y_ar, 15)

axes[0, 0].bar(range(16), acf_ar, color=BLUE, width=0.5)
axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
axes[0, 0].set_title('AR(p): ACF Decays', fontweight='bold')
axes[0, 0].set_ylabel('ACF')

axes[1, 0].bar(range(16), pacf_ar, color=BLUE, width=0.5)
axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
axes[1, 0].set_title('AR(p): PACF Cuts Off at p', fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('PACF')

# MA(1) example
y_ma = simulate_ma1(0.7, 300)
acf_ma = compute_acf(y_ma, 15)
pacf_ma = compute_pacf(y_ma, 15)

axes[0, 1].bar(range(16), acf_ma, color=GREEN, width=0.5)
axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
axes[0, 1].set_title('MA(q): ACF Cuts Off at q', fontweight='bold')

axes[1, 1].bar(range(16), pacf_ma, color=GREEN, width=0.5)
axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
axes[1, 1].set_title('MA(q): PACF Decays', fontweight='bold')
axes[1, 1].set_xlabel('Lag')

# ARMA(1,1) example - both decay
def simulate_arma11(phi, theta, n):
    eps = np.random.normal(0, 1, n+1)
    y = [0]
    for i in range(1, n):
        y.append(phi * y[-1] + eps[i] + theta * eps[i-1])
    return np.array(y)

y_arma = simulate_arma11(0.6, 0.4, 300)
acf_arma = compute_acf(y_arma, 15)
pacf_arma = compute_pacf(y_arma, 15)

axes[0, 2].bar(range(16), acf_arma, color=ORANGE, width=0.5)
axes[0, 2].axhline(y=0, color='black', linewidth=0.5)
axes[0, 2].set_title('ARMA(p,q): Both Decay', fontweight='bold')

axes[1, 2].bar(range(16), pacf_arma, color=ORANGE, width=0.5)
axes[1, 2].axhline(y=0, color='black', linewidth=0.5)
axes[1, 2].set_title('ARMA(p,q): Both Decay', fontweight='bold')
axes[1, 2].set_xlabel('Lag')

plt.tight_layout()
plt.savefig('charts/ch2_acf_pacf_patterns.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch2_acf_pacf_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch2_acf_pacf_patterns.pdf")

# Chart 6: Stationarity Region for AR(2)
fig, ax = plt.subplots(figsize=(10, 8))

# Stationarity region for AR(2): |phi2| < 1, phi1 + phi2 < 1, phi2 - phi1 < 1
phi1 = np.linspace(-2.5, 2.5, 500)

# Boundaries
ax.fill_between(phi1, -1, np.minimum(1 - phi1, 1 + phi1),
                where=(phi1 >= -2) & (phi1 <= 2),
                alpha=0.3, color=BLUE, label='Stationary Region')

# Draw boundary lines
ax.plot(phi1, 1 - phi1, 'r-', linewidth=2, label='$\\phi_1 + \\phi_2 = 1$')
ax.plot(phi1, -1 + phi1, 'g-', linewidth=2, label='$\\phi_2 - \\phi_1 = -1$')
ax.axhline(y=1, color=ORANGE, linewidth=2, linestyle='--', label='$\\phi_2 = 1$')
ax.axhline(y=-1, color=PURPLE, linewidth=2, linestyle='--', label='$\\phi_2 = -1$')

# Mark some points
ax.plot(0.5, 0.3, 'ko', markersize=10)
ax.annotate('Stationary\n(0.5, 0.3)', (0.5, 0.3), xytext=(0.8, 0.5),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=10)

ax.plot(1.5, 0.2, 'rx', markersize=12, markeredgewidth=3)
ax.annotate('Non-stationary\n(1.5, 0.2)', (1.5, 0.2), xytext=(1.8, 0.5),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('$\\phi_1$', fontsize=14)
ax.set_ylabel('$\\phi_2$', fontsize=14)
ax.set_title('Stationarity Region for AR(2)', fontweight='bold', fontsize=14)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)

plt.tight_layout()
plt.savefig('charts/ch2_ar2_stationarity.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch2_ar2_stationarity.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch2_ar2_stationarity.pdf")

# Chart 7: Model Diagnostics Example
n = 200
y = simulate_ar1(0.7, n)

# Pretend we fit an AR(1) model perfectly
fitted = np.array([0] + [0.7 * y[i] for i in range(n-1)])
residuals = y - fitted

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residual time plot
axes[0, 0].plot(residuals, color=BLUE, linewidth=0.8)
axes[0, 0].axhline(y=0, color='gray', linestyle='--')
axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Residual')

# Histogram
axes[0, 1].hist(residuals, bins=25, density=True, color=BLUE, alpha=0.7, edgecolor='white')
x = np.linspace(-4, 4, 100)
axes[0, 1].plot(x, stats.norm.pdf(x, 0, 1), color=RED, linewidth=2, label='N(0,1)')
axes[0, 1].set_title('Residual Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

# ACF of residuals
acf_resid = compute_acf(residuals, 20)
axes[1, 0].bar(range(21), acf_resid, color=BLUE, width=0.4)
axes[1, 0].axhline(y=1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=-1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
axes[1, 0].set_title('ACF of Residuals (Should Be White Noise)', fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')

# Q-Q plot (45-degree reference line)
(osm, osr), _ = stats.probplot(residuals, dist='norm', fit=True)
axes[1, 1].scatter(osm, osr, color=BLUE, alpha=0.6, s=30)
axes[1, 1].plot([-3, 3], [-3, 3], color=RED, linewidth=2, linestyle='--')
axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
axes[1, 1].set_xlabel('Theoretical Quantiles')
axes[1, 1].set_ylabel('Sample Quantiles')

plt.tight_layout()
plt.savefig('charts/ch2_diagnostics.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch2_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch2_diagnostics.pdf")

# Chart 8: Forecasting with AR(1)
n = 100
h = 20
phi = 0.8
y = simulate_ar1(phi, n)

# Generate forecasts
forecasts = []
last_val = y[-1]
for i in range(h):
    forecast = phi ** (i + 1) * last_val
    forecasts.append(forecast)
forecasts = np.array(forecasts)

# Forecast uncertainty (simplified)
sigma = 1
forecast_var = sigma**2 * np.array([(1 - phi**(2*(i+1))) / (1 - phi**2) for i in range(h)])
forecast_std = np.sqrt(forecast_var)

fig, ax = plt.subplots(figsize=(12, 6))

# Historical data
ax.plot(range(n), y, color=BLUE, linewidth=1.5, label='Historical')

# Forecasts
forecast_x = range(n, n + h)
ax.plot(forecast_x, forecasts, color=RED, linewidth=2, linestyle='--', label='Forecast')

# Confidence interval
ax.fill_between(forecast_x, forecasts - 1.96 * forecast_std, forecasts + 1.96 * forecast_std,
                color=RED, alpha=0.2, label='95% CI')

# Long-run mean
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.7, label='Long-run mean')
ax.axvline(x=n-1, color='gray', linestyle='--', alpha=0.5)

ax.set_title('AR(1) Forecasting: Mean Reversion', fontweight='bold', fontsize=14)
ax.set_xlabel('Time')
ax.set_ylabel('$Y_t$')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)

plt.tight_layout()
plt.savefig('charts/ch2_ar1_forecast.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch2_ar1_forecast.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch2_ar1_forecast.pdf")

print("\nAll Chapter 2 charts generated successfully!")
