"""
Generate charts for GARCH Volatility Models Chapter
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

COLORS = {
    'blue': '#1A3A6E',
    'red': '#DC3545',
    'green': '#2E7D32',
    'orange': '#E67E22',
    'gray': '#666666',
    'purple': '#6A1B9A'
}

# Create output directory
import os
os.makedirs('charts', exist_ok=True)

#=============================================================================
# Chart 1: Volatility Clustering
#=============================================================================
print("Generating Chart 1: Volatility Clustering...")

np.random.seed(42)
n = 1000

# Simulate GARCH(1,1)
omega = 0.00001
alpha = 0.1
beta = 0.85

sigma2 = np.zeros(n)
epsilon = np.zeros(n)
sigma2[0] = omega / (1 - alpha - beta)

for t in range(1, n):
    sigma2[t] = omega + alpha * epsilon[t-1]**2 + beta * sigma2[t-1]
    epsilon[t] = np.sqrt(sigma2[t]) * np.random.randn()

returns = epsilon * 100  # Scale for visualization

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Returns
axes[0].plot(returns, color=COLORS['blue'], linewidth=0.8, alpha=0.8)
axes[0].axhline(y=0, color='black', linewidth=0.5, linestyle='-')
axes[0].set_ylabel('Randamente (%)')
axes[0].set_title('Randamente Simulate GARCH(1,1)', fontweight='bold', fontsize=12)

# Conditional volatility
axes[1].plot(np.sqrt(sigma2) * 100, color=COLORS['red'], linewidth=1.2)
axes[1].fill_between(range(n), 0, np.sqrt(sigma2) * 100, color=COLORS['red'], alpha=0.3)
axes[1].set_ylabel('Volatilitate Condiționată (%)')
axes[1].set_xlabel('Timp')
axes[1].set_title('Volatility Clustering', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('charts/garch_volatility_clustering.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_volatility_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_volatility_clustering.pdf")

#=============================================================================
# Chart 2: Stylized Facts
#=============================================================================
print("Generating Chart 2: Stylized Facts...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ACF of returns
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(returns, lags=30, ax=axes[0, 0], color=COLORS['blue'],
         vlines_kwargs={'color': COLORS['blue']}, alpha=0.05)
axes[0, 0].set_title('ACF Randamente', fontweight='bold')
axes[0, 0].set_xlabel('Lag')

# ACF of squared returns
plot_acf(returns**2, lags=30, ax=axes[0, 1], color=COLORS['red'],
         vlines_kwargs={'color': COLORS['red']}, alpha=0.05)
axes[0, 1].set_title('ACF Randamente Pătrate', fontweight='bold')
axes[0, 1].set_xlabel('Lag')

# Histogram with normal overlay
axes[1, 0].hist(returns, bins=50, density=True, color=COLORS['blue'],
                alpha=0.7, edgecolor='white', label='Date')
x = np.linspace(returns.min(), returns.max(), 100)
axes[1, 0].plot(x, stats.norm.pdf(x, returns.mean(), returns.std()),
                color=COLORS['red'], linewidth=2, label='Normal')
axes[1, 0].set_title('Distribuție: Cozi Groase', fontweight='bold')
axes[1, 0].set_xlabel('Randament (%)')
axes[1, 0].set_ylabel('Densitate')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  frameon=False, ncol=2)

# QQ-plot
stats.probplot(returns, dist="norm", plot=axes[1, 1])
axes[1, 1].get_lines()[0].set_color(COLORS['blue'])
axes[1, 1].get_lines()[0].set_markersize(4)
axes[1, 1].get_lines()[1].set_color(COLORS['red'])
axes[1, 1].set_title('QQ-Plot vs Normal', fontweight='bold')

plt.tight_layout()
plt.savefig('charts/garch_stylized_facts.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_stylized_facts.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_stylized_facts.pdf")

#=============================================================================
# Chart 3: Conditional Variance
#=============================================================================
print("Generating Chart 3: Conditional Variance Dynamics...")

# Different alpha/beta combinations
params = [
    (0.05, 0.90, 'α=0.05, β=0.90'),
    (0.10, 0.85, 'α=0.10, β=0.85'),
    (0.20, 0.70, 'α=0.20, β=0.70'),
]

np.random.seed(123)
n = 300
omega = 0.0001

fig, ax = plt.subplots(figsize=(12, 6))

for i, (alpha, beta, label) in enumerate(params):
    sigma2 = np.zeros(n)
    epsilon = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, n):
        sigma2[t] = omega + alpha * epsilon[t-1]**2 + beta * sigma2[t-1]
        epsilon[t] = np.sqrt(sigma2[t]) * np.random.randn()

    colors = [COLORS['blue'], COLORS['red'], COLORS['green']]
    ax.plot(np.sqrt(sigma2) * 100, linewidth=1.5, color=colors[i], label=label)

ax.set_xlabel('Timp')
ax.set_ylabel('Volatilitate Condiționată (%)')
ax.set_title('GARCH(1,1): Impactul Parametrilor', fontweight='bold', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=3)

plt.tight_layout()
plt.savefig('charts/garch_conditional_variance.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_conditional_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_conditional_variance.pdf")

#=============================================================================
# Chart 4: Leverage Effect
#=============================================================================
print("Generating Chart 4: Leverage Effect...")

np.random.seed(456)
n = 500

# Simulate with leverage
omega = 0.0001
alpha = 0.05
gamma = 0.10  # leverage
beta = 0.85

sigma2 = np.zeros(n)
epsilon = np.zeros(n)
sigma2[0] = omega / (1 - alpha - gamma/2 - beta)

for t in range(1, n):
    indicator = 1 if epsilon[t-1] < 0 else 0
    sigma2[t] = omega + alpha * epsilon[t-1]**2 + gamma * epsilon[t-1]**2 * indicator + beta * sigma2[t-1]
    epsilon[t] = np.sqrt(sigma2[t]) * np.random.randn()

returns_lev = epsilon * 100

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Scatter plot: returns vs next volatility
axes[0].scatter(returns_lev[:-1], np.sqrt(sigma2[1:]) * 100,
                c=[COLORS['red'] if r < 0 else COLORS['blue'] for r in returns_lev[:-1]],
                alpha=0.5, s=15)
axes[0].set_xlabel('Randament la t (%)')
axes[0].set_ylabel('Volatilitate la t+1 (%)')
axes[0].set_title('Leverage Effect: Șocuri Negative vs Pozitive', fontweight='bold')
axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)

# Box plot comparison
negative_returns = returns_lev[returns_lev < 0]
positive_returns = returns_lev[returns_lev > 0]

# Get corresponding next-period volatilities
neg_mask = returns_lev[:-1] < 0
pos_mask = returns_lev[:-1] > 0
next_vol_neg = np.sqrt(sigma2[1:])[neg_mask] * 100
next_vol_pos = np.sqrt(sigma2[1:])[pos_mask] * 100

bp = axes[1].boxplot([next_vol_neg, next_vol_pos],
                      labels=['După Șoc Negativ', 'După Șoc Pozitiv'],
                      patch_artist=True)
bp['boxes'][0].set_facecolor(COLORS['red'])
bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor(COLORS['blue'])
bp['boxes'][1].set_alpha(0.6)
axes[1].set_ylabel('Volatilitate Următoare (%)')
axes[1].set_title('Distribuția Volatilității după Tipul Șocului', fontweight='bold')

plt.tight_layout()
plt.savefig('charts/garch_leverage_effect.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_leverage_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_leverage_effect.pdf")

#=============================================================================
# Chart 5: News Impact Curve
#=============================================================================
print("Generating Chart 5: News Impact Curve...")

# Parameters
omega = 0.0001
sigma2_prev = 0.0004  # Previous variance
epsilon_range = np.linspace(-0.05, 0.05, 200)

# GARCH (symmetric)
alpha_g = 0.10
beta_g = 0.85
sigma2_garch = omega + alpha_g * epsilon_range**2 + beta_g * sigma2_prev

# GJR-GARCH (asymmetric)
alpha_gjr = 0.05
gamma_gjr = 0.10
beta_gjr = 0.85
indicator = (epsilon_range < 0).astype(float)
sigma2_gjr = omega + alpha_gjr * epsilon_range**2 + gamma_gjr * epsilon_range**2 * indicator + beta_gjr * sigma2_prev

# EGARCH-like (for illustration)
omega_e = np.log(omega)
alpha_e = 0.15
gamma_e = -0.10  # negative for leverage
beta_e = 0.85
z = epsilon_range / np.sqrt(sigma2_prev)
log_sigma2_egarch = omega_e + alpha_e * (np.abs(z) - np.sqrt(2/np.pi)) + gamma_e * z + beta_e * np.log(sigma2_prev)
sigma2_egarch = np.exp(log_sigma2_egarch)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(epsilon_range * 100, np.sqrt(sigma2_garch) * 100,
        color=COLORS['blue'], linewidth=2, label='GARCH (simetric)')
ax.plot(epsilon_range * 100, np.sqrt(sigma2_gjr) * 100,
        color=COLORS['red'], linewidth=2, label='GJR-GARCH (asimetric)')
ax.plot(epsilon_range * 100, np.sqrt(sigma2_egarch) * 100,
        color=COLORS['green'], linewidth=2, linestyle='--', label='EGARCH')

ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('Șoc εₜ₋₁ (%)', fontsize=12)
ax.set_ylabel('Volatilitate Condiționată σₜ (%)', fontsize=12)
ax.set_title('News Impact Curve: Comparație Modele', fontweight='bold', fontsize=14)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=3)

plt.tight_layout()
plt.savefig('charts/garch_news_impact_curve.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_news_impact_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_news_impact_curve.pdf")

#=============================================================================
# Chart 6: Diagnostics
#=============================================================================
print("Generating Chart 6: Model Diagnostics...")

# Simulate standardized residuals (should be ~N(0,1) if model is correct)
np.random.seed(789)
n = 500
std_resid = np.random.standard_t(df=5, size=n)  # Use t-dist for realistic fat tails
std_resid = (std_resid - std_resid.mean()) / std_resid.std()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Time series of standardized residuals
axes[0, 0].plot(std_resid, color=COLORS['blue'], linewidth=0.8, alpha=0.8)
axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
axes[0, 0].axhline(y=2, color=COLORS['red'], linestyle='--', linewidth=0.8, alpha=0.5)
axes[0, 0].axhline(y=-2, color=COLORS['red'], linestyle='--', linewidth=0.8, alpha=0.5)
axes[0, 0].set_title('Reziduuri Standardizate', fontweight='bold')
axes[0, 0].set_xlabel('Timp')
axes[0, 0].set_ylabel('zₜ')

# ACF of squared standardized residuals
plot_acf(std_resid**2, lags=20, ax=axes[0, 1], color=COLORS['blue'],
         vlines_kwargs={'color': COLORS['blue']}, alpha=0.05)
axes[0, 1].set_title('ACF(zₜ²) - Efecte ARCH Reziduale', fontweight='bold')
axes[0, 1].set_xlabel('Lag')

# Histogram
axes[1, 0].hist(std_resid, bins=40, density=True, color=COLORS['blue'],
                alpha=0.7, edgecolor='white', label='Reziduuri')
x = np.linspace(-4, 4, 100)
axes[1, 0].plot(x, stats.norm.pdf(x), color=COLORS['red'],
                linewidth=2, label='N(0,1)')
axes[1, 0].set_title('Distribuția Reziduurilor Standardizate', fontweight='bold')
axes[1, 0].set_xlabel('zₜ')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=2)

# QQ-plot
stats.probplot(std_resid, dist="norm", plot=axes[1, 1])
axes[1, 1].get_lines()[0].set_color(COLORS['blue'])
axes[1, 1].get_lines()[0].set_markersize(4)
axes[1, 1].get_lines()[1].set_color(COLORS['red'])
axes[1, 1].set_title('QQ-Plot Reziduuri vs Normal', fontweight='bold')

plt.tight_layout()
plt.savefig('charts/garch_diagnostics.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_diagnostics.pdf")

#=============================================================================
# Chart 7: Forecast
#=============================================================================
print("Generating Chart 7: Volatility Forecast...")

np.random.seed(321)
n = 200
h = 50  # forecast horizon

omega = 0.0001
alpha = 0.08
beta = 0.90
unconditional_var = omega / (1 - alpha - beta)

# Generate historical data
sigma2 = np.zeros(n)
epsilon = np.zeros(n)
sigma2[0] = unconditional_var

for t in range(1, n):
    sigma2[t] = omega + alpha * epsilon[t-1]**2 + beta * sigma2[t-1]
    epsilon[t] = np.sqrt(sigma2[t]) * np.random.randn()

# Forecast
sigma2_forecast = np.zeros(h)
sigma2_forecast[0] = omega + alpha * epsilon[-1]**2 + beta * sigma2[-1]

for t in range(1, h):
    sigma2_forecast[t] = unconditional_var + (alpha + beta)**(t) * (sigma2_forecast[0] - unconditional_var)

fig, ax = plt.subplots(figsize=(12, 6))

# Historical
time_hist = np.arange(n)
ax.plot(time_hist, np.sqrt(sigma2) * 100, color=COLORS['blue'], linewidth=1.2, label='Volatilitate Istorică')

# Forecast
time_fore = np.arange(n-1, n + h)
forecast_line = np.concatenate([[np.sqrt(sigma2[-1]) * 100], np.sqrt(sigma2_forecast) * 100])
ax.plot(time_fore, forecast_line, color=COLORS['red'], linewidth=2, linestyle='--', label='Prognoză')

# Unconditional
ax.axhline(y=np.sqrt(unconditional_var) * 100, color=COLORS['green'],
           linestyle=':', linewidth=1.5, label=f'σ̄ = {np.sqrt(unconditional_var)*100:.2f}%')

# Vertical line at forecast start
ax.axvline(x=n-1, color='gray', linestyle='--', linewidth=1, alpha=0.5)

ax.set_xlabel('Timp')
ax.set_ylabel('Volatilitate (%)')
ax.set_title('Prognoza Volatilității GARCH(1,1)', fontweight='bold', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=3)
ax.set_xlim(0, n + h)

plt.tight_layout()
plt.savefig('charts/garch_forecast.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_forecast.pdf")

#=============================================================================
# Chart 8: S&P 500 Returns (Simulated realistic data)
#=============================================================================
print("Generating Chart 8: S&P 500 Returns (Simulated)...")

np.random.seed(2008)
n = 5000  # ~20 years of daily data

# Simulate with regime changes to mimic crises
omega = 0.00001
alpha = 0.085
beta = 0.905

sigma2 = np.zeros(n)
epsilon = np.zeros(n)
sigma2[0] = omega / (1 - alpha - beta)

# Add crisis periods (higher volatility shocks)
crisis_periods = [
    (1800, 2000, 3),   # 2008 crisis
    (4000, 4200, 2.5), # COVID-19
    (4600, 4750, 1.8), # 2022 volatility
]

for t in range(1, n):
    shock_mult = 1
    for start, end, mult in crisis_periods:
        if start <= t <= end:
            shock_mult = mult
            break

    z = np.random.randn() * shock_mult
    sigma2[t] = omega + alpha * epsilon[t-1]**2 + beta * sigma2[t-1]
    epsilon[t] = np.sqrt(sigma2[t]) * z

returns_sp = epsilon * 100

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Returns
axes[0].plot(returns_sp, color=COLORS['blue'], linewidth=0.5, alpha=0.8)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_ylabel('Randament Zilnic (%)')
axes[0].set_title('S&P 500 Randamente Zilnice (Simulate)', fontweight='bold', fontsize=12)

# Highlight crisis periods
for start, end, mult in crisis_periods:
    axes[0].axvspan(start, end, alpha=0.2, color=COLORS['red'])
    axes[1].axvspan(start, end, alpha=0.2, color=COLORS['red'])

# Volatility
axes[1].plot(np.sqrt(sigma2) * 100, color=COLORS['red'], linewidth=0.8)
axes[1].fill_between(range(n), 0, np.sqrt(sigma2) * 100, color=COLORS['red'], alpha=0.3)
axes[1].set_ylabel('Volatilitate Condiționată (%)')
axes[1].set_xlabel('Zile')
axes[1].set_title('Volatilitate GARCH(1,1)', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('charts/garch_sp500_returns.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_sp500_returns.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_sp500_returns.pdf")

#=============================================================================
# Chart 9: S&P 500 Volatility Detail
#=============================================================================
print("Generating Chart 9: S&P 500 Volatility Detail...")

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(np.sqrt(sigma2) * 100, color=COLORS['blue'], linewidth=1)
ax.fill_between(range(n), 0, np.sqrt(sigma2) * 100, color=COLORS['blue'], alpha=0.3)

# Add annotations
ax.annotate('Criză 2008', xy=(1900, np.sqrt(sigma2[1900])*100),
            xytext=(1600, np.sqrt(sigma2[1900])*100 + 1),
            fontsize=10, color=COLORS['red'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['red']))

ax.annotate('COVID-19', xy=(4100, np.sqrt(sigma2[4100])*100),
            xytext=(4300, np.sqrt(sigma2[4100])*100 + 0.5),
            fontsize=10, color=COLORS['red'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['red']))

ax.set_xlabel('Zile')
ax.set_ylabel('Volatilitate Condiționată (%)')
ax.set_title('Estimare GARCH(1,1): S&P 500', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('charts/garch_sp500_volatility.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_sp500_volatility.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_sp500_volatility.pdf")

#=============================================================================
# Chart 10: GARCH vs EGARCH Comparison
#=============================================================================
print("Generating Chart 10: GARCH vs EGARCH Comparison...")

np.random.seed(555)
n = 500

# GJR-GARCH
omega = 0.0001
alpha = 0.05
gamma = 0.12  # leverage
beta = 0.85

sigma2_gjr = np.zeros(n)
epsilon = np.zeros(n)
sigma2_gjr[0] = omega / (1 - alpha - gamma/2 - beta)

for t in range(1, n):
    indicator = 1 if epsilon[t-1] < 0 else 0
    sigma2_gjr[t] = omega + alpha * epsilon[t-1]**2 + gamma * epsilon[t-1]**2 * indicator + beta * sigma2_gjr[t-1]
    epsilon[t] = np.sqrt(sigma2_gjr[t]) * np.random.randn()

# Standard GARCH on same innovations
sigma2_garch = np.zeros(n)
sigma2_garch[0] = omega / (1 - alpha - beta)
alpha_g = alpha + gamma/2  # adjusted to have similar persistence

for t in range(1, n):
    sigma2_garch[t] = omega + alpha_g * epsilon[t-1]**2 + beta * sigma2_garch[t-1]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(np.sqrt(sigma2_garch) * 100, color=COLORS['blue'], linewidth=1.2,
        label='GARCH(1,1)', alpha=0.8)
ax.plot(np.sqrt(sigma2_gjr) * 100, color=COLORS['red'], linewidth=1.2,
        label='GJR-GARCH(1,1,1)', alpha=0.8)

ax.set_xlabel('Timp')
ax.set_ylabel('Volatilitate Condiționată (%)')
ax.set_title('Comparație GARCH vs GJR-GARCH', fontweight='bold', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=2)

plt.tight_layout()
plt.savefig('charts/garch_sp500_comparison.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_sp500_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_sp500_comparison.pdf")

print("\n" + "="*50)
print("All GARCH charts generated successfully!")
print("="*50)
