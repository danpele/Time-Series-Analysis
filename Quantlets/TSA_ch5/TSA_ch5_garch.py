"""
TSA_ch5_garch
=============
GARCH Models: Volatility Modeling

This script demonstrates:
- Volatility clustering in financial returns
- ARCH/GARCH model specification
- Model estimation with arch package
- News impact curves
- Volatility forecasting
- Value at Risk (VaR) calculation

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import arch package
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    print("Installing arch package...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'arch', '-q'])
    from arch import arch_model
    HAS_ARCH = True

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
print("GARCH MODELS: VOLATILITY ANALYSIS")
print("=" * 70)

# =============================================================================
# 1. Generate Financial Returns with Volatility Clustering
# =============================================================================
np.random.seed(42)
n = 2000

print("\n1. SIMULATING GARCH(1,1) PROCESS")
print("-" * 40)

# GARCH(1,1) parameters
omega = 0.00001  # constant
alpha = 0.10     # ARCH effect
beta = 0.85      # GARCH effect
print(f"   Parameters: ω = {omega}, α = {alpha}, β = {beta}")
print(f"   Persistence: α + β = {alpha + beta:.2f}")

# Simulate GARCH(1,1)
returns = np.zeros(n)
sigma2 = np.zeros(n)
sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance

for t in range(1, n):
    sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
    returns[t] = np.sqrt(sigma2[t]) * np.random.standard_normal()

returns = returns * 100  # Convert to percentage
sigma = np.sqrt(sigma2) * 100

dates = pd.date_range('2015-01-01', periods=n, freq='B')
df = pd.DataFrame({'returns': returns, 'volatility': sigma}, index=dates)

print(f"   Simulated {n} observations")
print(f"   Unconditional volatility: {np.sqrt(omega / (1 - alpha - beta)) * 100:.4f}%")

# =============================================================================
# 2. Stylized Facts Visualization
# =============================================================================
print("\n2. STYLIZED FACTS OF FINANCIAL RETURNS")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Returns time series
axes[0, 0].plot(df.index, df['returns'], color='#1A3A6E', linewidth=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=0.8)
axes[0, 0].set_title('Daily Returns (%)', fontweight='bold')
axes[0, 0].set_xlabel('Date')

# Volatility clustering
axes[0, 1].plot(df.index, df['volatility'], color='#DC3545', linewidth=0.8)
axes[0, 1].fill_between(df.index, 0, df['volatility'], alpha=0.3, color='#DC3545')
axes[0, 1].set_title('Conditional Volatility (True)', fontweight='bold')
axes[0, 1].set_xlabel('Date')

# Distribution
axes[1, 0].hist(df['returns'], bins=50, density=True, color='#1A3A6E',
                alpha=0.7, edgecolor='white')
x = np.linspace(df['returns'].min(), df['returns'].max(), 100)
axes[1, 0].plot(x, stats.norm.pdf(x, df['returns'].mean(), df['returns'].std()),
               'r-', lw=2, label='Normal')
axes[1, 0].set_title('Return Distribution (Fat Tails)', fontweight='bold')
axes[1, 0].set_xlabel('Return (%)')
kurtosis = stats.kurtosis(df['returns'])
axes[1, 0].text(0.95, 0.95, f'Kurtosis: {kurtosis:.2f}', transform=axes[1, 0].transAxes,
               ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ACF of squared returns
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['returns']**2, lags=30, ax=axes[1, 1], color='#1A3A6E')
axes[1, 1].set_title('ACF of Squared Returns (Volatility Clustering)', fontweight='bold')

plt.tight_layout()
plt.savefig('ch5_stylized_facts.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch5_stylized_facts.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch5_stylized_facts.pdf")

# =============================================================================
# 3. GARCH Model Estimation
# =============================================================================
print("\n3. GARCH MODEL ESTIMATION")
print("-" * 40)

# Fit GARCH(1,1)
model = arch_model(df['returns'], vol='Garch', p=1, q=1, dist='normal')
results = model.fit(disp='off')

print(f"\n   GARCH(1,1) Results:")
print(f"   ω (omega):  {results.params['omega']:.6f}")
print(f"   α (alpha):  {results.params['alpha[1]']:.4f}")
print(f"   β (beta):   {results.params['beta[1]']:.4f}")
print(f"   α + β:      {results.params['alpha[1]'] + results.params['beta[1]']:.4f}")
print(f"\n   Log-likelihood: {results.loglikelihood:.2f}")
print(f"   AIC: {results.aic:.2f}")
print(f"   BIC: {results.bic:.2f}")

# Conditional volatility
cond_vol = results.conditional_volatility

# =============================================================================
# 4. Model Comparison
# =============================================================================
print("\n4. MODEL COMPARISON")
print("-" * 40)

models_dict = {}

# GARCH(1,1) Normal
models_dict['GARCH-N'] = arch_model(df['returns'], vol='Garch', p=1, q=1, dist='normal').fit(disp='off')

# GARCH(1,1) Student-t
models_dict['GARCH-t'] = arch_model(df['returns'], vol='Garch', p=1, q=1, dist='t').fit(disp='off')

# GJR-GARCH (asymmetric)
models_dict['GJR-GARCH'] = arch_model(df['returns'], vol='Garch', p=1, o=1, q=1, dist='normal').fit(disp='off')

# EGARCH
models_dict['EGARCH'] = arch_model(df['returns'], vol='EGARCH', p=1, q=1, dist='normal').fit(disp='off')

print(f"   {'Model':<12} {'AIC':>10} {'BIC':>10} {'LogL':>12}")
print("   " + "-" * 46)
for name, res in models_dict.items():
    print(f"   {name:<12} {res.aic:>10.2f} {res.bic:>10.2f} {res.loglikelihood:>12.2f}")

best_model = min(models_dict.keys(), key=lambda x: models_dict[x].aic)
print(f"\n   Best model (AIC): {best_model}")

# =============================================================================
# 5. News Impact Curve
# =============================================================================
print("\n5. NEWS IMPACT CURVE")
print("-" * 40)

# Get parameters
omega_est = results.params['omega']
alpha_est = results.params['alpha[1]']
beta_est = results.params['beta[1]']

# Previous variance (unconditional)
sigma2_prev = omega_est / (1 - alpha_est - beta_est)

# Range of shocks
eps_range = np.linspace(-4, 4, 200)

# GARCH (symmetric)
sigma2_garch = omega_est + alpha_est * eps_range**2 + beta_est * sigma2_prev

# GJR-GARCH (asymmetric)
gjr_res = models_dict['GJR-GARCH']
omega_gjr = gjr_res.params['omega']
alpha_gjr = gjr_res.params['alpha[1]']
gamma_gjr = gjr_res.params['gamma[1]']
beta_gjr = gjr_res.params['beta[1]']
sigma2_prev_gjr = omega_gjr / (1 - alpha_gjr - gamma_gjr/2 - beta_gjr)

indicator = (eps_range < 0).astype(float)
sigma2_gjr = omega_gjr + alpha_gjr * eps_range**2 + gamma_gjr * eps_range**2 * indicator + beta_gjr * sigma2_prev_gjr

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(eps_range, np.sqrt(sigma2_garch), color='#1A3A6E', linewidth=2.5, label='GARCH(1,1)')
ax.plot(eps_range, np.sqrt(sigma2_gjr), color='#DC3545', linewidth=2.5, label='GJR-GARCH')
ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('Shock εₜ₋₁', fontsize=12)
ax.set_ylabel('Conditional Volatility σₜ', fontsize=12)
ax.set_title('News Impact Curve', fontweight='bold', fontsize=14)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch5_news_impact.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch5_news_impact.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch5_news_impact.pdf")
print(f"   GJR leverage effect (γ): {gamma_gjr:.4f}")

# =============================================================================
# 6. Volatility Forecast
# =============================================================================
print("\n6. VOLATILITY FORECASTING")
print("-" * 40)

forecast_horizon = 30
forecasts = results.forecast(horizon=forecast_horizon)
forecast_var = forecasts.variance.iloc[-1].values
forecast_vol = np.sqrt(forecast_var)

# Unconditional volatility
uncond_vol = np.sqrt(omega_est / (1 - alpha_est - beta_est))

fig, ax = plt.subplots(figsize=(12, 5))

# Historical volatility (last 100 days)
hist_vol = cond_vol[-100:]
hist_dates = df.index[-100:]

# Forecast dates
forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1),
                                periods=forecast_horizon, freq='B')

ax.plot(hist_dates, hist_vol, color='#1A3A6E', linewidth=1.2, label='Historical')
ax.plot(forecast_dates, forecast_vol, color='#DC3545', linewidth=2,
        linestyle='--', label='Forecast')
ax.axhline(y=uncond_vol, color='#2E7D32', linestyle=':', linewidth=1.5,
           label=f'Unconditional σ = {uncond_vol:.2f}%')

# Visual separator between historical and forecast
split_point = df.index[-1]
ax.axvline(x=split_point, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
y_pos = ax.get_ylim()[1] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
ax.text(split_point, y_pos, '  Forecast ', fontsize=9, ha='left', va='top',
        color='black', fontweight='bold', alpha=0.8)

ax.set_xlabel('Date')
ax.set_ylabel('Volatility (%)')
ax.set_title('GARCH(1,1) Volatility Forecast', fontweight='bold', fontsize=14)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch5_forecast.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch5_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch5_forecast.pdf")
print(f"   Forecast horizon: {forecast_horizon} days")
print(f"   1-day ahead volatility: {forecast_vol[0]:.4f}%")

# =============================================================================
# 7. Value at Risk (VaR)
# =============================================================================
print("\n7. VALUE AT RISK (VaR)")
print("-" * 40)

confidence_levels = [0.95, 0.99]
portfolio_value = 1000000  # $1 million

print(f"\n   Portfolio value: ${portfolio_value:,.0f}")
print(f"   Current volatility: {cond_vol.iloc[-1]:.4f}%")

for cl in confidence_levels:
    z_score = stats.norm.ppf(1 - cl)
    var_pct = -z_score * cond_vol.iloc[-1] / 100
    var_dollar = var_pct * portfolio_value
    print(f"\n   {cl*100:.0f}% 1-day VaR:")
    print(f"     Percentage: {var_pct*100:.2f}%")
    print(f"     Dollar amount: ${var_dollar:,.0f}")

# =============================================================================
# 8. Model Diagnostics
# =============================================================================
print("\n8. MODEL DIAGNOSTICS")
print("-" * 40)

std_resid = results.std_resid

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Standardized residuals
axes[0, 0].plot(df.index, std_resid, color='#1A3A6E', linewidth=0.5)
axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=0.8)
axes[0, 0].axhline(y=2, color='gray', linestyle=':', alpha=0.5)
axes[0, 0].axhline(y=-2, color='gray', linestyle=':', alpha=0.5)
axes[0, 0].set_title('Standardized Residuals', fontweight='bold')

# ACF of squared standardized residuals
plot_acf(std_resid**2, lags=20, ax=axes[0, 1], color='#1A3A6E')
axes[0, 1].set_title('ACF of z²ₜ (Check for remaining ARCH)', fontweight='bold')

# Histogram
axes[1, 0].hist(std_resid, bins=50, density=True, color='#1A3A6E', alpha=0.7, edgecolor='white')
x = np.linspace(-4, 4, 100)
axes[1, 0].plot(x, stats.norm.pdf(x), 'r-', lw=2, label='N(0,1)')
axes[1, 0].set_title('Standardized Residuals Distribution', fontweight='bold')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Q-Q plot
stats.probplot(std_resid, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot', fontweight='bold')

plt.tight_layout()
plt.savefig('ch5_diagnostics.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch5_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch5_diagnostics.pdf")

print("\n" + "=" * 70)
print("GARCH ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch5_stylized_facts.pdf: Volatility clustering evidence")
print("  - ch5_news_impact.pdf: News impact curves")
print("  - ch5_forecast.pdf: Volatility forecast")
print("  - ch5_diagnostics.pdf: Model diagnostics")
