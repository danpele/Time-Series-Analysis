"""
Generate charts for GARCH Volatility Models Chapter
Using REAL financial data from Yahoo Finance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance and arch
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance', '-q'])
    import yfinance as yf
    HAS_YFINANCE = True

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    print("Installing arch...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'arch', '-q'])
    from arch import arch_model
    HAS_ARCH = True

from statsmodels.graphics.tsaplots import plot_acf

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
# Download Real Financial Data
#=============================================================================
print("Downloading real financial data...")

# S&P 500 - long history
sp500 = yf.download('^GSPC', start='2000-01-01', end='2024-12-31', progress=False)
sp500_close = sp500['Close'].squeeze() if isinstance(sp500['Close'], pd.DataFrame) else sp500['Close']
sp500_returns = (sp500_close.pct_change() * 100).dropna()
sp500_returns = pd.Series(sp500_returns.values, index=sp500_returns.index, name='returns')

# Bitcoin - for comparison
btc = yf.download('BTC-USD', start='2015-01-01', end='2024-12-31', progress=False)
btc_close = btc['Close'].squeeze() if isinstance(btc['Close'], pd.DataFrame) else btc['Close']
btc_returns = (btc_close.pct_change() * 100).dropna()
btc_returns = pd.Series(btc_returns.values, index=btc_returns.index, name='returns')

# EUR/USD
eurusd = yf.download('EURUSD=X', start='2010-01-01', end='2024-12-31', progress=False)
eurusd_close = eurusd['Close'].squeeze() if isinstance(eurusd['Close'], pd.DataFrame) else eurusd['Close']
eurusd_returns = (eurusd_close.pct_change() * 100).dropna()
eurusd_returns = pd.Series(eurusd_returns.values, index=eurusd_returns.index, name='returns')

print(f"S&P 500: {len(sp500_returns)} observations ({sp500_returns.index[0].strftime('%Y-%m-%d')} to {sp500_returns.index[-1].strftime('%Y-%m-%d')})")
print(f"Bitcoin: {len(btc_returns)} observations")
print(f"EUR/USD: {len(eurusd_returns)} observations")

#=============================================================================
# Chart 1: Real Volatility Clustering - S&P 500
#=============================================================================
print("\nGenerating Chart 1: Real Volatility Clustering (S&P 500)...")

# Fit GARCH(1,1) model
model = arch_model(sp500_returns, vol='Garch', p=1, q=1, dist='normal')
results = model.fit(disp='off')
conditional_vol = results.conditional_volatility

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Returns
axes[0].plot(sp500_returns.index, sp500_returns.values, color=COLORS['blue'], linewidth=0.5, alpha=0.8)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_ylabel('Daily Return (%)')
axes[0].set_title('S&P 500 - Daily Returns (Real Data)', fontweight='bold', fontsize=12)

# Highlight crisis periods
crisis_periods = [
    ('2008-09-01', '2009-03-31', '2008 Crisis'),
    ('2020-02-20', '2020-04-30', 'COVID-19'),
    ('2022-01-01', '2022-10-31', '2022'),
]
for start, end, label in crisis_periods:
    axes[0].axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2, color=COLORS['red'])
    axes[1].axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.2, color=COLORS['red'])

# Conditional volatility
axes[1].plot(sp500_returns.index, conditional_vol, color=COLORS['red'], linewidth=0.8)
axes[1].fill_between(sp500_returns.index, 0, conditional_vol, color=COLORS['red'], alpha=0.3)
axes[1].set_ylabel('Conditional Volatility (%)')
axes[1].set_xlabel('Date')
axes[1].set_title('Volatility Clustering - GARCH(1,1)', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('charts/garch_volatility_clustering.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_volatility_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_volatility_clustering.pdf")

#=============================================================================
# Chart 2: Stylized Facts - Real S&P 500 Data
#=============================================================================
print("Generating Chart 2: Stylized Facts (S&P 500)...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ACF of returns
plot_acf(sp500_returns.values, lags=30, ax=axes[0, 0], color=COLORS['blue'],
         vlines_kwargs={'color': COLORS['blue']}, alpha=0.05)
axes[0, 0].set_title('ACF of S&P 500 Returns', fontweight='bold')
axes[0, 0].set_xlabel('Lag')

# ACF of squared returns
plot_acf(sp500_returns.values**2, lags=30, ax=axes[0, 1], color=COLORS['red'],
         vlines_kwargs={'color': COLORS['red']}, alpha=0.05)
axes[0, 1].set_title('ACF of Squared Returns (Volatility Clustering)', fontweight='bold')
axes[0, 1].set_xlabel('Lag')

# Histogram with normal overlay
axes[1, 0].hist(sp500_returns.values, bins=100, density=True, color=COLORS['blue'],
                alpha=0.7, edgecolor='white', label='S&P 500')
x = np.linspace(sp500_returns.min(), sp500_returns.max(), 200)
axes[1, 0].plot(x, stats.norm.pdf(x, sp500_returns.mean(), sp500_returns.std()),
                color=COLORS['red'], linewidth=2, label='Normal')
axes[1, 0].set_title('Distribution: Fat Tails (Leptokurtic)', fontweight='bold')
axes[1, 0].set_xlabel('Return (%)')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_xlim(-10, 10)
kurtosis = float(stats.kurtosis(sp500_returns.values))
skewness = float(stats.skew(sp500_returns.values))
axes[1, 0].text(0.95, 0.95, f'Kurtosis: {kurtosis:.2f}\nSkewness: {skewness:.2f}',
                transform=axes[1, 0].transAxes, fontsize=10, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=2)

# QQ-plot
returns_flat = sp500_returns.values.flatten()
stats.probplot(returns_flat, dist="norm", plot=axes[1, 1])
axes[1, 1].get_lines()[0].set_color(COLORS['blue'])
axes[1, 1].get_lines()[0].set_markersize(2)
axes[1, 1].get_lines()[1].set_color(COLORS['red'])
axes[1, 1].set_title('QQ-Plot vs Normal', fontweight='bold')

plt.tight_layout()
plt.savefig('charts/garch_stylized_facts.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_stylized_facts.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_stylized_facts.pdf")

#=============================================================================
# Chart 3: GARCH Parameter Comparison
#=============================================================================
print("Generating Chart 3: GARCH Model Estimation Results...")

# Estimate different GARCH specifications
models_results = {}

# GARCH(1,1)
model_garch = arch_model(sp500_returns, vol='Garch', p=1, q=1, dist='normal')
res_garch = model_garch.fit(disp='off')
models_results['GARCH(1,1)'] = res_garch

# GJR-GARCH(1,1)
model_gjr = arch_model(sp500_returns, vol='Garch', p=1, o=1, q=1, dist='normal')
res_gjr = model_gjr.fit(disp='off')
models_results['GJR-GARCH'] = res_gjr

# EGARCH(1,1)
model_egarch = arch_model(sp500_returns, vol='EGARCH', p=1, q=1, dist='normal')
res_egarch = model_egarch.fit(disp='off')
models_results['EGARCH'] = res_egarch

fig, ax = plt.subplots(figsize=(14, 6))

colors = [COLORS['blue'], COLORS['red'], COLORS['green']]
for i, (name, res) in enumerate(models_results.items()):
    vol = res.conditional_volatility
    ax.plot(sp500_returns.index[-1000:], vol[-1000:], linewidth=1.2,
            color=colors[i], label=name, alpha=0.8)

ax.set_xlabel('Date')
ax.set_ylabel('Conditional Volatility (%)')
ax.set_title('GARCH Models Comparison on S&P 500 (Last 1000 Days)', fontweight='bold', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=3)

plt.tight_layout()
plt.savefig('charts/garch_conditional_variance.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_conditional_variance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_conditional_variance.pdf")

#=============================================================================
# Chart 4: Leverage Effect - Real Data
#=============================================================================
print("Generating Chart 4: Leverage Effect (S&P 500)...")

# Calculate realized volatility (rolling std)
window = 22  # ~1 month
realized_vol = sp500_returns.rolling(window=window).std()

# Lag returns
lagged_returns = sp500_returns.shift(1)

# Create DataFrame for analysis
df_lev = pd.DataFrame({
    'returns': sp500_returns,
    'lagged_returns': lagged_returns,
    'future_vol': realized_vol.shift(-window)
}).dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
colors = [COLORS['red'] if r < 0 else COLORS['blue'] for r in df_lev['lagged_returns']]
axes[0].scatter(df_lev['lagged_returns'], df_lev['future_vol'],
                c=colors, alpha=0.3, s=10)
axes[0].set_xlabel('Return at t (%)')
axes[0].set_ylabel('Realized Volatility at t+22 (%)')
axes[0].set_title('Leverage Effect: Negative vs Positive Shocks', fontweight='bold')
axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
axes[0].set_xlim(-10, 10)

# Box plot comparison
neg_mask = df_lev['lagged_returns'] < -1
pos_mask = df_lev['lagged_returns'] > 1
neutral_mask = (df_lev['lagged_returns'] >= -1) & (df_lev['lagged_returns'] <= 1)

bp = axes[1].boxplot([df_lev.loc[neg_mask, 'future_vol'].values,
                       df_lev.loc[neutral_mask, 'future_vol'].values,
                       df_lev.loc[pos_mask, 'future_vol'].values],
                      labels=['Negative Shock\n(r < -1%)', 'Neutral\n(-1% < r < 1%)', 'Positive Shock\n(r > 1%)'],
                      patch_artist=True)
bp['boxes'][0].set_facecolor(COLORS['red'])
bp['boxes'][0].set_alpha(0.6)
bp['boxes'][1].set_facecolor(COLORS['gray'])
bp['boxes'][1].set_alpha(0.6)
bp['boxes'][2].set_facecolor(COLORS['blue'])
bp['boxes'][2].set_alpha(0.6)
axes[1].set_ylabel('Future Volatility (%)')
axes[1].set_title('Volatility Distribution by Shock Type', fontweight='bold')

plt.tight_layout()
plt.savefig('charts/garch_leverage_effect.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_leverage_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_leverage_effect.pdf")

#=============================================================================
# Chart 5: News Impact Curve - Estimated from Real Data
#=============================================================================
print("Generating Chart 5: News Impact Curve...")

# Get parameters from fitted models
omega_g = res_garch.params['omega']
alpha_g = res_garch.params['alpha[1]']
beta_g = res_garch.params['beta[1]']

omega_gjr = res_gjr.params['omega']
alpha_gjr = res_gjr.params['alpha[1]']
gamma_gjr = res_gjr.params['gamma[1]']
beta_gjr = res_gjr.params['beta[1]']

# Previous variance (use unconditional)
sigma2_prev_g = omega_g / (1 - alpha_g - beta_g)
sigma2_prev_gjr = omega_gjr / (1 - alpha_gjr - gamma_gjr/2 - beta_gjr)

# Range of shocks
epsilon_range = np.linspace(-5, 5, 200)

# GARCH (symmetric)
sigma2_garch = omega_g + alpha_g * epsilon_range**2 + beta_g * sigma2_prev_g

# GJR-GARCH (asymmetric)
indicator = (epsilon_range < 0).astype(float)
sigma2_gjr = omega_gjr + alpha_gjr * epsilon_range**2 + gamma_gjr * epsilon_range**2 * indicator + beta_gjr * sigma2_prev_gjr

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(epsilon_range, np.sqrt(sigma2_garch), color=COLORS['blue'], linewidth=2.5,
        label=f'GARCH(1,1): α={alpha_g:.3f}, β={beta_g:.3f}')
ax.plot(epsilon_range, np.sqrt(sigma2_gjr), color=COLORS['red'], linewidth=2.5,
        label=f'GJR-GARCH: α={alpha_gjr:.3f}, γ={gamma_gjr:.3f}, β={beta_gjr:.3f}')

ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('Shock εₜ₋₁ (%)', fontsize=12)
ax.set_ylabel('Conditional Volatility σₜ (%)', fontsize=12)
ax.set_title('News Impact Curve - Estimated on S&P 500', fontweight='bold', fontsize=14)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=2)

# Add text annotation
ax.text(0.02, 0.98, 'Negative shocks have greater\nimpact on volatility',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('charts/garch_news_impact_curve.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_news_impact_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_news_impact_curve.pdf")

#=============================================================================
# Chart 6: Model Diagnostics - Real Estimation
#=============================================================================
print("Generating Chart 6: Model Diagnostics...")

std_resid = res_garch.std_resid

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Time series of standardized residuals
axes[0, 0].plot(sp500_returns.index, std_resid, color=COLORS['blue'], linewidth=0.5, alpha=0.8)
axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
axes[0, 0].axhline(y=2, color=COLORS['red'], linestyle='--', linewidth=0.8, alpha=0.5)
axes[0, 0].axhline(y=-2, color=COLORS['red'], linestyle='--', linewidth=0.8, alpha=0.5)
axes[0, 0].set_title('Standardized Residuals GARCH(1,1)', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('zₜ = εₜ/σₜ')

# ACF of squared standardized residuals
plot_acf(std_resid**2, lags=30, ax=axes[0, 1], color=COLORS['blue'],
         vlines_kwargs={'color': COLORS['blue']}, alpha=0.05)
axes[0, 1].set_title('ACF(zₜ²) - Check for Residual ARCH Effects', fontweight='bold')
axes[0, 1].set_xlabel('Lag')

# Histogram
axes[1, 0].hist(std_resid, bins=80, density=True, color=COLORS['blue'],
                alpha=0.7, edgecolor='white', label='Std. Residuals')
x = np.linspace(-5, 5, 100)
axes[1, 0].plot(x, stats.norm.pdf(x), color=COLORS['red'],
                linewidth=2, label='N(0,1)')
# Add t-distribution
df_est = 2 / (stats.kurtosis(std_resid) - 3 + 2) if stats.kurtosis(std_resid) > 3 else 30
axes[1, 0].plot(x, stats.t.pdf(x, df=5), color=COLORS['green'],
                linewidth=2, linestyle='--', label='t(5)')
axes[1, 0].set_title('Standardized Residuals Distribution', fontweight='bold')
axes[1, 0].set_xlabel('zₜ')
axes[1, 0].set_xlim(-5, 5)
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False, ncol=3)

# QQ-plot
stats.probplot(std_resid, dist="norm", plot=axes[1, 1])
axes[1, 1].get_lines()[0].set_color(COLORS['blue'])
axes[1, 1].get_lines()[0].set_markersize(2)
axes[1, 1].get_lines()[1].set_color(COLORS['red'])
axes[1, 1].set_title('QQ-Plot Residuals vs Normal', fontweight='bold')

plt.tight_layout()
plt.savefig('charts/garch_diagnostics.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_diagnostics.pdf")

#=============================================================================
# Chart 7: Volatility Forecast
#=============================================================================
print("Generating Chart 7: Volatility Forecast...")

# Forecast next 30 days
forecast_horizon = 30
forecasts = res_garch.forecast(horizon=forecast_horizon)
forecast_var = forecasts.variance.iloc[-1].values
forecast_vol = np.sqrt(forecast_var)

# Historical volatility (last 250 days)
hist_vol = conditional_vol[-250:]
hist_dates = sp500_returns.index[-250:]

# Create forecast dates
last_date = sp500_returns.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')

fig, ax = plt.subplots(figsize=(14, 6))

# Historical
ax.plot(hist_dates, hist_vol, color=COLORS['blue'], linewidth=1.2, label='Historical Volatility')

# Forecast
ax.plot(forecast_dates, forecast_vol, color=COLORS['red'], linewidth=2.5,
        linestyle='--', label='GARCH(1,1) Forecast')

# Unconditional volatility
unconditional_vol = np.sqrt(omega_g / (1 - alpha_g - beta_g))
ax.axhline(y=unconditional_vol, color=COLORS['green'], linestyle=':', linewidth=1.5,
           label=f'σ̄ (unconditional) = {unconditional_vol:.2f}%')

# Confidence interval (approximate)
ax.fill_between(forecast_dates, forecast_vol * 0.7, forecast_vol * 1.3,
                color=COLORS['red'], alpha=0.2, label='70% Interval')

ax.axvline(x=last_date, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.text(last_date, ax.get_ylim()[1]*0.95, ' Forecast →', fontsize=10)

ax.set_xlabel('Date')
ax.set_ylabel('Volatility (%)')
ax.set_title(f'GARCH(1,1) Volatility Forecast - S&P 500 ({forecast_horizon} days)',
             fontweight='bold', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=4)

plt.tight_layout()
plt.savefig('charts/garch_forecast.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_forecast.pdf")

#=============================================================================
# Chart 8: S&P 500 Full History with Major Events
#=============================================================================
print("Generating Chart 8: S&P 500 Full History...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Returns
axes[0].plot(sp500_returns.index, sp500_returns.values, color=COLORS['blue'],
             linewidth=0.4, alpha=0.8)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_ylabel('Daily Return (%)')
axes[0].set_title('S&P 500 - Daily Returns (2000-2024)', fontweight='bold', fontsize=12)

# Add annotations for major events
events = [
    ('2001-09-11', 'Sep 11', -5),
    ('2008-10-15', 'Financial Crisis', 8),
    ('2010-05-06', 'Flash Crash', -5),
    ('2020-03-16', 'COVID-19', -12),
    ('2022-06-13', 'Inflation 2022', -4),
]
for date, label, y_offset in events:
    try:
        idx = pd.Timestamp(date)
        if idx in sp500_returns.index:
            val = sp500_returns.loc[idx]
            axes[0].annotate(label, xy=(idx, val), xytext=(idx, val + y_offset),
                           fontsize=8, color=COLORS['red'], fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=0.5))
    except:
        pass

# Volatility
axes[1].plot(sp500_returns.index, conditional_vol, color=COLORS['red'], linewidth=0.6)
axes[1].fill_between(sp500_returns.index, 0, conditional_vol, color=COLORS['red'], alpha=0.3)
axes[1].set_ylabel('GARCH Volatility (%)')
axes[1].set_xlabel('Date')
axes[1].set_title('Conditional Volatility GARCH(1,1)', fontweight='bold', fontsize=12)

# Add VIX-like threshold
axes[1].axhline(y=2, color=COLORS['green'], linestyle='--', linewidth=1, alpha=0.7, label='Normal level (~2%)')
axes[1].axhline(y=4, color=COLORS['orange'], linestyle='--', linewidth=1, alpha=0.7, label='High level (~4%)')

plt.tight_layout()
plt.savefig('charts/garch_sp500_returns.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_sp500_returns.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_sp500_returns.pdf")

#=============================================================================
# Chart 9: Bitcoin vs S&P 500 Volatility Comparison
#=============================================================================
print("Generating Chart 9: Bitcoin vs S&P 500 Volatility...")

# Fit GARCH to Bitcoin
model_btc = arch_model(btc_returns.dropna(), vol='Garch', p=1, q=1, dist='normal')
res_btc = model_btc.fit(disp='off')
btc_vol = res_btc.conditional_volatility

# Common date range
common_start = max(sp500_returns.index[0], btc_returns.index[0])
common_end = min(sp500_returns.index[-1], btc_returns.index[-1])

sp500_common = conditional_vol[(sp500_returns.index >= common_start) & (sp500_returns.index <= common_end)]
btc_common = btc_vol[(btc_returns.index >= common_start) & (btc_returns.index <= common_end)]

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(sp500_common.index, sp500_common.values, color=COLORS['blue'],
        linewidth=1, label='S&P 500', alpha=0.8)
ax.plot(btc_common.index, btc_common.values, color=COLORS['orange'],
        linewidth=1, label='Bitcoin', alpha=0.8)

ax.set_xlabel('Date')
ax.set_ylabel('GARCH(1,1) Volatility (%)')
ax.set_title('Volatility Comparison: S&P 500 vs Bitcoin', fontweight='bold', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False, ncol=2)

# Add statistics
sp500_mean_vol = sp500_common.mean()
btc_mean_vol = btc_common.mean()
ax.text(0.02, 0.98, f'Mean Volatility:\nS&P 500: {sp500_mean_vol:.2f}%\nBitcoin: {btc_mean_vol:.2f}%',
        transform=ax.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('charts/garch_sp500_volatility.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_sp500_volatility.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_sp500_volatility.pdf")

#=============================================================================
# Chart 10: Model Comparison Summary
#=============================================================================
print("Generating Chart 10: Model Comparison Summary...")

# Compare models with different distributions
models_compare = {}

# Normal distribution
model_n = arch_model(sp500_returns, vol='Garch', p=1, q=1, dist='normal')
res_n = model_n.fit(disp='off')
models_compare['GARCH-Normal'] = {'AIC': res_n.aic, 'BIC': res_n.bic, 'LL': res_n.loglikelihood}

# Student-t distribution
model_t = arch_model(sp500_returns, vol='Garch', p=1, q=1, dist='t')
res_t = model_t.fit(disp='off')
models_compare['GARCH-t'] = {'AIC': res_t.aic, 'BIC': res_t.bic, 'LL': res_t.loglikelihood}

# GJR-GARCH with t
model_gjr_t = arch_model(sp500_returns, vol='Garch', p=1, o=1, q=1, dist='t')
res_gjr_t = model_gjr_t.fit(disp='off')
models_compare['GJR-GARCH-t'] = {'AIC': res_gjr_t.aic, 'BIC': res_gjr_t.bic, 'LL': res_gjr_t.loglikelihood}

# EGARCH with t
model_e_t = arch_model(sp500_returns, vol='EGARCH', p=1, q=1, dist='t')
res_e_t = model_e_t.fit(disp='off')
models_compare['EGARCH-t'] = {'AIC': res_e_t.aic, 'BIC': res_e_t.bic, 'LL': res_e_t.loglikelihood}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AIC/BIC comparison
models_names = list(models_compare.keys())
aic_values = [models_compare[m]['AIC'] for m in models_names]
bic_values = [models_compare[m]['BIC'] for m in models_names]

x = np.arange(len(models_names))
width = 0.35

bars1 = axes[0].bar(x - width/2, aic_values, width, label='AIC', color=COLORS['blue'], alpha=0.8)
bars2 = axes[0].bar(x + width/2, bic_values, width, label='BIC', color=COLORS['red'], alpha=0.8)

axes[0].set_ylabel('Criterion Value')
axes[0].set_title('GARCH Models Comparison - Information Criteria', fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_names, rotation=15, ha='right')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)

# Highlight best model
min_aic_idx = np.argmin(aic_values)
axes[0].annotate('Best', xy=(x[min_aic_idx] - width/2, aic_values[min_aic_idx]),
                xytext=(x[min_aic_idx] - width/2, aic_values[min_aic_idx] - 500),
                fontsize=9, color=COLORS['green'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['green']))

# Volatility comparison for last 500 days
last_n = 500
axes[1].plot(sp500_returns.index[-last_n:], res_n.conditional_volatility[-last_n:],
             color=COLORS['blue'], linewidth=1, label='GARCH-Normal', alpha=0.7)
axes[1].plot(sp500_returns.index[-last_n:], res_t.conditional_volatility[-last_n:],
             color=COLORS['red'], linewidth=1, label='GARCH-t', alpha=0.7)
axes[1].plot(sp500_returns.index[-last_n:], res_gjr_t.conditional_volatility[-last_n:],
             color=COLORS['green'], linewidth=1, label='GJR-GARCH-t', alpha=0.7)

axes[1].set_xlabel('Date')
axes[1].set_ylabel('Volatility (%)')
axes[1].set_title('Estimated Volatility - Last 500 Days', fontweight='bold')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)

plt.tight_layout()
plt.savefig('charts/garch_sp500_comparison.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/garch_sp500_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: garch_sp500_comparison.pdf")

#=============================================================================
# Print Model Summary
#=============================================================================
print("\n" + "="*60)
print("GARCH Model Estimation Results (S&P 500)")
print("="*60)
print(f"\nGARCH(1,1) Parameters:")
print(f"  ω (omega) = {res_garch.params['omega']:.6f}")
print(f"  α (alpha) = {res_garch.params['alpha[1]']:.4f}")
print(f"  β (beta)  = {res_garch.params['beta[1]']:.4f}")
print(f"  α + β     = {res_garch.params['alpha[1]'] + res_garch.params['beta[1]']:.4f}")
print(f"  Unconditional volatility = {np.sqrt(omega_g / (1 - alpha_g - beta_g)):.4f}%")
print(f"\nGJR-GARCH Parameters:")
print(f"  γ (gamma) = {res_gjr.params['gamma[1]']:.4f} (leverage effect)")
print(f"\nModel Fit:")
print(f"  GARCH(1,1) AIC: {res_garch.aic:.2f}")
print(f"  GJR-GARCH  AIC: {res_gjr.aic:.2f}")
print(f"  EGARCH     AIC: {res_egarch.aic:.2f}")

print("\n" + "="*60)
print("All GARCH charts generated successfully with REAL DATA!")
print("="*60)
