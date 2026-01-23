#!/usr/bin/env python3
"""Generate charts for Chapter 10 slides"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Consistent style
plt.rcParams['figure.figsize'] = (10, 4)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

COLORS = {'blue': '#1A3A6E', 'red': '#DC3545', 'green': '#2E7D32',
          'orange': '#E67E22', 'gray': '#666666', 'purple': '#8E44AD'}

OUTPUT_DIR = 'charts/'

# =============================================================================
# 1. BITCOIN DATA AND CHARTS
# =============================================================================
print("Loading Bitcoin data...")
import yfinance as yf
btc = yf.download('BTC-USD', start='2019-01-01', end='2025-01-15', progress=False)
btc_returns = btc['Close'].pct_change() * 100

# Chart 1: Bitcoin returns showing volatility clustering
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(btc_returns.index, btc_returns.values, color=COLORS['blue'], linewidth=0.5)
ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
ax.set_title('Bitcoin Daily Returns: Volatility Clustering', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Return (%)')
ax.set_ylim(-25, 25)
plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{OUTPUT_DIR}btc_returns.pdf')
plt.savefig(f'{OUTPUT_DIR}btc_returns.png')
plt.close()
print("  - btc_returns.pdf")

# Chart 2: GARCH rolling forecast (simulated for illustration)
# We'll load the actual data from running GARCH
from arch import arch_model

btc_ret = btc_returns.dropna().values
n = len(btc_ret)
train_end = int(n * 0.85)
test_returns = btc_ret[train_end:]
test_dates = btc_returns.dropna().index[train_end:]

# Rolling one-step-ahead forecasts
print("  Computing GARCH rolling forecasts...")
all_returns = btc_ret
test_vol_forecast = np.zeros(len(test_returns))

for i in range(len(test_returns)):
    history = all_returns[:train_end + i]
    model = arch_model(history, mean='AR', lags=1, vol='Garch', p=1, q=1)
    res = model.fit(disp='off', show_warning=False)
    forecast = res.forecast(horizon=1, reindex=False)
    test_vol_forecast[i] = np.sqrt(forecast.variance.values[-1, 0])

realized_vol = np.abs(test_returns).flatten()

fig, ax = plt.subplots(figsize=(10, 4))
ax.fill_between(test_dates, 0, realized_vol, color=COLORS['blue'], alpha=0.4, label='Realized |Returns|')
ax.plot(test_dates, test_vol_forecast, color=COLORS['red'], linewidth=2, label='GARCH Rolling Forecast')
ax.set_title('GARCH(1,1): Rolling One-Step-Ahead Volatility Forecast', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Volatility')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=11)
plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.subplots_adjust(bottom=0.22)
plt.savefig(f'{OUTPUT_DIR}garch_forecast.pdf')
plt.savefig(f'{OUTPUT_DIR}garch_forecast.png')
plt.close()
print("  - garch_forecast.pdf")

# =============================================================================
# 2. SUNSPOT DATA AND CHARTS
# =============================================================================
print("Loading Sunspot data...")
import statsmodels.api as sm
sunspots = sm.datasets.sunspots.load_pandas().data
sunspots = sunspots[sunspots['YEAR'] >= 1900].copy()

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(sunspots['YEAR'], sunspots['SUNACTIVITY'], color=COLORS['blue'], linewidth=1.2)
ax.set_title('Yearly Sunspot Numbers: 11-Year Schwabe Cycle', fontweight='bold', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Sunspot Count')
# Add vertical lines at approximate cycle peaks
for year in [1906, 1917, 1928, 1937, 1947, 1958, 1968, 1979, 1989, 2000]:
    ax.axvline(x=year, color=COLORS['orange'], linestyle='--', alpha=0.3, linewidth=1)
plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{OUTPUT_DIR}sunspots.pdf')
plt.savefig(f'{OUTPUT_DIR}sunspots.png')
plt.close()
print("  - sunspots.pdf")

# =============================================================================
# 3. UNEMPLOYMENT DATA AND CHARTS
# =============================================================================
print("Loading Unemployment data...")
import pandas_datareader as pdr
unemp = pdr.get_data_fred('UNRATE', start='2010-01-01', end='2025-01-15')

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(unemp.index, unemp['UNRATE'], color=COLORS['blue'], linewidth=1.5)
ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-05-01'),
           alpha=0.3, color=COLORS['red'], label='COVID-19 Shock')
ax.set_title('US Unemployment Rate: COVID-19 Structural Break', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Unemployment Rate (%)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{OUTPUT_DIR}unemployment.pdf')
plt.savefig(f'{OUTPUT_DIR}unemployment.png')
plt.close()
print("  - unemployment.pdf")

# =============================================================================
# 4. ECONOMIC DATA (VAR)
# =============================================================================
print("Loading Economic data...")
gdp = pdr.get_data_fred('GDPC1', start='2000-01-01', end='2025-01-15')
unemp_q = pdr.get_data_fred('UNRATE', start='2000-01-01', end='2025-01-15')
inflation = pdr.get_data_fred('CPIAUCSL', start='2000-01-01', end='2025-01-15')
fed_rate = pdr.get_data_fred('FEDFUNDS', start='2000-01-01', end='2025-01-15')

# Calculate growth rates
gdp['gdp_growth'] = gdp['GDPC1'].pct_change(4) * 100
inflation['inflation'] = inflation['CPIAUCSL'].pct_change(12) * 100

fig, axes = plt.subplots(2, 2, figsize=(10, 6))

axes[0, 0].plot(gdp.index, gdp['gdp_growth'], color=COLORS['blue'], linewidth=1)
axes[0, 0].axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
axes[0, 0].set_title('GDP Growth (YoY %)', fontweight='bold')
axes[0, 0].set_ylabel('%')

axes[0, 1].plot(unemp_q.index, unemp_q['UNRATE'], color=COLORS['red'], linewidth=1)
axes[0, 1].set_title('Unemployment Rate (%)', fontweight='bold')
axes[0, 1].set_ylabel('%')

axes[1, 0].plot(inflation.index, inflation['inflation'], color=COLORS['green'], linewidth=1)
axes[1, 0].axhline(y=2, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
axes[1, 0].set_title('Inflation (CPI YoY %)', fontweight='bold')
axes[1, 0].set_ylabel('%')
axes[1, 0].set_xlabel('Date')

axes[1, 1].plot(fed_rate.index, fed_rate['FEDFUNDS'], color=COLORS['orange'], linewidth=1)
axes[1, 1].set_title('Federal Funds Rate (%)', fontweight='bold')
axes[1, 1].set_ylabel('%')
axes[1, 1].set_xlabel('Date')

# Add recession shading
for ax in axes.flat:
    ax.axvspan(pd.Timestamp('2008-01-01'), pd.Timestamp('2009-06-01'),
               alpha=0.1, color='gray')
    ax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-04-01'),
               alpha=0.1, color='red')

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{OUTPUT_DIR}economic_vars.pdf')
plt.savefig(f'{OUTPUT_DIR}economic_vars.png')
plt.close()
print("  - economic_vars.pdf")

# =============================================================================
# 5. MODEL COMPARISON VISUAL
# =============================================================================
print("Creating model comparison chart...")

models = ['GARCH(1,1)', 'GARCH(2,1)', 'GJR-GARCH']
val_mae = [2.63, 2.63, 2.66]
colors_bar = [COLORS['green'], COLORS['blue'], COLORS['blue']]

fig, ax = plt.subplots(figsize=(8, 3.5))
bars = ax.barh(models, val_mae, color=colors_bar, height=0.5)
ax.set_xlabel('Validation MAE')
ax.set_title('GARCH Model Comparison (Validation Set)', fontweight='bold', fontsize=14)
ax.set_xlim(2.5, 2.75)
bars[0].set_color(COLORS['green'])
ax.annotate('Best', xy=(val_mae[0], 0), xytext=(val_mae[0]+0.02, 0),
            fontsize=11, color=COLORS['green'], fontweight='bold', va='center')
plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{OUTPUT_DIR}garch_comparison.pdf')
plt.savefig(f'{OUTPUT_DIR}garch_comparison.png')
plt.close()
print("  - garch_comparison.pdf")

# =============================================================================
# 6. ROLLING VS MULTI-STEP ILLUSTRATION
# =============================================================================
print("Creating rolling vs multi-step illustration...")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# Multi-step (converges to flat line)
t = np.arange(100)
multi_step = 3 + 2 * np.exp(-t/15)  # Converges to 3
axes[0].plot(t, multi_step, color=COLORS['red'], linewidth=2)
axes[0].axhline(y=3, color='black', linestyle='--', alpha=0.5, label='Unconditional $\\bar{\\sigma}^2$')
axes[0].set_title('Multi-Step Forecast', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Forecast Horizon (h)')
axes[0].set_ylabel('$\\sigma^2_{t+h}$')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
axes[0].set_ylim(2.5, 5.5)

# Rolling one-step (dynamic)
np.random.seed(42)
rolling = 3 + np.cumsum(np.random.randn(100) * 0.3)
rolling = np.clip(rolling, 1.5, 6)
axes[1].plot(t, rolling, color=COLORS['green'], linewidth=2)
axes[1].set_title('Rolling One-Step-Ahead', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Time (t)')
axes[1].set_ylabel('$\\sigma^2_{t+1|t}$')
axes[1].set_ylim(2.5, 5.5)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{OUTPUT_DIR}rolling_vs_multistep.pdf')
plt.savefig(f'{OUTPUT_DIR}rolling_vs_multistep.png')
plt.close()
print("  - rolling_vs_multistep.pdf")

print("\nAll charts generated successfully!")
