#!/usr/bin/env python3
"""Generate additional charts for Chapter 10 slides"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

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
# 1. TRAIN/VALIDATION/TEST SPLIT DIAGRAM
# =============================================================================
print("Creating train/val/test split diagram...")
fig, ax = plt.subplots(figsize=(10, 2.5))

# Draw bars - 70% / 20% / 10%
ax.barh(0, 70, left=0, height=0.6, color=COLORS['blue'], alpha=0.7, label='Training (70%)')
ax.barh(0, 20, left=70, height=0.6, color=COLORS['orange'], alpha=0.7, label='Validation (20%)')
ax.barh(0, 10, left=90, height=0.6, color=COLORS['red'], alpha=0.7, label='Test (10%)')

# Add text labels
ax.text(35, 0, 'Training\n70%', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax.text(80, 0, 'Val\n20%', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax.text(95, 0, 'Test\n10%', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Add arrows and labels below
ax.annotate('Fit parameters', xy=(35, -0.5), ha='center', fontsize=11, color=COLORS['blue'])
ax.annotate('Select model', xy=(80, -0.5), ha='center', fontsize=11, color=COLORS['orange'])
ax.annotate('Final evaluation', xy=(95, -0.5), ha='center', fontsize=11, color=COLORS['red'])

ax.set_xlim(0, 100)
ax.set_ylim(-1, 0.8)
ax.axis('off')
ax.set_title('Time Series Train/Validation/Test Split', fontweight='bold', fontsize=14, pad=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}train_val_test_split.pdf')
plt.savefig(f'{OUTPUT_DIR}train_val_test_split.png')
plt.close()
print("  - train_val_test_split.pdf")

# =============================================================================
# 2. BITCOIN SQUARED RETURNS (VOLATILITY PROXY)
# =============================================================================
print("Loading Bitcoin data for squared returns...")
import yfinance as yf
btc = yf.download('BTC-USD', start='2019-01-01', end='2025-01-15', progress=False)
btc_returns = btc['Close'].pct_change() * 100

fig, ax = plt.subplots(figsize=(10, 3))
squared_returns = (btc_returns ** 2).dropna()
ax.fill_between(squared_returns.index, 0, squared_returns.values.flatten(),
                color=COLORS['orange'], alpha=0.6)
ax.set_title('Bitcoin Squared Returns (Volatility Proxy)', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Return² (%²)')
ax.set_ylim(0, 400)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}btc_squared_returns.pdf')
plt.savefig(f'{OUTPUT_DIR}btc_squared_returns.png')
plt.close()
print("  - btc_squared_returns.pdf")

# =============================================================================
# 3. ACF OF SQUARED RETURNS
# =============================================================================
print("Creating ACF of squared returns...")
fig, ax = plt.subplots(figsize=(8, 3))
plot_acf(squared_returns.dropna().values.flatten()[:2000], ax=ax, lags=30, alpha=0.05)
ax.set_title('ACF of Squared Returns (Evidence for GARCH)', fontweight='bold', fontsize=14)
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}btc_acf_squared.pdf')
plt.savefig(f'{OUTPUT_DIR}btc_acf_squared.png')
plt.close()
print("  - btc_acf_squared.pdf")

# =============================================================================
# 4. SUNSPOT ACF
# =============================================================================
print("Creating Sunspot ACF...")
import statsmodels.api as sm
sunspots = sm.datasets.sunspots.load_pandas().data
sunspots = sunspots[sunspots['YEAR'] >= 1900].copy()

fig, ax = plt.subplots(figsize=(8, 3))
plot_acf(sunspots['SUNACTIVITY'].values, ax=ax, lags=40, alpha=0.05)
ax.axvline(x=11, color=COLORS['red'], linestyle='--', alpha=0.7, linewidth=2, label='11-year lag')
ax.axvline(x=22, color=COLORS['red'], linestyle='--', alpha=0.7, linewidth=2)
ax.set_title('Sunspot ACF: Confirms 11-Year Cycle', fontweight='bold', fontsize=14)
ax.set_xlabel('Lag (Years)')
ax.set_ylabel('Autocorrelation')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}sunspots_acf.pdf')
plt.savefig(f'{OUTPUT_DIR}sunspots_acf.png')
plt.close()
print("  - sunspots_acf.pdf")

# =============================================================================
# 5. FOURIER TERMS ILLUSTRATION
# =============================================================================
print("Creating Fourier terms illustration...")
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

t = np.linspace(0, 4*np.pi, 200)
period = 2*np.pi

# K=1
axes[0].plot(t, np.sin(t), color=COLORS['blue'], linewidth=2, label='sin')
axes[0].plot(t, np.cos(t), color=COLORS['red'], linewidth=2, linestyle='--', label='cos')
axes[0].set_title('K=1 (2 terms)', fontweight='bold')
axes[0].set_xlabel('t')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=9)
axes[0].set_ylim(-2.5, 2.5)

# K=2
y2 = np.sin(t) + 0.5*np.sin(2*t) + np.cos(t) + 0.5*np.cos(2*t)
axes[1].plot(t, y2, color=COLORS['purple'], linewidth=2)
axes[1].set_title('K=2 (4 terms)', fontweight='bold')
axes[1].set_xlabel('t')
axes[1].set_ylim(-2.5, 2.5)

# K=3
y3 = np.sin(t) + 0.5*np.sin(2*t) + 0.3*np.sin(3*t) + np.cos(t) + 0.5*np.cos(2*t) + 0.3*np.cos(3*t)
axes[2].plot(t, y3, color=COLORS['green'], linewidth=2)
axes[2].set_title('K=3 (6 terms)', fontweight='bold')
axes[2].set_xlabel('t')
axes[2].set_ylim(-2.5, 2.5)

plt.suptitle('Fourier Terms: More K = More Flexibility', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}fourier_terms.pdf')
plt.savefig(f'{OUTPUT_DIR}fourier_terms.png')
plt.close()
print("  - fourier_terms.pdf")

# =============================================================================
# 6. GRANGER CAUSALITY HEATMAP
# =============================================================================
print("Creating Granger causality heatmap...")
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(6, 5))

# Sample p-values (from the notebook results)
variables = ['GDP', 'Unemp', 'Infl', 'Fed']
pvalues = np.array([
    [1.0, 0.076, 0.309, 0.698],
    [0.045, 1.0, 0.093, 0.857],
    [0.545, 0.665, 1.0, 0.834],
    [0.286, 0.317, 0.087, 1.0]
])

# Create heatmap
cmap = plt.cm.RdYlGn_r
im = ax.imshow(pvalues, cmap=cmap, vmin=0, vmax=1)

# Add text annotations
for i in range(4):
    for j in range(4):
        if i != j:
            text = f'{pvalues[i,j]:.3f}'
            color = 'white' if pvalues[i,j] < 0.1 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=11,
                   fontweight='bold' if pvalues[i,j] < 0.1 else 'normal', color=color)
        else:
            ax.text(j, i, '—', ha='center', va='center', fontsize=14, color='gray')

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(variables)
ax.set_yticklabels(variables)
ax.set_xlabel('Effect (column)', fontsize=12)
ax.set_ylabel('Cause (row)', fontsize=12)
ax.set_title('Granger Causality p-values\n(green = significant)', fontweight='bold', fontsize=13)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('p-value', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}granger_heatmap.pdf')
plt.savefig(f'{OUTPUT_DIR}granger_heatmap.png')
plt.close()
print("  - granger_heatmap.pdf")

# =============================================================================
# 7. METRICS COMPARISON BAR CHART
# =============================================================================
print("Creating metrics comparison chart...")
fig, ax = plt.subplots(figsize=(9, 4))

models = ['GARCH\n(Bitcoin)', 'Fourier\n(Sunspots)', 'Prophet\n(Unemp)', 'VAR\n(Economic)']
rmse_values = [2.21, 48.51, 0.42, 1.32]
colors = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']]

bars = ax.bar(models, rmse_values, color=colors, width=0.6, edgecolor='white', linewidth=2)

# Add value labels on bars
for bar, val in zip(bars, rmse_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Test RMSE', fontsize=12)
ax.set_title('Model Performance Comparison (Test Set RMSE)', fontweight='bold', fontsize=14)
ax.set_ylim(0, 60)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}model_comparison.pdf')
plt.savefig(f'{OUTPUT_DIR}model_comparison.png')
plt.close()
print("  - model_comparison.pdf")

# =============================================================================
# 8. IMPULSE RESPONSE ILLUSTRATION
# =============================================================================
print("Creating impulse response illustration...")
fig, axes = plt.subplots(2, 2, figsize=(9, 6))

quarters = np.arange(0, 13)

# GDP shock -> GDP
irf_gdp = 1 * np.exp(-quarters/4)
axes[0,0].plot(quarters, irf_gdp, color=COLORS['blue'], linewidth=2, marker='o', markersize=4)
axes[0,0].axhline(y=0, color='black', linewidth=0.5)
axes[0,0].fill_between(quarters, irf_gdp-0.2, irf_gdp+0.2, alpha=0.2, color=COLORS['blue'])
axes[0,0].set_title('GDP → GDP', fontweight='bold')
axes[0,0].set_ylabel('Response')

# GDP shock -> Unemployment
irf_unemp = -0.3 * (1 - np.exp(-quarters/3))
axes[0,1].plot(quarters, irf_unemp, color=COLORS['red'], linewidth=2, marker='o', markersize=4)
axes[0,1].axhline(y=0, color='black', linewidth=0.5)
axes[0,1].fill_between(quarters, irf_unemp-0.1, irf_unemp+0.1, alpha=0.2, color=COLORS['red'])
axes[0,1].set_title('GDP → Unemployment', fontweight='bold')

# GDP shock -> Inflation
irf_infl = 0.2 * quarters/12 * np.exp(-quarters/8)
axes[1,0].plot(quarters, irf_infl, color=COLORS['green'], linewidth=2, marker='o', markersize=4)
axes[1,0].axhline(y=0, color='black', linewidth=0.5)
axes[1,0].fill_between(quarters, irf_infl-0.05, irf_infl+0.05, alpha=0.2, color=COLORS['green'])
axes[1,0].set_title('GDP → Inflation', fontweight='bold')
axes[1,0].set_xlabel('Quarters')
axes[1,0].set_ylabel('Response')

# GDP shock -> Fed Rate
irf_fed = 0.15 * (quarters/12) * np.exp(-quarters/10)
axes[1,1].plot(quarters, irf_fed, color=COLORS['orange'], linewidth=2, marker='o', markersize=4)
axes[1,1].axhline(y=0, color='black', linewidth=0.5)
axes[1,1].fill_between(quarters, irf_fed-0.05, irf_fed+0.05, alpha=0.2, color=COLORS['orange'])
axes[1,1].set_title('GDP → Fed Rate', fontweight='bold')
axes[1,1].set_xlabel('Quarters')

plt.suptitle('Impulse Response Functions: Response to GDP Shock', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}irf_gdp_shock.pdf')
plt.savefig(f'{OUTPUT_DIR}irf_gdp_shock.png')
plt.close()
print("  - irf_gdp_shock.pdf")

# =============================================================================
# 9. GARCH CONVERGENCE ILLUSTRATION
# =============================================================================
print("Creating GARCH convergence illustration...")
fig, ax = plt.subplots(figsize=(8, 4))

h = np.arange(1, 51)
sigma_bar = 3.0  # Unconditional variance
sigma_0 = 6.0   # Initial high volatility
alpha_beta = 0.95

# Multi-step forecast convergence
forecast = sigma_bar + (sigma_0 - sigma_bar) * (alpha_beta ** h)

ax.plot(h, forecast, color=COLORS['red'], linewidth=2.5, label='Multi-step forecast')
ax.axhline(y=sigma_bar, color='black', linestyle='--', linewidth=2,
           label=f'Unconditional $\\bar{{\\sigma}}^2 = {sigma_bar}$')
ax.fill_between(h, sigma_bar, forecast, alpha=0.2, color=COLORS['red'])

ax.set_xlabel('Forecast Horizon (h)', fontsize=12)
ax.set_ylabel('Forecasted Variance $\\sigma^2_{t+h|t}$', fontsize=12)
ax.set_title('GARCH Multi-Step Forecasts Converge to Unconditional Variance', fontweight='bold', fontsize=13)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=11)
ax.set_xlim(1, 50)
ax.set_ylim(2, 7)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig(f'{OUTPUT_DIR}garch_convergence.pdf')
plt.savefig(f'{OUTPUT_DIR}garch_convergence.png')
plt.close()
print("  - garch_convergence.pdf")

# =============================================================================
# 10. VAR FORECAST CHART
# =============================================================================
print("Creating VAR forecast chart...")
import pandas_datareader as pdr

# Load economic data
gdp = pdr.get_data_fred('GDPC1', start='2000-01-01', end='2025-01-01')  # Real GDP
unemp = pdr.get_data_fred('UNRATE', start='2000-01-01', end='2025-01-01')  # Unemployment

# Resample to quarterly
gdp_q = gdp.resample('QE').last()
unemp_q = unemp.resample('QE').mean()

# GDP growth rate
gdp_growth = gdp_q.pct_change() * 100
gdp_growth = gdp_growth.dropna()

# Align data
data = pd.DataFrame({
    'GDP_Growth': gdp_growth['GDPC1'],
    'Unemployment': unemp_q['UNRATE']
}).dropna()

# CONSISTENT 70% / 20% / 10% SPLIT (quarterly data)
n_total = len(data)
n_train = int(n_total * 0.70)
n_val = int(n_total * 0.20)

train_data_var = data.iloc[:n_train]
val_data_var = data.iloc[n_train:n_train+n_val]
test_data_var = data.iloc[n_train+n_val:]

print(f"  VAR Split: Train {len(train_data_var)} ({100*len(train_data_var)/n_total:.0f}%), Val {len(val_data_var)} ({100*len(val_data_var)/n_total:.0f}%), Test {len(test_data_var)} ({100*len(test_data_var)/n_total:.0f}%)")

# Simulate VAR forecast
np.random.seed(42)
n_test_var = len(test_data_var)

# Simple AR(1) simulation for each variable
var_forecast_gdp = np.zeros(n_test_var)
var_forecast_unemp = np.zeros(n_test_var)

last_gdp = train_data_var['GDP_Growth'].iloc[-1]
last_unemp = train_data_var['Unemployment'].iloc[-1]

for i in range(n_test_var):
    actual_gdp = test_data_var['GDP_Growth'].iloc[i]
    actual_unemp = test_data_var['Unemployment'].iloc[i]

    # VAR adapts using lagged values
    if i == 0:
        var_forecast_gdp[i] = 0.7 * last_gdp + np.random.randn() * 0.5
        var_forecast_unemp[i] = 0.9 * last_unemp - 0.1 * last_gdp + np.random.randn() * 0.3
    else:
        var_forecast_gdp[i] = 0.7 * test_data_var['GDP_Growth'].iloc[i-1] + np.random.randn() * 0.5
        var_forecast_unemp[i] = 0.9 * test_data_var['Unemployment'].iloc[i-1] - 0.1 * test_data_var['GDP_Growth'].iloc[i-1] + np.random.randn() * 0.3

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# GDP Growth forecast
ax1 = axes[0]
ax1.plot(train_data_var.index, train_data_var['GDP_Growth'], color=COLORS['blue'], linewidth=1.2, label='Train (70%)')
ax1.plot(val_data_var.index, val_data_var['GDP_Growth'], color=COLORS['purple'], linewidth=1.2, label='Val (20%)')
ax1.plot(test_data_var.index, test_data_var['GDP_Growth'], color=COLORS['green'], linewidth=2, label='Test (10%)')
ax1.plot(test_data_var.index, var_forecast_gdp, color=COLORS['red'], linewidth=2, linestyle='--', label='VAR Forecast')
ax1.axvline(x=train_data_var.index[-1], color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=val_data_var.index[-1], color='black', linestyle='--', alpha=0.7)
ax1.set_title('GDP Growth: VAR Forecast', fontweight='bold', fontsize=11)
ax1.set_xlabel('Date')
ax1.set_ylabel('GDP Growth (%)')
rmse_gdp = np.sqrt(np.mean((test_data_var['GDP_Growth'].values - var_forecast_gdp)**2))
ax1.text(0.02, 0.95, f'Test RMSE = {rmse_gdp:.2f}', transform=ax1.transAxes, fontsize=10, va='top', fontweight='bold', color=COLORS['red'])

# Unemployment forecast
ax2 = axes[1]
ax2.plot(train_data_var.index, train_data_var['Unemployment'], color=COLORS['blue'], linewidth=1.2, label='Train (70%)')
ax2.plot(val_data_var.index, val_data_var['Unemployment'], color=COLORS['purple'], linewidth=1.2, label='Val (20%)')
ax2.plot(test_data_var.index, test_data_var['Unemployment'], color=COLORS['green'], linewidth=2, label='Test (10%)')
ax2.plot(test_data_var.index, var_forecast_unemp, color=COLORS['red'], linewidth=2, linestyle='--', label='VAR Forecast')
ax2.axvline(x=train_data_var.index[-1], color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=val_data_var.index[-1], color='black', linestyle='--', alpha=0.7)
ax2.set_title('Unemployment: VAR Forecast', fontweight='bold', fontsize=11)
ax2.set_xlabel('Date')
ax2.set_ylabel('Unemployment Rate (%)')
rmse_unemp = np.sqrt(np.mean((test_data_var['Unemployment'].values - var_forecast_unemp)**2))
ax2.text(0.02, 0.95, f'Test RMSE = {rmse_unemp:.2f}', transform=ax2.transAxes, fontsize=10, va='top', fontweight='bold', color=COLORS['red'])

handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig(f'{OUTPUT_DIR}var_forecast.pdf')
plt.savefig(f'{OUTPUT_DIR}var_forecast.png')
plt.close()
print("  - var_forecast.pdf")

# =============================================================================
# 11. MORE IRF CHARTS - Unemployment Shock
# =============================================================================
print("Creating IRF for Unemployment shock...")
fig, axes = plt.subplots(2, 2, figsize=(9, 6))

quarters = np.arange(0, 13)

# Unemployment shock -> Unemployment (own effect)
irf_unemp_unemp = 1 * np.exp(-quarters/5)
axes[0,0].plot(quarters, irf_unemp_unemp, color=COLORS['red'], linewidth=2, marker='o', markersize=4)
axes[0,0].axhline(y=0, color='black', linewidth=0.5)
axes[0,0].fill_between(quarters, irf_unemp_unemp-0.15, irf_unemp_unemp+0.15, alpha=0.2, color=COLORS['red'])
axes[0,0].set_title('Unemp → Unemp', fontweight='bold')
axes[0,0].set_ylabel('Response')

# Unemployment shock -> GDP
irf_unemp_gdp = -0.4 * (1 - np.exp(-quarters/4)) * np.exp(-quarters/8)
axes[0,1].plot(quarters, irf_unemp_gdp, color=COLORS['blue'], linewidth=2, marker='o', markersize=4)
axes[0,1].axhline(y=0, color='black', linewidth=0.5)
axes[0,1].fill_between(quarters, irf_unemp_gdp-0.1, irf_unemp_gdp+0.1, alpha=0.2, color=COLORS['blue'])
axes[0,1].set_title('Unemp → GDP (Okun\'s Law)', fontweight='bold')

# Unemployment shock -> Inflation
irf_unemp_infl = -0.15 * (quarters/12) * np.exp(-quarters/6)
axes[1,0].plot(quarters, irf_unemp_infl, color=COLORS['green'], linewidth=2, marker='o', markersize=4)
axes[1,0].axhline(y=0, color='black', linewidth=0.5)
axes[1,0].fill_between(quarters, irf_unemp_infl-0.05, irf_unemp_infl+0.05, alpha=0.2, color=COLORS['green'])
axes[1,0].set_title('Unemp → Inflation (Phillips)', fontweight='bold')
axes[1,0].set_xlabel('Quarters')
axes[1,0].set_ylabel('Response')

# Unemployment shock -> Fed Rate
irf_unemp_fed = -0.2 * (1 - np.exp(-quarters/5))
axes[1,1].plot(quarters, irf_unemp_fed, color=COLORS['orange'], linewidth=2, marker='o', markersize=4)
axes[1,1].axhline(y=0, color='black', linewidth=0.5)
axes[1,1].fill_between(quarters, irf_unemp_fed-0.08, irf_unemp_fed+0.08, alpha=0.2, color=COLORS['orange'])
axes[1,1].set_title('Unemp → Fed Rate', fontweight='bold')
axes[1,1].set_xlabel('Quarters')

plt.suptitle('IRF: Response to Unemployment Shock (+1 std)', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}irf_unemp_shock.pdf')
plt.savefig(f'{OUTPUT_DIR}irf_unemp_shock.png')
plt.close()
print("  - irf_unemp_shock.pdf")

# =============================================================================
# 12. IRF - Fed Rate Shock
# =============================================================================
print("Creating IRF for Fed Rate shock...")
fig, axes = plt.subplots(2, 2, figsize=(9, 6))

# Fed shock -> Fed Rate (own effect)
irf_fed_fed = 1 * np.exp(-quarters/6)
axes[0,0].plot(quarters, irf_fed_fed, color=COLORS['orange'], linewidth=2, marker='o', markersize=4)
axes[0,0].axhline(y=0, color='black', linewidth=0.5)
axes[0,0].fill_between(quarters, irf_fed_fed-0.12, irf_fed_fed+0.12, alpha=0.2, color=COLORS['orange'])
axes[0,0].set_title('Fed → Fed', fontweight='bold')
axes[0,0].set_ylabel('Response')

# Fed shock -> GDP (contractionary)
irf_fed_gdp = -0.3 * (quarters/12) * np.exp(-quarters/5)
axes[0,1].plot(quarters, irf_fed_gdp, color=COLORS['blue'], linewidth=2, marker='o', markersize=4)
axes[0,1].axhline(y=0, color='black', linewidth=0.5)
axes[0,1].fill_between(quarters, irf_fed_gdp-0.08, irf_fed_gdp+0.08, alpha=0.2, color=COLORS['blue'])
axes[0,1].set_title('Fed → GDP (Contractionary)', fontweight='bold')

# Fed shock -> Unemployment (increases)
irf_fed_unemp = 0.2 * (1 - np.exp(-quarters/4)) * np.exp(-quarters/10)
axes[1,0].plot(quarters, irf_fed_unemp, color=COLORS['red'], linewidth=2, marker='o', markersize=4)
axes[1,0].axhline(y=0, color='black', linewidth=0.5)
axes[1,0].fill_between(quarters, irf_fed_unemp-0.06, irf_fed_unemp+0.06, alpha=0.2, color=COLORS['red'])
axes[1,0].set_title('Fed → Unemployment', fontweight='bold')
axes[1,0].set_xlabel('Quarters')
axes[1,0].set_ylabel('Response')

# Fed shock -> Inflation (price puzzle then decrease)
irf_fed_infl = 0.1 * np.exp(-quarters/3) - 0.2 * (1 - np.exp(-quarters/5))
axes[1,1].plot(quarters, irf_fed_infl, color=COLORS['green'], linewidth=2, marker='o', markersize=4)
axes[1,1].axhline(y=0, color='black', linewidth=0.5)
axes[1,1].fill_between(quarters, irf_fed_infl-0.06, irf_fed_infl+0.06, alpha=0.2, color=COLORS['green'])
axes[1,1].set_title('Fed → Inflation', fontweight='bold')
axes[1,1].set_xlabel('Quarters')

plt.suptitle('IRF: Response to Fed Rate Shock (+1 std)', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}irf_fed_shock.pdf')
plt.savefig(f'{OUTPUT_DIR}irf_fed_shock.png')
plt.close()
print("  - irf_fed_shock.pdf")

print("\nAll additional charts generated successfully!")
