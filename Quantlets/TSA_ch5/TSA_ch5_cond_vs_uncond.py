"""
TSA_ch5_cond_vs_uncond
======================
Conditional vs Unconditional Volatility Comparison

This script generates a chart showing the difference between
conditional volatility (time-varying, from GARCH) and
unconditional volatility (constant long-run average).

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'arch', '-q'])
    from arch import arch_model

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance', '-q'])
    import yfinance as yf

# Chart style settings
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

CHARTS_DIR = '../../charts'

# Download S&P 500 data
print("Downloading S&P 500 data...")
sp500 = yf.download('^GSPC', start='2020-01-01', end='2026-02-01', progress=False)
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)
returns = 100 * sp500['Close'].pct_change().dropna()

# Fit GARCH(1,1)
print("Fitting GARCH(1,1)...")
model = arch_model(returns, vol='Garch', p=1, q=1, dist='normal', mean='Constant')
res = model.fit(disp='off')

# Extract conditional volatility
cond_vol = res.conditional_volatility

# Compute unconditional volatility
omega = res.params['omega']
alpha = res.params['alpha[1]']
beta = res.params['beta[1]']
uncond_var = omega / (1 - alpha - beta)
uncond_vol = np.sqrt(uncond_var)

print(f"GARCH(1,1) parameters: omega={omega:.6f}, alpha={alpha:.4f}, beta={beta:.4f}")
print(f"Unconditional volatility: {uncond_vol:.4f}%")
print(f"alpha + beta = {alpha + beta:.4f}")

# Create the chart
fig, ax = plt.subplots(figsize=(10, 3.5))

# Plot conditional volatility
ax.plot(cond_vol.index, cond_vol.values, color='#2166AC', linewidth=0.8,
        label=f'Conditional volatility $\\sigma_t$ (GARCH)', alpha=0.9)

# Plot unconditional volatility as horizontal line
ax.axhline(y=uncond_vol, color='#B2182B', linewidth=1.5, linestyle='--',
           label=f'Unconditional volatility $\\bar{{\\sigma}}$ = {uncond_vol:.2f}%', alpha=0.9)

# Fill between to show deviations
ax.fill_between(cond_vol.index, uncond_vol, cond_vol.values,
                where=(cond_vol.values > uncond_vol),
                color='#B2182B', alpha=0.15, interpolate=True)
ax.fill_between(cond_vol.index, uncond_vol, cond_vol.values,
                where=(cond_vol.values < uncond_vol),
                color='#2166AC', alpha=0.15, interpolate=True)

ax.set_xlabel('Date')
ax.set_ylabel('Volatility (%)')
ax.set_title('Conditional vs Unconditional Volatility: S&P 500 (GARCH(1,1))')
ax.legend(loc='upper right', frameon=False)

# Add annotations for high volatility periods
max_vol_idx = cond_vol.idxmax()
max_vol = cond_vol.max()
ax.annotate(f'Peak: {max_vol:.1f}%', xy=(max_vol_idx, max_vol),
            xytext=(max_vol_idx, max_vol + 0.5),
            fontsize=7, ha='center', color='#B2182B',
            arrowprops=dict(arrowstyle='->', color='#B2182B', lw=0.5))

plt.tight_layout()

# Save
plt.savefig(f'{CHARTS_DIR}/garch_cond_vs_uncond.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.savefig(f'{CHARTS_DIR}/garch_cond_vs_uncond.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

print(f"Chart saved to {CHARTS_DIR}/garch_cond_vs_uncond.pdf")
