#!/usr/bin/env python3
"""
Generate ARFIMA real data example using actual S&P 500 data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Style settings
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Colors
MAIN_BLUE = '#1A3A6E'
ACCENT_BLUE = '#4A90D9'
IDA_RED = '#DC3545'
FOREST = '#2E7D32'
ORANGE = '#E67E22'
GRAY = '#666666'

# =============================================================================
# Download real S&P 500 data
# =============================================================================
print("Downloading S&P 500 data...")
sp500 = yf.download('^GSPC', start='2015-01-01', end='2024-12-31', progress=False)

# Calculate returns and realized volatility (rolling 20-day std, annualized)
returns = np.log(sp500['Close'] / sp500['Close'].shift(1)).dropna()
realized_vol = returns.rolling(window=20).std() * np.sqrt(252) * 100  # Annualized %
realized_vol = realized_vol.dropna()

# Also calculate squared returns (proxy for volatility)
squared_returns = (returns ** 2) * 10000  # Scale for visibility

print(f"Data: {len(realized_vol)} observations from {realized_vol.index[0].date()} to {realized_vol.index[-1].date()}")

# =============================================================================
# Compute ACF
# =============================================================================
def compute_acf(x, max_lag=100):
    x = np.array(x)
    n = len(x)
    x = x - np.mean(x)
    var = np.var(x)
    acf = np.array([np.sum(x[:n-k] * x[k:]) / (n * var) for k in range(max_lag+1)])
    return acf

acf_vol = compute_acf(realized_vol.values, max_lag=100)
acf_sq_ret = compute_acf(squared_returns.dropna().values, max_lag=100)

# =============================================================================
# Estimate Hurst using R/S method
# =============================================================================
def estimate_hurst_rs(ts, min_lag=10, max_lag=None):
    ts = np.array(ts)
    n = len(ts)
    if max_lag is None:
        max_lag = min(n // 4, 500)

    lags = []
    rs_values = []

    for lag in range(min_lag, max_lag, 10):
        n_blocks = n // lag
        if n_blocks < 2:
            break

        rs_block = []
        for i in range(n_blocks):
            block = ts[i*lag:(i+1)*lag]
            mean_block = np.mean(block)
            cumdev = np.cumsum(block - mean_block)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(block, ddof=1)
            if S > 0:
                rs_block.append(R / S)

        if len(rs_block) > 0:
            lags.append(lag)
            rs_values.append(np.mean(rs_block))

    log_lags = np.log(lags)
    log_rs = np.log(rs_values)
    H = np.polyfit(log_lags, log_rs, 1)[0]

    return H, lags, rs_values

H_vol, lags_vol, rs_vol = estimate_hurst_rs(realized_vol.values)
d_vol = H_vol - 0.5

print(f"Realized Volatility: H = {H_vol:.3f}, d = {d_vol:.3f}")

# =============================================================================
# Create the figure - 2x2 layout, charts only
# =============================================================================
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

# Panel 1: Time series of realized volatility
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(realized_vol.index, realized_vol.values, color=MAIN_BLUE, lw=0.8, alpha=0.9)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Realized Volatility (%)', fontsize=12)
ax1.set_title('S&P 500 Realized Volatility (20-day, annualized)', fontsize=13, fontweight='bold', color=MAIN_BLUE)

# Panel 2: ACF of realized volatility
ax2 = fig.add_subplot(gs[0, 1])
lags_acf = np.arange(len(acf_vol))
ax2.bar(lags_acf[1:80], acf_vol[1:80], color=ACCENT_BLUE, alpha=0.7, width=0.8)

# Theoretical hyperbolic decay
k = np.arange(1, 80)
theoretical = k**(2*d_vol - 1)
theoretical = theoretical * acf_vol[1] / theoretical[0]
ax2.plot(k, theoretical, color=IDA_RED, lw=2.5, ls='--')

# Exponential decay for comparison
exp_decay = 0.95**k * acf_vol[1]
ax2.plot(k, exp_decay, color=ORANGE, lw=2, ls=':')

# Confidence bands
conf = 1.96 / np.sqrt(len(realized_vol))
ax2.axhline(conf, color=GRAY, ls=':', lw=1, alpha=0.7)
ax2.axhline(-conf, color=GRAY, ls=':', lw=1, alpha=0.7)
ax2.axhline(0, color=GRAY, lw=0.5)

ax2.set_xlabel('Lag (days)', fontsize=12)
ax2.set_ylabel('ACF', fontsize=12)
ax2.set_title('ACF: Slow Hyperbolic Decay', fontsize=13, fontweight='bold', color=MAIN_BLUE)
ax2.set_xlim(0, 80)

# Panel 3: R/S Analysis
ax3 = fig.add_subplot(gs[1, 0])
log_lags = np.log(lags_vol)
log_rs = np.log(rs_vol)
ax3.scatter(log_lags, log_rs, color=ACCENT_BLUE, s=50, alpha=0.8, zorder=3, edgecolors='white', lw=1)

# Regression line
z = np.polyfit(log_lags, log_rs, 1)
p = np.poly1d(z)
ax3.plot(log_lags, p(log_lags), color=IDA_RED, lw=2.5)

ax3.set_xlabel('log(n)', fontsize=12)
ax3.set_ylabel('log(R/S)', fontsize=12)
ax3.set_title('R/S Analysis: Hurst Exponent Estimation', fontsize=13, fontweight='bold', color=MAIN_BLUE)

# Add result annotation
ax3.text(0.95, 0.08, f'$H = {H_vol:.2f}$\n$d = H - 0.5 = {d_vol:.2f}$',
         transform=ax3.transAxes, fontsize=14, fontweight='bold', color=FOREST,
         ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=FOREST, alpha=0.9))

# Panel 4: ACF comparison - returns vs squared returns
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(lags_acf[1:50] - 0.2, compute_acf(returns.values, 50)[1:50],
        color=ACCENT_BLUE, alpha=0.6, width=0.4, label='Returns')
ax4.bar(lags_acf[1:50] + 0.2, acf_sq_ret[1:50],
        color=IDA_RED, alpha=0.6, width=0.4, label='Squared Returns')

ax4.axhline(0, color=GRAY, lw=0.5)
ax4.set_xlabel('Lag (days)', fontsize=12)
ax4.set_ylabel('ACF', fontsize=12)
ax4.set_title('Returns vs Squared Returns (Volatility Proxy)', fontsize=13, fontweight='bold', color=MAIN_BLUE)
ax4.set_xlim(0, 50)
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

# =============================================================================
# Legend at bottom center
# =============================================================================
legend_elements = [
    Line2D([0], [0], color=ACCENT_BLUE, lw=8, alpha=0.7, label='Sample ACF'),
    Line2D([0], [0], color=IDA_RED, lw=2.5, ls='--', label=f'Hyperbolic $k^{{2d-1}}$ (d={d_vol:.2f})'),
    Line2D([0], [0], color=ORANGE, lw=2, ls=':', label='Exponential (ARMA)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=ACCENT_BLUE, markersize=10, label='R/S statistic'),
]

fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.savefig('ch8_arfima_real_data.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.savefig('ch8_arfima_real_data.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()

print(f"\nGenerated: ch8_arfima_real_data.pdf")
print(f"Hurst exponent H = {H_vol:.3f}")
print(f"Fractional d = {d_vol:.3f}")
