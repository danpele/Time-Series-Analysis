"""Generate 3 theory-section charts to replace duplicates from S&P 500 / BTC case studies."""
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Download EUR/USD data for theory examples ───
eurusd = yf.download('EURUSD=X', start='2015-01-01', end='2024-12-31', auto_adjust=True)
ret_eur = (100 * np.log(eurusd['Close'] / eurusd['Close'].shift(1)).dropna()).squeeze()

# ─── Download DAX for the intro volatility clustering (replaces btc_returns.pdf in intro) ───
dax = yf.download('^GDAXI', start='2015-01-01', end='2024-12-31', auto_adjust=True)
ret_dax = (100 * np.log(dax['Close'] / dax['Close'].shift(1)).dropna()).squeeze()

# ═══════════════════════════════════════════════════════════════════════
# CHART 1: EGARCH vs GARCH comparison on EUR/USD  (replaces garch_sp500_comparison.pdf at line 1679)
# ═══════════════════════════════════════════════════════════════════════
am_g = arch_model(ret_eur, vol='Garch', p=1, q=1, dist='t')
res_g = am_g.fit(disp='off')

am_e = arch_model(ret_eur, vol='EGARCH', p=1, o=1, q=1, dist='t')
res_e = am_e.fit(disp='off')

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

axes[0].plot(ret_eur.index, ret_eur.values, color='#888888', linewidth=0.4, alpha=0.7)
axes[0].set_ylabel('Returns (%)', fontsize=11)
axes[0].set_title('EUR/USD Daily Returns (2015–2024)', fontsize=12, fontweight='bold')
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

axes[1].plot(res_g.conditional_volatility.index, res_g.conditional_volatility.values,
             color='#1A3A6E', linewidth=1.0, label='GARCH(1,1)-t', alpha=0.85)
axes[1].plot(res_e.conditional_volatility.index, res_e.conditional_volatility.values,
             color='#CD0000', linewidth=1.0, label='EGARCH(1,1,1)-t', alpha=0.85)
axes[1].set_ylabel('Conditional Volatility (%)', fontsize=11)
axes[1].set_xlabel('Date', fontsize=11)
axes[1].set_title('GARCH vs EGARCH Conditional Volatility', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10, frameon=False)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('/Users/danielpele/Documents/TSA/charts/ch5_egarch_eurusd_comparison.pdf',
            bbox_inches='tight', transparent=True, dpi=150)
plt.close()
print("Done: ch5_egarch_eurusd_comparison.pdf")

# ═══════════════════════════════════════════════════════════════════════
# CHART 2: Diagnostic plots for GARCH on EUR/USD  (replaces garch_diagnostics.pdf at line 2137)
# ═══════════════════════════════════════════════════════════════════════
std_resid = res_e.std_resid.dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# (a) Standardized residuals
axes[0, 0].plot(std_resid.index, std_resid.values, color='#1A3A6E', linewidth=0.4, alpha=0.7)
axes[0, 0].axhline(y=0, color='#CD0000', linewidth=0.8, linestyle='--')
axes[0, 0].set_title('(a) Standardized Residuals $\\hat{z}_t$', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('$\\hat{z}_t$', fontsize=10)
axes[0, 0].spines['top'].set_visible(False)
axes[0, 0].spines['right'].set_visible(False)

# (b) ACF of squared standardized residuals
from statsmodels.tsa.stattools import acf
acf_vals = acf(std_resid**2, nlags=30, fft=True)
n = len(std_resid)
ci = 1.96 / np.sqrt(n)
axes[0, 1].bar(range(len(acf_vals)), acf_vals, color='#1A3A6E', width=0.6, alpha=0.8)
axes[0, 1].axhline(y=ci, color='#CD0000', linestyle='--', linewidth=0.8)
axes[0, 1].axhline(y=-ci, color='#CD0000', linestyle='--', linewidth=0.8)
axes[0, 1].set_title('(b) ACF of $\\hat{z}_t^2$', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Lag', fontsize=10)
axes[0, 1].spines['top'].set_visible(False)
axes[0, 1].spines['right'].set_visible(False)

# (c) Histogram vs t-distribution
nu = res_e.params.get('nu', 5)
axes[1, 0].hist(std_resid, bins=60, density=True, color='#AAAAAA', edgecolor='white', alpha=0.7, label='Empirical')
x_range = np.linspace(-5, 5, 300)
axes[1, 0].plot(x_range, stats.t.pdf(x_range, df=nu), color='#CD0000', linewidth=1.5,
                label=f'Student-t ($\\nu$={nu:.1f})')
axes[1, 0].plot(x_range, stats.norm.pdf(x_range), color='#1A3A6E', linewidth=1.2,
                linestyle='--', label='Normal')
axes[1, 0].set_title('(c) Distribution of $\\hat{z}_t$', fontsize=11, fontweight='bold')
axes[1, 0].legend(fontsize=9, frameon=False)
axes[1, 0].spines['top'].set_visible(False)
axes[1, 0].spines['right'].set_visible(False)

# (d) Q-Q plot
theoretical_q = stats.t.ppf(np.linspace(0.001, 0.999, len(std_resid)), df=nu)
empirical_q = np.sort(std_resid.values)
axes[1, 1].scatter(theoretical_q, empirical_q, s=1, color='#1A3A6E', alpha=0.5)
lims = [-5, 5]
axes[1, 1].plot(lims, lims, color='#CD0000', linewidth=1.2, linestyle='--')
axes[1, 1].set_xlim(lims)
axes[1, 1].set_ylim(lims)
axes[1, 1].set_title('(d) Q-Q Plot (Student-t)', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Theoretical Quantiles', fontsize=10)
axes[1, 1].set_ylabel('Empirical Quantiles', fontsize=10)
axes[1, 1].spines['top'].set_visible(False)
axes[1, 1].spines['right'].set_visible(False)

fig.suptitle('EGARCH(1,1,1)-t Diagnostics — EUR/USD', fontsize=13, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('/Users/danielpele/Documents/TSA/charts/ch5_garch_diagnostic_eurusd.pdf',
            bbox_inches='tight', transparent=True, dpi=150)
plt.close()
print("Done: ch5_garch_diagnostic_eurusd.pdf")

# ═══════════════════════════════════════════════════════════════════════
# CHART 3: DAX returns with volatility clustering annotation (replaces btc_returns.pdf at line 164)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(ret_dax.index, ret_dax.values, color='#1A3A6E', linewidth=0.4, alpha=0.7)
ax.axhline(y=0, color='#CD0000', linewidth=0.6, linestyle='--', alpha=0.5)

# Highlight clustering periods
import pandas as pd
rolling_vol = ret_dax.rolling(21).std()
high_vol_mask = rolling_vol > rolling_vol.quantile(0.90)
for start_idx in range(len(high_vol_mask)):
    if high_vol_mask.iloc[start_idx]:
        ax.axvspan(ret_dax.index[start_idx], ret_dax.index[min(start_idx+1, len(ret_dax)-1)],
                   alpha=0.08, color='#CD0000', linewidth=0)

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Returns (%)', fontsize=11)
ax.set_title('DAX Daily Returns (2015–2024) — Volatility Clustering', fontsize=13, fontweight='bold')

# Add annotation
from matplotlib.patches import Patch
legend_elements = [
    plt.Line2D([0], [0], color='#1A3A6E', linewidth=1, label='Daily Returns (%)'),
    Patch(facecolor='#CD0000', alpha=0.15, label='High Volatility Regimes (top 10%)')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=2, fontsize=10, frameon=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
fig.savefig('/Users/danielpele/Documents/TSA/charts/ch5_dax_volatility_clustering.pdf',
            bbox_inches='tight', transparent=True, dpi=150)
plt.close()
print("Done: ch5_dax_volatility_clustering.pdf")
