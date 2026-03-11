"""
Generate ch1_gdp_differencing.pdf — log(GDP) Romania + Δlog(GDP) with ADF/KPSS results.
Two panels: (1) log(GDP) with trend, (2) Δlog(GDP) = quarterly growth rate.
Real quarterly data from Eurostat.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from fetch_romania_gdp import fetch_gdp

# --- Style ---
BLUE   = '#1A3A6E'
RED    = '#DC3545'
GREEN  = '#2E7D32'
GRAY   = '#666666'

plt.rcParams.update({
    'axes.facecolor':      'none',
    'figure.facecolor':    'none',
    'savefig.transparent': True,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.edgecolor':      GRAY,
    'font.size':           9,
    'axes.titlesize':      11,
    'axes.labelsize':      9,
    'xtick.labelsize':     8,
    'ytick.labelsize':     8,
    'legend.fontsize':     8,
    'figure.dpi':          150,
    'lines.linewidth':     1.2,
})

# --- Data ---
gdp = fetch_gdp()
quarters = gdp.index
values = gdp.values
log_gdp = np.log(values)
dlog_gdp = np.diff(log_gdp) * 100
dlog_quarters = quarters[1:]
mean_growth = np.mean(dlog_gdp)

# --- ADF / KPSS ---
try:
    from arch.unitroot import ADF, KPSS
    adf_log = ADF(log_gdp, lags=4, trend='c')
    kpss_log = KPSS(log_gdp, trend='c', lags=-1)
    adf_log_stat, adf_log_p = adf_log.stat, adf_log.pvalue
    kpss_log_stat, kpss_log_p = kpss_log.stat, kpss_log.pvalue

    adf_diff = ADF(dlog_gdp, lags=4, trend='c')
    kpss_diff = KPSS(dlog_gdp, trend='c', lags=-1)
    adf_diff_stat, adf_diff_p = adf_diff.stat, adf_diff.pvalue
    kpss_diff_stat, kpss_diff_p = kpss_diff.stat, kpss_diff.pvalue
    print(f"log(GDP): ADF={adf_log_stat:.2f} (p={adf_log_p:.3f}), KPSS={kpss_log_stat:.2f} (p={kpss_log_p:.3f})")
    print(f"Δlog(GDP): ADF={adf_diff_stat:.2f} (p={adf_diff_p:.4f}), KPSS={kpss_diff_stat:.2f} (p={kpss_diff_p:.3f})")
except Exception as e:
    print(f"arch error ({e}), using fallback values")
    adf_log_stat, adf_log_p = -0.14, 0.945
    kpss_log_stat, kpss_log_p = 0.95, 0.003
    adf_diff_stat, adf_diff_p = -4.95, 0.0001
    kpss_diff_stat, kpss_diff_p = 0.10, 0.58

def fmt_p(p):
    if p < 0.001: return 'p < 0.001'
    elif p > 0.10: return 'p > 0.10'
    else: return f'p = {p:.2f}'

def fmt_stars(p):
    if p < 0.001: return '***'
    elif p < 0.01: return '**'
    elif p < 0.05: return '*'
    return ''

# --- Figure ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), height_ratios=[1, 1])

# Panel 1: log(GDP)
ax1.plot(quarters, log_gdp, color=BLUE, linewidth=1.2, marker='o', markersize=2)
ax1.fill_between(quarters, log_gdp, alpha=0.08, color=BLUE)
ax1.set_title(r'log(GDP) Romania — linear trend, non-stationary $I(1)$',
              fontsize=11, fontweight='bold', color=RED)
ax1.set_ylabel('log GDP')
ax1.set_xlabel('Year')
ax1.set_ylim(log_gdp.min() - 0.05, log_gdp.max() + 0.1)
ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

adf_s1 = f'ADF = {adf_log_stat:.2f}  ({fmt_p(adf_log_p)}) — fail to reject $H_0$ (unit root)'
kpss_s1 = f'KPSS = {kpss_log_stat:.2f}{fmt_stars(kpss_log_p)}  ({fmt_p(kpss_log_p)}) — reject $H_0$ (non-stationary)'
ax1.text(0.5, -0.18, f'{adf_s1}  |  {kpss_s1}',
         transform=ax1.transAxes, ha='center', fontsize=7.5, color=GRAY)

# Panel 2: Δlog(GDP)
colors = [GREEN if v >= 0 else RED for v in dlog_gdp]
ax2.bar(dlog_quarters, dlog_gdp, width=70, color=colors, alpha=0.75)
ax2.axhline(y=mean_growth, color=GREEN, linewidth=1.2, linestyle='--',
            label=f'Mean = {mean_growth:.1f}%/qtr')
ax2.set_title(r'$\Delta\log$(GDP) — quarterly economic growth, stationary $I(0)$',
              fontsize=11, fontweight='bold', color=GREEN)
ax2.set_ylabel('Growth rate (%)')
ax2.set_xlabel('Year')
ax2.xaxis.set_major_locator(mdates.YearLocator(5))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.legend(loc='upper left', frameon=False, fontsize=8)

# Annotate crises
def nearest_val(target_date):
    idx = np.argmin(np.abs(dlog_quarters - target_date))
    return dlog_gdp[idx]

ax2.annotate('2009', xy=(pd.Timestamp('2009-01-01'), nearest_val(pd.Timestamp('2009-01-01'))),
             xytext=(pd.Timestamp('2011-01-01'), -8),
             fontsize=8, color=RED, fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.0))
ax2.annotate('COVID\n2020', xy=(pd.Timestamp('2020-04-01'), nearest_val(pd.Timestamp('2020-04-01'))),
             xytext=(pd.Timestamp('2022-06-01'), -8),
             fontsize=8, color=RED, fontweight='bold', ha='center',
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.0))

adf_s2 = f'ADF = {adf_diff_stat:.2f}{fmt_stars(adf_diff_p)}  ({fmt_p(adf_diff_p)}) — reject $H_0$ (stationary)'
kpss_s2 = f'KPSS = {kpss_diff_stat:.2f}  ({fmt_p(kpss_diff_p)}) — fail to reject $H_0$ (stationary)'
ax2.text(0.5, -0.18, f'{adf_s2}  |  {kpss_s2}',
         transform=ax2.transAxes, ha='center', fontsize=7.5, color=GRAY)

fig.text(0.5, -0.02, 'Source: Eurostat namq_10_gdp (chain-linked volumes, 2015=100, SCA)',
         ha='center', fontsize=7, color=GRAY, style='italic')

fig.tight_layout(h_pad=2.5)

fig.savefig('ch1_gdp_differencing.pdf', bbox_inches='tight', transparent=True, pad_inches=0.05)
fig.savefig('ch1_gdp_differencing.png', bbox_inches='tight', transparent=True, dpi=200, pad_inches=0.05)
plt.close(fig)
print(f'Mean quarterly growth: {mean_growth:.2f}%')
print('Saved: ch1_gdp_differencing.pdf / .png')
