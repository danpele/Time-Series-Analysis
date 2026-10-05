#!/usr/bin/env python3
"""
Generate 3 additional case-study charts for Chapter 5b: Multivariate GARCH.
1. Portfolio cumulative performance (DCC vs CCC vs Static)
2. Rolling hedging effectiveness (DCC vs CCC vs OLS)
3. VaR by VIX regime (grouped bar chart)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from arch import arch_model
from scipy.optimize import minimize
import os

# ── Style ─────────────────────────────────────────────────────────────────────
MainBlue = '#1A3A6E'
IDAred   = '#CD0000'
Forest   = '#2E7D32'
Crimson  = '#DC3545'
GoldC    = '#DAA520'
Purple   = '#6A0DAD'
Orange   = '#E67E22'

CHART_DIR = '/Users/danielpele/Documents/TSA/charts'
os.makedirs(CHART_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def save_chart(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, bbox_inches='tight', transparent=True, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

def make_transparent(fig, axes):
    fig.patch.set_alpha(0)
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    for ax in axes:
        ax.patch.set_alpha(0)

# ── Data ──────────────────────────────────────────────────────────────────────
print("Downloading data...")
import yfinance as yf

tickers = {'SP500': '^GSPC', 'DAX': '^GDAXI', 'Gold': 'GC=F', 'ES': 'ES=F'}
price_data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start='2004-01-01', end='2024-12-31',
                         auto_adjust=True, progress=False)
        if len(df) > 100:
            price_data[name] = df['Close'].squeeze()
            print(f"  {name}: {len(df)} rows")
    except Exception as e:
        print(f"  {name}: FAILED - {e}")

# VIX for regime chart
try:
    vix_df = yf.download('^VIX', start='2004-01-01', end='2024-12-31',
                         auto_adjust=True, progress=False)
    vix = vix_df['Close'].squeeze()
    print(f"  VIX: {len(vix_df)} rows")
except Exception as e:
    vix = None
    print(f"  VIX: FAILED - {e}")

prices = pd.DataFrame(price_data)
prices.index = pd.to_datetime(prices.index)
if hasattr(prices.index, 'tz') and prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
returns = prices.pct_change().dropna()

# Align VIX
if vix is not None:
    vix.index = pd.to_datetime(vix.index)
    if hasattr(vix.index, 'tz') and vix.index.tz is not None:
        vix.index = vix.index.tz_localize(None)
    vix = vix.reindex(returns.index).ffill()

# ── GARCH + DCC helpers ──────────────────────────────────────────────────────
def fit_garch(series):
    am = arch_model(series * 100, vol='Garch', p=1, q=1, dist='normal', mean='Constant')
    res = am.fit(disp='off')
    return res.conditional_volatility / 100, res.std_resid

def estimate_dcc_corr(z1, z2):
    z = np.column_stack([z1, z2])
    T = len(z)
    Qbar = np.corrcoef(z1, z2)
    qbar_12 = Qbar[0, 1]
    def negll(params):
        a, b = params
        if a < 0 or b < 0 or a + b >= 1:
            return 1e10
        q12 = np.zeros(T); q11 = np.ones(T); q22 = np.ones(T)
        rho = np.zeros(T)
        q12[0] = qbar_12; rho[0] = qbar_12
        ll = 0
        for t in range(1, T):
            q11[t] = 1 - a - b + a * z[t-1, 0]**2 + b * q11[t-1]
            q22[t] = 1 - a - b + a * z[t-1, 1]**2 + b * q22[t-1]
            q12[t] = (1 - a - b) * qbar_12 + a * z[t-1, 0] * z[t-1, 1] + b * q12[t-1]
            denom = np.sqrt(q11[t] * q22[t])
            rho[t] = q12[t] / denom if denom > 0 else qbar_12
            rho[t] = np.clip(rho[t], -0.999, 0.999)
            ll += np.log(1 - rho[t]**2) + (z[t,0]**2 + z[t,1]**2 - 2*rho[t]*z[t,0]*z[t,1])/(1-rho[t]**2) - z[t,0]**2 - z[t,1]**2
        return 0.5 * ll
    res = minimize(negll, [0.02, 0.95], method='Nelder-Mead',
                   options={'maxiter': 5000})
    a, b = res.x
    # Rebuild rho
    q12 = np.zeros(T); q11 = np.ones(T); q22 = np.ones(T)
    rho = np.zeros(T)
    q12[0] = qbar_12; rho[0] = qbar_12
    for t in range(1, T):
        q11[t] = 1 - a - b + a * z[t-1, 0]**2 + b * q11[t-1]
        q22[t] = 1 - a - b + a * z[t-1, 1]**2 + b * q22[t-1]
        q12[t] = (1 - a - b) * qbar_12 + a * z[t-1, 0] * z[t-1, 1] + b * q12[t-1]
        denom = np.sqrt(q11[t] * q22[t])
        rho[t] = q12[t] / denom if denom > 0 else qbar_12
        rho[t] = np.clip(rho[t], -0.999, 0.999)
    return rho

# ── Fit models ────────────────────────────────────────────────────────────────
print("\nFitting GARCH models...")
assets = ['SP500', 'DAX', 'Gold']
vol = {}
zres = {}
for a in assets:
    v, z = fit_garch(returns[a])
    vol[a] = v
    zres[a] = z
    print(f"  {a}: mean vol = {v.mean()*100:.2f}%")

# DCC correlations for each pair
print("Estimating DCC correlations...")
pairs = [('SP500', 'DAX'), ('SP500', 'Gold'), ('DAX', 'Gold')]
dcc_rho = {}
for a1, a2 in pairs:
    idx = vol[a1].dropna().index.intersection(vol[a2].dropna().index)
    idx = idx.intersection(zres[a1].dropna().index).intersection(zres[a2].dropna().index)
    z1 = zres[a1].reindex(idx).values
    z2 = zres[a2].reindex(idx).values
    rho = estimate_dcc_corr(z1, z2)
    dcc_rho[(a1, a2)] = pd.Series(rho, index=idx)
    print(f"  {a1}-{a2}: mean rho = {np.mean(rho):.3f}")

# CCC correlation = unconditional
ccc_rho = {}
for a1, a2 in pairs:
    idx = dcc_rho[(a1, a2)].index
    ccc_rho[(a1, a2)] = float(np.corrcoef(
        zres[a1].reindex(idx).values, zres[a2].reindex(idx).values)[0, 1])
    print(f"  CCC {a1}-{a2}: rho = {ccc_rho[(a1, a2)]:.3f}")

# Common index
common_idx = dcc_rho[('SP500', 'DAX')].index
for p in pairs:
    common_idx = common_idx.intersection(dcc_rho[p].index)
common_idx = common_idx.intersection(returns.index)
for a in assets:
    common_idx = common_idx.intersection(vol[a].dropna().index)

ret = returns[assets].reindex(common_idx)
T = len(common_idx)

# ── Build covariance matrices and portfolios ──────────────────────────────────
print("\nComputing portfolios...")

def build_cov(t, use_dcc=True):
    """Build 3x3 covariance matrix at time t."""
    H = np.zeros((3, 3))
    for i, ai in enumerate(assets):
        H[i, i] = vol[ai].reindex(common_idx).iloc[t]**2
    for a1, a2 in pairs:
        i = assets.index(a1)
        j = assets.index(a2)
        s1 = vol[a1].reindex(common_idx).iloc[t]
        s2 = vol[a2].reindex(common_idx).iloc[t]
        if use_dcc:
            rho = dcc_rho[(a1, a2)].reindex(common_idx).iloc[t]
        else:
            rho = ccc_rho[(a1, a2)]
        H[i, j] = rho * s1 * s2
        H[j, i] = H[i, j]
    return H

def mv_weights(H):
    """Minimum-variance weights."""
    try:
        Hinv = np.linalg.inv(H)
        ones = np.ones(3)
        w = Hinv @ ones / (ones @ Hinv @ ones)
        w = np.clip(w, 0, 1)
        w = w / w.sum()
        return w
    except:
        return np.array([1/3, 1/3, 1/3])

# Compute portfolio returns
port_ret_dcc = np.zeros(T)
port_ret_ccc = np.zeros(T)
port_ret_static = np.zeros(T)
w_static = np.array([1/3, 1/3, 1/3])

w_dcc_hist = np.zeros((T, 3))
w_ccc_hist = np.zeros((T, 3))

for t in range(T):
    r = ret.iloc[t].values
    H_dcc = build_cov(t, use_dcc=True)
    H_ccc = build_cov(t, use_dcc=False)
    w_dcc = mv_weights(H_dcc)
    w_ccc = mv_weights(H_ccc)
    port_ret_dcc[t] = w_dcc @ r
    port_ret_ccc[t] = w_ccc @ r
    port_ret_static[t] = w_static @ r
    w_dcc_hist[t] = w_dcc
    w_ccc_hist[t] = w_ccc

# Cumulative returns
cum_dcc = (1 + pd.Series(port_ret_dcc, index=common_idx)).cumprod()
cum_ccc = (1 + pd.Series(port_ret_ccc, index=common_idx)).cumprod()
cum_static = (1 + pd.Series(port_ret_static, index=common_idx)).cumprod()

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1: Portfolio Cumulative Performance
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Portfolio cumulative performance...")

fig, ax = plt.subplots(figsize=(10, 5))
make_transparent(fig, ax)
ax.plot(cum_dcc.index, cum_dcc.values, color=MainBlue, lw=1.8, label='MV Portfolio (DCC)')
ax.plot(cum_ccc.index, cum_ccc.values, color=IDAred, lw=1.3, label='MV Portfolio (CCC)', alpha=0.85)
ax.plot(cum_static.index, cum_static.values, color='gray', lw=1.3, ls='--', label='Equal-Weight (1/3 each)', alpha=0.7)

# Shade crisis periods
for start, end, lbl in [('2007-07-01', '2009-03-31', 'GFC'),
                         ('2020-02-01', '2020-06-30', 'COVID')]:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color=Crimson)
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    ypos = ax.get_ylim()[1] * 0.95
    ax.text(mid, ypos, lbl, ha='center', fontsize=9, color=Crimson, fontstyle='italic')

ax.set_ylabel('Cumulative Return (base = 1)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)

# Stats annotation
ann_vol_dcc = port_ret_dcc.std() * np.sqrt(252) * 100
ann_vol_ccc = port_ret_ccc.std() * np.sqrt(252) * 100
ann_vol_stat = port_ret_static.std() * np.sqrt(252) * 100
sharpe_dcc = (port_ret_dcc.mean() / port_ret_dcc.std()) * np.sqrt(252)
sharpe_ccc = (port_ret_ccc.mean() / port_ret_ccc.std()) * np.sqrt(252)
sharpe_stat = (port_ret_static.mean() / port_ret_static.std()) * np.sqrt(252)

stats_text = (f'DCC:    Vol={ann_vol_dcc:.1f}%  SR={sharpe_dcc:.2f}\n'
              f'CCC:    Vol={ann_vol_ccc:.1f}%  SR={sharpe_ccc:.2f}\n'
              f'Static: Vol={ann_vol_stat:.1f}%  SR={sharpe_stat:.2f}')
ax.text(0.98, 0.05, stats_text, transform=ax.transAxes, fontsize=9,
        va='bottom', ha='right', family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))

save_chart(fig, 'ch5b_portfolio_comparison.pdf')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2: Rolling Hedging Effectiveness
# ══════════════════════════════════════════════════════════════════════════════
print("[2/3] Hedging effectiveness...")

# SP500 spot vs ES futures
hedge_idx = returns[['SP500', 'ES']].dropna().index
r_spot = returns['SP500'].reindex(hedge_idx).values
r_fut = returns['ES'].reindex(hedge_idx).values

# Fit GARCH on both
vol_spot, z_spot = fit_garch(pd.Series(r_spot, index=hedge_idx))
vol_fut, z_fut = fit_garch(pd.Series(r_fut, index=hedge_idx))

hidx = vol_spot.dropna().index.intersection(vol_fut.dropna().index)
hidx = hidx.intersection(z_spot.dropna().index).intersection(z_fut.dropna().index)

z1h = z_spot.reindex(hidx).values
z2h = z_fut.reindex(hidx).values
rho_dcc_h = estimate_dcc_corr(z1h, z2h)
rho_ccc_h = float(np.corrcoef(z1h, z2h)[0, 1])

vs = vol_spot.reindex(hidx).values
vf = vol_fut.reindex(hidx).values
r_s = returns['SP500'].reindex(hidx).values
r_f = returns['ES'].reindex(hidx).values

# Hedge ratios
hr_dcc = rho_dcc_h * vs / vf
hr_ccc = rho_ccc_h * vs / vf
# OLS static
from numpy.polynomial.polynomial import polyfit
ols_beta = np.polyfit(r_f, r_s, 1)[0]
hr_ols = np.full(len(hidx), ols_beta)

# Hedged returns
hedged_dcc = r_s - hr_dcc * r_f
hedged_ccc = r_s - hr_ccc * r_f
hedged_ols = r_s - hr_ols * r_f

# Rolling HE (252 day window)
window = 252
he_dcc = np.full(len(hidx), np.nan)
he_ccc = np.full(len(hidx), np.nan)
he_ols = np.full(len(hidx), np.nan)

for t in range(window, len(hidx)):
    var_unhedged = np.var(r_s[t-window:t])
    if var_unhedged > 0:
        he_dcc[t] = 1 - np.var(hedged_dcc[t-window:t]) / var_unhedged
        he_ccc[t] = 1 - np.var(hedged_ccc[t-window:t]) / var_unhedged
        he_ols[t] = 1 - np.var(hedged_ols[t-window:t]) / var_unhedged

fig, ax = plt.subplots(figsize=(10, 5))
make_transparent(fig, ax)
ax.plot(hidx, he_dcc * 100, color=MainBlue, lw=1.5, label='DCC Hedge')
ax.plot(hidx, he_ccc * 100, color=IDAred, lw=1.2, label='CCC Hedge', alpha=0.85)
ax.plot(hidx, he_ols * 100, color='gray', lw=1.2, ls='--', label='OLS Static Hedge', alpha=0.7)

for start, end, lbl in [('2007-07-01', '2009-03-31', 'GFC'),
                         ('2020-02-01', '2020-06-30', 'COVID')]:
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color=Crimson)

ax.set_ylabel('Hedging Effectiveness HE (%)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
ax.set_ylim([85, 100])

save_chart(fig, 'ch5b_hedge_effectiveness.pdf')

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3: VaR by VIX Regime
# ══════════════════════════════════════════════════════════════════════════════
print("[3/3] VaR by VIX regime...")

if vix is not None:
    # Portfolio: 60% SP500, 40% DAX
    w_var = np.array([0.6, 0.4])
    var_assets = ['SP500', 'DAX']
    var_idx = returns[var_assets].dropna().index
    v1, z1v = fit_garch(returns['SP500'].reindex(var_idx))
    v2, z2v = fit_garch(returns['DAX'].reindex(var_idx))
    vidx = v1.dropna().index.intersection(v2.dropna().index)
    vidx = vidx.intersection(z1v.dropna().index).intersection(z2v.dropna().index)

    rho_dcc_v = estimate_dcc_corr(z1v.reindex(vidx).values, z2v.reindex(vidx).values)
    rho_ccc_v = float(np.corrcoef(z1v.reindex(vidx).values, z2v.reindex(vidx).values)[0, 1])

    s1 = v1.reindex(vidx).values
    s2 = v2.reindex(vidx).values

    # DCC portfolio vol
    port_vol_dcc = np.sqrt(w_var[0]**2 * s1**2 + w_var[1]**2 * s2**2 +
                           2 * w_var[0] * w_var[1] * rho_dcc_v * s1 * s2)
    port_vol_ccc = np.sqrt(w_var[0]**2 * s1**2 + w_var[1]**2 * s2**2 +
                           2 * w_var[0] * w_var[1] * rho_ccc_v * s1 * s2)

    var99_dcc = 2.326 * port_vol_dcc * 100  # in %
    var99_ccc = 2.326 * port_vol_ccc * 100

    # Actual portfolio returns
    port_ret_var = (w_var[0] * returns['SP500'].reindex(vidx) +
                    w_var[1] * returns['DAX'].reindex(vidx)).values * 100

    # VIX regimes
    vix_aligned = vix.reindex(vidx).ffill().values

    calm = vix_aligned < 15
    turbulent = (vix_aligned >= 15) & (vix_aligned <= 30)
    crisis = vix_aligned > 30

    regimes = ['Calm\n(VIX < 15)', 'Turbulent\n(VIX 15–30)', 'Crisis\n(VIX > 30)']
    masks = [calm, turbulent, crisis]

    # Mean VaR by regime
    var_dcc_means = [np.nanmean(var99_dcc[m]) for m in masks]
    var_ccc_means = [np.nanmean(var99_ccc[m]) for m in masks]

    # Mean correlation by regime
    rho_means = [np.nanmean(rho_dcc_v[m]) for m in masks]

    # Violation rates
    viol_dcc = [np.nanmean(port_ret_var[m] < -var99_dcc[m]) * 100 if m.sum() > 0 else 0 for m in masks]
    viol_ccc = [np.nanmean(port_ret_var[m] < -var99_ccc[m]) * 100 if m.sum() > 0 else 0 for m in masks]

    x = np.arange(3)
    width = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    make_transparent(fig, [ax1, ax2])

    # Left: Mean VaR by regime
    bars1 = ax1.bar(x - width/2, var_dcc_means, width, color=MainBlue, label='DCC VaR 99%', alpha=0.85)
    bars2 = ax1.bar(x + width/2, var_ccc_means, width, color=IDAred, label='CCC VaR 99%', alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, fontsize=10)
    ax1.set_ylabel('Mean 99% VaR (%)')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

    # Add rho annotation on bars
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        ax1.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.1,
                f'ρ={rho_means[i]:.2f}', ha='center', fontsize=8, color=MainBlue)

    # Right: Violation rates
    bars3 = ax2.bar(x - width/2, viol_dcc, width, color=MainBlue, label='DCC violations', alpha=0.85)
    bars4 = ax2.bar(x + width/2, viol_ccc, width, color=IDAred, label='CCC violations', alpha=0.85)
    ax2.axhline(y=1.0, color='black', ls='--', lw=1, label='Expected (1%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes, fontsize=10)
    ax2.set_ylabel('VaR Violation Rate (%)')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)

    fig.tight_layout(pad=2)
    save_chart(fig, 'ch5b_var_regime.pdf')
else:
    print("  SKIPPED: VIX data unavailable")

print("\nDone.")
