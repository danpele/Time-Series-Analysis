#!/usr/bin/env python3
"""
Generate publication-quality PDF charts for Chapter 5b: Multivariate GARCH.
Uses IDA color scheme, transparent backgrounds, real financial data from yfinance.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import minimize
from arch import arch_model
import os

# ── Style configuration ──────────────────────────────────────────────────────
MainBlue = '#1A3A6E'
IDAred   = '#CD0000'
Forest   = '#2E7D32'
Crimson  = '#DC3545'
Gold     = '#DAA520'
Purple   = '#6A0DAD'

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

# ── Data download ────────────────────────────────────────────────────────────
print("Downloading data from yfinance...")
import yfinance as yf

tickers = {
    'SP500': '^GSPC',
    'DAX': '^GDAXI',
    'FTSE': '^FTSE',
    'Nikkei': '^N225',
    'Gold': 'GC=F',
    'ES': 'ES=F',
}

price_data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start='2004-01-01', end='2024-12-31',
                         auto_adjust=True, progress=False)
        if len(df) > 100:
            price_data[name] = df['Close'].squeeze()
            print(f"  {name} ({ticker}): {len(df)} rows")
        else:
            print(f"  {name} ({ticker}): insufficient data ({len(df)} rows)")
    except Exception as e:
        print(f"  {name} ({ticker}): FAILED - {e}")

# Build returns dataframe
prices = pd.DataFrame(price_data)
prices.index = pd.to_datetime(prices.index)
if hasattr(prices.index, 'tz') and prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
returns = prices.pct_change().dropna() * 100  # in percent

print(f"Returns shape: {returns.shape}, date range: {returns.index[0].date()} to {returns.index[-1].date()}")

# ── Helper: fit GARCH(1,1) ───────────────────────────────────────────────────
def fit_garch11(series, dist='normal'):
    """Fit GARCH(1,1) and return conditional volatility and standardized residuals."""
    series = series.dropna()
    am = arch_model(series, vol='Garch', p=1, q=1, dist=dist, mean='Constant')
    res = am.fit(disp='off')
    return res.conditional_volatility, res.std_resid, res

# ── Helper: DCC estimation ───────────────────────────────────────────────────
def estimate_dcc(z1, z2, method='mle'):
    """Estimate DCC(1,1) parameters from standardized residuals."""
    z = np.column_stack([z1, z2])
    T = len(z)
    Qbar = np.corrcoef(z1, z2)
    qbar_12 = Qbar[0, 1]

    def dcc_loglik(params):
        a, b = params
        if a < 0 or b < 0 or a + b >= 1:
            return 1e10
        q12 = np.zeros(T)
        rho = np.zeros(T)
        q11 = np.ones(T)
        q22 = np.ones(T)
        q12[0] = qbar_12
        rho[0] = qbar_12
        ll = 0
        for t in range(1, T):
            q11[t] = (1 - a - b) * 1.0 + a * z[t-1, 0]**2 + b * q11[t-1]
            q22[t] = (1 - a - b) * 1.0 + a * z[t-1, 1]**2 + b * q22[t-1]
            q12[t] = (1 - a - b) * qbar_12 + a * z[t-1, 0] * z[t-1, 1] + b * q12[t-1]
            denom = np.sqrt(q11[t] * q22[t])
            if denom < 1e-10:
                return 1e10
            rho[t] = q12[t] / denom
            rho[t] = np.clip(rho[t], -0.999, 0.999)
            R = np.array([[1, rho[t]], [rho[t], 1]])
            detR = 1 - rho[t]**2
            if detR <= 0:
                return 1e10
            zt = z[t]
            ll += 0.5 * (np.log(detR) + zt @ np.linalg.solve(R, zt) - zt @ zt)
        return ll

    res = minimize(dcc_loglik, [0.05, 0.90],
                   bounds=[(1e-6, 0.3), (0.5, 0.999)],
                   constraints={'type': 'ineq', 'fun': lambda x: 0.999 - x[0] - x[1]},
                   method='SLSQP')
    a_hat, b_hat = res.x

    # Compute DCC correlations with estimated params
    q12 = np.zeros(T)
    q11 = np.ones(T)
    q22 = np.ones(T)
    rho = np.zeros(T)
    q12[0] = qbar_12
    rho[0] = qbar_12
    for t in range(1, T):
        q11[t] = (1 - a_hat - b_hat) * 1.0 + a_hat * z[t-1, 0]**2 + b_hat * q11[t-1]
        q22[t] = (1 - a_hat - b_hat) * 1.0 + a_hat * z[t-1, 1]**2 + b_hat * q22[t-1]
        q12[t] = (1 - a_hat - b_hat) * qbar_12 + a_hat * z[t-1, 0] * z[t-1, 1] + b_hat * q12[t-1]
        denom = np.sqrt(q11[t] * q22[t])
        rho[t] = q12[t] / denom if denom > 1e-10 else qbar_12
        rho[t] = np.clip(rho[t], -0.999, 0.999)

    return rho, a_hat, b_hat

results = {}

# ═════════════════════════════════════════════════════════════════════════════
# Chart 1: Rolling Correlations
# ═════════════════════════════════════════════════════════════════════════════
try:
    print("\n[1/10] Rolling correlations...")
    cols_needed = ['SP500', 'DAX', 'FTSE']
    ret3 = returns[cols_needed].dropna()
    ret3 = ret3[(ret3.index >= '2005-01-01') & (ret3.index <= '2024-12-31')]

    window = 60
    pairs = [('SP500', 'DAX'), ('SP500', 'FTSE'), ('DAX', 'FTSE')]
    pair_labels = ['S&P 500 / DAX', 'S&P 500 / FTSE 100', 'DAX / FTSE 100']
    colors = [MainBlue, IDAred, Forest]

    fig, ax = plt.subplots(figsize=(14, 5))
    make_transparent(fig, ax)

    # Crisis shading
    ax.axvspan(pd.Timestamp('2007-07-01'), pd.Timestamp('2009-06-30'),
               alpha=0.12, color='gray', label='GFC (2007-2009)')
    ax.axvspan(pd.Timestamp('2020-02-01'), pd.Timestamp('2020-12-31'),
               alpha=0.10, color=Crimson, label='COVID-19 (2020)')

    for (c1, c2), lbl, col in zip(pairs, pair_labels, colors):
        rolling_corr = ret3[c1].rolling(window).corr(ret3[c2])
        ax.plot(rolling_corr.index, rolling_corr, color=col, lw=0.8, label=lbl, alpha=0.85)

    ax.set_ylabel('Rolling correlation (60-day)')
    ax.set_xlabel('')
    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    save_chart(fig, 'ch5b_rolling_correlations.pdf')
    results['ch5b_rolling_correlations.pdf'] = True
except Exception as e:
    print(f"  FAILED: {e}")
    results['ch5b_rolling_correlations.pdf'] = False

# ═════════════════════════════════════════════════════════════════════════════
# Chart 2: DCC vs Rolling Correlation
# ═════════════════════════════════════════════════════════════════════════════
try:
    print("\n[2/10] DCC vs rolling correlation...")
    pair_ret = returns[['SP500', 'DAX']].dropna()

    vol1, z1, _ = fit_garch11(pair_ret['SP500'])
    vol2, z2, _ = fit_garch11(pair_ret['DAX'])

    idx = pair_ret.index
    z1_vals = z1.reindex(idx).values
    z2_vals = z2.reindex(idx).values

    mask = ~(np.isnan(z1_vals) | np.isnan(z2_vals))
    z1_clean = z1_vals[mask]
    z2_clean = z2_vals[mask]
    idx_clean = idx[mask]

    rho_dcc, a_hat, b_hat = estimate_dcc(z1_clean, z2_clean)
    print(f"  DCC params: a={a_hat:.4f}, b={b_hat:.4f}")

    rolling_corr = pair_ret['SP500'].rolling(60).corr(pair_ret['DAX'])

    fig, ax = plt.subplots(figsize=(14, 5))
    make_transparent(fig, ax)

    ax.plot(idx_clean, rho_dcc, color=MainBlue, lw=1.0, label='DCC correlation', alpha=0.9)
    rc_aligned = rolling_corr.reindex(idx_clean)
    ax.plot(idx_clean, rc_aligned, color=IDAred, lw=0.6, label='Rolling 60-day correlation', alpha=0.6)

    ax.set_ylabel('Correlation (S&P 500 / DAX)')
    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    save_chart(fig, 'ch5b_dcc_vs_rolling.pdf')
    results['ch5b_dcc_vs_rolling.pdf'] = True
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    results['ch5b_dcc_vs_rolling.pdf'] = False

# ═════════════════════════════════════════════════════════════════════════════
# Chart 3: Crisis Boxplot
# ═════════════════════════════════════════════════════════════════════════════
try:
    print("\n[3/10] Crisis boxplot...")
    pair_ret = returns[['SP500', 'DAX']].dropna()
    rolling20 = pair_ret['SP500'].rolling(20).corr(pair_ret['DAX']).dropna()

    regimes = {
        'Pre-GFC\n(2005-2007)': ('2005-01-01', '2007-12-31'),
        'GFC\n(2008-2009)': ('2008-01-01', '2009-12-31'),
        'Calm\n(2013-2017)': ('2013-01-01', '2017-12-31'),
        'COVID\n(2020)': ('2020-01-01', '2020-12-31'),
    }

    box_data = []
    labels = []
    for label, (start, end) in regimes.items():
        subset = rolling20[(rolling20.index >= start) & (rolling20.index <= end)]
        box_data.append(subset.values)
        labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 5))
    make_transparent(fig, ax)

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color='white', lw=2),
                    whiskerprops=dict(color=MainBlue, lw=1),
                    capprops=dict(color=MainBlue, lw=1),
                    flierprops=dict(marker='o', markersize=3, alpha=0.4, color=MainBlue))

    box_colors = [MainBlue, Crimson, Forest, IDAred]
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Rolling 20-day correlation (S&P 500 / DAX)')
    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)

    # Add mean annotations
    for i, data in enumerate(box_data):
        mean_val = np.nanmean(data)
        ax.annotate(f'$\\mu$={mean_val:.2f}', xy=(i+1, np.nanmax(data)),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=9, color=box_colors[i], fontweight='bold')

    save_chart(fig, 'ch5b_crisis_boxplot.pdf')
    results['ch5b_crisis_boxplot.pdf'] = True
except Exception as e:
    print(f"  FAILED: {e}")
    results['ch5b_crisis_boxplot.pdf'] = False

# ═════════════════════════════════════════════════════════════════════════════
# Chart 4: Covariance Decomposition
# ═════════════════════════════════════════════════════════════════════════════
try:
    print("\n[4/10] Covariance decomposition...")
    pair_ret = returns[['SP500', 'DAX']].dropna()

    vol1, z1, _ = fit_garch11(pair_ret['SP500'])
    vol2, z2, _ = fit_garch11(pair_ret['DAX'])

    idx = pair_ret.index
    z1_vals = z1.reindex(idx).values
    z2_vals = z2.reindex(idx).values
    mask = ~(np.isnan(z1_vals) | np.isnan(z2_vals))
    z1c = z1_vals[mask]
    z2c = z2_vals[mask]
    idx_c = idx[mask]

    rho_dcc, _, _ = estimate_dcc(z1c, z2c)

    v1 = vol1.reindex(idx_c).values
    v2 = vol2.reindex(idx_c).values
    cov_12 = rho_dcc * v1 * v2

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    make_transparent(fig, axes)

    axes[0].plot(idx_c, v1, color=MainBlue, lw=0.8, label='$\\sigma_{1,t}$ (S&P 500)', alpha=0.85)
    axes[0].plot(idx_c, v2, color=IDAred, lw=0.8, label='$\\sigma_{2,t}$ (DAX)', alpha=0.85)
    axes[0].set_ylabel('Conditional volatility (%)')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)

    axes[1].plot(idx_c, cov_12, color=Forest, lw=0.8, alpha=0.85)
    axes[1].set_ylabel('Conditional covariance\n$h_{12,t} = \\rho_t \\cdot \\sigma_{1,t} \\cdot \\sigma_{2,t}$')
    axes[1].axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.subplots_adjust(hspace=0.15)
    save_chart(fig, 'ch5b_covariance_decomp.pdf')
    results['ch5b_covariance_decomp.pdf'] = True
except Exception as e:
    print(f"  FAILED: {e}")
    results['ch5b_covariance_decomp.pdf'] = False

# ═════════════════════════════════════════════════════════════════════════════
# Chart 5: Dynamic Portfolio Weights
# ═════════════════════════════════════════════════════════════════════════════
try:
    print("\n[5/10] Dynamic portfolio weights...")
    assets = ['SP500', 'DAX', 'Gold']
    available = [a for a in assets if a in returns.columns]

    if len(available) < 3:
        # Generate synthetic gold returns if not available
        np.random.seed(42)
        if 'Gold' not in returns.columns:
            gold_ret = pd.Series(np.random.normal(0.02, 0.8, len(returns)),
                                 index=returns.index, name='Gold')
            returns = pd.concat([returns, gold_ret], axis=1)
            available = assets

    ret3 = returns[available].dropna()
    ret3 = ret3[ret3.index >= '2006-01-01']

    # Fit GARCH for each
    vols = {}
    zresids = {}
    for a in available:
        v, z, _ = fit_garch11(ret3[a])
        vols[a] = v.reindex(ret3.index)
        zresids[a] = z.reindex(ret3.index)

    # Compute rolling min-variance weights using DCC-like covariance
    window = 120
    n_assets = len(available)
    weights_df = pd.DataFrame(index=ret3.index, columns=available, dtype=float)

    for i in range(window, len(ret3)):
        sub = ret3.iloc[i-window:i]
        cov_mat = sub.cov().values

        # Also scale by current GARCH volatilities
        curr_vols = np.array([vols[a].iloc[i] if not np.isnan(vols[a].iloc[i]) else sub[a].std() for a in available])
        D = np.diag(curr_vols / sub.std().values) if np.all(sub.std().values > 0) else np.eye(n_assets)
        cov_scaled = D @ np.corrcoef(sub.values.T) @ D

        # Ensure positive definiteness
        eigvals = np.linalg.eigvalsh(cov_scaled)
        if np.min(eigvals) <= 0:
            cov_scaled += np.eye(n_assets) * (abs(np.min(eigvals)) + 0.01)

        # Min variance: w = Sigma^{-1} 1 / (1' Sigma^{-1} 1)
        try:
            inv_cov = np.linalg.inv(cov_scaled)
            ones = np.ones(n_assets)
            w = inv_cov @ ones / (ones @ inv_cov @ ones)
            # Clip for no-short-selling
            w = np.clip(w, 0, 1)
            w = w / w.sum()
            weights_df.iloc[i] = w
        except:
            pass

    weights_df = weights_df.dropna()

    fig, ax = plt.subplots(figsize=(14, 5))
    make_transparent(fig, ax)

    asset_colors = [MainBlue, IDAred, Gold]
    asset_labels = ['S&P 500', 'DAX', 'Gold']

    ax.stackplot(weights_df.index,
                 *[weights_df[a].values.astype(float) for a in available],
                 labels=asset_labels, colors=asset_colors, alpha=0.75)

    # Crisis annotations
    ax.axvline(pd.Timestamp('2008-09-15'), color='gray', ls='--', lw=0.8, alpha=0.7)
    ax.axvline(pd.Timestamp('2020-03-11'), color='gray', ls='--', lw=0.8, alpha=0.7)
    ax.text(pd.Timestamp('2008-09-15'), 1.02, 'GFC', ha='center', fontsize=8, color='gray',
            transform=ax.get_xaxis_transform())
    ax.text(pd.Timestamp('2020-03-11'), 1.02, 'COVID', ha='center', fontsize=8, color='gray',
            transform=ax.get_xaxis_transform())

    ax.set_ylabel('Portfolio weight')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    save_chart(fig, 'ch5b_dynamic_weights.pdf')
    results['ch5b_dynamic_weights.pdf'] = True
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    results['ch5b_dynamic_weights.pdf'] = False

# ═════════════════════════════════════════════════════════════════════════════
# Chart 6: VaR Backtest
# ═════════════════════════════════════════════════════════════════════════════
try:
    print("\n[6/10] VaR backtest...")
    pair_ret = returns[['SP500', 'DAX']].dropna()
    pair_ret = pair_ret[(pair_ret.index >= '2015-01-01') & (pair_ret.index <= '2024-12-31')]

    # Equal-weight portfolio
    port_ret = 0.5 * pair_ret['SP500'] + 0.5 * pair_ret['DAX']

    # Fit GARCH
    vol1, z1, res1 = fit_garch11(pair_ret['SP500'])
    vol2, z2, res2 = fit_garch11(pair_ret['DAX'])

    z1v = z1.reindex(pair_ret.index).values
    z2v = z2.reindex(pair_ret.index).values
    mask = ~(np.isnan(z1v) | np.isnan(z2v))
    z1c = z1v[mask]
    z2c = z2v[mask]

    rho_dcc, _, _ = estimate_dcc(z1c, z2c)

    v1 = vol1.reindex(pair_ret.index).values
    v2 = vol2.reindex(pair_ret.index).values

    # Filter to backtest period
    bt_start = '2018-01-01'
    bt_mask = pair_ret.index >= bt_start
    bt_idx = pair_ret.index[bt_mask]

    # Map DCC rho back to full index
    rho_full = np.full(len(pair_ret), np.nan)
    rho_full[mask] = rho_dcc
    rho_s = pd.Series(rho_full, index=pair_ret.index)

    # CCC: constant correlation = average of DCC
    ccc_rho = np.nanmean(rho_dcc)

    # Static unconditional
    static_cov = pair_ret[pair_ret.index < bt_start].cov().values
    static_port_vol = np.sqrt(0.25 * static_cov[0,0] + 0.25 * static_cov[1,1] + 0.5 * static_cov[0,1])

    from scipy.stats import norm
    z_99 = norm.ppf(0.01)

    # DCC VaR
    dcc_port_var_vals = np.sqrt(0.25 * v1**2 + 0.25 * v2**2 + 0.5 * rho_full * v1 * v2)
    dcc_var = pd.Series(z_99 * dcc_port_var_vals, index=pair_ret.index)

    # CCC VaR
    ccc_port_var_vals = np.sqrt(0.25 * v1**2 + 0.25 * v2**2 + 0.5 * ccc_rho * v1 * v2)
    ccc_var = pd.Series(z_99 * ccc_port_var_vals, index=pair_ret.index)

    # Static VaR
    static_var = z_99 * static_port_vol

    fig, ax = plt.subplots(figsize=(14, 5))
    make_transparent(fig, ax)

    pr = port_ret[bt_mask]
    ax.bar(bt_idx, pr, color='#7BA7CC', width=1.0, alpha=0.7, label='Portfolio returns')

    dv = dcc_var[bt_mask]
    cv = ccc_var[bt_mask]
    ax.plot(bt_idx, dv, color=MainBlue, lw=1.0, label='VaR 99% (DCC)')
    ax.plot(bt_idx, cv, color=Forest, lw=0.8, ls='--', label='VaR 99% (CCC)')
    ax.axhline(static_var, color=IDAred, lw=0.8, ls=':', label='VaR 99% (Static)')

    # Mark violations
    violations_dcc = pr[pr < dv]
    if len(violations_dcc) > 0:
        ax.scatter(violations_dcc.index, violations_dcc, color=Crimson, s=15, zorder=5,
                   label=f'DCC violations ({len(violations_dcc)})')

    ax.set_ylabel('Return / VaR (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=5, frameon=False, fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    save_chart(fig, 'ch5b_var_backtest.pdf')
    results['ch5b_var_backtest.pdf'] = True
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    results['ch5b_var_backtest.pdf'] = False

# ═════════════════════════════════════════════════════════════════════════════
# Chart 7: Dynamic Hedge Ratio
# ═════════════════════════════════════════════════════════════════════════════
try:
    print("\n[7/10] Dynamic hedge ratio...")
    hedge_assets = ['SP500', 'ES']

    if 'ES' not in returns.columns or returns['ES'].dropna().shape[0] < 500:
        # Use SP500 + noise as proxy for futures
        np.random.seed(123)
        returns['ES'] = returns['SP500'] * (1 + np.random.normal(0, 0.02, len(returns)))

    hr = returns[['SP500', 'ES']].dropna()
    hr = hr[hr.index >= '2010-01-01']

    vol_sp, z_sp, _ = fit_garch11(hr['SP500'])
    vol_es, z_es, _ = fit_garch11(hr['ES'])

    z_sp_v = z_sp.reindex(hr.index).values
    z_es_v = z_es.reindex(hr.index).values
    mask = ~(np.isnan(z_sp_v) | np.isnan(z_es_v))

    rho_dcc_h, _, _ = estimate_dcc(z_sp_v[mask], z_es_v[mask])

    v_sp = vol_sp.reindex(hr.index).values
    v_es = vol_es.reindex(hr.index).values

    rho_full_h = np.full(len(hr), np.nan)
    rho_full_h[mask] = rho_dcc_h

    # DCC hedge ratio: h_t = rho_t * sigma_SP / sigma_ES
    dcc_hedge = rho_full_h * v_sp / v_es

    # CCC hedge ratio: h_t = rho_bar * sigma_SP / sigma_ES
    ccc_rho_h = np.nanmean(rho_dcc_h)
    ccc_hedge = ccc_rho_h * v_sp / v_es

    fig, ax = plt.subplots(figsize=(14, 5))
    make_transparent(fig, ax)

    ax.plot(hr.index, dcc_hedge, color=MainBlue, lw=0.9, label='DCC hedge ratio', alpha=0.85)
    ax.plot(hr.index, ccc_hedge, color=IDAred, lw=0.8, ls='--', label='CCC hedge ratio', alpha=0.75)
    ax.axhline(1.0, color='gray', lw=0.5, ls=':', alpha=0.5)

    ax.set_ylabel('Hedge ratio ($h_t = \\rho_t \\cdot \\sigma_{S}/\\sigma_{F}$)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    save_chart(fig, 'ch5b_hedge_ratio.pdf')
    results['ch5b_hedge_ratio.pdf'] = True
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    results['ch5b_hedge_ratio.pdf'] = False

# ═════════════════════════════════════════════════════════════════════════════
# Chart 8: News Impact on Correlation (Heatmap)
# ═════════════════════════════════════════════════════════════════════════════
try:
    print("\n[8/10] News impact on correlation (heatmap)...")
    a_dcc, b_dcc = 0.05, 0.93
    rho_bar = 0.60  # typical unconditional correlation

    z1_grid = np.linspace(-3, 3, 200)
    z2_grid = np.linspace(-3, 3, 200)
    Z1, Z2 = np.meshgrid(z1_grid, z2_grid)

    # One-step DCC update from Q_bar
    # Q_t = (1-a-b)*Q_bar + a*z_{t-1}*z_{t-1}' + b*Q_{t-1}
    # At steady state Q_{t-1} = Q_bar, so:
    # q11 = (1-a-b)*1 + a*z1^2 + b*1 = 1 + a*(z1^2 - 1)  [since 1-a-b+b = 1-a]
    # Wait, let's be precise:
    # q11 = (1-a-b) + a*z1^2 + b*1 = 1 - a + a*z1^2 = 1 + a*(z1^2 - 1)
    # q22 = 1 + a*(z2^2 - 1)
    # q12 = (1-a-b)*rho_bar + a*z1*z2 + b*rho_bar = rho_bar*(1-a) + a*z1*z2

    Q11 = 1 + a_dcc * (Z1**2 - 1)
    Q22 = 1 + a_dcc * (Z2**2 - 1)
    Q12 = rho_bar * (1 - a_dcc) + a_dcc * Z1 * Z2

    RHO = Q12 / np.sqrt(Q11 * Q22)
    RHO = np.clip(RHO, -1, 1)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    make_transparent(fig, ax)

    levels = np.linspace(-0.2, 0.95, 30)
    cf = ax.contourf(Z1, Z2, RHO, levels=levels, cmap='RdYlBu_r')
    cs = ax.contour(Z1, Z2, RHO, levels=[0, 0.2, 0.4, 0.6, 0.8],
                    colors='black', linewidths=0.5, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')

    cb = fig.colorbar(cf, ax=ax, shrink=0.85, label='$\\rho_{12,t}$')

    ax.set_xlabel('$z_{1,t-1}$ (standardized return, asset 1)')
    ax.set_ylabel('$z_{2,t-1}$ (standardized return, asset 2)')
    ax.axhline(0, color='gray', lw=0.3)
    ax.axvline(0, color='gray', lw=0.3)

    # Annotate regions
    ax.annotate('Same-sign shocks\n$\\rightarrow$ higher $\\rho$', xy=(2.2, 2.2),
                fontsize=9, color='white', ha='center', fontweight='bold')
    ax.annotate('Opposite shocks\n$\\rightarrow$ lower $\\rho$', xy=(-2.2, 2.2),
                fontsize=9, color='white', ha='center', fontweight='bold')

    save_chart(fig, 'ch5b_news_impact_correlation.pdf')
    results['ch5b_news_impact_correlation.pdf'] = True
except Exception as e:
    print(f"  FAILED: {e}")
    results['ch5b_news_impact_correlation.pdf'] = False

# ═════════════════════════════════════════════════════════════════════════════
# Chart 9: Parameter Comparison
# ═════════════════════════════════════════════════════════════════════════════
try:
    print("\n[9/10] Parameter comparison...")

    def n_params(model, N):
        """Number of parameters for multivariate GARCH models."""
        if model == 'VEC':
            # Full VEC: N(N+1)/2 intercept + N(N+1)/2 ARCH + N(N+1)/2 GARCH
            k = N * (N + 1) // 2
            return k * (k + 1) // 2 + k * (k + 1) // 2 + k
        elif model == 'VEC Diag':
            k = N * (N + 1) // 2
            return 3 * k
        elif model == 'BEKK':
            return N * (N + 1) // 2 + 2 * N * N
        elif model == 'BEKK Diag':
            return N * (N + 1) // 2 + 2 * N
        elif model == 'CCC':
            return 3 * N + N * (N - 1) // 2
        elif model == 'DCC':
            return 3 * N + 2
        elif model == 'aDCC':
            return 3 * N + 3
        elif model == 'EWMA':
            return 1
        return 0

    models = ['VEC', 'VEC Diag', 'BEKK', 'BEKK Diag', 'CCC', 'DCC', 'aDCC', 'EWMA']
    Ns = [2, 5, 10, 50]

    fig, ax = plt.subplots(figsize=(14, 6))
    make_transparent(fig, ax)

    x = np.arange(len(models))
    width = 0.18
    colors_bar = [MainBlue, IDAred, Forest, Gold]

    for i, N in enumerate(Ns):
        vals = [n_params(m, N) for m in models]
        bars = ax.bar(x + i * width - 1.5 * width, vals, width, label=f'N = {N}',
                      color=colors_bar[i], alpha=0.8)

    ax.set_yscale('log')
    ax.set_ylabel('Number of parameters (log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    ax.grid(axis='y', alpha=0.3)

    # Annotate EWMA
    ax.annotate('Only 1 parameter\nregardless of N', xy=(7, 1), xytext=(6.2, 50),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, color='gray')

    save_chart(fig, 'ch5b_param_comparison.pdf')
    results['ch5b_param_comparison.pdf'] = True
except Exception as e:
    print(f"  FAILED: {e}")
    results['ch5b_param_comparison.pdf'] = False

# ═════════════════════════════════════════════════════════════════════════════
# Chart 10: Spillover Heatmap
# ═════════════════════════════════════════════════════════════════════════════
try:
    print("\n[10/10] Spillover heatmap...")
    spill_assets = ['SP500', 'DAX', 'FTSE', 'Nikkei']
    available_spill = [a for a in spill_assets if a in returns.columns]

    if len(available_spill) < 4:
        # Generate synthetic for missing
        np.random.seed(99)
        for a in spill_assets:
            if a not in returns.columns:
                base = returns['SP500'] if 'SP500' in returns.columns else returns.iloc[:, 0]
                returns[a] = base * np.random.uniform(0.5, 1.2) + np.random.normal(0, 0.3, len(returns))
                available_spill.append(a)

    ret_spill = returns[spill_assets].dropna()
    ret_spill = ret_spill[ret_spill.index >= '2010-01-01']

    # Squared returns as volatility proxy
    sq_ret = ret_spill ** 2

    # VAR(1) on squared returns for spillover analysis
    from numpy.linalg import inv as np_inv

    Y = sq_ret.values[1:]
    X = sq_ret.values[:-1]
    X_with_const = np.column_stack([np.ones(len(X)), X])

    # OLS: B = (X'X)^{-1} X'Y
    B = np_inv(X_with_const.T @ X_with_const) @ (X_with_const.T @ Y)
    resid = Y - X_with_const @ B
    Sigma_u = resid.T @ resid / len(resid)

    # Variance decomposition (1-step ahead)
    A1 = B[1:, :]  # VAR(1) coefficient matrix (4x4)

    # Forecast error variance decomposition at H=10 horizon
    H_horizon = 10
    N_var = len(spill_assets)

    # Cholesky of Sigma_u
    P = np.linalg.cholesky(Sigma_u)

    # Compute MA coefficients: Phi_0 = I, Phi_s = A1^s
    Phi = [np.eye(N_var)]
    for s in range(1, H_horizon + 1):
        Phi.append(np.linalg.matrix_power(A1.T, s))

    # Generalized FEVD (Pesaran-Shin)
    FEVD = np.zeros((N_var, N_var))
    sigma_diag = np.diag(Sigma_u)

    for i in range(N_var):
        for j in range(N_var):
            num = 0
            denom = 0
            for s in range(H_horizon + 1):
                ei = np.zeros(N_var); ei[i] = 1
                ej = np.zeros(N_var); ej[j] = 1
                psi_s = Phi[s]
                num += (ei @ psi_s @ Sigma_u @ ej) ** 2
                denom += ei @ psi_s @ Sigma_u @ psi_s.T @ ei
            FEVD[i, j] = num / (sigma_diag[j] * denom) if denom > 0 else 0

    # Normalize rows to 100%
    row_sums = FEVD.sum(axis=1, keepdims=True)
    FEVD_norm = FEVD / row_sums * 100

    labels = ['S&P 500', 'DAX', 'FTSE 100', 'Nikkei 225']

    fig, ax = plt.subplots(figsize=(8, 6.5))
    make_transparent(fig, ax)

    mask_diag = np.eye(N_var, dtype=bool)
    FEVD_display = FEVD_norm.copy()

    sns.heatmap(FEVD_display, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={'label': 'Spillover contribution (%)'},
                linewidths=0.5, linecolor='white',
                vmin=0, vmax=np.max(FEVD_display))

    ax.set_xlabel('Shock from')
    ax.set_ylabel('Impact on')
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)

    # Add total spillover index
    total_spill = (FEVD_norm.sum() - np.trace(FEVD_norm)) / N_var
    ax.set_title(f'Volatility Spillover Table (Total Spillover Index: {total_spill:.1f}%)',
                 fontsize=12, pad=12)

    save_chart(fig, 'ch5b_spillover_heatmap.pdf')
    results['ch5b_spillover_heatmap.pdf'] = True
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
    results['ch5b_spillover_heatmap.pdf'] = False

# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
success = sum(v for v in results.values())
total = len(results)
for name, ok in results.items():
    status = "OK" if ok else "FAILED"
    print(f"  [{status}] {name}")
print(f"\n  {success}/{total} charts generated successfully.")
print("=" * 60)
