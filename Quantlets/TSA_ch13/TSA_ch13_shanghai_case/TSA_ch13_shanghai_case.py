"""
LPPL Chapter 13 Charts — TSA Color Scheme + Real Data
- Transparent backgrounds everywhere
- Legends at the bottom (outside the plot)
- TSA color palette
- Real market data via yfinance + LPPL fitting via scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import yfinance as yf
from scipy.optimize import differential_evolution
import json, os
import warnings
warnings.filterwarnings('ignore')

# ── TSA color scheme ──
MainBlue   = '#1A3A6E'
Crimson    = '#DC3545'
Forest     = '#2E7D32'
Amber      = '#B5853F'
Orange     = '#E67E22'
Purple     = '#8E44AD'
DarkGray   = '#333333'
MediumGray = '#808080'

# ── Global style ──
plt.rcParams.update({
    'figure.facecolor':   'none',
    'axes.facecolor':     'none',
    'savefig.facecolor':  'none',
    'legend.facecolor':   'none',
    'legend.edgecolor':   'none',
    'legend.framealpha':  0,
    'font.size':          12,
    'axes.titlesize':     14,
    'axes.labelsize':     12,
    'legend.fontsize':    11,
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
})

OUTPUT_DIR = '/Users/danielpele/Documents/TSA/charts'


def save_fig(fig, name, dpi=150):
    path = f'{OUTPUT_DIR}/ch13_lppl_{name}.png'
    fig.savefig(path, dpi=dpi, bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.close(fig)
    print(f'  Saved {path}')


# =========================================================================
# REAL DATA INFRASTRUCTURE
# =========================================================================
_data_cache = {}
ALL_PARAMS = {}

def download_prices(ticker, start, end):
    key = f"{ticker}_{start}_{end}"
    if key not in _data_cache:
        df = yf.download(ticker, start=start, end=end, progress=False)
        close = df['Close']
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        _data_cache[key] = close.dropna()
    return _data_cache[key]


def _lppl_linear(t, y, tc, m, omega):
    dt = tc - t
    ok = dt > 0
    t_v, y_v, dt_v = t[ok], y[ok], dt[ok]
    if len(t_v) < 10:
        return None
    f = dt_v ** m
    X = np.column_stack([np.ones(len(t_v)), f,
                         f * np.cos(omega * np.log(dt_v)),
                         f * np.sin(omega * np.log(dt_v))])
    coeffs, _, _, _ = np.linalg.lstsq(X, y_v, rcond=None)
    fitted = X @ coeffs
    ssr = float(np.sum((y_v - fitted) ** 2))
    return coeffs, ssr, fitted


def fit_lppl(prices, tc_range=None, m_range=(0.1, 0.9), omega_range=(4, 25)):
    y = np.log(prices.values.astype(float))
    t = np.arange(len(y), dtype=float)
    if tc_range is None:
        tc_range = (len(y) - 5, len(y) + len(y) * 0.2)

    def objective(p):
        tc, m, omega = p
        res = _lppl_linear(t, y, tc, m, omega)
        if res is None:
            return 1e12
        coeffs, ssr, _ = res
        if coeffs[1] >= 0:
            return ssr * 100
        return ssr

    result = differential_evolution(objective, [tc_range, m_range, omega_range],
                                     seed=42, maxiter=1000, tol=1e-12,
                                     popsize=40, mutation=(0.5, 1.5), recombination=0.9)
    tc, m, omega = result.x
    coeffs, ssr, _ = _lppl_linear(t, y, tc, m, omega)
    A, B, C1, C2 = coeffs
    C = np.sqrt(C1**2 + C2**2)
    phi = np.arctan2(C2, C1)
    lam = np.exp(2 * np.pi / omega)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ssr / sst if sst > 0 else 0.0
    dt_all = np.maximum(tc - t, 0.01)
    f_all = dt_all ** m
    fitted_all = A + B*f_all + C1*f_all*np.cos(omega*np.log(dt_all)) + C2*f_all*np.sin(omega*np.log(dt_all))
    return {'tc': tc, 'm': m, 'omega': omega, 'A': A, 'B': B,
            'C': C, 'C1': C1, 'C2': C2, 'phi': phi,
            'lambda': lam, 'R2': r2, 'ssr': ssr,
            't': t, 'log_price': y, 'fitted_log': fitted_all, 'prices': prices}


def lppl_curve(t_arr, p):
    dt = np.maximum(p['tc'] - t_arr, 1e-6)
    f = dt ** p['m']
    return p['A'] + p['B']*f + p['C1']*f*np.cos(p['omega']*np.log(dt)) + p['C2']*f*np.sin(p['omega']*np.log(dt))


def bootstrap_ci(prices, p0, n_boot=200):
    y = np.log(prices.values.astype(float))
    t = np.arange(len(y), dtype=float)
    resid = y - p0['fitted_log']
    tc_b, m_b, w_b = [], [], []
    for i in range(n_boot):
        rng = np.random.RandomState(i)
        y_b = p0['fitted_log'] + rng.choice(resid, len(resid), replace=True)
        def obj(p):
            res = _lppl_linear(t, y_b, p[0], p[1], p[2])
            if res is None: return 1e12
            if res[0][1] >= 0: return res[1] * 100
            return res[1]
        try:
            r = differential_evolution(obj,
                [(max(p0['tc']-30, len(y)-5), p0['tc']+30),
                 (max(0.1, p0['m']-0.15), min(0.9, p0['m']+0.15)),
                 (max(4, p0['omega']-3), min(25, p0['omega']+3))],
                seed=i, maxiter=200, tol=1e-8, popsize=15)
            tc_b.append(r.x[0]); m_b.append(r.x[1]); w_b.append(r.x[2])
        except Exception:
            pass
    def ci(a):
        a = np.array(a)
        return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if len(a) > 10 else (np.nan, np.nan)
    return {'tc_ci': ci(tc_b), 'm_ci': ci(m_b), 'omega_ci': ci(w_b)}


# =========================================================================
# 1. Bubble Growth
# =========================================================================

# =========================================================================
# TSA_ch13_shanghai_case
# =========================================================================
def chart_shanghai_case():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    prices = download_prices('000001.SS', '2014-11-01', '2015-06-15')
    p = fit_lppl(prices)
    ALL_PARAMS['shanghai'] = p

    ax = axes[0]
    dates = prices.index
    ax.plot(dates, prices.values, color=MainBlue, linewidth=2, alpha=0.8, label='Shanghai Composite')
    ax.plot(dates, np.exp(p['fitted_log']), color=Crimson, linewidth=3, label='LPPL Fit')
    peak_i = int(prices.values.argmax())
    peak_date = dates[peak_i]
    ax.axvline(peak_date, color=Orange, linestyle='--', linewidth=2.5)
    ax.annotate(f'June 2015\nPeak: {prices.values[peak_i]:,.0f}',
                xy=(peak_date, prices.values[peak_i]),
                xytext=(-80, 10), textcoords='offset points', fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color=MediumGray),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_xlabel('Date'); ax.set_ylabel('Shanghai Composite Index')
    ax.set_title('Shanghai 2015: LPPL Fit (Real Data)', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)

    # Right panel: simplified CI from rolling LPPL R² on real data
    ax = axes[1]
    full_prices = download_prices('000001.SS', '2014-07-01', '2015-06-15')
    N = len(full_prices)
    ci_vals = np.zeros(N)
    for i in range(60, N):
        window = full_prices.iloc[max(0, i-120):i]
        if len(window) < 40:
            continue
        try:
            pf = fit_lppl(window, tc_range=(len(window)-5, len(window)+30),
                          m_range=(0.1, 0.9), omega_range=(4, 25))
            score = max(0, pf['R2']) if pf['B'] < 0 else 0
            ci_vals[i] = min(score, 1.0)
        except Exception:
            ci_vals[i] = 0
    ci_smooth = pd.Series(ci_vals, index=full_prices.index).rolling(10, min_periods=1).mean()
    ax.fill_between(full_prices.index, 0, ci_smooth.values, color=Crimson, alpha=0.4)
    ax.axhline(0.5, color=MediumGray, linestyle='--', linewidth=2, label='Warning')
    ax.axhline(0.7, color=Orange, linestyle='--', linewidth=2, label='High Alert')
    ax.axvline(pd.Timestamp('2015-04-15'), color=MainBlue, linestyle=':', linewidth=2,
               label='FCO Warning (Apr 2015)')
    ax.set_xlabel('Date'); ax.set_ylabel('Confidence Indicator')
    ax.set_title('CI Reached High Alert Before Peak', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
    ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

    plt.tight_layout()
    save_fig(fig, 'shanghai_case')


if __name__ == '__main__':
    print('Generating shanghai_case chart...')
    chart_shanghai_case()
    print('Done!')
