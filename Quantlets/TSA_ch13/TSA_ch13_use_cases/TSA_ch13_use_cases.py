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
# TSA_ch13_use_cases
# =========================================================================
def chart_use_cases():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Portfolio management
    ax = axes[0, 0]
    t = np.arange(100); np.random.seed(42)
    ci = np.clip(0.2 + 0.005*t + 0.1*np.sin(0.1*t) + 0.05*np.random.randn(100), 0, 1)
    ax.fill_between(t, 0, 1 - 0.8*ci, color=MainBlue, alpha=0.4, label='Position Size')
    ax.plot(t, ci, color=Crimson, linewidth=2, label='CI')
    ax.set_xlabel('Time'); ax.set_ylabel('Value')
    ax.set_title('Use Case 1: Dynamic Portfolio Management', fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.1)

    # Options hedging
    ax = axes[0, 1]
    ci_levels = [0.2, 0.4, 0.6, 0.8]
    hedge = [0.5, 1.0, 2.0, 4.0]
    colors = [Forest, Forest, Orange, Crimson]
    bars = ax.bar(range(4), hedge, color=colors, edgecolor='white', linewidth=2)
    ax.set_xticks(range(4)); ax.set_xticklabels([f'CI={v}' for v in ci_levels])
    ax.set_ylabel('Recommended Hedge Ratio (%)')
    ax.set_title('Use Case 2: Options Hedging Strategy', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, hedge):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val}%', ha='center', fontsize=11, fontweight='bold')

    # Early warning
    ax = axes[1, 0]
    t = np.arange(200); np.random.seed(123)
    lppl_ci = np.clip(0.25 + 0.002*t + 0.1*np.sin(0.03*t) + 0.05*np.random.randn(200), 0, 1)
    ax.plot(t, lppl_ci, color=MainBlue, linewidth=2, label='LPPL CI')
    ax.axhline(0.6, color=Crimson, linestyle='--', linewidth=2, label='Alert Level')
    ax.fill_between(t, 0.6, lppl_ci, where=lppl_ci > 0.6, color=Crimson, alpha=0.3)
    ax.set_xlabel('Time'); ax.set_ylabel('Systemic Risk Indicator')
    ax.set_title('Use Case 3: Central Bank Early Warning', fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

    # Backtest
    ax = axes[1, 1]
    crashes = ['2000\nDot-com', '2008\nFinancial', '2015\nShanghai', '2017\nBitcoin', '2021\nBitcoin']
    avoided = [23, 31, 18, 25, 22]
    bars = ax.bar(range(5), avoided, color=[Forest]*5, edgecolor='white', linewidth=2)
    ax.set_xticks(range(5)); ax.set_xticklabels(crashes, fontsize=10)
    ax.set_ylabel('Avoided Loss (%)')
    ax.set_title('Use Case 4: Backtest Results (Relative Performance)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, avoided):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'+{val}%', ha='center', fontsize=11, fontweight='bold', color=Forest)

    plt.tight_layout()
    save_fig(fig, 'use_cases')


if __name__ == '__main__':
    print('Generating use_cases chart...')
    chart_use_cases()
    print('Done!')
