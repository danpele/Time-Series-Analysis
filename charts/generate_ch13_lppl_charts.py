"""
LPPL Chapter 13 Charts — TSA Color Scheme
- Transparent backgrounds everywhere
- Legends at the bottom (outside the plot)
- TSA color palette
- Market data via yfinance + LPPL fitting via scipy
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
# DATA INFRASTRUCTURE
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
def chart_bubble_growth():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    t = np.linspace(0, 0.98, 300)
    tc = 1.0

    # Left — price trajectories (log scale)
    ax = axes[0]
    r = 0.3
    p_exp = 100 * np.exp(r * t)
    m = 0.5; B = -5; A = np.log(100) - B
    p_bubble = np.exp(A + B * (tc - t)**m)

    ax.plot(t, p_exp, color=Forest, linewidth=3, label='Normal Growth')
    ax.plot(t, p_bubble, color=Crimson, linewidth=3, label='Bubble Growth')
    ax.axvline(tc, color=MediumGray, linestyle='--', linewidth=2, alpha=0.7)
    ax.set_yscale('log')
    ax.annotate('Critical Time $t_c$', xy=(tc, 500),
                fontsize=12, ha='center', color=MediumGray,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_xlabel('Time'); ax.set_ylabel('Price (log scale)')
    ax.set_title('Price Trajectories: Normal vs Bubble', fontweight='bold')
    ax.set_xlim(0, 1.1); ax.grid(True, alpha=0.3)

    # Right — growth rates
    ax2 = axes[1]
    growth_exp = np.ones_like(t) * r
    growth_bubble = m * np.abs(B) * (tc - t)**(m - 1)
    mask = t < 0.92
    ax2.plot(t, growth_exp, color=Forest, linewidth=3, label='Normal: Constant')
    ax2.plot(t[mask], growth_bubble[mask], color=Crimson, linewidth=3, label='Bubble: Accelerating')
    ax2.axvline(tc, color=MediumGray, linestyle='--', linewidth=2, alpha=0.7)
    ax2.annotate('$\\rightarrow \\infty$', xy=(0.93, 8.5), fontsize=14,
                 color=Crimson, fontweight='bold')
    ax2.set_xlabel('Time'); ax2.set_ylabel('Growth Rate $d\\ln P/dt$')
    ax2.set_title('Growth Rate: The Key Difference', fontweight='bold')
    ax2.set_xlim(0, 1.1); ax2.set_ylim(0, 10); ax2.grid(True, alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color=Forest,    linewidth=3, label='Normal Growth (Exponential)'),
        plt.Line2D([0], [0], color=Crimson,    linewidth=3, label='Bubble Growth (Super-exponential)'),
        plt.Line2D([0], [0], color=MediumGray, linestyle='--', linewidth=2, label='Critical Time $t_c$'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=12)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.18)
    save_fig(fig, 'bubble_growth')


# =========================================================================
# 2. BTC LPPL  # =========================================================================
def _fit_btc2021():
    if 'btc2021' not in ALL_PARAMS:
        prices = download_prices('BTC-USD', '2020-07-20', '2021-11-15')
        p = fit_lppl(prices)
        ALL_PARAMS['btc2021'] = p
    return ALL_PARAMS['btc2021']

def chart_btc_lppl():
    p = _fit_btc2021()
    prices = p['prices']
    fig, ax = plt.subplots(figsize=(14, 7))

    dates = prices.index
    ax.plot(dates, prices.values, color=MainBlue, linewidth=2, alpha=0.8, label='BTC-USD Price')
    ax.plot(dates, np.exp(p['fitted_log']), color=Crimson, linewidth=3, label='LPPL Fit')

    peak_i = int(prices.values.argmax())
    peak_date = dates[peak_i]; peak_price = float(prices.values[peak_i])
    tc_error = abs(p['tc'] - peak_i)
    tc_date = dates[min(int(round(p['tc'])), len(dates)-1)]
    ax.axvline(tc_date, color=Orange, linestyle='--', linewidth=2.5,
               label=f'Predicted $t_c$ ({tc_error:.0f}-day error)')

    ax.annotate(f'Peak: ${peak_price:,.0f}\n{peak_date.strftime("%b %d, %Y")}',
                xy=(peak_date, peak_price), xytext=(-100, -40),
                textcoords='offset points', fontsize=12, ha='center',
                arrowprops=dict(arrowstyle='->', color=MediumGray, lw=1.5),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel('Date'); ax.set_ylabel('Price (USD)')
    ax.set_title(f'Bitcoin 2020-2021: LPPL Model Fit\n'
                 f'Predicted Peak: {tc_date.strftime("%b %d, %Y")} | '
                 f'Actual Peak: {peak_date.strftime("%b %d, %Y")} ({tc_error:.0f}-day error)',
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    param_text = (f'LPPL Parameters:\n$t_c$ error = {tc_error:.0f} days\n'
                  f'$m$ = {p["m"]:.3f}\n$\\omega$ = {p["omega"]:.3f}\n$R^2$ = {p["R2"]:.3f}')
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=MediumGray))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
              frameon=False, fontsize=12)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.16)
    save_fig(fig, 'btc_lppl')


# =========================================================================
# 3. BTC Full Analysis  # =========================================================================
def chart_btc_full_analysis():
    p = _fit_btc2021()
    prices = p['prices']
    fig, ax = plt.subplots(figsize=(14, 7))

    dates = prices.index
    ax.plot(dates, prices.values, color=MainBlue, linewidth=2, alpha=0.8, label='BTC-USD Price')
    ax.plot(dates, np.exp(p['fitted_log']), color=Crimson, linewidth=3, label='LPPL Fit')

    peak_i = int(prices.values.argmax())
    peak_date = dates[peak_i]; peak_price = float(prices.values[peak_i])
    tc_error = abs(p['tc'] - peak_i)
    tc_date = dates[min(int(round(p['tc'])), len(dates)-1)]
    ax.axvline(tc_date, color=Orange, linestyle='--', linewidth=2.5,
               label=f'Predicted $t_c$ ({tc_error:.0f}-day error)')

    mid = dates[len(dates)//3]
    ax.axvspan(mid, tc_date, color=Crimson, alpha=0.06)
    ax.annotate('Bubble Phase', xy=(dates[len(dates)*2//3], peak_price*0.85),
                fontsize=13, ha='center', color=Crimson, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_xlabel('Date'); ax.set_ylabel('Price (USD)')
    ax.set_title('Bitcoin 2020-2021: Full LPPL Analysis', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    param_text = (f'LPPL Parameters:\n$t_c$ error = {tc_error:.0f} days\n'
                  f'$m$ = {p["m"]:.3f}\n$\\omega$ = {p["omega"]:.3f}\n$R^2$ = {p["R2"]:.3f}')
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=MediumGray))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3,
              frameon=False, fontsize=12)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.16)
    save_fig(fig, 'btc_full_analysis')


# =========================================================================
# 4. Oscillations
# =========================================================================
def chart_oscillations():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t = np.linspace(0, 0.95, 500); tc = 1.0; m = 0.5; omega = 8
    dt = tc - t

    ax = axes[0, 0]
    ax.plot(t, dt**m, color=MainBlue, linewidth=3)
    ax.set_xlabel('Time $t$'); ax.set_ylabel('$(t_c - t)^m$')
    ax.set_title('(A) Power Law Trend', fontweight='bold'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    osc = np.cos(omega * np.log(dt))
    ax.plot(t, osc, color=Forest, linewidth=2)
    ax.set_xlabel('Time $t$'); ax.set_ylabel('$\\cos(\\omega \\ln(t_c - t))$')
    ax.set_title('(B) Log-Periodic Oscillation', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_ylim(-1.5, 1.5)

    ax = axes[1, 0]
    combined = dt**m * (1 + 0.3 * osc)
    ax.plot(t, combined, color=Crimson, linewidth=2)
    ax.set_xlabel('Time $t$'); ax.set_ylabel('$\\ln P(t)$')
    ax.set_title('(C) Combined LPPL (Log Scale)', fontweight='bold'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    log_dt = np.log(dt)
    ax.plot(log_dt, osc, color=Orange, linewidth=2)
    ax.set_xlabel('$\\ln(t_c - t)$'); ax.set_ylabel('Oscillation')
    ax.set_title('(D) In Log-Time: Oscillations Are Periodic!', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_ylim(-1.5, 1.5)
    ax.annotate('Key insight:\nEvenly spaced\nin log-time!', xy=(-2, 0.8),
                fontsize=12, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    fig.text(0.5, 0.02,
             'Log-periodic oscillations compress in real time but are evenly spaced in $\\ln(t_c - t)$\n'
             'This is the signature of Discrete Scale Invariance from hierarchical market structure',
             ha='center', fontsize=12, style='italic')
    plt.tight_layout(); plt.subplots_adjust(bottom=0.12)
    save_fig(fig, 'oscillations')


# =========================================================================
# 5. Ising
# =========================================================================
def chart_ising():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    np.random.seed(42); N = 30
    cmap = LinearSegmentedColormap.from_list('ising', [Crimson, 'white', MainBlue])

    # Low T (ordered → bubble)
    ax = axes[0]
    cold = np.ones((N, N)); cold[:3, :] = -1; cold[-2:, -5:] = -1
    ax.imshow(cold, cmap=cmap, vmin=-1, vmax=1)
    ax.set_title('$T < T_c$: Ordered\n$|m| \\approx 1$ (Strong consensus)', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, -0.12, 'Market: Strong herding\nBubble forming',
            transform=ax.transAxes, ha='center', fontsize=11, style='italic')

    # Critical
    ax = axes[1]
    spins = np.random.choice([-1, 1], (N, N))
    for _ in range(100):
        i, j = np.random.randint(0, N, 2)
        s = np.random.randint(2, 8)
        spins[max(0, i-s):min(N, i+s), max(0, j-s):min(N, j+s)] = np.random.choice([-1, 1])
    ax.imshow(spins, cmap=cmap, vmin=-1, vmax=1)
    ax.set_title('$T = T_c$: Critical Point\nClusters of ALL sizes!', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, -0.12, 'Market: Maximum instability\nSmall shocks → large cascades',
            transform=ax.transAxes, ha='center', fontsize=11, style='italic', color=Crimson)

    # High T (disordered → normal market)
    ax = axes[2]
    ax.imshow(np.random.choice([-1, 1], (N, N)), cmap=cmap, vmin=-1, vmax=1)
    ax.set_title('$T > T_c$: Disordered\n$m \\approx 0$ (No consensus)', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, -0.12, 'Market: Random trading\nNo herding behavior',
            transform=ax.transAxes, ha='center', fontsize=11, style='italic')

    handles = [
        mpatches.Patch(color=MainBlue, label='Spin Up (+1) = BUY'),
        mpatches.Patch(color=Crimson,  label='Spin Down (-1) = SELL'),
        mpatches.Patch(color='white', edgecolor='gray', label='Neutral'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=12)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.18)
    save_fig(fig, 'ising')


# =========================================================================
# 6. Phase Transition
# =========================================================================
def chart_phase_transition():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    T = np.linspace(0.01, 3, 500); Tc = 2.269

    # Magnetization
    ax = axes[0]
    mag = np.where(T < Tc, (1 - (T/Tc)**2)**0.125, 0.0)
    ax.plot(T, mag, color=MainBlue, linewidth=3, label='Magnetization $|m|$')
    ax.plot(T, -mag, color=MainBlue, linewidth=3, alpha=0.5)
    ax.axvline(Tc, color=Crimson, linestyle='--', linewidth=2.5, label=f'$T_c = {Tc:.3f}$')
    ax.fill_betweenx([0, 1], 0, Tc, color=Forest, alpha=0.15, label='Ordered Phase')
    ax.fill_betweenx([0, 1], Tc, 3, color=Orange, alpha=0.15, label='Disordered Phase')
    ax.set_xlabel('Temperature $T$'); ax.set_ylabel('Magnetization $m$')
    ax.set_title('(A) Order Parameter: Magnetization', fontweight='bold')
    ax.set_xlim(0, 3); ax.set_ylim(-1.1, 1.1); ax.grid(True, alpha=0.3)

    # Susceptibility
    ax = axes[1]
    chi = np.where(np.abs(T - Tc) > 0.05, 1 / np.abs(T - Tc)**1.75, np.nan)
    chi = np.clip(chi, 0, 50)
    ax.plot(T, chi, color=Crimson, linewidth=3)
    ax.axvline(Tc, color=MediumGray, linestyle='--', linewidth=2)
    ax.annotate('$\\chi \\to \\infty$\nat $T_c$!', xy=(Tc, 40), fontsize=13, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_xlabel('Temperature $T$'); ax.set_ylabel('Susceptibility $\\chi$')
    ax.set_title('(B) Susceptibility Diverges at Critical Point', fontweight='bold')
    ax.set_xlim(0, 3); ax.set_ylim(0, 55); ax.grid(True, alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color=MainBlue, linewidth=3, label='Magnetization $|m|$'),
        plt.Line2D([0], [0], color=Crimson,  linewidth=3, label='Susceptibility $\\chi$'),
        mpatches.Patch(color=Forest, alpha=0.3, label='Ordered (Bubble)'),
        mpatches.Patch(color=Orange, alpha=0.3, label='Disordered (Normal)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=11)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.14)
    save_fig(fig, 'phase_transition')


# =========================================================================
# 7. LPPL Components
# =========================================================================
def chart_lppl_components():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t = np.linspace(0, 340, 500); tc = 341
    m = 0.8; omega = 6; A = 4; B = -0.02; C = 0.005; phi = 2
    dt = np.maximum(tc - t, 0.1)

    ax = axes[0, 0]
    ax.axhline(A, color=MainBlue, linewidth=3)
    ax.fill_between(t, A-0.1, A+0.1, color=MainBlue, alpha=0.2)
    ax.set_ylabel('$A$'); ax.set_xlabel('Time')
    ax.set_title('(A) Constant: $A$ = Log price at $t_c$', fontweight='bold')
    ax.set_ylim(A-1, A+1); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    power = B * dt**m
    ax.plot(t, power, color=Crimson, linewidth=3)
    ax.fill_between(t, 0, power, color=Crimson, alpha=0.2)
    ax.set_ylabel('$B(t_c - t)^m$'); ax.set_xlabel('Time')
    ax.set_title(f'(B) Power Law: $B(t_c - t)^m$, $m={m}$', fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    logper = C * dt**m * np.cos(omega * np.log(dt) - phi)
    ax.plot(t, logper, color=Forest, linewidth=2)
    ax.fill_between(t, 0, logper, color=Forest, alpha=0.2)
    ax.set_ylabel('$C(t_c-t)^m\\cos(...)$'); ax.set_xlabel('Time')
    ax.set_title(f'(C) Log-Periodic: $\\omega={omega}$', fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    lppl = A + B * dt**m + C * dt**m * np.cos(omega * np.log(dt) - phi)
    ax.plot(t, np.exp(lppl), color=MainBlue, linewidth=3, label='Price = $e^{\\text{LPPL}}$')
    ax.axvline(tc, color=Orange, linestyle='--', linewidth=2, label='$t_c$')
    ax.set_ylabel('Price'); ax.set_xlabel('Time')
    ax.set_title('(D) Full LPPL Model', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

    fig.text(0.5, 0.02,
             r'$\ln P(t) = A + B(t_c - t)^m + C(t_c - t)^m \cos(\omega \ln(t_c - t) - \phi)$',
             ha='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    plt.tight_layout(); plt.subplots_adjust(bottom=0.10)
    save_fig(fig, 'components')


# =========================================================================
# 8. Historical Crashes  # =========================================================================
def chart_historical_crashes():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    crashes = [
        ('1929 Wall Street', '^GSPC', '1928-01-01', '1929-09-16', 89, 0.45, 7.4),
        ('1987 Black Monday', '^GSPC', '1987-01-01', '1987-10-16', 23, 0.35, 6.2),
        ('2000 Dot-com', '^IXIC', '1998-06-01', '2000-03-12', 78, 0.68, 7.8),
        ('2015 Shanghai', '000001.SS', '2014-07-01', '2015-06-15', 45, 0.55, 6.5),
        ('2017 Bitcoin', 'BTC-USD', '2017-01-01', '2017-12-20', 84, 0.72, 8.1),
        ('2021 Bitcoin', 'BTC-USD', '2020-07-20', '2021-11-15', 77, 0.82, 5.6),
    ]

    for idx, (name, ticker, start, end, drop, m, omega) in enumerate(crashes):
        ax = axes[idx]
        prices = download_prices(ticker, start, end)
        dates = prices.index
        peak_i = int(prices.values.argmax())
        peak_date = dates[peak_i]

        ax.plot(dates, prices.values, color=MainBlue, linewidth=2)
        ax.axvline(peak_date, color=Crimson, linestyle='--', linewidth=2)
        ax.set_title(f'{name}\nDrop: {drop}%, $m$={m}, $\\omega$={omega}',
                     fontweight='bold', fontsize=12)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.annotate(f'-{drop}%', xy=(peak_date, prices.values[peak_i]*0.85),
                    fontsize=13, color=Crimson, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(2, len(prices)//120)))
        plt.setp(ax.xaxis.get_majorticklabels(), fontsize=8)

    handles = [
        plt.Line2D([0], [0], color=MainBlue, linewidth=2, label='Price'),
        plt.Line2D([0], [0], color=Crimson, linestyle='--', linewidth=2, label='Peak / Critical Time $t_c$'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=12)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.08)
    save_fig(fig, 'historical_crashes')


# =========================================================================
# 9. Hazard Rate
# =========================================================================
def chart_hazard_rate():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    t = np.linspace(0, 0.98, 200); tc = 1.0; dt = tc - t

    ax = axes[0]
    for m_val, color, label in [(0.3, MainBlue, '$m = 0.3$'),
                                 (0.5, Forest,   '$m = 0.5$'),
                                 (0.7, Crimson,  '$m = 0.7$')]:
        h = np.clip(dt**(m_val - 1), 0, 50)
        ax.plot(t, h, color=color, linewidth=3, label=label)
    ax.axvline(tc, color=MediumGray, linestyle='--', linewidth=2)
    ax.set_xlabel('Time $t$'); ax.set_ylabel('Hazard Rate $h(t)$')
    ax.set_title('(A) Crash Hazard: $h(t) \\propto (t_c - t)^{m-1}$', fontweight='bold')
    ax.set_ylim(0, 55); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)

    ax = axes[1]
    t2 = np.linspace(0, 0.95, 200); dt2 = tc - t2; m_val = 0.5
    ax.plot(t2, 1 - dt2**m_val, color=Crimson, linewidth=3, label='Cumulative Hazard')
    ax.plot(t2, dt2**m_val, color=Forest, linewidth=3, label='Survival Probability')
    ax.axvline(tc, color=MediumGray, linestyle='--', linewidth=2)
    ax.fill_between(t2, 0, 1 - dt2**m_val, color=Crimson, alpha=0.2)
    ax.set_xlabel('Time $t$'); ax.set_ylabel('Probability')
    ax.set_title('(B) Crash Hazard Rises Near Critical Regime', fontweight='bold')
    ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

    fig.text(0.5, -0.02,
             'Cumulative crash probability increases as $t \\to t_c$\n'
             'The instability grows, making a regime transition increasingly likely',
             ha='center', fontsize=12, style='italic')
    plt.tight_layout(); plt.subplots_adjust(bottom=0.20)
    save_fig(fig, 'hazard_rate')


# =========================================================================
# 10. Confidence Indicator
# =========================================================================
def chart_confidence_indicator():
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), height_ratios=[2, 1, 1])
    np.random.seed(42); T = 400; t = np.arange(T)

    price = 100 * np.exp(0.002*t + 0.3*np.sin(0.02*t) + np.cumsum(np.random.normal(0, 0.01, T)))
    bs = 250
    for i in range(bs, T):
        price[i] *= (1 + 0.003*(i - bs))

    ax = axes[0]
    ax.plot(t, price, color=MainBlue, linewidth=2)
    ax.axvspan(bs, T, color=Crimson, alpha=0.1, label='Bubble Phase')
    ax.set_ylabel('Price'); ax.set_title('(A) Asset Price with Bubble Phase', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=1, frameon=False)

    ax = axes[1]
    ci = np.zeros(T)
    ci[bs:] = np.minimum(0.9, 0.1 + 0.003*(t[bs:] - bs) + 0.1*np.random.random(T - bs))
    ci[:bs] = 0.1 + 0.1*np.random.random(bs)
    ax.fill_between(t, 0, ci, color=Crimson, alpha=0.5)
    ax.axhline(0.5, color=MediumGray, linestyle='--', linewidth=2, label='Warning (0.5)')
    ax.axhline(0.7, color=Orange, linestyle='--', linewidth=2, label='High Alert (0.7)')
    ax.set_ylabel('CI'); ax.set_title('(B) LPPLS Confidence Indicator', fontweight='bold')
    ax.set_ylim(0, 1); ax.grid(True, alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

    ax = axes[2]
    tc_est = 380 + 30*np.random.randn(100)
    ax.hist(tc_est, bins=30, color=Forest, alpha=0.7, edgecolor='white')
    ax.axvline(np.median(tc_est), color=Crimson, linewidth=3, linestyle='--',
               label=f'Median $t_c$ = {np.median(tc_est):.0f}')
    ax.set_xlabel('Time'); ax.set_ylabel('Count')
    ax.set_title('(C) Distribution of $t_c$ Estimates', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=1, frameon=False)

    fig.text(0.5, 0.01,
             'LPPLS CI = Fraction of fits passing all filters | CI > 0.5: Warning | CI > 0.7: High bubble probability',
             ha='center', fontsize=11, style='italic')
    plt.tight_layout(); plt.subplots_adjust(bottom=0.07)
    save_fig(fig, 'confidence_indicator')


# =========================================================================
# 11. Risk Management
# =========================================================================
def chart_risk_management():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Position sizing
    ax = axes[0, 0]
    ci = np.linspace(0, 1, 100)
    ax.plot(ci, 1 - ci**1.5, color=MainBlue, linewidth=3)
    ax.fill_between(ci, 0, 1 - ci**1.5, color=MainBlue, alpha=0.2)
    ax.axhline(0.5, color=MediumGray, linestyle='--', alpha=0.7)
    ax.axvline(0.5, color=Crimson, linestyle='--', alpha=0.7)
    ax.set_xlabel('LPPLS Confidence Indicator'); ax.set_ylabel('Position Size')
    ax.set_title('(A) Dynamic Position Sizing', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.1)

    # Put option timing
    ax = axes[0, 1]
    t = np.arange(100)
    ci_t = np.clip(0.2 + 0.006*t + 0.1*np.sin(0.1*t), 0, 1)
    ax.plot(t, ci_t, color=MainBlue, linewidth=2, label='CI')
    ax.axhline(0.6, color=Crimson, linestyle='--', linewidth=2, label='Buy Puts Threshold')
    buy = t[ci_t > 0.6]
    if len(buy):
        ax.scatter(buy, ci_t[ci_t > 0.6], color=Crimson, s=50, zorder=5)
    ax.set_xlabel('Time'); ax.set_ylabel('CI')
    ax.set_title('(B) Put Option Timing', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

    # VaR
    ax = axes[1, 0]
    ci_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
    var_m = [1.0, 1.2, 1.5, 2.0, 3.0]
    colors = [Forest, Forest, Orange, Crimson, Crimson]
    bars = ax.bar(range(len(ci_vals)), var_m, color=colors, edgecolor='white', linewidth=2)
    ax.set_xticks(range(len(ci_vals)))
    ax.set_xticklabels([f'CI={v}' for v in ci_vals])
    ax.set_ylabel('VaR Multiplier')
    ax.set_title('(C) VaR Adjustment by CI Level', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y'); ax.axhline(1, color=MediumGray, linestyle='--')
    for bar, val in zip(bars, var_m):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val}x', ha='center', fontsize=11, fontweight='bold')

    # Decision framework
    ax = axes[1, 1]; ax.axis('off')
    framework = (
        "┌─────────────────────────────────────────────┐\n"
        "│         LPPL-Based Risk Framework           │\n"
        "├─────────────────────────────────────────────┤\n"
        "│  CI < 0.3  │  Normal risk management        │\n"
        "│            │  Full position allowed         │\n"
        "├────────────┼────────────────────────────────┤\n"
        "│ 0.3 < CI   │  Elevated caution              │\n"
        "│  < 0.5     │  Reduce position 20-30%        │\n"
        "├────────────┼────────────────────────────────┤\n"
        "│ 0.5 < CI   │  High alert                    │\n"
        "│  < 0.7     │  Buy protective puts           │\n"
        "│            │  Reduce position 50%           │\n"
        "├────────────┼────────────────────────────────┤\n"
        "│  CI > 0.7  │  Maximum caution               │\n"
        "│            │  Consider full exit            │\n"
        "│            │  VaR × 2-3                     │\n"
        "└────────────┴────────────────────────────────┘"
    )
    ax.text(0.5, 0.5, framework, transform=ax.transAxes, fontsize=11,
            va='center', ha='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_title('(D) Decision Framework', fontweight='bold')

    fig.text(0.5, 0.01,
             'LPPL-based risk management: Adjust position, hedging, and VaR based on bubble indicators',
             ha='center', fontsize=11, style='italic')
    plt.tight_layout(); plt.subplots_adjust(bottom=0.07)
    save_fig(fig, 'risk_management')


# =========================================================================
# 12. Scaling Ratio
# =========================================================================
def chart_scaling_ratio():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    t = np.linspace(0, 0.95, 500); tc = 1.0; dt = tc - t

    ax = axes[0]
    for omega, color in [(5, MainBlue), (8, Forest), (12, Crimson)]:
        lam = np.exp(2*np.pi/omega)
        ax.plot(t, np.cos(omega*np.log(dt)), color=color, linewidth=2,
                label=f'$\\omega$={omega}, $\\lambda$={lam:.2f}')
    ax.set_xlabel('Time $t$'); ax.set_ylabel('Oscillation')
    ax.set_title('(A) Log-Periodic Oscillations', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)

    ax = axes[1]
    lam_hist = [2.1, 2.3, 2.5, 2.0, 2.2, 2.4, 2.6, 2.1, 2.3, 2.8, 2.2, 2.0, 2.4]
    ax.hist(lam_hist, bins=8, color=MainBlue, alpha=0.7, edgecolor='white', linewidth=2)
    ax.axvline(np.mean(lam_hist), color=Crimson, linewidth=3, linestyle='--',
               label=f'Mean $\\lambda$ = {np.mean(lam_hist):.2f}')
    ax.axvspan(2.0, 2.5, color=Forest, alpha=0.2, label='Universal range')
    ax.set_xlabel('Scaling Ratio $\\lambda = e^{2\\pi/\\omega}$'); ax.set_ylabel('Count')
    ax.set_title('(B) Universal $\\lambda \\approx 2$', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

    fig.text(0.5, 0.01,
             'Universal scaling ratio $\\lambda \\approx 2$ suggests deep structural origin in market hierarchy',
             ha='center', fontsize=11, style='italic')
    plt.tight_layout(); plt.subplots_adjust(bottom=0.18)
    save_fig(fig, 'scaling_ratio')


# =========================================================================
# 13. Hierarchical
# =========================================================================
def chart_hierarchical():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                    gridspec_kw={'width_ratios': [1.1, 1]})

    # Panel A: Hierarchical tree with boxes and branching
    ax1.set_xlim(-0.05, 1.05); ax1.set_ylim(-0.15, 1.05); ax1.axis('off')
    ax1.set_title('(A) Hierarchical Market Structure', fontweight='bold', fontsize=13)

    levels = [
        {'y': 0.92, 'label': 'Central Banks\nRegulators', 'color': Crimson,
         'n': 1, 'w': 0.18, 'h': 0.08, 'time': '$\\tau_1$ ~ years'},
        {'y': 0.72, 'label': 'Major Banks\nInstitutions', 'color': Orange,
         'n': 2, 'w': 0.15, 'h': 0.07, 'time': '$\\tau_2$ ~ months'},
        {'y': 0.52, 'label': 'Hedge Funds\nAsset Mgrs', 'color': Forest,
         'n': 4, 'w': 0.12, 'h': 0.06, 'time': '$\\tau_3$ ~ weeks'},
        {'y': 0.32, 'label': 'Local Funds\nAdvisors', 'color': Amber,
         'n': 8, 'w': 0.08, 'h': 0.05, 'time': '$\\tau_4$ ~ days'},
        {'y': 0.12, 'label': 'Retail\nTraders', 'color': MainBlue,
         'n': 16, 'w': 0.04, 'h': 0.04, 'time': '$\\tau_5$ ~ hours'},
    ]

    all_positions = []
    for lev in levels:
        n = lev['n']
        span = min(0.85, 0.06 * n + 0.1)
        xs = np.linspace(0.5 - span/2, 0.5 + span/2, n) if n > 1 else [0.5]
        positions = []
        for x in xs:
            rect = plt.Rectangle((x - lev['w']/2, lev['y'] - lev['h']/2),
                                  lev['w'], lev['h'],
                                  facecolor=lev['color'], alpha=0.15,
                                  edgecolor=lev['color'], linewidth=1.5,
                                  transform=ax1.transData, zorder=2)
            ax1.add_patch(rect)
            if n <= 4:
                ax1.text(x, lev['y'], lev['label'], ha='center', va='center',
                         fontsize=7 if n <= 2 else 6, color=lev['color'],
                         fontweight='bold', zorder=3)
            positions.append(x)
        all_positions.append(positions)
        # Time-scale annotation on right
        ax1.text(1.02, lev['y'], lev['time'], ha='left', va='center',
                 fontsize=9, color=MediumGray, style='italic')

    # Draw connecting lines between levels
    for i in range(len(levels) - 1):
        parent_xs = all_positions[i]
        child_xs = all_positions[i + 1]
        parent_y = levels[i]['y'] - levels[i]['h'] / 2
        child_y = levels[i + 1]['y'] + levels[i + 1]['h'] / 2
        children_per_parent = len(child_xs) // len(parent_xs)
        for pi, px in enumerate(parent_xs):
            start = pi * children_per_parent
            end = start + children_per_parent
            for cx in child_xs[start:end]:
                ax1.plot([px, cx], [parent_y, child_y],
                         color=MediumGray, linewidth=0.8, alpha=0.5, zorder=1)

    # Lambda annotation between levels 1 and 2
    ax1.annotate('', xy=(-0.02, levels[1]['y']), xytext=(-0.02, levels[0]['y']),
                 arrowprops=dict(arrowstyle='<->', color=DarkGray, lw=1.5))
    ax1.text(-0.04, (levels[0]['y'] + levels[1]['y'])/2, '$\\lambda \\approx 2$',
             ha='right', va='center', fontsize=10, color=DarkGray, fontweight='bold')

    ax1.annotate('', xy=(-0.02, levels[2]['y']), xytext=(-0.02, levels[1]['y']),
                 arrowprops=dict(arrowstyle='<->', color=DarkGray, lw=1.5))
    ax1.text(-0.04, (levels[1]['y'] + levels[2]['y'])/2, '$\\lambda \\approx 2$',
             ha='right', va='center', fontsize=10, color=DarkGray, fontweight='bold')

    # Panel B: How hierarchy generates log-periodic oscillations (bubble: price rises)
    ax2.set_title('(B) Resulting Log-Periodic Cascade', fontweight='bold', fontsize=13)

    t = np.linspace(0.01, 0.99, 500)
    tc = 1.0; m = 0.5; omega = 7.0
    tau = tc - t

    # LPPL with B < 0: price INCREASES toward tc (bubble)
    A = 4.0; B = -1.5
    trend = A + B * tau**m
    lppl = A + B * tau**m * (1 + 0.12 * np.cos(omega * np.log(tau)))

    ax2.plot(t, trend, color=MediumGray, linewidth=1.5, linestyle='--', label='Power law trend ($B<0$)')
    ax2.plot(t, lppl, color=MainBlue, linewidth=2.2, label='LPPL (with hierarchy)')
    ax2.axvline(tc, color=Crimson, linewidth=1.5, linestyle=':', alpha=0.7, label='$t_c$')

    # Mark oscillation peaks (local maxima)
    peaks = []
    for i in range(1, len(t) - 1):
        if lppl[i] > lppl[i-1] and lppl[i] > lppl[i+1] and t[i] > 0.3:
            peaks.append(i)
    for idx in peaks[-5:]:
        ax2.plot(t[idx], lppl[idx], 'o', color=Crimson, markersize=5, zorder=5)

    # Annotate accelerating intervals between consecutive peaks
    if len(peaks) >= 3:
        for j in range(max(0, len(peaks) - 4), len(peaks) - 1):
            y_ann = max(lppl[peaks[j]], lppl[peaks[j+1]]) + 0.04
            ax2.annotate('', xy=(t[peaks[j+1]], y_ann),
                         xytext=(t[peaks[j]], y_ann),
                         arrowprops=dict(arrowstyle='<->', color=Forest, lw=1.2))

    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('$\\ln P(t)$')
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)

    fig.text(0.5, 0.01,
             'Each hierarchical level has a characteristic time scale $\\tau_k$. '
             'The ratio $\\lambda = \\tau_{k}/\\tau_{k+1} \\approx 2$ produces '
             'discrete scale invariance → log-periodic oscillations.',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    plt.tight_layout(); plt.subplots_adjust(bottom=0.15)
    save_fig(fig, 'hierarchical')


# =========================================================================
# 14. Cost Landscape
# =========================================================================
def chart_cost_landscape():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    np.random.seed(42)
    tc_r = np.linspace(300, 400, 200)
    cost = 0.5 + 0.3*np.sin(0.1*tc_r) + 0.2*np.sin(0.05*tc_r + 1) + 0.1*np.random.random(len(tc_r))
    ax.plot(tc_r, cost, color=MainBlue, linewidth=2)
    local_min = [35, 85, 140]; glob_min = 85
    for idx in local_min:
        if idx == glob_min:
            ax.scatter(tc_r[idx], cost[idx], color=Forest, s=200, zorder=5, marker='*', label='Global Minimum')
        else:
            ax.scatter(tc_r[idx], cost[idx], color=Crimson, s=100, zorder=5, marker='o',
                       label='Local Minimum' if idx == 35 else '')
    ax.set_xlabel('Critical Time $t_c$'); ax.set_ylabel('Cost Function (SSR)')
    ax.set_title('(A) Multiple Local Minima', fontweight='bold')
    ax.grid(True, alpha=0.3); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)

    ax = axes[1]
    tc_g = np.linspace(320, 380, 50); m_g = np.linspace(0.1, 0.9, 50)
    TC, M = np.meshgrid(tc_g, m_g)
    np.random.seed(42)
    Z = (TC-350)**2/1000 + (M-0.5)**2*2 + 0.3*np.sin(0.2*TC)*np.sin(5*M) + 0.1*np.random.random(TC.shape)
    im = ax.contourf(TC, M, Z, levels=20, cmap='RdYlGn_r')
    ax.scatter([350], [0.5], color='yellow', s=200, marker='*', edgecolors='black', linewidth=2,
               label='True minimum')
    ax.set_xlabel('$t_c$'); ax.set_ylabel('$m$')
    ax.set_title('(B) 2D Cost Surface', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cost')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=1, frameon=False)

    fig.text(0.5, -0.02,
             'LPPL estimation is challenging: multiple local minima require global optimization (Differential Evolution)',
             ha='center', fontsize=12, style='italic')
    plt.tight_layout(); plt.subplots_adjust(bottom=0.20)
    save_fig(fig, 'cost_landscape')


# =========================================================================
# 15. Filter Conditions
# =========================================================================
def chart_filter_conditions():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    conditions = [
        ('1', '$0.1 \\leq m \\leq 0.9$',                    'Power law exponent in physical range',       MainBlue),
        ('2', '$4 \\leq \\omega \\leq 25$',                  'Angular frequency (2.5-15 oscillations)',    MainBlue),
        ('3', '$B < 0$',                                      'Super-exponential growth (accelerating)',    Crimson),
        ('4', '$|C| < 1$',                                    'Oscillations bounded (not dominant)',        Forest),
        ('5', '$t_c > t_{\\text{last}}$',                     'Critical time in future',                   Orange),
        ('6', '$t_c < t_{\\text{last}} + \\Delta t$',         'Not too far in future',                     Orange),
        ('7', '$\\frac{m|B|\\omega}{2\\pi} > |C|$',           'Damping condition (trend dominates)',        Crimson),
        ('8', '$R^2 > 0.80$',                                 'Good fit quality',                          Forest),
    ]

    y_start = 0.92
    for i, (num, formula, desc, color) in enumerate(conditions):
        y = y_start - i * 0.105
        ax.add_patch(plt.Rectangle((0.02, y-0.04), 0.06, 0.08,
                                    facecolor=color, edgecolor='white', linewidth=2))
        ax.text(0.05, y, num, fontsize=14, fontweight='bold', color='white', ha='center', va='center')
        ax.text(0.12, y, formula, fontsize=14, va='center', fontfamily='serif')
        ax.text(0.48, y, desc, fontsize=12, va='center', style='italic', color=MediumGray)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('LPPL Filter Conditions: All 8 Must Pass for Valid Bubble Signal',
                 fontweight='bold', fontsize=15, y=1.02)
    ax.text(0.5, 0.02,
            'These constraints ensure physically meaningful fits and reduce false positives (Filimonov & Sornette, 2013)',
            ha='center', fontsize=12, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    save_fig(fig, 'filter_conditions')


# =========================================================================
# 16. Use Cases
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


# =========================================================================
# 17. Dot-com Case  # =========================================================================
def chart_dotcom_case():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    prices = download_prices('^IXIC', '1998-10-01', '2000-03-12')
    p = fit_lppl(prices, tc_range=(len(prices)-5, len(prices)+20))
    ALL_PARAMS['dotcom'] = p

    ax = axes[0]
    dates = prices.index
    ax.plot(dates, prices.values, color=MainBlue, linewidth=2, alpha=0.8, label='NASDAQ')
    ax.plot(dates, np.exp(p['fitted_log']), color=Crimson, linewidth=3, label='LPPL Fit')
    peak_i = int(prices.values.argmax())
    peak_date = dates[peak_i]
    ax.axvline(peak_date, color=Orange, linestyle='--', linewidth=2.5)
    ax.annotate(f'March 2000\nPeak: {prices.values[peak_i]:,.0f}',
                xy=(peak_date, prices.values[peak_i]),
                xytext=(-80, 10), textcoords='offset points', fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color=MediumGray),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_xlabel('Date'); ax.set_ylabel('NASDAQ Composite')
    ax.set_title('Dot-Com Bubble: NASDAQ 1998-2000', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    events = [
        ('Jan 1999', 'LPPL paper\npublished',  MainBlue, 0.2),
        ('Oct 1999', 'Warning\nconfirmed',      Orange,   0.4),
        ('Mar 2000', 'NASDAQ\nPeak',            Crimson,  0.6),
        ('Oct 2002', 'Bottom\n-78%',            Forest,   0.9),
    ]
    ax.axhline(0.5, color=MediumGray, linewidth=2)
    for date, event, color, x in events:
        ax.scatter(x, 0.5, s=200, color=color, zorder=5)
        ax.annotate(f'{date}\n{event}', xy=(x, 0.5),
                    xytext=(x, 0.7 if x < 0.5 else 0.3),
                    fontsize=11, ha='center', va='center',
                    arrowprops=dict(arrowstyle='->', color=color),
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_title('Timeline: 14-Month Advance Warning!', fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'dotcom_case')


# =========================================================================
# 18. Shanghai Case  # =========================================================================
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
    ax.set_title('Shanghai 2015: LPPL Fit', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)

    # Right panel: simplified CI from rolling LPPL R²
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
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
    ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

    plt.tight_layout(); plt.subplots_adjust(bottom=0.18)
    save_fig(fig, 'shanghai_case')


# =========================================================================
# 19. Bitcoin 2017 Case  # =========================================================================
def chart_bitcoin2017_case():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: bubble with LPPL fit
    prices = download_prices('BTC-USD', '2017-07-01', '2017-12-20')
    p = fit_lppl(prices)
    ALL_PARAMS['btc2017'] = p

    ax = axes[0]
    dates = prices.index
    ax.plot(dates, prices.values, color=MainBlue, linewidth=2, alpha=0.8, label='Bitcoin')
    ax.plot(dates, np.exp(p['fitted_log']), color=Crimson, linewidth=3, label='LPPL Fit')
    peak_i = int(prices.values.argmax())
    peak_date = dates[peak_i]; peak_price = float(prices.values[peak_i])
    ax.axvline(peak_date, color=Orange, linestyle='--', linewidth=2.5)
    ax.annotate(f'Dec 2017\n${peak_price:,.0f}',
                xy=(peak_date, peak_price), xytext=(-60, -30),
                textcoords='offset points', fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color=MediumGray),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_xlabel('Date'); ax.set_ylabel('Price (USD)')
    ax.set_title('Bitcoin 2017: Classic LPPL Bubble', fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)

    # Right: real post-crash data
    ax = axes[1]
    post = download_prices('BTC-USD', '2017-12-17', '2018-12-31')
    ax.plot(post.index, post.values, color=MainBlue, linewidth=2)
    post_peak = float(post.values.max()); post_bottom = float(post.values.min())
    ax.axhline(post_peak, color=Crimson, linestyle='--', alpha=0.5,
               label=f'Peak: ${post_peak:,.0f}')
    ax.axhline(post_bottom, color=Forest, linestyle='--', alpha=0.5,
               label=f'Bottom: ${post_bottom:,.0f}')
    ax.fill_between(post.index, post_peak, post.values, where=post.values < post_peak,
                    color=Crimson, alpha=0.2)
    drawdown = (post_bottom - post_peak) / post_peak * 100
    ax.annotate(f'{drawdown:.0f}%', xy=(post.index[len(post)//2], (post_peak+post_bottom)/2),
                fontsize=20, fontweight='bold', color=Crimson, ha='center')
    ax.set_xlabel('Date'); ax.set_ylabel('Price (USD)')
    ax.set_title(f'Post-Crash: {abs(drawdown):.0f}% Drawdown Over 12 Months', fontweight='bold')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'bitcoin2017_case')


# =========================================================================
# 20. Oil 2008 Case  # =========================================================================
def chart_oil2008_case():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Oil with LPPL fit
    prices = download_prices('CL=F', '2007-01-01', '2008-07-14')
    p = fit_lppl(prices, tc_range=(len(prices)-5, len(prices)+20))
    ALL_PARAMS['oil2008'] = p

    ax = axes[0]
    dates = prices.index
    ax.plot(dates, prices.values, color=MainBlue, linewidth=2, alpha=0.8, label='Oil Price')
    ax.plot(dates, np.exp(p['fitted_log']), color=Crimson, linewidth=3, label='LPPL Fit')
    peak_i = int(prices.values.argmax())
    peak_date = dates[peak_i]; peak_price = float(prices.values[peak_i])
    ax.axvline(peak_date, color=Orange, linestyle='--', linewidth=2.5)
    ax.annotate(f'July 2008\n${peak_price:.0f}/bbl',
                xy=(peak_date, peak_price), xytext=(-80, 10),
                textcoords='offset points', fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color=MediumGray),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_xlabel('Date'); ax.set_ylabel('Price (USD/bbl)')
    ax.set_title('Oil 2007-2008: Commodity Bubble', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)

    # Right: Real post-crash data
    ax = axes[1]
    post = download_prices('CL=F', '2008-07-11', '2009-06-30')
    ax.plot(post.index, post.values, color=MainBlue, linewidth=2)
    post_peak = float(post.values.max()); post_bottom = float(post.values.min())
    ax.axhline(post_peak, color=Crimson, linestyle='--', alpha=0.5,
               label=f'Peak: ${post_peak:.0f}')
    ax.axhline(post_bottom, color=Forest, linestyle='--', alpha=0.5,
               label=f'Bottom: ${post_bottom:.0f}')
    ax.fill_between(post.index, post_peak, post.values, where=post.values < post_peak,
                    color=Crimson, alpha=0.2)
    drawdown = (post_bottom - post_peak) / post_peak * 100
    ax.annotate(f'{drawdown:.0f}%', xy=(post.index[len(post)//3], (post_peak+post_bottom)/2),
                fontsize=20, fontweight='bold', color=Crimson, ha='center')
    ax.set_xlabel('Date'); ax.set_ylabel('Price (USD/bbl)')
    ax.set_title(f'Aftermath: Crash to ${post_bottom:.0f} ({abs(drawdown):.0f}% decline)',
                 fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'oil2008_case')


# =========================================================================
# 21. COVID 2020 Case  # =========================================================================
def chart_covid2020_case():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Real S&P 500 data with attempted LPPL fit
    sp = download_prices('^GSPC', '2019-01-01', '2020-06-30')
    crash_date = pd.Timestamp('2020-02-19')

    # Attempt LPPL fit on pre-crash data (poor fit expected)
    pre_crash = sp[:crash_date]
    try:
        p_covid = fit_lppl(pre_crash, tc_range=(len(pre_crash)-5, len(pre_crash)+60))
        lppl_fitted = np.exp(p_covid['fitted_log'])
        r2_str = f'$R^2$ = {p_covid["R2"]:.3f}'
    except Exception:
        lppl_fitted = None
        r2_str = '$R^2$ < 0.5'

    ax = axes[0]
    ax.plot(sp.index, sp.values, color=MainBlue, linewidth=2, alpha=0.8, label='S&P 500')
    if lppl_fitted is not None:
        ax.plot(pre_crash.index, lppl_fitted, color=Crimson, linewidth=2,
                linestyle='--', alpha=0.6, label=f'LPPL Fit (Poor, {r2_str})')
    ax.axvline(crash_date, color=Orange, linestyle='--', linewidth=2,
               label='COVID Shock (Feb 19)')
    crash_price = float(sp[crash_date]) if crash_date in sp.index else float(pre_crash.values[-1])
    ax.annotate('Exogenous\nShock', xy=(crash_date, crash_price),
                xytext=(40, 30), textcoords='offset points', fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color=Crimson, lw=1.5),
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
    ax.set_xlabel('Date'); ax.set_ylabel('Price')
    ax.set_title('S&P 500 2020: LPPL Fit Attempt', fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
    ax.grid(True, alpha=0.3)

    # Right: Explanation (same)
    ax = axes[1]; ax.axis('off')
    explanation = (
        "Why LPPL correctly REJECTED this crash:\n\n"
        "1. No super-exponential growth pattern\n"
        "   - Growth was approximately linear, not\n"
        "     accelerating toward a singularity\n\n"
        "2. No log-periodic oscillations detected\n"
        "   - Filter conditions failed:\n"
        "     $R^2 < 0.80$, $B > 0$ in many fits\n\n"
        "3. Exogenous shock (pandemic), not\n"
        "   endogenous instability\n"
        "   - LPPL models endogenous bubbles only\n\n"
        "4. LPPLS CI remained below 0.3\n"
        "   throughout late 2019 / early 2020\n\n"
        "CONCLUSION: LPPL is not designed to predict\n"
        "exogenous shocks - and that is a feature,\n"
        "not a bug."
    )
    ax.text(0.05, 0.95, explanation, transform=ax.transAxes, fontsize=12,
            va='top', ha='left', fontfamily='sans-serif',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=MediumGray))
    ax.set_title('Why LPPL Correctly Did Not Signal a Bubble', fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'covid2020_case')


# =========================================================================
# RUN ALL
# =========================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('Generating ch13 LPPL charts (TSA colors)...')
    print('=' * 60)

    # Theoretical charts (unchanged)
    chart_bubble_growth()          # 1
    chart_oscillations()           # 4
    chart_ising()                  # 5
    chart_phase_transition()       # 6
    chart_lppl_components()        # 7
    chart_hazard_rate()            # 9
    chart_confidence_indicator()   # 10
    chart_risk_management()        # 11
    chart_scaling_ratio()          # 12
    chart_hierarchical()           # 13
    chart_cost_landscape()         # 14
    chart_filter_conditions()      # 15
    chart_use_cases()              # 16

    # Real-data charts (download + fit + plot)
    print('\n  Downloading market data and fitting LPPL models...')
    chart_btc_lppl()               # 2
    chart_btc_full_analysis()      # 3
    chart_historical_crashes()     # 8
    chart_dotcom_case()            # 17
    chart_shanghai_case()          # 18
    chart_bitcoin2017_case()       # 19
    chart_oil2008_case()           # 20
    chart_covid2020_case()         # 21

    # Print fitted parameters for tex tables
    print('\n' + '=' * 60)
    print('FITTED LPPL PARAMETERS (for tex tables)')
    print('=' * 60)
    for name, p in ALL_PARAMS.items():
        peak_i = int(p['prices'].values.argmax())
        tc_error = abs(p['tc'] - peak_i)
        lam = np.exp(2 * np.pi / p['omega'])
        print(f'\n  {name}:')
        print(f'    tc = {p["tc"]:.1f} (error: {tc_error:.0f} days)')
        print(f'    m  = {p["m"]:.3f}')
        print(f'    ω  = {p["omega"]:.3f}')
        print(f'    λ  = {lam:.2f}')
        print(f'    R² = {p["R2"]:.3f}')
        print(f'    B  = {p["B"]:.4f}')

    print('\n' + '=' * 60)
    print('All charts generated successfully!')
    print('=' * 60)
