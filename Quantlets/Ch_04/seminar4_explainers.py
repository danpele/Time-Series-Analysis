"""
seminar4_explainers.py -- explanatory (primer) charts for Seminar 4 (TSA): seasonality and forecasting
=====================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 4, which takes
place BEFORE Lecture 4. All charts use SIMULATED data only (fixed seeds): they illustrate seasonal differences,
the ACF of seasonal models, the role of the seasonal MA parameter, SARIMA forecasts, Fourier terms, rolling-origin
cross-validation, the Diebold-Mariano test and forecast combination. They contain no exercise answers.

The figures are drawn at the size of their box on the slide (full width: 5.6 x 1.5 in; one column: 2.8 x 2.0 in),
so that 1 pt in the figure is about 1 pt on the slide (text >= 7 pt).

Output: charts/ch4_sem_primer_*.pdf and .png (transparent background, legend below the plot).
Run:    python3 Quantlets/Ch_04/seminar4_explainers.py
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import tsa_style as st   # noqa: E402

warnings.filterwarnings('ignore')
st.apply()
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7.5,
                     'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'lines.linewidth': 1.0,
                     'axes.titlelocation': 'left'})
B, R, G, A, P = st.MainBlue, st.IDAred, st.Forest, st.Amber, st.Purple
FULL = (5.6, 1.5)
HALF = (2.8, 2.0)


def save(fig, name):
    st.check_no_grey(fig)
    st.save_fig(f'ch4_sem_primer_{name}')


def ma_acf(theta, nlags):
    """Theoretical ACF of an MA process w_t = sum_j theta_j eps_{t-j} (theta_0 = 1 included)."""
    th = np.asarray(theta, float)
    g = [np.sum(th[:len(th) - k] * th[k:]) if k < len(th) else 0.0 for k in range(nlags + 1)]
    return np.array(g[1:]) / g[0]


def airline_simulate(theta, Theta, s=12, n=240, seed=1, base=None, sigma=1.0):
    """y_t with (1-L)(1-L^s) y_t = (1 + theta L)(1 + Theta L^s) eps_t; the first s+1 values from `base`."""
    rng = np.random.default_rng(seed)
    e = sigma * rng.standard_normal(n + s + 1)
    y = np.zeros(n + s + 1)
    y[:s + 1] = base if base is not None else 0
    for t in range(s + 1, n + s + 1):
        w = e[t] + theta * e[t - 1] + Theta * e[t - s] + theta * Theta * e[t - s - 1]
        y[t] = y[t - 1] + y[t - s] - y[t - s - 1] + w
    return y[s + 1:]


# =============================================================================
# 1. seasonal differences of a quarterly series
# =============================================================================
def fig_differences(seed=4):
    rng = np.random.default_rng(seed)
    n = 40
    t = np.arange(n)
    pattern = np.array([-6.0, 2.0, 5.0, -1.0])
    y = 100 + 0.8 * t + pattern[t % 4] + np.cumsum(0.15 * rng.standard_normal(n)) + 0.8 * rng.standard_normal(n)
    d4 = y[4:] - y[:-4]
    dd4 = d4[1:] - d4[:-1]
    fig, ax = plt.subplots(1, 3, figsize=FULL)
    ax[0].plot(t + 1, y, color=B, marker='o', ms=1.8, label='Series $y_t$')
    tf = np.arange(n, n + 4)
    ax[0].plot(tf + 1, y[-4:], color=R, ls='--', marker='o', ms=1.8, label='Seasonal naive forecast')
    ax[0].set_title('$y_t$: trend and season')
    ax[0].set_xlabel('Quarter $t$')
    ax[1].plot(t[4:] + 1, d4, color=G, marker='o', ms=1.8, label=r'$\Delta_4 y_t = y_t - y_{t-4}$')
    ax[1].axhline(4 * 0.8, color=A, ls='--', lw=0.9, label='$4 \\times$ trend slope')
    ax[1].set_title(r'$\Delta_4 y_t$: season removed')
    ax[1].set_xlabel('Quarter $t$')
    ax[2].plot(t[5:] + 1, dd4, color=P, marker='o', ms=1.8, label=r'$\Delta\Delta_4 y_t$')
    ax[2].axhline(0, color=st.DarkText, ls=':', lw=0.7)
    ax[2].set_title(r'$\Delta\Delta_4 y_t$: trend removed too')
    ax[2].set_xlabel('Quarter $t$')
    fig.tight_layout(w_pad=0.6)
    st.fig_legend_bottom(fig, ncol=5)
    save(fig, 'differences')
    return dict(mean_d4=float(d4.mean()), mean_dd4=float(dd4.mean()))


# =============================================================================
# 2. theoretical ACF of the airline model and of a seasonal AR(1)
# =============================================================================
def fig_acf(theta=-0.3, Theta=-0.7, Phi=0.5, s=12, nl=40):
    th = np.zeros(s + 2)
    th[0], th[1], th[s], th[s + 1] = 1, theta, Theta, theta * Theta
    acf_air = ma_acf(th, nl)
    lags = np.arange(1, nl + 1)
    acf_sar = np.array([Phi ** (k // s) if k % s == 0 else 0.0 for k in lags])
    pacf_sar = np.array([Phi if k == s else 0.0 for k in lags])
    fig, ax = plt.subplots(1, 3, figsize=FULL, sharey=True)
    for a, v, c, ttl in [(ax[0], acf_air, B, f'Airline, $\\theta = {theta}$, $\\Theta = {Theta}$: ACF'),
                         (ax[1], acf_sar, G, f'Seasonal AR(1), $\\Phi = {Phi}$: ACF'),
                         (ax[2], pacf_sar, R, f'Seasonal AR(1), $\\Phi = {Phi}$: PACF')]:
        a.vlines(lags, 0, v, color=c, lw=1.6)
        a.axhline(0, color=st.DarkText, lw=0.6)
        for k in (s, 2 * s, 3 * s):
            a.axvline(k, color=A, ls=':', lw=0.7)
        a.set_title(ttl.replace('-', '{-}'))
        a.set_xlabel('Lag $k$')
        a.set_xticks([1, 12, 24, 36])
    ax[0].set_ylabel('Autocorrelation')
    ax[0].set_ylim(-0.6, 0.65)
    fig.tight_layout(w_pad=0.6)
    save(fig, 'acf')
    return dict(rho1=float(acf_air[0]), rho11=float(acf_air[10]), rho12=float(acf_air[11]), rho13=float(acf_air[12]))


# =============================================================================
# 3. the role of Theta: a fixed or an evolving seasonal pattern
# =============================================================================
def fig_theta(theta=-0.4, s=12, years=12, seed=7):
    base = 10 * np.sin(2 * np.pi * np.arange(s + 1) / s)
    fig, ax = plt.subplots(1, 2, figsize=FULL, sharey=True)
    out = {}
    for a, Th in zip(ax, (-0.95, 0.0)):
        y = airline_simulate(theta, Th, s=s, n=years * s, seed=seed, base=base, sigma=2.0)
        for yr, c in zip((1, 6, 12), (B, A, R)):
            seg = y[(yr - 1) * s: yr * s]
            a.plot(np.arange(1, s + 1), seg - seg.mean(), color=c, marker='o', ms=2, label=f'Year {yr}')
        a.axhline(0, color=st.DarkText, lw=0.5, ls=':')
        a.set_title((f'$\\Theta = {{-}}{abs(Th)}$: a stable pattern' if Th < -0.5 else '$\\Theta = 0$: a pattern that changes'))
        a.set_xlabel('Month')
        a.set_xticks([1, 3, 6, 9, 12])
        seg1, seg12 = y[:s] - y[:s].mean(), y[-s:] - y[-s:].mean()
        out[Th] = float(np.corrcoef(seg1, seg12)[0, 1])
    ax[0].set_ylabel('Deviation from the year mean')
    fig.tight_layout(w_pad=0.8)
    st.fig_legend_bottom(fig, ncol=3)
    save(fig, 'theta')
    return {f'corr_year1_year12_Theta={k}': v for k, v in out.items()}


# =============================================================================
# 4. airline forecasts with 95% intervals against the seasonal naive forecast
# =============================================================================
def fig_forecast(seed=11, s=12, n=96, H=24):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    t = np.arange(n + s + 1)
    base = 100 + 0.3 * t[:s + 1] + 8 * np.sin(2 * np.pi * t[:s + 1] / s)
    y = airline_simulate(-0.4, -0.6, s=s, n=n, seed=seed, base=base, sigma=1.2)
    res = SARIMAX(y, order=(0, 1, 1), seasonal_order=(0, 1, 1, s)).fit(disp=False)
    fc = res.get_forecast(H)
    m, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
    sn = np.array([y[n - s + (h - 1) % s] for h in range(1, H + 1)])
    fig, ax = plt.subplots(figsize=FULL)
    tt = np.arange(1, n + 1)
    tf = np.arange(n + 1, n + H + 1)
    ax.plot(tt, y, color=B, label='Simulated monthly series')
    ax.fill_between(tf, ci[:, 0], ci[:, 1], color=BAND, lw=0, label='95% interval')
    ax.plot(tf, m, color=R, label='Airline forecast')
    ax.plot(tf, sn, color=G, ls='--', label='Seasonal naive forecast')
    ax.axvline(n + 0.5, color=A, ls=':', lw=0.9)
    ax.set_xlabel('Month $t$')
    ax.set_ylabel('$y_t$')
    ax.set_xlim(40, n + H + 1)
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=4)
    save(fig, 'forecast')
    width = ci[:, 1] - ci[:, 0]
    return dict(theta=float(res.params[0]), Theta=float(res.params[1]), width_h1=float(width[0]),
                width_h12=float(width[11]), width_h24=float(width[23]))


# =============================================================================
# 5. Fourier terms: harmonics and the approximation of an annual pattern
# =============================================================================
def fig_fourier(m=365.25):
    d = np.arange(1, 366)
    shape = 0.9 * np.cos(2 * np.pi * (d - 15) / m) + 0.35 * np.cos(4 * np.pi * (d - 200) / m)
    shape = shape + np.where((d > 355) | (d < 8), -0.6, 0.0)          # a holiday dip at the turn of the year
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    for k, c in zip((1, 2, 3), (B, R, G)):
        ax[0].plot(d, np.sin(2 * np.pi * k * d / m), color=c, label=f'$\\sin(2\\pi k t/m)$, $k = {k}$')
    ax[0].set_title('Harmonics, $m = 365.25$ days')
    ax[0].set_xlabel('Day of the year $t$')
    out = {}
    ax[1].plot(d, shape, color=st.DarkText, lw=1.3, label='Annual pattern')
    for K, c in zip((1, 2, 4), (A, B, R)):
        X = np.column_stack([np.ones_like(d, dtype=float)] + [f(2 * np.pi * k * d / m) for k in range(1, K + 1)
                                                               for f in (np.sin, np.cos)])
        b = np.linalg.lstsq(X, shape, rcond=None)[0]
        fit = X @ b
        ax[1].plot(d, fit, color=c, ls='--', label=f'$K = {K}$ ({2 * K} coef.)')
        out[K] = float(1 - np.var(shape - fit) / np.var(shape))
    ax[1].set_title('Fourier approximations with $K$ pairs')
    ax[1].set_xlabel('Day of the year $t$')
    fig.tight_layout(w_pad=0.8)
    st.fig_legend_bottom(fig, ncol=4)
    save(fig, 'fourier')
    return {f'R2_K{k}': v for k, v in out.items()}


# =============================================================================
# 6. rolling-origin (time-series) cross-validation
# =============================================================================
def fig_cv(n_orig=6, first=30, step=6, h=6, total=72):
    fig, ax = plt.subplots(figsize=HALF)
    for i in range(n_orig):
        T_i = first + i * step
        yy = n_orig - i
        ax.plot([1, T_i], [yy, yy], color=B, lw=4, solid_capstyle='butt', label='Estimation sample' if i == 0 else '_')
        ax.plot([T_i + 1, T_i + h], [yy, yy], color=R, lw=4, solid_capstyle='butt', label=f'Forecasts, $h = 1, \\dots, {h}$' if i == 0 else '_')
        ax.plot([T_i + h + 1, total], [yy, yy], color=BAND, lw=4, solid_capstyle='butt', label='Not used' if i == 0 else '_')
        ax.text(0, yy, f'$T_{i + 1}$ ', ha='right', va='center', fontsize=7.5)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(-6, total + 1)
    ax.set_xlabel('Time $t$')
    ax.set_title('Origins $T_1 < T_2 < \\dots$')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=1)
    save(fig, 'cv')
    return dict(n_origins=n_orig, n_errors=n_orig * h)


BAND = '#C5D2E8'


# =============================================================================
# 7. the Diebold-Mariano test: loss differential and the t reference distribution
# =============================================================================
def fig_dm(n=30, seed=5):
    rng = np.random.default_rng(seed)
    e1 = rng.standard_normal(n) * 1.0
    e2 = 0.6 * e1 + rng.standard_normal(n) * 0.95
    d = e1 ** 2 - e2 ** 2
    dbar = d.mean()
    s2 = np.mean((d - dbar) ** 2)
    dm = dbar / np.sqrt(s2 / n)
    hln = np.sqrt((n - 1) / n) * dm
    tc = stats.t.ppf(0.975, n - 1)
    p = 2 * stats.t.sf(abs(hln), n - 1)
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    ax[0].bar(np.arange(1, n + 1), d, color=np.where(d < 0, B, R), width=0.8)
    ax[0].axhline(dbar, color=A, lw=1.2, ls='--', label=f'$\\bar d = {dbar:.2f}$'.replace('-', '{-}'))
    ax[0].axhline(0, color=st.DarkText, lw=0.6)
    ax[0].set_title('$d_t = e_{1t}^2 - e_{2t}^2$ (blue: forecast 1 better)')
    ax[0].set_xlabel('Forecast $t$')
    x = np.linspace(-4.2, 4.2, 400)
    f = stats.t.pdf(x, n - 1)
    ax[1].plot(x, f, color=B, label=f'Student $t({n - 1})$ density')
    for sgn in (-1, 1):
        xx = x[sgn * x >= tc]
        ax[1].fill_between(xx, 0, stats.t.pdf(xx, n - 1), color=R, alpha=0.35, lw=0,
                           label='Rejection region, 5%' if sgn == 1 else '_')
    ax[1].axvline(hln, color=G, lw=1.4, label=f'HLN $= {hln:.2f}$'.replace('-', '{-}'))
    ax[1].set_title(f'Critical values $\\pm${tc:.3f}')
    ax[1].set_xlabel('Statistic')
    fig.tight_layout(w_pad=0.8)
    st.fig_legend_bottom(fig, ncol=4)
    save(fig, 'dm')
    return dict(dbar=float(dbar), dm=float(dm), hln=float(hln), tcrit=float(tc), p=float(p),
                mse1=float(np.mean(e1 ** 2)), mse2=float(np.mean(e2 ** 2)))


# =============================================================================
# 8. forecast combination: MSE(w) for three error correlations
# =============================================================================
def fig_combination(s1=1.0, s2=1.5):
    w = np.linspace(0, 1, 201)
    fig, ax = plt.subplots(figsize=HALF)
    out = {}
    for rho, c in zip((-0.5, 0.3, 0.6), (B, G, R)):
        mse = w ** 2 * s1 ** 2 + (1 - w) ** 2 * s2 ** 2 + 2 * w * (1 - w) * rho * s1 * s2
        ws = (s2 ** 2 - rho * s1 * s2) / (s1 ** 2 + s2 ** 2 - 2 * rho * s1 * s2)
        ws_c = min(max(ws, 0), 1)
        ms = ws_c ** 2 * s1 ** 2 + (1 - ws_c) ** 2 * s2 ** 2 + 2 * ws_c * (1 - ws_c) * rho * s1 * s2
        ax.plot(w, mse, color=c, label=f'$\\rho = {rho}$'.replace('-', '{-}'))
        ax.plot(ws_c, ms, 'o', color=c, ms=3.5)
        out[rho] = (float(ws), float(ms))
    ax.axhline(s1 ** 2, color=A, ls=':', lw=0.9, label='MSE of forecast 1')
    ax.set_xlabel('Weight $w$ of forecast 1')
    ax.set_ylabel('MSE$(w)$')
    ax.set_title(f'$\\sigma_1 = {s1:g}$, $\\sigma_2 = {s2:g}$; dots: $w^*$')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=2)
    save(fig, 'combination')
    return {f'rho={k}': {'w_star': v[0], 'mse_star': v[1]} for k, v in out.items()}


if __name__ == '__main__':
    for f in (fig_differences, fig_acf, fig_theta, fig_forecast, fig_fourier, fig_cv, fig_dm, fig_combination):
        print(f.__name__, f())
