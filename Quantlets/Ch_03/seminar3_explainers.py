"""
seminar3_explainers.py -- explanatory (primer) charts for Seminar 3 (TSA): unit roots and ARIMA models
=====================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 3, which takes
place BEFORE Lecture 3. All charts use SIMULATED data only (fixed seeds): trend-stationary and difference-
stationary paths, the orders of integration, the Dickey--Fuller distributions, a spurious regression, the KPSS
partial sums, a break in the mean and the forecast intervals of stationary and integrated models. They contain
no exercise answers.

Output: charts/ch3_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).
The charts are drawn at their size on the slide, so the text is 7.5-8.5 pt on the slide.

Run:  python3 Quantlets/Ch_03/seminar3_explainers.py

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
from statsmodels.tsa.stattools import adfuller, kpss

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import tsa_style as st   # noqa: E402

warnings.filterwarnings('ignore')
st.apply()
plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'lines.linewidth': 1.0})
HALF = (2.9, 2.3)
BLUE, RED, GREEN, AMBER, PURPLE, NAVY = st.MainBlue, st.IDAred, st.Forest, st.Amber, st.Purple, st.DarkText
BAND = '#C5D2E8'
COLS = [BLUE, RED, GREEN, AMBER, PURPLE]


def save(fig, name):
    st.check_no_grey(fig)
    st.save_fig(name)


def mm(s):
    return s.replace('-', '−')


# =============================================================================
# 1. trend-stationary against difference-stationary (random walk with drift)
# =============================================================================
def fig_trend_vs_rw(seed=10, T=200, a=0.0, b=0.1, n=5):
    rng = np.random.default_rng(seed)
    t = np.arange(1, T + 1)
    line = a + b * t
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.6))
    phi = 0.6
    sd_u = 1 / np.sqrt(1 - phi ** 2)
    for i in range(n):
        e = rng.standard_normal(T + 50)
        u = np.zeros(T + 50)
        for s in range(1, T + 50):
            u[s] = phi * u[s - 1] + e[s]
        axes[0].plot(t, line + u[50:], color=COLS[i], lw=0.6)
        axes[1].plot(t, np.cumsum(b + rng.standard_normal(T)), color=COLS[i], lw=0.6)
    axes[0].fill_between(t, line - 1.96 * sd_u, line + 1.96 * sd_u, color=BAND, alpha=0.8, lw=0, label='95% band around the line')
    axes[1].fill_between(t, line - 1.96 * np.sqrt(t), line + 1.96 * np.sqrt(t), color=BAND, alpha=0.8, lw=0)
    for ax, title in zip(axes, ('Trend-stationary: $y_t = 0.1\\,t + u_t$', 'Random walk with drift: $\\Delta y_t = 0.1 + \\varepsilon_t$')):
        ax.plot(t, line, color=NAVY, ls='--', lw=1.0, label='Line $0.1\\,t$' if ax is axes[0] else '_')
        ax.set_title(title, loc='left')
        ax.set_xlabel('$t$')
    axes[0].set_ylabel('$y_t$')
    fig.tight_layout(w_pad=0.8)
    st.fig_legend_bottom(fig, ncol=2)
    save(fig, 'ch3_sem_primer_trend_vs_rw')
    return dict(band_ts=1.96 * sd_u, band_rw200=1.96 * np.sqrt(T))


# =============================================================================
# 2. orders of integration: I(0), I(1), I(2)
# =============================================================================
def fig_integration(seed=4, T=200):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T)
    i0 = np.zeros(T)
    for s in range(1, T):
        i0[s] = 0.5 * i0[s - 1] + e[s]
    i1 = np.cumsum(e)
    i2 = np.cumsum(i1)
    t = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.55))
    for ax, x, col, title in ((axes[0], i0, BLUE, '$I(0)$: AR(1), $\\phi = 0.5$'), (axes[1], i1, RED, '$I(1)$: $\\Delta y_t$ stationary'),
                              (axes[2], i2, GREEN, '$I(2)$: $\\Delta^2 y_t$ stationary')):
        ax.plot(t, x, color=col, lw=0.8)
        ax.set_title(title, loc='left')
        ax.set_xlabel('$t$')
    fig.tight_layout(w_pad=0.6)
    save(fig, 'ch3_sem_primer_integration')


# =============================================================================
# 3. Dickey--Fuller distributions of tau, by Monte Carlo
# =============================================================================
def _tau(y, det):
    """tau of the DF regression Delta y_t = det + gamma y_{t-1} + e_t, for many series at once (rows)."""
    R, T = y.shape
    dy = np.diff(y, axis=1)
    ylag = y[:, :-1]
    n = T - 1
    cols = [ylag]
    if det in ('c', 'ct'):
        cols.append(np.ones((R, n)))
    if det == 'ct':
        cols.append(np.broadcast_to(np.arange(1, n + 1, dtype=float), (R, n)))
    X = np.stack(cols, axis=2)
    XtX = np.einsum('rti,rtj->rij', X, X)
    Xty = np.einsum('rti,rt->ri', X, dy)
    beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]
    res = dy - np.einsum('rti,ri->rt', X, beta)
    s2 = np.sum(res ** 2, axis=1) / (n - X.shape[2])
    se = np.sqrt(s2 * np.linalg.inv(XtX)[:, 0, 0])
    return beta[:, 0] / se


def fig_df_dist(seed=1, R=20000, T=250):
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.standard_normal((R, T)), axis=1)
    x = np.linspace(-5.5, 3.5, 500)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, stats.norm.pdf(x), color=NAVY, lw=1.0, ls='--', label='$N(0, 1)$, 5%: $-1.645$')
    out = {}
    for det, col, lab in (('n', BLUE, 'no constant'), ('c', RED, 'constant'), ('ct', GREEN, 'constant and trend')):
        tau = _tau(y, det)
        kde = stats.gaussian_kde(tau)
        cv = np.quantile(tau, 0.05)
        out[det] = cv
        ax.plot(x, kde(x), color=col, lw=1.4, label=mm(f'{lab}, 5%: {cv:.2f}'))
        ax.axvline(cv, color=col, lw=0.7, ls=':')
    ax.axvline(-1.645, color=NAVY, lw=0.7, ls=':')
    ax.set_xlabel('$\\tau$ under $H_0$ (unit root)')
    ax.set_ylabel('Density')
    ax.set_title(f'Simulated, $T$ = {T}', loc='left')
    st.legend_outside_bottom(ax, ncol=1)
    fig.tight_layout()
    save(fig, 'ch3_sem_primer_df_dist')
    return out


# =============================================================================
# 4. a spurious regression of two independent random walks
# =============================================================================
def fig_spurious(seed=7, T=200, R=2000):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(T))
    y = np.cumsum(rng.standard_normal(T))
    X = np.c_[np.ones(T), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    s2 = e @ e / (T - 2)
    t_one = b[1] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    r2_one = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    dw_one = np.sum(np.diff(e) ** 2) / (e @ e)
    xs = np.cumsum(rng.standard_normal((R, T)), axis=1)
    ys = np.cumsum(rng.standard_normal((R, T)), axis=1)
    xc = xs - xs.mean(axis=1, keepdims=True)
    yc = ys - ys.mean(axis=1, keepdims=True)
    bb = np.sum(xc * yc, axis=1) / np.sum(xc ** 2, axis=1)
    ee = yc - bb[:, None] * xc
    se = np.sqrt(np.sum(ee ** 2, axis=1) / (T - 2) / np.sum(xc ** 2, axis=1))
    tt = bb / se
    share = np.mean(np.abs(tt) > 1.96)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.65), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    ax.plot(np.arange(1, T + 1), x, color=BLUE, lw=0.9, label='$x_t$')
    ax.plot(np.arange(1, T + 1), y, color=RED, lw=0.9, label='$y_t$, independent of $x_t$')
    ax.set_title(mm(f'One pair: $t$ = {t_one:.1f}, $R^2$ = {r2_one:.2f}, DW = {dw_one:.2f}'), loc='left')
    ax.set_xlabel('$t$')
    ax = axes[1]
    ax.hist(np.clip(tt, -40, 40), bins=60, density=True, color=BAND, edgecolor=BLUE, lw=0.3, label=f'$t$ of {R:,} pairs')
    ax.axvspan(-1.96, 1.96, color=GREEN, alpha=0.35, lw=0, label='$|t| \\leq 1.96$: 95% of the cases if $t \\sim N(0, 1)$')
    ax.set_title(f'$|t| > 1.96$ in {100 * share:.0f}% of the pairs', loc='left')
    ax.set_xlabel('$t$-ratio of the slope')
    fig.tight_layout(w_pad=0.8)
    st.fig_legend_bottom(fig, ncol=2)
    save(fig, 'ch3_sem_primer_spurious')
    return dict(t=t_one, r2=r2_one, dw=dw_one, share=share)


# =============================================================================
# 5. KPSS: partial sums of the deviations
# =============================================================================
def fig_kpss(seed=12, T=200):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T)
    st_ = np.zeros(T)
    for s in range(1, T):
        st_[s] = 0.3 * st_[s - 1] + e[s]
    rw = np.cumsum(rng.standard_normal(T))
    t = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.6))
    out = {}
    for ax, x, col, name in ((axes[0], st_, BLUE, 'Stationary AR(1)'), (axes[1], rw, RED, 'Random walk')):
        S = np.cumsum(x - x.mean())
        eta = kpss(x, regression='c', nlags='auto')[0]
        out[name] = eta
        ax.plot(t, x - x.mean(), color=col, lw=0.6, alpha=0.6, label='Deviation $y_t - \\bar y$')
        ax.plot(t, S / np.sqrt(T), color=col, lw=1.6, label='Partial sum $S_t/\\sqrt{T}$')
        ax.axhline(0, color=NAVY, lw=0.5)
        ax.set_title(f'{name}: KPSS = {eta:.2f}', loc='left')
        ax.set_xlabel('$t$')
    fig.tight_layout(w_pad=0.8)
    st.fig_legend_bottom(fig, ncol=2)
    save(fig, 'ch3_sem_primer_kpss')
    return out


# =============================================================================
# 6. a break in the mean can look like a unit root
# =============================================================================
def fig_break(seed=3, T=200):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T)
    u = np.zeros(T)
    for s in range(1, T):
        u[s] = 0.5 * u[s - 1] + e[s]
    mu = np.where(np.arange(T) < T // 2, 0.0, 5.0)
    y = mu + u
    t = np.arange(1, T + 1)
    adf = adfuller(y, regression='c', autolag='AIC')
    from statsmodels.tsa.stattools import zivot_andrews
    za = zivot_andrews(y, regression='c', autolag='AIC')
    fig, ax = plt.subplots(figsize=(5.6, 1.5))
    ax.plot(t, y, color=BLUE, lw=0.8, label='$y_t = \\mu_t + u_t$, $u_t$ a stationary AR(1)')
    ax.plot(t, mu, color=RED, lw=1.4, ls='--', label='Mean $\\mu_t$: 0, then 5 from $t$ = 101')
    ax.set_title(mm(f'ADF (constant): $\\tau$ = {adf[0]:.2f}, p = {adf[1]:.2f};  Zivot\u2013Andrews: {za[0]:.2f}, break at $t$ = {za[4] + 1}'),
                 loc='left')
    ax.set_xlabel('$t$')
    fig.tight_layout()
    st.fig_legend_bottom(fig, ncol=2)
    save(fig, 'ch3_sem_primer_break')
    return dict(adf=adf[0], adfp=adf[1], za=za[0], zap=za[1], zab=za[4])


# =============================================================================
# 7. forecast intervals: stationary AR(1) against random walk with drift
# =============================================================================
def fig_fan(H=24, sigma=1.0, phi=0.5, c=0.2):
    h = np.arange(0, H + 1)
    psi2_ar = np.r_[0, np.cumsum(phi ** (2 * np.arange(H)))]
    w_ar = 1.96 * sigma * np.sqrt(psi2_ar)
    w_rw = 1.96 * sigma * np.sqrt(h)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.6), sharey=True)
    ax = axes[0]
    f_ar = 0.0 * h
    ax.fill_between(h, f_ar - w_ar, f_ar + w_ar, color=BAND, alpha=0.9, lw=0, label='95% interval')
    ax.plot(h, f_ar, color=BLUE, lw=1.4, label='Forecast')
    ax.set_title('Stationary AR(1), $\\phi = 0.5$ ($d = 0$)', loc='left')
    ax = axes[1]
    f_rw = c * h
    ax.fill_between(h, f_rw - w_rw, f_rw + w_rw, color=BAND, alpha=0.9, lw=0)
    ax.plot(h, f_rw, color=RED, lw=1.4, label='Forecast with drift')
    ax.set_title('Random walk with drift 0.2 ($d = 1$)', loc='left')
    for ax in axes:
        ax.set_xlabel('Horizon $h$')
        ax.axhline(0, color=NAVY, lw=0.5)
    axes[0].set_ylabel('Change from $y_T$')
    fig.tight_layout(w_pad=0.8)
    st.fig_legend_bottom(fig, ncol=3)
    save(fig, 'ch3_sem_primer_fan')
    return dict(w_ar_inf=1.96 * sigma / np.sqrt(1 - phi ** 2), w_rw4=w_rw[4], w_rw16=w_rw[16])


if __name__ == '__main__':
    print('trend    ', fig_trend_vs_rw())
    fig_integration()
    print('df       ', fig_df_dist())
    print('spurious ', fig_spurious())
    print('kpss     ', fig_kpss())
    print('break    ', fig_break())
    print('fan      ', fig_fan())
