"""
seminar15_explainers.py -- explanatory (primer) charts for Seminar 15 (TSA): review of Chapters 0-10
====================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 15, which takes place
BEFORE Lecture 15. All charts use SIMULATED data only (fixed seeds): they illustrate the concepts of the formula sheets
(ACF and PACF of AR(1) and MA(1), forecast intervals of ARIMA(0,1,1) and AR(1), a unit root against a stationary
series, seasonal differencing, GARCH volatility clustering and half-life, VaR with Normal and Student-t innovations,
cointegration and the error-correction spread) and contain no exercise answers.

Charts are drawn at the size of their box on the slide (1 pt in the figure = 1 pt on the slide).
Output: charts/ch15_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).
Run:  python3 Quantlets/Ch_15/seminar15_explainers.py
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
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.arima_process import ArmaProcess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import tsa_style as st                     # noqa: E402

warnings.filterwarnings('ignore')
st.apply()
# sizes for the slide: full width 5.45 in (box 0.98\textwidth x 0.66\textheight); the legend is added below
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.0, 'pdf.fonttype': 42})
FULL_L = (5.45, 1.55)


def legend(fig, ncol=3):
    fig.tight_layout()
    st.fig_legend_bottom(fig, ncol=ncol)


def save(name):
    st.check_no_grey(plt.gcf())
    st.save_fig(f'ch15_sem_primer_{name}')


# =============================================================================
# 1. theoretical ACF and PACF of an AR(1) and an MA(1)
# =============================================================================
def fig_acf_pacf(phi=0.7, theta=0.6, K=10):
    k = np.arange(1, K + 1)
    ar = ArmaProcess([1, -phi], [1])
    ma = ArmaProcess([1], [1, theta])
    fig, axes = plt.subplots(1, 4, figsize=FULL_L, sharey=True)
    panels = ((ar.acf(K + 1)[1:], f'AR(1), $\\phi$ = {phi}: ACF', st.MainBlue),
              (ar.pacf(K + 1)[1:], 'AR(1): PACF', st.MainBlue),
              (ma.acf(K + 1)[1:], f'MA(1), $\\theta$ = {theta}: ACF', st.IDAred),
              (ma.pacf(K + 1)[1:], 'MA(1): PACF', st.IDAred))
    for ax, (v, title, c) in zip(axes, panels):
        ax.bar(k, v, width=0.5, color=c)
        ax.axhline(0, color=st.DarkText, lw=0.5)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Lag $h$')
        ax.set_xticks([1, 5, 10])
    axes[0].set_ylabel('Autocorrelation')
    fig.tight_layout()
    save('acf_pacf')


# =============================================================================
# 2. forecast intervals: ARIMA(0,1,1) against a stationary AR(1)
# =============================================================================
def fig_arima_fan(theta=-0.6, phi=0.6, H=20, sigma=1.0):
    h = np.arange(1, H + 1)
    v_arima = sigma ** 2 * (1 + (h - 1) * (1 + theta) ** 2)
    v_ar = sigma ** 2 * (1 - phi ** (2 * h)) / (1 - phi ** 2)
    fig, axes = plt.subplots(1, 2, figsize=FULL_L, sharey=True)
    ax = axes[0]
    ax.fill_between(h, 100 - 1.96 * np.sqrt(v_arima), 100 + 1.96 * np.sqrt(v_arima), color=st.MainBlue, alpha=0.2, lw=0,
                    label='95% interval')
    ax.plot(h, np.full(H, 100.0), color=st.MainBlue, lw=1.3, label='Point forecast')
    ax.set_xlabel('Horizon $h$')
    ax.set_ylabel('Forecast')
    ax.set_title(f'ARIMA(0,1,1), $\\theta = {theta}$: keeps widening', loc='left')
    ax = axes[1]
    mean = 100 + 2.0 * phi ** h
    ax.fill_between(h, mean - 1.96 * np.sqrt(v_ar), mean + 1.96 * np.sqrt(v_ar), color=st.MainBlue, alpha=0.2, lw=0)
    ax.plot(h, mean, color=st.MainBlue, lw=1.3)
    ax.axhline(100, color=st.IDAred, lw=0.8, ls='--', label='Mean $\\mu$ = 100')
    ax.set_xlabel('Horizon $h$')
    ax.set_title(f'AR(1), $\\phi = {phi}$: bounded, back to the mean', loc='left')
    legend(fig, ncol=3)
    save('arima_fan')


# =============================================================================
# 3. a unit root against a stationary AR(1)
# =============================================================================
def fig_unit_root(n=300, seed=3, phi=0.8):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    rw = np.cumsum(e)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = phi * ar[t - 1] + e[t]
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    ax = axes[0]
    ax.plot(rw, color=st.IDAred, lw=0.9, label='Random walk: $y_t = y_{t-1} + \\varepsilon_t$ ($\\gamma = 0$)')
    ax.plot(ar, color=st.MainBlue, lw=0.9, label=f'AR(1): $y_t = {phi}\\,y_{{t-1}} + \\varepsilon_t$ ($\\gamma = {phi - 1:.1f}$)')
    ax.axhline(0, color=st.DarkText, lw=0.5, ls=':')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$y_t$')
    ax.set_title('The same shocks', loc='left')
    ax = axes[1]
    K = 30
    k = np.arange(1, K + 1)
    ax.bar(k - 0.2, sm_acf(rw, nlags=K)[1:], width=0.4, color=st.IDAred)
    ax.bar(k + 0.2, sm_acf(ar, nlags=K)[1:], width=0.4, color=st.MainBlue)
    ax.axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color=st.Forest, alpha=0.15, lw=0, label='$\\pm 1.96/\\sqrt{T}$')
    ax.set_xlabel('Lag $h$')
    ax.set_ylabel('Sample ACF')
    ax.set_title('Unit root: the ACF decays very slowly', loc='left')
    legend(fig, ncol=3)
    save('unit_root')


# =============================================================================
# 4. seasonal differencing of a monthly series
# =============================================================================
def fig_seasonal(n=144, seed=4, s=12):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    season = np.tile(np.array([-3, -2, 1, 2, 3, 4, 2, -4, 1, 2, 0, -6.0]), n // s)
    e = rng.standard_normal(n + 13)
    # airline model: (1 - L)(1 - L^12) y = (1 - 0.4L)(1 - 0.6L^12) e, plus a deterministic season for the picture
    u = e[13:] - 0.4 * e[12:-1] - 0.6 * e[1:-12] + 0.24 * e[:-13]
    z = u
    y = np.zeros(n)
    for i in range(n):
        y[i] = z[i] + (y[i - 1] if i >= 1 else 0) + (y[i - s] if i >= s else 0) - (y[i - s - 1] if i >= s + 1 else 0)
    y = 100 + 0.3 * t + season + 0.6 * y / y.std()
    zz = np.diff(y[s:] - y[:-s])
    K = 26
    k = np.arange(1, K + 1)
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    ax = axes[0]
    ax.plot(t, y, color=st.MainBlue, lw=0.9, label='$y_t$: trend + season + noise')
    ax.set_xlabel('Month $t$')
    ax.set_ylabel('$y_t$')
    ax.set_title('A monthly series ($s$ = 12)', loc='left')
    ax = axes[1]
    ax.bar(k, sm_acf(zz, nlags=K)[1:], width=0.5, color=st.IDAred, label='ACF of $z_t = \\Delta\\Delta_{12}y_t$')
    ax.axhspan(-1.96 / np.sqrt(len(zz)), 1.96 / np.sqrt(len(zz)), color=st.Forest, alpha=0.15, lw=0,
               label='$\\pm 1.96/\\sqrt{T}$')
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_xlabel('Lag $h$')
    ax.set_title('After both differences: spikes at lags 1 and 12', loc='left')
    legend(fig, ncol=3)
    save('seasonal')


# =============================================================================
# 5. GARCH(1,1): volatility clustering and the half-life of a shock
# =============================================================================
def fig_garch(n=1500, seed=5, omega=0.02, alpha=0.08, beta=0.90):
    rng = np.random.default_rng(seed)
    s2 = np.zeros(n)
    r = np.zeros(n)
    s2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.standard_normal()
    lr = omega / (1 - alpha - beta)
    p = alpha + beta
    hl = np.log(0.5) / np.log(p)
    h = np.arange(0, 151)
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    ax = axes[0]
    ax.plot(r, color=st.MainBlue, lw=0.35, label='Returns $\\varepsilon_t$')
    ax.plot(2 * np.sqrt(s2), color=st.IDAred, lw=0.9, label='$\\pm 2\\sigma_t$')
    ax.plot(-2 * np.sqrt(s2), color=st.IDAred, lw=0.9)
    ax.set_xlabel('Day $t$')
    ax.set_ylabel('Return (%)')
    ax.set_title(f'GARCH(1,1): $\\alpha$ = {alpha}, $\\beta$ = {beta}', loc='left')
    ax = axes[1]
    start = 4 * lr
    ax.plot(h, lr + p ** h * (start - lr), color=st.Purple, lw=1.3, label='$\\sigma^2_{T+h}$ forecast')
    ax.axhline(lr, color=st.Forest, lw=0.9, ls='--', label=f'Long-run variance $\\bar\\sigma^2$ = {lr:.1f}')
    ax.axvline(hl, color=st.Amber, lw=0.9, ls=':', label=f'Half-life $h_{{1/2}}$ = {hl:.0f} days')
    ax.set_xlabel('Horizon $h$ (days)')
    ax.set_ylabel('Variance')
    ax.set_title('After a shock: back to $\\bar\\sigma^2$ at the rate $\\alpha + \\beta$', loc='left')
    legend(fig, ncol=4)
    save('garch')
    return dict(lr=lr, hl=hl, p=p)


# =============================================================================
# 6. VaR 1%: Normal against a standardised Student-t
# =============================================================================
def fig_var_t(nu=5):
    x = np.linspace(-5, 4, 600)
    sc = np.sqrt((nu - 2) / nu)
    qn = stats.norm.ppf(0.01)
    qt = stats.t.ppf(0.01, nu) * sc
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    for ax, log in zip(axes, (False, True)):
        ax.plot(x, stats.norm.pdf(x), color=st.MainBlue, label='Normal $N(0, 1)$')
        ax.plot(x, stats.t.pdf(x / sc, nu) / sc, color=st.IDAred, label=f'Student-$t_{nu}$ rescaled to variance 1')
        ax.axvline(qn, color=st.MainBlue, lw=0.9, ls='--', label=f'1% quantile, Normal: ${qn:.3f}$')
        ax.axvline(qt, color=st.IDAred, lw=0.9, ls='--', label=f'1% quantile, $t_{nu}$: ${qt:.3f}$')
        ax.set_xlabel('Standardised innovation $z$')
        if log:
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 1)
            ax.set_title('Log scale: the $t$ tail is heavier', loc='left')
        else:
            ax.set_ylabel('Density')
            ax.set_title('Same variance, different tails', loc='left')
    legend(fig, ncol=2)
    save('var_t')
    return dict(qn=qn, qt=qt)


# =============================================================================
# 7. cointegration: two I(1) series and their stationary spread
# =============================================================================
def fig_coint(n=240, seed=7, beta=1.3, gamma=-0.08):
    rng = np.random.default_rng(seed)
    x = 2 + np.cumsum(0.25 * rng.standard_normal(n))
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = (1 + gamma) * u[t - 1] + 0.3 * rng.standard_normal()
    y = 3 + beta * x + u
    hl = np.log(0.5) / np.log(1 + gamma)
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    ax = axes[0]
    ax.plot(y, color=st.IDAred, lw=1.0, label='$y_t$, $I(1)$')
    ax.plot(x, color=st.MainBlue, lw=1.0, label='$x_t$, $I(1)$')
    ax.set_xlabel('Month $t$')
    ax.set_ylabel('Level')
    ax.set_title('Both wander, but together', loc='left')
    ax = axes[1]
    ax.plot(y - 3 - beta * x, color=st.Forest, lw=1.0, label=f'Spread $y_t - 3 - {beta}\\,x_t$, $I(0)$')
    ax.axhline(0, color=st.DarkText, lw=0.5, ls=':')
    ax.set_xlabel('Month $t$')
    ax.set_title(f'Back to 0; half-life {hl:.1f} months', loc='left')
    legend(fig, ncol=3)
    save('coint')
    return dict(hl=hl)


if __name__ == '__main__':
    for fn in (fig_acf_pacf, fig_arima_fan, fig_unit_root, fig_seasonal, fig_garch, fig_var_t, fig_coint):
        print(fn.__name__, fn())
