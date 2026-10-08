"""
seminar1_explainers.py -- explanatory (primer) charts for Seminar 1 (TSA): stochastic processes and stationarity
===============================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 1, which takes
place BEFORE Lecture 1. All charts use SIMULATED data only (fixed seeds): paths of white noise, MA, AR and random
walk processes, many paths at once (the ensemble), theoretical and sample ACFs, the 95% band, the chi-square
distribution of the portmanteau tests, volatility clustering, transformations and a rolling-window estimate.
They contain no exercise answers.

Output: charts/ch1_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).
The charts are drawn at their size on the slide, so the text is 7.5-8.5 pt on the slide.

Run:  python3 Quantlets/Ch_01/seminar1_explainers.py

Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import tsa_style as st   # noqa: E402

st.apply()
# drawn at the slide size (the beamer text width is 5.67 in): 8-8.5 pt text on the slide
plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'lines.linewidth': 1.0})
HALF = (2.9, 2.3)       # one column of a two-column frame
BLUE, RED, GREEN, AMBER, PURPLE, NAVY = st.MainBlue, st.IDAred, st.Forest, st.Amber, st.Purple, st.DarkText
BAND = '#C5D2E8'
MINUS = '−'


def save(fig, name):
    st.check_no_grey(fig)
    st.save_fig(name)


def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    d = np.sum(x ** 2)
    return np.array([np.sum(x[k:] * x[:len(x) - k]) / d for k in range(nlags + 1)])


def ar1(phi, n, rng, burn=200, sigma=1.0):
    e = sigma * rng.standard_normal(n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


# =============================================================================
# 1. four processes: one path each
# =============================================================================
def fig_processes(seed=3, T=200):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T + 1)
    wn = e[1:]
    ma = e[1:] + 0.8 * e[:-1]
    ar = ar1(0.7, T, rng)
    rw = np.cumsum(rng.standard_normal(T))
    t = np.arange(1, T + 1)
    fig, axes = plt.subplots(1, 4, figsize=(5.6, 1.6), sharex=True)
    for ax, x, col, title in ((axes[0], wn, BLUE, 'White noise'), (axes[1], ma, GREEN, 'MA(1), $\\theta = 0.8$'),
                              (axes[2], ar, AMBER, 'AR(1), $\\phi = 0.7$'), (axes[3], rw, RED, 'Random walk')):
        ax.plot(t, x, color=col, lw=0.7)
        ax.axhline(0, color=NAVY, lw=0.5, ls=':')
        ax.set_title(title, loc='left')
        ax.set_xlabel('$t$')
    axes[0].set_ylabel('$X_t$')
    fig.tight_layout(w_pad=0.6)
    save(fig, 'ch1_sem_primer_processes')


# =============================================================================
# 2. the ensemble: many paths, mean and variance at a fixed date
# =============================================================================
def fig_ensemble(seed=5, T=100, n=400):
    rng = np.random.default_rng(seed)
    t = np.arange(1, T + 1)
    phi = 0.7
    ARp = np.array([ar1(phi, T, rng) for _ in range(n)])
    RWp = np.cumsum(rng.standard_normal((n, T)), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.65))
    sd_ar = 1 / np.sqrt(1 - phi ** 2)
    for ax, P, col, title, band in ((axes[0], ARp, AMBER, 'Stationary AR(1), $\\phi = 0.7$', np.full(T, 1.96 * sd_ar)),
                                    (axes[1], RWp, RED, 'Random walk, $X_0 = 0$', 1.96 * np.sqrt(t))):
        lo, hi = np.percentile(P, [2.5, 97.5], axis=0)
        ax.fill_between(t, lo, hi, color=BAND, alpha=0.9, lw=0, label='Middle 95% of 400 paths')
        for i in range(4):
            ax.plot(t, P[i], color=col, lw=0.6, label='Four of the paths' if i == 0 else '_')
        ax.plot(t, band, color=NAVY, ls='--', lw=0.9, label='Theory: $\\pm 1.96$ standard deviations')
        ax.plot(t, -band, color=NAVY, ls='--', lw=0.9)
        ax.plot(t, P.mean(axis=0), color=GREEN, lw=1.3, label='Average over the paths')
        ax.set_title(title, loc='left')
        ax.set_xlabel('$t$')
    axes[0].set_ylabel('$X_t$')
    fig.tight_layout(w_pad=1.0)
    st.fig_legend_bottom(fig, ncol=2)
    save(fig, 'ch1_sem_primer_ensemble')
    return dict(sd_ar=sd_ar, sd_rw100=np.sqrt(100), emp_rw100=RWp[:, -1].std(), emp_ar100=ARp[:, -1].std())


# =============================================================================
# 3. theoretical ACFs and one sample ACF of each process
# =============================================================================
def fig_acf_theory(seed=9, T=300, K=10):
    rng = np.random.default_rng(seed)
    k = np.arange(0, K + 1)
    e = rng.standard_normal(T + 2)
    th = np.zeros(K + 1); th[0] = 1
    cases = []
    cases.append(('White noise', th.copy(), e[2:], BLUE))
    r = np.zeros(K + 1); r[0] = 1; r[1] = 0.8 / (1 + 0.64)
    cases.append(('MA(1), $\\theta = 0.8$', r, e[2:] + 0.8 * e[1:-1], GREEN))
    g0, g1, g2 = 1 + 0.36 + 0.16, 0.6 + 0.6 * 0.4, 0.4
    r = np.zeros(K + 1); r[0] = 1; r[1] = g1 / g0; r[2] = g2 / g0
    cases.append(('MA(2), $\\theta = (0.6, 0.4)$', r, e[2:] + 0.6 * e[1:-1] + 0.4 * e[:-2], PURPLE))
    cases.append(('AR(1), $\\phi = 0.7$', 0.7 ** k, ar1(0.7, T, rng), AMBER))
    fig, axes = plt.subplots(1, 4, figsize=(5.6, 1.7), sharey=True)
    b = 1.96 / np.sqrt(T)
    for ax, (title, rt, x, col) in zip(axes, cases):
        ax.axhspan(-b, b, color=BAND, alpha=0.9, lw=0)
        ax.vlines(k[1:], 0, rt[1:], color=col, lw=2.2, label='Theoretical $\\rho(h)$')
        ax.plot(k[1:], acf(x, K)[1:], 'o', color=NAVY, ms=2.6, label=f'Sample $\\hat\\rho(h)$, $T$ = {T}')
        ax.axhline(0, color=NAVY, lw=0.5)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Lag $h$')
        ax.set_xticks([1, 5, 10])
        ax.set_ylim(-0.35, 1.0)
    axes[0].fill_between([], [], color=BAND, label='Band $\\pm 1.96/\\sqrt{T}$')
    fig.tight_layout(w_pad=0.5)
    st.fig_legend_bottom(fig, ncol=3)
    save(fig, 'ch1_sem_primer_acf_theory')
    return dict(ma1=cases[1][1][1], ma2_1=cases[2][1][1], ma2_2=cases[2][1][2])


# =============================================================================
# 4. sample ACF of i.i.d. noise and the 95% band
# =============================================================================
def fig_band(seed=7, T=500, K=40):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(T)
    r = acf(x, K)
    b = 1.96 / np.sqrt(T)
    out = np.abs(r[1:]) > b
    fig, ax = plt.subplots(figsize=HALF)
    ax.axhspan(-b, b, color=BAND, alpha=0.9, lw=0, label=f'Band $\\pm 1.96/\\sqrt{{T}} = \\pm${b:.3f}')
    ks = np.arange(1, K + 1)
    ax.vlines(ks[~out], 0, r[1:][~out], color=BLUE, lw=1.6, label='Inside the band')
    ax.vlines(ks[out], 0, r[1:][out], color=RED, lw=1.6, label='Outside, by chance')
    ax.axhline(0, color=NAVY, lw=0.5)
    ax.set_xlabel('Lag $h$')
    ax.set_ylabel('$\\hat\\rho(h)$')
    ax.set_title(f'i.i.d. Normal noise, $T$ = {T}', loc='left')
    ax.set_ylim(-0.15, 0.15)
    st.legend_outside_bottom(ax, ncol=1)
    fig.tight_layout()
    save(fig, 'ch1_sem_primer_band')
    return dict(n_out=int(out.sum()), band=b, lags=[int(v) for v in ks[out]])


# =============================================================================
# 5. the chi-square distribution: critical value and p-value
# =============================================================================
def fig_chi2(m=10, q_obs=12.5):
    x = np.linspace(0, 32, 801)
    f = stats.chi2.pdf(x, m)
    cv = stats.chi2.ppf(0.95, m)
    p = stats.chi2.sf(q_obs, m)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, f, color=BLUE, lw=1.5, label=f'Density of $\\chi^2({m})$')
    ax.fill_between(x, 0, f, where=x >= q_obs, color=BAND, alpha=0.95, lw=0, label=f'p-value of $Q^* = {q_obs}$: {p:.2f}')
    ax.fill_between(x, 0, f, where=x >= cv, color=RED, alpha=0.6, lw=0, label='Rejection region, 5%')
    ax.axvline(cv, color=RED, lw=0.9, ls='--')
    ax.axvline(q_obs, color=GREEN, lw=1.1)
    ax.text(cv + 0.5, f.max() * 0.85, f'{cv:.2f}', color=RED, fontsize=8)
    ax.text(q_obs - 0.5, f.max() * 0.3, f'{q_obs}', color=GREEN, fontsize=8, ha='right')
    ax.set_xlabel('Value of the statistic')
    ax.set_ylabel('Density')
    ax.set_title(f'Chi-square, $m$ = {m} degrees of freedom', loc='left')
    st.legend_outside_bottom(ax, ncol=1)
    fig.tight_layout()
    save(fig, 'ch1_sem_primer_chi2')
    return dict(cv=cv, p=p, mean=m)


# =============================================================================
# 6. volatility clustering: returns uncorrelated, squares correlated
# =============================================================================
def fig_clustering(seed=8, T=1500, K=20):
    rng = np.random.default_rng(seed)
    w, a, b = 0.05, 0.10, 0.85
    s2 = w / (1 - a - b)
    r = np.zeros(T + 300)
    for t in range(1, T + 300):
        s2 = w + a * r[t - 1] ** 2 + b * s2
        r[t] = np.sqrt(s2) * rng.standard_normal()
    r = r[300:]
    band = 1.96 / np.sqrt(T)
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.65), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    ax = axes[0]
    ax.plot(np.arange(T), r, color=BLUE, lw=0.4)
    ax.set_title('Simulated returns $r_t$', loc='left')
    ax.set_xlabel('$t$ (days)')
    ra, r2 = acf(r, K), acf(r ** 2, K)
    ks = np.arange(1, K + 1)
    for ax, rr, col, title in ((axes[1], ra, BLUE, 'ACF of $r_t$'), (axes[2], r2, RED, 'ACF of $r_t^2$')):
        ax.axhspan(-band, band, color=BAND, alpha=0.9, lw=0)
        ax.vlines(ks, 0, rr[1:], color=col, lw=1.6)
        ax.axhline(0, color=NAVY, lw=0.5)
        ax.set_ylim(-0.1, 0.35)
        ax.set_title(title, loc='left')
        ax.set_xlabel('Lag $h$')
    axes[1].fill_between([], [], color=BAND, label=f'Band $\\pm 1.96/\\sqrt{{T}}$, $T$ = {T}')
    fig.tight_layout(w_pad=0.6)
    st.fig_legend_bottom(fig, ncol=1)
    save(fig, 'ch1_sem_primer_clustering')
    return dict(r1=ra[1], sq1=r2[1], sq10=r2[10])


# =============================================================================
# 7. transformations of a growing seasonal quarterly series
# =============================================================================
def fig_transform(seed=12, years=20):
    rng = np.random.default_rng(seed)
    n = 4 * years
    t = np.arange(n)
    season = np.tile([-0.25, 0.02, 0.12, 0.08], years)
    lny = np.log(50) + 0.012 * t + season + 0.012 * np.cumsum(rng.standard_normal(n)) * 0.6
    y = np.exp(lny)
    x = 2006 + t / 4
    d1 = 100 * np.diff(lny)
    d4 = 100 * (lny[4:] - lny[:-4])
    fig, axes = plt.subplots(1, 4, figsize=(5.6, 1.6))
    for ax, xx, yy, col, title in ((axes[0], x, y, BLUE, 'Level $Y_t$'), (axes[1], x, lny, GREEN, '$\\ln Y_t$'),
                                   (axes[2], x[1:], d1, AMBER, '$100\\,\\Delta\\ln Y_t$'),
                                   (axes[3], x[4:], d4, RED, '$100\\,\\Delta_4\\ln Y_t$')):
        ax.plot(xx, yy, color=col, lw=0.8)
        ax.set_title(title, loc='left')
        ax.set_xticks([2010, 2020])
    fig.tight_layout(w_pad=0.5)
    save(fig, 'ch1_sem_primer_transform')
    return dict(d1_sd=d1.std(ddof=1), d4_sd=d4.std(ddof=1), d4_mean=d4.mean())


# =============================================================================
# 8. a rolling-window estimate of rho(1)
# =============================================================================
def fig_rolling(seed=4, T=240, w=60):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T + 100)
    x = np.zeros(T + 100)
    phis = np.r_[np.full(100 + T // 2, 0.2), np.full(T // 2, 0.6)]
    for t in range(1, T + 100):
        x[t] = phis[t] * x[t - 1] + e[t]
    x, phis = x[100:], phis[100:]
    est = np.full(T, np.nan)
    for t in range(w - 1, T):
        est[t] = acf(x[t - w + 1:t + 1], 1)[1]
    tt = np.arange(1, T + 1)
    se = 1 / np.sqrt(w)
    fig, ax = plt.subplots(figsize=(5.6, 1.55))
    ax.fill_between(tt, est - 1.96 * se, est + 1.96 * se, color=BAND, alpha=0.9, lw=0,
                    label=f'Estimate $\\pm 1.96/\\sqrt{{{w}}}$')
    ax.plot(tt, est, color=BLUE, lw=1.2, label=f'Rolling $\\hat\\rho(1)$, window of {w} months')
    ax.plot(tt, phis, color=RED, lw=1.2, ls='--', label='True $\\rho(1)$')
    ax.axvline(w, color=NAVY, lw=0.6, ls=':')
    ax.set_xlabel('Month $t$ (end of the window)')
    ax.set_ylabel('$\\rho(1)$')
    ax.set_xlim(1, T)
    fig.tight_layout()
    st.fig_legend_bottom(fig, ncol=3)
    save(fig, 'ch1_sem_primer_rolling')
    return dict(se=se, first=w)


if __name__ == '__main__':
    fig_processes()
    print('ensemble ', {k: round(float(v), 2) for k, v in fig_ensemble().items()})
    print('acf      ', {k: round(float(v), 3) for k, v in fig_acf_theory().items()})
    print('band     ', fig_band())
    print('chi2     ', {k: round(float(v), 3) for k, v in fig_chi2().items()})
    print('cluster  ', {k: round(float(v), 3) for k, v in fig_clustering().items()})
    print('transform', {k: round(float(v), 2) for k, v in fig_transform().items()})
    print('rolling  ', fig_rolling())
