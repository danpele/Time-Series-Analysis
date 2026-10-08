"""
seminar5_explainers.py -- explanatory (primer) charts for Seminar 5 (TSA): conditional volatility, ARCH and GARCH
================================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 5, which takes
place BEFORE Lecture 5. All charts use SIMULATED data only (fixed seeds): volatility clustering, the ACF of returns
and of squared returns, mean reversion of GARCH variance forecasts, EWMA weights, VaR with Normal and Student-t
innovations, news impact curves, the QLIKE loss and VaR exceedances. They contain no exercise answers.

The figures are drawn at the size of their box on the slide (full width: 5.6 x 1.5 in; one column: 2.8 x 2.0 in),
so that 1 pt in the figure is about 1 pt on the slide (text >= 7 pt).

Output: charts/ch5_sem_primer_*.pdf and .png (transparent background, legend below the plot).
Run:    python3 Quantlets/Ch_05/seminar5_explainers.py
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
BAND = '#C5D2E8'
FULL = (5.6, 1.5)
HALF = (2.8, 2.0)


def save(fig, name):
    st.check_no_grey(fig)
    st.save_fig(f'ch5_sem_primer_{name}')


def garch_sim(n, omega, alpha, beta, seed, dist='normal', nu=5, burn=500):
    """GARCH(1,1) returns r_t = sigma_t z_t; z_t Normal or standardised Student-t."""
    rng = np.random.default_rng(seed)
    if dist == 'normal':
        z = rng.standard_normal(n + burn)
    else:
        z = rng.standard_t(nu, n + burn) * np.sqrt((nu - 2) / nu)
    s2 = np.empty(n + burn)
    r = np.empty(n + burn)
    s2[0] = omega / (1 - alpha - beta)
    r[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n + burn):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return r[burn:], np.sqrt(s2[burn:])


def acf(x, nl):
    x = np.asarray(x) - np.mean(x)
    d = np.sum(x ** 2)
    return np.array([np.sum(x[k:] * x[:-k]) / d for k in range(1, nl + 1)])


# =============================================================================
# 1. volatility clustering: GARCH returns against i.i.d. returns with the same variance
# =============================================================================
def fig_clustering(n=1500, omega=0.02, alpha=0.08, beta=0.90, seed=3):
    r, s = garch_sim(n, omega, alpha, beta, seed)
    lr = omega / (1 - alpha - beta)
    iid = np.sqrt(lr) * np.random.default_rng(seed + 1).standard_normal(n)
    fig, ax = plt.subplots(1, 2, figsize=FULL, sharey=True)
    ax[0].plot(iid, color=G, lw=0.5, label='Return $r_t$')
    ax[0].set_title('i.i.d. Normal returns, variance 1')
    ax[1].plot(r, color=B, lw=0.5, label='_')
    ax[1].plot(2 * s, color=R, lw=0.9, label=r'$\pm 2\sigma_t$')
    ax[1].plot(-2 * s, color=R, lw=0.9, label='_')
    ax[1].set_title(r'GARCH(1,1), $\alpha = 0.08$, $\beta = 0.90$, $\bar\sigma^2 = 1$')
    for a in ax:
        a.set_xlabel('Day $t$')
    ax[0].set_ylabel('$r_t$ (%)')
    fig.tight_layout(w_pad=0.8)
    st.fig_legend_bottom(fig, ncol=2)
    save(fig, 'clustering')
    k = lambda x: float(np.mean((x - x.mean()) ** 4) / np.var(x) ** 2)
    return dict(kurt_iid=k(iid), kurt_garch=k(r), sd_iid=float(iid.std()), sd_garch=float(r.std()))


# =============================================================================
# 2. ACF of r_t and of r_t^2 for a GARCH series
# =============================================================================
def fig_acf(n=2500, seed=8, nl=30):
    r, _ = garch_sim(n, 0.02, 0.08, 0.90, seed)
    a1, a2 = acf(r, nl), acf(r ** 2, nl)
    band = 1.96 / np.sqrt(n)
    lags = np.arange(1, nl + 1)
    fig, ax = plt.subplots(1, 2, figsize=FULL, sharey=True)
    for a, v, c, ttl in [(ax[0], a1, B, 'ACF of $r_t$'), (ax[1], a2, R, 'ACF of $r_t^2$')]:
        a.vlines(lags, 0, v, color=c, lw=1.6)
        a.axhline(0, color=st.DarkText, lw=0.6)
        a.fill_between([0.5, nl + 0.5], -band, band, color=BAND, lw=0, label=r'95% band $\pm 1.96/\sqrt{T}$' if a is ax[0] else '_')
        a.set_title(ttl)
        a.set_xlabel('Lag $k$')
    ax[0].set_ylabel('Autocorrelation')
    fig.tight_layout(w_pad=0.8)
    st.fig_legend_bottom(fig, ncol=1)
    save(fig, 'acf')
    q = lambda v: float(n * (n + 2) * np.sum(v[:10] ** 2 / (n - np.arange(1, 11))))
    return dict(band=float(band), acf_r1=float(a1[0]), acf_r2_1=float(a2[0]), acf_r2_10=float(a2[9]),
                Q10_r=q(a1), Q10_r2=q(a2))


# =============================================================================
# 3. mean reversion of the variance forecasts
# =============================================================================
def fig_forecast(H=120):
    h = np.arange(1, H + 1)
    fig, ax = plt.subplots(figsize=FULL)
    out = {}
    for p, c in zip((0.90, 0.97, 1.0), (G, B, R)):
        for start, ls in ((3.0, '-'), (0.4, '--')):
            f = 1 + p ** (h - 1) * (start - 1)
            lab = (f'$\\alpha + \\beta = {p:g}$' + (' (IGARCH)' if p == 1 else '')) if start == 3.0 else '_'
            ax.plot(h, f, color=c, ls=ls, label=lab)
        if p < 1:
            hl = np.log(0.5) / np.log(p)
            out[p] = float(hl)
            ax.plot(hl + 1, 1 + 0.5 * 2.0, 'o', color=c, ms=3.5)
    ax.axhline(1, color=A, ls=':', lw=0.9, label=r'long-run variance $\bar\sigma^2 = 1$')
    ax.set_xlabel('Horizon $h$ (days)')
    ax.set_ylabel(r'$E_t[\sigma^2_{t+h}]$')
    ax.set_title(r'Start $\sigma^2_{t+1} = 3$ (solid) or $0.4$ (dashed); dots: half of the excess gone')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=4)
    save(fig, 'forecast')
    return {f'half_life_{k}': v for k, v in out.items()}


# =============================================================================
# 4. EWMA weights on past squared returns
# =============================================================================
def fig_ewma(J=100):
    j = np.arange(1, J + 1)
    fig, ax = plt.subplots(figsize=HALF)
    out = {}
    for lam, c in zip((0.94, 0.97, 0.99), (R, B, G)):
        w = (1 - lam) * lam ** (j - 1)
        ax.plot(j, w, color=c, label=f'$\\lambda = {lam}$')
        out[lam] = float(np.sum(w[:20]))
    ax.set_xlabel('Days ago $j$')
    ax.set_ylabel('Weight of $r_{t-j}^2$')
    ax.set_title(r'EWMA weights $(1 - \lambda)\lambda^{j-1}$')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=3)
    save(fig, 'ewma')
    return {f'weight_last20_lambda={k}': v for k, v in out.items()}


# =============================================================================
# 5. VaR 1% with Normal and standardised Student-t innovations
# =============================================================================
def fig_var(nu=4):
    x = np.linspace(-5, 3.5, 600)
    sc = np.sqrt((nu - 2) / nu)
    fN = stats.norm.pdf(x)
    fT = stats.t.pdf(x / sc, nu) / sc
    qN = stats.norm.ppf(0.01)
    qT = stats.t.ppf(0.01, nu) * sc
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    ax[0].plot(x, fN, color=B, label='Normal $N(0, 1)$')
    ax[0].plot(x, fT, color=R, label=f'Standardised Student-t, $\\nu = {nu}$')
    for q, c in ((qN, B), (qT, R)):
        ax[0].axvline(q, color=c, ls='--', lw=0.9)
    ax[0].set_title('Densities of $z_t$, both with variance 1')
    ax[0].set_xlabel('$z$')
    xx = np.linspace(-6, -1.5, 300)
    ax[1].plot(xx, stats.norm.cdf(xx), color=B)
    ax[1].plot(xx, stats.t.cdf(xx / sc, nu), color=R)
    ax[1].axhline(0.01, color=A, ls=':', lw=0.9, label='Probability 1%')
    ax[1].plot([qN], [0.01], 'o', color=B, ms=3.5)
    ax[1].plot([qT], [0.01], 'o', color=R, ms=3.5)
    ax[1].annotate(f'$q_{{0.01}} = {qN:.3f}$', (qN, 0.01), xytext=(5, -14), textcoords='offset points', color=B, fontsize=7.5)
    ax[1].annotate(f'$q_{{0.01}} = {qT:.3f}$', (qT, 0.01), xytext=(-62, 14), textcoords='offset points', color=R, fontsize=7.5)
    ax[1].set_yscale('log')
    ax[1].set_ylim(1e-4, 0.1)
    ax[1].set_title('Left tail: $P(z \\leq x)$, log scale')
    ax[1].set_xlabel('$x$')
    fig.tight_layout(w_pad=0.8)
    st.fig_legend_bottom(fig, ncol=3)
    save(fig, 'var')
    return dict(qN=float(qN), qT=float(qT), ratio=float(qT / qN))


# =============================================================================
# 6. news impact curves: GARCH, GJR-GARCH and EGARCH
# =============================================================================
def fig_nic():
    e = np.linspace(-4, 4, 401)
    s2 = 1.0
    garch = 0.03 + 0.07 * e ** 2 + 0.90 * s2
    gjr = 0.03 + (0.02 + 0.10 * (e < 0)) * e ** 2 + 0.90 * s2
    Ez = np.sqrt(2 / np.pi)
    egarch = np.exp(0.0 + 0.15 * (np.abs(e) - Ez) - 0.06 * e + 0.97 * np.log(s2))
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(e, garch, color=B, label='GARCH')
    ax.plot(e, gjr, color=R, label='GJR-GARCH')
    ax.plot(e, egarch, color=G, label='EGARCH')
    ax.axvline(0, color=st.DarkText, lw=0.5, ls=':')
    ax.set_xlabel(r'Shock $\varepsilon_t$ (with $\sigma_t^2 = 1$)')
    ax.set_ylabel(r'$\sigma^2_{t+1}$')
    ax.set_title('News impact curves')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=3)
    save(fig, 'nic')
    f = lambda arr, x: float(np.interp(x, e, arr))
    return dict(garch_m2=f(garch, -2), garch_p2=f(garch, 2), gjr_m2=f(gjr, -2), gjr_p2=f(gjr, 2),
                eg_m2=f(egarch, -2), eg_p2=f(egarch, 2))


# =============================================================================
# 7. QLIKE and squared loss as functions of the forecast
# =============================================================================
def fig_qlike():
    h = np.linspace(0.2, 4, 400)
    r2 = 1.0
    q = r2 / h + np.log(h)
    se = (r2 - h) ** 2
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(h, q - q.min(), color=B, label=r'QLIKE $r_t^2/h_t + \ln h_t$')
    ax.plot(h, se, color=R, ls='--', label=r'Squared error $(r_t^2 - h_t)^2$')
    ax.axvline(1, color=A, ls=':', lw=0.9, label='$h_t = r_t^2 = 1$')
    ax.set_ylim(0, 3)
    ax.set_xlabel('Variance forecast $h_t$')
    ax.set_ylabel('Loss (minimum set to 0)')
    ax.set_title('Loss when $r_t^2 = 1$')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=1)
    save(fig, 'qlike')
    qq = lambda x: float(r2 / x + np.log(x) - 1)
    return dict(qlike_half=qq(0.5), qlike_double=qq(2.0), se_half=0.25, se_double=1.0)


# =============================================================================
# 8. VaR 1% exceedances: constant-variance VaR against GARCH VaR
# =============================================================================
def fig_exceed(n=1000, seed=21):
    r, s = garch_sim(n, 0.02, 0.10, 0.88, seed, dist='t', nu=5)
    q = stats.norm.ppf(0.01)
    qt = stats.t.ppf(0.01, 5) * np.sqrt(3 / 5)
    var_g = -qt * s
    var_c = -q * r.std() * np.ones(n)
    ex_g = r < -var_g
    ex_c = r < -var_c
    t = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=FULL)
    ax.plot(t, r, color=B, lw=0.5, label='Return $r_t$')
    ax.plot(t, -var_g, color=R, lw=0.9, label='$-$VaR 1%, GARCH-t')
    ax.plot(t, -var_c, color=G, lw=0.9, ls='--', label='$-$VaR 1%, constant variance, Normal')
    ax.plot(t[ex_c], r[ex_c], 'o', color=G, ms=3, mfc='none', label=f'exceedances, constant ({ex_c.sum()})')
    ax.plot(t[ex_g], r[ex_g], 'x', color=R, ms=3.5, label=f'exceedances, GARCH-t ({ex_g.sum()})')
    ax.set_xlabel('Day $t$')
    ax.set_ylabel('%')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=3)
    save(fig, 'exceed')
    return dict(n=n, expected=n * 0.01, exc_garch=int(ex_g.sum()), exc_const=int(ex_c.sum()))


if __name__ == '__main__':
    for f in (fig_clustering, fig_acf, fig_forecast, fig_ewma, fig_var, fig_nic, fig_qlike, fig_exceed):
        print(f.__name__, f())
