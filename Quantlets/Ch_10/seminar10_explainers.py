"""
seminar10_explainers.py -- explanatory (primer) charts for Seminar 10 (TSA): state space, Kalman filter, Markov switching
========================================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 10, which takes place
BEFORE Lecture 10. All charts use SIMULATED data only (fixed seeds): they illustrate the concepts (the local level
model and its signal-to-noise ratio, the Kalman filter with a missing block, the Kalman gain and its steady state,
filtered against smoothed estimates, the forecast variance, trend and cycle with the HP filter, a two-regime Markov
switching series, regime durations, one Hamilton-filter update) and contain no exercise answers.

Charts are drawn at the size of their box on the slide (1 pt in the figure = 1 pt on the slide).
Output: charts/ch10_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).
Run:  python3 Quantlets/Ch_10/seminar10_explainers.py
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
from statsmodels.tsa.filters.hp_filter import hpfilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import tsa_style as st                     # noqa: E402

warnings.filterwarnings('ignore')
st.apply()
# sizes for the slide: full width 5.45 in (box 0.98\textwidth x 0.66\textheight), a column chart 2.6 in wide
# (0.47\textwidth x 0.86\textheight); the legend is added below
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.0, 'pdf.fonttype': 42})
FULL_L = (5.45, 1.55)
HALF = (2.6, 1.95)


def legend(fig, ncol=3):
    fig.tight_layout()
    st.fig_legend_bottom(fig, ncol=ncol)


def save(name):
    st.check_no_grey(plt.gcf())
    st.save_fig(f'ch10_sem_primer_{name}')


def local_level(n, s2e, s2n, rng, mu0=0.0):
    mu = mu0 + np.r_[0, np.cumsum(np.sqrt(s2n) * rng.standard_normal(n - 1))]
    return mu + np.sqrt(s2e) * rng.standard_normal(n), mu


def kalman_ll(y, s2e, s2n, a1=0.0, P1=1e6):
    """Kalman filter of the local level model; y may contain np.nan (missing). Returns a_t|t, P_t|t, K_t, a_t, P_t."""
    n = len(y)
    a, P = a1, P1
    af, Pf, K, ap, Pp = (np.zeros(n) for _ in range(5))
    for t in range(n):
        ap[t], Pp[t] = a, P
        if np.isnan(y[t]):
            k, att, Ptt = 0.0, a, P
        else:
            F = P + s2e
            k = P / F
            att, Ptt = a + k * (y[t] - a), P * (1 - k)
        af[t], Pf[t], K[t] = att, Ptt, k
        a, P = att, Ptt + s2n
    return af, Pf, K, ap, Pp


def smoother_ll(af, Pf, ap, Pp):
    """Rauch-Tung-Striebel smoother of the local level model."""
    n = len(af)
    s = af.copy()
    for t in range(n - 2, -1, -1):
        J = Pf[t] / Pp[t + 1]
        s[t] = af[t] + J * (s[t + 1] - ap[t + 1])
    return s


def kbar(q):
    P = (q + np.sqrt(q ** 2 + 4 * q)) / 2
    return P / (P + 1)


# =============================================================================
# 1. the local level model for a small and a large signal-to-noise ratio
# =============================================================================
def fig_local_level(n=150, seed=1):
    fig, axes = plt.subplots(1, 2, figsize=FULL_L, sharey=False)
    for ax, q in zip(axes, (0.02, 1.0)):
        rng = np.random.default_rng(seed)
        y, mu = local_level(n, 1.0, q, rng)
        ax.plot(y, color=st.MainBlue, lw=0.6, marker='o', ms=1.2, label='Observation $y_t = \\mu_t + \\varepsilon_t$')
        ax.plot(mu, color=st.IDAred, lw=1.4, label='Level $\\mu_t$ (random walk)')
        ax.set_xlabel('Time $t$')
        ax.set_title(f'$q = \\sigma^2_\\eta/\\sigma^2_\\varepsilon = {q:g}$', loc='left')
    axes[0].set_ylabel('$y_t$')
    legend(fig, ncol=2)
    save('local_level')


# =============================================================================
# 2. the Kalman filter with a missing block
# =============================================================================
def fig_kalman(n=100, seed=3, q=0.1, gap=(55, 70)):
    rng = np.random.default_rng(seed)
    y, mu = local_level(n, 1.0, q, rng, mu0=5)
    yo = y.copy()
    yo[gap[0]:gap[1]] = np.nan
    af, Pf, K, ap, Pp = kalman_ll(yo, 1.0, q, a1=yo[0], P1=1.0)
    t = np.arange(n)
    fig, ax = plt.subplots(figsize=FULL_L)
    ax.fill_between(t, af - 1.96 * np.sqrt(Pf), af + 1.96 * np.sqrt(Pf), color=st.MainBlue, alpha=0.18, lw=0,
                    label='95% band $a_{t|t} \\pm 1.96\\sqrt{P_{t|t}}$')
    ax.plot(t, yo, 'o', ms=1.8, color=st.MainBlue, label='Observations $y_t$')
    ax.plot(t, mu, color=st.Forest, lw=1.0, ls='--', label='True level $\\mu_t$')
    ax.plot(t, af, color=st.IDAred, lw=1.3, label='Filtered level $a_{t|t}$')
    ax.axvspan(gap[0] - 0.5, gap[1] - 0.5, color=st.Amber, alpha=0.15, lw=0, label='Missing observations')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$y_t$')
    ax.set_title(f'Local level, $\\sigma^2_\\varepsilon$ = 1, $\\sigma^2_\\eta$ = {q}: no data, no update, a wider band', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=3)
    save('kalman')


# =============================================================================
# 3. the Kalman gain: convergence and the steady state against q
# =============================================================================
def fig_gain(n=15):
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    ax = axes[0]
    for q, c in ((0.04, st.MainBlue), (0.25, st.Forest), (1.0, st.IDAred)):
        _, _, K, _, _ = kalman_ll(np.zeros(n), 1.0, q, P1=1.0)
        ax.plot(np.arange(1, n + 1), K, color=c, marker='o', ms=2, label=f'$q$ = {q:g}: $\\bar K$ = {kbar(q):.3f}')
        ax.axhline(kbar(q), color=c, lw=0.6, ls=':')
    ax.set_xlabel('Step $t$')
    ax.set_ylabel('Kalman gain $K_t$')
    ax.set_title('$K_t$ converges to $\\bar K$ (start $P_1 = 1$)', loc='left')
    ax = axes[1]
    q = np.logspace(-3, 2, 300)
    ax.semilogx(q, kbar(q), color=st.Purple, lw=1.4, label='$\\bar K = \\alpha_{SES}$')
    ax.set_xlabel('$q = \\sigma^2_\\eta/\\sigma^2_\\varepsilon$ (log scale)')
    ax.set_ylabel('Steady-state gain $\\bar K$')
    ax.set_title('Noisy data (small $q$): small weight on $y_t$', loc='left')
    legend(fig, ncol=4)
    save('gain')


# =============================================================================
# 4. filtered against smoothed level
# =============================================================================
def fig_smoothed(n=120, seed=5, q=0.05):
    rng = np.random.default_rng(seed)
    y, mu = local_level(n, 1.0, q, rng)
    af, Pf, K, ap, Pp = kalman_ll(y, 1.0, q, a1=y[0], P1=1.0)
    sm = smoother_ll(af, Pf, ap, Pp)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(y, 'o', ms=1.3, color=st.MainBlue, label='$y_t$')
    ax.plot(mu, color=st.Forest, lw=1.0, ls='--', label='True $\\mu_t$')
    ax.plot(af, color=st.IDAred, lw=1.1, label='Filtered $a_{t|t}$ (data to $t$)')
    ax.plot(sm, color=st.Amber, lw=1.4, label='Smoothed $\\hat\\mu_t$ (all data)')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Level')
    ax.set_title('The filter lags; the smoother does not', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=2)
    save('smoothed')


# =============================================================================
# 5. forecasts of three state space models
# =============================================================================
def fig_forecasts(H=12):
    h = np.arange(0, H + 1)
    fig, axes = plt.subplots(1, 2, figsize=(5.45, 1.3))
    ax = axes[0]
    P, s2e, s2n, a = 1.0, 1.0, 0.25, 10.0
    var = P + h * s2n + s2e
    ax.fill_between(h[1:], a - 1.96 * np.sqrt(var[:-1]), a + 1.96 * np.sqrt(var[:-1]), color=st.MainBlue, alpha=0.18, lw=0,
                    label='95% interval')
    ax.plot(h[1:], np.full(H, a), color=st.MainBlue, lw=1.3, label='Local level: flat forecast $a_{n+1}$')
    ax.set_xlabel('Horizon $h$')
    ax.set_ylabel('Forecast')
    ax.set_title('Local level: the interval widens with $h$', loc='left')
    ax = axes[1]
    phi1, phi2 = 0.5, 0.3
    x = [1.0, 2.0]
    for _ in range(H):
        x.append(phi1 * x[-1] + phi2 * x[-2])
    ax.plot(h, x[1:], color=st.IDAred, marker='o', ms=2, label='AR(2): back to the mean 0')
    ax.plot(h, 2.0 + 0.5 * h, color=st.Forest, marker='o', ms=2, label='Local linear trend: a straight line')
    ax.axhline(0, color=st.DarkText, lw=0.5, ls=':')
    ax.set_xlabel('Horizon $h$')
    ax.set_title('$\\hat\\alpha_{T+h} = T\\,\\hat\\alpha_{T+h-1}$', loc='left')
    legend(fig, ncol=3)
    save('forecasts')


# =============================================================================
# 6. trend and cycle: the HP filter on a simulated log GDP
# =============================================================================
def fig_hp(n=160, seed=6):
    rng = np.random.default_rng(seed)
    slope = 0.6 + np.cumsum(0.02 * rng.standard_normal(n))
    trend = 100 + np.cumsum(slope)
    c = np.zeros(n)
    for t in range(2, n):
        c[t] = 1.4 * c[t - 1] - 0.5 * c[t - 2] + 0.6 * rng.standard_normal()
    y = trend + c
    cyc, tr = hpfilter(y, lamb=1600)
    end_cyc = [hpfilter(y[:k], lamb=1600)[0][-1] for k in range(40, n + 1)]
    t = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    ax = axes[0]
    ax.plot(t, c, color=st.MainBlue, lw=1.1, label='True cycle $\\psi_t$')
    ax.plot(t, cyc, color=st.IDAred, lw=1.1, label='HP gap (two-sided, full sample)')
    ax.axhline(0, color=st.DarkText, lw=0.5, ls=':')
    ax.set_xlabel('Quarter $t$')
    ax.set_ylabel('Gap (% of trend)')
    ax.set_title('$100\\log \\mathrm{GDP}_t = \\mu_t + \\psi_t$', loc='left')
    ax = axes[1]
    ax.plot(t, c, color=st.MainBlue, lw=1.1)
    ax.plot(np.arange(39, n), end_cyc, color=st.Amber, lw=1.1, label='HP gap at the end of each sample (real time)')
    ax.axhline(0, color=st.DarkText, lw=0.5, ls=':')
    ax.set_xlabel('Quarter $t$')
    ax.set_title('The last HP values are pulled to zero', loc='left')
    legend(fig, ncol=3)
    save('hp')
    return dict(corr_full=float(np.corrcoef(c, cyc)[0, 1]), corr_rt=float(np.corrcoef(c[39:], end_cyc)[0, 1]),
                sd_true_end=float(np.std(c[39:])), sd_rt=float(np.std(end_cyc)))


# =============================================================================
# 7. a two-regime Markov switching series with filtered and smoothed probabilities
# =============================================================================
def hamilton(y, mu, sig, P):
    n = len(y)
    xi = np.zeros((n, 2))
    pred = np.zeros((n, 2))
    p = np.array([(1 - P[1, 1]) / (2 - P[0, 0] - P[1, 1]), 0])
    p[1] = 1 - p[0]
    for t in range(n):
        pr = p @ P if t else p
        f = stats.norm.pdf(y[t], mu, sig)
        xi[t] = pr * f / (pr * f).sum()
        pred[t] = pr
        p = xi[t]
    sm = xi.copy()
    for t in range(n - 2, -1, -1):
        sm[t] = xi[t] * (P @ (sm[t + 1] / pred[t + 1]))
    return xi, sm


def fig_markov(n=200, seed=7):
    rng = np.random.default_rng(seed)
    P = np.array([[0.90, 0.10], [0.03, 0.97]])
    mu, sig = np.array([-0.5, 0.8]), np.array([1.2, 0.5])
    s = np.zeros(n, int)
    s[0] = 1
    for t in range(1, n):
        s[t] = rng.choice(2, p=P[s[t - 1]])
    y = mu[s] + sig[s] * rng.standard_normal(n)
    xi, sm = hamilton(y, mu, sig, P)
    t = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    for ax in axes:
        ax.fill_between(t, 0, 1, where=s == 0, transform=ax.get_xaxis_transform(), color=st.IDAred, alpha=0.12, lw=0,
                        step='mid', label='True regime 1 (recession)')
    ax = axes[0]
    ax.plot(t, y, color=st.MainBlue, lw=0.8, label='$y_t$')
    ax.set_xlabel('Quarter $t$')
    ax.set_ylabel('Growth $y_t$')
    ax.set_title('$y_t = \\mu_{S_t} + \\varepsilon_t$: two means, two variances', loc='left')
    ax = axes[1]
    ax.plot(t, xi[:, 0], color=st.Amber, lw=1.0, label='Filtered $\\Pr(S_t = 1 \\mid Y_t)$')
    ax.plot(t, sm[:, 0], color=st.Forest, lw=1.2, label='Smoothed $\\Pr(S_t = 1 \\mid Y_n)$')
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Quarter $t$')
    ax.set_ylabel('Probability of regime 1')
    ax.set_title('Hamilton filter and smoother', loc='left')
    legend(fig, ncol=4)
    save('markov')


# =============================================================================
# 8. regime durations: the geometric distribution
# =============================================================================
def fig_durations(K=30):
    k = np.arange(1, K + 1)
    fig, ax = plt.subplots(figsize=HALF)
    for p, c in ((0.75, st.IDAred), (0.95, st.MainBlue)):
        ax.bar(k + (-0.2 if p < 0.9 else 0.2), (1 - p) * p ** (k - 1), width=0.4, color=c,
               label=f'$p_{{ii}}$ = {p}: mean $1/(1-p_{{ii}})$ = {1 / (1 - p):.0f}')
    ax.set_xlabel('Duration $D$ (periods)')
    ax.set_ylabel('$\\Pr(D = k)$')
    ax.set_title('$\\Pr(D = k) = (1 - p_{ii})\\,p_{ii}^{k-1}$', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=1)
    save('durations')


# =============================================================================
# 9. one Hamilton-filter update: prior probability times density
# =============================================================================
def fig_update(mu1=-1.0, mu2=1.5, sig=1.0, prior=0.3, y=-0.2):
    x = np.linspace(-4.5, 5, 400)
    f1, f2 = stats.norm.pdf(x, mu1, sig), stats.norm.pdf(x, mu2, sig)
    fy1, fy2 = stats.norm.pdf(y, mu1, sig), stats.norm.pdf(y, mu2, sig)
    post = prior * fy1 / (prior * fy1 + (1 - prior) * fy2)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, prior * f1, color=st.IDAred, label='$\\Pr(S_t = 1 \\mid Y_{t-1})\\,f_1(y)$')
    ax.plot(x, (1 - prior) * f2, color=st.MainBlue, label='$\\Pr(S_t = 2 \\mid Y_{t-1})\\,f_2(y)$')
    ax.axvline(y, color=st.Forest, lw=1.0, ls='--', label=f'Observed $y_t = {y:g}$')
    ax.plot([y, y], [0, prior * fy1], color=st.IDAred, lw=2.5)
    ax.plot([y + 0.08, y + 0.08], [0, (1 - prior) * fy2], color=st.MainBlue, lw=2.5)
    ax.set_xlabel('$y$')
    ax.set_ylabel('Weighted density')
    ax.set_title(f'Prior {prior:g} for regime 1, after $y_t$: {post:.2f}', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=1)
    save('update')
    return dict(post=float(post), fy1=float(fy1), fy2=float(fy2))


if __name__ == '__main__':
    for fn in (fig_local_level, fig_kalman, fig_gain, fig_smoothed, fig_forecasts, fig_hp, fig_markov, fig_durations,
               fig_update):
        print(fn.__name__, fn())
