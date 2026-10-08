"""
seminar7_explainers.py -- explanatory (primer) charts for Seminar 7 (TSA): cointegration and VECM
================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 7, which takes
place BEFORE Lecture 7. All charts use SIMULATED data only (fixed seeds): a random walk against a stationary AR(1),
the spurious regression, a cointegrated pair and its spread, the null distributions of the Dickey-Fuller and
Engle-Granger statistics, the half-life of an error correction, a VECM with one adjusting variable and the
z-score rule of pairs trading. They contain no exercise answers.

The figures are drawn at the size of their box on the slide (full width: 5.6 x 1.5 in; one column: 2.8 x 2.0 in),
so that 1 pt in the figure is about 1 pt on the slide (text >= 7 pt).

Output: charts/ch7_sem_primer_*.pdf and .png (transparent background, legend below the plot).
Run:    python3 Quantlets/Ch_07/seminar7_explainers.py
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    st.save_fig(f'ch7_sem_primer_{name}')


def ols(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    s2 = e @ e / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, se, e


def df_tau(u, const):
    """Dickey-Fuller t statistic of gamma in  du_t = [c +] gamma u_{t-1} + e_t  (no lagged differences)."""
    du, ul = np.diff(u), u[:-1]
    X = np.column_stack([np.ones_like(ul), ul]) if const else ul[:, None]
    b, se, _ = ols(du, X)
    return b[-1] / se[-1]


# =============================================================================
# 1. a random walk against a stationary AR(1)
# =============================================================================
def fig_unit_root(n=300, seed=1):
    e = np.random.default_rng(seed).standard_normal(n)
    rw = np.cumsum(e)
    ar = np.zeros(n)
    for t in range(1, n):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    fig, ax = plt.subplots(figsize=FULL)
    ax.plot(rw, color=R, label=r'Random walk $y_t = y_{t-1} + \varepsilon_t$, $I(1)$')
    ax.plot(ar, color=B, label=r'AR(1) $y_t = 0.9\,y_{t-1} + \varepsilon_t$, $I(0)$')
    ax.axhline(0, color=st.DarkText, lw=0.5, ls=':')
    ax.set_xlabel('Period $t$')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=2)
    save(fig, 'unit_root')
    return dict(rw_range=[float(rw.min()), float(rw.max())], ar_range=[float(ar.min()), float(ar.max())])


# =============================================================================
# 2. the spurious regression
# =============================================================================
def fig_spurious(n=200, reps=2000, seed=2):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.standard_normal(n))
    y = np.cumsum(rng.standard_normal(n))
    ts, r2 = [], []
    for _ in range(reps):
        a = np.cumsum(rng.standard_normal(n))
        b = np.cumsum(rng.standard_normal(n))
        X = np.column_stack([np.ones(n), a])
        coef, se, e = ols(b, X)
        ts.append(coef[1] / se[1])
        r2.append(1 - e @ e / np.sum((b - b.mean()) ** 2))
    ts = np.array(ts)
    share = float(np.mean(np.abs(ts) > 1.96))
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    ax[0].plot(x, color=B, label='$x_t$')
    ax[0].plot(y, color=R, label='$y_t$')
    ax[0].set_title('Two independent random walks')
    ax[0].set_xlabel('Period $t$')
    ax[1].hist(ts, bins=60, density=True, color=A, label=f'$t$ statistic of $\\hat b$, {reps} pairs')
    g = np.linspace(-6, 6, 200)
    ax[1].plot(g, np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi), color=G, label='$N(0, 1)$, the usual reference')
    for v in (-1.96, 1.96):
        ax[1].axvline(v, color=R, ls='--', lw=0.8)
    ax[1].set_xlim(-30, 30)
    ax[1].set_title(f'Regression $y_t = a + b\\,x_t + u_t$: $|t| > 1.96$ in {100 * share:.0f}% of pairs')
    ax[1].set_xlabel('$t$')
    fig.tight_layout(w_pad=0.6)
    st.fig_legend_bottom(fig, ncol=4)
    save(fig, 'spurious')
    return dict(share_reject=share, median_R2=float(np.median(r2)))


# =============================================================================
# 3. a cointegrated pair and its equilibrium error
# =============================================================================
def fig_coint(n=300, seed=5):
    rng = np.random.default_rng(seed)
    x = 10 + np.cumsum(rng.standard_normal(n))
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.7 * u[t - 1] + rng.standard_normal()
    y = 2 + 0.8 * x + u
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    ax[0].plot(x, color=B, label='$x_t$, $I(1)$')
    ax[0].plot(y, color=R, label='$y_t = 2 + 0.8\\,x_t + u_t$')
    ax[0].set_title('Levels: both wander')
    ax[0].set_xlabel('Period $t$')
    ax[1].plot(y - 2 - 0.8 * x, color=G, label='$u_t = y_t - 2 - 0.8\\,x_t$')
    ax[1].axhline(0, color=st.DarkText, lw=0.5, ls=':')
    ax[1].set_title('Equilibrium error: stationary')
    ax[1].set_xlabel('Period $t$')
    fig.tight_layout(w_pad=0.6)
    st.fig_legend_bottom(fig, ncol=3)
    save(fig, 'coint')
    return dict(sd_u=float(u.std()), range_x=[float(x.min()), float(x.max())])


# =============================================================================
# 4. null distributions: Dickey-Fuller for one series against Engle-Granger for two
# =============================================================================
def fig_eg_dist(n=200, reps=4000, seed=7):
    rng = np.random.default_rng(seed)
    df, eg = [], []
    for _ in range(reps):
        w = np.cumsum(rng.standard_normal(n))
        df.append(df_tau(w, const=True))
        a = np.cumsum(rng.standard_normal(n))
        b = np.cumsum(rng.standard_normal(n))
        _, _, e = ols(b, np.column_stack([np.ones(n), a]))
        eg.append(df_tau(e, const=False))
    df, eg = np.array(df), np.array(eg)
    q_df, q_eg = np.quantile(df, 0.05), np.quantile(eg, 0.05)
    fig, ax = plt.subplots(figsize=FULL)
    bins = np.linspace(-6, 2, 70)
    ax.hist(df, bins=bins, density=True, histtype='step', color=B, lw=1.3, label='Dickey–Fuller $\\tau$, one random walk')
    ax.hist(eg, bins=bins, density=True, histtype='step', color=R, lw=1.3, label='Engle–Granger $\\tau$, residuals of two random walks')
    ax.axvline(q_df, color=B, ls='--', lw=0.9, label=f'5% quantile ${q_df:.2f}$')
    ax.axvline(q_eg, color=R, ls='--', lw=0.9, label=f'5% quantile ${q_eg:.2f}$')
    ax.set_xlabel('$\\tau$ under $H_0$ (simulated, $T = 200$)')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=2)
    save(fig, 'eg_dist')
    return dict(q5_df=float(q_df), q5_eg=float(q_eg))


# =============================================================================
# 5. error correction: how fast a gap closes
# =============================================================================
def fig_halflife(H=25):
    h = np.arange(H + 1)
    fig, ax = plt.subplots(figsize=HALF)
    out = {}
    for g, c in zip((-0.1, -0.3, -0.6), (B, G, R)):
        ax.plot(h, (1 + g) ** h, color=c, marker='o', ms=1.8, label=f'$\\gamma = {g}$')
        hl = np.log(0.5) / np.log(1 + g)
        out[g] = float(hl)
        ax.plot(hl, 0.5, 'o', color=c, ms=4, mfc='none')
    ax.axhline(0.5, color=A, ls=':', lw=0.9)
    ax.set_xlabel('Periods after the gap $h$')
    ax.set_ylabel('Remaining gap $(1 + \\gamma)^h$')
    ax.set_title('Circles: half-life')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=3)
    save(fig, 'halflife')
    return {f'half_life_{k}': v for k, v in out.items()}


# =============================================================================
# 6. a VECM in which only one variable adjusts
# =============================================================================
def fig_vecm(n=300, seed=11, a1=-0.2, a2=0.0):
    rng = np.random.default_rng(seed)
    y = np.zeros((n, 2))
    y[0] = [5, 5]
    for t in range(1, n):
        z = y[t - 1, 0] - y[t - 1, 1]
        y[t, 0] = y[t - 1, 0] + a1 * z + 0.6 * rng.standard_normal()
        y[t, 1] = y[t - 1, 1] + a2 * z + 0.6 * rng.standard_normal()
    fig, ax = plt.subplots(1, 2, figsize=FULL)
    ax[0].plot(y[:, 1], color=B, label=f'$y_{{2t}}$, $\\alpha_2 = {a2:g}$ (does not adjust)')
    ax[0].plot(y[:, 0], color=R, label=f'$y_{{1t}}$, $\\alpha_1 = {{-}}{abs(a1):g}$ (adjusts)')
    ax[0].set_title('Levels')
    ax[0].set_xlabel('Period $t$')
    ax[1].plot(y[:, 0] - y[:, 1], color=G, label=r'$z_t = \boldsymbol{\beta}^\top\mathbf{y}_t = y_{1t} - y_{2t}$')
    ax[1].axhline(0, color=st.DarkText, lw=0.5, ls=':')
    ax[1].set_title('Equilibrium error')
    ax[1].set_xlabel('Period $t$')
    fig.tight_layout(w_pad=0.6)
    st.fig_legend_bottom(fig, ncol=3)
    save(fig, 'vecm')
    z = y[:, 0] - y[:, 1]
    return dict(sd_z=float(z.std()), rho_z=float(1 + a1 - a2))


# =============================================================================
# 7. the z-score rule of pairs trading
# =============================================================================
def fig_pairs(n=500, seed=13):
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.95 * s[t - 1] + rng.standard_normal()
    z = (s - s.mean()) / s.std()
    pos = np.zeros(n)
    for t in range(1, n):
        if pos[t - 1] == 0:
            pos[t] = -1 if z[t] > 2 else (1 if z[t] < -2 else 0)
        elif pos[t - 1] == -1:
            pos[t] = 0 if z[t] <= 0 else -1
        else:
            pos[t] = 0 if z[t] >= 0 else 1
    fig, ax = plt.subplots(figsize=FULL)
    ax.fill_between(np.arange(n), -3.5, 3.5, where=pos == -1, color=st.Crimson, alpha=0.18, lw=0, label='short the spread')
    ax.fill_between(np.arange(n), -3.5, 3.5, where=pos == 1, color=BAND, lw=0, label='long the spread')
    ax.plot(z, color=B, label='$z_t$')
    for v, c in ((2, R), (-2, R), (0, A)):
        ax.axhline(v, color=c, ls='--' if v else ':', lw=0.9)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('Day $t$')
    ax.set_ylabel('$z_t$')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=3)
    save(fig, 'pairs')
    trades = int(np.sum((pos != 0) & (np.roll(pos, 1) == 0)))
    return dict(trades=trades, days_in_market=float(np.mean(pos != 0)))


if __name__ == '__main__':
    for f in (fig_unit_root, fig_spurious, fig_coint, fig_eg_dist, fig_halflife, fig_vecm, fig_pairs):
        print(f.__name__, f())
