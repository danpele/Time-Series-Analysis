"""
seminar6_explainers.py -- explanatory (primer) charts for Seminar 6 (TSA): VAR models and Granger causality
==========================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 6, which takes
place BEFORE Lecture 6. All charts use SIMULATED data only (fixed seeds): eigenvalues and stability, forecasts with
intervals, information criteria, cross-correlations, the F distribution of the Granger test,
orthogonalised impulse responses under two orderings and a forecast error variance decomposition.
They contain no exercise answers.

The figures are drawn at the size of their box on the slide (full width: 5.6 x 1.5 in; one column: 2.8 x 2.0 in),
so that 1 pt in the figure is about 1 pt on the slide (text >= 7 pt).

Output: charts/ch6_sem_primer_*.pdf and .png (transparent background, legend below the plot).
Run:    python3 Quantlets/Ch_06/seminar6_explainers.py
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

A1 = np.array([[0.5, 0.3], [0.1, 0.6]])
C1 = np.array([0.4, 0.2])
SIG = np.array([[1.0, 0.5], [0.5, 1.0]])


def save(fig, name):
    st.check_no_grey(fig)
    st.save_fig(f'ch6_sem_primer_{name}')


def simulate_var1(A, c, Sig, n, seed, burn=200):
    rng = np.random.default_rng(seed)
    P = np.linalg.cholesky(Sig)
    y = np.zeros((n + burn, 2))
    y[0] = np.linalg.solve(np.eye(2) - A, c)
    for t in range(1, n + burn):
        y[t] = c + A @ y[t - 1] + P @ rng.standard_normal(2)
    return y[burn:]


# =============================================================================
# 2. eigenvalues and stability
# =============================================================================
def fig_stability(H=20):
    mats = [('stable, real roots', np.array([[0.7, 0.1], [0.2, 0.5]]), B),
            ('stable, complex roots', np.array([[0.6, -0.5], [0.5, 0.6]]), G),
            ('explosive', np.array([[1.0, 0.2], [0.1, 0.9]]), R)]
    fig, ax = plt.subplots(1, 2, figsize=FULL, gridspec_kw=dict(width_ratios=[1, 2.2]))
    th = np.linspace(0, 2 * np.pi, 300)
    ax[0].plot(np.cos(th), np.sin(th), color=st.DarkText, lw=0.8)
    out = {}
    for lab, M, c in mats:
        ev = np.linalg.eigvals(M)
        ax[0].plot(ev.real, ev.imag, 'o', color=c, ms=4)
        out[lab] = np.abs(ev).round(3).tolist()
        resp = [np.linalg.matrix_power(M, h)[0, 0] for h in range(H + 1)]
        ax[1].plot(range(H + 1), resp, color=c, marker='o', ms=1.8, label=lab)
    ax[0].set_aspect('equal')
    ax[0].set_xlim(-1.3, 1.3)
    ax[0].set_ylim(-1.3, 1.3)
    ax[0].axhline(0, color=st.DarkText, lw=0.4, ls=':')
    ax[0].axvline(0, color=st.DarkText, lw=0.4, ls=':')
    ax[0].set_title('Eigenvalues')
    ax[0].set_xlabel('Real part')
    ax[0].set_ylabel('Imaginary part')
    ax[1].axhline(0, color=st.DarkText, lw=0.5)
    ax[1].set_ylim(-0.6, 3)
    ax[1].set_title('Response of $y_1$ to a unit shock in $\\varepsilon_1$, $(\\mathbf{A}^h)_{11}$', loc='right')
    ax[1].set_xlabel('Horizon $h$')
    fig.tight_layout(w_pad=0.6)
    st.fig_legend_bottom(fig, ncol=3)
    save(fig, 'stability')
    return out


# =============================================================================
# 3. forecasts and their 95% intervals
# =============================================================================
def fig_forecast(n=60, H=16, seed=9):
    y = simulate_var1(A1, C1, SIG, n, seed)
    mu = np.linalg.solve(np.eye(2) - A1, C1)
    f, V = [y[-1]], []
    S = np.zeros((2, 2))
    Phi = np.eye(2)
    for h in range(1, H + 1):
        f.append(C1 + A1 @ f[-1])
        S = S + Phi @ SIG @ Phi.T
        Phi = A1 @ Phi
        V.append(np.diag(S).copy())
    f = np.array(f[1:])
    V = np.array(V)
    t = np.arange(1, n + 1)
    tf = np.arange(n + 1, n + H + 1)
    fig, ax = plt.subplots(1, 2, figsize=FULL, sharey=True)
    for j, (a, c) in enumerate(zip(ax, (B, R))):
        a.plot(t, y[:, j], color=c, label='Observed' if j == 0 else '_')
        a.fill_between(tf, f[:, j] - 1.96 * np.sqrt(V[:, j]), f[:, j] + 1.96 * np.sqrt(V[:, j]), color=BAND, lw=0,
                       label='95% interval' if j == 0 else '_')
        a.plot(tf, f[:, j], color=A, lw=1.3, label='Forecast $\\hat y_{T+h}$' if j == 0 else '_')
        a.axhline(mu[j], color=G, ls='--', lw=0.9, label='Mean $\\mu$' if j == 0 else '_')
        a.set_title(f'$y_{{{j + 1}}}$')
        a.set_xlabel('Period $t$')
    fig.tight_layout(w_pad=0.6)
    st.fig_legend_bottom(fig, ncol=4)
    save(fig, 'forecast')
    return dict(last=y[-1].round(2).tolist(), f1=f[0].round(2).tolist(), f16=f[-1].round(2).tolist(),
                sd1=np.sqrt(V[0]).round(2).tolist(), sd16=np.sqrt(V[-1]).round(2).tolist())


# =============================================================================
# 4. information criteria against the lag order
# =============================================================================
def fig_ic(n=120, seed=4, pmax=8):
    rng = np.random.default_rng(seed)
    A_1 = np.array([[0.4, 0.2], [0.1, 0.3]])
    A_2 = np.array([[0.25, 0.0], [0.1, 0.2]])
    P = np.linalg.cholesky(SIG)
    y = np.zeros((n + 200 + pmax, 2))
    for t in range(2, len(y)):
        y[t] = A_1 @ y[t - 1] + A_2 @ y[t - 2] + P @ rng.standard_normal(2)
    y = y[200:]
    K = 2
    T = len(y) - pmax
    res = {}
    for p in range(pmax + 1):
        Y = y[pmax:]
        X = np.column_stack([np.ones(T)] + [y[pmax - j:len(y) - j] for j in range(1, p + 1)])
        Bh = np.linalg.lstsq(X, Y, rcond=None)[0]
        E = Y - X @ Bh
        res[p] = np.log(np.linalg.det(E.T @ E / T))
    p_ = np.arange(pmax + 1)
    ld = np.array([res[p] for p in p_])
    crit = {'AIC': ld + 2 * p_ * K ** 2 / T, 'BIC': ld + np.log(T) * p_ * K ** 2 / T,
            'HQ': ld + 2 * np.log(np.log(T)) * p_ * K ** 2 / T}
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(p_, ld, color=st.DarkText, ls=':', marker='o', ms=2, label=r'$\ln\det\tilde{\Sigma}(p)$')
    out = {}
    for (k, v), c in zip(crit.items(), (B, G, R)):
        ax.plot(p_, v, color=c, marker='o', ms=2, label=k)
        i = int(np.argmin(v))
        ax.plot(p_[i], v[i], 'o', color=c, ms=5, mfc='none')
        out[k] = i
    ax.set_xlabel('Lag order $p$')
    ax.set_title(f'$K = 2$, $T = {T}$; circles: minima')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=2)
    save(fig, 'ic')
    out['T'] = T
    return out


# =============================================================================
# 5. cross-correlations: x leads y by one period
# =============================================================================
def fig_ccf(n=500, seed=6, kmax=5):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n + 1)
    y = 0.5 * x[:-1] + rng.standard_normal(n)
    x = x[1:]
    ks = np.arange(-kmax, kmax + 1)

    def cc(k):
        if k >= 0:
            return np.corrcoef(y[k:], x[:n - k])[0, 1]
        return np.corrcoef(y[:n + k], x[-k:])[0, 1]
    v = np.array([cc(k) for k in ks])
    band = 1.96 / np.sqrt(n)
    fig, ax = plt.subplots(figsize=HALF)
    ax.vlines(ks, 0, v, color=B, lw=2)
    ax.fill_between([-kmax - 0.5, kmax + 0.5], -band, band, color=BAND, lw=0, label=r'$\pm 1.96/\sqrt{T}$')
    ax.axhline(0, color=st.DarkText, lw=0.5)
    ax.set_xlabel('$k$')
    ax.set_title(r'$\mathrm{Corr}(y_t, x_{t-k})$; $y_t = 0.5\,x_{t-1} + e_t$')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=1)
    save(fig, 'ccf')
    return dict(band=float(band), c0=float(v[kmax]), c1=float(v[kmax + 1]), cm1=float(v[kmax - 1]))


# =============================================================================
# 6. the F distribution of the Granger test
# =============================================================================
def fig_f(q=2, df2=100):
    x = np.linspace(0, 7, 500)
    f = stats.f.pdf(x, q, df2)
    cv = stats.f.ppf(0.95, q, df2)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, f, color=B, label=f'$F({q}, {df2})$ density')
    xx = x[x >= cv]
    ax.fill_between(xx, 0, stats.f.pdf(xx, q, df2), color=R, alpha=0.4, lw=0, label='5% rejection region')
    ax.axvline(cv, color=R, ls='--', lw=0.9)
    ax.annotate(f'{cv:.2f}', (cv, 0.5), xytext=(4, 0), textcoords='offset points', color=R, fontsize=7.5)
    ax.set_xlabel('$F$')
    ax.set_title('Reference distribution under $H_0$')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=1)
    save(fig, 'f')
    return dict(cv=float(cv))


# =============================================================================
# 7. orthogonalised impulse responses: the ordering matters
# =============================================================================
def irfs(A, Sig, H, order=(0, 1)):
    o = list(order)
    P = np.linalg.cholesky(Sig[np.ix_(o, o)])
    Pfull = np.zeros((2, 2))
    Pfull[np.ix_(o, o)] = P                       # back to the original variable order
    return np.array([np.linalg.matrix_power(A, h) @ Pfull for h in range(H + 1)])


def fig_irf(H=12):
    A = np.array([[0.5, 0.1], [0.3, 0.4]])
    th12 = irfs(A, SIG, H, (0, 1))
    th21 = irfs(A, SIG, H, (1, 0))
    gen = np.array([np.linalg.matrix_power(A, h) @ SIG[:, 0] / np.sqrt(SIG[0, 0]) for h in range(H + 1)])
    h = np.arange(H + 1)
    fig, ax = plt.subplots(1, 2, figsize=FULL, sharey=True)
    ax[0].plot(h, th12[:, 1, 0], color=B, marker='o', ms=2, label='Cholesky, order $(y_1, y_2)$')
    ax[0].plot(h, th21[:, 1, 0], color=R, marker='o', ms=2, label='Cholesky, order $(y_2, y_1)$')
    ax[0].plot(h, gen[:, 1], color=G, ls='--', label='Generalised')
    ax[0].set_title('Response of $y_2$ to a shock in $y_1$')
    ax[1].plot(h, th12[:, 0, 1], color=B, marker='o', ms=2)
    ax[1].plot(h, th21[:, 0, 1], color=R, marker='o', ms=2)
    ax[1].set_title('Response of $y_1$ to a shock in $y_2$')
    for a in ax:
        a.axhline(0, color=st.DarkText, lw=0.5)
        a.set_xlabel('Horizon $h$')
    fig.tight_layout(w_pad=0.6)
    st.fig_legend_bottom(fig, ncol=3)
    save(fig, 'irf')
    return dict(y2_on_y1_order12_h0=float(th12[0, 1, 0]), y2_on_y1_order21_h0=float(th21[0, 1, 0]),
                y1_on_y2_order12_h0=float(th12[0, 0, 1]), y1_on_y2_order21_h0=float(th21[0, 0, 1]))


# =============================================================================
# 8. forecast error variance decomposition
# =============================================================================
def fig_fevd(H=12):
    A = np.array([[0.5, 0.1], [0.3, 0.4]])
    th = irfs(A, SIG, H, (0, 1))
    hs = np.arange(1, H + 1)
    sh = []
    for hh in hs:
        num = np.sum(th[:hh, 1, :] ** 2, axis=0)
        sh.append(num / num.sum())
    sh = np.array(sh) * 100
    fig, ax = plt.subplots(figsize=HALF)
    ax.stackplot(hs, sh[:, 0], sh[:, 1], colors=[B, BAND], labels=['shock $u_1$', 'shock $u_2$'])
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 4, 8, 12])
    ax.set_xlabel('Horizon $h$')
    ax.set_ylabel('Share (%)')
    ax.set_title('FEVD of $y_2$, order $(y_1, y_2)$')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=2)
    save(fig, 'fevd')
    return dict(share_u1_h1=float(sh[0, 0]), share_u1_h4=float(sh[3, 0]), share_u1_h12=float(sh[-1, 0]))


if __name__ == '__main__':
    for f in (fig_stability, fig_forecast, fig_ic, fig_ccf, fig_f, fig_irf, fig_fevd):
        print(f.__name__, f())
