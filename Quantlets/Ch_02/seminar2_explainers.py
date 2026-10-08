"""
seminar2_explainers.py -- explanatory (primer) charts for Seminar 2 (TSA): ARMA models
=====================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 2, which takes
place BEFORE Lecture 2. All charts use SIMULATED data or theoretical formulas only (fixed seeds): AR(1) paths, the
AR(2) stationarity triangle, impulse responses and half-lives, invertibility, the ACF and PACF of AR, MA and ARMA
models, information criteria, residual diagnostics, heavy tails and the Jarque--Bera test, AR(1) forecasts with
intervals and the expanding-window forecast evaluation. They contain no exercise answers.

Output: charts/ch2_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).
The charts are drawn at their size on the slide, so the text is 7.5-8.5 pt on the slide.

Run:  python3 Quantlets/Ch_02/seminar2_explainers.py

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
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

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


def save(fig, name):
    st.check_no_grey(fig)
    st.save_fig(name)


def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    d = np.sum(x ** 2)
    return np.array([np.sum(x[k:] * x[:len(x) - k]) / d for k in range(nlags + 1)])


def arma(ar=(), ma=()):
    return ArmaProcess(np.r_[1, -np.asarray(ar, float)], np.r_[1, np.asarray(ma, float)])


def mm(s):
    return s.replace('-', '−')


# =============================================================================
# 1. AR(1) paths for three values of phi
# =============================================================================
def fig_ar1_paths(seed=6, T=150):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T + 100)
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.6), sharey=True)
    for ax, phi, col in zip(axes, (0.9, 0.3, -0.6), (BLUE, GREEN, RED)):
        x = np.zeros(T + 100)
        for t in range(1, T + 100):
            x[t] = phi * x[t - 1] + e[t]
        x = x[100:]
        ax.plot(np.arange(1, T + 1), x, color=col, lw=0.8)
        ax.axhline(0, color=NAVY, lw=0.5, ls=':')
        ax.set_title(mm(f'$\\phi$ = {phi}: $\\hat\\rho(1)$ = {acf(x, 1)[1]:.2f}'), loc='left')
        ax.set_xlabel('$t$')
    axes[0].set_ylabel('$X_t$')
    fig.tight_layout(w_pad=0.6)
    save(fig, 'ch2_sem_primer_ar1_paths')


# =============================================================================
# 2. impulse responses psi_j and the half-life
# =============================================================================
def fig_psi(J=12):
    j = np.arange(0, J + 1)
    fig, ax = plt.subplots(figsize=HALF)
    cases = [(arma([0.8]), BLUE, 'AR(1), $\\phi = 0.8$', 0.8), (arma([0.3]), GREEN, 'AR(1), $\\phi = 0.3$', 0.3),
             (arma([0.8], [-0.5]), AMBER, 'ARMA(1,1), $\\phi = 0.8$, $\\theta = -0.5$', None)]
    for proc, col, lab, phi in cases:
        psi = proc.arma2ma(J + 1)
        ax.plot(j, psi, 'o-', color=col, ms=2.8, lw=1.0, label=mm(lab))
    hl = np.log(0.5) / np.log(0.8)
    ax.axhline(0.5, color=NAVY, lw=0.6, ls='--')
    ax.axvline(hl, color=BLUE, lw=0.6, ls=':')
    ax.text(hl + 0.3, 0.75, f'half-life {hl:.1f}', color=BLUE, fontsize=8)
    ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_xlabel('Periods after the shock $j$')
    ax.set_ylabel('$\\psi_j$')
    ax.set_title('Effect of a unit shock', loc='left')
    st.legend_outside_bottom(ax, ncol=1)
    fig.tight_layout()
    save(fig, 'ch2_sem_primer_psi')
    return dict(hl=hl)


# =============================================================================
# 3. the AR(2) stationarity triangle and two ACFs
# =============================================================================
def fig_triangle(K=16):
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 1.85), gridspec_kw=dict(width_ratios=[1.45, 1, 1]))
    ax = axes[0]
    tri = np.array([[-2, -1], [2, -1], [0, 1], [-2, -1]])
    ax.fill(tri[:, 0], tri[:, 1], color=BAND, alpha=0.9, lw=0, label='Stationary')
    ax.plot(tri[:, 0], tri[:, 1], color=NAVY, lw=0.8)
    p1 = np.linspace(-2, 2, 200)
    ax.plot(p1, -p1 ** 2 / 4, color=PURPLE, lw=1.0, ls='--', label='$\\phi_1^2 + 4\\phi_2 = 0$')
    pts = [((0.5, 0.3), BLUE, 'A', -0.3), ((1.0, -0.5), GREEN, 'B', -0.32), ((1.3, 0.3), RED, 'C', 0.12)]
    for (a, b), col, lab, dx in pts:
        ax.plot(a, b, 'o', color=col, ms=4.5)
        ax.text(a + dx, b + 0.07, lab, color=col, fontsize=8.5, fontweight='bold')
    ax.text(0, -0.75, 'complex roots', ha='center', color=PURPLE, fontsize=7.5)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel('$\\phi_1$')
    ax.set_ylabel('$\\phi_2$')
    ax.set_title('AR(2) region', loc='left')
    k = np.arange(1, K + 1)
    for ax, (ar, col, lab) in zip(axes[1:], (((0.5, 0.3), BLUE, 'A: real roots'), ((1.0, -0.5), GREEN, 'B: complex roots'))):
        r = arma(ar).acf(K + 1)[1:]
        ax.vlines(k, 0, r, color=col, lw=1.8)
        ax.axhline(0, color=NAVY, lw=0.5)
        ax.set_ylim(-0.4, 1.0)
        ax.set_title(lab, loc='left')
        ax.set_xlabel('Lag $h$')
    axes[1].set_ylabel('$\\rho(h)$')
    fig.tight_layout(w_pad=0.6)
    st.fig_legend_bottom(fig, ncol=2)
    save(fig, 'ch2_sem_primer_triangle')
    rc = np.roots([-0.3, -0.5, 1])
    return dict(rootsA=np.abs(np.roots([-0.3, -0.5, 1])).round(2).tolist(), rootsB=np.abs(np.roots([0.5, -1.0, 1])).round(3).tolist(),
                rootsC=np.abs(np.roots([-0.3, -1.3, 1])).round(3).tolist())


# =============================================================================
# 4. invertibility: two MA(1) with the same ACF, their pi weights
# =============================================================================
def fig_invertible(J=8):
    j = np.arange(0, J + 1)
    fig, ax = plt.subplots(figsize=HALF)
    for th, col, lab in ((0.4, BLUE, '$\\theta = 0.4$: invertible'), (2.5, RED, '$\\theta = 2.5$: not invertible')):
        pi = (-th) ** j
        ax.plot(j, np.abs(pi), 'o-', color=col, ms=2.8, label=lab)
    ax.set_yscale('log')
    ax.set_xlabel('$j$')
    ax.set_ylabel('$|\\pi_j| = |\\theta|^j$ (log scale)')
    ax.set_title('Same $\\rho(1) = 0.345$; weights $\\pi_j = (-\\theta)^j$', loc='left')
    st.legend_outside_bottom(ax, ncol=1)
    fig.tight_layout()
    save(fig, 'ch2_sem_primer_invertible')
    return dict(r_04=0.4 / 1.16, r_25=2.5 / (1 + 6.25))


# =============================================================================
# 5. identification: theoretical ACF and PACF
# =============================================================================
def fig_identification(K=10):
    k = np.arange(1, K + 1)
    cases = [(arma([0.5, 0.3]), BLUE, 'AR(2)'), (arma([], [0.6]), GREEN, 'MA(1)'), (arma([0.7], [0.4]), AMBER, 'ARMA(1,1)')]
    fig, axes = plt.subplots(2, 3, figsize=(5.6, 2.25), sharex=True, sharey=True)
    for c, (proc, col, lab) in enumerate(cases):
        a = proc.acf(K + 1)[1:]
        p = proc.pacf(K + 1)[1:]
        for r, (vals, nm) in enumerate(((a, 'ACF'), (p, 'PACF'))):
            ax = axes[r, c]
            ax.vlines(k, 0, vals, color=col, lw=1.8)
            ax.axhline(0, color=NAVY, lw=0.5)
            ax.set_ylim(-0.5, 1.0)
            ax.set_title(f'{lab}: {nm}', loc='left')
        axes[1, c].set_xlabel('Lag $h$')
        axes[1, c].set_xticks([1, 5, 10])
    fig.tight_layout(h_pad=0.4, w_pad=0.5)
    save(fig, 'ch2_sem_primer_identification')


# =============================================================================
# 6. information criteria for AR(p), p = 0..6, on a simulated AR(2)
# =============================================================================
def fig_ic(seed=3, T=200):
    rng = np.random.default_rng(seed)
    x = arma([0.5, 0.3]).generate_sample(T, distrvs=rng.standard_normal, burnin=200)
    ps = np.arange(0, 7)
    m2, aic, bic = [], [], []
    for p in ps:
        res = ARIMA(x, order=(p, 0, 0), trend='c').fit()
        k = p + 2
        m2.append(-2 * res.llf)
        aic.append(-2 * res.llf + 2 * k)
        bic.append(-2 * res.llf + k * np.log(T))
    m2, aic, bic = map(np.array, (m2, aic, bic))
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(ps[1:], m2[1:], 'o-', color=NAVY, ms=3, label='$-2\\ln L$ (misfit)')
    ax.plot(ps[1:], aic[1:], 's-', color=BLUE, ms=3, label='AIC $= -2\\ln L + 2k$')
    ax.plot(ps[1:], bic[1:], '^-', color=RED, ms=3, label='BIC $= -2\\ln L + k\\ln T$')
    for v, col in ((aic, BLUE), (bic, RED)):
        i = int(np.argmin(v))
        ax.plot(ps[i], v[i], 'o', ms=8, mfc='none', mec=col, mew=1.3)
    ax.set_xticks(ps[1:])
    ax.set_xlabel('AR order $p$')
    ax.set_ylabel('Value')
    ax.set_title(f'Simulated AR(2), $T$ = {T}', loc='left')
    st.legend_outside_bottom(ax, ncol=1)
    fig.tight_layout()
    save(fig, 'ch2_sem_primer_ic')
    return dict(m2=m2.round(1).tolist(), aic=aic.round(1).tolist(), bic=bic.round(1).tolist(),
                aic_p=int(np.argmin(aic)), bic_p=int(np.argmin(bic)))


# =============================================================================
# 7. residual diagnostics: AR(1) and AR(2) fitted to an AR(2)
# =============================================================================
def fig_residuals(seed=3, T=200, K=12):
    rng = np.random.default_rng(seed)
    x = arma([0.5, 0.3]).generate_sample(T, distrvs=rng.standard_normal, burnin=200)
    k = np.arange(1, K + 1)
    b = 1.96 / np.sqrt(T)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.6), sharey=True)
    out = {}
    for ax, p, col in ((axes[0], 1, RED), (axes[1], 2, BLUE)):
        res = ARIMA(x, order=(p, 0, 0), trend='c').fit()
        e = res.resid[p:]
        r = acf(e, K)
        lb = acorr_ljungbox(e, lags=[8], model_df=p)
        q, pv = float(lb['lb_stat'].iloc[0]), float(lb['lb_pvalue'].iloc[0])
        out[p] = (q, pv)
        ax.axhspan(-b, b, color=BAND, alpha=0.9, lw=0)
        ax.vlines(k, 0, r[1:], color=col, lw=1.8)
        ax.axhline(0, color=NAVY, lw=0.5)
        ax.set_title(f'AR({p}) residuals: $Q^*(8)$ = {q:.1f}, ' + ('p < 0.001' if pv < 0.001 else f'p = {pv:.2f}'), loc='left')
        ax.set_xticks([1, 4, 8, 12])
        ax.set_xlabel('Lag $h$')
        ax.set_ylim(-0.25, 0.25)
    axes[0].set_ylabel('$\\hat\\rho_{\\hat\\varepsilon}(h)$')
    axes[0].fill_between([], [], color=BAND, label=f'Band $\\pm 1.96/\\sqrt{{T}}$, $T$ = {T}')
    fig.tight_layout(w_pad=0.8)
    st.fig_legend_bottom(fig, ncol=1)
    save(fig, 'ch2_sem_primer_residuals')
    return out


# =============================================================================
# 8. heavy tails: residual histogram against the Normal density, skewness, kurtosis, JB
# =============================================================================
def fig_jb(seed=5, T=1000, nu=4):
    rng = np.random.default_rng(seed)
    e = stats.t(nu).rvs(T, random_state=rng)
    e = e / e.std()
    z = (e - e.mean()) / e.std()
    S, K = np.mean(z ** 3), np.mean(z ** 4)
    JB = T / 6 * (S ** 2 + (K - 3) ** 2 / 4)
    x = np.linspace(-6, 6, 400)
    fig, ax = plt.subplots(figsize=HALF)
    ax.hist(z, bins=60, density=True, color=BAND, edgecolor=BLUE, lw=0.3, label='Simulated residuals')
    ax.plot(x, stats.norm.pdf(x), color=RED, lw=1.3, label='Normal density')
    ax.set_xlim(-6, 6)
    ax.set_xlabel('Standardised residual $z_t$')
    ax.set_ylabel('Density')
    ax.set_title(mm(f'$S$ = {S:.2f}, $K$ = {K:.1f}, JB = {JB:,.0f}'), loc='left')
    st.legend_outside_bottom(ax, ncol=1)
    fig.tight_layout()
    save(fig, 'ch2_sem_primer_jb')
    return dict(S=S, K=K, JB=JB, T=T)


# =============================================================================
# 9. AR(1) forecasts with 95% intervals
# =============================================================================
def fig_forecast(seed=1, T=40, H=16, mu=10.0, phi=0.8, sigma=1.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(T)
    x[0] = mu
    for t in range(1, T):
        x[t] = mu + phi * (x[t - 1] - mu) + sigma * rng.standard_normal()
    x[-1] = mu + 3.0
    h = np.arange(1, H + 1)
    f = mu + phi ** h * (x[-1] - mu)
    sd = sigma * np.sqrt((1 - phi ** (2 * h)) / (1 - phi ** 2))
    sd_inf = sigma / np.sqrt(1 - phi ** 2)
    fig, ax = plt.subplots(figsize=(5.6, 1.6))
    t = np.arange(1, T + 1)
    ax.plot(t, x, color=BLUE, lw=1.0, marker='o', ms=1.8, label='Observed $X_t$')
    ax.fill_between(T + h, f - 1.96 * sd, f + 1.96 * sd, color=BAND, alpha=0.9, lw=0, label='95% interval $\\hat X_{T+h} \\pm 1.96\\,\\sigma_h$')
    ax.plot(T + h, f, color=RED, lw=1.4, marker='o', ms=2, label='Forecast $\\hat X_{T+h}$')
    ax.axhline(mu, color=GREEN, lw=0.9, ls='--', label='Mean $\\mu$')
    ax.axhline(mu + 1.96 * sd_inf, color=NAVY, lw=0.6, ls=':')
    ax.axhline(mu - 1.96 * sd_inf, color=NAVY, lw=0.6, ls=':', label='$\\mu \\pm 1.96\\sqrt{\\gamma(0)}$')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$X_t$')
    fig.tight_layout()
    st.fig_legend_bottom(fig, ncol=3)
    save(fig, 'ch2_sem_primer_forecast')
    return dict(last=x[-1], f1=f[0], f2=f[1], sd1=sd[0], sd2=sd[1], sdinf=sd_inf, w_inf=1.96 * sd_inf)


# =============================================================================
# 10. expanding-window (rolling-origin) forecast evaluation
# =============================================================================
def fig_origins(n_or=6, T0=12, H=3, N=24):
    fig, ax = plt.subplots(figsize=HALF)
    for i in range(n_or):
        y = n_or - i
        o = T0 + 2 * i
        ax.barh(y, o, left=0, height=0.55, color=BLUE, label='Estimation sample' if i == 0 else '_')
        ax.barh(y, 1, left=o + H - 1, height=0.55, color=RED, label='Value forecast $h$ steps ahead' if i == 0 else '_')
        ax.text(o - 0.3, y, f'origin {i + 1}', ha='right', va='center', color='white', fontsize=7.5)
    ax.set_xlim(0, N + 1)
    ax.set_yticks([])
    ax.set_xlabel('Time')
    ax.set_title(f'Expanding window, $h$ = {H}', loc='left')
    ax.spines['left'].set_visible(False)
    st.legend_outside_bottom(ax, ncol=1)
    fig.tight_layout()
    save(fig, 'ch2_sem_primer_origins')


if __name__ == '__main__':
    fig_ar1_paths()
    print('psi      ', fig_psi())
    print('triangle ', fig_triangle())
    print('invert   ', fig_invertible())
    fig_identification()
    print('ic       ', fig_ic())
    print('resid    ', fig_residuals())
    print('jb       ', fig_jb())
    print('forecast ', {k: round(float(v), 3) for k, v in fig_forecast().items()})
    fig_origins()
