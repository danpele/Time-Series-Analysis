"""
generate_all_charts.py -- charts and numbers of Chapter 2 (TSA): ARMA models
============================================================================
Course data (tsa_data.py), chart style (tsa_style.py), statsmodels for the ARMA fits, the ACF, the PACF and the
portmanteau tests. Every number on the slides comes from here.
  * AR models      -- AR(1) paths and ACF for several phi; the AR(2) stationarity triangle, the inverse roots in the
                      unit circle and the damped-wave ACF of complex roots;
  * MA models      -- MA(1) paths, ACF and PACF; rho(1) = theta/(1 + theta^2) and the invertibility problem;
  * ARMA           -- theoretical and sample ACF/PACF of AR(2), MA(2) and ARMA(1,1); psi (impulse-response) weights;
  * estimation     -- Monte Carlo of Yule-Walker, conditional least squares and Gaussian maximum likelihood;
  * selection      -- how often AIC and BIC pick the true AR order; Ljung-Box on residuals with m and m - p - q
                      degrees of freedom;
  * forecasting    -- AR(1) and MA(1) forecasts with intervals (mean reversion);
  * real data      -- Romanian annual GDP growth (Box-Jenkins case: identification, model table, diagnostics,
                      forecast); Romanian 12-month HICP inflation (AR(2) forecast); BET and EUR/RON daily returns;
                      the sunspot numbers and Yule's AR(2).
Output: charts/tsa_ch2_*.pdf/.png, Quantlets/Ch_02/ch2_numbers.json, ch2_gdp_models.csv
References: Huang and Petukhina (2022), Applied Time Series Analysis and Forecasting with Python, Ch. 3-4;
Hyndman and Athanasopoulos, Forecasting: Principles and Practice (3rd ed.), Ch. 9; Brockwell and Davis (2016),
Introduction to Time Series and Forecasting, 3rd ed., Ch. 3 and 5.
Run:  python3 Quantlets/Ch_02/generate_all_charts.py
Time Series Analysis - Daniel Traian PELE
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from tsa_data import log_returns, read_eurostat, load_statsmodels   # noqa: E402
import tsa_style as st                                               # noqa: E402
from statsmodels.tsa.stattools import acf, pacf                       # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox               # noqa: E402
from statsmodels.tsa.arima.model import ARIMA                         # noqa: E402
from statsmodels.tsa.arima_process import ArmaProcess                 # noqa: E402
from statsmodels.regression.linear_model import yule_walker           # noqa: E402

warnings.filterwarnings('ignore')
SEED = 2026
GDP_NSA = ('namq_10_gdp', 'Q.CLV10_MEUR.NSA.B1GQ.RO')     # chain-linked volumes (2010), million EUR, not adjusted
GDP_SCA = ('namq_10_gdp', 'Q.CLV10_MEUR.SCA.B1GQ.RO')     # the same, seasonally and calendar adjusted
HICP = ('prc_hicp_minr', 'M.I15.TOTAL.RO')                 # harmonised index of consumer prices, 2015 = 100
GDP_START = '2000-01-01'                                   # annual GDP growth: sample start
INFL_START = '2005-01-01'                                  # 12-month inflation: sample start
BAND_COL = st.IDAred                                       # +-1.96/sqrt(T) bands


# =============================================================================
# DATA AND HELPERS
# =============================================================================
def ro_gdp_growth(kind='yoy', start=GDP_START):
    """Romanian real GDP growth in %: 'yoy' = 100 (ln Y_t - ln Y_{t-4}) of the unadjusted series,
    'qoq' = 100 (ln Y_t - ln Y_{t-1}) of the seasonally adjusted series."""
    if kind == 'yoy':
        y = 100 * np.log(read_eurostat(*GDP_NSA)).diff(4)
    else:
        y = 100 * np.log(read_eurostat(*GDP_SCA)).diff()
    y = y.dropna().loc[start:]
    y.index.freq = None
    return y.rename(f'gdp_{kind}')


def ro_inflation(start=INFL_START):
    """Romanian 12-month HICP inflation in %: 100 (ln P_t - ln P_{t-12})."""
    p = read_eurostat(*HICP)
    return (100 * np.log(p).diff(12)).dropna().loc[start:].rename('inflation')


def sample_acf(x, nlags=20):
    """Sample ACF rho(1..nlags) with the divisor T (Chapter 1)."""
    return acf(np.asarray(x, float), nlags=nlags, fft=True)[1:]


def sample_pacf(x, nlags=20):
    """Sample PACF phi_hh, h = 1..nlags (Durbin-Levinson on the sample ACF)."""
    return pacf(np.asarray(x, float), nlags=nlags, method='ywm')[1:]


def ljung_box(x, m=10, df=None):
    """Ljung-Box Q*(m) with its p-value from chi2(df); df = m by default, m - p - q for ARMA residuals."""
    x = np.asarray(x, float)
    T = len(x)
    r = sample_acf(x, m)
    q = T * (T + 2) * np.sum(r ** 2 / (T - np.arange(1, m + 1)))
    df = m if df is None else df
    return {'lb': float(q), 'df': int(df), 'lb_p': float(stats.chi2.sf(q, df))}


def acf_bars(ax, r, T, color=st.MainBlue, theory=None, label='sample', band=True, theory_label='theoretical'):
    """Bars at lags 1..len(r), the +-1.96/sqrt(T) band and optional theoretical values (dots)."""
    lags = np.arange(1, len(r) + 1)
    ax.bar(lags, r, width=0.55, color=color, label=label)
    if theory is not None:
        ax.plot(lags, theory, 'o', ms=4, color=st.Orange, label=theory_label)
    if band:
        b = 1.96 / np.sqrt(T)
        ax.axhline(b, color=BAND_COL, ls='--', lw=0.9, label=r'$\pm 1.96/\sqrt{T}$')
        ax.axhline(-b, color=BAND_COL, ls='--', lw=0.9, label='_nolegend_')
    ax.axhline(0, color=st.DarkText, lw=0.6)
    ax.set_xlim(0.3, len(r) + 0.7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def simulate_arma(phi=(), theta=(), n=500, sigma=1.0, mu=0.0, burn=500, rng=None):
    """X_t - mu = sum phi_i (X_{t-i} - mu) + e_t + sum theta_j e_{t-j}, e_t i.i.d. N(0, sigma^2), after a burn-in."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    e = rng.normal(0, sigma, n + burn)
    x = np.zeros(n + burn)
    p, q = len(phi), len(theta)
    for t in range(n + burn):
        x[t] = e[t] + sum(phi[i] * x[t - 1 - i] for i in range(p) if t - 1 - i >= 0) \
            + sum(theta[j] * e[t - 1 - j] for j in range(q) if t - 1 - j >= 0)
    return mu + x[burn:]


def arma_process(phi=(), theta=()):
    """statsmodels ArmaProcess for X_t = phi_1 X_{t-1} + ... + e_t + theta_1 e_{t-1} + ... (signs as in the slides)."""
    return ArmaProcess(np.r_[1, -np.asarray(phi, float)], np.r_[1, np.asarray(theta, float)])


def theoretical_acf(phi=(), theta=(), nlags=20):
    return arma_process(phi, theta).acf(nlags + 1)[1:]


def theoretical_pacf(phi=(), theta=(), nlags=20):
    return arma_process(phi, theta).pacf(nlags + 1)[1:]


def psi_weights(phi=(), theta=(), n=20):
    """psi_0, ..., psi_{n-1} of the causal representation X_t = sum psi_j e_{t-j}:
    psi_j = theta_j + sum_{i=1}^{min(j,p)} phi_i psi_{j-i}, psi_0 = 1."""
    p, q = len(phi), len(theta)
    psi = np.zeros(n)
    psi[0] = 1.0
    for j in range(1, n):
        psi[j] = (theta[j - 1] if j <= q else 0.0) + sum(phi[i] * psi[j - 1 - i] for i in range(min(j, p)))
    return psi


def inverse_roots(phi):
    """Inverse roots of 1 - phi_1 z - ... - phi_p z^p (stationary iff all have modulus < 1)."""
    roots = np.roots(np.r_[-np.asarray(phi, float)[::-1], 1.0])
    return 1 / roots


def fit_arma(x, p, q, trend='c'):
    """Gaussian maximum likelihood (statsmodels state-space ARIMA)."""
    return ARIMA(np.asarray(x, float), order=(p, 0, q), trend=trend).fit()


def save(name, save_it=True):
    if save_it:
        st.check_no_grey(plt.gcf())
        st.save_fig(name)
    else:
        plt.show()


# =============================================================================
# 1. AR MODELS
# =============================================================================
def fig_ar1(n=200, nlags=15, save_it=True):
    """AR(1) paths and sample/theoretical ACF for phi = 0.9, 0.3 and -0.7 (same shocks)."""
    phis = [0.9, 0.3, -0.7]
    cols = [st.MainBlue, st.Forest, st.IDAred]
    rng = np.random.default_rng(SEED)
    e = rng.normal(size=n + 300)
    fig, ax = plt.subplots(2, 3, figsize=(9.33, 3.25))
    out = {}
    for j, (ph, c) in enumerate(zip(phis, cols)):
        x = np.zeros(n + 300)
        for t in range(1, n + 300):
            x[t] = ph * x[t - 1] + e[t]
        x = x[300:]
        ax[0, j].plot(x, color=c, lw=0.8)
        ax[0, j].axhline(0, color=st.DarkText, lw=0.6)
        ax[0, j].set_title(rf'AR(1), $\phi = {ph}$')
        ax[0, j].set_ylim(-7, 7)
        r = sample_acf(x, nlags)
        acf_bars(ax[1, j], r, n, color=c, theory=ph ** np.arange(1, nlags + 1),
                 label='sample ACF' if j == 0 else '_nolegend_',
                 theory_label=r'theory $\phi^h$' if j == 0 else '_nolegend_')
        ax[1, j].set_ylim(-1, 1)
        ax[1, j].set_xlabel('lag h')
        if j:
            ax[1, j].get_lines()[1].set_label('_nolegend_')
        out[str(ph)] = {'r1': float(r[0]), 'r2': float(r[1]), 'r5': float(r[4]), 'var': float(x.var()),
                        'var_th': 1 / (1 - ph ** 2)}
    st.fig_legend_bottom(fig, ncol=3, y=-0.01)
    plt.tight_layout()
    save('tsa_ch2_ar1', save_it)
    return out


def fig_ar2(nlags=30, save_it=True, sun_phi=None):
    """The AR(2) stationarity triangle (real and complex roots), the inverse roots in the unit circle and the ACF
    of three AR(2) processes."""
    ex = {'A': (0.5, 0.3), 'B': (1.0, -0.6), 'C': (0.6, 0.5)}
    if sun_phi is not None:
        ex['Sunspots'] = tuple(sun_phi)
    cols = {'A': st.MainBlue, 'B': st.IDAred, 'C': st.Purple, 'Sunspots': st.Amber}
    fig, ax = plt.subplots(1, 3, figsize=(8.49, 2.95))
    f1 = np.linspace(-2, 2, 300)
    ax[0].fill([-2, 0, 2], [-1, 1, -1], color=st.Teal, alpha=0.12, label='stationary region')
    ax[0].plot([-2, 0, 2, -2], [-1, 1, -1, -1], color=st.Teal, lw=1.2, label='_nolegend_')
    ax[0].plot(f1, -f1 ** 2 / 4, color=st.Forest, ls='--', lw=1.1, label=r'$\phi_1^2 + 4\phi_2 = 0$')
    for k, (a, b) in ex.items():
        ax[0].plot(a, b, 'o', ms=7, color=cols[k], label=k)
        ax[0].annotate(k, (a, b), xytext=(5, 4), textcoords='offset points', color=cols[k], fontsize=11)
    ax[0].text(-0.75, 0.12, 'real roots', ha='center', fontsize=10, color=st.DarkText)
    ax[0].text(0, -0.75, 'complex roots', ha='center', fontsize=10, color=st.DarkText)
    ax[0].set_xlim(-2.2, 2.2)
    ax[0].set_ylim(-1.2, 1.2)
    ax[0].set_xlabel(r'$\phi_1$')
    ax[0].set_ylabel(r'$\phi_2$')
    ax[0].set_title('Stationarity triangle')
    th = np.linspace(0, 2 * np.pi, 400)
    ax[1].plot(np.cos(th), np.sin(th), color=st.DarkText, lw=0.9, label='_nolegend_')
    ax[1].axhline(0, color=st.DarkText, lw=0.4)
    ax[1].axvline(0, color=st.DarkText, lw=0.4)
    out = {}
    for k, ph in ex.items():
        ir = inverse_roots(ph)
        ax[1].plot(ir.real, ir.imag, 'o', ms=7, color=cols[k], label='_nolegend_')
        out[k] = {'phi': list(ph), 'inv_roots_mod': [float(abs(z)) for z in ir],
                  'roots': [[float(z.real), float(z.imag)] for z in 1 / ir], 'stationary': bool(np.all(abs(ir) < 1))}
    ax[1].set_aspect('equal')
    ax[1].set_xlim(-1.35, 1.35)
    ax[1].set_ylim(-1.35, 1.35)
    ax[1].set_title('Inverse roots and the unit circle')
    for k in ['A', 'B']:
        ax[2].plot(np.arange(1, nlags + 1), theoretical_acf(ex[k], nlags=nlags), 'o-', ms=3, lw=1, color=cols[k],
                   label='_nolegend_')
    ax[2].axhline(0, color=st.DarkText, lw=0.6)
    ax[2].set_xlabel('lag h')
    ax[2].set_title('Theoretical ACF of A and B')
    st.fig_legend_bottom(fig, ncol=6, y=-0.02)
    plt.tight_layout()
    save('tsa_ch2_ar2', save_it)
    a1, a2 = ex['B']
    out['B']['period'] = float(2 * np.pi / np.arccos(a1 / (2 * np.sqrt(-a2))))
    out['B']['damp'] = float(np.sqrt(-a2))
    out['A']['rho1'] = float(theoretical_acf(ex['A'], nlags=2)[0])
    out['A']['rho2'] = float(theoretical_acf(ex['A'], nlags=2)[1])
    return out


# =============================================================================
# 2. MA MODELS
# =============================================================================
def fig_ma1(n=300, nlags=12, save_it=True):
    """MA(1) with theta = 0.8 and -0.8: paths, sample ACF (cuts off after lag 1) and sample PACF (decays)."""
    rng = np.random.default_rng(SEED + 1)
    e = rng.normal(size=n + 1)
    fig, ax = plt.subplots(2, 3, figsize=(9.33, 3.27))
    out = {}
    for i, (th, c) in enumerate([(0.8, st.MainBlue), (-0.8, st.IDAred)]):
        x = e[1:] + th * e[:-1]
        ax[i, 0].plot(x, color=c, lw=0.7)
        ax[i, 0].set_title(rf'MA(1), $\theta = {th}$')
        ax[i, 0].set_ylim(-5, 5)
        r, pc = sample_acf(x, nlags), sample_pacf(x, nlags)
        acf_bars(ax[i, 1], r, n, color=c, theory=theoretical_acf(theta=[th], nlags=nlags),
                 label='sample' if i == 0 else '_nolegend_', theory_label='theoretical' if i == 0 else '_nolegend_')
        acf_bars(ax[i, 2], pc, n, color=c, theory=theoretical_pacf(theta=[th], nlags=nlags), label='_nolegend_',
                 theory_label='_nolegend_')
        for a in ax[i, 1:]:
            a.set_ylim(-0.65, 0.65)
            if i == 1:
                a.get_lines()[1].set_label('_nolegend_')
        ax[i, 1].set_title('ACF')
        ax[i, 2].set_title('PACF')
        out[str(th)] = {'r1': float(r[0]), 'r2': float(r[1]), 'p1': float(pc[0]), 'p2': float(pc[1]),
                        'p3': float(pc[2]), 'rho1_th': th / (1 + th ** 2)}
    st.fig_legend_bottom(fig, ncol=3, y=-0.01)
    plt.tight_layout()
    save('tsa_ch2_ma1', save_it)
    return out


def fig_ma1_rho(save_it=True):
    """rho(1) = theta / (1 + theta^2) of an MA(1): maximum 0.5 at theta = 1; theta and 1/theta give the same rho(1)."""
    th = np.linspace(-4, 4, 801)
    r = th / (1 + th ** 2)
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    ax.plot(th, r, color=st.MainBlue, lw=1.6, label=r'$\rho(1) = \theta/(1 + \theta^2)$')
    ax.axhline(0.5, color=st.IDAred, ls='--', lw=0.9, label=r'$\pm 0.5$: the largest possible $|\rho(1)|$')
    ax.axhline(-0.5, color=st.IDAred, ls='--', lw=0.9, label='_nolegend_')
    ax.axvspan(-1, 1, color=st.Forest, alpha=0.10, label=r'invertible: $|\theta| < 1$')
    for t0, c in [(0.5, st.Orange), (2.0, st.Purple)]:
        ax.plot(t0, t0 / (1 + t0 ** 2), 'o', ms=8, color=c, label=rf'$\theta = {t0}$: $\rho(1) = 0.4$')
    ax.axhline(0, color=st.DarkText, lw=0.6)
    ax.set_xlabel(r'$\theta$')
    ax.set_title(r'MA(1): $\theta$ and $1/\theta$ give the same autocorrelation')
    st.legend_outside_bottom(ax, ncol=3, y=-0.2)
    plt.tight_layout()
    save('tsa_ch2_ma1_rho', save_it)
    return {'r05': 0.5 / 1.25, 'r2': 2 / 5}


# =============================================================================
# 3. ARMA: ACF/PACF PATTERNS AND PSI WEIGHTS
# =============================================================================
MODELS = {'AR(2)': ([1.0, -0.6], []), 'MA(2)': ([], [0.6, 0.3]), 'ARMA(1,1)': ([0.7], [0.4])}


def fig_patterns(n=500, nlags=15, save_it=True):
    """Theoretical (dots) and sample (bars, one simulated path of n = 500) ACF and PACF of AR(2), MA(2), ARMA(1,1)."""
    rng = np.random.default_rng(SEED + 2)
    cols = [st.MainBlue, st.IDAred, st.Forest]
    fig, ax = plt.subplots(2, 3, figsize=(9.33, 3.27))
    out = {}
    for j, ((lab, (ph, th)), c) in enumerate(zip(MODELS.items(), cols)):
        x = simulate_arma(ph, th, n=n, rng=rng)
        r, pc = sample_acf(x, nlags), sample_pacf(x, nlags)
        acf_bars(ax[0, j], r, n, color=c, theory=theoretical_acf(ph, th, nlags),
                 label='sample' if j == 0 else '_nolegend_', theory_label='theoretical' if j == 0 else '_nolegend_')
        acf_bars(ax[1, j], pc, n, color=c, theory=theoretical_pacf(ph, th, nlags), label='_nolegend_',
                 theory_label='_nolegend_')
        ax[0, j].set_title(f'{lab}: ACF')
        ax[1, j].set_title(f'{lab}: PACF')
        for a in ax[:, j]:
            a.set_ylim(-0.8, 1.0)
            if j:
                a.get_lines()[1].set_label('_nolegend_')
        ax[1, j].get_lines()[1].set_label('_nolegend_')
        ax[1, j].set_xlabel('lag h')
        out[lab] = {'r': [float(v) for v in r[:4]], 'p': [float(v) for v in pc[:4]],
                    'rt': [float(v) for v in theoretical_acf(ph, th, 4)], 'pt': [float(v) for v in theoretical_pacf(ph, th, 4)]}
    st.fig_legend_bottom(fig, ncol=3, y=-0.01)
    plt.tight_layout()
    save('tsa_ch2_patterns', save_it)
    return out


def fig_psi(n=16, save_it=True):
    """psi weights (impulse responses) of AR(1) phi = 0.8, AR(2) (1.0, -0.6), MA(2) (0.6, 0.3) and ARMA(1,1)."""
    cases = {r'AR(1), $\phi = 0.8$': ([0.8], [], st.MainBlue), r'AR(2), $\phi = (1.0, -0.6)$': ([1.0, -0.6], [], st.IDAred),
             r'MA(2), $\theta = (0.6, 0.3)$': ([], [0.6, 0.3], st.Forest),
             r'ARMA(1,1), $\phi = 0.7, \theta = 0.4$': ([0.7], [0.4], st.Purple)}
    fig, ax = plt.subplots(1, 4, figsize=(8.91, 3.58), sharey=True)
    out = {}
    for a, (lab, (ph, th, c)) in zip(ax, cases.items()):
        psi = psi_weights(ph, th, n)
        a.bar(np.arange(n), psi, width=0.6, color=c)
        a.axhline(0, color=st.DarkText, lw=0.6)
        a.set_title(lab, fontsize=10.5)
        a.set_xlabel('j')
        out[lab.split(',')[0]] = [float(v) for v in psi[:6]]
    ax[0].set_ylabel(r'$\psi_j$')
    plt.tight_layout()
    save('tsa_ch2_psi', save_it)
    return out


# =============================================================================
# 4. ESTIMATION (Monte Carlo)
# =============================================================================
def css_ma1(x):
    """Conditional least squares for X_t = mu + e_t + theta e_{t-1}, with e_0 = 0: minimise the sum of squares."""
    from scipy.optimize import minimize_scalar
    x = np.asarray(x, float) - np.mean(x)

    def ssr(th):
        e, s = 0.0, 0.0
        for v in x:
            e = v - th * e
            s += e * e
        return s
    return float(minimize_scalar(ssr, bounds=(-0.99, 0.99), method='bounded').x)


def ar1_estimates(x):
    """Yule-Walker, conditional least squares (OLS of X_t on X_{t-1}) and Gaussian ML for an AR(1)."""
    yw = float(yule_walker(x, order=1, method='mle')[0][0])
    y, z = x[1:], x[:-1]
    Z = np.c_[np.ones_like(z), z]
    ols = float(np.linalg.lstsq(Z, y, rcond=None)[0][1])
    ml = float(fit_arma(x, 1, 0).params[1])
    return yw, ols, ml


def fig_estimators(nrep=400, T=100, phi=0.9, theta=0.5, save_it=True):
    """Sampling distributions of YW, CLS and ML for an AR(1) (phi = 0.9) and of CLS and ML for an MA(1)
    (theta = 0.5), T = 100."""
    rng = np.random.default_rng(SEED + 3)
    A = np.array([ar1_estimates(simulate_arma([phi], n=T, rng=rng)) for _ in range(nrep)])
    M = []
    for _ in range(nrep):
        x = simulate_arma([], [theta], n=T, rng=rng)
        M.append((css_ma1(x), float(fit_arma(x, 0, 1).params[1])))
    M = np.array(M)
    fig, ax = plt.subplots(1, 2, figsize=(9.33, 3.36))
    bins = np.linspace(0.55, 1.05, 40)
    for k, (lab, c) in enumerate([('Yule-Walker', st.Orange), ('conditional LS', st.MainBlue), ('Gaussian ML', st.IDAred)]):
        ax[0].hist(A[:, k], bins=bins, histtype='step', lw=1.6, color=c, label=lab)
    ax[0].axvline(phi, color=st.DarkText, ls='--', lw=1, label='true value')
    ax[0].set_title(rf'AR(1), $\phi = {phi}$, $T = {T}$: {nrep} samples')
    bins = np.linspace(0.05, 0.95, 40)
    ax[1].hist(M[:, 0], bins=bins, histtype='step', lw=1.6, color=st.MainBlue, label='_nolegend_')
    ax[1].hist(M[:, 1], bins=bins, histtype='step', lw=1.6, color=st.IDAred, label='_nolegend_')
    ax[1].axvline(theta, color=st.DarkText, ls='--', lw=1, label='_nolegend_')
    ax[1].set_title(rf'MA(1), $\theta = {theta}$, $T = {T}$: conditional LS and ML')
    st.fig_legend_bottom(fig, ncol=4, y=-0.01)
    plt.tight_layout()
    save('tsa_ch2_estimators', save_it)
    names = ['yw', 'cls', 'ml']
    out = {f'ar_{k}_mean': float(A[:, i].mean()) for i, k in enumerate(names)}
    out.update({f'ar_{k}_sd': float(A[:, i].std()) for i, k in enumerate(names)})
    out.update({'ma_cls_mean': float(M[:, 0].mean()), 'ma_ml_mean': float(M[:, 1].mean()),
                'ma_cls_sd': float(M[:, 0].std()), 'ma_ml_sd': float(M[:, 1].std()),
                'ar_se_asy': float(np.sqrt((1 - phi ** 2) / T)), 'ma_se_asy': float(np.sqrt((1 - theta ** 2) / T)),
                'nrep': nrep, 'T': T})
    return out


# =============================================================================
# 5. MODEL SELECTION AND RESIDUAL TESTS (Monte Carlo)
# =============================================================================
def ar_ols_ic(x, pmax=6):
    """AR(p), p = 0..pmax, fitted by OLS on the same sample t = pmax+1..T; AIC and BIC per observation count."""
    x = np.asarray(x, float)
    T = len(x)
    y = x[pmax:]
    n = len(y)
    aic, bic = [], []
    for p in range(pmax + 1):
        Z = np.c_[np.ones(n)] if p == 0 else np.c_[np.ones(n), np.column_stack([x[pmax - i:T - i] for i in range(1, p + 1)])]
        b = np.linalg.lstsq(Z, y, rcond=None)[0]
        s2 = np.mean((y - Z @ b) ** 2)
        ll = -0.5 * n * (np.log(2 * np.pi * s2) + 1)
        k = p + 2
        aic.append(-2 * ll + 2 * k)
        bic.append(-2 * ll + k * np.log(n))
    return int(np.argmin(aic)), int(np.argmin(bic))


def fig_ic(nrep=500, phi=(0.5, 0.25), pmax=6, save_it=True):
    """How often AIC and BIC choose each AR order when the truth is AR(2), for T = 100 and T = 1000."""
    rng = np.random.default_rng(SEED + 4)
    fig, ax = plt.subplots(1, 2, figsize=(9.33, 3.36), sharey=True)
    out = {}
    for a, T in zip(ax, [100, 1000]):
        ch = np.array([ar_ols_ic(simulate_arma(phi, n=T, rng=rng), pmax) for _ in range(nrep)])
        fa = np.bincount(ch[:, 0], minlength=pmax + 1) / nrep
        fb = np.bincount(ch[:, 1], minlength=pmax + 1) / nrep
        p = np.arange(pmax + 1)
        a.bar(p - 0.2, 100 * fa, width=0.4, color=st.MainBlue, label='AIC')
        a.bar(p + 0.2, 100 * fb, width=0.4, color=st.IDAred, label='BIC')
        a.axvline(2, color=st.Forest, ls='--', lw=1, label='true order p = 2')
        a.set_title(f'True AR(2), T = {T}: {nrep} samples')
        a.set_xlabel('chosen order p')
        a.set_xticks(p)
        out[str(T)] = {'aic_true': float(fa[2]), 'bic_true': float(fb[2]), 'aic_over': float(fa[3:].sum()),
                       'bic_over': float(fb[3:].sum()), 'aic_under': float(fa[:2].sum()), 'bic_under': float(fb[:2].sum())}
    ax[0].set_ylabel('% of samples')
    for a in ax[1:]:
        for b in a.patches + a.get_lines():
            b.set_label('_nolegend_')
    st.fig_legend_bottom(fig, ncol=3, y=-0.01)
    plt.tight_layout()
    save('tsa_ch2_ic', save_it)
    return out


def fig_lb_df(nrep=1000, T=200, m=10, save_it=True):
    """Ljung-Box on the residuals of a correctly specified model: rejection rate at 5% with m and with m - p - q
    degrees of freedom (AR(1) phi = 0.7 fitted by OLS; ARMA(1,1) phi = 0.7, theta = 0.4 fitted by ML)."""
    rng = np.random.default_rng(SEED + 5)
    rej = {'AR(1)': [0, 0], 'ARMA(1,1)': [0, 0]}
    for i in range(nrep):
        x = simulate_arma([0.7], n=T, rng=rng)
        Z = np.c_[np.ones(T - 1), x[:-1]]
        e = x[1:] - Z @ np.linalg.lstsq(Z, x[1:], rcond=None)[0]
        rej['AR(1)'][0] += ljung_box(e, m)['lb_p'] < 0.05
        rej['AR(1)'][1] += ljung_box(e, m, m - 1)['lb_p'] < 0.05
        if i < nrep // 2:
            x = simulate_arma([0.7], [0.4], n=T, rng=rng)
            e = fit_arma(x, 1, 1).resid[1:]
            rej['ARMA(1,1)'][0] += ljung_box(e, m)['lb_p'] < 0.05
            rej['ARMA(1,1)'][1] += ljung_box(e, m, m - 2)['lb_p'] < 0.05
    out = {'AR(1)': [100 * v / nrep for v in rej['AR(1)']], 'ARMA(1,1)': [100 * v / (nrep // 2) for v in rej['ARMA(1,1)']],
           'T': T, 'm': m, 'nrep': nrep}
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    xs = np.arange(2)
    ax.bar(xs - 0.18, [out['AR(1)'][0], out['ARMA(1,1)'][0]], width=0.36, color=st.Orange, label=f'df = m = {m}')
    ax.bar(xs + 0.18, [out['AR(1)'][1], out['ARMA(1,1)'][1]], width=0.36, color=st.MainBlue, label='df = m - p - q')
    ax.axhline(5, color=st.IDAred, ls='--', lw=1, label='nominal level 5%')
    ax.set_xticks(xs)
    ax.set_xticklabels(['AR(1) residuals', 'ARMA(1,1) residuals'])
    ax.set_ylabel('rejection rate (%)')
    ax.set_title(f'Ljung-Box on residuals of the true model, T = {T}, m = {m}')
    st.legend_outside_bottom(ax, ncol=3, y=-0.16)
    plt.tight_layout()
    save('tsa_ch2_lb_df', save_it)
    return out


# =============================================================================
# 6. FORECASTING
# =============================================================================
def fig_forecast_theory(H=20, save_it=True):
    """AR(1) (phi = 0.9 and 0.5) and MA(1) (theta = 0.6) forecasts from X_T = 3 (mu = 0, sigma = 1) with 95%
    intervals: the forecast reverts to the mean, the interval widens to the unconditional band."""
    h = np.arange(0, H + 1)
    fig, ax = plt.subplots(1, 3, figsize=(9.33, 3.31), sharey=True)
    out = {}
    for a, (lab, ph, th, c) in zip(ax, [(r'AR(1), $\phi = 0.9$', [0.9], [], st.MainBlue),
                                        (r'AR(1), $\phi = 0.5$', [0.5], [], st.Forest),
                                        (r'MA(1), $\theta = 0.6$, $\varepsilon_T = 2$', [], [0.6], st.Purple)]):
        psi = psi_weights(ph, th, H + 1)
        if ph:
            f = 3 * ph[0] ** h
        else:
            f = np.r_[3, 0.6 * 2, np.zeros(H - 1)]
        se = np.r_[0, np.sqrt(np.cumsum(psi[:H] ** 2))]
        a.plot(h, f, 'o-', ms=3, color=c, label='point forecast' if a is ax[0] else '_nolegend_')
        a.fill_between(h, f - 1.96 * se, f + 1.96 * se, color=c, alpha=0.15,
                       label='95% interval' if a is ax[0] else '_nolegend_')
        u = 1.96 * np.sqrt(np.sum(psi_weights(ph, th, 400) ** 2))
        a.axhline(u, color=st.IDAred, ls='--', lw=0.9, label=r'$\mu \pm 1.96\,\sigma_X$' if a is ax[0] else '_nolegend_')
        a.axhline(-u, color=st.IDAred, ls='--', lw=0.9, label='_nolegend_')
        a.axhline(0, color=st.DarkText, lw=0.6)
        a.set_title(lab)
        a.set_xlabel('horizon h')
        out[lab.split('$')[1] if ph else 'ma'] = {'f1': float(f[1]), 'f5': float(f[5]), 'se1': float(se[1]),
                                                  'se5': float(se[5]), 'se_inf': float(u / 1.96)}
    st.fig_legend_bottom(fig, ncol=3, y=-0.01)
    plt.tight_layout()
    save('tsa_ch2_forecast_theory', save_it)
    return out


# =============================================================================
# 7. ROMANIAN ANNUAL GDP GROWTH: THE BOX-JENKINS CASE
# =============================================================================
def gdp_models(y, pmax=3, qmax=3):
    """All ARMA(p, q), p <= pmax, q <= qmax, by Gaussian ML: log-likelihood, AIC, BIC, Ljung-Box on residuals."""
    rows = []
    for p in range(pmax + 1):
        for q in range(qmax + 1):
            try:
                r = fit_arma(y, p, q)
            except Exception:
                continue
            e = r.resid[max(p, q):]
            lb = ljung_box(e, 8, 8 - p - q)
            rows.append({'p': p, 'q': q, 'k': p + q + 2, 'loglik': float(r.llf), 'aic': float(r.aic), 'bic': float(r.bic),
                         'lb8': lb['lb'], 'lb8_df': lb['df'], 'lb8_p': lb['lb_p'], 'sigma': float(np.sqrt(r.params[-1]))})
    return pd.DataFrame(rows)


def fig_gdp_ident(nlags=16, save_it=True):
    """Identification: Romanian annual real GDP growth since 2000, its ACF and PACF."""
    y = ro_gdp_growth('yoy')
    T = len(y)
    fig = plt.figure(figsize=(9.33, 3.27))
    a0 = fig.add_subplot(2, 1, 1)
    a0.plot(y.index, y, color=st.Forest, lw=1.2)
    a0.axhline(y.mean(), color=st.Orange, ls='--', lw=1, label=f'mean {y.mean():.2f}%')
    a0.axhline(0, color=st.DarkText, lw=0.6)
    a0.set_title(r'Romania: real GDP growth, $100\,(\ln Y_t - \ln Y_{t-4})$, %')
    a1, a2 = fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)
    r, pc = sample_acf(y, nlags), sample_pacf(y, nlags)
    acf_bars(a1, r, T, color=st.Forest, label='_nolegend_')
    acf_bars(a2, pc, T, color=st.Forest, label='_nolegend_', band=True)
    a2.get_lines()[0].set_label('_nolegend_')
    a1.set_title('ACF')
    a2.set_title('PACF')
    for a in (a1, a2):
        a.set_ylim(-0.5, 0.8)
        a.set_xlabel('lag (quarters)')
    st.fig_legend_bottom(fig, ncol=2, y=-0.01)
    plt.tight_layout()
    save('tsa_ch2_gdp_ident', save_it)
    i = int(np.argmin(y.values))
    return {'T': T, 'first': y.index[0].strftime('%Y-%m-%d'), 'last': y.index[-1].strftime('%Y-%m-%d'),
            'mean': float(y.mean()), 'sd': float(y.std()), 'r': [float(v) for v in r[:6]], 'p': [float(v) for v in pc[:6]],
            'min': float(y.min()), 'min_d': y.index[i].strftime('%Y-%m-%d'), 'max': float(y.max()),
            'max_d': y.idxmax().strftime('%Y-%m-%d'), 'last_v': float(y.iloc[-1]), 'band': 1.96 / np.sqrt(T),
            'lb8': ljung_box(y, 8)}


def gdp_table(save_csv=True):
    y = ro_gdp_growth('yoy')
    t = gdp_models(y)
    if save_csv:
        t.round(4).to_csv(os.path.join(HERE, 'ch2_gdp_models.csv'), index=False)
    ia, ib = int(t['aic'].idxmin()), int(t['bic'].idxmin())
    sel = lambda p, q: t[(t.p == p) & (t.q == q)].iloc[0].to_dict()   # noqa: E731
    out = {'aic_best': [int(t.loc[ia, 'p']), int(t.loc[ia, 'q'])], 'bic_best': [int(t.loc[ib, 'p']), int(t.loc[ib, 'q'])],
           'rows': {f'{p}{q}': sel(p, q) for p, q in [(0, 0), (1, 0), (2, 0), (0, 3), (1, 1), (1, 3), (2, 3), (3, 3)]
                    if len(t[(t.p == p) & (t.q == q)])}}
    for k, (p, q) in [('aic', out['aic_best']), ('bic', out['bic_best'])]:
        r = fit_arma(y, p, q)
        out[f'{k}_params'] = {n: float(v) for n, v in zip(r.param_names, r.params)}
        out[f'{k}_se'] = {n: float(v) for n, v in zip(r.param_names, r.bse)}
        th = [v for n, v in zip(r.param_names, r.params) if n.startswith('ma.')]
        out[f'{k}_ma_root_mod'] = [float(v) for v in np.sort(np.abs(np.roots(np.r_[th[::-1], 1.0])))] if th else []
    return out


def fig_gdp_diag(p=0, q=3, save_it=True):
    """Residual diagnostics of the chosen model: residuals, residual ACF, Ljung-Box p-values with m - p - q degrees of
    freedom, and a Normal QQ plot."""
    y = ro_gdp_growth('yoy')
    r = fit_arma(y, p, q)
    e = pd.Series(r.resid, index=y.index).iloc[max(p, q):]
    T = len(e)
    fig, ax = plt.subplots(2, 2, figsize=(9.33, 3.27))
    ax[0, 0].plot(e.index, e, color=st.MainBlue, lw=1)
    ax[0, 0].axhline(0, color=st.DarkText, lw=0.6)
    ax[0, 0].set_title(f'Residuals of ARMA({p},{q})')
    acf_bars(ax[0, 1], sample_acf(e, 16), T, color=st.MainBlue, label='_nolegend_')
    ax[0, 1].get_lines()[0].set_label(r'$\pm 1.96/\sqrt{T}$')
    ax[0, 1].set_title('ACF of the residuals')
    ax[0, 1].set_ylim(-0.4, 0.4)
    ms = np.arange(p + q + 1, 17)
    pv = [ljung_box(e, m, m - p - q)['lb_p'] for m in ms]
    ax[1, 0].plot(ms, pv, 'o', color=st.MainBlue, label='_nolegend_')
    ax[1, 0].axhline(0.05, color=st.Orange, ls='--', lw=1, label='5% level')
    ax[1, 0].set_ylim(0, 1)
    ax[1, 0].set_xlabel('m')
    ax[1, 0].set_title(f'Ljung-Box p-values, df = m - {p + q}')
    (osm, osr), (sl, ic, _) = stats.probplot(e / e.std(), dist='norm')
    ax[1, 1].plot(osm, osr, 'o', ms=3.5, color=st.MainBlue, label='_nolegend_')
    ax[1, 1].plot(osm, sl * osm + ic, color=st.IDAred, lw=1, label='Normal reference line')
    ax[1, 1].set_title('Normal QQ plot of the standardised residuals')
    ax[1, 1].set_xlabel('theoretical quantiles')
    st.fig_legend_bottom(fig, ncol=3, y=-0.01)
    plt.tight_layout()
    save('tsa_ch2_gdp_diag', save_it)
    jb = stats.jarque_bera(e)
    i = int(np.argmax(np.abs(e.values)))
    lb8 = ljung_box(e, 8, 8 - p - q)
    lb8_wrong = ljung_box(e, 8)
    return {'lb8': lb8, 'lb8_wrong_p': lb8_wrong['lb_p'], 'lb16': ljung_box(e, 16, 16 - p - q), 'jb': float(jb.statistic),
            'jb_p': float(jb.pvalue), 'skew': float(stats.skew(e)), 'kurt': float(stats.kurtosis(e) + 3),
            'out_d': e.index[i].strftime('%Y-%m-%d'), 'out_v': float(e.iloc[i]), 'out_z': float(e.iloc[i] / e.std()),
            'min_p': float(min(pv)), 'lbsq8': ljung_box(e ** 2, 8)}


def fig_gdp_forecast(models=((0, 3), (1, 0)), H=8, save_it=True):
    """Forecasts of Romanian annual GDP growth, 8 quarters ahead, from MA(3) and AR(1), with 95% intervals."""
    y = ro_gdp_growth('yoy')
    idx = pd.date_range(y.index[-1] + pd.offsets.QuarterBegin(1, startingMonth=1), periods=H, freq='QS')
    fig, ax = plt.subplots(figsize=(9.33, 3.75))
    ax.plot(y.index[-40:], y.iloc[-40:], color=st.DarkText, lw=1.2, label='observed')
    out = {}
    for (p, q), c in zip(models, [st.IDAred, st.MainBlue]):
        r = fit_arma(y, p, q)
        f = r.get_forecast(H)
        m, ci = f.predicted_mean, f.conf_int(alpha=0.05)
        lab = f'ARMA({p},{q})'
        ax.plot(idx, m, 'o-', ms=3, color=c, label=f'{lab} forecast')
        ax.fill_between(idx, ci[:, 0], ci[:, 1], color=c, alpha=0.13, label=f'{lab}: 95% interval')
        out[lab] = {'f': [float(v) for v in m], 'lo': [float(v) for v in ci[:, 0]], 'hi': [float(v) for v in ci[:, 1]],
                    'mu': float(r.params[0])}
    ax.axhline(0, color=st.DarkText, lw=0.6)
    ax.set_title('Romania: annual real GDP growth (%), forecasts for the next 8 quarters')
    st.legend_outside_bottom(ax, ncol=3, y=-0.12)
    plt.tight_layout()
    save('tsa_ch2_gdp_forecast', save_it)
    out['first'] = idx[0].strftime('%Y-%m-%d')
    out['last'] = idx[-1].strftime('%Y-%m-%d')
    return out


# =============================================================================
# 8. ROMANIAN INFLATION: AR(2) AND MEAN REVERSION
# =============================================================================
def fig_inflation(H=24, save_it=True):
    """Romanian 12-month HICP inflation since 2005: AR(p) order by BIC (p <= 6), ML fit, 24-month forecast with 80% and
    95% intervals; the BNR target band 2.5% +- 1 p.p. for reference."""
    y = ro_inflation()
    bic = []
    for p in range(1, 7):
        bic.append(fit_arma(y.values, p, 0).bic)
    p = int(np.argmin(bic) + 1)
    r = fit_arma(y.values, p, 0)
    f = r.get_forecast(H)
    m, c95, c80 = f.predicted_mean, f.conf_int(alpha=0.05), f.conf_int(alpha=0.20)
    idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=H, freq='MS')
    fig, ax = plt.subplots(figsize=(9.33, 3.75))
    ax.plot(y.index, y, color=st.IDAred, lw=1.2, label='12-month inflation')
    ax.plot(idx, m, color=st.MainBlue, lw=1.6, label=f'AR({p}) forecast')
    ax.fill_between(idx, c95[:, 0], c95[:, 1], color=st.MainBlue, alpha=0.12, label='95% interval')
    ax.fill_between(idx, c80[:, 0], c80[:, 1], color=st.MainBlue, alpha=0.22, label='80% interval')
    mu = float(r.params[0])
    ax.axhline(mu, color=st.Orange, ls='--', lw=1, label=f'estimated mean {mu:.2f}%')
    ax.axhspan(1.5, 3.5, color=st.Forest, alpha=0.10, label='BNR target 2.5% $\\pm$ 1 p.p.')
    ax.set_title('Romania: HICP inflation, 12-month rate (%), and an AR forecast')
    st.legend_outside_bottom(ax, ncol=3, y=-0.12)
    plt.tight_layout()
    save('tsa_ch2_inflation', save_it)
    ph = r.params[1:1 + p]
    ir = inverse_roots(ph)
    return {'p': p, 'bic': [float(v) for v in bic], 'params': [float(v) for v in r.params], 'se': [float(v) for v in r.bse],
            'mu': mu, 'sum_phi': float(np.sum(ph)), 'inv_mod': [float(abs(z)) for z in ir], 'T': int(len(y)),
            'last': y.index[-1].strftime('%Y-%m-%d'), 'last_v': float(y.iloc[-1]), 'f': [float(v) for v in m],
            'lo95': [float(v) for v in c95[:, 0]], 'hi95': [float(v) for v in c95[:, 1]],
            'f_last': idx[-1].strftime('%Y-%m-%d'), 'max': float(y.max()), 'max_d': y.idxmax().strftime('%Y-%m-%d'),
            'lb12': ljung_box(r.resid[p:], 12, 12 - p), 'sigma': float(np.sqrt(r.params[-1]))}


# =============================================================================
# 9. DAILY RETURNS: BET AND EUR/RON
# =============================================================================
def returns_ar(name, start=None):
    """AR(1) for daily log returns (%): estimate, standard error, R^2, Ljung-Box on residuals and on squared residuals,
    Jarque-Bera; the orders chosen by AIC and BIC among ARMA(p, q), p, q <= 2."""
    r = log_returns(name, start=start)
    m = fit_arma(r.values, 1, 0)
    e = m.resid[1:]
    jb = stats.jarque_bera(e)
    ic = []
    for p in range(3):
        for q in range(3):
            try:
                f = fit_arma(r.values, p, q)
                ic.append((p, q, f.aic, f.bic))
            except Exception:
                pass
    ic = pd.DataFrame(ic, columns=['p', 'q', 'aic', 'bic'])
    a, b = ic.loc[ic.aic.idxmin()], ic.loc[ic.bic.idxmin()]
    return {'T': int(len(r)), 'first': r.index[0].strftime('%Y-%m-%d'), 'phi': float(m.params[1]), 'se': float(m.bse[1]),
            't': float(m.params[1] / m.bse[1]), 'r2': float(1 - np.var(e) / np.var(r.values[1:])),
            'lb10': ljung_box(e, 10, 9), 'lbsq10': ljung_box(e ** 2, 10), 'jb': float(jb.statistic), 'jb_p': float(jb.pvalue),
            'kurt': float(stats.kurtosis(e) + 3), 'aic': [int(a.p), int(a.q)], 'bic': [int(b.p), int(b.q)],
            'sd': float(r.std()), 'lb10_raw': ljung_box(r.values, 10)}


def fig_returns(nlags=20, save_it=True):
    """BET and EUR/RON daily returns: ACF of the returns, ACF of the squared residuals of an AR(1)."""
    fig, ax = plt.subplots(2, 2, figsize=(9.33, 3.27))
    out = {}
    for i, (k, lab, c) in enumerate([('bet', 'BET', st.IDAred), ('eurron', 'EUR/RON', st.Forest)]):
        r = log_returns(k)
        e = fit_arma(r.values, 1, 0).resid[1:]
        T = len(r)
        acf_bars(ax[i, 0], sample_acf(r, nlags), T, color=c, label='_nolegend_')
        acf_bars(ax[i, 1], sample_acf(e ** 2, nlags), T, color=c, label='_nolegend_')
        ax[i, 0].set_title(f'{lab}: ACF of daily returns')
        ax[i, 1].set_title(f'{lab}: ACF of squared AR(1) residuals')
        ax[i, 0].set_ylim(-0.1, 0.25)
        ax[i, 1].set_ylim(-0.1, 0.45)
        out[k] = returns_ar(k)
    ax[1, 0].set_xlabel('lag (days)')
    ax[1, 1].set_xlabel('lag (days)')
    ax[0, 0].get_lines()[0].set_label(r'$\pm 1.96/\sqrt{T}$')
    st.fig_legend_bottom(fig, ncol=1, y=-0.01)
    plt.tight_layout()
    save('tsa_ch2_returns', save_it)
    return out


# =============================================================================
# 10. SUNSPOTS: YULE'S AR(2)
# =============================================================================
def fig_sunspots(save_it=True):
    """Yearly sunspot numbers (statsmodels): AR(2) by least squares on 1749-1924 (the period of Yule, 1927), the
    implied period of the pseudo-cycle, and the AR order chosen by AIC and BIC on the full sample."""
    s = load_statsmodels('sunspots')
    y = s.loc['1749':'1924']
    x = y.values - y.values.mean()
    Z = np.c_[x[1:-1], x[:-2]]
    b = np.linalg.lstsq(Z, x[2:], rcond=None)[0]
    period = 2 * np.pi / np.arccos(b[0] / (2 * np.sqrt(-b[1])))
    aic, bic = [], []
    for p in range(1, 11):
        f = fit_arma(s.values, p, 0)
        aic.append(f.aic)
        bic.append(f.bic)
    r2 = fit_arma(s.values, 2, 0)
    fitted = s.values - r2.resid
    fig, ax = plt.subplots(1, 2, figsize=(9.33, 3.36), gridspec_kw={'width_ratios': [2.1, 1]})
    ax[0].plot(s.index, s, color=st.Amber, lw=1.1, label='sunspot number')
    ax[0].plot(s.index[2:], fitted[2:], color=st.MainBlue, lw=0.9, label='AR(2): one-step prediction')
    ax[0].axvspan(pd.Timestamp('1749'), pd.Timestamp('1924'), color=st.Teal, alpha=0.08, label="Yule's sample 1749-1924")
    ax[0].set_title('Yearly sunspot numbers and AR(2) one-step predictions')
    ax[1].plot(range(1, 11), aic, 'o-', color=st.MainBlue, label='AIC')
    ax[1].plot(range(1, 11), bic, 's-', color=st.IDAred, label='BIC')
    ax[1].set_xlabel('AR order p')
    ax[1].set_title('Information criteria, full sample')
    st.fig_legend_bottom(fig, ncol=5, y=-0.01)
    plt.tight_layout()
    save('tsa_ch2_sunspots', save_it)
    ir = inverse_roots(b)
    return {'phi1': float(b[0]), 'phi2': float(b[1]), 'period': float(period), 'mod': float(abs(ir[0])),
            'n_yule': int(len(y)), 'aic_p': int(np.argmin(aic) + 1), 'bic_p': int(np.argmin(bic) + 1),
            'full_phi': [float(v) for v in r2.params[1:3]], 'n': int(len(s)),
            'r2': float(1 - np.var(r2.resid[2:]) / np.var(s.values[2:]))}


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    st.apply()
    N = {}
    N['sun'] = fig_sunspots()
    for name, f in [('ar1', fig_ar1), ('ma1', fig_ma1), ('ma1rho', fig_ma1_rho), ('patterns', fig_patterns),
                    ('psi', fig_psi), ('est', fig_estimators), ('ic', fig_ic), ('lbdf', fig_lb_df),
                    ('fct', fig_forecast_theory), ('gdp', fig_gdp_ident), ('gdptab', gdp_table),
                    ('infl', fig_inflation), ('ret', fig_returns)]:
        print(name)
        N[name] = f()
    pb, qb = N['gdptab']['bic_best']
    N['gdpdiag'] = fig_gdp_diag(pb, qb)
    N['gdpfc'] = fig_gdp_forecast(models=((pb, qb), (1, 0)))
    N['ar2'] = fig_ar2(sun_phi=(N['sun']['phi1'], N['sun']['phi2']))
    with open(os.path.join(HERE, 'ch2_numbers.json'), 'w') as fh:
        json.dump(N, fh, indent=1, default=float)
    print('written ch2_numbers.json')
