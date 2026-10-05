"""
generate_all_charts.py -- charts and numbers of Chapter 14 (TSA): multivariate GARCH models
==========================================================================================
Course data (tsa_data.py), chart style (tsa_style.py). Every number on the slides comes from here.
  * motivation     -- rolling correlations (S&P 500 and DAX, BET and DAX, S&P 500 and Bitcoin); the effect of the
                      correlation on the risk of a two-asset portfolio;
  * models         -- the number of parameters of VEC, diagonal VEC, BEKK, diagonal and scalar BEKK, CCC and DCC
                      for N assets;
  * DCC            -- two-step estimation (Engle 2002): step 1 a GARCH(1,1) per series (arch package, or the numpy
                      fallback below), step 2 the parameters (a, b) of the correlation recursion by maximum
                      likelihood with correlation targeting; CCC (Bollerslev 1990) as the special case a = b = 0;
                      S&P 500 and DAX, daily, on common trading days since 2000; the crises of 2008 and 2020;
  * panel          -- a five-asset DCC on weekly returns (S&P 500, DAX, BET, EUR/RON, Bitcoin) since 2015;
  * portfolio VaR  -- VaR 1% of a 50/50 S&P 500 and DAX portfolio from DCC, CCC and a static covariance matrix,
                      parameters estimated on 2000-2014 and the filters run on 2015-2026; Kupiec (1995) test;
  * hedging        -- the minimum-variance hedge ratio of the BET with the DAX: DCC, rolling OLS and static OLS,
                      estimated on 2005-2014 and evaluated on 2015-2026 (hedging effectiveness).
Output: charts/tsa_ch14_*.pdf/.png, Quantlets/Ch_14/ch14_numbers.json
Partly adapted from the MFM course (Chapter 6, multivariate volatility and dependence).
Run:  python3 Quantlets/Ch_14/generate_all_charts.py
Time Series Analysis - Daniel Traian PELE
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from tsa_data import load_close   # noqa: E402
import tsa_style as st             # noqa: E402

warnings.filterwarnings('ignore')
NAME = {'sp500': 'S&P 500', 'dax': 'DAX', 'bet': 'BET', 'eurron': 'EUR/RON', 'btc': 'Bitcoin'}
COLORS = {'sp500': st.MainBlue, 'dax': st.Teal, 'bet': st.IDAred, 'eurron': st.Forest, 'btc': st.Amber}
PANEL = ['sp500', 'dax', 'bet', 'eurron', 'btc']
PANEL_START = '2015-01-01'                  # five-asset panel, daily on common trading days (Bitcoin from 2015)
PAIR_START = '2000-01-01'                   # S&P 500 and DAX, daily
HEDGE_START = '2005-01-01'                  # BET and DAX, daily
SPLIT = '2014-12-31'                        # estimation sample ends here; 2015-2026 is out of sample
ROLL = 250                                  # rolling window (trading days) of the rolling correlation and OLS
ALPHA_VAR = 0.01                            # VaR 1%
WEIGHTS = (0.5, 0.5)                        # the S&P 500 and DAX portfolio
CRISES = {'2008': ('2008-09-15', '2009-03-31'), '2020': ('2020-02-20', '2020-04-30')}
CALM = ('2017-01-01', '2019-12-31')
PANEL_CRISIS = ('2020-02-14', '2020-06-30')
EX = dict(s1=20.0, s2=20.0, rhos=(0.2, 0.8),                                  # diversification example
          a=0.05, b=0.93, qbar=0.5, q11=1.0, q22=1.0, q12=0.5, z=(-2.0, -2.5),  # DCC step by step
          var_s=(1.5, 1.2), var_rho=(0.3, 0.9),                                 # VaR example
          h_ss=1.2, h_sf=1.0, h_rho=0.8,                                        # hedge example
          bk=dict(c12=0.02, a11=0.30, a22=0.25, b11=0.94, b22=0.95, e=(-2.0, -1.5), h12=0.5))   # diagonal BEKK


def save(name, save_it=True):
    if save_it:
        st.check_no_grey(plt.gcf())
        st.save_fig(name)
    else:
        plt.show()


def shade(ax, periods=CRISES, alpha=0.15):
    for i, (a, b) in enumerate(periods.values()):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=st.Crimson, alpha=alpha, lw=0,
                   label='Crisis periods (2008, 2020)' if i == 0 else None)


# =============================================================================
# DATA
# =============================================================================
def common_returns(names, start, end=None):
    """Daily log returns in % computed on the days on which ALL markets trade: prices are aligned first, so each
    return covers the same calendar interval for every series."""
    P = pd.concat([load_close(k, start) for k in names], axis=1, join='inner')
    if end:
        P = P.loc[:end]
    return (100 * np.log(P).diff()).dropna()


def weekly_returns(names, start=PANEL_START):
    """Weekly log returns in % (Friday closes, or the last close of the week): the weekly horizon removes most of the
    asynchronous-trading problem of markets that close at different hours (New York, Frankfurt, Bucharest, crypto)."""
    P = pd.concat([load_close(k, start).resample('W-FRI').last() for k in names], axis=1).dropna()
    return (100 * np.log(P).diff()).dropna()


# =============================================================================
# STEP 1: UNIVARIATE GARCH(1,1)
# =============================================================================
def garch11_negloglik(theta, r):
    """Gaussian negative log-likelihood of r_t = mu + e_t, sigma2_t = omega + alpha e_{t-1}^2 + beta sigma2_{t-1}."""
    mu, omega, alpha, beta = theta
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e10
    e = r - mu
    s2 = garch11_variance(e, omega, alpha, beta)
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + e ** 2 / s2)


def garch11_variance(e, omega, alpha, beta, s0=None):
    """sigma2_t = omega + alpha e_{t-1}^2 + beta sigma2_{t-1}, started at the sample variance (one-step-ahead:
    sigma2_t uses only e_1, ..., e_{t-1})."""
    x = np.empty(len(e))
    x[0] = np.var(e) if s0 is None else s0
    x[1:] = omega + alpha * e[:-1] ** 2
    # sigma2_t - beta sigma2_{t-1} = x_t  -> a linear filter
    out = lfilter([1.0], [1.0, -beta], x)
    return out


def garch11_fit(r):
    """Step 1 of DCC: GARCH(1,1) with a constant mean, Gaussian QML. The arch package if installed, else a numpy
    maximum-likelihood fallback (the two give practically the same estimates)."""
    r = np.asarray(r, float)
    try:
        from arch import arch_model
        res = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='normal').fit(disp='off')
        p = res.params
        return dict(mu=float(p['mu']), omega=float(p['omega']), alpha=float(p['alpha[1]']), beta=float(p['beta[1]']))
    except ImportError:
        v = np.var(r)
        best = optimize.minimize(garch11_negloglik, [r.mean(), 0.05 * v, 0.08, 0.90], args=(r,), method='Nelder-Mead',
                                 options=dict(maxiter=4000, xatol=1e-8, fatol=1e-8))
        mu, omega, alpha, beta = best.x
        return dict(mu=float(mu), omega=float(omega), alpha=float(alpha), beta=float(beta))


def garch11_filter(r, p, s0=None):
    """Conditional volatility (one step ahead) and standardised residuals z_t = (r_t - mu) / sigma_t for given
    parameters p (from garch11_fit), on any sample (also out of sample)."""
    e = np.asarray(r, float) - p['mu']
    s2 = garch11_variance(e, p['omega'], p['alpha'], p['beta'], s0)
    s = np.sqrt(s2)
    return s, e / s


def step1(R, fit_until=None):
    """GARCH(1,1) for every column of R (estimated on R.loc[:fit_until]); volatilities and residuals on all of R."""
    P, S, Z = {}, {}, {}
    for c in R.columns:
        x = R[c].loc[:fit_until] if fit_until else R[c]
        P[c] = garch11_fit(x.values)
        s0 = float(np.var(x.values - P[c]['mu']))
        S[c], Z[c] = garch11_filter(R[c].values, P[c], s0)
    return P, pd.DataFrame(S, index=R.index), pd.DataFrame(Z, index=R.index)


# =============================================================================
# STEP 2: DCC (Engle 2002)  Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1},  R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
# =============================================================================
def dcc_path(Z, a, b, Qbar=None):
    """Correlation matrices R_t (T x N x N), one step ahead (R_t uses z_1, ..., z_{t-1}). Every element of Q_t follows
    the same linear recursion, computed with scipy's lfilter."""
    X = np.asarray(Z, float)
    T, N = X.shape
    Qbar = np.corrcoef(X.T) if Qbar is None else Qbar
    P = np.einsum('ti,tj->tij', X, X)                       # z_t z_t'
    drive = np.empty_like(P)
    drive[0] = Qbar
    drive[1:] = (1 - a - b) * Qbar + a * P[:-1]
    Q = lfilter([1.0], [1.0, -b], drive.reshape(T, -1), axis=0).reshape(T, N, N)
    Q[0] = Qbar
    d = np.sqrt(np.einsum('tii->ti', Q))
    return Q / (d[:, :, None] * d[:, None, :])


def dcc_loglik(Z, a, b, Qbar=None):
    """Correlation part of the Gaussian log-likelihood: -1/2 sum_t (log|R_t| + z_t' R_t^{-1} z_t - z_t' z_t)."""
    X = np.asarray(Z, float)
    R = dcc_path(X, a, b, Qbar)
    _, logdet = np.linalg.slogdet(R)
    quad = np.einsum('ti,tij,tj->t', X, np.linalg.inv(R), X)
    return float(-0.5 * np.sum(logdet + quad - np.sum(X ** 2, axis=1)))


def num_hessian(f, x, h=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    E = np.eye(k) * h
    for i in range(k):
        for j in range(k):
            H[i, j] = (f(x + E[i] + E[j]) - f(x + E[i] - E[j]) - f(x - E[i] + E[j]) + f(x - E[i] - E[j])) / (4 * h * h)
    return H


def dcc_fit(Z):
    """Step 2: (a, b) by maximum likelihood with Qbar fixed at the sample correlation of z (correlation targeting).
    Standard errors from the step-2 Hessian (they ignore the estimation error of step 1). CCC: a = b = 0."""
    X = np.asarray(Z, float)
    Qbar = np.corrcoef(X.T)

    def nll(p):
        a, b = p
        if a < 0 or b < 0 or a + b >= 0.9999:
            return 1e10
        return -dcc_loglik(X, a, b, Qbar)

    best = None
    for s in ((0.03, 0.95), (0.01, 0.98), (0.05, 0.90), (0.10, 0.80)):
        r = optimize.minimize(nll, s, method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-7, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    a, b = best.x
    try:
        se = np.sqrt(np.diag(np.linalg.inv(num_hessian(nll, best.x))))
    except Exception:
        se = np.array([np.nan, np.nan])
    ll, ll0 = -best.fun, dcc_loglik(X, 0.0, 0.0, Qbar)
    return dict(a=float(a), b=float(b), se_a=float(se[0]), se_b=float(se[1]), loglik=float(ll), ccc_loglik=float(ll0),
                lr=float(2 * (ll - ll0)), Qbar=Qbar)


def half_life(p):
    return float(np.log(0.5) / np.log(p))


# =============================================================================
# 1. MOTIVATION
# =============================================================================
def fig_rolling_corr(save_it=True):
    """Rolling 250-day correlations of daily returns on common trading days."""
    pairs = [('sp500', 'dax'), ('bet', 'dax'), ('sp500', 'btc')]
    cols = [st.MainBlue, st.IDAred, st.Amber]
    out = {}
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for (x, y), c in zip(pairs, cols):
        R = common_returns([x, y], PAIR_START if y != 'btc' else '2014-09-17')
        rc = R[x].rolling(ROLL).corr(R[y]).dropna()
        ax.plot(rc.index, rc, color=c, lw=1.1, label=f'{NAME[x]} and {NAME[y]}')
        out[f'{x}_{y}'] = dict(min=float(rc.min()), max=float(rc.max()), dmin=str(rc.idxmin().date()),
                               dmax=str(rc.idxmax().date()), last=float(rc.iloc[-1]), full=float(R.corr().iloc[0, 1]),
                               n=len(R))
    R = common_returns(['sp500', 'dax'], '2025-04-01', '2025-04-11')     # the tariff shock of April 2025
    out['apr2025'] = {str(d.date()): [float(R.loc[d, 'sp500']), float(R.loc[d, 'dax'])] for d in R.index}
    shade(ax)
    ax.axhline(0, color=st.DarkText, lw=0.6, ls='--')
    ax.set_ylabel('Correlation (250 trading days)')
    ax.set_ylim(-0.4, 1.0)
    st.legend_outside_bottom(ax, ncol=4)
    save('tsa_ch14_rolling_corr', save_it)
    return out


def worked_examples():
    """Numbers of the worked examples: diversification, one DCC step, a two-asset VaR 1%, a hedge ratio."""
    e = EX
    w = np.array(WEIGHTS)
    div = {}
    for rho in e['rhos']:
        v = (w[0] * e['s1']) ** 2 + (w[1] * e['s2']) ** 2 + 2 * w[0] * w[1] * rho * e['s1'] * e['s2']
        div[str(rho)] = float(np.sqrt(v))
    z1, z2 = e['z']
    c = 1 - e['a'] - e['b']
    q11 = c * 1.0 + e['a'] * z1 * z1 + e['b'] * e['q11']
    q22 = c * 1.0 + e['a'] * z2 * z2 + e['b'] * e['q22']
    q12 = c * e['qbar'] + e['a'] * z1 * z2 + e['b'] * e['q12']
    dcc = dict(c=c, q11=q11, q22=q22, q12=q12, rho=q12 / np.sqrt(q11 * q22),
               t1=c * e['qbar'], t2=e['a'] * z1 * z2, t3=e['b'] * e['q12'])
    zq = float(stats.norm.ppf(ALPHA_VAR))
    var = {'z': zq}
    s1, s2 = e['var_s']
    for rho in e['var_rho']:
        sp = np.sqrt((w[0] * s1) ** 2 + (w[1] * s2) ** 2 + 2 * w[0] * w[1] * rho * s1 * s2)
        var[str(rho)] = dict(sp=float(sp), var=float(-zq * sp))
    h = e['h_rho'] * e['h_ss'] / e['h_sf']
    hed = dict(h=h, cov=e['h_rho'] * e['h_ss'] * e['h_sf'], he=e['h_rho'] ** 2,
               sd_hedged=float(e['h_ss'] * np.sqrt(1 - e['h_rho'] ** 2)))
    k = e['bk']
    t1, t2, t3 = k['c12'], k['a11'] * k['a22'] * k['e'][0] * k['e'][1], k['b11'] * k['b22'] * k['h12']
    bekk = dict(aa=k['a11'] * k['a22'], bb=k['b11'] * k['b22'], ee=k['e'][0] * k['e'][1], t1=t1, t2=t2, t3=t3, h12=t1 + t2 + t3)
    return dict(div=div, dcc={k: float(v) for k, v in dcc.items()}, var=var, hedge={k: float(v) for k, v in hed.items()},
                bekk={k: float(v) for k, v in bekk.items()})


def fig_diversification(save_it=True):
    """Volatility of a 50/50 portfolio of two assets with 20% volatility each, as a function of the correlation."""
    rho = np.linspace(-1, 1, 201)
    s = EX['s1']
    sp = np.sqrt(0.25 * s ** 2 * 2 + 2 * 0.25 * rho * s * s)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(rho, sp, color=st.MainBlue, lw=2, label='Portfolio volatility (50/50, each asset 20%)')
    for r0, c in zip(EX['rhos'], (st.Forest, st.IDAred)):
        v = np.sqrt(0.5 * s ** 2 + 0.5 * r0 * s * s)
        ax.plot([r0], [v], 'o', color=c, ms=8, label=f'rho = {r0}: {v:.1f}%')
    ax.set_xlabel('Correlation rho')
    ax.set_ylabel('Annual volatility (%)')
    st.legend_outside_bottom(ax, ncol=3)
    save('tsa_ch14_diversification', save_it)
    return {}


# =============================================================================
# 2. MODELS: NUMBER OF PARAMETERS
# =============================================================================
def n_params(N):
    k = N * (N + 1) // 2
    return {'VEC': k + 2 * k * k, 'DVEC': 3 * k, 'BEKK': k + 2 * N * N, 'DBEKK': k + 2 * N, 'SBEKK': k + 2,
            'CCC': 3 * N + N * (N - 1) // 2, 'DCC': 3 * N + N * (N - 1) // 2 + 2}


def fig_param_count(save_it=True):
    Ns = np.arange(2, 51)
    lab = {'VEC': 'VEC(1,1)', 'BEKK': 'BEKK(1,1)', 'DVEC': 'Diagonal VEC', 'DBEKK': 'Diagonal BEKK', 'DCC': 'DCC',
           'CCC': 'CCC'}
    cols = {'VEC': st.IDAred, 'BEKK': st.Orange, 'DVEC': st.Purple, 'DBEKK': st.Amber, 'DCC': st.MainBlue, 'CCC': st.Forest}
    fig, ax = plt.subplots(figsize=(9, 4))
    for m in lab:
        ax.plot(Ns, [n_params(n)[m] for n in Ns], color=cols[m], lw=1.8, ls='--' if m == 'CCC' else '-', label=lab[m])
    ax.set_yscale('log')
    ax.set_xlabel('Number of assets N')
    ax.set_ylabel('Number of parameters (log scale)')
    st.legend_outside_bottom(ax, ncol=6)
    save('tsa_ch14_param_count', save_it)
    return {str(n): n_params(n) for n in (2, 5, 10, 50)}


# =============================================================================
# 3. DCC FOR THE S&P 500 AND THE DAX
# =============================================================================
def dcc_sp_dax():
    R = common_returns(['sp500', 'dax'], PAIR_START)
    P, S, Z = step1(R)
    f = dcc_fit(Z.values)
    Rt = dcc_path(Z.values, f['a'], f['b'], f['Qbar'])
    rho = pd.Series(Rt[:, 0, 1], index=R.index)
    return R, P, S, Z, f, rho


def fig_step1(save_it=True):
    R, P, S, Z, f, rho = dcc_sp_dax()
    fig, ax = plt.subplots(figsize=(10, 4))
    for k in ('sp500', 'dax'):
        ax.plot(S.index, S[k] * np.sqrt(252), color=COLORS[k], lw=0.9, label=f'{NAME[k]}: GARCH(1,1) volatility')
    shade(ax)
    ax.set_ylabel('Annualised volatility (%)')
    st.legend_outside_bottom(ax, ncol=3)
    save('tsa_ch14_garch_step1', save_it)
    out = {'n': len(R), 'first': str(R.index[0].date()), 'last': str(R.index[-1].date())}
    for k in ('sp500', 'dax'):
        out[k] = dict(P[k], pers=P[k]['alpha'] + P[k]['beta'], vmax=float(S[k].max() * np.sqrt(252)),
                      dmax=str(S[k].idxmax().date()), zstd=float(Z[k].std()), zkurt=float(stats.kurtosis(Z[k])),
                      rkurt=float(stats.kurtosis(R[k])))
    out['zcorr'] = float(Z.corr().iloc[0, 1])
    out['rcorr'] = float(R.corr().iloc[0, 1])
    return out


def fig_dcc_sp_dax(save_it=True):
    R, P, S, Z, f, rho = dcc_sp_dax()
    roll = R['sp500'].rolling(ROLL).corr(R['dax'])
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(rho.index, rho, color=st.IDAred, lw=0.8, label='DCC correlation')
    ax.plot(roll.index, roll, color=st.MainBlue, lw=1.3, label='Rolling correlation (250 days)')
    ax.axhline(f['Qbar'][0, 1], color=st.Forest, lw=1.5, ls='--', label='CCC (constant) correlation')
    shade(ax)
    ax.set_ylabel('Correlation')
    st.legend_outside_bottom(ax, ncol=4)
    save('tsa_ch14_dcc_sp_dax', save_it)
    out = dict(a=f['a'], b=f['b'], se_a=f['se_a'], se_b=f['se_b'], ab=f['a'] + f['b'], hl=half_life(f['a'] + f['b']),
               ll=f['loglik'], ll0=f['ccc_loglik'], lr=f['lr'], crit=float(stats.chi2.ppf(0.95, 2)), qbar=float(f['Qbar'][0, 1]),
               min=float(rho.min()), dmin=str(rho.idxmin().date()), max=float(rho.max()), dmax=str(rho.idxmax().date()),
               mean=float(rho.mean()), last=float(rho.iloc[-1]), lastd=str(rho.index[-1].date()),
               sd_dcc=float(rho.std()), sd_roll=float(roll.std()))
    for k, (a, b) in dict(CRISES, calm=CALM).items():
        out[f'mean_{k}'] = float(rho.loc[a:b].mean())
        out[f'vol_{k}'] = float(S.loc[a:b, 'sp500'].mean() * np.sqrt(252))
    return out


def fig_crisis_zoom(save_it=True):
    R, P, S, Z, f, rho = dcc_sp_dax()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    win = {'2008': ('2008-06-01', '2009-06-30'), '2020': ('2019-12-01', '2020-08-31')}
    out = {}
    for ax, (k, (a, b)) in zip(axes, win.items()):
        r = rho.loc[a:b]
        v = (S.loc[a:b, 'sp500'] * np.sqrt(252))
        ax.plot(r.index, r, color=st.IDAred, lw=1.4, label='DCC correlation (left axis)')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Correlation')
        ax2 = ax.twinx()
        ax2.plot(v.index, v, color=st.MainBlue, lw=1.2, label='S&P 500 GARCH volatility, % (right axis)')
        ax2.set_ylabel('Volatility (%)')
        ax2.spines['right'].set_visible(True)
        ax.set_title(f'Crisis of {k}')
        ax.tick_params(axis='x', labelrotation=30)
        out[k] = dict(rmax=float(r.max()), rdmax=str(r.idxmax().date()), r0=float(r.iloc[0]), vmax=float(v.max()),
                      vdmax=str(v.idxmax().date()))
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = axes[0].get_figure().axes[2].get_legend_handles_labels()
    st.fig_legend_bottom(fig, h1 + h2, l1 + l2, ncol=2, y=0.0)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    save('tsa_ch14_crisis_zoom', save_it)
    return out


# =============================================================================
# 4. FIVE-ASSET PANEL, WEEKLY
# =============================================================================
def dcc_panel():
    R = common_returns(PANEL, PANEL_START)
    P, S, Z = step1(R)
    f = dcc_fit(Z.values)
    Rt = dcc_path(Z.values, f['a'], f['b'], f['Qbar'])
    return R, P, S, Z, f, Rt


def fig_panel_corr(save_it=True):
    R, P, S, Z, f, Rt = dcc_panel()
    idx = {k: i for i, k in enumerate(PANEL)}
    pairs = [('bet', 'dax'), ('sp500', 'dax'), ('sp500', 'btc'), ('bet', 'eurron')]
    cols = [st.IDAred, st.MainBlue, st.Amber, st.Forest]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    out = dict(a=f['a'], b=f['b'], se_a=f['se_a'], se_b=f['se_b'], ab=f['a'] + f['b'], hl=half_life(f['a'] + f['b']),
               lr=f['lr'], n=len(R), first=str(R.index[0].date()), last=str(R.index[-1].date()),
               garch={k: dict(P[k], pers=P[k]['alpha'] + P[k]['beta']) for k in PANEL})
    W = weekly_returns(PANEL)
    for x, y in [('sp500', 'bet'), ('sp500', 'dax'), ('bet', 'dax')]:
        out[f'async_{x}_{y}'] = dict(daily=float(R[x].corr(R[y])), weekly=float(W[x].corr(W[y])))
    for (x, y), c in zip(pairs, cols):
        s = pd.Series(Rt[:, idx[x], idx[y]], index=R.index)
        ax.plot(s.index, s, color=c, lw=1.2, label=f'{NAME[x]} and {NAME[y]}')
        out[f'{x}_{y}'] = dict(mean=float(s.mean()), min=float(s.min()), max=float(s.max()), dmax=str(s.idxmax().date()),
                               dmin=str(s.idxmin().date()), last=float(s.iloc[-1]), qbar=float(f['Qbar'][idx[x], idx[y]]))
    ax.axvspan(pd.Timestamp(PANEL_CRISIS[0]), pd.Timestamp(PANEL_CRISIS[1]), color=st.Crimson, alpha=0.15, lw=0,
               label='COVID-19 crash (2020)')
    ax.axhline(0, color=st.DarkText, lw=0.6, ls='--')
    ax.set_ylabel('DCC correlation (daily returns)')
    st.legend_outside_bottom(ax, ncol=5)
    save('tsa_ch14_panel_corr', save_it)
    return out


def fig_panel_heatmap(save_it=True):
    R, P, S, Z, f, Rt = dcc_panel()
    T = pd.Series(range(len(R)), index=R.index)
    calm = Rt[T.loc[CALM[0]:CALM[1]].values].mean(axis=0)
    cris = Rt[T.loc[PANEL_CRISIS[0]:PANEL_CRISIS[1]].values].mean(axis=0)
    labs = [NAME[k] for k in PANEL]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, M, t in zip(axes, (calm, cris), ('Calm: 2017-2019', 'COVID-19 crash: Feb-Jun 2020')):
        im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(5))
        ax.set_xticklabels(labs, rotation=35)
        ax.set_yticks(range(5))
        ax.set_yticklabels(labs)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=10,
                        color='white' if abs(M[i, j]) > 0.6 else st.DarkText)
        ax.set_title(t)
    fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.25, label='Average DCC correlation')
    save('tsa_ch14_panel_heatmap', save_it)
    idx = {k: i for i, k in enumerate(PANEL)}
    off = ~np.eye(5, dtype=bool)
    out = dict(calm_avg=float(calm[off].mean()), crisis_avg=float(cris[off].mean()), nc=int(len(T.loc[PANEL_CRISIS[0]:PANEL_CRISIS[1]])))
    for x, y in [('bet', 'dax'), ('sp500', 'dax'), ('sp500', 'btc'), ('bet', 'eurron'), ('dax', 'btc'), ('sp500', 'bet')]:
        out[f'{x}_{y}'] = dict(calm=float(calm[idx[x], idx[y]]), crisis=float(cris[idx[x], idx[y]]))
    return out


# =============================================================================
# 5. PORTFOLIO VaR 1% (out of sample, 2015-2026)
# =============================================================================
def kupiec(viol, alpha=ALPHA_VAR):
    """Kupiec (1995) unconditional coverage test: LR = -2 log[(1-a)^(n-x) a^x / ((1-p)^(n-x) p^x)], chi2(1)."""
    n, x = len(viol), int(np.sum(viol))
    p = x / n
    l0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    l1 = (n - x) * np.log(1 - p) + (x * np.log(p) if x > 0 else 0.0)
    lr = -2 * (l0 - l1)
    return dict(n=n, x=x, rate=p, lr=float(lr), p=float(stats.chi2.sf(lr, 1)))


def christoffersen(viol):
    """Christoffersen (1998) independence test: does a violation today make a violation tomorrow more likely?
    LR = -2 log[L(pi) / L(pi01, pi11)], chi2(1)."""
    v = np.asarray(viol, int)
    a, b = v[:-1], v[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    p01, p11 = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1)
    p = (n01 + n11) / len(a)
    xl = lambda k, q: k * np.log(q) if k > 0 else 0.0
    l0 = xl(n00 + n10, 1 - p) + xl(n01 + n11, p)
    l1 = xl(n00, 1 - p01) + xl(n01, p01) + xl(n10, 1 - p11) + xl(n11, p11)
    lr = -2 * (l0 - l1)
    return dict(n11=int(n11), p01=float(p01), p11=float(p11), lr=float(lr), p=float(stats.chi2.sf(lr, 1)))


def portfolio_var():
    R = common_returns(['sp500', 'dax'], PAIR_START)
    P, S, Z = step1(R, fit_until=SPLIT)
    Zin = Z.loc[:SPLIT]
    f = dcc_fit(Zin.values)
    Rt = dcc_path(Z.values, f['a'], f['b'], f['Qbar'])
    w = np.array(WEIGHTS)
    mu = np.array([P[k]['mu'] for k in R.columns])
    s = S.values
    H = Rt * s[:, :, None] * s[:, None, :]
    sp_dcc = np.sqrt(np.einsum('i,tij,j->t', w, H, w))
    Rc = f['Qbar']                                             # CCC: constant correlation of step-1 residuals
    Hc = Rc[None] * s[:, :, None] * s[:, None, :]
    sp_ccc = np.sqrt(np.einsum('i,tij,j->t', w, Hc, w))
    sp_sta = float(np.sqrt(w @ np.cov(R.loc[:SPLIT].values.T) @ w))
    zq = stats.norm.ppf(ALPHA_VAR)
    rp = R.values @ w
    m = float(w @ mu)
    out_idx = R.index > SPLIT
    res = {'zq': float(zq), 'a': f['a'], 'b': f['b'], 'n_in': int((~out_idx).sum())}
    # filtered historical simulation: the 1% quantile of the in-sample standardised portfolio returns replaces z
    q_fhs = float(np.quantile(((rp - m) / sp_dcc)[~out_idx], ALPHA_VAR))
    res['q_fhs'] = q_fhs
    V = {'DCC': -(m + zq * sp_dcc), 'DCC-FHS': -(m + q_fhs * sp_dcc), 'CCC': -(m + zq * sp_ccc),
         'Static': -(m + zq * sp_sta) * np.ones(len(rp))}
    for k, v in V.items():
        res[k] = kupiec((rp < -v)[out_idx])
        res[k]['ind'] = christoffersen((rp < -v)[out_idx])
        res[k]['mean_var'] = float(v[out_idx].mean())
        res[k]['max_var'] = float(v[out_idx].max())
        for c, (a, b) in CRISES.items():
            if c == '2020':
                sel = (R.index >= a) & (R.index <= b)
                res[k]['viol2020'] = int(np.sum((rp < -v)[sel]))
    s_ = pd.Series
    return R.index, rp, {k: s_(v, index=R.index) for k, v in V.items()}, out_idx, res


def fig_var_backtest(save_it=True):
    idx, rp, V, out_idx, res = portfolio_var()
    rp = pd.Series(rp, index=idx).loc[idx[out_idx]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(rp.index, rp, color=st.MainBlue, lw=0.5, label='Portfolio return (50% S&P 500, 50% DAX), %')
    for k, c in (('DCC-FHS', st.IDAred), ('Static', st.Forest)):
        v = V[k].loc[rp.index]
        ax.plot(v.index, -v, color=c, lw=1.0, label=f'minus VaR 1%: {k}')
        hit = rp < -v
        ax.plot(rp.index[hit], rp[hit], 'v', color=c, ms=4, label=f'Violations: {k} ({int(hit.sum())})')
    ax.set_ylabel('Daily return (%)')
    st.legend_outside_bottom(ax, ncol=3)
    save('tsa_ch14_var_backtest', save_it)
    return res


# =============================================================================
# 6. HEDGING THE BET WITH THE DAX (out of sample, 2015-2026)
# =============================================================================
def hedge():
    R = common_returns(['bet', 'dax'], HEDGE_START)
    P, S, Z = step1(R, fit_until=SPLIT)
    f = dcc_fit(Z.loc[:SPLIT].values)
    Rt = dcc_path(Z.values, f['a'], f['b'], f['Qbar'])
    h_dcc = pd.Series(Rt[:, 0, 1] * S['bet'] / S['dax'], index=R.index)          # cov / var = rho sigma_s / sigma_f
    ins = R.loc[:SPLIT]
    h_sta = float(np.cov(ins['bet'], ins['dax'])[0, 1] / ins['dax'].var())
    h_rol = (R['bet'].rolling(ROLL).cov(R['dax']) / R['dax'].rolling(ROLL).var()).shift(1)   # past data only
    oos = R.loc[R.index > SPLIT]
    hd, hr = h_dcc.loc[oos.index], h_rol.loc[oos.index]
    un = oos['bet']
    out = {'a': f['a'], 'b': f['b'], 'static': h_sta, 'n_oos': len(oos), 'first': str(oos.index[0].date()),
           'dcc_min': float(hd.min()), 'dcc_max': float(hd.max()), 'dcc_mean': float(hd.mean()),
           'dcc_dmax': str(hd.idxmax().date()), 'dcc_dmin': str(hd.idxmin().date()), 'var_un': float(un.var())}
    for k, h in (('static', h_sta), ('rolling', hr), ('dcc', hd)):
        e = un - h * oos['dax']
        out[f'var_{k}'] = float(e.var())
        out[f'he_{k}'] = float(1 - e.var() / un.var())
    return R, h_dcc, h_rol, h_sta, out


def fig_hedge(save_it=True):
    R, h_dcc, h_rol, h_sta, out = hedge()
    a = '2015-01-01'
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(h_dcc.loc[a:].index, h_dcc.loc[a:], color=st.IDAred, lw=0.7, label='DCC hedge ratio')
    ax.plot(h_rol.loc[a:].index, h_rol.loc[a:], color=st.MainBlue, lw=1.4, label='Rolling OLS (250 days)')
    ax.axhline(h_sta, color=st.Forest, lw=1.5, ls='--', label='Static OLS (2005-2014)')
    ax.set_ylabel('Units of DAX sold per unit of BET')
    st.legend_outside_bottom(ax, ncol=3)
    save('tsa_ch14_hedge', save_it)
    return out


if __name__ == '__main__':
    st.apply()
    N = {}
    only = sys.argv[1:]
    path = os.path.join(HERE, 'ch14_numbers.json')
    if only and os.path.exists(path):
        N = json.load(open(path))
    for name, f in [('ex', worked_examples), ('roll', fig_rolling_corr), ('div', fig_diversification),
                    ('params', fig_param_count), ('step1', fig_step1), ('dcc', fig_dcc_sp_dax), ('zoom', fig_crisis_zoom),
                    ('panel', fig_panel_corr), ('heat', fig_panel_heatmap), ('var', fig_var_backtest), ('hedge', fig_hedge)]:
        if only and name not in only:
            continue
        print(name)
        N[name] = f()
        with open(path, 'w') as fh:
            json.dump(N, fh, indent=1, default=float)
    print('written ch14_numbers.json')
