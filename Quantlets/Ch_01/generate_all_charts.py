"""
generate_all_charts.py -- charts and numbers of Chapter 1 (TSA): stochastic processes and stationarity
=====================================================================================================
Course data (tsa_data.py), chart style (tsa_style.py), statsmodels for the ACF, the PACF and the portmanteau tests.
Every number on the slides comes from here.
  * real series    -- Romanian real GDP (Eurostat, quarterly, not seasonally adjusted), Romanian HICP inflation
                      (Eurostat, monthly), EUR/RON (BNR reference rate), BET and S&P 500 daily log returns;
  * processes      -- an ensemble of paths of a stationary AR(1) and of a random walk; four non-stationary
                      patterns; a weakly but not strictly stationary process; three kinds of white noise;
                      the random walk and its variance; "which series is real?" (BET among simulated random walks);
  * operators      -- S&P 500 log price and log returns; the Wold weights of four processes; ergodicity;
  * ACF and PACF   -- theoretical and sample ACF of four processes; the sampling distribution of the sample ACF
                      under white noise (Bartlett); ACF and PACF of AR(1), AR(2) and MA(1); BET returns;
  * tests          -- size of the Box-Pierce and Ljung-Box tests in small samples; Ljung-Box statistics of
                      real series against the 5% critical values;
  * transformations-- Romanian real GDP: log, quarterly and annual log differences and their ACF; Box-Cox with
                      Guerrero's lambda for Romanian nominal GDP; over-differencing white noise;
  * textbook data  -- the Nile flow (level shift in 1898) and the sunspot numbers (statsmodels data sets).
Output: charts/tsa_ch1_*.pdf/.png, Quantlets/Ch_01/ch1_numbers.json, ch1_real_series.csv
References: Huang and Petukhina (2022), Applied Time Series Analysis and Forecasting with Python, Ch. 1-2;
Hyndman and Athanasopoulos, Forecasting: Principles and Practice (3rd ed.), Ch. 2-3; Brockwell and Davis (2016),
Introduction to Time Series and Forecasting, 3rd ed., Ch. 1.
Run:  python3 Quantlets/Ch_01/generate_all_charts.py
Time Series Analysis - Daniel Traian PELE
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from tsa_data import load_close, log_returns, read_eurostat, load_statsmodels   # noqa: E402
import tsa_style as st                                                          # noqa: E402
from statsmodels.tsa.stattools import acf, pacf                                  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox                          # noqa: E402

SEED = 2026
GDP_REAL = ('namq_10_gdp', 'Q.CLV10_MEUR.NSA.B1GQ.RO')     # chain-linked volumes (2010), million EUR, not adjusted
GDP_NOMINAL = ('namq_10_gdp', 'Q.CP_MEUR.NSA.B1GQ.RO')     # current prices, million EUR, not adjusted
HICP = ('prc_hicp_minr', 'M.I15.TOTAL.RO')                 # harmonised index of consumer prices, 2015 = 100
START = '2000-01-01'                                       # S&P 500 and BET
BAND_COL = st.IDAred                                       # +-1.96/sqrt(T) bands
NAME = {'bet': 'BET', 'sp500': 'S&P 500'}


# =============================================================================
# DATA AND HELPERS
# =============================================================================
def ro_gdp(kind='real'):
    """Romanian quarterly GDP (Eurostat), million EUR, not seasonally adjusted: 'real' or 'nominal'."""
    ds, key = GDP_REAL if kind == 'real' else GDP_NOMINAL
    return read_eurostat(ds, key).rename(f'gdp_{kind}')


def ro_hicp():
    """Romanian HICP (Eurostat, monthly, 2015 = 100)."""
    return read_eurostat(*HICP).rename('hicp')


def sample_acf(x, nlags=20):
    """Sample ACF rho(1..nlags) with the divisor T (as in Brockwell and Davis)."""
    return acf(np.asarray(x, float), nlags=nlags, fft=True)[1:]


def sample_pacf(x, nlags=20):
    """Sample PACF phi_hh, h = 1..nlags (Durbin-Levinson on the sample ACF)."""
    return pacf(np.asarray(x, float), nlags=nlags, method='ywm')[1:]


def ljung_box(x, m=10):
    """Ljung-Box Q*(m) and Box-Pierce Q(m) with their chi-square(m) p-values."""
    t = acorr_ljungbox(np.asarray(x, float), lags=[m], boxpierce=True, return_df=True).iloc[0]
    return {'lb': float(t['lb_stat']), 'lb_p': float(t['lb_pvalue']), 'bp': float(t['bp_stat']), 'bp_p': float(t['bp_pvalue'])}


def acf_bars(ax, r, T, color=st.MainBlue, theory=None, label='sample ACF', band=True):
    """ACF as bars at lags 1..len(r), with the +-1.96/sqrt(T) bands and optional theoretical values."""
    lags = np.arange(1, len(r) + 1)
    ax.bar(lags, r, width=0.55, color=color, label=label)
    if theory is not None:
        ax.plot(lags, theory, 'o', ms=4, color=st.Orange, label='theoretical ACF')
    if band:
        b = 1.96 / np.sqrt(T)
        ax.axhline(b, color=BAND_COL, ls='--', lw=0.9, label=r'$\pm 1.96/\sqrt{T}$')
        ax.axhline(-b, color=BAND_COL, ls='--', lw=0.9, label='_nolegend_')
    ax.axhline(0, color=st.DarkText, lw=0.6)
    ax.set_xlim(0.3, len(r) + 0.7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))


def simulate_arma(phi=(), theta=(), n=500, sigma=1.0, burn=500, rng=None):
    """X_t = sum phi_i X_{t-i} + e_t + sum theta_j e_{t-j}, e_t i.i.d. N(0, sigma^2), after a burn-in."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    e = rng.normal(0, sigma, n + burn)
    x = np.zeros(n + burn)
    p, q = len(phi), len(theta)
    for t in range(n + burn):
        x[t] = e[t] + sum(phi[i] * x[t - 1 - i] for i in range(p) if t - 1 - i >= 0) \
            + sum(theta[j] * e[t - 1 - j] for j in range(q) if t - 1 - j >= 0)
    return x[burn:]


def theoretical_acf(phi=(), theta=(), nlags=20):
    """Theoretical ACF of an ARMA process (statsmodels ArmaProcess), lags 1..nlags."""
    from statsmodels.tsa.arima_process import ArmaProcess
    return ArmaProcess(np.r_[1, -np.asarray(phi, float)], np.r_[1, np.asarray(theta, float)]).acf(nlags + 1)[1:]


def save(name, save_it=True):
    if save_it:
        st.check_no_grey(plt.gcf())
        st.save_fig(name)
    else:
        plt.show()


# =============================================================================
# 1. REAL SERIES
# =============================================================================
def fig_four_series(save_it=True):
    """Four series of the course: Romanian real GDP, Romanian HICP inflation, EUR/RON, BET daily log returns."""
    gdp = ro_gdp('real') / 1000
    hicp = ro_hicp()
    infl = (100 * np.log(hicp).diff(12)).dropna()
    fx = load_close('eurron')
    bet = log_returns('bet', start=START)
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 5.1))
    ax[0, 0].plot(gdp.index, gdp, color=st.Forest)
    ax[0, 0].set_title('Romania: real GDP, quarterly (bn EUR, 2010 prices)')
    ax[0, 1].plot(infl.index, infl, color=st.IDAred)
    ax[0, 1].axhline(0, color=st.DarkText, lw=0.6)
    ax[0, 1].set_title('Romania: HICP inflation, 12-month (%)')
    ax[1, 0].plot(fx.index, fx, color=st.MainBlue)
    ax[1, 0].set_title('EUR/RON, BNR reference rate (daily)')
    ax[1, 1].plot(bet.index, bet, color=st.Amber, lw=0.5)
    ax[1, 1].set_title('BET: daily log returns (%)')
    plt.tight_layout()
    save('tsa_ch1_four_series', save_it)
    out = {'gdp_first': gdp.index[0].strftime('%Y-%m-%d'), 'gdp_last': gdp.index[-1].strftime('%Y-%m-%d'),
           'gdp_n': int(len(gdp)), 'gdp_v0': float(gdp.iloc[0]), 'gdp_v1': float(gdp.iloc[-1]),
           'infl_first': infl.index[0].strftime('%Y-%m-%d'), 'infl_last': infl.index[-1].strftime('%Y-%m-%d'),
           'infl_max': float(infl.max()), 'infl_max_d': infl.idxmax().strftime('%Y-%m-%d'),
           'infl_min': float(infl.min()), 'infl_min_d': infl.idxmin().strftime('%Y-%m-%d'),
           'infl_lastv': float(infl.iloc[-1]),
           'fx_first': fx.index[0].strftime('%Y-%m-%d'), 'fx_v0': float(fx.iloc[0]), 'fx_v1': float(fx.iloc[-1]),
           'fx_n': int(len(fx)), 'bet_n': int(len(bet)), 'bet_sd': float(bet.std()), 'bet_min': float(bet.min()),
           'bet_min_d': bet.idxmin().strftime('%Y-%m-%d'), 'bet_max': float(bet.max()),
           'bet_max_d': bet.idxmax().strftime('%Y-%m-%d'), 'bet_mean': float(bet.mean()),
           'bet_mean_0': float(bet[:'2012'].mean()), 'bet_mean_1': float(bet['2013':].mean()),
           'bet_sd_0': float(bet['2008-09':'2009-03'].std()), 'bet_sd_1': float(bet['2017'].std())}
    return out


# =============================================================================
# 2. PROCESSES
# =============================================================================
def fig_ensemble(n=100, paths=40, phi=0.7, save_it=True):
    """An ensemble of paths: stationary AR(1) and random walk, with the +-1.96 sd band of X_t."""
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    t = np.arange(1, n + 1)
    sd_ar = 1 / np.sqrt(1 - phi ** 2)
    for i in range(paths):
        e = rng.normal(size=n)
        x = np.empty(n)
        x[0] = rng.normal(0, sd_ar)
        for k in range(1, n):
            x[k] = phi * x[k - 1] + e[k]
        ax[0].plot(t, x, color=st.MainBlue, lw=0.6, alpha=0.45, label='one path' if i == 0 else '_nolegend_')
        ax[1].plot(t, np.cumsum(rng.normal(size=n)), color=st.Forest, lw=0.6, alpha=0.45,
                   label='one path' if i == 0 else '_nolegend_')
    ax[0].fill_between(t, -1.96 * sd_ar, 1.96 * sd_ar, color=st.Orange, alpha=0.18, label=r'$\pm 1.96\,$sd$(X_t)$')
    ax[1].fill_between(t, -1.96 * np.sqrt(t), 1.96 * np.sqrt(t), color=st.Orange, alpha=0.18, label='_nolegend_')
    ax[0].set_title(rf'Stationary AR(1), $\phi = {phi}$: same band at every $t$')
    ax[1].set_title(r'Random walk: the band widens like $\sqrt{t}$')
    for a in ax:
        a.set_xlabel('t')
    st.fig_legend_bottom(fig, ncol=2, y=-0.01)
    plt.tight_layout()
    save('tsa_ch1_ensemble', save_it)
    # cross-sectional variance at t = n from many paths
    big = rng.normal(size=(5000, n))
    rw = big.cumsum(axis=1)
    return {'ar_var': 1 / (1 - phi ** 2), 'ar_sd': sd_ar, 'rw_var_sim': float(rw[:, -1].var()), 'n': n, 'paths': paths}


def fig_nonstationary(n=300, save_it=True):
    """Four series: stationary AR(1), linear trend, changing variance, level shift (room question)."""
    rng = np.random.default_rng(SEED + 1)
    t = np.arange(n)
    a = simulate_arma([0.5], n=n, rng=rng)
    b = 0.04 * t + simulate_arma([0.5], n=n, rng=rng)
    c = simulate_arma([0.5], n=n, rng=rng) * np.linspace(0.4, 3.0, n)
    d = simulate_arma([0.5], n=n, rng=rng) + np.where(t >= n // 2, 4.0, 0.0)
    fig, ax = plt.subplots(2, 2, figsize=(9.6, 4.5), sharex=True)
    for axx, x, lab, col in zip(ax.ravel(), [a, b, c, d], ['Series A', 'Series B', 'Series C', 'Series D'],
                                [st.MainBlue, st.IDAred, st.Forest, st.Purple]):
        axx.plot(t, x, color=col, lw=0.9)
        axx.set_title(lab)
    plt.tight_layout()
    save('tsa_ch1_nonstationary', save_it)
    return {'slope': 0.04, 'sd0': 0.4, 'sd1': 3.0, 'shift': 4.0, 'n': n}


def fig_counterexample(n=4000, save_it=True):
    """Independent X_t: N(0,1) for even t, (chi2(5) - 5)/sqrt(10) for odd t: weakly, not strictly stationary."""
    rng = np.random.default_rng(SEED + 2)
    x = np.empty(n)
    x[0::2] = rng.normal(size=n // 2)
    x[1::2] = (rng.chisquare(5, size=n // 2) - 5) / np.sqrt(10)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2))
    ax[0].plot(np.arange(100), x[:100], color=st.MainBlue, lw=0.9)
    ax[0].set_title('First 100 observations')
    bins = np.linspace(-4, 6, 50)
    ax[1].hist(x[0::2], bins=bins, density=True, alpha=0.55, color=st.MainBlue, label='even t: N(0, 1)')
    ax[1].hist(x[1::2], bins=bins, density=True, alpha=0.55, color=st.IDAred,
               label=r'odd t: $(\chi^2_5 - 5)/\sqrt{10}$')
    ax[1].set_title('Distribution of $X_t$ for even and odd $t$')
    st.legend_outside_bottom(ax[1], ncol=2, y=-0.14)
    plt.tight_layout()
    save('tsa_ch1_counterexample', save_it)
    ev, od = x[0::2], x[1::2]
    return {'mean_e': float(ev.mean()), 'mean_o': float(od.mean()), 'var_e': float(ev.var()), 'var_o': float(od.var()),
            'skew_e': float(stats.skew(ev)), 'skew_o': float(stats.skew(od)), 'skew_th': float(np.sqrt(8 / 5)),
            'rho1': float(sample_acf(x, 1)[0])}


def simulate_garch(n, omega=0.05, alpha=0.10, beta=0.85, burn=500, rng=None):
    """GARCH(1,1) with Normal innovations: a weak white noise (uncorrelated, not independent)."""
    rng = rng if rng is not None else np.random.default_rng(SEED)
    z = rng.normal(size=n + burn)
    e = np.zeros(n + burn)
    s2 = np.full(n + burn, omega / (1 - alpha - beta))
    for t in range(1, n + burn):
        s2[t] = omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * z[t]
    return e[burn:]


def fig_white_noise(n=1000, save_it=True):
    """Gaussian white noise, i.i.d. Laplace white noise (heavier tails) and GARCH(1,1) weak white noise (variance 1),
    with the ACF of the series and of the squares."""
    rng = np.random.default_rng(SEED + 3)
    w = {'Gaussian i.i.d.': rng.normal(size=n),
         'Laplace i.i.d.': rng.laplace(0, 1 / np.sqrt(2), size=n),
         'GARCH(1,1): weak WN': simulate_garch(n, rng=rng)}
    cols = [st.MainBlue, st.Forest, st.IDAred]
    fig, ax = plt.subplots(2, 3, figsize=(7.4, 3.4))
    out = {}
    for j, ((lab, x), c) in enumerate(zip(w.items(), cols)):
        ax[0, j].plot(x[:500], color=c, lw=0.6)
        ax[0, j].set_title(lab)
        ax[0, j].set_ylim(-7, 7)
        acf_bars(ax[1, j], sample_acf(x ** 2, 20), n, color=c, label='ACF of squares' if j == 0 else '_nolegend_')
        ax[1, j].set_title('ACF of $X_t^2$')
        ax[1, j].set_ylim(-0.12, 0.35)
        out[lab] = {'lb_x': ljung_box(x, 10), 'lb_x2': ljung_box(x ** 2, 10), 'kurt': float(stats.kurtosis(x) + 3)}
    st.fig_legend_bottom(fig, ncol=3, y=-0.01)
    plt.tight_layout()
    save('tsa_ch1_white_noise', save_it)
    return out


def fig_random_walk(n=250, paths=60, nsim=5000, save_it=True):
    """Random walk paths with the +-1.96 sqrt(t) band, and the cross-sectional variance of X_t against t."""
    rng = np.random.default_rng(SEED + 4)
    x = rng.normal(size=(nsim, n)).cumsum(axis=1)
    t = np.arange(1, n + 1)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for i in range(paths):
        ax[0].plot(t, x[i], lw=0.5, alpha=0.5, color=st.PALETTE[i % 3], label='_nolegend_')
    ax[0].plot(t, 1.96 * np.sqrt(t), color=st.IDAred, ls='--', lw=1.2, label=r'$\pm 1.96\sqrt{t}$')
    ax[0].plot(t, -1.96 * np.sqrt(t), color=st.IDAred, ls='--', lw=1.2, label='_nolegend_')
    ax[0].set_title(f'{paths} random walks, $\\sigma = 1$')
    ax[0].set_xlabel('t')
    ax[1].plot(t, x.var(axis=0), color=st.MainBlue, label=f'variance across {nsim} paths')
    ax[1].plot(t, t, color=st.Orange, ls='--', label=r'theory: $t\sigma^2$')
    ax[1].set_title(r'$\mathrm{Var}(X_t)$ grows linearly in $t$')
    ax[1].set_xlabel('t')
    st.fig_legend_bottom(fig, ncol=3, y=-0.01)
    plt.tight_layout()
    save('tsa_ch1_random_walk', save_it)
    c = np.corrcoef(x[:, 99], x[:, 109])[0, 1]
    return {'var_end': float(x[:, -1].var()), 'n': n, 'corr_100_110': float(c), 'corr_th': float(np.sqrt(100 / 110)),
            'share_out': float(np.mean(np.abs(x[:, -1]) > 1.96 * np.sqrt(n)))}


def fig_spot_real(k='bet', n=500, save_it=True):
    """Which one is real? The BET log price over its last n days among five random walks with the same drift and
    the same standard deviation of the daily changes."""
    p = np.log(load_close(k)).iloc[-n - 1:]
    r = p.diff().dropna()
    rng = np.random.default_rng(SEED + 5)
    pos = int(rng.integers(0, 6))
    fig, ax = plt.subplots(2, 3, figsize=(9.6, 4.3), sharey=False)
    letters = 'ABCDEF'
    for i, a in enumerate(ax.ravel()):
        if i == pos:
            y = p.values - p.values[0]
        else:
            y = np.r_[0, np.cumsum(r.mean() + r.std() * rng.normal(size=n))]
        a.plot(np.arange(n + 1), 100 * y, color=st.MainBlue, lw=0.8)
        a.set_title(f'Series {letters[i]}')
        a.set_xticks([])
    fig.supylabel('change in log price since day 0 (%)', fontsize=11)
    plt.tight_layout()
    save('tsa_ch1_spot_real', save_it)
    return {'real': letters[pos], 'first': p.index[0].strftime('%Y-%m-%d'), 'last': p.index[-1].strftime('%Y-%m-%d'),
            'mu': float(100 * r.mean()), 'sd': float(100 * r.std()), 'n': n,
            'rho1': float(sample_acf(r, 1)[0]), 'lb10_p': ljung_box(r, 10)['lb_p']}


# =============================================================================
# 3. LAG OPERATOR, DIFFERENCING, WOLD, ERGODICITY
# =============================================================================
def fig_sp500_diff(save_it=True):
    """S&P 500: log price (a random-walk-like series) and daily log returns (its first difference)."""
    p = np.log(load_close('sp500', start=START))
    r = 100 * p.diff().dropna()
    fig, ax = plt.subplots(2, 1, figsize=(9.6, 4.3), sharex=True)
    ax[0].plot(p.index, p, color=st.MainBlue)
    ax[0].set_title(r'S&P 500: $\ln P_t$')
    ax[1].plot(r.index, r, color=st.IDAred, lw=0.5)
    ax[1].set_title(r'S&P 500: $r_t = 100\,\Delta \ln P_t = 100\,(1 - L)\ln P_t$ (%)')
    plt.tight_layout()
    save('tsa_ch1_sp500_diff', save_it)
    return {'n': int(len(r)), 'acf1_p': float(sample_acf(p, 1)[0]), 'acf50_p': float(sample_acf(p, 50)[-1]),
            'acf1_r': float(sample_acf(r, 1)[0]), 'mean_r': float(r.mean()), 'sd_r': float(r.std())}


def fig_wold(J=15, save_it=True):
    """Wold weights psi_j of AR(1) (phi = 0.8 and -0.6), MA(1) (theta = 0.6) and of the random walk (psi_j = 1)."""
    j = np.arange(J + 1)
    cases = [(r'AR(1), $\phi = 0.8$: $\psi_j = 0.8^j$', 0.8 ** j, st.MainBlue),
             (r'AR(1), $\phi = -0.6$: $\psi_j = (-0.6)^j$', (-0.6) ** j, st.Forest),
             (r'MA(1), $\theta = 0.6$: $\psi_1 = 0.6$, then 0', np.r_[1, 0.6, np.zeros(J - 1)], st.Purple),
             (r'Random walk: $\psi_j = 1$ (no decay)', np.ones(J + 1), st.IDAred)]
    fig, ax = plt.subplots(1, 4, figsize=(10.4, 2.7), sharey=True)
    for a, (lab, psi, c) in zip(ax, cases):
        a.bar(j, psi, color=c, width=0.6)
        a.axhline(0, color=st.DarkText, lw=0.6)
        a.set_title(lab, fontsize=11)
        a.set_xlabel('j')
    plt.tight_layout()
    save('tsa_ch1_wold', save_it)
    return {'sum_psi2_08': 1 / (1 - 0.64), 'psi5_08': 0.8 ** 5, 'psi10_08': 0.8 ** 10}


def fig_ergodicity(n=2000, paths=6, save_it=True):
    """Running time averages: ergodic AR(1) (all converge to 0) and the non-ergodic X_t = Z + e_t (each path
    converges to its own Z)."""
    rng = np.random.default_rng(SEED + 6)
    t = np.arange(1, n + 1)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2), sharey=True)
    ends_a, ends_b, zs = [], [], []
    for i in range(paths):
        x = simulate_arma([0.7], n=n, rng=rng)
        ax[0].plot(t, np.cumsum(x) / t, lw=0.9, color=st.PALETTE[i % len(st.PALETTE)])
        ends_a.append(float(x.mean()))
        z = rng.normal()
        y = z + rng.normal(size=n)
        ax[1].plot(t, np.cumsum(y) / t, lw=0.9, color=st.PALETTE[i % len(st.PALETTE)])
        ends_b.append(float(y.mean()))
        zs.append(float(z))
    for a in ax:
        a.axhline(0, color=st.DarkText, ls='--', lw=0.8)
        a.set_xscale('log')
        a.set_xlabel('T (log scale)')
    ax[0].set_title(r'AR(1), $\phi = 0.7$: $\bar X_T \to \mu = 0$ on every path')
    ax[1].set_title(r'$X_t = Z + \varepsilon_t$: $\bar X_T \to Z$, not $\mu = 0$')
    ax[0].set_ylabel(r'$\bar X_T$')
    plt.tight_layout()
    save('tsa_ch1_ergodicity', save_it)
    return {'a_min': min(ends_a), 'a_max': max(ends_a), 'b_min': min(ends_b), 'b_max': max(ends_b),
            'z_min': min(zs), 'z_max': max(zs), 'n': n}


# =============================================================================
# 4. ACF AND PACF
# =============================================================================
def fig_acf_models(n=500, nlags=20, save_it=True):
    """Sample ACF (T = 500) and theoretical ACF of white noise, AR(1) (0.8), MA(1) (0.6) and a random walk."""
    rng = np.random.default_rng(SEED + 7)
    cases = [('White noise', rng.normal(size=n), np.zeros(nlags)),
             (r'AR(1), $\phi = 0.8$', simulate_arma([0.8], n=n, rng=rng), theoretical_acf([0.8], nlags=nlags)),
             (r'MA(1), $\theta = 0.6$', simulate_arma([], [0.6], n=n, rng=rng), theoretical_acf([], [0.6], nlags=nlags)),
             ('Random walk', np.cumsum(rng.normal(size=n)), None)]
    fig, ax = plt.subplots(1, 4, figsize=(10.4, 2.9), sharey=True)
    out = {}
    for i, (a, (lab, x, th)) in enumerate(zip(ax, cases)):
        r = sample_acf(x, nlags)
        acf_bars(a, r, n, theory=th)
        a.set_title(lab)
        a.set_xlabel('lag h')
        out[lab.split(',')[0].replace('$', '')] = {'r1': float(r[0]), 'r2': float(r[1]), 'r10': float(r[9])}
    ax[0].set_ylim(-0.4, 1.05)
    st.fig_legend_bottom(fig, ncol=3, y=-0.03)
    plt.tight_layout()
    save('tsa_ch1_acf_models', save_it)
    out['th_ma1'] = 0.6 / 1.36
    return out


def fig_bartlett(T=100, nsim=10000, nlags=20, save_it=True):
    """Sample ACF of Gaussian white noise: the distribution of rho_hat(1) against N(0, 1/T), and the number of
    lags (out of 20) outside +-1.96/sqrt(T) against the Binomial(20, 0.05) distribution."""
    rng = np.random.default_rng(SEED + 8)
    r1, outside = np.empty(nsim), np.empty(nsim, int)
    b = 1.96 / np.sqrt(T)
    for i in range(nsim):
        x = rng.normal(size=T)
        r = sample_acf(x, nlags)
        r1[i] = r[0]
        outside[i] = int(np.sum(np.abs(r) > b))
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.2))
    ax[0].hist(r1, bins=60, density=True, color=st.MainBlue, alpha=0.6, label=rf'$\hat\rho(1)$, {nsim} samples of T = {T}')
    g = np.linspace(-0.45, 0.45, 300)
    ax[0].plot(g, stats.norm.pdf(g, 0, 1 / np.sqrt(T)), color=st.Orange, lw=1.8, label='N(0, 1/T)')
    for s in (-1, 1):
        ax[0].axvline(s * b, color=BAND_COL, ls='--', lw=0.9, label=r'$\pm 1.96/\sqrt{T}$' if s == 1 else '_nolegend_')
    ax[0].set_title(r'White noise: $\hat\rho(1) \approx N(0, 1/T)$')
    k = np.arange(0, 8)
    emp = np.array([np.mean(outside == v) for v in k])
    ax[1].bar(k - 0.18, emp, width=0.36, color=st.MainBlue, label='simulated share')
    ax[1].bar(k + 0.18, stats.binom.pmf(k, nlags, 0.05), width=0.36, color=st.Orange, label='Binomial(20, 0.05)')
    ax[1].set_title('Number of lags (of 20) outside the bands')
    ax[1].set_xlabel('lags outside')
    st.fig_legend_bottom(fig, ncol=5, y=-0.01)
    plt.tight_layout()
    save('tsa_ch1_bartlett', save_it)
    return {'sd_r1': float(r1.std()), 'sd_th': 1 / np.sqrt(T), 'mean_r1': float(r1.mean()), 'mean_th': -1 / T,
            'p_none': float(np.mean(outside == 0)), 'p_none_th': 0.95 ** nlags, 'p_one_plus': float(np.mean(outside >= 1)),
            'mean_out': float(outside.mean()), 'T': T, 'band': b}


def fig_acf_pacf(n=500, nlags=20, save_it=True):
    """Sample ACF and PACF (T = 500) of AR(1) (phi = 0.7), AR(2) (0.5, 0.3) and MA(1) (theta = 0.7)."""
    rng = np.random.default_rng(SEED + 9)
    cases = [(r'AR(1), $\phi = 0.7$', [0.7], []), (r'AR(2), $\phi_1 = 0.5$, $\phi_2 = 0.3$', [0.5, 0.3], []),
             (r'MA(1), $\theta = 0.7$', [], [0.7])]
    fig, ax = plt.subplots(2, 3, figsize=(7.4, 3.4), sharey=True)
    out = {}
    for j, (lab, ph, th) in enumerate(cases):
        x = simulate_arma(ph, th, n=n, rng=rng)
        r, p = sample_acf(x, nlags), sample_pacf(x, nlags)
        acf_bars(ax[0, j], r, n, color=st.MainBlue, label='sample ACF')
        acf_bars(ax[1, j], p, n, color=st.Forest, label='sample PACF')
        ax[0, j].set_title(lab + ': ACF', fontsize=10.5)
        ax[1, j].set_title('PACF', fontsize=10.5)
        ax[1, j].set_xlabel('lag h')
        key = ['ar1', 'ar2', 'ma1'][j]
        out[key] = {'r1': float(r[0]), 'r2': float(r[1]), 'p1': float(p[0]), 'p2': float(p[1]), 'p3': float(p[2])}
    ax[0, 0].set_ylim(-0.45, 1.0)
    st.fig_legend_bottom(fig, ncol=3, y=-0.01)
    plt.tight_layout()
    save('tsa_ch1_acf_pacf', save_it)
    out['ar2_rho1'] = 0.5 / (1 - 0.3)
    out['ar2_rho2'] = 0.5 * 0.5 / (1 - 0.3) + 0.3
    return out


def fig_returns_acf(k='bet', nlags=50, save_it=True):
    """ACF of daily log returns, absolute returns and squared returns (lags 1-50), with Ljung-Box tests."""
    r = log_returns(k, start=START)
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.0), sharey=True)
    out = {'n': int(len(r)), 'first': r.index[0].strftime('%Y-%m-%d')}
    for i, (a, (lab, x, c)) in enumerate(zip(ax, [('$r_t$', r, st.MainBlue), (r'$|r_t|$', r.abs(), st.Forest),
                                                  ('$r_t^2$', r ** 2, st.IDAred)])):
        rr = sample_acf(x, nlags)
        acf_bars(a, rr, len(r), color=c, label='_nolegend_')
        a.set_title(f'{NAME.get(k, k)}: ACF of {lab}')
        a.set_xlabel('lag h')
        out[['r', 'abs', 'sq'][i]] = {'r1': float(rr[0]), 'r5': float(rr[4]), 'r50': float(rr[-1]),
                                      'lb10': ljung_box(x, 10)}
    ax[0].set_ylim(-0.1, 0.45)
    st.fig_legend_bottom(fig, ncol=1, y=-0.03)
    plt.tight_layout()
    save(f'tsa_ch1_{k}_acf', save_it)
    out['band'] = 1.96 / np.sqrt(len(r))
    return out


# =============================================================================
# 5. PORTMANTEAU TESTS
# =============================================================================
def lb_size(T=50, m=20, nsim=5000, seed=SEED + 10):
    """Rejection rates at 5% of the Box-Pierce and Ljung-Box tests for Gaussian white noise."""
    rng = np.random.default_rng(seed)
    crit = stats.chi2.ppf(0.95, m)
    rej_bp = rej_lb = 0
    h = np.arange(1, m + 1)
    for _ in range(nsim):
        r = sample_acf(rng.normal(size=T), m)
        q_bp = T * np.sum(r ** 2)
        q_lb = T * (T + 2) * np.sum(r ** 2 / (T - h))
        rej_bp += q_bp > crit
        rej_lb += q_lb > crit
    return rej_bp / nsim, rej_lb / nsim


def fig_lb_size(save_it=True):
    """Size of Box-Pierce and Ljung-Box at the 5% level for T = 50, 100, 500 and m = 10, 20."""
    cases = [(50, 10), (50, 20), (100, 10), (100, 20), (500, 10), (500, 20)]
    res = {f'{T}_{m}': lb_size(T, m) for T, m in cases}
    fig, ax = plt.subplots(figsize=(8.8, 3.2))
    x = np.arange(len(cases))
    ax.bar(x - 0.18, [100 * res[f'{T}_{m}'][0] for T, m in cases], width=0.36, color=st.MainBlue, label='Box-Pierce Q')
    ax.bar(x + 0.18, [100 * res[f'{T}_{m}'][1] for T, m in cases], width=0.36, color=st.Orange, label='Ljung-Box Q*')
    ax.axhline(5, color=BAND_COL, ls='--', lw=1.0, label='nominal 5%')
    ax.set_xticks(x)
    ax.set_xticklabels([f'T = {T}, m = {m}' for T, m in cases])
    ax.set_ylabel('rejections of a true $H_0$ (%)')
    ax.set_title('Gaussian white noise, 5000 samples each')
    st.legend_outside_bottom(ax, ncol=3, y=-0.14)
    plt.tight_layout()
    save('tsa_ch1_lb_size', save_it)
    return {k: {'bp': v[0], 'lb': v[1]} for k, v in res.items()}


def real_series():
    """The real series of the chapter, transformed to (roughly) stationary form."""
    gdp = np.log(ro_gdp('real'))
    hicp = np.log(ro_hicp())
    nile = load_statsmodels('nile')
    return {'S&P 500 returns': log_returns('sp500', start=START),
            'S&P 500 squared returns': log_returns('sp500', start=START) ** 2,
            'BET returns': log_returns('bet', start=START),
            'BET squared returns': log_returns('bet', start=START) ** 2,
            'EUR/RON returns': (100 * np.log(load_close('eurron')).diff()).dropna(),
            'GDP growth, y/y': (100 * gdp.diff(4)).dropna(),
            'GDP growth, q/q': (100 * gdp.diff()).dropna(),
            'HICP inflation, m/m': (100 * hicp.diff())['2005':].dropna(),
            'Nile flow': nile,
            'Sunspots': load_statsmodels('sunspots')}


def fig_lb_stats(m_max=30, save_it=True):
    """Ljung-Box Q*(m), m = 1..30, of six series against the 5% critical value of chi-square(m)."""
    R = real_series()
    keys = ['S&P 500 returns', 'BET returns', 'EUR/RON returns', 'S&P 500 squared returns', 'BET squared returns',
            'GDP growth, q/q']
    cols = [st.MainBlue, st.IDAred, st.Forest, st.Teal, st.Orange, st.Purple]
    m = np.arange(1, m_max + 1)
    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    for k, c in zip(keys, cols):
        q = acorr_ljungbox(np.asarray(R[k], float), lags=list(m), return_df=True)['lb_stat'].values
        ax.plot(m, q, color=c, lw=1.6, label=k)
    ax.plot(m, stats.chi2.ppf(0.95, m), color=st.DarkText, ls='--', lw=1.2, label=r'5% critical value $\chi^2_{0.95}(m)$')
    ax.set_yscale('log')
    ax.set_xlabel('number of lags m')
    ax.set_ylabel('Ljung-Box $Q^*(m)$ (log scale)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.17)
    plt.tight_layout()
    save('tsa_ch1_lb_stats', save_it)
    table = {}
    for k, x in R.items():
        x = np.asarray(x, float)
        r = sample_acf(x, 12)
        table[k] = {'n': int(len(x)), 'r1': float(r[0]), 'r2': float(r[1]), 'r4': float(r[3]), 'r12': float(r[11]),
                    **{f'm{mm}': ljung_box(x, mm) for mm in (10, 20)}}
    pd.DataFrame({k: {'T': v['n'], 'rho1': v['r1'], 'Q*(10)': v['m10']['lb'], 'p': v['m10']['lb_p']}
                  for k, v in table.items()}).T.to_csv(os.path.join(globals().get('OUT_DIR', globals().get('HERE', '.')),
                                                    'ch1_real_series.csv'), float_format='%.4f')
    return table


# =============================================================================
# 6. TRANSFORMATIONS
# =============================================================================
def fig_gdp_transform(nlags=16, save_it=True):
    """Romanian real GDP (not seasonally adjusted): log level, quarterly and annual log differences, and their ACF."""
    g = np.log(ro_gdp('real'))
    d1 = (100 * g.diff()).dropna()
    d4 = (100 * g.diff(4)).dropna()
    fig, ax = plt.subplots(2, 3, figsize=(7.4, 3.4))
    for j, (lab, x, c) in enumerate([(r'$\ln Y_t$', g, st.MainBlue), (r'$100\,\Delta \ln Y_t$ (q/q, %)', d1, st.Forest),
                                     (r'$100\,\Delta_4 \ln Y_t$ (y/y, %)', d4, st.IDAred)]):
        ax[0, j].plot(x.index, x, color=c, lw=1.0)
        ax[0, j].set_title(lab, fontsize=11)
        import matplotlib.dates as mdates
        ax[0, j].xaxis.set_major_locator(mdates.YearLocator(10))
        ax[0, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        acf_bars(ax[1, j], sample_acf(x, nlags), len(x), color=c, label='_nolegend_')
        ax[1, j].set_title('ACF')
        ax[1, j].set_xlabel('lag (quarters)')
        ax[1, j].set_ylim(-0.6, 1.0)
    st.fig_legend_bottom(fig, ncol=1, y=-0.01)
    plt.tight_layout()
    save('tsa_ch1_gdp_transform', save_it)
    r0, r1, r4 = sample_acf(g, nlags), sample_acf(d1, nlags), sample_acf(d4, nlags)
    covid = d4['2020-04-01'] if pd.Timestamp('2020-04-01') in d4.index else float('nan')
    return {'n': int(len(g)), 'first': g.index[0].strftime('%Y-%m-%d'), 'last': g.index[-1].strftime('%Y-%m-%d'),
            'acf1_level': float(r0[0]), 'acf8_level': float(r0[7]), 'acf1_d1': float(r1[0]), 'acf4_d1': float(r1[3]),
            'acf2_d1': float(r1[1]), 'acf1_d4': float(r4[0]), 'acf4_d4': float(r4[3]), 'mean_d4': float(d4.mean()),
            'sd_d4': float(d4.std()), 'covid_d4': float(covid), 'mean_d1': float(d1.mean()), 'sd_d1': float(d1.std()),
            'last_d4': float(d4.iloc[-1]), 'band': 1.96 / np.sqrt(len(d1))}


def boxcox(y, lam):
    """Box-Cox transform: (y^lam - 1)/lam, or ln y for lam = 0."""
    y = np.asarray(y, float)
    return np.log(y) if abs(lam) < 1e-8 else (y ** lam - 1) / lam


def guerrero_cv(y, lam, period=4):
    """Guerrero (1993): coefficient of variation of sd_i / mean_i^(1 - lam) over non-overlapping blocks of one year."""
    y = np.asarray(y, float)
    nb = len(y) // period
    blocks = y[len(y) - nb * period:].reshape(nb, period)
    ratio = blocks.std(axis=1, ddof=1) / blocks.mean(axis=1) ** (1 - lam)
    return ratio.std(ddof=1) / ratio.mean()


def guerrero_lambda(y, period=4, grid=np.linspace(-1, 2, 301)):
    cv = np.array([guerrero_cv(y, l, period) for l in grid])
    return float(grid[np.argmin(cv)]), grid, cv


def fig_boxcox(save_it=True):
    """Romanian nominal GDP (not seasonally adjusted): the level, the Box-Cox transform with Guerrero's lambda,
    and the Guerrero criterion as a function of lambda."""
    y = ro_gdp('nominal')
    lam, grid, cv = guerrero_lambda(y.values)
    z = pd.Series(boxcox(y.values, lam), index=y.index)
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.0))
    ax[0].plot(y.index, y / 1000, color=st.MainBlue)
    ax[0].set_title('Nominal GDP (bn EUR): $Y_t$')
    ax[1].plot(z.index, z, color=st.Forest)
    ax[1].set_title(rf'Box-Cox, $\lambda = {lam:.2f}$')
    ax[2].plot(grid, cv, color=st.IDAred)
    ax[2].axvline(lam, color=st.DarkText, ls='--', lw=0.8)
    ax[2].set_title(r"Guerrero's criterion against $\lambda$")
    ax[2].set_xlabel(r'$\lambda$')
    plt.tight_layout()
    save('tsa_ch1_boxcox', save_it)
    # seasonal amplitude: range within each year, early and late, in levels and after the transform
    yy = y.groupby(y.index.year).agg(lambda s: (s.max() - s.min()) / s.mean() if len(s) == 4 else np.nan).dropna()
    return {'lambda': lam, 'cv_lam': float(cv.min()), 'cv_1': float(guerrero_cv(y.values, 1.0)),
            'cv_0': float(guerrero_cv(y.values, 0.0)), 'first': y.index[0].strftime('%Y-%m-%d'),
            'last': y.index[-1].strftime('%Y-%m-%d'), 'v0': float(y.iloc[0] / 1000), 'v1': float(y.iloc[-1] / 1000),
            'amp_rel_0': float(yy.iloc[:5].mean()), 'amp_rel_1': float(yy.iloc[-5:].mean())}


def fig_overdiff(n=500, nlags=10, save_it=True):
    """Over-differencing: white noise has no autocorrelation; its first difference is an MA(1) with rho(1) = -0.5."""
    rng = np.random.default_rng(SEED + 11)
    e = rng.normal(size=n + 1)
    d = np.diff(e)
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 2.9), sharey=True)
    acf_bars(ax[0], sample_acf(e, nlags), n, color=st.MainBlue, label='sample ACF')
    acf_bars(ax[1], sample_acf(d, nlags), n, color=st.IDAred, theory=np.r_[-0.5, np.zeros(nlags - 1)], label='_nolegend_')
    ax[0].set_title(r'White noise $\varepsilon_t$')
    ax[1].set_title(r'Over-differenced: $\Delta\varepsilon_t = \varepsilon_t - \varepsilon_{t-1}$')
    for a in ax:
        a.set_xlabel('lag h')
    ax[0].set_ylim(-0.65, 0.3)
    st.fig_legend_bottom(fig, ncol=3, y=-0.03)
    plt.tight_layout()
    save('tsa_ch1_overdiff', save_it)
    return {'r1_e': float(sample_acf(e, 1)[0]), 'r1_d': float(sample_acf(d, 1)[0]), 'var_e': float(e.var()),
            'var_d': float(d.var())}


# =============================================================================
# 7. TEXTBOOK SERIES (statsmodels)
# =============================================================================
def fig_textbook(nlags=40, save_it=True):
    """The Nile flow at Aswan (1871-1970, level shift in 1898) and the yearly sunspot numbers (1700-2008),
    with their sample ACF; for the Nile, also the ACF after removing the two means."""
    nile = load_statsmodels('nile')
    sun = load_statsmodels('sunspots')
    brk = pd.Timestamp('1899-01-01')
    m0, m1 = nile[nile.index < brk].mean(), nile[nile.index >= brk].mean()
    resid = nile - np.where(nile.index < brk, m0, m1)
    fig, ax = plt.subplots(2, 2, figsize=(10.4, 4.6))
    ax[0, 0].plot(nile.index, nile, color=st.MainBlue)
    ax[0, 0].plot(nile.index, np.where(nile.index < brk, m0, m1), color=st.IDAred, ls='--', label='mean before and after 1898')
    ax[0, 0].set_title('Nile at Aswan: yearly flow (10$^8$ m$^3$)')
    st.legend_outside_bottom(ax[0, 0], ncol=1, y=-0.12)
    rn, rr = sample_acf(nile, 20), sample_acf(resid, 20)
    lags = np.arange(1, 21)
    ax[1, 0].bar(lags - 0.18, rn, width=0.36, color=st.MainBlue, label='ACF of the flow')
    ax[1, 0].bar(lags + 0.18, rr, width=0.36, color=st.Orange, label='ACF after removing the two means')
    b = 1.96 / np.sqrt(len(nile))
    ax[1, 0].axhline(b, color=BAND_COL, ls='--', lw=0.9)
    ax[1, 0].axhline(-b, color=BAND_COL, ls='--', lw=0.9)
    ax[1, 0].axhline(0, color=st.DarkText, lw=0.6)
    ax[1, 0].set_xlabel('lag (years)')
    ax[1, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    st.legend_outside_bottom(ax[1, 0], ncol=2, y=-0.3)
    ax[0, 1].plot(sun.index, sun, color=st.Amber)
    ax[0, 1].set_title('Yearly sunspot numbers')
    rs = sample_acf(sun, nlags)
    acf_bars(ax[1, 1], rs, len(sun), color=st.Amber, label='_nolegend_', band=True)
    ax[1, 1].set_xlabel('lag (years)')
    plt.tight_layout()
    save('tsa_ch1_textbook', save_it)
    pk = int(np.argmax(rs[5:15]) + 6)
    return {'m0': float(m0), 'm1': float(m1), 'r1_nile': float(rn[0]), 'r10_nile': float(rn[9]),
            'r1_resid': float(rr[0]), 'n_nile': int(len(nile)), 'n_sun': int(len(sun)), 'sun_peak_lag': pk,
            'sun_peak': float(rs[pk - 1]), 'sun_r1': float(rs[0]), 'sun_min_lag': int(np.argmin(rs[:10]) + 1),
            'sun_min': float(rs[:10].min()), 'lb_nile': ljung_box(nile, 10), 'lb_resid': ljung_box(resid, 10)}


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    st.apply()
    N = {}
    for name, f in [('four', fig_four_series), ('ensemble', fig_ensemble), ('nonstat', fig_nonstationary),
                    ('counter', fig_counterexample), ('wn', fig_white_noise), ('rw', fig_random_walk),
                    ('spot', fig_spot_real), ('sp500', fig_sp500_diff), ('wold', fig_wold), ('ergo', fig_ergodicity),
                    ('acfm', fig_acf_models), ('bartlett', fig_bartlett), ('acfpacf', fig_acf_pacf),
                    ('bet', fig_returns_acf), ('lbsize', fig_lb_size), ('real', fig_lb_stats),
                    ('gdp', fig_gdp_transform), ('boxcox', fig_boxcox), ('overdiff', fig_overdiff),
                    ('textbook', fig_textbook)]:
        print(name)
        N[name] = f()
    with open(os.path.join(HERE, 'ch1_numbers.json'), 'w') as fh:
        json.dump(N, fh, indent=1, default=float)
    print('written ch1_numbers.json')
