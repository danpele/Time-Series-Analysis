"""
generate_all_charts.py -- charts and numbers of Chapter 5 (TSA): conditional volatility, ARCH and GARCH models
==============================================================================================================
Course data (tsa_data.py), chart style (tsa_style.py), the arch package for maximum-likelihood estimation.
Every number on the slides comes from here. Four daily series: S&P 500 and BET (since 2000), EUR/RON (BNR reference
rate, since July 2005) and Bitcoin (since September 2014).
  * stylised facts -- daily returns; moments, Ljung-Box tests on returns and squared returns, the ARCH-LM test
                      (Engle, 1982); ACF of returns and squared returns;
  * mean model     -- an AR model for the mean (Chapter 2) leaves residuals whose squares are autocorrelated;
  * simulation     -- i.i.d. Normal, ARCH(1) and GARCH(1,1) paths with the same unconditional variance; the
                      log-likelihood of a simulated ARCH(1) and of a simulated GARCH(1,1);
  * estimation     -- ARCH(q) and GARCH(1,1) on the S&P 500; Gaussian GARCH(1,1) step by step (scipy) and with arch;
                      classic and robust standard errors; Normal, Student-t and skewed-t innovations; QQ plots;
  * four markets   -- GARCH(1,1)-t: persistence, half-life, long-run volatility, conditional volatility;
                      EWMA (RiskMetrics) against GARCH in 2020;
  * ARMA-GARCH     -- AR(1)-GARCH(1,1)-t: the joint model of the conditional mean and the conditional variance;
                      constant and time-varying 95% intervals for the BET;
  * asymmetry      -- GJR-GARCH and EGARCH; news impact curves with a kernel estimate; the sign-bias test;
  * diagnostics    -- Ljung-Box tests on standardised residuals and their squares, ARCH-LM; nine models by AIC/BIC;
  * forecasts      -- multi-step forecasts and the volatility term structure; out-of-sample one-day forecasts of
                      GARCH-t, GJR-t and EWMA compared by QLIKE and the Diebold-Mariano test;
  * VaR            -- one-day VaR 1% from GARCH-t and from EWMA-Normal, and its exceedances.
Units: returns in % (r_t = 100 ln(P_t / P_{t-1})). EUR/RON returns are small (daily standard deviation about 0.3%), so
its models are estimated on 10 r_t and the results are converted back to % (SCALE); nothing else changes.
Output: charts/tsa_ch5_*.pdf/.png, Quantlets/Ch_05/ch5_numbers.json and ch5_*.csv
References: Huang and Petukhina (2022), Ch. 6; Tsay (2010), Ch. 3; Franke, Haerdle and Hafner (2019), Ch. 13.
Run:  python3 Quantlets/Ch_05/generate_all_charts.py
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from tsa_data import log_returns, periods_per_year   # noqa: E402
import tsa_style as st                               # noqa: E402
import statsmodels.api as sm                         # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox   # noqa: E402
from arch import arch_model                          # noqa: E402

warnings.filterwarnings('ignore')
SEED = 2026
ASSETS = ['sp500', 'bet', 'eurron', 'btc']
NAME = {'sp500': 'S&P 500', 'bet': 'BET', 'eurron': 'EUR/RON', 'btc': 'Bitcoin'}
START = {'sp500': '2000-01-01', 'bet': '2000-01-01', 'eurron': None, 'btc': None}
COLORS = {'sp500': st.MainBlue, 'bet': st.IDAred, 'eurron': st.Forest, 'btc': st.Amber}
SCALE = {'eurron': 10}       # estimation units: 10 r_t for EUR/RON (variance close to 1), r_t otherwise
EPISODES = {'2008': ('2008-09-01', '2009-03-31'), '2020': ('2020-02-15', '2020-05-31')}
OOS_START = '2015-01-01'     # out-of-sample period of the forecast comparison
REFIT = 250                  # the models are re-estimated every 250 observations (expanding window)
LAMBDA = 0.94                # EWMA (RiskMetrics) decay factor
ALPHA_VAR = 0.01             # VaR level: VaR 1%
LB_LAGS = 10                 # Ljung-Box Q(10)
ARCH_LAGS = 5                # ARCH-LM(5)
IG = 0.9995                  # persistence at or above this value: IGARCH (alpha + beta = 1 on the boundary)
TERM_DATES = ['2017-11-03', '2020-03-16']   # a calm day and the COVID-19 peak (plus the last day of the data)


# =============================================================================
# DATA AND HELPERS
# =============================================================================
def returns(k, start=None, end=None):
    """Daily log returns in % on the series' own calendar."""
    r = log_returns(k, start or START.get(k)) if end is None else log_returns(k, start or START.get(k), end)
    return r.rename(k)


def tidy_dates(fig, maxticks=6):
    """Date axes: at most `maxticks` ticks with concise labels, so that the dates never overlap."""
    import matplotlib.dates as mdates
    for ax in fig.axes:
        if isinstance(ax.xaxis.get_major_formatter(), (mdates.AutoDateFormatter, mdates.ConciseDateFormatter)):
            loc = mdates.AutoDateLocator(minticks=3, maxticks=maxticks)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def save(name, save_it=True):
    tidy_dates(plt.gcf())
    if save_it:
        st.check_no_grey(plt.gcf())
        st.save_fig(name)
    else:
        plt.show()


def acf_vals(x, nlags):
    """Sample autocorrelations rho_hat(1..nlags) = sum (x_t - m)(x_{t-k} - m) / sum (x_t - m)^2."""
    e = np.asarray(x, float) - np.mean(x)
    s = np.sum(e ** 2)
    return np.array([np.sum(e[k:] * e[:-k]) / s for k in range(1, nlags + 1)])


def ljung_box(x, lags=LB_LAGS, dof=0):
    """Ljung-Box Q(lags) and its p-value (chi-square with lags - dof degrees of freedom)."""
    lb = acorr_ljungbox(pd.Series(np.asarray(x, float)).dropna(), lags=[lags], model_df=dof)
    return float(lb['lb_stat'].iloc[0]), float(lb['lb_pvalue'].iloc[0])


def arch_lm(x, q=ARCH_LAGS):
    """Engle's (1982) ARCH-LM test: regress e_t^2 on a constant and e_{t-1}^2, ..., e_{t-q}^2 (e_t = x_t - mean);
    LM = n R^2 ~ chi-square(q) under no ARCH effects (n = number of regression observations)."""
    e2 = (np.asarray(x, float) - np.mean(x)) ** 2
    y = e2[q:]
    X = np.column_stack([np.ones(len(y))] + [e2[q - j:len(e2) - j] for j in range(1, q + 1)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ b
    r2 = 1 - np.sum(u ** 2) / np.sum((y - y.mean()) ** 2)
    lm = len(y) * r2
    return {'lm': float(lm), 'r2': float(r2), 'n': len(y), 'p': float(stats.chi2.sf(lm, q)),
            'crit': float(stats.chi2.ppf(0.95, q)), 'b': [float(v) for v in b]}


# =============================================================================
# STYLISED FACTS
# =============================================================================
def stylised_table(names=ASSETS):
    """Moments, autocorrelations of r, r^2 and |r| at lag 1, Ljung-Box Q(10) of r and r^2, ARCH-LM(5)."""
    out = {}
    for k in names:
        r = returns(k)
        x = r.values
        ppy = periods_per_year(r)
        jb = stats.jarque_bera(x)
        out[k] = {'n': len(x), 'first': r.index[0].date().isoformat(), 'last': r.index[-1].date().isoformat(),
                  'mean': float(x.mean()), 'sd': float(x.std(ddof=1)), 'vol': float(x.std(ddof=1) * np.sqrt(ppy)),
                  'ppy': float(ppy), 'skew': float(stats.skew(x)), 'kurt': float(stats.kurtosis(x, fisher=False)),
                  'jb': float(jb.statistic), 'min': float(x.min()), 'max': float(x.max()),
                  'date_min': r.idxmin().date().isoformat(), 'date_max': r.idxmax().date().isoformat(),
                  'rho_r': float(acf_vals(x, 1)[0]), 'rho_r2': float(acf_vals(x ** 2, 1)[0]),
                  'rho_abs': float(acf_vals(np.abs(x), 1)[0]), 'lb_r': ljung_box(x), 'lb_r2': ljung_box(x ** 2),
                  'arch': arch_lm(x)}
    return out


def fig_returns(names=ASSETS, save_it=True):
    """Daily log returns of the four series with the 2008 and 2020 episodes shaded."""
    fig, axes = plt.subplots(2, 2, figsize=(9.33, 2.59))
    out = {}
    for ax, k in zip(axes.ravel(), names):
        r = returns(k)
        ax.plot(r.index, r.values, color=COLORS[k], lw=0.45, label=f'{NAME[k]}: daily log return (%)')
        for (a, b), c in zip(EPISODES.values(), [st.Purple, st.Orange]):
            if pd.Timestamp(a) > r.index[0]:
                ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=0.18, lw=0)
        ax.set_ylabel('%')
        out[k] = {f'sd{e}': float(r.loc[a:b].std()) for e, (a, b) in EPISODES.items() if pd.Timestamp(a) > r.index[0]}
        out[k]['sd2017'] = float(r.loc['2017-01-01':'2017-12-31'].std())
        out[k]['sd_all'] = float(r.std())
    h, l = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            h.append(hh)
            l.append(ll)
    h += [plt.Rectangle((0, 0), 1, 1, color=st.Purple, alpha=0.3), plt.Rectangle((0, 0), 1, 1, color=st.Orange, alpha=0.3)]
    l += ['Sep 2008 - Mar 2009 (global financial crisis)', 'Feb - May 2020 (COVID-19)']
    st.fig_legend_bottom(fig, h, l, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save('tsa_ch5_returns', save_it)
    return out


def fig_acf_squares(names=ASSETS, lags=50, save_it=True):
    """ACF of returns and of squared returns, lags 1-50, with the i.i.d. 95% band, for the four series."""
    fig, axes = plt.subplots(2, 2, figsize=(9.33, 3.04), sharex=True)
    out = {}
    j = np.arange(1, lags + 1)
    for ax, k in zip(axes.ravel(), names):
        x = returns(k).values
        a_r, a_2 = acf_vals(x, lags), acf_vals(x ** 2, lags)
        band = 1.96 / np.sqrt(len(x))
        ax.bar(j - 0.2, a_r, width=0.4, color=st.MainBlue, label='returns r(t)')
        ax.bar(j + 0.2, a_2, width=0.4, color=st.IDAred, label='squared returns r(t)^2')
        ax.axhspan(-band, band, color=st.Teal, alpha=0.25, lw=0, label='95% band, +/- 1.96/sqrt(T)')
        ax.axhline(0, color=st.DarkText, lw=0.5)
        ax.set_title(NAME[k], loc='left', fontsize=12)
        out[k] = {'band': float(band), 'r2_1': float(a_2[0]), 'r2_10': float(a_2[9]), 'r2_50': float(a_2[-1]),
                  'r_1': float(a_r[0]), 'out_r2': int(np.sum(np.abs(a_2) > band)), 'out_r': int(np.sum(np.abs(a_r) > band))}
    for ax in axes[1]:
        ax.set_xlabel('lag k (days)')
    for ax in axes[:, 0]:
        ax.set_ylabel('ACF')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save('tsa_ch5_acf_squares', save_it)
    return out


# =============================================================================
# THE MEAN MODEL (CHAPTER 2) AND ITS RESIDUALS
# =============================================================================
def ar_order(r, pmax=5):
    """AR(p) for the mean with constant variance, p = 0..pmax chosen by BIC on the same observations."""
    bic = {}
    for p in range(pmax + 1):
        am = arch_model(r * SCALE.get(r.name, 1), mean='AR' if p else 'Constant', lags=p, vol='Constant',
                        hold_back=pmax, rescale=False)
        bic[p] = float(am.fit(disp='off').bic)
    return min(bic, key=bic.get), bic


def ols_ar1(r):
    """AR(1) for the mean by OLS: r_t = c + phi r_{t-1} + e_t; coefficients, classic and HAC-robust t statistics."""
    y, x = r.values[1:], r.values[:-1]
    X = sm.add_constant(x)
    o = sm.OLS(y, X).fit()
    h = sm.OLS(y, X).fit(cov_type='HC0')
    return {'c': float(o.params[0]), 'phi': float(o.params[1]), 'se': float(o.bse[1]), 'se_robust': float(h.bse[1]),
            'resid': pd.Series(o.resid, index=r.index[1:])}


def fig_arma_resid(k='bet', lags=30, save_it=True):
    """Left: ACF of the residuals of an AR(1) mean model and of their squares. Right: ACF of the standardised
    residuals of AR(1)-GARCH(1,1)-t and of their squares."""
    r = returns(k)
    o = ols_ar1(r)
    e = o['resid'].values
    res = fit(r, mean='AR', dist='t')
    z = res.std_resid.dropna().values
    j = np.arange(1, lags + 1)
    band = 1.96 / np.sqrt(len(e))
    fig, axes = plt.subplots(1, 2, figsize=(9.32, 2.62), sharey=True)
    for ax, (a, b), t in [(axes[0], (e, e ** 2), 'AR(1) with constant variance: residuals e(t)'),
                          (axes[1], (z, z ** 2), 'AR(1)-GARCH(1,1)-t: standardised residuals z(t)')]:
        ax.bar(j - 0.2, acf_vals(a, lags), width=0.4, color=st.MainBlue, label='residuals')
        ax.bar(j + 0.2, acf_vals(b, lags), width=0.4, color=st.IDAred, label='squared residuals')
        ax.axhspan(-band, band, color=st.Teal, alpha=0.25, lw=0, label='95% band, +/- 1.96/sqrt(T)')
        ax.axhline(0, color=st.DarkText, lw=0.5)
        ax.set_title(t, loc='left', fontsize=11)
        ax.set_xlabel('lag k (days)')
    axes[0].set_ylabel('autocorrelation')
    h, l = axes[0].get_legend_handles_labels()
    st.fig_legend_bottom(fig, h, l, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save('tsa_ch5_arma_resid', save_it)
    p_bic, bic = ar_order(r)
    return {'p_bic': int(p_bic), 'phi': o['phi'], 'phi_se': o['se'], 'phi_se_robust': o['se_robust'], 'c': o['c'],
            'lb_e': ljung_box(e, dof=1), 'lb_e2': ljung_box(e ** 2), 'archlm_e': arch_lm(e),
            'lb_z': ljung_box(z, dof=1), 'lb_z2': ljung_box(z ** 2), 'archlm_z': arch_lm(z),
            'acf_e2_1': float(acf_vals(e ** 2, 1)[0]), 'acf_z2_1': float(acf_vals(z ** 2, 1)[0])}


# =============================================================================
# SIMULATION AND LIKELIHOOD
# =============================================================================
def simulate_garch(n, omega, alpha, beta=0.0, burn=500, seed=SEED, dist='normal', nu=5.0):
    """eps_t = sigma_t z_t, sigma_t^2 = omega + alpha eps_{t-1}^2 + beta sigma_{t-1}^2 (beta = 0: ARCH(1));
    z_t standard Normal or standardised Student-t; returns (eps, sigma)."""
    rng = np.random.default_rng(seed)
    m = n + burn
    z = rng.standard_normal(m) if dist == 'normal' else rng.standard_t(nu, m) * np.sqrt((nu - 2) / nu)
    s2 = np.empty(m)
    e = np.empty(m)
    s2[0] = omega / (1 - alpha - beta)
    e[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, m):
        s2[t] = omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * z[t]
    return e[burn:], np.sqrt(s2[burn:])


def fig_simulated(n=1000, save_it=True):
    """Three paths with unconditional variance 1: i.i.d. Normal, ARCH(1) (omega = 0.5, alpha = 0.5) and
    GARCH(1,1) (omega = 0.02, alpha = 0.10, beta = 0.88, values typical of daily equity returns)."""
    rng = np.random.default_rng(SEED)
    paths = {'i.i.d. Normal(0, 1)': rng.standard_normal(n),
             'ARCH(1): omega = 0.5, alpha = 0.5': simulate_garch(n, 0.5, 0.5, 0.0, seed=SEED + 1)[0],
             'GARCH(1,1): omega = 0.02, alpha = 0.10, beta = 0.88': simulate_garch(n, 0.02, 0.10, 0.88, seed=SEED + 2)[0]}
    fig, axes = plt.subplots(3, 1, figsize=(9.33, 2.09), sharex=True, sharey=True)
    out = {}
    for ax, (lab, x), c in zip(axes, paths.items(), [st.MainBlue, st.IDAred, st.Forest]):
        ax.plot(np.arange(n), x, color=c, lw=0.7, label=lab)
        out[lab.split(':')[0]] = {'sd': float(np.std(x)), 'kurt': float(stats.kurtosis(x, fisher=False)),
                                  'max_abs': float(np.max(np.abs(x))), 'acf2_1': float(acf_vals(x ** 2, 1)[0])}
    axes[-1].set_xlabel('time t')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save('tsa_ch5_simulated', save_it)
    return out


def arch1_loglik(alpha, x):
    """Conditional Gaussian log-likelihood of ARCH(1) with omega = 1 - alpha (unconditional variance 1):
    l(alpha) = sum_{t>=2} [-0.5 ln s2_t - 0.5 x_t^2 / s2_t] (constant dropped), s2_t = omega + alpha x_{t-1}^2."""
    s2 = (1 - alpha) + alpha * x[:-1] ** 2
    return float(np.sum(-0.5 * np.log(s2) - 0.5 * x[1:] ** 2 / s2))


def fig_lik_arch1(ns=(100, 1000), alpha0=0.5, save_it=True):
    """Log-likelihood of a simulated ARCH(1) (omega = alpha = 0.5) as a function of alpha, for two sample sizes;
    each curve minus its maximum; standard errors from the curvature at the maximum."""
    grid = np.linspace(0.05, 0.95, 181)
    fig, ax = plt.subplots(figsize=(10.17, 1.69))
    out = {}
    for n, c in zip(ns, [st.MainBlue, st.IDAred]):
        x = simulate_garch(n, 1 - alpha0, alpha0, 0.0, seed=SEED + n)[0]
        ll = np.array([arch1_loglik(a, x) for a in grid])
        res = optimize.minimize_scalar(lambda a: -arch1_loglik(a, x), bounds=(0.001, 0.999), method='bounded')
        ax.plot(grid, ll - ll.max(), color=c, lw=1.6, label=f'n = {n}: log-likelihood minus its maximum')
        ax.axvline(res.x, color=c, ls='--', lw=1.0, label=f'n = {n}: maximum at alpha = {res.x:.3f}')
        h = 1e-4
        d2 = (arch1_loglik(res.x + h, x) - 2 * arch1_loglik(res.x, x) + arch1_loglik(res.x - h, x)) / h ** 2
        out[str(n)] = {'alpha_hat': float(res.x), 'se': float(1 / np.sqrt(-d2))}
    ax.axvline(alpha0, color='black', ls=':', lw=1.2, label=f'true alpha = {alpha0}')
    ax.set_ylim(-40, 2)
    ax.set_xlabel('alpha')
    ax.set_ylabel('log-likelihood minus maximum')
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    save('tsa_ch5_lik_arch1', save_it)
    return out


def garch_loglik_vt(a, b, x, s2bar):
    """Gaussian GARCH(1,1) log-likelihood with variance targeting omega = (1 - a - b) s2bar."""
    om = (1 - a - b) * s2bar
    s2 = np.empty(len(x))
    s2[0] = s2bar
    for t in range(1, len(x)):
        s2[t] = om + a * x[t - 1] ** 2 + b * s2[t - 1]
    return float(np.sum(-0.5 * np.log(s2[1:]) - 0.5 * x[1:] ** 2 / s2[1:]))


def fig_lik_garch(ns=(500, 2000), save_it=True):
    """Contours of the log-likelihood of a simulated GARCH(1,1) (omega = 0.1, alpha = 0.1, beta = 0.8) over a grid of
    (alpha, beta), omega set by variance targeting, for n = 500 and n = 2000."""
    A = np.linspace(0.01, 0.30, 59)
    B = np.linspace(0.50, 0.97, 48)
    fig, axes = plt.subplots(1, 2, figsize=(8.49, 1.84), sharey=True)
    out = {}
    cols = [st.Purple, st.MainBlue, st.Teal, st.Forest, st.Amber, st.Orange, st.IDAred]
    for ax, n in zip(axes, ns):
        x = simulate_garch(n, 0.1, 0.1, 0.8, seed=SEED + 7)[0]
        s2bar = float(np.var(x))
        L = np.full((len(B), len(A)), np.nan)
        for i, b in enumerate(B):
            for j, a in enumerate(A):
                if a + b < 0.995:
                    L[i, j] = garch_loglik_vt(a, b, x, s2bar)
        i, j = np.unravel_index(np.nanargmax(L), L.shape)
        lv = np.nanmax(L) - np.array([40, 20, 10, 5, 3, 2, 1])
        cs = ax.contour(A, B, L, levels=lv, colors=cols, linewidths=1.1)
        ax.clabel(cs, fmt=lambda v, m=np.nanmax(L): f'{v - m:.0f}', fontsize=10)
        ax.plot(A[j], B[i], 'o', color=st.IDAred, ms=8, label='maximum on the grid')
        ax.plot(0.1, 0.8, 'x', color='black', ms=9, mew=2, label='true values: alpha = 0.1, beta = 0.8')
        ax.plot(A, 1 - A, color=st.Purple, ls='--', lw=1.2, label='alpha + beta = 1 (IGARCH boundary)')
        ax.set_xlim(A[0], A[-1])
        ax.set_ylim(B[0], B[-1])
        ax.set_title(f'n = {n}: maximum at\nalpha = {A[j]:.2f}, beta = {B[i]:.2f}', loc='left', fontsize=10)
        ax.set_xlabel('alpha')
        out[str(n)] = {'a_hat': float(A[j]), 'b_hat': float(B[i])}
    axes[0].set_ylabel('beta')
    st.fig_legend_bottom(fig, ncol=3, y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save('tsa_ch5_lik_garch', save_it)
    return out


# =============================================================================
# ESTIMATION (arch package)
# =============================================================================
def fit(r, vol='GARCH', dist='t', o=0, p=1, q=1, mean='Constant', lags=1, last_obs=None, cov_type='robust'):
    """ML estimation with the arch package: constant or AR(lags) mean; GARCH (o = 0), GJR-GARCH (o = 1), EGARCH,
    ARCH(p). The series is multiplied by SCALE (EUR/RON: 10) and the result remembers the factor in res.sc."""
    sc = SCALE.get(r.name, 1)
    kw = dict(mean=mean, lags=lags if mean == 'AR' else 0, dist=dist, rescale=False)
    if vol == 'ARCH':
        am = arch_model(sc * r, vol='ARCH', p=p, **kw)
    elif vol == 'EGARCH':
        am = arch_model(sc * r, vol='EGARCH', p=1, o=1, q=1, **kw)
    else:
        am = arch_model(sc * r, vol='GARCH', p=p, o=o, q=q, **kw)
    res = am.fit(disp='off', last_obs=last_obs, cov_type=cov_type, options={'maxiter': 2000})
    res.sc = sc
    return res


def par(res):
    """Parameters in % units (the estimation scale removed): mu and Const / sc, omega / sc^2 (GARCH, GJR, ARCH)."""
    p = res.params.copy()
    for n in p.index:
        if n in ('mu', 'Const'):
            p[n] = p[n] / res.sc
        elif n == 'omega' and res.model.volatility.__class__.__name__ != 'EGARCH':
            p[n] = p[n] / res.sc ** 2
    return p


def sig(res):
    """Conditional volatility in %."""
    return res.conditional_volatility / res.sc


def persistence(res):
    """alpha + beta (GARCH), alpha + beta + gamma/2 (GJR with symmetric innovations), beta (EGARCH)."""
    p = res.params
    if res.model.volatility.__class__.__name__ == 'EGARCH':
        return float(p['beta[1]'])
    return float(sum(v for n, v in p.items() if n.startswith(('alpha', 'beta'))) + 0.5 * p.get('gamma[1]', 0.0))


def half_life(pers):
    """Days after which half of a variance shock has disappeared: ln 0.5 / ln(persistence)."""
    return float(np.log(0.5) / np.log(pers)) if 0 < pers < 1 else float('inf')


def summary(res, r):
    """Parameters and robust standard errors in % units, persistence, half-life, long-run and sample volatility."""
    p = par(res)
    se = res.std_err.copy()
    for n in se.index:
        if n in ('mu', 'Const'):
            se[n] = se[n] / res.sc
        elif n == 'omega':
            se[n] = se[n] / res.sc ** 2
    pers = persistence(res)
    ppy = periods_per_year(r)
    adj = res.nobs * np.log(res.sc)               # log-likelihood of r_t = log-likelihood of sc r_t + n ln sc
    out = {'n': int(res.nobs), 'first': r.index[0].date().isoformat(), 'ppy': float(ppy),
           'params': {k: float(v) for k, v in p.items()}, 'se': {k: float(v) for k, v in se.items()},
           'loglik': float(res.loglikelihood + adj), 'aic': float(res.aic - 2 * adj), 'bic': float(res.bic - 2 * adj),
           'k': int(len(p)), 'pers': pers, 'hl': half_life(pers), 'vol_sample': float(r.std() * np.sqrt(ppy))}
    if 'omega' in p and res.model.volatility.__class__.__name__ != 'EGARCH' and pers < IG:
        uv = p['omega'] / (1 - pers)
        out['uv'] = float(uv)
        out['vol_lr'] = float(np.sqrt(uv * ppy))
    return out


def garch11_negloglik(theta, x):
    """Minus the Gaussian log-likelihood of r_t = mu + eps_t, GARCH(1,1); sigma_1^2 = sample variance."""
    mu, om, a, b = theta
    if om <= 0 or a < 0 or b < 0 or a + b >= 1:
        return 1e10
    e = x - mu
    s2 = np.empty(len(x))
    s2[0] = np.var(x)
    for t in range(1, len(x)):
        s2[t] = om + a * e[t - 1] ** 2 + b * s2[t - 1]
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + e ** 2 / s2)


def fit_step_by_step(r):
    """Gaussian GARCH(1,1) by numerical optimisation of the log-likelihood (scipy, SLSQP, alpha + beta < 1), with
    classic standard errors from the inverse of the numerical Hessian."""
    x = np.asarray(r, float)
    v = np.var(x)
    th0 = np.array([x.mean(), 0.05 * v, 0.08, 0.90])
    bnds = [(None, None), (1e-6, None), (1e-6, 0.999), (1e-6, 0.999)]
    cons = [{'type': 'ineq', 'fun': lambda t: 0.9999 - t[2] - t[3]}]
    res = optimize.minimize(garch11_negloglik, th0, args=(x,), method='SLSQP', bounds=bnds, constraints=cons,
                            options={'maxiter': 1000, 'ftol': 1e-10})
    th = res.x
    from statsmodels.tools.numdiff import approx_hess
    H = approx_hess(th, garch11_negloglik, args=(x,))
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    names = ['mu', 'omega', 'alpha[1]', 'beta[1]']
    return {'params': dict(zip(names, map(float, th))), 'se': dict(zip(names, map(float, se))), 'loglik': float(-res.fun)}


def arch_q_table(k='sp500'):
    """ARCH(1), ARCH(5), ARCH(10) and GARCH(1,1), all Gaussian: log-likelihood, AIC, BIC, sum of the ARCH terms."""
    r = returns(k)
    out = {}
    for lab, kw in [('ARCH(1)', dict(vol='ARCH', p=1)), ('ARCH(5)', dict(vol='ARCH', p=5)),
                    ('ARCH(10)', dict(vol='ARCH', p=10)), ('GARCH(1,1)', dict(vol='GARCH'))]:
        res = fit(r, dist='normal', **kw)
        p = res.params
        out[lab] = {'k': int(len(p)), 'loglik': float(res.loglikelihood), 'aic': float(res.aic), 'bic': float(res.bic),
                    'sum_alpha': float(sum(v for n, v in p.items() if n.startswith('alpha'))),
                    'pers': float(sum(v for n, v in p.items() if n.startswith(('alpha', 'beta'))))}
    return out


def estimation_sp500(k='sp500'):
    """S&P 500 GARCH(1,1): Normal innovations step by step and with arch (classic and robust standard errors);
    Student-t and skewed-t innovations with arch; kurtosis of returns and of standardised residuals."""
    r = returns(k)
    out = {'step': fit_step_by_step(r)}
    rn = fit(r, dist='normal')
    rc = fit(r, dist='normal', cov_type='classic')
    out['normal'] = summary(rn, r)
    out['normal']['se_classic'] = {n: float(v) for n, v in rc.std_err.items()}
    for d in ['t', 'skewt']:
        out[d] = summary(fit(r, dist=d), r)
    z = rn.std_resid.dropna()
    out['kurt_r'] = float(stats.kurtosis(r, fisher=False))
    out['kurt_z'] = float(stats.kurtosis(z, fisher=False))
    out['kurt_zt'] = float(stats.kurtosis(fit(r, dist='t').std_resid.dropna(), fisher=False))
    return out


def markets_table(names=ASSETS):
    """GARCH(1,1)-t for each series: parameters, robust standard errors, persistence, half-life, volatility."""
    return {k: summary(fit(returns(k), dist='t'), returns(k)) for k in names}


def fig_vol(names=('sp500', 'bet'), fname='tsa_ch5_vol_sp500_bet', save_it=True):
    """Annualised GARCH(1,1)-t conditional volatility, with the 2008 and 2020 episodes shaded."""
    fig, axes = plt.subplots(len(names), 1, figsize=(9.33, 2.38))
    out = {}
    for ax, k in zip(np.atleast_1d(axes), names):
        r = returns(k)
        res = fit(r, dist='t')
        ann = np.sqrt(periods_per_year(r))
        v = sig(res) * ann
        ax.plot(v.index, v.values, color=COLORS[k], lw=0.9, label=f'{NAME[k]}: GARCH(1,1)-t volatility, annualised (%)')
        ax.axhline(r.std() * ann, color='black', ls='--', lw=0.9,
                   label='sample volatility, annualised (%)' if k == names[0] else '_')
        for (a, b), c in zip(EPISODES.values(), [st.Purple, st.Orange]):
            if pd.Timestamp(a) > r.index[0]:
                ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=0.15, lw=0)
        ax.set_ylabel('% per year')
        out[k] = {'last': float(v.iloc[-1]), 'max': float(v.max()), 'date_max': v.idxmax().date().isoformat(),
                  'min': float(v.min()), 'median': float(v.median())}
        for e, (a, b) in EPISODES.items():
            w = v.loc[a:b]
            if len(w) and pd.Timestamp(a) > r.index[0]:
                out[k][f'peak{e}'] = float(w.max())
                out[k][f'date{e}'] = w.idxmax().date().isoformat()
    h, l = [], []
    for ax in np.atleast_1d(axes):
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if not ll.startswith('_'):
                h.append(hh)
                l.append(ll)
    h += [plt.Rectangle((0, 0), 1, 1, color=st.Purple, alpha=0.3), plt.Rectangle((0, 0), 1, 1, color=st.Orange, alpha=0.3)]
    l += ['Sep 2008 - Mar 2009 (global financial crisis)', 'Feb - May 2020 (COVID-19)']
    st.fig_legend_bottom(fig, h, l, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fname, save_it)
    return out


def fig_persistence(T=None, horizon=250, save_it=True):
    """Share of a variance shock left after h days, persistence^h, with the half-lives in the legend."""
    T = T or markets_table()
    fig, ax = plt.subplots(figsize=(10.17, 1.78))
    h = np.arange(horizon + 1)
    for k, d in T.items():
        lab = (f"{NAME[k]}: alpha + beta = {d['pers']:.3f}, half-life {d['hl']:.0f} days" if d['pers'] < IG
               else f"{NAME[k]}: alpha + beta = 1 (IGARCH), no half-life")
        ax.plot(h, min(d['pers'], 1.0) ** h, color=COLORS[k], lw=1.8 if k != 'eurron' else 2.6,
                ls=':' if k == 'btc' else '-', label=lab)
    ax.axhline(0.5, color='black', ls=':', lw=1.0, label='half of the shock')
    ax.set_xlabel('days after the shock, h')
    ax.set_ylabel('share of the shock left')
    ax.set_ylim(0, 1.02)
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    save('tsa_ch5_persistence', save_it)
    return {k: {'pers': d['pers'], 'hl': d['hl']} for k, d in T.items()}


def fig_qq(k='sp500', save_it=True):
    """QQ plots of the standardised residuals: GARCH-N against the Normal, GARCH-t against the standardised t."""
    r = returns(k)
    rn, rt = fit(r, dist='normal'), fit(r, dist='t')
    nu = rt.params['nu']
    fig, axes = plt.subplots(1, 2, figsize=(9.33, 2.10))
    out = {}
    for ax, res, lab, ppf, c in [(axes[0], rn, 'GARCH(1,1)-N residuals against the Normal', stats.norm.ppf, st.MainBlue),
                                 (axes[1], rt, f'GARCH(1,1)-t residuals against the standardised t({nu:.1f})',
                                  lambda u: stats.t.ppf(u, nu) * np.sqrt((nu - 2) / nu), st.IDAred)]:
        z = np.sort(res.std_resid.dropna().values)
        u = (np.arange(1, len(z) + 1) - 0.5) / len(z)
        qth = ppf(u)
        ax.scatter(qth, z, s=5, color=c, alpha=0.6, label=lab)
        lim = [min(qth.min(), z.min()), max(qth.max(), z.max())]
        ax.plot(lim, lim, color='black', lw=1.0, ls='--', label='45-degree line' if c == st.MainBlue else '_')
        ax.set_xlabel('theoretical quantile')
        ax.set_ylabel('sample quantile')
        out[lab.split()[0]] = {'q001': float(np.quantile(z, 0.001)), 'th001': float(ppf(0.001))}
    st.fig_legend_bottom(fig, ncol=1, y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save('tsa_ch5_qq', save_it)
    return {'nu': float(nu), 'q': out}


# =============================================================================
# EWMA AGAINST GARCH
# =============================================================================
def ewma_variance(r, lam=LAMBDA, init=250):
    """EWMA variance forecast for day t made at t-1: h_t = lam h_{t-1} + (1 - lam) r_{t-1}^2 (zero mean)."""
    x = np.asarray(r, float)
    h = np.empty(len(x))
    h[0] = np.mean(x[:init] ** 2)
    for t in range(1, len(x)):
        h[t] = lam * h[t - 1] + (1 - lam) * x[t - 1] ** 2
    return pd.Series(h, index=r.index)


def fig_ewma_garch(k='sp500', a='2019-10-01', b='2021-03-31', save_it=True):
    """Annualised volatility around the COVID-19 crash: GARCH(1,1)-t (all data), EWMA (lambda = 0.94) and a
    63-day rolling standard deviation."""
    r = returns(k)
    ann = np.sqrt(periods_per_year(r))
    g = sig(fit(r, dist='t')) * ann
    e = np.sqrt(ewma_variance(r)) * ann
    w = r.rolling(63).std() * ann
    fig, ax = plt.subplots(figsize=(7.00, 2.73))
    ax.plot(g.loc[a:b].index, g.loc[a:b], color=st.MainBlue, lw=1.5, label='GARCH(1,1)-t')
    ax.plot(e.loc[a:b].index, e.loc[a:b], color=st.IDAred, lw=1.3, ls='--', label='EWMA, lambda = 0.94')
    ax.plot(w.loc[a:b].index, w.loc[a:b], color=st.Forest, lw=1.3, label='63-day rolling standard deviation')
    ax.set_ylabel('% per year')
    st.legend_outside_bottom(ax, ncol=3, y=-0.16)
    save('tsa_ch5_ewma_garch', save_it)
    d = '2020-06-30'
    return {'g_peak': float(g.loc[a:b].max()), 'g_dpeak': g.loc[a:b].idxmax().date().isoformat(),
            'e_peak': float(e.loc[a:b].max()), 'e_dpeak': e.loc[a:b].idxmax().date().isoformat(),
            'w_peak': float(w.loc[a:b].max()), 'w_dpeak': w.loc[a:b].idxmax().date().isoformat(),
            'g_jun': float(g.loc[:d].iloc[-1]), 'e_jun': float(e.loc[:d].iloc[-1]), 'w_jun': float(w.loc[:d].iloc[-1]),
            'hl_ewma': float(np.log(0.5) / np.log(LAMBDA))}


# =============================================================================
# ARMA-GARCH: THE JOINT MODEL OF MEAN AND VARIANCE
# =============================================================================
def arma_garch_table(names=ASSETS):
    """AR(1) mean: OLS with constant variance against AR(1)-GARCH(1,1)-t (robust SE); Ljung-Box Q(10) of the
    standardised residuals with a constant mean and with the AR(1) mean; BIC of both GARCH-t models."""
    out = {}
    for k in names:
        r = returns(k)
        o = ols_ar1(r)
        rc, ra = fit(r, dist='t'), fit(r, mean='AR', dist='t')
        ph = [n for n in ra.params.index if n.endswith('[1]') and n not in ('alpha[1]', 'beta[1]')][0]
        sc_, sa_ = summary(rc, r), summary(ra, r)
        out[k] = {'p_bic': int(ar_order(r)[0]), 'phi_ols': o['phi'], 'se_ols': o['se'], 'phi_g': float(ra.params[ph]),
                  'se_g': float(ra.std_err[ph]), 't_g': float(ra.tvalues[ph]),
                  'lbz_c': ljung_box(rc.std_resid.dropna()), 'lbz_a': ljung_box(ra.std_resid.dropna(), dof=1),
                  'bic_c': sc_['bic'], 'bic_a': sa_['bic'], 'pers_a': sa_['pers'], 'nu_a': float(ra.params['nu'])}
    return out


def fig_bands(k='bet', a='2019-07-01', b='2021-06-30', save_it=True):
    """One-day-ahead 95% intervals for the BET return: AR(1) with constant variance (fixed width) against
    AR(1)-GARCH(1,1)-t (width that follows the conditional volatility); share of returns outside each band."""
    r = returns(k)
    o = ols_ar1(r)
    s = float(np.std(o['resid'], ddof=2))
    m_ols = o['c'] + o['phi'] * r.shift(1)
    lo_c, hi_c = m_ols - 1.96 * s, m_ols + 1.96 * s
    res = fit(r, mean='AR', dist='t')
    p = par(res)
    ph = [n for n in p.index if n.endswith('[1]') and n not in ('alpha[1]', 'beta[1]')][0]
    nu = p['nu']
    q = stats.t.ppf(0.975, nu) * np.sqrt((nu - 2) / nu)
    m_g = p['Const'] + p[ph] * r.shift(1)
    sg = sig(res)
    lo_g, hi_g = m_g - q * sg, m_g + q * sg
    out_c = ((r < lo_c) | (r > hi_c)).iloc[1:]
    out_g = ((r < lo_g) | (r > hi_g)).iloc[1:]
    fig, ax = plt.subplots(figsize=(10.17, 2.52))
    w = slice(a, b)
    ax.plot(r.loc[w].index, r.loc[w], color=COLORS[k], lw=0.6, label=f'{NAME[k]}: daily log return (%)')
    ax.plot(lo_c.loc[w].index, lo_c.loc[w], color=st.Forest, lw=1.2, ls='--', label='95% interval, AR(1) with constant variance')
    ax.plot(hi_c.loc[w].index, hi_c.loc[w], color=st.Forest, lw=1.2, ls='--', label='_')
    ax.plot(lo_g.loc[w].index, lo_g.loc[w], color=st.MainBlue, lw=1.2, label='95% interval, AR(1)-GARCH(1,1)-t')
    ax.plot(hi_g.loc[w].index, hi_g.loc[w], color=st.MainBlue, lw=1.2, label='_')
    ax.set_ylabel('%')
    st.legend_outside_bottom(ax, ncol=2, y=-0.16)
    save('tsa_ch5_bands', save_it)
    yrs = {'calm2017': ('2017-01-01', '2017-12-31'), 'covid': ('2020-03-01', '2020-04-30')}
    return {'sd_const': s, 'out_c': float(out_c.mean()), 'out_g': float(out_g.mean()),
            **{f'out_c_{y}': float(out_c.loc[u:v].mean()) for y, (u, v) in yrs.items()},
            **{f'out_g_{y}': float(out_g.loc[u:v].mean()) for y, (u, v) in yrs.items()},
            'w_c': float(2 * 1.96 * s), 'w_g_min': float((hi_g - lo_g).loc[w].min()), 'w_g_max': float((hi_g - lo_g).loc[w].max())}


# =============================================================================
# ASYMMETRY
# =============================================================================
def news_impact(res, eps, s2bar):
    """sigma_t^2 as a function of the shock eps_{t-1}, with sigma_{t-1}^2 fixed at s2bar (Engle and Ng, 1993);
    for series estimated without scaling (SCALE = 1)."""
    p = res.params
    if res.model.volatility.__class__.__name__ == 'EGARCH':
        z = eps / np.sqrt(s2bar)
        return np.exp(p['omega'] + p['alpha[1]'] * (np.abs(z) - np.sqrt(2 / np.pi)) + p['gamma[1]'] * z
                      + p['beta[1]'] * np.log(s2bar))
    return p['omega'] + (p['alpha[1]'] + p.get('gamma[1]', 0.0) * (eps < 0)) * eps ** 2 + p['beta[1]'] * s2bar


def kernel_nic(r, grid, h=None):
    """Nadaraya-Watson estimate of E[r_t^2 | r_{t-1} = x] with a Gaussian kernel; bandwidth half a standard
    deviation of the returns."""
    x, y = r.values[:-1], r.values[1:] ** 2
    h = h or 0.5 * np.std(x)
    w = stats.norm.pdf((grid[:, None] - x[None, :]) / h)
    return (w * y).sum(1) / w.sum(1), h


def fig_nic(names=('sp500', 'btc'), save_it=True):
    """News impact curves of GARCH-t, GJR-t and EGARCH-t and a kernel estimate, for two series."""
    fig, axes = plt.subplots(1, 2, figsize=(9.33, 2.71))
    out = {}
    for ax, k in zip(axes, names):
        r = returns(k)
        s2bar = float(r.var())
        lo, hi = np.quantile(r, [0.01, 0.99])
        g = np.linspace(lo, hi, 201)
        for vol, o, lab, c in [('GARCH', 0, 'GARCH(1,1)-t', st.MainBlue), ('GARCH', 1, 'GJR-GARCH(1,1)-t', st.IDAred),
                               ('EGARCH', 0, 'EGARCH(1,1)-t', st.Forest)]:
            ax.plot(g, news_impact(fit(r, vol=vol, o=o, dist='t'), g, s2bar), color=c, lw=1.6, label=lab)
        kn, h = kernel_nic(r, g)
        ax.plot(g, kn, color=st.Amber, lw=1.4, ls='--', label='kernel estimate of E[r(t)^2 | r(t-1)]')
        ax.set_title(NAME[k], loc='left', fontsize=12)
        ax.set_xlabel('shock yesterday, eps(t-1) (%)')
        ax.set_ylabel('variance today (%^2)')
        rg = fit(r, o=1, dist='t')
        out[k] = {'ratio_gjr': float(news_impact(rg, np.array([-2.0]), s2bar)[0] / news_impact(rg, np.array([2.0]), s2bar)[0]),
                  'h': float(h)}
    st.fig_legend_bottom(fig, ncol=2, y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save('tsa_ch5_nic', save_it)
    return out


def sign_bias_test(z, eps):
    """Engle-Ng (1993) sign-bias tests: z_t^2 on a constant, S-_{t-1}, S-_{t-1} eps_{t-1}, S+_{t-1} eps_{t-1};
    t statistics and the joint test T R^2 ~ chi2(3)."""
    d = pd.concat([pd.Series(np.asarray(z)), pd.Series(np.asarray(eps))], axis=1, keys=['z', 'e']).dropna()
    zz, e = d['z'].values, d['e'].values
    y = zz[1:] ** 2
    el = e[:-1]
    sneg = (el < 0).astype(float)
    X = np.column_stack([np.ones_like(el), sneg, sneg * el, (1 - sneg) * el])
    ols = sm.OLS(y, X).fit()
    joint = len(y) * ols.rsquared
    return {'sign_t': float(ols.tvalues[1]), 'neg_t': float(ols.tvalues[2]), 'pos_t': float(ols.tvalues[3]),
            'joint': float(joint), 'joint_p': float(stats.chi2.sf(joint, 3))}


def asym_table(names=ASSETS):
    """GJR-GARCH(1,1)-t and EGARCH(1,1)-t: gamma with its robust t statistic, the likelihood-ratio test of GJR against
    GARCH, BIC, and the sign-bias test on the GARCH-t residuals."""
    out = {}
    for k in names:
        r = returns(k)
        rg, rj, re_ = fit(r, dist='t'), fit(r, o=1, dist='t'), fit(r, vol='EGARCH', dist='t')
        lr = 2 * (rj.loglikelihood - rg.loglikelihood)
        out[k] = {'gjr_gamma': float(rj.params['gamma[1]']), 'gjr_t': float(rj.tvalues['gamma[1]']),
                  'gjr_alpha': float(rj.params['alpha[1]']), 'gjr_beta': float(rj.params['beta[1]']),
                  'gjr_pers': persistence(rj), 'lr': float(lr), 'lr_p': float(stats.chi2.sf(lr, 1)),
                  'eg_gamma': float(re_.params['gamma[1]']), 'eg_t': float(re_.tvalues['gamma[1]']),
                  'eg_beta': float(re_.params['beta[1]']), 'sb': sign_bias_test(rg.std_resid, rg.resid),
                  'bic_g': float(rg.bic), 'bic_j': float(rj.bic), 'bic_e': float(re_.bic)}
    return out


def managed_rate_case(k='eurron', a='2025-04-01', b='2025-04-30'):
    """EUR/RON, GARCH(1,1)-t: the largest standardised residual (6 May 2025 after a calm month), the conditional
    volatility the day before, and the Ljung-Box Q(10) of z^2 with and without that day."""
    r = returns(k)
    res = fit(r, dist='t')
    z = res.std_resid.dropna()
    d = z.abs().idxmax()
    v = sig(res)
    i = r.index.get_loc(d)
    return {'date': d.date().isoformat(), 'z': float(z.loc[d]), 'r': float(r.loc[d]), 'sig_before': float(v.iloc[i]),
            'r_next': float(r.iloc[i + 1]), 'mean_abs_month': float(r.loc[a:b].abs().mean()),
            'z_second': float(z.abs().sort_values().iloc[-2]), 'lb_z2': ljung_box(z ** 2), 'lb_z2_wo': ljung_box((z ** 2).drop(d)),
            'ppy': float(periods_per_year(r))}


# =============================================================================
# DIAGNOSTICS AND MODEL SELECTION
# =============================================================================
def diagnostics(k='sp500'):
    """Ljung-Box Q(10) of r, r^2, z, z^2 and ARCH-LM(5) of r and z, for GARCH-N and GARCH-t."""
    r = returns(k)
    out = {'r': ljung_box(r), 'r2': ljung_box(r ** 2), 'archlm_r': arch_lm(r.values)}
    for d in ['normal', 't']:
        z = fit(r, dist=d).std_resid.dropna()
        out[d] = {'z': ljung_box(z), 'z2': ljung_box(z ** 2), 'archlm': arch_lm(z.values)}
    return out


def diag_markets(names=ASSETS):
    """Ljung-Box Q(10) of r^2 and of z (GARCH-t) and z^2 for each series."""
    out = {}
    for k in names:
        r = returns(k)
        z = fit(r, dist='t').std_resid.dropna()
        out[k] = {'r2': ljung_box(r ** 2), 'z2': ljung_box(z ** 2), 'z': ljung_box(z), 'lm_z': arch_lm(z.values)}
    return out


def fig_acf_diag(k='sp500', nlags=50, save_it=True):
    """ACF of squared returns and of squared standardised residuals (GARCH-t), with the 95% band."""
    r = returns(k)
    z = fit(r, dist='t').std_resid.dropna()
    a1, a2 = acf_vals(r ** 2, nlags), acf_vals(z ** 2, nlags)
    lags = np.arange(1, nlags + 1)
    band = 1.96 / np.sqrt(len(r))
    fig, ax = plt.subplots(figsize=(10.17, 1.97))
    ax.bar(lags - 0.2, a1, width=0.4, color=COLORS[k], label=f'{NAME[k]}: squared returns')
    ax.bar(lags + 0.2, a2, width=0.4, color=st.IDAred, label=f'{NAME[k]}: squared standardised residuals, GARCH(1,1)-t')
    ax.axhline(band, color='black', ls='--', lw=0.9, label='95% band, +/- 1.96/sqrt(T)')
    ax.axhline(-band, color='black', ls='--', lw=0.9)
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xlabel('lag k (days)')
    ax.set_ylabel('autocorrelation')
    st.legend_outside_bottom(ax, ncol=2, y=-0.2)
    save('tsa_ch5_acf_diag', save_it)
    return {'a1_1': float(a1[0]), 'a1_50': float(a1[-1]), 'a2_1': float(a2[0]), 'max_a2': float(np.max(np.abs(a2))),
            'band': float(band)}


def model_selection(k='sp500'):
    """Nine models (GARCH, GJR, EGARCH x Normal, t, skewed t): log-likelihood, number of parameters, AIC, BIC."""
    r = returns(k)
    rows = []
    for vol, o, vlab in [('GARCH', 0, 'GARCH'), ('GARCH', 1, 'GJR'), ('EGARCH', 0, 'EGARCH')]:
        for d, dlab in [('normal', 'Normal'), ('t', 't'), ('skewt', 'skewed t')]:
            res = fit(r, vol=vol, o=o, dist=d)
            rows.append({'model': vlab, 'dist': dlab, 'k': int(len(res.params)), 'loglik': float(res.loglikelihood),
                         'aic': float(res.aic), 'bic': float(res.bic)})
    df = pd.DataFrame(rows)
    df['d_aic'] = df['aic'] - df['aic'].min()
    df['d_bic'] = df['bic'] - df['bic'].min()
    return df


# =============================================================================
# FORECASTS
# =============================================================================
def garch_path_forecast(res, r, date, horizon=250):
    """Multi-step GARCH(1,1) variance forecasts (in %^2) made at the end of `date`:
    sigma^2_{t+h} = s2bar + (alpha + beta)^(h-1) (sigma^2_{t+1} - s2bar), s2bar = omega / (1 - alpha - beta)."""
    p = par(res)
    pers = p['alpha[1]'] + p['beta[1]']
    s2bar = p['omega'] / (1 - pers)
    i = r.index.get_loc(pd.Timestamp(date))
    s2t = sig(res).iloc[i] ** 2
    e = r.iloc[i] - p['mu']
    s2next = p['omega'] + p['alpha[1]'] * e ** 2 + p['beta[1]'] * s2t
    h = np.arange(1, horizon + 1)
    return s2bar + pers ** (h - 1) * (s2next - s2bar), s2bar, s2next


def fig_term_structure(k='sp500', horizon=250, save_it=True):
    """Forecasts of the daily volatility h days ahead (annualised) and of the average volatility over the next h days,
    made on a calm day, at the COVID-19 peak and on the last day of the data."""
    r = returns(k)
    res = fit(r, dist='t')
    ann = periods_per_year(r)
    dates = TERM_DATES + [r.index[-1].date().isoformat()]
    fig, ax = plt.subplots(figsize=(10.17, 1.84))
    out = {}
    h = np.arange(1, horizon + 1)
    for d, c in zip(dates, [st.Forest, st.IDAred, st.MainBlue]):
        f, s2bar, s2n = garch_path_forecast(res, r, d, horizon)
        avg = np.sqrt(np.cumsum(f) / h * ann)
        ax.plot(h, np.sqrt(f * ann), color=c, lw=1.6, label=f'forecast made on {d}')
        ax.plot(h, avg, color=c, lw=1.2, ls='--')
        out[d] = {'h1': float(np.sqrt(f[0] * ann)), 'h10': float(np.sqrt(f[9] * ann)), 'h22': float(np.sqrt(f[21] * ann)),
                  'h250': float(np.sqrt(f[-1] * ann)), 'avg10': float(avg[9]), 'avg22': float(avg[21]), 'avg250': float(avg[-1]),
                  's2next': float(s2n), 'sum10': float(np.sum(f[:10]))}
    ax.axhline(np.sqrt(s2bar * ann), color='black', ls=':', lw=1.0, label='long-run volatility')
    ax.set_xlabel('horizon h (trading days)')
    ax.set_ylabel('% per year')
    ax.plot([], [], color='black', lw=1.6, label='solid: volatility on day t+h')
    ax.plot([], [], color='black', lw=1.2, ls='--', label='dashed: average volatility up to day t+h')
    st.legend_outside_bottom(ax, ncol=3, y=-0.2)
    save('tsa_ch5_term_structure', save_it)
    out['lr'] = float(np.sqrt(s2bar * ann))
    out['s2bar'] = float(s2bar)
    out['params'] = {n: float(v) for n, v in par(res).items()}
    return out


def oos_forecasts(k, oos_start=OOS_START, refit=REFIT):
    """One-day-ahead variance forecasts on the out-of-sample period: GARCH-t and GJR-t re-estimated every `refit`
    observations on an expanding window (the forecast for day t uses data up to t-1), and EWMA; the one-day VaR 1% of
    GARCH-t and of EWMA-Normal (all in %)."""
    r = returns(k)
    sc = SCALE.get(k, 1)
    i0 = r.index.searchsorted(pd.Timestamp(oos_start))
    cols = {'garch': [], 'gjr': [], 'var_garch': []}
    for s in range(i0, len(r), refit):
        e = min(s + refit, len(r))
        fits = {}
        for lab, o in [('garch', 0), ('gjr', 1)]:
            am = arch_model(sc * r.iloc[:e], mean='Constant', vol='GARCH', p=1, o=o, q=1, dist='t', rescale=False)
            res = am.fit(disp='off', last_obs=r.index[s], options={'maxiter': 2000})
            fixed = am.fix(res.params)
            cols[lab].append(fixed.conditional_volatility.iloc[s:e] ** 2 / sc ** 2)
            fits[lab] = res.params
        p = fits['garch']
        q = stats.t.ppf(ALPHA_VAR, p['nu']) * np.sqrt((p['nu'] - 2) / p['nu'])
        cols['var_garch'].append(-(p['mu'] / sc + np.sqrt(cols['garch'][-1]) * q))
    df = pd.DataFrame({c: pd.concat(v) for c, v in cols.items()})
    df['ewma'] = ewma_variance(r).loc[df.index]
    df['var_ewma'] = -np.sqrt(df['ewma']) * stats.norm.ppf(ALPHA_VAR)
    df['r'] = r.loc[df.index]
    return df


def qlike(proxy, h):
    """QLIKE loss (Patton, 2011), written as proxy / h + ln h: it differs from proxy / h - ln(proxy / h) - 1 only by a
    term that does not depend on the forecast, and it allows a zero proxy."""
    return proxy / h + np.log(h)


def dm_test(la, lb, lags=5):
    """Diebold-Mariano: d = L_a - L_b; HAC (Newey-West) t statistic of the mean of d (negative: model a better)."""
    d = np.asarray(la) - np.asarray(lb)
    ols = sm.OLS(d, np.ones_like(d)).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    t = float(ols.tvalues[0])
    return {'mean': float(d.mean()), 't': t, 'p': float(2 * stats.norm.sf(abs(t)))}


def forecast_eval(names=ASSETS):
    """Mean QLIKE of GARCH-t, GJR-t and EWMA; DM tests against EWMA; VaR 1% exceedance rates."""
    out, frames = {}, {}
    for k in names:
        df = oos_forecasts(k)
        frames[k] = df
        px = df['r'] ** 2
        L = {m: qlike(px, df[m]) for m in ['garch', 'gjr', 'ewma']}
        eg, ee = df['r'] < -df['var_garch'], df['r'] < -df['var_ewma']
        out[k] = {'n': len(df), 'first': df.index[0].date().isoformat(),
                  'qlike': {m: float(v.mean()) for m, v in L.items()},
                  'dm_garch_ewma': dm_test(L['garch'], L['ewma']), 'dm_gjr_garch': dm_test(L['gjr'], L['garch']),
                  'exc_garch': float(eg.mean()), 'exc_ewma': float(ee.mean()),
                  'nexc_garch': int(eg.sum()), 'nexc_ewma': int(ee.sum())}
    return out, frames


def fig_forecast_eval(frames, save_it=True):
    """Cumulative QLIKE difference EWMA minus GARCH-t on the out-of-sample period (rising: GARCH-t better); EUR/RON
    in its own panel, since a few days with a near-zero EWMA forecast dominate its scale."""
    fig, axes = plt.subplots(1, 2, figsize=(7.70, 2.80), gridspec_kw={'width_ratios': [1.6, 1]})
    out = {}
    for k, df in frames.items():
        px = df['r'] ** 2
        dd = qlike(px, df['ewma']) - qlike(px, df['garch'])
        d = dd.cumsum()
        ax = axes[1] if k == 'eurron' else axes[0]
        ax.plot(d.index, d.values, color=COLORS[k], lw=1.4, label=f'{NAME[k]}')
        out[k] = {'total': float(d.iloc[-1]), 'max_day': dd.idxmax().date().isoformat(), 'max_val': float(dd.max()),
                  'share_max': float(dd.max() / d.iloc[-1]) if d.iloc[-1] != 0 else float('nan')}
    for ax in axes:
        ax.axhline(0, color='black', lw=0.8, ls='--')
    axes[0].set_ylabel('cumulative loss difference')
    axes[0].set_title('QLIKE(EWMA) - QLIKE(GARCH-t), cumulated', loc='left', fontsize=11)
    axes[1].set_title('EUR/RON', loc='left', fontsize=11)
    st.fig_legend_bottom(fig, ncol=4, y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save('tsa_ch5_forecast_eval', save_it)
    return out


def fig_var(frames, k='sp500', a='2019-07-01', b='2021-06-30', save_it=True):
    """Daily returns with the one-day VaR 1% of GARCH-t and of EWMA-Normal, and the exceedances."""
    df = frames[k].loc[a:b]
    fig, ax = plt.subplots(figsize=(10.17, 1.92))
    ax.plot(df.index, df['r'], color=COLORS[k], lw=0.7, label=f'{NAME[k]}: daily log return (%)')
    ax.plot(df.index, -df['var_garch'], color=st.IDAred, lw=1.3, label='minus VaR 1%, GARCH(1,1)-t')
    ax.plot(df.index, -df['var_ewma'], color=st.Forest, lw=1.1, ls='--', label='minus VaR 1%, EWMA-Normal')
    e1, e2 = df['r'] < -df['var_garch'], df['r'] < -df['var_ewma']
    ax.scatter(df.index[e1], df['r'][e1], color=st.IDAred, s=40, marker='o', zorder=5, label='exceedance of the GARCH-t VaR')
    ax.scatter(df.index[e2], df['r'][e2], color=st.Forest, s=60, marker='x', zorder=6, label='exceedance of the EWMA-Normal VaR')
    ax.set_ylabel('%')
    st.legend_outside_bottom(ax, ncol=2, y=-0.16)
    save('tsa_ch5_var', save_it)
    return {'n': int(len(df)), 'exc_garch': int(e1.sum()), 'exc_ewma': int(e2.sum())}


if __name__ == '__main__':
    st.apply()
    N = {}
    N['sty'] = stylised_table()
    N['returns'] = fig_returns()
    N['acf_sq'] = fig_acf_squares()
    N['arma'] = fig_arma_resid()
    N['sim'] = fig_simulated()
    N['lik_arch1'] = fig_lik_arch1()
    N['lik_garch'] = fig_lik_garch()
    N['archq'] = arch_q_table()
    N['est'] = estimation_sp500()
    T = markets_table()
    N['markets'] = T
    fig_persistence(T)
    N['vol1'] = fig_vol(('sp500', 'bet'), 'tsa_ch5_vol_sp500_bet')
    N['vol2'] = fig_vol(('eurron', 'btc'), 'tsa_ch5_vol_eurron_btc')
    N['qq'] = fig_qq()
    N['ewma'] = fig_ewma_garch()
    N['ag'] = arma_garch_table()
    N['bands'] = fig_bands()
    N['nic'] = fig_nic()
    N['asym'] = asym_table()
    N['diag'] = diagnostics()
    N['diag_m'] = diag_markets()
    N['acf_diag'] = fig_acf_diag()
    N['managed'] = managed_rate_case()
    ms = model_selection()
    ms.to_csv(os.path.join(HERE, 'ch5_model_selection.csv'), index=False, float_format='%.3f')
    N['ms'] = ms.to_dict(orient='records')
    N['term'] = fig_term_structure()
    fe, frames = forecast_eval()
    N['fe'] = fe
    N['fe_fig'] = fig_forecast_eval(frames)
    N['var_fig'] = fig_var(frames)
    N['end'] = returns('sp500').index[-1].date().isoformat()
    pd.DataFrame({NAME[k]: {'n': v['n'], 'mean (%)': v['mean'], 'sd (%)': v['sd'], 'skewness': v['skew'], 'kurtosis': v['kurt'],
                            'Q(10) r': v['lb_r'][0], 'Q(10) r^2': v['lb_r2'][0], 'ARCH-LM(5)': v['arch']['lm']}
                  for k, v in N['sty'].items()}).T.to_csv(os.path.join(HERE, 'ch5_stylised_facts.csv'), float_format='%.4f')
    pd.DataFrame({NAME[k]: {'n': v['n'], 'mu': v['params']['mu'], 'omega': v['params']['omega'],
                            'alpha': v['params']['alpha[1]'], 'beta': v['params']['beta[1]'], 'nu': v['params']['nu'],
                            'alpha+beta': v['pers'], 'half-life (days)': v['hl'], 'long-run vol (% p.a.)': v.get('vol_lr', np.nan),
                            'sample vol (% p.a.)': v['vol_sample']} for k, v in T.items()}).T.to_csv(
        os.path.join(HERE, 'ch5_garch_table.csv'), float_format='%.4f')
    pd.DataFrame({NAME[k]: {'n': v['n'], 'QLIKE GARCH-t': v['qlike']['garch'], 'QLIKE GJR-t': v['qlike']['gjr'],
                            'QLIKE EWMA': v['qlike']['ewma'], 'DM t (GARCH-t vs EWMA)': v['dm_garch_ewma']['t'],
                            'VaR 1% exceedances GARCH-t (%)': 100 * v['exc_garch'],
                            'VaR 1% exceedances EWMA-Normal (%)': 100 * v['exc_ewma']} for k, v in fe.items()}).T.to_csv(
        os.path.join(HERE, 'ch5_forecast_eval.csv'), float_format='%.4f')
    with open(os.path.join(HERE, 'ch5_numbers.json'), 'w') as f:
        json.dump(N, f, indent=1, default=float)
    print('done')
