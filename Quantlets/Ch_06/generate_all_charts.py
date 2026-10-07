"""
generate_all_charts.py -- charts and numbers of Chapter 6 (TSA): VAR models and Granger causality
=================================================================================================
Course data (tsa_data.py), chart style (tsa_style.py), statsmodels for the VAR models. Every number on the slides
comes from here.
  * multivariate data -- Romanian macro series (real GDP growth, HICP inflation, ROBOR 3M, unemployment, EUR/RON);
                         cross-correlations between markets (S&P 500, DAX, BET) and between inflation and ROBOR;
  * VAR(p)            -- a simulated bivariate VAR(1) (the worked example of the slides) and its impulse responses;
                         eigenvalues of companion matrices; lag selection by AIC, BIC and HQ; residual diagnostics;
  * Granger causality -- F tests in the Romanian VAR and in the market VAR; a Monte Carlo of the omitted-variable
                         pitfall;
  * IRF and FEVD      -- orthogonalised (Cholesky) and generalised impulse responses, the effect of the ordering,
                         forecast error variance decompositions, the Diebold-Yilmaz spillover index of S&P 500, DAX
                         and BET returns (rolling 200-day windows);
  * forecasting       -- VAR forecasts with intervals; pseudo out-of-sample comparison with AR models and the random
                         walk (Romania; the United States as in Stock and Watson 2001);
  * case study        -- the three-variable VAR of Stock and Watson (2001): inflation, unemployment, federal funds rate.
Output: charts/tsa_ch6_*.pdf/.png, Quantlets/Ch_06/ch6_numbers.json, ch6_granger_romania.csv, ch6_spillover_table.csv
References: Huang and Petukhina (2022), Ch. 7; Hyndman and Athanasopoulos, FPP3, Sec. 12.3; Luetkepohl (2005);
Hamilton (1994), Ch. 11; Sims (1980); Granger (1969); Stock and Watson (2001); Diebold and Yilmaz (2012).
Run:  python3 Quantlets/Ch_06/generate_all_charts.py
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
from tsa_data import load_close, load_panel, read_eurostat, read_fred   # noqa: E402
import tsa_style as st                                                   # noqa: E402
from statsmodels.tsa.api import VAR                                      # noqa: E402
from statsmodels.tsa.ar_model import AutoReg                             # noqa: E402

warnings.filterwarnings('ignore')
SEED = 2026
GDP_SA = ('namq_10_gdp', 'Q.CLV10_MEUR.SCA.B1GQ.RO')    # real GDP, chain-linked volumes (2010), seasonally adjusted
HICP = ('prc_hicp_minr', 'M.I15.TOTAL.RO')              # harmonised index of consumer prices, 2015 = 100
ROBOR = ('irt_st_m', 'M.IRT_M3.RO')                     # 3-month money market rate (ROBOR 3M), % per year
UNEMP = ('une_rt_m', 'M.SA.TOTAL.PC_ACT.T.RO')          # unemployment rate, seasonally adjusted, % of labour force
RO_START = '2005-01-01'          # inflation targeting since August 2005: the Romanian VAR starts in 2005Q1
RO_VARS = ['g', 'pi', 'i']       # GDP growth, inflation, ROBOR 3M (Cholesky order: slow to fast)
RO_P = 2                         # lag order of the Romanian VAR (chosen by HQ, see ic_romania)
MKT = ['sp500', 'dax', 'bet']
MKT_START = '2000-01-01'
MKT_P = 4                        # VAR(4), as in Diebold and Yilmaz (2012)
DY_H, DY_W = 10, 200             # Diebold-Yilmaz: 10-step forecast horizon, 200-day rolling windows
SW_VARS = ['pi', 'u', 'R']       # Stock and Watson (2001): inflation, unemployment, federal funds rate
SW_START, SW_END, SW_P = '1960-01-01', '2000-10-01', 4
LABEL = {'g': 'GDP growth', 'pi': 'inflation', 'i': 'ROBOR 3M', 'u': 'unemployment', 'ds': 'EUR/RON change',
         'R': 'fed funds rate', 'sp500': 'S&P 500', 'dax': 'DAX', 'bet': 'BET'}
COLV = {'g': st.Forest, 'pi': st.IDAred, 'i': st.MainBlue, 'u': st.Purple, 'ds': st.Amber, 'R': st.MainBlue,
        'sp500': st.COL['sp500'], 'dax': st.COL['dax'], 'bet': st.COL['bet']}
BAND = st.Teal


# =============================================================================
# DATA AND HELPERS
# =============================================================================
def ro_monthly():
    """Romanian monthly series: 12-month HICP inflation, ROBOR 3M, unemployment rate (SA), EUR/RON (monthly mean of the
    BNR reference rate)."""
    hicp = read_eurostat(*HICP)
    fx = load_close('eurron').resample('MS').mean()
    return pd.concat([(100 * np.log(hicp).diff(12)).rename('pi'), read_eurostat(*ROBOR).rename('i'),
                      read_eurostat(*UNEMP).rename('u'), fx.rename('eurron')], axis=1)


def ro_quarterly(start=RO_START):
    """Romanian quarterly data: real GDP growth g (q/q, %, SA), 12-month HICP inflation pi, ROBOR 3M i, unemployment u
    (quarterly means of monthly data) and the EUR/RON change ds = 100 ln(S_t / S_{t-1}) of the quarterly mean rate."""
    m = ro_monthly()
    q = m.resample('QS').mean()
    g = 100 * np.log(read_eurostat(*GDP_SA)).diff()
    d = pd.concat([g.rename('g'), q['pi'], q['i'], q['u'], (100 * np.log(q['eurron']).diff()).rename('ds')], axis=1)
    d.index.name = 'date'
    return d.loc[start:].dropna(subset=['g'])


def ro_var_data(cols=RO_VARS, start=RO_START):
    return ro_quarterly(start)[cols].dropna()


def market_returns(names=MKT, start=MKT_START):
    """Daily log returns (%) of several markets on their common trading days."""
    return load_panel(names, start=start, kind='returns').dropna()


def us_sw(start=SW_START, end=None):
    """Stock and Watson (2001) data from FRED: inflation pi = 400 ln(P_t / P_{t-1}) of the chain-type GDP price index
    (GDPCTPI), the civilian unemployment rate u (UNRATE) and the federal funds rate R (FEDFUNDS), quarterly means."""
    f = read_fred(['GDPCTPI', 'UNRATE', 'FEDFUNDS'])
    pi = 400 * np.log(f['GDPCTPI'].dropna()).diff()
    d = pd.concat([pi.rename('pi'), f['UNRATE'].resample('QS').mean().rename('u'),
                   f['FEDFUNDS'].resample('QS').mean().rename('R')], axis=1).dropna()
    d.index.name = 'date'
    return d.loc[start:end]


def fit_var(d, p):
    m = VAR(d.reset_index(drop=True))
    return m.fit(p)


def save(name, save_it=True):
    if save_it:
        st.check_no_grey(plt.gcf())
        st.save_fig(name)
    else:
        plt.show()


def years_axis(ax, step=5):
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator(step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))


def qlabel(ts):
    ts = pd.Timestamp(ts)
    return f'{ts.year}Q{(ts.month - 1) // 3 + 1}'


def ccf(y, x, kmax):
    """Cross-correlations rho_yx(k) = Corr(y_t, x_{t-k}), k = -kmax..kmax (k > 0: x leads y)."""
    return {k: float(pd.Series(y).corr(pd.Series(x).shift(k))) for k in range(-kmax, kmax + 1)}


def companion(res):
    """Companion matrix of a fitted VAR(p) (Kp x Kp)."""
    K, p = res.neqs, res.k_ar
    F = np.zeros((K * p, K * p))
    F[:K, :] = np.hstack(res.coefs)
    if p > 1:
        F[K:, :-K] = np.eye(K * (p - 1))
    return F


def ic_table(d, pmax=8):
    """AIC, BIC and HQ for p = 0..pmax on a common sample (T = N - pmax):
    IC(p) = ln det Sigma_ML(p) + c_T p K^2 / T, with c_T = 2 (AIC), ln T (BIC), 2 ln ln T (HQ)."""
    Y = d.values
    K, N = Y.shape[1], len(Y)
    T = N - pmax
    out = {}
    for p in range(pmax + 1):
        X = np.column_stack([np.ones(T)] + [Y[pmax - j:N - j] for j in range(1, p + 1)])
        Z = Y[pmax:]
        B = np.linalg.lstsq(X, Z, rcond=None)[0]
        U = Z - X @ B
        ld = float(np.log(np.linalg.det(U.T @ U / T)))
        pen = p * K ** 2 / T
        out[p] = {'logdet': ld, 'aic': ld + 2 * pen, 'bic': ld + np.log(T) * pen, 'hq': ld + 2 * np.log(np.log(T)) * pen}
    best = {c: int(min(out, key=lambda p: out[p][c])) for c in ('aic', 'bic', 'hq')}
    return {'T': T, 'K': K, 'pmax': pmax, 'table': out, 'best': best}


def granger_F(d, cause, effect, p):
    """Granger causality F test in the equation of `effect` of a VAR(p) with all variables of d:
    F = [(RSS_R - RSS_U)/p] / [RSS_U/(T - Kp - 1)], RSS_R without the p lags of `cause`."""
    Y = d.values
    cols = list(d.columns)
    K, N = Y.shape[1], len(Y)
    T = N - p
    lags = [Y[p - j:N - j] for j in range(1, p + 1)]
    X = np.column_stack([np.ones(T)] + lags)
    keep = [0] + [1 + (j - 1) * K + k for j in range(1, p + 1) for k in range(K) if cols[k] != cause]
    y = Y[p:, cols.index(effect)]
    rss = lambda Xm: float(np.sum((y - Xm @ np.linalg.lstsq(Xm, y, rcond=None)[0]) ** 2))
    rss_u, rss_r = rss(X), rss(X[:, keep])
    df2 = T - K * p - 1
    F = ((rss_r - rss_u) / p) / (rss_u / df2)
    return {'F': float(F), 'p': float(stats.f.sf(F, p, df2)), 'rss_u': rss_u, 'rss_r': rss_r, 'T': T, 'df1': p,
            'df2': df2, 'crit5': float(stats.f.ppf(0.95, p, df2))}


def granger_table(d, p):
    cols = list(d.columns)
    return {f'{c}->{e}': granger_F(d, c, e, p) for c in cols for e in cols if c != e}


def girf(res, H):
    """Generalised impulse responses (Pesaran and Shin 1998): response at horizon h to a shock of one standard deviation
    in variable j, Phi_h Sigma e_j / sqrt(sigma_jj). Array (H+1, K, K): [h, response, shock]."""
    S = np.asarray(res.sigma_u)
    ma = res.ma_rep(H)
    return np.array([ma[h] @ S / np.sqrt(np.diag(S))[None, :] for h in range(H + 1)])


def gfevd(res, H=DY_H):
    """Generalised FEVD normalised by rows (Diebold and Yilmaz 2012): theta[i, j] = share of the H-step forecast error
    variance of variable i due to shocks in variable j."""
    S = np.asarray(res.sigma_u)
    ma = res.ma_rep(H - 1)
    num = sum((ma[h] @ S) ** 2 for h in range(H)) / np.diag(S)[None, :]
    den = sum(np.diag(ma[h] @ S @ ma[h].T) for h in range(H))
    th = num / den[:, None]
    return th / th.sum(axis=1, keepdims=True)


def spill_index(th):
    K = th.shape[0]
    return 100 * (th.sum() - np.trace(th)) / K


def ar_forecast(y, h, pmax=4):
    """h-step forecast of an AR(p) with constant, p chosen by AIC (p <= pmax), iterated."""
    best = min(range(1, pmax + 1), key=lambda p: AutoReg(y, lags=p, trend='c').fit().aic)
    r = AutoReg(y, lags=best, trend='c').fit()
    return float(r.forecast(h)[-1]), best


def dm_test(e1, e2, h=1):
    """Diebold-Mariano test of equal squared-error loss, with a Newey-West variance of h-1 lags; returns (DM, p)."""
    dl = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(dl)
    m = dl.mean()
    g = [np.sum((dl[k:] - m) * (dl[:n - k] - m)) / n for k in range(h)]
    v = (g[0] + 2 * sum(g[1:])) / n
    dm = m / np.sqrt(v)
    return float(dm), float(2 * stats.norm.sf(abs(dm)))


# =============================================================================
# 1. MULTIVARIATE DATA
# =============================================================================
def fig_ro_macro(save_it=True):
    """Romanian macro series: GDP growth (quarterly), 12-month inflation and ROBOR 3M, unemployment, EUR/RON."""
    q = ro_quarterly('2000-01-01')
    m = ro_monthly().loc['2000-01-01':]
    fig, ax = plt.subplots(2, 2, figsize=(10.0, 5.4))
    a = ax[0, 0]
    a.bar(q.index, q['g'], width=70, color=st.Forest, label='real GDP growth, q/q (%)')
    a.axhline(0, color=st.DarkText, lw=0.6)
    a.set_title('Real GDP growth (quarterly, SA)')
    a = ax[0, 1]
    a.plot(m.index, m['pi'], color=st.IDAred, label='12-month HICP inflation (%)')
    a.plot(m.index, m['i'], color=st.MainBlue, label='ROBOR 3M (% p.a.)')
    a.set_ylim(-3, 32)
    a.set_title('Inflation and the 3-month interest rate')
    a = ax[1, 0]
    a.plot(m.index, m['u'], color=st.Purple, label='unemployment rate (%, SA)')
    a.set_title('Unemployment rate (monthly, SA)')
    a = ax[1, 1]
    a.plot(m.index, m['eurron'], color=st.Amber, label='EUR/RON (monthly mean)')
    a.set_title('EUR/RON, BNR reference rate')
    for a in ax.flat:
        years_axis(a, 5)
        a.set_xlim(pd.Timestamp('2000-01-01'), m.index[-1] + pd.Timedelta(days=60))
    st.fig_legend_bottom(fig, ncol=3, y=-0.01)
    plt.tight_layout()
    save('tsa_ch6_ro_macro', save_it)
    v = ro_var_data()
    mm = m.loc[RO_START:]
    return {'q_first': str(v.index[0])[:10], 'q_last': str(v.index[-1])[:10], 'T': len(v),
            'mean': v.mean().to_dict(), 'sd': v.std().to_dict(), 'corr': v.corr().round(4).to_dict(),
            'pi_max': float(mm['pi'].max()), 'pi_max_d': str(mm['pi'].idxmax())[:10],
            'i_max': float(mm['i'].max()), 'i_max_d': str(mm['i'].idxmax())[:10],
            'pi_last': float(m['pi'].dropna().iloc[-1]), 'pi_last_d': str(m['pi'].dropna().index[-1])[:10],
            'i_last': float(m['i'].dropna().iloc[-1]), 'u_last': float(m['u'].dropna().iloc[-1]),
            'u_last_d': str(m['u'].dropna().index[-1])[:10], 'fx_last': float(m['eurron'].dropna().iloc[-1]),
            'g_min': float(q['g'].loc[RO_START:].min()), 'g_min_d': str(q['g'].loc[RO_START:].idxmin())[:10],
            'g_min2_d': str(q['g'].loc[RO_START:].nsmallest(2).index[1])[:10],
            'g_min2': float(q['g'].loc[RO_START:].nsmallest(2).iloc[1])}


def fig_ccf(save_it=True):
    """Cross-correlation functions: BET and DAX against lags of the S&P 500 (daily returns); ROBOR 3M against lags of
    inflation (quarterly)."""
    r = market_returns()
    v = ro_var_data()
    kd, kq = 5, 8
    c_bet = ccf(r['bet'], r['sp500'], kd)
    c_dax = ccf(r['dax'], r['sp500'], kd)
    c_ip = ccf(v['i'], v['pi'], kq)
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.6))
    k = np.arange(-kd, kd + 1)
    ax[0].bar(k - 0.18, [c_dax[j] for j in k], width=0.36, color=st.COL['dax'], label=r'DAX$_t$ vs S&P 500$_{t-k}$')
    ax[0].bar(k + 0.18, [c_bet[j] for j in k], width=0.36, color=st.COL['bet'], label=r'BET$_t$ vs S&P 500$_{t-k}$')
    b = 1.96 / np.sqrt(len(r))
    ax[0].axhline(b, color=BAND, ls='--', lw=0.9, label=r'$\pm 1.96/\sqrt{T}$')
    ax[0].axhline(-b, color=BAND, ls='--', lw=0.9, label='_nolegend_')
    ax[0].axhline(0, color=st.DarkText, lw=0.6)
    ax[0].set_xlabel('lag k (days)')
    ax[0].set_title('Daily returns, 2000-2026')
    k2 = np.arange(-kq, kq + 1)
    ax[1].bar(k2, [c_ip[j] for j in k2], width=0.55, color=st.MainBlue, label=r'ROBOR$_t$ vs inflation$_{t-k}$')
    ax[1].axhline(0, color=st.DarkText, lw=0.6)
    ax[1].set_xlabel('lag k (quarters)')
    ax[1].set_title('Romania, quarterly, 2005-2026')
    for a in ax:
        a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    st.fig_legend_bottom(fig, ncol=4, y=-0.02)
    plt.tight_layout()
    save('tsa_ch6_ccf', save_it)
    kmax = max(c_ip, key=lambda j: c_ip[j])
    return {'bet': c_bet, 'dax': c_dax, 'i_pi': c_ip, 'band': float(b), 'T_daily': len(r), 'i_pi_kmax': int(kmax),
            'i_pi_max': float(c_ip[kmax]), 'first': str(r.index[0])[:10], 'last': str(r.index[-1])[:10]}


# =============================================================================
# 2. THE VAR(1) WORKED EXAMPLE
# =============================================================================
WX_A = np.array([[0.5, 0.2], [0.3, 0.4]])
WX_C = np.array([1.2, 0.6])
WX_S = np.array([[1.0, 0.5], [0.5, 1.0]])
WX_Y = np.array([5.0, 2.0])


def worked_example(A=WX_A, c=WX_C, S=WX_S, yT=WX_Y):
    """All numbers of the bivariate VAR(1) worked example: eigenvalues, mean, forecasts, impulse responses
    (simple and orthogonalised), forecast error variances and their decomposition."""
    lam = np.linalg.eigvals(A)
    mu = np.linalg.solve(np.eye(2) - A, c)
    f1 = c + A @ yT
    f2 = c + A @ f1
    A2 = A @ A
    P = np.linalg.cholesky(S)
    T0, T1 = P, A @ P
    mse1, mse2 = S, S + A @ S @ A.T
    fevd2 = (T0 ** 2 + T1 ** 2) / np.diag(mse2)[:, None]
    # reversed ordering: y2 first
    Pr = np.linalg.cholesky(S[::-1, ::-1])[::-1, ::-1]
    G = np.linalg.solve(np.eye(4) - np.kron(A, A), S.reshape(-1)).reshape(2, 2)   # Gamma(0): vec G = (I - A x A)^-1 vec S
    return {'lam': sorted(lam.real.tolist(), reverse=True), 'mu': mu.tolist(), 'f1': f1.tolist(), 'f2': f2.tolist(),
            'A2': A2.tolist(), 'P': P.tolist(), 'Theta1': T1.tolist(), 'mse2': mse2.tolist(), 'fevd2': fevd2.tolist(),
            'P_rev': Pr.tolist(), 'Gamma0': G.tolist(), 'det': float(np.linalg.det(np.eye(2) - A)),
            'tr': float(np.trace(A)), 'detA': float(np.linalg.det(A))}


def fig_var_sim(T=200, H=10, save_it=True):
    """A simulated path of the worked-example VAR(1) and its impulse responses Phi_h e_1 and the orthogonalised ones."""
    rng = np.random.default_rng(SEED)
    P = np.linalg.cholesky(WX_S)
    mu = np.linalg.solve(np.eye(2) - WX_A, WX_C)
    y = np.zeros((T + 100, 2))
    y[0] = mu
    for t in range(1, T + 100):
        y[t] = WX_C + WX_A @ y[t - 1] + P @ rng.standard_normal(2)
    y = y[100:]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.6), gridspec_kw={'width_ratios': [1.5, 1]})
    ax[0].plot(y[:, 0], color=st.MainBlue, label='$y_{1t}$')
    ax[0].plot(y[:, 1], color=st.IDAred, label='$y_{2t}$')
    ax[0].axhline(mu[0], color=st.MainBlue, ls='--', lw=0.9, label=r'$\mu_1$')
    ax[0].axhline(mu[1], color=st.IDAred, ls=':', lw=1.2, label=r'$\mu_2$')
    ax[0].set_xlabel('t')
    ax[0].set_title('Simulated VAR(1), T = 200')
    h = np.arange(H + 1)
    Ph = np.array([np.linalg.matrix_power(WX_A, j) for j in h])
    ax[1].plot(h, Ph[:, 0, 0], 'o-', color=st.MainBlue, ms=4, label=r'$y_1$ after a unit shock in $\varepsilon_1$')
    ax[1].plot(h, Ph[:, 1, 0], 's-', color=st.IDAred, ms=4, label=r'$y_2$ after a unit shock in $\varepsilon_1$')
    ax[1].axhline(0, color=st.DarkText, lw=0.6)
    ax[1].set_xlabel('horizon h')
    ax[1].set_title(r'Impulse responses $\Phi_h = A^h$')
    st.fig_legend_bottom(fig, ncol=3, y=-0.02)
    plt.tight_layout()
    save('tsa_ch6_var_sim', save_it)
    yy = pd.DataFrame(y, columns=['y1', 'y2'])
    r = VAR(yy).fit(1)
    return {'mean_sim': y.mean(axis=0).tolist(), 'corr_sim': float(np.corrcoef(y.T)[0, 1]),
            'A_hat': r.coefs[0].tolist(), 'c_hat': r.intercept.tolist(), 'phi_h5': Ph[5].tolist()}


# =============================================================================
# 3. STABILITY, LAG SELECTION, DIAGNOSTICS (ROMANIA)
# =============================================================================
def ic_romania(pmax=6):
    return ic_table(ro_var_data(), pmax)


def fig_ic(pmax=6, save_it=True):
    """AIC, BIC and HQ of the Romanian VAR for p = 0..pmax, on a common sample."""
    ic = ic_romania(pmax)
    p = np.arange(pmax + 1)
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    for c, col, mk, lab in [('aic', st.MainBlue, 'o', 'AIC'), ('hq', st.Forest, 's', 'HQ'), ('bic', st.IDAred, '^', 'BIC')]:
        vals = np.array([ic['table'][j][c] for j in p])
        ax.plot(p, vals, marker=mk, color=col, label=f"{lab} (minimum at p = {ic['best'][c]})")
        ax.plot(ic['best'][c], vals[ic['best'][c]], marker=mk, ms=12, mfc='none', mec=col, label='_nolegend_')
    ax.set_xlabel('lag order p')
    ax.set_ylabel('information criterion')
    ax.set_title('Romanian VAR (GDP growth, inflation, ROBOR 3M): lag selection')
    st.legend_outside_bottom(ax, ncol=3)
    plt.tight_layout()
    save('tsa_ch6_ic', save_it)
    return ic


def ro_fit(p=RO_P, cols=RO_VARS):
    return fit_var(ro_var_data(cols), p)


def fig_roots(save_it=True):
    """Eigenvalues of the companion matrices: Romanian VAR(2), Stock-Watson VAR(4), market VAR(4)."""
    fits = {'Romania VAR(2)': ro_fit(), 'US VAR(4), 1960-2000': fit_var(us_sw(SW_START, SW_END), SW_P),
            'S&P 500, DAX, BET VAR(4)': fit_var(market_returns(), MKT_P)}
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=st.DarkText, lw=0.9, label='unit circle')
    out = {}
    for (lab, r), col, mk in zip(fits.items(), [st.IDAred, st.MainBlue, st.Forest], ['o', 's', '^']):
        ev = np.linalg.eigvals(companion(r))
        ax.scatter(ev.real, ev.imag, color=col, marker=mk, s=36, label=lab, zorder=3)
        out[lab] = {'max_mod': float(np.abs(ev).max()), 'n': int(len(ev)), 'stable': bool(r.is_stable())}
    ax.axhline(0, color=st.DarkText, lw=0.5)
    ax.axvline(0, color=st.DarkText, lw=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel('real part')
    ax.set_ylabel('imaginary part')
    ax.set_title('Eigenvalues of the companion matrix')
    st.legend_outside_bottom(ax, ncol=2, y=-0.16)
    plt.tight_layout()
    save('tsa_ch6_roots', save_it)
    return out


def ro_estimates(p=RO_P):
    """The Romanian VAR(p): coefficients, standard errors, residual covariance and correlation, fit."""
    r = ro_fit(p)
    d = ro_var_data()
    return {'params': r.params.to_dict(), 'se': r.stderr.to_dict(), 'tvalues': r.tvalues.to_dict(),
            'sigma_u': np.asarray(r.sigma_u).tolist(), 'resid_corr': np.corrcoef(np.asarray(r.resid).T).tolist(),
            'nobs': int(r.nobs), 'first_used': str(d.index[p])[:10], 'last': str(d.index[-1])[:10],
            'n_coef': int(r.params.size), 'r2': {c: float(1 - np.var(np.asarray(r.resid)[:, j]) / np.var(d[c].values[p:]))
                                                 for j, c in enumerate(RO_VARS)}}


def fig_resid(p=RO_P, save_it=True):
    """Residuals of the Romanian VAR(2) with +-2 standard deviations; Portmanteau and normality tests."""
    r = ro_fit(p)
    d = ro_var_data()
    U = pd.DataFrame(np.asarray(r.resid), index=d.index[p:], columns=RO_VARS)
    fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.2))
    out = {'largest': {}}
    for a, c in zip(ax, RO_VARS):
        s = U[c].std()
        a.bar(U.index, U[c], width=70, color=COLV[c], label=f'residual, {LABEL[c]} equation')
        a.axhline(2 * s, color=BAND, ls='--', lw=0.9, label=r'$\pm 2\hat\sigma$' if c == 'g' else '_nolegend_')
        a.axhline(-2 * s, color=BAND, ls='--', lw=0.9, label='_nolegend_')
        a.axhline(0, color=st.DarkText, lw=0.6)
        a.set_title(f'{LABEL[c]} equation')
        years_axis(a, 5)
        j = U[c].abs().idxmax()
        out['largest'][c] = {'date': str(j)[:10], 'value': float(U[c].loc[j]), 'sd': float(s)}
    st.fig_legend_bottom(fig, ncol=4, y=-0.03)
    plt.tight_layout()
    save('tsa_ch6_resid', save_it)
    for h in (8, 12):
        w = r.test_whiteness(nlags=h, adjusted=True)
        out[f'lb{h}'] = {'stat': float(w.test_statistic), 'df': int(w.df), 'p': float(w.pvalue)}
    nt = r.test_normality()
    out['jb'] = {'stat': float(nt.test_statistic), 'df': int(nt.df), 'p': float(nt.pvalue)}
    out['skew'] = U.skew().to_dict()
    out['kurt'] = (U.kurt() + 3).to_dict()
    # without the two crisis quarters (2009Q1 and 2020Q2): dummies as exogenous variables
    dum = pd.DataFrame({'d09': (d.index == '2009-01-01').astype(float), 'd20': (d.index == '2020-04-01').astype(float)},
                       index=d.index)
    r2 = VAR(d.reset_index(drop=True), exog=dum.reset_index(drop=True)).fit(p)
    nt2 = r2.test_normality()
    out['jb_dummies'] = {'stat': float(nt2.test_statistic), 'p': float(nt2.pvalue)}
    return out


# =============================================================================
# 4. GRANGER CAUSALITY
# =============================================================================
def granger_romania(p=RO_P, write=False):
    d = ro_var_data()
    G = granger_table(d, p)
    r = ro_fit(p)
    inst = {c: float(r.test_inst_causality(c).pvalue) for c in RO_VARS}
    rows = [{'cause': k.split('->')[0], 'effect': k.split('->')[1], 'F': v['F'], 'p': v['p']} for k, v in G.items()]
    if write:
        pd.DataFrame(rows).to_csv(os.path.join(HERE, 'ch6_granger_romania.csv'), index=False, float_format='%.4f')
    # bivariate tests (the same pairs, two variables only)
    bi = {k: granger_F(d[[k.split('->')[0], k.split('->')[1]]], k.split('->')[0], k.split('->')[1], p)['p'] for k in G}
    sm = r.test_causality('i', ['g'], kind='wald')
    return {'tests': G, 'inst': inst, 'bivariate_p': bi, 'wald_g_i': {'stat': float(sm.test_statistic), 'p': float(sm.pvalue)}}


def granger_markets(p=MKT_P):
    r = market_returns()
    G = granger_table(r, p)
    res = fit_var(r, p)
    inst = float(res.test_inst_causality('sp500').pvalue)
    so = VAR(r.reset_index(drop=True)).select_order(10).selected_orders
    return {'tests': G, 'inst_sp500': inst, 'T': len(r), 'orders': {k: int(v) for k, v in so.items()},
            'coef_bet_sp1': float(res.params.loc['L1.sp500', 'bet']), 'coef_dax_sp1': float(res.params.loc['L1.sp500', 'dax']),
            'se_bet_sp1': float(res.stderr.loc['L1.sp500', 'bet']), 'corr': r.corr().to_dict()}


def granger_by_hand(p=RO_P):
    """The worked F test: does GDP growth Granger-cause ROBOR 3M in the Romanian VAR?"""
    return granger_F(ro_var_data(), 'g', 'i', p)


def omitted_sim(Ts=(50, 100, 200, 500), R=1000, p=2):
    """Monte Carlo of the omitted-variable pitfall: z_t = 0.5 z_{t-1} + e_t; x_t = 0.8 z_{t-1} + u_t;
    y_t = 0.8 z_{t-2} + v_t. x does not Granger-cause y given z, but in the bivariate (x, y) VAR it appears to."""
    rng = np.random.default_rng(SEED)
    out = {}
    for T in Ts:
        rej_bi = rej_tri = 0
        for _ in range(R):
            n = T + 50
            e = rng.standard_normal((n, 3))
            z = np.zeros(n)
            for t in range(1, n):
                z[t] = 0.5 * z[t - 1] + e[t, 0]
            x = np.r_[0, 0.8 * z[:-1]] + e[:, 1]
            y = np.r_[0, 0, 0.8 * z[:-2]] + e[:, 2]
            dd = pd.DataFrame({'x': x, 'y': y, 'z': z}).iloc[50:]
            rej_bi += granger_F(dd[['x', 'y']], 'x', 'y', p)['p'] < 0.05
            rej_tri += granger_F(dd, 'x', 'y', p)['p'] < 0.05
        out[T] = {'bivariate': rej_bi / R, 'trivariate': rej_tri / R}
    return out


def fig_granger_sim(save_it=True):
    sim = omitted_sim()
    Ts = list(sim)
    x = np.arange(len(Ts))
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.bar(x - 0.18, [100 * sim[T]['bivariate'] for T in Ts], width=0.36, color=st.IDAred,
           label='bivariate VAR (x, y): z omitted')
    ax.bar(x + 0.18, [100 * sim[T]['trivariate'] for T in Ts], width=0.36, color=st.MainBlue,
           label='trivariate VAR (x, y, z)')
    ax.axhline(5, color=BAND, ls='--', lw=1.0, label='nominal size 5%')
    ax.set_xticks(x)
    ax.set_xticklabels([f'T = {T}' for T in Ts])
    ax.set_ylabel('rejection rate (%)')
    ax.set_title('rejections of "x does not Granger-cause y"; x never helps once z is known', fontsize=11)
    st.legend_outside_bottom(ax, ncol=3)
    plt.tight_layout()
    save('tsa_ch6_granger_sim', save_it)
    return {str(k): v for k, v in sim.items()}


# =============================================================================
# 5. IMPULSE RESPONSES AND FEVD
# =============================================================================
def boot_irf(res, H, repl=500, signif=0.10, seed=SEED):
    """Residual bootstrap bands for orthogonalised impulse responses: resample the VAR residuals with replacement,
    rebuild the series from the first p observations, re-estimate the VAR(p), recompute the responses;
    percentile band of level 1 - signif."""
    rng = np.random.default_rng(seed)
    p, K = res.k_ar, res.neqs
    Y = np.asarray(res.endog)
    U = np.asarray(res.resid)
    U = U - U.mean(axis=0)
    c, A = res.intercept, res.coefs
    draws = []
    for _ in range(repl):
        y = np.zeros_like(Y)
        y[:p] = Y[:p]
        e = U[rng.integers(0, len(U), len(Y) - p)]
        for t in range(p, len(Y)):
            y[t] = c + sum(A[j] @ y[t - 1 - j] for j in range(p)) + e[t - p]
        draws.append(VAR(y).fit(p).irf(H).orth_irfs)
    draws = np.array(draws)
    return np.quantile(draws, signif / 2, axis=0), np.quantile(draws, 1 - signif / 2, axis=0)


def irf_panel(res, names, H, fname, title, repl=500, signif=0.10, save_it=True, scale=None):
    """Orthogonalised impulse responses (K x K panels) with residual-bootstrap bands."""
    irf = res.irf(H)
    lo, hi = boot_irf(res, H, repl, signif)
    K = len(names)
    fig, ax = plt.subplots(K, K, figsize=(10.0, 6.0), sharex=True)
    h = np.arange(H + 1)
    for i in range(K):
        for j in range(K):
            a = ax[i, j]
            a.fill_between(h, lo[:, i, j], hi[:, i, j], color=BAND, alpha=0.25,
                           label=f'{int(100 * (1 - signif))}% band' if (i, j) == (0, 0) else '_nolegend_')
            a.plot(h, irf.orth_irfs[:, i, j], color=COLV[names[j]], lw=1.6,
                   label='orthogonalised response' if (i, j) == (0, 0) else '_nolegend_')
            a.axhline(0, color=st.DarkText, lw=0.6)
            a.set_title(f'{LABEL[names[i]]} <- {LABEL[names[j]]} shock', fontsize=11)
            a.tick_params(labelsize=10)
            a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for a in ax[-1]:
        a.set_xlabel('quarters' if H < 50 else 'days', fontsize=11)
    fig.suptitle(title, fontsize=13)
    st.fig_legend_bottom(fig, ncol=2, y=-0.01)
    plt.tight_layout()
    save(fname, save_it)
    return irf.orth_irfs, lo, hi


def fig_irf_ro(H=12, save_it=True):
    r = ro_fit()
    o, lo, hi = irf_panel(r, RO_VARS, H, 'tsa_ch6_irf_ro', 'Romanian VAR(2), Cholesky order: GDP growth, inflation, ROBOR 3M',
                          save_it=save_it)
    k = {c: j for j, c in enumerate(RO_VARS)}
    pick = lambda i, j: [float(x) for x in o[:, k[i], k[j]]]
    sig = lambda i, j: [bool(lo[h, k[i], k[j]] > 0 or hi[h, k[i], k[j]] < 0) for h in range(H + 1)]
    P = np.linalg.cholesky(np.asarray(r.sigma_u))
    return {'pi_i': pick('pi', 'i'), 'i_pi': pick('i', 'pi'), 'g_i': pick('g', 'i'), 'i_g': pick('i', 'g'),
            'pi_pi': pick('pi', 'pi'), 'i_i': pick('i', 'i'), 'g_g': pick('g', 'g'), 'pi_g': pick('pi', 'g'),
            'g_pi': pick('g', 'pi'),
            'sig_pi_i': sig('pi', 'i'), 'sig_i_pi': sig('i', 'pi'), 'sig_g_i': sig('g', 'i'),
            'P': P.tolist(), 'sd': np.sqrt(np.diag(np.asarray(r.sigma_u))).tolist()}


def fig_irf_order(H=12, save_it=True):
    """The ordering matters: responses of ROBOR to an inflation shock and of inflation to a ROBOR shock with the order
    (g, pi, i) and with the order (g, i, pi), and the generalised responses."""
    d = ro_var_data()
    rA = fit_var(d[['g', 'pi', 'i']], RO_P)
    rB = fit_var(d[['g', 'i', 'pi']], RO_P)
    oA, oB = rA.irf(H).orth_irfs, rB.irf(H).orth_irfs
    G = girf(rA, H)
    h = np.arange(H + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.6))
    ax[0].plot(h, oA[:, 2, 1], 'o-', ms=3.5, color=st.MainBlue, label='order g, inflation, ROBOR')
    ax[0].plot(h, oB[:, 1, 2], 's--', ms=3.5, color=st.IDAred, label='order g, ROBOR, inflation')
    ax[0].plot(h, G[:, 2, 1], '^:', ms=3.5, color=st.Forest, label='generalised (order-free)')
    ax[0].set_title('ROBOR 3M <- inflation shock')
    ax[1].plot(h, oA[:, 1, 2], 'o-', ms=3.5, color=st.MainBlue, label='_nolegend_')
    ax[1].plot(h, oB[:, 2, 1], 's--', ms=3.5, color=st.IDAred, label='_nolegend_')
    ax[1].plot(h, G[:, 1, 2], '^:', ms=3.5, color=st.Forest, label='_nolegend_')
    ax[1].set_title('inflation <- ROBOR 3M shock')
    for a in ax:
        a.axhline(0, color=st.DarkText, lw=0.6)
        a.set_xlabel('quarters')
        a.set_ylabel('percentage points')
    st.fig_legend_bottom(fig, ncol=3, y=-0.02)
    plt.tight_layout()
    save('tsa_ch6_irf_order', save_it)
    return {'i_pi_A': oA[:, 2, 1].tolist(), 'i_pi_B': oB[:, 1, 2].tolist(), 'i_pi_G': G[:, 2, 1].tolist(),
            'pi_i_A': oA[:, 1, 2].tolist(), 'pi_i_B': oB[:, 2, 1].tolist(), 'pi_i_G': G[:, 1, 2].tolist(),
            'corr_pi_i': float(np.corrcoef(np.asarray(rA.resid).T)[1, 2])}


def fig_fevd_ro(H=12, save_it=True):
    r = ro_fit()
    fe = r.fevd(H).decomp            # [variable, horizon, shock]
    h = np.arange(1, H + 1)
    fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.4), sharey=True)
    for i, (a, c) in enumerate(zip(ax, RO_VARS)):
        bottom = np.zeros(H)
        for j, s in enumerate(RO_VARS):
            a.bar(h, 100 * fe[i, :, j], bottom=bottom, color=COLV[s], width=0.75,
                  label=f'{LABEL[s]} shock' if i == 0 else '_nolegend_')
            bottom += 100 * fe[i, :, j]
        a.set_title(f'{LABEL[c]}')
        a.set_xlabel('horizon (quarters)')
        a.set_ylim(0, 100)
    ax[0].set_ylabel('share of forecast error variance (%)')
    st.fig_legend_bottom(fig, ncol=3, y=-0.02)
    plt.tight_layout()
    save('tsa_ch6_fevd_ro', save_it)
    return {c: {str(hh): (100 * fe[i, hh - 1, :]).tolist() for hh in (1, 4, 8, 12)} for i, c in enumerate(RO_VARS)}


def fig_market_irf(H=5, save_it=True):
    """Responses of DAX and BET returns to an S&P 500 shock: generalised and Cholesky with two orderings
    (S&P 500 first; the order of the closing times BET, DAX, S&P 500)."""
    r = market_returns()
    rA = fit_var(r[['sp500', 'dax', 'bet']], MKT_P)
    rB = fit_var(r[['bet', 'dax', 'sp500']], MKT_P)
    oA, oB, G = rA.irf(H).orth_irfs, rB.irf(H).orth_irfs, girf(rA, H)
    h = np.arange(H + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.6))
    for a, (iA, iB, nm) in zip(ax, [(1, 1, 'dax'), (2, 0, 'bet')]):
        a.bar(h - 0.25, G[:, iA, 0], width=0.25, color=st.Forest, label='generalised')
        a.bar(h, oA[:, iA, 0], width=0.25, color=st.MainBlue, label='Cholesky, S&P 500 first')
        a.bar(h + 0.25, oB[:, iB, 2], width=0.25, color=st.IDAred, label='Cholesky, S&P 500 last (closing-time order)')
        a.axhline(0, color=st.DarkText, lw=0.6)
        a.set_title(f'{LABEL[nm]} <- S&P 500 shock')
        a.set_xlabel('days')
        a.set_ylabel('% return')
    st.fig_legend_bottom(fig, ncol=3, y=-0.02)
    plt.tight_layout()
    save('tsa_ch6_market_irf', save_it)
    sd = np.sqrt(np.diag(np.asarray(rA.sigma_u)))
    return {'G_dax': G[:, 1, 0].tolist(), 'G_bet': G[:, 2, 0].tolist(), 'A_dax': oA[:, 1, 0].tolist(),
            'A_bet': oA[:, 2, 0].tolist(), 'B_dax': oB[:, 1, 2].tolist(), 'B_bet': oB[:, 0, 2].tolist(),
            'sd': dict(zip(['sp500', 'dax', 'bet'], sd.tolist())),
            'corr_u': np.corrcoef(np.asarray(rA.resid).T).tolist()}


def spillover_table(r=None, write=False):
    r = market_returns() if r is None else r
    th = gfevd(fit_var(r, MKT_P), DY_H)
    K = th.shape[0]
    tab = pd.DataFrame(100 * th, index=[LABEL[c] for c in r.columns], columns=[LABEL[c] for c in r.columns])
    tab['from others'] = tab.sum(axis=1) - np.diag(tab.values)
    to = tab.iloc[:, :K].sum(axis=0) - np.diag(tab.values[:, :K])
    tab.loc['to others'] = list(to.values) + [spill_index(th)]
    if write:
        tab.to_csv(os.path.join(HERE, 'ch6_spillover_table.csv'), float_format='%.1f')
    return th, tab


def rolling_spillover(r=None, W=DY_W, step=1):
    r = market_returns() if r is None else r
    vals = {}
    for e in range(W, len(r) + 1, step):
        vals[r.index[e - 1]] = spill_index(gfevd(fit_var(r.iloc[e - W:e], MKT_P), DY_H))
    return pd.Series(vals)


def fig_spillover(save_it=True):
    r = market_returns()
    th, tab = spillover_table(r, write=save_it)
    s = rolling_spillover(r)
    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    ax.plot(s.index, s.values, color=st.MainBlue, lw=1.0, label='total spillover index, 200-day windows (%)')
    ax.axhline(spill_index(th), color=st.IDAred, ls='--', lw=1.0, label='full sample (%)')
    for dte, lab in [('2008-09-15', 'Lehman'), ('2011-08-05', 'US downgrade'), ('2020-03-11', 'COVID-19'),
                     ('2022-02-24', 'Ukraine')]:
        ax.axvline(pd.Timestamp(dte), color=st.Amber, lw=0.8, ls=':')
        ax.text(pd.Timestamp(dte), s.max() * 1.02, lab, color=st.DarkText, fontsize=10, ha='center')
    ax.set_ylim(0, s.max() * 1.12)
    ax.set_xlim(s.index[0] - pd.Timedelta(days=120), s.index[-1] + pd.Timedelta(days=120))
    years_axis(ax, 2)
    ax.set_title('Diebold-Yilmaz spillover index of S&P 500, DAX and BET returns')
    st.legend_outside_bottom(ax, ncol=2)
    plt.tight_layout()
    save('tsa_ch6_spillover', save_it)
    yr = s.groupby(s.index.year).mean()
    return {'table': {i: tab.loc[i].to_dict() for i in tab.index}, 'total': float(spill_index(th)),
            'roll_mean': float(s.mean()), 'roll_max': float(s.max()), 'roll_max_d': str(s.idxmax())[:10],
            'roll_min': float(s.min()), 'roll_min_d': str(s.idxmin())[:10], 'roll_last': float(s.iloc[-1]),
            'year_max': int(yr.idxmax()), 'year_min': int(yr.idxmin()), 'n_windows': int(len(s)),
            'v2007': float(s.loc['2007'].mean()), 'v2008q4': float(s.loc['2008-10':'2008-12'].mean()),
            'v2020': float(s.loc['2020-03':'2020-06'].mean())}


# =============================================================================
# 6. FORECASTING
# =============================================================================
def fig_forecast_ro(H=8, save_it=True):
    d = ro_var_data()
    r = fit_var(d, RO_P)
    pt, lo, hi = r.forecast_interval(d.values[-RO_P:], H, alpha=0.05)
    fidx = pd.date_range(d.index[-1] + pd.offsets.QuarterBegin(startingMonth=1), periods=H, freq='QS')
    fig, ax = plt.subplots(1, 3, figsize=(10.0, 3.4))
    for j, (a, c) in enumerate(zip(ax, RO_VARS)):
        hist = d[c].loc['2019-01-01':]
        a.plot(hist.index, hist, color=COLV[c], label='data' if j == 0 else '_nolegend_')
        a.plot(fidx, pt[:, j], color=st.DarkText, ls='--', label='VAR(2) forecast' if j == 0 else '_nolegend_')
        a.fill_between(fidx, lo[:, j], hi[:, j], color=BAND, alpha=0.25, label='95% interval' if j == 0 else '_nolegend_')
        a.set_title(LABEL[c])
        years_axis(a, 2)
    st.fig_legend_bottom(fig, ncol=3, y=-0.02)
    plt.tight_layout()
    save('tsa_ch6_forecast_ro', save_it)
    mu = np.linalg.solve(np.eye(3) - sum(r.coefs), r.intercept)
    return {'first_f': qlabel(fidx[0]), 'last_f': qlabel(fidx[-1]), 'last_obs': qlabel(d.index[-1]),
            'last': d.iloc[-1].to_dict(), 'pt': pt.tolist(), 'lo': lo.tolist(), 'hi': hi.tolist(),
            'mu': dict(zip(RO_VARS, mu.tolist()))}


def oos(d, p, start, hs, pmax_ar=4):
    """Pseudo out-of-sample forecasts with an expanding window: VAR(p), AR(p_AIC) for each variable and the random walk
    (no change). Returns RMSE by model, variable and horizon, and the forecast errors."""
    cols = list(d.columns)
    i0 = d.index.get_loc(pd.Timestamp(start))
    err = {m: {c: {h: [] for h in hs} for c in cols} for m in ('VAR', 'AR', 'RW')}
    for t in range(i0, len(d) + 1):          # origin: last observation at t-1
        train = d.iloc[:t]
        r = fit_var(train, p)
        fc = r.forecast(train.values[-p:], max(hs))
        for h in hs:
            if t - 1 + h >= len(d):
                continue
            actual = d.iloc[t - 1 + h]
            for j, c in enumerate(cols):
                y = train[c].values
                err['VAR'][c][h].append(actual[c] - fc[h - 1, j])
                err['AR'][c][h].append(actual[c] - ar_forecast(y, h, pmax_ar)[0])
                err['RW'][c][h].append(actual[c] - y[-1])
    rmse = {m: {c: {h: float(np.sqrt(np.mean(np.square(err[m][c][h])))) for h in hs} for c in cols} for m in err}
    return rmse, err


def fig_oos(save_it=True):
    """Relative RMSE (model / random walk) of VAR and AR forecasts: Romania (2015Q1-2026Q2, h = 1, 4) and the United
    States (Stock-Watson evaluation period 1985Q1-2000Q4, h = 2, 4, 8)."""
    dro = ro_var_data()
    rro, ero = oos(dro, RO_P, '2015-01-01', (1, 4))
    dus = us_sw(SW_START, SW_END)
    rus, eus = oos(dus, SW_P, '1985-01-01', (2, 4, 8))
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.8), gridspec_kw={'width_ratios': [1, 1.4]})
    for a, rm, cols, hs, ttl in [(ax[0], rro, RO_VARS, (1, 4), 'Romania, forecasts for 2015-2026'),
                                 (ax[1], rus, SW_VARS, (2, 4, 8), 'United States, forecasts for 1985-2000')]:
        groups = [(c, h) for c in cols for h in hs]
        x = np.arange(len(groups))
        a.bar(x - 0.18, [rm['AR'][c][h] / rm['RW'][c][h] for c, h in groups], width=0.36, color=st.Amber, label='AR / random walk')
        a.bar(x + 0.18, [rm['VAR'][c][h] / rm['RW'][c][h] for c, h in groups], width=0.36, color=st.MainBlue,
              label='VAR / random walk')
        a.axhline(1, color=st.IDAred, ls='--', lw=1.0, label='random walk = 1')
        a.set_xticks(x)
        a.set_xticklabels([f'{c}\nh={h}'.replace('pi', 'π') for c, h in groups], fontsize=10)
        a.set_title(ttl)
        a.set_ylabel('relative RMSE')
    handles, labels = ax[0].get_legend_handles_labels()
    st.fig_legend_bottom(fig, handles, labels, ncol=3, y=-0.02)
    plt.tight_layout()
    save('tsa_ch6_oos', save_it)
    dm = {c: dm_test(ero['VAR'][c][1], ero['AR'][c][1], 1) for c in RO_VARS}
    dmu = {c: dm_test(eus['VAR'][c][4], eus['AR'][c][4], 4) for c in SW_VARS}
    return {'ro': rro, 'us': rus, 'n_ro': len(ero['VAR']['g'][1]), 'n_us4': len(eus['VAR']['pi'][4]),
            'dm_ro': dm, 'dm_us4': dmu}


# =============================================================================
# 7. CASE STUDY: STOCK AND WATSON (2001)
# =============================================================================
def fig_us_data(save_it=True):
    d = us_sw(SW_START)
    fig, ax = plt.subplots(figsize=(10.0, 3.6))
    for c, col in [('pi', st.IDAred), ('u', st.Purple), ('R', st.MainBlue)]:
        ax.plot(d.index, d[c], color=col, lw=1.1, label={'pi': 'inflation (GDP price index, annualised %)',
                                                           'u': 'unemployment rate (%)', 'R': 'federal funds rate (%)'}[c])
    ax.axvspan(pd.Timestamp(SW_START), pd.Timestamp(SW_END), color=st.Teal, alpha=0.08, label='Stock-Watson sample 1960-2000')
    ax.axhline(0, color=st.DarkText, lw=0.6)
    years_axis(ax, 5)
    ax.set_title('United States, quarterly, FRED')
    st.legend_outside_bottom(ax, ncol=2)
    plt.tight_layout()
    save('tsa_ch6_us_data', save_it)
    return {'first': qlabel(d.index[0]), 'last': qlabel(d.index[-1]), 'T_sw': len(us_sw(SW_START, SW_END)),
            'T_all': len(d)}


def fig_sw_irf(H=24, save_it=True):
    d = us_sw(SW_START, SW_END)
    r = fit_var(d, SW_P)
    o, lo, hi = irf_panel(r, SW_VARS, H, 'tsa_ch6_sw_irf', 'Stock-Watson VAR(4), 1960Q1-2000Q4, order: inflation, unemployment, fed funds',
                          save_it=save_it)
    G = granger_table(d, SW_P)
    fe = r.fevd(12).decomp
    full = us_sw(SW_START)
    rf = fit_var(full, SW_P)
    of = rf.irf(H).orth_irfs
    Gf = granger_table(full, SW_P)
    k = {c: j for j, c in enumerate(SW_VARS)}
    return {'granger': {kk: v['p'] for kk, v in G.items()}, 'granger_full': {kk: v['p'] for kk, v in Gf.items()},
            'fevd': {c: {str(hh): (100 * fe[k[c], hh - 1, :]).tolist() for hh in (1, 4, 8, 12)} for c in SW_VARS},
            'u_R': o[:, k['u'], k['R']].tolist(), 'pi_R': o[:, k['pi'], k['R']].tolist(), 'R_R': o[:, k['R'], k['R']].tolist(),
            'R_pi': o[:, k['R'], k['pi']].tolist(), 'R_u': o[:, k['R'], k['u']].tolist(),
            'u_R_full': of[:, k['u'], k['R']].tolist(), 'pi_R_full': of[:, k['pi'], k['R']].tolist(),
            'sig_pi_R': [bool(lo[h, 0, 2] > 0 or hi[h, 0, 2] < 0) for h in range(H + 1)],
            'sig_u_R': [bool(lo[h, 1, 2] > 0 or hi[h, 1, 2] < 0) for h in range(H + 1)],
            'max_mod': float(np.abs(np.linalg.eigvals(companion(r))).max()),
            'max_mod_full': float(np.abs(np.linalg.eigvals(companion(rf))).max()), 'T': int(r.nobs),
            'T_full': int(rf.nobs), 'last_full': qlabel(full.index[-1]), 'n_coef': int(r.params.size)}


if __name__ == '__main__':
    st.apply()
    N = {}
    for name, f in [('ro', fig_ro_macro), ('ccf', fig_ccf), ('wx', worked_example), ('sim', fig_var_sim),
                    ('ic', fig_ic), ('roots', fig_roots), ('est', ro_estimates), ('resid', fig_resid),
                    ('gr_ro', lambda: granger_romania(write=True)), ('gr_mkt', granger_markets), ('gr_hand', granger_by_hand),
                    ('gr_sim', fig_granger_sim), ('irf', fig_irf_ro), ('order', fig_irf_order), ('fevd', fig_fevd_ro),
                    ('mirf', fig_market_irf), ('spill', fig_spillover), ('fc', fig_forecast_ro), ('oos', fig_oos),
                    ('us', fig_us_data), ('sw', fig_sw_irf)]:
        print(name)
        N[name] = f()
    with open(os.path.join(HERE, 'ch6_numbers.json'), 'w') as fh:
        json.dump(N, fh, indent=1, default=float)
    print('written ch6_numbers.json')
