"""
figs_practice.py -- charts, software outputs and numbers of the public TSA practice set
(exam/practice/probleme_examen_ro.tex, exam/practice/exam_problems_en.tex). Called by exam/make_figs.py:
    make(fig_dir, out_dir) -> dict of numbers (saved in exam/practice/practice_numbers.json)
Data: Quantlets/common/tsa_data.py (data/market, last day 18.09.2026; FRED, Eurostat, statsmodels data sets online).

Time Series Analysis - Daniel Traian PELE
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import exam_common as ec  # noqa: E402
from exam_common import td, st  # noqa: E402

warnings.filterwarnings('ignore')
END = '2026-09-18'
TOUR = ('tour_occ_nim', 'M.TOTAL.NR.I551-I553.RO')


def r(v, d=4):
    return round(float(v), d)


def lb_by_hand(a, n):
    return float(n * (n + 2) * sum(ak ** 2 / (n - k) for k, ak in enumerate(a, start=1)))


# ============================================================================================ chapter 1
def ch1(num, fig_dir, out_dir):
    from statsmodels.tsa.stattools import acf
    nile = td.load_statsmodels('nile')
    ec.write_out(ec.correlogram(nile.values, 10, 'Correlogram of the Nile annual flow, 1871-1970'), out_dir, 'p_nile_corr')
    a = acf(nile.values, nlags=10, fft=False)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(nile.values, lags=[10])
    num['nile_n'] = len(nile)
    num['nile_acf_1_3'] = [r(v, 3) for v in a[1:4]]
    num['nile_band'] = r(1.96 / np.sqrt(len(nile)), 3)
    num['nile_q10'] = r(lb['lb_stat'].iloc[0], 2)
    num['nile_nsig'] = int(np.sum(np.abs(a[1:11]) > 1.96 / np.sqrt(len(nile))))
    num['nile_mean'] = r(nile.mean(), 1)

    # US 10Y yield: level and daily change, 2015-2026
    y = td.load_close('us10y', start='2015-01-01')
    dy = (100 * y.diff()).dropna()     # basis points
    al = acf(y.values, nlags=5, fft=False)[1:]
    ad = acf(dy.values, nlags=5, fft=False)[1:]
    n = len(dy)
    num['us10y_n'] = n
    num['us10y_acf_level_1_5'] = [r(v, 3) for v in al]
    num['us10y_acf_diff_1_5'] = [r(v, 3) for v in ad]
    ad3 = [r(v, 3) for v in ad]
    num['us10y_q5_hand'] = r(lb_by_hand(ad3, n), 2)
    num['us10y_band'] = r(1.96 / np.sqrt(n), 3)
    num['us10y_first'] = r(y.iloc[0], 2)
    num['us10y_last'] = r(y.iloc[-1], 2)
    num['us10y_sd_diff_bp'] = r(dy.std(), 2)
    for lang in ('ro', 'en'):
        L = dict(ro=('randamentul la 10 ani (%)', 'variația zilnică (puncte de bază)'),
                 en=('10-year yield (%)', 'daily change (basis points)'))[lang]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 4.0), sharex=True)
        a1.plot(y.index, y.values, color=st.MainBlue, lw=1.0, label=L[0])
        a2.plot(dy.index, dy.values, color=st.IDAred, lw=0.5, label=L[1])
        a1.yaxis.set_major_formatter(ec.fmt(lang)(1))
        a2.yaxis.set_major_formatter(ec.fmt(lang)(0))
        st.fig_legend_bottom(fig, ncol=2, y=0.02)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        ec.save(fig, fig_dir, f'practice_us10y_{lang}')


# ============================================================================================ chapter 2
def ch2(num, fig_dir, out_dir):
    from statsmodels.tsa.arima.model import ARIMA
    s = td.load_statsmodels('sunspots')
    res = ARIMA(s.values, order=(2, 0, 0), trend='c').fit()
    c, p1, p2, s2 = res.params
    txt = ec.coef_table([('const', c, res.bse[0]), ('ar.L1', p1, res.bse[1]), ('ar.L2', p2, res.bse[2]),
                         ('sigma2', s2, res.bse[3])],
                        title='ARIMA(2,0,0) with constant, sunspots 1700-2008 (n = %d)\n'
                              'Log likelihood %.2f   AIC %.2f   BIC %.2f\n' % (len(s), res.llf, res.aic, res.bic))
    ec.write_out(txt, out_dir, 'p_sunspots_ar2')
    num.update(sun_n=len(s), sun_c=r(c, 3), sun_phi1=r(p1, 4), sun_phi2=r(p2, 4), sun_sigma2=r(s2, 2))
    P1, P2 = round(p1, 3), round(p2, 3)
    num['sun_disc'] = r(P1 ** 2 + 4 * P2, 4)
    num['sun_mu'] = r(c, 2)   # statsmodels 'const' is the mean
    cosw = P1 / (2 * np.sqrt(-P2))
    num['sun_cos'] = r(cosw, 4)
    num['sun_period'] = r(2 * np.pi / np.arccos(cosw), 2)
    num['sun_last2'] = [r(v, 1) for v in s.values[-2:]]
    mu = round(c, 2)
    x1, x0 = round(s.values[-1], 1), round(s.values[-2], 1)
    f1 = mu + P1 * (x1 - mu) + P2 * (x0 - mu)
    num['sun_f2009'] = r(f1, 2)

    # INDPRO monthly growth: ARMA order table
    ip = td.read_fred('INDPRO').loc['1990':'2025']
    g = (100 * np.log(ip).diff()).dropna()
    g.index.freq = 'MS'
    rows = []
    best = {}
    for (pp, qq) in [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1), (3, 0)]:
        m = ARIMA(g.values, order=(pp, 0, qq), trend='c').fit()
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(m.resid, lags=[12], model_df=pp + qq)
        rows.append((f'ARMA({pp},{qq})', m.aic, m.bic, float(lb['lb_pvalue'].iloc[0])))
        best[(pp, qq)] = m
    txt = ('Model selection, US industrial production growth (100 dlog INDPRO)\n'
           'monthly, 1990-02 to 2025-12, n = %d\n' % len(g)
           + f'{"model":<10}{"AIC":>11}{"BIC":>11}{"LB(12) p":>11}\n' + '-' * 43 + '\n'
           + '\n'.join(f'{a:<10}{b:>11.2f}{cc:>11.2f}{d:>11.3f}' for a, b, cc, d in rows))
    ec.write_out(txt, out_dir, 'p_indpro_orders')
    num['indpro_n'] = len(g)
    num['indpro_table'] = [(a, r(b, 2), r(cc, 2), r(d, 3)) for a, b, cc, d in rows]
    num['indpro_aic_best'] = min(rows, key=lambda t: t[1])[0]
    num['indpro_bic_best'] = min(rows, key=lambda t: t[2])[0]
    m1 = best[(1, 0)]
    num['indpro_ar1_mu'] = r(m1.params[0], 4)
    num['indpro_ar1_phi'] = r(m1.params[1], 4)
    num['indpro_last'] = r(g.iloc[-1], 3)
    num['indpro_last_date'] = str(g.index[-1].date())
    mu, ph, last = round(m1.params[0], 3), round(m1.params[1], 3), round(g.iloc[-1], 3)
    num['indpro_f1'] = r(mu + ph * (last - mu), 3)
    num['indpro_f2'] = r(mu + ph ** 2 * (last - mu), 3)
    num['indpro_sigma2'] = r(m1.params[2], 4)
    num['indpro_se2'] = r(np.sqrt(m1.params[2] * (1 + ph ** 2)), 3)


# ============================================================================================ chapter 3
def adf_block(x, reg, title, maxlag=None):
    from statsmodels.tsa.stattools import adfuller
    st_, p, lag, nobs, cv, _ = adfuller(x, regression=reg, autolag='AIC', maxlag=maxlag)
    lab = {'c': 'constant', 'ct': 'constant and trend', 'n': 'none'}[reg]
    t = (f'{title}\nAugmented Dickey-Fuller test, deterministic terms: {lab}\n'
         f'  ADF statistic  {st_:>9.4f}    p-value {p:>7.4f}\n'
         f'  lags (AIC) {lag:>3d}   observations {nobs:>5d}\n'
         f'  critical values: 1% {cv["1%"]:.3f}   5% {cv["5%"]:.3f}   10% {cv["10%"]:.3f}')
    return t, st_, p, lag, nobs, cv


def ch3(num, fig_dir, out_dir):
    from statsmodels.tsa.stattools import kpss
    gold = td.load_close('gold', start='2010-01-01')
    lg = np.log(gold.resample('MS').last().loc[:'2026-08'])
    t1, s1, p1, l1, n1, cv1 = adf_block(lg.values, 'ct', '(1) log gold price, monthly, 2010-01 to 2026-08')
    t2, s2, p2, l2, n2, cv2 = adf_block(np.diff(lg.values), 'c', '(2) first difference of the log price')
    ec.write_out(t1 + '\n\n' + t2, out_dir, 'p_gold_adf')
    num.update(gold_adf_level=r(s1, 3), gold_adf_level_p=r(p1, 3), gold_adf_level_cv5=r(cv1['5%'], 3),
               gold_adf_diff=r(s2, 3), gold_adf_diff_p=r(p2, 4), gold_adf_diff_cv5=r(cv2['5%'], 3),
               gold_n=len(lg))

    # ARIMA(0,1,1) for the log gold price, forecast by hand
    from statsmodels.tsa.arima.model import ARIMA
    m = ARIMA(100 * lg.values, order=(0, 1, 1), trend='t').fit()
    drift, th, s2_ = m.params
    e_last = m.resid[-1]
    num.update(gold_ima_drift=r(drift, 4), gold_ima_theta=r(th, 4), gold_ima_sigma2=r(s2_, 3),
               gold_ima_lastlog=r(100 * lg.values[-1], 3), gold_ima_lastres=r(e_last, 3),
               gold_last_date=str(lg.index[-1].date()), gold_last_price=r(gold.resample('MS').last().loc[:'2026-08'].iloc[-1], 2))
    D, T, E, X = round(drift, 3), round(th, 3), round(e_last, 3), round(100 * lg.values[-1], 3)
    f1 = X + D + T * E
    f2 = f1 + D
    num['gold_ima_f1'] = r(f1, 3)
    num['gold_ima_f2'] = r(f2, 3)
    S2 = round(s2_, 3)
    num['gold_ima_se1'] = r(np.sqrt(S2), 3)
    num['gold_ima_se2'] = r(np.sqrt(S2 * (1 + (1 + T) ** 2)), 3)
    num['gold_ima_f1_price'] = r(np.exp(f1 / 100), 1)
    txt = (f'ARIMA(0,1,1) with drift, y = 100 log(gold), monthly, 2010-01 to 2026-08\n'
           + ec.coef_table([('drift', drift, m.bse[0]), ('ma.L1', th, m.bse[1]), ('sigma2', s2_, m.bse[2])])
           + f'\nLast observation (2026-08):  y = {100 * lg.values[-1]:.3f}   residual = {e_last:.3f}')
    ec.write_out(txt, out_dir, 'p_gold_ima')

    # CPI: ADF and KPSS on log level, inflation (dlog) and its change
    cpi = td.read_fred('CPIAUCSL').loc['1990':'2025']
    lc = 100 * np.log(cpi)
    inf = 1200 * np.log(cpi).diff().dropna()     # annualised monthly inflation, %
    rows = []
    for name, x, reg in (('log CPI', lc, 'ct'), ('inflation', inf, 'c'), ('d(inflation)', inf.diff().dropna(), 'c')):
        a = adf_block(x.values, reg, '')
        k = kpss(x.values, regression=reg, nlags='auto')
        rows.append((name, reg, a[1], a[2], k[0], k[1]))
    txt = ('US CPI (CPIAUCSL), monthly, 1990-2025; inflation = 1200 dlog(CPI), % per year\n'
           f'{"series":<14}{"det.":>6}{"ADF":>9}{"ADF p":>8}{"KPSS":>9}{"KPSS p":>8}\n' + '-' * 54 + '\n'
           + '\n'.join(f'{a:<14}{b:>6}{c:>9.3f}{d:>8.3f}{e:>9.3f}{f:>8.3f}' for a, b, c, d, e, f in rows)
           + '\nKPSS 5% critical values: 0.463 (constant), 0.146 (constant and trend)\n'
             'KPSS p-values are truncated to the interval [0.01, 0.10]')
    ec.write_out(txt, out_dir, 'p_cpi_tests')
    num['cpi_tests'] = [(a, b, r(c, 3), r(d, 3), r(e, 3), r(f, 3)) for a, b, c, d, e, f in rows]


# ============================================================================================ chapter 4
def ch4(num, fig_dir, out_dir):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox
    co2 = td.load_statsmodels('co2').resample('MS').mean().interpolate()
    co2 = co2.loc['1960':'2000']
    co2.index.freq = 'MS'
    m = SARIMAX(co2, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    lb = acorr_ljungbox(m.resid[13:], lags=[24], model_df=2)
    txt = ('SARIMAX(0,1,1)x(0,1,1,12), CO2 at Mauna Loa (ppm), monthly, 1960-01 to 2000-12\n'
           f'No. observations {m.nobs}   Log likelihood {m.llf:.2f}   AIC {m.aic:.2f}\n'
           + ec.coef_table([('ma.L1', m.params['ma.L1'], m.bse['ma.L1']),
                            ('ma.S.L12', m.params['ma.S.L12'], m.bse['ma.S.L12']),
                            ('sigma2', m.params['sigma2'], m.bse['sigma2'])])
           + f'\nLjung-Box Q(24) of the residuals: {lb["lb_stat"].iloc[0]:.2f}   p-value {lb["lb_pvalue"].iloc[0]:.3f}')
    ec.write_out(txt, out_dir, 'p_co2_airline')
    num.update(co2_theta=r(m.params['ma.L1'], 4), co2_Theta=r(m.params['ma.S.L12'], 4),
               co2_sigma2=r(m.params['sigma2'], 4), co2_q24=r(lb['lb_stat'].iloc[0], 2),
               co2_q24_p=r(lb['lb_pvalue'].iloc[0], 3), co2_n=int(m.nobs))
    th, Th = round(m.params['ma.L1'], 3), round(m.params['ma.S.L12'], 3)
    num['co2_theta_prod'] = r(th * Th, 4)
    num['co2_rho1'] = r(th / (1 + th ** 2), 3)
    num['co2_rho12'] = r(Th / (1 + Th ** 2), 3)

    # tourism nights: seasonal naive against SARIMA on 2025
    tr = (td.read_eurostat(*TOUR) / 1e6).loc['2012':'2025']
    tr.index.freq = 'MS'
    train, test = tr.loc[:'2024-12'], tr.loc['2025-01':'2025-12']
    snaive = train.loc['2024-01':'2024-12'].values
    lt = np.log(train.loc['2015':])
    lt.index.freq = 'MS'
    # 2020-2021 pandemic months distort the model: intervention dummy for March 2020 - May 2021
    ex = pd.Series(0.0, index=lt.index)
    ex.loc['2020-03':'2021-05'] = 1.0
    ms = SARIMAX(lt, exog=ex, order=(1, 0, 0), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    fc = np.exp(ms.forecast(12, exog=np.zeros(12)).values)
    scale = float(np.mean(np.abs(train.loc['2022':].values[12:] - train.loc['2022':].values[:-12])))
    e1, e2 = test.values - snaive, test.values - fc
    num['tour_test'] = [r(v, 3) for v in test.values]
    num['tour_snaive'] = [r(v, 3) for v in snaive]
    num['tour_sarima'] = [r(v, 3) for v in fc]
    num['tour_mae_snaive'] = r(np.mean(np.abs(e1)), 4)
    num['tour_mae_sarima'] = r(np.mean(np.abs(e2)), 4)
    num['tour_scale'] = r(scale, 4)
    num['tour_mase_snaive'] = r(np.mean(np.abs(e1)) / scale, 3)
    num['tour_mase_sarima'] = r(np.mean(np.abs(e2)) / scale, 3)
    d = np.abs(e1) - np.abs(e2)
    num['tour_dm_dbar'] = r(d.mean(), 4)
    num['tour_dm'] = r(d.mean() / (d.std(ddof=1) / np.sqrt(12)), 3)
    num['tour_jul_aug'] = [r(v, 3) for v in test.loc['2025-07':'2025-08'].values]
    num['tour_2024'] = [r(v, 3) for v in train.loc['2024-01':'2024-12'].values]
    txt = ('Nights spent in tourist accommodation, Romania (millions), test year 2025\n'
           f'{"month":<8}{"actual":>9}{"s.naive":>9}{"SARIMA":>9}\n' + '-' * 35 + '\n'
           + '\n'.join(f'{str(i.date())[:7]:<8}{a:>9.3f}{b:>9.3f}{c:>9.3f}'
                       for i, a, b, c in zip(test.index, test.values, snaive, fc))
           + f'\n{"MAE":<8}{"":>9}{np.mean(np.abs(e1)):>9.3f}{np.mean(np.abs(e2)):>9.3f}'
           + f'\nIn-sample MAE of the seasonal naive method (2023-2024): {scale:.3f}'
           + '\nSARIMA = log-SARIMA(1,0,0)(0,1,1)12, 2015-2024, pandemic dummy 2020-03..2021-05')
    ec.write_out(txt, out_dir, 'p_tourism_eval')
    for lang in ('ro', 'en'):
        L = dict(ro=('înnoptări (milioane)', 'observat 2023--2025', 'naivă sezonieră', 'SARIMA'),
                 en=('nights (millions)', 'observed 2023--2025', 'seasonal naive', 'SARIMA'))[lang]
        fig, ax = plt.subplots(figsize=(7.0, 2.8))
        s = tr.loc['2023':]
        ax.plot(s.index, s.values, 'o-', color=st.MainBlue, ms=3, lw=1.2, label=L[1])
        ax.plot(test.index, snaive, 's--', color=st.IDAred, ms=3.5, lw=1.1, label=L[2])
        ax.plot(test.index, fc, '^--', color=st.Forest, ms=3.5, lw=1.1, label=L[3])
        ax.set_ylabel(L[0])
        ax.yaxis.set_major_formatter(ec.fmt(lang)(1))
        st.legend_outside_bottom(ax, ncol=3, y=-0.2)
        ec.save(fig, fig_dir, f'practice_tourism_{lang}')


# ============================================================================================ chapter 5
def ch5(num, fig_dir, out_dir):
    from arch import arch_model
    dax = td.log_returns('dax', start='2010-01-01')
    am = arch_model(dax, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
    res = am.fit(disp='off')
    pr, se = res.params, res.std_err
    txt = ('Constant mean - GARCH(1,1), Student-t innovations, DAX daily log returns (%)\n'
           f'2010-01-05 to 2026-09-18, n = {res.nobs}   Log likelihood {res.loglikelihood:.2f}\n'
           + ec.coef_table([('mu', pr['mu'], se['mu']), ('omega', pr['omega'], se['omega']),
                            ('alpha[1]', pr['alpha[1]'], se['alpha[1]']), ('beta[1]', pr['beta[1]'], se['beta[1]']),
                            ('nu', pr['nu'], se['nu'])], header=('coef', 'std err', 't', 'P>|t|')))
    ec.write_out(txt, out_dir, 'p_dax_garch')
    o, a, b = round(pr['omega'], 4), round(pr['alpha[1]'], 4), round(pr['beta[1]'], 4)
    num.update(dax_n=int(res.nobs), dax_mu=r(pr['mu'], 4), dax_omega=o, dax_alpha=a, dax_beta=b, dax_nu=r(pr['nu'], 3))
    num['dax_pers'] = r(a + b, 4)
    num['dax_lrvar'] = r(o / (1 - a - b), 4)
    num['dax_lrvol_d'] = r(np.sqrt(o / (1 - a - b)), 3)
    num['dax_lrvol_a'] = r(np.sqrt(252 * o / (1 - a - b)), 2)
    num['dax_half'] = r(np.log(0.5) / np.log(a + b), 1)

    # Ethereum: one GARCH(1,1) step and VaR 1% (Normal innovations), daily, 7 days a week
    eth = td.log_returns('eth', start='2018-01-01')
    re_ = arch_model(eth, mean='Constant', vol='GARCH', p=1, q=1, dist='normal').fit(disp='off')
    p = re_.params
    sig2_T = float(re_.conditional_volatility.iloc[-1] ** 2)
    eps_T = float(eth.iloc[-1] - p['mu'])
    num.update(eth_mu=r(p['mu'], 4), eth_omega=r(p['omega'], 4), eth_alpha=r(p['alpha[1]'], 4),
               eth_beta=r(p['beta[1]'], 4), eth_sig2T=r(sig2_T, 3), eth_rT=r(eth.iloc[-1], 3),
               eth_date=str(eth.index[-1].date()), eth_epsT=r(eps_T, 3))
    M, O, A, B, S2, E = (round(p['mu'], 3), round(p['omega'], 3), round(p['alpha[1]'], 3), round(p['beta[1]'], 3),
                         round(sig2_T, 3), round(eps_T, 3))
    num['eth_round'] = dict(mu=M, omega=O, alpha=A, beta=B, sig2T=S2, eps=E)
    s2n = O + A * E ** 2 + B * S2
    num['eth_sig2_next'] = r(s2n, 3)
    num['eth_sig_next'] = r(np.sqrt(s2n), 3)
    num['eth_var1'] = r(-(M - 2.326 * np.sqrt(s2n)), 2)
    num['eth_var1_eur'] = r(10000 * (1 - np.exp((M - 2.326 * np.sqrt(s2n)) / 100)), 0)

    # DAX GJR asymmetry
    gj = arch_model(dax, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='t').fit(disp='off')
    g = gj.params
    txt = ('Constant mean - GJR-GARCH(1,1,1), Student-t, DAX daily log returns (%), 2010-2026\n'
           + ec.coef_table([('omega', g['omega'], gj.std_err['omega']), ('alpha[1]', g['alpha[1]'], gj.std_err['alpha[1]']),
                            ('gamma[1]', g['gamma[1]'], gj.std_err['gamma[1]']),
                            ('beta[1]', g['beta[1]'], gj.std_err['beta[1]'])], header=('coef', 'std err', 't', 'P>|t|'))
           + f'\nLog likelihood {gj.loglikelihood:.2f}   (GARCH(1,1)-t: {res.loglikelihood:.2f})')
    ec.write_out(txt, out_dir, 'p_dax_gjr')
    O2, A2, G2, B2 = (round(g['omega'], 4), round(g['alpha[1]'], 4), round(g['gamma[1]'], 4), round(g['beta[1]'], 4))
    num.update(gjr_omega=O2, gjr_alpha=A2, gjr_gamma=G2, gjr_beta=B2, gjr_ll=r(gj.loglikelihood, 2),
               garch_ll=r(res.loglikelihood, 2))
    num['gjr_LR'] = r(2 * (gj.loglikelihood - res.loglikelihood), 2)
    num['gjr_pers'] = r(A2 + G2 / 2 + B2, 4)
    # news impact with sigma^2_{t-1} = 1: shocks of -2 and +2
    num['gjr_nic_minus2'] = r(O2 + (A2 + G2) * 4 + B2 * 1, 3)
    num['gjr_nic_plus2'] = r(O2 + A2 * 4 + B2 * 1, 3)


# ============================================================================================ chapter 6
def macro_growth():
    d = td.load_statsmodels('macrodata')
    x = pd.DataFrame({'gdp': 400 * np.log(d['realgdp']).diff(),
                      'infl': d['infl'],
                      'rate': d['tbilrate']}).dropna()
    x.index.freq = 'QS-OCT'
    return x


def ch6(num, fig_dir, out_dir):
    from statsmodels.tsa.api import VAR
    x = macro_growth()
    v = VAR(x)
    sel = v.select_order(8)
    tab = sel.ics
    lines = [f'VAR lag order selection, y = (gdp growth, inflation, T-bill rate), US 1959Q2-2009Q3',
             f'{"p":>3}{"AIC":>10}{"BIC":>10}{"HQIC":>10}', '-' * 33]
    for p_ in range(9):
        lines.append(f'{p_:>3}{tab["aic"][p_]:>10.3f}{tab["bic"][p_]:>10.3f}{tab["hqic"][p_]:>10.3f}')
    lines.append(f'chosen: AIC {sel.aic}, BIC {sel.bic}, HQIC {sel.hqic}')
    num['var_sel'] = dict(aic=int(sel.aic), bic=int(sel.bic), hqic=int(sel.hqic))
    num['var_sel_vals'] = {k: [r(tab[k][p_], 3) for p_ in range(9)] for k in ('aic', 'bic', 'hqic')}
    res = v.fit(2)
    gc1 = res.test_causality('gdp', ['rate'], kind='f')
    gc2 = res.test_causality('rate', ['gdp'], kind='f')
    gc3 = res.test_causality('rate', ['infl'], kind='f')
    lines += ['', 'Granger causality F-tests in the VAR(2) (H0: the cause does not Granger-cause)',
              f'{"cause -> effect":<18}{"F":>9}{"df":>12}{"p-value":>10}', '-' * 49]
    for name, t in (('rate -> gdp', gc1), ('gdp -> rate', gc2), ('infl -> rate', gc3)):
        lines.append(f'{name:<18}{t.test_statistic:>9.3f}{str(tuple(int(z) for z in t.df)):>12}{t.pvalue:>10.4f}')
    ec.write_out('\n'.join(lines), out_dir, 'p_macro_var')
    num['var_gc'] = {name: (r(t.test_statistic, 3), [int(z) for z in t.df], r(t.pvalue, 4))
                     for name, t in (('rate->gdp', gc1), ('gdp->rate', gc2), ('infl->rate', gc3))}
    num['var_nobs'] = int(res.nobs)
    num['var_k'] = 3

    # VAR(1) of (gdp growth, rate change) : stability and forecast
    y = pd.DataFrame({'gdp': x['gdp'], 'drate': x['rate'].diff()}).dropna()
    r1 = VAR(y).fit(1)
    A = r1.coefs[0]
    c = r1.intercept
    Ar = np.round(A, 3)
    cr = np.round(c, 3)
    ev = np.linalg.eigvals(Ar)
    last = np.round(y.values[-1], 2)
    f1 = cr + Ar @ last
    num.update(var1_A=Ar.tolist(), var1_c=cr.tolist(), var1_eig=[r(abs(e), 4) for e in ev],
               var1_eig_raw=[str(np.round(e, 4)) for e in ev], var1_last=last.tolist(), var1_f1=[r(v, 3) for v in f1],
               var1_lastdate='2009Q3')
    tr, det = Ar[0, 0] + Ar[1, 1], Ar[0, 0] * Ar[1, 1] - Ar[0, 1] * Ar[1, 0]
    num['var1_trace'] = r(tr, 4)
    num['var1_det'] = r(det, 4)
    mean = np.linalg.solve(np.eye(2) - Ar, cr)
    num['var1_mean'] = [r(v, 3) for v in mean]


# ============================================================================================ chapter 7
def ch7(num, fig_dir, out_dir):
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
    d = td.read_fred(['GS10', 'TB3MS']).loc['1990':'2025'].dropna()
    ols = sm.OLS(d['GS10'], sm.add_constant(d['TB3MS'])).fit()
    u = ols.resid
    a = adfuller(u.values, regression='n', autolag='AIC')
    txt = ('Step 1: OLS of GS10 on TB3MS, US monthly 1990-01 to 2025-12 (% per year)\n'
           + ec.coef_table([('const', ols.params['const'], ols.bse['const']),
                            ('TB3MS', ols.params['TB3MS'], ols.bse['TB3MS'])], header=('coef', 'std err', 't', 'P>|t|'))
           + f'\nR-squared {ols.rsquared:.3f}   Durbin-Watson {sm.stats.durbin_watson(u):.3f}   n = {len(u)}\n\n'
           + 'Step 2: ADF test on the residuals u_t (no constant, lags by AIC)\n'
           + f'  ADF statistic {a[0]:.4f}   lags {a[2]}   (Dickey-Fuller table p-value {a[1]:.4f})')
    ec.write_out(txt, out_dir, 'p_rates_eg')
    num.update(eg_c=r(ols.params['const'], 3), eg_b=r(ols.params['TB3MS'], 3), eg_r2=r(ols.rsquared, 3),
               eg_dw=r(sm.stats.durbin_watson(u), 3), eg_adf=r(a[0], 3), eg_n=len(u), eg_lags=int(a[2]))

    # Johansen + VECM
    d = td.read_fred(['GS10', 'TB3MS']).loc['1960':'2025'].dropna()
    j = coint_johansen(d[['GS10', 'TB3MS']].values, det_order=0, k_ar_diff=2)
    lines = ['Johansen test, (GS10, TB3MS), US monthly 1960-01 to 2025-12, constant,',
             '2 lagged differences',
             f'{"H0: rank <=":<13}{"trace":>9}{"5% c.v.":>9}{"max-eig":>9}{"5% c.v.":>9}', '-' * 49]
    for i in range(2):
        lines.append(f'{i:<13d}{j.lr1[i]:>9.2f}{j.cvt[i, 1]:>9.2f}{j.lr2[i]:>9.2f}{j.cvm[i, 1]:>9.2f}')
    vm = VECM(d[['GS10', 'TB3MS']], k_ar_diff=2, coint_rank=1, deterministic='ci').fit()
    beta = vm.beta[:, 0]
    alpha = vm.alpha[:, 0]
    ase = vm.stderr_alpha[:, 0]
    lines += ['', 'VECM, rank 1, constant in the cointegrating relation',
              f'beta (normalised): GS10 {beta[0]:.3f}   TB3MS {beta[1]:.3f}   const {vm.const_coint[0, 0]:.3f}',
              f'{"alpha":<10}{"coef":>10}{"std err":>10}{"z":>9}',
              f'{"D(GS10)":<10}{alpha[0]:>10.4f}{ase[0]:>10.4f}{alpha[0] / ase[0]:>9.2f}',
              f'{"D(TB3MS)":<10}{alpha[1]:>10.4f}{ase[1]:>10.4f}{alpha[1] / ase[1]:>9.2f}']
    ec.write_out('\n'.join(lines), out_dir, 'p_rates_vecm')
    num.update(jo_trace=[r(v, 2) for v in j.lr1], jo_cvt=[r(v, 2) for v in j.cvt[:, 1]],
               jo_max=[r(v, 2) for v in j.lr2], jo_cvm=[r(v, 2) for v in j.cvm[:, 1]],
               vecm_beta=[r(v, 3) for v in beta], vecm_const=r(vm.const_coint[0, 0], 3),
               vecm_alpha=[r(v, 4) for v in alpha], vecm_alpha_z=[r(alpha[i] / ase[i], 2) for i in range(2)])
    a1, a2 = round(alpha[0], 4), round(alpha[1], 4)
    b2 = round(beta[1], 3)
    lam = 1 + a1 + a2 * b2   # speed of the cointegrating error: z_t = GS10 + b2*TB3MS + c; dz = (a1 + b2 a2) z
    num['vecm_lambda'] = r(lam, 4)
    num['vecm_half'] = r(np.log(0.5) / np.log(lam), 1)
    num['spread_last'] = [r(v, 2) for v in d.iloc[-1].values]
    z = round(float(d['GS10'].iloc[-1] + b2 * d['TB3MS'].iloc[-1] + round(vm.const_coint[0, 0], 3)), 3)
    num['ect_last'] = z
    num['ect_adj'] = [r(a1 * z, 4), r(a2 * z, 4)]
    num['spread_date'] = str(d.index[-1].date())


# ============================================================================================ chapter 8
def gph(x, power=0.5):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    m = int(n ** power)
    f = np.fft.fft(x)
    j = np.arange(1, m + 1)
    lam = 2 * np.pi * j / n
    I = np.abs(f[j]) ** 2 / (2 * np.pi * n)
    X = np.log(4 * np.sin(lam / 2) ** 2)
    import statsmodels.api as sm
    res = sm.OLS(np.log(I), sm.add_constant(X)).fit()
    return -res.params[1], res.bse[1], m, n


def ch8(num, fig_dir, out_dir):
    from statsmodels.tsa.stattools import acf
    vix = np.log(td.load_close('vix', start='2004-01-01'))
    dg, se, m, n = gph(vix.values)
    dg2, se2, m2, _ = gph(np.diff(vix.values))
    lines = ['GPH log-periodogram regression, bandwidth m = n^0.5',
             f'{"series":<22}{"n":>6}{"m":>5}{"d_hat":>9}{"s.e.":>8}{"asy s.e.":>10}', '-' * 60]
    for name, nn, mm, dd, ss in (('log VIX', n, m, dg, se), ('d log VIX', n - 1, m2, dg2, se2)):
        lines.append(f'{name:<22}{nn:>6}{mm:>5}{dd:>9.3f}{ss:>8.3f}{np.pi / np.sqrt(24 * mm):>10.3f}')
    lines.append('asy s.e. = pi / sqrt(24 m); daily data 2004-01-02 to 2026-09-18')
    ec.write_out('\n'.join(lines), out_dir, 'p_vix_gph')
    num.update(vix_n=n, vix_m=m, vix_d=r(dg, 3), vix_se=r(se, 3), vix_asy=r(np.pi / np.sqrt(24 * m), 3),
               vix_d_diff=r(dg2, 3), vix_se_diff=r(se2, 3), vix_m2=m2)
    D = round(dg, 2)
    pis = [1.0]
    for k in range(1, 4):
        pis.append(pis[-1] * (k - 1 - D) / k)
    num['vix_pi_1_3'] = [r(v, 4) for v in pis[1:]]

    # gold absolute returns: ACF at lags 1, 10, 50, 100 and the AR(1) implied
    rg = td.log_returns('gold', start='2010-01-01')
    ab = np.abs(rg.values)
    a = acf(ab, nlags=100, fft=True)
    num['gold_abs_n'] = len(ab)
    num['gold_abs_acf'] = {k: r(a[k], 3) for k in (1, 5, 10, 20, 50, 100)}
    r1 = round(a[1], 3)
    num['gold_ar1_implied'] = {k: float(f'{r1 ** k:.2e}') for k in (10, 50, 100)}
    # power-law slope through lags 10 and 100: rho_k ~ C k^{2d-1}
    a10, a100 = round(a[10], 3), round(a[100], 3)
    slope = np.log(a100 / a10) / np.log(10)
    num['gold_slope'] = r(slope, 3)
    num['gold_d_from_slope'] = r((slope + 1) / 2, 3)
    for lang in ('ro', 'en'):
        L = dict(ro=('ACF a lui $|r_t|$, aur', 'AR(1) cu același $\\rho(1)$', 'decalaj $k$', 'autocorelație'),
                 en=('ACF of $|r_t|$, gold', 'AR(1) with the same $\\rho(1)$', 'lag $k$', 'autocorrelation'))[lang]
        fig, ax = plt.subplots(figsize=(6.8, 2.8))
        k = np.arange(1, 101)
        ax.bar(k, a[1:101], color=st.MainBlue, width=0.8, label=L[0])
        ax.plot(k, r1 ** k, color=st.IDAred, lw=1.6, label=L[1])
        ax.axhline(1.96 / np.sqrt(len(ab)), color=st.Forest, ls='--', lw=1.0, label=r'$1{,}96/\sqrt{n}$' if lang == 'ro' else r'$1.96/\sqrt{n}$')
        ax.set_xlabel(L[2])
        ax.set_ylabel(L[3])
        ax.yaxis.set_major_formatter(ec.fmt(lang)(2))
        st.legend_outside_bottom(ax, ncol=3, y=-0.3)
        ec.save(fig, fig_dir, f'practice_gold_absacf_{lang}')


# ============================================================================================ chapter 9
def ch9(num, fig_dir, out_dir):
    # Bitcoin: sign of tomorrow's return, walk-forward accuracy vs the right baseline (seeded, simple models)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    b = td.log_returns('btc', start='2018-01-01')
    X = pd.concat({f'lag{k}': b.shift(k) for k in range(1, 6)}, axis=1)
    X['vol20'] = b.rolling(20).std().shift(1)
    y = (b > 0).astype(int)
    df = pd.concat([X, y.rename('y')], axis=1).dropna()
    test_start = '2024-01-01'
    tr, te = df.loc[:'2023-12-31'], df.loc[test_start:]
    feats = [c for c in df.columns if c != 'y']
    lr = LogisticRegression(max_iter=1000).fit(tr[feats], tr['y'])
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50, random_state=1).fit(tr[feats], tr['y'])
    acc_lr = float((lr.predict(te[feats]) == te['y']).mean())
    acc_rf = float((rf.predict(te[feats]) == te['y']).mean())
    share_up_train = float(tr['y'].mean())
    share_up_test = float(te['y'].mean())
    acc_always_up = share_up_test
    num.update(btc_ntest=len(te), btc_acc_lr=r(acc_lr, 4), btc_acc_rf=r(acc_rf, 4),
               btc_up_train=r(share_up_train, 4), btc_up_test=r(share_up_test, 4), btc_ntrain=len(tr))
    n = len(te)
    num['btc_se_acc'] = r(np.sqrt(0.25 / n), 4)
    num['btc_z_rf'] = r((acc_rf - acc_always_up) / np.sqrt(acc_always_up * (1 - acc_always_up) / n), 2)
    num['btc_z_lr'] = r((acc_lr - acc_always_up) / np.sqrt(acc_always_up * (1 - acc_always_up) / n), 2)
    txt = ('Direction of the next-day Bitcoin return (1 = up), features: 5 lags, 20-day vol\n'
           f'train 2018-01 to 2023-12 (n = {len(tr)}), test 2024-01-01 to 2026-09-18 (n = {n})\n'
           f'{"model":<34}{"test accuracy":>14}\n' + '-' * 48 + '\n'
           f'{"logistic regression":<34}{acc_lr:>14.4f}\n'
           f'{"random forest (300 trees)":<34}{acc_rf:>14.4f}\n'
           f'{"always \'up\' (majority class)":<34}{acc_always_up:>14.4f}\n'
           f'share of up days in the training set: {share_up_train:.4f}')
    ec.write_out(txt, out_dir, 'p_btc_sign')

    # gold: quantile forecasts of the daily return; pinball loss and split conformal (small numbers)
    g = td.log_returns('gold', start='2025-01-01')
    # calibration residuals: |r_t - 0| for the last 20 days before the last 5 (absolute returns as conformity scores)
    cal = np.round(np.abs(g.values[-25:-5]), 2)
    test5 = np.round(g.values[-5:], 2)
    num['gold_cal_scores'] = sorted([float(v) for v in cal])
    num['gold_test5'] = [float(v) for v in test5]
    num['gold_test5_dates'] = [str(d.date()) for d in g.index[-5:]]
    nc = len(cal)
    k = int(np.ceil((nc + 1) * 0.9))
    qhat = sorted(cal)[k - 1]
    num['gold_conf_k'] = k
    num['gold_conf_q'] = float(qhat)
    num['gold_conf_cover'] = int(np.sum(np.abs(test5) <= qhat))
    # pinball losses of two 0.05-quantile forecasts on the 5 test days
    tau = 0.05
    qa, qb = -1.2, -2.5
    def pin(yv, q):
        return np.mean([tau * (v - q) if v >= q else (1 - tau) * (q - v) for v in yv])
    num['gold_pin_a'] = r(pin(test5, qa), 4)
    num['gold_pin_b'] = r(pin(test5, qb), 4)


# ============================================================================================ chapter 10
def ch10(num, fig_dir, out_dir):
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    u = td.read_fred('UNRATE').loc['2022':'2025'].asfreq('MS')   # October 2025 is missing (not published)
    m = UnobservedComponents(u, level='llevel').fit(disp=False)
    se_, sl_ = m.params['sigma2.irregular'], m.params['sigma2.level']
    txt = ('UnobservedComponents, local level, US unemployment rate (UNRATE, %),\n'
           f'monthly 2022-01 to 2025-12 (2025-10 missing), n = {int(m.nobs)}   Log likelihood {m.llf:.2f}\n'
           + ec.coef_table([('sigma2.irregular', se_, m.bse['sigma2.irregular']),
                            ('sigma2.level', sl_, m.bse['sigma2.level'])])
           + f'\nfiltered level 2025-12: {m.filtered_state[0, -1]:.4f}'
           f'   variance P(t|t): {m.filtered_state_cov[0, 0, -1]:.4f}')
    ec.write_out(txt, out_dir, 'p_unrate_ll')
    num.update(ll_se=r(se_, 4), ll_sl=r(sl_, 4), ll_a=r(m.filtered_state[0, -1], 4),
               ll_P=r(m.filtered_state_cov[0, 0, -1], 4), ll_q=r(sl_ / se_, 3) if se_ > 0 else None)
    SE, SL, A, P = round(se_, 4), round(sl_, 4), round(m.filtered_state[0, -1], 4), round(m.filtered_state_cov[0, 0, -1], 4)
    u1 = round(float(td.read_fred('UNRATE').loc['2026-01-01']), 1)
    Pp = P + SL
    K = Pp / (Pp + SE)
    a1 = A + K * (u1 - A)
    num.update(ll_y_jan26=u1, ll_Ppred=r(Pp, 4), ll_K=r(K, 4), ll_a_jan26=r(a1, 3), ll_P_upd=r((1 - K) * Pp, 4))

    # Nasdaq 100: two-regime Markov switching variance
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    rn = td.log_returns('ndx', start='2015-01-01')
    w = rn.resample('W-FRI').sum()
    np.random.seed(2027)  # search_reps draws random starting values
    ms = MarkovRegression(w.values, k_regimes=2, switching_variance=True).fit(disp=False, search_reps=20)
    pr = dict(zip(ms.model.param_names, ms.params))
    # identify the calm regime as the one with the smaller variance
    s0, s1 = pr['sigma2[0]'], pr['sigma2[1]']
    calm, turb = (0, 1) if s0 < s1 else (1, 0)
    P = ms.regime_transition[:, :, 0]   # P[i, j] = P(S_t = i | S_{t-1} = j)
    p_cc, p_tt = P[calm, calm], P[turb, turb]
    lines = ['MarkovRegression, 2 regimes, switching mean and variance,',
             f'Nasdaq 100 weekly log returns (%), {str(w.index[0].date())} to {str(w.index[-1].date())}, n = {len(w)}',
             f'{"regime":<10}{"mean":>10}{"variance":>11}{"stay prob.":>12}', '-' * 43]
    for name, k_, pkk in (('calm', calm, p_cc), ('turbulent', turb, p_tt)):
        lines.append(f'{name:<10}{pr[f"const[{k_}]"]:>10.3f}{pr[f"sigma2[{k_}]"]:>11.3f}{pkk:>12.4f}')
    sm_ = ms.smoothed_marginal_probabilities[:, turb]
    lines.append(f'share of weeks with smoothed P(turbulent) > 0.5: {np.mean(sm_ > 0.5):.3f}')
    ec.write_out('\n'.join(lines), out_dir, 'p_ndx_ms')
    pc, pt = round(p_cc, 3), round(p_tt, 3)
    num.update(ms_n=len(w), ms_mean=[r(pr[f'const[{calm}]'], 3), r(pr[f'const[{turb}]'], 3)],
               ms_var=[r(pr[f'sigma2[{calm}]'], 3), r(pr[f'sigma2[{turb}]'], 3)],
               ms_p=[r(p_cc, 4), r(p_tt, 4)], ms_share_turb=r(np.mean(sm_ > 0.5), 3))
    num['ms_dur'] = [r(1 / (1 - pc), 1), r(1 / (1 - pt), 1)]
    num['ms_ergodic_turb'] = r((1 - pc) / (2 - pc - pt), 3)
    num['ms_vol_ann'] = [r(np.sqrt(52 * round(pr[f'sigma2[{calm}]'], 3)), 1),
                         r(np.sqrt(52 * round(pr[f'sigma2[{turb}]'], 3)), 1)]


def make(fig_dir, out_dir):
    num = {}
    for f in (ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8, ch9, ch10):
        print('  practice', f.__name__)
        f(num, fig_dir, out_dir)
    return num


if __name__ == '__main__':
    import json
    ec.style()
    d = os.path.dirname(os.path.abspath(__file__))
    n = make(os.path.join(d, 'figs'), os.path.join(d, 'out'))
    print(json.dumps(n, indent=1, default=float, ensure_ascii=False))
