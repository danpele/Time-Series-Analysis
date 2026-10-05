"""
build_quantlets.py -- Quantlet folders of Chapter 6 (TSA): VAR models and Granger causality
==========================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_06/generate_all_charts.py && python3 Quantlets/Ch_06/seminar6.py
      python3 Quantlets/Ch_06/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 6
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import generate_all_charts as g                 # noqa: E402
from tsa_quantlets import build_all             # noqa: E402

SUBMITTED = 'Monday, 5 October 2026'
DATA = ('Daily market data from EODHD (S&P 500, DAX and BET since 2000), data/market of the TSA repository; EUR/RON '
        'reference rate of the BNR; Romanian quarterly real GDP, monthly HICP, ROBOR 3M and unemployment from Eurostat; '
        'US GDP price index, unemployment rate and federal funds rate from FRED')
NODATA = 'Simulated data only (no market data)'
CONSTS = ['from statsmodels.tsa.api import VAR', 'from statsmodels.tsa.ar_model import AutoReg',
          f'SEED = {g.SEED!r}', f'GDP_SA = {g.GDP_SA!r}', f'HICP = {g.HICP!r}', f'ROBOR = {g.ROBOR!r}',
          f'UNEMP = {g.UNEMP!r}', f'RO_START = {g.RO_START!r}', f'RO_VARS = {g.RO_VARS!r}', f'RO_P = {g.RO_P!r}',
          f'MKT = {g.MKT!r}', f'MKT_START = {g.MKT_START!r}', f'MKT_P = {g.MKT_P!r}', f'DY_H, DY_W = {g.DY_H!r}, {g.DY_W!r}',
          f'SW_VARS = {g.SW_VARS!r}', f'SW_START, SW_END, SW_P = {g.SW_START!r}, {g.SW_END!r}, {g.SW_P!r}',
          f'LABEL = {g.LABEL!r}',
          "COLV = {'g': st.Forest, 'pi': st.IDAred, 'i': st.MainBlue, 'u': st.Purple, 'ds': st.Amber, 'R': st.MainBlue, "
          "'sp500': st.COL['sp500'], 'dax': st.COL['dax'], 'bet': st.COL['bet']}",
          'BAND = st.Teal', f'WX_A = np.array({g.WX_A.tolist()!r})', f'WX_C = np.array({g.WX_C.tolist()!r})',
          f'WX_S = np.array({g.WX_S.tolist()!r})', f'WX_Y = np.array({g.WX_Y.tolist()!r})',
          'HERE = "."']
CORE = [g.ro_monthly, g.ro_quarterly, g.ro_var_data, g.market_returns, g.us_sw, g.fit_var, g.save, g.years_axis, g.qlabel,
        g.ccf, g.companion, g.ic_table, g.granger_F, g.granger_table, g.girf, g.gfevd, g.spill_index, g.ar_forecast,
        g.dm_test, g.boot_irf, g.irf_panel, g.ro_fit]

QUANTLETS = [
    dict(name='TSA_ch6_multivariate_data',
         desc='From one series to several: Romanian real GDP growth (Eurostat, seasonally adjusted), 12-month HICP inflation, '
              'the 3-month interbank rate ROBOR 3M and the unemployment rate (Eurostat), EUR/RON (BNR reference rate); '
              'cross-correlation functions of daily S&P 500, DAX and BET returns and of quarterly ROBOR 3M and inflation.',
         keywords='multivariate time series, cross-correlation, lead-lag, Romania, GDP, inflation, ROBOR, unemployment, '
                  'EUR/RON, S&P 500, DAX, BET, non-synchronous trading',
         consts=CONSTS, funcs=CORE + [g.fig_ro_macro, g.fig_ccf],
         run='print(fig_ro_macro())\nprint(fig_ccf())', charts=['tsa_ch6_ro_macro', 'tsa_ch6_ccf']),
    dict(name='TSA_ch6_var_stability',
         desc='The VAR(1) worked example (eigenvalues, mean, forecasts, impulse responses, Lyapunov variance), a simulated '
              'path, and the eigenvalues of the companion matrices of three estimated VARs: Romania VAR(2), the Stock-Watson '
              'VAR(4) for the United States and a daily VAR(4) of S&P 500, DAX and BET returns.',
         keywords='VAR, vector autoregression, stability, eigenvalues, companion matrix, unit circle, moving-average '
                  'representation, simulation',
         consts=CONSTS, funcs=CORE + [g.worked_example, g.fig_var_sim, g.fig_roots],
         run='print(worked_example())\nprint(fig_var_sim())\nprint(fig_roots())', charts=['tsa_ch6_var_sim', 'tsa_ch6_roots']),
    dict(name='TSA_ch6_estimation_diagnostics',
         desc='A VAR for Romanian GDP growth, 12-month inflation and ROBOR 3M (quarterly, 2005-2026): lag selection by AIC, '
              'BIC and HQ on a common sample, OLS estimates, residuals, multivariate portmanteau and Jarque-Bera tests.',
         keywords='VAR, OLS, lag selection, AIC, BIC, Hannan-Quinn, portmanteau test, normality test, residuals, Romania',
         consts=CONSTS, funcs=CORE + [g.ic_romania, g.fig_ic, g.ro_estimates, g.fig_resid],
         run='print(fig_ic())\nprint(ro_estimates())\nprint(fig_resid())', charts=['tsa_ch6_ic', 'tsa_ch6_resid']),
    dict(name='TSA_ch6_granger',
         desc='Granger causality F tests in the Romanian VAR(2) (GDP growth, inflation, ROBOR 3M) and in a daily VAR(4) of '
              'S&P 500, DAX and BET returns; a worked F test; a Monte Carlo of spurious Granger causality caused by an '
              'omitted common driver (bivariate against trivariate VAR).',
         keywords='Granger causality, F test, Wald test, omitted variable, instantaneous causality, Monte Carlo, Romania, '
                  'S&P 500, DAX, BET',
         consts=CONSTS, funcs=CORE + [g.granger_romania, g.granger_markets, g.granger_by_hand, g.omitted_sim, g.fig_granger_sim],
         run='print(granger_by_hand())\nprint(granger_romania())\nprint(granger_markets())\nprint(fig_granger_sim())',
         charts=['tsa_ch6_granger_sim'], extra=['ch6_granger_romania.csv']),
    dict(name='TSA_ch6_irf_fevd',
         desc='Orthogonalised (Cholesky) impulse responses of the Romanian VAR(2) with residual-bootstrap bands; the effect '
              'of the ordering and the generalised impulse responses of Pesaran and Shin (1998); the forecast error '
              'variance decomposition.',
         keywords='impulse response function, Cholesky, ordering, generalised impulse response, bootstrap bands, FEVD, '
                  'price puzzle, Romania, ROBOR, inflation',
         consts=CONSTS, funcs=CORE + [g.fig_irf_ro, g.fig_irf_order, g.fig_fevd_ro],
         run='print(fig_irf_ro())\nprint(fig_irf_order())\nprint(fig_fevd_ro())',
         charts=['tsa_ch6_irf_ro', 'tsa_ch6_irf_order', 'tsa_ch6_fevd_ro']),
    dict(name='TSA_ch6_market_spillovers',
         desc='Daily S&P 500, DAX and BET returns: responses of DAX and BET to an S&P 500 shock (generalised and Cholesky in '
              'two orderings); the Diebold-Yilmaz (2012) spillover table and the total spillover index in rolling 200-day '
              'windows (VAR(4), generalised FEVD at 10 days).',
         keywords='spillover index, Diebold-Yilmaz, generalised FEVD, connectedness, impulse response, S&P 500, DAX, BET, '
                  'contagion, rolling window',
         consts=CONSTS, funcs=CORE + [g.fig_market_irf, g.spillover_table, g.rolling_spillover, g.fig_spillover],
         run='print(fig_market_irf())\nprint(fig_spillover())', charts=['tsa_ch6_market_irf', 'tsa_ch6_spillover'],
         extra=['ch6_spillover_table.csv']),
    dict(name='TSA_ch6_forecasting',
         desc='VAR forecasts with 95% intervals for Romanian GDP growth, inflation and ROBOR 3M; pseudo out-of-sample '
              'comparison (expanding window) of the VAR with univariate AR models and the random walk for Romania '
              '(2015-2026) and for the Stock-Watson VAR of the United States (1985-2000); Diebold-Mariano tests.',
         keywords='VAR forecast, forecast interval, out-of-sample evaluation, RMSE, random walk, AR benchmark, '
                  'Diebold-Mariano, Romania, United States',
         consts=CONSTS, funcs=CORE + [g.fig_forecast_ro, g.oos, g.fig_oos],
         run='print(fig_forecast_ro())\nprint(fig_oos())', charts=['tsa_ch6_forecast_ro', 'tsa_ch6_oos']),
    dict(name='TSA_ch6_stock_watson',
         desc='The three-variable VAR of Stock and Watson (2001): inflation (GDP price index), unemployment and the federal '
              'funds rate, quarterly 1960Q1-2000Q4, VAR(4), Cholesky order inflation, unemployment, fed funds; impulse '
              'responses with bootstrap bands, Granger tests, FEVD, and the same VAR on the extended sample.',
         keywords='Stock and Watson, structural VAR, monetary policy shock, federal funds rate, inflation, unemployment, '
                  'impulse response, FEVD, FRED',
         consts=CONSTS, funcs=CORE + [g.fig_us_data, g.fig_sw_irf],
         run='print(fig_us_data())\nprint(fig_sw_irf())', charts=['tsa_ch6_us_data', 'tsa_ch6_sw_irf'],
         data='FRED series GDPCTPI, UNRATE and FEDFUNDS, read online'),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar6 as s
    QUANTLETS.append(dict(
        name='TSA_ch6_seminar',
        desc='Seminar 6 of Time Series Analysis: a VAR(1) by hand (stability, mean, forecasts), orthogonalised impulse '
             'responses and a variance decomposition by hand, a Granger F test and lag selection from given numbers; '
             'S&P 500 and BET (daily and weekly) Granger tests; a Romanian VAR of GDP growth, inflation and ROBOR 3M, '
             'extended with the EUR/RON rate; the price puzzle.',
        keywords='seminar, VAR, Granger causality, impulse response, FEVD, lag selection, S&P 500, BET, Romania, ROBOR, '
                 'inflation, EUR/RON',
        consts=CONSTS, funcs=CORE + [s.var1_by_hand, s.irf_by_hand, s.granger_rss, s.ic_by_hand, s.pair_returns,
                                     s.market_pair, s.b3_romania],
        run="print(var1_by_hand([[0.6, 0.2], [0.2, 0.6]], [0.4, 0.8], [3.0, 2.0]))\n"
            "print(irf_by_hand([[0.6, 0.2], [0.2, 0.6]], [[1.0, 0.4], [0.4, 1.0]], 3))\n"
            "print(granger_rss(52.8, 45.2, 100, 2, 5))\nprint(market_pair('D', 'ch6_sem_b1'))\nprint(b3_romania())",
        charts=['ch6_sem_b1', 'ch6_sem_b3']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 6, 'VAR models and Granger causality', HERE, data=DATA, submitted=SUBMITTED)
