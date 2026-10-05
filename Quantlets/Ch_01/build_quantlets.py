"""
build_quantlets.py -- Quantlet folders of Chapter 1 (TSA): stochastic processes and stationarity
===============================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_01/generate_all_charts.py && python3 Quantlets/Ch_01/seminar1.py
      python3 Quantlets/Ch_01/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 1
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
DATA = ('Daily market data from EODHD (BET and S&P 500 since 2000), data/market of the TSA repository; EUR/RON reference '
        'rate of the BNR; Romanian quarterly GDP and monthly HICP from Eurostat; statsmodels data sets (Nile, sunspots)')
NODATA = 'Simulated data only (no market data)'
CONSTS = ['from statsmodels.tsa.stattools import acf, pacf', 'from statsmodels.stats.diagnostic import acorr_ljungbox',
          f'SEED = {g.SEED!r}', f'GDP_REAL = {g.GDP_REAL!r}', f'GDP_NOMINAL = {g.GDP_NOMINAL!r}', f'HICP = {g.HICP!r}',
          f'START = {g.START!r}', 'BAND_COL = st.IDAred', f'NAME = {g.NAME!r}']
CORE = [g.ro_gdp, g.ro_hicp, g.sample_acf, g.sample_pacf, g.ljung_box, g.acf_bars, g.simulate_arma, g.theoretical_acf,
        g.save]

QUANTLETS = [
    dict(name='TSA_ch1_real_series',
         desc='Four series of the course: Romanian quarterly real GDP (Eurostat, chain-linked volumes, not seasonally '
              'adjusted), Romanian 12-month HICP inflation (Eurostat), the EUR/RON reference rate of the National Bank of '
              'Romania and the daily log returns of the BET index since 2000; summary numbers of each series.',
         keywords='time series, GDP, inflation, HICP, exchange rate, EUR/RON, BET, log returns, Eurostat, BNR, Romania',
         consts=CONSTS, funcs=CORE + [g.fig_four_series], run='print(fig_four_series())',
         charts=['tsa_ch1_four_series']),
    dict(name='TSA_ch1_processes',
         desc='Stochastic processes by simulation: an ensemble of paths of a stationary AR(1) and of a random walk; four '
              'series that violate stationarity in different ways (trend, changing variance, level shift); a weakly but '
              'not strictly stationary process; Gaussian, Laplace and GARCH(1,1) white noise with the ACF of their squares '
              'and Ljung-Box tests; random walks and their variance; the BET log price hidden among five random walks.',
         keywords='stochastic process, stationarity, strict stationarity, weak stationarity, white noise, random walk, '
                  'GARCH, ensemble, simulation, BET',
         consts=CONSTS, funcs=CORE + [g.fig_ensemble, g.fig_nonstationary, g.fig_counterexample, g.simulate_garch,
                                      g.fig_white_noise, g.fig_random_walk, g.fig_spot_real],
         run='print(fig_ensemble())\nprint(fig_nonstationary())\nprint(fig_counterexample())\nprint(fig_white_noise())\n'
             'print(fig_random_walk())\nprint(fig_spot_real())',
         charts=['tsa_ch1_ensemble', 'tsa_ch1_nonstationary', 'tsa_ch1_counterexample', 'tsa_ch1_white_noise',
                 'tsa_ch1_random_walk', 'tsa_ch1_spot_real']),
    dict(name='TSA_ch1_wold_ergodicity',
         desc='The Wold weights psi_j of AR(1) processes (phi = 0.8 and -0.6), of an MA(1) (theta = 0.6) and of the random '
              'walk; running time averages of an ergodic AR(1) and of the non-ergodic process X_t = Z + e_t.',
         keywords='Wold decomposition, impulse response, MA(infinity), ergodicity, time average, ensemble average',
         consts=CONSTS, funcs=CORE + [g.fig_wold, g.fig_ergodicity], run='print(fig_wold())\nprint(fig_ergodicity())',
         charts=['tsa_ch1_wold', 'tsa_ch1_ergodicity'], data=NODATA),
    dict(name='TSA_ch1_acf_pacf',
         desc='Sample and theoretical ACF of white noise, AR(1), MA(1) and a random walk; the sampling distribution of '
              'the sample ACF of white noise against Bartlett\'s N(0, 1/T) and the number of lags outside the 95% band; '
              'sample ACF and PACF of AR(1), AR(2) and MA(1); ACF of BET daily returns, absolute and squared returns.',
         keywords='autocorrelation function, ACF, partial autocorrelation, PACF, correlogram, Bartlett, confidence band, '
                  'volatility clustering, BET',
         consts=CONSTS, funcs=CORE + [g.fig_acf_models, g.fig_bartlett, g.fig_acf_pacf, g.fig_returns_acf],
         run="print(fig_acf_models())\nprint(fig_bartlett())\nprint(fig_acf_pacf())\nprint(fig_returns_acf('bet'))",
         charts=['tsa_ch1_acf_models', 'tsa_ch1_bartlett', 'tsa_ch1_acf_pacf', 'tsa_ch1_bet_acf']),
    dict(name='TSA_ch1_portmanteau',
         desc='Box-Pierce and Ljung-Box tests: their size at the 5% level for Gaussian white noise with T = 50, 100, 500 '
              'and m = 10, 20; Ljung-Box statistics Q*(m), m = 1..30, of S&P 500, BET and EUR/RON returns, of squared '
              'returns and of Romanian GDP growth; a table of ten real series (sample ACF and Q*(10)).',
         keywords='portmanteau test, Box-Pierce, Ljung-Box, white noise test, size, simulation, S&P 500, BET, EUR/RON, '
                  'GDP, inflation, Nile, sunspots',
         consts=CONSTS, funcs=CORE + [g.lb_size, g.fig_lb_size, g.real_series, g.fig_lb_stats],
         run='print(fig_lb_size())\nT = fig_lb_stats()\nprint(pd.DataFrame({k: {"T": v["n"], "rho1": v["r1"], '
             '"Q*(10)": v["m10"]["lb"], "p": v["m10"]["lb_p"]} for k, v in T.items()}).T.round(4))',
         charts=['tsa_ch1_lb_size', 'tsa_ch1_lb_stats'], extra=['ch1_real_series.csv']),
    dict(name='TSA_ch1_transformations',
         desc='Transformations: the S&P 500 log price and its first difference; Romanian real GDP in logs, quarterly and '
              'annual log differences with their ACF; the Box-Cox transformation of Romanian nominal GDP with the lambda '
              'of Guerrero (1993); over-differencing white noise.',
         keywords='transformation, logarithm, differencing, seasonal difference, Box-Cox, Guerrero, over-differencing, '
                  'GDP, Romania, S&P 500',
         consts=CONSTS, funcs=CORE + [g.fig_sp500_diff, g.fig_gdp_transform, g.boxcox, g.guerrero_cv, g.guerrero_lambda,
                                      g.fig_boxcox, g.fig_overdiff],
         run='print(fig_sp500_diff())\nprint(fig_gdp_transform())\nprint(fig_boxcox())\nprint(fig_overdiff())',
         charts=['tsa_ch1_sp500_diff', 'tsa_ch1_gdp_transform', 'tsa_ch1_boxcox', 'tsa_ch1_overdiff']),
    dict(name='TSA_ch1_textbook_series',
         desc='Two classic series shipped with statsmodels: the yearly Nile flow at Aswan (1871-1970), whose ACF decays '
              'slowly because of a level shift around 1898, and the yearly sunspot numbers (1700-2008), whose ACF is a '
              'damped wave with the 11-year solar cycle.',
         keywords='Nile, structural break, change point, sunspots, cycle, ACF, Ljung-Box, statsmodels data sets',
         consts=CONSTS, funcs=CORE + [g.fig_textbook], run='print(fig_textbook())', charts=['tsa_ch1_textbook'],
         data='statsmodels data sets: Nile (1871-1970) and sunspots (1700-2008)'),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar1 as s
    QUANTLETS.append(dict(
        name='TSA_ch1_seminar',
        desc='Seminar 1 of Time Series Analysis: autocovariances of an MA(1) and of a random walk, a sample ACF and a '
             'Ljung-Box statistic by hand; BET log price and returns (ACF, Ljung-Box on returns and squares); Romanian '
             'real GDP in logs, quarterly and annual growth with their ACF.',
        keywords='seminar, autocovariance, MA(1), random walk, sample ACF, Ljung-Box, BET, GDP, Romania',
        consts=CONSTS, funcs=CORE + [s.ma_acvf, s.rw_moments, s.acf_by_hand, s.b1_returns, s.b3_gdp],
        run="print(ma_acvf([0.5], 2.0))\nprint(rw_moments(0.25, 100, 120))\nprint(acf_by_hand([4, 6, 5, 8, 7, 9, 6, 7]))\n"
            "print(b1_returns('bet', fname='ch1_sem_b1'))\nprint(b3_gdp())",
        charts=['ch1_sem_b1', 'ch1_sem_b3']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 1, 'Stochastic processes and stationarity', HERE, data=DATA, submitted=SUBMITTED)
