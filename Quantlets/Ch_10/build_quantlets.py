"""
build_quantlets.py -- Quantlet folders of Chapter 10 (TSA): state space models, Kalman filter, Markov switching
==============================================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_10/generate_all_charts.py && python3 Quantlets/Ch_10/seminar10.py
      python3 Quantlets/Ch_10/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 10
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
DATA = ('Nile flow 1871-1970 (statsmodels); Romanian real GDP and HICP (Eurostat namq_10_gdp, prc_hicp_minr); US real '
        'GDP, potential GDP, industrial production, coincident indicators and NBER recession dates (FRED GDPC1, GDPPOT, '
        'INDPRO, PAYEMS, W875RX1, CMRMTSPL, USREC); weekly BET, Euro Stoxx 50 and S&P 500 from EODHD daily closes (data/market '
        'of the TSA repository)')
NODATA = 'Simulated data and FRED GDPC1 (US real GDP)'
INSTALL = ("# The arch package (GARCH estimation) is not preinstalled in Google Colab: pip install arch\n"
           "import importlib.util, subprocess, sys\n"
           "if importlib.util.find_spec('arch') is None:\n"
           "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'arch'])")
CONSTS = ['import statsmodels.api as sm',
          f'SEED = {g.SEED!r}', f'GDP_RO = {g.GDP_RO!r}', f'HP_LAMBDA = {g.HP_LAMBDA!r}', f'HAM_H, HAM_P = {g.HAM_H!r}, {g.HAM_P!r}',
          f'DIFFUSE = {g.DIFFUSE!r}', f'NILE_GAPS = {g.NILE_GAPS!r}', f'HAMILTON_SAMPLE = {g.HAMILTON_SAMPLE!r}',
          f'MS_SAMPLE = {g.MS_SAMPLE!r}', f'COINCIDENT = {g.COINCIDENT!r}', f'COINC_NAMES = {g.COINC_NAMES!r}',
          f'TVP_PAIR = {g.TVP_PAIR!r}', f'TVP_START = {g.TVP_START!r}', f'ROLL = {g.ROLL!r}', f'SEARCH = {g.SEARCH!r}',
          'SHADE = st.Amber']
CORE = [g.nile, g.ro_gdp, g.us_gdp_growth, g.nber, g.weekly_returns, g.save, g.shade, g.kalman_filter, g.rts_smoother,
        g.disturbance_smoother, g.local_level_filter, g.local_level_ml, g.steady_state, g.q_from_alpha, g.profile_loglik,
        g.ses_path, g.worked_example]
UC = [g.uc_fit, g.hp, g.hamilton_filter, g.hp_realtime]
MS = [g.ms_fit, g.regime_order, g.ms_summary, g.concordance]

QUANTLETS = [
    dict(name='TSA_ch10_state_space_examples',
         desc='The state space form: simulated local level paths for three signal-to-noise ratios (the same shocks) and a '
              'local linear trend; an AR(2) of US real GDP growth in state space form, whose Kalman-filter log-likelihood '
              'with the stationary initial covariance equals the exact likelihood of statsmodels SARIMAX.',
         keywords='state space form, local level, local linear trend, signal-to-noise ratio, AR(2), companion matrix, Kalman likelihood',
         consts=CONSTS, funcs=CORE + [g.fig_ss_examples, g.arma_ss_check],
         run='print(fig_ss_examples())\nprint(arma_ss_check())', charts=['tsa_ch10_ss_examples'], data=NODATA),
    dict(name='TSA_ch10_kalman_filter',
         desc='The Kalman filter written out in numpy: a three-period local level example by hand; the local level model of '
              'the Nile 1871-1970 by maximum likelihood (Durbin and Koopman 2012), filtered level and predictions with 90% '
              'bands, the Kalman gain and its steady state, and the equivalence with simple exponential smoothing (Muth 1960).',
         keywords='Kalman filter, Kalman gain, local level, Nile, steady state, simple exponential smoothing, maximum likelihood',
         consts=CONSTS, funcs=CORE + [g.fig_nile_filter, g.statsmodels_local_level, g.fig_gain_ses],
         run='print(worked_example())\nprint(fig_nile_filter())\nprint(statsmodels_local_level())\nprint(fig_gain_ses())',
         charts=['tsa_ch10_nile_filter', 'tsa_ch10_gain_ses']),
    dict(name='TSA_ch10_smoothing_likelihood',
         desc='Smoothing, likelihood and diagnostics of the Nile local level model: filtered against smoothed level '
              '(Rauch-Tung-Striebel), the profile log-likelihood over the signal-to-noise ratio, standardised prediction '
              'errors (Ljung-Box, Jarque-Bera) and the auxiliary residuals of the disturbance smoother, which locate the '
              'level shift of 1898-1899.',
         keywords='Kalman smoother, Rauch-Tung-Striebel, prediction-error decomposition, profile likelihood, auxiliary residuals, structural break, Nile',
         consts=CONSTS, funcs=CORE + [g.fig_nile_smooth, g.fig_likelihood, g.fig_diagnostics],
         run='print(fig_nile_smooth())\nprint(fig_likelihood())\nprint(fig_diagnostics())',
         charts=['tsa_ch10_nile_smooth', 'tsa_ch10_likelihood', 'tsa_ch10_diagnostics']),
    dict(name='TSA_ch10_missing_data',
         desc='Missing data and forecasting with the Kalman filter: the Nile with 1891-1910 and 1931-1950 removed (smoothed '
              'level with 90% bands) and forecasts for 1971-2000 obtained by treating the future as missing values.',
         keywords='missing values, Kalman smoother, forecasting, forecast intervals, local level, Nile',
         consts=CONSTS, funcs=CORE + [g.fig_missing], run='print(fig_missing())', charts=['tsa_ch10_missing']),
    dict(name='TSA_ch10_trend_cycle',
         desc='Trend and cycle of Romanian real GDP (Eurostat): an unobserved-components model (smooth trend plus AR(2) '
              'cycle, statsmodels UnobservedComponents) against the Hodrick-Prescott filter and the regression filter of '
              'Hamilton (2018); real-time (one-sided) against final (two-sided) output gaps.',
         keywords='unobserved components, output gap, Hodrick-Prescott filter, Hamilton filter, real time, revisions, Romania, GDP',
         consts=CONSTS, funcs=CORE + UC + [g.fig_ro_trend, g.fig_output_gap, g.fig_realtime],
         run='print(fig_ro_trend())\nprint(fig_output_gap())\nprint(fig_realtime())',
         charts=['tsa_ch10_ro_trend', 'tsa_ch10_output_gap', 'tsa_ch10_realtime']),
    dict(name='TSA_ch10_tvp_regression',
         desc='A regression with a time-varying (random-walk) coefficient by the Kalman filter: the beta of weekly BET '
              'returns on the Euro Stoxx 50, 2005-2026, smoothed with a 90% band, against 52-week rolling OLS and the '
              'constant OLS beta; likelihood-ratio test of a constant beta.',
         keywords='time-varying parameters, Kalman filter, beta, rolling regression, BET, Euro Stoxx 50, financial integration',
         consts=CONSTS, funcs=CORE + [g.tvp_filter, g.tvp_data, g.fig_tvp_beta], run='print(fig_tvp_beta())',
         charts=['tsa_ch10_tvp_beta']),
    dict(name='TSA_ch10_dynamic_factor',
         desc='A one-factor dynamic factor model of four US coincident indicators (Stock and Watson 1989) with statsmodels '
              'DynamicFactor: estimated on 1967-2019, run to the last month with a ragged edge (missing latest values), '
              'against the NBER recession dates.',
         keywords='dynamic factor model, nowcasting, ragged edge, coincident index, NBER recessions, Kalman filter',
         consts=CONSTS, funcs=CORE + [g.coincident, g.fig_dfm], run='print(fig_dfm())', charts=['tsa_ch10_dfm']),
    dict(name='TSA_ch10_markov_gdp',
         desc='Markov switching (Hamilton 1989) for GDP growth: the MS-AR(4) on 1951-1984 with two local maxima of the '
              'likelihood; a switching mean on US real GDP growth 1947-2019 against the NBER dates, extended to 2026; '
              'filtered against smoothed probabilities in 2008; specifications that find other regimes (variance, 2020); '
              'Romanian GDP growth regimes with switching mean and variance.',
         keywords='Markov switching, Hamilton filter, recession probabilities, NBER, expected duration, regime, Romania, GDP',
         consts=CONSTS, funcs=CORE + MS + [g.fig_ms_us, g.fig_ms_filtered, g.ms_us_pitfalls, g.fig_ms_ro],
         run='print(fig_ms_us())\nprint(fig_ms_filtered())\nprint(ms_us_pitfalls())\nprint(fig_ms_ro())',
         charts=['tsa_ch10_ms_us', 'tsa_ch10_ms_filtered', 'tsa_ch10_ms_ro']),
    dict(name='TSA_ch10_volatility_regimes',
         desc='Calm and turbulent regimes of weekly S&P 500 returns (switching mean and variance) against GARCH(1,1); '
              'series simulated from the fitted regime model reproduce the ACF of absolute returns, a positive local '
              'Whittle d and a high GARCH persistence (Lamoureux and Lastrapes 1990; Diebold and Inoue 2001).',
         keywords='volatility regimes, Markov switching, GARCH, persistence, spurious long memory, local Whittle, S&P 500',
         consts=CONSTS, funcs=CORE + MS + [g.ms_vol, g.local_whittle, g.simulate_ms, g.fig_vol_regimes, g.fig_regimes_memory],
         run='print(fig_vol_regimes())\nprint(fig_regimes_memory())', charts=['tsa_ch10_vol_regimes', 'tsa_ch10_regimes_memory']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar10 as s
    QUANTLETS.append(dict(
        name='TSA_ch10_seminar',
        desc='Seminar 10 of Time Series Analysis: the Kalman filter of a local level model by hand (with a missing value), '
             'exponential smoothing as a steady-state Kalman filter, state space forms and forecasts, durations and one '
             'step of the Hamilton filter; the underlying level of Romanian inflation; the US output gap against the CBO; '
             'recessions in US industrial production; volatility regimes of the BET.',
        keywords='seminar, Kalman filter, local level, exponential smoothing, output gap, Markov switching, NBER, BET, inflation',
        consts=CONSTS + [f'HICP = {s.HICP!r}', f'INFL_START = {s.INFL_START!r}', f'IP_SAMPLE = {s.IP_SAMPLE!r}'],
        funcs=CORE + UC + MS + [s.kalman_by_hand, s.ses_equivalence, s.state_space_forecasts, s.durations, s.hamilton_step,
                                s.ro_monthly_inflation, s.b1_inflation_level, s.b3_ip_regimes],
        run="print(kalman_by_hand([1.0, 3.0, 2.0], 0.0, 1.0, 1.0, 1.0))\nprint(ses_equivalence())\n"
            "print(durations(0.75, 0.95))\nprint(b1_inflation_level())\nprint(b3_ip_regimes())",
        charts=['ch10_sem_b1', 'ch10_sem_b3']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 10, 'State space models, Kalman filter and Markov switching', HERE, data=DATA, submitted=SUBMITTED,
              install=INSTALL)
