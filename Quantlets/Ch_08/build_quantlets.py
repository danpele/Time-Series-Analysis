"""
build_quantlets.py -- Quantlet folders of Chapter 8 (TSA): long memory and ARFIMA
================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_08/generate_all_charts.py && python3 Quantlets/Ch_08/seminar8.py
      python3 Quantlets/Ch_08/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 8
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
DATA = ('Nile flow 1871-1970 (statsmodels); Romanian monthly HICP (Eurostat prc_hicp_minr); US CPI and unemployment '
        '(FRED CPIAUCSL, UNRATE); daily S&P 500 and BET from EODHD (data/market of the TSA repository); EUR/RON '
        'reference rate of the BNR')
NODATA = 'Simulated data only (no market data)'
INSTALL = ("# The arch package (FIGARCH estimation) is not preinstalled in Google Colab: pip install arch\n"
           "import importlib.util, subprocess, sys\n"
           "if importlib.util.find_spec('arch') is None:\n"
           "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'arch'])")
CONSTS = ['from scipy import special', 'from statsmodels.tsa.stattools import adfuller',
          'from statsmodels.tsa.ar_model import AutoReg',
          f'SEED = {g.SEED!r}', f'HICP = {g.HICP!r}', f'INFL_START = {g.INFL_START!r}', f'BW = {g.BW!r}',
          f'NMIN = {g.NMIN!r}', f'WINDOW, STEP = {g.WINDOW!r}, {g.STEP!r}', f'MC_REPS = {g.MC_REPS!r}',
          f'LO_CRIT = {g.LO_CRIT!r}', f'NAME = {g.NAME!r}', f'NILE_BREAK = {g.NILE_BREAK!r}',
          f'SHIFTS = {g.SHIFTS!r}', f'SWITCH = {g.SWITCH!r}', f'RS_EXAMPLE = {g.RS_EXAMPLE!r}',
          "COLORS = {'nile': st.MainBlue, 'infl': st.IDAred, 'infl12': st.Orange, 'usinfl': st.Forest, 'unrate': st.Purple, "
          "'sp500': st.COL['sp500'], 'bet': st.COL['bet'], 'eurron': st.COL['eurron'], 'abs': st.Purple, 'sq': st.Orange}",
          'HERE = "."']
CORE = [g.nile, g.ro_hicp, g.ro_inflation12, g.seasonal_adjust, g.ro_inflation, g.us_inflation, g.us_unemployment,
        g.returns, g.monthly_rv, g.parkinson_vol, g.save, g.frac_weights, g.frac_diff, g.arfima_acvf0, g.arma_psi,
        g.arfima_acvf, g.arfima_acf, g.ma_inf_weights, g.ar_inf_weights, g.fgn_acvf, g.circulant_gaussian, g.sim_arfima,
        g.sim_fgn, g.acf, g.block_sizes, g.rs_curve, g.hurst_rs, g.rs_steps, g.dfa_curve, g.hurst_dfa, g.periodogram,
        g.bandwidth, g.gph, g.local_whittle, g.dl_quad_logdet, g._unpack, g._ok, g.arfima_exact_ml, g._se_numeric,
        g.arfima_spec, g.arfima_whittle, g.lo_test, g.all_estimates, g.arfima_forecast, g.ar_forecast, g.oos_inflation,
        g.rolling_d, g.rolling_hurst, g.mc_band, g.figarch_weights]

QUANTLETS = [
    dict(name='TSA_ch8_memory_data',
         desc='Four persistent series: the Nile flow 1871-1970, Romanian HICP inflation (12-month and monthly, seasonally '
              'adjusted), the US unemployment rate, monthly realised volatility of the S&P 500 and the BET; their sample '
              'ACF against the exponential decay of an AR(1) with the same lag-1 autocorrelation.',
         keywords='long memory, autocorrelation, hyperbolic decay, Nile, inflation, Romania, unemployment, realised volatility',
         consts=CONSTS, funcs=CORE + [g.fig_memory_data, g.fig_memory_acf],
         run='print(fig_memory_data())\nprint(fig_memory_acf())', charts=['tsa_ch8_memory_data', 'tsa_ch8_memory_acf']),
    dict(name='TSA_ch8_fractional_differencing',
         desc='The weights of the fractional difference (1 - L)^d and the impulse responses of (1 - L)^(-d) for d = 0.2, '
              '0.4, 0.7, 1 against an AR(1); fixed-window fractional differencing of the log S&P 500: ADF statistic and '
              'correlation with the level against d; the worked examples of the slides.',
         keywords='fractional differencing, fractional integration, binomial weights, ADF test, S&P 500, mean reversion',
         consts=CONSTS, funcs=CORE + [g.worked_examples, g.fig_weights, g.fig_ffd],
         run='print(worked_examples())\nprint(fig_weights())\nprint(fig_ffd())', charts=['tsa_ch8_weights', 'tsa_ch8_ffd']),
    dict(name='TSA_ch8_arfima_processes',
         desc='ARFIMA(0,d,0): theoretical ACF against an AR(1) (linear and log-log), exact simulation by circulant '
              'embedding for d = 0.4 and d = -0.4; fractional Brownian motion and fractional Gaussian noise for H = 0.3, '
              '0.5, 0.7.',
         keywords='ARFIMA, fractional Gaussian noise, fractional Brownian motion, Hurst exponent, circulant embedding, simulation',
         consts=CONSTS, funcs=CORE + [g.fig_decay, g.fig_arfima_paths, g.fig_fbm],
         run='print(fig_decay())\nprint(fig_arfima_paths())\nprint(fig_fbm())',
         charts=['tsa_ch8_decay', 'tsa_ch8_arfima_paths', 'tsa_ch8_fbm'], data=NODATA),
    dict(name='TSA_ch8_hurst_rs_dfa',
         desc='R/S analysis (Hurst 1951) and detrended fluctuation analysis (Peng et al. 1994) of US monthly inflation and '
              'of daily S&P 500 returns and absolute returns: log-log plots and slopes.',
         keywords='Hurst exponent, rescaled range, R/S, DFA, US inflation, S&P 500, absolute returns',
         consts=CONSTS, funcs=CORE + [g.fig_rs_dfa], run='print(fig_rs_dfa())', charts=['tsa_ch8_rs_dfa']),
    dict(name='TSA_ch8_estimation',
         desc='Estimating d: the GPH log-periodogram regression (Nile, Romanian inflation), the local Whittle estimator, '
              'sensitivity to the bandwidth, exact Gaussian maximum likelihood of ARFIMA(p,d,q) by the Durbin-Levinson '
              'algorithm against ARMA models (AIC, BIC), fitted autocorrelations, and a table of estimates for twelve series.',
         keywords='GPH, local Whittle, periodogram, bandwidth, exact maximum likelihood, ARFIMA, ARMA, model selection, inflation',
         consts=CONSTS, funcs=CORE + [g.fig_gph, g.fig_bandwidth, g.model_table, g.fig_arfima_fit, g.memory_table],
         run='print(fig_gph())\nprint(fig_bandwidth())\nprint(fig_arfima_fit())\nprint(memory_table())',
         charts=['tsa_ch8_gph', 'tsa_ch8_bandwidth', 'tsa_ch8_arfima_fit'], extra=['ch8_memory_table.csv']),
    dict(name='TSA_ch8_monte_carlo',
         desc='Monte Carlo of five estimators of d (GPH, local Whittle, Whittle ML of ARFIMA(0,d,0), R/S and DFA) for white '
              'noise, ARFIMA(0,0.3,0) and an AR(1) with phi = 0.6 (T = 500, 300 samples).',
         keywords='Monte Carlo, bias, variance, GPH, local Whittle, Whittle, R/S, DFA, short-memory contamination',
         consts=CONSTS, funcs=CORE + [g.fig_mc], run='print(fig_mc())', charts=['tsa_ch8_mc'], data=NODATA),
    dict(name='TSA_ch8_forecasting',
         desc='ARFIMA forecasts from the AR(infinity) form: US monthly inflation 36 months ahead (ARFIMA(1,d,0) against '
              'AR(p)); pseudo out-of-sample RMSE by horizon of ARFIMA, AR, the random walk and the mean for Romanian and US '
              'monthly inflation.',
         keywords='ARFIMA forecast, AR(infinity), out-of-sample evaluation, RMSE, inflation, Romania, United States',
         consts=CONSTS, funcs=CORE + [g.fig_forecast_path, g.fig_forecast],
         run='print(fig_forecast_path())\nprint(fig_forecast())', charts=['tsa_ch8_forecast_path', 'tsa_ch8_forecast']),
    dict(name='TSA_ch8_volatility_memory',
         desc='Long memory in volatility: ACF of r, |r| and r^2 for the S&P 500 and the BET (Ding, Granger and Engle 1993), '
              'the shuffle test, monthly log realised volatility with ARFIMA(1,d,0) and ARMA(1,1) by exact ML, GARCH(1,1) '
              'against FIGARCH(1,d,1) (arch package), and a HAR model for the daily range-based volatility.',
         keywords='volatility, absolute returns, realised volatility, FIGARCH, GARCH, HAR, long memory, S&P 500, BET',
         consts=CONSTS, funcs=CORE + [g.fig_vol_acf, g.fig_rv, g.fig_vol_models],
         run='print(fig_vol_acf())\nprint(fig_rv())\nprint(fig_vol_models())',
         charts=['tsa_ch8_vol_acf', 'tsa_ch8_rv', 'tsa_ch8_vol_models']),
    dict(name='TSA_ch8_spurious_memory',
         desc='Spurious long memory: Monte Carlo of a mean shift and of a Markov-switching mean (Diebold and Inoue 2001); '
              'the Nile and its 1898 break (Cobb 1978); rolling local Whittle d of US inflation and rolling DFA exponents '
              'of S&P 500 and BET returns with Monte Carlo bands.',
         keywords='spurious long memory, structural break, regime switching, Nile, rolling Hurst exponent, Monte Carlo bands',
         consts=CONSTS, funcs=CORE + [g.fig_spurious, g.fig_nile_break, g.fig_rolling],
         run='print(fig_spurious())\nprint(fig_nile_break())\nprint(fig_rolling())',
         charts=['tsa_ch8_spurious', 'tsa_ch8_nile_break', 'tsa_ch8_rolling']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar8 as s
    QUANTLETS.append(dict(
        name='TSA_ch8_seminar',
        desc='Seminar 8 of Time Series Analysis: fractional weights and ARFIMA autocorrelations, R/S and a GPH slope by '
             'hand, a one-step ARFIMA forecast, GARCH and FIGARCH weights; the Nile and its break; Romanian inflation; '
             'the memory of S&P 500 volatility and the shuffle test; BET and EUR/RON before and after 2008.',
        keywords='seminar, long memory, ARFIMA, Hurst exponent, GPH, local Whittle, Nile, inflation, volatility, BET, EUR/RON',
        consts=CONSTS, funcs=CORE + [s.weights_by_hand, s.hurst_from_points, s.gph_by_hand, s.forecast_by_hand,
                                     s.b1_nile, s.b3_sp500],
        run="print(weights_by_hand(0.25))\nprint(hurst_from_points([16, 64, 256], [4.1, 9.6, 22.4]))\n"
            "print(forecast_by_hand(0.4, 2.0, [3.0, 2.5, 2.8, 1.5]))\nprint(b1_nile())\nprint(b3_sp500())",
        charts=['ch8_sem_b1', 'ch8_sem_b3']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 8, 'Long memory and ARFIMA', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
