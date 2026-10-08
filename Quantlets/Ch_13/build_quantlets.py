"""
build_quantlets.py -- Quantlet folders of Chapter 13 (TSA): speculative bubbles and LPPL models
==============================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_13/generate_all_charts.py
      python3 Quantlets/Ch_13/build_quantlets.py
      python3 notebooks/add_colab_banner.py
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
DATA = ('Daily closes of the S&P 500, Nasdaq 100, BET, Shanghai Composite and Bitcoin from EODHD (data/market of the '
        'TSA repository)')
NODATA = 'Simulated data only (no market data)'
INSTALL = ("# The arch package (Phillips-Perron test of the LPPL residuals) is not preinstalled in Google Colab: pip install arch\n"
           "import importlib.util, subprocess, sys\n"
           "if importlib.util.find_spec('arch') is None:\n"
           "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'arch'])")
CONSTS = ['from scipy.signal import lombscargle',
          f'SEED = {g.SEED!r}', f'EPISODES = {g.EPISODES!r}',
          "COLORS = {'sp500': st.COL['sp500'], 'ndx': st.Teal, 'bet': st.Orange, 'ssec': st.Purple, 'btc17': st.COL['btc'], "
          "'btc21': st.COL['btc']}",
          f'FIT_LAG_DAYS = {g.FIT_LAG_DAYS!r}', f'PSY_SAMPLES = {g.PSY_SAMPLES!r}', f'PSY_LABEL = {g.PSY_LABEL!r}',
          f'PSY_REPS = {g.PSY_REPS!r}', f'LPPLS_SEARCH = {g.LPPLS_SEARCH!r}', f'LPPLS_FILTER = {g.LPPLS_FILTER!r}',
          'WINDOWS = list(range(750, 45, -25))', f'CI_STEP = {g.CI_STEP!r}', f'CRASH, HORIZON = {g.CRASH!r}, {g.HORIZON!r}',
          f'CI_LEVEL = {g.CI_LEVEL!r}', f'EVAL_SAMPLES = {g.EVAL_SAMPLES!r}', f'PARAM_KEYS = {g.PARAM_KEYS!r}',
          'HERE = "."']
CORE = [g.prices, g.weekly, g.yrs, g.todate, g.d2s, g.episode, g.drawdown, g.save, g._cums, g._adf_end, g.adf_window,
        g.min_window, g.psy, g.psy_cv, g.episodes_above, g.run_psy, g.lppl_design, g.lppl_linear, g._damping, g._grid,
        g.lppl_fit, g.lomb_pvalue, g.ar1_pass, g.lppl_conditions, g.qualified, g.lppl_path, g.fit_window, g._ci_point,
        g.confidence_series, g.cached_ci, g.future_fall, g.blanchard_watson, g.ar1_paths, g.crash_list, g.hit_rates,
        g.alarm_episodes, g.eval_ci, g.episode_ci, g.tc_path, g.windows_at, g.worked_examples]

QUANTLETS = [
    dict(name='TSA_ch13_episodes',
         desc='Six speculative run-ups and crashes in the course data: S&P 500 and Nasdaq 100 (2000), BET (2007), Shanghai '
              'Composite (2015), Bitcoin (2017 and 2021). The low before each run-up, the peak, the size of the run-up, the '
              'annual growth rate and the fall in the year after the peak.',
         keywords='speculative bubble, crash, run-up, drawdown, S&P 500, Nasdaq 100, BET, Shanghai Composite, Bitcoin',
         consts=CONSTS, funcs=CORE + [g.fig_episodes], run='print(fig_episodes())', charts=['tsa_ch13_episodes']),
    dict(name='TSA_ch13_rational_bubble',
         desc='A rational bubble of Blanchard and Watson (1982) added to a random-walk fundamental value; stationary, '
              'unit-root and explosive AR(1) paths with the same shocks; the worked examples of the chapter.',
         keywords='rational bubble, Blanchard-Watson, explosive root, unit root, AR(1), simulation',
         consts=CONSTS, funcs=CORE + [g.fig_rational], run='print(worked_examples())\nprint(fig_rational())',
         charts=['tsa_ch13_rational'], data=NODATA),
    dict(name='TSA_ch13_explosive_roots',
         desc='Right-tailed ADF, SADF (Phillips, Wu and Yu 2011), GSADF and BSADF date-stamping (Phillips, Shi and Yu '
              '2015) on weekly log prices of the Nasdaq 100 (1990-2004), Bitcoin (2014-2023), BET (2000-2012) and the '
              'Shanghai Composite (2010-2018), with Monte Carlo critical values.',
         keywords='explosive root, right-tailed unit root test, SADF, GSADF, BSADF, date-stamping, bubble, Nasdaq, Bitcoin, BET',
         consts=CONSTS, funcs=CORE + [g.fig_psy_ndx, g.fig_psy_panel], run='print(fig_psy_ndx())\nprint(fig_psy_panel())',
         charts=['tsa_ch13_psy_ndx', 'tsa_ch13_psy_panel']),
    dict(name='TSA_ch13_lppl_model',
         desc='The log-periodic power law (Johansen, Ledoit and Sornette 2000): exponential against super-exponential '
              'growth and their growth rates; the power-law term for several m, the log-periodic oscillation for several '
              'omega and the full LPPL path; the scaling ratio lambda = exp(2 pi / omega).',
         keywords='LPPL, super-exponential growth, finite-time singularity, log-periodic oscillations, discrete scale invariance',
         consts=CONSTS, funcs=CORE + [g.fig_growth, g.fig_lppl_components],
         run='print(fig_growth())\nprint(fig_lppl_components())', charts=['tsa_ch13_growth', 'tsa_ch13_lppl_components'],
         data=NODATA),
    dict(name='TSA_ch13_estimation',
         desc='LPPL estimation with the two-step method of Filimonov and Sornette (2013): OLS for A, B, C1, C2, grid search '
              'and Nelder-Mead for tc, m, omega; the cost landscape of the Nasdaq 100 window before March 2000; fits 30 '
              'days before six peaks with the filter conditions of Shu and Zhu (2020).',
         keywords='LPPL, Filimonov-Sornette, nonlinear least squares, critical time, filter conditions, Lomb periodogram',
         consts=CONSTS, funcs=CORE + [g.fig_cost, g.fig_fits], run='print(fig_cost())\nfits = fig_fits()\nprint(fits)',
         charts=['tsa_ch13_cost', 'tsa_ch13_fits']),
    dict(name='TSA_ch13_confidence',
         desc='Confidence from many windows: the estimated critical time for 29 windows ending at one date, the critical '
              'time as the end of the window moves, and the LPPLS confidence indicator around the peaks of the Nasdaq '
              '100 (2000), Shanghai Composite (2015), BET (2007) and Bitcoin (2017).',
         keywords='LPPLS confidence indicator, multiple windows, critical time, look-ahead bias, bubble diagnosis',
         consts=CONSTS, funcs=CORE + [g.fig_windows, g.fig_tc_path, g.fig_ci],
         run='print(fig_windows())\nprint(fig_tc_path())\nprint(fig_ci())',
         charts=['tsa_ch13_windows', 'tsa_ch13_tc_path', 'tsa_ch13_ci', 'tsa_ch13_ci_b'],
         extra=['ch13_ci_ep_ndx.csv', 'ch13_ci_ep_ssec.csv', 'ch13_ci_ep_bet.csv', 'ch13_ci_ep_btc17.csv']),
    dict(name='TSA_ch13_evaluation',
         desc='An honest evaluation of crash prediction: the LPPLS confidence indicator over the whole S&P 500 sample '
              '(since 1993) and the Bitcoin sample (since 2014); alarms against falls of at least 20% within 182 days, the '
              'hit rate by threshold against the base rate, alarm clusters and false alarms.',
         keywords='crash prediction, false alarms, base rate, hit rate, LPPLS confidence indicator, S&P 500, Bitcoin',
         consts=CONSTS, funcs=CORE + [g.fig_eval], run='ev = fig_eval()\nprint({k: v["hit"] for k, v in ev.items()})',
         charts=['tsa_ch13_eval', 'tsa_ch13_eval_btc'], extra=['ch13_ci_eval_sp500.csv', 'ch13_ci_eval_btc.csv']),
]

if __name__ == '__main__':
    build_all(QUANTLETS, 13, 'Speculative bubbles: LPPL models', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
