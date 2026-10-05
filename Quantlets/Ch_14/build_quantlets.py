"""
build_quantlets.py -- Quantlet folders of Chapter 14 (TSA): multivariate GARCH models
====================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_14/generate_all_charts.py
      python3 Quantlets/Ch_14/build_quantlets.py
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
DATA = ('Daily S&P 500, DAX, BET and Bitcoin from EODHD (data/market of the TSA repository); EUR/RON reference rate '
        'of the BNR (public XML archives)')
NODATA = 'No data (formulas only)'
INSTALL = ("# The arch package (step 1: univariate GARCH) is not preinstalled in Google Colab: pip install arch\n"
           "# (without it, garch11_fit falls back to a numpy maximum-likelihood estimator)\n"
           "import importlib.util, subprocess, sys\n"
           "if importlib.util.find_spec('arch') is None:\n"
           "    try:\n"
           "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'arch'])\n"
           "    except Exception as e:\n"
           "        print('arch not installed, the numpy fallback will be used:', e)")
CONSTS = ['from scipy.signal import lfilter',
          f'NAME = {g.NAME!r}', f'PANEL = {g.PANEL!r}', f'PANEL_START = {g.PANEL_START!r}', f'PAIR_START = {g.PAIR_START!r}',
          f'HEDGE_START = {g.HEDGE_START!r}', f'SPLIT = {g.SPLIT!r}', f'ROLL = {g.ROLL!r}', f'ALPHA_VAR = {g.ALPHA_VAR!r}',
          f'WEIGHTS = {g.WEIGHTS!r}', f'CRISES = {g.CRISES!r}', f'CALM = {g.CALM!r}', f'PANEL_CRISIS = {g.PANEL_CRISIS!r}',
          f'EX = {g.EX!r}',
          "COLORS = {'sp500': st.MainBlue, 'dax': st.Teal, 'bet': st.IDAred, 'eurron': st.Forest, 'btc': st.Amber}",
          'HERE = "."']
CORE = [g.save, g.shade, g.common_returns, g.weekly_returns, g.garch11_negloglik, g.garch11_variance, g.garch11_fit,
        g.garch11_filter, g.step1, g.dcc_path, g.dcc_loglik, g.num_hessian, g.dcc_fit, g.half_life]

QUANTLETS = [
    dict(name='TSA_ch14_correlations',
         desc='Why multivariate volatility: the risk of a 50/50 portfolio of two assets against the correlation, and rolling '
              '250-day correlations of daily returns on common trading days (S&P 500 and DAX, BET and DAX, S&P 500 and '
              'Bitcoin); the asynchronous closes of April 2025; the worked examples of the chapter.',
         keywords='correlation, rolling correlation, diversification, portfolio risk, asynchronous trading, S&P 500, DAX, BET, Bitcoin',
         consts=CONSTS, funcs=CORE + [g.worked_examples, g.fig_diversification, g.fig_rolling_corr],
         run='print(worked_examples())\nfig_diversification()\nprint(fig_rolling_corr())',
         charts=['tsa_ch14_diversification', 'tsa_ch14_rolling_corr']),
    dict(name='TSA_ch14_parameters',
         desc='The number of parameters of the variance equation of VEC(1,1), diagonal VEC, BEKK(1,1), diagonal and scalar '
              'BEKK, CCC and DCC models for N = 2 to 50 assets (the curse of dimensionality).',
         keywords='multivariate GARCH, VEC, BEKK, CCC, DCC, number of parameters, curse of dimensionality',
         consts=CONSTS, funcs=[g.save, g.n_params, g.fig_param_count], run='print(fig_param_count())',
         charts=['tsa_ch14_param_count'], data=NODATA),
    dict(name='TSA_ch14_dcc_estimation',
         desc='Two-step DCC estimation (Engle 2002) for daily S&P 500 and DAX returns on common trading days since 2000: '
              'step 1 a GARCH(1,1) per index, step 2 the DCC parameters (a, b) by maximum likelihood with correlation '
              'targeting; likelihood-ratio test against constant correlation (CCC, Bollerslev 1990); DCC against the '
              'rolling correlation; the crises of 2008 and 2020.',
         keywords='DCC, CCC, two-step estimation, GARCH, dynamic correlation, likelihood ratio, financial crisis, COVID-19, S&P 500, DAX',
         consts=CONSTS, funcs=CORE + [g.dcc_sp_dax, g.fig_step1, g.fig_dcc_sp_dax, g.fig_crisis_zoom],
         run='print(fig_step1())\nprint(fig_dcc_sp_dax())\nprint(fig_crisis_zoom())',
         charts=['tsa_ch14_garch_step1', 'tsa_ch14_dcc_sp_dax', 'tsa_ch14_crisis_zoom']),
    dict(name='TSA_ch14_dcc_panel',
         desc='A five-asset DCC model (S&P 500, DAX, BET, EUR/RON, Bitcoin) on daily returns on common trading days since '
              '2015: dynamic correlations of four pairs and the average correlation matrices of a calm period (2017-2019) '
              'and of the COVID-19 crash (February-June 2020); daily against weekly correlations.',
         keywords='DCC, correlation matrix, contagion, Romania, BET, EUR/RON, Bitcoin, DAX, S&P 500, COVID-19',
         consts=CONSTS, funcs=CORE + [g.dcc_panel, g.fig_panel_corr, g.fig_panel_heatmap],
         run='print(fig_panel_corr())\nprint(fig_panel_heatmap())',
         charts=['tsa_ch14_panel_corr', 'tsa_ch14_panel_heatmap']),
    dict(name='TSA_ch14_portfolio_var',
         desc='VaR 1% of a 50/50 S&P 500 and DAX portfolio from DCC (Normal and filtered historical simulation), CCC and a '
              'static covariance matrix; parameters estimated on 2000-2014, filters run on 2015-2026; Kupiec (1995) '
              'unconditional coverage and Christoffersen (1998) independence tests.',
         keywords='value at risk, VaR 1%, DCC, filtered historical simulation, backtesting, Kupiec test, Christoffersen test, portfolio',
         consts=CONSTS, funcs=CORE + [g.kupiec, g.christoffersen, g.portfolio_var, g.fig_var_backtest],
         run='print(fig_var_backtest())', charts=['tsa_ch14_var_backtest']),
    dict(name='TSA_ch14_hedging',
         desc='Minimum-variance hedge ratios of the BET with the DAX on daily returns since 2005: DCC, rolling OLS (250 days) '
              'and static OLS, estimated on 2005-2014 and evaluated out of sample on 2015-2026 (hedging effectiveness).',
         keywords='hedge ratio, minimum variance hedge, hedging effectiveness, DCC, OLS, BET, DAX, out of sample',
         consts=CONSTS, funcs=CORE + [g.hedge, g.fig_hedge], run='print(fig_hedge())', charts=['tsa_ch14_hedge']),
]

if __name__ == '__main__':
    build_all(QUANTLETS, 14, 'Multivariate GARCH models', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
