"""
build_quantlets.py -- Quantlet folders of Chapter 7 (TSA): cointegration and VECM
=================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_07/generate_all_charts.py && python3 Quantlets/Ch_07/seminar7.py
      python3 Quantlets/Ch_07/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 7
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
DATA = ('US Treasury yields (GS1, GS5, GS10, TB3MS) from FRED; US macro data (statsmodels macrodata); BNR reference rates '
        '(EUR, HUF, PLN); Romanian and euro-area HICP, ROBOR 3M, Euribor 3M, Romanian consumption and GDP from Eurostat; '
        'daily prices of Bucharest Stock Exchange shares and US banks from EODHD (data/market of the TSA repository)')
NODATA = 'Simulated data only (no market data)'
INSTALL = ("# The arch package (Phillips-Ouliaris test) is not preinstalled in Google Colab: pip install arch\n"
           "import importlib.util, subprocess, sys\n"
           "if importlib.util.find_spec('arch') is None:\n"
           "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'arch'])")
CONSTS = ['import itertools', 'import statsmodels.api as sm',
          'from statsmodels.tsa.stattools import adfuller, coint, acf',
          'from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp',
          'from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order',
          'from statsmodels.tsa.api import VAR', 'from arch.unitroot.cointegration import phillips_ouliaris',
          f'SEED = {g.SEED!r}', f'YIELDS = {g.YIELDS!r}', f'YIELD_START = {g.YIELD_START!r}', f'HICP_RO = {g.HICP_RO!r}',
          f'HICP_EA = {g.HICP_EA!r}', f'ROBOR = {g.ROBOR!r}', f'EURIBOR = {g.EURIBOR!r}', f'CONS_RO = {g.CONS_RO!r}',
          f'GDP_RO = {g.GDP_RO!r}', f'BVB = {g.BVB!r}', f'US_BANKS = {g.US_BANKS!r}', f'PAIR = {g.PAIR!r}',
          f'PAIR_START = {g.PAIR_START!r}', f'FORM, TRADE, ENTRY = {g.FORM!r}, {g.TRADE!r}, {g.ENTRY!r}',
          f'COST_BVB, COST_US = {g.COST_BVB!r}, {g.COST_US!r}']
CORE = [g.us_yields, g.us_macro, g.cee_fx, g.stock, g.stock_panel, g.save, g.years_axis, g.ts_index, g.adf_test, g.ols,
        g.eg_test, g.half_life, g.johansen, g.ecm_fit]

QUANTLETS = [
    dict(name='TSA_ch7_common_trends',
         desc='Cointegration as a common stochastic trend: the drunk and her dog of Murray (1994) by simulation (two error-'
              'correcting random walks and a stray, independent one, with the distances between them); four real pairs of '
              'I(1) series: US 3-month and 10-year Treasury yields (FRED), US real consumption and GDP (statsmodels macrodata), '
              'Banca Transilvania and BRD (BVB), EUR/RON and EUR/HUF (BNR), with Engle-Granger tests.',
         keywords='cointegration, common stochastic trend, random walk, error correction, drunk and her dog, interest rates, '
                  'consumption, income, Banca Transilvania, BRD, EUR/RON, EUR/HUF',
         consts=CONSTS, funcs=CORE + [g.fig_drunk_dog, g.fig_examples],
         run='print(fig_drunk_dog())\nprint(fig_examples())', charts=['tsa_ch7_drunk_dog', 'tsa_ch7_examples']),
    dict(name='TSA_ch7_engle_granger',
         desc='The Engle-Granger two-step method: the null distribution of the residual-based Dickey-Fuller statistic for 2 '
              'and 3 variables by simulation, against the Dickey-Fuller distribution and the MacKinnon (2010) critical '
              'values, with the size of the test when the Dickey-Fuller value is used; Engle-Granger and Phillips-Ouliaris '
              'tests on US consumption and income, US yields, two Romanian banks, EUR/RON and EUR/HUF and Romanian '
              'consumption and GDP; the step-1 residuals and their ACF.',
         keywords='Engle-Granger, Phillips-Ouliaris, cointegration test, residual-based test, MacKinnon critical values, '
                  'Monte Carlo, super-consistency, normalisation',
         consts=CONSTS, funcs=CORE + [g.eg_tau_batch, g.fig_eg_dist, g.eg_table, g.fig_eg_steps],
         run='print(fig_eg_dist())\nE = eg_table()\nprint(pd.DataFrame({k: {"beta": v["beta"][0], "tau": v["tau"], "p": v["p"], '
             '"PO Zt": v["po_zt"], "PO p": v["po_p"]} for k, v in E.items() if isinstance(v, dict)}).T)\nprint(fig_eg_steps())',
         charts=['tsa_ch7_eg_dist', 'tsa_ch7_eg_steps'], extra=['ch7_eg_tests.csv']),
    dict(name='TSA_ch7_error_correction',
         desc='Error correction models: the decay of an equilibrium error and its half-life ln(0.5)/ln(1 + gamma); single-'
              'equation ECMs for US consumption on income (quarterly) and for the 10-year on the 3-month Treasury yield '
              '(monthly), with the equation of the 3-month rate.',
         keywords='error correction model, ECM, speed of adjustment, half-life, consumption function, term structure',
         consts=CONSTS, funcs=CORE + [g.fig_ecm], run='print(fig_ecm())', charts=['tsa_ch7_ecm']),
    dict(name='TSA_ch7_johansen_vecm',
         desc='Johansen trace and maximum-eigenvalue tests (MacKinnon-Haug-Michelis critical values) for US Treasury yields '
              'at 1, 5 and 10 years, for US output, consumption and investment, and for EUR/RON, EUR/HUF and EUR/PLN, with '
              'three deterministic cases for the yields; the term-structure VECM with rank 2 and a restricted constant: '
              'cointegrating vectors, adjustment coefficients with t-statistics, half-lives of the equilibrium errors, '
              'orthogonalised impulse responses.',
         keywords='Johansen test, trace test, maximum eigenvalue test, VECM, cointegrating vector, adjustment coefficients, '
                  'weak exogeneity, term structure, impulse response',
         consts=CONSTS, funcs=CORE + [g.johansen_systems, g.vecm_rates, g.fig_rates, g.fig_vecm_irf],
         run='J = johansen_systems()\nprint({k: (v["trace"], v["rank_trace"]) for k, v in J.items() if k != "yields_det"})\n'
             'print(vecm_rates())\nprint(fig_rates())\nprint(fig_vecm_irf())',
         charts=['tsa_ch7_rates', 'tsa_ch7_vecm_irf'], extra=['ch7_johansen.csv']),
    dict(name='TSA_ch7_forecasting',
         desc='Forecasting US Treasury yields at 1, 5 and 10 years out of sample: a rank-2 VECM against a VAR in differences '
              'and the random walk, with re-estimation at origins every three months since 1990; RMSE ratios by horizon for '
              'the levels and for the 10-year minus 1-year spread (Christoffersen and Diebold 1998).',
         keywords='forecasting, VECM, VAR in differences, random walk, RMSE, long-horizon forecasting, interest rates',
         consts=CONSTS, funcs=CORE + [g.vecm_rates, g.forecast_compare], run='print(forecast_compare())',
         charts=['tsa_ch7_forecast']),
    dict(name='TSA_ch7_parity_conditions',
         desc='Economic examples: purchasing power parity for EUR/RON with Romanian and euro-area HICP (ADF on the real '
              'exchange rate, Engle-Granger, Johansen); ROBOR 3M and Euribor 3M since 1995 and since 2010 (Engle-Granger, '
              'Phillips-Ouliaris, ECM); EUR/RON, EUR/HUF and EUR/PLN (Johansen).',
         keywords='purchasing power parity, real exchange rate, Balassa-Samuelson, ROBOR, Euribor, interest rates, EUR/RON, '
                  'EUR/HUF, EUR/PLN, Johansen, Romania',
         consts=CONSTS, funcs=CORE + [g.fig_ppp, g.fig_ro_ea_rates, g.fig_cee_fx],
         run='print(fig_ppp())\nprint(fig_ro_ea_rates())\nprint(fig_cee_fx())',
         charts=['tsa_ch7_ppp', 'tsa_ch7_ro_ea_rates', 'tsa_ch7_cee_fx']),
    dict(name='TSA_ch7_pairs_trading',
         desc='Pairs trading with cointegration: Banca Transilvania and BRD since 2014 (full-sample spread, half-life); a '
              'rolling rule without look-ahead (252-day formation with Engle-Granger selection, 126-day trading, entry at '
              '|z| > 2, exit at 0) on 8 liquid BVB shares and on 6 US banks, gross and net of transaction costs, with the '
              'break-even cost and the in-sample illusion of full-sample parameters.',
         keywords='pairs trading, statistical arbitrage, cointegration, backtest, transaction costs, look-ahead bias, '
                  'Bucharest Stock Exchange, US banks',
         consts=CONSTS, funcs=CORE + [g.pairs_backtest, g.perf, g.fig_pair, g.fig_pairs_backtest],
         run='print(fig_pair())\nprint(fig_pairs_backtest())', charts=['tsa_ch7_pair', 'tsa_ch7_pairs_backtest']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar7 as s
    QUANTLETS.append(dict(
        name='TSA_ch7_seminar',
        desc='Seminar 7 of Time Series Analysis: Engle-Granger statistics by hand with two and three variables, an ECM and '
             'its half-life, a bivariate VECM, Johansen statistics from eigenvalues, a VAR(2) written as a VECM; Engle-'
             'Granger, Phillips-Ouliaris and ECM for the US 1-year and 10-year yields; Romanian consumption and GDP; '
             'Johansen for EUR/RON, EUR/HUF and EUR/PLN; a VECM for ROBOR, the Romanian 10-year yield and Euribor; pairs '
             'trading on Banca Transilvania and BRD.',
        keywords='seminar, cointegration, Engle-Granger, Phillips-Ouliaris, ECM, VECM, Johansen, interest rates, Romania, '
                 'pairs trading',
        consts=CONSTS, funcs=CORE + [s.eg_by_hand, s.ecm_by_hand, s.johansen_by_hand, s.b1_rates, s.b3_cee_fx],
        run="print(eg_by_hand(-0.118, 0.036, 200, 2))\nprint(johansen_by_hand([0.120, 0.045, 0.008], 200))\n"
            "print(b1_rates(save=True))\nprint(b3_cee_fx(save=True))",
        charts=['ch7_sem_b1', 'ch7_sem_b3']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 7, 'Cointegration and VECM', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
