"""
build_quantlets.py -- Quantlet folders of Chapter 15 (TSA): review and exam preparation
======================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_15/generate_all_charts.py && python3 Quantlets/Ch_15/seminar15.py
      python3 Quantlets/Ch_15/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 15
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
DATA = ('Romanian HICP and ROBOR 3M (Eurostat prc_hicp_minr, irt_st_m); daily BET closes from EODHD (data/market of the '
        'TSA repository)')
INSTALL = ("# The arch package (GARCH estimation) is not preinstalled in Google Colab: pip install arch\n"
           "import importlib.util, subprocess, sys\n"
           "if importlib.util.find_spec('arch') is None:\n"
           "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'arch'])")
CONSTS = ['import statsmodels.api as sm',
          'from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf',
          'from statsmodels.tsa.statespace.sarimax import SARIMAX',
          f'HICP = {g.HICP!r}', f'ROBOR = {g.ROBOR!r}', f'START = {g.START!r}', f'TRAIN_END = {g.TRAIN_END!r}',
          f'S = {g.S!r}', f'MAXLAG = {g.MAXLAG!r}', f'IDENT = {g.IDENT!r}', f'H = {g.H!r}']
CORE = [g.hicp, g.transforms, g.save, g.adf_kpss, g.ljung_box, g.dm_test, g.fit_sarima, g.label, g.model_summary]

QUANTLETS = [
    dict(name='TSA_ch15_box_jenkins',
         desc='The Box-Jenkins method from start to finish on the monthly HICP of Romania (Eurostat), 2005-2026: plots of '
              'the index, monthly and annual inflation; ADF and KPSS on ln P, its differences and z = d d12 ln P; the ACF '
              'and PACF of z; every SARIMA(p,1,q)(P,1,Q)12 with p, q <= 2 and P, Q <= 1 by AICc, BIC and Ljung-Box; '
              'residual diagnostics of the chosen model; one-step forecasts on a test sample against the seasonal naive '
              'and naive methods (RMSE, MAE, MASE, Diebold-Mariano with the HLN correction); 12-month forecasts of annual '
              'inflation with 95% intervals; the AR(1) x SAR(1) output and the forecast by hand of exam problem 3.',
         keywords='Box-Jenkins, SARIMA, ADF, KPSS, identification, ACF, PACF, Ljung-Box, Diebold-Mariano, MASE, inflation, HICP, Romania, exam',
         consts=CONSTS, funcs=CORE + [g.fig_bj_data, g.bj_tests, g.fig_bj_acf, g.bj_grid, g.fig_bj_diag, g.fig_bj_forecast,
                                      g.exam_p3],
         run="print(fig_bj_data())\nprint(bj_tests())\nprint(fig_bj_acf())\ngrid = bj_grid()\n"
             "print(pd.DataFrame(grid['rows']).head(8))\nprint(fig_bj_diag(grid))\nprint(fig_bj_forecast(grid))\nprint(exam_p3())",
         charts=['tsa_ch15_bj_data', 'tsa_ch15_bj_acf', 'tsa_ch15_bj_diag', 'tsa_ch15_bj_forecast']),
    dict(name='TSA_ch15_course_summary',
         desc='The course in a few functions, one per block of chapters, on Romanian data: SES and Holt-Winters for the '
              'HICP (Chapter 0), GARCH(1,1)-t for the BET (Chapter 5), a VAR for Romanian annual inflation and ROBOR 3M '
              'with Granger tests (Chapters 6-7), the local Whittle d of the BET returns and absolute returns (Chapter 8), '
              'a local level model of monthly inflation and its equivalent exponential smoothing weight (Chapter 10).',
         keywords='review, exponential smoothing, GARCH, VAR, Granger causality, long memory, local Whittle, local level, Kalman filter, Romania',
         consts=CONSTS, funcs=CORE + [g.course_smoothing, g.course_garch, g.course_var, g.local_whittle, g.course_memory,
                                      g.course_state_space],
         run="print(course_smoothing())\nprint(course_garch())\nprint(course_var())\nprint(course_memory())\n"
             "print(course_state_space())"),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar15 as s
    QUANTLETS.append(dict(
        name='TSA_ch15_seminar',
        desc='Seminar 15 of Time Series Analysis, exam practice: an ARIMA(0,1,1) by hand; ADF, KPSS and the correlogram of '
             'the Romanian unemployment rate; SARIMA output for Romanian GDP and a forecast by hand; VAR output and a '
             'Granger F test for Romanian inflation and ROBOR 3M; GARCH(1,1)-t output for the DAX and VaR 1%; Box-Jenkins '
             'for Romanian industrial production; the volatility of the BET since 2015; cointegration of the Romanian and '
             'German 10-year yields.',
        keywords='seminar, exam, ARIMA, unit roots, SARIMA, VAR, Granger causality, GARCH, VaR, cointegration, Romania',
        consts=CONSTS + [f'UNEMP = {s.UNEMP!r}', f'GDP_NSA = {s.GDP_NSA!r}', f'IP = {s.IP!r}', f'SEM_START = {s.SEM_START!r}',
                         f'IP_TEST = {s.IP_TEST!r}', f'BET_START = {s.BET_START!r}', f'YIELD_START = {s.YIELD_START!r}',
                         'from statsmodels.tsa.stattools import coint'],
        funcs=CORE + [s.quarter, s.a1_arima011, s.a2_unemployment, s.a3_gdp_sarima, s.ro_pi_i, s.a4_var, s.a5_garch,
                      s.b1_ip, s.b2_bet, s.b3_yields, s.c2_check],
        run="print(a1_arima011())\nprint(b1_ip())",
        charts=['ch15_sem_b1']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 15, 'Review and exam preparation', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
