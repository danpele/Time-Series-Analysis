"""
build_quantlets.py -- Quantlet folders of Chapter 0 (TSA): Introduction, components and exponential smoothing
=============================================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Lecture Quantlets take their code from generate_all_charts.py; the seminar Quantlet (TSA_ch0_seminar) from
seminar0.py (instructor file). After building, the seminar folder holds the full version: run
    python3 notebooks/split_seminar_notebooks.py 0
to replace it with the student version (the full copy goes to ../instructor/Quantlets/Ch_00).
Run:  python3 Quantlets/Ch_00/generate_all_charts.py && python3 Quantlets/Ch_00/seminar0.py
      python3 Quantlets/Ch_00/build_quantlets.py && python3 notebooks/add_colab_banner.py
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
DATA = ('Eurostat (Romanian real GDP namq_10_gdp, HICP prc_hicp_minr, electricity generation nrg_cb_pem), '
        'BNR reference rate EUR/RON, daily market data from EODHD (BET, S&P 500), statsmodels CO2 data set')
CONSTS = [f'GDP_NSA = {g.GDP_NSA!r}', f'GDP_SCA = {g.GDP_SCA!r}', f'HICP = {g.HICP!r}', f'ELEC = {g.ELEC!r}',
          f'INFL_START = {g.INFL_START!r}', f'TEST_H = {g.TEST_H!r}', f'SEASON = {g.SEASON!r}',
          f'SERIES_LABEL = {g.SERIES_LABEL!r}']
SERIES = [g.gdp_series, g.hicp_series, g.electricity_series, g.co2_series]

QUANTLETS = [
    dict(name='TSA_ch0_examples',
         desc='Six time series from economics, finance, energy and climate: Romanian real GDP (quarterly, unadjusted and '
              'seasonally adjusted, Eurostat), the Romanian HICP and its annual inflation rate (Eurostat), the EUR/RON '
              'reference rate of the BNR, the BET and S&P 500 indices since 2000 (log scale), net electricity generation in '
              'Romania (monthly, Eurostat) and atmospheric CO2 at Mauna Loa (statsmodels data set).',
         keywords='time series, GDP, inflation, HICP, exchange rate, EUR/RON, BET, S&P 500, electricity, CO2, seasonality, trend',
         consts=CONSTS, funcs=SERIES + [g.fig_gdp, g.fig_hicp, g.fig_eurron, g.fig_markets, g.fig_electricity, g.fig_co2],
         run="gdp = fig_gdp()\nhicp = fig_hicp()\nfx = fig_eurron()\nidx = fig_markets()\nel = fig_electricity()\nco2 = fig_co2()\n"
             "print(gdp.tail(4).round(0))\nprint(hicp.tail(3).round(2))",
         charts=['tsa_ch0_gdp', 'tsa_ch0_hicp', 'tsa_ch0_eurron', 'tsa_ch0_markets', 'tsa_ch0_electricity', 'tsa_ch0_co2']),
    dict(name='TSA_ch0_slutsky',
         desc="Slutsky's experiment (1927, 1937): a moving sum of 10 independent standard Normal shocks produces smooth, "
              'irregular waves that look like business cycles, although no cycle was put into the data.',
         keywords='Slutsky effect, moving average, white noise, business cycle, MA process, simulation',
         funcs=[g.fig_slutsky],
         run="eps, ma, up = fig_slutsky()\nprint('upward zero crossings of the moving sum:', up)",
         charts=['tsa_ch0_slutsky']),
    dict(name='TSA_ch0_decomposition',
         desc='Classical multiplicative decomposition (2x4 centred moving average, seasonal factors, remainder) of the '
              'Romanian real GDP, and robust STL decomposition (period 12) of the monthly CO2 series at Mauna Loa.',
         keywords='decomposition, trend, seasonality, seasonal factors, moving average, STL, LOESS, GDP, CO2',
         consts=CONSTS, funcs=[g.gdp_series, g.co2_series, g.classical_decomposition, g.fig_components, g.fig_stl],
         run="y, trend, seas, rem, q = fig_components()\nprint('seasonal factors by quarter:', {k: round(v, 3) for k, v in q.items()})\n"
             "c, res = fig_stl()\nprint('remainder standard deviation (ppm):', round(res.resid.std(), 3))",
         charts=['tsa_ch0_components', 'tsa_ch0_stl']),
    dict(name='TSA_ch0_acf',
         desc='From prices to returns (BET index, daily log returns since 2000) and a first look at the sample '
              'autocorrelation function: slow decay for a trending level (CO2), seasonal peaks (electricity generation), '
              'almost no memory (BET daily log returns), with the band +-1.96/sqrt(T).',
         keywords='log returns, autocorrelation, ACF, correlogram, trend, seasonality, BET, CO2, electricity',
         consts=CONSTS, funcs=[g.co2_series, g.electricity_series, g.fig_returns, g.sample_acf, g.fig_acf],
         run="p, r = fig_returns()\nout = fig_acf()\nfor k, (a, band, n) in out.items():\n"
             "    print(f'{k}: r1 = {a[0]:.3f}, r12 = {a[11]:.3f}, band = {band:.3f}, T = {n}')",
         charts=['tsa_ch0_returns', 'tsa_ch0_acf']),
    dict(name='TSA_ch0_smoothing',
         desc='Simple exponential smoothing of the monthly average EUR/RON reference rate (BNR) since January 2015 with '
              'smoothing constants 0.1 and 0.7, and the estimated smoothing constant (close to 1: the naive forecast).',
         keywords='exponential smoothing, SES, smoothing constant, naive forecast, EUR/RON, BNR',
         funcs=[g.ses_path, g.eurron_monthly, g.fig_ses],
         run="y, alpha = fig_ses()\nprint('estimated alpha:', round(alpha, 3))\nprint('one-step SES forecasts, last 3 months:')\n"
             "print(ses_path(y, 0.1).tail(3).round(4), ses_path(y, 0.7).tail(3).round(4))",
         charts=['tsa_ch0_ses']),
    dict(name='TSA_ch0_forecast',
         desc='Forecast evaluation on a test set: naive, seasonal naive, simple exponential smoothing and Holt-Winters '
              'forecasts of Romanian electricity generation for the last 24 months (MAE, RMSE, MAPE, MASE), and a small '
              'M-competition on four seasonal series (Romanian GDP, HICP, electricity generation, CO2).',
         keywords='forecast evaluation, naive forecast, seasonal naive, Holt-Winters, MAE, RMSE, MAPE, MASE, M4 competition',
         consts=CONSTS, funcs=SERIES + [g.method_forecasts, g.accuracy, g.fig_forecast, g.benchmark_data, g.fig_benchmarks],
         run="train, test, f, acc = fig_forecast()\nprint(pd.DataFrame(acc).T.round(3))\n"
             "tab, piv = fig_benchmarks()\nprint(piv.round(2))",
         charts=['tsa_ch0_forecast', 'tsa_ch0_benchmarks'], extra=['ch0_accuracy.csv']),
]

try:                                            # instructor file (git-ignored): seminar computations
    import seminar0 as s0
    QUANTLETS.append(dict(
        name='TSA_ch0_seminar',
        desc='Seminar 0 of Time Series Analysis: growth rates of a quarterly series, a sample autocorrelation by hand, '
             'naive and seasonal naive forecasts with MAE and RMSE; on real data: Romanian real GDP (Eurostat), the EUR/RON '
             'reference rate (BNR), the Romanian HICP (Eurostat) and electricity generation in Romania (Eurostat).',
        keywords='growth rate, year on year, autocorrelation, ACF, naive forecast, seasonal naive, MAE, RMSE, seminar',
        consts=[f'GDP_NSA = {s0.GDP_NSA!r}', f'HICP = {s0.HICP!r}', f'ELEC = {s0.ELEC!r}', f'START = {s0.START!r}'],
        funcs=[s0.growth_rates, s0.sample_acf, s0.paper_forecasts, s0.gdp_growth, s0.fig_gdp_growth, s0.eurron_acf,
               s0.fig_acf_pair],
        run="d = gdp_growth()\nprint(d.tail(4).round(2))\nfig_gdp_growth(d)\n"
            "b2 = eurron_acf()\nprint('EUR/RON: r1 level', round(b2['acf_level'][0], 3), '; r1 returns', round(b2['acf_ret'][0], 3))\n"
            "fig_acf_pair(b2['acf_level'], b2['acf_ret'], b2['band_level'], b2['band_ret'], 'EUR/RON level', "
            "'EUR/RON daily log returns', 'ch0_sem_b2_acf')",
        charts=['ch0_sem_b1_growth', 'ch0_sem_b2_acf'],
        data='Eurostat (Romanian real GDP, HICP, electricity generation), BNR reference rate EUR/RON'))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 0, 'Introduction: components and exponential smoothing', HERE, data=DATA, submitted=SUBMITTED)
