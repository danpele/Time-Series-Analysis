"""
build_quantlets.py -- Quantlet folders of Chapter 11 (TSA): foundation models for time series
============================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_11/generate_all_charts.py
      python3 Quantlets/Ch_11/build_quantlets.py
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
DATA = ('Hourly electricity load of Romania (ENTSO-E, extract of Chapter 4); Romanian industrial production, real GDP, '
        'HICP and unemployment (Eurostat sts_inpr_m, namq_10_gdp, prc_hicp_minr, une_rt_m); EUR/RON reference rate of '
        'the BNR; US retail sales (FRED RSXFSN). Open model weights: amazon/chronos-bolt-tiny, amazon/chronos-bolt-small, '
        'amazon/chronos-2 (Hugging Face, Apache-2.0)')
INSTALL = ("# Chronos models need the chronos-forecasting package (it installs PyTorch on a CPU-only machine if missing).\n"
           "# If the installation or the model download fails, the foundation models are skipped and the\n"
           "# statistical benchmarks (seasonal naive, ETS, ARIMA) still run.\n"
           "import importlib.util, subprocess, sys\n"
           "if importlib.util.find_spec('chronos') is None:\n"
           "    try:\n"
           "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'chronos-forecasting'])\n"
           "    except Exception as e:\n"
           "        print('chronos-forecasting could not be installed:', e)")
CONSTS = ['import time', 'from statsmodels.tsa.exponential_smoothing.ets import ETSModel',
          'from statsmodels.tsa.statespace.sarimax import SARIMAX',
          f'SEED = {g.SEED!r}', f'LEVELS = {g.LEVELS!r}', f'FM = {g.FM!r}', f'RELEASE = {g.RELEASE!r}',
          f'MODELS = {g.MODELS!r}', f'LOAD_FILE = {g.LOAD_FILE!r}', f'LOAD_RAW = {g.LOAD_RAW!r}',
          f'EVAL_SERIES = {g.EVAL_SERIES!r}', f'SHORT = {g.SHORT!r}', f'CONTEXTS = {g.CONTEXTS!r}', '_PIPES = {}',
          "COLORS = {'Seasonal naive': st.Amber, 'ETS': st.Forest, 'ARIMA': st.Purple, 'Chronos-Bolt tiny': st.Teal, "
          "'Chronos-Bolt small': st.MainBlue, 'Chronos-2': st.IDAred, 'actual': st.DarkText}",
          'HERE = "."']
CORE = [g.save, g.ro_load, g.get_series, g.chronos_pipeline, g.fm_quantiles, g.n_parameters, g.normal_q, g.snaive_q,
        g.ets_q, g.select_arima, g.arima_q, g.pinball, g.wql_parts, g.mase, g.crps_normal, g.origins_of, g.backtest,
        g.summarise, g.run_all, g.load_results]

QUANTLETS = [
    dict(name='TSA_ch11_tokens_scores',
         desc='The ideas behind time-series foundation models on real data: the seven evaluation series; mean scaling and '
              'quantisation of Chronos (tokens) on Romanian industrial production; patching of the hourly load of Romania '
              'into patches of 16 values; the pinball loss and the CRPS; the worked examples of the slides (scaled values '
              'and token numbers, attention weights, pinball losses, the Normal CRPS).',
         keywords='foundation model, Chronos, tokenisation, mean scaling, quantisation, patching, pinball loss, CRPS, attention',
         consts=CONSTS, funcs=CORE + [g.worked_examples, g.fig_series, g.fig_tokens, g.fig_patching, g.fig_scores],
         run='print(worked_examples())\nprint(fig_series())\nprint(fig_tokens())\nprint(fig_patching())\nprint(fig_scores())',
         charts=['tsa_ch11_series', 'tsa_ch11_tokens', 'tsa_ch11_patching', 'tsa_ch11_scores']),
    dict(name='TSA_ch11_zero_shot',
         desc='Zero-shot probabilistic forecasts of Chronos-Bolt (small), an open model run on a CPU without any training on '
              'the target series: 48 hours of Romanian electricity load against the seasonal naive forecast, 12 months of '
              'Romanian industrial production against ETS, 20 days of EUR/RON against the random walk; fan charts with the '
              '50% and 80% central intervals.',
         keywords='zero-shot forecasting, Chronos-Bolt, fan chart, quantile forecast, electricity load, Romania, EUR/RON, ETS',
         consts=CONSTS, funcs=CORE + [g.fan, g.fig_fan_load, g.fig_fan_monthly, g.fig_fan_eurron],
         run='print(fig_fan_load())\nprint(fig_fan_monthly())\nprint(fig_fan_eurron())',
         charts=['tsa_ch11_fan_load', 'tsa_ch11_fan_ip', 'tsa_ch11_fan_eurron']),
    dict(name='TSA_ch11_benchmark',
         desc='Rolling-origin evaluation of Chronos-Bolt (tiny, small) and Chronos-2 against the seasonal naive forecast, '
              'ETS and ARIMA on seven series (Romanian load, industrial production, real GDP, inflation, unemployment; '
              'EUR/RON; US retail sales): MASE, weighted quantile loss (a CRPS approximation), coverage of the 80% interval, '
              'error by horizon. The full run takes several minutes on a CPU; set max_origins to go faster.',
         keywords='forecast evaluation, rolling origin, MASE, CRPS, weighted quantile loss, coverage, seasonal naive, ETS, ARIMA, Chronos',
         consts=CONSTS, funcs=CORE + [g.table_results, g.bars, g.fig_benchmark, g.fig_coverage, g.fig_horizon],
         run='allr, infos = run_all(max_origins=None)\nR = fig_benchmark(allr)\nprint(R["_gm"])\nprint(fig_coverage(allr))\n'
             'print(fig_horizon(allr))',
         charts=['tsa_ch11_benchmark', 'tsa_ch11_coverage', 'tsa_ch11_horizon'], extra=['ch11_backtest.csv']),
    dict(name='TSA_ch11_context_contamination',
         desc='Three checks of a zero-shot forecaster: accuracy on Romanian hourly load against the context length (96 to '
              '2048 hours); relative weighted quantile loss for forecast origins before and after the release of the '
              'Chronos-Bolt weights (25 November 2024), a contamination check; accuracy against the number of parameters.',
         keywords='context length, data contamination, leakage, release date, model size, Chronos-Bolt, Chronos-2',
         consts=CONSTS, funcs=CORE + [g.table_results, g.fig_context, g.fig_prepost, g.fig_size],
         run='print(fig_context())\nallr = load_results()\nprint(fig_prepost(allr))\nprint(fig_size(allr))',
         charts=['tsa_ch11_context', 'tsa_ch11_prepost', 'tsa_ch11_size'], extra=['ch11_backtest.csv']),
]

if __name__ == '__main__':
    build_all(QUANTLETS, 11, 'Foundation models for time series', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
