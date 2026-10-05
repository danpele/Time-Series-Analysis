"""
build_quantlets.py -- Quantlet folders of Chapter 9 (TSA): machine learning for time series
==========================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_09/generate_all_charts.py && python3 Quantlets/Ch_09/seminar9.py
      python3 Quantlets/Ch_09/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 9
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import generate_all_charts as g                 # noqa: E402
from tsa_quantlets import build_all, module_body   # noqa: E402

SUBMITTED = 'Monday, 5 October 2026'
DATA = ('Hourly electricity load of Romania from ENTSO-E (the extract Quantlets/Ch_04/ch4_ro_load_hourly.csv of the TSA '
        'repository); monthly HICP of the 27 EU countries from Eurostat (prc_hicp_minr); daily S&P 500, DAX and BET '
        'prices from EODHD (data/market of the TSA repository); the official M4 evaluation file '
        '(github.com/Mcompetitions/M4-methods)')
INSTALL = ("import os\nos.environ['OMP_NUM_THREADS'] = '1'   # small data: one thread per model fit is faster\n"
           "# PyTorch is preinstalled in Google Colab; it is needed only for the LSTM")
CONSTS = ['import math',
          'from matplotlib.patches import Rectangle, Circle, FancyArrowPatch',
          'from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, '
          'RandomForestClassifier, RandomForestRegressor)',
          'from sklearn.inspection import permutation_importance',
          'from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, RidgeCV, lasso_path',
          'from sklearn.metrics import roc_auc_score',
          'from sklearn.model_selection import KFold, TimeSeriesSplit',
          'from sklearn.neural_network import MLPRegressor',
          'from sklearn.pipeline import make_pipeline',
          'from sklearn.preprocessing import StandardScaler',
          'from sklearn.tree import DecisionTreeRegressor, plot_tree',
          'from dateutil.easter import easter, EASTER_ORTHODOX',
          'HERE = "."',
          module_body(g, 'SEED = 2026', '\n\n\n# ===='),
          f'FEATS = {g.FEATS!r}', f'INF_FEATS = {g.INF_FEATS!r}', f'M4_FILE = {g.M4_FILE!r}',
          "HIGHLIGHT = {'lag0': st.MainBlue, 'same_dow': st.IDAred, 'mean7': st.Forest, 'sat': st.Orange, "
          "'holiday': st.Purple, 'easter': st.Amber, 'xmas': st.Teal}",
          "M4_TYPES = {'Statistical': st.MainBlue, 'Combination (S)': st.Teal, 'Combination (S & ML)': st.Purple, "
          "'Combination (ML)': st.Orange, 'Machine Learning': st.IDAred, 'Hybrid': st.Forest, 'Other': st.Amber}",
          f'GKX_MODELS = {g.GKX_MODELS!r}', f'GKX_R2 = {g.GKX_R2!r}', f'GKX_SR = {g.GKX_SR!r}']
DATAF = [g.ro_holidays, g.load_daily, g.calendar, g.direct_frame, g.origins, g.oos_r2, g.dm_raw]
MODELS = [g.make_ridge, g.make_rf, g.make_gb, g.direct_models, g.forecast_direct, g.forecast_recursive, g.mimo_inputs,
          g.MLPMulti, g.forecast_mlp, g.forecast_lstm]

QUANTLETS = [
    dict(name='TSA_ch9_supervised_learning',
         desc='Forecasting as supervised learning: the daily electricity load of Romania (ENTSO-E, 2022-2026) turned into a '
              'table of lags, rolling means, the same weekday of last week and calendar features of the target day '
              '(weekday, annual Fourier terms, Orthodox Easter, Christmas, public holidays); scatter plots of the target '
              'against three features; a diagram of the recursive, direct and multi-output (MIMO) strategies.',
         keywords='machine learning, supervised learning, lag features, rolling window, calendar features, recursive '
                  'forecast, direct forecast, MIMO, electricity load, ENTSO-E, Romania',
         consts=CONSTS, funcs=DATAF + [g.mini_table, g.fig_features, g.fig_strategies],
         run='print(mini_table())\nprint(fig_features())\nfig_strategies()',
         charts=['tsa_ch9_features', 'tsa_ch9_strategies']),
    dict(name='TSA_ch9_validation',
         desc='Validation without leakage: random K-fold, walk-forward with expanding and rolling windows and with a gap; '
              'a random forest for the sum of the next 21 daily returns of a simulated random walk and of the S&P 500 '
              'under random 5-fold, walk-forward and gapped walk-forward validation; bias and variance of regression '
              'trees of depth 1 to 10 (simulation).',
         keywords='walk-forward validation, time-series cross-validation, leakage, look-ahead bias, overlapping targets, '
                  'bias-variance trade-off, regression tree, random forest, S&P 500',
         consts=CONSTS, funcs=DATAF + [g.fig_cv_schemes, g.leakage_frame, g.cv_r2, g.leakage_experiment, g.fig_leakage,
                                       g.true_f, g.bias_variance, g.fig_bias_variance],
         run='fig_cv_schemes()\nprint(fig_leakage())\nprint(fig_bias_variance())',
         charts=['tsa_ch9_cv_schemes', 'tsa_ch9_leakage', 'tsa_ch9_bias_variance']),
    dict(name='TSA_ch9_regularisation_trees',
         desc='Ridge and lasso coefficient paths on the 30 load features; a depth-2 regression tree for tomorrow\'s load; '
              'a single tree, a random forest and histogram gradient boosting (two learning rates) on validation data; '
              'trees cannot extrapolate: random forests on the level and on the monthly changes of the Romanian HICP '
              'index after 2020.',
         keywords='ridge, lasso, regularisation, regression tree, random forest, gradient boosting, HistGradientBoosting, '
                  'learning rate, extrapolation, HICP, Romania, electricity load',
         consts=CONSTS, funcs=DATAF + [g.shrinkage_paths, g.fig_shrinkage, g.fig_tree, g.ensemble_curves,
                                       g.fig_ensembles, g.hicp, g.fig_extrapolation],
         run='print(fig_shrinkage())\nprint(fig_tree())\nprint(fig_ensembles())\nprint(fig_extrapolation())',
         charts=['tsa_ch9_shrinkage', 'tsa_ch9_tree', 'tsa_ch9_ensembles', 'tsa_ch9_extrapolation']),
    dict(name='TSA_ch9_neural_networks',
         desc='A multilayer perceptron and its activation functions; a recurrent network unrolled in time and the LSTM '
              'cell; a small multi-output MLP and a small LSTM (PyTorch, CPU, fixed seed) giving 14-day forecasts of the '
              'daily load of Romania from one forecast origin.',
         keywords='neural network, multilayer perceptron, activation function, recurrent neural network, LSTM, PyTorch, '
                  'multi-output forecast, electricity load',
         consts=CONSTS, funcs=DATAF + MODELS + [g.fig_mlp, g.fig_rnn],
         run="fig_mlp()\nfig_rnn()\ny = load_daily()\no = pd.Timestamp(CV_FIRST)\n"
             "f, losses = forecast_lstm(y, o)\nprint('LSTM training loss, first and last epoch:', losses[0], losses[-1])\n"
             "print(pd.DataFrame({'actual': y.loc[o + pd.Timedelta(days=1):].iloc[:CV_H].values, 'LSTM': f, "
             "'MLP': forecast_mlp(y, o)}).round(3))",
         charts=['tsa_ch9_mlp', 'tsa_ch9_rnn']),
    dict(name='TSA_ch9_load_forecasting',
         desc='Electricity load of Romania, the 26 forecast origins and the 14-day horizon of Chapter 4: seasonal naive, '
              'ETS, SARIMA, DHR and their combination (errors of Chapter 4) against ridge, random forest and gradient '
              'boosting (direct strategy), gradient boosting (recursive), an MLP and an LSTM (multi-output); MAE, RMSE, '
              'MASE and Diebold-Mariano tests; permutation importance; 90% intervals by quantile boosting and split '
              'conformal prediction.',
         keywords='electricity load, forecasting competition, walk-forward, direct strategy, recursive strategy, gradient '
                  'boosting, random forest, ridge, MLP, LSTM, MASE, Diebold-Mariano, quantile regression, pinball loss, '
                  'conformal prediction, permutation importance',
         consts=CONSTS, funcs=DATAF + MODELS + [g.load_ml_cv, g.ch9_file, g.saved_load_ml, g.ch4_errors, g.load_summary,
                                                g.fig_load_results, g.fig_load_forecasts, g.importance, g.fig_importance,
                                                g.load_intervals, g.fig_intervals],
         run="y = load_daily()\n# the ML models of the slides (about 10 minutes in Colab); saved copy: saved_load_ml()\n"
             "E_ml = load_ml_cv(y, save_csv=False)\nE_st = ch4_errors()\ntab, dm, scale, byh = load_summary(E_ml, E_st, y)\n"
             "print(tab.round(3))\nfig_load_results(tab, byh)\nprint(fig_load_forecasts(E_ml, E_st, y))\n"
             "print(fig_importance())\nR, out = load_intervals(y)\nprint(out)\nfig_intervals(R, out, y)",
         charts=['tsa_ch9_load_results', 'tsa_ch9_load_forecasts', 'tsa_ch9_importance', 'tsa_ch9_intervals'],
         extra=['ch9_load_ml.csv']),
    dict(name='TSA_ch9_inflation_global',
         desc='12-month HICP inflation of Romania (Eurostat), horizons 1, 3, 6 and 12 months, direct strategy on the change '
              'of inflation: random walk, AR, lasso, random forest and gradient boosting fitted on Romania only (local) '
              'and gradient boosting fitted on the 27 EU countries pooled (global); yearly walk-forward 2016-2026, RMSE '
              'and Diebold-Mariano tests.',
         keywords='inflation forecasting, HICP, Romania, European Union, global model, local model, gradient boosting, '
                  'random forest, lasso, random walk, Diebold-Mariano',
         consts=CONSTS, funcs=DATAF + [g.make_gb, g.hicp, g.inflation_panel, g.inflation_frame, g.inflation_models,
                                       g.inflation_forecasts, g.inflation_summary, g.fig_inflation],
         run="E, pi = inflation_forecasts(save=False)\nS = inflation_summary(E)\n"
             "print(pd.DataFrame({h: S[h]['rel'] for h in S}).round(3))\nprint(fig_inflation(E, pi, S))",
         charts=['tsa_ch9_inflation'], extra=['ch9_inflation.csv']),
    dict(name='TSA_ch9_finance',
         desc='Two financial applications. Weekly realised variance (Garman-Klass range proxy) of the S&P 500 and the DAX: '
              'HAR against lasso, random forest, gradient boosting and an MLP ensemble, yearly walk-forward from 2013 with '
              'a 5-day gap, out-of-sample R^2, QLIKE and Diebold-Mariano tests. The sign of the next daily return of the '
              'S&P 500 and the BET: logit, random forest and gradient boosting against the majority-class baseline.',
         keywords='realised volatility, HAR, QLIKE, Garman-Klass, lasso, random forest, gradient boosting, MLP, sign '
                  'prediction, classification, baseline, AUC, S&P 500, DAX, BET',
         consts=CONSTS, funcs=DATAF + [g.make_gb, g.daily_variance, g.rv_frame, g.MeanModel, g.MLPEnsemble, g.rv_models,
                                       g.walk_forward, g.qlike, g.dm_hac, g.rv_forecasts, g.rv_metrics, g.fig_rv,
                                       g.sign_frame, g.sign_models, g.sign_walk_forward, g.sign_metrics, g.fig_sign],
         run="X, P = rv_forecasts('sp500')\nprint(rv_metrics(X, P))\nfig_rv(X, P)\n"
             "S = {k: sign_metrics(*sign_walk_forward(k)) for k in ('sp500', 'bet')}\nprint(S)\nfig_sign(S)",
         charts=['tsa_ch9_rv', 'tsa_ch9_sign']),
    dict(name='TSA_ch9_case_studies',
         desc='Landmark case studies: the M4 competition, OWA of the 59 ranked methods from the official evaluation file '
              '(Makridakis, Spiliotis and Assimakopoulos, 2020), coloured by type; Gu, Kelly and Xiu (2020), monthly '
              'out-of-sample R^2 and Sharpe ratios of twelve methods (published numbers, Tables 1 and 7).',
         keywords='M4 competition, OWA, sMAPE, MASE, hybrid model, ES-RNN, combination, machine learning, Gu Kelly Xiu, '
                  'empirical asset pricing, neural networks',
         consts=CONSTS, funcs=[g.ch9_file, g.m4_table, g.fig_m4, g.fig_gkx],
         run='print(fig_m4())\nfig_gkx()', charts=['tsa_ch9_m4', 'tsa_ch9_gkx'], extra=['ch9_m4_owa.csv'],
         data='The official M4 evaluation file (github.com/Mcompetitions/M4-methods); numbers published by Gu, Kelly and '
              'Xiu (2020)'),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar9 as s
    QUANTLETS.append(dict(
        name='TSA_ch9_seminar',
        desc='Seminar 9 of Time Series Analysis: a lag table with recursive and direct forecasts, leakage in a rolling '
             'mean, one split of a regression tree, ridge and lasso in one dimension, a split conformal interval and the '
             'pinball loss, MASE and a Diebold-Mariano statistic; gradient boosting for the Romanian electricity load '
             'against the seasonal naive forecast; recursive against direct and the value of holiday features; Romanian '
             'inflation three months ahead with local and global models; the sign of BET returns.',
        keywords='seminar, machine learning, lag features, walk-forward, leakage, regression tree, ridge, lasso, '
                 'conformal prediction, gradient boosting, electricity load, inflation, BET',
        consts=CONSTS + [f'SEM_FIRST, SEM_LAST, SEM_H = {s.SEM_FIRST!r}, {s.SEM_LAST!r}, {s.SEM_H!r}',
                         "NO_CAL = [f for f in FEATS if f not in ('easter', 'xmas', 'holiday')]"],
        funcs=DATAF + MODELS + [g.hicp, g.inflation_panel, g.inflation_frame, g.inflation_models, g.inflation_forecasts,
                                g.sign_frame, g.sign_models, g.sign_metrics, s.lag_table, s.recursive_direct,
                                s.rolling_leak, s.best_split, s.ridge_lasso_1d, s.conformal_pinball, s.mase_dm,
                                g.calendar, s.sem_origins, s.b1_load, s.b3_inflation],
        run="print(lag_table([10, 12, 11, 13, 14, 13, 15, 16]))\nprint(recursive_direct([10, 12, 11, 13, 14, 13, 15, 16]))\n"
            "print(best_split([4.8, 5.0, 5.6, 6.0, 6.4, 6.9], [5.0, 5.2, 5.9, 6.0, 6.6, 7.0])['best'])\n"
            "print(conformal_pinball([0.12, 0.30, 0.05, 0.22, 0.41, 0.18, 0.09, 0.27, 0.35], 6.10))\n"
            "print(b1_load())\nprint(b3_inflation())",
        charts=['ch9_sem_b1', 'ch9_sem_b3']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 9, 'Machine learning for time series', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
