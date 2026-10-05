"""
build_notebooks_ch9.py -- lecture and seminar notebooks of Chapter 9 (TSA): machine learning for time series
===========================================================================================================
Output: notebooks/EN/chapter9_lecture_notebook.ipynb, notebooks/EN/chapter9_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_09/generate_all_charts.py and seminar9.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides. Small models, fixed seeds, one thread per fit: the lecture notebook
runs in about 15 minutes on a laptop CPU, the seminar notebook in about 5 minutes.
Run:  python3 notebooks/build_notebooks_ch9.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter9_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter9_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 9
Time Series Analysis - Daniel Traian PELE
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

QL = chapter_paths(9)
import generate_all_charts as g   # noqa: E402
import seminar9 as s              # noqa: E402
import build_quantlets as bq      # noqa: E402

CONSTS = '\n'.join(bq.CONSTS)
INSTALL = bq.INSTALL
N = json.load(open(os.path.join(QL, 'ch9_numbers.json')))
S = json.load(open(os.path.join(QL, 'sem9_results.json')))
CORE = bq.DATAF + bq.MODELS


def f(x, d=2):
    return f'{x:.{d}f}'


# =============================================================================
# LECTURE
# =============================================================================
LD = N['load']['tab']
LECTURE = [
    md("# Time Series Analysis — Chapter 9: Machine learning for time series\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Forecasting as supervised learning: lags, rolling windows, calendar features; recursive, direct and MIMO strategies.\n"
       "- Walk-forward validation, leakage, the bias–variance trade-off.\n"
       "- Ridge and lasso; regression trees, random forest, gradient boosting; trees cannot extrapolate.\n"
       "- Neural networks: the MLP and a small LSTM (PyTorch, CPU).\n"
       "- Electricity load of Romania: ML against the statistical models of Chapter 4, MASE and Diebold–Mariano; "
       "prediction intervals by quantile boosting and conformal prediction.\n"
       "- Romanian inflation with local and global models; realised volatility (HAR against ML); the sign of returns.\n"
       "- Case studies: the M4 competition and Gu, Kelly and Xiu (2020).\n"
       "- References: Huang and Petukhina (2022), *Applied Time Series Analysis and Forecasting with Python*, Ch. 10; "
       "Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.), Sec. 12.4; Hastie, Tibshirani and "
       "Friedman (2009); Makridakis, Spiliotis and Assimakopoulos (2020, 2022)."),
    *common_cells(INSTALL),
    md("## Definitions and helpers used in the whole notebook\n\n"
       "- `load_daily()`: daily mean load of Romania in GW (ENTSO-E, the file of Chapter 4), 2022–2026.\n"
       "- `calendar(idx)`: weekday dummies, annual Fourier terms, Orthodox Easter, Christmas, public holidays.\n"
       "- `direct_frame(y, h)`: the supervised table for horizon $h$ (14 lags, 7- and 28-day means, the same weekday, "
       "the calendar of the target day; target $y_{t+h}$).\n"
       "- `forecast_direct`, `forecast_recursive`, `forecast_mlp`, `forecast_lstm`: the four multi-step strategies.\n"
       "- `dm_raw(l1, l2)`: Diebold–Mariano test with the Harvey–Leybourne–Newbold correction (Chapter 4)."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. From a series to a table\n\n- The load, three features, a small lag table and the three multi-step strategies."),
    code(src(g.mini_table, g.fig_features, g.fig_strategies) + "\n\n\ny = load_daily()\nprint(pd.DataFrame(mini_table(y)['rows']))\n"
         "print(fig_features(y, save=False))\nfig_strategies(save=False)\nF = direct_frame(y, 1)\nF.dropna().head()"),
    md("## 2. Validation without leakage\n\n- Four validation schemes; random folds on overlapping targets; bias and variance of trees."),
    code(src(g.fig_cv_schemes, g.leakage_frame, g.cv_r2, g.leakage_experiment, g.fig_leakage, g.true_f, g.bias_variance,
             g.fig_bias_variance)
         + "\n\n\nfig_cv_schemes(save=False)\nprint(fig_leakage(save=False))\nbv = fig_bias_variance(save=False)\nprint(bv['best_depth'])"),
    md("## 3. Ridge and lasso\n\n- Coefficient paths on the 30 standardised load features; the lasso penalty chosen walk-forward."),
    code(src(g.shrinkage_paths, g.fig_shrinkage) + "\n\n\nprint(fig_shrinkage(save=False))"),
    md("## 4. Trees, random forest and gradient boosting\n\n- A depth-2 tree, ensembles on validation data, the extrapolation failure."),
    code(src(g.fig_tree, g.ensemble_curves, g.fig_ensembles, g.hicp, g.fig_extrapolation)
         + "\n\n\nprint(fig_tree(save=False))\nprint(fig_ensembles(save=False))\nprint(fig_extrapolation(save=False))"),
    md("## 5. Neural networks\n\n- The MLP and its activation functions, the RNN and the LSTM cell; a small LSTM and a multi-output MLP "
       "trained on the load up to the first origin."),
    code(src(g.fig_mlp, g.fig_rnn) + "\n\n\nfig_mlp(save=False)\nfig_rnn(save=False)\no = pd.Timestamp(CV_FIRST)\n"
         "f_lstm, losses = forecast_lstm(y, o)\nprint('LSTM loss: first epoch', round(losses[0], 3), 'last epoch', round(losses[-1], 3))\n"
         "pd.DataFrame({'actual': y.loc[o + pd.Timedelta(days=1):].iloc[:CV_H].values, 'LSTM': f_lstm, 'MLP': forecast_mlp(y, o)}).round(3)"),
    md("## 6. Electricity load: ML against Chapter 4\n\n"
       "- The 26 origins and the 14-day horizon of Chapter 4. Ridge and gradient boosting (direct and recursive) are "
       "recomputed here (a few minutes); the random forest, the MLP and the LSTM take longer, so their walk-forward forecasts "
       "are read from `ch9_load_ml.csv`, written by `load_ml_cv` in `generate_all_charts.py`.\n"
       f"- Slides: MASE of DHR {f(LD['DHR']['MASE'])}, ridge {f(LD['Ridge']['MASE'])}, GB direct {f(LD['GB direct']['MASE'])}, "
       f"LSTM {f(LD['LSTM']['MASE'])}."),
    code(src(g.load_ml_cv, g.ch9_file, g.saved_load_ml, g.ch4_errors, g.load_summary, g.fig_load_results, g.fig_load_forecasts)
         + "\n\n\nE_new = load_ml_cv(y, models=('Ridge', 'GB direct', 'GB recursive'), save_csv=False)\n"
           "E_saved = saved_load_ml()\nE_ml = pd.concat([E_new, E_saved[E_saved['model'].isin(['RF', 'MLP', 'LSTM'])]])\n"
           "E_st = ch4_errors()\ntab, dm, scale, byh = load_summary(E_ml, E_st, y)\nprint(tab.round(3))\n"
           "print({k: round(v['p'], 3) for k, v in dm.items() if k.endswith('|DHR')})\n"
           "fig_load_results(tab, byh, save=False)\nprint(fig_load_forecasts(E_ml, E_st, y, save=False))"),
    md("## 7. Feature importance and prediction intervals\n\n"
       "- Permutation importance of gradient boosting for horizons 1 and 14 days.\n"
       "- 90% intervals by quantile boosting and split conformal prediction; to keep the notebook short, every second origin "
       "is used here (the slides use all 26)."),
    code(src(g.importance, g.fig_importance, g.load_intervals, g.fig_intervals)
         + "\n\n\nprint(fig_importance(save=False))\nR, out = load_intervals(y, every=2)\nprint(out)\nfig_intervals(R, out, y, save=False)"),
    md("## 8. Romanian inflation: local and global models\n\n- 27 EU countries from Eurostat; yearly walk-forward from 2016; "
       "horizons 1, 3, 6, 12 months."),
    code(src(g.inflation_panel, g.inflation_frame, g.inflation_models, g.inflation_forecasts, g.inflation_summary,
             g.fig_inflation)
         + "\n\n\nE, pi = inflation_forecasts(save=False)\nSI = inflation_summary(E)\n"
           "print(pd.DataFrame({h: SI[h]['rel'] for h in SI}).round(3))\nprint(fig_inflation(E, pi, SI, save=False))"),
    md("## 9. Realised volatility and the sign of returns\n\n- HAR against lasso, random forest, GB and an MLP for the "
       "S&P 500 (the slides also show the DAX); the sign of the next daily return against the majority-class baseline."),
    code(src(g.daily_variance, g.rv_frame, g.MeanModel, g.MLPEnsemble, g.rv_models, g.walk_forward, g.qlike, g.dm_hac,
             g.rv_forecasts, g.rv_metrics, g.fig_rv)
         + "\n\n\nX, P = rv_forecasts('sp500')\nM = rv_metrics(X, P)\n"
           "print(pd.DataFrame({m: M[m] for m in ['HAR', 'Lasso', 'RF', 'GB', 'MLP']}).T.round(4))\nfig_rv(X, P, save=False)"),
    code(src(g.sign_frame, g.sign_models, g.sign_walk_forward, g.sign_metrics, g.fig_sign)
         + "\n\n\nSG = {k: sign_metrics(*sign_walk_forward(k)) for k in ('sp500', 'bet')}\n"
           "print({k: (round(v['base_acc'], 4), {m: round(v[m]['acc'], 4) for m in ('Logit', 'RF', 'GB')}) for k, v in SG.items()})\n"
           "fig_sign(SG, save=False)"),
    md("## 10. Case studies: M4 and Gu, Kelly and Xiu (2020)"),
    code(src(g.m4_table, g.fig_m4, g.fig_gkx) + "\n\n\nprint(fig_m4(save=False))\nfig_gkx(save=False)"),
    md("## Exercises\n\n1. Add the lags 21 and 28 to `direct_frame` and repeat the ridge and GB forecasts of Section 6.\n"
       "2. Train the GB models of Section 6 on the log of the load: does the MASE change?\n"
       "3. Replace the global GB of Section 8 by a global ridge model: does pooling help a linear model too?"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_FUNCS = [s.lag_table, s.recursive_direct, s.best_split, s.conformal_pinball, s.sem_origins, s.b1_load, s.b3_inflation]
SEMINAR = [
    md("# Time Series Analysis — Seminar 9: Machine learning for time series\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The seminar comes before Lecture 9; the slides give the definitions (\"What you need today\").\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class."),
    *common_cells(INSTALL),
    md("## Seminar functions\n\n- Helpers of Chapter 9 (`load_daily`, `calendar`, `direct_frame`, `forecast_direct`, "
       "`forecast_recursive`, `dm_raw`, `inflation_forecasts`, `sign_frame`) and the seminar functions: `lag_table`, "
       "`recursive_direct`, `best_split`, `conformal_pinball`, `b1_load`, `b3_inflation`."),
    code(CONSTS + "\nSEM_FIRST, SEM_LAST, SEM_H = '2026-01-05', '2026-06-22', 7\n"
         "NO_CAL = [f for f in FEATS if f not in ('easter', 'xmas', 'holiday')]\n\n\n"
         + src(*CORE, g.ch9_file, g.hicp, g.inflation_panel, g.inflation_frame, g.inflation_models, g.inflation_forecasts,
               g.sign_frame, g.sign_models, g.sign_metrics) + '\n\n\n' + src(*SEM_FUNCS)),
    code("# [solutions only]\n" + src(s.rolling_leak, s.ridge_lasso_1d, s.mase_dm, s.b2_strategies, s.b4_sign, s.c2_numbers)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: a lag table and two strategies\n\n"
       "**Context.** The series $y_1, \\dots, y_8$ = 10, 12, 11, 13, 14, 13, 15, 16.\n\n"
       "1. Build the table with the features $y_{t-1}$, $y_{t-2}$ and the target $y_t$; how many rows does it have?\n"
       "2. With the fitted model $\\hat y_t = 2 + 0.6\\,y_{t-1} + 0.3\\,y_{t-2}$, compute the recursive forecasts "
       "$\\hat y_9$, $\\hat y_{10}$, $\\hat y_{11}$.\n"
       "3. A direct two-step model is $\\hat y_{t+2} = 3 + 0.5\\,y_t + 0.3\\,y_{t-1}$: compute $\\hat y_{10}$ and the number "
       "of rows of its table.\n\n**Report:** the table, three recursive forecasts, one direct forecast, two row counts."),
    code("# Solution\nY = [10, 12, 11, 13, 14, 13, 15, 16]\nprint(pd.DataFrame(lag_table(Y)))\nrecursive_direct(Y)"),
    md(f"**Interpretation of the result.** Recursive: {f(S['A1']['rec'][0])}, {f(S['A1']['rec'][1])}, {f(S['A1']['rec'][2])}; "
       f"direct two-step: {f(S['A1']['direct2'])}. The two strategies differ because each is estimated for its own horizon."),
    md("## A2 [Proposed]: spot the leak\n\n"
       "**Context.** The daily series 10, 12, 11, 13, 14; the target is the last value. Model: A1.\n\n"
       "1. Compute the correct 3-day mean feature for this target, and the mean that `rolling(3).mean()` without `shift(1)` would give.\n"
       "2. Say whether each choice leaks: (a) a scaler fitted on 2022–2026 before the split; (b) holiday dummies of the "
       "target day; (c) the temperature measured on the target day; (d) shuffled 5-fold CV for a 21-day target.\n\n"
       "**Report:** two means and four verdicts with one reason each."),
    code("# Solution\nrolling_leak([10, 12, 11, 13, 14])"),
    md("## A3 [Solved]: one split of a regression tree\n\n"
       "**Context.** Six days, $x$ = load on the same weekday last week, $y$ = load today (GW): (4.8, 5.0), (5.0, 5.2), "
       "(5.6, 5.9), (6.0, 6.0), (6.4, 6.6), (6.9, 7.0).\n\n"
       "1. List the five candidate thresholds.\n2. For each, compute the two leaf means and the total SSE.\n"
       "3. Choose the split and give the forecast for $x = 6.2$.\n\n**Report:** a table of five rows, the chosen threshold, one forecast."),
    code("# Solution\nsp = best_split([4.8, 5.0, 5.6, 6.0, 6.4, 6.9], [5.0, 5.2, 5.9, 6.0, 6.6, 7.0])\n"
         "print(pd.DataFrame(sp['rows']).round(3))\nsp['best']"),
    md(f"**Interpretation of the result.** The best split is $x \\le {f(S['A3']['best']['thr'], 1)}$ (SSE "
       f"{f(S['A3']['best']['sse'], 3)} against {f(S['A3']['sst'], 3)} without a split); for $x = 6.2$ the forecast is "
       f"{f(S['A3']['best']['right_mean'])} GW."),
    md("## A4 [Proposed]: ridge and lasso in one dimension\n\n"
       "**Context.** One standardised feature with $S_{xy} = 40$, $S_{xx} = 50$. Model: A3.\n\n"
       "1. Compute the OLS coefficient.\n2. Compute the ridge coefficient for $\\lambda = 10$ and $\\lambda = 50$.\n"
       "3. Compute the lasso coefficient for $\\lambda = 20$ and $\\lambda = 100$.\n"
       "4. Explain which method can drop a lag from the model.\n\n**Report:** five coefficients and one sentence."),
    code("# Solution\nridge_lasso_1d(40.0, 50.0, [10.0, 50.0], [20.0, 100.0])"),
    md("## A5 [Solved]: a conformal interval and the pinball loss\n\n"
       "**Context.** Forecast $\\hat y = 6.10$ GW; absolute errors on 9 calibration days: 0.12, 0.30, 0.05, 0.22, 0.41, 0.18, "
       "0.09, 0.27, 0.35 GW.\n\n1. Build the 80% split conformal interval.\n"
       "2. A 90% quantile forecast is $q = 6.4$ GW: compute its pinball loss if $y = 6.0$ and if $y = 6.6$.\n\n"
       "**Report:** $k$, $\\hat q$, the interval, two losses."),
    code("# Solution\nconformal_pinball([0.12, 0.30, 0.05, 0.22, 0.41, 0.18, 0.09, 0.27, 0.35], 6.10)"),
    md(f"**Interpretation of the result.** $k = {S['A5']['k']}$, $\\hat q = {f(S['A5']['half'])}$, interval "
       f"[{f(S['A5']['lo'])}, {f(S['A5']['hi'])}] GW; pinball losses {f(S['A5']['pin']['6.0'])} and {f(S['A5']['pin']['6.6'])}: "
       "an outcome above a high quantile costs nine times more per GW."),
    md("## A6 [Proposed]: MASE and a Diebold–Mariano statistic\n\n"
       "**Context.** Average absolute errors (GW) at 6 origins: A = 0.30, 0.25, 0.40, 0.28, 0.35, 0.22; "
       "B = 0.32, 0.30, 0.38, 0.35, 0.41, 0.30; MASE scale 0.31 GW. Model: A5.\n\n"
       "1. Compute the MAE and the MASE of each model.\n2. Compute $d_t = A_t - B_t$, its mean and standard deviation, and "
       "$DM = \\bar d/(s/\\sqrt{6})$.\n3. Compare $|DM|$ with the 5% critical value of $t(5)$ and conclude.\n\n"
       "**Report:** four accuracy numbers, $\\bar d$, $DM$ and a decision."),
    code("# Solution\nmase_dm([0.30, 0.25, 0.40, 0.28, 0.35, 0.22], [0.32, 0.30, 0.38, 0.35, 0.41, 0.30], 0.31)"),
    # ---------------- Part B
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: gradient boosting for the electricity load\n\n"
       "**Question.** Does a gradient-boosting model forecast next week's daily load better than the same weekday last week?\n\n"
       "1. Build the direct tables with `direct_frame`.\n"
       "2. At each weekly origin from 5 January to 22 June 2026, train one `HistGradientBoostingRegressor` per horizon "
       "(1–7 days) on the data before the origin and forecast the next 7 days.\n"
       "3. Compute the MAE, RMSE and MASE of GB and of the seasonal naive forecast.\n"
       "4. Run the Diebold–Mariano test on the origin-average absolute errors.\n"
       "5. Interpretation: at which horizons does GB gain most, and why?\n\n"
       "**Report:** six accuracy numbers, a DM statistic with its p-value, one sentence."),
    code("# Solution\nb1 = b1_load(save=False)\nprint({k: b1[k] for k in ('GB direct', 'Seasonal naive', 'dm')})\n"
         "print(pd.DataFrame(b1['byh'], index=range(1, 8)).round(3))"),
    md(f"**Interpretation of the result.** MASE {f(S['B1']['GB direct']['mase'])} against {f(S['B1']['Seasonal naive']['mase'])} "
       f"(DM p = {f(S['B1']['dm']['p'], 3)}). The gain is largest one day ahead, where GB uses yesterday's level."),
    md("## B2 [Proposed]: recursive, direct and the value of holidays\n\n"
       "**Question.** Does the recursive strategy forecast as well as the direct one, and how much do the holiday features "
       "help? Model: B1.\n\n1. Repeat B1 with `forecast_recursive`.\n"
       "2. Repeat the direct model without the three holiday features (`feats=NO_CAL`).\n"
       "3. Report the MAE of the three models on all days, on the public-holiday days and on the other days.\n"
       "4. Interpretation: which feature group explains most of the difference, and on which days?\n\n"
       "**Report:** nine MAE values and one sentence."),
    code("# Solution\nb2_strategies(save=False)"),
    md("## B3 [Solved]: Romanian inflation three months ahead\n\n"
       "**Question.** Can machine learning forecast Romanian inflation three months ahead better than the random walk and a "
       "linear AR model?\n\n1. Build the features of each country and the target $\\pi_{t+3} - \\pi_t$ (`inflation_frame`).\n"
       "2. Every January from 2016, refit AR, lasso, random forest and GB on Romania only, and a global GB on the 27 EU "
       "countries; forecast every month of the year.\n3. Compute the RMSE of each model and its ratio to the random walk.\n"
       "4. Run the DM test for AR against RW and for the global against the local GB.\n"
       "5. Interpretation: what happened to all models in 2021–2023?\n\n"
       "**Report:** six RMSEs, two DM tests and one sentence."),
    code("# Solution\nb3 = b3_inflation(save=False)\nprint(pd.DataFrame({'RMSE': b3['rmse'], 'relative to RW': b3['rel'], "
         "'RMSE 2021-2023': b3['rmse_surge']}).round(3))\nprint('AR vs RW:', b3['dm_ar_rw'])\nprint('global vs local GB:', b3['dm_gl_loc'])"),
    md(f"**Interpretation of the result.** AR has the smallest RMSE ({f(S['B3']['rmse']['AR'])} pp, ratio "
       f"{f(S['B3']['rel']['AR'])}); the global GB beats the local one ({f(S['B3']['rmse']['GB global'])} against "
       f"{f(S['B3']['rmse']['GB local'])}), but no difference is significant. All errors jump in 2021–2023: the energy shock "
       "was not in the past of inflation."),
    md("## B4 [Proposed]: the sign of BET returns\n\n"
       "**Question.** Can a classifier predict whether the BET rises tomorrow better than \"it always rises\"? Model: B3.\n\n"
       "1. Build the features of `sign_frame` and the target $1\\{r_{t+1} > 0\\}$.\n"
       "2. Each January from 2015, fit a logistic regression and a gradient-boosting classifier on all earlier days; the "
       "baseline predicts the majority class of the training data.\n"
       "3. Report the accuracy of the three, the $z$ statistic against the baseline and the AUC.\n"
       "4. Interpretation: what does the result say about the weak-form efficiency of the Bucharest market?\n\n"
       "**Report:** three accuracies, two $z$ statistics, two AUCs and one sentence."),
    code("# Solution\nb4_sign(save=False)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: a global model for European electricity load\n\n"
       "**Idea.** ENTSO-E publishes the hourly load of every European country. Train one global GB model on about 20 countries "
       "(lags scaled by each country's mean, weekday, national holidays, a country identifier) and compare it, on the 26 "
       "origins of Chapter 4, with the local GB and with DHR. Does it forecast Orthodox Easter better? Which countries help "
       "Romania most? Models: B1, B3."),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant answered a question about machine learning for markets and electricity load:\n\n"
       "- (a) a random forest with 5-fold cross-validation explains about 37% of the next month's S&P 500 return, so the "
       "market is predictable;\n"
       "- (b) a logistic model predicts the daily sign with about 54% accuracy, well above 50%: a profitable signal;\n"
       "- (c) for the load, GB has MASE 0.97, so it is 3% more accurate than DHR;\n"
       "- (d) standardise the whole series first, then split it into training and test;\n"
       "- (e) an LSTM always beats ARIMA because it has long memory;\n"
       "- (f) calendar features of the target day, such as holidays, are allowed because they are known in advance.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number "
       "(lecture notebook, sections 2, 6, 9).\n\n**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2_numbers()"),
]

if __name__ == '__main__':
    build(LECTURE, 9, 'lecture')
    build(SEMINAR, 9, 'seminar')
