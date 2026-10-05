"""
build_notebooks_ch4.py -- lecture and seminar notebooks of Chapter 4 (TSA): seasonality and forecasting
======================================================================================================
Output: notebooks/EN/chapter4_lecture_notebook.ipynb, notebooks/EN/chapter4_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_04/generate_all_charts.py and seminar4.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides. TBATS and Prophet are optional: the first cell installs them if missing,
and every function that needs them checks that they are available.
Run:  python3 notebooks/build_notebooks_ch4.py
      jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/EN/chapter4_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/EN/chapter4_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 4
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(4)
import generate_all_charts as g   # noqa: E402
import seminar4 as s              # noqa: E402
import build_quantlets as bq      # noqa: E402

CONSTS = '\n'.join(bq.CONSTS)
CORE = bq.CORE + bq.LOAD
INSTALL = bq.INSTALL

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Time Series Analysis — Chapter 4: Seasonality and forecasting: SARIMA, TBATS, Prophet\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Seasonal data from Romania (Eurostat, ENTSO-E); seasonal differences and seasonal unit roots (HEGY, OCSB, Canova–Hansen).\n"
       "- SARIMA$(p,d,q)(P,D,Q)_s$: theory, the airline model, Romanian GDP; seasonal adjustment (NSA and SCA series).\n"
       "- Fourier terms and calendar effects (Orthodox Easter, working days); multiple seasonality in electricity load: MSTL, "
       "dynamic harmonic regression, TBATS, Prophet.\n"
       "- Time-series cross-validation, MASE, the Diebold–Mariano test and forecast combination.\n"
       "- References: Huang and Petukhina (2022), Ch. 4–5; Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* "
       "(3rd ed.), Ch. 9, 10, 12, 13; Box, Jenkins and Reinsel (2008), Ch. 9.\n"
       "- TBATS and Prophet are optional packages: the first code cell installs them if they are missing (about a minute in Colab)."),
    *common_cells(INSTALL),
    md("## Definitions used in the whole notebook\n\n"
       "- SARIMA: $\\phi(L)\\Phi(L^s)(1-L)^d(1-L^s)^D y_t = \\theta(L)\\Theta(L^s)\\varepsilon_t$; `fit_sarima(y, order, sorder)` "
       "is exact Gaussian maximum likelihood (`statsmodels` `SARIMAX`).\n"
       "- `ro_holidays(years)`: Romanian public holidays, with Orthodox Easter and Pentecost from `dateutil`.\n"
       "- `load_hourly()`, `load_daily()`: electricity load of Romania (ENTSO-E), in Romanian winter time (UTC+2).\n"
       "- `fourier(idx, period, K)`: Fourier terms; `easter_share(idx, w)`: the Easter regressor; `working_days(idx)`."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Seasonality in Romanian data"),
    code(src(g.airline, g.fig_series, g.fig_seasonal_plot) + "\n\n\nfig_series(save_it=False)\nfig_seasonal_plot(save_it=False)"),
    md("## 2. Seasonal differences and seasonal unit roots\n\n- The roots of $1 - z^s$; the ACF of log GDP after $\\Delta$, "
       "$\\Delta_4$ and $\\Delta\\Delta_4$; HEGY with simulated critical values; OCSB and Canova–Hansen (pmdarima, if installed)."),
    code(src(g.fig_unit_circle, g.fig_gdp_differencing, g.hegy_stats, g.hegy_critical, g.seasonal_tests)
         + "\n\n\nfig_unit_circle(save_it=False)\nfig_gdp_differencing(save_it=False)\ntests = seasonal_tests(save_csv=False)\n"
           "print(tests['hegy'], tests['hegy_cv'])\npd.DataFrame(tests['table'])"),
    md("## 3. SARIMA: theoretical ACF and PACF"),
    code(src(g.fig_theory_acf) + "\n\n\nth = fig_theory_acf(save_it=False)"),
    md("## 4. The airline model"),
    code(src(g.fig_airline, g.fig_airline_forecast) + "\n\n\nfig_airline(save_it=False)\nfig_airline_forecast(save_it=False)"),
    md("## 5. Romanian GDP (not adjusted): SARIMA"),
    code(src(g.gdp_models, g.parse_model, g.fig_gdp_diag, g.fig_gdp_forecast)
         + "\n\n\nT = gdp_models(save_csv=False)\nprint('AICc:', T['best_aicc'], ' BIC:', T['best_bic'])\n"
           "pd.DataFrame(T['rows'])[['model', 'k', 'aicc', 'bic', 'lb8_p', 'sigma']].round(3)"),
    code("o, so = parse_model(T['best_bic'])\ndiag = fig_gdp_diag(o, so, save_it=False)\nfc = fig_gdp_forecast(o, so, save_it=False)\ndiag"),
    md("## 6. Seasonal adjustment: Eurostat NSA and SCA series"),
    code(src(g.fig_sa_nsa) + "\n\n\nfig_sa_nsa(save_it=False)"),
    md("## 7. Fourier terms and calendar effects (Orthodox Easter, working days)"),
    code(src(g.fig_fourier, g.easter_regression, g.fig_easter) + "\n\n\nfig_fourier(save_it=False)\nea = fig_easter(save_it=False)\n"
         "pd.DataFrame({'food': ea['food']['params'], 'food se': ea['food']['se'], 'total': ea['total']['params'], "
         "'total se': ea['total']['se']}).round(4)"),
    md("## 8. Multiple seasonality: electricity load\n\n- Hourly ACF, MSTL with periods 24 and 168, daily load and holidays, "
       "a dynamic harmonic regression."),
    code(src(g.fig_load_hourly, g.fig_mstl, g.fig_load_daily, g.dhr_fit)
         + "\n\n\nfig_load_hourly(save_it=False)\nfig_mstl(save_it=False)\nfig_load_daily(save_it=False)\ndhr = dhr_fit()\n"
           "print(dhr['order'], {k: round(v, 3) for k, v in dhr['coef'].items()})"),
    md("## 9. TBATS and Prophet (optional packages)"),
    code(src(g.fig_prophet_components, g.tbats_fit) + "\n\n\nprint(fig_prophet_components(save_it=False))\nprint(tbats_fit())"),
    md("## 10. Forecast evaluation and combination\n\n- `FAST = True`: an origin every 56 days (a few minutes); `FAST = False`: "
       "every 14 days, the 26 origins of the slides (about 10 minutes, TBATS is refitted at each origin)."),
    code("FAST = True\n\n\n" + src(g.load_cv, g.cv_summary, g.fig_cv_scheme, g.fig_cv_results, g.fig_load_forecasts, g.fig_combination)
         + "\n\n\nfig_cv_scheme(save_it=False)\nE, orders = load_cv(step=56 if FAST else CV_STEP, save_csv=False)\n"
           "tab, dm, best, scale, byh = cv_summary(E)\nprint('orders:', orders, ' best single model:', best)\ntab.round(4)"),
    code("fig_cv_results(tab, byh, save_it=False)\nfig_load_forecasts(E, save_it=False)\n"
         "comb = fig_combination(E, 'DHR', 'Prophet', save_it=False) if 'Prophet' in tab.index else {}\n"
         "pd.DataFrame({k: v for k, v in dm.items() if k.endswith('|se')}).T[['hln', 'p']].round(3)"),
    code(src(g.gdp_cv) + "\n\n\ngc = gdp_cv()\npd.DataFrame({m: {h: v['mae'] for h, v in d.items()} for m, d in gc['tab'].items()}).round(3)"),
    md("## Exercises\n\n1. Fit the airline model to Romanian industrial production (`IPI`) with and without `working_days` as a regressor. "
       "Which one passes Ljung–Box?\n"
       "2. Add the Easter regressor to the DHR of daily load as a 7-day window before Easter and compare the cross-validation MAE.\n"
       "3. Combine DHR and Prophet with the weight estimated on the first half of the origins only, and evaluate it on the second half."),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_FUNCS = [s.seasonal_differences, s.airline_acf, s.dm_by_hand, s.b1_gdp, s.b3_load_cv]
SEMINAR = [
    md("# Time Series Analysis — Seminar 4: Seasonality and forecasting\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The seminar comes before Lecture 4; the slides give the formulas (\"What you need today\").\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class."),
    *common_cells(),
    md("## Seminar functions\n\n- Helpers of Chapter 4 (`fit_sarima`, `ljung_box`, `acf_bars`, `load_daily`, `load_exog`, `dm_raw`, "
       "`working_days`) and the seminar functions: `seasonal_differences`, `airline_acf`, `dm_by_hand`, `b1_gdp`, `b3_load_cv`."),
    code(CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_FUNCS)),
    code("# [solutions only]\nIPI_GRID = " + repr(s.IPI_GRID) + "\n\n\n"
         + src(s.sarima_polynomials, s.ma1_from_rho, s.combination_mse, s.b2_ipi, s.b4_hicp, s.c1_tourism, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: seasonal differences by hand\n\n"
       "**Context.** A quarterly series, $t = 1, \\dots, 9$: 10, 14, 16, 18, 12, 16, 19, 21, 14.\n\n"
       "1. Compute $\\Delta_4 y_t$ for $t = 5, \\dots, 9$.\n2. Compute $\\Delta\\Delta_4 y_t$ for $t = 6, \\dots, 9$.\n"
       "3. Expand $(1 - L)(1 - L^4)$ and check it on $t = 9$.\n4. Give the seasonal naive forecasts for $t = 10, \\dots, 13$.\n\n"
       "**Report:** thirteen numbers and one sentence on what $\\Delta_4$ removes."),
    code("# Solution\nseasonal_differences([10, 14, 16, 18, 12, 16, 19, 21, 14])"),
    md("**Interpretation of the result.** $\\Delta_4 y$ = 2, 2, 3, 3, 2: the seasonal pattern is gone and only the growth from one "
       "year to the next is left; the seasonal naive forecasts repeat the last four values."),
    md("## A2 [Proposed]: SARIMA polynomials\n\n**Context.** Monthly data, $s = 12$. Model: A1.\n\n"
       "1. Write the airline model SARIMA$(0,1,1)(0,1,1)_{12}$ with $\\theta = -0.4$, $\\Theta = -0.6$ as an equation for $y_t$.\n"
       "2. List the lags of $y$ and of $\\varepsilon$ that appear, and the coefficient of $\\varepsilon_{t-13}$.\n"
       "3. Expand $(1 - 0.5L)(1 - 0.3L^{12})$ for SARIMA$(1,0,0)(1,0,0)_{12}$.\n"
       "4. Count the parameters of SARIMA$(1,1,1)(1,1,1)_{12}$, including $\\sigma^2$.\n\n"
       "**Report:** two equations, two lists of lags and one count."),
    code("# Solution\nprint(sarima_polynomials(theta=-0.4, Theta=-0.6, s=12))\nprint(sarima_polynomials(phi=0.5, Phi=0.3, d=0, D=0, s=12))"),
    md("## A3 [Solved]: the ACF of the airline model\n\n"
       "**Context.** $w_t = \\Delta\\Delta_4 y_t = (1 - 0.5L)(1 - 0.6L^4)\\varepsilon_t$.\n\n"
       "1. Write $w_t$ as an MA(5) and list its nonzero coefficients.\n2. Compute $\\gamma(0)/\\sigma^2$.\n"
       "3. Compute $\\rho_1, \\dots, \\rho_5$.\n4. Say what the sample ACF of $w_t$ should look like.\n\n"
       "**Report:** five autocorrelations and one sentence."),
    code("# Solution\nairline_acf(-0.5, -0.6, s=4)"),
    md("**Interpretation of the result.** $\\rho_1 = -0.4$, $\\rho_4 = -0.441$ and the satellites $\\rho_3 = \\rho_5 = 0.176$; all "
       "other lags are zero: the signature of the airline model."),
    md("## A4 [Proposed]: a seasonal AR and moment estimates\n\n**Context.** Model: A3.\n\n"
       "1. For $y_t = 0.7y_{t-4} + \\varepsilon_t$, compute $\\rho_4$, $\\rho_8$, $\\rho_{12}$ and say which other lags are zero.\n"
       "2. Describe its PACF.\n"
       "3. The sample ACF of $\\Delta\\Delta_{12}y_t$ is $-0.40$ at lag 1, $-0.45$ at lag 12, small at lags 11 and 13; propose a model.\n"
       "4. Estimate $\\theta$ and $\\Theta$ from $\\rho_1$ and $\\rho_{12}$, keeping the invertible roots.\n\n"
       "**Report:** three autocorrelations, a model and two estimates."),
    code("# Solution\nprint(sarma_process(Phi=[0.7], s=4).acf(13)[1:].round(3))\nprint(ma1_from_rho(-0.40), ma1_from_rho(-0.45))"),
    md("## A5 [Solved]: MASE and the Diebold–Mariano test\n\n"
       "**Context.** Six non-overlapping windows; errors of method 1: 0.4, −0.6, 0.9, −0.3, 0.5, −0.8; of method 2: 1.1, −0.9, 0.7, "
       "−1.2, 0.8, −1.0; the in-sample MAE of the seasonal naive method is 1.\n\n"
       "1. Compute the MAE and the MASE of both methods.\n2. Compute $d_t = e_{1t}^2 - e_{2t}^2$, $\\bar d$ and $\\hat\\sigma_d^2$.\n"
       "3. Compute DM and HLN.\n4. Decide at 5% with $t(5)$.\n\n**Report:** four accuracy measures, the test statistic and a decision."),
    code("# Solution\ndm_by_hand([0.4, -0.6, 0.9, -0.3, 0.5, -0.8], [1.1, -0.9, 0.7, -1.2, 0.8, -1.0], scale=1.0)"),
    md("**Interpretation of the result.** Method 1 has the smaller errors in five windows out of six, but HLN = −2.28 is inside "
       "$\\pm 2.571$: with six windows the gain is not significant at 5%."),
    md("## A6 [Proposed]: two forecasts and their combination\n\n"
       "**Context.** Eight windows; $e_1$: 2.0, −1.5, 1.0, −2.5, 3.0, 0.5, −1.0, 1.5; $e_2$: 1.0, −2.5, 2.0, −0.5, 1.5, 2.0, −2.0, 0.5. Model: A5.\n\n"
       "1. Compute the MAE of both methods and the DM and HLN statistics with squared losses.\n2. Decide at 5% with $t(7)$.\n"
       "3. The equal-weight combination has errors $(e_1 + e_2)/2$: compute its MAE and RMSE.\n"
       "4. For unbiased forecasts with $\\sigma_1 = 2$, $\\sigma_2 = 3$, $\\rho = 0.3$, compute MSE(0.5), $w^*$ and MSE($w^*$).\n\n"
       "**Report:** six numbers, a decision and three numbers for the combination."),
    code("# Solution\ne1 = np.array([2.0, -1.5, 1.0, -2.5, 3.0, 0.5, -1.0, 1.5])\ne2 = np.array([1.0, -2.5, 2.0, -0.5, 1.5, 2.0, -2.0, 0.5])\n"
         "print(dm_by_hand(e1, e2))\nprint(dm_by_hand((e1 + e2) / 2, e1))\nprint(combination_mse(2.0, 3.0, 0.3))"),
    # ---------------- Part B
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: the airline model for Romanian GDP\n\n"
       "**Question.** How stable is the seasonal pattern of Romanian GDP, and what does the airline model forecast for the next four quarters?\n\n"
       "1. Estimate SARIMA$(0,1,1)(0,1,1)_4$ for $100\\ln Y_t$ by maximum likelihood and report $\\hat\\theta$, $\\hat\\Theta$ and their standard errors.\n"
       "2. Test the residuals with Ljung–Box $Q^*(8)$ on $8 - 2 = 6$ degrees of freedom.\n"
       "3. Forecast the next four quarters, in bn EUR, with 95% intervals.\n4. Compute the implied year-on-year growth for each forecast quarter.\n"
       "5. Interpretation: what does $\\hat\\Theta$ say about how fast the seasonal pattern of GDP changes?\n\n"
       "**Report:** two estimates with standard errors, a test, four forecasts with intervals and one sentence."),
    code("# Solution\nb1 = b1_gdp(save=False)\n{k: (np.round(v, 3) if isinstance(v, (float, list)) else v) for k, v in b1.items()}"),
    md("**Interpretation of the result.** $\\hat\\theta$ is not significant and $\\hat\\Theta \\approx -0.58$: the seasonal pattern of the "
       "forecast is an exponentially weighted average of past years, with weight $1 + \\hat\\Theta \\approx 0.42$ on the newest year. "
       "The pattern evolves, but slowly."),
    md("## B2 [Proposed]: industrial production and working days\n\n"
       "**Question.** Does the number of working days explain part of the monthly movements of Romanian industrial production? Model: B1.\n\n"
       "1. Estimate the airline model for $100\\ln Y_t$ (with dummies for April and May 2020) with and without `working_days` as a regressor, "
       "and compare AICc and $Q^*(24)$.\n"
       "2. Add two alternatives with working days, $(2,1,0)(0,1,1)_{12}$ and $(1,1,1)(0,1,1)_{12}$, and choose a model.\n"
       "3. Report the working-day coefficient with its standard error, and forecast the next 12 months.\n"
       "4. Interpretation: why must the working-day effect be removed before two months are compared?\n\n"
       "**Report:** a table of AICc and p-values, one coefficient, the forecast chart and one sentence."),
    code("# Solution\nb2 = b2_ipi(save=False)\nprint(pd.DataFrame(b2['table']).round(3))\nprint(b2['best'], b2['params']['x3'], b2['se']['x3'])"),
    md("## B3 [Solved]: cross-validation for daily electricity load\n\n"
       "**Question.** Does a regression with Fourier terms and holiday dummies forecast daily load better than SARIMA and the seasonal naive method?\n\n"
       "1. At each origin (every 28 days from 30 June 2025, horizon 14 days), fit the seasonal naive method (period 7), SARIMA$(1,0,1)(0,1,1)_7$ "
       "and a DHR (4 annual Fourier pairs, weekday and holiday dummies, ARMA(1,1) errors).\n"
       "2. Compute the MAE and the MASE of each method over all origins and horizons.\n"
       "3. Test DHR against SARIMA with the DM test on the window-average squared errors (one value per origin), with the HLN correction.\n"
       "4. Compare the MAE on public holidays with the MAE on ordinary days.\n"
       "5. Interpretation: why is the gain of the DHR largest on public holidays?\n\n"
       "**Report:** a table of MAE and MASE, one test, the chart and one sentence."),
    code("# Solution\nb3 = b3_load_cv(save=False)\nprint(pd.DataFrame({'MAE (GW)': b3['mae'], 'MASE': b3['mase'], 'holidays': b3['mae_hol'], "
         "'other days': b3['mae_nohol']}).round(3))\nprint('DM, DHR vs SARIMA:', b3['dm'])\nprint('holiday and Easter coefficients:', "
         "b3['coef']['holiday'], b3['coef']['easter'])"),
    md("**Interpretation of the result.** The DHR has the smallest MAE and MASE below 1; its gain is largest on holidays, which the "
       "weekly models treat as ordinary working days. Against SARIMA, the DM test is close to the 5% limit with 13 windows."),
    md("## B4 [Proposed]: seasonality in monthly inflation\n\n"
       "**Question.** Is the seasonal pattern of Romanian monthly inflation strong enough to improve one-month forecasts? Model: B3.\n\n"
       "1. Regress $\\pi_t = 100\\,\\Delta\\ln P_t$ (2010–2019) on 12 month dummies and test equal month means with an $F$-test.\n"
       "2. Estimate the airline model for $100\\ln P_t$ and report $\\hat\\theta$ and $\\hat\\Theta$.\n"
       "3. From monthly origins since January 2016, forecast next month's $\\pi$ with SARIMA, the seasonal naive method and the mean of the "
       "last 12 months; compare the MAE.\n"
       "4. Test SARIMA against both benchmarks with the DM test (squared errors).\n"
       "5. Interpretation: is the seasonal pattern strong enough to improve one-month forecasts?\n\n"
       "**Report:** an $F$-test, two estimates, three MAE values, two DM tests and one sentence."),
    code("# Solution\nb4 = b4_hicp(save=False)\nprint(b4['F'], b4['pF'], b4['theta'], b4['Theta'])\nprint(b4['mae'])\nprint(b4['dm_sn'], b4['dm_mean'])"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: tourism after the pandemic\n\n"
       "**Question.** Which training sample gives better forecasts of tourism nights for 2023–2025: data up to 2019, or data up to 2022? Model: B1.\n\n"
       "1. Fit the airline model to $\\ln y_t$ up to December 2019 and forecast 2023–2025.\n2. Repeat with data up to December 2022.\n"
       "3. Compare both with the seasonal naive forecast that repeats 2022, using the MAPE.\n"
       "4. Propose a better treatment of 2020–2021 (intervention dummies, a shorter sample, TBATS or Prophet) and sketch it as a team project.\n\n"
       "**Report:** three MAPE values, the chart and a project plan."),
    code("# Reference analysis\nc1_tourism(save=False)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant answered questions about seasonal models:\n\n"
       "- (a) SARIMA(0,1,1)(0,1,1)$_4$ has four MA parameters, at lags 1, 4 and 5 plus the noise variance;\n"
       "- (b) if the ACF of the differenced monthly series has one spike at lag 12 and the PACF decays at 12, 24, 36, use a seasonal AR(1);\n"
       "- (c) unadjusted Romanian GDP fell by about 34% in the first quarter of 2026 compared with the fourth quarter of 2025: a deep recession;\n"
       "- (d) DM = −2.3 with $d = L(e_1) - L(e_2)$ shows that forecast 2 is significantly more accurate;\n"
       "- (e) TBATS with periods 7 and 365.25 captures the Easter dip of electricity load;\n"
       "- (f) the equal-weight average of two unbiased forecasts never has a larger MSE than the average of their two MSEs.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 4, 'lecture')
    build(SEMINAR, 4, 'seminar')
