"""
build_notebooks_ch15.py -- lecture and seminar notebooks of Chapter 15 (TSA): review, the course in one notebook
================================================================================================================
Output: notebooks/EN/chapter15_lecture_notebook.ipynb, notebooks/EN/chapter15_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_15/generate_all_charts.py and seminar15.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch15.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter15_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter15_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 15
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(15)
import generate_all_charts as g   # noqa: E402
import seminar15 as s             # noqa: E402
import build_quantlets as bq      # noqa: E402

CONSTS = '\n'.join(bq.CONSTS)
CORE = bq.CORE
INSTALL = bq.INSTALL

# =============================================================================
# LECTURE: THE COURSE IN ONE NOTEBOOK
# =============================================================================
LECTURE = [
    md("# Time Series Analysis — Chapter 15: Review and exam preparation\n\n"
       "*Lecture notebook: the course in one notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Part 1: the Box–Jenkins method from start to finish on the monthly HICP of Romania (Eurostat), 2005–2026: plot, "
       "unit-root tests, identification, estimation, residual checks, forecasts and their evaluation (Chapters 0–4).\n"
       "- Part 2: the output of exam problem 3 (an AR(1) × SAR(1) model) and its forecast computed by hand.\n"
       "- Part 3: one short computation per block of chapters on Romanian data: exponential smoothing (Chapter 0), "
       "GARCH (Chapter 5), VAR and Granger causality (Chapters 6–7), long memory (Chapter 8), the local level model "
       "(Chapter 10).\n"
       "- Chapters 11–14 are for self-study.\n"
       "- The first code cell installs the `arch` package if it is missing (`pip install arch`).\n"
       "- References: Box, Jenkins and Reinsel (2008); Huang and Petukhina (2022); Hyndman and Athanasopoulos (2021)."),
    *common_cells(INSTALL),
    md("## Definitions and helpers used in the whole notebook\n\n"
       "- $y_t = 100\\ln P_t$; monthly inflation $100\\Delta\\ln P_t$; annual inflation $100\\Delta_{12}\\ln P_t$; "
       "$z_t = \\Delta\\Delta_{12}\\,100\\ln P_t$.\n"
       "- `adf_kpss(x, reg)`: ADF ($H_0$: unit root) and KPSS ($H_0$: stationarity) with the same deterministic terms.\n"
       "- `ljung_box(e, lags, df_adj)`: Ljung–Box with $m - k$ degrees of freedom for the residuals of a model with $k$ "
       "ARMA parameters.\n"
       "- `dm_test(e1, e2)`: Diebold–Mariano on squared errors with the Harvey–Leybourne–Newbold correction."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Step 1: plot the data"),
    code(src(g.fig_bj_data) + "\n\n\nprint(fig_bj_data(save_it=False))"),
    md("- The index trends up; monthly inflation is seasonal (high in January, low in June); tax changes move prices "
       "in a single month."),
    md("## 2. Step 2: unit-root and stationarity tests"),
    code(src(g.bj_tests) + "\n\n\npd.DataFrame(bj_tests()).T[['reg', 'n', 'adf', 'adf_p', 'adf_cv5', 'kpss', 'kpss_cv5']].round(3)"),
    md("- The level has a unit root; annual inflation is inconclusive; $z_t$ is stationary by both tests: $d = D = 1$."),
    md("## 3. Step 3: identification"),
    code(src(g.fig_bj_acf) + "\n\n\nac = fig_bj_acf(save_it=False)\nprint('lags outside the band, ACF:', ac['out_r'])\n"
         "print('lags outside the band, PACF:', ac['out_p'])"),
    md("- One regular spike and one seasonal spike at lag 12 in the ACF, a decaying seasonal PACF: SARIMA$(1,1,0)(0,1,1)_{12}$."),
    md("## 4. Step 4: estimation of every small SARIMA model (about a minute)"),
    code(src(g.bj_grid) + "\n\n\ngrid = bj_grid()\nprint('lowest BIC:', grid['best_bic'], '| chosen:', grid['chosen'])\n"
         "pd.DataFrame(grid['rows'])[['model', 'k', 'aicc', 'bic', 'lb12_p', 'lb24_p']].head(10).round(3)"),
    md("- The identified model fails the Ljung–Box test; the chosen model has the lowest BIC among the models whose "
       "residuals pass at lags 12 and 24."),
    md("## 5. Step 5: residual diagnostics"),
    code(src(g.fig_bj_diag) + "\n\n\nd = fig_bj_diag(grid, save_it=False)\n"
         "print(pd.DataFrame({'estimate': d['chosen']['params'], 'se': d['chosen']['se'], 'p': d['chosen']['p']}).round(4))\n"
         "print('Ljung-Box:', d['chosen']['lb'])\nprint('Jarque-Bera:', round(d['chosen']['jb'], 1), '| largest residuals:', d['chosen']['big'])"),
    md("- No autocorrelation left, but heavy tails: the two largest residuals are the tax changes of June 2015 and July 2010."),
    md("## 6. Step 6: forecasts and their evaluation"),
    code(src(g.fig_bj_forecast) + "\n\n\nfc = fig_bj_forecast(grid, save_it=False)\nprint(pd.DataFrame(fc['acc']).T.round(3))\n"
         "print('DM against seasonal naive:', fc['dm_sn'])\n"
         "print('annual inflation forecasts:', {k: round(fc[k], 2) for k in ('a1', 'a1_lo', 'a1_hi', 'a12', 'a12_lo', 'a12_hi')})"),
    md("- The SARIMA errors are the smallest, but the Diebold–Mariano test does not reject equal accuracy on 24 months."),
    md("## 7. Exam problem 3: SARIMA output and a forecast by hand"),
    code(src(g.exam_p3) + "\n\n\np3 = exam_p3()\n"
         "print({k: round(p3[k], 4) for k in ('phi', 'Phi', 'se_phi', 'se_Phi', 'sigma2')})\n"
         "hand = p3['phi'] * p3['zT'] + p3['Phi'] * p3['zT11'] - p3['phi'] * p3['Phi'] * p3['zT12']\n"
         "print('z forecast by hand:', round(hand, 4), '| statsmodels:', round(p3['z_next_sm'], 4))\n"
         "print('annual inflation forecast for', p3['next'], ':', round(p3['aT'] + hand, 2), '%')"),
    md("## 8. The course in a few functions (Chapters 0, 5, 6–7, 8, 10)"),
    code(src(g.course_smoothing, g.course_garch, g.course_var, g.local_whittle, g.course_memory, g.course_state_space)
         + "\n\n\nprint('Chapter 0, smoothing:', course_smoothing())\nprint('Chapter 5, GARCH(1,1)-t for the BET:', course_garch())\n"
           "print('Chapters 6-7, VAR of inflation and ROBOR:', course_var())\nprint('Chapter 8, local Whittle d of the BET:', course_memory())\n"
           "print('Chapter 10, local level of monthly inflation:', course_state_space())"),
    md("## Exercises\n\n"
       "1. Re-run the Box–Jenkins steps with the training sample ending in August 2022 and compare the chosen model.\n"
       "2. Add dummies for the VAT changes (July 2010, June 2015, August 2025) as regressors of the SARIMA model "
       "(`SARIMAX(..., exog=...)`) and compare the Jarque–Bera statistic of the residuals.\n"
       "3. Replace the seasonal differencing by twelve month dummies and compare the forecasts on the test sample."),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_CONSTS = (CONSTS + f"\nfrom statsmodels.tsa.stattools import coint\nUNEMP = {s.UNEMP!r}\nGDP_NSA = {s.GDP_NSA!r}\nIP = {s.IP!r}"
              f"\nSEM_START = {s.SEM_START!r}\nIP_TEST = {s.IP_TEST!r}\nBET_START = {s.BET_START!r}\nYIELD_START = {s.YIELD_START!r}")
SEM_VISIBLE = [s.quarter, s.a1_arima011, s.b1_ip]
SEM_SOLUTIONS = [s.a2_unemployment, s.a3_gdp_sarima, s.ro_pi_i, s.a4_var, s.a5_garch, s.b2_bet, s.b3_yields, s.c2_check]

SEMINAR = [
    md("# Time Series Analysis — Seminar 15: Review and exam preparation\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The seminar comes before Lecture 15; the slides \"What you need today\" are a formula sheet for Chapters 0–10.\n"
       "- Part A: five exam-type problems on paper (the output is on the slides), checked in code. Part B: three data "
       "tasks with an interpretation question. Part C: a project idea and an AI answer to audit.\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class.\n"
       "- The first code cell installs the `arch` package if it is missing."),
    *common_cells(INSTALL),
    md("## Seminar functions\n\n- Helpers of Chapter 15 (`adf_kpss`, `ljung_box`, `fit_sarima`) and the functions of the "
       "[Solved] tasks: `a1_arima011` (A1) and `b1_ip` (B1)."),
    code(SEM_CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_VISIBLE)),
    code("# [solutions only]\n" + src(*SEM_SOLUTIONS)),
    # ---------------- Part A
    md("# Part A: exam problems on paper\n\n- Solve each problem on paper first; then check it with the code."),
    md("## A1 [Solved]: an ARIMA(0,1,1) by hand\n\n"
       "**Context.** $\\Delta y_t = \\varepsilon_t - 0.6\\,\\varepsilon_{t-1}$, $\\varepsilon_t \\sim WN(0, 1)$; today $y_T = 100$, "
       "$\\hat\\varepsilon_T = 1.5$.\n\n"
       "1. Show that $\\Delta y_t$ is stationary and invertible, and compute $\\rho_{\\Delta y}(1)$.\n"
       "2. Compute $\\mathrm{Var}(y_t)$ for $t = 10$ and $t = 100$.\n"
       "3. Forecast $y_{T+1}$ and $y_{T+4}$ and give the 95% interval for $h = 4$.\n"
       "4. Give the weight of the equivalent exponential smoothing.\n\n"
       "**Report:** five numbers, an interval and one sentence."),
    code("# Solution\na1_arima011()"),
    md("**Interpretation of the result.** The forecast is flat (99.1), like that of exponential smoothing with "
       "$\\alpha = 0.4$, but its variance grows with the horizon: a unit root."),
    md("## A2 [Proposed]: identifying the Romanian unemployment rate\n\n"
       "**Context.** Unemployment rate (SA, %), 2005–2026 (Eurostat `une_rt_m`). Model: Seminar 3, A1; Seminar 2, A6.\n\n"
       "1. Run ADF and KPSS on $u_t$ and $\\Delta u_t$ and decide the order of integration.\n"
       "2. Compute the ACF and PACF of $\\Delta u_t$ and Ljung–Box $Q(4)$.\n"
       "3. Propose an ARIMA model.\n\n**Report:** two decisions, two numbers, a model."),
    code("# Solution\na2_unemployment()"),
    md("## A3 [Proposed]: SARIMA output for Romanian GDP\n\n"
       "**Context.** $z_t = \\Delta\\Delta_4\\,100\\ln \\mathrm{GDP}_t$, real GDP not seasonally adjusted (Eurostat). "
       "Model: Seminar 4, A4.\n\n"
       "1. Fit an AR(1) × SAR(1)$_4$ model without constant and write its equation.\n"
       "2. Test the residuals with Ljung–Box $Q(8)$.\n"
       "3. Forecast $z$ and the annual growth for the next quarter by hand; check with `get_forecast`.\n\n"
       "**Report:** the equation, one test, two forecasts."),
    code("# Solution\na3_gdp_sarima()"),
    md("## A4 [Proposed]: VAR output and Granger causality\n\n"
       "**Context.** VAR(2) for Romanian annual inflation and ROBOR 3M (Eurostat), monthly, 2005–2026. Model: Seminar 6, A5.\n\n"
       "1. Estimate the VAR equation by equation.\n"
       "2. Compute the Granger $F$ statistics from the restricted and unrestricted RSS.\n"
       "3. Compute the long-run effect on ROBOR of a permanent 1 pp rise in inflation.\n\n"
       "**Report:** the ROBOR equation, two $F$ tests, one long-run effect."),
    code("# Solution\na4_var()"),
    md("## A5 [Proposed]: GARCH output for the DAX and VaR 1%\n\n"
       "**Context.** GARCH(1,1)-t for daily DAX log returns in % since 2000. Model: Seminar 5, A1 and A4.\n\n"
       "1. Report the persistence, the half-life and the long-run annual volatility.\n"
       "2. Compute tomorrow's variance from the last day.\n"
       "3. Compute the one-day VaR 1% with the standardised t quantile and with the Normal one.\n\n"
       "**Report:** six numbers and one sentence."),
    code("# Solution\na5_garch()"),
    # ---------------- Part B
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: Box–Jenkins for Romanian industrial production\n\n"
       "**Question.** Which SARIMA model forecasts Romanian industrial production best, and does it beat the seasonal naive method?\n\n"
       "1. Run ADF and KPSS on $100\\ln y_t$, its first difference and $z_t = \\Delta\\Delta_{12}100\\ln y_t$.\n"
       "2. Plot the ACF of $z_t$ and propose a model.\n"
       "3. Fit the airline model and SARIMA$(1,1,1)(0,1,1)_{12}$ on the training sample; compare AICc, BIC and Ljung–Box.\n"
       "4. Forecast the 24 test months with both models and with the seasonal naive method; compare RMSE and MAE.\n"
       "5. Interpretation: which model would you use to forecast, and why?\n\n"
       "**Report:** a table of tests, a table of two models, three RMSE and one sentence."),
    code("# Solution\nb1 = b1_ip(save_it=False)\nprint(pd.DataFrame(b1['tests']).T[['adf', 'adf_p', 'kpss', 'kpss_cv5']].round(3))\n"
         "print({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ('model', 'aicc', 'bic', 'rmse', 'mae')} "
         "for k, v in b1['models'].items()})\nprint('seasonal naive:', b1['sn'])"),
    md("**Interpretation of the result.** The SARIMA$(1,1,1)(0,1,1)_{12}$ model wins in sample (lower BIC, clean "
       "residuals) but loses on the test sample; the airline model is slightly better than the seasonal naive method. "
       "Use the airline model and keep the seasonal naive method as the benchmark."),
    md("## B2 [Proposed]: volatility of the BET since 2015\n\n"
       "**Question.** Is the Bucharest market calmer or more turbulent today than usual, and what does it mean for "
       "tomorrow's VaR 1%? Model: Seminar 5, B1.\n\n"
       "1. Run the ARCH-LM test with five lags on the demeaned returns.\n"
       "2. Fit a GARCH(1,1) with Student-t innovations; report the persistence and the half-life.\n"
       "3. Compare the long-run, the sample and today's annualised volatility.\n"
       "4. Compute tomorrow's VaR 1% with the t and with the Normal quantile.\n"
       "5. Interpretation: is the BET calmer or more turbulent than usual today?\n\n"
       "**Report:** one test, five parameters, three volatilities, two VaR values and one sentence."),
    code("# Solution\nb2_bet()"),
    md("## B3 [Proposed]: the Romanian and German 10-year yields\n\n"
       "**Question.** Is the Romanian 10-year yield tied to the German one in the long run? Model: Seminar 7, B1 and A3.\n\n"
       "1. Run ADF on both yields and on their differences.\n"
       "2. Run the Engle–Granger test (Romania on Germany) and report $\\hat\\beta$.\n"
       "3. Run the Johansen trace and maximum-eigenvalue tests.\n"
       "4. Estimate the error-correction equation of each yield and compute the half-life.\n"
       "5. Interpretation: is the Romanian yield tied to the German one?\n\n"
       "**Report:** four ADF results, two cointegration tests, two adjustment coefficients and one sentence."),
    code("# Solution\nb3_yields()"),
    # ---------------- Part C
    md("# Part C: an open idea and AI critique"),
    md("## C1 [Proposed]: a forecast competition for Romanian inflation\n\n"
       "**Question.** Which method would have forecast Romanian monthly inflation best over the last ten years, one and "
       "twelve months ahead?\n\n"
       "1. List five competitors: seasonal naive, SES, SARIMA, ARFIMA, a ridge model with lags and month dummies.\n"
       "2. Describe a rolling-origin design: first origin, step, horizons, refits.\n"
       "3. Say how you would treat the tax changes of 2010, 2015 and 2025.\n"
       "4. Choose the error measures and the test for the comparison.\n\n"
       "**Report:** a one-page plan that a team could turn into its project."),
    code("# Solution\nplan = {'origins': 'monthly from 2015-01, expanding window', 'horizons': [1, 12],\n"
         "        'models': ['seasonal naive', 'SES', 'SARIMA (orders chosen before each origin)', 'ARFIMA', 'ridge with lags and month dummies'],\n"
         "        'tax changes': 'known VAT dates as dummies, results with and without those months',\n"
         "        'evaluation': 'MASE against the seasonal naive method, Diebold-Mariano with the HLN correction'}\nplan"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant checked the answers to Part A:\n\n"
       "- (a) for the unemployment rate, ADF gives p = 0.52, so the unit root is rejected at 5%;\n"
       "- (b) KPSS = 1.04 > 0.463 rejects, so the level of unemployment is stationary;\n"
       "- (c) for the DAX, alpha + beta = 0.992, so a shock halves in 1/(1 - 0.992) days;\n"
       "- (d) the VaR 99% of the DAX for tomorrow is negative;\n"
       "- (e) inflation Granger-causes ROBOR, which proves that inflation causes the central bank to raise rates;\n"
       "- (f) Ljung-Box Q(12) on the residuals of an AR(1) x SAR(1) model has 12 degrees of freedom.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nR = {'A2': a2_unemployment(), 'A4': a4_var(), 'A5': a5_garch()}\nc2_check(R)"),
]

if __name__ == '__main__':
    build(LECTURE, 15, 'lecture')
    build(SEMINAR, 15, 'seminar')
