"""
build_notebooks_ch3.py -- lecture and seminar notebooks of Chapter 3 (TSA): unit roots and ARIMA models
======================================================================================================
Output: notebooks/EN/chapter3_lecture_notebook.ipynb, notebooks/EN/chapter3_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_03/generate_all_charts.py and seminar3.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch3.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter3_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter3_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 3
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(3)
import generate_all_charts as g   # noqa: E402
import seminar3 as s              # noqa: E402

CONSTS = ('import itertools\nimport statsmodels.api as sm\n'
          'from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss, zivot_andrews\n'
          'from statsmodels.tsa.adfvalues import mackinnonp, mackinnoncrit\n'
          'from statsmodels.tsa.arima.model import ARIMA\nfrom statsmodels.tsa.arima_process import ArmaProcess\n'
          'from statsmodels.stats.diagnostic import acorr_ljungbox\n'
          f'SEED = {g.SEED!r}\nGDP_SA = {g.GDP_SA!r}\nHICP = {g.HICP!r}\nGDP_START = {g.GDP_START!r}\n'
          f'START = {g.START!r}\nBAND_COL = st.IDAred\nCASES = {g.CASES!r}')
CORE = [g.ro_gdp, g.ro_hicp, g.ro_inflation, g.us_gdp, g.quarterly, g.monthly, g.sample_acf, g.sample_pacf, g.acf_bars,
        g.save, g.years_axis, g.adf_test, g.pp_test, g.kpss_test, g.za_test, g.verdict, g.df_tau_batch, g.random_walks,
        g.ljung_box, g.arima_grid, g.best_order, g.gdp_series, g.gdp_order, g.gdp_fit, g.psi_weights, g.ols_summary]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Time Series Analysis — Chapter 3: Unit roots and ARIMA models\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Deterministic and stochastic trends; detrending a random walk.\n"
       "- Spurious regression: a Monte Carlo and a real example.\n"
       "- The Dickey–Fuller distributions, size and power; ADF, Phillips–Perron and KPSS on real series.\n"
       "- Structural breaks: Perron and Zivot–Andrews.\n"
       "- ARIMA for Romanian GDP: identification, AICc, diagnostics, forecasts; forecast intervals; automatic ARIMA; "
       "EUR/RON against the random walk; US GDP after 2007.\n"
       "- References: Huang and Petukhina (2022), *Applied Time Series Analysis and Forecasting with Python*, Ch. 4–5; "
       "Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.), Ch. 9; Hamilton (1994), Ch. 15–17."),
    *common_cells(),
    md("## Definitions and helpers used in the whole notebook\n\n"
       "- ADF regression: $\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum_{j=1}^{k}\\delta_j\\Delta y_{t-j} + \\varepsilon_t$, "
       "$H_0$: $\\gamma = 0$ (unit root); `adf_test(x, reg)` with `reg` = `'n'`, `'c'` or `'ct'`.\n"
       "- Phillips–Perron: `pp_test(x, reg)`, the Dickey–Fuller $t$-ratio corrected with the Newey–West long-run variance.\n"
       "- KPSS: `kpss_test(x, reg)`, $H_0$: stationarity; Zivot–Andrews: `za_test(x, reg)`.\n"
       "- ARIMA: `arima_grid(y, d, pmax, qmax, trend)` returns AICc, BIC and convergence for each $(p, q)$."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Deterministic and stochastic trends\n\n- Four trending series; a trend-stationary and a difference-stationary "
       "series with the same shocks; a random walk detrended by a straight line."),
    code(src(g.fig_four_series, g.fig_ts_ds, g.fig_detrend)
         + "\n\n\nfig_four_series(save_it=False)\nfig_ts_ds(save_it=False)\nfig_detrend(save_it=False)"),
    md("## 2. Spurious regression\n\n- Independent random walks regressed on each other; Romanian HICP against the S&P 500."),
    code(src(g.spurious_mc, g.fig_spurious_mc, g.fig_spurious_real)
         + "\n\n\nfig_spurious_mc(save_it=False)\nfig_spurious_real(save_it=False)"),
    md("## 3. The Dickey–Fuller test\n\n- Null distributions in three specifications; size and power; a worked example on "
       "Romanian GDP."),
    code(src(g.fig_df_dist, g.fig_adf_power, g.df_by_hand)
         + "\n\n\nfig_df_dist(save_it=False)\nfig_adf_power(save_it=False)\ndf_by_hand()"),
    md("## 4. Phillips–Perron and KPSS on real series\n\n- KPSS partial sums; ADF, PP and KPSS on fourteen series."),
    code(src(g.fig_kpss_sums, g.real_series, g.unit_root_table)
         + "\n\n\nfig_kpss_sums(save_it=False)\nU = unit_root_table()\n"
           "pd.DataFrame({k: {'det': v['reg'], 'T': v['n'], 'ADF': v['adf']['stat'], 'ADF p': v['adf']['p'], 'lags': v['adf']['lags'], "
           "'PP': v['pp']['stat'], 'KPSS': v['kpss']['stat'], 'verdict': v['verdict']} for k, v in U.items()}).T"),
    md("## 5. Structural breaks\n\n- A stationary series around a broken mean; Zivot–Andrews on the Nile and on EUR/RON."),
    code(src(g.fig_breaks_sim, g.fig_breaks_real) + "\n\n\nfig_breaks_sim(save_it=False)\nfig_breaks_real(save_it=False)"),
    md("## 6. An ARIMA model for Romanian GDP\n\n- Identification, model choice by AICc (two samples), diagnostics, forecasts, "
       "over-differencing."),
    code(src(g.fig_gdp_ident, g.gdp_models, g.fig_gdp_diag, g.fig_gdp_forecast, g.fig_overdiff_gdp)
         + "\n\n\nfig_gdp_ident(save_it=False)\nM = gdp_models()\nprint(M['main']['best'], M['full']['best_any'])\n"
           "fig_gdp_diag(save_it=False)\nfig_gdp_forecast(save_it=False)\nfig_overdiff_gdp(save_it=False)"),
    md("## 7. Forecast intervals\n\n- Half-widths of 95% intervals from the $\\psi$ weights; a worked ARIMA(1,1,0) example."),
    code(src(g.fig_interval_width, g.arima110_by_hand) + "\n\n\nfig_interval_width(save_it=False)\narima110_by_hand()"),
    md("## 8. ARIMA in practice\n\n- Romanian inflation with $d = 0$ and $d = 1$; EUR/RON against the random walk; "
       "US real GDP after 2007."),
    code(src(g.fig_inflation_d, g.fig_eurron_forecast, g.fig_us_gdp)
         + "\n\n\ninf = fig_inflation_d(save_it=False)\nprint(inf['d0']['order'], inf['d1']['order'], inf['d_hk'])\n"
           "fx = fig_eurron_forecast(save_it=False)\nprint(fx['rmse1'], fx['rmse20'])\nfig_us_gdp(save_it=False)"),
    md("## Exercises\n\n1. Run `adf_test` and `kpss_test` on the DAX log price (`load_close('dax')`) and on its returns.\n"
       "2. Repeat the Romanian GDP model choice with BIC instead of AICc: is the chosen model the same?\n"
       "3. Change the break size `delta` in `fig_breaks_sim` to 2 and to 8: how does the ADF rejection rate change?"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_FUNCS = [s.df_by_hand, s.arima_by_hand, s.ar_roots, s.unit_root_set, s.b1_tests, s.b3_gdp]
SEMINAR = [
    md("# Time Series Analysis — Seminar 3: Unit roots and ARIMA models\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The seminar comes before Lecture 3; the slides give the definitions (\"What you need today\").\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class."),
    *common_cells(),
    md("## Seminar functions\n\n- Helpers of Chapter 3 (`adf_test`, `pp_test`, `kpss_test`, `za_test`, `arima_grid`, `gdp_series`, "
       "`ro_inflation`) and the seminar functions: `df_by_hand`, `arima_by_hand`, `ar_roots`, `unit_root_set`, `b1_tests`, `b3_gdp`."),
    code(CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_FUNCS)),
    code("# [solutions only]\n" + src(s.adf_choice, s.b2_tests, s.b4_inflation, s.c1_breaks, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: a Dickey–Fuller statistic\n\n"
       "**Context.** A monthly interest rate, $T = 120$: OLS gives $\\Delta y_t = 0.42 - 0.061\\,y_{t-1}$, with "
       "$\\mathrm{SE}(\\hat\\gamma) = 0.025$.\n\n"
       "1. Compute $\\tau$ and $\\hat\\phi$.\n2. Compare $\\tau$ with the 5% Dickey–Fuller value for a regression with a constant, and decide.\n"
       "3. Say what a standard one-sided $t$-test would have concluded.\n"
       "4. Explain why a constant (and no trend) is the right choice for an interest rate.\n\n"
       "**Report:** two numbers, a decision and two sentences."),
    code("# Solution\ndf_by_hand(-0.061, 0.025, 120, 'c')"),
    md("**Interpretation of the result.** $\\tau = -2.44$ is above the Dickey–Fuller value (about $-2.89$) but below $-1.645$: "
       "the Dickey–Fuller table does not reject the unit root, a Normal table would wrongly reject it."),
    md("## A2 [Proposed]: choosing the deterministic terms\n\n"
       "**Context.** The log of a price index ($T = 200$) rises steadily. ADF: $\\tau = 1.85$ (no constant), $-1.21$ (constant), "
       "$-3.62$ (constant and trend); KPSS with trend: 0.09. Model: A1.\n\n"
       "1. Say which specification fits the data, and why.\n2. Decide at 5% in each specification "
       "(critical values: `mackinnoncrit(N=1, regression=..., nobs=200)`).\n"
       "3. Combine the chosen ADF result with KPSS (5% value 0.146).\n4. Say how you would model the series.\n\n"
       "**Report:** three decisions, a verdict and one sentence."),
    code("# Solution\nadf_choice({'n': 1.85, 'c': -1.21, 'ct': -3.62}, 200)"),
    md("## A3 [Solved]: ARIMA(1,1,0) forecasts\n\n"
       "**Context.** $\\Delta y_t = 0.6\\,\\Delta y_{t-1} + \\varepsilon_t$, $\\sigma = 2$; $y_{T-1} = 103$, $y_T = 108$.\n\n"
       "1. Compute $\\hat y_{T+1}$, $\\hat y_{T+2}$ and $\\hat y_{T+3}$.\n"
       "2. Compute $\\psi_1$ and $\\psi_2$, and the forecast error variances for $h = 1, 2, 3$.\n3. Give the 95% intervals.\n\n"
       "**Report:** three forecasts and three intervals."),
    code("# Solution\narima_by_hand(ar=(0.6,), d=1, y=(103.0, 108.0), sigma=2.0, H=3)"),
    md("**Interpretation of the result.** The forecasts rise by 3, 1.8 and 1.08: the momentum fades. The half-width grows from "
       "3.92 to 10.66 in three steps, faster than $\\sqrt{3}$, because $\\psi_j$ grows towards $1/(1 - \\phi) = 2.5$."),
    md("## A4 [Proposed]: ARIMA(0,1,1) and exponential smoothing\n\n"
       "**Context.** $\\Delta y_t = \\varepsilon_t - 0.6\\,\\varepsilon_{t-1}$, $\\sigma = 1$; $y_T = 50$, $\\hat\\varepsilon_T = 1.5$. Model: A3.\n\n"
       "1. Compute $\\hat y_{T+1}$ and $\\hat y_{T+h}$ for $h \\ge 2$.\n"
       "2. Compute the forecast error variances for $h = 1$, $2$ and $5$, and the 95% intervals.\n"
       "3. Show that the forecast equals simple exponential smoothing (Chapter 0) and give $\\alpha$.\n\n"
       "**Report:** two forecasts, three intervals and $\\alpha$."),
    code("# Solution\narima_by_hand(ma=(-0.6,), d=1, y=(50.0,), e_last=1.5, sigma=1.0, H=5)"),
    md("## A5 [Solved]: the order of integration of an equation\n\n"
       "**Context.** $y_t = 1.5\\,y_{t-1} - 0.5\\,y_{t-2} + \\varepsilon_t + 0.4\\,\\varepsilon_{t-1}$.\n\n"
       "1. Write the AR polynomial $\\phi(z)$ and find its roots.\n2. Say whether $y_t$ is stationary, and give $d$.\n"
       "3. Write the model as ARIMA$(p,d,q)$ for $\\Delta y_t$.\n\n**Report:** two roots, $d$ and the ARIMA equation."),
    code("# Solution\nar_roots([1.5, -0.5])"),
    md("**Interpretation of the result.** Roots 1 and 2: one unit root, so $d = 1$ and "
       "$\\Delta y_t = 0.5\\,\\Delta y_{t-1} + \\varepsilon_t + 0.4\\,\\varepsilon_{t-1}$, an ARIMA(1,1,1)."),
    md("## A6 [Proposed]: integrated or over-differenced?\n\n"
       "**Context.** (a) $y_t = 1.8\\,y_{t-1} - 0.8\\,y_{t-2} + \\varepsilon_t$; (b) $\\Delta y_t = \\varepsilon_t - \\varepsilon_{t-1}$. Model: A5.\n\n"
       "1. For (a), factor $\\phi(z)$, give $d$ and write the ARIMA model.\n"
       "2. For (b), find the MA root and say what $y_t$ really is.\n"
       "3. Compute $\\rho(1)$ of $\\Delta y_t$ in (b) and say how you would detect this case in data.\n\n"
       "**Report:** two orders of integration and two sentences."),
    code("# Solution\nprint(ar_roots([1.8, -0.8]))\nprint(np.roots([1, -1]))   # MA polynomial 1 - z (root 1)"),
    # ---------------- Part B
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: is the BET log price a random walk?\n\n"
       "**Question.** Does the BET log price have a unit root, and are its daily returns stationary?\n\n"
       "1. Plot $p_t = 100\\ln P_t$ with a fitted linear trend, and plot $r_t = \\Delta p_t$.\n"
       "2. Run ADF (lags by AIC), PP and KPSS on $p_t$ with constant and trend.\n"
       "3. Run the same tests on $r_t$ with a constant.\n"
       "4. Interpretation: if the BET is far below its fitted trend line, should you expect it to return to the line?\n\n"
       "**Report:** a table of six statistics, two verdicts and two sentences."),
    code("# Solution\nb1 = b1_tests('bet', save=False)\n"
         "pd.DataFrame({part: {'ADF': b1[part]['adf']['stat'], 'ADF p': b1[part]['adf']['p'], 'lags': b1[part]['adf']['lags'], "
         "'PP': b1[part]['pp']['stat'], 'KPSS': b1[part]['kpss']['stat'], 'verdict': b1[part]['verdict']} for part in ['level', 'diff']}).T"),
    md("**Interpretation of the result.** The log price has a unit root and the returns are stationary. With a unit root the "
       "trend line does not attract the series: a distance from the line carries no forecast of a return to it."),
    md("## B2 [Proposed]: the S&P 500 and EUR/RON\n\n"
       "**Question.** Do the S&P 500 and the EUR/RON rate give the same verdicts as the BET? Model: B1.\n\n"
       "1. Repeat steps 2 and 3 of B1 for both series (`unit_root_set(100 * np.log(load_close('sp500', start=START)))`, "
       "and the same for `'eurron'` without `start`).\n"
       "2. Compare the ADF and PP statistics of the returns, and explain why PP is much more negative.\n"
       "3. Interpretation: the EUR/RON rate is managed by the central bank; does the test result say that it is a random walk?\n\n"
       "**Report:** a table (two series, two rows each) and two sentences."),
    code("# Solution\nb2 = b2_tests()\n"
         "pd.DataFrame({(k, part): {'ADF': v[part]['adf']['stat'], 'PP': v[part]['pp']['stat'], 'KPSS': v[part]['kpss']['stat'], "
         "'verdict': v[part]['verdict']} for k, v in b2.items() for part in ['level', 'diff']}).T"),
    md("## B3 [Solved]: an ARIMA model for Romanian GDP\n\n"
       "**Question.** Which ARIMA model describes Romanian real GDP, and how uncertain is a two-year forecast?\n\n"
       "1. Choose $d$ with ADF and KPSS on $y_t = 100\\ln Y_t$ (constant and trend) and on $\\Delta y_t$ (constant).\n"
       "2. Fit ARIMA$(p,1,q)$ with drift for $p, q \\le 2$ and choose the model with the lowest AICc.\n"
       "3. Check the residuals with Ljung–Box $Q^*(8)$.\n4. Forecast 8 quarters with 80% and 95% intervals.\n"
       "5. Interpretation: why is the 95% interval after 8 quarters about twice as wide as after 2 quarters?\n\n"
       "**Report:** the tests, the AICc of six models, the chosen model, $Q^*(8)$, the chart and one sentence."),
    code("# Solution\nb3 = b3_gdp(save=False)\n"
         "print('verdicts:', b3['tests']['level']['verdict'], b3['tests']['diff']['verdict'])\n"
         "print('AICc:', {k: round(v['aicc'], 1) for k, v in b3['grid'].items()})\n"
         "print('model:', b3['order'], b3['params'], 'Q*(8):', b3['lb8'])\n"
         "print('half-widths after 2 and 8 quarters:', round(b3['half2'], 2), round(b3['half8'], 2))"),
    md("**Interpretation of the result.** The chosen model is a random walk with drift, whose error variance is $h\\sigma^2$: "
       "the interval grows like $\\sqrt{h}$, and $\\sqrt{8/2} = 2$."),
    md("## B4 [Proposed]: Romanian inflation, $d = 0$ or $d = 1$?\n\n"
       "**Question.** Is Romanian 12-month inflation stationary, and how much does the answer change the forecast? Model: B3.\n\n"
       "1. Run ADF, PP and KPSS on $\\pi_t$ (`ro_inflation()`) and on $\\Delta\\pi_t$, with a constant.\n"
       "2. Find the best ARIMA$(p,0,q)$ with a mean and the best ARIMA$(p,1,q)$ by AICc, with $p \\le 3$, $q \\le 2$.\n"
       "3. Forecast 12 months with both models, with 95% intervals, and run Ljung–Box $Q^*(24)$ on their residuals.\n"
       "4. Interpretation: why do the two models give similar point forecasts but different intervals?\n\n"
       "**Report:** a table of tests, two models, two forecasts with intervals, two Ljung–Box tests and two sentences."),
    code("# Solution\nb4_inflation(save=False)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: a break or a unit root?\n\n"
       "**Question.** Do Romanian inflation and GDP have a unit root, or are they stationary around a mean or a trend that "
       "shifted once? Models: B1, B3.\n\n"
       "1. Run the Zivot–Andrews test with a break in the level, then in the level and the trend, on both series.\n"
       "2. Compare each statistic with its 5% critical value and report the break dates.\n"
       "3. Match the dates with events (inflation targeting since 2005, the 2008–2009 crisis, the 2022 energy shock).\n"
       "4. Interpretation: why should a break date found by a test be checked against history before it is believed?\n\n"
       "**Report:** a table of four statistics with dates, the chart and a plan for a project."),
    code("# Reference analysis\nc1_breaks(save=False)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant analysed Romanian GDP and prices. It answered:\n\n"
       "- (a) the ADF p-value for log GDP is above 0.05, so GDP is proved to have a unit root;\n"
       "- (b) the KPSS statistic for log GDP is above 0.146, so KPSS rejects the unit root;\n"
       "- (c) regressing log HICP on the log S&P 500 gives a huge t-statistic: US stocks drive Romanian prices;\n"
       "- (d) GDP growth is stationary, so differencing it once more will make the ARIMA model even better;\n"
       "- (e) a random-walk forecast has the same 95% interval width at every horizon;\n"
       "- (f) the 5% Dickey–Fuller critical value with a constant is more negative than $-1.645$, because the Dickey–Fuller "
       "distribution is shifted to the left.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 3, 'lecture')
    build(SEMINAR, 3, 'seminar')
