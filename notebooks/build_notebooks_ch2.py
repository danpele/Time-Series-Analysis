"""
build_notebooks_ch2.py -- lecture and seminar notebooks of Chapter 2 (TSA): ARMA models
======================================================================================
Output: notebooks/EN/chapter2_lecture_notebook.ipynb, notebooks/EN/chapter2_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_02/generate_all_charts.py and seminar2.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch2.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter2_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter2_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 2
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(2)
import generate_all_charts as g   # noqa: E402
import seminar2 as s              # noqa: E402

CONSTS = ('from statsmodels.tsa.stattools import acf, pacf\nfrom statsmodels.stats.diagnostic import acorr_ljungbox\n'
          'from statsmodels.tsa.arima.model import ARIMA\nfrom statsmodels.tsa.arima_process import ArmaProcess\n'
          'from statsmodels.regression.linear_model import yule_walker\n'
          f'SEED = {g.SEED!r}\nGDP_NSA = {g.GDP_NSA!r}\nGDP_SCA = {g.GDP_SCA!r}\nHICP = {g.HICP!r}\n'
          f'GDP_START = {g.GDP_START!r}\nINFL_START = {g.INFL_START!r}\nBAND_COL = st.IDAred\nMODELS = {g.MODELS!r}\nHERE = "."')
CORE = [g.ro_gdp_growth, g.ro_inflation, g.sample_acf, g.sample_pacf, g.ljung_box, g.acf_bars, g.simulate_arma,
        g.arma_process, g.theoretical_acf, g.theoretical_pacf, g.psi_weights, g.inverse_roots, g.fit_arma, g.save]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Time Series Analysis — Chapter 2: ARMA models\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- AR($p$) and MA($q$): stationarity, invertibility, characteristic roots, ACF and PACF.\n"
       "- ARMA($p,q$): psi weights (impulse responses), identification patterns.\n"
       "- Estimation (Yule–Walker, conditional least squares, maximum likelihood), AIC and BIC, residual diagnostics.\n"
       "- Forecasts and intervals; the Box–Jenkins method on Romanian GDP growth; inflation, BET, EUR/RON, sunspots.\n"
       "- References: Huang and Petukhina (2022), *Applied Time Series Analysis and Forecasting with Python*, Ch. 3–4; "
       "Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.), Ch. 9; Brockwell and Davis (2016), Ch. 3 and 5."),
    *common_cells(),
    md("## Definitions used in the whole notebook\n\n"
       "- ARMA($p,q$): $\\phi(L)(X_t - \\mu) = \\theta(L)\\varepsilon_t$ with $\\phi(z) = 1 - \\phi_1 z - \\dots - \\phi_p z^p$, "
       "$\\theta(z) = 1 + \\theta_1 z + \\dots + \\theta_q z^q$ (the sign convention of statsmodels).\n"
       "- `fit_arma(x, p, q)`: exact Gaussian maximum likelihood (`statsmodels` `ARIMA`); `const` is the mean $\\mu$.\n"
       "- `ljung_box(x, m, df)`: Ljung–Box $Q^*(m)$ with $df$ degrees of freedom ($m - p - q$ for residuals).\n"
       "- `psi_weights`, `theoretical_acf`, `theoretical_pacf`, `inverse_roots`: the theory of the slides."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. AR(1) and AR(2)\n\n- AR(1) with three values of $\\phi$; the AR(2) stationarity triangle, inverse roots and damped waves."),
    code(src(g.fig_ar1, g.fig_ar2) + "\n\n\nfig_ar1(save_it=False)\nfig_ar2(save_it=False, sun_phi=(1.336, -0.650))"),
    code("# roots of an AR(2): stationary if all inverse roots have modulus < 1\n"
         "for phi in [(0.5, 0.3), (1.0, -0.6), (0.6, 0.5), (1.2, -0.32)]:\n"
         "    print(phi, np.round(np.abs(inverse_roots(phi)), 3))"),
    md("## 2. MA(1) and invertibility\n\n- ACF cuts off after lag 1, PACF decays; $\\theta$ and $1/\\theta$ give the same $\\rho(1)$."),
    code(src(g.fig_ma1, g.fig_ma1_rho) + "\n\n\nfig_ma1(save_it=False)\nfig_ma1_rho(save_it=False)"),
    md("## 3. ARMA: identification patterns and impulse responses"),
    code(src(g.fig_patterns, g.fig_psi) + "\n\n\nfig_patterns(save_it=False)\nfig_psi(save_it=False)"),
    md("## 4. Estimators by simulation\n\n- Yule–Walker, conditional least squares, maximum likelihood; $T = 100$."),
    code(src(g.css_ma1, g.ar1_estimates, g.fig_estimators) + "\n\n\nfig_estimators(save_it=False)"),
    md("## 5. Model selection and the Ljung–Box degrees of freedom"),
    code(src(g.ar_ols_ic, g.fig_ic, g.fig_lb_df) + "\n\n\nfig_ic(save_it=False)\nfig_lb_df(save_it=False)"),
    md("## 6. Forecasting: mean reversion and intervals"),
    code(src(g.fig_forecast_theory) + "\n\n\nfig_forecast_theory(save_it=False)"),
    md("## 7. Box–Jenkins: Romanian annual GDP growth\n\n"
       "- Identification, a grid of ARMA($p,q$) models with AIC, BIC and Ljung–Box, diagnostics, forecasts."),
    code(src(g.gdp_models, g.fig_gdp_ident, g.gdp_table, g.fig_gdp_diag, g.fig_gdp_forecast)
         + "\n\n\nfig_gdp_ident(save_it=False)\ntab = gdp_models(ro_gdp_growth('yoy'))\ntab.round(3)"),
    code("T = gdp_table(save_csv=False)\nprint('AIC:', T['aic_best'], 'BIC:', T['bic_best'])\n"
         "diag = fig_gdp_diag(*T['bic_best'], save_it=False)\nfc = fig_gdp_forecast(models=(tuple(T['bic_best']), (1, 0)), save_it=False)\ndiag"),
    md("## 8. More real series: inflation, daily returns, sunspots"),
    code(src(g.fig_inflation, g.returns_ar, g.fig_returns, g.fig_sunspots)
         + "\n\n\ninfl = fig_inflation(save_it=False)\nprint({k: infl[k] for k in ['p', 'params', 'mu', 'sum_phi']})\n"
           "ret = fig_returns(save_it=False)\npd.DataFrame({k: {'phi': v['phi'], 't': v['t'], 'R2': v['r2'], 'kurt': v['kurt']} for k, v in ret.items()}).T.round(3)"),
    code("sun = fig_sunspots(save_it=False)\nsun"),
    md("## Exercises\n\n1. Fit ARMA($p,q$), $p, q \\le 2$, to the DAX daily returns (`log_returns('dax')`) and compare the BIC choice with BET.\n"
       "2. Repeat the Box–Jenkins case for Romanian GDP growth on the sample 2000–2019 (before the pandemic). Does BIC still choose MA(3)?\n"
       "3. Compute the forecast of the inflation AR(2) 60 months ahead and explain the value it converges to."),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_FUNCS = [s.ar1_facts, s.ma1_facts, s.yw_ar2, s.ic_table, s.box_jenkins, s.b1_us_gdp, s.b3_returns]
SEMINAR = [
    md("# Time Series Analysis — Seminar 2: ARMA models\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The seminar comes before Lecture 2; the slides give the formulas (\"What you need today\").\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class."),
    *common_cells(),
    md("## Seminar functions\n\n- Helpers of Chapter 2 (`fit_arma`, `ljung_box`, `acf_bars`, `psi_weights`, `inverse_roots`, "
       "`ro_gdp_growth`, `ro_inflation`) and the seminar functions: `ar1_facts`, `ma1_facts`, `yw_ar2`, `ic_table`, "
       "`box_jenkins`, `b1_us_gdp`, `b3_returns`."),
    code(CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_FUNCS)),
    code("# [solutions only]\n" + src(s.ar2_facts, s.arma11_facts, s.b2_ro_gdp, s.c1_oos, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: an AR(1) process\n\n"
       "**Context.** $X_t = 1 + 0.6X_{t-1} + \\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 0.64)$, Gaussian.\n\n"
       "1. Check stationarity and compute the mean and $\\gamma(0)$.\n2. Compute $\\rho(2)$, $\\rho(3)$ and the half-life of a shock.\n"
       "3. With $X_T = 4$, forecast $X_{T+1}$ and $X_{T+2}$.\n4. Compute the 95% intervals for both forecasts.\n\n"
       "**Report:** eight numbers and one sentence on mean reversion."),
    code("# Solution\nar1_facts(c=1.0, phi=0.6, sigma2=0.64, x_T=4.0)"),
    md("**Interpretation of the result.** The mean is 2.5; from $X_T = 4$ the forecasts move back towards it (3.40, then 3.04), "
       "and the intervals widen towards the unconditional band $2.5 \\pm 1.96$."),
    md("## A2 [Proposed]: an AR(2) process\n\n"
       "**Context.** $X_t = 0.7X_{t-1} - 0.1X_{t-2} + \\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$. Model: A1.\n\n"
       "1. Write the model with the lag operator and factorise $\\phi(z)$.\n"
       "2. Find the roots and decide whether the process is stationary; check the three triangle conditions.\n"
       "3. Compute $\\rho(1)$, $\\rho(2)$ and $\\rho(3)$ from the Yule–Walker equations.\n4. Compute $\\psi_1$, $\\psi_2$ and $\\psi_3$.\n\n"
       "**Report:** the factorisation, two roots and six numbers."),
    code("# Solution\nar2_facts(0.7, -0.1)"),
    md("## A3 [Solved]: an MA(1) that is not invertible\n\n"
       "**Context.** $X_t = \\varepsilon_t + 2\\varepsilon_{t-1}$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$.\n\n"
       "1. Compute $\\gamma(0)$, $\\gamma(1)$ and $\\rho(1)$.\n2. Say whether the process is stationary and whether it is invertible.\n"
       "3. Find the invertible MA(1) with the same autocovariances.\n"
       "4. Write the first four weights of its AR($\\infty$) form and the forecast $\\hat X_{T+2}$.\n\n"
       "**Report:** five numbers, two verdicts and the new model."),
    code("# Solution\nma1_facts(theta=2.0, sigma2=1.0)"),
    md("**Interpretation of the result.** $\\theta = 2$ with $\\sigma^2 = 1$ and $\\theta = 0.5$ with $\\sigma^2 = 4$ have the same "
       "$\\gamma(0) = 5$ and $\\gamma(1) = 2$; only the second is invertible, so it is the one we report."),
    md("## A4 [Proposed]: an ARMA(1,1) process\n\n"
       "**Context.** $X_t = 0.5X_{t-1} + \\varepsilon_t + 0.3\\varepsilon_{t-1}$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$. Model: A1, A3.\n\n"
       "1. Check stationarity and invertibility.\n2. Compute $\\psi_1$, $\\psi_2$ and $\\psi_3$.\n"
       "3. Compute $\\gamma(0)$, $\\rho(1)$ and $\\rho(2)$.\n"
       "4. Show that $X_t = 0.5X_{t-1} + \\varepsilon_t - 0.5\\varepsilon_{t-1}$ is white noise, and say what an estimation program would report for it.\n\n"
       "**Report:** two verdicts, six numbers and one sentence."),
    code("# Solution\nprint(arma11_facts(0.5, 0.3))\n"
         "x = simulate_arma([0.5], [-0.5], n=300, rng=np.random.default_rng(1))\n"
         "r = fit_arma(x, 1, 1)\nprint(r.params.round(3), r.bse.round(3))"),
    md("## A5 [Solved]: Yule–Walker and information criteria\n\n"
       "**Context.** A stationary series has $T = 100$, $\\hat\\gamma(0) = 4$, $\\hat\\rho(1) = 0.75$, $\\hat\\rho(2) = 0.65$.\n\n"
       "1. Estimate an AR(2) by Yule–Walker: $\\hat\\phi_1$, $\\hat\\phi_2$, $\\hat\\sigma^2$.\n2. Check that the estimated model is stationary.\n"
       "3. Maximum likelihood gives $\\ln L = -214.6$ for AR(1) ($k = 3$) and $-210.9$ for AR(2) ($k = 4$); compute AIC and BIC.\n"
       "4. Choose a model with each criterion.\n\n**Report:** three estimates, four criteria and a choice."),
    code("# Solution\nprint(yw_ar2(0.75, 0.65, 4.0))\nprint(ic_table({'AR(1)': -214.6, 'AR(2)': -210.9}, {'AR(1)': 3, 'AR(2)': 4}, 100))"),
    md("**Interpretation of the result.** $\\hat\\phi = (0.6, 0.2)$, $\\hat\\sigma^2 = 1.68$; both AIC and BIC prefer AR(2), "
       "because the gain of 3.7 in $\\ln L$ exceeds the penalty of one more parameter."),
    md("## A6 [Proposed]: four candidates and the Ljung–Box test\n\n"
       "**Context.** $T = 120$; MLE gives $\\ln L$ = $-250.3$ (AR(1), $k = 3$), $-247.1$ (AR(2), $k = 4$), $-247.6$ (ARMA(1,1), $k = 4$), "
       "$-246.9$ (ARMA(2,1), $k = 5$). Model: A5.\n\n"
       "1. Compute AIC and BIC for the four models.\n2. Choose a model with each criterion.\n"
       "3. The residuals of AR(2) give $Q^*(10) = 14.2$; find the degrees of freedom and decide at 5%.\n"
       "4. Explain what changes if one uses 10 degrees of freedom.\n\n**Report:** a table of eight numbers, two choices and one decision."),
    code("# Solution\nprint(ic_table({'AR(1)': -250.3, 'AR(2)': -247.1, 'ARMA(1,1)': -247.6, 'ARMA(2,1)': -246.9},\n"
         "               {'AR(1)': 3, 'AR(2)': 4, 'ARMA(1,1)': 4, 'ARMA(2,1)': 5}, 120))\n"
         "print('p with 8 df:', stats.chi2.sf(14.2, 8), ' p with 10 df:', stats.chi2.sf(14.2, 10))"),
    # ---------------- Part B
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: US real GDP growth\n\n"
       "**Question.** Which ARMA model describes US quarterly real GDP growth, and how persistent are growth shocks?\n\n"
       "1. Plot $y_t = 400\\,\\Delta\\ln Y_t$ and its ACF and PACF up to lag 12, and propose candidate models.\n"
       "2. Estimate white noise, AR(1), AR(2), MA(2) and ARMA(1,1) by maximum likelihood, and tabulate AIC and BIC.\n"
       "3. Test the residuals of the BIC model with Ljung–Box $Q^*(8)$ on $8 - p - q$ degrees of freedom, and with Jarque–Bera.\n"
       "4. Interpretation: how many quarters does it take for half of a growth shock to fade?\n\n"
       "**Report:** the chart, a table of ten criteria, two test results and one sentence."),
    code("# Solution\nb1 = b1_us_gdp(save=False)\n"
         "print(pd.DataFrame({m: {'AIC': v['aic'], 'BIC': v['bic']} for m, v in b1['models'].items()}).T.round(2))\n"
         "print('BIC:', b1['bic_best'], ' AIC:', b1['aic_best'], ' LB:', b1['lb'], ' JB p:', b1['jb_p'])\n"
         "phi = b1['models']['ARMA(1,0)']['params'][1]\nprint('half-life (quarters):', np.log(0.5) / np.log(phi))"),
    md("**Interpretation of the result.** BIC chooses AR(1) with $\\hat\\phi \\approx 0.31$ (AIC: AR(2)); the residuals pass "
       "Ljung–Box. The half-life is below one quarter: shocks to the growth rate fade fast."),
    md("## B2 [Proposed]: Romanian real GDP growth\n\n"
       "**Question.** Does Romanian quarterly real GDP growth have any ARMA structure? Model: B1.\n\n"
       "1. Repeat the four steps of B1 for $y_t$ (`ro_gdp_growth('qoq')`).\n2. Run Ljung–Box $Q^*(8)$ on the series itself.\n"
       "3. Repeat the analysis without the four quarters of 2020, and compare.\n"
       "4. Interpretation: why is the best forecast of next quarter's growth simply the mean?\n\n"
       "**Report:** the chart, a table of ten criteria, the tests and two sentences."),
    code("# Solution\nb2 = b2_ro_gdp(save=False)\n"
         "print(pd.DataFrame({m: {'AIC': v['aic'], 'BIC': v['bic']} for m, v in b2['models'].items()}).T.round(2))\n"
         "print(b2['bic_best'], b2['lb_raw'], b2['ex2020']['lb_raw'])"),
    md("## B3 [Solved]: BET daily returns\n\n"
       "**Question.** Is the first-order autocorrelation of BET returns useful for forecasting?\n\n"
       "1. Estimate an AR(1) by maximum likelihood and report $\\hat\\phi$, its standard error and $t$-ratio, and $R^2$.\n"
       "2. Test the residuals with Ljung–Box $Q^*(10)$ on 9 degrees of freedom, and the squared residuals with $Q^*(10)$.\n"
       "3. Compute the kurtosis and the Jarque–Bera statistic of the residuals.\n4. Forecast tomorrow's return with its 95% interval.\n"
       "5. Interpretation: is the AR(1) coefficient statistically significant, and is it economically useful?\n\n"
       "**Report:** six numbers, the chart, a forecast with its interval and two sentences."),
    code("# Solution\nb3 = b3_returns('bet', fname='ch2_sem_b3', save=False)\nb3"),
    md("**Interpretation of the result.** $\\hat\\phi \\approx 0.11$ with $t \\approx 21$, but $R^2 \\approx 1\\%$: the forecast of "
       "tomorrow's return is tiny compared with its interval. The squared residuals are strongly autocorrelated (Chapter 5)."),
    md("## B4 [Proposed]: EUR/RON daily returns\n\n"
       "**Question.** Does the EUR/RON exchange rate behave like the BET? Model: B3.\n\n"
       "1. Repeat the four steps of B3 (`b3_returns('eurron')`).\n2. Compare $\\hat\\phi$, $R^2$ and the kurtosis with those of the BET.\n"
       "3. Interpretation: why can a managed exchange rate show a larger first-order autocorrelation than a stock index?\n\n"
       "**Report:** a table (two series, six numbers each) and two sentences."),
    code("# Solution\nb4 = b3_returns('eurron', fname='ch2_sem_b4', save=False)\n"
         "pd.DataFrame({'BET': b3, 'EUR/RON': b4}).loc[['phi', 't', 'r2', 'kurt', 'f1', 'lo', 'hi']]"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: does an AR model beat simple forecasts of inflation?\n\n"
       "**Question.** Is an AR(2) better than the historical mean and the last value at forecasting Romanian 12-month inflation "
       "one year ahead? Models: B1, A1.\n\n"
       "1. At each origin (every month from January 2015), estimate an AR(2) on all data up to that month and forecast 12 months ahead.\n"
       "2. Compute the same forecasts from the historical mean and from the last observed value.\n"
       "3. Compare RMSE and MAE of the three methods, over the full period and before July 2021.\n"
       "4. Interpretation: why do all three methods fail in 2022?\n\n**Report:** a table of RMSE and MAE, the chart and a plan for a project."),
    code("# Reference analysis\nc1_oos(save=False)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant answered questions about the ARMA models of this seminar:\n\n"
       "- (a) $X_t = 0.6X_{t-1} + 0.5X_{t-2} + \\varepsilon_t$ is stationary because both coefficients are below 1;\n"
       "- (b) an MA(1) with $\\theta = 2$ is not stationary, because $|\\theta| > 1$;\n"
       "- (c) the AR(1) for BET returns has $t = 21$, so yesterday's return explains most of today's return;\n"
       "- (d) for the residuals of an ARMA(1,1), Ljung–Box $Q(8)$ is compared with $\\chi^2(8)$;\n"
       "- (e) the forecast of an MA(2) three steps ahead equals the mean of the series;\n"
       "- (f) in an AR(1), $X_t = c + \\phi X_{t-1} + \\varepsilon_t$, the mean of the series is $c$.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nprint(np.abs(inverse_roots((0.6, 0.5))))\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 2, 'lecture')
    build(SEMINAR, 2, 'seminar')
