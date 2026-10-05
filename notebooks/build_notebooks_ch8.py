"""
build_notebooks_ch8.py -- lecture and seminar notebooks of Chapter 8 (TSA): long memory and ARFIMA
=================================================================================================
Output: notebooks/EN/chapter8_lecture_notebook.ipynb, notebooks/EN/chapter8_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_08/generate_all_charts.py and seminar8.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch8.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter8_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter8_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 8
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(8)
import generate_all_charts as g   # noqa: E402
import seminar8 as s              # noqa: E402
import build_quantlets as bq      # noqa: E402

CONSTS = '\n'.join(bq.CONSTS)
CORE = bq.CORE
INSTALL = bq.INSTALL

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Time Series Analysis — Chapter 8: Long memory and ARFIMA\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Short and long memory: exponential against hyperbolic decay of the ACF (Nile, inflation, volatility).\n"
       "- Fractional differencing $(1-L)^d$, fractional integration, the ARFIMA$(p,d,q)$ model, fBm and fGn.\n"
       "- Estimating $d$: R/S, DFA, GPH, local Whittle, exact ML (Durbin–Levinson) and Whittle; a Monte Carlo.\n"
       "- Forecasting with ARFIMA against AR; long memory in volatility (FIGARCH, HAR); spurious long memory.\n"
       "- The first code cell installs the `arch` package if it is missing (`pip install arch`).\n"
       "- References: Granger and Joyeux (1980); Hosking (1981); Geweke and Porter-Hudak (1983); Robinson (1995); "
       "Sowell (1992); Baillie (1996); Ding, Granger and Engle (1993); Diebold and Inoue (2001)."),
    *common_cells(INSTALL),
    md("## Definitions and helpers used in the whole notebook\n\n"
       "- `frac_weights(d, n)`: weights of $(1-L)^d$; `arfima_acvf(d, phi, theta, sigma2, n)`: ARFIMA autocovariances.\n"
       "- `gph(x, m)`, `local_whittle(x, m)`: semiparametric estimators with bandwidth $m = \\lfloor T^{0.65}\\rfloor$.\n"
       "- `arfima_exact_ml(x, p, q)`: exact Gaussian ML; `arfima_whittle(x, p, q)`: Whittle approximation.\n"
       "- `hurst_rs(x)`, `hurst_dfa(x)`: R/S and DFA exponents; `arfima_forecast(x, d, phi, theta, H)`: forecasts.\n"
       "- `statsmodels` has no ARFIMA class: every estimator is written out here."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Memory in data\n\n- Four persistent series and their ACF against an AR(1)."),
    code(src(g.fig_memory_data, g.fig_memory_acf) + "\n\n\nfig_memory_data(save_it=False)\nprint(fig_memory_acf(save_it=False)['infl'])"),
    md("## 2. Hyperbolic decay and fractional differencing\n\n- ARFIMA(0,0.4,0) against an AR(1); weights; fractional "
       "differencing of the log S&P 500."),
    code(src(g.worked_examples, g.fig_decay, g.fig_weights, g.fig_ffd)
         + "\n\n\nprint(worked_examples()['w04'])\nprint(fig_decay(save_it=False))\nfig_weights(save_it=False)\n"
           "ff = fig_ffd(save_it=False)\nprint('minimal d:', ff['dmin'], 'correlation:', round(ff['cor_dmin'], 3))"),
    md("## 3. ARFIMA processes, fBm and fGn"),
    code(src(g.fig_arfima_paths, g.fig_fbm) + "\n\n\nprint(fig_arfima_paths(save_it=False))\nprint(fig_fbm(save_it=False))"),
    md("## 4. Estimating d\n\n- R/S and DFA; GPH; local Whittle and the bandwidth; exact ML against ARMA; the table of "
       "twelve series."),
    code(src(g.fig_rs_dfa, g.fig_gph, g.fig_bandwidth, g.model_table, g.fig_arfima_fit, g.memory_table)
         + "\n\n\nprint(fig_rs_dfa(save_it=False))\nprint(fig_gph(save_it=False))\nfig_bandwidth(save_it=False)\n"
           "fit = fig_arfima_fit(save_it=False)\n"
           "print(pd.DataFrame(fit['table'])[['name', 'd', 'loglik', 'aic', 'bic']].round(3))\n"
           "print(pd.DataFrame(memory_table()).T[['n', 'rs', 'dfa', 'gph', 'lw', 'lw_se']].round(2))"),
    md("## 5. Monte Carlo of the estimators\n\n- 300 samples per design; it takes a few minutes. Reduce `reps` to go faster."),
    code(src(g.fig_mc) + "\n\n\nmc = fig_mc(reps=100, save_it=False)\n"
         "print({lab: {k: round(v[k]['mean'], 3) for k in ['GPH', 'LW', 'Whittle', 'R/S', 'DFA']} for lab, v in mc['res'].items()})"),
    md("## 6. Forecasting with ARFIMA"),
    code(src(g.fig_forecast_path, g.fig_forecast) + "\n\n\nprint(fig_forecast_path(save_it=False))\n"
         "fc = fig_forecast(save_it=False)\n"
         "print({k: [round(a / b, 3) for a, b in zip(fc[k]['rmse']['ARFIMA'][::6], fc[k]['rmse']['AR'][::6])] for k in ('ro', 'us')})"),
    md("## 7. Long memory in volatility\n\n- ACF of $r$, $|r|$, $r^2$; realised volatility; GARCH against FIGARCH; HAR."),
    code(src(g.fig_vol_acf, g.fig_rv, g.fig_vol_models)
         + "\n\n\nv = fig_vol_acf(save_it=False)\nprint({k: (round(v[k]['abs_lw'], 2), round(v[k]['abs_shuf_lw'], 2)) for k in ('sp500', 'bet')})\n"
           "print(fig_rv(save_it=False))\nvm = fig_vol_models(save_it=False)\nprint(vm['sp500']['figarch'], vm['har'])"),
    md("## 8. Spurious long memory\n\n- Breaks and regimes; the Nile in 1898; rolling estimates."),
    code(src(g.fig_spurious, g.fig_nile_break, g.fig_rolling)
         + "\n\n\nsp = fig_spurious(reps=100, save_it=False)\nprint(sp['break']['1.0'], sp['switch']['0.01'])\n"
           "print(fig_nile_break(save_it=False))\nro = fig_rolling(save_it=False)\nprint(ro['us'])"),
    md("## Exercises\n\n1. Estimate $d$ of Romanian monthly inflation on 2005–2019 only: does the 2021–2023 surge change it?\n"
       "2. Fit FIGARCH(1,d,1) to EUR/RON returns (`returns('eurron')`) and compare its BIC with GARCH(1,1).\n"
       "3. Simulate ARFIMA(1,0.3,0) with $\\phi = 0.5$ (`sim_arfima(1000, 0.3, rng, phi=[0.5])`) and compare GPH, local "
       "Whittle and exact ML of ARFIMA(1,d,0)."),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_FUNCS = [s.weights_by_hand, s.hurst_from_points, s.gph_by_hand, s.forecast_by_hand, s.figarch_by_hand, s.b1_nile,
             s.b3_sp500]
SEMINAR = [
    md("# Time Series Analysis — Seminar 8: Long memory and ARFIMA\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The seminar comes before Lecture 8; the slides give the definitions (\"What you need today\").\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class.\n"
       "- The first code cell installs the `arch` package if it is missing."),
    *common_cells(INSTALL),
    md("## Seminar functions\n\n- Helpers of Chapter 8 (`frac_weights`, `gph`, `local_whittle`, `hurst_rs`, "
       "`arfima_exact_ml`, `monthly_rv`) and the seminar functions: `weights_by_hand`, `hurst_from_points`, `gph_by_hand`, "
       "`forecast_by_hand`, `figarch_by_hand`, `b1_nile`, `b3_sp500`."),
    code(CONSTS + "\nSUBS = " + repr(s.SUBS) + "\nREGIMES = " + repr(s.REGIMES) + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_FUNCS)),
    code("# [solutions only]\n" + src(s.b2_romania, s.b4_bet_eurron, s.c1_us_regimes, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: fractional weights and the ACF\n\n"
       "**Context.** ARFIMA$(0, 0.25, 0)$ with $\\sigma^2 = 1$.\n\n"
       "1. Compute $\\pi_1, \\dots, \\pi_4$ of $(1-L)^{0.25}$.\n2. Compute $\\rho(1)$ and $\\rho(2)$, and give $H$ and the class.\n"
       "3. Compare $\\rho(5)$ with that of an AR(1) with the same $\\rho(1)$.\n\n"
       "**Report:** four weights, two autocorrelations, $H$, two values of $\\rho(5)$."),
    code("# Solution\nweights_by_hand(0.25)"),
    md("**Interpretation of the result.** $\\rho(1) = 1/3$, $\\rho(2) = 0.238$, $H = 0.75$: stationary long memory; "
       "$\\rho(5) = 0.151$ against $0.004$ for the AR(1)."),
    md("## A2 [Proposed]: which range is $d$ in?\n\n"
       "**Context.** (a) ARFIMA$(0, 0.6, 0)$; (b) ARFIMA$(0, -0.3, 0)$. Model: A1.\n\n"
       "1. Compute $\\pi_1$ and $\\pi_2$ of $(1-L)^d$ for each.\n2. Say whether each series is stationary, invertible and "
       "mean-reverting.\n3. For (b) compute $\\rho(1)$ and $\\rho(2)$; for (a) compute $\\rho(1)$ of the first difference.\n\n"
       "**Report:** four weights, two classifications, three autocorrelations."),
    code("# Solution\nprint(weights_by_hand(0.6))\nprint(weights_by_hand(-0.3))\nprint(weights_by_hand(-0.4)['rho1'])"),
    md("## A3 [Solved]: R/S by hand and $H$ from three points\n\n"
       "**Context.** $x = (0.6, 0.4, 0.5, 0.3, -0.4, -0.6, -0.2, -0.5)$; $(R/S)_n = 4.1, 9.6, 22.4$ for $n = 16, 64, 256$.\n\n"
       "1. Compute $\\bar x$, $Y_k$, $R$, $S$ and $R/S$.\n2. Compute $H$ as the slope of $\\log_{10}(R/S)_n$ on "
       "$\\log_{10}n$, and $d = H - 0.5$.\n\n**Report:** $R/S$, $H$, $d$ and one sentence."),
    code("# Solution\nprint(rs_steps([0.6, 0.4, 0.5, 0.3, -0.4, -0.6, -0.2, -0.5]))\n"
         "print(hurst_from_points([16, 64, 256], [4.1, 9.6, 22.4]))"),
    md("**Interpretation of the result.** $R/S = 3.83$ (grouped signs give a large range); $H = 0.61$, $d = 0.11$."),
    md("## A4 [Proposed]: a GPH slope by hand\n\n"
       "**Context.** $X_j = 0.5, 1.6, 2.3, 3.0$; $\\log I(\\lambda_j) = 1.2, 1.8, 2.0, 2.5$. Model: A3.\n\n"
       "1. Compute $\\bar X$, $\\bar y$, $S_{xy}$ and $S_{xx}$.\n2. Compute $\\hat d$ and its standard error $\\pi/\\sqrt{24m}$.\n"
       "3. Test $H_0$: $d = 0$ at 5% and comment on the size of $m$.\n\n"
       "**Report:** four sums, $\\hat d$, its SE, a decision and one sentence."),
    code("# Solution\ngph_by_hand([0.5, 1.6, 2.3, 3.0], [1.2, 1.8, 2.0, 2.5])"),
    md("## A5 [Solved]: a one-step ARFIMA forecast\n\n"
       "**Context.** ARFIMA$(0, 0.4, 0)$, $\\mu = 2$; $x_T = 3.0$, $x_{T-1} = 2.5$, $x_{T-2} = 2.8$, $x_{T-3} = 1.5$.\n\n"
       "1. Write the first four weights $-\\pi_k$.\n2. Compute $\\hat x_{T+1}$ with these four lags.\n"
       "3. Compare with an AR(1) with the same $\\rho(1)$.\n\n**Report:** four weights, two forecasts and one sentence."),
    code("# Solution\nforecast_by_hand(0.4, 2.0, [3.0, 2.5, 2.8, 1.5])"),
    md("**Interpretation of the result.** ARFIMA: 2.49; AR(1) with $\\phi = 0.667$: 2.67. The AR(1) uses only $x_T$; "
       "ARFIMA spreads the weight over the past."),
    md("## A6 [Proposed]: how long does a volatility shock count?\n\n"
       "**Context.** GARCH(1,1) with $\\alpha = 0.10$, $\\beta = 0.88$; FIGARCH$(0, 0.4, 0)$. Model: A5.\n\n"
       "1. Compute the weight of $\\varepsilon_{t-k}^2$ in $\\sigma_t^2$ for $k = 1, 2, 3$ in both models.\n"
       "2. Compute both weights at $k = 20$ and $k = 100$.\n3. Explain which model keeps a crisis longer.\n\n"
       "**Report:** ten weights and one sentence."),
    code("# Solution\nfigarch_by_hand(0.10, 0.88, 0.4)"),
    # ---------------- Part B
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: does the Nile have long memory?\n\n"
       "**Question.** Is the persistence of the Nile flow long memory, or the effect of the change in 1898?\n\n"
       "1. Plot the series and its ACF up to lag 20.\n2. Estimate $H$ by R/S, $d$ by GPH, local Whittle and exact ML of "
       "ARFIMA$(0,d,0)$.\n3. Subtract the means of 1871–1898 and 1899–1970 and repeat step 2.\n"
       "4. Interpretation: does the Nile have long memory?\n\n"
       "**Report:** two ACF values, four estimates before and after, one sentence."),
    code("# Solution\nb1 = b1_nile(save_it=False)\nprint(pd.DataFrame({k: b1[k] for k in ('raw', 'adj')}).round(3))"),
    md("**Interpretation of the result.** Raw: $d \\approx 0.36$–$0.44$; with the regime means removed: $d \\approx 0$. "
       "For 1871–1970 the evidence of long memory comes from the 1898 break."),
    md("## B2 [Proposed]: long memory in Romanian inflation\n\n"
       "**Question.** How persistent is Romanian inflation, and does the way we measure it change the answer? Model: B1.\n\n"
       "1. Estimate $d$ of the monthly SA rate (`ro_inflation()`) by GPH and local Whittle with $m = \\lfloor T^a\\rfloor$, "
       "$a = 0.5, 0.65, 0.8$.\n2. Fit ARFIMA$(0,d,0)$ and ARMA(1,1) by exact ML and compare their BIC.\n"
       "3. Estimate $d$ of the 12-month rate (`ro_inflation12()`) and compare the two ACFs.\n"
       "4. Interpretation: why does the 12-month rate give $d$ close to 1?\n\n"
       "**Report:** six estimates, two BIC values, one $d$, two sentences."),
    code("# Solution\nb2_romania(save_it=False)"),
    md("## B3 [Solved]: memory in S&P 500 volatility\n\n"
       "**Question.** Where is the memory of stock returns: in the returns or in their size?\n\n"
       "1. Estimate $d$ of $r_t$ and $|r_t|$ by local Whittle and GPH.\n2. Shuffle the days at random and re-estimate $d$ "
       "of $|r_t|$; plot both ACFs.\n3. Estimate $d$ of the monthly log realised volatility.\n"
       "4. Interpretation: what does the shuffle test show?\n\n**Report:** eight estimates, the chart and one sentence."),
    code("# Solution\nb3 = b3_sp500(save_it=False)\nprint({k: b3[k] for k in ('r', 'abs', 'shuf', 'rv')})"),
    md("**Interpretation of the result.** $d \\approx 0$ for $r_t$, about 0.53 for $|r_t|$ and 0 after the shuffle: the "
       "memory is in the order of calm and turbulent days, not in the distribution."),
    md("## B4 [Proposed]: BET and EUR/RON\n\n"
       "**Question.** Do the BET and the EUR/RON rate show the same memory, and is it stable before and after 2008? Model: B3.\n\n"
       "1. Estimate $d$ of $r_t$ and $|r_t|$ by local Whittle on the full sample.\n2. Repeat on the subsamples up to 2007 and "
       "2010–2026.\n3. Compare the estimates for $r_t$ with the 95% Monte Carlo band of i.i.d. series of 1900 days.\n"
       "4. Interpretation: is the full-sample memory of BET returns real?\n\n"
       "**Report:** twelve estimates, one band and two sentences."),
    code("# Solution\nb4_bet_eurron(save_it=False)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: US inflation and monetary regimes\n\n"
       "**Question.** Is the long memory of US inflation a property of inflation or of the changes in monetary policy? "
       "Models: B1, B2.\n\n"
       "1. Estimate $d$ by local Whittle on the full sample and in 1947–1984, 1985–2019, 2020–2026.\n"
       "2. Remove the three regime means and re-estimate $d$.\n3. Propose one more check.\n"
       "4. Interpretation: can a constant $d$ describe 80 years of inflation?\n\n"
       "**Report:** five estimates, one check and a project plan."),
    code("# Reference analysis\nc1_us_regimes(save_it=False)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant answered:\n\n"
       "- (a) monthly inflation has d = 0.31, so H = d - 0.5 < 0: anti-persistent;\n"
       "- (b) the 12-month rate gives d = 1.00, so Romanian inflation has a unit root;\n"
       "- (c) with d = 0.31 the series is stationary and its shocks die out;\n"
       "- (d) use `ARIMA(x, order=(0, 0.31, 0))` from statsmodels to fit ARFIMA;\n"
       "- (e) S&P 500 returns have d = -0.01 and |r| has d = 0.53, so returns are predictable;\n"
       "- (f) a significant GPH estimate proves long memory.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 8, 'lecture')
    build(SEMINAR, 8, 'seminar')
