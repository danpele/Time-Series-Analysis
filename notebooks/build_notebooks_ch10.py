"""
build_notebooks_ch10.py -- lecture and seminar notebooks of Chapter 10 (TSA): state space, Kalman filter, Markov switching
=========================================================================================================================
Output: notebooks/EN/chapter10_lecture_notebook.ipynb, notebooks/EN/chapter10_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_10/generate_all_charts.py and seminar10.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch10.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter10_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter10_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 10
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(10)
import generate_all_charts as g   # noqa: E402
import seminar10 as s             # noqa: E402
import build_quantlets as bq      # noqa: E402

CONSTS = '\n'.join(bq.CONSTS)
CORE = bq.CORE + bq.UC + bq.MS + [g.tvp_filter, g.tvp_data, g.coincident, g.ms_vol, g.local_whittle, g.simulate_ms]
INSTALL = bq.INSTALL

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Time Series Analysis — Chapter 10: State space models, Kalman filter and Markov switching\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The state space form: local level, local linear trend, ARMA, time-varying regression.\n"
       "- A Kalman filter written out in numpy: prediction and update, the gain, the steady state and exponential smoothing; "
       "the Rauch–Tung–Striebel smoother, the likelihood, diagnostics, missing values and forecasts (the Nile).\n"
       "- Trend and cycle of Romanian GDP: unobserved components, the HP filter and the Hamilton (2018) filter; a time-varying "
       "beta; a dynamic factor model.\n"
       "- Markov switching (Hamilton 1989): US recessions against the NBER dates, Romanian growth regimes, volatility regimes, "
       "and the links with GARCH and long memory.\n"
       "- The first code cell installs the `arch` package if it is missing (`pip install arch`).\n"
       "- References: Kalman (1960); Durbin and Koopman (2012); Harvey (1989); Hamilton (1989, 1994); Kim and Nelson (1999)."),
    *common_cells(INSTALL),
    md("## Definitions and helpers used in the whole notebook\n\n"
       "- `kalman_filter(y, Z, T, H, Q, a1, P1)`: the filter (missing values as NaN); `rts_smoother(kf)`; "
       "`disturbance_smoother(kf)`.\n"
       "- `local_level_ml(y)`: maximum likelihood of the local level model; `steady_state(q)`: steady-state gain.\n"
       "- `uc_fit`, `hp`, `hamilton_filter`: trend and cycle; `ms_fit`, `ms_summary`, `concordance`: Markov switching."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. The state space form\n\n- Simulated local level and local linear trend; AR(2) in state space form."),
    code(src(g.fig_ss_examples, g.arma_ss_check) + "\n\n\nprint(fig_ss_examples(save_it=False))\nprint(arma_ss_check())"),
    md("## 2. The Kalman filter\n\n- The three-period example; the Nile; the gain and exponential smoothing."),
    code(src(g.fig_nile_filter, g.statsmodels_local_level, g.fig_gain_ses)
         + "\n\n\nprint(pd.DataFrame(worked_example()['rows']).round(3))\nnf = fig_nile_filter(save_it=False)\n"
           "print({k: round(nf[k], 3) for k in ('s2_eps', 's2_eta', 'q', 'ss_K', 'loglik')})\n"
           "print(statsmodels_local_level())\nprint(fig_gain_ses(save_it=False))"),
    md("## 3. Smoothing, likelihood and diagnostics"),
    code(src(g.fig_nile_smooth, g.fig_likelihood, g.fig_diagnostics)
         + "\n\n\nprint(fig_nile_smooth(save_it=False))\nprint(fig_likelihood(save_it=False))\nprint(fig_diagnostics(save_it=False))"),
    md("## 4. Missing values and forecasts"),
    code(src(g.fig_missing) + "\n\n\nprint(fig_missing(save_it=False))"),
    md("## 5. Trend and cycle of Romanian GDP\n\n- UC model, HP filter, Hamilton filter; real time against hindsight."),
    code(src(g.fig_ro_trend, g.fig_output_gap, g.fig_realtime)
         + "\n\n\nprint(fig_ro_trend(save_it=False))\nprint(fig_output_gap(save_it=False))\nprint(fig_realtime(save_it=False))"),
    md("## 6. A time-varying beta and a dynamic factor"),
    code(src(g.fig_tvp_beta, g.fig_dfm) + "\n\n\nprint(fig_tvp_beta(save_it=False))\nprint(fig_dfm(save_it=False))"),
    md("## 7. Markov switching: US and Romanian GDP"),
    code(src(g.fig_ms_us, g.fig_ms_filtered, g.ms_us_pitfalls, g.fig_ms_ro)
         + "\n\n\nmu = fig_ms_us(save_it=False)\nprint(mu['ham'], mu['ham_b'], mu['ext'], mu['ext_conc'], sep='\\n')\n"
           "print(fig_ms_filtered(save_it=False))\npit = ms_us_pitfalls()\nprint(pit['sv'], pit['all'], sep='\\n')\n"
           "print(fig_ms_ro(save_it=False))"),
    md("## 8. Volatility regimes, GARCH and long memory\n\n- The simulation takes about a minute; reduce `reps` to go faster."),
    code(src(g.fig_vol_regimes, g.fig_regimes_memory)
         + "\n\n\nprint(fig_vol_regimes(save_it=False))\nprint(fig_regimes_memory(save_it=False, reps=20))"),
    md("## Exercises\n\n1. Fit a local linear trend (`UnobservedComponents(y, 'lltrend')`) to the Nile and compare its BIC with the local level.\n"
       "2. Estimate the Romanian output gap with a damped stochastic cycle (`cycle=True, stochastic_cycle=True, damped_cycle=True`) and compare it with the AR(2) cycle.\n"
       "3. Fit a three-regime volatility model to weekly S&P 500 returns and compare its BIC with two regimes."),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_FUNCS = [s.kalman_by_hand, s.ses_equivalence, s.state_space_forecasts, s.durations, s.hamilton_step,
             s.ro_monthly_inflation, s.b1_inflation_level, s.us_log_gdp, s.b3_ip_regimes]
SEMINAR = [
    md("# Time Series Analysis — Seminar 10: State space models, Kalman filter and Markov switching\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The seminar comes before Lecture 10; the slides give the definitions (\"What you need today\").\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class.\n"
       "- The first code cell installs the `arch` package if it is missing."),
    *common_cells(INSTALL),
    md("## Seminar functions\n\n- Helpers of Chapter 10 (`kalman_filter`, `local_level_ml`, `steady_state`, `uc_fit`, `hp`, "
       "`hamilton_filter`, `ms_fit`, `ms_summary`, `concordance`) and the seminar functions: `kalman_by_hand`, "
       "`ses_equivalence`, `state_space_forecasts`, `durations`, `hamilton_step`, `b1_inflation_level`, `b3_ip_regimes`."),
    code(CONSTS + f"\nHICP = {s.HICP!r}\nINFL_START = {s.INFL_START!r}\nIP_SAMPLE = {s.IP_SAMPLE!r}" + '\n\n\n' + src(*CORE)
         + '\n\n\n' + src(*SEM_FUNCS)),
    code("# [solutions only]\n" + src(s.b2_us_gap, s.b4_bet_regimes, s.c1_ro_inflation_regimes, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: the Kalman filter for three periods\n\n"
       "**Context.** Local level with $\\sigma^2_\\varepsilon = 1$, $\\sigma^2_\\eta = 1$; $a_1 = 0$, $P_1 = 1$; $y = (1, 3, 2)$.\n\n"
       "1. For $t = 1, 2, 3$ compute $F_t$, $K_t$, $v_t$, $a_{t|t}$, $P_{t|t}$ and the next prediction.\n"
       "2. Give the forecast of $y_4$ and its variance.\n3. Compute the steady-state gain for $q = 1$.\n\n"
       "**Report:** a table of the three steps, one forecast, $\\bar K$."),
    code("# Solution\nr = kalman_by_hand([1.0, 3.0, 2.0], 0.0, 1.0, 1.0, 1.0)\nprint(pd.DataFrame(r['rows']).round(3))\n"
         "print('a4 =', r['a_next'], ' P4 =', round(r['P_next'], 3), ' steady-state K =', round(r['ss_K'], 3))"),
    md("**Interpretation of the result.** The gain rises from 0.5 to 0.615, close to the steady state 0.618; the forecast "
       "of $y_4$ is 2 with variance $1.615 + 1$."),
    md("## A2 [Proposed]: a missing observation\n\n"
       "**Context.** $\\sigma^2_\\varepsilon = 2$, $\\sigma^2_\\eta = 0.5$; $a_1 = 5$, $P_1 = 2$; $y = (6, \\text{missing}, 4)$. Model: A1.\n\n"
       "1. Run the filter for $t = 1, 2, 3$, skipping the update at $t = 2$.\n2. Explain why $K_3$ equals $K_1$.\n"
       "3. Compute $\\bar K$ for this $q$ and the equivalent SES weight.\n\n**Report:** the three steps, one sentence, $\\bar K$."),
    code("# Solution\nr = kalman_by_hand([6.0, None, 4.0], 5.0, 2.0, 2.0, 0.5)\nprint(pd.DataFrame(r['rows']))\nprint(r['ss_K'])"),
    md("## A3 [Solved]: exponential smoothing is a steady-state Kalman filter\n\n"
       "1. Show that $a_{t+1} = a_t + \\bar K(y_t - a_t)$ is SES.\n2. Compute $\\bar K$ for $q = 1$ and $q = 0.25$.\n"
       "3. An SES model has $\\alpha = 0.3$: find $q$.\n\n**Report:** one line of algebra, two gains, one $q$."),
    code("# Solution\nprint(ses_equivalence())\nprint('q for alpha = 0.3:', round(q_from_alpha(0.3), 4))"),
    md("**Interpretation of the result.** $\\bar K = 0.618$ for $q = 1$ and $0.390$ for $q = 0.25$; $\\alpha = 0.3$ means "
       "$q = 0.129$: a level that moves little relative to the noise."),
    md("## A4 [Proposed]: state space forms and forecasts\n\n"
       "**Context.** AR(2) with $\\phi_1 = 0.5$, $\\phi_2 = 0.3$; a local linear trend. Model: A1.\n\n"
       "1. Write $Z$, $T$, $R$, $Q$, $H$ for the AR(2).\n2. From $(y_T, y_{T-1}) = (2, 1)$ forecast $h = 1, 2, 3$.\n"
       "3. Write $Z$, $T$ of the local linear trend and forecast three steps from $\\mu_T = 100$, $\\beta_T = 2$.\n\n"
       "**Report:** two sets of matrices and six forecasts."),
    code("# Solution\nstate_space_forecasts()"),
    md("## A5 [Solved]: durations of recessions and expansions\n\n"
       "**Context.** $p_{11} = 0.75$ (recession), $p_{22} = 0.95$ (expansion).\n\n"
       "1. Write the transition matrix and compute the expected durations.\n2. Compute the ergodic probabilities.\n"
       "3. From a recession, compute the probability of a recession two quarters later.\n\n"
       "**Report:** two durations, two probabilities, one two-step probability."),
    code("# Solution\ndurations(0.75, 0.95)"),
    md("**Interpretation of the result.** Recessions last 4 quarters and expansions 20 on average; one quarter in six is a "
       "recession quarter; after two quarters a recession continues with probability 0.575."),
    md("## A6 [Proposed]: one step of the Hamilton filter\n\n"
       "**Context.** The model of A5 with $\\mu_1 = -0.5$, $\\mu_2 = 1$, $\\sigma = 1$; $\\Pr(S_{t-1} = 1 \\mid Y_{t-1}) = 0.2$; "
       "$y_t = -1$. Model: A5.\n\n1. Compute the predicted probability.\n2. Compute the densities and the filtered probability.\n"
       "3. Give the likelihood contribution and next quarter's predicted probability.\n\n"
       "**Report:** four probabilities and one log-density."),
    code("# Solution\nhamilton_step(0.2, 0.75, 0.95, -1.0, -0.5, 1.0, 1.0)"),
    # ---------------- Part B
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: underlying Romanian inflation\n\n"
       "**Question.** How fast does the underlying level of Romanian inflation move, and which SES weight does the data choose?\n\n"
       "1. Fit the local level model by maximum likelihood.\n2. Compute $\\bar K$ and compare it with the SES weight estimated "
       "directly.\n3. Plot the filtered and smoothed level ($\\times 12$) and report its peak and last value.\n"
       "4. Interpretation: what does the size of $\\hat q$ say about monthly inflation?\n\n"
       "**Report:** three variances, two weights, two levels, one sentence."),
    code("# Solution\nb1 = b1_inflation_level(save_it=False)\nprint({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b1.items()})"),
    md("**Interpretation of the result.** $q \\approx 0.04$: most monthly movements are noise; $\\bar K \\approx 0.18$, close "
       "to the SES weight 0.175; the underlying inflation peaked at about 12% in 2022."),
    md("## B2 [Proposed]: the US output gap\n\n"
       "**Question.** How close are three statistical output gaps to the CBO gap, and which one would you use in real time? "
       "Model: B1.\n\n1. Compute the UC cycle, the HP gap and the Hamilton filter of $100\\log$ GDP (`us_log_gdp()`).\n"
       "2. Compute the CBO gap $100(\\mathrm{GDP}/\\mathrm{GDPPOT} - 1)$ (FRED GDPPOT) and the correlations, including the "
       "filtered UC cycle.\n3. Report the standard deviations and the latest values.\n"
       "4. Interpretation: which measure would you use to judge the economy today?\n\n"
       "**Report:** five correlations, four standard deviations, four latest values, one sentence."),
    code("# Solution\nb2_us_gap(save_it=False)"),
    md("## B3 [Solved]: recessions in US industrial production\n\n"
       "**Question.** Does a two-regime model of industrial production find the NBER recessions?\n\n"
       "1. Fit a two-regime model with switching mean and variance.\n2. Report the regime means and standard deviations, "
       "$p_{11}$, $p_{22}$ and the durations.\n3. Classify each month by the smoothed and by the filtered probability and "
       "compare with the NBER months.\n4. Interpretation: does the recession regime match the NBER recessions?\n\n"
       "**Report:** six parameters, two durations, two concordance tables, one sentence."),
    code("# Solution\nb3 = b3_ip_regimes(save_it=False)\nprint({k: v for k, v in b3.items()})"),
    md("**Interpretation of the result.** The recession regime (mean $-0.16\\%$, volatile) lasts about 8 months; it finds "
       "69% of the NBER months with 7% false alarms (smoothed), fewer with the filtered probabilities."),
    md("## B4 [Proposed]: volatility regimes of the BET\n\n"
       "**Question.** Does the Bucharest market have calm and turbulent regimes, and are they those of the S&P 500? Model: B3.\n\n"
       "1. Fit a two-regime model with switching mean and variance to weekly BET returns; report the annualised volatilities "
       "and durations.\n2. List the three years with the largest share of turbulent weeks.\n"
       "3. Fit a GARCH(1,1) and correlate its volatility with the regime-implied volatility.\n"
       "4. Interpretation: are the turbulent periods of the BET those of the S&P 500?\n\n"
       "**Report:** two volatilities, two durations, three years, two correlations, one sentence."),
    code("# Solution\nb4_bet_regimes(save_it=False)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: regimes of Romanian inflation\n\n"
       "**Question.** How many inflation regimes has Romania had since 1996, and when did the low-inflation regime begin? "
       "Model: B3.\n\n1. Fit models with two and three regimes and compare them by BIC.\n"
       "2. For three regimes, report the means, durations and the date when the low regime becomes permanent.\n"
       "3. Propose one more check.\n4. Interpretation: did inflation targeting start the low-inflation regime?\n\n"
       "**Report:** six parameters, a date, a check and a project plan."),
    code("# Reference analysis\nc1_ro_inflation_regimes(save_it=False)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant answered:\n\n"
       "- (a) for the Nile, q = 0.097, so the SES weight is alpha = q = 0.097;\n"
       "- (b) the Kalman gain is K = sigma2_eps / (P + sigma2_eps): noisier data get more weight;\n"
       "- (c) with p11 = 0.75, recessions last p11 / (1 - p11) = 3 quarters on average;\n"
       "- (d) to call a recession in real time, use `smoothed_marginal_probabilities`;\n"
       "- (e) the last value of the HP gap is a reliable estimate of today's output gap;\n"
       "- (f) in the local level model the forecasts of all horizons are equal; only their variance grows.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 10, 'lecture')
    build(SEMINAR, 10, 'seminar')
