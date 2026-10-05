"""
build_notebooks_ch1.py -- lecture and seminar notebooks of Chapter 1 (TSA): stochastic processes and stationarity
================================================================================================================
Output: notebooks/EN/chapter1_lecture_notebook.ipynb, notebooks/EN/chapter1_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_01/generate_all_charts.py and seminar1.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch1.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter1_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter1_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 1
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(1)
import generate_all_charts as g   # noqa: E402
import seminar1 as s              # noqa: E402

CONSTS = ('from statsmodels.tsa.stattools import acf, pacf\nfrom statsmodels.stats.diagnostic import acorr_ljungbox\n'
          f'SEED = {g.SEED!r}\nGDP_REAL = {g.GDP_REAL!r}\nGDP_NOMINAL = {g.GDP_NOMINAL!r}\nHICP = {g.HICP!r}\n'
          f'START = {g.START!r}\nBAND_COL = st.IDAred\nNAME = {g.NAME!r}')
CORE = [g.ro_gdp, g.ro_hicp, g.sample_acf, g.sample_pacf, g.ljung_box, g.acf_bars, g.simulate_arma, g.theoretical_acf,
        g.save]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Time Series Analysis — Chapter 1: Stochastic processes and stationarity\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Four real series: Romanian GDP and inflation, EUR/RON, BET returns.\n"
       "- Stochastic processes by simulation: ensembles, stationarity, white noise, the random walk.\n"
       "- The lag operator, differencing, the Wold decomposition, ergodicity.\n"
       "- Sample ACF and PACF with confidence bands; the Box–Pierce and Ljung–Box tests.\n"
       "- Transformations: log, differences, Box–Cox (Guerrero's lambda); over-differencing; the Nile and the sunspots.\n"
       "- References: Huang and Petukhina (2022), *Applied Time Series Analysis and Forecasting with Python*, Ch. 1–2; "
       "Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.), Ch. 2–3; Brockwell and Davis (2016), Ch. 1."),
    *common_cells(),
    md("## Definitions used in the whole notebook\n\n"
       "- Sample ACF with divisor $T$: $\\hat\\rho(h) = \\sum_{t=1}^{T-h}(x_t - \\bar x)(x_{t+h} - \\bar x)/\\sum_{t=1}^{T}(x_t - \\bar x)^2$; "
       "95% band $\\pm 1.96/\\sqrt{T}$.\n"
       "- Ljung–Box: $Q^*(m) = T(T+2)\\sum_{h=1}^{m}\\hat\\rho(h)^2/(T-h)$, approximately $\\chi^2(m)$ under white noise.\n"
       "- Helpers: `sample_acf`, `sample_pacf`, `ljung_box`, `acf_bars` (a correlogram with its band), `simulate_arma`."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Four series of this course\n\n- Romanian real GDP (Eurostat), HICP inflation (Eurostat), EUR/RON (BNR), BET daily log returns."),
    code(src(g.fig_four_series) + "\n\n\nfig_four_series(save_it=False)"),
    md("## 2. An ensemble of paths\n\n- 40 paths of a stationary AR(1) and of a random walk; the band is $\\pm 1.96$ standard deviations of $X_t$."),
    code(src(g.fig_ensemble) + "\n\n\nfig_ensemble(save_it=False)"),
    md("## 3. Stationarity\n\n- Four series that violate stationarity in different ways; a process that is weakly but not strictly stationary."),
    code(src(g.fig_nonstationary, g.fig_counterexample) + "\n\n\nfig_nonstationary(save_it=False)\nfig_counterexample(save_it=False)"),
    md("## 4. White noise and the random walk\n\n- Gaussian, Laplace and GARCH(1,1) white noise: the ACF of the squares tells them apart.\n"
       "- Random walks: the variance grows like $t\\sigma^2$.\n- Which panel is the BET?"),
    code(src(g.simulate_garch, g.fig_white_noise, g.fig_random_walk, g.fig_spot_real)
         + "\n\n\nfig_white_noise(save_it=False)\nfig_random_walk(save_it=False)\nspot = fig_spot_real(save_it=False)"),
    code("# the answer\nspot"),
    md("## 5. The lag operator, differencing, Wold and ergodicity\n\n"
       "- S&P 500: the log price and its difference.\n- Wold weights $\\psi_j$ of four processes.\n"
       "- Running time averages of an ergodic and a non-ergodic process."),
    code(src(g.fig_sp500_diff, g.fig_wold, g.fig_ergodicity)
         + "\n\n\nfig_sp500_diff(save_it=False)\nfig_wold(save_it=False)\nfig_ergodicity(save_it=False)"),
    md("## 6. Sample ACF and PACF\n\n- Theoretical and sample ACF; the sampling distribution under white noise (Bartlett); "
       "ACF and PACF of AR(1), AR(2), MA(1); BET returns, absolute and squared returns."),
    code(src(g.fig_acf_models, g.fig_bartlett, g.fig_acf_pacf, g.fig_returns_acf)
         + "\n\n\nfig_acf_models(save_it=False)\nfig_bartlett(save_it=False)\nfig_acf_pacf(save_it=False)\nfig_returns_acf('bet', save_it=False)"),
    md("## 7. Portmanteau tests\n\n- Size of Box–Pierce and Ljung–Box in small samples; Ljung–Box statistics of real series."),
    code(src(g.lb_size, g.fig_lb_size, g.real_series, g.fig_lb_stats)
         + "\n\n\nfig_lb_size(save_it=False)\nT = fig_lb_stats(save_it=False)\n"
           "pd.DataFrame({k: {'T': v['n'], 'rho1': v['r1'], 'rho4': v['r4'], 'Q*(10)': v['m10']['lb'], 'p': v['m10']['lb_p']} "
           "for k, v in T.items()}).T.round(4)"),
    md("## 8. Transformations\n\n- Romanian real GDP: log, quarterly and annual growth; Box–Cox with Guerrero's lambda for nominal GDP; "
       "over-differencing."),
    code(src(g.fig_gdp_transform, g.boxcox, g.guerrero_cv, g.guerrero_lambda, g.fig_boxcox, g.fig_overdiff)
         + "\n\n\nfig_gdp_transform(save_it=False)\nfig_boxcox(save_it=False)\nfig_overdiff(save_it=False)"),
    md("## 9. Two classic series: the Nile and the sunspots"),
    code(src(g.fig_textbook) + "\n\n\nfig_textbook(save_it=False)"),
    md("## Exercises\n\n1. Repeat the analysis of Section 6 for the DAX (`log_returns('dax')`): is the first-order autocorrelation "
       "positive or negative?\n2. Compute Guerrero's lambda for Romanian real GDP (`ro_gdp('real')`) and compare it with nominal GDP.\n"
       "3. Split the Nile series at 1898 and run Ljung–Box on each half."),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_FUNCS = [s.ma_acvf, s.rw_moments, s.trend_diff, s.acf_by_hand, s.portmanteau, s.b1_returns, s.b3_gdp]
SEMINAR = [
    md("# Time Series Analysis — Seminar 1: Stochastic processes and stationarity\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The seminar comes before Lecture 1; the slides give the definitions (\"What you need today\").\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class."),
    *common_cells(),
    md("## Seminar functions\n\n- Helpers of Chapter 1 (`sample_acf`, `ljung_box`, `acf_bars`, `ro_gdp`, `ro_hicp`) and the seminar "
       "functions: `ma_acvf`, `rw_moments`, `trend_diff`, `acf_by_hand`, `portmanteau`, `b1_returns`, `b3_gdp`."),
    code(CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_FUNCS)),
    code("# [solutions only]\n" + src(s.b4_hicp, s.c1_persistence, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: an MA(1) process\n\n"
       "**Context.** $X_t = \\varepsilon_t + 0.5\\,\\varepsilon_{t-1}$, with $\\varepsilon_t \\sim \\mathrm{WN}(0, 2)$.\n\n"
       "1. Compute the mean $E[X_t]$.\n2. Compute $\\gamma(0)$, $\\gamma(1)$ and $\\gamma(2)$.\n3. Compute $\\rho(1)$ and $\\rho(2)$.\n"
       "4. Say whether the process is weakly stationary, and why.\n\n**Report:** five numbers and one sentence."),
    code("# Solution\nma_acvf([0.5], sigma2=2.0)"),
    md("**Interpretation of the result.** $\\gamma(0) = 2.5$, $\\gamma(1) = 1$, $\\rho(1) = 0.4$ and nothing beyond lag 1: an MA(1) "
       "remembers exactly one period, at every date."),
    md("## A2 [Proposed]: an MA(2) process\n\n"
       "**Context.** $X_t = 2 + \\varepsilon_t + 0.4\\,\\varepsilon_{t-1} - 0.3\\,\\varepsilon_{t-2}$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$. Model: A1.\n\n"
       "1. Compute the mean.\n2. Compute $\\gamma(0)$, $\\gamma(1)$, $\\gamma(2)$ and $\\gamma(3)$.\n"
       "3. Compute $\\rho(1)$ and $\\rho(2)$, and sketch the ACF up to lag 4.\n4. Say how the constant 2 changes the autocovariances.\n\n"
       "**Report:** six numbers, the sketch and one sentence."),
    code("# Solution\nma_acvf([0.4, -0.3], sigma2=1.0, mu=2.0)"),
    md("## A3 [Solved]: a random walk\n\n"
       "**Context.** $X_t = X_{t-1} + \\varepsilon_t$, $X_0 = 0$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 0.25)$.\n\n"
       "1. Compute $E[X_{100}]$, $\\mathrm{Var}(X_{100})$ and $\\mathrm{Var}(X_{120})$.\n"
       "2. Compute $\\mathrm{Cov}(X_{100}, X_{120})$ and $\\mathrm{Corr}(X_{100}, X_{120})$.\n"
       "3. Say whether $X_t$ and $\\Delta X_t$ are stationary.\n4. With a drift $c = 0.1$, compute $E[X_{100}]$.\n\n"
       "**Report:** six numbers and two sentences."),
    code("# Solution\nprint(rw_moments(0.25, 100, 120))\nprint(rw_moments(0.25, 100, 120, drift=0.1))"),
    md("**Interpretation of the result.** The variance grows with $t$, so $X_t$ is not stationary; $\\Delta X_t = \\varepsilon_t$ is white noise. "
       "The correlation $\\sqrt{100/120} = 0.913$ depends on the dates, not only on their distance."),
    md("## A4 [Proposed]: a trend-stationary series\n\n"
       "**Context.** $Y_t = 5 + 0.2\\,t + \\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$. Model: A3.\n\n"
       "1. Compute $E[Y_t]$ and $\\mathrm{Var}(Y_t)$, and say whether $Y_t$ is stationary.\n"
       "2. Write $D_t = \\Delta Y_t$ and compute its mean, $\\gamma(0)$, $\\gamma(1)$ and $\\rho(1)$.\n"
       "3. Compare $Y_t$ with a random walk with drift 0.2: what happens to a shock $\\varepsilon_{50}$ in each?\n\n"
       "**Report:** five numbers and two sentences."),
    code("# Solution\ntrend_diff(5, 0.2, 1.0)"),
    md("## A5 [Solved]: a sample ACF by hand\n\n"
       "**Context.** Eight observations: $x = (4, 6, 5, 8, 7, 9, 6, 7)$.\n\n"
       "1. Compute $\\bar x$ and the deviations $x_t - \\bar x$.\n"
       "2. Compute $\\hat\\gamma(0)$, $\\hat\\gamma(1)$, $\\hat\\gamma(2)$ with divisor $T = 8$, then $\\hat\\rho(1)$ and $\\hat\\rho(2)$.\n"
       "3. Compute the band $\\pm 1.96/\\sqrt{T}$ and compare.\n4. Compute the Ljung–Box $Q^*(2)$ and decide at 5%.\n\n"
       "**Report:** seven numbers and one sentence."),
    code("# Solution\nacf_by_hand([4, 6, 5, 8, 7, 9, 6, 7], nlags=2)"),
    md("**Interpretation of the result.** $\\hat\\rho(2) = 0.389$ looks large, but with $T = 8$ the band is $\\pm 0.693$ and "
       "$Q^*(2) = 2.02 < 5.99$: eight observations cannot show any autocorrelation."),
    md("## A6 [Proposed]: Box–Pierce or Ljung–Box?\n\n"
       "**Context.** $T = 120$ monthly returns with $\\hat\\rho(1) = 0.21$, $\\hat\\rho(2) = -0.08$, $\\hat\\rho(3) = 0.12$. Model: A5.\n\n"
       "1. Compute the band and say which autocorrelations lie outside it.\n2. Compute $Q(3)$ and $Q^*(3)$.\n"
       "3. Compare both with $\\chi^2_{0.95}(3) = 7.81$ and decide.\n"
       "4. Explain why the two tests can disagree, and which one you trust here.\n\n**Report:** four numbers and two sentences."),
    code("# Solution\nportmanteau([0.21, -0.08, 0.12], 120)"),
    # ---------------- Part B
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: BET prices and returns\n\n"
       "**Question.** Is the BET log price stationary, and are BET daily returns white noise?\n\n"
       "1. Compute the sample ACF of $\\ln P_t$ at lags 1, 10 and 50.\n"
       "2. Draw the ACF of $r_t$ up to lag 20 with the band $\\pm 1.96/\\sqrt{T}$, and count the bars outside it.\n"
       "3. Compute Ljung–Box $Q^*(10)$ for $r_t$ and for $r_t^2$, with their p-values.\n"
       "4. Interpretation: can yesterday's return be used to forecast today's return?\n\n"
       "**Report:** three ACF values, the chart, two statistics with p-values and two sentences."),
    code("# Solution\nb1 = b1_returns('bet', fname='ch1_sem_b1', save=False)\nb1"),
    md("**Interpretation of the result.** The level is random-walk-like (ACF close to 1 at lag 50). Returns have a small "
       "positive first-order autocorrelation: significant, but it explains about 1% of the variance. The squares are "
       "strongly autocorrelated: today's risk can be forecast from yesterday's move."),
    md("## B2 [Proposed]: the S&P 500 and EUR/RON\n\n"
       "**Question.** Do the S&P 500 and the EUR/RON rate behave like the BET? Model: B1.\n\n"
       "1. Repeat the three steps of B1 for both series (`b1_returns('sp500')`, `b1_returns('eurron', start=None)`).\n"
       "2. Compare the sign and the size of $\\hat\\rho(1)$ of the returns across BET, S&P 500 and EUR/RON.\n"
       "3. Compare $Q^*(10)$ of the squared returns across the three series.\n"
       "4. Interpretation: which of the three return series is closest to white noise?\n\n"
       "**Report:** one table (three series, six numbers each) and two sentences."),
    code("# Solution\nb2 = {'BET': b1, 'S&P 500': b1_returns('sp500', fname='ch1_sem_b2', save=False), "
         "'EUR/RON': b1_returns('eurron', start=None)}\n"
         "pd.DataFrame({k: {'T': d['n'], 'ACF(1) ln P': d['acf_p1'], 'rho1 r': d['r1'], 'Q*(10) r': d['lb_r']['lb'], "
         "'rho1 r^2': d['sq1'], 'Q*(10) r^2': d['lb_r2']['lb']} for k, d in b2.items()}).T.round(3)"),
    md("## B3 [Solved]: Romanian real GDP\n\n"
       "**Question.** Which transformation makes Romanian quarterly GDP closest to stationary?\n\n"
       "1. Compute $\\ln Y_t$, $100\\,\\Delta\\ln Y_t$ and $100\\,\\Delta_4\\ln Y_t$.\n"
       "2. Draw the three series and their ACF up to lag 12.\n3. Report $\\hat\\rho(1)$, $\\hat\\rho(4)$ and $Q^*(8)$ for each.\n"
       "4. Interpretation: why is $\\hat\\rho(4)$ of the quarterly growth rate close to 1?\n\n**Report:** a table of nine numbers, the chart and two sentences."),
    code("# Solution\nb3 = b3_gdp(save=False)\n"
         "pd.DataFrame({k: {'rho1': b3[k]['r1'], 'rho4': b3[k]['r4'], 'Q*(8)': b3[k]['lb8']['lb']} for k in ['lev', 'd1', 'd4']}).T.round(2)"),
    md("**Interpretation of the result.** The quarterly rate of unadjusted GDP repeats the seasonal swing every year "
       "($\\hat\\rho(4) \\approx 1$). The annual rate removes the season; its ACF dies out after 2–3 quarters, but it reacts "
       "late to turning points and consecutive values overlap."),
    md("## B4 [Proposed]: Romanian inflation\n\n"
       "**Question.** Is monthly Romanian inflation since 2005 white noise around its mean? Model: B3.\n\n"
       "1. Compute the monthly inflation $100\\,\\Delta\\ln P_t$ and the 12-month inflation $100\\,\\Delta_{12}\\ln P_t$ "
       "from `ro_hicp()`.\n"
       "2. Draw the ACF of both up to lag 36 and report $\\hat\\rho(1)$, $\\hat\\rho(12)$ and $\\hat\\rho(24)$.\n"
       "3. Compute Ljung–Box $Q^*(12)$ for the monthly inflation.\n"
       "4. Interpretation: what does the ACF of the monthly inflation say about how fast an inflation shock fades?\n\n"
       "**Report:** six ACF values, one statistic with its p-value, the chart and two sentences."),
    code("# Solution\nb4_hicp(save=False)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: has Romanian inflation become more persistent?\n\n"
       "**Question.** Is monthly inflation more persistent after 2020 than in 2005–2019? Models: B1, B4.\n\n"
       "1. Compute $\\hat\\rho(1)$ and $Q^*(12)$ in each subperiod, with the band of each.\n"
       "2. Draw $\\hat\\rho(1)$ on rolling windows of 60 months.\n"
       "3. List the events of 2020–2024 that could change the persistence.\n"
       "4. Interpretation: why is the difference of two sample autocorrelations hard to judge with 80 observations?\n\n"
       "**Report:** a table, the chart and a plan for a project."),
    code("# Reference analysis\nc1_persistence(save=False)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant interpreted the S&P 500 daily data since 2000. It answered:\n\n"
       "- (a) the ACF of the log price at lag 1 is close to 1, so the log price is a stationary process with a very long memory;\n"
       "- (b) the return autocorrelation at lag 1 is outside the band, so a trader can forecast most of tomorrow's return;\n"
       "- (c) Ljung–Box Q(10) of the squared returns is very large, so the returns themselves are autocorrelated;\n"
       "- (d) a weakly stationary process is always strictly stationary, since its mean and variance are constant;\n"
       "- (e) differencing a series that is already stationary does no harm: white noise stays white noise;\n"
       "- (f) a random walk has $\\mathrm{Var}(X_t) = t\\sigma^2$, so it is not stationary.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 1, 'lecture')
    build(SEMINAR, 1, 'seminar')
