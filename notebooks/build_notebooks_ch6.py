"""
build_notebooks_ch6.py -- lecture and seminar notebooks of Chapter 6 (TSA): VAR models and Granger causality
===========================================================================================================
Output: notebooks/EN/chapter6_lecture_notebook.ipynb, notebooks/EN/chapter6_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_06/generate_all_charts.py and seminar6.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch6.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter6_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter6_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 6
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(6)
import generate_all_charts as g   # noqa: E402
import seminar6 as s              # noqa: E402
import build_quantlets as bq      # noqa: E402

CONSTS = '\n'.join(bq.CONSTS)
CORE = bq.CORE

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Time Series Analysis — Chapter 6: VAR models and Granger causality\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- From one series to several: Romanian macro data, cross-correlations of markets.\n"
       "- The VAR(p) model: a worked VAR(1), stability, the companion matrix.\n"
       "- Estimation by OLS, lag selection (AIC, BIC, HQ), residual diagnostics.\n"
       "- Granger causality: Romania, S&P 500 / DAX / BET, the omitted-variable pitfall.\n"
       "- Impulse responses (Cholesky, ordering, generalised), FEVD, the Diebold–Yilmaz spillover index.\n"
       "- Forecasting and out-of-sample evaluation; the Stock–Watson (2001) VAR.\n"
       "- References: Huang and Petukhina (2022), *Applied Time Series Analysis and Forecasting with Python*, Ch. 7; "
       "Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.), Sec. 12.3; Lütkepohl (2005); "
       "Hamilton (1994), Ch. 11; Stock and Watson (2001)."),
    *common_cells(),
    md("## Definitions and helpers used in the whole notebook\n\n"
       "- VAR(p): $\\mathbf{Y}_t = \\mathbf{c} + \\mathbf{A}_1\\mathbf{Y}_{t-1} + \\dots + \\mathbf{A}_p\\mathbf{Y}_{t-p} + "
       "\\boldsymbol{\\varepsilon}_t$, estimated with `VAR` from `statsmodels` (`fit_var(d, p)`).\n"
       "- `ro_var_data()`: Romanian quarterly GDP growth `g`, 12-month inflation `pi`, ROBOR 3M `i` (2005Q1–2026Q2).\n"
       "- `ic_table(d, pmax)`: AIC, BIC, HQ on a common sample; `granger_F(d, cause, effect, p)`: the Granger $F$ test.\n"
       "- `girf(res, H)`: generalised impulse responses; `gfevd(res, H)`: generalised FEVD (Diebold–Yilmaz); "
       "`boot_irf(res, H)`: residual-bootstrap bands."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. From one series to several\n\n- Romanian macro series; cross-correlations of markets and of ROBOR with inflation."),
    code(src(g.fig_ro_macro, g.fig_ccf) + "\n\n\nfig_ro_macro(save_it=False)\ncc = fig_ccf(save_it=False)\n"
         "print('BET vs S&P 500 at lags 0 and 1:', round(cc['bet'][0], 3), round(cc['bet'][1], 3))"),
    md("## 2. A VAR(1) by hand and stability\n\n- The worked example of the slides; a simulated path; eigenvalues of three "
       "estimated VARs."),
    code(src(g.worked_example, g.fig_var_sim, g.fig_roots)
         + "\n\n\nprint(worked_example())\nfig_var_sim(save_it=False)\nfig_roots(save_it=False)"),
    md("## 3. Estimation, lag selection, diagnostics\n\n- The Romanian VAR(2): information criteria, coefficients, residuals."),
    code(src(g.ic_romania, g.fig_ic, g.ro_estimates, g.fig_resid)
         + "\n\n\nic = fig_ic(save_it=False)\nprint('selected:', ic['best'])\nr = ro_fit()\nprint(r.summary())\n"
           "res = fig_resid(save_it=False)\nprint({k: res[k] for k in ['lb8', 'lb12', 'jb']})"),
    md("## 4. Granger causality\n\n- $F$ tests in the Romanian and market VARs; the omitted-variable Monte Carlo."),
    code(src(g.granger_romania, g.granger_markets, g.granger_by_hand, g.omitted_sim, g.fig_granger_sim)
         + "\n\n\nprint(granger_by_hand())\ngr = granger_romania()\n"
           "print(pd.DataFrame(gr['tests']).T[['F', 'p']])\ngm = granger_markets()\nprint(pd.DataFrame(gm['tests']).T[['F', 'p']])\n"
           "fig_granger_sim(save_it=False)"),
    md("## 5. Impulse responses and FEVD\n\n- Cholesky responses with bootstrap bands; two orderings and the generalised "
       "responses; the variance decomposition."),
    code(src(g.fig_irf_ro, g.fig_irf_order, g.fig_fevd_ro)
         + "\n\n\nfig_irf_ro(save_it=False)\nfig_irf_order(save_it=False)\nfe = fig_fevd_ro(save_it=False)\nprint(fe['i'])"),
    md("## 6. Market spillovers\n\n- Responses of DAX and BET to an S&P 500 shock; the Diebold–Yilmaz spillover index."),
    code(src(g.fig_market_irf, g.spillover_table, g.rolling_spillover, g.fig_spillover)
         + "\n\n\nfig_market_irf(save_it=False)\nsp = fig_spillover(save_it=False)\nprint(round(sp['total'], 1), sp['roll_max_d'])"),
    md("## 7. Forecasting\n\n- VAR forecasts with intervals; out-of-sample comparison with AR models and the random walk."),
    code(src(g.fig_forecast_ro, g.oos, g.fig_oos) + "\n\n\nfig_forecast_ro(save_it=False)\nev = fig_oos(save_it=False)\n"
         "print(pd.DataFrame({m: {c: ev['ro'][m][c][1] for c in RO_VARS} for m in ['VAR', 'AR', 'RW']}))"),
    md("## 8. Case study: Stock and Watson (2001)\n\n- Inflation, unemployment, federal funds rate; VAR(4), 1960Q1–2000Q4."),
    code(src(g.fig_us_data, g.fig_sw_irf) + "\n\n\nfig_us_data(save_it=False)\nsw = fig_sw_irf(save_it=False)\nprint(sw['granger'])"),
    md("## Exercises\n\n1. Add the unemployment rate `u` to the Romanian VAR (`ro_var_data(['g', 'u', 'pi', 'i'])`): does it "
       "Granger-cause anything?\n"
       "2. Recompute the Romanian responses with the order (i, pi, g): which conclusions survive?\n"
       "3. Compute the spillover index with the Nasdaq 100 (`'ndx'`) instead of the S&P 500."),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_FUNCS = [s.var1_by_hand, s.irf_by_hand, s.granger_rss, s.pair_returns, s.market_pair, s.b3_romania]
SEMINAR = [
    md("# Time Series Analysis — Seminar 6: VAR models and Granger causality\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The seminar comes before Lecture 6; the slides give the definitions (\"What you need today\").\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class."),
    *common_cells(),
    md("## Seminar functions\n\n- Helpers of Chapter 6 (`fit_var`, `ic_table`, `granger_F`, `girf`, `companion`, `ro_var_data`) "
       "and the seminar functions: `var1_by_hand`, `irf_by_hand`, `granger_rss`, `pair_returns`, `market_pair`, `b3_romania`."),
    code(CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_FUNCS)),
    code("# [solutions only]\n" + src(s.ic_by_hand, s.b4_exchange_rate, s.c1_price_puzzle, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: a VAR(1) by hand\n\n"
       "**Context.** $\\mathbf{c} = (0.4, 0.8)'$, $\\mathbf{A} = \\begin{pmatrix} 0.6 & 0.2 \\\\ 0.2 & 0.6 \\end{pmatrix}$, "
       "$\\mathbf{Y}_T = (3, 2)'$.\n\n"
       "1. Write the two equations.\n2. Find the eigenvalues of $\\mathbf{A}$ and decide whether the VAR is stable.\n"
       "3. Compute the mean $\\boldsymbol{\\mu}$.\n4. Compute $\\hat{\\mathbf{Y}}_{T+1}$ and $\\hat{\\mathbf{Y}}_{T+2}$.\n\n"
       "**Report:** two equations, two eigenvalues, $\\boldsymbol{\\mu}$, two forecasts."),
    code("# Solution\nvar1_by_hand([[0.6, 0.2], [0.2, 0.6]], [0.4, 0.8], [3.0, 2.0])"),
    md("**Interpretation of the result.** Eigenvalues 0.8 and 0.4: stable. The forecasts (2.6, 2.6) and (2.48, 2.88) move "
       "towards the mean (2.67, 3.33)."),
    md("## A2 [Proposed]: stable or not?\n\n"
       "**Context.** (a) $\\mathbf{A} = \\begin{pmatrix} 0.9 & 0.3 \\\\ 0.2 & 0.8 \\end{pmatrix}$; "
       "(b) $\\mathbf{A} = \\begin{pmatrix} 0.5 & -0.4 \\\\ 0.4 & 0.5 \\end{pmatrix}$. Model: A1.\n\n"
       "1. For each matrix, compute the trace, the determinant and the eigenvalues.\n2. Decide whether each VAR is stable.\n"
       "3. Describe how the responses to a shock behave in each case.\n\n"
       "**Report:** four eigenvalues (or moduli), two decisions and two sentences."),
    code("# Solution\nprint(var1_by_hand([[0.9, 0.3], [0.2, 0.8]], [0, 0], [0, 0]))\n"
         "print(var1_by_hand([[0.5, -0.4], [0.4, 0.5]], [0, 0], [0, 0]))"),
    md("## A3 [Solved]: orthogonalised impulse responses\n\n"
       "**Context.** The VAR(1) of A1 with $\\boldsymbol{\\Sigma} = \\begin{pmatrix} 1 & 0.4 \\\\ 0.4 & 1 \\end{pmatrix}$.\n\n"
       "1. Compute the Cholesky factor $\\mathbf{P}$.\n"
       "2. Compute $\\boldsymbol{\\Theta}_0 = \\mathbf{P}$, $\\boldsymbol{\\Theta}_1 = \\mathbf{A}\\mathbf{P}$ and $\\boldsymbol{\\Phi}_2 = \\mathbf{A}^2$.\n"
       "3. Compute $\\mathbf{P}$ again with the order $(y_2, y_1)$ and say what changes on impact.\n\n"
       "**Report:** three matrices and one sentence."),
    code("# Solution\nr = irf_by_hand([[0.6, 0.2], [0.2, 0.6]], [[1.0, 0.4], [0.4, 1.0]], 3)\n"
         "print('P =', np.round(r['P'], 3))\nprint('Theta_1 =', np.round(r['Theta'][1], 3))\nprint('Phi_2 =', np.round(r['Phi'][2], 3))\n"
         "print('P, order (y2, y1) =', np.round(r['P_rev'], 3))"),
    md("**Interpretation of the result.** With $y_1$ first, a shock $u_1$ moves $y_2$ by 0.4 at once; with $y_2$ first, "
       "it is $y_1$ that reacts on impact to the shock of $y_2$."),
    md("## A4 [Proposed]: a variance decomposition by hand\n\n"
       "**Context.** $\\boldsymbol{\\Theta}_0$ and $\\boldsymbol{\\Theta}_1$ of A3. Model: A3.\n\n"
       "1. Compute the share of $u_1$ in the 1-step forecast error variance of $y_2$.\n"
       "2. Compute the 2-step forecast error variance of $y_2$ and the share of $u_1$ in it.\n"
       "3. Compute the share of $u_2$ in the 2-step error variance of $y_1$, and explain why it is 0 at one step.\n\n"
       "**Report:** three percentages and one sentence."),
    code("# Solution\nr = irf_by_hand([[0.6, 0.2], [0.2, 0.6]], [[1.0, 0.4], [0.4, 1.0]], 3)\nprint(np.round(r['fevd'][1], 4), r['mse'][1])"),
    md("## A5 [Solved]: a Granger $F$ test\n\n"
       "**Context.** A bivariate VAR(2), $T = 100$; equation of $y_1$: $RSS_U = 45.2$, $RSS_R = 52.8$.\n\n"
       "1. Give $H_0$, the number of restrictions $q$ and the number of coefficients $k$.\n2. Compute $F$.\n"
       "3. Decide at 5% and state the conclusion in words.\n\n**Report:** $H_0$, $F$, a decision and one sentence."),
    code("# Solution\ngranger_rss(52.8, 45.2, 100, 2, 5)"),
    md("**Interpretation of the result.** $F \\approx 7.99 > 3.09$: the lags of $y_2$ help to forecast $y_1$."),
    md("## A6 [Proposed]: choosing $p$ by hand\n\n"
       "**Context.** $K = 3$, $T = 80$, $\\ln\\det\\tilde{\\boldsymbol{\\Sigma}}(p) = 1.50, 0.60, 0.25, 0.00, -0.15$ for $p = 0..4$. Model: A5.\n\n"
       "1. Compute the penalty per lag for AIC, BIC and HQ.\n2. Compute the three criteria for each $p$ and find the minima.\n"
       "3. Say which $p$ you would start from with 80 quarters, and why.\n\n"
       "**Report:** three penalties, a table of 15 values, three orders and one sentence."),
    code("# Solution\nic_by_hand([1.50, 0.60, 0.25, 0.00, -0.15], 80, 3)"),
    # ---------------- Part B
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: does the S&P 500 lead the BET?\n\n"
       "**Question.** Do yesterday's S&P 500 returns help to forecast today's BET returns, and the other way round?\n\n"
       "1. Compute the cross-correlations $\\mathrm{Corr}(\\mathrm{BET}_t, \\mathrm{S\\&P}_{t-k})$, $k = -5..5$.\n"
       "2. Choose $p$ with AIC, BIC and HQ (maximum 10) and estimate the VAR with the HQ order.\n"
       "3. Test Granger causality in both directions.\n"
       "4. Compute the generalised response of the BET to an S&P 500 shock for 5 days.\n"
       "5. Interpretation: why does New York lead Bucharest by one day?\n\n"
       "**Report:** three cross-correlations, the chosen $p$, two $F$ tests, two responses and one sentence."),
    code("# Solution\nb1 = market_pair('D', 'ch6_sem_b1', save=False)\n"
         "print('orders:', b1['orders'], 'p =', b1['p'])\n"
         "print('S&P -> BET:', round(b1['sp_bet']['F'], 1), b1['sp_bet']['p'])\n"
         "print('BET -> S&P:', round(b1['bet_sp']['F'], 2), round(b1['bet_sp']['p'], 3))\n"
         "print('GIRF of the BET:', np.round(b1['girf_bet'], 3))"),
    md("**Interpretation of the result.** The S&P 500 Granger-causes the BET and not the reverse. Wall Street closes after "
       "Bucharest, so the American afternoon is priced in Bucharest the next day: a time-zone effect."),
    md("## B2 [Proposed]: the same question with weekly returns\n\n"
       "**Question.** Does the lead of the S&P 500 survive with Friday-to-Friday weekly returns? Model: B1.\n\n"
       "1. Repeat steps 1–4 of B1 on weekly returns (`market_pair('W', ...)`).\n"
       "2. Compare the lag-1 cross-correlation and the $F$ statistic with the daily ones.\n"
       "3. Interpretation: why is the lead much weaker at the weekly frequency?\n\n"
       "**Report:** the same items as in B1 and two sentences."),
    code("# Solution\nb2 = market_pair('W', 'ch6_sem_b2', save=False)\nprint(b2['orders'], b2['p'], b2['sp_bet'], b2['ccf0'], b2['ccf1'])"),
    md("## B3 [Solved]: a VAR for the Romanian economy\n\n"
       "**Question.** How do GDP growth, inflation and the 3-month interest rate interact in Romania?\n\n"
       "1. Choose $p$ with AIC, BIC and HQ ($p \\le 6$) and estimate a VAR(2).\n"
       "2. Check stability and run the portmanteau test with $h = 8$ and $h = 12$.\n3. Run the six Granger $F$ tests.\n"
       "4. Plot the orthogonalised response of ROBOR to an inflation shock and the FEVD of ROBOR.\n"
       "5. Interpretation: do the results show that the central bank reacts to inflation?\n\n"
       "**Report:** three orders, one modulus, two portmanteau tests, a table of six tests, the chart and one sentence."),
    code("# Solution\nb3 = b3_romania(save=False)\nprint('orders:', b3['ic']['best'], 'largest modulus:', round(b3['max_mod'], 3))\n"
         "print('portmanteau:', b3['lb'])\nprint(pd.DataFrame(b3['granger']).T)\nprint('FEVD of ROBOR:', b3['fevd_i'])"),
    md("**Interpretation of the result.** ROBOR follows inflation (Granger test, impulse response, FEVD), as a reaction function "
       "would imply. ROBOR is a market rate, so the VAR cannot separate the central bank's decisions from market expectations."),
    md("## B4 [Proposed]: adding the EUR/RON exchange rate\n\n"
       "**Question.** Does the depreciation of the leu help to forecast inflation? Model: B3.\n\n"
       "1. Choose $p$ with the three criteria ($p \\le 4$) for `ro_var_data(['g', 'ds', 'pi', 'i'])` and estimate a VAR(2).\n"
       "2. Test whether `ds` Granger-causes `pi`, `i` and `g`, and whether `i` and `pi` Granger-cause `ds`.\n"
       "3. Compute the share of EUR/RON shocks in the FEVD of inflation at 1, 4, 8 and 12 quarters.\n"
       "4. Interpretation: does the result mean that the exchange rate does not affect Romanian prices?\n\n"
       "**Report:** three orders, five $F$ tests, four shares and two sentences."),
    code("# Solution\nb4_exchange_rate(save=False)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: the price puzzle\n\n"
       "**Question.** Why does inflation rise after a ROBOR shock in the VAR of B3, and does it change with the exchange rate or "
       "more lags? Models: B3, B4.\n\n"
       "1. Plot the response of inflation to a ROBOR shock in the VAR(2) of B3, the VAR(2) of B4 and a VAR(4) with the four "
       "variables of B4.\n2. Report the largest and the smallest response and their horizons.\n"
       "3. List two pieces of information that the central bank has and the VAR does not.\n"
       "4. Interpretation: is a positive response of inflation a proof that higher rates raise prices?\n\n"
       "**Report:** the chart, six numbers and a project plan."),
    code("# Reference analysis\nc1_price_puzzle(save=False)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant analysed the VAR of B3. It answered:\n\n"
       "- (a) AIC chooses p = 5, so the true lag order of the economy is 5;\n"
       "- (b) the largest eigenvalue modulus of A1 is above 1, so the VAR(2) is explosive;\n"
       "- (c) the portmanteau test at 8 lags has a p-value of about 0.02, so the residuals are white noise;\n"
       "- (d) GDP growth Granger-causes ROBOR, so faster growth causes the BNR to raise rates;\n"
       "- (e) Cholesky impulse responses do not depend on the order of the variables;\n"
       "- (f) in each row of the FEVD the shares add up to 100%.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 6, 'lecture')
    build(SEMINAR, 6, 'seminar')
