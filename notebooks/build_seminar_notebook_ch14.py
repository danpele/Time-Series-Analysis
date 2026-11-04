"""
build_seminar_notebook_ch14.py -- seminar notebook of Chapter 14 (TSA): multivariate GARCH, portfolio VaR, hedging
=================================================================================================================
Output: notebooks/EN/chapter14_seminar_notebook.ipynb
Chapter 14 is a self-study chapter; its lecture notebook is built by build_notebooks_ch14.py, the seminar notebook
here (a separate script, so that the lecture builder stays untouched). The code is taken from
Quantlets/Ch_14/generate_all_charts.py and seminar14.py (inspect.getsource).
Run:  python3 notebooks/build_seminar_notebook_ch14.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter14_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 14
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(14)
import generate_all_charts as g   # noqa: E402
import seminar14 as s             # noqa: E402
import build_quantlets as bq      # noqa: E402

CONSTS = '\n'.join(bq.CONSTS).replace('HERE = "."', 'HERE = next((p for p in ("../../Quantlets/Ch_14", "Quantlets/Ch_14", ".") '
                                                     'if os.path.isdir(p)), ".")')
SEM_CONSTS = (f"SHOCK_DAY = {s.SHOCK_DAY!r}\nHEDGE_DAY = {s.HEDGE_DAY!r}\nBOND_START = {s.BOND_START!r}\n"
              f"BOND_BREAK = {s.BOND_BREAK!r}\nVAR_PAIR = {s.VAR_PAIR!r}\nVAR_W = {s.VAR_W!r}\nBEKK_EX = {s.BEKK_EX!r}\n"
              f"HL_STEPS = {s.HL_STEPS!r}")
SEM_FUNCS = [g.kupiec, g.christoffersen, s.lecture_numbers, s.dcc_Q, s.sp_dax, s.a1_dcc_step, s.a3_portfolio,
             s.a5_half_life, s.b1_stock_bond]
SOL_FUNCS = [s.a2_bekk, s.a4_hedge, s.b2_var_backtest, s.c1_check]

SEMINAR = [
    md("# Time Series Analysis — Seminar 14: Multivariate GARCH models (CCC, DCC, BEKK) and portfolio VaR\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Chapter 14 is a self-study chapter: read it first; the slides recap the formulas used here.\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class.\n"
       "- The first code cell installs the `arch` package if it is missing (otherwise a numpy fallback is used)."),
    *common_cells(bq.INSTALL),
    md("## Seminar functions\n\n- Helpers of Chapter 14 (`common_returns`, `step1`, `dcc_fit`, `dcc_path`, `half_life`, "
       "`kupiec`, `christoffersen`) and the seminar functions: `dcc_Q`, `sp_dax`, `a1_dcc_step`, `a3_portfolio`, "
       "`a5_half_life`, `b1_stock_bond`."),
    code(CONSTS + '\n' + SEM_CONSTS + '\n\n\n' + src(*bq.CORE) + '\n\n\n' + src(*SEM_FUNCS)),
    code("# [solutions only]\n" + src(*SOL_FUNCS)),
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code. "
       "A1, A3 and A5 use the S&P 500 and DAX DCC estimated on 2000-2026: the next cell estimates it once."),
    code("data = sp_dax()        # daily S&P 500 and DAX on common trading days: GARCH(1,1) per index, then the DCC"),
    md("## A1 [Solved]: one step of the DCC recursion\n\n"
       "**Context.** The DCC of the S&P 500 and the DAX; the residuals $z$ of 16 March 2020 and $Q_{t-1}$.\n\n"
       "1. Compute $q_{11}$, $q_{22}$, $q_{12}$ for 17 March 2020, term by term.\n"
       "2. Compute the correlation of 16 and of 17 March 2020.\n3. Why does a fall of 12% move the correlation so little?\n\n"
       "**Report:** three elements of $Q_t$, two correlations, one sentence."),
    code("# Solution\na1 = a1_dcc_step(data)\nprint({k: (round(v, 4) if isinstance(v, float) else v) for k, v in a1.items()})"),
    md("**Interpretation of the result.** $b$ is close to 1 and the GARCH-standardised residuals are moderate, so the "
       "correlation moves from 0.705 to 0.707 only."),
    md("## A2 [Proposed]: one step of a diagonal BEKK\n\n"
       "**Context.** $c_{11} = 0.05$, $c_{12} = 0.01$, $c_{22} = 0.04$; $a_{11} = 0.25$, $a_{22} = 0.30$; $b_{11} = 0.95$, "
       "$b_{22} = 0.93$; $\\varepsilon_{t-1} = (1.5, -1.0)$; $h_{11} = 1.2$, $h_{12} = 0.3$, $h_{22} = 0.9$. Model: A1.\n\n"
       "1. Compute $h_{11,t}$, $h_{22,t}$, $h_{12,t}$ term by term.\n2. Compute the correlation at $t - 1$ and at $t$. Why does it fall?\n"
       "3. Check that $H_t$ is positive definite; give the persistences.\n\n"
       "**Report:** three elements, two correlations, one determinant, three persistences."),
    code("# Solution\na2_bekk()"),
    md("## A3 [Solved]: portfolio variance and VaR under CCC and DCC\n\n"
       "**Context.** S&P 500 and DAX on 17 March 2020: GARCH volatilities, the DCC correlation of A1, the constant CCC "
       "correlation and a static covariance matrix; weights (0.5, 0.5).\n\n"
       "1. Compute $h_{12} = \\rho_{12}\\sigma_1\\sigma_2$ for each model.\n2. Compute $\\sigma_p^2 = w^\\top Hw$, $\\sigma_p$ and "
       "the VaR 1% for each model.\n3. Which part of the VaR comes from the correlation, which from the volatilities?\n\n"
       "**Report:** three covariances, three VaRs, one sentence."),
    code("# Solution\na3 = a3_portfolio(data)\nprint(pd.DataFrame({k: a3[k] for k in ('dcc', 'ccc', 'static')}).round(3))\n"
         "print('portfolio return that day:', round(a3['ret'], 2))"),
    md("**Interpretation of the result.** The volatilities move the VaR from about 2.7% (static) to 12% (CCC); the crisis "
       "correlation adds about half a point (DCC)."),
    md("## A4 [Proposed]: the hedge ratio of the BET with the DAX\n\n"
       "**Context.** DCC of the BET and the DAX estimated on 2005-2014 (Chapter 14), run to 18 September 2026. Model: A3.\n\n"
       "1. Compute $h_t = \\rho_t\\sigma_{s,t}/\\sigma_{f,t}$ and the DAX position for 100 lei of BET.\n"
       "2. Compute $\\rho_t^2$ and $\\sigma_s\\sqrt{1 - \\rho_t^2}$.\n3. Compare with the static OLS ratio and with the "
       "out-of-sample effectiveness of the lecture.\n\n**Report:** two ratios, one position, two effectiveness figures, one sentence."),
    code("# Solution\na4_hedge()"),
    md("## A5 [Solved]: the half-life of a correlation shock\n\n"
       "**Context.** $a + b$ of the S&P 500 and DAX (A1), of the five-asset panel of the lecture and of the stock-bond pair (B1).\n\n"
       "1. Compute the half-life $\\ln 0.5/\\ln(a + b)$ and $\\ln 0.1/\\ln(a + b)$.\n"
       "2. Forecast the S&P 500 and DAX correlation 10 and 50 days after 17 March 2020 with "
       "$\\rho_{t+k} \\approx \\bar\\rho + (a + b)^k(\\rho_t - \\bar\\rho)$.\n3. What is the half-life under CCC?\n\n"
       "**Report:** six durations, two forecasts, one sentence. (B1 is estimated in the next cell.)"),
    code("# Solution\nb1 = b1_stock_bond(save_it=False)\na5_half_life(a1, b1)"),
    md("**Interpretation of the result.** Half of a correlation shock is still there after 85 trading days for the "
       "S&P 500 and DAX: diversification returns slowly after a crisis."),
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: stocks and US Treasuries\n\n"
       "**Question.** Do long Treasuries still hedge stocks, or did the stock-bond correlation change sign after 2022?\n\n"
       "1. Fit a GARCH(1,1) to each series and the DCC by two-step maximum likelihood.\n"
       "2. Test CCC against DCC with the likelihood ratio; compute the half-life.\n"
       "3. Plot the DCC and the 250-day rolling correlation; report the mean before and after January 2022.\n"
       "4. Interpretation: what does the change mean for a 60/40 stock-bond portfolio?\n\n"
       "**Report:** four parameters with standard errors, one test, one half-life, two means, one sentence."),
    code("# Solution\nb1 = b1_stock_bond(save_it=False)\n"
         "print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b1.items() if k != 'garch'})"),
    md("**Interpretation of the result.** The correlation was negative on almost every day before 2022 and has been "
       "positive on most days since: bonds hedge stocks less than before."),
    md("## B2 [Proposed]: a VaR backtest of a BET and DAX portfolio\n\n"
       "**Question.** Does a DCC VaR 1% give the right number of violations for a 50/50 BET and DAX portfolio? Model: A3.\n\n"
       "1. Compute the VaR 1% with DCC, CCC and the static covariance (parameters 2005-2014, VaR 2015-2026).\n"
       "2. Count the violations and compare with $0.01n$.\n3. Apply the Kupiec and Christoffersen tests.\n"
       "4. Interpretation: why can the static VaR pass the count and still be a poor risk measure?\n\n"
       "**Report:** three counts, three tests, one sentence."),
    code("# Solution\nb2 = b2_var_backtest(save_it=False)\n"
         "print(pd.DataFrame({k: {c: b2[k][c] for c in ('x', 'expected', 'rate', 'lr', 'p', 'mean_var')} for k in ('DCC', 'CCC', 'Static')}).T)"),
    md("# Part C: AI critique"),
    md("## C1 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant answered:\n\n"
       "- (a) a full BEKK(1,1) for 10 assets has about as many parameters as a CCC model;\n"
       "- (b) for the S&P 500 and the DAX, a + b = 0.992, so a correlation shock is gone within a week;\n"
       "- (c) CCC is DCC with a = b = 0; the LR statistic of 131.8 rejects constant correlation;\n"
       "- (d) the VaR of a portfolio is the weighted sum of the VaRs of its assets;\n"
       "- (e) 65 violations of a VaR 1% in 2893 days is close to 1%, so the model is well calibrated;\n"
       "- (f) the daily correlation of the S&P 500 and the BET is the right input for a weekly risk report.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc1_check()"),
]

if __name__ == '__main__':
    build(SEMINAR, 14, 'seminar')
