"""
build_notebooks_ch7.py -- lecture and seminar notebooks of Chapter 7 (TSA): cointegration and VECM
==================================================================================================
Output: notebooks/EN/chapter7_lecture_notebook.ipynb, notebooks/EN/chapter7_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_07/generate_all_charts.py and seminar7.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch7.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter7_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter7_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 7
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(7)
import generate_all_charts as g   # noqa: E402
import seminar7 as s              # noqa: E402

INSTALL = ("# The arch package (Phillips-Ouliaris test) is not preinstalled in Google Colab: pip install arch\n"
           "import importlib.util, subprocess, sys\n"
           "if importlib.util.find_spec('arch') is None:\n"
           "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'arch'])")
CONSTS = ('import itertools\nimport statsmodels.api as sm\n'
          'from statsmodels.tsa.stattools import adfuller, coint, acf\n'
          'from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp\n'
          'from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_order\n'
          'from statsmodels.tsa.api import VAR\nfrom arch.unitroot.cointegration import phillips_ouliaris\n'
          f'SEED = {g.SEED!r}\nYIELDS = {g.YIELDS!r}\nYIELD_START = {g.YIELD_START!r}\nHICP_RO = {g.HICP_RO!r}\n'
          f'HICP_EA = {g.HICP_EA!r}\nROBOR = {g.ROBOR!r}\nEURIBOR = {g.EURIBOR!r}\nCONS_RO = {g.CONS_RO!r}\n'
          f'GDP_RO = {g.GDP_RO!r}\nBVB = {g.BVB!r}\nUS_BANKS = {g.US_BANKS!r}\nPAIR = {g.PAIR!r}\n'
          f'PAIR_START = {g.PAIR_START!r}\nFORM, TRADE, ENTRY = {g.FORM!r}, {g.TRADE!r}, {g.ENTRY!r}\n'
          f'COST_BVB, COST_US = {g.COST_BVB!r}, {g.COST_US!r}')
CORE = [g.us_yields, g.us_macro, g.cee_fx, g.stock, g.stock_panel, g.save, g.years_axis, g.ts_index, g.adf_test, g.ols,
        g.eg_test, g.half_life, g.johansen, g.ecm_fit]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Time Series Analysis — Chapter 7: Cointegration and VECM\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Common stochastic trends: the drunk and her dog; four real pairs.\n"
       "- The Engle–Granger two-step method: why its critical values differ from Dickey–Fuller; Phillips–Ouliaris.\n"
       "- Error correction models and half-lives.\n"
       "- The Johansen tests, the term-structure VECM, impulse responses; VECM against a VAR in differences out of sample.\n"
       "- Applications: purchasing power parity for EUR/RON, ROBOR and Euribor, three Central European currencies, "
       "pairs trading on the Bucharest Stock Exchange and on US banks.\n"
       "- References: Huang and Petukhina (2022), *Applied Time Series Analysis and Forecasting with Python*, Ch. 9; "
       "Hamilton (1994), Ch. 19–20; Lütkepohl (2005), Ch. 6–8; Engle and Granger (1987); Johansen (1988, 1991).\n"
       "- The first code cell installs the `arch` package if it is missing (`pip install arch`)."),
    *common_cells(INSTALL),
    md("## Definitions and helpers used in the whole notebook\n\n"
       "- `eg_test(y, x)`: Engle–Granger (OLS of $y$ on a constant and $x$, ADF on the residuals with MacKinnon p-values) "
       "and Phillips–Ouliaris $Z_t$; $H_0$: no cointegration.\n"
       "- `ecm_fit(y, x)`: $\\Delta y_t = c + \\gamma\\hat u_{t-1} + \\delta_0\\Delta x_t + \\dots$; half-life "
       "$\\ln 0.5/\\ln(1 + \\gamma)$.\n"
       "- `johansen(df)`: trace and maximum-eigenvalue tests with an unrestricted constant and BIC lags."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Common stochastic trends\n\n- The drunk and her dog (Murray 1994); four real pairs."),
    code(src(g.fig_drunk_dog, g.fig_examples) + "\n\n\nprint(fig_drunk_dog(save_it=False)['adf_dist'])\n"
         "ex = fig_examples(save_it=False)\n{k: (round(v['tau'], 2), round(v['p'], 3)) for k, v in ex.items() if isinstance(v, dict) and 'tau' in v}"),
    md("## 2. The Engle–Granger method\n\n- The null distribution by simulation; tests on real pairs; the step-1 residuals."),
    code(src(g.eg_tau_batch, g.fig_eg_dist, g.eg_table, g.fig_eg_steps)
         + "\n\n\nED = fig_eg_dist(save_it=False)\nprint({k: ED[k] for k in (1, 2, 3)})\nE = eg_table(save_it=False)\n"
           "display(pd.DataFrame({k: {'beta': v['beta'][0], 'tau': v['tau'], 'lags': v['lags'], 'p': v['p'], 'PO Zt': v['po_zt'], "
           "'PO p': v['po_p']} for k, v in E.items() if isinstance(v, dict)}).T.round(3))\nfig_eg_steps(save_it=False)"),
    md("## 3. Error correction models\n\n- Half-lives; ECMs for consumption and for the 10-year yield."),
    code(src(g.fig_ecm) + "\n\n\nfig_ecm(save_it=False)"),
    md("## 4. Johansen tests and the term-structure VECM"),
    code(src(g.johansen_systems, g.vecm_rates, g.fig_rates, g.fig_vecm_irf)
         + "\n\n\nJ = johansen_systems(save_it=False)\n"
           "display(pd.DataFrame({k: {'trace': [round(x, 1) for x in v['trace']], 'max-eig': [round(x, 1) for x in v['maxeig']], "
           "'rank': v['rank_trace']} for k, v in J.items() if k != 'yields_det'}).T)\n"
           "VE = vecm_rates(save_it=False)\nprint('beta', VE['beta'])\nprint('alpha', VE['alpha'])\nprint('t(alpha)', VE['alpha_t'])\n"
           "fig_rates(save_it=False)\nfig_vecm_irf(save_it=False)"),
    md("## 5. Forecasting: VECM against a VAR in differences\n\n- Re-estimation at every origin; this cell takes about a minute."),
    code(src(g.forecast_compare) + "\n\n\nFC = forecast_compare(save_it=False)\nprint(FC['ratio_vecm'])\nprint(FC['spread_vecm'])"),
    md("## 6. Economic examples: PPP, ROBOR and Euribor, three currencies"),
    code(src(g.fig_ppp, g.fig_ro_ea_rates, g.fig_cee_fx)
         + "\n\n\nprint(fig_ppp(save_it=False)['eg'])\nprint(fig_ro_ea_rates(save_it=False)['since2010'])\n"
           "print(fig_cee_fx(save_it=False)['johansen'])"),
    md("## 7. Pairs trading\n\n- Banca Transilvania and BRD; a rolling rule on 8 BVB shares and on 6 US banks."),
    code(src(g.pairs_backtest, g.perf, g.fig_pair, g.fig_pairs_backtest)
         + "\n\n\nprint(fig_pair(save_it=False))\nPB = fig_pairs_backtest(save_it=False)\n"
           "{k: (PB[k]['gross'], PB[k]['net'], PB[k]['trades']) for k in ('bvb', 'us')}"),
    md("## Exercises\n\n1. Test PPP for EUR/PLN with Polish and euro-area HICP (`prc_hicp_minr`, key `M.I15.TOTAL.PL`).\n"
       "2. Re-run the Johansen test for the yields with 4 and 8 lagged differences: does the rank change?\n"
       "3. Change the entry threshold of `pairs_backtest` to 1.5 and 2.5 and the cost to 0.10%: how do the net results change?"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_FUNCS = [s.eg_by_hand, s.ecm_by_hand, s.vecm_2x2, s.johansen_by_hand, s.var_to_vecm, s.b1_rates, s.b3_cee_fx]
SEMINAR = [
    md("# Time Series Analysis — Seminar 7: Cointegration and VECM\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The seminar comes before Lecture 7; the slides give the definitions (\"What you need today\").\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class.\n"
       "- The first code cell installs the `arch` package if it is missing."),
    *common_cells(INSTALL),
    md("## Seminar functions\n\n- Helpers of Chapter 7 (`adf_test`, `eg_test`, `ecm_fit`, `johansen`, `cee_fx`, `stock_panel`) "
       "and the seminar functions: `eg_by_hand`, `ecm_by_hand`, `vecm_2x2`, `johansen_by_hand`, `var_to_vecm`, `b1_rates`, `b3_cee_fx`."),
    code(CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_FUNCS)),
    code("# [solutions only]\n" + src(s.b2_ro_consumption, s.b4_ro_rates, s.c1_pairs, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: an Engle–Granger statistic\n\n"
       "**Context.** Two $I(1)$ series, $T = 200$; step 2 gives $\\Delta\\hat u_t = -0.118\\,\\hat u_{t-1}$, with SE $= 0.036$.\n\n"
       "1. Compute $\\tau$.\n2. Compare $\\tau$ with the 5% and 10% Engle–Granger values for 2 variables, and decide.\n"
       "3. Say what the Dickey–Fuller value would have concluded.\n4. Explain why the two critical values differ.\n\n"
       "**Report:** one number, two decisions and one sentence."),
    code("# Solution\neg_by_hand(-0.118, 0.036, 200, 2)"),
    md("**Interpretation of the result.** $\\tau = -3.28$ lies between the Engle–Granger 5% value (about $-3.37$) and the "
       "Dickey–Fuller value (about $-2.88$): no cointegration at 5%, a rejection at 10%; the Dickey–Fuller table would "
       "wrongly report cointegration."),
    md("## A2 [Proposed]: three variables\n\n"
       "**Context.** $y_t$ on $x_{1t}$ and $x_{2t}$, $T = 150$; $\\hat\\gamma = -0.162$, SE $= 0.041$. Model: A1.\n\n"
       "1. Compute $\\tau$.\n2. Decide at 5% and at 1% with the values for 3 variables.\n"
       "3. Say which critical value a student would wrongly use if he forgot $x_{2t}$.\n"
       "4. Say how many cointegrating vectors the Engle–Granger method can find with three variables.\n\n"
       "**Report:** one number, two decisions and two sentences."),
    code("# Solution\nprint(eg_by_hand(-0.162, 0.041, 150, 3))\nprint(eg_by_hand(-0.162, 0.041, 150, 2))"),
    md("## A3 [Solved]: an error correction model\n\n"
       "**Context.** $\\Delta c_t = 0.2 + 0.5\\,\\Delta y_t - 0.25\\,(c_{t-1} - 0.9\\,y_{t-1})$.\n\n"
       "1. Give the long-run and the short-run effect of income on consumption.\n"
       "2. Interpret the coefficient $-0.25$ and compute the half-life.\n"
       "3. Compute $\\Delta c_t$ if $c_{t-1} - 0.9\\,y_{t-1} = 2$ and $\\Delta y_t = 1$.\n"
       "4. Say what a coefficient of $+0.25$ would mean.\n\n**Report:** two elasticities, one half-life, one forecast and one sentence."),
    code("# Solution\necm_by_hand(c=0.2, d0=0.5, g=-0.25, b=0.9, gap=2.0, dx=1.0)"),
    md("**Interpretation of the result.** A quarter of the gap closes each period (half-life 2.41 periods); here the error "
       "correction ($-0.5$) cancels the short-run effect of income ($+0.5$), so consumption grows only by the constant."),
    md("## A4 [Proposed]: a bivariate VECM\n\n"
       "**Context.** $\\Delta y_{1t} = -0.20\\,z_{t-1} + u_{1t}$, $\\Delta y_{2t} = 0.05\\,z_{t-1} + u_{2t}$, $z_t = y_{1t} - y_{2t}$. Model: A3.\n\n"
       "1. Write $\\alpha$, $\\beta$ and $\\Pi = \\alpha\\beta^\\top$, and give the rank of $\\Pi$.\n"
       "2. Explain which variable does most of the adjusting.\n"
       "3. Show that $z_t = (1 + \\beta^\\top\\alpha)z_{t-1} + (u_{1t} - u_{2t})$ and compute its half-life.\n\n"
       "**Report:** one matrix, one rank, one half-life and one sentence."),
    code("# Solution\nvecm_2x2([-0.20, 0.05], [1.0, -1.0])"),
    md("## A5 [Solved]: Johansen statistics from eigenvalues\n\n"
       "**Context.** Three $I(1)$ series, $T = 200$, case with a constant; eigenvalues 0.120, 0.045, 0.008.\n\n"
       "1. Compute the trace statistics for $r = 0$, $r \\le 1$, $r \\le 2$.\n2. Compute the maximum-eigenvalue statistics.\n"
       "3. Compare with the 5% values and choose the rank.\n4. Give the number of common trends.\n\n"
       "**Report:** six statistics, one rank and one number of trends."),
    code("# Solution\njohansen_by_hand([0.120, 0.045, 0.008], 200)"),
    md("**Interpretation of the result.** Only $r = 0$ is rejected (trace 36.38 > 29.80): rank 1, so $3 - 1 = 2$ common trends."),
    md("## A6 [Proposed]: from a VAR(2) to a VECM\n\n"
       "**Context.** $A_1 = \\begin{pmatrix} 0.7 & 0.2 \\\\ 0.1 & 0.7 \\end{pmatrix}$, $A_2 = \\begin{pmatrix} 0.1 & 0 \\\\ 0 & 0.2 \\end{pmatrix}$. Model: A5.\n\n"
       "1. Compute $\\Pi = A_1 + A_2 - I$ and $\\Gamma_1 = -A_2$.\n"
       "2. Show that $\\Pi$ has rank 1 and write it as $\\alpha\\beta^\\top$ with $\\beta = (1, -1)^\\top$.\n3. Interpret the signs of $\\alpha$.\n\n"
       "**Report:** two matrices, $\\alpha$ and one sentence."),
    code("# Solution\nvar_to_vecm([[0.7, 0.2], [0.1, 0.7]], [[0.1, 0.0], [0.0, 0.2]])"),
    # ---------------- Part B
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: the 1-year and the 10-year US yields\n\n"
       "**Question.** Are the 1-year and the 10-year Treasury yields cointegrated, how fast does their gap close, and which rate closes it?\n\n"
       "1. Run ADF (with a constant) on both yields and on the change of the 10-year yield.\n"
       "2. Run Engle–Granger (10-year on 1-year) and Phillips–Ouliaris, and compare with the 5% Engle–Granger value.\n"
       "3. Estimate the ECM of the 10-year yield (one lag of each difference) and the same equation for the 1-year yield.\n"
       "4. Interpretation: if the 10-year yield is 1 percentage point above its equilibrium, how long until half of the gap is closed, and by which rate?\n\n"
       "**Report:** three ADF p-values, two cointegration tests, two adjustment coefficients and one sentence."),
    code("# Solution\nb1 = b1_rates(save=False)\n"
         "print({k: round(b1[k]['p'], 3) for k in ('adf_gs1', 'adf_gs10', 'adf_dgs10', 'adf_spread')})\n"
         "print({k: round(b1['eg'][k], 3) for k in ('tau', 'p', 'crit5', 'po_zt', 'po_p')}, 'beta', round(b1['eg']['beta'][0], 3))\n"
         "print('10y: gamma', round(b1['ecm_long']['gamma'], 4), 't', round(b1['ecm_long']['gamma_t'], 2), 'half-life', round(b1['ecm_long']['half'], 1))\n"
         "print('1y: gamma', round(b1['ecm_short']['gamma'], 4), 't', round(b1['ecm_short']['gamma_t'], 2))"),
    md("**Interpretation of the result.** The yields are cointegrated; the 10-year yield closes about 3% of the gap per month "
       "(half-life of about two years), while the 1-year yield, driven by monetary policy, does not react."),
    md("## B2 [Proposed]: consumption and GDP in Romania\n\n"
       "**Question.** Is Romanian household consumption cointegrated with GDP, as the \"great ratios\" suggest? Model: B1.\n\n"
       "1. Run ADF (constant and trend) on $c_t$ and $y_t$, and ADF (constant) on their changes "
       "(`read_eurostat(*CONS_RO)`, `read_eurostat(*GDP_RO)`, as $100\\ln$).\n"
       "2. Run Engle–Granger ($c_t$ on $y_t$) and Phillips–Ouliaris, and report the slope.\n"
       "3. Plot the ratio of consumption to GDP and run ADF on $c_t - y_t$.\n"
       "4. Interpretation: does a stable consumption share make sense for Romania over 1995–2026?\n\n"
       "**Report:** four ADF tests, two cointegration tests, one chart and two sentences."),
    code("# Solution\nb2_ro_consumption(save=False)"),
    md("## B3 [Solved]: three Central European currencies\n\n"
       "**Question.** Do EUR/RON, EUR/HUF and EUR/PLN share a long-run equilibrium?\n\n"
       "1. Run ADF on each rate.\n2. Run the Johansen trace and maximum-eigenvalue tests with a constant and lags by BIC.\n"
       "3. Compare with the three pairwise Engle–Granger tests and with the correlations of levels and of changes.\n"
       "4. Interpretation: should the three rates be modelled with a VECM?\n\n"
       "**Report:** three ADF p-values, a table of six Johansen statistics, a rank and one sentence."),
    code("# Solution\nb3 = b3_cee_fx(save=False)\nj = b3['johansen']\n"
         "display(pd.DataFrame({'trace': j['trace'], 'trace 5%': j['trace_cv5'], 'max-eig': j['maxeig'], 'max-eig 5%': j['maxeig_cv5']}, "
         "index=['r = 0', 'r <= 1', 'r <= 2']).round(2))\nprint('rank', j['rank_trace'], {k: round(v['p'], 2) for k, v in b3['adf'].items()})\n"
         "print('EG p:', round(b3['eg_ron_huf']['p'], 2), round(b3['eg_ron_pln']['p'], 2), round(b3['eg_huf_pln']['p'], 2))\n"
         "print('correlation of levels', round(b3['corr_levels'], 2), 'of changes', round(b3['corr_changes'], 2))"),
    md("**Interpretation of the result.** Rank 0: no long-run equilibrium; a VAR in differences (Chapter 6) is the right model, "
       "although the levels are highly correlated."),
    md("## B4 [Proposed]: ROBOR, the Romanian 10-year yield and Euribor\n\n"
       "**Question.** How many long-run relations tie Romanian interest rates to each other and to Euribor, and which rate adjusts? Model: B3.\n\n"
       "1. Run the Johansen tests (constant, BIC lags) and choose the rank (`read_eurostat(*ROBOR)`, `read_eurostat(*EURIBOR)`, "
       "`load_close('ro10y')` as monthly means, since 2010).\n"
       "2. Estimate the VECM with that rank and the constant restricted to the cointegrating relation; normalise $\\beta$ on ROBOR.\n"
       "3. Report $\\hat\\alpha$ with $t$-statistics and run ADF on the equilibrium error.\n"
       "4. Interpretation: which rates are weakly exogenous, and what does this say about Romanian monetary conditions?\n\n"
       "**Report:** a table of Johansen statistics, $\\hat\\beta$, $\\hat\\alpha$ with $t$, one ADF test and two sentences."),
    code("# Solution\nb4_ro_rates(save=False)"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: pairs trading on two banks\n\n"
       "**Question.** Would a trader who found Banca Transilvania and BRD cointegrated at the end of 2021 have earned money "
       "afterwards? Models: B1, B3.\n\n"
       "1. On 2014–2021, run Engle–Granger and Phillips–Ouliaris (TLV on BRD) and keep $\\hat b$, the mean and the standard "
       "deviation of the spread (`stock_panel(PAIR, '2014-01-01')`).\n"
       "2. On January 2022–September 2026, apply the rule: open at $|z| > 2$, close at $z = 0$; compute the annual return "
       "gross and net of 0.20% per unit traded.\n"
       "3. Repeat the Engle–Granger test on the trading period alone.\n"
       "4. Interpretation: is the result evidence that pairs trading works on the BVB?\n\n"
       "**Report:** two tests, the number of trades, two returns and a plan for a project."),
    code("# Reference analysis\nc1_pairs(save=False)"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant answered questions on cointegration:\n\n"
       "- (a) the Engle–Granger statistic is $-3.0$ with two variables; since $-3.0$ is below the 5% Dickey–Fuller value, the series are cointegrated;\n"
       "- (b) the ECM coefficient of the lagged equilibrium error is $+0.15$, so 15% of the gap is corrected each period;\n"
       "- (c) Johansen rejects $r = 0$ and $r \\le 1$ but not $r \\le 2$ for three yields, so the yields share two common trends;\n"
       "- (d) EUR/RON and EUR/HUF are both $I(1)$ with a high $R^2$ in levels, so they are cointegrated;\n"
       "- (e) if two series are cointegrated, at least one of them Granger-causes the other;\n"
       "- (f) a pairs-trading rule with a Sharpe ratio of 0.44 on 2014–2026 data will keep this Sharpe ratio in the future.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 7, 'lecture')
    build(SEMINAR, 7, 'seminar')
