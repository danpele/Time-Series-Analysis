"""
build_notebooks_ch14.py -- lecture notebook of Chapter 14 (TSA): multivariate GARCH models (self-study, no seminar)
================================================================================================================
Output: notebooks/EN/chapter14_lecture_notebook.ipynb
The code is taken from Quantlets/Ch_14/generate_all_charts.py (inspect.getsource), so the notebook stays in sync with
the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch14.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter14_lecture_notebook.ipynb
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(14)
import generate_all_charts as g   # noqa: E402
import build_quantlets as bq      # noqa: E402

CONSTS = '\n'.join(bq.CONSTS)

LECTURE = [
    md("# Time Series Analysis — Chapter 14: Multivariate GARCH models\n\n"
       "*Lecture notebook (self-study chapter). Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Why multivariate volatility: portfolio risk, hedging, contagion; rolling correlations; asynchronous trading.\n"
       "- The number of parameters of VEC, BEKK, CCC and DCC models.\n"
       "- DCC (Engle 2002) in two steps: a GARCH(1,1) per series, then the correlation parameters $(a, b)$; the test "
       "against constant correlation (CCC, Bollerslev 1990); the crises of 2008 and 2020.\n"
       "- A five-asset DCC: S&P 500, DAX, BET, EUR/RON, Bitcoin.\n"
       "- Applications: portfolio VaR 1% with backtesting (Kupiec, Christoffersen) and dynamic hedge ratios.\n"
       "- The first code cell installs the `arch` package if it is missing; without it a numpy GARCH estimator is used.\n"
       "- References: Bollerslev, Engle and Wooldridge (1988); Bollerslev (1990); Engle and Kroner (1995); Engle (2002); "
       "Kroner and Sultan (1993); Kupiec (1995); Christoffersen (1998)."),
    *common_cells(bq.INSTALL),
    md("## Definitions and helpers used in the whole notebook\n\n"
       "- `common_returns(names, start)`: daily log returns in % on the days on which all markets trade.\n"
       "- `garch11_fit(r)`, `garch11_filter(r, p)`: step 1, a GARCH(1,1) per series (one-step-ahead volatility, "
       "standardised residuals).\n"
       "- `dcc_path(Z, a, b, Qbar)`: the correlation matrices $\\mathbf{R}_t$; `dcc_loglik`, `dcc_fit`: step 2 by maximum "
       "likelihood with correlation targeting ($\\bar{\\mathbf{Q}}$ = sample correlation of $\\mathbf{z}_t$).\n"
       "- The `arch` package has no DCC model: step 2 is written out here in a few lines of numpy."),
    code(CONSTS + '\n\n\n' + src(*bq.CORE)),
    md("## 1. Why multivariate volatility\n\n- The worked examples of the slides; diversification against the correlation; "
       "rolling correlations; the asynchronous closes of 9-10 April 2025."),
    code(src(g.worked_examples, g.fig_diversification, g.fig_rolling_corr)
         + "\n\n\nprint(worked_examples())\nfig_diversification(save_it=False)\nrc = fig_rolling_corr(save_it=False)\n"
           "print({k: v for k, v in rc.items() if k != 'apr2025'})\nprint(rc['apr2025'])"),
    md("## 2. The number of parameters"),
    code(src(g.n_params, g.fig_param_count) + "\n\n\nprint(pd.DataFrame(fig_param_count(save_it=False)))"),
    md("## 3. DCC for the S&P 500 and the DAX\n\n- Step 1: GARCH(1,1) for each index; step 2: $(a, b)$; LR test against CCC; "
       "the crises of 2008 and 2020."),
    code(src(g.dcc_sp_dax, g.fig_step1, g.fig_dcc_sp_dax, g.fig_crisis_zoom)
         + "\n\n\nprint(fig_step1(save_it=False))\nd = fig_dcc_sp_dax(save_it=False)\n"
           "print({k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()})\nprint(fig_crisis_zoom(save_it=False))"),
    md("### Check: does the code recover known parameters?\n\n- Simulate two series with a DCC correlation ($a = 0.04$, "
       "$b = 0.94$, $\\bar\\rho = 0.5$) and unit variances, then estimate $(a, b)$ by step 2."),
    code("rng = np.random.default_rng(2026)\nT, a0, b0, rho0 = 3000, 0.04, 0.94, 0.5\nQbar = np.array([[1, rho0], [rho0, 1]])\n"
         "Q, Zs = Qbar.copy(), np.empty((T, 2))\nfor t in range(T):\n"
         "    d = np.sqrt(np.diag(Q))\n    R = Q / np.outer(d, d)\n    Zs[t] = np.linalg.cholesky(R) @ rng.standard_normal(2)\n"
         "    Q = (1 - a0 - b0) * Qbar + a0 * np.outer(Zs[t], Zs[t]) + b0 * Q\n"
         "f = dcc_fit(Zs)\nprint('true a, b:', a0, b0, '  estimated:', round(f['a'], 4), round(f['b'], 4), '  SE:', round(f['se_a'], 4), round(f['se_b'], 4))"),
    md("## 4. Five markets\n\n- One DCC for the S&P 500, the DAX, the BET, EUR/RON and Bitcoin; calm against crisis."),
    code(src(g.dcc_panel, g.fig_panel_corr, g.fig_panel_heatmap)
         + "\n\n\np = fig_panel_corr(save_it=False)\nprint({k: p[k] for k in ('a', 'b', 'lr', 'n')})\n"
           "print({k: v for k, v in p.items() if k.startswith('async')})\nprint(fig_panel_heatmap(save_it=False))"),
    md("## 5. Portfolio VaR 1% and backtesting\n\n- Parameters estimated on 2000–2014; the filters run on 2015–2026 without "
       "re-estimation."),
    code(src(g.kupiec, g.christoffersen, g.portfolio_var, g.fig_var_backtest)
         + "\n\n\nv = fig_var_backtest(save_it=False)\n"
           "print(pd.DataFrame({k: {'violations': v[k]['x'], 'rate': v[k]['rate'], 'Kupiec p': v[k]['p'], "
           "'independence p': v[k]['ind']['p']} for k in ('DCC', 'DCC-FHS', 'CCC', 'Static')}).T.round(4))"),
    md("## 6. Dynamic hedge ratios\n\n- BET hedged with the DAX: DCC, rolling OLS and static OLS, out of sample."),
    code(src(g.hedge, g.fig_hedge) + "\n\n\nh = fig_hedge(save_it=False)\n"
         "print({k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()})"),
    md("## Exercises\n\n1. Re-estimate the S&P 500 and DAX DCC on weekly returns (`weekly_returns(['sp500', 'dax'], "
       "'2000-01-01')`): how do $a$, $b$ and the mean correlation change?\n"
       "2. Replace the Normal quantile of the VaR by the quantile of a Student $t$ distribution with 5 degrees of freedom "
       "(scaled to unit variance) and repeat the Kupiec test.\n"
       "3. Hedge the BET with the S&P 500 instead of the DAX: compare the hedging effectiveness and explain the difference "
       "with the asynchronous closes."),
]

if __name__ == '__main__':
    build(LECTURE, 14, 'lecture')
