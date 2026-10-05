"""
build_notebooks_ch5.py -- lecture and seminar notebooks of Chapter 5 (TSA): conditional volatility, ARCH and GARCH
=================================================================================================================
Output: notebooks/EN/chapter5_lecture_notebook.ipynb, notebooks/EN/chapter5_seminar_notebook.ipynb
The code is taken from Quantlets/Ch_05/generate_all_charts.py and seminar5.py (inspect.getsource), so the notebooks
stay in sync with the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch5.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter5_lecture_notebook.ipynb
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter5_seminar_notebook.ipynb
      python3 notebooks/split_seminar_notebooks.py 5
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(5)
import generate_all_charts as g   # noqa: E402
import seminar5 as s              # noqa: E402

INSTALL = ("# The arch package (ARCH/GARCH estimation) is not preinstalled in Google Colab: pip install arch\n"
           "import importlib.util, subprocess, sys\n"
           "if importlib.util.find_spec('arch') is None:\n"
           "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'arch'])")
CONSTS = ('import statsmodels.api as sm\nfrom statsmodels.stats.diagnostic import acorr_ljungbox\nfrom arch import arch_model\n'
          f'SEED = {g.SEED!r}\nASSETS = {g.ASSETS!r}\nNAME = {g.NAME!r}\nSTART = {g.START!r}\nCOLORS = {g.COLORS!r}\n'
          f'SCALE = {g.SCALE!r}\nEPISODES = {g.EPISODES!r}\nOOS_START = {g.OOS_START!r}\nREFIT = {g.REFIT!r}\n'
          f'LAMBDA = {g.LAMBDA!r}\nALPHA_VAR = {g.ALPHA_VAR!r}\nLB_LAGS = {g.LB_LAGS!r}\nARCH_LAGS = {g.ARCH_LAGS!r}\n'
          f'IG = {g.IG!r}\nTERM_DATES = {g.TERM_DATES!r}')
CORE = [g.returns, g.save, g.acf_vals, g.ljung_box, g.arch_lm, g.fit, g.par, g.sig, g.persistence, g.half_life,
        g.summary, g.ols_ar1, g.ar_order, g.ewma_variance, g.news_impact, g.sign_bias_test, g.oos_forecasts, g.qlike,
        g.dm_test]

# =============================================================================
# LECTURE
# =============================================================================
LECTURE = [
    md("# Time Series Analysis — Chapter 5: Conditional volatility, ARCH and GARCH\n\n"
       "*Lecture notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Stylised facts of daily returns (S&P 500, BET, EUR/RON, Bitcoin); the ACF of squared returns; the ARCH-LM test.\n"
       "- The mean model of Chapter 2 and its residuals; ARCH(1), ARCH(q) and GARCH(1,1); persistence, IGARCH and EWMA.\n"
       "- Maximum-likelihood estimation with the `arch` package; Student-t and skewed-t innovations; AR(1)-GARCH(1,1).\n"
       "- Asymmetry (GJR-GARCH, EGARCH, news impact curves); diagnostics; multi-step forecasts; QLIKE and the "
       "Diebold–Mariano test; VaR 1%.\n"
       "- References: Huang and Petukhina (2022), *Applied Time Series Analysis and Forecasting with Python*, Ch. 6; "
       "Tsay (2010), *Analysis of Financial Time Series*, Ch. 3; Bollerslev (1986)."),
    *common_cells(INSTALL),
    md("## Definitions and helpers used in the whole notebook\n\n"
       "- Returns in %: $r_t = 100(\\ln P_t - \\ln P_{t-1})$, `returns(k)` with `k` in `'sp500'`, `'bet'`, `'eurron'`, `'btc'`.\n"
       "- ARCH-LM: regress $\\hat\\varepsilon_t^2$ on a constant and $q$ of its lags; $\\mathrm{LM} = nR^2 \\sim \\chi^2(q)$: `arch_lm(x, q)`.\n"
       "- GARCH(1,1): $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$; `fit(r, vol, dist, o, mean)` wraps "
       "`arch_model`; EUR/RON is estimated on $10\\,r_t$ (`SCALE`) and `par`, `sig`, `summary` convert the results back to %."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Stylised facts\n\n- Moments, Ljung–Box on returns and squared returns, ARCH-LM; the four series; their ACFs."),
    code(src(g.stylised_table, g.fig_returns, g.fig_acf_squares)
         + "\n\n\nS = stylised_table()\n"
           "display(pd.DataFrame({k: {'n': v['n'], 'sd (%)': v['sd'], 'skewness': v['skew'], 'kurtosis': v['kurt'], "
           "'Q(10) r': v['lb_r'][0], 'Q(10) r^2': v['lb_r2'][0], 'ARCH-LM(5)': v['arch']['lm']} for k, v in S.items()}).T.round(3))\n"
           "fig_returns(save_it=False)\nfig_acf_squares(save_it=False)"),
    md("## 2. The mean model leaves ARCH effects\n\n- An AR(1) for the BET returns; ARCH-LM step by step; residual ACFs "
       "before and after AR(1)-GARCH(1,1)-t."),
    code(src(g.fig_arma_resid) + "\n\n\nA = fig_arma_resid(save_it=False)\n"
         "print('AR order by BIC:', A['p_bic'], ' phi =', round(A['phi'], 3))\n"
         "print('ARCH-LM on the AR(1) residuals: R2 =', round(A['archlm_e']['r2'], 4), ' n =', A['archlm_e']['n'], "
         "' LM = n R2 =', round(A['archlm_e']['lm'], 1), ' 5% critical value =', round(A['archlm_e']['crit'], 2))"),
    md("## 3. ARCH and GARCH by simulation; the likelihood\n\n- Paths with the same unconditional variance; the ARCH(1) "
       "log-likelihood for two sample sizes; the ridge of the GARCH(1,1) likelihood."),
    code(src(g.simulate_garch, g.fig_simulated, g.arch1_loglik, g.fig_lik_arch1, g.garch_loglik_vt, g.fig_lik_garch)
         + "\n\n\nprint(fig_simulated(save_it=False))\nprint(fig_lik_arch1(save_it=False))\nprint(fig_lik_garch(save_it=False))"),
    md("## 4. Estimation by maximum likelihood\n\n- ARCH(q) against GARCH(1,1); GARCH(1,1) step by step with scipy and with "
       "`arch`; classic and robust standard errors; three innovation distributions; QQ plots."),
    code(src(g.garch11_negloglik, g.fit_step_by_step, g.arch_q_table, g.estimation_sp500, g.fig_qq)
         + "\n\n\ndisplay(pd.DataFrame(arch_q_table()).T.round(3))\nE = estimation_sp500()\n"
           "display(pd.DataFrame({'step by step': E['step']['params'], 'arch': E['normal']['params']}).round(4))\n"
           "display(pd.DataFrame({'classic SE': E['normal']['se_classic'], 'robust SE': E['normal']['se']}).round(4))\n"
           "fig_qq(save_it=False)"),
    md("## 5. GARCH(1,1)-t in four markets; EWMA against GARCH\n\n- Persistence, half-life, long-run volatility; "
       "conditional volatility; the decay of a shock; the COVID-19 crash."),
    code(src(g.markets_table, g.fig_vol, g.fig_persistence, g.fig_ewma_garch)
         + "\n\n\nT = markets_table()\n"
           "display(pd.DataFrame({k: {'omega': v['params']['omega'], 'alpha': v['params']['alpha[1]'], 'beta': v['params']['beta[1]'], "
           "'nu': v['params']['nu'], 'persistence': v['pers'], 'half-life': v['hl'], 'long-run vol': v.get('vol_lr'), "
           "'sample vol': v['vol_sample']} for k, v in T.items()}).T)\n"
           "fig_vol(('sp500', 'bet'), save_it=False)\nfig_vol(('eurron', 'btc'), save_it=False)\n"
           "fig_persistence(T, save_it=False)\nfig_ewma_garch(save_it=False)"),
    md("## 6. ARMA-GARCH\n\n- AR(1) by OLS against AR(1)-GARCH(1,1)-t; intervals with a constant and with a GARCH variance."),
    code(src(g.arma_garch_table, g.fig_bands)
         + "\n\n\nAG = arma_garch_table()\ndisplay(pd.DataFrame({k: {'p (BIC)': v['p_bic'], 'phi OLS': v['phi_ols'], 'SE OLS': v['se_ols'], "
           "'phi AR-GARCH': v['phi_g'], 'SE AR-GARCH': v['se_g'], 'Q(10) z, AR(1)': v['lbz_a'][0]} for k, v in AG.items()}).T.round(3))\n"
           "print(fig_bands(save_it=False))"),
    md("## 7. Asymmetry\n\n- News impact curves; GJR and EGARCH on four series; the sign-bias test."),
    code(src(g.kernel_nic, g.fig_nic, g.asym_table)
         + "\n\n\nprint(fig_nic(save_it=False))\nAS = asym_table()\n"
           "display(pd.DataFrame({k: {'GJR alpha': v['gjr_alpha'], 'GJR gamma': v['gjr_gamma'], 't': v['gjr_t'], 'LR': v['lr'], "
           "'p': v['lr_p'], 'EGARCH gamma': v['eg_gamma'], 'sign bias': v['sb']['joint']} for k, v in AS.items()}).T.round(3))"),
    md("## 8. Diagnostics and model choice\n\n- Standardised residuals; the EUR/RON day of 6 May 2025; nine models by BIC."),
    code(src(g.diagnostics, g.diag_markets, g.fig_acf_diag, g.managed_rate_case, g.model_selection)
         + "\n\n\nprint(diagnostics())\nfig_acf_diag(save_it=False)\nprint(managed_rate_case())\n"
           "display(model_selection().round(1))"),
    md("## 9. Forecasts, their evaluation and VaR 1%\n\n- The term structure of volatility; GARCH-t, GJR-t and EWMA out of "
       "sample (QLIKE, Diebold–Mariano); VaR 1% exceedances."),
    code(src(g.garch_path_forecast, g.fig_term_structure, g.forecast_eval, g.fig_forecast_eval, g.fig_var)
         + "\n\n\nfig_term_structure(save_it=False)\nfe, frames = forecast_eval()\n"
           "display(pd.DataFrame({k: {'QLIKE GARCH-t': v['qlike']['garch'], 'QLIKE GJR-t': v['qlike']['gjr'], 'QLIKE EWMA': v['qlike']['ewma'], "
           "'DM t': v['dm_garch_ewma']['t'], 'VaR 1% exc. GARCH-t (%)': 100 * v['exc_garch'], "
           "'VaR 1% exc. EWMA (%)': 100 * v['exc_ewma']} for k, v in fe.items()}).T.round(3))\n"
           "fig_forecast_eval(frames, save_it=False)\nfig_var(frames, save_it=False)"),
    md("## Exercises\n\n1. Estimate GARCH(1,1)-t for the DAX (`returns('dax')`, after adding `'dax'` to `START`) and compare "
       "its persistence with the S&P 500.\n"
       "2. Re-estimate the S&P 500 GARCH(1,1)-t on 2000–2012 and on 2013–2026: is the persistence stable?\n"
       "3. Change `LAMBDA` to 0.97 and repeat the out-of-sample comparison: does EWMA improve?"),
]

# =============================================================================
# SEMINAR
# =============================================================================
SEM_FUNCS = [s.garch_algebra, s.archlm_by_hand, s.b1_estimate, s.b3_asymmetry, s.b5_oos]
SEMINAR = [
    md("# Time Series Analysis — Seminar 5: Conditional volatility, ARCH and GARCH\n\n"
       "*Seminar notebook. Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- The seminar comes before Lecture 5; the slides give the definitions (\"What you need today\").\n"
       "- [Solved] tasks: full solution here and in the slides; [Proposed] tasks: write your own code in the empty cell.\n"
       "- The first code cell installs the `arch` package if it is missing (`pip install arch`).\n"
       "- The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class."),
    *common_cells(INSTALL),
    md("## Seminar functions\n\n- Helpers of Chapter 5 (`returns`, `arch_lm`, `ljung_box`, `fit`, `par`, `sig`, `summary`, "
       "`ols_ar1`, `oos_forecasts`, `qlike`, `dm_test`) and the seminar functions `garch_algebra`, `archlm_by_hand`, "
       "`b1_estimate`, `b3_asymmetry`, `b5_oos`."),
    code(CONSTS + '\n\n\n' + src(*CORE) + '\n\n\n' + src(*SEM_FUNCS)),
    code("# [solutions only]\n" + src(s.garch_forecasts, s.asym_news, s.b2_table, s.b4_asymmetry, s.b6_oos, s.c1_eurron_periods, s.c2_check)),
    # ---------------- Part A
    md("# Part A: computations on paper\n\n- Do each computation on paper first; then check it with the code."),
    md("## A1 [Solved]: GARCH(1,1) algebra\n\n"
       "**Context.** Daily returns in %: $\\omega = 0.02$, $\\alpha = 0.08$, $\\beta = 0.90$, $\\mu = 0$, 252 trading days per year.\n\n"
       "1. Compute the persistence and the long-run variance.\n2. Compute the annualised long-run volatility.\n"
       "3. Compute the half-life of a variance shock.\n4. Today $\\sigma_t^2 = 1.5$ and $r_t = -3\\%$; compute $\\sigma_{t+1}^2$ and $\\sigma_{t+1}$.\n\n"
       "**Report:** four numbers and one sentence."),
    code("# Solution\ngarch_algebra(0.02, 0.08, 0.90, s2_t=1.5, r_t=-3.0)"),
    md("**Interpretation of the result.** Persistence 0.98, long-run variance 1 (15.9% per year), half-life about 34 days; "
       "a fall of 3% lifts tomorrow's variance from 1.5 to 2.09."),
    md("## A2 [Proposed]: a more reactive market\n\n"
       "**Context.** $\\omega = 0.05$, $\\alpha = 0.12$, $\\beta = 0.85$, $\\mu = 0$; today $\\sigma_t^2 = 2.0$ and $r_t = +1.5\\%$. Model: A1.\n\n"
       "1. Compute the persistence, the long-run variance and the annualised long-run volatility.\n2. Compute the half-life.\n"
       "3. Compute $\\sigma_{t+1}^2$.\n4. Write EWMA with $\\lambda = 0.94$ as a GARCH(1,1) and say which of the quantities in 1–2 exist for it.\n\n"
       "**Report:** four numbers and two sentences."),
    code("# Solution\ngarch_algebra(0.05, 0.12, 0.85, s2_t=2.0, r_t=1.5)"),
    md("## A3 [Solved]: forecasts and VaR 1%\n\n"
       "**Context.** The model of A1, with $\\sigma_{t+1}^2 = 2.09$ and Normal innovations.\n\n"
       "1. Compute $E_t[\\sigma_{t+5}^2]$ and $E_t[\\sigma_{t+10}^2]$.\n2. Compute the 10-day variance and volatility.\n"
       "3. Compare with the square-root-of-time rule $\\sqrt{10}\\,\\sigma_{t+1}$ and with $\\sqrt{10\\,\\bar\\sigma^2}$.\n"
       "4. Compute the one-day and the 10-day VaR 1%.\n\n**Report:** six numbers and one sentence."),
    code("# Solution\ns2n = garch_algebra(0.02, 0.08, 0.90, 1.5, -3.0)['s2_next']\n"
         "pers, s2bar = 0.98, 1.0\nf = {h: s2bar + pers ** (h - 1) * (s2n - s2bar) for h in (5, 10)}\n"
         "sum10 = sum(s2bar + pers ** (h - 1) * (s2n - s2bar) for h in range(1, 11))\nq = stats.norm.ppf(0.01)\n"
         "print('forecasts:', {h: round(v, 3) for h, v in f.items()})\n"
         "print('10-day variance:', round(sum10, 2), ' volatility:', round(np.sqrt(sum10), 2))\n"
         "print('sqrt-of-time:', round(np.sqrt(10 * s2n), 2), ' long run:', round(np.sqrt(10 * s2bar), 2))\n"
         "print('VaR 1%: one day', round(-q * np.sqrt(s2n), 2), ' 10 days', round(-q * np.sqrt(sum10), 2))"),
    md("**Interpretation of the result.** The high variance decays slowly towards 1: the 10-day volatility lies between the "
       "square-root-of-time value and the long-run value, closer to the first."),
    md("## A4 [Proposed]: forecasts with Student-t innovations\n\n"
       "**Context.** The model of A2, with $\\sigma_{t+1}^2 = 2.02$ and standardised Student-t innovations with $\\nu = 5$. Model: A3.\n\n"
       "1. Compute $E_t[\\sigma_{t+10}^2]$ and the 10-day variance.\n2. Compute the quantile $q_{0.01}$ of the standardised t.\n"
       "3. Compute the one-day and the 10-day VaR 1% with t and with Normal innovations.\n\n**Report:** six numbers and one sentence."),
    code("# Solution\nprint(garch_forecasts(0.05, 0.12, 0.85, 2.02, nu=5))\nprint(garch_forecasts(0.05, 0.12, 0.85, 2.02))"),
    md("## A5 [Solved]: an ARCH-LM test from regression output\n\n"
       "**Context.** An AR(1) is fitted to the daily returns of a stock index. The regression of $\\hat\\varepsilon_t^2$ on a "
       "constant and five lags uses $n = 1000$ observations and gives $R^2 = 0.062$; the Ljung–Box statistic of "
       "$\\hat\\varepsilon_t^2$ is $Q(10) = 95.3$.\n\n"
       "1. Compute the ARCH-LM statistic and compare it with the 5% value of $\\chi^2(5)$.\n"
       "2. Compare $Q(10)$ with the 5% value of $\\chi^2(10)$.\n3. Say which model you would fit next.\n"
       "4. For an ARCH(1) with $\\omega = \\alpha = 0.5$ and Normal $z_t$, compute the unconditional variance and the kurtosis.\n\n"
       "**Report:** two decisions, two numbers and one sentence."),
    code("# Solution\narchlm_by_hand(r2=0.062, n=1000, q=5, Q=95.3, m=10, omega=0.5, alpha=0.5)"),
    md("**Interpretation of the result.** LM = 62 and Q(10) = 95.3 are far above 11.07 and 18.31: strong ARCH effects, so "
       "the next model is an AR(1)-GARCH(1,1). The ARCH(1) of part 4 has variance 1 and kurtosis 9: Normal innovations, "
       "heavy-tailed returns."),
    md("## A6 [Proposed]: good and bad news\n\n"
       "**Context.** GJR-GARCH: $\\omega = 0.02$, $\\alpha = 0.03$, $\\gamma = 0.12$, $\\beta = 0.90$; EGARCH: $\\omega = 0$, "
       "$\\alpha = 0.12$, $\\gamma = -0.08$, $\\beta = 0.98$; in both $\\sigma_t^2 = 1$, $\\mu = 0$. Model: A1.\n\n"
       "1. Compute the GJR $\\sigma_{t+1}^2$ after $\\varepsilon_t = -2$ and after $\\varepsilon_t = +2$, and their ratio.\n"
       "2. Compute the GJR persistence and half-life.\n3. Compute the EGARCH $\\sigma_{t+1}^2$ for the same two shocks.\n\n"
       "**Report:** six numbers and one sentence."),
    code("# Solution\nasym_news(0.02, 0.03, 0.12, 0.90, 1.0, eg=(0.0, 0.12, -0.08, 0.98))"),
    # ---------------- Part B
    md("# Part B: real data and interpretation"),
    md("## B1 [Solved]: ARCH effects and AR(1)-GARCH(1,1)-t for the BET\n\n"
       "**Question.** Does the BET have ARCH effects, how persistent is its volatility, and where is it today relative to "
       "its long-run level?\n\n"
       "1. Fit an AR(1) for the mean by OLS, then run ARCH-LM(5) and Ljung–Box $Q(10)$ on its squared residuals.\n"
       "2. Estimate AR(1)-GARCH(1,1)-t and report the parameters with robust standard errors.\n"
       "3. Compute the persistence, the half-life and the annualised long-run volatility, and compare them with the sample volatility.\n"
       "4. Check $Q(10)$ of $\\hat z_t$ and of $\\hat z_t^2$, and draw the returns since 2018 with $\\pm 2\\hat\\sigma_t$ bands.\n"
       "5. Interpretation: is the BET volatility on the last day above or below its long-run level, and what does the model "
       "expect for the next months?\n\n"
       "**Report:** two test statistics, a table of six parameters, five numbers, the chart and two sentences."),
    code("# Solution\nb1 = b1_estimate('bet', save=False)\n"
         "print('ARCH-LM(5) on the AR(1) residuals:', round(b1['lm_e']['lm'], 1), ' Q(10) of squares:', round(b1['lb_e2'][0], 1))\n"
         "display(pd.DataFrame({'estimate': b1['params'], 'robust SE': b1['se']}).round(4))\n"
         "print('persistence', round(b1['pers'], 4), ' half-life', round(b1['hl'], 1), ' long-run vol', round(b1['vol_lr'], 1), "
         "' sample vol', round(b1['vol_sample'], 1), ' last day', round(b1['last_vol'], 1))\n"
         "print('Q(10) z:', b1['lb_z'], ' Q(10) z^2:', b1['lb_z2'])"),
    md("**Interpretation of the result.** The BET volatility on the last day is below its long-run level, so the model "
       "expects it to rise slowly; the long-run level is imprecise because $1 - \\alpha - \\beta$ is about 0.01."),
    md("## B2 [Proposed]: EUR/RON and Bitcoin\n\n"
       "**Question.** Do the EUR/RON rate and Bitcoin behave like the BET? Model: B1.\n\n"
       "1. Repeat steps 1–4 of B1 for both series (`b1_estimate('eurron', save=False)` estimates EUR/RON on $10\\,r_t$).\n"
       "2. Report $\\hat\\alpha + \\hat\\beta$ and say whether the half-life and the long-run volatility exist.\n"
       "3. Interpretation: what does $\\hat\\alpha + \\hat\\beta = 1$ mean for a forecast of the volatility one year ahead?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nB2 = b2_table()\n"
         "pd.DataFrame({k: {'ARCH-LM': v['lm_e']['lm'], 'phi': v['params'][v['ar_name']], 'alpha': v['params']['alpha[1]'], "
         "'beta': v['params']['beta[1]'], 'nu': v['params']['nu'], 'persistence': v['pers'], 'Q(10) z^2': v['lb_z2'][0]} for k, v in B2.items()}).T"),
    md("## B3 [Solved]: the leverage effect in the S&P 500\n\n"
       "**Question.** Do falls raise the volatility of the S&P 500 more than rises of the same size?\n\n"
       "1. Estimate GARCH(1,1)-t and GJR-GARCH(1,1)-t; report $\\gamma$ with its robust $t$ statistic.\n"
       "2. Compute the LR statistic of GJR against GARCH, its p-value and the two BIC values.\n"
       "3. Run the sign-bias test on the standardised residuals of GARCH-t.\n"
       "4. Draw both news impact curves and compute the GJR variance after shocks of −2% and +2%.\n"
       "5. Interpretation: how does the leverage effect change the VaR on the day after a fall?\n\n"
       "**Report:** four statistics with p-values, two variances, the chart and two sentences."),
    code("# Solution\nb3 = b3_asymmetry('sp500', save=False)\nb3"),
    md("**Interpretation of the result.** Only falls raise the variance ($\\hat\\alpha = 0$); after a fall of 2% the variance is "
       "about 1.6 times the variance after a rise of 2%, so the VaR is about 1.27 times larger."),
    md("## B4 [Proposed]: asymmetry in the BET, EUR/RON and Bitcoin\n\n"
       "**Question.** Is there a leverage effect in the BET, in EUR/RON and in Bitcoin? Model: B3.\n\n"
       "1. Estimate GJR-GARCH(1,1)-t and EGARCH(1,1)-t; report both values of $\\gamma$ with robust $t$ statistics.\n"
       "2. Compute the LR test of GJR against GARCH and the change in BIC.\n"
       "3. Interpretation: for EUR/RON a positive return means a weaker leu; what is \"bad news\" for this series?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nB4 = b4_asymmetry()\npd.DataFrame({k: {'gamma': v['gamma'], 't': v['t_gamma'], 'LR': v['lr'], 'p': v['p_lr'], "
         "'dBIC': v['bic_gjr'] - v['bic_garch'], 'EGARCH gamma': v['eg_gamma'], 't EGARCH': v['eg_t']} for k, v in B4.items()}).T"),
    md("## B5 [Solved]: GARCH against EWMA for the S&P 500\n\n"
       "**Question.** Does GARCH(1,1)-t forecast tomorrow's variance of the S&P 500 better than EWMA, and is its VaR 1% "
       "exceeded on 1% of the days?\n\n"
       "1. Re-estimate GARCH(1,1)-t every 250 days on all data up to that day and compute the one-day forecasts $h_t$; compute EWMA with $\\lambda = 0.94$.\n"
       "2. Compute the mean QLIKE of both models and the Diebold–Mariano $t$ statistic.\n"
       "3. Count the exceedances of the GARCH-t VaR 1% and of the EWMA-Normal VaR 1%, and compare them with the expected number.\n"
       "4. Draw the cumulative difference of the losses.\n"
       "5. Interpretation: which model would you give to a risk manager, and what would you still improve?\n\n"
       "**Report:** four numbers, two counts, the chart and two sentences."),
    code("# Solution\nb5 = b5_oos('sp500', save=False)\nb5"),
    md("**Interpretation of the result.** GARCH-t has the smaller mean QLIKE and the DM test rejects equal accuracy; both VaR "
       "models are exceeded more often than 1%, GARCH-t less so: add the leverage effect (B3)."),
    md("## B6 [Proposed]: GARCH against EWMA for the BET and EUR/RON\n\n"
       "**Question.** Does the result of B5 hold for the BET and for EUR/RON? Model: B5.\n\n"
       "1. Compute the out-of-sample GARCH(1,1)-t and EWMA forecasts as in B5.\n"
       "2. Compute the mean QLIKE of both models and the DM test.\n"
       "3. Find the day with the largest loss difference, and recompute the mean QLIKE without it.\n"
       "4. Count the VaR 1% exceedances of both models.\n"
       "5. Interpretation: why can the DM test be insignificant when one mean QLIKE is half the other?\n\n"
       "**Report:** one table and two sentences."),
    code("# Solution\nb6_oos()"),
    # ---------------- Part C
    md("# Part C: open questions and AI critique"),
    md("## C1 [Proposed]: is the leu calmer than it used to be?\n\n"
       "**Question.** Has the volatility of EUR/RON changed between 2005–2012, 2013–2019 and 2020–2026, and can one GARCH "
       "model describe all three periods? Model: B1.\n\n"
       "1. For each period compute the annualised sample volatility and the share of days with $|r_t| > 0.5\\%$.\n"
       "2. Estimate GARCH(1,1)-t on each period and report $\\alpha + \\beta$, $\\alpha$ and $\\nu$.\n"
       "3. Find the largest daily move of each period and its date.\n"
       "4. Interpretation: is the change in volatility a property of the market, or of the exchange-rate policy?\n\n"
       "**Report:** a table and a plan for a project."),
    code("# Reference analysis\npd.DataFrame(c1_eurron_periods())"),
    md("## C2 [Proposed]: audit an AI answer\n\n"
       "**Context.** An AI assistant interpreted a GARCH(1,1)-t fit of the BET (daily, since 2000). It answered:\n\n"
       "- (a) the half-life of a volatility shock is ln 0.5 / ln beta;\n"
       "- (b) the long-run variance is omega / (1 − alpha);\n"
       "- (c) an AR(1) by OLS gives a t statistic of about 9, so BET returns are strongly predictable;\n"
       "- (d) the Ljung–Box test of the squared standardised residuals does not reject, so the GARCH-t model is correct;\n"
       "- (e) with Student-t innovations the 1% quantile of z is closer to 0 than −2.326, so the VaR 1% is smaller;\n"
       "- (f) alpha + beta < 1, so the model is covariance stationary and the volatility reverts to its long-run level.\n\n"
       "1. For each statement, say whether it is correct; if not, give the correct statement and the correct number.\n\n"
       "**Report:** six verdicts with one line of justification each."),
    code("# Solution\nc2_check()"),
]

if __name__ == '__main__':
    build(LECTURE, 5, 'lecture')
    build(SEMINAR, 5, 'seminar')
