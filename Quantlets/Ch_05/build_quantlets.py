"""
build_quantlets.py -- Quantlet folders of Chapter 5 (TSA): conditional volatility, ARCH and GARCH models
=======================================================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_05/generate_all_charts.py && python3 Quantlets/Ch_05/seminar5.py
      python3 Quantlets/Ch_05/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 5
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import generate_all_charts as g                 # noqa: E402
from tsa_quantlets import build_all             # noqa: E402

SUBMITTED = 'Monday, 5 October 2026'
DATA = ('Daily market data from EODHD (S&P 500 and BET since 2000, Bitcoin since 2014), data/market of the TSA '
        'repository; the EUR/RON reference rate of the BNR (since July 2005)')
NODATA = 'Simulated data only (no market data)'
INSTALL = ("# The arch package (ARCH/GARCH estimation) is not preinstalled in Google Colab: pip install arch\n"
           "import importlib.util, subprocess, sys\n"
           "if importlib.util.find_spec('arch') is None:\n"
           "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'arch'])")
CONSTS = ['import statsmodels.api as sm', 'from statsmodels.stats.diagnostic import acorr_ljungbox',
          'from arch import arch_model',
          f'SEED = {g.SEED!r}', f'ASSETS = {g.ASSETS!r}', f'NAME = {g.NAME!r}', f'START = {g.START!r}',
          f'COLORS = {g.COLORS!r}', f'SCALE = {g.SCALE!r}', f'EPISODES = {g.EPISODES!r}', f'OOS_START = {g.OOS_START!r}',
          f'REFIT = {g.REFIT!r}', f'LAMBDA = {g.LAMBDA!r}', f'ALPHA_VAR = {g.ALPHA_VAR!r}', f'LB_LAGS = {g.LB_LAGS!r}',
          f'ARCH_LAGS = {g.ARCH_LAGS!r}', f'IG = {g.IG!r}', f'TERM_DATES = {g.TERM_DATES!r}']
CORE = [g.returns, g.save, g.acf_vals, g.ljung_box, g.arch_lm, g.fit, g.par, g.sig, g.persistence, g.half_life,
        g.summary, g.ols_ar1, g.ar_order]

QUANTLETS = [
    dict(name='TSA_ch5_stylised_facts',
         desc='Stylised facts of daily returns: S&P 500, BET, EUR/RON (BNR reference rate) and Bitcoin; moments, Ljung-Box '
              'tests on returns and squared returns, the ARCH-LM test (Engle 1982); the ACF of returns and of squared returns '
              '(volatility clustering).',
         keywords='stylised facts, volatility clustering, heavy tails, kurtosis, ACF, squared returns, Ljung-Box, McLeod-Li, '
                  'ARCH-LM, S&P 500, BET, EUR/RON, Bitcoin',
         consts=CONSTS, funcs=CORE + [g.stylised_table, g.fig_returns, g.fig_acf_squares],
         run="S = stylised_table()\nprint(pd.DataFrame({k: {'n': v['n'], 'sd': v['sd'], 'kurtosis': v['kurt'], 'Q(10) r': v['lb_r'][0], "
             "'Q(10) r^2': v['lb_r2'][0], 'ARCH-LM(5)': v['arch']['lm']} for k, v in S.items()}).T)\nprint(fig_returns())\nprint(fig_acf_squares())",
         charts=['tsa_ch5_returns', 'tsa_ch5_acf_squares'], extra=['ch5_stylised_facts.csv']),
    dict(name='TSA_ch5_mean_model',
         desc='The mean model of Chapter 2 and its residuals: an AR(1) for the BET returns (order chosen by BIC), the ARCH-LM '
              'test step by step on its residuals, and the ACF of the residuals and squared residuals before and after an '
              'AR(1)-GARCH(1,1)-t model.',
         keywords='AR model, residual diagnostics, ARCH-LM, Lagrange multiplier, squared residuals, AR-GARCH, BET',
         consts=CONSTS, funcs=CORE + [g.fig_arma_resid], run='print(fig_arma_resid())', charts=['tsa_ch5_arma_resid']),
    dict(name='TSA_ch5_simulation_likelihood',
         desc='Simulated i.i.d. Normal, ARCH(1) and GARCH(1,1) paths with the same unconditional variance; the conditional '
              'log-likelihood of a simulated ARCH(1) as a function of alpha for two sample sizes; contours of the GARCH(1,1) '
              'log-likelihood over (alpha, beta) with variance targeting.',
         keywords='ARCH, GARCH, simulation, kurtosis, maximum likelihood, log-likelihood, variance targeting',
         consts=CONSTS, funcs=CORE + [g.simulate_garch, g.fig_simulated, g.arch1_loglik, g.fig_lik_arch1, g.garch_loglik_vt,
                                      g.fig_lik_garch],
         run='print(fig_simulated())\nprint(fig_lik_arch1())\nprint(fig_lik_garch())',
         charts=['tsa_ch5_simulated', 'tsa_ch5_lik_arch1', 'tsa_ch5_lik_garch'], data=NODATA),
    dict(name='TSA_ch5_garch_estimation',
         desc='Estimation of ARCH(q) and GARCH(1,1) models for the S&P 500: AIC and BIC of ARCH(1), ARCH(5), ARCH(10) and '
              'GARCH(1,1); a Gaussian GARCH(1,1) estimated step by step with scipy and with the arch package, classic and '
              'robust (Bollerslev-Wooldridge) standard errors; Normal, Student-t and skewed-t innovations; QQ plots of the '
              'standardised residuals.',
         keywords='GARCH, ARCH, maximum likelihood, quasi-maximum likelihood, robust standard errors, Student-t, skewed t, '
                  'QQ plot, arch package, S&P 500',
         consts=CONSTS, funcs=CORE + [g.garch11_negloglik, g.fit_step_by_step, g.arch_q_table, g.estimation_sp500, g.fig_qq],
         run="print(arch_q_table())\nE = estimation_sp500()\nprint(E['step'])\nprint({d: E[d]['params'] for d in ['normal', 't', 'skewt']})\nprint(fig_qq())",
         charts=['tsa_ch5_qq']),
    dict(name='TSA_ch5_volatility_markets',
         desc='GARCH(1,1)-t on four daily series (S&P 500, BET, EUR/RON, Bitcoin): parameters, persistence, half-life, '
              'long-run and sample volatility; annualised conditional volatility with the 2008 and 2020 episodes; the decay '
              'of a variance shock; GARCH against EWMA (RiskMetrics) and a 63-day window around the COVID-19 crash.',
         keywords='GARCH, IGARCH, EWMA, RiskMetrics, persistence, half-life, conditional volatility, S&P 500, BET, EUR/RON, Bitcoin',
         consts=CONSTS, funcs=CORE + [g.markets_table, g.fig_vol, g.fig_persistence, g.ewma_variance, g.fig_ewma_garch],
         run="T = markets_table()\nprint(pd.DataFrame({k: {'alpha': v['params']['alpha[1]'], 'beta': v['params']['beta[1]'], "
             "'nu': v['params']['nu'], 'persistence': v['pers'], 'half-life': v['hl']} for k, v in T.items()}).T)\n"
             "fig_persistence(T)\nprint(fig_vol(('sp500', 'bet'), 'tsa_ch5_vol_sp500_bet'))\nprint(fig_vol(('eurron', 'btc'), 'tsa_ch5_vol_eurron_btc'))\n"
             "print(fig_ewma_garch())",
         charts=['tsa_ch5_persistence', 'tsa_ch5_vol_sp500_bet', 'tsa_ch5_vol_eurron_btc', 'tsa_ch5_ewma_garch'],
         extra=['ch5_garch_table.csv']),
    dict(name='TSA_ch5_arma_garch',
         desc='ARMA-GARCH: the AR(1) mean estimated by OLS with a constant variance against AR(1)-GARCH(1,1)-t on four series; '
              'Ljung-Box tests of the standardised residuals and BIC; one-day-ahead 95% intervals for the BET with a constant '
              'and with a GARCH variance.',
         keywords='ARMA-GARCH, AR-GARCH, joint estimation, forecast intervals, conditional variance, BET, S&P 500, EUR/RON, Bitcoin',
         consts=CONSTS, funcs=CORE + [g.arma_garch_table, g.fig_bands],
         run='print(arma_garch_table())\nprint(fig_bands())', charts=['tsa_ch5_bands']),
    dict(name='TSA_ch5_asymmetry',
         desc='Asymmetric volatility: GJR-GARCH(1,1)-t and EGARCH(1,1)-t on four series; news impact curves with a '
              'Nadaraya-Watson kernel estimate (S&P 500 and Bitcoin); likelihood-ratio and sign-bias tests.',
         keywords='leverage effect, GJR-GARCH, EGARCH, news impact curve, sign-bias test, likelihood ratio, kernel regression',
         consts=CONSTS, funcs=CORE + [g.news_impact, g.kernel_nic, g.fig_nic, g.sign_bias_test, g.asym_table],
         run='print(fig_nic())\nprint(asym_table())', charts=['tsa_ch5_nic']),
    dict(name='TSA_ch5_diagnostics',
         desc='Diagnostics of GARCH models: Ljung-Box tests of the standardised residuals and of their squares, ARCH-LM, the '
              'ACF of squared returns against squared standardised residuals (S&P 500); the EUR/RON day of 6 May 2025 that '
              'hides the remaining ARCH effects; nine models of the S&P 500 compared by AIC and BIC.',
         keywords='diagnostics, standardised residuals, Ljung-Box, ARCH-LM, model selection, AIC, BIC, EUR/RON, outlier',
         consts=CONSTS, funcs=CORE + [g.diagnostics, g.diag_markets, g.fig_acf_diag, g.managed_rate_case, g.model_selection],
         run='print(diagnostics())\nprint(diag_markets())\nprint(fig_acf_diag())\nprint(managed_rate_case())\nprint(model_selection())',
         charts=['tsa_ch5_acf_diag'], extra=['ch5_model_selection.csv']),
    dict(name='TSA_ch5_forecasts',
         desc='Volatility forecasts: the GARCH(1,1) term structure of volatility on three days (S&P 500); out-of-sample '
              'one-day forecasts of GARCH-t, GJR-t and EWMA since 2015 for four series, compared by QLIKE and the '
              'Diebold-Mariano test; the one-day VaR 1% of GARCH-t and EWMA-Normal and its exceedances.',
         keywords='volatility forecast, term structure, out of sample, QLIKE, Diebold-Mariano, EWMA, VaR 1%, exceedances',
         consts=CONSTS, funcs=CORE + [g.garch_path_forecast, g.fig_term_structure, g.ewma_variance, g.oos_forecasts, g.qlike,
                                      g.dm_test, g.forecast_eval, g.fig_forecast_eval, g.fig_var],
         run='print(fig_term_structure())\nfe, frames = forecast_eval()\nprint(fe)\nprint(fig_forecast_eval(frames))\nprint(fig_var(frames))',
         charts=['tsa_ch5_term_structure', 'tsa_ch5_forecast_eval', 'tsa_ch5_var'], extra=['ch5_forecast_eval.csv']),
]

try:                                   # seminar code (instructor files, see .gitignore)
    import seminar5 as s
    QUANTLETS.append(dict(
        name='TSA_ch5_seminar',
        desc='Seminar 5 of Time Series Analysis: GARCH(1,1) algebra, forecasts and VaR 1%, an ARCH-LM test by hand, '
             'GJR and EGARCH news impact; ARCH effects and AR(1)-GARCH(1,1)-t for the BET, EUR/RON and Bitcoin; the leverage '
             'effect in the S&P 500; GARCH against EWMA out of sample with VaR 1% exceedances.',
        keywords='seminar, ARCH-LM, GARCH, AR-GARCH, GJR, EGARCH, QLIKE, Diebold-Mariano, VaR 1%, BET, EUR/RON, Bitcoin, S&P 500',
        consts=CONSTS, funcs=CORE + [g.news_impact, g.sign_bias_test, g.ewma_variance, g.oos_forecasts, g.qlike, g.dm_test,
                                     s.garch_algebra, s.garch_forecasts, s.archlm_by_hand, s.b1_estimate, s.b3_asymmetry,
                                     s.b5_oos],
        run="print(garch_algebra(0.02, 0.08, 0.90, 1.5, -3.0))\nprint(archlm_by_hand(0.062, 1000, q=5, Q=95.3, m=10))\n"
            "print(b1_estimate('bet'))\nprint(b3_asymmetry('sp500'))\nprint(b5_oos('sp500'))",
        charts=['ch5_sem_b1', 'ch5_sem_b3', 'ch5_sem_b5']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 5, 'Conditional volatility: ARCH and GARCH', HERE, data=DATA, submitted=SUBMITTED, install=INSTALL)
