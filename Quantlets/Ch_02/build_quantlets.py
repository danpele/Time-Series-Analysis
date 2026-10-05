"""
build_quantlets.py -- Quantlet folders of Chapter 2 (TSA): ARMA models
=====================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_02/generate_all_charts.py && python3 Quantlets/Ch_02/seminar2.py
      python3 Quantlets/Ch_02/build_quantlets.py
      python3 notebooks/add_colab_banner.py && python3 notebooks/split_seminar_notebooks.py 2
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
DATA = ('Romanian quarterly real GDP and monthly HICP from Eurostat; daily market data from EODHD (BET since 2000), '
        'data/market of the TSA repository; EUR/RON reference rate of the BNR; statsmodels data sets (sunspots, macrodata)')
NODATA = 'Simulated data only (no market data)'
CONSTS = ['from statsmodels.tsa.stattools import acf, pacf', 'from statsmodels.stats.diagnostic import acorr_ljungbox',
          'from statsmodels.tsa.arima.model import ARIMA', 'from statsmodels.tsa.arima_process import ArmaProcess',
          'from statsmodels.regression.linear_model import yule_walker',
          f'SEED = {g.SEED!r}', f'GDP_NSA = {g.GDP_NSA!r}', f'GDP_SCA = {g.GDP_SCA!r}', f'HICP = {g.HICP!r}',
          f'GDP_START = {g.GDP_START!r}', f'INFL_START = {g.INFL_START!r}', 'BAND_COL = st.IDAred',
          f'MODELS = {g.MODELS!r}', 'HERE = "."']
CORE = [g.ro_gdp_growth, g.ro_inflation, g.sample_acf, g.sample_pacf, g.ljung_box, g.acf_bars, g.simulate_arma,
        g.arma_process, g.theoretical_acf, g.theoretical_pacf, g.psi_weights, g.inverse_roots, g.fit_arma, g.save]

QUANTLETS = [
    dict(name='TSA_ch2_ar_models',
         desc='Autoregressive models: AR(1) paths with phi = 0.9, 0.3 and -0.7 (the same shocks) with their sample and '
              'theoretical ACF; three AR(2) processes in the stationarity triangle (real roots, complex roots, explosive), '
              'their inverse roots in the unit circle and the damped-wave ACF of complex roots, with the AR(2) that Yule '
              '(1927) fitted to the sunspot numbers.',
         keywords='autoregressive model, AR(1), AR(2), stationarity, characteristic roots, unit circle, stationarity '
                  'triangle, complex roots, pseudo-cycle, ACF',
         consts=CONSTS, funcs=CORE + [g.fig_ar1, g.fig_ar2],
         run='print(fig_ar1())\nprint(fig_ar2(sun_phi=(1.336, -0.650)))', charts=['tsa_ch2_ar1', 'tsa_ch2_ar2'], data=NODATA),
    dict(name='TSA_ch2_ma_models',
         desc='Moving-average models: MA(1) paths with theta = 0.8 and -0.8, sample and theoretical ACF (cut-off after '
              'lag 1) and PACF (gradual decay); rho(1) = theta/(1 + theta^2) as a function of theta, its maximum 0.5 and '
              'the identification problem of theta and 1/theta (invertibility).',
         keywords='moving average, MA(1), invertibility, identification, ACF, PACF, AR(infinity)',
         consts=CONSTS, funcs=CORE + [g.fig_ma1, g.fig_ma1_rho], run='print(fig_ma1())\nprint(fig_ma1_rho())',
         charts=['tsa_ch2_ma1', 'tsa_ch2_ma1_rho'], data=NODATA),
    dict(name='TSA_ch2_arma_acf',
         desc='ARMA(p,q) identification: theoretical and sample ACF and PACF of an AR(2) with complex roots, an MA(2) '
              'and an ARMA(1,1) (one simulated path of 500 observations each); the psi weights (impulse responses) of '
              'AR(1), AR(2), MA(2) and ARMA(1,1) from the recursion psi_j = theta_j + sum phi_i psi_{j-i}.',
         keywords='ARMA, identification, ACF, PACF, impulse response, psi weights, causal representation, Wold',
         consts=CONSTS, funcs=CORE + [g.fig_patterns, g.fig_psi], run='print(fig_patterns())\nprint(fig_psi())',
         charts=['tsa_ch2_patterns', 'tsa_ch2_psi'], data=NODATA),
    dict(name='TSA_ch2_estimation',
         desc='Monte Carlo comparison of ARMA estimators with T = 100: Yule-Walker, conditional least squares and exact '
              'Gaussian maximum likelihood for an AR(1) with phi = 0.9 (small-sample bias), conditional least squares '
              'and maximum likelihood for an MA(1) with theta = 0.5.',
         keywords='Yule-Walker, conditional least squares, maximum likelihood, small-sample bias, Monte Carlo, AR(1), MA(1)',
         consts=CONSTS, funcs=CORE + [g.css_ma1, g.ar1_estimates, g.fig_estimators], run='print(fig_estimators())',
         charts=['tsa_ch2_estimators'], data=NODATA),
    dict(name='TSA_ch2_selection',
         desc='Model selection and residual diagnostics by simulation: how often AIC and BIC choose the true order of an '
              'AR(2) for T = 100 and T = 1000; the size of the Ljung-Box test on the residuals of the true AR(1) and '
              'ARMA(1,1) with m and with m - p - q degrees of freedom.',
         keywords='AIC, BIC, information criteria, order selection, consistency, Ljung-Box, degrees of freedom, residuals',
         consts=CONSTS, funcs=CORE + [g.ar_ols_ic, g.fig_ic, g.fig_lb_df], run='print(fig_ic())\nprint(fig_lb_df())',
         charts=['tsa_ch2_ic', 'tsa_ch2_lb_df'], data=NODATA),
    dict(name='TSA_ch2_forecasting',
         desc='ARMA forecasts and their 95% intervals from the psi weights: mean reversion of AR(1) forecasts with '
              'phi = 0.9 and 0.5, and an MA(1) whose forecast equals the mean after one step.',
         keywords='forecasting, forecast error variance, prediction interval, mean reversion, AR(1), MA(1)',
         consts=CONSTS, funcs=CORE + [g.fig_forecast_theory], run='print(fig_forecast_theory())',
         charts=['tsa_ch2_forecast_theory'], data=NODATA),
    dict(name='TSA_ch2_gdp_case',
         desc='The Box-Jenkins method on Romanian annual real GDP growth (Eurostat, quarterly, 100 times the 4-quarter log '
              'difference of chain-linked volumes, since 2000): identification by ACF and PACF, all ARMA(p,q) with '
              'p, q <= 3 by maximum likelihood with AIC, BIC and Ljung-Box on the residuals, residual diagnostics of the '
              'BIC model (MA(3)) and forecasts 8 quarters ahead with 95% intervals.',
         keywords='Box-Jenkins, GDP, Romania, Eurostat, ARMA, MA(3), AIC, BIC, Ljung-Box, Jarque-Bera, forecast',
         consts=CONSTS, funcs=CORE + [g.gdp_models, g.fig_gdp_ident, g.gdp_table, g.fig_gdp_diag, g.fig_gdp_forecast],
         run='print(fig_gdp_ident())\nT = gdp_table(save_csv=False)\nprint(T["aic_best"], T["bic_best"])\n'
             'print(gdp_models(ro_gdp_growth("yoy")).round(3))\nprint(fig_gdp_diag(*T["bic_best"]))\n'
             'print(fig_gdp_forecast(models=(tuple(T["bic_best"]), (1, 0))))',
         charts=['tsa_ch2_gdp_ident', 'tsa_ch2_gdp_diag', 'tsa_ch2_gdp_forecast'], extra=['ch2_gdp_models.csv']),
    dict(name='TSA_ch2_real_series',
         desc='ARMA models on real data: Romanian 12-month HICP inflation since 2005 (AR order by BIC, forecast 24 months '
              'ahead with 80% and 95% intervals, the BNR target band for reference); BET and EUR/RON daily returns (AR(1), '
              'Ljung-Box on residuals and squared residuals, Jarque-Bera, AIC and BIC choices); the yearly sunspot numbers '
              'with the AR(2) of Yule (1927) on 1749-1924 and its implied period.',
         keywords='inflation, HICP, Romania, BET, EUR/RON, daily returns, AR(1), sunspots, Yule, AR(2), forecast',
         consts=CONSTS, funcs=CORE + [g.fig_inflation, g.returns_ar, g.fig_returns, g.fig_sunspots],
         run='print(fig_inflation())\nprint(fig_returns())\nprint(fig_sunspots())',
         charts=['tsa_ch2_inflation', 'tsa_ch2_returns', 'tsa_ch2_sunspots']),
]

try:                                   # seminar code
    import seminar2 as s
    QUANTLETS.append(dict(
        name='TSA_ch2_seminar',
        desc='Seminar 2 of Time Series Analysis: AR(1) moments and forecasts, an MA(1) and its invertible twin, Yule-Walker '
             'estimates and information criteria on paper; the Box-Jenkins steps for US real GDP growth (statsmodels '
             'macrodata); an AR(1) for BET daily returns with residual diagnostics and a one-day forecast.',
        keywords='seminar, AR(1), MA(1), invertibility, Yule-Walker, AIC, BIC, Box-Jenkins, GDP, BET, Ljung-Box',
        consts=CONSTS, funcs=CORE + [s.ar1_facts, s.ma1_facts, s.yw_ar2, s.ic_table, s.box_jenkins, s.b1_us_gdp, s.b3_returns],
        run="print(ar1_facts(1.0, 0.6, 0.64, 4.0))\nprint(ma1_facts(2.0, 1.0))\nprint(yw_ar2(0.75, 0.65, 4.0))\n"
            "print(ic_table({'AR(1)': -214.6, 'AR(2)': -210.9}, {'AR(1)': 3, 'AR(2)': 4}, 100))\n"
            "print(b1_us_gdp())\nprint(b3_returns('bet', fname='ch2_sem_b3'))",
        charts=['ch2_sem_b1', 'ch2_sem_b3']))
except ImportError:
    pass

if __name__ == '__main__':
    build_all(QUANTLETS, 2, 'ARMA models', HERE, data=DATA, submitted=SUBMITTED)
