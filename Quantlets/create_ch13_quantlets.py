"""
Create TSA_ch13 quantlet directory structure.
Each quantlet gets a Metainfo.txt and a standalone Python script.
"""
import os, shutil, textwrap

BASE = '/Users/danielpele/Documents/TSA/Quantlets/TSA_ch13'
CHART_SCRIPT = '/Users/danielpele/Documents/TSA/charts/generate_ch13_lppl_charts.py'

# Read the full chart generation script
with open(CHART_SCRIPT, 'r') as f:
    full_script = f.read()

# Extract shared infrastructure (imports + helpers, lines 1-158)
lines = full_script.split('\n')
# Find where first chart function starts
shared_end = 0
for i, line in enumerate(lines):
    if line.startswith('def chart_bubble_growth'):
        shared_end = i
        break
shared_code = '\n'.join(lines[:shared_end])

# Map: quantlet_name -> (function_name, description, keywords, chart_output_name)
quantlets = {
    'TSA_ch13_bubble_growth': {
        'func': 'chart_bubble_growth',
        'desc': 'Super-exponential bubble growth vs normal exponential growth. Compares price trajectories and growth rates, showing how bubble growth accelerates toward a critical time tc.',
        'keywords': 'bubble, super-exponential growth, critical time, power law, LPPL',
        'chart': 'bubble_growth',
    },
    'TSA_ch13_btc_lppl': {
        'func': 'chart_btc_lppl',
        'desc': 'LPPL model fitted to Bitcoin 2021 bubble using market data from yfinance.Shows log-price with LPPL fit, residuals, and estimated critical time.',
        'keywords': 'Bitcoin, LPPL, bubble, cryptocurrency, critical time, model fitting',
        'chart': 'btc_lppl',
    },
    'TSA_ch13_btc2021_data': {
        'func': 'chart_btc_full_analysis',
        'desc': 'Complete Bitcoin Nov 2021 bubble analysis: log-price with LPPL fit, bootstrap confidence intervals for tc, parameter distributions, and CI evolution.',
        'keywords': 'Bitcoin, LPPL, bootstrap, confidence interval, bubble analysis, 2021',
        'chart': 'btc_full_analysis',
    },
    'TSA_ch13_btc2021_fit': {
        'func': 'chart_btc_lppl',
        'desc': 'LPPL fit to Bitcoin Nov 2021 bubble. Shows fitted vs actual log-price, estimated critical time, and residual analysis.',
        'keywords': 'Bitcoin, LPPL fit, differential evolution, partial linearization, 2021',
        'chart': 'btc_lppl',
    },
    'TSA_ch13_btc2021_bootstrap': {
        'func': 'chart_btc_full_analysis',
        'desc': 'Bootstrap confidence intervals for LPPL parameters fitted to Bitcoin Nov 2021 bubble. Includes tc distribution, parameter histograms, and CI time series.',
        'keywords': 'bootstrap, confidence interval, Bitcoin, LPPL, uncertainty quantification',
        'chart': 'btc_full_analysis',
    },
    'TSA_ch13_oscillations': {
        'func': 'chart_oscillations',
        'desc': 'Log-periodic oscillations in LPPL model. Demonstrates how oscillation frequency increases near the critical time on a log-time scale.',
        'keywords': 'log-periodicity, oscillations, discrete scale invariance, LPPL, critical time',
        'chart': 'oscillations',
    },
    'TSA_ch13_ising': {
        'func': 'chart_ising',
        'desc': 'Ising model simulation showing phase transition from disordered to ordered state. Financial analogy: independent traders vs herding behavior.',
        'keywords': 'Ising model, phase transition, herding, collective behavior, critical temperature',
        'chart': 'ising',
    },
    'TSA_ch13_phase_transition': {
        'func': 'chart_phase_transition',
        'desc': 'Phase transition diagram: magnetization, susceptibility, and correlation length as functions of temperature. Analogy to financial market regimes.',
        'keywords': 'phase transition, critical point, susceptibility, correlation length, order parameter',
        'chart': 'phase_transition',
    },
    'TSA_ch13_lppl_components': {
        'func': 'chart_lppl_components',
        'desc': 'Decomposition of the LPPL equation into its three components: power law trend, log-periodic oscillations, and combined LPPL signal.',
        'keywords': 'LPPL, decomposition, power law, log-periodicity, model components',
        'chart': 'lppl_components',
    },
    'TSA_ch13_historical_crashes': {
        'func': 'chart_historical_crashes',
        'desc': 'Historical financial crashes with market data: Dot-com 2000, Shanghai 2015, Bitcoin 2017, Oil 2008. Normalized price trajectories showing common super-exponential patterns.',
        'keywords': 'financial crashes, historical bubbles, dot-com, Shanghai, Bitcoin, oil, super-exponential',
        'chart': 'historical_crashes',
    },
    'TSA_ch13_hazard_rate': {
        'func': 'chart_hazard_rate',
        'desc': 'Crash hazard rate in the LPPL framework. Shows how the instantaneous probability of a crash increases as the system approaches the critical time.',
        'keywords': 'hazard rate, crash probability, LPPL, critical time, risk',
        'chart': 'hazard_rate',
    },
    'TSA_ch13_confidence_indicator': {
        'func': 'chart_confidence_indicator',
        'desc': 'LPPLS Confidence Indicator construction and interpretation. Shows multi-window LPPL fitting, filter application, and CI time series with traffic-light signals.',
        'keywords': 'confidence indicator, LPPLS, multi-window, filter conditions, bubble monitoring',
        'chart': 'confidence_indicator',
    },
    'TSA_ch13_risk_management': {
        'func': 'chart_risk_management',
        'desc': 'LPPL-based risk management applications: dynamic position sizing, put option timing, VaR adjustment by CI level, and decision framework.',
        'keywords': 'risk management, position sizing, hedging, VaR, LPPL, confidence indicator',
        'chart': 'risk_management',
    },
    'TSA_ch13_scaling_ratio': {
        'func': 'chart_scaling_ratio',
        'desc': 'Scaling ratio lambda = exp(2pi/omega) visualization. Shows the preferred scaling ratio in log-periodic oscillations and its connection to discrete scale invariance.',
        'keywords': 'scaling ratio, lambda, discrete scale invariance, log-periodicity, omega',
        'chart': 'scaling_ratio',
    },
    'TSA_ch13_hierarchical': {
        'func': 'chart_hierarchical',
        'desc': 'Hierarchical diamond lattice model generating discrete scale invariance. Shows how the recursive structure produces log-periodic oscillations in the partition function.',
        'keywords': 'hierarchical lattice, diamond lattice, discrete scale invariance, renormalization group',
        'chart': 'hierarchical',
    },
    'TSA_ch13_cost_landscape': {
        'func': 'chart_cost_landscape',
        'desc': 'LPPL cost function landscape showing multiple local minima. Demonstrates why differential evolution is needed instead of gradient-based optimization.',
        'keywords': 'cost function, optimization, differential evolution, local minima, LPPL estimation',
        'chart': 'cost_landscape',
    },
    'TSA_ch13_filter_conditions': {
        'func': 'chart_filter_conditions',
        'desc': 'Visualization of the 8 LPPL filter conditions: parameter bounds for m, omega, B, C, tc, damping, and R-squared. Shows valid vs invalid parameter regions.',
        'keywords': 'filter conditions, parameter validation, m range, omega range, damping, R-squared',
        'chart': 'filter_conditions',
    },
    'TSA_ch13_lppl_fit': {
        'func': 'chart_btc_lppl',
        'desc': 'LPPL estimation via partial linearization and differential evolution. Demonstrates the two-step procedure: fix nonlinear parameters, solve linear parameters by OLS.',
        'keywords': 'LPPL estimation, partial linearization, differential evolution, OLS, nonlinear optimization',
        'chart': 'btc_lppl',
    },
    'TSA_ch13_bootstrap_ci': {
        'func': 'chart_btc_full_analysis',
        'desc': 'Bootstrap confidence intervals for LPPL critical time estimation. Resamples residuals, re-estimates parameters, and constructs empirical distributions.',
        'keywords': 'bootstrap, confidence interval, critical time, residual resampling, uncertainty',
        'chart': 'btc_full_analysis',
    },
    'TSA_ch13_dotcom_case': {
        'func': 'chart_dotcom_case',
        'desc': 'LPPL case study: NASDAQ Dot-com bubble 2000. LPPL fit with bootstrap CI showing log-periodic oscillations before the March 2000 peak.',
        'keywords': 'dot-com bubble, NASDAQ, 2000, case study, LPPL fit, bootstrap',
        'chart': 'dotcom_case',
    },
    'TSA_ch13_shanghai_case': {
        'func': 'chart_shanghai_case',
        'desc': 'LPPL case study: Shanghai Composite 2015 bubble. LPPL fit with bootstrap CI showing the June 2015 crash.',
        'keywords': 'Shanghai, China, 2015, bubble, case study, LPPL fit, bootstrap',
        'chart': 'shanghai_case',
    },
    'TSA_ch13_bitcoin2017_case': {
        'func': 'chart_bitcoin2017_case',
        'desc': 'LPPL case study: Bitcoin December 2017 bubble. LPPL fit with bootstrap CI showing log-periodic oscillations before the December 2017 peak.',
        'keywords': 'Bitcoin, 2017, cryptocurrency, bubble, case study, LPPL fit, bootstrap',
        'chart': 'bitcoin2017_case',
    },
    'TSA_ch13_oil2008_case': {
        'func': 'chart_oil2008_case',
        'desc': 'LPPL case study: Oil price bubble 2008. LPPL fit with bootstrap CI showing the July 2008 peak at $147 per barrel.',
        'keywords': 'oil, crude oil, 2008, commodity bubble, case study, LPPL fit, bootstrap',
        'chart': 'oil2008_case',
    },
    'TSA_ch13_covid2020_case': {
        'func': 'chart_covid2020_case',
        'desc': 'LPPL negative control: S&P 500 during COVID-19 crash 2020. Shows that LPPL correctly fails to detect a bubble when the crash is exogenous.',
        'keywords': 'COVID-19, negative control, exogenous shock, S&P 500, 2020, LPPL, false positive',
        'chart': 'covid2020_case',
    },
    'TSA_ch13_use_cases': {
        'func': 'chart_use_cases',
        'desc': 'Summary of LPPL use cases: dynamic portfolio management, options hedging strategy, central bank early warning, and backtest results showing relative performance.',
        'keywords': 'use cases, portfolio management, hedging, central bank, backtest, LPPL applications',
        'chart': 'use_cases',
    },
}

# Create base directory
os.makedirs(BASE, exist_ok=True)

# Find function bodies in the script
def extract_function(func_name):
    """Extract a function body from the full script."""
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f'def {func_name}('):
            start = i
            break
    if start is None:
        return f'# Function {func_name} not found in source\npass'

    # Find end of function (next def at column 0 or end of file)
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith('def ') or lines[i].startswith('# ===='):
            end = i
            break

    # Trim trailing blank lines
    while end > start and lines[end - 1].strip() == '':
        end -= 1

    return '\n'.join(lines[start:end])

for qname, info in quantlets.items():
    qdir = os.path.join(BASE, qname)
    os.makedirs(qdir, exist_ok=True)

    # Metainfo.txt
    meta = f"""Name of QuantLet: '{qname}'

Published in: 'Time Series Analysis and Forecasting (TSA)'

Description: '{info["desc"]}'

Keywords: '{info["keywords"]}'

Author: 'Daniel Traian Pele'

Submitted: 'Monday, 19 May 2026'
"""
    with open(os.path.join(qdir, 'Metainfo.txt'), 'w') as f:
        f.write(meta)

    # Python script
    func_body = extract_function(info['func'])

    script = f'''{shared_code}

# =========================================================================
# {qname}
# =========================================================================
{func_body}


if __name__ == '__main__':
    print('Generating {info["chart"]} chart...')
    {info["func"]}()
    print('Done!')
'''
    with open(os.path.join(qdir, f'{qname}.py'), 'w') as f:
        f.write(script)

    print(f'  Created {qname}/')

print(f'\nCreated {len(quantlets)} quantlet directories in {BASE}')
