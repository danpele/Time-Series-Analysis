#!/usr/bin/env python3
"""
Generate ALL time series charts for Chapter 1
Uses real financial data from Yahoo Finance
All charts are publication-quality for LaTeX Beamer
- Transparent backgrounds
- Legends outside bottom
- Harvard-quality styling
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality style with transparent background
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.5,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.facecolor': 'none',
    'figure.facecolor': 'none',
    'savefig.facecolor': 'none',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'legend.facecolor': 'none',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.transparent': True,
})

# IDA-inspired color palette
COLORS = {
    'main_blue': '#1A3A6E',
    'accent_blue': '#2A528C',
    'ida_red': '#DC3545',
    'forest': '#2E7D32',
    'amber': '#B5853F',
    'slate': '#5E6A71',
    'purple': '#6A1B9A',
    'teal': '#00796B',
    'medium_gray': '#808080',
    'dark_gray': '#333333',
    'light_gray': '#E0E0E0',
}

def download_data():
    """Download financial data from Yahoo Finance"""
    print("Downloading data from Yahoo Finance...")

    def clean_df(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    sp500 = clean_df(yf.download('^GSPC', start='2019-01-01', end='2025-12-31', progress=False))
    aapl = clean_df(yf.download('AAPL', start='2019-01-01', end='2025-12-31', progress=False))
    eurusd = clean_df(yf.download('EURUSD=X', start='2019-01-01', end='2025-12-31', progress=False))
    btc = clean_df(yf.download('BTC-USD', start='2019-01-01', end='2025-12-31', progress=False))
    gold = clean_df(yf.download('GC=F', start='2019-01-01', end='2025-12-31', progress=False))
    tnx = clean_df(yf.download('^TNX', start='2019-01-01', end='2025-12-31', progress=False))

    try:
        airline = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv',
                              parse_dates=['Month'], index_col='Month')
        airline.columns = ['Passengers']
    except:
        dates = pd.date_range('1949-01', periods=144, freq='M')
        trend = np.linspace(100, 500, 144)
        seasonal = 50 * np.sin(np.arange(144) * 2 * np.pi / 12)
        noise = np.random.randn(144) * 20
        airline = pd.DataFrame({'Passengers': trend + seasonal + noise}, index=dates)

    return {
        'sp500': sp500, 'aapl': aapl, 'eurusd': eurusd,
        'btc': btc, 'gold': gold, 'tnx': tnx, 'airline': airline
    }

def setup_axis(ax):
    """Apply consistent styling to axis"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['dark_gray'])
    ax.spines['bottom'].set_color(COLORS['dark_gray'])
    ax.tick_params(colors=COLORS['dark_gray'])
    ax.xaxis.label.set_color(COLORS['dark_gray'])
    ax.yaxis.label.set_color(COLORS['dark_gray'])

# =============================================================================
# SECTION 1: What is a Time Series
# =============================================================================

def plot_timeseries_definition(data):
    """Plot showing what a time series looks like"""
    sp500 = data['sp500']['Close'].iloc[-252:]

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(sp500.index, sp500.values, color=COLORS['main_blue'], linewidth=0.8, alpha=0.7)
    ax.scatter(sp500.index[::5], sp500.values[::5], color=COLORS['ida_red'], s=12, zorder=5)
    ax.set_xlabel('Time ($t$)')
    ax.set_ylabel('$X_t$')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    setup_axis(ax)

    plt.tight_layout()
    plt.savefig('charts/timeseries_definition.pdf', transparent=True)
    plt.close()
    print("Saved: timeseries_definition.pdf")

def plot_multiple_assets(data):
    """Plot multiple financial time series"""
    fig, axes = plt.subplots(2, 3, figsize=(11, 5))

    assets = [
        ('sp500', 'S&P 500', COLORS['main_blue']),
        ('aapl', 'Apple', COLORS['ida_red']),
        ('btc', 'Bitcoin', COLORS['amber']),
        ('gold', 'Gold', COLORS['forest']),
        ('eurusd', 'EUR/USD', COLORS['accent_blue']),
        ('tnx', '10Y Yield', COLORS['slate']),
    ]

    for idx, (key, title, color) in enumerate(assets):
        ax = axes[idx // 3, idx % 3]
        df = data[key]
        if len(df) > 0:
            normalized = (df['Close'] / df['Close'].iloc[0]) * 100
            ax.plot(df.index, normalized, color=color, linewidth=0.6)
            ax.set_title(title, fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%y'))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.axhline(y=100, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4, alpha=0.5)
            setup_axis(ax)

    plt.tight_layout()
    plt.savefig('charts/multiple_assets.pdf', transparent=True)
    plt.close()
    print("Saved: multiple_assets.pdf")

def plot_data_types_comparison():
    """Visual comparison of data types"""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    np.random.seed(42)

    # Cross-sectional
    ax1 = axes[0]
    x = np.arange(1, 11)
    y = np.random.randn(10) * 10 + 50
    ax1.bar(x, y, color=COLORS['main_blue'], alpha=0.8)
    ax1.set_xlabel('Unit $i$')
    ax1.set_ylabel('$X_i$')
    ax1.set_title('Cross-Sectional', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    ax1.set_xticks(x)
    setup_axis(ax1)

    # Time series
    ax2 = axes[1]
    t = np.arange(1, 51)
    ts = np.cumsum(np.random.randn(50)) + 50
    ax2.plot(t, ts, color=COLORS['ida_red'], linewidth=1)
    ax2.scatter(t[::3], ts[::3], color=COLORS['ida_red'], s=8)
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('$X_t$')
    ax2.set_title('Time Series', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax2)

    # Panel
    ax3 = axes[2]
    for i, color in enumerate([COLORS['main_blue'], COLORS['ida_red'], COLORS['forest'], COLORS['amber']]):
        ts_panel = np.cumsum(np.random.randn(30)) + 50 + i*10
        ax3.plot(np.arange(1, 31), ts_panel, color=color, linewidth=0.8, label=f'Unit {i+1}')
    ax3.set_xlabel('Time $t$')
    ax3.set_ylabel('$X_{it}$')
    ax3.set_title('Panel', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, fontsize=7, frameon=False)
    setup_axis(ax3)

    plt.tight_layout()
    plt.savefig('charts/data_types_comparison.pdf', transparent=True)
    plt.close()
    print("Saved: data_types_comparison.pdf")

# =============================================================================
# SECTION 2: Time Series Decomposition
# =============================================================================

def plot_ts_components_synthetic():
    """Plot synthetic decomposition"""
    np.random.seed(42)
    t = np.arange(150)
    trend = 0.1 * t + 10
    seasonal = 4 * np.sin(2 * np.pi * t / 12)
    residual = np.random.randn(150) * 1.2
    observed = trend + seasonal + residual

    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

    components = [
        (observed, '$X_t$', COLORS['main_blue']),
        (trend, '$T_t$', COLORS['ida_red']),
        (seasonal, '$S_t$', COLORS['forest']),
        (residual, '$\\varepsilon_t$', COLORS['slate']),
    ]

    for ax, (series, ylabel, color) in zip(axes, components):
        ax.plot(t, series, color=color, linewidth=0.7)
        ax.set_ylabel(ylabel, fontsize=9)
        setup_axis(ax)
        if 'varepsilon' in ylabel:
            ax.axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)

    axes[-1].set_xlabel('Time $t$')
    plt.tight_layout()
    plt.savefig('charts/ts_components_synthetic.pdf', transparent=True)
    plt.close()
    print("Saved: ts_components_synthetic.pdf")

def plot_airline_decomposition(data):
    """Multiplicative decomposition of airline data"""
    airline = data['airline']
    decomp = seasonal_decompose(airline['Passengers'], model='multiplicative', period=12)

    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

    components = [
        (airline['Passengers'], '$X_t$', COLORS['main_blue']),
        (decomp.trend, '$T_t$', COLORS['ida_red']),
        (decomp.seasonal, '$S_t$', COLORS['forest']),
        (decomp.resid, '$\\varepsilon_t$', COLORS['slate']),
    ]

    for ax, (series, ylabel, color) in zip(axes, components):
        ax.plot(series.index, series.values, color=color, linewidth=0.7)
        ax.set_ylabel(ylabel, fontsize=9)
        setup_axis(ax)
        if 'varepsilon' in ylabel:
            ax.axhline(y=1, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig('charts/airline_decomposition.pdf', transparent=True)
    plt.close()
    print("Saved: airline_decomposition.pdf")

def plot_additive_vs_multiplicative(data):
    """Compare additive vs multiplicative"""
    airline = data['airline']
    add_decomp = seasonal_decompose(airline['Passengers'], model='additive', period=12)
    mult_decomp = seasonal_decompose(airline['Passengers'], model='multiplicative', period=12)

    fig, axes = plt.subplots(2, 2, figsize=(11, 5))

    # Seasonal components
    axes[0, 0].plot(add_decomp.seasonal.index, add_decomp.seasonal.values, color=COLORS['main_blue'], linewidth=0.7)
    axes[0, 0].set_title('Additive: $S_t$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[0, 0].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)
    setup_axis(axes[0, 0])

    axes[0, 1].plot(mult_decomp.seasonal.index, mult_decomp.seasonal.values, color=COLORS['ida_red'], linewidth=0.7)
    axes[0, 1].set_title('Multiplicative: $S_t$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[0, 1].axhline(y=1, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)
    setup_axis(axes[0, 1])

    # Residuals
    axes[1, 0].plot(add_decomp.resid.index, add_decomp.resid.values, color=COLORS['main_blue'], linewidth=0.5)
    axes[1, 0].set_title('Additive: $\\varepsilon_t$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 0].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)
    axes[1, 0].set_xlabel('Date')
    setup_axis(axes[1, 0])

    axes[1, 1].plot(mult_decomp.resid.index, mult_decomp.resid.values, color=COLORS['ida_red'], linewidth=0.5)
    axes[1, 1].set_title('Multiplicative: $\\varepsilon_t$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 1].axhline(y=1, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)
    axes[1, 1].set_xlabel('Date')
    setup_axis(axes[1, 1])

    plt.tight_layout()
    plt.savefig('charts/additive_vs_multiplicative.pdf', transparent=True)
    plt.close()
    print("Saved: additive_vs_multiplicative.pdf")

def plot_moving_average_trend():
    """Moving average smoothing"""
    np.random.seed(42)
    t = np.arange(100)
    trend_true = 0.1 * t + 10
    seasonal = 3 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(100) * 1
    observed = trend_true + seasonal + noise

    ma_5 = pd.Series(observed).rolling(window=5, center=True).mean()
    ma_12 = pd.Series(observed).rolling(window=12, center=True).mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, observed, color=COLORS['light_gray'], linewidth=0.6, label='Observed', alpha=0.8)
    ax.plot(t, ma_5, color=COLORS['forest'], linewidth=1, label='MA(5)')
    ax.plot(t, ma_12, color=COLORS['ida_red'], linewidth=1.2, label='MA(12)')
    ax.plot(t, trend_true, color=COLORS['main_blue'], linewidth=1.5, linestyle='--', label='True Trend')

    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$X_t$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    setup_axis(ax)

    plt.tight_layout()
    plt.savefig('charts/moving_average_trend.pdf', transparent=True)
    plt.close()
    print("Saved: moving_average_trend.pdf")

def plot_stl_decomposition(data):
    """STL decomposition"""
    airline = data['airline']
    stl = STL(airline['Passengers'], period=12, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

    components = [
        (airline['Passengers'], '$X_t$', COLORS['main_blue']),
        (result.trend, '$T_t$', COLORS['ida_red']),
        (result.seasonal, '$S_t$', COLORS['forest']),
        (result.resid, '$R_t$', COLORS['slate']),
    ]

    for ax, (series, ylabel, color) in zip(axes, components):
        ax.plot(series.index, series.values, color=color, linewidth=0.7)
        ax.set_ylabel(ylabel, fontsize=9)
        setup_axis(ax)
        if 'R_t' in ylabel:
            ax.axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)

    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    plt.savefig('charts/stl_decomposition.pdf', transparent=True)
    plt.close()
    print("Saved: stl_decomposition.pdf")

def plot_seasonal_pattern(data):
    """Seasonal indices"""
    airline = data['airline']
    decomp = seasonal_decompose(airline['Passengers'], model='multiplicative', period=12)
    seasonal_pattern = decomp.seasonal.iloc[:12]

    months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = [COLORS['forest'] if v > 1 else COLORS['ida_red'] for v in seasonal_pattern.values]
    bars = ax.bar(months, seasonal_pattern.values, color=colors, alpha=0.8, width=0.7)
    ax.axhline(y=1, color=COLORS['dark_gray'], linestyle='--', linewidth=0.8)
    ax.set_xlabel('Month')
    ax.set_ylabel('Seasonal Index $S_t$')
    setup_axis(ax)

    plt.tight_layout()
    plt.savefig('charts/seasonal_pattern.pdf', transparent=True)
    plt.close()
    print("Saved: seasonal_pattern.pdf")

# =============================================================================
# SECTION 3: Exponential Smoothing Methods
# =============================================================================

def plot_simple_exp_smoothing(data):
    """Simple Exponential Smoothing (SES)"""
    airline = data['airline']['Passengers']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Different alpha values
    ax1 = axes[0]
    ax1.plot(airline.index, airline.values, color=COLORS['light_gray'], linewidth=0.6, label='Observed', alpha=0.8)

    for alpha, color, ls in [(0.1, COLORS['main_blue'], '-'),
                              (0.5, COLORS['forest'], '-'),
                              (0.9, COLORS['ida_red'], '-')]:
        model = SimpleExpSmoothing(airline, initialization_method='estimated').fit(smoothing_level=alpha)
        ax1.plot(airline.index, model.fittedvalues, color=color, linewidth=1, linestyle=ls, label=f'$\\alpha={alpha}$')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Passengers')
    ax1.set_title('Simple Exponential Smoothing', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=8)
    setup_axis(ax1)

    # Weights decay
    ax2 = axes[1]
    t = np.arange(20)
    for alpha, color in [(0.1, COLORS['main_blue']), (0.5, COLORS['forest']), (0.9, COLORS['ida_red'])]:
        weights = alpha * (1 - alpha) ** t
        ax2.plot(t, weights, color=color, linewidth=1.2, marker='o', markersize=3, label=f'$\\alpha={alpha}$')

    ax2.set_xlabel('Lag $j$')
    ax2.set_ylabel('Weight $\\alpha(1-\\alpha)^j$')
    ax2.set_title('Exponential Decay of Weights', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8)
    setup_axis(ax2)

    plt.tight_layout()
    plt.savefig('charts/simple_exp_smoothing.pdf', transparent=True)
    plt.close()
    print("Saved: simple_exp_smoothing.pdf")

def plot_holt_method(data):
    """Holt's Linear Trend Method"""
    airline = data['airline']['Passengers']

    # Fit Holt's method
    model = Holt(airline, initialization_method='estimated').fit()
    forecast = model.forecast(24)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(airline.index, airline.values, color=COLORS['main_blue'], linewidth=0.8, label='Observed')
    ax.plot(airline.index, model.fittedvalues, color=COLORS['ida_red'], linewidth=1, label='Fitted')

    # Forecast
    forecast_idx = pd.date_range(airline.index[-1], periods=25, freq='M')[1:]
    ax.plot(forecast_idx, forecast, color=COLORS['forest'], linewidth=1.5, linestyle='--', label='Forecast')

    ax.axvline(x=airline.index[-1], color=COLORS['medium_gray'], linestyle=':', linewidth=0.8)
    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    setup_axis(ax)

    plt.tight_layout()
    plt.savefig('charts/holt_method.pdf', transparent=True)
    plt.close()
    print("Saved: holt_method.pdf")

def plot_holt_winters(data):
    """Holt-Winters with seasonality"""
    airline = data['airline']['Passengers']

    # Fit additive and multiplicative
    model_add = ExponentialSmoothing(airline, seasonal_periods=12, trend='add', seasonal='add',
                                      initialization_method='estimated').fit()
    model_mul = ExponentialSmoothing(airline, seasonal_periods=12, trend='add', seasonal='mul',
                                      initialization_method='estimated').fit()

    forecast_add = model_add.forecast(24)
    forecast_mul = model_mul.forecast(24)
    forecast_idx = pd.date_range(airline.index[-1], periods=25, freq='M')[1:]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Additive
    ax1 = axes[0]
    ax1.plot(airline.index, airline.values, color=COLORS['light_gray'], linewidth=0.6, alpha=0.8)
    ax1.plot(airline.index, model_add.fittedvalues, color=COLORS['main_blue'], linewidth=0.8)
    ax1.plot(forecast_idx, forecast_add, color=COLORS['ida_red'], linewidth=1.2, linestyle='--')
    ax1.axvline(x=airline.index[-1], color=COLORS['medium_gray'], linestyle=':', linewidth=0.6)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Passengers')
    ax1.set_title('Additive Seasonality', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax1)

    # Multiplicative
    ax2 = axes[1]
    ax2.plot(airline.index, airline.values, color=COLORS['light_gray'], linewidth=0.6, alpha=0.8)
    ax2.plot(airline.index, model_mul.fittedvalues, color=COLORS['main_blue'], linewidth=0.8)
    ax2.plot(forecast_idx, forecast_mul, color=COLORS['ida_red'], linewidth=1.2, linestyle='--')
    ax2.axvline(x=airline.index[-1], color=COLORS['medium_gray'], linestyle=':', linewidth=0.6)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Passengers')
    ax2.set_title('Multiplicative Seasonality', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax2)

    plt.tight_layout()
    plt.savefig('charts/holt_winters.pdf', transparent=True)
    plt.close()
    print("Saved: holt_winters.pdf")

def plot_ets_components(data):
    """ETS model components"""
    airline = data['airline']['Passengers']

    model = ExponentialSmoothing(airline, seasonal_periods=12, trend='add', seasonal='mul',
                                  initialization_method='estimated').fit()

    fig, axes = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True)

    # Level
    axes[0].plot(airline.index, model.level, color=COLORS['main_blue'], linewidth=0.8)
    axes[0].set_ylabel('Level $\\ell_t$', fontsize=9)
    setup_axis(axes[0])

    # Trend
    axes[1].plot(airline.index, model.trend, color=COLORS['ida_red'], linewidth=0.8)
    axes[1].set_ylabel('Trend $b_t$', fontsize=9)
    axes[1].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)
    setup_axis(axes[1])

    # Seasonal
    axes[2].plot(airline.index, model.season, color=COLORS['forest'], linewidth=0.8)
    axes[2].set_ylabel('Seasonal $s_t$', fontsize=9)
    axes[2].axhline(y=1, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)
    axes[2].set_xlabel('Date')
    setup_axis(axes[2])

    plt.tight_layout()
    plt.savefig('charts/ets_components.pdf', transparent=True)
    plt.close()
    print("Saved: ets_components.pdf")

# =============================================================================
# SECTION 3b: Forecast Evaluation
# =============================================================================

def plot_forecast_accuracy_metrics(data):
    """Compare forecast accuracy metrics"""
    airline = data['airline']['Passengers']

    # Split data
    train = airline[:-24]
    test = airline[-24:]

    # Fit models
    ses = SimpleExpSmoothing(train, initialization_method='estimated').fit()
    holt = Holt(train, initialization_method='estimated').fit()
    hw = ExponentialSmoothing(train, seasonal_periods=12, trend='add', seasonal='mul',
                              initialization_method='estimated').fit()

    # Forecasts
    fc_ses = ses.forecast(24)
    fc_holt = holt.forecast(24)
    fc_hw = hw.forecast(24)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Forecast comparison
    ax1 = axes[0]
    ax1.plot(test.index, test.values, color=COLORS['dark_gray'], linewidth=1.5, label='Actual')
    ax1.plot(test.index, fc_ses.values, color=COLORS['main_blue'], linewidth=1, linestyle='--', label='SES')
    ax1.plot(test.index, fc_holt.values, color=COLORS['forest'], linewidth=1, linestyle='--', label='Holt')
    ax1.plot(test.index, fc_hw.values, color=COLORS['ida_red'], linewidth=1, linestyle='--', label='Holt-Winters')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Passengers')
    ax1.set_title('Forecast Comparison', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=8)
    setup_axis(ax1)

    # Error metrics bar chart
    ax2 = axes[1]

    def calc_metrics(actual, forecast):
        errors = actual - forecast
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        mape = np.mean(np.abs(errors/actual)) * 100
        return mae, rmse, mape

    metrics_ses = calc_metrics(test.values, fc_ses.values)
    metrics_holt = calc_metrics(test.values, fc_holt.values)
    metrics_hw = calc_metrics(test.values, fc_hw.values)

    x = np.arange(3)
    width = 0.25

    ax2.bar(x - width, [metrics_ses[0], metrics_holt[0], metrics_hw[0]], width,
            color=COLORS['main_blue'], label='MAE', alpha=0.8)
    ax2.bar(x, [metrics_ses[1], metrics_holt[1], metrics_hw[1]], width,
            color=COLORS['forest'], label='RMSE', alpha=0.8)
    ax2.bar(x + width, [metrics_ses[2], metrics_holt[2], metrics_hw[2]], width,
            color=COLORS['ida_red'], label='MAPE (%)', alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(['SES', 'Holt', 'H-W'])
    ax2.set_ylabel('Error Value')
    ax2.set_title('Forecast Accuracy Metrics', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8)
    setup_axis(ax2)

    plt.tight_layout()
    plt.savefig('charts/forecast_accuracy_metrics.pdf', transparent=True)
    plt.close()
    print("Saved: forecast_accuracy_metrics.pdf")

def plot_residual_diagnostics(data):
    """Residual diagnostics for forecast evaluation"""
    airline = data['airline']['Passengers']

    model = ExponentialSmoothing(airline, seasonal_periods=12, trend='add', seasonal='mul',
                                  initialization_method='estimated').fit()
    residuals = model.resid

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Residuals over time
    axes[0, 0].plot(residuals.index, residuals.values, color=COLORS['main_blue'], linewidth=0.5)
    axes[0, 0].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.5)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].set_title('Residuals Over Time', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(axes[0, 0])

    # Histogram
    axes[0, 1].hist(residuals.dropna(), bins=25, color=COLORS['main_blue'], alpha=0.7, edgecolor='white', density=True)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    std = residuals.std()
    axes[0, 1].plot(x, 1/(std*np.sqrt(2*np.pi)) * np.exp(-0.5*(x/std)**2),
                    color=COLORS['ida_red'], linewidth=1.5)
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Residual Distribution', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(axes[0, 1])

    # ACF of residuals
    acf_resid = acf(residuals.dropna(), nlags=20)
    axes[1, 0].bar(range(len(acf_resid)), acf_resid, color=COLORS['forest'], width=0.6)
    axes[1, 0].axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
    n = len(residuals)
    axes[1, 0].axhline(y=1.96/np.sqrt(n), color=COLORS['ida_red'], linestyle='--', linewidth=0.5)
    axes[1, 0].axhline(y=-1.96/np.sqrt(n), color=COLORS['ida_red'], linestyle='--', linewidth=0.5)
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')
    axes[1, 0].set_title('ACF of Residuals', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(axes[1, 0])

    # Fitted vs Actual
    axes[1, 1].scatter(model.fittedvalues, airline, color=COLORS['main_blue'], s=8, alpha=0.6)
    min_val = min(model.fittedvalues.min(), airline.min())
    max_val = max(model.fittedvalues.max(), airline.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], color=COLORS['ida_red'],
                    linewidth=1.5, linestyle='--')
    axes[1, 1].set_xlabel('Fitted Values')
    axes[1, 1].set_ylabel('Actual Values')
    axes[1, 1].set_title('Fitted vs Actual', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(axes[1, 1])

    plt.tight_layout()
    plt.savefig('charts/residual_diagnostics.pdf', transparent=True)
    plt.close()
    print("Saved: residual_diagnostics.pdf")

def plot_cross_validation_forecast(data):
    """Time series cross-validation illustration"""
    np.random.seed(42)
    T = 100

    fig, ax = plt.subplots(figsize=(10, 4))

    # Create sample data
    t = np.arange(T)
    y = 0.1*t + 5*np.sin(2*np.pi*t/12) + np.random.randn(T)*2

    # Show CV folds
    fold_size = 20
    test_size = 5
    n_folds = 4

    colors_train = plt.cm.Blues(np.linspace(0.3, 0.7, n_folds))
    colors_test = plt.cm.Reds(np.linspace(0.4, 0.8, n_folds))

    for i in range(n_folds):
        train_end = 50 + i*10
        test_end = train_end + test_size

        # Training set
        ax.plot(t[:train_end], y[:train_end] + i*25, color=colors_train[i], linewidth=1.2,
                label=f'Train {i+1}' if i == 0 else '')
        # Test set
        ax.plot(t[train_end:test_end], y[train_end:test_end] + i*25, color=colors_test[i],
                linewidth=2, label=f'Test {i+1}' if i == 0 else '')
        # Vertical line
        ax.axvline(x=train_end, ymin=(i*25+5)/(n_folds*25+10), ymax=(i*25+20)/(n_folds*25+10),
                   color=COLORS['medium_gray'], linestyle=':', linewidth=0.8)

    ax.set_xlabel('Time')
    ax.set_ylabel('Fold')
    ax.set_yticks([12, 37, 62, 87])
    ax.set_yticklabels(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4'])
    ax.legend(['Training', 'Test'], loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=2, frameon=False, fontsize=9)
    setup_axis(ax)

    plt.tight_layout()
    plt.savefig('charts/cross_validation_forecast.pdf', transparent=True)
    plt.close()
    print("Saved: cross_validation_forecast.pdf")

def plot_train_test_validation():
    """Train/Test/Validation split visualization"""
    np.random.seed(42)
    T = 150
    t = np.arange(T)
    y = 0.08*t + 8*np.sin(2*np.pi*t/12) + np.random.randn(T)*3 + 50

    # Split points
    train_end = 100
    val_end = 125

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: The split
    ax1 = axes[0]
    ax1.plot(t[:train_end], y[:train_end], color=COLORS['main_blue'], linewidth=1, label='Training (67%)')
    ax1.plot(t[train_end:val_end], y[train_end:val_end], color=COLORS['forest'], linewidth=1.5, label='Validation (17%)')
    ax1.plot(t[val_end:], y[val_end:], color=COLORS['ida_red'], linewidth=1.5, label='Test (16%)')

    ax1.axvline(x=train_end, color=COLORS['dark_gray'], linestyle='--', linewidth=0.8)
    ax1.axvline(x=val_end, color=COLORS['dark_gray'], linestyle='--', linewidth=0.8)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('$X_t$')
    ax1.set_title('Train / Validation / Test Split', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=8)
    setup_axis(ax1)

    # Right: Purpose of each set
    ax2 = axes[1]
    purposes = ['Train', 'Validation', 'Test']
    colors = [COLORS['main_blue'], COLORS['forest'], COLORS['ida_red']]
    sizes = [67, 17, 16]

    bars = ax2.barh(purposes, sizes, color=colors, height=0.6)
    ax2.set_xlabel('Percentage of Data')
    ax2.set_title('Data Allocation', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])

    # Add annotations
    for bar, purpose in zip(bars, ['Fit model\nparameters', 'Tune hyper-\nparameters', 'Final\nevaluation']):
        ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                 purpose, va='center', ha='left', fontsize=7)

    ax2.set_xlim(0, 100)
    setup_axis(ax2)

    plt.tight_layout()
    plt.savefig('charts/train_test_validation.pdf', transparent=True)
    plt.close()
    print("Saved: train_test_validation.pdf")

def plot_seasonality_fourier_dummies(data):
    """Compare Fourier terms vs dummy variables for seasonality"""
    airline = data['airline']['Passengers']
    T = len(airline)
    t = np.arange(T)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    # Top left: Dummy variables visualization
    ax1 = axes[0, 0]
    months = np.tile(np.arange(12), T//12 + 1)[:T]
    dummy_matrix = np.zeros((T, 12))
    for i in range(T):
        dummy_matrix[i, months[i]] = 1

    ax1.imshow(dummy_matrix[:36].T, aspect='auto', cmap='Blues', interpolation='nearest')
    ax1.set_xlabel('Time (first 36 obs)')
    ax1.set_ylabel('Month')
    ax1.set_yticks(range(12))
    ax1.set_yticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax1.set_title('Seasonal Dummies $D_{jt}$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])

    # Top right: Fourier terms visualization
    ax2 = axes[0, 1]
    K = 3  # Number of Fourier pairs
    t_plot = np.arange(36)
    for k in range(1, K+1):
        sin_term = np.sin(2*np.pi*k*t_plot/12)
        cos_term = np.cos(2*np.pi*k*t_plot/12)
        ax2.plot(t_plot, sin_term + (k-1)*2.5, color=COLORS['main_blue'], linewidth=0.8,
                 label=f'sin(2π·{k}t/12)' if k==1 else '')
        ax2.plot(t_plot, cos_term + (k-1)*2.5, color=COLORS['ida_red'], linewidth=0.8, linestyle='--',
                 label=f'cos(2π·{k}t/12)' if k==1 else '')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Fourier Terms')
    ax2.set_title('Fourier Terms (K=3)', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=8)
    setup_axis(ax2)

    # Bottom left: Fitted seasonality with dummies
    ax3 = axes[1, 0]
    from sklearn.linear_model import LinearRegression

    # Create dummy features
    X_dummy = np.zeros((T, 12))
    for i in range(T):
        X_dummy[i, months[i]] = 1
    X_dummy = X_dummy[:, 1:]  # Drop first for identifiability

    # Add trend
    X_trend = np.column_stack([np.ones(T), t])
    X_full_dummy = np.column_stack([X_trend, X_dummy])

    model_dummy = LinearRegression().fit(X_full_dummy, airline.values)
    fitted_dummy = model_dummy.predict(X_full_dummy)

    ax3.plot(airline.index, airline.values, color=COLORS['light_gray'], linewidth=0.6, alpha=0.8)
    ax3.plot(airline.index, fitted_dummy, color=COLORS['main_blue'], linewidth=1)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Passengers')
    ax3.set_title('Dummy Model Fit', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax3)

    # Bottom right: Fitted seasonality with Fourier
    ax4 = axes[1, 1]

    # Create Fourier features
    X_fourier = [np.ones(T), t]
    for k in range(1, 4):  # 3 pairs
        X_fourier.append(np.sin(2*np.pi*k*t/12))
        X_fourier.append(np.cos(2*np.pi*k*t/12))
    X_full_fourier = np.column_stack(X_fourier)

    model_fourier = LinearRegression().fit(X_full_fourier, airline.values)
    fitted_fourier = model_fourier.predict(X_full_fourier)

    ax4.plot(airline.index, airline.values, color=COLORS['light_gray'], linewidth=0.6, alpha=0.8)
    ax4.plot(airline.index, fitted_fourier, color=COLORS['ida_red'], linewidth=1)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Passengers')
    ax4.set_title('Fourier Model Fit (K=3)', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax4)

    plt.tight_layout()
    plt.savefig('charts/seasonality_fourier_dummies.pdf', transparent=True)
    plt.close()
    print("Saved: seasonality_fourier_dummies.pdf")

def plot_real_data_forecast_comparison(data):
    """Compare forecasting methods on real airline data"""
    airline = data['airline']['Passengers']

    # 80/20 split
    n = len(airline)
    train_size = int(n * 0.8)
    train = airline.iloc[:train_size]
    test = airline.iloc[train_size:]
    h = len(test)

    # Fit models
    ses = SimpleExpSmoothing(train, initialization_method='estimated').fit()
    holt = Holt(train, initialization_method='estimated').fit()
    hw_add = ExponentialSmoothing(train, seasonal_periods=12, trend='add', seasonal='add',
                                   initialization_method='estimated').fit()
    hw_mul = ExponentialSmoothing(train, seasonal_periods=12, trend='add', seasonal='mul',
                                   initialization_method='estimated').fit()

    # Forecasts
    fc_ses = ses.forecast(h)
    fc_holt = holt.forecast(h)
    fc_hw_add = hw_add.forecast(h)
    fc_hw_mul = hw_mul.forecast(h)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: Forecasts
    ax1 = axes[0]
    ax1.plot(train.index, train.values, color=COLORS['main_blue'], linewidth=0.8, label='Training')
    ax1.plot(test.index, test.values, color=COLORS['dark_gray'], linewidth=2, label='Actual')
    ax1.plot(test.index, fc_ses.values, color=COLORS['slate'], linewidth=1, linestyle=':', label='SES')
    ax1.plot(test.index, fc_holt.values, color=COLORS['amber'], linewidth=1, linestyle='--', label='Holt')
    ax1.plot(test.index, fc_hw_add.values, color=COLORS['forest'], linewidth=1, linestyle='-.', label='HW-Add')
    ax1.plot(test.index, fc_hw_mul.values, color=COLORS['ida_red'], linewidth=1.5, label='HW-Mul')

    ax1.axvline(x=train.index[-1], color=COLORS['medium_gray'], linestyle='--', linewidth=0.6)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Passengers')
    ax1.set_title('Forecast Comparison', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=7)
    setup_axis(ax1)

    # Right: Error metrics table as bar chart
    ax2 = axes[1]

    def calc_rmse(actual, forecast):
        return np.sqrt(np.mean((actual - forecast)**2))

    def calc_mape(actual, forecast):
        return np.mean(np.abs((actual - forecast)/actual)) * 100

    methods = ['SES', 'Holt', 'HW-Add', 'HW-Mul']
    forecasts = [fc_ses, fc_holt, fc_hw_add, fc_hw_mul]
    rmses = [calc_rmse(test.values, fc.values) for fc in forecasts]
    mapes = [calc_mape(test.values, fc.values) for fc in forecasts]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax2.bar(x - width/2, rmses, width, color=COLORS['main_blue'], label='RMSE', alpha=0.8)
    ax2_twin = ax2.twinx()
    bars2 = ax2_twin.bar(x + width/2, mapes, width, color=COLORS['ida_red'], label='MAPE (%)', alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(methods)
    ax2.set_ylabel('RMSE', color=COLORS['main_blue'])
    ax2_twin.set_ylabel('MAPE (%)', color=COLORS['ida_red'])
    ax2.set_title('Forecast Accuracy', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])

    # Combined legend
    ax2.legend([bars1, bars2], ['RMSE', 'MAPE (%)'], loc='upper center',
               bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False, fontsize=8)
    setup_axis(ax2)
    ax2_twin.spines['top'].set_visible(False)
    ax2_twin.spines['right'].set_color(COLORS['ida_red'])

    plt.tight_layout()
    plt.savefig('charts/real_data_forecast_comparison.pdf', transparent=True)
    plt.close()
    print("Saved: real_data_forecast_comparison.pdf")

def plot_multiple_series_comparison(data):
    """Compare methods across multiple real datasets"""
    # Use different datasets
    datasets = {
        'Airline': data['airline']['Passengers'],
        'S&P 500': data['sp500']['Close'].resample('M').last().dropna(),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    for idx, (name, series) in enumerate(datasets.items()):
        row = idx

        # Ensure enough data
        if len(series) < 50:
            continue

        series = series.iloc[-120:]  # Last 120 observations

        n = len(series)
        train_size = int(n * 0.75)
        train = series.iloc[:train_size]
        test = series.iloc[train_size:]
        h = len(test)

        # Simple models
        if name == 'Airline':
            model = ExponentialSmoothing(train, seasonal_periods=12, trend='add', seasonal='mul',
                                          initialization_method='estimated').fit()
        else:
            model = Holt(train, initialization_method='estimated').fit()

        forecast = model.forecast(h)

        # Plot series and forecast
        ax1 = axes[row, 0]
        ax1.plot(train.index, train.values, color=COLORS['main_blue'], linewidth=0.8)
        ax1.plot(test.index, test.values, color=COLORS['dark_gray'], linewidth=1.5, label='Actual')
        ax1.plot(test.index, forecast.values, color=COLORS['ida_red'], linewidth=1.5, linestyle='--', label='Forecast')
        ax1.axvline(x=train.index[-1], color=COLORS['medium_gray'], linestyle=':', linewidth=0.6)
        ax1.set_title(f'{name}', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=7, frameon=False)
        setup_axis(ax1)

        # Residual plot
        ax2 = axes[row, 1]
        residuals = test.values - forecast.values
        ax2.bar(range(len(residuals)), residuals, color=COLORS['forest'], alpha=0.7, width=0.8)
        ax2.axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.5)

        rmse = np.sqrt(np.mean(residuals**2))
        mape = np.mean(np.abs(residuals/test.values)) * 100
        ax2.set_title(f'Errors (RMSE={rmse:.1f}, MAPE={mape:.1f}%)', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
        ax2.set_xlabel('Forecast Horizon')
        ax2.set_ylabel('Error')
        setup_axis(ax2)

    plt.tight_layout()
    plt.savefig('charts/multiple_series_comparison.pdf', transparent=True)
    plt.close()
    print("Saved: multiple_series_comparison.pdf")

# =============================================================================
# SECTION 3c: Trend and Seasonality Handling
# =============================================================================

def plot_detrending_methods():
    """Different detrending approaches"""
    np.random.seed(42)
    T = 150
    t = np.arange(T)

    # Create data with quadratic trend
    trend = 0.01*t**2 - t + 50
    seasonal = 5*np.sin(2*np.pi*t/12)
    noise = np.random.randn(T)*3
    y = trend + seasonal + noise

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    # Original
    axes[0, 0].plot(t, y, color=COLORS['main_blue'], linewidth=0.7)
    axes[0, 0].plot(t, trend, color=COLORS['ida_red'], linewidth=1.5, linestyle='--', label='True Trend')
    axes[0, 0].set_title('Original Series', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[0, 0].set_ylabel('$X_t$')
    axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8)
    setup_axis(axes[0, 0])

    # Linear detrending
    from scipy import stats as scipy_stats
    slope, intercept, _, _, _ = scipy_stats.linregress(t, y)
    linear_trend = slope*t + intercept
    detrended_linear = y - linear_trend

    axes[0, 1].plot(t, detrended_linear, color=COLORS['forest'], linewidth=0.7)
    axes[0, 1].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.5)
    axes[0, 1].set_title('Linear Detrending', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[0, 1].set_ylabel('Detrended')
    setup_axis(axes[0, 1])

    # Polynomial detrending
    poly_coef = np.polyfit(t, y, 2)
    poly_trend = np.polyval(poly_coef, t)
    detrended_poly = y - poly_trend

    axes[1, 0].plot(t, detrended_poly, color=COLORS['amber'], linewidth=0.7)
    axes[1, 0].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.5)
    axes[1, 0].set_title('Polynomial Detrending (degree=2)', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 0].set_xlabel('Time $t$')
    axes[1, 0].set_ylabel('Detrended')
    setup_axis(axes[1, 0])

    # HP Filter
    from statsmodels.tsa.filters.hp_filter import hpfilter
    cycle, hp_trend = hpfilter(y, lamb=1600)

    axes[1, 1].plot(t, cycle, color=COLORS['purple'], linewidth=0.7)
    axes[1, 1].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.5)
    axes[1, 1].set_title('HP Filter ($\\lambda=1600$)', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 1].set_xlabel('Time $t$')
    axes[1, 1].set_ylabel('Cyclical')
    setup_axis(axes[1, 1])

    plt.tight_layout()
    plt.savefig('charts/detrending_methods.pdf', transparent=True)
    plt.close()
    print("Saved: detrending_methods.pdf")

def plot_seasonal_adjustment(data):
    """Seasonal adjustment methods"""
    airline = data['airline']['Passengers']

    # Decomposition
    decomp = seasonal_decompose(airline, model='multiplicative', period=12)

    # Seasonal adjustment
    seasonally_adjusted = airline / decomp.seasonal

    # Seasonal differencing
    seasonal_diff = airline.diff(12).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    # Original
    axes[0, 0].plot(airline.index, airline.values, color=COLORS['main_blue'], linewidth=0.7)
    axes[0, 0].set_title('Original Series', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[0, 0].set_ylabel('Passengers')
    setup_axis(axes[0, 0])

    # Seasonal component
    axes[0, 1].plot(decomp.seasonal.index, decomp.seasonal.values, color=COLORS['forest'], linewidth=0.7)
    axes[0, 1].axhline(y=1, color=COLORS['medium_gray'], linestyle='--', linewidth=0.5)
    axes[0, 1].set_title('Seasonal Component $S_t$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[0, 1].set_ylabel('Seasonal Index')
    setup_axis(axes[0, 1])

    # Seasonally adjusted
    axes[1, 0].plot(seasonally_adjusted.index, seasonally_adjusted.values, color=COLORS['ida_red'], linewidth=0.7)
    axes[1, 0].set_title('Seasonally Adjusted: $X_t / S_t$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Adjusted')
    setup_axis(axes[1, 0])

    # Seasonal differencing
    axes[1, 1].plot(seasonal_diff.index, seasonal_diff.values, color=COLORS['amber'], linewidth=0.7)
    axes[1, 1].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.5)
    axes[1, 1].set_title('Seasonal Differencing: $\\Delta_{12} X_t$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Difference')
    setup_axis(axes[1, 1])

    plt.tight_layout()
    plt.savefig('charts/seasonal_adjustment.pdf', transparent=True)
    plt.close()
    print("Saved: seasonal_adjustment.pdf")

def plot_trend_estimation_comparison(data):
    """Compare trend estimation methods"""
    airline = data['airline']['Passengers']
    t = np.arange(len(airline))

    # Methods
    ma12 = airline.rolling(window=12, center=True).mean()

    # Polynomial fit
    poly_coef = np.polyfit(t, airline, 2)
    poly_trend = np.polyval(poly_coef, t)

    # Exponential smoothing level
    hw = ExponentialSmoothing(airline, seasonal_periods=12, trend='add', seasonal='mul',
                               initialization_method='estimated').fit()

    # LOESS (STL trend)
    stl = STL(airline, period=12, robust=True)
    result = stl.fit()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(airline.index, airline.values, color='#AAAAAA', linewidth=1.0, label='Observed', alpha=0.7)
    ax.plot(airline.index, ma12.values, color='#1A3A6E', linewidth=2.5, label='MA(12)')
    ax.plot(airline.index, poly_trend, color='#2E7D32', linewidth=2.5, label='Polynomial')
    ax.plot(airline.index, hw.level, color='#DC3545', linewidth=2.5, label='ETS Level')
    ax.plot(airline.index, result.trend, color='#E67E22', linewidth=2.5, label='STL/LOESS')

    ax.set_xlabel('Date')
    ax.set_ylabel('Passengers')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, frameon=False, fontsize=9)
    setup_axis(ax)

    plt.tight_layout()
    plt.savefig('charts/trend_estimation_comparison.pdf', transparent=True)
    plt.close()
    print("Saved: trend_estimation_comparison.pdf")

def plot_deterministic_trend_example():
    """Clear example of deterministic trend for students"""
    np.random.seed(42)
    T = 100
    t = np.arange(T)

    # Deterministic trend: y = 10 + 0.5*t + noise
    trend = 10 + 0.5 * t
    noise = np.random.randn(T) * 3
    y = trend + noise

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # Plot 1: The series with trend line
    ax1 = axes[0]
    ax1.plot(t, y, color=COLORS['main_blue'], linewidth=0.8, label='Observed $X_t$')
    ax1.plot(t, trend, color=COLORS['ida_red'], linewidth=2, linestyle='--', label='True trend')
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('$X_t$')
    ax1.set_title('Deterministic Trend', fontweight='bold', fontsize=10, color=COLORS['dark_gray'])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    setup_axis(ax1)

    # Plot 2: Detrended by regression
    ax2 = axes[1]
    detrended = y - trend
    ax2.plot(t, detrended, color=COLORS['forest'], linewidth=0.8)
    ax2.axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('$X_t - \\hat{T}_t$')
    ax2.set_title('After Detrending', fontweight='bold', fontsize=10, color=COLORS['dark_gray'])
    setup_axis(ax2)

    # Plot 3: ACF of detrended
    ax3 = axes[2]
    acf_vals = acf(detrended, nlags=20)
    ax3.bar(range(len(acf_vals)), acf_vals, color=COLORS['forest'], width=0.6)
    ax3.axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
    ax3.axhline(y=1.96/np.sqrt(T), color=COLORS['ida_red'], linestyle='--', linewidth=0.5)
    ax3.axhline(y=-1.96/np.sqrt(T), color=COLORS['ida_red'], linestyle='--', linewidth=0.5)
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('ACF')
    ax3.set_title('ACF: Stationary!', fontweight='bold', fontsize=10, color=COLORS['dark_gray'])
    setup_axis(ax3)

    plt.tight_layout()
    plt.savefig('charts/deterministic_trend_example.pdf', transparent=True)
    plt.close()
    print("Saved: deterministic_trend_example.pdf")

def plot_stochastic_trend_example():
    """Clear example of stochastic trend (random walk) for students"""
    np.random.seed(123)
    T = 100
    t = np.arange(T)

    # Stochastic trend: random walk
    shocks = np.random.randn(T) * 2
    y = np.cumsum(shocks) + 50  # Random walk starting at 50

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # Plot 1: The random walk
    ax1 = axes[0]
    ax1.plot(t, y, color=COLORS['main_blue'], linewidth=1)
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('$X_t$')
    ax1.set_title('Stochastic Trend', fontweight='bold', fontsize=10, color=COLORS['dark_gray'])
    setup_axis(ax1)

    # Plot 2: After differencing
    ax2 = axes[1]
    differenced = np.diff(y)
    ax2.plot(t[1:], differenced, color=COLORS['forest'], linewidth=0.8)
    ax2.axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.5)
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('$\\Delta X_t = X_t - X_{t-1}$')
    ax2.set_title('After Differencing', fontweight='bold', fontsize=10, color=COLORS['dark_gray'])
    setup_axis(ax2)

    # Plot 3: ACF of differenced
    ax3 = axes[2]
    acf_vals = acf(differenced, nlags=20)
    ax3.bar(range(len(acf_vals)), acf_vals, color=COLORS['forest'], width=0.6)
    ax3.axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
    ax3.axhline(y=1.96/np.sqrt(T), color=COLORS['ida_red'], linestyle='--', linewidth=0.5)
    ax3.axhline(y=-1.96/np.sqrt(T), color=COLORS['ida_red'], linestyle='--', linewidth=0.5)
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('ACF')
    ax3.set_title('ACF: Stationary!', fontweight='bold', fontsize=10, color=COLORS['dark_gray'])
    setup_axis(ax3)

    plt.tight_layout()
    plt.savefig('charts/stochastic_trend_example.pdf', transparent=True)
    plt.close()
    print("Saved: stochastic_trend_example.pdf")

def plot_trend_comparison_sidebyside():
    """Side by side comparison of both trend types"""
    np.random.seed(42)
    T = 150
    t = np.arange(T)

    # Deterministic: linear trend + noise
    det_trend = 20 + 0.3 * t
    det_series = det_trend + np.random.randn(T) * 5

    # Stochastic: random walk
    np.random.seed(55)
    stoch_series = np.cumsum(np.random.randn(T) * 2) + 50

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Top left: Deterministic trend
    ax1 = axes[0, 0]
    ax1.plot(t, det_series, color=COLORS['main_blue'], linewidth=0.7, label='$X_t$')
    ax1.plot(t, det_trend, color=COLORS['ida_red'], linewidth=2, linestyle='--', label='Trend')
    ax1.set_ylabel('$X_t$')
    ax1.set_title('Deterministic: $X_t = \\beta_0 + \\beta_1 t + \\varepsilon_t$',
                  fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8, frameon=False)
    setup_axis(ax1)

    # Top right: Stochastic trend
    ax2 = axes[0, 1]
    ax2.plot(t, stoch_series, color=COLORS['main_blue'], linewidth=0.7)
    ax2.set_ylabel('$X_t$')
    ax2.set_title('Stochastic: $X_t = X_{t-1} + \\varepsilon_t$',
                  fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax2)

    # Bottom left: Detrended (regression)
    ax3 = axes[1, 0]
    detrended = det_series - det_trend
    ax3.plot(t, detrended, color=COLORS['forest'], linewidth=0.7)
    ax3.axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.5)
    ax3.set_xlabel('Time $t$')
    ax3.set_ylabel('Residual')
    ax3.set_title('Solution: Regression $\\rightarrow$ Stationary',
                  fontweight='bold', fontsize=9, color=COLORS['forest'])
    setup_axis(ax3)

    # Bottom right: Differenced
    ax4 = axes[1, 1]
    differenced = np.diff(stoch_series)
    ax4.plot(t[1:], differenced, color=COLORS['forest'], linewidth=0.7)
    ax4.axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.5)
    ax4.set_xlabel('Time $t$')
    ax4.set_ylabel('$\\Delta X_t$')
    ax4.set_title('Solution: Differencing $\\rightarrow$ Stationary',
                  fontweight='bold', fontsize=9, color=COLORS['forest'])
    setup_axis(ax4)

    plt.tight_layout()
    plt.savefig('charts/trend_comparison_sidebyside.pdf', transparent=True)
    plt.close()
    print("Saved: trend_comparison_sidebyside.pdf")

# =============================================================================
# SECTION 4: Stochastic Processes
# =============================================================================

def plot_realizations_ensemble():
    """Multiple realizations"""
    np.random.seed(42)
    T = 100
    n_paths = 6

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax1 = axes[0]
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_paths))
    for i in range(n_paths):
        np.random.seed(i)
        path = np.cumsum(np.random.randn(T) * 0.5)
        ax1.plot(path, color=colors[i], linewidth=0.7, alpha=0.8)

    ax1.axvline(x=50, color=COLORS['ida_red'], linestyle='--', linewidth=1)
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('$X_t(\\omega)$')
    ax1.set_title('Multiple Realizations', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax1)

    ax2 = axes[1]
    ensemble_values = []
    for i in range(1000):
        np.random.seed(i)
        path = np.cumsum(np.random.randn(100) * 0.5)
        ensemble_values.append(path[50])

    ax2.hist(ensemble_values, bins=30, color=COLORS['main_blue'], alpha=0.7, edgecolor='white')
    ax2.axvline(x=np.mean(ensemble_values), color=COLORS['ida_red'], linewidth=1.5)
    ax2.set_xlabel('$X_{50}$')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution at $t=50$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax2)

    plt.tight_layout()
    plt.savefig('charts/realizations_ensemble.pdf', transparent=True)
    plt.close()
    print("Saved: realizations_ensemble.pdf")

# =============================================================================
# SECTION 5: White Noise and Random Walk
# =============================================================================

def plot_white_noise():
    """White noise process"""
    np.random.seed(42)
    T = 150
    wn = np.random.randn(T)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    ax1 = axes[0]
    ax1.plot(wn, color=COLORS['main_blue'], linewidth=0.5)
    ax1.axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.5)
    ax1.axhline(y=2, color=COLORS['ida_red'], linestyle='--', linewidth=0.5, alpha=0.6)
    ax1.axhline(y=-2, color=COLORS['ida_red'], linestyle='--', linewidth=0.5, alpha=0.6)
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('$\\varepsilon_t$')
    ax1.set_title('White Noise', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax1)

    ax2 = axes[1]
    ax2.hist(wn, bins=25, color=COLORS['main_blue'], alpha=0.7, edgecolor='white', density=True)
    x_norm = np.linspace(-4, 4, 100)
    ax2.plot(x_norm, 1/np.sqrt(2*np.pi) * np.exp(-x_norm**2/2), color=COLORS['ida_red'], linewidth=1.5)
    ax2.set_xlabel('$\\varepsilon_t$')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax2)

    plt.tight_layout()
    plt.savefig('charts/white_noise.pdf', transparent=True)
    plt.close()
    print("Saved: white_noise.pdf")

def plot_random_walk():
    """Random walk with variance growth"""
    np.random.seed(42)
    T = 200
    n_paths = 4

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax1 = axes[0]
    colors = [COLORS['main_blue'], COLORS['ida_red'], COLORS['forest'], COLORS['amber']]
    for i, color in enumerate(colors):
        np.random.seed(i*10)
        rw = np.cumsum(np.random.randn(T))
        ax1.plot(rw, color=color, linewidth=0.7, alpha=0.8)

    t = np.arange(1, T+1)
    ax1.fill_between(t, 2*np.sqrt(t), -2*np.sqrt(t), color=COLORS['slate'], alpha=0.15)
    ax1.plot(t, 2*np.sqrt(t), color=COLORS['slate'], linestyle='--', linewidth=0.8)
    ax1.plot(t, -2*np.sqrt(t), color=COLORS['slate'], linestyle='--', linewidth=0.8)
    ax1.axhline(y=0, color=COLORS['medium_gray'], linestyle='-', linewidth=0.4)
    ax1.set_xlabel('Time $t$')
    ax1.set_ylabel('$X_t$')
    ax1.set_title('Random Walk Paths', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax1)

    ax2 = axes[1]
    ax2.plot(t, t, color=COLORS['ida_red'], linewidth=1.5)
    ax2.fill_between(t, 0, t, color=COLORS['ida_red'], alpha=0.2)
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('$Var(X_t) = t\\sigma^2$')
    ax2.set_title('Variance Growth', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax2)

    plt.tight_layout()
    plt.savefig('charts/random_walk.pdf', transparent=True)
    plt.close()
    print("Saved: random_walk.pdf")

def plot_random_walk_vs_stationary():
    """Compare stationary vs non-stationary"""
    np.random.seed(42)
    T = 150

    rw = np.cumsum(np.random.randn(T))
    phi = 0.7
    ar1 = np.zeros(T)
    for t in range(1, T):
        ar1[t] = phi * ar1[t-1] + np.random.randn()

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.5))

    axes[0, 0].plot(rw, color=COLORS['ida_red'], linewidth=0.7)
    axes[0, 0].set_title('Random Walk', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[0, 0].set_ylabel('$X_t$')
    setup_axis(axes[0, 0])

    axes[0, 1].plot(ar1, color=COLORS['forest'], linewidth=0.7)
    axes[0, 1].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)
    axes[0, 1].set_title('AR(1): $\\phi=0.7$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[0, 1].set_ylabel('$X_t$')
    setup_axis(axes[0, 1])

    acf_rw = acf(rw, nlags=25)
    axes[1, 0].bar(range(len(acf_rw)), acf_rw, color=COLORS['ida_red'], width=0.6)
    axes[1, 0].axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
    axes[1, 0].axhline(y=1.96/np.sqrt(T), color=COLORS['slate'], linestyle='--', linewidth=0.5)
    axes[1, 0].axhline(y=-1.96/np.sqrt(T), color=COLORS['slate'], linestyle='--', linewidth=0.5)
    axes[1, 0].set_title('ACF: Slow Decay', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')
    setup_axis(axes[1, 0])

    acf_ar1 = acf(ar1, nlags=25)
    axes[1, 1].bar(range(len(acf_ar1)), acf_ar1, color=COLORS['forest'], width=0.6)
    axes[1, 1].axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
    axes[1, 1].axhline(y=1.96/np.sqrt(T), color=COLORS['slate'], linestyle='--', linewidth=0.5)
    axes[1, 1].axhline(y=-1.96/np.sqrt(T), color=COLORS['slate'], linestyle='--', linewidth=0.5)
    axes[1, 1].set_title('ACF: Fast Decay', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('ACF')
    setup_axis(axes[1, 1])

    plt.tight_layout()
    plt.savefig('charts/rw_vs_stationary.pdf', transparent=True)
    plt.close()
    print("Saved: rw_vs_stationary.pdf")

# =============================================================================
# SECTION 6: ACF and PACF
# =============================================================================

def plot_acf_pacf_examples():
    """ACF/PACF for different processes"""
    np.random.seed(42)
    T = 400

    wn = np.random.randn(T)
    ar1 = np.zeros(T)
    for t in range(1, T):
        ar1[t] = 0.8 * ar1[t-1] + np.random.randn()
    ma1 = np.zeros(T)
    eps = np.random.randn(T)
    for t in range(1, T):
        ma1[t] = eps[t] + 0.7 * eps[t-1]

    fig, axes = plt.subplots(3, 3, figsize=(11, 7))

    processes = [
        (wn, 'White Noise', COLORS['main_blue']),
        (ar1, 'AR(1)', COLORS['ida_red']),
        (ma1, 'MA(1)', COLORS['forest']),
    ]

    for i, (series, title, color) in enumerate(processes):
        # Time series
        axes[i, 0].plot(series[:80], color=color, linewidth=0.5)
        axes[i, 0].set_title(title, fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
        axes[i, 0].set_ylabel('$X_t$')
        if i == 2:
            axes[i, 0].set_xlabel('$t$')
        setup_axis(axes[i, 0])

        # ACF
        acf_vals = acf(series, nlags=15)
        axes[i, 1].bar(range(len(acf_vals)), acf_vals, color=color, width=0.6)
        axes[i, 1].axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
        axes[i, 1].axhline(y=1.96/np.sqrt(T), color=COLORS['slate'], linestyle='--', linewidth=0.4)
        axes[i, 1].axhline(y=-1.96/np.sqrt(T), color=COLORS['slate'], linestyle='--', linewidth=0.4)
        axes[i, 1].set_title('ACF', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
        axes[i, 1].set_ylim(-0.3, 1.05)
        if i == 2:
            axes[i, 1].set_xlabel('Lag')
        setup_axis(axes[i, 1])

        # PACF
        pacf_vals = pacf(series, nlags=15)
        axes[i, 2].bar(range(len(pacf_vals)), pacf_vals, color=color, width=0.6)
        axes[i, 2].axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
        axes[i, 2].axhline(y=1.96/np.sqrt(T), color=COLORS['slate'], linestyle='--', linewidth=0.4)
        axes[i, 2].axhline(y=-1.96/np.sqrt(T), color=COLORS['slate'], linestyle='--', linewidth=0.4)
        axes[i, 2].set_title('PACF', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
        axes[i, 2].set_ylim(-0.3, 1.05)
        if i == 2:
            axes[i, 2].set_xlabel('Lag')
        setup_axis(axes[i, 2])

    plt.tight_layout()
    plt.savefig('charts/acf_pacf_examples.pdf', transparent=True)
    plt.close()
    print("Saved: acf_pacf_examples.pdf")

def plot_acf_theoretical():
    """Theoretical ACF for AR(1)"""
    phi_values = [0.9, 0.5, -0.5, -0.9]
    lags = np.arange(0, 16)

    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5))
    axes = axes.flatten()
    colors = [COLORS['main_blue'], COLORS['ida_red'], COLORS['forest'], COLORS['amber']]

    for ax, phi, color in zip(axes, phi_values, colors):
        acf_theo = phi ** lags
        ax.bar(lags, acf_theo, color=color, width=0.6, alpha=0.8)
        ax.axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
        ax.set_xlabel('Lag $h$')
        ax.set_ylabel('$\\rho(h)$')
        ax.set_title(f'$\\phi = {phi}$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
        ax.set_ylim(-1.05, 1.05)
        setup_axis(ax)

    plt.tight_layout()
    plt.savefig('charts/acf_theoretical.pdf', transparent=True)
    plt.close()
    print("Saved: acf_theoretical.pdf")

# =============================================================================
# SECTION 7: Differencing
# =============================================================================

def plot_differencing_effect(data):
    """Effect of differencing"""
    sp500 = data['sp500']['Close'].dropna()
    diff1 = sp500.diff().dropna()
    log_returns = (np.log(sp500) - np.log(sp500.shift(1))).dropna() * 100

    fig, axes = plt.subplots(3, 2, figsize=(11, 7))

    # Original
    axes[0, 0].plot(sp500.index, sp500.values, color=COLORS['main_blue'], linewidth=0.5)
    axes[0, 0].set_title('Price $X_t$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[0, 0].set_ylabel('USD')
    setup_axis(axes[0, 0])

    acf_orig = acf(sp500, nlags=25)
    axes[0, 1].bar(range(len(acf_orig)), acf_orig, color=COLORS['main_blue'], width=0.6)
    axes[0, 1].axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
    axes[0, 1].set_title('ACF of $X_t$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(axes[0, 1])

    # First difference
    axes[1, 0].plot(diff1.index, diff1.values, color=COLORS['ida_red'], linewidth=0.3)
    axes[1, 0].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)
    axes[1, 0].set_title('$\\Delta X_t = X_t - X_{t-1}$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 0].set_ylabel('USD')
    setup_axis(axes[1, 0])

    acf_diff = acf(diff1, nlags=25)
    axes[1, 1].bar(range(len(acf_diff)), acf_diff, color=COLORS['ida_red'], width=0.6)
    axes[1, 1].axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
    axes[1, 1].axhline(y=1.96/np.sqrt(len(diff1)), color=COLORS['slate'], linestyle='--', linewidth=0.4)
    axes[1, 1].axhline(y=-1.96/np.sqrt(len(diff1)), color=COLORS['slate'], linestyle='--', linewidth=0.4)
    axes[1, 1].set_title('ACF of $\\Delta X_t$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(axes[1, 1])

    # Log returns
    axes[2, 0].plot(log_returns.index, log_returns.values, color=COLORS['forest'], linewidth=0.3)
    axes[2, 0].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)
    axes[2, 0].set_title('Log Returns $r_t$ (%)', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[2, 0].set_ylabel('%')
    axes[2, 0].set_xlabel('Date')
    setup_axis(axes[2, 0])

    acf_ret = acf(log_returns, nlags=25)
    axes[2, 1].bar(range(len(acf_ret)), acf_ret, color=COLORS['forest'], width=0.6)
    axes[2, 1].axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
    axes[2, 1].axhline(y=1.96/np.sqrt(len(log_returns)), color=COLORS['slate'], linestyle='--', linewidth=0.4)
    axes[2, 1].axhline(y=-1.96/np.sqrt(len(log_returns)), color=COLORS['slate'], linestyle='--', linewidth=0.4)
    axes[2, 1].set_title('ACF of $r_t$', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[2, 1].set_xlabel('Lag')
    setup_axis(axes[2, 1])

    plt.tight_layout()
    plt.savefig('charts/differencing_effect.pdf', transparent=True)
    plt.close()
    print("Saved: differencing_effect.pdf")

# =============================================================================
# SECTION 8: Stationarity Tests
# =============================================================================

def plot_adf_test_visualization(data):
    """ADF test visualization"""
    sp500 = data['sp500']['Close'].dropna()
    returns = (np.log(sp500) - np.log(sp500.shift(1))).dropna() * 100

    adf_price = adfuller(sp500, maxlag=20, autolag='AIC')
    adf_returns = adfuller(returns, maxlag=20, autolag='AIC')

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    # Price
    axes[0, 0].plot(sp500.index, sp500.values, color=COLORS['ida_red'], linewidth=0.5)
    axes[0, 0].set_title('Price Level', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[0, 0].set_ylabel('USD')
    setup_axis(axes[0, 0])

    axes[0, 1].barh(['ADF Stat', '1%', '5%', '10%'],
                    [adf_price[0], adf_price[4]['1%'], adf_price[4]['5%'], adf_price[4]['10%']],
                    color=[COLORS['ida_red'], COLORS['slate'], COLORS['slate'], COLORS['slate']], height=0.6)
    axes[0, 1].axvline(x=0, color=COLORS['medium_gray'], linewidth=0.5)
    axes[0, 1].set_title(f'ADF: p={adf_price[1]:.3f}', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(axes[0, 1])

    # Returns
    axes[1, 0].plot(returns.index, returns.values, color=COLORS['forest'], linewidth=0.3)
    axes[1, 0].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)
    axes[1, 0].set_title('Log Returns', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 0].set_ylabel('%')
    axes[1, 0].set_xlabel('Date')
    setup_axis(axes[1, 0])

    axes[1, 1].barh(['ADF Stat', '1%', '5%', '10%'],
                    [adf_returns[0], adf_returns[4]['1%'], adf_returns[4]['5%'], adf_returns[4]['10%']],
                    color=[COLORS['forest'], COLORS['slate'], COLORS['slate'], COLORS['slate']], height=0.6)
    axes[1, 1].axvline(x=0, color=COLORS['medium_gray'], linewidth=0.5)
    axes[1, 1].set_title(f'ADF: p={adf_returns[1]:.3f}', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 1].set_xlabel('Value')
    setup_axis(axes[1, 1])

    plt.tight_layout()
    plt.savefig('charts/adf_test_visualization.pdf', transparent=True)
    plt.close()
    print("Saved: adf_test_visualization.pdf")

# =============================================================================
# SECTION 9: Financial Data
# =============================================================================

def plot_sp500_analysis(data):
    """S&P 500 comprehensive analysis"""
    sp500 = data['sp500']['Close'].dropna()
    returns = (np.log(sp500) - np.log(sp500.shift(1))).dropna() * 100

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))

    axes[0, 0].plot(sp500.index, sp500.values, color=COLORS['main_blue'], linewidth=0.5)
    axes[0, 0].set_title('S&P 500 Price', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[0, 0].set_ylabel('USD')
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    setup_axis(axes[0, 0])

    axes[0, 1].plot(returns.index, returns.values, color=COLORS['main_blue'], linewidth=0.25)
    axes[0, 1].axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)
    high_vol = abs(returns) > 2
    axes[0, 1].fill_between(returns.index, returns.values, 0, where=high_vol.values,
                            color=COLORS['ida_red'], alpha=0.4)
    axes[0, 1].set_title('Daily Returns (%)', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[0, 1].set_ylabel('%')
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    setup_axis(axes[0, 1])

    acf_ret = acf(returns, nlags=25)
    axes[1, 0].bar(range(len(acf_ret)), acf_ret, color=COLORS['forest'], width=0.6)
    axes[1, 0].axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
    axes[1, 0].axhline(y=1.96/np.sqrt(len(returns)), color=COLORS['ida_red'], linestyle='--', linewidth=0.4)
    axes[1, 0].axhline(y=-1.96/np.sqrt(len(returns)), color=COLORS['ida_red'], linestyle='--', linewidth=0.4)
    axes[1, 0].set_title('ACF of Returns', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 0].set_xlabel('Lag')
    setup_axis(axes[1, 0])

    acf_sq = acf(returns**2, nlags=25)
    axes[1, 1].bar(range(len(acf_sq)), acf_sq, color=COLORS['amber'], width=0.6)
    axes[1, 1].axhline(y=0, color=COLORS['medium_gray'], linewidth=0.4)
    axes[1, 1].axhline(y=1.96/np.sqrt(len(returns)), color=COLORS['ida_red'], linestyle='--', linewidth=0.4)
    axes[1, 1].axhline(y=-1.96/np.sqrt(len(returns)), color=COLORS['ida_red'], linestyle='--', linewidth=0.4)
    axes[1, 1].set_title('ACF of $r_t^2$ (Volatility)', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    axes[1, 1].set_xlabel('Lag')
    setup_axis(axes[1, 1])

    plt.tight_layout()
    plt.savefig('charts/sp500_analysis.pdf', transparent=True)
    plt.close()
    print("Saved: sp500_analysis.pdf")

def plot_returns_distribution(data):
    """Returns distribution with equal axes QQ plot"""
    sp500 = data['sp500']['Close'].dropna()
    returns = (np.log(sp500) - np.log(sp500.shift(1))).dropna() * 100

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax1 = axes[0]
    ax1.hist(returns, bins=80, color=COLORS['main_blue'], alpha=0.7, edgecolor='white', density=True)
    x = np.linspace(returns.min(), returns.max(), 200)
    normal_pdf = 1/(returns.std()*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-returns.mean())/returns.std())**2)
    ax1.plot(x, normal_pdf, color=COLORS['ida_red'], linewidth=1.5)
    ax1.set_xlabel('Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax1)

    # Stats box
    stats_text = f'$\\mu$={returns.mean():.3f}%\n$\\sigma$={returns.std():.3f}%\nSkew={returns.skew():.2f}\nKurt={returns.kurtosis():.1f}'
    ax1.text(0.97, 0.97, stats_text, transform=ax1.transAxes, va='top', ha='right', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    # QQ plot with equal axes
    from scipy import stats
    ax2 = axes[1]

    # Standardize returns
    standardized = (returns - returns.mean()) / returns.std()
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.001, 0.999, len(standardized)))
    sample_quantiles = np.sort(standardized.values)

    # Use same limits for both axes
    limit = max(abs(theoretical_quantiles.min()), abs(theoretical_quantiles.max()),
                abs(sample_quantiles.min()), abs(sample_quantiles.max()))
    limit = min(limit, 5)  # Cap at 5 for visibility

    ax2.scatter(theoretical_quantiles, sample_quantiles, color=COLORS['main_blue'], s=3, alpha=0.5)
    ax2.plot([-limit, limit], [-limit, limit], color=COLORS['ida_red'], linewidth=1.5, linestyle='--')
    ax2.set_xlim(-limit, limit)
    ax2.set_ylim(-limit, limit)
    ax2.set_aspect('equal')
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Sample Quantiles')
    ax2.set_title('Q-Q Plot', fontweight='bold', fontsize=9, color=COLORS['dark_gray'])
    setup_axis(ax2)

    plt.tight_layout()
    plt.savefig('charts/returns_distribution.pdf', transparent=True)
    plt.close()
    print("Saved: returns_distribution.pdf")

def plot_volatility_clustering(data):
    """Volatility clustering"""
    sp500 = data['sp500']['Close'].dropna()
    returns = (np.log(sp500) - np.log(sp500.shift(1))).dropna() * 100

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(returns.index, returns.values, color=COLORS['main_blue'], linewidth=0.3, alpha=0.8)
    ax.axhline(y=0, color=COLORS['medium_gray'], linestyle='--', linewidth=0.4)

    high_vol = abs(returns) > 2
    ax.fill_between(returns.index, returns.values, 0, where=high_vol.values,
                    color=COLORS['ida_red'], alpha=0.4)

    ax.set_xlabel('Date')
    ax.set_ylabel('Return (%)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    setup_axis(ax)

    plt.tight_layout()
    plt.savefig('charts/volatility_clustering.pdf', transparent=True)
    plt.close()
    print("Saved: volatility_clustering.pdf")

# =============================================================================
# MAIN
# =============================================================================

def main():
    import os
    os.makedirs('charts', exist_ok=True)

    print("="*60)
    print("Generating Time Series Charts")
    print("="*60)

    data = download_data()

    print("\n--- Section 1: Time Series Definition ---")
    plot_timeseries_definition(data)
    plot_multiple_assets(data)
    plot_data_types_comparison()

    print("\n--- Section 2: Decomposition ---")
    plot_ts_components_synthetic()
    plot_airline_decomposition(data)
    plot_additive_vs_multiplicative(data)
    plot_moving_average_trend()
    plot_stl_decomposition(data)
    plot_seasonal_pattern(data)

    print("\n--- Section 3: Exponential Smoothing ---")
    plot_simple_exp_smoothing(data)
    plot_holt_method(data)
    plot_holt_winters(data)
    plot_ets_components(data)

    print("\n--- Section 3b: Forecast Evaluation ---")
    plot_forecast_accuracy_metrics(data)
    plot_residual_diagnostics(data)
    plot_cross_validation_forecast(data)
    plot_train_test_validation()
    plot_real_data_forecast_comparison(data)
    plot_multiple_series_comparison(data)

    print("\n--- Section 3c: Seasonality Modeling ---")
    plot_seasonality_fourier_dummies(data)

    print("\n--- Section 3d: Trend & Seasonality Handling ---")
    plot_detrending_methods()
    plot_seasonal_adjustment(data)
    plot_trend_estimation_comparison(data)
    plot_deterministic_trend_example()
    plot_stochastic_trend_example()
    plot_trend_comparison_sidebyside()

    print("\n--- Section 4: Stochastic Processes ---")
    plot_realizations_ensemble()

    print("\n--- Section 5: White Noise & Random Walk ---")
    plot_white_noise()
    plot_random_walk()
    plot_random_walk_vs_stationary()

    print("\n--- Section 6: ACF/PACF ---")
    plot_acf_pacf_examples()
    plot_acf_theoretical()

    print("\n--- Section 7: Differencing ---")
    plot_differencing_effect(data)

    print("\n--- Section 8: Stationarity Tests ---")
    plot_adf_test_visualization(data)

    print("\n--- Section 9: Financial Data ---")
    plot_sp500_analysis(data)
    plot_returns_distribution(data)
    plot_volatility_clustering(data)

    print("\n" + "="*60)
    print(f"Generated {len([f for f in os.listdir('charts') if f.endswith('.pdf')])} PDF charts")
    print("="*60)

if __name__ == '__main__':
    main()
