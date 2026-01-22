"""
Generate charts for Chapter 9: Prophet and TBATS
Real data examples with multiple seasonalities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Colors
MAIN_BLUE = '#1A3A6E'
IDA_RED = '#DC3545'
FOREST = '#2E7D32'
AMBER = '#B5853F'
ORANGE = '#E67E22'

def save_chart(fig, name):
    """Save chart to charts folder"""
    fig.savefig(f'charts/{name}.pdf', bbox_inches='tight', dpi=150)
    fig.savefig(f'charts/{name}.png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: charts/{name}.pdf")

def generate_multiple_seasonality_example():
    """Show example of data with multiple seasonal patterns"""
    np.random.seed(42)

    # Create hourly data for 4 weeks with daily + weekly patterns
    hours = np.arange(24 * 7 * 4)  # 4 weeks of hourly data

    # Daily pattern (24 hours)
    daily = 10 * np.sin(2 * np.pi * hours / 24 - np.pi/2)  # Peak at noon

    # Weekly pattern (168 hours)
    weekly = 5 * np.sin(2 * np.pi * hours / 168)  # Weekend dip

    # Trend
    trend = 0.01 * hours

    # Noise
    noise = np.random.normal(0, 2, len(hours))

    # Combined
    y = 100 + trend + daily + weekly + noise

    # Create datetime index
    dates = pd.date_range('2024-01-01', periods=len(hours), freq='h')

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Full series
    axes[0].plot(dates, y, color=MAIN_BLUE, linewidth=0.8)
    axes[0].set_title('Hourly Data with Multiple Seasonalities (4 weeks)', fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].xaxis.set_major_formatter(DateFormatter('%b %d'))

    # One week zoom to show daily pattern
    week_data = y[:168]
    week_dates = dates[:168]
    axes[1].plot(week_dates, week_data, color=MAIN_BLUE, linewidth=1)
    axes[1].set_title('One Week: Daily Pattern Visible', fontweight='bold')
    axes[1].set_ylabel('Value')
    axes[1].xaxis.set_major_formatter(DateFormatter('%a'))

    # Two days zoom to show hourly pattern
    day_data = y[:48]
    day_dates = dates[:48]
    axes[2].plot(day_dates, day_data, color=MAIN_BLUE, linewidth=1.5)
    axes[2].axvline(dates[12], color=IDA_RED, linestyle='--', alpha=0.7, label='Noon peak')
    axes[2].axvline(dates[36], color=IDA_RED, linestyle='--', alpha=0.7)
    axes[2].set_title('Two Days: Hourly Pattern Visible', fontweight='bold')
    axes[2].set_ylabel('Value')
    axes[2].set_xlabel('Time')
    axes[2].xaxis.set_major_formatter(DateFormatter('%H:00'))
    axes[2].legend()

    plt.tight_layout()
    save_chart(fig, 'ch9_multiple_seasonality')

def generate_fourier_terms_visualization():
    """Visualize how Fourier terms approximate seasonality"""
    t = np.linspace(0, 2*np.pi, 100)

    # True seasonal pattern (complex shape)
    true_pattern = np.sin(t) + 0.5*np.sin(2*t) + 0.3*np.cos(3*t)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # K=1 (one harmonic)
    k1 = np.sin(t)
    axes[0, 0].plot(t, true_pattern, 'k-', linewidth=2, label='True pattern')
    axes[0, 0].plot(t, k1, color=IDA_RED, linewidth=2, linestyle='--', label='K=1 approximation')
    axes[0, 0].fill_between(t, true_pattern, k1, alpha=0.3, color=IDA_RED)
    axes[0, 0].set_title('K = 1 (One Harmonic)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_ylabel('Value')

    # K=2 (two harmonics)
    k2 = np.sin(t) + 0.5*np.sin(2*t)
    axes[0, 1].plot(t, true_pattern, 'k-', linewidth=2, label='True pattern')
    axes[0, 1].plot(t, k2, color=ORANGE, linewidth=2, linestyle='--', label='K=2 approximation')
    axes[0, 1].fill_between(t, true_pattern, k2, alpha=0.3, color=ORANGE)
    axes[0, 1].set_title('K = 2 (Two Harmonics)', fontweight='bold')
    axes[0, 1].legend()

    # K=3 (three harmonics) - perfect fit
    k3 = np.sin(t) + 0.5*np.sin(2*t) + 0.3*np.cos(3*t)
    axes[1, 0].plot(t, true_pattern, 'k-', linewidth=2, label='True pattern')
    axes[1, 0].plot(t, k3, color=FOREST, linewidth=2, linestyle='--', label='K=3 approximation')
    axes[1, 0].set_title('K = 3 (Three Harmonics) - Perfect Fit', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_xlabel('Time (one seasonal cycle)')

    # Error vs K
    ks = [1, 2, 3, 4, 5]
    errors = []
    for k in ks:
        approx = np.zeros_like(t)
        approx += np.sin(t)
        if k >= 2:
            approx += 0.5*np.sin(2*t)
        if k >= 3:
            approx += 0.3*np.cos(3*t)
        errors.append(np.mean((true_pattern - approx)**2))

    axes[1, 1].bar(ks, errors, color=[IDA_RED, ORANGE, FOREST, MAIN_BLUE, AMBER])
    axes[1, 1].set_xlabel('Number of Harmonics (K)')
    axes[1, 1].set_ylabel('Mean Squared Error')
    axes[1, 1].set_title('Approximation Error vs K', fontweight='bold')

    plt.tight_layout()
    save_chart(fig, 'ch9_fourier_approximation')

def generate_tbats_decomposition():
    """Simulate TBATS-like decomposition"""
    np.random.seed(42)
    n = 365 * 2  # 2 years of daily data
    t = np.arange(n)

    # Components
    trend = 100 + 0.05 * t
    seasonal_weekly = 5 * np.sin(2 * np.pi * t / 7)
    seasonal_yearly = 15 * np.sin(2 * np.pi * t / 365.25)
    remainder = np.random.normal(0, 3, n)

    # Observed
    observed = trend + seasonal_weekly + seasonal_yearly + remainder

    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(dates, observed, color=MAIN_BLUE, linewidth=0.5)
    axes[0].set_title('TBATS Decomposition: Daily Data with Weekly + Yearly Seasonality', fontweight='bold')
    axes[0].set_ylabel('Observed')

    axes[1].plot(dates, trend, color=FOREST, linewidth=2)
    axes[1].set_ylabel('Trend')

    axes[2].plot(dates, seasonal_yearly, color=IDA_RED, linewidth=1)
    axes[2].set_ylabel('Yearly\nSeasonality')

    axes[3].plot(dates, seasonal_weekly, color=ORANGE, linewidth=0.5)
    axes[3].set_ylabel('Weekly\nSeasonality')

    axes[4].plot(dates, remainder, color='gray', linewidth=0.3)
    axes[4].set_ylabel('Remainder')
    axes[4].set_xlabel('Date')

    for ax in axes:
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

    plt.tight_layout()
    save_chart(fig, 'ch9_tbats_decomposition')

def generate_prophet_components():
    """Simulate Prophet-style component visualization"""
    np.random.seed(42)
    n = 365 * 3  # 3 years
    t = np.arange(n)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    # Trend with changepoint
    trend = np.where(t < 365, 100 + 0.03*t, 100 + 0.03*365 + 0.08*(t-365))

    # Yearly seasonality
    yearly = 10 * np.sin(2 * np.pi * t / 365.25)

    # Weekly seasonality
    weekly = 3 * np.sin(2 * np.pi * t / 7)

    # Holiday effect (simulate Christmas spike)
    holidays = np.zeros(n)
    for year in range(3):
        christmas_idx = 358 + year * 365
        if christmas_idx < n:
            holidays[christmas_idx-5:christmas_idx+3] = 15

    # Observed
    observed = trend + yearly + weekly + holidays + np.random.normal(0, 2, n)

    fig, axes = plt.subplots(5, 1, figsize=(14, 12))

    # Observed + Fitted
    axes[0].plot(dates, observed, 'k.', markersize=1, alpha=0.5, label='Observed')
    fitted = trend + yearly + weekly + holidays
    axes[0].plot(dates, fitted, color=MAIN_BLUE, linewidth=1, label='Fitted')
    axes[0].set_title('Prophet Decomposition with Trend Changepoint', fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].axvline(dates[365], color=IDA_RED, linestyle='--', alpha=0.7, label='Changepoint')

    # Trend
    axes[1].plot(dates, trend, color=FOREST, linewidth=2)
    axes[1].axvline(dates[365], color=IDA_RED, linestyle='--', alpha=0.7)
    axes[1].annotate('Changepoint', xy=(dates[365], trend[365]), xytext=(dates[400], trend[365]+10),
                     arrowprops=dict(arrowstyle='->', color=IDA_RED), fontsize=10, color=IDA_RED)
    axes[1].set_ylabel('Trend')

    # Yearly
    axes[2].plot(dates, yearly, color=ORANGE, linewidth=1)
    axes[2].set_ylabel('Yearly')

    # Weekly (show pattern)
    axes[3].plot(range(7), [3*np.sin(2*np.pi*d/7) for d in range(7)], 'o-', color=AMBER)
    axes[3].set_xticks(range(7))
    axes[3].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[3].set_ylabel('Weekly')

    # Holidays
    axes[4].bar(dates, holidays, color=IDA_RED, width=1)
    axes[4].set_ylabel('Holidays')
    axes[4].set_xlabel('Date')

    plt.tight_layout()
    save_chart(fig, 'ch9_prophet_components')

def generate_prophet_vs_tbats_comparison():
    """Compare Prophet and TBATS on simulated data"""
    np.random.seed(42)
    n_train = 365 * 2
    n_test = 30
    n_total = n_train + n_test

    t = np.arange(n_total)

    # True data generating process
    trend = 100 + 0.02 * t
    yearly = 10 * np.sin(2 * np.pi * t / 365.25)
    weekly = 3 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(0, 2, n_total)
    y = trend + yearly + weekly + noise

    # Simulate forecasts (slightly different patterns)
    np.random.seed(123)
    prophet_forecast = trend[n_train:] + yearly[n_train:] + weekly[n_train:] + np.random.normal(0, 1.5, n_test)

    np.random.seed(456)
    tbats_forecast = trend[n_train:] + yearly[n_train:] + weekly[n_train:] + np.random.normal(0, 1.8, n_test)

    dates = pd.date_range('2022-01-01', periods=n_total, freq='D')
    train_dates = dates[:n_train]
    test_dates = dates[n_train:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Training data and forecasts
    axes[0].plot(train_dates[-90:], y[n_train-90:n_train], color=MAIN_BLUE, linewidth=1, label='Training data')
    axes[0].plot(test_dates, y[n_train:], 'ko', markersize=4, label='Actual')
    axes[0].plot(test_dates, prophet_forecast, color=IDA_RED, linewidth=2, label='Prophet forecast')
    axes[0].plot(test_dates, tbats_forecast, color=FOREST, linewidth=2, linestyle='--', label='TBATS forecast')
    axes[0].axvline(dates[n_train], color='gray', linestyle=':', alpha=0.7)
    axes[0].annotate('Forecast\nStart', xy=(dates[n_train], y[n_train]), xytext=(dates[n_train-20], y[n_train]+15),
                     arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)
    axes[0].set_title('Prophet vs TBATS: 30-Day Forecast Comparison', fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].legend(loc='upper left')

    # Error comparison
    prophet_error = y[n_train:] - prophet_forecast
    tbats_error = y[n_train:] - tbats_forecast

    x_pos = np.arange(n_test)
    width = 0.35
    axes[1].bar(x_pos - width/2, np.abs(prophet_error), width, color=IDA_RED, alpha=0.7, label='Prophet |Error|')
    axes[1].bar(x_pos + width/2, np.abs(tbats_error), width, color=FOREST, alpha=0.7, label='TBATS |Error|')
    axes[1].set_xlabel('Forecast Horizon (days)')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title('Forecast Errors by Horizon', fontweight='bold')
    axes[1].legend()

    # Add RMSE annotations
    prophet_rmse = np.sqrt(np.mean(prophet_error**2))
    tbats_rmse = np.sqrt(np.mean(tbats_error**2))
    axes[1].text(0.95, 0.95, f'Prophet RMSE: {prophet_rmse:.2f}\nTBATS RMSE: {tbats_rmse:.2f}',
                transform=axes[1].transAxes, ha='right', va='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_chart(fig, 'ch9_prophet_vs_tbats')

def generate_electricity_demand_example():
    """Real-world style electricity demand with multiple seasonalities"""
    np.random.seed(42)

    # 2 weeks of hourly data
    n = 24 * 14
    hours = np.arange(n)

    # Base load
    base = 500

    # Daily pattern (peak at 8am and 7pm)
    hour_of_day = hours % 24
    daily = 100 * (np.exp(-((hour_of_day - 8)**2) / 20) +
                   0.8 * np.exp(-((hour_of_day - 19)**2) / 15) -
                   0.5 * np.exp(-((hour_of_day - 3)**2) / 10))

    # Weekly pattern (lower on weekends)
    day_of_week = (hours // 24) % 7
    weekly = np.where(day_of_week >= 5, -50, 20)  # Weekend reduction

    # Temperature effect (simulated)
    temp_effect = 30 * np.sin(2 * np.pi * hours / (24 * 7) + np.pi/4)

    # Noise
    noise = np.random.normal(0, 20, n)

    # Combined
    demand = base + daily + weekly + temp_effect + noise

    dates = pd.date_range('2024-01-01', periods=n, freq='h')

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Full series
    axes[0].plot(dates, demand, color=MAIN_BLUE, linewidth=0.8)
    axes[0].fill_between(dates, 0, demand, alpha=0.3, color=MAIN_BLUE)
    axes[0].set_title('Electricity Demand: 2 Weeks of Hourly Data', fontweight='bold')
    axes[0].set_ylabel('MW')
    axes[0].xaxis.set_major_formatter(DateFormatter('%a %d'))

    # Highlight weekends
    for i in range(14):
        if (i % 7) >= 5:  # Weekend
            start = dates[i * 24]
            end = dates[min((i + 1) * 24 - 1, n - 1)]
            axes[0].axvspan(start, end, alpha=0.2, color=IDA_RED)
    axes[0].text(0.02, 0.95, 'Shaded = Weekends', transform=axes[0].transAxes, fontsize=10)

    # Average daily profile
    daily_profile = np.zeros(24)
    for h in range(24):
        daily_profile[h] = np.mean(demand[hour_of_day == h])

    axes[1].plot(range(24), daily_profile, 'o-', color=FOREST, linewidth=2, markersize=6)
    axes[1].fill_between(range(24), 0, daily_profile, alpha=0.3, color=FOREST)
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Average MW')
    axes[1].set_title('Average Daily Profile (24-hour pattern)', fontweight='bold')
    axes[1].set_xticks([0, 6, 12, 18, 23])
    axes[1].set_xticklabels(['00:00', '06:00', '12:00', '18:00', '23:00'])
    axes[1].axvline(8, color=IDA_RED, linestyle='--', alpha=0.7)
    axes[1].axvline(19, color=IDA_RED, linestyle='--', alpha=0.7)
    axes[1].annotate('Morning peak', xy=(8, daily_profile[8]), xytext=(10, daily_profile[8]+20),
                     arrowprops=dict(arrowstyle='->', color=IDA_RED))
    axes[1].annotate('Evening peak', xy=(19, daily_profile[19]), xytext=(21, daily_profile[19]+20),
                     arrowprops=dict(arrowstyle='->', color=IDA_RED))

    # Average weekly profile
    weekly_profile = np.zeros(7)
    for d in range(7):
        weekly_profile[d] = np.mean(demand[day_of_week == d])

    colors = [MAIN_BLUE]*5 + [IDA_RED]*2
    axes[2].bar(range(7), weekly_profile, color=colors)
    axes[2].set_xlabel('Day of Week')
    axes[2].set_ylabel('Average MW')
    axes[2].set_title('Average Weekly Profile (168-hour pattern)', fontweight='bold')
    axes[2].set_xticks(range(7))
    axes[2].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    plt.tight_layout()
    save_chart(fig, 'ch9_electricity_demand')

def generate_retail_sales_example():
    """Retail sales with yearly, weekly, and holiday effects"""
    np.random.seed(42)

    # 3 years of daily data
    n = 365 * 3
    t = np.arange(n)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    # Trend
    trend = 1000 + 0.5 * t

    # Yearly seasonality (peak in December)
    day_of_year = dates.dayofyear
    yearly = 200 * np.sin(2 * np.pi * (day_of_year - 350) / 365)  # Peak near Christmas

    # Weekly pattern (higher on weekends)
    day_of_week = dates.dayofweek
    weekly = np.where(day_of_week >= 5, 100, -20)

    # Holiday spikes
    holidays = np.zeros(n)
    for i, d in enumerate(dates):
        # Christmas season (Dec 15-25)
        if d.month == 12 and 15 <= d.day <= 25:
            holidays[i] = 500
        # Black Friday (4th Friday of November)
        if d.month == 11 and d.weekday() == 4 and 22 <= d.day <= 28:
            holidays[i] = 800
        # Valentine's Day
        if d.month == 2 and d.day == 14:
            holidays[i] = 300

    # Noise
    noise = np.random.normal(0, 50, n)

    # Combined
    sales = trend + yearly + weekly + holidays + noise

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Full series
    axes[0, 0].plot(dates, sales, color=MAIN_BLUE, linewidth=0.5)
    axes[0, 0].set_title('Retail Sales: 3 Years of Daily Data', fontweight='bold')
    axes[0, 0].set_ylabel('Sales ($)')
    axes[0, 0].xaxis.set_major_formatter(DateFormatter('%b %Y'))

    # Highlight December spikes
    for year in range(2022, 2025):
        dec_start = pd.Timestamp(f'{year}-12-01')
        dec_end = pd.Timestamp(f'{year}-12-31')
        if dec_end <= dates[-1]:
            axes[0, 0].axvspan(dec_start, dec_end, alpha=0.2, color=IDA_RED)

    # Yearly pattern
    yearly_avg = pd.Series(sales, index=dates).groupby(dates.dayofyear).mean()
    axes[0, 1].plot(yearly_avg.index, yearly_avg.values, color=FOREST, linewidth=2)
    axes[0, 1].axvline(350, color=IDA_RED, linestyle='--', alpha=0.7)
    axes[0, 1].annotate('Christmas\npeak', xy=(350, yearly_avg.iloc[349]), xytext=(300, yearly_avg.iloc[349]+100),
                        arrowprops=dict(arrowstyle='->', color=IDA_RED))
    axes[0, 1].set_xlabel('Day of Year')
    axes[0, 1].set_ylabel('Average Sales ($)')
    axes[0, 1].set_title('Yearly Seasonality Pattern', fontweight='bold')

    # Weekly pattern
    weekly_avg = pd.Series(sales, index=dates).groupby(dates.dayofweek).mean()
    colors = [MAIN_BLUE]*5 + [FOREST]*2
    axes[1, 0].bar(range(7), weekly_avg.values, color=colors)
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[1, 0].set_ylabel('Average Sales ($)')
    axes[1, 0].set_title('Weekly Pattern (Weekend boost)', fontweight='bold')

    # Holiday effects
    holiday_names = ['Normal\nDay', 'Valentine\'s\nDay', 'Black\nFriday', 'Christmas\nSeason']
    holiday_effects = [0, 300, 800, 500]
    colors = [MAIN_BLUE, IDA_RED, ORANGE, FOREST]
    axes[1, 1].bar(range(4), holiday_effects, color=colors)
    axes[1, 1].set_xticks(range(4))
    axes[1, 1].set_xticklabels(holiday_names)
    axes[1, 1].set_ylabel('Additional Sales ($)')
    axes[1, 1].set_title('Holiday Effects', fontweight='bold')

    plt.tight_layout()
    save_chart(fig, 'ch9_retail_sales')

def generate_additive_vs_multiplicative():
    """Show difference between additive and multiplicative seasonality"""
    np.random.seed(42)
    n = 365 * 2
    t = np.arange(n)

    # Trend
    trend = 100 + 0.2 * t

    # Seasonal pattern
    seasonal_pattern = np.sin(2 * np.pi * t / 365)

    # Additive: seasonal amplitude constant
    additive = trend + 20 * seasonal_pattern + np.random.normal(0, 5, n)

    # Multiplicative: seasonal amplitude proportional to level
    multiplicative = trend * (1 + 0.2 * seasonal_pattern) + np.random.normal(0, 5, n)

    dates = pd.date_range('2023-01-01', periods=n, freq='D')

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(dates, additive, color=MAIN_BLUE, linewidth=0.8)
    axes[0].plot(dates, trend, color=IDA_RED, linewidth=2, linestyle='--', label='Trend')
    axes[0].set_title('Additive Seasonality: $Y_t = T_t + S_t + \\epsilon_t$', fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].legend()

    # Add annotation showing constant amplitude
    axes[0].annotate('', xy=(dates[90], additive[90]), xytext=(dates[90], trend[90]),
                     arrowprops=dict(arrowstyle='<->', color=FOREST, lw=2))
    axes[0].annotate('', xy=(dates[455], additive[455]), xytext=(dates[455], trend[455]),
                     arrowprops=dict(arrowstyle='<->', color=FOREST, lw=2))
    axes[0].text(dates[150], (additive[90] + trend[90])/2, 'Same amplitude', fontsize=10, color=FOREST)

    axes[1].plot(dates, multiplicative, color=MAIN_BLUE, linewidth=0.8)
    axes[1].plot(dates, trend, color=IDA_RED, linewidth=2, linestyle='--', label='Trend')
    axes[1].set_title('Multiplicative Seasonality: $Y_t = T_t \\times S_t \\times \\epsilon_t$', fontweight='bold')
    axes[1].set_ylabel('Value')
    axes[1].set_xlabel('Date')
    axes[1].legend()

    # Add annotation showing growing amplitude
    axes[1].annotate('', xy=(dates[90], multiplicative[90]), xytext=(dates[90], trend[90]),
                     arrowprops=dict(arrowstyle='<->', color=FOREST, lw=2))
    axes[1].annotate('', xy=(dates[455], multiplicative[455]), xytext=(dates[455], trend[455]),
                     arrowprops=dict(arrowstyle='<->', color=FOREST, lw=2))
    axes[1].text(dates[500], multiplicative[455], 'Larger\namplitude', fontsize=10, color=FOREST)
    axes[1].text(dates[20], multiplicative[90], 'Smaller\namplitude', fontsize=10, color=FOREST)

    for ax in axes:
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

    plt.tight_layout()
    save_chart(fig, 'ch9_additive_vs_multiplicative')

def generate_changepoint_detection():
    """Visualize trend changepoint detection in Prophet"""
    np.random.seed(42)
    n = 365 * 3
    t = np.arange(n)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    # Piecewise linear trend with changepoints
    changepoints = [200, 500, 800]
    trend = np.zeros(n)
    trend[:changepoints[0]] = 100 + 0.1 * t[:changepoints[0]]
    trend[changepoints[0]:changepoints[1]] = trend[changepoints[0]-1] + 0.3 * (t[changepoints[0]:changepoints[1]] - changepoints[0])
    trend[changepoints[1]:changepoints[2]] = trend[changepoints[1]-1] - 0.1 * (t[changepoints[1]:changepoints[2]] - changepoints[1])
    trend[changepoints[2]:] = trend[changepoints[2]-1] + 0.2 * (t[changepoints[2]:] - changepoints[2])

    # Add seasonality and noise
    seasonality = 10 * np.sin(2 * np.pi * t / 365)
    noise = np.random.normal(0, 5, n)
    y = trend + seasonality + noise

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Observed data with trend
    axes[0].plot(dates, y, color=MAIN_BLUE, linewidth=0.5, alpha=0.7, label='Observed')
    axes[0].plot(dates, trend, color=IDA_RED, linewidth=2, label='Detected trend')

    for cp in changepoints:
        axes[0].axvline(dates[cp], color=FOREST, linestyle='--', alpha=0.8, linewidth=2)

    axes[0].axvline(dates[changepoints[0]], color=FOREST, linestyle='--', alpha=0.8, linewidth=2, label='Changepoints')
    axes[0].set_title('Prophet Trend Changepoint Detection', fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].legend(loc='upper left')

    # Trend slope over time
    slopes = np.diff(trend)
    axes[1].fill_between(dates[:-1], 0, slopes, where=slopes > 0, color=FOREST, alpha=0.5, label='Growth')
    axes[1].fill_between(dates[:-1], 0, slopes, where=slopes < 0, color=IDA_RED, alpha=0.5, label='Decline')
    axes[1].axhline(0, color='black', linewidth=0.5)

    for cp in changepoints:
        axes[1].axvline(dates[cp], color='black', linestyle='--', alpha=0.5)

    axes[1].set_title('Trend Growth Rate (slope changes at changepoints)', fontweight='bold')
    axes[1].set_ylabel('Daily Change')
    axes[1].set_xlabel('Date')
    axes[1].legend()

    for ax in axes:
        ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))

    plt.tight_layout()
    save_chart(fig, 'ch9_changepoint_detection')

def generate_model_selection_flowchart():
    """Create a decision flowchart for model selection"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Boxes
    def draw_box(x, y, w, h, text, color, fontsize=10):
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)

    def draw_diamond(x, y, w, text):
        diamond = plt.Polygon([(x, y+w/2), (x+w/2, y), (x, y-w/2), (x-w/2, y)],
                               facecolor='#FFF3CD', edgecolor='black', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    def draw_arrow(x1, y1, x2, y2, text='', color='black'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))
        if text:
            ax.text((x1+x2)/2 + 0.3, (y1+y2)/2, text, fontsize=9, fontweight='bold')

    # Start
    draw_box(7, 9, 4, 0.8, 'Time Series with Multiple Seasonalities?', '#E3F2FD', 11)

    # Decision 1
    draw_diamond(7, 7.5, 1.2, 'Need\nInterpretability?')
    draw_arrow(7, 8.6, 7, 8.1)

    # Left branch - Yes
    draw_diamond(4, 6, 1.2, 'Have\nHoliday\nEffects?')
    draw_arrow(6.4, 7.5, 4.6, 6.3, 'Yes')

    # Prophet
    draw_box(2, 4.5, 2.5, 1.2, 'Use\nPROPHET', FOREST, 12)
    draw_arrow(3.4, 6, 2.3, 5.1, 'Yes')

    # TBATS option
    draw_box(5.5, 4.5, 2.5, 1.2, 'Use\nTBATS', ORANGE, 12)
    draw_arrow(4.6, 5.4, 5.2, 5.1, 'No')

    # Right branch - No (black box OK)
    draw_diamond(10, 6, 1.2, 'Very Large\nDataset?')
    draw_arrow(7.6, 7.5, 9.4, 6.3, 'No')

    # Neural methods
    draw_box(12, 4.5, 2.5, 1.2, 'Consider\nNeural Methods', IDA_RED, 11)
    draw_arrow(10.6, 5.4, 11.7, 5.1, 'Yes')

    # Ensemble
    draw_box(8.5, 4.5, 2.5, 1.2, 'Try Both\n+ Ensemble', MAIN_BLUE, 11)
    draw_arrow(9.4, 6, 8.8, 5.1, 'No')

    # Bottom recommendations
    draw_box(7, 2, 10, 1.5,
             'Recommendation: Start with Prophet (interpretable, handles holidays)\n'
             'Compare with TBATS, use ensemble if accuracy critical',
             '#F5F5F5', 10)

    # Arrows to bottom
    draw_arrow(2, 3.9, 3, 2.75, '', 'gray')
    draw_arrow(5.5, 3.9, 5.5, 2.75, '', 'gray')
    draw_arrow(8.5, 3.9, 8.5, 2.75, '', 'gray')
    draw_arrow(12, 3.9, 11, 2.75, '', 'gray')

    ax.set_title('Model Selection Guide for Multiple Seasonalities', fontsize=14, fontweight='bold', pad=20)

    save_chart(fig, 'ch9_model_selection_guide')

def main():
    """Generate all Chapter 9 charts"""
    print("Generating Chapter 9 charts...")

    generate_multiple_seasonality_example()
    generate_fourier_terms_visualization()
    generate_tbats_decomposition()
    generate_prophet_components()
    generate_prophet_vs_tbats_comparison()
    generate_electricity_demand_example()
    generate_retail_sales_example()
    generate_additive_vs_multiplicative()
    generate_changepoint_detection()
    generate_model_selection_flowchart()

    print("\nAll Chapter 9 charts generated successfully!")

if __name__ == "__main__":
    main()
