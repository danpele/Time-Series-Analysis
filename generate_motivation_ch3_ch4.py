#!/usr/bin/env python3
"""
Generate motivation charts for Chapter 3 (ARIMA) and Chapter 4 (SARIMA)
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Color palette
MAIN_BLUE = '#1a3a6e'
ACCENT_BLUE = '#2a528c'
CRIMSON = '#dc3545'
FOREST = '#2e7d32'
AMBER = '#b5853f'
ORANGE = '#e67e22'

output_dir = '/Users/danielpele/Documents/Time Series Analysis/charts/'

def save_fig(fig, name):
    """Save figure in both PDF and PNG formats"""
    fig.savefig(f'{output_dir}{name}.pdf', format='pdf', bbox_inches='tight', dpi=150)
    fig.savefig(f'{output_dir}{name}.png', format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {name}')

# =============================================================================
# CHAPTER 3 MOTIVATION CHARTS
# =============================================================================

def create_ch3_motivation_nonstationary():
    """Show examples of non-stationary series"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    np.random.seed(42)
    T = 200
    time = np.arange(T)

    # Random walk (stock price-like)
    ax1 = axes[0, 0]
    rw = 100 + np.cumsum(np.random.normal(0.05, 2, T))
    ax1.plot(time, rw, color=MAIN_BLUE, linewidth=1.5)
    ax1.set_title('Stock Price (Random Walk)', fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.axhline(y=np.mean(rw), color=CRIMSON, linestyle='--', label='Sample Mean')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    # Trending series (GDP-like)
    ax2 = axes[0, 1]
    trend = 100 * np.exp(0.01 * time) + np.cumsum(np.random.normal(0, 0.5, T))
    ax2.plot(time, trend, color=FOREST, linewidth=1.5)
    ax2.set_title('GDP (Exponential Trend)', fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Index')

    # Random walk with drift
    ax3 = axes[1, 0]
    rw_drift = 50 + np.cumsum(np.random.normal(0.2, 1.5, T))
    ax3.plot(time, rw_drift, color=ORANGE, linewidth=1.5)
    ax3.set_title('Random Walk with Drift', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')

    # Variance changing over time
    ax4 = axes[1, 1]
    var_change = np.zeros(T)
    for t in range(T):
        var_change[t] = np.random.normal(0, 0.5 + 0.02*t)
    var_change = np.cumsum(var_change)
    ax4.plot(time, var_change, color=CRIMSON, linewidth=1.5)
    ax4.set_title('Increasing Variance Over Time', fontweight='bold')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Value')

    fig.suptitle('Examples of Non-Stationary Time Series', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'ch3_motivation_nonstationary')

def create_ch3_motivation_differencing():
    """Show the effect of differencing"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    np.random.seed(123)
    T = 150
    time = np.arange(T)

    # Original random walk
    rw = 100 + np.cumsum(np.random.normal(0.1, 2, T))

    # First difference
    diff1 = np.diff(rw)

    # Original series
    ax1 = axes[0, 0]
    ax1.plot(time, rw, color=MAIN_BLUE, linewidth=1.5)
    ax1.set_title('Original Series: $Y_t$ (Non-Stationary)', fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.axhline(y=np.mean(rw), color=CRIMSON, linestyle='--', alpha=0.7)

    # ACF of original
    ax2 = axes[0, 1]
    from numpy.fft import fft, ifft
    def acf(x, nlags=30):
        n = len(x)
        x = x - np.mean(x)
        result = np.correlate(x, x, mode='full')
        result = result[n-1:n+nlags] / result[n-1]
        return result

    acf_orig = acf(rw, 30)
    ax2.bar(range(len(acf_orig)), acf_orig, color=MAIN_BLUE, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=1.96/np.sqrt(T), color=CRIMSON, linestyle='--')
    ax2.axhline(y=-1.96/np.sqrt(T), color=CRIMSON, linestyle='--')
    ax2.set_title('ACF of $Y_t$: Slow Decay (Unit Root)', fontweight='bold')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('ACF')

    # Differenced series
    ax3 = axes[1, 0]
    ax3.plot(time[1:], diff1, color=FOREST, linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.axhline(y=np.mean(diff1), color=CRIMSON, linestyle='--', alpha=0.7)
    ax3.set_title('Differenced: $\\Delta Y_t = Y_t - Y_{t-1}$ (Stationary)', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')

    # ACF of differenced
    ax4 = axes[1, 1]
    acf_diff = acf(diff1, 30)
    ax4.bar(range(len(acf_diff)), acf_diff, color=FOREST, alpha=0.7)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.axhline(y=1.96/np.sqrt(T), color=CRIMSON, linestyle='--')
    ax4.axhline(y=-1.96/np.sqrt(T), color=CRIMSON, linestyle='--')
    ax4.set_title('ACF of $\\Delta Y_t$: Quick Decay (Stationary)', fontweight='bold')
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('ACF')

    fig.suptitle('The Magic of Differencing: Converting Non-Stationary to Stationary',
                fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'ch3_motivation_differencing')

def create_ch3_motivation_realworld():
    """Show real-world applications"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    np.random.seed(456)
    T = 120
    time = np.arange(T)

    # Stock price
    ax1 = axes[0]
    stock = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, T)))
    ax1.plot(time, stock, color=MAIN_BLUE, linewidth=1.5)
    ax1.fill_between(time, stock, alpha=0.3, color=MAIN_BLUE)
    ax1.set_title('Stock Price\n(Geometric Random Walk)', fontweight='bold')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Price ($)')
    ax1.text(0.05, 0.95, 'I(1) in logs', transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Exchange rate
    ax2 = axes[1]
    fx = 1.1 + np.cumsum(np.random.normal(0, 0.005, T))
    ax2.plot(time, fx, color=FOREST, linewidth=1.5)
    ax2.fill_between(time, fx, alpha=0.3, color=FOREST)
    ax2.set_title('Exchange Rate\n(EUR/USD)', fontweight='bold')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Rate')
    ax2.text(0.05, 0.95, 'Random Walk', transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Interest rate
    ax3 = axes[2]
    ir = 3 + np.cumsum(np.random.normal(0, 0.1, T)) * 0.1
    ir = np.clip(ir, 0.5, 8)
    ax3.plot(time, ir, color=CRIMSON, linewidth=1.5)
    ax3.fill_between(time, ir, alpha=0.3, color=CRIMSON)
    ax3.set_title('Interest Rate\n(Central Bank Policy)', fontweight='bold')
    ax3.set_xlabel('Months')
    ax3.set_ylabel('Rate (%)')
    ax3.text(0.05, 0.95, 'Near Unit Root', transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.suptitle('Real-World Non-Stationary Series: Why We Need ARIMA',
                fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'ch3_motivation_realworld')

# =============================================================================
# CHAPTER 4 MOTIVATION CHARTS
# =============================================================================

def create_ch4_motivation_seasonal():
    """Show clear seasonal patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    np.random.seed(42)
    T = 72  # 6 years of monthly data
    time = np.arange(T)
    months = time % 12

    # Airline passengers (classic example)
    ax1 = axes[0, 0]
    trend = 100 + 2 * time
    seasonal = 30 * np.sin(2 * np.pi * time / 12) + 15 * np.cos(4 * np.pi * time / 12)
    passengers = trend + seasonal * (1 + 0.01 * time) + np.random.normal(0, 5, T)
    ax1.plot(time, passengers, color=MAIN_BLUE, linewidth=1.5)
    ax1.set_title('Airline Passengers (Monthly)', fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Thousands')
    for year in range(0, T, 12):
        ax1.axvline(x=year, color='gray', linestyle=':', alpha=0.5)

    # Retail sales
    ax2 = axes[0, 1]
    retail_trend = 500 + 3 * time
    retail_seasonal = np.zeros(T)
    for t in range(T):
        m = t % 12
        if m == 11:  # December spike
            retail_seasonal[t] = 150
        elif m == 10:  # November
            retail_seasonal[t] = 50
        elif m in [5, 6]:  # Summer
            retail_seasonal[t] = 30
        else:
            retail_seasonal[t] = 0
    retail = retail_trend + retail_seasonal + np.random.normal(0, 20, T)
    ax2.plot(time, retail, color=CRIMSON, linewidth=1.5)
    ax2.set_title('Retail Sales (Monthly)', fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('$ Millions')
    for year in range(0, T, 12):
        ax2.axvline(x=year, color='gray', linestyle=':', alpha=0.5)

    # Energy consumption
    ax3 = axes[1, 0]
    energy_trend = 1000 + 1 * time
    energy_seasonal = 200 * np.cos(2 * np.pi * time / 12)  # Peak in winter
    energy = energy_trend + energy_seasonal + np.random.normal(0, 30, T)
    ax3.plot(time, energy, color=FOREST, linewidth=1.5)
    ax3.set_title('Energy Consumption (Monthly)', fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('GWh')
    for year in range(0, T, 12):
        ax3.axvline(x=year, color='gray', linestyle=':', alpha=0.5)

    # Ice cream sales
    ax4 = axes[1, 1]
    ice_trend = 50 + 0.5 * time
    ice_seasonal = 40 * np.sin(2 * np.pi * (time - 3) / 12)  # Peak in summer
    ice_seasonal = np.maximum(ice_seasonal, -20)
    ice = ice_trend + ice_seasonal + np.random.normal(0, 5, T)
    ax4.plot(time, ice, color=ORANGE, linewidth=1.5)
    ax4.set_title('Ice Cream Sales (Monthly)', fontweight='bold')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Units (000s)')
    for year in range(0, T, 12):
        ax4.axvline(x=year, color='gray', linestyle=':', alpha=0.5)

    fig.suptitle('Seasonal Patterns Are Everywhere!', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'ch4_motivation_seasonal')

def create_ch4_motivation_decomposition():
    """Show seasonal decomposition"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    np.random.seed(123)
    T = 72
    time = np.arange(T)

    # Create components
    trend = 100 + 1.5 * time + 0.01 * time**2
    seasonal = 25 * np.sin(2 * np.pi * time / 12)
    residual = np.random.normal(0, 5, T)
    observed = trend + seasonal + residual

    # Observed
    axes[0].plot(time, observed, color=MAIN_BLUE, linewidth=1.5)
    axes[0].set_ylabel('Observed')
    axes[0].set_title('Decomposition: Observed = Trend + Seasonal + Residual', fontweight='bold')

    # Trend
    axes[1].plot(time, trend, color=FOREST, linewidth=2)
    axes[1].set_ylabel('Trend')

    # Seasonal
    axes[2].plot(time, seasonal, color=ORANGE, linewidth=1.5)
    axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[2].set_ylabel('Seasonal')

    # Residual
    axes[3].plot(time, residual, color=CRIMSON, linewidth=1)
    axes[3].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Month')

    for ax in axes:
        for year in range(0, T, 12):
            ax.axvline(x=year, color='gray', linestyle=':', alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'ch4_motivation_decomposition')

def create_ch4_motivation_monthly_pattern():
    """Show monthly seasonal pattern"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    np.random.seed(456)

    # Monthly boxplot-style visualization
    ax1 = axes[0]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Simulated seasonal factors
    seasonal_factors = [0.85, 0.82, 0.95, 1.00, 1.05, 1.15,
                       1.20, 1.18, 1.05, 0.95, 0.90, 1.10]

    colors = [MAIN_BLUE if sf < 1 else FOREST for sf in seasonal_factors]
    bars = ax1.bar(months, seasonal_factors, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=1, color=CRIMSON, linestyle='--', linewidth=2, label='Baseline')
    ax1.set_ylabel('Seasonal Factor')
    ax1.set_title('Monthly Seasonal Pattern (Tourism)', fontweight='bold')
    ax1.set_ylim(0.7, 1.3)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    # Year-over-year comparison
    ax2 = axes[1]
    years = 4
    T = 12 * years
    time = np.arange(T)

    base_pattern = np.array([0.85, 0.82, 0.95, 1.00, 1.05, 1.15,
                            1.20, 1.18, 1.05, 0.95, 0.90, 1.10])

    for year in range(years):
        year_data = 100 * (1.1 ** year) * base_pattern + np.random.normal(0, 3, 12)
        ax2.plot(range(12), year_data, 'o-', linewidth=1.5,
                label=f'Year {year+1}', alpha=0.8)

    ax2.set_xticks(range(12))
    ax2.set_xticklabels(months, rotation=45)
    ax2.set_ylabel('Value')
    ax2.set_title('Same Pattern Repeats Each Year', fontweight='bold')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False)

    fig.suptitle('Understanding Seasonal Patterns', fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'ch4_motivation_monthly')

def create_ch4_motivation_why_sarima():
    """Show why we need SARIMA"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    np.random.seed(789)
    T = 60
    time = np.arange(T)

    # Original seasonal data
    ax1 = axes[0]
    trend = 100 + 0.8 * time
    seasonal = 20 * np.sin(2 * np.pi * time / 12)
    y = trend + seasonal + np.random.normal(0, 3, T)
    ax1.plot(time, y, color=MAIN_BLUE, linewidth=1.5)
    ax1.set_title('Original Series\n(Trend + Seasonality)', fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Value')
    ax1.text(0.5, 0.95, 'Non-stationary:\nARMA fails!', transform=ax1.transAxes,
            fontsize=9, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))

    # After regular differencing
    ax2 = axes[1]
    diff1 = np.diff(y)
    ax2.plot(time[1:], diff1, color=ORANGE, linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.set_title('After $\\Delta Y_t$\n(Regular Differencing)', fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('$\\Delta Y_t$')
    ax2.text(0.5, 0.95, 'Still seasonal:\nARIMA fails!', transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8))

    # After seasonal differencing
    ax3 = axes[2]
    # Apply seasonal difference to original
    diff12 = y[12:] - y[:-12]
    # Then regular difference
    diff_both = np.diff(diff12)
    ax3.plot(range(len(diff_both)), diff_both, color=FOREST, linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_title('After $\\Delta\\Delta_{12} Y_t$\n(Both Differencings)', fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('$\\Delta\\Delta_{12} Y_t$')
    ax3.text(0.5, 0.95, 'Stationary:\nSARIMA works!', transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='#ccffcc', alpha=0.8))

    fig.suptitle('Why ARIMA Is Not Enough for Seasonal Data',
                fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'ch4_motivation_why_sarima')

# =============================================================================
# Main execution
# =============================================================================
if __name__ == '__main__':
    print('Generating Chapter 3 motivation charts...')
    create_ch3_motivation_nonstationary()
    create_ch3_motivation_differencing()
    create_ch3_motivation_realworld()

    print('\nGenerating Chapter 4 motivation charts...')
    create_ch4_motivation_seasonal()
    create_ch4_motivation_decomposition()
    create_ch4_motivation_monthly_pattern()
    create_ch4_motivation_why_sarima()

    print('\nAll motivation charts generated successfully!')
