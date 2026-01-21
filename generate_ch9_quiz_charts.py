#!/usr/bin/env python3
"""
Generate charts for Chapter 9 lecture quiz answers
Prophet and TBATS for Multiple Seasonalities
Time Series Analysis Course
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Set style for transparent backgrounds
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Output directory
output_dir = 'charts'
os.makedirs(output_dir, exist_ok=True)

def save_fig(name):
    plt.savefig(f'{output_dir}/{name}.pdf', format='pdf', bbox_inches='tight',
                transparent=True, dpi=150)
    plt.close()
    print(f"  Created {name}.pdf")

# =============================================================================
# CHAPTER 9: Prophet and TBATS Quiz Charts
# =============================================================================
print("Chapter 9: Prophet & TBATS Quiz Charts")


# Quiz 1: Multiple Seasonality Problem
def ch9_quiz1_multiple_seasonality():
    """Illustrate why SARIMA cannot handle multiple seasonalities"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    np.random.seed(42)
    n = 168 * 2  # Two weeks of hourly data

    t = np.arange(n)

    # Daily seasonality (period = 24)
    daily = 5 * np.sin(2 * np.pi * t / 24)

    # Weekly seasonality (period = 168)
    weekly = 3 * np.sin(2 * np.pi * t / 168)

    # Combined
    combined = daily + weekly + np.random.normal(0, 1, n)

    axes[0, 0].plot(t, daily, 'b-', linewidth=1)
    axes[0, 0].set_title('Daily Seasonality (s=24)', fontsize=10, color='blue')
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Value')

    axes[0, 1].plot(t, weekly, 'g-', linewidth=1)
    axes[0, 1].set_title('Weekly Seasonality (s=168)', fontsize=10, color='green')
    axes[0, 1].set_xlabel('Hour')

    axes[1, 0].plot(t, combined, 'purple', linewidth=0.8, alpha=0.8)
    axes[1, 0].set_title('Combined: Daily + Weekly + Noise', fontsize=10, color='purple')
    axes[1, 0].set_xlabel('Hour')
    axes[1, 0].set_ylabel('Value')

    # Show the problem
    axes[1, 1].axis('off')
    problem_text = """
    SARIMA Limitation:

    SARIMA$(p,d,q)(P,D,Q)_s$ handles
    only ONE seasonal period $s$.

    For hourly data with:
    • Daily pattern: s = 24
    • Weekly pattern: s = 168

    SARIMA cannot model both!

    Solutions:
    • TBATS
    • Prophet
    • Fourier terms + ARIMA
    """
    axes[1, 1].text(0.1, 0.9, problem_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    save_fig('ch9_quiz1_multiple_seasonality')

ch9_quiz1_multiple_seasonality()


# Quiz 2: TBATS acronym
def ch9_quiz2_tbats_components():
    """Visualize TBATS components"""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')

    # TBATS acronym explanation
    components = [
        ('T', 'Trigonometric', 'Fourier terms for seasonality\n$\\sum [a_n \\cos(\\frac{2\\pi nt}{m}) + b_n \\sin(\\frac{2\\pi nt}{m})]$', '#1f77b4'),
        ('B', 'Box-Cox', 'Variance stabilization\n$y^{(\\omega)} = (y^\\omega - 1)/\\omega$', '#ff7f0e'),
        ('A', 'ARMA', 'Error autocorrelation\n$\\phi(L)d_t = \\theta(L)\\varepsilon_t$', '#2ca02c'),
        ('T', 'Trend', 'Level + slope (possibly damped)\n$\\ell_t = \\ell_{t-1} + \\phi b_{t-1}$', '#d62728'),
        ('S', 'Seasonal', 'Multiple seasonal periods\n$m_1, m_2, ..., m_T$', '#9467bd')
    ]

    y_positions = [0.85, 0.68, 0.51, 0.34, 0.17]

    for (letter, name, desc, color), y in zip(components, y_positions):
        # Letter box
        ax.text(0.08, y, letter, fontsize=28, fontweight='bold', color='white',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='black'))

        # Name
        ax.text(0.18, y, name, fontsize=14, fontweight='bold', color=color,
                ha='left', va='center')

        # Description
        ax.text(0.38, y, desc, fontsize=10, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=color))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('TBATS: What Does It Stand For?', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_fig('ch9_quiz2_tbats_components')

ch9_quiz2_tbats_components()


# Quiz 3: Fourier terms / harmonics
def ch9_quiz3_fourier_harmonics():
    """Show effect of number of harmonics on seasonal pattern"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    t = np.linspace(0, 2, 200)  # Two periods

    # Target: complex seasonal pattern
    target = 3 * np.sin(2 * np.pi * t) + 1.5 * np.sin(4 * np.pi * t) + 0.8 * np.sin(6 * np.pi * t)

    harmonics = [1, 2, 3, 5]
    titles = ['K=1 (Too smooth)', 'K=2 (Better)', 'K=3 (Good fit)', 'K=5 (Risk of overfitting)']

    for ax, K, title in zip(axes.flat, harmonics, titles):
        approx = np.zeros_like(t)
        for k in range(1, K + 1):
            if k == 1:
                approx += 3 * np.sin(2 * np.pi * k * t)
            elif k == 2:
                approx += 1.5 * np.sin(2 * np.pi * k * t)
            elif k == 3:
                approx += 0.8 * np.sin(2 * np.pi * k * t)
            else:
                approx += 0.3 * np.sin(2 * np.pi * k * t)  # Small higher harmonics

        ax.plot(t, target, 'b--', linewidth=2, alpha=0.5, label='True pattern')
        ax.plot(t, approx, 'r-', linewidth=2, label=f'Fourier K={K}')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Period')
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    save_fig('ch9_quiz3_fourier_harmonics')

ch9_quiz3_fourier_harmonics()


# Quiz 4: Prophet decomposition
def ch9_quiz4_prophet_decomposition():
    """Illustrate Prophet's additive decomposition"""
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))

    np.random.seed(42)
    n = 365 * 2  # 2 years of daily data
    t = np.arange(n)

    # Trend with changepoint
    trend = np.zeros(n)
    trend[:200] = 0.05 * t[:200]
    trend[200:] = trend[199] + 0.02 * (t[200:] - 200)  # Slope change

    # Weekly seasonality
    weekly = 2 * np.sin(2 * np.pi * t / 7)

    # Yearly seasonality
    yearly = 5 * np.sin(2 * np.pi * t / 365)

    # Combined
    noise = np.random.normal(0, 1, n)
    y = trend + weekly + yearly + noise

    # Plot each component
    axes[0].plot(t, y, 'gray', linewidth=0.5, alpha=0.7)
    axes[0].set_title('$y(t) = g(t) + s(t) + h(t) + \\varepsilon_t$', fontsize=11)
    axes[0].set_ylabel('y(t)')

    axes[1].plot(t, trend, 'b-', linewidth=1.5)
    axes[1].axvline(x=200, color='red', linestyle='--', alpha=0.7, label='Changepoint')
    axes[1].set_title('g(t): Trend with Changepoint', fontsize=10, color='blue')
    axes[1].set_ylabel('g(t)')
    axes[1].legend(loc='upper left', fontsize=8)

    axes[2].plot(t[:50], weekly[:50], 'g-', linewidth=1.5)
    axes[2].set_title('s(t): Weekly Seasonality (period=7)', fontsize=10, color='green')
    axes[2].set_ylabel('s(t)')
    axes[2].set_xlabel('Day')

    axes[3].plot(t, yearly, 'orange', linewidth=1.5)
    axes[3].set_title('s(t): Yearly Seasonality (period=365)', fontsize=10, color='orange')
    axes[3].set_ylabel('s(t)')
    axes[3].set_xlabel('Day')

    plt.tight_layout()
    save_fig('ch9_quiz4_prophet_decomposition')

ch9_quiz4_prophet_decomposition()


# Quiz 5: Prophet vs TBATS comparison
def ch9_quiz5_prophet_vs_tbats():
    """Comparison table visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')

    # Comparison data
    table_data = [
        ['Feature', 'TBATS', 'Prophet'],
        ['Multiple seasonalities', 'Yes (automatic)', 'Yes (manual/auto)'],
        ['Holiday effects', 'No', 'Yes (built-in)'],
        ['External regressors', 'No', 'Yes'],
        ['Trend changepoints', 'No (smooth)', 'Yes (automatic)'],
        ['Missing data', 'Needs interpolation', 'Handles natively'],
        ['Interpretability', 'Moderate', 'High'],
        ['Computation speed', 'Slow', 'Fast'],
        ['High-frequency data', 'Good', 'Moderate'],
        ['Non-integer periods', 'Yes (e.g., 365.25)', 'Yes'],
        ['Best for', 'Technical/high-freq', 'Business/daily']
    ]

    # Create table
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     colWidths=[0.35, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Style header row
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Highlight key differences
    highlight_rows = [2, 3, 4, 5]  # Holiday, regressors, changepoints, missing data
    for row in highlight_rows:
        table[(row, 1)].set_facecolor('#FFE6E6')  # TBATS weakness
        table[(row, 2)].set_facecolor('#E6FFE6')  # Prophet strength

    ax.set_title('TBATS vs Prophet: Head-to-Head Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_fig('ch9_quiz5_prophet_vs_tbats')

ch9_quiz5_prophet_vs_tbats()


# Quiz 6: Seasonality mode (additive vs multiplicative)
def ch9_quiz6_seasonality_mode():
    """Show difference between additive and multiplicative seasonality"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    np.random.seed(42)
    t = np.arange(120)

    # Trend
    trend = 10 + 0.2 * t

    # Additive seasonality
    seasonal_add = 3 * np.sin(2 * np.pi * t / 12)
    y_add = trend + seasonal_add + np.random.normal(0, 0.5, 120)

    axes[0].plot(t, y_add, 'b-', linewidth=1, alpha=0.8)
    axes[0].plot(t, trend, 'r--', linewidth=2, alpha=0.7, label='Trend')
    axes[0].fill_between(t, trend - 3, trend + 3, alpha=0.2, color='blue', label='Constant amplitude')
    axes[0].set_title('Additive: $Y = T + S + \\varepsilon$\nConstant Seasonal Amplitude', fontsize=10, color='blue')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend(loc='upper left', fontsize=8)

    # Multiplicative seasonality
    seasonal_mult = 1 + 0.2 * np.sin(2 * np.pi * t / 12)  # +-20%
    y_mult = trend * seasonal_mult * (1 + np.random.normal(0, 0.03, 120))

    axes[1].plot(t, y_mult, 'g-', linewidth=1, alpha=0.8)
    axes[1].plot(t, trend, 'r--', linewidth=2, alpha=0.7, label='Trend')
    axes[1].fill_between(t, trend * 0.8, trend * 1.2, alpha=0.2, color='green', label='Growing amplitude')
    axes[1].set_title('Multiplicative: $Y = T \\times S \\times \\varepsilon$\nSeasonal Amplitude Grows', fontsize=10, color='green')
    axes[1].set_xlabel('Time')
    axes[1].legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    save_fig('ch9_quiz6_seasonality_mode')

ch9_quiz6_seasonality_mode()


# Quiz 7: Prophet changepoints
def ch9_quiz7_changepoints():
    """Illustrate Prophet's automatic changepoint detection"""
    fig, ax = plt.subplots(figsize=(10, 5))

    np.random.seed(42)
    n = 200
    t = np.arange(n)

    # Trend with multiple changepoints
    trend = np.zeros(n)
    changepoints = [50, 100, 150]
    slopes = [0.1, -0.05, 0.15, 0.02]

    current_slope = slopes[0]
    current_level = 10

    for i in range(n):
        if i in changepoints:
            cp_idx = changepoints.index(i)
            current_slope = slopes[cp_idx + 1]

        trend[i] = current_level
        current_level += current_slope

    # Add noise
    y = trend + np.random.normal(0, 1.5, n)

    ax.plot(t, y, 'gray', linewidth=0.8, alpha=0.6, label='Observed')
    ax.plot(t, trend, 'b-', linewidth=2.5, label='Trend')

    # Mark changepoints
    for cp in changepoints:
        ax.axvline(x=cp, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.scatter([cp], [trend[cp]], s=100, c='red', zorder=5)

    ax.scatter([], [], s=100, c='red', label='Changepoints')

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Prophet: Automatic Changepoint Detection\n$g(t) = (k + \\mathbf{a}(t)^T \\delta) \\cdot t + (m + \\mathbf{a}(t)^T \\gamma)$',
                 fontsize=11)
    ax.legend(loc='upper left', fontsize=9)

    # Add annotations for slope changes
    ax.annotate('Slope = 0.1', xy=(25, trend[25]+5), fontsize=9, color='blue')
    ax.annotate('Slope = -0.05', xy=(75, trend[75]+5), fontsize=9, color='blue')
    ax.annotate('Slope = 0.15', xy=(125, trend[125]+5), fontsize=9, color='blue')
    ax.annotate('Slope = 0.02', xy=(175, trend[175]+5), fontsize=9, color='blue')

    plt.tight_layout()
    save_fig('ch9_quiz7_changepoints')

ch9_quiz7_changepoints()


# Quiz 8: When to use which model
def ch9_quiz8_model_decision():
    """Decision flowchart for model selection"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')

    # Simple flowchart using text and boxes
    ax.text(0.5, 0.95, 'Multiple Seasonal Periods?', ha='center', va='center', fontsize=12,
            fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black'))

    # No branch -> SARIMA
    ax.annotate('No', xy=(0.25, 0.88), xytext=(0.25, 0.85), fontsize=10, ha='center')
    ax.text(0.15, 0.75, 'SARIMA', ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='green'))

    # Yes branch
    ax.annotate('Yes', xy=(0.75, 0.88), xytext=(0.75, 0.85), fontsize=10, ha='center')
    ax.text(0.75, 0.75, 'Holiday Effects\nImportant?', ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='black'))

    # No holidays -> TBATS
    ax.annotate('No', xy=(0.55, 0.68), xytext=(0.45, 0.60), fontsize=10, ha='center')
    ax.text(0.35, 0.50, 'TBATS', ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='green'))
    ax.text(0.35, 0.40, '• High-frequency data\n• Automatic selection\n• No external regressors',
            ha='center', va='top', fontsize=9)

    # Yes holidays -> Prophet
    ax.annotate('Yes', xy=(0.85, 0.68), xytext=(0.85, 0.55), fontsize=10, ha='center')
    ax.text(0.75, 0.50, 'External\nRegressors?', ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='black'))

    ax.annotate('No', xy=(0.65, 0.43), xytext=(0.55, 0.35), fontsize=10, ha='center')
    ax.text(0.50, 0.25, 'Prophet', ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='green'))

    ax.annotate('Yes', xy=(0.85, 0.43), xytext=(0.85, 0.35), fontsize=10, ha='center')
    ax.text(0.80, 0.25, 'Prophet +\nRegressors', ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='green'))

    # Draw arrows
    from matplotlib.patches import FancyArrowPatch
    arrows = [
        ((0.5, 0.89), (0.25, 0.80)),   # Main to SARIMA
        ((0.5, 0.89), (0.75, 0.80)),   # Main to Holidays?
        ((0.65, 0.70), (0.45, 0.55)),  # Holidays No to TBATS
        ((0.75, 0.70), (0.75, 0.55)),  # Holidays Yes to Regressors?
        ((0.65, 0.45), (0.55, 0.30)),  # Regressors No to Prophet
        ((0.85, 0.45), (0.85, 0.30)),  # Regressors Yes to Prophet+
    ]

    for start, end in arrows:
        arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15,
                                color='gray', linewidth=1.5)
        ax.add_patch(arrow)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Model Selection: Decision Flowchart', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_fig('ch9_quiz8_model_decision')

ch9_quiz8_model_decision()


# Quiz 9: Uncertainty in Prophet forecasts
def ch9_quiz9_prophet_uncertainty():
    """Show Prophet's uncertainty intervals"""
    fig, ax = plt.subplots(figsize=(10, 5))

    np.random.seed(42)

    # Historical data
    n_hist = 100
    t_hist = np.arange(n_hist)
    trend = 50 + 0.2 * t_hist
    seasonal = 5 * np.sin(2 * np.pi * t_hist / 30)
    y_hist = trend + seasonal + np.random.normal(0, 2, n_hist)

    # Forecast
    n_forecast = 30
    t_forecast = np.arange(n_hist, n_hist + n_forecast)
    trend_fc = 50 + 0.2 * t_forecast
    seasonal_fc = 5 * np.sin(2 * np.pi * t_forecast / 30)
    y_forecast = trend_fc + seasonal_fc

    # Uncertainty grows with horizon
    uncertainty = np.linspace(2, 8, n_forecast)

    ax.plot(t_hist, y_hist, 'b-', linewidth=1, label='Historical Data')
    ax.plot(t_forecast, y_forecast, 'r-', linewidth=2, label='Forecast (yhat)')
    ax.fill_between(t_forecast, y_forecast - 1.96*uncertainty, y_forecast + 1.96*uncertainty,
                    alpha=0.3, color='red', label='95% Interval')
    ax.fill_between(t_forecast, y_forecast - 1.28*uncertainty, y_forecast + 1.28*uncertainty,
                    alpha=0.3, color='red')

    ax.axvline(x=n_hist, color='black', linestyle='--', alpha=0.5, label='Forecast Start')

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Prophet: Uncertainty Grows with Forecast Horizon\n(Trend + Seasonality + Observation Uncertainty)',
                 fontsize=11)
    ax.legend(loc='upper left', fontsize=9)

    # Annotate uncertainty sources
    ax.annotate('Uncertainty grows\nwith horizon', xy=(n_hist + 25, y_forecast[-5] + 1.96*uncertainty[-5]),
                xytext=(n_hist + 15, y_forecast[-5] + 2.5*uncertainty[-5]),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')

    plt.tight_layout()
    save_fig('ch9_quiz9_prophet_uncertainty')

ch9_quiz9_prophet_uncertainty()


# Quiz 10: Real-world application example
def ch9_quiz10_energy_example():
    """Example: Energy demand with multiple seasonalities"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    np.random.seed(42)

    # Simulate hourly energy demand for 2 weeks
    n = 24 * 14  # 2 weeks hourly
    t = np.arange(n)
    hours = t % 24
    days = t // 24

    # Daily pattern (peak at noon and evening)
    daily = 20 * np.exp(-((hours - 12) ** 2) / 20) + 15 * np.exp(-((hours - 19) ** 2) / 10)

    # Weekly pattern (lower on weekends)
    weekly = np.where(days % 7 < 5, 0, -10)

    # Base load
    base = 100

    # Combined
    demand = base + daily + weekly + np.random.normal(0, 3, n)

    # Plot components
    axes[0, 0].plot(t[:48], demand[:48], 'b-', linewidth=1)
    axes[0, 0].set_title('First 2 Days: Daily Pattern Visible', fontsize=10)
    axes[0, 0].set_xlabel('Hour')
    axes[0, 0].set_ylabel('Demand (MW)')

    axes[0, 1].plot(t, demand, 'b-', linewidth=0.5, alpha=0.7)
    axes[0, 1].set_title('Full 2 Weeks: Weekly Pattern Visible', fontsize=10)
    axes[0, 1].set_xlabel('Hour')

    # Model comparison (simulated results)
    models = ['SARIMA\n(daily only)', 'TBATS', 'Prophet', 'Prophet +\nHolidays']
    mape = [8.5, 4.2, 4.8, 3.9]
    colors = ['red', 'blue', 'green', 'darkgreen']

    bars = axes[1, 0].bar(models, mape, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].set_title('Model Comparison: Energy Demand', fontsize=10)
    axes[1, 0].axhline(y=5, color='gray', linestyle='--', alpha=0.5)

    for bar, m in zip(bars, mape):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{m}%', ha='center', fontsize=9)

    # Key takeaway
    axes[1, 1].axis('off')
    takeaway_text = """
    Key Insights:

    1. SARIMA with s=24 misses weekly pattern
       → Higher error (MAPE = 8.5%)

    2. TBATS and Prophet capture both
       daily AND weekly seasonality
       → Much better (MAPE ~ 4-5%)

    3. Prophet + holidays adds value
       when special days matter
       → Best result (MAPE = 3.9%)

    Conclusion: Multiple seasonality
    models significantly outperform
    single-seasonality SARIMA!
    """
    axes[1, 1].text(0.1, 0.95, takeaway_text, transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    save_fig('ch9_quiz10_energy_example')

ch9_quiz10_energy_example()


print("\nAll Chapter 9 quiz charts created successfully!")
