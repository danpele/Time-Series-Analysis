"""
Generate educational charts for seminar quiz answers
Time Series Analysis Course
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import os

# Set style
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

# Create charts directory if needed
os.makedirs('charts', exist_ok=True)

# Colors
BLUE = '#1a3a6e'
RED = '#dc3545'
GREEN = '#2e7d32'
ORANGE = '#f57c00'
PURPLE = '#7b1fa2'

#=============================================================================
# CHAPTER 1 SEMINAR CHARTS
#=============================================================================

def ch1_acf_decay():
    """ACF patterns: stationary vs non-stationary"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    lags = np.arange(0, 21)

    # Stationary AR(1) - fast decay
    phi = 0.7
    acf_stationary = phi ** lags

    # Non-stationary (random walk approximation)
    acf_nonstationary = 1 - lags / 50  # Slow decay
    acf_nonstationary = np.maximum(acf_nonstationary, 0.5)

    # Plot stationary
    axes[0].bar(lags, acf_stationary, color=BLUE, alpha=0.7, width=0.6)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].axhline(y=1.96/np.sqrt(100), color=RED, linestyle='--', label='95% CI')
    axes[0].axhline(y=-1.96/np.sqrt(100), color=RED, linestyle='--')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].set_title('Stationary: Fast Decay\n(AR(1) with φ=0.7)', fontweight='bold')
    axes[0].set_ylim(-0.2, 1.1)
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    # Plot non-stationary
    axes[1].bar(lags, acf_nonstationary, color=RED, alpha=0.7, width=0.6)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].axhline(y=1.96/np.sqrt(100), color=BLUE, linestyle='--', label='95% CI')
    axes[1].axhline(y=-1.96/np.sqrt(100), color=BLUE, linestyle='--')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('ACF')
    axes[1].set_title('Non-Stationary: Slow Decay\n(Random Walk)', fontweight='bold')
    axes[1].set_ylim(-0.2, 1.1)
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    plt.tight_layout()
    plt.savefig('charts/sem1_acf_decay.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem1_acf_decay.pdf")

def ch1_forecast_intervals():
    """Forecast intervals widening with horizon"""
    fig, ax = plt.subplots(figsize=(10, 5))

    T = 50
    H = 30
    np.random.seed(42)

    # Historical data (random walk)
    y_hist = np.cumsum(np.random.randn(T)) + 100

    # Forecasts
    h = np.arange(1, H+1)
    y_forecast = np.full(H, y_hist[-1])

    # Confidence intervals (widen with sqrt(h))
    sigma = 1.5
    ci_upper = y_hist[-1] + 1.96 * sigma * np.sqrt(h)
    ci_lower = y_hist[-1] - 1.96 * sigma * np.sqrt(h)

    # Plot
    time_hist = np.arange(T)
    time_fc = np.arange(T-1, T+H)

    ax.plot(time_hist, y_hist, color=BLUE, linewidth=2, label='Historical Data')
    ax.plot(time_fc, [y_hist[-1]] + list(y_forecast), color=RED, linewidth=2,
            linestyle='--', label='Forecast')
    ax.fill_between(time_fc[1:], ci_lower, ci_upper, color=RED, alpha=0.2,
                    label='95% CI')

    ax.axvline(x=T-1, color='gray', linestyle=':', linewidth=1)
    ax.annotate('Forecast\nOrigin', xy=(T-1, y_hist[-1]), xytext=(T-5, y_hist[-1]+8),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Forecast Intervals Widen with Horizon\n(Random Walk: Var = h·σ²)',
                fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

    plt.tight_layout()
    plt.savefig('charts/sem1_forecast_intervals.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem1_forecast_intervals.pdf")

def ch1_timeseries_cv():
    """Time series cross-validation vs standard k-fold"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    n_obs = 20
    n_folds = 5

    # Standard k-fold (wrong for time series)
    ax = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, n_folds))

    for fold in range(n_folds):
        fold_indices = np.arange(fold, n_obs, n_folds)
        for i in range(n_obs):
            if i in fold_indices:
                ax.add_patch(plt.Rectangle((i, fold*0.8), 0.9, 0.7,
                            facecolor=RED, alpha=0.7, edgecolor='black'))
            else:
                ax.add_patch(plt.Rectangle((i, fold*0.8), 0.9, 0.7,
                            facecolor=BLUE, alpha=0.5, edgecolor='black'))

    ax.set_xlim(-0.5, n_obs+0.5)
    ax.set_ylim(-0.5, n_folds*0.8+0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Fold')
    ax.set_title('Standard K-Fold CV (WRONG for Time Series)\nTest data scattered randomly - violates temporal order!',
                fontweight='bold', color=RED)
    ax.set_yticks([i*0.8+0.35 for i in range(n_folds)])
    ax.set_yticklabels([f'Fold {i+1}' for i in range(n_folds)])

    # Time series CV (correct)
    ax = axes[1]

    for fold in range(n_folds):
        train_end = 8 + fold * 2
        test_start = train_end
        test_end = test_start + 2

        for i in range(n_obs):
            if i < train_end:
                ax.add_patch(plt.Rectangle((i, fold*0.8), 0.9, 0.7,
                            facecolor=BLUE, alpha=0.5, edgecolor='black'))
            elif i < test_end:
                ax.add_patch(plt.Rectangle((i, fold*0.8), 0.9, 0.7,
                            facecolor=GREEN, alpha=0.7, edgecolor='black'))

    ax.set_xlim(-0.5, n_obs+0.5)
    ax.set_ylim(-0.5, n_folds*0.8+0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Fold')
    ax.set_title('Time Series CV (CORRECT)\nTrain on past (blue), test on future (green) - respects temporal order',
                fontweight='bold', color=GREEN)
    ax.set_yticks([i*0.8+0.35 for i in range(n_folds)])
    ax.set_yticklabels([f'Fold {i+1}' for i in range(n_folds)])

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=BLUE, alpha=0.5, label='Training'),
                      Patch(facecolor=GREEN, alpha=0.7, label='Test'),
                      Patch(facecolor=RED, alpha=0.7, label='Test (k-fold)')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    plt.tight_layout()
    plt.savefig('charts/sem1_timeseries_cv.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem1_timeseries_cv.pdf")

#=============================================================================
# CHAPTER 3 SEMINAR CHARTS
#=============================================================================

def ch3_random_walk_variance():
    """Random walk variance grows linearly with time"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    np.random.seed(42)
    T = 100
    n_paths = 50

    # Simulate random walks
    ax = axes[0]
    for _ in range(n_paths):
        rw = np.cumsum(np.random.randn(T))
        ax.plot(rw, alpha=0.3, color=BLUE, linewidth=0.8)

    # Show expanding envelope
    t = np.arange(T)
    sigma = 1
    ax.fill_between(t, -1.96*sigma*np.sqrt(t+1), 1.96*sigma*np.sqrt(t+1),
                    color=RED, alpha=0.2, label='95% bounds')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('$Y_t$')
    ax.set_title('Random Walk Paths\n$Y_t = Y_{t-1} + \\varepsilon_t$', fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    # Variance plot
    ax = axes[1]
    t = np.arange(1, 101)
    var_t = t  # Var(Y_t) = t * sigma^2

    ax.plot(t, var_t, color=BLUE, linewidth=2, label='$Var(Y_t) = t\\sigma^2$')
    ax.fill_between(t, 0, var_t, color=BLUE, alpha=0.2)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Variance')
    ax.set_title('Variance Grows Linearly\nNon-stationary!', fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    plt.tight_layout()
    plt.savefig('charts/sem3_rw_variance.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem3_rw_variance.pdf")

def ch3_adf_kpss_comparison():
    """ADF vs KPSS test comparison"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Create a comparison diagram
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # ADF box
    ax.add_patch(plt.Rectangle((0.5, 4.5), 4, 3, facecolor=BLUE, alpha=0.3, edgecolor=BLUE, linewidth=2))
    ax.text(2.5, 7, 'ADF Test', fontsize=14, fontweight='bold', ha='center', color=BLUE)
    ax.text(2.5, 6.2, '$H_0$: Unit Root', fontsize=11, ha='center')
    ax.text(2.5, 5.5, '$H_1$: Stationary', fontsize=11, ha='center')
    ax.text(2.5, 4.8, 'Reject if t-stat < critical', fontsize=9, ha='center', style='italic')

    # KPSS box
    ax.add_patch(plt.Rectangle((5.5, 4.5), 4, 3, facecolor=GREEN, alpha=0.3, edgecolor=GREEN, linewidth=2))
    ax.text(7.5, 7, 'KPSS Test', fontsize=14, fontweight='bold', ha='center', color=GREEN)
    ax.text(7.5, 6.2, '$H_0$: Stationary', fontsize=11, ha='center')
    ax.text(7.5, 5.5, '$H_1$: Unit Root', fontsize=11, ha='center')
    ax.text(7.5, 4.8, 'Reject if LM > critical', fontsize=9, ha='center', style='italic')

    # Decision table
    ax.text(5, 3.5, 'Decision Matrix', fontsize=12, fontweight='bold', ha='center')

    # Table
    table_data = [
        ['ADF rejects', 'KPSS fails to reject', '→ Stationary', GREEN],
        ['ADF fails to reject', 'KPSS rejects', '→ Unit Root', RED],
        ['Both reject', 'or both fail', '→ Inconclusive', ORANGE]
    ]

    for i, (col1, col2, col3, color) in enumerate(table_data):
        y = 2.5 - i * 0.8
        ax.text(1.5, y, col1, fontsize=10, ha='center')
        ax.text(4, y, col2, fontsize=10, ha='center')
        ax.text(7, y, col3, fontsize=10, ha='center', fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig('charts/sem3_adf_kpss.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem3_adf_kpss.pdf")

def ch3_overdifferencing():
    """Overdifferencing creates negative autocorrelation"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    np.random.seed(42)
    T = 200

    # I(1) series
    eps = np.random.randn(T)
    y = np.cumsum(eps)  # Random walk

    # First difference (correct)
    dy = np.diff(y)  # Should be white noise

    # Second difference (overdifferenced)
    d2y = np.diff(dy)  # MA(1) with theta=-1

    # ACF function
    def compute_acf(x, nlags=20):
        acf = np.correlate(x - x.mean(), x - x.mean(), mode='full')
        acf = acf[len(acf)//2:]
        acf = acf / acf[0]
        return acf[:nlags+1]

    # Plot original
    axes[0].plot(y[:100], color=BLUE, linewidth=1)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')
    axes[0].set_title('Original: I(1) Series\nNeeds d=1', fontweight='bold')

    # ACF of first difference
    acf1 = compute_acf(dy)
    axes[1].bar(range(len(acf1)), acf1, color=GREEN, alpha=0.7)
    axes[1].axhline(y=1.96/np.sqrt(len(dy)), color='red', linestyle='--')
    axes[1].axhline(y=-1.96/np.sqrt(len(dy)), color='red', linestyle='--')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('ACF')
    axes[1].set_title('First Difference: White Noise\nCorrect differencing!',
                     fontweight='bold', color=GREEN)
    axes[1].set_ylim(-0.6, 1.1)

    # ACF of second difference (overdifferenced)
    acf2 = compute_acf(d2y)
    axes[2].bar(range(len(acf2)), acf2, color=RED, alpha=0.7)
    axes[2].axhline(y=1.96/np.sqrt(len(d2y)), color='blue', linestyle='--')
    axes[2].axhline(y=-1.96/np.sqrt(len(d2y)), color='blue', linestyle='--')
    axes[2].axhline(y=0, color='black', linewidth=0.5)
    axes[2].axhline(y=-0.5, color=ORANGE, linestyle=':', linewidth=2, label='$\\rho_1 \\approx -0.5$')
    axes[2].set_xlabel('Lag')
    axes[2].set_ylabel('ACF')
    axes[2].set_title('Second Difference: Overdifferenced!\nACF at lag 1 ≈ -0.5',
                     fontweight='bold', color=RED)
    axes[2].set_ylim(-0.6, 1.1)
    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    plt.tight_layout()
    plt.savefig('charts/sem3_overdifferencing.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem3_overdifferencing.pdf")

#=============================================================================
# CHAPTER 4 SEMINAR CHARTS
#=============================================================================

def ch4_seasonal_acf():
    """Seasonal ACF patterns"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    lags = np.arange(0, 37)

    # Create seasonal ACF pattern
    acf_seasonal = np.zeros(37)
    acf_seasonal[0] = 1.0
    for i in range(1, 37):
        if i % 12 == 0:
            acf_seasonal[i] = 0.9 ** (i // 12)  # Strong at seasonal lags
        else:
            acf_seasonal[i] = 0.1 * np.exp(-i/5)  # Decay at non-seasonal

    # Seasonal unit root ACF
    acf_unit = np.zeros(37)
    acf_unit[0] = 1.0
    for i in range(1, 37):
        if i % 12 == 0:
            acf_unit[i] = 0.95  # Slow decay at seasonal lags
        else:
            acf_unit[i] = 0.8 * (1 - i/50) if i < 37 else 0.5

    # Plot stationary seasonal
    colors = [GREEN if i % 12 == 0 else BLUE for i in lags]
    axes[0].bar(lags, acf_seasonal, color=colors, alpha=0.7, width=0.8)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].axhline(y=1.96/np.sqrt(100), color=RED, linestyle='--')
    axes[0].axhline(y=-1.96/np.sqrt(100), color=RED, linestyle='--')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].set_title('Stationary Seasonal\nSpikes at 12, 24, 36 (decay quickly)', fontweight='bold')

    # Highlight seasonal lags
    for lag in [12, 24, 36]:
        axes[0].annotate(f'Lag {lag}', xy=(lag, acf_seasonal[lag]),
                        xytext=(lag, acf_seasonal[lag]+0.15),
                        ha='center', fontsize=8, color=GREEN)

    # Plot seasonal unit root
    colors = [RED if i % 12 == 0 else BLUE for i in lags]
    axes[1].bar(lags, acf_unit, color=colors, alpha=0.7, width=0.8)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].axhline(y=1.96/np.sqrt(100), color=GREEN, linestyle='--')
    axes[1].axhline(y=-1.96/np.sqrt(100), color=GREEN, linestyle='--')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('ACF')
    axes[1].set_title('Seasonal Unit Root (D=1 needed)\nSlow decay at seasonal lags',
                     fontweight='bold', color=RED)

    plt.tight_layout()
    plt.savefig('charts/sem4_seasonal_acf.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem4_seasonal_acf.pdf")

def ch4_multiplicative_additive():
    """Multiplicative vs Additive seasonality"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    np.random.seed(42)
    t = np.arange(48)  # 4 years of monthly data

    # Trend
    trend = 100 + 2 * t

    # Seasonal pattern
    seasonal = 10 * np.sin(2 * np.pi * t / 12)

    # Additive
    y_add = trend + seasonal + np.random.randn(48) * 3

    # Multiplicative
    seasonal_mult = 1 + 0.1 * np.sin(2 * np.pi * t / 12)
    y_mult = trend * seasonal_mult * (1 + np.random.randn(48) * 0.02)

    # Plot additive
    axes[0, 0].plot(t, y_add, color=BLUE, linewidth=1.5)
    axes[0, 0].plot(t, trend, color=RED, linewidth=2, linestyle='--', label='Trend')
    axes[0, 0].fill_between(t, trend - 15, trend + 15, alpha=0.2, color=GREEN)
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Additive Seasonality\n$Y_t = T_t + S_t + \\varepsilon_t$', fontweight='bold')
    axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    axes[0, 0].annotate('Constant\namplitude', xy=(40, trend[40]+12), fontsize=9,
                       ha='center', color=GREEN)

    # Plot multiplicative
    axes[0, 1].plot(t, y_mult, color=BLUE, linewidth=1.5)
    axes[0, 1].plot(t, trend, color=RED, linewidth=2, linestyle='--', label='Trend')
    # Show growing amplitude
    upper = trend * 1.15
    lower = trend * 0.85
    axes[0, 1].fill_between(t, lower, upper, alpha=0.2, color=ORANGE)
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].set_title('Multiplicative Seasonality\n$Y_t = T_t \\times S_t \\times \\varepsilon_t$',
                        fontweight='bold')
    axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    axes[0, 1].annotate('Growing\namplitude', xy=(40, upper[40]), fontsize=9,
                       ha='center', color=ORANGE)

    # Subseries plots
    months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

    # Additive subseries
    ax = axes[1, 0]
    for m in range(12):
        values = y_add[m::12]
        years = np.arange(len(values))
        ax.plot(years, values, 'o-', markersize=4, label=months[m] if m < 3 else '')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title('Additive: Parallel Lines\n(Constant seasonal effect)', fontweight='bold', color=GREEN)

    # Multiplicative subseries
    ax = axes[1, 1]
    for m in range(12):
        values = y_mult[m::12]
        years = np.arange(len(values))
        ax.plot(years, values, 'o-', markersize=4, label=months[m] if m < 3 else '')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title('Multiplicative: Diverging Lines\n(Use log transform!)', fontweight='bold', color=ORANGE)

    plt.tight_layout()
    plt.savefig('charts/sem4_mult_add.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem4_mult_add.pdf")

#=============================================================================
# CHAPTER 5 SEMINAR CHARTS
#=============================================================================

def ch5_var_stability():
    """VAR stability - eigenvalues inside unit circle"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)

    # Stable VAR
    ax = axes[0]
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Unit Circle')
    ax.fill(np.cos(theta), np.sin(theta), color='lightgreen', alpha=0.3)

    # Eigenvalues inside
    eigenvalues_stable = [0.5+0.3j, 0.5-0.3j, -0.4]
    for ev in eigenvalues_stable:
        ax.plot(np.real(ev), np.imag(ev), 'o', markersize=15, color=GREEN,
               markeredgecolor='black', markeredgewidth=2)

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('STABLE VAR\nAll eigenvalues inside unit circle', fontweight='bold', color=GREEN)
    ax.text(0, -1.3, '$|\\lambda_i| < 1$ for all $i$', ha='center', fontsize=11)

    # Unstable VAR
    ax = axes[1]
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Unit Circle')
    ax.fill(np.cos(theta), np.sin(theta), color='lightcoral', alpha=0.3)

    # Eigenvalues - one outside
    eigenvalues_unstable = [0.3+0.2j, 0.3-0.2j, 1.1]
    for i, ev in enumerate(eigenvalues_unstable):
        color = RED if np.abs(ev) >= 1 else GREEN
        ax.plot(np.real(ev), np.imag(ev), 'o', markersize=15, color=color,
               markeredgecolor='black', markeredgewidth=2)

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('UNSTABLE VAR\nEigenvalue outside unit circle', fontweight='bold', color=RED)
    ax.annotate('$|\\lambda| > 1$', xy=(1.1, 0), xytext=(1.3, 0.3),
               fontsize=11, color=RED,
               arrowprops=dict(arrowstyle='->', color=RED))

    plt.tight_layout()
    plt.savefig('charts/sem5_var_stability.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem5_var_stability.pdf")

def ch5_irf_example():
    """Example impulse response functions"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    horizons = np.arange(0, 21)

    # Simulate IRFs
    # Shock to Y1 -> Y1 (own response)
    irf_11 = 0.8 ** horizons

    # Shock to Y1 -> Y2 (cross response, delayed)
    irf_21 = np.zeros(21)
    irf_21[1:] = 0.5 * 0.7 ** (horizons[1:] - 1)

    # Shock to Y2 -> Y1 (cross response)
    irf_12 = 0.3 * 0.75 ** horizons

    # Shock to Y2 -> Y2 (own response)
    irf_22 = 0.9 ** horizons

    # Plot
    titles = [('Response of $Y_1$ to shock in $Y_1$', irf_11),
              ('Response of $Y_1$ to shock in $Y_2$', irf_12),
              ('Response of $Y_2$ to shock in $Y_1$', irf_21),
              ('Response of $Y_2$ to shock in $Y_2$', irf_22)]

    for ax, (title, irf) in zip(axes.flat, titles):
        ax.plot(horizons, irf, 'o-', color=BLUE, linewidth=2, markersize=5)
        ax.fill_between(horizons, irf - 0.1, irf + 0.1, color=BLUE, alpha=0.2)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Horizon (h)')
        ax.set_ylabel('Response')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim(-0.5, 20.5)

    fig.suptitle('Impulse Response Functions (IRFs)\nEffect of 1-unit shock over time',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('charts/sem5_irf_example.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem5_irf_example.pdf")

def ch5_granger_diagram():
    """Granger causality concept diagram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.5, 'Granger Causality: Predictive, Not Causal!',
           fontsize=14, fontweight='bold', ha='center')

    # Box for X
    ax.add_patch(plt.Rectangle((1, 4), 2.5, 2, facecolor=BLUE, alpha=0.3,
                               edgecolor=BLUE, linewidth=2))
    ax.text(2.25, 5, 'Past X\n$X_{t-1}, X_{t-2}, ...$', ha='center', va='center', fontsize=11)

    # Box for Y past
    ax.add_patch(plt.Rectangle((1, 1), 2.5, 2, facecolor=GREEN, alpha=0.3,
                               edgecolor=GREEN, linewidth=2))
    ax.text(2.25, 2, 'Past Y\n$Y_{t-1}, Y_{t-2}, ...$', ha='center', va='center', fontsize=11)

    # Box for Y future
    ax.add_patch(plt.Rectangle((6.5, 2.5), 2.5, 2, facecolor=RED, alpha=0.3,
                               edgecolor=RED, linewidth=2))
    ax.text(7.75, 3.5, 'Future Y\n$Y_t$', ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrows
    ax.annotate('', xy=(6.4, 3.5), xytext=(3.6, 5),
               arrowprops=dict(arrowstyle='->', color=BLUE, lw=2))
    ax.annotate('', xy=(6.4, 3.5), xytext=(3.6, 2),
               arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))

    # Question mark
    ax.text(5, 4.8, 'Does this\nimprove\nprediction?', ha='center', va='center',
           fontsize=10, color=BLUE, style='italic')

    # Explanation box
    ax.add_patch(plt.Rectangle((0.5, -0.5), 9, 1.3, facecolor='lightyellow',
                               edgecolor='orange', linewidth=2))
    ax.text(5, 0.15, '"X Granger-causes Y" means: Past X helps predict future Y,\n'
                    'beyond what past Y alone provides. Does NOT imply true causation!',
           ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('charts/sem5_granger_diagram.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem5_granger_diagram.pdf")

def ch5_fevd_example():
    """FEVD example"""
    fig, ax = plt.subplots(figsize=(10, 5))

    horizons = [1, 2, 4, 8, 12, 20]

    # FEVD for Y1 (example values)
    y1_own = [95, 85, 70, 55, 50, 48]
    y1_other = [5, 15, 30, 45, 50, 52]

    x = np.arange(len(horizons))
    width = 0.6

    ax.bar(x, y1_own, width, label='Own shock ($\\varepsilon_1$)', color=BLUE, alpha=0.8)
    ax.bar(x, y1_other, width, bottom=y1_own, label='Other shock ($\\varepsilon_2$)',
          color=ORANGE, alpha=0.8)

    ax.set_xlabel('Forecast Horizon (h)', fontsize=11)
    ax.set_ylabel('Percentage of Variance Explained', fontsize=11)
    ax.set_title('Forecast Error Variance Decomposition (FEVD)\nHow much of $Y_1$ forecast uncertainty comes from each shock?',
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'h={h}' for h in horizons])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    ax.set_ylim(0, 105)

    # Add percentage labels
    for i, (own, other) in enumerate(zip(y1_own, y1_other)):
        ax.text(i, own/2, f'{own}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        ax.text(i, own + other/2, f'{other}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig('charts/sem5_fevd_example.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem5_fevd_example.pdf")

#=============================================================================
# MAIN
#=============================================================================

if __name__ == "__main__":
    print("Generating seminar educational charts...")
    print("=" * 50)

    # Chapter 1 charts
    ch1_acf_decay()
    ch1_forecast_intervals()
    ch1_timeseries_cv()

    # Chapter 3 charts
    ch3_random_walk_variance()
    ch3_adf_kpss_comparison()
    ch3_overdifferencing()

    # Chapter 4 charts
    ch4_seasonal_acf()
    ch4_multiplicative_additive()

    # Chapter 5 charts
    ch5_var_stability()
    ch5_irf_example()
    ch5_granger_diagram()
    ch5_fevd_example()

    print("=" * 50)
    print("All seminar charts generated successfully!")
