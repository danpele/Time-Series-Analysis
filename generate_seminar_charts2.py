"""
Generate additional educational charts for seminar quiz answers
Time Series Analysis Course - Part 2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ArrowStyle
import matplotlib.patches as mpatches
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True

# Create charts directory if needed
os.makedirs('charts', exist_ok=True)

# Colors
BLUE = '#1a3a6e'
RED = '#dc3545'
GREEN = '#2e7d32'
ORANGE = '#f57c00'
PURPLE = '#7b1fa2'
CYAN = '#0097a7'

#=============================================================================
# CHAPTER 1 ADDITIONAL CHARTS
#=============================================================================

def ch1_white_noise():
    """White noise properties visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    np.random.seed(42)
    T = 200
    wn = np.random.randn(T)

    # Time series plot
    axes[0, 0].plot(wn, color=BLUE, linewidth=0.8)
    axes[0, 0].axhline(y=0, color=RED, linewidth=2, label='Mean = 0')
    axes[0, 0].axhline(y=2, color=ORANGE, linestyle='--', alpha=0.7)
    axes[0, 0].axhline(y=-2, color=ORANGE, linestyle='--', alpha=0.7, label='±2σ')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('White Noise: $\\varepsilon_t \\sim WN(0, \\sigma^2)$', fontweight='bold')
    axes[0, 0].legend()

    # Histogram
    axes[0, 1].hist(wn, bins=30, color=BLUE, alpha=0.7, edgecolor='black', density=True)
    x = np.linspace(-4, 4, 100)
    axes[0, 1].plot(x, np.exp(-x**2/2)/np.sqrt(2*np.pi), color=RED, linewidth=2,
                   label='N(0,1) (not required!)')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Distribution\n(Normality NOT required)', fontweight='bold')
    axes[0, 1].legend(fontsize=8)

    # ACF
    from numpy.fft import fft, ifft
    def compute_acf(x, nlags=20):
        n = len(x)
        x = x - x.mean()
        acf = np.correlate(x, x, mode='full')[n-1:]
        acf = acf / acf[0]
        return acf[:nlags+1]

    acf = compute_acf(wn, 20)
    lags = np.arange(len(acf))
    axes[1, 0].bar(lags, acf, color=GREEN, alpha=0.7, width=0.6)
    axes[1, 0].axhline(y=1.96/np.sqrt(T), color=RED, linestyle='--', label='95% CI')
    axes[1, 0].axhline(y=-1.96/np.sqrt(T), color=RED, linestyle='--')
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')
    axes[1, 0].set_title('ACF: No Autocorrelation\n$Cov(\\varepsilon_t, \\varepsilon_s) = 0$ for $t \\neq s$', fontweight='bold')
    axes[1, 0].legend()

    # Properties summary
    axes[1, 1].axis('off')
    props = [
        ('1. $E[\\varepsilon_t] = 0$', 'Zero mean', GREEN),
        ('2. $Var(\\varepsilon_t) = \\sigma^2$', 'Constant variance', GREEN),
        ('3. $Cov(\\varepsilon_t, \\varepsilon_s) = 0$', 'Uncorrelated', GREEN),
        ('4. $\\varepsilon_t \\sim N(0,\\sigma^2)$', 'NOT required!', RED),
    ]

    for i, (formula, desc, color) in enumerate(props):
        y = 0.85 - i * 0.22
        axes[1, 1].text(0.1, y, formula, fontsize=14, transform=axes[1, 1].transAxes)
        axes[1, 1].text(0.55, y, desc, fontsize=12, color=color, fontweight='bold',
                       transform=axes[1, 1].transAxes)

    axes[1, 1].set_title('White Noise Properties', fontweight='bold')

    plt.tight_layout()
    plt.savefig('charts/sem1_white_noise.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem1_white_noise.pdf")

def ch1_holt_method():
    """Holt's exponential smoothing visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    np.random.seed(42)
    T = 40
    t = np.arange(T)

    # Data with trend
    trend = 50 + 1.5 * t
    y = trend + np.random.randn(T) * 3

    # Simple Exponential Smoothing (flat forecast)
    alpha = 0.3
    ses = np.zeros(T)
    ses[0] = y[0]
    for i in range(1, T):
        ses[i] = alpha * y[i] + (1-alpha) * ses[i-1]

    # Holt's method (trend forecast)
    alpha, beta = 0.3, 0.2
    level = np.zeros(T)
    slope = np.zeros(T)
    level[0] = y[0]
    slope[0] = 0
    for i in range(1, T):
        level[i] = alpha * y[i] + (1-alpha) * (level[i-1] + slope[i-1])
        slope[i] = beta * (level[i] - level[i-1]) + (1-beta) * slope[i-1]

    # Forecasts
    H = 15
    h = np.arange(1, H+1)
    ses_fc = np.full(H, ses[-1])
    holt_fc = level[-1] + slope[-1] * h

    # Plot SES
    axes[0].plot(t, y, 'o', color=BLUE, alpha=0.5, markersize=4, label='Data')
    axes[0].plot(t, ses, color=ORANGE, linewidth=2, label='SES Fit')
    axes[0].plot(np.arange(T-1, T+H), [ses[-1]] + list(ses_fc), '--', color=ORANGE,
                linewidth=2, label='SES Forecast')
    axes[0].axvline(x=T-1, color='gray', linestyle=':', alpha=0.7)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Simple Exponential Smoothing\nFlat Forecast (no trend)', fontweight='bold')
    axes[0].legend(fontsize=8)

    # Plot Holt
    axes[1].plot(t, y, 'o', color=BLUE, alpha=0.5, markersize=4, label='Data')
    axes[1].plot(t, level, color=GREEN, linewidth=2, label="Holt's Fit")
    axes[1].plot(np.arange(T-1, T+H), [level[-1]] + list(holt_fc), '--', color=GREEN,
                linewidth=2, label="Holt's Forecast")
    axes[1].axvline(x=T-1, color='gray', linestyle=':', alpha=0.7)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Value')
    axes[1].set_title("Holt's Method\nTrend-Following Forecast", fontweight='bold')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig('charts/sem1_holt_method.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem1_holt_method.pdf")

def ch1_decomposition():
    """Additive vs Multiplicative decomposition"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    np.random.seed(42)
    t = np.arange(48)  # 4 years monthly

    # Components
    trend = 100 + 2 * t
    seasonal = 15 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(48) * 5

    # Additive
    y_add = trend + seasonal + noise

    # Multiplicative
    seasonal_mult = 1 + 0.15 * np.sin(2 * np.pi * t / 12)
    y_mult = trend * seasonal_mult * (1 + np.random.randn(48) * 0.03)

    # Plot additive
    axes[0, 0].plot(t, y_add, color=BLUE, linewidth=1.5)
    axes[0, 0].set_title('Additive: $Y_t = T_t + S_t + \\varepsilon_t$', fontweight='bold')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Value')
    # Show constant amplitude
    axes[0, 0].annotate('', xy=(42, trend[42]+20), xytext=(42, trend[42]-20),
                       arrowprops=dict(arrowstyle='<->', color=GREEN, lw=2))
    axes[0, 0].text(44, trend[42], 'Constant\namplitude', fontsize=9, color=GREEN)

    # Plot multiplicative
    axes[0, 1].plot(t, y_mult, color=BLUE, linewidth=1.5)
    axes[0, 1].set_title('Multiplicative: $Y_t = T_t \\times S_t \\times \\varepsilon_t$', fontweight='bold')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Value')
    # Show growing amplitude
    axes[0, 1].annotate('', xy=(10, trend[10]*1.2), xytext=(10, trend[10]*0.8),
                       arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2))
    axes[0, 1].annotate('', xy=(40, trend[40]*1.2), xytext=(40, trend[40]*0.8),
                       arrowprops=dict(arrowstyle='<->', color=ORANGE, lw=2))
    axes[0, 1].text(25, 220, 'Growing amplitude', fontsize=9, color=ORANGE)

    # Decision diagram
    axes[1, 0].axis('off')
    axes[1, 0].set_xlim(0, 10)
    axes[1, 0].set_ylim(0, 8)

    # Question
    axes[1, 0].text(5, 7, 'When to use which?', fontsize=14, fontweight='bold', ha='center')

    # Additive box
    axes[1, 0].add_patch(FancyBboxPatch((0.5, 3.5), 4, 3, boxstyle="round,pad=0.1",
                                        facecolor='lightgreen', edgecolor=GREEN, linewidth=2))
    axes[1, 0].text(2.5, 5.8, 'ADDITIVE', fontsize=12, fontweight='bold', ha='center', color=GREEN)
    axes[1, 0].text(2.5, 5, 'Seasonal amplitude\nis CONSTANT', fontsize=10, ha='center')
    axes[1, 0].text(2.5, 4.2, 'Variance stable', fontsize=10, ha='center')

    # Multiplicative box
    axes[1, 0].add_patch(FancyBboxPatch((5.5, 3.5), 4, 3, boxstyle="round,pad=0.1",
                                        facecolor='lightyellow', edgecolor=ORANGE, linewidth=2))
    axes[1, 0].text(7.5, 5.8, 'MULTIPLICATIVE', fontsize=12, fontweight='bold', ha='center', color=ORANGE)
    axes[1, 0].text(7.5, 5, 'Seasonal amplitude\nGROWS with level', fontsize=10, ha='center')
    axes[1, 0].text(7.5, 4.2, 'Use log transform!', fontsize=10, ha='center', style='italic')

    # Visual guide
    axes[1, 1].axis('off')
    axes[1, 1].set_xlim(0, 10)
    axes[1, 1].set_ylim(0, 8)
    axes[1, 1].text(5, 7, 'Visual Diagnostic', fontsize=14, fontweight='bold', ha='center')

    # Draw patterns
    x = np.linspace(0, 8, 100)
    # Additive pattern
    y_pattern_add = 3 + 0.3 * x + 0.5 * np.sin(x * 3)
    axes[1, 1].plot(x + 1, y_pattern_add + 3, color=GREEN, linewidth=2)
    axes[1, 1].text(9.5, 6.5, 'Additive\n(parallel)', fontsize=9, color=GREEN, ha='right')

    # Multiplicative pattern
    y_pattern_mult = 3 + 0.3 * x + (0.2 + 0.1*x) * np.sin(x * 3)
    axes[1, 1].plot(x + 1, y_pattern_mult, color=ORANGE, linewidth=2)
    axes[1, 1].text(9.5, 3, 'Multiplicative\n(fan shape)', fontsize=9, color=ORANGE, ha='right')

    plt.tight_layout()
    plt.savefig('charts/sem1_decomposition.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem1_decomposition.pdf")

#=============================================================================
# CHAPTER 2 ADDITIONAL CHARTS
#=============================================================================

def ch2_acf_pacf_patterns():
    """ACF/PACF patterns for AR, MA, ARMA identification"""
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    lags = np.arange(0, 16)

    # AR(1) patterns
    phi = 0.7
    acf_ar1 = phi ** lags
    pacf_ar1 = np.zeros(16)
    pacf_ar1[0] = 1
    pacf_ar1[1] = phi

    axes[0, 0].bar(lags, acf_ar1, color=BLUE, alpha=0.7, width=0.6)
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 0].set_title('AR(1): ACF Decays Exponentially', fontweight='bold')
    axes[0, 0].set_ylabel('ACF')

    axes[0, 1].bar(lags, pacf_ar1, color=BLUE, alpha=0.7, width=0.6)
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].axhline(y=1.96/np.sqrt(100), color=RED, linestyle='--')
    axes[0, 1].axhline(y=-1.96/np.sqrt(100), color=RED, linestyle='--')
    axes[0, 1].set_title('AR(1): PACF Cuts Off at Lag 1', fontweight='bold')
    axes[0, 1].set_ylabel('PACF')
    axes[0, 1].annotate('Cut off!', xy=(2, 0.05), fontsize=10, color=GREEN, fontweight='bold')

    # MA(1) patterns
    theta = 0.6
    acf_ma1 = np.zeros(16)
    acf_ma1[0] = 1
    acf_ma1[1] = theta / (1 + theta**2)

    pacf_ma1 = np.zeros(16)
    pacf_ma1[0] = 1
    for k in range(1, 16):
        pacf_ma1[k] = (-1)**(k+1) * theta**k / (1 - theta**(2*(k+1))) * (1 - theta**2)
    pacf_ma1 = np.clip(pacf_ma1, -1, 1)

    axes[1, 0].bar(lags, acf_ma1, color=GREEN, alpha=0.7, width=0.6)
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].axhline(y=1.96/np.sqrt(100), color=RED, linestyle='--')
    axes[1, 0].axhline(y=-1.96/np.sqrt(100), color=RED, linestyle='--')
    axes[1, 0].set_title('MA(1): ACF Cuts Off at Lag 1', fontweight='bold')
    axes[1, 0].set_ylabel('ACF')
    axes[1, 0].annotate('Cut off!', xy=(2, 0.05), fontsize=10, color=GREEN, fontweight='bold')

    axes[1, 1].bar(lags, pacf_ma1, color=GREEN, alpha=0.7, width=0.6)
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].set_title('MA(1): PACF Decays', fontweight='bold')
    axes[1, 1].set_ylabel('PACF')

    # ARMA(1,1) patterns
    acf_arma = np.zeros(16)
    acf_arma[0] = 1
    for k in range(1, 16):
        acf_arma[k] = 0.5 * 0.7**(k-1)  # Decay after initial

    pacf_arma = np.zeros(16)
    pacf_arma[0] = 1
    for k in range(1, 16):
        pacf_arma[k] = 0.4 * 0.6**(k-1) * (-1)**(k+1)

    axes[2, 0].bar(lags, acf_arma, color=PURPLE, alpha=0.7, width=0.6)
    axes[2, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[2, 0].set_title('ARMA(1,1): ACF Decays', fontweight='bold')
    axes[2, 0].set_xlabel('Lag')
    axes[2, 0].set_ylabel('ACF')

    axes[2, 1].bar(lags, pacf_arma, color=PURPLE, alpha=0.7, width=0.6)
    axes[2, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[2, 1].set_title('ARMA(1,1): PACF Decays', fontweight='bold')
    axes[2, 1].set_xlabel('Lag')
    axes[2, 1].set_ylabel('PACF')

    # Add identification rule summary
    fig.text(0.5, 0.02, 'Identification Rule: ACF cuts off → MA(q), PACF cuts off → AR(p), Both decay → ARMA(p,q)',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('charts/sem2_acf_pacf_patterns.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem2_acf_pacf_patterns.pdf")

def ch2_stationarity_region():
    """AR(2) stationarity region (triangle)"""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Stationarity triangle vertices
    vertices = np.array([[-2, 1], [2, 1], [0, -1], [-2, 1]])

    # Fill the triangle
    ax.fill(vertices[:, 0], vertices[:, 1], color='lightgreen', alpha=0.5,
            edgecolor=GREEN, linewidth=2, label='Stationary Region')

    # Draw axes
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)

    # Mark boundaries
    ax.plot([-2, 2], [1, 1], 'r-', linewidth=2, label='$\\phi_2 = 1$ (unit root)')
    ax.plot([-2, 0], [1, -1], 'b-', linewidth=2, label='$\\phi_1 + \\phi_2 = -1$')
    ax.plot([0, 2], [-1, 1], 'purple', linewidth=2, label='$\\phi_1 - \\phi_2 = 1$')

    # Mark some example points
    examples = [
        (0.5, 0.3, GREEN, 'Stationary\n(0.5, 0.3)'),
        (1.5, 0.2, RED, 'Non-stationary\n(1.5, 0.2)'),
        (0, 0.9, ORANGE, 'Near boundary\n(0, 0.9)'),
    ]

    for x, y, color, label in examples:
        ax.plot(x, y, 'o', markersize=12, color=color, markeredgecolor='black', markeredgewidth=2)
        ax.annotate(label, xy=(x, y), xytext=(x+0.3, y+0.15), fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('$\\phi_1$', fontsize=12)
    ax.set_ylabel('$\\phi_2$', fontsize=12)
    ax.set_title('AR(2) Stationarity Region\n$Y_t = \\phi_1 Y_{t-1} + \\phi_2 Y_{t-2} + \\varepsilon_t$',
                fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add conditions
    ax.text(-2.3, -1.3, 'Conditions:\n$\\phi_2 + \\phi_1 < 1$\n$\\phi_2 - \\phi_1 < 1$\n$|\\phi_2| < 1$',
           fontsize=10, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lightyellow'))

    plt.tight_layout()
    plt.savefig('charts/sem2_stationarity_region.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem2_stationarity_region.pdf")

def ch2_invertibility():
    """MA invertibility visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)

    # Invertible MA(1)
    ax = axes[0]
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    ax.fill(np.cos(theta), np.sin(theta), color='lightcoral', alpha=0.3, label='Inside (non-invertible)')

    # Invertible root
    root_inv = -1/0.6  # theta = 0.6
    ax.plot(root_inv, 0, 'o', markersize=15, color=GREEN, markeredgecolor='black',
            markeredgewidth=2, label=f'Root at {root_inv:.2f}')
    ax.annotate('$\\theta = 0.6$\n$|\\theta| < 1$ ✓', xy=(root_inv, 0),
               xytext=(root_inv-0.3, 0.5), fontsize=10, color=GREEN,
               arrowprops=dict(arrowstyle='->', color=GREEN))

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Invertible MA(1): $\\theta = 0.6$\nRoot outside unit circle',
                fontweight='bold', color=GREEN)
    ax.legend(loc='lower right', fontsize=9)

    # Non-invertible MA(1)
    ax = axes[1]
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
    ax.fill(np.cos(theta), np.sin(theta), color='lightcoral', alpha=0.3, label='Inside (non-invertible)')

    # Non-invertible root
    root_noninv = -1/1.5  # theta = 1.5
    ax.plot(root_noninv, 0, 'o', markersize=15, color=RED, markeredgecolor='black',
            markeredgewidth=2, label=f'Root at {root_noninv:.2f}')
    ax.annotate('$\\theta = 1.5$\n$|\\theta| > 1$ ✗', xy=(root_noninv, 0),
               xytext=(root_noninv+0.3, 0.5), fontsize=10, color=RED,
               arrowprops=dict(arrowstyle='->', color=RED))

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Non-Invertible MA(1): $\\theta = 1.5$\nRoot inside unit circle',
                fontweight='bold', color=RED)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig('charts/sem2_invertibility.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem2_invertibility.pdf")

def ch2_information_criteria():
    """AIC vs BIC comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Model comparison
    p_values = np.arange(1, 8)
    n = 100

    # Simulated log-likelihood (improves then overfits)
    k = len(p_values)
    ll = -200 + 30 * (1 - np.exp(-p_values/2)) - 0.5 * p_values

    # AIC and BIC
    aic = -2 * ll + 2 * (p_values + 1)  # +1 for variance
    bic = -2 * ll + (p_values + 1) * np.log(n)

    # Normalize for plotting
    aic_norm = aic - aic.min() + 10
    bic_norm = bic - bic.min() + 10

    ax = axes[0]
    ax.plot(p_values, aic_norm, 'o-', color=BLUE, linewidth=2, markersize=8, label='AIC')
    ax.plot(p_values, bic_norm, 's-', color=GREEN, linewidth=2, markersize=8, label='BIC')
    ax.axvline(x=p_values[np.argmin(aic_norm)], color=BLUE, linestyle='--', alpha=0.5)
    ax.axvline(x=p_values[np.argmin(bic_norm)], color=GREEN, linestyle='--', alpha=0.5)
    ax.set_xlabel('Model Order (p)')
    ax.set_ylabel('Information Criterion (lower = better)')
    ax.set_title('AIC vs BIC for Model Selection', fontweight='bold')
    ax.legend()
    ax.annotate(f'AIC selects p={p_values[np.argmin(aic_norm)]}',
               xy=(p_values[np.argmin(aic_norm)], aic_norm.min()),
               xytext=(4, aic_norm.min()+5), color=BLUE, fontsize=10)
    ax.annotate(f'BIC selects p={p_values[np.argmin(bic_norm)]}',
               xy=(p_values[np.argmin(bic_norm)], bic_norm.min()),
               xytext=(2, bic_norm.min()+8), color=GREEN, fontsize=10)

    # Penalty comparison
    ax = axes[1]
    n_values = np.arange(10, 501)
    k_param = 5

    aic_penalty = 2 * k_param * np.ones_like(n_values)
    bic_penalty = k_param * np.log(n_values)

    ax.plot(n_values, aic_penalty, color=BLUE, linewidth=2, label=f'AIC penalty: $2k = {2*k_param}$')
    ax.plot(n_values, bic_penalty, color=GREEN, linewidth=2, label=f'BIC penalty: $k\\ln(n)$')
    ax.fill_between(n_values, aic_penalty, bic_penalty, where=bic_penalty > aic_penalty,
                   color='lightgreen', alpha=0.3, label='BIC penalizes more')

    ax.set_xlabel('Sample Size (n)')
    ax.set_ylabel('Penalty Term')
    ax.set_title('Penalty Comparison (k=5 parameters)', fontweight='bold')
    ax.legend(loc='lower right')
    ax.axvline(x=np.exp(2), color='gray', linestyle=':', alpha=0.7)
    ax.annotate(f'n = e² ≈ 7.4\n(equal penalties)', xy=(np.exp(2), 10), fontsize=9)

    plt.tight_layout()
    plt.savefig('charts/sem2_information_criteria.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem2_information_criteria.pdf")

#=============================================================================
# CHAPTER 3 ADDITIONAL CHARTS
#=============================================================================

def ch3_trend_vs_difference():
    """Trend stationary vs difference stationary"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    np.random.seed(42)
    T = 100
    t = np.arange(T)

    # Trend stationary
    trend_det = 50 + 0.5 * t
    eps = np.random.randn(T) * 5
    y_trend = trend_det + eps

    # Difference stationary (random walk with drift)
    drift = 0.5
    y_diff = np.cumsum(np.random.randn(T) + drift)

    # Plot trend stationary
    axes[0, 0].plot(y_trend, color=BLUE, linewidth=1)
    axes[0, 0].plot(trend_det, color=RED, linewidth=2, linestyle='--', label='Deterministic trend')
    axes[0, 0].set_title('Trend Stationary\n$Y_t = \\alpha + \\beta t + \\varepsilon_t$', fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].legend()
    axes[0, 0].annotate('Detrend by\nregression', xy=(80, trend_det[80]), fontsize=9,
                       xytext=(60, trend_det[60]+15),
                       arrowprops=dict(arrowstyle='->', color=GREEN))

    # Plot difference stationary
    axes[0, 1].plot(y_diff, color=BLUE, linewidth=1)
    axes[0, 1].plot(t * drift, color=RED, linewidth=2, linestyle='--', label='Expected path')
    axes[0, 1].set_title('Difference Stationary\n$Y_t = Y_{t-1} + \\mu + \\varepsilon_t$', fontweight='bold')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].legend()
    axes[0, 1].annotate('Apply\ndifferencing', xy=(80, y_diff[80]), fontsize=9,
                       xytext=(60, y_diff[60]+10),
                       arrowprops=dict(arrowstyle='->', color=GREEN))

    # After detrending vs differencing
    y_detrended = y_trend - trend_det
    y_differenced = np.diff(y_diff)

    axes[1, 0].plot(y_detrended, color=GREEN, linewidth=1)
    axes[1, 0].axhline(y=0, color=RED, linewidth=1, linestyle='--')
    axes[1, 0].set_title('After Detrending: Stationary!', fontweight='bold', color=GREEN)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Detrended $Y_t$')

    axes[1, 1].plot(y_differenced, color=GREEN, linewidth=1)
    axes[1, 1].axhline(y=drift, color=RED, linewidth=1, linestyle='--', label=f'Mean = {drift}')
    axes[1, 1].set_title('After Differencing: Stationary!', fontweight='bold', color=GREEN)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('$\\Delta Y_t$')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('charts/sem3_trend_vs_diff.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem3_trend_vs_diff.pdf")

def ch3_arima_flowchart():
    """ARIMA model selection flowchart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Helper function for boxes
    def draw_box(x, y, w, h, text, color, fontsize=10):
        ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor='black', linewidth=2, alpha=0.8))
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

    def draw_diamond(x, y, text, color):
        diamond = plt.Polygon([(x, y+0.6), (x+0.8, y), (x, y-0.6), (x-0.8, y)],
                             facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')

    # Title
    ax.text(6, 9.5, 'ARIMA Model Selection Flowchart', fontsize=14, fontweight='bold', ha='center')

    # Start
    draw_box(6, 8.5, 2.5, 0.7, 'Plot Data & ACF', 'lightblue')

    # Decision 1: Stationary?
    draw_diamond(6, 7.2, 'Stationary?', 'lightyellow')
    ax.annotate('', xy=(6, 6.6), xytext=(6, 8.1), arrowprops=dict(arrowstyle='->', lw=1.5))

    # No path
    ax.text(7.5, 7.2, 'No', fontsize=9, color=RED)
    draw_box(9, 7.2, 2, 0.7, 'Difference\n(d = d + 1)', 'lightcoral')
    ax.annotate('', xy=(8, 7.2), xytext=(6.8, 7.2), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(6, 8.1), xytext=(9, 7.6), arrowprops=dict(arrowstyle='->', lw=1.5, connectionstyle='arc3,rad=0.3'))

    # Yes path
    ax.text(4.5, 7.2, 'Yes', fontsize=9, color=GREEN)
    draw_box(3, 7.2, 2.2, 0.7, 'Examine\nACF/PACF', 'lightgreen')
    ax.annotate('', xy=(3.9, 7.2), xytext=(5.2, 7.2), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Three paths from ACF/PACF
    draw_box(1.5, 5.5, 2, 0.9, 'ACF cuts\noff at q', 'lightblue')
    draw_box(3, 5.5, 2, 0.9, 'PACF cuts\noff at p', 'lightblue')
    draw_box(4.5, 5.5, 2, 0.9, 'Both\ndecay', 'lightblue')

    ax.annotate('', xy=(1.5, 6), xytext=(2.5, 6.8), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(3, 6), xytext=(3, 6.8), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(4.5, 6), xytext=(3.5, 6.8), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Model choices
    draw_box(1.5, 4.3, 1.8, 0.7, 'MA(q)', GREEN, fontsize=11)
    draw_box(3, 4.3, 1.8, 0.7, 'AR(p)', GREEN, fontsize=11)
    draw_box(4.5, 4.3, 2, 0.7, 'ARMA(p,q)', GREEN, fontsize=11)

    ax.annotate('', xy=(1.5, 4.7), xytext=(1.5, 5.1), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(3, 4.7), xytext=(3, 5.1), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(4.5, 4.7), xytext=(4.5, 5.1), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Estimation and diagnostics
    draw_box(3, 3.2, 2.5, 0.7, 'Estimate Model', 'lightyellow')
    ax.annotate('', xy=(3, 3.6), xytext=(3, 3.9), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(3, 3.6), xytext=(1.5, 3.9), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(3, 3.6), xytext=(4.5, 3.9), arrowprops=dict(arrowstyle='->', lw=1.5))

    draw_diamond(3, 2, 'Residuals\nWhite Noise?', 'lightyellow')
    ax.annotate('', xy=(3, 2.6), xytext=(3, 2.85), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Final
    ax.text(4.5, 2, 'Yes', fontsize=9, color=GREEN)
    draw_box(6, 2, 2, 0.7, 'Done!', 'lightgreen', fontsize=11)
    ax.annotate('', xy=(5, 2), xytext=(3.8, 2), arrowprops=dict(arrowstyle='->', lw=1.5))

    ax.text(1.5, 2, 'No', fontsize=9, color=RED)
    ax.text(0.5, 2, 'Try different\np, d, q', fontsize=9, color=RED)
    ax.annotate('', xy=(1.2, 2.8), xytext=(2.2, 2), arrowprops=dict(arrowstyle='->', lw=1.5, connectionstyle='arc3,rad=-0.5'))

    plt.tight_layout()
    plt.savefig('charts/sem3_arima_flowchart.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem3_arima_flowchart.pdf")

#=============================================================================
# CHAPTER 4 ADDITIONAL CHARTS
#=============================================================================

def ch4_sarima_notation():
    """SARIMA notation breakdown"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(6, 7.5, 'SARIMA$(p,d,q) \\times (P,D,Q)_s$ Notation', fontsize=16, fontweight='bold', ha='center')

    # Main formula
    ax.text(6, 6.2, '$\\phi(L)\\Phi(L^s)(1-L)^d(1-L^s)^D Y_t = \\theta(L)\\Theta(L^s)\\varepsilon_t$',
           fontsize=14, ha='center', family='serif')

    # Regular component box
    ax.add_patch(FancyBboxPatch((0.5, 3), 5, 2.5, boxstyle="round,pad=0.1",
                                facecolor='lightblue', edgecolor=BLUE, linewidth=2))
    ax.text(3, 5.2, 'Regular (Non-Seasonal)', fontsize=12, fontweight='bold', ha='center', color=BLUE)

    params = [
        ('$p$', 'AR order', 'Number of AR lags'),
        ('$d$', 'Differencing', 'Regular differences'),
        ('$q$', 'MA order', 'Number of MA lags'),
    ]
    for i, (sym, name, desc) in enumerate(params):
        y = 4.5 - i * 0.6
        ax.text(1, y, sym, fontsize=12, fontweight='bold')
        ax.text(1.8, y, f'= {name}', fontsize=10)
        ax.text(3.5, y, f'({desc})', fontsize=9, color='gray')

    # Seasonal component box
    ax.add_patch(FancyBboxPatch((6.5, 3), 5, 2.5, boxstyle="round,pad=0.1",
                                facecolor='lightgreen', edgecolor=GREEN, linewidth=2))
    ax.text(9, 5.2, 'Seasonal', fontsize=12, fontweight='bold', ha='center', color=GREEN)

    params_s = [
        ('$P$', 'Seasonal AR', f'SAR lags at $s, 2s, ...$'),
        ('$D$', 'Seasonal Diff', f'$(1-L^s)^D$'),
        ('$Q$', 'Seasonal MA', f'SMA lags at $s, 2s, ...$'),
        ('$s$', 'Period', 'Seasonal period'),
    ]
    for i, (sym, name, desc) in enumerate(params_s):
        y = 4.7 - i * 0.5
        ax.text(7, y, sym, fontsize=12, fontweight='bold')
        ax.text(7.8, y, f'= {name}', fontsize=10)
        ax.text(9.3, y, f'({desc})', fontsize=9, color='gray')

    # Example
    ax.add_patch(FancyBboxPatch((1, 0.5), 10, 2, boxstyle="round,pad=0.1",
                                facecolor='lightyellow', edgecolor=ORANGE, linewidth=2))
    ax.text(6, 2.2, 'Example: SARIMA$(1,1,1) \\times (0,1,1)_{12}$', fontsize=12, fontweight='bold', ha='center')
    ax.text(6, 1.5, 'Monthly data with: AR(1), MA(1), one regular diff,', fontsize=10, ha='center')
    ax.text(6, 1, 'one seasonal diff at lag 12, seasonal MA(1)', fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig('charts/sem4_sarima_notation.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem4_sarima_notation.pdf")

def ch4_airline_model():
    """Famous Airline Model visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    np.random.seed(42)
    T = 144  # 12 years of monthly data
    t = np.arange(T)

    # Simulate airline-like data
    trend = np.exp(0.01 * t)
    seasonal = 1 + 0.3 * np.sin(2 * np.pi * t / 12)
    y = 100 * trend * seasonal * (1 + 0.05 * np.random.randn(T))

    # Plot original
    axes[0, 0].plot(y, color=BLUE, linewidth=1)
    axes[0, 0].set_title('Original: Airline Passengers', fontweight='bold')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Passengers')

    # Log transformed
    y_log = np.log(y)
    axes[0, 1].plot(y_log, color=GREEN, linewidth=1)
    axes[0, 1].set_title('Log Transform: $\\log(Y_t)$', fontweight='bold')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Log(Passengers)')

    # Differenced
    y_diff = np.diff(np.diff(y_log), 12)  # Regular then seasonal diff
    axes[1, 0].plot(y_diff, color=PURPLE, linewidth=1)
    axes[1, 0].axhline(y=0, color=RED, linestyle='--')
    axes[1, 0].set_title('$(1-L)(1-L^{12})\\log Y_t$: Stationary!', fontweight='bold')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Differenced')

    # Model box
    axes[1, 1].axis('off')
    axes[1, 1].set_xlim(0, 10)
    axes[1, 1].set_ylim(0, 10)

    axes[1, 1].add_patch(FancyBboxPatch((0.5, 5), 9, 4.5, boxstyle="round,pad=0.1",
                                        facecolor='lightyellow', edgecolor=ORANGE, linewidth=2))

    axes[1, 1].text(5, 9, 'The Airline Model', fontsize=14, fontweight='bold', ha='center')
    axes[1, 1].text(5, 8, 'SARIMA$(0,1,1) \\times (0,1,1)_{12}$', fontsize=12, ha='center', family='serif')
    axes[1, 1].text(5, 7, '$(1-L)(1-L^{12})Y_t = (1+\\theta L)(1+\\Theta L^{12})\\varepsilon_t$',
                   fontsize=11, ha='center', family='serif')
    axes[1, 1].text(5, 5.8, 'Only 2 parameters: $\\theta$ and $\\Theta$', fontsize=11, ha='center')

    axes[1, 1].text(5, 4, 'Why famous?', fontsize=12, fontweight='bold', ha='center')
    points = [
        '• Fits many seasonal economic series remarkably well',
        '• Extremely parsimonious (just 2 parameters)',
        '• Box & Jenkins (1970) airline passenger data',
    ]
    for i, point in enumerate(points):
        axes[1, 1].text(0.8, 3 - i*0.7, point, fontsize=10, ha='left')

    plt.tight_layout()
    plt.savefig('charts/sem4_airline_model.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem4_airline_model.pdf")

#=============================================================================
# CHAPTER 5 ADDITIONAL CHARTS
#=============================================================================

def ch5_cholesky_ordering():
    """Cholesky ordering impact visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.5, 'Cholesky Ordering: Order Matters!', fontsize=14, fontweight='bold', ha='center')

    # Ordering 1: Y1 first
    ax.add_patch(FancyBboxPatch((0.5, 4), 4, 3, boxstyle="round,pad=0.1",
                                facecolor='lightblue', edgecolor=BLUE, linewidth=2))
    ax.text(2.5, 6.7, 'Ordering 1: $Y_1$ first', fontsize=11, fontweight='bold', ha='center', color=BLUE)

    # Draw variables
    ax.add_patch(Circle((1.5, 5.5), 0.4, facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(1.5, 5.5, '$Y_1$', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.add_patch(Circle((3.5, 5.5), 0.4, facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax.text(3.5, 5.5, '$Y_2$', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.annotate('', xy=(3.1, 5.5), xytext=(1.9, 5.5),
               arrowprops=dict(arrowstyle='->', lw=2, color=GREEN))
    ax.text(2.5, 5.9, 'affects', fontsize=9, ha='center')

    ax.text(2.5, 4.5, '$Y_1$ can affect $Y_2$\ncontemporaneously', fontsize=9, ha='center')

    # Ordering 2: Y2 first
    ax.add_patch(FancyBboxPatch((5.5, 4), 4, 3, boxstyle="round,pad=0.1",
                                facecolor='lightyellow', edgecolor=ORANGE, linewidth=2))
    ax.text(7.5, 6.7, 'Ordering 2: $Y_2$ first', fontsize=11, fontweight='bold', ha='center', color=ORANGE)

    ax.add_patch(Circle((6.5, 5.5), 0.4, facecolor='lightyellow', edgecolor='black', linewidth=2))
    ax.text(6.5, 5.5, '$Y_2$', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.add_patch(Circle((8.5, 5.5), 0.4, facecolor='lightgreen', edgecolor='black', linewidth=2))
    ax.text(8.5, 5.5, '$Y_1$', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.annotate('', xy=(8.1, 5.5), xytext=(6.9, 5.5),
               arrowprops=dict(arrowstyle='->', lw=2, color=ORANGE))
    ax.text(7.5, 5.9, 'affects', fontsize=9, ha='center')

    ax.text(7.5, 4.5, '$Y_2$ can affect $Y_1$\ncontemporaneously', fontsize=9, ha='center')

    # Key message
    ax.add_patch(FancyBboxPatch((1, 0.5), 8, 2.8, boxstyle="round,pad=0.1",
                                facecolor='lightcoral', edgecolor=RED, linewidth=2))
    ax.text(5, 3, 'Key Points:', fontsize=11, fontweight='bold', ha='center')
    points = [
        '• Different orderings give DIFFERENT IRFs!',
        '• Ordering should be based on economic theory',
        '• "Fast-moving" variables should come first',
    ]
    for i, point in enumerate(points):
        ax.text(1.3, 2.3 - i*0.6, point, fontsize=10, ha='left')

    plt.tight_layout()
    plt.savefig('charts/sem5_cholesky_ordering.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem5_cholesky_ordering.pdf")

def ch5_var_diagnostics():
    """VAR residual diagnostics"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    np.random.seed(42)
    T = 200

    # Good residuals (white noise)
    resid_good = np.random.randn(T)

    # Bad residuals (autocorrelated)
    resid_bad = np.zeros(T)
    resid_bad[0] = np.random.randn()
    for i in range(1, T):
        resid_bad[i] = 0.6 * resid_bad[i-1] + np.random.randn()

    # ACF function
    def compute_acf(x, nlags=20):
        n = len(x)
        x = x - x.mean()
        acf = np.correlate(x, x, mode='full')[n-1:]
        acf = acf / acf[0]
        return acf[:nlags+1]

    # Good residuals plot
    axes[0, 0].plot(resid_good, color=GREEN, linewidth=0.8)
    axes[0, 0].axhline(y=0, color=RED, linewidth=1)
    axes[0, 0].set_title('Good Residuals: White Noise', fontweight='bold', color=GREEN)
    axes[0, 0].set_xlabel('Time')

    # Good ACF
    acf_good = compute_acf(resid_good)
    lags = np.arange(len(acf_good))
    axes[0, 1].bar(lags, acf_good, color=GREEN, alpha=0.7, width=0.6)
    axes[0, 1].axhline(y=1.96/np.sqrt(T), color=RED, linestyle='--', label='95% CI')
    axes[0, 1].axhline(y=-1.96/np.sqrt(T), color=RED, linestyle='--')
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].set_title('Good: ACF within bounds', fontweight='bold', color=GREEN)
    axes[0, 1].set_xlabel('Lag')
    axes[0, 1].set_ylabel('ACF')
    axes[0, 1].legend()

    # Bad residuals plot
    axes[1, 0].plot(resid_bad, color=RED, linewidth=0.8)
    axes[1, 0].axhline(y=0, color='black', linewidth=1)
    axes[1, 0].set_title('Bad Residuals: Autocorrelated', fontweight='bold', color=RED)
    axes[1, 0].set_xlabel('Time')

    # Bad ACF
    acf_bad = compute_acf(resid_bad)
    axes[1, 1].bar(lags, acf_bad, color=RED, alpha=0.7, width=0.6)
    axes[1, 1].axhline(y=1.96/np.sqrt(T), color=BLUE, linestyle='--', label='95% CI')
    axes[1, 1].axhline(y=-1.96/np.sqrt(T), color=BLUE, linestyle='--')
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].set_title('Bad: ACF exceeds bounds!', fontweight='bold', color=RED)
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('ACF')
    axes[1, 1].legend()
    axes[1, 1].annotate('Model misspecified!\nIncrease lag order', xy=(10, acf_bad[10]),
                       xytext=(12, 0.5), fontsize=9, color=RED,
                       arrowprops=dict(arrowstyle='->', color=RED))

    plt.tight_layout()
    plt.savefig('charts/sem5_var_diagnostics.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem5_var_diagnostics.pdf")

def ch5_var_parameters():
    """VAR parameter counting diagram"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(5, 7.5, 'VAR(p) Parameter Count: The Curse of Dimensionality',
           fontsize=14, fontweight='bold', ha='center')

    # Formula
    ax.text(5, 6.5, 'Parameters per equation: $1 + K \\times p$', fontsize=12, ha='center')
    ax.text(5, 5.9, 'Total parameters: $K(1 + Kp) + K(K+1)/2$', fontsize=12, ha='center')
    ax.text(5, 5.3, '(coefficients + covariance matrix)', fontsize=10, ha='center', color='gray')

    # Table
    table_data = [
        ['K=2, p=1', '2(1+2×1) = 6', '+ 3 = 9'],
        ['K=3, p=2', '3(1+3×2) = 21', '+ 6 = 27'],
        ['K=5, p=4', '5(1+5×4) = 105', '+ 15 = 120'],
        ['K=10, p=4', '10(1+10×4) = 410', '+ 55 = 465'],
    ]

    # Draw table
    ax.add_patch(Rectangle((1.5, 1.5), 7, 3.2, facecolor='lightgray', edgecolor='black', linewidth=2))

    # Header
    ax.text(3, 4.4, 'Model', fontsize=10, fontweight='bold', ha='center')
    ax.text(5.5, 4.4, 'Coefficients', fontsize=10, fontweight='bold', ha='center')
    ax.text(7.5, 4.4, 'Total', fontsize=10, fontweight='bold', ha='center')

    for i, (model, coef, total) in enumerate(table_data):
        y = 3.8 - i * 0.6
        color = RED if i == 3 else 'black'
        ax.text(3, y, model, fontsize=10, ha='center', color=color)
        ax.text(5.5, y, coef, fontsize=10, ha='center', color=color)
        ax.text(7.5, y, total, fontsize=10, ha='center', fontweight='bold', color=color)

    # Warning
    ax.text(5, 0.8, 'Warning: Parameters grow as $K^2 \\times p$ — need lots of data!',
           fontsize=11, ha='center', color=RED, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor=ORANGE))

    plt.tight_layout()
    plt.savefig('charts/sem5_var_parameters.pdf', bbox_inches='tight', dpi=150)
    plt.close()
    print("Created: sem5_var_parameters.pdf")

#=============================================================================
# MAIN
#=============================================================================

if __name__ == "__main__":
    print("Generating additional seminar charts...")
    print("=" * 50)

    # Chapter 1
    ch1_white_noise()
    ch1_holt_method()
    ch1_decomposition()

    # Chapter 2
    ch2_acf_pacf_patterns()
    ch2_stationarity_region()
    ch2_invertibility()
    ch2_information_criteria()

    # Chapter 3
    ch3_trend_vs_difference()
    ch3_arima_flowchart()

    # Chapter 4
    ch4_sarima_notation()
    ch4_airline_model()

    # Chapter 5
    ch5_cholesky_ordering()
    ch5_var_diagnostics()
    ch5_var_parameters()

    print("=" * 50)
    print("All additional charts generated successfully!")
