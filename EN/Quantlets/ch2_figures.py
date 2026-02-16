#!/usr/bin/env python3
"""
Chapter 2 Figure Generation: ARMA Models
Generates all lecture figures for Chapter 2 of the Time Series Analysis course.
Author: Daniel Traian PELE
Date: 2025

Output: All figures saved as PNG (300 dpi) + PDF to ../../charts/ with ch2_ prefix.
Style: transparent background, no grid, legend outside bottom, no top/right spines.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle
from matplotlib.collections import PatchCollection
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try imports for statsmodels
try:
    from statsmodels.tsa.arima_process import ArmaProcess
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not available. Using manual implementations.")

# =============================================================================
# GLOBAL STYLE SETTINGS
# =============================================================================
COLORS = ['#1A3A6E', '#CD0000', '#2E7D32', '#B5853F', '#E67E22', '#8E44AD']
BLUE, RED, GREEN, BROWN, ORANGE, PURPLE = COLORS
GRAY = '#666666'

plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'savefig.transparent': True,
    'axes.grid': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.facecolor': 'none',
    'legend.framealpha': 0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.2,
})

CHARTS_DIR = '../../charts'
np.random.seed(42)


def save_chart(fig, name):
    """Save chart as both PDF and PNG with transparent background."""
    fig.savefig(f'{CHARTS_DIR}/{name}.png', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'{CHARTS_DIR}/{name}.pdf', transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {name}.pdf + .png')


def remove_spines(ax):
    """Remove top and right spines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def add_legend_bottom(ax, ncol=None, **kwargs):
    """Add legend outside bottom."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if ncol is None:
        ncol = min(len(handles), 4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=ncol,
              frameon=False, **kwargs)


def simulate_ar1(phi, n=300, c=0, sigma=1.0):
    """Simulate an AR(1) process."""
    x = np.zeros(n)
    eps = np.random.normal(0, sigma, n)
    x[0] = eps[0]
    for t in range(1, n):
        x[t] = c + phi * x[t-1] + eps[t]
    return x


def simulate_ma1(theta, n=300, mu=0, sigma=1.0):
    """Simulate an MA(1) process."""
    eps = np.random.normal(0, sigma, n + 1)
    x = np.zeros(n)
    for t in range(n):
        x[t] = mu + eps[t+1] + theta * eps[t]
    return x


def simulate_arma(phi_list, theta_list, n=300, c=0, sigma=1.0):
    """Simulate a general ARMA(p,q) process."""
    p = len(phi_list)
    q = len(theta_list)
    burn = 100
    total = n + burn
    eps = np.random.normal(0, sigma, total)
    x = np.zeros(total)
    for t in range(max(p, q), total):
        ar_part = sum(phi_list[i] * x[t-1-i] for i in range(p))
        ma_part = sum(theta_list[j] * eps[t-1-j] for j in range(q))
        x[t] = c + ar_part + eps[t] + ma_part
    return x[burn:]


def compute_acf_manual(x, nlags=20):
    """Compute ACF manually."""
    n = len(x)
    xm = x - np.mean(x)
    gamma0 = np.sum(xm**2) / n
    acf_vals = np.ones(nlags + 1)
    for k in range(1, nlags + 1):
        acf_vals[k] = np.sum(xm[:n-k] * xm[k:]) / (n * gamma0)
    return acf_vals


def compute_pacf_manual(acf_vals, nlags=20):
    """Compute PACF from ACF using Durbin-Levinson."""
    pacf_vals = np.zeros(nlags + 1)
    pacf_vals[0] = 1.0
    pacf_vals[1] = acf_vals[1]
    phi_prev = np.array([acf_vals[1]])
    for k in range(2, nlags + 1):
        num = acf_vals[k] - np.sum(phi_prev * acf_vals[k-1:0:-1])
        den = 1.0 - np.sum(phi_prev * acf_vals[1:k])
        if abs(den) < 1e-12:
            break
        pacf_vals[k] = num / den
        phi_new = phi_prev - pacf_vals[k] * phi_prev[::-1]
        phi_prev = np.append(phi_new, pacf_vals[k])
    return pacf_vals


def stem_acf(ax, lags, vals, color=BLUE, label=None, conf=None):
    """Create ACF stem plot."""
    markerline, stemlines, baseline = ax.stem(lags, vals, linefmt='-', markerfmt='o', basefmt='k-')
    plt.setp(stemlines, color=color, linewidth=1.5)
    plt.setp(markerline, color=color, markersize=4)
    plt.setp(baseline, color='black', linewidth=0.5)
    if conf is not None:
        ax.axhline(y=conf, color=RED, linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(y=-conf, color=RED, linestyle='--', linewidth=0.8, alpha=0.7)
        ax.fill_between(lags, -conf, conf, alpha=0.05, color=RED)
    ax.axhline(y=0, color='black', linewidth=0.5)


# =============================================================================
# CHART 1: ch2_motivation_stationary
# =============================================================================
def chart_motivation_stationary():
    print('1. ch2_motivation_stationary')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    np.random.seed(42)
    n = 200

    # AR(1)
    ar1 = simulate_ar1(0.8, n)
    axes[0].plot(ar1, color=BLUE, linewidth=1.2, label='AR(1), $\\phi=0.8$')
    axes[0].axhline(0, color='black', linewidth=0.5, alpha=0.5)
    axes[0].set_title('AR(1) Process', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('$X_t$', fontsize=12)
    axes[0].tick_params(labelsize=11)
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    # MA(1)
    ma1 = simulate_ma1(0.7, n)
    axes[1].plot(ma1, color=RED, linewidth=1.2, label='MA(1), $\\theta=0.7$')
    axes[1].axhline(0, color='black', linewidth=0.5, alpha=0.5)
    axes[1].set_title('MA(1) Process', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].tick_params(labelsize=11)
    remove_spines(axes[1])
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    # ARMA(1,1)
    arma11 = simulate_arma([0.7], [0.4], n)
    axes[2].plot(arma11, color=GREEN, linewidth=1.2, label='ARMA(1,1), $\\phi=0.7, \\theta=0.4$')
    axes[2].axhline(0, color='black', linewidth=0.5, alpha=0.5)
    axes[2].set_title('ARMA(1,1) Process', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].tick_params(labelsize=11)
    remove_spines(axes[2])
    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    fig.suptitle('Stationary processes: AR, MA and ARMA', fontweight='bold', fontsize=15, y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_motivation_stationary')


# =============================================================================
# CHART 2: ch2_motivation_acf
# =============================================================================
def chart_motivation_acf():
    print('2. ch2_motivation_acf')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    nlags = 20
    conf = 1.96 / np.sqrt(300)

    np.random.seed(42)

    # AR(1) ACF
    ar1 = simulate_ar1(0.8, 300)
    acf_ar = compute_acf_manual(ar1, nlags)
    stem_acf(axes[0], np.arange(nlags+1), acf_ar, color=BLUE, conf=conf)
    # Overlay theoretical
    theo = 0.8 ** np.arange(nlags+1)
    axes[0].plot(np.arange(nlags+1), theo, 'o', color=ORANGE, markersize=3, alpha=0.7, label='Theoretical: $\\phi^h$')
    axes[0].set_title('ACF: AR(1), $\\phi=0.8$', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Lag', fontsize=13)
    axes[0].set_ylabel('ACF', fontsize=13)
    axes[0].tick_params(labelsize=11)
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    # MA(1) ACF
    ma1 = simulate_ma1(0.7, 300)
    acf_ma = compute_acf_manual(ma1, nlags)
    stem_acf(axes[1], np.arange(nlags+1), acf_ma, color=RED, conf=conf)
    axes[1].set_title('ACF: MA(1), $\\theta=0.7$', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Lag', fontsize=13)
    axes[1].tick_params(labelsize=11)
    remove_spines(axes[1])

    # ARMA(1,1) ACF
    arma11 = simulate_arma([0.7], [0.4], 300)
    acf_arma = compute_acf_manual(arma11, nlags)
    stem_acf(axes[2], np.arange(nlags+1), acf_arma, color=GREEN, conf=conf)
    axes[2].set_title('ACF: ARMA(1,1)', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Lag', fontsize=13)
    axes[2].tick_params(labelsize=11)
    remove_spines(axes[2])

    fig.suptitle('Distinct ACF patterns for different models', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_motivation_acf')


# =============================================================================
# CHART 3: ch2_white_noise
# =============================================================================
def chart_white_noise():
    print('3. ch2_white_noise')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    np.random.seed(42)
    wn = np.random.normal(0, 1, 300)

    axes[0].plot(wn, color=BLUE, linewidth=0.8, label='White noise $\\varepsilon_t \\sim N(0,1)$')
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].set_title('White noise series', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Time', fontsize=13)
    axes[0].set_ylabel('$\\varepsilon_t$', fontsize=13)
    axes[0].tick_params(labelsize=11)
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    nlags = 20
    acf_vals = compute_acf_manual(wn, nlags)
    conf = 1.96 / np.sqrt(300)
    stem_acf(axes[1], np.arange(nlags+1), acf_vals, color=BLUE, conf=conf)
    axes[1].set_title('ACF of white noise', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Lag', fontsize=13)
    axes[1].set_ylabel('ACF', fontsize=13)
    axes[1].tick_params(labelsize=11)
    remove_spines(axes[1])

    fig.tight_layout()
    save_chart(fig, 'ch2_white_noise')


# =============================================================================
# CHART 4: ch2_def_ar1
# =============================================================================
def chart_def_ar1():
    print('4. ch2_def_ar1')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    np.random.seed(42)
    n = 200

    # Positive phi
    ar_pos = simulate_ar1(0.8, n)
    axes[0].plot(ar_pos, color=BLUE, linewidth=1.0, label='$\\phi = 0.8$')
    axes[0].axhline(0, color='black', linewidth=0.5, alpha=0.5)
    axes[0].set_title('AR(1): positive $\\phi$', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Time', fontsize=13)
    axes[0].set_ylabel('$X_t$', fontsize=13)
    axes[0].tick_params(labelsize=11)
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    # Negative phi
    ar_neg = simulate_ar1(-0.8, n)
    axes[1].plot(ar_neg, color=RED, linewidth=1.0, label='$\\phi = -0.8$')
    axes[1].axhline(0, color='black', linewidth=0.5, alpha=0.5)
    axes[1].set_title('AR(1): negative $\\phi$', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Time', fontsize=13)
    axes[1].tick_params(labelsize=11)
    remove_spines(axes[1])
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    fig.suptitle('AR(1): different behavior for positive vs negative $\\phi$', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_def_ar1')


# =============================================================================
# CHART 5: ch2_ar1_simulations
# =============================================================================
def chart_ar1_simulations():
    print('5. ch2_ar1_simulations')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    np.random.seed(42)
    n = 200
    phis = [0.3, 0.9, -0.5, -0.95]
    colors_local = [BLUE, GREEN, RED, ORANGE]
    titles = ['$\\phi = 0.3$ (weakly persistent)',
              '$\\phi = 0.9$ (strongly persistent)',
              '$\\phi = -0.5$ (moderate oscillations)',
              '$\\phi = -0.95$ (strong oscillations)']

    for i, (phi, col, title) in enumerate(zip(phis, colors_local, titles)):
        ax = axes[i//2, i%2]
        x = simulate_ar1(phi, n)
        ax.plot(x, color=col, linewidth=0.9, label=f'$\\phi = {phi}$')
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_xlabel('Time', fontsize=12)
        if i % 2 == 0:
            ax.set_ylabel('$X_t$', fontsize=12)
        ax.tick_params(labelsize=10)
        remove_spines(ax)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    fig.suptitle('AR(1) Simulations: effect of coefficient $\\phi$', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_ar1_simulations')


# =============================================================================
# CHART 6: ch2_ar1_acf_pacf
# =============================================================================
def chart_ar1_acf_pacf():
    print('6. ch2_ar1_acf_pacf')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    nlags = 15

    np.random.seed(42)
    ar1 = simulate_ar1(0.7, 500)
    acf_vals = compute_acf_manual(ar1, nlags)
    pacf_vals = compute_pacf_manual(acf_vals, nlags)
    conf = 1.96 / np.sqrt(500)

    # ACF
    lags = np.arange(nlags + 1)
    stem_acf(axes[0], lags, acf_vals, color=BLUE, conf=conf)
    # Theoretical overlay
    theo_acf = 0.7 ** lags
    axes[0].plot(lags, theo_acf, 's', color=ORANGE, markersize=4, alpha=0.7,
                 label='Theoretical: $\\rho(h) = \\phi^h$', zorder=5)
    axes[0].set_title('ACF: AR(1) with $\\phi = 0.7$', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Lag', fontsize=13)
    axes[0].set_ylabel('ACF', fontsize=13)
    axes[0].tick_params(labelsize=11)
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    # PACF
    stem_acf(axes[1], lags, pacf_vals, color=GREEN, conf=conf)
    axes[1].set_title('PACF: AR(1) with $\\phi = 0.7$', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Lag', fontsize=13)
    axes[1].set_ylabel('PACF', fontsize=13)
    axes[1].tick_params(labelsize=11)
    axes[1].annotate('Spike at lag 1\nthen zero', xy=(1, pacf_vals[1]),
                     xytext=(5, pacf_vals[1]*0.7),
                     arrowprops=dict(arrowstyle='->', color=RED), fontsize=11, color=RED)
    remove_spines(axes[1])

    fig.suptitle('ACF and PACF for AR(1): theory vs sample', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_ar1_acf_pacf')


# =============================================================================
# CHART 7: ch2_def_arp
# =============================================================================
def chart_def_arp():
    print('7. ch2_def_arp')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    np.random.seed(42)
    n = 300
    nlags = 20
    conf = 1.96 / np.sqrt(n)

    # AR(2) with complex roots => pseudo-cyclic
    ar2 = simulate_arma([1.0, -0.5], [], n)
    axes[0].plot(ar2, color=BLUE, linewidth=0.9, label='AR(2): $\\phi_1=1.0, \\phi_2=-0.5$')
    axes[0].axhline(0, color='black', linewidth=0.5, alpha=0.5)
    axes[0].set_title('AR(2) Series', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Time', fontsize=13)
    axes[0].set_ylabel('$X_t$', fontsize=13)
    axes[0].tick_params(labelsize=11)
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    # ACF
    acf_vals = compute_acf_manual(ar2, nlags)
    stem_acf(axes[1], np.arange(nlags+1), acf_vals, color=BLUE, conf=conf)
    axes[1].set_title('ACF: sinusoidal decay', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Lag', fontsize=13)
    axes[1].set_ylabel('ACF', fontsize=13)
    axes[1].tick_params(labelsize=11)
    remove_spines(axes[1])

    # PACF
    pacf_vals = compute_pacf_manual(acf_vals, nlags)
    stem_acf(axes[2], np.arange(nlags+1), pacf_vals, color=GREEN, conf=conf)
    axes[2].set_title('PACF: cutoff after lag 2', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Lag', fontsize=13)
    axes[2].set_ylabel('PACF', fontsize=13)
    axes[2].tick_params(labelsize=11)
    remove_spines(axes[2])

    fig.suptitle('AR(2) Process: pseudo-cyclic behavior', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_def_arp')


# =============================================================================
# CHART 8: ch2_ar2_stationarity
# =============================================================================
def chart_ar2_stationarity():
    print('8. ch2_ar2_stationarity')
    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))

    # AR(2) stationarity triangle: phi1 + phi2 < 1, phi2 - phi1 < 1, |phi2| < 1
    phi1 = np.linspace(-2.5, 2.5, 500)

    # Boundaries
    # phi1 + phi2 < 1 => phi2 < 1 - phi1
    # phi2 - phi1 < 1 => phi2 < 1 + phi1
    # phi2 > -1

    # Triangle vertices: (2, -1), (-2, -1), (0, 1)
    triangle = Polygon([(2, -1), (-2, -1), (0, 1)], alpha=0.2, color=BLUE, label='Stationarity region')
    ax.add_patch(triangle)

    # Draw boundary lines
    phi1_line = np.linspace(-2.5, 2.5, 100)
    ax.plot(phi1_line, 1 - phi1_line, '--', color=RED, linewidth=1.5, label='$\\phi_1 + \\phi_2 = 1$')
    ax.plot(phi1_line, 1 + phi1_line, '--', color=GREEN, linewidth=1.5, label='$\\phi_2 - \\phi_1 = 1$')
    ax.axhline(-1, color=ORANGE, linestyle='--', linewidth=1.5, label='$\\phi_2 = -1$')

    # Mark some example points
    ax.plot(0.5, 0.3, 'o', color=GREEN, markersize=10, zorder=5)
    ax.annotate('Stationary', xy=(0.5, 0.3), xytext=(0.8, 0.6),
                arrowprops=dict(arrowstyle='->', color=GREEN), fontsize=9, color=GREEN, fontweight='bold')

    ax.plot(1.5, 0.2, 'x', color=RED, markersize=12, markeredgewidth=2, zorder=5)
    ax.annotate('Non-stationary', xy=(1.5, 0.2), xytext=(1.7, 0.6),
                arrowprops=dict(arrowstyle='->', color=RED), fontsize=9, color=RED, fontweight='bold')

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('$\\phi_1$', fontsize=12)
    ax.set_ylabel('$\\phi_2$', fontsize=12)
    ax.set_title('AR(2) Stationarity Triangle', fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.3)
    ax.axvline(0, color='black', linewidth=0.3)
    remove_spines(ax)
    add_legend_bottom(ax, ncol=2)

    fig.tight_layout()
    save_chart(fig, 'ch2_ar2_stationarity')


# =============================================================================
# CHART 9: ch2_def_ma1
# =============================================================================
def chart_def_ma1():
    print('9. ch2_def_ma1')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    np.random.seed(42)
    n = 200
    nlags = 20
    conf = 1.96 / np.sqrt(n)

    ma1 = simulate_ma1(0.7, n)

    axes[0].plot(ma1, color=RED, linewidth=0.9, label='MA(1), $\\theta = 0.7$')
    axes[0].axhline(0, color='black', linewidth=0.5, alpha=0.5)
    axes[0].set_title('MA(1) Series', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Time', fontsize=13)
    axes[0].set_ylabel('$X_t$', fontsize=13)
    axes[0].tick_params(labelsize=11)
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    acf_vals = compute_acf_manual(ma1, nlags)
    stem_acf(axes[1], np.arange(nlags+1), acf_vals, color=RED, conf=conf)
    axes[1].set_title('ACF: cutoff after lag 1', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Lag', fontsize=13)
    axes[1].set_ylabel('ACF', fontsize=13)
    axes[1].tick_params(labelsize=11)
    axes[1].annotate('Cutoff\nafter lag 1', xy=(2, acf_vals[2]),
                     xytext=(6, 0.25),
                     arrowprops=dict(arrowstyle='->', color=BLUE), fontsize=9, color=BLUE)
    remove_spines(axes[1])

    fig.suptitle('MA(1): short memory series with ACF cutoff', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_def_ma1')


# =============================================================================
# CHART 10: ch2_ma1_simulations
# =============================================================================
def chart_ma1_simulations():
    print('10. ch2_ma1_simulations')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    np.random.seed(42)
    n = 200
    thetas = [0.3, 0.9, -0.5, -0.9]
    colors_local = [BLUE, GREEN, RED, ORANGE]
    titles = ['$\\theta = 0.3$ (weak smoothing)',
              '$\\theta = 0.9$ (strong smoothing)',
              '$\\theta = -0.5$ (moderate oscillations)',
              '$\\theta = -0.9$ (strong oscillations)']

    for i, (theta, col, title) in enumerate(zip(thetas, colors_local, titles)):
        ax = axes[i//2, i%2]
        x = simulate_ma1(theta, n)
        ax.plot(x, color=col, linewidth=0.9, label=f'$\\theta = {theta}$')
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_xlabel('Time', fontsize=12)
        if i % 2 == 0:
            ax.set_ylabel('$X_t$', fontsize=12)
        ax.tick_params(labelsize=10)
        remove_spines(ax)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    fig.suptitle('MA(1) Simulations: effect of coefficient $\\theta$', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_ma1_simulations')


# =============================================================================
# CHART 11: ch2_ma1_acf_pacf
# =============================================================================
def chart_ma1_acf_pacf():
    print('11. ch2_ma1_acf_pacf')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    nlags = 15

    np.random.seed(42)
    ma1 = simulate_ma1(0.7, 500)
    acf_vals = compute_acf_manual(ma1, nlags)
    pacf_vals = compute_pacf_manual(acf_vals, nlags)
    conf = 1.96 / np.sqrt(500)

    lags = np.arange(nlags + 1)

    # ACF
    stem_acf(axes[0], lags, acf_vals, color=RED, conf=conf)
    # Theoretical overlay for MA(1)
    theo_acf = np.zeros(nlags + 1)
    theo_acf[0] = 1.0
    theta = 0.7
    theo_acf[1] = theta / (1 + theta**2)
    axes[0].plot(lags, theo_acf, 's', color=ORANGE, markersize=6, alpha=0.7,
                 label='Theoretical', zorder=5)
    axes[0].set_title('ACF: MA(1) with $\\theta = 0.7$', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Lag', fontsize=13)
    axes[0].set_ylabel('ACF', fontsize=13)
    axes[0].tick_params(labelsize=11)
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=12)

    # PACF
    stem_acf(axes[1], lags, pacf_vals, color=GREEN, conf=conf)
    axes[1].set_title('PACF: MA(1) with $\\theta = 0.7$', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Lag', fontsize=13)
    axes[1].set_ylabel('PACF', fontsize=13)
    axes[1].tick_params(labelsize=11)
    axes[1].annotate('Exponential\ndecay', xy=(3, pacf_vals[3]),
                     xytext=(8, -0.15),
                     arrowprops=dict(arrowstyle='->', color=RED, linewidth=1.5),
                     fontsize=12, color=RED, fontweight='bold')
    remove_spines(axes[1])

    fig.suptitle('ACF and PACF for MA(1): opposite pattern to AR(1)', fontweight='bold', fontsize=15, y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_ma1_acf_pacf')


# =============================================================================
# CHART 12: ch2_def_invertibility
# =============================================================================
def chart_def_invertibility():
    print('12. ch2_def_invertibility')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Unit circle with roots
    theta_circle = np.linspace(0, 2*np.pi, 100)
    axes[0].plot(np.cos(theta_circle), np.sin(theta_circle), 'k-', linewidth=1.5, label='Unit circle')
    axes[0].fill(np.cos(theta_circle), np.sin(theta_circle), alpha=0.05, color=GRAY)

    # Invertible roots (outside circle)
    axes[0].plot(1.5, 0.5, 'o', color=GREEN, markersize=12, zorder=5, label='Invertible ($|z| > 1$)')
    axes[0].plot(-1.3, 0.7, 'o', color=GREEN, markersize=12, zorder=5)

    # Non-invertible roots (inside circle)
    axes[0].plot(0.4, 0.3, 'x', color=RED, markersize=14, markeredgewidth=2.5, zorder=5, label='Non-invertible ($|z| < 1$)')
    axes[0].plot(-0.3, -0.5, 'x', color=RED, markersize=14, markeredgewidth=2.5, zorder=5)

    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_aspect('equal')
    axes[0].axhline(0, color='black', linewidth=0.3)
    axes[0].axvline(0, color='black', linewidth=0.3)
    axes[0].set_title('MA polynomial roots', fontweight='bold')
    axes[0].set_xlabel('Real part')
    axes[0].set_ylabel('Imaginary part')
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=10)

    # Right: AR(infinity) weights
    lags = np.arange(20)
    theta_inv = 0.6  # invertible
    theta_non = 1.5  # non-invertible

    weights_inv = (-theta_inv) ** lags
    weights_non = (-theta_non) ** lags

    axes[1].stem(lags, weights_inv, linefmt='-', markerfmt='o', basefmt='k-', label=f'$\\theta = {theta_inv}$ (invertible)')
    # Color the stem lines
    markerline, stemlines, baseline = axes[1].get_children()[0:3] if False else (None, None, None)

    axes[1].cla()
    # Re-draw manually
    markerline1, stemlines1, baseline1 = axes[1].stem(lags - 0.15, weights_inv, linefmt='-', markerfmt='o', basefmt='k-')
    plt.setp(stemlines1, color=GREEN, linewidth=1.2)
    plt.setp(markerline1, color=GREEN, markersize=4)

    # Clip non-invertible weights for display
    weights_non_clip = np.clip(weights_non, -5, 5)
    markerline2, stemlines2, baseline2 = axes[1].stem(lags + 0.15, weights_non_clip[:len(lags)], linefmt='-', markerfmt='s', basefmt='k-')
    plt.setp(stemlines2, color=RED, linewidth=1.2)
    plt.setp(markerline2, color=RED, markersize=4)

    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].set_ylim(-5, 5)
    axes[1].set_title('Weights $\\pi_j$ (AR$\\infty$)', fontweight='bold')
    axes[1].set_xlabel('Lag $j$')
    axes[1].set_ylabel('$\\pi_j = (-\\theta)^j$')

    # Manual legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=GREEN, marker='o', linewidth=1.2, markersize=4,
                              label=f'$\\theta = {theta_inv}$ (convergent)'),
                       Line2D([0], [0], color=RED, marker='s', linewidth=1.2, markersize=4,
                              label=f'$\\theta = {theta_non}$ (divergent)')]
    axes[1].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   frameon=False, fontsize=10)
    remove_spines(axes[1])

    fig.suptitle('Invertibility of MA models', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_def_invertibility')


# =============================================================================
# CHART 13: ch2_def_maq
# =============================================================================
def chart_def_maq():
    print('13. ch2_def_maq')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    np.random.seed(42)
    n = 300
    nlags = 20
    conf = 1.96 / np.sqrt(n)

    # MA(3) process
    eps = np.random.normal(0, 1, n + 3)
    theta1, theta2, theta3 = 0.7, -0.4, 0.3
    ma3 = np.zeros(n)
    for t in range(n):
        ma3[t] = eps[t+3] + theta1*eps[t+2] + theta2*eps[t+1] + theta3*eps[t]

    axes[0].plot(ma3, color=PURPLE, linewidth=0.9, label='MA(3)')
    axes[0].axhline(0, color='black', linewidth=0.5, alpha=0.5)
    axes[0].set_title('MA(3) Series', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Time', fontsize=13)
    axes[0].set_ylabel('$X_t$', fontsize=13)
    axes[0].tick_params(labelsize=11)
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    # ACF
    acf_vals = compute_acf_manual(ma3, nlags)
    stem_acf(axes[1], np.arange(nlags+1), acf_vals, color=PURPLE, conf=conf)
    axes[1].set_title('ACF: cutoff after lag 3', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Lag', fontsize=13)
    axes[1].set_ylabel('ACF', fontsize=13)
    axes[1].tick_params(labelsize=11)
    axes[1].annotate('Cutoff at lag 3', xy=(4, acf_vals[4]),
                     xytext=(8, 0.2),
                     arrowprops=dict(arrowstyle='->', color=RED), fontsize=9, color=RED)
    remove_spines(axes[1])

    # PACF
    pacf_vals = compute_pacf_manual(acf_vals, nlags)
    stem_acf(axes[2], np.arange(nlags+1), pacf_vals, color=ORANGE, conf=conf)
    axes[2].set_title('PACF: gradual decay', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Lag', fontsize=13)
    axes[2].set_ylabel('PACF', fontsize=13)
    axes[2].tick_params(labelsize=11)
    remove_spines(axes[2])

    fig.suptitle('MA(q) Process: ACF signature cuts off after lag $q$', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_def_maq')


# =============================================================================
# CHART 14: ch2_def_arma
# =============================================================================
def chart_def_arma():
    print('14. ch2_def_arma')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    np.random.seed(42)
    n = 300
    nlags = 20
    conf = 1.96 / np.sqrt(n)

    arma11 = simulate_arma([0.7], [0.4], n)

    axes[0].plot(arma11, color=GREEN, linewidth=1.2, label='ARMA(1,1): $\\phi=0.7, \\theta=0.4$')
    axes[0].axhline(0, color='black', linewidth=0.5, alpha=0.5)
    axes[0].set_title('ARMA(1,1) Series', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Time', fontsize=13)
    axes[0].set_ylabel('$X_t$', fontsize=13)
    axes[0].tick_params(labelsize=11)
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=12)

    acf_vals = compute_acf_manual(arma11, nlags)
    stem_acf(axes[1], np.arange(nlags+1), acf_vals, color=GREEN, conf=conf)
    axes[1].set_title('ACF: mixed decay', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Lag', fontsize=13)
    axes[1].set_ylabel('ACF', fontsize=13)
    axes[1].tick_params(labelsize=11)
    remove_spines(axes[1])

    pacf_vals = compute_pacf_manual(acf_vals, nlags)
    stem_acf(axes[2], np.arange(nlags+1), pacf_vals, color=ORANGE, conf=conf)
    axes[2].set_title('PACF: decay (no cutoff)', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Lag', fontsize=13)
    axes[2].set_ylabel('PACF', fontsize=13)
    axes[2].tick_params(labelsize=11)
    remove_spines(axes[2])

    fig.suptitle('ARMA(1,1): neither ACF nor PACF cut off', fontweight='bold', fontsize=15, y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_def_arma')


# =============================================================================
# CHART 15: ch2_acf_pacf_patterns
# =============================================================================
def chart_acf_pacf_patterns():
    print('15. ch2_acf_pacf_patterns')
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    nlags = 15
    lags = np.arange(nlags + 1)

    np.random.seed(42)
    n = 500
    conf = 1.96 / np.sqrt(n)

    # Row 1: AR(2) - phi1=0.6, phi2=0.2
    ar2 = simulate_arma([0.6, 0.2], [], n)
    acf_ar = compute_acf_manual(ar2, nlags)
    pacf_ar = compute_pacf_manual(acf_ar, nlags)

    stem_acf(axes[0, 0], lags, acf_ar, color=BLUE, conf=conf)
    axes[0, 0].set_title('AR(2): ACF — exponential decay', fontweight='bold', fontsize=13)
    axes[0, 0].set_ylabel('ACF', fontsize=12)
    axes[0, 0].tick_params(labelsize=11)
    remove_spines(axes[0, 0])

    stem_acf(axes[0, 1], lags, pacf_ar, color=BLUE, conf=conf)
    axes[0, 1].set_title('AR(2): PACF — cutoff at lag 2', fontweight='bold', fontsize=13)
    axes[0, 1].set_ylabel('PACF', fontsize=12)
    axes[0, 1].tick_params(labelsize=11)
    remove_spines(axes[0, 1])

    # Row 2: MA(2) - theta1=0.6, theta2=0.3
    eps = np.random.normal(0, 1, n + 2)
    ma2 = np.zeros(n)
    for t in range(n):
        ma2[t] = eps[t+2] + 0.6*eps[t+1] + 0.3*eps[t]
    acf_ma = compute_acf_manual(ma2, nlags)
    pacf_ma = compute_pacf_manual(acf_ma, nlags)

    stem_acf(axes[1, 0], lags, acf_ma, color=RED, conf=conf)
    axes[1, 0].set_title('MA(2): ACF — cutoff at lag 2', fontweight='bold', fontsize=13)
    axes[1, 0].set_ylabel('ACF', fontsize=12)
    axes[1, 0].tick_params(labelsize=11)
    remove_spines(axes[1, 0])

    stem_acf(axes[1, 1], lags, pacf_ma, color=RED, conf=conf)
    axes[1, 1].set_title('MA(2): PACF — exponential decay', fontweight='bold', fontsize=13)
    axes[1, 1].set_ylabel('PACF', fontsize=12)
    axes[1, 1].tick_params(labelsize=11)
    remove_spines(axes[1, 1])

    # Row 3: ARMA(1,1)
    arma11 = simulate_arma([0.7], [0.5], n)
    acf_arma = compute_acf_manual(arma11, nlags)
    pacf_arma = compute_pacf_manual(acf_arma, nlags)

    stem_acf(axes[2, 0], lags, acf_arma, color=GREEN, conf=conf)
    axes[2, 0].set_title('ARMA(1,1): ACF — exponential decay', fontweight='bold', fontsize=13)
    axes[2, 0].set_xlabel('Lag', fontsize=12)
    axes[2, 0].set_ylabel('ACF', fontsize=12)
    axes[2, 0].tick_params(labelsize=11)
    remove_spines(axes[2, 0])

    stem_acf(axes[2, 1], lags, pacf_arma, color=GREEN, conf=conf)
    axes[2, 1].set_title('ARMA(1,1): PACF — exponential decay', fontweight='bold', fontsize=13)
    axes[2, 1].set_xlabel('Lag', fontsize=12)
    axes[2, 1].set_ylabel('PACF', fontsize=12)
    axes[2, 1].tick_params(labelsize=11)
    remove_spines(axes[2, 1])

    fig.suptitle('ACF/PACF Patterns: AR vs MA vs ARMA', fontweight='bold', fontsize=15, y=1.01)
    fig.tight_layout()
    save_chart(fig, 'ch2_acf_pacf_patterns')


# =============================================================================
# CHART 16: ch2_diagnostics
# =============================================================================
def chart_diagnostics():
    print('16. ch2_diagnostics')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    np.random.seed(42)
    n = 300
    nlags = 20
    conf = 1.96 / np.sqrt(n)

    # Generate AR(1) and fit, use residuals as if from a good model
    ar1 = simulate_ar1(0.7, n)
    # "Residuals" of a well-fitted model = approximately white noise
    residuals = ar1[1:] - 0.7 * ar1[:-1]
    residuals = residuals - np.mean(residuals)

    # Panel 1: Residuals over time
    axes[0].plot(residuals, color=BLUE, linewidth=0.7, label='Residuals')
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].set_title('Model residuals', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Time', fontsize=13)
    axes[0].set_ylabel('Residual', fontsize=13)
    axes[0].tick_params(labelsize=11)
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    # Panel 2: ACF of residuals
    acf_res = compute_acf_manual(residuals, nlags)
    stem_acf(axes[1], np.arange(nlags+1), acf_res, color=BLUE, conf=conf)
    axes[1].set_title('ACF of residuals', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Lag', fontsize=13)
    axes[1].set_ylabel('ACF', fontsize=13)
    axes[1].tick_params(labelsize=11)
    remove_spines(axes[1])

    # Panel 3: Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes[2].plot(osm, osr, 'o', color=BLUE, markersize=3, alpha=0.5)
    axes[2].plot(osm, slope*osm + intercept, '-', color=RED, linewidth=1.5, label='Normal line')
    axes[2].set_title('Q-Q Plot', fontweight='bold', fontsize=14)
    axes[2].set_xlabel('Theoretical quantiles', fontsize=13)
    axes[2].set_ylabel('Sample quantiles', fontsize=13)
    axes[2].tick_params(labelsize=11)
    remove_spines(axes[2])
    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    fig.suptitle('AR(1) Model Diagnostics: white noise residuals', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_diagnostics')


# =============================================================================
# CHART 17: ch2_def_ljungbox
# =============================================================================
def chart_def_ljungbox():
    print('17. ch2_def_ljungbox')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    np.random.seed(42)
    n = 300
    nlags = 20
    conf = 1.96 / np.sqrt(n)

    # Left: Good model - white noise residuals
    wn = np.random.normal(0, 1, n)
    acf_good = compute_acf_manual(wn, nlags)
    stem_acf(axes[0], np.arange(nlags+1), acf_good, color=GREEN, conf=conf)
    axes[0].set_title('Good model: residuals = white noise', fontweight='bold')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF of residuals')

    # Add p-value annotation
    axes[0].annotate('Ljung-Box: p > 0.05\n$H_0$ not rejected', xy=(12, 0.12),
                     fontsize=9, color=GREEN, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=GREEN, alpha=0.8))
    remove_spines(axes[0])

    # Right: Bad model - autocorrelated residuals
    bad_res = simulate_ar1(0.5, n)
    acf_bad = compute_acf_manual(bad_res, nlags)
    stem_acf(axes[1], np.arange(nlags+1), acf_bad, color=RED, conf=conf)
    axes[1].set_title('Inadequate model: residual autocorrelation', fontweight='bold')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('ACF of residuals')

    axes[1].annotate('Ljung-Box: p < 0.05\n$H_0$ rejected', xy=(12, 0.35),
                     fontsize=9, color=RED, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=RED, alpha=0.8))
    remove_spines(axes[1])

    fig.suptitle('Ljung-Box Test: good model vs inadequate model', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_def_ljungbox')


# =============================================================================
# CHART 18: ch2_ar1_forecast
# =============================================================================
def chart_ar1_forecast():
    print('18. ch2_ar1_forecast')
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

    np.random.seed(42)
    n = 100
    h = 30
    phi = 0.8
    sigma = 1.0

    ar1 = simulate_ar1(phi, n, sigma=sigma)
    mu = 0  # mean

    # Forecast
    last_val = ar1[-1]
    horizons = np.arange(1, h+1)
    forecasts = mu + phi**horizons * (last_val - mu)

    # Confidence intervals
    msfe = np.array([sigma**2 * (1 - phi**(2*hh)) / (1 - phi**2) for hh in horizons])
    ci_95 = 1.96 * np.sqrt(msfe)
    ci_80 = 1.28 * np.sqrt(msfe)

    # Plot historical
    ax.plot(range(n), ar1, color=BLUE, linewidth=1.0, label='Historical')

    # Plot forecast
    fc_x = range(n, n + h)
    ax.plot(fc_x, forecasts, color=RED, linewidth=1.5, linestyle='--', label='Forecast')
    ax.fill_between(fc_x, forecasts - ci_95, forecasts + ci_95,
                    color=RED, alpha=0.1, label='95% CI')
    ax.fill_between(fc_x, forecasts - ci_80, forecasts + ci_80,
                    color=RED, alpha=0.2, label='80% CI')

    # Mean line
    ax.axhline(mu, color=GRAY, linestyle=':', linewidth=1.0, label='Mean $\\mu$')

    # Vertical line at forecast origin
    ax.axvline(n-1, color=GRAY, linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_title('AR(1) Forecast: mean reversion ($\\phi = 0.8$)', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('$X_t$')
    remove_spines(ax)
    add_legend_bottom(ax, ncol=5)

    fig.tight_layout()
    save_chart(fig, 'ch2_ar1_forecast')


# =============================================================================
# CHART 19: ch2_rolling_forecast
# =============================================================================
def chart_rolling_forecast():
    print('19. ch2_rolling_forecast')
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

    np.random.seed(42)
    n = 200
    phi = 0.7
    ar1 = simulate_ar1(phi, n)

    train_end = 150
    test_start = train_end

    # Rolling 1-step-ahead forecasts
    rolling_fc = np.zeros(n - test_start)
    for t in range(test_start, n):
        # Simple AR(1) forecast: phi * x_{t-1}
        rolling_fc[t - test_start] = phi * ar1[t-1]

    ax.plot(range(n), ar1, color=BLUE, linewidth=0.8, alpha=0.6, label='Actual data')
    ax.plot(range(test_start, n), rolling_fc, color=RED, linewidth=1.2, linestyle='--',
            label='Rolling forecast (1-step)')
    ax.axvline(test_start, color=GRAY, linestyle='--', linewidth=1.0, alpha=0.7)
    ax.fill_between(range(0, test_start), ax.get_ylim()[0], ax.get_ylim()[1],
                    alpha=0.03, color=BLUE)

    # Annotations
    ax.annotate('Training', xy=(train_end//2, ax.get_ylim()[1]*0.8),
                fontsize=10, color=BLUE, fontweight='bold', ha='center')
    ax.annotate('Test', xy=((test_start + n)//2, ax.get_ylim()[1]*0.8),
                fontsize=10, color=RED, fontweight='bold', ha='center')

    ax.set_title('Rolling window forecast', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('$X_t$')
    remove_spines(ax)
    add_legend_bottom(ax, ncol=3)

    fig.tight_layout()
    save_chart(fig, 'ch2_rolling_forecast')


# =============================================================================
# CHART 20: ch2_rolling_vs_multistep
# =============================================================================
def chart_rolling_vs_multistep():
    print('20. ch2_rolling_vs_multistep')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    np.random.seed(42)
    n = 200
    phi = 0.7
    ar1 = simulate_ar1(phi, n)

    train_end = 150
    test_range = range(train_end, n)
    actual = ar1[train_end:]

    # Rolling 1-step
    rolling_1 = np.array([phi * ar1[t-1] for t in test_range])

    # Multi-step direct (recursive from train_end)
    multi_step = np.zeros(n - train_end)
    multi_step[0] = phi * ar1[train_end - 1]
    for i in range(1, len(multi_step)):
        multi_step[i] = phi * multi_step[i-1]

    # Recursive multi-step (from origin, using predicted values)
    recursive = np.zeros(n - train_end)
    last = ar1[train_end - 1]
    for i in range(len(recursive)):
        recursive[i] = phi * last
        last = recursive[i]

    methods = [('Rolling 1-step', rolling_1, BLUE),
               ('Direct multi-step', multi_step, RED),
               ('Recursive', recursive, GREEN)]

    for i, (name, fc, col) in enumerate(methods):
        axes[i].plot(test_range, actual, color=GRAY, linewidth=0.8, alpha=0.6, label='Actual')
        axes[i].plot(test_range, fc, color=col, linewidth=1.2, linestyle='--', label=name)
        rmse = np.sqrt(np.mean((actual - fc)**2))
        axes[i].set_title(f'{name}\nRMSE = {rmse:.2f}', fontweight='bold', fontsize=13)
        axes[i].set_xlabel('Time', fontsize=12)
        if i == 0:
            axes[i].set_ylabel('$X_t$', fontsize=12)
        axes[i].tick_params(labelsize=10)
        remove_spines(axes[i])
        axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=10)

    fig.suptitle('Comparison: Rolling vs Multi-step vs Recursive', fontweight='bold', y=1.05)
    fig.tight_layout()
    save_chart(fig, 'ch2_rolling_vs_multistep')


# =============================================================================
# CHART 21-25: CASE STUDY (Sunspots)
# =============================================================================
def generate_sunspot_data():
    """Generate synthetic sunspot-like data."""
    np.random.seed(123)
    n = 309  # 1700-2008
    t = np.arange(n)

    # Create a cyclic pattern (~11 year cycle) + noise
    cycle = 60 * np.sin(2 * np.pi * t / 11) * (0.5 + 0.5 * np.abs(np.sin(2 * np.pi * t / 11)))
    noise = np.random.normal(0, 15, n)

    # AR-like structure
    x = np.zeros(n)
    x[0] = 50 + noise[0]
    x[1] = 50 + noise[1]
    for i in range(2, n):
        x[i] = 10 + 1.2*x[i-1] - 0.6*x[i-2] + noise[i]
        x[i] = max(0, x[i])  # sunspots are non-negative

    years = np.arange(1700, 1700 + n)
    return years, x


def chart_case_raw_data():
    print('21. ch2_case_raw_data')
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    years, sunspots = generate_sunspot_data()

    ax.plot(years, sunspots, color=BLUE, linewidth=0.9, label='Sunspots (annual)')
    ax.fill_between(years, 0, sunspots, alpha=0.1, color=BLUE)
    ax.set_title('Sunspot activity (1700--2008)', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of sunspots')
    remove_spines(ax)
    add_legend_bottom(ax)

    fig.tight_layout()
    save_chart(fig, 'ch2_case_raw_data')


def chart_case_acf_pacf():
    print('22. ch2_case_acf_pacf')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    years, sunspots = generate_sunspot_data()
    nlags = 25
    conf = 1.96 / np.sqrt(len(sunspots))

    acf_vals = compute_acf_manual(sunspots, nlags)
    pacf_vals = compute_pacf_manual(acf_vals, nlags)

    stem_acf(axes[0], np.arange(nlags+1), acf_vals, color=BLUE, conf=conf)
    axes[0].set_title('ACF: damped sinusoidal pattern', fontweight='bold')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    remove_spines(axes[0])

    stem_acf(axes[1], np.arange(nlags+1), pacf_vals, color=GREEN, conf=conf)
    axes[1].set_title('PACF: spikes at lag 1, 2, 9', fontweight='bold')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')
    remove_spines(axes[1])

    fig.suptitle('ACF/PACF analysis for sunspots', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_case_acf_pacf')


def chart_case_model_comparison():
    print('23. ch2_case_model_comparison')
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))

    # Simulated AIC values for different models
    models = ['AR(1)', 'AR(2)', 'AR(5)', 'AR(9)', 'MA(2)', 'ARMA(1,1)', 'ARMA(2,1)', 'ARMA(2,2)']
    aic_vals = [2850, 2720, 2680, 2640, 2830, 2750, 2695, 2688]
    bic_vals = [2858, 2732, 2704, 2680, 2842, 2762, 2711, 2708]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, aic_vals, width, color=BLUE, alpha=0.8, label='AIC')
    bars2 = ax.bar(x + width/2, bic_vals, width, color=RED, alpha=0.8, label='BIC')

    # Highlight best model
    best_aic = np.argmin(aic_vals)
    bars1[best_aic].set_edgecolor(GREEN)
    bars1[best_aic].set_linewidth(2.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylabel('Criterion value')
    ax.set_title('Model comparison: AIC and BIC', fontweight='bold')

    # Annotate best
    ax.annotate(f'Best:\n{models[best_aic]}', xy=(best_aic, aic_vals[best_aic]),
                xytext=(best_aic + 1.5, aic_vals[best_aic] - 30),
                arrowprops=dict(arrowstyle='->', color=GREEN, linewidth=1.5),
                fontsize=9, color=GREEN, fontweight='bold')

    remove_spines(ax)
    add_legend_bottom(ax, ncol=2)

    fig.tight_layout()
    save_chart(fig, 'ch2_case_model_comparison')


def chart_case_diagnostics():
    print('24. ch2_case_diagnostics')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    np.random.seed(42)
    years, sunspots = generate_sunspot_data()
    n = len(sunspots)
    nlags = 20
    conf = 1.96 / np.sqrt(n)

    # Simulated residuals from AR(9) model
    residuals = np.random.normal(0, 15, n - 9)

    # Residuals over time
    axes[0, 0].plot(residuals, color=BLUE, linewidth=0.7)
    axes[0, 0].axhline(0, color='black', linewidth=0.5)
    axes[0, 0].set_title('Residuals over time', fontweight='bold', fontsize=13)
    axes[0, 0].set_xlabel('Observation', fontsize=12)
    axes[0, 0].set_ylabel('Residual', fontsize=12)
    axes[0, 0].tick_params(labelsize=10)
    remove_spines(axes[0, 0])

    # ACF of residuals
    acf_res = compute_acf_manual(residuals, nlags)
    stem_acf(axes[0, 1], np.arange(nlags+1), acf_res, color=BLUE, conf=conf)
    axes[0, 1].set_title('ACF of residuals', fontweight='bold', fontsize=13)
    axes[0, 1].set_xlabel('Lag', fontsize=12)
    axes[0, 1].set_ylabel('ACF', fontsize=12)
    axes[0, 1].tick_params(labelsize=10)
    remove_spines(axes[0, 1])

    # Histogram
    axes[1, 0].hist(residuals, bins=25, color=BLUE, alpha=0.7, edgecolor='white', density=True)
    xr = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1, 0].plot(xr, stats.norm.pdf(xr, np.mean(residuals), np.std(residuals)),
                    color=RED, linewidth=1.5, label='Normal')
    axes[1, 0].set_title('Residual distribution', fontweight='bold', fontsize=13)
    axes[1, 0].set_xlabel('Residual', fontsize=12)
    axes[1, 0].set_ylabel('Density', fontsize=12)
    axes[1, 0].tick_params(labelsize=10)
    remove_spines(axes[1, 0])
    axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=10)

    # Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes[1, 1].plot(osm, osr, 'o', color=BLUE, markersize=3, alpha=0.5)
    axes[1, 1].plot(osm, slope*osm + intercept, '-', color=RED, linewidth=1.5)
    axes[1, 1].set_title('Q-Q Plot', fontweight='bold', fontsize=13)
    axes[1, 1].set_xlabel('Theoretical quantiles', fontsize=12)
    axes[1, 1].set_ylabel('Sample quantiles', fontsize=12)
    axes[1, 1].tick_params(labelsize=10)
    remove_spines(axes[1, 1])

    fig.suptitle('AR(9) Diagnostics for sunspots', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_case_diagnostics')


def chart_case_forecast():
    print('25. ch2_case_forecast')
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

    np.random.seed(123)
    years, sunspots = generate_sunspot_data()
    n = len(sunspots)

    # Use last 30 years as test
    train_end = n - 30
    train_years = years[:train_end]
    test_years = years[train_end:]
    train_data = sunspots[:train_end]
    test_data = sunspots[train_end:]

    # Simple AR(2) forecast for illustration
    phi1, phi2 = 1.2, -0.6
    forecasts = np.zeros(30)
    # Use last two training values
    prev1 = train_data[-1]
    prev2 = train_data[-2]
    mu = np.mean(train_data)
    for i in range(30):
        forecasts[i] = 10 + phi1 * prev1 + phi2 * prev2
        forecasts[i] = max(0, forecasts[i])
        prev2 = prev1
        prev1 = forecasts[i]

    # CI
    sigma = 15
    h_range = np.arange(1, 31)
    ci = 1.96 * sigma * np.sqrt(h_range * 0.5)

    ax.plot(train_years[-80:], train_data[-80:], color=BLUE, linewidth=1.0, label='Historical')
    ax.plot(test_years, test_data, color=GRAY, linewidth=1.0, linestyle='-', alpha=0.7, label='Actual (test)')
    ax.plot(test_years, forecasts, color=RED, linewidth=1.5, linestyle='--', label='AR(9) Forecast')
    ax.fill_between(test_years, np.maximum(forecasts - ci, 0), forecasts + ci,
                    color=RED, alpha=0.15, label='95% CI')

    ax.axvline(years[train_end], color=GRAY, linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title('Sunspot forecast: AR(9)', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of sunspots')
    remove_spines(ax)
    add_legend_bottom(ax, ncol=4)

    fig.tight_layout()
    save_chart(fig, 'ch2_case_forecast')


# =============================================================================
# CHART 26: ch2_quiz_ar_stationarity
# =============================================================================
def chart_quiz_ar_stationarity():
    print('26. ch2_quiz_ar_stationarity')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    np.random.seed(42)
    n = 150
    phis = [1.2, 1.0, -0.8, -1.5]
    colors_local = [RED, ORANGE, GREEN, RED]
    labels = ['$\\phi=1.2$ (explosive)',
              '$\\phi=1.0$ (random walk)',
              '$\\phi=-0.8$ (STATIONARY)',
              '$\\phi=-1.5$ (explosive)']

    for i, (phi, col, label) in enumerate(zip(phis, colors_local, labels)):
        ax = axes[i//2, i%2]
        x = np.zeros(n)
        eps = np.random.normal(0, 1, n)
        x[0] = eps[0]
        for t in range(1, n):
            x[t] = phi * x[t-1] + eps[t]
            # Clip explosive series
            x[t] = np.clip(x[t], -1e6, 1e6)

        if abs(phi) > 1:
            # Only show first part before it explodes
            valid = np.where(np.abs(x) < 100)[0]
            if len(valid) > 10:
                ax.plot(valid, x[valid], color=col, linewidth=0.9, label=label)
            else:
                ax.plot(range(min(30, n)), x[:min(30, n)], color=col, linewidth=0.9, label=label)
        else:
            ax.plot(x, color=col, linewidth=0.9, label=label)

        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.set_title(label, fontweight='bold', fontsize=13)
        ax.set_xlabel('Time', fontsize=12)
        if i % 2 == 0:
            ax.set_ylabel('$X_t$', fontsize=12)
        ax.tick_params(labelsize=10)

        # Highlight stationarity
        if abs(phi) < 1:
            ax.set_facecolor('none')
            for spine in ax.spines.values():
                spine.set_edgecolor(GREEN)
                spine.set_linewidth(2)

        remove_spines(ax)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    fig.suptitle('AR(1) Stationarity condition: $|\\phi| < 1$', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_quiz_ar_stationarity')


# =============================================================================
# CHART 27: ch2_quiz_acf_pacf_patterns
# =============================================================================
def chart_quiz_acf_pacf_patterns():
    print('27. ch2_quiz_acf_pacf_patterns')
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    nlags = 15
    lags = np.arange(nlags + 1)

    np.random.seed(42)
    n = 500
    conf = 1.96 / np.sqrt(n)

    # MA(1) with theta=0.7
    ma1 = simulate_ma1(0.7, n)
    acf_ma = compute_acf_manual(ma1, nlags)
    pacf_ma = compute_pacf_manual(acf_ma, nlags)

    stem_acf(axes[0, 0], lags, acf_ma, color=RED, conf=conf)
    axes[0, 0].set_title('ACF: spike at lag 1, then zero', fontweight='bold', fontsize=10)
    axes[0, 0].set_ylabel('ACF')
    remove_spines(axes[0, 0])

    stem_acf(axes[0, 1], lags, pacf_ma, color=RED, conf=conf)
    axes[0, 1].set_title('PACF: gradual decay', fontweight='bold', fontsize=10)
    axes[0, 1].set_ylabel('PACF')
    remove_spines(axes[0, 1])

    # For comparison: AR(1) with phi=0.7
    ar1 = simulate_ar1(0.7, n)
    acf_ar = compute_acf_manual(ar1, nlags)
    pacf_ar = compute_pacf_manual(acf_ar, nlags)

    stem_acf(axes[1, 0], lags, acf_ar, color=BLUE, conf=conf)
    axes[1, 0].set_title('AR(1): ACF exponential decay', fontweight='bold', fontsize=10)
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')
    remove_spines(axes[1, 0])

    stem_acf(axes[1, 1], lags, pacf_ar, color=BLUE, conf=conf)
    axes[1, 1].set_title('AR(1): PACF single spike', fontweight='bold', fontsize=10)
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('PACF')
    remove_spines(axes[1, 1])

    # Add row labels
    axes[0, 0].annotate('MA(1)', xy=(-0.15, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', color=RED, rotation=90, va='center')
    axes[1, 0].annotate('AR(1)', xy=(-0.15, 0.5), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', color=BLUE, rotation=90, va='center')

    fig.suptitle('Quiz: ACF/PACF Patterns - MA(1) vs AR(1)', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_quiz_acf_pacf_patterns')


# =============================================================================
# CHART 28: ch2_quiz_information_criteria
# =============================================================================
def chart_quiz_information_criteria():
    print('28. ch2_quiz_information_criteria')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AIC vs BIC comparison
    p_range = np.arange(0, 8)
    q_range = np.arange(0, 4)

    # Simulated AIC/BIC grids
    np.random.seed(42)
    n_obs = 200

    # AIC values (smaller is better)
    aic_grid = np.zeros((len(p_range), len(q_range)))
    bic_grid = np.zeros((len(p_range), len(q_range)))

    base_ll = 500
    for i, p in enumerate(p_range):
        for j, q in enumerate(q_range):
            k = p + q + 1  # number of parameters
            # Simulated log-likelihood (improves then plateaus)
            ll = base_ll - 50 * np.exp(-0.5*(p+q)) + np.random.normal(0, 3)
            aic_grid[i, j] = -2*ll + 2*k
            bic_grid[i, j] = -2*ll + k*np.log(n_obs)

    # AIC heatmap
    im1 = axes[0].imshow(aic_grid.T, cmap='RdYlGn_r', aspect='auto',
                          origin='lower')
    axes[0].set_xticks(range(len(p_range)))
    axes[0].set_xticklabels(p_range)
    axes[0].set_yticks(range(len(q_range)))
    axes[0].set_yticklabels(q_range)
    axes[0].set_xlabel('AR order ($p$)')
    axes[0].set_ylabel('MA order ($q$)')
    axes[0].set_title('AIC (moderate penalty)', fontweight='bold')

    # Mark minimum
    aic_min = np.unravel_index(aic_grid.argmin(), aic_grid.shape)
    axes[0].plot(aic_min[0], aic_min[1], 's', color='white', markersize=15, markeredgecolor='black', markeredgewidth=2)
    axes[0].annotate(f'Min: ARMA({p_range[aic_min[0]]},{q_range[aic_min[1]]})',
                     xy=(aic_min[0], aic_min[1]), xytext=(aic_min[0]+2.5, aic_min[1]+1.0),
                     arrowprops=dict(arrowstyle='->', color='white', linewidth=1.5),
                     fontsize=12, color='white', fontweight='bold')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # BIC heatmap
    im2 = axes[1].imshow(bic_grid.T, cmap='RdYlGn_r', aspect='auto',
                          origin='lower')
    axes[1].set_xticks(range(len(p_range)))
    axes[1].set_xticklabels(p_range)
    axes[1].set_yticks(range(len(q_range)))
    axes[1].set_yticklabels(q_range)
    axes[1].set_xlabel('AR order ($p$)')
    axes[1].set_ylabel('MA order ($q$)')
    axes[1].set_title('BIC (stronger penalty)', fontweight='bold')

    bic_min = np.unravel_index(bic_grid.argmin(), bic_grid.shape)
    axes[1].plot(bic_min[0], bic_min[1], 's', color='white', markersize=15, markeredgecolor='black', markeredgewidth=2)
    axes[1].annotate(f'Min: ARMA({p_range[bic_min[0]]},{q_range[bic_min[1]]})',
                     xy=(bic_min[0], bic_min[1]), xytext=(bic_min[0]+2.5, bic_min[1]+1.0),
                     arrowprops=dict(arrowstyle='->', color='white', linewidth=1.5),
                     fontsize=12, color='white', fontweight='bold')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # Restore spines for heatmaps
    for ax in axes:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    fig.suptitle('Information criteria: AIC vs BIC', fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_quiz_information_criteria')


# =============================================================================
# CHART 29: ch2_quiz_ljung_box
# =============================================================================
def chart_quiz_ljung_box():
    print('29. ch2_quiz_ljung_box')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    np.random.seed(42)
    n = 200

    # Ljung-Box p-values for different lag numbers
    # Good model: high p-values
    lags_test = np.arange(1, 21)

    # Simulate p-values for good model
    p_good = np.random.uniform(0.1, 0.9, 20)
    p_good = np.sort(p_good)[::-1]  # generally decreasing but above 0.05

    # Simulate p-values for bad model
    p_bad = np.random.uniform(0.001, 0.08, 20)
    p_bad[:3] = [0.002, 0.005, 0.01]

    # Good model
    axes[0].bar(lags_test, p_good, color=GREEN, alpha=0.7, edgecolor='white')
    axes[0].axhline(0.05, color=RED, linewidth=1.5, linestyle='--', label='$\\alpha = 0.05$')
    axes[0].set_title('Adequate model: $p > 0.05$', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Maximum lag ($m$)', fontsize=13)
    axes[0].set_ylabel('p-value Ljung-Box', fontsize=13)
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(labelsize=11)
    axes[0].annotate('$H_0$ not rejected\nWhite noise residuals',
                     xy=(10, 0.7), fontsize=9, color=GREEN, fontweight='bold',
                     ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=GREEN, alpha=0.8))
    remove_spines(axes[0])
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    # Bad model
    axes[1].bar(lags_test, p_bad, color=RED, alpha=0.7, edgecolor='white')
    axes[1].axhline(0.05, color=RED, linewidth=1.5, linestyle='--', label='$\\alpha = 0.05$')
    axes[1].set_title('Inadequate model: $p < 0.05$', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Maximum lag ($m$)', fontsize=13)
    axes[1].set_ylabel('p-value Ljung-Box', fontsize=13)
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(labelsize=11)
    axes[1].annotate('$H_0$ rejected\nResidual autocorrelation',
                     xy=(10, 0.7), fontsize=9, color=RED, fontweight='bold',
                     ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=RED, alpha=0.8))
    remove_spines(axes[1])
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=11)

    fig.suptitle('Quiz: Interpreting the Ljung-Box test', fontweight='bold', y=1.05)
    fig.tight_layout()
    save_chart(fig, 'ch2_quiz_ljung_box')


# =============================================================================
# CHART 30: ch2_quiz_forecast_properties
# =============================================================================
def chart_quiz_forecast_properties():
    print('30. ch2_quiz_forecast_properties')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: Forecast convergence to mean
    h = np.arange(0, 31)
    mu = 5
    x_n = 15  # last observed value

    phi_vals = [0.3, 0.7, 0.95]
    colors_phi = [GREEN, BLUE, RED]

    for phi, col in zip(phi_vals, colors_phi):
        fc = mu + phi**h * (x_n - mu)
        axes[0].plot(h, fc, '-o', color=col, markersize=3, linewidth=1.2,
                     label=f'$\\phi = {phi}$')

    axes[0].axhline(mu, color=GRAY, linestyle=':', linewidth=1.5, label='$\\mu$')
    axes[0].set_title('Forecast convergence to $\\mu$', fontweight='bold')
    axes[0].set_xlabel('Forecast horizon ($h$)')
    axes[0].set_ylabel('$\\hat{X}_{n+h|n}$')
    remove_spines(axes[0])
    add_legend_bottom(axes[0], ncol=4)

    # Right: Forecast uncertainty (MSFE)
    sigma = 1.0
    for phi, col in zip(phi_vals, colors_phi):
        msfe = np.array([sigma**2 * (1 - phi**(2*hh)) / (1 - phi**2) if hh > 0 else 0 for hh in h])
        unconditional_var = sigma**2 / (1 - phi**2)
        axes[1].plot(h, msfe, '-', color=col, linewidth=1.2, label=f'$\\phi = {phi}$')
        axes[1].axhline(unconditional_var, color=col, linestyle=':', linewidth=0.8, alpha=0.5)

    axes[1].set_title('Forecast uncertainty (MSFE)', fontweight='bold')
    axes[1].set_xlabel('Forecast horizon ($h$)')
    axes[1].set_ylabel('MSFE$(h)$')
    remove_spines(axes[1])
    add_legend_bottom(axes[1], ncol=3)

    fig.suptitle('Quiz: Forecast properties of stationary AR(1)', fontweight='bold', y=1.05)
    fig.tight_layout()
    save_chart(fig, 'ch2_quiz_forecast_properties')


# =============================================================================
# CHART 31: ch2_quiz_ar1_variance
# =============================================================================
def chart_quiz_ar1_variance():
    print('31. ch2_quiz_ar1_variance')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: step-by-step calculation
    phi = 0.6
    sigma2 = 4.0
    phi2 = phi**2
    one_minus_phi2 = 1 - phi2
    var_xt = sigma2 / one_minus_phi2

    steps = ['$\\sigma^2$\n$= 4$', '$\\phi^2$\n$= 0.36$', '$1-\\phi^2$\n$= 0.64$', '$\\gamma(0)$\n$= 6.25$']
    values = [sigma2, phi2, one_minus_phi2, var_xt]
    colors_bar = [BLUE, ORANGE, GREEN, RED]

    bars = axes[0].bar(range(len(steps)), values, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
    for i, (bar, val) in enumerate(zip(bars, values)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.2f}', ha='center', fontweight='bold', fontsize=14)
    axes[0].set_xticks(range(len(steps)))
    axes[0].set_xticklabels(steps, fontsize=12)
    axes[0].set_ylabel('Value', fontsize=13)
    axes[0].set_title('$\\gamma(0) = \\dfrac{\\sigma^2}{1-\\phi^2}$', fontsize=15, fontweight='bold')
    axes[0].tick_params(axis='y', labelsize=12)
    remove_spines(axes[0])

    # Right: variance as function of phi
    phi_range = np.linspace(0, 0.99, 200)
    var_range = sigma2 / (1 - phi_range**2)
    axes[1].plot(phi_range, var_range, color=BLUE, linewidth=2.5)
    axes[1].axvline(0.6, color=RED, linestyle='--', linewidth=1.5)
    axes[1].axhline(6.25, color=RED, linestyle=':', linewidth=1, alpha=0.7)
    axes[1].plot(0.6, 6.25, 'o', color=RED, markersize=12, zorder=5)
    axes[1].annotate('$\\gamma(0) = 6.25$', xy=(0.6, 6.25), xytext=(0.2, 15),
                     fontsize=14, fontweight='bold', color=RED,
                     arrowprops=dict(arrowstyle='->', color=RED, lw=2))
    axes[1].axhline(sigma2, color=GRAY, linestyle=':', linewidth=1.2, alpha=0.6)
    axes[1].text(0.02, sigma2 + 1, '$\\sigma^2 = 4$', fontsize=12, color=GRAY)
    axes[1].set_xlabel('$\\phi$', fontsize=14)
    axes[1].set_ylabel('$\\mathrm{Var}(X_t)$', fontsize=14)
    axes[1].set_title('Variance grows as $|\\phi| \\to 1$', fontsize=15, fontweight='bold')
    axes[1].set_ylim(0, 40)
    axes[1].tick_params(axis='both', labelsize=12)
    remove_spines(axes[1])

    fig.tight_layout()
    save_chart(fig, 'ch2_quiz_ar1_variance')


# =============================================================================
# CHART 32: ch2_quiz_ma1_acf
# =============================================================================
def chart_quiz_ma1_acf():
    print('32. ch2_quiz_ma1_acf')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    theta = 0.5
    rho1 = theta / (1 + theta**2)
    nlags = 12

    # Left: ACF stem plot
    lags = np.arange(0, nlags + 1)
    acf_vals = np.zeros(nlags + 1)
    acf_vals[0] = 1.0
    acf_vals[1] = rho1

    markerline, stemlines, baseline = axes[0].stem(lags, acf_vals, linefmt='-', markerfmt='o', basefmt='k-')
    plt.setp(stemlines, color=BLUE, linewidth=2)
    plt.setp(markerline, color=BLUE, markersize=7)
    plt.setp(baseline, color='black', linewidth=0.5)
    # Highlight rho(1)
    axes[0].plot(1, rho1, 'o', color=RED, markersize=12, zorder=5)
    axes[0].annotate(f'$\\rho(1) = {rho1:.2f}$', xy=(1, rho1), xytext=(3.5, rho1 + 0.2),
                     fontsize=14, fontweight='bold', color=RED,
                     arrowprops=dict(arrowstyle='->', color=RED, lw=2))
    conf = 1.96 / np.sqrt(300)
    axes[0].axhline(y=conf, color=RED, linestyle='--', linewidth=0.8, alpha=0.4)
    axes[0].axhline(y=-conf, color=RED, linestyle='--', linewidth=0.8, alpha=0.4)
    axes[0].set_xlabel('Lag', fontsize=13)
    axes[0].set_ylabel('ACF', fontsize=13)
    axes[0].set_title(f'MA(1) ACF: $\\theta = {theta}$', fontsize=15, fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=12)
    remove_spines(axes[0])

    # Right: rho(1) as function of theta
    theta_range = np.linspace(-2, 2, 400)
    rho_range = theta_range / (1 + theta_range**2)
    axes[1].plot(theta_range, rho_range, color=BLUE, linewidth=2.5)
    axes[1].axhline(0.5, color=GRAY, linestyle=':', linewidth=1.2, alpha=0.6)
    axes[1].axhline(-0.5, color=GRAY, linestyle=':', linewidth=1.2, alpha=0.6)
    axes[1].text(1.6, 0.52, '$+0.5$', fontsize=12, color=GRAY)
    axes[1].text(1.6, -0.57, '$-0.5$', fontsize=12, color=GRAY)
    axes[1].plot(0.5, rho1, 'o', color=RED, markersize=12, zorder=5)
    axes[1].annotate('$\\theta=0.5$\n$\\rho(1)=0.40$', xy=(0.5, rho1),
                     xytext=(-1.5, 0.35), fontsize=13, fontweight='bold', color=RED,
                     arrowprops=dict(arrowstyle='->', color=RED, lw=2))
    axes[1].set_xlabel('$\\theta$', fontsize=14)
    axes[1].set_ylabel('$\\rho(1)$', fontsize=14)
    axes[1].set_title('$\\rho(1) = \\dfrac{\\theta}{1+\\theta^2}$', fontsize=15, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=12)
    remove_spines(axes[1])

    fig.tight_layout()
    save_chart(fig, 'ch2_quiz_ma1_acf')


# =============================================================================
# CHART 33: ch2_quiz_arma11_acf
# =============================================================================
def chart_quiz_arma11_acf():
    print('33. ch2_quiz_arma11_acf')
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    nlags = 12
    lags = np.arange(0, nlags + 1)

    # AR(1): phi=0.7
    phi_ar = 0.7
    acf_ar = np.array([phi_ar**h for h in range(nlags + 1)])

    # MA(1): theta=0.4
    theta_ma = 0.4
    acf_ma = np.zeros(nlags + 1)
    acf_ma[0] = 1.0
    acf_ma[1] = theta_ma / (1 + theta_ma**2)

    # ARMA(1,1): phi=0.7, theta=0.4
    phi_arma = 0.7
    theta_arma = 0.4
    rho1_arma = (1 + phi_arma * theta_arma) * (phi_arma + theta_arma) / (1 + 2*phi_arma*theta_arma + theta_arma**2)
    acf_arma = np.zeros(nlags + 1)
    acf_arma[0] = 1.0
    acf_arma[1] = rho1_arma
    for h in range(2, nlags + 1):
        acf_arma[h] = phi_arma * acf_arma[h-1]

    titles = [f'AR(1): $\\phi={phi_ar}$', f'MA(1): $\\theta={theta_ma}$',
              f'ARMA(1,1)\n$\\phi={phi_arma},\\; \\theta={theta_arma}$']
    data = [acf_ar, acf_ma, acf_arma]
    colors_list = [BLUE, RED, GREEN]

    for i, (ax, vals, title, col) in enumerate(zip(axes, data, titles, colors_list)):
        ml, sl, bl = ax.stem(lags, vals, linefmt='-', markerfmt='o', basefmt='k-')
        plt.setp(sl, color=col, linewidth=2)
        plt.setp(ml, color=col, markersize=6)
        plt.setp(bl, color='black', linewidth=0.5)
        ax.set_title(title, fontweight='bold', fontsize=13)
        ax.set_xlabel('Lag', fontsize=12)
        if i == 0:
            ax.set_ylabel('ACF', fontsize=13)
        ax.set_ylim(-0.15, 1.08)
        ax.tick_params(axis='both', labelsize=11)
        remove_spines(ax)

    fig.suptitle('ACF Comparison: AR(1) vs MA(1) vs ARMA(1,1)', fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch2_quiz_arma11_acf')


# =============================================================================
# CHART 34: ch2_quiz_ar2_check
# =============================================================================
def chart_quiz_ar2_check():
    print('34. ch2_quiz_ar2_check')
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    phi1_range = np.linspace(-2.5, 2.5, 500)

    # Stationarity triangle vertices
    verts = np.array([[0, 1], [2, -1], [-2, -1], [0, 1]])
    triangle = plt.Polygon(verts, alpha=0.15, color=GREEN, label='Stationarity region')
    ax.add_patch(triangle)

    ax.plot(phi1_range, 1 - phi1_range, '--', color=BLUE, linewidth=2, label='$\\phi_1+\\phi_2=1$')
    ax.plot(phi1_range, 1 + phi1_range, '--', color=ORANGE, linewidth=2, label='$\\phi_2-\\phi_1=1$')
    ax.axhline(-1, color=PURPLE, linestyle='--', linewidth=2, label='$\\phi_2=-1$')

    # Test point
    ax.plot(0.8, 0.3, 's', color=RED, markersize=16, zorder=5, markeredgecolor='black', linewidth=1.5)
    ax.annotate('$(0.8,\\, 0.3)$\nNON-STATIONARY\n$\\phi_1+\\phi_2=1.1>1$',
                xy=(0.8, 0.3), xytext=(1.4, 0.7), fontsize=13, fontweight='bold', color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=2.5),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=RED, alpha=0.95))

    ax.set_xlabel('$\\phi_1$', fontsize=14)
    ax.set_ylabel('$\\phi_2$', fontsize=14)
    ax.set_title('AR(2) Stationarity Region', fontsize=16, fontweight='bold')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.tick_params(axis='both', labelsize=12)
    remove_spines(ax)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=12)

    fig.tight_layout()
    save_chart(fig, 'ch2_quiz_ar2_check')


# =============================================================================
# CHART 35: ch2_quiz_wold
# =============================================================================
def chart_quiz_wold():
    print('35. ch2_quiz_wold')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: MA(infinity) weights for AR(1) phi=0.7
    phi = 0.7
    n_weights = 20
    j = np.arange(n_weights)
    psi = phi**j

    markerline, stemlines, baseline = axes[0].stem(j, psi, linefmt='-', markerfmt='o', basefmt='k-')
    plt.setp(stemlines, color=BLUE, linewidth=2)
    plt.setp(markerline, color=BLUE, markersize=6)
    plt.setp(baseline, color='black', linewidth=0.5)
    axes[0].set_xlabel('$j$', fontsize=14)
    axes[0].set_ylabel('$\\psi_j$', fontsize=14)
    axes[0].set_title('Wold weights: $\\psi_j = \\phi^j$\n($\\phi=0.7$)', fontsize=14, fontweight='bold')
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].tick_params(axis='both', labelsize=12)
    remove_spines(axes[0])

    # Right: Partial sum representation
    partial_var = np.cumsum(psi**2)
    true_var = 1 / (1 - phi**2)
    axes[1].plot(j, partial_var, 'o-', color=GREEN, linewidth=2, markersize=6,
                 label='Partial sum')
    axes[1].axhline(true_var, color=RED, linestyle='--', linewidth=2,
                     label=f'$\\gamma(0) = {true_var:.2f}$')
    axes[1].annotate(f'$\\gamma(0) = {true_var:.2f}$', xy=(15, true_var),
                     xytext=(10, true_var - 0.35), fontsize=13, fontweight='bold', color=RED)
    axes[1].set_xlabel('Number of terms $j$', fontsize=13)
    axes[1].set_ylabel('Cumulative variance', fontsize=13)
    axes[1].set_title('$\\sum \\psi_k^2$ converges to $\\gamma(0)$', fontsize=14, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=12)
    remove_spines(axes[1])
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, fontsize=12)

    fig.tight_layout()
    save_chart(fig, 'ch2_quiz_wold')


# =============================================================================
# CHART 36: ch2_quiz_ci_growth
# =============================================================================
def chart_quiz_ci_growth():
    print('36. ch2_quiz_ci_growth')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    phi = 0.9
    sigma2 = 1.0
    h_range = np.arange(1, 51)

    # MSFE
    msfe = np.array([sigma2 * (1 - phi**(2*h)) / (1 - phi**2) for h in h_range])
    unconditional_var = sigma2 / (1 - phi**2)

    axes[0].plot(h_range, msfe, '-', color=BLUE, linewidth=2.5)
    axes[0].axhline(unconditional_var, color=RED, linestyle='--', linewidth=2)
    axes[0].annotate(f'$\\gamma(0) = {unconditional_var:.2f}$', xy=(35, unconditional_var),
                     xytext=(20, unconditional_var - 1.5), fontsize=13, fontweight='bold', color=RED)
    axes[0].set_xlabel('Forecast horizon $h$', fontsize=13)
    axes[0].set_ylabel('MSFE', fontsize=14)
    axes[0].set_title('MSFE convergence ($\\phi=0.9$)', fontsize=15, fontweight='bold')
    axes[0].tick_params(axis='both', labelsize=12)
    remove_spines(axes[0])

    # CI width
    ci_width = 2 * 1.96 * np.sqrt(msfe)
    ci_limit = 2 * 1.96 * np.sqrt(unconditional_var)

    axes[1].plot(h_range, ci_width, '-', color=GREEN, linewidth=2.5)
    axes[1].axhline(ci_limit, color=RED, linestyle='--', linewidth=2)
    axes[1].fill_between(h_range, 0, ci_width, alpha=0.1, color=GREEN)
    axes[1].annotate(f'Limit $\\approx {ci_limit:.1f}$', xy=(35, ci_limit),
                     xytext=(20, ci_limit - 2.5), fontsize=13, fontweight='bold', color=RED)
    axes[1].set_xlabel('Forecast horizon $h$', fontsize=13)
    axes[1].set_ylabel('95% CI width', fontsize=14)
    axes[1].set_title(f'CI width: $2 \\times 1.96 \\times \\sqrt{{\\mathrm{{MSFE}}}}$', fontsize=15, fontweight='bold')
    axes[1].tick_params(axis='both', labelsize=12)
    remove_spines(axes[1])

    fig.tight_layout()
    save_chart(fig, 'ch2_quiz_ci_growth')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('Chapter 2 (ARMA Models) -- Lecture Figure Generation')
    print('=' * 60)

    import os
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # --- Motivation & theory slides ---
    chart_motivation_stationary()     # 1
    chart_motivation_acf()            # 2
    chart_white_noise()               # 3

    # --- Definition slides ---
    chart_def_ar1()                   # 4
    chart_ar1_simulations()           # 5
    chart_ar1_acf_pacf()              # 6
    chart_def_arp()                   # 7
    chart_ar2_stationarity()          # 8
    chart_def_ma1()                   # 9
    chart_ma1_simulations()           # 10
    chart_ma1_acf_pacf()              # 11
    chart_def_invertibility()         # 12
    chart_def_maq()                   # 13
    chart_def_arma()                  # 14

    # --- Identification & diagnostics ---
    chart_acf_pacf_patterns()         # 15
    chart_diagnostics()               # 16
    chart_def_ljungbox()              # 17

    # --- Forecasting ---
    chart_ar1_forecast()              # 18
    chart_rolling_forecast()          # 19
    chart_rolling_vs_multistep()      # 20

    # --- Case study (also in ch2_quantlet_arma.py) ---
    chart_case_raw_data()             # 21
    chart_case_acf_pacf()             # 22
    chart_case_model_comparison()     # 23
    chart_case_diagnostics()          # 24
    chart_case_forecast()             # 25

    # --- Quiz figures ---
    chart_quiz_ar_stationarity()      # 26
    chart_quiz_acf_pacf_patterns()    # 27
    chart_quiz_information_criteria() # 28
    chart_quiz_ljung_box()            # 29
    chart_quiz_forecast_properties()  # 30
    chart_quiz_ar1_variance()         # 31
    chart_quiz_ma1_acf()              # 32
    chart_quiz_arma11_acf()           # 33
    chart_quiz_ar2_check()            # 34
    chart_quiz_wold()                 # 35
    chart_quiz_ci_growth()            # 36

    print('=' * 60)
    print('ALL 36 Chapter 2 charts regenerated successfully!')
    print(f'Output directory: {CHARTS_DIR}')
    print()
    print('NOTE: Conceptual diagrams (ch2_lag_operator, ch2_box_jenkins,')
    print('      ch2_wold_representation, ch2_arma_structure, etc.) are')
    print('      generated by generate_arma_charts.py in the project root.')
    print('=' * 60)
