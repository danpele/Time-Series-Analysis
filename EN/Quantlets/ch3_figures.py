#!/usr/bin/env python3
"""
Chapter 3: ARIMA Models -- Figure Generation Script
Generates all lecture figures for Chapter 3 of the Time Series Analysis course.
Author: Daniel Traian PELE
Date: 2025

Output: All figures saved as PNG (300 dpi) + PDF to ../../charts/ with ch3_ prefix.
Style: transparent background, no grid, legend outside bottom, no top/right spines.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try imports for statsmodels
try:
    from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf, adfuller
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not available. Some charts may fail.")

# =============================================================================
# GLOBAL STYLE SETTINGS
# =============================================================================
COLORS = ['#1A3A6E', '#CD0000', '#2E7D32', '#B5853F', '#E67E22', '#8E44AD']
BLUE, RED, GREEN, BROWN, ORANGE, PURPLE = COLORS
GRAY = '#666666'

COLORS_MULTI = [BLUE, RED, GREEN, ORANGE, PURPLE, GRAY,
                '#2980B9', '#C0392B', '#27AE60', '#D35400']

plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'savefig.transparent': True,
    'axes.grid': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.facecolor': 'none',
    'legend.framealpha': 0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.5,
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


def bottom_legend(ax, ncol=2, **kw):
    """Add legend outside bottom."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if ncol is None:
        ncol = min(len(handles), 4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=ncol, frameon=False, **kw)


# ===================================================================
# 1. ch3_motivation_nonstationary
#    Three non-stationary examples with sample mean shown
# ===================================================================
def chart_motivation_nonstationary():
    T = 300
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Stock-price-like random walk
    rw = np.cumsum(np.random.normal(0.0003, 0.01, T)) + 4.5
    axes[0].plot(rw, color=BLUE, lw=1.3, label='Stock price (log)')
    axes[0].axhline(rw.mean(), color=RED, ls='--', lw=1.2, label=f'Mean = {rw.mean():.2f}')
    axes[0].set_title('Stock price (log)')
    axes[0].set_xlabel('Time')
    bottom_legend(axes[0], ncol=1)

    # GDP-like trending series
    gdp = np.cumsum(np.random.normal(0.005, 0.008, T)) + 9.0
    axes[1].plot(gdp, color=GREEN, lw=1.3, label='GDP (log)')
    axes[1].axhline(gdp.mean(), color=RED, ls='--', lw=1.2, label=f'Mean = {gdp.mean():.2f}')
    axes[1].set_title('GDP (log)')
    axes[1].set_xlabel('Time')
    bottom_legend(axes[1], ncol=1)

    # Exchange rate random walk
    fx = np.cumsum(np.random.normal(0, 0.006, T)) + 1.1
    axes[2].plot(fx, color=ORANGE, lw=1.3, label='Exchange rate')
    axes[2].axhline(fx.mean(), color=RED, ls='--', lw=1.2, label=f'Mean = {fx.mean():.2f}')
    axes[2].set_title('Exchange rate')
    axes[2].set_xlabel('Time')
    bottom_legend(axes[2], ncol=1)

    fig.suptitle('Non-stationary data: sample mean is meaningless', fontsize=15, y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch3_motivation_nonstationary')


# ===================================================================
# 2. ch3_motivation_realworld
#    Real-world-style examples: stock, FX, interest rate
# ===================================================================
def chart_motivation_realworld():
    T = 250
    fig, axes = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True)

    stock = np.cumsum(np.random.normal(0.0005, 0.015, T)) + 4.0
    axes[0].plot(stock, color=BLUE, lw=1.3)
    axes[0].set_title('Stock prices (log) — Random walk')
    axes[0].set_ylabel('Log Price')

    fx = np.cumsum(np.random.normal(0, 0.005, T)) + 0.0
    axes[1].plot(fx, color=GREEN, lw=1.3)
    axes[1].set_title('Exchange rate — Random walk')
    axes[1].set_ylabel('Log FX')

    # Near unit root AR(1) for interest rate
    ir = np.zeros(T)
    ir[0] = 3.0
    for t in range(1, T):
        ir[t] = 0.01 + 0.995 * ir[t-1] + np.random.normal(0, 0.08)
    axes[2].plot(ir, color=ORANGE, lw=1.3)
    axes[2].set_title('Interest rate — Highly persistent')
    axes[2].set_ylabel('Rate (%)')
    axes[2].set_xlabel('Time')

    fig.tight_layout()
    save_chart(fig, 'ch3_motivation_realworld')


# ===================================================================
# 3. ch3_motivation_differencing
#    Top: original series + ACF slow decay. Bottom: differenced + ACF cutoff
# ===================================================================
def chart_motivation_differencing():
    T = 300
    rw = np.cumsum(np.random.normal(0, 1, T))
    drw = np.diff(rw)
    nlags = 30

    acf_rw  = sm_acf(rw,  nlags=nlags, fft=True)
    acf_drw = sm_acf(drw, nlags=nlags, fft=True)
    ci = 1.96 / np.sqrt(T)

    fig, axes = plt.subplots(2, 2, figsize=(13, 6.5))

    axes[0, 0].plot(rw, color=BLUE, lw=1.0)
    axes[0, 0].set_title('Original series (random walk)')
    axes[0, 0].set_xlabel('Time')

    axes[0, 1].bar(range(nlags+1), acf_rw, color=BLUE, width=0.6, alpha=0.8)
    axes[0, 1].axhline(ci, color=RED, ls='--', lw=0.8)
    axes[0, 1].axhline(-ci, color=RED, ls='--', lw=0.8)
    axes[0, 1].set_title('ACF — slow decay')
    axes[0, 1].set_xlabel('Lag')

    axes[1, 0].plot(drw, color=GREEN, lw=1.0)
    axes[1, 0].set_title('After differencing ($\\Delta Y_t$)')
    axes[1, 0].set_xlabel('Time')

    axes[1, 1].bar(range(nlags+1), acf_drw, color=GREEN, width=0.6, alpha=0.8)
    axes[1, 1].axhline(ci, color=RED, ls='--', lw=0.8)
    axes[1, 1].axhline(-ci, color=RED, ls='--', lw=0.8)
    axes[1, 1].set_title('ACF — cuts off quickly')
    axes[1, 1].set_xlabel('Lag')

    fig.tight_layout()
    save_chart(fig, 'ch3_motivation_differencing')


# ===================================================================
# 4. ch3_gdp_levels
#    US Real GDP (simulated) with upward trend
# ===================================================================
def chart_gdp_levels():
    T = 138
    t = np.arange(T)
    # Simulate quarterly log GDP with trend + noise + COVID dip
    log_gdp = 9.2 + 0.0057 * t + np.cumsum(np.random.normal(0, 0.002, T))
    # COVID dip around obs 120
    log_gdp[120:125] -= np.array([0.10, 0.05, 0.02, 0.01, 0.005])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    quarters = [f'{1990 + i//4}Q{i%4+1}' for i in range(T)]
    ax.plot(t, log_gdp, color=BLUE, lw=1.5, label='Log US Real GDP')
    ax.axhline(log_gdp.mean(), color=RED, ls='--', lw=1, alpha=0.7,
               label=f'Mean = {log_gdp.mean():.2f}')
    # Mark COVID
    ax.annotate('COVID-19', xy=(121, log_gdp[121]), fontsize=10,
                xytext=(100, log_gdp[121]-0.06),
                arrowprops=dict(arrowstyle='->', color=RED), color=RED)
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Log GDP')
    ax.set_title('US Real GDP — Non-stationary series with trend')
    # Sparse x-ticks
    tick_idx = list(range(0, T, 20)) + [T-1]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([quarters[i] for i in tick_idx], rotation=45, fontsize=9)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.tight_layout()
    save_chart(fig, 'ch3_gdp_levels')


# ===================================================================
# 5. ch3_trend_comparison
#    Left: deterministic trend. Right: stochastic trend (random walk)
# ===================================================================
def chart_trend_comparison():
    T = 200
    t = np.arange(T)
    eps = np.random.normal(0, 1, T)

    # Deterministic trend
    det = 0.05 * t + eps
    # Stochastic trend (random walk)
    sto = np.cumsum(np.random.normal(0.05, 1, T))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(det, color=BLUE, lw=1.2, label='$Y_t = \\alpha + \\beta t + \\varepsilon_t$')
    axes[0].plot(0.05 * t, color=RED, ls='--', lw=1.5, label='Deterministic trend')
    axes[0].set_title('Deterministic Trend')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')
    bottom_legend(axes[0], ncol=1)

    axes[1].plot(sto, color=GREEN, lw=1.2, label='$Y_t = Y_{t-1} + \\varepsilon_t$')
    axes[1].set_title('Stochastic Trend (Unit Root)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('$Y_t$')
    bottom_legend(axes[1], ncol=1)

    fig.tight_layout()
    save_chart(fig, 'ch3_trend_comparison')


# ===================================================================
# 6. ch3_def_random_walk
#    Left: multiple RW trajectories. Right: variance growth
# ===================================================================
def chart_def_random_walk():
    T = 200
    n_paths = 50
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    paths = np.cumsum(np.random.normal(0, 1, (n_paths, T)), axis=1)
    for i in range(n_paths):
        axes[0].plot(paths[i], lw=0.5, alpha=0.4, color=BLUE)
    axes[0].axhline(0, color=RED, ls='--', lw=1, label='$E[Y_t] = 0$')
    axes[0].set_title('Random walk trajectories')
    axes[0].set_xlabel('Time ($t$)')
    axes[0].set_ylabel('$Y_t$')
    bottom_legend(axes[0], ncol=1)

    # Variance: theoretical vs empirical
    t_arr = np.arange(1, T+1)
    emp_var = np.var(paths, axis=0)
    axes[1].plot(t_arr, emp_var, color=BLUE, lw=1.5, label='Empirical variance')
    axes[1].plot(t_arr, t_arr * 1.0, color=RED, ls='--', lw=1.5,
                 label='$Var(Y_t) = t\\sigma^2$')
    axes[1].set_title('Variance growth')
    axes[1].set_xlabel('Time ($t$)')
    axes[1].set_ylabel('Variance')
    bottom_legend(axes[1], ncol=1)

    fig.tight_layout()
    save_chart(fig, 'ch3_def_random_walk')


# ===================================================================
# 7. ch3_def_random_walk_drift
#    Left: pure RW (no drift). Right: RW with drift
# ===================================================================
def chart_def_random_walk_drift():
    T = 200
    n_paths = 8
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i in range(n_paths):
        rw = np.cumsum(np.random.normal(0, 1, T))
        axes[0].plot(rw, lw=1.0, alpha=0.7, color=COLORS_MULTI[i % len(COLORS_MULTI)])
    axes[0].axhline(0, color=GRAY, ls='--', lw=1)
    axes[0].set_title('Random walk (no drift, $\\mu=0$)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')

    mu = 0.15
    for i in range(n_paths):
        rw_d = np.cumsum(np.random.normal(mu, 1, T))
        axes[1].plot(rw_d, lw=1.0, alpha=0.7, color=COLORS_MULTI[i % len(COLORS_MULTI)])
    axes[1].plot(mu * np.arange(T), color=RED, ls='--', lw=2, label=f'Drift trend ($\\mu={mu}$)')
    axes[1].set_title(f'Random walk with drift ($\\mu={mu}$)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('$Y_t$')
    bottom_legend(axes[1], ncol=1)

    fig.tight_layout()
    save_chart(fig, 'ch3_def_random_walk_drift')


# ===================================================================
# 8. ch3_random_walk
#    Left: pure RW simulations. Right: RW with drift simulations
# ===================================================================
def chart_random_walk():
    T = 300
    n_paths = 10
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i in range(n_paths):
        rw = np.cumsum(np.random.normal(0, 1, T))
        axes[0].plot(rw, lw=0.8, alpha=0.7, color=COLORS_MULTI[i % len(COLORS_MULTI)])
    axes[0].axhline(0, color=GRAY, ls='--', lw=1)
    axes[0].set_title('Random walk: no drift')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')

    mu = 0.1
    for i in range(n_paths):
        rwd = np.cumsum(np.random.normal(mu, 1, T))
        axes[1].plot(rwd, lw=0.8, alpha=0.7, color=COLORS_MULTI[i % len(COLORS_MULTI)])
    axes[1].plot(mu * np.arange(T), 'k--', lw=2, label=f'$E[Y_t] = \\mu t$ ($\\mu={mu}$)')
    axes[1].set_title(f'Random walk: with drift ($\\mu={mu}$)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('$Y_t$')
    bottom_legend(axes[1], ncol=1)

    fig.tight_layout()
    save_chart(fig, 'ch3_random_walk')


# ===================================================================
# 9. ch3_variance_growth
#    Left: fan of trajectories. Right: variance grows linearly
# ===================================================================
def chart_variance_growth():
    T = 200
    n_paths = 200
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    paths = np.cumsum(np.random.normal(0, 1, (n_paths, T)), axis=1)
    # Fan chart
    for i in range(min(n_paths, 80)):
        axes[0].plot(paths[i], lw=0.3, alpha=0.25, color=BLUE)
    pcts = [5, 25, 75, 95]
    for p, ls in zip([5, 95], ['--', '--']):
        axes[0].plot(np.percentile(paths, p, axis=0), color=RED, ls=ls, lw=1.2,
                     label=f'Percentile {p}%')
    axes[0].plot(np.percentile(paths, 50, axis=0), color=RED, lw=1.5, label='Median')
    axes[0].set_title('Fan of trajectories')
    axes[0].set_xlabel('Time ($t$)')
    axes[0].set_ylabel('$Y_t$')
    bottom_legend(axes[0], ncol=3)

    t_arr = np.arange(1, T+1)
    emp_var = np.var(paths, axis=0)
    axes[1].plot(t_arr, emp_var, color=BLUE, lw=1.5, label='Empirical variance')
    axes[1].plot(t_arr, t_arr, color=RED, ls='--', lw=1.5,
                 label='Theoretical: $Var(Y_t)=t\\sigma^2$')
    axes[1].set_title('Variance grows linearly')
    axes[1].set_xlabel('Time ($t$)')
    axes[1].set_ylabel('$Var(Y_t)$')
    bottom_legend(axes[1], ncol=1)

    fig.tight_layout()
    save_chart(fig, 'ch3_variance_growth')


# ===================================================================
# 10. ch3_def_integrated
#     Three subplots: I(0), I(1), I(2) processes
# ===================================================================
def chart_def_integrated():
    T = 200
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # I(0): stationary AR(1)
    i0 = np.zeros(T)
    for t in range(1, T):
        i0[t] = 0.5 * i0[t-1] + np.random.normal(0, 1)
    axes[0].plot(i0, color=GREEN, lw=1.2)
    axes[0].axhline(0, color=GRAY, ls='--', lw=1)
    axes[0].set_title('$I(0)$ — Stationary')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')
    axes[0].text(0.05, 0.92, 'No differencing', transform=axes[0].transAxes,
                 fontsize=11, color=GREEN, fontweight='bold', va='top')

    # I(1): random walk
    i1 = np.cumsum(np.random.normal(0, 1, T))
    axes[1].plot(i1, color=BLUE, lw=1.2)
    axes[1].set_title('$I(1)$ — One difference needed')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('$Y_t$')
    axes[1].text(0.05, 0.92, '$\\Delta Y_t$ stationary', transform=axes[1].transAxes,
                 fontsize=11, color=BLUE, fontweight='bold', va='top')

    # I(2): cumsum of cumsum
    i2 = np.cumsum(np.cumsum(np.random.normal(0, 1, T)))
    axes[2].plot(i2, color=RED, lw=1.2)
    axes[2].set_title('$I(2)$ — Two differences needed')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('$Y_t$')
    axes[2].text(0.05, 0.92, '$\\Delta^2 Y_t$ stationary', transform=axes[2].transAxes,
                 fontsize=11, color=RED, fontweight='bold', va='top')

    fig.tight_layout()
    save_chart(fig, 'ch3_def_integrated')


# ===================================================================
# 11. ch3_def_difference
#     Left: non-stationary series. Right: after first differencing
# ===================================================================
def chart_def_difference():
    T = 200
    rw = np.cumsum(np.random.normal(0.03, 1, T))
    drw = np.diff(rw)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(rw, color=BLUE, lw=1.3, label='$Y_t$ (non-stationary)')
    axes[0].set_title('Original series — $I(1)$')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')
    bottom_legend(axes[0], ncol=1)

    axes[1].plot(drw, color=GREEN, lw=1.0, label='$\\Delta Y_t = Y_t - Y_{t-1}$')
    axes[1].axhline(0, color=GRAY, ls='--', lw=1)
    axes[1].set_title('After first difference — $I(0)$')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('$\\Delta Y_t$')
    bottom_legend(axes[1], ncol=1)

    fig.tight_layout()
    save_chart(fig, 'ch3_def_difference')


# ===================================================================
# 12. ch3_acf_nonstationary
#     Top: ACF of random walk (slow decay). Bottom: ACF after differencing
# ===================================================================
def chart_acf_nonstationary():
    T = 300
    nlags = 25
    rw = np.cumsum(np.random.normal(0, 1, T))
    drw = np.diff(rw)
    acf_rw  = sm_acf(rw,  nlags=nlags, fft=True)
    acf_drw = sm_acf(drw, nlags=nlags, fft=True)
    ci = 1.96 / np.sqrt(T)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].bar(range(nlags+1), acf_rw, color=BLUE, width=0.6, alpha=0.85)
    axes[0].axhline(ci, color=RED, ls='--', lw=0.8, label='95% CI')
    axes[0].axhline(-ci, color=RED, ls='--', lw=0.8)
    axes[0].set_title('ACF — Random walk (slow decay $\\Rightarrow$ unit root)')
    axes[0].set_ylabel('ACF')
    axes[0].set_xlabel('Lag')
    bottom_legend(axes[0], ncol=1)

    axes[1].bar(range(nlags+1), acf_drw, color=GREEN, width=0.6, alpha=0.85)
    axes[1].axhline(ci, color=RED, ls='--', lw=0.8, label='95% CI')
    axes[1].axhline(-ci, color=RED, ls='--', lw=0.8)
    axes[1].set_title('ACF — After differencing (cuts off $\\Rightarrow$ stationary)')
    axes[1].set_ylabel('ACF')
    axes[1].set_xlabel('Lag')
    bottom_legend(axes[1], ncol=1)

    fig.tight_layout()
    save_chart(fig, 'ch3_acf_nonstationary')


# ===================================================================
# 13. ch3_differencing
#     Left: GDP in levels (trending). Right: GDP growth rate (stationary)
# ===================================================================
def chart_differencing():
    T = 138
    t = np.arange(T)
    log_gdp = 9.2 + 0.0057 * t + np.cumsum(np.random.normal(0, 0.002, T))
    log_gdp[120:125] -= np.array([0.10, 0.05, 0.02, 0.01, 0.005])
    growth = np.diff(log_gdp) * 100  # percentage

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(log_gdp, color=BLUE, lw=1.3, label='Log GDP')
    axes[0].set_title('GDP in levels (non-stationary)')
    axes[0].set_xlabel('Quarter')
    axes[0].set_ylabel('Log GDP')
    bottom_legend(axes[0], ncol=1)

    axes[1].plot(growth, color=GREEN, lw=1.0, label='$\\Delta$ Log GDP (%)')
    axes[1].axhline(0, color=GRAY, ls='--', lw=1)
    axes[1].set_title('GDP growth rate (stationary)')
    axes[1].set_xlabel('Quarter')
    axes[1].set_ylabel('Growth (%)')
    bottom_legend(axes[1], ncol=1)

    fig.tight_layout()
    save_chart(fig, 'ch3_differencing')


# ===================================================================
# 14. ch3_def_arima
#     Top: ARIMA series (non-stationary). Bottom: differenced + ACF/PACF
# ===================================================================
def chart_def_arima():
    T = 300
    # Simulate ARIMA(1,1,1): phi=0.6, theta=0.4
    eps = np.random.normal(0, 1, T+1)
    z = np.zeros(T+1)  # differenced series (ARMA(1,1))
    for t in range(1, T+1):
        z[t] = 0.6 * z[t-1] + eps[t] + 0.4 * eps[t-1]
    y = np.cumsum(z)  # integrate

    dy = np.diff(y)
    nlags = 20
    acf_dy  = sm_acf(dy[1:], nlags=nlags, fft=True)
    pacf_dy = sm_pacf(dy[1:], nlags=nlags, method='ywm')
    ci = 1.96 / np.sqrt(len(dy))

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.plot(y, color=BLUE, lw=1.0, label='ARIMA(1,1,1) — original series')
    ax_top.set_title('Original ARIMA(1,1,1) series (non-stationary)')
    ax_top.set_xlabel('Time')
    ax_top.set_ylabel('$Y_t$')
    bottom_legend(ax_top, ncol=1)

    ax_acf = fig.add_subplot(gs[1, 0])
    ax_acf.bar(range(nlags+1), acf_dy, color=GREEN, width=0.6, alpha=0.85)
    ax_acf.axhline(ci, color=RED, ls='--', lw=0.8)
    ax_acf.axhline(-ci, color=RED, ls='--', lw=0.8)
    ax_acf.set_title('ACF — differenced series')
    ax_acf.set_xlabel('Lag')
    ax_acf.set_ylabel('ACF')

    ax_pacf = fig.add_subplot(gs[1, 1])
    ax_pacf.bar(range(nlags+1), pacf_dy, color=PURPLE, width=0.6, alpha=0.85)
    ax_pacf.axhline(ci, color=RED, ls='--', lw=0.8)
    ax_pacf.axhline(-ci, color=RED, ls='--', lw=0.8)
    ax_pacf.set_title('PACF — differenced series')
    ax_pacf.set_xlabel('Lag')
    ax_pacf.set_ylabel('PACF')

    save_chart(fig, 'ch3_def_arima')


# ===================================================================
# 15. ch3_def_adf
#     Left: stationary series (ADF rejects). Right: non-stationary (ADF fails)
# ===================================================================
def chart_def_adf():
    T = 200
    # Stationary AR(1)
    stat = np.zeros(T)
    for t in range(1, T):
        stat[t] = 0.5 * stat[t-1] + np.random.normal(0, 1)
    # Non-stationary random walk
    nonstat = np.cumsum(np.random.normal(0, 1, T))

    adf_s = adfuller(stat, autolag='AIC')
    adf_n = adfuller(nonstat, autolag='AIC')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(stat, color=GREEN, lw=1.2)
    axes[0].axhline(0, color=GRAY, ls='--', lw=0.8)
    axes[0].set_title('Stationary — ADF rejects $H_0$')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')
    txt_s = f'ADF stat = {adf_s[0]:.2f}\np-value = {adf_s[1]:.4f}'
    axes[0].text(0.05, 0.95, txt_s, transform=axes[0].transAxes, fontsize=11,
                 va='top', bbox=dict(boxstyle='round', facecolor=GREEN, alpha=0.15))

    axes[1].plot(nonstat, color=RED, lw=1.2)
    axes[1].set_title('Non-stationary — ADF fails to reject $H_0$')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('$Y_t$')
    txt_n = f'ADF stat = {adf_n[0]:.2f}\np-value = {adf_n[1]:.4f}'
    axes[1].text(0.05, 0.95, txt_n, transform=axes[1].transAxes, fontsize=11,
                 va='top', bbox=dict(boxstyle='round', facecolor=RED, alpha=0.15))

    fig.tight_layout()
    save_chart(fig, 'ch3_def_adf')


# ===================================================================
# 16. ch3_rolling_forecast
#     Rolling window forecast illustration
# ===================================================================
def chart_rolling_forecast():
    T = 200
    window = 80
    rw = np.cumsum(np.random.normal(0.01, 0.5, T))

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(range(T), rw, color=BLUE, lw=1.3, label='Actual data')

    # Show 4 example windows
    starts = [20, 50, 80, 110]
    win_colors = [GREEN, ORANGE, PURPLE, RED]
    for s, c in zip(starts, win_colors):
        end = s + window
        if end + 1 < T:
            # Training window
            ax.axvspan(s, end, alpha=0.08, color=c)
            ax.plot([s, s], [rw.min()-1, rw.max()+1], color=c, ls=':', lw=0.8, alpha=0.6)
            # Forecast point
            ax.plot(end, rw[end], 'o', color=c, ms=6, zorder=5)
            ax.annotate(f'  F{starts.index(s)+1}', xy=(end, rw[end]), fontsize=9, color=c)

    ax.set_title('Rolling Forecast — Sliding Window')
    ax.set_xlabel('Time')
    ax.set_ylabel('$Y_t$')
    ax.set_ylim(rw.min()-2, rw.max()+2)
    # Custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=BLUE, lw=1.5, label='Actual data'),
        Patch(facecolor=GREEN, alpha=0.2, label='Training window'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=ORANGE,
               markersize=6, label='Forecast point')
    ]
    ax.legend(handles=legend_elements, loc='upper center',
              bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    save_chart(fig, 'ch3_rolling_forecast')


# ===================================================================
# 17. ch3_acf_pacf  (Case study: ACF/PACF of differenced GDP)
# ===================================================================
def chart_acf_pacf():
    T = 138
    t_arr = np.arange(T)
    log_gdp = 9.2 + 0.0057 * t_arr + np.cumsum(np.random.normal(0, 0.002, T))
    log_gdp[120:125] -= np.array([0.10, 0.05, 0.02, 0.01, 0.005])
    growth = np.diff(log_gdp)

    nlags = 20
    acf_v  = sm_acf(growth, nlags=nlags, fft=True)
    pacf_v = sm_pacf(growth, nlags=nlags, method='ywm')
    ci = 1.96 / np.sqrt(len(growth))

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    # Top-left: ACF of levels
    acf_lev = sm_acf(log_gdp, nlags=nlags, fft=True)
    axes[0, 0].bar(range(nlags+1), acf_lev, color=BLUE, width=0.6, alpha=0.85)
    axes[0, 0].axhline(ci, color=RED, ls='--', lw=0.8)
    axes[0, 0].axhline(-ci, color=RED, ls='--', lw=0.8)
    axes[0, 0].set_title('ACF — GDP levels')
    axes[0, 0].set_xlabel('Lag')
    axes[0, 0].set_ylabel('ACF')

    # Top-right: PACF of levels
    pacf_lev = sm_pacf(log_gdp, nlags=nlags, method='ywm')
    axes[0, 1].bar(range(nlags+1), pacf_lev, color=BLUE, width=0.6, alpha=0.85)
    axes[0, 1].axhline(ci, color=RED, ls='--', lw=0.8)
    axes[0, 1].axhline(-ci, color=RED, ls='--', lw=0.8)
    axes[0, 1].set_title('PACF — GDP levels')
    axes[0, 1].set_xlabel('Lag')
    axes[0, 1].set_ylabel('PACF')

    # Bottom-left: ACF of differenced
    axes[1, 0].bar(range(nlags+1), acf_v, color=GREEN, width=0.6, alpha=0.85)
    axes[1, 0].axhline(ci, color=RED, ls='--', lw=0.8)
    axes[1, 0].axhline(-ci, color=RED, ls='--', lw=0.8)
    axes[1, 0].set_title('ACF — Differenced GDP ($\\Delta$ log GDP)')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')

    # Bottom-right: PACF of differenced
    axes[1, 1].bar(range(nlags+1), pacf_v, color=GREEN, width=0.6, alpha=0.85)
    axes[1, 1].axhline(ci, color=RED, ls='--', lw=0.8)
    axes[1, 1].axhline(-ci, color=RED, ls='--', lw=0.8)
    axes[1, 1].set_title('PACF — Differenced GDP ($\\Delta$ log GDP)')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('PACF')

    fig.tight_layout()
    save_chart(fig, 'ch3_acf_pacf')


# ===================================================================
# 18. ch3_diagnostics  (Case study: residual diagnostics 2x2)
# ===================================================================
def chart_diagnostics():
    T = 138
    t_arr = np.arange(T)
    log_gdp = 9.2 + 0.0057 * t_arr + np.cumsum(np.random.normal(0, 0.002, T))
    log_gdp[120:125] -= np.array([0.10, 0.05, 0.02, 0.01, 0.005])
    growth = np.diff(log_gdp)

    # Fit ARIMA(1,1,1) on differenced series
    model = ARIMA(log_gdp, order=(1, 1, 1)).fit()
    resid = model.resid[1:]  # skip first

    nlags = 20
    acf_r = sm_acf(resid, nlags=nlags, fft=True)
    ci = 1.96 / np.sqrt(len(resid))

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    # Residuals time series
    axes[0, 0].plot(resid, color=BLUE, lw=0.8)
    axes[0, 0].axhline(0, color=RED, ls='--', lw=1)
    axes[0, 0].set_title('Standardized residuals')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Residual')

    # Histogram
    axes[0, 1].hist(resid, bins=25, density=True, color=BLUE, alpha=0.7, edgecolor='white')
    x_range = np.linspace(resid.min(), resid.max(), 100)
    axes[0, 1].plot(x_range, stats.norm.pdf(x_range, resid.mean(), resid.std()),
                    color=RED, lw=1.5, label='Normal')
    axes[0, 1].set_title('Histogram of residuals')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Density')
    bottom_legend(axes[0, 1], ncol=1)

    # ACF of residuals
    axes[1, 0].bar(range(nlags+1), acf_r, color=GREEN, width=0.6, alpha=0.85)
    axes[1, 0].axhline(ci, color=RED, ls='--', lw=0.8)
    axes[1, 0].axhline(-ci, color=RED, ls='--', lw=0.8)
    axes[1, 0].set_title('ACF of residuals')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')

    # Q-Q plot
    stats.probplot(resid, dist='norm', plot=axes[1, 1])
    axes[1, 1].get_lines()[0].set(color=BLUE, markersize=3)
    axes[1, 1].get_lines()[1].set(color=RED, lw=1.5)
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].set_xlabel('Theoretical quantiles')
    axes[1, 1].set_ylabel('Sample quantiles')

    fig.tight_layout()
    save_chart(fig, 'ch3_diagnostics')


# ===================================================================
# 19. ch3_arima_forecast  (Case study: forecast with confidence intervals)
# ===================================================================
def chart_arima_forecast():
    T = 138
    t_arr = np.arange(T)
    log_gdp = 9.2 + 0.0057 * t_arr + np.cumsum(np.random.normal(0, 0.002, T))
    log_gdp[120:125] -= np.array([0.10, 0.05, 0.02, 0.01, 0.005])

    # Split: train on first 126 obs, test on last 12
    train = log_gdp[:126]
    test  = log_gdp[126:]
    model = ARIMA(train, order=(1, 1, 1)).fit()

    # Forecast 12 steps
    fc = model.get_forecast(steps=12)
    fc_mean = fc.predicted_mean
    fc_ci   = fc.conf_int(alpha=0.05)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot last 40 training points + test
    show_from = 90
    ax.plot(range(show_from, 126), train[show_from:], color=BLUE, lw=1.5,
            label='Training data')
    ax.plot(range(126, 138), test, color=GRAY, lw=1.5, ls='--',
            label='Actual data (test)')
    ax.plot(range(126, 138), fc_mean, color=RED, lw=1.5,
            label='ARIMA(1,1,1) forecast')
    # fc_ci may be DataFrame or ndarray
    try:
        ci_lo = fc_ci.iloc[:, 0]
        ci_hi = fc_ci.iloc[:, 1]
    except AttributeError:
        ci_lo = fc_ci[:, 0]
        ci_hi = fc_ci[:, 1]
    ax.fill_between(range(126, 138), ci_lo, ci_hi,
                    color=RED, alpha=0.15, label='95% CI')
    ax.axvline(126, color=GRAY, ls=':', lw=1, alpha=0.7)
    ax.set_title('ARIMA(1,1,1) Forecast — US Real GDP')
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Log GDP')
    bottom_legend(ax, ncol=2)
    fig.tight_layout()
    save_chart(fig, 'ch3_arima_forecast')


# ===================================================================
# 20. ch3_adf_test  (Case study: ADF results for levels vs differenced)
# ===================================================================
def chart_adf_test():
    T = 138
    t_arr = np.arange(T)
    log_gdp = 9.2 + 0.0057 * t_arr + np.cumsum(np.random.normal(0, 0.002, T))
    log_gdp[120:125] -= np.array([0.10, 0.05, 0.02, 0.01, 0.005])
    growth = np.diff(log_gdp)

    adf_lev = adfuller(log_gdp, autolag='AIC')
    adf_dif = adfuller(growth, autolag='AIC')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Levels
    axes[0].plot(log_gdp, color=BLUE, lw=1.3)
    axes[0].set_title('Log GDP — Levels')
    axes[0].set_xlabel('Quarter')
    axes[0].set_ylabel('Log GDP')
    info_l = (f'ADF stat = {adf_lev[0]:.2f}\n'
              f'p-value = {adf_lev[1]:.3f}\n'
              f'CV 1% = {adf_lev[4]["1%"]:.2f}\n'
              f'CV 5% = {adf_lev[4]["5%"]:.2f}')
    axes[0].text(0.05, 0.95, info_l, transform=axes[0].transAxes, fontsize=10.5,
                 va='top', bbox=dict(boxstyle='round', facecolor=RED, alpha=0.12))
    axes[0].text(0.05, 0.48, 'Cannot reject $H_0$\n$\\rightarrow$ Unit root',
                 transform=axes[0].transAxes, fontsize=11, color=RED, fontweight='bold')

    # Differenced
    axes[1].plot(growth, color=GREEN, lw=1.0)
    axes[1].axhline(0, color=GRAY, ls='--', lw=0.8)
    axes[1].set_title('$\\Delta$ Log GDP — Differenced')
    axes[1].set_xlabel('Quarter')
    axes[1].set_ylabel('$\\Delta$ Log GDP')
    info_d = (f'ADF stat = {adf_dif[0]:.2f}\n'
              f'p-value = {adf_dif[1]:.4f}\n'
              f'CV 1% = {adf_dif[4]["1%"]:.2f}\n'
              f'CV 5% = {adf_dif[4]["5%"]:.2f}')
    axes[1].text(0.05, 0.95, info_d, transform=axes[1].transAxes, fontsize=10.5,
                 va='top', bbox=dict(boxstyle='round', facecolor=GREEN, alpha=0.12))
    axes[1].text(0.05, 0.48, 'Reject $H_0$\n$\\rightarrow$ Stationary',
                 transform=axes[1].transAxes, fontsize=11, color=GREEN, fontweight='bold')

    fig.tight_layout()
    save_chart(fig, 'ch3_adf_test')


# ===================================================================
# 21. ch3_quiz1_rw_variance
#     Quiz: RW variance grows linearly
# ===================================================================
def chart_quiz1_rw_variance():
    T = 200
    n_paths = 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    paths = np.cumsum(np.random.normal(0, 1, (n_paths, T)), axis=1)
    for i in range(min(n_paths, 40)):
        axes[0].plot(paths[i], lw=0.3, alpha=0.3, color=BLUE)
    axes[0].fill_between(range(T),
                         np.percentile(paths, 5, axis=0),
                         np.percentile(paths, 95, axis=0),
                         alpha=0.15, color=BLUE, label='Percentile 5-95%')
    axes[0].axhline(0, color=RED, ls='--', lw=1)
    axes[0].set_title('Random walk trajectories')
    axes[0].set_xlabel('Time ($t$)')
    axes[0].set_ylabel('$Y_t$')
    bottom_legend(axes[0], ncol=1)

    t_arr = np.arange(1, T+1)
    emp_var = np.var(paths, axis=0)
    axes[1].plot(t_arr, emp_var, color=BLUE, lw=1.5, label='Empirical variance')
    axes[1].plot(t_arr, t_arr, color=RED, ls='--', lw=2,
                 label='$Var(Y_t) = t \\cdot \\sigma^2$')
    axes[1].set_title('Answer: (B) $Var(Y_t) = t \\cdot \\sigma^2$')
    axes[1].set_xlabel('Time ($t$)')
    axes[1].set_ylabel('Variance')
    bottom_legend(axes[1], ncol=1)

    fig.tight_layout()
    save_chart(fig, 'ch3_quiz1_rw_variance')


# ===================================================================
# 22. ch3_quiz2_differencing
#     Quiz: I(2) needs 2 differences to become stationary
# ===================================================================
def chart_quiz2_differencing():
    T = 200
    eps = np.random.normal(0, 1, T)
    i2 = np.cumsum(np.cumsum(eps))
    d1 = np.diff(i2)
    d2 = np.diff(d1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].plot(i2, color=RED, lw=1.2)
    axes[0].set_title('$Y_t \\sim I(2)$ — Original')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')
    axes[0].text(0.05, 0.92, 'Non-stationary', transform=axes[0].transAxes,
                 fontsize=11, color=RED, fontweight='bold', va='top')

    axes[1].plot(d1, color=ORANGE, lw=1.0)
    axes[1].axhline(0, color=GRAY, ls='--', lw=0.8)
    axes[1].set_title('$\\Delta Y_t \\sim I(1)$ — One difference')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('$\\Delta Y_t$')
    axes[1].text(0.05, 0.92, 'Still non-stationary', transform=axes[1].transAxes,
                 fontsize=11, color=ORANGE, fontweight='bold', va='top')

    axes[2].plot(d2, color=GREEN, lw=1.0)
    axes[2].axhline(0, color=GRAY, ls='--', lw=0.8)
    axes[2].set_title('$\\Delta^2 Y_t \\sim I(0)$ — Two differences')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('$\\Delta^2 Y_t$')
    axes[2].text(0.05, 0.92, 'Stationary!', transform=axes[2].transAxes,
                 fontsize=11, color=GREEN, fontweight='bold', va='top')

    fig.suptitle('Correct answer: (C) Two differences needed', fontsize=13, y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch3_quiz2_differencing')


# ===================================================================
# 23. ch3_quiz3_adf_test
#     Quiz: ADF test interpretation — stat = -2.1
# ===================================================================
def chart_quiz3_adf_test():
    fig, ax = plt.subplots(figsize=(10, 5))

    # Critical values (with constant, no trend)
    cvs = {'1%': -3.43, '5%': -2.86, '10%': -2.57}
    test_stat = -2.1

    # Draw number line
    x_range = np.linspace(-4.5, 0, 500)
    ax.axhline(0.5, color=GRAY, lw=0.5, alpha=0.3)

    # Rejection regions
    ax.axvspan(-4.5, cvs['1%'], alpha=0.15, color=RED, label='Rejection 1%')
    ax.axvspan(cvs['1%'], cvs['5%'], alpha=0.12, color=ORANGE, label='Rejection 5%')
    ax.axvspan(cvs['5%'], cvs['10%'], alpha=0.10, color='#F1C40F', label='Rejection 10%')
    ax.axvspan(cvs['10%'], 0, alpha=0.08, color=GREEN, label='Cannot reject $H_0$')

    # Critical values as vertical lines
    for label, val in cvs.items():
        ax.axvline(val, color=GRAY, ls='--', lw=1)
        ax.text(val, 1.05, f'CV {label}\n({val})', ha='center', fontsize=10, color=GRAY)

    # Test statistic
    ax.axvline(test_stat, color=BLUE, lw=2.5)
    ax.plot(test_stat, 0.5, 'D', color=BLUE, ms=10, zorder=5)
    ax.text(test_stat + 0.05, 0.7, f'Stat = {test_stat}', fontsize=12,
            color=BLUE, fontweight='bold')

    ax.set_xlim(-4.5, 0.3)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.set_xlabel('ADF Statistic')
    ax.set_title('Answer: (C) Cannot reject $H_0$ — Statistic $-2.1 > -2.57$ (CV 10%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=4, frameon=False, fontsize=10)

    fig.tight_layout()
    save_chart(fig, 'ch3_quiz3_adf_test')


# ===================================================================
# 24. ch3_quiz4_acf_decay
#     Quiz: ACF exponential decay for AR(1) (differenced ARIMA(1,1,0))
# ===================================================================
def chart_quiz4_acf_decay():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    nlags = 15
    phi = 0.7
    k = np.arange(nlags + 1)

    # Theoretical ACF
    acf_th = phi ** k
    axes[0].bar(k, acf_th, color=BLUE, width=0.6, alpha=0.85)
    axes[0].set_title(f'Theoretical ACF — AR(1), $\\phi_1 = {phi}$')
    axes[0].set_xlabel('Lag ($k$)')
    axes[0].set_ylabel('$\\rho_k = \\phi_1^k$')
    axes[0].text(0.5, 0.85, 'Exponential decay',
                 transform=axes[0].transAxes, fontsize=11, ha='center',
                 color=BLUE, fontweight='bold')

    # Simulated
    T = 500
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = phi * y[t-1] + np.random.normal(0, 1)
    acf_emp = sm_acf(y, nlags=nlags, fft=True)
    ci = 1.96 / np.sqrt(T)

    axes[1].bar(k, acf_emp, color=GREEN, width=0.6, alpha=0.85, label='Empirical ACF')
    axes[1].plot(k, acf_th, 'o-', color=RED, ms=4, lw=1.2, label='Theoretical ACF')
    axes[1].axhline(ci, color=GRAY, ls='--', lw=0.8)
    axes[1].axhline(-ci, color=GRAY, ls='--', lw=0.8)
    axes[1].set_title(f'Empirical vs theoretical ACF (T={T})')
    axes[1].set_xlabel('Lag ($k$)')
    axes[1].set_ylabel('ACF')
    bottom_legend(axes[1], ncol=2)

    fig.suptitle('Answer: (B) ACF decays exponentially', fontsize=13, y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch3_quiz4_acf_decay')


# ===================================================================
# 25. ch3_quiz5_forecast_ci
#     Quiz: CI widens unboundedly for I(1) vs bounded for I(0)
# ===================================================================
def chart_quiz5_forecast_ci():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    h_max = 30
    h = np.arange(1, h_max + 1)
    sigma2 = 1.0

    # I(1): variance = h * sigma^2, CI width = 2 * 1.96 * sqrt(h*sigma^2)
    ci_width_i1 = 2 * 1.96 * np.sqrt(h * sigma2)
    axes[0].fill_between(h, -ci_width_i1/2, ci_width_i1/2, alpha=0.2, color=RED)
    axes[0].plot(h, ci_width_i1/2, color=RED, lw=1.5)
    axes[0].plot(h, -ci_width_i1/2, color=RED, lw=1.5)
    axes[0].axhline(0, color=BLUE, lw=1.5, label='Point forecast')
    axes[0].set_title('$I(1)$: CI widens unboundedly')
    axes[0].set_xlabel('Horizon ($h$)')
    axes[0].set_ylabel('95% CI')
    axes[0].text(0.55, 0.85, '$\\propto \\sqrt{h}$', transform=axes[0].transAxes,
                 fontsize=13, color=RED, fontweight='bold')
    bottom_legend(axes[0], ncol=1)

    # I(0) AR(1): variance converges
    phi = 0.7
    psi_cum = np.array([sum(phi**j for j in range(i+1)) for i in range(h_max)])
    var_i0 = sigma2 * np.array([sum(phi**(2*j) for j in range(i+1)) for i in range(h_max)])
    ci_width_i0 = 2 * 1.96 * np.sqrt(var_i0)
    limit = 2 * 1.96 * np.sqrt(sigma2 / (1 - phi**2))

    axes[1].fill_between(h, -ci_width_i0/2, ci_width_i0/2, alpha=0.2, color=GREEN)
    axes[1].plot(h, ci_width_i0/2, color=GREEN, lw=1.5)
    axes[1].plot(h, -ci_width_i0/2, color=GREEN, lw=1.5)
    axes[1].axhline(0, color=BLUE, lw=1.5, label='Point forecast')
    axes[1].axhline(limit/2, color=ORANGE, ls='--', lw=1.2, label='CI limit')
    axes[1].axhline(-limit/2, color=ORANGE, ls='--', lw=1.2)
    axes[1].set_title('$I(0)$: CI converges to a limit')
    axes[1].set_xlabel('Horizon ($h$)')
    axes[1].set_ylabel('95% CI')
    bottom_legend(axes[1], ncol=2)

    fig.suptitle('Answer: (C) Widens unboundedly for I(1)', fontsize=13, y=1.02)
    fig.tight_layout()
    save_chart(fig, 'ch3_quiz5_forecast_ci')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print('=' * 60)
    print('Chapter 3 (ARIMA Models) -- Lecture Figure Generation')
    print('=' * 60)

    import os
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # --- Motivation slides ---
    chart_motivation_nonstationary()      # 1
    chart_motivation_realworld()          # 2
    chart_motivation_differencing()       # 3

    # --- Definition slides ---
    chart_gdp_levels()                    # 4
    chart_trend_comparison()              # 5
    chart_def_random_walk()               # 6
    chart_def_random_walk_drift()         # 7
    chart_random_walk()                   # 8
    chart_variance_growth()               # 9
    chart_def_integrated()                # 10
    chart_def_difference()                # 11

    # --- ACF & unit root tests ---
    chart_acf_nonstationary()             # 12
    chart_differencing()                  # 13
    chart_def_arima()                     # 14
    chart_def_adf()                       # 15

    # --- Forecasting ---
    chart_rolling_forecast()              # 16

    # --- Case study ---
    chart_acf_pacf()                      # 17
    chart_diagnostics()                   # 18
    chart_arima_forecast()                # 19
    chart_adf_test()                      # 20

    # --- Quiz figures ---
    chart_quiz1_rw_variance()             # 21
    chart_quiz2_differencing()            # 22
    chart_quiz3_adf_test()                # 23
    chart_quiz4_acf_decay()               # 24
    chart_quiz5_forecast_ci()             # 25

    print('=' * 60)
    print('ALL 25 Chapter 3 charts regenerated successfully!')
    print(f'Output directory: {CHARTS_DIR}')
    print()
    print('NOTE: This script generates the following chart families:')
    print('  ch3_motivation_*    — Non-stationarity motivation (3 charts)')
    print('  ch3_gdp_*           — GDP level/trend illustrations (1 chart)')
    print('  ch3_trend_*         — Deterministic vs stochastic trend (1 chart)')
    print('  ch3_def_*           — Definition slides: RW, integrated, ARIMA, ADF (6 charts)')
    print('  ch3_random_walk     — Random walk simulations (1 chart)')
    print('  ch3_variance_growth — Variance growth illustration (1 chart)')
    print('  ch3_acf_*           — ACF analysis for non-stationary data (2 charts)')
    print('  ch3_differencing    — Differencing GDP example (1 chart)')
    print('  ch3_rolling_*       — Rolling forecast illustration (1 chart)')
    print('  ch3_diagnostics     — Residual diagnostics (1 chart)')
    print('  ch3_arima_forecast  — ARIMA forecast with CI (1 chart)')
    print('  ch3_adf_test        — ADF test case study (1 chart)')
    print('  ch3_quiz*           — Quiz illustrations (5 charts)')
    print('=' * 60)
