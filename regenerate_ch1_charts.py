#!/usr/bin/env python3
"""
Regenerate ALL Chapter 1 (Intro to Time Series) charts.

Style requirements:
  - Transparent background
  - Legend outside bottom
  - Consistent color scheme
  - No top/right spines
  - Save both PDF and PNG to charts/ directory
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

# =============================================================================
# GLOBAL STYLE CONFIGURATION
# =============================================================================
BLUE = '#1A3A6E'
RED = '#DC3545'
GREEN = '#2E7D32'
ORANGE = '#E67E22'
GRAY = '#666666'
PURPLE = '#8E44AD'

# Additional shades
LIGHT_BLUE = '#4A7ABE'
LIGHT_RED = '#E8808A'
LIGHT_GREEN = '#66BB6A'

plt.rcParams.update({
    'axes.facecolor': 'none',
    'figure.facecolor': 'none',
    'savefig.transparent': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': GRAY,
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'lines.linewidth': 1.2,
})

CHARTS_DIR = '/Users/danielpele/Documents/Time Series Analysis/charts'

def save_chart(fig, name):
    """Save chart as both PDF and PNG with transparent background."""
    fig.savefig(f'{CHARTS_DIR}/{name}.pdf', bbox_inches='tight', transparent=True, pad_inches=0.05)
    fig.savefig(f'{CHARTS_DIR}/{name}.png', bbox_inches='tight', transparent=True, dpi=200, pad_inches=0.05)
    plt.close(fig)
    print(f'  Saved: {name}.pdf + .png')

def add_legend_bottom(ax, ncol=None, **kwargs):
    """Add legend outside bottom."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if ncol is None:
        ncol = min(len(handles), 4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=ncol,
              frameon=False, **kwargs)


# =============================================================================
# HELPER: Generate simulated data
# =============================================================================
np.random.seed(42)

def generate_ar1(n, phi, sigma=1.0):
    """Generate AR(1) process."""
    x = np.zeros(n)
    eps = np.random.normal(0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t-1] + eps[t]
    return x

def generate_random_walk(n, sigma=1.0, drift=0.0):
    """Generate random walk with optional drift."""
    eps = np.random.normal(0, sigma, n)
    return np.cumsum(eps) + drift * np.arange(n)

def generate_white_noise(n, sigma=1.0):
    """Generate white noise."""
    return np.random.normal(0, sigma, n)

def generate_seasonal(n, period=12, amplitude=1.0):
    """Generate seasonal component."""
    return amplitude * np.sin(2 * np.pi * np.arange(n) / period)


# =============================================================================
# REAL DATA LOADING (with synthetic fallback)
# =============================================================================
SP500_PRICES = None
SP500_RETURNS = None
SP500_DATES = None
SP500_REAL = False

def load_sp500_data():
    """Load S&P 500 data from Yahoo Finance (2020-2025), fallback to synthetic."""
    global SP500_PRICES, SP500_RETURNS, SP500_DATES, SP500_REAL
    if SP500_PRICES is not None:
        return  # already loaded
    if YF_AVAILABLE:
        try:
            df = yf.download('^GSPC', start='2020-01-01', end='2025-12-31', progress=False)
            close = df['Close'].squeeze().dropna()
            if len(close) > 100:
                SP500_PRICES = close.values
                SP500_RETURNS = np.diff(np.log(close.values))
                SP500_DATES = close.index
                SP500_REAL = True
                print(f'  [DATA] S&P 500 loaded: {len(close)} observations ({close.index[0].strftime("%Y-%m-%d")} to {close.index[-1].strftime("%Y-%m-%d")})')
                return
        except Exception as e:
            print(f'  [DATA] Yahoo Finance failed: {e}')
    # Fallback: synthetic
    print('  [DATA] Using synthetic S&P 500 data')
    np.random.seed(42)
    n = 1250
    log_ret = np.random.normal(0.0003, 0.012, n)
    vol = np.ones(n) * 0.012
    for t in range(1, n):
        vol[t] = np.sqrt(0.00001 + 0.1 * log_ret[t-1]**2 + 0.85 * vol[t-1]**2)
        log_ret[t] = 0.0003 + vol[t] * np.random.normal()
    prices = 3000 * np.exp(np.cumsum(log_ret))
    dates = pd.date_range('2020-01-02', periods=n, freq='B')
    SP500_PRICES = prices
    SP500_RETURNS = log_ret[1:]
    SP500_DATES = dates
    SP500_REAL = False


# =============================================================================
# CHART 1: ch1_stationary_nonstationary_examples
# S&P 500 prices (nonstationary) vs log returns (stationary)
# =============================================================================
def chart_stationary_nonstationary_examples():
    print('Generating: ch1_stationary_nonstationary_examples')
    load_sp500_data()
    prices = SP500_PRICES
    log_returns = SP500_RETURNS
    dates = SP500_DATES

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))

    axes[0].plot(dates[:len(prices)], prices, color=BLUE, linewidth=0.9)
    axes[0].set_title('S&P 500 Prices (Non-stationary)', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price')
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].plot(dates[1:len(log_returns)+1], log_returns, color=RED, linewidth=0.5, alpha=0.8)
    axes[1].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[1].set_title('S&P 500 Log returns (Stationary)', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Return')
    axes[1].tick_params(axis='x', rotation=30)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_stationary_nonstationary_examples')


# =============================================================================
# CHART 2: ch1_def_stochastic
# Multiple realizations of a stochastic process
# =============================================================================
def chart_def_stochastic():
    print('Generating: ch1_def_stochastic')
    np.random.seed(42)
    n = 200
    colors = [BLUE, RED, GREEN, ORANGE, PURPLE, GRAY, LIGHT_BLUE, LIGHT_RED]

    fig, ax = plt.subplots(figsize=(7, 3.0))
    for i in range(8):
        y = generate_ar1(n, phi=0.7, sigma=1.0)
        ax.plot(y, color=colors[i % len(colors)], alpha=0.6, linewidth=0.8,
                label=f'Realization {i+1}' if i < 5 else None)

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.set_title('Multiple realizations of an AR(1) stochastic process', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel(r'$X_t(\omega)$')
    add_legend_bottom(ax, ncol=5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_def_stochastic')


# =============================================================================
# CHART 3: ch1_def_strict_stationarity
# Strict stationarity: distribution invariant under time shift
# =============================================================================
def chart_def_strict_stationarity():
    print('Generating: ch1_def_strict_stationarity')
    np.random.seed(42)
    n = 300
    y = generate_ar1(n, phi=0.5, sigma=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))

    # Full series
    axes[0].plot(y, color=BLUE, linewidth=0.7)
    axes[0].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[0].axvspan(50, 100, alpha=0.15, color=RED)
    axes[0].axvspan(180, 230, alpha=0.15, color=GREEN)
    axes[0].set_title('Stationary series', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$X_t$')

    # Window 1 histogram
    w1 = y[50:100]
    axes[1].hist(w1, bins=15, color=RED, alpha=0.6, edgecolor='white', density=True)
    x_range = np.linspace(w1.min()-1, w1.max()+1, 100)
    axes[1].plot(x_range, stats.norm.pdf(x_range, w1.mean(), w1.std()), color=RED, linewidth=1.5)
    axes[1].set_title(f'Window 1 (t=50-100)', fontsize=9, fontweight='bold')
    axes[1].set_xlabel(r'$X_t$')
    axes[1].set_ylabel('Density')

    # Window 2 histogram
    w2 = y[180:230]
    axes[2].hist(w2, bins=15, color=GREEN, alpha=0.6, edgecolor='white', density=True)
    x_range2 = np.linspace(w2.min()-1, w2.max()+1, 100)
    axes[2].plot(x_range2, stats.norm.pdf(x_range2, w2.mean(), w2.std()), color=GREEN, linewidth=1.5)
    axes[2].set_title(f'Window 2 (t=180-230)', fontsize=9, fontweight='bold')
    axes[2].set_xlabel(r'$X_t$')
    axes[2].set_ylabel('Density')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_def_strict_stationarity')


# =============================================================================
# CHART 4: ch1_def_weak_stationarity
# Weak stationarity: constant mean, variance, and autocovariance
# =============================================================================
def chart_def_weak_stationarity():
    print('Generating: ch1_def_weak_stationarity')
    np.random.seed(42)
    n = 300
    y = generate_ar1(n, phi=0.5, sigma=1.0)

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.8))

    # Time series with mean and bands
    mu = y.mean()
    sigma = y.std()
    axes[0].plot(y, color=BLUE, linewidth=0.7, label=r'$X_t$')
    axes[0].axhline(mu, color=RED, linewidth=1.0, linestyle='--', label=r'$\mu$')
    axes[0].axhline(mu + 2*sigma, color=ORANGE, linewidth=0.8, linestyle=':', label=r'$\mu \pm 2\sigma$')
    axes[0].axhline(mu - 2*sigma, color=ORANGE, linewidth=0.8, linestyle=':')
    axes[0].set_title(r'$E[X_t] = \mu$, $Var(X_t) = \sigma^2$', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$X_t$')
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    # Rolling mean
    window = 50
    rolling_mean = pd.Series(y).rolling(window).mean()
    rolling_var = pd.Series(y).rolling(window).var()
    axes[1].plot(rolling_mean, color=RED, linewidth=1.0, label='Rolling mean')
    axes[1].axhline(mu, color=GRAY, linewidth=0.8, linestyle='--', label=r'$\mu$ total')
    axes[1].set_title('Constant mean over time', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Mean')
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    # ACF
    acf_vals = acf(y, nlags=20)
    axes[2].bar(range(len(acf_vals)), acf_vals, color=BLUE, width=0.5, alpha=0.7)
    axes[2].axhline(1.96/np.sqrt(n), color=RED, linewidth=0.8, linestyle='--', label=r'$\pm 1.96/\sqrt{T}$')
    axes[2].axhline(-1.96/np.sqrt(n), color=RED, linewidth=0.8, linestyle='--')
    axes[2].set_title(r'ACF: $\gamma(h)$ depends only on $h$', fontsize=9, fontweight='bold')
    axes[2].set_xlabel('Lag')
    axes[2].set_ylabel(r'$\hat{\rho}(h)$')
    axes[2].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'ch1_def_weak_stationarity')


# =============================================================================
# CHART 5: ch1_counterexample_stationarity
# Weak stationary but NOT strictly stationary
# =============================================================================
def chart_counterexample_stationarity():
    print('Generating: ch1_counterexample_stationarity')
    np.random.seed(42)
    n = 200

    # Even indices: N(0,1), Odd indices: scaled chi-squared
    x = np.zeros(n)
    for t in range(n):
        if t % 2 == 0:
            x[t] = np.random.normal(0, 1)
        else:
            x[t] = (np.random.chisquare(5) - 5) / np.sqrt(10)

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    axes[0].plot(x, color=BLUE, linewidth=0.6)
    axes[0].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[0].set_title('Complete series', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$X_t$')

    # Even indices (Normal)
    x_even = x[0::2]
    axes[1].hist(x_even, bins=20, color=GREEN, alpha=0.6, edgecolor='white', density=True, label='t even: N(0,1)')
    xr = np.linspace(-4, 4, 100)
    axes[1].plot(xr, stats.norm.pdf(xr, 0, 1), color=GREEN, linewidth=1.5)
    axes[1].set_title('t even: Symmetric', fontsize=9, fontweight='bold')
    axes[1].set_xlabel(r'$X_t$')
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    # Odd indices (Chi-squared)
    x_odd = x[1::2]
    axes[2].hist(x_odd, bins=20, color=RED, alpha=0.6, edgecolor='white', density=True, label=r't odd: scaled $\chi^2$')
    xr2 = np.linspace(-3, 5, 100)
    chi_pdf = stats.chi2.pdf(xr2 * np.sqrt(10) + 5, 5) * np.sqrt(10)
    axes[2].plot(xr2, chi_pdf, color=RED, linewidth=1.5)
    axes[2].set_title('t odd: Asymmetric', fontsize=9, fontweight='bold')
    axes[2].set_xlabel(r'$X_t$')
    axes[2].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'ch1_counterexample_stationarity')


# =============================================================================
# CHART 6: ch1_ergodicity
# Ergodicity illustration: time average converges to ensemble average
# =============================================================================
def chart_ergodicity():
    print('Generating: ch1_ergodicity')
    np.random.seed(42)
    n = 500
    mu_true = 2.0
    num_realizations = 30

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))

    # Left: Time average of single realization
    y = generate_ar1(n, phi=0.5, sigma=1.0) + mu_true
    cumulative_mean = np.cumsum(y) / np.arange(1, n+1)
    axes[0].plot(cumulative_mean, color=BLUE, linewidth=1.0, label=r'$\bar{X}_T$ (time average)')
    axes[0].axhline(mu_true, color=RED, linewidth=1.2, linestyle='--', label=r'$\mu = 2.0$ (population mean)')
    axes[0].set_title('Time average (one realization)', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('T (no. observations)')
    axes[0].set_ylabel(r'$\bar{X}_T$')
    axes[0].set_ylim(mu_true - 2, mu_true + 2)
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    # Right: Ensemble average across realizations
    ensemble_means = []
    for i in range(num_realizations):
        yi = generate_ar1(n, phi=0.5, sigma=1.0) + mu_true
        if i < 8:
            axes[1].plot(yi[:100], color=GRAY, alpha=0.2, linewidth=0.5)
        ensemble_means.append(yi.mean())

    cumulative_ensemble = np.cumsum(ensemble_means) / np.arange(1, num_realizations+1)
    ax2 = axes[1].twinx()
    ax2.plot(range(1, num_realizations+1), cumulative_ensemble, color=GREEN, linewidth=1.5,
             marker='o', markersize=3, label='Ensemble mean')
    ax2.axhline(mu_true, color=RED, linewidth=1.2, linestyle='--', label=r'$\mu = 2.0$')
    ax2.set_ylabel('Ensemble mean', color=GREEN)
    ax2.spines['top'].set_visible(False)

    axes[1].set_title('Ensemble mean (multiple realizations)', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Time / No. realizations')
    axes[1].set_ylabel(r'$X_t$')

    # Combined legend
    ax2.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'ch1_ergodicity')


# =============================================================================
# CHART 7: ch1_wold_decomposition
# Wold decomposition: stochastic (MA infinity) + deterministic
# =============================================================================
def chart_wold_decomposition():
    print('Generating: ch1_wold_decomposition')
    np.random.seed(42)
    n = 200

    # Wold coefficients (exponentially decaying)
    psi = np.array([0.7**j for j in range(50)])

    # Generate process
    eps = np.random.normal(0, 1, n + 50)
    stochastic = np.zeros(n)
    for t in range(n):
        stochastic[t] = sum(psi[j] * eps[t + 50 - j] for j in range(min(t+1, 50)))

    deterministic = 0.5 * np.sin(2 * np.pi * np.arange(n) / 50) + 1.0
    total = stochastic + deterministic

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.0))

    # Wold coefficients
    axes[0, 0].bar(range(20), psi[:20], color=BLUE, alpha=0.7, width=0.6)
    axes[0, 0].set_title(r'Wold coefficients $\psi_j$', fontsize=9, fontweight='bold')
    axes[0, 0].set_xlabel('j')
    axes[0, 0].set_ylabel(r'$\psi_j$')

    # Total process
    axes[0, 1].plot(total, color=BLUE, linewidth=0.8)
    axes[0, 1].set_title(r'$X_t = \sum \psi_j \varepsilon_{t-j} + \eta_t$', fontsize=9, fontweight='bold')
    axes[0, 1].set_xlabel('Time')

    # Stochastic component
    axes[1, 0].plot(stochastic, color=RED, linewidth=0.8)
    axes[1, 0].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[1, 0].set_title(r'Stochastic: $\sum \psi_j \varepsilon_{t-j}$', fontsize=9, fontweight='bold')
    axes[1, 0].set_xlabel('Time')

    # Deterministic component
    axes[1, 1].plot(deterministic, color=GREEN, linewidth=1.2)
    axes[1, 1].set_title(r'Deterministic: $\eta_t$', fontsize=9, fontweight='bold')
    axes[1, 1].set_xlabel('Time')

    fig.tight_layout(h_pad=2.0)
    save_chart(fig, 'ch1_wold_decomposition')


# =============================================================================
# CHART 8: ch1_def_lag_operator
# Lag operator illustration
# =============================================================================
def chart_def_lag_operator():
    print('Generating: ch1_def_lag_operator')
    np.random.seed(42)
    n = 100
    y = generate_ar1(n, phi=0.8, sigma=1.0) + 5.0
    t = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.0))

    # Original vs Lagged
    axes[0].plot(t, y, color=BLUE, linewidth=1.0, label=r'$X_t$')
    axes[0].plot(t[1:], y[:-1], color=RED, linewidth=1.0, linestyle='--', label=r'$LX_t = X_{t-1}$')
    axes[0].set_title(r'$LX_t = X_{t-1}$', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    # First difference
    diff1 = np.diff(y)
    axes[1].plot(diff1, color=GREEN, linewidth=0.8)
    axes[1].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[1].set_title(r'$\Delta X_t = (1-L)X_t$', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel(r'$\Delta X_t$')

    # Second difference
    diff2 = np.diff(diff1)
    axes[2].plot(diff2, color=PURPLE, linewidth=0.8)
    axes[2].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[2].set_title(r'$\Delta^2 X_t = (1-L)^2 X_t$', fontsize=9, fontweight='bold')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel(r'$\Delta^2 X_t$')

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'ch1_def_lag_operator')


# =============================================================================
# CHART 9: differencing_effect
# S&P 500 prices vs log returns (differencing effect)
# =============================================================================
def chart_differencing_effect():
    print('Generating: differencing_effect')
    load_sp500_data()
    prices = SP500_PRICES
    log_ret = SP500_RETURNS
    dates = SP500_DATES

    fig, axes = plt.subplots(2, 1, figsize=(7, 3.5))

    axes[0].plot(dates[:len(prices)], prices, color=BLUE, linewidth=0.8)
    axes[0].set_title('S&P 500: Prices (non-stationary, I(1))', fontsize=9, fontweight='bold')
    axes[0].set_ylabel('Price')
    axes[0].tick_params(axis='x', rotation=30)

    axes[1].plot(dates[1:len(log_ret)+1], log_ret, color=RED, linewidth=0.4, alpha=0.8)
    axes[1].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[1].set_title(r'Log returns: $r_t = \ln P_t - \ln P_{t-1}$ (stationary)', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Return')
    axes[1].tick_params(axis='x', rotation=30)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'differencing_effect')


# =============================================================================
# CHART 10: ch1_transform_sequence_ro
# Transformation sequence: raw -> log -> diff
# =============================================================================
def chart_transform_sequence_ro():
    print('Generating: ch1_transform_sequence_ro')
    load_sp500_data()
    prices = SP500_PRICES
    log_prices = np.log(prices)
    returns = SP500_RETURNS
    dates = SP500_DATES

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    axes[0].plot(dates[:len(prices)], prices, color=BLUE, linewidth=0.8, label=r'$P_t$')
    axes[0].set_title(r'S&P 500 $P_t$', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    axes[1].plot(dates[:len(log_prices)], log_prices, color=GREEN, linewidth=0.8, label=r'$\ln(P_t)$')
    axes[1].set_title(r'$\ln(P_t)$', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Log price')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    axes[2].plot(dates[1:len(returns)+1], returns, color=RED, linewidth=0.4, alpha=0.8, label=r'$r_t = \Delta \ln(P_t)$')
    axes[2].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[2].set_title(r'Returns $r_t$', fontsize=9, fontweight='bold')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Return')
    axes[2].tick_params(axis='x', rotation=30)
    axes[2].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'ch1_transform_sequence_ro')


# =============================================================================
# CHART 11: ch1_def_white_noise
# White noise process with ACF
# =============================================================================
def chart_def_white_noise():
    print('Generating: ch1_def_white_noise')
    np.random.seed(42)
    n = 300
    wn = generate_white_noise(n)

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))

    axes[0].plot(wn, color=BLUE, linewidth=0.5, alpha=0.8)
    axes[0].axhline(0, color=RED, linewidth=0.8, linestyle='--', label=r'$\mu = 0$')
    axes[0].axhline(2, color=ORANGE, linewidth=0.6, linestyle=':', label=r'$\pm 2\sigma$')
    axes[0].axhline(-2, color=ORANGE, linewidth=0.6, linestyle=':')
    axes[0].set_title(r'White noise: $\varepsilon_t \sim WN(0, \sigma^2)$', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$\varepsilon_t$')
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    # ACF
    acf_vals = acf(wn, nlags=25)
    axes[1].bar(range(len(acf_vals)), acf_vals, color=BLUE, width=0.5, alpha=0.7)
    ci = 1.96 / np.sqrt(n)
    axes[1].axhline(ci, color=RED, linewidth=0.8, linestyle='--', label=r'$\pm 1.96/\sqrt{T}$')
    axes[1].axhline(-ci, color=RED, linewidth=0.8, linestyle='--')
    axes[1].axhline(0, color=GRAY, linewidth=0.3)
    axes[1].set_title('ACF white noise', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel(r'$\hat{\rho}(h)$')
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'ch1_def_white_noise')


# =============================================================================
# CHART 12: random_walk
# Random walk realizations showing growing variance
# =============================================================================
def chart_random_walk():
    print('Generating: random_walk')
    np.random.seed(42)
    n = 300

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))

    colors_rw = [BLUE, RED, GREEN, ORANGE, PURPLE]
    for i in range(5):
        rw = generate_random_walk(n)
        axes[0].plot(rw, color=colors_rw[i], linewidth=0.8, alpha=0.7,
                     label=f'Realization {i+1}')

    axes[0].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    # Variance envelope
    t_arr = np.arange(1, n+1)
    axes[0].fill_between(t_arr-1, -2*np.sqrt(t_arr), 2*np.sqrt(t_arr),
                         alpha=0.08, color=GRAY, label=r'$\pm 2\sqrt{t}\sigma$')
    axes[0].set_title('Random walk: multiple realizations', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$X_t$')
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    # Variance grows linearly
    variances = []
    num_sims = 500
    for _ in range(num_sims):
        rw = generate_random_walk(n)
        variances.append(rw**2)
    var_empirical = np.mean(variances, axis=0)
    axes[1].plot(var_empirical, color=BLUE, linewidth=1.0, label=r'$\widehat{Var}(X_t)$ (empirical)')
    axes[1].plot(t_arr, color=RED, linewidth=1.0, linestyle='--', label=r'$t \cdot \sigma^2$ (theoretical)')
    axes[1].set_title(r'$Var(X_t) = t\sigma^2$ grows linearly', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Time (t)')
    axes[1].set_ylabel('Variance')
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'random_walk')


# =============================================================================
# CHART 13: ch1_wn_rw
# White noise vs random walk side by side
# =============================================================================
def chart_wn_rw():
    print('Generating: ch1_wn_rw')
    np.random.seed(42)
    n = 300
    wn = generate_white_noise(n)
    rw = np.cumsum(wn)  # RW is cumulative sum of WN

    fig, axes = plt.subplots(2, 2, figsize=(7, 4.0))

    # WN series
    axes[0, 0].plot(wn, color=BLUE, linewidth=0.5)
    axes[0, 0].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[0, 0].set_title('White noise (WN)', fontsize=9, fontweight='bold')
    axes[0, 0].set_ylabel(r'$\varepsilon_t$')

    # WN ACF
    acf_wn = acf(wn, nlags=20)
    axes[0, 1].bar(range(len(acf_wn)), acf_wn, color=BLUE, width=0.5, alpha=0.7)
    ci = 1.96 / np.sqrt(n)
    axes[0, 1].axhline(ci, color=RED, linewidth=0.8, linestyle='--')
    axes[0, 1].axhline(-ci, color=RED, linewidth=0.8, linestyle='--')
    axes[0, 1].set_title('ACF: White noise', fontsize=9, fontweight='bold')
    axes[0, 1].set_ylabel(r'$\hat{\rho}(h)$')

    # RW series
    axes[1, 0].plot(rw, color=RED, linewidth=0.8)
    axes[1, 0].set_title('Random walk (RW)', fontsize=9, fontweight='bold')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel(r'$X_t$')

    # RW ACF
    acf_rw = acf(rw, nlags=20)
    axes[1, 1].bar(range(len(acf_rw)), acf_rw, color=RED, width=0.5, alpha=0.7)
    axes[1, 1].set_title('ACF: Random walk (slow decay)', fontsize=9, fontweight='bold')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel(r'$\hat{\rho}(h)$')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_wn_rw')


# =============================================================================
# CHART 14: rw_vs_stationary
# ACF comparison: stationary vs random walk
# =============================================================================
def chart_rw_vs_stationary():
    print('Generating: rw_vs_stationary')
    np.random.seed(42)
    n = 300
    stationary = generate_ar1(n, phi=0.7, sigma=1.0)
    rw = generate_random_walk(n)
    nlags = 25

    fig, axes = plt.subplots(2, 2, figsize=(8, 4.0))

    # Stationary series
    axes[0, 0].plot(stationary, color=BLUE, linewidth=0.7)
    axes[0, 0].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[0, 0].set_title('Stationary AR(1) series', fontsize=9, fontweight='bold')
    axes[0, 0].set_ylabel(r'$X_t$')

    # Stationary ACF
    acf_stat = acf(stationary, nlags=nlags)
    axes[0, 1].bar(range(len(acf_stat)), acf_stat, color=BLUE, width=0.5, alpha=0.7)
    ci = 1.96 / np.sqrt(n)
    axes[0, 1].axhline(ci, color=RED, linewidth=0.8, linestyle='--')
    axes[0, 1].axhline(-ci, color=RED, linewidth=0.8, linestyle='--')
    axes[0, 1].set_title('ACF: Fast decay', fontsize=9, fontweight='bold')
    axes[0, 1].set_ylabel(r'$\hat{\rho}(h)$')

    # Random walk series
    axes[1, 0].plot(rw, color=RED, linewidth=0.8)
    axes[1, 0].set_title('Random walk', fontsize=9, fontweight='bold')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel(r'$X_t$')

    # RW ACF
    acf_rw = acf(rw, nlags=nlags)
    axes[1, 1].bar(range(len(acf_rw)), acf_rw, color=RED, width=0.5, alpha=0.7)
    axes[1, 1].set_title('ACF: Very slow decay', fontsize=9, fontweight='bold')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel(r'$\hat{\rho}(h)$')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'rw_vs_stationary')


# =============================================================================
# CHART 15: ch1_acf_examples
# ACF patterns for different processes
# =============================================================================
def chart_acf_examples():
    print('Generating: ch1_acf_examples')
    np.random.seed(42)
    n = 300
    nlags = 25

    wn = generate_white_noise(n)
    ar1 = generate_ar1(n, phi=0.7, sigma=1.0)
    rw = generate_random_walk(n)
    seasonal = generate_ar1(n, phi=0.5) + 2 * np.sin(2 * np.pi * np.arange(n) / 12)

    fig, axes = plt.subplots(2, 4, figsize=(9, 3.8))

    series = [wn, ar1, rw, seasonal]
    titles_top = ['White noise', 'AR(1), phi=0.7', 'Random walk', 'Seasonal (s=12)']
    colors_s = [BLUE, GREEN, RED, PURPLE]

    for i, (s, title, col) in enumerate(zip(series, titles_top, colors_s)):
        axes[0, i].plot(s, color=col, linewidth=0.5)
        axes[0, i].set_title(title, fontsize=8, fontweight='bold')
        if i == 0:
            axes[0, i].set_ylabel(r'$X_t$')
        axes[0, i].axhline(0 if i != 2 else s.mean(), color=GRAY, linewidth=0.3, linestyle='--')

        acf_vals = acf(s, nlags=nlags)
        axes[1, i].bar(range(len(acf_vals)), acf_vals, color=col, width=0.5, alpha=0.7)
        ci = 1.96 / np.sqrt(n)
        axes[1, i].axhline(ci, color=RED, linewidth=0.6, linestyle='--')
        axes[1, i].axhline(-ci, color=RED, linewidth=0.6, linestyle='--')
        axes[1, i].set_xlabel('Lag')
        if i == 0:
            axes[1, i].set_ylabel('ACF')
        axes[1, i].set_title(f'ACF: {title}', fontsize=8, fontweight='bold')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_acf_examples')


# =============================================================================
# CHART 16: acf_pacf_examples
# ACF and PACF patterns for AR, MA, ARMA
# =============================================================================
def chart_acf_pacf_examples():
    print('Generating: acf_pacf_examples')
    np.random.seed(42)
    n = 500
    nlags = 20

    # AR(2)
    ar2 = np.zeros(n)
    eps = np.random.normal(0, 1, n)
    for t in range(2, n):
        ar2[t] = 0.6 * ar2[t-1] - 0.2 * ar2[t-2] + eps[t]

    # MA(2)
    ma2 = np.zeros(n)
    for t in range(2, n):
        ma2[t] = eps[t] + 0.7 * eps[t-1] + 0.3 * eps[t-2]

    # ARMA(1,1)
    arma11 = np.zeros(n)
    for t in range(1, n):
        arma11[t] = 0.6 * arma11[t-1] + eps[t] + 0.4 * eps[t-1]

    fig, axes = plt.subplots(2, 3, figsize=(8, 4.0))
    titles = ['AR(2)', 'MA(2)', 'ARMA(1,1)']
    series_list = [ar2, ma2, arma11]
    colors_s = [BLUE, RED, GREEN]

    for i, (s, title, col) in enumerate(zip(series_list, titles, colors_s)):
        acf_vals = acf(s, nlags=nlags)
        pacf_vals = pacf(s, nlags=nlags)

        axes[0, i].bar(range(len(acf_vals)), acf_vals, color=col, width=0.5, alpha=0.7)
        ci = 1.96 / np.sqrt(n)
        axes[0, i].axhline(ci, color=RED, linewidth=0.6, linestyle='--')
        axes[0, i].axhline(-ci, color=RED, linewidth=0.6, linestyle='--')
        axes[0, i].set_title(f'ACF: {title}', fontsize=9, fontweight='bold')
        if i == 0:
            axes[0, i].set_ylabel('ACF')

        axes[1, i].bar(range(len(pacf_vals)), pacf_vals, color=col, width=0.5, alpha=0.7)
        axes[1, i].axhline(ci, color=RED, linewidth=0.6, linestyle='--')
        axes[1, i].axhline(-ci, color=RED, linewidth=0.6, linestyle='--')
        axes[1, i].set_title(f'PACF: {title}', fontsize=9, fontweight='bold')
        axes[1, i].set_xlabel('Lag')
        if i == 0:
            axes[1, i].set_ylabel('PACF')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'acf_pacf_examples')


# =============================================================================
# CHART 17: acf_theoretical
# Theoretical ACF decay patterns
# =============================================================================
def chart_acf_theoretical():
    print('Generating: acf_theoretical')
    nlags = 20
    lags = np.arange(nlags + 1)

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    # Exponential decay (phi > 0)
    phi_pos = 0.8
    acf_exp = phi_pos ** lags
    axes[0].bar(lags, acf_exp, color=BLUE, width=0.5, alpha=0.7)
    axes[0].set_title(r'Exponential decay ($\phi = 0.8$)', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel(r'$\rho(h)$')

    # Oscillatory decay (phi < 0)
    phi_neg = -0.7
    acf_osc = phi_neg ** lags
    colors_bars = [BLUE if v >= 0 else RED for v in acf_osc]
    axes[1].bar(lags, acf_osc, color=colors_bars, width=0.5, alpha=0.7)
    axes[1].set_title(r'Oscillatory decay ($\phi = -0.7$)', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel(r'$\rho(h)$')

    # Slow decay (near unit root)
    phi_slow = 0.99
    acf_slow = phi_slow ** lags
    axes[2].bar(lags, acf_slow, color=RED, width=0.5, alpha=0.7)
    axes[2].set_title(r'Slow decay ($\phi = 0.99$)', fontsize=9, fontweight='bold')
    axes[2].set_xlabel('Lag')
    axes[2].set_ylabel(r'$\rho(h)$')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'acf_theoretical')


# =============================================================================
# CHART 18: adf_test_visualization
# ADF test visualization with prices and returns
# =============================================================================
def chart_adf_test_visualization():
    print('Generating: adf_test_visualization')
    load_sp500_data()
    prices = SP500_PRICES
    log_ret = SP500_RETURNS
    dates = SP500_DATES

    # Compute actual ADF statistics
    adf_prices = adfuller(prices, maxlag=20, autolag='AIC')
    adf_returns = adfuller(log_ret, maxlag=20, autolag='AIC')

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))

    axes[0].plot(dates[:len(prices)], prices, color=BLUE, linewidth=0.8)
    axes[0].set_title(f'S&P 500 Prices: ADF = {adf_prices[0]:.2f} (p = {adf_prices[1]:.2f})\nNon-stationary',
                      fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].annotate('Do not reject $H_0$\n(unit root)',
                     xy=(0.5, 0.85), xycoords='axes fraction',
                     fontsize=8, color=RED, fontweight='bold',
                     ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=RED, alpha=0.8))

    p_str = f'p = {adf_returns[1]:.4f}' if adf_returns[1] >= 0.01 else 'p < 0.01'
    axes[1].plot(dates[1:len(log_ret)+1], log_ret, color=GREEN, linewidth=0.4, alpha=0.8)
    axes[1].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[1].set_title(f'S&P 500 Returns: ADF = {adf_returns[0]:.1f} ({p_str})\nStationary',
                      fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Return')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].annotate('Reject $H_0$\n(stationary)',
                     xy=(0.5, 0.85), xycoords='axes fraction',
                     fontsize=8, color=GREEN, fontweight='bold',
                     ha='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=GREEN, alpha=0.8))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'adf_test_visualization')


# =============================================================================
# CHART 19: sp500_analysis
# S&P 500 comprehensive analysis (4 panels)
# =============================================================================
def chart_sp500_analysis():
    print('Generating: sp500_analysis')
    load_sp500_data()
    prices = SP500_PRICES
    log_ret = SP500_RETURNS
    dates = SP500_DATES
    n = len(log_ret)

    fig, axes = plt.subplots(2, 2, figsize=(8, 4.0))

    # Prices
    axes[0, 0].plot(dates[:len(prices)], prices, color=BLUE, linewidth=0.8)
    axes[0, 0].set_title('S&P 500 Prices', fontsize=9, fontweight='bold')
    axes[0, 0].set_ylabel('Price')
    axes[0, 0].tick_params(axis='x', rotation=30)

    # Returns
    axes[0, 1].plot(dates[1:n+1], log_ret, color=RED, linewidth=0.4, alpha=0.8)
    axes[0, 1].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[0, 1].set_title('Log returns', fontsize=9, fontweight='bold')
    axes[0, 1].set_ylabel('Return')
    axes[0, 1].tick_params(axis='x', rotation=30)

    # ACF of returns
    acf_ret = acf(log_ret, nlags=25)
    axes[1, 0].bar(range(len(acf_ret)), acf_ret, color=GREEN, width=0.5, alpha=0.7)
    ci = 1.96 / np.sqrt(n)
    axes[1, 0].axhline(ci, color=RED, linewidth=0.6, linestyle='--')
    axes[1, 0].axhline(-ci, color=RED, linewidth=0.6, linestyle='--')
    axes[1, 0].set_title(r'ACF returns ($\approx 0$)', fontsize=9, fontweight='bold')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')

    # ACF of squared returns
    acf_ret2 = acf(log_ret**2, nlags=25)
    axes[1, 1].bar(range(len(acf_ret2)), acf_ret2, color=ORANGE, width=0.5, alpha=0.7)
    axes[1, 1].axhline(ci, color=RED, linewidth=0.6, linestyle='--')
    axes[1, 1].axhline(-ci, color=RED, linewidth=0.6, linestyle='--')
    axes[1, 1].set_title(r'ACF $r_t^2$ (volatility clustering)', fontsize=9, fontweight='bold')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('ACF')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'sp500_analysis')


# =============================================================================
# CHART 20: returns_distribution
# Returns distribution: histogram vs normal
# =============================================================================
def chart_returns_distribution():
    print('Generating: returns_distribution')
    load_sp500_data()
    returns = SP500_RETURNS

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))

    # Histogram
    axes[0].hist(returns, bins=60, color=BLUE, alpha=0.6, edgecolor='white', density=True, label='S&P 500 returns')
    x_range = np.linspace(returns.min(), returns.max(), 200)
    axes[0].plot(x_range, stats.norm.pdf(x_range, returns.mean(), returns.std()),
                 color=RED, linewidth=1.5, linestyle='--', label='Normal')
    # Fit Student-t
    t_params = stats.t.fit(returns)
    axes[0].plot(x_range, stats.t.pdf(x_range, *t_params),
                 color=GREEN, linewidth=1.5, label=f'Student-t (df={t_params[0]:.1f})')
    axes[0].set_title('S&P 500 returns distribution', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Return')
    axes[0].set_ylabel('Density')
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    # QQ plot
    stats.probplot(returns, dist='norm', plot=axes[1])
    axes[1].get_lines()[0].set_color(BLUE)
    axes[1].get_lines()[0].set_markersize(2)
    axes[1].get_lines()[1].set_color(RED)
    axes[1].set_title('QQ-Plot vs Normal', fontsize=9, fontweight='bold')

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'returns_distribution')


# =============================================================================
# CHART 21: volatility_clustering
# Volatility clustering in financial returns
# =============================================================================
def chart_volatility_clustering():
    print('Generating: volatility_clustering')
    load_sp500_data()
    returns = SP500_RETURNS
    dates = SP500_DATES

    fig, axes = plt.subplots(2, 1, figsize=(7, 3.5))

    axes[0].plot(dates[1:len(returns)+1], returns, color=BLUE, linewidth=0.4, alpha=0.8)
    axes[0].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[0].set_title('S&P 500 returns (volatility clustering)', fontsize=9, fontweight='bold')
    axes[0].set_ylabel('Return')
    axes[0].tick_params(axis='x', rotation=30)

    abs_ret = pd.Series(np.abs(returns), index=dates[1:len(returns)+1])
    axes[1].plot(abs_ret.index, abs_ret.values, color=ORANGE, linewidth=0.5, alpha=0.7, label=r'$|r_t|$')
    axes[1].plot(abs_ret.rolling(20).mean(), color=RED, linewidth=1.2, label='MA(20) $|r_t|$')
    axes[1].set_title('Volatility: |returns| and rolling mean', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel(r'$|r_t|$')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(h_pad=2.0)
    save_chart(fig, 'volatility_clustering')


# =============================================================================
# CHART 22: ch1_case_gdp_raw
# Romania quarterly GDP raw data
# =============================================================================
def chart_case_gdp_raw():
    print('Generating: ch1_case_gdp_raw')
    np.random.seed(42)
    # Simulate Romania GDP-like quarterly data 2010-2023
    quarters = pd.date_range('2010-01-01', '2023-12-31', freq='QS')
    n = len(quarters)
    trend = np.linspace(30, 80, n)
    seasonal = 3.0 * np.sin(2 * np.pi * np.arange(n) / 4 + 0.5)
    # COVID shock
    covid_shock = np.zeros(n)
    covid_idx = 40  # ~Q2 2020
    covid_shock[covid_idx] = -12
    covid_shock[covid_idx+1] = -5
    covid_shock[covid_idx+2] = 3
    noise = np.random.normal(0, 1.5, n)
    gdp = trend + seasonal + covid_shock + noise

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(quarters, gdp, color=BLUE, linewidth=1.0, marker='o', markersize=2, label='GDP Romania (bn RON)')
    ax.axvspan(pd.Timestamp('2020-03-01'), pd.Timestamp('2020-09-01'), alpha=0.15, color=RED, label='COVID-19')
    ax.set_title('Quarterly GDP Romania (2010-2023)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Period')
    ax.set_ylabel('GDP (bn RON)')
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    add_legend_bottom(ax, ncol=2)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_case_gdp_raw')


# =============================================================================
# CHART 23: ch1_quiz2_stationarity
# Quiz: Stationary vs non-stationary comparison
# =============================================================================
def chart_quiz2_stationarity():
    print('Generating: ch1_quiz2_stationarity')
    np.random.seed(42)
    n = 200

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    # Stationary
    y_stat = generate_ar1(n, phi=0.5, sigma=1.0)
    axes[0].plot(y_stat, color=GREEN, linewidth=0.7)
    axes[0].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[0].axhline(2, color=ORANGE, linewidth=0.5, linestyle=':')
    axes[0].axhline(-2, color=ORANGE, linewidth=0.5, linestyle=':')
    axes[0].set_title('Stationary', fontsize=9, fontweight='bold', color=GREEN)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$X_t$')

    # Non-stationary (trend)
    y_trend = 0.05 * np.arange(n) + np.random.normal(0, 1, n)
    axes[1].plot(y_trend, color=RED, linewidth=0.7)
    axes[1].set_title('Non-stationary (trend)', fontsize=9, fontweight='bold', color=RED)
    axes[1].set_xlabel('Time')

    # Non-stationary (changing variance)
    y_var = np.random.normal(0, 1, n) * np.linspace(0.5, 4, n)
    axes[2].plot(y_var, color=RED, linewidth=0.7)
    axes[2].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[2].set_title('Non-stationary (variance)', fontsize=9, fontweight='bold', color=RED)
    axes[2].set_xlabel('Time')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_quiz2_stationarity')


# =============================================================================
# CHART 24: ch1_quiz4_wn_rw
# Quiz: White noise vs random walk
# =============================================================================
def chart_quiz4_wn_rw():
    print('Generating: ch1_quiz4_wn_rw')
    np.random.seed(42)
    n = 200
    wn = generate_white_noise(n)
    rw = np.cumsum(wn)

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))

    axes[0].plot(wn, color=BLUE, linewidth=0.5)
    axes[0].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[0].fill_between(range(n), -2, 2, alpha=0.08, color=BLUE)
    axes[0].set_title(r'White noise: $Var = \sigma^2$ (const.)', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$\varepsilon_t$')

    axes[1].plot(rw, color=RED, linewidth=0.8)
    t_arr = np.arange(1, n+1)
    axes[1].fill_between(range(n), -2*np.sqrt(t_arr), 2*np.sqrt(t_arr),
                         alpha=0.08, color=RED)
    axes[1].set_title(r'Random walk: $Var = t\sigma^2$ (grows!)', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel(r'$X_t$')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_quiz4_wn_rw')


# =============================================================================
# CHART 25: ch1_stationarity
# Stationarity illustration (for quiz answer 3 - KPSS)
# =============================================================================
def chart_stationarity():
    print('Generating: ch1_stationarity')
    np.random.seed(42)
    n = 200

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))

    # Stationary (KPSS H0 not rejected)
    y_stat = generate_ar1(n, phi=0.5, sigma=1.0)
    axes[0].plot(y_stat, color=GREEN, linewidth=0.7)
    axes[0].axhline(y_stat.mean(), color=RED, linewidth=0.8, linestyle='--')
    axes[0].set_title('KPSS: Do not reject $H_0$\n(Stationary)', fontsize=9, fontweight='bold', color=GREEN)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$X_t$')
    axes[0].annotate('$H_0$: Stationary',
                     xy=(0.5, 0.9), xycoords='axes fraction', ha='center',
                     fontsize=8, color=GREEN, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor=GREEN, alpha=0.8))

    # Non-stationary (KPSS H0 rejected)
    y_ns = generate_random_walk(n)
    axes[1].plot(y_ns, color=RED, linewidth=0.8)
    axes[1].set_title('KPSS: Reject $H_0$\n(Non-stationary)', fontsize=9, fontweight='bold', color=RED)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel(r'$X_t$')
    axes[1].annotate('$H_1$: Unit root',
                     xy=(0.5, 0.9), xycoords='axes fraction', ha='center',
                     fontsize=8, color=RED, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', edgecolor=RED, alpha=0.8))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_stationarity')


# =============================================================================
# ADDITIONAL CHARTS that exist in the charts/ directory
# =============================================================================

# CHART 26: ch1_motivation_real
def chart_motivation_real():
    print('Generating: ch1_motivation_real')
    np.random.seed(42)
    n = 200

    fig, axes = plt.subplots(2, 2, figsize=(8, 4.0))

    # GDP-like
    t = np.arange(n)
    gdp = 100 + 0.3 * t + 5 * np.sin(2*np.pi*t/40) + np.random.normal(0, 2, n)
    axes[0, 0].plot(gdp, color=BLUE, linewidth=0.8)
    axes[0, 0].set_title('GDP (trend + cycle)', fontsize=9, fontweight='bold')
    axes[0, 0].set_ylabel('Value')

    # Temperature-like
    temp = 15 + 10 * np.sin(2*np.pi*t/12) + np.random.normal(0, 2, n)
    axes[0, 1].plot(temp, color=RED, linewidth=0.8)
    axes[0, 1].set_title('Temperature (seasonal)', fontsize=9, fontweight='bold')

    # Stock-like
    stock = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n)))
    axes[1, 0].plot(stock, color=GREEN, linewidth=0.8)
    axes[1, 0].set_title('Stock price (random walk)', fontsize=9, fontweight='bold')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Value')

    # Electricity-like
    elec = 500 + 100 * np.sin(2*np.pi*t/12) + 30 * np.sin(2*np.pi*t/52) + 0.5*t + np.random.normal(0, 20, n)
    axes[1, 1].plot(elec, color=ORANGE, linewidth=0.8)
    axes[1, 1].set_title('Energy consumption (multiple seasonality)', fontsize=9, fontweight='bold')
    axes[1, 1].set_xlabel('Time')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_motivation_real')


# CHART 27: ch1_motivation_components
def chart_motivation_components():
    print('Generating: ch1_motivation_components')
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    trend = 50 + 0.2 * t
    seasonal = 8 * np.sin(2*np.pi*t/12)
    cyclical = 5 * np.sin(2*np.pi*t/50)
    irregular = np.random.normal(0, 2, n)
    total = trend + seasonal + cyclical + irregular

    fig, axes = plt.subplots(5, 1, figsize=(7, 5.5), sharex=True)

    axes[0].plot(total, color=BLUE, linewidth=0.8)
    axes[0].set_title('Original series = Trend + Seasonal + Cycle + Irregular', fontsize=9, fontweight='bold')
    axes[0].set_ylabel('Total')

    axes[1].plot(trend, color=RED, linewidth=1.2)
    axes[1].set_ylabel('Trend')

    axes[2].plot(seasonal, color=GREEN, linewidth=0.8)
    axes[2].set_ylabel('Seasonal')

    axes[3].plot(cyclical, color=ORANGE, linewidth=0.8)
    axes[3].set_ylabel('Cycle')

    axes[4].plot(irregular, color=GRAY, linewidth=0.5)
    axes[4].set_ylabel('Irregular')
    axes[4].set_xlabel('Time')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_motivation_components')


# CHART 28: ch1_motivation_everywhere
def chart_motivation_everywhere():
    print('Generating: ch1_motivation_everywhere')
    np.random.seed(42)
    n = 150

    fig, axes = plt.subplots(2, 3, figsize=(9, 3.5))

    labels = ['Financial markets', 'Macroeconomics', 'Energy',
              'Climate/Weather', 'Health', 'Retail/Sales']
    colors_list = [BLUE, RED, GREEN, ORANGE, PURPLE, GRAY]

    for i, (ax, label, col) in enumerate(zip(axes.flat, labels, colors_list)):
        t = np.arange(n)
        if i == 0:  # financial
            y = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, n)))
        elif i == 1:  # macro
            y = 100 + 0.3*t + np.random.normal(0, 3, n)
        elif i == 2:  # energy
            y = 500 + 100*np.sin(2*np.pi*t/12) + 0.5*t + np.random.normal(0, 20, n)
        elif i == 3:  # climate
            y = 15 + 12*np.sin(2*np.pi*t/12) + np.random.normal(0, 2, n)
        elif i == 4:  # health
            y = np.maximum(5 + 0.1*t + 15*np.exp(-((t-75)/10)**2) + np.random.normal(0, 2, n), 0)
        else:  # retail
            y = 200 + 50*np.sin(2*np.pi*t/12) + 0.3*t + np.random.normal(0, 15, n)
        ax.plot(y, color=col, linewidth=0.7)
        ax.set_title(label, fontsize=8, fontweight='bold')
        if i >= 3:
            ax.set_xlabel('Time')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_motivation_everywhere')


# CHART 29: ch1_motivation_forecast
def chart_motivation_forecast():
    print('Generating: ch1_motivation_forecast')
    np.random.seed(42)
    n = 100
    h = 20
    y = 50 + 0.3*np.arange(n) + 5*np.sin(2*np.pi*np.arange(n)/12) + np.random.normal(0, 2, n)

    # Simple forecast
    t_future = np.arange(n, n+h)
    forecast = 50 + 0.3*t_future + 5*np.sin(2*np.pi*t_future/12)
    ci_lower = forecast - 2 * np.linspace(2, 5, h)
    ci_upper = forecast + 2 * np.linspace(2, 5, h)

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(range(n), y, color=BLUE, linewidth=0.8, label='Observed data')
    ax.plot(range(n, n+h), forecast, color=RED, linewidth=1.2, linestyle='--', label='Forecast')
    ax.fill_between(range(n, n+h), ci_lower, ci_upper, alpha=0.15, color=RED, label='95% CI')
    ax.axvline(n, color=GRAY, linewidth=0.8, linestyle=':', alpha=0.5)
    ax.set_title('Time series forecasting', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    add_legend_bottom(ax, ncol=3)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_motivation_forecast')


# CHART 30: ch1_def_timeseries
def chart_def_timeseries():
    print('Generating: ch1_def_timeseries')
    np.random.seed(42)
    n = 50
    y = 10 + 0.2*np.arange(n) + 3*np.sin(2*np.pi*np.arange(n)/12) + np.random.normal(0, 1, n)

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(y, color=BLUE, linewidth=1.0, marker='o', markersize=3)
    ax.set_title(r'Time series: $\{X_t\}_{t=1}^{T}$ - observations ordered in time', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel(r'$X_t$')

    # Annotate a few points
    for i in [5, 15, 25, 35, 45]:
        if i < n:
            ax.annotate(f'$X_{{{i}}}$', xy=(i, y[i]), xytext=(i+2, y[i]+2),
                       fontsize=7, color=RED,
                       arrowprops=dict(arrowstyle='->', color=RED, lw=0.5))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_def_timeseries')


# CHART 31: ch1_def_stochastic_ro (Romanian version)
def chart_def_stochastic_ro():
    print('Generating: ch1_def_stochastic_ro')
    # Same as ch1_def_stochastic but with Romanian labels
    np.random.seed(42)
    n = 200
    colors = [BLUE, RED, GREEN, ORANGE, PURPLE, GRAY, LIGHT_BLUE, LIGHT_RED]

    fig, ax = plt.subplots(figsize=(7, 3.0))
    for i in range(8):
        y = generate_ar1(n, phi=0.7, sigma=1.0)
        ax.plot(y, color=colors[i % len(colors)], alpha=0.6, linewidth=0.8,
                label=f'Realization {i+1}' if i < 5 else None)

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4)
    ax.set_title('Multiple realizations of a stochastic process', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel(r'$X_t(\omega)$')
    add_legend_bottom(ax, ncol=5)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_def_stochastic_ro')


# CHART 32: ch1_def_moving_average
def chart_def_moving_average():
    print('Generating: ch1_def_moving_average')
    np.random.seed(42)
    n = 200
    y = 10 + 0.05*np.arange(n) + 3*np.sin(2*np.pi*np.arange(n)/12) + np.random.normal(0, 2, n)
    ma5 = pd.Series(y).rolling(5).mean()
    ma20 = pd.Series(y).rolling(20).mean()

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(y, color=GRAY, linewidth=0.5, alpha=0.6, label='Raw data')
    ax.plot(ma5, color=BLUE, linewidth=1.0, label='MA(5)')
    ax.plot(ma20, color=RED, linewidth=1.2, label='MA(20)')
    ax.set_title('Moving Average', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    add_legend_bottom(ax, ncol=3)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_def_moving_average')


# CHART 33: ch1_def_ets
def chart_def_ets():
    print('Generating: ch1_def_ets')
    np.random.seed(42)
    n = 100
    y = 50 + 0.3*np.arange(n) + 8*np.sin(2*np.pi*np.arange(n)/12) + np.random.normal(0, 2, n)

    # Simple exponential smoothing
    alpha_vals = [0.1, 0.5, 0.9]
    colors_ets = [BLUE, GREEN, RED]

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(y, color=GRAY, linewidth=0.5, alpha=0.5, label='Raw data')

    for alpha, col in zip(alpha_vals, colors_ets):
        ses = np.zeros(n)
        ses[0] = y[0]
        for t in range(1, n):
            ses[t] = alpha * y[t] + (1-alpha) * ses[t-1]
        ax.plot(ses, color=col, linewidth=1.0, label=rf'$\alpha = {alpha}$')

    ax.set_title('Simple exponential smoothing (ETS)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    add_legend_bottom(ax, ncol=4)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_def_ets')


# CHART 34: ch1_def_stl
def chart_def_stl():
    print('Generating: ch1_def_stl')
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    trend = 50 + 0.2*t
    seasonal = 8*np.sin(2*np.pi*t/12)
    remainder = np.random.normal(0, 2, n)
    total = trend + seasonal + remainder

    fig, axes = plt.subplots(4, 1, figsize=(7, 4.5), sharex=True)

    axes[0].plot(total, color=BLUE, linewidth=0.7)
    axes[0].set_ylabel('Original')
    axes[0].set_title('STL Decomposition', fontsize=10, fontweight='bold')

    axes[1].plot(trend, color=RED, linewidth=1.0)
    axes[1].set_ylabel('Trend')

    axes[2].plot(seasonal, color=GREEN, linewidth=0.8)
    axes[2].set_ylabel('Seasonal')

    axes[3].plot(remainder, color=GRAY, linewidth=0.5)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_def_stl')


# CHART 35: ch1_differencing
def chart_differencing():
    print('Generating: ch1_differencing')
    np.random.seed(42)
    n = 200
    rw = generate_random_walk(n, drift=0.1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))

    axes[0].plot(rw, color=BLUE, linewidth=0.8, label=r'$X_t$ (random walk)')
    axes[0].set_title('Original series (non-stationary)', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$X_t$')
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    diff = np.diff(rw)
    axes[1].plot(diff, color=GREEN, linewidth=0.5, label=r'$\Delta X_t = X_t - X_{t-1}$')
    axes[1].axhline(diff.mean(), color=RED, linewidth=0.8, linestyle='--', label=f'Mean = {diff.mean():.2f}')
    axes[1].set_title('After differencing (stationary)', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel(r'$\Delta X_t$')
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'ch1_differencing')


# CHART 36: ch1_trend_types
def chart_trend_types():
    print('Generating: ch1_trend_types')
    np.random.seed(42)
    n = 200
    t = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    # Linear trend
    y_lin = 10 + 0.3*t + np.random.normal(0, 3, n)
    axes[0].plot(y_lin, color=BLUE, linewidth=0.6)
    axes[0].plot(10 + 0.3*t, color=RED, linewidth=1.2, linestyle='--')
    axes[0].set_title('Linear trend', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$X_t$')

    # Exponential trend
    y_exp = 10 * np.exp(0.01*t) + np.random.normal(0, 3, n)
    axes[1].plot(y_exp, color=BLUE, linewidth=0.6)
    axes[1].plot(10 * np.exp(0.01*t), color=RED, linewidth=1.2, linestyle='--')
    axes[1].set_title('Exponential trend', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Time')

    # Stochastic trend (random walk)
    y_stoch = generate_random_walk(n, drift=0.1)
    axes[2].plot(y_stoch, color=BLUE, linewidth=0.8)
    axes[2].plot(0.1*t, color=RED, linewidth=1.2, linestyle='--')
    axes[2].set_title('Stochastic trend (RW)', fontsize=9, fontweight='bold')
    axes[2].set_xlabel('Time')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_trend_types')


# CHART 37: ch1_unit_root_series
def chart_unit_root_series():
    print('Generating: ch1_unit_root_series')
    np.random.seed(42)
    n = 200

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    # Pure random walk
    rw = generate_random_walk(n)
    axes[0].plot(rw, color=BLUE, linewidth=0.8)
    axes[0].set_title(r'RW: $X_t = X_{t-1} + \varepsilon_t$', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$X_t$')

    # RW with drift
    rw_d = generate_random_walk(n, drift=0.1)
    axes[1].plot(rw_d, color=RED, linewidth=0.8)
    axes[1].set_title(r'RW + drift: $X_t = c + X_{t-1} + \varepsilon_t$', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Time')

    # Near unit root
    y_near = np.zeros(n)
    eps = np.random.normal(0, 1, n)
    for t in range(1, n):
        y_near[t] = 0.98 * y_near[t-1] + eps[t]
    axes[2].plot(y_near, color=GREEN, linewidth=0.8)
    axes[2].set_title(r'Near unit root: $\phi = 0.98$', fontsize=9, fontweight='bold')
    axes[2].set_xlabel('Time')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_unit_root_series')


# CHART 53: ch1_white_noise_types
def chart_white_noise_types():
    """3 types of white noise: weak, strong (iid), Gaussian."""
    print('Generating: ch1_white_noise_types')
    np.random.seed(42)
    n = 300

    # 1) Weak WN: uncorrelated but dependent (GARCH-like)
    # Generate using squared dependence: eps_t = sigma_t * z_t, z~N(0,1)
    z = np.random.normal(0, 1, n)
    sigma2 = np.ones(n)
    for t in range(1, n):
        sigma2[t] = 0.1 + 0.7 * z[t-1]**2 * sigma2[t-1]
    weak_wn = np.sqrt(sigma2) * np.random.normal(0, 1, n)
    # Center and scale for display
    weak_wn = (weak_wn - weak_wn.mean()) / weak_wn.std()

    # 2) Strong WN (iid): t-distribution (non-Gaussian but iid)
    strong_wn = stats.t.rvs(df=5, size=n, random_state=123)
    strong_wn = (strong_wn - strong_wn.mean()) / strong_wn.std()

    # 3) Gaussian WN: N(0,1)
    gauss_wn = np.random.normal(0, 1, n)

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))

    titles = ['Weak White Noise', 'Strong White Noise (i.i.d.)', 'Gaussian White Noise']
    data = [weak_wn, strong_wn, gauss_wn]
    colors = [ORANGE, BLUE, GREEN]
    subtitles = ['Uncorrelated, nonlinear dependence', r'$\varepsilon_t \sim t_5$ (i.i.d.)',
                 r'$\varepsilon_t \sim N(0,1)$ (i.i.d.)']

    for i, (ax, d, c, t, sub) in enumerate(zip(axes, data, colors, titles, subtitles)):
        ax.plot(d, color=c, linewidth=0.4, alpha=0.8)
        ax.axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
        ax.set_title(t, fontsize=8, fontweight='bold')
        ax.set_xlabel('Time', fontsize=7)
        ax.set_ylabel(r'$\varepsilon_t$', fontsize=7)
        ax.tick_params(labelsize=6)
        # Add subtitle
        ax.text(0.5, -0.22, sub, transform=ax.transAxes, fontsize=7,
                ha='center', va='top', style='italic', color=GRAY)

    fig.tight_layout(w_pad=2.5)
    fig.subplots_adjust(bottom=0.22)
    save_chart(fig, 'ch1_white_noise_types')


# CHART 38: ch1_white_noise_test
def chart_white_noise_test():
    print('Generating: ch1_white_noise_test')
    np.random.seed(42)
    n = 300
    wn = generate_white_noise(n)

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    axes[0].plot(wn, color=BLUE, linewidth=0.4)
    axes[0].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[0].set_title('White noise', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel(r'$\varepsilon_t$')

    # ACF
    acf_vals = acf(wn, nlags=20)
    axes[1].bar(range(len(acf_vals)), acf_vals, color=BLUE, width=0.5, alpha=0.7)
    ci = 1.96 / np.sqrt(n)
    axes[1].axhline(ci, color=RED, linewidth=0.8, linestyle='--', label='95% CI')
    axes[1].axhline(-ci, color=RED, linewidth=0.8, linestyle='--')
    axes[1].set_title('ACF (all within bands)', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('ACF')
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    # Histogram
    axes[2].hist(wn, bins=30, color=BLUE, alpha=0.6, edgecolor='white', density=True)
    x_r = np.linspace(-4, 4, 100)
    axes[2].plot(x_r, stats.norm.pdf(x_r, 0, 1), color=RED, linewidth=1.5, label='N(0,1)')
    axes[2].set_title('Distribution', fontsize=9, fontweight='bold')
    axes[2].set_xlabel(r'$\varepsilon_t$')
    axes[2].set_ylabel('Density')
    axes[2].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'ch1_white_noise_test')


# CHART 39: ch1_transform_sequence (English version)
def chart_transform_sequence():
    print('Generating: ch1_transform_sequence')
    load_sp500_data()
    prices = SP500_PRICES
    log_prices = np.log(prices)
    returns = SP500_RETURNS
    dates = SP500_DATES

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    axes[0].plot(dates[:len(prices)], prices, color=BLUE, linewidth=0.8, label=r'$P_t$')
    axes[0].set_title(r'S&P 500 $P_t$', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    axes[1].plot(dates[:len(log_prices)], log_prices, color=GREEN, linewidth=0.8, label=r'$\ln(P_t)$')
    axes[1].set_title(r'$\ln(P_t)$', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Log price')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    axes[2].plot(dates[1:len(returns)+1], returns, color=RED, linewidth=0.4, alpha=0.8, label=r'$r_t = \Delta \ln(P_t)$')
    axes[2].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[2].set_title(r'Returns $r_t$', fontsize=9, fontweight='bold')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Return')
    axes[2].tick_params(axis='x', rotation=30)
    axes[2].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'ch1_transform_sequence')


# CHART 40: ch1_hp_filter_cycle
def chart_hp_filter_cycle():
    print('Generating: ch1_hp_filter_cycle')
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    trend = 100 + 0.3*t + 0.005*t**1.3
    cycle = 8*np.sin(2*np.pi*t/40) + 5*np.sin(2*np.pi*t/20)
    noise = np.random.normal(0, 2, n)
    y = trend + cycle + noise

    # Simple HP filter approximation
    from scipy.signal import filtfilt
    # Use Hodrick-Prescott: approximate with low-pass filter
    lam = 1600  # quarterly
    # Simple smoothing
    smooth_trend = pd.Series(y).rolling(20, center=True, min_periods=1).mean().values
    hp_cycle = y - smooth_trend

    fig, axes = plt.subplots(2, 1, figsize=(7, 3.5), sharex=True)

    axes[0].plot(y, color=BLUE, linewidth=0.7, label='Original series')
    axes[0].plot(smooth_trend, color=RED, linewidth=1.5, label='Trend (HP)')
    axes[0].set_title('Hodrick-Prescott filter: Trend', fontsize=9, fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    axes[1].plot(hp_cycle, color=GREEN, linewidth=0.7, label='Cycle (HP)')
    axes[1].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[1].set_title('Cyclical component', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Cycle')
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(h_pad=2.0)
    save_chart(fig, 'ch1_hp_filter_cycle')


# CHART 41: ch1_hp_filter_lambda
def chart_hp_filter_lambda():
    print('Generating: ch1_hp_filter_lambda')
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    y = 100 + 0.3*t + 8*np.sin(2*np.pi*t/30) + np.random.normal(0, 3, n)

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(y, color=GRAY, linewidth=0.5, alpha=0.5, label='Raw data')

    lambdas = [10, 100, 1600]
    colors_hp = [BLUE, RED, GREEN]
    labels_hp = [r'$\lambda = 10$ (flexible)', r'$\lambda = 100$', r'$\lambda = 1600$ (rigid)']

    for lam, col, lab in zip(lambdas, colors_hp, labels_hp):
        window = max(3, int(np.sqrt(lam / 10)))
        smooth = pd.Series(y).rolling(window, center=True, min_periods=1).mean().values
        ax.plot(smooth, color=col, linewidth=1.2, label=lab)

    ax.set_title(r'HP filter: effect of $\lambda$', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    add_legend_bottom(ax, ncol=4)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_hp_filter_lambda')


# CHART 42: ch1_cyclical_component
def chart_cyclical_component():
    print('Generating: ch1_cyclical_component')
    np.random.seed(42)
    n = 200
    t = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))

    # Business cycle
    bc = 5*np.sin(2*np.pi*t/40) + 3*np.sin(2*np.pi*t/80) + np.random.normal(0, 1, n)
    axes[0].plot(bc, color=BLUE, linewidth=0.8)
    axes[0].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[0].fill_between(t, 0, bc, where=bc>0, alpha=0.1, color=GREEN, label='Expansion')
    axes[0].fill_between(t, 0, bc, where=bc<0, alpha=0.1, color=RED, label='Contraction')
    axes[0].set_title('Business cycle', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Cyclical component')
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    # Different cycle lengths
    c1 = np.sin(2*np.pi*t/20)
    c2 = np.sin(2*np.pi*t/50)
    c3 = np.sin(2*np.pi*t/100)
    axes[1].plot(c1, color=BLUE, linewidth=0.8, label='Short cycle (20)')
    axes[1].plot(c2, color=RED, linewidth=1.0, label='Medium cycle (50)')
    axes[1].plot(c3, color=GREEN, linewidth=1.2, label='Long cycle (100)')
    axes[1].set_title('Cycles of different lengths', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Time')
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'ch1_cyclical_component')


# CHART 43: ch1_quiz1_components
def chart_quiz1_components():
    print('Generating: ch1_quiz1_components')
    np.random.seed(42)
    n = 150
    t = np.arange(n)
    trend = 0.2*t
    seasonal = 5*np.sin(2*np.pi*t/12)
    noise = np.random.normal(0, 1.5, n)
    total = 50 + trend + seasonal + noise

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(total, color=BLUE, linewidth=0.8, label='Observed series')
    ax.plot(50 + trend, color=RED, linewidth=1.2, linestyle='--', label='Trend')
    ax.set_title('Identify the components of the series', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    add_legend_bottom(ax, ncol=2)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_quiz1_components')


# CHART 44: ch1_quiz3_acf
def chart_quiz3_acf():
    print('Generating: ch1_quiz3_acf')
    np.random.seed(42)
    n = 300

    fig, axes = plt.subplots(1, 3, figsize=(8, 2.5))

    # WN ACF
    wn = generate_white_noise(n)
    acf_wn = acf(wn, nlags=20)
    axes[0].bar(range(len(acf_wn)), acf_wn, color=BLUE, width=0.5, alpha=0.7)
    ci = 1.96 / np.sqrt(n)
    axes[0].axhline(ci, color=RED, linewidth=0.6, linestyle='--')
    axes[0].axhline(-ci, color=RED, linewidth=0.6, linestyle='--')
    axes[0].set_title('ACF: Proces A', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')

    # AR(1) ACF
    ar = generate_ar1(n, phi=0.8)
    acf_ar = acf(ar, nlags=20)
    axes[1].bar(range(len(acf_ar)), acf_ar, color=GREEN, width=0.5, alpha=0.7)
    axes[1].axhline(ci, color=RED, linewidth=0.6, linestyle='--')
    axes[1].axhline(-ci, color=RED, linewidth=0.6, linestyle='--')
    axes[1].set_title('ACF: Proces B', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Lag')

    # RW ACF
    rw = generate_random_walk(n)
    acf_rw = acf(rw, nlags=20)
    axes[2].bar(range(len(acf_rw)), acf_rw, color=RED, width=0.5, alpha=0.7)
    axes[2].axhline(ci, color=RED, linewidth=0.6, linestyle='--')
    axes[2].axhline(-ci, color=RED, linewidth=0.6, linestyle='--')
    axes[2].set_title('ACF: Proces C', fontsize=9, fontweight='bold')
    axes[2].set_xlabel('Lag')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_quiz3_acf')


# CHART 45: ch1_quiz5_forecast_errors
def chart_quiz5_forecast_errors():
    print('Generating: ch1_quiz5_forecast_errors')
    np.random.seed(42)
    n = 80
    h = 20
    y = 50 + 0.2*np.arange(n+h) + np.random.normal(0, 3, n+h)
    y_obs = y[:n]
    y_true = y[n:]

    # Forecast
    forecast = np.full(h, y_obs[-1] + 0.2 * np.arange(1, h+1))
    errors = y_true - forecast

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))

    axes[0].plot(range(n), y_obs, color=BLUE, linewidth=0.8, label='Observed')
    axes[0].plot(range(n, n+h), y_true, color=GRAY, linewidth=0.8, linestyle=':', label='Actual')
    axes[0].plot(range(n, n+h), forecast, color=RED, linewidth=1.2, linestyle='--', label='Forecast')
    axes[0].axvline(n, color=GRAY, linewidth=0.5, linestyle=':')
    axes[0].set_title('Forecast vs Actual', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    # Error metrics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    axes[1].bar(range(h), errors, color=[GREEN if e >= 0 else RED for e in errors], alpha=0.7, width=0.6)
    axes[1].axhline(0, color=GRAY, linewidth=0.5, linestyle='--')
    axes[1].set_title(f'Forecast errors (MAE={mae:.1f}, RMSE={rmse:.1f})', fontsize=9, fontweight='bold')
    axes[1].set_xlabel('Horizon')
    axes[1].set_ylabel('Error')

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'ch1_quiz5_forecast_errors')


# CHART 46: ch1_quiz6_decomposition
def chart_quiz6_decomposition():
    print('Generating: ch1_quiz6_decomposition')
    np.random.seed(42)
    n = 150
    t = np.arange(n)
    trend = 0.15*t
    seasonal = 5*np.sin(2*np.pi*t/12)
    noise = np.random.normal(0, 1, n)
    y = 30 + trend + seasonal + noise

    fig, axes = plt.subplots(4, 1, figsize=(7, 4.5), sharex=True)

    axes[0].plot(y, color=BLUE, linewidth=0.7)
    axes[0].set_title('Original series', fontsize=9, fontweight='bold')
    axes[0].set_ylabel('Total')

    axes[1].plot(30 + trend, color=RED, linewidth=1.0)
    axes[1].set_ylabel('Trend')

    axes[2].plot(seasonal, color=GREEN, linewidth=0.8)
    axes[2].set_ylabel('Seasonal')

    axes[3].plot(noise, color=GRAY, linewidth=0.5)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_quiz6_decomposition')


# CHART 47: ch1_ts_patterns
def chart_ts_patterns():
    print('Generating: ch1_ts_patterns')
    np.random.seed(42)
    n = 150

    fig, axes = plt.subplots(2, 3, figsize=(9, 3.5))

    # Trend
    t = np.arange(n)
    axes[0, 0].plot(50 + 0.3*t + np.random.normal(0, 2, n), color=BLUE, linewidth=0.7)
    axes[0, 0].set_title('Trend', fontsize=9, fontweight='bold')

    # Sezonalitate
    axes[0, 1].plot(10*np.sin(2*np.pi*t/12) + np.random.normal(0, 1, n), color=RED, linewidth=0.7)
    axes[0, 1].set_title('Seasonality', fontsize=9, fontweight='bold')

    # Cyclicality
    axes[0, 2].plot(8*np.sin(2*np.pi*t/40) + np.random.normal(0, 1, n), color=GREEN, linewidth=0.7)
    axes[0, 2].set_title('Cyclicality', fontsize=9, fontweight='bold')

    # White noise
    axes[1, 0].plot(np.random.normal(0, 1, n), color=ORANGE, linewidth=0.5)
    axes[1, 0].set_title('White noise', fontsize=9, fontweight='bold')
    axes[1, 0].set_xlabel('Time')

    # Random walk
    axes[1, 1].plot(np.cumsum(np.random.normal(0, 1, n)), color=PURPLE, linewidth=0.8)
    axes[1, 1].set_title('Random walk', fontsize=9, fontweight='bold')
    axes[1, 1].set_xlabel('Time')

    # Trend + seasonal
    axes[1, 2].plot(50 + 0.2*t + 5*np.sin(2*np.pi*t/12) + np.random.normal(0, 1, n),
                    color=GRAY, linewidth=0.7)
    axes[1, 2].set_title('Trend + Seasonal', fontsize=9, fontweight='bold')
    axes[1, 2].set_xlabel('Time')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_ts_patterns')


# CHART 48: ch1_decomposition
def chart_decomposition():
    print('Generating: ch1_decomposition')
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    trend = 100 + 0.5*t
    seasonal = 15*np.sin(2*np.pi*t/12)
    noise = np.random.normal(0, 3, n)
    y = trend + seasonal + noise

    fig, axes = plt.subplots(4, 1, figsize=(7, 4.5), sharex=True)

    axes[0].plot(y, color=BLUE, linewidth=0.7)
    axes[0].set_ylabel('Original')
    axes[0].set_title('Additive decomposition: Y = T + S + R', fontsize=10, fontweight='bold')

    axes[1].plot(trend, color=RED, linewidth=1.2)
    axes[1].set_ylabel('Trend')

    axes[2].plot(seasonal, color=GREEN, linewidth=0.8)
    axes[2].set_ylabel('Seasonal')

    axes[3].plot(noise, color=GRAY, linewidth=0.5)
    axes[3].axhline(0, color=GRAY, linewidth=0.3, linestyle='--')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_decomposition')


# CHART 49: ch1_exponential_smoothing
def chart_exponential_smoothing():
    print('Generating: ch1_exponential_smoothing')
    np.random.seed(42)
    n = 100
    y = 20 + 0.1*np.arange(n) + np.random.normal(0, 3, n)

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(y, color=GRAY, linewidth=0.5, alpha=0.5, label='Raw data')

    for alpha, col, lab in [(0.1, BLUE, '0.1 (smooth)'), (0.3, GREEN, '0.3'), (0.8, RED, '0.8 (reactive)')]:
        ses = np.zeros(n)
        ses[0] = y[0]
        for t in range(1, n):
            ses[t] = alpha * y[t] + (1-alpha) * ses[t-1]
        ax.plot(ses, color=col, linewidth=1.0, label=rf'$\alpha = {lab}$')

    ax.set_title('Simple exponential smoothing', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    add_legend_bottom(ax, ncol=4)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_exponential_smoothing')


# CHART 50: ch1_forecast_eval
def chart_forecast_eval():
    print('Generating: ch1_forecast_eval')
    np.random.seed(42)
    n = 80
    h = 20
    y = 50 + 0.2*np.arange(n+h) + 5*np.sin(2*np.pi*np.arange(n+h)/12) + np.random.normal(0, 2, n+h)
    y_train = y[:n]
    y_test = y[n:]

    # Two models
    f1 = np.mean(y_train[-12:]) + 0.2*np.arange(1, h+1)
    f2 = y_test + np.random.normal(0, 1.5, h)  # better model

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))

    axes[0].plot(range(n), y_train, color=BLUE, linewidth=0.8, label='Training')
    axes[0].plot(range(n, n+h), y_test, color=GRAY, linewidth=0.8, linestyle=':', label='Test')
    axes[0].plot(range(n, n+h), f1, color=RED, linewidth=1.0, linestyle='--', label='Model 1')
    axes[0].plot(range(n, n+h), f2, color=GREEN, linewidth=1.0, linestyle='--', label='Model 2')
    axes[0].axvline(n, color=GRAY, linewidth=0.5, linestyle=':')
    axes[0].set_title('Model comparison', fontsize=9, fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    # Error comparison
    metrics = ['MAE', 'RMSE', 'MAPE']
    m1_vals = [np.mean(np.abs(y_test-f1)), np.sqrt(np.mean((y_test-f1)**2)),
               np.mean(np.abs((y_test-f1)/y_test))*100]
    m2_vals = [np.mean(np.abs(y_test-f2)), np.sqrt(np.mean((y_test-f2)**2)),
               np.mean(np.abs((y_test-f2)/y_test))*100]

    x_pos = np.arange(len(metrics))
    w = 0.35
    axes[1].bar(x_pos - w/2, m1_vals, w, color=RED, alpha=0.7, label='Model 1')
    axes[1].bar(x_pos + w/2, m2_vals, w, color=GREEN, alpha=0.7, label='Model 2')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(metrics)
    axes[1].set_title('Error metrics', fontsize=9, fontweight='bold')
    axes[1].set_ylabel('Value')
    axes[1].legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    fig.tight_layout(w_pad=2.0)
    save_chart(fig, 'ch1_forecast_eval')


# CHART 51: ch1_moving_average
def chart_moving_average():
    print('Generating: ch1_moving_average')
    np.random.seed(42)
    n = 200
    y = 10 + 0.05*np.arange(n) + 3*np.sin(2*np.pi*np.arange(n)/12) + np.random.normal(0, 2, n)

    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.plot(y, color=GRAY, linewidth=0.5, alpha=0.5, label='Original series')

    for w, col, lab in [(5, BLUE, 'MA(5)'), (12, RED, 'MA(12)'), (24, GREEN, 'MA(24)')]:
        ma = pd.Series(y).rolling(w, center=True).mean()
        ax.plot(ma, color=col, linewidth=1.2, label=lab)

    ax.set_title('Moving average: smoothing', fontsize=10, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    add_legend_bottom(ax, ncol=4)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_moving_average')


# CHART 52: ch1_case_gdp_decomposition
def chart_case_gdp_decomposition():
    print('Generating: ch1_case_gdp_decomposition')
    np.random.seed(42)
    quarters = pd.date_range('2010-01-01', '2023-12-31', freq='QS')
    n = len(quarters)
    trend = np.linspace(30, 80, n)
    seasonal = 3.0 * np.sin(2 * np.pi * np.arange(n) / 4 + 0.5)
    covid_shock = np.zeros(n)
    covid_idx = 40
    covid_shock[covid_idx] = -12
    covid_shock[covid_idx+1] = -5
    covid_shock[covid_idx+2] = 3
    noise = np.random.normal(0, 1.5, n)
    gdp = trend + seasonal + covid_shock + noise

    fig, axes = plt.subplots(4, 1, figsize=(7, 4.5), sharex=True)

    axes[0].plot(quarters, gdp, color=BLUE, linewidth=0.8)
    axes[0].set_ylabel('GDP')
    axes[0].set_title('GDP Decomposition Romania', fontsize=10, fontweight='bold')

    axes[1].plot(quarters, trend, color=RED, linewidth=1.0)
    axes[1].set_ylabel('Trend')

    axes[2].plot(quarters, seasonal, color=GREEN, linewidth=0.8)
    axes[2].set_ylabel('Seasonal')

    axes[3].plot(quarters, covid_shock + noise, color=GRAY, linewidth=0.5)
    axes[3].axhline(0, color=GRAY, linewidth=0.3, linestyle='--')
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Period')

    axes[3].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
    axes[3].xaxis.set_major_locator(matplotlib.dates.YearLocator(2))

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_chart(fig, 'ch1_case_gdp_decomposition')


# =============================================================================
# RUN ALL CHARTS
# =============================================================================
if __name__ == '__main__':
    import os
    os.makedirs(CHARTS_DIR, exist_ok=True)

    print('='*60)
    print('REGENERATING ALL CHAPTER 1 CHARTS')
    print('='*60)

    # Referenced directly in tex file
    chart_stationary_nonstationary_examples()  # 1
    chart_def_stochastic()                     # 2
    chart_def_strict_stationarity()            # 3
    chart_def_weak_stationarity()              # 4
    chart_counterexample_stationarity()        # 5
    chart_ergodicity()                         # 6
    chart_wold_decomposition()                 # 7
    chart_def_lag_operator()                   # 8
    chart_differencing_effect()                # 9
    chart_transform_sequence_ro()              # 10
    chart_def_white_noise()                    # 11
    chart_random_walk()                        # 12
    chart_wn_rw()                              # 13
    chart_rw_vs_stationary()                   # 14
    chart_acf_examples()                       # 15
    chart_acf_pacf_examples()                  # 16
    chart_acf_theoretical()                    # 17
    chart_adf_test_visualization()             # 18
    chart_sp500_analysis()                     # 19
    chart_returns_distribution()               # 20
    chart_volatility_clustering()              # 21
    chart_case_gdp_raw()                       # 22
    chart_quiz2_stationarity()                 # 23
    chart_quiz4_wn_rw()                        # 24
    chart_stationarity()                       # 25

    # Additional charts in charts/ directory
    chart_motivation_real()                    # 26
    chart_motivation_components()              # 27
    chart_motivation_everywhere()              # 28
    chart_motivation_forecast()                # 29
    chart_def_timeseries()                     # 30
    chart_def_stochastic_ro()                  # 31
    chart_def_moving_average()                 # 32
    chart_def_ets()                            # 33
    chart_def_stl()                            # 34
    chart_differencing()                       # 35
    chart_trend_types()                        # 36
    chart_unit_root_series()                   # 37
    chart_white_noise_test()                   # 38
    chart_transform_sequence()                 # 39
    chart_hp_filter_cycle()                    # 40
    chart_hp_filter_lambda()                   # 41
    chart_cyclical_component()                 # 42
    chart_quiz1_components()                   # 43
    chart_quiz3_acf()                          # 44
    chart_quiz5_forecast_errors()              # 45
    chart_quiz6_decomposition()                # 46
    chart_ts_patterns()                        # 47
    chart_decomposition()                      # 48
    chart_exponential_smoothing()              # 49
    chart_forecast_eval()                      # 50
    chart_moving_average()                     # 51
    chart_case_gdp_decomposition()             # 52
    chart_white_noise_types()                   # 53

    print('='*60)
    print(f'ALL 53 CHAPTER 1 CHARTS REGENERATED SUCCESSFULLY')
    print(f'Output directory: {CHARTS_DIR}')
    print('='*60)
