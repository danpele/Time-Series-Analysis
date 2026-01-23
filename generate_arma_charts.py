"""
Generate Charts for Chapter 2: ARMA Models
Time Series Analysis Course

Charts are saved in the charts/ folder for use in LaTeX Beamer slides.
Style: Transparent background, no grid, legend outside bottom
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from scipy import stats

# Set random seed for reproducibility
np.random.seed(42)

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================
BLUE = '#1A3A6E'
RED = '#DC3545'
GREEN = '#2E7D32'
ORANGE = '#E07B00'
PURPLE = '#7B1FA2'
GRAY = '#666666'

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

def save_chart(fig, name):
    """Save chart in both PDF and PNG formats"""
    fig.savefig(f'charts/{name}.pdf', bbox_inches='tight', dpi=150, transparent=True)
    fig.savefig(f'charts/{name}.png', bbox_inches='tight', dpi=150, transparent=True)
    plt.close(fig)
    print(f"Saved: {name}")

# =============================================================================
# 1. AR(1) PROCESSES WITH DIFFERENT PHI VALUES
# =============================================================================
def plot_ar1_comparison():
    """Compare AR(1) processes with different phi values"""
    n = 200
    phi_values = [0.9, 0.5, -0.5, -0.9]
    colors = [BLUE, GREEN, ORANGE, RED]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (phi, color) in enumerate(zip(phi_values, colors)):
        ar = np.array([1, -phi])
        ma = np.array([1])
        process = ArmaProcess(ar, ma)
        data = process.generate_sample(nsample=n)

        axes[idx].plot(data, color=color, linewidth=0.8)
        axes[idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[idx].set_title(f'AR(1): $\\phi = {phi}$', fontweight='bold')
        axes[idx].set_xlabel('Time')
        axes[idx].set_ylabel('$X_t$')

        # Add persistence info
        if phi > 0:
            persistence = "High persistence" if phi > 0.7 else "Moderate persistence"
        else:
            persistence = "Oscillating" + (" strongly" if abs(phi) > 0.7 else "")
        axes[idx].text(0.02, 0.98, persistence, transform=axes[idx].transAxes,
                      fontsize=10, va='top', color=color)

    plt.tight_layout()
    save_chart(fig, 'ar1_comparison')

# =============================================================================
# 2. MA(1) PROCESSES WITH DIFFERENT THETA VALUES
# =============================================================================
def plot_ma1_comparison():
    """Compare MA(1) processes with different theta values"""
    n = 200
    theta_values = [0.8, 0.3, -0.3, -0.8]
    colors = [BLUE, GREEN, ORANGE, RED]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (theta, color) in enumerate(zip(theta_values, colors)):
        ar = np.array([1])
        ma = np.array([1, theta])
        process = ArmaProcess(ar, ma)
        data = process.generate_sample(nsample=n)

        axes[idx].plot(data, color=color, linewidth=0.8)
        axes[idx].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[idx].set_title(f'MA(1): $\\theta = {theta}$', fontweight='bold')
        axes[idx].set_xlabel('Time')
        axes[idx].set_ylabel('$X_t$')

    plt.tight_layout()
    save_chart(fig, 'ma1_comparison')

# =============================================================================
# 3. ARMA(1,1) PROCESS
# =============================================================================
def plot_arma11():
    """Show ARMA(1,1) process"""
    n = 300

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ARMA(1,1) with phi=0.7, theta=0.4
    ar = np.array([1, -0.7])
    ma = np.array([1, 0.4])
    process = ArmaProcess(ar, ma)
    data = process.generate_sample(nsample=n)

    axes[0].plot(data, color=BLUE, linewidth=0.8)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('ARMA(1,1): $\\phi=0.7$, $\\theta=0.4$', fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$X_t$')

    # ARMA(2,1)
    ar2 = np.array([1, -0.5, -0.3])
    ma2 = np.array([1, 0.4])
    process2 = ArmaProcess(ar2, ma2)
    data2 = process2.generate_sample(nsample=n)

    axes[1].plot(data2, color=GREEN, linewidth=0.8)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_title('ARMA(2,1): $\\phi_1=0.5$, $\\phi_2=0.3$, $\\theta=0.4$', fontweight='bold')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('$X_t$')

    plt.tight_layout()
    save_chart(fig, 'arma_examples')

# =============================================================================
# 4. ACF/PACF PATTERNS FOR MODEL IDENTIFICATION
# =============================================================================
def plot_acf_pacf_patterns():
    """Show ACF and PACF patterns for AR, MA, ARMA"""
    n = 500
    nlags = 15

    # Generate processes
    # AR(2)
    ar_ar2 = np.array([1, -0.6, -0.2])
    ma_ar2 = np.array([1])
    ar2_data = ArmaProcess(ar_ar2, ma_ar2).generate_sample(nsample=n)

    # MA(2)
    ar_ma2 = np.array([1])
    ma_ma2 = np.array([1, 0.6, 0.3])
    ma2_data = ArmaProcess(ar_ma2, ma_ma2).generate_sample(nsample=n)

    # ARMA(1,1)
    ar_arma = np.array([1, -0.7])
    ma_arma = np.array([1, 0.5])
    arma_data = ArmaProcess(ar_arma, ma_arma).generate_sample(nsample=n)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    datasets = [
        ('AR(2)', ar2_data, BLUE),
        ('MA(2)', ma2_data, GREEN),
        ('ARMA(1,1)', arma_data, ORANGE)
    ]

    for i, (name, data, color) in enumerate(datasets):
        # ACF
        acf_vals = acf(data, nlags=nlags)
        axes[i, 0].bar(range(len(acf_vals)), acf_vals, color=color, width=0.4)
        axes[i, 0].axhline(y=0, color='black', linewidth=0.5)
        axes[i, 0].axhline(y=1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
        axes[i, 0].axhline(y=-1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
        axes[i, 0].set_title(f'{name} - ACF', fontweight='bold')
        axes[i, 0].set_xlabel('Lag')
        axes[i, 0].set_ylabel('ACF')
        axes[i, 0].set_ylim(-0.4, 1.1)

        # PACF
        pacf_vals = pacf(data, nlags=nlags)
        axes[i, 1].bar(range(len(pacf_vals)), pacf_vals, color=color, width=0.4)
        axes[i, 1].axhline(y=0, color='black', linewidth=0.5)
        axes[i, 1].axhline(y=1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
        axes[i, 1].axhline(y=-1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
        axes[i, 1].set_title(f'{name} - PACF', fontweight='bold')
        axes[i, 1].set_xlabel('Lag')
        axes[i, 1].set_ylabel('PACF')
        axes[i, 1].set_ylim(-0.4, 1.1)

    # Add annotations
    axes[0, 0].text(0.95, 0.95, 'Decays', transform=axes[0, 0].transAxes,
                   ha='right', va='top', fontsize=12, color=BLUE, fontweight='bold')
    axes[0, 1].text(0.95, 0.95, 'Cuts off at lag 2', transform=axes[0, 1].transAxes,
                   ha='right', va='top', fontsize=12, color=BLUE, fontweight='bold')

    axes[1, 0].text(0.95, 0.95, 'Cuts off at lag 2', transform=axes[1, 0].transAxes,
                   ha='right', va='top', fontsize=12, color=GREEN, fontweight='bold')
    axes[1, 1].text(0.95, 0.95, 'Decays', transform=axes[1, 1].transAxes,
                   ha='right', va='top', fontsize=12, color=GREEN, fontweight='bold')

    axes[2, 0].text(0.95, 0.95, 'Decays', transform=axes[2, 0].transAxes,
                   ha='right', va='top', fontsize=12, color=ORANGE, fontweight='bold')
    axes[2, 1].text(0.95, 0.95, 'Decays', transform=axes[2, 1].transAxes,
                   ha='right', va='top', fontsize=12, color=ORANGE, fontweight='bold')

    plt.tight_layout()
    save_chart(fig, 'acf_pacf_patterns')

# =============================================================================
# 5. UNIT CIRCLE FOR STATIONARITY
# =============================================================================
def plot_unit_circle():
    """Visualize stationarity conditions with unit circle"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Stationary AR(2)
    phi1, phi2 = 0.5, 0.3
    roots_stat = np.roots([1, -phi1, -phi2])

    ax = axes[0]
    circle = Circle((0, 0), 1, fill=False, color=GRAY, linestyle='--', linewidth=2)
    ax.add_patch(circle)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)

    for root in roots_stat:
        ax.plot(root.real, root.imag, 'o', color=GREEN, markersize=15)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title(f'Stationary AR(2)\n$\\phi_1={phi1}$, $\\phi_2={phi2}$', fontweight='bold')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.text(0.05, 0.95, 'Roots OUTSIDE\nunit circle', transform=ax.transAxes,
            fontsize=12, va='top', color=GREEN, fontweight='bold')
    ax.fill_between(np.linspace(-1, 1, 100),
                    -np.sqrt(1 - np.linspace(-1, 1, 100)**2),
                    np.sqrt(1 - np.linspace(-1, 1, 100)**2),
                    alpha=0.1, color=RED)

    # Non-stationary AR(2)
    phi1_ns, phi2_ns = 1.2, -0.5
    roots_nonstat = np.roots([1, -phi1_ns, -phi2_ns])

    ax = axes[1]
    circle = Circle((0, 0), 1, fill=False, color=GRAY, linestyle='--', linewidth=2)
    ax.add_patch(circle)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)

    for root in roots_nonstat:
        ax.plot(root.real, root.imag, 'o', color=RED, markersize=15)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title(f'Non-Stationary AR(2)\n$\\phi_1={phi1_ns}$, $\\phi_2={phi2_ns}$', fontweight='bold')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.text(0.05, 0.95, 'Root INSIDE\nunit circle', transform=ax.transAxes,
            fontsize=12, va='top', color=RED, fontweight='bold')
    ax.fill_between(np.linspace(-1, 1, 100),
                    -np.sqrt(1 - np.linspace(-1, 1, 100)**2),
                    np.sqrt(1 - np.linspace(-1, 1, 100)**2),
                    alpha=0.1, color=RED)

    plt.tight_layout()
    save_chart(fig, 'unit_circle_stationarity')

# =============================================================================
# 6. IMPULSE RESPONSE FUNCTION
# =============================================================================
def plot_impulse_response():
    """Show impulse response functions for different models"""
    n_lags = 20

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # AR(1) with phi=0.8
    phi = 0.8
    irf_ar1 = [phi**i for i in range(n_lags)]
    markerline, stemlines, baseline = axes[0].stem(range(n_lags), irf_ar1, basefmt='gray')
    plt.setp(stemlines, color=BLUE)
    plt.setp(markerline, color=BLUE)
    axes[0].axhline(y=0, color='gray', linewidth=0.5)
    axes[0].set_title(f'AR(1): $\\phi = {phi}$', fontweight='bold')
    axes[0].set_xlabel('Lag $j$')
    axes[0].set_ylabel('$\\psi_j$')
    axes[0].set_ylim(-0.2, 1.1)

    # AR(1) with phi=-0.8
    phi_neg = -0.8
    irf_ar1_neg = [phi_neg**i for i in range(n_lags)]
    markerline, stemlines, baseline = axes[1].stem(range(n_lags), irf_ar1_neg, basefmt='gray')
    plt.setp(stemlines, color=GREEN)
    plt.setp(markerline, color=GREEN)
    axes[1].axhline(y=0, color='gray', linewidth=0.5)
    axes[1].set_title(f'AR(1): $\\phi = {phi_neg}$', fontweight='bold')
    axes[1].set_xlabel('Lag $j$')
    axes[1].set_ylabel('$\\psi_j$')

    # MA(1)
    theta = 0.6
    irf_ma1 = [1] + [theta] + [0]*(n_lags-2)
    markerline, stemlines, baseline = axes[2].stem(range(n_lags), irf_ma1, basefmt='gray')
    plt.setp(stemlines, color=ORANGE)
    plt.setp(markerline, color=ORANGE)
    axes[2].axhline(y=0, color='gray', linewidth=0.5)
    axes[2].set_title(f'MA(1): $\\theta = {theta}$', fontweight='bold')
    axes[2].set_xlabel('Lag $j$')
    axes[2].set_ylabel('$\\psi_j$')
    axes[2].set_ylim(-0.2, 1.3)

    plt.tight_layout()
    save_chart(fig, 'impulse_response')

# =============================================================================
# 7. THEORETICAL ACF FOR AR(1)
# =============================================================================
def plot_ar1_theoretical_acf():
    """Show theoretical ACF for AR(1) with different phi values"""
    lags = np.arange(0, 15)
    phi_values = [0.9, 0.5, -0.5, -0.9]
    colors = [BLUE, GREEN, ORANGE, RED]

    fig, ax = plt.subplots(figsize=(12, 6))

    for phi, color in zip(phi_values, colors):
        acf_theoretical = phi ** lags
        ax.plot(lags, acf_theoretical, 'o-', color=color, linewidth=2,
                markersize=8, label=f'$\\phi = {phi}$')

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Lag $h$', fontsize=12)
    ax.set_ylabel('$\\rho(h) = \\phi^h$', fontsize=12)
    ax.set_title('Theoretical ACF of AR(1): $\\rho(h) = \\phi^h$', fontweight='bold', fontsize=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks(lags)

    plt.tight_layout()
    save_chart(fig, 'ar1_theoretical_acf')

# =============================================================================
# 8. MODEL SELECTION: AIC VS BIC
# =============================================================================
def plot_aic_bic_comparison():
    """Compare AIC and BIC for model selection"""
    n = 300

    # Generate AR(2) data
    ar = np.array([1, -0.6, -0.2])
    ma = np.array([1])
    data = ArmaProcess(ar, ma).generate_sample(nsample=n)

    # Fit different models
    results = []
    for p in range(5):
        for q in range(5):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(data, order=(p, 0, q))
                fitted = model.fit()
                results.append({'p': p, 'q': q, 'AIC': fitted.aic, 'BIC': fitted.bic})
            except:
                pass

    results_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AIC heatmap
    aic_pivot = results_df.pivot(index='p', columns='q', values='AIC')
    im1 = axes[0].imshow(aic_pivot.values, cmap='RdYlGn_r', aspect='auto')
    axes[0].set_xticks(range(len(aic_pivot.columns)))
    axes[0].set_xticklabels(aic_pivot.columns)
    axes[0].set_yticks(range(len(aic_pivot.index)))
    axes[0].set_yticklabels(aic_pivot.index)
    axes[0].set_xlabel('q (MA order)')
    axes[0].set_ylabel('p (AR order)')
    axes[0].set_title('AIC Values', fontweight='bold')
    plt.colorbar(im1, ax=axes[0])

    # Mark best
    best_aic_idx = results_df['AIC'].idxmin()
    best_aic = results_df.loc[best_aic_idx]
    axes[0].plot(int(best_aic['q']), int(best_aic['p']), 's', color='white',
                 markersize=20, markeredgecolor='black', markeredgewidth=2)

    # BIC heatmap
    bic_pivot = results_df.pivot(index='p', columns='q', values='BIC')
    im2 = axes[1].imshow(bic_pivot.values, cmap='RdYlGn_r', aspect='auto')
    axes[1].set_xticks(range(len(bic_pivot.columns)))
    axes[1].set_xticklabels(bic_pivot.columns)
    axes[1].set_yticks(range(len(bic_pivot.index)))
    axes[1].set_yticklabels(bic_pivot.index)
    axes[1].set_xlabel('q (MA order)')
    axes[1].set_ylabel('p (AR order)')
    axes[1].set_title('BIC Values', fontweight='bold')
    plt.colorbar(im2, ax=axes[1])

    # Mark best
    best_bic_idx = results_df['BIC'].idxmin()
    best_bic = results_df.loc[best_bic_idx]
    axes[1].plot(int(best_bic['q']), int(best_bic['p']), 's', color='white',
                 markersize=20, markeredgecolor='black', markeredgewidth=2)

    plt.tight_layout()
    save_chart(fig, 'aic_bic_comparison')

# =============================================================================
# 9. FORECASTING WITH CONFIDENCE INTERVALS
# =============================================================================
def plot_arma_forecast():
    """Show ARMA forecasting with confidence intervals"""
    n = 200
    h = 30

    # Generate ARMA(1,1) data
    ar = np.array([1, -0.7])
    ma = np.array([1, 0.3])
    data = ArmaProcess(ar, ma).generate_sample(nsample=n)

    # Fit model
    model = ARIMA(data, order=(1, 0, 1))
    fitted = model.fit()

    # Forecast
    forecast = fitted.get_forecast(steps=h)
    forecast_mean = forecast.predicted_mean
    conf_int = forecast.conf_int()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Historical data
    ax.plot(range(n), data, color=BLUE, linewidth=1, label='Observed')

    # Forecast
    forecast_idx = range(n, n + h)
    ax.plot(forecast_idx, forecast_mean, color=RED, linewidth=2, label='Forecast')
    if hasattr(conf_int, 'iloc'):
        ax.fill_between(forecast_idx, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                        color=RED, alpha=0.2, label='95% CI')
    else:
        ax.fill_between(forecast_idx, conf_int[:, 0], conf_int[:, 1],
                        color=RED, alpha=0.2, label='95% CI')

    # Vertical line at forecast origin
    ax.axvline(x=n-1, color='gray', linestyle='--', alpha=0.5)

    # Add unconditional mean
    mu = fitted.params[0] if len(fitted.params) > 0 else 0
    ax.axhline(y=mu, color=GREEN, linestyle=':', alpha=0.7, label=f'Mean = {mu:.2f}')

    ax.set_title('ARMA(1,1) Forecast with 95% Confidence Interval', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('$X_t$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)

    # Add annotation
    ax.annotate('Forecasts revert\nto mean', xy=(n + h - 5, mu),
                xytext=(n + h - 15, mu + 2), fontsize=10,
                arrowprops=dict(arrowstyle='->', color=GREEN))

    plt.tight_layout()
    save_chart(fig, 'arma_forecast')

# =============================================================================
# 10. RESIDUAL DIAGNOSTICS
# =============================================================================
def plot_residual_diagnostics_arma():
    """Show residual diagnostic plots for ARMA model"""
    n = 300

    # Generate and fit
    ar = np.array([1, -0.6, -0.2])
    ma = np.array([1])
    data = ArmaProcess(ar, ma).generate_sample(nsample=n)

    model = ARIMA(data, order=(2, 0, 0))
    fitted = model.fit()
    resid = fitted.resid

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Residuals over time
    axes[0, 0].plot(resid, color=BLUE, linewidth=0.8)
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Residuals over Time', fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Residual')

    # Histogram
    axes[0, 1].hist(resid, bins=30, density=True, color=BLUE, alpha=0.7, edgecolor='white')
    x_range = np.linspace(resid.min(), resid.max(), 100)
    axes[0, 1].plot(x_range, stats.norm.pdf(x_range, resid.mean(), resid.std()),
                   color=RED, linewidth=2, label='Normal')
    axes[0, 1].set_title('Residual Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

    # ACF of residuals
    acf_resid = acf(resid, nlags=20)
    axes[1, 0].bar(range(len(acf_resid)), acf_resid, color=BLUE, width=0.4)
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].axhline(y=1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
    axes[1, 0].axhline(y=-1.96/np.sqrt(n), color=RED, linestyle='--', alpha=0.7)
    axes[1, 0].set_title('ACF of Residuals', fontweight='bold')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')
    axes[1, 0].text(0.95, 0.95, 'White noise', transform=axes[1, 0].transAxes,
                   ha='right', va='top', fontsize=11, color=GREEN, fontweight='bold')

    # Q-Q plot
    (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist='norm', fit=True)
    axes[1, 1].scatter(osm, osr, color=BLUE, alpha=0.6, s=20)
    q_range = abs(osm).max() * 1.1
    x_line = np.array([-q_range, q_range])
    axes[1, 1].plot(x_line, slope * x_line + intercept, color=RED, linewidth=2)
    axes[1, 1].set_xlim(-q_range, q_range)
    axes[1, 1].set_ylim(-q_range * slope + intercept - 0.5, q_range * slope + intercept + 0.5)
    axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
    axes[1, 1].set_xlabel('Theoretical Quantiles')
    axes[1, 1].set_ylabel('Sample Quantiles')

    plt.tight_layout()
    save_chart(fig, 'arma_residual_diagnostics')

# =============================================================================
# 11. LAG OPERATOR VISUALIZATION
# =============================================================================
def plot_lag_operator():
    """Visualize lag operator"""
    n = 50
    t = np.arange(n)
    x = np.sin(t * 0.3) + 0.5 * np.random.randn(n)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original series
    axes[0].plot(t, x, 'o-', color=BLUE, linewidth=1.5, markersize=5)
    axes[0].set_title('Original: $X_t$', fontweight='bold')
    axes[0].set_xlabel('Time $t$')
    axes[0].set_ylabel('$X_t$')

    # Lagged series
    axes[1].plot(t[:-1], x[:-1], 'o-', color=BLUE, linewidth=1.5, markersize=5, alpha=0.5)
    axes[1].plot(t[1:], x[:-1], 'o-', color=GREEN, linewidth=1.5, markersize=5, label='$LX_t = X_{t-1}$')
    axes[1].set_title('Lag Operator: $LX_t = X_{t-1}$', fontweight='bold')
    axes[1].set_xlabel('Time $t$')
    axes[1].set_ylabel('$LX_t$')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

    # Difference
    diff = x[1:] - x[:-1]
    axes[2].plot(t[1:], diff, 'o-', color=RED, linewidth=1.5, markersize=5)
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_title('First Difference: $(1-L)X_t = X_t - X_{t-1}$', fontweight='bold')
    axes[2].set_xlabel('Time $t$')
    axes[2].set_ylabel('$\\Delta X_t$')

    plt.tight_layout()
    save_chart(fig, 'lag_operator')

# =============================================================================
# 12. YULE-WALKER VISUALIZATION
# =============================================================================
def plot_yule_walker():
    """Visualize Yule-Walker equations for AR(2)"""
    phi1, phi2 = 0.6, 0.2

    # Theoretical ACF
    rho = np.zeros(10)
    rho[0] = 1
    rho[1] = phi1 / (1 - phi2)
    for k in range(2, 10):
        rho[k] = phi1 * rho[k-1] + phi2 * rho[k-2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ACF plot
    axes[0].bar(range(len(rho)), rho, color=BLUE, width=0.5)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_title(f'Theoretical ACF for AR(2)\n$\\phi_1={phi1}$, $\\phi_2={phi2}$', fontweight='bold')
    axes[0].set_xlabel('Lag $h$')
    axes[0].set_ylabel('$\\rho(h)$')

    # Yule-Walker equations visualization
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Yule-Walker Equations', fontweight='bold')

    # Add equations as text
    equations = [
        r'$\rho(1) = \phi_1 + \phi_2 \rho(1)$',
        r'$\rho(2) = \phi_1 \rho(1) + \phi_2$',
        '',
        r'Matrix form: $R \cdot \phi = \rho$',
        '',
        'R = autocorrelation matrix',
        '',
        r'Solution: $\hat{\phi} = R^{-1} \rho$'
    ]

    for i, eq in enumerate(equations):
        ax.text(5, 9 - i * 1.1, eq, ha='center', va='top', fontsize=13)

    plt.tight_layout()
    save_chart(fig, 'yule_walker')

# =============================================================================
# 13. AR(1) VARIANCE DERIVATION
# =============================================================================
def plot_ar1_variance():
    """Visualize AR(1) variance as function of phi"""
    phi_values = np.linspace(-0.99, 0.99, 100)
    sigma_sq = 1

    variance = sigma_sq / (1 - phi_values**2)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(phi_values, variance, color=BLUE, linewidth=2.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Mark some points
    for phi in [0.5, 0.8, 0.9]:
        var = sigma_sq / (1 - phi**2)
        ax.plot(phi, var, 'o', color=RED, markersize=10)
        ax.annotate(f'$\\phi={phi}$\n$\\gamma(0)={var:.2f}$',
                   xy=(phi, var), xytext=(phi+0.1, var+0.5),
                   fontsize=10, ha='left')

    ax.set_xlabel('$\\phi$', fontsize=12)
    ax.set_ylabel('$\\gamma(0) = \\sigma^2/(1-\\phi^2)$', fontsize=12)
    ax.set_title('AR(1) Variance as Function of $\\phi$ (with $\\sigma^2=1$)', fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 10)

    # Add note
    ax.text(0.02, 0.98, 'Variance increases\nas $|\\phi| \\to 1$',
            transform=ax.transAxes, fontsize=11, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_chart(fig, 'ar1_variance')

# =============================================================================
# 14. MODEL IDENTIFICATION SUMMARY
# =============================================================================
def plot_model_identification_table():
    """Create visual summary of model identification"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Table data
    table_data = [
        ['Model', 'ACF Pattern', 'PACF Pattern'],
        ['AR(p)', 'Exponential decay\nor damped oscillation', 'Cuts off after lag p'],
        ['MA(q)', 'Cuts off after lag q', 'Exponential decay\nor damped oscillation'],
        ['ARMA(p,q)', 'Exponential decay\nafter lag q-p', 'Exponential decay\nafter lag p-q']
    ]

    # Colors
    colors = [
        [BLUE, BLUE, BLUE],
        ['white', '#e8f4e8', '#ffe8e8'],
        ['white', '#ffe8e8', '#e8f4e8'],
        ['white', '#e8f0ff', '#e8f0ff']
    ]

    table = ax.table(cellText=table_data, cellColours=colors,
                     loc='center', cellLoc='center',
                     colWidths=[0.2, 0.4, 0.4])

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)

    # Style header
    for j in range(3):
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    for i in range(1, 4):
        table[(i, 0)].set_text_props(fontweight='bold')

    ax.set_title('Model Identification: ACF/PACF Patterns', fontweight='bold',
                 fontsize=14, pad=20)

    plt.tight_layout()
    save_chart(fig, 'model_identification_table')

# =============================================================================
# 15. BOX-JENKINS METHODOLOGY FLOWCHART (IMPROVED)
# =============================================================================
def plot_box_jenkins_flowchart():
    """Create professional Box-Jenkins methodology flowchart"""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Colors
    step_colors = ['#2E86AB', '#28A745', '#F39C12', '#9B59B6']  # Blue, Green, Orange, Purple
    decision_color = '#E74C3C'
    detail_color = '#34495E'

    # Draw main process boxes
    def draw_box(ax, x, y, width, height, text, color, step_num=None):
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                             boxstyle="round,pad=0.02,rounding_size=0.2",
                             facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(box)
        if step_num:
            ax.text(x, y + height/2 - 0.3, f'STEP {step_num}', ha='center', va='top',
                   fontsize=11, color='white', fontweight='bold', alpha=0.9)
            ax.text(x, y - 0.15, text, ha='center', va='center', fontsize=10,
                   color='white', fontweight='bold', wrap=True)
        else:
            ax.text(x, y, text, ha='center', va='center', fontsize=9,
                   color='white', fontweight='bold', wrap=True)

    # Draw diamond decision box
    def draw_diamond(ax, x, y, size, text, color):
        diamond = Polygon([(x, y + size), (x + size, y), (x, y - size), (x - size, y)],
                         facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(diamond)
        ax.text(x, y, text, ha='center', va='center', fontsize=10,
               color='white', fontweight='bold')

    # Main steps (left column)
    draw_box(ax, 4, 10, 4.5, 1.6, 'IDENTIFICATION\nAnalyze ACF/PACF\nDetermine p, d, q', step_colors[0], 1)
    draw_box(ax, 4, 7.5, 4.5, 1.6, 'ESTIMATION\nEstimate parameters\nMLE or Yule-Walker', step_colors[1], 2)
    draw_box(ax, 4, 5, 4.5, 1.6, 'DIAGNOSTIC CHECK\nResidual analysis\nLjung-Box test', step_colors[2], 3)
    draw_box(ax, 4, 2, 4.5, 1.6, 'FORECASTING\nGenerate predictions\nConfidence intervals', step_colors[3], 4)

    # Decision diamond (right of Step 3)
    draw_diamond(ax, 9.5, 5, 1.0, 'Model\nOK?', decision_color)

    # Side boxes with details (right column)
    draw_box(ax, 11, 10, 3.8, 1.3, 'Check stationarity\nTransform if needed', detail_color)
    draw_box(ax, 11, 7.5, 3.8, 1.3, 'Maximum Likelihood\nLeast Squares', detail_color)
    draw_box(ax, 11, 2, 3.8, 1.3, 'Point forecasts\n95% intervals', detail_color)

    # Arrow style
    arrow_main = dict(arrowstyle='->', color='#555555', lw=2, mutation_scale=15)
    arrow_yes = dict(arrowstyle='->', color='#28A745', lw=2.5, mutation_scale=15)
    arrow_no = dict(arrowstyle='->', color='#E74C3C', lw=2.5, mutation_scale=15)

    # Main flow arrows (Step 1 → 2 → 3)
    ax.annotate('', xy=(4, 9.2), xytext=(4, 8.3), arrowprops=arrow_main)
    ax.annotate('', xy=(4, 6.7), xytext=(4, 5.8), arrowprops=arrow_main)

    # Step 3 → Decision diamond
    ax.annotate('', xy=(8.5, 5), xytext=(6.25, 5), arrowprops=arrow_main)

    # YES arrow: Decision → Step 4
    ax.annotate('', xy=(4, 2.8), xytext=(4, 4.2), arrowprops=arrow_yes)
    ax.text(3.3, 3.5, 'YES', fontsize=11, color='#28A745', fontweight='bold')

    # NO arrow: Decision → back to Step 1 (curved)
    ax.annotate('', xy=(9.5, 6.0), xytext=(9.5, 8.8),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5,
                              connectionstyle='arc3,rad=0.3', mutation_scale=15))
    ax.annotate('', xy=(6.25, 10), xytext=(9.5, 9.2),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5, mutation_scale=15))
    ax.text(10.2, 7.5, 'NO', fontsize=11, color='#E74C3C', fontweight='bold')
    ax.text(10.2, 7.0, 'Revise', fontsize=9, color='#E74C3C')

    # Dotted lines to detail boxes
    ax.plot([6.25, 9.1], [10, 10], '--', color='#888888', alpha=0.5, linewidth=1.5)
    ax.plot([6.25, 9.1], [7.5, 7.5], '--', color='#888888', alpha=0.5, linewidth=1.5)
    ax.plot([6.25, 9.1], [2, 2], '--', color='#888888', alpha=0.5, linewidth=1.5)

    # Title
    ax.text(7, 11.5, 'Box-Jenkins Methodology', ha='center', va='center',
           fontsize=16, fontweight='bold', color='#2E86AB')

    plt.tight_layout()
    save_chart(fig, 'box_jenkins_flowchart')

# =============================================================================
# 16. LJUNG-BOX TEST VISUALIZATION
# =============================================================================
def plot_ljung_box():
    """Visualize Ljung-Box test results"""
    n = 300

    # Good model (residuals are white noise)
    ar = np.array([1, -0.7])
    ma = np.array([1])
    data_good = ArmaProcess(ar, ma).generate_sample(nsample=n)
    model_good = ARIMA(data_good, order=(1, 0, 0))
    fit_good = model_good.fit()
    resid_good = fit_good.resid

    # Bad model (underfit - missing AR component)
    ar2 = np.array([1, -0.7, -0.3])
    ma2 = np.array([1])
    data_bad = ArmaProcess(ar2, ma2).generate_sample(nsample=n)
    model_bad = ARIMA(data_bad, order=(1, 0, 0))  # Deliberately underfit
    fit_bad = model_bad.fit()
    resid_bad = fit_bad.resid

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Good model ACF
    acf_good = acf(resid_good, nlags=20)
    axes[0, 0].bar(range(len(acf_good)), acf_good, color=GREEN, width=0.4)
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 0].axhline(y=1.96/np.sqrt(n), color=RED, linestyle='--')
    axes[0, 0].axhline(y=-1.96/np.sqrt(n), color=RED, linestyle='--')
    axes[0, 0].set_title('Good Fit: Residual ACF', fontweight='bold')
    axes[0, 0].set_xlabel('Lag')

    # Bad model ACF
    acf_bad = acf(resid_bad, nlags=20)
    axes[0, 1].bar(range(len(acf_bad)), acf_bad, color=RED, width=0.4)
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].axhline(y=1.96/np.sqrt(n), color=RED, linestyle='--')
    axes[0, 1].axhline(y=-1.96/np.sqrt(n), color=RED, linestyle='--')
    axes[0, 1].set_title('Bad Fit: Residual ACF (Underfit)', fontweight='bold')
    axes[0, 1].set_xlabel('Lag')

    # Ljung-Box p-values - Good
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lags_test = range(1, 21)
    lb_good = acorr_ljungbox(resid_good, lags=list(lags_test), return_df=True)
    axes[1, 0].bar(lags_test, lb_good['lb_pvalue'], color=GREEN, width=0.6)
    axes[1, 0].axhline(y=0.05, color=RED, linestyle='--', linewidth=2, label='$\\alpha = 0.05$')
    axes[1, 0].set_title('Ljung-Box p-values (Good Fit)', fontweight='bold')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('p-value')
    axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    axes[1, 0].text(0.95, 0.95, 'All p > 0.05\nRESIDUALS OK', transform=axes[1, 0].transAxes,
                   ha='right', va='top', fontsize=11, color=GREEN, fontweight='bold')

    # Ljung-Box p-values - Bad
    lb_bad = acorr_ljungbox(resid_bad, lags=list(lags_test), return_df=True)
    axes[1, 1].bar(lags_test, lb_bad['lb_pvalue'], color=RED, width=0.6)
    axes[1, 1].axhline(y=0.05, color=RED, linestyle='--', linewidth=2, label='$\\alpha = 0.05$')
    axes[1, 1].set_title('Ljung-Box p-values (Bad Fit)', fontweight='bold')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('p-value')
    axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    axes[1, 1].text(0.95, 0.95, 'p < 0.05\nMODEL INADEQUATE', transform=axes[1, 1].transAxes,
                   ha='right', va='top', fontsize=11, color=RED, fontweight='bold')

    plt.tight_layout()
    save_chart(fig, 'ljung_box_test')

# =============================================================================
# 17. AR(2) STATIONARITY TRIANGLE
# =============================================================================
def plot_ar2_stationarity_triangle():
    """Visualize AR(2) stationarity region as triangle"""
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(10, 10))

    # Stationarity triangle vertices
    # phi1 + phi2 < 1, phi2 - phi1 < 1, |phi2| < 1
    vertices = np.array([[-2, 1], [2, 1], [0, -1]])
    triangle = Polygon(vertices, facecolor=GREEN, alpha=0.3, edgecolor=GREEN, linewidth=2)
    ax.add_patch(triangle)

    # Plot boundary lines
    phi1 = np.linspace(-2.5, 2.5, 100)
    ax.plot(phi1, 1 - phi1, '--', color=BLUE, linewidth=2, label='$\\phi_1 + \\phi_2 = 1$')
    ax.plot(phi1, phi1 - 1, '--', color=RED, linewidth=2, label='$\\phi_2 - \\phi_1 = 1$')
    ax.axhline(y=1, color=ORANGE, linestyle='--', linewidth=2, label='$\\phi_2 = 1$')
    ax.axhline(y=-1, color=PURPLE, linestyle='--', linewidth=2, label='$\\phi_2 = -1$')

    # Sample points
    stationary_points = [(0.5, 0.3), (0.3, -0.2), (-0.5, 0.2), (0.8, 0.1)]
    nonstationary_points = [(1.5, 0.5), (0.5, 0.8), (-0.5, -0.8)]

    for p in stationary_points:
        ax.plot(p[0], p[1], 'o', color=GREEN, markersize=12)
    for p in nonstationary_points:
        ax.plot(p[0], p[1], 'x', color=RED, markersize=15, mew=3)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('$\\phi_1$', fontsize=14)
    ax.set_ylabel('$\\phi_2$', fontsize=14)
    ax.set_title('AR(2) Stationarity Region', fontweight='bold', fontsize=16)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)

    # Add labels
    ax.text(0, 0, 'STATIONARY\nREGION', ha='center', va='center', fontsize=14,
           color=GREEN, fontweight='bold', alpha=0.8)
    ax.text(1.8, 0.8, 'Non-stationary', ha='center', va='center', fontsize=11, color=RED)

    ax.set_aspect('equal')
    plt.tight_layout()
    save_chart(fig, 'ar2_stationarity_triangle')

# =============================================================================
# 18. ARMA MODEL STRUCTURE DIAGRAM
# =============================================================================
def plot_arma_structure():
    """Create ARMA model structure diagram"""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Helper function
    def draw_box(x, y, w, h, text, color):
        box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                            boxstyle="round,pad=0.02,rounding_size=0.2",
                            facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=11,
               color='white', fontweight='bold')

    def draw_circle(x, y, r, text, color):
        circle = Circle((x, y), r, facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, text, ha='center', va='center', fontsize=14,
               color='white', fontweight='bold')

    # White noise input
    draw_box(1.5, 4, 2, 1.2, '$\\varepsilon_t$\nWhite Noise', GRAY)

    # MA component
    draw_box(5, 4, 2.5, 1.5, 'MA(q)\n$\\theta(L)\\varepsilon_t$', GREEN)

    # Summation
    draw_circle(8, 4, 0.5, '+', ORANGE)

    # AR component (feedback)
    draw_box(8, 1.5, 2.5, 1.2, 'AR(p)\n$\\phi(L)$', BLUE)

    # Output
    draw_box(11, 4, 2, 1.2, '$X_t$\nOutput', PURPLE)

    # Delay block
    draw_box(13.5, 4, 1.5, 1, '$L$\nLag', GRAY)

    # Arrows
    arrow_style = dict(arrowstyle='->', color=GRAY, lw=2, mutation_scale=15)

    ax.annotate('', xy=(3.6, 4), xytext=(2.5, 4), arrowprops=arrow_style)
    ax.annotate('', xy=(7.5, 4), xytext=(6.3, 4), arrowprops=arrow_style)
    ax.annotate('', xy=(10, 4), xytext=(8.5, 4), arrowprops=arrow_style)
    ax.annotate('', xy=(12.8, 4), xytext=(12, 4), arrowprops=arrow_style)

    # Feedback loop
    ax.annotate('', xy=(14.3, 4), xytext=(14.3, 1.5),
                arrowprops=dict(arrowstyle='-', color=GRAY, lw=2))
    ax.annotate('', xy=(9.3, 1.5), xytext=(14.3, 1.5),
                arrowprops=dict(arrowstyle='-', color=GRAY, lw=2))
    ax.annotate('', xy=(8, 3.5), xytext=(8, 2.1),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=2, mutation_scale=15))
    ax.annotate('', xy=(6.7, 1.5), xytext=(8, 1.5),
                arrowprops=dict(arrowstyle='-', color=GRAY, lw=2))

    # Title and equation
    ax.text(8, 7.3, 'ARMA(p,q) Model Structure', ha='center', va='center',
           fontsize=16, fontweight='bold', color=BLUE)
    ax.text(8, 6.5, '$X_t = \\phi_1 X_{t-1} + ... + \\phi_p X_{t-p} + \\varepsilon_t + \\theta_1 \\varepsilon_{t-1} + ... + \\theta_q \\varepsilon_{t-q}$',
           ha='center', va='center', fontsize=12)

    plt.tight_layout()
    save_chart(fig, 'arma_structure')

# =============================================================================
# 19. WOLD REPRESENTATION
# =============================================================================
def plot_wold_representation():
    """Visualize Wold's decomposition theorem"""
    n = 100

    # AR(1) process
    phi = 0.8
    ar = np.array([1, -phi])
    ma = np.array([1])
    data = ArmaProcess(ar, ma).generate_sample(nsample=n)

    # MA(infinity) coefficients (psi weights)
    n_psi = 20
    psi = [phi**k for k in range(n_psi)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Original AR(1) series
    axes[0].plot(data, color=BLUE, linewidth=1)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('AR(1) Process: $X_t = 0.8X_{t-1} + \\varepsilon_t$', fontweight='bold')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$X_t$')

    # Psi weights (MA infinity representation)
    markerline, stemlines, baseline = axes[1].stem(range(n_psi), psi, basefmt='gray')
    plt.setp(stemlines, color=GREEN)
    plt.setp(markerline, color=GREEN, markersize=8)
    axes[1].axhline(y=0, color='gray', linewidth=0.5)
    axes[1].set_title('$\\psi$-weights: $\\psi_j = \\phi^j$', fontweight='bold')
    axes[1].set_xlabel('Lag $j$')
    axes[1].set_ylabel('$\\psi_j$')

    # Add equation
    axes[1].text(0.95, 0.95, '$X_t = \\sum_{j=0}^{\\infty} \\psi_j \\varepsilon_{t-j}$',
                transform=axes[1].transAxes, ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Cumulative effect
    cumsum_psi = np.cumsum(psi)
    axes[2].plot(range(n_psi), cumsum_psi, 'o-', color=ORANGE, linewidth=2, markersize=8)
    axes[2].axhline(y=1/(1-phi), color=RED, linestyle='--', linewidth=2,
                   label=f'Long-run: $1/(1-\\phi) = {1/(1-phi):.1f}$')
    axes[2].set_title('Cumulative Impulse Response', fontweight='bold')
    axes[2].set_xlabel('Lag $j$')
    axes[2].set_ylabel('$\\sum_{k=0}^{j} \\psi_k$')
    axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

    plt.tight_layout()
    save_chart(fig, 'wold_representation')

# =============================================================================
# 20. CHARACTERISTIC POLYNOMIAL ROOTS
# =============================================================================
def plot_characteristic_roots():
    """Visualize characteristic polynomial roots"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cases = [
        ('AR(1): $\\phi=0.8$', [1, -0.8], GREEN),
        ('AR(2): Complex roots', [1, -1.0, 0.5], BLUE),
        ('AR(2): Real roots', [1, -0.5, -0.3], ORANGE)
    ]

    for idx, (title, coeffs, color) in enumerate(cases):
        ax = axes[idx]

        # Unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), '--', color=GRAY, linewidth=2)
        ax.fill(np.cos(theta), np.sin(theta), alpha=0.1, color=RED)

        # Roots
        roots = np.roots(coeffs)
        for root in roots:
            ax.plot(root.real, root.imag, 'o', color=color, markersize=15)
            ax.plot([0, root.real], [0, root.imag], '-', color=color, alpha=0.5)
            # Label
            ax.annotate(f'  $|z|={abs(root):.2f}$',
                       xy=(root.real, root.imag), fontsize=10, color=color)

        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(title, fontweight='bold')

        # Stationarity label
        is_stationary = all(abs(r) > 1 for r in roots)
        status = 'STATIONARY' if is_stationary else 'NON-STATIONARY'
        status_color = GREEN if is_stationary else RED
        ax.text(0.05, 0.95, status, transform=ax.transAxes,
               fontsize=11, va='top', color=status_color, fontweight='bold')

    plt.tight_layout()
    save_chart(fig, 'characteristic_roots')

# =============================================================================
# 21. ESTIMATION METHODS COMPARISON
# =============================================================================
def plot_estimation_comparison():
    """Compare different estimation methods"""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def draw_method_box(x, y, title, pros, cons, color):
        # Title box
        title_box = FancyBboxPatch((x-2.5, y+1.2), 5, 0.8,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(title_box)
        ax.text(x, y+1.6, title, ha='center', va='center',
               fontsize=12, color='white', fontweight='bold')

        # Content box
        content_box = FancyBboxPatch((x-2.5, y-2), 5, 3.2,
                                    boxstyle="round,pad=0.02",
                                    facecolor='white', edgecolor=color, linewidth=2)
        ax.add_patch(content_box)

        # Pros
        ax.text(x, y+0.9, 'Pros:', ha='center', va='top', fontsize=10,
               color=GREEN, fontweight='bold')
        for i, pro in enumerate(pros):
            ax.text(x, y+0.5-i*0.4, f'+ {pro}', ha='center', va='top', fontsize=9)

        # Cons
        ax.text(x, y-0.5, 'Cons:', ha='center', va='top', fontsize=10,
               color=RED, fontweight='bold')
        for i, con in enumerate(cons):
            ax.text(x, y-0.9-i*0.4, f'- {con}', ha='center', va='top', fontsize=9)

    # Method boxes
    draw_method_box(3, 6, 'Yule-Walker',
                   ['Simple computation', 'Closed-form solution'],
                   ['AR only', 'Less efficient'],
                   BLUE)

    draw_method_box(8, 6, 'Maximum Likelihood',
                   ['Most efficient', 'Works for ARMA'],
                   ['Iterative', 'Local optima risk'],
                   GREEN)

    draw_method_box(13, 6, 'Conditional LS',
                   ['Simple to implement', 'Fast computation'],
                   ['Biased for small n', 'Ignores initial values'],
                   ORANGE)

    # Title
    ax.text(8, 9.5, 'ARMA Parameter Estimation Methods', ha='center',
           fontsize=16, fontweight='bold', color=BLUE)

    # Bottom recommendation
    rec_box = FancyBboxPatch((3.5, 0.5), 9, 1.2,
                             boxstyle="round,pad=0.02",
                             facecolor='#f0f0f0', edgecolor=PURPLE, linewidth=2)
    ax.add_patch(rec_box)
    ax.text(8, 1.1, 'Recommendation: Use MLE for final estimation,\nYule-Walker for initial values',
           ha='center', va='center', fontsize=11, fontweight='bold', color=PURPLE)

    plt.tight_layout()
    save_chart(fig, 'estimation_comparison')

# =============================================================================
# 22. INVERTIBILITY CONDITIONS
# =============================================================================
def plot_invertibility():
    """Visualize MA invertibility conditions"""
    n = 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Invertible MA(1)
    theta_inv = 0.6
    ar = np.array([1])
    ma = np.array([1, theta_inv])
    data_inv = ArmaProcess(ar, ma).generate_sample(nsample=n)

    axes[0, 0].plot(data_inv, color=GREEN, linewidth=1)
    axes[0, 0].set_title(f'Invertible MA(1): $\\theta = {theta_inv}$', fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].text(0.02, 0.98, '$|\\theta| < 1$ ✓', transform=axes[0, 0].transAxes,
                   va='top', fontsize=12, color=GREEN, fontweight='bold')

    # Non-invertible MA(1)
    theta_noninv = 1.5

    axes[0, 1].text(0.5, 0.5, f'MA(1) with $\\theta = {theta_noninv}$\n\nNOT INVERTIBLE\n$|\\theta| > 1$',
                   ha='center', va='center', transform=axes[0, 1].transAxes,
                   fontsize=14, color=RED, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='#ffe0e0'))
    axes[0, 1].set_title(f'Non-Invertible MA(1): $\\theta = {theta_noninv}$', fontweight='bold')
    axes[0, 1].axis('off')

    # Pi weights for invertible
    n_pi = 15
    pi_weights = [(-theta_inv)**k for k in range(n_pi)]
    markerline, stemlines, baseline = axes[1, 0].stem(range(n_pi), pi_weights, basefmt='gray')
    plt.setp(stemlines, color=BLUE)
    plt.setp(markerline, color=BLUE)
    axes[1, 0].axhline(y=0, color='gray', linewidth=0.5)
    axes[1, 0].set_title('$\\pi$-weights: $\\pi_j = (-\\theta)^j$', fontweight='bold')
    axes[1, 0].set_xlabel('Lag $j$')
    axes[1, 0].set_ylabel('$\\pi_j$')
    axes[1, 0].text(0.95, 0.95, '$\\varepsilon_t = \\sum_{j=0}^{\\infty} \\pi_j X_{t-j}$',
                   transform=axes[1, 0].transAxes, ha='right', va='top', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Unit circle for MA
    ax = axes[1, 1]
    theta_circle = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta_circle), np.sin(theta_circle), '--', color=GRAY, linewidth=2)
    ax.fill(np.cos(theta_circle), np.sin(theta_circle), alpha=0.1, color=RED)

    # Plot roots
    root_inv = -1/theta_inv
    root_noninv = -1/theta_noninv

    ax.plot(root_inv, 0, 'o', color=GREEN, markersize=15, label=f'$\\theta={theta_inv}$: $|z|={abs(root_inv):.2f}$')
    ax.plot(root_noninv, 0, 'x', color=RED, markersize=15, mew=3, label=f'$\\theta={theta_noninv}$: $|z|={abs(root_noninv):.2f}$')

    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_title('MA Invertibility: Roots of $\\theta(z)$', fontweight='bold')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.text(0, -0.3, 'Invertible if roots\nOUTSIDE unit circle', ha='center', fontsize=10)

    plt.tight_layout()
    save_chart(fig, 'invertibility')

# =============================================================================
# 23. ARMA SIMULATION STEPS
# =============================================================================
def plot_arma_simulation_steps():
    """Show step-by-step ARMA simulation"""
    np.random.seed(123)
    n = 50
    phi = 0.7
    theta = 0.4

    # Generate components
    eps = np.random.randn(n)
    x = np.zeros(n)

    for t in range(1, n):
        x[t] = phi * x[t-1] + eps[t] + theta * eps[t-1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # White noise
    markerline, stemlines, baseline = axes[0, 0].stem(range(n), eps, basefmt='gray')
    plt.setp(stemlines, color=GRAY)
    plt.setp(markerline, color=GRAY)
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Step 1: Generate White Noise $\\varepsilon_t$', fontweight='bold')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('$\\varepsilon_t$')

    # AR component
    ar_component = np.zeros(n)
    for t in range(1, n):
        ar_component[t] = phi * x[t-1]
    axes[0, 1].plot(ar_component, color=BLUE, linewidth=1.5)
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_title(f'Step 2: AR Component $\\phi X_{{t-1}}$ ($\\phi={phi}$)', fontweight='bold')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('$\\phi X_{t-1}$')

    # MA component
    ma_component = np.zeros(n)
    for t in range(1, n):
        ma_component[t] = theta * eps[t-1]
    axes[1, 0].plot(ma_component, color=GREEN, linewidth=1.5)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_title(f'Step 3: MA Component $\\theta \\varepsilon_{{t-1}}$ ($\\theta={theta}$)', fontweight='bold')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('$\\theta \\varepsilon_{t-1}$')

    # Final ARMA process
    axes[1, 1].plot(x, color=PURPLE, linewidth=1.5)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_title(f'Step 4: ARMA(1,1) = AR + MA + $\\varepsilon_t$', fontweight='bold')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('$X_t$')
    axes[1, 1].text(0.02, 0.98, '$X_t = \\phi X_{t-1} + \\varepsilon_t + \\theta \\varepsilon_{t-1}$',
                   transform=axes[1, 1].transAxes, va='top', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_chart(fig, 'arma_simulation_steps')

# =============================================================================
# 24. PARSIMONY PRINCIPLE
# =============================================================================
def plot_parsimony():
    """Illustrate parsimony principle in model selection"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Number of parameters
    n_params = np.arange(1, 11)

    # Simulated metrics
    np.random.seed(42)
    in_sample_error = 100 * np.exp(-0.5 * n_params) + np.random.randn(10) * 2
    out_sample_error = 100 * np.exp(-0.3 * n_params) + 0.5 * (n_params - 3)**2 + np.random.randn(10) * 3

    ax.plot(n_params, in_sample_error, 'o-', color=BLUE, linewidth=2, markersize=10,
           label='In-sample error (Training)')
    ax.plot(n_params, out_sample_error, 's-', color=RED, linewidth=2, markersize=10,
           label='Out-of-sample error (Test)')

    # Optimal point
    optimal_idx = np.argmin(out_sample_error)
    ax.axvline(x=n_params[optimal_idx], color=GREEN, linestyle='--', linewidth=2,
              label=f'Optimal complexity (p+q = {n_params[optimal_idx]})')
    ax.plot(n_params[optimal_idx], out_sample_error[optimal_idx], '*',
           color=GREEN, markersize=20)

    # Regions
    ax.axvspan(1, n_params[optimal_idx], alpha=0.1, color=ORANGE)
    ax.axvspan(n_params[optimal_idx], 10, alpha=0.1, color=RED)
    ax.text(2, 60, 'UNDERFITTING\n(High bias)', ha='center', fontsize=11,
           color=ORANGE, fontweight='bold')
    ax.text(8, 60, 'OVERFITTING\n(High variance)', ha='center', fontsize=11,
           color=RED, fontweight='bold')

    ax.set_xlabel('Model Complexity (Number of Parameters p + q)', fontsize=12)
    ax.set_ylabel('Prediction Error', fontsize=12)
    ax.set_title('Parsimony Principle: Bias-Variance Trade-off', fontweight='bold', fontsize=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    ax.set_xlim(0.5, 10.5)

    plt.tight_layout()
    save_chart(fig, 'parsimony_principle')

# =============================================================================
# 25. FORECAST ERROR DECOMPOSITION
# =============================================================================
def plot_forecast_error_decomposition():
    """Show forecast error variance decomposition over horizon"""
    phi = 0.8
    sigma2 = 1

    horizons = np.arange(1, 21)

    # Forecast error variance for AR(1)
    var_forecast = sigma2 * (1 - phi**(2*horizons)) / (1 - phi**2)
    unconditional_var = sigma2 / (1 - phi**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Forecast error variance
    axes[0].plot(horizons, var_forecast, 'o-', color=BLUE, linewidth=2, markersize=8)
    axes[0].axhline(y=unconditional_var, color=RED, linestyle='--', linewidth=2,
                   label=f'Unconditional variance = {unconditional_var:.2f}')
    axes[0].fill_between(horizons, 0, var_forecast, alpha=0.2, color=BLUE)
    axes[0].set_xlabel('Forecast Horizon $h$', fontsize=12)
    axes[0].set_ylabel('$Var(e_t(h))$', fontsize=12)
    axes[0].set_title(f'Forecast Error Variance: AR(1) with $\\phi={phi}$', fontweight='bold')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
    axes[0].set_ylim(0, unconditional_var * 1.2)

    # Confidence interval width
    ci_width = 2 * 1.96 * np.sqrt(var_forecast)
    axes[1].fill_between(horizons, -1.96*np.sqrt(var_forecast), 1.96*np.sqrt(var_forecast),
                        alpha=0.3, color=BLUE, label='95% CI')
    axes[1].plot(horizons, 1.96*np.sqrt(var_forecast), color=BLUE, linewidth=2)
    axes[1].plot(horizons, -1.96*np.sqrt(var_forecast), color=BLUE, linewidth=2)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Forecast Horizon $h$', fontsize=12)
    axes[1].set_ylabel('Prediction Interval Bounds', fontsize=12)
    axes[1].set_title('Forecast Uncertainty Grows with Horizon', fontweight='bold')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

    # Add annotation
    axes[1].annotate('Uncertainty\nincreases', xy=(15, 1.96*np.sqrt(var_forecast[14])),
                    xytext=(17, 3), fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color=BLUE))

    plt.tight_layout()
    save_chart(fig, 'forecast_error_decomposition')

# =============================================================================
# 26. TRAIN/TEST/VALIDATION FORECAST USE CASE
# =============================================================================
def plot_forecast_use_case():
    """Show practical train/test/validation forecast with confidence intervals"""
    np.random.seed(42)

    # Generate realistic-looking ARMA data
    n_total = 200
    n_train = 140
    n_val = 30
    n_test = 30

    # Generate ARMA(1,1) process
    ar = np.array([1, -0.7])
    ma = np.array([1, 0.3])
    data = ArmaProcess(ar, ma).generate_sample(nsample=n_total)
    data = data + 50 + np.linspace(0, 10, n_total)  # Add trend and level

    # Split data
    train = data[:n_train]
    val_actual = data[n_train:n_train+n_val]
    test_actual = data[n_train+n_val:]

    # Fit model on training data
    model = ARIMA(train, order=(1, 0, 1))
    fitted = model.fit()

    # Generate forecasts for validation period
    val_forecast = fitted.get_forecast(steps=n_val)
    val_mean = val_forecast.predicted_mean
    val_ci = val_forecast.conf_int()

    # Refit on train+val and forecast test
    train_val = data[:n_train+n_val]
    model2 = ARIMA(train_val, order=(1, 0, 1))
    fitted2 = model2.fit()
    test_forecast = fitted2.get_forecast(steps=n_test)
    test_mean = test_forecast.predicted_mean
    test_ci = test_forecast.conf_int()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot training data
    ax.plot(range(n_train), train, color=BLUE, linewidth=1.5, label='Training Data')

    # Plot validation actual and forecast
    val_idx = range(n_train, n_train+n_val)
    ax.plot(val_idx, val_actual, color=GREEN, linewidth=1.5, label='Validation (Actual)')
    ax.plot(val_idx, val_mean, '--', color=GREEN, linewidth=2, alpha=0.8, label='Validation Forecast')
    if hasattr(val_ci, 'iloc'):
        ax.fill_between(val_idx, val_ci.iloc[:, 0], val_ci.iloc[:, 1],
                        color=GREEN, alpha=0.15)
    else:
        ax.fill_between(val_idx, val_ci[:, 0], val_ci[:, 1],
                        color=GREEN, alpha=0.15)

    # Plot test actual and forecast
    test_idx = range(n_train+n_val, n_total)
    ax.plot(test_idx, test_actual, color=RED, linewidth=1.5, label='Test (Actual)')
    ax.plot(test_idx, test_mean, '--', color=RED, linewidth=2, alpha=0.8, label='Test Forecast')
    if hasattr(test_ci, 'iloc'):
        ax.fill_between(test_idx, test_ci.iloc[:, 0], test_ci.iloc[:, 1],
                        color=RED, alpha=0.15, label='95% CI')
    else:
        ax.fill_between(test_idx, test_ci[:, 0], test_ci[:, 1],
                        color=RED, alpha=0.15, label='95% CI')

    # Vertical lines for splits
    ax.axvline(x=n_train-0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=n_train+n_val-0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7)

    # Add region labels
    ax.text(n_train/2, ax.get_ylim()[1]*0.95, 'TRAINING\n(70%)',
           ha='center', va='top', fontsize=12, color=BLUE, fontweight='bold')
    ax.text(n_train + n_val/2, ax.get_ylim()[1]*0.95, 'VALIDATION\n(15%)',
           ha='center', va='top', fontsize=12, color=GREEN, fontweight='bold')
    ax.text(n_train + n_val + n_test/2, ax.get_ylim()[1]*0.95, 'TEST\n(15%)',
           ha='center', va='top', fontsize=12, color=RED, fontweight='bold')

    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('ARMA Forecasting: Train/Validation/Test Split with 95% Confidence Intervals',
                fontweight='bold', fontsize=14)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)

    plt.tight_layout()
    save_chart(fig, 'forecast_train_val_test')

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    print("Generating ARMA charts for Chapter 2...")
    print("=" * 50)

    # Generate all charts
    plot_ar1_comparison()
    plot_ma1_comparison()
    plot_arma11()
    plot_acf_pacf_patterns()
    plot_unit_circle()
    plot_impulse_response()
    plot_ar1_theoretical_acf()
    plot_aic_bic_comparison()
    plot_arma_forecast()
    plot_residual_diagnostics_arma()
    plot_lag_operator()
    plot_yule_walker()
    plot_ar1_variance()
    plot_model_identification_table()
    plot_box_jenkins_flowchart()
    plot_ljung_box()

    # New charts
    plot_ar2_stationarity_triangle()
    plot_arma_structure()
    plot_wold_representation()
    plot_characteristic_roots()
    plot_estimation_comparison()
    plot_invertibility()
    plot_arma_simulation_steps()
    plot_parsimony()
    plot_forecast_error_decomposition()
    plot_forecast_use_case()

    print("=" * 50)
    print("All charts generated successfully!")
