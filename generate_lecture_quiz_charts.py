#!/usr/bin/env python3
"""
Generate charts for lecture quiz answers
Time Series Analysis Course
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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
# CHAPTER 2: ARMA Models Quiz Charts
# =============================================================================
print("Chapter 2: ARMA Quiz Charts")

# Quiz: AR Stationarity - Compare stationary vs non-stationary
def ch2_ar_stationarity():
    np.random.seed(42)
    n = 100
    eps = np.random.normal(0, 1, n)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Stationary: phi = -0.8
    y_stat = np.zeros(n)
    for t in range(1, n):
        y_stat[t] = -0.8 * y_stat[t-1] + eps[t]
    axes[0, 0].plot(y_stat, 'b-', linewidth=1)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title(r'$\phi = -0.8$ (Stationary, $|\phi| < 1$)', fontsize=10, color='green')
    axes[0, 0].set_ylabel('$Y_t$')

    # Non-stationary: phi = 1.0 (random walk)
    y_rw = np.zeros(n)
    for t in range(1, n):
        y_rw[t] = 1.0 * y_rw[t-1] + eps[t]
    axes[0, 1].plot(y_rw, 'r-', linewidth=1)
    axes[0, 1].set_title(r'$\phi = 1.0$ (Unit Root, Non-stationary)', fontsize=10, color='red')

    # Non-stationary: phi = 1.2 (explosive)
    y_exp = np.zeros(n)
    for t in range(1, min(50, n)):
        y_exp[t] = 1.2 * y_exp[t-1] + eps[t]
    axes[1, 0].plot(y_exp[:50], 'r-', linewidth=1)
    axes[1, 0].set_title(r'$\phi = 1.2$ (Explosive, $|\phi| > 1$)', fontsize=10, color='red')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('$Y_t$')

    # Non-stationary: phi = -1.5 (explosive oscillating)
    y_osc = np.zeros(n)
    for t in range(1, min(30, n)):
        y_osc[t] = -1.5 * y_osc[t-1] + eps[t]
    axes[1, 1].plot(y_osc[:30], 'r-', linewidth=1)
    axes[1, 1].set_title(r'$\phi = -1.5$ (Explosive Oscillation)', fontsize=10, color='red')
    axes[1, 1].set_xlabel('Time')

    plt.tight_layout()
    save_fig('ch2_quiz_ar_stationarity')

ch2_ar_stationarity()

# Quiz: ACF/PACF Pattern Recognition
def ch2_acf_pacf_patterns():
    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    lags = np.arange(0, 11)

    # AR(1) with phi=0.7
    phi = 0.7
    acf_ar1 = phi ** lags
    pacf_ar1 = np.zeros(11)
    pacf_ar1[0] = 1
    pacf_ar1[1] = phi

    axes[0, 0].bar(lags, acf_ar1, color='steelblue', alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 0].set_title('AR(1): ACF Decays', fontsize=9)
    axes[0, 0].set_ylim(-0.3, 1.1)

    axes[1, 0].bar(lags, pacf_ar1, color='coral', alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 0].set_title('AR(1): PACF Cuts Off', fontsize=9)
    axes[1, 0].set_ylim(-0.3, 1.1)
    axes[1, 0].set_xlabel('Lag')

    # MA(1) with theta=0.7
    acf_ma1 = np.zeros(11)
    acf_ma1[0] = 1
    acf_ma1[1] = 0.7 / (1 + 0.7**2)

    pacf_ma1 = np.zeros(11)
    pacf_ma1[0] = 1
    for k in range(1, 11):
        pacf_ma1[k] = (-0.7)**k * (1 - 0.7**2) / (1 - 0.7**(2*(k+1)))

    axes[0, 1].bar(lags, acf_ma1, color='steelblue', alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 1].set_title('MA(1): ACF Cuts Off', fontsize=9)
    axes[0, 1].set_ylim(-0.3, 1.1)

    axes[1, 1].bar(lags, pacf_ma1, color='coral', alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 1].set_title('MA(1): PACF Decays', fontsize=9)
    axes[1, 1].set_ylim(-0.5, 1.1)
    axes[1, 1].set_xlabel('Lag')

    # ARMA(1,1)
    acf_arma = np.zeros(11)
    acf_arma[0] = 1
    for k in range(1, 11):
        acf_arma[k] = 0.6 * 0.7**(k-1)

    axes[0, 2].bar(lags, acf_arma, color='steelblue', alpha=0.7)
    axes[0, 2].axhline(y=0, color='black', linewidth=0.5)
    axes[0, 2].set_title('ARMA(1,1): Both Decay', fontsize=9)
    axes[0, 2].set_ylim(-0.3, 1.1)

    pacf_arma = np.zeros(11)
    pacf_arma[0] = 1
    for k in range(1, 11):
        pacf_arma[k] = 0.5 * (-0.3)**(k-1)

    axes[1, 2].bar(lags, pacf_arma, color='coral', alpha=0.7)
    axes[1, 2].axhline(y=0, color='black', linewidth=0.5)
    axes[1, 2].set_title('ARMA(1,1): Both Decay', fontsize=9)
    axes[1, 2].set_ylim(-0.5, 1.1)
    axes[1, 2].set_xlabel('Lag')

    plt.tight_layout()
    save_fig('ch2_quiz_acf_pacf_patterns')

ch2_acf_pacf_patterns()

# Quiz: Information Criteria
def ch2_information_criteria():
    fig, ax = plt.subplots(figsize=(8, 4))

    orders = np.arange(0, 8)
    n = 200

    # Simulated log-likelihood and criteria for different model orders
    # True model is AR(2)
    loglik = -100 + 15 * (1 - np.exp(-orders/2))
    k = orders + 1  # number of parameters

    aic = -2 * loglik + 2 * k
    bic = -2 * loglik + np.log(n) * k

    ax.plot(orders, aic, 'b-o', label='AIC', linewidth=2, markersize=8)
    ax.plot(orders, bic, 'r-s', label='BIC', linewidth=2, markersize=8)

    # Mark minimum
    aic_min = np.argmin(aic)
    bic_min = np.argmin(bic)
    ax.axvline(x=aic_min, color='b', linestyle='--', alpha=0.5)
    ax.axvline(x=bic_min, color='r', linestyle='--', alpha=0.5)

    ax.scatter([aic_min], [aic[aic_min]], s=150, c='blue', zorder=5, edgecolors='black')
    ax.scatter([bic_min], [bic[bic_min]], s=150, c='red', zorder=5, edgecolors='black')

    ax.set_xlabel('Model Order (p)')
    ax.set_ylabel('Information Criterion')
    ax.set_title('AIC vs BIC: Lower is Better')
    ax.legend()
    ax.set_xticks(orders)

    plt.tight_layout()
    save_fig('ch2_quiz_information_criteria')

ch2_information_criteria()

# Quiz: Ljung-Box Test
def ch2_ljung_box():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    lags = np.arange(1, 21)

    # Good fit: residuals are white noise
    acf_good = np.random.normal(0, 0.1, 20)
    ci = 1.96 / np.sqrt(100)

    axes[0].bar(lags, acf_good, color='green', alpha=0.7)
    axes[0].axhline(y=ci, color='red', linestyle='--', label='95% CI')
    axes[0].axhline(y=-ci, color='red', linestyle='--')
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_title('Good Model: Residuals ≈ White Noise', fontsize=10, color='green')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF of Residuals')
    axes[0].legend()
    axes[0].set_ylim(-0.5, 0.5)

    # Bad fit: significant autocorrelation
    acf_bad = 0.4 * 0.8**lags + np.random.normal(0, 0.05, 20)

    axes[1].bar(lags, acf_bad, color='red', alpha=0.7)
    axes[1].axhline(y=ci, color='blue', linestyle='--', label='95% CI')
    axes[1].axhline(y=-ci, color='blue', linestyle='--')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title('Poor Model: Significant Autocorrelation', fontsize=10, color='red')
    axes[1].set_xlabel('Lag')
    axes[1].legend()
    axes[1].set_ylim(-0.5, 0.5)

    plt.tight_layout()
    save_fig('ch2_quiz_ljung_box')

ch2_ljung_box()

# Quiz: Forecast Properties
def ch2_forecast_properties():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(9, 5))

    # Simulated AR(1) series
    n = 100
    h_max = 30
    phi = 0.7
    sigma = 1

    y = np.zeros(n)
    eps = np.random.normal(0, sigma, n)
    for t in range(1, n):
        y[t] = phi * y[t-1] + eps[t]

    # Forecasts
    h = np.arange(1, h_max + 1)
    y_last = y[-1]
    forecasts = y_last * phi**h

    # Forecast variance (converges to unconditional variance)
    var_h = sigma**2 * (1 - phi**(2*h)) / (1 - phi**2)
    unconditional_var = sigma**2 / (1 - phi**2)

    ci_upper = forecasts + 1.96 * np.sqrt(var_h)
    ci_lower = forecasts - 1.96 * np.sqrt(var_h)

    # Plot
    time_hist = np.arange(n)
    time_fore = np.arange(n, n + h_max)

    ax.plot(time_hist[-30:], y[-30:], 'b-', linewidth=1.5, label='Observed')
    ax.plot(time_fore, forecasts, 'r-', linewidth=2, label='Forecast')
    ax.fill_between(time_fore, ci_lower, ci_upper, color='red', alpha=0.2, label='95% CI')

    # Unconditional mean
    ax.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Unconditional Mean')

    # Unconditional variance bounds
    ax.axhline(y=1.96 * np.sqrt(unconditional_var), color='purple', linestyle=':', alpha=0.7)
    ax.axhline(y=-1.96 * np.sqrt(unconditional_var), color='purple', linestyle=':', alpha=0.7,
               label='Unconditional CI Limit')

    ax.axvline(x=n, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('$Y_t$')
    ax.set_title('AR(1) Forecasts Converge to Unconditional Mean')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    save_fig('ch2_quiz_forecast_properties')

ch2_forecast_properties()

# =============================================================================
# CHAPTER 3: ARIMA Models Quiz Charts
# =============================================================================
print("\nChapter 3: ARIMA Quiz Charts")

# Quiz 1: Random walk variance grows linearly
def ch3_quiz1_rw_variance():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n = 100
    n_paths = 50

    # Multiple random walk paths
    for i in range(n_paths):
        eps = np.random.normal(0, 1, n)
        y = np.cumsum(eps)
        axes[0].plot(y, alpha=0.3, linewidth=0.8)

    axes[0].set_xlabel('Time (t)')
    axes[0].set_ylabel('$Y_t$')
    axes[0].set_title('Random Walk Paths: Variance Increases')

    # Variance vs time
    t = np.arange(1, n + 1)
    theoretical_var = t * 1  # sigma^2 = 1

    axes[1].plot(t, theoretical_var, 'r-', linewidth=2, label=r'$\mathrm{Var}(Y_t) = t \cdot \sigma^2$')
    axes[1].fill_between(t, 0, theoretical_var, alpha=0.3, color='red')
    axes[1].set_xlabel('Time (t)')
    axes[1].set_ylabel('Variance')
    axes[1].set_title('Variance Grows Linearly with Time')
    axes[1].legend()

    plt.tight_layout()
    save_fig('ch3_quiz1_rw_variance')

ch3_quiz1_rw_variance()

# Quiz 2: I(2) differencing
def ch3_quiz2_differencing():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    n = 150
    eps = np.random.normal(0, 1, n)

    # Create I(2) series: double integrated
    z = np.cumsum(eps)  # I(1)
    y = np.cumsum(z)    # I(2)

    # First difference
    dy = np.diff(y)  # I(1)

    # Second difference
    d2y = np.diff(dy)  # I(0)

    axes[0].plot(y, 'b-', linewidth=1)
    axes[0].set_title('Original I(2) Series')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')

    axes[1].plot(dy, 'orange', linewidth=1)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title(r'First Difference $\Delta Y_t$ (Still I(1))')
    axes[1].set_xlabel('Time')

    axes[2].plot(d2y, 'green', linewidth=1)
    axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[2].set_title(r'Second Difference $\Delta^2 Y_t$ (Stationary I(0))')
    axes[2].set_xlabel('Time')

    plt.tight_layout()
    save_fig('ch3_quiz2_differencing')

ch3_quiz2_differencing()

# Quiz 3: ADF test interpretation
def ch3_quiz3_adf_test():
    fig, ax = plt.subplots(figsize=(8, 4))

    # Critical values
    cv_1 = -3.43
    cv_5 = -2.86
    cv_10 = -2.57
    test_stat = -2.1

    # Plot regions
    x = np.linspace(-5, 0, 1000)

    # Color regions
    ax.axvspan(-5, cv_1, alpha=0.3, color='green', label='Reject at 1%')
    ax.axvspan(cv_1, cv_5, alpha=0.3, color='lightgreen', label='Reject at 5%')
    ax.axvspan(cv_5, cv_10, alpha=0.3, color='yellow', label='Reject at 10%')
    ax.axvspan(cv_10, 0, alpha=0.3, color='red', label='Cannot Reject')

    # Critical values lines
    ax.axvline(x=cv_1, color='darkgreen', linestyle='--', linewidth=2)
    ax.axvline(x=cv_5, color='green', linestyle='--', linewidth=2)
    ax.axvline(x=cv_10, color='olive', linestyle='--', linewidth=2)

    # Test statistic
    ax.axvline(x=test_stat, color='blue', linestyle='-', linewidth=3, label=f'Test Stat = {test_stat}')
    ax.scatter([test_stat], [0.5], s=200, c='blue', zorder=5, marker='v')

    # Labels
    ax.text(cv_1, 0.9, '1%\n-3.43', ha='center', fontsize=9)
    ax.text(cv_5, 0.9, '5%\n-2.86', ha='center', fontsize=9)
    ax.text(cv_10, 0.9, '10%\n-2.57', ha='center', fontsize=9)
    ax.text(test_stat, 0.6, 'Test\nStat', ha='center', fontsize=9, color='blue')

    ax.set_xlim(-5, 0)
    ax.set_ylim(0, 1)
    ax.set_xlabel('ADF Test Statistic')
    ax.set_title('ADF Test: Cannot Reject Unit Root (Non-stationary)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_yticks([])

    plt.tight_layout()
    save_fig('ch3_quiz3_adf_test')

ch3_quiz3_adf_test()

# Quiz 4: ACF of AR(1) decays exponentially
def ch3_quiz4_acf_decay():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    phi = 0.7
    lags = np.arange(0, 16)
    acf = phi ** lags

    colors = ['green' if a > 0.1 else 'gray' for a in acf]
    axes[0].bar(lags, acf, color=colors, alpha=0.7, edgecolor='black')
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Lag k')
    axes[0].set_ylabel(r'$\rho_k = \phi^k$')
    axes[0].set_title(r'ACF of AR(1) with $\phi = 0.7$: Exponential Decay')

    # Decay rate visualization
    axes[1].semilogy(lags[1:], acf[1:], 'bo-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Lag k')
    axes[1].set_ylabel(r'$\rho_k$ (log scale)')
    axes[1].set_title('Log Scale: Linear = Exponential Decay')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('ch3_quiz4_acf_decay')

ch3_quiz4_acf_decay()

# Quiz 5: Forecast CI widening for I(1)
def ch3_quiz5_forecast_ci():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    n = 80
    h_max = 40
    sigma = 1

    # Random walk
    eps = np.random.normal(0, sigma, n)
    y_rw = np.cumsum(eps)

    # Forecast
    h = np.arange(1, h_max + 1)
    y_last = y_rw[-1]
    forecast_rw = np.full(h_max, y_last)
    ci_rw = 1.96 * sigma * np.sqrt(h)

    time_hist = np.arange(n)
    time_fore = np.arange(n, n + h_max)

    axes[0].plot(time_hist, y_rw, 'b-', linewidth=1.5)
    axes[0].plot(time_fore, forecast_rw, 'r-', linewidth=2)
    axes[0].fill_between(time_fore, forecast_rw - ci_rw, forecast_rw + ci_rw,
                         color='red', alpha=0.2)
    axes[0].axvline(x=n, color='gray', linestyle='-', alpha=0.5)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')
    axes[0].set_title('I(1): CI Widens Without Bound')

    # CI width comparison
    axes[1].plot(h, 2 * ci_rw, 'r-', linewidth=2, label=r'I(1): $\propto \sqrt{h}$ (unbounded)')

    # Stationary comparison
    phi = 0.7
    var_stat = sigma**2 * (1 - phi**(2*h)) / (1 - phi**2)
    ci_stat = 1.96 * np.sqrt(var_stat)
    unconditional = 1.96 * np.sqrt(sigma**2 / (1 - phi**2))

    axes[1].plot(h, 2 * ci_stat, 'g-', linewidth=2, label=r'I(0): converges to limit')
    axes[1].axhline(y=2 * unconditional, color='green', linestyle='--', alpha=0.7)

    axes[1].set_xlabel('Forecast Horizon h')
    axes[1].set_ylabel('95% CI Width')
    axes[1].set_title('CI Width: I(1) vs I(0)')
    axes[1].legend()

    plt.tight_layout()
    save_fig('ch3_quiz5_forecast_ci')

ch3_quiz5_forecast_ci()

# =============================================================================
# CHAPTER 4: SARIMA Models Quiz Charts
# =============================================================================
print("\nChapter 4: SARIMA Quiz Charts")

# Quiz 1: Seasonal periods
def ch4_quiz1_seasonal_periods():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    np.random.seed(42)

    # Quarterly data (s=4)
    t = np.arange(20)
    seasonal_q = 3 * np.sin(2 * np.pi * t / 4)
    y_q = seasonal_q + np.random.normal(0, 0.5, 20)
    axes[0, 0].plot(t, y_q, 'bo-', markersize=6)
    axes[0, 0].set_title('Quarterly Data: s = 4', fontsize=10)
    axes[0, 0].set_xlabel('Quarter')

    # Monthly data (s=12)
    t = np.arange(36)
    seasonal_m = 3 * np.sin(2 * np.pi * t / 12)
    y_m = seasonal_m + np.random.normal(0, 0.5, 36)
    axes[0, 1].plot(t, y_m, 'go-', markersize=4)
    axes[0, 1].set_title('Monthly Data: s = 12', fontsize=10)
    axes[0, 1].set_xlabel('Month')

    # Weekly data (s=52)
    t = np.arange(104)
    seasonal_w = 3 * np.sin(2 * np.pi * t / 52)
    y_w = seasonal_w + np.random.normal(0, 0.5, 104)
    axes[1, 0].plot(t, y_w, 'r-', linewidth=1)
    axes[1, 0].set_title('Weekly Data: s = 52', fontsize=10)
    axes[1, 0].set_xlabel('Week')

    # Daily data (s=7)
    t = np.arange(28)
    seasonal_d = 2 * np.sin(2 * np.pi * t / 7)
    y_d = seasonal_d + np.random.normal(0, 0.3, 28)
    axes[1, 1].plot(t, y_d, 'mo-', markersize=5)
    axes[1, 1].set_title('Daily Data (Weekly Pattern): s = 7', fontsize=10)
    axes[1, 1].set_xlabel('Day')

    plt.tight_layout()
    save_fig('ch4_quiz1_seasonal_periods')

ch4_quiz1_seasonal_periods()

# Quiz 2: Seasonal differencing
def ch4_quiz2_seasonal_diff():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    n = 72  # 6 years of monthly data
    t = np.arange(n)

    # Original series with trend and seasonality
    trend = 0.05 * t
    seasonal = 3 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 0.5, n)
    y = trend + seasonal + noise

    # Seasonal difference
    dy_12 = y[12:] - y[:-12]

    axes[0].plot(t, y, 'b-', linewidth=1)
    axes[0].set_title('Original: Trend + Seasonality')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('$Y_t$')

    axes[1].plot(t[12:], dy_12, 'g-', linewidth=1)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title(r'$(1-L^{12})Y_t = Y_t - Y_{t-12}$')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel(r'$\Delta_{12} Y_t$')

    # Show year-over-year comparison
    axes[2].plot(y[:12], 'b-o', label='Year 1', markersize=4)
    axes[2].plot(y[12:24], 'r-s', label='Year 2', markersize=4)
    axes[2].plot(y[24:36], 'g-^', label='Year 3', markersize=4)
    axes[2].set_title('Same Month, Different Years')
    axes[2].set_xlabel('Month of Year')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    save_fig('ch4_quiz2_seasonal_diff')

ch4_quiz2_seasonal_diff()

# Quiz 5: ACF with seasonal spikes
def ch4_quiz5_seasonal_acf():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    lags = np.arange(0, 49)

    # ACF with seasonal pattern (before differencing)
    acf_before = np.zeros(49)
    acf_before[0] = 1
    for k in range(1, 49):
        if k % 12 == 0:
            acf_before[k] = 0.9 ** (k // 12)
        else:
            acf_before[k] = 0.1 * np.exp(-k/10) * np.cos(2*np.pi*k/12)

    colors_before = ['red' if k % 12 == 0 and k > 0 else 'steelblue' for k in lags]
    axes[0].bar(lags, acf_before, color=colors_before, alpha=0.7)
    axes[0].axhline(y=1.96/np.sqrt(100), color='black', linestyle='--', alpha=0.5)
    axes[0].axhline(y=-1.96/np.sqrt(100), color='black', linestyle='--', alpha=0.5)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].set_title('ACF with Seasonal Spikes at 12, 24, 36...')

    # After seasonal differencing
    acf_after = np.zeros(49)
    acf_after[0] = 1
    acf_after[1] = 0.3  # MA(1) component
    acf_after[12] = 0.25  # SMA(1) component

    colors_after = ['green' if acf_after[k] > 0.15 else 'steelblue' for k in lags]
    axes[1].bar(lags, acf_after, color=colors_after, alpha=0.7)
    axes[1].axhline(y=1.96/np.sqrt(100), color='black', linestyle='--', alpha=0.5)
    axes[1].axhline(y=-1.96/np.sqrt(100), color='black', linestyle='--', alpha=0.5)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Lag')
    axes[1].set_title('After Differencing: Spikes at 1 and 12')

    plt.tight_layout()
    save_fig('ch4_quiz5_seasonal_acf')

ch4_quiz5_seasonal_acf()

# =============================================================================
# CHAPTER 5: VAR Models Quiz Charts
# =============================================================================
print("\nChapter 5: VAR Quiz Charts")

# Quiz 1: VAR stability - eigenvalues in unit circle
def ch5_quiz1_var_stability():
    fig, ax = plt.subplots(figsize=(6, 6))

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Unit Circle')

    # Eigenvalues from the quiz
    eigenvalues = [0.879, 0.421]

    ax.scatter(eigenvalues, [0, 0], s=200, c='green', zorder=5,
               label=f'Eigenvalues: {eigenvalues[0]:.3f}, {eigenvalues[1]:.3f}', marker='o')

    # Annotations
    ax.annotate(f'$\\lambda_1 = {eigenvalues[0]}$', (eigenvalues[0], 0.1), fontsize=11)
    ax.annotate(f'$\\lambda_2 = {eigenvalues[1]}$', (eigenvalues[1], 0.1), fontsize=11)

    # Shade stable region
    circle = plt.Circle((0, 0), 1, color='lightgreen', alpha=0.3, label='Stable Region')
    ax.add_patch(circle)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Real Part')
    ax.set_ylabel('Imaginary Part')
    ax.set_title('VAR(1) Stability: Both Eigenvalues Inside Unit Circle')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('ch5_quiz1_var_stability')

ch5_quiz1_var_stability()

# Quiz 2: Granger causality concept
def ch5_quiz2_granger_causality():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    np.random.seed(42)

    n = 100
    x = np.zeros(n)
    y = np.zeros(n)
    eps_x = np.random.normal(0, 1, n)
    eps_y = np.random.normal(0, 1, n)

    # X Granger-causes Y: Y depends on lagged X
    for t in range(1, n):
        x[t] = 0.5 * x[t-1] + eps_x[t]
        y[t] = 0.3 * y[t-1] + 0.4 * x[t-1] + eps_y[t]  # X_{t-1} helps predict Y_t

    axes[0].plot(x, 'b-', alpha=0.7, label='X')
    axes[0].plot(y, 'r-', alpha=0.7, label='Y')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].set_title('X Granger-causes Y')
    axes[0].legend()

    # Diagram
    axes[1].axis('off')
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 6)

    # Boxes
    axes[1].add_patch(plt.Rectangle((1, 4), 2, 1.5, fill=True, facecolor='lightblue', edgecolor='blue', linewidth=2))
    axes[1].add_patch(plt.Rectangle((7, 4), 2, 1.5, fill=True, facecolor='lightcoral', edgecolor='red', linewidth=2))

    axes[1].text(2, 4.75, '$X_{t-1}$', ha='center', va='center', fontsize=14)
    axes[1].text(8, 4.75, '$Y_t$', ha='center', va='center', fontsize=14)

    # Arrow
    axes[1].annotate('', xy=(6.9, 4.75), xytext=(3.1, 4.75),
                     arrowprops=dict(arrowstyle='->', color='green', lw=3))
    axes[1].text(5, 5.2, 'Predictive\nInformation', ha='center', fontsize=10, color='green')

    axes[1].text(5, 2, 'Past X helps predict Y\n(beyond Y\'s own past)',
                 ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    save_fig('ch5_quiz2_granger_causality')

ch5_quiz2_granger_causality()

# Quiz 3: Cholesky ordering
def ch5_quiz3_cholesky_ordering():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Ordering 1: GDP first, then Interest Rate
    axes[0].axis('off')
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 8)

    axes[0].text(5, 7.5, 'Ordering: (GDP, Interest Rate)', ha='center', fontsize=11, fontweight='bold')

    # GDP box
    axes[0].add_patch(plt.Rectangle((1, 4), 3, 2, fill=True, facecolor='lightblue', edgecolor='blue', linewidth=2))
    axes[0].text(2.5, 5, 'GDP', ha='center', va='center', fontsize=12)

    # Interest Rate box
    axes[0].add_patch(plt.Rectangle((6, 4), 3, 2, fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2))
    axes[0].text(7.5, 5, 'Interest\nRate', ha='center', va='center', fontsize=11)

    # Arrows showing contemporaneous effects
    axes[0].annotate('', xy=(5.9, 5), xytext=(4.1, 5),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

    axes[0].text(5, 2.5, 'GDP shock → IR responds at t=0\nIR shock → GDP responds at t=1',
                 ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    # Ordering 2: Interest Rate first, then GDP
    axes[1].axis('off')
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 8)

    axes[1].text(5, 7.5, 'Ordering: (Interest Rate, GDP)', ha='center', fontsize=11, fontweight='bold')

    # Interest Rate box
    axes[1].add_patch(plt.Rectangle((1, 4), 3, 2, fill=True, facecolor='lightgreen', edgecolor='green', linewidth=2))
    axes[1].text(2.5, 5, 'Interest\nRate', ha='center', va='center', fontsize=11)

    # GDP box
    axes[1].add_patch(plt.Rectangle((6, 4), 3, 2, fill=True, facecolor='lightblue', edgecolor='blue', linewidth=2))
    axes[1].text(7.5, 5, 'GDP', ha='center', va='center', fontsize=12)

    # Arrows
    axes[1].annotate('', xy=(5.9, 5), xytext=(4.1, 5),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))

    axes[1].text(5, 2.5, 'IR shock → GDP responds at t=0\nGDP shock → IR responds at t=1',
                 ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

    plt.tight_layout()
    save_fig('ch5_quiz3_cholesky_ordering')

ch5_quiz3_cholesky_ordering()

# Quiz 5: FEVD interpretation
def ch5_quiz5_fevd():
    fig, ax = plt.subplots(figsize=(8, 5))

    horizons = [1, 4, 8, 12, 20]

    # FEVD for variable 1 (e.g., GDP)
    fevd_own = [0.95, 0.75, 0.60, 0.55, 0.50]  # Own shocks
    fevd_other = [0.05, 0.25, 0.40, 0.45, 0.50]  # Shocks from variable 2

    x = np.arange(len(horizons))
    width = 0.6

    bars1 = ax.bar(x, fevd_own, width, label='Own Shocks (Variable 1)', color='steelblue')
    bars2 = ax.bar(x, fevd_other, width, bottom=fevd_own, label='Shocks from Variable 2', color='coral')

    # Add 35% marker for h=12
    ax.axhline(y=0.65, color='red', linestyle='--', alpha=0.5, xmin=0.55, xmax=0.85)
    ax.annotate('FEVD₁₂(12) = 35%', xy=(3, 0.78), fontsize=10, color='red')

    ax.set_xlabel('Forecast Horizon (h)')
    ax.set_ylabel('Fraction of Forecast Error Variance')
    ax.set_title('FEVD: Decomposition of Forecast Uncertainty')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    save_fig('ch5_quiz5_fevd')

ch5_quiz5_fevd()

print("\nAll lecture quiz charts created successfully!")
print(f"Total charts created in {output_dir}/")
