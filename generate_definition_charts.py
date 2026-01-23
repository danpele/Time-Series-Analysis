#!/usr/bin/env python3
"""
Generate illustration charts for definitions in Time Series Analysis Course
Charts are added after each definition to help visualize the concept
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

print("Generating Definition Illustration Charts...")

# =============================================================================
# CHAPTER 1: Introduction Definitions
# =============================================================================
print("\nChapter 1 Definition Charts:")

# 1. Time Series Definition
def ch1_def_timeseries():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(8, 3))

    n = 50
    t = np.arange(n)
    y = 10 + 0.1*t + 2*np.sin(2*np.pi*t/12) + np.random.normal(0, 0.5, n)

    ax.plot(t, y, 'b-o', markersize=4, linewidth=1.5)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$X_t$')
    ax.set_title('Time Series: Sequence of observations indexed by time')

    # Annotate a few points
    for i in [10, 20, 30]:
        ax.annotate(f'$X_{{{i}}}$', (t[i], y[i]), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    save_fig('ch1_def_timeseries')

ch1_def_timeseries()

# 2. Stochastic Process
def ch1_def_stochastic():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(8, 3.5))

    n = 100
    t = np.arange(n)

    # Show multiple realizations
    for i, seed in enumerate([42, 123, 456, 789]):
        np.random.seed(seed)
        y = np.zeros(n)
        for j in range(1, n):
            y[j] = 0.7 * y[j-1] + np.random.normal(0, 1)
        ax.plot(t, y, alpha=0.6, linewidth=1, label=f'Realization {i+1}')

    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$X_t(\\omega)$')
    ax.set_title('Stochastic Process: Multiple realizations from same underlying process')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

    plt.tight_layout()
    save_fig('ch1_def_stochastic')

ch1_def_stochastic()

# 3. Strict Stationarity
def ch1_def_strict_stationarity():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 150

    # Stationary process
    y_stat = np.random.normal(0, 1, n)
    for t in range(1, n):
        y_stat[t] = 0.5 * y_stat[t-1] + np.random.normal(0, 0.87)

    axes[0].plot(y_stat, 'b-', linewidth=1)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].fill_between(range(n), -2, 2, alpha=0.1, color='blue')
    axes[0].set_title('Strictly Stationary: Joint distribution unchanged by time shift')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$X_t$')

    # Highlight two windows
    axes[0].axvspan(20, 50, alpha=0.2, color='green')
    axes[0].axvspan(80, 110, alpha=0.2, color='orange')
    axes[0].text(35, 2.5, '$t_1, t_2, ..., t_k$', ha='center', fontsize=9)
    axes[0].text(95, 2.5, '$t_1+h, t_2+h, ..., t_k+h$', ha='center', fontsize=9)

    # Non-stationary
    np.random.seed(42)
    y_nonstat = np.cumsum(np.random.normal(0, 1, n))

    axes[1].plot(y_nonstat, 'r-', linewidth=1)
    axes[1].axhline(y=0, color='blue', linestyle='--', alpha=0.5)
    axes[1].set_title('Non-Stationary: Distribution changes with time')
    axes[1].set_xlabel('Time')

    plt.tight_layout()
    save_fig('ch1_def_strict_stationarity')

ch1_def_strict_stationarity()

# 4. Weak Stationarity
def ch1_def_weak_stationarity():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 150

    # Weakly stationary
    eps = np.random.normal(0, 1, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t-1] + eps[t]

    axes[0].plot(y, 'b-', linewidth=1)
    axes[0].axhline(y=np.mean(y), color='red', linestyle='-', linewidth=2, label=f'Mean = {np.mean(y):.2f}')
    axes[0].axhline(y=np.mean(y) + np.std(y), color='orange', linestyle='--', alpha=0.7)
    axes[0].axhline(y=np.mean(y) - np.std(y), color='orange', linestyle='--', alpha=0.7)
    axes[0].set_title('Weakly Stationary: Constant mean and autocovariance')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$X_t$')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)

    # Show autocovariance depends only on lag
    lags = np.arange(0, 20)
    acf = 0.7 ** lags

    axes[1].bar(lags, acf, color='steelblue', alpha=0.7)
    axes[1].set_title('$\\gamma(h) = \\gamma(|s-t|)$: Depends only on lag')
    axes[1].set_xlabel('Lag $h$')
    axes[1].set_ylabel('Autocovariance')

    plt.tight_layout()
    save_fig('ch1_def_weak_stationarity')

ch1_def_weak_stationarity()

# 5. White Noise
def ch1_def_white_noise():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 100
    wn = np.random.normal(0, 1, n)

    axes[0].plot(wn, 'b-', linewidth=1, alpha=0.8)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0].axhline(y=2, color='orange', linestyle=':', alpha=0.7)
    axes[0].axhline(y=-2, color='orange', linestyle=':', alpha=0.7)
    axes[0].set_title('White Noise: $\\varepsilon_t \\sim WN(0, \\sigma^2)$')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$\\varepsilon_t$')
    axes[0].text(50, 2.5, 'Mean = 0, Var = $\\sigma^2$', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ACF of white noise
    lags = np.arange(0, 16)
    acf = np.zeros(16)
    acf[0] = 1
    ci = 1.96 / np.sqrt(n)

    axes[1].bar(lags, acf, color='steelblue', alpha=0.7)
    axes[1].axhline(y=ci, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=-ci, color='red', linestyle='--', alpha=0.7)
    axes[1].set_title('ACF: No autocorrelation ($\\gamma(h) = 0$ for $h \\neq 0$)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('ACF')
    axes[1].set_ylim(-0.3, 1.1)

    plt.tight_layout()
    save_fig('ch1_def_white_noise')

ch1_def_white_noise()

# 6. Lag Operator
def ch1_def_lag_operator():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(9, 3.5))

    n = 20
    t = np.arange(n)
    y = 10 + 0.5*t + np.random.normal(0, 1, n)

    ax.plot(t, y, 'b-o', markersize=6, linewidth=1.5, label='$X_t$')
    ax.plot(t[1:], y[:-1], 'r--s', markersize=5, linewidth=1.5, alpha=0.7, label='$LX_t = X_{t-1}$')

    # Draw arrows showing the shift
    for i in [5, 10, 15]:
        ax.annotate('', xy=(t[i], y[i-1]), xytext=(t[i], y[i]),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))

    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Value')
    ax.set_title('Lag Operator: $L X_t = X_{t-1}$, shifts series back by one period')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=9, frameon=False)

    plt.tight_layout()
    save_fig('ch1_def_lag_operator')

ch1_def_lag_operator()

# 7. Centered Moving Average
def ch1_def_moving_average():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(9, 3.5))

    n = 60
    t = np.arange(n)
    y = 10 + 0.1*t + 3*np.sin(2*np.pi*t/12) + np.random.normal(0, 1.5, n)

    # Compute centered moving average (q=2, so 2q+1=5)
    q = 2
    ma = np.convolve(y, np.ones(2*q+1)/(2*q+1), mode='valid')
    t_ma = t[q:-q]

    ax.plot(t, y, 'b-', linewidth=1, alpha=0.7, label='Original $X_t$')
    ax.plot(t_ma, ma, 'r-', linewidth=2.5, label=f'Moving Average (order {2*q+1})')

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Centered Moving Average: $\\hat{m}_t = \\frac{1}{2q+1}\\sum_{j=-q}^{q} X_{t+j}$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=9, frameon=False)

    plt.tight_layout()
    save_fig('ch1_def_moving_average')

ch1_def_moving_average()

# 8. STL Decomposition
def ch1_def_stl():
    np.random.seed(42)
    fig, axes = plt.subplots(4, 1, figsize=(9, 6), sharex=True)

    n = 96
    t = np.arange(n)

    # Components
    trend = 10 + 0.1*t
    seasonal = 3*np.sin(2*np.pi*t/12)
    noise = np.random.normal(0, 0.5, n)
    y = trend + seasonal + noise

    axes[0].plot(t, y, 'b-', linewidth=1)
    axes[0].set_ylabel('Data')
    axes[0].set_title('STL Decomposition: $Y_t = T_t + S_t + R_t$')

    axes[1].plot(t, trend, 'r-', linewidth=2)
    axes[1].set_ylabel('Trend')

    axes[2].plot(t, seasonal, 'g-', linewidth=1.5)
    axes[2].set_ylabel('Seasonal')

    axes[3].plot(t, noise, 'gray', linewidth=1, alpha=0.7)
    axes[3].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[3].set_ylabel('Remainder')
    axes[3].set_xlabel('Time')

    plt.tight_layout()
    save_fig('ch1_def_stl')

ch1_def_stl()

# 9. ETS Models
def ch1_def_ets():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 50
    t = np.arange(n)

    # Generate data
    y = 10 + 0.2*t + np.random.normal(0, 1, n)

    # Simple exponential smoothing
    alpha = 0.3
    ses = np.zeros(n)
    ses[0] = y[0]
    for i in range(1, n):
        ses[i] = alpha * y[i] + (1 - alpha) * ses[i-1]

    axes[0].plot(t, y, 'b-o', markersize=3, linewidth=1, alpha=0.7, label='Data')
    axes[0].plot(t, ses, 'r-', linewidth=2, label=f'ETS (SES, $\\alpha$={alpha})')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Exponential Smoothing')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)

    # Forecast
    n_fore = 10
    t_fore = np.arange(n, n + n_fore)
    forecast = np.full(n_fore, ses[-1])

    axes[1].plot(t[-20:], y[-20:], 'b-o', markersize=4, label='Historical')
    axes[1].plot(t_fore, forecast, 'r--', linewidth=2, label='Forecast')
    axes[1].axvline(x=n-0.5, color='gray', linestyle=':', alpha=0.7)
    axes[1].set_xlabel('Time')
    axes[1].set_title('ETS Forecast')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)

    plt.tight_layout()
    save_fig('ch1_def_ets')

ch1_def_ets()

# =============================================================================
# CHAPTER 2: ARMA Definitions
# =============================================================================
print("\nChapter 2 Definition Charts:")

# 1. AR(1) Process
def ch2_def_ar1():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 150

    # Positive phi
    phi1 = 0.8
    y1 = np.zeros(n)
    eps = np.random.normal(0, 1, n)
    for t in range(1, n):
        y1[t] = phi1 * y1[t-1] + eps[t]

    axes[0].plot(y1, 'b-', linewidth=1)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title(f'AR(1): $X_t = {phi1} X_{{t-1}} + \\varepsilon_t$')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$X_t$')
    axes[0].text(75, max(y1)-0.5, f'$\\phi = {phi1}$: Persistent, slow decay', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Negative phi
    phi2 = -0.7
    y2 = np.zeros(n)
    for t in range(1, n):
        y2[t] = phi2 * y2[t-1] + eps[t]

    axes[1].plot(y2, 'g-', linewidth=1)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_title(f'AR(1): $X_t = {phi2} X_{{t-1}} + \\varepsilon_t$')
    axes[1].set_xlabel('Time')
    axes[1].text(75, max(y2)-0.5, f'$\\phi = {phi2}$: Alternating pattern', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    save_fig('ch2_def_ar1')

ch2_def_ar1()

# 2. AR(p) Process
def ch2_def_arp():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 150
    eps = np.random.normal(0, 1, n)

    # AR(2) with cyclical behavior
    phi1, phi2 = 0.6, -0.3
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = phi1 * y[t-1] + phi2 * y[t-2] + eps[t]

    axes[0].plot(y, 'b-', linewidth=1)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title(f'AR(2): $\\phi_1 = {phi1}$, $\\phi_2 = {phi2}$')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$X_t$')

    # ACF decay pattern
    lags = np.arange(0, 20)
    # Theoretical ACF for AR(2) - damped sinusoidal
    acf = np.zeros(20)
    acf[0] = 1
    acf[1] = phi1 / (1 - phi2)
    for k in range(2, 20):
        acf[k] = phi1 * acf[k-1] + phi2 * acf[k-2]

    axes[1].bar(lags, acf, color='steelblue', alpha=0.7)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title('AR(p) ACF: Decays (exponential or damped sine)')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('ACF')

    plt.tight_layout()
    save_fig('ch2_def_arp')

ch2_def_arp()

# 3. MA(1) Process
def ch2_def_ma1():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 150
    eps = np.random.normal(0, 1, n)

    # MA(1)
    theta = 0.7
    y = eps[1:] + theta * eps[:-1]

    axes[0].plot(y, 'b-', linewidth=1)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title(f'MA(1): $X_t = \\varepsilon_t + {theta}\\varepsilon_{{t-1}}$')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$X_t$')

    # ACF cuts off after lag 1
    lags = np.arange(0, 12)
    acf = np.zeros(12)
    acf[0] = 1
    acf[1] = theta / (1 + theta**2)

    ci = 1.96 / np.sqrt(n)
    axes[1].bar(lags, acf, color='green', alpha=0.7)
    axes[1].axhline(y=ci, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=-ci, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title('MA(1) ACF: Cuts off after lag 1')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('ACF')
    axes[1].set_ylim(-0.3, 1.1)

    plt.tight_layout()
    save_fig('ch2_def_ma1')

ch2_def_ma1()

# 4. MA(q) Process
def ch2_def_maq():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 150
    eps = np.random.normal(0, 1, n + 3)

    # MA(3)
    theta = [0.6, -0.4, 0.3]
    y = np.zeros(n)
    for t in range(n):
        y[t] = eps[t+3] + theta[0]*eps[t+2] + theta[1]*eps[t+1] + theta[2]*eps[t]

    axes[0].plot(y, 'b-', linewidth=1)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title('MA(3): $X_t = \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\theta_2\\varepsilon_{t-2} + \\theta_3\\varepsilon_{t-3}$')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$X_t$')

    # ACF cuts off after lag q
    lags = np.arange(0, 12)
    acf = np.zeros(12)
    acf[0] = 1
    acf[1] = 0.45
    acf[2] = 0.25
    acf[3] = 0.15

    ci = 1.96 / np.sqrt(n)
    axes[1].bar(lags, acf, color='purple', alpha=0.7)
    axes[1].axhline(y=ci, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=-ci, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title('MA(q) ACF: Cuts off after lag $q$')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('ACF')
    axes[1].set_ylim(-0.3, 1.1)
    axes[1].annotate('Cuts off', xy=(4, 0.05), fontsize=9, color='red')

    plt.tight_layout()
    save_fig('ch2_def_maq')

ch2_def_maq()

# 5. Invertibility
def ch2_def_invertibility():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    theta_values = np.linspace(-1.5, 1.5, 100)

    # Unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='blue', linewidth=2)
    axes[0].add_patch(circle)
    axes[0].set_xlim(-1.8, 1.8)
    axes[0].set_ylim(-1.8, 1.8)
    axes[0].set_aspect('equal')
    axes[0].axhline(y=0, color='gray', linewidth=0.5)
    axes[0].axvline(x=0, color='gray', linewidth=0.5)

    # Plot invertible and non-invertible points
    axes[0].plot(0.5, 0, 'go', markersize=12, label='Invertible: $|\\theta| < 1$')
    axes[0].plot(1.3, 0, 'rx', markersize=12, mew=3, label='Non-invertible: $|\\theta| > 1$')
    axes[0].set_title('Invertibility: Root outside unit circle')
    axes[0].legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    axes[0].set_xlabel('Real')
    axes[0].set_ylabel('Imaginary')

    # Show AR representation weights
    lags = np.arange(0, 15)
    theta_inv = 0.6
    theta_non = 1.2

    weights_inv = (-theta_inv) ** lags
    weights_non = (-theta_non) ** lags

    axes[1].plot(lags, np.abs(weights_inv), 'g-o', markersize=5, label=f'$\\theta = {theta_inv}$: Decaying')
    axes[1].plot(lags[:8], np.abs(weights_non[:8]), 'r--s', markersize=5, label=f'$\\theta = {theta_non}$: Exploding')
    axes[1].set_xlabel('Lag $j$')
    axes[1].set_ylabel('$|\\pi_j|$')
    axes[1].set_title('AR($\\infty$) weights: $X_t = \\sum \\pi_j X_{t-j} + \\varepsilon_t$')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)
    axes[1].set_ylim(0, 5)

    plt.tight_layout()
    save_fig('ch2_def_invertibility')

ch2_def_invertibility()

# 6. ARMA(p,q) Process
def ch2_def_arma():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 150
    eps = np.random.normal(0, 1, n)

    # ARMA(1,1)
    phi, theta = 0.6, 0.4
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t-1] + eps[t] + theta * eps[t-1]

    axes[0].plot(y, 'purple', linewidth=1)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title(f'ARMA(1,1): $X_t = {phi}X_{{t-1}} + \\varepsilon_t + {theta}\\varepsilon_{{t-1}}$')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$X_t$')

    # ACF pattern
    lags = np.arange(0, 15)
    acf = np.zeros(15)
    acf[0] = 1
    # ARMA(1,1) ACF starts with spike then decays
    acf[1] = 0.75
    for k in range(2, 15):
        acf[k] = phi * acf[k-1]

    ci = 1.96 / np.sqrt(n)
    axes[1].bar(lags, acf, color='purple', alpha=0.7)
    axes[1].axhline(y=ci, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=-ci, color='red', linestyle='--', alpha=0.7)
    axes[1].set_title('ARMA ACF: Decay after lag $q$')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('ACF')
    axes[1].set_ylim(-0.3, 1.1)

    plt.tight_layout()
    save_fig('ch2_def_arma')

ch2_def_arma()

# 7. Ljung-Box Test
def ch2_def_ljungbox():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 100
    lags = np.arange(1, 21)
    ci = 1.96 / np.sqrt(n)

    # Good residuals (white noise)
    np.random.seed(42)
    resid_good = np.random.normal(0, 1, n)
    acf_good = np.array([np.corrcoef(resid_good[k:], resid_good[:-k])[0,1] for k in range(1, 21)])

    axes[0].bar(lags, acf_good, color='green', alpha=0.7)
    axes[0].axhline(y=ci, color='red', linestyle='--', alpha=0.7)
    axes[0].axhline(y=-ci, color='red', linestyle='--', alpha=0.7)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_title('Good Model: Residuals are white noise')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF of Residuals')
    axes[0].set_ylim(-0.4, 0.4)
    axes[0].text(10, 0.35, 'Ljung-Box: Fail to reject $H_0$', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Bad residuals (autocorrelated)
    np.random.seed(42)
    eps = np.random.normal(0, 1, n)
    resid_bad = np.zeros(n)
    for t in range(1, n):
        resid_bad[t] = 0.5 * resid_bad[t-1] + eps[t]
    acf_bad = np.array([np.corrcoef(resid_bad[k:], resid_bad[:-k])[0,1] for k in range(1, 21)])

    axes[1].bar(lags, acf_bad, color='red', alpha=0.7)
    axes[1].axhline(y=ci, color='blue', linestyle='--', alpha=0.7)
    axes[1].axhline(y=-ci, color='blue', linestyle='--', alpha=0.7)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title('Poor Model: Residuals are autocorrelated')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylim(-0.4, 0.6)
    axes[1].text(10, 0.5, 'Ljung-Box: Reject $H_0$', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    plt.tight_layout()
    save_fig('ch2_def_ljungbox')

ch2_def_ljungbox()

# =============================================================================
# CHAPTER 3: ARIMA Definitions
# =============================================================================
print("\nChapter 3 Definition Charts:")

# 1. Random Walk
def ch3_def_random_walk():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 150

    # Multiple random walks
    for i, seed in enumerate([42, 123, 456]):
        np.random.seed(seed)
        rw = np.cumsum(np.random.normal(0, 1, n))
        axes[0].plot(rw, linewidth=1.5, alpha=0.8, label=f'Path {i+1}')

    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title('Random Walk: $Y_t = Y_{t-1} + \\varepsilon_t$')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), fontsize=8, frameon=False, ncol=3)

    # Variance grows with time
    np.random.seed(42)
    n_sim = 1000
    t_points = [10, 50, 100, 150]
    variances = []

    for t in t_points:
        endpoints = [np.sum(np.random.normal(0, 1, t)) for _ in range(n_sim)]
        variances.append(np.var(endpoints))

    axes[1].plot(t_points, variances, 'b-o', markersize=8, linewidth=2)
    axes[1].plot(t_points, t_points, 'r--', linewidth=1.5, label='Theoretical: $Var = t\\sigma^2$')
    axes[1].set_title('Variance grows linearly with time')
    axes[1].set_xlabel('Time $t$')
    axes[1].set_ylabel('Variance')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), fontsize=8, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    save_fig('ch3_def_random_walk')

ch3_def_random_walk()

# 2. Random Walk with Drift
def ch3_def_random_walk_drift():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(9, 3.5))

    n = 150
    eps = np.random.normal(0, 1, n)

    # Random walk without drift
    rw = np.cumsum(eps)

    # Random walk with drift
    mu = 0.3
    rw_drift = mu * np.arange(n) + np.cumsum(eps)

    ax.plot(rw, 'b-', linewidth=1.5, label='No drift: $Y_t = Y_{t-1} + \\varepsilon_t$')
    ax.plot(rw_drift, 'r-', linewidth=1.5, label=f'With drift: $Y_t = {mu} + Y_{{t-1}} + \\varepsilon_t$')
    ax.plot(mu * np.arange(n), 'r--', linewidth=1, alpha=0.5, label='Drift trend')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    ax.set_title('Random Walk: With and Without Drift')
    ax.set_xlabel('Time')
    ax.set_ylabel('$Y_t$')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), fontsize=8, frameon=False, ncol=3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    save_fig('ch3_def_random_walk_drift')

ch3_def_random_walk_drift()

# 3. Integrated Process
def ch3_def_integrated():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    n = 150
    eps = np.random.normal(0, 1, n)

    # I(0) - Stationary
    y0 = np.zeros(n)
    for t in range(1, n):
        y0[t] = 0.7 * y0[t-1] + eps[t]

    axes[0].plot(y0, 'g-', linewidth=1)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title('$I(0)$: Stationary')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')

    # I(1) - Random walk
    y1 = np.cumsum(eps)

    axes[1].plot(y1, 'b-', linewidth=1)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('$I(1)$: One difference to stationary')
    axes[1].set_xlabel('Time')

    # I(2)
    y2 = np.cumsum(np.cumsum(eps))

    axes[2].plot(y2, 'r-', linewidth=1)
    axes[2].set_title('$I(2)$: Two differences to stationary')
    axes[2].set_xlabel('Time')

    plt.tight_layout()
    save_fig('ch3_def_integrated')

ch3_def_integrated()

# 4. First Difference
def ch3_def_difference():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 100

    # Non-stationary series
    eps = np.random.normal(0, 1, n)
    y = np.cumsum(eps) + 10

    axes[0].plot(y, 'b-', linewidth=1.5)
    axes[0].set_title('Original: $Y_t$ (non-stationary)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')

    # Differenced series
    dy = np.diff(y)

    axes[1].plot(dy, 'g-', linewidth=1)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_title('Differenced: $\\Delta Y_t = Y_t - Y_{t-1}$ (stationary)')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('$\\Delta Y_t$')

    plt.tight_layout()
    save_fig('ch3_def_difference')

ch3_def_difference()

# 5. ARIMA(p,d,q)
def ch3_def_arima():
    np.random.seed(42)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    n = 150
    eps = np.random.normal(0, 1, n)

    # Generate ARIMA(1,1,1)
    # First create stationary ARMA(1,1)
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.6 * z[t-1] + eps[t] + 0.4 * eps[t-1]

    # Integrate to get I(1)
    y = np.cumsum(z) + 50

    axes[0, 0].plot(y, 'b-', linewidth=1)
    axes[0, 0].set_title('Original: $Y_t$ (ARIMA(1,1,1))')
    axes[0, 0].set_ylabel('$Y_t$')

    # First difference
    dy = np.diff(y)
    axes[0, 1].plot(dy, 'g-', linewidth=1)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('After $d=1$ difference: ARMA(1,1)')

    # ACF of differenced series
    lags = np.arange(0, 15)
    acf = np.zeros(15)
    acf[0] = 1
    acf[1] = 0.65
    for k in range(2, 15):
        acf[k] = 0.6 * acf[k-1]

    ci = 1.96 / np.sqrt(n)
    axes[1, 0].bar(lags, acf, color='steelblue', alpha=0.7)
    axes[1, 0].axhline(y=ci, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].axhline(y=-ci, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('ACF of $\\Delta Y_t$')
    axes[1, 0].set_xlabel('Lag')
    axes[1, 0].set_ylabel('ACF')

    # PACF
    pacf = np.zeros(15)
    pacf[0] = 1
    pacf[1] = 0.65
    pacf[2:] = np.random.normal(0, 0.05, 13)

    axes[1, 1].bar(lags, pacf, color='orange', alpha=0.7)
    axes[1, 1].axhline(y=ci, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].axhline(y=-ci, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('PACF of $\\Delta Y_t$')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('PACF')

    plt.tight_layout()
    save_fig('ch3_def_arima')

ch3_def_arima()

# 6. ADF Test
def ch3_def_adf():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 100

    # Stationary series (reject unit root)
    eps = np.random.normal(0, 1, n)
    y_stat = np.zeros(n)
    for t in range(1, n):
        y_stat[t] = 0.5 * y_stat[t-1] + eps[t]

    axes[0].plot(y_stat, 'g-', linewidth=1)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title('Stationary: Reject $H_0$ (no unit root)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('$Y_t$')
    axes[0].text(50, max(y_stat)-0.5, 'ADF stat < critical value\np-value < 0.05', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Non-stationary (fail to reject)
    y_nonstat = np.cumsum(eps)

    axes[1].plot(y_nonstat, 'r-', linewidth=1)
    axes[1].axhline(y=0, color='blue', linestyle='--', alpha=0.5)
    axes[1].set_title('Non-Stationary: Fail to reject $H_0$ (unit root)')
    axes[1].set_xlabel('Time')
    axes[1].text(50, max(y_nonstat)-2, 'ADF stat > critical value\np-value > 0.05', ha='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    plt.tight_layout()
    save_fig('ch3_def_adf')

ch3_def_adf()

# =============================================================================
# CHAPTER 4: SARIMA Definitions
# =============================================================================
print("\nChapter 4 Definition Charts:")

# 1. Seasonality
def ch4_def_seasonality():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Monthly seasonality (s=12)
    t = np.arange(48)
    trend = 100 + 0.5*t
    seasonal = 15*np.sin(2*np.pi*t/12)
    noise = np.random.normal(0, 3, 48)
    y = trend + seasonal + noise

    axes[0].plot(t, y, 'b-o', markersize=4, linewidth=1)
    axes[0].set_title('Monthly Data: Period $s=12$')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Value')

    # Highlight seasonal pattern
    for i in range(4):
        axes[0].axvspan(i*12, (i+1)*12 - 0.5, alpha=0.1, color=['blue', 'orange'][i % 2])

    # Quarterly seasonality (s=4)
    t2 = np.arange(24)
    seasonal2 = 10*np.sin(2*np.pi*t2/4)
    y2 = 50 + 0.8*t2 + seasonal2 + np.random.normal(0, 2, 24)

    axes[1].plot(t2, y2, 'g-s', markersize=5, linewidth=1)
    axes[1].set_title('Quarterly Data: Period $s=4$')
    axes[1].set_xlabel('Quarter')

    for i in range(6):
        axes[1].axvspan(i*4, (i+1)*4 - 0.5, alpha=0.1, color=['green', 'purple'][i % 2])

    plt.tight_layout()
    save_fig('ch4_def_seasonality')

ch4_def_seasonality()

# 2. Seasonal Difference
def ch4_def_seasonal_diff():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Monthly data with seasonality
    n = 48
    t = np.arange(n)
    trend = 100 + t
    seasonal = 20*np.sin(2*np.pi*t/12)
    noise = np.random.normal(0, 3, n)
    y = trend + seasonal + noise

    axes[0].plot(t, y, 'b-', linewidth=1.5)
    axes[0].set_title('Original: $Y_t$ with seasonal pattern')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('$Y_t$')

    # Seasonal difference
    s = 12
    dy = y[s:] - y[:-s]
    t_diff = t[s:]

    axes[1].plot(t_diff, dy, 'g-', linewidth=1.5)
    axes[1].axhline(y=np.mean(dy), color='red', linestyle='--', alpha=0.7)
    axes[1].set_title(f'Seasonal Diff: $\\Delta_{{12}} Y_t = Y_t - Y_{{t-12}}$')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('$\\Delta_{12} Y_t$')

    plt.tight_layout()
    save_fig('ch4_def_seasonal_diff')

ch4_def_seasonal_diff()

# 3. SARIMA
def ch4_def_sarima():
    np.random.seed(42)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    n = 96
    t = np.arange(n)

    # Generate SARIMA-like data
    trend = 50 + 0.3*t
    seasonal = 15*np.sin(2*np.pi*t/12)
    ar_component = np.zeros(n)
    eps = np.random.normal(0, 2, n)
    for i in range(1, n):
        ar_component[i] = 0.5 * ar_component[i-1] + eps[i]

    y = trend + seasonal + ar_component

    axes[0, 0].plot(t, y, 'b-', linewidth=1)
    axes[0, 0].set_title('SARIMA$(1,1,1)\\times(1,1,1)_{12}$ Series')
    axes[0, 0].set_ylabel('$Y_t$')

    # After differencing
    dy = np.diff(y)
    axes[0, 1].plot(dy, 'r-', linewidth=1)
    axes[0, 1].set_title('After regular difference $\\Delta Y_t$')

    # After seasonal differencing
    s = 12
    dsy = y[s:] - y[:-s]
    axes[1, 0].plot(dsy, 'g-', linewidth=1)
    axes[1, 0].set_title('After seasonal difference $\\Delta_{12} Y_t$')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Value')

    # Both differences
    ddsy = np.diff(dsy)
    axes[1, 1].plot(ddsy, 'purple', linewidth=1)
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].set_title('Both differences: $\\Delta\\Delta_{12} Y_t$')
    axes[1, 1].set_xlabel('Time')

    plt.tight_layout()
    save_fig('ch4_def_sarima')

ch4_def_sarima()

# =============================================================================
# CHAPTER 5: VAR Definitions
# =============================================================================
print("\nChapter 5 Definition Charts:")

# 1. Cross-Correlation Function
def ch5_def_ccf():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    n = 150

    # Generate two related series
    eps1 = np.random.normal(0, 1, n)
    eps2 = np.random.normal(0, 1, n)

    x = np.zeros(n)
    y = np.zeros(n)

    for t in range(1, n):
        x[t] = 0.6 * x[t-1] + eps1[t]
        y[t] = 0.3 * x[t-1] + 0.5 * y[t-1] + eps2[t]  # y depends on lagged x

    axes[0].plot(x, 'b-', linewidth=1, label='$X_t$')
    axes[0].plot(y, 'r-', linewidth=1, alpha=0.7, label='$Y_t$')
    axes[0].set_title('Two Related Time Series')
    axes[0].set_xlabel('Time')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, frameon=False)

    # Cross-correlation using numpy correlate for simplicity
    from scipy import signal
    ccf_full = np.correlate(x - np.mean(x), y - np.mean(y), mode='full')
    ccf_full = ccf_full / (n * np.std(x) * np.std(y))
    mid = len(ccf_full) // 2
    lags = np.arange(-15, 16)
    ccf = ccf_full[mid-15:mid+16]

    ci = 1.96 / np.sqrt(n)
    axes[1].bar(lags, ccf, color='purple', alpha=0.7)
    axes[1].axhline(y=ci, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=-ci, color='red', linestyle='--', alpha=0.7)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_title('Cross-Correlation: $\\rho_{XY}(k)$')
    axes[1].set_xlabel('Lag $k$')
    axes[1].set_ylabel('CCF')
    axes[1].text(8, 0.35, '$X$ leads $Y$', fontsize=9)

    plt.tight_layout()
    save_fig('ch5_def_ccf')

ch5_def_ccf()

print("\nAll definition illustration charts created successfully!")
