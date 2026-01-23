#!/usr/bin/env python3
"""
Generate charts for Chapter 6: Cointegration and VECM
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# Set style
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.grid'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

# Colors
MAIN_BLUE = '#1A3A6E'
IDA_RED = '#DC3545'
FOREST = '#2E7D32'
AMBER = '#B5853F'
ORANGE = '#E67E22'

output_dir = 'charts'
os.makedirs(output_dir, exist_ok=True)

np.random.seed(42)

# =============================================================================
# 1. Spurious Regression Example
# =============================================================================
def plot_spurious_regression():
    """Two independent random walks that appear correlated"""
    T = 200

    # Generate two independent random walks
    e1 = np.random.normal(0, 1, T)
    e2 = np.random.normal(0, 1, T)

    y1 = np.cumsum(e1)
    y2 = np.cumsum(e2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Time series plot
    ax1 = axes[0]
    ax1.plot(y1, color=MAIN_BLUE, linewidth=1.5, label='$Y_t$ (Random Walk 1)')
    ax1.plot(y2, color=IDA_RED, linewidth=1.5, label='$X_t$ (Random Walk 2)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_title('Two Independent Random Walks')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)

    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(y2, y1, alpha=0.5, color=MAIN_BLUE, s=20)

    # Fit line
    coef = np.polyfit(y2, y1, 1)
    x_line = np.linspace(min(y2), max(y2), 100)
    y_line = coef[0] * x_line + coef[1]
    ax2.plot(x_line, y_line, color=IDA_RED, linewidth=2, label=f'OLS: $R^2$ = {np.corrcoef(y1, y2)[0,1]**2:.3f}')

    ax2.set_xlabel('$X_t$')
    ax2.set_ylabel('$Y_t$')
    ax2.set_title('Spurious Regression: High $R^2$ but No True Relationship!')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/spurious_regression.pdf', bbox_inches='tight', transparent=True)
    plt.close()

# =============================================================================
# 2. Cointegrated Series Example
# =============================================================================
def plot_cointegrated_series():
    """Two cointegrated series sharing a common trend"""
    T = 200

    # Common stochastic trend
    trend = np.cumsum(np.random.normal(0, 1, T))

    # Stationary components
    s1 = np.zeros(T)
    s2 = np.zeros(T)
    for t in range(1, T):
        s1[t] = 0.7 * s1[t-1] + np.random.normal(0, 0.5)
        s2[t] = 0.6 * s2[t-1] + np.random.normal(0, 0.5)

    # Cointegrated series
    y1 = trend + s1
    y2 = 0.8 * trend + s2 + 5  # Different loading on trend

    # Cointegrating relationship (spread)
    spread = y1 - 1.25 * y2 + 6.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Time series
    ax1 = axes[0]
    ax1.plot(y1, color=MAIN_BLUE, linewidth=1.5, label='$Y_{1t}$')
    ax1.plot(y2, color=IDA_RED, linewidth=1.5, label='$Y_{2t}$')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_title('Cointegrated Series: Share Common Trend')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)

    # Spread (error correction term)
    ax2 = axes[1]
    ax2.plot(spread, color=FOREST, linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.fill_between(range(T), spread, 0, alpha=0.3, color=FOREST)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Spread')
    ax2.set_title('Cointegrating Relation: $Y_1 - 1.25 Y_2$ is Stationary')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/cointegrated_series.pdf', bbox_inches='tight', transparent=True)
    plt.close()

# =============================================================================
# 3. Error Correction Mechanism
# =============================================================================
def plot_error_correction():
    """Illustration of error correction mechanism"""
    T = 150

    # Parameters
    alpha1 = -0.15  # Y1 adjusts
    alpha2 = 0.10   # Y2 adjusts
    beta = 1.0      # Cointegrating coefficient

    y1 = np.zeros(T)
    y2 = np.zeros(T)
    y1[0] = 10
    y2[0] = 10

    # Equilibrium
    equilibrium = 0

    # Shock at t=30
    for t in range(1, T):
        error = y1[t-1] - beta * y2[t-1] - equilibrium

        if t == 30:
            # Shock to y1
            y1[t] = y1[t-1] + alpha1 * error + 3 + np.random.normal(0, 0.2)
        else:
            y1[t] = y1[t-1] + alpha1 * error + np.random.normal(0, 0.2)

        y2[t] = y2[t-1] + alpha2 * error + np.random.normal(0, 0.2)

    spread = y1 - beta * y2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Series
    ax1 = axes[0]
    ax1.plot(y1, color=MAIN_BLUE, linewidth=1.5, label='$Y_{1t}$')
    ax1.plot(y2, color=IDA_RED, linewidth=1.5, label='$Y_{2t}$')
    ax1.axvline(x=30, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax1.annotate('Shock', xy=(30, y1[30]), xytext=(45, y1[30]+1),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_title('Error Correction: Variables Adjust After Shock')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)

    # Spread
    ax2 = axes[1]
    ax2.plot(spread, color=FOREST, linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.axvline(x=30, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.fill_between(range(T), spread, 0, alpha=0.3, color=FOREST)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Deviation from Equilibrium')
    ax2.set_title('Error Correction Term: Reverts to Zero')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/error_correction.pdf', bbox_inches='tight', transparent=True)
    plt.close()

# =============================================================================
# 4. Real Example: Interest Rates
# =============================================================================
def plot_interest_rates():
    """Simulated short and long term interest rates"""
    T = 250

    # Generate realistic interest rate data
    # Common trend (level of rates)
    trend = np.cumsum(np.random.normal(0, 0.1, T))
    trend = trend - trend[0] + 5  # Start at 5%

    # Short rate follows trend with more volatility
    short_rate = trend + np.random.normal(0, 0.3, T)
    for t in range(1, T):
        short_rate[t] = 0.95 * short_rate[t-1] + 0.05 * trend[t] + np.random.normal(0, 0.1)

    # Long rate is smoother, leads short rate slightly
    long_rate = np.zeros(T)
    long_rate[0] = short_rate[0] + 1
    for t in range(1, T):
        long_rate[t] = 0.98 * long_rate[t-1] + 0.02 * (short_rate[t] + 1.5) + np.random.normal(0, 0.05)

    # Spread
    spread = long_rate - short_rate

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Interest rates
    ax1 = axes[0]
    ax1.plot(short_rate, color=MAIN_BLUE, linewidth=1.5, label='Short Rate (3-month)')
    ax1.plot(long_rate, color=IDA_RED, linewidth=1.5, label='Long Rate (10-year)')
    ax1.set_xlabel('Time (months)')
    ax1.set_ylabel('Interest Rate (%)')
    ax1.set_title('Term Structure: Short and Long Interest Rates')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)

    # Spread
    ax2 = axes[1]
    ax2.plot(spread, color=FOREST, linewidth=1.5)
    ax2.axhline(y=spread.mean(), color=IDA_RED, linestyle='--', linewidth=1, label=f'Mean = {spread.mean():.2f}%')
    ax2.fill_between(range(T), spread, spread.mean(), alpha=0.3, color=FOREST)
    ax2.set_xlabel('Time (months)')
    ax2.set_ylabel('Spread (%)')
    ax2.set_title('Term Spread: Stationary (Cointegrating Relation)')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/interest_rates_coint.pdf', bbox_inches='tight', transparent=True)
    plt.close()

# =============================================================================
# 5. Real Example: Stock Prices (Pairs Trading)
# =============================================================================
def plot_pairs_trading():
    """Two related stock prices for pairs trading"""
    T = 250

    # Common market factor
    market = np.cumsum(np.random.normal(0.0005, 0.015, T))

    # Two correlated stocks (e.g., Coca-Cola and Pepsi)
    stock1 = 100 * np.exp(market + np.cumsum(np.random.normal(0, 0.005, T)))
    stock2 = 80 * np.exp(0.9 * market + np.cumsum(np.random.normal(0, 0.005, T)))

    # Log prices
    log_p1 = np.log(stock1)
    log_p2 = np.log(stock2)

    # Spread (hedge ratio estimated)
    beta = np.cov(log_p1, log_p2)[0,1] / np.var(log_p2)
    spread = log_p1 - beta * log_p2
    spread = spread - spread.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Stock prices
    ax1 = axes[0]
    ax1.plot(stock1, color=MAIN_BLUE, linewidth=1.5, label='Stock A (e.g., Coca-Cola)')
    ax1.plot(stock2, color=IDA_RED, linewidth=1.5, label='Stock B (e.g., Pepsi)')
    ax1.set_xlabel('Trading Days')
    ax1.set_ylabel('Price ($)')
    ax1.set_title('Related Stocks: Share Common Market Factor')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)

    # Spread for pairs trading
    ax2 = axes[1]
    ax2.plot(spread, color=FOREST, linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.axhline(y=2*spread.std(), color=IDA_RED, linestyle=':', linewidth=1, alpha=0.7)
    ax2.axhline(y=-2*spread.std(), color=IDA_RED, linestyle=':', linewidth=1, alpha=0.7)
    ax2.fill_between(range(T), spread, 0, alpha=0.3, color=FOREST)
    ax2.set_xlabel('Trading Days')
    ax2.set_ylabel('Spread (log prices)')
    ax2.set_title('Pairs Trading: Spread Mean-Reverts')
    ax2.annotate('Sell spread', xy=(T*0.7, 2*spread.std()), fontsize=9, color=IDA_RED)
    ax2.annotate('Buy spread', xy=(T*0.7, -2*spread.std()), fontsize=9, color=FOREST)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/pairs_trading.pdf', bbox_inches='tight', transparent=True)
    plt.close()

# =============================================================================
# 6. PPP Example: Exchange Rate and Price Levels
# =============================================================================
def plot_ppp_example():
    """Exchange rate and relative price levels (PPP)"""
    T = 200

    # Common trend (real exchange rate drift)
    trend = np.cumsum(np.random.normal(0, 0.01, T))

    # Log exchange rate (e.g., USD/EUR)
    exchange_rate = 0.1 + trend + np.cumsum(np.random.normal(0, 0.005, T))

    # Relative price level (P_US / P_EUR in logs)
    price_diff = 0.05 + 0.9 * trend + np.cumsum(np.random.normal(0, 0.003, T))

    # Real exchange rate (should be stationary under PPP)
    real_rate = exchange_rate - price_diff

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Nominal series
    ax1 = axes[0]
    ax1.plot(exchange_rate, color=MAIN_BLUE, linewidth=1.5, label='Log Exchange Rate')
    ax1.plot(price_diff, color=IDA_RED, linewidth=1.5, label='Log Price Differential')
    ax1.set_xlabel('Time (quarters)')
    ax1.set_ylabel('Log Value')
    ax1.set_title('Exchange Rate and Relative Prices')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)

    # Real exchange rate
    ax2 = axes[1]
    ax2.plot(real_rate, color=FOREST, linewidth=1.5)
    ax2.axhline(y=real_rate.mean(), color=IDA_RED, linestyle='--', linewidth=1)
    ax2.fill_between(range(T), real_rate, real_rate.mean(), alpha=0.3, color=FOREST)
    ax2.set_xlabel('Time (quarters)')
    ax2.set_ylabel('Real Exchange Rate')
    ax2.set_title('PPP: Real Exchange Rate is Stationary')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ppp_cointegration.pdf', bbox_inches='tight', transparent=True)
    plt.close()

# =============================================================================
# 7. VECM Impulse Response
# =============================================================================
def plot_vecm_irf():
    """Impulse response in a cointegrated system"""
    T = 50

    # VECM parameters
    alpha1 = -0.2
    alpha2 = 0.1

    # Response to shock in y1
    y1_resp = np.zeros(T)
    y2_resp = np.zeros(T)
    y1_resp[0] = 1  # Unit shock

    for t in range(1, T):
        error = y1_resp[t-1] - y2_resp[t-1]
        y1_resp[t] = y1_resp[t-1] + alpha1 * error
        y2_resp[t] = y2_resp[t-1] + alpha2 * error

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # IRF
    ax1 = axes[0]
    ax1.plot(y1_resp, color=MAIN_BLUE, linewidth=2, label='$Y_1$ response', marker='o', markersize=3)
    ax1.plot(y2_resp, color=IDA_RED, linewidth=2, label='$Y_2$ response', marker='s', markersize=3)
    ax1.axhline(y=y1_resp[-1], color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Periods after shock')
    ax1.set_ylabel('Response')
    ax1.set_title('IRF: Shock to $Y_1$ in Cointegrated System')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=2)
    ax1.set_xlim(0, T-1)

    # Spread response
    spread_resp = y1_resp - y2_resp
    ax2 = axes[1]
    ax2.plot(spread_resp, color=FOREST, linewidth=2, marker='o', markersize=3)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.fill_between(range(T), spread_resp, 0, alpha=0.3, color=FOREST)
    ax2.set_xlabel('Periods after shock')
    ax2.set_ylabel('Spread ($Y_1 - Y_2$)')
    ax2.set_title('Error Correction Term Returns to Zero')
    ax2.set_xlim(0, T-1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/vecm_irf.pdf', bbox_inches='tight', transparent=True)
    plt.close()

# =============================================================================
# 8. Johansen Test Visualization
# =============================================================================
def plot_eigenvalues():
    """Visualization of Johansen eigenvalues"""
    # Example eigenvalues for 3-variable system
    eigenvalues = [0.35, 0.12, 0.03]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(1, 4)
    bars = ax.bar(x, eigenvalues, color=[IDA_RED, AMBER, MAIN_BLUE], width=0.6)

    # Add threshold line
    ax.axhline(y=0.15, color='gray', linestyle='--', linewidth=1.5, label='Significance threshold (approx)')

    ax.set_xlabel('Eigenvalue Number')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Johansen Test: Eigenvalues Indicate Cointegrating Rank')
    ax.set_xticks(x)
    ax.set_xticklabels(['$\\lambda_1$ = 0.35\n(Significant)', '$\\lambda_2$ = 0.12\n(Not significant)', '$\\lambda_3$ = 0.03\n(Not significant)'])

    # Annotate
    ax.annotate('r = 1 cointegrating\nrelationship', xy=(1, 0.35), xytext=(1.5, 0.42),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/johansen_eigenvalues.pdf', bbox_inches='tight', transparent=True)
    plt.close()

# =============================================================================
# Run all
# =============================================================================
if __name__ == '__main__':
    print("Generating Chapter 6 charts...")

    plot_spurious_regression()
    print("  - spurious_regression.pdf")

    plot_cointegrated_series()
    print("  - cointegrated_series.pdf")

    plot_error_correction()
    print("  - error_correction.pdf")

    plot_interest_rates()
    print("  - interest_rates_coint.pdf")

    plot_pairs_trading()
    print("  - pairs_trading.pdf")

    plot_ppp_example()
    print("  - ppp_cointegration.pdf")

    plot_vecm_irf()
    print("  - vecm_irf.pdf")

    plot_eigenvalues()
    print("  - johansen_eigenvalues.pdf")

    print("Done!")
