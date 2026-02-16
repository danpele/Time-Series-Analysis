"""
TSA_ch1_python_viz
==================
Python Exercise 1: Loading and Visualization

Task: Load S&P 500 data and create visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

print("=" * 60)
print("PYTHON EXERCISE 1: Time Series Visualization")
print("=" * 60)

# Download S&P 500 data
print("\nDownloading S&P 500 data...")
sp500 = yf.download('^GSPC', start='2020-01-01', end='2025-01-01', progress=False)
print(f"Downloaded {len(sp500)} observations")
print(f"Date range: {sp500.index[0].date()} to {sp500.index[-1].date()}")

# Extract closing prices
prices = sp500['Close'].squeeze()

# Calculate returns
returns = prices.pct_change().dropna() * 100  # in percentage

# Basic statistics
print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)

print("\nPRICES:")
print(f"  Mean:     {prices.mean():.2f}")
print(f"  Std Dev:  {prices.std():.2f}")
print(f"  Min:      {prices.min():.2f}")
print(f"  Max:      {prices.max():.2f}")
print(f"  Range:    {prices.max() - prices.min():.2f}")

print("\nRETURNS (%):")
print(f"  Mean:     {returns.mean():.4f}%")
print(f"  Std Dev:  {returns.std():.4f}%")
print(f"  Min:      {returns.min():.4f}%")
print(f"  Max:      {returns.max():.4f}%")
print(f"  Skewness: {returns.skew():.4f}")
print(f"  Kurtosis: {returns.kurtosis():.4f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: Prices
ax1 = axes[0, 0]
ax1.plot(prices.index, prices.values, 'b-', linewidth=0.8)
ax1.set_title('S&P 500 Closing Prices', fontsize=12)
ax1.set_xlabel('Date')
ax1.set_ylabel('Price ($)')
ax1.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(range(len(prices)), prices.values, 1)
p = np.poly1d(z)
ax1.plot(prices.index, p(range(len(prices))), 'r--', linewidth=2, label='Linear Trend')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

# Top-right: Returns
ax2 = axes[0, 1]
ax2.plot(returns.index, returns.values, 'b-', linewidth=0.5, alpha=0.8)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax2.axhline(y=returns.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean = {returns.mean():.3f}%')
ax2.set_title('S&P 500 Daily Returns (%)', fontsize=12)
ax2.set_xlabel('Date')
ax2.set_ylabel('Return (%)')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

# Bottom-left: Histogram of returns
ax3 = axes[1, 0]
ax3.hist(returns, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
# Overlay normal distribution
from scipy import stats

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

x = np.linspace(returns.min(), returns.max(), 100)
ax3.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()), 'r-', linewidth=2, label='Normal')
ax3.set_title('Distribution of Returns', fontsize=12)
ax3.set_xlabel('Return (%)')
ax3.set_ylabel('Density')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax3.grid(True, alpha=0.3)

# Bottom-right: Rolling statistics
ax4 = axes[1, 1]
window = 60  # 60-day rolling window
rolling_mean = prices.rolling(window=window).mean()
rolling_std = prices.rolling(window=window).std()

ax4.plot(prices.index, prices.values, 'b-', linewidth=0.5, alpha=0.5, label='Prices')
ax4.plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2, label=f'{window}-day MA')
ax4.fill_between(rolling_mean.index,
                 rolling_mean.values - 2*rolling_std.values,
                 rolling_mean.values + 2*rolling_std.values,
                 alpha=0.2, color='red', label='±2 SD')
ax4.set_title(f'{window}-Day Rolling Statistics', fontsize=12)
ax4.set_xlabel('Date')
ax4.set_ylabel('Price ($)')
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../charts/ch1_python_viz.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch1_python_viz.pdf', bbox_inches='tight', transparent=True)
plt.show()

# Stationarity assessment
print("\n" + "=" * 60)
print("STATIONARITY ASSESSMENT")
print("=" * 60)

print("\nPRICES - Does the series appear stationary?")
print("  - Clear upward TREND visible")
print("  - Mean is NOT constant over time")
print("  - Variance appears to increase with level")
print("  → CONCLUSION: NOT stationary")

print("\nRETURNS - Does the series appear stationary?")
print("  - Fluctuates around zero (constant mean)")
print("  - No obvious trend")
print("  - Some volatility clustering (variance not perfectly constant)")
print("  → CONCLUSION: Approximately stationary")
print("    (may have conditional heteroscedasticity)")

print("\n" + "=" * 60)
print("KEY TAKEAWAYS:")
print("=" * 60)
print("""
1. Financial PRICES are typically non-stationary
   - They show trending behavior
   - Variance grows with the level

2. Financial RETURNS are approximately stationary
   - Fluctuate around a constant mean (near zero)
   - No trend component

3. This is why we model RETURNS, not prices!

4. Next step: Formal stationarity tests (ADF, KPSS)
""")
