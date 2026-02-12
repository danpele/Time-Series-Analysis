"""
TSA_ch1_python_adf
==================
Python Exercise 2: Stationarity Testing

Task: Test stationarity using ADF and KPSS tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("PYTHON EXERCISE 2: Stationarity Testing")
print("=" * 60)

# Download S&P 500 data
print("\nDownloading S&P 500 data...")
sp500 = yf.download('^GSPC', start='2020-01-01', end='2025-01-01', progress=False)
prices = sp500['Close']
returns = prices.pct_change().dropna()

print(f"Prices: {len(prices)} observations")
print(f"Returns: {len(returns)} observations")

def run_adf_test(series, name):
    """Run ADF test and print results"""
    result = adfuller(series, autolag='AIC')
    print(f"\nADF Test for {name}:")
    print("-" * 40)
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value:        {result[1]:.6f}")
    print(f"  Lags Used:      {result[2]}")
    print(f"  Observations:   {result[3]}")
    print("  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.4f}")

    # Interpretation
    print("\n  INTERPRETATION:")
    print(f"    H₀: Series has unit root (non-stationary)")
    if result[1] < 0.05:
        print(f"    p-value ({result[1]:.4f}) < 0.05 → REJECT H₀")
        print(f"    → Evidence for STATIONARITY")
        conclusion = "STATIONARY"
    else:
        print(f"    p-value ({result[1]:.4f}) ≥ 0.05 → FAIL TO REJECT H₀")
        print(f"    → Evidence for NON-STATIONARITY (unit root)")
        conclusion = "NON-STATIONARY"

    return result[0], result[1], conclusion

def run_kpss_test(series, name):
    """Run KPSS test and print results"""
    result = kpss(series, regression='c', nlags='auto')
    print(f"\nKPSS Test for {name}:")
    print("-" * 40)
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value:        {result[1]:.4f}")
    print(f"  Lags Used:      {result[2]}")
    print("  Critical Values:")
    for key, value in result[3].items():
        print(f"    {key}: {value:.4f}")

    # Interpretation
    print("\n  INTERPRETATION:")
    print(f"    H₀: Series is stationary")
    if result[1] < 0.05:
        print(f"    p-value ({result[1]:.4f}) < 0.05 → REJECT H₀")
        print(f"    → Evidence for NON-STATIONARITY")
        conclusion = "NON-STATIONARY"
    else:
        print(f"    p-value ({result[1]:.4f}) ≥ 0.05 → FAIL TO REJECT H₀")
        print(f"    → Evidence for STATIONARITY")
        conclusion = "STATIONARY"

    return result[0], result[1], conclusion

# Run tests
print("\n" + "=" * 60)
print("TESTING PRICES")
print("=" * 60)
adf_stat_p, adf_pval_p, adf_conc_p = run_adf_test(prices, "S&P 500 Prices")
kpss_stat_p, kpss_pval_p, kpss_conc_p = run_kpss_test(prices, "S&P 500 Prices")

print("\n" + "=" * 60)
print("TESTING RETURNS")
print("=" * 60)
adf_stat_r, adf_pval_r, adf_conc_r = run_adf_test(returns, "S&P 500 Returns")
kpss_stat_r, kpss_pval_r, kpss_conc_r = run_kpss_test(returns, "S&P 500 Returns")

# Summary table
print("\n" + "=" * 60)
print("SUMMARY OF RESULTS")
print("=" * 60)

print("\n" + "-" * 70)
print(f"{'Series':<15} {'ADF stat':>10} {'ADF p-val':>12} {'KPSS stat':>12} {'KPSS p-val':>12}")
print("-" * 70)
print(f"{'Prices':<15} {adf_stat_p:>10.4f} {adf_pval_p:>12.4f} {kpss_stat_p:>12.4f} {kpss_pval_p:>12.4f}")
print(f"{'Returns':<15} {adf_stat_r:>10.4f} {adf_pval_r:>12.6f} {kpss_stat_r:>12.4f} {kpss_pval_r:>12.4f}")
print("-" * 70)

# Decision matrix
print("\n" + "=" * 60)
print("DECISION MATRIX")
print("=" * 60)
print("""
                  | KPSS: Fail to Reject | KPSS: Reject H₀
                  | (stationary)         | (non-stationary)
------------------|----------------------|-------------------
ADF: Reject H₀    | STATIONARY           | Trend stationary
(no unit root)    |                      | (need detrending)
------------------|----------------------|-------------------
ADF: Fail to      | Inconclusive         | NON-STATIONARY
Reject H₀         | (more tests needed)  | (unit root)
(unit root)       |                      |
""")

# Apply decision matrix
print("\nAPPLYING TO OUR DATA:")
print("-" * 40)

for name, adf_p, kpss_p in [("Prices", adf_pval_p, kpss_pval_p),
                             ("Returns", adf_pval_r, kpss_pval_r)]:
    adf_reject = adf_p < 0.05
    kpss_reject = kpss_p < 0.05

    if adf_reject and not kpss_reject:
        conclusion = "STATIONARY"
    elif not adf_reject and kpss_reject:
        conclusion = "NON-STATIONARY (unit root)"
    elif adf_reject and kpss_reject:
        conclusion = "Trend stationary"
    else:
        conclusion = "Inconclusive"

    print(f"\n{name}:")
    print(f"  ADF: {'Reject H₀' if adf_reject else 'Fail to reject H₀'} (p={adf_p:.4f})")
    print(f"  KPSS: {'Reject H₀' if kpss_reject else 'Fail to reject H₀'} (p={kpss_p:.4f})")
    print(f"  → CONCLUSION: {conclusion}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Prices with test results
ax1 = axes[0, 0]
ax1.plot(prices.index, prices.values, 'b-', linewidth=0.8)
ax1.set_title(f'S&P 500 Prices\nADF p-val: {adf_pval_p:.4f}, KPSS p-val: {kpss_pval_p:.4f}', fontsize=11)
ax1.set_xlabel('Date')
ax1.set_ylabel('Price ($)')
ax1.grid(True, alpha=0.3)
ax1.text(0.02, 0.98, 'NON-STATIONARY', transform=ax1.transAxes, fontsize=12,
         verticalalignment='top', color='red', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Returns with test results
ax2 = axes[0, 1]
ax2.plot(returns.index, returns.values, 'g-', linewidth=0.5, alpha=0.8)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax2.set_title(f'S&P 500 Returns\nADF p-val: {adf_pval_r:.6f}, KPSS p-val: {kpss_pval_r:.4f}', fontsize=11)
ax2.set_xlabel('Date')
ax2.set_ylabel('Return')
ax2.grid(True, alpha=0.3)
ax2.text(0.02, 0.98, 'STATIONARY', transform=ax2.transAxes, fontsize=12,
         verticalalignment='top', color='green', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ACF of prices
ax3 = axes[1, 0]
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(prices, ax=ax3, lags=30, alpha=0.05)
ax3.set_title('ACF of Prices (slow decay = non-stationary)', fontsize=11)

# ACF of returns
ax4 = axes[1, 1]
plot_acf(returns, ax=ax4, lags=30, alpha=0.05)
ax4.set_title('ACF of Returns (quick decay = stationary)', fontsize=11)

plt.tight_layout()
plt.savefig('../../charts/ch1_python_adf.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("FINAL CONCLUSIONS")
print("=" * 60)
print("""
1. S&P 500 PRICES are NON-STATIONARY
   - ADF fails to reject unit root (p ≈ 0.81)
   - KPSS rejects stationarity (p < 0.05)
   - ACF shows very slow decay
   → Cannot model directly with ARMA

2. S&P 500 RETURNS are STATIONARY
   - ADF strongly rejects unit root (p < 0.001)
   - KPSS fails to reject stationarity (p ≈ 0.1)
   - ACF quickly drops to near zero
   → CAN model with ARMA

3. RECOMMENDATION:
   - Difference prices to get returns: r_t = p_t - p_{t-1}
   - Or use log returns: r_t = log(p_t) - log(p_{t-1})
   - Then apply time series models to returns
""")
