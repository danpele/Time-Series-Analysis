"""
TSA_ch1_unit_root_tests
=======================
Unit Root Tests: ADF and KPSS

This script demonstrates:
- Augmented Dickey-Fuller (ADF) test: H0 = unit root (non-stationary)
- KPSS test: H0 = stationary
- Complementary use of both tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Set random seed
np.random.seed(42)

n = 500

# Generate test series
# 1. Stationary (AR(1) with |phi| < 1)
ar1_stationary = np.zeros(n)
phi = 0.7
for t in range(1, n):
    ar1_stationary[t] = phi * ar1_stationary[t-1] + np.random.normal(0, 1)

# 2. Unit root (random walk)
random_walk = np.cumsum(np.random.normal(0, 1, n))

# 3. Trend stationary
t = np.arange(n)
trend_stationary = 0.05 * t + np.random.normal(0, 1, n)

# 4. Difference stationary (random walk with drift)
rw_drift = 0.1 + np.cumsum(np.random.normal(0, 1, n))

def run_tests(series, name):
    """Run ADF and KPSS tests"""
    # ADF test
    adf_result = adfuller(series, autolag='AIC')
    adf_stat, adf_pvalue = adf_result[0], adf_result[1]

    # KPSS test (with 'c' for level stationarity)
    kpss_result = kpss(series, regression='c', nlags='auto')
    kpss_stat, kpss_pvalue = kpss_result[0], kpss_result[1]

    return {
        'name': name,
        'adf_stat': adf_stat,
        'adf_pval': adf_pvalue,
        'kpss_stat': kpss_stat,
        'kpss_pval': kpss_pvalue
    }

# Run tests
results = [
    run_tests(ar1_stationary, 'AR(1) Stationary'),
    run_tests(random_walk, 'Random Walk'),
    run_tests(trend_stationary, 'Trend Stationary'),
    run_tests(rw_drift, 'RW with Drift')
]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

series_list = [
    (ar1_stationary, 'AR(1) Stationary', 'blue'),
    (random_walk, 'Random Walk (Unit Root)', 'red'),
    (trend_stationary, 'Trend Stationary', 'green'),
    (rw_drift, 'Random Walk with Drift', 'purple')
]

for ax, (series, title, color) in zip(axes.flatten(), series_list):
    ax.plot(series, color=color, linewidth=0.8, alpha=0.8, label=title)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# Add legend outside bottom
fig.legend(['AR(1) Stationary', 'Random Walk', 'Trend Stationary', 'RW with Drift'],
           loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
plt.subplots_adjust(bottom=0.1)

# Set transparent background
fig.patch.set_facecolor('none')
for ax in fig.axes:
    ax.patch.set_facecolor('none')

# Save both PDF and PNG with transparent background
plt.savefig('../../charts/ch1_unit_root_series.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_unit_root_series.png', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_unit_root_series.pdf', transparent=True, bbox_inches='tight')
plt.show()

# Print test results
print("=" * 80)
print("UNIT ROOT TEST RESULTS")
print("=" * 80)
print("\nADF Test: H0 = Series has unit root (non-stationary)")
print("KPSS Test: H0 = Series is stationary")
print("-" * 80)
print(f"{'Series':<20} {'ADF Stat':>10} {'ADF p-val':>10} {'KPSS Stat':>10} {'KPSS p-val':>10} {'Conclusion':>15}")
print("-" * 80)

for r in results:
    # Determine conclusion
    adf_reject = r['adf_pval'] < 0.05  # Reject unit root
    kpss_reject = r['kpss_pval'] < 0.05  # Reject stationarity

    if adf_reject and not kpss_reject:
        conclusion = "STATIONARY"
    elif not adf_reject and kpss_reject:
        conclusion = "UNIT ROOT"
    elif adf_reject and kpss_reject:
        conclusion = "TREND STAT."
    else:
        conclusion = "UNCERTAIN"

    print(f"{r['name']:<20} {r['adf_stat']:>10.3f} {r['adf_pval']:>10.4f} {r['kpss_stat']:>10.3f} {r['kpss_pval']:>10.4f} {conclusion:>15}")

print("\n" + "=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)
print("""
             | KPSS: Fail to Reject | KPSS: Reject
-------------|---------------------|-------------------
ADF: Reject  | STATIONARY          | TREND STATIONARY
ADF: Fail    | UNCERTAIN           | UNIT ROOT

Decision Framework:
1. Both agree -> Clear conclusion
2. Conflicting -> Need more investigation
   - Check for structural breaks
   - Try different lag specifications
   - Consider trend specifications

Practical Tips:
- Always run BOTH tests (they have opposite null hypotheses)
- ADF is more commonly used but has low power
- KPSS is more conservative (rejects more often)
- If unit root detected -> difference the series
- If trend stationary -> detrend or include trend in model
""")
