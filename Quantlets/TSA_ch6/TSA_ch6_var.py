"""
TSA_ch6_var
===========
VAR Models and Granger Causality

This script demonstrates:
- Vector Autoregression (VAR) models
- Granger causality testing
- Impulse Response Functions (IRF)
- Forecast Error Variance Decomposition (FEVD)
- VAR forecasting

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# Chart style settings - Nature journal quality
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75

def save_fig(name):
    """Save figure with transparent background."""
    plt.savefig(f'{name}.pdf', bbox_inches='tight', transparent=True, dpi=300)
    plt.savefig(f'{name}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf")

print("=" * 70)
print("VAR MODELS AND GRANGER CAUSALITY")
print("=" * 70)

# =============================================================================
# 1. Generate Multivariate Time Series
# =============================================================================
np.random.seed(42)
n = 500

print("\n1. SIMULATING VAR(2) PROCESS")
print("-" * 40)

# VAR(2) coefficients
# Y1_t = 0.5*Y1_{t-1} + 0.1*Y2_{t-1} - 0.2*Y1_{t-2} + e1_t
# Y2_t = 0.3*Y1_{t-1} + 0.4*Y2_{t-1} + 0.1*Y1_{t-2} - 0.1*Y2_{t-2} + e2_t

A1 = np.array([[0.5, 0.1],
               [0.3, 0.4]])
A2 = np.array([[-0.2, 0.0],
               [0.1, -0.1]])

print("   Coefficient matrices:")
print(f"   A1 = {A1.tolist()}")
print(f"   A2 = {A2.tolist()}")

# Simulate
Y = np.zeros((n, 2))
errors = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], n)

for t in range(2, n):
    Y[t] = A1 @ Y[t-1] + A2 @ Y[t-2] + errors[t]

dates = pd.date_range('2000-01-01', periods=n, freq='M')
df = pd.DataFrame(Y, columns=['GDP_Growth', 'Unemployment'], index=dates)

print(f"   Simulated {n} observations of 2 variables")
print(f"   Variables: GDP Growth, Unemployment")

# =============================================================================
# 2. Data Visualization
# =============================================================================
print("\n2. DATA VISUALIZATION")
print("-" * 40)

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axes[0].plot(df.index, df['GDP_Growth'], color='#1A3A6E', linewidth=1)
axes[0].set_ylabel('GDP Growth')
axes[0].set_title('Multivariate Time Series', fontweight='bold')
axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

axes[1].plot(df.index, df['Unemployment'], color='#DC3545', linewidth=1)
axes[1].set_ylabel('Unemployment')
axes[1].set_xlabel('Date')
axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('ch6_data.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch6_data.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch6_data.pdf")

# =============================================================================
# 3. Stationarity Testing
# =============================================================================
print("\n3. STATIONARITY TESTING")
print("-" * 40)

for col in df.columns:
    adf_result = adfuller(df[col])
    status = "Stationary" if adf_result[1] < 0.05 else "Non-stationary"
    print(f"   {col}:")
    print(f"     ADF statistic: {adf_result[0]:.4f}")
    print(f"     p-value: {adf_result[1]:.4f} ({status})")

# =============================================================================
# 4. VAR Model Estimation
# =============================================================================
print("\n4. VAR MODEL ESTIMATION")
print("-" * 40)

# Fit VAR model
model = VAR(df)

# Lag order selection
lag_order = model.select_order(maxlags=8)
print("\n   Lag Order Selection:")
print(f"   {'Lag':>4} {'AIC':>12} {'BIC':>12} {'HQIC':>12}")
print("   " + "-" * 44)
for i in range(1, 9):
    print(f"   {i:>4} {lag_order.ics['aic'][i]:>12.2f} {lag_order.ics['bic'][i]:>12.2f} {lag_order.ics['hqic'][i]:>12.2f}")

optimal_lag = lag_order.selected_orders['aic']
print(f"\n   Optimal lag (AIC): {optimal_lag}")

# Fit VAR with optimal lag
results = model.fit(optimal_lag)

print(f"\n   VAR({optimal_lag}) Estimation Results:")
print(f"   AIC: {results.aic:.2f}")
print(f"   BIC: {results.bic:.2f}")

# =============================================================================
# 5. Granger Causality Tests
# =============================================================================
print("\n5. GRANGER CAUSALITY TESTS")
print("-" * 40)

# Test: Does Unemployment Granger-cause GDP Growth?
print("\n   H0: Unemployment does NOT Granger-cause GDP Growth")
gc_test1 = grangercausalitytests(df[['GDP_Growth', 'Unemployment']], maxlag=4, verbose=False)
for lag in [1, 2, 3, 4]:
    f_stat = gc_test1[lag][0]['ssr_ftest'][0]
    p_val = gc_test1[lag][0]['ssr_ftest'][1]
    print(f"   Lag {lag}: F = {f_stat:.3f}, p-value = {p_val:.4f}")

# Test: Does GDP Growth Granger-cause Unemployment?
print("\n   H0: GDP Growth does NOT Granger-cause Unemployment")
gc_test2 = grangercausalitytests(df[['Unemployment', 'GDP_Growth']], maxlag=4, verbose=False)
for lag in [1, 2, 3, 4]:
    f_stat = gc_test2[lag][0]['ssr_ftest'][0]
    p_val = gc_test2[lag][0]['ssr_ftest'][1]
    print(f"   Lag {lag}: F = {f_stat:.3f}, p-value = {p_val:.4f}")

# =============================================================================
# 6. Impulse Response Functions
# =============================================================================
print("\n6. IMPULSE RESPONSE FUNCTIONS")
print("-" * 40)

irf = results.irf(periods=20)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Number of periods in IRF
n_periods = len(irf.irfs[:, 0, 0])

# Response of GDP to GDP shock
axes[0, 0].plot(irf.irfs[:, 0, 0], color='#1A3A6E', linewidth=2)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[0, 0].fill_between(range(n_periods), irf.irfs[:, 0, 0] - 1.96*irf.stderr()[:, 0, 0],
                        irf.irfs[:, 0, 0] + 1.96*irf.stderr()[:, 0, 0], alpha=0.2, color='#1A3A6E')
axes[0, 0].set_title('GDP → GDP', fontweight='bold')
axes[0, 0].set_xlabel('Periods')

# Response of GDP to Unemployment shock
axes[0, 1].plot(irf.irfs[:, 0, 1], color='#DC3545', linewidth=2)
axes[0, 1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[0, 1].fill_between(range(n_periods), irf.irfs[:, 0, 1] - 1.96*irf.stderr()[:, 0, 1],
                        irf.irfs[:, 0, 1] + 1.96*irf.stderr()[:, 0, 1], alpha=0.2, color='#DC3545')
axes[0, 1].set_title('Unemployment → GDP', fontweight='bold')
axes[0, 1].set_xlabel('Periods')

# Response of Unemployment to GDP shock
axes[1, 0].plot(irf.irfs[:, 1, 0], color='#1A3A6E', linewidth=2)
axes[1, 0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[1, 0].fill_between(range(n_periods), irf.irfs[:, 1, 0] - 1.96*irf.stderr()[:, 1, 0],
                        irf.irfs[:, 1, 0] + 1.96*irf.stderr()[:, 1, 0], alpha=0.2, color='#1A3A6E')
axes[1, 0].set_title('GDP → Unemployment', fontweight='bold')
axes[1, 0].set_xlabel('Periods')

# Response of Unemployment to Unemployment shock
axes[1, 1].plot(irf.irfs[:, 1, 1], color='#DC3545', linewidth=2)
axes[1, 1].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[1, 1].fill_between(range(n_periods), irf.irfs[:, 1, 1] - 1.96*irf.stderr()[:, 1, 1],
                        irf.irfs[:, 1, 1] + 1.96*irf.stderr()[:, 1, 1], alpha=0.2, color='#DC3545')
axes[1, 1].set_title('Unemployment → Unemployment', fontweight='bold')
axes[1, 1].set_xlabel('Periods')

plt.suptitle('Impulse Response Functions (with 95% CI)', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('ch6_irf.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch6_irf.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch6_irf.pdf")

# =============================================================================
# 7. Forecast Error Variance Decomposition
# =============================================================================
print("\n7. FORECAST ERROR VARIANCE DECOMPOSITION")
print("-" * 40)

fevd = results.fevd(20)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Get FEVD data - shape is (periods, variables, shocks)
n_fevd_periods = fevd.decomp.shape[0]
periods = range(1, n_fevd_periods + 1)

# FEVD for GDP Growth
axes[0].stackplot(periods, fevd.decomp[:, 0, 0], fevd.decomp[:, 0, 1],
                  labels=['GDP', 'Unemployment'],
                  colors=['#1A3A6E', '#DC3545'], alpha=0.8)
axes[0].set_xlabel('Forecast Horizon')
axes[0].set_ylabel('Proportion')
axes[0].set_title('FEVD: GDP Growth', fontweight='bold')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
axes[0].set_ylim(0, 1)

# FEVD for Unemployment
axes[1].stackplot(periods, fevd.decomp[:, 1, 0], fevd.decomp[:, 1, 1],
                  labels=['GDP', 'Unemployment'],
                  colors=['#1A3A6E', '#DC3545'], alpha=0.8)
axes[1].set_xlabel('Forecast Horizon')
axes[1].set_ylabel('Proportion')
axes[1].set_title('FEVD: Unemployment', fontweight='bold')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('ch6_fevd.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch6_fevd.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch6_fevd.pdf")

horizon_idx = min(9, n_fevd_periods - 1)  # Use horizon 10 or last available
print(f"\n   FEVD at horizon {horizon_idx + 1}:")
print(f"   GDP variance explained by:")
print(f"     - GDP shocks: {fevd.decomp[horizon_idx, 0, 0]*100:.1f}%")
print(f"     - Unemployment shocks: {fevd.decomp[horizon_idx, 0, 1]*100:.1f}%")
print(f"   Unemployment variance explained by:")
print(f"     - GDP shocks: {fevd.decomp[horizon_idx, 1, 0]*100:.1f}%")
print(f"     - Unemployment shocks: {fevd.decomp[horizon_idx, 1, 1]*100:.1f}%")

# =============================================================================
# 8. VAR Forecasting
# =============================================================================
print("\n8. VAR FORECASTING")
print("-" * 40)

forecast_steps = 24
forecast = results.forecast(df.values[-optimal_lag:], steps=forecast_steps)
forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1),
                                periods=forecast_steps, freq='M')

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Visual separator between historical and forecast
split_point = df.index[-1]

# GDP Growth
axes[0].plot(df.index[-100:], df['GDP_Growth'].iloc[-100:], color='#1A3A6E', linewidth=1.5, label='Historical')
axes[0].plot(forecast_dates, forecast[:, 0], color='#DC3545', linewidth=2, linestyle='--', label='Forecast')
axes[0].axvline(x=split_point, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
y_pos = axes[0].get_ylim()[1] - 0.05 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0])
axes[0].text(split_point, y_pos, '  Forecast ', fontsize=9, ha='left', va='top',
             color='black', fontweight='bold', alpha=0.8)
axes[0].set_ylabel('GDP Growth')
axes[0].set_title('VAR Forecast', fontweight='bold')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Unemployment
axes[1].plot(df.index[-100:], df['Unemployment'].iloc[-100:], color='#1A3A6E', linewidth=1.5, label='Historical')
axes[1].plot(forecast_dates, forecast[:, 1], color='#DC3545', linewidth=2, linestyle='--', label='Forecast')
axes[1].axvline(x=split_point, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].set_ylabel('Unemployment')
axes[1].set_xlabel('Date')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
plt.savefig('ch6_forecast.pdf', dpi=150, bbox_inches='tight')
plt.savefig('ch6_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved: ch6_forecast.pdf")

print("\n" + "=" * 70)
print("VAR ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch6_data.pdf: Multivariate time series")
print("  - ch6_irf.pdf: Impulse response functions")
print("  - ch6_fevd.pdf: Forecast error variance decomposition")
print("  - ch6_forecast.pdf: VAR forecasts")
