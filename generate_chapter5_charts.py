"""
Generate charts for Chapter 5: VAR Models and Granger Causality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

COLORS = {
    'blue': '#1A3A6E',
    'red': '#DC3545',
    'green': '#2E7D32',
    'orange': '#E67E22',
    'gray': '#666666'
}

# Create output directory
import os
os.makedirs('charts', exist_ok=True)

#=============================================================================
# Chart 1: Simulated Bivariate VAR
#=============================================================================
print("Generating Chart 1: Simulated VAR...")

np.random.seed(42)
n = 200

# VAR(1) coefficients
A = np.array([[0.7, 0.2],
              [0.1, 0.5]])
c = np.array([0.5, 0.3])

# Generate data
Y = np.zeros((n, 2))
Y[0] = [10, 5]  # Initial values

for t in range(1, n):
    eps = np.random.randn(2) * 0.5
    Y[t] = c + A @ Y[t-1] + eps

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(Y[:, 0], color=COLORS['blue'], linewidth=1.5, label='Y₁')
axes[0].set_title('Simulated VAR(1): Variable Y₁', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Value')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

axes[1].plot(Y[:, 1], color=COLORS['red'], linewidth=1.5, label='Y₂')
axes[1].set_title('Simulated VAR(1): Variable Y₂', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_var_simulation.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_var_simulation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_var_simulation.pdf")

#=============================================================================
# Chart 2: Cross-correlation
#=============================================================================
print("Generating Chart 2: Cross-correlation...")

from scipy.stats import pearsonr

lags = range(-20, 21)
ccf = []
for lag in lags:
    if lag < 0:
        corr, _ = pearsonr(Y[-lag:, 0], Y[:lag, 1])
    elif lag > 0:
        corr, _ = pearsonr(Y[:-lag, 0], Y[lag:, 1])
    else:
        corr, _ = pearsonr(Y[:, 0], Y[:, 1])
    ccf.append(corr)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(lags, ccf, color=COLORS['blue'], alpha=0.7, width=0.8)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(y=1.96/np.sqrt(n), color=COLORS['red'], linestyle='--', alpha=0.7, label='95% CI')
ax.axhline(y=-1.96/np.sqrt(n), color=COLORS['red'], linestyle='--', alpha=0.7)
ax.set_xlabel('Lag', fontsize=11)
ax.set_ylabel('Cross-correlation', fontsize=11)
ax.set_title('Cross-correlation: Y₁ and Y₂', fontweight='bold', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_cross_correlation.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_cross_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_cross_correlation.pdf")

#=============================================================================
# Chart 3: Impulse Response Functions
#=============================================================================
print("Generating Chart 3: IRFs...")

# Fit VAR to simulated data
data = pd.DataFrame(Y, columns=['Y1', 'Y2'])
model = VAR(data)
results = model.fit(maxlags=4, ic='aic')

# Get IRFs
irf = results.irf(periods=20)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Response of Y1 to shock in Y1
axes[0, 0].plot(irf.irfs[:, 0, 0], color=COLORS['blue'], linewidth=2, marker='o', markersize=4)
axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
axes[0, 0].fill_between(range(21), irf.irfs[:, 0, 0] - 1.96*irf.stderr()[:, 0, 0],
                        irf.irfs[:, 0, 0] + 1.96*irf.stderr()[:, 0, 0],
                        color=COLORS['blue'], alpha=0.2)
axes[0, 0].set_title('Response of Y₁ to Y₁ shock', fontweight='bold')
axes[0, 0].set_xlabel('Horizon')

# Response of Y1 to shock in Y2
axes[0, 1].plot(irf.irfs[:, 0, 1], color=COLORS['red'], linewidth=2, marker='o', markersize=4)
axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
axes[0, 1].fill_between(range(21), irf.irfs[:, 0, 1] - 1.96*irf.stderr()[:, 0, 1],
                        irf.irfs[:, 0, 1] + 1.96*irf.stderr()[:, 0, 1],
                        color=COLORS['red'], alpha=0.2)
axes[0, 1].set_title('Response of Y₁ to Y₂ shock', fontweight='bold')
axes[0, 1].set_xlabel('Horizon')

# Response of Y2 to shock in Y1
axes[1, 0].plot(irf.irfs[:, 1, 0], color=COLORS['blue'], linewidth=2, marker='o', markersize=4)
axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
axes[1, 0].fill_between(range(21), irf.irfs[:, 1, 0] - 1.96*irf.stderr()[:, 1, 0],
                        irf.irfs[:, 1, 0] + 1.96*irf.stderr()[:, 1, 0],
                        color=COLORS['blue'], alpha=0.2)
axes[1, 0].set_title('Response of Y₂ to Y₁ shock', fontweight='bold')
axes[1, 0].set_xlabel('Horizon')

# Response of Y2 to shock in Y2
axes[1, 1].plot(irf.irfs[:, 1, 1], color=COLORS['red'], linewidth=2, marker='o', markersize=4)
axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
axes[1, 1].fill_between(range(21), irf.irfs[:, 1, 1] - 1.96*irf.stderr()[:, 1, 1],
                        irf.irfs[:, 1, 1] + 1.96*irf.stderr()[:, 1, 1],
                        color=COLORS['red'], alpha=0.2)
axes[1, 1].set_title('Response of Y₂ to Y₂ shock', fontweight='bold')
axes[1, 1].set_xlabel('Horizon')

plt.suptitle('Impulse Response Functions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_irf.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_irf.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_irf.pdf")

#=============================================================================
# Chart 4: Forecast Error Variance Decomposition
#=============================================================================
print("Generating Chart 4: FEVD...")

fevd = results.fevd(periods=20)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Extract FEVD data - shape is (n_periods, n_vars, n_vars)
# fevd.decomp[h, i, j] = contribution of shock j to variance of variable i at horizon h
horizons = np.arange(fevd.decomp.shape[0])

# FEVD for Y1 (variable 0)
y1_from_y1 = fevd.decomp[:, 0, 0] * 100
y1_from_y2 = fevd.decomp[:, 0, 1] * 100
axes[0].stackplot(horizons, y1_from_y1, y1_from_y2,
                  colors=[COLORS['blue'], COLORS['red']], alpha=0.7,
                  labels=['Y₁ shock', 'Y₂ shock'])
axes[0].set_title('Variance Decomposition: Y₁', fontweight='bold')
axes[0].set_xlabel('Horizon')
axes[0].set_ylabel('Percentage')
axes[0].set_ylim(0, 100)
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# FEVD for Y2 (variable 1)
y2_from_y1 = fevd.decomp[:, 1, 0] * 100
y2_from_y2 = fevd.decomp[:, 1, 1] * 100
axes[1].stackplot(horizons, y2_from_y1, y2_from_y2,
                  colors=[COLORS['blue'], COLORS['red']], alpha=0.7,
                  labels=['Y₁ shock', 'Y₂ shock'])
axes[1].set_title('Variance Decomposition: Y₂', fontweight='bold')
axes[1].set_xlabel('Horizon')
axes[1].set_ylabel('Percentage')
axes[1].set_ylim(0, 100)
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.suptitle('Forecast Error Variance Decomposition', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_fevd.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_fevd.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_fevd.pdf")

#=============================================================================
# Chart 5: Real Data - GDP and Unemployment
#=============================================================================
print("Generating Chart 5: Real data example...")

try:
    import pandas_datareader as pdr

    # Download GDP growth and unemployment
    gdp = pdr.get_data_fred('A191RL1Q225SBEA', start='1990-01-01', end='2024-12-31')  # Real GDP growth
    unemp = pdr.get_data_fred('UNRATE', start='1990-01-01', end='2024-12-31')  # Unemployment rate

    # Resample unemployment to quarterly (use QS to match GDP's quarter-start dates)
    unemp_q = unemp.resample('QS').mean()

    # Align dates by merging on index
    combined = gdp.join(unemp_q, how='inner')
    combined.columns = ['GDP_Growth', 'Unemployment']

    data_source = "FRED"
except Exception as e:
    print(f"Error fetching FRED data: {e}")
    # Simulate realistic data
    np.random.seed(123)
    n = 140
    gdp_growth = 2.5 + np.random.randn(n) * 2
    unemp = 5 + np.cumsum(np.random.randn(n) * 0.3)
    unemp = np.clip(unemp, 3, 12)

    combined = pd.DataFrame({
        'GDP_Growth': gdp_growth,
        'Unemployment': unemp
    }, index=pd.date_range('1990-01-01', periods=n, freq='QE'))

    data_source = "Simulated"

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(combined.index, combined['GDP_Growth'], color=COLORS['blue'], linewidth=1.5)
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[0].set_title('US Real GDP Growth Rate', fontweight='bold')
axes[0].set_ylabel('Growth Rate (%)')
axes[0].fill_between(combined.index, 0, combined['GDP_Growth'],
                     where=combined['GDP_Growth'] < 0, color=COLORS['red'], alpha=0.3)

axes[1].plot(combined.index, combined['Unemployment'], color=COLORS['red'], linewidth=1.5)
axes[1].set_title('US Unemployment Rate', fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Rate (%)')

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_gdp_unemployment.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_gdp_unemployment.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_gdp_unemployment.pdf")

#=============================================================================
# Chart 6: Motivation - Economic Dynamics
#=============================================================================
print("Generating Chart 6: Motivation Economic...")

np.random.seed(42)
n = 120
dates = pd.date_range('2000-01-01', periods=n, freq='QE')

# Simulate correlated macro variables
gdp_growth = 2.5 + np.cumsum(np.random.randn(n) * 0.3) * 0.1
gdp_growth = gdp_growth - np.mean(gdp_growth) + 2.5
inflation = 2.0 + 0.3 * gdp_growth + np.random.randn(n) * 0.5
unemployment = 5.5 - 0.4 * gdp_growth + np.random.randn(n) * 0.3
unemployment = np.clip(unemployment, 3, 10)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axes[0].plot(dates, gdp_growth, color=COLORS['blue'], linewidth=1.5, label='GDP Growth')
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
axes[0].set_ylabel('GDP Growth (%)')
axes[0].set_title('Macroeconomic Dynamics: Interconnected Variables', fontweight='bold', fontsize=12)
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

axes[1].plot(dates, inflation, color=COLORS['red'], linewidth=1.5, label='Inflation')
axes[1].set_ylabel('Inflation (%)')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

axes[2].plot(dates, unemployment, color=COLORS['green'], linewidth=1.5, label='Unemployment')
axes[2].set_ylabel('Unemployment (%)')
axes[2].set_xlabel('Date')
axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_motivation_econ.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_motivation_econ.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_motivation_econ.pdf")

#=============================================================================
# Chart 7: Motivation - Scatter plots
#=============================================================================
print("Generating Chart 7: Motivation Scatter...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# GDP vs Unemployment (Okun's Law)
axes[0].scatter(gdp_growth, unemployment, alpha=0.6, color=COLORS['blue'], s=30)
z = np.polyfit(gdp_growth, unemployment, 1)
p = np.poly1d(z)
x_line = np.linspace(min(gdp_growth), max(gdp_growth), 100)
axes[0].plot(x_line, p(x_line), color=COLORS['red'], linewidth=2, label='Trend')
axes[0].set_xlabel('GDP Growth (%)')
axes[0].set_ylabel('Unemployment (%)')
axes[0].set_title("Okun's Law", fontweight='bold')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

# GDP vs Inflation
axes[1].scatter(gdp_growth, inflation, alpha=0.6, color=COLORS['green'], s=30)
z = np.polyfit(gdp_growth, inflation, 1)
p = np.poly1d(z)
axes[1].plot(x_line, p(x_line), color=COLORS['red'], linewidth=2, label='Trend')
axes[1].set_xlabel('GDP Growth (%)')
axes[1].set_ylabel('Inflation (%)')
axes[1].set_title('GDP vs Inflation', fontweight='bold')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

# Unemployment vs Inflation (Phillips Curve)
axes[2].scatter(unemployment, inflation, alpha=0.6, color=COLORS['orange'], s=30)
z = np.polyfit(unemployment, inflation, 1)
p = np.poly1d(z)
x_line2 = np.linspace(min(unemployment), max(unemployment), 100)
axes[2].plot(x_line2, p(x_line2), color=COLORS['red'], linewidth=2, label='Trend')
axes[2].set_xlabel('Unemployment (%)')
axes[2].set_ylabel('Inflation (%)')
axes[2].set_title('Phillips Curve', fontweight='bold')
axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_motivation_scatter.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_motivation_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_motivation_scatter.pdf")

#=============================================================================
# Chart 8: Motivation - Lead-Lag relationships
#=============================================================================
print("Generating Chart 8: Lead-Lag relationships...")

np.random.seed(123)
n = 100
stock_returns = np.random.randn(n) * 2
unemployment_change = np.zeros(n)
for t in range(4, n):
    unemployment_change[t] = -0.3 * stock_returns[t-4] + np.random.randn() * 0.5

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Time series
ax1 = axes[0]
ax2 = ax1.twinx()
ax1.plot(stock_returns, color=COLORS['blue'], linewidth=1.5, label='Stock Returns')
ax2.plot(unemployment_change, color=COLORS['red'], linewidth=1.5, label='Unemployment Change')
ax1.set_xlabel('Time')
ax1.set_ylabel('Stock Returns (%)', color=COLORS['blue'])
ax2.set_ylabel('Unemployment Change (%)', color=COLORS['red'])
ax1.set_title('Lead-Lag Relationship: Stocks Lead Unemployment', fontweight='bold', fontsize=12)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
           bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

# Cross-correlation
from scipy.stats import pearsonr
lags_range = range(-12, 13)
ccf_vals = []
for lag in lags_range:
    if lag < 0:
        corr, _ = pearsonr(stock_returns[-lag:], unemployment_change[:lag])
    elif lag > 0:
        corr, _ = pearsonr(stock_returns[:-lag], unemployment_change[lag:])
    else:
        corr, _ = pearsonr(stock_returns, unemployment_change)
    ccf_vals.append(corr)

axes[1].bar(lags_range, ccf_vals, color=COLORS['blue'], alpha=0.7)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].axhline(y=1.96/np.sqrt(n), color=COLORS['red'], linestyle='--', alpha=0.7, label='95% CI')
axes[1].axhline(y=-1.96/np.sqrt(n), color=COLORS['red'], linestyle='--', alpha=0.7)
axes[1].axvline(x=4, color=COLORS['green'], linestyle=':', linewidth=2, label='Max correlation at lag 4')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Cross-correlation')
axes[1].set_title('Cross-correlation Function', fontweight='bold')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_motivation_leadlag.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_motivation_leadlag.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_motivation_leadlag.pdf")

#=============================================================================
# Chart 9: Motivation - Univariate limitation
#=============================================================================
print("Generating Chart 9: Univariate limitation...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Univariate approach
axes[0].plot(dates[:60], gdp_growth[:60], color=COLORS['blue'], linewidth=1.5, label='GDP Growth')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('GDP Growth (%)')
axes[0].set_title('Univariate: GDP Only', fontweight='bold')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

# Right: Multivariate approach
ax1 = axes[1]
ax2 = ax1.twinx()
ax1.plot(dates[:60], gdp_growth[:60], color=COLORS['blue'], linewidth=1.5, label='GDP Growth')
ax2.plot(dates[:60], unemployment[:60], color=COLORS['red'], linewidth=1.5, label='Unemployment')
ax1.set_xlabel('Date')
ax1.set_ylabel('GDP Growth (%)', color=COLORS['blue'])
ax2.set_ylabel('Unemployment (%)', color=COLORS['red'])
ax1.set_title('Multivariate: GDP + Unemployment', fontweight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
           bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_motivation_univariate.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_motivation_univariate.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_motivation_univariate.pdf")

#=============================================================================
# Chart 10: Lag Selection
#=============================================================================
print("Generating Chart 10: Lag Selection...")

lags_test = np.arange(1, 9)
# Simulated criteria values
np.random.seed(42)
aic_vals = 5.5 - 0.3 * lags_test + 0.05 * lags_test**2 + np.random.randn(8) * 0.05
bic_vals = 5.6 - 0.25 * lags_test + 0.06 * lags_test**2 + np.random.randn(8) * 0.05
hq_vals = 5.55 - 0.27 * lags_test + 0.055 * lags_test**2 + np.random.randn(8) * 0.05

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lags_test, aic_vals, 'o-', color=COLORS['blue'], linewidth=2, markersize=8, label='AIC')
ax.plot(lags_test, bic_vals, 's-', color=COLORS['red'], linewidth=2, markersize=8, label='BIC')
ax.plot(lags_test, hq_vals, '^-', color=COLORS['green'], linewidth=2, markersize=8, label='HQ')

# Mark minima
aic_min = np.argmin(aic_vals)
bic_min = np.argmin(bic_vals)
ax.scatter([lags_test[aic_min]], [aic_vals[aic_min]], s=200, c=COLORS['blue'], zorder=5, edgecolors='black', linewidth=2)
ax.scatter([lags_test[bic_min]], [bic_vals[bic_min]], s=200, c=COLORS['red'], zorder=5, edgecolors='black', linewidth=2)

ax.set_xlabel('Lag Order (p)', fontsize=11)
ax.set_ylabel('Information Criterion Value', fontsize=11)
ax.set_title('Lag Selection: Information Criteria', fontweight='bold', fontsize=12)
ax.set_xticks(lags_test)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_lag_selection.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_lag_selection.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_lag_selection.pdf")

#=============================================================================
# Chart 11: Stability Roots
#=============================================================================
print("Generating Chart 11: Stability Roots...")

fig, ax = plt.subplots(figsize=(8, 8))

# Draw unit circle
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Unit Circle')
ax.fill(np.cos(theta), np.sin(theta), alpha=0.1, color='green')

# Eigenvalues from example VAR
eigenvalues = [0.65 + 0.132j, 0.65 - 0.132j, 0.4, 0.2]
for i, ev in enumerate(eigenvalues):
    ax.scatter(np.real(ev), np.imag(ev), s=150, c=COLORS['blue'], zorder=5, edgecolors='black')
    ax.annotate(f'λ{i+1}', (np.real(ev)+0.05, np.imag(ev)+0.05), fontsize=10)

ax.axhline(y=0, color='gray', linewidth=0.5)
ax.axvline(x=0, color='gray', linewidth=0.5)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('Real Part', fontsize=11)
ax.set_ylabel('Imaginary Part', fontsize=11)
ax.set_title('VAR Stability: Eigenvalues Inside Unit Circle', fontweight='bold', fontsize=12)
ax.set_aspect('equal')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_stability_roots.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_stability_roots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_stability_roots.pdf")

#=============================================================================
# Chart 12: Structural IRF
#=============================================================================
print("Generating Chart 12: Structural IRF...")

# Simulate orthogonalized IRF
horizons = np.arange(21)
irf_11 = np.exp(-horizons * 0.15) * 1.0
irf_12 = np.exp(-horizons * 0.12) * 0.4 * (1 - np.exp(-horizons * 0.3))
irf_21 = -np.exp(-horizons * 0.1) * 0.3 * (1 - np.exp(-horizons * 0.2))
irf_22 = np.exp(-horizons * 0.18) * 0.8

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(horizons, irf_11, color=COLORS['blue'], linewidth=2, marker='o', markersize=4)
axes[0, 0].fill_between(horizons, irf_11 - 0.15, irf_11 + 0.15, color=COLORS['blue'], alpha=0.2)
axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
axes[0, 0].set_title('Response of Y₁ to Structural Shock 1', fontweight='bold')
axes[0, 0].set_xlabel('Horizon')

axes[0, 1].plot(horizons, irf_12, color=COLORS['red'], linewidth=2, marker='o', markersize=4)
axes[0, 1].fill_between(horizons, irf_12 - 0.1, irf_12 + 0.1, color=COLORS['red'], alpha=0.2)
axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
axes[0, 1].set_title('Response of Y₁ to Structural Shock 2', fontweight='bold')
axes[0, 1].set_xlabel('Horizon')

axes[1, 0].plot(horizons, irf_21, color=COLORS['blue'], linewidth=2, marker='o', markersize=4)
axes[1, 0].fill_between(horizons, irf_21 - 0.1, irf_21 + 0.1, color=COLORS['blue'], alpha=0.2)
axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
axes[1, 0].set_title('Response of Y₂ to Structural Shock 1', fontweight='bold')
axes[1, 0].set_xlabel('Horizon')

axes[1, 1].plot(horizons, irf_22, color=COLORS['red'], linewidth=2, marker='o', markersize=4)
axes[1, 1].fill_between(horizons, irf_22 - 0.12, irf_22 + 0.12, color=COLORS['red'], alpha=0.2)
axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
axes[1, 1].set_title('Response of Y₂ to Structural Shock 2', fontweight='bold')
axes[1, 1].set_xlabel('Horizon')

plt.suptitle('Structural Impulse Response Functions (Cholesky)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_structural_irf.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_structural_irf.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_structural_irf.pdf")

#=============================================================================
# Chart 13: VAR Forecast
#=============================================================================
print("Generating Chart 13: VAR Forecast...")

np.random.seed(42)
n_obs = 80
n_fcst = 20

# Generate VAR data
Y_hist = np.zeros((n_obs, 2))
Y_hist[0] = [10, 5]
for t in range(1, n_obs):
    eps = np.random.randn(2) * 0.5
    Y_hist[t] = c + A @ Y_hist[t-1] + eps

# Forecast
Y_fcst = np.zeros((n_fcst, 2))
Y_fcst[0] = c + A @ Y_hist[-1]
for t in range(1, n_fcst):
    Y_fcst[t] = c + A @ Y_fcst[t-1]

# Confidence bands (widening)
ci_width = np.sqrt(np.arange(1, n_fcst + 1)) * 0.4

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Y1
time_hist = np.arange(n_obs)
time_fcst = np.arange(n_obs - 1, n_obs + n_fcst)
axes[0].plot(time_hist, Y_hist[:, 0], color=COLORS['blue'], linewidth=1.5, label='Observed')
axes[0].plot(time_fcst, np.concatenate([[Y_hist[-1, 0]], Y_fcst[:, 0]]),
             color=COLORS['red'], linewidth=2, linestyle='--', label='Forecast')
axes[0].fill_between(time_fcst[1:], Y_fcst[:, 0] - 1.96*ci_width, Y_fcst[:, 0] + 1.96*ci_width,
                     color=COLORS['red'], alpha=0.2, label='95% CI')
axes[0].axvline(x=n_obs-1, color='gray', linestyle=':', alpha=0.7)
axes[0].set_ylabel('Y₁')
axes[0].set_title('VAR Forecast: Variable Y₁', fontweight='bold')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

# Y2
axes[1].plot(time_hist, Y_hist[:, 1], color=COLORS['blue'], linewidth=1.5, label='Observed')
axes[1].plot(time_fcst, np.concatenate([[Y_hist[-1, 1]], Y_fcst[:, 1]]),
             color=COLORS['red'], linewidth=2, linestyle='--', label='Forecast')
axes[1].fill_between(time_fcst[1:], Y_fcst[:, 1] - 1.96*ci_width*0.8, Y_fcst[:, 1] + 1.96*ci_width*0.8,
                     color=COLORS['red'], alpha=0.2, label='95% CI')
axes[1].axvline(x=n_obs-1, color='gray', linestyle=':', alpha=0.7)
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Y₂')
axes[1].set_title('VAR Forecast: Variable Y₂', fontweight='bold')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_var_forecast.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_var_forecast.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_var_forecast.pdf")

#=============================================================================
# Chart 14: VAR Results Summary
#=============================================================================
print("Generating Chart 14: VAR Results...")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Coefficient matrix visualization
coef_matrix = np.array([[0.7, 0.2], [-0.1, 0.6]])
im = axes[0].imshow(coef_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Y₁,t-1', 'Y₂,t-1'])
axes[0].set_yticklabels(['Y₁,t', 'Y₂,t'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f'{coef_matrix[i, j]:.2f}', ha='center', va='center',
                    color='white' if abs(coef_matrix[i, j]) > 0.5 else 'black', fontsize=14)
axes[0].set_title('VAR(1) Coefficient Matrix', fontweight='bold')
plt.colorbar(im, ax=axes[0], shrink=0.8)

# Model fit statistics
stats_names = ['Log-Lik', 'AIC', 'BIC', 'HQ']
stats_vals = [-245.3, 5.12, 5.28, 5.18]
colors = [COLORS['blue'], COLORS['red'], COLORS['green'], COLORS['orange']]
bars = axes[1].bar(stats_names, stats_vals, color=colors, alpha=0.7)
axes[1].set_ylabel('Value')
axes[1].set_title('Model Fit Statistics', fontweight='bold')
axes[1].axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_var_results.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_var_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_var_results.pdf")

#=============================================================================
# Chart 15: Diagnostics
#=============================================================================
print("Generating Chart 15: Diagnostics...")

np.random.seed(42)
residuals = np.random.randn(100)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, ax=axes[0, 0], lags=20, alpha=0.05)
axes[0, 0].set_title('ACF of Residuals', fontweight='bold')

# Histogram
axes[0, 1].hist(residuals, bins=20, color=COLORS['blue'], alpha=0.7, density=True, edgecolor='white')
x = np.linspace(-4, 4, 100)
from scipy.stats import norm
axes[0, 1].plot(x, norm.pdf(x), color=COLORS['red'], linewidth=2, label='Normal')
axes[0, 1].set_title('Residual Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

# Q-Q plot
from scipy.stats import probplot
probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
axes[1, 0].get_lines()[0].set_color(COLORS['blue'])
axes[1, 0].get_lines()[1].set_color(COLORS['red'])

# Residuals over time
axes[1, 1].plot(residuals, color=COLORS['blue'], linewidth=1)
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].axhline(y=2, color=COLORS['red'], linestyle=':', alpha=0.7)
axes[1, 1].axhline(y=-2, color=COLORS['red'], linestyle=':', alpha=0.7)
axes[1, 1].set_title('Residuals Over Time', fontweight='bold')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Residual')

plt.suptitle('VAR Diagnostic Plots', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_diagnostics.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_diagnostics.pdf")

#=============================================================================
# Chart 16: Historical Decomposition
#=============================================================================
print("Generating Chart 16: Historical Decomposition...")

np.random.seed(42)
n = 100
time = np.arange(n)

# Simulated contributions
shock1_contrib = np.cumsum(np.random.randn(n) * 0.3)
shock2_contrib = np.cumsum(np.random.randn(n) * 0.2)
actual = shock1_contrib + shock2_contrib

fig, ax = plt.subplots(figsize=(12, 6))

ax.fill_between(time, 0, shock1_contrib, alpha=0.6, color=COLORS['blue'], label='Shock 1 Contribution')
ax.fill_between(time, shock1_contrib, shock1_contrib + shock2_contrib, alpha=0.6,
                color=COLORS['red'], label='Shock 2 Contribution')
ax.plot(time, actual, 'k-', linewidth=2, label='Actual')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Historical Decomposition of Y₁', fontweight='bold', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_historical_decomp.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_historical_decomp.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_historical_decomp.pdf")

#=============================================================================
# Chart 17: Monetary Policy IRF
#=============================================================================
print("Generating Chart 17: Monetary Policy IRF...")

horizons = np.arange(21)

# Simulate monetary policy transmission
# Shock: Interest rate increase
ir_shock = np.exp(-horizons * 0.15) * 1.0
output_response = -0.5 * (1 - np.exp(-horizons * 0.2)) * np.exp(-horizons * 0.08)
inflation_response = -0.3 * (1 - np.exp(-horizons * 0.15)) * np.exp(-horizons * 0.05)

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Interest rate shock
axes[0].plot(horizons, ir_shock, color=COLORS['blue'], linewidth=2, marker='o', markersize=4)
axes[0].fill_between(horizons, ir_shock - 0.15, ir_shock + 0.15, color=COLORS['blue'], alpha=0.2)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_title('Interest Rate Shock', fontweight='bold')
axes[0].set_xlabel('Quarters')
axes[0].set_ylabel('Percentage Points')

# Output response
axes[1].plot(horizons, output_response, color=COLORS['red'], linewidth=2, marker='o', markersize=4)
axes[1].fill_between(horizons, output_response - 0.1, output_response + 0.1, color=COLORS['red'], alpha=0.2)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_title('Output Gap Response', fontweight='bold')
axes[1].set_xlabel('Quarters')
axes[1].set_ylabel('Percentage Points')

# Inflation response
axes[2].plot(horizons, inflation_response, color=COLORS['green'], linewidth=2, marker='o', markersize=4)
axes[2].fill_between(horizons, inflation_response - 0.08, inflation_response + 0.08, color=COLORS['green'], alpha=0.2)
axes[2].axhline(y=0, color='black', linewidth=0.5)
axes[2].set_title('Inflation Response', fontweight='bold')
axes[2].set_xlabel('Quarters')
axes[2].set_ylabel('Percentage Points')

plt.suptitle('Monetary Policy Transmission: Response to 1pp Interest Rate Shock',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch5_monetary_irf.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_monetary_irf.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_monetary_irf.pdf")

print("\nAll Chapter 5 charts generated successfully!")
