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
plt.rcParams['axes.grid'] = True
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
axes[0].set_title(f'US Real GDP Growth Rate ({data_source})', fontweight='bold')
axes[0].set_ylabel('Growth Rate (%)')
axes[0].fill_between(combined.index, 0, combined['GDP_Growth'],
                     where=combined['GDP_Growth'] < 0, color=COLORS['red'], alpha=0.3)

axes[1].plot(combined.index, combined['Unemployment'], color=COLORS['red'], linewidth=1.5)
axes[1].set_title('US Unemployment Rate', fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Rate (%)')

plt.tight_layout()
plt.savefig('charts/ch5_gdp_unemployment.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch5_gdp_unemployment.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch5_gdp_unemployment.pdf")

print("\nAll Chapter 5 charts generated successfully!")
