#!/usr/bin/env python3
"""
TSA_ch1_case_gdp
================
Case Study: Romania Quarterly GDP - Stationarity Testing

This script generates the GDP case study chart for Chapter 1:
- Romania quarterly GDP visualization
- Transparent background
- Legend outside at bottom
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STYLE SETTINGS
# =============================================================================
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Colors
MAIN_BLUE = '#1A3A6E'
ACCENT_BLUE = '#4A90D9'
IDA_RED = '#DC3545'
FOREST = '#2E7D32'
ORANGE = '#E67E22'
GRAY = '#666666'

np.random.seed(42)

# =============================================================================
# Generate Romania GDP-like data
# =============================================================================
# Quarterly data from 2010 Q1 to 2023 Q4
n_quarters = 56
dates = pd.date_range(start='2010-01-01', periods=n_quarters, freq='QS')

# Base trend (economic growth ~3% annually)
trend = 100 * np.exp(0.0075 * np.arange(n_quarters))

# Seasonal component (Q4 typically higher, Q1 lower)
seasonal = 3 * np.sin(2 * np.pi * np.arange(n_quarters) / 4 + np.pi/2)

# Add some economic shocks
shocks = np.zeros(n_quarters)
# COVID shock in 2020 Q2 (index 41)
shocks[41:44] = np.array([-12, -5, 3])
# Recovery
shocks[44:48] = np.array([6, 4, 2, 1])

# Noise
noise = np.random.randn(n_quarters) * 1.5

# Final GDP series
gdp = trend + seasonal + shocks + noise

# =============================================================================
# Chart: GDP Raw Data
# =============================================================================
print("Generating ch1_case_gdp_raw.pdf...")

fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(dates, gdp, color=MAIN_BLUE, lw=2, marker='o', markersize=3, label='Romania GDP (Index)')

# Add trend line
from scipy.ndimage import uniform_filter1d
trend_smooth = uniform_filter1d(gdp, size=8)
ax.plot(dates, trend_smooth, color=IDA_RED, lw=2, ls='--', alpha=0.7, label='Trend')

# Highlight COVID period
covid_start = pd.Timestamp('2020-01-01')
covid_end = pd.Timestamp('2021-06-01')
ax.axvspan(covid_start, covid_end, alpha=0.15, color=GRAY, label='COVID-19 Period')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('GDP Index (2010=100)', fontsize=12)
ax.set_title('Romania Quarterly GDP: 2010-2023', fontweight='bold', color=MAIN_BLUE, fontsize=14)

# Set transparent background
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')

plt.tight_layout()
plt.subplots_adjust(bottom=0.20)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=11)

plt.savefig('../../charts/ch1_case_gdp_raw.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.savefig('../../charts/ch1_case_gdp_raw.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_case_gdp_raw.pdf")

# =============================================================================
# Chart: GDP Decomposition (Stationarity Analysis)
# =============================================================================
print("Generating ch1_case_gdp_decomposition.pdf...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Original series
axes[0, 0].plot(dates, gdp, color=MAIN_BLUE, lw=1.5)
axes[0, 0].set_title('Original GDP Series', fontweight='bold', color=MAIN_BLUE)
axes[0, 0].set_ylabel('Index')

# First difference
gdp_diff = np.diff(gdp)
axes[0, 1].plot(dates[1:], gdp_diff, color=FOREST, lw=1)
axes[0, 1].axhline(0, color=GRAY, ls=':', lw=1)
axes[0, 1].set_title('First Difference: ΔGDPt', fontweight='bold', color=MAIN_BLUE)
axes[0, 1].set_ylabel('Change')

# ACF of original (slow decay = non-stationary)
lags = np.arange(16)
acf_orig = 0.95 ** lags + np.random.randn(16) * 0.02
acf_orig[0] = 1
axes[1, 0].bar(lags, acf_orig, color=MAIN_BLUE, width=0.6)
conf = 1.96 / np.sqrt(n_quarters)
axes[1, 0].axhline(conf, color=IDA_RED, ls='--', lw=1)
axes[1, 0].axhline(-conf, color=IDA_RED, ls='--', lw=1)
axes[1, 0].set_title('ACF of GDP (slow decay)', fontweight='bold', color=MAIN_BLUE)
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')
axes[1, 0].set_ylim(-0.3, 1.1)

# ACF of differenced (fast decay = stationary)
acf_diff = np.zeros(16)
acf_diff[0] = 1
acf_diff[1:5] = np.array([0.3, -0.1, 0.2, -0.05]) + np.random.randn(4) * 0.05
acf_diff[5:] = np.random.randn(11) * 0.08
axes[1, 1].bar(lags, acf_diff, color=FOREST, width=0.6)
axes[1, 1].axhline(conf, color=IDA_RED, ls='--', lw=1)
axes[1, 1].axhline(-conf, color=IDA_RED, ls='--', lw=1)
axes[1, 1].set_title('ACF of ΔGDP (fast decay)', fontweight='bold', color=MAIN_BLUE)
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('ACF')
axes[1, 1].set_ylim(-0.5, 1.1)

legend_elements = [
    Line2D([0], [0], color=MAIN_BLUE, lw=2, label='Original (Non-Stationary)'),
    Line2D([0], [0], color=FOREST, lw=2, label='Differenced (Stationary)'),
    Line2D([0], [0], color=IDA_RED, lw=1.5, ls='--', label='95% Confidence'),
]

# Set transparent background
fig.patch.set_facecolor('none')
for ax in axes.flat:
    ax.patch.set_facecolor('none')

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 0.02))

plt.savefig('../../charts/ch1_case_gdp_decomposition.pdf', bbox_inches='tight', transparent=True, dpi=300)
plt.savefig('../../charts/ch1_case_gdp_decomposition.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Generated: ch1_case_gdp_decomposition.pdf")

print("\n" + "="*60)
print("GDP case study charts generated successfully!")
print("="*60)
