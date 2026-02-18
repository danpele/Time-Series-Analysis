#!/usr/bin/env python3
"""
Generate chart showing transformation sequence: Prices -> Log -> Returns
With transparent background, legend outside bottom, Quantlet style
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style - Quantlet compatible
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'

# Generate realistic price data with increasing variance
np.random.seed(42)
n = 500
returns_true = np.random.normal(0.0005, 0.015, n)  # Daily returns ~1.5% vol
log_prices = np.cumsum(returns_true) + np.log(100)  # Start at price 100
prices = np.exp(log_prices)
log_returns = np.diff(log_prices)

# Create figure with 3 panels - transparent background
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
fig.patch.set_alpha(0.0)

# Colors
color_price = '#1a3a6e'
color_log = '#2e7d32'
color_ret = '#c62828'

# Panel 1: Raw Prices
ax1 = axes[0]
ax1.set_facecolor('none')
line1, = ax1.plot(prices, color=color_price, linewidth=1.2, label='Prices $P_t$')
ax1.axhline(y=np.mean(prices), color='red', linestyle='--', linewidth=1, alpha=0.7)
ax1.fill_between(range(len(prices)), prices.min(), prices.max(), alpha=0.08, color=color_price)
ax1.set_title('1. Prices $P_t$ (non-stationary, increasing variance)', fontsize=11, fontweight='bold', loc='left')
ax1.set_ylabel('Price', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, len(prices))
ax1.text(0.98, 0.92, '✗ Trend\n✗ Increasing variance', transform=ax1.transAxes,
         ha='right', va='top', fontsize=9, color='red', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red'))

# Panel 2: Log Prices
ax2 = axes[1]
ax2.set_facecolor('none')
line2, = ax2.plot(log_prices, color=color_log, linewidth=1.2, label='Log-Prices $\\log(P_t)$')
ax2.axhline(y=np.mean(log_prices), color='red', linestyle='--', linewidth=1, alpha=0.7)
ax2.set_title('2. Log-Prices $\\log(P_t)$ (non-stationary, stabilized variance)', fontsize=11, fontweight='bold', loc='left')
ax2.set_ylabel('Log(Price)', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, len(log_prices))
ax2.text(0.98, 0.92, '✗ Persistent trend\n✓ Stabilized variance', transform=ax2.transAxes,
         ha='right', va='top', fontsize=9, color='orange', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='orange'))

# Panel 3: Log Returns
ax3 = axes[2]
ax3.set_facecolor('none')
line3, = ax3.plot(log_returns, color=color_ret, linewidth=0.8, alpha=0.8, label='Log-Returns $r_t = \\Delta\\log(P_t)$')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.axhline(y=np.mean(log_returns), color='blue', linestyle='--', linewidth=1, alpha=0.7)
band_upper = ax3.axhline(y=2*np.std(log_returns), color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax3.axhline(y=-2*np.std(log_returns), color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax3.fill_between(range(len(log_returns)), -2*np.std(log_returns), 2*np.std(log_returns),
                  alpha=0.12, color='gray')
ax3.set_title('3. Log-Returns $r_t = \\Delta\\log(P_t)$ (STATIONARY!)', fontsize=11, fontweight='bold', loc='left')
ax3.set_ylabel('Return', fontsize=10)
ax3.set_xlabel('Time (days)', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, len(log_returns))
ax3.text(0.98, 0.92, '✓ Mean ≈ 0\n✓ Constant variance\n✓ STATIONARY!', transform=ax3.transAxes,
         ha='right', va='top', fontsize=9, color='green', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='green'))

# Create legend outside at bottom - transparent background
fig.legend([line1, line2, line3],
           ['Prices $P_t$ (non-stationary)',
            'Log-Prices $\\log(P_t)$ (stabilized variance)',
            'Log-Returns $r_t = \\Delta\\log(P_t)$ (stationary)'],
           loc='lower center', ncol=3, fontsize=10, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

# Add transformation arrows on the left
fig.text(0.015, 0.67, '→ log( )', fontsize=11, fontweight='bold', color=color_log, rotation=0,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
fig.text(0.015, 0.35, '→ Δ', fontsize=11, fontweight='bold', color=color_ret, rotation=0,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)  # Make room for legend

# Save with transparent background - RO version uses separate filename
plt.savefig('../../charts/ch1_transform_sequence_ro.pdf', dpi=300, bbox_inches='tight',
            transparent=True, edgecolor='none')
plt.savefig('../../charts/ch1_transform_sequence_ro.png', dpi=150, bbox_inches='tight',
            transparent=True, edgecolor='none')
print("Chart saved: ch1_transform_sequence_ro.pdf/.png (transparent, legend bottom)")
plt.close()
