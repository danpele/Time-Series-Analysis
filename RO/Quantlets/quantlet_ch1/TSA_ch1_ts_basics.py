"""
TSA_ch1_ts_basics
=================
Time Series Basics and Patterns

This script demonstrates:
- Different types of time series patterns
- Trend, seasonality, cycles, and noise
- Visual identification of patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Generate time index
n = 200
t = np.arange(n)

# Create different pattern types
# 1. Trend only
trend_data = 50 + 0.5 * t + np.random.normal(0, 5, n)

# 2. Seasonal only (no trend)
seasonal_data = 100 + 20 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 3, n)

# 3. Trend + Seasonality
trend_seasonal = 50 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 3, n)

# 4. Random walk (no pattern - stochastic trend)
random_walk = np.cumsum(np.random.normal(0, 1, n)) + 100

# 5. Cyclical (longer cycles)
cyclical = 100 + 30 * np.sin(2 * np.pi * t / 50) + np.random.normal(0, 5, n)

# 6. White noise (stationary, no pattern)
white_noise = np.random.normal(100, 10, n)

# Create figure
fig, axes = plt.subplots(3, 2, figsize=(14, 13))

# Plot each pattern
patterns = [
    (trend_data, 'Trend Pattern', 'Linear growth over time', 'blue'),
    (seasonal_data, 'Seasonal Pattern', 'Regular 12-period cycle', 'green'),
    (trend_seasonal, 'Trend + Seasonality', 'Growth with regular cycles', 'purple'),
    (random_walk, 'Random Walk', 'Stochastic trend (unpredictable)', 'red'),
    (cyclical, 'Cyclical Pattern', 'Long irregular cycles', 'orange'),
    (white_noise, 'White Noise', 'Stationary, no pattern', 'brown')
]

for ax, (data, title, description, color) in zip(axes.flatten(), patterns):
    ax.plot(data, color=color, linewidth=0.8, alpha=0.8, label=title)
    ax.set_title(f'{title}\n({description})', fontsize=11)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

    # Add mean line for stationary series
    if 'White Noise' in title:
        ax.axhline(y=np.mean(data), color='red', linestyle='--',
                   label=f'Mean = {np.mean(data):.1f}')

plt.tight_layout()

# Add legend outside bottom
fig.legend(['Trend', 'Seasonal', 'Trend+Seasonal', 'Random Walk', 'Cyclical', 'White Noise'],
           loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=6, frameon=False)
plt.subplots_adjust(bottom=0.08)

# Set transparent background
fig.patch.set_facecolor('none')
for ax in fig.axes:
    ax.patch.set_facecolor('none')

# Save both PDF and PNG with transparent background
plt.savefig('../../charts/ch1_ts_patterns.pdf', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_ts_patterns.png', transparent=True, bbox_inches='tight', dpi=300)
plt.savefig('../../charts/ch1_ts_patterns.pdf', transparent=True, bbox_inches='tight')
plt.show()

# Print pattern identification guide
print("=" * 60)
print("TIME SERIES PATTERN IDENTIFICATION GUIDE")
print("=" * 60)
print("""
Pattern       | Visual Cue                    | Stationarity
--------------|-------------------------------|-------------
Trend         | Upward/downward movement      | Non-stationary
Seasonality   | Regular repeating cycles      | Non-stationary
Cyclical      | Irregular long-term waves     | Non-stationary
Random Walk   | Wandering, no return to mean  | Non-stationary
White Noise   | Random around constant mean   | STATIONARY

Key Questions for Identification:
1. Is there a long-term direction? -> Trend
2. Are there regular, predictable cycles? -> Seasonality
3. Does the series return to its mean? -> If yes, possibly stationary
4. Does variance change over time? -> If yes, non-stationary
""")

# Summary statistics
print("\nSummary Statistics:")
print("-" * 60)
for data, title, _, _ in patterns:
    print(f"{title:20} | Mean: {np.mean(data):7.2f} | Std: {np.std(data):7.2f}")
