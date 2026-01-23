"""
Generate charts for Chapter 8: Modern Extensions (ARFIMA, ML, LSTM)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from statsmodels.tsa.stattools import acf
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
plt.rcParams['legend.frameon'] = False

COLORS = {
    'blue': '#1A3A6E',
    'red': '#DC3545',
    'green': '#2E7D32',
    'orange': '#E67E22',
    'gray': '#666666',
    'light_blue': '#5B8BD4'
}

# Create output directory
import os
os.makedirs('charts', exist_ok=True)

#=============================================================================
# Chart 1: Short Memory vs Long Memory ACF
#=============================================================================
print("Generating Chart 1: ACF Comparison...")

np.random.seed(42)
n = 500

# Short memory: AR(1) with phi=0.7
ar1 = np.zeros(n)
for i in range(1, n):
    ar1[i] = 0.7 * ar1[i-1] + np.random.randn()

# Long memory simulation (fractional differencing approximation)
# Using simple method: add persistent shocks
long_mem = np.zeros(n)
d = 0.35  # fractional differencing parameter
weights = np.array([1.0])
for k in range(1, 100):
    weights = np.append(weights, weights[-1] * (d + k - 1) / k)

noise = np.random.randn(n + 100)
for i in range(n):
    long_mem[i] = np.sum(weights[:min(i+1, 100)] * noise[100+i:100+i-min(i+1,100):-1])

# Calculate ACFs
max_lag = 50
acf_short = acf(ar1, nlags=max_lag, fft=True)
acf_long = acf(long_mem, nlags=max_lag, fft=True)

# Theoretical decay
lags = np.arange(max_lag + 1)
exp_decay = 0.7 ** lags  # Exponential decay for AR(1)
hyp_decay = (lags + 1).astype(float) ** (2*d - 1)  # Hyperbolic decay
hyp_decay = hyp_decay / hyp_decay[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Short memory ACF
axes[0].bar(lags, acf_short, color=COLORS['blue'], alpha=0.7, width=0.8, label='Empirical ACF')
axes[0].plot(lags, exp_decay, 'r--', linewidth=2, label='Exponential decay')
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].axhline(y=1.96/np.sqrt(n), color='gray', linestyle=':', alpha=0.7)
axes[0].axhline(y=-1.96/np.sqrt(n), color='gray', linestyle=':', alpha=0.7)
axes[0].set_title('Short Memory (AR(1), φ=0.7)\nFast Exponential Decay', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
axes[0].set_xlim(-1, max_lag)

# Long memory ACF
axes[1].bar(lags, acf_long, color=COLORS['red'], alpha=0.7, width=0.8, label='Empirical ACF')
axes[1].plot(lags, hyp_decay * 0.8, 'b--', linewidth=2, label='Hyperbolic decay')
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].axhline(y=1.96/np.sqrt(n), color='gray', linestyle=':', alpha=0.7)
axes[1].axhline(y=-1.96/np.sqrt(n), color='gray', linestyle=':', alpha=0.7)
axes[1].set_title('Long Memory (ARFIMA, d=0.35)\nSlow Hyperbolic Decay', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('ACF')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
axes[1].set_xlim(-1, max_lag)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch8_acf_comparison.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch8_acf_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch8_acf_comparison.pdf")

#=============================================================================
# Chart 2: Hurst Exponent Interpretation
#=============================================================================
print("Generating Chart 2: Hurst Exponent...")

np.random.seed(123)
n = 300

# Generate three types of series
# H < 0.5: Mean-reverting
mean_rev = np.zeros(n)
for i in range(1, n):
    mean_rev[i] = -0.3 * mean_rev[i-1] + np.random.randn()
mean_rev = np.cumsum(mean_rev) * 0.3

# H = 0.5: Random walk
random_walk = np.cumsum(np.random.randn(n))

# H > 0.5: Trending/Persistent
persistent = np.zeros(n)
trend = 0
for i in range(n):
    trend += 0.3 * np.sign(trend) + np.random.randn() * 0.5
    persistent[i] = trend
persistent = np.cumsum(persistent) * 0.5

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(mean_rev, color=COLORS['green'], linewidth=1.2)
axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0].set_title('Anti-Persistent (H < 0.5)\nMean-Reverting', fontweight='bold', fontsize=11, color=COLORS['green'])
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')

axes[1].plot(random_walk, color=COLORS['gray'], linewidth=1.2)
axes[1].set_title('Random Walk (H = 0.5)\nNo Memory', fontweight='bold', fontsize=11, color=COLORS['gray'])
axes[1].set_xlabel('Time')

axes[2].plot(persistent, color=COLORS['red'], linewidth=1.2)
axes[2].set_title('Persistent (H > 0.5)\nTrending', fontweight='bold', fontsize=11, color=COLORS['red'])
axes[2].set_xlabel('Time')

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch8_hurst_interpretation.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch8_hurst_interpretation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch8_hurst_interpretation.pdf")

#=============================================================================
# Chart 3: Long Memory in Volatility (Financial Stylized Fact)
#=============================================================================
print("Generating Chart 3: Long Memory in Volatility...")

np.random.seed(456)
n = 1000

# Simulate GARCH-like returns with volatility clustering
returns = np.zeros(n)
volatility = np.ones(n)

for i in range(1, n):
    volatility[i] = 0.1 + 0.85 * volatility[i-1] + 0.1 * returns[i-1]**2
    returns[i] = np.sqrt(volatility[i]) * np.random.randn()

abs_returns = np.abs(returns)

# ACFs
max_lag = 100
acf_ret = acf(returns, nlags=max_lag, fft=True)
acf_abs = acf(abs_returns, nlags=max_lag, fft=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Returns series
axes[0, 0].plot(returns, color=COLORS['blue'], linewidth=0.5, alpha=0.8)
axes[0, 0].set_title('Simulated Returns', fontweight='bold')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Return')

# Absolute returns (volatility proxy)
axes[0, 1].plot(abs_returns, color=COLORS['red'], linewidth=0.5, alpha=0.8)
axes[0, 1].set_title('Absolute Returns (Volatility Proxy)', fontweight='bold')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('|Return|')

# ACF of returns (short memory)
lags = np.arange(max_lag + 1)
axes[1, 0].bar(lags, acf_ret, color=COLORS['blue'], alpha=0.7, width=0.8)
axes[1, 0].axhline(y=1.96/np.sqrt(n), color='red', linestyle='--', alpha=0.5)
axes[1, 0].axhline(y=-1.96/np.sqrt(n), color='red', linestyle='--', alpha=0.5)
axes[1, 0].set_title('ACF of Returns (Short Memory)', fontweight='bold', color=COLORS['blue'])
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')
axes[1, 0].set_xlim(-1, max_lag)

# ACF of absolute returns (long memory)
axes[1, 1].bar(lags, acf_abs, color=COLORS['red'], alpha=0.7, width=0.8)
axes[1, 1].axhline(y=1.96/np.sqrt(n), color='blue', linestyle='--', alpha=0.5)
axes[1, 1].axhline(y=-1.96/np.sqrt(n), color='blue', linestyle='--', alpha=0.5)
axes[1, 1].set_title('ACF of |Returns| (Long Memory!)', fontweight='bold', color=COLORS['red'])
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('ACF')
axes[1, 1].set_xlim(-1, max_lag)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch8_volatility_long_memory.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch8_volatility_long_memory.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch8_volatility_long_memory.pdf")

#=============================================================================
# Chart 4: Feature Engineering for Time Series
#=============================================================================
print("Generating Chart 4: Feature Engineering...")

np.random.seed(789)
n = 100

# Sample time series
y = np.cumsum(np.random.randn(n)) + 50

# Create features
lag1 = np.roll(y, 1)
lag1[0] = np.nan
lag2 = np.roll(y, 2)
lag2[:2] = np.nan

# Rolling mean (window=7)
rolling_mean = pd.Series(y).rolling(7).mean().values

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Original series
axes[0, 0].plot(y, color=COLORS['blue'], linewidth=1.5, label='y_t')
axes[0, 0].set_title('Original Series y_t', fontweight='bold')
axes[0, 0].set_xlabel('Time (t)')
axes[0, 0].set_ylabel('Value')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

# Lag features
axes[0, 1].plot(y, color=COLORS['blue'], linewidth=1.5, alpha=0.5, label='y_t')
axes[0, 1].plot(lag1, color=COLORS['red'], linewidth=1.5, label='y_{t-1} (lag 1)')
axes[0, 1].plot(lag2, color=COLORS['orange'], linewidth=1.5, label='y_{t-2} (lag 2)')
axes[0, 1].set_title('Lag Features', fontweight='bold')
axes[0, 1].set_xlabel('Time (t)')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

# Rolling statistics
axes[1, 0].plot(y, color=COLORS['blue'], linewidth=1, alpha=0.5, label='y_t')
axes[1, 0].plot(rolling_mean, color=COLORS['green'], linewidth=2, label='Rolling Mean (7)')
axes[1, 0].set_title('Rolling Statistics', fontweight='bold')
axes[1, 0].set_xlabel('Time (t)')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

# Feature matrix illustration
feature_names = ['y_{t-1}', 'y_{t-2}', 'y_{t-3}', 'MA_7', 'STD_7']
feature_importance = [0.35, 0.20, 0.12, 0.18, 0.15]
colors = [COLORS['blue'], COLORS['blue'], COLORS['blue'], COLORS['green'], COLORS['orange']]

bars = axes[1, 1].barh(feature_names, feature_importance, color=colors, alpha=0.8)
axes[1, 1].set_title('Example Feature Importance', fontweight='bold')
axes[1, 1].set_xlabel('Importance')
axes[1, 1].set_xlim(0, 0.45)

for bar, val in zip(bars, feature_importance):
    axes[1, 1].text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.0%}',
                    va='center', fontsize=10)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch8_feature_engineering.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch8_feature_engineering.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch8_feature_engineering.pdf")

#=============================================================================
# Chart 5: Random Forest Prediction Example
#=============================================================================
print("Generating Chart 5: Random Forest Prediction...")

np.random.seed(321)
n = 200

# Generate data with trend and seasonality
t = np.arange(n)
trend = 0.05 * t
seasonal = 5 * np.sin(2 * np.pi * t / 30)
noise = np.random.randn(n) * 2
y = 50 + trend + seasonal + noise

# Simulate train/test split
train_size = 150
y_train = y[:train_size]
y_test = y[train_size:]

# Simulate RF predictions (with some error)
y_pred = y_test + np.random.randn(len(y_test)) * 1.5  # Add some prediction error

fig, ax = plt.subplots(figsize=(14, 5))

# Training data
ax.plot(range(train_size), y_train, color=COLORS['blue'], linewidth=1.2, label='Training Data')

# Test data
ax.plot(range(train_size, n), y_test, color=COLORS['green'], linewidth=1.5, label='Actual (Test)')

# Predictions
ax.plot(range(train_size, n), y_pred, color=COLORS['red'], linewidth=1.5, linestyle='--', label='RF Prediction')

# Highlight test region
ax.axvspan(train_size, n, alpha=0.1, color='gray')
ax.axvline(x=train_size, color='black', linestyle=':', alpha=0.7)
ax.text(train_size + 2, ax.get_ylim()[1] - 3, 'Test Period', fontsize=10)

ax.set_title('Random Forest Time Series Prediction', fontweight='bold', fontsize=12)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch8_rf_prediction.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch8_rf_prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch8_rf_prediction.pdf")

#=============================================================================
# Chart 6: Time Series Cross-Validation
#=============================================================================
print("Generating Chart 6: Time Series CV...")

fig, ax = plt.subplots(figsize=(12, 5))

folds = 5
total_len = 10

for i in range(folds):
    train_end = 4 + i
    test_start = train_end
    test_end = test_start + 1

    y_pos = folds - i - 1

    # Training set
    ax.barh(y_pos, train_end, left=0, height=0.6, color=COLORS['blue'], alpha=0.7)

    # Test set
    ax.barh(y_pos, 1, left=test_start, height=0.6, color=COLORS['red'], alpha=0.7)

    ax.text(-0.3, y_pos, f'Fold {i+1}', ha='right', va='center', fontsize=11)

# Legend
ax.barh(-0.8, 0.8, left=2, height=0.4, color=COLORS['blue'], alpha=0.7, label='Training')
ax.barh(-0.8, 0.8, left=4, height=0.4, color=COLORS['red'], alpha=0.7, label='Test')
ax.text(2.9, -0.8, 'Train', va='center', fontsize=10, color='white', fontweight='bold')
ax.text(4.9, -0.8, 'Test', va='center', fontsize=10, color='white', fontweight='bold')

ax.set_xlim(-1, total_len)
ax.set_ylim(-1.5, folds)
ax.set_xlabel('Time →', fontsize=12)
ax.set_title('Time Series Cross-Validation (Walk-Forward)', fontweight='bold', fontsize=13)
ax.set_yticks([])
ax.spines['left'].set_visible(False)

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch8_timeseries_cv.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch8_timeseries_cv.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch8_timeseries_cv.pdf")

#=============================================================================
# Chart 7: LSTM Cell Architecture (Simplified)
#=============================================================================
print("Generating Chart 7: LSTM Architecture...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Main cell box
cell = FancyBboxPatch((3, 1.5), 8, 5, boxstyle="round,pad=0.1",
                       facecolor='lightblue', edgecolor=COLORS['blue'], linewidth=2)
ax.add_patch(cell)
ax.text(7, 6.2, 'LSTM Cell', ha='center', fontsize=14, fontweight='bold', color=COLORS['blue'])

# Gates (circles)
gate_r = 0.4
# Forget gate
ax.add_patch(plt.Circle((4.5, 4), gate_r, color=COLORS['red'], alpha=0.8))
ax.text(4.5, 4, 'f', ha='center', va='center', fontsize=12, color='white', fontweight='bold')
ax.text(4.5, 3.2, 'Forget', ha='center', fontsize=9)

# Input gate
ax.add_patch(plt.Circle((6.5, 4), gate_r, color=COLORS['green'], alpha=0.8))
ax.text(6.5, 4, 'i', ha='center', va='center', fontsize=12, color='white', fontweight='bold')
ax.text(6.5, 3.2, 'Input', ha='center', fontsize=9)

# Output gate
ax.add_patch(plt.Circle((8.5, 4), gate_r, color=COLORS['orange'], alpha=0.8))
ax.text(8.5, 4, 'o', ha='center', va='center', fontsize=12, color='white', fontweight='bold')
ax.text(8.5, 3.2, 'Output', ha='center', fontsize=9)

# Cell state (horizontal line at top)
ax.annotate('', xy=(10.5, 5.5), xytext=(3.5, 5.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=2))
ax.text(7, 5.8, 'Cell State (C_t)', ha='center', fontsize=10, style='italic')

# Hidden state output
ax.annotate('', xy=(11.5, 4), xytext=(9.5, 4),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))
ax.text(12.5, 4, 'h_t\n(output)', ha='center', fontsize=10)

# Inputs
ax.annotate('', xy=(3.5, 4), xytext=(1.5, 4),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))
ax.text(0.5, 4, 'x_t\n(input)', ha='center', fontsize=10)

ax.annotate('', xy=(3.5, 2.5), xytext=(1.5, 2.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=2))
ax.text(0.5, 2.5, 'h_{t-1}\n(prev)', ha='center', fontsize=10)

# Previous cell state
ax.annotate('', xy=(3.5, 5.5), xytext=(1.5, 5.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['blue'], lw=2))
ax.text(0.5, 5.5, 'C_{t-1}', ha='center', fontsize=10, color=COLORS['blue'])

# Key insight box
ax.add_patch(FancyBboxPatch((3, 0.3), 8, 0.8, boxstyle="round,pad=0.05",
                            facecolor='lightyellow', edgecolor=COLORS['orange'], linewidth=1))
ax.text(7, 0.7, 'Key: Gates control information flow → Solves vanishing gradient',
        ha='center', fontsize=10, style='italic')

plt.savefig('charts/ch8_lstm_architecture.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch8_lstm_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch8_lstm_architecture.pdf")

#=============================================================================
# Chart 8: Model Comparison
#=============================================================================
print("Generating Chart 8: Model Comparison...")

models = ['ARIMA', 'ARFIMA', 'Random\nForest', 'LSTM']
rmse = [2.45, 2.32, 2.18, 2.25]
training_time = [1, 2, 15, 120]  # relative units

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# RMSE comparison
colors_bar = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['red']]
bars1 = axes[0].bar(models, rmse, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
axes[0].set_ylabel('RMSE (lower is better)')
axes[0].set_title('Prediction Accuracy Comparison', fontweight='bold', fontsize=12)
axes[0].set_ylim(0, 3)

for bar, val in zip(bars1, rmse):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}',
                 ha='center', fontsize=11, fontweight='bold')

# Training time comparison (log scale for visibility)
bars2 = axes[1].bar(models, training_time, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
axes[1].set_ylabel('Relative Training Time')
axes[1].set_title('Computational Cost Comparison', fontweight='bold', fontsize=12)
axes[1].set_yscale('log')
axes[1].set_ylim(0.5, 200)

for bar, val in zip(bars2, training_time):
    axes[1].text(bar.get_x() + bar.get_width()/2, val * 1.2, f'{val}x',
                 ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch8_model_comparison.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch8_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch8_model_comparison.pdf")

#=============================================================================
# Chart 9: ARFIMA d Parameter Effect
#=============================================================================
print("Generating Chart 9: ARFIMA d Parameter...")

fig, ax = plt.subplots(figsize=(12, 5))

d_values = [0, 0.2, 0.4, 0.49]
colors_d = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['red']]
max_lag = 30
lags = np.arange(1, max_lag + 1)

for d, color in zip(d_values, colors_d):
    if d == 0:
        # Exponential decay for ARMA
        acf_theoretical = 0.7 ** lags
    else:
        # Hyperbolic decay for ARFIMA
        acf_theoretical = lags.astype(float) ** (2*d - 1)
        acf_theoretical = acf_theoretical / acf_theoretical[0] * 0.8

    label = f'd = {d}' if d > 0 else 'd = 0 (ARMA)'
    ax.plot(lags, acf_theoretical, color=color, linewidth=2, marker='o',
            markersize=4, label=label)

ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Lag (k)', fontsize=11)
ax.set_ylabel('ACF ρ(k)', fontsize=11)
ax.set_title('Effect of Fractional Differencing Parameter d on ACF Decay', fontweight='bold', fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)
ax.set_xlim(0, max_lag + 1)

# Add annotations
ax.annotate('Fast decay\n(Short memory)', xy=(10, 0.7**10), xytext=(15, 0.25),
            arrowprops=dict(arrowstyle='->', color=COLORS['blue']),
            fontsize=9, color=COLORS['blue'])
ax.annotate('Slow decay\n(Long memory)', xy=(20, 0.15), xytext=(22, 0.35),
            arrowprops=dict(arrowstyle='->', color=COLORS['red']),
            fontsize=9, color=COLORS['red'])

plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
plt.savefig('charts/ch8_arfima_d_effect.pdf', dpi=150, bbox_inches='tight')
plt.savefig('charts/ch8_arfima_d_effect.png', dpi=150, bbox_inches='tight')
plt.close()
print("Created: ch8_arfima_d_effect.pdf")

print("\n" + "="*50)
print("All Chapter 8 charts generated successfully!")
print("="*50)
