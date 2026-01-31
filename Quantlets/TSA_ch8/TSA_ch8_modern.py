"""
TSA_ch8_modern
==============
Modern Extensions: ARFIMA, Machine Learning, and LSTM

This script demonstrates:
- Long memory and ARFIMA models
- Random Forest for time series
- LSTM neural networks
- Model comparison

Author: Time Series Analysis Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
print("MODERN EXTENSIONS: ARFIMA, ML, AND LSTM")
print("=" * 70)

np.random.seed(42)

# =============================================================================
# 1. Long Memory and ARFIMA
# =============================================================================
print("\n1. LONG MEMORY AND ARFIMA")
print("-" * 40)

def simulate_arfima(n, d, phi=0, theta=0, sigma=1):
    """Simulate ARFIMA(p, d, q) process using fractional differencing."""
    # Generate white noise
    eps = np.random.normal(0, sigma, n + 1000)

    # Fractional integration weights (truncated)
    k = np.arange(1000)
    weights = np.zeros(1000)
    weights[0] = 1
    for j in range(1, 1000):
        weights[j] = weights[j-1] * (d + j - 1) / j

    # Apply fractional integration
    y = np.convolve(eps, weights, mode='full')[:n+1000]

    # Apply AR if phi != 0
    if phi != 0:
        for t in range(1, len(y)):
            y[t] = y[t] + phi * y[t-1]

    return y[-n:]

# Simulate different d values
n = 500
d_values = [0, 0.2, 0.4]
colors = ['#1A3A6E', '#2E7D32', '#DC3545']

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for idx, (d, color) in enumerate(zip(d_values, colors)):
    y = simulate_arfima(n, d)

    # Time series
    axes[0, idx].plot(y, color=color, linewidth=0.8, label=f'd = {d}')
    axes[0, idx].set_title(f'ARFIMA(0, {d}, 0)', fontweight='bold')
    axes[0, idx].set_xlabel('Time')
    axes[0, idx].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

    # ACF (manual calculation)
    acf_vals = [1.0]
    for lag in range(1, 31):
        acf_vals.append(np.corrcoef(y[lag:], y[:-lag])[0, 1])

    axes[1, idx].bar(range(31), acf_vals, color=color, alpha=0.7, edgecolor='white')
    axes[1, idx].axhline(y=1.96/np.sqrt(n), color='red', linestyle='--', linewidth=1)
    axes[1, idx].axhline(y=-1.96/np.sqrt(n), color='red', linestyle='--', linewidth=1)
    axes[1, idx].set_title(f'ACF: d = {d}', fontweight='bold')
    axes[1, idx].set_xlabel('Lag')

plt.suptitle('ARFIMA: Effect of Fractional Differencing Parameter d', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout()
save_fig('ch8_arfima')

print("   ARFIMA(p, d, q) with 0 < d < 0.5:")
print("   - Long memory: slow hyperbolic ACF decay")
print("   - Stationary but with persistent dependence")
print("   - ACF decays like k^(2d-1)")

# =============================================================================
# 2. ACF Comparison: Short vs Long Memory
# =============================================================================
print("\n2. SHORT VS LONG MEMORY")
print("-" * 40)

# Short memory AR(1)
ar1 = np.zeros(n)
for t in range(1, n):
    ar1[t] = 0.7 * ar1[t-1] + np.random.normal(0, 1)

# Long memory ARFIMA
arfima = simulate_arfima(n, d=0.35)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# ACF comparison
lags = 50
acf_ar1 = [np.corrcoef(ar1[lag:], ar1[:-lag])[0, 1] for lag in range(1, lags+1)]
acf_arfima = [np.corrcoef(arfima[lag:], arfima[:-lag])[0, 1] for lag in range(1, lags+1)]

axes[0].plot(range(1, lags+1), acf_ar1, color='#1A3A6E', linewidth=2, marker='o', markersize=3, label='AR(1) φ=0.7')
axes[0].plot(range(1, lags+1), acf_arfima, color='#DC3545', linewidth=2, marker='s', markersize=3, label='ARFIMA d=0.35')
axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')
axes[0].set_title('ACF: Short Memory vs Long Memory', fontweight='bold')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

# Log-log plot to show decay rate
axes[1].loglog(range(1, lags+1), np.abs(acf_ar1), color='#1A3A6E', linewidth=2, marker='o', markersize=3, label='AR(1): Exponential decay')
axes[1].loglog(range(1, lags+1), np.abs(acf_arfima), color='#DC3545', linewidth=2, marker='s', markersize=3, label='ARFIMA: Hyperbolic decay')
axes[1].set_xlabel('Lag (log scale)')
axes[1].set_ylabel('|ACF| (log scale)')
axes[1].set_title('Log-Log Plot: Decay Comparison', fontweight='bold')
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
save_fig('ch8_memory_comparison')

# =============================================================================
# 3. Feature Engineering for ML
# =============================================================================
print("\n3. FEATURE ENGINEERING FOR ML")
print("-" * 40)

# Generate time series with pattern
n_ml = 500
t = np.arange(n_ml)
trend = 0.01 * t
seasonality = 5 * np.sin(2 * np.pi * t / 30)
y_ml = trend + seasonality + np.random.normal(0, 1, n_ml)

# Create features
def create_features(y, lags=5):
    """Create lag features and rolling statistics."""
    df = pd.DataFrame({'y': y})

    # Lag features
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)

    # Rolling features
    df['rolling_mean_5'] = df['y'].rolling(window=5).mean().shift(1)
    df['rolling_std_5'] = df['y'].rolling(window=5).std().shift(1)
    df['rolling_mean_10'] = df['y'].rolling(window=10).mean().shift(1)

    return df.dropna()

df_features = create_features(y_ml, lags=5)
print(f"   Features created: {list(df_features.columns[1:])}")

# Visualize features
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(y_ml, color='#1A3A6E', linewidth=0.8, label='Original series')
axes[0, 0].set_title('Original Time Series', fontweight='bold')
axes[0, 0].set_xlabel('Time')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

axes[0, 1].scatter(df_features['lag_1'], df_features['y'], color='#1A3A6E', alpha=0.5, s=10, label='Yₜ vs Yₜ₋₁')
axes[0, 1].set_xlabel('Yₜ₋₁')
axes[0, 1].set_ylabel('Yₜ')
axes[0, 1].set_title('Lag 1 Feature', fontweight='bold')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

axes[1, 0].plot(df_features['rolling_mean_5'].values, color='#DC3545', linewidth=1.5, label='Rolling Mean (5)')
axes[1, 0].plot(df_features['y'].values, color='#1A3A6E', linewidth=0.5, alpha=0.5, label='Original')
axes[1, 0].set_title('Rolling Mean Feature', fontweight='bold')
axes[1, 0].set_xlabel('Time')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

axes[1, 1].plot(df_features['rolling_std_5'].values, color='#2E7D32', linewidth=1.5, label='Rolling Std (5)')
axes[1, 1].set_title('Rolling Std Feature (Volatility)', fontweight='bold')
axes[1, 1].set_xlabel('Time')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

plt.tight_layout()
save_fig('ch8_features')

# =============================================================================
# 4. Random Forest for Time Series
# =============================================================================
print("\n4. RANDOM FOREST FOR TIME SERIES")
print("-" * 40)

# Train/test split (time series aware)
train_size = int(len(df_features) * 0.8)
train = df_features.iloc[:train_size]
test = df_features.iloc[train_size:]

X_train = train.drop('y', axis=1)
y_train = train['y']
X_test = test.drop('y', axis=1)
y_test = test['y']

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"   Random Forest Performance:")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE: {mae:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 5 Important Features:")
for _, row in feature_importance.head().iterrows():
    print(f"   - {row['feature']}: {row['importance']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Predictions vs Actual - show training context
train_plot_size = min(50, len(y_train))  # Show last 50 training points for context
combined_actual = np.concatenate([y_train.values[-train_plot_size:], y_test.values])
combined_pred = np.concatenate([np.full(train_plot_size, np.nan), y_pred])
time_idx = np.arange(len(combined_actual))
split_idx = train_plot_size

axes[0].plot(time_idx[:split_idx], combined_actual[:split_idx], color='#1A3A6E', linewidth=1.5, label='Training')
axes[0].plot(time_idx[split_idx:], combined_actual[split_idx:], color='#2E7D32', linewidth=1.5, label='Test (Actual)')
axes[0].plot(time_idx[split_idx:], y_pred, color='#DC3545', linewidth=1.5, linestyle='--', label='RF Predicted')

# Visual separator between train and test
axes[0].axvline(x=split_idx, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
y_pos = axes[0].get_ylim()[1] - 0.05 * (axes[0].get_ylim()[1] - axes[0].get_ylim()[0])
axes[0].text(split_idx, y_pos, '  Test ', fontsize=9, ha='left', va='top',
             color='black', fontweight='bold', alpha=0.8)

axes[0].set_title('Random Forest: Actual vs Predicted', fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

# Feature importance
colors_fi = ['#1A3A6E' if i < 3 else '#CCCCCC' for i in range(len(feature_importance))]
axes[1].barh(feature_importance['feature'], feature_importance['importance'], color=colors_fi)
axes[1].set_xlabel('Importance')
axes[1].set_title('Feature Importance', fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
save_fig('ch8_random_forest')

# =============================================================================
# 5. Time Series Cross-Validation
# =============================================================================
print("\n5. TIME SERIES CROSS-VALIDATION")
print("-" * 40)

fig, ax = plt.subplots(figsize=(12, 5))

n_splits = 5
fold_size = len(df_features) // (n_splits + 1)

colors_cv = plt.cm.Blues(np.linspace(0.3, 0.9, n_splits))

for i in range(n_splits):
    train_end = (i + 2) * fold_size
    test_start = train_end
    test_end = test_start + fold_size

    # Training data
    ax.barh(i, train_end, height=0.4, color=colors_cv[i], alpha=0.7, label='Train' if i == 0 else '')
    # Test data
    ax.barh(i, test_end - test_start, left=test_start, height=0.4, color='#DC3545', alpha=0.7, label='Test' if i == 0 else '')

ax.set_xlabel('Time Index')
ax.set_ylabel('Fold')
ax.set_yticks(range(n_splits))
ax.set_yticklabels([f'Fold {i+1}' for i in range(n_splits)])
ax.set_title('Time Series Cross-Validation (Expanding Window)', fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

plt.tight_layout()
save_fig('ch8_cross_validation')

print("   Time series CV rules:")
print("   - Never use future data for training")
print("   - Expanding or sliding window approach")
print("   - Maintain temporal order")

# =============================================================================
# 6. LSTM Concept Visualization
# =============================================================================
print("\n6. LSTM ARCHITECTURE")
print("-" * 40)

# Simplified LSTM simulation (conceptual)
def simple_lstm_demo(sequence, hidden_size=10):
    """Simplified LSTM demonstration."""
    n = len(sequence)

    # Simulated cell state and hidden state evolution
    cell_state = np.zeros(n)
    hidden_state = np.zeros(n)

    # Simulate gates (simplified)
    forget_gate = 1 / (1 + np.exp(-0.5 * sequence))  # sigmoid
    input_gate = 1 / (1 + np.exp(-0.3 * sequence))
    output_gate = 1 / (1 + np.exp(-0.4 * sequence))

    for t in range(1, n):
        # Simplified update
        cell_state[t] = forget_gate[t] * cell_state[t-1] + input_gate[t] * np.tanh(sequence[t])
        hidden_state[t] = output_gate[t] * np.tanh(cell_state[t])

    return cell_state, hidden_state, forget_gate, input_gate, output_gate

# Demo sequence
demo_seq = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.2 * np.random.randn(100)
cell, hidden, fg, ig, og = simple_lstm_demo(demo_seq)

fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Input sequence
axes[0, 0].plot(demo_seq, color='#1A3A6E', linewidth=1.5, label='Input Xₜ')
axes[0, 0].set_title('Input Sequence', fontweight='bold')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# Gates
axes[0, 1].plot(fg, color='#DC3545', linewidth=1.5, alpha=0.8, label='Forget Gate')
axes[0, 1].plot(ig, color='#2E7D32', linewidth=1.5, alpha=0.8, label='Input Gate')
axes[0, 1].plot(og, color='#E67E22', linewidth=1.5, alpha=0.8, label='Output Gate')
axes[0, 1].set_title('LSTM Gates', fontweight='bold')
axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)

# Cell state
axes[1, 0].plot(cell, color='#9B59B6', linewidth=2, label='Cell State Cₜ')
axes[1, 0].set_title('Cell State (Long-term Memory)', fontweight='bold')
axes[1, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# Hidden state
axes[1, 1].plot(hidden, color='#1ABC9C', linewidth=2, label='Hidden State hₜ')
axes[1, 1].set_title('Hidden State (Short-term Memory)', fontweight='bold')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)

# LSTM Architecture diagram (conceptual)
axes[2, 0].text(0.5, 0.9, 'LSTM Cell Architecture', fontsize=14, fontweight='bold',
               ha='center', transform=axes[2, 0].transAxes)
axes[2, 0].text(0.5, 0.7, 'Forget Gate: fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)', fontsize=10,
               ha='center', transform=axes[2, 0].transAxes, family='monospace')
axes[2, 0].text(0.5, 0.55, 'Input Gate: iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)', fontsize=10,
               ha='center', transform=axes[2, 0].transAxes, family='monospace')
axes[2, 0].text(0.5, 0.4, 'Cell State: Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙tanh(Wc·[hₜ₋₁, xₜ])', fontsize=10,
               ha='center', transform=axes[2, 0].transAxes, family='monospace')
axes[2, 0].text(0.5, 0.25, 'Output Gate: oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)', fontsize=10,
               ha='center', transform=axes[2, 0].transAxes, family='monospace')
axes[2, 0].text(0.5, 0.1, 'Hidden State: hₜ = oₜ⊙tanh(Cₜ)', fontsize=10,
               ha='center', transform=axes[2, 0].transAxes, family='monospace')
axes[2, 0].axis('off')

# Advantages
axes[2, 1].text(0.5, 0.9, 'LSTM Advantages for Time Series', fontsize=14, fontweight='bold',
               ha='center', transform=axes[2, 1].transAxes)
advantages = [
    '✓ Captures long-term dependencies',
    '✓ Handles variable-length sequences',
    '✓ Mitigates vanishing gradient problem',
    '✓ Learns complex nonlinear patterns',
    '✓ Automatic feature learning'
]
for i, adv in enumerate(advantages):
    axes[2, 1].text(0.1, 0.7 - i*0.12, adv, fontsize=11,
                   transform=axes[2, 1].transAxes)
axes[2, 1].axis('off')

plt.tight_layout()
save_fig('ch8_lstm')

# =============================================================================
# 7. Model Comparison Summary
# =============================================================================
print("\n7. MODEL COMPARISON SUMMARY")
print("-" * 40)

comparison = pd.DataFrame({
    'Model': ['ARIMA', 'ARFIMA', 'Random Forest', 'LSTM'],
    'Complexity': ['Low', 'Medium', 'Medium', 'High'],
    'Interpretability': ['High', 'High', 'Medium', 'Low'],
    'Long Memory': ['No', 'Yes', 'With features', 'Yes'],
    'Nonlinearity': ['No', 'No', 'Yes', 'Yes'],
    'Data Required': ['Small', 'Medium', 'Medium', 'Large']
})

print("\n   Model Comparison:")
print(comparison.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))

models = ['ARIMA', 'ARFIMA', 'Random Forest', 'LSTM']
metrics = ['Interpretability', 'Flexibility', 'Data Efficiency', 'Long Memory']
scores = np.array([
    [5, 2, 5, 1],  # ARIMA
    [4, 3, 4, 5],  # ARFIMA
    [3, 4, 3, 3],  # RF
    [1, 5, 1, 5]   # LSTM
])

x = np.arange(len(metrics))
width = 0.2
colors_bar = ['#1A3A6E', '#2E7D32', '#DC3545', '#E67E22']

for i, (model, color) in enumerate(zip(models, colors_bar)):
    ax.bar(x + i*width, scores[i], width, label=model, color=color, alpha=0.8)

ax.set_ylabel('Score (1-5)')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics)
ax.set_title('Model Comparison by Characteristics', fontweight='bold')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
ax.set_ylim(0, 6)

plt.tight_layout()
save_fig('ch8_model_comparison')

print("\n" + "=" * 70)
print("MODERN EXTENSIONS ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch8_arfima.pdf: ARFIMA long memory demonstration")
print("  - ch8_memory_comparison.pdf: Short vs long memory")
print("  - ch8_features.pdf: Feature engineering")
print("  - ch8_random_forest.pdf: Random Forest predictions")
print("  - ch8_cross_validation.pdf: Time series CV")
print("  - ch8_lstm.pdf: LSTM architecture")
print("  - ch8_model_comparison.pdf: Model comparison")
