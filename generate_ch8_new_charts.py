#!/usr/bin/env python3
"""
Generate new charts for Chapter 8: Modern Extensions
EUR/RON example, LSTM training, RF feature importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
MAIN_BLUE = '#1A3A6E'
ACCENT_BLUE = '#2A528C'
IDA_RED = '#DC3545'
FOREST = '#2E7D32'
AMBER = '#B5853F'

def set_style():
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14

set_style()

# ============================================================================
# Chart 1: EUR/RON Series
# ============================================================================
def generate_eurron_series():
    """Generate simulated EUR/RON series chart"""
    np.random.seed(42)

    # Simulate EUR/RON-like series (2015-2024)
    dates = pd.date_range('2015-01-01', '2024-12-31', freq='B')
    n = len(dates)

    # Start at ~4.5, trend upward with volatility clustering
    trend = np.linspace(4.45, 4.98, n)

    # Add GARCH-like volatility
    returns = np.zeros(n)
    volatility = np.zeros(n)
    volatility[0] = 0.005

    for t in range(1, n):
        volatility[t] = 0.0001 + 0.1 * returns[t-1]**2 + 0.85 * volatility[t-1]
        returns[t] = np.random.normal(0, np.sqrt(volatility[t]))

    # Cumulate returns to get price
    price = trend * np.exp(np.cumsum(returns))

    # Create DataFrame
    df = pd.DataFrame({'EURRON': price, 'Returns': returns * 100}, index=dates)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Price
    axes[0].plot(df.index, df['EURRON'], color=MAIN_BLUE, linewidth=0.8)
    axes[0].set_ylabel('Curs EUR/RON')
    axes[0].set_title('Evoluția Cursului EUR/RON (2015-2024)', fontweight='bold')
    axes[0].axhline(y=df['EURRON'].mean(), color=IDA_RED, linestyle='--',
                    alpha=0.7, label=f'Media: {df["EURRON"].mean():.2f}')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    # Returns
    axes[1].plot(df.index, df['Returns'], color=ACCENT_BLUE, linewidth=0.5, alpha=0.8)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_ylabel('Randamente (%)')
    axes[1].set_xlabel('Data')
    axes[1].set_title('Randamente Zilnice', fontweight='bold')

    # Highlight volatility clusters
    high_vol = df['Returns'].abs() > 1.5
    axes[1].scatter(df.index[high_vol], df['Returns'][high_vol],
                   color=IDA_RED, s=10, alpha=0.6, label='Volatilitate ridicată')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

    plt.tight_layout()
    plt.savefig('charts/ch8_eurron_series.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: ch8_eurron_series.pdf")

# ============================================================================
# Chart 2: LSTM Training Curves
# ============================================================================
def generate_lstm_training():
    """Generate LSTM training curves"""
    np.random.seed(42)

    epochs = np.arange(1, 51)

    # Simulate training loss (decreasing with noise)
    train_loss = 0.5 * np.exp(-0.08 * epochs) + 0.02 + np.random.normal(0, 0.005, 50)
    train_loss = np.maximum(train_loss, 0.015)

    # Validation loss (slightly higher, with more noise)
    val_loss = 0.55 * np.exp(-0.07 * epochs) + 0.025 + np.random.normal(0, 0.008, 50)
    val_loss = np.maximum(val_loss, 0.02)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(epochs, train_loss, color=MAIN_BLUE, linewidth=2, label='Training Loss')
    axes[0].plot(epochs, val_loss, color=IDA_RED, linewidth=2, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Curba de Învățare LSTM', fontweight='bold')
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    axes[0].set_xlim(1, 50)

    # Learning rate effect (simulated)
    lr_epochs = np.arange(1, 51)
    lr_fast = 0.5 * np.exp(-0.15 * lr_epochs) + 0.03
    lr_medium = 0.5 * np.exp(-0.08 * lr_epochs) + 0.02
    lr_slow = 0.5 * np.exp(-0.03 * lr_epochs) + 0.025

    axes[1].plot(lr_epochs, lr_fast, color=IDA_RED, linewidth=2, label='LR=0.01 (rapid)')
    axes[1].plot(lr_epochs, lr_medium, color=MAIN_BLUE, linewidth=2, label='LR=0.001 (optim)')
    axes[1].plot(lr_epochs, lr_slow, color=FOREST, linewidth=2, label='LR=0.0001 (lent)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (MSE)')
    axes[1].set_title('Efectul Learning Rate', fontweight='bold')
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    axes[1].set_xlim(1, 50)

    plt.tight_layout()
    plt.savefig('charts/ch8_lstm_training.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: ch8_lstm_training.pdf")

# ============================================================================
# Chart 3: Random Forest Feature Importance
# ============================================================================
def generate_rf_feature_importance():
    """Generate RF feature importance chart"""

    features = ['lag_1', 'lag_2', 'rolling_std_5', 'lag_3', 'rolling_mean_5',
                'lag_4', 'lag_5', 'rolling_mean_20', 'dayofweek', 'month']
    importance = [0.28, 0.18, 0.15, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.03]

    # Sort by importance
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = [MAIN_BLUE if imp > 0.1 else ACCENT_BLUE for imp in np.array(importance)[sorted_idx]]

    bars = ax.barh([features[i] for i in sorted_idx],
                   [importance[i] for i in sorted_idx],
                   color=colors, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Importanță')
    ax.set_title('Importanța Features în Random Forest', fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, [importance[i] for i in sorted_idx]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontsize=9)

    ax.set_xlim(0, 0.35)

    plt.tight_layout()
    plt.savefig('charts/ch8_rf_feature_importance.pdf', dpi=150, bbox_inches='tight')
    plt.close()
    print("Generated: ch8_rf_feature_importance.pdf")

# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("Generating Chapter 8 new charts...")
    generate_eurron_series()
    generate_lstm_training()
    generate_rf_feature_importance()
    print("Done!")
