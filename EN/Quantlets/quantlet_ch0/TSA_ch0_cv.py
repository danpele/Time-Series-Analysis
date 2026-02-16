"""
TSA_ch0_cv
==========
Time Series Cross-Validation

This script demonstrates:
- Why standard k-fold CV doesn't work for time series
- Rolling/expanding window cross-validation
- Proper train-test splits respecting temporal order
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Set random seed
np.random.seed(42)

def plot_cv_splits(ax, n_samples, n_splits, min_train_size, method='expanding'):
    """Visualize cross-validation splits"""
    colors_train = plt.cm.Blues(np.linspace(0.3, 0.7, n_splits))
    colors_test = plt.cm.Oranges(np.linspace(0.5, 0.8, n_splits))

    for i in range(n_splits):
        if method == 'expanding':
            train_end = min_train_size + i * ((n_samples - min_train_size) // n_splits)
            train_start = 0
        else:  # rolling
            train_end = min_train_size + i * ((n_samples - min_train_size) // n_splits)
            train_start = max(0, train_end - min_train_size)

        test_start = train_end
        test_end = min(train_end + (n_samples - min_train_size) // n_splits, n_samples)

        # Draw training set
        ax.barh(i, train_end - train_start, left=train_start, height=0.6,
                color=colors_train[i], edgecolor='black', linewidth=0.5,
                label='Train' if i == 0 else '')

        # Draw test set
        ax.barh(i, test_end - test_start, left=test_start, height=0.6,
                color=colors_test[i], edgecolor='black', linewidth=0.5,
                label='Test' if i == 0 else '')

    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f'Split {i+1}' for i in range(n_splits)])
    ax.set_xlabel('Time Index')
    ax.set_xlim(0, n_samples)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Standard k-fold (WRONG for time series)
ax1 = axes[0, 0]
n = 50
k = 5
fold_size = n // k
for i in range(k):
    for j in range(k):
        if i == j:
            color = 'orange'
            label = 'Test' if i == 0 else ''
        else:
            color = 'blue'
            label = 'Train' if i == 0 and j == 1 else ''
        ax1.barh(i, fold_size, left=j*fold_size, height=0.6,
                color=color, edgecolor='black', linewidth=0.5, alpha=0.7)

ax1.set_title('Standard K-Fold CV (WRONG for Time Series!)', fontsize=12, color='red')
ax1.set_yticks(range(k))
ax1.set_yticklabels([f'Fold {i+1}' for i in range(k)])
ax1.set_xlabel('Time Index')
ax1.text(0.5, -0.15, '⚠ Future data used to predict past!', transform=ax1.transAxes,
         ha='center', fontsize=11, color='red', fontweight='bold')

# 2. Expanding window CV (CORRECT)
ax2 = axes[0, 1]
plot_cv_splits(ax2, 50, 5, 20, method='expanding')
ax2.set_title('Expanding Window CV (CORRECT)', fontsize=12, color='green')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

# 3. Rolling window CV (CORRECT)
ax3 = axes[1, 0]
plot_cv_splits(ax3, 50, 5, 20, method='rolling')
ax3.set_title('Rolling Window CV (CORRECT)', fontsize=12, color='green')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)

# 4. Explanation
ax4 = axes[1, 1]
ax4.axis('off')
explanation = """
Time Series Cross-Validation Rules:

✗ NEVER use standard k-fold CV
  - Randomly mixes observations
  - Allows future data in training set
  - Violates temporal ordering

✓ Use Expanding Window CV
  - Training set grows over time
  - Test set always AFTER training
  - Simulates real forecasting scenario

✓ Use Rolling Window CV
  - Fixed training window size
  - Slides through time
  - Good for non-stationary data

Key Principle:
"Never use future data to predict the past"
"""
ax4.text(0.1, 0.95, explanation, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('../../charts/ch8_timeseries_cv.png', dpi=150, bbox_inches='tight')
plt.show()

print("Time Series Cross-Validation Summary:")
print("-" * 50)
print("Standard k-fold: ❌ Violates temporal order")
print("Expanding window: ✓ Training grows, test always after")
print("Rolling window: ✓ Fixed window slides through time")
