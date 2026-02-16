"""
TSA_ch0_ses
===========
Simple Exponential Smoothing (SES)

This script demonstrates:
- SES formula: X_hat(t+1) = alpha * X_t + (1-alpha) * X_hat_t
- Effect of smoothing parameter alpha
- Small alpha = more smoothing, large alpha = more reactive
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

# Generate sample data with level shifts
n = 100
data = np.concatenate([
    np.random.normal(10, 1, 30),
    np.random.normal(15, 1, 40),
    np.random.normal(12, 1, 30)
])

def simple_exp_smoothing(data, alpha, initial=None):
    """Simple Exponential Smoothing"""
    n = len(data)
    smoothed = np.zeros(n)
    smoothed[0] = initial if initial is not None else data[0]

    for t in range(1, n):
        smoothed[t] = alpha * data[t-1] + (1 - alpha) * smoothed[t-1]

    return smoothed

# Apply SES with different alpha values
alphas = [0.1, 0.3, 0.5, 0.9]
colors = ['blue', 'green', 'orange', 'red']

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, alpha in enumerate(alphas):
    smoothed = simple_exp_smoothing(data, alpha)

    axes[i].plot(data, 'gray', alpha=0.5, linewidth=1, label='Observed')
    axes[i].plot(smoothed, colors[i], linewidth=2, label=f'SES (α={alpha})')
    axes[i].set_title(f'α = {alpha}: {"Very smooth" if alpha < 0.3 else "Reactive" if alpha > 0.7 else "Moderate"}',
                      fontsize=12)
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Value')
    axes[i].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
    axes[i].grid(True, alpha=0.3)

    # Add annotation about weight on recent observation
    axes[i].text(0.02, 0.98, f'Weight on X_t: {alpha*100:.0f}%\nWeight on history: {(1-alpha)*100:.0f}%',
                 transform=axes[i].transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('../../charts/ch1_exponential_smoothing.png', dpi=150, bbox_inches='tight')
plt.show()

# Demonstrate the formula step by step
print("SES Formula: X_hat(t+1) = α * X_t + (1-α) * X_hat_t")
print("\nExample with α = 0.3:")
print("Data: [10, 12, 11, 14, 13]")
print("Starting with X_hat_1 = 10:")

example_data = [10, 12, 11, 14, 13]
alpha = 0.3
x_hat = [10]  # Initial value

for t in range(1, len(example_data)):
    new_forecast = alpha * example_data[t-1] + (1 - alpha) * x_hat[-1]
    print(f"X_hat_{t+1} = {alpha} * {example_data[t-1]} + {1-alpha} * {x_hat[-1]:.2f} = {new_forecast:.2f}")
    x_hat.append(new_forecast)

# Forecast for t=6
forecast_6 = alpha * example_data[-1] + (1 - alpha) * x_hat[-1]
print(f"X_hat_6 = {alpha} * {example_data[-1]} + {1-alpha} * {x_hat[-1]:.2f} = {forecast_6:.2f}")
