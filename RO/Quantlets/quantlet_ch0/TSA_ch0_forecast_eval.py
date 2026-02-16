"""
TSA_ch0_forecast_eval
=====================
Forecast Evaluation Metrics

This script demonstrates:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed
np.random.seed(42)

def calculate_metrics(actual, forecast):
    """Calculate all forecast error metrics"""
    errors = actual - forecast
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2
    pct_errors = np.abs(errors / actual) * 100

    mae = np.mean(abs_errors)
    mse = np.mean(sq_errors)
    rmse = np.sqrt(mse)
    mape = np.mean(pct_errors)

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'errors': errors
    }

# Example 1: Well-behaved forecast
actual1 = np.array([100, 110, 105, 120, 115, 125, 130, 128])
forecast1 = np.array([98, 108, 107, 118, 117, 123, 128, 130])

# Example 2: Forecast with large errors
actual2 = np.array([100, 110, 105, 120, 115, 125, 130, 128])
forecast2 = np.array([95, 105, 115, 110, 125, 115, 140, 120])

# Example 3: Different scale (for MAPE comparison)
actual3 = np.array([1000, 1100, 1050, 1200, 1150, 1250, 1300, 1280])
forecast3 = np.array([980, 1080, 1070, 1180, 1170, 1230, 1280, 1300])

metrics1 = calculate_metrics(actual1, forecast1)
metrics2 = calculate_metrics(actual2, forecast2)
metrics3 = calculate_metrics(actual3, forecast3)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Comparison of forecasts
ax1 = axes[0, 0]
t = np.arange(len(actual1))
ax1.plot(t, actual1, 'ko-', linewidth=2, markersize=8, label='Actual')
ax1.plot(t, forecast1, 'b^--', linewidth=2, markersize=8, label='Good Forecast')
ax1.plot(t, forecast2, 'rs--', linewidth=2, markersize=8, label='Poor Forecast')
ax1.set_title('Forecast Comparison', fontsize=12)
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax1.grid(True, alpha=0.3)

# Plot 2: Error distribution
ax2 = axes[0, 1]
width = 0.35
x = np.arange(4)
metrics_names = ['MAE', 'MSE', 'RMSE', 'MAPE']
values1 = [metrics1['MAE'], metrics1['MSE'], metrics1['RMSE'], metrics1['MAPE']]
values2 = [metrics2['MAE'], metrics2['MSE'], metrics2['RMSE'], metrics2['MAPE']]

bars1 = ax2.bar(x - width/2, values1, width, label='Good Forecast', color='blue', alpha=0.7)
bars2 = ax2.bar(x + width/2, values2, width, label='Poor Forecast', color='red', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics_names)
ax2.set_title('Error Metrics Comparison', fontsize=12)
ax2.set_ylabel('Value')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars1, values1):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, values2):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}', ha='center', va='bottom', fontsize=8)

# Plot 3: MAPE comparison across scales
ax3 = axes[1, 0]
x = np.arange(2)
mapes = [metrics1['MAPE'], metrics3['MAPE']]
scales = ['Scale: ~100', 'Scale: ~1000']
bars = ax3.bar(x, mapes, color=['blue', 'green'], alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(scales)
ax3.set_title('MAPE: Scale-Independent Metric', fontsize=12)
ax3.set_ylabel('MAPE (%)')
ax3.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, mapes):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=10)

# Plot 4: Formula summary
ax4 = axes[1, 1]
ax4.axis('off')
formulas = """
Error Metrics Formulas:

MAE = (1/n) Σ |eₜ|
      Mean Absolute Error
      - Same units as data
      - Treats all errors equally

MSE = (1/n) Σ eₜ²
      Mean Squared Error
      - Squared units
      - Penalizes large errors more

RMSE = √MSE
       Root Mean Squared Error
       - Same units as data
       - Penalizes large errors more

MAPE = (100/n) Σ |eₜ/Xₜ|
       Mean Absolute Percentage Error
       - Scale-independent (%)
       - Fails when Xₜ ≈ 0
"""
ax4.text(0.1, 0.95, formulas, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('../../charts/ch1_forecast_eval.png', dpi=150, bbox_inches='tight')
plt.show()

# Print detailed results
print("=" * 50)
print("Forecast Evaluation Results")
print("=" * 50)
print(f"\nGood Forecast: MAE={metrics1['MAE']:.2f}, RMSE={metrics1['RMSE']:.2f}, MAPE={metrics1['MAPE']:.2f}%")
print(f"Poor Forecast: MAE={metrics2['MAE']:.2f}, RMSE={metrics2['RMSE']:.2f}, MAPE={metrics2['MAPE']:.2f}%")
print(f"\nNote: RMSE penalizes large errors more than MAE")
print(f"RMSE/MAE ratio (Good): {metrics1['RMSE']/metrics1['MAE']:.2f}")
print(f"RMSE/MAE ratio (Poor): {metrics2['RMSE']/metrics2['MAE']:.2f}")
