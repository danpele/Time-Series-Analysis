"""
TSA_ch0_ex2_metrics
===================
Seminar Exercise 2: Forecast Error Metrics

Problem: Given actual and forecast values, calculate MAE, MSE, RMSE, MAPE
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Given data
actual = np.array([100, 110, 105, 120])
forecast = np.array([95, 108, 110, 115])
n = len(actual)

print("=" * 60)
print("FORECAST ERROR METRICS - Step by Step Solution")
print("=" * 60)

# Step 1: Calculate errors
errors = actual - forecast

print("\nStep 1: Calculate Forecast Errors")
print("-" * 60)
print(f"e_t = X_t - X̂_t")
print()
for t in range(n):
    print(f"  e_{t+1} = {actual[t]} - {forecast[t]} = {errors[t]}")

# Step 2: Absolute errors
abs_errors = np.abs(errors)

print("\nStep 2: Absolute Errors |e_t|")
print("-" * 60)
for t in range(n):
    print(f"  |e_{t+1}| = |{errors[t]}| = {abs_errors[t]}")

# Step 3: Squared errors
sq_errors = errors ** 2

print("\nStep 3: Squared Errors e_t²")
print("-" * 60)
for t in range(n):
    print(f"  e_{t+1}² = ({errors[t]})² = {sq_errors[t]}")

# Step 4: Percentage errors
pct_errors = np.abs(errors / actual) * 100

print("\nStep 4: Absolute Percentage Errors |e_t/X_t| × 100")
print("-" * 60)
for t in range(n):
    print(f"  |e_{t+1}/X_{t+1}| × 100 = |{errors[t]}/{actual[t]}| × 100 = {pct_errors[t]:.2f}%")

# Summary table
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'t':>3} {'X_t':>8} {'X̂_t':>8} {'e_t':>8} {'|e_t|':>8} {'e_t²':>10} {'%err':>8}")
print("-" * 60)
for t in range(n):
    print(f"{t+1:>3} {actual[t]:>8.0f} {forecast[t]:>8.0f} {errors[t]:>8.0f} {abs_errors[t]:>8.0f} {sq_errors[t]:>10.0f} {pct_errors[t]:>7.2f}%")
print("-" * 60)
print(f"{'Sum':>3} {'':>8} {'':>8} {'':>8} {sum(abs_errors):>8.0f} {sum(sq_errors):>10.0f} {sum(pct_errors):>7.2f}%")

# Calculate metrics
mae = np.mean(abs_errors)
mse = np.mean(sq_errors)
rmse = np.sqrt(mse)
mape = np.mean(pct_errors)

print("\n" + "=" * 60)
print("METRIC CALCULATIONS")
print("=" * 60)

print("\na) MAE (Mean Absolute Error):")
print(f"   MAE = (1/n) × Σ|e_t|")
print(f"   MAE = (1/{n}) × ({' + '.join([str(int(e)) for e in abs_errors])})")
print(f"   MAE = (1/{n}) × {sum(abs_errors)}")
print(f"   MAE = {mae:.2f}")

print("\nb) MSE (Mean Squared Error):")
print(f"   MSE = (1/n) × Σe_t²")
print(f"   MSE = (1/{n}) × ({' + '.join([str(int(e)) for e in sq_errors])})")
print(f"   MSE = (1/{n}) × {sum(sq_errors)}")
print(f"   MSE = {mse:.2f}")

print("\nc) RMSE (Root Mean Squared Error):")
print(f"   RMSE = √MSE")
print(f"   RMSE = √{mse:.2f}")
print(f"   RMSE = {rmse:.2f}")

print("\nd) MAPE (Mean Absolute Percentage Error):")
print(f"   MAPE = (100/n) × Σ|e_t/X_t|")
print(f"   MAPE = (100/{n}) × ({' + '.join([f'{e/100:.4f}' for e in pct_errors])})")
print(f"   MAPE = {mape:.2f}%")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Actual vs Forecast
ax1 = axes[0]
t_vals = np.arange(1, n + 1)
width = 0.35
ax1.bar(t_vals - width/2, actual, width, label='Actual $X_t$', color='blue', alpha=0.7)
ax1.bar(t_vals + width/2, forecast, width, label='Forecast $\\hat{X}_t$', color='red', alpha=0.7)
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Value')
ax1.set_title('Actual vs Forecast Values')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax1.set_xticks(t_vals)
ax1.grid(True, alpha=0.3, axis='y')

# Right: Error metrics comparison
ax2 = axes[1]
metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']
values = [mae, mse, rmse, mape]
colors = ['blue', 'green', 'orange', 'purple']
bars = ax2.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Value')
ax2.set_title('Error Metrics Comparison')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val, metric in zip(bars, values, metrics):
    label = f'{val:.2f}%' if metric == 'MAPE' else f'{val:.2f}'
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             label, ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('../../charts/ch0_seminar_ex2_metrics.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch0_seminar_ex2_metrics.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("\n" + "=" * 60)
print("FINAL ANSWERS:")
print("=" * 60)
print(f"a) Errors: e₁={errors[0]}, e₂={errors[1]}, e₃={errors[2]}, e₄={errors[3]}")
print(f"b) MAE  = {mae:.2f}")
print(f"c) MSE  = {mse:.2f}")
print(f"d) RMSE = {rmse:.2f}")
print(f"e) MAPE = {mape:.2f}%")

print("\n" + "=" * 60)
print("INTERPRETATION:")
print("=" * 60)
print(f"""
- MAE ({mae:.2f}): Average absolute error is {mae:.2f} units
- RMSE ({rmse:.2f}): Penalizes large errors more than MAE
  Note: RMSE > MAE indicates some large errors present
  RMSE/MAE ratio = {rmse/mae:.2f}
- MAPE ({mape:.2f}%): Average error is about {mape:.1f}% of actual value
  This is scale-independent, good for comparison
""")
