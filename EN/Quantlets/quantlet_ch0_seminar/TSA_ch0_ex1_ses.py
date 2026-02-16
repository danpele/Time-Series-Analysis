"""
TSA_ch0_ex1_ses
===============
Seminar Exercise 1: Simple Exponential Smoothing

Problem: Given data [10, 12, 11, 14, 13] with α = 0.3
Calculate forecasts, errors, MAE and RMSE
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


# Given data
data = np.array([10, 12, 11, 14, 13])
alpha = 0.3
n = len(data)

# Initialize arrays
forecasts = np.zeros(n + 1)  # +1 for X_hat_6
errors = np.zeros(n)

# Starting value: X_hat_1 = X_1 = 10
forecasts[0] = data[0]

print("=" * 60)
print("SIMPLE EXPONENTIAL SMOOTHING - Step by Step Solution")
print("=" * 60)
print(f"\nFormula: X_hat(t+1) = α·X_t + (1-α)·X_hat_t")
print(f"         X_hat(t+1) = {alpha}·X_t + {1-alpha}·X_hat_t")
print(f"\nStarting value: X_hat_1 = X_1 = {data[0]}")
print("-" * 60)

# Calculate forecasts step by step
for t in range(1, n + 1):
    # Forecast for time t (made at time t-1)
    forecasts[t] = alpha * data[t-1] + (1 - alpha) * forecasts[t-1]

    if t < n:
        errors[t] = data[t] - forecasts[t]
        print(f"\nStep {t}: Forecast for t={t+1}")
        print(f"  X_hat_{t+1} = {alpha} × X_{t} + {1-alpha} × X_hat_{t}")
        print(f"  X_hat_{t+1} = {alpha} × {data[t-1]} + {1-alpha} × {forecasts[t-1]:.2f}")
        print(f"  X_hat_{t+1} = {alpha * data[t-1]:.2f} + {(1-alpha) * forecasts[t-1]:.2f}")
        print(f"  X_hat_{t+1} = {forecasts[t]:.2f}")
        print(f"  Error e_{t+1} = X_{t+1} - X_hat_{t+1} = {data[t]} - {forecasts[t]:.2f} = {errors[t]:.2f}")
    else:
        print(f"\nStep {t}: Forecast for t=6 (out-of-sample)")
        print(f"  X_hat_6 = {alpha} × X_5 + {1-alpha} × X_hat_5")
        print(f"  X_hat_6 = {alpha} × {data[-1]} + {1-alpha} × {forecasts[-2]:.2f}")
        print(f"  X_hat_6 = {forecasts[-1]:.2f}")

# Summary table
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'t':>3} {'X_t':>8} {'X_hat_t':>10} {'e_t':>8} {'|e_t|':>8} {'e_t^2':>10}")
print("-" * 60)

abs_errors = []
sq_errors = []

for t in range(n):
    if t == 0:
        print(f"{t+1:>3} {data[t]:>8.0f} {forecasts[t]:>10.2f} {'--':>8} {'--':>8} {'--':>10}")
    else:
        abs_e = abs(errors[t])
        sq_e = errors[t]**2
        abs_errors.append(abs_e)
        sq_errors.append(sq_e)
        print(f"{t+1:>3} {data[t]:>8.0f} {forecasts[t]:>10.2f} {errors[t]:>8.2f} {abs_e:>8.2f} {sq_e:>10.2f}")

print(f"{'6':>3} {'?':>8} {forecasts[-1]:>10.2f} {'--':>8} {'--':>8} {'--':>10}")
print("-" * 60)

# Calculate MAE and RMSE
mae = np.mean(abs_errors)
mse = np.mean(sq_errors)
rmse = np.sqrt(mse)

print(f"\nERROR METRICS:")
print(f"  MAE  = (|{abs_errors[0]:.2f}| + |{abs_errors[1]:.2f}| + |{abs_errors[2]:.2f}| + |{abs_errors[3]:.2f}|) / 4")
print(f"       = {sum(abs_errors):.2f} / 4 = {mae:.3f}")
print(f"\n  MSE  = ({sq_errors[0]:.2f} + {sq_errors[1]:.2f} + {sq_errors[2]:.2f} + {sq_errors[3]:.2f}) / 4")
print(f"       = {sum(sq_errors):.2f} / 4 = {mse:.3f}")
print(f"\n  RMSE = √{mse:.3f} = {rmse:.3f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Data vs Forecasts
ax1 = axes[0]
t_vals = np.arange(1, n + 2)
ax1.plot(t_vals[:-1], data, 'bo-', markersize=10, linewidth=2, label='Actual $X_t$')
ax1.plot(t_vals, forecasts, 'r^--', markersize=8, linewidth=2, label='Forecast $\\hat{X}_t$')
ax1.axvline(x=5.5, color='gray', linestyle=':', alpha=0.5)
ax1.text(5.7, 10.5, 'Forecast\nhorizon', fontsize=9, color='gray')
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('Value')
ax1.set_title(f'Simple Exponential Smoothing (α = {alpha})')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, 7))

# Right: Forecast errors
ax2 = axes[1]
ax2.bar(range(2, n+1), errors[1:], color=['green' if e > 0 else 'red' for e in errors[1:]],
        alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linewidth=1)
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Forecast Error ($e_t = X_t - \\hat{X}_t$)')
ax2.set_title('Forecast Errors')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(range(2, n+1))

plt.tight_layout()
plt.savefig('../../charts/ch0_seminar_ex1_ses.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch0_seminar_ex1_ses.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("\n" + "=" * 60)
print("ANSWERS:")
print("=" * 60)
print(f"a) Forecasts: X̂₂={forecasts[1]:.2f}, X̂₃={forecasts[2]:.2f}, X̂₄={forecasts[3]:.2f}, X̂₅={forecasts[4]:.2f}")
print(f"b) Forecast for t=6: X̂₆ = {forecasts[5]:.2f}")
print(f"c) Errors: e₂={errors[1]:.2f}, e₃={errors[2]:.2f}, e₄={errors[3]:.2f}, e₅={errors[4]:.2f}")
print(f"d) MAE = {mae:.3f}, RMSE = {rmse:.3f}")
