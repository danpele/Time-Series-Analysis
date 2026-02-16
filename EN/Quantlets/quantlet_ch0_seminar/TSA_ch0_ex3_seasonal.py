"""
TSA_ch0_ex3_seasonal
====================
Seminar Exercise 3: Seasonal Indices

Problem: Verify seasonal indices normalization, calculate seasonal forecast,
and deseasonalize actual values
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


# Given seasonal indices (multiplicative model)
quarters = ['Q1', 'Q2', 'Q3', 'Q4']
seasonal_indices = np.array([0.85, 1.05, 0.90, 1.20])

# Given values
trend_forecast_q4 = 1000  # Trend forecast for Q4
actual_q4_sales = 1150    # Actual Q4 sales

print("=" * 60)
print("SEASONAL INDICES - Step by Step Solution")
print("=" * 60)

# Part a: Verify normalization
print("\na) Verify Seasonal Indices are Properly Normalized")
print("-" * 60)
print("For multiplicative model with m periods:")
print("  - Sum of indices should equal m")
print("  - Or average should equal 1")
print()
print("Given indices:")
for q, idx in zip(quarters, seasonal_indices):
    print(f"  S_{q} = {idx}")

sum_indices = sum(seasonal_indices)
avg_indices = np.mean(seasonal_indices)

print(f"\nSum of indices = {' + '.join([str(s) for s in seasonal_indices])} = {sum_indices}")
print(f"For m = 4 quarters, sum should = 4")
print(f"Result: {sum_indices} = 4 ✓ CORRECTLY NORMALIZED")
print(f"\nAlternatively: Average = {avg_indices} = 1 ✓")

# Part b: Seasonal forecast
print("\n" + "=" * 60)
print("b) Calculate Seasonally Adjusted Forecast for Q4")
print("-" * 60)
print("Multiplicative model: X̂_t = T_t × S_t")
print()
print(f"Given: Trend forecast T_Q4 = {trend_forecast_q4}")
print(f"       Seasonal index S_Q4 = {seasonal_indices[3]}")
print()
print(f"Seasonal forecast:")
print(f"  X̂_Q4 = T_Q4 × S_Q4")
print(f"  X̂_Q4 = {trend_forecast_q4} × {seasonal_indices[3]}")

seasonal_forecast_q4 = trend_forecast_q4 * seasonal_indices[3]
print(f"  X̂_Q4 = {seasonal_forecast_q4:.0f} units")

print("\nInterpretation:")
print(f"  Q4 has a seasonal index of {seasonal_indices[3]}, meaning")
print(f"  Q4 sales are typically {(seasonal_indices[3]-1)*100:.0f}% ABOVE the trend level.")
print(f"  So if trend is {trend_forecast_q4}, seasonal forecast is {seasonal_forecast_q4:.0f}.")

# Part c: Deseasonalize
print("\n" + "=" * 60)
print("c) Calculate Deseasonalized Value for Q4")
print("-" * 60)
print("To deseasonalize (remove seasonal effect):")
print("  X_deseasonalized = X_actual / S_t")
print()
print(f"Given: Actual Q4 sales = {actual_q4_sales}")
print(f"       Seasonal index S_Q4 = {seasonal_indices[3]}")
print()
print(f"Deseasonalized value:")
print(f"  X_deseas = X_actual / S_Q4")
print(f"  X_deseas = {actual_q4_sales} / {seasonal_indices[3]}")

deseasonalized_q4 = actual_q4_sales / seasonal_indices[3]
print(f"  X_deseas = {deseasonalized_q4:.2f} units")

print("\nInterpretation:")
print(f"  Actual Q4 sales: {actual_q4_sales} units")
print(f"  After removing the seasonal boost ({(seasonal_indices[3]-1)*100:.0f}%): {deseasonalized_q4:.2f} units")
print(f"  Trend forecast was: {trend_forecast_q4} units")
print(f"  Difference: {deseasonalized_q4:.2f} - {trend_forecast_q4} = {deseasonalized_q4 - trend_forecast_q4:.2f} units")
print(f"  → Underlying performance was {abs(deseasonalized_q4 - trend_forecast_q4):.2f} units BELOW trend")
print(f"  → Despite the seasonal boost, actual performance was slightly weak")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Left: Seasonal indices
ax1 = axes[0]
colors = ['red' if s < 1 else 'green' for s in seasonal_indices]
bars = ax1.bar(quarters, seasonal_indices, color=colors, alpha=0.7, edgecolor='black')
ax1.axhline(y=1, color='black', linestyle='--', linewidth=2, label='Neutral (1.0)')
ax1.set_ylabel('Seasonal Index')
ax1.set_title('Quarterly Seasonal Indices')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim(0, 1.5)

# Add labels
for bar, val in zip(bars, seasonal_indices):
    pct = (val - 1) * 100
    label = f'{val}\n({pct:+.0f}%)'
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             label, ha='center', va='bottom', fontsize=9)

# Middle: Q4 comparison
ax2 = axes[1]
categories = ['Trend\nForecast', 'Seasonal\nForecast', 'Actual\nSales']
values = [trend_forecast_q4, seasonal_forecast_q4, actual_q4_sales]
colors = ['blue', 'orange', 'green']
bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Units')
ax2.set_title('Q4 Forecast vs Actual')
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'{val:.0f}', ha='center', va='bottom', fontsize=10)

# Right: Deseasonalization
ax3 = axes[2]
categories = ['Actual Q4\n(with seasonal)', 'Deseasonalized\nQ4', 'Trend\n(target)']
values = [actual_q4_sales, deseasonalized_q4, trend_forecast_q4]
colors = ['green', 'purple', 'blue']
bars = ax3.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('Units')
ax3.set_title('Deseasonalization: Q4')
ax3.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             f'{val:.1f}', ha='center', va='bottom', fontsize=10)

# Add arrow showing deseasonalization
ax3.annotate('', xy=(1, deseasonalized_q4), xytext=(0, actual_q4_sales),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax3.text(0.5, (actual_q4_sales + deseasonalized_q4)/2 + 20, f'÷ {seasonal_indices[3]}',
         ha='center', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('../../charts/ch0_seminar_ex3_seasonal.png', dpi=150, bbox_inches='tight', transparent=True)
plt.savefig('../../charts/ch0_seminar_ex3_seasonal.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("\n" + "=" * 60)
print("FINAL ANSWERS:")
print("=" * 60)
print(f"a) Sum of indices = {sum_indices} = 4 ✓ Properly normalized")
print(f"b) Seasonally adjusted Q4 forecast = {seasonal_forecast_q4:.0f} units")
print(f"c) Deseasonalized Q4 value = {deseasonalized_q4:.2f} units")
print(f"   (Below trend of {trend_forecast_q4}, indicating weak underlying performance)")
