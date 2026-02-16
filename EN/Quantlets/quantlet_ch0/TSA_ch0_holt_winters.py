"""
TSA_ch0_holt_winters
====================
Holt-Winters Exponential Smoothing

This script demonstrates:
- Simple Exponential Smoothing (level only)
- Holt's method (level + trend)
- Holt-Winters (level + trend + seasonality)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

# Set random seed
np.random.seed(42)

# Generate data with trend and seasonality
n = 60  # 5 years of monthly data
t = np.arange(n)

# Components
trend = 100 + 1.5 * t
seasonal_pattern = 15 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 5, n)
data = trend + seasonal_pattern + noise

# Create pandas series with date index
dates = pd.date_range(start='2019-01', periods=n, freq='ME')
ts = pd.Series(data, index=dates)

# Split into train and test
train = ts[:-12]
test = ts[-12:]

# Fit models
# 1. Simple Exponential Smoothing
ses = SimpleExpSmoothing(train).fit(smoothing_level=0.3)
ses_forecast = ses.forecast(12)

# 2. Holt's method (trend)
holt = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
holt_forecast = holt.forecast(12)

# 3. Holt-Winters (trend + seasonality)
hw = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12).fit()
hw_forecast = hw.forecast(12)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data and SES
ax1 = axes[0, 0]
ax1.plot(train.index, train, 'b-', linewidth=1, label='Training Data')
ax1.plot(test.index, test, 'b--', linewidth=1, alpha=0.5, label='Test Data')
ax1.plot(ses_forecast.index, ses_forecast, 'r-', linewidth=2, label='SES Forecast')
ax1.axvline(x=train.index[-1], color='gray', linestyle=':', alpha=0.5)
ax1.set_title('Simple Exponential Smoothing\n(Level only - no trend/seasonality)', fontsize=11)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax1.grid(True, alpha=0.3)

# Plot 2: Data and Holt
ax2 = axes[0, 1]
ax2.plot(train.index, train, 'b-', linewidth=1, label='Training Data')
ax2.plot(test.index, test, 'b--', linewidth=1, alpha=0.5, label='Test Data')
ax2.plot(holt_forecast.index, holt_forecast, 'g-', linewidth=2, label="Holt's Forecast")
ax2.axvline(x=train.index[-1], color='gray', linestyle=':', alpha=0.5)
ax2.set_title("Holt's Method\n(Level + Trend)", fontsize=11)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax2.grid(True, alpha=0.3)

# Plot 3: Data and Holt-Winters
ax3 = axes[1, 0]
ax3.plot(train.index, train, 'b-', linewidth=1, label='Training Data')
ax3.plot(test.index, test, 'b--', linewidth=1, alpha=0.5, label='Test Data')
ax3.plot(hw_forecast.index, hw_forecast, 'orange', linewidth=2, label='Holt-Winters Forecast')
ax3.axvline(x=train.index[-1], color='gray', linestyle=':', alpha=0.5)
ax3.set_title('Holt-Winters Method\n(Level + Trend + Seasonality)', fontsize=11)
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax3.grid(True, alpha=0.3)

# Plot 4: Comparison of forecasts
ax4 = axes[1, 1]
ax4.plot(test.index, test, 'ko-', linewidth=2, markersize=6, label='Actual')
ax4.plot(ses_forecast.index, ses_forecast, 'r^--', linewidth=1.5, markersize=6, label='SES')
ax4.plot(holt_forecast.index, holt_forecast, 'gs--', linewidth=1.5, markersize=6, label='Holt')
ax4.plot(hw_forecast.index, hw_forecast, 'mD--', linewidth=1.5, markersize=6, label='Holt-Winters')
ax4.set_title('Forecast Comparison (Test Period)', fontsize=11)
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../charts/holt_winters.png', dpi=150, bbox_inches='tight')
plt.show()

# Calculate forecast errors
def rmse(actual, forecast):
    return np.sqrt(np.mean((actual - forecast)**2))

print("=" * 50)
print("Forecast Accuracy (RMSE)")
print("=" * 50)
print(f"Simple Exponential Smoothing: {rmse(test, ses_forecast):.2f}")
print(f"Holt's Method:                {rmse(test, holt_forecast):.2f}")
print(f"Holt-Winters:                 {rmse(test, hw_forecast):.2f}")

print("\n" + "=" * 50)
print("Method Comparison")
print("=" * 50)
print("""
Method              | Components      | Best For
--------------------|-----------------|---------------------------
SES                 | Level           | No trend, no seasonality
Holt                | Level + Trend   | Trend, no seasonality
Holt-Winters (Add)  | All three       | Constant seasonal amplitude
Holt-Winters (Mult) | All three       | Growing seasonal amplitude
""")

# Print smoothing parameters
print("Estimated Smoothing Parameters (Holt-Winters):")
print(f"  Alpha (level):      {hw.params['smoothing_level']:.4f}")
print(f"  Beta (trend):       {hw.params['smoothing_trend']:.4f}")
print(f"  Gamma (seasonal):   {hw.params['smoothing_seasonal']:.4f}")
