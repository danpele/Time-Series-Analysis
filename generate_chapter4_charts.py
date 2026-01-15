#!/usr/bin/env python3
"""Generate charts for Chapter 4: SARIMA Models"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['font.size'] = 12

# Colors
BLUE = '#1A3A6E'
RED = '#DC3545'
GREEN = '#2E7D32'
ORANGE = '#FF8C00'

np.random.seed(42)

# Load airline passengers data
try:
    from statsmodels.datasets import get_rdataset
    airline = get_rdataset("AirPassengers").data
    airline.index = pd.date_range(start='1949-01', periods=len(airline), freq='M')
    passengers = airline['value']
except:
    # Synthetic airline-like data
    n = 144
    dates = pd.date_range(start='1949-01-01', periods=n, freq='M')
    trend = np.linspace(100, 500, n)
    seasonal = 50 * np.sin(2 * np.pi * np.arange(n) / 12) * (1 + np.arange(n) / n)
    noise = np.random.normal(0, 10, n)
    passengers = pd.Series(trend + seasonal + noise, index=dates)

print(f"Airline passengers data: {len(passengers)} observations")

# Chart 1: Original time series
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(passengers.index, passengers.values, color=BLUE, linewidth=1.5)
ax.set_title('International Airline Passengers (1949-1960)', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Passengers (thousands)')
plt.tight_layout()
plt.savefig('charts/ch4_airline_data.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch4_airline_data.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch4_airline_data.pdf")

# Chart 2: Seasonal decomposition
decomposition = seasonal_decompose(passengers, model='multiplicative', period=12)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(passengers.index, passengers.values, color=BLUE, linewidth=1)
axes[0].set_title('Original Series', fontweight='bold')
axes[0].set_ylabel('Passengers')

axes[1].plot(passengers.index, decomposition.trend, color=GREEN, linewidth=1.5)
axes[1].set_title('Trend Component', fontweight='bold')
axes[1].set_ylabel('Trend')

axes[2].plot(passengers.index, decomposition.seasonal, color=ORANGE, linewidth=1)
axes[2].set_title('Seasonal Component', fontweight='bold')
axes[2].set_ylabel('Seasonal')

axes[3].plot(passengers.index, decomposition.resid, color=RED, linewidth=0.8)
axes[3].set_title('Residual Component', fontweight='bold')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Date')

plt.tight_layout()
plt.savefig('charts/ch4_decomposition.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch4_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch4_decomposition.pdf")

# Chart 3: ACF/PACF after differencing
# Apply both regular and seasonal differencing
log_passengers = np.log(passengers)
diff_data = log_passengers.diff().diff(12).dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ACF
acf_vals = acf(diff_data, nlags=36)
axes[0].bar(range(len(acf_vals)), acf_vals, color=BLUE, width=0.3)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].axhline(y=1.96/np.sqrt(len(diff_data)), color=RED, linestyle='--', alpha=0.7)
axes[0].axhline(y=-1.96/np.sqrt(len(diff_data)), color=RED, linestyle='--', alpha=0.7)
# Highlight seasonal lags
for lag in [12, 24, 36]:
    if lag < len(acf_vals):
        axes[0].axvline(x=lag, color=ORANGE, linestyle=':', alpha=0.5)
axes[0].set_title('ACF of $\\Delta\\Delta_{12}\\log(Y_t)$', fontweight='bold')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')

# PACF
pacf_vals = pacf(diff_data, nlags=36)
axes[1].bar(range(len(pacf_vals)), pacf_vals, color=GREEN, width=0.3)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].axhline(y=1.96/np.sqrt(len(diff_data)), color=RED, linestyle='--', alpha=0.7)
axes[1].axhline(y=-1.96/np.sqrt(len(diff_data)), color=RED, linestyle='--', alpha=0.7)
for lag in [12, 24, 36]:
    if lag < len(pacf_vals):
        axes[1].axvline(x=lag, color=ORANGE, linestyle=':', alpha=0.5)
axes[1].set_title('PACF of $\\Delta\\Delta_{12}\\log(Y_t)$', fontweight='bold')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('PACF')

plt.tight_layout()
plt.savefig('charts/ch4_acf_pacf.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch4_acf_pacf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch4_acf_pacf.pdf")

# Chart 4: SARIMA forecast
train = log_passengers[:-24]
test = log_passengers[-24:]

# Fit SARIMA(0,1,1)(0,1,1)[12] - Airline model
model = SARIMAX(train, order=(0,1,1), seasonal_order=(0,1,1,12))
fitted = model.fit(disp=False)

# Forecast
forecast = fitted.get_forecast(steps=len(test))
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

fig, ax = plt.subplots(figsize=(14, 6))

# Training data (last 48 obs)
ax.plot(train.index[-48:], np.exp(train.values[-48:]), color=BLUE, linewidth=1.5, label='Training Data')

# Actual test data
ax.plot(test.index, np.exp(test.values), color=GREEN, linewidth=2, label='Actual', marker='o', markersize=4)

# Forecast
ax.plot(test.index, np.exp(forecast_mean.values), color=RED, linewidth=2, linestyle='--', label='SARIMA Forecast')

# Confidence interval
ax.fill_between(test.index, np.exp(conf_int.iloc[:, 0].values), np.exp(conf_int.iloc[:, 1].values),
                color=RED, alpha=0.2, label='95% CI')

ax.axvline(x=train.index[-1], color='gray', linestyle=':', alpha=0.7, linewidth=2)
ax.set_title('Airline Passengers: SARIMA$(0,1,1)\\times(0,1,1)_{12}$ Forecast', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Passengers (thousands)')
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('charts/ch4_sarima_forecast.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch4_sarima_forecast.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch4_sarima_forecast.pdf")

# Chart 5: Diagnostics
resid = fitted.resid

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals over time
axes[0, 0].plot(resid.index, resid.values, color=BLUE, linewidth=0.8)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Residuals over Time', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residual')

# Histogram
axes[0, 1].hist(resid.dropna(), bins=25, density=True, color=BLUE, alpha=0.7, edgecolor='white')
x_range = np.linspace(resid.min(), resid.max(), 100)
axes[0, 1].plot(x_range, stats.norm.pdf(x_range, resid.mean(), resid.std()),
               color=RED, linewidth=2, label='Normal')
axes[0, 1].set_title('Residual Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()

# ACF of residuals
acf_resid = acf(resid.dropna(), nlags=36)
axes[1, 0].bar(range(len(acf_resid)), acf_resid, color=BLUE, width=0.3)
axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
axes[1, 0].axhline(y=1.96/np.sqrt(len(resid)), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=-1.96/np.sqrt(len(resid)), color=RED, linestyle='--', alpha=0.7)
for lag in [12, 24, 36]:
    if lag < len(acf_resid):
        axes[1, 0].axvline(x=lag, color=ORANGE, linestyle=':', alpha=0.3)
axes[1, 0].set_title('ACF of Residuals', fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')

# Q-Q plot
(osm, osr), (slope, intercept, r) = stats.probplot(resid.dropna(), dist='norm', fit=True)
axes[1, 1].scatter(osm, osr, color=BLUE, alpha=0.6, s=30)
axes[1, 1].plot(osm, slope * osm + intercept, color=RED, linewidth=2)
axes[1, 1].set_title('Q-Q Plot', fontweight='bold')
axes[1, 1].set_xlabel('Theoretical Quantiles')
axes[1, 1].set_ylabel('Sample Quantiles')

plt.tight_layout()
plt.savefig('charts/ch4_diagnostics.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch4_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch4_diagnostics.pdf")

print("\nAll Chapter 4 charts generated successfully!")
