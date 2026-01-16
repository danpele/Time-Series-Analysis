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
# Skip first 13 residuals (burn-in period for seasonal model with s=12)
resid = fitted.resid[13:]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals over time
axes[0, 0].plot(resid.index, resid.values, color=BLUE, linewidth=0.8)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Residuals over Time (after burn-in)', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residual')

# Histogram
resid_clean = resid.dropna()
axes[0, 1].hist(resid_clean, bins=25, density=True, color=BLUE, alpha=0.7, edgecolor='white')
x_range = np.linspace(resid_clean.min(), resid_clean.max(), 100)
axes[0, 1].plot(x_range, stats.norm.pdf(x_range, resid_clean.mean(), resid_clean.std()),
               color=RED, linewidth=2, label='Normal')
axes[0, 1].set_title('Residual Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()

# ACF of residuals
acf_resid = acf(resid_clean, nlags=36)
axes[1, 0].bar(range(len(acf_resid)), acf_resid, color=BLUE, width=0.3)
axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
axes[1, 0].axhline(y=1.96/np.sqrt(len(resid_clean)), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=-1.96/np.sqrt(len(resid_clean)), color=RED, linestyle='--', alpha=0.7)
for lag in [12, 24, 36]:
    if lag < len(acf_resid):
        axes[1, 0].axvline(x=lag, color=ORANGE, linestyle=':', alpha=0.3)
axes[1, 0].set_title('ACF of Residuals', fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')

# Q-Q plot
(osm, osr), (slope, intercept, r) = stats.probplot(resid_clean, dist='norm', fit=True)
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

# Chart 6: ACF of original data (showing seasonal patterns)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

acf_orig = acf(passengers, nlags=48)
axes[0].bar(range(len(acf_orig)), acf_orig, color=BLUE, width=0.4)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].axhline(y=1.96/np.sqrt(len(passengers)), color=RED, linestyle='--', alpha=0.7)
axes[0].axhline(y=-1.96/np.sqrt(len(passengers)), color=RED, linestyle='--', alpha=0.7)
# Highlight seasonal lags
for lag in [12, 24, 36, 48]:
    if lag < len(acf_orig):
        axes[0].axvline(x=lag, color=ORANGE, linestyle=':', alpha=0.5, linewidth=2)
        axes[0].annotate(f'Lag {lag}', (lag, acf_orig[lag]+0.05), ha='center', fontsize=9, color=ORANGE)
axes[0].set_title('ACF of Original Series (Before Differencing)', fontweight='bold')
axes[0].set_xlabel('Lag')
axes[0].set_ylabel('ACF')
axes[0].set_ylim(-0.3, 1.1)

# Also show slow decay characteristic
acf_seasonal = acf(passengers, nlags=48)
seasonal_lags = [0, 12, 24, 36, 48]
axes[1].plot(seasonal_lags[:len([x for x in seasonal_lags if x < len(acf_seasonal)])],
             [acf_seasonal[l] for l in seasonal_lags if l < len(acf_seasonal)],
             'o-', color=BLUE, linewidth=2, markersize=10, label='ACF at seasonal lags')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].axhline(y=1.96/np.sqrt(len(passengers)), color=RED, linestyle='--', alpha=0.7, label='95% CI')
axes[1].axhline(y=-1.96/np.sqrt(len(passengers)), color=RED, linestyle='--', alpha=0.7)
axes[1].set_title('ACF Slow Decay at Seasonal Lags → Needs Differencing', fontweight='bold')
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('ACF')
axes[1].legend()
axes[1].set_xticks(seasonal_lags[:len([x for x in seasonal_lags if x < len(acf_seasonal)])])

plt.tight_layout()
plt.savefig('charts/ch4_acf_seasonality.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch4_acf_seasonality.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch4_acf_seasonality.pdf")

# Chart 7: Seasonal subseries plot
fig, ax = plt.subplots(figsize=(14, 6))

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
colors_months = plt.cm.tab10(np.linspace(0, 1, 12))

for i in range(12):
    month_data = passengers[passengers.index.month == i + 1]
    years = month_data.index.year
    ax.plot(years, month_data.values, 'o-', color=colors_months[i], linewidth=1.5,
            markersize=5, label=months[i])
    # Add horizontal mean line for each month
    mean_val = month_data.mean()
    ax.axhline(y=mean_val, xmin=(i)/12 + 0.01, xmax=(i+1)/12 - 0.01,
               color=colors_months[i], linestyle='--', alpha=0.3)

ax.set_title('Seasonal Subseries Plot: Each Month Across Years', fontweight='bold', fontsize=14)
ax.set_xlabel('Year')
ax.set_ylabel('Passengers (thousands)')
ax.legend(loc='upper left', ncol=6, fontsize=8)

plt.tight_layout()
plt.savefig('charts/ch4_seasonal_subseries.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch4_seasonal_subseries.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch4_seasonal_subseries.pdf")

# Chart 8: Seasonal box plot
fig, ax = plt.subplots(figsize=(12, 6))

month_data = [passengers[passengers.index.month == m].values for m in range(1, 13)]
bp = ax.boxplot(month_data, labels=months, patch_artist=True)

# Color boxes
colors_box = plt.cm.Blues(np.linspace(0.3, 0.9, 12))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_edgecolor(BLUE)

ax.set_title('Seasonal Box Plot: Distribution by Month', fontweight='bold', fontsize=14)
ax.set_xlabel('Month')
ax.set_ylabel('Passengers (thousands)')

# Add mean line
means = [np.mean(d) for d in month_data]
ax.plot(range(1, 13), means, 'o--', color=RED, linewidth=2, markersize=8, label='Monthly Mean')
ax.legend()

plt.tight_layout()
plt.savefig('charts/ch4_seasonal_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch4_seasonal_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch4_seasonal_boxplot.pdf")

# Chart 9: Effect of seasonal differencing
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original series
axes[0, 0].plot(passengers.index, passengers.values, color=BLUE, linewidth=1)
axes[0, 0].set_title('Original Series $Y_t$', fontweight='bold')
axes[0, 0].set_ylabel('Passengers')

# Regular difference only
diff1 = passengers.diff().dropna()
axes[0, 1].plot(diff1.index, diff1.values, color=GREEN, linewidth=1)
axes[0, 1].set_title('Regular Difference $\\Delta Y_t = Y_t - Y_{t-1}$', fontweight='bold')
axes[0, 1].set_ylabel('$\\Delta Y$')
axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Seasonal difference only
diff12 = passengers.diff(12).dropna()
axes[1, 0].plot(diff12.index, diff12.values, color=ORANGE, linewidth=1)
axes[1, 0].set_title('Seasonal Difference $\\Delta_{12} Y_t = Y_t - Y_{t-12}$', fontweight='bold')
axes[1, 0].set_ylabel('$\\Delta_{12} Y$')
axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Both differences
diff_both = passengers.diff().diff(12).dropna()
axes[1, 1].plot(diff_both.index, diff_both.values, color=RED, linewidth=1)
axes[1, 1].set_title('Both Differences $\\Delta\\Delta_{12} Y_t$', fontweight='bold')
axes[1, 1].set_ylabel('$\\Delta\\Delta_{12} Y$')
axes[1, 1].set_xlabel('Date')
axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('charts/ch4_differencing_effect.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch4_differencing_effect.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch4_differencing_effect.pdf")

# Chart 10: ACF comparison before and after differencing
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ACF of original
acf_orig = acf(passengers, nlags=36)
axes[0, 0].bar(range(len(acf_orig)), acf_orig, color=BLUE, width=0.4)
axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
axes[0, 0].axhline(y=1.96/np.sqrt(len(passengers)), color=RED, linestyle='--', alpha=0.7)
axes[0, 0].axhline(y=-1.96/np.sqrt(len(passengers)), color=RED, linestyle='--', alpha=0.7)
axes[0, 0].set_title('ACF of Original $Y_t$ (Slow Decay)', fontweight='bold')
axes[0, 0].set_ylabel('ACF')
axes[0, 0].set_ylim(-0.4, 1.1)

# ACF after regular difference
acf_diff1 = acf(diff1.dropna(), nlags=36)
axes[0, 1].bar(range(len(acf_diff1)), acf_diff1, color=GREEN, width=0.4)
axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
axes[0, 1].axhline(y=1.96/np.sqrt(len(diff1)), color=RED, linestyle='--', alpha=0.7)
axes[0, 1].axhline(y=-1.96/np.sqrt(len(diff1)), color=RED, linestyle='--', alpha=0.7)
for lag in [12, 24, 36]:
    if lag < len(acf_diff1):
        axes[0, 1].axvline(x=lag, color=ORANGE, linestyle=':', alpha=0.5)
axes[0, 1].set_title('ACF of $\\Delta Y_t$ (Seasonal Spikes Remain)', fontweight='bold')
axes[0, 1].set_ylabel('ACF')
axes[0, 1].set_ylim(-0.4, 1.1)

# ACF after seasonal difference
acf_diff12 = acf(diff12.dropna(), nlags=36)
axes[1, 0].bar(range(len(acf_diff12)), acf_diff12, color=ORANGE, width=0.4)
axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
axes[1, 0].axhline(y=1.96/np.sqrt(len(diff12)), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=-1.96/np.sqrt(len(diff12)), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].set_title('ACF of $\\Delta_{12} Y_t$ (Trend Decay Remains)', fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')
axes[1, 0].set_ylim(-0.4, 1.1)

# ACF after both differences
acf_both = acf(diff_both.dropna(), nlags=36)
axes[1, 1].bar(range(len(acf_both)), acf_both, color=RED, width=0.4)
axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
axes[1, 1].axhline(y=1.96/np.sqrt(len(diff_both)), color=RED, linestyle='--', alpha=0.7)
axes[1, 1].axhline(y=-1.96/np.sqrt(len(diff_both)), color=RED, linestyle='--', alpha=0.7)
axes[1, 1].set_title('ACF of $\\Delta\\Delta_{12} Y_t$ (Stationary)', fontweight='bold')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('ACF')
axes[1, 1].set_ylim(-0.4, 0.6)

plt.tight_layout()
plt.savefig('charts/ch4_acf_differencing.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch4_acf_differencing.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch4_acf_differencing.pdf")

print("\nAll Chapter 4 charts generated successfully!")
