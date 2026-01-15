#!/usr/bin/env python3
"""Generate charts for Chapter 3: ARIMA Models"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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

np.random.seed(42)

# Try to get real data, fallback to synthetic
try:
    import pandas_datareader as pdr
    # US Real GDP (quarterly)
    gdp = pdr.get_data_fred('GDPC1', start='1990-01-01', end='2024-06-30')
    gdp_data = gdp['GDPC1'].dropna()
    data_source = "US Real GDP (Billions of Chained 2017 Dollars)"
except:
    # Synthetic GDP-like data
    n = 140
    dates = pd.date_range(start='1990-01-01', periods=n, freq='Q')
    trend = np.linspace(8000, 22000, n)
    noise = np.cumsum(np.random.normal(0, 100, n))
    gdp_data = pd.Series(trend + noise, index=dates)
    data_source = "US Real GDP (Simulated)"

print(f"Using: {data_source}")

# Chart 1: Non-stationary time series (GDP in levels)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(gdp_data.index, gdp_data.values, color=BLUE, linewidth=1.5)
ax.set_title('US Real GDP: A Non-Stationary Time Series', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Billions of 2017 Dollars')
plt.tight_layout()
plt.savefig('charts/ch3_gdp_levels.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_gdp_levels.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_gdp_levels.pdf")

# Chart 2: GDP growth rate (first difference of log)
gdp_growth = np.log(gdp_data).diff().dropna() * 100  # Percentage growth

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Levels
axes[0].plot(gdp_data.index, gdp_data.values, color=BLUE, linewidth=1.5)
axes[0].set_title('GDP in Levels (Non-Stationary)', fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Billions $')

# Growth rate
axes[1].plot(gdp_growth.index, gdp_growth.values, color=GREEN, linewidth=1)
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].axhline(y=gdp_growth.mean(), color=RED, linestyle='--', alpha=0.7, label=f'Mean = {gdp_growth.mean():.2f}%')
axes[1].set_title('GDP Growth Rate (Stationary)', fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Growth Rate (%)')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig('charts/ch3_differencing.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_differencing.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_differencing.pdf")

# Chart 3: ACF/PACF comparison (levels vs differenced)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ACF of levels
acf_levels = acf(gdp_data.dropna(), nlags=20)
axes[0, 0].bar(range(len(acf_levels)), acf_levels, color=BLUE, width=0.3)
axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
axes[0, 0].axhline(y=1.96/np.sqrt(len(gdp_data)), color=RED, linestyle='--', alpha=0.7)
axes[0, 0].axhline(y=-1.96/np.sqrt(len(gdp_data)), color=RED, linestyle='--', alpha=0.7)
axes[0, 0].set_title('ACF of GDP Levels', fontweight='bold')
axes[0, 0].set_xlabel('Lag')
axes[0, 0].set_ylabel('ACF')

# PACF of levels
pacf_levels = pacf(gdp_data.dropna(), nlags=20)
axes[0, 1].bar(range(len(pacf_levels)), pacf_levels, color=BLUE, width=0.3)
axes[0, 1].axhline(y=0, color='black', linewidth=0.5)
axes[0, 1].axhline(y=1.96/np.sqrt(len(gdp_data)), color=RED, linestyle='--', alpha=0.7)
axes[0, 1].axhline(y=-1.96/np.sqrt(len(gdp_data)), color=RED, linestyle='--', alpha=0.7)
axes[0, 1].set_title('PACF of GDP Levels', fontweight='bold')
axes[0, 1].set_xlabel('Lag')
axes[0, 1].set_ylabel('PACF')

# ACF of growth
acf_diff = acf(gdp_growth.dropna(), nlags=20)
axes[1, 0].bar(range(len(acf_diff)), acf_diff, color=GREEN, width=0.3)
axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
axes[1, 0].axhline(y=1.96/np.sqrt(len(gdp_growth)), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=-1.96/np.sqrt(len(gdp_growth)), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].set_title('ACF of GDP Growth (Differenced)', fontweight='bold')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')

# PACF of growth
pacf_diff = pacf(gdp_growth.dropna(), nlags=20)
axes[1, 1].bar(range(len(pacf_diff)), pacf_diff, color=GREEN, width=0.3)
axes[1, 1].axhline(y=0, color='black', linewidth=0.5)
axes[1, 1].axhline(y=1.96/np.sqrt(len(gdp_growth)), color=RED, linestyle='--', alpha=0.7)
axes[1, 1].axhline(y=-1.96/np.sqrt(len(gdp_growth)), color=RED, linestyle='--', alpha=0.7)
axes[1, 1].set_title('PACF of GDP Growth (Differenced)', fontweight='bold')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('PACF')

plt.tight_layout()
plt.savefig('charts/ch3_acf_pacf.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_acf_pacf.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_acf_pacf.pdf")

# Chart 4: ARIMA forecasting with train/test
train = gdp_data[:-12]
test = gdp_data[-12:]

# Fit ARIMA(1,1,1) on log GDP
log_train = np.log(train)
log_test = np.log(test)

model = ARIMA(log_train, order=(1, 1, 1))
fitted = model.fit()

# Forecast
forecast = fitted.get_forecast(steps=len(test))
forecast_mean = np.exp(forecast.predicted_mean)
conf_int = np.exp(forecast.conf_int())

fig, ax = plt.subplots(figsize=(14, 6))

# Training data (last 40 obs)
ax.plot(train.index[-40:], train.values[-40:], color=BLUE, linewidth=1.5, label='Training Data')

# Actual test data
ax.plot(test.index, test.values, color=GREEN, linewidth=2, label='Actual', marker='o', markersize=5)

# Forecast
ax.plot(test.index, forecast_mean.values, color=RED, linewidth=2, linestyle='--', label='ARIMA(1,1,1) Forecast')

# Confidence interval
ax.fill_between(test.index, conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values,
                color=RED, alpha=0.2, label='95% CI')

ax.axvline(x=train.index[-1], color='gray', linestyle=':', alpha=0.7, linewidth=2)
ax.set_title('US GDP: ARIMA(1,1,1) Forecast vs Actual', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Billions of 2017 Dollars')
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('charts/ch3_arima_forecast.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_arima_forecast.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_arima_forecast.pdf")

# Chart 5: Unit root test visualization (ADF results)
adf_levels = adfuller(gdp_data.dropna(), autolag='AIC')
adf_diff = adfuller(gdp_growth.dropna(), autolag='AIC')

fig, ax = plt.subplots(figsize=(10, 6))

# Create bar chart for ADF statistics vs critical values
categories = ['GDP Levels', 'GDP Growth\n(Differenced)', '1% Critical', '5% Critical', '10% Critical']
values = [adf_levels[0], adf_diff[0], adf_levels[4]['1%'], adf_levels[4]['5%'], adf_levels[4]['10%']]
colors = [BLUE, GREEN, RED, RED, RED]

bars = ax.bar(categories, values, color=colors, width=0.6, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.5)

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}',
            ha='center', va='bottom' if height > 0 else 'top', fontsize=11)

ax.set_title('Augmented Dickey-Fuller Test Results', fontweight='bold', fontsize=14)
ax.set_ylabel('Test Statistic / Critical Value')
ax.set_ylim(min(values) - 1, max(values) + 1)

# Add interpretation
ax.text(0.02, 0.98, f'GDP Levels: p-value = {adf_levels[1]:.4f} (Non-stationary)',
        transform=ax.transAxes, fontsize=10, verticalalignment='top', color=BLUE)
ax.text(0.02, 0.92, f'GDP Growth: p-value = {adf_diff[1]:.4f} (Stationary)',
        transform=ax.transAxes, fontsize=10, verticalalignment='top', color=GREEN)

plt.tight_layout()
plt.savefig('charts/ch3_adf_test.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_adf_test.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_adf_test.pdf")

# Chart 6: Residual diagnostics
resid = fitted.resid

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residuals over time
axes[0, 0].plot(resid.index, resid.values, color=BLUE, linewidth=0.8)
axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[0, 0].set_title('ARIMA Residuals', fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Residual')

# Histogram
axes[0, 1].hist(resid, bins=25, density=True, color=BLUE, alpha=0.7, edgecolor='white')
from scipy import stats
x_range = np.linspace(resid.min(), resid.max(), 100)
axes[0, 1].plot(x_range, stats.norm.pdf(x_range, resid.mean(), resid.std()),
               color=RED, linewidth=2, label='Normal')
axes[0, 1].set_title('Residual Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Residual')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()

# ACF of residuals
acf_resid = acf(resid.dropna(), nlags=20)
axes[1, 0].bar(range(len(acf_resid)), acf_resid, color=BLUE, width=0.3)
axes[1, 0].axhline(y=0, color='black', linewidth=0.5)
axes[1, 0].axhline(y=1.96/np.sqrt(len(resid)), color=RED, linestyle='--', alpha=0.7)
axes[1, 0].axhline(y=-1.96/np.sqrt(len(resid)), color=RED, linestyle='--', alpha=0.7)
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
plt.savefig('charts/ch3_diagnostics.pdf', dpi=300, bbox_inches='tight')
plt.savefig('charts/ch3_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ch3_diagnostics.pdf")

print("\nAll Chapter 3 charts generated successfully!")
