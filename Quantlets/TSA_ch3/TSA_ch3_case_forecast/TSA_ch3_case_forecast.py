"""
TSA_ch3_case_forecast
=====================
ARIMA Forecasting with Confidence Intervals

Data Source: FRED (GDPC1) - US Real GDP
8-quarter ahead forecast with 95% confidence bands
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from statsmodels.tsa.arima.model import ARIMA

# Chart style settings - Nature journal quality
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75

BLUE, RED = '#1A3A6E', '#DC3545'

# Get data
gdp = pdr.get_data_fred('GDPC1', start='1960-01-01', end='2024-09-30')
gdp_data = gdp['GDPC1'].dropna()
log_gdp = np.log(gdp_data)

# Fit model
model = ARIMA(log_gdp, order=(1, 1, 1))
fit = model.fit()

# Forecast
forecast_steps = 8
forecast = fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Create future dates
last_date = gdp_data.index[-1]
future_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq='Q')[1:]

fig, ax = plt.subplots(figsize=(14, 6))

# Plot last 40 observations
plot_data = log_gdp.iloc[-40:]
ax.plot(plot_data.index, plot_data.values, color=BLUE, linewidth=2, label='Observed')

# Forecast
ax.plot(future_dates, forecast_mean.values, color=RED, linewidth=2,
        linestyle='--', label='Forecast')
ax.fill_between(future_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                color=RED, alpha=0.2, label='95% CI')

# Visual separator between historical and forecast
split_point = plot_data.index[-1]
ax.axvline(x=split_point, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
y_pos = ax.get_ylim()[1] - 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
ax.text(split_point, y_pos, '  Forecast ', fontsize=9, ha='left', va='top',
        color='black', fontweight='bold', alpha=0.8)

ax.set_title('US Real GDP: ARIMA Forecast', fontweight='bold', fontsize=14)
ax.set_xlabel('Date')
ax.set_ylabel('Log(GDP)')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)
ax.text(0.02, 0.98, 'Model: ARIMA(1,1,1)\nSource: FRED (GDPC1)',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        style='italic', color='gray')

plt.tight_layout()
plt.savefig('ch3_case_forecast.pdf', dpi=300, bbox_inches='tight')
plt.show()
