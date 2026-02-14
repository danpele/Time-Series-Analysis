"""
TSA_ch2_case_study
==================
Complete ARMA modeling case study with sunspot data.

Description:
- Load and visualize sunspot data
- ACF/PACF analysis
- Model selection with AIC/BIC
- Diagnostic checking
- Forecasting
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd

# Load sunspot data
from statsmodels.datasets import sunspots
data = sunspots.load_pandas().data
y = data['SUNACTIVITY'].values

# Create comprehensive figure
fig = plt.figure(figsize=(14, 10))

# Plot 1: Raw data
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(y, 'b-', linewidth=0.8)
ax1.set_title('Sunspot Numbers (1700-2008)', fontsize=11)
ax1.set_xlabel('Year')
ax1.set_ylabel('Sunspot Count')
ax1.grid(True, alpha=0.3)

# Plot 2: ACF
ax2 = fig.add_subplot(2, 2, 2)
plot_acf(y, lags=30, ax=ax2)
ax2.set_title('ACF - Suggests AR or ARMA', fontsize=11)

# Fit AR(2) model
model = ARIMA(y, order=(2, 0, 0)).fit()

# Plot 3: Residual ACF
ax3 = fig.add_subplot(2, 2, 3)
plot_acf(model.resid, lags=20, ax=ax3)
ax3.set_title('Residual ACF (AR(2) fit)', fontsize=11)

# Plot 4: Forecast
ax4 = fig.add_subplot(2, 2, 4)
forecast = model.get_forecast(steps=20)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

ax4.plot(y[-50:], 'b-', linewidth=1, label='Observed')
ax4.plot(range(len(y[-50:]), len(y[-50:]) + 20), forecast_mean, 'r-', linewidth=2, label='Forecast')
ax4.fill_between(range(len(y[-50:]), len(y[-50:]) + 20),
                  forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                  color='red', alpha=0.2, label='95% CI')
ax4.set_title('20-Year Forecast', fontsize=11)
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../charts/ch2_case_study.pdf', bbox_inches='tight')
plt.show()

# Model summary
print("AR(2) Model Summary:")
print(f"  phi1 = {model.params[1]:.4f}")
print(f"  phi2 = {model.params[2]:.4f}")
print(f"  AIC = {model.aic:.2f}")
print(f"  BIC = {model.bic:.2f}")

# Ljung-Box test
lb_test = acorr_ljungbox(model.resid, lags=[10], return_df=True)
print(f"  Ljung-Box(10) p-value = {lb_test['lb_pvalue'].values[0]:.4f}")
