"""
TSA_ch2_python_selection
========================
Python Exercise 2: Model Selection

Tasks:
1. Load a time series and check stationarity (ADF test)
2. Compare AIC/BIC for AR(1), MA(1), ARMA(1,1), ARMA(2,1)
3. Select the best model
4. Generate forecasts with confidence intervals
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_process import ArmaProcess

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


np.random.seed(42)

print("=" * 60)
print("PYTHON EXERCISE 2: MODEL SELECTION")
print("=" * 60)

# Generate an ARMA(1,1) series for demonstration
ar_params = np.array([1, -0.7])
ma_params = np.array([1, 0.4])
arma_process = ArmaProcess(ar_params, ma_params)
y = arma_process.generate_sample(nsample=300)

# Step 1: Stationarity test
adf_result = adfuller(y)
print(f"\n1. ADF Test for Stationarity:")
print(f"   ADF Statistic: {adf_result[0]:.4f}")
print(f"   p-value: {adf_result[1]:.6f}")
print(f"   → {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'}")

# Step 2: Compare models
models = {
    'AR(1)':     (1, 0, 0),
    'MA(1)':     (0, 0, 1),
    'ARMA(1,1)': (1, 0, 1),
    'ARMA(2,1)': (2, 0, 1),
}

print(f"\n2. Model Comparison:")
print(f"   {'Model':<12} {'AIC':>10} {'BIC':>10}")
print("   " + "-" * 34)

results = {}
for name, order in models.items():
    try:
        fit = ARIMA(y, order=order).fit()
        results[name] = fit
        print(f"   {name:<12} {fit.aic:>10.2f} {fit.bic:>10.2f}")
    except Exception as e:
        print(f"   {name:<12} {'FAILED':>10}")

# Step 3: Select best model
best_aic = min(results, key=lambda k: results[k].aic)
best_bic = min(results, key=lambda k: results[k].bic)
print(f"\n3. Best Model:")
print(f"   By AIC: {best_aic} (AIC = {results[best_aic].aic:.2f})")
print(f"   By BIC: {best_bic} (BIC = {results[best_bic].bic:.2f})")

# Step 4: Forecast with best model
best_model = results[best_aic]
forecast = best_model.get_forecast(steps=20)
fc_mean = forecast.predicted_mean
fc_ci = np.array(forecast.conf_int(alpha=0.05))
fc_mean_arr = np.array(fc_mean)

print(f"\n4. Forecast (next 5 periods):")
for i in range(5):
    print(f"   h={i+1}: {fc_mean_arr[i]:.4f} [{fc_ci[i, 0]:.4f}, {fc_ci[i, 1]:.4f}]")
