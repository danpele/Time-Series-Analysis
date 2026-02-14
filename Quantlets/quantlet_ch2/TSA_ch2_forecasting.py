"""
TSA_ch2_forecasting
===================
Forecasting with ARMA Models

This script demonstrates:
- Point forecasts from AR, MA, and ARMA models
- Forecast confidence intervals
- Mean reversion in stationary models
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess

# Set random seed
np.random.seed(42)

n = 200
h = 20  # forecast horizon

print("=" * 60)
print("FORECASTING WITH ARMA MODELS")
print("=" * 60)

print("""
Key Forecasting Formulas:

AR(1): X_t = c + φX_{t-1} + ε_t
  Point forecast: X̂_{n+h|n} = μ + φ^h(X_n - μ)
  As h → ∞: X̂ → μ (mean reversion)

MA(1): X_t = ε_t + θε_{t-1}
  X̂_{n+1|n} = θε_n
  X̂_{n+h|n} = 0 for h > 1 (no long memory)

Forecast Variance:
  Grows with horizon (uncertainty increases)
  MSFE(h) = σ² × Σ(ψ_j²) for j=0 to h-1
""")

# Generate and fit AR(1)
phi_ar = 0.8
ar = np.array([1, -phi_ar])
ma = np.array([1])
ar_process = ArmaProcess(ar, ma)
data_ar = ar_process.generate_sample(nsample=n)

# Fit model
model_ar = ARIMA(data_ar, order=(1, 0, 0)).fit()

# Generate forecasts
forecast_ar = model_ar.get_forecast(steps=h)
mean_ar = forecast_ar.predicted_mean
ci_ar = forecast_ar.conf_int(alpha=0.05)

# Theoretical long-run mean
mu_ar = model_ar.params[0] / (1 - model_ar.params[1]) if abs(model_ar.params[1]) < 1 else 0

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: AR(1) forecast
ax1 = axes[0, 0]
ax1.plot(range(n), data_ar, 'b-', linewidth=0.8, alpha=0.8, label='Observed')
forecast_idx = range(n, n + h)
ax1.plot(forecast_idx, mean_ar, 'r-', linewidth=2, label='Forecast')
ax1.fill_between(forecast_idx, ci_ar.iloc[:, 0], ci_ar.iloc[:, 1],
                 alpha=0.2, color='red', label='95% CI')
ax1.axhline(y=mu_ar, color='green', linestyle='--', linewidth=2, label=f'μ = {mu_ar:.2f}')
ax1.axvline(x=n-1, color='gray', linestyle=':', alpha=0.5)
ax1.set_title(f'AR(1) Forecast: φ = {model_ar.params[1]:.2f}', fontsize=12)
ax1.set_xlabel('Time')
ax1.set_ylabel('Value')
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(n-50, n+h)

# Plot 2: Forecast convergence to mean
ax2 = axes[0, 1]
horizons = np.arange(1, h+1)
forecasts_from_high = mu_ar + phi_ar**horizons * (data_ar[-1] - mu_ar)
forecasts_from_low = mu_ar + phi_ar**horizons * (data_ar.min() - mu_ar)

ax2.plot(horizons, mean_ar, 'ro-', markersize=6, linewidth=2, label=f'From X_n = {data_ar[-1]:.2f}')
ax2.plot(horizons, forecasts_from_low, 'bs--', markersize=6, linewidth=2,
         label=f'From X_n = {data_ar.min():.2f}')
ax2.axhline(y=mu_ar, color='green', linestyle='-', linewidth=2, label=f'Long-run μ = {mu_ar:.2f}')
ax2.set_title('AR(1) Mean Reversion', fontsize=12)
ax2.set_xlabel('Forecast Horizon (h)')
ax2.set_ylabel('E[X_{n+h}|X_n]')
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax2.grid(True, alpha=0.3)

# Plot 3: Forecast variance growth
ax3 = axes[1, 0]
# Theoretical MSFE for AR(1)
sigma_sq = model_ar.params[2]  # residual variance
msfe = np.array([sigma_sq * sum(phi_ar**(2*j) for j in range(i+1)) for i in range(h)])
ci_width = 2 * 1.96 * np.sqrt(msfe)

ax3.plot(horizons, np.sqrt(msfe), 'b-o', markersize=6, linewidth=2, label='Forecast Std Dev')
ax3.plot(horizons, ci_width/2, 'r--', linewidth=2, label='95% CI half-width')
ax3.set_title('Forecast Uncertainty Grows with Horizon', fontsize=12)
ax3.set_xlabel('Forecast Horizon (h)')
ax3.set_ylabel('Standard Deviation')
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax3.grid(True, alpha=0.3)

# Plot 4: Comparison with MA(1)
ax4 = axes[1, 1]
# Generate MA(1) data
theta_ma = 0.6
ar_ma = np.array([1])
ma_ma = np.array([1, theta_ma])
ma_process = ArmaProcess(ar_ma, ma_ma)
data_ma = ma_process.generate_sample(nsample=n)

model_ma = ARIMA(data_ma, order=(0, 0, 1)).fit()
forecast_ma = model_ma.get_forecast(steps=h)
mean_ma = forecast_ma.predicted_mean

ax4.plot(horizons, mean_ar - mu_ar, 'b-o', markersize=6, linewidth=2, label='AR(1): slow decay')
ax4.plot(horizons, mean_ma - mean_ma.mean(), 'r-s', markersize=6, linewidth=2, label='MA(1): instant to mean')
ax4.axhline(y=0, color='gray', linestyle='--')
ax4.set_title('Forecast Decay: AR(1) vs MA(1)', fontsize=12)
ax4.set_xlabel('Forecast Horizon (h)')
ax4.set_ylabel('Deviation from Mean')
ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=False)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../../charts/ch2_forecasting.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("KEY FORECASTING INSIGHTS")
print("=" * 60)
print("""
1. AR Models: Gradual mean reversion
   - Forecast decays toward μ as φ^h
   - Persistent processes (φ ≈ 1) decay slowly

2. MA Models: Quick mean reversion
   - Forecast = μ after q steps
   - "Short memory" models

3. ARMA Models: Combined behavior
   - Initial dynamics from MA component
   - Long-run decay from AR component

4. Confidence Intervals:
   - ALWAYS expand with horizon
   - Reflect increasing uncertainty
   - Wider for more persistent processes

5. Practical Consideration:
   - Long-horizon forecasts → unconditional mean
   - ARMA best for short-term forecasting
""")
