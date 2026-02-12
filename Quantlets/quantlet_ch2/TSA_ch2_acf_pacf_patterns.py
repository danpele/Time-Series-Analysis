"""
TSA_ch2_acf_pacf_patterns
=========================
Demonstrate ACF/PACF identification patterns for AR, MA, ARMA.

Description:
- Simulate AR(1), MA(1), ARMA(1,1)
- Plot ACF and PACF for each
- Show characteristic patterns for model identification
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess

# Set seed
np.random.seed(42)

n = 500

# AR(1): phi = 0.7
ar1_ar = np.array([1, -0.7])
ar1_ma = np.array([1])
ar1_process = ArmaProcess(ar1_ar, ar1_ma)
ar1_data = ar1_process.generate_sample(nsample=n)

# MA(1): theta = 0.7
ma1_ar = np.array([1])
ma1_ma = np.array([1, 0.7])
ma1_process = ArmaProcess(ma1_ar, ma1_ma)
ma1_data = ma1_process.generate_sample(nsample=n)

# ARMA(1,1): phi = 0.7, theta = 0.4
arma_ar = np.array([1, -0.7])
arma_ma = np.array([1, 0.4])
arma_process = ArmaProcess(arma_ar, arma_ma)
arma_data = arma_process.generate_sample(nsample=n)

# Create figure
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# AR(1)
plot_acf(ar1_data, lags=15, ax=axes[0, 0])
axes[0, 0].set_title('AR(1): ACF - Exponential Decay', fontsize=11)
plot_pacf(ar1_data, lags=15, ax=axes[0, 1], method='ywm')
axes[0, 1].set_title('AR(1): PACF - Cuts off at lag 1', fontsize=11)

# MA(1)
plot_acf(ma1_data, lags=15, ax=axes[1, 0])
axes[1, 0].set_title('MA(1): ACF - Cuts off at lag 1', fontsize=11)
plot_pacf(ma1_data, lags=15, ax=axes[1, 1], method='ywm')
axes[1, 1].set_title('MA(1): PACF - Exponential Decay', fontsize=11)

# ARMA(1,1)
plot_acf(arma_data, lags=15, ax=axes[2, 0])
axes[2, 0].set_title('ARMA(1,1): ACF - Decays (no cutoff)', fontsize=11)
plot_pacf(arma_data, lags=15, ax=axes[2, 1], method='ywm')
axes[2, 1].set_title('ARMA(1,1): PACF - Decays (no cutoff)', fontsize=11)

plt.tight_layout()
plt.savefig('../../charts/ch2_acf_pacf_patterns.pdf', bbox_inches='tight')
plt.show()

print("Model Identification Guide:")
print("  AR(p):    ACF decays,    PACF cuts off at lag p")
print("  MA(q):    ACF cuts off,  PACF decays")
print("  ARMA:     Both decay,    Use AIC/BIC for selection")
