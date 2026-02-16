"""
TSA_ch2_white_noise
===================
Visualize white noise process and its ACF properties.

Description:
- Generate white noise series
- Plot time series and ACF
- Demonstrate no autocorrelation structure
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# --- Standard chart style ---
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False


# Set seed for reproducibility
np.random.seed(42)

# Generate white noise
n = 200
sigma = 1.0
epsilon = np.random.normal(0, sigma, n)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: White noise series
ax1 = axes[0]
ax1.plot(epsilon, 'b-', linewidth=0.8, alpha=0.8)
ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
ax1.axhline(y=2*sigma, color='gray', linestyle=':', alpha=0.7)
ax1.axhline(y=-2*sigma, color='gray', linestyle=':', alpha=0.7)
ax1.set_title('White Noise Process', fontsize=12)
ax1.set_xlabel('Time')
ax1.set_ylabel(r'$\varepsilon_t$')
ax1.grid(True, alpha=0.3)

# Plot 2: ACF
ax2 = axes[1]
plot_acf(epsilon, lags=20, ax=ax2)
ax2.set_title('ACF of White Noise', fontsize=12)
ax2.set_xlabel('Lag')

plt.tight_layout()
plt.savefig('../../charts/ch2_white_noise.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("White Noise Properties:")
print(f"  Mean: {np.mean(epsilon):.4f} (theoretical: 0)")
print(f"  Variance: {np.var(epsilon):.4f} (theoretical: {sigma**2})")
print(f"  ACF(1): {np.corrcoef(epsilon[:-1], epsilon[1:])[0,1]:.4f} (theoretical: 0)")
