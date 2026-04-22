import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Clean style for Beamer
mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

# Parameters
T = 1000
omega_bar = 0.0002
alpha = 0.05
beta = 0.85
rho = 0.995
phi = 0.03
np.random.seed(42)

# Pre-allocate
sigma2 = np.zeros(T)
q = np.zeros(T)
eps = np.zeros(T)
returns = np.zeros(T)

# Initial values
sigma2[0] = omega_bar
q[0] = omega_bar

z = np.random.standard_normal(T)

for t in range(1, T):
    # Permanent component
    q[t] = omega_bar + rho * (q[t-1] - omega_bar) + phi * (eps[t-1]**2 - sigma2[t-1])
    # Transitory equation: conditional variance
    sigma2[t] = q[t] + alpha * (eps[t-1]**2 - q[t-1]) + beta * (sigma2[t-1] - q[t-1])
    # Floor to avoid negative variance
    sigma2[t] = max(sigma2[t], 1e-8)
    q[t] = max(q[t], 1e-8)
    # Simulate
    eps[t] = np.sqrt(sigma2[t]) * z[t]
    returns[t] = eps[t]

sigma = np.sqrt(sigma2)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top panel: returns with +-2sigma bands
ax1.plot(returns, color='gray', linewidth=0.5, alpha=0.8)
ax1.plot(2 * sigma, color='red', linewidth=1.2, linestyle='--', label=r'$\pm 2\sigma_t$')
ax1.plot(-2 * sigma, color='red', linewidth=1.2, linestyle='--')
ax1.set_title('CGARCH: Simulated returns with volatility bands', fontweight='bold')
ax1.set_ylabel('Return')
## legend moved to bottom of figure
ax1.axhline(0, color='black', linewidth=0.5, alpha=0.4)

# Bottom panel: permanent component vs conditional variance
ax2.plot(q, color='#2060B0', linewidth=2.2, label='Permanent component $q_t$')
ax2.plot(sigma2, color='red', linewidth=0.9, alpha=0.85, label='Conditional variance $\\sigma^2_t$')
ax2.set_title('CGARCH: Permanent component vs conditional variance', fontweight='bold')
ax2.set_ylabel('Variance')
ax2.set_xlabel('Time')
## legend moved to bottom of figure

# Transparent background
fig.patch.set_alpha(0)
ax1.patch.set_alpha(0)
ax2.patch.set_alpha(0)

# Combined legend at bottom
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=4, frameon=False, fontsize=12)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('/Users/danielpele/Documents/TSA/charts/garch_cgarch_simulation.pdf',
            bbox_inches='tight', dpi=150, transparent=True)
plt.close()
print("Chart saved successfully.")
