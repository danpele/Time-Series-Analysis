import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Parameters
T = 1000
np.random.seed(42)
omega = 0.00002
alpha = 0.08
beta = 0.87
lam = 0.94

# Simulate GARCH(1,1) returns
eps = np.random.standard_normal(T)
eps[300] *= 4  # large negative shock

sigma2 = np.zeros(T)
r = np.zeros(T)
sigma2[0] = omega / (1 - alpha - beta)  # start at unconditional variance
for t in range(1, T):
    sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
    r[t] = np.sqrt(sigma2[t]) * eps[t]

# GARCH conditional variance (re-estimate on realized returns)
garch_var = np.zeros(T)
garch_var[0] = omega / (1 - alpha - beta)
for t in range(1, T):
    garch_var[t] = omega + alpha * r[t-1]**2 + beta * garch_var[t-1]

# EWMA conditional variance
ewma_var = np.zeros(T)
ewma_var[0] = np.var(r[:50]) if np.var(r[:50]) > 0 else omega / (1 - alpha - beta)
for t in range(1, T):
    ewma_var[t] = lam * ewma_var[t-1] + (1 - lam) * r[t-1]**2

garch_vol = np.sqrt(garch_var)
ewma_vol = np.sqrt(ewma_var)
uncond_vol = np.sqrt(omega / (1 - alpha - beta))

# Plot
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Top panel: returns + bands
ax1.plot(r, color='0.7', linewidth=0.5, label='Returns')
ax1.plot(2 * ewma_vol, color='#2166ac', linestyle='--', linewidth=1.3, label=r'EWMA $\pm 2\sigma$')
ax1.plot(-2 * ewma_vol, color='#2166ac', linestyle='--', linewidth=1.3)
ax1.plot(2 * garch_vol, color='#b2182b', linestyle='--', linewidth=1.3, label=r'GARCH $\pm 2\sigma$')
ax1.plot(-2 * garch_vol, color='#b2182b', linestyle='--', linewidth=1.3)
ax1.set_title('Simulated returns with EWMA (blue) vs GARCH (red) bands', fontweight='bold')
ax1.set_ylabel('Return')
## legend moved to bottom of figure
ax1.grid(True, alpha=0.3)
ax1.axvline(300, color='0.5', linestyle=':', linewidth=0.8, alpha=0.5)

# Bottom panel: conditional volatilities
ax2.plot(ewma_vol, color='#2166ac', linewidth=1.5, label=r'EWMA ($\lambda=0.94$)')
ax2.plot(garch_vol, color='#b2182b', linewidth=1.5, label='GARCH(1,1)')
ax2.axhline(uncond_vol, color='black', linestyle='--', linewidth=1.2, label=r'$\bar{\sigma}$ unconditional')
ax2.set_title('Conditional volatility: EWMA vs GARCH', fontweight='bold')
ax2.set_xlabel('t')
ax2.set_ylabel(r'$\sigma_t$')
## legend moved to bottom of figure
ax2.grid(True, alpha=0.3)
ax2.axvline(300, color='0.5', linestyle=':', linewidth=0.8, alpha=0.5)

# Transparent background
fig.patch.set_alpha(0)
ax1.patch.set_alpha(0)
ax2.patch.set_alpha(0)

# Combined legend at bottom
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
# Deduplicate
seen = set()
handles, labels = [], []
for h, l in zip(h1 + h2, l1 + l2):
    if l not in seen:
        seen.add(l)
        handles.append(h)
        labels.append(l)
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=5, frameon=False, fontsize=11)

plt.tight_layout(rect=[0, 0.06, 1, 1], h_pad=1.5)
plt.savefig('/Users/danielpele/Documents/TSA/charts/garch_ewma_comparison.pdf',
            bbox_inches='tight', dpi=150, transparent=True)
plt.close()
print('Chart saved.')
