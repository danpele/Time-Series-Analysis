import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

np.random.seed(42)

# --- Parameters ---
T = 1000
p11, p22 = 0.98, 0.95  # transition probabilities
omega = [0.00001, 0.0001]
alpha = [0.05, 0.15]
beta = [0.90, 0.80]

# --- Generate Markov chain regime sequence ---
regimes = np.zeros(T, dtype=int)
regimes[0] = 0  # start in calm regime
for t in range(1, T):
    if regimes[t-1] == 0:
        regimes[t] = 0 if np.random.rand() < p11 else 1
    else:
        regimes[t] = 1 if np.random.rand() < p22 else 0

# --- Simulate GARCH(1,1) with regime-dependent parameters ---
sigma2 = np.zeros(T)
returns = np.zeros(T)
sigma2[0] = omega[regimes[0]] / (1 - alpha[regimes[0]] - beta[regimes[0]])
returns[0] = np.sqrt(sigma2[0]) * np.random.randn()

for t in range(1, T):
    s = regimes[t]
    sigma2[t] = omega[s] + alpha[s] * returns[t-1]**2 + beta[s] * sigma2[t-1]
    returns[t] = np.sqrt(sigma2[t]) * np.random.randn()

sigma_t = np.sqrt(sigma2)

# --- Smoothed regime probability (simple moving average filter as proxy) ---
regime_prob = uniform_filter1d(regimes.astype(float), size=20)

# --- Plot ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
plt.rcParams.update({'font.size': 13})
time = np.arange(T)

# Colors
c_calm = '#2166AC'
c_turb = '#D6604D'

# Top: returns colored by regime
ax = axes[0]
for t in range(T):
    color = c_turb if regimes[t] == 1 else c_calm
    ax.bar(t, returns[t], width=1.0, color=color, edgecolor='none', alpha=0.85)
ax.set_ylabel('Returns', fontsize=14)
ax.set_title('Simulated returns: calm regime (blue) vs turbulent (red)',
             fontsize=15, fontweight='bold', pad=10)
ax.tick_params(labelsize=12)
ax.axhline(0, color='grey', lw=0.5)

# Middle: conditional volatility colored by regime
ax = axes[1]
for t in range(1, T):
    color = c_turb if regimes[t] == 1 else c_calm
    ax.plot([t-1, t], [sigma_t[t-1], sigma_t[t]], color=color, lw=1.2)
ax.set_ylabel(r'$\sigma_t$', fontsize=14)
ax.set_title(r'Conditional volatility $\sigma_t$ by regime',
             fontsize=15, fontweight='bold', pad=10)
ax.tick_params(labelsize=12)

# Bottom: probability of turbulent regime
ax = axes[2]
ax.fill_between(time, 0, regime_prob, color=c_turb, alpha=0.3, label=r'$P(s_t=2)$')
ax.plot(time, regime_prob, color=c_turb, lw=1.2)
ax.set_ylabel(r'$P(s_t = 2)$', fontsize=14)
ax.set_xlabel('Time (t)', fontsize=14)
ax.set_title(r'Turbulent regime probability $P(s_t = 2)$',
             fontsize=15, fontweight='bold', pad=10)
ax.set_ylim(-0.02, 1.05)
ax.tick_params(labelsize=12)

# Transparent background
fig.patch.set_alpha(0)
for ax in axes:
    ax.patch.set_alpha(0)

# Combined legend at bottom
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_handles = [
    Patch(facecolor=c_calm, alpha=0.85, label='Calm regime ($s_t=1$)'),
    Patch(facecolor=c_turb, alpha=0.85, label='Turbulent regime ($s_t=2$)'),
    Line2D([0], [0], color=c_turb, lw=1.2, label=r'$P(s_t=2)$'),
]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=3, frameon=False, fontsize=12)

plt.tight_layout(rect=[0, 0.06, 1, 1], h_pad=1.5)
plt.savefig('/Users/danielpele/Documents/TSA/charts/garch_markov_switching.pdf',
            bbox_inches='tight', dpi=150, transparent=True)
plt.close()
print('Chart saved.')
