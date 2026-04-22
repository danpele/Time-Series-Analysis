"""
Generate a 2x2 APARCH simulation comparison chart.
"""

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

np.random.seed(42)

T = 1000
alpha = 0.08
beta = 0.88

configs = [
    {"delta": 2, "gamma": 0.0, "omega": 0.00001, "title": "Standard GARCH(1,1)"},
    {"delta": 2, "gamma": 0.5, "omega": 0.00001, "title": "GJR-GARCH(1,1)"},
    {"delta": 1, "gamma": 0.0, "omega": 0.001,   "title": "AVGARCH / Taylor(1,1)"},
    {"delta": 1, "gamma": 0.5, "omega": 0.001,   "title": "TGARCH(1,1)"},
]

# Single innovation sequence for all panels
z = np.random.standard_normal(T)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for ax, cfg in zip(axes, configs):
    delta = cfg["delta"]
    gamma = cfg["gamma"]
    omega = cfg["omega"]

    # APARCH: sigma^delta_t = omega + alpha*(|e_{t-1}| - gamma*e_{t-1})^delta + beta*sigma^delta_{t-1}
    sigma_d = np.zeros(T)  # sigma^delta
    sigma = np.zeros(T)
    returns = np.zeros(T)

    # Initialize
    sigma_d[0] = omega / (1 - alpha - beta) if (alpha + beta) < 1 else omega
    sigma[0] = sigma_d[0] ** (1.0 / delta)
    returns[0] = sigma[0] * z[0]

    for t in range(1, T):
        leverage_term = (abs(returns[t-1]) - gamma * returns[t-1]) ** delta
        sigma_d[t] = omega + alpha * leverage_term + beta * sigma_d[t-1]
        sigma[t] = sigma_d[t] ** (1.0 / delta)
        returns[t] = sigma[t] * z[t]

    # Plot
    ax.fill_between(range(T), -2*sigma, 2*sigma, color='#e74c3c', alpha=0.12, label=r'$\pm 2\sigma_t$')
    ax.plot(range(T), returns, color='#7f8c8d', linewidth=0.4, alpha=0.85, label='Returns')
    ax.plot(range(T), 2*sigma, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.plot(range(T), -2*sigma, color='#e74c3c', linewidth=0.8, linestyle='--', alpha=0.7)

    subtitle = rf"$\delta={delta},\ \gamma_1={gamma}$"
    ax.set_title(f"{cfg['title']}  ({subtitle})", fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Return', fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.8)

    # Minimal style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, T)

# Transparent background
fig.patch.set_alpha(0)
for ax in axes:
    ax.patch.set_alpha(0)

# Combined legend at very bottom of figure
handles, labels = axes[0].get_legend_handles_labels()
# Remove per-panel legends
for ax in axes:
    leg = ax.get_legend()
    if leg:
        leg.remove()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=3, frameon=False, fontsize=11)

fig.suptitle('APARCH Model: Effect of $\\delta$ and $\\gamma_1$ Parameters', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.06, 1, 0.94])
plt.savefig('/Users/danielpele/Documents/TSA/charts/garch_aparch_simulation.pdf', bbox_inches='tight', dpi=150, transparent=True)
plt.close()
print("Chart saved.")
