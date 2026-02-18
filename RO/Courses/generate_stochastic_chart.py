#!/usr/bin/env python3
"""
Generate RO version of stochastic process chart with Romanian labels
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set style - transparent backgrounds
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

np.random.seed(42)
fig, ax = plt.subplots(figsize=(8, 3.5))

n = 100
t = np.arange(n)

# Show multiple realizations
for i, seed in enumerate([42, 123, 456, 789]):
    np.random.seed(seed)
    y = np.zeros(n)
    for j in range(1, n):
        y[j] = 0.7 * y[j-1] + np.random.normal(0, 1)
    ax.plot(t, y, alpha=0.6, linewidth=1, label=f'Realization {i+1}')

ax.set_xlabel('Time $t$')
ax.set_ylabel('$X_t(\\omega)$')
ax.set_title('Stochastic Process: Multiple Realizations from the Same Underlying Process')
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False)

plt.tight_layout()
plt.savefig('../../charts/ch1_def_stochastic_ro.pdf', format='pdf', bbox_inches='tight',
            transparent=True, dpi=150)
plt.savefig('../../charts/ch1_def_stochastic_ro.png', format='png', bbox_inches='tight',
            transparent=True, dpi=150)
print("Chart saved: ch1_def_stochastic_ro.pdf/.png")
plt.close()
