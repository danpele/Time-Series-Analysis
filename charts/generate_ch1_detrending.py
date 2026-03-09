"""
Generate ch1_detrending.pdf — Stationarizing a series with deterministic trend.
Left: original series + fitted trend; Right: detrended (stationary) residuals.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Style (matches regenerate_ch1_charts.py) ---
BLUE   = '#1A3A6E'
RED    = '#DC3545'
GREEN  = '#2E7D32'
ORANGE = '#E67E22'
GRAY   = '#666666'

plt.rcParams.update({
    'axes.facecolor':      'none',
    'figure.facecolor':    'none',
    'savefig.transparent': True,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.edgecolor':      GRAY,
    'font.size':           9,
    'axes.titlesize':      10,
    'axes.labelsize':      9,
    'xtick.labelsize':     8,
    'ytick.labelsize':     8,
    'legend.fontsize':     8,
    'figure.dpi':          150,
    'lines.linewidth':     1.2,
})

def add_legend_bottom(ax, ncol=None, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ncol = ncol or min(len(handles), 4)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
                  ncol=ncol, frameon=False, fontsize=7, **kwargs)

# --- Data ---
np.random.seed(42)
n = 200
t = np.arange(n)
alpha, beta, sigma = 5.0, 0.15, 3.0
eps = np.random.normal(0, sigma, n)
Y = alpha + beta * t + eps            # TS process

# OLS fit
slope, intercept, r, p, se = stats.linregress(t, Y)
trend_hat = intercept + slope * t
residuals = Y - trend_hat              # detrended series

# --- Figure ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.8))

# Left panel: original + fitted trend
ax1.plot(t, Y, color=BLUE, linewidth=0.6, alpha=0.85, label=r'$Y_t = \alpha + \beta\,t + \varepsilon_t$')
ax1.plot(t, trend_hat, color=RED, linewidth=1.4, linestyle='--',
         label=rf'$\hat{{Y}}_t = {intercept:.1f} + {slope:.3f}\,t$')
ax1.set_title('Series with deterministic trend', fontsize=9, fontweight='bold')
ax1.set_xlabel('Time')
ax1.set_ylabel(r'$Y_t$')
add_legend_bottom(ax1)

# Right panel: detrended residuals
ax2.plot(t, residuals, color=GREEN, linewidth=0.6, alpha=0.85,
         label=r'$\hat{\varepsilon}_t = Y_t - \hat{\alpha} - \hat{\beta}\,t$')
ax2.axhline(0, color=GRAY, linewidth=0.5, linestyle=':')
ax2.set_title('Stationarized series (residuals)', fontsize=9, fontweight='bold')
ax2.set_xlabel('Time')
ax2.set_ylabel(r'$\hat{\varepsilon}_t$')
add_legend_bottom(ax2)

fig.tight_layout(rect=[0, 0.08, 1, 1])

fig.savefig('ch1_detrending.pdf', bbox_inches='tight', transparent=True, pad_inches=0.05)
fig.savefig('ch1_detrending.png', bbox_inches='tight', transparent=True, dpi=200, pad_inches=0.05)
plt.close(fig)
print('Saved: ch1_detrending.pdf / .png')
