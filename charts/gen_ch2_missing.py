"""Generate missing ch2 seminar charts."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib
matplotlib.use('Agg')

# Style settings - transparent, no grid, legend outside bottom
plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'savefig.transparent': True,
    'axes.grid': False,
    'font.size': 10,
    'figure.dpi': 150,
})

IDA_RED = '#DC3545'
ACCENT_BLUE = '#0056b3'

# ============================================================
# 1. ch2_ar1.pdf — AR(1) stationarity region + simulations
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: stationarity region
ax = axes[0]
ax.axhspan(-1, 1, alpha=0.15, color='green')
ax.axhline(1, color=IDA_RED, ls='--', lw=1.5)
ax.axhline(-1, color=IDA_RED, ls='--', lw=1.5)
ax.axhline(0, color='black', lw=0.5)
for phi in [-0.8, 0.5, 0.9]:
    ax.plot(0.5, phi, 'o', ms=10, color=ACCENT_BLUE)
    ax.annotate(f'$\\phi={phi}$', (0.55, phi), fontsize=9)
for phi in [1.2, -1.5]:
    ax.plot(0.5, phi, 'x', ms=10, color=IDA_RED, mew=2)
    ax.annotate(f'$\\phi={phi}$', (0.55, phi), fontsize=9, color=IDA_RED)
ax.set_xlim(0, 1)
ax.set_ylim(-1.7, 1.7)
ax.set_ylabel('$\\phi$')
ax.set_title('Regiune staționară AR(1)', fontsize=11)
ax.set_xticks([])
handles = [
    plt.Line2D([0], [0], color='green', alpha=0.3, lw=8, label='Regiune staționară'),
    plt.Line2D([0], [0], color=IDA_RED, ls='--', lw=1.5, label='$|\\phi|=1$ (rădăcină unitară)'),
]
ax.legend(handles=handles, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.25),
          ncol=2, frameon=False)

# Right: simulations
np.random.seed(42)
T = 100
eps = np.random.normal(0, 1, T)
ax = axes[1]
for phi, color, ls in [(0.5, ACCENT_BLUE, '-'), (0.9, 'orange', '-'), (-0.8, 'green', '--')]:
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t-1] + eps[t]
    ax.plot(x, color=color, ls=ls, lw=1.2, label=f'$\\phi={phi}$')
ax.set_title('Simulări AR(1)', fontsize=11)
ax.set_xlabel('Timp')
ax.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.25),
          ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('ch2_ar1.pdf', bbox_inches='tight', transparent=True)
plt.savefig('ch2_ar1.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Created ch2_ar1.pdf")

# ============================================================
# 2. ch2_ma1.pdf — MA(1) ACF cutoff + unit circle
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: ACF of MA(1)
ax = axes[0]
lags = np.arange(0, 11)
theta = 0.7
acf_vals = np.zeros(11)
acf_vals[0] = 1.0
acf_vals[1] = theta / (1 + theta**2)
ax.bar(lags, acf_vals, color=ACCENT_BLUE, width=0.3, zorder=3)
ci = 1.96 / np.sqrt(100)
ax.axhline(ci, color=IDA_RED, ls='--', lw=1, alpha=0.7)
ax.axhline(-ci, color=IDA_RED, ls='--', lw=1, alpha=0.7)
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Lag')
ax.set_ylabel('ACF')
ax.set_title(f'ACF — MA(1), $\\theta={theta}$', fontsize=11)
ax.annotate('Se anulează\ndupă lag 1', xy=(2, 0), xytext=(4, 0.25),
            fontsize=9, color=IDA_RED, arrowprops=dict(arrowstyle='->', color=IDA_RED))
handles = [
    plt.Line2D([0], [0], color=ACCENT_BLUE, lw=6, label='ACF'),
    plt.Line2D([0], [0], color=IDA_RED, ls='--', lw=1, label='±1.96/√T'),
]
ax.legend(handles=handles, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.25),
          ncol=2, frameon=False)

# Right: unit circle for invertibility
ax = axes[1]
circle = Circle((0, 0), 1, fill=False, color='black', lw=1.5, ls='--')
ax.add_patch(circle)
ax.set_xlim(-1.8, 1.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect('equal')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)

# Invertible root (|z| > 1)
z_inv = -1/0.7
ax.plot(z_inv, 0, 'o', ms=10, color='green', zorder=5)
ax.annotate(f'$z={z_inv:.2f}$\n(invertibil)', (z_inv, 0.15), fontsize=9, color='green', ha='center')

# Non-invertible root (|z| < 1)
z_noninv = -1/1.5
ax.plot(z_noninv, 0, 'x', ms=10, color=IDA_RED, mew=2, zorder=5)
ax.annotate(f'$z={z_noninv:.2f}$\n(neinvertibil)', (z_noninv, -0.25), fontsize=9, color=IDA_RED, ha='center')

ax.set_title('Cercul unitate — Invertibilitate MA', fontsize=11)
ax.set_xlabel('Re')
ax.set_ylabel('Im')
handles = [
    plt.Line2D([0], [0], marker='o', color='green', ls='', ms=8, label='Invertibil ($|z|>1$)'),
    plt.Line2D([0], [0], marker='x', color=IDA_RED, ls='', ms=8, mew=2, label='Neinvertibil ($|z|<1$)'),
]
ax.legend(handles=handles, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.25),
          ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('ch2_ma1.pdf', bbox_inches='tight', transparent=True)
plt.savefig('ch2_ma1.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Created ch2_ma1.pdf")

# ============================================================
# 3. ch2_arma.pdf — ARMA(1,1) simulation + ACF/PACF
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Simulate ARMA(1,1): X_t = 0.7*X_{t-1} + eps_t + 0.4*eps_{t-1}
np.random.seed(123)
T = 200
phi, theta = 0.7, 0.4
eps = np.random.normal(0, 1, T)
x = np.zeros(T)
for t in range(1, T):
    x[t] = phi * x[t-1] + eps[t] + theta * eps[t-1]

ax = axes[0]
ax.plot(x, color=ACCENT_BLUE, lw=0.8, label=f'ARMA(1,1)')
ax.set_title(f'ARMA(1,1): $\\phi={phi}$, $\\theta={theta}$', fontsize=11)
ax.set_xlabel('Timp')
ax.set_ylabel('$X_t$')
ax.axhline(0, color='black', lw=0.5)
ax.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.25),
          ncol=1, frameon=False)

# ACF
ax = axes[1]
n_lags = 15
x_centered = x - x.mean()
acf = np.array([1.0] + [np.corrcoef(x_centered[k:], x_centered[:-k])[0, 1] for k in range(1, n_lags+1)])
lags = np.arange(n_lags+1)
ax.bar(lags, acf, color=ACCENT_BLUE, width=0.3, zorder=3)
ci = 1.96 / np.sqrt(T)
ax.axhline(ci, color=IDA_RED, ls='--', lw=1, alpha=0.7)
ax.axhline(-ci, color=IDA_RED, ls='--', lw=1, alpha=0.7)
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Lag')
ax.set_ylabel('ACF')
ax.set_title('ACF — ARMA(1,1)', fontsize=11)
ax.annotate('Descreștere\ngraduală\n(componenta AR)', xy=(5, acf[5]), xytext=(8, 0.4),
            fontsize=9, color=IDA_RED, arrowprops=dict(arrowstyle='->', color=IDA_RED))
handles = [
    plt.Line2D([0], [0], color=ACCENT_BLUE, lw=6, label='ACF'),
    plt.Line2D([0], [0], color=IDA_RED, ls='--', lw=1, label='±1.96/√T'),
]
ax.legend(handles=handles, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.25),
          ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('ch2_arma.pdf', bbox_inches='tight', transparent=True)
plt.savefig('ch2_arma.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Created ch2_arma.pdf")

# ============================================================
# 4. ch2_ar2.pdf — AR(2) stationarity triangle + roots
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: stationarity triangle
ax = axes[0]
phi1 = np.linspace(-2.5, 2.5, 500)
ax.fill_between(phi1, np.maximum(-1, phi1 - 1), np.minimum(1 - phi1, -phi1 - 1),
                where=(np.maximum(-1, phi1 - 1) < np.minimum(1 - phi1, -phi1 - 1)),
                alpha=0.2, color='green')
ax.plot(phi1, 1 - phi1, 'b--', lw=1)
ax.plot(phi1, -1 - phi1, 'r--', lw=1)
ax.axhline(-1, color='gray', ls=':', lw=1)
ax.axhline(1, color='gray', ls=':', lw=1)

# Example points
ax.plot(0.6, 0.3, 'o', ms=8, color='green', zorder=5)
ax.annotate('Staționar', (0.65, 0.35), fontsize=8, color='green')
ax.plot(0.8, 0.5, 'x', ms=8, color=IDA_RED, mew=2, zorder=5)
ax.annotate('Nestaționar', (0.85, 0.55), fontsize=8, color=IDA_RED)

ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('$\\phi_1$')
ax.set_ylabel('$\\phi_2$')
ax.set_title('Triunghiul de staționaritate AR(2)', fontsize=11)
ax.axhline(0, color='black', lw=0.3)
ax.axvline(0, color='black', lw=0.3)
handles = [
    plt.Line2D([0], [0], color='green', alpha=0.3, lw=8, label='Regiune staționară'),
    plt.Line2D([0], [0], color='blue', ls='--', lw=1, label='$\\phi_1+\\phi_2=1$'),
    plt.Line2D([0], [0], color='red', ls='--', lw=1, label='$\\phi_2-\\phi_1=1$'),
    plt.Line2D([0], [0], color='gray', ls=':', lw=1, label='$\\phi_2=\\pm 1$'),
]
ax.legend(handles=handles, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.35),
          ncol=2, frameon=False)

# Right: unit circle with complex roots
ax = axes[1]
circle = Circle((0, 0), 1, fill=False, color='black', lw=1.5, ls='--')
ax.add_patch(circle)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)

# Complex roots outside unit circle (stationary case)
z1 = 1.2 * np.exp(1j * np.pi/4)
z2 = 1.2 * np.exp(-1j * np.pi/4)
ax.plot(z1.real, z1.imag, 'o', ms=10, color='green', zorder=5)
ax.plot(z2.real, z2.imag, 'o', ms=10, color='green', zorder=5)
ax.annotate('$z_1$ (|z|>1)', (z1.real+0.1, z1.imag+0.1), fontsize=9, color='green')
ax.annotate('$z_2$ (|z|>1)', (z2.real+0.1, z2.imag-0.2), fontsize=9, color='green')

# Roots inside unit circle (non-stationary)
z3 = 0.7 * np.exp(1j * np.pi/3)
z4 = 0.7 * np.exp(-1j * np.pi/3)
ax.plot(z3.real, z3.imag, 'x', ms=10, color=IDA_RED, mew=2, zorder=5)
ax.plot(z4.real, z4.imag, 'x', ms=10, color=IDA_RED, mew=2, zorder=5)
ax.annotate('$z_3$ (|z|<1)', (z3.real+0.1, z3.imag+0.1), fontsize=9, color=IDA_RED)

ax.set_title('Rădăcini complexe AR(2)', fontsize=11)
ax.set_xlabel('Re')
ax.set_ylabel('Im')
ax.annotate('Rădăcini complexe\n→ cicluri', xy=(0.5, 1.5), fontsize=9,
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
handles = [
    plt.Line2D([0], [0], marker='o', color='green', ls='', ms=8, label='Staționar ($|z|>1$)'),
    plt.Line2D([0], [0], marker='x', color=IDA_RED, ls='', ms=8, mew=2, label='Nestaționar ($|z|<1$)'),
]
ax.legend(handles=handles, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.25),
          ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('ch2_ar2.pdf', bbox_inches='tight', transparent=True)
plt.savefig('ch2_ar2.png', bbox_inches='tight', transparent=True, dpi=300)
plt.close()
print("Created ch2_ar2.pdf")

print("\nAll 4 charts generated successfully!")
