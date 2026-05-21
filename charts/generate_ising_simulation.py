"""
Generate ch13_lppl_ising_simulation.png — Phase Transition and Financial Interpretation.
Row 1 (Bubble): T/Tc = 0.50 → 0.80 → 0.94 → 1.00  (heating toward regime break)
Row 2 (Recovery): T/Tc = 1.00 → 1.10 → 1.50 → 2.00  (through break to normal)
Bottom: magnetization curve + financial interpretation text.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

MainBlue   = '#1A3A6E'
Crimson    = '#DC3545'
Forest     = '#2E7D32'
Amber      = '#B5853F'
MediumGray = '#808080'

OUTPUT = '/Users/danielpele/Documents/TSA/charts/ch13_lppl_ising_simulation.png'

N = 60
Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))
SWEEPS = 3000
cmap = ListedColormap([Crimson, MainBlue])


def checkerboard_sweep(spins, beta, parity):
    n = spins.shape[0]
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    mask = (rows + cols) % 2 == parity
    nb = (np.roll(spins, 1, 0) + np.roll(spins, -1, 0) +
          np.roll(spins, 1, 1) + np.roll(spins, -1, 1))
    dE = 2.0 * spins * nb
    prob = np.exp(-beta * dE)
    flip = (dE <= 0) | (np.random.random((n, n)) < prob)
    spins[mask & flip] *= -1
    return spins


def metropolis(spins, T, sweeps):
    if T < 0.01:
        return spins
    beta = 1.0 / T
    for _ in range(sweeps):
        checkerboard_sweep(spins, beta, 0)
        checkerboard_sweep(spins, beta, 1)
    return spins


def magnetization(spins):
    return abs(spins.mean())


# === Row 1: Bubble formation — heating from T/Tc=0.50 toward T_c ===
bubble_panels = [
    (0.50, 'Strong herding',        MainBlue),
    (0.80, 'Fluctuations grow',     Amber),
    (0.94, 'Approaching $T_c$',     Crimson),
    (1.00, 'Regime break',          Crimson),
]

# === Row 2: Recovery — from T_c to normal market ===
crash_panels = [
    (1.00, 'Regime break',          Crimson),
    (1.10, 'Herding collapses',     Crimson),
    (1.50, 'Approaching normal',    Forest),
    (2.00, 'Normal market',         Forest),
]

# Generate bubble snapshots
np.random.seed(42)
spins_b = np.ones((N, N), dtype=int)
spins_b = metropolis(spins_b, 0.50 * Tc, SWEEPS * 2)  # equilibrate
bubble_snapshots = []
for ratio, _, _ in bubble_panels:
    spins_b = metropolis(spins_b, ratio * Tc, SWEEPS)
    bubble_snapshots.append((spins_b.copy(), magnetization(spins_b)))

# Generate crash snapshots
np.random.seed(123)
spins_c = np.ones((N, N), dtype=int)
spins_c = metropolis(spins_c, 0.50 * Tc, SWEEPS * 2)  # equilibrate
crash_snapshots = []
for ratio, _, _ in crash_panels:
    spins_c = metropolis(spins_c, ratio * Tc, SWEEPS)
    crash_snapshots.append((spins_c.copy(), magnetization(spins_c)))

# === Magnetization curve ===
np.random.seed(99)
T_range = np.linspace(0.25, 2.5, 40)
mag_curve = []
spins_m = np.ones((N, N), dtype=int)
spins_m = metropolis(spins_m, 0.25 * Tc, SWEEPS)
for T_ratio in T_range:
    spins_m = metropolis(spins_m, T_ratio * Tc, 1000)
    mag_curve.append(magnetization(spins_m))
mag_curve = np.array(mag_curve)

# === Build figure ===
fig = plt.figure(figsize=(14, 12))
fig.patch.set_alpha(0)
gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1.2],
              hspace=0.35, wspace=0.15)

# Row 1: Bubble
for col, ((snap, mag), (ratio, label, color)) in enumerate(
        zip(bubble_snapshots, bubble_panels)):
    ax = fig.add_subplot(gs[0, col])
    ax.imshow((snap + 1) // 2, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'$T/T_c = {ratio:.2f}$\n{label}', fontsize=9,
                 fontweight='bold', color=color, pad=4)
    ax.text(0.5, -0.08, f'$|M| = {mag:.2f}$', transform=ax.transAxes,
            ha='center', fontsize=8, color=MediumGray)

fig.text(0.02, 0.82, 'Bubble\nFormation', fontsize=11, fontweight='bold',
         color=Crimson, ha='center', va='center', rotation=90)

# Row 2: Crash & Recovery
for col, ((snap, mag), (ratio, label, color)) in enumerate(
        zip(crash_snapshots, crash_panels)):
    ax = fig.add_subplot(gs[1, col])
    ax.imshow((snap + 1) // 2, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'$T/T_c = {ratio:.2f}$\n{label}', fontsize=9,
                 fontweight='bold', color=color, pad=4)
    ax.text(0.5, -0.08, f'$|M| = {mag:.2f}$', transform=ax.transAxes,
            ha='center', fontsize=8, color=MediumGray)

fig.text(0.02, 0.52, 'Break &\nRecovery', fontsize=11, fontweight='bold',
         color=Forest, ha='center', va='center', rotation=90)

# Bottom left: Magnetization curve
ax_mag = fig.add_subplot(gs[2, :2])
ax_mag.plot(T_range, mag_curve, color=MainBlue, lw=2, label='Magnetization $|M|$')
ax_mag.axvline(1.0, color=Crimson, ls='--', lw=1.5, label='$T_c$ (critical)')
ax_mag.axvspan(0.25, 1.0, alpha=0.10, color=Crimson, label='Ordered (Bubble)')
ax_mag.axvspan(1.0, 2.5, alpha=0.10, color=Forest, label='Disordered (Normal)')
ax_mag.set_xlabel('$T / T_c$', fontsize=10)
ax_mag.set_ylabel('Magnetization $|M|$', fontsize=10)
ax_mag.set_title('Phase Transition: Order Parameter', fontsize=11, fontweight='bold')
ax_mag.legend(fontsize=8, loc='upper right', framealpha=0)
ax_mag.set_xlim(0.25, 2.5)
ax_mag.set_ylim(-0.05, 1.05)

# Bottom right: Financial interpretation text
ax_txt = fig.add_subplot(gs[2, 2:])
ax_txt.axis('off')
ax_txt.set_title('Financial Interpretation', fontsize=11, fontweight='bold')
lines = [
    (r'$T \ll T_c$: Bubble regime', MainBlue,
     'Herding dominates, one group controls\n→ Super-exponential price growth'),
    (f'$T \\to T_c$: Approaching regime break', Crimson,
     'Susceptibility diverges, correlations span market\n→ Maximum fragility, log-periodic oscillations'),
    (f'$T = T_c$: Critical point', Crimson,
     'System-wide correlations shatter\n→ Regime break'),
    (f'$T \\gg T_c$: Normal market', Forest,
     'Traders act independently\n→ Efficient market (EMH)'),
]
for i, (title, color, desc) in enumerate(lines):
    y = 0.88 - i * 0.24
    ax_txt.text(0.05, y, title, fontsize=10, fontweight='bold', color=color,
                transform=ax_txt.transAxes, va='top')
    ax_txt.text(0.05, y - 0.08, desc, fontsize=9, color=MediumGray,
                transform=ax_txt.transAxes, va='top')

# Caption
fig.text(0.5, 0.01,
         f'2D Ising model ({N}×{N}). Blue = buy, red = sell. '
         f'Top row: bubble forms as herding strengthens approaching $T_c$. '
         f'Bottom row: regime break at $T_c$, recovery above.',
         ha='center', fontsize=9, color=MediumGray)

fig.savefig(OUTPUT, dpi=150, bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.close(fig)
print(f'Saved {OUTPUT}')
