"""
Regenerate Ising simulation frames for Ch13 LPPL.
Bubble sequence: T=0 (perfect bubble) → T≈Tc (approaching crash).
Crash sequence: T=Tc (crash) → T>>Tc (normal market restored).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

MainBlue = '#1A3A6E'
Crimson  = '#DC3545'
Forest   = '#2E7D32'
Amber    = '#B5853F'

OUTPUT = '/Users/danielpele/Documents/TSA/charts/ising_frames'
os.makedirs(OUTPUT, exist_ok=True)

N = 80
Tc = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ~2.269
SWEEPS = 5000

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
        return spins  # T≈0: no flips, ground state
    beta = 1.0 / T
    for _ in range(sweeps):
        checkerboard_sweep(spins, beta, 0)
        checkerboard_sweep(spins, beta, 1)
    return spins

def magnetization(spins):
    return abs(spins.mean())

def save_frame(spins, ratio, label, color, idx, prefix):
    fig, ax = plt.subplots(figsize=(6, 6.8))
    fig.patch.set_alpha(0)
    ax.imshow((spins + 1) // 2, cmap=cmap, vmin=0, vmax=1,
              interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    mag = magnetization(spins)
    fig.text(0.03, 0.96,
             f'$T/T_c$ = {ratio:.2f}',
             fontsize=15, style='italic', color=Forest,
             transform=fig.transFigure, va='top')
    fig.text(0.38, 0.96,
             f'—   {label}',
             fontsize=14, fontweight='bold', color=color,
             transform=fig.transFigure, va='top')
    fig.text(0.50, 0.03,
             f'Magnetization $|M|$ = {mag:.2f}',
             fontsize=12, ha='center', color='gray',
             transform=fig.transFigure)
    path = f'{OUTPUT}/{prefix}_{idx:02d}.png'
    fig.savefig(path, dpi=100, bbox_inches='tight', transparent=True,
                pad_inches=0.08)
    plt.close(fig)
    print(f'  {path}  (T/Tc={ratio:.2f}, |M|={mag:.2f})')

# ── Bubble sequence: ordered market heating toward Tc ──
bubble_frames = [
    (0.50, 'Strong herding — stable bubble',          MainBlue),
    (0.60, 'Bubble intact — small fluctuations',      MainBlue),
    (0.70, 'First cracks — domains appear',           Amber),
    (0.78, 'Herding weakens — domains grow',          Amber),
    (0.84, 'Bubble destabilizing',                    Amber),
    (0.88, 'Large fluctuations — consensus fading',   Amber),
    (0.92, 'Approaching $T_c$ — instability grows',   Crimson),
    (0.94, 'Near-critical — susceptibility diverges',  Crimson),
    (0.96, 'Bubble fragile — large domains flip',     Crimson),
    (0.98, 'On the edge of $T_c$ — loss of order',   Crimson),
    (0.99, 'Imminent regime break',                   Crimson),
    (1.00, 'Critical point $T_c$ — regime break',     Crimson),
]

# ── Crash sequence: heating from Tc to T>>Tc ──
crash_frames = [
    (1.00, 'Critical point $T_c$ — regime break',   Crimson),
    (1.02, 'Post-break — loss of consensus',        Crimson),
    (1.05, 'Disorder spreads rapidly',              Crimson),
    (1.10, 'Herding collapses',                     Crimson),
    (1.15, 'Recovery — correlations dissolving',    Amber),
    (1.25, 'Market stabilizing',                    Amber),
    (1.35, 'Weak correlations remain',              Amber),
    (1.50, 'Approaching normal market',             Forest),
    (1.65, 'Near-random trading',                   Forest),
    (1.80, 'Efficient market regime',               Forest),
    (1.90, 'Independent trading restored',          Forest),
    (2.00, 'Normal market — random trading',        Forest),
]

print('=== Generating BUBBLE frames (heating from T/Tc=0.50 toward Tc) ===')
np.random.seed(42)
spins = np.ones((N, N), dtype=int)
# Equilibrate at T/Tc = 0.50 first — produces a realistic ordered state with minor fluctuations
spins = metropolis(spins, 0.50 * Tc, SWEEPS * 2)
for idx, (ratio, label, color) in enumerate(bubble_frames, 1):
    T = ratio * Tc
    spins = metropolis(spins, T, SWEEPS)
    save_frame(spins, ratio, label, color, idx, 'ising_bubble')

print('\n=== Generating CRASH frames (heating from Tc to T>>Tc) ===')
np.random.seed(123)
spins_crash = np.ones((N, N), dtype=int)
spins_crash = metropolis(spins_crash, 0.50 * Tc, SWEEPS * 2)
for idx, (ratio, label, color) in enumerate(crash_frames, 1):
    T = ratio * Tc
    spins_crash = metropolis(spins_crash, T, SWEEPS)
    save_frame(spins_crash, ratio, label, color, idx, 'ising_crash')

print('\nDone!')
