"""
Generate FIGARCH vs GARCH vs IGARCH comparison chart.
Three panels: stochastic simulation (left) + impulse response (right).

The left column shows the full stochastic path of sigma2_t.
The right column shows the impulse response: expected sigma2_t path
after a shock at t=0, compared to the no-shock baseline.
This cleanly illustrates exponential vs hyperbolic vs infinite persistence.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl

# Use a font that supports Romanian diacritics
mpl.rcParams['font.family'] = 'DejaVu Sans'

np.random.seed(42)
T = 1000

# Common innovations
z = np.random.standard_normal(T)
z[300] = 5.0  # large shock

# ─────────────────────────────────────────────────
# GARCH(1,1): alpha + beta = 0.95
# ─────────────────────────────────────────────────
omega_g, alpha_g, beta_g = 0.00002, 0.08, 0.87
uncond_garch = omega_g / (1 - alpha_g - beta_g)

sigma2_garch = np.zeros(T)
sigma2_garch[0] = uncond_garch
for t in range(1, T):
    eps2_prev = sigma2_garch[t-1] * z[t-1]**2
    sigma2_garch[t] = omega_g + alpha_g * eps2_prev + beta_g * sigma2_garch[t-1]

# ─────────────────────────────────────────────────
# IGARCH(1,1): alpha + beta = 1.0
# ─────────────────────────────────────────────────
omega_i, alpha_i, beta_i = 0.00001, 0.08, 0.92

sigma2_igarch = np.zeros(T)
sigma2_igarch[0] = 0.0004
for t in range(1, T):
    eps2_prev = sigma2_igarch[t-1] * z[t-1]**2
    sigma2_igarch[t] = omega_i + alpha_i * eps2_prev + beta_i * sigma2_igarch[t-1]

# ─────────────────────────────────────────────────
# FIGARCH(1,d,1) via truncated ARCH(inf)
# Parameters chosen so lambda_1 > 0: need phi - beta + d > 0
# ─────────────────────────────────────────────────
d_f = 0.4
phi_f = 0.1
beta_f = 0.3
omega_f = 0.00002
trunc = 500

# delta_k: fractional coefficients
delta = np.zeros(trunc + 1)
delta[1] = d_f
for k in range(2, trunc + 1):
    delta[k] = delta[k-1] * (k - 1 - d_f) / k

# lambda_k: ARCH(inf) weights (BBM recursion)
lam = np.zeros(trunc + 1)
lam[1] = phi_f - beta_f + d_f  # = 0.2
for k in range(2, trunc + 1):
    lam[k] = beta_f * lam[k-1] + (delta[k] - phi_f * delta[k-1])
    lam[k] = max(lam[k], 0.0)

omega_star = omega_f / (1 - beta_f)

sigma2_figarch = np.zeros(T)
eps2_fig = np.zeros(T)
sigma2_figarch[0] = 0.0004
eps2_fig[0] = sigma2_figarch[0] * z[0]**2

for t in range(1, T):
    arch_sum = 0.0
    upper = min(t, trunc)
    for k in range(1, upper + 1):
        arch_sum += lam[k] * eps2_fig[t - k]
    sigma2_figarch[t] = omega_star + arch_sum
    sigma2_figarch[t] = max(sigma2_figarch[t], 1e-10)
    eps2_fig[t] = sigma2_figarch[t] * z[t]**2

# ─────────────────────────────────────────────────
# Impulse Response Functions (IRF)
# Show E[sigma2_{t+h} | shock at t] vs E[sigma2_{t+h} | no shock]
# We set all future z = 1 (so eps2 = sigma2, the expected value)
# and compare paths with vs without z_0 = 5.
# ─────────────────────────────────────────────────
H = 500  # horizon

# GARCH IRF
def garch_irf(omega, alpha, beta, sigma2_0, z_shock, H):
    """Expected sigma2 path: z_0 = z_shock, then E[z^2]=1 for t>0."""
    s = np.zeros(H)
    s[0] = omega + alpha * sigma2_0 * z_shock**2 + beta * sigma2_0
    for h in range(1, H):
        # E[sigma2_{h+1}] = omega + (alpha + beta) * E[sigma2_h]
        s[h] = omega + (alpha + beta) * s[h-1]
    return s

sig0 = uncond_garch
irf_garch_shock = garch_irf(omega_g, alpha_g, beta_g, sig0, 5.0, H)
irf_garch_base  = garch_irf(omega_g, alpha_g, beta_g, sig0, 1.0, H)

# IGARCH IRF
sig0_i = 0.0004
irf_igarch_shock = garch_irf(omega_i, alpha_i, beta_i, sig0_i, 5.0, H)
irf_igarch_base  = garch_irf(omega_i, alpha_i, beta_i, sig0_i, 1.0, H)

# FIGARCH IRF: use the ARCH(inf) representation with E[eps2_k] = sigma2_k
def figarch_irf(omega_star, lam, sigma2_0, z_shock, H, trunc):
    s = np.zeros(H)
    e2 = np.zeros(H)
    # t = -1: eps2 = sigma2_0 * z_shock^2
    # before that: eps2 = sigma2_0 (steady state)
    # We store a buffer of past eps2 values
    buf_len = trunc
    # Pre-shock steady state eps2
    eps2_history = np.full(buf_len, sigma2_0)  # long pre-history at steady state

    # Shock period (h=0): eps2 at h=-1 was sigma2_0 * z_shock^2
    eps2_history[-1] = sigma2_0 * z_shock**2

    for h in range(H):
        arch_sum = 0.0
        for k in range(1, min(buf_len, trunc) + 1):
            if k <= len(eps2_history):
                arch_sum += lam[k] * eps2_history[-k]
        s[h] = omega_star + arch_sum
        s[h] = max(s[h], 1e-10)
        e2[h] = s[h]  # E[eps2] = sigma2 when E[z^2] = 1
        # Append to history
        eps2_history = np.append(eps2_history, e2[h])
    return s

# Steady-state sigma2 for FIGARCH: omega_star / (1 - sum(lam))
lam_sum = lam[1:trunc+1].sum()
figarch_ss = omega_star / (1 - lam_sum) if lam_sum < 1 else 0.0004

irf_figarch_shock = figarch_irf(omega_star, lam, figarch_ss, 5.0, H, trunc)
irf_figarch_base  = figarch_irf(omega_star, lam, figarch_ss, 1.0, H, trunc)

# ─────────────────────────────────────────────────
# Plotting: 3 rows x 2 columns
# Left: stochastic simulation | Right: impulse response
# ─────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 10),
                          gridspec_kw={'width_ratios': [1.2, 1]})
plt.subplots_adjust(hspace=0.35, wspace=0.30)

model_titles = [
    "GARCH(1,1): " + r"$\alpha+\beta = 0.95$"
    + " \u2014 exponential decay",
    "FIGARCH(1,d,1): " + r"$d = 0.4$"
    + " \u2014 hyperbolic decay",
    "IGARCH(1,1): " + r"$\alpha+\beta = 1.0$"
    + " \u2014 infinite persistence",
]

sim_data = [sigma2_garch, sigma2_figarch, sigma2_igarch]
unconds_left = [uncond_garch, None, None]

irf_shocks = [irf_garch_shock, irf_figarch_shock, irf_igarch_shock]
irf_bases  = [irf_garch_base,  irf_figarch_base,  irf_igarch_base]

for i in range(3):
    # ── Left panel: stochastic simulation ──
    ax = axes[i, 0]
    ax.plot(range(T), sim_data[i], color='#2166AC', linewidth=0.7, alpha=0.85)
    if unconds_left[i] is not None:
        ax.axhline(unconds_left[i], color='#B2182B', linestyle='--', linewidth=1.2,
                    label=r'$E[\sigma^2_t]$')
    ax.axvline(300, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_title(model_titles[i], fontsize=12, fontweight='bold', pad=6)
    ax.set_ylabel(r'$\sigma^2_t$', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if i == 2:
        ax.set_xlabel('t', fontsize=11)

    # ── Right panel: impulse response ──
    ax2 = axes[i, 1]
    ax2.plot(range(H), irf_shocks[i], color='#B2182B', linewidth=1.5,
             label=r'$z_0 = 5$' + ' (shock)')
    ax2.plot(range(H), irf_bases[i], color='#2166AC', linewidth=1.5,
             linestyle='--', label=r'$z_0 = 1$' + ' (baseline)')
    ax2.set_title("Impulse response", fontsize=12, fontweight='bold', pad=6)
    ## legend moved to bottom of figure
    ax2.set_ylabel(r'$E[\sigma^2_{t+h}]$', fontsize=11)
    ax2.tick_params(labelsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    if i == 2:
        ax2.set_xlabel('h (horizon)', fontsize=11)

# Annotation for the shock on first left panel
axes[0, 0].annotate(r'$z_{300}=5$', xy=(300, sim_data[0][300]),
                     xytext=(360, sim_data[0][300] * 0.85),
                     fontsize=10, color='#555555',
                     arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2))

# Transparent background
fig.patch.set_alpha(0)
for row in axes:
    for a in row:
        a.patch.set_alpha(0)

# Combined legend at bottom from right-panel labels (IRF) + left-panel unconditional
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color='#B2182B', linewidth=1.5, label=r'$z_0 = 5$ (shock)'),
    Line2D([0], [0], color='#2166AC', linewidth=1.5, linestyle='--', label=r'$z_0 = 1$ (baseline)'),
    Line2D([0], [0], color='#B2182B', linewidth=1.2, linestyle='--', label=r'$E[\sigma^2_t]$'),
]
fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=3, frameon=False, fontsize=10)

plt.subplots_adjust(bottom=0.08)
plt.savefig('/Users/danielpele/Documents/TSA/charts/garch_figarch_simulation.pdf',
            bbox_inches='tight', dpi=150, transparent=True)
plt.close()

print("Chart saved.")
print(f"GARCH   IRF: starts {irf_garch_shock[0]:.6f}, ends {irf_garch_shock[-1]:.6f}, "
      f"uncond {uncond_garch:.6f}")
print(f"FIGARCH IRF: starts {irf_figarch_shock[0]:.6f}, ends {irf_figarch_shock[-1]:.6f}, "
      f"steady {figarch_ss:.6f}")
print(f"IGARCH  IRF: starts {irf_igarch_shock[0]:.6f}, ends {irf_igarch_shock[-1]:.6f}")
print(f"Lambda sum: {lam_sum:.4f}, lambda[1]={lam[1]:.4f}")
