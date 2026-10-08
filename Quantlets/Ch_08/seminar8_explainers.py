"""
seminar8_explainers.py -- explanatory (primer) charts for Seminar 8 (TSA): long memory and ARFIMA
=================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 8, which takes place
BEFORE Lecture 8. All charts use SIMULATED data only (fixed seeds): they illustrate the concepts (short and long
memory, ARFIMA paths, fractional weights, R/S, the periodogram and GPH, the shuffle test, GARCH and FIGARCH weights,
spurious memory from a break) and contain no exercise answers.

Charts are drawn at the size of their box on the slide (1 pt in the figure = 1 pt on the slide), so the text is
at least 6.5 pt on the slide.
Output: charts/ch8_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).
Run:  python3 Quantlets/Ch_08/seminar8_explainers.py
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import special

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import tsa_style as st                     # noqa: E402
from generate_all_charts import (frac_weights, arfima_acf, sim_arfima, rs_curve, periodogram, gph,   # noqa: E402
                                 local_whittle, acf, bandwidth)

st.apply()
# sizes for the slide: a full-width chart is 5.45 x 1.75 in (box 0.98\\textwidth x 0.60\\textheight), a column chart 2.6 x 2.3 in (0.46\\textwidth x 0.80\\textheight)
plt.rcParams.update({'font.size': 7.5, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.0, 'pdf.fonttype': 42})
FULL = (5.45, 1.75)
HALF = (2.6, 1.95)        # plus the legend below
FULL_L = (5.45, 1.55)     # full width, plus a legend below
C = [st.MainBlue, st.IDAred, st.Forest, st.Amber, st.Purple]


def legend(fig, ncol=3):
    fig.tight_layout()
    st.fig_legend_bottom(fig, ncol=ncol)


def save(name):
    st.check_no_grey(plt.gcf())
    st.save_fig(f'ch8_sem_primer_{name}')


# =============================================================================
# 1. short and long memory: the ACF of an AR(1) against ARFIMA(0, d, 0)
# =============================================================================
def fig_acf_decay(d=0.3, K=60):
    r = arfima_acf(d, K + 1)
    phi = r[1]                                     # AR(1) with the same rho(1)
    k = np.arange(1, K + 1)
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    ax = axes[0]
    ax.plot(k, r[1:], color=st.IDAred, label=f'ARFIMA(0, {d}, 0): $\\rho(k) \\sim Ck^{{2d-1}}$')
    ax.plot(k, phi ** k, color=st.MainBlue, label=f'AR(1), $\\phi$ = {phi:.3f}: $\\rho(k) = \\phi^k$')
    ax.axhline(0, color=st.DarkText, lw=0.5, ls=':')
    ax.set_xlabel('Lag $k$')
    ax.set_ylabel('$\\rho(k)$')
    ax.set_title('Same $\\rho(1)$, linear scale', loc='left')
    ax = axes[1]
    ax.loglog(k, r[1:], color=st.IDAred)
    ax.loglog(k, phi ** k, color=st.MainBlue)
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel('Lag $k$ (log scale)')
    ax.set_ylabel('$\\rho(k)$ (log scale)')
    ax.set_title(f'Log--log: a line of slope $2d - 1 = {2 * d - 1:.1f}$', loc='left')
    legend(fig, ncol=2)
    save('acf_decay')
    return dict(phi=phi, r10=r[10], ar10=phi ** 10, r50=r[50])


# =============================================================================
# 2. simulated ARFIMA(0, d, 0) paths for three values of d
# =============================================================================
def fig_paths(n=500, seed=8):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + 3000)
    fig, axes = plt.subplots(1, 3, figsize=FULL, sharey=True)
    out = {}
    for ax, d, c in zip(axes, (-0.3, 0.0, 0.4), (st.Forest, st.MainBlue, st.IDAred)):
        psi = frac_weights(-d, 3000)              # (1 - L)^(-d): the MA(infinity) weights psi_k
        x = np.convolve(z, psi)[3000:3000 + n]     # the same shocks for the three values of d
        ax.plot(np.arange(n), x, color=c, lw=0.6)
        ax.axhline(0, color=st.DarkText, lw=0.5, ls=':')
        H = d + 0.5
        lab = 'anti-persistent' if d < 0 else 'white noise' if d == 0 else 'long memory'
        ax.set_title(f'$d = {d:g}$, $H = {H:g}$: {lab}', loc='left')
        ax.set_xlabel('Time $t$')
        out[d] = float(np.corrcoef(x[1:], x[:-1])[0, 1])
    axes[0].set_ylabel('$x_t$')
    fig.tight_layout()
    save('paths')
    return out


# =============================================================================
# 3. fractional weights pi_k and the impulse response psi_k
# =============================================================================
def fig_weights(K=50):
    k = np.arange(1, K + 1)
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    ax = axes[0]
    for d, c in ((0.25, st.Forest), (0.4, st.IDAred), (0.8, st.Purple)):
        ax.loglog(k, -frac_weights(d, K + 1)[1:], color=c, marker='o', ms=1.6, lw=0.8, label=f'$d$ = {d:g}')
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel('Lag $k$ (log scale)')
    ax.set_ylabel('$-\\pi_k$ (log scale)')
    ax.set_title('Weights of $(1-L)^d$ on $x_{t-k}$', loc='left')
    ax = axes[1]
    d = 0.4
    psi = frac_weights(-d, K + 1)
    phi = d / (1 - d)
    ax.plot(np.r_[0, k], psi, color=st.IDAred, label='ARFIMA(0, 0.4, 0): $\\psi_k$')
    ax.plot(np.r_[0, k], phi ** np.r_[0, k], color=st.MainBlue, label=f'AR(1), $\\phi$ = {phi:.3f}')
    ax.set_xlabel('Periods after the shock $k$')
    ax.set_ylabel('Effect $\\psi_k$')
    ax.set_title('Impulse response, $d$ = 0.4', loc='left')
    legend(fig, ncol=4)
    save('weights')
    return dict(psi10=psi[10], ar10=phi ** 10, psi50=psi[50])


# =============================================================================
# 4. R/S: the cumulative deviations of a block and its range
# =============================================================================
def fig_rs_range(n=200, seed=3):
    rng = np.random.default_rng(seed)
    x_lm = sim_arfima(n, 0.4, rng)[0]
    x_wn = rng.standard_normal(n)
    fig, ax = plt.subplots(figsize=HALF)
    out = {}
    for x, c, lab in ((x_wn, st.MainBlue, 'white noise, $d$ = 0'), (x_lm, st.IDAred, 'ARFIMA, $d$ = 0.4')):
        x = x / x.std()
        Y = np.cumsum(x - x.mean())
        ax.plot(np.arange(1, n + 1), Y, color=c, lw=0.9, label=lab)
        i1, i0 = int(np.argmax(Y)), int(np.argmin(Y))
        xm = n + 6 if c == st.IDAred else n + 14
        ax.annotate('', xy=(xm, Y.max()), xytext=(xm, Y.min()),
                    arrowprops=dict(arrowstyle='<->', color=c, lw=0.8))
        out[lab] = float(Y.max() - Y.min())
    ax.axhline(0, color=st.DarkText, lw=0.5, ls=':')
    ax.set_xlim(0, n + 22)
    ax.set_xlabel('$k$ (position in the block)')
    ax.set_ylabel('$Y_k$ (standardised data)')
    ax.set_title('Cumulative deviations; arrows: range $R$', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=1)
    save('rs_range')
    return out


# =============================================================================
# 5. R/S on a log-log scale: the slope is H
# =============================================================================
def fig_rs_loglog(n=4096, seed=11):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=HALF)
    out = {}
    for d, c in ((0.0, st.MainBlue), (0.3, st.IDAred)):
        x = sim_arfima(n, d, rng)[0] if d else rng.standard_normal(n)
        sz, rs = rs_curve(x)
        b, a = np.polyfit(np.log10(sz), np.log10(rs), 1)
        ax.plot(np.log10(sz), np.log10(rs), 'o', ms=2.2, color=c)
        ax.plot(np.log10(sz), a + b * np.log10(sz), color=c, lw=0.9, label=f'$d$ = {d:g}: slope $\\hat H$ = {b:.2f}')
        out[d] = float(b)
    ax.set_xlabel('$\\log_{10} n$ (block size)')
    ax.set_ylabel('$\\log_{10}(R/S)_n$')
    ax.set_title('Simulated, $T$ = 4096', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=1)
    save('rs_loglog')
    return out


# =============================================================================
# 6. the periodogram and the GPH regression
# =============================================================================
def fig_periodogram(n=2048, d=0.3, seed=32):
    rng = np.random.default_rng(seed)
    x = sim_arfima(n, d, rng)[0]
    lam, I = periodogram(x)
    m = bandwidth(n)
    f = (4 * np.sin(lam / 2) ** 2) ** (-d) / (2 * np.pi)       # spectral density of ARFIMA(0, d, 0), sigma^2 = 1
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    ax = axes[0]
    ax.loglog(lam, I, color=st.MainBlue, lw=0.4, label='Periodogram $I(\\lambda_j)$')
    ax.loglog(lam, f, color=st.IDAred, lw=1.2, label='Spectral density $f(\\lambda)$')
    ax.axvline(lam[m - 1], color=st.Forest, lw=0.8, ls='--', label=f'Last frequency used, $j = m$ = {m}')
    ax.set_xlabel('Frequency $\\lambda$ (radians, log scale)')
    ax.set_ylabel('Power (log scale)')
    ax.set_title(f'ARFIMA(0, {d}, 0), $T$ = {n}', loc='left')
    ax = axes[1]
    X = -np.log(4 * np.sin(lam[:m] / 2) ** 2)
    y = np.log(I[:m])
    b, a = np.polyfit(X, y, 1)
    ax.plot(X, y, 'o', ms=1.6, color=st.MainBlue)
    xs = np.linspace(X.min(), X.max(), 10)
    ax.plot(xs, a + b * xs, color=st.Amber, lw=1.3, label=f'OLS line: slope $\\hat d$ = {b:.2f}')
    ax.set_xlabel('$X_j = -\\log(4\\sin^2(\\lambda_j/2))$')
    ax.set_ylabel('$\\log I(\\lambda_j)$')
    ax.set_title(f'GPH regression on the first $m$ = {m} points', loc='left')
    legend(fig, ncol=2)
    save('periodogram')
    return dict(m=m, gph=float(b), lw=local_whittle(x)['d'])


# =============================================================================
# 7. memory in the size of returns, and the shuffle test
# =============================================================================
def fig_shuffle(n=5000, seed=5, K=200):
    rng = np.random.default_rng(seed)
    h = sim_arfima(n, 0.4, rng)[0]                   # log volatility with long memory
    sig = np.exp(0.8 * h / h.std())
    r = sig * rng.standard_normal(n)
    a = np.abs(r)
    a_sh = rng.permutation(a)
    k = np.arange(1, K + 1)
    fig, axes = plt.subplots(1, 2, figsize=FULL_L, gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    ax.plot(np.arange(n), r, color=st.MainBlue, lw=0.3)
    ax.set_xlabel('Day $t$')
    ax.set_ylabel('$r_t$')
    ax.set_title('Simulated returns: calm and turbulent periods', loc='left')
    ax = axes[1]
    band = 1.96 / np.sqrt(n)
    ax.fill_between(k, -band, band, color=st.MainBlue, alpha=0.15, lw=0, label='$\\pm 1.96/\\sqrt{T}$')
    ax.plot(k, acf(r, K), color=st.Forest, lw=0.7, label='ACF of $r_t$')
    ax.plot(k, acf(a, K), color=st.IDAred, lw=1.0, label='ACF of $|r_t|$')
    ax.plot(k, acf(a_sh, K), color=st.Amber, lw=1.0, label='ACF of $|r_t|$ shuffled')
    ax.set_xlabel('Lag $k$ (days)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('The shuffle removes the memory', loc='left')
    legend(fig, ncol=4)
    save('shuffle')
    return dict(lw_abs=local_whittle(a)['d'], lw_shuf=local_whittle(a_sh)['d'], lw_r=local_whittle(r)['d'])


# =============================================================================
# 8. ARCH(infinity) weights: GARCH(1,1) against FIGARCH(0, d, 0)
# =============================================================================
def fig_garch_figarch(alpha=0.08, beta=0.90, d=0.4, K=250):
    k = np.arange(1, K + 1)
    g = alpha * beta ** (k - 1)
    fw = -frac_weights(d, K + 1)[1:]
    fig, ax = plt.subplots(figsize=HALF)
    ax.semilogy(k, g, color=st.MainBlue, label=f'GARCH(1,1): $\\alpha\\beta^{{k-1}}$, $\\alpha$ = {alpha}, $\\beta$ = {beta}')
    ax.semilogy(k, fw, color=st.IDAred, label=f'FIGARCH: $\\lambda_k = -\\pi_k$, $d$ = {d}')
    ax.semilogy(k, d * k ** (-1 - d) / special.gamma(1 - d), color=st.Amber, lw=0.9, ls='--',
                label='$d\\,k^{-1-d}/\\Gamma(1-d)$')
    ax.set_ylim(1e-7, 1)
    ax.set_xlabel('Days ago $k$')
    ax.set_ylabel('Weight of $\\varepsilon_{t-k}^2$ (log)')
    ax.set_title('How long a shock counts in $\\sigma_t^2$', loc='left')
    fig.tight_layout()
    st.legend_outside_bottom(ax, ncol=1)
    save('garch_figarch')
    return dict(g100=float(g[99]), f100=float(fw[99]))


# =============================================================================
# 9. spurious long memory: a short-memory series with one break in the mean
# =============================================================================
def fig_spurious(n=400, seed=4, shift=1.0, K=40):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    mu = np.where(np.arange(n) < n // 3, shift, 0.0)
    x = mu + e
    adj = x.copy()
    adj[:n // 3] -= x[:n // 3].mean()
    adj[n // 3:] -= x[n // 3:].mean()
    k = np.arange(1, K + 1)
    fig, axes = plt.subplots(1, 2, figsize=FULL_L)
    ax = axes[0]
    ax.plot(np.arange(n), x, color=st.MainBlue, lw=0.5, label='$x_t = \\mu_t + \\varepsilon_t$')
    ax.plot(np.arange(n), mu, color=st.IDAred, lw=1.3, label='Mean $\\mu_t$ (one break)')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('$x_t$')
    ax.set_title('White noise plus one change of level', loc='left')
    ax = axes[1]
    band = 1.96 / np.sqrt(n)
    ax.fill_between(k, -band, band, color=st.MainBlue, alpha=0.15, lw=0, label='$\\pm 1.96/\\sqrt{T}$')
    ax.plot(k, acf(x, K), color=st.IDAred, marker='o', ms=1.8, lw=0.8, label='ACF of $x_t$')
    ax.plot(k, acf(adj, K), color=st.Forest, marker='o', ms=1.8, lw=0.8, label='ACF after removing the two means')
    ax.set_xlabel('Lag $k$')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'$\\hat d$ (LW): {local_whittle(x)["d"]:.2f} before, {local_whittle(adj)["d"]:.2f} after', loc='left')
    legend(fig, ncol=3)
    save('spurious')
    return dict(lw=local_whittle(x)['d'], lw_adj=local_whittle(adj)['d'])


if __name__ == '__main__':
    for f in (fig_acf_decay, fig_paths, fig_weights, fig_rs_range, fig_rs_loglog, fig_periodogram, fig_shuffle,
              fig_garch_figarch, fig_spurious):
        print(f.__name__, f())
