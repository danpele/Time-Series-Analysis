"""
seminar0_explainers.py -- explanatory (primer) charts for Seminar 0 (TSA): first steps with time series
=======================================================================================================
Teaching charts for the slides "Prerequisites for Today" / "Noțiuni necesare azi" of Seminar 0, which takes
place BEFORE Lecture 0. All charts use SIMULATED data only (fixed seeds): they illustrate the concepts (trend and
seasonality, growth rates, the log approximation, autocorrelation and the lag plot, the ACF band, typical ACF
patterns, naive forecasts, MAE and RMSE) and contain no exercise answers.

Output: charts/ch0_sem_primer_*.pdf and .png (transparent background, legend outside at the bottom).
The charts are drawn at their size on the slide, so the text is 7-8 pt on the slide.

Run:  python3 Quantlets/Ch_00/seminar0_explainers.py

Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
import tsa_style as st   # noqa: E402

st.apply()
# drawn at the slide size: 8 pt text on the slide (the beamer text width is 5.67 in)
plt.rcParams.update({'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'lines.linewidth': 1.1})
FULL = (5.5, 1.5)       # full slide width, chart above the bullets (saved about 5.5 x 1.9 in)
HALF = (2.9, 2.3)       # one column of a two-column frame
BLUE, RED, GREEN, AMBER, PURPLE, NAVY = st.MainBlue, st.IDAred, st.Forest, st.Amber, st.Purple, st.DarkText
BAND = '#C5D2E8'


def save(fig, name):
    st.check_no_grey(fig)
    st.save_fig(name)


def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    d = np.sum(x ** 2)
    return np.array([np.sum(x[k:] * x[:len(x) - k]) / d for k in range(nlags + 1)])


def acf_panel(ax, r, T, color, title):
    k = np.arange(1, len(r))
    b = 1.96 / np.sqrt(T)
    ax.axhspan(-b, b, color=BAND, alpha=0.9, lw=0)
    ax.vlines(k, 0, r[1:], color=color, lw=1.4)
    ax.axhline(0, color=NAVY, lw=0.5)
    ax.set_title(title, loc='left')
    ax.set_xlabel('Lag $k$')
    ax.set_ylim(-0.6, 1.05)


def seasonal_series(seed=4, years=10, m=4):
    """Quarterly series: trend x seasonal pattern x small noise (multiplicative, like real GDP)."""
    rng = np.random.default_rng(seed)
    n = years * m
    t = np.arange(n)
    trend = 100 * np.exp(0.006 * t)
    season = np.tile([0.80, 1.02, 1.12, 1.06], years)
    y = trend * season * np.exp(0.012 * rng.standard_normal(n))
    return t, trend, y


# =============================================================================
# 1. level, trend and seasonality; the two growth rates
# =============================================================================
def fig_growth():
    t, trend, y = seasonal_series()
    x = 2016 + t / 4
    qq = 100 * (y[1:] / y[:-1] - 1)
    yy = 100 * (y[4:] / y[:-4] - 1)
    fig, axes = plt.subplots(1, 2, figsize=FULL)
    ax = axes[0]
    ax.plot(x, y, color=BLUE, marker='o', ms=2.2, label='Series $y_t$')
    ax.plot(x, trend, color=AMBER, ls='--', lw=1.2, label='Trend')
    q1 = np.arange(0, len(y), 4)
    ax.plot(x[q1], y[q1], 'o', color=RED, ms=3.2, label='First quarters')
    ax.set_title('Level: trend and seasonality ($m = 4$)', loc='left')
    ax.set_xlabel('Year')
    ax = axes[1]
    ax.axhline(0, color=NAVY, lw=0.5)
    ax.plot(x[1:], qq, color=BLUE, lw=1.0, label='q/q: $100\\,(y_t/y_{t-1} - 1)$')
    ax.plot(x[4:], yy, color=RED, lw=1.6, label='y/y: $100\\,(y_t/y_{t-4} - 1)$')
    ax.set_title('Growth rates, %', loc='left')
    ax.set_xlabel('Year')
    fig.tight_layout(w_pad=1.5)
    st.fig_legend_bottom(fig, ncol=3)
    save(fig, 'ch0_sem_primer_growth')
    return dict(qq_min=qq.min(), qq_max=qq.max(), yy_min=yy.min(), yy_max=yy.max(), yy_mean=yy.mean())


# =============================================================================
# 2. the log approximation 100 ln(1 + g/100) ~ g
# =============================================================================
def fig_logdiff():
    g = np.linspace(-40, 40, 401)
    d = 100 * np.log(1 + g / 100)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(g, g, color=NAVY, ls='--', lw=1.0, label='Growth $g$ (45° line)')
    ax.plot(g, d, color=BLUE, lw=1.6, label='Log difference $100\\,\\ln(1 + g/100)$')
    ax.axvspan(-5, 5, color=BAND, alpha=0.9, lw=0, label='$|g| \\leq 5\\%$: gap below 0.13')
    for gv in (-30, 20):
        dv = 100 * np.log(1 + gv / 100)
        lab = f'{gv:+d}% gives {dv:+.1f}'.replace('-', '\u2212')
        ax.annotate(lab, (gv, dv), xytext=(14 if gv < 0 else -80, -12 if gv < 0 else 14),
                    textcoords='offset points', fontsize=8, color=RED,
                    arrowprops=dict(arrowstyle='->', color=RED, lw=0.7))
        ax.plot(gv, dv, 'o', color=RED, ms=3)
    ax.set_ylim(-62, 48)
    ax.set_xlabel('Growth over one period $g$ (%)')
    ax.set_ylabel('Value (%)')
    ax.set_title('Log difference against growth rate', loc='left')
    st.legend_outside_bottom(ax, ncol=1)
    fig.tight_layout()
    save(fig, 'ch0_sem_primer_logdiff')
    return dict(m30=100 * np.log(0.7), p20=100 * np.log(1.2), gap5=5 - 100 * np.log(1.05))


# =============================================================================
# 3. the lag plot: y_t against y_{t-1}, and the sign of the products of deviations
# =============================================================================
def fig_lagplot(seed=11, T=150):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T + 200)
    noise = e[200:]
    p = np.zeros_like(e)
    for i in range(1, len(e)):
        p[i] = 0.8 * p[i - 1] + e[i]
    pers = p[200:]
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.75))
    out = {}
    for ax, z, col, name in ((axes[0], noise, BLUE, 'Independent noise'), (axes[1], pers, RED, 'Persistent series')):
        zb = z.mean()
        lim = np.max(np.abs(z - zb)) * 1.08
        ax.add_patch(plt.Rectangle((zb, zb), lim, lim, color=BAND, alpha=0.8, lw=0))
        ax.add_patch(plt.Rectangle((zb - lim, zb - lim), lim, lim, color=BAND, alpha=0.8, lw=0,
                                   label='Product of deviations $> 0$'))
        ax.plot(z[:-1], z[1:], 'o', color=col, ms=2.4, alpha=0.85)
        ax.axhline(zb, color=NAVY, lw=0.5)
        ax.axvline(zb, color=NAVY, lw=0.5)
        r1 = acf(z, 1)[1]
        out[name] = r1
        ax.set_title(f'{name}: $r_1$ = {r1:.2f}', loc='left')
        ax.set_xlabel('$y_{t-1}$')
        ax.set_ylabel('$y_t$')
        ax.set_xlim(zb - lim, zb + lim)
        ax.set_ylim(zb - lim, zb + lim)
        ax.set_aspect('equal')
    fig.tight_layout(w_pad=2.0)
    st.fig_legend_bottom(fig, ncol=1)
    save(fig, 'ch0_sem_primer_lagplot')
    return out


# =============================================================================
# 4. the standard Normal distribution and the 95% band
# =============================================================================
def fig_normal_band():
    x = np.linspace(-4, 4, 801)
    f = stats.norm.pdf(x)
    fig, ax = plt.subplots(figsize=HALF)
    ax.plot(x, f, color=BLUE, lw=1.5, label='Standard Normal density')
    ax.fill_between(x, 0, f, where=np.abs(x) <= 1.96, color=BAND, alpha=0.9, lw=0, label='Central 95%')
    ax.fill_between(x, 0, f, where=x >= 1.96, color=RED, alpha=0.55, lw=0, label='2.5% in each tail')
    ax.fill_between(x, 0, f, where=x <= -1.96, color=RED, alpha=0.55, lw=0)
    for v in (-1.96, 1.96):
        ax.axvline(v, color=RED, lw=0.8, ls='--')
        ax.text(v + (0.15 if v > 0 else -0.15), 0.40, f'{v:+.2f}'.replace('-', '\u2212'),
                ha='left' if v > 0 else 'right', color=RED, fontsize=8)
    ax.set_ylim(0, 0.46)
    ax.set_xlabel('$z$')
    ax.set_ylabel('Density')
    ax.set_title('Quantiles $\\pm 1.96$ of $N(0, 1)$', loc='left')
    st.legend_outside_bottom(ax, ncol=1)
    fig.tight_layout()
    save(fig, 'ch0_sem_primer_normal_band')


# =============================================================================
# 5. typical ACF patterns: noise, trend (random walk), seasonality
# =============================================================================
def fig_acf_patterns(seed=7, T=200):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(T)
    rw = np.cumsum(rng.standard_normal(T))
    m = np.arange(T)
    seas = 3 * np.sin(2 * np.pi * m / 12) + rng.standard_normal(T)
    K = 30
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.85), sharey=True)
    out = {}
    for ax, z, col, title in ((axes[0], noise, BLUE, 'Independent noise'), (axes[1], rw, RED, 'Trend (random walk)'),
                              (axes[2], seas, GREEN, 'Monthly seasonality')):
        r = acf(z, K)
        acf_panel(ax, r, T, col, title)
        out[title] = r
    axes[0].set_ylabel('$r_k$')
    axes[0].fill_between([], [], color=BAND, label='Band $\\pm 1.96/\\sqrt{T}$, $T$ = 200')
    fig.tight_layout(w_pad=1.0)
    st.fig_legend_bottom(fig, ncol=1)
    save(fig, 'ch0_sem_primer_acf_patterns')
    noise_out = int(np.sum(np.abs(out['Independent noise'][1:]) > 1.96 / np.sqrt(T)))
    return dict(noise_out=noise_out, rw1=out['Trend (random walk)'][1], rw30=out['Trend (random walk)'][30],
                s12=out['Monthly seasonality'][12], s6=out['Monthly seasonality'][6])


# =============================================================================
# 6. naive and seasonal naive forecasts with a split by time
# =============================================================================
def fig_naive():
    t, trend, y = seasonal_series(seed=21, years=6)
    x = 2020 + t / 4
    ntr = 20
    train, test = y[:ntr], y[ntr:]
    h = len(test)
    naive = np.repeat(train[-1], h)
    snaive = np.array([train[ntr - 4 + (j % 4)] for j in range(h)])
    fig, ax = plt.subplots(figsize=(5.5, 1.75))
    ax.axvspan(x[ntr] - 0.125, x[-1] + 0.1, color=BAND, alpha=0.6, lw=0, label='Test set')
    ax.plot(x[:ntr], train, color=BLUE, marker='o', ms=2.3, label='Training set')
    ax.plot(x[ntr:], test, color=NAVY, marker='o', ms=2.3, label='Test values')
    ax.plot(x[ntr:], naive, color=RED, ls='--', marker='s', ms=2.6, label='Naive $\\hat y_{T+h} = y_T$')
    ax.plot(x[ntr:], snaive, color=GREEN, ls='--', marker='^', ms=2.8, label='Seasonal naive $\\hat y_{T+h} = y_{T+h-m}$')
    ax.set_xlabel('Year')
    ax.set_ylabel('$y_t$')
    ax.set_title('Forecasts made at the end of the training set, $T$ = 20 quarters, $h$ = 1, ..., 4', loc='left')
    fig.tight_layout()
    st.fig_legend_bottom(fig, ncol=3)
    save(fig, 'ch0_sem_primer_naive')
    e1, e2 = test - naive, test - snaive
    return dict(mae_n=np.mean(np.abs(e1)), rmse_n=np.sqrt(np.mean(e1 ** 2)),
                mae_s=np.mean(np.abs(e2)), rmse_s=np.sqrt(np.mean(e2 ** 2)))


# =============================================================================
# 7. MAE and RMSE: same MAE, different RMSE
# =============================================================================
def fig_errors():
    a = np.array([2, -2, 2, -2])
    b = np.array([0.5, -0.5, 0.5, -6.5])
    fig, axes = plt.subplots(1, 2, figsize=HALF, sharey=True)
    for ax, e, col, name in ((axes[0], a, BLUE, 'Method 1'), (axes[1], b, RED, 'Method 2')):
        ax.bar(np.arange(1, 5), e, color=col, width=0.6)
        ax.axhline(0, color=NAVY, lw=0.5)
        mae, rmse = np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2))
        ax.set_title(f'{name}\nMAE {mae:.1f}, RMSE {rmse:.1f}', loc='left')
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel('Test period')
    axes[0].set_ylabel('Error $e = y - \\hat y$')
    fig.tight_layout()
    save(fig, 'ch0_sem_primer_errors')
    return dict(mae=(np.mean(np.abs(a)), np.mean(np.abs(b))), rmse=(np.sqrt(np.mean(a ** 2)), np.sqrt(np.mean(b ** 2))))


if __name__ == '__main__':
    print('growth   ', {k: round(float(v), 2) for k, v in fig_growth().items()})
    print('logdiff  ', {k: round(float(v), 2) for k, v in fig_logdiff().items()})
    print('lagplot  ', {k: round(float(v), 2) for k, v in fig_lagplot().items()})
    fig_normal_band()
    print('acf      ', {k: round(float(v), 2) for k, v in fig_acf_patterns().items()})
    print('naive    ', {k: round(float(v), 2) for k, v in fig_naive().items()})
    print('errors   ', fig_errors())
