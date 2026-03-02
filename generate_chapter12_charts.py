#!/usr/bin/env python3
"""
Generate charts for Chapter 12: Spectral Analysis
Publication-quality charts for Beamer slides (16:9 aspect ratio)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('/Users/danielpele/Documents/TSA/charts', exist_ok=True)

# ---------------------------------------------------------------------------
# Color scheme matching course theme
# ---------------------------------------------------------------------------
MAIN_BLUE = '#1A3A6E'
CRIMSON   = '#DC3545'
FOREST    = '#2E7D32'
AMBER     = '#B5853F'
ORANGE    = '#E6802E'
PURPLE    = '#8E44AD'
GRAY      = '#666666'
LIGHT_BLUE = '#5B8BD4'

COLORS = [MAIN_BLUE, CRIMSON, FOREST, AMBER, ORANGE, PURPLE]

# ---------------------------------------------------------------------------
# Global rcParams – transparent background, no top/right spine, no grid
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'figure.figsize':       (12, 5),
    'font.size':            11,
    'axes.titlesize':       13,
    'axes.labelsize':       12,
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
    'axes.facecolor':       'none',
    'figure.facecolor':     'none',
    'savefig.facecolor':    'none',
    'axes.grid':            False,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'legend.frameon':       False,
    'legend.fontsize':      10,
    'lines.linewidth':      1.8,
})

CHARTS_DIR = '/Users/danielpele/Documents/TSA/charts'


def save_chart(fig, name):
    """Save chart as PDF with transparent background."""
    path = f'{CHARTS_DIR}/{name}.pdf'
    fig.savefig(path, bbox_inches='tight', dpi=150, transparent=True)
    plt.close(fig)
    print(f"  Saved: {path}")


# ===========================================================================
# DATA GENERATORS
# ===========================================================================

def make_sunspot_data(seed=42):
    """Monthly sunspot numbers 1960-2023 with realistic 11-year cycle."""
    np.random.seed(seed)
    dates = pd.date_range('1960-01-01', '2023-12-01', freq='MS')
    n = len(dates)
    t = np.arange(n)
    cycle = 11 * 12  # 132 months
    # Sunspot amplitudes vary between cycles
    base = (75 + 55 * np.sin(2 * np.pi * t / cycle - np.pi / 2)
            + 15 * np.sin(4 * np.pi * t / cycle))
    noise = np.random.gamma(1, 15, n)
    sunspots = np.maximum(base + noise, 0)
    # Rough amplitude modulation (Gleissberg cycle ~80yr)
    gleissberg = 1 + 0.25 * np.sin(2 * np.pi * t / (80 * 12))
    sunspots *= gleissberg
    return pd.DataFrame({'date': dates, 'sunspots': sunspots})


def make_gdp_data(seed=7):
    """Log US-like real GDP quarterly 1960-2023."""
    np.random.seed(seed)
    dates = pd.date_range('1960-01-01', '2023-10-01', freq='QS')
    n = len(dates)
    t = np.arange(n)
    # Long-run trend: ~3% annual = 0.75% quarterly
    drift = 0.0075
    eps = np.random.normal(0, 0.005, n)
    rw = np.cumsum(eps) + drift * t
    # Business cycle (4-8 year = 16-32 quarters) – add two harmonics
    cycle = (0.012 * np.sin(2 * np.pi * t / 24 + 0.5)
             + 0.008 * np.sin(2 * np.pi * t / 18 + 1.2))
    # COVID dip
    covid_idx = np.where(dates >= pd.Timestamp('2020-04-01'))[0]
    cycle[covid_idx[:2]] -= 0.08
    log_gdp = 7.5 + rw + cycle  # Start near log(1800) billion
    return pd.DataFrame({'date': dates, 'log_gdp': log_gdp})


def ar1_process(phi, n=512, seed=0):
    np.random.seed(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + np.random.randn()
    return x


def ar2_process(phi1, phi2, n=512, seed=0):
    np.random.seed(seed)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + np.random.randn()
    return x


def theoretical_ar1_spectrum(phi, freqs):
    """Theoretical spectral density of AR(1)."""
    sigma2 = 1.0
    return sigma2 / (1 - 2 * phi * np.cos(2 * np.pi * freqs) + phi ** 2)


def theoretical_ar2_spectrum(phi1, phi2, freqs):
    """Theoretical spectral density of AR(2)."""
    sigma2 = 1.0
    omega = 2 * np.pi * freqs
    denom = np.abs(1 - phi1 * np.exp(-1j * omega) - phi2 * np.exp(-2j * omega)) ** 2
    return sigma2 / denom


# ===========================================================================
# CHART 1: Sunspot Time Series
# ===========================================================================
def ch12_sunspot_timeseries():
    print("  [1/26] ch12_sunspot_timeseries")
    df = make_sunspot_data()
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.fill_between(df['date'], df['sunspots'], alpha=0.25, color=AMBER)
    ax.plot(df['date'], df['sunspots'], color=AMBER, linewidth=1.2)
    ax.set_title('Monthly Sunspot Numbers (1960–2023)', fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Sunspot Number')
    # Annotate solar maxima
    for yr, label in [(1969, 'Max'), (1980, 'Max'), (1990, 'Max'),
                      (2001, 'Max'), (2014, 'Max')]:
        mask = (df['date'].dt.year == yr)
        idx = df.loc[mask, 'sunspots'].idxmax()
        ax.annotate(label, xy=(df['date'][idx], df['sunspots'][idx]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8.5, color=CRIMSON,
                    arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=0.8))
    ax.text(0.01, 0.95, '≈ 11-year cycle', transform=ax.transAxes,
            fontsize=10, color=MAIN_BLUE, va='top', style='italic')
    plt.tight_layout()
    save_chart(fig, 'ch12_sunspot_timeseries')


# ===========================================================================
# CHART 2: Sunspot Periodogram
# ===========================================================================
def ch12_sunspot_periodogram():
    print("  [2/26] ch12_sunspot_periodogram")
    df = make_sunspot_data()
    x = df['sunspots'].values
    n = len(x)
    # Detrend lightly
    x = x - x.mean()
    # Periodogram via FFT
    X = fft(x)
    power = (np.abs(X) ** 2) / n
    freqs = fftfreq(n, d=1 / 12)  # cycles per year
    # Keep positive freqs
    pos = freqs > 0
    freqs_pos = freqs[pos]
    power_pos = power[pos]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(freqs_pos, power_pos, color=MAIN_BLUE, linewidth=0.8, alpha=0.85)
    # Mark 11-year peak
    peak_idx = np.argmax(power_pos[(freqs_pos > 0.05) & (freqs_pos < 0.2)])
    peak_freq = freqs_pos[(freqs_pos > 0.05) & (freqs_pos < 0.2)][peak_idx]
    peak_pow = power_pos[(freqs_pos > 0.05) & (freqs_pos < 0.2)][peak_idx]
    ax.axvline(peak_freq, color=CRIMSON, linestyle='--', linewidth=1.5,
               label=f'Peak ≈ {1/peak_freq:.1f} yr cycle')
    ax.scatter([peak_freq], [peak_pow], color=CRIMSON, zorder=5, s=40)
    ax.set_xlabel('Frequency (cycles per year)')
    ax.set_ylabel('Power (log scale)')
    ax.set_title('Periodogram of Monthly Sunspot Numbers', fontweight='bold')
    ax.set_xlim(0, 2)
    ax.legend()
    ax.text(peak_freq + 0.03, peak_pow * 0.3, f'~{1/peak_freq:.0f} years',
            color=CRIMSON, fontsize=10)
    plt.tight_layout()
    save_chart(fig, 'ch12_sunspot_periodogram')


# ===========================================================================
# CHART 3: Fourier Decomposition
# ===========================================================================
def ch12_fourier_decomposition():
    print("  [3/26] ch12_fourier_decomposition")
    t = np.linspace(0, 1, 400)
    f1 = np.sin(2 * np.pi * 3 * t)
    f2 = 0.5 * np.sin(2 * np.pi * 7 * t)
    f3 = 0.3 * np.sin(2 * np.pi * 15 * t + 0.5)
    composite = f1 + f2 + f3

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    components = [
        (composite, MAIN_BLUE, 'Composite Signal  $x(t) = f_1 + f_2 + f_3$', True),
        (f1, CRIMSON,  r'$f_1(t) = \sin(2\pi \cdot 3t)$  [3 Hz]', False),
        (f2, FOREST,   r'$f_2(t) = 0.5\sin(2\pi \cdot 7t)$  [7 Hz]', False),
        (f3, AMBER,    r'$f_3(t) = 0.3\sin(2\pi \cdot 15t + 0.5)$  [15 Hz]', False),
    ]
    for ax, (y, c, lbl, bold) in zip(axes, components):
        ax.plot(t, y, color=c, linewidth=1.8 if bold else 1.4)
        ax.axhline(0, color='black', linewidth=0.4, alpha=0.4)
        ax.set_ylabel(lbl, fontsize=9.5, color=c, fontweight='bold' if bold else 'normal')
        ax.set_ylim(-2.0, 2.0)

    axes[-1].set_xlabel('Time')
    fig.suptitle('Fourier Decomposition: Signal as Sum of Sine Waves',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    save_chart(fig, 'ch12_fourier_decomposition')


# ===========================================================================
# CHART 4: Aliasing
# ===========================================================================
def ch12_aliasing():
    print("  [4/26] ch12_aliasing")
    # Sampling rate fs = 10 Hz
    fs = 10
    t_cont = np.linspace(0, 1, 2000)
    t_samp = np.arange(0, 1, 1 / fs)

    f_true = 3        # 3 Hz – true signal
    f_alias = fs - f_true  # 7 Hz – looks identical at these sample points

    y_true  = np.sin(2 * np.pi * f_true  * t_cont)
    y_alias = np.sin(2 * np.pi * f_alias * t_cont)
    y_samp  = np.sin(2 * np.pi * f_true  * t_samp)  # same at sample points!

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_cont, y_true,  color=MAIN_BLUE, linewidth=1.8, label=f'True signal ({f_true} Hz)', zorder=3)
    ax.plot(t_cont, y_alias, color=CRIMSON, linewidth=1.8, linestyle='--',
            label=f'Aliased signal ({f_alias} Hz)', zorder=2)
    ax.scatter(t_samp, y_samp, color=FOREST, zorder=5, s=60, label='Sample points (fs=10 Hz)')
    ax.axhline(0, color='black', linewidth=0.4, alpha=0.4)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Aliasing: Two Different Frequencies Appear Identical at Sample Points',
                 fontweight='bold')
    ax.legend(loc='upper right')
    ax.text(0.02, 0.05,
            f'Nyquist freq = {fs/2} Hz\n{f_alias} Hz aliases to {f_true} Hz',
            transform=ax.transAxes, fontsize=10, color=CRIMSON,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
    plt.tight_layout()
    save_chart(fig, 'ch12_aliasing')


# ===========================================================================
# CHART 5: Spectral Density Gallery (2×2)
# ===========================================================================
def ch12_spectral_density_gallery():
    print("  [5/26] ch12_spectral_density_gallery")
    np.random.seed(42)
    n = 256
    freqs = np.linspace(0, 0.5, 300)

    processes = [
        ('White Noise',            np.random.randn(n),    None,           None),
        ('AR(1)  φ = 0.9',         ar1_process(0.9, n),  0.9,            None),
        ('AR(1)  φ = −0.7',        ar1_process(-0.7, n), -0.7,           None),
        ('AR(2)  φ₁=1.0, φ₂=−0.5', ar2_process(1.0, -0.5, n), None, (1.0, -0.5)),
    ]
    colors_g = [MAIN_BLUE, CRIMSON, FOREST, PURPLE]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for col, (name, x, phi, phi2) in enumerate(processes):
        # Time series
        ax_ts = axes[0, col]
        ax_ts.plot(x[:100], color=colors_g[col], linewidth=1.0)
        ax_ts.set_title(name, fontweight='bold', fontsize=10.5)
        ax_ts.set_xlabel('t', fontsize=9)
        if col == 0:
            ax_ts.set_ylabel('Value', fontsize=9)
        ax_ts.tick_params(labelsize=8)

        # Spectral density
        ax_sp = axes[1, col]
        f_pg, P_pg = signal.periodogram(x, fs=1.0)
        ax_sp.semilogy(f_pg[1:], P_pg[1:], color=colors_g[col], linewidth=0.7, alpha=0.6)
        # Overlay theoretical
        if phi is not None:
            S_th = theoretical_ar1_spectrum(phi, freqs)
            scale = np.mean(P_pg[1:]) / np.mean(S_th)
            ax_sp.semilogy(freqs, S_th * scale, 'k--', linewidth=1.4, label='Theoretical')
        elif phi2 is not None:
            S_th = theoretical_ar2_spectrum(phi2[0], phi2[1], freqs)
            scale = np.mean(P_pg[1:]) / np.mean(S_th)
            ax_sp.semilogy(freqs, S_th * scale, 'k--', linewidth=1.4, label='Theoretical')
        ax_sp.set_xlabel('Frequency', fontsize=9)
        if col == 0:
            ax_sp.set_ylabel('Spectral density', fontsize=9)
        ax_sp.tick_params(labelsize=8)

    axes[0, 0].set_title('White Noise', fontweight='bold', fontsize=10.5)
    fig.text(0.5, 1.01, 'Spectral Density Gallery: Four Processes',
             ha='center', fontweight='bold', fontsize=13)
    axes[1, 1].legend(fontsize=8)
    plt.tight_layout()
    save_chart(fig, 'ch12_spectral_density_gallery')


# ===========================================================================
# CHART 6: Periodogram Inconsistency
# ===========================================================================
def ch12_periodogram_inconsistency():
    print("  [6/26] ch12_periodogram_inconsistency")
    np.random.seed(0)
    phi = 0.8
    n = 512
    x = ar1_process(phi, n)

    f_raw, P_raw = signal.periodogram(x, fs=1.0)
    # Daniell smoother
    window = signal.windows.hann(51)
    window /= window.sum()
    P_smooth = np.convolve(P_raw, window, mode='same')

    freqs_th = np.linspace(0.001, 0.499, 400)
    S_th = theoretical_ar1_spectrum(phi, freqs_th)
    scale = np.mean(P_raw[1:]) / np.mean(S_th)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (P_plot, title, col) in zip(axes, [
        (P_raw,    'Raw Periodogram  (highly variable)', MAIN_BLUE),
        (P_smooth, 'Smoothed Estimate  (consistent)',   FOREST),
    ]):
        ax.semilogy(f_raw[1:], P_plot[1:], color=col, linewidth=0.9, alpha=0.85)
        ax.semilogy(freqs_th, S_th * scale, color=CRIMSON, linewidth=2.0,
                    linestyle='--', label='True spectrum')
        ax.set_title(title, fontweight='bold', fontsize=11.5)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Spectral density')
        ax.legend()

    fig.suptitle('Periodogram Inconsistency: Raw vs. Smoothed (AR(1), φ = 0.8)',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_chart(fig, 'ch12_periodogram_inconsistency')


# ===========================================================================
# CHART 7: Smoothing Method Comparison
# ===========================================================================
def ch12_smoothing_comparison():
    print("  [7/26] ch12_smoothing_comparison")
    np.random.seed(1)
    phi = 0.7
    n = 512
    x = ar1_process(phi, n)

    # Raw periodogram
    f_raw, P_raw = signal.periodogram(x, fs=1.0)
    pos = f_raw > 0

    # Daniell (hanning convolution)
    win_d = signal.windows.hann(31); win_d /= win_d.sum()
    P_daniell = np.convolve(P_raw, win_d, mode='same')

    # Welch
    f_welch, P_welch = signal.welch(x, fs=1.0, nperseg=128, noverlap=64,
                                     window='hann')

    # Multitaper proxy: average periodograms using several DPSS tapers
    tapers = signal.windows.dpss(len(x), 4, Kmax=4)
    P_mt_sum = np.zeros(len(f_raw))
    for taper in tapers:
        _, P_tap = signal.periodogram(x * taper, fs=1.0)
        P_mt_sum += P_tap
    P_mt2 = P_mt_sum / len(tapers)
    f_mt2 = f_raw

    freqs_th = np.linspace(0.001, 0.499, 400)
    S_th = theoretical_ar1_spectrum(phi, freqs_th)
    scale = np.mean(P_raw[pos]) / np.mean(S_th)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(f_raw[pos], P_raw[pos], color=GRAY, linewidth=0.6, alpha=0.5,
                label='Raw periodogram')
    ax.semilogy(f_raw[pos], P_daniell[pos], color=MAIN_BLUE, linewidth=1.8,
                label='Daniell smoother')
    ax.semilogy(f_welch, P_welch, color=FOREST, linewidth=1.8,
                label='Welch (50% overlap)')
    ax.semilogy(f_mt2, P_mt2, color=ORANGE, linewidth=1.8, linestyle='-.',
                label='Multitaper (DPSS, K=4)')
    ax.semilogy(freqs_th, S_th * scale, color=CRIMSON, linewidth=2.2,
                linestyle='--', label='True spectrum')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Spectral density')
    ax.set_title('Spectral Estimation Methods Compared (AR(1), φ = 0.7)',
                 fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    save_chart(fig, 'ch12_smoothing_comparison')


# ===========================================================================
# CHART 8: Lag Windows
# ===========================================================================
def ch12_lag_windows():
    print("  [8/26] ch12_lag_windows")
    M = 50
    lags = np.arange(-M, M + 1)
    tau = lags / M

    bartlett = np.where(np.abs(tau) <= 1, 1 - np.abs(tau), 0)
    # Parzen
    parzen = np.where(
        np.abs(tau) <= 0.5,
        1 - 6 * tau**2 + 6 * np.abs(tau)**3,
        np.where(np.abs(tau) <= 1, 2 * (1 - np.abs(tau))**3, 0)
    )
    # Tukey-Hanning
    tukey = np.where(np.abs(tau) <= 1, 0.5 * (1 + np.cos(np.pi * tau)), 0)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(lags, bartlett, color=MAIN_BLUE,  linewidth=2.2, label='Bartlett')
    ax.plot(lags, parzen,   color=CRIMSON,    linewidth=2.2, label='Parzen')
    ax.plot(lags, tukey,    color=FOREST,     linewidth=2.2, label='Tukey–Hanning')
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.4)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.4)
    ax.set_xlabel('Lag  τ')
    ax.set_ylabel('Window weight  w(τ)')
    ax.set_title('Common Lag Windows (M = 50)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_chart(fig, 'ch12_lag_windows')


# ===========================================================================
# CHART 9: Multitaper DPSS Tapers
# ===========================================================================
def ch12_multitaper_dpss():
    print("  [9/26] ch12_multitaper_dpss")
    N = 256
    NW = 4        # time-half-bandwidth
    K  = 5        # number of tapers
    tapers, _ = signal.windows.dpss(N, NW, Kmax=K, return_ratios=True)

    taper_colors = [MAIN_BLUE, CRIMSON, FOREST, AMBER, PURPLE]
    fig, axes = plt.subplots(K, 1, figsize=(12, 9), sharex=True)
    for k, (taper, ax, col) in enumerate(zip(tapers, axes, taper_colors)):
        ax.plot(taper, color=col, linewidth=1.5)
        ax.axhline(0, color='black', linewidth=0.4, alpha=0.3)
        ax.set_ylabel(f'Taper {k}', fontsize=10, color=col)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel('Sample index')
    fig.suptitle(f'DPSS Slepian Tapers  (N={N}, NW={NW})', fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_chart(fig, 'ch12_multitaper_dpss')


# ===========================================================================
# CHART 10: Estimation Method Comparison with True Spectrum
# ===========================================================================
def ch12_estimation_comparison():
    print("  [10/26] ch12_estimation_comparison")
    np.random.seed(5)
    phi1, phi2 = 1.0, -0.5
    n = 512
    x = ar2_process(phi1, phi2, n)

    f_raw, P_raw = signal.periodogram(x, fs=1.0)
    f_welch, P_welch = signal.welch(x, fs=1.0, nperseg=128, noverlap=64)
    # Multitaper
    try:
        tapers, _ = signal.windows.dpss(128, 4, Kmax=6, return_ratios=True)
        mt_psds = []
        for tap in tapers:
            _, P_tap = signal.welch(x, fs=1.0, nperseg=128, noverlap=0,
                                     window=tap)
            mt_psds.append(P_tap)
        f_mt = f_welch
        P_mt = np.mean(mt_psds, axis=0)
    except Exception:
        f_mt, P_mt = f_welch, P_welch

    freqs_th = np.linspace(0.001, 0.499, 400)
    S_th = theoretical_ar2_spectrum(phi1, phi2, freqs_th)
    pos = f_raw > 0
    scale = np.mean(P_raw[pos]) / np.mean(S_th)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(f_raw[pos],  P_raw[pos],  color=GRAY,      linewidth=0.6, alpha=0.5, label='Raw periodogram')
    ax.semilogy(f_welch,     P_welch,     color=MAIN_BLUE, linewidth=1.8, label='Welch')
    ax.semilogy(f_mt,        P_mt,        color=ORANGE,    linewidth=1.8, linestyle='-.', label='Multitaper')
    ax.semilogy(freqs_th, S_th * scale,   color=CRIMSON,   linewidth=2.5, linestyle='--', label='True spectrum')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Spectral density')
    ax.set_title('Spectral Estimators vs. True Spectrum  (AR(2), φ₁=1.0, φ₂=−0.5)',
                 fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_chart(fig, 'ch12_estimation_comparison')


# ===========================================================================
# CHART 11: Coherence Example
# ===========================================================================
def ch12_coherence_example():
    print("  [11/26] ch12_coherence_example")
    np.random.seed(20)
    n = 512
    t = np.arange(n)
    # Common driving process
    common = ar1_process(0.8, n, seed=20)
    x = common + 0.5 * np.random.randn(n)
    # y is x delayed by 5 steps + independent noise
    delay = 8
    y = np.roll(common, delay) + 0.8 * np.random.randn(n)

    f_c, Cxy = signal.coherence(x, y, fs=1.0, nperseg=128, noverlap=64)
    # Phase from cross-spectrum
    f_p, Pxy = signal.csd(x, y, fs=1.0, nperseg=128, noverlap=64)
    phase = np.angle(Pxy, deg=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(f_c, Cxy, color=MAIN_BLUE, linewidth=1.8)
    ax1.axhline(0.5, color=CRIMSON, linestyle='--', linewidth=1.2, alpha=0.8,
                label='Threshold 0.5')
    ax1.set_ylabel('Squared Coherence')
    ax1.set_title('Coherence and Phase Between Related Processes', fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.legend()

    ax2.plot(f_p, phase, color=FOREST, linewidth=1.4)
    ax2.axhline(0, color='black', linewidth=0.5, alpha=0.4)
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Phase (degrees)')
    # Annotate expected phase from delay
    ax2.text(0.6, 0.85, f'y = x shifted by {delay} steps',
             transform=ax2.transAxes, fontsize=10, color=MAIN_BLUE, style='italic')

    plt.tight_layout()
    save_chart(fig, 'ch12_coherence_example')


# ===========================================================================
# CHART 12: Filter Transfer Functions
# ===========================================================================
def ch12_filter_transfer():
    print("  [12/26] ch12_filter_transfer")
    freqs = np.linspace(0, 0.5, 1000)
    omega = 2 * np.pi * freqs

    # First difference: H(e^iw) = 1 - e^{-iw}
    H_fd = np.abs(1 - np.exp(-1j * omega)) ** 2

    # 5-point moving average
    M = 5
    # H(w) = (1/M) sum_{k=0}^{M-1} e^{-ikw}
    H_ma = np.zeros(len(omega), dtype=complex)
    for k in range(M):
        H_ma += np.exp(-1j * k * omega)
    H_ma = (np.abs(H_ma) / M) ** 2

    # Ideal band-pass (business cycle: 6–32 quarters)
    f_low  = 1 / 32
    f_high = 1 / 6
    H_bp = np.where((freqs >= f_low) & (freqs <= f_high), 1.0, 0.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, H_fd, color=MAIN_BLUE, linewidth=2.0, label='First difference')
    ax.plot(freqs, H_ma, color=FOREST,   linewidth=2.0, label=f'{M}-pt Moving average')
    ax.plot(freqs, H_bp, color=CRIMSON,  linewidth=2.0, linestyle='--',
            label='Ideal band-pass (6–32 periods)')
    ax.axvline(f_low,  color=CRIMSON, linewidth=0.8, alpha=0.5)
    ax.axvline(f_high, color=CRIMSON, linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Frequency (cycles per period)')
    ax.set_ylabel('Squared Gain  |H(ω)|²')
    ax.set_title('Filter Transfer Functions', fontweight='bold')
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.05, 4.2)
    ax.legend()
    ax.text(0.18, 0.7, 'Business\ncycle band', transform=ax.transAxes,
            fontsize=9.5, color=CRIMSON, ha='center')
    plt.tight_layout()
    save_chart(fig, 'ch12_filter_transfer')


# ===========================================================================
# CHART 13: HP, BK, CF Filter Transfer Functions
# ===========================================================================
def ch12_hp_bk_cf_transfer():
    print("  [13/26] ch12_hp_bk_cf_transfer")
    freqs = np.linspace(1e-4, 0.5, 2000)
    omega = 2 * np.pi * freqs

    # HP filter gain (quarterly, lambda=1600)
    lam = 1600
    H_hp = (4 * lam * (1 - np.cos(omega)) ** 2 /
            (1 + 4 * lam * (1 - np.cos(omega)) ** 2))

    # BK filter: approximate as cosine band-pass (6-32 quarters, K=12 leads/lags)
    f_low, f_high = 1 / 32, 1 / 6
    K = 12
    # Ideal minus truncated leakage – approximate analytically
    H_bk = np.zeros_like(freqs)
    for k in range(-K, K + 1):
        if k == 0:
            a_k = 2 * (f_high - f_low)
        else:
            a_k = (np.sin(2 * np.pi * f_high * k) - np.sin(2 * np.pi * f_low * k)) / (np.pi * k)
        H_bk += a_k * np.cos(k * omega)
    H_bk = np.clip(H_bk, 0, None)

    # CF filter: approximate as asymmetric band-pass
    H_cf = np.where((freqs >= f_low) & (freqs <= f_high),
                    0.9 * np.ones_like(freqs),
                    0.1 * np.exp(-50 * (freqs - np.where(freqs < f_low, f_low, f_high)) ** 2))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(freqs, H_hp, color=MAIN_BLUE, linewidth=2.2, label='HP filter (λ=1600)')
    ax.plot(freqs, H_bk, color=CRIMSON,   linewidth=2.2, label='BK filter (6–32 qtrs, K=12)')
    ax.plot(freqs, H_cf, color=FOREST,    linewidth=2.2, linestyle='-.', label='CF filter')
    ax.axvspan(f_low, f_high, alpha=0.1, color=AMBER, label='Business cycle band')
    ax.axvline(f_low,  color=GRAY, linewidth=0.8, linestyle=':')
    ax.axvline(f_high, color=GRAY, linewidth=0.8, linestyle=':')
    ax.set_xlabel('Frequency (cycles per quarter)')
    ax.set_ylabel('Gain  |H(ω)|')
    ax.set_title('Transfer Functions: HP, BK, and CF Filters', fontweight='bold')
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc='upper left')
    ax.text(0.5 * (f_low + f_high), -0.04, '6–32 qtrs', ha='center', fontsize=9,
            color=AMBER, va='top')
    plt.tight_layout()
    save_chart(fig, 'ch12_hp_bk_cf_transfer')


# ===========================================================================
# CHART 14: Spectral Leakage
# ===========================================================================
def ch12_spectral_leakage():
    print("  [14/26] ch12_spectral_leakage")
    np.random.seed(3)
    n = 128
    # Signal: single sinusoid at non-integer frequency
    f0 = 0.123
    t = np.arange(n)
    x = np.sin(2 * np.pi * f0 * t) + 0.3 * np.random.randn(n)

    # No taper
    X_rect  = fft(x)
    P_rect  = (np.abs(X_rect[:n // 2]) ** 2) / n

    # Hanning taper
    win = signal.windows.hann(n)
    X_hann = fft(x * win)
    P_hann = (np.abs(X_hann[:n // 2]) ** 2) / np.sum(win ** 2)

    freqs_dft = np.arange(n // 2) / n

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (P, title, col) in zip(axes, [
        (P_rect, 'No Taper (rectangular window) — heavy leakage', MAIN_BLUE),
        (P_hann, 'Hanning Taper — reduced leakage',              FOREST),
    ]):
        ax.semilogy(freqs_dft, P, color=col, linewidth=1.0)
        ax.axvline(f0, color=CRIMSON, linestyle='--', linewidth=1.5, label=f'True freq = {f0}')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')
        ax.legend()

    fig.suptitle('Spectral Leakage: Effect of Tapering', fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_chart(fig, 'ch12_spectral_leakage')


# ===========================================================================
# CHART 15: Wavelet vs. Fourier – Time-Frequency Resolution
# ===========================================================================
def ch12_wavelet_vs_fourier():
    print("  [15/26] ch12_wavelet_vs_fourier")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, (title, x_boxes, y_boxes, col, subtitle) in zip(axes, [
        ('Fourier Analysis',
         [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
         [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)],
         MAIN_BLUE,
         'Full time extent, narrow frequency bins'),
        ('Wavelet Analysis',
         [(0.0, 1.0), (0.0, 0.5), (0.0, 0.25), (0.0, 0.125)],
         [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)],
         CRIMSON,
         'Adaptive time-frequency tiling'),
    ]):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Draw tiling boxes (time on x, frequency on y)
        widths  = [x[1] - x[0] for x in x_boxes]
        heights = [y[1] - y[0] for y in y_boxes]
        for i, (xb, yb, w, h) in enumerate(zip(x_boxes, y_boxes, widths, heights)):
            rect = plt.Rectangle((xb[0], yb[0]), w, h,
                                  edgecolor=col, facecolor=col,
                                  alpha=0.15 + 0.1 * i, linewidth=1.5)
            ax.add_patch(rect)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontweight='bold', fontsize=12.5, color=col)
        ax.text(0.5, -0.12, subtitle, ha='center', transform=ax.transAxes,
                fontsize=9.5, style='italic', color=GRAY)

    fig.suptitle('Time–Frequency Resolution: Fourier vs. Wavelet',
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    save_chart(fig, 'ch12_wavelet_vs_fourier')


# ===========================================================================
# CHART 16: Morlet Wavelet at Different Scales
# ===========================================================================
def ch12_morlet_wavelet():
    print("  [16/26] ch12_morlet_wavelet")
    t = np.linspace(-4, 4, 800)
    f0 = 1.0  # center frequency
    scales = [0.5, 1.0, 2.0, 4.0]
    colors_m = [MAIN_BLUE, CRIMSON, FOREST, PURPLE]

    def morlet(t_vals, scale):
        """Real part of Morlet wavelet at given scale."""
        t_scaled = t_vals / scale
        return (np.pi * scale) ** (-0.25) * np.exp(-0.5 * t_scaled ** 2) * np.cos(2 * np.pi * f0 * t_scaled)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=False)
    for ax, scale, col in zip(axes.flat, scales, colors_m):
        psi = morlet(t, scale)
        envelope = (np.pi * scale) ** (-0.25) * np.exp(-0.5 * (t / scale) ** 2)
        ax.plot(t, psi, color=col, linewidth=1.8)
        ax.fill_between(t, -envelope, envelope, alpha=0.15, color=col)
        ax.axhline(0, color='black', linewidth=0.4, alpha=0.4)
        ax.set_title(f'Scale s = {scale}', fontweight='bold', fontsize=11, color=col)
        ax.set_xlabel('Time')
        ax.tick_params(labelsize=8)

    fig.suptitle('Morlet Wavelet at Different Scales', fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_chart(fig, 'ch12_morlet_wavelet')


# ===========================================================================
# CHART 17: Scalogram of Synthetic Chirp Signal
# ===========================================================================
def ch12_scalogram_synthetic():
    print("  [17/26] ch12_scalogram_synthetic")
    np.random.seed(9)
    fs = 200
    T = 2.0
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    # Chirp: frequency sweeps from 5 to 50 Hz
    chirp = signal.chirp(t, f0=5, f1=50, t1=T, method='linear')
    # Add a burst at 80 Hz between 0.5 and 1.0 seconds
    burst = np.zeros_like(t)
    burst_mask = (t >= 0.5) & (t <= 1.0)
    burst[burst_mask] = 0.6 * np.sin(2 * np.pi * 80 * t[burst_mask])
    sig = chirp + burst + 0.1 * np.random.randn(len(t))

    # Compute scalogram using STFT
    f_stft, t_stft, Sxx = signal.spectrogram(sig, fs=fs, nperseg=64, noverlap=56,
                                               window='hann')

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 2])
    axes[0].plot(t, sig, color=MAIN_BLUE, linewidth=0.8)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Signal: Linear Chirp (5→50 Hz) + 80 Hz Burst (0.5–1.0 s)',
                      fontweight='bold', fontsize=11.5)
    axes[0].set_xlim(0, T)

    im = axes[1].pcolormesh(t_stft, f_stft, 10 * np.log10(Sxx + 1e-10),
                             shading='gouraud', cmap='viridis')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Scalogram (STFT Magnitude)', fontweight='bold', fontsize=11.5)
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('Power (dB)')

    plt.tight_layout()
    save_chart(fig, 'ch12_scalogram_synthetic')


# ===========================================================================
# CHART 18: GDP Time Series
# ===========================================================================
def ch12_gdp_timeseries():
    print("  [18/26] ch12_gdp_timeseries")
    df = make_gdp_data()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    axes[0].plot(df['date'], df['log_gdp'], color=MAIN_BLUE, linewidth=1.8)
    axes[0].set_title('Log Real GDP (US-like, Quarterly 1960–2023)',
                      fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Log GDP')

    # Growth rate
    growth = df['log_gdp'].diff() * 100
    axes[1].bar(df['date'], growth, color=np.where(growth >= 0, MAIN_BLUE, CRIMSON),
                width=60, alpha=0.75)
    axes[1].axhline(0, color='black', linewidth=0.6)
    axes[1].set_ylabel('Quarter-on-Quarter Growth (%)')
    axes[1].set_xlabel('Year')
    axes[1].set_title('Quarterly Growth Rate', fontweight='bold', fontsize=12)

    # Mark COVID
    for ax in axes:
        ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01'),
                   alpha=0.15, color=CRIMSON, label='COVID-19')
    axes[0].legend(loc='upper left')

    plt.tight_layout()
    save_chart(fig, 'ch12_gdp_timeseries')


# ===========================================================================
# CHART 19: GDP Periodogram + Multitaper
# ===========================================================================
def ch12_gdp_periodogram():
    print("  [19/26] ch12_gdp_periodogram")
    df = make_gdp_data()
    x = df['log_gdp'].diff().dropna().values

    f_raw, P_raw = signal.periodogram(x, fs=4.0)  # 4 quarters/year
    f_mt, P_mt   = signal.welch(x, fs=4.0, nperseg=min(64, len(x)//4), noverlap=32)
    pos = f_raw > 0

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(f_raw[pos], P_raw[pos], color=GRAY, linewidth=0.7, alpha=0.6,
                label='Raw periodogram')
    ax.semilogy(f_mt, P_mt, color=MAIN_BLUE, linewidth=2.2, label='Multitaper estimate')
    # Mark business cycle band
    ax.axvspan(4/32, 4/6, alpha=0.15, color=AMBER, label='Business cycle (6–32 qtrs)')
    ax.set_xlabel('Frequency (cycles per year)')
    ax.set_ylabel('Spectral density')
    ax.set_title('Spectral Analysis of GDP Growth (Quarterly)', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save_chart(fig, 'ch12_gdp_periodogram')


# ===========================================================================
# CHART 20: GDP Business Cycle Extraction (HP, BK, CF)
# ===========================================================================
def ch12_gdp_bandpass():
    print("  [20/26] ch12_gdp_bandpass")
    df = make_gdp_data()
    y = df['log_gdp'].values
    n = len(y)

    # ---- HP filter ----
    lam = 1600
    # Construct HP filter matrix (tridiagonal)
    from scipy.linalg import solve_banded
    e = np.ones(n)
    D2 = (np.diag(e[:-2], -2) - 4 * np.diag(e[:-1], -1) + 6 * np.diag(e, 0)
          - 4 * np.diag(e[:-1], 1) + np.diag(e[:-2], 2))
    D2[0, :] = D2[-1, :] = D2[1, :] = D2[-2, :] = 0
    D2[0, 0] = D2[-1, -1] = 1
    D2[1, 1] = D2[-2, -2] = 1
    A = np.eye(n) + lam * D2.T @ D2
    try:
        from numpy.linalg import solve
        trend_hp = solve(A, y)
    except Exception:
        trend_hp = y.copy()
    cycle_hp = y - trend_hp

    # ---- BK filter (approximate with butter band-pass) ----
    f_low, f_high = 1/32, 1/6  # in cycles/quarter
    try:
        sos = signal.butter(4, [f_low * 2, f_high * 2], btype='band', output='sos')
        cycle_bk = signal.sosfiltfilt(sos, y)
    except Exception:
        cycle_bk = cycle_hp.copy()

    # ---- CF filter (approximate with band-pass FIR) ----
    try:
        K = 24
        taps = signal.firwin(2 * K + 1, [f_low * 2, f_high * 2], pass_zero=False)
        cycle_cf = np.convolve(y, taps, mode='same')
    except Exception:
        cycle_cf = cycle_bk.copy()

    dates = df['date']
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(dates, y, color=MAIN_BLUE, linewidth=1.5, label='Log GDP')
    axes[0].plot(dates, trend_hp, color=CRIMSON, linewidth=1.8,
                 linestyle='--', label='HP Trend')
    axes[0].set_ylabel('Log GDP')
    axes[0].set_title('HP Filter: Trend Extraction', fontweight='bold')
    axes[0].legend()

    axes[1].plot(dates, cycle_hp, color=CRIMSON,   linewidth=1.6, label='HP cycle')
    axes[1].plot(dates, cycle_bk, color=FOREST,    linewidth=1.6, label='BK cycle')
    axes[1].plot(dates, cycle_cf, color=ORANGE,    linewidth=1.6, linestyle='-.', label='CF cycle')
    axes[1].axhline(0, color='black', linewidth=0.5, alpha=0.4)
    axes[1].axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-01-01'),
                    alpha=0.15, color=CRIMSON)
    axes[1].set_ylabel('Cyclical Component')
    axes[1].set_xlabel('Year')
    axes[1].set_title('Business Cycle: HP, BK, and CF Components', fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    save_chart(fig, 'ch12_gdp_bandpass')


# ===========================================================================
# CHART 21: GDP Wavelet Scalogram
# ===========================================================================
def ch12_gdp_scalogram():
    print("  [21/26] ch12_gdp_scalogram")
    df = make_gdp_data()
    # Detrend: remove HP-filter trend (lambda=1600 for quarterly data)
    from scipy.signal import filtfilt
    log_gdp = df['log_gdp'].values
    # Simple HP detrend via double-pass Butterworth low-pass (approx HP)
    # HP cycle ≈ log_gdp - trend; use direct HP implementation
    T = len(log_gdp)
    # HP filter matrix approach
    q = 1600  # standard quarterly lambda
    I_T = np.eye(T)
    D2 = np.zeros((T - 2, T))
    for i in range(T - 2):
        D2[i, i] = 1
        D2[i, i + 1] = -2
        D2[i, i + 2] = 1
    trend = np.linalg.solve(I_T + q * D2.T @ D2, log_gdp)
    cycle = log_gdp - trend  # detrended log GDP

    dates = df['date'].values
    fs = 4.0  # quarters per year

    # Morlet CWT using PyWavelets
    # Scales correspond to periods from 2 to 64 quarters
    periods = np.arange(2, 65, 0.5)  # in quarters
    # For cmor wavelet: scale = (center_freq * fs) / pseudo_freq
    # Using cmor1.5-1.0 (bandwidth=1.5, center=1.0)
    wavelet = 'cmor1.5-1.0'
    central_freq = pywt.central_frequency(wavelet)
    scales = central_freq * fs / (1.0 / periods)  # scale = central_freq * fs * period
    # Simplify: scale = central_freq * fs * period_in_samples / fs = central_freq * period_in_quarters
    scales = central_freq * periods

    coefs, freqs = pywt.cwt(cycle, scales, wavelet, sampling_period=1.0/fs)
    power = np.abs(coefs) ** 2

    fig, ax = plt.subplots(figsize=(13, 5.5))
    # Convert to periods in quarters for y-axis
    periods_out = 1.0 / freqs  # freqs in cycles/quarter → periods in quarters

    im = ax.pcolormesh(dates, periods_out, power,
                       shading='gouraud', cmap='jet')
    ax.set_ylabel('Period (quarters)')
    ax.set_xlabel('Year')
    ax.set_title('Morlet Wavelet Scalogram — Detrended Log Real GDP',
                 fontweight='bold')
    ax.set_yscale('log')
    ax.set_ylim(2, 64)
    ax.set_yticks([2, 4, 6, 8, 16, 32, 64])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    # Business cycle band: 6–32 quarters
    ax.axhline(6, color='white', linewidth=1.2, linestyle='--', alpha=0.9)
    ax.axhline(32, color='white', linewidth=1.2, linestyle='--', alpha=0.9)
    ax.text(dates[5], 7, 'BC band: 6 quarters', color='white', fontsize=8.5,
            fontweight='bold')
    ax.text(dates[5], 36, 'BC band: 32 quarters', color='white', fontsize=8.5,
            fontweight='bold')
    # Cone of influence (approximate)
    coi = np.sqrt(2) * np.arange(T) / fs  # in quarters
    coi_right = np.sqrt(2) * np.arange(T)[::-1] / fs
    coi_full = np.minimum(coi, coi_right)
    ax.fill_between(dates, coi_full, 0.1, alpha=0.15, color='black',
                    hatch='//', label='COI')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Wavelet Power')
    plt.tight_layout()
    save_chart(fig, 'ch12_gdp_scalogram')


# ===========================================================================
# CHART 22: Quiz 1 – Spectral Density & Autocovariance
# ===========================================================================
def ch12_quiz1_spectral_density():
    print("  [22/26] ch12_quiz1_spectral_density")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: autocovariance of AR(1)
    phi = 0.7
    lags = np.arange(0, 30)
    gamma = phi ** lags  # ACVF (sigma^2=1)
    ax = axes[0]
    ax.stem(lags, gamma, linefmt=MAIN_BLUE, markerfmt='o', basefmt='k-')
    ax.set_xlabel('Lag k')
    ax.set_ylabel('γ(k)')
    ax.set_title('Autocovariance Function γ(k)', fontweight='bold')

    # Right: spectral density via the relationship S(f) = sum_k gamma(k) e^{-i2πfk}
    freqs = np.linspace(0, 0.5, 400)
    S_th = theoretical_ar1_spectrum(phi, freqs)
    # Also show the Fourier pair annotation
    ax2 = axes[1]
    ax2.plot(freqs, S_th, color=CRIMSON, linewidth=2.2)
    ax2.fill_between(freqs, S_th, alpha=0.15, color=CRIMSON)
    ax2.set_xlabel('Frequency f')
    ax2.set_ylabel('S(f)')
    ax2.set_title('Spectral Density  S(f)', fontweight='bold')
    ax2.text(0.55, 0.85,
             r'$S(f) = \sum_{k=-\infty}^{\infty} \gamma(k)\,e^{-i2\pi f k}$',
             transform=ax2.transAxes, fontsize=11, color=MAIN_BLUE,
             ha='center')

    # Arrow between panels
    fig.text(0.5, 0.55, '⟷  Fourier transform pair', ha='center',
             fontsize=11.5, color=FOREST, fontweight='bold')

    fig.suptitle('Quiz: Autocovariance ↔ Spectral Density Relationship',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_chart(fig, 'ch12_quiz1_spectral_density')


# ===========================================================================
# CHART 23: Quiz 2 – Periodogram Inconsistency
# ===========================================================================
def ch12_quiz2_inconsistency():
    print("  [23/26] ch12_quiz2_inconsistency")
    phi = 0.7
    n = 256
    n_realiz = 6
    freqs_th = np.linspace(0.001, 0.499, 400)
    S_th = theoretical_ar1_spectrum(phi, freqs_th)
    scale_ref = None

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        x = ar1_process(phi, n, seed=i * 7)
        f_raw, P_raw = signal.periodogram(x, fs=1.0)
        pos = f_raw > 0
        if scale_ref is None:
            scale_ref = np.mean(P_raw[pos]) / np.mean(S_th)
        ax.semilogy(f_raw[pos], P_raw[pos], color=MAIN_BLUE, linewidth=0.7, alpha=0.75)
        ax.semilogy(freqs_th, S_th * scale_ref, color=CRIMSON, linewidth=1.8,
                    linestyle='--')
        ax.set_title(f'Realization {i+1}', fontsize=10.5, fontweight='bold')
        ax.tick_params(labelsize=8)

    fig.text(0.5, 0.0, 'Frequency', ha='center', fontsize=12)
    fig.text(0.0, 0.5, 'Spectral density', va='center', rotation='vertical', fontsize=12)
    fig.suptitle('Periodogram Inconsistency: Variance Does Not Shrink Across Realizations\n'
                 '(red dashed = true spectrum)', fontweight='bold', fontsize=12.5)
    plt.tight_layout()
    save_chart(fig, 'ch12_quiz2_inconsistency')


# ===========================================================================
# CHART 24: Quiz 3 – Seasonal Spectrum
# ===========================================================================
def ch12_quiz3_seasonal_spectrum():
    print("  [24/26] ch12_quiz3_seasonal_spectrum")
    np.random.seed(11)
    n = 480  # 40 years monthly
    t = np.arange(n)
    # Monthly seasonal: peaks at f = k/12 for k = 1, 2, ..., 6
    x = np.zeros(n)
    amplitudes = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1]
    for k, amp in enumerate(amplitudes, 1):
        x += amp * np.sin(2 * np.pi * k * t / 12 + np.random.uniform(0, 2 * np.pi))
    x += 0.4 * np.random.randn(n)

    f_raw, P_raw = signal.periodogram(x, fs=12.0)  # cycles per year
    pos = (f_raw > 0) & (f_raw <= 6.5)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    axes[0].plot(t / 12, x, color=MAIN_BLUE, linewidth=0.8)
    axes[0].set_ylabel('Value')
    axes[0].set_xlabel('Year')
    axes[0].set_title('Monthly Seasonal Process (40 years)', fontweight='bold')

    axes[1].semilogy(f_raw[pos], P_raw[pos], color=MAIN_BLUE, linewidth=0.9)
    # Mark seasonal harmonics
    for k in range(1, 7):
        axes[1].axvline(k, color=CRIMSON, linewidth=1.4, linestyle='--', alpha=0.7,
                        label=f'f={k}/yr' if k == 1 else None)
        axes[1].text(k + 0.05, P_raw[pos].max() * 0.5 ** k,
                     f'k={k}', fontsize=8.5, color=CRIMSON)
    axes[1].set_xlabel('Frequency (cycles per year)')
    axes[1].set_ylabel('Power')
    axes[1].set_title('Periodogram: Seasonal Peaks at Integer Harmonics', fontweight='bold')
    axes[1].legend()

    fig.suptitle('Quiz: Seasonal Spectrum — Peaks at f = k / period',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_chart(fig, 'ch12_quiz3_seasonal_spectrum')


# ===========================================================================
# CHART 25: Quiz 4 – Wavelet Advantage (non-stationary)
# ===========================================================================
def ch12_quiz4_wavelet_advantage():
    print("  [25/26] ch12_quiz4_wavelet_advantage")
    np.random.seed(13)
    n = 512
    t = np.linspace(0, 1, n)
    fs = n

    # Stationary: single frequency throughout
    x_stat = np.sin(2 * np.pi * 20 * t) + 0.3 * np.random.randn(n)

    # Non-stationary: frequency changes at midpoint
    x_nstat = np.zeros(n)
    x_nstat[:n // 2] = np.sin(2 * np.pi * 10 * t[:n // 2])
    x_nstat[n // 2:] = np.sin(2 * np.pi * 40 * t[n // 2:])
    x_nstat += 0.3 * np.random.randn(n)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Stationary time series
    axes[0, 0].plot(t, x_stat, color=MAIN_BLUE, linewidth=0.7)
    axes[0, 0].set_title('Stationary Signal (20 Hz constant)', fontweight='bold', fontsize=11)
    axes[0, 0].set_ylabel('Amplitude')

    # Stationary spectrum (Fourier is fine)
    f_s, P_s = signal.periodogram(x_stat, fs=fs)
    pos_s = (f_s > 0) & (f_s < 80)
    axes[1, 0].semilogy(f_s[pos_s], P_s[pos_s], color=MAIN_BLUE, linewidth=1.2)
    axes[1, 0].axvline(20, color=CRIMSON, linestyle='--', linewidth=1.5, label='True freq 20 Hz')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Power')
    axes[1, 0].set_title('Fourier: Clear single peak', fontweight='bold', fontsize=11)
    axes[1, 0].legend(fontsize=9)

    # Non-stationary time series
    axes[0, 1].plot(t[:n // 2], x_nstat[:n // 2], color=CRIMSON,  linewidth=0.7,
                    label='10 Hz segment')
    axes[0, 1].plot(t[n // 2:], x_nstat[n // 2:], color=FOREST,   linewidth=0.7,
                    label='40 Hz segment')
    axes[0, 1].set_title('Non-stationary Signal (10 Hz → 40 Hz)', fontweight='bold', fontsize=11)
    axes[0, 1].legend(fontsize=9)

    # Non-stationary scalogram
    f_st, t_st, Sxx = signal.spectrogram(x_nstat, fs=fs, nperseg=64, noverlap=56)
    pos_f = f_st < 80
    im = axes[1, 1].pcolormesh(t_st, f_st[pos_f], Sxx[pos_f, :],
                                shading='gouraud', cmap='viridis')
    axes[1, 1].axvline(0.5, color='white', linewidth=1.5, linestyle='--')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    axes[1, 1].set_title('Wavelet/STFT: Reveals Time-Varying Frequency', fontweight='bold', fontsize=11)
    plt.colorbar(im, ax=axes[1, 1], label='Power')

    fig.suptitle('Quiz: Fourier vs. Wavelet — Stationary vs. Non-stationary Signals',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_chart(fig, 'ch12_quiz4_wavelet_advantage')


# ===========================================================================
# CHART 26: Quiz 5 – HP Filter Transfer Function
# ===========================================================================
def ch12_quiz5_hp_filter():
    print("  [26/26] ch12_quiz5_hp_filter")
    freqs = np.linspace(1e-4, 0.5, 2000)
    omega = 2 * np.pi * freqs

    for lam, col, label in [
        (100,  FOREST,    'λ = 100  (annual)'),
        (1600, MAIN_BLUE, 'λ = 1600 (quarterly)'),
        (14400, ORANGE,   'λ = 14400 (monthly)'),
    ]:
        H_hp = 4 * lam * (1 - np.cos(omega)) ** 2 / (1 + 4 * lam * (1 - np.cos(omega)) ** 2)
        pass

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for lam, col, label in [
        (100,  FOREST,    'λ = 100  (annual)'),
        (1600, MAIN_BLUE, 'λ = 1600 (quarterly)'),
        (14400, ORANGE,   'λ = 14400 (monthly)'),
    ]:
        H_hp = 4 * lam * (1 - np.cos(omega)) ** 2 / (1 + 4 * lam * (1 - np.cos(omega)) ** 2)
        ax1.plot(freqs, H_hp, color=col, linewidth=2.0, label=label)

    ax1.axvspan(1/32, 1/6, alpha=0.12, color=AMBER, label='Business cycle')
    ax1.set_xlabel('Frequency (cycles per period)')
    ax1.set_ylabel('Gain  |H(ω)|')
    ax1.set_title('HP Filter: High-Pass Gain', fontweight='bold')
    ax1.legend(fontsize=9.5)
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(-0.02, 1.05)

    # Right: show what gets removed (trend) vs. retained (cycle)
    lam = 1600
    H_hp_q = 4 * lam * (1 - np.cos(omega)) ** 2 / (1 + 4 * lam * (1 - np.cos(omega)) ** 2)
    H_lp_q = 1 - H_hp_q
    ax2.fill_between(freqs, H_lp_q, alpha=0.25, color=CRIMSON, label='Trend (removed by HP)')
    ax2.fill_between(freqs, H_hp_q, alpha=0.25, color=MAIN_BLUE, label='Cycle (kept by HP)')
    ax2.plot(freqs, H_hp_q, color=MAIN_BLUE, linewidth=2.0)
    ax2.plot(freqs, H_lp_q, color=CRIMSON,   linewidth=2.0)
    ax2.set_xlabel('Frequency (cycles per quarter)')
    ax2.set_ylabel('Gain')
    ax2.set_title('HP Filter = High-Pass  (λ=1600)', fontweight='bold')
    ax2.legend(fontsize=9.5)
    ax2.set_xlim(0, 0.5)

    fig.suptitle('Quiz: HP Filter Transfer Function — It IS a High-Pass Filter',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    save_chart(fig, 'ch12_quiz5_hp_filter')


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("Generating Chapter 12: Spectral Analysis Charts")
    print("=" * 60)

    chart_functions = [
        ch12_sunspot_timeseries,
        ch12_sunspot_periodogram,
        ch12_fourier_decomposition,
        ch12_aliasing,
        ch12_spectral_density_gallery,
        ch12_periodogram_inconsistency,
        ch12_smoothing_comparison,
        ch12_lag_windows,
        ch12_multitaper_dpss,
        ch12_estimation_comparison,
        ch12_coherence_example,
        ch12_filter_transfer,
        ch12_hp_bk_cf_transfer,
        ch12_spectral_leakage,
        ch12_wavelet_vs_fourier,
        ch12_morlet_wavelet,
        ch12_scalogram_synthetic,
        ch12_gdp_timeseries,
        ch12_gdp_periodogram,
        ch12_gdp_bandpass,
        ch12_gdp_scalogram,
        ch12_quiz1_spectral_density,
        ch12_quiz2_inconsistency,
        ch12_quiz3_seasonal_spectrum,
        ch12_quiz4_wavelet_advantage,
        ch12_quiz5_hp_filter,
    ]

    success = 0
    failed  = []
    for fn in chart_functions:
        try:
            fn()
            success += 1
        except Exception as e:
            print(f"  [FAILED] {fn.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed.append(fn.__name__)

    print()
    print("=" * 60)
    print(f"Done: {success}/{len(chart_functions)} charts generated successfully.")
    if failed:
        print(f"Failed: {failed}")
    print("=" * 60)


if __name__ == '__main__':
    main()
