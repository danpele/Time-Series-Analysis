"""
generate_all_charts.py -- charts and numbers of Chapter 12 (TSA): spectral analysis
==================================================================================
Course data (tsa_data.py), chart style (tsa_style.py). Every number on the slides comes from here.
Conventions (Shumway and Stoffer 2017, ch. 4): frequency nu in cycles per observation, 0 <= nu <= 1/2, period 1/nu;
spectral density f(nu) = sum_h gamma(h) exp(-2 pi i nu h), so that the integral of f over [-1/2, 1/2] is the variance;
periodogram I(nu_j) = |d(nu_j)|^2 with d(nu_j) = T^(-1/2) sum_t x_t exp(-2 pi i nu_j t), nu_j = j / T.
  * Fourier analysis -- a series as a sum of cosines; the DFT and the periodogram of a small series by hand; aliasing;
  * spectral densities -- white noise, AR(1), MA(1), AR(2) with a pseudo-cycle, ARMA(1,1);
  * the periodogram -- sunspots 1700-2008 (statsmodels); its inconsistency for white noise; Fisher's g test;
  * estimation -- leakage and tapering; Daniell and Welch smoothing; chi-square confidence bands;
  * cycles in data -- US and Romanian real GDP growth and HP cycles (FRED GDPC1, Eurostat namq_10_gdp); the hourly
                      electricity load of Romania (ENTSO-E extract of Chapter 4): daily and weekly cycles;
  * long memory -- the spectral pole at zero of S&P 500 absolute returns, GPH (Chapter 8);
  * filters -- gains of the first and seasonal differences and of the Hodrick-Prescott filter;
  * two series -- coherence and phase of US industrial production and unemployment (FRED INDPRO, UNRATE);
  * wavelets -- a Morlet scalogram of the sunspots (pointer only).
Output: charts/tsa_ch12_*.pdf/.png, Quantlets/Ch_12/ch12_numbers.json
Run:  python3 Quantlets/Ch_12/generate_all_charts.py
Time Series Analysis - Daniel Traian PELE
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from tsa_data import load_statsmodels, log_returns, read_eurostat, read_fred   # noqa: E402
import tsa_style as st                                                        # noqa: E402
from statsmodels.tsa.filters.hp_filter import hpfilter                         # noqa: E402

warnings.filterwarnings('ignore')
SEED = 2026
RO_GDP = ('namq_10_gdp', 'Q.CLV10_MEUR.SCA.B1GQ.RO')   # Romanian real GDP, SA, chain-linked 2010 volumes (Eurostat)
GDP_END = '2019-12-31'                                  # spectra before the 2020 outlier
BC_BAND = (6, 32)                                       # business-cycle periods in quarters (Burns and Mitchell)
HP_LAMBDA = 1600                                        # Hodrick-Prescott smoothing parameter, quarterly data
LOAD_FILE = 'ch4_ro_load_hourly.csv'                    # hourly load of Romania (ENTSO-E extract, Chapter 4)
LOAD_RAW = 'https://raw.githubusercontent.com/danpele/Time-Series-Analysis/main/Quantlets/Ch_04/' + LOAD_FILE
BW = 0.65                                               # GPH bandwidth m = [T^0.65] (as in Chapter 8)
AR2 = (1.5, -0.75)                                      # AR(2) with complex roots: a pseudo-cycle
DFT_EXAMPLE = [4.0, 2.0, 0.0, 2.0]                      # the series of the worked DFT example


def save(name, save_it=True):
    if save_it:
        st.check_no_grey(plt.gcf())
        st.save_fig(name)
    else:
        plt.show()


# =============================================================================
# TOOLS
# =============================================================================
def periodogram(x, taper=False):
    """Periodogram I(nu_j) = |d(nu_j)|^2, d(nu_j) = T^(-1/2) sum_t x_t exp(-2 pi i nu_j t), at the Fourier
    frequencies nu_j = j / T, j = 1..[T/2] (mean removed). With taper=True a Hann (cosine bell) taper is applied
    and rescaled so that the total power is unchanged."""
    x = np.asarray(x, float)
    x = x - x.mean()
    T = len(x)
    if taper:
        w = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(T) + 0.5) / T))
        x = x * w / np.sqrt(np.mean(w ** 2))
    I = np.abs(np.fft.fft(x)) ** 2 / T
    j = np.arange(1, T // 2 + 1)
    return j / T, I[j]


def daniell(I, m):
    """Daniell smoother: the average of the 2m + 1 periodogram ordinates around each frequency (reflected at the
    ends); approximately chi-square with df = 2(2m + 1) degrees of freedom."""
    if m == 0:
        return np.asarray(I, float)
    k = np.ones(2 * m + 1) / (2 * m + 1)
    return np.convolve(np.pad(np.asarray(I, float), m, mode='reflect'), k, mode='valid')


def chi2_band(fhat, df, level=0.95):
    """Approximate confidence interval for f(nu): [df fhat / chi2_df(1 - a/2), df fhat / chi2_df(a/2)]."""
    a = 1 - level
    return df * fhat / stats.chi2.ppf(1 - a / 2, df), df * fhat / stats.chi2.ppf(a / 2, df)


def welch(x, nperseg):
    """Welch estimate (Hann windows, 50% overlap), rescaled to the convention f(nu) = sum_h gamma(h) e^(-2 pi i nu h)."""
    nu, P = signal.welch(np.asarray(x, float) - np.mean(x), fs=1.0, window='hann', nperseg=nperseg)
    return nu[1:], P[1:] / 2


def arma_spectrum(nu, phi=(), theta=(), sigma2=1.0):
    """Spectral density of an ARMA(p,q): sigma^2 |theta(e^(-2 pi i nu))|^2 / |phi(e^(-2 pi i nu))|^2."""
    z = np.exp(-2j * np.pi * np.asarray(nu, float))
    num = np.ones_like(z) + sum(t * z ** (k + 1) for k, t in enumerate(theta))
    den = 1 - sum(p * z ** (k + 1) for k, p in enumerate(phi))
    return sigma2 * np.abs(num) ** 2 / np.abs(den) ** 2


def simulate_arma(phi, theta, T, seed=SEED, burn=500):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T + burn)
    x = np.zeros(T + burn)
    for t in range(T + burn):
        x[t] = e[t] + sum(th * e[t - k - 1] for k, th in enumerate(theta) if t - k - 1 >= 0) \
            + sum(p * x[t - k - 1] for k, p in enumerate(phi) if t - k - 1 >= 0)
    return x[burn:]


def fisher_g(I):
    """Fisher's (1929) g test of a hidden periodicity: g = max I_j / sum I_j over j = 1..m (nu_j < 1/2), with the
    exact p-value P(g > x) = sum_k (-1)^(k-1) C(m, k) (1 - kx)^(m-1), k <= 1/x."""
    I = np.asarray(I, float)
    m = len(I)
    g = I.max() / I.sum()
    from math import comb
    p = sum((-1) ** (k - 1) * comb(m, k) * (1 - k * g) ** (m - 1) for k in range(1, int(1 / g) + 1))
    return {'g': float(g), 'm': int(m), 'p': float(min(max(p, 0.0), 1.0)), 'j': int(np.argmax(I) + 1)}


def gph_spec(x, m=None):
    """GPH (Geweke and Porter-Hudak 1983): OLS slope of log I(nu_j) on -log(4 sin^2(pi nu_j)), j = 1..m, is d;
    standard error pi / sqrt(24 m)."""
    nu, I = periodogram(x)
    m = m or int(np.floor(len(x) ** BW))
    X = -np.log(4 * np.sin(np.pi * nu[:m]) ** 2)
    b = np.polyfit(X, np.log(I[:m]), 1)
    return {'d': float(b[0]), 'c': float(b[1]), 'se': float(np.pi / np.sqrt(24 * m)), 'm': int(m),
            'nu_m': float(nu[m - 1])}


def morlet_cwt(x, periods, w0=6.0):
    """Continuous wavelet transform with a Morlet wavelet (Torrence and Compo 1998), computed by FFT;
    returns the wavelet power |W(s, t)|^2 / variance for the given Fourier periods (in observations)."""
    x = np.asarray(x, float)
    x = (x - x.mean()) / x.std()
    T = len(x)
    n = 1 << int(np.ceil(np.log2(2 * T)))
    xf = np.fft.fft(x, n)
    om = 2 * np.pi * np.fft.fftfreq(n)
    fourier_factor = 4 * np.pi / (w0 + np.sqrt(2 + w0 ** 2))
    P = np.zeros((len(periods), T))
    for i, per in enumerate(periods):
        s = per / fourier_factor
        psi = np.pi ** -0.25 * np.sqrt(2 * np.pi * s) * np.exp(-0.5 * (s * om - w0) ** 2) * (om > 0)
        P[i] = np.abs(np.fft.ifft(xf * psi)[:T]) ** 2
    return P


# =============================================================================
# DATA
# =============================================================================
def sunspots():
    """Yearly mean sunspot number 1700-2008 (Wolf / SILSO series as shipped with statsmodels)."""
    s = load_statsmodels('sunspots')
    s.index = s.index.year
    return s


def gdp_growth(end=GDP_END):
    """Quarterly growth of real GDP, 100 x log difference: United States (FRED GDPC1) and Romania (Eurostat)."""
    us = read_fred('GDPC1').loc[:end]
    ro = read_eurostat(*RO_GDP).loc[:end]
    return {'US': (100 * np.log(us)).diff().dropna(), 'RO': (100 * np.log(ro)).diff().dropna(),
            'US_level': 100 * np.log(us), 'RO_level': 100 * np.log(ro)}


def hp_cycle(level, lam=HP_LAMBDA):
    cyc, trend = hpfilter(level, lamb=lam)
    return cyc


def load_hourly():
    """Hourly electricity load of Romania, MW (ENTSO-E extract of Chapter 4), on a regular UTC hourly grid."""
    s = None
    local = [os.path.join(HERE, '..', 'Ch_04', LOAD_FILE)] + [os.path.join(d, 'Quantlets', 'Ch_04', LOAD_FILE)
                                                                for d in ('.', '..', '../..', '../../..')]
    for src in [p for p in local if os.path.exists(p)] + [LOAD_RAW]:
        try:
            s = pd.read_csv(src, index_col=0, parse_dates=True).iloc[:, 0]
            break
        except Exception:
            continue
    s = s[~s.index.duplicated()].sort_index()
    s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq='h')).interpolate()
    return s.rename('load_MW')


# =============================================================================
# 1. FOURIER ANALYSIS
# =============================================================================
def worked_examples():
    """Numbers of the worked examples: DFT of (4, 2, 0, 2); AR(1) and MA(1) spectra; the AR(2) peak; aliasing of a
    4-month cycle observed quarterly; a Daniell confidence interval."""
    x = np.array(DFT_EXAMPLE)
    T = len(x)
    y = x - x.mean()
    d = np.fft.fft(y) / np.sqrt(T)
    I = np.abs(d) ** 2
    phi1, phi2 = AR2
    c = phi1 * (phi2 - 1) / (4 * phi2)
    nu_star = float(np.arccos(c) / (2 * np.pi))
    L = 5
    lo, hi = chi2_band(1.0, 2 * L)
    nu_true = 1 / 4                         # cycles per month: a 4-month cycle
    nu_q = nu_true * 3                      # cycles per quarter when observed every 3 months
    alias = abs(nu_q - round(nu_q))
    return {'dft': {'x': x.tolist(), 'mean': float(x.mean()), 'I': I.tolist(), 'var': float(np.mean(y ** 2)),
                    'sumI': float(I[1:].sum())},
            'ar1': {'phi': 0.6, 'f0': float(arma_spectrum(0, [0.6])), 'f5': float(arma_spectrum(0.5, [0.6]))},
            'ma1': {'theta': 0.5, 'f0': float(arma_spectrum(0, [], [0.5])), 'f5': float(arma_spectrum(0.5, [], [0.5]))},
            'ar2': {'phi1': phi1, 'phi2': phi2, 'cos': float(c), 'nu': nu_star, 'period': 1 / nu_star,
                    'fpeak': float(arma_spectrum(nu_star, AR2)), 'f0': float(arma_spectrum(0, AR2))},
            'ci': {'L': L, 'df': 2 * L, 'lo': float(lo), 'hi': float(hi),
                   'q_lo': float(stats.chi2.ppf(0.025, 2 * L)), 'q_hi': float(stats.chi2.ppf(0.975, 2 * L))},
            'alias': {'nu_month': nu_true, 'nu_quarter': nu_q, 'alias': float(alias), 'period_q': float(1 / alias),
                      'period_m': float(3 / alias)}}


def fig_fourier(save_it=True):
    """A series built from two cosines plus noise, its components and its periodogram (spikes at 1/12 and 1/4)."""
    rng = np.random.default_rng(SEED)
    T = 120
    t = np.arange(1, T + 1)
    c1 = 2.0 * np.cos(2 * np.pi * t / 12)
    c2 = 1.0 * np.cos(2 * np.pi * t / 4 + 1.0)
    x = c1 + c2 + rng.normal(0, 0.7, T)
    nu, I = periodogram(x)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1.35, 1]})
    ax = axes[0]
    ax.plot(t, x, color=st.MainBlue, lw=1.2, label='series x_t')
    ax.plot(t, c1, color=st.IDAred, lw=1.1, ls='--', label='2 cos(2 pi t / 12): period 12')
    ax.plot(t, c2, color=st.Forest, lw=1.0, ls=':', label='cos(2 pi t / 4 + 1): period 4')
    ax.set_xlabel('t')
    ax.set_title('Two cycles plus noise')
    ax = axes[1]
    ax.vlines(nu, 0, I, color=st.MainBlue, lw=1.4, label='periodogram I(nu_j)')
    ax.set_xlabel('frequency nu (cycles per observation)')
    ax.set_title('Periodogram')
    for v, lab in ((1 / 12, '1/12'), (1 / 4, '1/4')):
        ax.annotate(lab, (v, I[np.argmin(abs(nu - v))]), textcoords='offset points', xytext=(6, -4), color=st.IDAred)
    st.fig_legend_bottom(fig, ncol=4, y=0.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save('tsa_ch12_fourier', save_it)
    j1, j2 = np.argmin(abs(nu - 1 / 12)), np.argmin(abs(nu - 1 / 4))
    var = np.mean((x - x.mean()) ** 2)
    return {'T': T, 'I12': float(I[j1]), 'I4': float(I[j2]), 'share': float((I[j1] + I[j2]) / I.sum()),
            'var': float(var), 'theory12': float(2.0 ** 2 * T / 4), 'theory4': float(1.0 ** 2 * T / 4)}


def fig_aliasing(save_it=True):
    """Aliasing: a 4-month cycle observed once a quarter looks like a 12-month cycle."""
    tm = np.linspace(0, 36, 1000)
    tq = np.arange(0, 37, 3)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(tm, np.cos(2 * np.pi * tm / 4), color=st.MainBlue, lw=1.1, label='true cycle: period 4 months')
    ax.plot(tm, np.cos(2 * np.pi * tm / 12), color=st.IDAred, lw=1.6, ls='--', label='alias: period 12 months')
    ax.plot(tq, np.cos(2 * np.pi * tq / 4), 'o', ms=7, color=st.Amber, label='quarterly observations')
    ax.set_xlabel('months')
    ax.set_xticks(np.arange(0, 37, 3))
    st.legend_outside_bottom(ax, ncol=3)
    fig.tight_layout()
    save('tsa_ch12_aliasing', save_it)
    return {'nyquist_q': 0.5, 'match': bool(np.allclose(np.cos(2 * np.pi * tq / 4), np.cos(2 * np.pi * tq / 12)))}


# =============================================================================
# 2. SPECTRAL DENSITIES OF ARMA MODELS
# =============================================================================
SPECTRA = [('white noise', (), ()), ('AR(1), phi = 0.6', (0.6,), ()), ('AR(1), phi = -0.6', (-0.6,), ()),
           ('MA(1), theta = 0.5', (), (0.5,)), ('AR(2), phi = (1.5, -0.75)', AR2, ()),
           ('ARMA(1,1), phi = 0.6, theta = 0.4', (0.6,), (0.4,))]


def fig_spectra(save_it=True):
    """Theoretical spectral densities (sigma^2 = 1) of six models, with their variance (area under f)."""
    nu = np.linspace(0, 0.5, 501)
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.2))
    out = {}
    for i, (ax, (lab, phi, theta)) in enumerate(zip(axes.flat, SPECTRA)):
        f = arma_spectrum(nu, phi, theta)
        ax.plot(nu, f, color=st.PALETTE[i], lw=1.8, label=lab)
        ax.fill_between(nu, 0, f, color=st.PALETTE[i], alpha=0.12, lw=0)
        ax.set_title(lab, fontsize=11.5)
        ax.set_ylim(0, f.max() * 1.12)
        if i >= 3:
            ax.set_xlabel('frequency nu')
        var = 2 * np.trapezoid(f, nu)
        out[lab] = {'f0': float(f[0]), 'f05': float(f[-1]), 'var': float(var), 'argmax': float(nu[np.argmax(f)])}
    fig.tight_layout()
    save('tsa_ch12_spectra', save_it)
    return out


# =============================================================================
# 3. THE PERIODOGRAM
# =============================================================================
def fig_sunspots(save_it=True):
    """Yearly sunspot numbers 1700-2008 and their raw periodogram against the period in years."""
    s = sunspots()
    nu, I = periodogram(s.values)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={'width_ratios': [1.3, 1]})
    axes[0].plot(s.index, s.values, color=st.Amber, lw=1.1, label='yearly mean sunspot number')
    axes[0].set_xlabel('year')
    axes[0].set_title(f'Sunspots, {s.index[0]}-{s.index[-1]}')
    ax = axes[1]
    ax.vlines(nu, 0, I, color=st.MainBlue, lw=1.2, label='periodogram I(nu_j)')
    j = int(np.argmax(I))
    ax.annotate(f'peak: nu = {nu[j]:.4f}, period {1 / nu[j]:.1f} years', (nu[j], I[j]), textcoords='offset points',
                xytext=(10, -8), color=st.IDAred)
    ax.set_xlabel('frequency nu (cycles per year)')
    ax.set_title('Raw periodogram')
    st.fig_legend_bottom(fig, ncol=2, y=0.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save('tsa_ch12_sunspots', save_it)
    band = (1 / nu >= 9) & (1 / nu <= 13)
    order = np.argsort(I)[::-1][:3]
    return {'T': int(len(s)), 'y0': int(s.index[0]), 'y1': int(s.index[-1]), 'j': j + 1, 'nu': float(nu[j]),
            'period': float(1 / nu[j]), 'share_band': float(I[band].sum() / I.sum()),
            'top3': [float(1 / nu[k]) for k in order], 'fisher': fisher_g(I[nu < 0.5]),
            'mean': float(s.mean()), 'max': float(s.max()), 'max_year': int(s.idxmax())}


def fig_inconsistency(save_it=True):
    """Periodogram of Gaussian white noise (sigma^2 = 1) for T = 128 and T = 2048: it does not settle down."""
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
    out = {}
    for ax, T, c in zip(axes, (128, 2048), (st.MainBlue, st.Purple)):
        x = rng.standard_normal(T)
        nu, I = periodogram(x)
        ax.plot(nu, I, color=c, lw=0.8, label=f'periodogram, T = {T}' if T == 128 else 'periodogram, T = 2048')
        ax.axhline(1.0, color=st.IDAred, lw=1.8, ls='--', label='true spectrum f(nu) = 1' if T == 128 else '_')
        ax.set_title(f'White noise, T = {T}')
        ax.set_xlabel('frequency nu')
        I0 = I[nu < 0.5]
        out[str(T)] = {'mean': float(I0.mean()), 'sd': float(I0.std()), 'max': float(I0.max()),
                       'q95': float(np.quantile(I0, 0.95))}
    st.fig_legend_bottom(fig, ncol=3, y=0.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save('tsa_ch12_inconsistency', save_it)
    out['chi2_q95'] = float(stats.chi2.ppf(0.95, 2) / 2)
    return out


# =============================================================================
# 4. LEAKAGE, TAPERING, SMOOTHING
# =============================================================================
def fig_leakage(save_it=True):
    """A strong cosine between two Fourier frequencies plus a weak one: raw and Hann-tapered periodograms (log scale)."""
    rng = np.random.default_rng(SEED)
    T = 256
    t = np.arange(T)
    nu1, nu2, a2 = 25.5 / T, 0.30, 0.02
    x = np.cos(2 * np.pi * nu1 * t) + a2 * np.cos(2 * np.pi * nu2 * t) + rng.normal(0, 0.002, T)
    nu, I = periodogram(x)
    _, It = periodogram(x, taper=True)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.semilogy(nu, I, color=st.IDAred, lw=1.2, label='raw periodogram (rectangular window)')
    ax.semilogy(nu, It, color=st.MainBlue, lw=1.4, label='Hann-tapered periodogram')
    ax.axvline(nu1, color=st.Amber, lw=1, ls='--', label=f'strong cycle, nu = {nu1:.4f}')
    ax.axvline(nu2, color=st.Forest, lw=1, ls='--', label=f'weak cycle, nu = {nu2:.2f} (amplitude {a2})')
    ax.set_xlabel('frequency nu')
    ax.set_ylabel('power (log scale)')
    st.legend_outside_bottom(ax, ncol=2)
    fig.tight_layout()
    save('tsa_ch12_leakage', save_it)
    k = np.argmin(abs(nu - nu2))
    near = (abs(nu - nu2) > 3 / T) & (abs(nu - nu2) < 10 / T)
    return {'T': T, 'nu1': nu1, 'nu2': nu2, 'a2': a2, 'raw_at2': float(I[k]), 'tap_at2': float(It[k]),
            'raw_bg': float(np.median(I[near])), 'tap_bg': float(np.median(It[near])),
            'raw_ratio': float(I[k] / np.median(I[near])), 'tap_ratio': float(It[k] / np.median(It[near]))}


def fig_smoothing(save_it=True):
    """AR(2) with a pseudo-cycle, T = 512: raw periodogram, Daniell (m = 2 and m = 10) and Welch against the truth."""
    T = 512
    x = simulate_arma(AR2, (), T)
    nu, I = periodogram(x)
    f = arma_spectrum(nu, AR2)
    est = {'Daniell, m = 2 (L = 5)': daniell(I, 2), 'Daniell, m = 10 (L = 21)': daniell(I, 10)}
    nw, fw = welch(x, 128)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    axes[0].plot(nu, I, color=st.Teal, lw=0.8, label='raw periodogram')
    axes[0].set_title('Raw periodogram')
    axes[1].plot(nu, est['Daniell, m = 2 (L = 5)'], color=st.MainBlue, lw=1.3, label='Daniell, m = 2 (L = 5)')
    axes[1].plot(nu, est['Daniell, m = 10 (L = 21)'], color=st.Purple, lw=1.6, label='Daniell, m = 10 (L = 21)')
    axes[1].plot(nw, fw, color=st.Orange, lw=1.4, ls='-.', label='Welch, segments of 128')
    axes[1].set_title('Smoothed estimates')
    for ax in axes:
        ax.plot(nu, f, color=st.IDAred, lw=2, ls='--', label='true AR(2) spectrum' if ax is axes[0] else '_')
        ax.set_xlabel('frequency nu')
        ax.set_xlim(0, 0.5)
    st.fig_legend_bottom(fig, ncol=5, y=0.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save('tsa_ch12_smoothing', save_it)
    out = {'T': T, 'true_peak': float(nu[np.argmax(f)]), 'raw_peak': float(nu[np.argmax(I)])}
    lf = np.log(f)
    out['mse_raw'] = float(np.mean((np.log(I) - lf) ** 2))
    for k, (lab, v) in zip(('d2', 'd10'), est.items()):
        out[k] = {'peak': float(nu[np.argmax(v)]), 'mse': float(np.mean((np.log(v) - lf) ** 2)),
                  'fpeak': float(v.max())}
    fwt = arma_spectrum(nw, AR2)
    out['welch'] = {'peak': float(nw[np.argmax(fw)]), 'mse': float(np.mean((np.log(fw) - np.log(fwt)) ** 2)),
                    'K': int((T - 128) // 64 + 1)}
    out['ftrue_peak'] = float(f.max())
    return out


def fig_sunspot_ci(save_it=True):
    """Daniell-smoothed periodogram of the sunspots (m = 2, df = 10) with a 95% chi-square band, log scale."""
    s = sunspots()
    nu, I = periodogram(s.values)
    m = 2
    fh = daniell(I, m)
    df = 2 * (2 * m + 1)
    lo, hi = chi2_band(fh, df)
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.fill_between(nu, lo, hi, color=st.Teal, alpha=0.2, lw=0, label='95% confidence band (chi-square, df = 10)')
    ax.semilogy(nu, I, color=st.Amber, lw=0.7, label='raw periodogram')
    ax.semilogy(nu, fh, color=st.MainBlue, lw=1.8, label='Daniell smoother, m = 2')
    j = int(np.argmax(fh))
    ax.annotate(f'period {1 / nu[j]:.1f} years', (nu[j], fh[j]), textcoords='offset points', xytext=(8, 6),
                color=st.IDAred)
    ax.set_xlabel('frequency nu (cycles per year)')
    ax.set_ylabel('spectrum (log scale)')
    st.legend_outside_bottom(ax, ncol=3)
    fig.tight_layout()
    save('tsa_ch12_sunspot_ci', save_it)
    j2 = np.argmin(abs(nu - 0.18))
    return {'m': m, 'L': 2 * m + 1, 'df': df, 'B': float((2 * m + 1) / len(s)), 'nu': float(nu[j]),
            'period': float(1 / nu[j]), 'f': float(fh[j]), 'lo': float(lo[j]), 'hi': float(hi[j]),
            'ratio_hi': float(hi[j] / fh[j]), 'ratio_lo': float(lo[j] / fh[j]), 'f018': float(fh[j2]),
            'hi018': float(hi[j2])}


# =============================================================================
# 5. CYCLES IN DATA: GDP, ELECTRICITY LOAD
# =============================================================================
def fig_gdp_series(save_it=True):
    """US and Romanian real GDP: quarterly growth and HP cycle (lambda = 1600), up to 2019 Q4."""
    G = gdp_growth()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    out = {}
    for ax, k, c in zip(axes, ('US', 'RO'), (st.MainBlue, st.IDAred)):
        cyc = hp_cycle(G[k + '_level'])
        ax.plot(G[k].index, G[k].values, color=c, lw=0.9, label='quarterly growth, %' if k == 'US' else '_')
        ax.plot(cyc.index, cyc.values, color=st.Forest, lw=1.5, label='HP cycle, % of trend' if k == 'US' else '_')
        ax.axhline(0, color=st.DarkText, lw=0.5)
        ax.set_title('United States' if k == 'US' else 'Romania')
        out[k] = {'T': int(len(G[k])), 'start': str(G[k].index[0].date()), 'end': str(G[k].index[-1].date()),
                  'mean': float(G[k].mean()), 'sd': float(G[k].std()), 'cyc_sd': float(cyc.std())}
    st.fig_legend_bottom(fig, ncol=2, y=0.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save('tsa_ch12_gdp_series', save_it)
    return out


def fig_gdp_spectra(save_it=True):
    """Smoothed spectra (Daniell, m = 2) of GDP growth and of the HP cycle, as a share of the variance, with the
    business-cycle band of 6-32 quarters."""
    G = gdp_growth()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    out = {}
    lo_nu, hi_nu = 1 / BC_BAND[1], 1 / BC_BAND[0]
    for ax, kind in zip(axes, ('growth', 'HP cycle')):
        ax.axvspan(lo_nu, hi_nu, color=st.Amber, alpha=0.15, lw=0, label='business-cycle band, 6-32 quarters' if kind == 'growth' else '_')
        for k, c in (('US', st.MainBlue), ('RO', st.IDAred)):
            x = G[k] if kind == 'growth' else hp_cycle(G[k + '_level'])
            nu, I = periodogram(x.values)
            fh = daniell(I, 2)
            ax.plot(nu, fh / np.mean((x - x.mean()) ** 2), color=c, lw=1.6,
                    label=('United States' if k == 'US' else 'Romania') if kind == 'growth' else '_')
            band = (nu >= lo_nu) & (nu <= hi_nu)
            j = int(np.argmax(fh))
            out[f'{k}.{kind}'] = {'share_bc': float(I[band].sum() / I.sum()), 'share_low': float(I[nu < lo_nu].sum() / I.sum()),
                                  'peak_period': float(1 / nu[j]), 'T': int(len(x))}
        ax.set_title(f'Spectrum of the {kind} / variance')
        ax.set_xlabel('frequency nu (cycles per quarter)')
        ax.set_xlim(0, 0.5)
    st.fig_legend_bottom(fig, ncol=3, y=0.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save('tsa_ch12_gdp_spectra', save_it)
    return out


def fig_load(save_it=True):
    """Hourly electricity load of Romania: two weeks of data and the Welch spectrum against the period in hours."""
    s = load_hourly()
    x = s.values
    nw, fw = welch(x, 24 * 7 * 8)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={'width_ratios': [1, 1.2]})
    w = s.loc['2025-03-03':'2025-03-16 23:00']
    axes[0].plot(w.index, w.values / 1000, color=st.MainBlue, lw=1.1, label='load, GW (UTC hours)')
    axes[0].set_title('Two weeks of March 2025')
    axes[0].tick_params(axis='x', labelrotation=30)
    ax = axes[1]
    ax.loglog(1 / nw, fw, color=st.IDAred, lw=1.2, label='Welch spectrum (8-week segments)')
    for p, lab in ((168, '168 h'), (24, '24 h'), (12, '12 h'), (8, '8 h')):
        ax.axvline(p, color=st.Forest, lw=0.8, ls=':')
        ax.text(p, ax.get_ylim()[1] if False else fw.max() * 2, lab, rotation=90, va='top', ha='right', color=st.Forest, fontsize=10)
    ax.set_xlabel('period (hours, log scale)')
    ax.set_ylabel('spectrum (log scale)')
    ax.set_title('Spectrum of the hourly load')
    ax.invert_xaxis()
    st.fig_legend_bottom(fig, ncol=2, y=0.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save('tsa_ch12_load', save_it)
    nu, I = periodogram(x)
    T = len(x)
    tot = I.sum()

    def share(periods, width=2):
        idx = set()
        for p in periods:
            j = int(round(T / p))
            idx.update(range(max(j - width, 1) - 1, min(j + width, len(I) - 1)))
        return float(I[sorted(idx)].sum() / tot)
    daily = share([24 / k for k in range(1, 7)])
    weekly = share([168 / k for k in range(1, 7)] + [168 * 2])
    order = np.argsort(I)[::-1]
    top = []
    for k in order:
        p = 1 / nu[k]
        if all(abs(p - q) / q > 0.05 for q in top):
            top.append(float(p))
        if len(top) == 6:
            break
    return {'T': int(T), 'start': str(s.index[0].date()), 'end': str(s.index[-1].date()), 'mean': float(s.mean()),
            'share_daily': daily, 'share_weekly': weekly, 'share_long': float(I[1 / nu > 24 * 30].sum() / tot),
            'top': top}


# =============================================================================
# 6. LONG MEMORY: THE POLE AT ZERO
# =============================================================================
def fig_long_memory(save_it=True):
    """Log-log periodogram of daily S&P 500 returns and absolute returns, with the GPH line on the lowest frequencies."""
    r = log_returns('sp500').values
    S = [('returns r_t', r, st.MainBlue), ('absolute returns |r_t|', np.abs(r), st.IDAred)]
    fig, ax = plt.subplots(figsize=(10, 4.0))
    out = {}
    for lab, x, c in S:
        nu, I = periodogram(x)
        edges = np.geomspace(nu[0], 0.5, 41)                 # 40 bins of equal width in log frequency
        k = np.digitize(nu, edges)
        nb = np.array([np.exp(np.mean(np.log(nu[k == b]))) for b in np.unique(k)])
        Ib = np.array([I[k == b].mean() for b in np.unique(k)])
        ax.loglog(nb, Ib, 'o-', color=c, lw=1.0, ms=3.5, label=f'{lab}: periodogram, averaged in log-frequency bins')
        g = gph_spec(x)
        m = g['m']
        xx = nu[:m]
        ax.loglog(xx, np.exp(g['c'] - g['d'] * np.log(4 * np.sin(np.pi * xx) ** 2)), color=c, lw=2.4, ls='--',
                  label=f"{lab}: GPH line, d = {g['d']:.2f}")
        out['abs' if 'abs' in lab else 'r'] = g
    ax.set_xlabel('frequency nu (cycles per day, log scale)')
    ax.set_ylabel('power (log scale)')
    st.legend_outside_bottom(ax, ncol=2)
    fig.tight_layout()
    save('tsa_ch12_long_memory', save_it)
    out['T'] = int(len(r))
    return out


# =============================================================================
# 7. FILTERS
# =============================================================================
def hp_gain(nu, lam=HP_LAMBDA):
    """Squared gain of the HP cycle filter: 4 lam (1 - cos w)^2 / (1 + 4 lam (1 - cos w)^2), w = 2 pi nu."""
    a = 4 * lam * (1 - np.cos(2 * np.pi * nu)) ** 2
    return a / (1 + a)


def fig_filters(save_it=True):
    """Squared gains of the first difference, the seasonal difference (s = 4) and the HP filter (lambda = 1600)."""
    nu = np.linspace(0, 0.5, 1001)
    g1 = 4 * np.sin(np.pi * nu) ** 2
    g4 = 4 * np.sin(4 * np.pi * nu) ** 2
    gh = hp_gain(nu)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    axes[0].plot(nu, g1, color=st.MainBlue, lw=1.8, label='first difference 1 - L')
    axes[0].plot(nu, g4, color=st.Orange, lw=1.6, ls='--', label='seasonal difference 1 - L^4')
    axes[0].set_title('Differencing filters')
    axes[1].plot(nu, gh, color=st.Forest, lw=1.8, label='HP cycle filter, lambda = 1600')
    axes[1].plot(nu, 1 - gh, color=st.Purple, lw=1.6, ls='--', label='HP trend filter')
    for ax in axes:
        ax.axvspan(1 / BC_BAND[1], 1 / BC_BAND[0], color=st.Amber, alpha=0.15, lw=0,
                   label='business-cycle band' if ax is axes[1] else '_')
        ax.set_xlabel('frequency nu (cycles per quarter)')
        ax.set_ylabel('squared gain |H(nu)|^2')
    st.fig_legend_bottom(fig, ncol=5, y=0.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save('tsa_ch12_filters', save_it)
    half = float(nu[np.argmin(abs(gh - 0.5))])
    return {'d1_40': float(4 * np.sin(np.pi / 40) ** 2), 'd1_4': float(4 * np.sin(np.pi / 4) ** 2),
            'hp_8': float(hp_gain(1 / 8)), 'hp_32': float(hp_gain(1 / 32)), 'hp_40': float(hp_gain(1 / 40)),
            'hp_half_nu': half, 'hp_half_period': 1 / half}


# =============================================================================
# 8. TWO SERIES: COHERENCE AND PHASE
# =============================================================================
def fig_coherence(save_it=True):
    """Squared coherence and phase of US industrial production growth and the fall in unemployment (monthly,
    1948-2019), Welch segments of 96 months without overlap."""
    F = read_fred(['INDPRO', 'UNRATE']).loc['1948-01-01':'2019-12-31'].dropna()
    ip = 100 * np.log(F['INDPRO']).diff()
    du = -F['UNRATE'].diff()
    d = pd.concat([ip, du], axis=1).dropna()
    x, y = d.iloc[:, 0].values, d.iloc[:, 1].values
    n = 96
    nu, C = signal.coherence(x, y, fs=1.0, window='hann', nperseg=n, noverlap=0)
    _, Pxy = signal.csd(x, y, fs=1.0, window='hann', nperseg=n, noverlap=0)
    ph = np.angle(Pxy)
    K = len(x) // n
    thr = 1 - 0.05 ** (1 / (K - 1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    band = (nu >= 1 / 96) & (nu <= 1 / 18)
    for ax in axes:
        ax.axvspan(1 / 96, 1 / 18, color=st.Amber, alpha=0.15, lw=0, label='business-cycle band, 18-96 months' if ax is axes[0] else '_')
    axes[0].plot(nu, C, color=st.MainBlue, lw=1.6, marker='o', ms=3, label='squared coherence')
    axes[0].axhline(thr, color=st.IDAred, lw=1.2, ls='--', label=f'5% significance threshold ({thr:.2f})')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Squared coherence')
    axes[1].plot(nu, ph, color=st.Purple, lw=1.4, marker='o', ms=3, label='phase (radians)')
    axes[1].axhline(0, color=st.DarkText, lw=0.5)
    axes[1].set_title('Phase')
    for ax in axes:
        ax.set_xlabel('frequency nu (cycles per month)')
    st.fig_legend_bottom(fig, ncol=4, y=0.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save('tsa_ch12_coherence', save_it)
    j = np.argmin(abs(nu - 1 / 48))
    lag48 = float(-ph[j] / (2 * np.pi * nu[j]))      # > 0: the fall in unemployment lags production
    hi = (nu > 1 / 6)
    return {'T': int(len(x)), 'start': str(d.index[0].date()), 'end': str(d.index[-1].date()), 'K': int(K),
            'thr': float(thr), 'coh_bc': float(C[band].mean()), 'coh_bc_max': float(C[band].max()),
            'coh_hi': float(C[hi].mean()), 'phase48': float(ph[j]), 'nu48': float(nu[j]), 'lag48': lag48,
            'corr0': float(np.corrcoef(x, y)[0, 1])}


# =============================================================================
# 9. WAVELETS (POINTER)
# =============================================================================
def fig_wavelet(save_it=True):
    """Morlet wavelet power of the yearly sunspot numbers: how the strength and length of the cycle change in time."""
    s = sunspots()
    periods = np.geomspace(2, 64, 60)
    P = morlet_cwt(s.values, periods)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    cs = ax.contourf(s.index, periods, P, levels=12, cmap='viridis')
    ax.set_yscale('log', base=2)
    ax.set_yticks([2, 4, 8, 11, 16, 32, 64])
    ax.set_yticklabels(['2', '4', '8', '11', '16', '32', '64'])
    ax.invert_yaxis()
    ax.set_ylabel('period (years)')
    ax.set_xlabel('year')
    fig.colorbar(cs, ax=ax, label='wavelet power / variance', pad=0.01)
    fig.tight_layout()
    save('tsa_ch12_wavelet', save_it)
    k = np.argmin(abs(periods - 11))
    band = (periods >= 8) & (periods <= 14)
    pw = P[band].mean(axis=0)
    yrs = s.index.values
    weak = (yrs >= 1795) & (yrs <= 1830)
    return {'p11_mean': float(P[k].mean()), 'weak_1795_1830': float(pw[weak].mean()), 'all': float(pw.mean()),
            'max_year': int(yrs[np.argmax(pw)])}


if __name__ == '__main__':
    st.apply()
    N = {}
    only = sys.argv[1:]
    path = os.path.join(HERE, 'ch12_numbers.json')
    if only and os.path.exists(path):
        N = json.load(open(path))
    for name, f in [('ex', worked_examples), ('fourier', fig_fourier), ('alias', fig_aliasing), ('spectra', fig_spectra),
                    ('sun', fig_sunspots), ('incons', fig_inconsistency), ('leak', fig_leakage),
                    ('smooth', fig_smoothing), ('sunci', fig_sunspot_ci), ('gdp', fig_gdp_series),
                    ('gdpspec', fig_gdp_spectra), ('load', fig_load), ('lm', fig_long_memory), ('filt', fig_filters),
                    ('coh', fig_coherence), ('wav', fig_wavelet)]:
        if only and name not in only:
            continue
        print(name)
        N[name] = f()
        with open(path, 'w') as fh:
            json.dump(N, fh, indent=1, default=float)
    print('written ch12_numbers.json')
