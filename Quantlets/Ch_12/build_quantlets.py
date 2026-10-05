"""
build_quantlets.py -- Quantlet folders of Chapter 12 (TSA): spectral analysis
============================================================================
Metainfo.txt + self-contained Colab notebook + charts for each Quantlet (Quantlets/common/tsa_quantlets.py).
Run:  python3 Quantlets/Ch_12/generate_all_charts.py
      python3 Quantlets/Ch_12/build_quantlets.py
      python3 notebooks/add_colab_banner.py
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
sys.path.insert(0, HERE)
import generate_all_charts as g                 # noqa: E402
from tsa_quantlets import build_all             # noqa: E402

SUBMITTED = 'Monday, 5 October 2026'
DATA = ('Yearly sunspot numbers 1700-2008 (statsmodels); US real GDP, industrial production and unemployment (FRED '
        'GDPC1, INDPRO, UNRATE); Romanian real GDP (Eurostat namq_10_gdp); hourly electricity load of Romania (ENTSO-E '
        'extract, Quantlets/Ch_04); daily S&P 500 from EODHD (data/market of the TSA repository)')
NODATA = 'Simulated data only (no market data)'
CONSTS = ['from scipy import signal', 'from statsmodels.tsa.filters.hp_filter import hpfilter',
          f'SEED = {g.SEED!r}', f'RO_GDP = {g.RO_GDP!r}', f'GDP_END = {g.GDP_END!r}', f'BC_BAND = {g.BC_BAND!r}',
          f'HP_LAMBDA = {g.HP_LAMBDA!r}', f'LOAD_FILE = {g.LOAD_FILE!r}', f'LOAD_RAW = {g.LOAD_RAW!r}',
          f'BW = {g.BW!r}', f'AR2 = {g.AR2!r}', f'DFT_EXAMPLE = {g.DFT_EXAMPLE!r}', 'HERE = "."']
CORE = [g.save, g.periodogram, g.daniell, g.chi2_band, g.welch, g.arma_spectrum, g.simulate_arma, g.fisher_g,
        g.gph_spec, g.morlet_cwt, g.sunspots, g.gdp_growth, g.hp_cycle, g.load_hourly]

QUANTLETS = [
    dict(name='TSA_ch12_fourier',
         desc='Cycles and the discrete Fourier transform: a series built from two cosines plus noise and its periodogram; '
              'the DFT of the series (4, 2, 0, 2) by hand and Parseval; aliasing of a 4-month cycle observed quarterly; '
              'the numbers of the worked examples (AR(1), MA(1) and AR(2) spectra, a Daniell confidence interval).',
         keywords='Fourier frequencies, discrete Fourier transform, periodogram, Parseval, aliasing, Nyquist frequency',
         consts=CONSTS, funcs=CORE + [g.worked_examples, g.fig_fourier, g.fig_aliasing],
         run='print(worked_examples())\nprint(fig_fourier())\nprint(fig_aliasing())',
         charts=['tsa_ch12_fourier', 'tsa_ch12_aliasing'], data=NODATA),
    dict(name='TSA_ch12_spectral_densities',
         desc='Theoretical spectral densities of white noise, AR(1) with phi = 0.6 and -0.6, MA(1), an AR(2) with complex '
              'roots (a pseudo-cycle of about 12 periods) and ARMA(1,1), with their variances (the area under f).',
         keywords='spectral density, white noise, AR(1), MA(1), AR(2), ARMA, pseudo-cycle, Wiener-Khinchin',
         consts=CONSTS, funcs=CORE + [g.fig_spectra], run='SPECTRA = ' + repr(g.SPECTRA) + '\nprint(fig_spectra())',
         charts=['tsa_ch12_spectra'], data=NODATA),
    dict(name='TSA_ch12_periodogram',
         desc='The raw periodogram of the yearly sunspot numbers 1700-2008 (an 11-year cycle) with Fisher\'s g test of a '
              'hidden periodicity; the periodogram of Gaussian white noise for T = 128 and T = 2048 (unbiased but not '
              'consistent).',
         keywords='periodogram, sunspots, Fisher g test, hidden periodicity, inconsistency, chi-square distribution',
         consts=CONSTS, funcs=CORE + [g.fig_sunspots, g.fig_inconsistency],
         run='print(fig_sunspots())\nprint(fig_inconsistency())',
         charts=['tsa_ch12_sunspots', 'tsa_ch12_inconsistency']),
    dict(name='TSA_ch12_spectral_estimation',
         desc='Estimating the spectrum: leakage and the Hann taper; Daniell smoothing (m = 2 and m = 10) and Welch '
              'segment averaging of a simulated AR(2) against its true spectrum; the Daniell-smoothed sunspot spectrum '
              'with a 95% chi-square confidence band.',
         keywords='spectral leakage, tapering, Hann window, Daniell smoother, Welch method, bandwidth, confidence band',
         consts=CONSTS, funcs=CORE + [g.fig_leakage, g.fig_smoothing, g.fig_sunspot_ci],
         run='print(fig_leakage())\nprint(fig_smoothing())\nprint(fig_sunspot_ci())',
         charts=['tsa_ch12_leakage', 'tsa_ch12_smoothing', 'tsa_ch12_sunspot_ci']),
    dict(name='TSA_ch12_business_cycle',
         desc='Business cycles in the frequency domain: quarterly growth and Hodrick-Prescott cycles of US real GDP '
              '(1947-2019) and Romanian real GDP (1995-2019); smoothed spectra and the share of the variance in the '
              'business-cycle band of 6-32 quarters.',
         keywords='business cycle, GDP, Romania, United States, Hodrick-Prescott filter, spectrum, band of frequencies',
         consts=CONSTS, funcs=CORE + [g.fig_gdp_series, g.fig_gdp_spectra],
         run='print(fig_gdp_series())\nprint(fig_gdp_spectra())',
         charts=['tsa_ch12_gdp_series', 'tsa_ch12_gdp_spectra']),
    dict(name='TSA_ch12_electricity_load',
         desc='The hourly electricity load of Romania 2022-2026 (ENTSO-E): two weeks of data and the Welch spectrum '
              'against the period in hours; the daily (24, 12, 8 hours) and weekly (168, 84 hours) cycles and their '
              'shares of the variance.',
         keywords='electricity load, Romania, ENTSO-E, daily cycle, weekly cycle, harmonics, Welch spectrum',
         consts=CONSTS, funcs=CORE + [g.fig_load], run='print(fig_load())', charts=['tsa_ch12_load']),
    dict(name='TSA_ch12_long_memory',
         desc='The spectral pole at zero: log-log periodogram of daily S&P 500 returns and absolute returns (2000-2026), '
              'with the GPH (Geweke and Porter-Hudak 1983) regression on the lowest frequencies.',
         keywords='long memory, spectral pole, GPH, log-periodogram regression, volatility, S&P 500',
         consts=CONSTS, funcs=CORE + [g.fig_long_memory], run='print(fig_long_memory())',
         charts=['tsa_ch12_long_memory']),
    dict(name='TSA_ch12_filters',
         desc='Squared gains of linear filters: the first difference, the seasonal difference (s = 4) and the '
              'Hodrick-Prescott trend and cycle filters (lambda = 1600), with the business-cycle band.',
         keywords='linear filter, transfer function, squared gain, differencing, seasonal difference, Hodrick-Prescott',
         consts=CONSTS, funcs=CORE + [g.hp_gain, g.fig_filters], run='print(fig_filters())',
         charts=['tsa_ch12_filters'], data=NODATA),
    dict(name='TSA_ch12_coherence',
         desc='Cross-spectral analysis of US monthly industrial production growth and the fall in the unemployment rate '
              '(1948-2019): squared coherence with a 5% significance threshold, and the phase (Welch estimates).',
         keywords='cross-spectrum, coherence, phase, industrial production, unemployment, Okun law',
         consts=CONSTS, funcs=CORE + [g.fig_coherence], run='print(fig_coherence())',
         charts=['tsa_ch12_coherence']),
    dict(name='TSA_ch12_wavelet',
         desc='A pointer to wavelets: the Morlet wavelet power (scalogram) of the yearly sunspot numbers 1700-2008, '
              'computed by FFT as in Torrence and Compo (1998).',
         keywords='wavelet, Morlet, scalogram, time-frequency analysis, sunspots',
         consts=CONSTS, funcs=CORE + [g.fig_wavelet], run='print(fig_wavelet())', charts=['tsa_ch12_wavelet']),
]

if __name__ == '__main__':
    build_all(QUANTLETS, chapter=12, chapter_title='Spectral analysis', ql_dir=HERE, data=DATA, submitted=SUBMITTED)
