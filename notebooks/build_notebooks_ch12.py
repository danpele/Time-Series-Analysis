"""
build_notebooks_ch12.py -- lecture notebook of Chapter 12 (TSA): spectral analysis (self-study chapter, no seminar)
================================================================================================================
Output: notebooks/EN/chapter12_lecture_notebook.ipynb
The code is taken from Quantlets/Ch_12/generate_all_charts.py (inspect.getsource), so the notebook stays in sync with
the Quantlets and the slides.
Run:  python3 notebooks/build_notebooks_ch12.py
      jupyter nbconvert --to notebook --execute --inplace notebooks/EN/chapter12_lecture_notebook.ipynb
Time Series Analysis - Daniel Traian PELE
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_notebook import build, chapter_paths, code, common_cells, md, src   # noqa: E402

chapter_paths(12)
import generate_all_charts as g   # noqa: E402
import build_quantlets as bq      # noqa: E402

CONSTS = '\n'.join(bq.CONSTS)
CORE = bq.CORE

LECTURE = [
    md("# Time Series Analysis — Chapter 12: Spectral analysis\n\n"
       "*Lecture notebook (self-study chapter). Daniel Traian Pele, Bucharest University of Economic Studies.*\n\n"
       "- Cycles, Fourier frequencies, the discrete Fourier transform and aliasing.\n"
       "- Spectral densities of white noise and ARMA models; the periodogram and its properties; Fisher's test.\n"
       "- Leakage and tapering; Daniell and Welch smoothing; chi-square confidence bands.\n"
       "- Cycles in data: sunspots, US and Romanian GDP, the hourly electricity load of Romania.\n"
       "- The spectral pole of long memory; filter gains; coherence and phase; a wavelet scalogram.\n"
       "- Convention (Shumway and Stoffer 2017, ch. 4): frequency $\\nu$ in cycles per observation, period $1/\\nu$, "
       "$f(\\nu) = \\sum_h \\gamma(h) e^{-2\\pi i \\nu h}$, periodogram $I(\\nu_j) = |T^{-1/2}\\sum_t x_t e^{-2\\pi i \\nu_j t}|^2$.\n"
       "- References: Schuster (1898); Yule (1927); Fisher (1929); Welch (1967); Granger (1966); Torrence and Compo (1998)."),
    *common_cells(),
    md("## Definitions and helpers used in the whole notebook\n\n"
       "- `periodogram(x, taper)`: raw periodogram at the Fourier frequencies (optional Hann taper).\n"
       "- `daniell(I, m)`: average of $2m+1$ neighbouring ordinates; `chi2_band(f, df)`: 95% band; `welch(x, n)`: Welch estimate.\n"
       "- `arma_spectrum(nu, phi, theta)`: theoretical ARMA spectrum; `fisher_g(I)`: Fisher's test; `gph_spec(x)`: GPH estimate of $d$.\n"
       "- `morlet_cwt(x, periods)`: Morlet wavelet power; data loaders for sunspots, GDP and the electricity load."),
    code(CONSTS + '\n\n\n' + src(*CORE)),
    md("## 1. Cycles, the DFT and aliasing\n\n- Worked examples of the slides; two cycles plus noise; a 4-month cycle seen quarterly."),
    code(src(g.worked_examples, g.fig_fourier, g.fig_aliasing)
         + "\n\n\nex = worked_examples()\nprint('DFT of (4, 2, 0, 2):', ex['dft'])\nprint('AR(2) peak:', ex['ar2'])\n"
           "print(fig_fourier(save_it=False))\nfig_aliasing(save_it=False)\n"
           "# check: the FFT gives the same periodogram as the hand computation\n"
           "print(np.abs(np.fft.fft([2, 0, -2, 0])) ** 2 / 4)"),
    md("## 2. Spectral densities of ARMA models"),
    code("SPECTRA = " + repr(g.SPECTRA) + "\n\n\n" + src(g.fig_spectra)
         + "\n\n\nsp = fig_spectra(save_it=False)\nprint(pd.DataFrame(sp).T.round(3))"),
    md("## 3. The periodogram: sunspots and white noise"),
    code(src(g.fig_sunspots, g.fig_inconsistency)
         + "\n\n\nsun = fig_sunspots(save_it=False)\nprint('peak period:', round(sun['period'], 2), 'years; Fisher g:', sun['fisher'])\n"
           "print(fig_inconsistency(save_it=False))"),
    md("## 4. Leakage, smoothing and confidence bands\n\n- Try other values of `m` in `daniell` and other segment lengths in `welch`."),
    code(src(g.fig_leakage, g.fig_smoothing, g.fig_sunspot_ci)
         + "\n\n\nprint(fig_leakage(save_it=False))\nprint(fig_smoothing(save_it=False))\nprint(fig_sunspot_ci(save_it=False))"),
    md("## 5. Business cycles: US and Romanian GDP\n\n- Samples end in 2019 Q4: the 2020 collapse would dominate the periodogram."),
    code(src(g.fig_gdp_series, g.fig_gdp_spectra)
         + "\n\n\nprint(fig_gdp_series(save_it=False))\nprint(pd.DataFrame(fig_gdp_spectra(save_it=False)).T.round(3))"),
    md("## 6. The daily and weekly rhythm of electricity load"),
    code(src(g.fig_load) + "\n\n\nprint(fig_load(save_it=False))"),
    md("## 7. Long memory: the pole at zero (Chapter 8)"),
    code(src(g.fig_long_memory) + "\n\n\nprint(fig_long_memory(save_it=False))"),
    md("## 8. Linear filters: differencing and the HP filter"),
    code(src(g.hp_gain, g.fig_filters) + "\n\n\nprint(fig_filters(save_it=False))"),
    md("## 9. Two series: coherence and phase"),
    code(src(g.fig_coherence) + "\n\n\nprint(fig_coherence(save_it=False))"),
    md("## 10. A pointer: wavelets"),
    code(src(g.fig_wavelet) + "\n\n\nprint(fig_wavelet(save_it=False))"),
    md("## Exercises for self-study\n\n"
       "1. Compute by hand the periodogram of $x = (1, -1, 1, -1)$; check it with `np.fft.fft`.\n"
       "2. Plot the spectrum of an AR(2) with $\\phi = (1.0, -0.5)$; find the peak period with the formula of the slides.\n"
       "3. Apply the Daniell smoother with $m = 1, 2, 5, 10$ to the sunspot periodogram; which $m$ would you report?\n"
       "4. Repeat Section 5 for another EU country (Eurostat key `Q.CLV10_MEUR.SCA.B1GQ.<country code>`).\n"
       "5. Compute the coherence between Romanian and euro-area GDP growth (`EA20`); interpret the business-cycle band."),
]

if __name__ == '__main__':
    build(LECTURE, 12, 'lecture')
