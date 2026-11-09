"""Build the interactive pages of Chapter 12 (spectral analysis): interactive/ch12.html (EN) and interactive/ch12_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its interactive version on
the right. The worked examples of the lecture (frequency and period, the DFT of four numbers, the folding of an aliased
frequency, the AR(1) and MA(1) spectra, the AR(2) peak, the Daniell estimator, a chi-square band, the HP gain), its result
tables (dominant periods of the sunspots, spectral peaks with their bandwidths, the GDP spectra, the shares of the load
spectrum, filter gains at selected periods) and an extra chart where the lecture gives only numbers (the distribution of
periodogram ordinates) get blocks of their own (EXTRA), placed by slide; on the web lecture each of them travels inside a
chart block and ch12.js moves it to its own frame. Logic in interactive/ch12.js, inputs in interactive/ch12_data.js
(Quantlets/Ch_12/export_interactive.py), slide images in interactive/slides/.
"""
import json
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 12

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_12'
TEX = {'ro': 'RO/Cursuri/capitol12_analiza_spectrala', 'en': 'EN/Courses/chapter12_spectral_analysis'}

# (anchor, chart file, section key); order of the slides
BLOCKS = [('fourier', 'tsa_ch12_fourier', 'cf'), ('aliasing', 'tsa_ch12_aliasing', 'cf'),
          ('spectra', 'tsa_ch12_spectra', 'sd'),
          ('sunspots', 'tsa_ch12_sunspots', 'pg'), ('noise', 'tsa_ch12_inconsistency', 'pg'),
          ('leakage', 'tsa_ch12_leakage', 'es'), ('smoothing', 'tsa_ch12_smoothing', 'es'), ('sun-ci', 'tsa_ch12_sunspot_ci', 'es'),
          ('gdp-series', 'tsa_ch12_gdp_series', 'cy'), ('gdp-spectra', 'tsa_ch12_gdp_spectra', 'cy'), ('load', 'tsa_ch12_load', 'cy'),
          ('long-memory', 'tsa_ch12_long_memory', 'lm'),
          ('filters', 'tsa_ch12_filters', 'fi'),
          ('coherence', 'tsa_ch12_coherence', 'co'),
          ('wavelet', 'tsa_ch12_wavelet', 'wv')]

# worked examples, result tables and extra charts: (anchor, frame title EN, frame title RO, section key, host chart block)
EXTRA = [('wx-cycle', 'A cycle: amplitude, frequency, phase', 'Un ciclu: amplitudine, frecvență, fază', 'cf', 'fourier'),
         ('wx-dft', 'Worked example: the DFT of four numbers', 'Exemplu rezolvat: DFT pentru patru numere', 'cf', 'fourier'),
         ('wx-alias', 'Interpreting aliasing', 'Interpretarea fenomenului de aliasing', 'cf', 'aliasing'),
         ('wx-arma', 'Worked example: AR(1) and MA(1) spectra', 'Exemplu rezolvat: spectrele AR(1) și MA(1)', 'sd', 'spectra'),
         ('wx-ar2', 'Worked example: the peak of an AR(2) spectrum', 'Exemplu rezolvat: vîrful spectrului unui AR(2)', 'sd', 'spectra'),
         ('t-sun', 'Interpreting the sunspot periodogram', 'Interpretarea periodogramei petelor solare', 'pg', 'sunspots'),
         ('x-prop', 'Statistical properties of the periodogram', 'Proprietățile statistice ale periodogramei', 'pg', 'noise'),
         ('wx-daniell', 'Smoothing the periodogram (1/2): the Daniell estimator', 'Netezirea periodogramei (1/2): estimatorul Daniell', 'es', 'smoothing'),
         ('t-smooth', 'Interpreting the smoothed estimates', 'Interpretarea estimărilor netezite', 'es', 'smoothing'),
         ('wx-band', 'Worked example: a 95% band with L = 5', 'Exemplu rezolvat: o bandă de 95% pentru L = 5', 'es', 'sun-ci'),
         ('t-gdp', 'Interpreting the GDP spectra', 'Interpretarea spectrelor PIB', 'cy', 'gdp-spectra'),
         ('t-load', 'Interpreting the load spectrum', 'Interpretarea spectrului consumului', 'cy', 'load'),
         ('wx-hp', 'Linear filters and their gain (2/2)', 'Filtrele liniare și cîștigul lor (2/2)', 'fi', 'filters'),
         ('t-filt', 'Interpreting the filter gains', 'Interpretarea cîștigurilor filtrelor', 'fi', 'filters')]

PFX = {'fourier': 'fo', 'aliasing': 'al', 'spectra': 'sp', 'sunspots': 'ss', 'noise': 'wn', 'leakage': 'lk', 'smoothing': 'sm', 'sun-ci': 'sc',
       'gdp-series': 'gs', 'gdp-spectra': 'gp', 'load': 'ld', 'long-memory': 'lm', 'filters': 'fl', 'coherence': 'co', 'wavelet': 'wv',
       'wx-cycle': 'wy', 'wx-dft': 'wd', 'wx-alias': 'wa', 'wx-arma': 'wr', 'wx-ar2': 'w2', 't-sun': 'ts', 'x-prop': 'xp', 'wx-daniell': 'wl',
       't-smooth': 'tm', 'wx-band': 'wb', 't-gdp': 'tg', 't-load': 'tl', 'wx-hp': 'wh', 't-filt': 'tf'}

SECTIONS = {
 'ro': dict(cf='Cicluri și frecvențe', sd='Densitatea spectrală', pg='Periodograma', es='Estimarea spectrului', cy='Cicluri în datele economice și energetice',
            lm='Memoria lungă: polul de la zero', fi='Filtre liniare', co='Două serii: coerență și fază', wv='Extensie: wavelets'),
 'en': dict(cf='Cycles and frequencies', sd='The spectral density', pg='The periodogram', es='Estimating the spectrum', cy='Cycles in economic and energy data',
            lm='Long memory: the pole at zero', fi='Linear filters', co='Two series: coherence and phase', wv='Extension: wavelets'),
}

L = {
 'ro': dict(file='ch12_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 12 · Analiza spectrală', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'casetele „Exemplu rezolvat” refac calculul cu cifrele capitolului.',
   src='Simulările capitolului (sămînța 2026): extragerile Normale din exemplul cu două cicluri (T = 120), din periodogramele zgomotului alb (T = 128 și T = 2048), '
       'din exemplul de scurgere spectrală (T = 256) și seria AR(2) simulată din exemplul de netezire (T = 512). Petele solare anuale 1700–2008 (statsmodels), consumul '
       'orar de energie electrică al României 1 ianuarie 2022 – 30 iunie 2026 (extrasul ENTSO-E din Capitolul 4, pus pe o grilă orară regulată de funcția capitolului) '
       'și randamentele zilnice S&P 500 2000–2026 (EODHD) nu sînt revizuite și reproduc exact cifrele din curs. PIB-ul real al SUA (FRED GDPC1) și producția '
       'industrială și rata șomajului din SUA (FRED INDPRO, UNRATE), citite pe 4 noiembrie 2026 și tăiate la 2019 ca în curs, reproduc exact cifrele din curs. '
       'PIB-ul real al României (Eurostat namq_10_gdp) a fost revizuit după realizarea slide-urilor: pagina folosește varianta din slide-uri, citită din graficul '
       'vectorial charts/tsa_ch12_gdp_series.pdf, în care fiecare dintre cele 99 de rate trimestriale de creștere (1995 T2 – 2019 T4) este un vîrf al liniei; ele '
       'reproduc cifrele din curs pînă la a șasea cifră semnificativă; valorile cu datele Eurostat din 4 noiembrie 2026 sînt date alături. Variantele care nu apar '
       'în curs (alte amplitudini, alte eșantioane, alte lățimi de bandă, alte valori ale lui λ, alte segmente, alt ω₀) sînt rulări suplimentare ale codului '
       'capitolului pe aceleași date; graficele din „Interpretarea fenomenului de aliasing”, „Interpretarea periodogramei petelor solare”, „Proprietățile '
       'statistice ale periodogramei”, „Interpretarea estimărilor netezite”, „Interpretarea spectrelor PIB”, „Interpretarea spectrului consumului” și '
       '„Interpretarea cîștigurilor filtrelor” sînt grafice suplimentare pentru slide-uri care dau doar cifre.'),
 'en': dict(file='ch12.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 12 · Spectral Analysis', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; '
        'the “Worked example” boxes redo the calculation with the numbers of the chapter.',
   src='Simulations of the chapter (seed 2026): the Normal draws of the two-cycles example (T = 120), of the white-noise periodograms (T = 128 and T = 2048), of the '
       'leakage example (T = 256) and the simulated AR(2) of the smoothing example (T = 512). The yearly sunspots 1700–2008 (statsmodels), the hourly electricity load '
       'of Romania 1 January 2022 – 30 June 2026 (the ENTSO-E extract of Chapter 4, put on a regular hourly grid by the chapter’s loader) and the daily S&P 500 returns '
       '2000–2026 (EODHD) are not revised and reproduce the numbers of the lecture exactly. US real GDP (FRED GDPC1) and US industrial production and unemployment '
       '(FRED INDPRO, UNRATE), read on 4 November 2026 and cut at 2019 as in the lecture, reproduce the numbers of the lecture exactly. Romanian real GDP (Eurostat '
       'namq_10_gdp) has been revised since the slides were made: the page uses the vintage of the slides, read from the vector chart charts/tsa_ch12_gdp_series.pdf, in '
       'which every one of the 99 quarterly growth rates (1995 Q2 – 2019 Q4) is a vertex of the line; they reproduce the numbers of the lecture to the sixth significant '
       'digit; the values with the Eurostat data of 4 November 2026 are given next to them. Variants that do not appear in the lecture (other amplitudes, samples, '
       'bandwidths, values of λ, segments, ω₀) are extra runs of the chapter’s code on the same data; the charts of “Interpreting aliasing”, “Interpreting the sunspot '
       'periodogram”, “Statistical properties of the periodogram”, “Interpreting the smoothed estimates”, “Interpreting the GDP spectra”, “Interpreting the load '
       'spectrum” and “Interpreting the filter gains” are extra charts for slides that give only numbers.'),
}

CSS = ('<style>.c12x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c12x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c12x .c12f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c12x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c12x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c12x .wctl .ctl{max-width:360px;margin:.15rem 0 .2rem}'
       '.c12t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c12t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c12t th,.c12t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c12t thead tr:last-child th{border-bottom:1px solid #1A3A6E}'
       '.c12t tbody tr:nth-child(even){background:#F5F7FB}.c12t tbody tr.hi{background:#E3F4F7}.c12t .r{text-align:right}.c12cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}</style>')


def tbl(i):
    return f'<div class="tbl c12t" id="{i}"></div>'


def wx(i, title, formula, ctl=''):
    """Worked-example box; `ctl` holds the controls that only drive the box (they stay out of the chart's controls bar)."""
    return f'<div class="c12x"><h4>{title}</h4>' + (f'<div class="wctl">{ctl}</div>' if ctl else '') + f'<div class="c12f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def box(ctl, inner):
    """A box with controls that only drive the table inside it."""
    return f'<div class="c12x"><div class="wctl">{ctl}</div>{inner}</div>'


LAMS = [100, 400, 1600, 6400, 25600]


def widgets_all(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    lec = s(' (curs)', ' (lecture)')
    W = s('Exemplu rezolvat', 'Worked example')
    lam = lambda i, lab=None: select(i, lab or s('Parametrul de netezire λ', 'Smoothing parameter λ'), [(str(v), str(v) + (lec if v == 1600 else '')) for v in LAMS], '1600')  # noqa: E731
    return {
     'fourier': (slider('fo-a', s('Amplitudinea ciclului de perioadă 4 (curs: 1)', 'Amplitude of the period-4 cycle (lecture: 1)'), 0, 2, 0.25, 1)
                 + slider('fo-s', s('Abaterea standard a zgomotului (curs: 0,7)', 'Standard deviation of the noise (lecture: 0.7)'), 0, 1.5, 0.1, 0.7) + play('fo'), CSS + charts('fo-chart')),
     'aliasing': (slider('al-d', s('Intervalul de observare Δ (luni; curs: 3)', 'Sampling interval Δ (months; lecture: 3)'), 1, 6, 1, 3)
                  + slider('al-p', s('Perioada ciclului adevărat (luni; curs: 4)', 'Period of the true cycle (months; lecture: 4)'), 2, 12, 0.5, 4) + play('al'), charts('al-chart')),
     'spectra': (slider('sp-n', s('Limita superioară ν a ariei hașurate (curs: 0,5)', 'Upper limit ν of the shaded area (lecture: 0.5)'), 0.05, 0.5, 0.05, 0.5)
                 + slider('sp-s', s('Varianța zgomotului σ² (curs: 1)', 'Noise variance σ² (lecture: 1)'), 0.5, 2, 0.5, 1) + play('sp'), charts('sp-chart')),
     'sunspots': (slider('ss-a', s('Primul an (curs: 1700)', 'First year (lecture: 1700)'), 1700, 1900, 10, 1700)
                  + slider('ss-b', s('Ultimul an (curs: 2008)', 'Last year (lecture: 2008)'), 1808, 2008, 25, 2008) + play('ss'), charts('ss-chart')),
     'noise': (slider('wn-m', s('Netezirea Daniell m a ambelor periodograme (curs: 0, periodograma brută)', 'Daniell smoothing m of both periodograms (lecture: 0, the raw periodogram)'), 0, 10, 1, 0)
               + select('wn-T', s('Eșantionul din dreapta', 'Sample on the right'), [('2048', 'T = 2048' + lec), ('1024', s('primele 1024 de extrageri', 'the first 1024 draws')),
                                                                                     ('512', s('primele 512 extrageri', 'the first 512 draws')), ('256', s('primele 256 de extrageri', 'the first 256 draws'))], '2048')
               + play('wn'), charts('wn-chart')),
     'leakage': (slider('lk-j', s('Frecvența ciclului puternic, în intervale Fourier j (ν = j/256; curs: 25,5)', 'Frequency of the strong cycle in Fourier bins j (ν = j/256; lecture: 25.5)'), 25, 26, 0.1, 25.5)
                 + slider('lk-a', s('Amplitudinea ciclului slab (curs: 0,02)', 'Amplitude of the weak cycle (lecture: 0.02)'), 0.005, 0.05, 0.005, 0.02) + play('lk'), charts('lk-chart')),
     'smoothing': (slider('sm-m', s('m al estimației Daniell mai late (curs: 10)', 'm of the wider Daniell estimate (lecture: 10)'), 1, 20, 1, 10)
                   + select('sm-w', s('Lungimea segmentelor Welch', 'Welch segment length'), [('128', '128' + lec), ('64', '64'), ('256', '256')], '128') + play('sm'), charts('sm-chart')),
     'sun-ci': (slider('sc-m', s('Netezirea Daniell m (curs: 2)', 'Daniell smoothing m (lecture: 2)'), 0, 8, 1, 2)
                + select('sc-c', s('Nivelul de încredere', 'Confidence level'), [('95', '95%' + lec), ('90', '90%'), ('99', '99%')], '95') + play('sc'), charts('sc-chart')),
     'gdp-series': (lam('gs-l') + play('gs'), charts('gs-chart')),
     'gdp-spectra': (slider('gp-m', s('Netezirea Daniell m (curs: 2)', 'Daniell smoothing m (lecture: 2)'), 0, 6, 1, 2) + lam('gp-l', s('λ al ciclului HP', 'λ of the HP cycle')) + play('gp'), charts('gp-chart')),
     'load': (select('ld-y', s('Eșantionul', 'Sample'), [('all', '2022–2026' + lec)] + [(str(y), str(y)) for y in (2022, 2023, 2024, 2025)] + [('2026', s('2026 (ianuarie–iunie)', '2026 (January–June)'))], 'all')
              + select('ld-w', s('Segmentele Welch', 'Welch segments'), [('8', s('8 săptămîni', '8 weeks') + lec), ('2', s('2 săptămîni', '2 weeks')), ('4', s('4 săptămîni', '4 weeks')), ('12', s('12 săptămîni', '12 weeks'))], '8')
              + play('ld'), charts('ld-chart')),
     'long-memory': (slider('lm-b', s('Exponentul b al benzii GPH, m = ⌊T<sup>b</sup>⌋ (curs: 0,65)', 'Exponent b of the GPH bandwidth, m = ⌊T<sup>b</sup>⌋ (lecture: 0.65)'), 0.45, 0.85, 0.05, 0.65)
                     + select('lm-k', s('Intervalele de frecvență logaritmică', 'Log-frequency bins'), [('40', '40' + lec), ('20', '20'), ('80', '80')], '40') + play('lm'), charts('lm-chart')),
     'filters': (slider('fl-p', s('Perioada marcată (trimestre)', 'Period marked (quarters)'), 4, 80, 2, 32) + lam('fl-l') + play('fl'), charts('fl-chart')),
     'coherence': (select('co-n', s('Lungimea segmentelor Welch', 'Welch segment length'), [(str(v), s(f'{v} de luni', f'{v} months') + (lec if v == 96 else '')) for v in (48, 72, 96, 120, 144)], '96')
                   + play('co'), charts('co-chart')),
     'wavelet': (select('wv-w', s('Parametrul Morlet ω₀', 'Morlet parameter ω₀'), [(str(v), str(v) + (lec if v == 6 else '')) for v in (4, 5, 6, 8, 10)], '6')
                 + slider('wv-a', s('Fereastra de 36 de ani comparată (curs: 1795–1830)', 'The 36-year window compared (lecture: 1795–1830)'), 1700, 1970, 5, 1795) + play('wv'), charts('wv-chart')),
     # ---------------------------------------------------------------- EXTRA
     'wx-cycle': (slider('wy-p', s('Perioada 1/ν (observații; exemplul din curs: 12)', 'Period 1/ν (observations; example of the lecture: 12)'), 2, 24, 1, 12)
                  + slider('wy-a', s('Amplitudinea A', 'Amplitude A'), 1, 3, 0.5, 2) + slider('wy-f', s('Faza φ (radiani)', 'Phase φ (radians)'), 0, 3, 0.25, 0) + play('wy'),
                  charts('wy-chart') + wx('wy-ws', W, r'\nu=\frac{1}{P},\qquad \omega=2\pi\nu,\qquad A\cos(2\pi\nu t+\varphi)=a\cos(2\pi\nu t)+b\sin(2\pi\nu t),\quad a=A\cos\varphi,\ b=-A\sin\varphi',
                                          select('wy-u', s('Frecvența datelor', 'Frequency of the data'), [('m', s('lunare', 'monthly') + lec), ('h', s('orare', 'hourly')), ('q', s('trimestriale', 'quarterly')), ('w', s('săptămînale', 'weekly'))], 'm'))),
     'wx-dft': (slider('wd-a', s('Valoarea x₁ (curs: 4)', 'Value x₁ (lecture: 4)'), 0, 8, 1, 4) + slider('wd-b', s('Valoarea x₃ (curs: 0)', 'Value x₃ (lecture: 0)'), -4, 4, 1, 0) + play('wd'),
                charts('wd-chart') + wx('wd-ws', W, r'd(\nu_j)=T^{-1/2}\sum_{t=1}^{T}y_t\,e^{-2\pi i\nu_j t},\qquad y_t=x_t-\bar x,\qquad \sum_{t=1}^{T}y_t^2=\sum_{j=1}^{T-1}|d(\nu_j)|^2',
                                        select('wd-j', s('Frecvența din calcul', 'Frequency of the calculation'), [('1', 'ν = 1/4' + lec), ('2', 'ν = 1/2'), ('3', 'ν = 3/4')], '1'))),
     'wx-alias': (slider('wa-p', s('Perioada ciclului adevărat (luni; curs: 4)', 'Period of the true cycle (months; lecture: 4)'), 2, 12, 0.5, 4) + play('wa'),
                  charts('wa-chart') + wx('wa-ws', W, r'\nu_{\text{obs}}=\Delta\,\nu,\qquad \nu_{\text{alias}}=\bigl|\nu_{\text{obs}}-\operatorname{round}(\nu_{\text{obs}})\bigr|\le\tfrac12,\qquad \text{period}=\frac{\Delta}{\nu_{\text{alias}}}\ \text{months}',
                                          select('wa-d', s('Intervalul de observare Δ', 'Sampling interval Δ'), [('3', s('3 luni', '3 months') + lec), ('2', s('2 luni', '2 months')), ('1', s('1 lună', '1 month')), ('6', s('6 luni', '6 months'))], '3'))),
     'wx-arma': (slider('wr-p', s('φ al AR(1) (curs: 0,6)', 'φ of the AR(1) (lecture: 0.6)'), -0.9, 0.9, 0.1, 0.6) + slider('wr-t', s('θ al MA(1) (curs: 0,5)', 'θ of the MA(1) (lecture: 0.5)'), -0.9, 0.9, 0.1, 0.5) + play('wr'),
                 charts('wr-chart') + wx('wr-ws', W, r'f_{AR}(\nu)=\frac{\sigma^2}{1-2\phi\cos 2\pi\nu+\phi^2},\qquad f_{MA}(\nu)=\sigma^2(1+2\theta\cos 2\pi\nu+\theta^2),\qquad \int_{-1/2}^{1/2}f=\gamma(0)')),
     'wx-ar2': (slider('w2-a', s('φ₁ (curs: 1,5)', 'φ₁ (lecture: 1.5)'), 0.2, 1.6, 0.1, 1.5) + slider('w2-b', s('φ₂ (curs: −0,75)', 'φ₂ (lecture: −0.75)'), -0.95, -0.65, 0.05, -0.75) + play('w2'),
                charts('w2-chart') + wx('w2-ws', W, r'\cos(2\pi\nu^*)=\frac{\phi_1(\phi_2-1)}{4\phi_2},\qquad f(\nu)=\frac{\sigma^2}{|1-\phi_1e^{-2\pi i\nu}-\phi_2e^{-4\pi i\nu}|^2},\qquad f(0)=\frac{\sigma^2}{(1-\phi_1-\phi_2)^2}')),
     't-sun': (slider('ts-a', s('Limita inferioară a benzii (ani; curs: 9)', 'Lower limit of the band (years; lecture: 9)'), 5, 11, 1, 9)
               + slider('ts-b', s('Limita superioară a benzii (ani; curs: 13)', 'Upper limit of the band (years; lecture: 13)'), 11, 20, 1, 13) + play('ts'),
               charts('ts-chart') + box(select('ts-k', s('Rîndurile tabelului', 'Rows of the table'), [('3', s('cele mai mari 3 ordonate', 'the 3 largest ordinates') + lec), ('5', s('cele mai mari 5', 'the 5 largest')), ('10', s('cele mai mari 10', 'the 10 largest'))], '3'), tbl('ts-tb'))),
     'x-prop': (select('xp-T', s('Eșantionul de zgomot alb', 'White-noise sample'), [('2048', 'T = 2048'), ('128', 'T = 128')], '2048')
                + slider('xp-c', s('Pragul c, multiplu al lui f (curs: 3,00)', 'Threshold c, a multiple of f (lecture: 3.00)'), 1, 5, 0.5, 3) + play('xp'), charts('xp-chart')),
     'wx-daniell': (slider('wl-m', s('Vecini de fiecare parte m (curs: 2)', 'Neighbours on each side m (lecture: 2)'), 0, 8, 1, 2) + play('wl'),
                    charts('wl-chart') + wx('wl-ws', W, r'\hat f(\nu_j)=\frac1L\sum_{k=-m}^{m}I(\nu_{j+k}),\qquad L=2m+1,\qquad B=\frac{L}{T},\qquad \operatorname{Var}\hat f\approx\frac{f^2}{L}',
                                            slider('wl-j', s('Ordonata j (vîrful netezit din curs: 30)', 'Ordinate j (the smoothed peak of the lecture: 30)'), 20, 40, 1, 30))),
     't-smooth': (slider('tm-m', s('m al rîndului suplimentar (rîndurile din curs: 2 și 10)', 'm of the extra row (rows of the lecture: 2 and 10)'), 0, 30, 1, 10) + play('tm'),
                  charts('tm-chart') + box(select('tm-w', s('Segmentele Welch din tabel', 'Welch segments in the table'), [('128', '128' + lec), ('64', '64'), ('256', '256')], '128'), tbl('tm-tb'))),
     'wx-band': (slider('wb-L', s('Numărul de ordonate L (curs: 5)', 'Number of ordinates L (lecture: 5)'), 1, 21, 2, 5) + play('wb'),
                 charts('wb-chart') + wx('wb-ws', W, r'\frac{2L\,\hat f(\nu)}{f(\nu)}\approx\chi^2_{2L},\qquad \left[\frac{df\,\hat f}{\chi^2_{df}(1-\alpha/2)},\ \frac{df\,\hat f}{\chi^2_{df}(\alpha/2)}\right],\quad df=2L',
                                         select('wb-c', s('Nivelul de încredere', 'Confidence level'), [('95', '95%' + lec), ('90', '90%'), ('99', '99%')], '95'))),
     't-gdp': (slider('tg-a', s('Cea mai scurtă perioadă a benzii (trimestre; curs: 6)', 'Shortest period of the band (quarters; lecture: 6)'), 4, 10, 1, 6)
               + slider('tg-b', s('Cea mai lungă perioadă a benzii (trimestre; curs: 32)', 'Longest period of the band (quarters; lecture: 32)'), 16, 48, 4, 32) + play('tg'),
               charts('tg-chart') + box(select('tg-m', s('Netezirea Daniell m pentru vîrf', 'Daniell smoothing m for the peak'), [(str(v), str(v) + (lec if v == 2 else '')) for v in range(0, 7)], '2'), tbl('tg-tb'))),
     't-load': (slider('tl-w', s('Semilățimea benzii în jurul fiecărei armonici (ordonate; curs: 2)', 'Half-width of the band around each harmonic (ordinates; lecture: 2)'), 1, 10, 1, 2) + play('tl'),
                charts('tl-chart') + box(select('tl-k', s('Armonicele zilnice și săptămînale incluse', 'Daily and weekly harmonics included'), [('6', '6' + lec), ('3', '3'), ('1', s('1 (doar ciclul de bază)', '1 (the fundamental only)'))], '6'), tbl('tl-tb'))),
     'wx-hp': (lam('wh-l') + play('wh'),
               charts('wh-chart') + wx('wh-ws', W, r'G(\nu)=\frac{4\lambda(1-\cos 2\pi\nu)^2}{1+4\lambda(1-\cos 2\pi\nu)^2},\qquad |H_{\text{cycle}}|^2=G^2,\quad |H_{\text{trend}}|^2=(1-G)^2,\quad |1-e^{-2\pi i\nu}|^2=4\sin^2(\pi\nu)',
                                       slider('wh-p', s('Perioada P (trimestre; curs: 8, 32, 40)', 'Period P (quarters; lecture: 8, 32, 40)'), 2, 80, 1, 32))),
     't-filt': (lam('tf-l') + play('tf'),
                charts('tf-chart') + box(select('tf-q', s('Valorile din tabel', 'Values in the table'), [('sq', s('cîștigurile pătratice |H(ν)|²', 'squared gains |H(ν)|²') + lec), ('g', s('cîștigurile |H(ν)|', 'gains |H(ν)|'))], 'sq'), tbl('tf-tb'))),
    }


def _frame_page(lang, title):
    from interactive_common import ROOT, frames_of, latex_text
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    assert len(frames) == len(pages)
    k = [latex_text(t) for t, _ in frames].index(title)
    return int(pages[k][0]), latex_text(frames[k][0])


def _extra_pages(lang):
    """Page and title of the frames of EXTRA, from the .tex and the beamer .nav file of each language."""
    return {a: _frame_page(lang, ten if lang == 'en' else tro) for a, ten, tro, _, _ in EXTRA}


def _extra_live(anchor, lang, W):
    t = L[lang]
    ctl, ch = W[anchor]
    q, a = QA[lang][anchor]
    live = f'<div class="live"><div class="ctls">{ctl}</div>{ch}<ul class="out" id="{PFX[anchor]}-out"></ul></div>'
    tail = f'<div class="ask"><span class="ask-tag">{t["ask"]}</span> {q}</div>\n  <details><summary>{t["answer"]}</summary><p>{a}</p></details>'
    return live, tail


def widgets(lang):
    """Chart blocks for build_web_lecture.py: each host carries the web version of its EXTRA blocks (moved by ch12.js)."""
    W = widgets_all(lang)
    xp = _extra_pages(lang)
    res = {}
    for a, _, _ in BLOCKS:
        ctl, ch = W[a]
        held = ''
        for x, _, _, _, host in EXTRA:
            if host == a:
                live, tail = _extra_live(x, lang, W)
                held += f'<section class="blk wlive xblk" id="{x}" data-frame="{xp[x][0]}">{live}{tail}</section>'
        res[a] = (ctl, ch + (f'<div class="xhold">{held}</div>' if held else ''))
    return res


MATHJAX = ('<script>window.MathJax = { tex: { inlineMath: [[\'\\\\(\', \'\\\\)\']], displayMath: [[\'\\\\[\', \'\\\\]\']] }, '
           'startup: { pageReady: () => MathJax.startup.defaultPageReady().then(() => document.body.setAttribute(\'data-mj\', \'done\')) } };</script>\n'
           '<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>\n')


def _post(lang):
    """Insert the EXTRA blocks in slide order (with their section headings) and MathJax; render their slide images."""
    import pymupdf
    import interactive_common as ic
    sp = sys.modules['__main__']
    t, W = L[lang], widgets_all(lang)
    path = os.path.join(ic.OUT, t['file'])
    html = open(path, encoding='utf-8').read()
    viewer = '../pdfjs/web/viewer.html?file=../../' + TEX[lang] + '.pdf'
    chart_pages = ic.slide_pages(sp, lang)
    blocks = {m.group(1): m.group(0) for m in re.finditer(r'\n<section class="blk" id="([^"]+)">.*?</section>', html, flags=re.S)}
    items = [(chart_pages[c][0], 0, a, k, blocks[a]) for a, c, k in BLOCKS]
    xp = _extra_pages(lang)
    doc = pymupdf.open(os.path.join(ic.ROOT, TEX[lang] + '.pdf'))
    for j, (anchor, _, _, skey, _) in enumerate(EXTRA):
        pg, title = xp[anchor]
        img = f'ch12x_{anchor}_{lang}.jpg'
        doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', img), jpg_quality=82)
        live, tail = _extra_live(anchor, lang, W)
        items.append((pg, j + 1, anchor, skey, f'''
<section class="blk" id="{anchor}">
  <h3>{title}</h3>
  <div class="pair">
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/{img}" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    {live}
  </div>
  {tail}
</section>'''))
    items.sort(key=lambda x: (x[0], x[1]))
    body, nav, sec = [], [], None
    for _, _, anchor, skey, h in items:
        if skey != sec:
            sec = skey
            body.append(f'<h2 class="sec" id="s-{skey}">{SECTIONS[lang][skey]}</h2>')
            nav.append(f'<a href="#s-{skey}">{SECTIONS[lang][skey]}</a>')
        body.append(h)
    i0, i1 = html.index('<p class="toc">'), html.index('<p class="src foot">')
    html = html[:i0] + f'<p class="toc">{" · ".join(nav)}</p>\n' + ''.join(body) + '\n' + html[i1:]
    if 'MathJax-script' not in html:
        html = html.replace('</head>', MATHJAX + '</head>', 1)
    open(path, 'w', encoding='utf-8').write(html)
    print('extra blocks', lang, len(EXTRA))


# ================================================================= readouts
TEXT_JS = {
 'ro': r"""(function () {
const MON = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
const de = n => { const r = Math.abs(Math.round(n)) % 100; return Number.isInteger(+n) && ((r === 0 && +n !== 0) || r >= 20) ? ' de' : ''; };
const big = n => String(Math.round(n)).replace('-', '−').replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
const SP = ['zgomotul alb', 'AR(1), φ = 0,6', 'AR(1), φ = −0,6', 'MA(1), θ = 0,5', 'AR(2), φ = (1,5; −0,75)', 'ARMA(1,1), φ = 0,6, θ = 0,4'];
const lst = (a, f) => a.map((v, i) => f(v, i)).join('; ');
const luni = n => `${n}${de(n)} ${n === 1 ? 'lună' : 'luni'}`;
window.CH12_TEXT = {
 sep: '; ', semi: '; ', big,
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 fo1: (a, b, c, d) => `Vîrful la ν = 1/12: I = ${a} (teoria A²T/4 = ${c}); vîrful la ν = 1/4: I = ${b} (teoria ${d}).`,
 fo2: (s, v, vt) => `Cele două vîrfuri concentrează ${s} din puterea totală; varianța seriei ${v} (în teorie 2 + A²/2 + σ² = ${vt}).`,
 al1: (dt, P, v, ab) => `O observație la ${luni(dt)}: ciclul de ${P} luni are ${v} cicluri pe observație, ${ab ? 'peste frecvența Nyquist 1/2' : 'cel mult frecvența Nyquist 1/2'}.`,
 al2: (a, po, pm, ok) => ok ? `Frecvența rămîne ${a} cicluri pe observație: perioada aparentă este ${po} observații, adică ${pm} luni (ciclul adevărat).` : `Frecvența se pliază la ${a} cicluri pe observație: perioada aparentă este ${po} observații, adică ${pm} luni.`,
 sp1: (s2, v) => `σ² = ${s2}; varianța (aria sub f pe [−1/2, 1/2]): ${lst(v, (x, i) => `${SP[i]} ${x}`)}.`,
 sp2: (n, v) => `Ponderea varianței la frecvențele 0 ≤ ν ≤ ${n}: ${lst(v, (x, i) => `${SP[i]} ${x}`)}.`,
 ss1: (T, j, nu, p, o) => `${T}${de(T)} ani; cea mai mare ordonată este la j = ${j}: ν = ${nu} cicluri pe an, perioada ${p} ani; următoarele: ${o[0]} și ${o[1]} ani.`,
 ss2: s => `Perioadele între 9 și 13 ani concentrează ${s} din varianță.`,
 ss3: (m, g, i, lo, p) => `Testul lui Fisher: m = ${m}, g = ${g} (față de 1/m = ${i}), valoarea p ${lo ? '&lt; 0,001' : '= ' + p}.`,
 wn1: (T, m, s, x) => `T = ${T}: media ordonatelor ${m}, abaterea standard ${s}, maximul ${x}.`,
 wn2: (m, L, r, q, t) => m ? `Daniell m = ${m} mediază L = ${L} ordonate: abaterea standard scade aproximativ la 1/√L = ${r}; cuantila de 95% a ordonatelor ${q} (teoria χ²<sub>2L</sub>/(2L): ${t}).` : `Periodograma brută: abaterea standard rămîne aproximativ 1 pentru orice T; cuantila de 95% a ordonatelor ${q} (teoria χ²₂/2: ${t}).`,
 lk1: (nu, j, fo) => `Ciclul puternic la ν = ${nu} (j = ${j})${fo ? ': o frecvență Fourier, ciclul încape de un număr întreg de ori în eșantion' : ': între două frecvențe Fourier'}.`,
 lk2: (r, t) => `La ν = 0,3 ciclul slab este de ${r} ori peste fondul din periodograma brută și de ${t} ori peste fondul din cea cu fereastră Hann.`,
 sm1: (t, r, e) => `Vîrful adevărat la ν = ${t}; cea mai mare ordonată brută la ν = ${r}; eroarea pătratică medie a lui log I: ${e}.`,
 sm2: (m, p, e, h, ht) => `Daniell m = ${m}: vîrful la ν = ${p}, înălțimea ${h} (adevărată ${ht}), eroarea pătratică medie a lui log f̂ ${e}.`,
 sm3: (n, K, p, e) => `Welch cu segmente de ${n}${de(n)} observații (K = ${K}): vîrful la ν = ${p}, eroarea pătratică medie ${e}.`,
 sc1: (m, L, df, B) => `m = ${m}: L = ${L}${de(L)} ordonate, df = ${df}, lățimea de bandă B = L/T = ${B} cicluri pe an.`,
 sc2: (p, f, lv, lo, hi, rl, rh) => `Vîrful netezit la perioada de ${p} ani: f̂ = ${f}; intervalul de ${lv}% [${lo}; ${hi}], adică de la ${rl} la ${rh} ori f̂.`,
 sc3: (f, h, ok) => `La ν ≈ 0,18 (armonica de 5,5 ani): f̂ = ${f}, limita superioară ${h}; ${ok ? 'banda vîrfului este în întregime deasupra ei' : 'banda vîrfului nu mai este în întregime deasupra ei'}.`,
 gs1: (k, m, s, c, l) => `${k === 'US' ? 'SUA' : 'România'}: creșterea medie ${m}% pe trimestru, abaterea standard ${s}; abaterea standard a ciclului HP (λ = ${l}) ${c}% din trend.`,
 gs2: (m, s, c, ms, ss, cs, d) => `PIB-ul României a fost revizuit după realizarea slide-urilor: cu datele Eurostat din ${d} media este ${m}, abaterea standard ${s}, a ciclului HP ${c} (în slide ${ms}, ${ss}, ${cs}); pagina folosește valorile din slide.`,
 gp1: (m, us, up, ro, rp) => `Daniell m = ${m}. Creșterea SUA: ${us} din varianță în banda de 6–32 de trimestre, vîrful la ${up} trimestre; creșterea României: ${ro}, vîrful la ${rp} trimestre.`,
 gp2: (l, us, up, uy, ro, rl, rp) => `Ciclul HP (λ = ${l}) al SUA: ${us} în bandă, vîrful la ${up} trimestre (${uy} ani); al României: ${ro} în bandă, ${rl} la perioade de peste 32 de trimestre, vîrful la ${rp} trimestre.`,
 ld1: (y, T, m, K, w) => `${y === 'all' ? '2022–2026' : y}: ${big(T)}${de(T)} ore, media ${m} MW; ${K}${de(K)} segmente de ${w} săptămîni, cu suprapunere de 50%.`,
 ld2: l => `Cele mai mari ordonate Welch pînă la perioada de 336 de ore: ${l.join('; ')} ore.`,
 lm1: (b, T, m, p) => `T = ${T}, m = ⌊T<sup>${b}</sup>⌋ = ${m}${de(m)} frecvențe (perioade de cel puțin ${p} zile).`,
 lm2: (d, s, a) => `Randamentele: d̂ = ${d} (eroarea standard ${s}); randamentele absolute: d̂ = ${a}.`,
 fl1: (P, a, b) => `Perioada de ${P}${de(P)} trimestre: cîștigul pătratic al diferenței de ordinul întîi ${a}, al diferenței sezoniere 1 − L⁴ ${b}.`,
 fl2: (P, l, g, t) => `Filtrul HP (λ = ${l}) la ${P}${de(P)} trimestre: ciclul G² = ${g}, trendul (1 − G)² = ${t}.`,
 fl3: (l, p, y) => `Jumătate din putere (G² = 0,5) la perioada de ${p} trimestre (${y} ani) pentru λ = ${l}.`,
 co1: (n, K, t) => `${K}${de(K)} segmente de ${n}${de(n)} luni; pragul de semnificație de 5%: 1 − 0,05<sup>1/(K−1)</sup> = ${t}.`,
 co2: (b, x, h, c) => `Coerența pătratică în banda de 18–96 de luni: media ${b}, maximul ${x}; la perioade sub 6 luni: ${h}; corelația obișnuită ${c}.`,
 co3: (p, f, l) => `La ciclul de ${p} luni: faza ${f} radiani, o întîrziere de ${l} luni a scăderii șomajului.`,
 wv1: (w, a, b, x, y, m) => `ω₀ = ${w}: puterea medie în banda de 8–14 ani este ${x} în ${a}–${b}, față de ${y} pe tot eșantionul; maximul în ${m}.`,
 wys1: (P, nu, om) => `Pasul 1: ν = 1/${P} = ${nu} cicluri pe observație; frecvența unghiulară ω = 2πν = ${om} radiani pe observație.`,
 wys2: (u, P) => `Pasul 2: ${({ m: `în date lunare, perioada este de ${P} luni${P === 12 ? ' (ciclul anual)' : ''}`, h: `în date orare, perioada este de ${P} ore${P === 24 ? ' (ciclul zilnic)' : ''}`, q: `în date trimestriale, perioada este de ${P} trimestre, adică ${(P / 4).toFixed(2).replace('.', ',')} ani${P === 24 ? ' (ciclul de 6 ani din curs)' : ''}`, w: `în date săptămînale, perioada este de ${P} săptămîni, adică ${(P / 52).toFixed(2).replace('.', ',')} ani` })[u]}.`,
 wys3: (A, f, a, b, s, A2) => `Pasul 3: a = A cos φ = ${A}·cos ${f} = ${a}; b = −A sin φ = ${b}; a² + b² = ${s} = A² = ${A2}.`,
 wys4: (P, ok) => ok ? `Frecvența ν = 1/${P} este sub frecvența Nyquist 1/2: ciclul este vizibil în date.` : 'ν = 1/2 este chiar frecvența Nyquist: un ciclu la două observații, cea mai rapidă oscilație vizibilă.',
 wy1: (P, nu, c) => `Perioada ${P}, ν = ${nu}: în cele 48 de observații încap ${String(+c.toFixed(2)).replace('.', ',')}${de(+c.toFixed(2))} cicluri.`,
 wds1: (x, m, y) => `Pasul 1: x = (${x}), media x̄ = ${m}, abaterile y = (${y}).`,
 wds2: (nu, t, s) => `Pasul 2, la ν = ${nu}: Σ y<sub>t</sub> e<sup>−2πiνt</sup> = ${t} = ${s}.`,
 wds3: (nu, d, I) => `Pasul 3: d(${nu}) = suma/√4 = ${d}, deci |d(${nu})|² = <b>${I}</b>.`,
 wds4: (l, s, ss, v) => `Parseval: ${l} = ${s} = Σ y<sub>t</sub>² = ${ss}; varianța ${v}.`,
 wd1: (x, I) => `x = (${x}): |d(ν<sub>j</sub>)|² la ν = 0, 1/4, 1/2, 3/4: ${I}.`,
 was1: (P, nu, dt, v) => `Pasul 1: ciclul de ${P} luni are ν = 1/${P} = ${nu} cicluri pe lună; cu o observație la ${luni(dt)}: ${dt} × ${nu} = ${v} cicluri pe observație.`,
 was2: (v, ab, a) => ab ? `Pasul 2: ${v} este peste 1/2, deci se pliază: |${v} − round(${v})| = <b>${a}</b> cicluri pe observație.` : `Pasul 2: ${v} ≤ 1/2, deci frecvența rămîne <b>${a}</b> cicluri pe observație (fără aliasing).`,
 was3: (po, pm, dt) => `Pasul 3: perioada aparentă 1/${'ν'} = ${po} observații = ${pm} luni.`,
 wa1: (P, dt, pm, same) => same ? `Ciclul de ${P} luni, observat la ${luni(dt)}, apare cu perioada lui adevărată.` : `Ciclul de ${P} luni, observat la ${luni(dt)}, apare ca un ciclu de ${pm} luni.`,
 wrs1: (p, a, b, f) => `AR(1), ν = 0: cos 0 = 1, f(0) = 1/(1 − ${p})² = 1/${a}² = 1/${b} = <b>${f}</b>.`,
 wrs2: (p, a, b, f, r) => `ν = 1/2: cos π = −1, f(1/2) = 1/(1 + ${p})² = 1/${a}² = 1/${b} = <b>${f}</b>; raportul f(0)/f(1/2) = ${r}.`,
 wrs3: (t, a, b) => `MA(1) cu θ = ${t}: f(0) = (1 + θ)² = <b>${a}</b>, f(1/2) = (1 − θ)² = <b>${b}</b>.`,
 wrs4: (p, a, v, n) => `Aria: γ(0) = 1/(1 − φ²) = 1/${a} = ${v}; integrala numerică a formulei dă ${n}.`,
 wr1: (p, r) => p > 0 ? `φ > 0: puterea este la frecvențele joase, f(0)/f(1/2) = ${r}.` : p < 0 ? `φ < 0: puterea este la frecvențele înalte, f(0)/f(1/2) = ${r}.` : 'φ = 0: spectrul este plat (zgomot alb).',
 w2s1: (a, b, d, c) => `Pasul 1: φ₁² + 4φ₂ = ${a}² + 4·(${b}) = ${d}${c ? ' &lt; 0: rădăcini complexe, un pseudo-ciclu' : ' ≥ 0: rădăcini reale'}.`,
 w2s2: (a, b, c, ins, nu, p) => ins ? `Pasul 2: cos(2πν*) = ${a}·(${b} − 1)/(4·(${b})) = ${c}, deci ν* = arccos(${c})/(2π) = <b>${nu}</b> și perioada 1/ν* = <b>${p}</b> observații.` : `Pasul 2: φ₁(φ₂ − 1)/(4φ₂) = ${c} este în afara intervalului [−1, 1]: spectrul nu are vîrf interior, maximul este la ν = 0.`,
 w2s3: (fp, f0, d) => `Pasul 3: f(ν*) = ${fp}, față de f(0) = 1/(${d})² = ${f0}.`,
 w21: (ins, p, r) => ins ? `Vîrful la perioada de ${p} observații este de ${r} ori mai înalt decît f(0).` : 'Fără vîrf interior: maximul spectrului este la ν = 0.',
 tsHead: ['Rangul', 'j', 'ν<sub>j</sub>', 'Perioada (ani)', 'I(ν<sub>j</sub>)', 'Ponderea în varianță'],
 tsNote: (a, b, s, t) => `Periodograma brută a petelor solare 1700–2008 (T = 309). Perioadele între ${a} și ${b} ani: ${s} din varianță; ordonatele din tabel: ${t}.`,
 ts1: (a, b, s, n) => `Banda de ${a}–${b} ani concentrează ${s} din varianță.`,
 xp1: (T, n, m, v) => `T = ${T}: ${n}${de(n)} ordonate în (0, 1/2); media lui I/f este ${m}, varianța ${v} (distribuția exponențială: 1 și 1).`,
 xp2: (c, k, n, s, t) => `${k} din ${n}${de(n)} ordonate depășesc ${c}·f: ${s}, față de e<sup>−c</sup> = ${t} în teorie.`,
 xp3: (q, t) => `Cuantila de 95% a ordonatelor: ${q}, față de ${t} în teorie.`,
 wls1: (j, nu, p, m, L) => `Ordonata j = ${j}: ν = ${nu}, perioada ${p} ani; m = ${m}, deci L = 2m + 1 = ${L}.`,
 wls2: (l, s, L, a) => `Ordonatele mediate: ${l}; suma ${s}, împărțită la ${L}: f̂ = <b>${a}</b>.`,
 wls3: (L, B, df, r) => `Lățimea de bandă B = ${L}/309 = ${B}; df = 2L = ${df}; abaterea standard relativă aproximativ 1/√L = ${r}.`,
 wl1: (m, L, p, f) => `m = ${m} (L = ${L}): vîrful estimației Daniell este la perioada de ${p} ani, f̂ = ${f}.`,
 tmRaw: 'periodograma brută',
 tmHead: ['Estimatorul', 'L', 'B = L/T', 'df = 2L', 'Vîrful ν', 'Perioada', 'Înălțimea', 'EPM log'],
 tmNote: (n, p, h) => `AR(2) simulat, φ = (1,5; −0,75), T = 512; vîrful adevărat la ν = ${n} (perioada ${p}), înălțimea ${h}. EPM log: eroarea pătratică medie a lui log f̂ față de log f.`,
 tm1: (m, L, B, e, h, b) => `m = ${m}: L = ${L}, B = ${B}, eroarea pătratică medie ${e}, vîrful ${h}; pe grila m = 0–30 eroarea minimă este la m = ${b}.`,
 wbs1: (L, m, df, a, q0, b, q1) => `Pasul 1: L = ${L} (m = ${m}), df = 2L = ${df}; cuantilele χ²: χ²<sub>${df}</sub>(${a}) = ${q0}, χ²<sub>${df}</sub>(${b}) = ${q1}.`,
 wbs2: (df, q1, lo, q0, hi) => `Pasul 2: limita inferioară ${df}/${q1} = <b>${lo}</b> × f̂; limita superioară ${df}/${q0} = <b>${hi}</b> × f̂.`,
 wbs3: (m, p, f, lv, lo, hi) => `Pasul 3, petele solare (Daniell m = ${m}): vîrful netezit la perioada de ${p} ani, f̂ = ${f}; intervalul de ${lv}%: [${lo}; ${hi}].`,
 wb1: (L, lv, lo, hi, r) => `L = ${L}: banda de ${lv}% merge de la ${lo} la ${hi} ori f̂; limita superioară este de ${r} ori limita inferioară.`,
 tgHead: ['Seria', 'T', 'În bandă', 'Perioade mai lungi', 'Vîrful (trimestre)', 'Vîrful (ani)'],
 tgLab: ['Creșterea SUA', 'Creșterea României', 'Ciclul HP al SUA', 'Ciclul HP al României'],
 tgNote: m => `Ponderile din periodograma brută; vîrful din estimația Daniell cu m = ${m}; λ = 1600; datele pînă în 2019 T4.`,
 tg1: (a, b, l, h, u, r, uh, rh) => `Banda de ${a}–${b}${de(b)} trimestre (${l} ≤ ν ≤ ${h}): creșterea SUA ${u}, a României ${r}; ciclul HP al SUA ${uh}, al României ${rh}.`,
 tlHead: ['Componenta', 'Ponderea în varianță'],
 tlRow: [n => `ciclul zilnic ${n > 1 ? `și primele ${n - 1} armonici (24/k ore, k = 1, …, ${n})` : '(24 de ore)'}`, n => `ciclul săptămînal ${n > 1 ? `și primele ${n - 1} armonici (168/k ore, k = 1, …, ${n})` : '(168 de ore)'} și 336 de ore`, 'perioadele mai lungi de o lună (peste 720 de ore)'],
 tlNote: (l, T) => `Periodograma celor ${big(T)}${de(T)} ore. Cele mai mari ordonate, la perioade distincte: ${l} ore.`,
 tl1: (w, b, d, k, l) => `Semilățimea ${w}: în jurul fiecărei armonici intră ${b}${de(b)} ordonate; zilnic ${d}, săptămînal ${k}, peste o lună ${l}.`,
 whs1: (P, nu, om, c) => `Pasul 1: P = ${P}, ν = 1/${P} = ${nu}, 2πν = ${om}; 1 − cos 2πν = ${c}.`,
 whs2: (l, c, A) => `Pasul 2: 4λ(1 − cos 2πν)² = 4·${l}·${c}² = ${A}.`,
 whs3: (A, G, g, t) => `Pasul 3: G = ${A}/(1 + ${A}) = <b>${G}</b>; ciclul G² = <b>${g}</b>, trendul (1 − G)² = <b>${t}</b>.`,
 whs4: (P, d) => `Pentru comparație, diferența de ordinul întîi la ${P}${de(P)} trimestre: 4 sin²(π/${P}) = ${d}.`,
 wh1: (l, p, y) => `λ = ${l}: jumătate din putere la ${p} trimestre (${y} ani).`,
 tfHead: sq => ['Perioada (trimestre)', 'ν', '1 − L', '1 − L⁴', sq ? 'Ciclul HP, G²' : 'Ciclul HP, G', sq ? 'Trendul HP, (1 − G)²' : 'Trendul HP, 1 − G'],
 tfNote: (sq, l) => `${sq ? 'Cîștigurile pătratice' : 'Cîștigurile'}; λ = ${l}; rîndurile evidențiate sînt perioadele din curs.`,
 tf1: (l, a, b, c, h) => `λ = ${l}: G² = ${a} la 8 trimestre, ${b} la 32, ${c} la 40; jumătate din putere la ${h} trimestre.`,
};
})();
""",
 'en': r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const big = n => String(Math.round(n)).replace('-', '−').replace(/\B(?=(\d{3})+(?!\d))/g, ',');
const SP = ['white noise', 'AR(1), φ = 0.6', 'AR(1), φ = −0.6', 'MA(1), θ = 0.5', 'AR(2), φ = (1.5, −0.75)', 'ARMA(1,1), φ = 0.6, θ = 0.4'];
const lst = (a, f) => a.map((v, i) => f(v, i)).join('; ');
const mon = n => `${n} ${n === 1 ? 'month' : 'months'}`;
window.CH12_TEXT = {
 sep: '; ', semi: ', ', big,
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 fo1: (a, b, c, d) => `Spike at ν = 1/12: I = ${a} (theory A²T/4 = ${c}); spike at ν = 1/4: I = ${b} (theory ${d}).`,
 fo2: (s, v, vt) => `The two spikes hold ${s} of the total power; variance of the series ${v} (in theory 2 + A²/2 + σ² = ${vt}).`,
 al1: (dt, P, v, ab) => `One observation every ${mon(dt)}: the ${P}-month cycle has ${v} cycles per observation, ${ab ? 'above the Nyquist frequency 1/2' : 'at most the Nyquist frequency 1/2'}.`,
 al2: (a, po, pm, ok) => ok ? `The frequency stays ${a} cycles per observation: the apparent period is ${po} observations, i.e. ${pm} months (the true cycle).` : `The frequency folds to ${a} cycles per observation: the apparent period is ${po} observations, i.e. ${pm} months.`,
 sp1: (s2, v) => `σ² = ${s2}; variance (area under f over [−1/2, 1/2]): ${lst(v, (x, i) => `${SP[i]} ${x}`)}.`,
 sp2: (n, v) => `Share of the variance at frequencies 0 ≤ ν ≤ ${n}: ${lst(v, (x, i) => `${SP[i]} ${x}`)}.`,
 ss1: (T, j, nu, p, o) => `${T} years; the highest ordinate is at j = ${j}: ν = ${nu} cycles per year, period ${p} years; the next: ${o[0]} and ${o[1]} years.`,
 ss2: s => `Periods between 9 and 13 years carry ${s} of the variance.`,
 ss3: (m, g, i, lo, p) => `Fisher's test: m = ${m}, g = ${g} (against 1/m = ${i}), p-value ${lo ? '&lt; 0.001' : '= ' + p}.`,
 wn1: (T, m, s, x) => `T = ${T}: mean of the ordinates ${m}, standard deviation ${s}, maximum ${x}.`,
 wn2: (m, L, r, q, t) => m ? `Daniell m = ${m} averages L = ${L} ordinates: the standard deviation falls to about 1/√L = ${r}; 95% quantile of the ordinates ${q} (theory χ²<sub>2L</sub>/(2L): ${t}).` : `Raw periodogram: the standard deviation stays about 1 for any T; 95% quantile of the ordinates ${q} (theory χ²₂/2: ${t}).`,
 lk1: (nu, j, fo) => `Strong cycle at ν = ${nu} (j = ${j})${fo ? ': a Fourier frequency, the cycle fits a whole number of times in the sample' : ': between two Fourier frequencies'}.`,
 lk2: (r, t) => `At ν = 0.3 the weak cycle stands ${r} times above the background of the raw periodogram and ${t} times above that of the Hann-tapered one.`,
 sm1: (t, r, e) => `True peak at ν = ${t}; highest raw ordinate at ν = ${r}; mean squared error of log I ${e}.`,
 sm2: (m, p, e, h, ht) => `Daniell m = ${m}: peak at ν = ${p}, height ${h} (true ${ht}), mean squared error of log f̂ ${e}.`,
 sm3: (n, K, p, e) => `Welch with segments of ${n} (K = ${K}): peak at ν = ${p}, mean squared error ${e}.`,
 sc1: (m, L, df, B) => `m = ${m}: L = ${L} ordinates, df = ${df}, bandwidth B = L/T = ${B} cycles per year.`,
 sc2: (p, f, lv, lo, hi, rl, rh) => `Smoothed peak at a period of ${p} years: f̂ = ${f}; ${lv}% interval [${lo}, ${hi}], i.e. from ${rl} to ${rh} times f̂.`,
 sc3: (f, h, ok) => `At ν ≈ 0.18 (the 5.5-year harmonic): f̂ = ${f}, upper bound ${h}; ${ok ? 'the band of the peak lies entirely above it' : 'the band of the peak no longer lies entirely above it'}.`,
 gs1: (k, m, s, c, l) => `${k === 'US' ? 'United States' : 'Romania'}: mean growth ${m}% per quarter, standard deviation ${s}; standard deviation of the HP cycle (λ = ${l}) ${c}% of trend.`,
 gs2: (m, s, c, ms, ss, cs, d) => `Romanian GDP has been revised since the slides were made: with the Eurostat data of ${d} the mean is ${m}, the standard deviation ${s}, that of the HP cycle ${c} (slide: ${ms}, ${ss}, ${cs}); the page uses the slide values.`,
 gp1: (m, us, up, ro, rp) => `Daniell m = ${m}. US growth: ${us} of the variance in the band of 6–32 quarters, peak at ${up} quarters; Romanian growth: ${ro}, peak at ${rp} quarters.`,
 gp2: (l, us, up, uy, ro, rl, rp) => `HP cycle (λ = ${l}) of the US: ${us} in the band, peak at ${up} quarters (${uy} years); of Romania: ${ro} in the band, ${rl} at periods longer than 32 quarters, peak at ${rp} quarters.`,
 ld1: (y, T, m, K, w) => `${y === 'all' ? '2022–2026' : y}: ${big(T)} hours, mean ${m} MW; ${K} segments of ${w} weeks, overlapping by 50%.`,
 ld2: l => `Highest Welch ordinates up to a period of 336 hours: ${l.join(', ')} hours.`,
 lm1: (b, T, m, p) => `T = ${T}, m = ⌊T<sup>${b}</sup>⌋ = ${m} frequencies (periods of at least ${p} days).`,
 lm2: (d, s, a) => `Returns: d̂ = ${d} (standard error ${s}); absolute returns: d̂ = ${a}.`,
 fl1: (P, a, b) => `Period of ${P} quarters: squared gain of the first difference ${a}, of the seasonal difference 1 − L⁴ ${b}.`,
 fl2: (P, l, g, t) => `HP filter (λ = ${l}) at ${P} quarters: cycle G² = ${g}, trend (1 − G)² = ${t}.`,
 fl3: (l, p, y) => `Half power (G² = 0.5) at a period of ${p} quarters (${y} years) for λ = ${l}.`,
 co1: (n, K, t) => `${K} segments of ${n} months; 5% significance threshold: 1 − 0.05<sup>1/(K−1)</sup> = ${t}.`,
 co2: (b, x, h, c) => `Squared coherence in the band of 18–96 months: mean ${b}, maximum ${x}; at periods below 6 months: ${h}; ordinary correlation ${c}.`,
 co3: (p, f, l) => `At the ${p}-month cycle: phase ${f} radians, a lag of ${l} months of the fall in unemployment.`,
 wv1: (w, a, b, x, y, m) => `ω₀ = ${w}: mean power in the 8–14-year band ${x} in ${a}–${b}, against ${y} over the whole sample; maximum in ${m}.`,
 wys1: (P, nu, om) => `Step 1: ν = 1/${P} = ${nu} cycles per observation; angular frequency ω = 2πν = ${om} radians per observation.`,
 wys2: (u, P) => `Step 2: ${({ m: `in monthly data the period is ${P} months${P === 12 ? ' (the annual cycle)' : ''}`, h: `in hourly data the period is ${P} hours${P === 24 ? ' (the daily cycle)' : ''}`, q: `in quarterly data the period is ${P} quarters, i.e. ${(P / 4).toFixed(2)} years${P === 24 ? ' (the 6-year cycle of the lecture)' : ''}`, w: `in weekly data the period is ${P} weeks, i.e. ${(P / 52).toFixed(2)} years` })[u]}.`,
 wys3: (A, f, a, b, s, A2) => `Step 3: a = A cos φ = ${A}·cos ${f} = ${a}; b = −A sin φ = ${b}; a² + b² = ${s} = A² = ${A2}.`,
 wys4: (P, ok) => ok ? `The frequency ν = 1/${P} is below the Nyquist frequency 1/2: the cycle is visible in the data.` : 'ν = 1/2 is the Nyquist frequency itself: one cycle every two observations, the fastest visible oscillation.',
 wy1: (P, nu, c) => `Period ${P}, ν = ${nu}: the 48 observations hold ${+c.toFixed(2)} cycles.`,
 wds1: (x, m, y) => `Step 1: x = (${x}), mean x̄ = ${m}, deviations y = (${y}).`,
 wds2: (nu, t, s) => `Step 2, at ν = ${nu}: Σ y<sub>t</sub> e<sup>−2πiνt</sup> = ${t} = ${s}.`,
 wds3: (nu, d, I) => `Step 3: d(${nu}) = sum/√4 = ${d}, so |d(${nu})|² = <b>${I}</b>.`,
 wds4: (l, s, ss, v) => `Parseval: ${l} = ${s} = Σ y<sub>t</sub>² = ${ss}; variance ${v}.`,
 wd1: (x, I) => `x = (${x}): |d(ν<sub>j</sub>)|² at ν = 0, 1/4, 1/2, 3/4: ${I}.`,
 was1: (P, nu, dt, v) => `Step 1: the ${P}-month cycle has ν = 1/${P} = ${nu} cycles per month; with one observation every ${mon(dt)}: ${dt} × ${nu} = ${v} cycles per observation.`,
 was2: (v, ab, a) => ab ? `Step 2: ${v} is above 1/2, so it folds: |${v} − round(${v})| = <b>${a}</b> cycles per observation.` : `Step 2: ${v} ≤ 1/2, so the frequency stays <b>${a}</b> cycles per observation (no aliasing).`,
 was3: (po, pm, dt) => `Step 3: apparent period 1/ν = ${po} observations = ${pm} months.`,
 wa1: (P, dt, pm, same) => same ? `The ${P}-month cycle, observed every ${mon(dt)}, appears with its true period.` : `The ${P}-month cycle, observed every ${mon(dt)}, appears as a ${pm}-month cycle.`,
 wrs1: (p, a, b, f) => `AR(1), ν = 0: cos 0 = 1, f(0) = 1/(1 − ${p})² = 1/${a}² = 1/${b} = <b>${f}</b>.`,
 wrs2: (p, a, b, f, r) => `ν = 1/2: cos π = −1, f(1/2) = 1/(1 + ${p})² = 1/${a}² = 1/${b} = <b>${f}</b>; ratio f(0)/f(1/2) = ${r}.`,
 wrs3: (t, a, b) => `MA(1) with θ = ${t}: f(0) = (1 + θ)² = <b>${a}</b>, f(1/2) = (1 − θ)² = <b>${b}</b>.`,
 wrs4: (p, a, v, n) => `Area: γ(0) = 1/(1 − φ²) = 1/${a} = ${v}; the numerical integral of the formula gives ${n}.`,
 wr1: (p, r) => p > 0 ? `φ > 0: the power sits at low frequencies, f(0)/f(1/2) = ${r}.` : p < 0 ? `φ < 0: the power sits at high frequencies, f(0)/f(1/2) = ${r}.` : 'φ = 0: the spectrum is flat (white noise).',
 w2s1: (a, b, d, c) => `Step 1: φ₁² + 4φ₂ = ${a}² + 4·(${b}) = ${d}${c ? ' &lt; 0: complex roots, a pseudo-cycle' : ' ≥ 0: real roots'}.`,
 w2s2: (a, b, c, ins, nu, p) => ins ? `Step 2: cos(2πν*) = ${a}·(${b} − 1)/(4·(${b})) = ${c}, so ν* = arccos(${c})/(2π) = <b>${nu}</b> and the period 1/ν* = <b>${p}</b> observations.` : `Step 2: φ₁(φ₂ − 1)/(4φ₂) = ${c} lies outside [−1, 1]: no interior peak, the maximum is at ν = 0.`,
 w2s3: (fp, f0, d) => `Step 3: f(ν*) = ${fp}, against f(0) = 1/(${d})² = ${f0}.`,
 w21: (ins, p, r) => ins ? `The peak at a period of ${p} observations is ${r} times higher than f(0).` : 'No interior peak: the maximum of the spectrum is at ν = 0.',
 tsHead: ['Rank', 'j', 'ν<sub>j</sub>', 'Period (years)', 'I(ν<sub>j</sub>)', 'Share of the variance'],
 tsNote: (a, b, s, t) => `Raw periodogram of the sunspots 1700–2008 (T = 309). Periods between ${a} and ${b} years: ${s} of the variance; the ordinates of the table: ${t}.`,
 ts1: (a, b, s, n) => `The band of ${a}–${b} years carries ${s} of the variance.`,
 xp1: (T, n, m, v) => `T = ${T}: ${n} ordinates in (0, 1/2); mean of I/f ${m}, variance ${v} (exponential distribution: 1 and 1).`,
 xp2: (c, k, n, s, t) => `${k} of ${n} ordinates exceed ${c}·f: ${s}, against e<sup>−c</sup> = ${t} in theory.`,
 xp3: (q, t) => `95% quantile of the ordinates: ${q}, against ${t} in theory.`,
 wls1: (j, nu, p, m, L) => `Ordinate j = ${j}: ν = ${nu}, period ${p} years; m = ${m}, so L = 2m + 1 = ${L}.`,
 wls2: (l, s, L, a) => `Ordinates averaged: ${l}; sum ${s}, divided by ${L}: f̂ = <b>${a}</b>.`,
 wls3: (L, B, df, r) => `Bandwidth B = ${L}/309 = ${B}; df = 2L = ${df}; relative standard deviation about 1/√L = ${r}.`,
 wl1: (m, L, p, f) => `m = ${m} (L = ${L}): the peak of the Daniell estimate is at a period of ${p} years, f̂ = ${f}.`,
 tmRaw: 'raw periodogram',
 tmHead: ['Estimator', 'L', 'B = L/T', 'df = 2L', 'Peak ν', 'Period', 'Height', 'MSE log'],
 tmNote: (n, p, h) => `Simulated AR(2), φ = (1.5, −0.75), T = 512; true peak at ν = ${n} (period ${p}), height ${h}. MSE log: mean squared error of log f̂ against log f.`,
 tm1: (m, L, B, e, h, b) => `m = ${m}: L = ${L}, B = ${B}, mean squared error ${e}, peak ${h}; on the grid m = 0–30 the smallest error is at m = ${b}.`,
 wbs1: (L, m, df, a, q0, b, q1) => `Step 1: L = ${L} (m = ${m}), df = 2L = ${df}; chi-square quantiles: χ²<sub>${df}</sub>(${a}) = ${q0}, χ²<sub>${df}</sub>(${b}) = ${q1}.`,
 wbs2: (df, q1, lo, q0, hi) => `Step 2: lower bound ${df}/${q1} = <b>${lo}</b> × f̂; upper bound ${df}/${q0} = <b>${hi}</b> × f̂.`,
 wbs3: (m, p, f, lv, lo, hi) => `Step 3, the sunspots (Daniell m = ${m}): smoothed peak at a period of ${p} years, f̂ = ${f}; ${lv}% interval: [${lo}, ${hi}].`,
 wb1: (L, lv, lo, hi, r) => `L = ${L}: the ${lv}% band runs from ${lo} to ${hi} times f̂; the upper bound is ${r} times the lower bound.`,
 tgHead: ['Series', 'T', 'In the band', 'Longer periods', 'Peak (quarters)', 'Peak (years)'],
 tgLab: ['US growth', 'Romanian growth', 'US HP cycle', 'Romanian HP cycle'],
 tgNote: m => `Shares from the raw periodogram; peak from the Daniell estimate with m = ${m}; λ = 1600; data up to 2019 Q4.`,
 tg1: (a, b, l, h, u, r, uh, rh) => `Band of ${a}–${b} quarters (${l} ≤ ν ≤ ${h}): US growth ${u}, Romanian growth ${r}; US HP cycle ${uh}, Romanian HP cycle ${rh}.`,
 tlHead: ['Component', 'Share of the variance'],
 tlRow: [n => `daily cycle ${n > 1 ? `and its first ${n - 1} harmonics (24/k hours, k = 1, …, ${n})` : '(24 hours)'}`, n => `weekly cycle ${n > 1 ? `and its first ${n - 1} harmonics (168/k hours, k = 1, …, ${n})` : '(168 hours)'} and 336 hours`, 'periods longer than a month (above 720 hours)'],
 tlNote: (l, T) => `Periodogram of the ${big(T)} hours. Largest ordinates, at distinct periods: ${l} hours.`,
 tl1: (w, b, d, k, l) => `Half-width ${w}: ${b} ordinates around each harmonic; daily ${d}, weekly ${k}, longer than a month ${l}.`,
 whs1: (P, nu, om, c) => `Step 1: P = ${P}, ν = 1/${P} = ${nu}, 2πν = ${om}; 1 − cos 2πν = ${c}.`,
 whs2: (l, c, A) => `Step 2: 4λ(1 − cos 2πν)² = 4·${l}·${c}² = ${A}.`,
 whs3: (A, G, g, t) => `Step 3: G = ${A}/(1 + ${A}) = <b>${G}</b>; cycle G² = <b>${g}</b>, trend (1 − G)² = <b>${t}</b>.`,
 whs4: (P, d) => `For comparison, the first difference at ${P} quarters: 4 sin²(π/${P}) = ${d}.`,
 wh1: (l, p, y) => `λ = ${l}: half power at ${p} quarters (${y} years).`,
 tfHead: sq => ['Period (quarters)', 'ν', '1 − L', '1 − L⁴', sq ? 'HP cycle, G²' : 'HP cycle, G', sq ? 'HP trend, (1 − G)²' : 'HP trend, 1 − G'],
 tfNote: (sq, l) => `${sq ? 'Squared gains' : 'Gains'}; λ = ${l}; the highlighted rows are the periods of the lecture.`,
 tf1: (l, a, b, c, h) => `λ = ${l}: G² = ${a} at 8 quarters, ${b} at 32, ${c} at 40; half power at ${h} quarters.`,
};
})();
""",
}

QA = {
 'ro': {
  'fourier': ('Puneți amplitudinea ciclului de perioadă 4 la 2, cît amplitudinea ciclului anual. Cît de înalt devine vîrful de la ν = 1/4?',
              '119,2, de patru ori mai înalt decît cu amplitudinea 1 (29,6) și aproape de teoria A²T/4 = 4·120/4 = 120: vîrful crește cu pătratul amplitudinii. Vîrful de la 1/12 rămîne 139,7, peste valoarea teoretică 120, pentru că la aceeași frecvență contribuie și zgomotul; fără zgomot cele două vîrfuri sînt exact 120 și 30.'),
  'aliasing': ('▶ modifică intervalul de observare. La Δ = 4 luni (o observație pe ciclu), ce arată observațiile?',
               'O constantă: ciclul are 4/4 = 1 ciclu pe observație, care se pliază la |1 − 1| = 0, deci fiecare observație cade în același punct al ciclului și ciclul de 4 luni dispare din date (un alias de perioadă infinită). La Δ = 5 luni același ciclu apare ca un ciclu de 20 de luni.'),
  'spectra': ('Mutați limita superioară la ν = 0,1. Ce pondere din varianța AR(1) cu φ = 0,6 se află la frecvențele sub 0,1 (perioade de peste 10 observații)?',
              '58%, față de 20% pentru zgomotul alb, care își împrăștie varianța uniform: banda [0; 0,1] este o cincime din [0; 0,5]. Pentru AR(2) ponderea este 81%, pentru AR(1) cu φ = −0,6 doar 5%.'),
  'sunspots': ('▶ scurtează eșantionul. Cu anii 1700–1808 (109 ani), la ce perioadă este cea mai mare ordonată?',
               'La 10,9 ani (j = 10), aproape de cei 11,0 ani ai eșantionului întreg: ciclul se vede și într-un secol de date, dar grila Fourier este mai rară (perioadele 109/j: 12,1; 10,9; 9,9 ani). Testul lui Fisher dă g = 0,222 și valoarea p 0,00009.'),
  'noise': ('▶ crește m-ul Daniell. La m = 10 (L = 21), cît este abaterea standard a ordonatelor netezite pentru T = 2048?',
            '0,21, aproape de 1/√21 = 0,22: media a L ordonate aproape independente împarte abaterea standard la √L; pentru T = 128 este 0,20. Un eșantion mai lung nu reușea singur acest lucru: periodogramele brute au abaterile standard 1,03 și 1,00.'),
  'leakage': ('Mutați ciclul puternic la j = 26 (ν = 26/256, o frecvență Fourier). De cîte ori este ciclul slab peste fond în periodograma brută?',
              'De 755 de ori, față de 7,7 la j = 25,5: un ciclu care încape de un număr întreg de ori în eșantion nu produce scurgere, iar fondul coboară la nivelul zgomotului foarte mic. Cu fereastra Hann raportul este 4 176, aproape același pentru orice j.'),
  'smoothing': ('▶ lărgește a doua fereastră Daniell. La m = 20 (L = 41), unde este vîrful ei și cît de înalt este?',
                'La ν = 0,0625 (perioada 16,0), cu înălțimea 36,1, față de vîrful adevărat de la ν = 0,080 cu înălțimea 64: fereastra are lățimea 41/512 = 0,08, cît frecvența vîrfului, deci îl aplatizează și îl deplasează. Eroarea pătratică medie a lui log f̂ crește din nou, la 0,11 (0,09 la m = 10 și la m = 15).'),
  'sun-ci': ('Puneți m = 0 (periodograma brută, df = 2). De cîte ori f̂ este limita superioară a benzii de 95% la vîrf?',
             'De 39,5 ori (2/χ²₂(0,025) = 2/0,0506), iar limita inferioară este 0,27 × f̂: o singură ordonată spune foarte puțin despre înălțimea spectrului. Cu m = 2 multiplicatorii sînt 0,49 și 3,08, cu m = 8 sînt 0,65 și 1,72.'),
  'gdp-series': ('Puneți λ = 100. Abaterea standard a ciclului HP al SUA crește sau scade față de 1,58?',
                 'Scade, la 1,03: un λ mai mic lasă trendul să urmeze mai îndeaproape datele, deci ciclul păstrează doar fluctuațiile mai rapide; cu λ = 25 600 crește la 2,23. Ratele de creștere nu depind de λ (abaterea standard 0,93).'),
  'gdp-spectra': ('Puneți m = 0 (periodogramele brute). La ce perioadă este cea mai mare ordonată a ciclului HP al SUA?',
                  'La 26,5 trimestre (6,6 ani), față de 24,3 cu m = 2: o ordonată brută este zgomotoasă, iar vîrful netezit se poate muta cu o frecvență Fourier (T = 292: 292/11 = 26,5, 292/12 = 24,3). Vîrful creșterii SUA rămîne la 8,8 trimestre.'),
  'load': ('Alegeți 2026 (ianuarie–iunie, 4 344 de ore). La ce perioadă este cea mai mare ordonată Welch cu segmente de 8 săptămîni?',
           'La 12 ore, înaintea celei de 24 de ore (la fel în 2025); pe 2022–2026 cea mai mare este ordonata de 24 de ore. Profilul zilnic cu două vîrfuri, dimineața și seara, pune multă putere în armonica de 12 ore, iar ordinea celor două se poate schimba de la un an la altul.'),
  'long-memory': ('Puneți exponentul la 0,45 (m = 52 de frecvențe, perioade de peste 129 de zile). Cît este d̂ pentru randamentele absolute?',
                  '0,38, față de 0,56 cu m = 307, cu eroarea standard 0,089; cu exponentul 0,80 (m = 1 152) este 0,41. Estimația depinde de lățimea benzii: puține frecvențe dau o eroare standard mare, multe frecvențe aduc efecte de memorie scurtă. Pentru randamente d̂ rămîne între −0,04 și 0,14.'),
  'filters': ('▶ mută perioada marcată. La 4 trimestre, cît este cîștigul pătratic al diferenței sezoniere 1 − L⁴?',
              '0: 4 sin²(4π/4) = 4 sin²(π) = 0. Diferența sezonieră elimină ciclul anual al datelor trimestriale (și armonica lui de 2 trimestre), în timp ce diferența de ordinul întîi îl amplifică de 2 ori. La 8 trimestre diferența sezonieră are cîștigul maxim, 4.'),
  'coherence': ('Alegeți segmente de 48 de luni (K = 17). Ce se întîmplă cu pragul de semnificație de 5%?',
                'Scade de la 0,35 la 0,17 (1 − 0,05<sup>1/16</sup>): mai multe segmente fac coerența netezită mai precisă, deci o valoare mai mică este deja semnificativă. Coerența medie în banda de 18–96 de luni crește la 0,90, iar faza la ciclul de 48 de luni dă o întîrziere de 2,1 luni. Cu segmente de 144 de luni (K = 5) pragul este 0,53.'),
  'wavelet': ('Alegeți ω₀ = 10 (mai multe oscilații în undă). Cum se schimbă puterea medie de 8–14 ani în 1795–1830?',
              'Crește de la 2,0 la 2,5, iar media pe tot eșantionul trece de la 5,7 la 5,9: o undă mai lungă mediază pe mai mulți ani, deci minimul slab din jurul lui 1800 se amestecă cu vecinii lui. Anul maximului rămîne în jurul lui 1956.'),
  'wx-cycle': ('Puneți φ = 1,5 radiani, cu A = 2. Cît sînt a și b?',
               'a = 2 cos 1,5 = 0,14 și b = −2 sin 1,5 = −1,99: cu o fază apropiată de π/2 ciclul este aproape un sinus pur, cu semn negativ. Oricare ar fi faza, a² + b² = 4 = A².'),
  'wx-dft': ('▶ mută x₃. Cu x = (4, 2, 4, 2), unde se află varianța?',
             'Toată la ν = 1/2: abaterile (1, −1, 1, −1) alternează, deci |d(1/2)|² = (−4)²/4 = 4, iar |d(1/4)|² = |d(3/4)|² = 0; Parseval: 4 = 1 + 1 + 1 + 1. Seria are perioada 2.'),
  'wx-alias': ('Puneți perioada ciclului adevărat la 2,5 luni, cu observații trimestriale (Δ = 3). Ce perioadă pare să aibă ciclul?',
               '15 luni: 3/2,5 = 1,2 cicluri pe trimestru, care se pliază la |1,2 − 1| = 0,2, adică 5 trimestre. Un ciclu mai rapid decît observațiile apare ca un ciclu mai lent, fals.'),
  'wx-arma': ('Puneți φ = −0,6. Cît sînt f(0) și f(1/2)?',
              'f(0) = 1/1,6² = 0,39 și f(1/2) = 1/0,4² = 6,25: imaginea în oglindă a cazului φ = 0,6. Un φ negativ face seria să alterneze, deci puterea trece la frecvențele cele mai înalte; varianța rămîne 1,5625.'),
  'wx-ar2': ('▶ mută φ₂ spre −0,95, cu φ₁ = 1,5. Vîrful se mută spre o perioadă mai lungă sau mai scurtă?',
             'Spre una mai scurtă: 9,1 observații la φ₂ = −0,95 (12,4 la −0,75 și 20,2 la −0,65), iar vîrful devine mult mai înalt (981 față de 64). Un φ₂ mai negativ aduce rădăcinile complexe mai aproape de cercul unitate (modulul lor este √−φ₂) și accelerează oscilația ACF.'),
  't-sun': ('Lărgiți banda la 9–20 de ani. Ce pondere din varianță concentrează?',
            '63%, față de 60% pentru 9–13 ani: perioadele dintre 13 și 20 de ani adaugă doar 3 puncte procentuale. Aproape toată puterea ciclului solar este între 9 și 13 ani; cu banda de 5–20 de ani ponderea ajunge la 73%, pentru că intră și armonica de aproximativ 5,5 ani.'),
  'x-prop': ('Puneți c = 1 (chiar valoarea adevărată f). Ce pondere din ordonatele zgomotului alb depășește f?',
             '37,6% pentru T = 2048, aproape de e<sup>−1</sup> = 36,8%, nu 50%: distribuția exponențială are media 1, dar mediana ln 2 = 0,69, deci majoritatea ordonatelor sînt sub f, iar puținele valori mari ridică media. Pentru c = 3 ponderea este 5,1% (teoria 5,0%).'),
  'wx-daniell': ('Puneți m = 0. La ce perioadă este vîrful?',
                 'La 11,0 ani (j = 28), cea mai mare ordonată brută; cu m = 2 vîrful netezit trece la 10,3 ani (j = 30), pentru că media celor cinci ordonate din jurul lui j = 30 este cea mai mare. Cu m = 4 vîrful revine la 11,0 ani: poziția vîrfului netezit variază cu una-două frecvențe Fourier.'),
  't-smooth': ('▶ trece prin m = 0, 2, …, 30. La ce m este cea mai mică eroare pătratică medie a lui log f̂?',
               'La m = 12 (0,089), foarte aproape de m = 10 (0,092); la m = 30 eroarea urcă la 0,22. Pînă la m = 12 scade varianța, apoi domină distorsiunea: înălțimea vîrfului scade de la 64 la 44,7 la m = 12 și la 31,5 la m = 30.'),
  'wx-band': ('Puneți L = 21. Cît este multiplicatorul superior al benzii de 95%?',
              '1,62 (și cel inferior 0,68), față de 3,08 și 0,49 pentru L = 5: banda se îngustează aproximativ ca 1/√L. Banda rămîne asimetrică, mai largă în sus, pentru că distribuția χ² are coada spre dreapta.'),
  't-gdp': ('Puneți cea mai lungă perioadă a benzii la 48 de trimestre. Ce pondere din varianța ciclului HP al SUA intră în bandă?',
            '90%, față de 80% pentru 6–32 de trimestre: ciclul HP al SUA are încă multă putere între 8 și 12 ani. Pentru creșterea României ponderea rămîne 15,6%, pentru că varianța ei este la frecvențele înalte.'),
  't-load': ('Puneți semilățimea la 1. Cît scade ponderea ciclului zilnic și a armonicelor lui?',
             'Foarte puțin: de la 33,7% la 33,4%. Ciclul zilnic este aproape perfect periodic, deci puterea lui stă în cîteva ordonate foarte înguste la 24/k ore; ciclul săptămînal scade mai mult, de la 15,7% la 14,9%: puterea lui este răspîndită pe mai multe ordonate vecine.'),
  'wx-hp': ('Alegeți λ = 6400. La ce perioadă filtrul HP pentru ciclu păstrează jumătate din putere?',
            'La 45,0 trimestre (11,3 ani), față de 31,8 trimestre pentru λ = 1600: un λ de 4 ori mai mare mută pragul spre ciclurile mai lente, aproximativ cu factorul 4<sup>1/4</sup> = 1,41. La 32 de trimestre G² crește de la 0,49 la 0,82.'),
  't-filt': ('Alegeți λ = 100. Cît este cîștigul pătratic al ciclului HP la 32 de trimestre?',
             '0,017, față de 0,49 pentru λ = 1600: cu λ = 100 trendul urmează aproape toate ciclurile de 8 ani, iar ciclul HP păstrează doar perioadele scurte (jumătate din putere la 15,8 trimestre). La 8 trimestre cîștigul pătratic este încă 0,94.'),
 },
 'en': {
  'fourier': ('Set the amplitude of the period-4 cycle to 2, as large as the annual cycle. How high is the spike at ν = 1/4 now?',
              '119.2, four times higher than with amplitude 1 (29.6) and close to the theory A²T/4 = 4·120/4 = 120: the spike grows with the square of the amplitude. The spike at 1/12 stays at 139.7, above its theoretical 120, because the noise also contributes at that frequency; without noise the two spikes are exactly 120 and 30.'),
  'aliasing': ('▶ changes the sampling interval. At Δ = 4 months (one observation per cycle), what do the observations show?',
               'A constant: the cycle has 4/4 = 1 cycle per observation, which folds to |1 − 1| = 0, so every observation falls at the same point of the cycle and the 4-month cycle disappears from the data (an alias of infinite period). At Δ = 5 months the same cycle appears as a 20-month cycle.'),
  'spectra': ('Move the upper limit to ν = 0.1. What share of the variance of the AR(1) with φ = 0.6 lies at frequencies below 0.1 (periods longer than 10 observations)?',
              '58%, against 20% for white noise, which spreads its variance evenly: the band [0, 0.1] is a fifth of [0, 0.5]. For the AR(2) the share is 81%, for the AR(1) with φ = −0.6 only 5%.'),
  'sunspots': ('▶ shortens the sample. With 1700–1808 (109 years), at what period is the highest ordinate?',
               'At 10.9 years (j = 10), close to the 11.0 years of the full sample: the cycle shows in a century of data, but the Fourier grid is coarser (periods 109/j: 12.1, 10.9, 9.9 years). Fisher’s test gives g = 0.222 and a p-value of 0.00009.'),
  'noise': ('▶ raises the Daniell m. At m = 10 (L = 21), what is the standard deviation of the smoothed ordinates for T = 2048?',
            '0.21, close to 1/√21 = 0.22: averaging L nearly independent ordinates divides the standard deviation by √L; for T = 128 it is 0.20. A longer sample alone did not achieve this: the raw periodograms have standard deviations 1.03 and 1.00.'),
  'leakage': ('Move the strong cycle to j = 26 (ν = 26/256, a Fourier frequency). How many times above the background does the weak cycle stand in the raw periodogram?',
              '755 times, against 7.7 at j = 25.5: a cycle that fits a whole number of times in the sample does not leak, and the background falls to the level of the tiny noise. With the Hann taper the ratio is 4,176, almost the same for any j.'),
  'smoothing': ('▶ widens the second Daniell window. At m = 20 (L = 41), where is its peak and how high is it?',
                'At ν = 0.0625 (period 16.0), height 36.1, against the true peak at ν = 0.080 with height 64: the window is 41/512 = 0.08 wide, as wide as the peak frequency, so it flattens and shifts the peak. The mean squared error of log f̂ rises again, to 0.11 (0.09 at m = 10 and at m = 15).'),
  'sun-ci': ('Set m = 0 (the raw periodogram, df = 2). How many times f̂ is the upper bound of the 95% band at the peak?',
             '39.5 times (2/χ²₂(0.025) = 2/0.0506), and the lower bound is 0.27 × f̂: a single ordinate says very little about the height of the spectrum. With m = 2 the multipliers are 0.49 and 3.08, with m = 8 they are 0.65 and 1.72.'),
  'gdp-series': ('Set λ = 100. Does the standard deviation of the US HP cycle rise or fall from 1.58?',
                 'It falls, to 1.03: a smaller λ lets the trend follow the data more closely, so the cycle keeps only the faster fluctuations; with λ = 25,600 it rises to 2.23. The growth rates do not depend on λ (standard deviation 0.93).'),
  'gdp-spectra': ('Set m = 0 (the raw periodograms). At what period is the highest ordinate of the US HP cycle?',
                  'At 26.5 quarters (6.6 years), against 24.3 with m = 2: a raw ordinate is noisy, and the smoothed peak can move by one Fourier frequency (T = 292: 292/11 = 26.5, 292/12 = 24.3). The peak of US growth stays at 8.8 quarters.'),
  'load': ('Choose 2026 (January–June, 4,344 hours). At what period is the highest Welch ordinate with 8-week segments?',
           'At 12 hours, ahead of 24 hours (as in 2025); over 2022–2026 the 24-hour ordinate is the highest. The daily profile with two peaks, in the morning and in the evening, puts much power into the 12-hour harmonic, and the order of the two can change from one year to the next.'),
  'long-memory': ('Set the exponent to 0.45 (m = 52 frequencies, periods above 129 days). What is d̂ for the absolute returns?',
                  '0.38, against 0.56 with m = 307, with standard error 0.089; with exponent 0.80 (m = 1,152) it is 0.41. The estimate depends on the bandwidth: few frequencies give a large standard error, many frequencies bring in short-memory effects. For the returns d̂ stays between −0.04 and 0.14.'),
  'filters': ('▶ moves the marked period. At 4 quarters, what is the squared gain of the seasonal difference 1 − L⁴?',
              '0: 4 sin²(4π/4) = 4 sin²(π) = 0. The seasonal difference removes the yearly cycle of quarterly data (and its 2-quarter harmonic), while the first difference amplifies it by 2. At 8 quarters the seasonal difference has its largest gain, 4.'),
  'coherence': ('Choose segments of 48 months (K = 17). What happens to the 5% significance threshold?',
                'It falls from 0.35 to 0.17 (1 − 0.05<sup>1/16</sup>): more segments make the smoothed coherence more precise, so a smaller value is already significant. The mean coherence in the 18–96-month band rises to 0.90, and the phase at the 48-month cycle gives a lag of 2.1 months. With 144-month segments (K = 5) the threshold is 0.53.'),
  'wavelet': ('Choose ω₀ = 10 (more oscillations inside the wavelet). How does the mean 8–14-year power in 1795–1830 change?',
              'It rises from 2.0 to 2.5, while the mean over the whole sample goes from 5.7 to 5.9: a longer wavelet averages over more years, so the weak minimum around 1800 is blurred with its neighbours. The year of the maximum stays around 1956.'),
  'wx-cycle': ('Set φ = 1.5 radians, with A = 2. What are a and b?',
               'a = 2 cos 1.5 = 0.14 and b = −2 sin 1.5 = −1.99: with a phase close to π/2 the cycle is almost a pure sine with a negative sign. Whatever the phase, a² + b² = 4 = A².'),
  'wx-dft': ('▶ moves x₃. With x = (4, 2, 4, 2), where does the variance sit?',
             'All at ν = 1/2: the deviations (1, −1, 1, −1) alternate, so |d(1/2)|² = (−4)²/4 = 4 and |d(1/4)|² = |d(3/4)|² = 0; Parseval: 4 = 1 + 1 + 1 + 1. The series has period 2.'),
  'wx-alias': ('Set the period of the true cycle to 2.5 months, with quarterly observations (Δ = 3). What period does the cycle appear to have?',
               '15 months: 3/2.5 = 1.2 cycles per quarter, which folds to |1.2 − 1| = 0.2, i.e. 5 quarters. A cycle faster than the observations appears as a slower, false cycle.'),
  'wx-arma': ('Set φ = −0.6. What are f(0) and f(1/2)?',
              'f(0) = 1/1.6² = 0.39 and f(1/2) = 1/0.4² = 6.25: the mirror image of φ = 0.6. A negative φ makes the series alternate, so the power moves to the highest frequencies; the variance stays 1.5625.'),
  'wx-ar2': ('▶ moves φ₂ towards −0.95, with φ₁ = 1.5. Does the peak move to a longer or a shorter period?',
             'To a shorter one: 9.1 observations at φ₂ = −0.95 (12.4 at −0.75 and 20.2 at −0.65), and the peak becomes much higher (981 against 64). A more negative φ₂ brings the complex roots closer to the unit circle (their modulus is √−φ₂) and speeds up the oscillation of the ACF.'),
  't-sun': ('Widen the band to 9–20 years. What share of the variance does it hold?',
            '63%, against 60% for 9–13 years: the periods between 13 and 20 years add only 3 percentage points. Almost all the power of the solar cycle lies between 9 and 13 years; with the band 5–20 years the share reaches 73%, because the harmonic of about 5.5 years comes in as well.'),
  'x-prop': ('Set c = 1 (the true value f itself). What share of the white-noise ordinates exceeds f?',
             '37.6% for T = 2048, close to e<sup>−1</sup> = 36.8%, not 50%: the exponential distribution has mean 1 but median ln 2 = 0.69, so most ordinates lie below f and a few large values pull the mean up. For c = 3 the share is 5.1% (theory 5.0%).'),
  'wx-daniell': ('Set m = 0. At what period is the peak?',
                 'At 11.0 years (j = 28), the highest raw ordinate; with m = 2 the smoothed peak moves to 10.3 years (j = 30), because the mean of the five ordinates around j = 30 is the largest. With m = 4 the peak returns to 11.0 years: the position of the smoothed peak varies by one or two Fourier frequencies.'),
  't-smooth': ('▶ steps through m = 0, 2, …, 30. At which m is the mean squared error of log f̂ smallest?',
               'At m = 12 (0.089), very close to m = 10 (0.092); at m = 30 the error climbs to 0.22. Up to m = 12 the variance falls, then the bias dominates: the height of the peak drops from 64 to 44.7 at m = 12 and to 31.5 at m = 30.'),
  'wx-band': ('Set L = 21. What is the upper multiplier of the 95% band?',
              '1.62 (and the lower one 0.68), against 3.08 and 0.49 for L = 5: the band narrows roughly like 1/√L. It stays asymmetric, wider upwards, because the chi-square distribution has its tail to the right.'),
  't-gdp': ('Set the longest period of the band to 48 quarters. What share of the variance of the US HP cycle falls in the band?',
            '90%, against 80% for 6–32 quarters: the US HP cycle still has much power between 8 and 12 years. For Romanian growth the share hardly changes (15.6% in both cases), because its variance sits at high frequencies.'),
  't-load': ('Set the half-width to 1. How much does the share of the daily cycle and its harmonics fall?',
             'Very little: from 33.7% to 33.4%. The daily cycle is almost perfectly periodic, so its power sits in a few very narrow ordinates at 24/k hours; the weekly cycle falls more, from 15.7% to 14.9%: its power is spread over more neighbouring ordinates.'),
  'wx-hp': ('Choose λ = 6400. At what period does the HP cycle filter keep half of the power?',
            'At 45.0 quarters (11.3 years), against 31.8 quarters for λ = 1600: a 4 times larger λ moves the cut-off to slower cycles, by a factor of about 4<sup>1/4</sup> = 1.41. At 32 quarters G² rises from 0.49 to 0.82.'),
  't-filt': ('Choose λ = 100. What is the squared gain of the HP cycle filter at 32 quarters?',
             '0.017, against 0.49 for λ = 1600: with λ = 100 the trend follows almost every 8-year cycle and the HP cycle keeps only the short periods (half power at 15.8 quarters). At 8 quarters the squared gain is still 0.94.'),
 },
}

# Ce credeți? / What do you think?: (index of the correct option, [(RO, EN), ...]); keys checked against the page (node)
PREDICT = {
 'fourier': (0, [('119,2: de patru ori mai înalt (A²T/4 = 120).', '119.2: four times higher (A²T/4 = 120).'), ('59,2: de două ori mai înalt, ca amplitudinea.', '59.2: twice as high, like the amplitude.'),
                 ('29,6: înălțimea depinde doar de frecvență.', '29.6: the height depends only on the frequency.')]),
 'aliasing': (0, [('O constantă: ciclul dispare.', 'A constant: the cycle disappears.'), ('Un ciclu de 4 luni: o observație pe ciclu ajunge.', 'A 4-month cycle: one observation per cycle is enough.'),
                  ('Un ciclu de 12 luni, ca la observațiile trimestriale.', 'A 12-month cycle, as with quarterly observations.')]),
 'spectra': (0, [('58%.', '58%.'), ('20%: banda acoperă o cincime din frecvențe.', '20%: the band covers a fifth of the frequencies.'), ('100%: un AR(1) cu φ pozitiv are toată puterea la frecvențe joase.', '100%: an AR(1) with a positive φ has all its power at low frequencies.')]),
 'sunspots': (0, [('10,9 ani.', '10.9 years.'), ('Nu există un vîrf clar: 109 ani sînt prea puțini.', 'There is no clear peak: 109 years are too few.'), ('54,5 ani: într-un eșantion scurt domină schimbările lente.', '54.5 years: in a short sample the slow changes dominate.')]),
 'noise': (0, [('0,21.', '0.21.'), ('1,00: netezirea nu schimbă varianța ordonatelor.', '1.00: smoothing does not change the variance of the ordinates.'), ('0,05 = 1/21.', '0.05 = 1/21.')]),
 'leakage': (0, [('755.', '755.'), ('7,7, ca la j = 25,5: scurgerea nu depinde de frecvență.', '7.7, as at j = 25.5: leakage does not depend on the frequency.'), ('4 230, ca după fereastra Hann.', '4,230, as after the Hann taper.')]),
 'smoothing': (0, [('La ν = 0,0625, înălțimea 36.', 'At ν = 0.0625, height 36.'), ('La ν = 0,080, înălțimea 64: mai multă netezire apropie mereu estimația de adevăr.', 'At ν = 0.080, height 64: more smoothing always brings the estimate closer to the truth.'),
                   ('La ν = 0,102, ca periodograma brută.', 'At ν = 0.102, as the raw periodogram.')]),
 'sun-ci': (0, [('39,5.', '39.5.'), ('3,08, ca pentru m = 2: banda nu depinde de m.', '3.08, as for m = 2: the band does not depend on m.'), ('2: dublul estimației, pentru 2 grade de libertate.', '2: twice the estimate, for 2 degrees of freedom.')]),
 'gdp-series': (0, [('Scade, la 1,03.', 'It falls, to 1.03.'), ('Crește: un λ mai mic dă un ciclu mai puternic.', 'It rises: a smaller λ gives a stronger cycle.'), ('Rămîne 1,58: λ schimbă doar trendul.', 'It stays at 1.58: λ changes only the trend.')]),
 'gdp-spectra': (0, [('26,5 trimestre.', '26.5 quarters.'), ('24,3 trimestre: netezirea nu mută vîrful.', '24.3 quarters: smoothing does not move the peak.'), ('8,8 trimestre, ca la creștere.', '8.8 quarters, as for growth.')]),
 'load': (0, [('12 ore.', '12 hours.'), ('24 de ore, ca pe tot eșantionul: ordinea vîrfurilor nu depinde de an.', '24 hours, as over the whole sample: the order of the peaks does not depend on the year.'),
              ('168 de ore: iarna domină ciclul săptămînal.', '168 hours: in winter the weekly cycle dominates.')]),
 'long-memory': (0, [('0,38.', '0.38.'), ('0,56: d nu depinde de numărul de frecvențe.', '0.56: d does not depend on the number of frequencies.'), ('0: cu puține frecvențe memoria lungă nu se vede.', '0: with few frequencies long memory cannot be seen.')]),
 'filters': (0, [('0.', '0.'), ('2, ca diferența de ordinul întîi.', '2, as the first difference.'), ('4: diferența sezonieră amplifică ciclul anual.', '4: the seasonal difference amplifies the yearly cycle.')]),
 'coherence': (0, [('Scade, la 0,17.', 'It falls, to 0.17.'), ('Rămîne 0,35: pragul nu depinde de segmente.', 'It stays at 0.35: the threshold does not depend on the segments.'), ('Crește: segmentele scurte sînt mai zgomotoase.', 'It rises: short segments are noisier.')]),
 'wavelet': (0, [('Crește, la 2,5.', 'It rises, to 2.5.'), ('Rămîne 2,0: ω₀ nu schimbă puterea.', 'It stays at 2.0: ω₀ does not change the power.'), ('Scade spre 0: o undă mai lungă arată mai clar minimul.', 'It falls towards 0: a longer wavelet shows the minimum more clearly.')]),
 'wx-cycle': (0, [('a = 0,14, b = −1,99.', 'a = 0.14, b = −1.99.'), ('a = 2, b = 0: faza nu schimbă a și b.', 'a = 2, b = 0: the phase does not change a and b.'), ('a = 1,5, b = 2.', 'a = 1.5, b = 2.')]),
 'wx-dft': (0, [('Toată la ν = 1/2.', 'All at ν = 1/2.'), ('Toată la ν = 1/4, ca pentru x = (4, 2, 0, 2).', 'All at ν = 1/4, as for x = (4, 2, 0, 2).'), ('Împărțită egal între cele trei frecvențe.', 'Split equally over the three frequencies.')]),
 'wx-alias': (0, [('15 luni.', '15 months.'), ('2,5 luni: perioada adevărată.', '2.5 months: the true period.'), ('7,5 luni = 3 × 2,5.', '7.5 months = 3 × 2.5.')]),
 'wx-arma': (0, [('f(0) = 0,39, f(1/2) = 6,25.', 'f(0) = 0.39, f(1/2) = 6.25.'), ('f(0) = 6,25, f(1/2) = 0,39: semnul lui φ nu contează.', 'f(0) = 6.25, f(1/2) = 0.39: the sign of φ does not matter.'), ('Ambele negative.', 'Both negative.')]),
 'wx-ar2': (0, [('Spre o perioadă mai scurtă (9,1).', 'To a shorter period (9.1).'), ('Spre o perioadă mai lungă: un φ₂ mai negativ încetinește ciclul.', 'To a longer period: a more negative φ₂ slows the cycle.'), ('Nu se mută: perioada depinde doar de φ₁.', 'It does not move: the period depends only on φ₁.')]),
 't-sun': (0, [('63%.', '63%.'), ('100%: toată varianța este la ciclul de 11 ani.', '100%: all the variance is at the 11-year cycle.'), ('Aproape dublu, 120%: banda e aproape de două ori mai lată.', 'Almost double, 120%: the band is almost twice as wide.')]),
 'x-prop': (0, [('Aproximativ 37%.', 'About 37%.'), ('50%: f este media, deci jumătate din ordonate sînt peste ea.', '50%: f is the mean, so half of the ordinates lie above it.'), ('5%, ca pentru pragul 3.', '5%, as for the threshold 3.')]),
 'wx-daniell': (0, [('11,0 ani.', '11.0 years.'), ('10,3 ani, ca pentru m = 2.', '10.3 years, as for m = 2.'), ('Nu există un vîrf fără netezire.', 'There is no peak without smoothing.')]),
 't-smooth': (0, [('m = 12.', 'm = 12.'), ('m = 30: mai multă netezire înseamnă mereu o eroare mai mică.', 'm = 30: more smoothing always means a smaller error.'), ('m = 0: periodograma brută este nedistorsionată.', 'm = 0: the raw periodogram is unbiased.')]),
 'wx-band': (0, [('1,62.', '1.62.'), ('3,08, ca pentru L = 5.', '3.08, as for L = 5.'), ('0,15 = 3,08/21: banda se îngustează cu 1/L.', '0.15 = 3.08/21: the band narrows like 1/L.')]),
 't-gdp': (0, [('90%.', '90%.'), ('80%, ca pentru 6–32 de trimestre.', '80%, as for 6–32 quarters.'), ('100%: ciclul HP nu are perioade de peste 12 ani.', '100%: the HP cycle has no periods above 12 years.')]),
 't-load': (0, [('Foarte puțin: la 33,4%.', 'Very little: to 33.4%.'), ('La jumătate: banda are jumătate din ordonate.', 'To half: the band has half the ordinates.'), ('La 0: o singură ordonată nu poate purta un ciclu.', 'To 0: a single ordinate cannot carry a cycle.')]),
 'wx-hp': (0, [('45,0 trimestre.', '45.0 quarters.'), ('31,8 trimestre: λ nu mută pragul.', '31.8 quarters: λ does not move the cut-off.'), ('127 de trimestre = 4 × 31,8.', '127 quarters = 4 × 31.8.')]),
 't-filt': (0, [('0,017.', '0.017.'), ('0,49, ca pentru λ = 1600.', '0.49, as for λ = 1600.'), ('0,99: un λ mic păstrează tot ciclul.', '0.99: a small λ keeps the whole cycle.')]),
}
VEIL_CHARTS = set()

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slides „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('Niște date săptămînale au un vîrf la ν = 0,0192. Care este perioada?', 'Weekly data show a peak at ν = 0.0192. What is the period?'), key=1, opts=[
  (('0,0192 săptămîni.', '0.0192 weeks.'), ('ν este frecvența (cicluri pe săptămînă); perioada este 1/ν.', 'ν is the frequency (cycles per week); the period is 1/ν.')),
  (('Aproximativ 52 de săptămîni: un ciclu anual.', 'About 52 weeks: an annual cycle.'), ('1/0,0192 ≈ 52 de săptămîni.', '1/0.0192 ≈ 52 weeks.')),
  (('Aproximativ 2 săptămîni.', 'About 2 weeks.'), ('2 săptămîni ar însemna ν = 1/2, frecvența Nyquist.', '2 weeks would mean ν = 1/2, the Nyquist frequency.'))]),
 dict(q=('Cît sînt f(0) și f(1/2) pentru un AR(1) cu φ = −0,5 și σ² = 1?', 'What are f(0) and f(1/2) of an AR(1) with φ = −0.5 and σ² = 1?'), key=0, opts=[
  (('f(0) = 0,44 și f(1/2) = 4.', 'f(0) = 0.44 and f(1/2) = 4.'), ('f(0) = 1/1,5² = 0,44 și f(1/2) = 1/0,5² = 4: puterea este la frecvențele înalte.', 'f(0) = 1/1.5² = 0.44 and f(1/2) = 1/0.5² = 4: the power sits at high frequencies.')),
  (('f(0) = 4 și f(1/2) = 0,44.', 'f(0) = 4 and f(1/2) = 0.44.'), ('Aceasta este situația pentru φ = +0,5; semnul lui φ inversează spectrul.', 'This is the case φ = +0.5; the sign of φ mirrors the spectrum.')),
  (('Ambele 1,33 = 1/(1 − φ²).', 'Both 1.33 = 1/(1 − φ²).'), ('1/(1 − φ²) este varianța, aria sub f, nu valoarea lui f la o frecvență.', '1/(1 − φ²) is the variance, the area under f, not the value of f at one frequency.'))]),
 dict(q=('O serie lunară observată doar o dată la două luni are un ciclu de 3 luni. Unde apare acesta?', 'A monthly series observed only every second month has a 3-month cycle. Where does it appear?'), key=2, opts=[
  (('Tot la 3 luni.', 'Still at 3 months.'), ('2/3 cicluri pe observație este peste frecvența Nyquist 1/2, deci frecvența se pliază.', '2/3 cycles per observation is above the Nyquist frequency 1/2, so the frequency folds.')),
  (('Nu apare deloc: este mai rapid decît observațiile.', 'It does not appear at all: it is faster than the observations.'), ('Ciclul nu dispare, ci apare cu altă perioadă (aliasing).', 'The cycle does not vanish; it appears with another period (aliasing).')),
  (('Ca un ciclu fals de 3 observații, adică 6 luni.', 'As a false cycle of 3 observations, i.e. 6 months.'), ('ν = 2/3 se pliază la 1/3: perioada 3 observații = 6 luni.', 'ν = 2/3 folds to 1/3: a period of 3 observations = 6 months.'))]),
 dict(q=('Ce efect are prima diferență asupra unui ciclu de 40 de trimestre?', 'What does the first difference do to a cycle of 40 quarters?'), key=1, opts=[
  (('Îi dublează puterea.', 'It doubles its power.'), ('Cîștigul pătratic 2 apare la perioada de 4 trimestre, nu la 40.', 'The squared gain 2 occurs at a period of 4 quarters, not at 40.')),
  (('Îi înmulțește puterea cu 4 sin²(π/40) = 0,025: ciclul aproape dispare.', 'It multiplies its power by 4 sin²(π/40) = 0.025: the cycle almost disappears.'), ('Prima diferență are cîștigul pătratic 4 sin²(πν), mic la frecvențele joase.', 'The first difference has squared gain 4 sin²(πν), small at low frequencies.')),
  (('Nu o schimbă: diferențierea elimină doar trendul.', 'It leaves it unchanged: differencing removes only the trend.'), ('Diferențierea slăbește și ciclurile lente, nu doar trendul.', 'Differencing also weakens slow cycles, not only the trend.'))]),
 dict(q=('Periodograma log–log a unei serii are panta −0,6 în apropierea lui zero. Cît este d?', 'The log–log periodogram of a series has slope −0.6 near zero. What is d?'), key=0, opts=[
  (('d = 0,3: memorie lungă staționară.', 'd = 0.3: stationary long memory.'), ('Panta este −2d, deci d = 0,6/2 = 0,3.', 'The slope is −2d, so d = 0.6/2 = 0.3.')),
  (('d = 0,6: nestaționar.', 'd = 0.6: nonstationary.'), ('Panta este −2d, nu −d.', 'The slope is −2d, not −d.')),
  (('d = −0,6: antipersistență.', 'd = −0.6: antipersistence.'), ('O pantă negativă înseamnă putere care crește spre ν = 0, deci d este pozitiv.', 'A negative slope means power rising towards ν = 0, so d is positive.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {})) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch12-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] = TEXT_JS[lang].replace('\n};\n})();\n', '\n};\n') + ('window.CH12_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                                                                        + 'window.CH12_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n})();\n')


_predict_js()


if __name__ == '__main__':
    import interactive_common
    page_spec = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA', 'TEXT_JS', 'PREDICT', 'TICKET')})
    page_spec.widgets = lambda lang: {a: v for a, v in widgets_all(lang).items() if a in {b for b, _, _ in BLOCKS}}
    interactive_common.build(page_spec)
    for lg in ('en', 'ro'):
        _post(lg)
