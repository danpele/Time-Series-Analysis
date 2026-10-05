r"""
build_chapter12.py -- Capitolul 12 (Analiză spectrală), EN + RO dintr-o singură sursă
====================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_12/ch12_numbers.json (generate_all_charts.py).
Nicio cifră nu este scrisă de mînă. Capitol de studiu individual: exemple rezolvate, slide-uri de interpretare,
recapitulare pe secțiuni și autoevaluare cu răspunsuri. Reface vechiul capitol 12 (analiză spectrală) la nivel de
licență, cu convenția de frecvență din Shumway și Stoffer (2017, cap. 4): cicluri pe observație.
Ieșire:
  EN/Courses/chapter12_spectral_analysis.tex
  RO/Cursuri/capitol12_analiza_spectrala.tex
Rulare:
  python3 Quantlets/Ch_12/generate_all_charts.py
  python3 latex/build_chapter12.py && python3 latex/tsa_build.py compile 12
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, photo   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch12_common import QLURL, REFS, T, bib, finalize, load, pv   # noqa: E402


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


N = load()
V = Values()
D = Deck(12, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\quantlet{{{folder.replace("_", chr(92) + "_")}}}{{\\qlurl{{{folder}}}}}'


def chart(title, fig, folder, bullets, h='0.6\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def interp(title, bullets, size='small'):
    D.frame(T(f'Interpreting {title[0]}', f'Interpretarea {title[1]}'), items(*bullets), size)


def solved(title, bullets, size='small'):
    D.frame(T(f'Worked example: {title[0]}', f'Exemplu rezolvat: {title[1]}'), items(*bullets), size)


PH = {
    'fourier': ('ch12_joseph_fourier.jpg', C + 'Joseph_Fourier.jpg',
                T('Engraving', 'Gravură') + ': A. F. B. Geille, ' + T('after', 'după') + ' J.-L. Boilly (1839--1840); '
                + T('public domain', 'domeniu public') + '; Wikimedia Commons'),
    'schuster': ('ch12_arthur_schuster.jpg', C + 'Arthur_Schuster.jpg',
                 T('Photo', 'Foto') + ': ' + T('unknown author', 'autor necunoscut') + ' (1900s); '
                 + T('public domain', 'domeniu public') + '; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


def two(left, right, wl='0.36', wr='0.62'):
    return (f'\\begin{{columns}}[T]\n\\begin{{column}}{{{wl}\\textwidth}}\n{left}\n\\end{{column}}\n'
            f'\\begin{{column}}{{{wr}\\textwidth}}\n{right}\n\\end{{column}}\n\\end{{columns}}')


# =============================================================================
# CIFRE
# =============================================================================
P = V.put
EX = N['ex']
P('ex.ar1.f0', EX['ar1']['f0'], 2)
P('ex.ar1.f5', EX['ar1']['f5'], 2)
P('ex.ar1.ratio', EX['ar1']['f0'] / EX['ar1']['f5'], 0)
P('ex.ma1.f0', EX['ma1']['f0'], 2)
P('ex.ma1.f5', EX['ma1']['f5'], 2)
P('ex.ar2.cos', EX['ar2']['cos'], 3)
P('ex.ar2.nu', EX['ar2']['nu'], 4)
P('ex.ar2.per', EX['ar2']['period'], 1)
P('ex.ar2.fp', EX['ar2']['fpeak'], 0)
P('ex.ar2.f0', EX['ar2']['f0'], 0)
P('ex.ci.qlo', EX['ci']['q_lo'], 2)
P('ex.ci.qhi', EX['ci']['q_hi'], 2)
P('ex.ci.lo', EX['ci']['lo'], 2)
P('ex.ci.hi', EX['ci']['hi'], 2)

F = N['fourier']
P('fo.I12', F['I12'], 1)
P('fo.I4', F['I4'], 1)
P('fo.t12', F['theory12'], 0)
P('fo.t4', F['theory4'], 0)
P('fo.share', 100 * F['share'], 0)

S = N['spectra']
P('sp.ar2.var', S['AR(2), phi = (1.5, -0.75)']['var'], 2)
P('sp.ar1.var', S['AR(1), phi = 0.6']['var'], 4)
P('sp.arma.f0', S['ARMA(1,1), phi = 0.6, theta = 0.4']['f0'], 2)

SU = N['sun']
V.raw('su.T', str(SU['T']))
V.raw('su.j', str(SU['j']))
P('su.nu', SU['nu'], 4)
P('su.per', SU['period'], 1)
P('su.share', 100 * SU['share_band'], 0)
P('su.p2', SU['top3'][1], 1)
P('su.p3', SU['top3'][2], 1)
P('su.g', SU['fisher']['g'], 3)
V.raw('su.m', str(SU['fisher']['m']))
V.raw('su.pv', pv(SU['fisher']['p']))
P('su.gcrit', 1 / SU['fisher']['m'], 4)
P('su.mean', SU['mean'], 1)
P('su.max', SU['max'], 1)
V.raw('su.maxy', str(SU['max_year']))

IC = N['incons']
for k in ('128', '2048'):
    P(f'ic.{k}.m', IC[k]['mean'], 2)
    P(f'ic.{k}.sd', IC[k]['sd'], 2)
    P(f'ic.{k}.max', IC[k]['max'], 1)
P('ic.q95', IC['chi2_q95'], 2)

LK = N['leak']
P('lk.nu1', LK['nu1'], 4)
P('lk.rr', LK['raw_ratio'], 1)
V.int('lk.tr', round(LK['tap_ratio']))

SM = N['smooth']
P('sm.tp', SM['true_peak'], 3)
P('sm.rp', SM['raw_peak'], 3)
P('sm.d2p', SM['d2']['peak'], 3)
P('sm.d10p', SM['d10']['peak'], 3)
P('sm.wp', SM['welch']['peak'], 3)
P('sm.mr', SM['mse_raw'], 2)
P('sm.m2', SM['d2']['mse'], 2)
P('sm.m10', SM['d10']['mse'], 2)
P('sm.mw', SM['welch']['mse'], 2)
V.raw('sm.K', str(SM['welch']['K']))
P('sm.ftp', SM['ftrue_peak'], 0)
P('sm.f10', SM['d10']['fpeak'], 0)

SC = N['sunci']
P('sc.per', SC['period'], 1)
P('sc.B', SC['B'], 3)
V.int('sc.f', round(SC['f']))
V.int('sc.lo', round(SC['lo']))
V.int('sc.hi', round(SC['hi']))

G = N['gdp']
for k in ('US', 'RO'):
    V.raw(f'g.{k}.T', str(G[k]['T']))
    P(f'g.{k}.sd', G[k]['sd'], 2)
    P(f'g.{k}.csd', G[k]['cyc_sd'], 2)
    P(f'g.{k}.mean', G[k]['mean'], 2)
GS = N['gdpspec']
for k in ('US', 'RO'):
    P(f'gs.{k}.g.bc', 100 * GS[f'{k}.growth']['share_bc'], 0)
    P(f'gs.{k}.c.bc', 100 * GS[f'{k}.HP cycle']['share_bc'], 0)
    P(f'gs.{k}.c.low', 100 * GS[f'{k}.HP cycle']['share_low'], 0)
P('gs.US.c.per', GS['US.HP cycle']['peak_period'], 1)
P('gs.US.c.yrs', GS['US.HP cycle']['peak_period'] / 4, 1)
P('gs.US.g.per', GS['US.growth']['peak_period'], 1)

LO = N['load']
V.int('lo.T', LO['T'])
V.int('lo.mean', round(LO['mean']))
P('lo.daily', 100 * LO['share_daily'], 0)
P('lo.weekly', 100 * LO['share_weekly'], 0)
P('lo.long', 100 * LO['share_long'], 0)
P('lo.p168', LO['top'][4], 1)
P('lo.p84', LO['top'][5], 1)
P('lo.pyr', LO['top'][2] / 24, 0)

LM = N['lm']
V.int('lm.T', LM['T'])
for k in ('r', 'abs'):
    P(f'lm.{k}.d', LM[k]['d'], 2)
P('lm.se', LM['r']['se'], 3)
V.raw('lm.m', str(LM['r']['m']))
P('lm.per', 1 / LM['r']['nu_m'], 0)

FI = N['filt']
P('fi.d40', FI['d1_40'], 3)
P('fi.d4', FI['d1_4'], 0)
P('fi.h8', FI['hp_8'], 3)
P('fi.h32', FI['hp_32'], 2)
P('fi.h40', FI['hp_40'], 2)
P('fi.hhalf', FI['hp_half_period'], 0)

CO = N['coh']
V.raw('co.T', str(CO['T']))
V.raw('co.K', str(CO['K']))
P('co.thr', CO['thr'], 2)
P('co.bc', CO['coh_bc'], 2)
P('co.bcmax', CO['coh_bc_max'], 2)
P('co.hi', CO['coh_hi'], 2)
P('co.ph', CO['phase48'], 2)
P('co.lag', CO['lag48'], 1)
P('co.r0', CO['corr0'], 2)

WV = N['wav']
P('wv.weak', WV['weak_1795_1830'], 1)
P('wv.all', WV['all'], 1)
V.raw('wv.maxy', str(WV['max_year']))

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("The chapter's question and route", 'Întrebarea capitolului și traseul'), items(
    (T('\\textbf{Question}: which cycles does a series contain, and how much of its variance does each cycle explain?',
       '\\textbf{Întrebarea}: ce cicluri conține o serie și cît din varianța ei explică fiecare ciclu?'),
     [T('examples: the 11-year sunspot cycle, the business cycle, the daily and weekly rhythm of electricity demand',
        'exemple: ciclul de 11 ani al petelor solare, ciclul economic, ritmul zilnic și săptămînal al consumului de energie electrică')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('cycles, frequencies and the discrete Fourier transform', 'cicluri, frecvențe și transformata Fourier discretă'),
      T('the spectral density of white noise and of ARMA models', 'densitatea spectrală a zgomotului alb și a modelelor ARMA'),
      T('the periodogram, its weaknesses, and how to smooth it', 'periodograma, slăbiciunile ei și netezirea ei'),
      T('cycles in data; long memory; filters; two series; wavelets as a pointer', 'cicluri în date; memoria lungă; filtre; două serii; o trimitere spre wavelets')]),
    T('Self-study chapter: worked examples, interpretation slides and a recap after each section; self-assessment with answers at the end',
      'Capitol de studiu individual: exemple rezolvate, slide-uri de interpretare și o recapitulare după fiecare secțiune; autoevaluare cu răspunsuri la final')))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Convert between frequency, period and angular frequency, and find the Fourier frequencies of a sample', 'Treceți de la frecvență la perioadă și la frecvența unghiulară și găsiți frecvențele Fourier ale unui eșantion'),
    T('Compute the spectral density of white noise, AR(1), MA(1) and AR(2) models and read its shape', 'Calculați densitatea spectrală a zgomotului alb și a modelelor AR(1), MA(1) și AR(2) și interpretați forma ei'),
    T('Compute a periodogram by hand for a short series and explain why the raw periodogram is not consistent', 'Calculați de mînă periodograma unei serii scurte și explicați de ce periodograma brută nu este consistentă'),
    T('Smooth the periodogram (Daniell, Bartlett, Welch), taper the data and build chi-square confidence bands', 'Netezați periodograma (Daniell, Bartlett, Welch), aplicați o fereastră de atenuare și construiți benzi de încredere chi-pătrat'),
    T('Identify cycles in sunspots, GDP and electricity load; read coherence and phase between two series', 'Identificați cicluri în petele solare, în PIB și în consumul de energie electrică; interpretați coerența și faza dintre două serii')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Main reference for this chapter: \\refSS, Chapter 4 (spectral analysis and filtering)', 'Referința principală a capitolului: \\refSS, capitolul 4 (analiză spectrală și filtrare)'),
     [T('theory: \\refBDtm, Chapters 4 and 10; \\refHamTS, Chapter 6; applications in Python: \\refHP', 'teorie: \\refBDtm, capitolele 4 și 10; \\refHamTS, capitolul 6; aplicații în Python: \\refHP'),
      T('\\refSS\\ measure frequency in cycles per observation; Chapter 8 used the angular frequency $\\lambda = 2\\pi\\nu$', '\\refSS\\ măsoară frecvența în cicluri pe observație; în Capitolul 8 am folosit frecvența unghiulară $\\lambda = 2\\pi\\nu$')]),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_12}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_12}'),
     [T('the periodogram, the Daniell smoother and the Morlet wavelet are written out in \\texttt{numpy}; Welch and coherence come from \\texttt{scipy.signal}',
        'periodograma, netezirea Daniell și wavelet-ul Morlet sînt scrise explicit în \\texttt{numpy}; Welch și coerența provin din \\texttt{scipy.signal}')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter12_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter12_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. CICLURI ȘI FRECVENȚE
# =============================================================================
D.section('Cycles and frequencies', 'Cicluri și frecvențe')

D.frame(T('Joseph Fourier and the idea of the chapter', 'Joseph Fourier și ideea capitolului'), two(
    ph('fourier', 'J. B. J. Fourier (1768--1830)', h='0.5\\textheight'),
    items((T('Fourier (1807, heat conduction): a function can be written as a sum of sines and cosines', 'Fourier (1807, conducția căldurii): o funcție se poate scrie ca sumă de sinusuri și cosinusuri'),
           [T('each term is a pure cycle with its own frequency and amplitude', 'fiecare termen este un ciclu pur, cu frecvența și amplitudinea lui')]),
          (T('Applied to a time series: how much variance sits at each frequency?', 'Aplicat unei serii de timp: cîtă varianță se află la fiecare frecvență?'),
           [T('the \\textbf{time domain} (Chapters 1--8) describes the series through its autocorrelations', '\\textbf{domeniul timpului} (Capitolele 1--8) descrie seria prin autocorelațiile ei'),
            T('the \\textbf{frequency domain} describes the same information through the spectrum', '\\textbf{domeniul frecvenței} descrie aceeași informație prin spectru')]),
          T('The two views are equivalent; the spectrum makes cycles visible at a glance', 'Cele două perspective sînt echivalente; spectrul face ciclurile vizibile dintr-o privire'))))

D.frame(T('A cycle: amplitude, frequency, phase', 'Un ciclu: amplitudine, frecvență, fază'), items(
    (T('A pure cycle: $x_t = A\\cos(2\\pi\\nu t + \\varphi)$, $t = 1, 2, \\ldots$', 'Un ciclu pur: $x_t = A\\cos(2\\pi\\nu t + \\varphi)$, $t = 1, 2, \\ldots$'),
     [T('$A$: \\textbf{amplitude} (height of the wave); $\\varphi$: \\textbf{phase} (where the wave starts)', '$A$: \\textbf{amplitudinea} (înălțimea undei); $\\varphi$: \\textbf{faza} (punctul de pornire al undei)'),
      T('$\\nu$: \\textbf{frequency}, in cycles per observation; the \\textbf{period} $1/\\nu$ is the number of observations per cycle', '$\\nu$: \\textbf{frecvența}, în cicluri pe observație; \\textbf{perioada} $1/\\nu$ este numărul de observații dintr-un ciclu'),
      T('angular frequency $\\omega = 2\\pi\\nu$, in radians per observation', 'frecvența unghiulară $\\omega = 2\\pi\\nu$, în radiani pe observație')]),
    (T('Example: monthly data with an annual cycle: $\\nu = 1/12$ cycles per month, period 12 months', 'Exemplu: date lunare cu un ciclu anual: $\\nu = 1/12$ cicluri pe lună, perioada 12 luni'),
     [T('hourly data with a daily cycle: $\\nu = 1/24$; quarterly data with a 6-year cycle: $\\nu = 1/24$ as well', 'date orare cu un ciclu zilnic: $\\nu = 1/24$; date trimestriale cu un ciclu de 6 ani: tot $\\nu = 1/24$')]),
    (T('Equivalent form: $A\\cos(2\\pi\\nu t + \\varphi) = a\\cos(2\\pi\\nu t) + b\\sin(2\\pi\\nu t)$', 'Forma echivalentă: $A\\cos(2\\pi\\nu t + \\varphi) = a\\cos(2\\pi\\nu t) + b\\sin(2\\pi\\nu t)$'),
     [T('with $a = A\\cos\\varphi$, $b = -A\\sin\\varphi$ and $A^2 = a^2 + b^2$: linear in $a$ and $b$, so it can be fitted by OLS', 'cu $a = A\\cos\\varphi$, $b = -A\\sin\\varphi$ și $A^2 = a^2 + b^2$: liniară în $a$ și $b$, deci se poate estima prin OLS')]),
    T('The highest frequency visible in discrete data is $\\nu = 1/2$ (one cycle every two observations): the \\textbf{Nyquist frequency}', 'Cea mai mare frecvență vizibilă în date discrete este $\\nu = 1/2$ (un ciclu la două observații): \\textbf{frecvența Nyquist}')))

chart(T('Two cycles plus noise', 'Două cicluri plus zgomot'), 'tsa_ch12_fourier', 'TSA_ch12_fourier', [
    T('Simulated series, $T = 120$: $x_t = 2\\cos(2\\pi t/12) + \\cos(2\\pi t/4 + 1) + \\varepsilon_t$, $\\varepsilon_t \\sim N(0, 0.49)$; right: its periodogram (defined in Section 3)',
      'Serie simulată, $T = 120$: $x_t = 2\\cos(2\\pi t/12) + \\cos(2\\pi t/4 + 1) + \\varepsilon_t$, $\\varepsilon_t \\sim N(0;\\ 0.49)$; dreapta: periodograma ei (definită în secțiunea 3)')],
    h='0.62\\textheight')

interp(('two cycles plus noise', 'celor două cicluri plus zgomot'), [
    (T('In the time domain the two cycles are hard to separate by eye', 'În domeniul timpului cele două cicluri sînt greu de separat cu ochiul liber'),
     [T('in the frequency domain they are two isolated spikes, at $\\nu = 1/12$ and $\\nu = 1/4$', 'în domeniul frecvenței ele sînt două vîrfuri izolate, la $\\nu = 1/12$ și $\\nu = 1/4$')]),
    (T('Height of the spikes: @{fo.I12} and @{fo.I4}; theory for a cycle of amplitude $A$ at a Fourier frequency: $A^2T/4$, i.e.\\ @{fo.t12} and @{fo.t4}', 'Înălțimea vîrfurilor: @{fo.I12} și @{fo.I4}; teoria, pentru un ciclu de amplitudine $A$ la o frecvență Fourier: $A^2T/4$, adică @{fo.t12} și @{fo.t4}'),
     [T('the spike grows with the squared amplitude: the larger cycle dominates', 'vîrful crește cu pătratul amplitudinii: ciclul mai mare domină')]),
    T('The two spikes hold @{fo.share}\\% of the total power; the noise spreads the rest evenly over all frequencies', 'Cele două vîrfuri concentrează @{fo.share}\\% din puterea totală; zgomotul împrăștie restul uniform pe toate frecvențele')])

D.frame(T('Fourier frequencies and the discrete Fourier transform', 'Frecvențele Fourier și transformata Fourier discretă'), items(
    (T('For a sample $x_1, \\ldots, x_T$, the \\textbf{Fourier frequencies} are $\\nu_j = j/T$, $j = 0, 1, \\ldots, \\lfloor T/2 \\rfloor$', 'Pentru un eșantion $x_1, \\ldots, x_T$, \\textbf{frecvențele Fourier} sînt $\\nu_j = j/T$, $j = 0, 1, \\ldots, \\lfloor T/2 \\rfloor$'),
     [T('cycles that fit an integer number of times ($j$ times) in the sample', 'ciclurile care încap de un număr întreg de ori ($j$ ori) în eșantion')]),
    (T('\\textbf{Discrete Fourier transform} (DFT): $d(\\nu_j) = T^{-1/2}\\sum_{t=1}^{T} x_t\\, e^{-2\\pi i \\nu_j t}$', '\\textbf{Transformata Fourier discretă} (DFT): $d(\\nu_j) = T^{-1/2}\\sum_{t=1}^{T} x_t\\, e^{-2\\pi i \\nu_j t}$'),
     [T('$e^{-i\\theta} = \\cos\\theta - i\\sin\\theta$: the DFT correlates the series with a cosine and a sine of frequency $\\nu_j$', '$e^{-i\\theta} = \\cos\\theta - i\\sin\\theta$: DFT corelează seria cu un cosinus și un sinus de frecvență $\\nu_j$'),
      T('the $T$ values $d(\\nu_j)$ contain exactly the same information as the $T$ data points', 'cele $T$ valori $d(\\nu_j)$ conțin exact aceeași informație ca cele $T$ observații')]),
    (T('\\textbf{Parseval}: $\\sum_{t=1}^{T}(x_t - \\bar x)^2 = \\sum_{j=1}^{T-1}|d(\\nu_j)|^2$: the variance is split over frequencies', '\\textbf{Parseval}: $\\sum_{t=1}^{T}(x_t - \\bar x)^2 = \\sum_{j=1}^{T-1}|d(\\nu_j)|^2$: varianța se împarte pe frecvențe'),
     [T('the \\textbf{fast Fourier transform} (FFT) of \\refCT\\ computes all $d(\\nu_j)$ in $O(T\\log T)$ operations instead of $O(T^2)$', '\\textbf{transformata Fourier rapidă} (FFT) a lui \\refCT\\ calculează toate valorile $d(\\nu_j)$ în $O(T\\log T)$ operații în loc de $O(T^2)$')])))

solved(('the DFT of four numbers', 'DFT pentru patru numere'), [
    (T('Data: $x = (4, 2, 0, 2)$, $T = 4$; mean $\\bar x = 2$, deviations $y = (2, 0, -2, 0)$', 'Datele: $x = (4, 2, 0, 2)$, $T = 4$; media $\\bar x = 2$, abaterile $y = (2, 0, -2, 0)$'),
     [T('Fourier frequencies: $\\nu_1 = 1/4$ (period 4) and $\\nu_2 = 1/2$ (period 2)', 'Frecvențele Fourier: $\\nu_1 = 1/4$ (perioada 4) și $\\nu_2 = 1/2$ (perioada 2)')]),
    (T('At $\\nu = 1/4$: $e^{-2\\pi i t/4} = e^{-i\\pi t/2}$ equals $-i$ for $t = 1$ and $i$ for $t = 3$', 'La $\\nu = 1/4$: $e^{-2\\pi i t/4} = e^{-i\\pi t/2}$ este $-i$ pentru $t = 1$ și $i$ pentru $t = 3$'),
     [T('$\\sum_t y_t e^{-i\\pi t/2} = 2(-i) + (-2)(i) = -4i$, so $d(1/4) = -4i/\\sqrt{4} = -2i$ and $|d(1/4)|^2 = 4$', '$\\sum_t y_t e^{-i\\pi t/2} = 2(-i) + (-2)(i) = -4i$, deci $d(1/4) = -4i/\\sqrt{4} = -2i$ și $|d(1/4)|^2 = 4$')]),
    (T('At $\\nu = 1/2$: $e^{-i\\pi t} = (-1)^t$, so $\\sum_t y_t(-1)^t = -2 + 2 = 0$ and $|d(1/2)|^2 = 0$', 'La $\\nu = 1/2$: $e^{-i\\pi t} = (-1)^t$, deci $\\sum_t y_t(-1)^t = -2 + 2 = 0$ și $|d(1/2)|^2 = 0$'),
     [T('by symmetry $|d(3/4)|^2 = |d(1/4)|^2 = 4$', 'prin simetrie $|d(3/4)|^2 = |d(1/4)|^2 = 4$')]),
    T('Parseval: $4 + 0 + 4 = 8 = 2^2 + 0^2 + (-2)^2 + 0^2$; the whole variance ($8/4 = 2$) sits at period 4', 'Parseval: $4 + 0 + 4 = 8 = 2^2 + 0^2 + (-2)^2 + 0^2$; toată varianța ($8/4 = 2$) se află la perioada 4'),
    T('Check in Python: \\texttt{abs(np.fft.fft([2, 0, -2, 0]))**2 / 4} gives $(0, 4, 0, 4)$', 'Verificare în Python: \\texttt{abs(np.fft.fft([2, 0, -2, 0]))**2 / 4} dă $(0, 4, 0, 4)$')])

chart(T('Aliasing', 'Aliasing'), 'tsa_ch12_aliasing', 'TSA_ch12_aliasing', [
    T('A cycle of 4 months ($\\nu = 1/4$ per month) observed only once a quarter (every 3 months)', 'Un ciclu de 4 luni ($\\nu = 1/4$ pe lună), observat doar o dată pe trimestru (la fiecare 3 luni)')],
    h='0.6\\textheight')

interp(('aliasing', 'fenomenului de aliasing'), [
    (T('Per quarter, the true frequency is $3 \\times 1/4 = 3/4$ cycles, above the Nyquist frequency $1/2$', 'Pe trimestru, frecvența adevărată este $3 \\times 1/4 = 3/4$ cicluri, peste frecvența Nyquist $1/2$'),
     [T('it is folded back to $|3/4 - 1| = 1/4$ cycles per quarter: a period of 4 quarters, i.e.\\ 12 months', 'ea se pliază înapoi la $|3/4 - 1| = 1/4$ cicluri pe trimestru: o perioadă de 4 trimestre, adică 12 luni')]),
    (T('The quarterly points lie exactly on a 12-month wave: the data cannot tell the two cycles apart', 'Punctele trimestriale cad exact pe o undă de 12 luni: datele nu pot deosebi cele două cicluri'),
     [T('this is \\textbf{aliasing}: cycles faster than two observations appear as slower, false cycles', 'acesta este \\textbf{aliasing}: ciclurile mai rapide decît două observații apar ca cicluri mai lente, false')]),
    T('Practical rule: sample at least twice per cycle of interest; average (rather than pick) values when you aggregate to a lower frequency', 'Regula practică: eșantionați de cel puțin două ori pe ciclul de interes; la agregarea pe o frecvență mai mică, faceți media valorilor, nu alegeți una dintre ele')])

D.recap(('Cycles and frequencies', 'cicluri și frecvențe'), [
    T('A cycle has an amplitude, a frequency $\\nu$ (cycles per observation), a period $1/\\nu$ and a phase', 'Un ciclu are o amplitudine, o frecvență $\\nu$ (cicluri pe observație), o perioadă $1/\\nu$ și o fază'),
    T('The DFT at the Fourier frequencies $j/T$ rewrites the data as a sum of cycles; Parseval splits the variance by frequency', 'DFT la frecvențele Fourier $j/T$ rescrie datele ca sumă de cicluri; Parseval împarte varianța pe frecvențe'),
    T('Only frequencies up to $1/2$ are visible; faster cycles are aliased', 'Sînt vizibile doar frecvențele pînă la $1/2$; ciclurile mai rapide apar ca aliasuri')])

# =============================================================================
# 2. DENSITATEA SPECTRALĂ
# =============================================================================
D.section('The spectral density', 'Densitatea spectrală')

D.frame(T('From autocovariances to the spectrum', 'De la autocovarianțe la spectru'), items(
    (T('For a stationary series with autocovariances $\\gamma(h)$, $\\sum_h|\\gamma(h)| < \\infty$, the \\textbf{spectral density} is', 'Pentru o serie staționară cu autocovarianțele $\\gamma(h)$, $\\sum_h|\\gamma(h)| < \\infty$, \\textbf{densitatea spectrală} este'),
     [T('$f(\\nu) = \\sum_{h=-\\infty}^{\\infty}\\gamma(h)e^{-2\\pi i\\nu h} = \\gamma(0) + 2\\sum_{h=1}^{\\infty}\\gamma(h)\\cos(2\\pi\\nu h)$, $-1/2 \\le \\nu \\le 1/2$',
        '$f(\\nu) = \\sum_{h=-\\infty}^{\\infty}\\gamma(h)e^{-2\\pi i\\nu h} = \\gamma(0) + 2\\sum_{h=1}^{\\infty}\\gamma(h)\\cos(2\\pi\\nu h)$, $-1/2 \\le \\nu \\le 1/2$')]),
    (T('The inverse relation: $\\gamma(h) = \\int_{-1/2}^{1/2} f(\\nu)e^{2\\pi i\\nu h}\\,d\\nu$', 'Relația inversă: $\\gamma(h) = \\int_{-1/2}^{1/2} f(\\nu)e^{2\\pi i\\nu h}\\,d\\nu$'),
     [T('the pair $\\gamma \\leftrightarrow f$ is the \\textbf{Wiener--Khinchin} theorem: the ACF and the spectrum carry the same information', 'perechea $\\gamma \\leftrightarrow f$ este teorema \\textbf{Wiener--Khinchin}: ACF și spectrul conțin aceeași informație'),
      T('for $h = 0$: $\\mathrm{Var}(x_t) = \\gamma(0) = \\int_{-1/2}^{1/2} f(\\nu)\\,d\\nu$: the area under $f$ is the variance', 'pentru $h = 0$: $\\mathrm{Var}(x_t) = \\gamma(0) = \\int_{-1/2}^{1/2} f(\\nu)\\,d\\nu$: aria de sub $f$ este varianța')]),
    (T('Properties: $f(\\nu) \\ge 0$, $f(-\\nu) = f(\\nu)$; it is enough to draw $f$ on $[0, 1/2]$', 'Proprietăți: $f(\\nu) \\ge 0$, $f(-\\nu) = f(\\nu)$; este suficient să desenăm $f$ pe $[0, 1/2]$'),
     [T('$f(\\nu)\\,d\\nu$ is the share of the variance due to cycles with frequencies in $[\\nu, \\nu + d\\nu]$', '$f(\\nu)\\,d\\nu$ este partea din varianță datorată ciclurilor cu frecvențe în $[\\nu, \\nu + d\\nu]$')])))

D.frame(T('White noise, AR(1) and MA(1)', 'Zgomot alb, AR(1) și MA(1)'), items(
    (T('\\textbf{White noise}, variance $\\sigma^2$: $\\gamma(h) = 0$ for $h \\ne 0$, so $f(\\nu) = \\sigma^2$ for every $\\nu$', '\\textbf{Zgomot alb}, varianța $\\sigma^2$: $\\gamma(h) = 0$ pentru $h \\ne 0$, deci $f(\\nu) = \\sigma^2$ pentru orice $\\nu$'),
     [T('a flat spectrum: all frequencies contribute equally, like white light', 'un spectru plat: toate frecvențele contribuie în mod egal, ca lumina albă')]),
    (T('\\textbf{AR(1)} $x_t = \\phi x_{t-1} + w_t$: $f(\\nu) = \\dfrac{\\sigma^2}{1 - 2\\phi\\cos(2\\pi\\nu) + \\phi^2}$', '\\textbf{AR(1)} $x_t = \\phi x_{t-1} + w_t$: $f(\\nu) = \\dfrac{\\sigma^2}{1 - 2\\phi\\cos(2\\pi\\nu) + \\phi^2}$'),
     [T('$\\phi > 0$: power at low frequencies (smooth, persistent series); $\\phi < 0$: power at high frequencies (a zig-zag series)', '$\\phi > 0$: putere la frecvențele joase (serie netedă, persistentă); $\\phi < 0$: putere la frecvențele înalte (serie în zigzag)')]),
    (T('\\textbf{MA(1)} $x_t = w_t + \\theta w_{t-1}$: $f(\\nu) = \\sigma^2\\,[1 + 2\\theta\\cos(2\\pi\\nu) + \\theta^2]$', '\\textbf{MA(1)} $x_t = w_t + \\theta w_{t-1}$: $f(\\nu) = \\sigma^2\\,[1 + 2\\theta\\cos(2\\pi\\nu) + \\theta^2]$'),
     [T('directly from the definition: $\\gamma(0) = \\sigma^2(1 + \\theta^2)$, $\\gamma(1) = \\sigma^2\\theta$, so $f = \\gamma(0) + 2\\gamma(1)\\cos(2\\pi\\nu)$', 'direct din definiție: $\\gamma(0) = \\sigma^2(1 + \\theta^2)$, $\\gamma(1) = \\sigma^2\\theta$, deci $f = \\gamma(0) + 2\\gamma(1)\\cos(2\\pi\\nu)$')]),
    (T('\\textbf{ARMA}$(p,q)$, $\\phi(L)x_t = \\theta(L)w_t$: $f(\\nu) = \\sigma^2\\dfrac{|\\theta(e^{-2\\pi i\\nu})|^2}{|\\phi(e^{-2\\pi i\\nu})|^2}$', '\\textbf{ARMA}$(p,q)$, $\\phi(L)x_t = \\theta(L)w_t$: $f(\\nu) = \\sigma^2\\dfrac{|\\theta(e^{-2\\pi i\\nu})|^2}{|\\phi(e^{-2\\pi i\\nu})|^2}$'),
     [T('a ratio of two polynomials in $\\cos(2\\pi\\nu)$: smooth and easy to compute (\\refBDtm, Sec.~4.4)', 'un raport de două polinoame în $\\cos(2\\pi\\nu)$: neted și ușor de calculat (\\refBDtm, secț.~4.4)')])), size='footnotesize')

solved(('AR(1) and MA(1) spectra', 'spectrele AR(1) și MA(1)'), [
    (T('AR(1) with $\\phi = 0.6$, $\\sigma^2 = 1$', 'AR(1) cu $\\phi = 0.6$, $\\sigma^2 = 1$'),
     [T('$\\nu = 0$: $\\cos 0 = 1$, $f(0) = 1/(1 - 0.6)^2 = 1/0.16 = @{ex.ar1.f0}$', '$\\nu = 0$: $\\cos 0 = 1$, $f(0) = 1/(1 - 0.6)^2 = 1/0.16 = @{ex.ar1.f0}$'),
      T('$\\nu = 1/2$: $\\cos\\pi = -1$, $f(1/2) = 1/(1 + 0.6)^2 = 1/2.56 = @{ex.ar1.f5}$', '$\\nu = 1/2$: $\\cos\\pi = -1$, $f(1/2) = 1/(1 + 0.6)^2 = 1/2.56 = @{ex.ar1.f5}$'),
      T('ratio @{ex.ar1.ratio}: slow cycles carry far more variance than fast ones', 'raportul @{ex.ar1.ratio}: ciclurile lente poartă mult mai multă varianță decît cele rapide')]),
    (T('MA(1) with $\\theta = 0.5$, $\\sigma^2 = 1$', 'MA(1) cu $\\theta = 0.5$, $\\sigma^2 = 1$'),
     [T('$f(0) = (1 + 0.5)^2 = @{ex.ma1.f0}$ and $f(1/2) = (1 - 0.5)^2 = @{ex.ma1.f5}$', '$f(0) = (1 + 0.5)^2 = @{ex.ma1.f0}$ și $f(1/2) = (1 - 0.5)^2 = @{ex.ma1.f5}$')]),
    (T('Check of the area for the AR(1): $\\int_{-1/2}^{1/2} f = \\gamma(0) = 1/(1 - \\phi^2) = 1/0.64 = @{sp.ar1.var}$', 'Verificarea ariei pentru AR(1): $\\int_{-1/2}^{1/2} f = \\gamma(0) = 1/(1 - \\phi^2) = 1/0.64 = @{sp.ar1.var}$'),
     [T('the numerical integral of the formula gives the same value', 'integrala numerică a formulei dă aceeași valoare')])])

D.frame(T('AR(2) and pseudo-cycles', 'AR(2) și pseudo-ciclurile'), items(
    (T('\\textbf{AR(2)} $x_t = \\phi_1 x_{t-1} + \\phi_2 x_{t-2} + w_t$: $f(\\nu) = \\dfrac{\\sigma^2}{|1 - \\phi_1 e^{-2\\pi i\\nu} - \\phi_2 e^{-4\\pi i\\nu}|^2}$', '\\textbf{AR(2)} $x_t = \\phi_1 x_{t-1} + \\phi_2 x_{t-2} + w_t$: $f(\\nu) = \\dfrac{\\sigma^2}{|1 - \\phi_1 e^{-2\\pi i\\nu} - \\phi_2 e^{-4\\pi i\\nu}|^2}$'),
     [T('with complex roots ($\\phi_1^2 + 4\\phi_2 < 0$) the ACF is a damped wave and the spectrum has a peak inside $(0, 1/2)$', 'pentru rădăcini complexe ($\\phi_1^2 + 4\\phi_2 < 0$) ACF este o undă amortizată, iar spectrul are un vîrf în interiorul intervalului $(0, 1/2)$'),
      T('the peak is at $\\cos(2\\pi\\nu^*) = \\phi_1(\\phi_2 - 1)/(4\\phi_2)$, when the right-hand side lies in $[-1, 1]$', 'vîrful se află la $\\cos(2\\pi\\nu^*) = \\phi_1(\\phi_2 - 1)/(4\\phi_2)$, cînd membrul drept este în $[-1, 1]$')]),
    (T('A \\textbf{pseudo-cycle}: an irregular cycle whose length and amplitude vary from one round to the next', 'Un \\textbf{pseudo-ciclu}: un ciclu neregulat, a cărui lungime și amplitudine variază de la o rundă la alta'),
     [T('\\refYule\\ fitted an AR(2) to Wolfer\'s sunspot numbers: the first autoregressive model, an alternative to fixed sine waves', '\\refYule\\ a estimat un AR(2) pentru numerele de pete solare ale lui Wolfer: primul model autoregresiv, o alternativă la undele sinusoidale fixe')]),
    T('Business cycles are pseudo-cycles of this kind: recurrent, but never with the same length (Section 5)', 'Ciclurile economice sînt pseudo-cicluri de acest fel: recurente, dar niciodată de aceeași lungime (secțiunea 5)')))

solved(('the peak of an AR(2) spectrum', 'vîrful spectrului unui AR(2)'), [
    (T('$\\phi_1 = 1.5$, $\\phi_2 = -0.75$, $\\sigma^2 = 1$; complex roots, since $1.5^2 + 4(-0.75) = -0.75 < 0$', '$\\phi_1 = 1.5$, $\\phi_2 = -0.75$, $\\sigma^2 = 1$; rădăcini complexe, deoarece $1.5^2 + 4(-0.75) = -0.75 < 0$'),
     [T('$\\cos(2\\pi\\nu^*) = 1.5 \\cdot (-1.75)/(-3) = @{ex.ar2.cos}$, so $\\nu^* = \\arccos(@{ex.ar2.cos})/(2\\pi) = @{ex.ar2.nu}$', '$\\cos(2\\pi\\nu^*) = 1.5 \\cdot (-1.75)/(-3) = @{ex.ar2.cos}$, deci $\\nu^* = \\arccos(@{ex.ar2.cos})/(2\\pi) = @{ex.ar2.nu}$'),
      T('period $1/\\nu^* = @{ex.ar2.per}$ observations; for yearly data, a cycle of about 12 years', 'perioada $1/\\nu^* = @{ex.ar2.per}$ observații; pentru date anuale, un ciclu de aproximativ 12 ani')]),
    (T('Height: $f(\\nu^*) = @{ex.ar2.fp}$ against $f(0) = 1/(1 - 1.5 + 0.75)^2 = @{ex.ar2.f0}$', 'Înălțimea: $f(\\nu^*) = @{ex.ar2.fp}$, față de $f(0) = 1/(1 - 1.5 + 0.75)^2 = @{ex.ar2.f0}$'),
     [T('a clear peak: the series oscillates with a typical period of about 12 observations', 'un vîrf clar: seria oscilează cu o perioadă tipică de aproximativ 12 observații')]),
    T('The same AR(2) is used in Section 4 to test the estimators of the spectrum', 'Același AR(2) este folosit în secțiunea 4 pentru a testa estimatorii spectrului')])

chart(T('A gallery of spectral densities', 'O galerie de densități spectrale'), 'tsa_ch12_spectra', 'TSA_ch12_spectra', [
    T('Theoretical spectra with $\\sigma^2 = 1$ on $0 \\le \\nu \\le 1/2$; the shaded area equals half the variance', 'Spectre teoretice, cu $\\sigma^2 = 1$, pe $0 \\le \\nu \\le 1/2$; aria hașurată este jumătate din varianță')],
    h='0.66\\textheight')

interp(('the gallery', 'galeriei'), [
    (T('Where the power sits tells the type of dependence', 'Locul în care se află puterea arată tipul de dependență'),
     [T('flat: no dependence (white noise); decreasing: positive persistence (AR(1) with $\\phi > 0$, MA(1) with $\\theta > 0$, ARMA(1,1))', 'plat: nicio dependență (zgomot alb); descrescător: persistență pozitivă (AR(1) cu $\\phi > 0$, MA(1) cu $\\theta > 0$, ARMA(1,1))'),
      T('increasing: alternation of signs (AR(1) with $\\phi < 0$); a hump inside the band: a pseudo-cycle (AR(2))', 'crescător: alternanță de semne (AR(1) cu $\\phi < 0$); o cocoașă în interiorul benzii: un pseudo-ciclu (AR(2))')]),
    (T('The AR(2) concentrates its variance @{sp.ar2.var} in a narrow band around $\\nu = @{ex.ar2.nu}$', 'AR(2) își concentrează varianța @{sp.ar2.var} într-o bandă îngustă în jurul lui $\\nu = @{ex.ar2.nu}$'),
     [T('the ARMA(1,1) adds the MA term to the AR(1): $f(0) = @{sp.arma.f0}$, a steeper fall', 'ARMA(1,1) adaugă termenul MA la AR(1): $f(0) = @{sp.arma.f0}$, o scădere mai abruptă')]),
    T('Reading direction: estimate the spectrum from data, look at its shape, then choose a model whose spectrum looks the same', 'Ordinea de lucru: estimați spectrul din date, priviți forma lui, apoi alegeți un model care are un spectru asemănător')])

D.recap(('The spectral density', 'densitatea spectrală'), [
    T('$f(\\nu) = \\sum_h\\gamma(h)e^{-2\\pi i\\nu h}$; the area under $f$ is the variance; $f$ and the ACF are equivalent', '$f(\\nu) = \\sum_h\\gamma(h)e^{-2\\pi i\\nu h}$; aria de sub $f$ este varianța; $f$ și ACF sînt echivalente'),
    T('White noise is flat; AR(1) with $\\phi > 0$ puts power at low frequencies; AR(2) with complex roots gives a peak', 'Zgomotul alb are spectrul plat; AR(1) cu $\\phi > 0$ pune puterea la frecvențele joase; AR(2) cu rădăcini complexe dă un vîrf'),
    T('ARMA spectra: $\\sigma^2|\\theta(e^{-2\\pi i\\nu})|^2/|\\phi(e^{-2\\pi i\\nu})|^2$', 'Spectrele ARMA: $\\sigma^2|\\theta(e^{-2\\pi i\\nu})|^2/|\\phi(e^{-2\\pi i\\nu})|^2$')])

# =============================================================================
# 3. PERIODOGRAMA
# =============================================================================
D.section('The periodogram', 'Periodograma')

D.frame(T('Arthur Schuster and hidden periodicities', 'Arthur Schuster și periodicitățile ascunse'), two(
    ph('schuster', 'A. Schuster (1851--1934)', h='0.5\\textheight'),
    items((T('\\refSchuster\\ introduced the \\textbf{periodogram} to search for ``hidden periodicities\'\' in meteorological data', '\\refSchuster\\ a introdus \\textbf{periodograma} pentru a căuta „periodicități ascunse” în date meteorologice'),
           [T('a claimed 26-day cycle; he showed how to judge whether a peak could be due to chance', 'un presupus ciclu de 26 de zile; a arătat cum se judecă dacă un vîrf poate fi întîmplător')]),
          (T('Sunspots: counted daily since the 18th century; Rudolf Wolf built the standard index in 1848', 'Petele solare: numărate zilnic încă din secolul al XVIII-lea; Rudolf Wolf a construit indicele standard în 1848'),
           [T('their 11-year cycle is the classic test case of spectral analysis (Chapter 0)', 'ciclul lor de 11 ani este cazul clasic de test al analizei spectrale (Capitolul 0)')]),
          T('The question of the section: how do we estimate $f(\\nu)$ from one sample, and how reliable is the estimate?', 'Întrebarea secțiunii: cum estimăm $f(\\nu)$ dintr-un singur eșantion și cît de sigură este estimarea?'))))

D.frame(T('The periodogram', 'Periodograma'), items(
    (T('\\textbf{Periodogram}: $I(\\nu_j) = |d(\\nu_j)|^2$ at the Fourier frequencies $\\nu_j = j/T$ (after removing the mean)', '\\textbf{Periodograma}: $I(\\nu_j) = |d(\\nu_j)|^2$ la frecvențele Fourier $\\nu_j = j/T$ (după eliminarea mediei)'),
     [T('equivalently $I(\\nu_j) = \\sum_{|h| < T}\\hat\\gamma(h)e^{-2\\pi i\\nu_j h}$: the spectral formula with the sample autocovariances $\\hat\\gamma(h)$', 'echivalent $I(\\nu_j) = \\sum_{|h| < T}\\hat\\gamma(h)e^{-2\\pi i\\nu_j h}$: formula spectrului cu autocovarianțele de selecție $\\hat\\gamma(h)$')]),
    (T('Regression view: regress $x_t$ on $\\cos(2\\pi\\nu_j t)$ and $\\sin(2\\pi\\nu_j t)$ by OLS, with coefficients $a_j, b_j$', 'Interpretarea prin regresie: regresați $x_t$ pe $\\cos(2\\pi\\nu_j t)$ și $\\sin(2\\pi\\nu_j t)$ prin OLS, cu coeficienții $a_j, b_j$'),
     [T('$I(\\nu_j) = \\frac{T}{4}(a_j^2 + b_j^2)$: the squared amplitude of the best-fitting cycle at $\\nu_j$, scaled by $T/4$', '$I(\\nu_j) = \\frac{T}{4}(a_j^2 + b_j^2)$: pătratul amplitudinii celui mai bun ciclu la $\\nu_j$, înmulțit cu $T/4$')]),
    (T('Variance decomposition: $\\hat\\gamma(0) = \\frac{1}{T}\\sum_{j=1}^{T-1} I(\\nu_j)$', 'Descompunerea varianței: $\\hat\\gamma(0) = \\frac{1}{T}\\sum_{j=1}^{T-1} I(\\nu_j)$'),
     [T('the share of a band of frequencies is the sum of its ordinates divided by the total', 'ponderea unei benzi de frecvențe este suma ordonatelor ei împărțită la total')]),
    T('In Python: \\texttt{np.abs(np.fft.fft(x - x.mean()))**2 / T}; keep $j = 1, \\ldots, \\lfloor T/2\\rfloor$', 'În Python: \\texttt{np.abs(np.fft.fft(x - x.mean()))**2 / T}; se păstrează $j = 1, \\ldots, \\lfloor T/2\\rfloor$')))

chart(T('Sunspots and their periodogram', 'Petele solare și periodograma lor'), 'tsa_ch12_sunspots', 'TSA_ch12_sunspots', [
    T('Yearly mean sunspot number, 1700--2008, @{su.T} observations (Wolf / SILSO series, statsmodels); right: the raw periodogram', 'Numărul mediu anual de pete solare, 1700--2008, @{su.T} observații (seria Wolf / SILSO, statsmodels); dreapta: periodograma brută')],
    h='0.62\\textheight')

interp(('the sunspot periodogram', 'periodogramei petelor solare'), [
    (T('The highest ordinate is at $j = @{su.j}$: $\\nu = @{su.j}/@{su.T} = @{su.nu}$ cycles per year, a period of @{su.per} years', 'Cea mai mare ordonată este la $j = @{su.j}$: $\\nu = @{su.j}/@{su.T} = @{su.nu}$ cicluri pe an, o perioadă de @{su.per} ani'),
     [T('the next largest ordinates are at @{su.p2} and @{su.p3} years: the cycle is not exactly regular', 'următoarele ordonate ca mărime sînt la @{su.p2} și @{su.p3} ani: ciclul nu este perfect regulat')]),
    (T('Periods between 9 and 13 years carry @{su.share}\\% of the variance', 'Perioadele între 9 și 13 ani explică @{su.share}\\% din varianță'),
     [T('the small ordinates near $\\nu = 0$ reflect slow changes in the height of the cycles (for example the weak cycles around 1800)', 'ordonatele mici din apropierea lui $\\nu = 0$ reflectă schimbările lente ale înălțimii ciclurilor (de exemplu ciclurile slabe din jurul anului 1800)')]),
    T('The raw periodogram is very spiky: neighbouring ordinates jump up and down; the next slides explain why', 'Periodograma brută este foarte zimțată: ordonatele vecine sar în sus și în jos; slide-urile următoare explică de ce')])

D.frame(T('Statistical properties of the periodogram', 'Proprietățile statistice ale periodogramei'), items(
    (T('For large $T$, at a Fourier frequency $0 < \\nu_j < 1/2$ (\\refBDtm, Sec.~10.3; \\refSS, Sec.~4.3):', 'Pentru $T$ mare, la o frecvență Fourier $0 < \\nu_j < 1/2$ (\\refBDtm, secț.~10.3; \\refSS, secț.~4.3):'),
     [T('$\\dfrac{2I(\\nu_j)}{f(\\nu_j)} \\approx \\chi^2_2$: so $E[I(\\nu_j)] \\approx f(\\nu_j)$ (unbiased) and $\\mathrm{Var}[I(\\nu_j)] \\approx f(\\nu_j)^2$', '$\\dfrac{2I(\\nu_j)}{f(\\nu_j)} \\approx \\chi^2_2$: deci $E[I(\\nu_j)] \\approx f(\\nu_j)$ (nedeplasată) și $\\mathrm{Var}[I(\\nu_j)] \\approx f(\\nu_j)^2$'),
      T('ordinates at different Fourier frequencies are approximately independent', 'ordonatele de la frecvențe Fourier diferite sînt aproximativ independente')]),
    (T('The variance does \\textbf{not} shrink when $T$ grows: the periodogram is \\textbf{not a consistent} estimator of $f$', 'Varianța \\textbf{nu} scade cînd $T$ crește: periodograma \\textbf{nu este un estimator consistent} al lui $f$'),
     [T('a larger $T$ gives more ordinates (a finer grid), not more precise ones', 'un $T$ mai mare dă mai multe ordonate (o grilă mai fină), nu ordonate mai precise')]),
    (T('$\\chi^2_2/2$ is an exponential distribution with mean 1: a single ordinate can easily be 3 times the true value', '$\\chi^2_2/2$ este o distribuție exponențială cu media 1: o singură ordonată poate depăși ușor de 3 ori valoarea adevărată'),
     [T('its 95\\% quantile is @{ic.q95}: about 1 ordinate in 20 exceeds $@{ic.q95}\\,f(\\nu)$ by chance', 'cuantila ei de 95\\% este @{ic.q95}: aproximativ 1 ordonată din 20 depășește întîmplător $@{ic.q95}\\,f(\\nu)$')])))

chart(T('The periodogram of white noise', 'Periodograma zgomotului alb'), 'tsa_ch12_inconsistency', 'TSA_ch12_inconsistency', [
    T('Gaussian white noise with $\\sigma^2 = 1$, so the true spectrum is flat at 1; samples of $T = 128$ and $T = 2048$', 'Zgomot alb gaussian cu $\\sigma^2 = 1$, deci spectrul adevărat este constant, egal cu 1; eșantioane de $T = 128$ și $T = 2048$')],
    h='0.6\\textheight')

interp(('the white-noise periodogram', 'periodogramei zgomotului alb'), [
    (T('$T = 128$: mean of the ordinates @{ic.128.m}, standard deviation @{ic.128.sd}', '$T = 128$: media ordonatelor @{ic.128.m}, abaterea standard @{ic.128.sd}'),
     [T('$T = 2048$: mean @{ic.2048.m}, standard deviation @{ic.2048.sd}: sixteen times more data, the same scatter', '$T = 2048$: media @{ic.2048.m}, abaterea standard @{ic.2048.sd}: de 16 ori mai multe date, aceeași împrăștiere')]),
    (T('The largest ordinate for $T = 2048$ is @{ic.2048.max}: with 1024 ordinates, isolated tall spikes appear by chance', 'Cea mai mare ordonată pentru $T = 2048$ este @{ic.2048.max}: din 1024 de ordonate, apar întîmplător vîrfuri înalte izolate'),
     [T('a tall spike is not yet a cycle: it needs a test (next slide) or a smoothed estimate (Section 4)', 'un vîrf înalt nu este încă un ciclu: are nevoie de un test (slide-ul următor) sau de o estimare netezită (secțiunea 4)')]),
    T('The mean is right (about 1): the periodogram is unbiased but noisy; averaging neighbouring ordinates will reduce the noise', 'Media este corectă (aproximativ 1): periodograma este nedeplasată, dar zgomotoasă; media ordonatelor vecine va reduce zgomotul')])

D.frame(T("Fisher's test of a hidden periodicity", 'Testul lui Fisher pentru o periodicitate ascunsă'), items(
    (T('$H_0$: Gaussian white noise; $H_1$: white noise plus one cycle at an unknown Fourier frequency', '$H_0$: zgomot alb gaussian; $H_1$: zgomot alb plus un ciclu la o frecvență Fourier necunoscută'),
     [T('statistic of \\refFisher: $g = \\max_j I(\\nu_j)\\,/\\,\\sum_{j=1}^{m} I(\\nu_j)$, with $m$ ordinates in $(0, 1/2)$', 'statistica lui \\refFisher: $g = \\max_j I(\\nu_j)\\,/\\,\\sum_{j=1}^{m} I(\\nu_j)$, cu $m$ ordonate în $(0, 1/2)$'),
      T('under $H_0$ each ordinate is about $1/m$ of the total; $g$ much larger than $1/m$ signals a cycle', 'sub $H_0$ fiecare ordonată reprezintă aproximativ $1/m$ din total; un $g$ mult mai mare decît $1/m$ semnalează un ciclu'),
      T('exact p-value: $P(g > x) = \\sum_{k \\ge 1,\\ kx < 1}(-1)^{k-1}\\binom{m}{k}(1 - kx)^{m-1}$', 'p-value exactă: $P(g > x) = \\sum_{k \\ge 1,\\ kx < 1}(-1)^{k-1}\\binom{m}{k}(1 - kx)^{m-1}$')]),
    (T('Sunspots: $m = @{su.m}$, $g = @{su.g}$, against $1/m = @{su.gcrit}$; p-value @{su.pv}', 'Petele solare: $m = @{su.m}$, $g = @{su.g}$, față de $1/m = @{su.gcrit}$; p-value @{su.pv}'),
     [T('the 11-year peak is far too large to be chance', 'vîrful de 11 ani este mult prea mare pentru a fi întîmplător')]),
    T('Caution: $H_0$ is white noise; against a red (AR(1)-like) background, low-frequency peaks are more likely by chance, so compare with a smooth background spectrum', 'Atenție: $H_0$ este zgomotul alb; pe un fond „roșu” (de tip AR(1)), vîrfurile de joasă frecvență apar mai ușor din întîmplare, deci comparați cu un spectru de fond neted')))

D.recap(('The periodogram', 'periodograma'), [
    T('$I(\\nu_j) = |d(\\nu_j)|^2$: the squared amplitude of the cycle at each Fourier frequency; it sums to the variance', '$I(\\nu_j) = |d(\\nu_j)|^2$: pătratul amplitudinii ciclului la fiecare frecvență Fourier; suma dă varianța'),
    T('$2I/f \\approx \\chi^2_2$: unbiased but not consistent; neighbouring ordinates are nearly independent', '$2I/f \\approx \\chi^2_2$: nedeplasată, dar neconsistentă; ordonatele vecine sînt aproape independente'),
    T("Sunspots: a peak at @{su.per} years; Fisher's $g$ rejects white noise", 'Petele solare: un vîrf la @{su.per} ani; testul $g$ al lui Fisher respinge zgomotul alb')])

# =============================================================================
# 4. ESTIMAREA SPECTRULUI
# =============================================================================
D.section('Estimating the spectrum', 'Estimarea spectrului')

D.frame(T('Leakage and tapering', 'Scurgerea spectrală și ferestrele de atenuare'), items(
    (T('A cycle whose frequency is not a Fourier frequency does not fit a whole number of times in the sample', 'Un ciclu a cărui frecvență nu este o frecvență Fourier nu încape de un număr întreg de ori în eșantion'),
     [T('cutting the series at $t = 1$ and $t = T$ creates a jump; its power \\textbf{leaks} into all other frequencies', 'tăierea seriei la $t = 1$ și $t = T$ creează un salt; puterea lui se \\textbf{scurge} la toate celelalte frecvențe'),
      T('leakage from a strong peak can hide a weak cycle elsewhere', 'scurgerea dintr-un vîrf puternic poate ascunde un ciclu slab în altă parte')]),
    (T('\\textbf{Tapering}: multiply the data by a window $h_t$ that goes smoothly to 0 at both ends before the DFT', '\\textbf{Atenuarea} (tapering): înmulțiți datele cu o fereastră $h_t$ care scade lin spre 0 la ambele capete, înainte de DFT'),
     [T('Hann (cosine bell) window: $h_t = \\frac12[1 - \\cos(2\\pi(t - 0.5)/T)]$, rescaled to keep the total power', 'fereastra Hann (clopot cosinus): $h_t = \\frac12[1 - \\cos(2\\pi(t - 0.5)/T)]$, rescalată pentru a păstra puterea totală'),
      T('the price: a slightly wider main peak (less resolution) for much smaller leakage', 'prețul: un vîrf principal puțin mai lat (rezoluție mai mică) pentru o scurgere mult mai mică')]),
    T('Default in practice: always taper a little; always remove the mean (and a trend, if there is one) first', 'În practică: aplicați întotdeauna o atenuare ușoară; eliminați întîi media (și trendul, dacă există)')))

chart(T('Leakage: raw and tapered periodograms', 'Scurgerea: periodograma brută și cea cu atenuare'), 'tsa_ch12_leakage', 'TSA_ch12_leakage', [
    T('$T = 256$: $\\cos(2\\pi\\cdot @{lk.nu1}\\,t)$ (between two Fourier frequencies) plus $0.02\\cos(2\\pi\\cdot 0.3\\,t)$ and tiny noise; log scale', '$T = 256$: $\\cos(2\\pi\\cdot @{lk.nu1}\\,t)$ (între două frecvențe Fourier) plus $0.02\\cos(2\\pi\\cdot 0.3\\,t)$ și un zgomot foarte mic; scară logaritmică')],
    h='0.6\\textheight')

interp(('leakage', 'scurgerii spectrale'), [
    (T('Raw periodogram: the strong cycle leaks power everywhere; at $\\nu = 0.3$ the weak cycle stands only @{lk.rr} times above the leaked background', 'Periodograma brută: ciclul puternic își scurge puterea peste tot; la $\\nu = 0.3$ ciclul slab este doar de @{lk.rr} ori peste fondul scurs'),
     [T('with real noise added, this weak peak would be invisible', 'cu un zgomot real adăugat, acest vîrf slab ar fi invizibil')]),
    (T('Hann taper: the background falls by several orders of magnitude and the weak cycle stands @{lk.tr} times above it', 'Fereastra Hann: fondul scade cu mai multe ordine de mărime, iar ciclul slab ajunge de @{lk.tr} ori peste el'),
     [T('the main peak is a little wider: tapering trades resolution for less leakage', 'vîrful principal este puțin mai lat: atenuarea schimbă rezoluție pentru o scurgere mai mică')]),
    T('Log scale is essential: on a linear scale both curves look like a single spike', 'Scara logaritmică este esențială: pe o scară liniară ambele curbe arată ca un singur vîrf')])

D.frame(T('Smoothing the periodogram', 'Netezirea periodogramei'), items(
    (T('Idea: $f$ is smooth, the ordinates are nearly independent: average $L = 2m + 1$ neighbouring ordinates', 'Ideea: $f$ este netedă, iar ordonatele sînt aproape independente: faceți media a $L = 2m + 1$ ordonate vecine'),
     [T('\\textbf{Daniell} estimator: $\\hat f(\\nu_j) = \\frac{1}{L}\\sum_{k=-m}^{m} I(\\nu_{j+k})$; variance about $f^2/L$', 'estimatorul \\textbf{Daniell}: $\\hat f(\\nu_j) = \\frac{1}{L}\\sum_{k=-m}^{m} I(\\nu_{j+k})$; varianța aproximativ $f^2/L$'),
      T('\\textbf{bandwidth} $B = L/T$: the width of the frequency band that is averaged', '\\textbf{lățimea de bandă} $B = L/T$: lățimea benzii de frecvențe peste care se face media')]),
    (T('Bias--variance trade-off: a larger $L$ lowers the variance but flattens narrow peaks (bias)', 'Compromisul deplasare--varianță: un $L$ mai mare reduce varianța, dar aplatizează vîrfurile înguste (deplasare)'),
     [T('consistency needs $L \\to \\infty$ and $L/T \\to 0$, e.g.\\ $L \\approx \\sqrt T$', 'consistența cere $L \\to \\infty$ și $L/T \\to 0$, de exemplu $L \\approx \\sqrt T$')]),
    (T('Equivalent views: smoothing in frequency $=$ down-weighting distant autocovariances (lag windows)', 'Perspective echivalente: netezirea în frecvență $=$ ponderi mai mici pentru autocovarianțele îndepărtate (ferestre de decalaj)'),
     [T('Bartlett window: $\\hat f(\\nu) = \\sum_{|h| \\le M}(1 - |h|/M)\\hat\\gamma(h)e^{-2\\pi i\\nu h}$ (\\refBartlett; \\refBT)', 'fereastra Bartlett: $\\hat f(\\nu) = \\sum_{|h| \\le M}(1 - |h|/M)\\hat\\gamma(h)e^{-2\\pi i\\nu h}$ (\\refBartlett; \\refBT)')]),
    (T('\\textbf{Welch} \\refWelch: split the series into $K$ overlapping segments, taper each, average their periodograms', '\\textbf{Welch} \\refWelch: împărțiți seria în $K$ segmente care se suprapun, aplicați fereastra pe fiecare, faceți media periodogramelor'),
     [T('\\texttt{scipy.signal.welch}; the segment length sets the resolution', '\\texttt{scipy.signal.welch}; lungimea segmentului fixează rezoluția')])), size='footnotesize')

chart(T('Smoothed estimates of an AR(2) spectrum', 'Estimări netezite ale spectrului unui AR(2)'), 'tsa_ch12_smoothing', 'TSA_ch12_smoothing', [
    T('Simulated AR(2), $\\phi = (1.5, -0.75)$, $T = 512$; true spectrum (dashed) with the raw periodogram (left) and three smoothed estimates (right)', 'AR(2) simulat, $\\phi = (1.5;\\ -0.75)$, $T = 512$; spectrul adevărat (linie întreruptă), periodograma brută (stînga) și trei estimări netezite (dreapta)')],
    h='0.6\\textheight')

interp(('the smoothed estimates', 'estimărilor netezite'), [
    (T('True peak at $\\nu = @{sm.tp}$; the highest raw ordinate is at $\\nu = @{sm.rp}$: a single noisy ordinate misplaces the peak', 'Vîrful adevărat este la $\\nu = @{sm.tp}$; cea mai mare ordonată brută este la $\\nu = @{sm.rp}$: o singură ordonată zgomotoasă plasează greșit vîrful'),
     [T('peak of Daniell $m = 2$: $\\nu = @{sm.d2p}$; of Daniell $m = 10$: $\\nu = @{sm.d10p}$; of Welch ($K = @{sm.K}$ segments of 128): $\\nu = @{sm.wp}$', 'vîrful Daniell $m = 2$: $\\nu = @{sm.d2p}$; Daniell $m = 10$: $\\nu = @{sm.d10p}$; Welch ($K = @{sm.K}$ segmente de 128): $\\nu = @{sm.wp}$')]),
    (T('Mean squared error of $\\log\\hat f$: raw @{sm.mr}; Daniell $m = 2$: @{sm.m2}; $m = 10$: @{sm.m10}; Welch: @{sm.mw}', 'Eroarea pătratică medie a lui $\\log\\hat f$: brută @{sm.mr}; Daniell $m = 2$: @{sm.m2}; $m = 10$: @{sm.m10}; Welch: @{sm.mw}'),
     [T('the price of heavy smoothing: the peak height drops from @{sm.ftp} to @{sm.f10} (bias)', 'prețul netezirii puternice: înălțimea vîrfului scade de la @{sm.ftp} la @{sm.f10} (deplasare)')]),
    T('Choose the bandwidth by looking at several values: keep the narrowest one whose estimate is no longer dominated by noise', 'Alegeți lățimea de bandă privind mai multe valori: păstrați-o pe cea mai îngustă la care estimarea nu mai este dominată de zgomot')])

D.frame(T('Confidence bands for the spectrum', 'Benzi de încredere pentru spectru'), items(
    (T('Daniell estimate with $L$ terms: $\\dfrac{2L\\,\\hat f(\\nu)}{f(\\nu)} \\approx \\chi^2_{2L}$ (degrees of freedom $df = 2L$)', 'Estimarea Daniell cu $L$ termeni: $\\dfrac{2L\\,\\hat f(\\nu)}{f(\\nu)} \\approx \\chi^2_{2L}$ (grade de libertate $df = 2L$)'),
     [T('95\\% interval: $\\left[\\dfrac{df\\,\\hat f(\\nu)}{\\chi^2_{df}(0.975)},\\ \\dfrac{df\\,\\hat f(\\nu)}{\\chi^2_{df}(0.025)}\\right]$', 'intervalul de 95\\%: $\\left[\\dfrac{df\\,\\hat f(\\nu)}{\\chi^2_{df}(0.975)};\\ \\dfrac{df\\,\\hat f(\\nu)}{\\chi^2_{df}(0.025)}\\right]$')]),
    (T('The interval is a fixed multiple of $\\hat f$: on a log scale its width is the same at every frequency', 'Intervalul este un multiplu fix al lui $\\hat f$: pe scară logaritmică lățimea lui este aceeași la toate frecvențele'),
     [T('a peak is significant if the band at the peak lies above the band of the background around it', 'un vîrf este semnificativ dacă banda de la vîrf se află deasupra benzii fondului din jur')]),
    T('Welch with $K$ non-overlapping segments: $df \\approx 2K$; overlap and tapering change $df$ a little', 'Welch cu $K$ segmente care nu se suprapun: $df \\approx 2K$; suprapunerea și atenuarea modifică puțin $df$')))

solved(('a 95\\% band with $L = 5$', 'o bandă de 95\\% pentru $L = 5$'), [
    (T('$L = 5$ ($m = 2$), so $df = 10$; from the $\\chi^2_{10}$ table: $\\chi^2_{10}(0.025) = @{ex.ci.qlo}$ and $\\chi^2_{10}(0.975) = @{ex.ci.qhi}$', '$L = 5$ ($m = 2$), deci $df = 10$; din tabelul $\\chi^2_{10}$: $\\chi^2_{10}(0.025) = @{ex.ci.qlo}$ și $\\chi^2_{10}(0.975) = @{ex.ci.qhi}$'),
     [T('lower bound: $10/@{ex.ci.qhi} = @{ex.ci.lo}$ times $\\hat f$; upper bound: $10/@{ex.ci.qlo} = @{ex.ci.hi}$ times $\\hat f$', 'limita inferioară: $10/@{ex.ci.qhi} = @{ex.ci.lo}$ ori $\\hat f$; limita superioară: $10/@{ex.ci.qlo} = @{ex.ci.hi}$ ori $\\hat f$')]),
    (T('Sunspots: at the smoothed peak (period @{sc.per} years) $\\hat f = @{sc.f}$', 'Petele solare: la vîrful netezit (perioada @{sc.per} ani) $\\hat f = @{sc.f}$'),
     [T('95\\% interval: [@{sc.lo}, @{sc.hi}]: asymmetric, wider upwards', 'intervalul de 95\\%: [@{sc.lo}, @{sc.hi}]: asimetric, mai lat în sus')]),
    T('With only 5 ordinates the estimate is still uncertain by a factor of 2--3; more smoothing narrows the band but blurs the peak', 'Cu doar 5 ordonate estimarea are încă o incertitudine de un factor de 2--3; o netezire mai puternică îngustează banda, dar estompează vîrful')])

chart(T('Sunspots: smoothed spectrum with a confidence band', 'Petele solare: spectrul netezit cu bandă de încredere'), 'tsa_ch12_sunspot_ci', 'TSA_ch12_sunspot_ci', [
    T('Daniell smoother with $m = 2$ ($L = 5$, $df = 10$, bandwidth $B = 5/@{su.T} = @{sc.B}$ cycles per year) and its 95\\% band; log scale', 'Netezire Daniell cu $m = 2$ ($L = 5$, $df = 10$, lățimea de bandă $B = 5/@{su.T} = @{sc.B}$ cicluri pe an) și banda de 95\\%; scară logaritmică')],
    h='0.6\\textheight')

interp(('the smoothed sunspot spectrum', 'spectrului netezit al petelor solare'), [
    (T('The band at the peak (period @{sc.per} years) lies entirely above the band of the background at higher frequencies', 'Banda de la vîrf (perioada @{sc.per} ani) se află în întregime deasupra benzii fondului de la frecvențele mai înalte'),
     [T('the 11-year cycle is a robust feature, not a chance spike', 'ciclul de 11 ani este o trăsătură robustă, nu un vîrf întîmplător')]),
    (T('A second, smaller hump near $\\nu \\approx 0.18$ (about 5.5 years) is the first \\textbf{harmonic}', 'O a doua cocoașă, mai mică, în jurul lui $\\nu \\approx 0.18$ (aproximativ 5,5 ani) este prima \\textbf{armonică}'),
     [T('the sunspot cycle is not a sine wave: it rises fast and decays slowly, and a non-sinusoidal cycle has power at multiples of its frequency', 'ciclul petelor solare nu este o sinusoidă: crește repede și scade lent, iar un ciclu nesinusoidal are putere la multiplii frecvenței lui')]),
    T('Power also rises towards $\\nu = 0$: the height of the cycles changes over decades and centuries', 'Puterea crește și spre $\\nu = 0$: înălțimea ciclurilor se schimbă de-a lungul deceniilor și secolelor')])

D.recap(('Estimating the spectrum', 'estimarea spectrului'), [
    T('Taper to reduce leakage; plot spectra on a log scale', 'Aplicați o fereastră de atenuare pentru a reduce scurgerea; reprezentați spectrele pe scară logaritmică'),
    T('Smooth over $L$ ordinates (Daniell, Bartlett, Welch): variance $\\approx f^2/L$, but narrow peaks are flattened', 'Netezați peste $L$ ordonate (Daniell, Bartlett, Welch): varianța $\\approx f^2/L$, dar vîrfurile înguste sînt aplatizate'),
    T('95\\% band: $[df\\,\\hat f/\\chi^2_{df}(0.975),\\ df\\,\\hat f/\\chi^2_{df}(0.025)]$ with $df = 2L$', 'Banda de 95\\%: $[df\\,\\hat f/\\chi^2_{df}(0.975);\\ df\\,\\hat f/\\chi^2_{df}(0.025)]$, cu $df = 2L$')])

# =============================================================================
# 5. CICLURI ÎN DATE
# =============================================================================
D.section('Cycles in economic and energy data', 'Cicluri în datele economice și energetice')

D.frame(T('The typical spectral shape and the business cycle', 'Forma spectrală tipică și ciclul economic'), items(
    (T('\\refGranger: the spectrum of most economic levels (GDP, prices) falls steeply from $\\nu = 0$: the \\textbf{typical spectral shape}', '\\refGranger: spectrul majorității nivelurilor economice (PIB, prețuri) scade abrupt de la $\\nu = 0$: \\textbf{forma spectrală tipică}'),
     [T('trends and near unit roots put almost all the variance at the lowest frequencies and hide the cycles', 'trendurile și rădăcinile aproape unitare pun aproape toată varianța la frecvențele cele mai joase și ascund ciclurile'),
      T('so we first make the series stationary: growth rates (differences) or a detrended cycle', 'de aceea transformăm întîi seria într-una staționară: rate de creștere (diferențe) sau un ciclu fără trend')]),
    (T('\\textbf{Business-cycle band}: fluctuations lasting 1.5 to 8 years, i.e.\\ 6 to 32 quarters (\\refBK)', '\\textbf{Banda ciclului economic}: fluctuații care durează între 1,5 și 8 ani, adică între 6 și 32 de trimestre (\\refBK)'),
     [T('in frequency: $1/32 \\le \\nu \\le 1/6$ cycles per quarter', 'în frecvență: $1/32 \\le \\nu \\le 1/6$ cicluri pe trimestru')]),
    (T('Data: US real GDP (FRED GDPC1, @{g.US.T} quarterly growth rates) and Romanian real GDP (Eurostat, seasonally adjusted, @{g.RO.T} growth rates)', 'Datele: PIB-ul real al SUA (FRED GDPC1, @{g.US.T} de rate de creștere trimestriale) și PIB-ul real al României (Eurostat, ajustat sezonier, @{g.RO.T} de rate de creștere)'),
     [T('both up to 2019 Q4: the 2020 collapse and rebound would dominate every periodogram', 'ambele pînă în T4 2019: prăbușirea și revenirea din 2020 ar domina orice periodogramă'),
      T('the cycle is also extracted with the HP filter, $\\lambda = 1600$ (\\refHPf; Chapter 10)', 'ciclul este extras și cu filtrul HP, $\\lambda = 1600$ (\\refHPf; Capitolul 10)')])))

chart(T('GDP growth and HP cycles', 'Creșterea PIB și ciclurile HP'), 'tsa_ch12_gdp_series', 'TSA_ch12_gdp_series', [
    T('Quarterly growth $100\\,\\Delta\\ln \\mathrm{GDP}_t$ and the HP cycle ($100\\ln\\mathrm{GDP}_t$ minus its HP trend), United States 1947--2019 and Romania 1995--2019', 'Creșterea trimestrială $100\\,\\Delta\\ln \\mathrm{PIB}_t$ și ciclul HP ($100\\ln\\mathrm{PIB}_t$ minus trendul HP), Statele Unite 1947--2019 și România 1995--2019')],
    h='0.6\\textheight')

interp(('the GDP series', 'seriilor PIB'), [
    (T('United States: mean growth @{g.US.mean}\\% per quarter, standard deviation @{g.US.sd}; HP cycle standard deviation @{g.US.csd}\\% of trend', 'Statele Unite: creșterea medie @{g.US.mean}\\% pe trimestru, abaterea standard @{g.US.sd}; abaterea standard a ciclului HP @{g.US.csd}\\% din trend'),
     [T('the cycle shows the recessions as deep troughs (1958, 1975, 1982, 2009)', 'ciclul arată recesiunile ca minime adînci (1958, 1975, 1982, 2009)')]),
    (T('Romania: mean growth @{g.RO.mean}\\%, standard deviation @{g.RO.sd}: three times more volatile', 'România: creșterea medie @{g.RO.mean}\\%, abaterea standard @{g.RO.sd}: de trei ori mai volatilă'),
     [T('one large boom--bust (2004--2010) dominates a sample of only 25 years', 'un singur ciclu mare de expansiune și recesiune (2004--2010) domină un eșantion de doar 25 de ani')]),
    T('The growth rate is noisy and short-lived; the HP cycle is smooth and persistent: their spectra will differ', 'Rata de creștere este zgomotoasă și de scurtă durată; ciclul HP este neted și persistent: spectrele lor vor fi diferite')])

chart(T('Spectra of GDP growth and HP cycles', 'Spectrele creșterii PIB și ale ciclurilor HP'), 'tsa_ch12_gdp_spectra', 'TSA_ch12_gdp_spectra', [
    T('Daniell smoother, $m = 2$; each spectrum divided by the variance of its series; shaded: periods of 6 to 32 quarters', 'Netezire Daniell, $m = 2$; fiecare spectru este împărțit la varianța seriei sale; hașurat: perioade între 6 și 32 de trimestre')],
    h='0.6\\textheight')

interp(('the GDP spectra', 'spectrelor PIB'), [
    (T('US growth: @{gs.US.g.bc}\\% of the variance in the business-cycle band; the smoothed peak is at about @{gs.US.g.per} quarters', 'Creșterea SUA: @{gs.US.g.bc}\\% din varianță în banda ciclului economic; vîrful netezit este la aproximativ @{gs.US.g.per} trimestre'),
     [T('Romanian growth: only @{gs.RO.g.bc}\\%; most of its variance is at high frequencies (quarter-to-quarter noise and revisions)', 'creșterea României: doar @{gs.RO.g.bc}\\%; cea mai mare parte a varianței este la frecvențe înalte (zgomot de la un trimestru la altul și revizuiri)')]),
    (T('US HP cycle: @{gs.US.c.bc}\\% of the variance in the band, peak at @{gs.US.c.per} quarters (about @{gs.US.c.yrs} years)', 'Ciclul HP al SUA: @{gs.US.c.bc}\\% din varianță în bandă, vîrful la @{gs.US.c.per} trimestre (aproximativ @{gs.US.c.yrs} ani)'),
     [T('Romanian HP cycle: @{gs.RO.c.bc}\\% in the band and @{gs.RO.c.low}\\% at periods longer than 8 years: one long cycle in a short sample', 'ciclul HP al României: @{gs.RO.c.bc}\\% în bandă și @{gs.RO.c.low}\\% la perioade de peste 8 ani: un singur ciclu lung într-un eșantion scurt')]),
    T('Caution: the HP filter itself removes low frequencies and can shape the peak (Section 7); 99 Romanian observations give a very wide confidence band', 'Atenție: filtrul HP elimină el însuși frecvențele joase și poate modela vîrful (secțiunea 7); cele 99 de observații ale României dau o bandă de încredere foarte largă')])

chart(T('The rhythm of electricity demand', 'Ritmul consumului de energie electrică'), 'tsa_ch12_load', 'TSA_ch12_load', [
    T('Hourly electricity load of Romania (ENTSO-E, MW, UTC hours), 2022--2026, @{lo.T} hours, mean @{lo.mean} MW (Chapter 4); Welch spectrum with 8-week segments', 'Consumul orar de energie electrică al României (ENTSO-E, MW, ore UTC), 2022--2026, @{lo.T} de ore, media @{lo.mean} MW (Capitolul 4); spectrul Welch cu segmente de 8 săptămîni')],
    h='0.58\\textheight')

interp(('the load spectrum', 'spectrului consumului'), [
    (T('Sharp peaks at 24, 12 and 8 hours: the daily cycle and its harmonics (morning and evening peaks are not a sine wave)', 'Vîrfuri ascuțite la 24, 12 și 8 ore: ciclul zilnic și armonicele lui (vîrfurile de dimineață și de seară nu formează o sinusoidă)'),
     [T('the daily cycle and its harmonics hold @{lo.daily}\\% of the variance', 'ciclul zilnic și armonicele lui explică @{lo.daily}\\% din varianță')]),
    (T('A peak at @{lo.p168} hours (one week) and its harmonic at @{lo.p84} hours: working days against weekends', 'Un vîrf la @{lo.p168} ore (o săptămînă) și armonica lui la @{lo.p84} ore: zilele lucrătoare față de sfîrșitul de săptămînă'),
     [T('weekly components: @{lo.weekly}\\%; periods longer than a month (winter against summer): @{lo.long}\\%', 'componentele săptămînale: @{lo.weekly}\\%; perioadele mai lungi de o lună (iarna față de vară): @{lo.long}\\%')]),
    T('This is why Chapter 4 modelled the load with daily and weekly Fourier terms: the spectrum shows which periods to include', 'De aceea în Capitolul 4 am modelat consumul cu termeni Fourier zilnici și săptămînali: spectrul arată ce perioade trebuie incluse')])

D.recap(('Cycles in data', 'ciclurile din date'), [
    T('Remove trends first (growth rates, detrended cycles): levels have the typical spectral shape', 'Eliminați întîi trendurile (rate de creștere, cicluri fără trend): nivelurile au forma spectrală tipică'),
    T('US GDP: a business-cycle peak of about @{gs.US.c.yrs} years in the HP cycle; Romania: too short a sample for a sharp answer', 'PIB-ul SUA: un vîrf al ciclului economic de aproximativ @{gs.US.c.yrs} ani în ciclul HP; România: un eșantion prea scurt pentru un răspuns clar'),
    T('Electricity load: daily and weekly cycles with harmonics; the spectrum chooses the Fourier terms of a forecasting model', 'Consumul de energie electrică: cicluri zilnice și săptămînale cu armonice; spectrul alege termenii Fourier ai unui model de prognoză')])

# =============================================================================
# 6. MEMORIA LUNGĂ
# =============================================================================
D.section('Long memory: the pole at zero', 'Memoria lungă: polul de la zero')

D.frame(T('Long memory in the frequency domain', 'Memoria lungă în domeniul frecvenței'), items(
    (T('Chapter 8: a long-memory series has $\\rho(h) \\sim Ch^{2d-1}$; then $\\sum_h\\gamma(h) = \\infty$ and $f(0) = \\infty$', 'Capitolul 8: o serie cu memorie lungă are $\\rho(h) \\sim Ch^{2d-1}$; atunci $\\sum_h\\gamma(h) = \\infty$ și $f(0) = \\infty$'),
     [T('ARFIMA$(0,d,0)$: $f(\\nu) = \\sigma^2\\,|2\\sin(\\pi\\nu)|^{-2d} \\approx \\sigma^2(2\\pi\\nu)^{-2d}$ near $\\nu = 0$: a \\textbf{pole} at zero', 'ARFIMA$(0,d,0)$: $f(\\nu) = \\sigma^2\\,|2\\sin(\\pi\\nu)|^{-2d} \\approx \\sigma^2(2\\pi\\nu)^{-2d}$ în apropierea lui $\\nu = 0$: un \\textbf{pol} la zero')]),
    (T('On a log--log plot the low-frequency spectrum is a straight line with slope $-2d$', 'Pe un grafic log--log spectrul de joasă frecvență este o dreaptă cu panta $-2d$'),
     [T('\\refGPH\\ regress $\\log I(\\nu_j)$ on $-\\log(4\\sin^2(\\pi\\nu_j))$ for $j = 1, \\ldots, m$: the slope is $\\hat d$, standard error $\\pi/\\sqrt{24m}$', '\\refGPH\\ regresează $\\log I(\\nu_j)$ pe $-\\log(4\\sin^2(\\pi\\nu_j))$ pentru $j = 1, \\ldots, m$: panta este $\\hat d$, cu eroarea standard $\\pi/\\sqrt{24m}$'),
      T('only the lowest $m = \\lfloor T^{0.65}\\rfloor$ frequencies are used, where the pole dominates', 'se folosesc doar cele mai joase $m = \\lfloor T^{0.65}\\rfloor$ frecvențe, unde polul domină')]),
    T('Short memory (ARMA) has a finite $f(0)$: the log--log plot flattens at low frequencies', 'Memoria scurtă (ARMA) are $f(0)$ finit: graficul log--log devine orizontal la frecvențele joase')))

chart(T('The pole at zero in stock-market data', 'Polul de la zero în datele bursiere'), 'tsa_ch12_long_memory', 'TSA_ch12_long_memory', [
    T('Daily S\\&P 500 log returns, 2000--2026, $T = @{lm.T}$ (EODHD); periodogram averaged in 40 log-frequency bins; GPH lines on the lowest $m = @{lm.m}$ frequencies (periods above @{lm.per} days)', 'Randamentele logaritmice zilnice S\\&P 500, 2000--2026, $T = @{lm.T}$ (EODHD); periodograma mediată în 40 de intervale logaritmice de frecvență; dreptele GPH pe cele mai joase $m = @{lm.m}$ frecvențe (perioade de peste @{lm.per} de zile)')],
    h='0.58\\textheight')

interp(('the log--log periodogram', 'periodogramei log--log'), [
    (T('Returns $r_t$: flat at low frequencies, GPH $\\hat d = @{lm.r.d}$ (standard error @{lm.se}): no long memory in the returns', 'Randamentele $r_t$: spectru plat la frecvențele joase, GPH $\\hat d = @{lm.r.d}$ (eroarea standard @{lm.se}): nicio memorie lungă în randamente'),
     [T('a flat spectrum is what market efficiency predicts for returns (Chapter 1)', 'un spectru plat este ceea ce prevede eficiența pieței pentru randamente (Capitolul 1)')]),
    (T('Absolute returns $|r_t|$: power rises steadily towards $\\nu = 0$, slope $-2\\hat d$ with $\\hat d = @{lm.abs.d}$', 'Randamentele absolute $|r_t|$: puterea crește constant spre $\\nu = 0$, panta $-2\\hat d$, cu $\\hat d = @{lm.abs.d}$'),
     [T('the long memory of volatility (Chapter 8); $\\hat d$ near or above 0.5 also reflects breaks such as 2008 and 2020', 'memoria lungă a volatilității (Capitolul 8); un $\\hat d$ apropiat de 0,5 sau peste reflectă și rupturi precum 2008 și 2020')]),
    T('The same spectral picture explains why GPH and local Whittle (Chapter 8) are frequency-domain estimators', 'Aceeași imagine spectrală explică de ce GPH și Whittle local (Capitolul 8) sînt estimatori din domeniul frecvenței')])

# =============================================================================
# 7. FILTRE
# =============================================================================
D.section('Linear filters', 'Filtre liniare')

D.frame(T('Linear filters and their gain', 'Filtrele liniare și cîștigul lor'), items(
    (T('A \\textbf{linear filter}: $y_t = \\sum_j a_j x_{t-j}$; its \\textbf{frequency response} is $A(\\nu) = \\sum_j a_j e^{-2\\pi i\\nu j}$', 'Un \\textbf{filtru liniar}: $y_t = \\sum_j a_j x_{t-j}$; \\textbf{răspunsul lui în frecvență} este $A(\\nu) = \\sum_j a_j e^{-2\\pi i\\nu j}$'),
     [T('key result: $f_y(\\nu) = |A(\\nu)|^2 f_x(\\nu)$; $|A(\\nu)|^2$ is the \\textbf{squared gain}', 'rezultatul de bază: $f_y(\\nu) = |A(\\nu)|^2 f_x(\\nu)$; $|A(\\nu)|^2$ este \\textbf{cîștigul pătratic}'),
      T('the filter multiplies the power at each frequency: it amplifies some cycles and removes others', 'filtrul înmulțește puterea de la fiecare frecvență: amplifică unele cicluri și le elimină pe altele')]),
    (T('First difference $y_t = x_t - x_{t-1}$: $|A(\\nu)|^2 = |1 - e^{-2\\pi i\\nu}|^2 = 4\\sin^2(\\pi\\nu)$', 'Prima diferență $y_t = x_t - x_{t-1}$: $|A(\\nu)|^2 = |1 - e^{-2\\pi i\\nu}|^2 = 4\\sin^2(\\pi\\nu)$'),
     [T('zero at $\\nu = 0$ (removes the trend), 4 at $\\nu = 1/2$ (amplifies noise); at a period of 40 quarters: @{fi.d40}; at 4 quarters: @{fi.d4}', 'zero la $\\nu = 0$ (elimină trendul), 4 la $\\nu = 1/2$ (amplifică zgomotul); la o perioadă de 40 de trimestre: @{fi.d40}; la 4 trimestre: @{fi.d4}')]),
    (T('Seasonal difference $1 - L^4$: $4\\sin^2(4\\pi\\nu)$, zero at $\\nu = 0, 1/4, 1/2$ (the trend and the seasonal cycles)', 'Diferența sezonieră $1 - L^4$: $4\\sin^2(4\\pi\\nu)$, zero la $\\nu = 0, 1/4, 1/2$ (trendul și ciclurile sezoniere)'),
     [T('HP cycle filter: squared gain $\\dfrac{4\\lambda(1 - \\cos 2\\pi\\nu)^2}{1 + 4\\lambda(1 - \\cos 2\\pi\\nu)^2}$, a high-pass filter', 'filtrul HP pentru ciclu: cîștigul pătratic $\\dfrac{4\\lambda(1 - \\cos 2\\pi\\nu)^2}{1 + 4\\lambda(1 - \\cos 2\\pi\\nu)^2}$, un filtru trece-sus')])), size='footnotesize')

chart(T('Squared gains of common filters', 'Cîștigurile pătratice ale filtrelor uzuale'), 'tsa_ch12_filters', 'TSA_ch12_filters', [
    T('Left: first difference and seasonal difference ($s = 4$); right: HP cycle and trend filters, $\\lambda = 1600$ (quarterly data); shaded: 6--32 quarters', 'Stînga: prima diferență și diferența sezonieră ($s = 4$); dreapta: filtrele HP pentru ciclu și pentru trend, $\\lambda = 1600$ (date trimestriale); hașurat: 6--32 de trimestre')],
    h='0.6\\textheight')

interp(('the filter gains', 'cîștigurilor filtrelor'), [
    (T('Differencing is not neutral: it weakens business cycles and amplifies the highest frequencies', 'Diferențierea nu este neutră: slăbește ciclurile economice și amplifică frecvențele cele mai înalte'),
     [T('this is why growth rates look noisier than levels; filtering white noise can even create spurious cycles (the Slutsky effect, Chapter 0)', 'de aceea ratele de creștere par mai zgomotoase decît nivelurile; filtrarea unui zgomot alb poate crea chiar cicluri false (efectul Slutsky, Capitolul 0)')]),
    (T('HP cycle filter: gain @{fi.h8} at 8 quarters, @{fi.h32} at 32 quarters, @{fi.h40} at 40 quarters; half power at about @{fi.hhalf} quarters', 'Filtrul HP pentru ciclu: cîștig @{fi.h8} la 8 trimestre, @{fi.h32} la 32 de trimestre, @{fi.h40} la 40 de trimestre; jumătate din putere la aproximativ @{fi.hhalf} de trimestre'),
     [T('it keeps everything faster than 10 years, including noise: not a band-pass filter (\\refBK)', 'păstrează tot ce este mai rapid de 10 ani, inclusiv zgomotul: nu este un filtru trece-bandă (\\refBK)')]),
    T('Applied to a random walk, the HP filter produces a cycle with a peak that comes from the filter, not from the data (\\refHamHP)', 'Aplicat unui mers aleator, filtrul HP produce un ciclu cu un vîrf care provine din filtru, nu din date (\\refHamHP)')])

# =============================================================================
# 8. DOUĂ SERII
# =============================================================================
D.section('Two series: coherence and phase', 'Două serii: coerență și fază')

D.frame(T('Cross-spectrum, coherence and phase', 'Spectrul încrucișat, coerența și faza'), items(
    (T('For two stationary series with cross-covariances $\\gamma_{xy}(h) = \\mathrm{Cov}(x_{t+h}, y_t)$:', 'Pentru două serii staționare cu covarianțele încrucișate $\\gamma_{xy}(h) = \\mathrm{Cov}(x_{t+h}, y_t)$:'),
     [T('\\textbf{cross-spectrum} $f_{xy}(\\nu) = \\sum_h\\gamma_{xy}(h)e^{-2\\pi i\\nu h}$, a complex number at each frequency', '\\textbf{spectrul încrucișat} $f_{xy}(\\nu) = \\sum_h\\gamma_{xy}(h)e^{-2\\pi i\\nu h}$, un număr complex la fiecare frecvență')]),
    (T('\\textbf{Squared coherence}: $\\rho^2_{xy}(\\nu) = \\dfrac{|f_{xy}(\\nu)|^2}{f_x(\\nu)f_y(\\nu)} \\in [0, 1]$', '\\textbf{Coerența pătratică}: $\\rho^2_{xy}(\\nu) = \\dfrac{|f_{xy}(\\nu)|^2}{f_x(\\nu)f_y(\\nu)} \\in [0, 1]$'),
     [T('a squared correlation frequency by frequency: how well the cycles of $x$ and $y$ at $\\nu$ move together', 'un pătrat al corelației, frecvență cu frecvență: cît de bine se mișcă împreună ciclurile lui $x$ și $y$ la frecvența $\\nu$')]),
    (T('\\textbf{Phase} $\\varphi(\\nu) = \\arg f_{xy}(\\nu)$: the shift between the two cycles, in radians', '\\textbf{Faza} $\\varphi(\\nu) = \\arg f_{xy}(\\nu)$: decalajul dintre cele două cicluri, în radiani'),
     [T('a time lag of $k$ periods gives a phase that is linear in frequency: $\\varphi(\\nu) = \\pm 2\\pi\\nu k$; so the lag is $|\\varphi|/(2\\pi\\nu)$', 'un decalaj de $k$ perioade dă o fază liniară în frecvență: $\\varphi(\\nu) = \\pm 2\\pi\\nu k$; deci decalajul este $|\\varphi|/(2\\pi\\nu)$')]),
    T('Estimation: smooth the cross-periodogram like the periodogram; the raw coherence is always 1, so smoothing is compulsory', 'Estimarea: netezim periodograma încrucișată la fel ca periodograma; coerența brută este întotdeauna 1, deci netezirea este obligatorie')), size='footnotesize')

chart(T('Industrial production and unemployment', 'Producția industrială și șomajul'), 'tsa_ch12_coherence', 'TSA_ch12_coherence', [
    T('US monthly data, 1948--2019 (FRED INDPRO, UNRATE), $T = @{co.T}$: $x_t = 100\\,\\Delta\\ln \\mathrm{IP}_t$ and $y_t = -\\Delta u_t$ (the fall in the unemployment rate); Welch with $K = @{co.K}$ segments of 96 months', 'Date lunare din SUA, 1948--2019 (FRED INDPRO, UNRATE), $T = @{co.T}$: $x_t = 100\\,\\Delta\\ln \\mathrm{IP}_t$ și $y_t = -\\Delta u_t$ (scăderea ratei șomajului); Welch cu $K = @{co.K}$ segmente de 96 de luni')],
    h='0.56\\textheight')

interp(('coherence and phase', 'coerenței și fazei'), [
    (T('In the business-cycle band (18--96 months) the squared coherence averages @{co.bc} (maximum @{co.bcmax}), far above the 5\\% threshold @{co.thr}', 'În banda ciclului economic (18--96 de luni) coerența pătratică are media @{co.bc} (maximum @{co.bcmax}), mult peste pragul de 5\\%, @{co.thr}'),
     [T('at frequencies faster than 6 months it averages only @{co.hi}: the monthly noise of the two series is unrelated', 'la frecvențele mai rapide de 6 luni media este doar @{co.hi}: zgomotul lunar al celor două serii nu este legat'),
      T('the ordinary correlation, @{co.r0}, mixes the strong link of the cycles with the weak link of the noise', 'corelația obișnuită, @{co.r0}, amestecă legătura puternică a ciclurilor cu legătura slabă a zgomotului')]),
    (T('Phase at the 48-month cycle: @{co.ph} radians, a lag of about @{co.lag} months', 'Faza la ciclul de 48 de luni: @{co.ph} radiani, un decalaj de aproximativ @{co.lag} luni'),
     [T('the fall in unemployment follows the recovery of production with a short delay (Okun\'s law, with a lag)', 'scăderea șomajului urmează revenirea producției cu o mică întîrziere (legea lui Okun, cu decalaj)')]),
    T('Where the coherence is low, the phase is meaningless: read the phase only in bands with significant coherence', 'Acolo unde coerența este mică, faza nu are sens: citiți faza doar în benzile cu coerență semnificativă')])

# =============================================================================
# 9. WAVELETS
# =============================================================================
D.section('A pointer: wavelets', 'O trimitere: wavelets')

D.frame(T('Wavelets: frequency that changes over time', 'Wavelets: frecvența care se schimbă în timp'), items(
    (T('The spectrum assumes stationarity: the same cycles at every date', 'Spectrul presupune staționaritate: aceleași cicluri la orice dată'),
     [T('the \\textbf{short-time Fourier transform} (STFT) computes spectra in moving windows of fixed length', '\\textbf{transformata Fourier pe ferestre scurte} (STFT) calculează spectre în ferestre mobile de lungime fixă')]),
    (T('A \\textbf{wavelet} is a short wave packet; the \\textbf{continuous wavelet transform} (CWT) stretches it to each period and slides it along the series', 'Un \\textbf{wavelet} este un pachet scurt de unde; \\textbf{transformata wavelet continuă} (CWT) îl întinde la fiecare perioadă și îl glisează de-a lungul seriei'),
     [T('the result, the \\textbf{scalogram}, shows the power at each period and each date (\\refTC)', 'rezultatul, \\textbf{scalograma}, arată puterea la fiecare perioadă și la fiecare dată (\\refTC)'),
      T('short windows for fast cycles, long windows for slow cycles: good resolution at all scales', 'ferestre scurte pentru ciclurile rapide, ferestre lungi pentru ciclurile lente: rezoluție bună la toate scările')]),
    T('Beyond this course; a natural project topic (Python: \\texttt{pywt}, or the Morlet code of this chapter)', 'Dincolo de acest curs; o temă naturală de proiect (Python: \\texttt{pywt} sau codul Morlet al acestui capitol)')))

chart(T('Wavelet power of the sunspots', 'Puterea wavelet a petelor solare'), 'tsa_ch12_wavelet', 'TSA_ch12_wavelet', [
    T('Morlet wavelet power of the standardised yearly sunspot numbers, 1700--2008, periods 2 to 64 years; bright colours mean high power', 'Puterea wavelet Morlet a numerelor anuale standardizate de pete solare, 1700--2008, perioade între 2 și 64 de ani; culorile deschise indică putere mare')],
    h='0.56\\textheight')

interp(('the scalogram', 'scalogramei'), [
    (T('The 11-year band is present throughout, but its strength changes', 'Banda de 11 ani este prezentă tot timpul, dar intensitatea ei se schimbă'),
     [T('average power in the 8--14-year band: @{wv.weak} in 1795--1830 (the weak Dalton minimum) against @{wv.all} over the whole sample; the maximum is around @{wv.maxy}', 'puterea medie în banda de 8--14 ani: @{wv.weak} în 1795--1830 (minimul Dalton, cu cicluri slabe), față de @{wv.all} pe întregul eșantion; maximul este în jurul anului @{wv.maxy}')]),
    T('The periodogram averaged these changes into one peak; the scalogram shows when the cycle was strong or weak', 'Periodograma a mediat aceste schimbări într-un singur vîrf; scalograma arată cînd ciclul a fost puternic sau slab'),
    T('Edges of the chart are less reliable (the wavelet runs out of data): this region is called the cone of influence', 'Marginile graficului sînt mai puțin sigure (wavelet-ul rămîne fără date): această regiune se numește conul de influență')])

D.recap(('Long memory, filters, two series, wavelets', 'memoria lungă, filtre, două serii, wavelets'), [
    T('Long memory is a pole at $\\nu = 0$: slope $-2d$ on a log--log plot (GPH)', 'Memoria lungă este un pol la $\\nu = 0$: panta $-2d$ pe un grafic log--log (GPH)'),
    T('A filter multiplies the spectrum by its squared gain: differencing and the HP filter reshape cycles', 'Un filtru înmulțește spectrul cu cîștigul lui pătratic: diferențierea și filtrul HP remodelează ciclurile'),
    T('Coherence is a correlation by frequency; the phase gives the lead or the lag', 'Coerența este o corelație pe frecvențe; faza arată avansul sau întîrzierea'),
    T('Wavelets show how cycles change over time', 'Wavelets arată cum se schimbă ciclurile în timp')])

# =============================================================================
# 10. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Code}: a first draft of a periodogram, a Daniell smoother, a Welch estimate or a coherence plot', '\\textbf{Cod}: o primă versiune a unei periodograme, a unei neteziri Daniell, a unei estimări Welch sau a unui grafic al coerenței'),
    T('\\textbf{Explanation}: a second explanation of leakage, aliasing or the bias--variance trade-off of the bandwidth', '\\textbf{Explicații}: o a doua explicație a scurgerii spectrale, a fenomenului de aliasing sau a compromisului deplasare--varianță al lățimii de bandă'),
    T('\\textbf{Exploration}: spectra of many series at once (load of several countries, GDP of all EU members) and a first reading of the peaks', '\\textbf{Explorare}: spectrele multor serii deodată (consumul mai multor țări, PIB-ul tuturor statelor UE) și o primă interpretare a vîrfurilor'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that reads hourly electricity load, removes the mean, computes the periodogram with a Hann taper and a Welch estimate with 8-week segments, plots both on a log-log scale against the period in hours, and marks 24, 12 and 168 hours.}',
        '\\aiprompt{Write Python code that reads hourly electricity load, removes the mean, computes the periodogram with a Hann taper and a Welch estimate with 8-week segments, plots both on a log-log scale against the period in hours, and marks 24, 12 and 168 hours.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('The units: frequency in cycles per observation or radians; \\texttt{scipy.signal.welch} returns a one-sided density (twice our $f$)', 'Unitățile: frecvența în cicluri pe observație sau în radiani; \\texttt{scipy.signal.welch} întoarce o densitate unilaterală (dublul lui $f$)'),
    T('The period: $1/\\nu$ in observations; convert to years, quarters or hours yourself', 'Perioada: $1/\\nu$ în observații; transformați-o singuri în ani, trimestre sau ore'),
    T("Parseval: the ordinates must add up to $T$ times the variance; test the code on a cosine with a known frequency", 'Parseval: ordonatele trebuie să se adune la $T$ înmulțit cu varianța; testați codul pe un cosinus cu frecvență cunoscută'),
    T('A peak is a cycle only with a test or a confidence band; a raw periodogram always has tall spikes', 'Un vîrf este un ciclu doar cu un test sau cu o bandă de încredere; o periodogramă brută are întotdeauna vîrfuri înalte'),
    T('Trends, breaks and filters create low-frequency power and false peaks: check the data and the transformation first', 'Trendurile, rupturile și filtrele creează putere la frecvențele joase și vîrfuri false: verificați întîi datele și transformarea'),
    T('Every cited reference: it must exist; check the DOI', 'Fiecare referință citată: trebuie să existe; verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('The spectrum splits the variance of a stationary series by frequency; it carries the same information as the ACF', 'Spectrul împarte varianța unei serii staționare pe frecvențe; conține aceeași informație ca ACF'),
    T('Read its shape: flat (white noise), falling (persistence), peaked (cycles), a pole at zero (long memory)', 'Interpretați forma lui: plat (zgomot alb), descrescător (persistență), cu vîrfuri (cicluri), cu pol la zero (memorie lungă)'),
    T('The raw periodogram is unbiased but inconsistent: taper, smooth and add confidence bands', 'Periodograma brută este nedeplasată, dar neconsistentă: aplicați atenuarea, netezirea și benzile de încredere'),
    T('Real cycles: 11 years in sunspots, 4--8 years in GDP, 24 hours and 1 week in electricity load', 'Cicluri reale: 11 ani la petele solare, 4--8 ani în PIB, 24 de ore și o săptămînă în consumul de energie electrică'),
    T('Filters reshape spectra; coherence and phase relate the cycles of two series', 'Filtrele remodelează spectrele; coerența și faza leagă ciclurile a două serii')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.35}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Spectral density', 'Densitatea spectrală') + ' & $f(\\nu) = \\sum_h\\gamma(h)e^{-2\\pi i\\nu h}$, \\quad $\\gamma(0) = \\int_{-1/2}^{1/2} f(\\nu)\\,d\\nu$',
     'AR(1), MA(1) & $\\sigma^2/(1 - 2\\phi\\cos 2\\pi\\nu + \\phi^2)$, \\quad $\\sigma^2(1 + 2\\theta\\cos 2\\pi\\nu + \\theta^2)$',
     'ARMA & $\\sigma^2|\\theta(e^{-2\\pi i\\nu})|^2/|\\phi(e^{-2\\pi i\\nu})|^2$',
     T('Periodogram', 'Periodograma') + ' & $I(\\nu_j) = |T^{-1/2}\\sum_t x_t e^{-2\\pi i\\nu_j t}|^2$, \\quad $\\nu_j = j/T$, \\quad $2I/f \\approx \\chi^2_2$',
     'Daniell & $\\hat f(\\nu_j) = \\frac1L\\sum_{|k| \\le m} I(\\nu_{j+k})$, \\quad $L = 2m + 1$, \\quad $B = L/T$',
     T('95\\% band', 'Banda de 95\\%') + ' & $[2L\\hat f/\\chi^2_{2L}(0.975),\\ 2L\\hat f/\\chi^2_{2L}(0.025)]$',
     T('Linear filter', 'Filtru liniar') + ' & $f_y(\\nu) = |A(\\nu)|^2 f_x(\\nu)$, \\quad $|1 - e^{-2\\pi i\\nu}|^2 = 4\\sin^2(\\pi\\nu)$',
     T('Coherence, phase', 'Coerența, faza') + ' & $\\rho^2_{xy} = |f_{xy}|^2/(f_x f_y)$, \\quad $\\varphi = \\arg f_{xy}$, \\quad ' + T('lag', 'decalaj') + ' $= |\\varphi|/(2\\pi\\nu)$'],
    size='scriptsize') + '}')

D.frame(T('Self-assessment (1/2)', 'Autoevaluare (1/2)'), items(
    (T('\\textbf{Question}: weekly data show a peak at $\\nu = 0.0192$. What is the period?', '\\textbf{Întrebare}: niște date săptămînale au un vîrf la $\\nu = 0.0192$. Care este perioada?'),
     [T('\\textbf{Answer}: $1/0.0192 \\approx 52$ weeks: an annual cycle', '\\textbf{Răspuns}: $1/0.0192 \\approx 52$ de săptămîni: un ciclu anual')]),
    (T('\\textbf{Question}: what are $f(0)$ and $f(1/2)$ of an AR(1) with $\\phi = -0.5$, $\\sigma^2 = 1$?', '\\textbf{Întrebare}: cît sînt $f(0)$ și $f(1/2)$ pentru un AR(1) cu $\\phi = -0.5$, $\\sigma^2 = 1$?'),
     [T('\\textbf{Answer}: $f(0) = 1/1.5^2 = 0.44$ and $f(1/2) = 1/0.5^2 = 4$: power at high frequencies', '\\textbf{Răspuns}: $f(0) = 1/1.5^2 = 0.44$ și $f(1/2) = 1/0.5^2 = 4$: puterea este la frecvențele înalte')]),
    (T('\\textbf{Question}: why does a longer sample not make the raw periodogram more precise?', '\\textbf{Întrebare}: de ce un eșantion mai lung nu face periodograma brută mai precisă?'),
     [T('\\textbf{Answer}: each ordinate still has variance about $f^2$; more data only adds ordinates; precision needs averaging', '\\textbf{Răspuns}: fiecare ordonată are în continuare varianța aproximativ $f^2$; mai multe date adaugă doar ordonate; precizia cere o mediere')]),
    (T('\\textbf{Question}: with $L = 5$, how wide is the 95\\% band relative to $\\hat f$?', '\\textbf{Întrebare}: pentru $L = 5$, cît de largă este banda de 95\\% în raport cu $\\hat f$?'),
     [T('\\textbf{Answer}: from @{ex.ci.lo} to @{ex.ci.hi} times $\\hat f$ ($df = 10$)', '\\textbf{Răspuns}: de la @{ex.ci.lo} la @{ex.ci.hi} ori $\\hat f$ ($df = 10$)')])), size='footnotesize')

D.frame(T('Self-assessment (2/2)', 'Autoevaluare (2/2)'), items(
    (T('\\textbf{Question}: a monthly series observed only every second month has a 3-month cycle. Where does it appear?', '\\textbf{Întrebare}: o serie lunară observată doar o dată la două luni are un ciclu de 3 luni. Unde apare acesta?'),
     [T('\\textbf{Answer}: $\\nu = 2/3$ cycles per two months is above $1/2$; it folds to $1/3$: a false cycle of 3 observations, i.e.\\ 6 months', '\\textbf{Răspuns}: $\\nu = 2/3$ cicluri la două luni este peste $1/2$; se pliază la $1/3$: un ciclu fals de 3 observații, adică 6 luni')]),
    (T('\\textbf{Question}: what does the first difference do to a cycle of 40 quarters?', '\\textbf{Întrebare}: ce efect are prima diferență asupra unui ciclu de 40 de trimestre?'),
     [T('\\textbf{Answer}: it multiplies its power by $4\\sin^2(\\pi/40) = @{fi.d40}$: the cycle almost disappears', '\\textbf{Răspuns}: îi înmulțește puterea cu $4\\sin^2(\\pi/40) = @{fi.d40}$: ciclul aproape dispare')]),
    (T('\\textbf{Question}: the log--log periodogram of a series has slope $-0.6$ near zero. What is $d$?', '\\textbf{Întrebare}: periodograma log--log a unei serii are panta $-0.6$ în apropierea lui zero. Cît este $d$?'),
     [T('\\textbf{Answer}: slope $= -2d$, so $d = 0.3$: stationary long memory (Chapter 8)', '\\textbf{Răspuns}: panta $= -2d$, deci $d = 0.3$: memorie lungă staționară (Capitolul 8)')]),
    (T('\\textbf{Question}: the squared coherence of two series is 0.9 at 5-year cycles and 0.1 at monthly cycles. What does it mean?', '\\textbf{Întrebare}: coerența pătratică a două serii este 0,9 la ciclurile de 5 ani și 0,1 la ciclurile lunare. Ce înseamnă aceasta?'),
     [T('\\textbf{Answer}: their business cycles move together; their short-run noise does not', '\\textbf{Răspuns}: ciclurile lor economice se mișcă împreună; zgomotul lor pe termen scurt nu')]),
    T('Next: Chapter 13, speculative bubbles and LPPL models', 'Urmează: Capitolul 13, bule speculative și modele LPPL')), size='footnotesize')

D.references(bib(), per=9)

if __name__ == '__main__':
    finalize(D.write(V))
