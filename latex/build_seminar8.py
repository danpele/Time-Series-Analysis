r"""
build_seminar8.py -- Seminarul 8 (Memorie lungă și ARFIMA), EN + RO dintr-o singură sursă
=========================================================================================
Seminarul are loc ÎNAINTEA cursului 8: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu o întrebare de interpretare, C o întrebare deschisă și
critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în versiunea
profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_08/sem8_results.json (seminar8.py).
Ieșire:
  EN/Seminars/seminar8_long_memory_arfima.tex          (+ _solutions.tex)
  RO/Seminarii/seminar8_memorie_lunga_arfima_ro.tex    (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_08/seminar8.py && python3 latex/build_seminar8.py && python3 latex/tsa_build.py compile 8
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, fig   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch8_common import REFS, T, bib, finalize, load_sem   # noqa: E402

S = load_sem()
V = Values()
D = Deck(8, 'seminar', refs=REFS)
P = V.put


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


def qlsem():
    return '\\quantlet{TSA\\_ch8\\_seminar}{\\qlurl{TSA_ch8_seminar}}'


# =============================================================================
# CIFRE
# =============================================================================
A1 = S['A1']
for i, w in enumerate(A1['pi'][:4], 1):
    P(f'a1.p{i}', w, 4)
for k in ('rho1', 'rho2', 'rho5', 'rho10'):
    P(f'a1.{k}', A1[k], 3)
P('a1.ar5', A1['ar5'], 4)
P('a1.var', A1['var'], 3)
A2a, A2b = S['A2a'], S['A2b']
P('a2a.p1', A2a['pi'][0], 2)
P('a2a.p2', A2a['pi'][1], 3)
P('a2b.p1', A2b['pi'][0], 2)
P('a2b.p2', A2b['pi'][1], 3)
P('a2b.r1', A2b['rho1'], 3)
P('a2b.r2', A2b['rho2'], 3)
A2c = S['A2c']
P('a2c.r1', A2c['rho1'], 3)
A3 = S['A3']
P('a3.mean', A3['mean'], 4)
P('a3.R', A3['R'], 3)
P('a3.S', A3['S'], 3)
P('a3.RS', A3['RS'], 2)
for i, y in enumerate(A3['Y'], 1):
    P(f'a3.y{i}', y + 0.0, 4)
AP = S['A3pts']
for i in range(3):
    P(f'ap.ln{i}', AP['log_n'][i], 3)
    P(f'ap.lr{i}', AP['log_rs'][i], 3)
P('ap.H', AP['H'], 2)
P('ap.d', AP['H'] - 0.5, 2)
A4 = S['A4']
for k in ('xbar', 'ybar', 'sxy', 'sxx', 'd', 'se', 't'):
    P(f'a4.{k}', A4[k], 3)
A5 = S['A5']
for i, t in enumerate(A5['terms']):
    P(f'a5.t{i}', t, 4)
P('a5.fa', A5['f_arfima'], 3)
P('a5.fr', A5['f_ar1'], 3)
P('a5.phi', A5['phi'], 3)
A6 = S['A6']
for k in ('1', '2', '3', '20', '100'):
    P(f'a6.g{k}', A6[k]['garch'], 4)
    P(f'a6.f{k}', A6[k]['figarch'], 4)
    P(f'a6.a{k}', A6[k]['approx'], 4)
V.raw('a6.g100', '⁅3.2⁆ \\cdot 10^{-7}')
P('a6.f100x', A6['100']['figarch'] * 1e4, 1)
P('a6.ratio', A6['100']['figarch'] / A6['100']['garch'], 0)

B1 = S['B1']
P('b1.pre', B1['pre'], 0)
P('b1.post', B1['post'], 0)
for k in ('raw', 'adj'):
    for c in ('r1', 'r10', 'H', 'gph', 'gph_se', 'lw', 'lw_se', 'ml', 'ml_se'):
        P(f'b1.{k}.{c}', B1[k][c], 2)
V.raw('b1.m', str(B1['raw']['m']))
B2 = S['B2']
V.raw('b2.T', str(B2['T']))
for a in ('0.5', '0.65', '0.8'):
    k = a.replace('0.', '')
    V.raw(f'b2.m{k}', str(B2['bw'][a]['m']))
    P(f'b2.g{k}', B2['bw'][a]['gph'], 2)
    P(f'b2.l{k}', B2['bw'][a]['lw'], 2)
    P(f'b2.s{k}', B2['bw'][a]['lw_se'], 2)
for k in ('ml_d', 'ml_se', 'a_phi', 'a_theta', 'lw12', 'gph12', 'r1', 'r1_12'):
    P(f'b2.{k}', B2[k], 2)
P('b2.ml_se', B2['ml_se'], 3)
P('b2.bicf', B2['bic_f'], 1)
P('b2.bica', B2['bic_a'], 1)
B3 = S['B3']
V.int('b3.T', B3['T'])
V.raw('b3.Trv', str(B3['Trv']))
for k in ('r', 'abs', 'shuf', 'rv'):
    for c in ('lw', 'lw_se', 'gph', 'gph_se'):
        P(f'b3.{k}.{c}', B3[k][c], 2)
for k in ('1', '50', '200'):
    P(f'b3.acf{k}', B3['acf_abs'][k], 3)
B4 = S['B4']
for a in ('bet', 'eurron'):
    for s in ('full', '2000-2007', '2010-2026'):
        k = s.replace('-', '')
        V.int(f'b4.{a}.{k}.n', B4[a][s]['n'])
        P(f'b4.{a}.{k}.r', B4[a][s]['r'], 2)
        P(f'b4.{a}.{k}.abs', B4[a][s]['abs'], 2)
P('b4.lo', B4['band']['q025'], 2)
P('b4.hi', B4['band']['q975'], 2)
C1 = S['C1']
for s in ('full', '1947-1984', '1985-2019', '2020-2026', 'adj'):
    k = s.replace('-', '')
    P(f'c1.{k}.lw', C1[s]['lw'], 2)
    if 'mean' in C1[s]:
        P(f'c1.{k}.mean', C1[s]['mean'], 2)
    if 'n' in C1[s]:
        V.raw(f'c1.{k}.n', str(C1[s]['n']))
C2 = S['C2']
for k in ('ml', 'lw12', 'sp_r', 'sp_abs'):
    P(f'c2.{k}', C2[k], 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: does a shock to a series fade at an exponential rate (ARMA) or much more slowly (long memory), and how do we measure the difference?',
       '\\textbf{Întrebarea}: se stinge un șoc într-o serie exponențial (ARMA) sau mult mai lent (memorie lungă) și cum măsurăm diferența?'),
     [T('this seminar comes \\textbf{before} Lecture 8: the section ``What you need today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 8: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: fractional weights and ARFIMA autocorrelations; R/S and GPH by hand; an ARFIMA forecast; GARCH and FIGARCH weights',
        'Partea A: ponderi fracționare și autocorelații ARFIMA; R/S și GPH calculate de mînă; o prognoză ARFIMA; ponderile GARCH și FIGARCH'),
      T('Part B: the Nile and its break; Romanian inflation; S\\&P 500 volatility; BET and EUR/RON', 'Partea B: Nilul și ruptura lui; inflația din România; volatilitatea S\\&P 500; BET și EUR/RON'),
      T('Part C: US inflation and monetary regimes; an AI answer to audit', 'Partea C: inflația din SUA și regimurile monetare; un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('weights of $(1-L)^d$, ACF of ARFIMA$(0,d,0)$, ranges of $d$', 'ponderile lui $(1-L)^d$, ACF a unui ARFIMA$(0,d,0)$, intervalele lui $d$') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('R/S and $H$ by hand; a GPH slope by hand', 'R/S și $H$ de mînă; o pantă GPH de mînă') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('an ARFIMA forecast; GARCH and FIGARCH weights', 'o prognoză ARFIMA; ponderile GARCH și FIGARCH') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('the Nile and its break; Romanian inflation', 'Nilul și ruptura lui; inflația din România') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('S\\&P 500 volatility; BET and EUR/RON', 'volatilitatea S\\&P 500; BET și EUR/RON') + ' & ' + SP + ' & B3',
     'C1, C2 & ' + T('US inflation and regimes; what is wrong in an AI answer?', 'inflația din SUA și regimurile; ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, B2'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați dumneavoastră, după model')))

D.frame(T('Data used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Frequency}', '\\textbf{Frecvența}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    [T('Nile flow at Aswan', 'Debitul Nilului la Aswan') + ' & statsmodels (nile) & ' + T('yearly', 'anual') + ' & 1871--1970',
     T('HICP, Romania', 'IAPC, România') + ' & Eurostat (prc\\_hicp\\_minr) & ' + T('monthly, 2015 = 100', 'lunar, 2015 = 100') + ' & 2005--2026',
     T('CPI, United States', 'IPC, Statele Unite') + ' & FRED (CPIAUCSL) & ' + T('monthly, SA', 'lunar, ajustat sezonier') + ' & 1947--2026',
     'S\\&P 500, BET & EODHD & ' + T('daily, close', 'zilnic, închidere') + ' & 2000--2026',
     'EUR/RON & ' + T('BNR reference rate', 'cursul de referință BNR') + ' & ' + T('daily', 'zilnic') + ' & 2005--2026'],
    size='footnotesize') + items(
    T('Romanian inflation: monthly rate $100\\Delta\\ln P_t$ minus its monthly means (seasonal adjustment), and the 12-month rate $100(\\ln P_t - \\ln P_{t-12})$',
      'Inflația din România: rata lunară $100\\Delta\\ln P_t$ din care se scad mediile lunare (ajustare sezonieră) și rata anuală $100(\\ln P_t - \\ln P_{t-12})$'),
    T('In the notebook: \\texttt{gph}, \\texttt{local\\_whittle}, \\texttt{hurst\\_rs}, \\texttt{arfima\\_exact\\_ml}, \\texttt{monthly\\_rv}; no account or key is needed',
      'În notebook: \\texttt{gph}, \\texttt{local\\_whittle}, \\texttt{hurst\\_rs}, \\texttt{arfima\\_exact\\_ml}, \\texttt{monthly\\_rv}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What you need today', 'Noțiuni necesare azi')

D.frame(T('What you need today (1/4): short and long memory', 'Noțiuni necesare azi (1/4): memorie scurtă și memorie lungă'), items(
    (T('\\textbf{Short memory} (ARMA, Chapter 2): $|\\rho(k)| \\le Cr^k$, $0 < r < 1$; the autocorrelations are summable', '\\textbf{Memoria scurtă} (ARMA, Capitolul 2): $|\\rho(k)| \\le Cr^k$, $0 < r < 1$; autocorelațiile sînt sumabile'),
     [T('AR(1): $\\rho(k) = \\phi^k$', 'AR(1): $\\rho(k) = \\phi^k$')]),
    (T('\\textbf{Long memory}: $\\rho(k) \\sim Ck^{2d-1}$, $0 < d < 1/2$: hyperbolic decay, $\\sum_k\\rho(k) = \\infty$', '\\textbf{Memoria lungă}: $\\rho(k) \\sim Ck^{2d-1}$, $0 < d < 1/2$: descreștere hiperbolică, $\\sum_k\\rho(k) = \\infty$'),
     [T('equivalently, the spectral density has a pole at zero: $f(\\lambda) \\sim G\\lambda^{-2d}$ as $\\lambda \\to 0$', 'echivalent, densitatea spectrală are un pol la zero: $f(\\lambda) \\sim G\\lambda^{-2d}$ cînd $\\lambda \\to 0$')]),
    (T('\\textbf{Hurst exponent} $H = d + 1/2$: $H = 0.5$ no memory, $H > 0.5$ persistence, $H < 0.5$ anti-persistence', '\\textbf{Exponentul Hurst} $H = d + 1/2$: $H = 0{,}5$ fără memorie, $H > 0{,}5$ persistență, $H < 0{,}5$ antipersistență'),
     [T('Hurst found $H \\approx 0.7$ for the Nile \\refHurst', 'Hurst a găsit $H \\approx 0{,}7$ pentru Nil \\refHurst')])))

D.frame(T('What you need today (2/4): fractional differencing and ARFIMA', 'Noțiuni necesare azi (2/4): diferențierea fracționară și ARFIMA'), items(
    (T('$(1-L)^d = \\sum_k\\pi_kL^k$ with $\\pi_0 = 1$, $\\pi_k = \\pi_{k-1}(k-1-d)/k$; $(1-L)^{-d}$: $\\psi_k = \\psi_{k-1}(k-1+d)/k$', '$(1-L)^d = \\sum_k\\pi_kL^k$, cu $\\pi_0 = 1$, $\\pi_k = \\pi_{k-1}(k-1-d)/k$; $(1-L)^{-d}$: $\\psi_k = \\psi_{k-1}(k-1+d)/k$'),
     [T('\\textbf{ARFIMA}$(p,d,q)$: $\\phi(L)(1-L)^d(x_t - \\mu) = \\theta(L)\\varepsilon_t$ \\refGJ, \\refHosking', '\\textbf{ARFIMA}$(p,d,q)$: $\\phi(L)(1-L)^d(x_t - \\mu) = \\theta(L)\\varepsilon_t$ \\refGJ, \\refHosking')]),
    (T('Ranges: invertible for $d > -1/2$; stationary for $d < 1/2$; mean-reverting for $d < 1$; $d = 1$ unit root', 'Intervale: inversabil pentru $d > -1/2$; staționar pentru $d < 1/2$; cu revenire la medie pentru $d < 1$; $d = 1$ rădăcină unitară'),
     [T('if $1/2 \\le d < 3/2$, difference once: $(1-L)x_t$ is ARFIMA with $d - 1$', 'dacă $1/2 \\le d < 3/2$, diferențiem o dată: $(1-L)x_t$ este ARFIMA cu $d - 1$')]),
    (T('ARFIMA$(0,d,0)$: $\\rho(1) = d/(1-d)$, $\\rho(k) = \\rho(k-1)(k-1+d)/(k-d)$, $\\gamma(0) = \\sigma^2\\Gamma(1-2d)/\\Gamma(1-d)^2$', 'ARFIMA$(0,d,0)$: $\\rho(1) = d/(1-d)$, $\\rho(k) = \\rho(k-1)(k-1+d)/(k-d)$, $\\gamma(0) = \\sigma^2\\Gamma(1-2d)/\\Gamma(1-d)^2$'),
     [T('one-step forecast: $\\hat x_{T+1} = \\mu - \\sum_{k\\ge1}\\pi_k(x_{T+1-k} - \\mu)$', 'prognoza la un pas: $\\hat x_{T+1} = \\mu - \\sum_{k\\ge1}\\pi_k(x_{T+1-k} - \\mu)$')])))

D.frame(T('What you need today (3/4): estimators of $d$ and $H$', 'Noțiuni necesare azi (3/4): estimatori ai lui $d$ și $H$'), items(
    (T('\\textbf{R/S}: in a block, $Y_k = \\sum_{i\\le k}(x_i - \\bar x)$, $R = \\max Y_k - \\min Y_k$, $S$ the standard deviation (divisor $n$)', '\\textbf{R/S}: într-un bloc, $Y_k = \\sum_{i\\le k}(x_i - \\bar x)$, $R = \\max Y_k - \\min Y_k$, $S$ abaterea standard (împărțitor $n$)'),
     [T('$H$ = slope of $\\log_{10}(R/S)_n$ on $\\log_{10}n$; \\textbf{DFA}: slope of the detrended fluctuation $\\log F(n)$', '$H$ = panta lui $\\log_{10}(R/S)_n$ față de $\\log_{10}n$; \\textbf{DFA}: panta fluctuației fără tendință $\\log F(n)$')]),
    (T('\\textbf{GPH}: OLS of $\\log I(\\lambda_j)$ on $X_j = -\\log(4\\sin^2(\\lambda_j/2))$, $j = 1..m$; slope $= \\hat d$, SE $= \\pi/\\sqrt{24m}$', '\\textbf{GPH}: OLS a lui $\\log I(\\lambda_j)$ pe $X_j = -\\log(4\\sin^2(\\lambda_j/2))$, $j = 1..m$; panta $= \\hat d$, SE $= \\pi/\\sqrt{24m}$'),
     [T('$I(\\lambda_j)$: periodogram at $\\lambda_j = 2\\pi j/T$; OLS slope $= \\sum(X_j - \\bar X)(y_j - \\bar y)/\\sum(X_j - \\bar X)^2$', '$I(\\lambda_j)$: periodograma la $\\lambda_j = 2\\pi j/T$; panta OLS $= \\sum(X_j - \\bar X)(y_j - \\bar y)/\\sum(X_j - \\bar X)^2$')]),
    (T('\\textbf{Local Whittle} \\refRob: a local likelihood on the same $m$ frequencies, SE $= 1/(2\\sqrt m)$; bandwidth $m = \\lfloor T^{0.65}\\rfloor$', '\\textbf{Whittle local} \\refRob: o verosimilitate locală pe aceleași $m$ frecvențe, SE $= 1/(2\\sqrt m)$; lățimea de bandă $m = \\lfloor T^{0{,}65}\\rfloor$'),
     [T('\\textbf{Exact ML} \\refSowell: the full ARFIMA$(p,d,q)$ likelihood; compare models by AIC and BIC', '\\textbf{ML exactă} \\refSowell: verosimilitatea completă ARFIMA$(p,d,q)$; comparăm modelele după AIC și BIC')])))

D.frame(T('What you need today (4/4): volatility and spurious memory', 'Noțiuni necesare azi (4/4): volatilitate și memorie aparentă'), items(
    (T('Returns $r_t$ are almost uncorrelated; $|r_t|$ and realised volatility have long memory \\refDGE', 'Randamentele $r_t$ sînt aproape necorelate; $|r_t|$ și volatilitatea realizată au memorie lungă \\refDGE'),
     [T('monthly \\textbf{realised volatility}: $\\sqrt{RV_m}$, $RV_m = \\sum_{t\\in m} r_t^2$; we model its logarithm', '\\textbf{volatilitatea realizată} lunară: $\\sqrt{RV_m}$, $RV_m = \\sum_{t\\in m} r_t^2$; modelăm logaritmul ei'),
      T('\\textbf{shuffle test}: permute the days at random; real memory disappears, the distribution stays', '\\textbf{testul permutării}: permutăm zilele aleator; memoria reală dispare, distribuția rămîne')]),
    (T('GARCH(1,1) as ARCH($\\infty$): weight $\\alpha\\beta^{k-1}$ on $\\varepsilon_{t-k}^2$; FIGARCH \\refBBM: weights $\\approx k^{-1-d}$', 'GARCH(1,1) ca ARCH($\\infty$): ponderea $\\alpha\\beta^{k-1}$ pentru $\\varepsilon_{t-k}^2$; FIGARCH \\refBBM: ponderi $\\approx k^{-1-d}$'),
     [T('with $\\phi = \\beta = 0$, FIGARCH weights are $\\lambda_k = -\\pi_k$ of $(1-L)^d$', 'cu $\\phi = \\beta = 0$, ponderile FIGARCH sînt $\\lambda_k = -\\pi_k$ ale lui $(1-L)^d$')]),
    T('\\textbf{Spurious long memory}: level shifts and rare regime switches give $\\hat d > 0$ for short-memory series \\refDI; re-estimate after removing the breaks', '\\textbf{Memoria lungă aparentă}: schimbările de nivel și schimbările rare de regim dau $\\hat d > 0$ pentru serii cu memorie scurtă \\refDI; reestimați după eliminarea rupturilor')))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: computations on paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: fractional weights and the ACF', 'A1: ponderi fracționare și ACF'),
         items(T('A series is ARFIMA$(0, 0.25, 0)$ with $\\sigma^2 = 1$.', 'O serie este ARFIMA$(0;\\ 0{,}25;\\ 0)$, cu $\\sigma^2 = 1$.'),
               T('1. Compute $\\pi_1, \\dots, \\pi_4$ of $(1-L)^{0.25}$.', '1. Calculați $\\pi_1, \\dots, \\pi_4$ pentru $(1-L)^{0{,}25}$.'),
               T('2. Compute $\\rho(1)$ and $\\rho(2)$, and give $H$ and the class of the process.', '2. Calculați $\\rho(1)$ și $\\rho(2)$ și precizați $H$ și clasa procesului.'),
               T('3. Compare $\\rho(5)$ with that of an AR(1) with the same $\\rho(1)$.', '3. Comparați $\\rho(5)$ cu valoarea unui AR(1) cu același $\\rho(1)$.'),
               T('Report: four weights, two autocorrelations, $H$, two values of $\\rho(5)$.', 'Raportați: patru ponderi, două autocorelații, $H$, două valori ale lui $\\rho(5)$.')),
         items(T('1. $\\pi_1 = -0.25$; $\\pi_2 = -0.25 \\cdot 0.75/2 = @{a1.p2}$; $\\pi_3 = @{a1.p2} \\cdot 1.75/3 = @{a1.p3}$; $\\pi_4 = @{a1.p3} \\cdot 2.75/4 = @{a1.p4}$', '1. $\\pi_1 = -0{,}25$; $\\pi_2 = -0{,}25 \\cdot 0{,}75/2 = @{a1.p2}$; $\\pi_3 = @{a1.p2} \\cdot 1{,}75/3 = @{a1.p3}$; $\\pi_4 = @{a1.p3} \\cdot 2{,}75/4 = @{a1.p4}$'),
               T('2. $\\rho(1) = 0.25/0.75 = @{a1.rho1}$; $\\rho(2) = @{a1.rho1} \\cdot 1.25/1.75 = @{a1.rho2}$; $H = 0.75$: stationary, long memory', '2. $\\rho(1) = 0{,}25/0{,}75 = @{a1.rho1}$; $\\rho(2) = @{a1.rho1} \\cdot 1{,}25/1{,}75 = @{a1.rho2}$; $H = 0{,}75$: staționar, memorie lungă'),
               T('3. ARFIMA: $\\rho(5) = @{a1.rho5}$; AR(1): $@{a1.rho1}^5 = @{a1.ar5}$, about 37 times smaller', '3. ARFIMA: $\\rho(5) = @{a1.rho5}$; AR(1): $@{a1.rho1}^5 = @{a1.ar5}$, de aproximativ 37 de ori mai mic')),
         size='scriptsize')

D.proposed(T('A2: which range is $d$ in?', 'A2: în ce interval se află $d$?'),
           items(T('Two series: (a) ARFIMA$(0, 0.6, 0)$; (b) ARFIMA$(0, -0.3, 0)$. Model: A1.', 'Două serii: (a) ARFIMA$(0;\\ 0{,}6;\\ 0)$; (b) ARFIMA$(0;\\ -0{,}3;\\ 0)$. Model: A1.'),
                 T('1. Compute $\\pi_1$ and $\\pi_2$ of $(1-L)^d$ for each.', '1. Calculați $\\pi_1$ și $\\pi_2$ ale lui $(1-L)^d$ pentru fiecare.'),
                 T('2. Say whether each series is stationary, invertible and mean-reverting.', '2. Precizați dacă fiecare serie este staționară, inversabilă și cu revenire la medie.'),
                 T('3. For (b) compute $\\rho(1)$ and $\\rho(2)$; for (a) compute $\\rho(1)$ of the first difference.', '3. Pentru (b) calculați $\\rho(1)$ și $\\rho(2)$; pentru (a) calculați $\\rho(1)$ al primei diferențe.'),
                 T('Report: four weights, two classifications, three autocorrelations.', 'Raportați: patru ponderi, două clasificări, trei autocorelații.')),
           items(T('1. (a) $\\pi_1 = @{a2a.p1}$, $\\pi_2 = @{a2a.p2}$; (b) $\\pi_1 = @{a2b.p1}$, $\\pi_2 = @{a2b.p2}$', '1. (a) $\\pi_1 = @{a2a.p1}$, $\\pi_2 = @{a2a.p2}$; (b) $\\pi_1 = @{a2b.p1}$, $\\pi_2 = @{a2b.p2}$'),
                 T('2. (a) not stationary ($d \\ge 0.5$), invertible, mean-reverting ($d < 1$); (b) stationary, invertible, anti-persistent ($H = 0.2$)', '2. (a) nestaționară ($d \\ge 0{,}5$), inversabilă, cu revenire la medie ($d < 1$); (b) staționară, inversabilă, antipersistentă ($H = 0{,}2$)'),
                 T('3. (b) $\\rho(1) = -0.3/1.3 = @{a2b.r1}$, $\\rho(2) = @{a2b.r1} \\cdot 0.7/2.3 = @{a2b.r2}$; (a) the difference is ARFIMA$(0, -0.4, 0)$: $\\rho(1) = -0.4/1.4 = @{a2c.r1}$', '3. (b) $\\rho(1) = -0{,}3/1{,}3 = @{a2b.r1}$, $\\rho(2) = @{a2b.r1} \\cdot 0{,}7/2{,}3 = @{a2b.r2}$; (a) diferența este ARFIMA$(0;\\ -0{,}4;\\ 0)$: $\\rho(1) = -0{,}4/1{,}4 = @{a2c.r1}$')),
           size='scriptsize')

D.solved(T('A3: R/S by hand and $H$ from three points', 'A3: R/S de mînă și $H$ din trei puncte'),
         items(T('Block: $x = (0.6, 0.4, 0.5, 0.3, -0.4, -0.6, -0.2, -0.5)$. An R/S plot gives $(R/S)_n = 4.1;\\ 9.6;\\ 22.4$ for $n = 16, 64, 256$.', 'Blocul: $x = (0{,}6;\\ 0{,}4;\\ 0{,}5;\\ 0{,}3;\\ -0{,}4;\\ -0{,}6;\\ -0{,}2;\\ -0{,}5)$. Un grafic R/S dă $(R/S)_n = 4{,}1;\\ 9{,}6;\\ 22{,}4$ pentru $n = 16;\\ 64;\\ 256$.'),
               T('1. Compute $\\bar x$, the cumulative deviations $Y_k$, $R$, $S$ and $R/S$.', '1. Calculați $\\bar x$, abaterile cumulate $Y_k$, $R$, $S$ și $R/S$.'),
               T('2. Compute $H$ as the slope of $\\log_{10}(R/S)_n$ on $\\log_{10}n$, and $d = H - 0.5$.', '2. Calculați $H$ ca pantă a lui $\\log_{10}(R/S)_n$ față de $\\log_{10}n$ și $d = H - 0{,}5$.'),
               T('Report: $R/S$, $H$, $d$ and one sentence.', 'Raportați: $R/S$, $H$, $d$ și o frază.')),
         items(T('1. $\\bar x = @{a3.mean}$; $Y_k$: @{a3.y1}, @{a3.y2}, @{a3.y3}, @{a3.y4}, @{a3.y5}, @{a3.y6}, @{a3.y7}, 0; $R = @{a3.R}$, $S = @{a3.S}$, $R/S = @{a3.RS}$', '1. $\\bar x = @{a3.mean}$; $Y_k$: @{a3.y1}; @{a3.y2}; @{a3.y3}; @{a3.y4}; @{a3.y5}; @{a3.y6}; @{a3.y7}; 0; $R = @{a3.R}$, $S = @{a3.S}$, $R/S = @{a3.RS}$'),
               T('the signs are grouped (four positive, then four negative): the range is large, as for a persistent series', 'semnele sînt grupate (patru pozitive, apoi patru negative): amplitudinea este mare, ca pentru o serie persistentă'),
               T('2. $\\log_{10}n$: @{ap.ln0}, @{ap.ln1}, @{ap.ln2}; $\\log_{10}(R/S)$: @{ap.lr0}, @{ap.lr1}, @{ap.lr2}; slope $H = @{ap.H}$, $d = @{ap.d}$: moderate persistence', '2. $\\log_{10}n$: @{ap.ln0}; @{ap.ln1}; @{ap.ln2}; $\\log_{10}(R/S)$: @{ap.lr0}; @{ap.lr1}; @{ap.lr2}; panta $H = @{ap.H}$, $d = @{ap.d}$: persistență moderată')),
         size='scriptsize')

D.proposed(T('A4: a GPH slope by hand', 'A4: o pantă GPH de mînă'),
           items(T('For $m = 4$ frequencies: $X_j = 0.5;\\ 1.6;\\ 2.3;\\ 3.0$ and $\\log I(\\lambda_j) = 1.2;\\ 1.8;\\ 2.0;\\ 2.5$. Model: A3.', 'Pentru $m = 4$ frecvențe: $X_j = 0{,}5;\\ 1{,}6;\\ 2{,}3;\\ 3{,}0$ și $\\log I(\\lambda_j) = 1{,}2;\\ 1{,}8;\\ 2{,}0;\\ 2{,}5$. Model: A3.'),
                 T('1. Compute $\\bar X$, $\\bar y$, $\\sum(X_j - \\bar X)(y_j - \\bar y)$ and $\\sum(X_j - \\bar X)^2$.', '1. Calculați $\\bar X$, $\\bar y$, $\\sum(X_j - \\bar X)(y_j - \\bar y)$ și $\\sum(X_j - \\bar X)^2$.'),
                 T('2. Compute $\\hat d$ and its standard error $\\pi/\\sqrt{24m}$.', '2. Calculați $\\hat d$ și eroarea standard $\\pi/\\sqrt{24m}$.'),
                 T('3. Test $H_0$: $d = 0$ at 5\\% and comment on the size of $m$.', '3. Testați $H_0$: $d = 0$ la 5\\% și comentați mărimea lui $m$.'),
                 T('Report: four sums, $\\hat d$, its SE, a decision and one sentence.', 'Raportați: patru sume, $\\hat d$, SE, o decizie și o frază.')),
           items(T('1. $\\bar X = @{a4.xbar}$, $\\bar y = @{a4.ybar}$, $S_{xy} = @{a4.sxy}$, $S_{xx} = @{a4.sxx}$', '1. $\\bar X = @{a4.xbar}$, $\\bar y = @{a4.ybar}$, $S_{xy} = @{a4.sxy}$, $S_{xx} = @{a4.sxx}$'),
                 T('2. $\\hat d = @{a4.sxy}/@{a4.sxx} = @{a4.d}$; SE $= \\pi/\\sqrt{96} = @{a4.se}$', '2. $\\hat d = @{a4.sxy}/@{a4.sxx} = @{a4.d}$; SE $= \\pi/\\sqrt{96} = @{a4.se}$'),
                 T('3. $t = @{a4.t} < 1.96$: do not reject $d = 0$; with only 4 frequencies the estimate is very imprecise: real applications use $m$ of 20--300', '3. $t = @{a4.t} < 1{,}96$: nu respingem $d = 0$; cu doar 4 frecvențe estimarea este foarte imprecisă: aplicațiile reale folosesc $m$ între 20 și 300')),
           size='scriptsize')

D.solved(T('A5: a one-step ARFIMA forecast', 'A5: o prognoză ARFIMA la un pas'),
         items(T('ARFIMA$(0, 0.4, 0)$ with $\\mu = 2$; the last four values: $x_T = 3.0$, $x_{T-1} = 2.5$, $x_{T-2} = 2.8$, $x_{T-3} = 1.5$.', 'ARFIMA$(0;\\ 0{,}4;\\ 0)$, cu $\\mu = 2$; ultimele patru valori: $x_T = 3{,}0$, $x_{T-1} = 2{,}5$, $x_{T-2} = 2{,}8$, $x_{T-3} = 1{,}5$.'),
               T('1. Write the first four weights $-\\pi_k$ of $(1-L)^{0.4}$.', '1. Scrieți primele patru ponderi $-\\pi_k$ ale lui $(1-L)^{0{,}4}$.'),
               T('2. Compute $\\hat x_{T+1}$ using these four lags.', '2. Calculați $\\hat x_{T+1}$ folosind aceste patru decalaje.'),
               T('3. Compare with an AR(1) with the same $\\rho(1)$.', '3. Comparați cu un AR(1) cu același $\\rho(1)$.'),
               T('Report: four weights, two forecasts and one sentence.', 'Raportați: patru ponderi, două prognoze și o frază.')),
         items(T('1. $-\\pi_k$: 0.4; 0.12; 0.064; 0.0416', '1. $-\\pi_k$: 0,4; 0,12; 0,064; 0,0416'),
               T('2. $\\hat x_{T+1} = 2 + 0.4(1.0) + 0.12(0.5) + 0.064(0.8) + 0.0416(-0.5) = 2 + @{a5.t0} + @{a5.t1} + @{a5.t2} @{a5.t3} = @{a5.fa}$', '2. $\\hat x_{T+1} = 2 + 0{,}4(1{,}0) + 0{,}12(0{,}5) + 0{,}064(0{,}8) + 0{,}0416(-0{,}5) = 2 + @{a5.t0} + @{a5.t1} + @{a5.t2} @{a5.t3} = @{a5.fa}$'),
               T('3. AR(1): $\\phi = \\rho(1) = 0.4/0.6 = @{a5.phi}$, $\\hat x_{T+1} = 2 + @{a5.phi} \\cdot 1.0 = @{a5.fr}$; the AR(1) looks only at $x_T$, ARFIMA spreads the weight over the past (in practice over all $T$ values)', '3. AR(1): $\\phi = \\rho(1) = 0{,}4/0{,}6 = @{a5.phi}$, $\\hat x_{T+1} = 2 + @{a5.phi} \\cdot 1{,}0 = @{a5.fr}$; AR(1) se uită doar la $x_T$, ARFIMA distribuie ponderea pe trecut (în practică, pe toate cele $T$ valori)')),
         size='scriptsize')

D.proposed(T('A6: how long does a volatility shock count?', 'A6: cît timp contează un șoc al volatilității?'),
           items(T('GARCH(1,1) with $\\alpha = 0.10$, $\\beta = 0.88$; FIGARCH$(0, d, 0)$ with $d = 0.4$ (weights $\\lambda_k = -\\pi_k$). Model: A5.', 'GARCH(1,1) cu $\\alpha = 0{,}10$, $\\beta = 0{,}88$; FIGARCH$(0, d, 0)$ cu $d = 0{,}4$ (ponderile $\\lambda_k = -\\pi_k$). Model: A5.'),
                 T('1. Compute the weight of $\\varepsilon_{t-k}^2$ in $\\sigma_t^2$ for $k = 1, 2, 3$ in both models.', '1. Calculați ponderea lui $\\varepsilon_{t-k}^2$ în $\\sigma_t^2$ pentru $k = 1, 2, 3$ în ambele modele.'),
                 T('2. Compute both weights at $k = 20$ and $k = 100$; for FIGARCH use $\\lambda_k \\approx d\\,k^{-1-d}/\\Gamma(1-d)$, $\\Gamma(0.6) = 1.489$.', '2. Calculați ambele ponderi la $k = 20$ și $k = 100$; pentru FIGARCH folosiți $\\lambda_k \\approx d\\,k^{-1-d}/\\Gamma(1-d)$, $\\Gamma(0{,}6) = 1{,}489$.'),
                 T('3. Explain which model keeps a crisis in today\'s volatility longer.', '3. Explicați ce model păstrează mai mult timp o criză în volatilitatea de azi.'),
                 T('Report: ten weights and one sentence.', 'Raportați: zece ponderi și o frază.')),
           items(T('1. GARCH $\\alpha\\beta^{k-1}$: @{a6.g1}, @{a6.g2}, @{a6.g3}; FIGARCH: @{a6.f1}, @{a6.f2}, @{a6.f3}', '1. GARCH $\\alpha\\beta^{k-1}$: @{a6.g1}; @{a6.g2}; @{a6.g3}; FIGARCH: @{a6.f1}; @{a6.f2}; @{a6.f3}'),
                 T('2. $k = 20$: GARCH @{a6.g20}, FIGARCH @{a6.a20} (exact @{a6.f20}); $k = 100$: GARCH $@{a6.g100}$, FIGARCH $@{a6.f100x} \\cdot 10^{-4}$', '2. $k = 20$: GARCH @{a6.g20}, FIGARCH @{a6.a20} (exact @{a6.f20}); $k = 100$: GARCH $@{a6.g100}$, FIGARCH $@{a6.f100x} \\cdot 10^{-4}$'),
                 T('3. FIGARCH: after 100 days the weight is about @{a6.ratio} times the GARCH weight; GARCH forgets a crisis within weeks, FIGARCH within years', '3. FIGARCH: după 100 de zile ponderea este de aproximativ @{a6.ratio} de ori ponderea GARCH; GARCH uită o criză în cîteva săptămîni, FIGARCH în ani')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: real data and interpretation', 'Partea B: date reale și interpretare')

D.task(T('B1: does the Nile have long memory? [Solved]', 'B1: are Nilul memorie lungă? [Rezolvat]'),
       T('is the persistence of the Nile flow long memory, or the effect of the change in 1898?', 'este persistența debitului Nilului memorie lungă sau efectul schimbării din 1898?'),
       T('yearly flow at Aswan, 1871--1970 ($T = 100$)', 'debitul anual la Aswan, 1871--1970 ($T = 100$)'),
       [T('Plot the series and its ACF up to lag 20.', 'Reprezentați grafic seria și ACF pînă la decalajul 20.'),
        T('Estimate $H$ by R/S, $d$ by GPH and local Whittle ($m = \\lfloor T^{0.65}\\rfloor$) and by exact ML of ARFIMA$(0,d,0)$.', 'Estimați $H$ prin R/S, $d$ prin GPH și Whittle local ($m = \\lfloor T^{0{,}65}\\rfloor$) și prin ML exactă pentru ARFIMA$(0,d,0)$.'),
        T('Subtract the mean of 1871--1898 and the mean of 1899--1970 and repeat step 2.', 'Scădeți media din 1871--1898 și media din 1899--1970 și repetați pasul 2.'),
        T('Interpretation: does the Nile have long memory?', 'Interpretare: are Nilul memorie lungă?')],
       T('two ACF values, four estimates before and after, one sentence', 'două valori ale ACF, patru estimări înainte și după, o frază'), size='footnotesize', nb='B1')

D.frame(T('B1: solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch8_sem_b1', h='0.42') + items(
    T('Raw: $\\hat\\rho(1) = @{b1.raw.r1}$, $\\hat\\rho(10) = @{b1.raw.r10}$; $H_{R/S} = @{b1.raw.H}$; GPH @{b1.raw.gph} (SE @{b1.raw.gph_se}), LW @{b1.raw.lw} (SE @{b1.raw.lw_se}), ML @{b1.raw.ml} (SE @{b1.raw.ml_se}); $m = @{b1.m}$',
      'Seria brută: $\\hat\\rho(1) = @{b1.raw.r1}$, $\\hat\\rho(10) = @{b1.raw.r10}$; $H_{R/S} = @{b1.raw.H}$; GPH @{b1.raw.gph} (SE @{b1.raw.gph_se}), LW @{b1.raw.lw} (SE @{b1.raw.lw_se}), ML @{b1.raw.ml} (SE @{b1.raw.ml_se}); $m = @{b1.m}$'),
    T('Means @{b1.pre} and @{b1.post}; after removing them: $\\hat\\rho(1) = @{b1.adj.r1}$; $H_{R/S} = @{b1.adj.H}$; GPH @{b1.adj.gph}, LW @{b1.adj.lw}, ML @{b1.adj.ml} (SE @{b1.adj.ml_se})',
      'Mediile @{b1.pre} și @{b1.post}; după eliminarea lor: $\\hat\\rho(1) = @{b1.adj.r1}$; $H_{R/S} = @{b1.adj.H}$; GPH @{b1.adj.gph}, LW @{b1.adj.lw}, ML @{b1.adj.ml} (SE @{b1.adj.ml_se})'),
    T('Interpretation: for 1871--1970 the evidence of long memory comes from one break; R/S stays above 0.5 because it is biased upwards in short samples', 'Interpretare: pentru 1871--1970 dovada de memorie lungă provine dintr-o singură ruptură; R/S rămîne peste 0,5 deoarece este deplasat în sus în eșantioane scurte')) + qlsem(),
    'scriptsize')

D.task(T('B2: long memory in Romanian inflation [Proposed]', 'B2: memoria lungă a inflației din România [Propus]'),
       T('how persistent is Romanian inflation, and does the way we measure inflation change the answer?', 'cît de persistentă este inflația din România și schimbă modul de măsurare a inflației răspunsul?'),
       T('monthly HICP, 2005--2026: the monthly rate seasonally adjusted with the monthly means, and the 12-month rate; model: B1', 'IAPC lunar, 2005--2026: rata lunară ajustată sezonier cu mediile lunare și rata anuală; model: B1'),
       [T('Estimate $d$ of the monthly rate by GPH and local Whittle with $m = \\lfloor T^a\\rfloor$, $a = 0.5;\\ 0.65;\\ 0.8$.', 'Estimați $d$ al ratei lunare prin GPH și Whittle local, cu $m = \\lfloor T^a\\rfloor$, $a = 0{,}5;\\ 0{,}65;\\ 0{,}8$.'),
        T('Fit ARFIMA$(0,d,0)$ and ARMA(1,1) by exact ML and compare their BIC.', 'Estimați ARFIMA$(0,d,0)$ și ARMA(1,1) prin ML exactă și comparați BIC.'),
        T('Estimate $d$ of the 12-month rate by local Whittle and compare the two ACFs.', 'Estimați $d$ al ratei anuale prin Whittle local și comparați cele două ACF.'),
        T('Interpretation: why does the 12-month rate give $d$ close to 1?', 'Interpretare: de ce dă rata anuală un $d$ apropiat de 1?')],
       T('six estimates, two BIC values, one $d$, two sentences', 'șase estimări, două valori BIC, un $d$, două fraze'), size='footnotesize', nb='B2')

D.frame(T('B2: solution [Proposed]', 'B2: rezolvare [Propus]'), fig('ch8_sem_b2', h='0.42') + items(
    T('$T = @{b2.T}$; GPH / LW: @{b2.g5} / @{b2.l5} ($m = @{b2.m5}$), @{b2.g65} / @{b2.l65} ($m = @{b2.m65}$), @{b2.g8} / @{b2.l8} ($m = @{b2.m8}$)',
      '$T = @{b2.T}$; GPH / LW: @{b2.g5} / @{b2.l5} ($m = @{b2.m5}$), @{b2.g65} / @{b2.l65} ($m = @{b2.m65}$), @{b2.g8} / @{b2.l8} ($m = @{b2.m8}$)'),
    T('Exact ML: $\\hat d = @{b2.ml_d}$ (SE @{b2.ml_se}), BIC @{b2.bicf}; ARMA(1,1): $\\phi = @{b2.a_phi}$, $\\theta = @{b2.a_theta}$, BIC @{b2.bica}: ARFIMA preferred',
      'ML exactă: $\\hat d = @{b2.ml_d}$ (SE @{b2.ml_se}), BIC @{b2.bicf}; ARMA(1,1): $\\phi = @{b2.a_phi}$, $\\theta = @{b2.a_theta}$, BIC @{b2.bica}: preferăm ARFIMA'),
    T('12-month rate: LW @{b2.lw12}, GPH @{b2.gph12}; $\\hat\\rho(1)$ @{b2.r1_12} against @{b2.r1} for the monthly rate',
      'Rata anuală: LW @{b2.lw12}, GPH @{b2.gph12}; $\\hat\\rho(1)$ @{b2.r1_12}, față de @{b2.r1} pentru rata lunară'),
    T('Interpretation: consecutive 12-month rates share 11 months; the overlap is a filter that removes power at the frequencies near $2\\pi/12$ and steepens the log-periodogram: $\\hat d$ is pushed towards 1',
      'Interpretare: două rate anuale consecutive au 11 luni comune; suprapunerea este un filtru care elimină puterea la frecvențele din jurul lui $2\\pi/12$ și înclină log-periodograma: $\\hat d$ este împins spre 1')) + qlsem(),
    'scriptsize', instructor_only=True)

D.task(T('B3: memory in S\\&P 500 volatility [Solved]', 'B3: memoria volatilității S\\&P 500 [Rezolvat]'),
       T('where is the memory of stock returns: in the returns or in their size?', 'unde se află memoria randamentelor bursiere: în randamente sau în mărimea lor?'),
       T('daily log returns of the S\\&P 500 since 2000 and their monthly realised volatility', 'randamentele logaritmice zilnice S\\&P 500 din 2000 și volatilitatea lor realizată lunară'),
       [T('Estimate $d$ of $r_t$ and of $|r_t|$ by local Whittle and GPH.', 'Estimați $d$ pentru $r_t$ și pentru $|r_t|$ prin Whittle local și GPH.'),
        T('Shuffle the days at random and re-estimate $d$ of $|r_t|$; plot both ACFs of $|r_t|$.', 'Permutați aleator zilele și reestimați $d$ pentru $|r_t|$; reprezentați grafic ambele ACF ale lui $|r_t|$.'),
        T('Estimate $d$ of the monthly log realised volatility.', 'Estimați $d$ pentru logaritmul volatilității realizate lunare.'),
        T('Interpretation: what does the shuffle test show?', 'Interpretare: ce arată testul permutării?')],
       T('eight estimates, the chart and one sentence', 'opt estimări, graficul și o frază'), size='footnotesize', nb='B3')

D.frame(T('B3: solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch8_sem_b3', h='0.42') + items(
    T('$T = @{b3.T}$ days; $r_t$: LW @{b3.r.lw}, GPH @{b3.r.gph} (SE @{b3.r.lw_se}, @{b3.r.gph_se}); $|r_t|$: LW @{b3.abs.lw}, GPH @{b3.abs.gph}',
      '$T = @{b3.T}$ de zile; $r_t$: LW @{b3.r.lw}, GPH @{b3.r.gph} (SE @{b3.r.lw_se}, @{b3.r.gph_se}); $|r_t|$: LW @{b3.abs.lw}, GPH @{b3.abs.gph}'),
    T('Shuffled $|r_t|$: LW @{b3.shuf.lw}, GPH @{b3.shuf.gph}; ACF of $|r_t|$: @{b3.acf1} (lag 1), @{b3.acf50} (lag 50), @{b3.acf200} (lag 200); log RV ($@{b3.Trv}$ months): LW @{b3.rv.lw}, GPH @{b3.rv.gph}',
      '$|r_t|$ permutat: LW @{b3.shuf.lw}, GPH @{b3.shuf.gph}; ACF a lui $|r_t|$: @{b3.acf1} (decalajul 1), @{b3.acf50} (decalajul 50), @{b3.acf200} (decalajul 200); log RV (@{b3.Trv} de luni): LW @{b3.rv.lw}, GPH @{b3.rv.gph}'),
    T('Interpretation: the shuffle keeps every daily value and destroys their order; the memory disappears, so it lives in the clustering of calm and turbulent periods, not in the distribution of returns',
      'Interpretare: permutarea păstrează fiecare valoare zilnică și le distruge ordinea; memoria dispare, deci ea se află în gruparea perioadelor calme și agitate, nu în distribuția randamentelor')) + qlsem(),
    'scriptsize')

D.task(T('B4: BET and EUR/RON [Proposed]', 'B4: BET și EUR/RON [Propus]'),
       T('do the BET and the EUR/RON rate show the same memory, and is it stable before and after the 2008 crisis?', 'au BET și cursul EUR/RON aceeași memorie și este ea stabilă înainte și după criza din 2008?'),
       T('daily log returns of the BET (since 2000) and of the EUR/RON reference rate (since July 2005); model: B3', 'randamentele logaritmice zilnice ale BET (din 2000) și ale cursului de referință EUR/RON (din iulie 2005); model: B3'),
       [T('Estimate $d$ of $r_t$ and $|r_t|$ by local Whittle on the full sample.', 'Estimați $d$ pentru $r_t$ și $|r_t|$ prin Whittle local pe tot eșantionul.'),
        T('Repeat on the subsamples up to 2007 and 2010--2026.', 'Repetați pe subeșantioanele pînă în 2007 și 2010--2026.'),
        T('Compare the estimates for $r_t$ with the 95\\% Monte Carlo band of i.i.d. series of 1900 days.', 'Comparați estimările pentru $r_t$ cu banda Monte Carlo de 95\\% a seriilor i.i.d. de 1900 de zile.'),
        T('Interpretation: is the full-sample memory of BET returns real?', 'Interpretare: este reală memoria randamentelor BET pe tot eșantionul?')],
       T('twelve estimates, one band and two sentences', 'douăsprezece estimări, o bandă și două fraze'), size='footnotesize', nb='B4')

D.frame(T('B4: solution [Proposed]', 'B4: rezolvare [Propus]'), fig('ch8_sem_b4', h='0.42') + items(
    T('BET $r_t$: @{b4.bet.full.r} (full), @{b4.bet.20002007.r} (to 2007), @{b4.bet.20102026.r} (2010--2026); $|r_t|$: @{b4.bet.full.abs}, @{b4.bet.20002007.abs}, @{b4.bet.20102026.abs}',
      'BET $r_t$: @{b4.bet.full.r} (tot), @{b4.bet.20002007.r} (pînă în 2007), @{b4.bet.20102026.r} (2010--2026); $|r_t|$: @{b4.bet.full.abs}, @{b4.bet.20002007.abs}, @{b4.bet.20102026.abs}'),
    T('EUR/RON $r_t$: @{b4.eurron.full.r}, @{b4.eurron.20002007.r}, @{b4.eurron.20102026.r}; $|r_t|$: @{b4.eurron.full.abs}, @{b4.eurron.20002007.abs}, @{b4.eurron.20102026.abs}; band [@{b4.lo}; @{b4.hi}]',
      'EUR/RON $r_t$: @{b4.eurron.full.r}, @{b4.eurron.20002007.r}, @{b4.eurron.20102026.r}; $|r_t|$: @{b4.eurron.full.abs}, @{b4.eurron.20002007.abs}, @{b4.eurron.20102026.abs}; banda [@{b4.lo}; @{b4.hi}]'),
    T('Interpretation: the BET return memory of the full sample (@{b4.bet.full.r}) vanishes inside each subsample: it comes from the change of level and volatility around 2008, a spurious long memory; the memory of $|r_t|$ survives in both markets',
      'Interpretare: memoria randamentelor BET pe tot eșantionul (@{b4.bet.full.r}) dispare în fiecare subeșantion: provine din schimbarea de nivel și de volatilitate din jurul anului 2008, o memorie lungă aparentă; memoria lui $|r_t|$ rezistă pe ambele piețe')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: open questions and AI critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: US inflation and monetary regimes [Proposed]', 'C1: inflația din SUA și regimurile monetare [Propus]'),
       T('is the long memory of US inflation a property of inflation, or of the changes in monetary policy?', 'este memoria lungă a inflației din SUA o proprietate a inflației sau a schimbărilor de politică monetară?'),
       T('US monthly CPI inflation, annualised, 1947--2026; regimes 1947--1984, 1985--2019, 2020--2026; models: B1, B2', 'inflația lunară IPC din SUA, anualizată, 1947--2026; regimurile 1947--1984, 1985--2019, 2020--2026; modele: B1, B2'),
       [T('Estimate $d$ by local Whittle on the full sample and in each regime.', 'Estimați $d$ prin Whittle local pe tot eșantionul și în fiecare regim.'),
        T('Remove the three regime means and re-estimate $d$.', 'Eliminați cele trei medii de regim și reestimați $d$.'),
        T('Propose one more check (a break test, a Markov-switching model, a forecast comparison).', 'Propuneți încă o verificare (un test de ruptură, un model Markov switching, o comparație de prognoze).'),
        T('Interpretation: can a constant $d$ describe 80 years of inflation?', 'Interpretare: poate un $d$ constant să descrie 80 de ani de inflație?')],
       T('five estimates, one check and a project plan', 'cinci estimări, o verificare și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: reference analysis [Proposed]', 'C1: analiză de referință [Propus]'), fig('ch8_sem_c1', h='0.42') + items(
    T('$\\hat d$: full sample @{c1.full.lw}; 1947--1984 @{c1.19471984.lw} (mean @{c1.19471984.mean}\\%); 1985--2019 @{c1.19852019.lw} (mean @{c1.19852019.mean}\\%); 2020--2026 @{c1.20202026.lw} (@{c1.20202026.n} months)',
      '$\\hat d$: tot eșantionul @{c1.full.lw}; 1947--1984 @{c1.19471984.lw} (media @{c1.19471984.mean}\\%); 1985--2019 @{c1.19852019.lw} (media @{c1.19852019.mean}\\%); 2020--2026 @{c1.20202026.lw} (@{c1.20202026.n} de luni)'),
    T('Regime means removed: @{c1.adj.lw}: the mean shifts alone do not remove the memory; it sits inside the 1947--1984 regime (the Great Inflation)',
      'Fără mediile de regim: @{c1.adj.lw}: schimbările de medie singure nu elimină memoria; ea se află în interiorul regimului 1947--1984 (Marea Inflație)'),
    T('Interpretation: no; memory is strong when policy lets inflation drift and weak under inflation targeting; $d$ is a property of the policy regime, a natural link to Chapter 10',
      'Interpretare: nu; memoria este puternică atunci cînd politica lasă inflația să derive și slabă sub țintirea inflației; $d$ este o proprietate a regimului de politică, o legătură naturală cu Capitolul 10'),
    T('Project: the same analysis for Romania (monthly HICP since 1997) with the 2005 switch to inflation targeting', 'Proiect: aceeași analiză pentru România (IAPC lunar din 1997), cu trecerea la țintirea inflației din 2005')) + qlsem(),
    'scriptsize', instructor_only=True)

D.frame(T('C2: audit an AI answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant about the long memory of Romanian inflation and of the S\\&P 500. The answer:', 'Un student a întrebat un asistent AI despre memoria lungă a inflației din România și a S\\&P 500. Răspunsul:'),
    T('\\aiprompt{(a) Monthly inflation has d = @{c2.ml}, so its Hurst exponent is H = d - 0.5 < 0: it is anti-persistent.}', '\\aiprompt{(a) Inflația lunară are d = @{c2.ml}, deci exponentul Hurst este H = d - 0,5 < 0: este antipersistentă.}'),
    T('\\aiprompt{(b) The 12-month rate gives d = @{c2.lw12}, so Romanian inflation has a unit root.}', '\\aiprompt{(b) Rata anuală dă d = @{c2.lw12}, deci inflația din România are rădăcină unitară.}'),
    T('\\aiprompt{(c) With d = @{c2.ml} the series is stationary and its shocks die out.}', '\\aiprompt{(c) Cu d = @{c2.ml} seria este staționară, iar șocurile ei se sting.}'),
    T('\\aiprompt{(d) Use statsmodels.tsa.arima.model.ARIMA(x, order=(0, 0.31, 0)) to fit the ARFIMA model.}', '\\aiprompt{(d) Folosiți statsmodels.tsa.arima.model.ARIMA(x, order=(0, 0.31, 0)) pentru a estima modelul ARFIMA.}'),
    T('\\aiprompt{(e) S\\&P 500 returns have d = @{c2.sp_r} and |r| has d = @{c2.sp_abs}, so returns are predictable.}', '\\aiprompt{(e) Randamentele S\\&P 500 au d = @{c2.sp_r}, iar |r| are d = @{c2.sp_abs}, deci randamentele sînt previzibile.}'),
    T('\\aiprompt{(f) A significant GPH estimate proves long memory.}', '\\aiprompt{(f) O estimare GPH semnificativă demonstrează memoria lungă.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: $H = d + 0.5 = 0.81$: persistent', '(a) Greșit: $H = d + 0{,}5 = 0{,}81$: persistentă'),
    T('(b) Wrong: the overlap of 12-month rates pushes $\\hat d$ towards 1 (B2); the monthly rate gives $d = @{c2.ml}$', '(b) Greșit: suprapunerea ratelor anuale împinge $\\hat d$ spre 1 (B2); rata lunară dă $d = @{c2.ml}$'),
    T('(c) Correct: $0 < d < 0.5$; shocks die out hyperbolically, slower than in ARMA', '(c) Corect: $0 < d < 0{,}5$; șocurile se sting hiperbolic, mai lent decît în ARMA'),
    T('(d) Wrong: the ARIMA order must be an integer; statsmodels has no ARFIMA class (use exact ML or Whittle, as in the notebook)', '(d) Greșit: ordinul ARIMA trebuie să fie întreg; statsmodels nu are o clasă ARFIMA (folosiți ML exactă sau Whittle, ca în notebook)'),
    T('(e) Wrong: $d \\approx 0$ for returns means no memory in returns; the memory of $|r_t|$ makes volatility predictable, not the sign', '(e) Greșit: $d \\approx 0$ pentru randamente înseamnă că randamentele nu au memorie; memoria lui $|r_t|$ face volatilitatea previzibilă, nu semnul'),
    T('(f) Wrong: breaks and regime shifts also give a significant $\\hat d$ (B1, B4)', '(f) Greșit: rupturile și schimbările de regim dau și ele un $\\hat d$ semnificativ (B1, B4)')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-up', 'Încheiere')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('$(1-L)^d$: weights by the recursion $\\pi_k = \\pi_{k-1}(k-1-d)/k$; ARFIMA$(0,d,0)$: $\\rho(1) = d/(1-d)$', '$(1-L)^d$: ponderile prin recurența $\\pi_k = \\pi_{k-1}(k-1-d)/k$; ARFIMA$(0,d,0)$: $\\rho(1) = d/(1-d)$'),
    T('$H = d + 1/2$; R/S and GPH are slopes of log--log regressions', '$H = d + 1/2$; R/S și GPH sînt pante ale unor regresii log--log'),
    T('Romanian monthly inflation: $d \\approx 0.3$; S\\&P 500: memory in $|r_t|$, not in $r_t$', 'Inflația lunară din România: $d \\approx 0{,}3$; S\\&P 500: memorie în $|r_t|$, nu în $r_t$'),
    T('Breaks can create long memory: the Nile, BET returns, US inflation', 'Rupturile pot crea memorie lungă: Nilul, randamentele BET, inflația din SUA'),
    T('An AI answer is a draft: check $H$ against $d$, the library, and the data transformation', 'Un răspuns AI este o ciornă: verificați relația dintre $H$ și $d$, biblioteca și transformarea datelor')))

D.frame(T('After the seminar', 'După seminar'), items(
    T('Lecture 8 develops each topic of today: hyperbolic decay, fractional differencing, ARFIMA, the estimators, forecasting, volatility, spurious memory',
      'Cursul 8 dezvoltă fiecare temă de azi: descreșterea hiperbolică, diferențierea fracționară, ARFIMA, estimatorii, prognoza, volatilitatea, memoria aparentă'),
    T('Try the [Proposed] tasks in the notebook', 'Încercați cerințele [Propus] în notebook'),
    T('C1 can grow into a team project: inflation memory and monetary regimes in Romania', 'C1 poate deveni un proiect de echipă: memoria inflației și regimurile monetare din România'),
    T('Reading: \\refBaillie; \\refGraves; \\refDGE', 'Lectură: \\refBaillie; \\refGraves; \\refDGE'),
    T('\\textbf{The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class}',
      '\\textbf{Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar}')))

D.references(bib(['Baillie', 'BBM', 'DI', 'DGE', 'GJ', 'Graves', 'Hosking', 'Hurst', 'Rob', 'Sowell']), per=16)

if __name__ == '__main__':
    finalize(D.write(V))
