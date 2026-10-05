r"""
build_seminar3.py -- Seminarul 3 (Rădăcini unitare și modele ARIMA), EN + RO dintr-o singură sursă
==================================================================================================
Seminarul are loc ÎNAINTEA cursului 3: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu o întrebare de interpretare, C o întrebare deschisă și
critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în versiunea
profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_03/sem3_results.json (seminar3.py).
Ieșire:
  EN/Seminars/seminar3_unit_roots_arima_models.tex          (+ _solutions.tex)
  RO/Seminarii/seminar3_radacini_unitare_modele_arima_ro.tex (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_03/seminar3.py && python3 latex/build_seminar3.py && python3 latex/tsa_build.py compile 3
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, fig   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch3_common import REFS, T, bib, date, finalize, load_sem, pv, qtr, month   # noqa: E402

S = load_sem()
V = Values()
D = Deck(3, 'seminar', refs=REFS)


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


def qlsem():
    return '\\quantlet{TSA\\_ch3\\_seminar}{\\qlurl{TSA_ch3_seminar}}'


# =============================================================================
# CIFRE
# =============================================================================
A1 = S['A1']
V.put('a1.tau', A1['tau'], 2)
V.put('a1.cv', A1['crit5'], 2)
V.put('a1.cv10', A1['crit10'], 2)
V.put('a1.p', A1['p'], 2)
V.put('a1.phi', A1['phi'], 3)
A2 = S['A2']
for reg in ['n', 'c', 'ct']:
    V.put(f'a2.{reg}', A2[reg]['crit5'], 2)
A3 = S['A3']
for i in range(3):
    V.put(f'a3.f{i}', A3['f'][i], 2)
    V.put(f'a3.v{i}', A3['var'][i], 2)
    V.put(f'a3.h{i}', A3['half'][i], 2)
    V.put(f'a3.lo{i}', A3['lo'][i], 2)
    V.put(f'a3.hi{i}', A3['hi'][i], 2)
A4 = S['A4']
for i in [0, 1, 4]:
    V.put(f'a4.v{i}', A4['var'][i], 2)
    V.put(f'a4.h{i}', A4['half'][i], 2)
    V.put(f'a4.lo{i}', A4['lo'][i], 2)
    V.put(f'a4.hi{i}', A4['hi'][i], 2)
V.put('a4.f', A4['f'][0], 1)


def put_set(key, d):
    for part in ['level', 'diff']:
        x = d[part]
        V.put(f'{key}.{part}.adf', x['adf']['stat'], 2)
        V.raw(f'{key}.{part}.adfp', pv(x['adf']['p']))
        V.raw(f'{key}.{part}.lags', str(x['adf']['lags']))
        V.put(f'{key}.{part}.pp', x['pp']['stat'], 2)
        V.put(f'{key}.{part}.kpss', x['kpss']['stat'], 3)
        V.put(f'{key}.{part}.kcv', x['kpss']['crit5'], 3)
        V.put(f'{key}.{part}.cv', x['adf']['crit5'], 2)
    V.int(f'{key}.n', d['n'])


B1 = S['B1']
put_set('b1', B1)
V.put('b1.dev', B1['dev_last'], 0)
B2 = S['B2']
put_set('b2s', B2['sp500'])
put_set('b2e', B2['eurron'])
B3 = S['B3']
put_set('b3', B3['tests'])
for k in ['00', '01', '10', '11', '02', '20']:
    V.put(f'b3.g{k}', B3['grid'][k]['aicc'], 1)
V.raw('b3.o', f"ARIMA({B3['order'][0]},1,{B3['order'][2]})")
V.put('b3.c', B3['params']['x1'], 2)
V.put('b3.cse', B3['se']['x1'], 2)
V.put('b3.sig', B3['params']['sigma2'] ** 0.5, 2)
V.put('b3.q8', B3['lb8']['lb'], 1)
V.put('b3.q8p', B3['lb8']['lb_p'], 2)
V.put('b3.last', B3['last'], 1)
V.raw('b3.lq', qtr(B3['last_q']))
V.raw('b3.fq', qtr(B3['last_fq']))
V.put('b3.f8', B3['f8'], 1)
V.put('b3.h1', B3['half1'], 1)
V.put('b3.h2', B3['half2'], 1)
V.put('b3.h8', B3['half8'], 1)
V.put('b3.lo8', B3['lo8'], 1)
V.put('b3.hi8', B3['hi8'], 1)
V.put('b3.g8', B3['growth8'], 1)
V.put('b3.ratio', B3['half8'] / B3['half2'], 2)
B4 = S['B4']
put_set('b4', B4['tests'])
V.raw('b4.lm', month(B4['last_m']))
V.put('b4.last', B4['last'], 1)
for d in ['d0', 'd1']:
    o = B4[d]['order']
    V.raw(f'b4.{d}.o', f'ARIMA({o[0]},{o[1]},{o[2]})')
    V.put(f'b4.{d}.aicc', B4[d]['aicc'], 1)
    V.put(f'b4.{d}.f', B4[d]['f12'], 1)
    V.put(f'b4.{d}.lo', B4[d]['lo12'], 1)
    V.put(f'b4.{d}.hi', B4[d]['hi12'], 1)
    V.put(f'b4.{d}.q', B4[d]['lb12']['lb'], 0)
    V.raw(f'b4.{d}.qp', pv(B4[d]['lb12']['lb_p']))
V.put('b4.sum', B4['d0']['sum_ar'], 3)
V.put('b4.mu', B4['d0']['params']['const'], 1)
C1 = S['C1']
V.put('c1.pi.za', C1['pi']['za_c']['stat'], 2)
V.raw('c1.pi.d', month(C1['pi']['za_c']['break'][:7]))
V.put('c1.pi.zap', C1['pi']['za_c']['p'], 2)
V.put('c1.cv', C1['pi']['za_c']['crit5'], 2)
V.put('c1.cvct', C1['gdp']['za_ct']['crit5'], 2)
V.put('c1.g.za', C1['gdp']['za_c']['stat'], 2)
V.put('c1.g.zact', C1['gdp']['za_ct']['stat'], 2)
V.raw('c1.g.dct', qtr(C1['gdp']['za_ct']['break'][:4] + 'Q' + str((int(C1['gdp']['za_ct']['break'][5:7]) - 1) // 3 + 1)))
V.put('c1.g.zactp', C1['gdp']['za_ct']['p'], 3)
C2 = S['C2']
V.put('c2.adfp', C2['adf_gdp']['p'], 2)
V.put('c2.kpss', C2['kpss_gdp']['stat'], 3)
V.put('c2.t', C2['lev']['t'], 1)
V.put('c2.r2', C2['lev']['r2'], 2)
V.put('c2.dw', C2['lev']['dw'], 2)
V.put('c2.dt', C2['dif']['t'], 2)
V.put('c2.cv', C2['crit_c'], 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: does a series have a unit root, how many times must we difference it, and how wide are the forecast intervals of the resulting ARIMA model?',
       '\\textbf{Întrebarea}: are o serie rădăcină unitară, de cîte ori trebuie diferențiată și cît de largi sînt intervalele de prognoză ale modelului ARIMA obținut?'),
     [T('this seminar comes \\textbf{before} Lecture 3: the section ``What you need today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 3: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: a Dickey--Fuller statistic by hand; deterministic terms; ARIMA forecasts and intervals; the order of integration of an equation',
        'Partea A: o statistică Dickey--Fuller calculată de mînă; termenii determiniști; prognoze și intervale ARIMA; ordinul de integrare al unei ecuații'),
      T('Part B: unit-root tests on the BET, the S\\&P 500 and EUR/RON; an ARIMA for Romanian GDP; Romanian inflation with $d = 0$ and $d = 1$',
        'Partea B: teste de rădăcină unitară pentru BET, S\\&P 500 și EUR/RON; un model ARIMA pentru PIB-ul României; inflația din România cu $d = 0$ și $d = 1$'),
      T('Part C: an open question on structural breaks and an AI answer to audit', 'Partea C: o întrebare deschisă despre rupturi structurale și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('a Dickey--Fuller statistic by hand; choosing the deterministic terms', 'o statistică Dickey--Fuller calculată de mînă; alegerea termenilor determiniști') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('forecasts and intervals of ARIMA(1,1,0) and ARIMA(0,1,1)', 'prognoze și intervale pentru ARIMA(1,1,0) și ARIMA(0,1,1)') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('the order of integration read from an equation', 'ordinul de integrare citit dintr-o ecuație') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('unit-root tests: BET; S\\&P 500 and EUR/RON', 'teste de rădăcină unitară: BET; S\\&P 500 și EUR/RON') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('ARIMA: Romanian real GDP; Romanian inflation', 'ARIMA: PIB-ul real al României; inflația din România') + ' & ' + SP + ' & B3',
     'C1, C2 & ' + T('a break or a unit root? what is wrong in an AI answer?', 'ruptură sau rădăcină unitară? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, B3'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați dumneavoastră, după model')))

D.frame(T('Data used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Frequency}', '\\textbf{Frecvența}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['BET, S\\&P 500 & EODHD & ' + T('daily, close', 'zilnic, închidere') + ' & 2000--2026',
     'EUR/RON & ' + T('BNR reference rate', 'cursul de referință BNR') + ' & ' + T('daily', 'zilnic') + ' & 2005--2026',
     T('Real GDP, Romania', 'PIB real, România') + ' & Eurostat (namq\\_10\\_gdp) & ' + T('quarterly, seasonally adjusted', 'trimestrial, ajustat sezonier') + ' & 2000--2026',
     T('HICP, Romania', 'IAPC, România') + ' & Eurostat (prc\\_hicp\\_minr) & ' + T('monthly, 2015 = 100', 'lunar, 2015 = 100') + ' & 2004--2026'],
    size='footnotesize') + items(
    T('Prices and GDP as $100\\ln$: differences are then log returns or growth rates in \\%; 12-month inflation $\\pi_t = 100(\\ln P_t - \\ln P_{t-12})$',
      'Prețurile și PIB-ul ca $100\\ln$: diferențele sînt atunci randamente logaritmice sau rate de creștere în \\%; inflația anuală $\\pi_t = 100(\\ln P_t - \\ln P_{t-12})$'),
    T('In the notebook: \\texttt{adf\\_test}, \\texttt{pp\\_test}, \\texttt{kpss\\_test}, \\texttt{za\\_test}, \\texttt{arima\\_grid}; no account or key is needed',
      'În notebook: \\texttt{adf\\_test}, \\texttt{pp\\_test}, \\texttt{kpss\\_test}, \\texttt{za\\_test}, \\texttt{arima\\_grid}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What you need today', 'Noțiuni necesare azi')

D.frame(T('What you need today (1/4): trends and integration', 'Noțiuni necesare azi (1/4): trenduri și integrare'), items(
    (T('\\textbf{Trend-stationary}: $y_t = \\alpha + \\beta t + u_t$, $u_t$ stationary: shocks fade, the series returns to the line',
       '\\textbf{Staționară în jurul trendului}: $y_t = \\alpha + \\beta t + u_t$, $u_t$ staționar: șocurile se sting, seria revine la dreaptă'),
     [T('\\textbf{difference-stationary}: $\\Delta y_t = \\beta + u_t$: shocks are permanent (random walk with drift: $u_t$ white noise)', '\\textbf{staționară în diferențe}: $\\Delta y_t = \\beta + u_t$: șocurile sînt permanente (mers aleator cu derivă: $u_t$ zgomot alb)')]),
    (T('$y_t \\sim I(d)$, \\textbf{integrated of order $d$}: $\\Delta^d y_t$ is stationary, $\\Delta^{d-1}y_t$ is not', '$y_t \\sim I(d)$, \\textbf{integrat de ordinul $d$}: $\\Delta^d y_t$ este staționar, $\\Delta^{d-1}y_t$ nu este'),
     [T('$\\Delta y_t = y_t - y_{t-1}$; $\\Delta^2 y_t = y_t - 2y_{t-1} + y_{t-2}$', '$\\Delta y_t = y_t - y_{t-1}$; $\\Delta^2 y_t = y_t - 2y_{t-1} + y_{t-2}$')]),
    (T('\\textbf{Unit root}: the AR polynomial $\\phi(z) = 1 - \\phi_1 z - \\dots - \\phi_p z^p$ has the root $z = 1$', '\\textbf{Rădăcină unitară}: polinomul AR $\\phi(z) = 1 - \\phi_1 z - \\dots - \\phi_p z^p$ are rădăcina $z = 1$'),
     [T('then $\\phi(z) = (1 - z)\\phi^*(z)$ and $\\phi^*(L)\\Delta y_t = \\varepsilon_t$; a stationary AR has all roots with $|z| > 1$ (Chapter 2)', 'atunci $\\phi(z) = (1 - z)\\phi^*(z)$ și $\\phi^*(L)\\Delta y_t = \\varepsilon_t$; un AR staționar are toate rădăcinile cu $|z| > 1$ (Capitolul 2)'),
      T('\\textbf{spurious regression}: regressing one $I(1)$ series on another, unrelated one gives large $t$ and $R^2$', '\\textbf{regresie falsă}: regresia unei serii $I(1)$ pe o alta, fără legătură cu ea, dă valori mari pentru $t$ și $R^2$')])))

D.frame(T('What you need today (2/4): the Dickey--Fuller test', 'Noțiuni necesare azi (2/4): testul Dickey--Fuller'), items(
    (T('\\textbf{ADF regression}: $\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum_{j=1}^{k}\\delta_j\\Delta y_{t-j} + \\varepsilon_t$, $\\gamma = \\phi - 1$',
       '\\textbf{Regresia ADF}: $\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum_{j=1}^{k}\\delta_j\\Delta y_{t-j} + \\varepsilon_t$, $\\gamma = \\phi - 1$'),
     [T('$H_0$: $\\gamma = 0$ (unit root); $H_1$: $\\gamma < 0$ (stationary); $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$; reject if $\\tau$ is below the critical value',
        '$H_0$: $\\gamma = 0$ (rădăcină unitară); $H_1$: $\\gamma < 0$ (staționar); $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$; respingem dacă $\\tau$ este sub valoarea critică'),
      T('$k = 0$: the Dickey--Fuller (DF) test; $k$ chosen by AIC, at most $12(T/100)^{1/4}$', '$k = 0$: testul Dickey--Fuller (DF); $k$ ales după AIC, cel mult $12(T/100)^{1/4}$')]),
    (T('$\\tau$ does not follow a $t$ distribution: Dickey--Fuller critical values (5\\%, large $T$)', '$\\tau$ nu urmează o distribuție $t$: valorile critice Dickey--Fuller (5\\%, $T$ mare)'),
     [T('no constant $-1.94$; constant $-2.86$; constant and trend $-3.41$ (Normal: $-1.645$)', 'fără constantă $-1{,}94$; constantă $-2{,}86$; constantă și trend $-3{,}41$ (distribuția Normală: $-1{,}645$)')]),
    (T('Deterministic terms: trend for trending series (log GDP, log prices); constant for series around a level (inflation, returns)', 'Termenii determiniști: trend pentru seriile cu trend (logaritmul PIB, logaritmii prețurilor); constantă pentru seriile din jurul unui nivel (inflație, randamente)'),
     [T('\\textbf{Phillips--Perron} (PP): no lags, a corrected $\\tau$; same $H_0$ and critical values', '\\textbf{Phillips--Perron} (PP): fără decalaje, un $\\tau$ corectat; aceeași $H_0$ și aceleași valori critice')])))

D.frame(T('What you need today (3/4): KPSS and breaks', 'Noțiuni necesare azi (3/4): KPSS și rupturi'), items(
    (T('\\textbf{KPSS}: $H_0$: stationary (around a level or a trend); $H_1$: unit root; reject for \\textbf{large} values', '\\textbf{KPSS}: $H_0$: staționar (în jurul unui nivel sau al unui trend); $H_1$: rădăcină unitară; respingem pentru valori \\textbf{mari}'),
     [T('5\\% critical values: 0.463 (level), 0.146 (trend)', 'valori critice de 5\\%: 0,463 (nivel), 0,146 (trend)')]),
    (T('\\textbf{ADF and KPSS together}', '\\textbf{ADF și KPSS împreună}'),
     [T('ADF rejects, KPSS does not: $I(0)$; ADF does not, KPSS rejects: $I(1)$', 'ADF respinge, KPSS nu: $I(0)$; ADF nu respinge, KPSS respinge: $I(1)$'),
      T('neither rejects: inconclusive; both reject: conflict (often a break)', 'niciunul nu respinge: neconcludent; ambele resping: conflict (adesea o ruptură)')]),
    (T('\\textbf{Zivot--Andrews} (ZA): ADF with one break at an unknown date; the statistic is the minimum $t$ over all dates', '\\textbf{Zivot--Andrews} (ZA): ADF cu o ruptură la o dată necunoscută; statistica este $t$ minim pe toate datele'),
     [T('5\\% critical values: $-4.81$ (break in the level), $-5.07$ (level and trend)', 'valori critice de 5\\%: $-4{,}81$ (ruptură în nivel), $-5{,}07$ (nivel și trend)')])))

D.frame(T('What you need today (4/4): ARIMA models and their forecasts', 'Noțiuni necesare azi (4/4): modele ARIMA și prognozele lor'), items(
    (T('\\textbf{ARIMA$(p,d,q)$}: $\\phi(L)(1 - L)^d y_t = c + \\theta(L)\\varepsilon_t$: an ARMA$(p,q)$ for $\\Delta^d y_t$ (Chapter 2)', '\\textbf{ARIMA$(p,d,q)$}: $\\phi(L)(1 - L)^d y_t = c + \\theta(L)\\varepsilon_t$: un ARMA$(p,q)$ pentru $\\Delta^d y_t$ (Capitolul 2)'),
     [T('with $d = 1$, the constant $c$ is the drift: the average change per period', 'cu $d = 1$, constanta $c$ este deriva: variația medie pe perioadă')]),
    (T('\\textbf{Forecasts}: iterate the model for the levels with future shocks set to 0', '\\textbf{Prognozele}: aplicăm recursiv modelul pentru niveluri, cu șocurile viitoare egale cu 0'),
     [T('error variance at horizon $h$: $\\sigma^2(\\psi_0^2 + \\dots + \\psi_{h-1}^2)$, with $\\psi_0 = 1$; 95\\% interval: $\\pm 1.96\\times$ its square root', 'varianța erorii la orizontul $h$: $\\sigma^2(\\psi_0^2 + \\dots + \\psi_{h-1}^2)$, cu $\\psi_0 = 1$; intervalul de 95\\%: $\\pm 1{,}96\\times$ rădăcina ei pătrată'),
      T('ARIMA(1,1,0): $\\psi_j = 1 + \\phi + \\dots + \\phi^j$; ARIMA(0,1,1): $\\psi_j = 1 + \\theta$ for $j \\ge 1$; random walk: $\\psi_j = 1$', 'ARIMA(1,1,0): $\\psi_j = 1 + \\phi + \\dots + \\phi^j$; ARIMA(0,1,1): $\\psi_j = 1 + \\theta$ pentru $j \\ge 1$; mers aleator: $\\psi_j = 1$')]),
    (T('\\textbf{Model choice}: AICc (Chapter 2), lower is better, only between models with the same $d$', '\\textbf{Alegerea modelului}: AICc (Capitolul 2), valoarea mai mică este mai bună, doar între modele cu același $d$'),
     [T('residual check: Ljung--Box with $m - p - q$ degrees of freedom', 'verificarea reziduurilor: Ljung--Box cu $m - p - q$ grade de libertate')])))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: computations on paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: a Dickey--Fuller statistic', 'A1: o statistică Dickey--Fuller'),
         items(T('A monthly interest rate, $T = 120$: OLS gives $\\Delta y_t = 0.42 - 0.061\\,y_{t-1}$, with $\\mathrm{SE}(\\hat\\gamma) = 0.025$.', 'O rată lunară a dobînzii, $T = 120$: OLS dă $\\Delta y_t = 0{,}42 - 0{,}061\\,y_{t-1}$, cu $\\mathrm{SE}(\\hat\\gamma) = 0{,}025$.'),
               T('1. Compute $\\tau$ and $\\hat\\phi$.', '1. Calculați $\\tau$ și $\\hat\\phi$.'),
               T('2. Compare $\\tau$ with the 5\\% Dickey--Fuller value for a regression with a constant, $@{a1.cv}$, and decide.', '2. Comparați $\\tau$ cu valoarea Dickey--Fuller de 5\\% pentru o regresie cu constantă, $@{a1.cv}$, și decideți.'),
               T('3. Say what a standard one-sided $t$-test would have concluded.', '3. Precizați ce ar fi concluzionat un test $t$ unilateral obișnuit.'),
               T('4. Explain why a constant (and no trend) is the right choice for an interest rate.', '4. Explicați de ce o constantă (fără trend) este alegerea potrivită pentru o rată a dobînzii.'),
               T('Report: two numbers, a decision and two sentences.', 'Raportați: două valori, o decizie și două fraze.')),
         items(T('1. $\\tau = -0.061/0.025 = @{a1.tau}$; $\\hat\\phi = 1 - 0.061 = @{a1.phi}$', '1. $\\tau = -0{,}061/0{,}025 = @{a1.tau}$; $\\hat\\phi = 1 - 0{,}061 = @{a1.phi}$'),
               T('2. $@{a1.tau} > @{a1.cv}$: do not reject the unit root at 5\\% (p = @{a1.p}); also not at 10\\% ($@{a1.cv10}$)', '2. $@{a1.tau} > @{a1.cv}$: nu respingem rădăcina unitară la 5\\% (p = @{a1.p}); nici la 10\\% ($@{a1.cv10}$)'),
               T('3. $@{a1.tau} < -1.645$: the $t$-test would wrongly reject; under $H_0$ the statistic is not Student', '3. $@{a1.tau} < -1{,}645$: testul $t$ ar respinge greșit; în ipoteza $H_0$ statistica nu urmează legea Student'),
               T('4. An interest rate has no long-run trend; the plausible alternative is ``stationary around a mean\'\', which the constant represents.', '4. O rată a dobînzii nu are trend pe termen lung; alternativa plauzibilă este „staționară în jurul unei medii”, reprezentată de constantă.')),
         size='scriptsize')

D.proposed(T('A2: choosing the deterministic terms', 'A2: alegerea termenilor determiniști'),
           items(T('The log of a price index ($T = 200$) rises steadily. ADF: $\\tau = 1.85$ (no constant), $-1.21$ (constant), $-3.62$ (constant and trend); KPSS with trend: 0.09. Model: A1.',
                   'Logaritmul unui indice de preț ($T = 200$) crește constant. ADF: $\\tau = 1{,}85$ (fără constantă), $-1{,}21$ (constantă), $-3{,}62$ (constantă și trend); KPSS cu trend: 0,09. Model: A1.'),
                 T('1. Say which specification fits the data, and why.', '1. Precizați ce specificație se potrivește datelor și de ce.'),
                 T('2. Decide at 5\\% in each specification, with the critical values $@{a2.n}$ (no constant), $@{a2.c}$ (constant) and $@{a2.ct}$ (constant and trend).', '2. Decideți la 5\\% în fiecare specificație, cu valorile critice $@{a2.n}$ (fără constantă), $@{a2.c}$ (constantă) și $@{a2.ct}$ (constantă și trend).'),
                 T('3. Combine the chosen ADF result with KPSS (5\\% value 0.146).', '3. Combinați rezultatul ADF ales cu KPSS (valoarea de 5\\%: 0,146).'),
                 T('4. Say how you would model the series.', '4. Precizați cum ați modela seria.'),
                 T('Report: three decisions, a verdict and one sentence.', 'Raportați: trei decizii, un verdict și o frază.')),
           items(T('1. The series trends: constant and trend; without the trend, the alternative cannot describe the data', '1. Seria are trend: constantă și trend; fără trend, alternativa nu poate descrie datele'),
                 T('2. No constant: $1.85 > @{a2.n}$, do not reject; constant: $-1.21 > @{a2.c}$, do not reject; trend: $-3.62 < @{a2.ct}$, reject', '2. Fără constantă: $1{,}85 > @{a2.n}$, nu respingem; constantă: $-1{,}21 > @{a2.c}$, nu respingem; trend: $-3{,}62 < @{a2.ct}$, respingem'),
                 T('3. ADF rejects, KPSS $0.09 < 0.146$ does not: trend-stationary, $I(0)$ around a line', '3. ADF respinge, KPSS $0{,}09 < 0{,}146$ nu respinge: staționară în jurul trendului, $I(0)$ în jurul unei drepte'),
                 T('4. Regress on a linear trend and model the residuals as ARMA; differencing would over-difference.', '4. Estimăm regresia pe un trend liniar și modelăm reziduurile ca ARMA; diferențierea ar duce la supradiferențiere.')),
           size='scriptsize')

D.solved(T('A3: ARIMA(1,1,0) forecasts', 'A3: prognoze ARIMA(1,1,0)'),
         items(T('$\\Delta y_t = 0.6\\,\\Delta y_{t-1} + \\varepsilon_t$, $\\sigma = 2$; the last two values are $y_{T-1} = 103$ and $y_T = 108$.', '$\\Delta y_t = 0{,}6\\,\\Delta y_{t-1} + \\varepsilon_t$, $\\sigma = 2$; ultimele două valori sînt $y_{T-1} = 103$ și $y_T = 108$.'),
               T('1. Compute $\\hat y_{T+1}$, $\\hat y_{T+2}$ and $\\hat y_{T+3}$.', '1. Calculați $\\hat y_{T+1}$, $\\hat y_{T+2}$ și $\\hat y_{T+3}$.'),
               T('2. Compute $\\psi_1$ and $\\psi_2$, and the forecast error variances for $h = 1, 2, 3$.', '2. Calculați $\\psi_1$ și $\\psi_2$ și varianțele erorilor de prognoză pentru $h = 1, 2, 3$.'),
               T('3. Give the 95\\% intervals.', '3. Dați intervalele de 95\\%.'),
               T('Report: three forecasts and three intervals.', 'Raportați: trei prognoze și trei intervale.')),
         items(T('1. $\\Delta y_T = 5$; $\\widehat{\\Delta y}_{T+1} = 3$, $\\hat y_{T+1} = @{a3.f0}$; $\\widehat{\\Delta y}_{T+2} = 1.8$, $\\hat y_{T+2} = @{a3.f1}$; $\\widehat{\\Delta y}_{T+3} = 1.08$, $\\hat y_{T+3} = @{a3.f2}$',
                 '1. $\\Delta y_T = 5$; $\\widehat{\\Delta y}_{T+1} = 3$, $\\hat y_{T+1} = @{a3.f0}$; $\\widehat{\\Delta y}_{T+2} = 1{,}8$, $\\hat y_{T+2} = @{a3.f1}$; $\\widehat{\\Delta y}_{T+3} = 1{,}08$, $\\hat y_{T+3} = @{a3.f2}$'),
               T('2. $\\psi_1 = 1.6$, $\\psi_2 = 1.96$; variances $4$, $4(1 + 2.56) = @{a3.v1}$, $4(1 + 2.56 + 3.8416) = @{a3.v2}$', '2. $\\psi_1 = 1{,}6$, $\\psi_2 = 1{,}96$; varianțele $4$, $4(1 + 2{,}56) = @{a3.v1}$, $4(1 + 2{,}56 + 3{,}8416) = @{a3.v2}$'),
               T('3. $[@{a3.lo0}, @{a3.hi0}]$; $[@{a3.lo1}, @{a3.hi1}]$; $[@{a3.lo2}, @{a3.hi2}]$: half-widths @{a3.h0}, @{a3.h1}, @{a3.h2}', '3. $[@{a3.lo0}, @{a3.hi0}]$; $[@{a3.lo1}, @{a3.hi1}]$; $[@{a3.lo2}, @{a3.hi2}]$: semilățimi @{a3.h0}, @{a3.h1}, @{a3.h2}')),
         size='scriptsize')

D.proposed(T('A4: ARIMA(0,1,1) and exponential smoothing', 'A4: ARIMA(0,1,1) și netezirea exponențială'),
           items(T('$\\Delta y_t = \\varepsilon_t - 0.6\\,\\varepsilon_{t-1}$, $\\sigma = 1$; $y_T = 50$ and the last residual is $\\hat\\varepsilon_T = 1.5$. Model: A3.', '$\\Delta y_t = \\varepsilon_t - 0{,}6\\,\\varepsilon_{t-1}$, $\\sigma = 1$; $y_T = 50$, iar ultimul reziduu este $\\hat\\varepsilon_T = 1{,}5$. Model: A3.'),
                 T('1. Compute $\\hat y_{T+1}$ and $\\hat y_{T+h}$ for $h \\ge 2$.', '1. Calculați $\\hat y_{T+1}$ și $\\hat y_{T+h}$ pentru $h \\ge 2$.'),
                 T('2. Compute the forecast error variances for $h = 1$, $h = 2$ and $h = 5$, and the 95\\% intervals.', '2. Calculați varianțele erorilor de prognoză pentru $h = 1$, $h = 2$ și $h = 5$ și intervalele de 95\\%.'),
                 T('3. Show that the forecast equals simple exponential smoothing (Chapter 0) and give $\\alpha$.', '3. Arătați că prognoza coincide cu netezirea exponențială simplă (Capitolul 0) și dați $\\alpha$.'),
                 T('Report: two forecasts, three intervals and $\\alpha$.', 'Raportați: două prognoze, trei intervale și $\\alpha$.')),
           items(T('1. $\\hat y_{T+1} = y_T - 0.6\\,\\hat\\varepsilon_T = @{a4.f}$; then $\\hat y_{T+h} = @{a4.f}$: a flat forecast', '1. $\\hat y_{T+1} = y_T - 0{,}6\\,\\hat\\varepsilon_T = @{a4.f}$; apoi $\\hat y_{T+h} = @{a4.f}$: o prognoză orizontală'),
                 T('2. $\\psi_j = 1 - 0.6 = 0.4$: variances $1$, $1.16$, $1 + 4 \\cdot 0.16 = @{a4.v4}$; intervals $[@{a4.lo0}, @{a4.hi0}]$, $[@{a4.lo1}, @{a4.hi1}]$, $[@{a4.lo4}, @{a4.hi4}]$',
                   '2. $\\psi_j = 1 - 0{,}6 = 0{,}4$: varianțele $1$; $1{,}16$; $1 + 4 \\cdot 0{,}16 = @{a4.v4}$; intervalele $[@{a4.lo0}, @{a4.hi0}]$, $[@{a4.lo1}, @{a4.hi1}]$, $[@{a4.lo4}, @{a4.hi4}]$'),
                 T('3. $\\hat y_{T+1} = y_T - 0.6(y_T - \\hat y_T) = 0.4\\,y_T + 0.6\\,\\hat y_T$: SES with $\\alpha = 1 + \\theta = 0.4$', '3. $\\hat y_{T+1} = y_T - 0{,}6(y_T - \\hat y_T) = 0{,}4\\,y_T + 0{,}6\\,\\hat y_T$: SES cu $\\alpha = 1 + \\theta = 0{,}4$')),
           size='scriptsize')

D.solved(T('A5: the order of integration of an equation', 'A5: ordinul de integrare al unei ecuații'),
         items(T('$y_t = 1.5\\,y_{t-1} - 0.5\\,y_{t-2} + \\varepsilon_t + 0.4\\,\\varepsilon_{t-1}$.', '$y_t = 1{,}5\\,y_{t-1} - 0{,}5\\,y_{t-2} + \\varepsilon_t + 0{,}4\\,\\varepsilon_{t-1}$.'),
               T('1. Write the AR polynomial $\\phi(z)$ and find its roots.', '1. Scrieți polinomul AR $\\phi(z)$ și aflați rădăcinile lui.'),
               T('2. Say whether $y_t$ is stationary, and give $d$.', '2. Precizați dacă $y_t$ este staționar și dați $d$.'),
               T('3. Write the model as ARIMA$(p,d,q)$ for $\\Delta y_t$.', '3. Scrieți modelul ca ARIMA$(p,d,q)$ pentru $\\Delta y_t$.'),
               T('Report: two roots, $d$ and the ARIMA equation.', 'Raportați: două rădăcini, $d$ și ecuația ARIMA.')),
         items(T('1. $\\phi(z) = 1 - 1.5z + 0.5z^2 = (1 - z)(1 - 0.5z)$: roots $z = 1$ and $z = 2$', '1. $\\phi(z) = 1 - 1{,}5z + 0{,}5z^2 = (1 - z)(1 - 0{,}5z)$: rădăcinile $z = 1$ și $z = 2$'),
               T('2. One root on the unit circle: not stationary; the other root is outside: $d = 1$', '2. O rădăcină pe cercul unitate: nestaționar; cealaltă rădăcină este în afara cercului: $d = 1$'),
               T('3. $(1 - 0.5L)\\Delta y_t = (1 + 0.4L)\\varepsilon_t$: ARIMA(1,1,1), $\\Delta y_t = 0.5\\,\\Delta y_{t-1} + \\varepsilon_t + 0.4\\,\\varepsilon_{t-1}$', '3. $(1 - 0{,}5L)\\Delta y_t = (1 + 0{,}4L)\\varepsilon_t$: ARIMA(1,1,1), $\\Delta y_t = 0{,}5\\,\\Delta y_{t-1} + \\varepsilon_t + 0{,}4\\,\\varepsilon_{t-1}$')),
         size='scriptsize')

D.proposed(T('A6: integrated or over-differenced?', 'A6: integrat sau supradiferențiat?'),
           items(T('Two models: (a) $y_t = 1.8\\,y_{t-1} - 0.8\\,y_{t-2} + \\varepsilon_t$; (b) $\\Delta y_t = \\varepsilon_t - \\varepsilon_{t-1}$. Model: A5.', 'Două modele: (a) $y_t = 1{,}8\\,y_{t-1} - 0{,}8\\,y_{t-2} + \\varepsilon_t$; (b) $\\Delta y_t = \\varepsilon_t - \\varepsilon_{t-1}$. Model: A5.'),
                 T('1. For (a), factor $\\phi(z)$, give $d$ and write the ARIMA model.', '1. Pentru (a), factorizați $\\phi(z)$, dați $d$ și scrieți modelul ARIMA.'),
                 T('2. For (b), find the MA root and say what $y_t$ really is.', '2. Pentru (b), aflați rădăcina MA și precizați ce este de fapt $y_t$.'),
                 T('3. Compute $\\rho(1)$ of $\\Delta y_t$ in (b) and say how you would detect this case in data.', '3. Calculați $\\rho(1)$ al lui $\\Delta y_t$ în (b) și precizați cum ați detecta acest caz în date.'),
                 T('Report: two orders of integration and two sentences.', 'Raportați: două ordine de integrare și două fraze.')),
           items(T('1. $1 - 1.8z + 0.8z^2 = (1 - z)(1 - 0.8z)$: roots 1 and 1.25; $d = 1$: ARIMA(1,1,0), $\\Delta y_t = 0.8\\,\\Delta y_{t-1} + \\varepsilon_t$', '1. $1 - 1{,}8z + 0{,}8z^2 = (1 - z)(1 - 0{,}8z)$: rădăcinile 1 și 1,25; $d = 1$: ARIMA(1,1,0), $\\Delta y_t = 0{,}8\\,\\Delta y_{t-1} + \\varepsilon_t$'),
                 T('2. $\\theta(z) = 1 - z$ has the root 1: the $(1 - L)$ cancels, $y_t = \\varepsilon_t$ plus a constant: white noise, $d = 0$', '2. $\\theta(z) = 1 - z$ are rădăcina 1: factorul $(1 - L)$ se simplifică, $y_t = \\varepsilon_t$ plus o constantă: zgomot alb, $d = 0$'),
                 T('3. $\\rho(1) = -1/2$; in data: $\\hat\\rho(1) \\approx -0.5$ after differencing, a larger variance, an MA estimate near $-1$', '3. $\\rho(1) = -1/2$; în date: $\\hat\\rho(1) \\approx -0{,}5$ după diferențiere, o varianță mai mare, o estimare MA apropiată de $-1$')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: real data and interpretation', 'Partea B: date reale și interpretare')

D.task(T('B1: is the BET log price a random walk? [Solved]', 'B1: este logaritmul prețului BET un mers aleator? [Rezolvat]'),
       T('does the BET log price have a unit root, and are its daily returns stationary?', 'are logaritmul prețului BET o rădăcină unitară și sînt randamentele lui zilnice staționare?'),
       T('BET closes since 2000; $p_t = 100\\ln P_t$ and $r_t = \\Delta p_t$', 'închiderile BET din 2000; $p_t = 100\\ln P_t$ și $r_t = \\Delta p_t$'),
       [T('Plot $p_t$ with a fitted linear trend, and plot $r_t$.', 'Reprezentați grafic $p_t$ împreună cu un trend liniar estimat și reprezentați grafic $r_t$.'),
        T('Run ADF (lags by AIC), PP and KPSS on $p_t$ with constant and trend.', 'Aplicați ADF (decalaje după AIC), PP și KPSS pe $p_t$, cu constantă și trend.'),
        T('Run the same tests on $r_t$ with a constant.', 'Aplicați aceleași teste pe $r_t$, cu constantă.'),
        T('Interpretation: if the BET is far below its fitted trend line, should you expect it to return to the line?', 'Interpretare: dacă BET se află mult sub dreapta de trend estimată, ar trebui să vă așteptați ca el să revină la dreaptă?')],
       T('a table of six statistics, two verdicts and two sentences', 'un tabel cu șase statistici, două verdicte și două fraze'), size='footnotesize', nb='B1')

D.frame(T('B1: solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch3_sem_b1', h='0.36') + table(
    'lrrrl', T('& ADF (p, $k$) & PP & KPSS & ' + '\\textbf{verdict}', '& ADF (p, $k$) & PP & KPSS & \\textbf{verdict}'),
    [T('$p_t$ (c, t)', '$p_t$ (c, t)') + ' & $@{b1.level.adf}$ (@{b1.level.adfp}, @{b1.level.lags}) & $@{b1.level.pp}$ & @{b1.level.kpss} & $I(1)$',
     T('$r_t$ (c)', '$r_t$ (c)') + ' & $@{b1.diff.adf}$ (@{b1.diff.adfp}, @{b1.diff.lags}) & $@{b1.diff.pp}$ & @{b1.diff.kpss} & $I(0)$'],
    size='scriptsize') + items(
    T('$p_t$: ADF and PP far above $@{b1.level.cv}$, KPSS $> @{b1.level.kcv}$: unit root; $r_t$: both reject strongly, KPSS $< @{b1.diff.kcv}$: stationary',
      '$p_t$: ADF și PP mult peste $@{b1.level.cv}$, KPSS $> @{b1.level.kcv}$: rădăcină unitară; $r_t$: ambele resping puternic, KPSS $< @{b1.diff.kcv}$: staționar'),
    T('Interpretation: no; with a unit root the trend line is not an attractor; at the end of the sample the index is @{b1.dev} log points from the line, and nothing pulls it back',
      'Interpretare: nu; cu o rădăcină unitară, dreapta de trend nu atrage seria; la sfîrșitul eșantionului indicele se află la @{b1.dev} puncte logaritmice de dreaptă și nimic nu îl readuce')) + qlsem(),
    'scriptsize')

D.task(T('B2: the S\\&P 500 and EUR/RON [Proposed]', 'B2: S\\&P 500 și EUR/RON [Propus]'),
       T('do the S\\&P 500 and the EUR/RON rate give the same verdicts as the BET?', 'dau S\\&P 500 și cursul EUR/RON aceleași verdicte ca BET?'),
       T('S\\&P 500 since 2000; EUR/RON (BNR reference rate) since July 2005; both as $100\\ln$; model: B1', 'S\\&P 500 din 2000; EUR/RON (cursul de referință BNR) din iulie 2005; ambele ca $100\\ln$; model: B1'),
       [T('Repeat steps 2 and 3 of B1 for both series.', 'Repetați pașii 2 și 3 din B1 pentru ambele serii.'),
        T('Compare the ADF and PP statistics of the returns, and explain why PP is much more negative.', 'Comparați statisticile ADF și PP ale randamentelor și explicați de ce PP este mult mai negativ.'),
        T('Interpretation: the EUR/RON rate is managed by the central bank; does the test result say that it is a random walk?', 'Interpretare: cursul EUR/RON este administrat de banca centrală; spune rezultatul testului că el este un mers aleator?')],
       T('a table (two series, two rows each) and two sentences', 'un tabel (două serii, cîte două rînduri) și două fraze'), size='footnotesize', nb='B2')


def b2row(lab, k, part, v):
    return f'{lab} & $@{{{k}.{part}.adf}}$ (@{{{k}.{part}.adfp}}, @{{{k}.{part}.lags}}) & $@{{{k}.{part}.pp}}$ & @{{{k}.{part}.kpss}} & {v}'


D.frame(T('B2: solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrl', T('& ADF (p, $k$) & PP & KPSS & \\textbf{verdict}', '& ADF (p, $k$) & PP & KPSS & \\textbf{verdict}'),
    [b2row('S\\&P 500, $p_t$ (c, t)', 'b2s', 'level', '$I(1)$'), b2row(T('S\\&P 500, $r_t$ (c)', 'S\\&P 500, $r_t$ (c)'), 'b2s', 'diff', '$I(0)$'),
     b2row('EUR/RON, $s_t$ (c, t)', 'b2e', 'level', '$I(1)$'), b2row('EUR/RON, $\\Delta s_t$ (c)', 'b2e', 'diff', '$I(0)$')],
    size='scriptsize') + items(
    T('Both levels: unit root (ADF and PP do not reject, KPSS rejects); both returns: stationary', 'Ambele niveluri: rădăcină unitară (ADF și PP nu resping, KPSS respinge); ambele serii de randamente: staționare'),
    T('PP uses no lags and corrects $\\tau$ with the long-run variance; with thousands of observations both reject, PP more strongly; the verdict is the same',
      'PP nu folosește decalaje și corectează $\\tau$ prin varianța pe termen lung; cu mii de observații ambele resping, PP mai puternic; verdictul este același'),
    T('Interpretation: no; the tests only say that the rate does not return to a fixed mean or line; a managed rate with rare large adjustments can look like this (Lecture 3, Zivot--Andrews)',
      'Interpretare: nu; testele spun doar că seria nu revine la o medie sau la o dreaptă fixă; un curs administrat, cu ajustări mari și rare, poate arăta la fel (Cursul 3, Zivot--Andrews)')) + qlsem(),
    'scriptsize', instructor_only=True)

D.task(T('B3: an ARIMA model for Romanian GDP [Solved]', 'B3: un model ARIMA pentru PIB-ul României [Rezolvat]'),
       T('which ARIMA model describes Romanian real GDP, and how uncertain is a two-year forecast?', 'ce model ARIMA descrie PIB-ul real al României și cît de incertă este o prognoză pe doi ani?'),
       T('real GDP, chain-linked volumes (2010), seasonally adjusted, Eurostat, from 2000Q1; $y_t = 100\\ln Y_t$', 'PIB-ul real, volume înlănțuite (2010), ajustat sezonier, Eurostat, din T1 2000; $y_t = 100\\ln Y_t$'),
       [T('Choose $d$ with ADF and KPSS on $y_t$ (constant and trend) and on $\\Delta y_t$ (constant).', 'Alegeți $d$ cu ADF și KPSS pe $y_t$ (constantă și trend) și pe $\\Delta y_t$ (constantă).'),
        T('Fit ARIMA$(p,1,q)$ with drift for $p, q \\le 2$ and choose the model with the lowest AICc.', 'Estimați ARIMA$(p,1,q)$ cu derivă pentru $p, q \\le 2$ și alegeți modelul cu cel mai mic AICc.'),
        T('Check the residuals with Ljung--Box $Q^*(8)$.', 'Verificați reziduurile cu Ljung--Box $Q^*(8)$.'),
        T('Forecast 8 quarters with 80\\% and 95\\% intervals.', 'Prognozați 8 trimestre, cu intervale de 80\\% și 95\\%.'),
        T('Interpretation: why is the 95\\% interval after 8 quarters about twice as wide as after 2 quarters?', 'Interpretare: de ce este intervalul de 95\\% după 8 trimestre de aproximativ două ori mai larg decît după 2 trimestre?')],
       T('the tests, the AICc of six models, the chosen model, $Q^*(8)$, the chart and one sentence', 'testele, AICc pentru șase modele, modelul ales, $Q^*(8)$, graficul și o frază'), size='footnotesize', nb='B3')

D.frame(T('B3: solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch3_sem_b3', h='0.34') + items(
    T('$y_t$: ADF $@{b3.level.adf}$ (p = @{b3.level.adfp}), KPSS @{b3.level.kpss} $> 0.146$: $I(1)$; $\\Delta y_t$: ADF $@{b3.diff.adf}$, KPSS @{b3.diff.kpss}: $I(0)$; so $d = 1$',
      '$y_t$: ADF $@{b3.level.adf}$ (p = @{b3.level.adfp}), KPSS @{b3.level.kpss} $> 0{,}146$: $I(1)$; $\\Delta y_t$: ADF $@{b3.diff.adf}$, KPSS @{b3.diff.kpss}: $I(0)$; deci $d = 1$'),
    T('AICc: (0,1,0) @{b3.g00}; (1,1,0) @{b3.g10}; (0,1,1) @{b3.g01}; (1,1,1) @{b3.g11}; (2,1,0) @{b3.g20}; (0,1,2) @{b3.g02}: @{b3.o} with drift $\\hat c = @{b3.c}$ (SE @{b3.cse}), $\\hat\\sigma = @{b3.sig}$',
      'AICc: (0,1,0) @{b3.g00}; (1,1,0) @{b3.g10}; (0,1,1) @{b3.g01}; (1,1,1) @{b3.g11}; (2,1,0) @{b3.g20}; (0,1,2) @{b3.g02}: @{b3.o} cu deriva $\\hat c = @{b3.c}$ (SE @{b3.cse}), $\\hat\\sigma = @{b3.sig}$'),
    T('$Q^*(8) = @{b3.q8}$ (p = @{b3.q8p}): white-noise residuals; forecast for @{b3.fq}: @{b3.f8}, i.e.\\ $+@{b3.g8}$\\% in two years, 95\\% interval $[@{b3.lo8}, @{b3.hi8}]$',
      '$Q^*(8) = @{b3.q8}$ (p = @{b3.q8p}): reziduuri de tip zgomot alb; prognoza pentru @{b3.fq}: @{b3.f8}, adică $+@{b3.g8}$\\% în doi ani, intervalul de 95\\% $[@{b3.lo8}, @{b3.hi8}]$'),
    T('Interpretation: for a random walk the error variance is $h\\sigma^2$, so the width grows like $\\sqrt{h}$: $\\sqrt{8/2} = 2$ (half-widths @{b3.h2} and @{b3.h8}, ratio @{b3.ratio})',
      'Interpretare: pentru un mers aleator varianța erorii este $h\\sigma^2$, deci lățimea crește ca $\\sqrt{h}$: $\\sqrt{8/2} = 2$ (semilățimile @{b3.h2} și @{b3.h8}, raportul @{b3.ratio})')) + qlsem(),
    'scriptsize')

D.task(T('B4: Romanian inflation, $d = 0$ or $d = 1$? [Proposed]', 'B4: inflația din România, $d = 0$ sau $d = 1$? [Propus]'),
       T('is Romanian 12-month inflation stationary, and how much does the answer change the forecast?', 'este inflația anuală din România staționară și cît de mult schimbă răspunsul prognoza?'),
       T('$\\pi_t = 100(\\ln P_t - \\ln P_{t-12})$, HICP, January 2005 to @{b4.lm}; model: B3', '$\\pi_t = 100(\\ln P_t - \\ln P_{t-12})$, IAPC, ianuarie 2005--@{b4.lm}; model: B3'),
       [T('Run ADF, PP and KPSS on $\\pi_t$ and on $\\Delta\\pi_t$, with a constant.', 'Aplicați ADF, PP și KPSS pe $\\pi_t$ și pe $\\Delta\\pi_t$, cu constantă.'),
        T('Find the best ARIMA$(p,0,q)$ with a mean and the best ARIMA$(p,1,q)$ by AICc, with $p \\le 3$, $q \\le 2$.', 'Găsiți cel mai bun ARIMA$(p,0,q)$ cu medie și cel mai bun ARIMA$(p,1,q)$ după AICc, cu $p \\le 3$, $q \\le 2$.'),
        T('Forecast 12 months with both models, with 95\\% intervals, and run Ljung--Box $Q^*(24)$ on their residuals.', 'Prognozați 12 luni cu ambele modele, cu intervale de 95\\%, și aplicați Ljung--Box $Q^*(24)$ pe reziduurile lor.'),
        T('Interpretation: why do the two models give similar point forecasts but different intervals?', 'Interpretare: de ce dau cele două modele prognoze punctuale similare, dar intervale diferite?')],
       T('a table of tests, two models, two forecasts with intervals, two Ljung--Box tests and two sentences', 'un tabel cu testele, două modele, două prognoze cu intervale, două teste Ljung--Box și două fraze'), size='footnotesize', nb='B4')

D.frame(T('B4: solution [Proposed]', 'B4: rezolvare [Propus]'), fig('ch3_sem_b4', h='0.34') + items(
    T('$\\pi_t$: ADF $@{b4.level.adf}$ (p = @{b4.level.adfp}), KPSS @{b4.level.kpss} $< 0.463$: inconclusive; $\\Delta\\pi_t$: ADF $@{b4.diff.adf}$, KPSS @{b4.diff.kpss}: $I(0)$',
      '$\\pi_t$: ADF $@{b4.level.adf}$ (p = @{b4.level.adfp}), KPSS @{b4.level.kpss} $< 0{,}463$: neconcludent; $\\Delta\\pi_t$: ADF $@{b4.diff.adf}$, KPSS @{b4.diff.kpss}: $I(0)$'),
    T('$d = 0$: @{b4.d0.o}, AR sum @{b4.sum}, mean @{b4.mu}\\%; 12 months: @{b4.d0.f}\\% in $[@{b4.d0.lo}, @{b4.d0.hi}]$. $d = 1$: @{b4.d1.o}; @{b4.d1.f}\\% in $[@{b4.d1.lo}, @{b4.d1.hi}]$',
      '$d = 0$: @{b4.d0.o}, suma coeficienților AR @{b4.sum}, media @{b4.mu}\\%; 12 luni: @{b4.d0.f}\\% în $[@{b4.d0.lo}, @{b4.d0.hi}]$. $d = 1$: @{b4.d1.o}; @{b4.d1.f}\\% în $[@{b4.d1.lo}, @{b4.d1.hi}]$'),
    T('$Q^*(24)$: @{b4.d0.q} (p @{b4.d0.qp}) and @{b4.d1.q} (p @{b4.d1.qp}): both leave autocorrelation, since 12-month rates overlap (Chapter 4 models monthly inflation)',
      '$Q^*(24)$: @{b4.d0.q} (p @{b4.d0.qp}) și @{b4.d1.q} (p @{b4.d1.qp}): ambele lasă autocorelație, deoarece ratele anuale se suprapun (Capitolul 4 modelează inflația lunară)'),
    T('Interpretation: the AR sum @{b4.sum} is almost 1, so in the short run both models move alike; only $d = 1$ lets the error variance grow without bound, hence the wider interval',
      'Interpretare: suma coeficienților AR, @{b4.sum}, este aproape 1, deci pe termen scurt modelele evoluează la fel; doar $d = 1$ lasă varianța erorii să crească nelimitat, de aici intervalul mai larg')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: open questions and AI critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: a break or a unit root? [Proposed]', 'C1: ruptură sau rădăcină unitară? [Propus]'),
       T('do Romanian inflation and GDP have a unit root, or are they stationary around a mean or a trend that shifted once?', 'au inflația și PIB-ul României o rădăcină unitară sau sînt staționare în jurul unei medii ori al unui trend care s-a schimbat o dată?'),
       T('$\\pi_t$ of B4 and $y_t$ of B3; models: B1, B3', '$\\pi_t$ din B4 și $y_t$ din B3; modele: B1, B3'),
       [T('Run the Zivot--Andrews test with a break in the level, then in the level and the trend, on both series.', 'Aplicați testul Zivot--Andrews cu ruptură în nivel, apoi în nivel și în trend, pe ambele serii.'),
        T('Compare each statistic with its 5\\% critical value and report the break dates.', 'Comparați fiecare statistică cu valoarea critică de 5\\% și raportați datele rupturilor.'),
        T('Match the dates with events (inflation targeting since 2005, the 2008--2009 crisis, the 2022 energy shock).', 'Asociați datele cu evenimente (țintirea inflației din 2005, criza din 2008--2009, șocul energetic din 2022).'),
        T('Interpretation: why should a break date found by a test be checked against history before it is believed?', 'Interpretare: de ce trebuie comparată cu istoria o dată de ruptură găsită de un test înainte de a o accepta?')],
       T('a table of four statistics with dates, the chart and a plan for a project', 'un tabel cu patru statistici și datele lor, graficul și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: reference analysis [Proposed]', 'C1: analiză de referință [Propus]'), fig('ch3_sem_c1', h='0.34') + items(
    T('Inflation: ZA (level) $= @{c1.pi.za}$ (p = @{c1.pi.zap}), above $@{c1.cv}$, break in @{c1.pi.d}: no evidence against the unit root',
      'Inflația: ZA (nivel) $= @{c1.pi.za}$ (p = @{c1.pi.zap}), peste $@{c1.cv}$, ruptură în @{c1.pi.d}: nicio dovadă împotriva rădăcinii unitare'),
    T('GDP: ZA (level) $= @{c1.g.za}$, not significant; ZA (level and trend) $= @{c1.g.zact}$ $< @{c1.cvct}$ (p = @{c1.g.zactp}), break in @{c1.g.dct}: stationary around a trend that broke in the 2008 crisis?',
      'PIB: ZA (nivel) $= @{c1.g.za}$, nesemnificativ; ZA (nivel și trend) $= @{c1.g.zact}$ $< @{c1.cvct}$ (p = @{c1.g.zactp}), ruptură în @{c1.g.dct}: staționar în jurul unui trend care s-a rupt în criza din 2008?'),
    T('Interpretation: the test picks the date that best fits the data; a date without an economic event, or found after trying several specifications, may be an accident of the sample',
      'Interpretare: testul alege data care se potrivește cel mai bine datelor; o dată fără un eveniment economic sau găsită după încercarea mai multor specificații poate fi un accident al eșantionului'),
    T('Project: breaks and persistence of inflation in Romania and in the region, with Bai--Perron for several breaks', 'Proiect: rupturi și persistența inflației în România și în regiune, cu testul Bai--Perron pentru mai multe rupturi')) + qlsem(),
    'scriptsize', instructor_only=True)

D.frame(T('C2: audit an AI answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to analyse Romanian GDP and prices. The answer:', 'Un student a cerut unui asistent AI să analizeze PIB-ul și prețurile din România. Răspunsul:'),
    T('\\aiprompt{(a) The ADF p-value for log GDP is @{c2.adfp}, so GDP is proved to have a unit root.}', '\\aiprompt{(a) Valoarea p ADF pentru logaritmul PIB este @{c2.adfp}, deci s-a demonstrat că PIB-ul are rădăcină unitară.}'),
    T('\\aiprompt{(b) The KPSS statistic for log GDP is @{c2.kpss}, above 0.146, so KPSS rejects the unit root.}', '\\aiprompt{(b) Statistica KPSS pentru logaritmul PIB este @{c2.kpss}, peste 0,146, deci KPSS respinge rădăcina unitară.}'),
    T('\\aiprompt{(c) Regressing log HICP on the log S\\&P 500 gives t = @{c2.t} and R2 = @{c2.r2}: US stocks drive Romanian prices.}', '\\aiprompt{(c) Regresia logaritmului IAPC pe logaritmul S\\&P 500 dă t = @{c2.t} și R2 = @{c2.r2}: acțiunile americane determină prețurile din România.}'),
    T('\\aiprompt{(d) GDP growth is stationary, so differencing it once more will make the ARIMA model even better.}', '\\aiprompt{(d) Creșterea PIB este staționară, deci încă o diferențiere va face modelul ARIMA și mai bun.}'),
    T('\\aiprompt{(e) A random-walk forecast has the same 95\\% interval width at every horizon.}', '\\aiprompt{(e) O prognoză de tip mers aleator are aceeași lățime a intervalului de 95\\% la orice orizont.}'),
    T('\\aiprompt{(f) The 5\\% Dickey-Fuller critical value with a constant is about @{c2.cv}, more negative than -1.645, because the Dickey-Fuller distribution is shifted to the left.}', '\\aiprompt{(f) Valoarea critică Dickey-Fuller de 5\\% cu constantă este circa @{c2.cv}, mai negativă decît -1,645, pentru că distribuția Dickey-Fuller este deplasată la stînga.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: not rejecting is not proof; with 106 quarters the test has little power against $\\phi$ near 1', '(a) Greșit: nerespingerea nu este o demonstrație; cu 106 trimestre testul are putere mică împotriva unui $\\phi$ apropiat de 1'),
    T('(b) Wrong: the KPSS null is stationarity; @{c2.kpss} $> 0.146$ rejects stationarity, which supports the unit root', '(b) Greșit: ipoteza nulă KPSS este staționaritatea; @{c2.kpss} $> 0{,}146$ respinge staționaritatea, ceea ce susține rădăcina unitară'),
    T('(c) Wrong: a spurious regression of two trending series; DW = @{c2.dw} $< R^2$; in differences $t = @{c2.dt}$', '(c) Greșit: o regresie falsă între două serii cu trend; DW = @{c2.dw} $< R^2$; în diferențe $t = @{c2.dt}$'),
    T('(d) Wrong: over-differencing; the MA root moves to the unit circle and the variance grows', '(d) Greșit: supradiferențiere; rădăcina MA ajunge pe cercul unitate, iar varianța crește'),
    T('(e) Wrong: the variance is $h\\sigma^2$, so the width grows like $\\sqrt{h}$', '(e) Greșit: varianța este $h\\sigma^2$, deci lățimea crește ca $\\sqrt{h}$'),
    T('(f) Correct', '(f) Corect')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-up', 'Încheiere')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Compare $\\tau$ with Dickey--Fuller critical values, never with Normal ones; choose the deterministic terms from the plot', 'Comparați $\\tau$ cu valorile critice Dickey--Fuller, niciodată cu cele ale distribuției Normale; alegeți termenii determiniști după grafic'),
    T('ADF and KPSS have opposite null hypotheses: read them together', 'ADF și KPSS au ipoteze nule opuse: interpretați-le împreună'),
    T('Prices, exchange rates and GDP in logs are $I(1)$; returns and growth rates are $I(0)$', 'Prețurile, cursurile de schimb și PIB-ul în logaritmi sînt $I(1)$; randamentele și ratele de creștere sînt $I(0)$'),
    T('With $d = 1$, forecast intervals grow with the horizon; the choice of $d$ matters most for long horizons', 'Cu $d = 1$, intervalele de prognoză cresc cu orizontul; alegerea lui $d$ contează mai ales pe orizonturi lungi'),
    T('An AI answer is a draft: check the direction of each test and the critical values used', 'Un răspuns AI este o ciornă: verificați sensul fiecărui test și valorile critice folosite')))

D.frame(T('After the seminar', 'După seminar'), items(
    T('Lecture 3 develops each topic of today: trends, spurious regression, the Dickey--Fuller distributions, PP, KPSS, breaks, ARIMA forecasting and automatic selection',
      'Cursul 3 dezvoltă fiecare temă de azi: trendurile, regresia falsă, distribuțiile Dickey--Fuller, PP, KPSS, rupturile, prognoza ARIMA și selecția automată'),
    T('Try the [Proposed] tasks in the notebook', 'Încercați cerințele [Propus] în notebook'),
    T('C1 can grow into a team project: breaks and persistence of inflation in Romania, Hungary, Poland and Czechia', 'C1 poate deveni un proiect de echipă: rupturi și persistența inflației în România, Ungaria, Polonia și Cehia'),
    T('Reading: \\refHP, Ch.~4--5; \\refFPP, Ch.~9; \\refHamilton, Ch.~15--17', 'Lectură: \\refHP, cap.~4--5; \\refFPP, cap.~9; \\refHamilton, cap.~15--17'),
    T('\\textbf{The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class}',
      '\\textbf{Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar}')))

D.references(bib(['DF', 'FPP', 'Hamilton', 'HP', 'KPSS', 'PP', 'ZA', 'GN', 'MacKinnon', 'MacKinnonb']), per=16)

if __name__ == '__main__':
    finalize(D.write(V))
