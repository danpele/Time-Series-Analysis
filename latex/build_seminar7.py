r"""
build_seminar7.py -- Seminarul 7 (Cointegrare și VECM), EN + RO dintr-o singură sursă
====================================================================================
Seminarul are loc ÎNAINTEA cursului 7: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu o întrebare de interpretare, C o întrebare deschisă și
critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în versiunea
profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_07/sem7_results.json (seminar7.py).
Ieșire:
  EN/Seminars/seminar7_cointegration_vecm.tex          (+ _solutions.tex)
  RO/Seminarii/seminar7_cointegrare_vecm_ro.tex        (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_07/seminar7.py && python3 latex/build_seminar7.py && python3 latex/tsa_build.py compile 7
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, fig   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch7_common import REFS, T, bib, finalize, load_sem, pv, month   # noqa: E402

S = load_sem()
V = Values()
D = Deck(7, 'seminar', refs=REFS)


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


def qlsem():
    return '\\quantlet{TSA\\_ch7\\_seminar}{\\qlurl{TSA_ch7_seminar}}'


# =============================================================================
# CIFRE
# =============================================================================
A1 = S['A1']
for k in ['tau', 'eg1', 'eg5', 'eg10', 'df5', 'p']:
    V.put(f'a1.{k}', A1[k], 2 if k != 'p' else 3)
A2 = S['A2']
for k in ['tau', 'eg1', 'eg5', 'eg10', 'df5', 'p']:
    V.put(f'a2.{k}', A2[k], 2 if k != 'p' else 3)
V.put('a2.eg5n2', S['A2b']['eg5'], 2)
A3 = S['A3']
V.put('a3.dy', A3['dy'], 1)
V.put('a3.half', A3['half'], 2)
A4 = S['A4']
V.put('a4.ba', A4['ba'], 2)
V.put('a4.rho', A4['rho'], 2)
V.put('a4.half', A4['half'], 2)
A5 = S['A5']
for r in range(3):
    V.put(f'a5.t{r}', A5['trace'][r], 2)
    V.put(f'a5.m{r}', A5['maxeig'][r], 2)
    V.put(f'a5.ct{r}', A5['cvt'][r], 2)
    V.put(f'a5.cm{r}', A5['cvm'][r], 2)
V.raw('a5.r', str(A5['rank_trace']))
A6 = S['A6']
V.put('a6.r2', A6['roots'][1], 2)
V.put('a6.r3', A6['roots'][2], 2)
V.put('a6.r4', A6['roots'][3], 2)

B1 = S['B1']
V.int('b1.n', B1['n'])
V.raw('b1.first', month(B1['first']))
V.raw('b1.last', month(B1['last']))
V.put('b1.p1', B1['adf_gs1']['p'], 2)
V.put('b1.p10', B1['adf_gs10']['p'], 2)
V.raw('b1.pd', pv(B1['adf_dgs10']['p']))
V.raw('b1.psp', pv(B1['adf_spread']['p']))
E = B1['eg']
V.put('b1.b', E['beta'][0], 3)
V.put('b1.a', E['const'], 2)
V.put('b1.tau', E['tau'], 2)
V.raw('b1.k', str(E['lags']))
V.put('b1.cv', E['crit5'], 2)
V.put('b1.adfcv', E['adf_crit5'], 2)
V.put('b1.p', E['p'], 3)
V.put('b1.po', E['po_zt'], 2)
V.put('b1.pop', E['po_p'], 3)
V.put('b1.g', B1['ecm_long']['gamma'], 4)
V.put('b1.gt', B1['ecm_long']['gamma_t'], 2)
V.put('b1.h', B1['ecm_long']['half'], 1)
V.put('b1.d0', B1['ecm_long']['delta0'], 2)
V.put('b1.gs', B1['ecm_short']['gamma'], 4)
V.put('b1.gst', B1['ecm_short']['gamma_t'], 2)

B2 = S['B2']
V.int('b2.n', B2['n'])
V.put('b2.pc', B2['adf_c']['p'], 2)
V.put('b2.py', B2['adf_y']['p'], 2)
V.raw('b2.pdc', pv(B2['adf_dc']['p']))
V.raw('b2.pdy', pv(B2['adf_dy']['p']))
E2 = B2['eg']
V.put('b2.b', E2['beta'][0], 2)
V.put('b2.tau', E2['tau'], 2)
V.raw('b2.k', str(E2['lags']))
V.put('b2.cv', E2['crit5'], 2)
V.put('b2.p', E2['p'], 2)
V.put('b2.po', E2['po_zt'], 2)
V.raw('b2.pop', pv(E2['po_p']))
V.put('b2.psh', B2['adf_share']['p'], 2)
V.put('b2.s0', 100 * B2['share_first'], 0)
V.put('b2.s1', 100 * B2['share_last'], 0)
V.put('b2.g', B2['ecm']['gamma'], 3)
V.put('b2.gt', B2['ecm']['gamma_t'], 2)
V.put('b2.h', B2['ecm']['half'], 1)

B3 = S['B3']
J = B3['johansen']
for r in range(3):
    V.put(f'b3.t{r}', J['trace'][r], 2)
    V.put(f'b3.m{r}', J['maxeig'][r], 2)
    V.put(f'b3.ct{r}', J['trace_cv5'][r], 2)
    V.put(f'b3.cm{r}', J['maxeig_cv5'][r], 2)
V.raw('b3.r', str(J['rank_trace']))
V.raw('b3.k', str(J['k_ar_diff']))
V.int('b3.n', J['n'])
for c, k in [('EUR/RON', 'ron'), ('EUR/HUF', 'huf'), ('EUR/PLN', 'pln')]:
    V.put(f'b3.p.{k}', B3['adf'][c]['p'], 2)
V.put('b3.egrh', B3['eg_ron_huf']['p'], 2)
V.put('b3.egrp', B3['eg_ron_pln']['p'], 2)
V.put('b3.eghp', B3['eg_huf_pln']['p'], 2)
V.put('b3.cl', B3['corr_levels'], 2)
V.put('b3.cc', B3['corr_changes'], 2)

B4 = S['B4']
J4 = B4['johansen']
for r in range(3):
    V.put(f'b4.t{r}', J4['trace'][r], 2)
    V.put(f'b4.ct{r}', J4['trace_cv5'][r], 2)
    V.put(f'b4.m{r}', J4['maxeig'][r], 2)
V.raw('b4.r', str(B4['rank']))
V.raw('b4.k', str(J4['k_ar_diff']))
V.int('b4.n', B4['n'])
V.raw('b4.last', month(B4['last']))
V.put('b4.b1', -B4['beta'][1], 2)
V.put('b4.b2', B4['beta'][2], 3)
V.put('b4.c', B4['const'][0], 2)
for i in range(3):
    V.put(f'b4.a{i}', B4['alpha'][i], 3)
    V.put(f'b4.at{i}', B4['alpha_t'][i], 2)
V.raw('b4.pec', pv(B4['ec_adf']['p']))

C1 = S['C1']
V.put('c1.fb', C1['form']['beta'], 2)
V.put('c1.fp', C1['form']['eg']['p'], 3)
V.put('c1.fpo', C1['form']['eg']['po_p'], 3)
V.put('c1.tp', C1['trade']['eg']['p'], 2)
V.put('c1.tb', C1['trade']['eg']['beta'][0], 2)
V.raw('c1.n', str(C1['trade']['n_trades']))
V.put('c1.gm', C1['trade']['gross']['mean'], 2)
V.put('c1.gs', C1['trade']['gross']['sharpe'], 2)
V.put('c1.nm', C1['trade']['net']['mean'], 2)
V.put('c1.ns', C1['trade']['net']['sharpe'], 2)
V.put('c1.zmin', C1['trade']['z_min'], 2)
V.put('c1.out', 100 * C1['trade']['share_out'], 1)
V.put('c1.im', C1['insample']['net']['mean'], 2)
V.put('c1.is', C1['insample']['net']['sharpe'], 2)

import math   # noqa: E402
from statsmodels.tsa.adfvalues import mackinnoncrit   # noqa: E402
from statsmodels.tsa.vector_ar.vecm import c_sjt, c_sja   # noqa: E402
for k in (1, 2, 3):
    V.put(f'cv{k}', mackinnoncrit(N=k, regression='c', nobs=math.inf)[1], 2)
for j, m in enumerate((3, 2, 1)):
    V.put(f'jt{j}', c_sjt(m, 0)[1], 2)
    V.put(f'jm{j}', c_sja(m, 0)[1], 2)
V.put('hl01', math.log(0.5) / math.log(0.9), 1)
V.put('a6.ba', -0.2 - 0.1, 1)
V.put('a6.h', math.log(0.5) / math.log(0.7), 1)

C2 = S['C2']
V.put('c2.r2', C2['r2'], 2)
V.put('c2.p', C2['eg_p'], 2)
V.put('c2.df5', C2['df5'], 2)
V.put('c2.eg5', C2['eg5'], 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: are two or more $I(1)$ series tied by a long-run equilibrium, how fast do they return to it, and who does the adjusting?',
       '\\textbf{Întrebarea}: sînt două sau mai multe serii $I(1)$ legate printr-un echilibru pe termen lung, cît de repede revin la el și cine face ajustarea?'),
     [T('this seminar comes \\textbf{before} Lecture 7: the section ``What you need today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 7: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: Engle--Granger statistics by hand; an ECM and its half-life; a bivariate VECM; Johansen statistics; from a VAR to a VECM',
        'Partea A: statistici Engle--Granger calculate de mînă; un ECM și timpul lui de înjumătățire; un VECM cu două variabile; statistici Johansen; de la VAR la VECM'),
      T('Part B: US interest rates; Romanian consumption and GDP; EUR/RON, EUR/HUF and EUR/PLN; ROBOR, the Romanian 10-year yield and Euribor',
        'Partea B: ratele dobînzii din SUA; consumul și PIB-ul României; EUR/RON, EUR/HUF și EUR/PLN; ROBOR, randamentul la 10 ani al României și Euribor'),
      T('Part C: pairs trading on two Romanian banks, and an AI answer to audit', 'Partea C: pairs trading pe două bănci din România și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('an Engle--Granger statistic with two and with three variables', 'o statistică Engle--Granger cu două și cu trei variabile') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('an ECM and a bivariate VECM: speed and half-life', 'un ECM și un VECM cu două variabile: viteză și timp de înjumătățire') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('Johansen statistics from eigenvalues; a VAR(2) written as a VECM', 'statistici Johansen din valori proprii; un VAR(2) scris ca VECM') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('Engle--Granger and ECM: US yields; Romanian consumption and GDP', 'Engle--Granger și ECM: randamentele din SUA; consumul și PIB-ul României') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('Johansen and VECM: three currencies; three interest rates', 'Johansen și VECM: trei monede; trei rate ale dobînzii') + ' & ' + SP + ' & B3',
     'C1, C2 & ' + T('pairs trading out of sample; what is wrong in an AI answer?', 'pairs trading în afara eșantionului; ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, B3'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați dumneavoastră, după model')))

D.frame(T('Data used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Frequency}', '\\textbf{Frecvența}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    [T('US Treasury yields, 1 and 10 years', 'Randamentele titlurilor de stat SUA, 1 și 10 ani') + ' & FRED (GS1, GS10) & ' + T('monthly', 'lunar') + ' & 1960--2026',
     T('Household consumption and GDP, Romania', 'Consumul gospodăriilor și PIB-ul, România') + ' & Eurostat (namq\\_10\\_gdp) & ' + T('quarterly, SA', 'trimestrial, ajustat sezonier') + ' & 1995--2026',
     'EUR/RON, EUR/HUF, EUR/PLN & ' + T('BNR reference rates', 'cursurile de referință BNR') + ' & ' + T('month-end', 'sfîrșitul lunii') + ' & 2005--2026',
     T('ROBOR 3M, Euribor 3M', 'ROBOR 3M, Euribor 3M') + ' & Eurostat (irt\\_st\\_m) & ' + T('monthly', 'lunar') + ' & 2010--2026',
     T('Romanian 10-year yield', 'Randamentul la 10 ani, România') + ' & EODHD & ' + T('daily, monthly mean', 'zilnic, media lunară') + ' & 2010--2026',
     T('Banca Transilvania, BRD', 'Banca Transilvania, BRD') + ' & EODHD & ' + T('daily, adjusted close', 'zilnic, închidere ajustată') + ' & 2014--2026'],
    size='footnotesize') + items(
    T('Levels as $100\\ln$ (consumption, GDP, exchange rates, share prices); interest rates in \\% per year', 'Nivelurile ca $100\\ln$ (consum, PIB, cursuri, prețuri ale acțiunilor); ratele dobînzii în \\% pe an'),
    T('In the notebook: \\texttt{adf\\_test}, \\texttt{eg\\_test} (Engle--Granger and Phillips--Ouliaris), \\texttt{ecm\\_fit}, \\texttt{johansen}; no account or key is needed',
      'În notebook: \\texttt{adf\\_test}, \\texttt{eg\\_test} (Engle--Granger și Phillips--Ouliaris), \\texttt{ecm\\_fit}, \\texttt{johansen}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What you need today', 'Noțiuni necesare azi')

D.frame(T('What you need today (1/4): cointegration', 'Noțiuni necesare azi (1/4): cointegrarea'), items(
    (T('$y_t \\sim I(1)$: $y_t$ has a unit root and $\\Delta y_t = y_t - y_{t-1}$ is stationary (Chapter 3; ADF test, $H_0$: unit root)', '$y_t \\sim I(1)$: $y_t$ are o rădăcină unitară, iar $\\Delta y_t = y_t - y_{t-1}$ este staționar (Capitolul 3; testul ADF, $H_0$: rădăcină unitară)'),
     [T('regressing one $I(1)$ series on another, unrelated one gives a \\textbf{spurious regression}: large $t$, high $R^2$', 'regresia unei serii $I(1)$ pe o alta, fără legătură cu ea, este o \\textbf{regresie falsă}: $t$ mare, $R^2$ mare')]),
    (T('\\textbf{Cointegration}: $I(1)$ series $y_{1t}, \\dots, y_{nt}$ for which a combination $\\beta^\\top\\mathbf y_t$ is stationary', '\\textbf{Cointegrarea}: serii $I(1)$ $y_{1t}, \\dots, y_{nt}$ pentru care o combinație $\\beta^\\top\\mathbf y_t$ este staționară'),
     [T('$\\beta$: the cointegrating vector, normalised with one coefficient equal to 1; $u_t = \\beta^\\top\\mathbf y_t - \\mu$: the equilibrium error', '$\\beta$: vectorul de cointegrare, normalizat cu un coeficient egal cu 1; $u_t = \\beta^\\top\\mathbf y_t - \\mu$: eroarea de echilibru'),
      T('the series share common stochastic trends: $n$ variables, $r$ cointegrating vectors, $n - r$ common trends', 'seriile au trenduri stochastice comune: $n$ variabile, $r$ vectori de cointegrare, $n - r$ trenduri comune')]),
    T('Example: the 1-year and the 10-year interest rate wander, but their spread stays within a band', 'Exemplu: rata dobînzii la 1 an și cea la 10 ani rătăcesc, dar diferența dintre ele rămîne într-o bandă')))

D.frame(T('What you need today (2/4): the Engle--Granger test', 'Noțiuni necesare azi (2/4): testul Engle--Granger'), items(
    (T('\\textbf{Step 1}: OLS in levels, $y_t = a + b x_t + u_t$; residuals $\\hat u_t$', '\\textbf{Pasul 1}: OLS în niveluri, $y_t = a + b x_t + u_t$; reziduurile $\\hat u_t$'),
     [T('\\textbf{Step 2}: ADF on $\\hat u_t$ without constant: $\\Delta\\hat u_t = \\gamma\\hat u_{t-1} + \\sum_j\\delta_j\\Delta\\hat u_{t-j} + e_t$, $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$', '\\textbf{Pasul 2}: ADF pe $\\hat u_t$ fără constantă: $\\Delta\\hat u_t = \\gamma\\hat u_{t-1} + \\sum_j\\delta_j\\Delta\\hat u_{t-j} + e_t$, $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$'),
      T('$H_0$: no cointegration ($\\gamma = 0$); reject if $\\tau$ is below the critical value', '$H_0$: fără cointegrare ($\\gamma = 0$); respingem dacă $\\tau$ este sub valoarea critică')]),
    (T('\\textbf{Critical values} (MacKinnon, with a constant, large $T$): more negative than Dickey--Fuller, because OLS makes $\\hat u_t$ look stationary',
       '\\textbf{Valorile critice} (MacKinnon, cu constantă, $T$ mare): mai negative decît cele Dickey--Fuller, deoarece OLS face ca $\\hat u_t$ să pară staționar'),
     [T('5\\%: $@{cv1}$ (one series, Dickey--Fuller); $@{cv2}$ (2 variables); $@{cv3}$ (3 variables)', '5\\%: $@{cv1}$ (o serie, Dickey--Fuller); $@{cv2}$ (2 variabile); $@{cv3}$ (3 variabile)')]),
    (T('\\textbf{Phillips--Ouliaris} $Z_t$: the same $H_0$ and critical values; no lags, a long-run variance correction', '\\textbf{Phillips--Ouliaris} $Z_t$: aceeași $H_0$ și aceleași valori critice; fără decalaje, cu o corecție prin varianța pe termen lung'),
     [T('in Python: \\texttt{coint} (\\texttt{statsmodels}), \\texttt{phillips\\_ouliaris} (\\texttt{arch})', 'în Python: \\texttt{coint} (\\texttt{statsmodels}), \\texttt{phillips\\_ouliaris} (\\texttt{arch})')])))

D.frame(T('What you need today (3/4): error correction', 'Noțiuni necesare azi (3/4): corecția erorii'), items(
    (T('\\textbf{ECM} (error correction model): $\\Delta y_t = c + \\gamma\\,\\hat u_{t-1} + \\delta_0\\,\\Delta x_t + \\dots + \\varepsilon_t$, with $\\hat u_{t-1} = y_{t-1} - \\hat a - \\hat b x_{t-1}$',
       '\\textbf{ECM} (error correction model, modelul cu corecția erorii): $\\Delta y_t = c + \\gamma\\,\\hat u_{t-1} + \\delta_0\\,\\Delta x_t + \\dots + \\varepsilon_t$, cu $\\hat u_{t-1} = y_{t-1} - \\hat a - \\hat b x_{t-1}$'),
     [T('$\\gamma < 0$: \\textbf{speed of adjustment}, the share of last period\'s gap closed by $y$; $\\delta_0$: short-run effect; $b$: long-run effect', '$\\gamma < 0$: \\textbf{viteza de ajustare}, fracțiunea din distanța din perioada anterioară închisă de $y$; $\\delta_0$: efectul pe termen scurt; $b$: efectul pe termen lung')]),
    (T('\\textbf{Half-life}: a gap shrinks like $(1 + \\gamma)^h$; half of it is gone after $h_{1/2} = \\ln 0.5/\\ln(1 + \\gamma)$ periods', '\\textbf{Timpul de înjumătățire}: o distanță scade ca $(1 + \\gamma)^h$; jumătate din ea dispare după $h_{1/2} = \\ln 0{,}5/\\ln(1 + \\gamma)$ perioade'),
     [T('example: $\\gamma = -0.5$ gives $h_{1/2} = 1$; $\\gamma = -0.1$ gives about @{hl01} periods', 'exemplu: $\\gamma = -0{,}5$ dă $h_{1/2} = 1$; $\\gamma = -0{,}1$ dă circa @{hl01} perioade')]),
    T('All the terms of an ECM are stationary: OLS and $t$-tests are valid', 'Toți termenii unui ECM sînt staționari: OLS și testele $t$ sînt valide')))

D.frame(T('What you need today (4/4): VECM and Johansen', 'Noțiuni necesare azi (4/4): VECM și Johansen'), items(
    (T('\\textbf{VECM}: $\\Delta\\mathbf y_t = \\Pi\\mathbf y_{t-1} + \\sum_{i=1}^{p-1}\\Gamma_i\\Delta\\mathbf y_{t-i} + \\mathbf u_t$; from a VAR($p$) (Chapter 6): $\\Pi = A_1 + \\dots + A_p - I$, $\\Gamma_i = -(A_{i+1} + \\dots + A_p)$',
       '\\textbf{VECM}: $\\Delta\\mathbf y_t = \\Pi\\mathbf y_{t-1} + \\sum_{i=1}^{p-1}\\Gamma_i\\Delta\\mathbf y_{t-i} + \\mathbf u_t$; dintr-un VAR($p$) (Capitolul 6): $\\Pi = A_1 + \\dots + A_p - I$, $\\Gamma_i = -(A_{i+1} + \\dots + A_p)$'),
     [T('$r = \\mathrm{rank}(\\Pi)$: 0 = no cointegration; $0 < r < n$: $\\Pi = \\alpha\\beta^\\top$, $r$ cointegrating vectors (columns of $\\beta$) and loadings $\\alpha$', '$r = \\mathrm{rang}(\\Pi)$: 0 = fără cointegrare; $0 < r < n$: $\\Pi = \\alpha\\beta^\\top$, $r$ vectori de cointegrare (coloanele lui $\\beta$) și coeficienții de ajustare $\\alpha$'),
      T('$\\alpha_i = 0$: $y_i$ does not adjust (\\textbf{weakly exogenous})', '$\\alpha_i = 0$: $y_i$ nu se ajustează (\\textbf{slab exogenă})')]),
    (T('\\textbf{Johansen}: eigenvalues $\\hat\\lambda_1 \\ge \\dots \\ge \\hat\\lambda_n$; trace $= -T\\sum_{i>r}\\ln(1 - \\hat\\lambda_i)$, max-eigenvalue $= -T\\ln(1 - \\hat\\lambda_{r+1})$',
       '\\textbf{Johansen}: valorile proprii $\\hat\\lambda_1 \\ge \\dots \\ge \\hat\\lambda_n$; urma $= -T\\sum_{i>r}\\ln(1 - \\hat\\lambda_i)$, valoarea proprie maximă $= -T\\ln(1 - \\hat\\lambda_{r+1})$'),
     [T('test $r = 0$, $r \\le 1$, \\dots: the rank is the first $r$ not rejected; 5\\% critical values with a constant, $n = 3$: trace @{jt0}, @{jt1}, @{jt2}; max @{jm0}, @{jm1}, @{jm2}',
        'testăm $r = 0$, $r \\le 1$ etc.: rangul este primul $r$ nerespins; valori critice de 5\\% cu constantă, $n = 3$: urma @{jt0}, @{jt1}, @{jt2}; valoarea maximă @{jm0}, @{jm1}, @{jm2}')])))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: computations on paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: an Engle--Granger statistic', 'A1: o statistică Engle--Granger'),
         items(T('Two $I(1)$ series, $T = 200$. Step 1 gives residuals $\\hat u_t$; step 2 gives $\\Delta\\hat u_t = -0.118\\,\\hat u_{t-1}$, with $\\mathrm{SE} = 0.036$.', 'Două serii $I(1)$, $T = 200$. Pasul 1 dă reziduurile $\\hat u_t$; pasul 2 dă $\\Delta\\hat u_t = -0{,}118\\,\\hat u_{t-1}$, cu $\\mathrm{SE} = 0{,}036$.'),
               T('1. Compute $\\tau$.', '1. Calculați $\\tau$.'),
               T('2. Compare $\\tau$ with the 5\\% and 10\\% Engle--Granger values for 2 variables, $@{a1.eg5}$ and $@{a1.eg10}$, and decide.', '2. Comparați $\\tau$ cu valorile Engle--Granger de 5\\% și 10\\% pentru 2 variabile, $@{a1.eg5}$ și $@{a1.eg10}$, și decideți.'),
               T('3. Say what the Dickey--Fuller value $@{a1.df5}$ would have concluded.', '3. Precizați ce ar fi concluzionat valoarea Dickey--Fuller $@{a1.df5}$.'),
               T('4. Explain why the two critical values differ.', '4. Explicați de ce diferă cele două valori critice.'),
               T('Report: one number, two decisions and one sentence.', 'Raportați: o valoare, două decizii și o frază.')),
         items(T('1. $\\tau = -0.118/0.036 = @{a1.tau}$', '1. $\\tau = -0{,}118/0{,}036 = @{a1.tau}$'),
               T('2. $@{a1.tau} > @{a1.eg5}$: do not reject ``no cointegration\'\' at 5\\% (p = @{a1.p}); $@{a1.tau} < @{a1.eg10}$: reject at 10\\%', '2. $@{a1.tau} > @{a1.eg5}$: nu respingem „fără cointegrare” la 5\\% (p = @{a1.p}); $@{a1.tau} < @{a1.eg10}$: respingem la 10\\%'),
               T('3. $@{a1.tau} < @{a1.df5}$: the Dickey--Fuller table would wrongly report cointegration at 5\\%', '3. $@{a1.tau} < @{a1.df5}$: tabelul Dickey--Fuller ar raporta greșit cointegrare la 5\\%'),
               T('4. OLS chooses $\\hat b$ to minimise the residual variance, so $\\hat u_t$ looks more stationary than a true error: the null distribution shifts to the left.', '4. OLS alege $\\hat b$ astfel încît să minimizeze varianța reziduurilor, deci $\\hat u_t$ pare mai staționar decît o eroare adevărată: distribuția din ipoteza nulă se deplasează la stînga.')),
         size='scriptsize')

D.proposed(T('A2: three variables', 'A2: trei variabile'),
           items(T('Three $I(1)$ series, $T = 150$: $y_t$ regressed on $x_{1t}$ and $x_{2t}$. Step 2 gives $\\hat\\gamma = -0.162$, $\\mathrm{SE} = 0.041$. Model: A1.', 'Trei serii $I(1)$, $T = 150$: $y_t$ regresat pe $x_{1t}$ și $x_{2t}$. Pasul 2 dă $\\hat\\gamma = -0{,}162$, $\\mathrm{SE} = 0{,}041$. Model: A1.'),
                 T('1. Compute $\\tau$.', '1. Calculați $\\tau$.'),
                 T('2. Decide at 5\\% and at 1\\% with the values for 3 variables, $@{a2.eg5}$ and $@{a2.eg1}$.', '2. Decideți la 5\\% și la 1\\% cu valorile pentru 3 variabile, $@{a2.eg5}$ și $@{a2.eg1}$.'),
                 T('3. Say which critical value a student would wrongly use if he forgot $x_{2t}$ (2 variables: $@{a2.eg5n2}$).', '3. Precizați ce valoare critică ar folosi greșit un student care ar uita de $x_{2t}$ (2 variabile: $@{a2.eg5n2}$).'),
                 T('4. Say how many cointegrating vectors the Engle--Granger method can find with three variables.', '4. Precizați cîți vectori de cointegrare poate găsi metoda Engle--Granger cu trei variabile.'),
                 T('Report: one number, two decisions and two sentences.', 'Raportați: o valoare, două decizii și două fraze.')),
           items(T('1. $\\tau = -0.162/0.041 = @{a2.tau}$', '1. $\\tau = -0{,}162/0{,}041 = @{a2.tau}$'),
                 T('2. $@{a2.tau} < @{a2.eg5}$: reject at 5\\% (p = @{a2.p}); $@{a2.tau} > @{a2.eg1}$: not at 1\\%', '2. $@{a2.tau} < @{a2.eg5}$: respingem la 5\\% (p = @{a2.p}); $@{a2.tau} > @{a2.eg1}$: nu și la 1\\%'),
                 T('3. $@{a2.eg5n2}$: less negative than the right value; here the decision would be the same, but the test would be too liberal in general', '3. $@{a2.eg5n2}$: mai puțin negativă decît valoarea corectă; aici decizia ar fi aceeași, dar în general testul ar respinge prea des'),
                 T('4. One: the residuals of one regression; with three variables there may be two vectors, which only the Johansen method can find.', '4. Unul: reziduurile unei singure regresii; cu trei variabile pot exista doi vectori, pe care doar metoda Johansen îi poate găsi.')),
           size='scriptsize')

D.solved(T('A3: an error correction model', 'A3: un model cu corecția erorii'),
         items(T('Estimated ECM (log consumption $c_t$, log income $y_t$, in \\%): $\\Delta c_t = 0.2 + 0.5\\,\\Delta y_t - 0.25\\,(c_{t-1} - 0.9\\,y_{t-1})$.', 'ECM estimat (logaritmul consumului $c_t$ și al venitului $y_t$, în \\%): $\\Delta c_t = 0{,}2 + 0{,}5\\,\\Delta y_t - 0{,}25\\,(c_{t-1} - 0{,}9\\,y_{t-1})$.'),
               T('1. Give the long-run and the short-run effect of income on consumption.', '1. Precizați efectul pe termen lung și efectul pe termen scurt al venitului asupra consumului.'),
               T('2. Interpret the coefficient $-0.25$ and compute the half-life.', '2. Interpretați coeficientul $-0{,}25$ și calculați timpul de înjumătățire.'),
               T('3. Compute $\\Delta c_t$ if $c_{t-1} - 0.9\\,y_{t-1} = 2$ and $\\Delta y_t = 1$.', '3. Calculați $\\Delta c_t$ dacă $c_{t-1} - 0{,}9\\,y_{t-1} = 2$ și $\\Delta y_t = 1$.'),
               T('4. Say what a coefficient of $+0.25$ would mean.', '4. Precizați ce ar însemna un coeficient de $+0{,}25$.'),
               T('Report: two elasticities, one half-life, one forecast and one sentence.', 'Raportați: două elasticități, un timp de înjumătățire, o prognoză și o frază.')),
         items(T('1. Long run: 0.9 (from the equilibrium $c = 0.9y$ + const.); short run: 0.5 (same period)', '1. Termen lung: 0,9 (din echilibrul $c = 0{,}9y$ + const.); termen scurt: 0,5 (în aceeași perioadă)'),
               T('2. A quarter of last period\'s gap is closed each period; $h_{1/2} = \\ln 0.5/\\ln 0.75 = @{a3.half}$ periods', '2. Un sfert din distanța din perioada anterioară se închide în fiecare perioadă; $h_{1/2} = \\ln 0{,}5/\\ln 0{,}75 = @{a3.half}$ perioade'),
               T('3. $\\Delta c_t = 0.2 + 0.5 \\cdot 1 - 0.25 \\cdot 2 = @{a3.dy}$: the error correction cancels the short-run effect', '3. $\\Delta c_t = 0{,}2 + 0{,}5 \\cdot 1 - 0{,}25 \\cdot 2 = @{a3.dy}$: corecția erorii anulează efectul pe termen scurt'),
               T('4. Consumption would move away from equilibrium after a gap: no error correction, no cointegration.', '4. Consumul s-ar îndepărta de echilibru după o abatere: fără corecția erorii, fără cointegrare.')),
         size='scriptsize')

D.proposed(T('A4: a bivariate VECM', 'A4: un VECM cu două variabile'),
           items(T('$\\Delta y_{1t} = -0.20\\,z_{t-1} + u_{1t}$, $\\Delta y_{2t} = 0.05\\,z_{t-1} + u_{2t}$, with $z_t = y_{1t} - y_{2t}$. Model: A3.', '$\\Delta y_{1t} = -0{,}20\\,z_{t-1} + u_{1t}$, $\\Delta y_{2t} = 0{,}05\\,z_{t-1} + u_{2t}$, cu $z_t = y_{1t} - y_{2t}$. Model: A3.'),
                 T('1. Write $\\alpha$, $\\beta$ and $\\Pi = \\alpha\\beta^\\top$, and give the rank of $\\Pi$.', '1. Scrieți $\\alpha$, $\\beta$ și $\\Pi = \\alpha\\beta^\\top$ și precizați rangul lui $\\Pi$.'),
                 T('2. Explain which variable does most of the adjusting.', '2. Explicați care variabilă face cea mai mare parte a ajustării.'),
                 T('3. Show that $z_t = (1 + \\beta^\\top\\alpha)\\,z_{t-1} + (u_{1t} - u_{2t})$ and compute its half-life.', '3. Arătați că $z_t = (1 + \\beta^\\top\\alpha)\\,z_{t-1} + (u_{1t} - u_{2t})$ și calculați timpul de înjumătățire.'),
                 T('Report: one matrix, one rank, one half-life and one sentence.', 'Raportați: o matrice, un rang, un timp de înjumătățire și o frază.')),
           items(T('1. $\\alpha = (-0.20, 0.05)^\\top$, $\\beta = (1, -1)^\\top$; $\\Pi = \\begin{pmatrix} -0.20 & 0.20 \\\\ 0.05 & -0.05 \\end{pmatrix}$, $\\det\\Pi = 0$, rank 1', '1. $\\alpha = (-0{,}20; 0{,}05)^\\top$, $\\beta = (1, -1)^\\top$; $\\Pi = \\begin{pmatrix} -0{,}20 & 0{,}20 \\\\ 0{,}05 & -0{,}05 \\end{pmatrix}$, $\\det\\Pi = 0$, rangul 1'),
                 T('2. $y_1$: it closes 20\\% of the gap per period, $y_2$ only 5\\%; $y_2$ is close to weakly exogenous', '2. $y_1$: închide 20\\% din distanță pe perioadă, $y_2$ doar 5\\%; $y_2$ este aproape slab exogenă'),
                 T('3. $\\Delta z_t = \\Delta y_{1t} - \\Delta y_{2t} = (-0.20 - 0.05)\\,z_{t-1} + \\dots$; $\\beta^\\top\\alpha = @{a4.ba}$, $z_t = @{a4.rho}\\,z_{t-1} + \\dots$; half-life @{a4.half} periods', '3. $\\Delta z_t = \\Delta y_{1t} - \\Delta y_{2t} = (-0{,}20 - 0{,}05)\\,z_{t-1} + \\dots$; $\\beta^\\top\\alpha = @{a4.ba}$, $z_t = @{a4.rho}\\,z_{t-1} + \\dots$; timpul de înjumătățire @{a4.half} perioade')),
           size='scriptsize')

D.solved(T('A5: Johansen statistics from eigenvalues', 'A5: statistici Johansen din valori proprii'),
         items(T('Three $I(1)$ series, $T = 200$, case with a constant; the Johansen eigenvalues are 0.120, 0.045 and 0.008.', 'Trei serii $I(1)$, $T = 200$, cazul cu constantă; valorile proprii Johansen sînt 0,120; 0,045 și 0,008.'),
               T('1. Compute the trace statistics for $r = 0$, $r \\le 1$ and $r \\le 2$.', '1. Calculați statisticile urmei pentru $r = 0$, $r \\le 1$ și $r \\le 2$.'),
               T('2. Compute the maximum-eigenvalue statistics.', '2. Calculați statisticile valorii proprii maxime.'),
               T('3. Compare with the 5\\% values (trace $@{a5.ct0}$, $@{a5.ct1}$, $@{a5.ct2}$; maximum $@{a5.cm0}$, $@{a5.cm1}$, $@{a5.cm2}$) and choose the rank.', '3. Comparați cu valorile de 5\\% (urma $@{a5.ct0}$; $@{a5.ct1}$; $@{a5.ct2}$; maxima $@{a5.cm0}$; $@{a5.cm1}$; $@{a5.cm2}$) și alegeți rangul.'),
               T('4. Give the number of common trends.', '4. Precizați numărul trendurilor comune.'),
               T('Report: six statistics, one rank and one number of trends.', 'Raportați: șase statistici, un rang și un număr de trenduri.')),
         items(T('1. $-200[\\ln 0.880 + \\ln 0.955 + \\ln 0.992] = @{a5.t0}$; $-200[\\ln 0.955 + \\ln 0.992] = @{a5.t1}$; $-200\\ln 0.992 = @{a5.t2}$', '1. $-200[\\ln 0{,}880 + \\ln 0{,}955 + \\ln 0{,}992] = @{a5.t0}$; $-200[\\ln 0{,}955 + \\ln 0{,}992] = @{a5.t1}$; $-200\\ln 0{,}992 = @{a5.t2}$'),
               T('2. @{a5.m0}; @{a5.m1}; @{a5.m2}', '2. @{a5.m0}; @{a5.m1}; @{a5.m2}'),
               T('3. $r = 0$ rejected by both ($@{a5.t0} > @{a5.ct0}$, $@{a5.m0} > @{a5.cm0}$); $r \\le 1$ not rejected: rank @{a5.r}', '3. $r = 0$ respins de ambele teste ($@{a5.t0} > @{a5.ct0}$, $@{a5.m0} > @{a5.cm0}$); $r \\le 1$ nerespins: rangul @{a5.r}'),
               T('4. $n - r = 3 - 1 = 2$ common stochastic trends.', '4. $n - r = 3 - 1 = 2$ trenduri stochastice comune.')),
         size='scriptsize')

D.proposed(T('A6: from a VAR(2) to a VECM', 'A6: de la VAR(2) la VECM'),
           items(T('$\\mathbf y_t = A_1\\mathbf y_{t-1} + A_2\\mathbf y_{t-2} + \\mathbf u_t$ with $A_1 = \\begin{pmatrix} 0.7 & 0.2 \\\\ 0.1 & 0.7 \\end{pmatrix}$, $A_2 = \\begin{pmatrix} 0.1 & 0 \\\\ 0 & 0.2 \\end{pmatrix}$. Model: A5.',
                   '$\\mathbf y_t = A_1\\mathbf y_{t-1} + A_2\\mathbf y_{t-2} + \\mathbf u_t$ cu $A_1 = \\begin{pmatrix} 0{,}7 & 0{,}2 \\\\ 0{,}1 & 0{,}7 \\end{pmatrix}$, $A_2 = \\begin{pmatrix} 0{,}1 & 0 \\\\ 0 & 0{,}2 \\end{pmatrix}$. Model: A5.'),
                 T('1. Compute $\\Pi = A_1 + A_2 - I$ and $\\Gamma_1 = -A_2$.', '1. Calculați $\\Pi = A_1 + A_2 - I$ și $\\Gamma_1 = -A_2$.'),
                 T('2. Show that $\\Pi$ has rank 1 and write it as $\\alpha\\beta^\\top$ with $\\beta = (1, -1)^\\top$.', '2. Arătați că $\\Pi$ are rangul 1 și scrieți-o ca $\\alpha\\beta^\\top$, cu $\\beta = (1, -1)^\\top$.'),
                 T('3. Interpret the signs of $\\alpha$.', '3. Interpretați semnele lui $\\alpha$.'),
                 T('Report: two matrices, $\\alpha$ and one sentence.', 'Raportați: două matrice, $\\alpha$ și o frază.')),
           items(T('1. $\\Pi = \\begin{pmatrix} -0.2 & 0.2 \\\\ 0.1 & -0.1 \\end{pmatrix}$, $\\Gamma_1 = \\begin{pmatrix} -0.1 & 0 \\\\ 0 & -0.2 \\end{pmatrix}$', '1. $\\Pi = \\begin{pmatrix} -0{,}2 & 0{,}2 \\\\ 0{,}1 & -0{,}1 \\end{pmatrix}$, $\\Gamma_1 = \\begin{pmatrix} -0{,}1 & 0 \\\\ 0 & -0{,}2 \\end{pmatrix}$'),
                 T('2. $\\det\\Pi = 0.02 - 0.02 = 0$, rows proportional: rank 1; $\\alpha = (-0.2, 0.1)^\\top$; check: the VAR has one root equal to 1 (the others @{a6.r2}, @{a6.r3}, @{a6.r4})', '2. $\\det\\Pi = 0{,}02 - 0{,}02 = 0$, rînduri proporționale: rangul 1; $\\alpha = (-0{,}2; 0{,}1)^\\top$; verificare: VAR-ul are o rădăcină egală cu 1 (celelalte @{a6.r2}; @{a6.r3}; @{a6.r4})'),
                 T('3. When $y_1 > y_2$: $y_1$ falls and $y_2$ rises: both correct the gap; $\\beta^\\top\\alpha = @{a6.ba}$, half-life about @{a6.h} periods.', '3. Cînd $y_1 > y_2$: $y_1$ scade, iar $y_2$ crește: ambele corectează distanța; $\\beta^\\top\\alpha = @{a6.ba}$, timp de înjumătățire de circa @{a6.h} perioade.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: real data and interpretation', 'Partea B: date reale și interpretare')

D.task(T('B1: the 1-year and the 10-year US yields [Solved]', 'B1: randamentele din SUA la 1 an și la 10 ani [Rezolvat]'),
       T('are the 1-year and the 10-year Treasury yields cointegrated, how fast does their gap close, and which rate closes it?', 'sînt cointegrate randamentele titlurilor de stat la 1 an și la 10 ani, cît de repede se închide distanța dintre ele și care rată o închide?'),
       T('FRED GS1 and GS10, monthly averages, % per year, @{b1.first}--@{b1.last}', 'FRED GS1 și GS10, medii lunare, \\% pe an, @{b1.first}--@{b1.last}'),
       [T('Run ADF (with a constant) on both yields and on the change of the 10-year yield.', 'Aplicați ADF (cu constantă) pe ambele randamente și pe variația randamentului la 10 ani.'),
        T('Run Engle--Granger (10-year on 1-year) and Phillips--Ouliaris, and compare with the 5\\% Engle--Granger value.', 'Aplicați Engle--Granger (10 ani pe 1 an) și Phillips--Ouliaris și comparați cu valoarea Engle--Granger de 5\\%.'),
        T('Estimate the ECM of the 10-year yield (one lag of each difference) and the same equation for the 1-year yield.', 'Estimați ECM pentru randamentul la 10 ani (cîte un decalaj din fiecare diferență) și aceeași ecuație pentru randamentul la 1 an.'),
        T('Interpretation: if the 10-year yield is 1 percentage point above its equilibrium, how long until half of the gap is closed, and by which rate?', 'Interpretare: dacă randamentul la 10 ani este cu 1 punct procentual peste echilibru, cît timp trece pînă se închide jumătate din distanță și prin care rată?')],
       T('three ADF p-values, two cointegration tests, two adjustment coefficients and one sentence', 'trei valori p ADF, două teste de cointegrare, doi coeficienți de ajustare și o frază'), size='footnotesize', nb='B1')

D.frame(T('B1: solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch7_sem_b1', h='0.34') + items(
    T('ADF p = @{b1.p1} (1 year), @{b1.p10} (10 years): unit roots; the change: p @{b1.pd}; both yields $I(1)$', 'ADF p = @{b1.p1} (1 an), @{b1.p10} (10 ani): rădăcini unitare; variația: p @{b1.pd}; ambele randamente sînt $I(1)$'),
    T('Step 1: $\\hat\\imath^{(10)}_t = @{b1.a} + @{b1.b}\\,i^{(1)}_t$; step 2: $\\tau = @{b1.tau}$ (@{b1.k} lags) $< @{b1.cv}$: cointegration at 5\\% (p = @{b1.p}); Phillips--Ouliaris $Z_t = @{b1.po}$ (p = @{b1.pop})',
      'Pasul 1: $\\hat\\imath^{(10)}_t = @{b1.a} + @{b1.b}\\,i^{(1)}_t$; pasul 2: $\\tau = @{b1.tau}$ (@{b1.k} decalaje) $< @{b1.cv}$: cointegrare la 5\\% (p = @{b1.p}); Phillips--Ouliaris $Z_t = @{b1.po}$ (p = @{b1.pop})'),
    T('ECM, 10 years: $\\hat\\gamma = @{b1.g}$ ($t = @{b1.gt}$), $\\hat\\delta_0 = @{b1.d0}$; 1 year: $\\hat\\gamma = @{b1.gs}$ ($t = @{b1.gst}$), not significant', 'ECM, 10 ani: $\\hat\\gamma = @{b1.g}$ ($t = @{b1.gt}$), $\\hat\\delta_0 = @{b1.d0}$; 1 an: $\\hat\\gamma = @{b1.gs}$ ($t = @{b1.gst}$), nesemnificativ'),
    T('Interpretation: half-life $\\ln 0.5/\\ln(1 @{b1.g}) = @{b1.h}$ months, about two years; the 10-year yield closes the gap, the 1-year yield (set by monetary policy) does not react',
      'Interpretare: timpul de înjumătățire $\\ln 0{,}5/\\ln(1 @{b1.g}) = @{b1.h}$ luni, circa doi ani; randamentul la 10 ani închide distanța, iar cel la 1 an (stabilit de politica monetară) nu reacționează')) + qlsem(),
    'scriptsize')

D.task(T('B2: consumption and GDP in Romania [Proposed]', 'B2: consumul și PIB-ul în România [Propus]'),
       T('is Romanian household consumption cointegrated with GDP, as the ``great ratios\'\' suggest?', 'este consumul gospodăriilor din România cointegrat cu PIB-ul, cum sugerează „marile rapoarte”?'),
       T('Eurostat, real household consumption and real GDP, chain-linked volumes, seasonally adjusted, 1995--2026; $c_t$, $y_t$ as $100\\ln$; model: B1', 'Eurostat, consumul real al gospodăriilor și PIB-ul real, volume înlănțuite, ajustate sezonier, 1995--2026; $c_t$, $y_t$ ca $100\\ln$; model: B1'),
       [T('Run ADF (constant and trend) on $c_t$ and $y_t$, and ADF (constant) on their changes.', 'Aplicați ADF (constantă și trend) pe $c_t$ și $y_t$ și ADF (constantă) pe variațiile lor.'),
        T('Run Engle--Granger ($c_t$ on $y_t$) and Phillips--Ouliaris, and report the slope.', 'Aplicați Engle--Granger ($c_t$ pe $y_t$) și Phillips--Ouliaris și raportați panta.'),
        T('Plot the ratio of consumption to GDP and run ADF on $c_t - y_t$.', 'Reprezentați grafic raportul dintre consum și PIB și aplicați ADF pe $c_t - y_t$.'),
        T('Interpretation: does a stable consumption share make sense for Romania over 1995--2026?', 'Interpretare: are sens o pondere stabilă a consumului pentru România în perioada 1995--2026?')],
       T('four ADF tests, two cointegration tests, one chart and two sentences', 'patru teste ADF, două teste de cointegrare, un grafic și două fraze'), size='footnotesize', nb='B2')

D.frame(T('B2: solution [Proposed]', 'B2: rezolvare [Propus]'), fig('ch7_sem_b2', h='0.34') + items(
    T('ADF p = @{b2.pc} ($c_t$), @{b2.py} ($y_t$); changes: p @{b2.pdc} and @{b2.pdy}: both $I(1)$ ($T = @{b2.n}$)', 'ADF p = @{b2.pc} ($c_t$), @{b2.py} ($y_t$); variațiile: p @{b2.pdc} și @{b2.pdy}: ambele $I(1)$ ($T = @{b2.n}$)'),
    T('Slope @{b2.b}; Engle--Granger $\\tau = @{b2.tau}$ (@{b2.k} lags), p = @{b2.p}: no cointegration; Phillips--Ouliaris $Z_t = @{b2.po}$, p @{b2.pop}: cointegration; the tests disagree',
      'Panta @{b2.b}; Engle--Granger $\\tau = @{b2.tau}$ (@{b2.k} decalaje), p = @{b2.p}: fără cointegrare; Phillips--Ouliaris $Z_t = @{b2.po}$, p @{b2.pop}: cointegrare; testele nu concordă'),
    T('$c_t - y_t$: ADF p = @{b2.psh}; the share rose from about @{b2.s0}\\% to @{b2.s1}\\% (ratio of chain-linked volumes, a rough measure)', '$c_t - y_t$: ADF p = @{b2.psh}; ponderea a crescut de la circa @{b2.s0}\\% la @{b2.s1}\\% (raport de volume înlănțuite, o măsură aproximativă)'),
    T('Interpretation: no; a converging economy with growing credit and remittances raised consumption faster than GDP; a slope of @{b2.b} is not a ``great ratio\'\', and the verdict depends on the lag choice',
      'Interpretare: nu; o economie în convergență, cu credit și remitențe în creștere, a mărit consumul mai repede decît PIB-ul; o pantă de @{b2.b} nu este un „mare raport”, iar verdictul depinde de numărul de decalaje')) + qlsem(),
    'scriptsize', instructor_only=True)

D.task(T('B3: three Central European currencies [Solved]', 'B3: trei monede central-europene [Rezolvat]'),
       T('do EUR/RON, EUR/HUF and EUR/PLN share a long-run equilibrium?', 'au EUR/RON, EUR/HUF și EUR/PLN un echilibru comun pe termen lung?'),
       T('BNR reference rates (cross rates for HUF and PLN), month-end, $100\\ln$, since July 2005', 'cursurile de referință BNR (cursuri încrucișate pentru HUF și PLN), la sfîrșitul lunii, $100\\ln$, din iulie 2005'),
       [T('Run ADF on each rate.', 'Aplicați ADF pe fiecare curs.'),
        T('Run the Johansen trace and maximum-eigenvalue tests with a constant and lags by BIC.', 'Aplicați testele Johansen ale urmei și ale valorii proprii maxime, cu constantă și decalaje după BIC.'),
        T('Compare with the three pairwise Engle--Granger tests and with the correlations of levels and of changes.', 'Comparați cu cele trei teste Engle--Granger pe perechi și cu corelațiile nivelurilor și ale variațiilor.'),
        T('Interpretation: should the three rates be modelled with a VECM?', 'Interpretare: ar trebui modelate cele trei cursuri cu un VECM?')],
       T('three ADF p-values, a table of six Johansen statistics, a rank and one sentence', 'trei valori p ADF, un tabel cu șase statistici Johansen, un rang și o frază'), size='footnotesize', nb='B3')

D.frame(T('B3: solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch7_sem_b3', h='0.32') + table(
    'lccc', T('& $r = 0$ & $r \\le 1$ & $r \\le 2$', '& $r = 0$ & $r \\le 1$ & $r \\le 2$'),
    [T('trace (5\\% value)', 'urma (valoarea de 5\\%)') + ' & @{b3.t0} (@{b3.ct0}) & @{b3.t1} (@{b3.ct1}) & @{b3.t2} (@{b3.ct2})',
     T('maximum eigenvalue (5\\% value)', 'valoarea proprie maximă (valoarea de 5\\%)') + ' & @{b3.m0} (@{b3.cm0}) & @{b3.m1} (@{b3.cm1}) & @{b3.m2} (@{b3.cm2})'],
    size='scriptsize') + items(
    T('ADF p = @{b3.p.ron}; @{b3.p.huf}; @{b3.p.pln}: three $I(1)$ rates; Johansen ($T = @{b3.n}$, $k = @{b3.k}$): no statistic exceeds its critical value: rank @{b3.r}', 'ADF p = @{b3.p.ron}; @{b3.p.huf}; @{b3.p.pln}: trei cursuri $I(1)$; Johansen ($T = @{b3.n}$, $k = @{b3.k}$): nicio statistică nu depășește valoarea critică: rangul @{b3.r}'),
    T('Engle--Granger p = @{b3.egrh} (RON, HUF), @{b3.egrp} (RON, PLN), @{b3.eghp} (HUF, PLN); correlation @{b3.cl} in levels but only @{b3.cc} in monthly changes', 'Engle--Granger p = @{b3.egrh} (RON, HUF), @{b3.egrp} (RON, PLN), @{b3.eghp} (HUF, PLN); corelația este @{b3.cl} în niveluri, dar doar @{b3.cc} în variațiile lunare'),
    T('Interpretation: no; with rank 0 there is no $\\beta$ to estimate; model the changes with a VAR in differences (Chapter 6)', 'Interpretare: nu; cu rangul 0 nu există niciun $\\beta$ de estimat; modelăm variațiile cu un VAR în diferențe (Capitolul 6)')) + qlsem(),
    'scriptsize')

D.task(T('B4: ROBOR, the Romanian 10-year yield and Euribor [Proposed]', 'B4: ROBOR, randamentul la 10 ani al României și Euribor [Propus]'),
       T('how many long-run relations tie Romanian interest rates to each other and to Euribor, and which rate adjusts?', 'cîte relații pe termen lung leagă ratele dobînzii din România între ele și de Euribor și care rată se ajustează?'),
       T('ROBOR 3M and Euribor 3M (Eurostat), Romanian 10-year yield (monthly mean of daily data), January 2010--@{b4.last}; model: B3', 'ROBOR 3M și Euribor 3M (Eurostat), randamentul la 10 ani al României (media lunară a datelor zilnice), ianuarie 2010--@{b4.last}; model: B3'),
       [T('Run the Johansen tests (constant, BIC lags) and choose the rank.', 'Aplicați testele Johansen (constantă, decalaje după BIC) și alegeți rangul.'),
        T('Estimate the VECM with that rank and the constant restricted to the cointegrating relation; normalise $\\beta$ on ROBOR.', 'Estimați VECM cu acest rang și cu constanta restricționată la relația de cointegrare; normalizați $\\beta$ pe ROBOR.'),
        T('Report $\\hat\\alpha$ with $t$-statistics and run ADF on the equilibrium error.', 'Raportați $\\hat\\alpha$ cu statisticile $t$ și aplicați ADF pe eroarea de echilibru.'),
        T('Interpretation: which rates are weakly exogenous, and what does this say about Romanian monetary conditions?', 'Interpretare: care rate sînt slab exogene și ce spune acest lucru despre condițiile monetare din România?')],
       T('a table of Johansen statistics, $\\hat\\beta$, $\\hat\\alpha$ with $t$, one ADF test and two sentences', 'un tabel cu statisticile Johansen, $\\hat\\beta$, $\\hat\\alpha$ cu $t$, un test ADF și două fraze'), size='footnotesize', nb='B4')

D.frame(T('B4: solution [Proposed]', 'B4: rezolvare [Propus]'), fig('ch7_sem_b4', h='0.32') + items(
    T('$T = @{b4.n}$, $k = @{b4.k}$: trace @{b4.t0} $> @{b4.ct0}$; @{b4.t1} $< @{b4.ct1}$: rank @{b4.r} (the maximum-eigenvalue test agrees)', '$T = @{b4.n}$, $k = @{b4.k}$: urma @{b4.t0} $> @{b4.ct0}$; @{b4.t1} $< @{b4.ct1}$: rangul @{b4.r} (testul valorii proprii maxime este de acord)'),
    T('$\\hat\\beta^\\top\\mathbf y_t = \\mathrm{ROBOR}_t - @{b4.b1}\\,i^{(10)}_t + @{b4.b2}\\,\\mathrm{Euribor}_t + @{b4.c}$: ROBOR tied to the Romanian long rate; Euribor almost absent; ADF on the error: p @{b4.pec}',
      '$\\hat\\beta^\\top\\mathbf y_t = \\mathrm{ROBOR}_t - @{b4.b1}\\,i^{(10)}_t + @{b4.b2}\\,\\mathrm{Euribor}_t + @{b4.c}$: ROBOR legat de rata pe termen lung din România; Euribor aproape absent; ADF pe eroare: p @{b4.pec}'),
    T('$\\hat\\alpha$: ROBOR $@{b4.a0}$ ($t = @{b4.at0}$); 10-year yield $@{b4.a1}$ ($t = @{b4.at1}$); Euribor $@{b4.a2}$ ($t = @{b4.at2}$)', '$\\hat\\alpha$: ROBOR $@{b4.a0}$ ($t = @{b4.at0}$); randamentul la 10 ani $@{b4.a1}$ ($t = @{b4.at1}$); Euribor $@{b4.a2}$ ($t = @{b4.at2}$)'),
    T('Interpretation: the 10-year yield and Euribor are weakly exogenous; ROBOR does the adjusting; the long rate reflects inflation and risk premia; Euribor is not the anchor of Romanian money market rates',
      'Interpretare: randamentul la 10 ani și Euribor sînt slab exogene; ROBOR face ajustarea; rata pe termen lung reflectă inflația și primele de risc; Euribor nu este ancora ratelor de pe piața monetară din România')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: open questions and AI critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: pairs trading on two banks [Proposed]', 'C1: pairs trading pe două bănci [Propus]'),
       T('would a trader who found Banca Transilvania and BRD cointegrated at the end of 2021 have earned money afterwards?', 'ar fi cîștigat bani un investitor care ar fi găsit Banca Transilvania și BRD cointegrate la sfîrșitul lui 2021?'),
       T('adjusted daily prices, formation 2014--2021, trading January 2022--September 2026; models: B1, B3', 'prețuri zilnice ajustate, formare 2014--2021, tranzacționare ianuarie 2022--septembrie 2026; modele: B1, B3'),
       [T('On the formation period, run Engle--Granger and Phillips--Ouliaris (TLV on BRD) and keep $\\hat b$, the mean and the standard deviation of the spread.', 'Pe perioada de formare, aplicați Engle--Granger și Phillips--Ouliaris (TLV pe BRD) și păstrați $\\hat b$, media și abaterea standard a spread-ului.'),
        T('On the trading period, apply the rule: open at $|z| > 2$, close at $z = 0$; compute the annual return gross and net of 0.20\\% per unit traded.', 'Pe perioada de tranzacționare, aplicați regula: deschidere la $|z| > 2$, închidere la $z = 0$; calculați randamentul anual brut și net (după un cost de 0,20\\% pe unitatea tranzacționată).'),
        T('Repeat the Engle--Granger test on the trading period alone.', 'Repetați testul Engle--Granger doar pe perioada de tranzacționare.'),
        T('Interpretation: is the result evidence that pairs trading works on the BVB?', 'Interpretare: este rezultatul o dovadă că pairs trading funcționează la BVB?')],
       T('two tests, the number of trades, two returns and a plan for a project', 'două teste, numărul de tranzacții, două randamente și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: reference analysis [Proposed]', 'C1: analiză de referință [Propus]'), fig('ch7_sem_c1', h='0.32') + items(
    T('Formation: $\\hat b = @{c1.fb}$, Engle--Granger p = @{c1.fp}, Phillips--Ouliaris p = @{c1.fpo}: cointegrated at 5\\% by one test only', 'Formarea: $\\hat b = @{c1.fb}$, Engle--Granger p = @{c1.fp}, Phillips--Ouliaris p = @{c1.fpo}: cointegrate la 5\\% doar după un test'),
    T('Trading: @{c1.n} trades; gross @{c1.gm}\\% per year (Sharpe @{c1.gs}), net @{c1.nm}\\% (Sharpe @{c1.ns}); the spread fell to $z = @{c1.zmin}$ before it came back', 'Tranzacționarea: @{c1.n} tranzacții; brut @{c1.gm}\\% pe an (Sharpe @{c1.gs}), net @{c1.nm}\\% (Sharpe @{c1.ns}); spread-ul a coborît pînă la $z = @{c1.zmin}$ înainte să revină'),
    T('On 2022--2026 alone: Engle--Granger p = @{c1.tp}, slope @{c1.tb}: the relation weakened; in-sample, full-period parameters: @{c1.im}\\% per year', 'Doar pe 2022--2026: Engle--Granger p = @{c1.tp}, panta @{c1.tb}: relația s-a slăbit; în eșantion, cu parametrii întregii perioade: @{c1.im}\\% pe an'),
    T('Interpretation: no; @{c1.n} trades are an anecdote, one pair was chosen after the fact, and a spread at $z = @{c1.zmin}$ could also have kept falling; Lecture 7 tests 28 pairs with rolling windows',
      'Interpretare: nu; @{c1.n} tranzacții sînt o anecdotă, perechea a fost aleasă a posteriori, iar un spread la $z = @{c1.zmin}$ ar fi putut continua să scadă; Cursul 7 testează 28 de perechi cu ferestre mobile'),
    T('Project: pairs trading on the BVB with rolling selection, several thresholds, realistic costs and limits on short selling', 'Proiect: pairs trading la BVB cu selecție pe ferestre mobile, mai multe praguri, costuri realiste și limite ale vînzării în lipsă')) + qlsem(),
    'scriptsize', instructor_only=True)

D.frame(T('C2: audit an AI answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant about cointegration. The answer:', 'Un student a întrebat un asistent AI despre cointegrare. Răspunsul:'),
    T('\\aiprompt{(a) The Engle-Granger statistic is -3.0 with two variables; since -3.0 < @{c2.df5}, the 5\\% Dickey-Fuller value, the series are cointegrated.}', '\\aiprompt{(a) Statistica Engle-Granger este -3,0 cu două variabile; deoarece -3,0 < @{c2.df5}, valoarea Dickey-Fuller de 5\\%, seriile sînt cointegrate.}'),
    T('\\aiprompt{(b) The ECM coefficient of the lagged equilibrium error is +0.15, so 15\\% of the gap is corrected each period.}', '\\aiprompt{(b) Coeficientul erorii de echilibru decalate din ECM este +0,15, deci 15\\% din distanță se corectează în fiecare perioadă.}'),
    T('\\aiprompt{(c) Johansen rejects r = 0 and r <= 1 but not r <= 2 for three yields, so the yields share two common trends.}', '\\aiprompt{(c) Johansen respinge r = 0 și r <= 1, dar nu și r <= 2 pentru trei randamente, deci randamentele au două trenduri comune.}'),
    T('\\aiprompt{(d) EUR/RON and EUR/HUF are both I(1) and R2 = @{c2.r2} in levels, so they are cointegrated.}', '\\aiprompt{(d) EUR/RON și EUR/HUF sînt ambele I(1), iar R2 = @{c2.r2} în niveluri, deci sînt cointegrate.}'),
    T('\\aiprompt{(e) If two series are cointegrated, at least one of them Granger-causes the other.}', '\\aiprompt{(e) Dacă două serii sînt cointegrate, cel puțin una dintre ele o cauzează Granger pe cealaltă.}'),
    T('\\aiprompt{(f) A pairs-trading rule with a Sharpe ratio of 0.44 on 2014-2026 data will keep this Sharpe ratio in the future.}', '\\aiprompt{(f) O regulă de pairs trading cu raportul Sharpe 0,44 pe datele din 2014-2026 își va păstra acest raport Sharpe și în viitor.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: residual-based tests need the Engle--Granger value, about $@{c2.eg5}$; $-3.0 > @{c2.eg5}$: no cointegration at 5\\%', '(a) Greșit: testele pe reziduuri cer valoarea Engle--Granger, circa $@{c2.eg5}$; $-3{,}0 > @{c2.eg5}$: fără cointegrare la 5\\%'),
    T('(b) Wrong: a positive coefficient pushes $y$ further from equilibrium; error correction needs a negative coefficient', '(b) Greșit: un coeficient pozitiv îndepărtează $y$ și mai mult de echilibru; corecția erorii cere un coeficient negativ'),
    T('(c) Wrong: rank 2 means two cointegrating vectors and $3 - 2 = 1$ common trend', '(c) Greșit: rangul 2 înseamnă doi vectori de cointegrare și $3 - 2 = 1$ trend comun'),
    T('(d) Wrong: a high $R^2$ between $I(1)$ series proves nothing; Engle--Granger p = @{c2.p}', '(d) Greșit: un $R^2$ mare între serii $I(1)$ nu dovedește nimic; Engle--Granger p = @{c2.p}'),
    T('(e) Correct: by the Granger representation theorem, at least one adjustment coefficient is non-zero, so the lagged levels help predict at least one variable', '(e) Corect: conform teoremei de reprezentare a lui Granger, cel puțin un coeficient de ajustare este nenul, deci nivelurile decalate ajută la prognoza cel puțin a unei variabile'),
    T('(f) Wrong: the 0.44 uses full-sample parameters and a pair chosen after the fact; out of sample the rolling rule earns close to zero after costs (Lecture 7)', '(f) Greșit: valoarea 0,44 folosește parametrii întregului eșantion și o pereche aleasă a posteriori; în afara eșantionului, regula pe ferestre mobile cîștigă aproape zero după costuri (Cursul 7)')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-up', 'Încheiere')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Test that every series is $I(1)$ first; then test the equilibrium error, never the correlation of levels', 'Testați întîi că fiecare serie este $I(1)$; apoi testați eroarea de echilibru, niciodată corelația nivelurilor'),
    T('Residual-based tests need Engle--Granger critical values, which depend on the number of variables', 'Testele pe reziduuri cer valorile critice Engle--Granger, care depind de numărul de variabile'),
    T('The ECM gives the speed of adjustment; the half-life $\\ln 0.5/\\ln(1 + \\gamma)$ is in the units of the data', 'ECM dă viteza de ajustare; timpul de înjumătățire $\\ln 0{,}5/\\ln(1 + \\gamma)$ este în unitățile datelor'),
    T('Johansen gives the rank; $n - r$ is the number of common trends; $\\alpha$ shows who adjusts', 'Johansen dă rangul; $n - r$ este numărul trendurilor comune; $\\alpha$ arată cine se ajustează'),
    T('An AI answer is a draft: check the critical values, the signs and the number of trends', 'Un răspuns AI este o ciornă: verificați valorile critice, semnele și numărul trendurilor')))

D.frame(T('After the seminar', 'După seminar'), items(
    T('Lecture 7 develops each topic of today: common trends, the Engle--Granger distribution, ECM, the VECM and the Granger representation theorem, the Johansen tests, forecasting and pairs trading',
      'Cursul 7 dezvoltă fiecare temă de azi: trendurile comune, distribuția Engle--Granger, ECM, modelul VECM și teorema de reprezentare a lui Granger, testele Johansen, prognoza și pairs trading'),
    T('Try the [Proposed] tasks in the notebook', 'Încercați cerințele [Propus] în notebook'),
    T('C1 can grow into a team project: pairs trading on the BVB, out of sample and after costs', 'C1 poate deveni un proiect de echipă: pairs trading la BVB, în afara eșantionului și după costuri'),
    T('Reading: \\refHP, Ch.~9; \\refHamilton, Ch.~19--20; \\refEG', 'Lectură: \\refHP, cap.~9; \\refHamilton, cap.~19--20; \\refEG'),
    T('\\textbf{The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class}',
      '\\textbf{Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar}')))

D.references(bib(['EG', 'GGR', 'Hamilton', 'HP', 'Joh88', 'Joh91', 'MacKinnonb', 'PO']), per=16)

if __name__ == '__main__':
    finalize(D.write(V))
