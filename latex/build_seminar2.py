r"""
build_seminar2.py -- Seminarul 2 (Modele ARMA), EN + RO dintr-o singură sursă
==============================================================================
Seminarul are loc ÎNAINTEA cursului 2: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu o întrebare de interpretare, C o întrebare deschisă și
critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în versiunea
profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_02/sem2_results.json (seminar2.py).
Ieșire:
  EN/Seminars/seminar2_arma_models.tex          (+ _solutions.tex)
  RO/Seminarii/seminar2_modele_arma_ro.tex      (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_02/seminar2.py && python3 latex/build_seminar2.py && python3 latex/tsa_build.py compile 2
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, items, table, fig   # noqa: E402
from ch2_common import REFS, T, bib, date, finalize, load_sem, pv, quarter   # noqa: E402

S = load_sem()
V = Values()
D = Deck(2, 'seminar', refs=REFS)


def qlsem():
    return '\\quantlet{TSA\\_ch2\\_seminar}{\\qlurl{TSA_ch2_seminar}}'


# =============================================================================
# CIFRE
# =============================================================================
A1 = S['A1']
V.put('a1.mu', A1['mu'], 1)
V.put('a1.g0', A1['gamma0'], 2)
V.put('a1.r2', A1['rho'][1], 2)
V.put('a1.r3', A1['rho'][2], 3)
V.put('a1.f1', A1['f'][0], 2)
V.put('a1.f2', A1['f'][1], 2)
V.put('a1.m2', A1['mse'][1], 4)
V.put('a1.lo1', A1['lo'][0], 3)
V.put('a1.hi1', A1['hi'][0], 3)
V.put('a1.lo2', A1['lo'][1], 2)
V.put('a1.hi2', A1['hi'][1], 2)
V.put('a1.sd2', math.sqrt(A1['mse'][1]), 3)
V.put('a1.hl', A1['half_life'], 2)
A2 = S['A2']
V.put('a2.z1', A2['roots'][0], 0)
V.put('a2.z2', A2['roots'][1], 0)
V.put('a2.r1', A2['rho'][0], 3)
V.put('a2.r2', A2['rho'][1], 3)
V.put('a2.r3', A2['rho'][2], 3)
V.put('a2.c1', A2['cond'][0], 1)
V.put('a2.c2', A2['cond'][1], 1)
V.put('a2.p2', A2['psi'][2], 2)
V.put('a2.p3', A2['psi'][3], 3)
A3 = S['A3']
V.put('a3.g0', A3['g0'], 0)
V.put('a3.g1', A3['g1'], 0)
V.put('a3.r1', A3['rho1'], 1)
V.put('a3.s2', A3['sigma2_inv'], 0)
A4 = S['A4']
V.put('a4.p1', A4['psi'][1], 1)
V.put('a4.p2', A4['psi'][2], 1)
V.put('a4.p3', A4['psi'][3], 1)
V.put('a4.g0', A4['g0'], 4)
V.put('a4.r1', A4['rho1'], 4)
V.put('a4.r2', A4['rho2'], 4)
A5 = S['A5']
V.put('a5.det', A5['det'], 4)
V.put('a5.f1', A5['phi1'], 1)
V.put('a5.f2', A5['phi2'], 1)
V.put('a5.s2', A5['sigma2'], 2)
V.put('a5.c1', A5['cond'][0], 1)
V.put('a5.c2', A5['cond'][1], 1)
AI = S['A5ic']
for m, k in [('AR(1)', 'ar1'), ('AR(2)', 'ar2')]:
    V.put(f'a5.{k}.aic', AI[m]['aic'], 1)
    V.put(f'a5.{k}.bic', AI[m]['bic'], 2)
V.put('ln100', math.log(100), 3)
A6 = S['A6']
for m, k in [('AR(1)', 'ar1'), ('AR(2)', 'ar2'), ('ARMA(1,1)', 'arma11'), ('ARMA(2,1)', 'arma21')]:
    V.put(f'a6.{k}.aic', A6[m]['aic'], 1)
    V.put(f'a6.{k}.bic', A6[m]['bic'], 2)
V.put('ln120', math.log(120), 3)
V.put('a6.c8', A6['crit8'], 2)
V.put('a6.c10', A6['crit10'], 2)
V.put('a6.p8', A6['p8'], 3)
V.put('a6.p10', A6['p10'], 3)


def put_bj(key, d):
    V.int(f'{key}.T', d['T'])
    V.put(f'{key}.mean', d['mean'], 2)
    V.put(f'{key}.sd', d['sd'], 2)
    for i in range(3):
        V.put(f'{key}.r{i + 1}', d['r'][i], 2)
        V.put(f'{key}.p{i + 1}', d['p'][i], 2)
    V.put(f'{key}.band', d['band'], 2)
    for m, v in d['models'].items():
        k = m.replace('ARMA(', '').replace(',', '').replace(')', '')
        V.put(f'{key}.{k}.aic', v['aic'], 2)
        V.put(f'{key}.{k}.bic', v['bic'], 2)
    V.put(f'{key}.q', d['lb']['lb'], 2)
    V.raw(f'{key}.qdf', str(d['lb']['df']))
    V.raw(f'{key}.qp', pv(d['lb']['lb_p']))
    V.raw(f'{key}.qpw', pv(d['lb_wrong']['lb_p']))
    V.put(f'{key}.jb', d['jb'], 1)
    V.raw(f'{key}.jbp', pv(d['jb_p']))
    V.put(f'{key}.q0', d['lb_raw']['lb'], 1)
    V.raw(f'{key}.q0p', pv(d['lb_raw']['lb_p']))


B1, B2 = S['B1'], S['B2']
put_bj('b1', B1)
put_bj('b2', B2)
assert B1['bic_best'] == 'ARMA(1,0)' and B1['aic_best'] == 'ARMA(2,0)'
assert B2['bic_best'] == 'ARMA(0,0)' and B2['aic_best'] == 'ARMA(0,0)'
m10 = B1['models']['ARMA(1,0)']
V.put('b1.mu', m10['params'][0], 2)
V.put('b1.phi', m10['params'][1], 3)
V.put('b1.phis', m10['se'][1], 3)
V.put('b1.sig', math.sqrt(m10['params'][2]), 2)
V.put('b1.hl', math.log(0.5) / math.log(m10['params'][1]), 2)
m20 = B1['models']['ARMA(2,0)']
V.put('b1.a2.p1', m20['params'][1], 3)
V.put('b1.a2.p2', m20['params'][2], 3)
V.put('b1.a2.s2', m20['se'][2], 3)
V.raw('b1.q0d', quarter(B1['first']))
V.raw('b1.q1d', quarter(B1['last']))
V.raw('b2.q0d', quarter(B2['first']))
V.raw('b2.q1d', quarter(B2['last']))
m0 = B2['models']['ARMA(0,0)']
V.put('b2.mu', m0['params'][0], 2)
V.put('b2.mus', m0['se'][0], 2)
V.put('b2.min', B2['min'], 1)
V.raw('b2.mind', quarter(B2['min_d']))
X = B2['ex2020']
V.put('b2x.r2', X['r'][1], 2)
V.raw('b2x.q0p', pv(X['lb_raw']['lb_p']))
V.put('b2x.sd', X['sd'], 2)


def put_ret(key, d):
    V.int(f'{key}.T', d['T'])
    V.raw(f'{key}.last', date(d['last']))
    V.put(f'{key}.lastr', d['last_r'], 2)
    V.put(f'{key}.c', d['c'], 3)
    V.put(f'{key}.phi', d['phi'], 3)
    V.put(f'{key}.se', d['se'], 4)
    V.put(f'{key}.t', d['t'], 1)
    V.put(f'{key}.sig', d['sigma'], 3)
    V.put(f'{key}.r2', 100 * d['r2'], 1)
    V.put(f'{key}.q', d['lb10']['lb'], 1)
    V.raw(f'{key}.qp', pv(d['lb10']['lb_p']))
    V.put(f'{key}.q2', d['lbsq10']['lb'], 0)
    V.raw(f'{key}.q2p', pv(d['lbsq10']['lb_p']))
    V.put(f'{key}.jb', d['jb'], 0)
    V.put(f'{key}.kurt', d['kurt'], 1)
    V.put(f'{key}.f1', d['f1'], 3)
    V.put(f'{key}.lo', d['lo'], 2)
    V.put(f'{key}.hi', d['hi'], 2)
    V.put(f'{key}.mean', d['mean'], 3)
    V.put(f'{key}.band', 1.96 / math.sqrt(d['T']), 3)


put_ret('b3', S['B3'])
put_ret('b4', S['B4'])
C1 = S['C1']
V.raw('c1.n', str(C1['n']))
V.raw('c1.d0', date(C1['first'], day=False))
V.raw('c1.d1', date(C1['last'], day=False))
for k in ['ar', 'mean', 'naive']:
    V.put(f'c1.rm.{k}', C1['rmse'][k], 2)
    V.put(f'c1.ma.{k}', C1['mae'][k], 2)
    V.put(f'c1.pre.{k}', C1['rmse_pre'][k], 2)
V.raw('c1.npre', str(C1['n_pre']))
C2 = S['C2']
V.put('c2.phi', C2['phi'], 2)
V.raw('c2.pw', pv(C2['lb_wrong']['lb_p']))
V.raw('c2.pr', pv(C2['lb_right']['lb_p']))
V.put('c2.q', C2['lb_wrong']['lb'], 2)
V.put('c2.bphi', C2['bet_phi'], 3)
V.put('c2.br2', 100 * C2['bet_r2'], 1)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how do we choose, estimate, check and use an ARMA model for a stationary series?',
       '\\textbf{Întrebarea}: cum alegem, estimăm, verificăm și folosim un model ARMA pentru o serie staționară?'),
     [T('this seminar comes \\textbf{before} Lecture 2: the section ``What you need today\'\' gives every formula the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 2: secțiunea „Noțiuni necesare azi” dă toate formulele folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: AR(1), AR(2), MA(1) and ARMA(1,1) on paper; Yule--Walker; AIC and BIC from given log-likelihoods',
        'Partea A: AR(1), AR(2), MA(1) și ARMA(1,1) pe hîrtie; Yule--Walker; AIC și BIC din log-verosimilități date'),
      T('Part B: the Box--Jenkins steps on US and Romanian GDP growth, and on BET and EUR/RON returns',
        'Partea B: pașii Box--Jenkins pentru creșterea PIB-ului SUA și al României și pentru randamentele BET și EUR/RON'),
      T('Part C: an out-of-sample test for Romanian inflation and an AI answer to audit', 'Partea C: un test în afara eșantionului pentru inflația din România și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('AR(1): moments and forecasts; AR(2): roots and autocorrelations', 'AR(1): momente și prognoze; AR(2): rădăcini și autocorelații') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('MA(1) invertibility; ARMA(1,1) weights and a common factor', 'invertibilitatea MA(1); ponderile ARMA(1,1) și un factor comun') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('Yule--Walker and information criteria; criteria and Ljung--Box degrees of freedom', 'Yule--Walker și criterii informaționale; criterii și gradele de libertate Ljung--Box') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('Box--Jenkins: US real GDP growth; Romanian real GDP growth', 'Box--Jenkins: creșterea PIB-ului real al SUA; creșterea PIB-ului real al României') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('AR(1) for daily returns: BET; EUR/RON', 'AR(1) pentru randamente zilnice: BET; EUR/RON') + ' & ' + SP + ' & B3',
     'C1, C2 & ' + T('does an AR beat simple forecasts of inflation? what is wrong in an AI answer?', 'bate un AR prognozele simple ale inflației? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, A1'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați dumneavoastră, după model')))

D.frame(T('Data used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Frequency}', '\\textbf{Frecvența}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    [T('US real GDP', 'PIB real, SUA') + ' & ' + T('statsmodels \\texttt{macrodata}', 'statsmodels \\texttt{macrodata}') + ' & ' + T('quarterly', 'trimestrial') + ' & 1959--2009',
     T('Real GDP, Romania', 'PIB real, România') + ' & Eurostat (namq\\_10\\_gdp) & ' + T('quarterly, seasonally adjusted', 'trimestrial, ajustat sezonier') + ' & 2000--2026',
     'BET & EODHD & ' + T('daily, close', 'zilnic, închidere') + ' & 2000--2026',
     'EUR/RON & ' + T('BNR reference rate', 'cursul de referință BNR') + ' & ' + T('daily', 'zilnic') + ' & 2005--2026',
     T('HICP, Romania', 'IAPC, România') + ' & Eurostat (prc\\_hicp\\_minr) & ' + T('monthly, 2015 = 100', 'lunar, 2015 = 100') + ' & 2005--2026'],
    size='footnotesize') + items(
    T('Growth rates in \\%: $400\\,\\Delta\\ln Y_t$ (US, annualised), $100\\,\\Delta\\ln Y_t$ (Romania, quarter on quarter); daily log returns $100\\,\\Delta\\ln P_t$',
      'Ratele de creștere în \\%: $400\\,\\Delta\\ln Y_t$ (SUA, anualizat), $100\\,\\Delta\\ln Y_t$ (România, față de trimestrul anterior); randamente logaritmice zilnice $100\\,\\Delta\\ln P_t$'),
    T('12-month inflation: $100\\,(\\ln P_t - \\ln P_{t-12})$', 'Inflația pe 12 luni: $100\\,(\\ln P_t - \\ln P_{t-12})$'),
    T('In the notebook: \\texttt{fit\\_arma(x, p, q)}, \\texttt{box\\_jenkins(y, name)}, \\texttt{b3\\_returns(\'bet\')}; no account or key is needed',
      'În notebook: \\texttt{fit\\_arma(x, p, q)}, \\texttt{box\\_jenkins(y, name)}, \\texttt{b3\\_returns(\'bet\')}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What you need today', 'Noțiuni necesare azi')

D.frame(T('What you need today (1/4): the models', 'Noțiuni necesare azi (1/4): modelele'), items(
    (T('$\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$ white noise; lag operator $LX_t = X_{t-1}$ (Seminar 1)', '$\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$ zgomot alb; operatorul lag $LX_t = X_{t-1}$ (Seminarul 1)'),
     [T('\\textbf{AR($p$)}: $X_t = c + \\phi_1X_{t-1} + \\dots + \\phi_pX_{t-p} + \\varepsilon_t$, i.e. $\\phi(L)X_t = c + \\varepsilon_t$, $\\phi(z) = 1 - \\phi_1z - \\dots - \\phi_pz^p$',
        '\\textbf{AR($p$)}: $X_t = c + \\phi_1X_{t-1} + \\dots + \\phi_pX_{t-p} + \\varepsilon_t$, adică $\\phi(L)X_t = c + \\varepsilon_t$, $\\phi(z) = 1 - \\phi_1z - \\dots - \\phi_pz^p$'),
      T('\\textbf{MA($q$)}: $X_t = \\mu + \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\dots + \\theta_q\\varepsilon_{t-q} = \\mu + \\theta(L)\\varepsilon_t$',
        '\\textbf{MA($q$)}: $X_t = \\mu + \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\dots + \\theta_q\\varepsilon_{t-q} = \\mu + \\theta(L)\\varepsilon_t$'),
      T('\\textbf{ARMA($p,q$)}: $\\phi(L)(X_t - \\mu) = \\theta(L)\\varepsilon_t$', '\\textbf{ARMA($p,q$)}: $\\phi(L)(X_t - \\mu) = \\theta(L)\\varepsilon_t$')]),
    (T('\\textbf{Stationary (causal)}: all roots of $\\phi(z) = 0$ satisfy $|z| > 1$; then $X_t = \\mu + \\sum_{j \\ge 0}\\psi_j\\varepsilon_{t-j}$', '\\textbf{Staționar (cauzal)}: toate rădăcinile lui $\\phi(z) = 0$ au $|z| > 1$; atunci $X_t = \\mu + \\sum_{j \\ge 0}\\psi_j\\varepsilon_{t-j}$'),
     [T('AR(1): $|\\phi| < 1$; AR(2): $\\phi_1 + \\phi_2 < 1$, $\\phi_2 - \\phi_1 < 1$, $|\\phi_2| < 1$', 'AR(1): $|\\phi| < 1$; AR(2): $\\phi_1 + \\phi_2 < 1$, $\\phi_2 - \\phi_1 < 1$, $|\\phi_2| < 1$')]),
    (T('\\textbf{Invertible}: all roots of $\\theta(z) = 0$ satisfy $|z| > 1$; MA(1): $|\\theta| < 1$', '\\textbf{Invertibil}: toate rădăcinile lui $\\theta(z) = 0$ au $|z| > 1$; MA(1): $|\\theta| < 1$'),
     [T('then $\\varepsilon_t = \\sum_j\\pi_jX_{t-j}$; MA(1): $\\pi_j = (-\\theta)^j$', 'atunci $\\varepsilon_t = \\sum_j\\pi_jX_{t-j}$; MA(1): $\\pi_j = (-\\theta)^j$')])))

D.frame(T('What you need today (2/4): moments', 'Noțiuni necesare azi (2/4): momente'), items(
    (T('\\textbf{AR(1)}: $\\mu = c/(1 - \\phi)$, $\\gamma(0) = \\sigma^2/(1 - \\phi^2)$, $\\rho(h) = \\phi^h$, $\\psi_j = \\phi^j$', '\\textbf{AR(1)}: $\\mu = c/(1 - \\phi)$, $\\gamma(0) = \\sigma^2/(1 - \\phi^2)$, $\\rho(h) = \\phi^h$, $\\psi_j = \\phi^j$'),
     [T('half-life of a shock: $\\ln 0.5/\\ln\\phi$', 'timpul de înjumătățire al unui șoc: $\\ln 0{,}5/\\ln\\phi$')]),
    (T('\\textbf{AR(2)}, Yule--Walker: $\\rho(1) = \\phi_1/(1 - \\phi_2)$, $\\rho(h) = \\phi_1\\rho(h - 1) + \\phi_2\\rho(h - 2)$, $h \\ge 2$', '\\textbf{AR(2)}, Yule--Walker: $\\rho(1) = \\phi_1/(1 - \\phi_2)$, $\\rho(h) = \\phi_1\\rho(h - 1) + \\phi_2\\rho(h - 2)$, $h \\ge 2$'),
     [T('complex roots when $\\phi_1^2 + 4\\phi_2 < 0$: the ACF is a damped wave', 'rădăcini complexe cînd $\\phi_1^2 + 4\\phi_2 < 0$: ACF este o undă amortizată')]),
    (T('\\textbf{MA(1)}: $\\gamma(0) = \\sigma^2(1 + \\theta^2)$, $\\gamma(1) = \\theta\\sigma^2$, $\\rho(1) = \\theta/(1 + \\theta^2)$, $\\rho(h) = 0$ for $h \\ge 2$', '\\textbf{MA(1)}: $\\gamma(0) = \\sigma^2(1 + \\theta^2)$, $\\gamma(1) = \\theta\\sigma^2$, $\\rho(1) = \\theta/(1 + \\theta^2)$, $\\rho(h) = 0$ pentru $h \\ge 2$'),
     [T('$\\theta$ with $\\sigma^2$ and $1/\\theta$ with $\\theta^2\\sigma^2$ have the same autocovariances', '$\\theta$ cu $\\sigma^2$ și $1/\\theta$ cu $\\theta^2\\sigma^2$ au aceleași autocovarianțe')]),
    (T('\\textbf{ARMA(1,1)} $X_t = \\phi X_{t-1} + \\varepsilon_t + \\theta\\varepsilon_{t-1}$: $\\psi_j = \\phi^{j-1}(\\phi + \\theta)$, $j \\ge 1$', '\\textbf{ARMA(1,1)} $X_t = \\phi X_{t-1} + \\varepsilon_t + \\theta\\varepsilon_{t-1}$: $\\psi_j = \\phi^{j-1}(\\phi + \\theta)$, $j \\ge 1$'),
     [T('$\\gamma(0) = \\sigma^2\\frac{1 + 2\\phi\\theta + \\theta^2}{1 - \\phi^2}$, $\\rho(1) = \\frac{(1 + \\phi\\theta)(\\phi + \\theta)}{1 + 2\\phi\\theta + \\theta^2}$, $\\rho(h) = \\phi\\rho(h - 1)$', '$\\gamma(0) = \\sigma^2\\frac{1 + 2\\phi\\theta + \\theta^2}{1 - \\phi^2}$, $\\rho(1) = \\frac{(1 + \\phi\\theta)(\\phi + \\theta)}{1 + 2\\phi\\theta + \\theta^2}$, $\\rho(h) = \\phi\\rho(h - 1)$')])))

D.frame(T('What you need today (3/4): identification and estimation', 'Noțiuni necesare azi (3/4): identificare și estimare'), items(
    (T('\\textbf{Identification}: AR($p$): PACF cuts off after $p$; MA($q$): ACF cuts off after $q$; ARMA: both decay', '\\textbf{Identificarea}: AR($p$): PACF se anulează după $p$; MA($q$): ACF se anulează după $q$; ARMA: ambele descresc'),
     [T('band $\\pm 1.96/\\sqrt{T}$ for the sample ACF and PACF', 'banda $\\pm 1{,}96/\\sqrt{T}$ pentru ACF și PACF de selecție')]),
    (T('\\textbf{Yule--Walker} for AR(2), with sample autocorrelations $\\hat\\rho_1, \\hat\\rho_2$:', '\\textbf{Yule--Walker} pentru AR(2), cu autocorelațiile de selecție $\\hat\\rho_1, \\hat\\rho_2$:'),
     [T('$\\hat\\phi_1 = \\hat\\rho_1(1 - \\hat\\rho_2)/(1 - \\hat\\rho_1^2)$, $\\hat\\phi_2 = (\\hat\\rho_2 - \\hat\\rho_1^2)/(1 - \\hat\\rho_1^2)$, $\\hat\\sigma^2 = \\hat\\gamma(0)(1 - \\hat\\phi_1\\hat\\rho_1 - \\hat\\phi_2\\hat\\rho_2)$',
        '$\\hat\\phi_1 = \\hat\\rho_1(1 - \\hat\\rho_2)/(1 - \\hat\\rho_1^2)$, $\\hat\\phi_2 = (\\hat\\rho_2 - \\hat\\rho_1^2)/(1 - \\hat\\rho_1^2)$, $\\hat\\sigma^2 = \\hat\\gamma(0)(1 - \\hat\\phi_1\\hat\\rho_1 - \\hat\\phi_2\\hat\\rho_2)$')]),
    (T('\\textbf{Maximum likelihood} (MLE): the default of \\texttt{statsmodels} \\texttt{ARIMA(x, order=(p, 0, q))}', '\\textbf{Verosimilitatea maximă} (MLE): metoda implicită a lui \\texttt{ARIMA(x, order=(p, 0, q))} din \\texttt{statsmodels}'),
     [T('\\texttt{const} in the output is the mean $\\mu$; standard errors give $t$-ratios as in regression', '\\texttt{const} din rezultate este media $\\mu$; erorile standard dau rapoarte $t$, ca în regresie')]),
    (T('\\textbf{Information criteria}, $k$ = number of parameters ($\\phi$, $\\theta$, $\\mu$, $\\sigma^2$):', '\\textbf{Criterii informaționale}, $k$ = numărul parametrilor ($\\phi$, $\\theta$, $\\mu$, $\\sigma^2$):'),
     [T('AIC $= -2\\ln L + 2k$, BIC $= -2\\ln L + k\\ln T$; the smallest value wins; BIC prefers smaller models', 'AIC $= -2\\ln L + 2k$, BIC $= -2\\ln L + k\\ln T$; cîștigă valoarea cea mai mică; BIC preferă modele mai mici')])))

D.frame(T('What you need today (4/4): diagnostics and forecasts', 'Noțiuni necesare azi (4/4): diagnosticare și prognoză'), items(
    (T('\\textbf{Ljung--Box on residuals}: $Q^*(m) = T(T + 2)\\sum_{h=1}^{m}\\hat\\rho_{\\hat\\varepsilon}(h)^2/(T - h) \\approx \\chi^2(m - p - q)$', '\\textbf{Ljung--Box pentru reziduuri}: $Q^*(m) = T(T + 2)\\sum_{h=1}^{m}\\hat\\rho_{\\hat\\varepsilon}(h)^2/(T - h) \\approx \\chi^2(m - p - q)$'),
     [T('5\\% critical values: $\\chi^2_{0.95}(5) = 11.07$, $\\chi^2_{0.95}(7) = 14.07$, $\\chi^2_{0.95}(8) = 15.51$, $\\chi^2_{0.95}(9) = 16.92$, $\\chi^2_{0.95}(10) = 18.31$',
        'valori critice de 5\\%: $\\chi^2_{0{,}95}(5) = 11{,}07$, $\\chi^2_{0{,}95}(7) = 14{,}07$, $\\chi^2_{0{,}95}(8) = 15{,}51$, $\\chi^2_{0{,}95}(9) = 16{,}92$, $\\chi^2_{0{,}95}(10) = 18{,}31$'),
      T('\\textbf{Jarque--Bera}: $\\frac{T}{6}(S^2 + (K - 3)^2/4) \\approx \\chi^2(2)$, $S$ skewness, $K$ kurtosis; critical value 5.99', '\\textbf{Jarque--Bera}: $\\frac{T}{6}(S^2 + (K - 3)^2/4) \\approx \\chi^2(2)$, $S$ asimetria, $K$ boltirea; valoarea critică 5,99')]),
    (T('\\textbf{Forecasts}: replace future shocks by 0 and future values by their forecasts', '\\textbf{Prognoze}: înlocuim șocurile viitoare cu 0 și valorile viitoare cu prognozele lor'),
     [T('AR(1): $\\hat X_{T+h} = \\mu + \\phi^h(X_T - \\mu)$; MA($q$): $\\hat X_{T+h} = \\mu$ for $h > q$', 'AR(1): $\\hat X_{T+h} = \\mu + \\phi^h(X_T - \\mu)$; MA($q$): $\\hat X_{T+h} = \\mu$ pentru $h > q$'),
      T('error variance $\\sigma_h^2 = \\sigma^2(1 + \\psi_1^2 + \\dots + \\psi_{h-1}^2)$; 95\\% interval $\\hat X_{T+h} \\pm 1.96\\,\\sigma_h$', 'varianța erorii $\\sigma_h^2 = \\sigma^2(1 + \\psi_1^2 + \\dots + \\psi_{h-1}^2)$; intervalul de 95\\% $\\hat X_{T+h} \\pm 1{,}96\\,\\sigma_h$')]),
    T('\\textbf{RMSE} (root mean squared error) $= \\sqrt{\\frac1n\\sum e_i^2}$; \\textbf{MAE} (mean absolute error) $= \\frac1n\\sum|e_i|$ (Chapter 0)', '\\textbf{RMSE} (rădăcina erorii pătratice medii) $= \\sqrt{\\frac1n\\sum e_i^2}$; \\textbf{MAE} (eroarea absolută medie) $= \\frac1n\\sum|e_i|$ (Capitolul 0)')))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: computations on paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: an AR(1) process', 'A1: un proces AR(1)'),
         items(T('$X_t = 1 + 0.6X_{t-1} + \\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 0.64)$, Gaussian.', '$X_t = 1 + 0{,}6X_{t-1} + \\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0; 0{,}64)$, gaussian.'),
               T('1. Check stationarity and compute the mean and $\\gamma(0)$.', '1. Verificați staționaritatea și calculați media și $\\gamma(0)$.'),
               T('2. Compute $\\rho(2)$, $\\rho(3)$ and the half-life of a shock.', '2. Calculați $\\rho(2)$, $\\rho(3)$ și timpul de înjumătățire al unui șoc.'),
               T('3. With $X_T = 4$, forecast $X_{T+1}$ and $X_{T+2}$.', '3. Cu $X_T = 4$, prognozați $X_{T+1}$ și $X_{T+2}$.'),
               T('4. Compute the 95\\% intervals for both forecasts.', '4. Calculați intervalele de 95\\% pentru ambele prognoze.'),
               T('Report: eight numbers and one sentence on mean reversion.', 'Raportați: opt valori și o frază despre revenirea la medie.')),
         items(T('1. $|0.6| < 1$: stationary; $\\mu = 1/0.4 = @{a1.mu}$; $\\gamma(0) = 0.64/(1 - 0.36) = @{a1.g0}$', '1. $|0{,}6| < 1$: staționar; $\\mu = 1/0{,}4 = @{a1.mu}$; $\\gamma(0) = 0{,}64/(1 - 0{,}36) = @{a1.g0}$'),
               T('2. $\\rho(2) = 0.6^2 = @{a1.r2}$, $\\rho(3) = @{a1.r3}$; half-life $\\ln 0.5/\\ln 0.6 = @{a1.hl}$ periods', '2. $\\rho(2) = 0{,}6^2 = @{a1.r2}$, $\\rho(3) = @{a1.r3}$; timpul de înjumătățire $\\ln 0{,}5/\\ln 0{,}6 = @{a1.hl}$ perioade'),
               T('3. $\\hat X_{T+1} = 2.5 + 0.6 \\cdot 1.5 = @{a1.f1}$; $\\hat X_{T+2} = 2.5 + 0.36 \\cdot 1.5 = @{a1.f2}$', '3. $\\hat X_{T+1} = 2{,}5 + 0{,}6 \\cdot 1{,}5 = @{a1.f1}$; $\\hat X_{T+2} = 2{,}5 + 0{,}36 \\cdot 1{,}5 = @{a1.f2}$'),
               T('4. $\\sigma_1 = 0.8$: $[@{a1.lo1}, @{a1.hi1}]$; $\\sigma_2^2 = 0.64(1 + 0.36) = @{a1.m2}$, $\\sigma_2 = @{a1.sd2}$: $[@{a1.lo2}, @{a1.hi2}]$', '4. $\\sigma_1 = 0{,}8$: $[@{a1.lo1}; @{a1.hi1}]$; $\\sigma_2^2 = 0{,}64(1 + 0{,}36) = @{a1.m2}$, $\\sigma_2 = @{a1.sd2}$: $[@{a1.lo2}; @{a1.hi2}]$'),
               T('The forecasts move from 4 back towards 2.5, and the intervals widen towards $\\pm 1.96$.', 'Prognozele revin de la 4 spre 2,5, iar intervalele se lărgesc spre $\\pm 1{,}96$.')),
         size='scriptsize')

D.proposed(T('A2: an AR(2) process', 'A2: un proces AR(2)'),
           items(T('$X_t = 0.7X_{t-1} - 0.1X_{t-2} + \\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$. Model: A1.', '$X_t = 0{,}7X_{t-1} - 0{,}1X_{t-2} + \\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$. Model: A1.'),
                 T('1. Write the model with the lag operator and factorise $\\phi(z)$.', '1. Scrieți modelul cu operatorul lag și factorizați $\\phi(z)$.'),
                 T('2. Find the roots and decide whether the process is stationary; check the three triangle conditions.', '2. Găsiți rădăcinile și decideți dacă procesul este staționar; verificați cele trei condiții ale triunghiului.'),
                 T('3. Compute $\\rho(1)$, $\\rho(2)$ and $\\rho(3)$ from the Yule--Walker equations.', '3. Calculați $\\rho(1)$, $\\rho(2)$ și $\\rho(3)$ din ecuațiile Yule--Walker.'),
                 T('4. Compute $\\psi_1$, $\\psi_2$ and $\\psi_3$.', '4. Calculați $\\psi_1$, $\\psi_2$ și $\\psi_3$.'),
                 T('Report: the factorisation, two roots and six numbers.', 'Raportați: factorizarea, două rădăcini și șase valori.')),
           items(T('1. $(1 - 0.7L + 0.1L^2)X_t = \\varepsilon_t$; $1 - 0.7z + 0.1z^2 = (1 - 0.5z)(1 - 0.2z)$', '1. $(1 - 0{,}7L + 0{,}1L^2)X_t = \\varepsilon_t$; $1 - 0{,}7z + 0{,}1z^2 = (1 - 0{,}5z)(1 - 0{,}2z)$'),
                 T('2. Roots $@{a2.z1}$ and $@{a2.z2}$, both outside the circle: stationary; $\\phi_1 + \\phi_2 = @{a2.c1} < 1$, $\\phi_2 - \\phi_1 = @{a2.c2} < 1$, $|\\phi_2| = 0.1 < 1$', '2. Rădăcinile $@{a2.z1}$ și $@{a2.z2}$, ambele în afara cercului: staționar; $\\phi_1 + \\phi_2 = @{a2.c1} < 1$, $\\phi_2 - \\phi_1 = @{a2.c2} < 1$, $|\\phi_2| = 0{,}1 < 1$'),
                 T('3. $\\rho(1) = 0.7/1.1 = @{a2.r1}$; $\\rho(2) = 0.7\\rho(1) - 0.1 = @{a2.r2}$; $\\rho(3) = 0.7\\rho(2) - 0.1\\rho(1) = @{a2.r3}$', '3. $\\rho(1) = 0{,}7/1{,}1 = @{a2.r1}$; $\\rho(2) = 0{,}7\\rho(1) - 0{,}1 = @{a2.r2}$; $\\rho(3) = 0{,}7\\rho(2) - 0{,}1\\rho(1) = @{a2.r3}$'),
                 T('4. $\\psi_1 = 0.7$, $\\psi_2 = 0.7^2 - 0.1 = @{a2.p2}$, $\\psi_3 = 0.7\\psi_2 - 0.1\\psi_1 = @{a2.p3}$', '4. $\\psi_1 = 0{,}7$, $\\psi_2 = 0{,}7^2 - 0{,}1 = @{a2.p2}$, $\\psi_3 = 0{,}7\\psi_2 - 0{,}1\\psi_1 = @{a2.p3}$')),
           size='scriptsize')

D.solved(T('A3: an MA(1) that is not invertible', 'A3: un MA(1) care nu este invertibil'),
         items(T('$X_t = \\varepsilon_t + 2\\varepsilon_{t-1}$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$.', '$X_t = \\varepsilon_t + 2\\varepsilon_{t-1}$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$.'),
               T('1. Compute $\\gamma(0)$, $\\gamma(1)$ and $\\rho(1)$.', '1. Calculați $\\gamma(0)$, $\\gamma(1)$ și $\\rho(1)$.'),
               T('2. Say whether the process is stationary and whether it is invertible.', '2. Precizați dacă procesul este staționar și dacă este invertibil.'),
               T('3. Find the invertible MA(1) with the same autocovariances.', '3. Găsiți procesul MA(1) invertibil cu aceleași autocovarianțe.'),
               T('4. Write the first four weights of its AR($\\infty$) form and the forecast $\\hat X_{T+2}$.', '4. Scrieți primele patru ponderi ale formei lui AR($\\infty$) și prognoza $\\hat X_{T+2}$.'),
               T('Report: five numbers, two verdicts and the new model.', 'Raportați: cinci valori, două verdicte și noul model.')),
         items(T('1. $\\gamma(0) = 1 + 4 = @{a3.g0}$, $\\gamma(1) = 2 \\cdot 1 = @{a3.g1}$, $\\rho(1) = 2/5 = @{a3.r1}$', '1. $\\gamma(0) = 1 + 4 = @{a3.g0}$, $\\gamma(1) = 2 \\cdot 1 = @{a3.g1}$, $\\rho(1) = 2/5 = @{a3.r1}$'),
               T('2. Stationary (every finite MA is); not invertible: the root of $1 + 2z$ is $-0.5$, inside the circle', '2. Staționar (orice MA finit este); nu este invertibil: rădăcina lui $1 + 2z$ este $-0{,}5$, în interiorul cercului'),
               T('3. $\\theta^* = 1/2$, $\\sigma^{*2} = 2^2 \\cdot 1 = @{a3.s2}$: $\\gamma(0) = 4 \\cdot 1.25 = 5$, $\\gamma(1) = 0.5 \\cdot 4 = 2$', '3. $\\theta^* = 1/2$, $\\sigma^{*2} = 2^2 \\cdot 1 = @{a3.s2}$: $\\gamma(0) = 4 \\cdot 1{,}25 = 5$, $\\gamma(1) = 0{,}5 \\cdot 4 = 2$'),
               T('4. $u_t = X_t - 0.5X_{t-1} + 0.25X_{t-2} - 0.125X_{t-3} + \\dots$; $\\hat X_{T+2} = \\mu = 0$ (beyond $q = 1$)', '4. $u_t = X_t - 0{,}5X_{t-1} + 0{,}25X_{t-2} - 0{,}125X_{t-3} + \\dots$; $\\hat X_{T+2} = \\mu = 0$ (după $q = 1$)')),
         size='scriptsize')

D.proposed(T('A4: an ARMA(1,1) process', 'A4: un proces ARMA(1,1)'),
           items(T('$X_t = 0.5X_{t-1} + \\varepsilon_t + 0.3\\varepsilon_{t-1}$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$. Model: A1, A3.', '$X_t = 0{,}5X_{t-1} + \\varepsilon_t + 0{,}3\\varepsilon_{t-1}$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$. Model: A1, A3.'),
                 T('1. Check stationarity and invertibility.', '1. Verificați staționaritatea și invertibilitatea.'),
                 T('2. Compute $\\psi_1$, $\\psi_2$ and $\\psi_3$.', '2. Calculați $\\psi_1$, $\\psi_2$ și $\\psi_3$.'),
                 T('3. Compute $\\gamma(0)$, $\\rho(1)$ and $\\rho(2)$.', '3. Calculați $\\gamma(0)$, $\\rho(1)$ și $\\rho(2)$.'),
                 T('4. Show that $X_t = 0.5X_{t-1} + \\varepsilon_t - 0.5\\varepsilon_{t-1}$ is white noise, and say what an estimation program would report for it.', '4. Arătați că $X_t = 0{,}5X_{t-1} + \\varepsilon_t - 0{,}5\\varepsilon_{t-1}$ este zgomot alb și precizați ce ar raporta un program de estimare pentru el.'),
                 T('Report: two verdicts, six numbers and one sentence.', 'Raportați: două verdicte, șase valori și o frază.')),
           items(T('1. $|0.5| < 1$: stationary; $|0.3| < 1$: invertible; no common factor', '1. $|0{,}5| < 1$: staționar; $|0{,}3| < 1$: invertibil; fără factor comun'),
                 T('2. $\\psi_1 = 0.5 + 0.3 = @{a4.p1}$, $\\psi_2 = 0.5 \\cdot 0.8 = @{a4.p2}$, $\\psi_3 = @{a4.p3}$', '2. $\\psi_1 = 0{,}5 + 0{,}3 = @{a4.p1}$, $\\psi_2 = 0{,}5 \\cdot 0{,}8 = @{a4.p2}$, $\\psi_3 = @{a4.p3}$'),
                 T('3. $\\gamma(0) = (1 + 0.3 + 0.09)/0.75 = @{a4.g0}$; $\\rho(1) = 1.15 \\cdot 0.8/1.39 = @{a4.r1}$; $\\rho(2) = 0.5\\rho(1) = @{a4.r2}$', '3. $\\gamma(0) = (1 + 0{,}3 + 0{,}09)/0{,}75 = @{a4.g0}$; $\\rho(1) = 1{,}15 \\cdot 0{,}8/1{,}39 = @{a4.r1}$; $\\rho(2) = 0{,}5\\rho(1) = @{a4.r2}$'),
                 T('4. $(1 - 0.5L)X_t = (1 - 0.5L)\\varepsilon_t$, so $X_t = \\varepsilon_t$; the program reports $\\hat\\phi \\approx -\\hat\\theta$ with huge standard errors: fit white noise instead.', '4. $(1 - 0{,}5L)X_t = (1 - 0{,}5L)\\varepsilon_t$, deci $X_t = \\varepsilon_t$; programul raportează $\\hat\\phi \\approx -\\hat\\theta$ cu erori standard foarte mari: se estimează un zgomot alb.')),
           size='scriptsize')

D.solved(T('A5: Yule--Walker and information criteria', 'A5: Yule--Walker și criterii informaționale'),
         items(T('A stationary series has $T = 100$, $\\hat\\gamma(0) = 4$, $\\hat\\rho(1) = 0.75$, $\\hat\\rho(2) = 0.65$.', 'O serie staționară are $T = 100$, $\\hat\\gamma(0) = 4$, $\\hat\\rho(1) = 0{,}75$, $\\hat\\rho(2) = 0{,}65$.'),
               T('1. Estimate an AR(2) by Yule--Walker: $\\hat\\phi_1$, $\\hat\\phi_2$, $\\hat\\sigma^2$.', '1. Estimați un AR(2) prin Yule--Walker: $\\hat\\phi_1$, $\\hat\\phi_2$, $\\hat\\sigma^2$.'),
               T('2. Check that the estimated model is stationary.', '2. Verificați că modelul estimat este staționar.'),
               T('3. Maximum likelihood gives $\\ln L = -214.6$ for AR(1) ($k = 3$) and $-210.9$ for AR(2) ($k = 4$); compute AIC and BIC.', '3. Verosimilitatea maximă dă $\\ln L = -214{,}6$ pentru AR(1) ($k = 3$) și $-210{,}9$ pentru AR(2) ($k = 4$); calculați AIC și BIC.'),
               T('4. Choose a model with each criterion.', '4. Alegeți un model după fiecare criteriu.'),
               T('Report: three estimates, four criteria and a choice.', 'Raportați: trei estimări, patru valori ale criteriilor și o alegere.')),
         items(T('1. $1 - 0.75^2 = @{a5.det}$; $\\hat\\phi_1 = 0.75 \\cdot 0.35/@{a5.det} = @{a5.f1}$; $\\hat\\phi_2 = (0.65 - 0.5625)/@{a5.det} = @{a5.f2}$', '1. $1 - 0{,}75^2 = @{a5.det}$; $\\hat\\phi_1 = 0{,}75 \\cdot 0{,}35/@{a5.det} = @{a5.f1}$; $\\hat\\phi_2 = (0{,}65 - 0{,}5625)/@{a5.det} = @{a5.f2}$'),
               T('$\\hat\\sigma^2 = 4\\,(1 - 0.6 \\cdot 0.75 - 0.2 \\cdot 0.65) = @{a5.s2}$', '$\\hat\\sigma^2 = 4\\,(1 - 0{,}6 \\cdot 0{,}75 - 0{,}2 \\cdot 0{,}65) = @{a5.s2}$'),
               T('2. $\\hat\\phi_1 + \\hat\\phi_2 = @{a5.c1} < 1$, $\\hat\\phi_2 - \\hat\\phi_1 = @{a5.c2} < 1$, $|\\hat\\phi_2| < 1$: stationary', '2. $\\hat\\phi_1 + \\hat\\phi_2 = @{a5.c1} < 1$, $\\hat\\phi_2 - \\hat\\phi_1 = @{a5.c2} < 1$, $|\\hat\\phi_2| < 1$: staționar'),
               T('3. $\\ln 100 = @{ln100}$; AR(1): AIC $= @{a5.ar1.aic}$, BIC $= @{a5.ar1.bic}$; AR(2): AIC $= @{a5.ar2.aic}$, BIC $= @{a5.ar2.bic}$', '3. $\\ln 100 = @{ln100}$; AR(1): AIC $= @{a5.ar1.aic}$, BIC $= @{a5.ar1.bic}$; AR(2): AIC $= @{a5.ar2.aic}$, BIC $= @{a5.ar2.bic}$'),
               T('4. Both criteria choose AR(2): the gain in $\\ln L$ (3.7) exceeds both penalties per parameter (1 and $\\ln(100)/2$).', '4. Ambele criterii aleg AR(2): cîștigul în $\\ln L$ (3,7) depășește ambele penalizări pe parametru (1 și $\\ln(100)/2$).')),
         size='scriptsize')

D.proposed(T('A6: four candidates and the Ljung--Box test', 'A6: patru candidați și testul Ljung--Box'),
           items(T('$T = 120$; MLE gives $\\ln L$ = $-250.3$ (AR(1), $k = 3$), $-247.1$ (AR(2), $k = 4$), $-247.6$ (ARMA(1,1), $k = 4$), $-246.9$ (ARMA(2,1), $k = 5$). Model: A5.', '$T = 120$; MLE dă $\\ln L$ = $-250{,}3$ (AR(1), $k = 3$), $-247{,}1$ (AR(2), $k = 4$), $-247{,}6$ (ARMA(1,1), $k = 4$), $-246{,}9$ (ARMA(2,1), $k = 5$). Model: A5.'),
                 T('1. Compute AIC and BIC for the four models.', '1. Calculați AIC și BIC pentru cele patru modele.'),
                 T('2. Choose a model with each criterion.', '2. Alegeți un model după fiecare criteriu.'),
                 T('3. The residuals of AR(2) give $Q^*(10) = 14.2$; find the degrees of freedom and decide at 5\\%.', '3. Reziduurile modelului AR(2) dau $Q^*(10) = 14{,}2$; găsiți gradele de libertate și decideți la 5\\%.'),
                 T('4. Explain what changes if one uses 10 degrees of freedom.', '4. Explicați ce se schimbă dacă se folosesc 10 grade de libertate.'),
                 T('Report: a table of eight numbers, two choices and one decision.', 'Raportați: un tabel cu opt valori, două alegeri și o decizie.')),
           items(T('1. $\\ln 120 = @{ln120}$; AIC: @{a6.ar1.aic}; @{a6.ar2.aic}; @{a6.arma11.aic}; @{a6.arma21.aic}', '1. $\\ln 120 = @{ln120}$; AIC: @{a6.ar1.aic}; @{a6.ar2.aic}; @{a6.arma11.aic}; @{a6.arma21.aic}'),
                 T('BIC: @{a6.ar1.bic}; @{a6.ar2.bic}; @{a6.arma11.bic}; @{a6.arma21.bic}', 'BIC: @{a6.ar1.bic}; @{a6.ar2.bic}; @{a6.arma11.bic}; @{a6.arma21.bic}'),
                 T('2. Both choose AR(2); ARMA(2,1) gains only 0.2 in $\\ln L$ for one more parameter', '2. Ambele aleg AR(2); ARMA(2,1) cîștigă doar 0,2 în $\\ln L$ pentru un parametru în plus'),
                 T('3. $df = 10 - 2 = 8$, critical value @{a6.c8}: $14.2 < @{a6.c8}$, p = @{a6.p8}: do not reject (borderline)', '3. $df = 10 - 2 = 8$, valoarea critică @{a6.c8}: $14{,}2 < @{a6.c8}$, p = @{a6.p8}: nu respingem (la limită)'),
                 T('4. With 10 df: p = @{a6.p10}, the residuals look cleaner than they are; the right df brings the test close to rejection.', '4. Cu 10 grade de libertate: p = @{a6.p10}, reziduurile par mai curate decît sînt; gradele de libertate corecte aduc testul aproape de respingere.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: real data and interpretation', 'Partea B: date reale și interpretare')

D.task(T('B1: US real GDP growth [Solved]', 'B1: creșterea PIB-ului real al SUA [Rezolvat]'),
       T('which ARMA model describes US quarterly real GDP growth, and how persistent are growth shocks?', 'ce model ARMA descrie creșterea trimestrială a PIB-ului real al SUA și cît de persistente sînt șocurile de creștere?'),
       T('statsmodels \\texttt{macrodata}, real GDP, @{b1.q0d}--@{b1.q1d}; $y_t = 400\\,\\Delta\\ln Y_t$ (annualised \\%)', 'statsmodels \\texttt{macrodata}, PIB real, @{b1.q0d}--@{b1.q1d}; $y_t = 400\\,\\Delta\\ln Y_t$ (\\% anualizat)'),
       [T('Plot $y_t$ and its ACF and PACF up to lag 12, and propose candidate models.', 'Reprezentați grafic $y_t$ și ACF și PACF pînă la decalajul 12 și propuneți modele candidate.'),
        T('Estimate white noise, AR(1), AR(2), MA(2) and ARMA(1,1) by maximum likelihood, and tabulate AIC and BIC.', 'Estimați zgomotul alb, AR(1), AR(2), MA(2) și ARMA(1,1) prin verosimilitate maximă și tabelați AIC și BIC.'),
        T('Test the residuals of the BIC model with Ljung--Box $Q^*(8)$ on $8 - p - q$ degrees of freedom, and with Jarque--Bera.', 'Testați reziduurile modelului ales de BIC cu Ljung--Box $Q^*(8)$, cu $8 - p - q$ grade de libertate, și cu Jarque--Bera.'),
        T('Interpretation: how many quarters does it take for half of a growth shock to fade?', 'Interpretare: cîte trimestre trec pînă cînd se stinge jumătate dintr-un șoc de creștere?')],
       T('the chart, a table of ten criteria, two test results and one sentence', 'graficul, un tabel cu zece valori ale criteriilor, două rezultate de test și o frază'), size='footnotesize', nb='B1')

D.frame(T('B1: solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch2_sem_b1', h='0.30') + table(
    'lrrrrr', ' & ' + 'WN' + ' & AR(1) & AR(2) & MA(2) & ARMA(1,1)',
    ['AIC & @{b1.00.aic} & @{b1.10.aic} & \\textbf{@{b1.20.aic}} & @{b1.02.aic} & @{b1.11.aic}',
     'BIC & @{b1.00.bic} & \\textbf{@{b1.10.bic}} & @{b1.20.bic} & @{b1.02.bic} & @{b1.11.bic}'], size='scriptsize') + items(
    T('$T = @{b1.T}$, ACF $@{b1.r1}$, $@{b1.r2}$, $@{b1.r3}$ and PACF $@{b1.p1}$, $@{b1.p2}$, $@{b1.p3}$ (band $\\pm @{b1.band}$): AR(1) or AR(2)', '$T = @{b1.T}$, ACF $@{b1.r1}$; $@{b1.r2}$; $@{b1.r3}$ și PACF $@{b1.p1}$; $@{b1.p2}$; $@{b1.p3}$ (banda $\\pm @{b1.band}$): AR(1) sau AR(2)'),
    T('BIC: AR(1), $\\hat\\mu = @{b1.mu}$, $\\hat\\phi = @{b1.phi}$ (@{b1.phis}); AIC: AR(2), $\\hat\\phi_2 = @{b1.a2.p2}$ (@{b1.a2.s2}); the BIC values differ by 0.02', 'BIC: AR(1), $\\hat\\mu = @{b1.mu}$, $\\hat\\phi = @{b1.phi}$ (@{b1.phis}); AIC: AR(2), $\\hat\\phi_2 = @{b1.a2.p2}$ (@{b1.a2.s2}); valorile BIC diferă cu 0,02'),
    T('AR(1) residuals: $Q^*(8) = @{b1.q}$ on @{b1.qdf} df, p = @{b1.qp}; Jarque--Bera @{b1.jb}, p @{b1.jbp}', 'Reziduurile AR(1): $Q^*(8) = @{b1.q}$ cu @{b1.qdf} grade de libertate, p = @{b1.qp}; Jarque--Bera @{b1.jb}, p @{b1.jbp}'),
    T('Interpretation: half-life $\\ln 0.5/\\ln @{b1.phi} = @{b1.hl}$ quarters: growth shocks are short-lived; output levels, not growth rates, carry the persistence', 'Interpretare: timpul de înjumătățire $\\ln 0{,}5/\\ln @{b1.phi} = @{b1.hl}$ trimestre: șocurile de creștere sînt de scurtă durată; persistența se află în nivelul producției, nu în ritmul de creștere')) + qlsem(),
    'scriptsize')

D.task(T('B2: Romanian real GDP growth [Proposed]', 'B2: creșterea PIB-ului real al României [Propus]'),
       T('does Romanian quarterly real GDP growth have any ARMA structure?', 'are creșterea trimestrială a PIB-ului real al României vreo structură ARMA?'),
       T('Eurostat, chain-linked volumes, seasonally and calendar adjusted, @{b2.q0d}--@{b2.q1d}; $y_t = 100\\,\\Delta\\ln Y_t$; model: B1', 'Eurostat, volume înlănțuite, ajustate sezonier și pentru zilele lucrătoare, @{b2.q0d}--@{b2.q1d}; $y_t = 100\\,\\Delta\\ln Y_t$; model: B1'),
       [T('Repeat the four steps of B1 for $y_t$ (\\texttt{ro\\_gdp\\_growth(\'qoq\')}).', 'Repetați cei patru pași din B1 pentru $y_t$ (\\texttt{ro\\_gdp\\_growth(\'qoq\')}).'),
        T('Run Ljung--Box $Q^*(8)$ on the series itself.', 'Aplicați testul Ljung--Box $Q^*(8)$ seriei însăși.'),
        T('Repeat the analysis without the four quarters of 2020, and compare.', 'Repetați analiza fără cele patru trimestre din 2020 și comparați.'),
        T('Interpretation: why is the best forecast of next quarter\'s growth simply the mean?', 'Interpretare: de ce cea mai bună prognoză a creșterii din trimestrul următor este pur și simplu media?')],
       T('the chart, a table of ten criteria, the tests and two sentences', 'graficul, un tabel cu zece valori ale criteriilor, testele și două fraze'), size='footnotesize', nb='B2')

D.frame(T('B2: solution [Proposed]', 'B2: rezolvare [Propus]'), fig('ch2_sem_b2', h='0.30') + table(
    'lrrrrr', ' & ' + 'WN' + ' & AR(1) & AR(2) & MA(2) & ARMA(1,1)',
    ['AIC & \\textbf{@{b2.00.aic}} & @{b2.10.aic} & @{b2.20.aic} & @{b2.02.aic} & @{b2.11.aic}',
     'BIC & \\textbf{@{b2.00.bic}} & @{b2.10.bic} & @{b2.20.bic} & @{b2.02.bic} & @{b2.11.bic}'], size='scriptsize') + items(
    T('$T = @{b2.T}$, mean @{b2.mean}\\%, sd @{b2.sd}\\%; ACF $@{b2.r1}$, $@{b2.r2}$, $@{b2.r3}$ (band $\\pm @{b2.band}$); minimum $@{b2.min}\\%$ in @{b2.mind}', '$T = @{b2.T}$, media @{b2.mean}\\%, abaterea standard @{b2.sd}\\%; ACF $@{b2.r1}$; $@{b2.r2}$; $@{b2.r3}$ (banda $\\pm @{b2.band}$); minimul $@{b2.min}\\%$ în @{b2.mind}'),
    T('Both criteria choose white noise; $Q^*(8) = @{b2.q0}$, p = @{b2.q0p}; Jarque--Bera @{b2.jb}, p @{b2.jbp}; without 2020: p = @{b2x.q0p}, sd @{b2x.sd}\\%', 'Ambele criterii aleg zgomotul alb; $Q^*(8) = @{b2.q0}$, p = @{b2.q0p}; Jarque--Bera @{b2.jb}, p @{b2.jbp}; fără 2020: p = @{b2x.q0p}, abaterea standard @{b2x.sd}\\%'),
    T('Interpretation: past quarterly growth carries no linear information about the next quarter; the forecast is $\\hat\\mu = @{b2.mu}\\%$ (se @{b2.mus}); the annual rate of the lecture is an MA(3) because it sums four such quarters', 'Interpretare: creșterea trimestrială trecută nu conține informație liniară despre trimestrul următor; prognoza este $\\hat\\mu = @{b2.mu}\\%$ (eroarea standard @{b2.mus}); rata anuală din curs este un MA(3) pentru că însumează patru astfel de trimestre')),
    'scriptsize', instructor_only=True)

D.task(T('B3: BET daily returns [Solved]', 'B3: randamentele zilnice ale BET [Rezolvat]'),
       T('is the first-order autocorrelation of BET returns useful for forecasting?', 'este autocorelația de ordinul 1 a randamentelor BET utilă pentru prognoză?'),
       T('BET closes since 2000 (EODHD); $r_t = 100\\,\\Delta\\ln P_t$, $T = @{b3.T}$, last day @{b3.last}', 'închiderile BET din 2000 (EODHD); $r_t = 100\\,\\Delta\\ln P_t$, $T = @{b3.T}$, ultima zi @{b3.last}'),
       [T('Estimate an AR(1) by maximum likelihood and report $\\hat\\phi$, its standard error and $t$-ratio, and $R^2$.', 'Estimați un AR(1) prin verosimilitate maximă și raportați $\\hat\\phi$, eroarea standard, raportul $t$ și $R^2$.'),
        T('Test the residuals with Ljung--Box $Q^*(10)$ on 9 degrees of freedom, and the squared residuals with $Q^*(10)$.', 'Testați reziduurile cu Ljung--Box $Q^*(10)$, cu 9 grade de libertate, și pătratele reziduurilor cu $Q^*(10)$.'),
        T('Compute the kurtosis and the Jarque--Bera statistic of the residuals.', 'Calculați boltirea și statistica Jarque--Bera a reziduurilor.'),
        T('Forecast tomorrow\'s return with its 95\\% interval.', 'Prognozați randamentul de mîine cu intervalul de 95\\%.'),
        T('Interpretation: is the AR(1) coefficient statistically significant, and is it economically useful?', 'Interpretare: este coeficientul AR(1) semnificativ statistic și este el util din punct de vedere economic?')],
       T('six numbers, the chart, a forecast with its interval and two sentences', 'șase valori, graficul, o prognoză cu intervalul ei și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch2_sem_b3', h='0.40') + items(
    T('$\\hat\\phi = @{b3.phi}$ (se @{b3.se}), $t = @{b3.t}$; $R^2 = @{b3.r2}\\%$; $\\hat\\sigma = @{b3.sig}\\%$', '$\\hat\\phi = @{b3.phi}$ (eroarea standard @{b3.se}), $t = @{b3.t}$; $R^2 = @{b3.r2}\\%$; $\\hat\\sigma = @{b3.sig}\\%$'),
    T('Residuals: $Q^*(10) = @{b3.q}$, p @{b3.qp} (some correlation remains); squares: $Q^*(10) = @{b3.q2}$; kurtosis @{b3.kurt}, Jarque--Bera @{b3.jb}', 'Reziduurile: $Q^*(10) = @{b3.q}$, p @{b3.qp} (rămîne puțină corelație); pătratele: $Q^*(10) = @{b3.q2}$; boltirea @{b3.kurt}, Jarque--Bera @{b3.jb}'),
    T('Last return $@{b3.lastr}\\%$: forecast $@{b3.f1}\\%$, 95\\% interval $[@{b3.lo}, @{b3.hi}]$ (Normal, too narrow in the tails)', 'Ultimul randament $@{b3.lastr}\\%$: prognoza $@{b3.f1}\\%$, intervalul de 95\\% $[@{b3.lo}; @{b3.hi}]$ (construit cu distribuția Normală, prea îngust în cozi)'),
    T('Interpretation: highly significant, but it explains about 1\\% of the variance and the forecast is tiny relative to the interval; the large $Q^*$ of the squares points to volatility models (Chapter 5)', 'Interpretare: foarte semnificativ, dar explică aproximativ 1\\% din varianță, iar prognoza este foarte mică în raport cu intervalul; valoarea mare a lui $Q^*$ pentru pătrate trimite la modelele de volatilitate (Capitolul 5)')) + qlsem(),
    'scriptsize')

D.task(T('B4: EUR/RON daily returns [Proposed]', 'B4: randamentele zilnice EUR/RON [Propus]'),
       T('does the EUR/RON exchange rate behave like the BET?', 'se comportă cursul EUR/RON ca BET?'),
       T('BNR reference rate since July 2005, $T = @{b4.T}$ returns; model: B3', 'cursul de referință BNR din iulie 2005, $T = @{b4.T}$ randamente; model: B3'),
       [T('Repeat the four steps of B3 (\\texttt{b3\\_returns(\'eurron\')}).', 'Repetați cei patru pași din B3 (\\texttt{b3\\_returns(\'eurron\')}).'),
        T('Compare $\\hat\\phi$, $R^2$ and the kurtosis with those of the BET.', 'Comparați $\\hat\\phi$, $R^2$ și boltirea cu cele ale BET.'),
        T('Interpretation: why can a managed exchange rate show a larger first-order autocorrelation than a stock index?', 'Interpretare: de ce poate un curs de schimb administrat să aibă o autocorelație de ordinul 1 mai mare decît un indice bursier?')],
       T('a table (two series, six numbers each) and two sentences', 'un tabel (două serii, cîte șase valori) și două fraze'), size='footnotesize', nb='B4')

D.frame(T('B4: solution [Proposed]', 'B4: rezolvare [Propus]'), fig('ch2_sem_b4', h='0.34') + table(
    'lrrrrrr', ' & $\\hat\\phi$ & $t$ & $R^2$ (\\%) & $Q^*(10)$ & $Q^*_{sq}(10)$ & ' + T('kurtosis', 'boltirea'),
    ['BET & $@{b3.phi}$ & @{b3.t} & @{b3.r2} & @{b3.q} & @{b3.q2} & @{b3.kurt}',
     'EUR/RON & $@{b4.phi}$ & @{b4.t} & @{b4.r2} & @{b4.q} & @{b4.q2} & @{b4.kurt}'], size='scriptsize') + items(
    T('EUR/RON: larger $\\hat\\phi$ and $R^2$, residual correlation left ($Q^*(10)$ p @{b4.qp}), fatter tails; forecast $@{b4.f1}\\%$, interval $[@{b4.lo}, @{b4.hi}]$', 'EUR/RON: $\\hat\\phi$ și $R^2$ mai mari, corelație reziduală ($Q^*(10)$, p @{b4.qp}), cozi mai groase; prognoza $@{b4.f1}\\%$, intervalul $[@{b4.lo}; @{b4.hi}]$'),
    T('Interpretation: the central bank smooths the rate, so moves continue for a day or two; rare large jumps give the high kurtosis', 'Interpretare: banca centrală netezește cursul, deci mișcările continuă una sau două zile; salturile mari și rare dau boltirea ridicată')),
    'scriptsize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: open questions and AI critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: does an AR model beat simple forecasts of inflation? [Proposed]', 'C1: bate un model AR prognozele simple ale inflației? [Propus]'),
       T('is an AR(2) better than the historical mean and the last value at forecasting Romanian 12-month inflation one year ahead?', 'este un AR(2) mai bun decît media istorică și ultima valoare la prognoza inflației pe 12 luni din România, cu un an înainte?'),
       T('HICP inflation since 2005 (Eurostat); forecast origins every month from January 2015; models: B1, A1', 'inflația IAPC din 2005 (Eurostat); origini de prognoză în fiecare lună începînd cu ianuarie 2015; modele: B1, A1'),
       [T('At each origin, estimate an AR(2) on all data up to that month and forecast 12 months ahead.', 'La fiecare origine, estimați un AR(2) pe toate datele pînă în luna respectivă și prognozați cu 12 luni înainte.'),
        T('Compute the same forecasts from the historical mean and from the last observed value.', 'Calculați aceleași prognoze din media istorică și din ultima valoare observată.'),
        T('Compare RMSE and MAE of the three methods, over the full period and before July 2021.', 'Comparați RMSE și MAE pentru cele trei metode, pe întreaga perioadă și înainte de iulie 2021.'),
        T('Interpretation: why do all three methods fail in 2022?', 'Interpretare: de ce eșuează toate cele trei metode în 2022?')],
       T('a table of RMSE and MAE, the chart and a plan for a project', 'un tabel cu RMSE și MAE, graficul și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: reference analysis [Proposed]', 'C1: analiză de referință [Propus]'), fig('ch2_sem_c1', h='0.36') + table(
    'lrrr', ' & AR(2) & ' + T('mean', 'media') + ' & ' + T('last value', 'ultima valoare'),
    [T('RMSE, all', 'RMSE, total') + ' & @{c1.rm.ar} & @{c1.rm.mean} & @{c1.rm.naive}',
     T('MAE, all', 'MAE, total') + ' & @{c1.ma.ar} & @{c1.ma.mean} & @{c1.ma.naive}',
     T('RMSE, before July 2021', 'RMSE, înainte de iulie 2021') + ' & @{c1.pre.ar} & @{c1.pre.mean} & @{c1.pre.naive}'], size='scriptsize') + items(
    T('@{c1.n} forecasts for @{c1.d0}--@{c1.d1}; the AR(2) has the smallest errors, but its gain over the last value is small', '@{c1.n} prognoze pentru @{c1.d0}--@{c1.d1}; AR(2) are cele mai mici erori, dar avantajul față de ultima valoare este mic'),
    T('Interpretation: the 2022 energy shock was outside the history of the series; a univariate model learns it only afterwards. A project: add a formal test (Diebold--Mariano), other horizons and other countries', 'Interpretare: șocul energetic din 2022 a fost în afara istoriei seriei; un model univariat îl învață abia după aceea. Un proiect: adăugați un test formal (Diebold--Mariano), alte orizonturi și alte țări')),
    'scriptsize', instructor_only=True)

D.frame(T('C2: audit an AI answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant about the ARMA models of this seminar. The answer:', 'Un student a cerut unui asistent AI informații despre modelele ARMA din acest seminar. Răspunsul:'),
    T('\\aiprompt{(a) The process X(t) = 0.6 X(t-1) + 0.5 X(t-2) + e(t) is stationary because both coefficients are below 1.}', '\\aiprompt{(a) Procesul X(t) = 0,6 X(t-1) + 0,5 X(t-2) + e(t) este staționar, deoarece ambii coeficienți sînt sub 1.}'),
    T('\\aiprompt{(b) An MA(1) with theta = 2 is not stationary, because |theta| > 1.}', '\\aiprompt{(b) Un MA(1) cu theta = 2 nu este staționar, deoarece |theta| > 1.}'),
    T('\\aiprompt{(c) The AR(1) for BET returns has t = 21, so yesterday\'s return explains most of today\'s return.}', '\\aiprompt{(c) Modelul AR(1) pentru randamentele BET are t = 21, deci randamentul de ieri explică cea mai mare parte a randamentului de azi.}'),
    T('\\aiprompt{(d) For the residuals of an ARMA(1,1), Ljung-Box Q(8) is compared with the chi-square distribution with 8 degrees of freedom.}', '\\aiprompt{(d) Pentru reziduurile unui ARMA(1,1), statistica Ljung-Box Q(8) se compară cu distribuția hi-pătrat cu 8 grade de libertate.}'),
    T('\\aiprompt{(e) The forecast of an MA(2) three steps ahead equals the mean of the series.}', '\\aiprompt{(e) Prognoza unui MA(2) cu trei pași înainte este egală cu media seriei.}'),
    T('\\aiprompt{(f) In an AR(1), X(t) = c + phi X(t-1) + e(t), the mean of the series is c.}', '\\aiprompt{(f) Într-un AR(1), X(t) = c + phi X(t-1) + e(t), media seriei este c.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu, dați afirmația corectă și, unde se poate, valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])), 'scriptsize')

D.frame(T('C2: solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: $\\phi_1 + \\phi_2 = 1.1 > 1$; one root of $1 - 0.6z - 0.5z^2$ lies inside the circle: explosive (the lecture example C)', '(a) Greșit: $\\phi_1 + \\phi_2 = 1{,}1 > 1$; o rădăcină a lui $1 - 0{,}6z - 0{,}5z^2$ este în interiorul cercului: proces exploziv (exemplul C din curs)'),
    T('(b) Wrong: every finite MA is stationary; $|\\theta| > 1$ breaks invertibility, not stationarity (A3)', '(b) Greșit: orice MA finit este staționar; $|\\theta| > 1$ încalcă invertibilitatea, nu staționaritatea (A3)'),
    T('(c) Wrong: significant is not large: $\\hat\\phi = @{c2.bphi}$, $\\hat\\phi^2 \\approx @{c2.br2}\\%$ of the variance (B3)', '(c) Greșit: semnificativ nu înseamnă mare: $\\hat\\phi = @{c2.bphi}$, $\\hat\\phi^2 \\approx @{c2.br2}\\%$ din varianță (B3)'),
    T('(d) Wrong: $8 - p - q = 6$ degrees of freedom; with 8 the test is too lenient (A6)', '(d) Greșit: $8 - p - q = 6$ grade de libertate; cu 8 testul este prea indulgent (A6)'),
    T('(e) Correct: beyond $q = 2$ steps the forecast is $\\mu$', '(e) Corect: după $q = 2$ pași prognoza este $\\mu$'),
    T('(f) Wrong: the mean is $c/(1 - \\phi)$; $c$ is the intercept (A1)', '(f) Greșit: media este $c/(1 - \\phi)$; $c$ este termenul liber (A1)')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-up', 'Încheiere')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Stationarity is read from the roots of $\\phi(z)$, invertibility from the roots of $\\theta(z)$, never from single coefficients', 'Staționaritatea se citește din rădăcinile lui $\\phi(z)$, invertibilitatea din rădăcinile lui $\\theta(z)$, niciodată din coeficienți luați separat'),
    T('The mean of an AR model is $c/\\phi(1)$, not the intercept $c$', 'Media unui model AR este $c/\\phi(1)$, nu termenul liber $c$'),
    T('Box--Jenkins: ACF and PACF propose, AIC and BIC choose, residual tests check, forecasts are compared with simple benchmarks', 'Box--Jenkins: ACF și PACF propun, AIC și BIC aleg, testele pe reziduuri verifică, prognozele se compară cu repere simple'),
    T('Ljung--Box on residuals uses $m - p - q$ degrees of freedom', 'Testul Ljung--Box pentru reziduuri folosește $m - p - q$ grade de libertate'),
    T('Statistically significant is not economically large: compare $\\hat\\phi^2$ with the forecast interval', 'Semnificativ statistic nu înseamnă mare din punct de vedere economic: comparați $\\hat\\phi^2$ cu intervalul de prognoză')))

D.frame(T('After the seminar', 'După seminar'), items(
    T('Lecture 2 derives today\'s formulas: roots, invertibility, $\\psi$ weights, estimators, information criteria, forecast intervals, the Box--Jenkins case of Romanian GDP',
      'Cursul 2 deduce formulele de azi: rădăcini, invertibilitate, ponderi $\\psi$, estimatori, criterii informaționale, intervale de prognoză, studiul Box--Jenkins pentru PIB-ul României'),
    T('Try the [Proposed] tasks in the notebook', 'Încercați cerințele [Propus] în notebook'),
    T('C1 can grow into a team project: ARMA forecasts of inflation in Romania and the region against simple benchmarks', 'C1 poate deveni un proiect de echipă: prognoze ARMA ale inflației în România și în regiune, comparate cu repere simple'),
    T('Reading: \\refHP, Ch.~3--4; \\refFPP, Ch.~9; \\refBD, Ch.~3', 'Lectură: \\refHP, cap.~3--4; \\refFPP, cap.~9; \\refBD, cap.~3'),
    T('\\textbf{The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class}',
      '\\textbf{Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar}')))

D.references(bib(['BD', 'BJ', 'BP', 'FPP', 'HP', 'JB', 'LB', 'Akaike', 'Schwarz']), per=16)

if __name__ == '__main__':
    finalize(D.write(V))
