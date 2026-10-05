r"""
build_seminar1.py -- Seminarul 1 (Procese stochastice și staționaritate), EN + RO dintr-o singură sursă
====================================================================================================
Seminarul are loc ÎNAINTEA cursului 1: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu o întrebare de interpretare, C o întrebare deschisă și
critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în versiunea
profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_01/sem1_results.json (seminar1.py).
Ieșire:
  EN/Seminars/seminar1_stochastic_processes_stationarity.tex          (+ _solutions.tex)
  RO/Seminarii/seminar1_procese_stochastice_stationaritate_ro.tex     (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_01/seminar1.py && python3 latex/build_seminar1.py && python3 latex/tsa_build.py compile 1
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, items, table, fig   # noqa: E402
from ch1_common import REFS, T, bib, date, finalize, load_sem, pv, quarter   # noqa: E402

S = load_sem()
V = Values()
D = Deck(1, 'seminar', refs=REFS)


def qlsem():
    return '\\quantlet{TSA\\_ch1\\_seminar}{\\qlurl{TSA_ch1_seminar}}'


# =============================================================================
# CIFRE
# =============================================================================
A1, A2 = S['A1'], S['A2']
V.put('a1.g0', A1['gamma'][0], 2)
V.put('a1.g1', A1['gamma'][1], 2)
V.put('a1.r1', A1['rho'][1], 2)
V.put('a2.g0', A2['gamma'][0], 2)
V.put('a2.g1', A2['gamma'][1], 2)
V.put('a2.g2', A2['gamma'][2], 2)
V.put('a2.r1', A2['rho'][1], 3)
V.put('a2.r2', A2['rho'][2], 3)
A3, A3d = S['A3'], S['A3d']
V.put('a3.var', A3['var_t'], 0)
V.put('a3.vars', A3['var_s'], 0)
V.put('a3.cov', A3['cov'], 0)
V.put('a3.corr', A3['corr'], 3)
V.put('a3.mean', A3d['mean_t'], 0)
A5 = S['A5']
V.put('a5.mean', A5['mean'], 1)
V.put('a5.ss', A5['ss'], 0)
V.put('a5.s1', sum(A5['prods1']), 2)
V.put('a5.s2', sum(A5['prods2']), 2)
V.put('a5.g0', A5['gamma'][0], 3)
V.put('a5.g1', A5['gamma'][1], 4)
V.put('a5.g2', A5['gamma'][2], 3)
V.put('a5.r1', A5['rho'][1], 3)
V.put('a5.r2', A5['rho'][2], 3)
V.put('a5.band', A5['band'], 3)
V.put('a5.q', A5['q_lb'], 2)
V.put('a5.crit', A5['crit'], 2)
V.put('a5.p', A5['p_lb'], 2)
A6 = S['A6']
V.put('a6.bp', A6['q_bp'], 2)
V.put('a6.lb', A6['q_lb'], 2)
V.put('a6.crit', A6['crit'], 2)
V.put('a6.pbp', A6['p_bp'], 3)
V.put('a6.plb', A6['p_lb'], 3)
V.put('a6.band', A6['band'], 3)


def put_ret(key, d):
    V.int(f'{key}.n', d['n'])
    V.put(f'{key}.p1', d['acf_p1'], 3)
    V.put(f'{key}.p10', d['acf_p10'], 3)
    V.put(f'{key}.p50', d['acf_p50'], 3)
    V.put(f'{key}.r1', d['r1'], 3)
    V.put(f'{key}.r1sq', 100 * d['r1sq'], 1)
    V.put(f'{key}.band', d['band'], 3)
    V.raw(f'{key}.nout', str(d['n_out']))
    V.put(f'{key}.sq1', d['sq1'], 2)
    V.put(f'{key}.q', d['lb_r']['lb'], 1)
    V.raw(f'{key}.qp', pv(d['lb_r']['lb_p']))
    V.put(f'{key}.q2', d['lb_r2']['lb'], 0)
    V.raw(f'{key}.q2p', pv(d['lb_r2']['lb_p']))
    V.put(f'{key}.sd', d['sd'], 2)


put_ret('b1', S['B1'])
put_ret('b2s', S['B2']['sp500'])
put_ret('b2e', S['B2']['eurron'])
V.raw('b1.y0', date(S['B1']['first']))
V.raw('end', date('2026-09-18'))
B3 = S['B3']
for k in ['lev', 'd1', 'd4']:
    for c in ['r1', 'r2', 'r4', 'r8']:
        V.put(f'b3.{k}.{c}', B3[k][c], 2)
    V.put(f'b3.{k}.q', B3[k]['lb8']['lb'], 1)
    V.raw(f'b3.{k}.p', pv(B3[k]['lb8']['lb_p']))
    V.put(f'b3.{k}.mean', B3[k]['mean'], 2)
    V.put(f'b3.{k}.sd', B3[k]['sd'], 1)
V.put('b3.min', B3['min_d4'], 1)
V.raw('b3.mind', quarter(B3['min_d4_d']))
V.put('b3.last', B3['last_d4'], 1)
V.raw('b3.lastq', quarter(B3['last']))
V.put('b3.band', B3['band'], 2)
B4 = S['B4']
V.int('b4.n', B4['n'])
V.raw('b4.last', date(B4['last'], day=False))
for c in ['mean_m', 'sd_m', 'r1', 'r12', 'r24', 'band', 'a_r1', 'a_r12', 'a_r24', 'a_max', 'a_last']:
    V.put(f'b4.{c}', B4[c], 2)
V.raw('b4.amaxd', date(B4['a_max_d'], day=False))
V.put('b4.q', B4['lb12']['lb'], 1)
V.raw('b4.p', pv(B4['lb12']['lb_p']))
V.put('b4.ann', 12 * B4['mean_m'], 1)
C1 = S['C1']
for c in ['r1_0', 'r1_1', 'mean0', 'mean1', 'sd0', 'sd1', 'roll_min', 'roll_max', 'roll_last']:
    V.put(f'c1.{c}', C1[c], 2)
V.raw('c1.n0', str(C1['n0']))
V.raw('c1.n1', str(C1['n1']))
V.put('c1.q0', C1['lb0']['lb'], 1)
V.put('c1.q1', C1['lb1']['lb'], 1)
V.raw('c1.p0', pv(C1['lb0']['lb_p']))
V.raw('c1.p1', pv(C1['lb1']['lb_p']))
V.put('c1.band0', 1.96 / C1['n0'] ** 0.5, 3)
V.put('c1.band1', 1.96 / C1['n1'] ** 0.5, 3)
C2 = S['C2']
V.put('c2.p1', C2['acf_p1'], 3)
V.put('c2.r1', C2['rho1'], 3)
V.put('c2.band', C2['band'], 3)
V.put('c2.r2', 100 * C2['r2'], 1)
V.put('c2.q', C2['lb_r']['lb'], 0)
V.put('c2.q2', C2['lb_r2']['lb'], 0)
V.int('c2.n', C2['n'])

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: is a series stationary, is it white noise, and how do the ACF and the Ljung--Box test answer these questions?',
       '\\textbf{Întrebarea}: este o serie staționară, este ea zgomot alb și cum răspund la aceste întrebări ACF și testul Ljung--Box?'),
     [T('this seminar comes \\textbf{before} Lecture 1: the section ``What you need today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 1: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: autocovariances of moving averages and of a random walk, a sample ACF and a Ljung--Box statistic by hand',
        'Partea A: autocovarianțele unor medii mobile și ale unui mers aleator, o ACF de selecție și o statistică Ljung--Box calculate de mînă'),
      T('Part B: BET, S\\&P 500 and EUR/RON returns; Romanian GDP and inflation, each with an interpretation question',
        'Partea B: randamentele BET, S\\&P 500 și EUR/RON; PIB-ul și inflația României, fiecare cu o întrebare de interpretare'),
      T('Part C: an open question for a project and an AI answer to audit', 'Partea C: o întrebare deschisă pentru proiect și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('mean, autocovariances and ACF of MA(1) and MA(2) processes', 'media, autocovarianțele și ACF ale unor procese MA(1) și MA(2)') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('a random walk; a trend-stationary series and its first difference', 'un mers aleator; o serie staționară în jurul trendului și prima ei diferență') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('sample ACF, bands, Box--Pierce and Ljung--Box by hand', 'ACF de selecție, benzi, Box--Pierce și Ljung--Box calculate de mînă') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('prices and returns: BET; S\\&P 500 and EUR/RON', 'prețuri și randamente: BET; S\\&P 500 și EUR/RON') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('transformations: Romanian real GDP; Romanian HICP inflation', 'transformări: PIB-ul real al României; inflația IAPC a României') + ' & ' + SP + ' & B3',
     'C1, C2 & ' + T('has inflation become more persistent? what is wrong in an AI answer?', 'a devenit inflația mai persistentă? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, B4'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați dumneavoastră, după model')))

D.frame(T('Data used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Frequency}', '\\textbf{Frecvența}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['BET, S\\&P 500 & EODHD & ' + T('daily, close', 'zilnic, închidere') + ' & 2000--2026',
     'EUR/RON & ' + T('BNR reference rate', 'cursul de referință BNR') + ' & ' + T('daily', 'zilnic') + ' & 2005--2026',
     T('Real GDP, Romania', 'PIB real, România') + ' & Eurostat (namq\\_10\\_gdp) & ' + T('quarterly, not adjusted', 'trimestrial, neajustat') + ' & 1995--2026',
     T('HICP, Romania', 'IAPC, România') + ' & Eurostat (prc\\_hicp\\_minr) & ' + T('monthly, 2015 = 100', 'lunar, 2015 = 100') + ' & 1996--2026'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekends and repeated holiday closes are dropped; the last day is @{end}',
      'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekendurile și închiderile repetate din zilele libere se elimină; ultima zi este @{end}'),
    T('Growth and inflation rates in \\%: $100\\,\\Delta\\ln Y_t$ (one period), $100\\,\\Delta_4\\ln Y_t$ and $100\\,\\Delta_{12}\\ln P_t$ (one year)',
      'Ratele de creștere și de inflație în \\%: $100\\,\\Delta\\ln Y_t$ (o perioadă), $100\\,\\Delta_4\\ln Y_t$ și $100\\,\\Delta_{12}\\ln P_t$ (un an)'),
    T('In the notebook: \\texttt{b1\\_returns(\'bet\')}, \\texttt{sample\\_acf(x)}, \\texttt{ljung\\_box(x, 10)}; no account or key is needed',
      'În notebook: \\texttt{b1\\_returns(\'bet\')}, \\texttt{sample\\_acf(x)}, \\texttt{ljung\\_box(x, 10)}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What you need today', 'Noțiuni necesare azi')

D.frame(T('What you need today (1/4): processes and moments', 'Noțiuni necesare azi (1/4): procese și momente'), items(
    (T('\\textbf{Stochastic process} $\\{X_t\\}$: one random variable for each date $t$; a time series $x_1, \\dots, x_T$ is one observed path',
       '\\textbf{Proces stochastic} $\\{X_t\\}$: o variabilă aleatoare pentru fiecare dată $t$; o serie de timp $x_1, \\dots, x_T$ este o traiectorie observată'),
     [T('mean $\\mu_t = E[X_t]$; autocovariance $\\gamma(t, s) = \\mathrm{Cov}(X_t, X_s)$', 'media $\\mu_t = E[X_t]$; autocovarianța $\\gamma(t, s) = \\mathrm{Cov}(X_t, X_s)$')]),
    (T('\\textbf{Weakly stationary}: $E[X_t^2] < \\infty$, $E[X_t] = \\mu$ and $\\mathrm{Cov}(X_t, X_{t+h}) = \\gamma(h)$ for every $t$',
       '\\textbf{Slab staționar}: $E[X_t^2] < \\infty$, $E[X_t] = \\mu$ și $\\mathrm{Cov}(X_t, X_{t+h}) = \\gamma(h)$ pentru orice $t$'),
     [T('\\textbf{ACF}: $\\rho(h) = \\gamma(h)/\\gamma(0)$, $\\rho(0) = 1$, $|\\rho(h)| \\le 1$, $\\rho(-h) = \\rho(h)$', '\\textbf{ACF}: $\\rho(h) = \\gamma(h)/\\gamma(0)$, $\\rho(0) = 1$, $|\\rho(h)| \\le 1$, $\\rho(-h) = \\rho(h)$')]),
    (T('Rules for covariances (with constants $a, b, c$):', 'Reguli pentru covarianțe (cu constantele $a, b, c$):'),
     [T('$\\mathrm{Cov}(aX + bY, Z) = a\\,\\mathrm{Cov}(X, Z) + b\\,\\mathrm{Cov}(Y, Z)$; $\\mathrm{Cov}(X + c, Y) = \\mathrm{Cov}(X, Y)$', '$\\mathrm{Cov}(aX + bY, Z) = a\\,\\mathrm{Cov}(X, Z) + b\\,\\mathrm{Cov}(Y, Z)$; $\\mathrm{Cov}(X + c, Y) = \\mathrm{Cov}(X, Y)$'),
      T('$\\mathrm{Var}(\\sum_i a_iX_i) = \\sum_i\\sum_j a_ia_j\\mathrm{Cov}(X_i, X_j)$', '$\\mathrm{Var}(\\sum_i a_iX_i) = \\sum_i\\sum_j a_ia_j\\mathrm{Cov}(X_i, X_j)$')])))

D.frame(T('What you need today (2/4): white noise, moving averages, random walk', 'Noțiuni necesare azi (2/4): zgomot alb, medii mobile, mers aleator'), items(
    (T('\\textbf{White noise} $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$: mean 0, variance $\\sigma^2$, $\\mathrm{Cov}(\\varepsilon_t, \\varepsilon_s) = 0$ for $t \\ne s$',
       '\\textbf{Zgomot alb} $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$: media 0, varianța $\\sigma^2$, $\\mathrm{Cov}(\\varepsilon_t, \\varepsilon_s) = 0$ pentru $t \\ne s$'),
     [T('weak (uncorrelated), i.i.d. (independent), Gaussian (i.i.d. Normal)', 'slab (necorelat), i.i.d. (independent), gaussian (i.i.d. Normal)')]),
    (T('\\textbf{MA($q$)} (moving average): $X_t = \\mu + \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\dots + \\theta_q\\varepsilon_{t-q}$',
       '\\textbf{MA($q$)} (medie mobilă): $X_t = \\mu + \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\dots + \\theta_q\\varepsilon_{t-q}$'),
     [T('with $\\theta_0 = 1$: $\\gamma(h) = \\sigma^2\\sum_{j=0}^{q-h}\\theta_j\\theta_{j+h}$ for $0 \\le h \\le q$, and $\\gamma(h) = 0$ for $h > q$',
        'cu $\\theta_0 = 1$: $\\gamma(h) = \\sigma^2\\sum_{j=0}^{q-h}\\theta_j\\theta_{j+h}$ pentru $0 \\le h \\le q$ și $\\gamma(h) = 0$ pentru $h > q$')]),
    (T('\\textbf{Random walk}: $X_t = X_{t-1} + \\varepsilon_t$, $X_0 = 0$, so $X_t = \\varepsilon_1 + \\dots + \\varepsilon_t$; with drift: $X_t = c + X_{t-1} + \\varepsilon_t$',
       '\\textbf{Mers aleator}: $X_t = X_{t-1} + \\varepsilon_t$, $X_0 = 0$, deci $X_t = \\varepsilon_1 + \\dots + \\varepsilon_t$; cu derivă: $X_t = c + X_{t-1} + \\varepsilon_t$'),
     [T('\\textbf{difference}: $\\Delta X_t = X_t - X_{t-1}$; \\textbf{seasonal difference}: $\\Delta_sX_t = X_t - X_{t-s}$', '\\textbf{diferența}: $\\Delta X_t = X_t - X_{t-1}$; \\textbf{diferența sezonieră}: $\\Delta_sX_t = X_t - X_{t-s}$')])))

D.frame(T('What you need today (3/4): sample ACF and its bands', 'Noțiuni necesare azi (3/4): ACF de selecție și benzile ei'), items(
    (T('Sample mean $\\bar x$, \\textbf{sample autocovariance} and \\textbf{sample ACF} (divisor $T$):', 'Media de selecție $\\bar x$, \\textbf{autocovarianța de selecție} și \\textbf{ACF de selecție} (împărțitorul $T$):'),
     [T('$\\hat\\gamma(h) = \\frac1T\\sum_{t=1}^{T-h}(x_t - \\bar x)(x_{t+h} - \\bar x)$, \\quad $\\hat\\rho(h) = \\hat\\gamma(h)/\\hat\\gamma(0)$', '$\\hat\\gamma(h) = \\frac1T\\sum_{t=1}^{T-h}(x_t - \\bar x)(x_{t+h} - \\bar x)$, \\quad $\\hat\\rho(h) = \\hat\\gamma(h)/\\hat\\gamma(0)$')]),
    (T('If the series is i.i.d.: $\\hat\\rho(h) \\approx N(0, 1/T)$; \\textbf{95\\% band} $\\pm 1.96/\\sqrt{T}$', 'Dacă seria este i.i.d.: $\\hat\\rho(h) \\approx N(0, 1/T)$; \\textbf{banda de 95\\%} $\\pm 1{,}96/\\sqrt{T}$'),
     [T('a bar outside the band: $\\rho(h) \\ne 0$ at 5\\%, for that lag alone; with 20 lags, about 1 bar crosses by chance', 'o bară în afara benzii: $\\rho(h) \\ne 0$ la 5\\%, doar pentru acel decalaj; cu 20 de decalaje, aproximativ o bară iese din întîmplare')]),
    (T('Typical patterns', 'Tipare tipice'),
     [T('white noise: no bar outside; MA($q$): bars up to lag $q$, then nothing; AR(1): $\\rho(h) = \\phi^h$, geometric decay', 'zgomot alb: nicio bară în afară; MA($q$): bare pînă la decalajul $q$, apoi nimic; AR(1): $\\rho(h) = \\phi^h$, descreștere geometrică'),
      T('random walk or trend: very slow, almost linear decay; seasonal data: peaks at lags $s, 2s, \\dots$', 'mers aleator sau trend: descreștere foarte lentă, aproape liniară; date sezoniere: vîrfuri la decalajele $s, 2s, \\dots$')])))

D.frame(T('What you need today (4/4): portmanteau tests and transformations', 'Noțiuni necesare azi (4/4): teste portmanteau și transformări'), items(
    (T('$H_0$: $\\rho(1) = \\dots = \\rho(m) = 0$; under $H_0$ both statistics are approximately $\\chi^2(m)$', '$H_0$: $\\rho(1) = \\dots = \\rho(m) = 0$; în ipoteza $H_0$, ambele statistici urmează aproximativ $\\chi^2(m)$'),
     [T('\\textbf{Box--Pierce}: $Q(m) = T\\sum_{h=1}^{m}\\hat\\rho(h)^2$; \\textbf{Ljung--Box}: $Q^*(m) = T(T + 2)\\sum_{h=1}^{m}\\hat\\rho(h)^2/(T - h)$',
        '\\textbf{Box--Pierce}: $Q(m) = T\\sum_{h=1}^{m}\\hat\\rho(h)^2$; \\textbf{Ljung--Box}: $Q^*(m) = T(T + 2)\\sum_{h=1}^{m}\\hat\\rho(h)^2/(T - h)$'),
      T('5\\% critical values: $\\chi^2_{0.95}(2) = 5.99$, $\\chi^2_{0.95}(3) = 7.81$, $\\chi^2_{0.95}(8) = 15.51$, $\\chi^2_{0.95}(10) = 18.31$, $\\chi^2_{0.95}(12) = 21.03$',
        'valori critice de 5\\%: $\\chi^2_{0{,}95}(2) = 5{,}99$, $\\chi^2_{0{,}95}(3) = 7{,}81$, $\\chi^2_{0{,}95}(8) = 15{,}51$, $\\chi^2_{0{,}95}(10) = 18{,}31$, $\\chi^2_{0{,}95}(12) = 21{,}03$')]),
    (T('\\textbf{Transformations} of a positive series $Y_t$', '\\textbf{Transformări} ale unei serii pozitive $Y_t$'),
     [T('log: $\\ln Y_t$ stabilises a variance that grows with the level; $100\\,\\Delta\\ln Y_t \\approx$ growth in \\%', 'logaritmul: $\\ln Y_t$ stabilizează o varianță care crește cu nivelul; $100\\,\\Delta\\ln Y_t \\approx$ creșterea în \\%'),
      T('differences remove trends and random walks; $\\Delta_4$ and $\\Delta_{12}$ remove the season of quarterly and monthly data', 'diferențele elimină trendurile și mersurile aleatoare; $\\Delta_4$ și $\\Delta_{12}$ elimină sezonul datelor trimestriale și lunare'),
      T('Box--Cox: $(Y_t^\\lambda - 1)/\\lambda$, with $\\lambda = 0$ meaning $\\ln Y_t$', 'Box--Cox: $(Y_t^\\lambda - 1)/\\lambda$, iar $\\lambda = 0$ înseamnă $\\ln Y_t$')])))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: computations on paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: an MA(1) process', 'A1: un proces MA(1)'),
         items(T('$X_t = \\varepsilon_t + 0.5\\,\\varepsilon_{t-1}$, with $\\varepsilon_t \\sim \\mathrm{WN}(0, 2)$.', '$X_t = \\varepsilon_t + 0{,}5\\,\\varepsilon_{t-1}$, cu $\\varepsilon_t \\sim \\mathrm{WN}(0, 2)$.'),
               T('1. Compute the mean $E[X_t]$.', '1. Calculați media $E[X_t]$.'),
               T('2. Compute $\\gamma(0)$, $\\gamma(1)$ and $\\gamma(2)$.', '2. Calculați $\\gamma(0)$, $\\gamma(1)$ și $\\gamma(2)$.'),
               T('3. Compute $\\rho(1)$ and $\\rho(2)$.', '3. Calculați $\\rho(1)$ și $\\rho(2)$.'),
               T('4. Say whether the process is weakly stationary, and why.', '4. Precizați dacă procesul este slab staționar și de ce.'),
               T('Report: five numbers and one sentence.', 'Raportați: cinci valori și o frază.')),
         items(T('1. $E[X_t] = 0 + 0.5 \\cdot 0 = 0$', '1. $E[X_t] = 0 + 0{,}5 \\cdot 0 = 0$'),
               T('2. $\\gamma(0) = \\mathrm{Var}(\\varepsilon_t) + 0.25\\,\\mathrm{Var}(\\varepsilon_{t-1}) = 2 + 0.5 = @{a1.g0}$', '2. $\\gamma(0) = \\mathrm{Var}(\\varepsilon_t) + 0{,}25\\,\\mathrm{Var}(\\varepsilon_{t-1}) = 2 + 0{,}5 = @{a1.g0}$'),
               T('$\\gamma(1) = \\mathrm{Cov}(\\varepsilon_t + 0.5\\varepsilon_{t-1}, \\varepsilon_{t+1} + 0.5\\varepsilon_t) = 0.5 \\cdot 2 = @{a1.g1}$; $\\gamma(2) = 0$ (no common shock)',
                 '$\\gamma(1) = \\mathrm{Cov}(\\varepsilon_t + 0{,}5\\varepsilon_{t-1}, \\varepsilon_{t+1} + 0{,}5\\varepsilon_t) = 0{,}5 \\cdot 2 = @{a1.g1}$; $\\gamma(2) = 0$ (niciun șoc comun)'),
               T('3. $\\rho(1) = @{a1.g1}/@{a1.g0} = @{a1.r1}$; $\\rho(2) = 0$', '3. $\\rho(1) = @{a1.g1}/@{a1.g0} = @{a1.r1}$; $\\rho(2) = 0$'),
               T('4. Yes: the mean and all $\\gamma(h)$ are finite and do not depend on $t$.', '4. Da: media și toate valorile $\\gamma(h)$ sînt finite și nu depind de $t$.')),
         size='scriptsize')

D.proposed(T('A2: an MA(2) process', 'A2: un proces MA(2)'),
           items(T('$X_t = 2 + \\varepsilon_t + 0.4\\,\\varepsilon_{t-1} - 0.3\\,\\varepsilon_{t-2}$, with $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$. Model: A1.',
                   '$X_t = 2 + \\varepsilon_t + 0{,}4\\,\\varepsilon_{t-1} - 0{,}3\\,\\varepsilon_{t-2}$, cu $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$. Model: A1.'),
                 T('1. Compute the mean.', '1. Calculați media.'),
                 T('2. Compute $\\gamma(0)$, $\\gamma(1)$, $\\gamma(2)$ and $\\gamma(3)$.', '2. Calculați $\\gamma(0)$, $\\gamma(1)$, $\\gamma(2)$ și $\\gamma(3)$.'),
                 T('3. Compute $\\rho(1)$ and $\\rho(2)$, and sketch the ACF up to lag 4.', '3. Calculați $\\rho(1)$ și $\\rho(2)$ și schițați ACF pînă la decalajul 4.'),
                 T('4. Say how the constant 2 changes the autocovariances.', '4. Precizați cum schimbă constanta 2 autocovarianțele.'),
                 T('Report: six numbers, the sketch and one sentence.', 'Raportați: șase valori, schița și o frază.')),
           items(T('1. $E[X_t] = 2$', '1. $E[X_t] = 2$'),
                 T('2. $\\gamma(0) = 1 + 0.16 + 0.09 = @{a2.g0}$; $\\gamma(1) = 0.4 + 0.4 \\cdot (-0.3) = @{a2.g1}$; $\\gamma(2) = -0.3 = @{a2.g2}$; $\\gamma(3) = 0$',
                   '2. $\\gamma(0) = 1 + 0{,}16 + 0{,}09 = @{a2.g0}$; $\\gamma(1) = 0{,}4 + 0{,}4 \\cdot (-0{,}3) = @{a2.g1}$; $\\gamma(2) = -0{,}3 = @{a2.g2}$; $\\gamma(3) = 0$'),
                 T('3. $\\rho(1) = @{a2.r1}$, $\\rho(2) = @{a2.r2}$, $\\rho(h) = 0$ for $h \\ge 3$: the ACF cuts off after lag 2', '3. $\\rho(1) = @{a2.r1}$, $\\rho(2) = @{a2.r2}$, $\\rho(h) = 0$ pentru $h \\ge 3$: ACF se anulează după decalajul 2'),
                 T('4. Not at all: a constant shifts the mean, not the covariances.', '4. Deloc: o constantă deplasează media, nu și covarianțele.')),
           size='scriptsize')

D.solved(T('A3: a random walk', 'A3: un mers aleator'),
         items(T('$X_t = X_{t-1} + \\varepsilon_t$, $X_0 = 0$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 0.25)$.', '$X_t = X_{t-1} + \\varepsilon_t$, $X_0 = 0$, $\\varepsilon_t \\sim \\mathrm{WN}(0; 0{,}25)$.'),
               T('1. Compute $E[X_{100}]$, $\\mathrm{Var}(X_{100})$ and $\\mathrm{Var}(X_{120})$.', '1. Calculați $E[X_{100}]$, $\\mathrm{Var}(X_{100})$ și $\\mathrm{Var}(X_{120})$.'),
               T('2. Compute $\\mathrm{Cov}(X_{100}, X_{120})$ and $\\mathrm{Corr}(X_{100}, X_{120})$.', '2. Calculați $\\mathrm{Cov}(X_{100}, X_{120})$ și $\\mathrm{Corr}(X_{100}, X_{120})$.'),
               T('3. Say whether $X_t$ and $\\Delta X_t$ are stationary.', '3. Precizați dacă $X_t$ și $\\Delta X_t$ sînt staționare.'),
               T('4. With a drift $c = 0.1$, compute $E[X_{100}]$.', '4. Cu o derivă $c = 0{,}1$, calculați $E[X_{100}]$.'),
               T('Report: six numbers and two sentences.', 'Raportați: șase valori și două fraze.')),
         items(T('1. $X_t = \\sum_{i=1}^{t}\\varepsilon_i$: $E[X_{100}] = 0$; $\\mathrm{Var}(X_{100}) = 100 \\cdot 0.25 = @{a3.var}$; $\\mathrm{Var}(X_{120}) = @{a3.vars}$',
                 '1. $X_t = \\sum_{i=1}^{t}\\varepsilon_i$: $E[X_{100}] = 0$; $\\mathrm{Var}(X_{100}) = 100 \\cdot 0{,}25 = @{a3.var}$; $\\mathrm{Var}(X_{120}) = @{a3.vars}$'),
               T('2. The common shocks are $\\varepsilon_1, \\dots, \\varepsilon_{100}$: $\\mathrm{Cov} = @{a3.cov}$; $\\mathrm{Corr} = @{a3.cov}/\\sqrt{@{a3.var} \\cdot @{a3.vars}} = \\sqrt{100/120} = @{a3.corr}$',
                 '2. Șocurile comune sînt $\\varepsilon_1, \\dots, \\varepsilon_{100}$: $\\mathrm{Cov} = @{a3.cov}$; $\\mathrm{Corr} = @{a3.cov}/\\sqrt{@{a3.var} \\cdot @{a3.vars}} = \\sqrt{100/120} = @{a3.corr}$'),
               T('3. $X_t$: no, its variance grows with $t$; $\\Delta X_t = \\varepsilon_t$: yes, it is white noise.', '3. $X_t$: nu, varianța lui crește cu $t$; $\\Delta X_t = \\varepsilon_t$: da, este zgomot alb.'),
               T('4. $E[X_{100}] = 100 \\cdot 0.1 = @{a3.mean}$; the variance does not change.', '4. $E[X_{100}] = 100 \\cdot 0{,}1 = @{a3.mean}$; varianța nu se schimbă.')),
         size='scriptsize')

D.proposed(T('A4: a trend-stationary series', 'A4: o serie staționară în jurul trendului'),
           items(T('$Y_t = 5 + 0.2\\,t + \\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$. Model: A3.', '$Y_t = 5 + 0{,}2\\,t + \\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, 1)$. Model: A3.'),
                 T('1. Compute $E[Y_t]$ and $\\mathrm{Var}(Y_t)$, and say whether $Y_t$ is stationary.', '1. Calculați $E[Y_t]$ și $\\mathrm{Var}(Y_t)$ și precizați dacă $Y_t$ este staționar.'),
                 T('2. Write $D_t = \\Delta Y_t$ and compute its mean, $\\gamma(0)$, $\\gamma(1)$ and $\\rho(1)$.', '2. Scrieți $D_t = \\Delta Y_t$ și calculați media, $\\gamma(0)$, $\\gamma(1)$ și $\\rho(1)$.'),
                 T('3. Compare $Y_t$ with a random walk with drift $0.2$: what happens to a shock $\\varepsilon_{50}$ in each?', '3. Comparați $Y_t$ cu un mers aleator cu deriva $0{,}2$: ce se întîmplă cu un șoc $\\varepsilon_{50}$ în fiecare caz?'),
                 T('Report: five numbers and two sentences.', 'Raportați: cinci valori și două fraze.')),
           items(T('1. $E[Y_t] = 5 + 0.2t$ depends on $t$; $\\mathrm{Var}(Y_t) = 1$; not stationary (stationary around the trend)', '1. $E[Y_t] = 5 + 0{,}2t$ depinde de $t$; $\\mathrm{Var}(Y_t) = 1$; nu este staționar (este staționar în jurul trendului)'),
                 T('2. $D_t = 0.2 + \\varepsilon_t - \\varepsilon_{t-1}$: mean 0.2, $\\gamma(0) = 2$, $\\gamma(1) = -1$, $\\rho(1) = -0.5$', '2. $D_t = 0{,}2 + \\varepsilon_t - \\varepsilon_{t-1}$: media 0,2, $\\gamma(0) = 2$, $\\gamma(1) = -1$, $\\rho(1) = -0{,}5$'),
                 T('$D_t$ is stationary but over-differenced: an MA(1) with $\\theta = -1$', '$D_t$ este staționar, dar supradiferențiat: un MA(1) cu $\\theta = -1$'),
                 T('3. In $Y_t$ the shock affects only $Y_{50}$; in the random walk it stays in every later level.', '3. În $Y_t$ șocul afectează doar $Y_{50}$; în mersul aleator rămîne în toate nivelurile ulterioare.')),
           size='scriptsize')

D.solved(T('A5: a sample ACF by hand', 'A5: o ACF de selecție calculată de mînă'),
         items(T('Eight observations: $x = (4, 6, 5, 8, 7, 9, 6, 7)$.', 'Opt observații: $x = (4, 6, 5, 8, 7, 9, 6, 7)$.'),
               T('1. Compute $\\bar x$ and the deviations $x_t - \\bar x$.', '1. Calculați $\\bar x$ și abaterile $x_t - \\bar x$.'),
               T('2. Compute $\\hat\\gamma(0)$, $\\hat\\gamma(1)$, $\\hat\\gamma(2)$ with divisor $T = 8$, then $\\hat\\rho(1)$ and $\\hat\\rho(2)$.', '2. Calculați $\\hat\\gamma(0)$, $\\hat\\gamma(1)$, $\\hat\\gamma(2)$ cu împărțitorul $T = 8$, apoi $\\hat\\rho(1)$ și $\\hat\\rho(2)$.'),
               T('3. Compute the band $\\pm 1.96/\\sqrt{T}$ and compare.', '3. Calculați banda $\\pm 1{,}96/\\sqrt{T}$ și comparați.'),
               T('4. Compute the Ljung--Box $Q^*(2)$ and decide at 5\\%.', '4. Calculați statistica Ljung--Box $Q^*(2)$ și decideți la 5\\%.'),
               T('Report: seven numbers and one sentence.', 'Raportați: șapte valori și o frază.')),
         items(T('1. $\\bar x = 52/8 = @{a5.mean}$; deviations $(-2.5, -0.5, -1.5, 1.5, 0.5, 2.5, -0.5, 0.5)$', '1. $\\bar x = 52/8 = @{a5.mean}$; abaterile $(-2{,}5; -0{,}5; -1{,}5; 1{,}5; 0{,}5; 2{,}5; -0{,}5; 0{,}5)$'),
               T('2. sum of squares @{a5.ss}: $\\hat\\gamma(0) = @{a5.g0}$; products at lag 1 sum to @{a5.s1}: $\\hat\\gamma(1) = @{a5.g1}$; at lag 2 to @{a5.s2}: $\\hat\\gamma(2) = @{a5.g2}$',
                 '2. suma pătratelor @{a5.ss}: $\\hat\\gamma(0) = @{a5.g0}$; produsele la decalajul 1 au suma @{a5.s1}: $\\hat\\gamma(1) = @{a5.g1}$; la decalajul 2: @{a5.s2}, $\\hat\\gamma(2) = @{a5.g2}$'),
               T('$\\hat\\rho(1) = @{a5.r1}$, $\\hat\\rho(2) = @{a5.r2}$', '$\\hat\\rho(1) = @{a5.r1}$, $\\hat\\rho(2) = @{a5.r2}$'),
               T('3. Band $\\pm @{a5.band}$: both inside', '3. Banda $\\pm @{a5.band}$: ambele în interior'),
               T('4. $Q^*(2) = 8 \\cdot 10\\,(\\hat\\rho(1)^2/7 + \\hat\\rho(2)^2/6) = @{a5.q} < @{a5.crit}$ (p = @{a5.p}): do not reject white noise; with $T = 8$ the test has almost no power.',
                 '4. $Q^*(2) = 8 \\cdot 10\\,(\\hat\\rho(1)^2/7 + \\hat\\rho(2)^2/6) = @{a5.q} < @{a5.crit}$ (p = @{a5.p}): nu respingem ipoteza de zgomot alb; cu $T = 8$, testul nu are aproape nicio putere.')),
         size='scriptsize')

D.proposed(T('A6: Box--Pierce or Ljung--Box?', 'A6: Box--Pierce sau Ljung--Box?'),
           items(T('A series of $T = 120$ monthly returns has $\\hat\\rho(1) = 0.21$, $\\hat\\rho(2) = -0.08$, $\\hat\\rho(3) = 0.12$. Model: A5.',
                   'O serie de $T = 120$ de randamente lunare are $\\hat\\rho(1) = 0{,}21$, $\\hat\\rho(2) = -0{,}08$, $\\hat\\rho(3) = 0{,}12$. Model: A5.'),
                 T('1. Compute the band and say which autocorrelations lie outside it.', '1. Calculați banda și precizați care autocorelații sînt în afara ei.'),
                 T('2. Compute $Q(3)$ and $Q^*(3)$.', '2. Calculați $Q(3)$ și $Q^*(3)$.'),
                 T('3. Compare both with $\\chi^2_{0.95}(3) = 7.81$ and decide.', '3. Comparați ambele statistici cu $\\chi^2_{0{,}95}(3) = 7{,}81$ și decideți.'),
                 T('4. Explain why the two tests can disagree, and which one you trust here.', '4. Explicați de ce cele două teste pot da concluzii diferite și în care aveți încredere aici.'),
                 T('Report: four numbers and two sentences.', 'Raportați: patru valori și două fraze.')),
           items(T('1. $\\pm 1.96/\\sqrt{120} = \\pm @{a6.band}$: only $\\hat\\rho(1)$ is outside', '1. $\\pm 1{,}96/\\sqrt{120} = \\pm @{a6.band}$: doar $\\hat\\rho(1)$ este în afară'),
                 T('2. $Q(3) = 120\\,(0.0441 + 0.0064 + 0.0144) = @{a6.bp}$; $Q^*(3) = 120 \\cdot 122\\,(0.0441/119 + 0.0064/118 + 0.0144/117) = @{a6.lb}$',
                   '2. $Q(3) = 120\\,(0{,}0441 + 0{,}0064 + 0{,}0144) = @{a6.bp}$; $Q^*(3) = 120 \\cdot 122\\,(0{,}0441/119 + 0{,}0064/118 + 0{,}0144/117) = @{a6.lb}$'),
                 T('3. $Q = @{a6.bp} < 7.81$ (p = @{a6.pbp}): do not reject; $Q^* = @{a6.lb} > 7.81$ (p = @{a6.plb}): reject', '3. $Q = @{a6.bp} < 7{,}81$ (p = @{a6.pbp}): nu respingem; $Q^* = @{a6.lb} > 7{,}81$ (p = @{a6.plb}): respingem'),
                 T('4. $Q$ is too small in small samples; $Q^*$ corrects it. Trust Ljung--Box, but note that the evidence is borderline.', '4. $Q$ este prea mic în eșantioane mici; $Q^*$ corectează acest lucru. Avem încredere în Ljung--Box, dar dovezile sînt la limită.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: real data and interpretation', 'Partea B: date reale și interpretare')

D.task(T('B1: BET prices and returns [Solved]', 'B1: prețurile și randamentele BET [Rezolvat]'),
       T('is the BET log price stationary, and are BET daily returns white noise?', 'este logaritmul prețului BET staționar și sînt randamentele zilnice BET zgomot alb?'),
       T('BET closes since 2000; $\\ln P_t$ and $r_t = 100\\,\\Delta\\ln P_t$', 'închiderile BET din 2000; $\\ln P_t$ și $r_t = 100\\,\\Delta\\ln P_t$'),
       [T('Compute the sample ACF of $\\ln P_t$ at lags 1, 10 and 50.', 'Calculați ACF de selecție a lui $\\ln P_t$ la decalajele 1, 10 și 50.'),
        T('Draw the ACF of $r_t$ up to lag 20 with the band $\\pm 1.96/\\sqrt{T}$, and count the bars outside it.', 'Desenați ACF a lui $r_t$ pînă la decalajul 20 cu banda $\\pm 1{,}96/\\sqrt{T}$ și numărați barele din afara ei.'),
        T('Compute Ljung--Box $Q^*(10)$ for $r_t$ and for $r_t^2$, with their p-values.', 'Calculați statistica Ljung--Box $Q^*(10)$ pentru $r_t$ și pentru $r_t^2$, cu valorile p.'),
        T('Interpretation: can yesterday\'s return be used to forecast today\'s return?', 'Interpretare: poate fi folosit randamentul de ieri pentru a prognoza randamentul de azi?')],
       T('three ACF values, the chart, two statistics with p-values and two sentences', 'trei valori ale ACF, graficul, două statistici cu valorile p și două fraze'), size='footnotesize', nb='B1')

D.frame(T('B1: solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch1_sem_b1', h='0.42') + items(
    T('$\\ln P_t$: $\\hat\\rho(1) = @{b1.p1}$, $\\hat\\rho(10) = @{b1.p10}$, $\\hat\\rho(50) = @{b1.p50}$: almost no decay, a random-walk-like level, not stationary',
      '$\\ln P_t$: $\\hat\\rho(1) = @{b1.p1}$, $\\hat\\rho(10) = @{b1.p10}$, $\\hat\\rho(50) = @{b1.p50}$: aproape nicio descreștere, un nivel de tip mers aleator, nestaționar'),
    T('$r_t$ ($T = @{b1.n}$, band $\\pm @{b1.band}$): @{b1.nout} of 20 bars outside, the largest $\\hat\\rho(1) = @{b1.r1}$; $Q^*(10) = @{b1.q}$, p @{b1.qp}',
      '$r_t$ ($T = @{b1.n}$, banda $\\pm @{b1.band}$): @{b1.nout} din 20 de bare în afară, cea mai mare $\\hat\\rho(1) = @{b1.r1}$; $Q^*(10) = @{b1.q}$, p @{b1.qp}'),
    T('$r_t^2$: $\\hat\\rho(1) = @{b1.sq1}$; $Q^*(10) = @{b1.q2}$, p @{b1.q2p}: much stronger dependence', '$r_t^2$: $\\hat\\rho(1) = @{b1.sq1}$; $Q^*(10) = @{b1.q2}$, p @{b1.q2p}: o dependență mult mai puternică'),
    T('Interpretation: yesterday\'s return explains only $\\hat\\rho(1)^2 = @{b1.r1sq}\\%$ of today\'s variance, too little for a trading rule after costs; the size of yesterday\'s move does predict today\'s risk',
      'Interpretare: randamentul de ieri explică doar $\\hat\\rho(1)^2 = @{b1.r1sq}\\%$ din varianța randamentului de azi, prea puțin pentru o regulă de tranzacționare după costuri; mărimea mișcării de ieri anticipează însă riscul de azi')) + qlsem(),
    'scriptsize')

D.task(T('B2: the S\\&P 500 and EUR/RON [Proposed]', 'B2: S\\&P 500 și EUR/RON [Propus]'),
       T('do the S\\&P 500 and the EUR/RON rate behave like the BET?', 'se comportă S\\&P 500 și cursul EUR/RON ca BET?'),
       T('S\\&P 500 closes since 2000; EUR/RON (BNR reference rate) since July 2005; model: B1', 'închiderile S\\&P 500 din 2000; EUR/RON (cursul de referință BNR) din iulie 2005; model: B1'),
       [T('Repeat the three steps of B1 for both series.', 'Repetați cei trei pași din B1 pentru ambele serii.'),
        T('Compare the sign and the size of $\\hat\\rho(1)$ of the returns across BET, S\\&P 500 and EUR/RON.', 'Comparați semnul și mărimea lui $\\hat\\rho(1)$ al randamentelor pentru BET, S\\&P 500 și EUR/RON.'),
        T('Compare $Q^*(10)$ of the squared returns across the three series.', 'Comparați $Q^*(10)$ al randamentelor la pătrat pentru cele trei serii.'),
        T('Interpretation: which of the three return series is closest to white noise?', 'Interpretare: care dintre cele trei serii de randamente este cea mai apropiată de zgomotul alb?')],
       T('one table (three series, six numbers each) and two sentences', 'un tabel (trei serii, cîte șase valori) și două fraze'), size='footnotesize', nb='B2')


def b2row(lab, k):
    return f'{lab} & @{{{k}.n}} & $@{{{k}.p1}}$ & $@{{{k}.r1}}$ & @{{{k}.q}} & $@{{{k}.sq1}}$ & @{{{k}.q2}}'


D.frame(T('B2: solution [Proposed]', 'B2: rezolvare [Propus]'), fig('ch1_sem_b2', h='0.34') + table(
    'lrrrrrr', T('& $T$ & ACF(1) $\\ln P_t$ & $\\hat\\rho(1)$ $r_t$ & $Q^*(10)$ $r_t$ & $\\hat\\rho(1)$ $r_t^2$ & $Q^*(10)$ $r_t^2$', '& $T$ & ACF(1) $\\ln P_t$ & $\\hat\\rho(1)$ $r_t$ & $Q^*(10)$ $r_t$ & $\\hat\\rho(1)$ $r_t^2$ & $Q^*(10)$ $r_t^2$'),
    [b2row('BET', 'b1'), b2row('S\\&P 500', 'b2s'), b2row('EUR/RON', 'b2e')], size='scriptsize') + items(
    T('All three levels are random-walk-like; all three return series reject white noise, and their squares reject much more strongly',
      'Toate trei nivelurile sînt de tip mers aleator; toate trei seriile de randamente resping ipoteza de zgomot alb, iar pătratele lor o resping mult mai puternic'),
    T('S\\&P 500: $\\hat\\rho(1) < 0$ (a small reversal, a very liquid market); BET and EUR/RON: $\\hat\\rho(1) > 0$ (slow adjustment; EUR/RON is a managed rate)',
      'S\\&P 500: $\\hat\\rho(1) < 0$ (o mică revenire, o piață foarte lichidă); BET și EUR/RON: $\\hat\\rho(1) > 0$ (ajustare lentă; EUR/RON este un curs administrat)'),
    T('Interpretation: the S\\&P 500 is closest to a weak white noise in the mean; none is i.i.d., since the squares are correlated in all three',
      'Interpretare: S\\&P 500 este cel mai apropiat de un zgomot alb slab în medie; niciuna nu este i.i.d., deoarece pătratele sînt corelate în toate trei')) + qlsem(),
    'scriptsize', instructor_only=True)

D.task(T('B3: Romanian real GDP [Solved]', 'B3: PIB-ul real al României [Rezolvat]'),
       T('which transformation makes Romanian quarterly GDP closest to stationary?', 'ce transformare apropie cel mai mult de staționaritate PIB-ul trimestrial al României?'),
       T('real GDP, chain-linked volumes (2010), not seasonally adjusted, Eurostat, from 1995', 'PIB-ul real, volume înlănțuite (2010), neajustat sezonier, Eurostat, din 1995'),
       [T('Compute $\\ln Y_t$, $100\\,\\Delta\\ln Y_t$ and $100\\,\\Delta_4\\ln Y_t$.', 'Calculați $\\ln Y_t$, $100\\,\\Delta\\ln Y_t$ și $100\\,\\Delta_4\\ln Y_t$.'),
        T('Draw the three series and their ACF up to lag 12.', 'Desenați cele trei serii și ACF a fiecăreia pînă la decalajul 12.'),
        T('Report $\\hat\\rho(1)$, $\\hat\\rho(4)$ and $Q^*(8)$ for each.', 'Raportați $\\hat\\rho(1)$, $\\hat\\rho(4)$ și $Q^*(8)$ pentru fiecare.'),
        T('Interpretation: why is $\\hat\\rho(4)$ of the quarterly growth rate close to 1?',
          'Interpretare: de ce este $\\hat\\rho(4)$ al ratei trimestriale de creștere aproape de 1?')],
       T('a table of nine numbers, the chart and two sentences', 'un tabel cu nouă valori, graficul și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch1_sem_b3', h='0.40') + table(
    'lrrr', T('& $\\hat\\rho(1)$ & $\\hat\\rho(4)$ & $Q^*(8)$', '& $\\hat\\rho(1)$ & $\\hat\\rho(4)$ & $Q^*(8)$'),
    [f'{lab} & $@{{b3.{k}.r1}}$ & $@{{b3.{k}.r4}}$ & @{{b3.{k}.q}}' for lab, k in [('$\\ln Y_t$', 'lev'), ('$100\\,\\Delta\\ln Y_t$', 'd1'), ('$100\\,\\Delta_4\\ln Y_t$', 'd4')]],
    size='scriptsize') + items(
    T('$\\ln Y_t$: trend and season, peaks at lags 4 and 8; $\\Delta\\ln Y_t$: the season dominates (standard deviation @{b3.d1.sd}\\%); $\\Delta_4\\ln Y_t$: mean @{b3.d4.mean}\\%, the ACF dies out after 2--3 quarters',
      '$\\ln Y_t$: trend și sezon, vîrfuri la decalajele 4 și 8; $\\Delta\\ln Y_t$: sezonul domină (abaterea standard @{b3.d1.sd}\\%); $\\Delta_4\\ln Y_t$: media @{b3.d4.mean}\\%, ACF se stinge după 2--3 trimestre'),
    T('Interpretation: a q/q rate of unadjusted data repeats the same seasonal swing every year, hence $\\hat\\rho(4) \\approx 1$; the annual rate removes the season but reacts late to turning points and overlaps across quarters (positive $\\hat\\rho(1)$)',
      'Interpretare: rata t/t a datelor neajustate repetă în fiecare an aceeași oscilație sezonieră, de aici $\\hat\\rho(4) \\approx 1$; rata anuală elimină sezonul, dar reacționează tîrziu la punctele de întoarcere și se suprapune de la un trimestru la altul ($\\hat\\rho(1)$ pozitiv)')) + qlsem(),
    'scriptsize')

D.task(T('B4: Romanian inflation [Proposed]', 'B4: inflația în România [Propus]'),
       T('is monthly Romanian inflation since 2005 white noise around its mean?', 'este inflația lunară din România după 2005 un zgomot alb în jurul mediei?'),
       T('HICP, 2015 = 100, Eurostat, January 2005 to the last month available; model: B3', 'IAPC, 2015 = 100, Eurostat, din ianuarie 2005 pînă la ultima lună disponibilă; model: B3'),
       [T('Compute the monthly inflation $100\\,\\Delta\\ln P_t$ and the 12-month inflation $100\\,\\Delta_{12}\\ln P_t$.', 'Calculați inflația lunară $100\\,\\Delta\\ln P_t$ și inflația pe 12 luni $100\\,\\Delta_{12}\\ln P_t$.'),
        T('Draw the ACF of both up to lag 36 and report $\\hat\\rho(1)$, $\\hat\\rho(12)$ and $\\hat\\rho(24)$.', 'Desenați ACF a ambelor serii pînă la decalajul 36 și raportați $\\hat\\rho(1)$, $\\hat\\rho(12)$ și $\\hat\\rho(24)$.'),
        T('Compute Ljung--Box $Q^*(12)$ for the monthly inflation.', 'Calculați statistica Ljung--Box $Q^*(12)$ pentru inflația lunară.'),
        T('Interpretation: what does the ACF of the monthly inflation say about how fast an inflation shock fades?', 'Interpretare: ce spune ACF a inflației lunare despre cît de repede se stinge un șoc inflaționist?')],
       T('six ACF values, one statistic with its p-value, the chart and two sentences', 'șase valori ale ACF, o statistică cu valoarea p, graficul și două fraze'), size='footnotesize', nb='B4')

D.frame(T('B4: solution [Proposed]', 'B4: rezolvare [Propus]'), fig('ch1_sem_b4', h='0.46') + items(
    T('Monthly inflation ($T = @{b4.n}$, to @{b4.last}): mean @{b4.mean_m}\\% per month (about @{b4.ann}\\% per year), $\\hat\\rho(1) = @{b4.r1}$, $\\hat\\rho(12) = @{b4.r12}$, $\\hat\\rho(24) = @{b4.r24}$, band $\\pm @{b4.band}$',
      'Inflația lunară ($T = @{b4.n}$, pînă în @{b4.last}): media @{b4.mean_m}\\% pe lună (circa @{b4.ann}\\% pe an), $\\hat\\rho(1) = @{b4.r1}$, $\\hat\\rho(12) = @{b4.r12}$, $\\hat\\rho(24) = @{b4.r24}$, banda $\\pm @{b4.band}$'),
    T('$Q^*(12) = @{b4.q}$, p @{b4.p}: not white noise; 12-month inflation: $\\hat\\rho(1) = @{b4.a_r1}$, $\\hat\\rho(12) = @{b4.a_r12}$ (overlapping windows); peak @{b4.a_max}\\% in @{b4.amaxd}',
      '$Q^*(12) = @{b4.q}$, p @{b4.p}: nu este zgomot alb; inflația pe 12 luni: $\\hat\\rho(1) = @{b4.a_r1}$, $\\hat\\rho(12) = @{b4.a_r12}$ (ferestre suprapuse); vîrf de @{b4.a_max}\\% în @{b4.amaxd}'),
    T('Interpretation: the ACF stays positive for two years: inflation shocks fade slowly (persistence, inflation expectations); part of the correlation comes from shifts in the mean level, such as the 2022 surge',
      'Interpretare: ACF rămîne pozitivă timp de doi ani: șocurile inflaționiste se sting lent (persistență, așteptări inflaționiste); o parte din corelație provine din schimbările nivelului mediu, cum este creșterea din 2022')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: open questions and AI critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: has Romanian inflation become more persistent? [Proposed]', 'C1: a devenit inflația din România mai persistentă? [Propus]'),
       T('is monthly inflation more persistent after 2020 than in 2005--2019?', 'este inflația lunară mai persistentă după 2020 decît în 2005--2019?'),
       T('monthly HICP inflation of B4, split in January 2020; models: B1, B4', 'inflația lunară IAPC din B4, împărțită în ianuarie 2020; modele: B1, B4'),
       [T('Compute $\\hat\\rho(1)$ and $Q^*(12)$ in each subperiod, with the band of each.', 'Calculați $\\hat\\rho(1)$ și $Q^*(12)$ în fiecare subperioadă, cu banda corespunzătoare.'),
        T('Draw $\\hat\\rho(1)$ on rolling windows of 60 months.', 'Desenați $\\hat\\rho(1)$ pe ferestre mobile de 60 de luni.'),
        T('List the events of 2020--2024 that could change the persistence.', 'Enumerați evenimentele din 2020--2024 care ar putea schimba persistența.'),
        T('Interpretation: why is the difference of two sample autocorrelations hard to judge with 80 observations?', 'Interpretare: de ce este greu de evaluat diferența dintre două autocorelații de selecție cu 80 de observații?')],
       T('a table, the chart and a plan for a project', 'un tabel, graficul și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: reference analysis [Proposed]', 'C1: analiză de referință [Propus]'), fig('ch1_sem_c1', h='0.36') + table(
    'lrrrrr', T('Period & $T$ & mean (\\%) & $\\hat\\rho(1)$ & band & $Q^*(12)$ (p)', 'Perioada & $T$ & media (\\%) & $\\hat\\rho(1)$ & banda & $Q^*(12)$ (p)'),
    ['2005--2019 & @{c1.n0} & @{c1.mean0} & $@{c1.r1_0}$ & $\\pm @{c1.band0}$ & @{c1.q0} (@{c1.p0})',
     '2020--2026 & @{c1.n1} & @{c1.mean1} & $@{c1.r1_1}$ & $\\pm @{c1.band1}$ & @{c1.q1} (@{c1.p1})'], size='scriptsize') + items(
    T('$\\hat\\rho(1)$ rises from @{c1.r1_0} to @{c1.r1_1}; the rolling estimate moves between @{c1.roll_min} and @{c1.roll_max} (last: @{c1.roll_last})', '$\\hat\\rho(1)$ crește de la @{c1.r1_0} la @{c1.r1_1}; estimarea pe ferestre mobile variază între @{c1.roll_min} și @{c1.roll_max} (ultima: @{c1.roll_last})'),
    T('Events: the pandemic (2020), energy prices and the war in Ukraine (2022), the energy price caps and their removal (2025)', 'Evenimente: pandemia (2020), prețurile energiei și războiul din Ucraina (2022), plafonarea prețurilor la energie și eliminarea ei (2025)'),
    T('Interpretation: each $\\hat\\rho(1)$ has a standard error of about $1/\\sqrt{T}$ (0.11 after 2020); a break in the mean also raises $\\hat\\rho(1)$; a project needs a test (Chapter 3) and a model (Chapter 2)',
      'Interpretare: fiecare $\\hat\\rho(1)$ are o eroare standard de circa $1/\\sqrt{T}$ (0,11 după 2020); o ruptură în medie mărește și ea $\\hat\\rho(1)$; un proiect are nevoie de un test (Capitolul 3) și de un model (Capitolul 2)')) + qlsem(),
    'scriptsize', instructor_only=True)

D.frame(T('C2: audit an AI answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to interpret the S\\&P 500 daily data since 2000 ($T = @{c2.n}$ returns). The answer:', 'Un student a cerut unui asistent AI să interpreteze datele zilnice S\\&P 500 din 2000 ($T = @{c2.n}$ de randamente). Răspunsul:'),
    T('\\aiprompt{(a) The ACF of the log price at lag 1 is @{c2.p1}, so the log price is a stationary process with a very long memory.}', '\\aiprompt{(a) ACF a logaritmului prețului la decalajul 1 este @{c2.p1}, deci logaritmul prețului este un proces staționar cu memorie foarte lungă.}'),
    T('\\aiprompt{(b) The return autocorrelation at lag 1 is @{c2.r1}, outside the band of @{c2.band}, so a trader can forecast most of tomorrow\'s return.}', '\\aiprompt{(b) Autocorelația randamentelor la decalajul 1 este @{c2.r1}, în afara benzii de @{c2.band}, deci un trader poate prognoza cea mai mare parte a randamentului de mîine.}'),
    T('\\aiprompt{(c) Ljung-Box Q(10) of the squared returns is @{c2.q2}, so the returns themselves are autocorrelated.}', '\\aiprompt{(c) Ljung-Box Q(10) al randamentelor la pătrat este @{c2.q2}, deci randamentele însele sînt autocorelate.}'),
    T('\\aiprompt{(d) A weakly stationary process is always strictly stationary, since its mean and variance are constant.}', '\\aiprompt{(d) Un proces slab staționar este întotdeauna strict staționar, deoarece media și varianța lui sînt constante.}'),
    T('\\aiprompt{(e) Differencing a series that is already stationary does no harm: white noise stays white noise.}', '\\aiprompt{(e) Diferențierea unei serii deja staționare nu strică nimic: zgomotul alb rămîne zgomot alb.}'),
    T('\\aiprompt{(f) A random walk has Var(X(t)) = t * sigma\\textasciicircum{}2, so it is not stationary.}', '\\aiprompt{(f) Un mers aleator are Var(X(t)) = t * sigma\\textasciicircum{}2, deci nu este staționar.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: an ACF close to 1 that barely decays signals a random walk (non-stationary level); the sample ACF of a non-stationary series does not estimate any $\\rho(h)$',
      '(a) Greșit: o ACF apropiată de 1, care aproape nu scade, semnalează un mers aleator (nivel nestaționar); ACF de selecție a unei serii nestaționare nu estimează niciun $\\rho(h)$'),
    T('(b) Wrong: significant is not large: $\\hat\\rho(1)^2 = @{c2.r2}\\%$ of the variance; the band also assumes i.i.d. data, which returns are not',
      '(b) Greșit: semnificativ nu înseamnă mare: $\\hat\\rho(1)^2 = @{c2.r2}\\%$ din varianță; în plus, banda presupune date i.i.d., iar randamentele nu sînt'),
    T('(c) Wrong: correlated squares show dependence in the variance (volatility clustering), not in the returns; for the returns $Q^*(10) = @{c2.q}$',
      '(c) Greșit: pătratele corelate arată o dependență în varianță (volatility clustering), nu în randamente; pentru randamente $Q^*(10) = @{c2.q}$'),
    T('(d) Wrong: weak stationarity fixes only the first two moments; strict stationarity needs all joint distributions (the converse holds with finite variance)',
      '(d) Greșit: staționaritatea slabă fixează doar primele două momente; staționaritatea strictă cere toate distribuțiile comune (reciproca este adevărată dacă varianța este finită)'),
    T('(e) Wrong: $\\Delta\\varepsilon_t = \\varepsilon_t - \\varepsilon_{t-1}$ is an MA(1) with $\\rho(1) = -0.5$ and double variance: over-differencing',
      '(e) Greșit: $\\Delta\\varepsilon_t = \\varepsilon_t - \\varepsilon_{t-1}$ este un MA(1) cu $\\rho(1) = -0{,}5$ și varianță dublă: supradiferențiere'),
    T('(f) Correct: the variance depends on $t$', '(f) Corect: varianța depinde de $t$')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-up', 'Încheiere')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Autocovariances of linear processes follow from the covariance rules: only common shocks count', 'Autocovarianțele proceselor liniare rezultă din regulile covarianței: contează doar șocurile comune'),
    T('A random walk has a growing variance; its difference is white noise; a trend-stationary series becomes an over-differenced MA(1)', 'Un mers aleator are o varianță crescătoare; diferența lui este zgomot alb; o serie staționară în jurul trendului devine, prin diferențiere, un MA(1) supradiferențiat'),
    T('Look at the level, the changes and the squares: prices wander, returns are almost uncorrelated, squares are strongly correlated', 'Analizați nivelul, variațiile și pătratele: prețurile rătăcesc, randamentele sînt aproape necorelate, pătratele sînt puternic corelate'),
    T('For macro data, choose the transformation first (log, $\\Delta$, $\\Delta_4$, $\\Delta_{12}$), then read the ACF', 'Pentru datele macroeconomice, alegeți întîi transformarea (logaritm, $\\Delta$, $\\Delta_4$, $\\Delta_{12}$), apoi interpretați ACF'),
    T('An AI answer is a draft: check what is stationary, what ``significant\'\' means, and whether squares or levels were tested', 'Un răspuns AI este o ciornă: verificați ce este staționar, ce înseamnă „semnificativ” și dacă s-au testat pătratele sau nivelurile')))

D.frame(T('After the seminar', 'După seminar'), items(
    T('Lecture 1 develops each topic of today: stationarity, white noise and the random walk, Wold, ergodicity, ACF and PACF, the portmanteau tests, transformations',
      'Cursul 1 dezvoltă fiecare temă de azi: staționaritatea, zgomotul alb și mersul aleator, descompunerea Wold, ergodicitatea, ACF și PACF, testele portmanteau, transformările'),
    T('Try the [Proposed] tasks in the notebook', 'Încercați cerințele [Propus] în notebook'),
    T('C1 can grow into a team project: inflation persistence in Romania and in the region (Hungary, Poland, Czechia)', 'C1 poate deveni un proiect de echipă: persistența inflației în România și în regiune (Ungaria, Polonia, Cehia)'),
    T('Reading: \\refHP, Ch.~1--2; \\refFPP, Ch.~2--3; \\refBD, Ch.~1', 'Lectură: \\refHP, cap.~1--2; \\refFPP, cap.~2--3; \\refBD, cap.~1'),
    T('\\textbf{The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class}',
      '\\textbf{Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar}')))

D.references(bib(['BD', 'BP', 'FPP', 'HP', 'LB']), per=16)

if __name__ == '__main__':
    finalize(D.write(V))
