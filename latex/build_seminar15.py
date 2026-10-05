r"""
build_seminar15.py -- Seminarul 15 (Recapitulare: exercițiu pentru examen), EN + RO dintr-o singură sursă
========================================================================================================
Seminarul are loc ÎNAINTEA cursului 15: secțiunea „Noțiuni necesare azi” este o fișă compactă de formule pentru
Capitolele 0--10. Zece probleme mixte în formatul examenului: Partea A pe hîrtie, cu rezultate obținute cu software
(ARIMA, identificare, SARIMA, VAR și Granger, GARCH), Partea B pe date reale (producția industrială, BET, randamentele
la 10 ani ale României și Germaniei), Partea C o idee de proiect și critica unui răspuns AI.
[Rezolvat]: rezolvarea vizibilă pentru toți (A1, B1, modelele de urmat); [Propus]: rezolvarea doar în versiunea
profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_15/sem15_results.json (seminar15.py).
Ieșire:
  EN/Seminars/seminar15_review.tex              (+ _solutions.tex)
  RO/Seminarii/seminar15_recapitulare_ro.tex    (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_15/seminar15.py && python3 latex/build_seminar15.py && python3 latex/tsa_build.py compile 15
"""

import math
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, fig   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch15_common import REFS, T, bib, finalize, load_sem, month, neg, pv, quarter   # noqa: E402

S = load_sem()
V = Values()
D = Deck(15, 'seminar', refs=REFS)
P = V.put
CLOSE = T('\\textbf{The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class}',
          '\\textbf{Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar}')


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


def otab(spec, header, rows, size='scriptsize'):
    """Software output table, shrunk to the column width when it is wider (narrow task column of the instructor
    version), never enlarged."""
    return (f'\\begin{{center}}\n\\{size}\n\\resizebox{{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}}{{!}}{{%\n'
            f'\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n' + '\n'.join(r + ' \\\\' for r in rows)
            + '\n\\bottomrule\n\\end{tabular}}\n\\end{center}\n')


def qlsem():
    return '\\quantlet{TSA\\_ch15\\_seminar}{\\qlurl{TSA_ch15_seminar}}'


# =============================================================================
# CIFRE
# =============================================================================
A1 = S['A1']
for k, d in (('rho1', 3), ('inv_root', 3), ('f', 2), ('alpha_ses', 1), ('var_y10', 2), ('var_y100', 2), ('lo4', 2), ('hi4', 2)):
    P(f'a1.{k}', A1[k], d)
P('a1.v1', A1['var']['1'], 2)
P('a1.v4', A1['var']['4'], 2)

A2 = S['A2']
for tag in ('level', 'diff'):
    t = A2[tag]
    P(f'a2.{tag}.adf', t['adf'], 2)
    V.raw(f'a2.{tag}.p', pv(t['adf_p']))
    P(f'a2.{tag}.cv5', t['adf_cv5'], 2)
    P(f'a2.{tag}.cv1', t['adf_cv1'], 2)
    P(f'a2.{tag}.kpss', t['kpss'], 3)
    P(f'a2.{tag}.kcv', t['kpss_cv5'], 3)
for i in range(4):
    P(f'a2.r{i + 1}', A2['r'][i], 2)
    P(f'a2.p{i + 1}', A2['p'][i], 2)
V.raw('a2.T', str(A2['T']))
P('a2.band', A2['band'], 3)
P('a2.q4', A2['q4'], 2)
V.raw('a2.q4p', pv(A2['q4_p']))
P('a2.chi4', A2['chi4'], 2)
V.raw('a2.first', month(A2['first']))
V.raw('a2.last', month(A2['last']))
P('a2.lastu', A2['last_u'], 1)
T2 = A2['T']
r4 = A2['r4']
P('a2.sum', sum(r ** 2 / (T2 - h) for h, r in enumerate(r4, 1)) * 1e4, 3)

A3 = S['A3']
for k, d in (('phi', 3), ('Phi', 3), ('se_phi', 3), ('se_Phi', 3), ('z_phi', 2), ('z_Phi', 2), ('sigma2', 2), ('zT', 3), ('zT3', 3), ('zT4', 3),
             ('aT', 2), ('z_next', 3), ('a_next', 2)):
    P(f'a3.{k}', A3[k], d)
V.raw('a3.p_phi', pv(A3['p_phi']))
V.raw('a3.p_Phi', pv(A3['p_Phi']))
P('a3.lb', A3['lb8']['q'], 2)
V.raw('a3.lbdf', str(A3['lb8']['df']))
V.raw('a3.lbp', pv(A3['lb8']['p']))
P('a3.chi6', stats.chi2.ppf(0.95, A3['lb8']['df']), 2)
V.raw('a3.n', str(A3['n']))
V.raw('a3.first', quarter(A3['first'][:4] + '-' + f"{3 * int(A3['first'][-1]) - 2:02d}-01"))
V.raw('a3.last', quarter(A3['last'][:4] + '-' + f"{3 * int(A3['last'][-1]) - 2:02d}-01"))
TAIL = A3['tail']
for i, (q, v) in enumerate(TAIL):
    V.raw(f'a3.q{i}', quarter(q[:4] + '-' + f"{3 * int(q[-1]) - 2:02d}-01"))
    P(f'a3.v{i}', v, 3)
V.raw('a3.next', quarter(A3['next'][:4] + '-' + f"{3 * int(A3['next'][-1]) - 2:02d}-01"))
P('a3.t1', A3['phi'] * A3['zT'], 3)
P('a3.t2', A3['Phi'] * A3['zT3'], 3)
P('a3.t3', -A3['phi'] * A3['Phi'] * A3['zT4'], 3)
P('a3.pP', A3['phi'] * A3['Phi'], 3)

A4 = S['A4']
B_I, T_I = A4['params']['i'], A4['t']['i']
B_P, T_P = A4['params']['pi'], A4['t']['pi']
for k in ('const', 'L1.pi', 'L1.i', 'L2.pi', 'L2.i'):
    P(f'a4.bi.{k}', B_I[k], 3)
    P(f'a4.ti.{k}', T_I[k], 2)
    P(f'a4.bp.{k}', B_P[k], 3)
    P(f'a4.tp.{k}', T_P[k], 2)
G1, G2 = A4['granger']['pi->i'], A4['granger']['i->pi']
for tag, g in (('g1', G1), ('g2', G2)):
    P(f'a4.{tag}.rr', g['rss_r'], 2)
    P(f'a4.{tag}.ru', g['rss_u'], 2)
    P(f'a4.{tag}.F', g['F'], 2)
    V.raw(f'a4.{tag}.p', pv(g['p']))
    V.raw(f'a4.{tag}.df2', str(g['df2']))
    P(f'a4.{tag}.crit', g['crit5'], 2)
P('a4.num', A4['lr_num'], 3)
P('a4.den', A4['lr_den'], 3)
P('a4.lr', A4['lr'], 2)
V.raw('a4.T', str(A4['T']))
V.raw('a4.first', month(A4['first']))
V.raw('a4.last', month(A4['last']))
P('a4.r2i', A4['r2']['i'], 2)
P('a4.r2p', A4['r2']['pi'], 2)

A5 = S['A5']
pa = A5['params']
nu5 = round(pa['nu'])
q5 = stats.t.ppf(0.01, nu5) * math.sqrt((nu5 - 2) / nu5)
s2n = pa['omega'] + pa['alpha[1]'] * A5['eT'] ** 2 + pa['beta[1]'] * A5['s2T']
for k, kk, d in (('mu', 'mu', 3), ('omega', 'om', 4), ('alpha[1]', 'al', 3), ('beta[1]', 'be', 3), ('nu', 'nu', 2)):
    P(f'a5.{kk}', pa[k], d)
    P(f'a5.se.{kk}', A5['se'][k], 3 if kk != 'om' else 4)
P('a5.s2T', A5['s2T'], 3)
P('a5.eT', A5['eT'], 3)
P('a5.e2', A5['eT'] ** 2, 3)
P('a5.ae2', pa['alpha[1]'] * A5['eT'] ** 2, 3)
P('a5.bs2', pa['beta[1]'] * A5['s2T'], 3)
P('a5.s2n', s2n, 3)
P('a5.sn', math.sqrt(s2n), 3)
P('a5.pers', A5['pers'], 3)
P('a5.hl', A5['hl'], 0)
P('a5.lv', A5['lr_var'], 2)
P('a5.ppy', A5['ppy'], 0)
P('a5.ann', A5['lr_vol'], 1)
V.raw('a5.nu5', str(nu5))
P('a5.tq', stats.t.ppf(0.01, nu5), 3)
P('a5.sc', math.sqrt((nu5 - 2) / nu5), 3)
P('a5.q', q5, 3)
P('a5.var', -(pa['mu'] + math.sqrt(s2n) * q5), 2)
P('a5.varn', -(pa['mu'] + math.sqrt(s2n) * stats.norm.ppf(0.01)), 2)
V.raw('a5.n', f"{A5['n']:,}".replace(',', '\\,'))
V.raw('a5.last', month(A5['last']))

B1 = S['B1']
for k in ('y', 'd1', 'z'):
    t = B1['tests'][k]
    P(f'b1.{k}.adf', t['adf'], 2)
    V.raw(f'b1.{k}.p', pv(t['adf_p']))
    P(f'b1.{k}.kpss', t['kpss'], 3)
    P(f'b1.{k}.kcv', t['kpss_cv5'], 3)
for j in (1, 2, 12, 13):
    P(f'b1.r{j}', B1['r'][j - 1], 2)
P('b1.band', B1['band'], 3)
for nm in ('airline', 'alt'):
    m = B1['models'][nm]
    for k in ('aicc', 'bic', 'rmse', 'mae'):
        P(f'b1.{nm}.{k}', m[k], 1 if k in ('aicc', 'bic') else 2)
    V.raw(f'b1.{nm}.lb12', pv(m['lb']['12']['p']))
    V.raw(f'b1.{nm}.lb24', pv(m['lb']['24']['p']))
    for k, v in m['params'].items():
        P(f'b1.{nm}.{k}', v, 3)
        if k in m['se']:
            P(f'b1.{nm}.se.{k}', m['se'][k], 3)
P('b1.sn.rmse', B1['sn']['rmse'], 2)
P('b1.sn.mae', B1['sn']['mae'], 2)
P('b1.covid', B1['covid'], 1)
V.raw('b1.first', month(B1['first']))
V.raw('b1.last', month(B1['last']))
V.raw('b1.tf', month(B1['test_first']))
V.raw('b1.n', str(B1['n']))

B2 = S['B2']
for k, kk, d in (('mu', 'mu', 3), ('omega', 'om', 3), ('alpha[1]', 'al', 3), ('beta[1]', 'be', 3), ('nu', 'nu', 2)):
    P(f'b2.{kk}', B2['params'][k], d)
for k, d in (('lm', 1), ('pers', 3), ('hl', 1), ('lr_vol', 1), ('vol_now', 1), ('vol_sample', 1), ('var1', 2), ('var1_n', 2)):
    P(f'b2.{k}', B2[k], d)
V.raw('b2.lmp', pv(B2['lm_p']))
V.raw('b2.n', f"{B2['n']:,}".replace(',', '\\,'))
V.raw('b2.last', month(B2['last']))

B3 = S['B3']
for k in ('ro', 'de', 'dro', 'dde'):
    P(f'b3.{k}.adf', B3['adf'][k]['adf'], 2)
    V.raw(f'b3.{k}.p', pv(B3['adf'][k]['adf_p']))
P('b3.eg', B3['eg']['stat'], 2)
V.raw('b3.egp', pv(B3['eg']['p']))
P('b3.egcv', B3['eg']['cv5'], 2)
P('b3.beta', B3['beta'], 2)
P('b3.const', B3['const'], 2)
E = B3['ecm']
P('b3.g', E['gamma'], 3)
P('b3.tg', E['t_gamma'], 2)
P('b3.d', E['delta'], 3)
P('b3.td', E['t_delta'], 2)
P('b3.half', E['half'], 0)
P('b3.gde', E['gamma_de'], 3)
P('b3.tgde', E['t_gamma_de'], 2)
J = B3['joh']
for i in range(2):
    P(f'b3.tr{i}', J['trace'][i], 2)
    P(f'b3.cv{i}', J['cv5'][i], 2)
    P(f'b3.mx{i}', J['maxeig'][i], 2)
    P(f'b3.cvm{i}', J['cv5_max'][i], 2)
P('b3.spread', B3['spread_mean'], 2)
P('b3.ro', B3['ro_last'], 2)
P('b3.de', B3['de_last'], 2)
V.raw('b3.n', str(B3['n']))
V.raw('b3.first', month(B3['first']))
V.raw('b3.last', month(B3['last']))

C2 = S['C2']
P('c2.hlw', C2['hl_wrong'], 0)
P('c2.hl', C2['hl'], 0)
P('c2.var', C2['var1'], 2)
P('c2.adfp', C2['adf_p_level'], 2)
P('c2.kpss', C2['kpss_level'], 2)
P('c2.gp', C2['g_p'], 3)
P('c2.pers', A5['pers'], 3)
neg(V)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: can you solve, on paper and on data, an exam problem from any core chapter?',
       '\\textbf{Întrebarea}: puteți rezolva, pe hîrtie și pe date, o problemă de examen din orice capitol de bază?'),
     [T('this seminar comes \\textbf{before} Lecture 15: the section ``What you need today\'\' is a formula sheet for Chapters 0--10',
        'seminarul are loc \\textbf{înaintea} Cursului 15: secțiunea „Noțiuni necesare azi” este o fișă de formule pentru Capitolele 0--10')]),
    (T('Route', 'Traseul'),
     [T('Part A: five exam-type problems on paper, with software output: ARIMA, identification, SARIMA, VAR and Granger, GARCH', 'Partea A: cinci probleme de tip examen, pe hîrtie, cu rezultate obținute cu software: ARIMA, identificare, SARIMA, VAR și Granger, GARCH'),
      T('Part B: three data tasks (industrial production, the BET, two 10-year yields), each with an interpretation question', 'Partea B: trei cerințe pe date reale (producția industrială, BET, două randamente la 10 ani), fiecare cu o întrebare de interpretare'),
      T('Part C: a project idea and an AI answer to audit', 'Partea C: o idee de proiect și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook'),
    CLOSE))

TB = '>{\\raggedright\\arraybackslash}'
SOL, PRO = T('Solved', 'Rezolvat'), T('Proposed', 'Propus')
D.frame(T('Exercise map', 'Harta exercițiilor'), table(
    TB + 'p{0.8cm}' + TB + 'p{6.9cm}' + TB + 'p{1.4cm}' + TB + 'p{2.2cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Topic}', '\\textbf{Tema}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & ' + T('\\textbf{Chapters}', '\\textbf{Capitole}'),
    ['A1 & ' + T('an ARIMA(0,1,1) by hand: moments, ACF, forecasts, exponential smoothing', 'un ARIMA(0,1,1) calculat de mînă: momente, ACF, prognoze, netezire exponențială') + f' & {SOL} & 0, 1, 3',
     'A2 & ' + T('identifying the Romanian unemployment rate: ADF, KPSS, correlogram, Ljung--Box', 'identificarea ratei șomajului din România: ADF, KPSS, corelogramă, Ljung--Box') + f' & {PRO} & 1--3',
     'A3 & ' + T('SARIMA output for Romanian GDP and a forecast by hand', 'rezultate SARIMA pentru PIB-ul României și o prognoză de mînă') + f' & {PRO} & 4',
     'A4 & ' + T('VAR output, Granger $F$ from two RSS, a long-run effect', 'rezultate VAR, testul Granger $F$ din două RSS, un efect pe termen lung') + f' & {PRO} & 6',
     'A5 & ' + T('GARCH(1,1)-$t$ output for the DAX and VaR 1\\%', 'rezultate GARCH(1,1)-$t$ pentru DAX și VaR 1\\%') + f' & {PRO} & 5',
     'B1 & ' + T('Box--Jenkins for Romanian industrial production', 'Box--Jenkins pentru producția industrială a României') + f' & {SOL} & 3, 4',
     'B2 & ' + T('volatility of the BET since 2015', 'volatilitatea BET din 2015') + f' & {PRO} & 5',
     'B3 & ' + T('the Romanian and German 10-year yields: cointegration', 'randamentele la 10 ani ale României și Germaniei: cointegrare') + f' & {PRO} & 3, 7',
     'C1, C2 & ' + T('a forecast competition for inflation; audit an AI answer', 'o competiție de prognoză pentru inflație; verificarea unui răspuns AI') + f' & {PRO} & 0--10'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați dumneavoastră, după model')))

D.frame(T('Data used', 'Datele folosite'), table(
    '>{\\raggedright\\arraybackslash}p{4.4cm}>{\\raggedright\\arraybackslash}p{3.2cm}>{\\raggedright\\arraybackslash}p{2.6cm}l', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Frequency}', '\\textbf{Frecvența}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    [T('unemployment rate, Romania (SA)', 'rata șomajului, România (SA)') + ' & Eurostat (une\\_rt\\_m) & ' + T('monthly', 'lunar') + ' & 2005--2026',
     T('real GDP, Romania (not adjusted)', 'PIB real, România (neajustat)') + ' & Eurostat (namq\\_10\\_gdp) & ' + T('quarterly', 'trimestrial') + ' & 2000--2026',
     T('HICP inflation and ROBOR 3M, Romania', 'inflația IAPC și ROBOR 3M, România') + ' & Eurostat & ' + T('monthly', 'lunar') + ' & 2005--2026',
     T('industrial production, Romania (calendar adjusted)', 'producția industrială, România (ajustată calendaristic)') + ' & Eurostat (sts\\_inpr\\_m) & ' + T('monthly', 'lunar') + ' & 2005--2026',
     'DAX, BET & EODHD & ' + T('daily', 'zilnic') + ' & 2000--2026',
     T('10-year yields, Romania and Germany', 'randamente la 10 ani, România și Germania') + ' & EODHD & ' + T('monthly means of daily data', 'medii lunare ale datelor zilnice') + ' & 2008--2026'],
    size='footnotesize') + items(
    T('Part A needs only a calculator: the output tables are on the slides; Part B uses the notebook', 'Partea A are nevoie doar de un calculator: tabelele cu rezultate sînt pe slide-uri; Partea B folosește notebook-ul'),
    T('Growth rates in \\%: $100\\Delta\\ln y_t$; annual rates: $100\\Delta_{12}\\ln y_t$ (monthly), $100\\Delta_4\\ln y_t$ (quarterly); returns: $100\\Delta\\ln P_t$', 'Ratele de creștere în \\%: $100\\Delta\\ln y_t$; ratele anuale: $100\\Delta_{12}\\ln y_t$ (lunar), $100\\Delta_4\\ln y_t$ (trimestrial); randamentele: $100\\Delta\\ln P_t$')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What you need today', 'Noțiuni necesare azi')
FH = T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}')


def sheet(title, rows):
    D.frame(title, '{\\renewcommand{\\arraystretch}{1.3}' + table('l>{\\raggedright\\arraybackslash}p{9.4cm}', FH, rows, size='footnotesize') + '}')


sheet(T('What you need today (1/4): description, stationarity, ARMA', 'Noțiuni necesare azi (1/4): descriere, staționaritate, ARMA'), [
    'SES, MASE & $\\ell_t = \\alpha y_t + (1 - \\alpha)\\ell_{t-1}$; \\ MASE $=$ MAE / MAE ' + T('(seasonal naive, in sample)', '(naiv sezonier, în eșantion)'),
    T('ACF, band', 'ACF, banda') + ' & $\\hat\\rho(h) = \\hat\\gamma(h)/\\hat\\gamma(0)$, \\ $\\pm 1.96/\\sqrt{T}$',
    'Ljung--Box & $Q^*(m) = T(T + 2)\\sum_{h=1}^{m}\\hat\\rho(h)^2/(T - h) \\sim \\chi^2(m - k)$, ' + T('$k$ = ARMA parameters', '$k$ = parametri ARMA'),
    'AR(1), MA(1) & $\\rho(h) = \\phi^h$; \\ $\\rho(1) = \\theta/(1 + \\theta^2)$, $\\rho(h) = 0$ ' + T('for', 'pentru') + ' $h \\ge 2$',
    T('Identification', 'Identificarea') + ' & ' + T('ACF cuts off: MA; PACF cuts off: AR; both decay: ARMA', 'ACF se anulează: MA; PACF se anulează: AR; ambele descresc: ARMA'),
    T('Criteria', 'Criterii') + ' & AIC $= -2\\ln L + 2k$, \\ BIC $= -2\\ln L + k\\ln T$; ' + T('the smallest wins', 'cîștigă valoarea cea mai mică'),
    T('Forecast interval', 'Intervalul de prognoză') + ' & $\\hat y_{T+h} \\pm 1.96\\,\\sigma\\sqrt{\\sum_{j<h}\\psi_j^2}$'])

sheet(T('What you need today (2/4): unit roots, ARIMA, SARIMA, evaluation', 'Noțiuni necesare azi (2/4): rădăcini unitare, ARIMA, SARIMA, evaluare'), [
    'ADF & $\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum_j\\delta_j\\Delta y_{t-j} + \\varepsilon_t$; \\ $H_0$: $\\gamma = 0$ ' + T('(unit root)', '(rădăcină unitară)'),
    T('ADF, 5\\% critical values', 'ADF, valori critice la 5\\%') + ' & ' + T('about $-2.87$ (constant), $-3.43$ (constant and trend); reject if $\\tau$ is smaller', 'circa $-2.87$ (constantă), $-3.43$ (constantă și trend); respingem dacă $\\tau$ este mai mic'),
    'KPSS & $H_0$: ' + T('stationarity; 5\\% critical values 0.463 (constant), 0.146 (trend); reject if larger', 'staționaritate; valori critice la 5\\%: 0,463 (constantă), 0,146 (trend); respingem dacă este mai mare'),
    'ARIMA(0,1,1) & $\\Delta y_t = \\varepsilon_t + \\theta\\varepsilon_{t-1}$; \\ $\\hat y_{T+h} = y_T + \\theta\\varepsilon_T$, \\ $\\alpha_{SES} = 1 + \\theta$',
    T('Its error variance', 'Varianța erorii') + ' & $\\sigma^2[1 + (h - 1)(1 + \\theta)^2]$',
    'AR(1)$\\times$SAR(1)$_s$ & $z_t = \\phi z_{t-1} + \\Phi z_{t-s} - \\phi\\Phi z_{t-s-1} + \\varepsilon_t$',
    T('Annual rate from $z$', 'Rata anuală din $z$') + ' & $z_t = \\Delta\\Delta_s\\ln y_t$: \\ $\\Delta_s\\ln y_{T+1} = \\Delta_s\\ln y_T + \\hat z_{T+1}$',
    'Diebold--Mariano & $d_t = e_{1t}^2 - e_{2t}^2$, \\ DM $= \\bar d/(s_d/\\sqrt{n})$; ' + T('reject equal accuracy if $|\\mathrm{DM}| > t_{0.975}(n - 1)$', 'respingem acuratețea egală dacă $|\\mathrm{DM}| > t_{0.975}(n - 1)$')])

sheet(T('What you need today (3/4): volatility and several series', 'Noțiuni necesare azi (3/4): volatilitate și mai multe serii'), [
    'GARCH(1,1) & $\\sigma_{t+1}^2 = \\omega + \\alpha\\varepsilon_t^2 + \\beta\\sigma_t^2$, \\ $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$, \\ $h_{1/2} = \\ln 0.5/\\ln(\\alpha + \\beta)$',
    T('Annual volatility', 'Volatilitatea anuală') + ' & $\\sqrt{q\\,\\bar\\sigma^2}$, ' + T('$q$ = trading days a year', '$q$ = zile de tranzacționare pe an'),
    'VaR 1\\% & $-(\\mu + \\sigma_{t+1}q_{0.01}(z))$; \\ Student-$t_\\nu$: $q = t_\\nu^{-1}(0.01)\\sqrt{(\\nu - 2)/\\nu}$; \\ Normal: $-2.326$',
    'VAR$(p)$ & $\\mathbf y_t = \\mathbf c + \\mathbf A_1\\mathbf y_{t-1} + \\dots + \\mathbf A_p\\mathbf y_{t-p} + \\mathbf u_t$, ' + T('OLS equation by equation', 'OLS ecuație cu ecuație'),
    'Granger $F$ & $[(RSS_R - RSS_U)/p]\\,/\\,[RSS_U/(T - Kp - 1)] \\sim F(p, T - Kp - 1)$',
    T('Long-run effect', 'Efectul pe termen lung') + ' & $\\partial y/\\partial x = \\sum_j b_{yx,j}\\,/\\,(1 - \\sum_j b_{yy,j})$',
    'Engle--Granger, ECM & ' + T('ADF on OLS residuals, critical value about $-3.34$ (two variables); $h_{1/2} = \\ln 0.5/\\ln(1 + \\gamma)$', 'ADF pe reziduurile OLS, valoare critică circa $-3.34$ (două variabile); $h_{1/2} = \\ln 0.5/\\ln(1 + \\gamma)$'),
    'Johansen & ' + T('trace $-T\\sum_{i>r}\\ln(1 - \\hat\\lambda_i)$; test $r = 0, 1, \\dots$ in turn; stop at the first non-rejection', 'urma $-T\\sum_{i>r}\\ln(1 - \\hat\\lambda_i)$; testăm pe rînd $r = 0, 1, \\dots$; ne oprim la prima nerespingere')])

sheet(T('What you need today (4/4): extensions', 'Noțiuni necesare azi (4/4): extensii'), [
    'ARFIMA$(0,d,0)$ & $\\rho(1) = d/(1 - d)$, \\ $H = d + 1/2$; ' + T('stationary for $d < 1/2$, mean-reverting for $d < 1$', 'staționar pentru $d < 1/2$, cu revenire la medie pentru $d < 1$'),
    'Local level & $F_t = P_t + \\sigma^2_\\varepsilon$, \\ $K_t = P_t/F_t$, \\ $a_{t|t} = a_t + K_t(y_t - a_t)$, \\ $P_{t+1} = P_t(1 - K_t) + \\sigma^2_\\eta$',
    T('Steady state', 'Starea de echilibru') + ' & $\\bar P/\\sigma^2_\\varepsilon = (q + \\sqrt{q^2 + 4q})/2$, \\ $\\bar K = \\alpha_{SES}$, \\ $q = \\sigma^2_\\eta/\\sigma^2_\\varepsilon$',
    'Markov switching & $E(D_i) = 1/(1 - p_{ii})$, \\ $\\pi_1 = (1 - p_{22})/(2 - p_{11} - p_{22})$',
    T('Machine learning', 'Învățare automată') + ' & ' + T('walk-forward validation only; compare with a simple benchmark', 'doar validare walk-forward; comparație cu un reper simplu'),
    T('Values', 'Valori') + ' & $\\chi^2_{0.95}$: ' + T('3.84 (1), 5.99 (2), 9.49 (4), 12.59 (6), 18.31 (10)', '3,84 (1); 5,99 (2); 9,49 (4); 12,59 (6); 18,31 (10)') + '; \\ $F_{0.95}(2, 250) = 3.03$'])

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: exam problems on paper', 'Partea A: probleme de examen pe hîrtie')

D.solved(T('A1: an ARIMA(0,1,1) by hand', 'A1: un ARIMA(0,1,1) calculat de mînă'),
         items(T('$\\Delta y_t = \\varepsilon_t - 0.6\\,\\varepsilon_{t-1}$, $\\varepsilon_t \\sim WN(0, 1)$; $y_0$ given, $\\varepsilon_0 = 0$; today $y_T = 100$, $\\hat\\varepsilon_T = 1.5$.', '$\\Delta y_t = \\varepsilon_t - 0{,}6\\,\\varepsilon_{t-1}$, $\\varepsilon_t \\sim WN(0, 1)$; $y_0$ dat, $\\varepsilon_0 = 0$; azi $y_T = 100$, $\\hat\\varepsilon_T = 1{,}5$.'),
               T('1. Show that $\\Delta y_t$ is stationary and invertible, and compute $\\rho_{\\Delta y}(1)$.', '1. Arătați că $\\Delta y_t$ este staționar și invertibil și calculați $\\rho_{\\Delta y}(1)$.'),
               T('2. Compute $\\mathrm{Var}(y_t)$ for $t = 10$ and $t = 100$ and say whether $y_t$ is stationary.', '2. Calculați $\\mathrm{Var}(y_t)$ pentru $t = 10$ și $t = 100$ și precizați dacă $y_t$ este staționar.'),
               T('3. Forecast $y_{T+1}$ and $y_{T+4}$; give the 95\\% interval for $h = 4$.', '3. Prognozați $y_{T+1}$ și $y_{T+4}$; dați intervalul de 95\\% pentru $h = 4$.'),
               T('4. Give the weight of the equivalent exponential smoothing.', '4. Dați ponderea netezirii exponențiale echivalente.'),
               T('Report: five numbers, an interval and one sentence.', 'Raportați: cinci valori, un interval și o frază.')),
         items(T('1. An MA(1) is always stationary; root $1/0.6 = @{a1.inv_root} > 1$: invertible; $\\rho(1) = -0.6/1.36 = @{a1.rho1}$', '1. Un MA(1) este întotdeauna staționar; rădăcina $1/0{,}6 = @{a1.inv_root} > 1$: invertibil; $\\rho(1) = -0{,}6/1{,}36 = @{a1.rho1}$'),
               T('2. $y_t = y_0 + \\varepsilon_t + 0.4\\sum_{j<t}\\varepsilon_j$: $\\mathrm{Var} = 1 + (t - 1)0.16$: $@{a1.var_y10}$ and $@{a1.var_y100}$: grows with $t$, not stationary', '2. $y_t = y_0 + \\varepsilon_t + 0{,}4\\sum_{j<t}\\varepsilon_j$: $\\mathrm{Var} = 1 + (t - 1)0{,}16$: $@{a1.var_y10}$ și $@{a1.var_y100}$: crește cu $t$, nestaționar'),
               T('3. $\\hat y_{T+h} = 100 - 0.6 \\times 1.5 = @{a1.f}$ for every $h$; variances $@{a1.v1}$ and $@{a1.v4}$; interval $[@{a1.lo4}, @{a1.hi4}]$', '3. $\\hat y_{T+h} = 100 - 0{,}6 \\times 1{,}5 = @{a1.f}$ pentru orice $h$; varianțe $@{a1.v1}$ și $@{a1.v4}$; intervalul $[@{a1.lo4}; @{a1.hi4}]$'),
               T('4. $\\alpha = 1 + \\theta = @{a1.alpha_ses}$', '4. $\\alpha = 1 + \\theta = @{a1.alpha_ses}$'),
               T('Interpretation: the forecast is flat, like that of exponential smoothing, but its interval keeps widening: a unit root', 'Interpretare: prognoza este orizontală, ca a netezirii exponențiale, dar intervalul ei se lărgește continuu: o rădăcină unitară')),
         size='scriptsize')

A2OUT = otab('lcccc', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ADF $\\tau$ & $p$ & ' + T('ADF crit. 5\\%', 'ADF val. critică 5\\%') + ' & KPSS (' + T('crit. 0.463', 'val. critică 0,463') + ')',
              ['$u_t$ & $@{a2.level.adf}$ & @{a2.level.p} & $@{a2.level.cv5}$ & @{a2.level.kpss}',
               '$\\Delta u_t$ & $@{a2.diff.adf}$ & @{a2.diff.p} & $@{a2.diff.cv5}$ & @{a2.diff.kpss}'], size='scriptsize')
D.proposed(T('A2: identifying the Romanian unemployment rate', 'A2: identificarea ratei șomajului din România'),
           A2OUT + items(T('Unemployment rate $u_t$ (SA, \\%), @{a2.first} -- @{a2.last}; both tests with a constant. $\\Delta u_t$ ($T = @{a2.T}$): ACF $@{a2.r1}$, $@{a2.r2}$, $@{a2.r3}$, $@{a2.r4}$; PACF $@{a2.p1}$, $@{a2.p2}$, $@{a2.p3}$, $@{a2.p4}$ (lags 1--4). Model: Seminar 3, A1; Seminar 2, A6.',
                           'Rata șomajului $u_t$ (SA, \\%), @{a2.first} -- @{a2.last}; ambele teste cu constantă. $\\Delta u_t$ ($T = @{a2.T}$): ACF $@{a2.r1}$; $@{a2.r2}$; $@{a2.r3}$; $@{a2.r4}$; PACF $@{a2.p1}$; $@{a2.p2}$; $@{a2.p3}$; $@{a2.p4}$ (decalajele 1--4). Model: Seminarul 3, A1; Seminarul 2, A6.'),
                         T('1. State $H_0$ of each test and decide the order of integration of $u_t$.', '1. Precizați $H_0$ pentru fiecare test și decideți ordinul de integrare al lui $u_t$.'),
                         T('2. Compute the band $\\pm 1.96/\\sqrt{T}$ and Ljung--Box $Q(4)$ from the four ACF values; decide at 5\\%.', '2. Calculați banda $\\pm 1{,}96/\\sqrt{T}$ și Ljung--Box $Q(4)$ din cele patru valori ACF; decideți la 5\\%.'),
                         T('3. Propose an ARIMA model for $u_t$.', '3. Propuneți un model ARIMA pentru $u_t$.'),
                         T('Report: two decisions, two numbers, a model.', 'Raportați: două decizii, două valori, un model.')),
           items(T('1. ADF ($H_0$ unit root): $u_t$ not rejected ($p = @{a2.level.p}$), $\\Delta u_t$ rejected; KPSS ($H_0$ stationarity): $@{a2.level.kpss} > 0.463$ rejects for $u_t$, $@{a2.diff.kpss}$ does not for $\\Delta u_t$: $u_t \\sim I(1)$', '1. ADF ($H_0$ rădăcină unitară): pentru $u_t$ nu se respinge ($p = @{a2.level.p}$), pentru $\\Delta u_t$ se respinge; KPSS ($H_0$ staționaritate): $@{a2.level.kpss} > 0{,}463$ respinge pentru $u_t$, $@{a2.diff.kpss}$ nu respinge pentru $\\Delta u_t$: $u_t \\sim I(1)$'),
                 T('2. Band $\\pm @{a2.band}$; $Q(4) = T(T + 2)\\sum_h\\hat\\rho_h^2/(T - h) = @{a2.q4} > @{a2.chi4}$ ($p = @{a2.q4p}$): $\\Delta u_t$ is not white noise', '2. Banda $\\pm @{a2.band}$; $Q(4) = T(T + 2)\\sum_h\\hat\\rho_h^2/(T - h) = @{a2.q4} > @{a2.chi4}$ ($p = @{a2.q4p}$): $\\Delta u_t$ nu este zgomot alb'),
                 T('3. Only lag 1 is clearly outside the band in both ACF and PACF, with a negative sign: ARIMA(1,1,0) or ARIMA(0,1,1); choose by BIC and check the residuals', '3. Doar decalajul 1 iese clar din bandă, atît în ACF, cît și în PACF, cu semn negativ: ARIMA(1,1,0) sau ARIMA(0,1,1); alegem după BIC și verificăm reziduurile')),
           size='scriptsize')

A3OUT = otab('lcccc', T('\\textbf{Dependent: $z_t$}', '\\textbf{Variabila dependentă: $z_t$}') + ' & ' + T('Coef.', 'Coef.') + ' & SE & $z$ & $p$',
              ['AR(1) & $@{a3.phi}$ & $@{a3.se_phi}$ & $@{a3.z_phi}$ & @{a3.p_phi}',
               'SAR(4) & $@{a3.Phi}$ & $@{a3.se_Phi}$ & $@{a3.z_Phi}$ & @{a3.p_Phi}',
               'SIGMASQ & $@{a3.sigma2}$ & & &', 'Ljung--Box $Q(8)$ & $@{a3.lb}$ & & & @{a3.lbp}'], size='scriptsize')
D.proposed(T('A3: SARIMA output for Romanian GDP', 'A3: rezultate SARIMA pentru PIB-ul României'),
           A3OUT + items(T('$z_t = \\Delta\\Delta_4\\,100\\ln \\mathrm{GDP}_t$, real GDP not seasonally adjusted, @{a3.first} -- @{a3.last} ($T = @{a3.n}$). Last values of $z$: $@{a3.v1}$ (@{a3.q1}), $@{a3.v2}$ (@{a3.q2}), $@{a3.v3}$ (@{a3.q3}), $@{a3.v4}$ (@{a3.q4}), $@{a3.v5}$ (@{a3.q5}); annual growth $\\Delta_4\\,100\\ln \\mathrm{GDP}$ in @{a3.q5}: $@{a3.aT}\\%$. Model: Seminar 4, A4.',
                           '$z_t = \\Delta\\Delta_4\\,100\\ln \\mathrm{PIB}_t$, PIB real neajustat sezonier, @{a3.first} -- @{a3.last} ($T = @{a3.n}$). Ultimele valori ale lui $z$: $@{a3.v1}$ (@{a3.q1}); $@{a3.v2}$ (@{a3.q2}); $@{a3.v3}$ (@{a3.q3}); $@{a3.v4}$ (@{a3.q4}); $@{a3.v5}$ (@{a3.q5}); creșterea anuală $\\Delta_4\\,100\\ln \\mathrm{PIB}$ în @{a3.q5}: $@{a3.aT}\\%$. Model: Seminarul 4, A4.'),
                         T('1. Write the model equation and say which coefficients are significant.', '1. Scrieți ecuația modelului și precizați ce coeficienți sînt semnificativi.'),
                         T('2. Test whether the residuals are white noise ($\\chi^2_{0.95}(@{a3.lbdf}) = @{a3.chi6}$).', '2. Testați dacă reziduurile sînt zgomot alb ($\\chi^2_{0,95}(@{a3.lbdf}) = @{a3.chi6}$).'),
                         T('3. Forecast $z$ and the annual growth rate for @{a3.next}.', '3. Prognozați $z$ și rata anuală de creștere pentru @{a3.next}.')),
           items(T('1. $z_t = \\phi z_{t-1} + \\Phi z_{t-4} - \\phi\\Phi z_{t-5} + \\varepsilon_t$; only $\\hat\\Phi = @{a3.Phi}$ is significant ($p$ @{a3.p_Phi}); $\\hat\\phi$ has $p = @{a3.p_phi}$', '1. $z_t = \\phi z_{t-1} + \\Phi z_{t-4} - \\phi\\Phi z_{t-5} + \\varepsilon_t$; doar $\\hat\\Phi = @{a3.Phi}$ este semnificativ ($p$ @{a3.p_Phi}); $\\hat\\phi$ are $p = @{a3.p_phi}$'),
                 T('2. $Q(8) = @{a3.lb} > @{a3.chi6}$ on $8 - 2 = @{a3.lbdf}$ df: white noise rejected; a seasonal MA term would fit better', '2. $Q(8) = @{a3.lb} > @{a3.chi6}$ cu $8 - 2 = @{a3.lbdf}$ grade de libertate: ipoteza de zgomot alb se respinge; un termen MA sezonier s-ar potrivi mai bine'),
                 T('3. $\\hat z = @{a3.phi}(@{a3.zT}) + (@{a3.Phi})(@{a3.zT3}) - (@{a3.pP})(@{a3.zT4}) = @{a3.z_next}$; growth $@{a3.aT} + @{a3.z_next} = @{a3.a_next}\\%$', '3. $\\hat z = @{a3.phi}(@{a3.zT}) + (@{a3.Phi})(@{a3.zT3}) - (@{a3.pP})(@{a3.zT4}) = @{a3.z_next}$; creșterea $@{a3.aT} + @{a3.z_next} = @{a3.a_next}\\%$')),
           size='scriptsize')

A4OUT = otab('lcc', '& ' + T('$\\pi_t$ equation', 'ecuația $\\pi_t$') + ' & ' + T('$i_t$ equation', 'ecuația $i_t$'),
              ['$\\pi_{t-1}$ & $@{a4.bp.L1.pi}$ [$@{a4.tp.L1.pi}$] & $@{a4.bi.L1.pi}$ [$@{a4.ti.L1.pi}$]',
               '$\\pi_{t-2}$ & $@{a4.bp.L2.pi}$ [$@{a4.tp.L2.pi}$] & $@{a4.bi.L2.pi}$ [$@{a4.ti.L2.pi}$]',
               '$i_{t-1}$ & $@{a4.bp.L1.i}$ [$@{a4.tp.L1.i}$] & $@{a4.bi.L1.i}$ [$@{a4.ti.L1.i}$]',
               '$i_{t-2}$ & $@{a4.bp.L2.i}$ [$@{a4.tp.L2.i}$] & $@{a4.bi.L2.i}$ [$@{a4.ti.L2.i}$]',
               'const. & $@{a4.bp.const}$ [$@{a4.tp.const}$] & $@{a4.bi.const}$ [$@{a4.ti.const}$]',
               '$R^2$ & @{a4.r2p} & @{a4.r2i}'], size='scriptsize')
D.proposed(T('A4: VAR output and Granger causality', 'A4: rezultate VAR și cauzalitate Granger'),
           A4OUT + items(T('VAR(2) for Romanian annual inflation $\\pi$ and ROBOR 3M $i$ (\\%), @{a4.first} -- @{a4.last}, $T = @{a4.T}$; $t$ in brackets. Without the lags of $\\pi$, the RSS of the $i$ equation rises from @{a4.g1.ru} to @{a4.g1.rr}. Model: Seminar 6, A5.',
                           'VAR(2) pentru inflația anuală a României $\\pi$ și ROBOR 3M $i$ (\\%), @{a4.first} -- @{a4.last}, $T = @{a4.T}$; statisticile $t$ între paranteze. Fără decalajele lui $\\pi$, RSS a ecuației lui $i$ crește de la @{a4.g1.ru} la @{a4.g1.rr}. Model: Seminarul 6, A5.'),
                         T('1. Write the $i$ equation.', '1. Scrieți ecuația lui $i$.'),
                         T('2. Compute the Granger $F$ statistic for ``$\\pi$ does not Granger-cause $i$\'\' and decide at 5\\% ($F_{0.95} = @{a4.g1.crit}$).', '2. Calculați statistica Granger $F$ pentru „$\\pi$ nu cauzează Granger $i$” și decideți la 5\\% ($F_{0,95} = @{a4.g1.crit}$).'),
                         T('3. Compute the long-run effect on ROBOR of a permanent 1 pp rise in inflation.', '3. Calculați efectul pe termen lung asupra ROBOR al unei creșteri permanente a inflației cu 1 pp.')),
           items(T('1. $i_t = @{a4.bi.const} + @{a4.bi.L1.pi}\\pi_{t-1} + @{a4.bi.L2.pi}\\pi_{t-2} + @{a4.bi.L1.i}i_{t-1} + (@{a4.bi.L2.i})i_{t-2} + u_t$', '1. $i_t = @{a4.bi.const} + @{a4.bi.L1.pi}\\pi_{t-1} + @{a4.bi.L2.pi}\\pi_{t-2} + @{a4.bi.L1.i}i_{t-1} + (@{a4.bi.L2.i})i_{t-2} + u_t$'),
                 T('2. $F = [(@{a4.g1.rr} - @{a4.g1.ru})/2]/[@{a4.g1.ru}/@{a4.g1.df2}] = @{a4.g1.F} > @{a4.g1.crit}$ ($p = @{a4.g1.p}$): rejected; the reverse test: $F = @{a4.g2.F}$, $p = @{a4.g2.p}$', '2. $F = [(@{a4.g1.rr} - @{a4.g1.ru})/2]/[@{a4.g1.ru}/@{a4.g1.df2}] = @{a4.g1.F} > @{a4.g1.crit}$ ($p = @{a4.g1.p}$): se respinge; testul invers: $F = @{a4.g2.F}$, $p = @{a4.g2.p}$'),
                 T('3. $(@{a4.bi.L1.pi} + @{a4.bi.L2.pi})/(1 - @{a4.bi.L1.i} - (@{a4.bi.L2.i})) = @{a4.num}/@{a4.den} = @{a4.lr}$ pp; individually the inflation lags are not significant, jointly they are', '3. $(@{a4.bi.L1.pi} + @{a4.bi.L2.pi})/(1 - @{a4.bi.L1.i} - (@{a4.bi.L2.i})) = @{a4.num}/@{a4.den} = @{a4.lr}$ pp; individual, decalajele inflației nu sînt semnificative, împreună sînt')),
           size='scriptsize')

A5OUT = otab('lccccc', T('\\textbf{DAX}', '\\textbf{DAX}') + ' & $\\mu$ & $\\omega$ & $\\alpha$ & $\\beta$ & $\\nu$',
              [T('estimate', 'estimare') + ' & $@{a5.mu}$ & $@{a5.om}$ & $@{a5.al}$ & $@{a5.be}$ & $@{a5.nu}$',
               'SE & $@{a5.se.mu}$ & $@{a5.se.om}$ & $@{a5.se.al}$ & $@{a5.se.be}$ & $@{a5.se.nu}$'], size='scriptsize')
D.proposed(T('A5: GARCH output for the DAX and VaR 1\\%', 'A5: rezultate GARCH pentru DAX și VaR 1\\%'),
           A5OUT + items(T('GARCH(1,1)-$t$, daily DAX log returns in \\% since 2000 ($n = @{a5.n}$); last day (@{a5.last}): $\\sigma_T^2 = @{a5.s2T}$, $\\hat\\varepsilon_T = @{a5.eT}$; about @{a5.ppy} trading days a year; use $\\nu = @{a5.nu5}$: $t_{@{a5.nu5}}^{-1}(0.01) = @{a5.tq}$. Model: Seminar 5, A1 and A4.',
                           'GARCH(1,1)-$t$, randamentele logaritmice zilnice DAX în \\%, din 2000 ($n = @{a5.n}$); ultima zi (@{a5.last}): $\\sigma_T^2 = @{a5.s2T}$, $\\hat\\varepsilon_T = @{a5.eT}$; circa @{a5.ppy} de zile de tranzacționare pe an; folosiți $\\nu = @{a5.nu5}$: $t_{@{a5.nu5}}^{-1}(0{,}01) = @{a5.tq}$. Model: Seminarul 5, A1 și A4.'),
                         T('1. Compute the persistence, the half-life and the long-run annual volatility.', '1. Calculați persistența, timpul de înjumătățire și volatilitatea anuală pe termen lung.'),
                         T('2. Compute $\\sigma_{T+1}^2$.', '2. Calculați $\\sigma_{T+1}^2$.'),
                         T('3. Compute the one-day VaR 1\\% and compare it with the Normal VaR 1\\%.', '3. Calculați VaR 1\\% pe o zi și comparați-l cu VaR 1\\% Normal.')),
           items(T('1. $\\alpha + \\beta = @{a5.pers}$; $h_{1/2} = @{a5.hl}$ days; $\\bar\\sigma^2 = @{a5.om}/(1 - @{a5.pers}) = @{a5.lv}$; $\\sqrt{@{a5.ppy} \\times @{a5.lv}} = @{a5.ann}\\%$', '1. $\\alpha + \\beta = @{a5.pers}$; $h_{1/2} = @{a5.hl}$ de zile; $\\bar\\sigma^2 = @{a5.om}/(1 - @{a5.pers}) = @{a5.lv}$; $\\sqrt{@{a5.ppy} \\times @{a5.lv}} = @{a5.ann}\\%$'),
                 T('2. $@{a5.om} + @{a5.al} \\times @{a5.e2} + @{a5.be} \\times @{a5.s2T} = @{a5.om} + @{a5.ae2} + @{a5.bs2} = @{a5.s2n}$, $\\sigma_{T+1} = @{a5.sn}\\%$', '2. $@{a5.om} + @{a5.al} \\times @{a5.e2} + @{a5.be} \\times @{a5.s2T} = @{a5.om} + @{a5.ae2} + @{a5.bs2} = @{a5.s2n}$, $\\sigma_{T+1} = @{a5.sn}\\%$'),
                 T('3. $q = @{a5.tq} \\times @{a5.sc} = @{a5.q}$; VaR 1\\% $= -(@{a5.mu} + @{a5.sn}(@{a5.q})) = @{a5.var}\\%$; Normal $@{a5.varn}\\%$: heavier tails, larger VaR', '3. $q = @{a5.tq} \\times @{a5.sc} = @{a5.q}$; VaR 1\\% $= -(@{a5.mu} + @{a5.sn}(@{a5.q})) = @{a5.var}\\%$; Normal $@{a5.varn}\\%$: cozi mai groase, VaR mai mare')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: real data and interpretation', 'Partea B: date reale și interpretare')

D.task(T('B1: Box--Jenkins for Romanian industrial production [Solved]', 'B1: Box--Jenkins pentru producția industrială a României [Rezolvat]'),
       T('which SARIMA model forecasts Romanian industrial production best, and does it beat the seasonal naive method?', 'ce model SARIMA prognozează cel mai bine producția industrială a României și bate el metoda naivă sezonieră?'),
       T('industrial production (B--D), calendar adjusted, monthly, @{b1.first} -- @{b1.last}; test sample: the last 24 months', 'producția industrială (B--D), ajustată calendaristic, lunar, @{b1.first} -- @{b1.last}; eșantionul de test: ultimele 24 de luni'),
       [T('Run ADF and KPSS on $100\\ln y_t$, its first difference and $z_t = \\Delta\\Delta_{12}100\\ln y_t$.', 'Aplicați ADF și KPSS pentru $100\\ln y_t$, prima diferență și $z_t = \\Delta\\Delta_{12}100\\ln y_t$.'),
        T('Plot the ACF of $z_t$ and propose a model.', 'Reprezentați ACF a lui $z_t$ și propuneți un model.'),
        T('Fit the airline model and SARIMA$(1,1,1)(0,1,1)_{12}$ on the training sample; compare AICc, BIC and Ljung--Box.', 'Estimați modelul airline și SARIMA$(1,1,1)(0,1,1)_{12}$ pe eșantionul de antrenare; comparați AICc, BIC și Ljung--Box.'),
        T('Forecast the 24 test months with both models and with the seasonal naive method; compare RMSE and MAE.', 'Prognozați cele 24 de luni de test cu ambele modele și cu metoda naivă sezonieră; comparați RMSE și MAE.'),
        T('Interpretation: which model would you use to forecast, and why?', 'Interpretare: ce model ați folosi pentru prognoză și de ce?')],
       T('a table of tests, a table of two models, three RMSE and one sentence', 'un tabel cu teste, un tabel cu două modele, trei valori RMSE și o frază'), size='footnotesize', nb='B1')

D.frame(T('B1: solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch15_sem_b1', h='0.36') + table(
    'lccccc', T('\\textbf{Model}', '\\textbf{Modelul}') + ' & AICc & BIC & LB(12) $p$ & LB(24) $p$ & ' + T('test RMSE', 'RMSE test'),
    ['airline $(0,1,1)(0,1,1)_{12}$ & @{b1.airline.aicc} & @{b1.airline.bic} & @{b1.airline.lb12} & @{b1.airline.lb24} & @{b1.airline.rmse}',
     '$(1,1,1)(0,1,1)_{12}$ & @{b1.alt.aicc} & @{b1.alt.bic} & @{b1.alt.lb12} & @{b1.alt.lb24} & @{b1.alt.rmse}',
     T('seasonal naive', 'naiv sezonier') + ' & & & & & @{b1.sn.rmse}'], size='scriptsize') + items(
    T('Tests: ADF $\\tau = @{b1.y.adf}$ ($p = @{b1.y.p}$) and KPSS @{b1.y.kpss} $>$ @{b1.y.kcv} for the level; $z_t$: ADF $@{b1.z.adf}$, KPSS @{b1.z.kpss}: $d = D = 1$; ACF of $z_t$: $@{b1.r1}$, $@{b1.r2}$ at lags 1, 2, $@{b1.r12}$ at 12',
      'Teste: ADF $\\tau = @{b1.y.adf}$ ($p = @{b1.y.p}$) și KPSS @{b1.y.kpss} $>$ @{b1.y.kcv} pentru nivel; $z_t$: ADF $@{b1.z.adf}$, KPSS @{b1.z.kpss}: $d = D = 1$; ACF a lui $z_t$: $@{b1.r1}$; $@{b1.r2}$ la decalajele 1, 2; $@{b1.r12}$ la 12'),
    T('Airline: $\\hat\\theta = @{b1.airline.ma.L1}$, $\\hat\\Theta = @{b1.airline.ma.S.L12}$; the $(1,1,1)$ model: $\\hat\\phi = @{b1.alt.ar.L1}$, $\\hat\\theta = @{b1.alt.ma.L1}$, almost cancelling factors',
      'Airline: $\\hat\\theta = @{b1.airline.ma.L1}$, $\\hat\\Theta = @{b1.airline.ma.S.L12}$; modelul $(1,1,1)$: $\\hat\\phi = @{b1.alt.ar.L1}$, $\\hat\\theta = @{b1.alt.ma.L1}$, factori care aproape se anulează'),
    T('Interpretation: the $(1,1,1)$ model wins in sample (BIC, clean residuals) but loses out of sample; the airline model is slightly better than the seasonal naive method: use the airline model, and keep the seasonal naive method as the benchmark',
      'Interpretare: modelul $(1,1,1)$ cîștigă în eșantion (BIC, reziduuri curate), dar pierde în afara eșantionului; modelul airline este puțin mai bun decît metoda naivă sezonieră: folosim modelul airline și păstrăm metoda naivă sezonieră ca reper')) + qlsem(),
    'scriptsize')

D.task(T('B2: volatility of the BET since 2015 [Proposed]', 'B2: volatilitatea BET din 2015 [Propus]'),
       T('is the Bucharest market calmer or more turbulent today than usual, and what does it mean for tomorrow\'s VaR 1\\%? Model: Seminar 5, B1.', 'este bursa de la București azi mai liniștită sau mai agitată decît de obicei și ce înseamnă acest lucru pentru VaR 1\\% de mîine? Model: Seminarul 5, B1.'),
       T('BET daily log returns in \\%, January 2015 -- @{b2.last} ($n = @{b2.n}$)', 'randamentele logaritmice zilnice ale BET în \\%, ianuarie 2015 -- @{b2.last} ($n = @{b2.n}$)'),
       [T('Run the ARCH-LM test with five lags on the demeaned returns.', 'Aplicați testul ARCH-LM cu cinci decalaje pe randamentele centrate.'),
        T('Fit a GARCH(1,1) with Student-$t$ innovations; report the persistence and the half-life.', 'Estimați un GARCH(1,1) cu inovații Student-$t$; raportați persistența și timpul de înjumătățire.'),
        T('Compare the long-run, the sample and today\'s annualised volatility.', 'Comparați volatilitatea anualizată pe termen lung, cea de selecție și cea de azi.'),
        T('Compute tomorrow\'s VaR 1\\% with the $t$ and with the Normal quantile.', 'Calculați VaR 1\\% de mîine cu cuantila $t$ și cu cea Normală.'),
        T('Interpretation: is the BET calmer or more turbulent than usual today?', 'Interpretare: este BET azi mai liniștit sau mai agitat decît de obicei?')],
       T('one test, five parameters, three volatilities, two VaR values and one sentence', 'un test, cinci parametri, trei volatilități, două valori VaR și o frază'), size='footnotesize', nb='B2')

D.frame(T('B2: solution [Proposed]', 'B2: rezolvare [Propus]'), items(
    T('ARCH-LM(5) $= @{b2.lm}$, $p$ @{b2.lmp}: strong ARCH effects', 'ARCH-LM(5) $= @{b2.lm}$, $p$ @{b2.lmp}: efecte ARCH puternice'),
    T('GARCH(1,1)-$t$: $\\mu = @{b2.mu}$, $\\omega = @{b2.om}$, $\\alpha = @{b2.al}$, $\\beta = @{b2.be}$, $\\nu = @{b2.nu}$; persistence @{b2.pers}, half-life @{b2.hl} days', 'GARCH(1,1)-$t$: $\\mu = @{b2.mu}$, $\\omega = @{b2.om}$, $\\alpha = @{b2.al}$, $\\beta = @{b2.be}$, $\\nu = @{b2.nu}$; persistența @{b2.pers}, timpul de înjumătățire @{b2.hl} zile'),
    T('Volatility: long run @{b2.lr_vol}\\%, sample @{b2.vol_sample}\\%, today @{b2.vol_now}\\% a year', 'Volatilitatea: pe termen lung @{b2.lr_vol}\\%, de selecție @{b2.vol_sample}\\%, azi @{b2.vol_now}\\% pe an'),
    T('VaR 1\\% for tomorrow: @{b2.var1}\\% with the $t$ quantile, @{b2.var1_n}\\% with the Normal one', 'VaR 1\\% pentru mîine: @{b2.var1}\\% cu cuantila $t$, @{b2.var1_n}\\% cu cea Normală'),
    T('Interpretation: today the BET is more turbulent than usual (@{b2.vol_now}\\% against @{b2.lr_vol}\\%); with a half-life of about @{b2.hl} days the excess fades within weeks; until then the VaR is higher than its average level',
      'Interpretare: azi BET este mai agitat decît de obicei (@{b2.vol_now}\\%, față de @{b2.lr_vol}\\%); cu un timp de înjumătățire de circa @{b2.hl} zile, excesul se stinge în cîteva săptămîni; pînă atunci VaR este peste nivelul lui mediu')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: the Romanian and German 10-year yields [Proposed]', 'B3: randamentele la 10 ani ale României și Germaniei [Propus]'),
       T('is the Romanian 10-year yield tied to the German one in the long run? Model: Seminar 7, B1 and A3.', 'este randamentul la 10 ani al României legat pe termen lung de cel al Germaniei? Model: Seminarul 7, B1 și A3.'),
       T('monthly means of daily 10-year yields (\\%), @{b3.first} -- @{b3.last} ($T = @{b3.n}$)', 'medii lunare ale randamentelor zilnice la 10 ani (\\%), @{b3.first} -- @{b3.last} ($T = @{b3.n}$)'),
       [T('Run ADF on both yields and on their differences.', 'Aplicați ADF pentru ambele randamente și pentru diferențele lor.'),
        T('Run the Engle--Granger test (Romania on Germany) and report $\\hat\\beta$.', 'Aplicați testul Engle--Granger (România în funcție de Germania) și raportați $\\hat\\beta$.'),
        T('Run the Johansen trace and maximum-eigenvalue tests.', 'Aplicați testele Johansen ale urmei și ale valorii proprii maxime.'),
        T('Estimate the error-correction equation of each yield and compute the half-life.', 'Estimați ecuația cu corecția erorii pentru fiecare randament și calculați timpul de înjumătățire.'),
        T('Interpretation: is the Romanian yield tied to the German one?', 'Interpretare: este randamentul României legat de cel al Germaniei?')],
       T('four ADF results, two cointegration tests, two adjustment coefficients and one sentence', 'patru rezultate ADF, două teste de cointegrare, doi coeficienți de ajustare și o frază'), size='footnotesize', nb='B3')

D.frame(T('B3: solution [Proposed]', 'B3: rezolvare [Propus]'), items(
    T('ADF: levels $@{b3.ro.adf}$ ($p = @{b3.ro.p}$) and $@{b3.de.adf}$ ($p = @{b3.de.p}$): $I(1)$; differences $@{b3.dro.adf}$ and $@{b3.dde.adf}$: stationary', 'ADF: niveluri $@{b3.ro.adf}$ ($p = @{b3.ro.p}$) și $@{b3.de.adf}$ ($p = @{b3.de.p}$): $I(1)$; diferențe $@{b3.dro.adf}$ și $@{b3.dde.adf}$: staționare'),
    T('Engle--Granger: $\\tau = @{b3.eg} < @{b3.egcv}$ ($p = @{b3.egp}$): cointegrated; $\\hat y^{RO} = @{b3.const} + @{b3.beta}\\,y^{DE}$', 'Engle--Granger: $\\tau = @{b3.eg} < @{b3.egcv}$ ($p = @{b3.egp}$): cointegrate; $\\hat y^{RO} = @{b3.const} + @{b3.beta}\\,y^{DE}$'),
    T('Johansen: trace @{b3.tr0} $>$ @{b3.cv0} ($r = 0$ rejected, at the border), @{b3.tr1} $<$ @{b3.cv1}; maximum eigenvalue @{b3.mx0} $<$ @{b3.cvm0}: not rejected', 'Johansen: urma @{b3.tr0} $>$ @{b3.cv0} ($r = 0$ respins, la limită), @{b3.tr1} $<$ @{b3.cv1}; valoarea proprie maximă @{b3.mx0} $<$ @{b3.cvm0}: nu se respinge'),
    T('ECM: Romania $\\gamma = @{b3.g}$ ($t = @{b3.tg}$), half-life @{b3.half} months; Germany $\\gamma = @{b3.gde}$ ($t = @{b3.tgde}$); short run $\\Delta y^{RO}$ on $\\Delta y^{DE}$: @{b3.d} ($t = @{b3.td}$)', 'ECM: România $\\gamma = @{b3.g}$ ($t = @{b3.tg}$), timpul de înjumătățire @{b3.half} luni; Germania $\\gamma = @{b3.gde}$ ($t = @{b3.tgde}$); pe termen scurt $\\Delta y^{RO}$ în funcție de $\\Delta y^{DE}$: @{b3.d} ($t = @{b3.td}$)'),
    T('Interpretation: a weak and slow long-run link: the spread (mean @{b3.spread} pp) returns to equilibrium in years, $\\hat\\beta > 1$ means Romanian yields move more than German ones; the evidence depends on the test, so the honest answer is ``probably, but weakly\'\'',
      'Interpretare: o legătură pe termen lung slabă și lentă: spread-ul (media @{b3.spread} pp) revine la echilibru în cîțiva ani, $\\hat\\beta > 1$ înseamnă că randamentele României se mișcă mai mult decît cele ale Germaniei; dovezile depind de test, deci răspunsul onest este „probabil, dar slab”')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: an open idea and AI critique', 'Partea C: o idee deschisă și critica unui răspuns AI')

D.task(T('C1: a forecast competition for Romanian inflation [Proposed]', 'C1: o competiție de prognoză pentru inflația din România [Propus]'),
       T('which method would have forecast Romanian monthly inflation best over the last ten years, one and twelve months ahead?', 'ce metodă ar fi prognozat cel mai bine inflația lunară din România în ultimii zece ani, cu o lună și cu douăsprezece luni înainte?'),
       T('the Romanian HICP (Eurostat), monthly, 2005--2026; Lecture 15 runs one SARIMA on one test sample', 'IAPC al României (Eurostat), lunar, 2005--2026; Cursul 15 aplică un singur SARIMA pe un singur eșantion de test'),
       [T('List five competitors: seasonal naive, SES, SARIMA, ARFIMA, a ridge model with lags and month dummies.', 'Enumerați cinci concurenți: naiv sezonier, SES, SARIMA, ARFIMA, un model ridge cu decalaje și variabile dummy pentru luni.'),
        T('Describe a rolling-origin design: first origin, step, horizons, refits.', 'Descrieți o schemă cu origini mobile: prima origine, pasul, orizonturile, reestimările.'),
        T('Say how you would treat the tax changes of 2010, 2015 and 2025.', 'Precizați cum ați trata modificările de taxe din 2010, 2015 și 2025.'),
        T('Choose the error measures and the test for the comparison.', 'Alegeți măsurile de eroare și testul pentru comparație.')],
       T('a one-page plan that a team could turn into its project', 'un plan de o pagină pe care o echipă îl poate transforma în proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: a reference plan [Proposed]', 'C1: un plan de referință [Propus]'), items(
    T('Origins every month from January 2015; horizons 1 and 12; every model refitted at every origin, orders chosen on data before the origin', 'Origini lunare din ianuarie 2015; orizonturile 1 și 12; fiecare model reestimat la fiecare origine, cu ordinele alese pe datele dinaintea originii'),
    T('Tax changes: dummies known in advance (VAT dates are announced) or a robust loss; report results with and without the tax months', 'Modificările de taxe: variabile dummy cunoscute dinainte (datele TVA sînt anunțate) sau o funcție de pierdere robustă; raportăm rezultatele cu și fără lunile cu modificări de taxe'),
    T('MASE against the seasonal naive method; Diebold--Mariano with the HLN correction for each pair; a table by horizon', 'MASE față de metoda naivă sezonieră; Diebold--Mariano cu corecția HLN pentru fiecare pereche; un tabel pe orizonturi'),
    T('Expected result: small gains at one month, almost none at twelve months; ARFIMA and SARIMA close; a combination hard to beat', 'Rezultatul așteptat: cîștiguri mici la o lună, aproape niciunul la douăsprezece luni; ARFIMA și SARIMA apropiate; o combinație greu de depășit')),
    'footnotesize', instructor_only=True)

D.frame(T('C2: audit an AI answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to check the answers to Part A. The answer:', 'Un student a cerut unui asistent AI să verifice răspunsurile la Partea A. Răspunsul:'),
    T('\\aiprompt{(a) For the unemployment rate, ADF gives p = @{c2.adfp}, so the unit root is rejected at 5\\%.}', '\\aiprompt{(a) Pentru rata șomajului, ADF dă p = @{c2.adfp}, deci rădăcina unitară se respinge la 5\\%.}'),
    T('\\aiprompt{(b) KPSS = @{c2.kpss} > 0.463 rejects, so the level of unemployment is stationary.}', '\\aiprompt{(b) KPSS = @{c2.kpss} > 0,463 respinge, deci nivelul șomajului este staționar.}'),
    T('\\aiprompt{(c) For the DAX, alpha + beta = @{c2.pers}, so a shock halves in 1/(1 - @{c2.pers}) = @{c2.hlw} days.}', '\\aiprompt{(c) Pentru DAX, alpha + beta = @{c2.pers}, deci un șoc se înjumătățește în 1/(1 - @{c2.pers}) = @{c2.hlw} de zile.}'),
    T('\\aiprompt{(d) The VaR 99\\% of the DAX for tomorrow is -@{c2.var}\\%.}', '\\aiprompt{(d) VaR 99\\% al DAX pentru mîine este -@{c2.var}\\%.}'),
    T('\\aiprompt{(e) Inflation Granger-causes ROBOR (p = @{c2.gp}), which proves that inflation causes the central bank to raise rates.}', '\\aiprompt{(e) Inflația cauzează Granger ROBOR (p = @{c2.gp}), ceea ce dovedește că inflația determină banca centrală să crească dobînzile.}'),
    T('\\aiprompt{(f) Ljung-Box Q(12) on the residuals of an AR(1) x SAR(1) model has 12 degrees of freedom.}', '\\aiprompt{(f) Ljung-Box Q(12) pe reziduurile unui model AR(1) x SAR(1) are 12 grade de libertate.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and the correct number (notebook, section C2).', '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și dați valoarea corectă (notebook, secțiunea C2).'),
      T('2. Report: six verdicts with one line of justification each.', '2. Raportați: șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: $p = @{c2.adfp} > 0.05$: the unit root is not rejected; $u_t \\sim I(1)$', '(a) Greșit: $p = @{c2.adfp} > 0{,}05$: rădăcina unitară nu se respinge; $u_t \\sim I(1)$'),
    T('(b) Wrong: $H_0$ of KPSS is stationarity; rejecting it points to a unit root', '(b) Greșit: la KPSS, $H_0$ este staționaritatea; respingerea ei indică o rădăcină unitară'),
    T('(c) Wrong: $h_{1/2} = \\ln 0.5/\\ln @{c2.pers} = @{c2.hl}$ days; $1/(1 - \\alpha - \\beta)$ is not a half-life', '(c) Greșit: $h_{1/2} = \\ln 0{,}5/\\ln @{c2.pers} = @{c2.hl}$ de zile; $1/(1 - \\alpha - \\beta)$ nu este un timp de înjumătățire'),
    T('(d) Wrong twice: the level is the tail probability, VaR 1\\%, and VaR is a positive loss: @{c2.var}\\%', '(d) Greșit de două ori: nivelul este probabilitatea cozii, VaR 1\\%, iar VaR este o pierdere pozitivă: @{c2.var}\\%'),
    T('(e) Wrong: Granger causality is extra predictability, not proof of a causal policy effect', '(e) Greșit: cauzalitatea Granger înseamnă predictibilitate suplimentară, nu dovada unui efect cauzal al politicii'),
    T('(f) Wrong: $12 - 2 = 10$ degrees of freedom (two estimated ARMA parameters)', '(f) Greșit: $12 - 2 = 10$ grade de libertate (doi parametri ARMA estimați)')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-up', 'Încheiere')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Every exam problem has three parts: the formula with the numbers, the result with its unit and sign, the interpretation', 'Orice problemă de examen are trei părți: formula cu cifrele înlocuite, rezultatul cu unitate și semn, interpretarea'),
    T('Tests: state $H_0$ first; ADF and KPSS have opposite nulls; Ljung--Box on residuals loses $k$ degrees of freedom', 'Testele: precizați întîi $H_0$; ADF și KPSS au ipoteze nule opuse; Ljung--Box pe reziduuri pierde $k$ grade de libertate'),
    T('Forecasts by hand: write the model in levels of $z_t$, substitute the last values, then undo the differences', 'Prognozele de mînă: scrieți modelul pentru $z_t$, înlocuiți ultimele valori, apoi refaceți diferențierile'),
    T('A better in-sample model is not a better forecaster (B1); judge it on a test sample against a benchmark', 'Un model mai bun în eșantion nu prognozează neapărat mai bine (B1); judecați-l pe un eșantion de test, față de un reper'),
    T('An AI answer is a draft: check the null hypotheses, the half-life, the VaR level and the degrees of freedom', 'Un răspuns AI este o ciornă: verificați ipotezele nule, timpul de înjumătățire, nivelul VaR și gradele de libertate')))

D.frame(T('After the seminar', 'După seminar'), items(
    T('Lecture 15 gathers the course: the course map, one recap slide per chapter, the toolbox, Box--Jenkins on Romanian inflation, eight solved exam-type problems, the project',
      'Cursul 15 adună întreaga materie: harta cursului, cîte un slide de recapitulare pentru fiecare capitol, trusa de instrumente, Box--Jenkins pe inflația din România, opt probleme de tip examen rezolvate, proiectul'),
    T('Try the [Proposed] tasks on paper first, then check them in the notebook', 'Încercați cerințele [Propus] întîi pe hîrtie, apoi verificați-le în notebook'),
    T('C1 can grow into a team project', 'C1 poate deveni un proiect de echipă'),
    T('Reading: \\refHP; \\refFPP; \\refBJ', 'Lectură: \\refHP; \\refFPP; \\refBJ'),
    CLOSE))

D.references(bib(['HP', 'FPP', 'BJ', 'DF', 'KPSS', 'LB', 'Granger', 'EG', 'Joh', 'Boll', 'DM']), per=16)

if __name__ == '__main__':
    finalize(D.write(V))
