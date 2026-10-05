r"""
build_seminar4.py -- Seminarul 4 (Sezonalitate și prognoză: SARIMA, TBATS, Prophet), EN + RO dintr-o singură sursă
==================================================================================================================
Seminarul are loc ÎNAINTEA cursului 4: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu o întrebare de interpretare, C o întrebare deschisă și
critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în versiunea
profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_04/sem4_results.json (seminar4.py).
Ieșire:
  EN/Seminars/seminar4_seasonality_forecasting.tex          (+ _solutions.tex)
  RO/Seminarii/seminar4_sezonalitate_prognoza_ro.tex        (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_04/seminar4.py && python3 latex/build_seminar4.py && python3 latex/tsa_build.py compile 4
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, items, table, fig   # noqa: E402
from ch4_common import REFS, T, bib, date, finalize, load_sem, pv, quarter   # noqa: E402

S = load_sem()
V = Values()
D = Deck(4, 'seminar', refs=REFS)


def qlsem():
    return '\\quantlet{TSA\\_ch4\\_seminar}{\\qlurl{TSA_ch4_seminar}}'


# =============================================================================
# CIFRE
# =============================================================================
A1 = S['A1']
for i, v in enumerate(A1['ds']):
    V.put(f'a1.ds{i + 5}', v, 0)
for i, v in enumerate(A1['dds']):
    V.put(f'a1.dd{i + 6}', v, 0)
for i, v in enumerate(A1['snaive']):
    V.put(f'a1.f{i + 1}', v, 0)
A2 = S['A2']
V.put('a2.ma13', A2['airline']['ma']['13'], 2)
V.put('a2.ar13', A2['ar']['ar']['13'], 2)
A3 = S['A3']
V.put('a3.g0', A3['g0'], 2)
V.put('a3.r1', A3['rho']['1'], 3)
V.put('a3.r3', A3['rho']['3'], 3)
V.put('a3.r4', A3['rho']['4'], 3)
V.put('a3.r5', A3['rho']['5'], 3)
A4 = S['A4']
V.put('a4.r4', A4['sar'][3], 2)
V.put('a4.r8', A4['sar'][7], 2)
V.put('a4.r12', A4['sar'][11], 3)
V.put('a4.th', A4['theta'], 2)
V.put('a4.Th', A4['Theta'], 3)
A5 = S['A5']
V.put('a5.mae1', A5['mae1'], 3)
V.put('a5.mae2', A5['mae2'], 3)
for i, v in enumerate(A5['d']):
    V.put(f'a5.d{i + 1}', v, 2)
V.put('a5.dbar', A5['dbar'], 3)
V.put('a5.s2', A5['s2'], 4)
V.put('a5.dm', A5['dm'], 2)
V.put('a5.hln', A5['hln'], 2)
V.put('a5.tc', A5['t_crit'], 3)
V.put('a5.p', A5['p'], 3)
V.put('a5.corr', ((A5['n'] - 1) / A5['n']) ** 0.5, 3)
A6 = S['A6']
V.put('a6.mae1', A6['mae1'], 3)
V.put('a6.mae2', A6['mae2'], 3)
V.put('a6.dbar', A6['dbar'], 3)
V.put('a6.s2', A6['s2'], 3)
V.put('a6.dm', A6['dm'], 3)
V.put('a6.hln', A6['hln'], 3)
V.put('a6.tc', A6['t_crit'], 3)
V.put('a6.p', A6['p'], 2)
V.put('a6.cmae', A6['comb']['mae1'], 4)
V.put('a6.crmse', A6['comb']['rmse1'], 3)
V.put('a6.rmse1', A6['rmse1'], 3)
V.put('a6.rmse2', A6['rmse2'], 3)
TH = A6['theory']
V.put('a6.mh', TH['mse_half'], 2)
V.put('a6.w', TH['w_opt'], 3)
V.put('a6.mo', TH['mse_opt'], 3)

B1 = S['B1']
V.put('b1.th', B1['theta'], 3)
V.put('b1.ths', B1['theta_se'], 3)
V.put('b1.Th', B1['Theta'], 3)
V.put('b1.Ths', B1['Theta_se'], 3)
V.put('b1.al', 1 + B1['Theta'], 2)
V.put('b1.sig', B1['sigma'], 2)
V.put('b1.q', B1['lb8']['lb'], 2)
V.raw('b1.qp', pv(B1['lb8']['lb_p']))
V.raw('b1.q0', quarter(B1['first_f']))
V.raw('b1.last', quarter(B1['last']))
V.int('b1.T', B1['T'])
for i in range(4):
    V.put(f'b1.f{i + 1}', B1['f_level'][i], 1)
    V.put(f'b1.g{i + 1}', B1['g'][i], 2)
V.put('b1.lo1', B1['lo'][0], 1)
V.put('b1.hi1', B1['hi'][0], 1)
V.put('b1.lo4', B1['lo'][3], 1)
V.put('b1.hi4', B1['hi'][3], 1)

B2 = S['B2']
TB2 = {r['model']: r for r in B2['table']}
assert B2['best'] == '(0,1,1)(0,1,1)12 + WD'
for key, m in [('a', '(0,1,1)(0,1,1)12'), ('aw', '(0,1,1)(0,1,1)12 + WD'), ('b', '(2,1,0)(0,1,1)12'),
               ('bw', '(2,1,0)(0,1,1)12 + WD'), ('cw', '(1,1,1)(0,1,1)12 + WD'), ('dw', '(0,1,1)(1,1,1)12 + WD')]:
    V.put(f'b2.{key}.aicc', TB2[m]['aicc'], 1)
    V.raw(f'b2.{key}.p', pv(TB2[m]['lb24_p']))
V.put('b2.a.Th', TB2['(0,1,1)(0,1,1)12']['Theta'], 3)
V.put('b2.b.Th', TB2['(2,1,0)(0,1,1)12']['Theta'], 3)
P2, S2 = B2['params'], B2['se']
V.put('b2.wd', P2['x3'], 2)
V.put('b2.wds', S2['x3'], 2)
V.put('b2.th', P2['ma.L1'], 3)
V.put('b2.Th', P2['ma.S.L12'], 3)
V.put('b2.ao', P2['x1'], 1)
V.raw('b2.last', date(B2['last'], day=False))
V.int('b2.T', B2['T'])
V.put('b2.q', B2['lb24_best']['lb'], 1)
V.put('b2.aug', B2['mom_month'][7], 1)
V.put('b2.dec', B2['mom_month'][11], 1)

B3 = S['B3']
for k, key in [('Seasonal naive', 'sn'), ('SARIMA', 'sa'), ('DHR', 'dh')]:
    V.put(f'b3.{key}.mae', 1000 * B3['mae'][k], 0)
    V.put(f'b3.{key}.mase', B3['mase'][k], 2)
    V.put(f'b3.{key}.hol', 1000 * B3['mae_hol'][k], 0)
    V.put(f'b3.{key}.nh', 1000 * B3['mae_nohol'][k], 0)
V.put('b3.scale', 1000 * B3['scale'], 0)
V.raw('b3.n', str(B3['n_origins']))
V.raw('b3.d0', date(B3['first']))
V.raw('b3.d1', date(B3['last']))
V.put('b3.dm', B3['dm']['hln'], 2)
V.raw('b3.dmp', pv(B3['dm']['p']))
V.put('b3.dmsn', B3['dm_sn']['hln'], 2)
V.raw('b3.dmsnp', pv(B3['dm_sn']['p']))
V.put('b3.dbar', 1e6 * B3['dm']['dbar'], 0)
V.raw('b3.nhol', str(B3['n_hol']))
V.put('b3.hol', -B3['coef']['holiday'], 2)
V.put('b3.eas', -B3['coef']['easter'], 2)
V.put('b2.wd2', 2 * B2['params']['x3'], 1)

B4 = S['B4']
V.put('b4.F', B4['F'], 2)
V.raw('b4.pF', pv(B4['pF']))
V.put('b4.jun', B4['month_mean'][5], 2)
V.put('b4.jan', B4['month_mean'][0], 2)
V.put('b4.oct', B4['month_mean'][9], 2)
V.put('b4.aug', B4['month_mean'][7], 2)
V.put('b4.th', B4['theta'], 3)
V.put('b4.ths', B4['theta_se'], 3)
V.put('b4.Th', B4['Theta'], 3)
V.put('b4.Ths', B4['Theta_se'], 3)
for k, key in [('SARIMA', 'sa'), ('Seasonal naive', 'sn'), ('Mean of last 12', 'm12')]:
    V.put(f'b4.{key}', B4['mae'][k], 3)
V.put('b4.dmm', B4['dm_mean']['hln'], 2)
V.raw('b4.dmmp', pv(B4['dm_mean']['p']))
V.put('b4.dms', B4['dm_sn']['hln'], 2)
V.raw('b4.dmsp', pv(B4['dm_sn']['p']))
V.raw('b4.n', str(B4['n_fc']))
V.raw('b4.d0', date(B4['first_fc'], day=False))
V.raw('b4.d1', date(B4['last_fc'], day=False))

C1 = S['C1']
V.put('c1.m19', C1['mape']['trained to 2019'], 1)
V.put('c1.m22', C1['mape']['trained to 2022'], 1)
V.put('c1.sn', C1['mape']['seasonal naive 2022'], 1)
V.put('c1.y25', C1['act_2025'], 1)
V.put('c1.y19', C1['act_2019'], 1)
C2 = S['C2']
V.put('c2.nsa', C2['qoq_nsa'], 1)
V.put('c2.sca', C2['qoq_sca'], 2)
V.raw('c2.q', quarter(C2['q']))

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how do we model, forecast and compare forecasts of series that repeat a pattern every year, week or day?',
       '\\textbf{Întrebarea}: cum modelăm, prognozăm și comparăm prognozele unor serii care repetă un tipar în fiecare an, săptămînă sau zi?'),
     [T('this seminar comes \\textbf{before} Lecture 4: the section ``What you need today\'\' gives every formula the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 4: secțiunea „Noțiuni necesare azi” dă toate formulele folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: seasonal differences, SARIMA polynomials and autocorrelations, the Diebold--Mariano test and forecast combination on paper',
        'Partea A: diferențe sezoniere, polinoamele și autocorelațiile SARIMA, testul Diebold--Mariano și combinarea prognozelor, pe hîrtie'),
      T('Part B: Romanian GDP, industrial production, daily electricity load and monthly inflation',
        'Partea B: PIB-ul României, producția industrială, consumul zilnic de electricitate și inflația lunară'),
      T('Part C: tourism after the pandemic and an AI answer to audit', 'Partea C: turismul după pandemie și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('seasonal differences and seasonal naive forecasts; SARIMA polynomials', 'diferențe sezoniere și prognoze sezoniere naive; polinoamele SARIMA') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('the ACF of the airline model; a seasonal AR and moment estimates', 'ACF a modelului airline; un AR sezonier și estimări prin metoda momentelor') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('MASE and the Diebold--Mariano test; combining two forecasts', 'MASE și testul Diebold--Mariano; combinarea a două prognoze') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('SARIMA: Romanian GDP; industrial production with working days', 'SARIMA: PIB-ul României; producția industrială cu zile lucrătoare') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('cross-validation: electricity load; monthly inflation', 'validare încrucișată: consumul de electricitate; inflația lunară') + ' & ' + SP + ' & B3',
     'C1, C2 & ' + T('tourism after the pandemic; what is wrong in an AI answer?', 'turismul după pandemie; ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, A5'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați dumneavoastră, după model')))

D.frame(T('Data used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Frequency}', '\\textbf{Frecvența}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    [T('Real GDP, not adjusted', 'PIB real, neajustat') + ' & Eurostat (namq\\_10\\_gdp) & ' + T('quarterly', 'trimestrial') + ' & 2000--2026',
     T('Industrial production', 'Producția industrială') + ' & Eurostat (sts\\_inpr\\_m) & ' + T('monthly, 2021 = 100', 'lunar, 2021 = 100') + ' & 2010--2026',
     T('Electricity load', 'Consumul de electricitate') + ' & ENTSO-E & ' + T('daily mean of hourly data', 'media zilnică a datelor orare') + ' & 2022--2026',
     T('HICP, Romania', 'IAPC, România') + ' & Eurostat (prc\\_hicp\\_minr) & ' + T('monthly, 2015 = 100', 'lunar, 2015 = 100') + ' & 2010--2026',
     T('Tourism nights', 'Înnoptări turistice') + ' & Eurostat (tour\\_occ\\_nim) & ' + T('monthly', 'lunar') + ' & 2005--2025'],
    size='footnotesize') + items(
    T('Logs times 100: $100\\ln Y_t$, so that differences are growth rates in \\%', 'Logaritmi înmulțiți cu 100: $100\\ln Y_t$, astfel încît diferențele sînt rate de creștere în \\%'),
    T('ENTSO-E: European Network of Transmission System Operators for Electricity; load = electricity consumed, in MW (hourly average)',
      'ENTSO-E: rețeaua europeană a operatorilor de transport și de sistem pentru energie electrică; consumul este exprimat în MW (media orară)'),
    T('In the notebook: \\texttt{fit\\_sarima}, \\texttt{b1\\_gdp()}, \\texttt{b3\\_load\\_cv()}; no account or key is needed',
      'În notebook: \\texttt{fit\\_sarima}, \\texttt{b1\\_gdp()}, \\texttt{b3\\_load\\_cv()}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What you need today', 'Noțiuni necesare azi')

D.frame(T('What you need today (1/4): seasonality and seasonal differences', 'Noțiuni necesare azi (1/4): sezonalitate și diferențe sezoniere'), items(
    (T('\\textbf{Seasonality}: a pattern that repeats with a fixed \\textbf{period} $s$: $s = 4$ (quarters), $s = 12$ (months), $s = 7$ (days of the week), $s = 24$ (hours of the day)',
       '\\textbf{Sezonalitatea}: un tipar care se repetă cu o \\textbf{perioadă} fixă $s$: $s = 4$ (trimestre), $s = 12$ (luni), $s = 7$ (zilele săptămînii), $s = 24$ (orele zilei)'),
     [T('causes: weather, the calendar (working days, holidays), institutions (tax dates, school year)', 'cauze: vremea, calendarul (zile lucrătoare, sărbători), instituțiile (termene fiscale, anul școlar)')]),
    (T('\\textbf{Seasonal difference}: $\\Delta_s y_t = (1 - L^s)y_t = y_t - y_{t-s}$; with logs, $100\\,\\Delta_4\\ln Y_t$ is the year-on-year growth rate',
       '\\textbf{Diferența sezonieră}: $\\Delta_s y_t = (1 - L^s)y_t = y_t - y_{t-s}$; cu logaritmi, $100\\,\\Delta_4\\ln Y_t$ este rata de creștere față de același trimestru al anului anterior'),
     [T('both differences: $\\Delta\\Delta_s y_t = (1 - L)(1 - L^s)y_t = y_t - y_{t-1} - y_{t-s} + y_{t-s-1}$', 'ambele diferențe: $\\Delta\\Delta_s y_t = (1 - L)(1 - L^s)y_t = y_t - y_{t-1} - y_{t-s} + y_{t-s-1}$')]),
    (T('\\textbf{Seasonal naive forecast}: $\\hat y_{T+h} = y_{T+h-s}$ for $h \\le s$: the value of the same season one period earlier (Chapter 0)',
       '\\textbf{Prognoza sezonieră naivă}: $\\hat y_{T+h} = y_{T+h-s}$ pentru $h \\le s$: valoarea din același sezon cu o perioadă în urmă (Capitolul 0)'),
     [T('the benchmark every seasonal model must beat', 'reperul pe care orice model sezonier trebuie să îl bată')])))

D.frame(T('What you need today (2/4): SARIMA models', 'Noțiuni necesare azi (2/4): modele SARIMA'), items(
    (T('\\textbf{SARIMA$(p,d,q)(P,D,Q)_s$}: $\\phi(L)\\Phi(L^s)(1 - L)^d(1 - L^s)^D y_t = \\theta(L)\\Theta(L^s)\\varepsilon_t$',
       '\\textbf{SARIMA$(p,d,q)(P,D,Q)_s$}: $\\phi(L)\\Phi(L^s)(1 - L)^d(1 - L^s)^D y_t = \\theta(L)\\Theta(L^s)\\varepsilon_t$'),
     [T('$\\Phi(L^s) = 1 - \\Phi_1L^s - \\dots - \\Phi_PL^{Ps}$ (seasonal AR), $\\Theta(L^s) = 1 + \\Theta_1L^s + \\dots + \\Theta_QL^{Qs}$ (seasonal MA)',
        '$\\Phi(L^s) = 1 - \\Phi_1L^s - \\dots - \\Phi_PL^{Ps}$ (AR sezonier), $\\Theta(L^s) = 1 + \\Theta_1L^s + \\dots + \\Theta_QL^{Qs}$ (MA sezonier)'),
      T('the polynomials are \\textbf{multiplied}: cross terms such as $\\theta\\Theta L^{s+1}$ appear without new parameters', 'polinoamele se \\textbf{înmulțesc}: apar termeni încrucișați precum $\\theta\\Theta L^{s+1}$, fără parametri noi')]),
    (T('\\textbf{Airline model} (Box and Jenkins): SARIMA$(0,1,1)(0,1,1)_s$', '\\textbf{Modelul airline} (Box și Jenkins): SARIMA$(0,1,1)(0,1,1)_s$'),
     [T('$w_t = \\Delta\\Delta_s y_t = (1 + \\theta L)(1 + \\Theta L^s)\\varepsilon_t$: two parameters, $\\theta$ and $\\Theta$', '$w_t = \\Delta\\Delta_s y_t = (1 + \\theta L)(1 + \\Theta L^s)\\varepsilon_t$: doi parametri, $\\theta$ și $\\Theta$'),
      T('ACF of $w_t$: $\\rho_1 = \\frac{\\theta}{1 + \\theta^2}$, $\\rho_s = \\frac{\\Theta}{1 + \\Theta^2}$, $\\rho_{s-1} = \\rho_{s+1} = \\rho_1\\rho_s$, all other lags 0',
        'ACF pentru $w_t$: $\\rho_1 = \\frac{\\theta}{1 + \\theta^2}$, $\\rho_s = \\frac{\\Theta}{1 + \\Theta^2}$, $\\rho_{s-1} = \\rho_{s+1} = \\rho_1\\rho_s$, celelalte decalaje 0')]),
    (T('\\textbf{Identification} at seasonal lags $s, 2s, 3s$: as in Chapter 2, but one season apart', '\\textbf{Identificarea} la decalajele sezoniere $s, 2s, 3s$: ca în Capitolul 2, dar cu un sezon distanță'),
     [T('ACF cuts off after lag $s$, PACF decays at $s, 2s, \\dots$: seasonal MA(1); the reverse: seasonal AR(1)', 'ACF se anulează după decalajul $s$, PACF descrește la $s, 2s, \\dots$: MA(1) sezonier; invers: AR(1) sezonier')])))

D.frame(T('What you need today (3/4): calendar regressors and Fourier terms', 'Noțiuni necesare azi (3/4): regresori de calendar și termeni Fourier'), items(
    (T('\\textbf{Regression with SARIMA errors}: $y_t = \\beta^\\top x_t + u_t$, $u_t \\sim$ SARIMA; $x_t$ known in advance: working days, holidays, outliers',
       '\\textbf{Regresia cu erori SARIMA}: $y_t = \\beta^\\top x_t + u_t$, $u_t \\sim$ SARIMA; $x_t$ cunoscut dinainte: zile lucrătoare, sărbători, valori aberante'),
     [T('\\textbf{working days}: number of Monday--Friday days of the month that are not public holidays', '\\textbf{zile lucrătoare}: numărul zilelor de luni pînă vineri din lună care nu sînt sărbători legale'),
      T('with $100\\ln y_t$, the coefficient of a dummy is approximately the effect in \\%', 'cu $100\\ln y_t$, coeficientul unei variabile dummy este aproximativ efectul în \\%')]),
    (T('\\textbf{Fourier terms} for a period $m$: $\\sin(2\\pi kt/m)$, $\\cos(2\\pi kt/m)$, $k = 1, \\dots, K$', '\\textbf{Termeni Fourier} pentru o perioadă $m$: $\\sin(2\\pi kt/m)$, $\\cos(2\\pi kt/m)$, $k = 1, \\dots, K$'),
     [T('$2K$ coefficients instead of $m - 1$ dummies; $m$ may be non-integer ($365.25$ days in a year)', '$2K$ coeficienți în locul celor $m - 1$ variabile dummy; $m$ poate fi neîntreg ($365{,}25$ de zile într-un an)'),
      T('\\textbf{dynamic harmonic regression} (DHR): Fourier terms plus calendar dummies, with ARMA errors', '\\textbf{regresia armonică dinamică} (DHR): termeni Fourier plus variabile dummy de calendar, cu erori ARMA')]),
    T('TBATS and Prophet (Lecture 4) are two automatic versions of the same idea: several seasonal periods, Fourier-type seasonality',
      'TBATS și Prophet (Cursul 4) sînt două variante automate ale aceleiași idei: mai multe perioade sezoniere, sezonalitate de tip Fourier')))

D.frame(T('What you need today (4/4): comparing forecasts', 'Noțiuni necesare azi (4/4): compararea prognozelor'), items(
    (T('\\textbf{Time-series cross-validation}: origins $T_1 < T_2 < \\dots$; at each origin, fit on the data up to it and forecast $h$ steps',
       '\\textbf{Validarea încrucișată pentru serii de timp}: originile $T_1 < T_2 < \\dots$; la fiecare origine, estimăm pe datele pînă la ea și prognozăm $h$ pași'),
     [T('MAE $= \\frac1n\\sum|e_i|$; RMSE $= \\sqrt{\\frac1n\\sum e_i^2}$; MASE = MAE / (in-sample MAE of the seasonal naive method) (Chapter 0)',
        'MAE $= \\frac1n\\sum|e_i|$; RMSE $= \\sqrt{\\frac1n\\sum e_i^2}$; MASE = MAE / (MAE în eșantion a metodei sezoniere naive) (Capitolul 0)')]),
    (T('\\textbf{Diebold--Mariano} (DM) test of equal accuracy: $d_t = L(e_{1t}) - L(e_{2t})$, loss $L(e) = e^2$ or $|e|$',
       '\\textbf{Testul Diebold--Mariano} (DM) al acurateței egale: $d_t = L(e_{1t}) - L(e_{2t})$, pierderea $L(e) = e^2$ sau $|e|$'),
     [T('$\\mathrm{DM} = \\bar d/\\sqrt{\\hat\\sigma_d^2/n}$, $\\hat\\sigma_d^2 = \\frac1n\\sum(d_t - \\bar d)^2$ for one-step (or non-overlapping) losses',
        '$\\mathrm{DM} = \\bar d/\\sqrt{\\hat\\sigma_d^2/n}$, $\\hat\\sigma_d^2 = \\frac1n\\sum(d_t - \\bar d)^2$ pentru pierderi la un pas (sau care nu se suprapun)'),
      T('small-sample correction (Harvey, Leybourne, Newbold): $\\mathrm{HLN} = \\sqrt{(n - 1)/n}\\;\\mathrm{DM}$, compared with Student $t(n - 1)$',
        'corecția de eșantion mic (Harvey, Leybourne, Newbold): $\\mathrm{HLN} = \\sqrt{(n - 1)/n}\\;\\mathrm{DM}$, comparată cu distribuția Student $t(n - 1)$'),
      T('$\\bar d < 0$: forecast 1 is more accurate; 5\\% critical values $t_{0.975}(5) = 2.571$, $t_{0.975}(7) = 2.365$',
        '$\\bar d < 0$: prognoza 1 este mai precisă; valori critice de 5\\%: $t_{0{,}975}(5) = 2{,}571$, $t_{0{,}975}(7) = 2{,}365$')]),
    (T('\\textbf{Forecast combination}: $f_c = w f_1 + (1 - w)f_2$; for unbiased forecasts', '\\textbf{Combinarea prognozelor}: $f_c = w f_1 + (1 - w)f_2$; pentru prognoze nedeplasate'),
     [T('$\\mathrm{MSE}(w) = w^2\\sigma_1^2 + (1 - w)^2\\sigma_2^2 + 2w(1 - w)\\rho\\sigma_1\\sigma_2$; best $w^* = \\frac{\\sigma_2^2 - \\rho\\sigma_1\\sigma_2}{\\sigma_1^2 + \\sigma_2^2 - 2\\rho\\sigma_1\\sigma_2}$',
        '$\\mathrm{MSE}(w) = w^2\\sigma_1^2 + (1 - w)^2\\sigma_2^2 + 2w(1 - w)\\rho\\sigma_1\\sigma_2$; cea mai bună pondere $w^* = \\frac{\\sigma_2^2 - \\rho\\sigma_1\\sigma_2}{\\sigma_1^2 + \\sigma_2^2 - 2\\rho\\sigma_1\\sigma_2}$')])), size='footnotesize')

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: computations on paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: seasonal differences by hand', 'A1: diferențe sezoniere calculate de mînă'),
         items(T('A quarterly series, $t = 1, \\dots, 9$: 10, 14, 16, 18, 12, 16, 19, 21, 14.', 'O serie trimestrială, $t = 1, \\dots, 9$: 10, 14, 16, 18, 12, 16, 19, 21, 14.'),
               T('1. Compute $\\Delta_4 y_t$ for $t = 5, \\dots, 9$.', '1. Calculați $\\Delta_4 y_t$ pentru $t = 5, \\dots, 9$.'),
               T('2. Compute $\\Delta\\Delta_4 y_t$ for $t = 6, \\dots, 9$.', '2. Calculați $\\Delta\\Delta_4 y_t$ pentru $t = 6, \\dots, 9$.'),
               T('3. Expand $(1 - L)(1 - L^4)$ and check it on $t = 9$.', '3. Dezvoltați $(1 - L)(1 - L^4)$ și verificați rezultatul pentru $t = 9$.'),
               T('4. Give the seasonal naive forecasts for $t = 10, \\dots, 13$.', '4. Dați prognozele sezoniere naive pentru $t = 10, \\dots, 13$.'),
               T('Report: thirteen numbers and one sentence on what $\\Delta_4$ removes.', 'Raportați: treisprezece valori și o frază despre ce elimină $\\Delta_4$.')),
         items(T('1. $\\Delta_4 y$: $@{a1.ds5}, @{a1.ds6}, @{a1.ds7}, @{a1.ds8}, @{a1.ds9}$ (e.g. $12 - 10$, $16 - 14$)', '1. $\\Delta_4 y$: $@{a1.ds5}; @{a1.ds6}; @{a1.ds7}; @{a1.ds8}; @{a1.ds9}$ (de exemplu $12 - 10$, $16 - 14$)'),
               T('2. $\\Delta\\Delta_4 y$: $@{a1.dd6}, @{a1.dd7}, @{a1.dd8}, @{a1.dd9}$', '2. $\\Delta\\Delta_4 y$: $@{a1.dd6}; @{a1.dd7}; @{a1.dd8}; @{a1.dd9}$'),
               T('3. $1 - L - L^4 + L^5$; $t = 9$: $14 - 21 - 12 + 18 = @{a1.dd9}$', '3. $1 - L - L^4 + L^5$; $t = 9$: $14 - 21 - 12 + 18 = @{a1.dd9}$'),
               T('4. $\\hat y_{10}, \\dots, \\hat y_{13} = @{a1.f1}, @{a1.f2}, @{a1.f3}, @{a1.f4}$ (the last four values)', '4. $\\hat y_{10}, \\dots, \\hat y_{13} = @{a1.f1}; @{a1.f2}; @{a1.f3}; @{a1.f4}$ (ultimele patru valori)'),
               T('$\\Delta_4$ removes a seasonal pattern that repeats exactly; the small, stable $\\Delta_4 y$ shows the year-on-year growth.', '$\\Delta_4$ elimină un tipar sezonier care se repetă exact; valorile mici și stabile ale lui $\\Delta_4 y$ arată creșterea de la un an la altul.')),
         size='scriptsize')

D.proposed(T('A2: SARIMA polynomials', 'A2: polinoamele SARIMA'),
           items(T('Monthly data, $s = 12$. Model: A1.', 'Date lunare, $s = 12$. Model: A1.'),
                 T('1. Write the airline model SARIMA$(0,1,1)(0,1,1)_{12}$ with $\\theta = -0.4$, $\\Theta = -0.6$ as an equation for $y_t$.', '1. Scrieți modelul airline SARIMA$(0,1,1)(0,1,1)_{12}$ cu $\\theta = -0{,}4$, $\\Theta = -0{,}6$ ca ecuație pentru $y_t$.'),
                 T('2. List the lags of $y$ and of $\\varepsilon$ that appear, and the coefficient of $\\varepsilon_{t-13}$.', '2. Enumerați decalajele lui $y$ și ale lui $\\varepsilon$ care apar și coeficientul lui $\\varepsilon_{t-13}$.'),
                 T('3. Expand $(1 - 0.5L)(1 - 0.3L^{12})$ for SARIMA$(1,0,0)(1,0,0)_{12}$.', '3. Dezvoltați $(1 - 0{,}5L)(1 - 0{,}3L^{12})$ pentru SARIMA$(1,0,0)(1,0,0)_{12}$.'),
                 T('4. Count the parameters of SARIMA$(1,1,1)(1,1,1)_{12}$, including $\\sigma^2$.', '4. Numărați parametrii modelului SARIMA$(1,1,1)(1,1,1)_{12}$, inclusiv $\\sigma^2$.'),
                 T('Report: two equations, two lists of lags and one count.', 'Raportați: două ecuații, două liste de decalaje și un număr.')),
           items(T('1. $y_t = y_{t-1} + y_{t-12} - y_{t-13} + \\varepsilon_t - 0.4\\varepsilon_{t-1} - 0.6\\varepsilon_{t-12} + @{a2.ma13}\\varepsilon_{t-13}$', '1. $y_t = y_{t-1} + y_{t-12} - y_{t-13} + \\varepsilon_t - 0{,}4\\varepsilon_{t-1} - 0{,}6\\varepsilon_{t-12} + @{a2.ma13}\\varepsilon_{t-13}$'),
                 T('2. $y$: lags 1, 12, 13; $\\varepsilon$: lags 0, 1, 12, 13; the coefficient of $\\varepsilon_{t-13}$ is $\\theta\\Theta = @{a2.ma13}$, not a free parameter', '2. $y$: decalajele 1, 12, 13; $\\varepsilon$: decalajele 0, 1, 12, 13; coeficientul lui $\\varepsilon_{t-13}$ este $\\theta\\Theta = @{a2.ma13}$, nu un parametru liber'),
                 T('3. $1 - 0.5L - 0.3L^{12} + @{a2.ar13}L^{13}$: $y_t = 0.5y_{t-1} + 0.3y_{t-12} - @{a2.ar13}y_{t-13} + \\varepsilon_t$', '3. $1 - 0{,}5L - 0{,}3L^{12} + @{a2.ar13}L^{13}$: $y_t = 0{,}5y_{t-1} + 0{,}3y_{t-12} - @{a2.ar13}y_{t-13} + \\varepsilon_t$'),
                 T('4. $\\phi, \\theta, \\Phi, \\Theta, \\sigma^2$: five parameters (no constant once $d = D = 1$)', '4. $\\phi, \\theta, \\Phi, \\Theta, \\sigma^2$: cinci parametri (fără constantă cînd $d = D = 1$)')),
           size='scriptsize')

D.solved(T('A3: the ACF of the airline model', 'A3: ACF a modelului airline'),
         items(T('Quarterly airline model: $w_t = \\Delta\\Delta_4 y_t = (1 - 0.5L)(1 - 0.6L^4)\\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$.', 'Modelul airline trimestrial: $w_t = \\Delta\\Delta_4 y_t = (1 - 0{,}5L)(1 - 0{,}6L^4)\\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$.'),
               T('1. Write $w_t$ as an MA(5) and list its nonzero coefficients.', '1. Scrieți $w_t$ ca MA(5) și enumerați coeficienții nenuli.'),
               T('2. Compute $\\gamma(0)/\\sigma^2$.', '2. Calculați $\\gamma(0)/\\sigma^2$.'),
               T('3. Compute $\\rho_1, \\dots, \\rho_5$.', '3. Calculați $\\rho_1, \\dots, \\rho_5$.'),
               T('4. Say what the sample ACF of $w_t$ should look like.', '4. Precizați cum ar trebui să arate ACF de selecție a lui $w_t$.'),
               T('Report: five autocorrelations and one sentence.', 'Raportați: cinci autocorelații și o frază.')),
         items(T('1. $w_t = \\varepsilon_t - 0.5\\varepsilon_{t-1} - 0.6\\varepsilon_{t-4} + 0.3\\varepsilon_{t-5}$', '1. $w_t = \\varepsilon_t - 0{,}5\\varepsilon_{t-1} - 0{,}6\\varepsilon_{t-4} + 0{,}3\\varepsilon_{t-5}$'),
               T('2. $\\gamma(0)/\\sigma^2 = 1 + 0.25 + 0.36 + 0.09 = (1 + 0.25)(1 + 0.36) = @{a3.g0}$', '2. $\\gamma(0)/\\sigma^2 = 1 + 0{,}25 + 0{,}36 + 0{,}09 = (1 + 0{,}25)(1 + 0{,}36) = @{a3.g0}$'),
               T('3. $\\rho_1 = -0.5/1.25 = @{a3.r1}$; $\\rho_4 = -0.6/1.36 = @{a3.r4}$; $\\rho_3 = \\rho_5 = \\rho_1\\rho_4 = @{a3.r3}$; $\\rho_2 = 0$', '3. $\\rho_1 = -0{,}5/1{,}25 = @{a3.r1}$; $\\rho_4 = -0{,}6/1{,}36 = @{a3.r4}$; $\\rho_3 = \\rho_5 = \\rho_1\\rho_4 = @{a3.r3}$; $\\rho_2 = 0$'),
               T('4. Spikes at lags 1 and 4 with small ``satellites\'\' at 3 and 5; nothing after lag 5: the signature of the airline model.', '4. Valori semnificative la decalajele 1 și 4, cu „sateliți” mici la 3 și 5; nimic după decalajul 5: semnătura modelului airline.')),
         size='scriptsize')

D.proposed(T('A4: a seasonal AR and moment estimates', 'A4: un AR sezonier și estimări prin metoda momentelor'),
           items(T('Model: A3.', 'Model: A3.'),
                 T('1. For $y_t = 0.7y_{t-4} + \\varepsilon_t$, compute $\\rho_4$, $\\rho_8$, $\\rho_{12}$ and say which other lags are zero.', '1. Pentru $y_t = 0{,}7y_{t-4} + \\varepsilon_t$, calculați $\\rho_4$, $\\rho_8$, $\\rho_{12}$ și precizați care alte decalaje sînt zero.'),
                 T('2. Describe its PACF.', '2. Descrieți PACF a acestui proces.'),
                 T('3. The sample ACF of $\\Delta\\Delta_{12}y_t$ is $-0.40$ at lag 1, $-0.45$ at lag 12, small at lags 11 and 13, and inside the band elsewhere; propose a model.', '3. ACF de selecție a lui $\\Delta\\Delta_{12}y_t$ este $-0{,}40$ la decalajul 1, $-0{,}45$ la decalajul 12, mică la decalajele 11 și 13 și în interiorul benzii în rest; propuneți un model.'),
                 T('4. Estimate $\\theta$ and $\\Theta$ from $\\rho_1$ and $\\rho_{12}$, keeping the invertible roots.', '4. Estimați $\\theta$ și $\\Theta$ din $\\rho_1$ și $\\rho_{12}$, păstrînd rădăcinile invertibile.'),
                 T('Report: three autocorrelations, a model and two estimates.', 'Raportați: trei autocorelații, un model și două estimări.')),
           items(T('1. $\\rho_4 = @{a4.r4}$, $\\rho_8 = @{a4.r8}$, $\\rho_{12} = @{a4.r12}$; all lags that are not multiples of 4 are zero', '1. $\\rho_4 = @{a4.r4}$, $\\rho_8 = @{a4.r8}$, $\\rho_{12} = @{a4.r12}$; toate decalajele care nu sînt multipli de 4 sînt zero'),
                 T('2. One spike, $\\phi_{44} = 0.7$, and zero elsewhere', '2. O singură valoare nenulă, $\\phi_{44} = 0{,}7$, și zero în rest'),
                 T('3. The airline model SARIMA$(0,1,1)(0,1,1)_{12}$ (A3 with $s = 12$)', '3. Modelul airline SARIMA$(0,1,1)(0,1,1)_{12}$ (A3 cu $s = 12$)'),
                 T('4. $\\theta/(1 + \\theta^2) = -0.40$: $\\hat\\theta = @{a4.th}$ (the other root, $-2$, is not invertible); $\\Theta/(1 + \\Theta^2) = -0.45$: $\\hat\\Theta = @{a4.Th}$', '4. $\\theta/(1 + \\theta^2) = -0{,}40$: $\\hat\\theta = @{a4.th}$ (cealaltă rădăcină, $-2$, nu este invertibilă); $\\Theta/(1 + \\Theta^2) = -0{,}45$: $\\hat\\Theta = @{a4.Th}$')),
           size='scriptsize')

D.solved(T('A5: MASE and the Diebold--Mariano test', 'A5: MASE și testul Diebold--Mariano'),
         items(T('Six non-overlapping forecast windows; errors of method 1: $0.4, -0.6, 0.9, -0.3, 0.5, -0.8$; of method 2: $1.1, -0.9, 0.7, -1.2, 0.8, -1.0$; the in-sample MAE of the seasonal naive method is 1.', 'Șase ferestre de prognoză care nu se suprapun; erorile metodei 1: $0{,}4; -0{,}6; 0{,}9; -0{,}3; 0{,}5; -0{,}8$; ale metodei 2: $1{,}1; -0{,}9; 0{,}7; -1{,}2; 0{,}8; -1{,}0$; MAE în eșantion a metodei sezoniere naive este 1.'),
               T('1. Compute the MAE and the MASE of both methods.', '1. Calculați MAE și MASE pentru ambele metode.'),
               T('2. Compute $d_t = e_{1t}^2 - e_{2t}^2$, $\\bar d$ and $\\hat\\sigma_d^2$.', '2. Calculați $d_t = e_{1t}^2 - e_{2t}^2$, $\\bar d$ și $\\hat\\sigma_d^2$.'),
               T('3. Compute DM and HLN.', '3. Calculați DM și HLN.'),
               T('4. Decide at 5\\% with $t(5)$.', '4. Decideți la 5\\% cu distribuția $t(5)$.'),
               T('Report: four accuracy measures, the test statistic and a decision.', 'Raportați: patru măsuri de acuratețe, statistica testului și o decizie.')),
         items(T('1. MAE: @{a5.mae1} and @{a5.mae2}; MASE: the same (scale 1): both below 1, method 1 better', '1. MAE: @{a5.mae1} și @{a5.mae2}; MASE: aceleași (scala 1): ambele sub 1, metoda 1 mai bună'),
               T('2. $d$: $@{a5.d1}, @{a5.d2}, @{a5.d3}, @{a5.d4}, @{a5.d5}, @{a5.d6}$; $\\bar d = @{a5.dbar}$; $\\hat\\sigma_d^2 = @{a5.s2}$', '2. $d$: $@{a5.d1}; @{a5.d2}; @{a5.d3}; @{a5.d4}; @{a5.d5}; @{a5.d6}$; $\\bar d = @{a5.dbar}$; $\\hat\\sigma_d^2 = @{a5.s2}$'),
               T('3. $\\mathrm{DM} = @{a5.dbar}/\\sqrt{@{a5.s2}/6} = @{a5.dm}$; $\\mathrm{HLN} = @{a5.corr} \\cdot (@{a5.dm}) = @{a5.hln}$', '3. $\\mathrm{DM} = @{a5.dbar}/\\sqrt{@{a5.s2}/6} = @{a5.dm}$; $\\mathrm{HLN} = @{a5.corr} \\cdot (@{a5.dm}) = @{a5.hln}$'),
               T('4. $|@{a5.hln}| < @{a5.tc}$ (p = @{a5.p}): equal accuracy is not rejected; six windows are too few to prove the visible gain.', '4. $|@{a5.hln}| < @{a5.tc}$ (p = @{a5.p}): acuratețea egală nu se respinge; șase ferestre sînt prea puține pentru a dovedi avantajul vizibil.')),
         size='scriptsize')

D.proposed(T('A6: two forecasts and their combination', 'A6: două prognoze și combinarea lor'),
           items(T('Eight windows; errors $e_1$: $2.0, -1.5, 1.0, -2.5, 3.0, 0.5, -1.0, 1.5$; $e_2$: $1.0, -2.5, 2.0, -0.5, 1.5, 2.0, -2.0, 0.5$. Model: A5.', 'Opt ferestre; erorile $e_1$: $2{,}0; -1{,}5; 1{,}0; -2{,}5; 3{,}0; 0{,}5; -1{,}0; 1{,}5$; $e_2$: $1{,}0; -2{,}5; 2{,}0; -0{,}5; 1{,}5; 2{,}0; -2{,}0; 0{,}5$. Model: A5.'),
                 T('1. Compute the MAE of both methods and the DM and HLN statistics with squared losses.', '1. Calculați MAE pentru ambele metode și statisticile DM și HLN cu pierderi pătratice.'),
                 T('2. Decide at 5\\% with $t(7)$.', '2. Decideți la 5\\% cu distribuția $t(7)$.'),
                 T('3. The equal-weight combination has errors $(e_1 + e_2)/2$: compute its MAE and RMSE.', '3. Combinarea cu ponderi egale are erorile $(e_1 + e_2)/2$: calculați MAE și RMSE ale acesteia.'),
                 T('4. For unbiased forecasts with $\\sigma_1 = 2$, $\\sigma_2 = 3$, $\\rho = 0.3$, compute $\\mathrm{MSE}(0.5)$, $w^*$ and $\\mathrm{MSE}(w^*)$.', '4. Pentru prognoze nedeplasate cu $\\sigma_1 = 2$, $\\sigma_2 = 3$, $\\rho = 0{,}3$, calculați $\\mathrm{MSE}(0{,}5)$, $w^*$ și $\\mathrm{MSE}(w^*)$.'),
                 T('Report: six numbers, a decision and three numbers for the combination.', 'Raportați: șase valori, o decizie și trei valori pentru combinare.')),
           items(T('1. MAE @{a6.mae1} and @{a6.mae2}; $\\bar d = @{a6.dbar}$, $\\hat\\sigma_d^2 = @{a6.s2}$, DM $= @{a6.dm}$, HLN $= @{a6.hln}$', '1. MAE @{a6.mae1} și @{a6.mae2}; $\\bar d = @{a6.dbar}$, $\\hat\\sigma_d^2 = @{a6.s2}$, DM $= @{a6.dm}$, HLN $= @{a6.hln}$'),
                 T('2. $|@{a6.hln}| < @{a6.tc}$ (p = @{a6.p}): no significant difference', '2. $|@{a6.hln}| < @{a6.tc}$ (p = @{a6.p}): nicio diferență semnificativă'),
                 T('3. Combination: MAE @{a6.cmae}, RMSE @{a6.crmse} (RMSE of the two: @{a6.rmse1}, @{a6.rmse2}): better than both, because the errors partly cancel', '3. Combinarea: MAE @{a6.cmae}, RMSE @{a6.crmse} (RMSE pentru cele două: @{a6.rmse1}; @{a6.rmse2}): mai bună decît ambele, pentru că erorile se compensează parțial'),
                 T('4. $\\mathrm{MSE}(0.5) = (4 + 9 + 2 \\cdot 0.3 \\cdot 6)/4 = @{a6.mh}$; $w^* = (9 - 1.8)/(13 - 3.6) = @{a6.w}$; $\\mathrm{MSE}(w^*) = @{a6.mo}$ (both below 4)', '4. $\\mathrm{MSE}(0{,}5) = (4 + 9 + 2 \\cdot 0{,}3 \\cdot 6)/4 = @{a6.mh}$; $w^* = (9 - 1{,}8)/(13 - 3{,}6) = @{a6.w}$; $\\mathrm{MSE}(w^*) = @{a6.mo}$ (ambele sub 4)')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: real data and interpretation', 'Partea B: date reale și interpretare')

D.task(T('B1: the airline model for Romanian GDP [Solved]', 'B1: modelul airline pentru PIB-ul României [Rezolvat]'),
       T('how stable is the seasonal pattern of Romanian GDP, and what does the airline model forecast for the next four quarters?', 'cît de stabil este tiparul sezonier al PIB-ului României și ce prognozează modelul airline pentru următoarele patru trimestre?'),
       T('Eurostat, real GDP not adjusted, 2000Q1--@{b1.last}, $T = @{b1.T}$; $y_t = 100\\ln Y_t$', 'Eurostat, PIB real neajustat, T1 2000--@{b1.last}, $T = @{b1.T}$; $y_t = 100\\ln Y_t$'),
       [T('Estimate SARIMA$(0,1,1)(0,1,1)_4$ by maximum likelihood and report $\\hat\\theta$, $\\hat\\Theta$ and their standard errors.', 'Estimați SARIMA$(0,1,1)(0,1,1)_4$ prin verosimilitate maximă și raportați $\\hat\\theta$, $\\hat\\Theta$ și erorile lor standard.'),
        T('Test the residuals with Ljung--Box $Q^*(8)$ on $8 - 2 = 6$ degrees of freedom.', 'Testați reziduurile cu Ljung--Box $Q^*(8)$, cu $8 - 2 = 6$ grade de libertate.'),
        T('Forecast the next four quarters, in bn EUR, with 95\\% intervals.', 'Prognozați următoarele patru trimestre, în mld. EUR, cu intervale de 95\\%.'),
        T('Compute the implied year-on-year growth for each forecast quarter.', 'Calculați creșterea anuală implicată pentru fiecare trimestru prognozat.'),
        T('Interpretation: what does $\\hat\\Theta$ say about how fast the seasonal pattern of GDP changes?', 'Interpretare: ce spune $\\hat\\Theta$ despre cît de repede se schimbă tiparul sezonier al PIB-ului?')],
       T('two estimates with standard errors, a test, four forecasts with intervals and one sentence', 'două estimări cu erorile standard, un test, patru prognoze cu intervale și o frază'), size='footnotesize', nb='B1')

D.frame(T('B1: solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch4_sem_b1', h='0.33') + items(
    T('$\\hat\\theta = @{b1.th}$ (@{b1.ths}): not significant; $\\hat\\Theta = @{b1.Th}$ (@{b1.Ths}); $\\hat\\sigma = @{b1.sig}\\%$; $Q^*(8) = @{b1.q}$, p = @{b1.qp}: no residual autocorrelation',
      '$\\hat\\theta = @{b1.th}$ (@{b1.ths}): nesemnificativ; $\\hat\\Theta = @{b1.Th}$ (@{b1.Ths}); $\\hat\\sigma = @{b1.sig}\\%$; $Q^*(8) = @{b1.q}$, p = @{b1.qp}: fără autocorelație în reziduuri'),
    T('Forecasts from @{b1.q0}: @{b1.f1}, @{b1.f2}, @{b1.f3}, @{b1.f4} bn EUR; first interval $[@{b1.lo1}, @{b1.hi1}]$, fourth $[@{b1.lo4}, @{b1.hi4}]$',
      'Prognozele din @{b1.q0}: @{b1.f1}; @{b1.f2}; @{b1.f3}; @{b1.f4} mld. EUR; primul interval $[@{b1.lo1}; @{b1.hi1}]$, al patrulea $[@{b1.lo4}; @{b1.hi4}]$'),
    T('Implied year-on-year growth: @{b1.g1}\\%, @{b1.g2}\\%, @{b1.g3}\\%, @{b1.g4}\\%', 'Creșterea anuală implicată: @{b1.g1}\\%; @{b1.g2}\\%; @{b1.g3}\\%; @{b1.g4}\\%'),
    T('Interpretation: the seasonal forecast is an exponentially weighted average of past seasonal patterns, with weight $1 + \\hat\\Theta = @{b1.al}$ on the newest year; $\\Theta = -1$ would mean a fixed pattern, $\\Theta = 0$ a pattern that changes freely: the pattern of GDP evolves slowly',
      'Interpretare: prognoza sezonieră este o medie ponderată exponențial a tiparelor sezoniere trecute, cu ponderea $1 + \\hat\\Theta = @{b1.al}$ pentru anul cel mai recent; $\\Theta = -1$ ar însemna un tipar fix, $\\Theta = 0$ un tipar care se schimbă liber: tiparul PIB-ului evoluează lent')) + qlsem(),
    'scriptsize')

D.task(T('B2: industrial production and working days [Proposed]', 'B2: producția industrială și zilele lucrătoare [Propus]'),
       T('does the number of working days explain part of the monthly movements of Romanian industrial production?', 'explică numărul zilelor lucrătoare o parte din mișcările lunare ale producției industriale din România?'),
       T('Eurostat, industrial production not adjusted, 2010--@{b2.last}, $y_t = 100\\ln Y_t$; dummies for April and May 2020; model: B1', 'Eurostat, producția industrială neajustată, 2010--@{b2.last}, $y_t = 100\\ln Y_t$; variabile dummy pentru aprilie și mai 2020; model: B1'),
       [T('Estimate the airline model with and without the number of working days as a regressor, and compare AICc and $Q^*(24)$.', 'Estimați modelul airline cu și fără numărul zilelor lucrătoare ca regresor și comparați AICc și $Q^*(24)$.'),
        T('Add two alternatives with working days, $(2,1,0)(0,1,1)_{12}$ and $(1,1,1)(0,1,1)_{12}$, and choose a model.', 'Adăugați două alternative cu zile lucrătoare, $(2,1,0)(0,1,1)_{12}$ și $(1,1,1)(0,1,1)_{12}$, și alegeți un model.'),
        T('Report the working-day coefficient with its standard error, and forecast the next 12 months.', 'Raportați coeficientul zilelor lucrătoare cu eroarea lui standard și prognozați următoarele 12 luni.'),
        T('Interpretation: why must the working-day effect be removed before two months are compared?', 'Interpretare: de ce trebuie eliminat efectul zilelor lucrătoare înainte de a compara două luni?')],
       T('a table of AICc and p-values, one coefficient, the forecast chart and one sentence', 'un tabel cu AICc și valori p, un coeficient, graficul prognozei și o frază'), size='footnotesize', nb='B2')

D.frame(T('B2: solution [Proposed]', 'B2: rezolvare [Propus]'), fig('ch4_sem_b2', h='0.30') + table(
    'lrr', T('Model', 'Modelul') + ' & AICc & ' + T('$Q^*(24)$, p', '$Q^*(24)$, p'),
    ['$(0,1,1)(0,1,1)_{12}$ & @{b2.a.aicc} & @{b2.a.p}', '$(0,1,1)(0,1,1)_{12}$ + WD & \\textbf{@{b2.aw.aicc}} & @{b2.aw.p}',
     '$(2,1,0)(0,1,1)_{12}$ & @{b2.b.aicc} & @{b2.b.p}', '$(2,1,0)(0,1,1)_{12}$ + WD & @{b2.bw.aicc} & @{b2.bw.p}',
     '$(1,1,1)(0,1,1)_{12}$ + WD & @{b2.cw.aicc} & @{b2.cw.p}'], size='scriptsize') + items(
    T('Best: airline + working days (WD); one more working day raises production by @{b2.wd}\\% (se @{b2.wds}); $\\hat\\theta = @{b2.th}$, $\\hat\\Theta = @{b2.Th}$; without WD the residuals fail Ljung--Box and $\\hat\\Theta$ drifts to @{b2.b.Th}',
      'Cel mai bun: airline + zile lucrătoare (WD); o zi lucrătoare în plus crește producția cu @{b2.wd}\\% (eroarea standard @{b2.wds}); $\\hat\\theta = @{b2.th}$, $\\hat\\Theta = @{b2.Th}$; fără WD reziduurile nu trec testul Ljung--Box, iar $\\hat\\Theta$ ajunge la @{b2.b.Th}'),
    T('Interpretation: a month with two more working days has about @{b2.wd2}\\% more output for purely calendar reasons; comparing months without this correction mistakes the calendar for the business cycle',
      'Interpretare: o lună cu două zile lucrătoare în plus are o producție cu aproximativ @{b2.wd2}\\% mai mare doar din motive de calendar; comparația fără această corecție confundă calendarul cu ciclul economic')),
    'scriptsize', instructor_only=True)

D.task(T('B3: cross-validation for daily electricity load [Solved]', 'B3: validare încrucișată pentru consumul zilnic de electricitate [Rezolvat]'),
       T('does a regression with Fourier terms and holiday dummies forecast daily load better than SARIMA and the seasonal naive method?', 'prognozează o regresie cu termeni Fourier și variabile dummy pentru sărbători consumul zilnic mai bine decît SARIMA și metoda sezonieră naivă?'),
       T('ENTSO-E, daily mean load in Romania (GW), 2022--2026; @{b3.n} origins every 28 days from @{b3.d0} to @{b3.d1}, horizon 14 days', 'ENTSO-E, consumul mediu zilnic în România (GW), 2022--2026; @{b3.n} origini la 28 de zile, de la @{b3.d0} la @{b3.d1}, orizont de 14 zile'),
       [T('At each origin, fit the seasonal naive method (period 7), SARIMA$(1,0,1)(0,1,1)_7$ and a DHR (4 annual Fourier pairs, weekday and holiday dummies, ARMA(1,1) errors).', 'La fiecare origine, estimați metoda sezonieră naivă (perioada 7), SARIMA$(1,0,1)(0,1,1)_7$ și o DHR (4 perechi Fourier anuale, variabile dummy pentru zilele săptămînii și sărbători, erori ARMA(1,1)).'),
        T('Compute the MAE and the MASE of each method over all origins and horizons.', 'Calculați MAE și MASE pentru fiecare metodă, pe toate originile și orizonturile.'),
        T('Test DHR against SARIMA with the DM test on the window-average squared errors (one value per origin), with the HLN correction.', 'Testați DHR față de SARIMA cu testul DM pe media erorilor pătratice din fiecare fereastră (o valoare pe origine), cu corecția HLN.'),
        T('Compare the MAE on public holidays with the MAE on ordinary days.', 'Comparați MAE în zilele de sărbătoare legală cu MAE în zilele obișnuite.'),
        T('Interpretation: why is the gain of the DHR largest on public holidays?', 'Interpretare: de ce este cîștigul DHR cel mai mare în zilele de sărbătoare legală?')],
       T('a table of MAE and MASE, one test, the chart and one sentence', 'un tabel cu MAE și MASE, un test, graficul și o frază'), size='footnotesize', nb='B3')

D.frame(T('B3: solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch4_sem_b3', h='0.32') + table(
    'lrrrr', T('Method', 'Metoda') + ' & MAE (MW) & MASE & ' + T('holidays', 'sărbători') + ' & ' + T('other days', 'alte zile'),
    [T('Seasonal naive', 'Sezonieră naivă') + ' & @{b3.sn.mae} & @{b3.sn.mase} & @{b3.sn.hol} & @{b3.sn.nh}',
     'SARIMA & @{b3.sa.mae} & @{b3.sa.mase} & @{b3.sa.hol} & @{b3.sa.nh}',
     'DHR & \\textbf{@{b3.dh.mae}} & \\textbf{@{b3.dh.mase}} & @{b3.dh.hol} & @{b3.dh.nh}'], size='scriptsize') + items(
    T('MASE scale: in-sample MAE of the weekly seasonal naive method, @{b3.scale} MW; DM (HLN), DHR against SARIMA: @{b3.dm}, p = @{b3.dmp}; against the seasonal naive method: @{b3.dmsn}, p = @{b3.dmsnp}',
      'Scala MASE: MAE în eșantion a metodei sezoniere naive săptămînale, @{b3.scale} MW; DM (HLN), DHR față de SARIMA: @{b3.dm}, p = @{b3.dmp}; față de metoda sezonieră naivă: @{b3.dmsn}, p = @{b3.dmsnp}'),
    T('Interpretation: SARIMA and the seasonal naive method only know ``one week ago\'\'; a holiday that falls on a weekday looks to them like a normal working day, while the DHR has dummies that remove @{b3.hol} GW on a public holiday and @{b3.eas} GW on the Easter days',
      'Interpretare: SARIMA și metoda sezonieră naivă cunosc doar „acum o săptămînă”; o sărbătoare care cade într-o zi lucrătoare arată pentru ele ca o zi obișnuită, în timp ce DHR are variabile dummy care scad @{b3.hol} GW într-o zi de sărbătoare legală și @{b3.eas} GW în zilele de Paște')) + qlsem(),
    'scriptsize')

D.task(T('B4: seasonality in monthly inflation [Proposed]', 'B4: sezonalitatea inflației lunare [Propus]'),
       T('is the seasonal pattern of Romanian monthly inflation strong enough to improve one-month forecasts?', 'este tiparul sezonier al inflației lunare din România suficient de puternic pentru a îmbunătăți prognozele pe o lună?'),
       T('HICP, Romania, 2010--2026; $\\pi_t = 100\\,\\Delta\\ln P_t$ (\\% per month); model: B3', 'IAPC, România, 2010--2026; $\\pi_t = 100\\,\\Delta\\ln P_t$ (\\% pe lună); model: B3'),
       [T('Regress $\\pi_t$ (2010--2019) on 12 month dummies and test equal month means with an $F$-test.', 'Regresați $\\pi_t$ (2010--2019) pe 12 variabile dummy lunare și testați egalitatea mediilor lunare cu un test $F$.'),
        T('Estimate the airline model for $100\\ln P_t$ and report $\\hat\\theta$ and $\\hat\\Theta$.', 'Estimați modelul airline pentru $100\\ln P_t$ și raportați $\\hat\\theta$ și $\\hat\\Theta$.'),
        T('From monthly origins since January 2016, forecast next month\'s $\\pi$ with SARIMA, the seasonal naive method and the mean of the last 12 months; compare the MAE.', 'Din origini lunare începînd cu ianuarie 2016, prognozați $\\pi$ din luna următoare cu SARIMA, metoda sezonieră naivă și media ultimelor 12 luni; comparați MAE.'),
        T('Test SARIMA against both benchmarks with the DM test (squared errors).', 'Testați SARIMA față de ambele repere cu testul DM (erori pătratice).'),
        T('Interpretation: is the seasonal pattern strong enough to improve one-month forecasts?', 'Interpretare: este tiparul sezonier suficient de puternic pentru a îmbunătăți prognozele pe o lună?')],
       T('an $F$-test, two estimates, three MAE values, two DM tests and one sentence', 'un test $F$, două estimări, trei valori MAE, două teste DM și o frază'), size='footnotesize', nb='B4')

D.frame(T('B4: solution [Proposed]', 'B4: rezolvare [Propus]'), fig('ch4_sem_b4', h='0.30') + items(
    T('$F = @{b4.F}$, p = @{b4.pF}: the month means differ; January @{b4.jan}\\%, June @{b4.jun}\\% (fresh food), August @{b4.aug}\\%, October @{b4.oct}\\%',
      '$F = @{b4.F}$, p = @{b4.pF}: mediile lunare diferă; ianuarie @{b4.jan}\\%, iunie @{b4.jun}\\% (alimente proaspete), august @{b4.aug}\\%, octombrie @{b4.oct}\\%'),
    T('Airline: $\\hat\\theta = @{b4.th}$ (@{b4.ths}), $\\hat\\Theta = @{b4.Th}$ (@{b4.Ths}); @{b4.n} forecasts, @{b4.d0}--@{b4.d1}: MAE SARIMA @{b4.sa}, seasonal naive @{b4.sn}, mean of 12 months @{b4.m12}',
      'Airline: $\\hat\\theta = @{b4.th}$ (@{b4.ths}), $\\hat\\Theta = @{b4.Th}$ (@{b4.Ths}); @{b4.n} prognoze, @{b4.d0}--@{b4.d1}: MAE SARIMA @{b4.sa}, sezonieră naivă @{b4.sn}, media pe 12 luni @{b4.m12}'),
    T('DM (HLN): SARIMA against the seasonal naive method @{b4.dms} (p = @{b4.dmsp}); against the 12-month mean @{b4.dmm} (p = @{b4.dmmp})',
      'DM (HLN): SARIMA față de metoda sezonieră naivă @{b4.dms} (p = @{b4.dmsp}); față de media pe 12 luni @{b4.dmm} (p = @{b4.dmmp})'),
    T('Interpretation: the seasonal pattern is real but small next to the monthly noise; SARIMA beats copying last year\'s month, yet it does not beat the simple 12-month mean',
      'Interpretare: tiparul sezonier există, dar este mic în raport cu zgomotul lunar; SARIMA bate copierea lunii de anul trecut, dar nu bate media simplă pe 12 luni')),
    'scriptsize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: open questions and AI critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: tourism after the pandemic [Proposed]', 'C1: turismul după pandemie [Propus]'),
       T('which training sample gives better forecasts of tourism nights for 2023--2025: data up to 2019, or data up to 2022?', 'ce eșantion de antrenare dă prognoze mai bune ale înnoptărilor turistice pentru 2023--2025: datele pînă în 2019 sau datele pînă în 2022?'),
       T('Eurostat, nights in tourist accommodation, Romania, monthly, 2005--2025; model: B1', 'Eurostat, înnoptări în unitățile de cazare turistică, România, lunar, 2005--2025; model: B1'),
       [T('Fit the airline model to $\\ln y_t$ up to December 2019 and forecast 2023--2025.', 'Estimați modelul airline pentru $\\ln y_t$ pînă în decembrie 2019 și prognozați 2023--2025.'),
        T('Repeat with data up to December 2022.', 'Repetați cu datele pînă în decembrie 2022.'),
        T('Compare both with the seasonal naive forecast that repeats 2022, using the MAPE (mean absolute percentage error).', 'Comparați ambele cu prognoza sezonieră naivă care repetă anul 2022, folosind MAPE (eroarea procentuală absolută medie).'),
        T('Propose a better treatment of 2020--2021 (intervention dummies, a shorter sample, TBATS or Prophet) and sketch it as a team project.', 'Propuneți un tratament mai bun pentru 2020--2021 (variabile dummy de intervenție, un eșantion mai scurt, TBATS sau Prophet) și schițați-l ca proiect de echipă.')],
       T('three MAPE values, the chart and a project plan', 'trei valori MAPE, graficul și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: reference analysis [Proposed]', 'C1: analiză de referință [Propus]'), fig('ch4_sem_c1', h='0.36') + items(
    T('MAPE 2023--2025: airline trained to 2019 @{c1.m19}\\%; trained to 2022 @{c1.m22}\\%; seasonal naive (2022 repeated) @{c1.sn}\\%; nights: @{c1.y19} million in 2019, @{c1.y25} million in 2025',
      'MAPE 2023--2025: airline estimat pînă în 2019 @{c1.m19}\\%; estimat pînă în 2022 @{c1.m22}\\%; sezonieră naivă (2022 repetat) @{c1.sn}\\%; înnoptări: @{c1.y19} milioane în 2019, @{c1.y25} milioane în 2025'),
    T('Interpretation: the collapse of 2020 enters the differenced data as huge shocks, and the model trained to 2022 extrapolates the rebound; the old model ignores the break; the simplest forecast wins. A project: intervention analysis with dummies for 2020--2021, compared by rolling origins',
      'Interpretare: prăbușirea din 2020 intră în datele diferențiate ca șocuri foarte mari, iar modelul estimat pînă în 2022 extrapolează revenirea; modelul vechi ignoră ruptura; cea mai simplă prognoză cîștigă. Un proiect: analiza intervenției cu variabile dummy pentru 2020--2021, comparată cu origini mobile')),
    'scriptsize', instructor_only=True)

D.frame(T('C2: audit an AI answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant about seasonal models. The answer:', 'Un student a cerut unui asistent AI informații despre modelele sezoniere. Răspunsul:'),
    T('\\aiprompt{(a) SARIMA(0,1,1)(0,1,1)4 has four MA parameters, at lags 1, 4 and 5 plus the noise variance.}', '\\aiprompt{(a) SARIMA(0,1,1)(0,1,1)4 are patru parametri MA, la decalajele 1, 4 și 5, plus varianța zgomotului.}'),
    T('\\aiprompt{(b) If the ACF of the differenced monthly series has one spike at lag 12 and the PACF decays at 12, 24, 36, use a seasonal AR(1).}', '\\aiprompt{(b) Dacă ACF a seriei lunare diferențiate are o singură valoare semnificativă la decalajul 12, iar PACF descrește la 12, 24, 36, folosiți un AR(1) sezonier.}'),
    T('\\aiprompt{(c) Unadjusted Romanian GDP fell by about 34\\% in the first quarter of 2026 compared with the fourth quarter of 2025: a deep recession.}', '\\aiprompt{(c) PIB-ul neajustat al României a scăzut cu aproximativ 34\\% în primul trimestru din 2026 față de al patrulea trimestru din 2025: o recesiune profundă.}'),
    T('\\aiprompt{(d) DM = -2.3 with d = L(e1) - L(e2) shows that forecast 2 is significantly more accurate.}', '\\aiprompt{(d) DM = -2,3 cu d = L(e1) - L(e2) arată că prognoza 2 este semnificativ mai precisă.}'),
    T('\\aiprompt{(e) TBATS with periods 7 and 365.25 captures the Easter dip of electricity load.}', '\\aiprompt{(e) TBATS cu perioadele 7 și 365,25 surprinde scăderea consumului de electricitate de Paște.}'),
    T('\\aiprompt{(f) The equal-weight average of two unbiased forecasts never has a larger MSE than the average of their two MSEs.}', '\\aiprompt{(f) Media cu ponderi egale a două prognoze nedeplasate nu are niciodată un MSE mai mare decît media celor două MSE.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu, dați afirmația corectă și, unde se poate, valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])), 'scriptsize')

D.frame(T('C2: solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: two MA parameters, $\\theta$ and $\\Theta$; the lag-5 coefficient is the product $\\theta\\Theta$ (A2, A3)', '(a) Greșit: doi parametri MA, $\\theta$ și $\\Theta$; coeficientul de la decalajul 5 este produsul $\\theta\\Theta$ (A2, A3)'),
    T('(b) Wrong: an ACF that cuts off at lag 12 with a decaying seasonal PACF is a seasonal MA(1) (A4)', '(b) Greșit: o ACF care se anulează după decalajul 12, cu PACF sezonieră descrescătoare, indică un MA(1) sezonier (A4)'),
    T('(c) Wrong: the fall is seasonal (Q1 is always the lowest quarter): unadjusted @{c2.nsa}\\%, adjusted (SCA) @{c2.sca}\\% in @{c2.q}', '(c) Greșit: scăderea este sezonieră (T1 este mereu cel mai slab trimestru): neajustat @{c2.nsa}\\%, ajustat (SCA) @{c2.sca}\\% în @{c2.q}'),
    T('(d) Wrong: $\\bar d < 0$ favours forecast 1; significance also needs $|\\mathrm{HLN}|$ above the $t(n - 1)$ critical value (A5)', '(d) Greșit: $\\bar d < 0$ favorizează prognoza 1; semnificația cere și ca $|\\mathrm{HLN}|$ să depășească valoarea critică $t(n - 1)$ (A5)'),
    T('(e) Wrong: Orthodox Easter moves between April and May; a fixed period cannot follow it; it needs a holiday regressor, as in the DHR of B3', '(e) Greșit: Paștele ortodox se mută între aprilie și mai; o perioadă fixă nu îl poate urmări; este nevoie de un regresor de sărbătoare, ca în DHR din B3'),
    T('(f) Correct: $\\mathrm{MSE}(0.5) = (\\sigma_1^2 + \\sigma_2^2 + 2\\rho\\sigma_1\\sigma_2)/4 \\le (\\sigma_1^2 + \\sigma_2^2)/2$, because $2\\rho\\sigma_1\\sigma_2 \\le \\sigma_1^2 + \\sigma_2^2$ (A6)', '(f) Corect: $\\mathrm{MSE}(0{,}5) = (\\sigma_1^2 + \\sigma_2^2 + 2\\rho\\sigma_1\\sigma_2)/4 \\le (\\sigma_1^2 + \\sigma_2^2)/2$, deoarece $2\\rho\\sigma_1\\sigma_2 \\le \\sigma_1^2 + \\sigma_2^2$ (A6)')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-up', 'Încheiere')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('$\\Delta_s$ removes a stable seasonal pattern; $\\Delta\\Delta_s$ also removes the trend; the seasonal naive forecast is the benchmark', '$\\Delta_s$ elimină un tipar sezonier stabil; $\\Delta\\Delta_s$ elimină și trendul; prognoza sezonieră naivă este reperul'),
    T('SARIMA multiplies a regular and a seasonal polynomial: the airline model has two parameters and ACF spikes at $1$, $s - 1$, $s$, $s + 1$', 'SARIMA înmulțește un polinom obișnuit și unul sezonier: modelul airline are doi parametri și valori nenule ale ACF la $1$, $s - 1$, $s$, $s + 1$'),
    T('Calendar effects (working days, Easter, public holidays) are known in advance: put them in as regressors', 'Efectele de calendar (zile lucrătoare, Paște, sărbători legale) sînt cunoscute dinainte: introduceți-le ca regresori'),
    T('Compare forecasts out of sample, with rolling origins, MASE and the DM test; a visible gain is not always a significant one', 'Comparați prognozele în afara eșantionului, cu origini mobile, MASE și testul DM; un cîștig vizibil nu este întotdeauna unul semnificativ'),
    T('Combining forecasts is cheap insurance: it is rarely the worst and often near the best', 'Combinarea prognozelor este o asigurare ieftină: rareori este cea mai slabă și deseori este aproape de cea mai bună')))

D.frame(T('After the seminar', 'După seminar'), items(
    T('Lecture 4 derives today\'s formulas and adds seasonal unit roots, seasonal adjustment (X-13ARIMA-SEATS), multiple seasonality, TBATS and Prophet',
      'Cursul 4 deduce formulele de azi și adaugă rădăcinile unitare sezoniere, ajustarea sezonieră (X-13ARIMA-SEATS), sezonalitatea multiplă, TBATS și Prophet'),
    T('Try the [Proposed] tasks in the notebook; TBATS and Prophet are optional installs there', 'Încercați cerințele [Propus] în notebook; TBATS și Prophet se instalează opțional acolo'),
    T('C1 can grow into a team project: forecasting tourism or electricity load in Romania with a rolling-origin comparison of several models', 'C1 poate deveni un proiect de echipă: prognoza turismului sau a consumului de electricitate din România, cu o comparație pe origini mobile a mai multor modele'),
    T('Reading: \\refHP, Ch.~4--5; \\refFPPsar; \\refFPPcv; \\refDM', 'Lectură: \\refHP, cap.~4--5; \\refFPPsar; \\refFPPcv; \\refDM'),
    T('\\textbf{The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class}',
      '\\textbf{Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar}')))

D.references(bib(['BG', 'BJ', 'DM', 'FPP', 'HLN', 'HP', 'HKo', 'LB', 'Tashman']), per=16)

if __name__ == '__main__':
    finalize(D.write(V))
