r"""
build_seminar10.py -- Seminarul 10 (Modele în spațiul stărilor, filtrul Kalman și modele Markov switching), EN + RO
=================================================================================================================
Seminarul are loc ÎNAINTEA cursului 10: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu o întrebare de interpretare, C o întrebare deschisă și
critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în versiunea
profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_10/sem10_results.json (seminar10.py).
Ieșire:
  EN/Seminars/seminar10_state_space_kalman_markov_switching.tex             (+ _solutions.tex)
  RO/Seminarii/seminar10_spatiul_starilor_kalman_markov_switching_ro.tex    (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_10/seminar10.py && python3 latex/build_seminar10.py && python3 latex/tsa_build.py compile 10
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, fig   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch10_common import REFS, T, bib, finalize, load_sem, month   # noqa: E402

S = load_sem()
V = Values()
D = Deck(10, 'seminar', refs=REFS)
P = V.put


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


def qlsem():
    return '\\quantlet{TSA\\_ch10\\_seminar}{\\qlurl{TSA_ch10_seminar}}'


# =============================================================================
# CIFRE
# =============================================================================
for tag in ('A1', 'A2'):
    for r in S[tag]['rows']:
        t = r['t']
        for k in ('y', 'a', 'P', 'F', 'K', 'v', 'af', 'Pf'):
            if r[k] is not None:
                P(f'{tag.lower()}.{t}.{k}', r[k], 3 if k == 'K' else 2)
    P(f'{tag.lower()}.anext', S[tag]['a_next'], 2)
    P(f'{tag.lower()}.Pnext', S[tag]['P_next'], 2)
    P(f'{tag.lower()}.ssK', S[tag]['ss_K'], 3)
    P(f'{tag.lower()}.ssP', S[tag]['ss_P'], 3)
A3 = S['A3']
P('a3.P1', A3['K']['1.0']['P'], 3)
P('a3.K1', A3['K']['1.0']['K'], 3)
P('a3.P025', A3['K']['0.25']['P'], 3)
P('a3.K025', A3['K']['0.25']['K'], 3)
P('a3.q03', A3['q']['0.3'], 3)
P('a3.q05', A3['q']['0.5'], 2)
A4 = S['A4']
for i, x in enumerate(A4['ar'], 1):
    P(f'a4.ar{i}', x, 3)
for i, x in enumerate(A4['llt'], 1):
    P(f'a4.llt{i}', x, 0)
A5 = S['A5']
P('a5.d1', A5['d1'], 0)
P('a5.d2', A5['d2'], 0)
P('a5.pi1', A5['pi1'], 3)
P('a5.pi2', A5['pi2'], 3)
P('a5.p112', A5['p11_2'], 3)
P('a5.p212', A5['p21_2'], 3)
A6 = S['A6']
for k in ('pred1', 'f1', 'f2', 'lik', 'filt1', 'next_pred1', 'loglik'):
    P(f'a6.{k}', A6[k], 3)

B1 = S['B1']
P('b1.se', B1['s2_eps'], 3)
P('b1.sn', B1['s2_eta'], 4)
P('b1.q', B1['q'], 3)
P('b1.K', B1['K'], 3)
P('b1.ses', B1['alpha_ses'], 3)
P('b1.peak', B1['peak'], 1)
V.raw('b1.peakd', month(B1['peak_d']))
P('b1.low', B1['low'], 1)
V.raw('b1.lowd', month(B1['low_d']))
P('b1.last', B1['last_s'], 1)
P('b1.sdlast', B1['sd_last'], 1)
P('b1.rawsd', B1['raw_sd'], 1)
V.raw('b1.n', str(B1['n']))
V.raw('b1.lastd', month(B1['last']))
B2 = S['B2']
for k in ('c_uc', 'c_hp', 'c_ham', 'c_ucf'):
    P(f'b2.{k}', B2[k], 2)
for k in ('sd_cbo', 'sd_uc', 'sd_hp', 'sd_ham', 'last_cbo', 'last_uc', 'last_hp', 'last_ham'):
    P(f'b2.{k}', B2[k], 1)
B3 = S['B3']
for k in ('p11', 'p22'):
    P(f'b3.{k}', B3[k], 3)
P('b3.d1', B3['dur1'], 1)
P('b3.d2', B3['dur2'], 1)
P('b3.m1', B3['const_1'], 2)
P('b3.m2', B3['const_2'], 2)
P('b3.s1', B3['sd1'], 2)
P('b3.s2', B3['sd2'], 2)
P('b3.erg', 100 * B3['ergodic1'], 0)
for k in ('concord', 'hit', 'false'):
    P(f'b3.c.{k}', 100 * B3['conc'][k], 0)
    P(f'b3.f.{k}', 100 * B3['conc_f'][k], 0)
V.raw('b3.nrec', str(B3['n_rec']))
B4 = S['B4']
P('b4.d1', B4['dur1'], 0)
P('b4.d2', B4['dur2'], 0)
P('b4.s1', B4['sd1_ann'], 0)
P('b4.s2', B4['sd2_ann'], 0)
P('b4.gab', B4['garch_ab'], 2)
P('b4.cg', B4['corr_garch'], 2)
P('b4.cs', B4['corr_sp'], 2)
P('b4.agree', 100 * B4['agree_sp'], 0)
P('b4.share', 100 * B4['share_turb'], 0)
TY = list(B4['top_years'].items())
for i, (y, s) in enumerate(TY[:3]):
    V.raw(f'b4.y{i}', y)
    P(f'b4.ys{i}', 100 * s, 0)
C1 = S['C1']
for lab in ('low', 'medium', 'high'):
    P(f'c1.{lab}.m', C1[lab]['mean'], 2)
    P(f'c1.{lab}.s', C1[lab]['sd'], 2)
    P(f'c1.{lab}.d', C1[lab]['dur'], 1)
    P(f'c1.{lab}.share', 100 * C1[lab]['share'], 0)
P('c1.ann', C1['annual_low'], 1)
P('c1.bic2', C1['bic2'], 1)
P('c1.bic3', C1['bic3'], 1)
V.raw('c1.lowyear', str(C1['low_all_year']))
P('b4.bet22', 100 * B4['bet_2022'], 0)
P('b4.sp22', 100 * B4['sp_2022'], 0)
V.raw('c1.firstlow', month(C1['first_low']))
C2 = S['C2']
P('c2.K', C2['K_nile'], 3)
for _k, _v in list(V.items()):                 # negative numbers: a real minus sign in text and in math mode
    if isinstance(_v, str) and _v.startswith('⁅-'):
        V[_k] = '⁅\\ensuremath{-}' + _v[2:]

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how do we estimate a quantity we never observe (a level, a trend, a regime) from noisy data, one period at a time?',
       '\\textbf{Întrebarea}: cum estimăm o mărime pe care nu o observăm niciodată (un nivel, un trend, un regim) din date zgomotoase, perioadă cu perioadă?'),
     [T('this seminar comes \\textbf{before} Lecture 10: the section ``What you need today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 10: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: the Kalman filter by hand; exponential smoothing as a steady state; state space forms; durations; the Hamilton filter',
        'Partea A: filtrul Kalman de mînă; netezirea exponențială ca stare de echilibru; forme în spațiul stărilor; durate; filtrul Hamilton'),
      T('Part B: Romanian inflation; the US output gap; US industrial production and recessions; BET volatility regimes', 'Partea B: inflația din România; deviația PIB a SUA; producția industrială din SUA și recesiunile; regimurile de volatilitate ale BET'),
      T('Part C: regimes of Romanian inflation since 1996; an AI answer to audit', 'Partea C: regimurile inflației din România din 1996; un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('the Kalman filter of a local level model for three periods; a missing value', 'filtrul Kalman al unui model local level pentru trei perioade; o valoare lipsă') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('steady state and exponential smoothing; state space forms and forecasts', 'starea de echilibru și netezirea exponențială; forme în spațiul stărilor și prognoze') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('durations of regimes; one step of the Hamilton filter', 'duratele regimurilor; un pas al filtrului Hamilton') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('underlying Romanian inflation; the US output gap', 'inflația de fond din România; deviația PIB a SUA') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('recessions in US industrial production; BET volatility regimes', 'recesiunile în producția industrială din SUA; regimurile de volatilitate ale BET') + ' & ' + SP + ' & B3',
     'C1, C2 & ' + T('Romanian inflation regimes; what is wrong in an AI answer?', 'regimurile inflației din România; ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B3, A1--A6'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați dumneavoastră, după model')))

D.frame(T('Data used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Frequency}', '\\textbf{Frecvența}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    [T('HICP, Romania', 'IAPC, România') + ' & Eurostat (prc\\_hicp\\_minr) & ' + T('monthly, 2015 = 100', 'lunar, 2015 = 100') + ' & 1996--2026',
     T('real GDP and potential GDP, US', 'PIB real și PIB potențial, SUA') + ' & FRED (GDPC1, GDPPOT) & ' + T('quarterly', 'trimestrial') + ' & 1947--2026',
     T('industrial production, US', 'producția industrială, SUA') + ' & FRED (INDPRO) & ' + T('monthly', 'lunar') + ' & 1960--2019',
     T('NBER recession indicator', 'indicatorul de recesiune NBER') + ' & FRED (USREC) & ' + T('monthly', 'lunar') + ' & 1960--2019',
     'BET, S\\&P 500 & EODHD & ' + T('weekly, from daily closes', 'săptămînal, din închideri zilnice') + ' & 2000--2026'],
    size='footnotesize') + items(
    T('Monthly inflation: $100\\Delta\\ln P_t$, seasonally adjusted (SA) by removing the monthly means (Chapter 4); CBO output gap: $100(\\mathrm{GDP}_t/\\mathrm{GDPPOT}_t - 1)$',
      'Inflația lunară: $100\\Delta\\ln P_t$, ajustată sezonier (SA) prin eliminarea mediilor lunare (Capitolul 4); deviația PIB a CBO: $100(\\mathrm{PIB}_t/\\mathrm{GDPPOT}_t - 1)$'),
    T('In the notebook: \\texttt{kalman\\_by\\_hand}, \\texttt{local\\_level\\_ml}, \\texttt{uc\\_fit}, \\texttt{hamilton\\_filter}, \\texttt{MarkovRegression}; no account or key is needed',
      'În notebook: \\texttt{kalman\\_by\\_hand}, \\texttt{local\\_level\\_ml}, \\texttt{uc\\_fit}, \\texttt{hamilton\\_filter}, \\texttt{MarkovRegression}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What you need today', 'Noțiuni necesare azi')

D.frame(T('What you need today (1/4): the state space form', 'Noțiuni necesare azi (1/4): forma în spațiul stărilor'), items(
    (T('\\textbf{State space form}: $y_t = Z\\alpha_t + \\varepsilon_t$ (measurement), $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$ (transition); $\\varepsilon_t \\sim N(0, H)$, $\\eta_t \\sim N(0, Q)$',
       '\\textbf{Forma în spațiul stărilor}: $y_t = Z\\alpha_t + \\varepsilon_t$ (măsurare), $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$ (tranziție); $\\varepsilon_t \\sim N(0, H)$, $\\eta_t \\sim N(0, Q)$'),
     [T('$\\alpha_t$: the \\textbf{state}, not observed; forecasts iterate the transition: $\\hat\\alpha_{T+h} = T^h\\hat\\alpha_T$', '$\\alpha_t$: \\textbf{starea}, neobservată; prognozele iterează tranziția: $\\hat\\alpha_{T+h} = T^h\\hat\\alpha_T$')]),
    (T('\\textbf{Local level}: $y_t = \\mu_t + \\varepsilon_t$, $\\mu_{t+1} = \\mu_t + \\eta_t$; signal-to-noise ratio $q = \\sigma^2_\\eta/\\sigma^2_\\varepsilon$', '\\textbf{Local level}: $y_t = \\mu_t + \\varepsilon_t$, $\\mu_{t+1} = \\mu_t + \\eta_t$; raportul semnal--zgomot $q = \\sigma^2_\\eta/\\sigma^2_\\varepsilon$'),
     [T('\\textbf{local linear trend}: $\\mu_{t+1} = \\mu_t + \\beta_t + \\eta_t$, $\\beta_{t+1} = \\beta_t + \\zeta_t$; state $(\\mu_t, \\beta_t)\'$', '\\textbf{local linear trend}: $\\mu_{t+1} = \\mu_t + \\beta_t + \\eta_t$, $\\beta_{t+1} = \\beta_t + \\zeta_t$; starea $(\\mu_t, \\beta_t)\'$')]),
    (T('\\textbf{AR(2)} in state space form: $\\alpha_t = (y_t, y_{t-1})\'$, $T = \\begin{pmatrix}\\phi_1 & \\phi_2\\\\ 1 & 0\\end{pmatrix}$, $Z = (1,\\ 0)$, $H = 0$', '\\textbf{AR(2)} în forma în spațiul stărilor: $\\alpha_t = (y_t, y_{t-1})\'$, $T = \\begin{pmatrix}\\phi_1 & \\phi_2\\\\ 1 & 0\\end{pmatrix}$, $Z = (1,\\ 0)$, $H = 0$'),
     [T('\\textbf{simple exponential smoothing} (SES, Chapter 0): $\\ell_t = \\alpha y_t + (1-\\alpha)\\ell_{t-1}$, forecast $\\hat y_{t+1} = \\ell_t$', '\\textbf{netezirea exponențială simplă} (SES, Capitolul 0): $\\ell_t = \\alpha y_t + (1-\\alpha)\\ell_{t-1}$, prognoza $\\hat y_{t+1} = \\ell_t$')])))

D.frame(T('What you need today (2/4): the Kalman filter of the local level model', 'Noțiuni necesare azi (2/4): filtrul Kalman al modelului local level'), items(
    (T('Start with a prediction $a_t$ of $\\mu_t$ and its variance $P_t$; when $y_t$ arrives \\refKalman:', 'Pornim de la o predicție $a_t$ a lui $\\mu_t$ și varianța ei $P_t$; cînd sosește $y_t$ \\refKalman:'),
     [T('$v_t = y_t - a_t$ (surprise), \\quad $F_t = P_t + \\sigma^2_\\varepsilon$, \\quad $K_t = P_t/F_t$ (\\textbf{Kalman gain})', '$v_t = y_t - a_t$ (surpriza), \\quad $F_t = P_t + \\sigma^2_\\varepsilon$, \\quad $K_t = P_t/F_t$ (\\textbf{cîștigul Kalman})'),
      T('update: $a_{t|t} = a_t + K_tv_t$, \\quad $P_{t|t} = P_t(1 - K_t)$', 'actualizarea: $a_{t|t} = a_t + K_tv_t$, \\quad $P_{t|t} = P_t(1 - K_t)$'),
      T('prediction: $a_{t+1} = a_{t|t}$, \\quad $P_{t+1} = P_{t|t} + \\sigma^2_\\eta$', 'predicția: $a_{t+1} = a_{t|t}$, \\quad $P_{t+1} = P_{t|t} + \\sigma^2_\\eta$')]),
    (T('\\textbf{Missing} $y_t$: no update, $a_{t|t} = a_t$, $P_{t|t} = P_t$; forecasts of every horizon equal $a_{n+1}$', '\\textbf{Lipsește} $y_t$: fără actualizare, $a_{t|t} = a_t$, $P_{t|t} = P_t$; prognozele pentru orice orizont sînt egale cu $a_{n+1}$'),
     [T('\\textbf{steady state}: $\\bar P = \\sigma^2_\\varepsilon(q + \\sqrt{q^2 + 4q})/2$, $\\bar K = \\bar P/(\\bar P + \\sigma^2_\\varepsilon)$; then $a_{t+1} = \\bar Ky_t + (1 - \\bar K)a_t$ is SES with $\\alpha = \\bar K$ \\refMuth',
        '\\textbf{starea de echilibru}: $\\bar P = \\sigma^2_\\varepsilon(q + \\sqrt{q^2 + 4q})/2$, $\\bar K = \\bar P/(\\bar P + \\sigma^2_\\varepsilon)$; atunci $a_{t+1} = \\bar Ky_t + (1 - \\bar K)a_t$ este SES cu $\\alpha = \\bar K$ \\refMuth'),
      T('inverse map: $q = \\alpha^2/(1 - \\alpha)$', 'relația inversă: $q = \\alpha^2/(1 - \\alpha)$')])))

D.frame(T('What you need today (3/4): likelihood, smoothing, trend and cycle', 'Noțiuni necesare azi (3/4): verosimilitate, netezire, trend și ciclu'), items(
    (T('\\textbf{Likelihood} (prediction-error decomposition): $\\log L = -\\frac12\\sum_t(\\log 2\\pi + \\log F_t + v_t^2/F_t)$, maximised over the variances', '\\textbf{Verosimilitatea} (descompunerea erorilor de predicție): $\\log L = -\\frac12\\sum_t(\\log 2\\pi + \\log F_t + v_t^2/F_t)$, maximizată după varianțe'),
     [T('\\textbf{filtered} $a_{t|t}$ uses $y_1..y_t$ (real time); \\textbf{smoothed} $\\hat\\mu_t$ uses the whole sample (history) \\refDK', '\\textbf{filtrat} $a_{t|t}$ folosește $y_1..y_t$ (timp real); \\textbf{netezit} $\\hat\\mu_t$ folosește tot eșantionul (istorie) \\refDK')]),
    (T('\\textbf{Output gap}: the cycle $\\psi_t$ in $100\\log\\mathrm{GDP}_t = \\mu_t + \\psi_t$ (trend plus cycle)', '\\textbf{Deviația PIB} (output gap): ciclul $\\psi_t$ din $100\\log\\mathrm{PIB}_t = \\mu_t + \\psi_t$ (trend plus ciclu)'),
     [T('\\textbf{UC model} \\refHarvey: smooth trend ($\\Delta^2\\mu_{t+1} = \\zeta_t$) plus an AR(2) cycle, estimated by the Kalman filter', '\\textbf{modelul UC} \\refHarvey: trend neted ($\\Delta^2\\mu_{t+1} = \\zeta_t$) plus un ciclu AR(2), estimat cu filtrul Kalman'),
      T('\\textbf{HP filter} \\refHPf: $\\min\\sum(y_t - \\tau_t)^2 + 1600\\sum(\\Delta^2\\tau_t)^2$, a two-sided smoother', '\\textbf{filtrul HP} \\refHPf: $\\min\\sum(y_t - \\tau_t)^2 + 1600\\sum(\\Delta^2\\tau_t)^2$, un netezitor bilateral'),
      T('\\textbf{Hamilton filter} \\refHamHP: residual of OLS of $y_{t+8}$ on $1, y_t, \\dots, y_{t-3}$', '\\textbf{filtrul Hamilton} \\refHamHP: reziduul regresiei OLS a lui $y_{t+8}$ pe $1, y_t, \\dots, y_{t-3}$')])))

D.frame(T('What you need today (4/4): Markov switching', 'Noțiuni necesare azi (4/4): modele Markov switching'), items(
    (T('\\textbf{Markov switching} \\refHamMS: $y_t = \\mu_{S_t} + \\varepsilon_t$, $\\varepsilon_t \\sim N(0, \\sigma^2_{S_t})$, regime $S_t \\in \\{1, 2\\}$ follows a Markov chain', '\\textbf{Markov switching} \\refHamMS: $y_t = \\mu_{S_t} + \\varepsilon_t$, $\\varepsilon_t \\sim N(0, \\sigma^2_{S_t})$, regimul $S_t \\in \\{1, 2\\}$ urmează un lanț Markov'),
     [T('$p_{ij} = \\Pr(S_t = j \\mid S_{t-1} = i)$; \\textbf{expected duration} of regime $i$: $1/(1 - p_{ii})$', '$p_{ij} = \\Pr(S_t = j \\mid S_{t-1} = i)$; \\textbf{durata așteptată} a regimului $i$: $1/(1 - p_{ii})$'),
      T('\\textbf{ergodic} probability $\\pi_1 = (1 - p_{22})/(2 - p_{11} - p_{22})$; $h$-step transitions: $\\mathbf{P}^h$', 'probabilitatea \\textbf{ergodică} $\\pi_1 = (1 - p_{22})/(2 - p_{11} - p_{22})$; tranziții în $h$ pași: $\\mathbf{P}^h$')]),
    (T('\\textbf{Hamilton filter}: predict $\\Pr(S_t = 1 \\mid Y_{t-1}) = p_{11}\\xi_{t-1} + (1 - p_{22})(1 - \\xi_{t-1})$, with $\\xi_{t-1} = \\Pr(S_{t-1} = 1 \\mid Y_{t-1})$', '\\textbf{Filtrul Hamilton}: prezicem $\\Pr(S_t = 1 \\mid Y_{t-1}) = p_{11}\\xi_{t-1} + (1 - p_{22})(1 - \\xi_{t-1})$, cu $\\xi_{t-1} = \\Pr(S_{t-1} = 1 \\mid Y_{t-1})$'),
     [T('update with the Normal densities $f_j(y_t)$: $\\xi_t = \\dfrac{\\Pr(S_t = 1 \\mid Y_{t-1})f_1(y_t)}{\\Pr(S_t = 1 \\mid Y_{t-1})f_1(y_t) + \\Pr(S_t = 2 \\mid Y_{t-1})f_2(y_t)}$', 'actualizăm cu densitățile distribuției Normale $f_j(y_t)$: $\\xi_t = \\dfrac{\\Pr(S_t = 1 \\mid Y_{t-1})f_1(y_t)}{\\Pr(S_t = 1 \\mid Y_{t-1})f_1(y_t) + \\Pr(S_t = 2 \\mid Y_{t-1})f_2(y_t)}$'),
      T('\\textbf{filtered} probabilities for real time, \\textbf{smoothed} ones for history; \\texttt{statsmodels}: \\texttt{MarkovRegression}', 'probabilitățile \\textbf{filtrate} pentru timp real, cele \\textbf{netezite} pentru istorie; \\texttt{statsmodels}: \\texttt{MarkovRegression}')])))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: computations on paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: the Kalman filter for three periods', 'A1: filtrul Kalman pentru trei perioade'),
         items(T('Local level with $\\sigma^2_\\varepsilon = 1$, $\\sigma^2_\\eta = 1$; prior $a_1 = 0$, $P_1 = 1$; data $y = (1, 3, 2)$.', 'Local level cu $\\sigma^2_\\varepsilon = 1$, $\\sigma^2_\\eta = 1$; a priori $a_1 = 0$, $P_1 = 1$; datele $y = (1, 3, 2)$.'),
               T('1. For $t = 1, 2, 3$ compute $F_t$, $K_t$, $v_t$, $a_{t|t}$, $P_{t|t}$ and the next prediction.', '1. Pentru $t = 1, 2, 3$ calculați $F_t$, $K_t$, $v_t$, $a_{t|t}$, $P_{t|t}$ și predicția următoare.'),
               T('2. Give the forecast of $y_4$ and its variance.', '2. Dați prognoza lui $y_4$ și varianța ei.'),
               T('3. Compute the steady-state gain $\\bar K$ for $q = 1$.', '3. Calculați cîștigul de echilibru $\\bar K$ pentru $q = 1$.'),
               T('Report: a table of the three steps, one forecast, $\\bar K$.', 'Raportați: un tabel cu cei trei pași, o prognoză, $\\bar K$.')),
         items(T('1. $t = 1$: $F = 2$, $K = 0.5$, $v = 1$, $a_{1|1} = 0.5$, $P_{1|1} = 0.5$; $a_2 = 0.5$, $P_2 = 1.5$', '1. $t = 1$: $F = 2$, $K = 0{,}5$, $v = 1$, $a_{1|1} = 0{,}5$, $P_{1|1} = 0{,}5$; $a_2 = 0{,}5$, $P_2 = 1{,}5$'),
               T('$t = 2$: $F = 2.5$, $K = 0.6$, $v = 2.5$, $a_{2|2} = 0.5 + 0.6 \\cdot 2.5 = 2$, $P_{2|2} = 0.6$; $a_3 = 2$, $P_3 = 1.6$', '$t = 2$: $F = 2{,}5$, $K = 0{,}6$, $v = 2{,}5$, $a_{2|2} = 0{,}5 + 0{,}6 \\cdot 2{,}5 = 2$, $P_{2|2} = 0{,}6$; $a_3 = 2$, $P_3 = 1{,}6$'),
               T('$t = 3$: $F = 2.6$, $K = @{a1.3.K}$, $v = 0$, $a_{3|3} = 2$, $P_{3|3} = @{a1.3.Pf}$', '$t = 3$: $F = 2{,}6$, $K = @{a1.3.K}$, $v = 0$, $a_{3|3} = 2$, $P_{3|3} = @{a1.3.Pf}$'),
               T('2. $\\hat y_4 = a_4 = @{a1.anext}$; $P_4 = @{a1.Pnext}$; $\\Var = P_4 + \\sigma^2_\\varepsilon = @{a1.Pnext} + 1$', '2. $\\hat y_4 = a_4 = @{a1.anext}$; $P_4 = @{a1.Pnext}$; $\\Var = P_4 + \\sigma^2_\\varepsilon = @{a1.Pnext} + 1$'),
               T('3. $\\bar P = (1 + \\sqrt5)/2 = @{a1.ssP}$ (the golden ratio), $\\bar K = @{a1.ssK}$; $K_t$ is already @{a1.3.K} at $t = 3$', '3. $\\bar P = (1 + \\sqrt5)/2 = @{a1.ssP}$ (numărul de aur), $\\bar K = @{a1.ssK}$; $K_t$ este deja @{a1.3.K} la $t = 3$')),
         size='scriptsize')

D.proposed(T('A2: a missing observation', 'A2: o observație lipsă'),
           items(T('Local level with $\\sigma^2_\\varepsilon = 2$, $\\sigma^2_\\eta = 0.5$; $a_1 = 5$, $P_1 = 2$; data $y = (6, \\text{missing}, 4)$. Model: A1.', 'Local level cu $\\sigma^2_\\varepsilon = 2$, $\\sigma^2_\\eta = 0{,}5$; $a_1 = 5$, $P_1 = 2$; datele $y = (6, \\text{lipsă}, 4)$. Model: A1.'),
                 T('1. Run the filter for $t = 1, 2, 3$, skipping the update at $t = 2$.', '1. Aplicați filtrul pentru $t = 1, 2, 3$, sărind peste actualizare la $t = 2$.'),
                 T('2. Explain why $K_3$ equals $K_1$.', '2. Explicați de ce $K_3$ este egal cu $K_1$.'),
                 T('3. Compute $\\bar K$ for this $q$ and the equivalent SES weight.', '3. Calculați $\\bar K$ pentru acest $q$ și ponderea SES echivalentă.'),
                 T('Report: the three steps, one sentence, $\\bar K$.', 'Raportați: cei trei pași, o frază, $\\bar K$.')),
           items(T('1. $t = 1$: $F = 4$, $K = 0.5$, $v = 1$, $a_{1|1} = 5.5$, $P_{1|1} = 1$; $a_2 = 5.5$, $P_2 = 1.5$', '1. $t = 1$: $F = 4$, $K = 0{,}5$, $v = 1$, $a_{1|1} = 5{,}5$, $P_{1|1} = 1$; $a_2 = 5{,}5$, $P_2 = 1{,}5$'),
                 T('$t = 2$: no update; $a_3 = 5.5$, $P_3 = 1.5 + 0.5 = 2$', '$t = 2$: fără actualizare; $a_3 = 5{,}5$, $P_3 = 1{,}5 + 0{,}5 = 2$'),
                 T('$t = 3$: $F = 4$, $K = 0.5$, $v = -1.5$, $a_{3|3} = @{a2.3.af}$, $P_{3|3} = 1$; $a_4 = @{a2.anext}$, $P_4 = @{a2.Pnext}$', '$t = 3$: $F = 4$, $K = 0{,}5$, $v = -1{,}5$, $a_{3|3} = @{a2.3.af}$, $P_{3|3} = 1$; $a_4 = @{a2.anext}$, $P_4 = @{a2.Pnext}$'),
                 T('2. The missing year adds $\\sigma^2_\\eta$ without an update: $P_3 = P_1 = 2$, so the gain is the same; uncertainty that grows makes the next observation count more', '2. Anul lipsă adaugă $\\sigma^2_\\eta$ fără actualizare: $P_3 = P_1 = 2$, deci cîștigul este același; incertitudinea care crește face ca următoarea observație să conteze mai mult'),
                 T('3. $q = 0.25$: $\\bar P = @{a2.ssP}$, $\\bar K = @{a2.ssK} = \\alpha_{SES}$', '3. $q = 0{,}25$: $\\bar P = @{a2.ssP}$, $\\bar K = @{a2.ssK} = \\alpha_{SES}$')),
           size='scriptsize')

D.solved(T('A3: exponential smoothing is a steady-state Kalman filter', 'A3: netezirea exponențială este un filtru Kalman în echilibru'),
         items(T('A local level model and simple exponential smoothing (SES).', 'Un model local level și netezirea exponențială simplă (SES).'),
               T('1. Show that the steady-state update $a_{t+1} = a_t + \\bar K(y_t - a_t)$ is SES.', '1. Arătați că actualizarea în echilibru $a_{t+1} = a_t + \\bar K(y_t - a_t)$ este SES.'),
               T('2. Compute $\\bar K$ for $q = 1$ and $q = 0.25$.', '2. Calculați $\\bar K$ pentru $q = 1$ și $q = 0{,}25$.'),
               T('3. An SES model has $\\alpha = 0.3$. Find the $q$ of the local level model behind it.', '3. Un model SES are $\\alpha = 0{,}3$. Găsiți $q$ al modelului local level din spatele lui.'),
               T('Report: one line of algebra, two gains, one $q$.', 'Raportați: un rînd de calcul, două cîștiguri, un $q$.')),
         items(T('1. $a_{t+1} = \\bar Ky_t + (1 - \\bar K)a_t$: the SES recursion with $\\alpha = \\bar K$ and $\\ell_t = a_{t+1}$', '1. $a_{t+1} = \\bar Ky_t + (1 - \\bar K)a_t$: recurența SES cu $\\alpha = \\bar K$ și $\\ell_t = a_{t+1}$'),
               T('2. $q = 1$: $\\bar P/\\sigma^2_\\varepsilon = (1 + \\sqrt5)/2 = @{a3.P1}$, $\\bar K = @{a3.P1}/(@{a3.P1} + 1) = @{a3.K1}$; $q = 0.25$: $\\bar P/\\sigma^2_\\varepsilon = @{a3.P025}$, $\\bar K = @{a3.K025}$', '2. $q = 1$: $\\bar P/\\sigma^2_\\varepsilon = (1 + \\sqrt5)/2 = @{a3.P1}$, $\\bar K = @{a3.P1}/(@{a3.P1} + 1) = @{a3.K1}$; $q = 0{,}25$: $\\bar P/\\sigma^2_\\varepsilon = @{a3.P025}$, $\\bar K = @{a3.K025}$'),
               T('3. From $\\bar P = \\bar P(1 - \\bar K) + \\sigma^2_\\eta$: $\\sigma^2_\\eta = \\bar K\\bar P$ and $\\bar P = \\alpha\\sigma^2_\\varepsilon/(1 - \\alpha)$, so $q = \\alpha^2/(1 - \\alpha) = 0.09/0.7 = @{a3.q03}$', '3. Din $\\bar P = \\bar P(1 - \\bar K) + \\sigma^2_\\eta$: $\\sigma^2_\\eta = \\bar K\\bar P$ și $\\bar P = \\alpha\\sigma^2_\\varepsilon/(1 - \\alpha)$, deci $q = \\alpha^2/(1 - \\alpha) = 0{,}09/0{,}7 = @{a3.q03}$'),
               T('a small $\\alpha$ means a level that moves little relative to the noise', 'un $\\alpha$ mic înseamnă un nivel care se mișcă puțin în raport cu zgomotul')),
         size='scriptsize')

D.proposed(T('A4: state space forms and forecasts', 'A4: forme în spațiul stărilor și prognoze'),
           items(T('An AR(2) with $\\phi_1 = 0.5$, $\\phi_2 = 0.3$, $\\sigma^2 = 1$; a local linear trend. Model: A1 (the prediction step).', 'Un AR(2) cu $\\phi_1 = 0{,}5$, $\\phi_2 = 0{,}3$, $\\sigma^2 = 1$; un local linear trend. Model: A1 (pasul de predicție).'),
                 T('1. Write $Z$, $T$, $R$, $Q$ and $H$ for the AR(2).', '1. Scrieți $Z$, $T$, $R$, $Q$ și $H$ pentru AR(2).'),
                 T('2. With the last state $(y_T, y_{T-1}) = (2, 1)$, compute the forecasts for $h = 1, 2, 3$ by $\\hat\\alpha_{T+h} = T\\hat\\alpha_{T+h-1}$.', '2. Cu ultima stare $(y_T, y_{T-1}) = (2, 1)$, calculați prognozele pentru $h = 1, 2, 3$ prin $\\hat\\alpha_{T+h} = T\\hat\\alpha_{T+h-1}$.'),
                 T('3. Write $Z$ and $T$ of the local linear trend and forecast three steps from $\\mu_T = 100$, $\\beta_T = 2$.', '3. Scrieți $Z$ și $T$ pentru local linear trend și prognozați trei pași de la $\\mu_T = 100$, $\\beta_T = 2$.'),
                 T('Report: two sets of matrices and six forecasts.', 'Raportați: două seturi de matrice și șase prognoze.')),
           items(T('1. $Z = (1,\\ 0)$, $T = \\begin{pmatrix}0.5 & 0.3\\\\ 1 & 0\\end{pmatrix}$, $R = (1,\\ 0)\'$, $Q = 1$, $H = 0$', '1. $Z = (1,\\ 0)$, $T = \\begin{pmatrix}0{,}5 & 0{,}3\\\\ 1 & 0\\end{pmatrix}$, $R = (1,\\ 0)\'$, $Q = 1$, $H = 0$'),
                 T('2. $0.5 \\cdot 2 + 0.3 \\cdot 1 = @{a4.ar1}$; $0.5 \\cdot @{a4.ar1} + 0.3 \\cdot 2 = @{a4.ar2}$; $0.5 \\cdot @{a4.ar2} + 0.3 \\cdot @{a4.ar1} = @{a4.ar3}$', '2. $0{,}5 \\cdot 2 + 0{,}3 \\cdot 1 = @{a4.ar1}$; $0{,}5 \\cdot @{a4.ar1} + 0{,}3 \\cdot 2 = @{a4.ar2}$; $0{,}5 \\cdot @{a4.ar2} + 0{,}3 \\cdot @{a4.ar1} = @{a4.ar3}$'),
                 T('3. $Z = (1,\\ 0)$, $T = \\begin{pmatrix}1 & 1\\\\ 0 & 1\\end{pmatrix}$; forecasts @{a4.llt1}, @{a4.llt2}, @{a4.llt3}: a straight line with slope $\\beta_T$', '3. $Z = (1,\\ 0)$, $T = \\begin{pmatrix}1 & 1\\\\ 0 & 1\\end{pmatrix}$; prognozele @{a4.llt1}, @{a4.llt2}, @{a4.llt3}: o dreaptă cu panta $\\beta_T$')),
           size='scriptsize')

D.solved(T('A5: durations of recessions and expansions', 'A5: durata recesiunilor și a expansiunilor'),
         items(T('A two-regime model of GDP growth: regime 1 (recession) with $p_{11} = 0.75$, regime 2 (expansion) with $p_{22} = 0.95$.', 'Un model cu două regimuri pentru creșterea PIB: regimul 1 (recesiune) cu $p_{11} = 0{,}75$, regimul 2 (expansiune) cu $p_{22} = 0{,}95$.'),
               T('1. Write the transition matrix and compute the expected durations.', '1. Scrieți matricea de tranziție și calculați duratele așteptate.'),
               T('2. Compute the ergodic probabilities.', '2. Calculați probabilitățile ergodice.'),
               T('3. Starting in a recession, compute the probability of a recession two quarters later.', '3. Pornind dintr-o recesiune, calculați probabilitatea unei recesiuni după două trimestre.'),
               T('Report: two durations, two probabilities, one two-step probability.', 'Raportați: două durate, două probabilități, o probabilitate în doi pași.')),
         items(T('1. $\\mathbf{P} = \\begin{pmatrix}0.75 & 0.25\\\\ 0.05 & 0.95\\end{pmatrix}$; durations $1/0.25 = @{a5.d1}$ and $1/0.05 = @{a5.d2}$ quarters', '1. $\\mathbf{P} = \\begin{pmatrix}0{,}75 & 0{,}25\\\\ 0{,}05 & 0{,}95\\end{pmatrix}$; durate $1/0{,}25 = @{a5.d1}$ și $1/0{,}05 = @{a5.d2}$ de trimestre'),
               T('2. $\\pi_1 = 0.05/(0.25 + 0.05) = @{a5.pi1}$, $\\pi_2 = @{a5.pi2}$: one quarter in six is a recession quarter', '2. $\\pi_1 = 0{,}05/(0{,}25 + 0{,}05) = @{a5.pi1}$, $\\pi_2 = @{a5.pi2}$: un trimestru din șase este de recesiune'),
               T('3. $p^{(2)}_{11} = 0.75^2 + 0.25 \\cdot 0.05 = @{a5.p112}$; the chance of an expansion is $@{a5.p212}$', '3. $p^{(2)}_{11} = 0{,}75^2 + 0{,}25 \\cdot 0{,}05 = @{a5.p112}$; șansa unei expansiuni este $@{a5.p212}$'),
               T('the geometric duration: many recessions are short, a few are long; the mean is 4 quarters', 'durata geometrică: multe recesiuni sînt scurte, cîteva sînt lungi; media este 4 trimestre')),
         size='scriptsize')

D.proposed(T('A6: one step of the Hamilton filter', 'A6: un pas al filtrului Hamilton'),
           items(T('The model of A5 with $\\mu_1 = -0.5$, $\\mu_2 = 1$, $\\sigma = 1$; last quarter $\\Pr(S_{t-1} = 1 \\mid Y_{t-1}) = 0.2$; this quarter $y_t = -1$. Model: A5.', 'Modelul din A5 cu $\\mu_1 = -0{,}5$, $\\mu_2 = 1$, $\\sigma = 1$; trimestrul trecut $\\Pr(S_{t-1} = 1 \\mid Y_{t-1}) = 0{,}2$; acest trimestru $y_t = -1$. Model: A5.'),
                 T('1. Compute the predicted probability $\\Pr(S_t = 1 \\mid Y_{t-1})$.', '1. Calculați probabilitatea prezisă $\\Pr(S_t = 1 \\mid Y_{t-1})$.'),
                 T('2. Compute the densities $f_1(y_t)$, $f_2(y_t)$ (standard Normal $\\phi(0.5) = 0.352$, $\\phi(2) = 0.054$) and the filtered probability.', '2. Calculați densitățile $f_1(y_t)$, $f_2(y_t)$ (distribuția Normală standard: $\\phi(0{,}5) = 0{,}352$, $\\phi(2) = 0{,}054$) și probabilitatea filtrată.'),
                 T('3. Give the likelihood contribution and the predicted probability for next quarter.', '3. Dați contribuția la verosimilitate și probabilitatea prezisă pentru trimestrul următor.'),
                 T('Report: four probabilities and one log-density.', 'Raportați: patru probabilități și o log-densitate.')),
           items(T('1. $0.75 \\cdot 0.2 + 0.05 \\cdot 0.8 = @{a6.pred1}$', '1. $0{,}75 \\cdot 0{,}2 + 0{,}05 \\cdot 0{,}8 = @{a6.pred1}$'),
                 T('2. $f_1 = \\phi(-0.5) = @{a6.f1}$, $f_2 = \\phi(-2) = @{a6.f2}$; $\\xi_t = @{a6.pred1} \\cdot @{a6.f1}/@{a6.lik} = @{a6.filt1}$', '2. $f_1 = \\phi(-0{,}5) = @{a6.f1}$, $f_2 = \\phi(-2) = @{a6.f2}$; $\\xi_t = @{a6.pred1} \\cdot @{a6.f1}/@{a6.lik} = @{a6.filt1}$'),
                 T('3. $p(y_t \\mid Y_{t-1}) = @{a6.lik}$, $\\log = @{a6.loglik}$; next prediction $0.75 \\cdot @{a6.filt1} + 0.05 \\cdot (1 - @{a6.filt1}) = @{a6.next_pred1}$', '3. $p(y_t \\mid Y_{t-1}) = @{a6.lik}$, $\\log = @{a6.loglik}$; predicția următoare $0{,}75 \\cdot @{a6.filt1} + 0{,}05 \\cdot (1 - @{a6.filt1}) = @{a6.next_pred1}$'),
                 T('one bad quarter moves the recession probability from 0.19 to 0.60', 'un singur trimestru slab mută probabilitatea de recesiune de la 0,19 la 0,60')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: real data and interpretation', 'Partea B: date reale și interpretare')

D.task(T('B1: underlying Romanian inflation [Solved]', 'B1: inflația de fond din România [Rezolvat]'),
       T('how fast does the underlying level of Romanian inflation move, and which exponential smoothing weight does the data choose?', 'cît de repede se mișcă nivelul de fond al inflației din România și ce pondere de netezire exponențială aleg datele?'),
       T('monthly HICP inflation, seasonally adjusted, 2005--2026 ($T = @{b1.n}$)', 'inflația lunară IAPC, ajustată sezonier, 2005--2026 ($T = @{b1.n}$)'),
       [T('Fit the local level model by maximum likelihood and report $\\hat\\sigma^2_\\varepsilon$, $\\hat\\sigma^2_\\eta$ and $\\hat q$.', 'Estimați modelul local level prin verosimilitate maximă și raportați $\\hat\\sigma^2_\\varepsilon$, $\\hat\\sigma^2_\\eta$ și $\\hat q$.'),
        T('Compute the steady-state gain $\\bar K$ and compare it with the SES weight estimated directly.', 'Calculați cîștigul de echilibru $\\bar K$ și comparați-l cu ponderea SES estimată direct.'),
        T('Plot the filtered and smoothed level ($\\times 12$, in \\% per year) and report its peak and its last value.', 'Reprezentați grafic nivelul filtrat și netezit ($\\times 12$, în \\% pe an) și raportați maximul și ultima valoare.'),
        T('Interpretation: what does the size of $\\hat q$ say about monthly inflation?', 'Interpretare: ce spune mărimea lui $\\hat q$ despre inflația lunară?')],
       T('three variances, two weights, two levels, one sentence', 'trei varianțe, două ponderi, două niveluri, o frază'), size='footnotesize', nb='B1')

D.frame(T('B1: solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch10_sem_b1', h='0.42') + items(
    T('$\\hat\\sigma^2_\\varepsilon = @{b1.se}$, $\\hat\\sigma^2_\\eta = @{b1.sn}$, $\\hat q = @{b1.q}$; $\\bar K = @{b1.K}$; SES by least squares: $\\hat\\alpha = @{b1.ses}$',
      '$\\hat\\sigma^2_\\varepsilon = @{b1.se}$, $\\hat\\sigma^2_\\eta = @{b1.sn}$, $\\hat q = @{b1.q}$; $\\bar K = @{b1.K}$; SES prin cele mai mici pătrate: $\\hat\\alpha = @{b1.ses}$'),
    T('Smoothed level $\\times 12$: peak @{b1.peak}\\% (@{b1.peakd}), low @{b1.low}\\% (@{b1.lowd}), last @{b1.last}\\% (@{b1.lastd}, SE @{b1.sdlast}); the raw series $\\times 12$ has SD @{b1.rawsd}',
      'Nivelul netezit $\\times 12$: maxim @{b1.peak}\\% (@{b1.peakd}), minim @{b1.low}\\% (@{b1.lowd}), ultima valoare @{b1.last}\\% (@{b1.lastd}, SE @{b1.sdlast}); seria brută $\\times 12$ are SD @{b1.rawsd}'),
    T('Interpretation: $q$ is small: most monthly movements are noise (tax changes, food and energy prices); each month moves the underlying level by about @{b1.K} of the surprise',
      'Interpretare: $q$ este mic: majoritatea mișcărilor lunare sînt zgomot (modificări de taxe, prețuri la alimente și energie); fiecare lună mută nivelul de fond cu aproximativ @{b1.K} din surpriză')) + qlsem(),
    'scriptsize')

D.task(T('B2: the US output gap [Proposed]', 'B2: deviația PIB a SUA [Propus]'),
       T('how close are three statistical output gaps to the official CBO gap, and which one would you use in real time?', 'cît de aproape sînt trei deviații PIB statistice de deviația oficială a CBO și pe care ați folosi-o în timp real?'),
       T('US real GDP and the CBO potential GDP, quarterly, 1949--2026; model: B1', 'PIB-ul real al SUA și PIB-ul potențial al CBO, trimestrial, 1949--2026; model: B1'),
       [T('Compute the UC cycle (smooth trend plus AR(2)), the HP gap ($\\lambda = 1600$) and the Hamilton filter ($h = 8$, four lags) of $100\\log\\mathrm{GDP}$.', 'Calculați ciclul UC (trend neted plus AR(2)), deviația HP ($\\lambda = 1600$) și filtrul Hamilton ($h = 8$, patru decalaje) pentru $100\\log\\mathrm{PIB}$.'),
        T('Compute the CBO gap $100(\\mathrm{GDP}/\\mathrm{GDPPOT} - 1)$ and the correlation of each measure with it, including the filtered UC cycle.', 'Calculați deviația CBO $100(\\mathrm{PIB}/\\mathrm{GDPPOT} - 1)$ și corelația fiecărei măsuri cu ea, inclusiv a ciclului UC filtrat.'),
        T('Report the standard deviations and the latest values.', 'Raportați abaterile standard și ultimele valori.'),
        T('Interpretation: which measure would you use to judge the economy today?', 'Interpretare: ce măsură ați folosi pentru a judeca economia astăzi?')],
       T('five correlations, four standard deviations, four latest values, one sentence', 'cinci corelații, patru abateri standard, patru valori recente, o frază'), size='footnotesize', nb='B2')

D.frame(T('B2: solution [Proposed]', 'B2: rezolvare [Propus]'), fig('ch10_sem_b2', h='0.42') + items(
    T('Correlation with the CBO gap: UC smoothed @{b2.c_uc}, UC filtered @{b2.c_ucf}, HP @{b2.c_hp}, Hamilton @{b2.c_ham}; SD: CBO @{b2.sd_cbo}, UC @{b2.sd_uc}, HP @{b2.sd_hp}, Hamilton @{b2.sd_ham}',
      'Corelația cu deviația CBO: UC netezit @{b2.c_uc}, UC filtrat @{b2.c_ucf}, HP @{b2.c_hp}, Hamilton @{b2.c_ham}; SD: CBO @{b2.sd_cbo}, UC @{b2.sd_uc}, HP @{b2.sd_hp}, Hamilton @{b2.sd_ham}'),
    T('Latest: CBO @{b2.last_cbo}\\%, UC @{b2.last_uc}\\%, HP @{b2.last_hp}\\%, Hamilton @{b2.last_ham}\\%',
      'Ultimele valori: CBO @{b2.last_cbo}\\%, UC @{b2.last_uc}\\%, HP @{b2.last_hp}\\%, Hamilton @{b2.last_ham}\\%'),
    T('Interpretation: the HP gap is the smallest and pulled to zero at the end of the sample; for today use a one-sided measure (filtered UC, Hamilton) and report its uncertainty',
      'Interpretare: deviația HP este cea mai mică și este trasă spre zero la capătul eșantionului; pentru astăzi folosiți o măsură unilaterală (UC filtrat, Hamilton) și raportați-i incertitudinea')) + qlsem(),
    'scriptsize', instructor_only=True)

D.task(T('B3: recessions in US industrial production [Solved]', 'B3: recesiunile în producția industrială din SUA [Rezolvat]'),
       T('does a two-regime model of industrial production find the NBER recessions?', 'găsește un model cu două regimuri pentru producția industrială recesiunile NBER?'),
       T('monthly growth of US industrial production, 1960--2019 (@{b3.nrec} NBER recession months)', 'creșterea lunară a producției industriale din SUA, 1960--2019 (@{b3.nrec} de luni de recesiune NBER)'),
       [T('Fit a two-regime model with switching mean and variance (\\texttt{MarkovRegression}).', 'Estimați un model cu două regimuri, cu medie și varianță variabile (\\texttt{MarkovRegression}).'),
        T('Report the regime means and standard deviations, $p_{11}$, $p_{22}$ and the expected durations.', 'Raportați mediile și abaterile standard ale regimurilor, $p_{11}$, $p_{22}$ și duratele așteptate.'),
        T('Classify each month by the smoothed probability ($> 0.5$) and by the filtered probability, and compare with the NBER months.', 'Clasificați fiecare lună după probabilitatea netezită ($> 0{,}5$) și după cea filtrată și comparați cu lunile NBER.'),
        T('Interpretation: does the recession regime match the NBER recessions?', 'Interpretare: se potrivește regimul de recesiune cu recesiunile NBER?')],
       T('six parameters, two durations, two concordance tables, one sentence', 'șase parametri, două durate, două tabele de concordanță, o frază'), size='footnotesize', nb='B3')

D.frame(T('B3: solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch10_sem_b3', h='0.4') + items(
    T('Recession regime: mean @{b3.m1}\\%, SD @{b3.s1}; expansion: mean @{b3.m2}\\%, SD @{b3.s2}; $\\hat p_{11} = @{b3.p11}$, $\\hat p_{22} = @{b3.p22}$; durations @{b3.d1} and @{b3.d2} months; ergodic share @{b3.erg}\\%',
      'Regimul de recesiune: media @{b3.m1}\\%, SD @{b3.s1}; expansiunea: media @{b3.m2}\\%, SD @{b3.s2}; $\\hat p_{11} = @{b3.p11}$, $\\hat p_{22} = @{b3.p22}$; durate de @{b3.d1} și @{b3.d2} luni; ponderea ergodică @{b3.erg}\\%'),
    T('Smoothed: @{b3.c.concord}\\% of months agree, @{b3.c.hit}\\% of NBER months found, @{b3.c.false}\\% false alarms; filtered: @{b3.f.concord}\\%, @{b3.f.hit}\\%, @{b3.f.false}\\%',
      'Netezit: @{b3.c.concord}\\% din luni coincid, @{b3.c.hit}\\% din lunile NBER găsite, @{b3.c.false}\\% alarme false; filtrat: @{b3.f.concord}\\%, @{b3.f.hit}\\%, @{b3.f.false}\\%'),
    T('Interpretation: yes for the deep recessions; the regime is ``low and volatile growth\'\', so it also flags some slowdowns, and in real time it finds fewer recession months',
      'Interpretare: da, pentru recesiunile profunde; regimul este „creștere scăzută și volatilă”, deci semnalează și unele încetiniri, iar în timp real găsește mai puține luni de recesiune')) + qlsem(),
    'scriptsize')

D.task(T('B4: volatility regimes of the BET [Proposed]', 'B4: regimurile de volatilitate ale BET [Propus]'),
       T('does the Bucharest market have calm and turbulent regimes, and are they the same as those of the S\\&P 500?', 'are piața de la București regimuri calme și agitate și sînt ele aceleași cu cele ale S\\&P 500?'),
       T('weekly log returns of the BET and of the S\\&P 500, 2000--2026; model: B3', 'randamentele logaritmice săptămînale ale BET și S\\&P 500, 2000--2026; model: B3'),
       [T('Fit a two-regime model with switching mean and variance to BET returns; report the annualised volatilities and durations.', 'Estimați un model cu două regimuri, cu medie și varianță variabile, pentru randamentele BET; raportați volatilitățile anualizate și duratele.'),
        T('List the three years with the largest share of turbulent weeks.', 'Enumerați cei trei ani cu cea mai mare pondere de săptămîni agitate.'),
        T('Fit a GARCH(1,1) (Chapter 5) and correlate its volatility with the regime-implied volatility.', 'Estimați un GARCH(1,1) (Capitolul 5) și corelați volatilitatea lui cu volatilitatea implicată de regimuri.'),
        T('Interpretation: are the turbulent periods of the BET those of the S\\&P 500?', 'Interpretare: sînt perioadele agitate ale BET aceleași cu cele ale S\\&P 500?')],
       T('two volatilities, two durations, three years, two correlations, one sentence', 'două volatilități, două durate, trei ani, două corelații, o frază'), size='footnotesize', nb='B4')

D.frame(T('B4: solution [Proposed]', 'B4: rezolvare [Propus]'), fig('ch10_sem_b4', h='0.4') + items(
    T('Calm: @{b4.s2}\\% per year, @{b4.d2} weeks; turbulent: @{b4.s1}\\% per year, @{b4.d1} weeks; turbulent in @{b4.share}\\% of the weeks; years: @{b4.y0} (@{b4.ys0}\\%), @{b4.y1} (@{b4.ys1}\\%), @{b4.y2} (@{b4.ys2}\\%)',
      'Calm: @{b4.s2}\\% pe an, @{b4.d2} de săptămîni; agitat: @{b4.s1}\\% pe an, @{b4.d1} săptămîni; agitat în @{b4.share}\\% din săptămîni; anii: @{b4.y0} (@{b4.ys0}\\%), @{b4.y1} (@{b4.ys1}\\%), @{b4.y2} (@{b4.ys2}\\%)'),
    T('GARCH(1,1): $\\alpha + \\beta = @{b4.gab}$; correlation with the regime volatility @{b4.cg}; correlation of the turbulent probabilities of BET and S\\&P 500: @{b4.cs} (same classification in @{b4.agree}\\% of the weeks)',
      'GARCH(1,1): $\\alpha + \\beta = @{b4.gab}$; corelația cu volatilitatea regimurilor @{b4.cg}; corelația probabilităților de regim agitat pentru BET și S\\&P 500: @{b4.cs} (aceeași clasificare în @{b4.agree}\\% din săptămîni)'),
    T('Interpretation: only partly: global crises (2008--2009) are common, but the BET has its own turbulent years (2001--2002) and was mostly calm in 2022 (@{b4.bet22}\\% turbulent weeks, against @{b4.sp22}\\% for the S\\&P 500)',
      'Interpretare: doar parțial: crizele globale (2008--2009) sînt comune, dar BET are propriii ani agitați (2001--2002) și a fost mai ales calm în 2022 (@{b4.bet22}\\% săptămîni agitate, față de @{b4.sp22}\\% pentru S\\&P 500)')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: open questions and AI critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: regimes of Romanian inflation [Proposed]', 'C1: regimurile inflației din România [Propus]'),
       T('how many inflation regimes has Romania had since 1996, and when did the low-inflation regime begin?', 'cîte regimuri de inflație a avut România din 1996 și cînd a început regimul de inflație scăzută?'),
       T('monthly HICP inflation, not seasonally adjusted, 1996--2026; inflation targeting since August 2005; model: B3', 'inflația lunară IAPC, neajustată sezonier, 1996--2026; țintirea inflației din august 2005; model: B3'),
       [T('Fit models with two and three regimes (switching mean and variance) and compare them by BIC.', 'Estimați modele cu două și trei regimuri (medie și varianță variabile) și comparați-le după BIC.'),
        T('For three regimes, report the means, durations and the date when the low regime becomes permanent.', 'Pentru trei regimuri, raportați mediile, duratele și data de la care regimul scăzut devine permanent.'),
        T('Propose one more check (seasonality, a regime with an AR term, out-of-sample forecasts).', 'Propuneți încă o verificare (sezonalitatea, un regim cu termen AR, prognoze în afara eșantionului).'),
        T('Interpretation: did inflation targeting start the low-inflation regime?', 'Interpretare: a declanșat țintirea inflației regimul de inflație scăzută?')],
       T('six parameters, a date, a check and a project plan', 'șase parametri, o dată, o verificare și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: reference analysis [Proposed]', 'C1: analiză de referință [Propus]'), fig('ch10_sem_c1', h='0.42') + items(
    T('Means (\\% per month): low @{c1.low.m} (SD @{c1.low.s}, about @{c1.ann}\\% per year), medium @{c1.medium.m}, high @{c1.high.m}; durations @{c1.low.d}, @{c1.medium.d}, @{c1.high.d} months',
      'Medii (\\% pe lună): scăzut @{c1.low.m} (SD @{c1.low.s}, aproximativ @{c1.ann}\\% pe an), mediu @{c1.medium.m}, ridicat @{c1.high.m}; durate de @{c1.low.d}, @{c1.medium.d}, @{c1.high.d} luni'),
    T('BIC: @{c1.bic2} with two regimes, @{c1.bic3} with three. The low regime appears in @{c1.firstlow} and is the regime of almost every month from @{c1.lowyear}, before inflation targeting; medium months return in 2010, 2022 and July--August 2025 (VAT increases in 2010 and 2025)',
      'BIC: @{c1.bic2} cu două regimuri, @{c1.bic3} cu trei. Regimul scăzut apare în @{c1.firstlow} și este regimul aproape tuturor lunilor din @{c1.lowyear}, înainte de țintirea inflației; lunile de regim mediu revin în 2010, 2022 și iulie--august 2025 (majorările TVA din 2010 și 2025)'),
    T('Interpretation: disinflation started before 2005; targeting consolidated it. Project: add a regime-dependent AR term and compare forecasts with a local level model (B1)',
      'Interpretare: dezinflația a început înainte de 2005; țintirea a consolidat-o. Proiect: adăugați un termen AR dependent de regim și comparați prognozele cu un model local level (B1)')) + qlsem(),
    'scriptsize', instructor_only=True)

D.frame(T('C2: audit an AI answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant about state space and regime models. The answer:', 'Un student a întrebat un asistent AI despre modelele în spațiul stărilor și cu regimuri. Răspunsul:'),
    T('\\aiprompt{(a) For the Nile, q = 0.097, so the equivalent exponential smoothing weight is alpha = q = 0.097.}', '\\aiprompt{(a) Pentru Nil, q = 0,097, deci ponderea echivalentă de netezire exponențială este alpha = q = 0,097.}'),
    T('\\aiprompt{(b) The Kalman gain is K = sigma2\\_eps / (P + sigma2\\_eps): the noisier the data, the more weight they get.}', '\\aiprompt{(b) Cîștigul Kalman este K = sigma2\\_eps / (P + sigma2\\_eps): cu cît datele sînt mai zgomotoase, cu atît primesc o pondere mai mare.}'),
    T('\\aiprompt{(c) With p11 = 0.75, recessions last p11 / (1 - p11) = 3 quarters on average.}', '\\aiprompt{(c) Cu p11 = 0,75, recesiunile durează în medie p11 / (1 - p11) = 3 trimestre.}'),
    T('\\aiprompt{(d) To call a recession in real time, use smoothed\\_marginal\\_probabilities from statsmodels.}', '\\aiprompt{(d) Pentru a anunța o recesiune în timp real, folosiți smoothed\\_marginal\\_probabilities din statsmodels.}'),
    T('\\aiprompt{(e) The last value of the HP gap is a reliable estimate of today\'s output gap.}', '\\aiprompt{(e) Ultima valoare a deviației HP este o estimare de încredere a deviației PIB de azi.}'),
    T('\\aiprompt{(f) In the local level model the forecasts of all horizons are equal; only their variance grows with the horizon.}', '\\aiprompt{(f) În modelul local level prognozele pentru toate orizonturile sînt egale; doar varianța lor crește cu orizontul.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number (Part A, notebook section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă (Partea A, secțiunea C2 din notebook).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: $\\alpha = \\bar K = @{c2.K}$, not $q$; the map is $q = \\alpha^2/(1 - \\alpha)$ (A3)', '(a) Greșit: $\\alpha = \\bar K = @{c2.K}$, nu $q$; relația este $q = \\alpha^2/(1 - \\alpha)$ (A3)'),
    T('(b) Wrong: $K = P/(P + \\sigma^2_\\varepsilon)$; noisy data get \\textbf{less} weight (A1)', '(b) Greșit: $K = P/(P + \\sigma^2_\\varepsilon)$; datele zgomotoase primesc o pondere \\textbf{mai mică} (A1)'),
    T('(c) Wrong: the expected duration is $1/(1 - p_{11}) = 4$ quarters (A5)', '(c) Greșit: durata așteptată este $1/(1 - p_{11}) = 4$ trimestre (A5)'),
    T('(d) Wrong: smoothed probabilities use future data; in real time use \\texttt{filtered\\_marginal\\_probabilities} (B3)', '(d) Greșit: probabilitățile netezite folosesc date viitoare; în timp real folosiți \\texttt{filtered\\_marginal\\_probabilities} (B3)'),
    T('(e) Wrong: HP is two-sided and its last values are revised when new quarters arrive \\refHamHP; use a one-sided measure (B2)', '(e) Greșit: HP este bilateral, iar ultimele lui valori se revizuiesc cînd sosesc trimestre noi \\refHamHP; folosiți o măsură unilaterală (B2)'),
    T('(f) Correct: $\\hat y_{n+h} = a_{n+1}$ for all $h$, with variance $P_{n+1} + (h - 1)\\sigma^2_\\eta + \\sigma^2_\\varepsilon$', '(f) Corect: $\\hat y_{n+h} = a_{n+1}$ pentru orice $h$, cu varianța $P_{n+1} + (h - 1)\\sigma^2_\\eta + \\sigma^2_\\varepsilon$')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-up', 'Încheiere')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Kalman filter: $K_t = P_t/(P_t + \\sigma^2_\\varepsilon)$, $a_{t|t} = a_t + K_tv_t$; a missing value means no update', 'Filtrul Kalman: $K_t = P_t/(P_t + \\sigma^2_\\varepsilon)$, $a_{t|t} = a_t + K_tv_t$; o valoare lipsă înseamnă fără actualizare'),
    T('Exponential smoothing is a local level filter in steady state: $\\alpha = \\bar K$, $q = \\alpha^2/(1 - \\alpha)$', 'Netezirea exponențială este un filtru local level în echilibru: $\\alpha = \\bar K$, $q = \\alpha^2/(1 - \\alpha)$'),
    T('Romanian monthly inflation: small $q$, a slowly moving underlying level', 'Inflația lunară din România: $q$ mic, un nivel de fond care se mișcă lent'),
    T('Markov switching: durations $1/(1 - p_{ii})$; filtered probabilities for real time, smoothed ones for history', 'Markov switching: durate $1/(1 - p_{ii})$; probabilitățile filtrate pentru timp real, cele netezite pentru istorie'),
    T('An AI answer is a draft: check the gain, the duration formula, filtered against smoothed', 'Un răspuns AI este o ciornă: verificați cîștigul, formula duratei, filtrat față de netezit')))

D.frame(T('After the seminar', 'După seminar'), items(
    T('Lecture 10 develops each topic of today: the state space form, the filter and the smoother, the likelihood, trend and cycle, Markov switching',
      'Cursul 10 dezvoltă fiecare temă de azi: forma în spațiul stărilor, filtrul și netezitorul, verosimilitatea, trend și ciclu, modelele Markov switching'),
    T('Try the [Proposed] tasks in the notebook', 'Încercați cerințele [Propus] în notebook'),
    T('C1 can grow into a team project: inflation regimes in Romania and in the region', 'C1 poate deveni un proiect de echipă: regimurile inflației în România și în regiune'),
    T('Reading: \\refDK, Ch.~2; \\refHamMS; \\refHamHP', 'Lectură: \\refDK, cap.~2; \\refHamMS; \\refHamHP'),
    T('\\textbf{The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class}',
      '\\textbf{Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar}')))

D.references(bib(['DK', 'HamMS', 'HamHP', 'Harvey', 'HPf', 'Kalman', 'Muth']), per=16)

if __name__ == '__main__':
    finalize(D.write(V))
