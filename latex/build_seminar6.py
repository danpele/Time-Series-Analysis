r"""
build_seminar6.py -- Seminarul 6 (Modele VAR și cauzalitate Granger), EN + RO dintr-o singură sursă
===================================================================================================
Seminarul are loc ÎNAINTEA cursului 6: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu o întrebare de interpretare, C o întrebare deschisă și
critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în versiunea
profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_06/sem6_results.json (seminar6.py).
Ieșire:
  EN/Seminars/seminar6_var_models_granger_causality.tex          (+ _solutions.tex)
  RO/Seminarii/seminar6_modele_var_cauzalitate_granger_ro.tex    (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_06/seminar6.py && python3 latex/build_seminar6.py && python3 latex/tsa_build.py compile 6
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, fig   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch6_common import REFS, T, bib, date, finalize, load_sem, pv, qtr   # noqa: E402

S = load_sem()
V = Values()
D = Deck(6, 'seminar', refs=REFS)


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


def qlsem():
    return '\\quantlet{TSA\\_ch6\\_seminar}{\\qlurl{TSA_ch6_seminar}}'


def mat2(M, d=2):
    f = lambda x: '⁅' + f'{x:.{d}f}' + '⁆'
    return f'\\begin{{pmatrix}} {f(M[0][0])} & {f(M[0][1])} \\\\ {f(M[1][0])} & {f(M[1][1])} \\end{{pmatrix}}'


def vec2(v, d=2):
    f = lambda x: '⁅' + f'{x:.{d}f}' + '⁆'
    return f'\\begin{{pmatrix}} {f(v[0])} \\\\ {f(v[1])} \\end{{pmatrix}}'


# =============================================================================
# CIFRE
# =============================================================================
A1 = S['A1']
V.put('a1.l1', A1['lam_re'][0], 1)
V.put('a1.l2', A1['lam_re'][1], 1)
V.put('a1.tr', A1['tr'], 1)
V.put('a1.det', A1['det'], 2)
V.raw('a1.mu', vec2(A1['mu'], 3))
V.raw('a1.f1', vec2(A1['f1']))
V.raw('a1.f2', vec2(A1['f2']))
A2a, A2b = S['A2a'], S['A2b']
V.put('a2a.l1', A2a['lam_re'][0], 1)
V.put('a2a.l2', A2a['lam_re'][1], 1)
V.put('a2a.tr', A2a['tr'], 1)
V.put('a2a.det', A2a['det'], 2)
V.put('a2b.re', A2b['lam_re'][0], 1)
V.put('a2b.im', abs(A2b['lam_im'][0]), 1)
V.put('a2b.mod', A2b['mod'][0], 3)
V.put('a2b.det', A2b['det'], 2)
A3 = S['A3']
V.raw('a3.P', mat2(A3['P'], 3))
V.raw('a3.Th1', mat2(A3['Theta'][1], 3))
V.raw('a3.Phi2', mat2(A3['Phi'][2], 2))
V.raw('a3.Pr', mat2(A3['P_rev'], 3))
V.put('a3.p22', A3['P'][1][1], 3)
V.put('a3.t21', A3['Theta'][1][1][0], 2)
V.put('a3.t22', A3['Theta'][1][1][1], 3)
V.put('a3.t11', A3['Theta'][1][0][0], 2)
V.put('a3.t12', A3['Theta'][1][0][1], 3)
V.put('a3.r11', A3['P_rev'][0][0], 3)
V.put('a3.mse2', A3['mse'][1][1], 3)
V.put('a3.mse1', A3['mse'][1][0], 3)
V.put('a3.fe1', 100 * A3['fevd'][0][1][0], 1)
V.put('a3.fe2', 100 * A3['fevd'][1][1][0], 1)
V.put('a3.fe2b', 100 * A3['fevd'][1][1][1], 1)
V.put('a3.fe12', 100 * A3['fevd'][1][0][1], 1)
A5 = S['A5']
V.put('a5.F', A5['F'], 2)
V.put('a5.cv', A5['crit5'], 2)
V.raw('a5.p', pv(A5['p']))
A6 = S['A6']
for p in range(5):
    for c in ('aic', 'bic', 'hq'):
        V.put(f'a6.{p}.{c}', A6['table'][str(p)][c], 3)
for c in ('aic', 'bic', 'hq'):
    V.raw(f'a6.b.{c}', str(A6['best'][c]))
    V.put(f'a6.pen.{c}', A6['pen'][c], 3)

B1, B2 = S['B1'], S['B2']
for k, B in (('b1', B1), ('b2', B2)):
    V.int(f'{k}.T', B['T'])
    V.raw(f'{k}.p', str(B['p']))
    for o in ('aic', 'bic', 'hqic'):
        V.raw(f'{k}.o.{o}', str(B['orders'][o]))
    V.put(f'{k}.F1', B['sp_bet']['F'], 1)
    V.raw(f'{k}.p1', pv(B['sp_bet']['p']))
    V.put(f'{k}.cv', B['sp_bet']['crit5'], 2)
    V.put(f'{k}.F2', B['bet_sp']['F'], 2)
    V.raw(f'{k}.p2', pv(B['bet_sp']['p']))
    V.put(f'{k}.c1', B['ccf1'], 3)
    V.put(f'{k}.c0', B['ccf0'], 2)
    V.put(f'{k}.cm1', B['ccfm1'], 3)
    V.put(f'{k}.g0', B['girf_bet'][0], 2)
    V.put(f'{k}.g1', B['girf_bet'][1], 2)
    V.put(f'{k}.g2', B['girf_bet'][2], 3)
    V.put(f'{k}.b', B['coef1'], 3)
    V.put(f'{k}.se', B['se1'], 3)
    V.put(f'{k}.cu', B['corr_u'], 2)
B3 = S['B3']
for c in ('aic', 'bic', 'hq'):
    V.raw(f'b3.ic.{c}', str(B3['ic']['best'][c]))
V.put('b3.mod', B3['max_mod'], 3)
V.raw('b3.n', str(B3['nobs']))
V.raw('b3.q0', qtr(B3['first']))
V.raw('b3.q1', qtr(B3['last']))
for h in ('8', '12'):
    V.put(f'b3.q{h}', B3['lb'][h]['stat'], 1)
    V.raw(f'b3.q{h}df', str(B3['lb'][h]['df']))
    V.raw(f'b3.q{h}p', pv(B3['lb'][h]['p']))
for k, v in B3['granger'].items():
    kk = k.replace('->', '')
    V.put(f'b3.g.{kk}.F', v['F'], 2)
    V.raw(f'b3.g.{kk}.p', pv(v['p']))
V.put('b3.cv', B3['crit5'], 2)
V.put('b3.ipi0', B3['i_pi'][0], 2)
V.put('b3.ipimax', max(B3['i_pi']), 2)
V.raw('b3.ipih', str(int(np.argmax(B3['i_pi']))))
V.put('b3.ipi12', B3['i_pi'][12], 2)
V.put('b3.lo12', B3['i_pi_lo'][12], 2)
for h in ('1', '4', '12'):
    for j, s in enumerate(['g', 'pi', 'i']):
        V.put(f'b3.fe.{h}.{s}', B3['fevd_i'][h][j], 0)
B4 = S['B4']
for k in ('ds->pi', 'ds->i', 'ds->g', 'i->ds', 'pi->ds'):
    kk = k.replace('->', '')
    V.put(f'b4.{kk}.F', B4['granger'][k]['F'], 2)
    V.raw(f'b4.{kk}.p', pv(B4['granger'][k]['p']))
V.raw('b4.n', str(B4['nobs']))
V.raw('b4.q0', qtr(B4['first']))
for h in ('1', '4', '8', '12'):
    V.put(f'b4.fe.{h}', B4['fevd_pi'][h][1], 1)
V.put('b4.r1', B4['pi_ds'][1], 2)
V.put('b4.rmin', min(B4['pi_ds']), 2)
V.raw('b4.nsig', str(sum(B4['pi_ds_sig'])))
for c in ('aic', 'bic', 'hq'):
    V.raw(f'b4.ic.{c}', str(B4['ic_best'][c]))
V.put('b4.mod', B4['max_mod'], 3)
C1 = S['C1']
labs = list(C1)
for j, lab in enumerate(labs):
    V.put(f'c1.{j}.max', C1[lab]['max'], 2)
    V.raw(f'c1.{j}.h', str(C1[lab]['hmax']))
    V.put(f'c1.{j}.min', C1[lab]['min'], 2)
    V.raw(f'c1.{j}.hmin', str(C1[lab]['hmin']))
C2 = S['C2']
V.raw('c2.aic', str(C2['aic_p']))
V.raw('c2.bic', str(C2['bic_p']))
V.raw('c2.lb', pv(C2['lb8_p']))
V.put('c2.eA1', C2['eig_A1'], 2)
V.put('c2.eF', C2['eig_comp'], 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: when we model several series together, which of them helps to forecast the others, and how does a shock in one spread to the rest?',
       '\\textbf{Întrebarea}: cînd modelăm împreună mai multe serii, care dintre ele ajută la prognoza celorlalte și cum se propagă un șoc dintr-o serie în celelalte?'),
     [T('this seminar comes \\textbf{before} Lecture 6: the section ``What you need today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 6: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: a VAR(1) by hand (stability, mean, forecasts); impulse responses with the Cholesky factor; a variance decomposition; a Granger $F$ test; lag selection',
        'Partea A: un VAR(1) calculat de mînă (stabilitate, medie, prognoze); răspunsuri la impuls cu factorul Cholesky; o descompunere a varianței; un test Granger $F$; alegerea numărului de decalaje'),
      T('Part B: does the S\\&P 500 lead the BET (daily and weekly)? a VAR for Romanian GDP growth, inflation and ROBOR 3M; the EUR/RON rate added to it',
        'Partea B: precedă S\\&P 500 indicele BET (zilnic și săptămînal)? un VAR pentru creșterea PIB-ului, inflația și ROBOR 3M din România; cursul EUR/RON adăugat în model'),
      T('Part C: an open question on the ``price puzzle\'\' and an AI answer to audit', 'Partea C: o întrebare deschisă despre „enigma prețurilor” și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('a VAR(1) by hand: stability, mean, forecasts', 'un VAR(1) calculat de mînă: stabilitate, medie, prognoze') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('orthogonalised impulse responses; a variance decomposition', 'răspunsuri la impuls ortogonalizate; o descompunere a varianței') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('a Granger $F$ test; lag selection by AIC, BIC and HQ', 'un test Granger $F$; alegerea numărului de decalaje după AIC, BIC și HQ') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('S\\&P 500 and BET: daily; weekly', 'S\\&P 500 și BET: zilnic; săptămînal') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('a Romanian macro VAR; the EUR/RON rate added', 'un VAR macroeconomic românesc; cursul EUR/RON adăugat') + ' & ' + SP + ' & B3',
     'C1, C2 & ' + T('the price puzzle; what is wrong in an AI answer?', 'enigma prețurilor; ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B3'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați dumneavoastră, după model')))

D.frame(T('Data used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Frequency}', '\\textbf{Frecvența}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['S\\&P 500, BET & EODHD & ' + T('daily, close', 'zilnic, închidere') + ' & 2000--2026',
     T('Real GDP, Romania', 'PIB real, România') + ' & Eurostat (namq\\_10\\_gdp) & ' + T('quarterly, seasonally adjusted', 'trimestrial, ajustat sezonier') + ' & 2005--2026',
     T('HICP, Romania', 'IAPC, România') + ' & Eurostat (prc\\_hicp\\_minr) & ' + T('monthly, 2015 = 100', 'lunar, 2015 = 100') + ' & 2004--2026',
     'ROBOR 3M & Eurostat (irt\\_st\\_m) & ' + T('monthly, \\% per year', 'lunar, \\% pe an') + ' & 2005--2026',
     'EUR/RON & ' + T('BNR reference rate', 'cursul de referință BNR') + ' & ' + T('daily', 'zilnic') + ' & 2005--2026'],
    size='footnotesize') + items(
    T('Quarterly VAR variables: $g_t = 100\\Delta\\ln\\mathrm{GDP}_t$; $\\pi_t$: 12-month HICP inflation (quarterly mean); $i_t$: ROBOR 3M (quarterly mean); $s_t$: $100\\Delta\\ln$ of the quarterly mean EUR/RON',
      'Variabilele VAR-ului trimestrial: $g_t = 100\\Delta\\ln\\mathrm{PIB}_t$; $\\pi_t$: inflația anuală IAPC (media trimestrială); $i_t$: ROBOR 3M (media trimestrială); $s_t$: $100\\Delta\\ln$ al mediei trimestriale EUR/RON'),
    T('In the notebook: \\texttt{VAR} from \\texttt{statsmodels}, \\texttt{granger\\_F}, \\texttt{ic\\_table}, \\texttt{girf}; no account or key is needed',
      'În notebook: \\texttt{VAR} din \\texttt{statsmodels}, \\texttt{granger\\_F}, \\texttt{ic\\_table}, \\texttt{girf}; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What you need today', 'Noțiuni necesare azi')

D.frame(T('What you need today (1/4): the VAR model and its stability', 'Noțiuni necesare azi (1/4): modelul VAR și stabilitatea lui'), items(
    (T('\\textbf{VAR$(p)$}: $\\bY_t = \\bc + \\bA_1\\bY_{t-1} + \\dots + \\bA_p\\bY_{t-p} + \\bepsilon_t$; $\\bY_t$: $K$ series; $\\bA_j$: $K \\times K$; $\\bepsilon_t$ white noise, $E[\\bepsilon_t\\bepsilon_t\'] = \\bSigma$',
       '\\textbf{VAR$(p)$}: $\\bY_t = \\bc + \\bA_1\\bY_{t-1} + \\dots + \\bA_p\\bY_{t-p} + \\bepsilon_t$; $\\bY_t$: $K$ serii; $\\bA_j$: $K \\times K$; $\\bepsilon_t$ zgomot alb, $E[\\bepsilon_t\\bepsilon_t\'] = \\bSigma$'),
     [T('equation by equation: each variable is regressed on a constant and the $p$ lags of all variables; $a_{jk}$: effect of $y_{k,t-1}$ on $y_{jt}$', 'ecuație cu ecuație: fiecare variabilă este regresată pe o constantă și pe cele $p$ decalaje ale tuturor variabilelor; $a_{jk}$: efectul lui $y_{k,t-1}$ asupra lui $y_{jt}$')]),
    (T('\\textbf{Stability} of a VAR(1): all eigenvalues of $\\bA$ have $|\\lambda| < 1$; for $K = 2$: $\\lambda^2 - \\mathrm{tr}(\\bA)\\lambda + \\det(\\bA) = 0$', '\\textbf{Stabilitatea} unui VAR(1): toate valorile proprii ale lui $\\bA$ au $|\\lambda| < 1$; pentru $K = 2$: $\\lambda^2 - \\mathrm{tr}(\\bA)\\lambda + \\det(\\bA) = 0$'),
     [T('complex $\\lambda = a \\pm bi$: modulus $\\sqrt{a^2 + b^2}$; VAR$(p)$: the eigenvalues of the companion matrix $\\begin{pmatrix} \\bA_1 & \\bA_2 \\\\ \\mathbf{I} & \\mathbf{0} \\end{pmatrix}$ (for $p = 2$)', '$\\lambda = a \\pm bi$ complex: modulul $\\sqrt{a^2 + b^2}$; VAR$(p)$: valorile proprii ale matricei companion $\\begin{pmatrix} \\bA_1 & \\bA_2 \\\\ \\mathbf{I} & \\mathbf{0} \\end{pmatrix}$ (pentru $p = 2$)')]),
    (T('\\textbf{Mean} of a stable VAR(1): $\\boldsymbol{\\mu} = (\\mathbf{I} - \\bA)^{-1}\\bc$; \\textbf{forecasts}: $\\hat\\bY_{T+1} = \\bc + \\bA\\bY_T$, $\\hat\\bY_{T+2} = \\bc + \\bA\\hat\\bY_{T+1}$', '\\textbf{Media} unui VAR(1) stabil: $\\boldsymbol{\\mu} = (\\mathbf{I} - \\bA)^{-1}\\bc$; \\textbf{prognoze}: $\\hat\\bY_{T+1} = \\bc + \\bA\\bY_T$, $\\hat\\bY_{T+2} = \\bc + \\bA\\hat\\bY_{T+1}$'),
     [T('inverse of a $2 \\times 2$ matrix: $\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}^{-1} = \\frac{1}{ad - bc}\\begin{pmatrix} d & -b \\\\ -c & a \\end{pmatrix}$', 'inversa unei matrice $2 \\times 2$: $\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}^{-1} = \\frac{1}{ad - bc}\\begin{pmatrix} d & -b \\\\ -c & a \\end{pmatrix}$')])))

D.frame(T('What you need today (2/4): estimation, lag order and diagnostics', 'Noțiuni necesare azi (2/4): estimare, numărul de decalaje și diagnosticare'), items(
    (T('\\textbf{Estimation}: OLS equation by equation (all equations have the same regressors); $K + pK^2$ coefficients', '\\textbf{Estimarea}: OLS ecuație cu ecuație (toate ecuațiile au aceiași regresori); $K + pK^2$ coeficienți'),
     [T('the variables must be stationary (Chapter 3): growth rates, returns, or rates that do not trend', 'variabilele trebuie să fie staționare (Capitolul 3): rate de creștere, randamente sau rate fără trend')]),
    (T('\\textbf{Lag order}: $\\mathrm{IC}(p) = \\ln\\det\\tilde\\bSigma(p) + c_T\\,pK^2/T$, smallest is best, on a common sample', '\\textbf{Numărul de decalaje}: $\\mathrm{IC}(p) = \\ln\\det\\tilde\\bSigma(p) + c_T\\,pK^2/T$, cea mai mică valoare este cea mai bună, pe un eșantion comun'),
     [T('$c_T = 2$ (AIC), $\\ln T$ (BIC), $2\\ln\\ln T$ (HQ); BIC chooses the smallest $p$, AIC the largest', '$c_T = 2$ (AIC), $\\ln T$ (BIC), $2\\ln\\ln T$ (HQ); BIC alege cel mai mic $p$, AIC cel mai mare')]),
    (T('\\textbf{Portmanteau test}: $H_0$: residuals without auto- and cross-correlation up to lag $h$; $Q_h \\sim \\chi^2(K^2(h - p))$', '\\textbf{Testul portmanteau}: $H_0$: reziduuri fără autocorelații și corelații încrucișate pînă la decalajul $h$; $Q_h \\sim \\chi^2(K^2(h - p))$'),
     [T('a small p-value: add lags or variables; normality (Jarque--Bera) often fails because of crises', 'o valoare p mică: adăugăm decalaje sau variabile; normalitatea (Jarque--Bera) eșuează adesea din cauza crizelor')])))

D.frame(T('What you need today (3/4): Granger causality', 'Noțiuni necesare azi (3/4): cauzalitatea Granger'), items(
    (T('$x$ \\textbf{Granger-causes} $y$ if the past of $x$ improves the forecast of $y$ beyond the past of $y$ and of the other variables', '$x$ \\textbf{cauzează în sens Granger} pe $y$ dacă trecutul lui $x$ îmbunătățește prognoza lui $y$ dincolo de trecutul lui $y$ și al celorlalte variabile'),
     [T('in a VAR$(p)$: $H_0$: the $p$ coefficients of $x_{t-1}, \\dots, x_{t-p}$ in the equation of $y$ are all zero', 'într-un VAR$(p)$: $H_0$: cei $p$ coeficienți ai lui $x_{t-1}, \\dots, x_{t-p}$ din ecuația lui $y$ sînt toți zero')]),
    (T('\\textbf{$F$ test}: $F = \\dfrac{(RSS_R - RSS_U)/q}{RSS_U/(T - k)}$; $q$: number of restrictions; $k$: coefficients of the unrestricted equation', '\\textbf{Testul $F$}: $F = \\dfrac{(RSS_R - RSS_U)/q}{RSS_U/(T - k)}$; $q$: numărul de restricții; $k$: coeficienții ecuației nerestricționate'),
     [T('reject $H_0$ if $F$ exceeds the critical value of $F(q, T - k)$ (about 3.1 for $q = 2$ and $T - k \\approx 80$--100)', 'respingem $H_0$ dacă $F$ depășește valoarea critică a distribuției $F(q, T - k)$ (circa 3,1 pentru $q = 2$ și $T - k \\approx 80$--100)')]),
    (T('\\textbf{Not} cause and effect: a statement about predictability, relative to the chosen variables', '\\textbf{Nu} înseamnă cauză și efect: este o afirmație despre predictibilitate, relativă la variabilele alese'),
     [T('pitfalls: an omitted common driver; effects within the same period (instantaneous causality, in $\\bSigma$); non-stationary series', 'capcane: un factor comun omis; efecte în aceeași perioadă (cauzalitate instantanee, în $\\bSigma$); serii nestaționare')])))

D.frame(T('What you need today (4/4): impulse responses and FEVD', 'Noțiuni necesare azi (4/4): răspunsuri la impuls și FEVD'), items(
    (T('\\textbf{Impulse responses} of a VAR(1): $\\bPhi_h = \\bA^h$; column $k$: the path of all variables after a unit shock to $\\varepsilon_k$', '\\textbf{Răspunsurile la impuls} ale unui VAR(1): $\\bPhi_h = \\bA^h$; coloana $k$: traiectoria tuturor variabilelor după un șoc unitar în $\\varepsilon_k$'),
     [T('the shocks are correlated, so we use $\\bSigma = \\mathbf{P}\\mathbf{P}\'$ (\\textbf{Cholesky}, $\\mathbf{P}$ lower triangular) and $\\boldsymbol{\\Theta}_h = \\bPhi_h\\mathbf{P}$', 'șocurile sînt corelate, deci folosim $\\bSigma = \\mathbf{P}\\mathbf{P}\'$ (\\textbf{Cholesky}, $\\mathbf{P}$ inferior triunghiulară) și $\\boldsymbol{\\Theta}_h = \\bPhi_h\\mathbf{P}$'),
      T('$2 \\times 2$: $p_{11} = \\sqrt{\\sigma_{11}}$, $p_{21} = \\sigma_{21}/p_{11}$, $p_{22} = \\sqrt{\\sigma_{22} - p_{21}^2}$', '$2 \\times 2$: $p_{11} = \\sqrt{\\sigma_{11}}$, $p_{21} = \\sigma_{21}/p_{11}$, $p_{22} = \\sqrt{\\sigma_{22} - p_{21}^2}$')]),
    (T('\\textbf{The ordering matters}: the first variable does not react within the period to the shocks of the others', '\\textbf{Ordinea contează}: prima variabilă nu reacționează în aceeași perioadă la șocurile celorlalte'),
     [T('the \\textbf{generalised} responses \\refPS\\ do not depend on the order: $\\bPhi_h\\bSigma\\mathbf{e}_k/\\sqrt{\\sigma_{kk}}$', 'răspunsurile \\textbf{generalizate} \\refPS\\ nu depind de ordine: $\\bPhi_h\\bSigma\\mathbf{e}_k/\\sqrt{\\sigma_{kk}}$')]),
    (T('\\textbf{FEVD}: share of shock $k$ in the $h$-step forecast error variance of $y_j$: $\\sum_{s<h}\\theta_{jk,s}^2 / \\sum_{s<h}\\sum_m\\theta_{jm,s}^2$', '\\textbf{FEVD}: ponderea șocului $k$ în varianța erorii de prognoză la $h$ pași a lui $y_j$: $\\sum_{s<h}\\theta_{jk,s}^2 / \\sum_{s<h}\\sum_m\\theta_{jm,s}^2$'),
     [T('each row sums to 100\\%', 'fiecare rînd însumează 100\\%')])))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: computations on paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: a VAR(1) by hand', 'A1: un VAR(1) calculat de mînă'),
         items(T('$\\bc = (0.4, 0.8)\'$, $\\bA = \\begin{pmatrix} 0.6 & 0.2 \\\\ 0.2 & 0.6 \\end{pmatrix}$; last observation $\\bY_T = (3, 2)\'$.', '$\\bc = (0{,}4;\\ 0{,}8)\'$, $\\bA = \\begin{pmatrix} 0{,}6 & 0{,}2 \\\\ 0{,}2 & 0{,}6 \\end{pmatrix}$; ultima observație $\\bY_T = (3;\\ 2)\'$.'),
               T('1. Write the two equations.', '1. Scrieți cele două ecuații.'),
               T('2. Find the eigenvalues of $\\bA$ and decide whether the VAR is stable.', '2. Aflați valorile proprii ale lui $\\bA$ și decideți dacă VAR-ul este stabil.'),
               T('3. Compute the mean $\\boldsymbol{\\mu}$.', '3. Calculați media $\\boldsymbol{\\mu}$.'),
               T('4. Compute $\\hat\\bY_{T+1}$ and $\\hat\\bY_{T+2}$.', '4. Calculați $\\hat\\bY_{T+1}$ și $\\hat\\bY_{T+2}$.'),
               T('Report: two equations, two eigenvalues, $\\boldsymbol{\\mu}$, two forecasts.', 'Raportați: două ecuații, două valori proprii, $\\boldsymbol{\\mu}$, două prognoze.')),
         items(T('1. $y_{1t} = 0.4 + 0.6y_{1,t-1} + 0.2y_{2,t-1} + \\varepsilon_{1t}$; $y_{2t} = 0.8 + 0.2y_{1,t-1} + 0.6y_{2,t-1} + \\varepsilon_{2t}$', '1. $y_{1t} = 0{,}4 + 0{,}6y_{1,t-1} + 0{,}2y_{2,t-1} + \\varepsilon_{1t}$; $y_{2t} = 0{,}8 + 0{,}2y_{1,t-1} + 0{,}6y_{2,t-1} + \\varepsilon_{2t}$'),
               T('2. $\\mathrm{tr} = @{a1.tr}$, $\\det = 0.36 - 0.04 = @{a1.det}$: $\\lambda^2 - 1.2\\lambda + 0.32 = 0$, $\\lambda = @{a1.l1}$ and $@{a1.l2}$; both below 1: stable', '2. $\\mathrm{tr} = @{a1.tr}$, $\\det = 0{,}36 - 0{,}04 = @{a1.det}$: $\\lambda^2 - 1{,}2\\lambda + 0{,}32 = 0$, $\\lambda = @{a1.l1}$ și $@{a1.l2}$; ambele sub 1: stabil'),
               T('3. $\\mathbf{I} - \\bA = \\begin{pmatrix} 0.4 & -0.2 \\\\ -0.2 & 0.4 \\end{pmatrix}$, $\\det = 0.12$; $\\boldsymbol{\\mu} = \\frac{1}{0.12}\\begin{pmatrix} 0.4 & 0.2 \\\\ 0.2 & 0.4 \\end{pmatrix}\\begin{pmatrix} 0.4 \\\\ 0.8 \\end{pmatrix} = @{a1.mu}$',
                 '3. $\\mathbf{I} - \\bA = \\begin{pmatrix} 0{,}4 & -0{,}2 \\\\ -0{,}2 & 0{,}4 \\end{pmatrix}$, $\\det = 0{,}12$; $\\boldsymbol{\\mu} = \\frac{1}{0{,}12}\\begin{pmatrix} 0{,}4 & 0{,}2 \\\\ 0{,}2 & 0{,}4 \\end{pmatrix}\\begin{pmatrix} 0{,}4 \\\\ 0{,}8 \\end{pmatrix} = @{a1.mu}$'),
               T('4. $\\hat\\bY_{T+1} = (0.4 + 1.8 + 0.4,\\ 0.8 + 0.6 + 1.2)\' = @{a1.f1}$; $\\hat\\bY_{T+2} = @{a1.f2}$: moving towards $\\boldsymbol{\\mu}$', '4. $\\hat\\bY_{T+1} = (0{,}4 + 1{,}8 + 0{,}4;\\ 0{,}8 + 0{,}6 + 1{,}2)\' = @{a1.f1}$; $\\hat\\bY_{T+2} = @{a1.f2}$: se apropie de $\\boldsymbol{\\mu}$')),
         size='scriptsize')

D.proposed(T('A2: stable or not?', 'A2: stabil sau nu?'),
           items(T('Two VAR(1) matrices: (a) $\\bA = \\begin{pmatrix} 0.9 & 0.3 \\\\ 0.2 & 0.8 \\end{pmatrix}$; (b) $\\bA = \\begin{pmatrix} 0.5 & -0.4 \\\\ 0.4 & 0.5 \\end{pmatrix}$. Model: A1.', 'Două matrice VAR(1): (a) $\\bA = \\begin{pmatrix} 0{,}9 & 0{,}3 \\\\ 0{,}2 & 0{,}8 \\end{pmatrix}$; (b) $\\bA = \\begin{pmatrix} 0{,}5 & -0{,}4 \\\\ 0{,}4 & 0{,}5 \\end{pmatrix}$. Model: A1.'),
                 T('1. For each matrix, compute the trace, the determinant and the eigenvalues.', '1. Pentru fiecare matrice, calculați urma, determinantul și valorile proprii.'),
                 T('2. Decide whether each VAR is stable.', '2. Decideți dacă fiecare VAR este stabil.'),
                 T('3. Describe how the responses to a shock behave in each case.', '3. Descrieți cum se comportă răspunsurile la un șoc în fiecare caz.'),
                 T('Report: four eigenvalues (or moduli), two decisions and two sentences.', 'Raportați: patru valori proprii (sau module), două decizii și două fraze.')),
           items(T('1. (a) $\\mathrm{tr} = @{a2a.tr}$, $\\det = @{a2a.det}$: $\\lambda = @{a2a.l1}$, $@{a2a.l2}$; (b) $\\mathrm{tr} = 1$, $\\det = @{a2b.det}$: $\\lambda = @{a2b.re} \\pm @{a2b.im}i$, $|\\lambda| = @{a2b.mod}$', '1. (a) $\\mathrm{tr} = @{a2a.tr}$, $\\det = @{a2a.det}$: $\\lambda = @{a2a.l1}$, $@{a2a.l2}$; (b) $\\mathrm{tr} = 1$, $\\det = @{a2b.det}$: $\\lambda = @{a2b.re} \\pm @{a2b.im}i$, $|\\lambda| = @{a2b.mod}$'),
                 T('2. (a) $@{a2a.l1} > 1$: not stable (explosive); (b) $@{a2b.mod} < 1$: stable', '2. (a) $@{a2a.l1} > 1$: nestabil (exploziv); (b) $@{a2b.mod} < 1$: stabil'),
                 T('3. (a) a shock is amplified by about 10\\% per period, forever; (b) complex roots: the responses oscillate and shrink by a factor @{a2b.mod} per period', '3. (a) un șoc este amplificat cu circa 10\\% pe perioadă, la nesfîrșit; (b) rădăcini complexe: răspunsurile oscilează și se micșorează cu factorul @{a2b.mod} pe perioadă')),
           size='scriptsize')

D.solved(T('A3: orthogonalised impulse responses', 'A3: răspunsuri la impuls ortogonalizate'),
         items(T('The VAR(1) of A1 with $\\bSigma = \\begin{pmatrix} 1 & 0.4 \\\\ 0.4 & 1 \\end{pmatrix}$, order $(y_1, y_2)$.', 'VAR(1) din A1 cu $\\bSigma = \\begin{pmatrix} 1 & 0{,}4 \\\\ 0{,}4 & 1 \\end{pmatrix}$, ordinea $(y_1, y_2)$.'),
               T('1. Compute the Cholesky factor $\\mathbf{P}$.', '1. Calculați factorul Cholesky $\\mathbf{P}$.'),
               T('2. Compute $\\boldsymbol{\\Theta}_0 = \\mathbf{P}$ and $\\boldsymbol{\\Theta}_1 = \\bA\\mathbf{P}$, and $\\bPhi_2 = \\bA^2$.', '2. Calculați $\\boldsymbol{\\Theta}_0 = \\mathbf{P}$, $\\boldsymbol{\\Theta}_1 = \\bA\\mathbf{P}$ și $\\bPhi_2 = \\bA^2$.'),
               T('3. Compute $\\mathbf{P}$ again with the order $(y_2, y_1)$ and say what changes on impact.', '3. Calculați din nou $\\mathbf{P}$ cu ordinea $(y_2, y_1)$ și precizați ce se schimbă la impact.'),
               T('Report: three matrices and one sentence.', 'Raportați: trei matrice și o frază.')),
         items(T('1. $p_{11} = 1$, $p_{21} = 0.4$, $p_{22} = \\sqrt{1 - 0.16} = @{a3.p22}$: $\\mathbf{P} = @{a3.P}$', '1. $p_{11} = 1$, $p_{21} = 0{,}4$, $p_{22} = \\sqrt{1 - 0{,}16} = @{a3.p22}$: $\\mathbf{P} = @{a3.P}$'),
               T('2. $\\boldsymbol{\\Theta}_1 = @{a3.Th1}$; $\\bPhi_2 = @{a3.Phi2}$; after $u_1 = 1$: $y_2$ moves by 0.4 at once and by @{a3.t21} one period later', '2. $\\boldsymbol{\\Theta}_1 = @{a3.Th1}$; $\\bPhi_2 = @{a3.Phi2}$; după $u_1 = 1$: $y_2$ se mișcă imediat cu 0,4 și cu @{a3.t21} o perioadă mai tîrziu'),
               T('3. Order $(y_2, y_1)$: $\\mathbf{P} = @{a3.Pr}$ (in the original order); now $y_1$ reacts on impact to the $y_2$ shock (0.4) and $y_2$ does not react to the $y_1$ shock', '3. Ordinea $(y_2, y_1)$: $\\mathbf{P} = @{a3.Pr}$ (în ordinea inițială); acum $y_1$ reacționează la impact la șocul lui $y_2$ (0,4), iar $y_2$ nu reacționează la șocul lui $y_1$')),
         size='scriptsize')

D.proposed(T('A4: a variance decomposition by hand', 'A4: o descompunere a varianței calculată de mînă'),
           items(T('Use $\\boldsymbol{\\Theta}_0$ and $\\boldsymbol{\\Theta}_1$ of A3 (order $y_1, y_2$). Model: A3.', 'Folosiți $\\boldsymbol{\\Theta}_0$ și $\\boldsymbol{\\Theta}_1$ din A3 (ordinea $y_1, y_2$). Model: A3.'),
                 T('1. Compute the share of $u_1$ in the 1-step forecast error variance of $y_2$.', '1. Calculați ponderea lui $u_1$ în varianța erorii de prognoză la un pas a lui $y_2$.'),
                 T('2. Compute the 2-step forecast error variance of $y_2$ and the share of $u_1$ in it.', '2. Calculați varianța erorii de prognoză la doi pași a lui $y_2$ și ponderea lui $u_1$ în ea.'),
                 T('3. Compute the share of $u_2$ in the 2-step error variance of $y_1$, and explain why it is 0 at one step.', '3. Calculați ponderea lui $u_2$ în varianța erorii la doi pași a lui $y_1$ și explicați de ce este 0 la un pas.'),
                 T('Report: three percentages and one sentence.', 'Raportați: trei procente și o frază.')),
           items(T('1. $\\theta_{21,0}^2 = 0.16$ out of $0.16 + 0.84 = 1$: @{a3.fe1}\\%', '1. $\\theta_{21,0}^2 = 0{,}16$ din $0{,}16 + 0{,}84 = 1$: @{a3.fe1}\\%'),
                 T('2. variance $1 + 0.44^2 + @{a3.t22}^2 = @{a3.mse2}$; share of $u_1$: $(0.16 + 0.1936)/@{a3.mse2} = @{a3.fe2}$\\%', '2. varianța $1 + 0{,}44^2 + @{a3.t22}^2 = @{a3.mse2}$; ponderea lui $u_1$: $(0{,}16 + 0{,}1936)/@{a3.mse2} = @{a3.fe2}$\\%'),
                 T('3. $y_1$: $(0 + @{a3.t12}^2)/@{a3.mse1} = @{a3.fe12}$\\%; at one step it is 0 because $y_1$ is ordered first and does not react to $u_2$ on impact', '3. $y_1$: $(0 + @{a3.t12}^2)/@{a3.mse1} = @{a3.fe12}$\\%; la un pas este 0 deoarece $y_1$ este așezată prima și nu reacționează la $u_2$ la impact')),
           size='scriptsize')

D.solved(T('A5: a Granger $F$ test', 'A5: un test Granger $F$'),
         items(T('A bivariate VAR(2) on $T = 100$ observations. In the equation of $y_1$: $RSS_U = 45.2$ (with the two lags of $y_2$) and $RSS_R = 52.8$ (without them).', 'Un VAR(2) cu două variabile, pe $T = 100$ de observații. În ecuația lui $y_1$: $RSS_U = 45{,}2$ (cu cele două decalaje ale lui $y_2$) și $RSS_R = 52{,}8$ (fără ele).'),
               T('1. Give $H_0$, the number of restrictions $q$ and the number of coefficients $k$ of the unrestricted equation.', '1. Precizați $H_0$, numărul de restricții $q$ și numărul de coeficienți $k$ ai ecuației nerestricționate.'),
               T('2. Compute $F$.', '2. Calculați $F$.'),
               T('3. Decide at 5\\% and state the conclusion in words.', '3. Decideți la 5\\% și formulați concluzia în cuvinte.'),
               T('Report: $H_0$, $F$, a decision and one sentence.', 'Raportați: $H_0$, $F$, o decizie și o frază.')),
         items(T('1. $H_0$: $a_{12}^{(1)} = a_{12}^{(2)} = 0$ ($y_2$ does not Granger-cause $y_1$); $q = 2$; $k = 1 + 2 \\cdot 2 = 5$', '1. $H_0$: $a_{12}^{(1)} = a_{12}^{(2)} = 0$ ($y_2$ nu cauzează $y_1$ în sens Granger); $q = 2$; $k = 1 + 2 \\cdot 2 = 5$'),
               T('2. $F = \\dfrac{(52.8 - 45.2)/2}{45.2/95} = \\dfrac{3.8}{0.476} = @{a5.F}$', '2. $F = \\dfrac{(52{,}8 - 45{,}2)/2}{45{,}2/95} = \\dfrac{3{,}8}{0{,}476} = @{a5.F}$'),
               T('3. $@{a5.F} > @{a5.cv}$, the 5\\% value of $F(2, 95)$ (p @{a5.p}): reject; the past of $y_2$ helps to forecast $y_1$, which does not prove that $y_2$ causes $y_1$', '3. $@{a5.F} > @{a5.cv}$, valoarea de 5\\% a distribuției $F(2, 95)$ (p @{a5.p}): respingem; trecutul lui $y_2$ ajută la prognoza lui $y_1$, ceea ce nu demonstrează că $y_2$ îl cauzează pe $y_1$')),
         size='scriptsize')

D.proposed(T('A6: choosing $p$ by hand', 'A6: alegerea lui $p$ de mînă'),
           items(T('A VAR with $K = 3$ on a common sample of $T = 80$ quarters gives $\\ln\\det\\tilde\\bSigma(p) = 1.50; 0.60; 0.25; 0.00; -0.15$ for $p = 0, \\dots, 4$. Model: A5.', 'Un VAR cu $K = 3$, pe un eșantion comun de $T = 80$ de trimestre, dă $\\ln\\det\\tilde\\bSigma(p) = 1{,}50;\\ 0{,}60;\\ 0{,}25;\\ 0{,}00;\\ -0{,}15$ pentru $p = 0, \\dots, 4$. Model: A5.'),
                 T('1. Compute the penalty per lag $c_T K^2/T$ for AIC, BIC and HQ.', '1. Calculați penalizarea pe decalaj $c_T K^2/T$ pentru AIC, BIC și HQ.'),
                 T('2. Compute the three criteria for each $p$ and find the minima.', '2. Calculați cele trei criterii pentru fiecare $p$ și aflați minimele.'),
                 T('3. Say which $p$ you would start from with 80 quarters, and why.', '3. Precizați de la ce $p$ ați porni cu 80 de trimestre și de ce.'),
                 T('Report: three penalties, a table of 15 values, three orders and one sentence.', 'Raportați: trei penalizări, un tabel cu 15 valori, trei ordine și o frază.')),
           items(T('1. AIC $2 \\cdot 9/80 = @{a6.pen.aic}$; BIC $\\ln 80 \\cdot 9/80 = @{a6.pen.bic}$; HQ $2\\ln\\ln 80 \\cdot 9/80 = @{a6.pen.hq}$', '1. AIC $2 \\cdot 9/80 = @{a6.pen.aic}$; BIC $\\ln 80 \\cdot 9/80 = @{a6.pen.bic}$; HQ $2\\ln\\ln 80 \\cdot 9/80 = @{a6.pen.hq}$'),
                 T('2. AIC: @{a6.1.aic}, @{a6.2.aic}, @{a6.3.aic}, @{a6.4.aic} ($p = 1..4$): min at $p = @{a6.b.aic}$; BIC: @{a6.1.bic}, @{a6.2.bic}: min at $p = @{a6.b.bic}$; HQ: @{a6.1.hq}, @{a6.2.hq}, @{a6.3.hq}: min at $p = @{a6.b.hq}$',
                   '2. AIC: @{a6.1.aic}, @{a6.2.aic}, @{a6.3.aic}, @{a6.4.aic} ($p = 1..4$): minim la $p = @{a6.b.aic}$; BIC: @{a6.1.bic}, @{a6.2.bic}: minim la $p = @{a6.b.bic}$; HQ: @{a6.1.hq}, @{a6.2.hq}, @{a6.3.hq}: minim la $p = @{a6.b.hq}$'),
                 T('3. Start from $p = 2$ (HQ): $3 + 2 \\cdot 9 = 21$ coefficients; check the residuals and move to $p = 3$ only if they are autocorrelated', '3. Pornim de la $p = 2$ (HQ): $3 + 2 \\cdot 9 = 21$ de coeficienți; verificăm reziduurile și trecem la $p = 3$ doar dacă sînt autocorelate')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: real data and interpretation', 'Partea B: date reale și interpretare')

D.task(T('B1: does the S\\&P 500 lead the BET? [Solved]', 'B1: precedă S\\&P 500 indicele BET? [Rezolvat]'),
       T('do yesterday\'s S\\&P 500 returns help to forecast today\'s BET returns, and the other way round?', 'ajută randamentele S\\&P 500 de ieri la prognoza randamentelor BET de azi și invers?'),
       T('daily log returns (\\%) of the S\\&P 500 and the BET since 2000, on common trading days', 'randamentele logaritmice zilnice (\\%) ale S\\&P 500 și BET din 2000, în zilele comune de tranzacționare'),
       [T('Compute the cross-correlations $\\Corr(\\text{BET}_t, \\text{S\\&P}_{t-k})$ for $k = -5, \\dots, 5$.', 'Calculați corelațiile încrucișate $\\Corr(\\text{BET}_t, \\text{S\\&P}_{t-k})$ pentru $k = -5, \\dots, 5$.'),
        T('Choose $p$ for a bivariate VAR with AIC, BIC and HQ (maximum 10) and estimate the VAR with the HQ order.', 'Alegeți $p$ pentru un VAR cu două variabile după AIC, BIC și HQ (maximum 10) și estimați VAR-ul cu ordinul ales de HQ.'),
        T('Test Granger causality in both directions.', 'Testați cauzalitatea Granger în ambele sensuri.'),
        T('Compute the generalised response of the BET to an S\\&P 500 shock for 5 days.', 'Calculați răspunsul generalizat al BET la un șoc S\\&P 500 pentru 5 zile.'),
        T('Interpretation: why does New York lead Bucharest by one day?', 'Interpretare: de ce precedă New York-ul Bucureștiul cu o zi?')],
       T('three cross-correlations, the chosen $p$, two $F$ tests, two responses and one sentence', 'trei corelații încrucișate, $p$ ales, două teste $F$, două răspunsuri și o frază'), size='footnotesize', nb='B1')

D.frame(T('B1: solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch6_sem_b1', h='0.36') + items(
    T('$T = @{b1.T}$ days; CCF: @{b1.c0} ($k = 0$), @{b1.c1} ($k = 1$), @{b1.cm1} ($k = -1$); orders: AIC @{b1.o.aic}, BIC @{b1.o.bic}, HQ @{b1.o.hqic}: VAR(@{b1.p})',
      '$T = @{b1.T}$ de zile; CCF: @{b1.c0} ($k = 0$), @{b1.c1} ($k = 1$), @{b1.cm1} ($k = -1$); ordine: AIC @{b1.o.aic}, BIC @{b1.o.bic}, HQ @{b1.o.hqic}: VAR(@{b1.p})'),
    T('S\\&P 500 $\\to$ BET: $F = @{b1.F1}$ (p @{b1.p1}); BET $\\to$ S\\&P 500: $F = @{b1.F2}$ (p @{b1.p2}); coefficient of S\\&P$_{t-1}$: @{b1.b} (SE @{b1.se})',
      'S\\&P 500 $\\to$ BET: $F = @{b1.F1}$ (p @{b1.p1}); BET $\\to$ S\\&P 500: $F = @{b1.F2}$ (p @{b1.p2}); coeficientul lui S\\&P$_{t-1}$: @{b1.b} (SE @{b1.se})'),
    T('Generalised response of the BET: @{b1.g0}\\% on the same day, @{b1.g1}\\% the next day, @{b1.g2}\\% after two days',
      'Răspunsul generalizat al BET: @{b1.g0}\\% în aceeași zi, @{b1.g1}\\% în ziua următoare, @{b1.g2}\\% după două zile'),
    T('Interpretation: Wall Street closes after Bucharest, so news of the American afternoon reaches the BET only the next day: a time-zone effect, not a slow market',
      'Interpretare: Wall Street se închide după București, deci știrile din după-amiaza americană ajung la BET abia a doua zi: un efect al fusurilor orare, nu o piață lentă')) + qlsem(),
    'scriptsize')

D.task(T('B2: the same question with weekly returns [Proposed]', 'B2: aceeași întrebare cu randamente săptămînale [Propus]'),
       T('does the lead of the S\\&P 500 survive when we use Friday-to-Friday weekly returns?', 'rezistă precedența S\\&P 500 dacă folosim randamente săptămînale, de vineri pînă vineri?'),
       T('weekly log returns (\\%) of the S\\&P 500 and the BET from the Friday closes; model: B1', 'randamente logaritmice săptămînale (\\%) ale S\\&P 500 și BET, din închiderile de vineri; model: B1'),
       [T('Repeat steps 1--4 of B1 on weekly returns.', 'Repetați pașii 1--4 din B1 pe randamentele săptămînale.'),
        T('Compare the lag-1 cross-correlation and the $F$ statistic with the daily ones.', 'Comparați corelația încrucișată la decalajul 1 și statistica $F$ cu cele zilnice.'),
        T('Interpretation: why is the lead much weaker at the weekly frequency?', 'Interpretare: de ce este precedența mult mai slabă la frecvența săptămînală?')],
       T('the same items as in B1 and two sentences', 'aceleași elemente ca în B1 și două fraze'), size='footnotesize', nb='B2')

D.frame(T('B2: solution [Proposed]', 'B2: rezolvare [Propus]'), fig('ch6_sem_b2', h='0.34') + items(
    T('$T = @{b2.T}$ weeks; CCF: @{b2.c0} ($k = 0$), @{b2.c1} ($k = 1$); orders: AIC @{b2.o.aic}, BIC @{b2.o.bic}, HQ @{b2.o.hqic}: VAR(@{b2.p})',
      '$T = @{b2.T}$ de săptămîni; CCF: @{b2.c0} ($k = 0$), @{b2.c1} ($k = 1$); ordine: AIC @{b2.o.aic}, BIC @{b2.o.bic}, HQ @{b2.o.hqic}: VAR(@{b2.p})'),
    T('S\\&P 500 $\\to$ BET: $F = @{b2.F1}$ (p @{b2.p1}), against $F = @{b1.F1}$ daily; BET $\\to$ S\\&P 500: p @{b2.p2}', 'S\\&P 500 $\\to$ BET: $F = @{b2.F1}$ (p @{b2.p1}), față de $F = @{b1.F1}$ zilnic; BET $\\to$ S\\&P 500: p @{b2.p2}'),
    T('Interpretation: the one-day delay now falls inside the same week, so it shows up in the same-week correlation (@{b2.c0}) instead of the lag; a weak lead remains', 'Interpretare: întîrzierea de o zi cade acum în aceeași săptămînă, deci apare în corelația din aceeași săptămînă (@{b2.c0}), nu la decalaj; rămîne o precedență slabă')) + qlsem(),
    'scriptsize', instructor_only=True)

D.task(T('B3: a VAR for the Romanian economy [Solved]', 'B3: un VAR pentru economia României [Rezolvat]'),
       T('how do GDP growth, inflation and the 3-month interest rate interact in Romania?', 'cum interacționează în România creșterea PIB-ului, inflația și dobînda la 3 luni?'),
       T('$\\bY_t = (g_t, \\pi_t, i_t)\'$, quarterly, @{b3.q0}--@{b3.q1}', '$\\bY_t = (g_t, \\pi_t, i_t)\'$, trimestrial, @{b3.q0}--@{b3.q1}'),
       [T('Choose $p$ with AIC, BIC and HQ ($p \\le 6$) and estimate a VAR(2).', 'Alegeți $p$ după AIC, BIC și HQ ($p \\le 6$) și estimați un VAR(2).'),
        T('Check stability (largest modulus of the companion eigenvalues) and run the portmanteau test with $h = 8$ and $h = 12$.', 'Verificați stabilitatea (cel mai mare modul al valorilor proprii ale matricei companion) și aplicați testul portmanteau cu $h = 8$ și $h = 12$.'),
        T('Run the six Granger $F$ tests.', 'Aplicați cele șase teste Granger $F$.'),
        T('Plot the orthogonalised response of ROBOR to an inflation shock (order $g, \\pi, i$) and the FEVD of ROBOR.', 'Reprezentați grafic răspunsul ortogonalizat al ROBOR la un șoc al inflației (ordinea $g, \\pi, i$) și FEVD pentru ROBOR.'),
        T('Interpretation: do the results show that the central bank reacts to inflation?', 'Interpretare: arată rezultatele că banca centrală reacționează la inflație?')],
       T('three orders, one modulus, two portmanteau tests, a table of six tests, the chart and one sentence', 'trei ordine, un modul, două teste portmanteau, un tabel cu șase teste, graficul și o frază'), size='footnotesize', nb='B3')

D.frame(T('B3: solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch6_sem_b3', h='0.32') + items(
    T('Orders: AIC @{b3.ic.aic}, BIC @{b3.ic.bic}, HQ @{b3.ic.hq}; VAR(2) on @{b3.n} quarters; largest modulus @{b3.mod}: stable; $Q_8 = @{b3.q8}$ (p @{b3.q8p}), $Q_{12} = @{b3.q12}$ (p @{b3.q12p})',
      'Ordine: AIC @{b3.ic.aic}, BIC @{b3.ic.bic}, HQ @{b3.ic.hq}; VAR(2) pe @{b3.n} de trimestre; cel mai mare modul @{b3.mod}: stabil; $Q_8 = @{b3.q8}$ (p @{b3.q8p}), $Q_{12} = @{b3.q12}$ (p @{b3.q12p})'),
    T('Granger (p): $g \\to i$ @{b3.g.gi.p}, $\\pi \\to i$ @{b3.g.pii.p}, $i \\to g$ @{b3.g.ig.p}, $\\pi \\to g$ @{b3.g.pig.p}, $i \\to \\pi$ @{b3.g.ipi.p}, $g \\to \\pi$ @{b3.g.gpi.p}',
      'Granger (p): $g \\to i$ @{b3.g.gi.p}, $\\pi \\to i$ @{b3.g.pii.p}, $i \\to g$ @{b3.g.ig.p}, $\\pi \\to g$ @{b3.g.pig.p}, $i \\to \\pi$ @{b3.g.ipi.p}, $g \\to \\pi$ @{b3.g.gpi.p}'),
    T('ROBOR after an inflation shock: @{b3.ipi0} pp on impact, @{b3.ipimax} pp after @{b3.ipih} quarters; FEVD of ROBOR at 12 quarters: inflation @{b3.fe.12.pi}\\%, own @{b3.fe.12.i}\\%',
      'ROBOR după un șoc al inflației: @{b3.ipi0} pp la impact, @{b3.ipimax} pp după @{b3.ipih} trimestre; FEVD pentru ROBOR la 12 trimestre: inflația @{b3.fe.12.pi}\\%, propriile șocuri @{b3.fe.12.i}\\%'),
    T('Interpretation: they show that ROBOR \\textbf{follows} inflation, as a reaction function would imply; ROBOR is a market rate, and the VAR cannot separate the BNR\'s decisions from market expectations',
      'Interpretare: arată că ROBOR \\textbf{urmează} inflația, cum ar implica o funcție de reacție; ROBOR este o dobîndă de piață, iar VAR-ul nu poate separa deciziile BNR de așteptările pieței')) + qlsem(),
    'scriptsize')

D.task(T('B4: adding the EUR/RON exchange rate [Proposed]', 'B4: adăugarea cursului EUR/RON [Propus]'),
       T('does the depreciation of the leu help to forecast inflation (exchange-rate pass-through)?', 'ajută deprecierea leului la prognoza inflației (transmiterea cursului de schimb în prețuri)?'),
       T('$\\bY_t = (g_t, s_t, \\pi_t, i_t)\'$ with $s_t$ the quarterly EUR/RON change, from @{b4.q0}; model: B3', '$\\bY_t = (g_t, s_t, \\pi_t, i_t)\'$, cu $s_t$ variația trimestrială a cursului EUR/RON, din @{b4.q0}; model: B3'),
       [T('Choose $p$ with the three criteria ($p \\le 4$) and estimate a VAR(2).', 'Alegeți $p$ după cele trei criterii ($p \\le 4$) și estimați un VAR(2).'),
        T('Test whether $s$ Granger-causes $\\pi$, $i$ and $g$, and whether $i$ and $\\pi$ Granger-cause $s$.', 'Testați dacă $s$ cauzează în sens Granger pe $\\pi$, $i$ și $g$ și dacă $i$ și $\\pi$ cauzează pe $s$.'),
        T('Compute the share of EUR/RON shocks in the FEVD of inflation at 1, 4, 8 and 12 quarters.', 'Calculați ponderea șocurilor EUR/RON în FEVD pentru inflație la 1, 4, 8 și 12 trimestre.'),
        T('Interpretation: does the result mean that the exchange rate does not affect Romanian prices?', 'Interpretare: înseamnă rezultatul că cursul de schimb nu influențează prețurile din România?')],
       T('three orders, five $F$ tests, four shares and two sentences', 'trei ordine, cinci teste $F$, patru ponderi și două fraze'), size='footnotesize', nb='B4')

D.frame(T('B4: solution [Proposed]', 'B4: rezolvare [Propus]'), fig('ch6_sem_b4', h='0.32') + items(
    T('Orders: AIC @{b4.ic.aic}, BIC @{b4.ic.bic}, HQ @{b4.ic.hq}; VAR(2) on @{b4.n} quarters, largest modulus @{b4.mod}',
      'Ordine: AIC @{b4.ic.aic}, BIC @{b4.ic.bic}, HQ @{b4.ic.hq}; VAR(2) pe @{b4.n} de trimestre, cel mai mare modul @{b4.mod}'),
    T('$s \\to \\pi$: $F = @{b4.dspi.F}$ (p @{b4.dspi.p}); $s \\to i$: p @{b4.dsi.p}; $s \\to g$: p @{b4.dsg.p}; $i \\to s$: p @{b4.ids.p}; $\\pi \\to s$: p @{b4.pids.p}',
      '$s \\to \\pi$: $F = @{b4.dspi.F}$ (p @{b4.dspi.p}); $s \\to i$: p @{b4.dsi.p}; $s \\to g$: p @{b4.dsg.p}; $i \\to s$: p @{b4.ids.p}; $\\pi \\to s$: p @{b4.pids.p}'),
    T('EUR/RON shocks explain @{b4.fe.1}\\%, @{b4.fe.4}\\%, @{b4.fe.8}\\% and @{b4.fe.12}\\% of the inflation forecast variance; the response band never excludes zero',
      'Șocurile EUR/RON explică @{b4.fe.1}\\%, @{b4.fe.4}\\%, @{b4.fe.8}\\% și @{b4.fe.12}\\% din varianța prognozei inflației; banda răspunsului nu exclude niciodată zero'),
    T('Interpretation: no; the leu moved little (a managed float), quarterly averages blur the timing, and 12-month inflation reacts slowly: the test has little power, absence of evidence is not evidence of absence',
      'Interpretare: nu; leul s-a mișcat puțin (curs flotant administrat), mediile trimestriale estompează momentul șocurilor, iar inflația anuală reacționează lent: testul are putere mică, iar lipsa dovezilor nu este o dovadă a lipsei efectului')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: open questions and AI critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: the price puzzle [Proposed]', 'C1: enigma prețurilor [Propus]'),
       T('why does inflation rise after a ROBOR shock in the VAR of B3, and does it change when the exchange rate or more lags are added?', 'de ce crește inflația după un șoc ROBOR în VAR-ul din B3 și se schimbă acest lucru dacă adăugăm cursul de schimb sau mai multe decalaje?'),
       T('the VARs of B3 and B4; ROBOR ordered last; models: B3, B4', 'VAR-urile din B3 și B4; ROBOR așezat ultimul; modele: B3, B4'),
       [T('Plot the response of inflation to a ROBOR shock in the VAR(2) of B3, the VAR(2) of B4 and a VAR(4) with the four variables of B4.', 'Reprezentați grafic răspunsul inflației la un șoc ROBOR în VAR(2) din B3, în VAR(2) din B4 și într-un VAR(4) cu cele patru variabile din B4.'),
        T('Report the largest and the smallest response and their horizons.', 'Raportați cel mai mare și cel mai mic răspuns și orizonturile lor.'),
        T('List two pieces of information that the central bank has and the VAR does not.', 'Enumerați două informații pe care banca centrală le are, iar VAR-ul nu.'),
        T('Interpretation: is a positive response of inflation a proof that higher rates raise prices?', 'Interpretare: este un răspuns pozitiv al inflației o dovadă că dobînzile mai mari cresc prețurile?')],
       T('the chart, six numbers and a project plan', 'graficul, șase valori și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: reference analysis [Proposed]', 'C1: analiză de referință [Propus]'), fig('ch6_sem_c1', h='0.34') + items(
    T('Peak response: @{c1.0.max} pp after @{c1.0.h} quarters (B3); @{c1.1.max} pp after @{c1.1.h} (B4); @{c1.2.max} pp after @{c1.2.h} (VAR(4)), then @{c1.2.min} pp after @{c1.2.hmin}',
      'Răspunsul maxim: @{c1.0.max} pp după @{c1.0.h} trimestre (B3); @{c1.1.max} pp după @{c1.1.h} (B4); @{c1.2.max} pp după @{c1.2.h} (VAR(4)), apoi @{c1.2.min} pp după @{c1.2.hmin}'),
    T('The bank sees inflation expectations, energy and food price forecasts, wage data and fiscal plans; when it raises the rate because it expects inflation, the VAR labels the move a ``shock\'\' followed by inflation',
      'Banca vede așteptările inflaționiste, prognozele prețurilor la energie și alimente, datele despre salarii și planurile fiscale; cînd crește dobînda pentru că anticipează inflația, VAR-ul numește mișcarea „șoc” urmat de inflație'),
    T('Interpretation: no; it signals a badly identified shock (omitted information), the price puzzle of \\refSimsPP', 'Interpretare: nu; semnalează un șoc identificat greșit (informație omisă), enigma prețurilor din \\refSimsPP'),
    T('Project: add a commodity price index, the euro-area rate or survey expectations, and compare the responses', 'Proiect: adăugați un indice al prețurilor materiilor prime, dobînda din zona euro sau așteptările din anchete și comparați răspunsurile')) + qlsem(),
    'scriptsize', instructor_only=True)

D.frame(T('C2: audit an AI answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to analyse the VAR of B3. The answer:', 'Un student a cerut unui asistent AI să analizeze VAR-ul din B3. Răspunsul:'),
    T('\\aiprompt{(a) AIC chooses p = @{c2.aic}, so the true lag order of the economy is @{c2.aic}.}', '\\aiprompt{(a) AIC alege p = @{c2.aic}, deci ordinul adevărat al economiei este @{c2.aic}.}'),
    T('\\aiprompt{(b) The largest eigenvalue modulus of A1 is @{c2.eA1}, so the VAR(2) is explosive.}', '\\aiprompt{(b) Cel mai mare modul al valorilor proprii ale lui A1 este @{c2.eA1}, deci VAR(2) este exploziv.}'),
    T('\\aiprompt{(c) The portmanteau test at 8 lags has p = @{c2.lb}, so the residuals are white noise.}', '\\aiprompt{(c) Testul portmanteau la 8 decalaje are p = @{c2.lb}, deci reziduurile sînt zgomot alb.}'),
    T('\\aiprompt{(d) GDP growth Granger-causes ROBOR, so faster growth causes the BNR to raise rates.}', '\\aiprompt{(d) Creșterea PIB-ului cauzează ROBOR în sens Granger, deci creșterea mai rapidă determină BNR să crească dobînzile.}'),
    T('\\aiprompt{(e) Cholesky impulse responses do not depend on the order of the variables.}', '\\aiprompt{(e) Răspunsurile la impuls Cholesky nu depind de ordinea variabilelor.}'),
    T('\\aiprompt{(f) In each row of the FEVD the shares add up to 100\\%.}', '\\aiprompt{(f) Pe fiecare rînd al FEVD ponderile însumează 100\\%.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: criteria choose a model for a sample; BIC chooses @{c2.bic}, HQ @{b3.ic.hq}; there is no ``true\'\' order to discover', '(a) Greșit: criteriile aleg un model pentru un eșantion; BIC alege @{c2.bic}, HQ @{b3.ic.hq}; nu există un ordin „adevărat” de descoperit'),
    T('(b) Wrong: the stability of a VAR(2) is decided by the companion matrix; its largest modulus is @{c2.eF} $< 1$: stable', '(b) Greșit: stabilitatea unui VAR(2) se decide pe matricea companion; cel mai mare modul al ei este @{c2.eF} $< 1$: stabil'),
    T('(c) Wrong: p = @{c2.lb} $< 0.05$ rejects white noise at 8 lags (at 12 lags it is not rejected)', '(c) Greșit: p = @{c2.lb} $< 0{,}05$ respinge zgomotul alb la 8 decalaje (la 12 decalaje nu se respinge)'),
    T('(d) Wrong: Granger causality is predictability; ROBOR is a market rate and both may react to a third factor', '(d) Greșit: cauzalitatea Granger înseamnă predictibilitate; ROBOR este o dobîndă de piață și ambele pot reacționa la un al treilea factor'),
    T('(e) Wrong: the order decides which variable reacts on impact (A3, B3)', '(e) Greșit: ordinea decide ce variabilă reacționează la impact (A3, B3)'),
    T('(f) Correct', '(f) Corect')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-up', 'Încheiere')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Stability: eigenvalues of $\\bA$ (or of the companion matrix) inside the unit circle; then forecasts converge to $\\boldsymbol{\\mu}$', 'Stabilitatea: valorile proprii ale lui $\\bA$ (sau ale matricei companion) în interiorul cercului unitate; atunci prognozele converg spre $\\boldsymbol{\\mu}$'),
    T('Granger $F$ test: compare the equation with and without the lags of the candidate variable', 'Testul Granger $F$: comparăm ecuația cu și fără decalajele variabilei candidate'),
    T('Cholesky responses depend on the ordering; the generalised responses do not', 'Răspunsurile Cholesky depind de ordine; cele generalizate nu'),
    T('S\\&P 500 leads the BET by one day because of the time zones; ROBOR follows Romanian inflation', 'S\\&P 500 precedă BET cu o zi din cauza fusurilor orare; ROBOR urmează inflația din România'),
    T('An AI answer is a draft: check the direction of each claim and which matrix decides stability', 'Un răspuns AI este o ciornă: verificați sensul fiecărei afirmații și ce matrice decide stabilitatea')))

D.frame(T('After the seminar', 'După seminar'), items(
    T('Lecture 6 develops each topic of today: vector processes, the companion form, lag selection, diagnostics, Granger causality and its pitfalls, impulse responses, FEVD, spillovers, forecasting, structural VARs',
      'Cursul 6 dezvoltă fiecare temă de azi: procese vectoriale, forma companion, alegerea numărului de decalaje, diagnosticarea, cauzalitatea Granger și capcanele ei, răspunsurile la impuls, FEVD, spillover-ul, prognoza, VAR-urile structurale'),
    T('Try the [Proposed] tasks in the notebook', 'Încercați cerințele [Propus] în notebook'),
    T('C1 can grow into a team project: monetary policy shocks in Romania with an extended VAR', 'C1 poate deveni un proiect de echipă: șocurile de politică monetară din România într-un VAR extins'),
    T('Reading: \\refHP, Ch.~7; \\refFPP, Sec.~12.3; \\refSW', 'Lectură: \\refHP, cap.~7; \\refFPP, secț.~12.3; \\refSW'),
    T('\\textbf{The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class}',
      '\\textbf{Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar}')))

D.references(bib(['FPP', 'Granger', 'HP', 'Lut', 'PS', 'Sims', 'SimsPP', 'SW']), per=16)

if __name__ == '__main__':
    finalize(D.write(V))
