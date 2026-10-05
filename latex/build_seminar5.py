r"""
build_seminar5.py -- Seminarul 5 (Volatilitate condiționată: ARCH și GARCH), EN + RO dintr-o singură sursă
==========================================================================================================
Seminarul are loc ÎNAINTEA cursului 5: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu o întrebare de interpretare, C o întrebare deschisă și
critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în versiunea
profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_05/sem5_results.json (seminar5.py).
Ieșire:
  EN/Seminars/seminar5_conditional_volatility_garch.tex          (+ _solutions.tex)
  RO/Seminarii/seminar5_volatilitate_conditionata_garch_ro.tex   (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_05/seminar5.py && python3 latex/build_seminar5.py && python3 latex/tsa_build.py compile 5
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, fig   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch5_common import IG, NAMES, REFS, T, bib, date, finalize, load_sem, pv   # noqa: E402

S = load_sem()
V = Values()
D = Deck(5, 'seminar', refs=REFS)


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


def qlsem():
    return '\\quantlet{TSA\\_ch5\\_seminar}{\\qlurl{TSA_ch5_seminar}}'


# =============================================================================
# CIFRE
# =============================================================================
A = S['A']
for a in ['a1', 'a2']:
    d = A[a]
    V.put(f'{a}.pers', d['pers'], 2)
    V.put(f'{a}.uv', d['uv'], 3)
    V.put(f'{a}.vol', d['vol_lr'], 2)
    V.put(f'{a}.hl', d['hl'], 1)
    V.put(f'{a}.lnp', math.log(d['pers']), 5)
    V.put(f'{a}.s2', d['s2_next'], 3)
    V.put(f'{a}.s', math.sqrt(d['s2_next']), 3)
for a in ['a3', 'a4', 'a4n']:
    d = A[a]
    V.put(f'{a}.s2bar', d['s2bar'], 3)
    for h, v in d['f'].items():
        V.put(f'{a}.f{h}', v, 3)
    V.put(f'{a}.sum', d['sumH'], 2)
    V.put(f'{a}.vol', d['volH'], 2)
    V.put(f'{a}.volsq', d['volH_sqrt'], 2)
    V.put(f'{a}.vollr', d['volH_lr'], 2)
    V.put(f'{a}.q', d['q'], 3)
    V.put(f'{a}.qa', -d['q'], 3)
    V.put(f'{a}.v1', d['var1'], 2)
    V.put(f'{a}.vH', d['varH'], 2)
    V.put(f'{a}.fac', d['factor'], 3)
V.put('a3.p4', 0.98 ** 4, 4)
V.put('a3.ex', A['a1']['s2_next'] - 1, 2)
V.put('a4.inc', 100 * (A['a4']['var1'] / A['a4n']['var1'] - 1), 0)
V.put('a1.inc', 100 * (A['a1']['s2_next'] / 1.5 - 1), 0)
V.put('a1.wk', A['a1']['hl'] / 5, 0)
d = A['a5']
V.put('a5.lm', d['lm'], 1)
V.put('a5.crit', d['crit'], 2)
V.raw('a5.p', pv(d['p']))
V.put('a5.qcrit', d['Qcrit'], 2)
V.raw('a5.qp', pv(d['Qp']))
V.put('a5.uv', d['uv'], 1)
V.put('a5.k', d['kurt'], 1)
d = A['a6']
V.put('a6.neg', d['gjr']['-2.0'], 2)
V.put('a6.pos', d['gjr']['2.0'], 2)
V.put('a6.ratio', d['ratio'], 2)
V.put('a6.pers', d['pers'], 2)
V.put('a6.hl', d['hl'], 1)
V.put('a6.egneg.ln', d['eg']['-2.0']['ln'], 4)
V.put('a6.egneg', d['eg']['-2.0']['s2'], 3)
V.put('a6.egpos.ln', d['eg']['2.0']['ln'], 4)
V.put('a6.egpos', d['eg']['2.0']['s2'], 3)


def put_b1(key, d):
    p, se = d['params'], d['se']
    ar = d['ar_name']
    for a, b in [('c', 'Const'), ('phi', ar), ('om', 'omega'), ('a', 'alpha[1]'), ('b', 'beta[1]'), ('nu', 'nu')]:
        dd = 2 if a == 'nu' else (4 if a == 'om' else 3)
        V.put(f'{key}.{a}', p[b], dd)
        V.put(f'{key}.{a}.se', se[b], dd)
    V.int(f'{key}.n', d['n'])
    V.raw(f'{key}.y0', d['first'][:4])
    V.put(f'{key}.vs', d['vol_sample'], 1)
    V.put(f'{key}.last', d['last_vol'], 1)
    V.put(f'{key}.out', 100 * d['share_out'], 1)
    V.put(f'{key}.ab', p['alpha[1]'] + p['beta[1]'], 4)
    V.put(f'{key}.lm', d['lm_e']['lm'], 0)
    V.put(f'{key}.lmr2', d['lm_e']['r2'], 3)
    V.put(f'{key}.q2e', d['lb_e2'][0], 0)
    V.put(f'{key}.qz', d['lb_z'][0], 1)
    V.raw(f'{key}.qzp', pv(d['lb_z'][1]))
    V.put(f'{key}.qz2', d['lb_z2'][0], 1 if d['lb_z2'][0] > 0.05 else 3)
    V.raw(f'{key}.qz2p', pv(d['lb_z2'][1]))
    V.put(f'{key}.phio', d['phi_ols'], 3)
    V.put(f'{key}.seo', d['se_ols'], 3)
    if d['pers'] >= IG:
        V.raw(f'{key}.pers', '⁅1.000⁆')
        V.raw(f'{key}.hl', '--')
        V.raw(f'{key}.vlr', '--')
    else:
        V.put(f'{key}.pers', d['pers'], 3)
        V.put(f'{key}.hl', d['hl'], 1)
        V.put(f'{key}.vlr', d['vol_lr'], 1)
        V.put(f'{key}.uv', d['uv'], 3)
        V.put(f'{key}.1mp', 1 - d['pers'], 4)
        V.put(f'{key}.lnp', math.log(d['pers']), 5)
    for e in ['2008', '2020']:
        if f'peak{e}' in d:
            V.put(f'{key}.p{e}', d[f'peak{e}'], 0)
        else:
            V.raw(f'{key}.p{e}', '--')


put_b1('b1', S['B1'])
for k, d in S['B2'].items():
    put_b1(f'b2.{k}', d)


def put_b3(key, d):
    V.put(f'{key}.g', d['gamma'], 3)
    V.put(f'{key}.gt', d['t_gamma'], 1)
    V.put(f'{key}.a', d['alpha_gjr'], 3)
    V.put(f'{key}.lr', d['lr'], 1)
    V.raw(f'{key}.lrp', pv(d['p_lr']))
    V.put(f'{key}.db', d['bic_gjr'] - d['bic_garch'], 1)
    V.put(f'{key}.sb', d['sb']['joint'], 1)
    V.raw(f'{key}.sbp', pv(d['sb']['joint_p']))
    V.put(f'{key}.sbt', d['sb']['sign_t'], 2)
    if 'eg_gamma' in d:
        V.put(f'{key}.eg', d['eg_gamma'], 3)
        V.put(f'{key}.egt', d['eg_t'], 1)


B3 = S['B3']
put_b3('b3', B3)
V.put('b3.b', B3['beta_gjr'], 3)
V.put('b3.bg', B3['bic_garch'], 1)
V.put('b3.bj', B3['bic_gjr'], 1)
V.put('b3.neg', B3['nic_neg'], 2)
V.put('b3.pos', B3['nic_pos'], 2)
V.put('b3.ratio', B3['ratio'], 2)
V.put('b3.sq', math.sqrt(B3['ratio']), 2)
V.put('b3.pj', B3['pers_gjr'], 3)
for k, d in S['B4'].items():
    put_b3(f'b4.{k}', d)


def put_b5(key, d):
    V.int(f'{key}.n', d['n'])
    V.put(f'{key}.qg', d['q_garch'], 3)
    V.put(f'{key}.qe', d['q_ewma'], 3)
    V.put(f'{key}.qgw', d['q_garch_wo'], 3)
    V.put(f'{key}.qew', d['q_ewma_wo'], 3)
    V.put(f'{key}.dm', d['dm']['t'], 2)
    V.put(f'{key}.dmm', d['dm']['mean'], 3)
    V.raw(f'{key}.dmp', pv(d['dm']['p']))
    V.raw(f'{key}.ng', str(d['nexc_garch']))
    V.raw(f'{key}.ne', str(d['nexc_ewma']))
    V.put(f'{key}.eg', 100 * d['exc_garch'], 2)
    V.put(f'{key}.ee', 100 * d['exc_ewma'], 2)
    V.raw(f'{key}.exp', str(round(d['expected'])))
    V.raw(f'{key}.day', date(d['max_day']))
    V.put(f'{key}.share', 100 * d['share_max'], 0)


put_b5('b5', S['B5'])
for k, d in S['B6'].items():
    put_b5(f'b6.{k}', d)
c1rows = []
for i, r in enumerate(S['C1']):
    pers = '⁅1.000⁆' if r['pers'] >= IG else f'⁅{r["pers"]:.3f}⁆'
    c1rows.append(f'{r["from"][:4]}--{r["to"][:4]} & {r["n"]:,} & ⁅{r["vol"]:.1f}⁆ & ⁅{100 * r["big"]:.1f}⁆ & {pers} & '
                  f'$⁅{r["alpha"]:.3f}⁆$ & $⁅{r["nu"]:.2f}⁆$ & ⁅{r["maxabs"]:.2f}⁆'.replace(',', '\\,'))
    V.put(f'c1.v{i}', r['vol'], 1)
    V.put(f'c1.b{i}', 100 * r['big'], 1)
    V.raw(f'c1.d{i}', date(r['date_max']))
C2 = S['C2']
V.put('c2.a', C2['alpha'], 3)
V.put('c2.b', C2['beta'], 3)
V.put('c2.om', C2['omega'], 4)
V.put('c2.pers', C2['pers'], 3)
V.put('c2.hl', C2['hl'], 0)
V.put('c2.hlw', C2['hl_wrong'], 1)
V.put('c2.uvw', C2['uv_wrong'], 3)
V.put('c2.volw', C2['vol_wrong'], 1)
V.put('c2.uv', C2['uv'], 2)
V.put('c2.vol', C2['vol_lr'], 1)
V.put('c2.nu', C2['nu'], 2)
V.put('c2.qt', C2['q_t'], 3)
V.put('c2.qta', -C2['q_t'], 3)
V.put('c2.qn', C2['q_n'], 3)
V.put('c2.lb', C2['lb_z2'][0], 1)
V.put('c2.lbp', C2['lb_z2'][1], 2)
V.put('c2.lbz', C2['lb_z'][0], 1)
V.put('c2.phio', C2['phi_ols'], 3)
V.put('c2.seo', C2['se_ols'], 3)
V.put('c2.to', C2['phi_ols'] / C2['se_ols'], 1)
V.put('c2.sehc', C2['se_hc'], 3)
V.put('c2.thc', C2['phi_ols'] / C2['se_hc'], 1)
V.put('c2.phig', C2['phi_g'], 3)
V.put('c2.tg', C2['phi_g'] / C2['se_g'], 1)
V.put('c2.r2', C2['phi_ols'] ** 2, 3)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how do we detect, model, estimate and forecast a variance that changes every day?',
       '\\textbf{Întrebarea}: cum detectăm, modelăm, estimăm și prognozăm o varianță care se schimbă în fiecare zi?'),
     [T('this seminar comes \\textbf{before} Lecture 5: the section ``What you need today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 5: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: GARCH algebra, forecasts and VaR 1\\%, an ARCH-LM test, asymmetric news, on paper', 'Partea A: algebra GARCH, prognoze și VaR 1\\%, un test ARCH-LM, știri asimetrice, pe hîrtie'),
      T('Part B: ARCH effects and AR(1)-GARCH(1,1)-t, asymmetry, out-of-sample forecasts on the BET, S\\&P 500, EUR/RON and Bitcoin',
        'Partea B: efecte ARCH și AR(1)-GARCH(1,1)-t, asimetria, prognoze în afara eșantionului pentru BET, S\\&P 500, EUR/RON și Bitcoin'),
      T('Part C: an open question on the leu and an AI answer to audit', 'Partea C: o întrebare deschisă despre leu și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; its first cell installs the \\texttt{arch} package (\\texttt{pip install arch})',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; prima celulă instalează pachetul \\texttt{arch} (\\texttt{pip install arch})')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('persistence, long-run variance, half-life and one GARCH(1,1) step', 'persistența, varianța de lungă durată, timpul de înjumătățire și un pas GARCH(1,1)') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('multi-step forecasts, 10-day volatility and VaR 1\\%', 'prognoze pe mai mulți pași, volatilitatea pe 10 zile și VaR 1\\%') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('an ARCH-LM test from regression output; news impact in GJR and EGARCH', 'un test ARCH-LM din rezultatele unei regresii; impactul știrilor în GJR și EGARCH') + ' & ' + SP + ' & A5, A1',
     'B1, B2 & ' + T('ARCH effects and AR(1)-GARCH(1,1)-t: BET; EUR/RON and Bitcoin', 'efecte ARCH și AR(1)-GARCH(1,1)-t: BET; EUR/RON și Bitcoin') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('asymmetry and the leverage effect: S\\&P 500; BET, EUR/RON, Bitcoin', 'asimetria și efectul de levier: S\\&P 500; BET, EUR/RON, Bitcoin') + ' & ' + SP + ' & B3',
     'B5, B6 & ' + T('GARCH against EWMA out of sample, VaR 1\\%: S\\&P 500; BET, EUR/RON', 'GARCH față de EWMA în afara eșantionului, VaR 1\\%: S\\&P 500; BET, EUR/RON') + ' & ' + SP + ' & B5',
     'C1, C2 & ' + T('is the leu calmer than it used to be? what is wrong in an AI answer?', 'este leul mai liniștit decît înainte? ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, A1'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați dumneavoastră, după model')))

D.frame(T('Data used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Frequency}', '\\textbf{Frecvența}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    ['S\\&P 500, BET & EODHD & ' + T('daily, close', 'zilnic, închidere') + ' & 2000--2026',
     'EUR/RON & ' + T('BNR reference rate', 'cursul de referință BNR') + ' & ' + T('daily', 'zilnic') + ' & 2005--2026',
     'Bitcoin & EODHD & ' + T('daily, 7 days a week', 'zilnic, 7 zile pe săptămînă') + ' & 2014--2026'],
    size='footnotesize') + items(
    T('Daily log returns in \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekends and repeated holiday closes are dropped (except Bitcoin)',
      'Randamente logaritmice zilnice în \\%: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$; weekendurile și închiderile repetate din zilele libere se elimină (cu excepția Bitcoin)'),
    T('EUR/RON moves about 0.3\\% a day: the notebook estimates its models on $10\\,r_t$ and converts the results back to \\%',
      'EUR/RON variază cu circa 0,3\\% pe zi: notebook-ul estimează modelele pe $10\\,r_t$ și convertește rezultatele înapoi în \\%'),
    T('In the notebook: \\texttt{returns(\'bet\')}, \\texttt{arch\\_lm(x)}, \\texttt{fit(r, mean=\'AR\', dist=\'t\')} (package \\texttt{arch}); no account or key is needed',
      'În notebook: \\texttt{returns(\'bet\')}, \\texttt{arch\\_lm(x)}, \\texttt{fit(r, mean=\'AR\', dist=\'t\')} (pachetul \\texttt{arch}); nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What you need today', 'Noțiuni necesare azi')

D.frame(T('What you need today (1/4): ARCH effects', 'Noțiuni necesare azi (1/4): efecte ARCH'), items(
    (T('$r_t = \\mu_t + \\varepsilon_t$, $\\varepsilon_t = \\sigma_t z_t$; $z_t$ i.i.d., mean 0, variance 1 (the \\textbf{innovations})', '$r_t = \\mu_t + \\varepsilon_t$, $\\varepsilon_t = \\sigma_t z_t$; $z_t$ i.i.d., media 0, varianța 1 (\\textbf{inovațiile})'),
     [T('$\\mu_t$: the \\textbf{conditional mean} (an ARMA model, Chapter 2); $\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\text{past})$: the \\textbf{conditional variance}',
        '$\\mu_t$: \\textbf{media condiționată} (un model ARMA, Capitolul 2); $\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\text{trecut})$: \\textbf{varianța condiționată}'),
      T('daily returns are almost uncorrelated, but their squares are correlated: \\textbf{volatility clustering}', 'randamentele zilnice sînt aproape necorelate, dar pătratele lor sînt corelate: \\textbf{volatility clustering}')]),
    (T('\\textbf{ARCH-LM test} \\refEngle: regress $\\hat\\varepsilon_t^2$ on a constant and $\\hat\\varepsilon_{t-1}^2, \\dots, \\hat\\varepsilon_{t-q}^2$ by OLS',
       '\\textbf{Testul ARCH-LM} \\refEngle: regresia prin OLS a lui $\\hat\\varepsilon_t^2$ pe o constantă și pe $\\hat\\varepsilon_{t-1}^2, \\dots, \\hat\\varepsilon_{t-q}^2$'),
     [T('$H_0$: no ARCH effects; $\\mathrm{LM} = nR^2 \\sim \\chi^2(q)$; 5\\% values: $\\chi^2(5)$ 11.07, $\\chi^2(10)$ 18.31', '$H_0$: fără efecte ARCH; $\\mathrm{LM} = nR^2 \\sim \\chi^2(q)$; valorile de 5\\%: $\\chi^2(5)$ 11,07, $\\chi^2(10)$ 18,31'),
      T('the Ljung--Box $Q(m)$ of $\\hat\\varepsilon_t^2$ \\refLB\\ is a second test with the same $H_0$', 'statistica Ljung--Box $Q(m)$ a lui $\\hat\\varepsilon_t^2$ \\refLB\\ este un al doilea test cu aceeași $H_0$')]),
    T('\\textbf{Kurtosis} $K = E[(r - \\mu)^4]/\\sigma^4$: 3 for the Normal distribution, larger for heavy tails', '\\textbf{Coeficientul de boltire} $K = E[(r - \\mu)^4]/\\sigma^4$: 3 pentru distribuția Normală, mai mare pentru cozi groase')))

D.frame(T('What you need today (2/4): ARCH, GARCH and EWMA', 'Noțiuni necesare azi (2/4): ARCH, GARCH și EWMA'), items(
    (T('\\textbf{ARCH(1)} \\refEngle: $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2$, $\\omega > 0$, $0 \\le \\alpha < 1$', '\\textbf{ARCH(1)} \\refEngle: $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2$, $\\omega > 0$, $0 \\le \\alpha < 1$'),
     [T('unconditional variance $\\omega/(1 - \\alpha)$; with Normal $z_t$ and $3\\alpha^2 < 1$: kurtosis $3(1 - \\alpha^2)/(1 - 3\\alpha^2)$',
        'varianța necondiționată $\\omega/(1 - \\alpha)$; cu $z_t$ Normale și $3\\alpha^2 < 1$: coeficientul de boltire $3(1 - \\alpha^2)/(1 - 3\\alpha^2)$')]),
    (T('\\textbf{GARCH(1,1)} \\refBoll: $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, $\\omega > 0$, $\\alpha, \\beta \\ge 0$',
       '\\textbf{GARCH(1,1)} \\refBoll: $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, $\\omega > 0$, $\\alpha, \\beta \\ge 0$'),
     [T('\\textbf{persistence} $\\alpha + \\beta$; \\textbf{long-run variance} $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$ if $\\alpha + \\beta < 1$; annual volatility $\\sqrt{252\\,\\bar\\sigma^2}$',
        '\\textbf{persistența} $\\alpha + \\beta$; \\textbf{varianța de lungă durată} $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$ dacă $\\alpha + \\beta < 1$; volatilitatea anuală $\\sqrt{252\\,\\bar\\sigma^2}$'),
      T('\\textbf{half-life} $h_{1/2} = \\ln 0.5/\\ln(\\alpha + \\beta)$ days: the time for half of a variance shock to disappear', '\\textbf{timpul de înjumătățire} $h_{1/2} = \\ln 0{,}5/\\ln(\\alpha + \\beta)$ zile: timpul în care dispare jumătate dintr-un șoc de varianță')]),
    (T('\\textbf{EWMA} \\refRM: $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1 - \\lambda)r_{t-1}^2$ (exponential smoothing of $r_t^2$, Chapter 0)', '\\textbf{EWMA} \\refRM: $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1 - \\lambda)r_{t-1}^2$ (netezirea exponențială a lui $r_t^2$, Capitolul 0)'),
     [T('a GARCH with $\\omega = 0$ and $\\alpha + \\beta = 1$: an \\textbf{IGARCH} (integrated GARCH): no half-life, no long-run variance', 'un GARCH cu $\\omega = 0$ și $\\alpha + \\beta = 1$: un \\textbf{IGARCH} (GARCH integrat): fără timp de înjumătățire și fără varianță de lungă durată'),
      T('\\textbf{AR(1)-GARCH(1,1)}: $r_t = c + \\phi r_{t-1} + \\varepsilon_t$ with GARCH errors, both parts estimated together', '\\textbf{AR(1)-GARCH(1,1)}: $r_t = c + \\phi r_{t-1} + \\varepsilon_t$ cu erori GARCH, ambele părți estimate împreună')])))

D.frame(T('What you need today (3/4): forecasts, VaR and estimation', 'Noțiuni necesare azi (3/4): prognoze, VaR și estimare'), items(
    (T('Forecasts made at the end of day $t$', 'Prognoze făcute la sfîrșitul zilei $t$'),
     [T('one day: $\\sigma_{t+1}^2 = \\omega + \\alpha(r_t - \\mu)^2 + \\beta\\sigma_t^2$; $h$ days: $E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$',
        'o zi: $\\sigma_{t+1}^2 = \\omega + \\alpha(r_t - \\mu)^2 + \\beta\\sigma_t^2$; $h$ zile: $E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$'),
      T('$H$-day variance: $\\sum_{h=1}^{H}E_t[\\sigma_{t+h}^2] = H\\bar\\sigma^2 + (\\sigma_{t+1}^2 - \\bar\\sigma^2)\\dfrac{1 - (\\alpha + \\beta)^H}{1 - \\alpha - \\beta}$',
        'varianța pe $H$ zile: $\\sum_{h=1}^{H}E_t[\\sigma_{t+h}^2] = H\\bar\\sigma^2 + (\\sigma_{t+1}^2 - \\bar\\sigma^2)\\dfrac{1 - (\\alpha + \\beta)^H}{1 - \\alpha - \\beta}$')]),
    (T('\\textbf{VaR 1\\%} (value at risk): the loss exceeded with probability 1\\%: $\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}\\,q_{0.01}(z))$',
       '\\textbf{VaR 1\\%} (value at risk, valoarea expusă la risc): pierderea depășită cu probabilitatea 1\\%: $\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}\\,q_{0{,}01}(z))$'),
     [T('Normal: $q_{0.01} = -2.326$; standardised Student-t: $q_{0.01} = t_\\nu^{-1}(0.01)\\sqrt{(\\nu - 2)/\\nu}$, for example $t_5^{-1}(0.01) = -3.365$',
        'Normală: $q_{0{,}01} = -2{,}326$; Student-t standardizată: $q_{0{,}01} = t_\\nu^{-1}(0{,}01)\\sqrt{(\\nu - 2)/\\nu}$, de exemplu $t_5^{-1}(0{,}01) = -3{,}365$')]),
    (T('\\textbf{MLE} (maximum likelihood estimation): $\\hat\\theta$ maximises $\\ell(\\theta) = -\\frac12\\sum_t[\\ln 2\\pi + \\ln\\sigma_t^2 + (r_t - \\mu_t)^2/\\sigma_t^2]$; \\texttt{arch} reports robust SE (standard errors) \\refBW',
       '\\textbf{MLE} (maximum likelihood estimation, estimarea prin verosimilitate maximă): $\\hat\\theta$ maximizează $\\ell(\\theta) = -\\frac12\\sum_t[\\ln 2\\pi + \\ln\\sigma_t^2 + (r_t - \\mu_t)^2/\\sigma_t^2]$; \\texttt{arch} raportează SE (erori standard) robuste \\refBW'),
     [T('heavy-tailed innovations: standardised Student-t with $\\nu$ degrees of freedom \\refBollT; LR (likelihood ratio) test $2(\\ell_1 - \\ell_0) \\sim \\chi^2(r)$, $\\chi^2_{0.95}(1) = 3.84$',
        'inovații cu cozi groase: Student-t standardizată cu $\\nu$ grade de libertate \\refBollT; testul LR (likelihood ratio, raportul de verosimilitate) $2(\\ell_1 - \\ell_0) \\sim \\chi^2(r)$, $\\chi^2_{0{,}95}(1) = 3{,}84$')])))

D.frame(T('What you need today (4/4): asymmetry and forecast comparison', 'Noțiuni necesare azi (4/4): asimetria și compararea prognozelor'), items(
    (T('\\textbf{GJR-GARCH} \\refGJR: $\\sigma_t^2 = \\omega + (\\alpha + \\gamma I_{t-1})\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, $I_{t-1} = 1$ if $\\varepsilon_{t-1} < 0$',
       '\\textbf{GJR-GARCH} \\refGJR: $\\sigma_t^2 = \\omega + (\\alpha + \\gamma I_{t-1})\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, $I_{t-1} = 1$ dacă $\\varepsilon_{t-1} < 0$'),
     [T('\\textbf{leverage effect} \\refChristie: $\\gamma > 0$, falls raise volatility more than rises; persistence $\\alpha + \\beta + \\gamma/2$', '\\textbf{efectul de levier} \\refChristie: $\\gamma > 0$, scăderile cresc volatilitatea mai mult decît creșterile; persistența $\\alpha + \\beta + \\gamma/2$'),
      T('\\textbf{EGARCH} \\refNelson: $\\ln\\sigma_t^2 = \\omega + \\alpha(|z_{t-1}| - E|z_{t-1}|) + \\gamma z_{t-1} + \\beta\\ln\\sigma_{t-1}^2$; $E|z| = 0.7979$ for Normal $z$; leverage: $\\gamma < 0$',
        '\\textbf{EGARCH} \\refNelson: $\\ln\\sigma_t^2 = \\omega + \\alpha(|z_{t-1}| - E|z_{t-1}|) + \\gamma z_{t-1} + \\beta\\ln\\sigma_{t-1}^2$; $E|z| = 0{,}7979$ pentru $z$ Normal; levier: $\\gamma < 0$'),
      T('\\textbf{sign-bias test} \\refEN: $\\chi^2(3)$ on the standardised residuals $\\hat z_t = (r_t - \\hat\\mu_t)/\\hat\\sigma_t$; a rejection means a missing asymmetry',
        '\\textbf{testul de asimetrie (sign bias)} \\refEN: $\\chi^2(3)$ pe reziduurile standardizate $\\hat z_t = (r_t - \\hat\\mu_t)/\\hat\\sigma_t$; o respingere înseamnă că lipsește asimetria')]),
    (T('\\textbf{Out of sample}: the forecast $h_t$ for day $t$ uses only data up to $t-1$; the proxy of the true variance is $r_t^2$',
       '\\textbf{În afara eșantionului}: prognoza $h_t$ pentru ziua $t$ folosește doar date pînă la $t-1$; aproximarea varianței adevărate este $r_t^2$'),
     [T('\\textbf{QLIKE} loss \\refPatton: $L_t = r_t^2/h_t + \\ln h_t$; the smaller the mean, the better', 'pierderea \\textbf{QLIKE} \\refPatton: $L_t = r_t^2/h_t + \\ln h_t$; cu cît media este mai mică, cu atît mai bine'),
      T('\\textbf{DM} (Diebold--Mariano) test \\refDM: $t$ statistic of the mean of $L_t^A - L_t^B$ with HAC (autocorrelation-robust) SE; $t < -1.96$: A is better',
        'testul \\textbf{DM} (Diebold--Mariano) \\refDM: statistica $t$ a mediei lui $L_t^A - L_t^B$, cu SE de tip HAC (robuste la autocorelație); $t < -1{,}96$: A este mai bun'),
      T('\\textbf{VaR exceedance}: a day with $r_t < -\\mathrm{VaR}_t$; a correct VaR 1\\% is exceeded on about 1\\% of the days', '\\textbf{depășire VaR}: o zi cu $r_t < -\\mathrm{VaR}_t$; un VaR 1\\% corect este depășit în circa 1\\% dintre zile')])))

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: computations on paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: GARCH(1,1) algebra', 'A1: algebra GARCH(1,1)'),
         items(T('Daily returns in \\%: $\\omega = 0.02$, $\\alpha = 0.08$, $\\beta = 0.90$, $\\mu = 0$, 252 trading days per year.',
                 'Randamente zilnice în \\%: $\\omega = 0{,}02$, $\\alpha = 0{,}08$, $\\beta = 0{,}90$, $\\mu = 0$, 252 de zile de tranzacționare pe an.'),
               T('1. Compute the persistence and the long-run variance.', '1. Calculați persistența și varianța de lungă durată.'),
               T('2. Compute the annualised long-run volatility.', '2. Calculați volatilitatea de lungă durată anualizată.'),
               T('3. Compute the half-life of a variance shock.', '3. Calculați timpul de înjumătățire al unui șoc de varianță.'),
               T('4. Today $\\sigma_t^2 = 1.5$ and $r_t = -3\\%$; compute $\\sigma_{t+1}^2$ and $\\sigma_{t+1}$.', '4. Azi $\\sigma_t^2 = 1{,}5$ și $r_t = -3\\%$; calculați $\\sigma_{t+1}^2$ și $\\sigma_{t+1}$.'),
               T('Report: four numbers and one sentence.', 'Raportați: patru valori și o frază.')),
         items(T('1. $\\alpha + \\beta = @{a1.pers}$; $\\bar\\sigma^2 = 0.02/(1 - @{a1.pers}) = @{a1.uv}$', '1. $\\alpha + \\beta = @{a1.pers}$; $\\bar\\sigma^2 = 0.02/(1 - @{a1.pers}) = @{a1.uv}$'),
               T('2. $\\sqrt{252 \\times @{a1.uv}} = @{a1.vol}\\%$ per year', '2. $\\sqrt{252 \\times @{a1.uv}} = @{a1.vol}\\%$ pe an'),
               T('3. $\\ln 0.5/\\ln @{a1.pers} = -0.69315/(@{a1.lnp}) = @{a1.hl}$ days', '3. $\\ln 0.5/\\ln @{a1.pers} = -0.69315/(@{a1.lnp}) = @{a1.hl}$ zile'),
               T('4. $0.02 + 0.08 \\times 9 + 0.90 \\times 1.5 = @{a1.s2}$; $\\sigma_{t+1} = @{a1.s}\\%$', '4. $0.02 + 0.08 \\times 9 + 0.90 \\times 1.5 = @{a1.s2}$; $\\sigma_{t+1} = @{a1.s}\\%$'),
               T('A fall of 3\\% raises the variance by @{a1.inc}\\%; half of the excess is gone after @{a1.hl} trading days, about @{a1.wk} weeks.', 'O scădere de 3\\% crește varianța cu @{a1.inc}\\%; jumătate din excedent dispare după @{a1.hl} zile de tranzacționare, circa @{a1.wk} săptămîni.')),
         size='scriptsize')

D.proposed(T('A2: a more reactive market', 'A2: o piață mai reactivă'),
           items(T('$\\omega = 0.05$, $\\alpha = 0.12$, $\\beta = 0.85$, $\\mu = 0$; today $\\sigma_t^2 = 2.0$ and $r_t = +1.5\\%$. Model: A1.',
                   '$\\omega = 0{,}05$, $\\alpha = 0{,}12$, $\\beta = 0{,}85$, $\\mu = 0$; azi $\\sigma_t^2 = 2{,}0$ și $r_t = +1{,}5\\%$. Model: A1.'),
                 T('1. Compute the persistence, the long-run variance and the annualised long-run volatility.', '1. Calculați persistența, varianța de lungă durată și volatilitatea de lungă durată anualizată.'),
                 T('2. Compute the half-life.', '2. Calculați timpul de înjumătățire.'),
                 T('3. Compute $\\sigma_{t+1}^2$.', '3. Calculați $\\sigma_{t+1}^2$.'),
                 T('4. Write EWMA with $\\lambda = 0.94$ as a GARCH(1,1) and say which of the quantities in 1--2 exist for it.', '4. Scrieți EWMA cu $\\lambda = 0{,}94$ ca un GARCH(1,1) și precizați care dintre mărimile de la 1--2 există pentru el.'),
                 T('Report: four numbers and two sentences.', 'Raportați: patru valori și două fraze.')),
           items(T('1. $@{a2.pers}$; $\\bar\\sigma^2 = 0.05/0.03 = @{a2.uv}$; $\\sqrt{252 \\times @{a2.uv}} = @{a2.vol}\\%$', '1. $@{a2.pers}$; $\\bar\\sigma^2 = 0.05/0.03 = @{a2.uv}$; $\\sqrt{252 \\times @{a2.uv}} = @{a2.vol}\\%$'),
                 T('2. $\\ln 0.5/\\ln @{a2.pers} = @{a2.hl}$ days', '2. $\\ln 0.5/\\ln @{a2.pers} = @{a2.hl}$ zile'),
                 T('3. $0.05 + 0.12 \\times 2.25 + 0.85 \\times 2 = @{a2.s2}$: a rise also raises the variance; here it stays almost unchanged', '3. $0.05 + 0.12 \\times 2.25 + 0.85 \\times 2 = @{a2.s2}$: și o creștere mărește varianța; aici ea rămîne aproape neschimbată'),
                 T('4. $\\omega = 0$, $\\alpha = 0.06$, $\\beta = 0.94$: IGARCH; no long-run variance and no half-life (the factor is 1).', '4. $\\omega = 0$, $\\alpha = 0.06$, $\\beta = 0.94$: IGARCH; nu există varianță de lungă durată și nici timp de înjumătățire (factorul este 1).')),
           size='scriptsize')

D.solved(T('A3: forecasts and VaR 1\\%', 'A3: prognoze și VaR 1\\%'),
         items(T('The model of A1, with $\\sigma_{t+1}^2 = @{a1.s2}$ and Normal innovations.', 'Modelul din A1, cu $\\sigma_{t+1}^2 = @{a1.s2}$ și inovații Normale.'),
               T('1. Compute $E_t[\\sigma_{t+5}^2]$ and $E_t[\\sigma_{t+10}^2]$.', '1. Calculați $E_t[\\sigma_{t+5}^2]$ și $E_t[\\sigma_{t+10}^2]$.'),
               T('2. Compute the 10-day variance and volatility.', '2. Calculați varianța și volatilitatea pe 10 zile.'),
               T('3. Compare with the square-root-of-time rule $\\sqrt{10}\\,\\sigma_{t+1}$ and with $\\sqrt{10\\,\\bar\\sigma^2}$.', '3. Comparați cu regula rădăcinii pătrate a timpului, $\\sqrt{10}\\,\\sigma_{t+1}$, și cu $\\sqrt{10\\,\\bar\\sigma^2}$.'),
               T('4. Compute the one-day and the 10-day VaR 1\\%.', '4. Calculați VaR 1\\% pe o zi și pe 10 zile.'),
               T('Report: six numbers and one sentence.', 'Raportați: șase valori și o frază.')),
         items(T('1. $1 + 0.98^{4}(@{a1.s2} - 1) = 1 + @{a3.p4} \\times @{a3.ex} = @{a3.f5}$; $1 + 0.98^{9} \\times @{a3.ex} = @{a3.f10}$', '1. $1 + 0.98^{4}(@{a1.s2} - 1) = 1 + @{a3.p4} \\times @{a3.ex} = @{a3.f5}$; $1 + 0.98^{9} \\times @{a3.ex} = @{a3.f10}$'),
               T('2. $10 \\times 1 + @{a3.ex} \\times (1 - 0.98^{10})/0.02 = 10 + @{a3.ex} \\times @{a3.fac} = @{a3.sum}$; volatility $@{a3.vol}\\%$', '2. $10 \\times 1 + @{a3.ex} \\times (1 - 0.98^{10})/0.02 = 10 + @{a3.ex} \\times @{a3.fac} = @{a3.sum}$; volatilitatea $@{a3.vol}\\%$'),
               T('3. $\\sqrt{10 \\times @{a1.s2}} = @{a3.volsq}\\%$ (too high), $\\sqrt{10} = @{a3.vollr}\\%$ (too low)', '3. $\\sqrt{10 \\times @{a1.s2}} = @{a3.volsq}\\%$ (prea mare), $\\sqrt{10} = @{a3.vollr}\\%$ (prea mică)'),
               T('4. $2.326 \\times @{a1.s} = @{a3.v1}\\%$; $2.326 \\times @{a3.vol} = @{a3.vH}\\%$', '4. $2.326 \\times @{a1.s} = @{a3.v1}\\%$; $2.326 \\times @{a3.vol} = @{a3.vH}\\%$'),
               T('The high variance decays slowly: the 10-day risk lies between the two simple rules, closer to the first.', 'Varianța ridicată scade lent: riscul pe 10 zile este între cele două reguli simple, mai aproape de prima.')),
         size='scriptsize')

D.proposed(T('A4: forecasts with Student-t innovations', 'A4: prognoze cu inovații Student-t'),
           items(T('The model of A2, with $\\sigma_{t+1}^2 = @{a2.s2}$ and standardised Student-t innovations with $\\nu = 5$ ($t_5^{-1}(0.01) = -3.365$). Model: A3.',
                   'Modelul din A2, cu $\\sigma_{t+1}^2 = @{a2.s2}$ și inovații Student-t standardizate cu $\\nu = 5$ ($t_5^{-1}(0{,}01) = -3{,}365$). Model: A3.'),
                 T('1. Compute $E_t[\\sigma_{t+10}^2]$ and the 10-day variance.', '1. Calculați $E_t[\\sigma_{t+10}^2]$ și varianța pe 10 zile.'),
                 T('2. Compute the quantile $q_{0.01}$ of the standardised t.', '2. Calculați cuantila $q_{0{,}01}$ a distribuției t standardizate.'),
                 T('3. Compute the one-day and the 10-day VaR 1\\% with t and with Normal innovations.', '3. Calculați VaR 1\\% pe o zi și pe 10 zile, cu inovații t și cu inovații Normale.'),
                 T('Report: six numbers and one sentence.', 'Raportați: șase valori și o frază.')),
           items(T('1. $@{a4.s2bar} + 0.97^{9}(@{a2.s2} - @{a4.s2bar}) = @{a4.f10}$; sum $= @{a4.sum}$, volatility $@{a4.vol}\\%$', '1. $@{a4.s2bar} + 0.97^{9}(@{a2.s2} - @{a4.s2bar}) = @{a4.f10}$; suma $= @{a4.sum}$, volatilitatea $@{a4.vol}\\%$'),
                 T('2. $q = -3.365 \\times \\sqrt{3/5} = @{a4.q}$', '2. $q = -3.365 \\times \\sqrt{3/5} = @{a4.q}$'),
                 T('3. t: $@{a4.qa} \\times @{a2.s} = @{a4.v1}\\%$, 10 days $@{a4.vH}\\%$; Normal: $@{a4n.v1}\\%$ and $@{a4n.vH}\\%$', '3. t: $@{a4.qa} \\times @{a2.s} = @{a4.v1}\\%$, 10 zile $@{a4.vH}\\%$; Normală: $@{a4n.v1}\\%$ și $@{a4n.vH}\\%$'),
                 T('With the same variance, heavy tails raise VaR 1\\% by about @{a4.inc}\\%.', 'La aceeași varianță, cozile groase cresc VaR 1\\% cu circa @{a4.inc}\\%.')),
           size='scriptsize')

D.solved(T('A5: an ARCH-LM test from regression output', 'A5: un test ARCH-LM din rezultatele unei regresii'),
         items(T('An AR(1) is fitted to the daily returns of a stock index. The regression of $\\hat\\varepsilon_t^2$ on a constant and five lags uses $n = 1000$ observations and gives $R^2 = 0.062$; the Ljung--Box statistic of $\\hat\\varepsilon_t^2$ is $Q(10) = 95.3$.',
                 'Pe randamentele zilnice ale unui indice bursier se estimează un AR(1). Regresia lui $\\hat\\varepsilon_t^2$ pe o constantă și cinci decalaje folosește $n = 1000$ de observații și dă $R^2 = 0{,}062$; statistica Ljung--Box a lui $\\hat\\varepsilon_t^2$ este $Q(10) = 95{,}3$.'),
               T('1. Compute the ARCH-LM statistic and compare it with the 5\\% value of $\\chi^2(5)$.', '1. Calculați statistica ARCH-LM și comparați-o cu valoarea de 5\\% a distribuției $\\chi^2(5)$.'),
               T('2. Compare $Q(10)$ with the 5\\% value of $\\chi^2(10)$.', '2. Comparați $Q(10)$ cu valoarea de 5\\% a distribuției $\\chi^2(10)$.'),
               T('3. Say which model you would fit next.', '3. Precizați ce model ați estima în continuare.'),
               T('4. For an ARCH(1) with $\\omega = \\alpha = 0.5$ and Normal $z_t$, compute the unconditional variance and the kurtosis.', '4. Pentru un ARCH(1) cu $\\omega = \\alpha = 0{,}5$ și $z_t$ Normale, calculați varianța necondiționată și coeficientul de boltire.'),
               T('Report: two decisions, two numbers and one sentence.', 'Raportați: două decizii, două valori și o frază.')),
         items(T('1. $\\mathrm{LM} = 1000 \\times 0.062 = @{a5.lm} > @{a5.crit}$ (p @{a5.p}): reject $H_0$, ARCH effects', '1. $\\mathrm{LM} = 1000 \\times 0.062 = @{a5.lm} > @{a5.crit}$ (p @{a5.p}): respingem $H_0$, efecte ARCH'),
               T('2. $95.3 > @{a5.qcrit}$ (p @{a5.qp}): the squares are autocorrelated; same conclusion', '2. $95.3 > @{a5.qcrit}$ (p @{a5.qp}): pătratele sînt autocorelate; aceeași concluzie'),
               T('3. AR(1)-GARCH(1,1), estimated jointly; then the same tests on $\\hat z_t$', '3. AR(1)-GARCH(1,1), estimat simultan; apoi aceleași teste pe $\\hat z_t$'),
               T('4. $\\bar\\sigma^2 = 0.5/(1 - 0.5) = @{a5.uv}$; $K = 3 \\times 0.75/0.25 = @{a5.k}$: Normal innovations, yet heavy-tailed returns', '4. $\\bar\\sigma^2 = 0.5/(1 - 0.5) = @{a5.uv}$; $K = 3 \\times 0.75/0.25 = @{a5.k}$: inovații Normale, dar randamente cu cozi groase')),
         size='scriptsize')

D.proposed(T('A6: good and bad news', 'A6: știri bune și știri proaste'),
           items(T('GJR-GARCH: $\\omega = 0.02$, $\\alpha = 0.03$, $\\gamma = 0.12$, $\\beta = 0.90$; EGARCH: $\\omega = 0$, $\\alpha = 0.12$, $\\gamma = -0.08$, $\\beta = 0.98$; in both $\\sigma_t^2 = 1$, $\\mu = 0$. Model: A1.',
                   'GJR-GARCH: $\\omega = 0{,}02$, $\\alpha = 0{,}03$, $\\gamma = 0{,}12$, $\\beta = 0{,}90$; EGARCH: $\\omega = 0$, $\\alpha = 0{,}12$, $\\gamma = -0{,}08$, $\\beta = 0{,}98$; în ambele $\\sigma_t^2 = 1$, $\\mu = 0$. Model: A1.'),
                 T('1. Compute the GJR $\\sigma_{t+1}^2$ after $\\varepsilon_t = -2$ and after $\\varepsilon_t = +2$, and their ratio.', '1. Calculați $\\sigma_{t+1}^2$ GJR după $\\varepsilon_t = -2$ și după $\\varepsilon_t = +2$, precum și raportul lor.'),
                 T('2. Compute the GJR persistence and half-life.', '2. Calculați persistența și timpul de înjumătățire GJR.'),
                 T('3. Compute the EGARCH $\\sigma_{t+1}^2$ for the same two shocks (use $E|z| = 0.7979$).', '3. Calculați $\\sigma_{t+1}^2$ EGARCH pentru aceleași două șocuri (folosiți $E|z| = 0{,}7979$).'),
                 T('Report: six numbers and one sentence.', 'Raportați: șase valori și o frază.')),
           items(T('1. $0.02 + 0.15 \\times 4 + 0.90 = @{a6.neg}$; $0.02 + 0.03 \\times 4 + 0.90 = @{a6.pos}$; ratio $@{a6.ratio}$', '1. $0.02 + 0.15 \\times 4 + 0.90 = @{a6.neg}$; $0.02 + 0.03 \\times 4 + 0.90 = @{a6.pos}$; raportul $@{a6.ratio}$'),
                 T('2. $0.03 + 0.90 + 0.06 = @{a6.pers}$; $\\ln 0.5/\\ln @{a6.pers} = @{a6.hl}$ days', '2. $0.03 + 0.90 + 0.06 = @{a6.pers}$; $\\ln 0.5/\\ln @{a6.pers} = @{a6.hl}$ zile'),
                 T('3. $z = -2$: $0.12(2 - 0.7979) + 0.16 = @{a6.egneg.ln}$, $e^{@{a6.egneg.ln}} = @{a6.egneg}$; $z = +2$: $@{a6.egpos.ln}$, $\\sigma_{t+1}^2 = @{a6.egpos}$',
                   '3. $z = -2$: $0.12(2 - 0.7979) + 0.16 = @{a6.egneg.ln}$, $e^{@{a6.egneg.ln}} = @{a6.egneg}$; $z = +2$: $@{a6.egpos.ln}$, $\\sigma_{t+1}^2 = @{a6.egpos}$'),
                 T('In EGARCH a good shock of 2 can even lower the variance: the sign term dominates.', 'În EGARCH, un șoc pozitiv de 2 poate chiar să scadă varianța: termenul de semn domină.')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: real data and interpretation', 'Partea B: date reale și interpretare')

D.task(T('B1: ARCH effects and AR(1)-GARCH(1,1)-t for the BET [Solved]', 'B1: efecte ARCH și AR(1)-GARCH(1,1)-t pentru BET [Rezolvat]'),
       T('does the BET have ARCH effects, how persistent is its volatility, and where is it today relative to its long-run level?', 'are BET efecte ARCH, cît de persistentă este volatilitatea lui și unde se află azi față de nivelul de lungă durată?'),
       T('BET closes since 2000, daily log returns in \\%', 'închiderile BET din 2000, randamente logaritmice zilnice în \\%'),
       [T('Fit an AR(1) for the mean by OLS, then run ARCH-LM(5) and Ljung--Box $Q(10)$ on its squared residuals.', 'Estimați un AR(1) pentru medie prin OLS, apoi aplicați ARCH-LM(5) și Ljung--Box $Q(10)$ pe pătratele reziduurilor.'),
        T('Estimate AR(1)-GARCH(1,1)-t and report the parameters with robust standard errors.', 'Estimați AR(1)-GARCH(1,1)-t și raportați parametrii cu erorile standard robuste.'),
        T('Compute the persistence, the half-life and the annualised long-run volatility, and compare them with the sample volatility.', 'Calculați persistența, timpul de înjumătățire și volatilitatea de lungă durată anualizată și comparați-le cu volatilitatea de selecție.'),
        T('Check $Q(10)$ of $\\hat z_t$ and of $\\hat z_t^2$, and draw the returns since 2018 with $\\pm 2\\hat\\sigma_t$ bands.', 'Verificați $Q(10)$ pentru $\\hat z_t$ și $\\hat z_t^2$ și desenați randamentele din 2018 cu benzile $\\pm 2\\hat\\sigma_t$.'),
        T('Interpretation: is the BET volatility on the last day above or below its long-run level, and what does the model expect for the next months?', 'Interpretare: este volatilitatea BET din ultima zi peste sau sub nivelul de lungă durată și ce așteaptă modelul pentru lunile următoare?')],
       T('two test statistics, a table of six parameters, five numbers, the chart and two sentences', 'două statistici de test, un tabel cu șase parametri, cinci valori, graficul și două fraze'), size='footnotesize', nb='B1')

D.frame(T('B1: solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch5_sem_b1', h='0.28') + table(
    'lrrrrrr', '& $c$ & $\\phi$ & $\\omega$ & $\\alpha$ & $\\beta$ & $\\nu$',
    [T('estimate', 'estimare') + ' & $@{b1.c}$ & $@{b1.phi}$ & $@{b1.om}$ & $@{b1.a}$ & $@{b1.b}$ & $@{b1.nu}$',
     T('robust SE', 'SE robustă') + ' & $@{b1.c.se}$ & $@{b1.phi.se}$ & $@{b1.om.se}$ & $@{b1.a.se}$ & $@{b1.b.se}$ & $@{b1.nu.se}$'], size='scriptsize') + items(
    T('AR(1) residuals: ARCH-LM $= nR^2 = @{b1.lm}$ ($R^2 = @{b1.lmr2}$), $Q(10)$ of squares $= @{b1.q2e}$: strong ARCH effects', 'Reziduurile AR(1): ARCH-LM $= nR^2 = @{b1.lm}$ ($R^2 = @{b1.lmr2}$), $Q(10)$ al pătratelor $= @{b1.q2e}$: efecte ARCH puternice'),
    T('$T = @{b1.n}$; $\\alpha + \\beta = @{b1.pers}$, half-life $\\ln 0.5/(@{b1.lnp}) = @{b1.hl}$ days; long-run volatility @{b1.vlr}\\% against @{b1.vs}\\% in the sample; $\\hat z_t$: $Q(10) = @{b1.qz}$ (p @{b1.qzp}), $\\hat z_t^2$: $@{b1.qz2}$ (p = @{b1.qz2p})',
      '$T = @{b1.n}$; $\\alpha + \\beta = @{b1.pers}$, timpul de înjumătățire $\\ln 0{,}5/(@{b1.lnp}) = @{b1.hl}$ zile; volatilitatea de lungă durată @{b1.vlr}\\%, față de @{b1.vs}\\% în eșantion; $\\hat z_t$: $Q(10) = @{b1.qz}$ (p @{b1.qzp}), $\\hat z_t^2$: $@{b1.qz2}$ (p = @{b1.qz2p})'),
    T('Peaks: @{b1.p2008}\\% (2008), @{b1.p2020}\\% (2020); last day: @{b1.last}\\%; @{b1.out}\\% of the days fall outside $\\pm 2\\hat\\sigma_t$',
      'Vîrfuri: @{b1.p2008}\\% (2008), @{b1.p2020}\\% (2020); ultima zi: @{b1.last}\\%; @{b1.out}\\% dintre zile ies din banda $\\pm 2\\hat\\sigma_t$'),
    T('Interpretation: below (@{b1.last}\\% against @{b1.vlr}\\%); the model expects the volatility to rise slowly towards the long-run level, but that level is imprecise ($1 - \\alpha - \\beta = @{b1.1mp}$)',
      'Interpretare: sub nivelul de lungă durată (@{b1.last}\\% față de @{b1.vlr}\\%); modelul se așteaptă ca volatilitatea să crească lent spre acest nivel, dar nivelul este imprecis ($1 - \\alpha - \\beta = @{b1.1mp}$)')) + qlsem(), 'scriptsize')

D.task(T('B2: EUR/RON and Bitcoin [Proposed]', 'B2: EUR/RON și Bitcoin [Propus]'),
       T('do the EUR/RON rate and Bitcoin behave like the BET? Model: B1.', 'se comportă cursul EUR/RON și Bitcoin la fel ca BET? Model: B1.'),
       T('EUR/RON (BNR reference rate) since July 2005, Bitcoin since September 2014; daily log returns in \\%', 'EUR/RON (cursul de referință BNR) din iulie 2005, Bitcoin din septembrie 2014; randamente logaritmice zilnice în \\%'),
       [T('Repeat steps 1--4 of B1 for both series (EUR/RON: estimate on $10\\,r_t$).', 'Repetați pașii 1--4 din B1 pentru ambele serii (EUR/RON: estimați pe $10\\,r_t$).'),
        T('Report $\\hat\\alpha + \\hat\\beta$ and say whether the half-life and the long-run volatility exist.', 'Raportați $\\hat\\alpha + \\hat\\beta$ și precizați dacă timpul de înjumătățire și volatilitatea de lungă durată există.'),
        T('Interpretation: what does $\\hat\\alpha + \\hat\\beta = 1$ mean for a forecast of the volatility one year ahead?', 'Interpretare: ce înseamnă $\\hat\\alpha + \\hat\\beta = 1$ pentru o prognoză a volatilității peste un an?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B2')


def b2row(k):
    return (f'{NAMES[k]} & @{{b2.{k}.n}} & @{{b2.{k}.lm}} & $@{{b2.{k}.phi}}$ & $@{{b2.{k}.a}}$ & $@{{b2.{k}.b}}$ & $@{{b2.{k}.nu}}$ & '
            f'$@{{b2.{k}.ab}}$ & @{{b2.{k}.vs}} & $@{{b2.{k}.qz2}}$')


D.frame(T('B2: solution [Proposed]', 'B2: rezolvare [Propus]'), table(
    'lrrrrrrrrr', T('& $T$ & ARCH-LM & $\\phi$ & $\\alpha$ & $\\beta$ & $\\nu$ & $\\alpha + \\beta$ & sample vol. & $Q(10)$, $\\hat z_t^2$',
                    '& $T$ & ARCH-LM & $\\phi$ & $\\alpha$ & $\\beta$ & $\\nu$ & $\\alpha + \\beta$ & vol. selecție & $Q(10)$, $\\hat z_t^2$'),
    [b2row(k) for k in ['eurron', 'btc']], size='scriptsize') + items(
    T('Both: strong ARCH effects before the model; $\\hat\\alpha + \\hat\\beta = 1$ (IGARCH): no half-life and no long-run volatility; heavy tails ($\\hat\\nu$ @{b2.eurron.nu} and @{b2.btc.nu})',
      'Ambele: efecte ARCH puternice înainte de model; $\\hat\\alpha + \\hat\\beta = 1$ (IGARCH): fără timp de înjumătățire și fără volatilitate de lungă durată; cozi groase ($\\hat\\nu$ @{b2.eurron.nu} și @{b2.btc.nu})'),
    T('EUR/RON: $\\hat\\phi = @{b2.eurron.phi}$ (OLS: @{b2.eurron.phio}): the AR term almost disappears once the variance is modelled; $Q(10)$ of $\\hat z_t^2$ tiny because of one day (@{b6.eurron.day}, Lecture 5)',
      'EUR/RON: $\\hat\\phi = @{b2.eurron.phi}$ (OLS: @{b2.eurron.phio}): termenul AR aproape dispare cînd modelăm varianța; $Q(10)$ al lui $\\hat z_t^2$ este foarte mic din cauza unei singure zile (@{b6.eurron.day}, Cursul 5)'),
    T('Interpretation: no mean reversion: the forecast for every horizon equals tomorrow\'s variance, as for EWMA; a one-year forecast simply repeats today\'s level',
      'Interpretare: nu există revenire la medie: prognoza pentru orice orizont este egală cu varianța de mîine, ca la EWMA; o prognoză pe un an repetă pur și simplu nivelul de azi')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B3: the leverage effect in the S\\&P 500 [Solved]', 'B3: efectul de levier la S\\&P 500 [Rezolvat]'),
       T('do falls raise the volatility of the S\\&P 500 more than rises of the same size?', 'cresc scăderile volatilitatea S\\&P 500 mai mult decît creșterile de aceeași mărime?'),
       T('S\\&P 500 closes since 2000, daily log returns in \\%', 'închiderile S\\&P 500 din 2000, randamente logaritmice zilnice în \\%'),
       [T('Estimate GARCH(1,1)-t and GJR-GARCH(1,1)-t; report $\\gamma$ with its robust $t$ statistic.', 'Estimați GARCH(1,1)-t și GJR-GARCH(1,1)-t; raportați $\\gamma$ cu statistica $t$ robustă.'),
        T('Compute the LR statistic of GJR against GARCH, its p-value and the two BIC values.', 'Calculați statistica LR pentru GJR față de GARCH, p-valoarea ei și cele două valori BIC.'),
        T('Run the sign-bias test on the standardised residuals of GARCH-t.', 'Aplicați testul de asimetrie (sign bias) pe reziduurile standardizate ale GARCH-t.'),
        T('Draw both news impact curves and compute the GJR variance after shocks of $-2\\%$ and $+2\\%$.', 'Desenați ambele curbe de impact ale știrilor și calculați varianța GJR după șocuri de $-2\\%$ și $+2\\%$.'),
        T('Interpretation: how does the leverage effect change the VaR on the day after a fall?', 'Interpretare: cum schimbă efectul de levier VaR-ul din ziua de după o scădere?')],
       T('four statistics with p-values, two variances, the chart and two sentences', 'patru statistici cu p-valori, două varianțe, graficul și două fraze'), size='footnotesize', nb='B3')

D.frame(T('B3: solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch5_sem_b3', h='0.34') + items(
    T('GJR: $\\hat\\alpha = @{b3.a}$, $\\hat\\gamma = @{b3.g}$ ($t = @{b3.gt}$), $\\hat\\beta = @{b3.b}$; LR $= @{b3.lr}$ (p @{b3.lrp}); BIC $@{b3.bg} \\to @{b3.bj}$',
      'GJR: $\\hat\\alpha = @{b3.a}$, $\\hat\\gamma = @{b3.g}$ ($t = @{b3.gt}$), $\\hat\\beta = @{b3.b}$; LR $= @{b3.lr}$ (p @{b3.lrp}); BIC $@{b3.bg} \\to @{b3.bj}$'),
    T('Sign bias on the GARCH-t residuals: $\\chi^2(3) = @{b3.sb}$ (p @{b3.sbp}), sign $t = @{b3.sbt}$', 'Testul de asimetrie pe reziduurile GARCH-t: $\\chi^2(3) = @{b3.sb}$ (p @{b3.sbp}), $t$ al semnului $= @{b3.sbt}$'),
    T('GJR variance after $-2\\%$: $@{b3.neg}$; after $+2\\%$: $@{b3.pos}$; ratio $@{b3.ratio}$', 'Varianța GJR după $-2\\%$: $@{b3.neg}$; după $+2\\%$: $@{b3.pos}$; raportul $@{b3.ratio}$'),
    T('Interpretation: only falls raise the variance ($\\hat\\alpha = 0$); after a fall the VaR is about $\\sqrt{@{b3.ratio}} = @{b3.sq}$ times the VaR after an equal rise; a symmetric GARCH understates the risk after bad days',
      'Interpretare: doar scăderile cresc varianța ($\\hat\\alpha = 0$); după o scădere, VaR este de circa $\\sqrt{@{b3.ratio}} = @{b3.sq}$ ori VaR-ul de după o creștere egală; un GARCH simetric subestimează riscul după zilele proaste')) + qlsem(), 'scriptsize')

D.task(T('B4: asymmetry in the BET, EUR/RON and Bitcoin [Proposed]', 'B4: asimetria la BET, EUR/RON și Bitcoin [Propus]'),
       T('is there a leverage effect in the BET, in EUR/RON and in Bitcoin? Model: B3.', 'există un efect de levier la BET, la EUR/RON și la Bitcoin? Model: B3.'),
       T('BET since 2000, EUR/RON since 2005, Bitcoin since 2014; daily log returns in \\%', 'BET din 2000, EUR/RON din 2005, Bitcoin din 2014; randamente logaritmice zilnice în \\%'),
       [T('Estimate GJR-GARCH(1,1)-t and EGARCH(1,1)-t; report both values of $\\gamma$ with robust $t$ statistics.', 'Estimați GJR-GARCH(1,1)-t și EGARCH(1,1)-t; raportați ambele valori $\\gamma$, cu statisticile $t$ robuste.'),
        T('Compute the LR test of GJR against GARCH and the change in BIC.', 'Calculați testul LR pentru GJR față de GARCH și modificarea BIC.'),
        T('Interpretation: for EUR/RON a positive return means a weaker leu; what is ``bad news\'\' for this series?', 'Interpretare: pentru EUR/RON un randament pozitiv înseamnă un leu mai slab; ce sînt „știrile proaste” pentru această serie?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B4')


def b4row(k):
    return (f'{NAMES[k]} & $@{{b4.{k}.a}}$ & $@{{b4.{k}.g}}$ & ${{@{{b4.{k}.gt}}}}$ & $@{{b4.{k}.lr}}$ & @{{b4.{k}.lrp}} & ${{@{{b4.{k}.db}}}}$ & ${{@{{b4.{k}.eg}}}}$ & ${{@{{b4.{k}.egt}}}}$')


D.frame(T('B4: solution [Proposed]', 'B4: rezolvare [Propus]'), table(
    'lrrrrrrrr', T('& GJR $\\alpha$ & GJR $\\gamma$ & $t$ & LR & p & $\\Delta$BIC & EGARCH $\\gamma$ & $t$', '& GJR $\\alpha$ & GJR $\\gamma$ & $t$ & LR & p & $\\Delta$BIC & EGARCH $\\gamma$ & $t$'),
    [b4row(k) for k in ['bet', 'eurron', 'btc']], size='footnotesize') + items(
    T('BET: $\\gamma > 0$ (EGARCH $\\gamma < 0$), significant by LR at 5\\%, but BIC rises: a real but small effect', 'BET: $\\gamma > 0$ (în EGARCH $\\gamma < 0$), semnificativ după testul LR la 5\\%, dar BIC crește: un efect real, dar mic'),
    T('Bitcoin: no asymmetry ($t = @{b4.btc.gt}$); EUR/RON: $\\gamma > 0$ and significant, with BIC almost unchanged', 'Bitcoin: nicio asimetrie ($t = @{b4.btc.gt}$); EUR/RON: $\\gamma > 0$ și semnificativ, cu BIC aproape neschimbat'),
    T('Interpretation: in GJR the ``bad news\'\' are $\\varepsilon_{t-1} < 0$, days when the leu strengthens; for a currency the sign has no leverage meaning: a large move in either direction signals policy news',
      'Interpretare: în GJR „știrile proaste” sînt $\\varepsilon_{t-1} < 0$, zilele în care leul se întărește; pentru o valută semnul nu are sensul efectului de levier: o mișcare mare în orice direcție semnalează știri de politică monetară')) + qlsem(),
    'footnotesize', instructor_only=True)

D.task(T('B5: GARCH against EWMA for the S\\&P 500 [Solved]', 'B5: GARCH față de EWMA pentru S\\&P 500 [Rezolvat]'),
       T('does GARCH(1,1)-t forecast tomorrow\'s variance of the S\\&P 500 better than EWMA, and is its VaR 1\\% exceeded on 1\\% of the days?',
         'prognozează GARCH(1,1)-t varianța de mîine a S\\&P 500 mai bine decît EWMA și este VaR 1\\% al lui depășit în 1\\% dintre zile?'),
       T('S\\&P 500 since 2000; forecasts for every day since 2 January 2015', 'S\\&P 500 din 2000; prognoze pentru fiecare zi de la 2 ianuarie 2015'),
       [T('Re-estimate GARCH(1,1)-t every 250 days on all data up to that day and compute the one-day forecasts $h_t$; compute EWMA with $\\lambda = 0.94$.',
          'Reestimați GARCH(1,1)-t la fiecare 250 de zile pe toate datele pînă în acea zi și calculați prognozele pe o zi $h_t$; calculați EWMA cu $\\lambda = 0{,}94$.'),
        T('Compute the mean QLIKE of both models and the Diebold--Mariano $t$ statistic.', 'Calculați media QLIKE pentru ambele modele și statistica $t$ Diebold--Mariano.'),
        T('Count the exceedances of the GARCH-t VaR 1\\% and of the EWMA-Normal VaR 1\\%, and compare them with the expected number.', 'Numărați depășirile VaR 1\\% GARCH-t și VaR 1\\% EWMA-Normal și comparați-le cu numărul așteptat.'),
        T('Draw the cumulative difference of the losses.', 'Desenați diferența cumulată a pierderilor.'),
        T('Interpretation: which model would you give to a risk manager, and what would you still improve?', 'Interpretare: ce model i-ați da unui manager de risc și ce ați mai îmbunătăți?')],
       T('four numbers, two counts, the chart and two sentences', 'patru valori, două numărători, graficul și două fraze'), size='footnotesize', nb='B5')

D.frame(T('B5: solution [Solved]', 'B5: rezolvare [Rezolvat]'), fig('ch5_sem_b5', h='0.32') + items(
    T('@{b5.n} days; mean QLIKE: GARCH-t $@{b5.qg}$, EWMA $@{b5.qe}$; DM: mean difference $@{b5.dmm}$, $t = @{b5.dm}$ (p @{b5.dmp}): GARCH-t is better',
      '@{b5.n} zile; media QLIKE: GARCH-t $@{b5.qg}$, EWMA $@{b5.qe}$; DM: diferența medie $@{b5.dmm}$, $t = @{b5.dm}$ (p @{b5.dmp}): GARCH-t este mai bun'),
    T('VaR 1\\% exceedances: GARCH-t @{b5.ng} (@{b5.eg}\\%), EWMA-Normal @{b5.ne} (@{b5.ee}\\%); expected about @{b5.exp}',
      'Depășiri VaR 1\\%: GARCH-t @{b5.ng} (@{b5.eg}\\%), EWMA-Normal @{b5.ne} (@{b5.ee}\\%); așteptat circa @{b5.exp}'),
    T('Interpretation: GARCH-t, but both VaR models are exceeded too often; add the leverage effect (B3) and test the number of exceedances formally (\\refKupiec)',
      'Interpretare: GARCH-t, dar ambele modele VaR sînt depășite prea des; adăugăm efectul de levier (B3) și testăm formal numărul de depășiri (\\refKupiec)')) + qlsem(), 'scriptsize')

D.task(T('B6: GARCH against EWMA for the BET and EUR/RON [Proposed]', 'B6: GARCH față de EWMA pentru BET și EUR/RON [Propus]'),
       T('does the result of B5 hold for the BET and for EUR/RON? Model: B5.', 'se păstrează rezultatul din B5 pentru BET și pentru EUR/RON? Model: B5.'),
       T('BET since 2000, EUR/RON since 2005; forecasts for every day since January 2015', 'BET din 2000, EUR/RON din 2005; prognoze pentru fiecare zi din ianuarie 2015'),
       [T('Compute the out-of-sample GARCH(1,1)-t and EWMA forecasts as in B5.', 'Calculați prognozele GARCH(1,1)-t și EWMA în afara eșantionului, ca în B5.'),
        T('Compute the mean QLIKE of both models and the DM test.', 'Calculați media QLIKE pentru ambele modele și testul DM.'),
        T('Find the day with the largest loss difference, and recompute the mean QLIKE without it.', 'Găsiți ziua cu cea mai mare diferență a pierderilor și recalculați media QLIKE fără ea.'),
        T('Count the VaR 1\\% exceedances of both models.', 'Numărați depășirile VaR 1\\% pentru ambele modele.'),
        T('Interpretation: why can the DM test be insignificant when one mean QLIKE is half the other?', 'Interpretare: de ce poate fi testul DM nesemnificativ cînd o medie QLIKE este jumătate din cealaltă?')],
       T('one table and two sentences', 'un tabel și două fraze'), size='footnotesize', nb='B6')


def b6row(k):
    return (f'{NAMES[k]} & $@{{b6.{k}.qg}}$ & $@{{b6.{k}.qe}}$ & ${{@{{b6.{k}.dm}}}}$ (@{{b6.{k}.dmp}}) & @{{b6.{k}.day}} & @{{b6.{k}.share}}\\% & '
            f'@{{b6.{k}.ng}} & @{{b6.{k}.ne}}')


D.frame(T('B6: solution [Proposed]', 'B6: rezolvare [Propus]'), table(
    'lrrrrrrr', T('& QLIKE, GARCH-t & EWMA & DM $t$ (p) & largest day & share & exc., GARCH-t & EWMA',
                  '& QLIKE, GARCH-t & EWMA & DM $t$ (p) & ziua maximă & pondere & dep., GARCH-t & EWMA'),
    [b6row(k) for k in ['bet', 'eurron']], size='scriptsize') + items(
    T('Expected VaR 1\\% exceedances: about @{b6.bet.exp} for each series; BET: GARCH-t better (DM $t = @{b6.bet.dm}$), exceedances: @{b6.bet.ng}, EWMA-Normal about twice too many',
      'Depășiri VaR 1\\% așteptate: circa @{b6.bet.exp} pentru fiecare serie; BET: GARCH-t mai bun (DM $t = @{b6.bet.dm}$), depășiri: @{b6.bet.ng}, EWMA-Normal de circa două ori prea multe'),
    T('EUR/RON: mean QLIKE @{b6.eurron.qg} against @{b6.eurron.qe}, but DM $t = @{b6.eurron.dm}$: @{b6.eurron.share}\\% of the difference comes from @{b6.eurron.day}; without it: $@{b6.eurron.qgw}$ against $@{b6.eurron.qew}$',
      'EUR/RON: media QLIKE @{b6.eurron.qg} față de @{b6.eurron.qe}, dar DM $t = @{b6.eurron.dm}$: @{b6.eurron.share}\\% din diferență provine din @{b6.eurron.day}; fără această zi: $@{b6.eurron.qgw}$ față de $@{b6.eurron.qew}$'),
    T('Interpretation: the DM statistic divides the mean difference by its standard error; one enormous loss inflates both, so the evidence rests on one day and is weak',
      'Interpretare: statistica DM împarte diferența medie la eroarea ei standard; o singură pierdere enormă le mărește pe amîndouă, deci dovada se sprijină pe o zi și este slabă')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: open questions and AI critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.task(T('C1: is the leu calmer than it used to be? [Proposed]', 'C1: este leul mai liniștit decît înainte? [Propus]'),
       T('has the volatility of EUR/RON changed between 2005--2012, 2013--2019 and 2020--2026, and can one GARCH model describe all three periods?',
         's-a schimbat volatilitatea EUR/RON între perioadele 2005--2012, 2013--2019 și 2020--2026 și poate un singur model GARCH să descrie toate trei perioadele?'),
       T('EUR/RON daily log returns in \\% (BNR reference rate); model: B1', 'randamentele logaritmice zilnice EUR/RON în \\% (cursul de referință BNR); model: B1'),
       [T('For each period compute the annualised sample volatility and the share of days with $|r_t| > 0.5\\%$.', 'Pentru fiecare perioadă calculați volatilitatea de selecție anualizată și ponderea zilelor cu $|r_t| > 0{,}5\\%$.'),
        T('Estimate GARCH(1,1)-t on each period and report $\\alpha + \\beta$, $\\alpha$ and $\\nu$.', 'Estimați GARCH(1,1)-t pe fiecare perioadă și raportați $\\alpha + \\beta$, $\\alpha$ și $\\nu$.'),
        T('Find the largest daily move of each period and its date.', 'Găsiți cea mai mare variație zilnică din fiecare perioadă și data ei.'),
        T('Interpretation: is the change in volatility a property of the market, or of the exchange-rate policy?', 'Interpretare: este schimbarea volatilității o proprietate a pieței sau a politicii de curs de schimb?')],
       T('a table and a plan for a project', 'un tabel și un plan de proiect'), size='footnotesize', nb='C1')

D.frame(T('C1: reference analysis [Proposed]', 'C1: analiză de referință [Propus]'), table(
    'lrrrrrrr', T('Period & $T$ & vol.\\ (\\% p.a.) & $|r| > 0.5\\%$ (\\%) & $\\alpha + \\beta$ & $\\alpha$ & $\\nu$ & max $|r|$',
                  'Perioada & $T$ & vol.\\ (\\% pe an) & $|r| > 0{,}5\\%$ (\\%) & $\\alpha + \\beta$ & $\\alpha$ & $\\nu$ & max $|r|$'),
    c1rows, size='scriptsize') + items(
    T('Volatility falls from @{c1.v0}\\% to @{c1.v1}\\% and @{c1.v2}\\% per year; large days from @{c1.b0}\\% to @{c1.b2}\\% of the days',
      'Volatilitatea scade de la @{c1.v0}\\% la @{c1.v1}\\% și @{c1.v2}\\% pe an; zilele cu variații mari, de la @{c1.b0}\\% la @{c1.b2}\\% dintre zile'),
    T('Every period is IGARCH ($\\alpha + \\beta = 1$) with very heavy tails: long calm stretches broken by rare jumps (largest moves: @{c1.d0}, @{c1.d1}, @{c1.d2})',
      'Fiecare perioadă este IGARCH ($\\alpha + \\beta = 1$), cu cozi foarte groase: perioade lungi de calm întrerupte de salturi rare (cele mai mari variații: @{c1.d0}, @{c1.d1}, @{c1.d2})'),
    T('Interpretation: the policy of the BNR, not the market, sets the level; a GARCH with constant parameters fits none of the periods well; project: Markov switching (Chapter 10) or a GARCH with policy dummies',
      'Interpretare: nivelul este fixat de politica BNR, nu de piață; un GARCH cu parametri constanți nu se potrivește bine niciunei perioade; proiect: Markov switching (Capitolul 10) sau un GARCH cu variabile de politică')) + qlsem(),
    'footnotesize', instructor_only=True)

D.frame(T('C2: audit an AI answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant to interpret a GARCH(1,1)-t fit of the BET (daily, since 2000): $\\hat\\omega = @{c2.om}$, $\\hat\\alpha = @{c2.a}$, $\\hat\\beta = @{c2.b}$, $\\hat\\nu = @{c2.nu}$. The answer:',
      'Un student a cerut unui asistent AI să interpreteze un GARCH(1,1)-t estimat pentru BET (date zilnice, din 2000): $\\hat\\omega = @{c2.om}$, $\\hat\\alpha = @{c2.a}$, $\\hat\\beta = @{c2.b}$, $\\hat\\nu = @{c2.nu}$. Răspunsul:'),
    T('\\aiprompt{(a) The half-life of a volatility shock is ln 0.5 / ln beta = @{c2.hlw} days.}', '\\aiprompt{(a) Timpul de înjumătățire al unui șoc de volatilitate este ln 0,5 / ln beta = @{c2.hlw} zile.}'),
    T('\\aiprompt{(b) The long-run variance is omega/(1 - alpha) = @{c2.uvw}, an annual volatility of @{c2.volw}\\%.}', '\\aiprompt{(b) Varianța de lungă durată este omega/(1 - alpha) = @{c2.uvw}, o volatilitate anuală de @{c2.volw}\\%.}'),
    T('\\aiprompt{(c) An AR(1) by OLS gives phi = @{c2.phio} with SE @{c2.seo}, t = @{c2.to}: BET returns are strongly predictable.}', '\\aiprompt{(c) Un AR(1) estimat prin OLS dă phi = @{c2.phio} cu SE @{c2.seo}, t = @{c2.to}: randamentele BET sînt puternic previzibile.}'),
    T('\\aiprompt{(d) Ljung-Box Q(10) of the squared standardised residuals = @{c2.lb}, p = @{c2.lbp}: the GARCH-t model is correct.}', '\\aiprompt{(d) Ljung-Box Q(10) al pătratelor reziduurilor standardizate = @{c2.lb}, p = @{c2.lbp}: modelul GARCH-t este corect.}'),
    T('\\aiprompt{(e) With Student-t innovations the 1\\% quantile of z is closer to 0 than the Normal -2.326, so the VaR 1\\% is smaller.}', '\\aiprompt{(e) Cu inovații Student-t, cuantila de 1\\% a lui z este mai aproape de 0 decît valoarea Normală -2,326, deci VaR 1\\% este mai mic.}'),
    T('\\aiprompt{(f) alpha + beta = @{c2.pers} < 1, so the model is covariance stationary and the volatility reverts to its long-run level.}', '\\aiprompt{(f) alpha + beta = @{c2.pers} < 1, deci modelul este staționar în covarianță, iar volatilitatea revine la nivelul de lungă durată.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the notebook (section C2).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook (secțiunea C2).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: the half-life uses the persistence, $\\ln 0.5/\\ln(\\alpha + \\beta) = \\ln 0.5/\\ln @{c2.pers} = @{c2.hl}$ days', '(a) Greșit: timpul de înjumătățire folosește persistența, $\\ln 0{,}5/\\ln(\\alpha + \\beta) = \\ln 0{,}5/\\ln @{c2.pers} = @{c2.hl}$ zile'),
    T('(b) Wrong: $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta) = @{c2.uv}$, a long-run volatility of @{c2.vol}\\% (imprecise, since $1 - \\alpha - \\beta$ is small)',
      '(b) Greșit: $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta) = @{c2.uv}$, o volatilitate de lungă durată de @{c2.vol}\\% (imprecisă, deoarece $1 - \\alpha - \\beta$ este mic)'),
    T('(c) Wrong: with GARCH errors the OLS SE is too small; robust SE @{c2.sehc} ($t = @{c2.thc}$); AR(1)-GARCH-t: $\\hat\\phi = @{c2.phig}$ ($t = @{c2.tg}$); significant, but $\\phi^2 \\approx @{c2.r2}$ of the variance: weak predictability',
      '(c) Greșit: cu erori GARCH, SE din OLS este prea mică; SE robustă @{c2.sehc} ($t = @{c2.thc}$); AR(1)-GARCH-t: $\\hat\\phi = @{c2.phig}$ ($t = @{c2.tg}$); semnificativ, dar $\\phi^2 \\approx @{c2.r2}$ din varianță: o previzibilitate slabă'),
    T('(d) Wrong: no ARCH effect is left, but that does not prove the model correct; $Q(10)$ of $\\hat z_t$ is @{c2.lbz}: autocorrelation is left in the mean (Lecture 5: AR(1)-GARCH)',
      '(d) Greșit: nu a rămas niciun efect ARCH, dar asta nu dovedește că modelul este corect; $Q(10)$ al lui $\\hat z_t$ este @{c2.lbz}: a rămas autocorelație în medie (Cursul 5: AR(1)-GARCH)'),
    T('(e) Wrong: for the standardised $t$ with $\\nu = @{c2.nu}$ the quantile is $@{c2.qt}$, further from 0 than $@{c2.qn}$: the VaR 1\\% is larger', '(e) Greșit: pentru $t$ standardizată cu $\\nu = @{c2.nu}$ cuantila este $@{c2.qt}$, mai departe de 0 decît $@{c2.qn}$: VaR 1\\% este mai mare'),
    T('(f) Correct', '(f) Corect')) + qlsem(),
    'footnotesize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-up', 'Încheiere')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Test for ARCH effects on the residuals of the mean model: ARCH-LM $= nR^2$ and Ljung--Box on the squares', 'Testăm efectele ARCH pe reziduurile modelului pentru medie: ARCH-LM $= nR^2$ și Ljung--Box pe pătrate'),
    T('$\\alpha + \\beta$ decides the long run: half-life, long-run variance, the shape of the forecasts; $\\alpha + \\beta = 1$: IGARCH', '$\\alpha + \\beta$ decide termenul lung: timpul de înjumătățire, varianța de lungă durată, forma prognozelor; $\\alpha + \\beta = 1$: IGARCH'),
    T('Multi-day risk is the sum of the daily variance forecasts, not $\\sqrt{H}$ times today\'s volatility', 'Riscul pe mai multe zile este suma prognozelor zilnice ale varianței, nu $\\sqrt{H}$ ori volatilitatea de azi'),
    T('Heavy-tailed innovations change the VaR quantile; asymmetry changes the VaR after bad days', 'Inovațiile cu cozi groase schimbă cuantila VaR; asimetria schimbă VaR-ul după zilele proaste'),
    T('Compare forecasts out of sample (QLIKE, DM) and look for single days that drive the result', 'Comparăm prognozele în afara eșantionului (QLIKE, DM) și căutăm zilele care determină singure rezultatul')))

D.frame(T('After the seminar', 'După seminar'), items(
    T('Lecture 5 develops each topic of today: stylised facts, ARCH and GARCH, estimation, ARMA-GARCH, asymmetry, diagnostics, forecasts, VaR 1\\%',
      'Cursul 5 dezvoltă fiecare temă de azi: faptele stilizate, ARCH și GARCH, estimarea, ARMA-GARCH, asimetria, diagnosticarea, prognozele, VaR 1\\%'),
    T('Try the [Proposed] tasks in the notebook', 'Încercați cerințele [Propus] în notebook'),
    T('C1 can grow into a team project: volatility regimes of the leu and of other currencies of the region', 'C1 poate deveni un proiect de echipă: regimurile de volatilitate ale leului și ale altor valute din regiune'),
    T('Reading: \\refHP, Ch.~6; \\refTsay, Ch.~3', 'Lectură: \\refHP, cap.~6; \\refTsay, cap.~3'),
    T('\\textbf{The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class}',
      '\\textbf{Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar}')))

D.references(bib(['Boll', 'BollT', 'BW', 'Christie', 'DM', 'Engle', 'EN', 'GJR', 'HP', 'Kupiec', 'LB', 'Nelson', 'Patton', 'RM', 'Tsay']), per=16)

if __name__ == '__main__':
    finalize(D.write(V))
