r"""
build_chapter15.py -- Capitolul 15 (Recapitulare și pregătire pentru examen), EN + RO dintr-o singură sursă
==========================================================================================================
Harta cursului, cîte un slide de recapitulare pentru fiecare capitol 0--10 (formule, fapte empirice pe serii
românești și internaționale, greșeli frecvente), cîte un slide scurt pentru capitolele de studiu individual 11--14,
trusa de instrumente, metoda Box--Jenkins de la un capăt la altul pe IAPC-ul României, examenul (format, conținut,
criterii, opt probleme de tip examen rezolvate pas cu pas), proiectul de echipă, prezența și secțiunea finală
„Contribuția posibilă a AI”.
Cifrele @{cheie}: faptele empirice vin din fișierele de cifre ale fiecărui capitol (Quantlets/Ch_NN/chN_numbers.json),
cazul Box--Jenkins și Problema 3 din Quantlets/Ch_15/ch15_numbers.json, iar celelalte rezolvări sînt calculate aici.
Ieșire:
  EN/Courses/chapter15_review.tex
  RO/Cursuri/capitol15_recapitulare.tex
Rulare:
  python3 Quantlets/Ch_15/generate_all_charts.py
  python3 latex/build_chapter15.py && python3 latex/tsa_build.py compile 15
"""

import math
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, photo, enum   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch15_common import (QLURL, REFS, T, bib, chapter_numbers, date, facts, finalize, load, month, neg, pv,   # noqa: E402
                         quarter)


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


N = load()
C = chapter_numbers()
V = facts(Values())
P = V.put
D = Deck(15, 'lecture', refs=REFS)
CM = 'https://commons.wikimedia.org/wiki/File:'
SITE = 'https://danpele.github.io/Time-Series-Analysis/'


def ql(folder):
    return f'\\quantlet{{{folder.replace("_", chr(92) + "_")}}}{{\\qlurl{{{folder}}}}}'


def chart(title, fig, folder, bullets, h='0.6\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def interp(title, bullets, size='small'):
    D.frame(T(f'Interpreting {title[0]}', f'Interpretarea {title[1]}'), items(*bullets), size)


PH = {
    'ase': ('ch0_ase_2014.jpg', CM + 'Bucharest_-_Academie_de_Studii_Economice_01.jpg',
            T('Photo', 'Foto') + ': Joe Mabel (2014); CC BY 3.0; Wikimedia Commons'),
    'box': ('ch0_box.jpg', CM + 'GeorgeEPBox_(cropped).jpg',
            T('Photo', 'Foto') + ': DavidMCEddy (2011); CC BY-SA 3.0; Wikimedia Commons'),
    'ins': ('ch2_ins_bucharest_2009.jpg', CM + 'Institutul_Na\\%C8\\%9Bional_de_Statistic\\%C4\\%83.jpg',
            T('Photo', 'Foto') + ': Dan Mihai Pitea (2009); CC BY-SA 3.0; Wikimedia Commons'),
    'granger': ('ch3_clive_granger_2008.jpg', CM + 'Clive_Granger_by_Olaf_Storbeck_(3x4_cropped).jpg',
                T('Photo', 'Foto') + ': Olaf Storbeck (2008); CC BY-SA 2.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


def two(left, right, wl='0.58', wr='0.38'):
    return (f'\\begin{{columns}}[T]\n\\begin{{column}}{{{wl}\\textwidth}}\n{left}\n\\end{{column}}\n'
            f'\\begin{{column}}{{{wr}\\textwidth}}\n{right}\n\\end{{column}}\n\\end{{columns}}')


def chap(n, ro_case='Capitolul'):
    """'Chapter n' / 'Capitolul n' (appendix_links.py turns the mentions into links)."""
    return T(f'Chapter {n}', f'{ro_case} {n}')


FORM = T('\\textbf{Key formulas}', '\\textbf{Formule-cheie}')
FACT = T('\\textbf{Empirical facts}', '\\textbf{Fapte empirice}')
MIST = T('\\textbf{Common mistakes}', '\\textbf{Greșeli frecvente}')


def review(k, title, formulas, fct, mistakes, size='footnotesize'):
    """One recap slide per chapter: key formulas, empirical facts, common mistakes."""
    D.frame(T(f'Chapter {k}: {title[0]}', f'Capitolul {k}: {title[1]}'),
            items((FORM, formulas), (FACT, fct), (MIST, mistakes)), size)


# =============================================================================
# CIFRE: BOX-JENKINS PE IAPC (Quantlets/Ch_15)
# =============================================================================
BD_ = N['data']
P('bj.n', BD_['n'], 0)
V.raw('bj.first', month(BD_['first']))
V.raw('bj.last', month(BD_['last']))
P('bj.lasta', BD_['last_a'], 2)
P('bj.maxa', BD_['max_a'], 1)
V.raw('bj.maxad', month(BD_['max_a_d']))
P('bj.mina', BD_['min_a'], 1)
V.raw('bj.minad', month(BD_['min_a_d']))
P('bj.meanm', BD_['mean_m'], 2)
P('bj.mjan', BD_['m_by_month']['1'], 2)
P('bj.mjun', BD_['m_by_month']['6'], 2)
P('bj.a2507', BD_['a_2025_07'], 1)
P('bj.a2508', BD_['a_2025_08'], 1)
P('bj.m2508', BD_['m_2025_08'], 2)
TS = N['tests']
for k in ('y', 'm', 'a', 'z'):
    t = TS[k]
    P(f'ts.{k}.adf', t['adf'], 2)
    V.raw(f'ts.{k}.p', pv(t['adf_p']))
    P(f'ts.{k}.cv', t['adf_cv5'], 2)
    P(f'ts.{k}.kpss', t['kpss'], 3)
    P(f'ts.{k}.kcv', t['kpss_cv5'], 3)
    V.raw(f'ts.{k}.n', str(t['n']))
AC = N['acf']
P('ac.band', AC['band'], 3)
for j in (1, 2, 12, 13, 24, 36):
    P(f'ac.r{j}', AC['r'][j - 1], 2)
    P(f'ac.p{j}', AC['p'][j - 1], 2)
V.raw('ac.n', str(AC['n']))
GR = N['grid']
V.raw('gr.n', str(GR['n_models']))
V.raw('gr.ntr', str(GR['n_train']))
V.raw('gr.trlast', month(GR['train_last']))
ROWS = {r['model']: r for r in GR['rows']}
SHOW = [GR['ident']['model']] + [r['model'] for r in GR['rows'][:5] if r['model'] != GR['ident']['model']][:4]
for i, mname in enumerate(SHOW):
    r = ROWS[mname]
    V.raw(f'gr{i}.m', mname)
    P(f'gr{i}.aicc', r['aicc'], 1)
    P(f'gr{i}.bic', r['bic'], 1)
    V.raw(f'gr{i}.lb12', pv(r['lb12_p']))
    V.raw(f'gr{i}.lb24', pv(r['lb24_p']))
    V.raw(f'gr{i}.k', str(r['k']))
V.raw('gr.best', GR['best_bic'])
V.raw('gr.chosen', GR['chosen'])
DG = N['diag']['chosen']
for k in ('ar.L1', 'ar.L2', 'ma.L1', 'ma.S.L12'):
    P(f'dg.{k}', DG['params'][k], 3)
    P(f'dg.se.{k}', DG['se'][k], 3)
    P(f'dg.z.{k}', DG['z'][k], 2)
    V.raw(f'dg.p.{k}', pv(DG['p'][k]))
P('dg.s2', DG['params']['sigma2'], 3)
P('dg.lb12', DG['lb']['12']['q'], 2)
V.raw('dg.lb12df', str(DG['lb']['12']['df']))
V.raw('dg.lb12p', pv(DG['lb']['12']['p']))
P('dg.lb24', DG['lb']['24']['q'], 2)
V.raw('dg.lb24df', str(DG['lb']['24']['df']))
V.raw('dg.lb24p', pv(DG['lb']['24']['p']))
P('dg.jb', DG['jb'], 0)
P('dg.kurt', DG['kurt'], 1)
V.raw('dg.big1', month(DG['big'][0][0]))
P('dg.big1v', DG['big'][0][1], 1)
V.raw('dg.big2', month(DG['big'][1][0]))
P('dg.big2v', DG['big'][1][1], 1)
P('dg.arroot', min(DG['ar_roots']), 3)
DI = N['diag']['ident']
V.raw('di.lb12p', pv(DI['lb']['12']['p']))
V.raw('di.lb24p', pv(DI['lb']['24']['p']))
FC = N['fc']
V.raw('fc.ntest', str(FC['n_test']))
V.raw('fc.tf', month(FC['test_first']))
V.raw('fc.tl', month(FC['test_last']))
for nm, kk in (('SARIMA', 's'), ('Seasonal naive', 'sn'), ('Naive', 'nv')):
    for m in ('rmse', 'mae', 'mase'):
        P(f'fc.{kk}.{m}', FC['acc'][nm][m], 2 if m != 'mae' else 2)
P('fc.dmsn', FC['dm_sn']['hln'], 2)
V.raw('fc.dmsnp', pv(FC['dm_sn']['p']))
P('fc.dmnv', FC['dm_nv']['hln'], 2)
V.raw('fc.dmnvp', pv(FC['dm_nv']['p']))
V.raw('fc.w1', month(FC['worst'][0][0]))
P('fc.w1a', FC['worst'][0][1], 2)
P('fc.w1f', FC['worst'][0][2], 2)
V.raw('fc.w2', month(FC['worst'][1][0]))
P('fc.w2a', FC['worst'][1][1], 2)
P('fc.w2f', FC['worst'][1][2], 2)
V.raw('fc.f1d', month(FC['fc_first']))
V.raw('fc.f12d', month(FC['fc_last']))
for k in ('a1', 'a1_lo', 'a1_hi', 'a6', 'a12', 'a12_lo', 'a12_hi', 'last_a'):
    P(f'fc.{k}', FC[k], 1)

# =============================================================================
# PROBLEMELE DE EXAMEN (rezolvări calculate aici sau citite din capitole)
# =============================================================================
# P1: trend determinist cu erori AR(1); supradiferențierea
a1, b1, phi1, Tn, yT = 10.0, 0.5, 0.6, 100, 62.0
uT = yT - a1 - b1 * Tn
P('e1.var', 1 / (1 - phi1 ** 2), 4)
P('e1.vdu', 2 * (1 - phi1) / (1 - phi1 ** 2), 2)
P('e1.cdu', (2 * phi1 - 1 - phi1 ** 2) / (1 - phi1 ** 2), 2)
P('e1.rdu', (2 * phi1 - 1 - phi1 ** 2) / (2 * (1 - phi1)), 2)
P('e1.uT', uT, 1)
P('e1.f1', a1 + b1 * (Tn + 1) + phi1 * uT, 2)
P('e1.f10', a1 + b1 * (Tn + 10) + phi1 ** 10 * uT, 2)
P('e1.p10', phi1 ** 10 * uT, 3)
P('e1.v10', (1 - phi1 ** 20) / (1 - phi1 ** 2), 3)

# P2: PIB-ul României (Capitolul 3)
c3 = C[3]
U1, U2 = c3['urt']['Romania real GDP, log'], c3['urt']['Romania real GDP, growth']
for tag, u in (('l', U1), ('g', U2)):
    P(f'e2.{tag}.adf', u['adf']['stat'], 2)
    V.raw(f'e2.{tag}.p', pv(u['adf']['p']))
    P(f'e2.{tag}.cv1', u['adf']['crit1'], 2)
    P(f'e2.{tag}.cv5', u['adf']['crit5'], 2)
    P(f'e2.{tag}.cv10', u['adf']['crit10'], 2)
    P(f'e2.{tag}.kpss', u['kpss']['stat'], 3)
    P(f'e2.{tag}.kcv', u['kpss']['crit5'], 3)
    V.raw(f'e2.{tag}.n', str(u['adf']['nobs']))
GI = c3['gdp_id']
for k in ('r1', 'r2', 'p1', 'p2', 'band'):
    P(f'e2.{k}', GI[k], 2)
P('e2.lb8', GI['lb8']['lb'], 2)
V.raw('e2.lb8p', pv(GI['lb8']['lb_p']))
P('e2.chi8', stats.chi2.ppf(0.95, 8), 2)
GD = c3['gdp_diag']
P('e2.c', GD['params']['x1'], 2)
P('e2.cse', GD['se']['x1'], 2)
P('e2.ct', GD['params']['x1'] / GD['se']['x1'], 2)
P('e2.ann', 4 * GD['params']['x1'], 1)
P('e2.s', GD['sigma'], 2)
V.raw('e2.first', GI['first'])
V.raw('e2.last', GI['last'])

# P3: AR(1) x SAR(1) pentru z = ΔΔ12 ln IAPC (Quantlets/Ch_15)
E3 = N['p3']
for k in ('phi', 'Phi'):
    P(f'e3.{k}', E3[k], 3)
    P(f'e3.se{k}', E3['se_' + k], 3)
    P(f'e3.z{k}', E3['z_' + k], 2)
    V.raw(f'e3.p{k}', pv(E3['p_' + k]))
P('e3.s2', E3['sigma2'], 3)
P('e3.lb12', E3['lb']['12']['q'], 2)
V.raw('e3.lb12p', pv(E3['lb']['12']['p']))
P('e3.aic', E3['aic'], 1)
V.raw('e3.n', str(E3['n']))
V.raw('e3.first', month(E3['first']))
V.raw('e3.last', month(E3['last']))
for k in ('zT', 'zT11', 'zT12'):
    P(f'e3.{k}', E3[k], 3)
V.raw('e3.dT', month(E3['dT']))
V.raw('e3.dT11', month(E3['dT11']))
V.raw('e3.dT12', month(E3['dT12']))
V.raw('e3.next', month(E3['next']))
P('e3.aT', E3['aT'], 3)
P('e3.t1', E3['phi'] * E3['zT'], 3)
P('e3.t2', E3['Phi'] * E3['zT11'], 3)
P('e3.t3', -E3['phi'] * E3['Phi'] * E3['zT12'], 3)
P('e3.pP', E3['phi'] * E3['Phi'], 3)
P('e3.znext', E3['z_next_hand'], 3)
P('e3.anext', E3['a_next'], 2)
P('e3.inv', abs(E3['phi']), 2)
P('e3.invs', abs(E3['Phi']) ** (1 / 12), 2)

# P4: VAR-ul României (Capitolul 6)
c6 = C[6]
EI = c6['est']['params']['i']
TI = c6['est']['tvalues']['i']
for k in ('const', 'L1.g', 'L1.pi', 'L1.i', 'L2.g', 'L2.pi', 'L2.i'):
    P(f'e4.b.{k}', EI[k], 3)
    P(f'e4.t.{k}', TI[k], 2)
P('e4.r2', c6['est']['r2']['i'], 2)
V.raw('e4.n', str(c6['est']['nobs']))
V.raw('e4.first', quarter(c6['est']['first_used']))
V.raw('e4.last', quarter(c6['est']['last']))
GT = c6['gr_ro']['tests']
for k, tag in (('g->i', 'gi'), ('pi->i', 'pii'), ('i->g', 'ig'), ('i->pi', 'ipi'), ('g->pi', 'gpi'), ('pi->g', 'pig')):
    P(f'e4.F.{tag}', GT[k]['F'], 2)
    V.raw(f'e4.p.{tag}', pv(GT[k]['p']))
gh = c6['gr_hand']
P('e4.rssr', gh['rss_r'], 2)
P('e4.rssu', gh['rss_u'], 2)
P('e4.F', gh['F'], 2)
V.raw('e4.df2', str(gh['df2']))
V.raw('e4.T', str(gh['T']))
P('e4.crit', gh['crit5'], 2)
num_pi = EI['L1.pi'] + EI['L2.pi']
den = 1 - EI['L1.i'] - EI['L2.i']
P('e4.num', num_pi, 3)
P('e4.den', den, 3)
P('e4.lr', num_pi / den, 2)

# P5: VECM pentru randamentele SUA (Capitolul 7)
c7 = C[7]
VE = c7['vecm']
JY = c7['joh']['US yields 1y, 5y, 10y']
P('e5.b13', -VE['beta'][2][0], 3)
P('e5.b23', -VE['beta'][2][1], 3)
P('e5.c1', VE['const_coint'][0], 3)
P('e5.c2', VE['const_coint'][1], 3)
for i, nm in enumerate(('1', '5', '10')):
    for j in range(2):
        P(f'e5.a{nm}{j + 1}', VE['alpha'][i][j], 3)
        P(f'e5.t{nm}{j + 1}', VE['alpha_t'][i][j], 2)
for r in range(3):
    P(f'e5.tr{r}', JY['trace'][r], 2)
    P(f'e5.cv{r}', JY['trace_cv5'][r], 2)
V.raw('e5.n', str(VE['n']))
V.raw('e5.first', month(VE['first'] + '-01'))
V.raw('e5.last', month(VE['last'] + '-01'))
P('e5.h1', VE['ec_half'][0], 1)
P('e5.h2', VE['ec_half'][1], 1)
P('e5.e1', VE['ec_eig'][0], 3)
P('e5.e2', VE['ec_eig'][1], 3)

# P6: GARCH(1,1)-t pentru BET (Capitolul 5)
GB = C[5]['markets']['bet']
mu6, om6, al6, be6, nu6 = (GB['params'][k] for k in ('mu', 'omega', 'alpha[1]', 'beta[1]', 'nu'))
for k, v in (('mu', mu6), ('om', om6), ('al', al6), ('be', be6), ('nu', nu6)):
    P(f'e6.{k}', v, 3 if k != 'om' else 4)
for k, kk in (('mu', 'mu'), ('omega', 'om'), ('alpha[1]', 'al'), ('beta[1]', 'be'), ('nu', 'nu')):
    P(f'e6.se.{kk}', GB['se'][k], 3 if kk != 'om' else 4)
V.raw('e6.n', f"{GB['n']:,}".replace(',', '\\,'))
s2T6, eT6 = 1.0, -3.0
s2n6 = om6 + al6 * eT6 ** 2 + be6 * s2T6
q6 = stats.t.ppf(0.01, round(nu6)) * math.sqrt((round(nu6) - 2) / round(nu6))
P('e6.a9', al6 * eT6 ** 2, 3)
P('e6.s2n', s2n6, 3)
P('e6.sn', math.sqrt(s2n6), 3)
P('e6.pers', al6 + be6, 3)
P('e6.hl', math.log(0.5) / math.log(al6 + be6), 0)
P('e6.lv', om6 / (1 - al6 - be6), 2)
P('e6.ann', math.sqrt(250 * om6 / (1 - al6 - be6)), 1)
P('e6.ann_ch', GB['vol_lr'], 1)
P('e6.vs', GB['vol_sample'], 1)
P('e6.tq', stats.t.ppf(0.01, round(nu6)), 3)
P('e6.sc', math.sqrt((round(nu6) - 2) / round(nu6)), 3)
P('e6.q', q6, 3)
P('e6.var', -(mu6 + math.sqrt(s2n6) * q6), 2)
P('e6.varn', -(mu6 + math.sqrt(s2n6) * stats.norm.ppf(0.01)), 2)
V.raw('e6.nu5', str(round(nu6)))

# P7: evaluarea prognozelor și testul DM
EA = [0.3, -0.5, 0.2, 0.6, -0.4, 0.1]
EB = [0.5, -0.7, 0.6, 0.4, -0.9, 0.3]
SC = 0.6
n7 = len(EA)
rm = lambda e: math.sqrt(sum(x * x for x in e) / len(e))   # noqa: E731
ma = lambda e: sum(abs(x) for x in e) / len(e)             # noqa: E731
P('e7.rA', rm(EA), 3)
P('e7.rB', rm(EB), 3)
P('e7.mA', ma(EA), 3)
P('e7.mB', ma(EB), 3)
P('e7.qA', ma(EA) / SC, 2)
P('e7.qB', ma(EB) / SC, 2)
d7 = [a * a - b * b for a, b in zip(EA, EB)]
db = sum(d7) / n7
sd = math.sqrt(sum((x - db) ** 2 for x in d7) / (n7 - 1))
for i, x in enumerate(d7, 1):
    P(f'e7.d{i}', x, 2)
P('e7.db', db, 3)
P('e7.sd', sd, 3)
P('e7.dm', db / (sd / math.sqrt(n7)), 2)
P('e7.crit', stats.t.ppf(0.975, n7 - 1), 3)
P('e7.kf', C[9]['leakage']['sp500']['kfold'], 2)
P('e7.wf', C[9]['leakage']['sp500']['wf'], 2)

# P8: Kalman, Markov, ARFIMA
s2e8, s2n8, a8, P8, y8 = 4.0, 1.0, 10.0, 2.0, 13.0
F8 = P8 + s2e8
K8 = P8 / F8
P('e8.F', F8, 0)
P('e8.K', K8, 3)
P('e8.af', a8 + K8 * (y8 - a8), 2)
P('e8.Pf', P8 * (1 - K8), 3)
P('e8.Pn', P8 * (1 - K8) + s2n8, 3)
q8 = s2n8 / s2e8
Pb = (q8 + math.sqrt(q8 ** 2 + 4 * q8)) / 2
P('e8.Pb', Pb, 4)
P('e8.Kb', Pb / (Pb + 1), 3)
MR = C[10]['msro']
P('e8.p11', MR['p11'], 3)
P('e8.p22', MR['p22'], 3)
P('e8.d1', 1 / (1 - MR['p11']), 1)
P('e8.d2', 1 / (1 - MR['p22']), 1)
P('e8.pi1', (1 - MR['p22']) / (2 - MR['p11'] - MR['p22']), 3)
P('e8.pi1p', 100 * (1 - MR['p22']) / (2 - MR['p11'] - MR['p22']), 0)
ARF = next(r for r in C[8]['fit']['table'] if r['name'] == 'ARFIMA(0,d,0)')
P('e8.d', ARF['d'], 2)
P('e8.dse', ARF['se_d'], 2)
P('e8.H', ARF['d'] + 0.5, 2)
P('e8.r1', ARF['d'] / (1 - ARF['d']), 2)
neg(V)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), two(items(
    (T('\\textbf{Question}: what does the whole course say about a time series, and how is it examined?',
       '\\textbf{Întrebarea}: ce spune întregul curs despre o serie de timp și cum se evaluează aceste cunoștințe la examen?'),
     [T('one chapter that ties Chapters 0--14 together', 'un capitol care leagă Capitolele 0--14')]),
    (T('\\textbf{Route}', '\\textbf{Traseul}'),
     [T('Part I: the course map, one recap slide per chapter, the toolbox of tests and models', 'Partea I: harta cursului, cîte un slide de recapitulare pentru fiecare capitol, trusa de teste și modele'),
      T('Part II: the Box--Jenkins method from start to finish on Romanian inflation', 'Partea a II-a: metoda Box--Jenkins de la un capăt la altul, pe inflația din România'),
      T('Part III: the exam (format, grading, eight solved problems), the team project and attendance', 'Partea a III-a: examenul (format, criterii, opt probleme rezolvate), proiectul de echipă și prezența')]),
    T('Chapters 11--14 are for self-study', 'Capitolele 11--14 sînt pentru studiu individual')),
    ph('ase', T('Bucharest University of Economic Studies', 'Academia de Studii Economice din București'), h='0.40\\textheight')))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Place every chapter on the path from describing a series to modelling and forecasting it', 'Situați fiecare capitol pe drumul de la descrierea unei serii la modelarea și prognoza ei'),
    T('Recall the key formula, the key empirical fact and the most common mistake of each chapter', 'Reamintiți formula-cheie, faptul empiric principal și greșeala cea mai frecventă din fiecare capitol'),
    T('Choose the right test or model for a given question about a time series', 'Alegeți testul sau modelul potrivit pentru o întrebare dată despre o serie de timp'),
    T('Run the Box--Jenkins method on a real series: transform, test, identify, estimate, check, forecast', 'Aplicați metoda Box--Jenkins pe o serie reală: transformare, testare, identificare, estimare, verificare, prognoză'),
    T('Read software output (estimates, tests, correlograms) and interpret it in three to five sentences', 'Citiți rezultatele obținute cu software (estimări, teste, corelograme) și interpretați-le în trei pînă la cinci fraze'),
    T('Plan the team project, its oral defence and the declaration of AI use', 'Planificați proiectul de echipă, susținerea orală și declararea utilizării AI')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refHP; free companion: \\refFPP', 'Manual: \\refHP; manual însoțitor gratuit: \\refFPP'),
     [T('theory: \\refBD; \\refHamilton; the method: \\refBJ', 'teorie: \\refBD; \\refHamilton; metoda: \\refBJ')]),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_15}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_15}'),
     [T('the Box--Jenkins case on the Romanian HICP and the output of exam problem 3', 'cazul Box--Jenkins pe IAPC-ul României și rezultatele pentru problema 3 de examen')]),
    T('Lecture notebook, the course in one notebook: \\href{\\colaburl{notebooks/EN/chapter15_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului, cursul într-un singur notebook: \\href{\\colaburl{notebooks/EN/chapter15_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Self-assessment: the quiz of this chapter, 20 questions drawn from all chapters', 'Autoevaluare: quiz-ul acestui capitol, 20 de întrebări extrase din toate capitolele')))

# =============================================================================
# 1. HARTA CURSULUI
# =============================================================================
D.section('The course map', 'Harta cursului')

MAP = r"""\begin{center}
\resizebox{0.97\textwidth}{!}{%
\begin{tikzpicture}[
  box/.style={draw=MainBlue, rounded corners=2pt, fill=white, font=\scriptsize, align=center, minimum height=0.95cm, text width=2.15cm},
  key/.style={box, draw=IDAred, line width=0.9pt},
  self/.style={box, dashed, draw=Purple},
  lab/.style={font=\scriptsize\bfseries, text=MainBlue, anchor=east, align=right},
  arr/.style={-{Stealth[length=2mm]}, MainBlue, line width=0.5pt},
  karr/.style={-{Stealth[length=2.4mm]}, IDAred, line width=1.1pt},
  sarr/.style={-{Stealth[length=2mm]}, Purple, dashed, line width=0.5pt}]
\node[lab] at (-0.2, 0) {⟦One series:\\ the mean||O serie:\\ media⟧};
\node[key] (c0) at (1.3, 0) {0 ⟦Description:\\ components, smoothing||Descriere:\\ componente, netezire⟧};
\node[key] (c1) at (3.9, 0) {1 ⟦Stationarity,\\ ACF, PACF||Staționaritate,\\ ACF, PACF⟧};
\node[key] (c2) at (6.5, 0) {2 ARMA};
\node[key] (c3) at (9.1, 0) {3 ⟦Unit roots,\\ ARIMA||Rădăcini\\ unitare, ARIMA⟧};
\node[key] (c4) at (11.7, 0) {4 ⟦Seasonality,\\ SARIMA, evaluation||Sezonalitate,\\ SARIMA, evaluare⟧};
\node[lab] at (-0.2, -1.6) {⟦Variance and\\ several series||Varianța și\\ mai multe serii⟧};
\node[key] (c5) at (3.9, -1.6) {5 ⟦Volatility:\\ ARCH, GARCH||Volatilitate:\\ ARCH, GARCH⟧};
\node[key] (c6) at (6.5, -1.6) {6 ⟦VAR, Granger,\\ IRF, FEVD||VAR, Granger,\\ IRF, FEVD⟧};
\node[key] (c7) at (9.1, -1.6) {7 ⟦Cointegration,\\ VECM||Cointegrare,\\ VECM⟧};
\node[lab] at (-0.2, -3.2) {⟦Extensions||Extensii⟧};
\node[box] (c8) at (3.9, -3.2) {8 ⟦Long memory,\\ ARFIMA||Memorie lungă,\\ ARFIMA⟧};
\node[box] (c9) at (6.5, -3.2) {9 ⟦Machine\\ learning||Învățare\\ automată⟧};
\node[box] (c10) at (9.1, -3.2) {10 ⟦State space,\\ Kalman, regimes||Spațiul stărilor,\\ Kalman, regimuri⟧};
\node[lab] at (-0.2, -4.8) {⟦Self-study||Studiu\\ individual⟧};
\node[self] (c11) at (3.9, -4.8) {11 Foundation\\ models};
\node[self] (c12) at (6.5, -4.8) {12 ⟦Spectral\\ analysis||Analiză\\ spectrală⟧};
\node[self] (c13) at (9.1, -4.8) {13 ⟦Bubbles:\\ LPPL||Bule:\\ LPPL⟧};
\node[self] (c14) at (11.7, -4.8) {14 ⟦Multivariate\\ GARCH||GARCH\\ multivariat⟧};
\draw[karr] (c0) -- (c1); \draw[karr] (c1) -- (c2); \draw[karr] (c2) -- (c3); \draw[karr] (c3) -- (c4);
\draw[karr] (c2.south west) -- (c5.north east); \draw[karr] (c3) -- (c7); \draw[arr] (c2) -- (c6); \draw[arr] (c6) -- (c7);
\draw[arr] (c5) -- (c8); \draw[arr] (c6) -- (c9); \draw[arr] (c4.south) -- (11.7, -3.2) -- (c10.east);
\end{tikzpicture}}
\end{center}"""

D.frame(T('The course map: Chapters 0--14', 'Harta cursului: Capitolele 0--14'), MAP + '\n\\vspace{-0.2cm}\n' + items(
    T('Red: the core of the exam, from describing one series to modelling several series together', 'Roșu: nucleul examenului, de la descrierea unei serii la modelarea împreună a mai multor serii'),
    T('Blue: extensions (Chapters 8--10); dashed: self-study (Chapters 11--14), each built on a core chapter: 11 on 9, 12 on 1 and 8, 13 on 3, 14 on 5 and 6', 'Albastru: extensii (Capitolele 8--10); linie întreruptă: studiu individual (Capitolele 11--14), fiecare construit pe un capitol de bază: 11 pe 9, 12 pe 1 și 8, 13 pe 3, 14 pe 5 și 6')), 'footnotesize')

D.frame(T('The series of the course', 'Seriile cursului'), table(
    '>{\\raggedright\\arraybackslash}p{3.6cm}>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{4.2cm}l',
    T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Topic illustrated}', '\\textbf{Tema ilustrată}') + ' & ' + T('\\textbf{Ch.}', '\\textbf{Cap.}'),
    [T('Romanian real GDP (quarterly)', 'PIB-ul real al României (trimestrial)') + ' & Eurostat & ' + T('trend, seasonality, a unit root, regimes', 'trend, sezonalitate, rădăcină unitară, regimuri') + ' & 0--4, 10',
     T('Romanian HICP inflation (monthly)', 'Inflația IAPC a României (lunar)') + ' & Eurostat & ' + T('persistence, seasonality, long memory', 'persistență, sezonalitate, memorie lungă') + ' & 2--4, 8',
     T('ROBOR 3M, unemployment', 'ROBOR 3M, șomaj') + ' & Eurostat & ' + T('a VAR for the Romanian economy', 'un VAR pentru economia României') + ' & 6',
     'EUR/RON & BNR & ' + T('a random walk that is hard to beat', 'un mers aleator greu de depășit') + ' & 0, 3, 5',
     T('BET, S\\&P 500, DAX (daily)', 'BET, S\\&P 500, DAX (zilnic)') + ' & EODHD & ' + T('volatility clustering, spillovers', 'volatility clustering, spillover') + ' & 1, 5, 6',
     T('electricity load (daily, hourly)', 'consumul de electricitate (zilnic, orar)') + ' & ENTSO-E & ' + T('multiple seasonality, machine learning', 'sezonalitate multiplă, învățare automată') + ' & 4, 9',
     T('Nile, sunspots, US yields and GDP', 'Nilul, petele solare, randamentele și PIB-ul SUA') + ' & statsmodels, FRED & ' + T('the classic examples of the textbooks', 'exemplele clasice ale manualelor') + ' & 1, 2, 7, 8, 10'],
    size='footnotesize') + items(
    T('Every number on the recap slides is the number of its chapter: same data, same window, same code', 'Fiecare cifră de pe slide-urile de recapitulare este cifra din capitolul ei: aceleași date, aceeași fereastră, același cod')))

# =============================================================================
# 2. CAPITOLELE 0--3
# =============================================================================
D.section('Chapters 0--3: description, stationarity, ARMA, ARIMA', 'Capitolele 0--3: descriere, staționaritate, ARMA, ARIMA')

review(0, ('Introduction: components and exponential smoothing', 'introducere: componente și netezire exponențială'),
       [T('additive $y_t = T_t + S_t + R_t$ or multiplicative $y_t = T_t S_t R_t$; the logarithm turns one into the other', 'aditiv $y_t = T_t + S_t + R_t$ sau multiplicativ $y_t = T_t S_t R_t$; logaritmul transformă un model în celălalt'),
        T('SES $\\ell_t = \\alpha y_t + (1 - \\alpha)\\ell_{t-1}$; MASE $=$ MAE $/$ MAE of the in-sample seasonal naive method', 'SES $\\ell_t = \\alpha y_t + (1 - \\alpha)\\ell_{t-1}$; MASE $=$ MAE $/$ MAE metodei naive sezoniere în eșantion')],
       [T('Romanian GDP grew @{f0.mult} times since 1995 (@{f0.g}\\% a year); Q1 is about $@{f0.q1}\\%$ and Q4 $+@{f0.q4}\\%$ around the trend', 'PIB-ul României a crescut de @{f0.mult} ori din 1995 (@{f0.g}\\% pe an); trimestrul I este cu circa $@{f0.q1}\\%$, iar trimestrul IV cu $+@{f0.q4}\\%$ față de trend'),
        T('EUR/RON: $\\hat\\alpha_{SES} = @{f0.ses}$, the last value is the best forecast; electricity: Holt--Winters MASE @{f0.hw} against @{f0.nv} for the naive method', 'EUR/RON: $\\hat\\alpha_{SES} = @{f0.ses}$, ultima valoare este cea mai bună prognoză; electricitatea: Holt--Winters are MASE @{f0.hw}, față de @{f0.nv} pentru metoda naivă')],
       [T('judging a forecast on the data used to fit it', 'judecarea unei prognoze pe datele folosite la estimare'),
        T('MAPE for series that cross zero (monthly inflation); an additive model for a seasonal swing that grows with the level', 'MAPE pentru serii care trec prin zero (inflația lunară); un model aditiv pentru o oscilație sezonieră care crește odată cu nivelul')])

review(1, ('Stochastic processes and stationarity', 'procese stochastice și staționaritate'),
       [T('weak stationarity: $E X_t = \\mu$, $\\mathrm{Cov}(X_t, X_{t+h}) = \\gamma(h)$; $\\hat\\rho(h)$ against $\\pm 1.96/\\sqrt{T}$', 'staționaritate slabă: $E X_t = \\mu$, $\\mathrm{Cov}(X_t, X_{t+h}) = \\gamma(h)$; $\\hat\\rho(h)$ comparat cu $\\pm 1.96/\\sqrt{T}$'),
        T('Ljung--Box $Q^*(m) = T(T + 2)\\sum_{h=1}^{m}\\hat\\rho(h)^2/(T - h) \\sim \\chi^2(m)$ \\refLB; random walk $\\mathrm{Var}(X_t) = t\\sigma^2$', 'Ljung--Box $Q^*(m) = T(T + 2)\\sum_{h=1}^{m}\\hat\\rho(h)^2/(T - h) \\sim \\chi^2(m)$ \\refLB; mers aleator $\\mathrm{Var}(X_t) = t\\sigma^2$')],
       [T('BET returns: $\\hat\\rho_1 = @{f1.r1}$, absolute returns $\\hat\\rho_1 = @{f1.a1}$ (band $\\pm @{f1.band}$): uncorrelated is not independent', 'randamentele BET: $\\hat\\rho_1 = @{f1.r1}$, randamentele absolute $\\hat\\rho_1 = @{f1.a1}$ (banda $\\pm @{f1.band}$): necorelat nu înseamnă independent'),
        T('Romanian GDP: Box--Cox $\\hat\\lambda = @{f1.lam}$, the logarithm; one difference too many gives $\\hat\\rho_1 = @{f1.od}$', 'PIB-ul României: Box--Cox $\\hat\\lambda = @{f1.lam}$, adică logaritmul; o diferențiere în plus dă $\\hat\\rho_1 = @{f1.od}$')],
       [T('reading one bar outside the band as a signal (one in twenty is outside by chance)', 'interpretarea unei singure bare din afara benzii ca semnal (una din douăzeci iese din întîmplare)'),
        T('differencing a series that is already stationary', 'diferențierea unei serii care este deja staționară')])

review(2, ('ARMA models', 'modele ARMA'),
       [T('$\\phi(L)(X_t - \\mu) = \\theta(L)\\varepsilon_t$; AR(1) $\\rho(h) = \\phi^h$; MA(1) $\\rho(1) = \\theta/(1 + \\theta^2)$', '$\\phi(L)(X_t - \\mu) = \\theta(L)\\varepsilon_t$; AR(1) $\\rho(h) = \\phi^h$; MA(1) $\\rho(1) = \\theta/(1 + \\theta^2)$'),
        T('AIC $= -2\\ln L + 2k$ \\refAkaike, BIC $= -2\\ln L + k\\ln T$ \\refSchwarz; residual check $Q^*(m) \\sim \\chi^2(m - p - q)$', 'AIC $= -2\\ln L + 2k$ \\refAkaike, BIC $= -2\\ln L + k\\ln T$ \\refSchwarz; verificarea reziduurilor $Q^*(m) \\sim \\chi^2(m - p - q)$')],
       [T('annual growth of Romanian GDP: MA(3) by BIC, a product of overlapping quarters; residual Ljung--Box $p = @{f2.lb}$', 'creșterea anuală a PIB-ului României: MA(3) după BIC, efectul trimestrelor suprapuse; Ljung--Box pe reziduuri $p = @{f2.lb}$'),
        T('Romanian inflation: AR(2) with $\\hat\\phi_1 + \\hat\\phi_2 = @{f2.sumphi}$, close to a unit root; sunspots: an AR(2) cycle of @{f2.period} years', 'inflația din România: AR(2) cu $\\hat\\phi_1 + \\hat\\phi_2 = @{f2.sumphi}$, aproape de o rădăcină unitară; petele solare: un ciclu AR(2) de @{f2.period} ani')],
       [T('reading the AR order from the ACF (it is the PACF that cuts off for an AR)', 'citirea ordinului AR din ACF (la un AR se anulează PACF)'),
        T('Ljung--Box on residuals with $m$ instead of $m - p - q$ degrees of freedom; a common factor in $\\phi(L)$ and $\\theta(L)$', 'Ljung--Box pe reziduuri cu $m$ în loc de $m - p - q$ grade de libertate; un factor comun în $\\phi(L)$ și $\\theta(L)$')])

review(3, ('Unit roots and ARIMA models', 'rădăcini unitare și modele ARIMA'),
       [T('ADF: $\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum_j\\delta_j\\Delta y_{t-j} + \\varepsilon_t$, $H_0$: $\\gamma = 0$ \\refDF; KPSS: $H_0$ stationarity \\refKPSS', 'ADF: $\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum_j\\delta_j\\Delta y_{t-j} + \\varepsilon_t$, $H_0$: $\\gamma = 0$ \\refDF; KPSS: $H_0$ staționaritate \\refKPSS'),
        T('ARIMA$(p,d,q)$: $\\phi(L)(1 - L)^d y_t = c + \\theta(L)\\varepsilon_t$; forecast variance grows with $h$ when $d \\ge 1$', 'ARIMA$(p,d,q)$: $\\phi(L)(1 - L)^d y_t = c + \\theta(L)\\varepsilon_t$; varianța prognozei crește cu $h$ cînd $d \\ge 1$')],
       [T('log Romanian GDP: $\\tau = @{f3.adf1}$ ($p = @{f3.adf1p}$, 5\\% critical value $@{f3.cv}$); growth: $\\tau = @{f3.adf2}$; ARIMA(0,1,0) with drift @{f3.drift}\\% a quarter', 'logaritmul PIB-ului României: $\\tau = @{f3.adf1}$ ($p = @{f3.adf1p}$, valoarea critică la 5\\% $@{f3.cv}$); creșterea: $\\tau = @{f3.adf2}$; ARIMA(0,1,0) cu derivă de @{f3.drift}\\% pe trimestru'),
        T('two independent random walks: a significant $t$ in @{f3.spur}\\% of regressions \\refGN; ADF power only @{f3.pow}\\% for $\\phi = 0.95$, $T = 100$', 'două mersuri aleatoare independente: un $t$ semnificativ în @{f3.spur}\\% din regresii \\refGN; puterea ADF doar @{f3.pow}\\% pentru $\\phi = 0.95$, $T = 100$')],
       [T('Student or Normal critical values for the ADF statistic', 'valori critice Student sau Normale pentru statistica ADF'),
        T('reading a non-rejection as proof of a unit root; a regression in levels of $I(1)$ series without a cointegration test', 'interpretarea nerespingerii ca dovadă a rădăcinii unitare; o regresie în niveluri între serii $I(1)$ fără test de cointegrare')])

D.recap(('Chapters 0--3', 'Capitolele 0--3'), [
    T('Plot first, then transform: logarithm for a growing swing, differences for a stochastic trend', 'Întîi graficul, apoi transformarea: logaritm pentru o oscilație care crește, diferențe pentru un trend stochastic'),
    T('ACF and PACF identify an ARMA; AIC and BIC choose; Ljung--Box on residuals checks', 'ACF și PACF identifică un ARMA; AIC și BIC aleg; Ljung--Box pe reziduuri verifică'),
    T('ADF and KPSS together decide $d$; Dickey--Fuller critical values, never Student ones', 'ADF și KPSS împreună decid $d$; valori critice Dickey--Fuller, niciodată Student')])

# =============================================================================
# 3. CAPITOLELE 4--7
# =============================================================================
D.section('Chapters 4--7: seasonality, volatility, VAR, VECM', 'Capitolele 4--7: sezonalitate, volatilitate, VAR, VECM')

review(4, ('Seasonality and forecasting', 'sezonalitate și prognoză'),
       [T('SARIMA $\\phi(L)\\Phi(L^s)(1 - L)^d(1 - L^s)^D y_t = \\theta(L)\\Theta(L^s)\\varepsilon_t$; airline $(0,1,1)(0,1,1)_s$', 'SARIMA $\\phi(L)\\Phi(L^s)(1 - L)^d(1 - L^s)^D y_t = \\theta(L)\\Theta(L^s)\\varepsilon_t$; airline $(0,1,1)(0,1,1)_s$'),
        T('Diebold--Mariano $\\bar d/\\sqrt{\\hat V/n}$, $d_t = L(e_{1t}) - L(e_{2t})$ \\refDM, with the HLN correction \\refHLN', 'Diebold--Mariano $\\bar d/\\sqrt{\\hat V/n}$, $d_t = L(e_{1t}) - L(e_{2t})$ \\refDM, cu corecția HLN \\refHLN')],
       [T('airline passengers: MAPE @{f4.mape}\\% for the airline model against @{f4.mapesn}\\% for the seasonal naive method', 'pasagerii aerieni: MAPE @{f4.mape}\\% pentru modelul airline, față de @{f4.mapesn}\\% pentru metoda naivă sezonieră'),
        T('Orthodox Easter raises Romanian food sales by about @{f4.east}\\%; daily load: DHR MASE @{f4.dhr} against @{f4.snv} (seasonal naive)', 'Paștele ortodox crește vînzările de alimente din România cu circa @{f4.east}\\%; consumul zilnic: DHR are MASE @{f4.dhr}, față de @{f4.snv} (naiv sezonier)')],
       [T('comparing AICc of models with different $d$ or $D$', 'compararea AICc pentru modele cu $d$ sau $D$ diferite'),
        T('judging forecasts at a single origin; seasonally adjusted data in a SARIMA forecast of raw data', 'judecarea prognozelor la o singură origine; date ajustate sezonier într-o prognoză SARIMA a datelor brute')])

review(5, ('Conditional volatility: ARCH and GARCH', 'volatilitate condiționată: ARCH și GARCH'),
       [T('GARCH(1,1) $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$ \\refEngle, \\refBoll; $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$, $h_{1/2} = \\ln 0.5/\\ln(\\alpha + \\beta)$', 'GARCH(1,1) $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$ \\refEngle, \\refBoll; $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$, $h_{1/2} = \\ln 0.5/\\ln(\\alpha + \\beta)$'),
        T('ARCH-LM $= nR^2 \\sim \\chi^2(q)$; VaR 1\\% $= -(\\mu + \\sigma_{t+1}q_{0.01}(z))$', 'ARCH-LM $= nR^2 \\sim \\chi^2(q)$; VaR 1\\% $= -(\\mu + \\sigma_{t+1}q_{0.01}(z))$')],
       [T('GARCH(1,1)-$t$: $\\alpha + \\beta = @{f5.psp}$ for the S\\&P 500 (half-life @{f5.hsp} days), $@{f5.pbet}$ for the BET (@{f5.hbet} days)', 'GARCH(1,1)-$t$: $\\alpha + \\beta = @{f5.psp}$ pentru S\\&P 500 (timp de înjumătățire @{f5.hsp} de zile), $@{f5.pbet}$ pentru BET (@{f5.hbet} de zile)'),
        T('S\\&P 500: GJR $\\hat\\gamma = @{f5.gj}$ ($t = @{f5.gjt}$); kurtosis @{f5.kr} for returns, @{f5.kz} for standardised residuals', 'S\\&P 500: GJR $\\hat\\gamma = @{f5.gj}$ ($t = @{f5.gjt}$); coeficientul de boltire @{f5.kr} pentru randamente, @{f5.kz} pentru reziduurile standardizate')],
       [T('the half-life written as $1/(1 - \\alpha - \\beta)$; a long-run variance reported when $\\alpha + \\beta \\ge 1$', 'timpul de înjumătățire scris $1/(1 - \\alpha - \\beta)$; o varianță pe termen lung raportată cînd $\\alpha + \\beta \\ge 1$'),
        T('``VaR 99\\%\'\' or a negative VaR: the course writes VaR 1\\%, a positive loss', '„VaR 99\\%” sau un VaR negativ: în curs scriem VaR 1\\%, o pierdere pozitivă')])

review(6, ('VAR models and Granger causality', 'modele VAR și cauzalitate Granger'),
       [T('VAR$(p)$: $\\bY_t = \\bc + \\bA_1\\bY_{t-1} + \\dots + \\bA_p\\bY_{t-p} + \\bepsilon_t$ \\refSims; stable if all eigenvalues of the companion matrix are inside the unit circle', 'VAR$(p)$: $\\bY_t = \\bc + \\bA_1\\bY_{t-1} + \\dots + \\bA_p\\bY_{t-p} + \\bepsilon_t$ \\refSims; stabil dacă toate valorile proprii ale matricei companion sînt în interiorul cercului unitate'),
        T('Granger $F = [(RSS_R - RSS_U)/p]\\,/\\,[RSS_U/(T - Kp - 1)]$ \\refGranger; IRF and FEVD with a Cholesky ordering', 'Granger $F = [(RSS_R - RSS_U)/p]\\,/\\,[RSS_U/(T - Kp - 1)]$ \\refGranger; IRF și FEVD cu o ordine Cholesky')],
       [T('Romanian VAR(2) of growth, inflation and ROBOR 3M: growth Granger-causes ROBOR, $F = @{f6.F}$, $p =$ @{f6.p}', 'VAR(2) pentru România (creștere, inflație, ROBOR 3M): creșterea cauzează Granger ROBOR, $F = @{f6.F}$, $p =$ @{f6.p}'),
        T('spillovers among the S\\&P 500, DAX and BET: @{f6.spt}\\% on average, @{f6.spm}\\% on @{f6.spd}', 'spillover între S\\&P 500, DAX și BET: în medie @{f6.spt}\\%, @{f6.spm}\\% la @{f6.spd}')],
       [T('reading Granger causality as causality (it is extra predictability)', 'interpretarea cauzalității Granger drept cauzalitate (este doar predictibilitate suplimentară)'),
        T('IRFs reported without the ordering behind them; too many lags for a short sample', 'IRF raportate fără ordinea care stă la baza lor; prea multe decalaje pentru un eșantion scurt')])

review(7, ('Cointegration and VECM', 'cointegrare și VECM'),
       [T('Engle--Granger: OLS in levels, then ADF on residuals with MacKinnon critical values \\refEG; ECM half-life $\\ln 0.5/\\ln(1 + \\gamma)$', 'Engle--Granger: OLS în niveluri, apoi ADF pe reziduuri cu valorile critice MacKinnon \\refEG; timpul de înjumătățire al ECM $\\ln 0.5/\\ln(1 + \\gamma)$'),
        T('VECM $\\Delta\\mathbf y_t = \\alpha\\beta^\\top\\mathbf y_{t-1} + \\sum_i\\Gamma_i\\Delta\\mathbf y_{t-i} + \\mathbf u_t$; trace $-T\\sum_{i>r}\\ln(1 - \\hat\\lambda_i)$ \\refJoh', 'VECM $\\Delta\\mathbf y_t = \\alpha\\beta^\\top\\mathbf y_{t-1} + \\sum_i\\Gamma_i\\Delta\\mathbf y_{t-i} + \\mathbf u_t$; statistica urmei $-T\\sum_{i>r}\\ln(1 - \\hat\\lambda_i)$ \\refJoh')],
       [T('US yields at 1, 5 and 10 years: trace @{f7.tr0} $>$ @{f7.cv0} for $r = 0$, @{f7.tr2} $<$ @{f7.cv2} for $r \\le 2$: rank 2; long-rate ECM half-life @{f7.hl} months', 'randamentele SUA la 1, 5 și 10 ani: urma @{f7.tr0} $>$ @{f7.cv0} pentru $r = 0$, @{f7.tr2} $<$ @{f7.cv2} pentru $r \\le 2$: rangul 2; timpul de înjumătățire al ECM pentru dobînda lungă @{f7.hl} luni'),
        T('PPP for the leu: ADF $p = @{f7.ppp}$ on the real exchange rate, not stationary; pairs trading on the BVB: Sharpe @{f7.sg} gross, @{f7.sn} net', 'PPP pentru leu: ADF $p = @{f7.ppp}$ pentru cursul real, nestaționar; pairs trading la BVB: Sharpe @{f7.sg} brut, @{f7.sn} net')],
       [T('Dickey--Fuller critical values for Engle--Granger residuals (they must be more negative)', 'valorile critice Dickey--Fuller pentru reziduurile Engle--Granger (trebuie să fie mai negative)'),
        T('a VAR in differences for cointegrated series (it omits the error correction term)', 'un VAR în diferențe pentru serii cointegrate (omite termenul de corecție a erorii)')])

D.recap(('Chapters 4--7', 'Capitolele 4--7'), [
    T('Seasonality: $\\Delta_s$ or Fourier terms; calendar effects as regressors; evaluation with rolling origins, MASE and DM', 'Sezonalitatea: $\\Delta_s$ sau termeni Fourier; efectele de calendar ca regresori; evaluare cu origini mobile, MASE și DM'),
    T('Volatility: ARCH-LM, then GARCH(1,1)-$t$; persistence close to 1; VaR 1\\% with the right quantile', 'Volatilitatea: ARCH-LM, apoi GARCH(1,1)-$t$; persistență aproape de 1; VaR 1\\% cu cuantila potrivită'),
    T('Several series: VAR for stationary data, VECM for cointegrated $I(1)$ data; Granger is predictability', 'Mai multe serii: VAR pentru date staționare, VECM pentru date $I(1)$ cointegrate; Granger înseamnă predictibilitate')])

# =============================================================================
# 4. CAPITOLELE 8--10
# =============================================================================
D.section('Chapters 8--10: long memory, machine learning, state space', 'Capitolele 8--10: memorie lungă, învățare automată, spațiul stărilor')

review(8, ('Long memory and ARFIMA', 'memorie lungă și ARFIMA'),
       [T('$(1 - L)^d$ with $\\pi_k = \\pi_{k-1}(k - 1 - d)/k$ \\refGJ, \\refHosking; $\\rho(k) \\sim Ck^{2d-1}$; $H = d + 1/2$', '$(1 - L)^d$ cu $\\pi_k = \\pi_{k-1}(k - 1 - d)/k$ \\refGJ, \\refHosking; $\\rho(k) \\sim Ck^{2d-1}$; $H = d + 1/2$'),
        T('stationary for $d < 1/2$, mean-reverting for $d < 1$; GPH and local Whittle use the lowest $m$ frequencies', 'staționar pentru $d < 1/2$, cu revenire la medie pentru $d < 1$; GPH și Whittle local folosesc cele mai joase $m$ frecvențe')],
       [T('Romanian monthly inflation: ARFIMA(0,$d$,0) with $\\hat d = @{f8.d}$ (SE @{f8.dse}) beats ARMA by AIC and BIC', 'inflația lunară din România: ARFIMA(0,$d$,0) cu $\\hat d = @{f8.d}$ (SE @{f8.dse}) bate ARMA după AIC și BIC'),
        T('S\\&P 500: $H$ (R/S) @{f8.hr} for returns, @{f8.ha} for $|r_t|$; the Nile: $\\hat d = @{f8.nraw}$, but $@{f8.nadj}$ after the break of 1898', 'S\\&P 500: $H$ (R/S) @{f8.hr} pentru randamente, @{f8.ha} pentru $|r_t|$; Nilul: $\\hat d = @{f8.nraw}$, dar $@{f8.nadj}$ după ruptura din 1898')],
       [T('a break or a regime change read as long memory', 'o ruptură sau o schimbare de regim interpretată drept memorie lungă'),
        T('one bandwidth $m$ only; long memory in returns confused with long memory in volatility', 'o singură lățime de bandă $m$; memoria lungă a randamentelor confundată cu memoria lungă a volatilității')])

review(9, ('Machine learning for time series', 'învățare automată pentru serii de timp'),
       [T('direct forecast $\\hat y_{T+h} = \\hat f_h(y_T, \\dots, y_{T-p+1}, \\mathbf z_{T+h})$; ridge $+\\lambda\\sum\\beta_j^2$, lasso $+\\lambda\\sum|\\beta_j|$', 'prognoza directă $\\hat y_{T+h} = \\hat f_h(y_T, \\dots, y_{T-p+1}, \\mathbf z_{T+h})$; ridge $+\\lambda\\sum\\beta_j^2$, lasso $+\\lambda\\sum|\\beta_j|$'),
        T('bias--variance $E(y_0 - \\hat f)^2 = \\text{bias}^2 + \\mathrm{Var}(\\hat f) + \\sigma^2$; walk-forward validation only', 'deplasare--varianță $E(y_0 - \\hat f)^2 = \\text{deplasare}^2 + \\mathrm{Var}(\\hat f) + \\sigma^2$; doar validare walk-forward')],
       [T('S\\&P 500, the same model: $R^2 = @{f9.kf}$ with random $k$-fold, $@{f9.wf}$ walk-forward: leakage', 'S\\&P 500, același model: $R^2 = @{f9.kf}$ cu $k$-fold aleator, $@{f9.wf}$ walk-forward: scurgere de informație'),
        T('sign of returns: majority class @{f9.base}\\%, boosting @{f9.gb}\\%; daily load: ridge MASE @{f9.ridge}, LSTM @{f9.lstm} \\refMfour', 'semnul randamentelor: clasa majoritară @{f9.base}\\%, boosting @{f9.gb}\\%; consumul zilnic: ridge MASE @{f9.ridge}, LSTM @{f9.lstm} \\refMfour')],
       [T('random cross-validation and scaling on the whole sample (look-ahead)', 'validarea încrucișată aleatoare și scalarea pe tot eșantionul (look-ahead)'),
        T('trees on trending levels (they cannot extrapolate); no simple benchmark in the comparison', 'arbori pe niveluri cu trend (nu pot extrapola); comparații fără un reper simplu')])

review(10, ('State space models, Kalman filter and Markov switching', 'spațiul stărilor, filtrul Kalman și modele Markov switching'),
       [T('$y_t = Z\\alpha_t + \\varepsilon_t$, $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$; $K_t = P_tZ\'F_t^{-1}$, $a_{t|t} = a_t + K_tv_t$ \\refKalman', '$y_t = Z\\alpha_t + \\varepsilon_t$, $\\alpha_{t+1} = T\\alpha_t + R\\eta_t$; $K_t = P_tZ\'F_t^{-1}$, $a_{t|t} = a_t + K_tv_t$ \\refKalman'),
        T('Markov switching: durations $1/(1 - p_{ii})$, ergodic $\\pi_1 = (1 - p_{22})/(2 - p_{11} - p_{22})$ \\refHamMS', 'Markov switching: durate $1/(1 - p_{ii})$, ergodic $\\pi_1 = (1 - p_{22})/(2 - p_{11} - p_{22})$ \\refHamMS')],
       [T('the Nile: $\\hat q = @{f10.q}$, steady-state gain $\\bar K = @{f10.K}$, the weight of exponential smoothing', 'Nilul: $\\hat q = @{f10.q}$, cîștigul de echilibru $\\bar K = @{f10.K}$, ponderea netezirii exponențiale'),
        T('US GDP: the regimes match the NBER dates in @{f10.con}\\% of quarters; Romanian GDP: stable regime @{f10.d2} quarters, volatile @{f10.d1}', 'PIB-ul SUA: regimurile coincid cu datările NBER în @{f10.con}\\% din trimestre; PIB-ul României: regimul stabil @{f10.d2} trimestre, cel volatil @{f10.d1}')],
       [T('smoothed probabilities used as if known in real time', 'probabilități netezite folosite ca și cum ar fi fost cunoscute în timp real'),
        T('durations written as $p_{ii}/(1 - p_{ii})$; the last value of the HP gap read as a real-time estimate', 'durate scrise $p_{ii}/(1 - p_{ii})$; ultima valoare a deviației HP interpretată ca estimare în timp real')])

D.recap(('Chapters 8--10', 'Capitolele 8--10'), [
    T('Long memory: hyperbolic ACF, $d$ between 0 and 1/2; check breaks and regimes before believing it', 'Memoria lungă: ACF hiperbolică, $d$ între 0 și 1/2; verificați rupturile și regimurile înainte de a o accepta'),
    T('Machine learning: the same supervised table, walk-forward validation, simple benchmarks', 'Învățarea automată: același tabel supervizat, validare walk-forward, repere simple'),
    T('State space: the Kalman filter for hidden levels and trends; Markov switching for regimes', 'Spațiul stărilor: filtrul Kalman pentru niveluri și trenduri ascunse; Markov switching pentru regimuri')])

# =============================================================================
# 5. CAPITOLELE 11--14 (STUDIU INDIVIDUAL)
# =============================================================================
D.section('Chapters 11--14: self-study', 'Capitolele 11--14: studiu individual')

SELF = T('Chapters 11--14 are for self-study (not examined unless the instructor decides otherwise)', 'Capitolele 11--14 sînt pentru studiu individual (nu se examinează, dacă titularul nu decide altfel)')


def selfstudy(k, title, ideas, links_to):
    D.frame(T(f'Chapter {k}: {title[0]} (self-study)', f'Capitolul {k}: {title[1]} (studiu individual)'), items(
        (T('\\textbf{Key ideas}', '\\textbf{Idei principale}'), ideas),
        (T('\\textbf{Link with the core of the course}', '\\textbf{Legătura cu nucleul cursului}'), links_to),
        SELF), 'footnotesize')


selfstudy(11, ('Foundation models for time series', 'foundation models pentru serii de timp'),
          [T('transformers pre-trained on many series: tokenisation and patching of the values', 'modele transformer pre-antrenate pe foarte multe serii: tokenizarea și împărțirea valorilor în segmente (patching)'),
           T('Chronos, TimesFM, Moirai, Lag-Llama: zero-shot forecasts and fine-tuning', 'Chronos, TimesFM, Moirai, Lag-Llama: prognoze zero-shot și fine-tuning'),
           T('a fair comparison: the same test sample, MASE, probabilistic scores, simple benchmarks', 'o comparație corectă: același eșantion de test, MASE, scoruri probabilistice, repere simple')],
          [T('a global model of many series (Chapter 9); judged like any forecast (Chapter 4)', 'un model global pe multe serii (Capitolul 9); evaluat ca orice prognoză (Capitolul 4)')])

selfstudy(12, ('Spectral analysis', 'analiză spectrală'),
          [T('the Fourier transform, the periodogram $I(\\lambda_j)$ and the spectral density', 'transformata Fourier, periodograma $I(\\lambda_j)$ și densitatea spectrală'),
           T('smoothed periodograms (Welch), filters (Hodrick--Prescott) and business cycles', 'periodograme netezite (Welch), filtre (Hodrick--Prescott) și ciclul economic'),
           T('coherence between two series; wavelets as a time--frequency tool', 'coerența dintre două serii; wavelets ca instrument timp--frecvență')],
          [T('the same information as the ACF, by frequency (Chapter 1); the spectral pole of long memory (Chapter 8)', 'aceeași informație ca ACF, pe frecvențe (Capitolul 1); polul spectral al memoriei lungi (Capitolul 8)')])

selfstudy(13, ('Speculative bubbles: LPPL models', 'bule speculative: modele LPPL'),
          [T('bubbles as super-exponential growth of prices', 'bulele ca o creștere superexponențială a prețurilor'),
           T('the LPPLS model: power law plus log-periodic oscillations; estimation and filter conditions', 'modelul LPPLS: o lege putere plus oscilații log-periodice; estimare și condiții de filtrare'),
           T('the LPPLS confidence indicator on historical bubbles and crashes', 'indicatorul de încredere LPPLS pe bule și crahuri istorice')],
          [T('explosive roots, the opposite of unit roots (Chapter 3); volatility before crashes (Chapter 5)', 'rădăcini explozive, opusul rădăcinilor unitare (Capitolul 3); volatilitatea dinaintea crahurilor (Capitolul 5)')])

selfstudy(14, ('Multivariate GARCH models', 'modele GARCH multivariate'),
          [T('time-varying covariances and correlations; the number of parameters grows fast', 'covarianțe și corelații variabile în timp; numărul de parametri crește foarte repede'),
           T('VEC, BEKK, CCC and DCC; estimation of DCC in two steps', 'VEC, BEKK, CCC și DCC; estimarea DCC în doi pași'),
           T('dynamic correlations, portfolio variance and VaR 1\\%', 'corelații dinamice, varianța portofoliului și VaR 1\\%')],
          [T('GARCH for each series (Chapter 5) and the multivariate view of the VAR (Chapter 6)', 'GARCH pentru fiecare serie (Capitolul 5) și perspectiva multivariată a VAR (Capitolul 6)')])

# =============================================================================
# 6. TRUSA DE INSTRUMENTE
# =============================================================================
D.section('The toolbox', 'Trusa de instrumente')

TB = '>{\\raggedright\\arraybackslash}'
TH = (T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Tool}', '\\textbf{Instrumentul}') + ' & '
      + T('\\textbf{Decision}', '\\textbf{Decizia}') + ' & \\textbf{' + T('Ch.', 'Cap.') + '}')
SPEC = TB + 'p{3.5cm}' + TB + 'p{3.5cm}' + TB + 'p{3.6cm}' + 'c'
D.frame(T('Toolbox (1/2): one series, the mean', 'Trusa de instrumente (1/2): o serie, media'), table(
    SPEC, TH,
    [T('Is the series stationary?', 'Este seria staționară?') + ' & ' + T('plot, ACF; ADF and KPSS', 'grafic, ACF; ADF și KPSS') + ' & ' + T('ADF rejects, KPSS does not: stationary', 'ADF respinge, KPSS nu: staționară') + ' & 1, 3',
     T('Trend: deterministic or stochastic?', 'Trend: determinist sau stochastic?') + ' & ' + T('ADF and KPSS with a trend', 'ADF și KPSS cu trend') + ' & ' + T('detrend, or difference', 'eliminarea trendului sau diferențierea') + ' & 3',
     T('Is there seasonality?', 'Există sezonalitate?') + ' & ' + T('seasonal plot, ACF at $s$, $2s$; OCSB, HEGY', 'graficul sezonier, ACF la $s$, $2s$; OCSB, HEGY') + ' & ' + T('$\\Delta_s$, or dummies and Fourier terms', '$\\Delta_s$ sau variabile dummy și termeni Fourier') + ' & 4',
     T('Which ARMA?', 'Ce model ARMA?') + ' & ' + T('ACF, PACF; AIC, BIC', 'ACF, PACF; AIC, BIC') + ' & ' + T('smallest BIC with white-noise residuals', 'cel mai mic BIC cu reziduuri zgomot alb') + ' & 2',
     T('Are the residuals adequate?', 'Sînt reziduurile adecvate?') + ' & ' + T('Ljung--Box on $m - k$ df; Jarque--Bera', 'Ljung--Box cu $m - k$ grade de libertate; Jarque--Bera') + ' & ' + T('$p > 0.05$: no autocorrelation left', '$p > 0{,}05$: nu mai rămîne autocorelație') + ' & 2',
     T('Which forecast is better?', 'Care prognoză este mai bună?') + ' & ' + T('test sample, rolling origins; MASE, DM', 'eșantion de test, origini mobile; MASE, DM') + ' & ' + T('DM $p < 0.05$; otherwise a tie', 'DM cu $p < 0{,}05$; altfel egalitate') + ' & 0, 4'],
    size='footnotesize'))

D.frame(T('Toolbox (2/2): variance, several series, extensions', 'Trusa de instrumente (2/2): varianță, mai multe serii, extensii'), table(
    SPEC, TH,
    [T('Does the variance change?', 'Se schimbă varianța?') + ' & ' + T('Ljung--Box on $r_t^2$, ARCH-LM', 'Ljung--Box pe $r_t^2$, ARCH-LM') + ' & ' + T('reject: GARCH(1,1)-$t$, GJR', 'respingem: GARCH(1,1)-$t$, GJR') + ' & 5',
     T('Does $x$ help to forecast $y$?', 'Ajută $x$ la prognoza lui $y$?') + ' & ' + T('VAR, Granger $F$ test', 'VAR, testul Granger $F$') + ' & ' + T('$p < 0.05$: Granger causality', '$p < 0{,}05$: cauzalitate Granger') + ' & 6',
     T('What does a shock do?', 'Ce efect are un șoc?') + ' & IRF, FEVD (Cholesky) & ' + T('bootstrap bands; try other orderings', 'benzi bootstrap; încercați alte ordini') + ' & 6',
     T('A long-run equilibrium?', 'Există un echilibru pe termen lung?') + ' & ' + T('Engle--Granger, Johansen', 'Engle--Granger, Johansen') + ' & ' + T('VECM: $\\beta$, $\\alpha$, half-life', 'VECM: $\\beta$, $\\alpha$, timpul de înjumătățire') + ' & 7',
     T('Long memory?', 'Memorie lungă?') + ' & ' + T('GPH, local Whittle, ARFIMA', 'GPH, Whittle local, ARFIMA') + ' & ' + T('several $m$; check breaks', 'mai multe valori $m$; verificați rupturile') + ' & 8',
     T('Many predictors, non-linear effects?', 'Mulți predictori, efecte neliniare?') + ' & ' + T('ridge, lasso, trees, boosting', 'ridge, lasso, arbori, boosting') + ' & ' + T('walk-forward against a benchmark', 'walk-forward față de un reper') + ' & 9',
     T('A hidden level or regime?', 'Un nivel sau un regim ascuns?') + ' & ' + T('Kalman filter, Markov switching', 'filtrul Kalman, Markov switching') + ' & ' + T('filtered for real time, smoothed for history', 'filtrat pentru timp real, netezit pentru istorie') + ' & 10'],
    size='footnotesize'))

D.frame(T('Conventions used throughout the course', 'Convențiile folosite în tot cursul'), items(
    (T('\\textbf{Transformations}: $100\\ln y_t$; growth $100\\Delta\\ln y_t$; annual rates $100\\Delta_{12}\\ln y_t$ (monthly), $100\\Delta_4\\ln y_t$ (quarterly)', '\\textbf{Transformările}: $100\\ln y_t$; creșterea $100\\Delta\\ln y_t$; ratele anuale $100\\Delta_{12}\\ln y_t$ (lunar), $100\\Delta_4\\ln y_t$ (trimestrial)'),
     [T('``DLOG(IPC,1,12)\'\' in EViews output is $\\Delta\\Delta_{12}\\ln$ IPC', '„DLOG(IPC,1,12)” în rezultatele EViews înseamnă $\\Delta\\Delta_{12}\\ln$ IPC')]),
    (T('\\textbf{Tests}: state $H_0$, the statistic, its distribution, the critical value or $p$-value, the decision at 5\\%', '\\textbf{Testele}: precizăm $H_0$, statistica, distribuția ei, valoarea critică sau $p$-valoarea, decizia la 5\\%'),
     [T('then one sentence on what the decision means for the series', 'apoi o frază despre semnificația deciziei pentru serie')]),
    (T('\\textbf{Model choice}: AIC, AICc and BIC only on the same sample and the same $d$, $D$', '\\textbf{Alegerea modelului}: AIC, AICc și BIC doar pe același eșantion și cu aceleași $d$, $D$'),
     [T('forecasts are judged out of sample, against the naive and seasonal naive methods', 'prognozele se judecă în afara eșantionului, față de metodele naivă și naivă sezonieră')]),
    T('\\textbf{Risk}: the level is the tail probability: VaR 1\\%, a positive loss, $\\mathrm{VaR}_\\alpha = -q_\\alpha$', '\\textbf{Riscul}: nivelul este probabilitatea cozii: VaR 1\\%, o pierdere pozitivă, $\\mathrm{VaR}_\\alpha = -q_\\alpha$')))

# =============================================================================
# 7. BOX-JENKINS PE INFLAȚIA DIN ROMÂNIA
# =============================================================================
D.section('Box--Jenkins from start to finish: Romanian inflation', 'Box--Jenkins de la un capăt la altul: inflația din România')

D.frame(T('The method in six steps', 'Metoda în șase pași'), two(enum(
    T('\\textbf{Plot} and transform: logarithm, $\\Delta$, $\\Delta_{12}$', '\\textbf{Graficul} și transformarea: logaritm, $\\Delta$, $\\Delta_{12}$'),
    T('\\textbf{Test}: ADF and KPSS for each transformation', '\\textbf{Testarea}: ADF și KPSS pentru fiecare transformare'),
    T('\\textbf{Identify}: ACF and PACF of the stationary series', '\\textbf{Identificarea}: ACF și PACF ale seriei staționare'),
    T('\\textbf{Estimate}: candidate SARIMA models by maximum likelihood; AICc and BIC', '\\textbf{Estimarea}: modele SARIMA candidate, prin verosimilitate maximă; AICc și BIC'),
    T('\\textbf{Check}: residuals (Ljung--Box, Jarque--Bera, outliers); back to step 3 if they fail', '\\textbf{Verificarea}: reziduurile (Ljung--Box, Jarque--Bera, valori extreme); înapoi la pasul 3 dacă eșuează'),
    T('\\textbf{Forecast and evaluate}: a test sample, benchmarks, then intervals', '\\textbf{Prognoza și evaluarea}: un eșantion de test, repere, apoi intervale')) + '\n' + items(
    T('Data: the HICP of Romania (Eurostat), @{bj.first} -- @{bj.last}, $T = @{bj.n}$ months; \\refBJ', 'Datele: IAPC al României (Eurostat), @{bj.first} -- @{bj.last}, $T = @{bj.n}$ de luni; \\refBJ')),
    ph('box', T('George E. P. Box (1919--2013)', 'George E. P. Box (1919--2013)'), h='0.42\\textheight'), '0.62', '0.34'), 'footnotesize')

chart(T('Step 1: the data', 'Pasul 1: datele'), 'tsa_ch15_bj_data', 'TSA_ch15_box_jenkins', [
    T('Top: HICP, 2015 = 100; bottom: monthly inflation $100\\Delta\\ln P_t$ (bars) and annual inflation $100\\Delta_{12}\\ln P_t$ (line)', 'Sus: IAPC, 2015 = 100; jos: inflația lunară $100\\Delta\\ln P_t$ (bare) și inflația anuală $100\\Delta_{12}\\ln P_t$ (linie)')],
    h='0.6\\textheight')

interp(('the data', 'datelor'), [
    (T('\\textbf{Trend}: the index rises almost every month; the logarithm is the natural scale', '\\textbf{Trend}: indicele crește aproape în fiecare lună; logaritmul este scala firească'),
     [T('monthly inflation averages @{bj.meanm}\\%; annual inflation ranges from $@{bj.mina}\\%$ (@{bj.minad}) to @{bj.maxa}\\% (@{bj.maxad})', 'inflația lunară are media @{bj.meanm}\\%; inflația anuală variază între $@{bj.mina}\\%$ (@{bj.minad}) și @{bj.maxa}\\% (@{bj.maxad})')]),
    (T('\\textbf{Seasonality}: January averages @{bj.mjan}\\%, June $@{bj.mjun}\\%$ (fresh food)', '\\textbf{Sezonalitatea}: ianuarie are în medie @{bj.mjan}\\%, iunie $@{bj.mjun}\\%$ (alimentele proaspete)'),
     [T('a seasonal pattern in $\\Delta\\ln P_t$: we will need $\\Delta_{12}$ or seasonal terms', 'un tipar sezonier în $\\Delta\\ln P_t$: vom avea nevoie de $\\Delta_{12}$ sau de termeni sezonieri')]),
    (T('\\textbf{Shocks}: tax changes move prices in one month', '\\textbf{Șocurile}: modificările de taxe mută prețurile într-o singură lună'),
     [T('July--August 2025: end of the electricity price caps and a higher VAT; annual inflation from @{bj.a2507}\\% to @{bj.a2508}\\%', 'iulie--august 2025: încetarea plafonării prețului la electricitate și creșterea TVA; inflația anuală de la @{bj.a2507}\\% la @{bj.a2508}\\%')])])

D.frame(T('Step 2: unit-root and stationarity tests', 'Pasul 2: teste de rădăcină unitară și de staționaritate'), table(
    'lccccc', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Terms}', '\\textbf{Termeni}') + ' & ADF $\\tau$ & $p$ & ' + T('KPSS $\\eta$ (5\\%: crit.)', 'KPSS $\\eta$ (5\\%: val. critică)') + ' & ' + T('\\textbf{Verdict}', '\\textbf{Verdict}'),
    ['$100\\ln P_t$ & ' + T('const., trend', 'const., trend') + ' & $@{ts.y.adf}$ & @{ts.y.p} & @{ts.y.kpss} (@{ts.y.kcv}) & ' + T('unit root', 'rădăcină unitară'),
     '$\\Delta\\ln P_t$ & ' + T('const.', 'const.') + ' & $@{ts.m.adf}$ & @{ts.m.p} & @{ts.m.kpss} (@{ts.m.kcv}) & ' + T('stationary, seasonal', 'staționară, sezonieră'),
     '$\\Delta_{12}\\ln P_t$ & ' + T('const.', 'const.') + ' & $@{ts.a.adf}$ & @{ts.a.p} & @{ts.a.kpss} (@{ts.a.kcv}) & ' + T('inconclusive', 'neconcludent'),
     '$z_t = \\Delta\\Delta_{12}\\ln P_t$ & ' + T('const.', 'const.') + ' & $@{ts.z.adf}$ & @{ts.z.p} & @{ts.z.kpss} (@{ts.z.kcv}) & ' + T('stationary', 'staționară')],
    size='footnotesize') + items(
    T('ADF: $H_0$ unit root, lags by AIC \\refDF; KPSS: $H_0$ stationarity \\refKPSS; 5\\% ADF critical values $@{ts.y.cv}$ (with trend), $@{ts.z.cv}$ (constant)', 'ADF: $H_0$ rădăcină unitară, decalaje după AIC \\refDF; KPSS: $H_0$ staționaritate \\refKPSS; valorile critice ADF la 5\\%: $@{ts.y.cv}$ (cu trend), $@{ts.z.cv}$ (cu constantă)'),
    T('Annual inflation: neither ADF nor KPSS rejects: an inconclusive result, as in Chapter 3', 'Inflația anuală: nici ADF, nici KPSS nu respinge: un rezultat neconcludent, ca în Capitolul 3'),
    T('Decision: $d = 1$, $D = 1$: we model $z_t$, both tests agree that it is stationary', 'Decizia: $d = 1$, $D = 1$: modelăm $z_t$, iar ambele teste arată că este staționară')) + ql('TSA_ch15_box_jenkins'), 'footnotesize')

chart(T('Step 3: identification from the correlogram of $z_t$', 'Pasul 3: identificarea din corelograma lui $z_t$'), 'tsa_ch15_bj_acf', 'TSA_ch15_box_jenkins', [
    T('ACF and PACF of $z_t = \\Delta\\Delta_{12}\\ln P_t$, lags 1--36, $T = @{ac.n}$; seasonal lags in red; band $\\pm @{ac.band}$', 'ACF și PACF pentru $z_t = \\Delta\\Delta_{12}\\ln P_t$, decalajele 1--36, $T = @{ac.n}$; decalajele sezoniere în roșu; banda $\\pm @{ac.band}$')],
    h='0.55\\textheight')

interp(('the correlogram', 'corelogramei'), [
    (T('\\textbf{Regular part}: $\\hat\\rho_1 = @{ac.r1}$, $\\hat\\rho_2 = @{ac.r2}$; PACF $\\hat\\phi_{11} = @{ac.p1}$, then small: an AR(1)', '\\textbf{Partea obișnuită}: $\\hat\\rho_1 = @{ac.r1}$, $\\hat\\rho_2 = @{ac.r2}$; PACF $\\hat\\phi_{11} = @{ac.p1}$, apoi mici: un AR(1)'),
     [T('an MA(1) is also possible: ACF and PACF both shrink after lag 1', 'este posibil și un MA(1): ACF și PACF scad amîndouă după decalajul 1')]),
    (T('\\textbf{Seasonal part}: ACF $@{ac.r12}$ at lag 12, then $@{ac.r24}$ at 24; PACF $@{ac.p12}$, $@{ac.p24}$, $@{ac.p36}$ at 12, 24, 36', '\\textbf{Partea sezonieră}: ACF $@{ac.r12}$ la decalajul 12, apoi $@{ac.r24}$ la 24; PACF $@{ac.p12}$, $@{ac.p24}$, $@{ac.p36}$ la 12, 24, 36'),
     [T('one seasonal ACF spike and a decaying seasonal PACF: a seasonal MA(1)', 'o singură valoare sezonieră mare în ACF și o PACF sezonieră care descrește: un MA(1) sezonier')]),
    T('First candidate: SARIMA$(1,1,0)(0,1,1)_{12}$ for $\\ln P_t$; the satellites at lags 11 and 13 are the product $\\rho_1\\rho_{12}$', 'Primul candidat: SARIMA$(1,1,0)(0,1,1)_{12}$ pentru $\\ln P_t$; sateliții de la decalajele 11 și 13 sînt produsul $\\rho_1\\rho_{12}$')])

GH = (T('\\textbf{Model}', '\\textbf{Modelul}') + ' & ' + T('param.', 'param.') + ' & AICc & BIC & ' + T('LB(12) $p$', 'LB(12) $p$') + ' & ' + T('LB(24) $p$', 'LB(24) $p$'))
D.frame(T('Step 4: estimating the candidates', 'Pasul 4: estimarea candidaților'), table(
    'lccccc', GH,
    ['$(1,1,0)(0,1,1)_{12}$, ' + T('identified', 'identificat') + ' & @{gr0.k} & @{gr0.aicc} & @{gr0.bic} & @{gr0.lb12} & @{gr0.lb24}']
    + [f'${{@{{gr{i}.m}}}}_{{12}}$ & @{{gr{i}.k}} & @{{gr{i}.aicc}} & @{{gr{i}.bic}} & @{{gr{i}.lb12}} & @{{gr{i}.lb24}}' for i in range(1, 5)],
    size='footnotesize') + items(
    T('All @{gr.n} SARIMA$(p,1,q)(P,1,Q)_{12}$ with $p, q \\le 2$, $P, Q \\le 1$, on the training sample to @{gr.trlast} ($T = @{gr.ntr}$); sorted by BIC', 'Toate cele @{gr.n} de modele SARIMA$(p,1,q)(P,1,Q)_{12}$ cu $p, q \\le 2$, $P, Q \\le 1$, pe eșantionul de antrenare pînă în @{gr.trlast} ($T = @{gr.ntr}$); ordonate după BIC'),
    T('param.: all estimated parameters, the variance included; Ljung--Box on the residuals with $m - k$ degrees of freedom, $k$ = the ARMA parameters', 'param.: toți parametrii estimați, inclusiv varianța; Ljung--Box pe reziduuri cu $m - k$ grade de libertate, $k$ = parametrii ARMA'),
    T('The identified model fails the residual check; the lowest BIC, $@{gr.best}$, fails at lag 12; the chosen model is $@{gr.chosen}_{12}$, the lowest BIC among the models that pass', 'Modelul identificat nu trece verificarea reziduurilor; cel mai mic BIC, $@{gr.best}$, eșuează la decalajul 12; modelul ales este $@{gr.chosen}_{12}$, cel mai mic BIC dintre modelele care trec')) + ql('TSA_ch15_box_jenkins'), 'footnotesize')

D.frame(T('Step 4: the chosen model', 'Pasul 4: modelul ales'), table(
    'lcccc', T('\\textbf{Parameter}', '\\textbf{Parametrul}') + ' & ' + T('\\textbf{Estimate}', '\\textbf{Estimarea}') + ' & SE & $z$ & $p$',
    ['$\\phi_1$ (ar.L1) & $@{dg.ar.L1}$ & $@{dg.se.ar.L1}$ & $@{dg.z.ar.L1}$ & @{dg.p.ar.L1}',
     '$\\phi_2$ (ar.L2) & $@{dg.ar.L2}$ & $@{dg.se.ar.L2}$ & $@{dg.z.ar.L2}$ & @{dg.p.ar.L2}',
     '$\\theta_1$ (ma.L1) & $@{dg.ma.L1}$ & $@{dg.se.ma.L1}$ & $@{dg.z.ma.L1}$ & @{dg.p.ma.L1}',
     '$\\Theta_1$ (ma.S.L12) & $@{dg.ma.S.L12}$ & $@{dg.se.ma.S.L12}$ & $@{dg.z.ma.S.L12}$ & @{dg.p.ma.S.L12}',
     '$\\sigma^2$ & $@{dg.s2}$ & & &'],
    size='footnotesize') + items(
    T('$(1 - \\phi_1L - \\phi_2L^2)\\,z_t = (1 + \\theta_1L)(1 + \\Theta_1L^{12})\\,\\varepsilon_t$, $z_t = \\Delta\\Delta_{12}\\,100\\ln P_t$', '$(1 - \\phi_1L - \\phi_2L^2)\\,z_t = (1 + \\theta_1L)(1 + \\Theta_1L^{12})\\,\\varepsilon_t$, $z_t = \\Delta\\Delta_{12}\\,100\\ln P_t$'),
    T('$\\hat\\Theta_1$ close to $-1$: the seasonal pattern is almost fixed, $\\Delta_{12}$ nearly cancels; seasonal dummies would be an alternative', '$\\hat\\Theta_1$ aproape de $-1$: tiparul sezonier este aproape fix, iar $\\Delta_{12}$ aproape se anulează; variabilele dummy sezoniere ar fi o alternativă'),
    T('The smallest AR root has modulus @{dg.arroot}: inflation shocks are very persistent, as in Chapters 2 and 8', 'Cea mai mică rădăcină AR are modulul @{dg.arroot}: șocurile inflației sînt foarte persistente, ca în Capitolele 2 și 8')) + ql('TSA_ch15_box_jenkins'), 'footnotesize')

chart(T('Step 5: checking the residuals', 'Pasul 5: verificarea reziduurilor'), 'tsa_ch15_bj_diag', 'TSA_ch15_box_jenkins', [
    T('Standardised residuals of the chosen model; ACF of the residuals of both models; QQ plot against $N(0, 1)$', 'Reziduurile standardizate ale modelului ales; ACF a reziduurilor ambelor modele; QQ plot față de $N(0, 1)$')],
    h='0.55\\textheight')

interp(('the residual checks', 'verificării reziduurilor'), [
    (T('\\textbf{No autocorrelation left}: Ljung--Box $Q(12) = @{dg.lb12}$ ($@{dg.lb12df}$ df, $p = @{dg.lb12p}$), $Q(24) = @{dg.lb24}$ ($p = @{dg.lb24p}$)', '\\textbf{Nu mai rămîne autocorelație}: Ljung--Box $Q(12) = @{dg.lb12}$ ($@{dg.lb12df}$ grade de libertate, $p = @{dg.lb12p}$), $Q(24) = @{dg.lb24}$ ($p = @{dg.lb24p}$)'),
     [T('the identified model had $p = @{di.lb12p}$ and @{di.lb24p}: the extra AR and MA terms reduce the autocorrelation at lags 6--7', 'modelul identificat avea $p = @{di.lb12p}$ și @{di.lb24p}: termenii AR și MA în plus reduc autocorelația de la decalajele 6--7')]),
    (T('\\textbf{Not Normal}: Jarque--Bera @{dg.jb}, kurtosis @{dg.kurt}', '\\textbf{Nu este Normal}: Jarque--Bera @{dg.jb}, coeficientul de boltire @{dg.kurt}'),
     [T('the two largest residuals: @{dg.big1} ($@{dg.big1v}\\sigma$, the VAT cut on food) and @{dg.big2} ($@{dg.big2v}\\sigma$, the VAT increase)', 'cele mai mari două reziduuri: @{dg.big1} ($@{dg.big1v}\\sigma$, reducerea TVA la alimente) și @{dg.big2} ($@{dg.big2v}\\sigma$, creșterea TVA)')]),
    T('Consequence: point forecasts are fine, Normal intervals are too narrow when taxes change; dummies for known tax changes would help', 'Consecința: prognozele punctuale sînt bune, dar intervalele Normale sînt prea înguste cînd se schimbă taxele; variabile dummy pentru modificările de taxe cunoscute ar ajuta')])

chart(T('Step 6: forecasts', 'Pasul 6: prognoze'), 'tsa_ch15_bj_forecast', 'TSA_ch15_box_jenkins', [
    T('Left: one-step forecasts of monthly inflation on the test sample (@{fc.tf} -- @{fc.tl}), parameters fixed at the training estimates; right: annual inflation forecasts to @{fc.f12d} with 95\\% intervals, model re-estimated on all data', 'Stînga: prognoze pe un pas ale inflației lunare pe eșantionul de test (@{fc.tf} -- @{fc.tl}), cu parametrii estimați pe eșantionul de antrenare; dreapta: prognoze ale inflației anuale pînă în @{fc.f12d}, cu intervale de 95\\%, modelul reestimat pe toate datele')],
    h='0.55\\textheight')

D.frame(T('Interpreting the forecasts', 'Interpretarea prognozelor'), table(
    'lccc', T('\\textbf{Monthly inflation, test sample}', '\\textbf{Inflația lunară, eșantion de test}') + ' & RMSE & MAE & MASE',
    [f'SARIMA$@{{gr.chosen}}_{{12}}$ & @{{fc.s.rmse}} & @{{fc.s.mae}} & @{{fc.s.mase}}',
     T('seasonal naive', 'naiv sezonier') + ' & @{fc.sn.rmse} & @{fc.sn.mae} & @{fc.sn.mase}',
     T('naive', 'naiv') + ' & @{fc.nv.rmse} & @{fc.nv.mae} & @{fc.nv.mase}'],
    size='footnotesize') + items(
    (T('SARIMA has the smallest errors, but Diebold--Mariano (HLN) does not reject equal accuracy: $@{fc.dmsn}$ ($p = @{fc.dmsnp}$) against the seasonal naive, $@{fc.dmnv}$ ($p = @{fc.dmnvp}$) against the naive method', 'SARIMA are cele mai mici erori, dar testul Diebold--Mariano (HLN) nu respinge acuratețea egală: $@{fc.dmsn}$ ($p = @{fc.dmsnp}$) față de naivul sezonier, $@{fc.dmnv}$ ($p = @{fc.dmnvp}$) față de metoda naivă'),
     [T('@{fc.ntest} months are few; the largest misses: @{fc.w1} (@{fc.w1a}\\% against @{fc.w1f}\\%) and @{fc.w2} (@{fc.w2a}\\% against @{fc.w2f}\\%): tax and energy decisions that no univariate model can foresee', '@{fc.ntest} luni sînt puține; cele mai mari erori: @{fc.w1} (@{fc.w1a}\\%, față de @{fc.w1f}\\%) și @{fc.w2} (@{fc.w2a}\\%, față de @{fc.w2f}\\%): decizii privind taxele și energia pe care niciun model univariat nu le poate anticipa')]),
    (T('Annual inflation: @{fc.last_a}\\% in @{bj.last}; forecast @{fc.a1}\\% for @{fc.f1d} [@{fc.a1_lo}, @{fc.a1_hi}], @{fc.a12}\\% for @{fc.f12d} [@{fc.a12_lo}, @{fc.a12_hi}]', 'Inflația anuală: @{fc.last_a}\\% în @{bj.last}; prognoza @{fc.a1}\\% pentru @{fc.f1d} [@{fc.a1_lo}, @{fc.a1_hi}], @{fc.a12}\\% pentru @{fc.f12d} [@{fc.a12_lo}, @{fc.a12_hi}]'),
     [T('the August 2025 VAT jump leaves the 12-month window in August 2026; the 12-month interval is wide: a near unit root in $\\Delta\\ln P_t$', 'saltul TVA din august 2025 iese din fereastra de 12 luni în august 2026; intervalul la 12 luni este larg: o rădăcină aproape unitară în $\\Delta\\ln P_t$')])) + ql('TSA_ch15_box_jenkins'), 'footnotesize')

D.recap(('Box--Jenkins on Romanian inflation', 'Box--Jenkins pe inflația din România'), [
    T('Plot, transform, test: $\\ln P_t$ needs $d = 1$ and $D = 1$', 'Grafic, transformare, testare: $\\ln P_t$ are nevoie de $d = 1$ și $D = 1$'),
    T('The correlogram suggests $(1,1,0)(0,1,1)_{12}$; the residual check rejects it; the chosen model is $@{gr.chosen}_{12}$', 'Corelograma sugerează $(1,1,0)(0,1,1)_{12}$; verificarea reziduurilor îl respinge; modelul ales este $@{gr.chosen}_{12}$'),
    T('Out of sample the gain over the seasonal naive method is not significant; tax shocks dominate the errors', 'În afara eșantionului, cîștigul față de metoda naivă sezonieră nu este semnificativ; șocurile fiscale domină erorile')])

# =============================================================================
# 8. EXAMENUL
# =============================================================================
D.section('The exam', 'Examenul')

D.frame(T('The written exam: format', 'Examenul scris: formatul'), two(items(
    (T('\\textbf{70\\% of the final grade}; written, 2 hours', '\\textbf{70\\% din nota finală}; scris, 2 ore'),
     [T('five subjects, all compulsory; 9 points for the subjects plus 1 point ex officio', 'cinci subiecte, toate obligatorii; 9 puncte pentru subiecte plus 1 punct din oficiu')]),
    (T('\\textbf{Three kinds of subjects}', '\\textbf{Trei tipuri de subiecte}'),
     [T('a short derivation: moments, stationarity, the ACF or the form of a process', 'o derivare scurtă: momente, staționaritate, ACF sau forma unui proces'),
      T('the interpretation of software output: estimates, tests, correlograms, VAR and VECM tables', 'interpretarea rezultatelor obținute cu software: estimări, teste, corelograme, tabele VAR și VECM'),
      T('model choice and a forecast computed by hand from the output', 'alegerea modelului și o prognoză calculată de mînă din rezultate')]),
    T('The critical values you need are printed in the output, as in the problems below', 'Valorile critice necesare sînt tipărite în rezultate, ca în problemele de mai jos')),
    ph('ins', T('The National Institute of Statistics, Bucharest: the source of many exam series', 'Institutul Național de Statistică, București: sursa multor serii de examen'), h='0.40\\textheight')))

D.frame(T('Content assessed', 'Conținutul evaluat'), items(
    (T('\\textbf{Chapters 0--10}: lectures and seminars', '\\textbf{Capitolele 0--10}: cursuri și seminarii'),
     [T('the definitions and formulas of the ``What you need today\'\' slides of every seminar', 'definițiile și formulele de pe slide-urile „Noțiuni necesare azi” ale fiecărui seminar'),
      T('the computations of Part A of the seminars, on paper', 'calculele din Partea A a seminariilor, pe hîrtie')]),
    (T('\\textbf{Reading output}: tables like those of Part B and of this chapter', '\\textbf{Citirea rezultatelor}: tabele ca în Partea B și ca în acest capitol'),
     [T('EViews or Python output: the same numbers under different names (AR(1), ar.L1; SAR(12), ar.S.L12)', 'rezultate EViews sau Python: aceleași cifre sub alte nume (AR(1), ar.L1; SAR(12), ar.S.L12)'),
      T('no code is written at the exam', 'la examen nu se scrie cod')]),
    (T('\\textbf{Core weight}: ARIMA and SARIMA, unit roots, VAR and Granger causality, cointegration and VECM, GARCH', '\\textbf{Ponderea principală}: ARIMA și SARIMA, rădăcini unitare, VAR și cauzalitate Granger, cointegrare și VECM, GARCH'),
     [SELF])))

D.frame(T('Grading criteria', 'Criterii de notare'), items(
    (T('\\textbf{The method}: the formula or the equation, with the numbers substituted', '\\textbf{Metoda}: formula sau ecuația, cu cifrele înlocuite'),
     [T('a correct method with an arithmetic slip receives most of the points', 'o metodă corectă cu o greșeală de calcul primește cea mai mare parte a punctajului')]),
    (T('\\textbf{The result}: the number with its unit (\\%, percentage points, months) and its sign', '\\textbf{Rezultatul}: cifra, cu unitatea de măsură (\\%, puncte procentuale, luni) și semnul ei'),
     [T('for a test: $H_0$, the statistic, the critical value or the $p$-value, the decision', 'pentru un test: $H_0$, statistica, valoarea critică sau $p$-valoarea, decizia')]),
    (T('\\textbf{The interpretation}: three to five sentences, statistical and economic', '\\textbf{Interpretarea}: trei pînă la cinci fraze, statistice și economice'),
     [T('a correct number with a wrong interpretation does not receive all the points', 'o cifră corectă cu o interpretare greșită nu primește tot punctajul'),
      T('say what the model cannot show (causality, structural breaks, the future of policy)', 'precizați ce nu poate arăta modelul (cauzalitatea, rupturile structurale, viitorul politicilor)')])))

D.frame(T('Typical problem types', 'Tipuri de probleme'), table(
    TB + 'p{4.6cm}' + TB + 'p{1.4cm}' + TB + 'p{5.2cm}',
    T('\\textbf{Type}', '\\textbf{Tipul}') + ' & ' + T('\\textbf{Chapters}', '\\textbf{Capitole}') + ' & ' + T('\\textbf{Example in this chapter}', '\\textbf{Exemplu în acest capitol}'),
    [T('derivation: moments, stationarity, differencing', 'derivare: momente, staționaritate, diferențiere') + ' & 1--3 & ' + T('Problem 1: trend plus AR(1) errors', 'Problema 1: trend plus erori AR(1)'),
     T('correlogram and unit-root output: identification', 'corelogramă și test de rădăcină unitară: identificare') + ' & 1--3 & ' + T('Problem 2: Romanian GDP', 'Problema 2: PIB-ul României'),
     T('SARIMA output and a forecast by hand', 'rezultate SARIMA și o prognoză de mînă') + ' & 2--4 & ' + T('Problem 3: Romanian inflation', 'Problema 3: inflația din România'),
     T('VAR and Granger output', 'rezultate VAR și Granger') + ' & 6 & ' + T('Problem 4: growth, inflation, ROBOR', 'Problema 4: creștere, inflație, ROBOR'),
     T('cointegration and VECM output', 'rezultate de cointegrare și VECM') + ' & 7 & ' + T('Problem 5: US yields', 'Problema 5: randamentele SUA'),
     T('GARCH output and VaR 1\\%', 'rezultate GARCH și VaR 1\\%') + ' & 5 & ' + T('Problem 6: the BET', 'Problema 6: BET'),
     T('forecast evaluation; state space and regimes', 'evaluarea prognozelor; spațiul stărilor și regimuri') + ' & 4, 8--10 & ' + T('Problems 7 and 8', 'Problemele 7 și 8')],
    size='footnotesize') + items(
    T('The eight problems below are review examples with full solutions; the exam has its own subjects', 'Cele opt probleme de mai jos sînt exemple de recapitulare, cu rezolvări complete; examenul are subiecte proprii')))


def problem(k, title, context, tasks, size='footnotesize', extra=''):
    ctx = '\n'.join(f'        \\item {c}' for c in context)
    body = ('\\begin{itemize}\n    \\item \\textbf{' + T('Context', 'Context') + '}\n    \\begin{itemize}\n' + ctx + '\n    \\end{itemize}\n'
            '    \\item ' + T('\\textbf{Tasks}', '\\textbf{Cerințe}') + '\n' + enum(*tasks) + '\n\\end{itemize}')
    D.frame(T(f'Problem {k}: {title[0]}', f'Problema {k}: {title[1]}'), extra + body, size)


def solution(k, steps, interpretation, size='footnotesize'):
    D.frame(T(f'Problem {k}: solution', f'Problema {k}: rezolvare'),
            enum(*steps) + '\n' + items((T('\\textbf{Interpretation}', '\\textbf{Interpretare}'), interpretation)), size)


# ---- Problema 1
problem(1, ('a trend with AR(1) errors', 'un trend cu erori AR(1)'),
        [T('$Y_t = 10 + 0.5t + u_t$, $u_t = 0.6u_{t-1} + \\varepsilon_t$, $\\varepsilon_t \\sim WN(0, 1)$', '$Y_t = 10 + 0.5t + u_t$, $u_t = 0.6u_{t-1} + \\varepsilon_t$, $\\varepsilon_t \\sim WN(0, 1)$'),
         T('At $T = 100$ we observe $Y_T = 62$', 'La $T = 100$ observăm $Y_T = 62$')],
        [T('Compute $E(Y_t)$ and $\\mathrm{Var}(Y_t)$ and say whether $Y_t$ is stationary.', 'Calculați $E(Y_t)$ și $\\mathrm{Var}(Y_t)$ și precizați dacă $Y_t$ este staționar.'),
         T('Show that $\\Delta Y_t$ is an ARMA(1,1) with $\\theta = -1$ and compute $\\rho_{\\Delta Y}(1)$.', 'Arătați că $\\Delta Y_t$ este un ARMA(1,1) cu $\\theta = -1$ și calculați $\\rho_{\\Delta Y}(1)$.'),
         T('Forecast $Y_{T+1}$ and $Y_{T+10}$ with the correct model and give the variance of the 10-step error.', 'Prognozați $Y_{T+1}$ și $Y_{T+10}$ cu modelul corect și dați varianța erorii pe 10 pași.'),
         T('Explain what goes wrong if the analyst differences $Y_t$.', 'Explicați ce nu este în regulă dacă analistul diferențiază $Y_t$.')])
solution(1, [
    T('$E(Y_t) = 10 + 0.5t$ depends on $t$: not stationary; $\\mathrm{Var}(Y_t) = 1/(1 - 0.36) = @{e1.var}$ is constant: \\textbf{trend-stationary}', '$E(Y_t) = 10 + 0.5t$ depinde de $t$: nestaționar; $\\mathrm{Var}(Y_t) = 1/(1 - 0.36) = @{e1.var}$ este constantă: \\textbf{staționar în jurul trendului}'),
    T('$\\Delta Y_t = 0.5 + \\Delta u_t$ and $(1 - 0.6L)\\Delta u_t = (1 - L)\\varepsilon_t$: ARMA(1,1), $\\theta = -1$; $\\gamma(0) = @{e1.vdu}$, $\\gamma(1) = @{e1.cdu}$, $\\rho(1) = @{e1.rdu}$', '$\\Delta Y_t = 0.5 + \\Delta u_t$ și $(1 - 0.6L)\\Delta u_t = (1 - L)\\varepsilon_t$: ARMA(1,1), $\\theta = -1$; $\\gamma(0) = @{e1.vdu}$, $\\gamma(1) = @{e1.cdu}$, $\\rho(1) = @{e1.rdu}$'),
    T('$u_T = 62 - 10 - 50 = @{e1.uT}$; $\\hat Y_{T+1} = 10 + 50.5 + 0.6 \\cdot 2 = @{e1.f1}$; $\\hat Y_{T+10} = 10 + 55 + 0.6^{10} \\cdot 2 = @{e1.f10}$; variance $(1 - 0.6^{20})/(1 - 0.36) = @{e1.v10}$', '$u_T = 62 - 10 - 50 = @{e1.uT}$; $\\hat Y_{T+1} = 10 + 50.5 + 0.6 \\cdot 2 = @{e1.f1}$; $\\hat Y_{T+10} = 10 + 55 + 0.6^{10} \\cdot 2 = @{e1.f10}$; varianța $(1 - 0.6^{20})/(1 - 0.36) = @{e1.v10}$'),
    T('The MA root is 1: $\\Delta Y_t$ is not invertible, $\\hat\\rho_1 < 0$ and an MA coefficient near $-1$ are the signs of over-differencing', 'Rădăcina MA este 1: $\\Delta Y_t$ nu este invertibil; $\\hat\\rho_1 < 0$ și un coeficient MA aproape de $-1$ sînt semnele supradiferențierii')],
    [T('A shock dies out: after 10 periods only $@{e1.p10}$ of the deviation is left, and the error variance stays bounded', 'Un șoc se stinge: după 10 perioade mai rămîne doar $@{e1.p10}$ din abatere, iar varianța erorii rămîne mărginită'),
     T('The treatment follows the type of trend: detrend a trend-stationary series, difference a series with a unit root (Chapter 3)', 'Tratamentul urmează tipul trendului: eliminăm trendul dintr-o serie staționară în jurul trendului, diferențiem o serie cu rădăcină unitară (Capitolul 3)')])

# ---- Problema 2
P2OUT = table('lcccccc',
              T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('Terms', 'Termeni') + ' & ADF $\\tau$ & $p$ & ' + T('crit. 1\\%; 5\\%; 10\\%', 'val. critice 1\\%; 5\\%; 10\\%') + ' & KPSS & ' + T('crit. 5\\%', 'val. critică 5\\%'),
              ['$100\\ln Y_t$ & c, t & $@{e2.l.adf}$ & @{e2.l.p} & $@{e2.l.cv1}$; $@{e2.l.cv5}$; $@{e2.l.cv10}$ & @{e2.l.kpss} & @{e2.l.kcv}',
               '$100\\Delta\\ln Y_t$ & c & $@{e2.g.adf}$ & @{e2.g.p} & $@{e2.g.cv1}$; $@{e2.g.cv5}$; $@{e2.g.cv10}$ & @{e2.g.kpss} & @{e2.g.kcv}'],
              size='scriptsize')
problem(2, ('identifying Romanian GDP', 'identificarea PIB-ului României'),
        [T('Romanian real GDP, seasonally adjusted (Eurostat), @{e2.first} -- @{e2.last}', 'PIB-ul real al României, ajustat sezonier (Eurostat), @{e2.first} -- @{e2.last}'),
         T('Growth $g_t = 100\\Delta\\ln Y_t$: ACF $@{e2.r1}$, $@{e2.r2}$; PACF $@{e2.p1}$, $@{e2.p2}$ (lags 1, 2; band $\\pm @{e2.band}$); Ljung--Box $Q(8) = @{e2.lb8}$, $p = @{e2.lb8p}$; $\\chi^2_{0.95}(8) = @{e2.chi8}$', 'Creșterea $g_t = 100\\Delta\\ln Y_t$: ACF $@{e2.r1}$; $@{e2.r2}$; PACF $@{e2.p1}$; $@{e2.p2}$ (decalajele 1, 2; banda $\\pm @{e2.band}$); Ljung--Box $Q(8) = @{e2.lb8}$, $p = @{e2.lb8p}$; $\\chi^2_{0,95}(8) = @{e2.chi8}$')],
        [T('Describe the ADF test and decide whether $\\ln Y_t$ and $g_t$ are stationary.', 'Descrieți testul ADF și decideți dacă $\\ln Y_t$ și $g_t$ sînt staționare.'),
         T('Say whether KPSS agrees.', 'Precizați dacă testul KPSS confirmă rezultatul.'),
         T('Identify the process with the Box--Jenkins method.', 'Identificați procesul cu metoda Box--Jenkins.'),
         T('The estimated drift is $@{e2.c}$ (SE $@{e2.cse}$): write the model and interpret the drift.', 'Deriva estimată este $@{e2.c}$ (SE $@{e2.cse}$): scrieți modelul și interpretați deriva.')],
        extra=P2OUT)
solution(2, [
    T('ADF: $\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum\\delta_j\\Delta y_{t-j} + \\varepsilon_t$, $H_0$: $\\gamma = 0$ (unit root); $\\ln Y_t$: $@{e2.l.adf} > @{e2.l.cv5}$: not rejected; $g_t$: $@{e2.g.adf} < @{e2.g.cv1}$: rejected', 'ADF: $\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum\\delta_j\\Delta y_{t-j} + \\varepsilon_t$, $H_0$: $\\gamma = 0$ (rădăcină unitară); $\\ln Y_t$: $@{e2.l.adf} > @{e2.l.cv5}$: nu se respinge; $g_t$: $@{e2.g.adf} < @{e2.g.cv1}$: se respinge'),
    T('KPSS ($H_0$: stationarity): $@{e2.l.kpss} > @{e2.l.kcv}$ rejects for the level, $@{e2.g.kpss} < @{e2.g.kcv}$ does not for growth: $\\ln Y_t \\sim I(1)$', 'KPSS ($H_0$: staționaritate): $@{e2.l.kpss} > @{e2.l.kcv}$ respinge pentru nivel, $@{e2.g.kpss} < @{e2.g.kcv}$ nu respinge pentru creștere: $\\ln Y_t \\sim I(1)$'),
    T('ACF and PACF of $g_t$ inside the band, $Q(8) = @{e2.lb8} < @{e2.chi8}$: growth is white noise around its mean', 'ACF și PACF ale lui $g_t$ sînt în bandă, $Q(8) = @{e2.lb8} < @{e2.chi8}$: creșterea este zgomot alb în jurul mediei'),
    T('ARIMA(0,1,0) with drift: $100\\ln Y_t = 100\\ln Y_{t-1} + @{e2.c} + \\varepsilon_t$, $\\hat\\sigma = @{e2.s}$; $t = @{e2.ct}$', 'ARIMA(0,1,0) cu derivă: $100\\ln Y_t = 100\\ln Y_{t-1} + @{e2.c} + \\varepsilon_t$, $\\hat\\sigma = @{e2.s}$; $t = @{e2.ct}$')],
    [T('The economy grows by about @{e2.c}\\% a quarter, about @{e2.ann}\\% a year; quarterly surprises are not predictable', 'Economia crește cu circa @{e2.c}\\% pe trimestru, adică aproximativ @{e2.ann}\\% pe an; surprizele trimestriale nu sînt previzibile'),
     T('A shock to the level is permanent: the 2009 and 2020 recessions moved the path of GDP for good', 'Un șoc asupra nivelului este permanent: recesiunile din 2009 și 2020 au mutat definitiv traiectoria PIB-ului')])

# ---- Problema 3
P3OUT = table('lcccc',
              T('\\textbf{Dependent: $z_t$}', '\\textbf{Variabila dependentă: $z_t$}') + ' & ' + T('Coefficient', 'Coeficient') + ' & SE & $z$ & $p$',
              ['AR(1) & $@{e3.phi}$ & $@{e3.sephi}$ & $@{e3.zphi}$ & @{e3.pphi}',
               'SAR(12) & $@{e3.Phi}$ & $@{e3.sePhi}$ & $@{e3.zPhi}$ & @{e3.pPhi}',
               'SIGMASQ & $@{e3.s2}$ & & &',
               T('Ljung--Box $Q(12)$', 'Ljung--Box $Q(12)$') + ' & $@{e3.lb12}$ & & & @{e3.lb12p}'],
              size='scriptsize')
problem(3, ('SARIMA output and a forecast by hand', 'rezultate SARIMA și o prognoză de mînă'),
        [T('$z_t = \\Delta\\Delta_{12}\\,100\\ln \\mathrm{HICP}_t$ (``DLOG(IPC,1,12)\'\'), Romania, @{e3.first} -- @{e3.last}, $T = @{e3.n}$', '$z_t = \\Delta\\Delta_{12}\\,100\\ln \\mathrm{IAPC}_t$ („DLOG(IPC,1,12)”), România, @{e3.first} -- @{e3.last}, $T = @{e3.n}$'),
         T('Last values: $z$ = $@{e3.zT}$ (@{e3.dT}), $@{e3.zT11}$ (@{e3.dT11}), $@{e3.zT12}$ (@{e3.dT12}); annual inflation in @{e3.dT}: @{e3.aT}\\%', 'Ultimele valori: $z$ = $@{e3.zT}$ (@{e3.dT}), $@{e3.zT11}$ (@{e3.dT11}), $@{e3.zT12}$ (@{e3.dT12}); inflația anuală în @{e3.dT}: @{e3.aT}\\%')],
        [T('Write the equation of the model and interpret the coefficients.', 'Scrieți ecuația modelului și interpretați coeficienții.'),
         T('Test whether the residuals are white noise.', 'Testați dacă reziduurile sînt zgomot alb.'),
         T('Forecast $z$ and the annual inflation rate for @{e3.next}.', 'Prognozați $z$ și rata anuală a inflației pentru @{e3.next}.')],
        extra=P3OUT)
solution(3, [
    T('$(1 - \\phi L)(1 - \\Phi L^{12})z_t = \\varepsilon_t$, so $z_t = \\phi z_{t-1} + \\Phi z_{t-12} - \\phi\\Phi z_{t-13} + \\varepsilon_t$', '$(1 - \\phi L)(1 - \\Phi L^{12})z_t = \\varepsilon_t$, deci $z_t = \\phi z_{t-1} + \\Phi z_{t-12} - \\phi\\Phi z_{t-13} + \\varepsilon_t$'),
    T('$\\hat\\phi = @{e3.phi}$: a monthly surprise continues one month later; $\\hat\\Phi = @{e3.Phi} < 0$: a surprise in a month is partly reversed in the same month a year later; both significant; inverted AR roots @{e3.inv} and @{e3.invs} $< 1$: stationary', '$\\hat\\phi = @{e3.phi}$: o surpriză lunară continuă și luna următoare; $\\hat\\Phi = @{e3.Phi} < 0$: o surpriză dintr-o lună se inversează parțial în aceeași lună a anului următor; ambele semnificative; rădăcinile AR inversate @{e3.inv} și @{e3.invs} $< 1$: staționar'),
    T('Ljung--Box $Q(12) = @{e3.lb12}$ on $12 - 2 = 10$ df, $p = @{e3.lb12p}$: white noise is rejected at 5\\%; the model leaves some autocorrelation (compare Step 5 above)', 'Ljung--Box $Q(12) = @{e3.lb12}$ cu $12 - 2 = 10$ grade de libertate, $p = @{e3.lb12p}$: ipoteza de zgomot alb se respinge la 5\\%; modelul lasă o parte din autocorelație (comparați cu Pasul 5 de mai sus)'),
    T('$\\hat z_{T+1} = @{e3.phi}(@{e3.zT}) + (@{e3.Phi})(@{e3.zT11}) - (@{e3.pP})(@{e3.zT12}) = @{e3.t1} + (@{e3.t2}) + @{e3.t3} = @{e3.znext}$', '$\\hat z_{T+1} = @{e3.phi}(@{e3.zT}) + (@{e3.Phi})(@{e3.zT11}) - (@{e3.pP})(@{e3.zT12}) = @{e3.t1} + (@{e3.t2}) + @{e3.t3} = @{e3.znext}$'),
    T('Annual inflation: $\\Delta_{12}\\ln P_{T+1} = \\Delta_{12}\\ln P_T + z_{T+1} = @{e3.aT} + (@{e3.znext}) = @{e3.anext}\\%$', 'Inflația anuală: $\\Delta_{12}\\ln P_{T+1} = \\Delta_{12}\\ln P_T + z_{T+1} = @{e3.aT} + (@{e3.znext}) = @{e3.anext}\\%$')],
    [T('Inflation is expected to keep falling in @{e3.next} as last year\'s tax shock leaves the 12-month window', 'Se așteaptă ca inflația să scadă în continuare în @{e3.next}, pe măsură ce șocul fiscal de anul trecut iese din fereastra de 12 luni'),
     T('Residual autocorrelation: a richer model (as in Step 4) would change the forecast slightly; a 95\\% interval needs $\\hat\\sigma$', 'Autocorelația reziduală: un model mai bogat (ca la Pasul 4) ar schimba puțin prognoza; un interval de 95\\% are nevoie de $\\hat\\sigma$')], size='scriptsize')

# ---- Problema 4
P4OUT = (table('lccc', T('\\textbf{ROBOR equation}', '\\textbf{Ecuația ROBOR}') + ' & ' + T('coef. [$t$]', 'coef. [$t$]') + ' & & ' + T('coef. [$t$]', 'coef. [$t$]'),
               ['$g_{t-1}$ & $@{e4.b.L1.g}$ [$@{e4.t.L1.g}$] & $g_{t-2}$ & $@{e4.b.L2.g}$ [$@{e4.t.L2.g}$]',
                '$\\pi_{t-1}$ & $@{e4.b.L1.pi}$ [$@{e4.t.L1.pi}$] & $\\pi_{t-2}$ & $@{e4.b.L2.pi}$ [$@{e4.t.L2.pi}$]',
                '$i_{t-1}$ & $@{e4.b.L1.i}$ [$@{e4.t.L1.i}$] & $i_{t-2}$ & $@{e4.b.L2.i}$ [$@{e4.t.L2.i}$]',
                T('constant', 'constanta') + ' & $@{e4.b.const}$ [$@{e4.t.const}$] & $R^2$ & @{e4.r2}'], size='scriptsize')
         + table('lcc', T('\\textbf{Null hypothesis}', '\\textbf{Ipoteza nulă}') + ' & $F$ & $p$',
                 [T('$g$ does not Granger-cause $i$', '$g$ nu cauzează Granger $i$') + ' & @{e4.F.gi} & @{e4.p.gi}',
                  T('$\\pi$ does not Granger-cause $i$', '$\\pi$ nu cauzează Granger $i$') + ' & @{e4.F.pii} & @{e4.p.pii}',
                  T('$i$ does not Granger-cause $\\pi$', '$i$ nu cauzează Granger $\\pi$') + ' & @{e4.F.ipi} & @{e4.p.ipi}'], size='scriptsize'))
problem(4, ('VAR output and Granger causality', 'rezultate VAR și cauzalitate Granger'),
        [T('VAR(2) for Romania, @{e4.first} -- @{e4.last} ($T = @{e4.n}$): GDP growth $g$, annual inflation $\\pi$, ROBOR 3M $i$', 'VAR(2) pentru România, @{e4.first} -- @{e4.last} ($T = @{e4.n}$): creșterea PIB $g$, inflația anuală $\\pi$, ROBOR 3M $i$')],
        [T('Describe Granger causality and interpret the table.', 'Descrieți cauzalitatea Granger și interpretați tabelul.'),
         T('Write the ROBOR equation and quantify the long-run effect of a permanent 1 pp rise in inflation on ROBOR.', 'Scrieți ecuația ROBOR și cuantificați efectul pe termen lung al unei creșteri permanente a inflației cu 1 pp asupra ROBOR.'),
         T('Recompute the first $F$ from $RSS_R = @{e4.rssr}$ and $RSS_U = @{e4.rssu}$.', 'Recalculați primul $F$ din $RSS_R = @{e4.rssr}$ și $RSS_U = @{e4.rssu}$.')],
        size='scriptsize', extra=P4OUT)
solution(4, [
    T('$x$ Granger-causes $y$ if the lags of $x$ improve the forecast of $y$ given the lags of $y$ (joint $F$ test) \\refGranger; growth and inflation help to forecast ROBOR ($p =$ @{e4.p.gi} and @{e4.p.pii}); ROBOR does not help to forecast inflation ($p = @{e4.p.ipi}$)', '$x$ cauzează Granger $y$ dacă decalajele lui $x$ îmbunătățesc prognoza lui $y$, dată fiind istoria lui $y$ (test $F$ comun) \\refGranger; creșterea și inflația ajută la prognoza ROBOR ($p =$ @{e4.p.gi} și @{e4.p.pii}); ROBOR nu ajută la prognoza inflației ($p = @{e4.p.ipi}$)'),
    T('$i_t = @{e4.b.const} + @{e4.b.L1.g}g_{t-1} + @{e4.b.L2.g}g_{t-2} + @{e4.b.L1.pi}\\pi_{t-1} + (@{e4.b.L2.pi})\\pi_{t-2} + @{e4.b.L1.i}i_{t-1} + (@{e4.b.L2.i})i_{t-2} + u_t$', '$i_t = @{e4.b.const} + @{e4.b.L1.g}g_{t-1} + @{e4.b.L2.g}g_{t-2} + @{e4.b.L1.pi}\\pi_{t-1} + (@{e4.b.L2.pi})\\pi_{t-2} + @{e4.b.L1.i}i_{t-1} + (@{e4.b.L2.i})i_{t-2} + u_t$'),
    T('Long run ($i_t = i_{t-1}$, $\\pi_t = \\pi_{t-1}$): $\\partial i/\\partial\\pi = (@{e4.b.L1.pi} + (@{e4.b.L2.pi}))/(1 - @{e4.b.L1.i} - (@{e4.b.L2.i})) = @{e4.num}/@{e4.den} = @{e4.lr}$ pp', 'Pe termen lung ($i_t = i_{t-1}$, $\\pi_t = \\pi_{t-1}$): $\\partial i/\\partial\\pi = (@{e4.b.L1.pi} + (@{e4.b.L2.pi}))/(1 - @{e4.b.L1.i} - (@{e4.b.L2.i})) = @{e4.num}/@{e4.den} = @{e4.lr}$ pp'),
    T('$F = \\frac{(@{e4.rssr} - @{e4.rssu})/2}{@{e4.rssu}/@{e4.df2}} = @{e4.F} > F_{0.95}(2, @{e4.df2}) = @{e4.crit}$: $H_0$ rejected', '$F = \\frac{(@{e4.rssr} - @{e4.rssu})/2}{@{e4.rssu}/@{e4.df2}} = @{e4.F} > F_{0,95}(2, @{e4.df2}) = @{e4.crit}$: $H_0$ se respinge')],
    [T('The money market rate follows the economy and inflation, as a central bank reaction would imply; the reverse link is not visible in quarterly data', 'Dobînda interbancară urmează economia și inflația, așa cum ar implica reacția unei bănci centrale; legătura inversă nu se vede în datele trimestriale'),
     T('The long-run effect divides by $@{e4.den}$, a number close to zero: it is imprecise; Granger causality is predictability, not proof of a policy effect', 'Efectul pe termen lung împarte la $@{e4.den}$, un număr apropiat de zero: este imprecis; cauzalitatea Granger înseamnă predictibilitate, nu dovada unui efect de politică')], size='scriptsize')

# ---- Problema 5
P5OUT = (table('lcccc', T('\\textbf{Johansen trace test}', '\\textbf{Testul urmei Johansen}') + ' & $r = 0$ & $r \\le 1$ & $r \\le 2$ &',
               [T('statistic', 'statistica') + ' & @{e5.tr0} & @{e5.tr1} & @{e5.tr2} &', T('5\\% critical value', 'valoarea critică 5\\%') + ' & @{e5.cv0} & @{e5.cv1} & @{e5.cv2} &'], size='scriptsize')
         + table('lccc', T('\\textbf{Adjustment} $\\alpha$ [$t$]', '\\textbf{Ajustarea} $\\alpha$ [$t$]') + ' & $\\Delta y^{(1)}$ & $\\Delta y^{(5)}$ & $\\Delta y^{(10)}$',
                 ['EC1 & $@{e5.a11}$ [$@{e5.t11}$] & $@{e5.a51}$ [$@{e5.t51}$] & $@{e5.a101}$ [$@{e5.t101}$]',
                  'EC2 & $@{e5.a12}$ [$@{e5.t12}$] & $@{e5.a52}$ [$@{e5.t52}$] & $@{e5.a102}$ [$@{e5.t102}$]'], size='scriptsize'))
problem(5, ('cointegration and VECM output', 'rezultate de cointegrare și VECM'),
        [T('US Treasury yields at 1, 5 and 10 years (monthly, FRED), @{e5.first} -- @{e5.last}, VECM with two lagged differences', 'Randamentele titlurilor de stat ale SUA la 1, 5 și 10 ani (lunar, FRED), @{e5.first} -- @{e5.last}, VECM cu două diferențe decalate'),
         T('Cointegrating equations: EC1 $= y^{(1)}_{t-1} - @{e5.b13}\\,y^{(10)}_{t-1} + @{e5.c1}$; EC2 $= y^{(5)}_{t-1} - @{e5.b23}\\,y^{(10)}_{t-1} + @{e5.c2}$', 'Ecuațiile de cointegrare: EC1 $= y^{(1)}_{t-1} - @{e5.b13}\\,y^{(10)}_{t-1} + @{e5.c1}$; EC2 $= y^{(5)}_{t-1} - @{e5.b23}\\,y^{(10)}_{t-1} + @{e5.c2}$')],
        [T('Define cointegration and decide the rank.', 'Definiți cointegrarea și decideți rangul.'),
         T('Write the error-correction part of the equation for $\\Delta y^{(10)}_t$ and say which yields adjust.', 'Scrieți partea de corecție a erorii din ecuația pentru $\\Delta y^{(10)}_t$ și precizați care randamente se ajustează.'),
         T('Interpret the long-run relations economically.', 'Interpretați economic relațiile pe termen lung.')],
        size='scriptsize', extra=P5OUT)
solution(5, [
    T('$I(1)$ series are cointegrated if a linear combination is $I(0)$ \\refEG; trace: $@{e5.tr0} > @{e5.cv0}$, $@{e5.tr1} > @{e5.cv1}$, $@{e5.tr2} < @{e5.cv2}$: rank $r = 2$, one common stochastic trend \\refJoh', 'Seriile $I(1)$ sînt cointegrate dacă o combinație liniară a lor este $I(0)$ \\refEG; urma: $@{e5.tr0} > @{e5.cv0}$, $@{e5.tr1} > @{e5.cv1}$, $@{e5.tr2} < @{e5.cv2}$: rangul $r = 2$, un singur trend stochastic comun \\refJoh'),
    T('$\\Delta y^{(10)}_t = @{e5.a101}\\,\\mathrm{EC1}_{t-1} + (@{e5.a102})\\,\\mathrm{EC2}_{t-1} + \\dots$; on EC1 the 5- and 10-year yields have $|t| > 2$, the 1-year yield $|t| < 1$ on both relations: the short rate does not adjust (weakly exogenous)', '$\\Delta y^{(10)}_t = @{e5.a101}\\,\\mathrm{EC1}_{t-1} + (@{e5.a102})\\,\\mathrm{EC2}_{t-1} + \\dots$; pe EC1, randamentele la 5 și 10 ani au $|t| > 2$, iar cel la 1 an are $|t| < 1$ pe ambele relații: dobînda scurtă nu se ajustează (exogenitate slabă)'),
    T('$\\beta \\approx 1$: the spreads $y^{(1)} - y^{(10)}$ and $y^{(5)} - y^{(10)}$ are stationary, as the expectations hypothesis implies; half-lives of the equilibrium errors @{e5.h1} and @{e5.h2} months', '$\\beta \\approx 1$: spread-urile $y^{(1)} - y^{(10)}$ și $y^{(5)} - y^{(10)}$ sînt staționare, cum implică ipoteza așteptărilor; timpii de înjumătățire ai erorilor de echilibru: @{e5.h1} și @{e5.h2} luni')],
    [T('The three yields share one trend (the level of rates); the curve can bend, but spreads return to their means', 'Cele trei randamente au un singur trend comun (nivelul dobînzilor); curba se poate deforma, dar spread-urile revin la mediile lor'),
     T('The long rates do the adjusting: markets move the long end toward the policy-driven short end', 'Dobînzile lungi se ajustează: piețele mișcă partea lungă a curbei spre partea scurtă, determinată de politica monetară')], size='scriptsize')

# ---- Problema 6
P6OUT = table('lccccc', T('\\textbf{BET}', '\\textbf{BET}') + ' & $\\mu$ & $\\omega$ & $\\alpha$ & $\\beta$ & $\\nu$',
              [T('estimate', 'estimare') + ' & $@{e6.mu}$ & $@{e6.om}$ & $@{e6.al}$ & $@{e6.be}$ & $@{e6.nu}$',
               'SE & $@{e6.se.mu}$ & $@{e6.se.om}$ & $@{e6.se.al}$ & $@{e6.se.be}$ & $@{e6.se.nu}$'], size='scriptsize')
problem(6, ('GARCH output and VaR 1\\%', 'rezultate GARCH și VaR 1\\%'),
        [T('GARCH(1,1) with Student-$t$ innovations for daily BET log returns in \\%, since 2000 ($n = @{e6.n}$)', 'GARCH(1,1) cu inovații Student-$t$ pentru randamentele logaritmice zilnice ale BET în \\%, din 2000 ($n = @{e6.n}$)'),
         T('Today: $\\sigma_t^2 = 1.0$ and $\\varepsilon_t = -3.0$; $t_5^{-1}(0.01) = @{e6.tq}$; about 250 trading days a year', 'Azi: $\\sigma_t^2 = 1{,}0$ și $\\varepsilon_t = -3{,}0$; $t_5^{-1}(0{,}01) = @{e6.tq}$; circa 250 de zile de tranzacționare pe an')],
        [T('Compute the persistence, the half-life and the long-run annual volatility.', 'Calculați persistența, timpul de înjumătățire și volatilitatea anuală pe termen lung.'),
         T('Compute $\\sigma_{t+1}^2$.', 'Calculați $\\sigma_{t+1}^2$.'),
         T('Compute the one-day VaR 1\\% with the standardised $t_5$ quantile and compare it with the Normal one.', 'Calculați VaR 1\\% pe o zi cu cuantila $t_5$ standardizată și comparați-l cu cel Normal.')],
        extra=P6OUT)
solution(6, [
    T('$\\alpha + \\beta = @{e6.pers}$; $h_{1/2} = \\ln 0.5/\\ln @{e6.pers} = @{e6.hl}$ days; $\\bar\\sigma^2 = @{e6.om}/(1 - @{e6.pers}) = @{e6.lv}$, $\\sqrt{250 \\times @{e6.lv}} = @{e6.ann}\\%$ a year', '$\\alpha + \\beta = @{e6.pers}$; $h_{1/2} = \\ln 0{,}5/\\ln @{e6.pers} = @{e6.hl}$ de zile; $\\bar\\sigma^2 = @{e6.om}/(1 - @{e6.pers}) = @{e6.lv}$, $\\sqrt{250 \\times @{e6.lv}} = @{e6.ann}\\%$ pe an'),
    T('$\\sigma_{t+1}^2 = @{e6.om} + @{e6.al} \\times 9 + @{e6.be} \\times 1.0 = @{e6.om} + @{e6.a9} + @{e6.be} = @{e6.s2n}$, $\\sigma_{t+1} = @{e6.sn}\\%$', '$\\sigma_{t+1}^2 = @{e6.om} + @{e6.al} \\times 9 + @{e6.be} \\times 1{,}0 = @{e6.om} + @{e6.a9} + @{e6.be} = @{e6.s2n}$, $\\sigma_{t+1} = @{e6.sn}\\%$'),
    T('$q_{0.01}(z) = @{e6.tq} \\times \\sqrt{3/5} = @{e6.tq} \\times @{e6.sc} = @{e6.q}$; VaR 1\\% $= -(@{e6.mu} + @{e6.sn} \\times (@{e6.q})) = @{e6.var}\\%$; Normal: $@{e6.varn}\\%$', '$q_{0,01}(z) = @{e6.tq} \\times \\sqrt{3/5} = @{e6.tq} \\times @{e6.sc} = @{e6.q}$; VaR 1\\% $= -(@{e6.mu} + @{e6.sn} \\times (@{e6.q})) = @{e6.var}\\%$; Normal: $@{e6.varn}\\%$')],
    [T('Persistence close to 1: a shock to volatility halves only after about @{e6.hl} trading days', 'Persistență aproape de 1: un șoc al volatilității se înjumătățește abia după circa @{e6.hl} de zile de tranzacționare'),
     T('The long-run volatility (@{e6.ann}\\%) is far above the sample one (@{e6.vs}\\%): when $\\alpha + \\beta \\approx 1$, $\\omega/(1 - \\alpha - \\beta)$ is very imprecise', 'Volatilitatea pe termen lung (@{e6.ann}\\%) este mult peste cea de selecție (@{e6.vs}\\%): cînd $\\alpha + \\beta \\approx 1$, raportul $\\omega/(1 - \\alpha - \\beta)$ este foarte imprecis'),
     T('Heavy tails raise VaR 1\\% above the Normal value: a loss larger than @{e6.var}\\% is expected on one day in 100', 'Cozile groase ridică VaR 1\\% peste valoarea Normală: o pierdere mai mare de @{e6.var}\\% este așteptată într-o zi din 100')])

# ---- Problema 7
P7OUT = table('lcccccc', T('\\textbf{Test month}', '\\textbf{Luna de test}') + ' & 1 & 2 & 3 & 4 & 5 & 6',
              [T('errors, SARIMA', 'erori, SARIMA') + ' & ' + ' & '.join(f'${x}$' for x in EA),
               T('errors, seasonal naive', 'erori, naiv sezonier') + ' & ' + ' & '.join(f'${x}$' for x in EB)], size='scriptsize')
problem(7, ('evaluating forecasts', 'evaluarea prognozelor'),
        [T('Monthly inflation forecasts on a test sample of six months; the in-sample MAE of the seasonal naive method is 0.6', 'Prognoze ale inflației lunare pe un eșantion de test de șase luni; MAE în eșantion al metodei naive sezoniere este 0,6'),
         T('A colleague reports $R^2 = @{e7.kf}$ for a boosting model of daily S\\&P 500 returns, with random 5-fold cross-validation', 'Un coleg raportează $R^2 = @{e7.kf}$ pentru un model boosting al randamentelor zilnice S\\&P 500, cu validare încrucișată aleatoare în 5 grupuri')],
        [T('Compute RMSE, MAE and MASE for both methods.', 'Calculați RMSE, MAE și MASE pentru ambele metode.'),
         T('Compute the Diebold--Mariano statistic on squared errors, $\\bar d/(s_d/\\sqrt{n})$, and decide at 5\\% ($t_{0.975}(5) = @{e7.crit}$).', 'Calculați statistica Diebold--Mariano pe erorile pătratice, $\\bar d/(s_d/\\sqrt{n})$, și decideți la 5\\% ($t_{0,975}(5) = @{e7.crit}$).'),
         T('Comment on the colleague\'s $R^2$.', 'Comentați valoarea $R^2$ raportată de coleg.')],
        extra=P7OUT)
solution(7, [
    T('SARIMA: RMSE @{e7.rA}, MAE @{e7.mA}, MASE @{e7.qA}; seasonal naive: RMSE @{e7.rB}, MAE @{e7.mB}, MASE @{e7.qB}', 'SARIMA: RMSE @{e7.rA}, MAE @{e7.mA}, MASE @{e7.qA}; naiv sezonier: RMSE @{e7.rB}, MAE @{e7.mB}, MASE @{e7.qB}'),
    T('$d_t = e_{A,t}^2 - e_{B,t}^2$: $@{e7.d1}$, $@{e7.d2}$, $@{e7.d3}$, $@{e7.d4}$, $@{e7.d5}$, $@{e7.d6}$; $\\bar d = @{e7.db}$, $s_d = @{e7.sd}$; DM $= @{e7.dm}$, $|@{e7.dm}| < @{e7.crit}$: equal accuracy is not rejected', '$d_t = e_{A,t}^2 - e_{B,t}^2$: $@{e7.d1}$; $@{e7.d2}$; $@{e7.d3}$; $@{e7.d4}$; $@{e7.d5}$; $@{e7.d6}$; $\\bar d = @{e7.db}$, $s_d = @{e7.sd}$; DM $= @{e7.dm}$, $|@{e7.dm}| < @{e7.crit}$: acuratețea egală nu se respinge'),
    T('Random folds put future days in the training set: leakage; walk-forward validation gives $R^2 = @{e7.wf}$ for the same model (Chapter 9)', 'Grupurile aleatoare pun zile din viitor în setul de antrenare: scurgere de informație; validarea walk-forward dă $R^2 = @{e7.wf}$ pentru același model (Capitolul 9)')],
    [T('MASE below 1: SARIMA beats the seasonal naive method on average, but six months cannot show that the gain is real', 'MASE sub 1: SARIMA bate în medie metoda naivă sezonieră, dar șase luni nu pot arăta că acest cîștig este real'),
     T('Only out-of-sample, time-ordered evaluation counts as evidence of forecasting skill', 'Doar evaluarea în afara eșantionului, în ordinea timpului, este o dovadă a capacității de prognoză')])

# ---- Problema 8
problem(8, ('state space, regimes and long memory', 'spațiul stărilor, regimuri și memorie lungă'),
        [T('Local level model: $\\sigma^2_\\varepsilon = 4$, $\\sigma^2_\\eta = 1$; prediction $a_t = 10$, $P_t = 2$; observation $y_t = 13$', 'Modelul local level: $\\sigma^2_\\varepsilon = 4$, $\\sigma^2_\\eta = 1$; predicția $a_t = 10$, $P_t = 2$; observația $y_t = 13$'),
         T('Romanian GDP growth, two regimes: $p_{11} = @{e8.p11}$ (volatile), $p_{22} = @{e8.p22}$ (stable) (Chapter 10)', 'Creșterea PIB-ului României, două regimuri: $p_{11} = @{e8.p11}$ (volatil), $p_{22} = @{e8.p22}$ (stabil) (Capitolul 10)'),
         T('Romanian monthly inflation: ARFIMA(0,$d$,0) with $\\hat d = @{e8.d}$ (SE @{e8.dse}) (Chapter 8)', 'Inflația lunară din România: ARFIMA(0,$d$,0) cu $\\hat d = @{e8.d}$ (SE @{e8.dse}) (Capitolul 8)')],
        [T('Run one Kalman update and prediction; compute the steady-state gain and the equivalent SES weight.', 'Aplicați un pas de actualizare și de predicție Kalman; calculați cîștigul de echilibru și ponderea SES echivalentă.'),
         T('Compute the expected durations of the two regimes and the long-run share of the volatile regime.', 'Calculați duratele așteptate ale celor două regimuri și ponderea pe termen lung a regimului volatil.'),
         T('Compute $H$ and $\\rho(1)$ for the ARFIMA model and say whether inflation is stationary and mean-reverting.', 'Calculați $H$ și $\\rho(1)$ pentru modelul ARFIMA și precizați dacă inflația este staționară și cu revenire la medie.')])
solution(8, [
    T('$F_t = 2 + 4 = @{e8.F}$, $K_t = 2/6 = @{e8.K}$; $a_{t|t} = 10 + @{e8.K} \\times 3 = @{e8.af}$, $P_{t|t} = 2(1 - @{e8.K}) = @{e8.Pf}$; $a_{t+1} = @{e8.af}$, $P_{t+1} = @{e8.Pn}$', '$F_t = 2 + 4 = @{e8.F}$, $K_t = 2/6 = @{e8.K}$; $a_{t|t} = 10 + @{e8.K} \\times 3 = @{e8.af}$, $P_{t|t} = 2(1 - @{e8.K}) = @{e8.Pf}$; $a_{t+1} = @{e8.af}$, $P_{t+1} = @{e8.Pn}$'),
    T('$q = 0.25$: $\\bar P/\\sigma^2_\\varepsilon = (q + \\sqrt{q^2 + 4q})/2 = @{e8.Pb}$, $\\bar K = @{e8.Pb}/(1 + @{e8.Pb}) = @{e8.Kb} = \\alpha_{SES}$', '$q = 0{,}25$: $\\bar P/\\sigma^2_\\varepsilon = (q + \\sqrt{q^2 + 4q})/2 = @{e8.Pb}$, $\\bar K = @{e8.Pb}/(1 + @{e8.Pb}) = @{e8.Kb} = \\alpha_{SES}$'),
    T('$E(D_1) = 1/(1 - @{e8.p11}) = @{e8.d1}$ quarters, $E(D_2) = @{e8.d2}$ quarters; $\\pi_1 = (1 - @{e8.p22})/(2 - @{e8.p11} - @{e8.p22}) = @{e8.pi1}$', '$E(D_1) = 1/(1 - @{e8.p11}) = @{e8.d1}$ trimestre, $E(D_2) = @{e8.d2}$ trimestre; $\\pi_1 = (1 - @{e8.p22})/(2 - @{e8.p11} - @{e8.p22}) = @{e8.pi1}$'),
    T('$H = d + 0.5 = @{e8.H}$; $\\rho(1) = d/(1 - d) = @{e8.r1}$; $0 < d < 0.5$: stationary with long memory, mean-reverting', '$H = d + 0{,}5 = @{e8.H}$; $\\rho(1) = d/(1 - d) = @{e8.r1}$; $0 < d < 0{,}5$: staționar cu memorie lungă, cu revenire la medie')],
    [T('The filter moves the level one third of the way to the surprise; in the long run the weight is that of exponential smoothing', 'Filtrul mută nivelul cu o treime din surpriză; pe termen lung, ponderea este cea a netezirii exponențiale'),
     T('In the long run Romanian growth is in the volatile regime about @{e8.pi1p}\\% of the time: transition, 2009 and 2020', 'Pe termen lung, creșterea din România se află în regimul volatil aproximativ @{e8.pi1p}\\% din timp: tranziția, 2009 și 2020'),
     T('Inflation shocks fade slowly, hyperbolically: forecasts return to the mean more slowly than those of an AR model', 'Șocurile inflației se sting lent, hiperbolic: prognozele revin la medie mai încet decît cele ale unui model AR')])

D.recap(('The exam', 'examenul'), [
    T('Written, 2 hours, five subjects, 70\\% of the grade; Chapters 0--10', 'Scris, 2 ore, cinci subiecte, 70\\% din notă; Capitolele 0--10'),
    T('Points for the method, the result with its unit and sign, and the interpretation', 'Punctajul se acordă pentru metodă, pentru rezultat (cu unitate și semn) și pentru interpretare'),
    T('Practise with Part A of every seminar and with Seminar 15', 'Exersați cu Partea A a fiecărui seminar și cu Seminarul 15')])

# =============================================================================
# 9. PROIECTUL DE ECHIPĂ ȘI PREZENȚA
# =============================================================================
D.section('The team project and attendance', 'Proiectul de echipă și prezența')

D.frame(T('The team project: content and deliverables', 'Proiectul de echipă: conținut și livrabile'), items(
    (T('\\textbf{20\\% of the final grade}: a team of 2--4 students analyses real series with the methods of the course', '\\textbf{20\\% din nota finală}: o echipă de 2--4 studenți analizează serii reale cu metodele cursului'),
     [T('one concrete question: forecasting EUR/RON, modelling Romanian inflation, monetary policy and bank rates', 'o întrebare concretă: prognoza cursului EUR/RON, modelarea inflației din România, politica monetară și dobînzile bancare'),
      T('one series: trend and stationarity, smoothing, ARIMA or SARIMA, a training set, a test set and a horizon, point and interval forecasts, two methods compared', 'o serie: trend și staționaritate, netezire, ARIMA sau SARIMA, set de antrenare, set de test și orizont, prognoze punctuale și pe interval, comparația a două metode'),
      T('several series: unit roots, cointegration, VAR or VECM, Granger causality, IRF and FEVD', 'mai multe serii: rădăcini unitare, cointegrare, VAR sau VECM, cauzalitate Granger, IRF și FEVD')]),
    (T('\\textbf{Deliverables}', '\\textbf{Livrabile}'),
     [T('a repository whose code reproduces every number and every chart from the data', 'un repository al cărui cod reproduce, din date, fiecare rezultat numeric și fiecare grafic'),
      T('a short report: the question and the literature, data sources, models, conclusions, references', 'un raport scurt: întrebarea și literatura, sursele de date, modelele, concluziile, bibliografia'),
      T('a presentation; the file \\texttt{AI\\_USE.md}', 'o prezentare; fișierul \\texttt{AI\\_USE.md}')])))

D.frame(T('Project grading criteria and the oral defence', 'Criterii de evaluare a proiectului și susținerea orală'), items(
    (T('\\textbf{The question}: clear, answerable with the data, useful to someone', '\\textbf{Întrebarea}: clară, cu răspuns posibil pe baza datelor, utilă cuiva'),
     [T('``Can a SARIMA forecast Romanian inflation better than the seasonal naive method?\'\' is a question; ``an analysis of inflation\'\' is not', '„Poate un SARIMA să prognozeze inflația din România mai bine decît metoda naivă sezonieră?” este o întrebare; „o analiză a inflației” nu este')]),
    (T('\\textbf{Methods, checks, interpretation, reproducibility}', '\\textbf{Metode, verificări, interpretare, reproductibilitate}'),
     [T('the right test for the question (the toolbox); residual diagnostics; out-of-sample evaluation against a benchmark', 'testul potrivit pentru întrebare (trusa de instrumente); diagnosticarea reziduurilor; evaluare în afara eșantionului, față de un reper'),
      T('what the numbers mean and what the data cannot show; the repository runs from the data to every chart', 'ce înseamnă cifrele și ce nu pot arăta datele; repository-ul rulează de la date pînă la fiecare grafic')]),
    (T('\\textbf{The oral defence}: the project is graded only after its presentation', '\\textbf{Susținerea orală}: proiectul se notează numai după prezentarea lui'),
     [T('each member explains the code and the results: what does this line compute? why this test and not another?', 'fiecare membru explică codul și rezultatele: ce calculează această linie? de ce acest test și nu altul?'),
      T('a line of code that cannot be explained does not count as your own work, with or without AI', 'o linie de cod pe care nu o puteți explica nu este considerată muncă proprie, cu sau fără AI')])))

D.frame(T('The file AI\\_USE.md', 'Fișierul AI\\_USE.md'), items(
    (T('\\textbf{AI tools are allowed and must be declared}', '\\textbf{Instrumentele AI sînt permise și trebuie declarate}'),
     [T('for every use: the tool, the prompt, what was kept and what was corrected', 'pentru fiecare utilizare: instrumentul, promptul, ce s-a păstrat și ce s-a corectat'),
      T('the errors of the AI tool that you found and how you found them', 'erorile instrumentului AI pe care le-ați găsit și modul în care le-ați găsit')]),
    (T('\\textbf{You are responsible for every number and every reference}', '\\textbf{Răspundeți pentru fiecare cifră și pentru fiecare referință}'),
     [T('every number: recomputed with your own code from the data', 'fiecare cifră: recalculată cu propriul cod, din date'),
      T('every reference: its DOI opened and its title checked', 'fiecare referință: DOI-ul deschis și titlul verificat')]),
    (T('\\textbf{Typical AI errors in this course}', '\\textbf{Erori AI tipice în acest curs}'),
     [T('the wrong null hypothesis of ADF or KPSS, Ljung--Box with $m$ degrees of freedom on residuals, AIC compared across different $d$, ``VaR 99\\%\'\', invented references', 'ipoteza nulă greșită pentru ADF sau KPSS, Ljung--Box cu $m$ grade de libertate pe reziduuri, AIC comparat între $d$ diferite, „VaR 99\\%”, referințe inventate')])))

D.frame(T('The final grade', 'Nota finală'), table(
    TB + 'p{3.2cm}cc' + TB + 'p{5.6cm}',
    T('\\textbf{Component}', '\\textbf{Componenta}') + ' & ' + T('\\textbf{Weight}', '\\textbf{Pondere}') + ' & ' + T('\\textbf{When}', '\\textbf{Cînd}') + ' & ' + T('\\textbf{Criteria}', '\\textbf{Criterii}'),
    [T('written exam', 'examenul scris') + ' & 70\\% & ' + T('exam session', 'sesiunea') + ' & ' + T('method, result, interpretation', 'metodă, rezultat, interpretare'),
     T('team project', 'proiectul de echipă') + ' & 20\\% & ' + T('end of semester', 'finalul semestrului') + ' & ' + T('question, methods, checks, oral defence', 'întrebare, metode, verificări, susținere orală'),
     T('attendance', 'prezența') + ' & 10\\% & ' + T('every week', 'în fiecare săptămînă') + ' & ' + T('presence at the lectures and seminars', 'prezența la cursuri și seminarii')],
    size='footnotesize') + items(
    T('The seminars are for practice and are not graded; students hand in nothing from the seminars', 'Seminariile au rol de exercițiu și nu se notează; studenții nu predau nimic de la seminarii'),
    T('The quizzes on the course site are for self-assessment', 'Quiz-urile de pe site-ul cursului sînt pentru autoevaluare')))

# =============================================================================
# 10. CONTRIBUȚIA POSIBILĂ A AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Practice}: new exercises in the style of the exam, with output generated from real data by your own code', '\\textbf{Exercițiu}: probleme noi în stilul examenului, cu rezultate generate din date reale de propriul cod'),
    T('\\textbf{Explanation}: a second explanation of an output table, a derivation or a test you did not understand', '\\textbf{Explicații}: o a doua explicație pentru un tabel de rezultate, o derivare sau un test pe care nu l-ați înțeles'),
    T('\\textbf{Project}: a first draft of the code of the Box--Jenkins steps, of a rolling evaluation, of a VECM', '\\textbf{Proiect}: o primă versiune a codului pentru pașii Box--Jenkins, pentru o evaluare cu origini mobile, pentru un VECM'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that downloads the Romanian HICP from Eurostat (prc\\_hicp\\_minr, M.I15.TOTAL.RO), fits every SARIMA(p,1,q)(P,1,Q)12 with p, q <= 2 and P, Q <= 1 to 100 log HICP up to August 2024, reports AICc, BIC and Ljung-Box p-values with m - k degrees of freedom, and compares one-step forecasts with the seasonal naive method by MASE and the Diebold-Mariano test.}',
        '\\aiprompt{Write Python code that downloads the Romanian HICP from Eurostat (prc\\_hicp\\_minr, M.I15.TOTAL.RO), fits every SARIMA(p,1,q)(P,1,Q)12 with p, q <= 2 and P, Q <= 1 to 100 log HICP up to August 2024, reports AICc, BIC and Ljung-Box p-values with m - k degrees of freedom, and compares one-step forecasts with the seasonal naive method by MASE and the Diebold-Mariano test.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('The null hypotheses: ADF has a unit root under $H_0$, KPSS has stationarity; an answer that swaps them is wrong', 'Ipotezele nule: la ADF, $H_0$ este rădăcina unitară, la KPSS staționaritatea; un răspuns care le inversează este greșit'),
    T('The degrees of freedom: Ljung--Box on residuals uses $m - k$; Granger $F$ uses $(p, T - Kp - 1)$', 'Gradele de libertate: Ljung--Box pe reziduuri folosește $m - k$; testul Granger $F$ folosește $(p, T - Kp - 1)$'),
    T('The signs: the EViews and Python conventions for MA terms and for the error-correction coefficient', 'Semnele: convențiile EViews și Python pentru termenii MA și pentru coeficientul de corecție a erorii'),
    T('The time order: no information from the test sample in the estimation, the scaling or the model choice', 'Ordinea timpului: nicio informație din eșantionul de test în estimare, în scalare sau în alegerea modelului'),
    T('Every number recomputed with your own code; every reference checked by its DOI', 'Fiecare cifră recalculată cu propriul cod; fiecare referință verificată prin DOI')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Look at the data first: components, transformations, the correlogram', 'Întîi priviți datele: componente, transformări, corelogramă'),
    T('Decide $d$ and $D$ with ADF and KPSS together; identify with ACF and PACF; choose with BIC; check the residuals', 'Decideți $d$ și $D$ cu ADF și KPSS împreună; identificați cu ACF și PACF; alegeți cu BIC; verificați reziduurile'),
    T('Model the variance when it clusters (GARCH); model several series together with VAR or VECM', 'Modelați varianța cînd se grupează (GARCH); modelați mai multe serii împreună cu VAR sau VECM'),
    T('Judge forecasts out of sample against simple benchmarks; a better in-sample fit is not a better forecast', 'Judecați prognozele în afara eșantionului, față de repere simple; o potrivire mai bună în eșantion nu înseamnă o prognoză mai bună'),
    T('The exam rewards method, result and interpretation; the project rewards a clear question answered honestly', 'Examenul răsplătește metoda, rezultatul și interpretarea; proiectul răsplătește o întrebare clară, cu un răspuns onest')))

D.frame(T('Self-assessment', 'Autoevaluare'), items(
    (T('\\textbf{Question}: ADF does not reject and KPSS rejects. What do you conclude?', '\\textbf{Întrebare}: ADF nu respinge, iar KPSS respinge. Ce concluzie trageți?'),
     [T('\\textbf{Answer}: both point to a unit root: difference the series', '\\textbf{Răspuns}: ambele indică o rădăcină unitară: diferențiem seria')]),
    (T('\\textbf{Question}: the residuals of an ARMA(1,1) give Ljung--Box $Q(10) = 16$. Is the model adequate at 5\\%?', '\\textbf{Întrebare}: reziduurile unui ARMA(1,1) dau Ljung--Box $Q(10) = 16$. Este modelul adecvat la 5\\%?'),
     [T('\\textbf{Answer}: with $10 - 2 = 8$ df the critical value is 15.51: no, autocorrelation is left', '\\textbf{Răspuns}: cu $10 - 2 = 8$ grade de libertate valoarea critică este 15,51: nu, rămîne autocorelație')]),
    (T('\\textbf{Question}: two $I(1)$ series are cointegrated. Why not a VAR in differences?', '\\textbf{Întrebare}: două serii $I(1)$ sînt cointegrate. De ce nu un VAR în diferențe?'),
     [T('\\textbf{Answer}: it omits the error-correction term and loses the long-run information', '\\textbf{Răspuns}: omite termenul de corecție a erorii și pierde informația pe termen lung')])))

D.references(bib())

if __name__ == '__main__':
    finalize(D.write(V))
