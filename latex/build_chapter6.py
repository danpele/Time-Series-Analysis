r"""
build_chapter6.py -- Capitolul 6 (Modele VAR și cauzalitate Granger), EN + RO dintr-o singură sursă
==================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_06/ch6_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter6_var_models_granger_causality.tex
  RO/Cursuri/capitol6_modele_var_cauzalitate_granger.tex
Rulare:
  python3 Quantlets/Ch_06/generate_all_charts.py
  python3 latex/build_chapter6.py && python3 latex/tsa_build.py compile 6
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, photo   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch6_common import QLURL, REFS, T, bib, date, finalize, load, pv, qtr, month   # noqa: E402


def items(*xs):
    """tsa_build.items, with (text, []) treated as a plain bullet."""
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


N = load()
V = Values()
D = Deck(6, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\quantlet{{{folder.replace("_", chr(92) + "_")}}}{{\\qlurl{{{folder}}}}}'


def chart(title, fig, folder, bullets, h='0.62\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def interp(title, bullets, size='small'):
    D.frame(T(f'Interpreting {title[0]}', f'Interpretarea {title[1]}'), items(*bullets), size)


PH = {
    'sims': ('ch6_christopher_sims_2011.jpg', C + 'Christopher_A._Sims_close-up_(cropped).jpg',
             T('Photo', 'Foto') + ': Holger Motzkau (2011); CC BY-SA 3.0; Wikimedia Commons'),
    'nobel': ('ch6_sims_nobel_lecture_2011.jpg', C + 'Nobel_Prize_2011-Nobel_lectures_KVA-DSC_8085.jpg',
              T('Photo', 'Foto') + ': Holger Motzkau (2011); CC BY-SA 3.0; Wikimedia Commons'),
    'granger': ('ch3_clive_granger_2008.jpg', C + 'Clive_Granger_by_Olaf_Storbeck_(3x4_cropped).jpg',
                T('Photo', 'Foto') + ': Olaf Storbeck (2008); CC BY-SA 2.0; Wikimedia Commons'),
    'fed': ('ch6_eccles_building_2011.jpg', C + 'Eccles_Building_(26088200676).jpg',
            T('Photo', 'Foto') + ': Federal Reserve (2011); public domain; Wikimedia Commons'),
    'fse': ('ch6_frankfurt_exchange_2015.jpg', C + 'Frankfurt_Stock_Exchange_(Ank_Kumar)_01.jpg',
            T('Photo', 'Foto') + ': Ank Kumar (2015); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


def mat2(M, d=2):
    """A 2 x 2 matrix as a LaTeX pmatrix with RO decimal marks."""
    f = lambda x: '⁅' + f'{x:.{d}f}' + '⁆'
    return f'\\begin{{pmatrix}} {f(M[0][0])} & {f(M[0][1])} \\\\ {f(M[1][0])} & {f(M[1][1])} \\end{{pmatrix}}'


def vec2(v, d=2):
    f = lambda x: '⁅' + f'{x:.{d}f}' + '⁆'
    return f'\\begin{{pmatrix}} {f(v[0])} \\\\ {f(v[1])} \\end{{pmatrix}}'


# =============================================================================
# CIFRE
# =============================================================================
R = N['ro']
V.raw('ro.q0', qtr(R['q_first']))
V.raw('ro.q1', qtr(R['q_last']))
V.raw('ro.T', str(R['T']))
for c in ['g', 'pi', 'i']:
    V.put(f'ro.m.{c}', R['mean'][c], 2)
    V.put(f'ro.s.{c}', R['sd'][c], 2)
V.put('ro.c.pii', R['corr']['pi']['i'], 2)
V.put('ro.c.gi', R['corr']['g']['i'], 2)
V.put('ro.pimax', R['pi_max'], 1)
V.raw('ro.pimaxd', month(R['pi_max_d']))
V.put('ro.imax', R['i_max'], 1)
V.raw('ro.imaxd', month(R['i_max_d']))
V.put('ro.pil', R['pi_last'], 1)
V.raw('ro.pild', month(R['pi_last_d']))
V.put('ro.il', R['i_last'], 2)
V.put('ro.ul', R['u_last'], 1)
V.put('ro.fx', R['fx_last'], 2)
V.put('ro.gmin', R['g_min'], 1)
V.raw('ro.gmind', qtr(R['g_min_d']))
V.put('ro.gmin2', R['g_min2'], 1)
V.raw('ro.gmin2d', qtr(R['g_min2_d']))

CC = N['ccf']
V.put('cc.bet0', CC['bet']['0'], 2)
V.put('cc.bet1', CC['bet']['1'], 2)
V.put('cc.betm1', CC['bet']['-1'], 2)
V.put('cc.dax0', CC['dax']['0'], 2)
V.put('cc.dax1', CC['dax']['1'], 2)
V.put('cc.band', CC['band'], 3)
V.int('cc.T', CC['T_daily'])
V.raw('cc.k', str(CC['i_pi_kmax']))
V.put('cc.ipmax', CC['i_pi_max'], 2)
V.put('cc.ip0', CC['i_pi']['0'], 2)
V.put('cc.ipm4', CC['i_pi']['-4'], 2)
V.put('cc.ip4', CC['i_pi']['4'], 2)

W = N['wx']
V.put('wx.l1', W['lam'][0], 1)
V.put('wx.l2', W['lam'][1], 1)
V.put('wx.tr', W['tr'], 1)
V.put('wx.dA', W['detA'], 2)
V.put('wx.det', W['det'], 2)
V.put('wx.mu1', W['mu'][0], 2)
V.put('wx.mu2', W['mu'][1], 2)
V.raw('wx.f1', vec2(W['f1']))
V.raw('wx.f2', vec2(W['f2']))
V.raw('wx.A2', mat2(W['A2']))
V.raw('wx.P', mat2(W['P'], 3))
V.raw('wx.Th1', mat2(W['Theta1'], 3))
V.raw('wx.Prev', mat2(W['P_rev'], 3))
V.raw('wx.G0', mat2(W['Gamma0']))
V.put('wx.g11', W['Gamma0'][0][0], 2)
V.put('wx.g22', W['Gamma0'][1][1], 2)
V.put('wx.mse22', W['mse2'][1][1], 2)
V.put('wx.mse11', W['mse2'][0][0], 2)
V.put('wx.fe21', 100 * W['fevd2'][1][0], 1)
V.put('wx.fe12', 100 * W['fevd2'][0][1], 1)
V.put('wx.p21', W['P'][1][0], 3)
V.put('wx.p22', W['P'][1][1], 3)
V.put('wx.th21', W['Theta1'][1][0], 3)
SI = N['sim']
V.put('sim.c', SI['corr_sim'], 2)

IC = N['ic']
for p in range(IC['pmax'] + 1):
    for c in ('aic', 'bic', 'hq'):
        V.put(f'ic.{p}.{c}', IC['table'][str(p)][c], 3)
for c in ('aic', 'bic', 'hq'):
    V.raw(f'ic.b.{c}', str(IC['best'][c]))
V.raw('ic.T', str(IC['T']))
V.raw('ic.pmax', str(IC['pmax']))

RT = N['roots']
V.put('rt.ro', RT['Romania VAR(2)']['max_mod'], 3)
V.put('rt.us', RT['US VAR(4), 1960-2000']['max_mod'], 3)
V.put('rt.mk', RT['S&P 500, DAX, BET VAR(4)']['max_mod'], 3)

E = N['est']
for c in ['g', 'pi', 'i']:
    for k in ['const', 'L1.g', 'L1.pi', 'L1.i', 'L2.g', 'L2.pi', 'L2.i']:
        kk = k.replace('.', '')
        V.put(f'e.{c}.{kk}', E['params'][c][k], 2)
        V.put(f'e.{c}.{kk}.t', E['tvalues'][c][k], 1)
    V.put(f'e.r2.{c}', E['r2'][c], 2)
V.raw('e.n', str(E['nobs']))
V.raw('e.k', str(E['n_coef']))
V.raw('e.q0', qtr(E['first_used']))
V.put('e.cpii', E['resid_corr'][1][2], 2)
V.put('e.cgpi', E['resid_corr'][0][1], 2)
V.put('e.cgi', E['resid_corr'][0][2], 2)
V.put('e.sg', E['sigma_u'][0][0] ** 0.5, 2)
V.put('e.spi', E['sigma_u'][1][1] ** 0.5, 2)
V.put('e.si', E['sigma_u'][2][2] ** 0.5, 2)

RS = N['resid']
V.raw('rs.gd', qtr(RS['largest']['g']['date']))
V.put('rs.gv', RS['largest']['g']['value'], 1)
V.raw('rs.pid', qtr(RS['largest']['pi']['date']))
V.put('rs.piv', RS['largest']['pi']['value'], 1)
V.raw('rs.id', qtr(RS['largest']['i']['date']))
V.put('rs.iv', RS['largest']['i']['value'], 1)
for h in ('8', '12'):
    V.put(f'rs.q{h}', RS[f'lb{h}']['stat'], 1)
    V.raw(f'rs.q{h}df', str(RS[f'lb{h}']['df']))
    V.raw(f'rs.q{h}p', pv(RS[f'lb{h}']['p']))
V.put('rs.jb', RS['jb']['stat'], 0)
V.raw('rs.jbp', pv(RS['jb']['p']))
V.put('rs.jbd', RS['jb_dummies']['stat'], 0)
V.put('rs.kg', RS['kurt']['g'], 1)
V.put('rs.ki', RS['kurt']['i'], 1)

GR = N['gr_ro']['tests']
for k, v in GR.items():
    kk = k.replace('->', '')
    V.put(f'gr.{kk}.F', v['F'], 2)
    V.raw(f'gr.{kk}.p', pv(v['p']))
V.put('gr.cv', GR['g->i']['crit5'], 2)
V.raw('gr.df2', str(GR['g->i']['df2']))
BI = N['gr_ro']['bivariate_p']
V.raw('gr.bi.ig', pv(BI['i->g']))
V.raw('gr.bi.pig', pv(BI['pi->g']))
INS = N['gr_ro']['inst']
V.raw('gr.inst.i', pv(INS['i']))
GH = N['gr_hand']
V.put('gh.ru', GH['rss_u'], 2)
V.put('gh.rr', GH['rss_r'], 2)
V.raw('gh.T', str(GH['T']))
V.raw('gh.df2', str(GH['df2']))
V.put('gh.num', (GH['rss_r'] - GH['rss_u']) / 2, 3)
V.put('gh.den', GH['rss_u'] / GH['df2'], 4)
V.put('gh.F', GH['F'], 2)
V.raw('gh.p', pv(GH['p']))
V.put('gh.cv', GH['crit5'], 2)

GM = N['gr_mkt']
for k, v in GM['tests'].items():
    kk = k.replace('->', '_')
    V.put(f'gm.{kk}.F', v['F'], 1)
    V.raw(f'gm.{kk}.p', pv(v['p']))
V.int('gm.T', GM['tests']['sp500->bet']['T'])
V.put('gm.b1', GM['coef_bet_sp1'], 3)
V.put('gm.b1se', GM['se_bet_sp1'], 3)
V.put('gm.d1', GM['coef_dax_sp1'], 3)
for k in ('aic', 'bic', 'hqic'):
    V.raw(f'gm.o.{k}', str(GM['orders'][k]))

GS = N['gr_sim']
for T_ in ('50', '100', '200', '500'):
    V.put(f'gs.{T_}.b', 100 * GS[T_]['bivariate'], 0)
    V.put(f'gs.{T_}.t', 100 * GS[T_]['trivariate'], 1)

IR = N['irf']
V.put('ir.ipi0', IR['i_pi'][0], 2)
V.put('ir.ipimax', max(IR['i_pi']), 2)
V.raw('ir.ipih', str(int(np.argmax(IR['i_pi']))))
V.put('ir.ipi12', IR['i_pi'][12], 2)
V.put('ir.pii1', IR['pi_i'][1], 2)
V.put('ir.piimax', max(IR['pi_i']), 2)
V.raw('ir.piih', str(int(np.argmax(IR['pi_i']))))
V.put('ir.gi1', abs(IR['g_i'][1]), 2)
V.put('ir.ig2', IR['i_g'][2], 2)
V.put('ir.sg', IR['sd'][0], 2)
V.put('ir.spi', IR['P'][1][1], 2)
V.put('ir.si', IR['P'][2][2], 2)
V.put('ir.pi0', IR['pi_pi'][0], 2)
V.put('ir.pi4', IR['pi_pi'][4], 2)
V.put('ir.pi12', IR['pi_pi'][12], 2)
V.raw('ir.sig', str(sum(IR['sig_i_pi'])))

OR = N['order']
V.put('or.A0', OR['i_pi_A'][0], 2)
V.put('or.Amax', max(OR['i_pi_A']), 2)
V.put('or.Bmax', max(OR['i_pi_B']), 2)
V.put('or.Gmax', max(OR['i_pi_G']), 2)
V.put('or.pA', max(OR['pi_i_A']), 2)
V.put('or.pB', max(OR['pi_i_B']), 2)
V.put('or.pB0', OR['pi_i_B'][0], 2)
V.put('or.pG', max(OR['pi_i_G']), 2)
V.put('or.r', OR['corr_pi_i'], 2)

FE = N['fevd']
for c in ['g', 'pi', 'i']:
    for h in ('1', '4', '8', '12'):
        for j, s in enumerate(['g', 'pi', 'i']):
            V.put(f'fe.{c}.{h}.{s}', FE[c][h][j], 0)

MI = N['mirf']
V.put('mi.d0', MI['G_dax'][0], 2)
V.put('mi.d1', MI['G_dax'][1], 2)
V.put('mi.b0', MI['G_bet'][0], 2)
V.put('mi.b1', MI['G_bet'][1], 2)
V.put('mi.Bb1', MI['B_bet'][1], 2)
V.put('mi.Bd1', MI['B_dax'][1], 2)
V.put('mi.sd', MI['sd']['sp500'], 2)
V.put('mi.cu', MI['corr_u'][0][1], 2)

SP = N['spill']
tab = SP['table']
for i_ in ['S&P 500', 'DAX', 'BET']:
    k = {'S&P 500': 'sp', 'DAX': 'dax', 'BET': 'bet'}[i_]
    for j_ in ['S&P 500', 'DAX', 'BET', 'from others']:
        kj = {'S&P 500': 'sp', 'DAX': 'dax', 'BET': 'bet', 'from others': 'from'}[j_]
        V.put(f'sp.{k}.{kj}', tab[i_][j_], 1)
    V.put(f'sp.to.{k}', tab['to others'][i_], 1)
V.put('sp.tot', SP['total'], 1)
V.put('sp.mean', SP['roll_mean'], 1)
V.put('sp.max', SP['roll_max'], 1)
V.raw('sp.maxd', date(SP['roll_max_d']))
V.put('sp.min', SP['roll_min'], 1)
V.put('sp.last', SP['roll_last'], 1)
V.put('sp.v07', SP['v2007'], 0)
V.put('sp.v08', SP['v2008q4'], 0)
V.put('sp.v20', SP['v2020'], 0)

FC = N['fc']
V.raw('fc.f0', qtr(FC['first_f']))
V.raw('fc.f1', qtr(FC['last_f']))
V.raw('fc.lo', qtr(FC['last_obs']))
for j, c in enumerate(['g', 'pi', 'i']):
    V.put(f'fc.{c}.last', FC['last'][c], 1)
    for h in (1, 4, 8):
        V.put(f'fc.{c}.{h}', FC['pt'][h - 1][j], 1)
        V.put(f'fc.{c}.{h}.lo', FC['lo'][h - 1][j], 1)
        V.put(f'fc.{c}.{h}.hi', FC['hi'][h - 1][j], 1)
    V.put(f'fc.mu.{c}', FC['mu'][c], 1)

OO = N['oos']
for c in ['g', 'pi', 'i']:
    for h in ('1', '4'):
        for m in ('VAR', 'AR', 'RW'):
            V.put(f'oo.ro.{m}.{c}.{h}', OO['ro'][m][c][h], 2)
        V.put(f'oo.ro.rv.{c}.{h}', OO['ro']['VAR'][c][h] / OO['ro']['RW'][c][h], 2)
        V.put(f'oo.ro.ra.{c}.{h}', OO['ro']['AR'][c][h] / OO['ro']['RW'][c][h], 2)
    V.put(f'oo.dm.{c}', OO['dm_ro'][c][0], 2)
    V.raw(f'oo.dm.{c}.p', pv(OO['dm_ro'][c][1]))
for c in ['pi', 'u', 'R']:
    for h in ('2', '4', '8'):
        V.put(f'oo.us.rv.{c}.{h}', OO['us']['VAR'][c][h] / OO['us']['RW'][c][h], 2)
        V.put(f'oo.us.ra.{c}.{h}', OO['us']['AR'][c][h] / OO['us']['RW'][c][h], 2)
V.raw('oo.n', str(OO['n_ro']))
V.raw('oo.n0', str((2014 - int(R['q_first'][:4]) + 1) * 4 - 2))   # quarters 2005Q3-2014Q4 used by the first VAR(2)

US = N['us']
V.raw('us.T', str(US['T_sw']))
V.raw('us.last', qtr(US['last']))
SW = N['sw']
for k, v in SW['granger'].items():
    V.raw(f'sw.g.{k.replace("->", "")}', pv(v))
for k, v in SW['granger_full'].items():
    V.raw(f'sw.gf.{k.replace("->", "")}', pv(v))
for c in ['pi', 'u', 'R']:
    for h in ('1', '4', '8', '12'):
        for j, s in enumerate(['pi', 'u', 'R']):
            V.put(f'sw.fe.{c}.{h}.{s}', SW['fevd'][c][h][j], 0)
V.put('sw.uRmax', max(SW['u_R']), 2)
V.raw('sw.uRh', str(int(np.argmax(SW['u_R']))))
V.put('sw.piR1', SW['pi_R'][1], 2)
V.put('sw.piR12', SW['pi_R'][12], 2)
V.put('sw.piRmin', min(SW['pi_R']), 2)
V.put('sw.RR0', SW['R_R'][0], 2)
V.put('sw.mod', SW['max_mod'], 3)
V.put('sw.modf', SW['max_mod_full'], 3)
V.put('sw.uRf', min(SW['u_R_full'][:6]), 2)
V.put('sw.piRf', max(SW['pi_R_full']), 2)
V.raw('sw.Tf', str(SW['T_full']))
V.raw('sw.k', str(SW['n_coef']))

# ---- worked examples (computed here)
V.put('ex.k5p4', 5 + 4 * 25, 0)
V.put('ex.k3p2', 3 + 2 * 9, 0)
V.put('ex.ar1', 0.5 ** 1, 1)


# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: when several series move together (output, prices, interest rates; three stock markets), how do we model them jointly, and what can we learn about who leads whom?',
       '\\textbf{Întrebarea}: cînd mai multe serii evoluează împreună (producția, prețurile, dobînzile; trei piețe bursiere), cum le modelăm împreună și ce putem afla despre cine pe cine precedă?'),
     [T('one equation per variable, each with the past of all variables: the vector autoregression (VAR)', 'cîte o ecuație pentru fiecare variabilă, fiecare cu trecutul tuturor variabilelor: modelul vector autoregresiv (VAR)')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('from one series to several: vector processes and cross-correlations; the VAR$(p)$ model and its stability', 'de la o serie la mai multe: procese vectoriale și corelații încrucișate; modelul VAR$(p)$ și stabilitatea lui'),
      T('estimation, lag selection and diagnostics; Granger causality and its pitfalls', 'estimare, alegerea numărului de laguri și diagnosticare; cauzalitatea Granger și capcanele ei'),
      T('impulse responses, variance decomposition, forecasts; the structural VAR idea of Sims (1980)', 'funcții de răspuns la impuls, descompunerea varianței, prognoze; ideea VAR-ului structural, Sims (1980)')]),
    T('We build on Chapter 2 (AR models) and Chapter 3 (stationarity tests); Seminar 6 comes before this lecture',
      'Pornim de la Capitolul 2 (modele AR) și de la Capitolul 3 (teste de staționaritate); Seminarul 6 are loc înaintea acestui curs')))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Write a VAR$(p)$ in matrix form and equation by equation, and count its parameters', 'Scrieți un VAR$(p)$ în formă matriceală și ecuație cu ecuație și numărați-i parametrii'),
    T('Check stability with the eigenvalues of the companion matrix, and compute the mean of a stable VAR', 'Verificați stabilitatea cu valorile proprii ale matricei companion și calculați media unui VAR stabil'),
    T('Estimate a VAR by OLS, choose $p$ with AIC, BIC and HQ, and check the residuals', 'Estimați un VAR prin OLS, alegeți $p$ cu AIC, BIC și HQ și verificați reziduurile'),
    T('Test Granger causality and explain what it does not prove', 'Testați cauzalitatea Granger și explicați ce nu demonstrează ea'),
    T('Compute and read orthogonalised impulse responses and variance decompositions, and explain why the ordering matters', 'Calculați și interpretați funcțiile de răspuns la impuls ortogonalizate și descompunerea varianței și explicați de ce contează ordinea variabilelor'),
    T('Forecast with a VAR and compare it fairly with univariate benchmarks', 'Faceți prognoze cu un VAR și comparați-le corect cu metode univariate de referință')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refHP, Ch.~7 (multivariate time series)', 'Manual: \\refHP, cap.~7 (serii de timp multivariate)'),
     [T('companion, free online: \\refFPP, Sec.~12.3 (vector autoregressions)', 'manual însoțitor, gratuit online: \\refFPP, secț.~12.3 (modele vectoriale autoregresive)')]),
    T('Theory: \\refLut, Ch.~2--4; \\refHamilton, Ch.~11; an accessible survey: \\refSW', 'Teorie: \\refLut, cap.~2--4; \\refHamilton, cap.~11; o sinteză accesibilă: \\refSW'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_06}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_06}'),
     [T('\\texttt{VAR} from \\texttt{statsmodels}: \\texttt{select\\_order}, \\texttt{fit}, \\texttt{test\\_whiteness}, \\texttt{irf}, \\texttt{fevd}, \\texttt{forecast\\_interval}',
        '\\texttt{VAR} din \\texttt{statsmodels}: \\texttt{select\\_order}, \\texttt{fit}, \\texttt{test\\_whiteness}, \\texttt{irf}, \\texttt{fevd}, \\texttt{forecast\\_interval}')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter6_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter6_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. DE LA O SERIE LA MAI MULTE
# =============================================================================
D.section('From one series to several', 'De la o serie la mai multe')

chart(T('Romania: four macro series that move together', 'România: patru serii macroeconomice care evoluează împreună'), 'tsa_ch6_ro_macro', 'TSA_ch6_multivariate_data', [
    T('Real GDP growth (q/q) and unemployment rate, seasonally adjusted; 12-month HICP inflation; ROBOR 3M, the 3-month interbank rate; source: Eurostat; EUR/RON: the BNR reference rate',
      'Creșterea PIB-ului real (față de trimestrul anterior) și rata șomajului, ajustate sezonier; inflația anuală IAPC; ROBOR 3M, dobînda la 3 luni; sursa: Eurostat; EUR/RON: cursul de referință BNR')],
    h='0.6\\textheight')

interp(('the Romanian series', 'seriilor românești'), [
    (T('Inflation and ROBOR 3M rise and fall together: correlation @{ro.c.pii} over @{ro.q0}--@{ro.q1} (quarterly means)', 'Inflația și ROBOR 3M cresc și scad împreună: corelația este @{ro.c.pii} în @{ro.q0}--@{ro.q1} (medii trimestriale)'),
     [T('inflation peaked at @{ro.pimax}\\% in @{ro.pimaxd}; ROBOR 3M peaked at @{ro.imax}\\% in @{ro.imaxd} (the 2008 liquidity squeeze)', 'inflația a atins @{ro.pimax}\\% în @{ro.pimaxd}; ROBOR 3M a atins @{ro.imax}\\% în @{ro.imaxd} (criza de lichiditate din 2008)')]),
    (T('GDP growth is noisy, with two deep falls: @{ro.gmin}\\% in @{ro.gmind} and @{ro.gmin2}\\% in @{ro.gmin2d}', 'Creșterea PIB-ului este zgomotoasă, cu două scăderi mari: @{ro.gmin}\\% în @{ro.gmind} și @{ro.gmin2}\\% în @{ro.gmin2d}'),
     [T('latest values: inflation @{ro.pil}\\% (@{ro.pild}), ROBOR 3M @{ro.il}\\%, unemployment @{ro.ul}\\%, EUR/RON @{ro.fx}', 'ultimele valori: inflația @{ro.pil}\\% (@{ro.pild}), ROBOR 3M @{ro.il}\\%, șomajul @{ro.ul}\\%, EUR/RON @{ro.fx}')]),
    (T('Questions for this chapter', 'Întrebările capitolului'),
     [T('Does inflation help to forecast the interest rate?', 'Ajută inflația la prognoza dobînzii?'),
      T('How does ROBOR react to an inflation surprise, and for how long?', 'Cum reacționează ROBOR la o surpriză a inflației și cît timp?')])])

D.frame(T('Vector time series', 'Serii de timp vectoriale'), items(
    (T('\\textbf{Definition}: $\\bY_t = (Y_{1t}, \\dots, Y_{Kt})\'$ collects $K$ series observed at the same dates', '\\textbf{Definiție}: $\\bY_t = (Y_{1t}, \\dots, Y_{Kt})\'$ grupează $K$ serii observate la aceleași date'),
     [T('example: $\\bY_t = (g_t, \\pi_t, i_t)\'$, GDP growth, inflation and ROBOR 3M of quarter $t$; $K = 3$', 'exemplu: $\\bY_t = (g_t, \\pi_t, i_t)\'$, creșterea PIB-ului, inflația și ROBOR 3M din trimestrul $t$; $K = 3$'),
      T('$Y_{jt}$: the value of series $j$ at date $t$; the prime $\'$ denotes the transpose (a column vector)', '$Y_{jt}$: valoarea seriei $j$ la data $t$; apostroful $\'$ notează transpusa (un vector coloană)')]),
    (T('\\textbf{Mean vector}: $\\boldsymbol{\\mu} = E[\\bY_t]$, $K \\times 1$', '\\textbf{Vectorul mediilor}: $\\boldsymbol{\\mu} = E[\\bY_t]$, de dimensiune $K \\times 1$'),
     [T('\\textbf{autocovariance matrix} at lag $h$: $\\bGamma(h) = \\Cov(\\bY_t, \\bY_{t-h}) = E[(\\bY_t - \\boldsymbol{\\mu})(\\bY_{t-h} - \\boldsymbol{\\mu})\']$, $K \\times K$',
        '\\textbf{matricea de autocovarianță} la lagul $h$: $\\bGamma(h) = \\Cov(\\bY_t, \\bY_{t-h}) = E[(\\bY_t - \\boldsymbol{\\mu})(\\bY_{t-h} - \\boldsymbol{\\mu})\']$, $K \\times K$'),
      T('diagonal: the autocovariances of each series (Chapter 1); off-diagonal: \\textbf{cross-covariances} $\\gamma_{jk}(h) = \\Cov(Y_{jt}, Y_{k,t-h})$', 'diagonala: autocovarianțele fiecărei serii (Capitolul 1); în afara diagonalei: \\textbf{covarianțele încrucișate} $\\gamma_{jk}(h) = \\Cov(Y_{jt}, Y_{k,t-h})$')]),
    (T('\\textbf{Weak stationarity}: $\\boldsymbol{\\mu}$ and $\\bGamma(h)$ do not depend on $t$', '\\textbf{Staționaritate slabă}: $\\boldsymbol{\\mu}$ și $\\bGamma(h)$ nu depind de $t$'),
     [T('$\\bGamma(h)$ is not symmetric, but $\\bGamma(-h) = \\bGamma(h)\'$: ``$Y_1$ leads $Y_2$\'\' and ``$Y_2$ leads $Y_1$\'\' are different statements', '$\\bGamma(h)$ nu este simetrică, dar $\\bGamma(-h) = \\bGamma(h)\'$: „$Y_1$ precedă $Y_2$” și „$Y_2$ precedă $Y_1$” sînt afirmații diferite')])))

D.frame(T('The cross-correlation function', 'Funcția de corelație încrucișată'), items(
    (T('\\textbf{Definition}: $\\rho_{yx}(k) = \\Corr(y_t, x_{t-k}) = \\gamma_{yx}(k)/(\\sigma_y\\sigma_x)$, for $k = 0, \\pm 1, \\pm 2, \\dots$', '\\textbf{Definiție}: $\\rho_{yx}(k) = \\Corr(y_t, x_{t-k}) = \\gamma_{yx}(k)/(\\sigma_y\\sigma_x)$, pentru $k = 0, \\pm 1, \\pm 2, \\dots$'),
     [T('the cross-correlation function (CCF): $k > 0$: past $x$ with today\'s $y$ ($x$ leads); $k < 0$: $y$ leads', 'funcția de corelație încrucișată (CCF): $k > 0$: $x$ din trecut cu $y$ de azi ($x$ precedă); $k < 0$: $y$ precedă'),
      T('$\\gamma_{yx}(k) = \\Cov(y_t, x_{t-k})$: the cross-covariance; $\\sigma_y$, $\\sigma_x$: the standard deviations of the two series', '$\\gamma_{yx}(k) = \\Cov(y_t, x_{t-k})$: covarianța încrucișată; $\\sigma_y$, $\\sigma_x$: abaterile standard ale celor două serii')]),
    (T('Sample version: $\\hat\\rho_{yx}(k)$, the correlation of the pairs $(y_t, x_{t-k})$', 'Versiunea de eșantion: $\\hat\\rho_{yx}(k)$, corelația perechilor $(y_t, x_{t-k})$'),
     [T('if the two series are independent white noises, $\\hat\\rho_{yx}(k) \\approx N(0, 1/T)$, with $T$ the number of observations: bands $\\pm 1.96/\\sqrt{T}$', 'dacă cele două serii sînt zgomote albe independente, $\\hat\\rho_{yx}(k) \\approx N(0, 1/T)$, unde $T$ este numărul de observații: benzile $\\pm 1{,}96/\\sqrt{T}$'),
      T('with autocorrelated series the bands are too narrow: spurious lead--lag patterns (compare Chapter 3, spurious regression)', 'pentru serii autocorelate benzile sînt prea înguste: apar relații false de precedență (comparați cu Capitolul 3, regresia falsă)')]),
    T('The CCF looks at one pair and one lag at a time; a VAR looks at all variables and all lags together', 'CCF privește o pereche și un lag pe rînd; un VAR privește împreună toate variabilele și toate lagurile')))

chart(T('Cross-correlations: markets and Romanian macro data', 'Corelații încrucișate: piețe și date macroeconomice românești'), 'tsa_ch6_ccf', 'TSA_ch6_multivariate_data', [
    T('Left: daily log returns (EODHD), @{cc.T} common trading days, 2000--2026; bars: $\\Corr(\\text{DAX}_t, \\text{S\\&P}_{t-k})$ and $\\Corr(\\text{BET}_t, \\text{S\\&P}_{t-k})$. Right: quarterly ROBOR 3M against lags of inflation',
      'Stînga: randamente logaritmice zilnice (EODHD), @{cc.T} de zile comune de tranzacționare, 2000--2026; bare: $\\Corr(\\text{DAX}_t, \\text{S\\&P}_{t-k})$ și $\\Corr(\\text{BET}_t, \\text{S\\&P}_{t-k})$. Dreapta: ROBOR 3M trimestrial față de lagurile inflației')],
    h='0.56\\textheight')

interp(('the cross-correlations', 'corelațiilor încrucișate'), [
    (T('BET with the S\\&P 500: @{cc.bet0} on the same day and @{cc.bet1} with the previous day ($k = 1$); with $k = -1$: @{cc.betm1}', 'BET cu S\\&P 500: @{cc.bet0} în aceeași zi și @{cc.bet1} cu ziua precedentă ($k = 1$); pentru $k = -1$: @{cc.betm1}'),
     [T('the band is $\\pm$@{cc.band}: yesterday\'s New York return is visible in today\'s Bucharest and Frankfurt returns, not the reverse', 'banda este $\\pm$@{cc.band}: randamentul de ieri de la New York se vede în randamentele de azi de la București și Frankfurt, nu invers'),
      T('reason: Wall Street closes at 22:00 in Frankfurt, after the European markets; its news reaches them the next morning', 'motivul: Wall Street se închide la ora 22:00 (ora Frankfurtului), după piețele europene; știrile ajung la ele a doua zi dimineață')]),
    (T('ROBOR with inflation: the largest correlation, @{cc.ipmax}, is at $k = @{cc.k}$ quarters (inflation leads)', 'ROBOR cu inflația: cea mai mare corelație, @{cc.ipmax}, apare la $k = @{cc.k}$ trimestre (inflația precedă)'),
     [T('but every value is large (@{cc.ipm4} at $k = -4$, @{cc.ip4} at $k = 4$): two persistent series; the CCF alone cannot separate lead from persistence', 'dar toate valorile sînt mari (@{cc.ipm4} la $k = -4$, @{cc.ip4} la $k = 4$): două serii persistente; CCF singură nu poate separa precedența de persistență')])])

D.frame(T('The advantage of one model for all series', 'Avantajul unui singur model pentru toate seriile'), items(
    (T('Univariate models (Chapters 2--3) forecast each series from its own past', 'Modelele univariate (Capitolele 2--3) prognozează fiecare serie din propriul trecut'),
     [T('they ignore that last quarter\'s inflation may help to forecast this quarter\'s interest rate', 'ele ignoră faptul că inflația din trimestrul trecut poate ajuta la prognoza dobînzii din acest trimestru')]),
    (T('\\textbf{Feedback}: the central bank reacts to inflation, inflation reacts to the interest rate', '\\textbf{Feedback}: banca centrală reacționează la inflație, inflația reacționează la dobîndă'),
     [T('a single regression of $\\pi_t$ on $i_t$ treats $i_t$ as given (exogenous); here both are \\textbf{endogenous}, determined inside the system', 'o singură regresie a lui $\\pi_t$ pe $i_t$ tratează $i_t$ ca fiind dat (exogen); aici ambele sînt \\textbf{endogene}, determinate în interiorul sistemului')]),
    (T('A VAR treats all variables symmetrically: each depends on the past of all', 'Un VAR tratează simetric toate variabilele: fiecare depinde de trecutul tuturor'),
     [T('no economic theory is needed to write it down; theory comes back when we interpret shocks (Section 7)', 'nu este nevoie de teorie economică pentru a-l scrie; teoria revine cînd interpretăm șocurile (secțiunea 7)')])))

D.recap(('From one series to several', 'de la o serie la mai multe'), [
    T('A vector series $\\bY_t$ has a mean vector and autocovariance matrices $\\bGamma(h)$, with $\\bGamma(-h) = \\bGamma(h)\'$', 'O serie vectorială $\\bY_t$ are un vector al mediilor și matrice de autocovarianță $\\bGamma(h)$, cu $\\bGamma(-h) = \\bGamma(h)\'$'),
    T('The CCF $\\rho_{yx}(k)$ shows lead--lag patterns; $k > 0$: $x$ leads', 'CCF $\\rho_{yx}(k)$ arată relațiile de precedență; $k > 0$: $x$ precedă'),
    T('S\\&P 500 returns lead DAX and BET returns by one day', 'Randamentele S\\&P 500 precedă cu o zi randamentele DAX și BET'),
    T('Feedback between variables calls for a joint model', 'Feedback-ul dintre variabile cere un model comun')])

# =============================================================================
# 2. MODELUL VAR(p)
# =============================================================================
D.section('The VAR$(p)$ model', 'Modelul VAR$(p)$')

D.frame(T('The VAR$(p)$ model', 'Modelul VAR$(p)$'), items(
    (T('\\textbf{Definition}: $\\bY_t = \\bc + \\bA_1\\bY_{t-1} + \\dots + \\bA_p\\bY_{t-p} + \\bepsilon_t$', '\\textbf{Definiție}: $\\bY_t = \\bc + \\bA_1\\bY_{t-1} + \\dots + \\bA_p\\bY_{t-p} + \\bepsilon_t$'),
     [T('$\\bc$: $K \\times 1$ vector of constants; $\\bA_1, \\dots, \\bA_p$: $K \\times K$ coefficient matrices', '$\\bc$: vectorul termenilor liberi, $K \\times 1$; $\\bA_1, \\dots, \\bA_p$: matricele coeficienților, $K \\times K$'),
      T('$\\bepsilon_t$: vector white noise: $E[\\bepsilon_t] = \\mathbf{0}$, $E[\\bepsilon_t\\bepsilon_t\'] = \\bSigma$, $E[\\bepsilon_t\\bepsilon_s\'] = \\mathbf{0}$ for $t \\ne s$', '$\\bepsilon_t$: zgomot alb vectorial: $E[\\bepsilon_t] = \\mathbf{0}$, $E[\\bepsilon_t\\bepsilon_t\'] = \\bSigma$, $E[\\bepsilon_t\\bepsilon_s\'] = \\mathbf{0}$ pentru $t \\ne s$')]),
    (T('$\\bSigma$ is not diagonal in general: shocks of different equations are correlated \\textbf{at the same date}', '$\\bSigma$ nu este, în general, diagonală: șocurile din ecuații diferite sînt corelate \\textbf{la aceeași dată}'),
     [T('no lagged errors are correlated: all the dynamics is in the matrices $\\bA_j$', 'erorile de la date diferite nu sînt corelate: toată dinamica se află în matricele $\\bA_j$')]),
    (T('With $K = 1$ we recover the AR$(p)$ of Chapter 2', 'Pentru $K = 1$ regăsim modelul AR$(p)$ din Capitolul 2'),
     [T('this form is called the \\textbf{reduced form}: it describes the data, not the economic shocks (Section 10)', 'această formă se numește \\textbf{forma redusă}: descrie datele, nu șocurile economice (secțiunea 10)')])))

D.frame(T('A bivariate VAR(1), equation by equation', 'Un VAR(1) cu două variabile, ecuație cu ecuație'), items(
    (T('In matrix form:', 'În formă matriceală:'),
     [T('$\\begin{pmatrix} y_{1t} \\\\ y_{2t} \\end{pmatrix} = \\begin{pmatrix} c_1 \\\\ c_2 \\end{pmatrix} + \\begin{pmatrix} a_{11} & a_{12} \\\\ a_{21} & a_{22} \\end{pmatrix}\\begin{pmatrix} y_{1,t-1} \\\\ y_{2,t-1} \\end{pmatrix} + \\begin{pmatrix} \\varepsilon_{1t} \\\\ \\varepsilon_{2t} \\end{pmatrix}$',
        '$\\begin{pmatrix} y_{1t} \\\\ y_{2t} \\end{pmatrix} = \\begin{pmatrix} c_1 \\\\ c_2 \\end{pmatrix} + \\begin{pmatrix} a_{11} & a_{12} \\\\ a_{21} & a_{22} \\end{pmatrix}\\begin{pmatrix} y_{1,t-1} \\\\ y_{2,t-1} \\end{pmatrix} + \\begin{pmatrix} \\varepsilon_{1t} \\\\ \\varepsilon_{2t} \\end{pmatrix}$')]),
    (T('Equation by equation:', 'Ecuație cu ecuație:'),
     [T('$y_{1t} = c_1 + a_{11}y_{1,t-1} + a_{12}y_{2,t-1} + \\varepsilon_{1t}$', '$y_{1t} = c_1 + a_{11}y_{1,t-1} + a_{12}y_{2,t-1} + \\varepsilon_{1t}$'),
      T('$y_{2t} = c_2 + a_{21}y_{1,t-1} + a_{22}y_{2,t-1} + \\varepsilon_{2t}$', '$y_{2t} = c_2 + a_{21}y_{1,t-1} + a_{22}y_{2,t-1} + \\varepsilon_{2t}$')]),
    (T('Reading the coefficients: $a_{jk}$ = effect of $y_{k,t-1}$ on $y_{jt}$ (row: equation; column: regressor)', 'Interpretarea coeficienților: $a_{jk}$ = efectul lui $y_{k,t-1}$ asupra lui $y_{jt}$ (rîndul: ecuația; coloana: regresorul)'),
     [T('$a_{12} = a_{21} = 0$: two separate AR(1) models; $a_{12} \\ne 0$: the past of $y_2$ helps to predict $y_1$', '$a_{12} = a_{21} = 0$: două modele AR(1) separate; $a_{12} \\ne 0$: trecutul lui $y_2$ ajută la prognoza lui $y_1$')])))

D.frame(T('Worked example: a VAR(1)', 'Exemplu rezolvat: un VAR(1)'), items(
    (T('Model: $\\bc = (1.2, 0.6)\'$, $\\bA = \\begin{pmatrix} 0.5 & 0.2 \\\\ 0.3 & 0.4 \\end{pmatrix}$, $\\bSigma = \\begin{pmatrix} 1 & 0.5 \\\\ 0.5 & 1 \\end{pmatrix}$',
       'Modelul: $\\bc = (1{,}2;\\ 0{,}6)\'$, $\\bA = \\begin{pmatrix} 0{,}5 & 0{,}2 \\\\ 0{,}3 & 0{,}4 \\end{pmatrix}$, $\\bSigma = \\begin{pmatrix} 1 & 0{,}5 \\\\ 0{,}5 & 1 \\end{pmatrix}$'),
     [T('$y_{1t} = 1.2 + 0.5y_{1,t-1} + 0.2y_{2,t-1} + \\varepsilon_{1t}$; \\quad $y_{2t} = 0.6 + 0.3y_{1,t-1} + 0.4y_{2,t-1} + \\varepsilon_{2t}$', '$y_{1t} = 1{,}2 + 0{,}5y_{1,t-1} + 0{,}2y_{2,t-1} + \\varepsilon_{1t}$; \\quad $y_{2t} = 0{,}6 + 0{,}3y_{1,t-1} + 0{,}4y_{2,t-1} + \\varepsilon_{2t}$')]),
    (T('Reading: $a_{21} = 0.3$: one unit more of $y_1$ last period raises $y_2$ by 0.3 today, other things fixed', 'Interpretare: $a_{21} = 0{,}3$: o unitate în plus a lui $y_1$ în perioada trecută crește $y_2$ cu 0,3 azi, celelalte fiind fixe'),
     [T('the shocks are correlated: $\\Corr(\\varepsilon_{1t}, \\varepsilon_{2t}) = 0.5$', 'șocurile sînt corelate: $\\Corr(\\varepsilon_{1t}, \\varepsilon_{2t}) = 0{,}5$')]),
    (T('Forecasts from $\\bY_T = (5, 2)\'$: $\\hat\\bY_{T+1} = \\bc + \\bA\\bY_T = @{wx.f1}$, \\quad $\\hat\\bY_{T+2} = \\bc + \\bA\\hat\\bY_{T+1} = @{wx.f2}$',
       'Prognoze din $\\bY_T = (5;\\ 2)\'$: $\\hat\\bY_{T+1} = \\bc + \\bA\\bY_T = @{wx.f1}$, \\quad $\\hat\\bY_{T+2} = \\bc + \\bA\\hat\\bY_{T+1} = @{wx.f2}$'),
     [T('the forecasts move towards the mean $\\boldsymbol{\\mu} = (@{wx.mu1}, @{wx.mu2})\'$ (next section)', 'prognozele se îndreaptă spre medie, $\\boldsymbol{\\mu} = (@{wx.mu1}, @{wx.mu2})\'$ (secțiunea următoare)')])))

chart(T('The worked example, simulated', 'Exemplul rezolvat, simulat'), 'tsa_ch6_var_sim', 'TSA_ch6_var_stability', [
    T('Left: 200 periods of the VAR(1) of the previous slide, with its two means; right: the responses of $y_1$ and $y_2$ to a unit shock in $\\varepsilon_1$ at $h = 0$, the columns of $\\bA^h$',
      'Stînga: 200 de perioade din modelul VAR(1) de pe slide-ul anterior, cu cele două medii; dreapta: răspunsurile lui $y_1$ și $y_2$ la un șoc unitar în $\\varepsilon_1$ la $h = 0$, coloanele lui $\\bA^h$')],
    h='0.56\\textheight')

interp(('the simulated VAR(1)', 'VAR(1) simulat'), [
    (T('The two series wander around their means and move together: sample correlation @{sim.c}', 'Cele două serii oscilează în jurul mediilor și evoluează împreună: corelația de eșantion este @{sim.c}'),
     [T('two sources of co-movement: the correlated shocks ($\\Corr = 0.5$) and the cross effects $a_{12}$, $a_{21}$', 'două surse ale evoluției comune: șocurile corelate ($\\Corr = 0{,}5$) și efectele încrucișate $a_{12}$, $a_{21}$')]),
    (T('A unit shock to $y_1$ reaches $y_2$ one period later ($a_{21} = 0.3$), then both fade', 'Un șoc unitar în $y_1$ ajunge la $y_2$ după o perioadă ($a_{21} = 0{,}3$), apoi ambele se sting'),
     [T('the decay is governed by the largest eigenvalue of $\\bA$, @{wx.l1}: after 5 periods about $0.7^5 \\approx 0.17$ of the effect remains', 'stingerea este guvernată de cea mai mare valoare proprie a lui $\\bA$, @{wx.l1}: după 5 perioade rămîne circa $0{,}7^5 \\approx 0{,}17$ din efect')])])

D.frame(T('The number of parameters', 'Numărul de parametri'), items(
    (T('A VAR$(p)$ with $K$ variables and a constant: $K + pK^2$ coefficients, plus $K(K+1)/2$ in $\\bSigma$', 'Un VAR$(p)$ cu $K$ variabile și termen liber: $K + pK^2$ coeficienți, plus $K(K+1)/2$ în $\\bSigma$'),
     [T('each equation has $1 + pK$ regressors', 'fiecare ecuație are $1 + pK$ regresori'),
      T('Romanian VAR: $K = 3$, $p = 2$: $3 + 2 \\cdot 9 = @{ex.k3p2}$ coefficients estimated on @{e.n} quarters', 'VAR-ul românesc: $K = 3$, $p = 2$: $3 + 2 \\cdot 9 = @{ex.k3p2}$ de coeficienți estimați pe @{e.n} de trimestre')]),
    (T('\\textbf{The curse of dimensionality}: $K = 5$, $p = 4$: $5 + 4 \\cdot 25 = @{ex.k5p4}$ coefficients', '\\textbf{Blestemul dimensionalității}: $K = 5$, $p = 4$: $5 + 4 \\cdot 25 = @{ex.k5p4}$ de coeficienți'),
     [T('with 25 years of quarterly data (100 observations) each equation has 21 regressors and 79 residual degrees of freedom', 'cu 25 de ani de date trimestriale (100 de observații) fiecare ecuație are 21 de regresori și 79 de grade de libertate reziduale')]),
    T('Hence small VARs (3--6 variables), few lags, and estimates with large standard errors; individual coefficients are rarely interpreted', 'De aici VAR-uri mici (3--6 variabile), puține laguri și estimări cu erori standard mari; coeficienții individuali se interpretează rar')))

D.frame(T('Christopher Sims and "Macroeconomics and Reality"', 'Christopher Sims și „Macroeconomics and Reality”'), '\\begin{columns}[T]\n\\begin{column}{0.38\\textwidth}\n'
        + ph('sims', T('Christopher A. Sims, Nobel Prize in Economics 2011', 'Christopher A. Sims, Premiul Nobel pentru economie 2011'), h='0.56\\textheight')
        + '\n\\end{column}\n\\begin{column}{0.6\\textwidth}\n' + items(
            (T('\\refSims: the large macro models of the 1970s imposed ``incredible\'\' restrictions', '\\refSims: marile modele macroeconomice din anii 1970 impuneau restricții „incredibile”'),
             [T('which variables are exogenous, which lags are zero', 'ce variabile sînt exogene, ce laguri sînt nule'),
              T('his proposal: an unrestricted VAR that lets the data speak', 'propunerea lui: un VAR nerestricționat, fără restricții impuse a priori'),
              T('then only the few assumptions needed to interpret the shocks', 'apoi doar ipotezele minime necesare pentru interpretarea șocurilor')]),
            T('The VAR became the standard tool of central banks for forecasting and for measuring the effects of monetary policy', 'VAR-ul a devenit instrumentul standard al băncilor centrale pentru prognoză și pentru măsurarea efectelor politicii monetare'),
            T('Nobel Prize 2011, with Thomas Sargent, ``for their empirical research on cause and effect in the macroeconomy\'\'', 'Premiul Nobel 2011, împreună cu Thomas Sargent, „pentru cercetarea empirică a cauzei și efectului în macroeconomie”'))
        + '\n\\end{column}\n\\end{columns}')

D.recap(('The VAR$(p)$ model', 'modelul VAR$(p)$'), [
    T('$\\bY_t = \\bc + \\sum_{j=1}^{p}\\bA_j\\bY_{t-j} + \\bepsilon_t$, $\\bepsilon_t$ white noise with covariance $\\bSigma$', '$\\bY_t = \\bc + \\sum_{j=1}^{p}\\bA_j\\bY_{t-j} + \\bepsilon_t$, $\\bepsilon_t$ zgomot alb cu covarianța $\\bSigma$'),
    T('$a_{jk}$: effect of the lag of variable $k$ in the equation of variable $j$', '$a_{jk}$: efectul lagului variabilei $k$ în ecuația variabilei $j$'),
    T('$K + pK^2$ coefficients: keep $K$ and $p$ small', '$K + pK^2$ coeficienți: păstrăm $K$ și $p$ mici'),
    T('\\refSims: VARs as an alternative to ``incredible\'\' restrictions', '\\refSims: VAR-ul ca alternativă la restricțiile „incredibile”')])

# =============================================================================
# 3. STABILITATE
# =============================================================================
D.section('Stationarity and the companion form', 'Staționaritate și forma companion')

D.frame(T('Stability of a VAR(1)', 'Stabilitatea unui VAR(1)'), items(
    (T('Substitute repeatedly: $\\bY_t = \\bc + \\bA\\bc + \\dots + \\bA^{h-1}\\bc + \\bA^h\\bY_{t-h} + \\sum_{j=0}^{h-1}\\bA^j\\bepsilon_{t-j}$', 'Substituim repetat: $\\bY_t = \\bc + \\bA\\bc + \\dots + \\bA^{h-1}\\bc + \\bA^h\\bY_{t-h} + \\sum_{j=0}^{h-1}\\bA^j\\bepsilon_{t-j}$'),
     [T('the past is forgotten only if $\\bA^h \\to \\mathbf{0}$', 'trecutul este uitat doar dacă $\\bA^h \\to \\mathbf{0}$')]),
    (T('\\textbf{Stability condition}: all eigenvalues $\\lambda$ of $\\bA$ satisfy $|\\lambda| < 1$', '\\textbf{Condiția de stabilitate}: toate valorile proprii $\\lambda$ ale lui $\\bA$ au $|\\lambda| < 1$'),
     [T('eigenvalues: the roots of $\\det(\\bA - \\lambda\\mathbf{I}) = 0$; for $K = 2$: $\\lambda^2 - \\mathrm{tr}(\\bA)\\lambda + \\det(\\bA) = 0$', 'valorile proprii: rădăcinile ecuației $\\det(\\bA - \\lambda\\mathbf{I}) = 0$; pentru $K = 2$: $\\lambda^2 - \\mathrm{tr}(\\bA)\\lambda + \\det(\\bA) = 0$'),
      T('$\\mathbf{I}$: the identity matrix; $\\mathrm{tr}(\\bA) = a_{11} + a_{22}$: the trace; $\\det(\\bA) = a_{11}a_{22} - a_{12}a_{21}$: the determinant; $|\\lambda|$: the modulus (eigenvalues can be complex)', '$\\mathbf{I}$: matricea identitate; $\\mathrm{tr}(\\bA) = a_{11} + a_{22}$: urma; $\\det(\\bA) = a_{11}a_{22} - a_{12}a_{21}$: determinantul; $|\\lambda|$: modulul (valorile proprii pot fi complexe)'),
      T('a stable VAR is weakly stationary (if it started long ago); this is the multivariate version of $|\\phi| < 1$ (Chapter 2)', 'un VAR stabil este slab staționar (dacă a pornit demult); aceasta este versiunea multivariată a condiției $|\\phi| < 1$ (Capitolul 2)')]),
    (T('\\textbf{Example}: $\\mathrm{tr}(\\bA) = @{wx.tr}$, $\\det(\\bA) = 0.5 \\cdot 0.4 - 0.2 \\cdot 0.3 = @{wx.dA}$: $\\lambda^2 - 0.9\\lambda + 0.14 = 0$', '\\textbf{Exemplu}: $\\mathrm{tr}(\\bA) = @{wx.tr}$, $\\det(\\bA) = 0{,}5 \\cdot 0{,}4 - 0{,}2 \\cdot 0{,}3 = @{wx.dA}$: $\\lambda^2 - 0{,}9\\lambda + 0{,}14 = 0$'),
     [T('$\\lambda = (0.9 \\pm 0.5)/2$: $\\lambda_1 = @{wx.l1}$, $\\lambda_2 = @{wx.l2}$; both inside the unit circle: stable', '$\\lambda = (0{,}9 \\pm 0{,}5)/2$: $\\lambda_1 = @{wx.l1}$, $\\lambda_2 = @{wx.l2}$; ambele în interiorul cercului unitate: stabil')])))

D.frame(T('Mean and variance of a stable VAR(1)', 'Media și varianța unui VAR(1) stabil'), items(
    (T('\\textbf{Mean}: take expectations, $\\boldsymbol{\\mu} = \\bc + \\bA\\boldsymbol{\\mu}$, so $\\boldsymbol{\\mu} = (\\mathbf{I} - \\bA)^{-1}\\bc$', '\\textbf{Media}: luăm speranța matematică în ambii membri, $\\boldsymbol{\\mu} = \\bc + \\bA\\boldsymbol{\\mu}$, deci $\\boldsymbol{\\mu} = (\\mathbf{I} - \\bA)^{-1}\\bc$'),
     [T('example: $\\mathbf{I} - \\bA = \\begin{pmatrix} 0.5 & -0.2 \\\\ -0.3 & 0.6 \\end{pmatrix}$, determinant @{wx.det}; $\\boldsymbol{\\mu} = \\frac{1}{0.24}\\begin{pmatrix} 0.6 & 0.2 \\\\ 0.3 & 0.5 \\end{pmatrix}\\begin{pmatrix} 1.2 \\\\ 0.6 \\end{pmatrix} = \\begin{pmatrix} @{wx.mu1} \\\\ @{wx.mu2} \\end{pmatrix}$',
        'exemplu: $\\mathbf{I} - \\bA = \\begin{pmatrix} 0{,}5 & -0{,}2 \\\\ -0{,}3 & 0{,}6 \\end{pmatrix}$, determinantul @{wx.det}; $\\boldsymbol{\\mu} = \\frac{1}{0{,}24}\\begin{pmatrix} 0{,}6 & 0{,}2 \\\\ 0{,}3 & 0{,}5 \\end{pmatrix}\\begin{pmatrix} 1{,}2 \\\\ 0{,}6 \\end{pmatrix} = \\begin{pmatrix} @{wx.mu1} \\\\ @{wx.mu2} \\end{pmatrix}$')]),
    (T('\\textbf{Variance}: $\\bGamma(0) = \\bA\\bGamma(0)\\bA\' + \\bSigma$ (a discrete Lyapunov equation); $\\bGamma(h) = \\bA^h\\bGamma(0)$', '\\textbf{Varianța}: $\\bGamma(0) = \\bA\\bGamma(0)\\bA\' + \\bSigma$ (o ecuație Lyapunov discretă); $\\bGamma(h) = \\bA^h\\bGamma(0)$'),
     [T('solved by $\\mathrm{vec}\\,\\bGamma(0) = (\\mathbf{I} - \\bA \\otimes \\bA)^{-1}\\mathrm{vec}\\,\\bSigma$; example: $\\bGamma(0) = @{wx.G0}$', 'se rezolvă prin $\\mathrm{vec}\\,\\bGamma(0) = (\\mathbf{I} - \\bA \\otimes \\bA)^{-1}\\mathrm{vec}\\,\\bSigma$; exemplu: $\\bGamma(0) = @{wx.G0}$'),
      T('$\\otimes$: the Kronecker product; $\\mathrm{vec}$ stacks the columns of a matrix', '$\\otimes$: produsul Kronecker; $\\mathrm{vec}$ așază coloanele unei matrice una sub alta')]),
    T('The variances @{wx.g11} and @{wx.g22} exceed those of the shocks (1): the dynamics accumulates past shocks', 'Varianțele @{wx.g11} și @{wx.g22} depășesc varianța șocurilor (1): dinamica acumulează șocurile trecute')))

D.frame(T('The companion form of a VAR$(p)$', 'Forma companion a unui VAR$(p)$'), items(
    (T('Stack $p$ consecutive vectors: $\\boldsymbol{\\xi}_t = (\\bY_t\', \\bY_{t-1}\', \\dots, \\bY_{t-p+1}\')\'$, of size $Kp$', 'Grupăm $p$ vectori consecutivi: $\\boldsymbol{\\xi}_t = (\\bY_t\', \\bY_{t-1}\', \\dots, \\bY_{t-p+1}\')\'$, de dimensiune $Kp$'),
     [T('then $\\boldsymbol{\\xi}_t = \\boldsymbol{\\nu} + \\mathbf{F}\\boldsymbol{\\xi}_{t-1} + \\mathbf{v}_t$: every VAR$(p)$ is a VAR(1) in $\\boldsymbol{\\xi}_t$', 'atunci $\\boldsymbol{\\xi}_t = \\boldsymbol{\\nu} + \\mathbf{F}\\boldsymbol{\\xi}_{t-1} + \\mathbf{v}_t$: orice VAR$(p)$ este un VAR(1) în $\\boldsymbol{\\xi}_t$'),
      T('$\\boldsymbol{\\nu}$: $\\bc$ followed by zeros; $\\mathbf{v}_t$: $\\bepsilon_t$ followed by zeros; $\\mathbf{F}$: the $Kp \\times Kp$ companion matrix', '$\\boldsymbol{\\nu}$: $\\bc$ urmat de zerouri; $\\mathbf{v}_t$: $\\bepsilon_t$ urmat de zerouri; $\\mathbf{F}$: matricea companion, $Kp \\times Kp$')]),
    (T('For $p = 2$: $\\begin{pmatrix} \\bY_t \\\\ \\bY_{t-1} \\end{pmatrix} = \\begin{pmatrix} \\bc \\\\ \\mathbf{0} \\end{pmatrix} + \\underbrace{\\begin{pmatrix} \\bA_1 & \\bA_2 \\\\ \\mathbf{I}_K & \\mathbf{0} \\end{pmatrix}}_{\\mathbf{F}}\\begin{pmatrix} \\bY_{t-1} \\\\ \\bY_{t-2} \\end{pmatrix} + \\begin{pmatrix} \\bepsilon_t \\\\ \\mathbf{0} \\end{pmatrix}$',
       'Pentru $p = 2$: $\\begin{pmatrix} \\bY_t \\\\ \\bY_{t-1} \\end{pmatrix} = \\begin{pmatrix} \\bc \\\\ \\mathbf{0} \\end{pmatrix} + \\underbrace{\\begin{pmatrix} \\bA_1 & \\bA_2 \\\\ \\mathbf{I}_K & \\mathbf{0} \\end{pmatrix}}_{\\mathbf{F}}\\begin{pmatrix} \\bY_{t-1} \\\\ \\bY_{t-2} \\end{pmatrix} + \\begin{pmatrix} \\bepsilon_t \\\\ \\mathbf{0} \\end{pmatrix}$'),
     [T('the second block row is the identity $\\bY_{t-1} = \\bY_{t-1}$', 'al doilea bloc de rînduri este identitatea $\\bY_{t-1} = \\bY_{t-1}$')]),
    (T('\\textbf{Stability of a VAR$(p)$}: all $Kp$ eigenvalues of the companion matrix $\\mathbf{F}$ have modulus below 1', '\\textbf{Stabilitatea unui VAR$(p)$}: toate cele $Kp$ valori proprii ale matricei companion $\\mathbf{F}$ au modulul sub 1'),
     [T('equivalently: all roots $z$ (complex numbers) of $\\det(\\mathbf{I}_K - \\bA_1z - \\dots - \\bA_pz^p) = 0$ lie outside the unit circle', 'echivalent: toate rădăcinile $z$ (numere complexe) ale ecuației $\\det(\\mathbf{I}_K - \\bA_1z - \\dots - \\bA_pz^p) = 0$ sînt în afara cercului unitate'),
      T('the eigenvalues of $\\bA_1$ alone say nothing about the stability of a VAR(2)', 'valorile proprii ale lui $\\bA_1$ singure nu spun nimic despre stabilitatea unui VAR(2)')])))

chart(T('Eigenvalues of three estimated VARs', 'Valorile proprii a trei VAR-uri estimate'), 'tsa_ch6_roots', 'TSA_ch6_var_stability', [
    T('Companion matrices: Romanian VAR(2) (GDP growth, inflation, ROBOR 3M; 6 eigenvalues), the US VAR(4) of Section 10 (12), the daily S\\&P 500, DAX and BET VAR(4) (12)',
      'Matricele companion: VAR(2) românesc (creșterea PIB-ului, inflația, ROBOR 3M; 6 valori proprii), VAR(4) american din secțiunea 10 (12), VAR(4) zilnic pentru S\\&P 500, DAX și BET (12)')],
    h='0.6\\textheight')

interp(('the eigenvalues', 'valorilor proprii'), [
    (T('All three VARs are stable; largest moduli: Romania @{rt.ro}, US @{rt.us}, markets @{rt.mk}', 'Toate cele trei VAR-uri sînt stabile; cele mai mari module: România @{rt.ro}, SUA @{rt.us}, piețe @{rt.mk}'),
     [T('the largest modulus sets the speed at which shocks fade: $0.97^{20} \\approx 0.54$ after 20 quarters, $0.46^{2} \\approx 0.21$ after 2 days', 'cel mai mare modul stabilește viteza cu care se sting șocurile: $0{,}97^{20} \\approx 0{,}54$ după 20 de trimestre, $0{,}46^{2} \\approx 0{,}21$ după 2 zile')]),
    (T('Macro VARs have eigenvalues close to 1 (persistent inflation and interest rates); return VARs have small ones', 'VAR-urile macroeconomice au valori proprii apropiate de 1 (inflație și dobînzi persistente); VAR-urile de randamente au valori mici'),
     [T('complex pairs produce damped oscillations in the responses', 'perechile complexe produc oscilații amortizate în răspunsuri')]),
    T('An eigenvalue very close to 1 signals a unit root, perhaps cointegration: Chapter 7', 'O valoare proprie foarte apropiată de 1 semnalează o rădăcină unitară, poate cointegrare: Capitolul 7')])

D.frame(T('The moving-average representation', 'Reprezentarea de medie mobilă'), items(
    (T('A stable VAR can be written as an infinite moving average of the shocks (Wold, Chapter 1):', 'Un VAR stabil se poate scrie ca o medie mobilă infinită a șocurilor (Wold, Capitolul 1):'),
     [T('$\\bY_t = \\boldsymbol{\\mu} + \\bepsilon_t + \\bPhi_1\\bepsilon_{t-1} + \\bPhi_2\\bepsilon_{t-2} + \\dots = \\boldsymbol{\\mu} + \\sum_{h=0}^{\\infty}\\bPhi_h\\bepsilon_{t-h}$, $\\bPhi_0 = \\mathbf{I}$', '$\\bY_t = \\boldsymbol{\\mu} + \\bepsilon_t + \\bPhi_1\\bepsilon_{t-1} + \\bPhi_2\\bepsilon_{t-2} + \\dots = \\boldsymbol{\\mu} + \\sum_{h=0}^{\\infty}\\bPhi_h\\bepsilon_{t-h}$, $\\bPhi_0 = \\mathbf{I}$'),
      T('$\\bPhi_h$: the $K \\times K$ matrix that weights the shock of $h$ periods ago', '$\\bPhi_h$: matricea $K \\times K$ care ponderează șocul de acum $h$ perioade')]),
    (T('VAR(1): $\\bPhi_h = \\bA^h$; VAR$(p)$: $\\bPhi_h = \\sum_{j=1}^{\\min(h,p)}\\bA_j\\bPhi_{h-j}$', 'VAR(1): $\\bPhi_h = \\bA^h$; VAR$(p)$: $\\bPhi_h = \\sum_{j=1}^{\\min(h,p)}\\bA_j\\bPhi_{h-j}$'),
     [T('example: $\\bPhi_2 = \\bA^2 = @{wx.A2}$', 'exemplu: $\\bPhi_2 = \\bA^2 = @{wx.A2}$')]),
    (T('Element $(j, k)$ of $\\bPhi_h$: the effect on $Y_{j,t+h}$ of a unit change in $\\varepsilon_{kt}$, all other shocks fixed', 'Elementul $(j, k)$ al lui $\\bPhi_h$: efectul asupra lui $Y_{j,t+h}$ al unei modificări unitare a lui $\\varepsilon_{kt}$, celelalte șocuri fiind fixe'),
     [T('these are the \\textbf{impulse responses} (Section 7); they also give the forecast errors (Sections 8--9)', 'acestea sînt \\textbf{răspunsurile la impuls} (secțiunea 7); tot din ele obținem erorile de prognoză (secțiunile 8--9)')])))

D.recap(('Stationarity and the companion form', 'staționaritate și forma companion'), [
    T('VAR(1): stable if all eigenvalues of $\\bA$ have $|\\lambda| < 1$; VAR$(p)$: the same for the companion matrix', 'VAR(1): stabil dacă toate valorile proprii ale lui $\\bA$ au $|\\lambda| < 1$; VAR$(p)$: aceeași condiție pentru matricea companion'),
    T('Mean $\\boldsymbol{\\mu} = (\\mathbf{I} - \\bA_1 - \\dots - \\bA_p)^{-1}\\bc$; forecasts converge to it', 'Media $\\boldsymbol{\\mu} = (\\mathbf{I} - \\bA_1 - \\dots - \\bA_p)^{-1}\\bc$; prognozele converg spre ea'),
    T('Moving-average form $\\bY_t = \\boldsymbol{\\mu} + \\sum_h\\bPhi_h\\bepsilon_{t-h}$: the basis of impulse responses', 'Forma de medie mobilă $\\bY_t = \\boldsymbol{\\mu} + \\sum_h\\bPhi_h\\bepsilon_{t-h}$: baza răspunsurilor la impuls'),
    T('Estimated macro VARs have eigenvalues close to 1', 'VAR-urile macroeconomice estimate au valori proprii apropiate de 1')])

# =============================================================================
# 4. ESTIMARE ȘI ALEGEREA LUI p
# =============================================================================
D.section('Estimation and lag selection', 'Estimare și alegerea numărului de laguri')

D.frame(T('Estimation by OLS, equation by equation', 'Estimarea prin OLS, ecuație cu ecuație'), items(
    (T('Each equation is a linear regression of $Y_{jt}$ on a constant and $\\bY_{t-1}, \\dots, \\bY_{t-p}$', 'Fiecare ecuație este o regresie liniară a lui $Y_{jt}$ pe o constantă și pe $\\bY_{t-1}, \\dots, \\bY_{t-p}$'),
     [T('the same $1 + Kp$ regressors in every equation; OLS (ordinary least squares) on each equation separately', 'aceiași $1 + Kp$ regresori în fiecare ecuație; OLS (ordinary least squares, metoda celor mai mici pătrate) pe fiecare ecuație separat')]),
    (T('\\textbf{OLS is enough}: with identical regressors, system estimation (GLS, generalised least squares, on all equations) gives exactly the OLS estimates', '\\textbf{OLS este suficient}: cu regresori identici, estimarea sistemului (GLS, generalized least squares, metoda celor mai mici pătrate generalizate, pe toate ecuațiile) dă exact estimările OLS'),
     [T('the correlation of the shocks does not improve the estimates; under Normal errors OLS is also the maximum likelihood estimator', 'corelația șocurilor nu îmbunătățește estimările; cu erori Normale, OLS este și estimatorul de verosimilitate maximă')]),
    (T('\\textbf{Covariance of the shocks}: $\\hat\\bSigma = \\frac{1}{T - Kp - 1}\\sum_t\\hat\\bepsilon_t\\hat\\bepsilon_t\'$, from the residuals of all equations', '\\textbf{Covarianța șocurilor}: $\\hat\\bSigma = \\frac{1}{T - Kp - 1}\\sum_t\\hat\\bepsilon_t\\hat\\bepsilon_t\'$, din reziduurile tuturor ecuațiilor'),
     [T('$\\hat\\bepsilon_t$: the vector of residuals at date $t$; $T - Kp - 1$: the residual degrees of freedom of each equation', '$\\hat\\bepsilon_t$: vectorul reziduurilor la data $t$; $T - Kp - 1$: gradele de libertate reziduale ale fiecărei ecuații'),
      T('stationarity matters: with stationary series the usual $t$ and $F$ tests are valid in large samples', 'staționaritatea contează: pentru serii staționare testele $t$ și $F$ obișnuite sînt valide în eșantioane mari')])))

D.frame(T('Choosing $p$: information criteria', 'Alegerea lui $p$: criterii informaționale'), items(
    (T('Fit VAR$(p)$ for $p = 0, \\dots, p_{\\max}$ on the \\textbf{same sample}; $\\tilde\\bSigma(p)$: residual covariance divided by $T$', 'Estimăm VAR$(p)$ pentru $p = 0, \\dots, p_{\\max}$ pe \\textbf{același eșantion}; $\\tilde\\bSigma(p)$: covarianța reziduurilor împărțită la $T$'),
     [T('AIC (Akaike) $= \\ln\\det\\tilde\\bSigma(p) + \\frac{2}{T}pK^2$ \\refAkaike', 'AIC (Akaike) $= \\ln\\det\\tilde\\bSigma(p) + \\frac{2}{T}pK^2$ \\refAkaike'),
      T('BIC (Schwarz) $= \\ln\\det\\tilde\\bSigma(p) + \\frac{\\ln T}{T}pK^2$ \\refSchwarz', 'BIC (Schwarz) $= \\ln\\det\\tilde\\bSigma(p) + \\frac{\\ln T}{T}pK^2$ \\refSchwarz'),
      T('HQ (Hannan--Quinn) $= \\ln\\det\\tilde\\bSigma(p) + \\frac{2\\ln\\ln T}{T}pK^2$ \\refHQ', 'HQ (Hannan--Quinn) $= \\ln\\det\\tilde\\bSigma(p) + \\frac{2\\ln\\ln T}{T}pK^2$ \\refHQ'),
      T('$\\det\\tilde\\bSigma(p)$: a generalised residual variance; $pK^2$: the number of lag coefficients; $p_{\\max}$: the largest order tried', '$\\det\\tilde\\bSigma(p)$: o varianță generalizată a reziduurilor; $pK^2$: numărul coeficienților lagurilor; $p_{\\max}$: ordinul maxim încercat')]),
    (T('First term: fit (smaller residual ``volume\'\'); second: a penalty per coefficient; choose the $p$ with the smallest value', 'Primul termen: calitatea ajustării („volumul” reziduurilor mai mic); al doilea: o penalizare pe coeficient; alegem $p$ cu cea mai mică valoare'),
     [T('penalties: AIC $<$ HQ $<$ BIC (for $T \\ge 16$): BIC chooses the smallest $p$, AIC the largest', 'penalizările: AIC $<$ HQ $<$ BIC (pentru $T \\ge 16$): BIC alege cel mai mic $p$, AIC cel mai mare'),
      T('BIC and HQ are consistent (they find the true $p$ as $T \\to \\infty$); AIC tends to overfit, but often forecasts well', 'BIC și HQ sînt consistente (găsesc $p$ adevărat cînd $T \\to \\infty$); AIC tinde să supraajusteze, dar prognozează adesea bine')])))

chart(T('Romanian VAR: which $p$?', 'VAR-ul românesc: alegerea lui $p$'), 'tsa_ch6_ic', 'TSA_ch6_estimation_diagnostics', [
    T('GDP growth, 12-month inflation and ROBOR 3M, @{ro.q0}--@{ro.q1}; VAR$(p)$ for $p = 0, \\dots, @{ic.pmax}$ on the common sample of @{ic.T} quarters; circles mark the minima',
      'Creșterea PIB-ului, inflația anuală și ROBOR 3M, @{ro.q0}--@{ro.q1}; VAR$(p)$ pentru $p = 0, \\dots, @{ic.pmax}$ pe eșantionul comun de @{ic.T} de trimestre; cercurile marchează minimele')],
    h='0.56\\textheight')

interp(('the lag selection', 'alegerii numărului de laguri'), [
    (T('BIC chooses $p = @{ic.b.bic}$, HQ $p = @{ic.b.hq}$, AIC $p = @{ic.b.aic}$: the three criteria disagree, as they often do in short samples', 'BIC alege $p = @{ic.b.bic}$, HQ $p = @{ic.b.hq}$, AIC $p = @{ic.b.aic}$: cele trei criterii nu coincid, cum se întîmplă adesea în eșantioane scurte'),
     [T('AIC: @{ic.1.aic} at $p = 1$, @{ic.2.aic} at $p = 2$, @{ic.5.aic} at $p = 5$: a flat curve after $p = 2$', 'AIC: @{ic.1.aic} pentru $p = 1$, @{ic.2.aic} pentru $p = 2$, @{ic.5.aic} pentru $p = 5$: o curbă plată după $p = 2$')]),
    (T('$p = 5$ would mean $3 + 5 \\cdot 9 = 48$ coefficients for about 80 quarters', '$p = 5$ ar însemna $3 + 5 \\cdot 9 = 48$ de coeficienți pentru circa 80 de trimestre'),
     [T('we take $p = 2$ (HQ), then check whether the residuals are white noise; if not, we add lags', 'alegem $p = 2$ (HQ), apoi verificăm dacă reziduurile sînt zgomot alb; dacă nu, adăugăm laguri')]),
    T('A rule of thumb: start from the criteria, never stop at them', 'O regulă practică: pornim de la criterii, dar nu ne oprim la ele')])

D.frame(T('The estimated Romanian VAR(2)', 'VAR(2) românesc estimat'), table(
    'lrrr', T('\\textbf{Regressor}', '\\textbf{Regresorul}') + ' & ' + T('\\textbf{equation $g_t$}', '\\textbf{ecuația $g_t$}') + ' & ' + T('\\textbf{equation $\\pi_t$}', '\\textbf{ecuația $\\pi_t$}') + ' & ' + T('\\textbf{equation $i_t$}', '\\textbf{ecuația $i_t$}'),
    [T('constant', 'termen liber') + ' & ' + ' & '.join(f'$@{{e.{c}.const}}$ ($@{{e.{c}.const.t}}$)' for c in ['g', 'pi', 'i'])]
    + [f'${lab}$ & ' + ' & '.join(f'$@{{e.{c}.{k}}}$ ($@{{e.{c}.{k}.t}}$)' for c in ['g', 'pi', 'i'])
       for k, lab in [('L1g', 'g_{t-1}'), ('L1pi', '\\pi_{t-1}'), ('L1i', 'i_{t-1}'), ('L2g', 'g_{t-2}'), ('L2pi', '\\pi_{t-2}'), ('L2i', 'i_{t-2}')]]
    + ['$R^2$ & @{e.r2.g} & @{e.r2.pi} & @{e.r2.i}'],
    size='scriptsize') + items(
    T('OLS on @{e.n} quarters (@{e.q0}--@{ro.q1}); $t$-statistics in parentheses; @{e.k} coefficients', 'OLS pe @{e.n} de trimestre (@{e.q0}--@{ro.q1}); statisticile $t$ între paranteze; @{e.k} de coeficienți'),
    T('Residual standard deviations: @{e.sg} ($g$), @{e.spi} ($\\pi$), @{e.si} ($i$); residual correlation of $\\pi$ and $i$: @{e.cpii}', 'Abaterile standard ale reziduurilor: @{e.sg} ($g$), @{e.spi} ($\\pi$), @{e.si} ($i$); corelația reziduurilor lui $\\pi$ și $i$: @{e.cpii}')),
    'small')

interp(('the coefficients', 'coeficienților'), [
    (T('Inflation is very persistent: $@{e.pi.L1pi}\\,\\pi_{t-1} @{e.pi.L2pi}\\,\\pi_{t-2}$; ROBOR too: $@{e.i.L1i}\\,i_{t-1}$', 'Inflația este foarte persistentă: $@{e.pi.L1pi}\\,\\pi_{t-1} @{e.pi.L2pi}\\,\\pi_{t-2}$; la fel ROBOR: $@{e.i.L1i}\\,i_{t-1}$'),
     [T('$R^2$ of @{e.r2.pi} and @{e.r2.i}: most of it is the series\' own past', '$R^2$ de @{e.r2.pi} și @{e.r2.i}: cea mai mare parte provine din propriul trecut al seriilor')]),
    (T('In the ROBOR equation, past GDP growth ($t = @{e.i.L1g.t}$) and past inflation ($t = @{e.i.L1pi.t}$) matter: the interest rate follows the economy', 'În ecuația ROBOR contează creșterea PIB-ului din trecut ($t = @{e.i.L1g.t}$) și inflația din trecut ($t = @{e.i.L1pi.t}$): dobînda urmează economia'),
     [T('GDP growth is hard to predict: $R^2 = @{e.r2.g}$', 'Creșterea PIB-ului este greu de prognozat: $R^2 = @{e.r2.g}$')]),
    T('Single coefficients of correlated lags are unstable (e.g.\\ $i_{t-1}$ and $i_{t-2}$ with opposite signs in the $g$ equation): read the system through tests, responses and forecasts, not one coefficient at a time',
      'Coeficienții individuali ai unor laguri corelate sînt instabili (de exemplu $i_{t-1}$ și $i_{t-2}$ cu semne opuse în ecuația lui $g$): interpretăm sistemul prin teste, răspunsuri și prognoze, nu coeficient cu coeficient')])

D.recap(('Estimation and lag selection', 'estimare și alegerea numărului de laguri'), [
    T('OLS equation by equation is efficient: all equations have the same regressors', 'OLS ecuație cu ecuație este eficient: toate ecuațiile au aceiași regresori'),
    T('AIC, BIC, HQ: $\\ln\\det\\tilde\\bSigma(p)$ plus a penalty; compare on a common sample', 'AIC, BIC, HQ: $\\ln\\det\\tilde\\bSigma(p)$ plus o penalizare; comparăm pe un eșantion comun'),
    T('BIC $\\le$ HQ $\\le$ AIC in the chosen $p$; Romanian VAR: $p = 2$', 'BIC $\\le$ HQ $\\le$ AIC ca $p$ ales; VAR-ul românesc: $p = 2$'),
    T('Interpret the system, not single coefficients', 'Interpretăm sistemul, nu coeficienții individuali')])

# =============================================================================
# 5. DIAGNOSTICARE
# =============================================================================
D.section('Residual diagnostics', 'Diagnosticarea reziduurilor')

D.frame(T('Are the residuals white noise?', 'Verificarea reziduurilor: zgomot alb'), items(
    (T('\\textbf{Multivariate portmanteau test} \\refHosking', '\\textbf{Testul portmanteau multivariat} \\refHosking'),
     [T('$H_0$: no autocorrelation and no cross-correlation of the residuals up to lag $h$', '$H_0$: nicio autocorelație și nicio corelație încrucișată a reziduurilor pînă la lagul $h$'),
      T('$Q_h = T^2\\sum_{j=1}^{h}\\frac{1}{T - j}\\mathrm{tr}(\\hat{\\mathbf{C}}_j\'\\hat{\\mathbf{C}}_0^{-1}\\hat{\\mathbf{C}}_j\\hat{\\mathbf{C}}_0^{-1})$, $\\hat{\\mathbf{C}}_j = \\frac{1}{T}\\sum_t\\hat\\bepsilon_t\\hat\\bepsilon_{t-j}\'$: the residual autocovariance matrix at lag $j$; $\\mathrm{tr}$: the trace', '$Q_h = T^2\\sum_{j=1}^{h}\\frac{1}{T - j}\\mathrm{tr}(\\hat{\\mathbf{C}}_j\'\\hat{\\mathbf{C}}_0^{-1}\\hat{\\mathbf{C}}_j\\hat{\\mathbf{C}}_0^{-1})$, $\\hat{\\mathbf{C}}_j = \\frac{1}{T}\\sum_t\\hat\\bepsilon_t\\hat\\bepsilon_{t-j}\'$: matricea de autocovarianță a reziduurilor la lagul $j$; $\\mathrm{tr}$: urma'),
      T('under $H_0$: $\\chi^2$ with $K^2(h - p)$ degrees of freedom; the multivariate Ljung--Box test (Chapter 2); a small p-value: the residuals are still autocorrelated', 'în ipoteza $H_0$: $\\chi^2$ cu $K^2(h - p)$ grade de libertate; testul Ljung--Box multivariat (Capitolul 2); un p-value mic: reziduurile sînt încă autocorelate')]),
    (T('\\textbf{Normality}: a multivariate Jarque--Bera test \\refJB\\ on the standardised residuals (skewness and kurtosis)', '\\textbf{Normalitatea}: un test Jarque--Bera multivariat \\refJB\\ pe reziduurile standardizate (asimetrie și boltire)'),
     [T('Normality is not needed for OLS; it matters for exact intervals and for small samples', 'normalitatea nu este necesară pentru OLS; contează pentru intervale exacte și pentru eșantioane mici')]),
    (T('If the portmanteau test rejects: add lags, add a variable, or model breaks and outliers (dummy variables)', 'Dacă testul portmanteau respinge: adăugăm laguri, adăugăm o variabilă sau modelăm rupturile și valorile extreme (variabile dummy)'),
     [T('also check stability (eigenvalues) and, for long samples, parameter stability over time', 'verificăm și stabilitatea (valorile proprii) și, pentru eșantioane lungi, stabilitatea parametrilor în timp')])))

chart(T('Residuals of the Romanian VAR(2)', 'Reziduurile VAR(2) românesc'), 'tsa_ch6_resid', 'TSA_ch6_estimation_diagnostics', [
    T('Residuals of the three equations, @{e.q0}--@{ro.q1}, with $\\pm 2$ residual standard deviations', 'Reziduurile celor trei ecuații, @{e.q0}--@{ro.q1}, cu $\\pm 2$ abateri standard ale reziduurilor')],
    h='0.5\\textheight')

interp(('the residuals', 'reziduurilor'), [
    (T('Largest residuals: GDP @{rs.gv} in @{rs.gd} (the pandemic), ROBOR $+@{rs.iv}$ in @{rs.id} (the 2008 liquidity squeeze), inflation @{rs.piv} in @{rs.pid}', 'Cele mai mari reziduuri: PIB @{rs.gv} în @{rs.gd} (pandemia), ROBOR $+@{rs.iv}$ în @{rs.id} (criza de lichiditate din 2008), inflația @{rs.piv} în @{rs.pid}'),
     [T('kurtosis @{rs.kg} ($g$) and @{rs.ki} ($i$), against 3 under the Normal distribution: Jarque--Bera @{rs.jb} (p @{rs.jbp})', 'boltire @{rs.kg} ($g$) și @{rs.ki} ($i$), față de 3 pentru distribuția Normală: Jarque--Bera @{rs.jb} (p @{rs.jbp})')]),
    (T('Portmanteau: $Q_8 = @{rs.q8}$ ($@{rs.q8df}$ d.f., p @{rs.q8p}); $Q_{12} = @{rs.q12}$ ($@{rs.q12df}$ d.f., p @{rs.q12p})', 'Portmanteau: $Q_8 = @{rs.q8}$ ($@{rs.q8df}$ de grade de libertate, p @{rs.q8p}); $Q_{12} = @{rs.q12}$ ($@{rs.q12df}$ de grade de libertate, p @{rs.q12p})'),
     [T('borderline at 8 lags, fine at 12: some short-run correlation is left, partly from overlapping 12-month inflation rates', 'la limită pentru 8 laguri, acceptabil pentru 12: rămîne o corelație pe termen scurt, în parte din cauza ratelor anuale ale inflației care se suprapun')]),
    T('Consequence: point estimates are usable, but the intervals of the next sections rely on large-sample approximations; we use bootstrap bands for the responses',
      'Consecință: estimările punctuale pot fi folosite, dar intervalele din secțiunile următoare se bazează pe aproximări pentru eșantioane mari; pentru răspunsuri folosim benzi bootstrap')])

D.recap(('Residual diagnostics', 'diagnosticarea reziduurilor'), [
    T('Portmanteau test: $H_0$ white-noise residuals, $\\chi^2$ with $K^2(h - p)$ d.f.', 'Testul portmanteau: $H_0$ reziduuri zgomot alb, $\\chi^2$ cu $K^2(h - p)$ grade de libertate'),
    T('Normality fails in macro data: crises produce outliers', 'Normalitatea nu se verifică în datele macroeconomice: crizele produc valori extreme'),
    T('Remedies: more lags, more variables, dummies for crisis quarters', 'Remedii: mai multe laguri, mai multe variabile, dummy pentru trimestrele de criză')])

# =============================================================================
# 6. CAUZALITATE GRANGER
# =============================================================================
D.section('Granger causality', 'Cauzalitatea Granger')

D.frame(T('Granger causality: the definition', 'Cauzalitatea Granger: definiția'), '\\begin{columns}[T]\n\\begin{column}{0.3\\textwidth}\n'
        + ph('granger', T('Clive Granger (1934--2009), Nobel Prize in Economics 2003', 'Clive Granger (1934--2009), Premiul Nobel pentru economie 2003'), h='0.44\\textheight')
        + '\n\\end{column}\n\\begin{column}{0.68\\textwidth}\n' + items(
            (T('\\textbf{Definition} \\refGranger: $x$ \\textbf{Granger-causes} $y$', '\\textbf{Definiție} \\refGranger: $x$ \\textbf{cauzează în sens Granger} pe $y$'),
             [T('if the past of $x$ improves the forecast of $y$ made from the past of $y$ (and of the other variables)', 'dacă trecutul lui $x$ îmbunătățește prognoza lui $y$ făcută din trecutul lui $y$ (și al celorlalte variabile)'),
              T('formally: the MSE (mean squared error) of the optimal forecast of $y_{t+1}$ is smaller when $x_t, x_{t-1}, \\dots$ are used', 'formal: MSE (mean squared error, eroarea pătratică medie) a prognozei optime pentru $y_{t+1}$ este mai mică atunci cînd folosim $x_t, x_{t-1}, \\dots$')]),
            (T('A statement about \\textbf{predictability}, not about cause and effect', 'O afirmație despre \\textbf{predictibilitate}, nu despre cauză și efect'),
             [T('the cause comes before the effect, but ``before\'\' does not imply ``because\'\'', 'cauza vine înaintea efectului, dar „înainte” nu înseamnă „din cauza”')]),
            T('Four cases: no causality, $x \\to y$, $y \\to x$, feedback $x \\leftrightarrow y$', 'Patru cazuri: nicio cauzalitate, $x \\to y$, $y \\to x$, feedback $x \\leftrightarrow y$'))
        + '\n\\end{column}\n\\end{columns}')

D.frame(T('Testing Granger causality in a VAR', 'Testarea cauzalității Granger într-un VAR'), items(
    (T('In a VAR$(p)$, $x$ does not Granger-cause $y$ if all coefficients of $x_{t-1}, \\dots, x_{t-p}$ in the equation of $y$ are zero', 'Într-un VAR$(p)$, $x$ nu cauzează în sens Granger pe $y$ dacă toți coeficienții lui $x_{t-1}, \\dots, x_{t-p}$ din ecuația lui $y$ sînt zero'),
     [T('bivariate VAR(2): $H_0$: $a_{12}^{(1)} = a_{12}^{(2)} = 0$ in $y_t = c + a_{11}^{(1)}y_{t-1} + a_{12}^{(1)}x_{t-1} + a_{11}^{(2)}y_{t-2} + a_{12}^{(2)}x_{t-2} + \\varepsilon_t$',
        'VAR(2) cu două variabile: $H_0$: $a_{12}^{(1)} = a_{12}^{(2)} = 0$ în $y_t = c + a_{11}^{(1)}y_{t-1} + a_{12}^{(1)}x_{t-1} + a_{11}^{(2)}y_{t-2} + a_{12}^{(2)}x_{t-2} + \\varepsilon_t$'),
      T('$a_{12}^{(j)}$: the coefficient of $x_{t-j}$ in the equation of $y$ (the superscript is the lag)', '$a_{12}^{(j)}$: coeficientul lui $x_{t-j}$ în ecuația lui $y$ (indicele de sus este lagul)')]),
    (T('\\textbf{$F$ test}: $F = \\dfrac{(RSS_R - RSS_U)/p}{RSS_U/(T - Kp - 1)} \\sim F(p, T - Kp - 1)$ under $H_0$', '\\textbf{Testul $F$}: $F = \\dfrac{(RSS_R - RSS_U)/p}{RSS_U/(T - Kp - 1)} \\sim F(p, T - Kp - 1)$ în ipoteza $H_0$'),
     [T('$RSS_U$: residual sum of squares of the equation of $y$; $RSS_R$: the same without the lags of $x$', '$RSS_U$: suma pătratelor reziduurilor din ecuația lui $y$; $RSS_R$: aceeași sumă fără lagurile lui $x$'),
      T('a large $F$ (a small p-value): dropping the lags of $x$ raises the RSS too much, so $H_0$ is rejected', 'un $F$ mare (un p-value mic): eliminarea lagurilor lui $x$ crește prea mult RSS, deci respingem $H_0$')]),
    (T('\\textbf{Wald test}: $W = (\\mathbf{R}\\hat{\\boldsymbol{\\beta}})\'[\\mathbf{R}\\hat{\\mathbf{V}}\\mathbf{R}\']^{-1}(\\mathbf{R}\\hat{\\boldsymbol{\\beta}}) \\sim \\chi^2(p)$; here $W = pF$', '\\textbf{Testul Wald}: $W = (\\mathbf{R}\\hat{\\boldsymbol{\\beta}})\'[\\mathbf{R}\\hat{\\mathbf{V}}\\mathbf{R}\']^{-1}(\\mathbf{R}\\hat{\\boldsymbol{\\beta}}) \\sim \\chi^2(p)$; aici $W = pF$'),
     [T('$\\hat{\\boldsymbol{\\beta}}$: the estimated coefficients of the equation of $y$; $\\mathbf{R}$ selects the tested ones; $\\hat{\\mathbf{V}}$: the estimated covariance of $\\hat{\\boldsymbol{\\beta}}$; both versions need stationary variables', '$\\hat{\\boldsymbol{\\beta}}$: coeficienții estimați ai ecuației lui $y$; $\\mathbf{R}$ îi selectează pe cei testați; $\\hat{\\mathbf{V}}$: covarianța estimată a lui $\\hat{\\boldsymbol{\\beta}}$; ambele versiuni cer variabile staționare')])))

D.frame(T('Worked example: does GDP growth Granger-cause ROBOR?', 'Exemplu rezolvat: este creșterea PIB-ului o cauză Granger pentru ROBOR?'), items(
    (T('Equation of $i_t$ in the Romanian VAR(2), $T = @{gh.T}$, $K = 3$, $p = 2$', 'Ecuația lui $i_t$ din VAR(2) românesc, $T = @{gh.T}$, $K = 3$, $p = 2$'),
     [T('unrestricted (with $g_{t-1}$, $g_{t-2}$): $RSS_U = @{gh.ru}$; restricted (without them): $RSS_R = @{gh.rr}$', 'nerestricționat (cu $g_{t-1}$, $g_{t-2}$): $RSS_U = @{gh.ru}$; restricționat (fără ele): $RSS_R = @{gh.rr}$')]),
    (T('$F = \\dfrac{(@{gh.rr} - @{gh.ru})/2}{@{gh.ru}/@{gh.df2}} = \\dfrac{@{gh.num}}{@{gh.den}} = @{gh.F}$', '$F = \\dfrac{(@{gh.rr} - @{gh.ru})/2}{@{gh.ru}/@{gh.df2}} = \\dfrac{@{gh.num}}{@{gh.den}} = @{gh.F}$'),
     [T('degrees of freedom $(2, @{gh.T} - 6 - 1) = (2, @{gh.df2})$; 5\\% critical value @{gh.cv}; p-value @{gh.p}', 'grade de libertate $(2, @{gh.T} - 6 - 1) = (2, @{gh.df2})$; valoarea critică de 5\\%: @{gh.cv}; p-value-ul: @{gh.p}')]),
    (T('\\textbf{Decision}: reject $H_0$: past GDP growth helps to forecast ROBOR 3M, given past inflation and past ROBOR', '\\textbf{Decizia}: respingem $H_0$: creșterea PIB-ului din trecut ajută la prognoza ROBOR 3M, dat fiind trecutul inflației și al ROBOR'),
     [T('consistent with a central bank that tightens when the economy grows fast', 'compatibil cu o bancă centrală care înăsprește politica atunci cînd economia crește repede')])))

D.frame(T('Granger causality in the Romanian VAR(2)', 'Cauzalitatea Granger în VAR(2) românesc'), table(
    'llrr', T('\\textbf{Cause} & \\textbf{Effect} & $F$ & \\textbf{p}', '\\textbf{Cauza} & \\textbf{Efectul} & $F$ & \\textbf{p}'),
    [T('GDP growth & inflation', 'creșterea PIB & inflația') + ' & @{gr.gpi.F} & @{gr.gpi.p}',
     T('GDP growth & ROBOR 3M', 'creșterea PIB & ROBOR 3M') + ' & @{gr.gi.F} & @{gr.gi.p}',
     T('inflation & GDP growth', 'inflația & creșterea PIB') + ' & @{gr.pig.F} & @{gr.pig.p}',
     T('inflation & ROBOR 3M', 'inflația & ROBOR 3M') + ' & @{gr.pii.F} & @{gr.pii.p}',
     T('ROBOR 3M & GDP growth', 'ROBOR 3M & creșterea PIB') + ' & @{gr.ig.F} & @{gr.ig.p}',
     T('ROBOR 3M & inflation', 'ROBOR 3M & inflația') + ' & @{gr.ipi.F} & @{gr.ipi.p}'],
    size='footnotesize') + items(
    T('$F(2, @{gr.df2})$ tests in the VAR(2), @{e.q0}--@{ro.q1}; 5\\% critical value @{gr.cv}', 'Teste $F(2, @{gr.df2})$ în VAR(2), @{e.q0}--@{ro.q1}; valoarea critică de 5\\%: @{gr.cv}'),
    T('Significant at 5\\%: GDP growth $\\to$ ROBOR, inflation $\\to$ ROBOR, ROBOR $\\to$ GDP growth', 'Semnificative la 5\\%: creșterea PIB $\\to$ ROBOR, inflația $\\to$ ROBOR, ROBOR $\\to$ creșterea PIB')) + ql('TSA_ch6_granger'),
    'small')

interp(('the Granger tests', 'testelor Granger'), [
    (T('The interest rate follows the economy: past growth and past inflation help to forecast ROBOR', 'Dobînda urmează economia: creșterea și inflația din trecut ajută la prognoza ROBOR'),
     [T('the pattern of a reaction function (the central bank responds to inflation and activity); the test alone cannot prove it is the BNR\'s rule', 'tiparul unei funcții de reacție (banca centrală răspunde la inflație și la activitate); testul singur nu poate demonstra că aceasta este regula BNR')]),
    (T('ROBOR does not Granger-cause inflation (p @{gr.ipi.p}): no evidence that past rates improve inflation forecasts', 'ROBOR nu cauzează inflația în sens Granger (p @{gr.ipi.p}): nicio dovadă că dobînzile din trecut îmbunătățesc prognoza inflației'),
     [T('this does not say that monetary policy is ineffective: the effect may be slow, small relative to the noise, or already anticipated', 'aceasta nu înseamnă că politica monetară este ineficientă: efectul poate fi lent, mic față de zgomot sau deja anticipat')]),
    T('ROBOR $\\to$ GDP growth (p @{gr.ig.p}): higher rates come before weaker growth; in the bivariate VAR (GDP, ROBOR) alone the p-value is @{gr.bi.ig}',
      'ROBOR $\\to$ creșterea PIB (p @{gr.ig.p}): dobînzile mai mari preced o creștere mai slabă; în VAR-ul cu două variabile (PIB, ROBOR) p-value-ul este @{gr.bi.ig}')])

D.frame(T('Markets: does New York lead Frankfurt and Bucharest?', 'Piețe: New York-ul precedă Frankfurtul și Bucureștiul'), '\\begin{columns}[T]\n\\begin{column}{0.36\\textwidth}\n'
        + ph('fse', T('Trading floor of the Frankfurt Stock Exchange (DAX)', 'Sala de tranzacționare a Bursei din Frankfurt (DAX)'), h='0.34\\textheight')
        + '\n\\end{column}\n\\begin{column}{0.62\\textwidth}\n' + table(
            'llrr', T('\\textbf{Cause} & \\textbf{Effect} & $F$ & \\textbf{p}', '\\textbf{Cauza} & \\textbf{Efectul} & $F$ & \\textbf{p}'),
            ['S\\&P 500 & DAX & @{gm.sp500_dax.F} & @{gm.sp500_dax.p}', 'S\\&P 500 & BET & @{gm.sp500_bet.F} & @{gm.sp500_bet.p}',
             'DAX & S\\&P 500 & @{gm.dax_sp500.F} & @{gm.dax_sp500.p}', 'DAX & BET & @{gm.dax_bet.F} & @{gm.dax_bet.p}',
             'BET & S\\&P 500 & @{gm.bet_sp500.F} & @{gm.bet_sp500.p}', 'BET & DAX & @{gm.bet_dax.F} & @{gm.bet_dax.p}'],
            size='scriptsize') + items(
            T('Daily log returns, VAR(4) (the order of \\refDYb), @{gm.T} days, 2000--2026; $F(4, \\cdot)$ tests; HQ would choose $p = @{gm.o.hqic}$, AIC $p = @{gm.o.aic}$',
              'Randamente logaritmice zilnice, VAR(4) (ordinul din \\refDYb), @{gm.T} de zile, 2000--2026; teste $F(4, \\cdot)$; HQ ar alege $p = @{gm.o.hqic}$, AIC $p = @{gm.o.aic}$'),
            T('Coefficient of the S\\&P 500 of yesterday in the BET equation: @{gm.b1} (SE @{gm.b1se}); in the DAX equation: @{gm.d1}', 'Coeficientul S\\&P 500 de ieri în ecuația BET: @{gm.b1} (SE @{gm.b1se}); în ecuația DAX: @{gm.d1}')) + '\n\\end{column}\n\\end{columns}\n' + ql('TSA_ch6_granger'),
        'footnotesize')

interp(('the market tests', 'testelor pentru piețe'), [
    (T('Strong one-way causality from the S\\&P 500 to the DAX and the BET: a 1\\% S\\&P return yesterday adds about @{gm.b1}\\% to the BET today', 'Cauzalitate puternică, într-un singur sens, de la S\\&P 500 spre DAX și BET: un randament de 1\\% al S\\&P 500 ieri adaugă circa @{gm.b1}\\% la BET azi'),
     [T('the BET does not Granger-cause the S\\&P 500 (p @{gm.bet_sp500.p})', 'BET nu cauzează S\\&P 500 în sens Granger (p @{gm.bet_sp500.p})')]),
    (T('Cause: the closing times (non-synchronous trading), not a slow market', 'Cauza: orele de închidere (tranzacționare nesincronă), nu o reacție lentă a pieței'),
     [T('New York keeps trading after Europe closes; ``yesterday\'s\'\' S\\&P return partly happened while Europe was closed', 'New York tranzacționează și după închiderea Europei; randamentul S\\&P „de ieri” s-a produs parțial cînd Europa era închisă'),
      T('on weekly data the effect is much weaker (Seminar 6, B2)', 'pe date săptămînale efectul este mult mai slab (Seminarul 6, B2)')]),
    T('Can you trade on it? You would buy at the Bucharest open, after prices have already adjusted: Granger causality in closing prices is not a free profit',
      'Regularitatea nu aduce un profit sigur: ați cumpăra la deschiderea Bursei de la București, după ce prețurile s-au ajustat deja; cauzalitatea Granger în prețurile de închidere nu este o sursă de profit fără risc')])

D.frame(T('Pitfall 1: omitted variables', 'Capcana 1: variabilele omise'), items(
    (T('Granger causality is relative to the information set: adding or removing a variable can create or destroy it \\refLutOm', 'Cauzalitatea Granger depinde de setul de informații: adăugarea sau eliminarea unei variabile o poate crea sau distruge \\refLutOm'),
     [T('a common driver $z$ that affects $x$ earlier than $y$ makes $x$ look like a cause of $y$', 'un factor comun $z$ care afectează pe $x$ mai devreme decît pe $y$ îl face pe $x$ să pară o cauză a lui $y$')]),
    (T('Simulation: $z_t = 0.5z_{t-1} + e_t$, $x_t = 0.8z_{t-1} + u_t$, $y_t = 0.8z_{t-2} + v_t$', 'Simulare: $z_t = 0{,}5z_{t-1} + e_t$, $x_t = 0{,}8z_{t-1} + u_t$, $y_t = 0{,}8z_{t-2} + v_t$'),
     [T('$e_t$, $u_t$, $v_t$: independent standard Normal white noises', '$e_t$, $u_t$, $v_t$: zgomote albe independente, cu distribuția Normală standard'),
      T('$x$ never enters the equation of $y$; but $x_{t-1}$ is a noisy measure of $z_{t-2}$, which drives $y_t$', '$x$ nu apare niciodată în ecuația lui $y$; dar $x_{t-1}$ este o măsură zgomotoasă a lui $z_{t-2}$, care determină pe $y_t$')]),
    T('Example: ice-cream sales Granger-cause drownings in a bivariate model; temperature, the omitted driver, explains both', 'Exemplu: vînzările de înghețată cauzează înecurile în sens Granger într-un model cu două variabile; temperatura, factorul omis, le explică pe amîndouă')))

chart(T('Spurious Granger causality from an omitted variable', 'Cauzalitate Granger falsă din cauza unei variabile omise'), 'tsa_ch6_granger_sim', 'TSA_ch6_granger', [
    T('1\\,000 simulations per sample size of the system of the previous slide; $F$ test of ``$x$ does not Granger-cause $y$\'\' at 5\\% in a VAR(2), with and without $z$',
      '1\\,000 de simulări pentru fiecare volum al eșantionului, cu sistemul de pe slide-ul anterior; testul $F$ pentru „$x$ nu cauzează $y$ în sens Granger” la 5\\% într-un VAR(2), cu și fără $z$')],
    h='0.52\\textheight')

interp(('the simulation', 'simulării'), [
    (T('Without $z$, the test rejects in @{gs.50.b}\\% of samples with $T = 50$ and @{gs.200.b}\\% with $T = 200$: almost always a ``cause\'\'', 'Fără $z$, testul respinge în @{gs.50.b}\\% din eșantioane pentru $T = 50$ și în @{gs.200.b}\\% pentru $T = 200$: aproape mereu o „cauză”'),
     [T('more data makes it worse: the test becomes more sure of a wrong conclusion', 'mai multe date înrăutățesc situația: testul devine tot mai sigur de o concluzie greșită')]),
    (T('With $z$ in the VAR, the rejection rate is @{gs.50.t}\\%--@{gs.500.t}\\%, close to the nominal 5\\%', 'Cu $z$ în VAR, rata de respingere este @{gs.50.t}\\%--@{gs.500.t}\\%, apropiată de nivelul nominal de 5\\%'),
     [T('Granger causality is a property of the chosen system, not of the world', 'cauzalitatea Granger este o proprietate a sistemului ales, nu a lumii')]),
    T('Romanian example: inflation $\\to$ GDP growth has p @{gr.pig.p} in the three-variable VAR and @{gr.bi.pig} in the bivariate one', 'Exemplu românesc: inflația $\\to$ creșterea PIB are p @{gr.pig.p} în VAR-ul cu trei variabile și @{gr.bi.pig} în cel cu două variabile')])

D.frame(T('Pitfall 2: instantaneous causality', 'Capcana 2: cauzalitatea instantanee'), items(
    (T('Granger tests only use lags; effects within the same period are hidden in the correlation of the shocks $\\bSigma$', 'Testele Granger folosesc doar laguri; efectele din aceeași perioadă sînt ascunse în corelația șocurilor $\\bSigma$'),
     [T('\\textbf{instantaneous causality} between $x$ and $y$: $\\Cov(\\varepsilon_{xt}, \\varepsilon_{yt}) \\ne 0$; tested by a Wald test on the off-diagonal elements of $\\bSigma$', '\\textbf{cauzalitate instantanee} între $x$ și $y$: $\\Cov(\\varepsilon_{xt}, \\varepsilon_{yt}) \\ne 0$; testată printr-un test Wald pe elementele din afara diagonalei lui $\\bSigma$')]),
    (T('It has \\textbf{no direction}: the data cannot tell whether $x$ moved $y$ or $y$ moved $x$ within the quarter', 'Nu are \\textbf{direcție}: datele nu pot spune dacă $x$ l-a mișcat pe $y$ sau $y$ pe $x$ în cursul trimestrului'),
     [T('Romania: ROBOR shocks are instantaneously correlated with the others (p @{gr.inst.i}); the residual correlation of inflation and ROBOR is @{e.cpii}', 'România: șocurile ROBOR sînt corelate instantaneu cu celelalte (p @{gr.inst.i}); corelația reziduurilor inflației și ROBOR este @{e.cpii}')]),
    T('With quarterly data, many true effects happen within the quarter: a VAR in which nothing is Granger-significant can still have strong links (Section 7)',
      'În datele trimestriale, multe efecte reale au loc în cursul trimestrului: un VAR în care nimic nu este semnificativ în sens Granger poate avea totuși legături puternice (secțiunea 7)')))

D.frame(T('Pitfall 3: trends, expectations and the meaning of "cause"', 'Capcana 3: trenduri, așteptări și sensul cuvîntului „cauză”'), items(
    (T('\\textbf{Non-stationary variables}: with $I(1)$ series the $F$ and Wald tests do not have their usual distributions \\refSSW', '\\textbf{Variabile nestaționare}: pentru serii $I(1)$, testele $F$ și Wald nu au distribuțiile obișnuite \\refSSW'),
     [T('remedies: test on differences (if there is no cointegration), or the Toda--Yamamoto approach \\refTY: estimate a VAR$(p + d_{\\max})$ in levels and test only the first $p$ lags', 'remedii: testăm pe diferențe (dacă nu există cointegrare) sau abordarea Toda--Yamamoto \\refTY: estimăm un VAR$(p + d_{\\max})$ în niveluri și testăm doar primele $p$ laguri'),
      T('$d_{\\max}$: the highest order of integration of the series (usually 1); the extra lags are estimated but not tested', '$d_{\\max}$: ordinul maxim de integrare al seriilor (de obicei 1); lagurile suplimentare se estimează, dar nu se testează'),
      T('cointegrated series need a VECM: Chapter 7', 'seriile cointegrate cer un VECM: Capitolul 7')]),
    (T('\\textbf{Expectations}: forward-looking prices move before the events they anticipate', '\\textbf{Așteptările}: prețurile orientate spre viitor se mișcă înaintea evenimentelor pe care le anticipează'),
     [T('stock prices Granger-cause GDP, and weather forecasts ``cause\'\' rain; neither is a cause in the everyday sense', 'prețurile acțiunilor cauzează PIB-ul în sens Granger, iar prognozele meteo „cauzează” ploaia; niciuna nu este o cauză în sensul obișnuit')]),
    T('Report: ``$x$ helps to predict $y$, given these variables and this sample\'\'; never ``$x$ causes $y$\'\'', 'Raportăm: „$x$ ajută la prognoza lui $y$, date fiind aceste variabile și acest eșantion”; niciodată „$x$ cauzează pe $y$”')))

D.recap(('Granger causality', 'cauzalitatea Granger'), [
    T('$x \\to y$ (Granger): the lags of $x$ improve the forecast of $y$; $F$ or Wald test of zero restrictions', '$x \\to y$ (Granger): lagurile lui $x$ îmbunătățesc prognoza lui $y$; test $F$ sau Wald al restricțiilor nule'),
    T('Romania: GDP growth and inflation $\\to$ ROBOR; markets: S\\&P 500 $\\to$ DAX, BET (time zones)', 'România: creșterea PIB și inflația $\\to$ ROBOR; piețe: S\\&P 500 $\\to$ DAX, BET (fusurile orare)'),
    T('Pitfalls: omitted variables, instantaneous effects, non-stationarity, expectations', 'Capcane: variabile omise, efecte instantanee, nestaționaritate, așteptări'),
    T('Granger causality is predictability, not causation', 'Cauzalitatea Granger înseamnă predictibilitate, nu cauzalitate')])

# =============================================================================
# 7. RĂSPUNS LA IMPULS ȘI FEVD
# =============================================================================
D.section('Impulse responses', 'Funcții de răspuns la impuls')

D.frame(T('Impulse responses and why we orthogonalise', 'Răspunsurile la impuls și necesitatea ortogonalizării'), items(
    (T('\\textbf{Impulse response function} (IRF): the path of $Y_{j,t+h}$, $h = 0, 1, 2, \\dots$, after a one-time shock to variable $k$ at time $t$', '\\textbf{Funcția de răspuns la impuls} (IRF): traiectoria lui $Y_{j,t+h}$, $h = 0, 1, 2, \\dots$, după un șoc unic în variabila $k$ la momentul $t$'),
     [T('from the moving-average form: the response to a unit change in $\\varepsilon_{kt}$ is the column $k$ of $\\bPhi_h$', 'din forma de medie mobilă: răspunsul la o modificare unitară a lui $\\varepsilon_{kt}$ este coloana $k$ a lui $\\bPhi_h$')]),
    (T('\\textbf{Problem}: the shocks are correlated ($\\bSigma$ not diagonal)', '\\textbf{Problema}: șocurile sînt corelate ($\\bSigma$ nu este diagonală)'),
     [T('``a shock to inflation with no shock to ROBOR\'\' is an experiment that the data rarely show: the two shocks usually come together (correlation @{e.cpii})', '„un șoc al inflației fără niciun șoc al ROBOR” este un experiment pe care datele îl arată rar: cele două șocuri vin de obicei împreună (corelația @{e.cpii})')]),
    (T('\\textbf{Solution}: rewrite the shocks as $\\bepsilon_t = \\mathbf{P}\\mathbf{u}_t$, with $\\mathbf{u}_t$ uncorrelated, unit variance: $\\bSigma = \\mathbf{P}\\mathbf{P}\'$', '\\textbf{Soluția}: rescriem șocurile ca $\\bepsilon_t = \\mathbf{P}\\mathbf{u}_t$, cu $\\mathbf{u}_t$ necorelate, de varianță unitară: $\\bSigma = \\mathbf{P}\\mathbf{P}\'$'),
     [T('the \\textbf{orthogonalised} IRF: $\\boldsymbol{\\Theta}_h = \\bPhi_h\\mathbf{P}$; element $(j, k)$: the response of $Y_j$ to a one-standard-deviation shock $u_k$', '\\textbf{IRF ortogonalizată}: $\\boldsymbol{\\Theta}_h = \\bPhi_h\\mathbf{P}$; elementul $(j, k)$: răspunsul lui $Y_j$ la un șoc $u_k$ de o abatere standard'),
      T('$\\mathbf{P}$: a $K \\times K$ matrix that turns the uncorrelated shocks $\\mathbf{u}_t$ into the observed shocks $\\bepsilon_t$', '$\\mathbf{P}$: o matrice $K \\times K$ care transformă șocurile necorelate $\\mathbf{u}_t$ în șocurile observate $\\bepsilon_t$')])))

D.frame(T('The Cholesky decomposition and the ordering', 'Descompunerea Cholesky și ordinea variabilelor'), items(
    (T('\\textbf{Cholesky}: the unique lower triangular $\\mathbf{P}$ with positive diagonal such that $\\bSigma = \\mathbf{P}\\mathbf{P}\'$', '\\textbf{Cholesky}: matricea unică $\\mathbf{P}$, inferior triunghiulară, cu diagonala pozitivă, pentru care $\\bSigma = \\mathbf{P}\\mathbf{P}\'$'),
     [T('example: $\\bSigma = \\begin{pmatrix} 1 & 0.5 \\\\ 0.5 & 1 \\end{pmatrix}$: $\\mathbf{P} = @{wx.P}$, since $p_{11} = 1$, $p_{21} = 0.5/1$, $p_{22} = \\sqrt{1 - 0.25}$', 'exemplu: $\\bSigma = \\begin{pmatrix} 1 & 0{,}5 \\\\ 0{,}5 & 1 \\end{pmatrix}$: $\\mathbf{P} = @{wx.P}$, deoarece $p_{11} = 1$, $p_{21} = 0{,}5/1$, $p_{22} = \\sqrt{1 - 0{,}25}$')]),
    (T('\\textbf{Recursive structure}: $\\varepsilon_{1t} = p_{11}u_{1t}$, $\\varepsilon_{2t} = p_{21}u_{1t} + p_{22}u_{2t}$', '\\textbf{Structura recursivă}: $\\varepsilon_{1t} = p_{11}u_{1t}$, $\\varepsilon_{2t} = p_{21}u_{1t} + p_{22}u_{2t}$'),
     [T('the first variable is not moved by $u_2$ in the same period; the second reacts to both at once', 'prima variabilă nu este mișcată de $u_2$ în aceeași perioadă; a doua reacționează imediat la ambele'),
      T('the whole correlation is attributed to the variable ordered first', 'întreaga corelație este atribuită variabilei așezate prima')]),
    (T('\\textbf{The ordering matters}: with $y_2$ first, $\\mathbf{P} = @{wx.Prev}$ (upper triangular in the original order)', '\\textbf{Ordinea contează}: cu $y_2$ prima, $\\mathbf{P} = @{wx.Prev}$ (superior triunghiulară în ordinea inițială)'),
     [T('rule: order from the slowest variable (does not react within the period) to the fastest; Romania: GDP growth, inflation, ROBOR', 'regula: ordonăm de la variabila cea mai lentă (nu reacționează în aceeași perioadă) la cea mai rapidă; România: creșterea PIB, inflația, ROBOR')])))

D.frame(T('Worked example: orthogonalised responses', 'Exemplu rezolvat: răspunsuri ortogonalizate'), items(
    (T('VAR(1) of Section 2, order $(y_1, y_2)$: $\\boldsymbol{\\Theta}_0 = \\mathbf{P} = @{wx.P}$', 'VAR(1) din secțiunea 2, ordinea $(y_1, y_2)$: $\\boldsymbol{\\Theta}_0 = \\mathbf{P} = @{wx.P}$'),
     [T('a shock $u_1 = 1$ moves $y_1$ by 1 and $y_2$ by @{wx.p21} at once; $u_2 = 1$ moves only $y_2$, by @{wx.p22}', 'un șoc $u_1 = 1$ mișcă imediat $y_1$ cu 1 și $y_2$ cu @{wx.p21}; $u_2 = 1$ mișcă doar $y_2$, cu @{wx.p22}')]),
    (T('One period later: $\\boldsymbol{\\Theta}_1 = \\bA\\mathbf{P} = @{wx.Th1}$', 'O perioadă mai tîrziu: $\\boldsymbol{\\Theta}_1 = \\bA\\mathbf{P} = @{wx.Th1}$'),
     [T('$y_2$ responds to $u_1$ by @{wx.th21}: $0.3 \\cdot 1 + 0.4 \\cdot 0.5$, the direct effect $a_{21}$ plus its own persistence', '$y_2$ răspunde la $u_1$ cu @{wx.th21}: $0{,}3 \\cdot 1 + 0{,}4 \\cdot 0{,}5$, efectul direct $a_{21}$ plus propria persistență')]),
    T('In general $\\boldsymbol{\\Theta}_h = \\bPhi_h\\mathbf{P}$; the \\textbf{cumulative} response $\\sum_{s \\le h}\\boldsymbol{\\Theta}_s$ gives the effect on the level when the VAR is in growth rates',
      'În general $\\boldsymbol{\\Theta}_h = \\bPhi_h\\mathbf{P}$; răspunsul \\textbf{cumulat} $\\sum_{s \\le h}\\boldsymbol{\\Theta}_s$ dă efectul asupra nivelului cînd VAR-ul este în rate de creștere')))

chart(T('Impulse responses of the Romanian VAR(2)', 'Răspunsurile la impuls ale VAR(2) românesc'), 'tsa_ch6_irf_ro', 'TSA_ch6_irf_fevd', [
    T('Row: responding variable; column: shock of one standard deviation; Cholesky order $g, \\pi, i$; 90\\% bands from a residual bootstrap (500 replications) \\refKilian',
      'Rîndul: variabila care răspunde; coloana: șocul de o abatere standard; ordinea Cholesky $g, \\pi, i$; benzi de 90\\% dintr-un bootstrap pe reziduuri (500 de replicări) \\refKilian')],
    h='0.7\\textheight')

interp(('the Romanian responses', 'răspunsurilor românești'), [
    (T('An inflation shock (@{ir.pi0} pp on impact) raises ROBOR by @{ir.ipi0} pp at once and by up to @{ir.ipimax} pp after @{ir.ipih} quarters; still @{ir.ipi12} pp after 3 years', 'Un șoc al inflației (@{ir.pi0} puncte procentuale la impact) crește ROBOR imediat cu @{ir.ipi0} pp și pînă la @{ir.ipimax} pp după @{ir.ipih} trimestre; încă @{ir.ipi12} pp după 3 ani'),
     [T('the band excludes zero for @{ir.sig} of the 13 horizons: the clearest result of the VAR', 'banda exclude zero la @{ir.sig} din cele 13 orizonturi: cel mai clar rezultat al VAR-ului')]),
    (T('A ROBOR shock lowers GDP growth by @{ir.gi1} pp the next quarter, then the effect vanishes', 'Un șoc ROBOR reduce creșterea PIB-ului cu @{ir.gi1} pp în trimestrul următor, apoi efectul dispare'),
     [T('but inflation \\textbf{rises} after a ROBOR shock (up to @{ir.piimax} pp, not significant): the ``price puzzle\'\' \\refSimsPP', 'dar inflația \\textbf{crește} după un șoc ROBOR (pînă la @{ir.piimax} pp, nesemnificativ): „enigma prețurilor” \\refSimsPP'),
      T('a likely reason: the rate rises when the central bank expects inflation; information omitted from the VAR (exchange rate, energy prices) shows up as a ``shock\'\'', 'un motiv probabil: dobînda crește atunci cînd banca centrală anticipează inflația; informația omisă din VAR (cursul de schimb, prețurile energiei) apare ca „șoc”')]),
    T('Inflation shocks themselves are persistent: @{ir.pi4} pp after a year, @{ir.pi12} pp after three', 'Șocurile inflației sînt persistente: @{ir.pi4} pp după un an, @{ir.pi12} pp după trei')])

chart(T('Does the ordering change the answer?', 'Efectul ordinii asupra răspunsurilor'), 'tsa_ch6_irf_order', 'TSA_ch6_irf_fevd', [
    T('Same VAR(2), two Cholesky orderings (inflation before ROBOR, ROBOR before inflation) and the generalised responses of \\refPS, which do not depend on the order',
      'Același VAR(2), două ordini Cholesky (inflația înaintea ROBOR, ROBOR înaintea inflației) și răspunsurile generalizate din \\refPS, care nu depind de ordine')],
    h='0.52\\textheight')

interp(('the two orderings', 'celor două ordini'), [
    (T('ROBOR after an inflation shock: peak @{or.Amax} pp (inflation first) against @{or.Bmax} pp (ROBOR first)', 'ROBOR după un șoc al inflației: maximul @{or.Amax} pp (inflația prima), față de @{or.Bmax} pp (ROBOR primul)'),
     [T('with ROBOR first, the impact response is zero by construction', 'cu ROBOR primul, răspunsul la impact este zero prin construcție')]),
    (T('Inflation after a ROBOR shock: @{or.pA} pp against @{or.pB} pp, with @{or.pB0} pp already on impact', 'Inflația după un șoc ROBOR: @{or.pA} pp, față de @{or.pB} pp, cu @{or.pB0} pp deja la impact'),
     [T('the second ordering assigns the common part of the two shocks (correlation @{or.r}) to ROBOR, so the ``price puzzle\'\' gets larger', 'a doua ordine atribuie partea comună a celor două șocuri (corelația @{or.r}) lui ROBOR, deci „enigma prețurilor” crește')]),
    T('The ordering is an assumption about which variable reacts within the quarter; it must be stated and defended, and results that change with it are fragile',
      'Ordinea este o ipoteză despre ce variabilă reacționează în cursul trimestrului; trebuie precizată și justificată, iar rezultatele care se schimbă odată cu ea sînt fragile')])

D.frame(T('Generalised impulse responses', 'Răspunsuri la impuls generalizate'), items(
    (T('\\textbf{Generalised IRF} \\refPS\\ (after \\refKPP): shock variable $k$ by one standard deviation and let the other shocks move as they usually do with it', '\\textbf{IRF generalizată} \\refPS\\ (după \\refKPP): aplicăm variabilei $k$ un șoc de o abatere standard și lăsăm celelalte șocuri să se miște așa cum se mișcă de obicei odată cu ea'),
     [T('$\\boldsymbol{\\psi}_k(h) = \\bPhi_h\\bSigma\\mathbf{e}_k/\\sqrt{\\sigma_{kk}}$; $\\mathbf{e}_k$: column $k$ of the identity matrix; $\\sigma_{kk}$: the variance of shock $k$', '$\\boldsymbol{\\psi}_k(h) = \\bPhi_h\\bSigma\\mathbf{e}_k/\\sqrt{\\sigma_{kk}}$; $\\mathbf{e}_k$: coloana $k$ a matricei identitate; $\\sigma_{kk}$: varianța șocului $k$')]),
    (T('Properties', 'Proprietăți'),
     [T('no ordering is needed; for the variable ordered first, it equals the Cholesky response', 'nu este nevoie de nicio ordine; pentru variabila așezată prima coincide cu răspunsul Cholesky'),
      T('the shocks are not orthogonal: the responses to different shocks overlap and cannot be added up', 'șocurile nu sînt ortogonale: răspunsurile la șocuri diferite se suprapun și nu se pot aduna')]),
    T('Useful for markets, where no ordering is natural; it describes typical co-movements, not structural shocks', 'Util pentru piețe, unde nicio ordine nu este naturală; descrie evoluții comune tipice, nu șocuri structurale')))

chart(T('Markets: responses to an S\\&P 500 shock', 'Piețe: răspunsuri la un șoc S\\&P 500'), 'tsa_ch6_market_irf', 'TSA_ch6_market_spillovers', [
    T('VAR(4) of daily S\\&P 500, DAX and BET returns; S\\&P shock of one standard deviation (@{mi.sd}\\%); generalised responses, Cholesky with the S\\&P 500 first, and Cholesky in the order of the closing times (BET, DAX, S\\&P 500)',
      'VAR(4) al randamentelor zilnice S\\&P 500, DAX și BET; șoc S\\&P de o abatere standard (@{mi.sd}\\%); răspunsuri generalizate, Cholesky cu S\\&P 500 primul și Cholesky în ordinea orelor de închidere (BET, DAX, S\\&P 500)')],
    h='0.52\\textheight')

interp(('the market responses', 'răspunsurilor piețelor'), [
    (T('Generalised: DAX @{mi.d0}\\% on the same day and @{mi.d1}\\% the next; BET @{mi.b0}\\% and @{mi.b1}\\%', 'Generalizate: DAX @{mi.d0}\\% în aceeași zi și @{mi.d1}\\% în ziua următoare; BET @{mi.b0}\\% și @{mi.b1}\\%'),
     [T('identical to the Cholesky responses with the S\\&P 500 first, as the theory says', 'identice cu răspunsurile Cholesky cu S\\&P 500 primul, conform teoriei')]),
    (T('With the S\\&P 500 last (it closes last), the same-day response is zero by assumption and the next-day responses are @{mi.Bd1}\\% (DAX) and @{mi.Bb1}\\% (BET)', 'Cu S\\&P 500 ultimul (se închide ultimul), răspunsul din aceeași zi este zero prin ipoteză, iar cele din ziua următoare sînt @{mi.Bd1}\\% (DAX) și @{mi.Bb1}\\% (BET)'),
     [T('the same-day correlation (@{mi.cu} with the DAX) is attributed to Europe in this order', 'corelația din aceeași zi (@{mi.cu} cu DAX) este atribuită Europei în această ordine')]),
    T('Either way, the effect is over after two days: markets absorb the news quickly', 'În ambele variante, efectul dispare după două zile: piețele încorporează rapid informația')])

D.recap(('Impulse responses', 'funcții de răspuns la impuls'), [
    T('IRF: columns of $\\bPhi_h$; orthogonalised: $\\boldsymbol{\\Theta}_h = \\bPhi_h\\mathbf{P}$, $\\bSigma = \\mathbf{P}\\mathbf{P}\'$', 'IRF: coloanele lui $\\bPhi_h$; ortogonalizată: $\\boldsymbol{\\Theta}_h = \\bPhi_h\\mathbf{P}$, $\\bSigma = \\mathbf{P}\\mathbf{P}\'$'),
    T('Cholesky = recursive ordering; the ordering is an assumption and can change the results', 'Cholesky = ordine recursivă; ordinea este o ipoteză și poate schimba rezultatele'),
    T('Romania: ROBOR rises for years after an inflation shock; a ``price puzzle\'\' appears', 'România: ROBOR crește ani la rînd după un șoc al inflației; apare „enigma prețurilor”'),
    T('Generalised IRF: order-free, but not additive; bootstrap bands for inference', 'IRF generalizată: nu depinde de ordine, dar nu este aditivă; benzi bootstrap pentru inferență')])

# =============================================================================
# 7b. FEVD
# =============================================================================
D.section('Forecast error variance decomposition', 'Descompunerea varianței erorii de prognoză')

D.frame(T('Forecast error variance decomposition', 'Descompunerea varianței erorii de prognoză'), items(
    (T('$h$-step forecast error: $\\bY_{t+h} - \\hat\\bY_{t+h|t} = \\sum_{s=0}^{h-1}\\bPhi_s\\bepsilon_{t+h-s} = \\sum_{s=0}^{h-1}\\boldsymbol{\\Theta}_s\\mathbf{u}_{t+h-s}$', 'Eroarea de prognoză la $h$ pași: $\\bY_{t+h} - \\hat\\bY_{t+h|t} = \\sum_{s=0}^{h-1}\\bPhi_s\\bepsilon_{t+h-s} = \\sum_{s=0}^{h-1}\\boldsymbol{\\Theta}_s\\mathbf{u}_{t+h-s}$'),
     [T('the orthogonal shocks $u_k$ are uncorrelated, so the variance splits into $K$ parts', 'șocurile ortogonale $u_k$ sînt necorelate, deci varianța se împarte în $K$ părți')]),
    (T('\\textbf{FEVD}: $\\omega_{jk}(h) = \\dfrac{\\sum_{s=0}^{h-1}\\theta_{jk,s}^2}{\\sum_{s=0}^{h-1}\\sum_{m=1}^{K}\\theta_{jm,s}^2}$, the share of shock $k$ in the $h$-step error variance of $Y_j$', '\\textbf{FEVD}: $\\omega_{jk}(h) = \\dfrac{\\sum_{s=0}^{h-1}\\theta_{jk,s}^2}{\\sum_{s=0}^{h-1}\\sum_{m=1}^{K}\\theta_{jm,s}^2}$, ponderea șocului $k$ în varianța erorii de prognoză la $h$ pași a lui $Y_j$'),
     [T('$\\theta_{jk,s}$: element $(j, k)$ of $\\boldsymbol{\\Theta}_s$; $\\hat\\bY_{t+h|t}$: the forecast of $\\bY_{t+h}$ made at $t$', '$\\theta_{jk,s}$: elementul $(j, k)$ al lui $\\boldsymbol{\\Theta}_s$; $\\hat\\bY_{t+h|t}$: prognoza lui $\\bY_{t+h}$ făcută la momentul $t$'),
      T('each row sums to 100\\%; it depends on the Cholesky ordering; at $h = 1$ the first variable is explained 100\\% by its own shock', 'fiecare rînd însumează 100\\%; depinde de ordinea Cholesky; la $h = 1$ prima variabilă este explicată 100\\% de propriul șoc')]),
    (T('Example (VAR(1) of Section 2), $h = 2$: the variance of the error of $y_2$ is @{wx.mse22}; shock $u_1$ explains @{wx.fe21}\\%', 'Exemplu (VAR(1) din secțiunea 2), $h = 2$: varianța erorii pentru $y_2$ este @{wx.mse22}; șocul $u_1$ explică @{wx.fe21}\\%'),
     [T('$(0.5^2 + 0.5^2)/@{wx.mse22}$: the impact $p_{21} = 0.5$ and the next-period response @{wx.th21}', '$(0{,}5^2 + 0{,}5^2)/@{wx.mse22}$: impactul $p_{21} = 0{,}5$ și răspunsul din perioada următoare @{wx.th21}')])))

chart(T('FEVD of the Romanian VAR(2)', 'FEVD pentru VAR(2) românesc'), 'tsa_ch6_fevd_ro', 'TSA_ch6_irf_fevd', [
    T('Share of each orthogonal shock in the forecast error variance, horizons 1--12 quarters, Cholesky order $g, \\pi, i$', 'Ponderea fiecărui șoc ortogonal în varianța erorii de prognoză, orizonturi de 1--12 trimestre, ordinea Cholesky $g, \\pi, i$')],
    h='0.5\\textheight')

interp(('the variance decomposition', 'descompunerii varianței'), [
    (T('ROBOR: own shocks explain @{fe.i.1.i}\\% at one quarter, but only @{fe.i.12.i}\\% after three years; inflation shocks explain @{fe.i.12.pi}\\%, GDP shocks @{fe.i.12.g}\\%', 'ROBOR: propriile șocuri explică @{fe.i.1.i}\\% la un trimestru, dar doar @{fe.i.12.i}\\% după trei ani; șocurile inflației explică @{fe.i.12.pi}\\%, cele ale PIB-ului @{fe.i.12.g}\\%'),
     [T('in the long run, the interest rate is mostly driven by inflation', 'pe termen lung, dobînda este determinată mai ales de inflație')]),
    (T('Inflation: @{fe.pi.12.pi}\\% own shocks even after 12 quarters; GDP growth: @{fe.g.12.g}\\% own shocks', 'Inflația: @{fe.pi.12.pi}\\% propriile șocuri chiar și după 12 trimestre; creșterea PIB: @{fe.g.12.g}\\% propriile șocuri'),
     [T('in this small VAR, inflation is driven by forces outside the model (energy, food, exchange rate, taxes)', 'în acest VAR mic, inflația este determinată de factori din afara modelului (energie, alimente, curs de schimb, taxe)')]),
    T('FEVD and Granger tests agree: the information flows from inflation and output to the interest rate', 'FEVD și testele Granger sînt în acord: informația circulă de la inflație și producție spre dobîndă')])

D.frame(T('Case study: Diebold and Yilmaz (2009, 2012)', 'Studiu de caz: Diebold și Yilmaz (2009, 2012)'), items(
    (T('\\textbf{Question}: how much of the forecast uncertainty of one market comes from shocks in other markets, and how does it change over time?', '\\textbf{Întrebarea}: cît din incertitudinea prognozei unei piețe provine din șocurile altor piețe și cum se schimbă aceasta în timp?'),
     [T('\\refDY: Cholesky FEVD of weekly returns of 19 stock markets; \\refDYb: generalised FEVD, which removes the dependence on the ordering', '\\refDY: FEVD Cholesky pentru randamentele săptămînale a 19 piețe bursiere; \\refDYb: FEVD generalizată, care elimină dependența de ordine')]),
    (T('\\textbf{Spillover index}: $S(H) = \\frac{100}{K}\\sum_{j \\ne k}\\tilde\\omega_{jk}(H)$, the average share of other markets\' shocks', '\\textbf{Indicele de spillover}: $S(H) = \\frac{100}{K}\\sum_{j \\ne k}\\tilde\\omega_{jk}(H)$, ponderea medie a șocurilor celorlalte piețe'),
     [T('$\\tilde\\omega_{jk}$: generalised FEVD rescaled so that each row sums to 1; the specification of \\refDYb: VAR(4), $H = 10$ days, rolling 200-day windows', '$\\tilde\\omega_{jk}$: FEVD generalizată rescalată astfel încît fiecare rînd să însumeze 1; specificația din \\refDYb: VAR(4), $H = 10$ zile, ferestre mobile de 200 de zile')]),
    (T('\\textbf{Finding}: spillovers rise sharply in crises; markets become more connected exactly when diversification is most needed', '\\textbf{Rezultatul}: spillover-ul crește puternic în crize; piețele devin mai conectate exact atunci cînd diversificarea este cea mai necesară'),
     [T('next: the index for the S\\&P 500, the DAX and the BET', 'urmează: indicele pentru S\\&P 500, DAX și BET')])))

D.frame(T('Spillover table: S\\&P 500, DAX, BET', 'Tabelul de spillover: S\\&P 500, DAX, BET'), table(
    'lrrrr', T('\\textbf{To $\\leftarrow$ from (\\%)} & S\\&P 500 & DAX & BET & \\textbf{from others}', '\\textbf{Spre $\\leftarrow$ de la (\\%)} & S\\&P 500 & DAX & BET & \\textbf{de la ceilalți}'),
    ['S\\&P 500 & @{sp.sp.sp} & @{sp.sp.dax} & @{sp.sp.bet} & @{sp.sp.from}',
     'DAX & @{sp.dax.sp} & @{sp.dax.dax} & @{sp.dax.bet} & @{sp.dax.from}',
     'BET & @{sp.bet.sp} & @{sp.bet.dax} & @{sp.bet.bet} & @{sp.bet.from}',
     T('\\textbf{to others}', '\\textbf{spre ceilalți}') + ' & @{sp.to.sp} & @{sp.to.dax} & @{sp.to.bet} & ' + T('index', 'indice') + ' @{sp.tot}'],
    size='footnotesize') + items(
    T('Row $j$: shares of the 10-day forecast error variance of market $j$ due to shocks in each market (generalised FEVD, full sample 2000--2026)', 'Rîndul $j$: ponderile varianței erorii de prognoză la 10 zile a pieței $j$ datorate șocurilor din fiecare piață (FEVD generalizată, întregul eșantion 2000--2026)'),
    T('The S\\&P 500 receives @{sp.sp.dax}\\% from the DAX and the DAX @{sp.dax.sp}\\% from the S\\&P 500; the BET receives @{sp.bet.from}\\% and gives only @{sp.to.bet}\\%: a small, net receiving market', 'S\\&P 500 primește @{sp.sp.dax}\\% de la DAX, iar DAX @{sp.dax.sp}\\% de la S\\&P 500; BET primește @{sp.bet.from}\\% și transmite doar @{sp.to.bet}\\%: o piață mică, receptoare netă')) + ql('TSA_ch6_market_spillovers'),
    'small')

chart(T('The spillover index over time', 'Indicele de spillover în timp'), 'tsa_ch6_spillover', 'TSA_ch6_market_spillovers', [
    T('Total spillover index of S\\&P 500, DAX and BET daily returns: generalised FEVD at 10 days from a VAR(4), rolling windows of 200 trading days (@{sp.mean}\\% on average)',
      'Indicele total de spillover pentru randamentele zilnice S\\&P 500, DAX și BET: FEVD generalizată la 10 zile dintr-un VAR(4), ferestre mobile de 200 de zile de tranzacționare (în medie @{sp.mean}\\%)')],
    h='0.52\\textheight')

interp(('the spillover index', 'indicelui de spillover'), [
    (T('From @{sp.v07}\\% in 2007 to @{sp.v08}\\% in 2008Q4 (after Lehman); maximum @{sp.max}\\% on @{sp.maxd}, in the COVID-19 crash', 'De la @{sp.v07}\\% în 2007 la @{sp.v08}\\% în T4 2008 (după Lehman); maximul @{sp.max}\\% pe @{sp.maxd}, în prăbușirea din pandemia COVID-19'),
     [T('the same pattern as in \\refDY: connectedness jumps in crises and decays slowly', 'același tipar ca în \\refDY: conectarea piețelor crește brusc în crize și scade lent')]),
    (T('Most recent value: @{sp.last}\\%, below the full-sample index (@{sp.tot}\\%)', 'Ultima valoare: @{sp.last}\\%, sub indicele din întregul eșantion (@{sp.tot}\\%)'),
     [T('the jumps are partly mechanical: a 200-day window that contains a crash is dominated by it for the next 200 days', 'salturile sînt în parte mecanice: o fereastră de 200 de zile care conține o prăbușire este dominată de ea timp de 200 de zile')]),
    T('For a portfolio holding the BET: diversification across these markets helps least when it is needed most', 'Pentru un portofoliu care conține BET: diversificarea între aceste piețe ajută cel mai puțin atunci cînd este cea mai necesară')])

D.recap(('Variance decomposition', 'descompunerea varianței'), [
    T('FEVD: share of each orthogonal shock in the $h$-step forecast error variance; rows sum to 100\\%', 'FEVD: ponderea fiecărui șoc ortogonal în varianța erorii de prognoză la $h$ pași; rîndurile însumează 100\\%'),
    T('Romania: ROBOR is driven by inflation at long horizons; inflation mainly by its own shocks', 'România: ROBOR este determinat de inflație pe orizonturi lungi; inflația mai ales de propriile șocuri'),
    T('Diebold--Yilmaz spillover index: the off-diagonal mass of the generalised FEVD', 'Indicele de spillover Diebold--Yilmaz: ponderea totală din afara diagonalei a FEVD generalizate'),
    T('Spillovers among S\\&P 500, DAX and BET peak in crises (2008, 2020)', 'Spillover-ul dintre S\\&P 500, DAX și BET atinge maximele în crize (2008, 2020)')])

# =============================================================================
# 8. PROGNOZA
# =============================================================================
D.section('Forecasting with a VAR', 'Prognoza cu un VAR')

D.frame(T('VAR forecasts and their uncertainty', 'Prognozele VAR și incertitudinea lor'), items(
    (T('\\textbf{Point forecasts} by recursion: $\\hat\\bY_{T+h|T} = \\bc + \\bA_1\\hat\\bY_{T+h-1|T} + \\dots + \\bA_p\\hat\\bY_{T+h-p|T}$, with $\\hat\\bY_{T+j|T} = \\bY_{T+j}$ for $j \\le 0$', '\\textbf{Prognozele punctuale} prin recurență: $\\hat\\bY_{T+h|T} = \\bc + \\bA_1\\hat\\bY_{T+h-1|T} + \\dots + \\bA_p\\hat\\bY_{T+h-p|T}$, cu $\\hat\\bY_{T+j|T} = \\bY_{T+j}$ pentru $j \\le 0$'),
     [T('for a stable VAR they converge to the mean $\\boldsymbol{\\mu}$ as $h$ grows', 'pentru un VAR stabil ele converg spre media $\\boldsymbol{\\mu}$ cînd $h$ crește')]),
    (T('\\textbf{Error covariance}: $\\bSigma(h) = \\sum_{s=0}^{h-1}\\bPhi_s\\bSigma\\bPhi_s\'$; it grows with $h$ towards $\\bGamma(0)$', '\\textbf{Covarianța erorii}: $\\bSigma(h) = \\sum_{s=0}^{h-1}\\bPhi_s\\bSigma\\bPhi_s\'$; crește cu $h$ spre $\\bGamma(0)$'),
     [T('95\\% interval for $Y_j$: $\\hat Y_{j,T+h|T} \\pm 1.96\\sqrt{\\sigma_{jj}(h)}$, $\\sigma_{jj}(h)$: element $(j, j)$ of $\\bSigma(h)$ (Normal shocks; parameter uncertainty ignored)', 'intervalul de 95\\% pentru $Y_j$: $\\hat Y_{j,T+h|T} \\pm 1{,}96\\sqrt{\\sigma_{jj}(h)}$, $\\sigma_{jj}(h)$: elementul $(j, j)$ al lui $\\bSigma(h)$ (șocuri Normale; incertitudinea parametrilor este ignorată)'),
      T('example (Section 2): $\\bSigma(2) = \\bSigma + \\bA\\bSigma\\bA\'$, with variances @{wx.mse11} and @{wx.mse22}', 'exemplu (secțiunea 2): $\\bSigma(2) = \\bSigma + \\bA\\bSigma\\bA\'$, cu varianțele @{wx.mse11} și @{wx.mse22}')]),
    T('Same logic as the ARMA forecasts of Chapter 2, with matrices in place of numbers', 'Aceeași logică precum prognozele ARMA din Capitolul 2, cu matrice în locul numerelor')))

chart(T('Romanian VAR(2): forecasts for two years', 'VAR(2) românesc: prognoze pe doi ani'), 'tsa_ch6_forecast_ro', 'TSA_ch6_forecasting', [
    T('Data until @{fc.lo}; forecasts @{fc.f0}--@{fc.f1} with 95\\% intervals', 'Date pînă în @{fc.lo}; prognoze pentru @{fc.f0}--@{fc.f1}, cu intervale de 95\\%')],
    h='0.5\\textheight')

interp(('the forecasts', 'prognozelor'), [
    (T('Inflation: from @{fc.pi.last}\\% in @{fc.lo} to @{fc.pi.4}\\% after a year and @{fc.pi.8}\\% after two (95\\%: $[@{fc.pi.8.lo}, @{fc.pi.8.hi}]$)', 'Inflația: de la @{fc.pi.last}\\% în @{fc.lo} la @{fc.pi.4}\\% după un an și @{fc.pi.8}\\% după doi (95\\%: $[@{fc.pi.8.lo}, @{fc.pi.8.hi}]$)'),
     [T('the forecasts move towards the VAR mean (inflation @{fc.mu.pi}\\%, ROBOR @{fc.mu.i}\\%), slowly because the largest eigenvalue is @{rt.ro}', 'prognozele se îndreaptă spre media VAR-ului (inflația @{fc.mu.pi}\\%, ROBOR @{fc.mu.i}\\%), lent, deoarece cea mai mare valoare proprie este @{rt.ro}')]),
    (T('ROBOR: @{fc.i.4}\\% after a year, @{fc.i.8}\\% after two; GDP growth: @{fc.g.4}\\% per quarter, with an interval $[@{fc.g.4.lo}, @{fc.g.4.hi}]$', 'ROBOR: @{fc.i.4}\\% după un an, @{fc.i.8}\\% după doi; creșterea PIB: @{fc.g.4}\\% pe trimestru, cu intervalul $[@{fc.g.4.lo}, @{fc.g.4.hi}]$'),
     [T('the GDP interval is wide because the 2009 and 2020 outliers inflate $\\hat\\sigma_g$', 'intervalul PIB-ului este larg pentru că valorile extreme din 2009 și 2020 măresc $\\hat\\sigma_g$')]),
    T('A mechanical forecast: it knows nothing of the budget, energy prices or BNR decisions; its value is as a transparent benchmark', 'O prognoză mecanică: nu știe nimic despre buget, prețurile energiei sau deciziile BNR; valoarea ei este aceea de reper transparent')])

D.frame(T('Evaluating VAR forecasts fairly', 'Evaluarea corectă a prognozelor VAR'), items(
    (T('\\textbf{Pseudo out-of-sample}: re-estimate on data up to $t$ only, forecast $t + h$, move $t$ forward (expanding window)', '\\textbf{Evaluarea pseudo în afara eșantionului} (pseudo out-of-sample): reestimăm doar pe datele pînă la $t$, prognozăm $t + h$, avansăm $t$ (fereastră care crește)'),
     [T('never evaluate on the data used for estimation (Chapter 0)', 'nu evaluăm niciodată pe datele folosite la estimare (Capitolul 0)')]),
    (T('\\textbf{Benchmarks}: the random walk (no change) and a univariate AR$(p)$ for each variable, $p$ by AIC', '\\textbf{Metode de referință}: mersul aleator (nicio schimbare) și un AR$(p)$ univariat pentru fiecare variabilă, $p$ după AIC'),
     [T('the question is not ``is the VAR good?\'\', but ``do the other variables add anything to the series\' own past?\'\'', 'întrebarea nu este „este VAR-ul bun?”, ci „adaugă ceva celelalte variabile la propriul trecut al seriei?”')]),
    (T('Measures: RMSE relative to the random walk (below 1: better); the Diebold--Mariano test \\refDM\\ for equal accuracy', 'Măsuri: RMSE relativ la mersul aleator (sub 1: mai bun); testul Diebold--Mariano \\refDM\\ pentru acuratețe egală'),
     [T('DM statistic: the mean difference of squared errors divided by its standard error; approximately $N(0, 1)$ under $H_0$', 'statistica DM: media diferenței erorilor pătratice împărțită la eroarea ei standard; aproximativ $N(0, 1)$ în ipoteza $H_0$')])))

chart(T('VAR against univariate benchmarks', 'VAR față de metodele univariate de referință'), 'tsa_ch6_oos', 'TSA_ch6_forecasting', [
    T('Relative RMSE (model / random walk). Left: Romania, @{oo.n} forecasts per horizon, origins 2014Q4--2026Q1; right: the US VAR of Section 10, forecasts for 1985Q1--2000Q4, the evaluation period of \\refSW',
      'RMSE relativ (model / mers aleator). Stînga: România, @{oo.n} de prognoze pe orizont, originile T4 2014--T1 2026; dreapta: VAR-ul american din secțiunea 10, prognoze pentru T1 1985--T4 2000, perioada de evaluare din \\refSW')],
    h='0.5\\textheight')

interp(('the forecast comparison', 'comparației prognozelor'), [
    (T('Romania, one quarter ahead: the VAR loses to the AR for GDP growth (@{oo.ro.rv.g.1} against @{oo.ro.ra.g.1}; DM = @{oo.dm.g}, p @{oo.dm.g.p}) and is no better for inflation or ROBOR', 'România, orizontul de un trimestru: VAR-ul este mai puțin precis decît AR pentru creșterea PIB-ului (@{oo.ro.rv.g.1} față de @{oo.ro.ra.g.1}; DM = @{oo.dm.g}, p @{oo.dm.g.p}) și nu este mai bun pentru inflație sau ROBOR'),
     [T('@{ex.k3p2} coefficients on @{oo.n0} to @{e.n} quarters: the estimation noise eats the information of the other variables', '@{ex.k3p2} de coeficienți pe @{oo.n0}--@{e.n} de trimestre: erorile de estimare anulează informația adusă de celelalte variabile')]),
    (T('United States, 8 quarters ahead: the VAR beats the AR for unemployment (@{oo.us.rv.u.8} against @{oo.us.ra.u.8}) and the fed funds rate (@{oo.us.rv.R.8} against @{oo.us.ra.R.8})', 'Statele Unite, orizontul de 8 trimestre: VAR-ul este mai precis decît AR pentru șomaj (@{oo.us.rv.u.8} față de @{oo.us.ra.u.8}) și pentru dobînda federală (@{oo.us.rv.R.8} față de @{oo.us.ra.R.8})'),
     [T('for inflation, nothing beats the random walk (@{oo.us.rv.pi.8}) at this horizon in this period', 'pentru inflație, niciun model nu este mai precis decît mersul aleator (@{oo.us.rv.pi.8}) la acest orizont, în această perioadă')]),
    T('VARs help when the sample is long and the links are strong; in short, noisy samples a simple AR is hard to beat', 'VAR-urile ajută cînd eșantionul este lung și legăturile sînt puternice; în eșantioane scurte și zgomotoase, un AR simplu este greu de depășit')])

D.recap(('Forecasting with a VAR', 'prognoza cu un VAR'), [
    T('Recursive point forecasts; error covariance $\\sum_s\\bPhi_s\\bSigma\\bPhi_s\'$', 'Prognoze punctuale recursive; covarianța erorii $\\sum_s\\bPhi_s\\bSigma\\bPhi_s\'$'),
    T('Evaluate out of sample against the random walk and univariate AR models', 'Evaluăm în afara eșantionului, față de mersul aleator și de modele AR univariate'),
    T('Romania: no gain from the VAR; US 1985--2000: gains for unemployment and the interest rate', 'România: niciun cîștig din VAR; SUA 1985--2000: cîștiguri pentru șomaj și dobîndă')])

# =============================================================================
# 9. VAR STRUCTURAL ȘI STUDIU DE CAZ
# =============================================================================
D.section('Structural VARs and a landmark case study', 'VAR-uri structurale și un studiu de caz de referință')

D.frame(T('From reduced form to structural VAR', 'De la forma redusă la VAR-ul structural'), items(
    (T('\\textbf{Structural VAR} (SVAR): $\\mathbf{B}_0\\bY_t = \\mathbf{d} + \\mathbf{B}_1\\bY_{t-1} + \\dots + \\mathbf{B}_p\\bY_{t-p} + \\mathbf{u}_t$, $\\mathbf{u}_t$ uncorrelated \\textbf{structural shocks}', '\\textbf{VAR structural} (SVAR): $\\mathbf{B}_0\\bY_t = \\mathbf{d} + \\mathbf{B}_1\\bY_{t-1} + \\dots + \\mathbf{B}_p\\bY_{t-p} + \\mathbf{u}_t$, $\\mathbf{u}_t$ \\textbf{șocuri structurale} necorelate'),
     [T('$\\mathbf{d}$: constants; $\\mathbf{B}_1, \\dots, \\mathbf{B}_p$: structural lag coefficients; $\\mathbf{B}_0$: the contemporaneous effects (e.g.\\ the policy rate reacts to this quarter\'s inflation)', '$\\mathbf{d}$: termenii liberi; $\\mathbf{B}_1, \\dots, \\mathbf{B}_p$: coeficienții structurali ai lagurilor; $\\mathbf{B}_0$: efectele contemporane (de exemplu, dobînda de politică monetară reacționează la inflația din acest trimestru)'),
      T('multiplying by $\\mathbf{B}_0^{-1}$ gives the reduced form, with $\\bepsilon_t = \\mathbf{B}_0^{-1}\\mathbf{u}_t$', 'înmulțind cu $\\mathbf{B}_0^{-1}$ obținem forma redusă, cu $\\bepsilon_t = \\mathbf{B}_0^{-1}\\mathbf{u}_t$')]),
    (T('\\textbf{Identification}: $\\bSigma$ has $K(K+1)/2$ distinct elements, $\\mathbf{B}_0^{-1}$ has $K^2$; we need $K(K-1)/2$ restrictions', '\\textbf{Identificarea}: $\\bSigma$ are $K(K+1)/2$ elemente distincte, $\\mathbf{B}_0^{-1}$ are $K^2$; avem nevoie de $K(K-1)/2$ restricții'),
     [T('Cholesky: $\\mathbf{B}_0^{-1} = \\mathbf{P}$ lower triangular: exactly $K(K-1)/2$ zeros; the recursive SVAR of \\refSims', 'Cholesky: $\\mathbf{B}_0^{-1} = \\mathbf{P}$ inferior triunghiulară: exact $K(K-1)/2$ zerouri; SVAR-ul recursiv din \\refSims'),
      T('other schemes: long-run restrictions, sign restrictions, external instruments (beyond this course)', 'alte scheme: restricții pe termen lung, restricții de semn, instrumente externe (dincolo de acest curs)')]),
    T('The data identify the reduced form; the structural interpretation always rests on assumptions that must be argued with economics', 'Datele identifică forma redusă; interpretarea structurală se bazează întotdeauna pe ipoteze care trebuie argumentate economic')))

D.frame(T('Case study: Stock and Watson (2001)', 'Studiu de caz: Stock și Watson (2001)'), '\\begin{columns}[T]\n\\begin{column}{0.4\\textwidth}\n'
        + ph('fed', T('Eccles Building, Washington: the Federal Reserve Board', 'Clădirea Eccles, Washington: Consiliul Rezervei Federale'), h='0.3\\textheight')
        + '\n\\end{column}\n\\begin{column}{0.58\\textwidth}\n' + items(
            (T('\\refSW: ``Vector autoregressions\'\', a widely used survey of the method', '\\refSW: „Vector autoregressions”, o sinteză a metodei, folosită pe scară largă'),
             [T('three variables: inflation $\\pi_t$, unemployment $u_t$, the federal funds rate $R_t$', 'trei variabile: inflația $\\pi_t$, șomajul $u_t$, dobînda federală $R_t$'),
              T('$\\pi_t = 400\\ln(P_t/P_{t-1})$, $P_t$: the GDP price index; 400 annualises, in \\%', '$\\pi_t = 400\\ln(P_t/P_{t-1})$, $P_t$: indicele de preț al PIB; factorul 400 anualizează, în \\%'),
              T('quarterly, 1960Q1--2000Q4, VAR(4), Cholesky order $\\pi, u, R$', 'trimestrial, T1 1960--T4 2000, VAR(4), ordinea Cholesky $\\pi, u, R$')]),
            (T('Four tasks: data description, forecasting, structural inference, policy analysis', 'Patru sarcini: descrierea datelor, prognoza, inferența structurală, analiza politicilor'),
             [T('their verdict: VARs are good at the first two; the last two depend on the identifying assumptions', 'verdictul lor: VAR-urile sînt bune la primele două; ultimele două depind de ipotezele de identificare')]),
            T('We re-estimate it with today\'s FRED data (series GDPCTPI, UNRATE, FEDFUNDS): the numbers differ slightly from the published ones because the data have been revised',
              'Îl reestimăm cu datele FRED de azi (seriile GDPCTPI, UNRATE, FEDFUNDS): cifrele diferă puțin de cele publicate, deoarece datele au fost revizuite'))
        + '\n\\end{column}\n\\end{columns}', 'footnotesize')

chart(T('The three US series', 'Cele trei serii americane'), 'tsa_ch6_us_data', 'TSA_ch6_stock_watson', [
    T('Quarterly, 1960Q1--@{us.last} (FRED); shaded: the sample of \\refSW, @{us.T} quarters', 'Trimestrial, T1 1960--@{us.last} (FRED); zona colorată: eșantionul din \\refSW, @{us.T} de trimestre')],
    h='0.52\\textheight')

chart(T('Stock--Watson VAR: impulse responses', 'VAR-ul Stock--Watson: răspunsuri la impuls'), 'tsa_ch6_sw_irf', 'TSA_ch6_stock_watson', [
    T('VAR(4), 1960Q1--2000Q4, @{sw.k} coefficients; Cholesky order inflation, unemployment, fed funds rate; 24 quarters; 90\\% bootstrap bands',
      'VAR(4), T1 1960--T4 2000, @{sw.k} de coeficienți; ordinea Cholesky inflație, șomaj, dobînda federală; 24 de trimestre; benzi bootstrap de 90\\%')],
    h='0.7\\textheight')

interp(('the Stock--Watson VAR', 'VAR-ului Stock--Watson'), [
    (T('A monetary shock ($+@{sw.RR0}$ pp in the fed funds rate) raises unemployment by up to @{sw.uRmax} pp after @{sw.uRh} quarters', 'Un șoc monetar ($+@{sw.RR0}$ pp în dobînda federală) crește șomajul cu pînă la @{sw.uRmax} pp după @{sw.uRh} trimestre'),
     [T('inflation first rises (@{sw.piR1} pp, a price puzzle), then falls slowly (@{sw.piR12} pp after 3 years)', 'inflația crește întîi (@{sw.piR1} pp, o enigmă a prețurilor), apoi scade lent (@{sw.piR12} pp după 3 ani)')]),
    (T('Granger tests: unemployment $\\to$ inflation (p @{sw.g.upi}), inflation and unemployment $\\to$ fed funds (p @{sw.g.piR}, @{sw.g.uR}), fed funds $\\to$ inflation not significant (p @{sw.g.Rpi})', 'Testele Granger: șomajul $\\to$ inflația (p @{sw.g.upi}), inflația și șomajul $\\to$ dobînda federală (p @{sw.g.piR}, @{sw.g.uR}), dobînda federală $\\to$ inflația nesemnificativ (p @{sw.g.Rpi})'),
     [T('FEVD after 12 quarters: monetary shocks explain @{sw.fe.u.12.R}\\% of unemployment and @{sw.fe.pi.12.R}\\% of inflation', 'FEVD după 12 trimestre: șocurile monetare explică @{sw.fe.u.12.R}\\% din șomaj și @{sw.fe.pi.12.R}\\% din inflație')]),
    T('On the extended sample 1960--@{us.last} ($T = @{sw.Tf}$) the unemployment response has the wrong sign in the first quarters (@{sw.uRf} pp): zero rates after 2008 and the pandemic change the system',
      'Pe eșantionul extins 1960--@{us.last} ($T = @{sw.Tf}$) răspunsul șomajului are semnul greșit în primele trimestre (@{sw.uRf} pp): dobînzile zero de după 2008 și pandemia schimbă sistemul')])

D.frame(T('The legacy of the VAR', 'Moștenirea modelului VAR'), '\\begin{columns}[T]\n\\begin{column}{0.5\\textwidth}\n'
        + ph('nobel', T('Christopher Sims giving his Nobel lecture, Stockholm, 8 December 2011', 'Christopher Sims susținînd prelegerea Nobel, Stockholm, 8 decembrie 2011'), h='0.4\\textheight')
        + '\n\\end{column}\n\\begin{column}{0.48\\textwidth}\n' + items(
            (T('Central banks use VARs for forecasts and policy scenarios', 'Băncile centrale folosesc VAR-uri pentru prognoze și scenarii de politică'),
             [T('and their descendants: Bayesian VARs, DSGE models compared with VARs', 'și modelele derivate din ele: VAR-uri bayesiene, modele DSGE comparate cu VAR-uri')]),
            (T('The price puzzle \\refSimsPP', 'Enigma prețurilor \\refSimsPP'),
             [T('led to adding commodity prices and expectations to monetary VARs', 'a dus la adăugarea prețurilor materiilor prime și a așteptărilor în VAR-urile monetare')]),
            (T('Non-stationary variables that move together need the error-correction form', 'Variabilele nestaționare care evoluează împreună cer forma cu corecția erorii'),
             [T('Chapter 7: cointegration and VECM \\refEG', 'Capitolul 7: cointegrare și VECM \\refEG')]))
        + '\n\\end{column}\n\\end{columns}')

D.recap(('Structural VARs', 'VAR-uri structurale'), [
    T('SVAR: $\\bepsilon_t = \\mathbf{B}_0^{-1}\\mathbf{u}_t$; $K(K-1)/2$ restrictions identify the structural shocks', 'SVAR: $\\bepsilon_t = \\mathbf{B}_0^{-1}\\mathbf{u}_t$; $K(K-1)/2$ restricții identifică șocurile structurale'),
    T('Cholesky is the simplest (recursive) identification', 'Cholesky este cea mai simplă identificare (recursivă)'),
    T('Stock and Watson (2001): a monetary shock raises unemployment; inflation reacts slowly', 'Stock și Watson (2001): un șoc monetar crește șomajul; inflația reacționează lent'),
    T('Results depend on the sample and on the identification', 'Rezultatele depind de eșantion și de identificare')])

# =============================================================================
# 10. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Code}: a first draft of a script that downloads several series, aligns their dates, selects $p$ and estimates a VAR', '\\textbf{Cod}: o primă versiune a unui script care descarcă mai multe serii, le aliniază datele, alege $p$ și estimează un VAR'),
    T('\\textbf{Explanation}: a second explanation of the Cholesky decomposition or of a FEVD table', '\\textbf{Explicații}: o a doua explicație a descompunerii Cholesky sau a unui tabel FEVD'),
    T('\\textbf{Exploration}: Granger tests and spillover indices for many pairs of markets or EU countries at once', '\\textbf{Explorare}: teste Granger și indici de spillover pentru multe perechi de piețe sau de țări din UE deodată'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that builds a quarterly data set for Romania (real GDP growth from Eurostat namq\\_10\\_gdp, 12-month HICP inflation from prc\\_hicp\\_minr, ROBOR 3M from irt\\_st\\_m), chooses the VAR lag order by AIC, BIC and HQ, checks stability and residuals, runs Granger tests and plots orthogonalised impulse responses with bootstrap bands.}',
        '\\aiprompt{Write Python code that builds a quarterly data set for Romania (real GDP growth from Eurostat namq\\_10\\_gdp, 12-month HICP inflation from prc\\_hicp\\_minr, ROBOR 3M from irt\\_st\\_m), chooses the VAR lag order by AIC, BIC and HQ, checks stability and residuals, runs Granger tests and plots orthogonalised impulse responses with bootstrap bands.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('The dates: series aligned on the same quarters, no forward-filled or overlapping values that create spurious leads', 'Datele calendaristice: serii aliniate pe aceleași trimestre, fără valori propagate înainte (forward fill) sau suprapuse, care creează precedențe false'),
    T('Stationarity of each variable (Chapter 3) before Granger tests; with $I(1)$ variables, differences, Toda--Yamamoto or a VECM', 'Staționaritatea fiecărei variabile (Capitolul 3) înaintea testelor Granger; pentru variabile $I(1)$: diferențe, Toda--Yamamoto sau un VECM'),
    T('Stability from the companion matrix, not from $\\bA_1$ alone', 'Stabilitatea din matricea companion, nu doar din $\\bA_1$'),
    T('The Cholesky ordering used and its justification; the same results with another ordering or with generalised responses', 'Ordinea Cholesky folosită și justificarea ei; aceleași rezultate cu altă ordine sau cu răspunsuri generalizate'),
    T('That ``Granger-causes\'\' is never reported as ``causes\'\'', 'Că „cauzează în sens Granger” nu este raportat niciodată drept „cauzează”'),
    T('Every cited reference: it must exist; check the DOI', 'Fiecare referință citată: trebuie să existe; verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('A VAR$(p)$ regresses each variable on the past of all; OLS equation by equation; few variables and lags', 'Un VAR$(p)$ regresează fiecare variabilă pe trecutul tuturor; OLS ecuație cu ecuație; puține variabile și laguri'),
    T('Stability: all eigenvalues of the companion matrix inside the unit circle; then a mean and a moving-average form exist', 'Stabilitatea: toate valorile proprii ale matricei companion în interiorul cercului unitate; atunci există o medie și o formă de medie mobilă'),
    T('Choose $p$ with AIC, BIC, HQ, then check the residuals with the portmanteau test', 'Alegem $p$ cu AIC, BIC, HQ, apoi verificăm reziduurile cu testul portmanteau'),
    T('Granger causality = incremental predictability, relative to the chosen variables', 'Cauzalitatea Granger = predictibilitate suplimentară, relativă la variabilele alese'),
    T('IRF and FEVD need orthogonal shocks: the Cholesky ordering is an identifying assumption', 'IRF și FEVD cer șocuri ortogonale: ordinea Cholesky este o ipoteză de identificare'),
    T('VAR forecasts must beat AR and random-walk benchmarks out of sample; often they do not', 'Prognozele VAR trebuie să fie mai precise, în afara eșantionului, decît modelele AR și mersul aleator; adesea nu sînt')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.35}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    ['VAR$(p)$ & $\\bY_t = \\bc + \\bA_1\\bY_{t-1} + \\dots + \\bA_p\\bY_{t-p} + \\bepsilon_t$, \\quad $E[\\bepsilon_t\\bepsilon_t\'] = \\bSigma$',
     T('Stability, mean', 'Stabilitate, medie') + ' & $|\\lambda_i(\\mathbf{F})| < 1$; \\quad $\\boldsymbol{\\mu} = (\\mathbf{I} - \\bA_1 - \\dots - \\bA_p)^{-1}\\bc$',
     T('Criteria', 'Criterii') + ' & $\\ln\\det\\tilde\\bSigma(p) + c_T\\,pK^2/T$, \\quad $c_T = 2,\\ \\ln T,\\ 2\\ln\\ln T$',
     T('Portmanteau', 'Portmanteau') + ' & $Q_h \\sim \\chi^2(K^2(h - p))$',
     'Granger $F$ & $\\dfrac{(RSS_R - RSS_U)/p}{RSS_U/(T - Kp - 1)} \\sim F(p, T - Kp - 1)$',
     T('MA form, IRF', 'Forma MA, IRF') + ' & $\\bPhi_h = \\sum_{j=1}^{\\min(h,p)}\\bA_j\\bPhi_{h-j}$, \\quad $\\boldsymbol{\\Theta}_h = \\bPhi_h\\mathbf{P}$, \\quad $\\bSigma = \\mathbf{P}\\mathbf{P}\'$',
     'FEVD & $\\omega_{jk}(h) = \\sum_{s<h}\\theta_{jk,s}^2 / \\sum_{s<h}\\sum_m\\theta_{jm,s}^2$',
     T('Forecast error', 'Eroarea de prognoză') + ' & $\\bSigma(h) = \\sum_{s=0}^{h-1}\\bPhi_s\\bSigma\\bPhi_s\'$'],
    size='footnotesize') + '}')

D.frame(T('Self-assessment', 'Autoevaluare'), items(
    (T('\\textbf{Question}: is the VAR(1) with $\\bA = \\begin{pmatrix} 0.9 & 0.3 \\\\ 0.2 & 0.8 \\end{pmatrix}$ stable?', '\\textbf{Întrebare}: este stabil VAR(1) cu $\\bA = \\begin{pmatrix} 0{,}9 & 0{,}3 \\\\ 0{,}2 & 0{,}8 \\end{pmatrix}$?'),
     [T('\\textbf{Answer}: no: $\\lambda^2 - 1.7\\lambda + 0.66 = 0$ gives $\\lambda = 1.1$ and $0.6$', '\\textbf{Răspuns}: nu: $\\lambda^2 - 1{,}7\\lambda + 0{,}66 = 0$ dă $\\lambda = 1{,}1$ și $0{,}6$')]),
    (T('\\textbf{Question}: a Granger test gives p = 0.30 for $x \\to y$. Does $x$ have no effect on $y$?', '\\textbf{Întrebare}: un test Granger dă p = 0,30 pentru $x \\to y$. Nu are $x$ niciun efect asupra lui $y$?'),
     [T('\\textbf{Answer}: no such conclusion: the lags of $x$ do not improve the forecast; effects within the period, slow effects or low power remain possible', '\\textbf{Răspuns}: nu putem concluziona asta: lagurile lui $x$ nu îmbunătățesc prognoza; rămîn posibile efecte în aceeași perioadă, efecte lente sau o putere mică a testului')]),
    (T('\\textbf{Question}: in a Cholesky FEVD, what share of the 1-step error variance of the first variable comes from its own shock?', '\\textbf{Întrebare}: într-o FEVD Cholesky, ce pondere din varianța erorii la un pas a primei variabile provine din propriul șoc?'),
     [T('\\textbf{Answer}: 100\\%, by construction of the ordering', '\\textbf{Răspuns}: 100\\%, prin construcția ordinii')]),
    T('Next: Chapter 7, cointegration and the vector error correction model (VECM)', 'Urmează: Capitolul 7, cointegrarea și modelul vectorial cu corecția erorii (VECM)')))

D.references(bib())

if __name__ == '__main__':
    finalize(D.write(V))
