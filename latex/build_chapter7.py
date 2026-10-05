r"""
build_chapter7.py -- Capitolul 7 (Cointegrare și VECM), EN + RO dintr-o singură sursă
====================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_07/ch7_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter7_cointegration_vecm.tex
  RO/Cursuri/capitol7_cointegrare_vecm.tex
Rulare:
  python3 Quantlets/Ch_07/generate_all_charts.py
  python3 latex/build_chapter7.py && python3 latex/tsa_build.py compile 7
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, cols, table, photo   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch7_common import QLURL, REFS, T, bib, finalize, load, pv, month, qtr   # noqa: E402


def items(*xs):
    """tsa_build.items, with (text, []) treated as a plain bullet."""
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


N = load()
V = Values()
D = Deck(7, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\quantlet{{{folder.replace("_", chr(92) + "_")}}}{{\\qlurl{{{folder}}}}}'


def chart(title, fig, folder, bullets, h='0.60\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def interp(title, bullets, size='small'):
    D.frame(T(f'Interpreting {title[0]}', f'Interpretarea {title[1]}'), items(*bullets), size)


PH = {
    'dog': ('ch7_dog_walking_2018.jpg', C + 'Dog_walking_woman.jpg', T('Photo', 'Foto') + ': Amin (2018); CC BY-SA 4.0; Wikimedia Commons'),
    'engle': ('ch7_robert_engle_2022.jpg', C + '0603_KRBN-RobertEngle-JonDemske-10.jpg',
              T('Photo', 'Foto') + ': Jon Demske (2022); CC BY-SA 4.0; Wikimedia Commons'),
    'granger': ('ch3_clive_granger_2008.jpg', C + 'Clive_Granger_by_Olaf_Storbeck_(3x4_cropped).jpg',
                T('Photo', 'Foto') + ': Olaf Storbeck (2008); CC BY-SA 2.0; Wikimedia Commons'),
    'nottingham': ('ch7_granger_building_2012.jpg', C + 'University_Park_MMB_\\%C2\\%AB24_Sir_Clive_Granger_Building.jpg',
                   T('Photo', 'Foto') + ': mattbuck (2012); CC BY-SA 3.0; Wikimedia Commons'),
    'copenhagen': ('ch7_copenhagen_university_2011.jpg', C + 'Copenhagen_University_Main_Entrance_DSC09700.jpg',
                   T('Photo', 'Foto') + ': Per Meistrup (2011); CC BY-SA 4.0; Wikimedia Commons'),
    'bnr': ('ch7_bnr_palace_2015.jpg', C + 'Bucharest_-_BNR_Palace_(19644434340).jpg',
            T('Photo', 'Foto') + ': Ștefan Jurcă (2015); CC BY 2.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# CIFRE
# =============================================================================
DR = N['drunk']
V.put('dr.rho', DR['rho'], 1)
V.put('dr.c', DR['c'], 2)
V.put('dr.d', DR['d'], 2)
V.put('dr.half', DR['half'], 1)
V.raw('dr.T', str(DR['T']))
V.raw('dr.pd', pv(DR['adf_dist']['p']))
V.put('dr.ps', DR['adf_stray']['p'], 2)
V.put('dr.px', DR['adf_x']['p'], 2)
V.put('dr.sdd', DR['sd_dist'], 1)
V.put('dr.sds', DR['sd_stray'], 1)
V.put('dr.egtau', DR['eg']['tau'], 2)
V.put('dr.egb', DR['eg']['beta'][0], 2)

EX = N['examples']
for k in ['rates', 'cons', 'banks', 'fx']:
    V.put(f'ex.{k}.tau', EX[k]['tau'], 2)
    V.raw(f'ex.{k}.p', pv(EX[k]['p']))
    V.put(f'ex.{k}.b', EX[k]['beta'][0], 2)
    V.put(f'ex.{k}.cv', EX[k]['crit5'], 2)
V.put('ex.fx.r2', EX['fx']['r2'], 2)
V.put('ex.tb3.p', EX['adf_rates_tb3']['p'], 2)
V.put('ex.gs10.p', EX['adf_rates_gs10']['p'], 2)
V.put('ex.cc.p', EX['adf_cons_c']['p'], 2)
V.put('ex.cy.p', EX['adf_cons_y']['p'], 2)

ED = N['egdist']
V.int('ed.R', ED['R'])
V.raw('ed.T', str(ED['T']))
for k in ['1', '2', '3']:
    V.put(f'ed.{k}.q05', ED[k]['q05'], 2)
    V.put(f'ed.{k}.mk5', ED[k]['mk5'], 2)
for k in ['2', '3']:
    V.put(f'ed.{k}.size', 100 * ED[k]['size_with_df'], 1)
for k in ['1', '2', '3', '4']:
    for j, lev in enumerate(['1', '5', '10']):
        V.put(f'cv.c.{k}.{lev}', ED['table'][k][j], 2)
        V.put(f'cv.ct.{k}.{lev}', ED['table_ct'][k][j], 2)

EGT = N['egtab']
EGROWS = [('US consumption on income', T('US consumption on income', 'SUA: consumul pe venit'), T('quarterly', 'trimestrial')),
          ('US 10-year on 3-month yield', T('US: 10-year on 3-month yield', 'SUA: 10 ani pe 3 luni'), T('monthly', 'lunar')),
          ('TLV on BRD', T('TLV on BRD (since 2014)', 'TLV pe BRD (din 2014)'), T('daily', 'zilnic')),
          ('TLV on BRD, reversed', T('BRD on TLV (since 2014)', 'BRD pe TLV (din 2014)'), T('daily', 'zilnic')),
          ('TLV on BRD, from 2010', T('TLV on BRD (since 2010)', 'TLV pe BRD (din 2010)'), T('daily', 'zilnic')),
          ('EUR/RON on EUR/HUF', T('EUR/RON on EUR/HUF', 'EUR/RON pe EUR/HUF'), T('monthly', 'lunar')),
          ('Romania consumption on GDP', T('Romania: consumption on GDP', 'România: consumul pe PIB'), T('quarterly', 'trimestrial'))]


def egrow(i):
    key, lab, freq = EGROWS[i]
    e = EGT[key]
    V.int(f'eg{i}.n', e['n'])
    V.put(f'eg{i}.b', e['beta'][0], 2)
    V.put(f'eg{i}.tau', e['tau'], 2)
    V.raw(f'eg{i}.p', pv(e['p']))
    V.put(f'eg{i}.cv', e['crit5'], 2)
    V.put(f'eg{i}.po', e['po_zt'], 2)
    V.raw(f'eg{i}.pop', pv(e['po_p']))
    V.raw(f'eg{i}.k', str(e['lags']))
    return (f'{lab} & {freq} & @{{eg{i}.n}} & @{{eg{i}.b}} & $@{{eg{i}.tau}}$ (@{{eg{i}.k}}) & @{{eg{i}.p}} & '
            f'$@{{eg{i}.po}}$ & @{{eg{i}.pop}}')


EGR = [egrow(i) for i in range(len(EGROWS))]
V.put('eg.cons.b', EGT['US consumption on income']['beta'][0], 3)
V.put('eg.cons.a', EGT['US consumption on income']['const'], 1)
V.put('eg.cons.dw', EGT['US consumption on income']['dw'], 2)
V.put('eg.cons.r2', EGT['US consumption on income']['r2'], 4)
V.put('eg.cons.cv', EGT['US consumption on income']['crit5'], 2)
V.put('eg.cons.adfcv', EGT['US consumption on income']['adf_crit5'], 2)
V.put('eg.inc.tau', EGT['US income on consumption']['tau'], 2)
V.put('eg.inc.b', EGT['US income on consumption']['beta'][0], 3)
V.put('eg.inc.b1', 1 / EGT['US income on consumption']['beta'][0], 3)
V.raw('eg.ro.k', str(EGT['Romania consumption on GDP']['lags']))

ES = N['egsteps']
V.put('es.c1', ES['acf1_cons'], 2)
V.put('es.r1', ES['acf1_rates'], 2)
V.put('es.r12', ES['acf12_rates'], 2)
V.put('es.c8', ES['acf8_cons'], 2)

EC = N['ecm']
for k in ['cons', 'long']:
    V.put(f'ec.{k}.g', EC[k]['gamma'], 3)
    V.put(f'ec.{k}.t', EC[k]['gamma_t'], 2)
    V.put(f'ec.{k}.h', EC[k]['half'], 1)
    V.put(f'ec.{k}.d0', EC[k]['delta0'], 2)
    V.put(f'ec.{k}.b', EC[k]['beta'], 3)
    V.put(f'ec.{k}.c', EC[k]['const'], 3)
V.put('ec.short.g', EC['short']['gamma'], 3)
V.put('ec.short.t', EC['short']['gamma_t'], 2)
# worked example: consumption 2% above equilibrium, income growth 1% this quarter
gap, dy = 2.0, 1.0
V.put('wx.gap', gap, 0)
V.put('wx.dy', dy, 0)
V.put('wx.corr', EC['cons']['gamma'] * gap, 3)
V.put('wx.short', EC['cons']['delta0'] * dy, 3)
V.put('wx.dc', EC['cons']['const'] + EC['cons']['delta0'] * dy + EC['cons']['gamma'] * gap, 2)
V.put('wx.hl', math.log(0.5) / math.log(0.8), 1)

JO = N['joh']
SYS = [('US yields 1y, 5y, 10y', T('US yields 1y, 5y, 10y', 'SUA: randamentele la 1, 5 și 10 ani')),
       ('US output, consumption, investment', T('US output, consumption, investment', 'SUA: PIB, consum, investiții')),
       ('EUR/RON, EUR/HUF, EUR/PLN', 'EUR/RON, EUR/HUF, EUR/PLN')]
JROWS = []
for i, (k, lab) in enumerate(SYS):
    j = JO[k]
    V.int(f'j{i}.n', j['n'])
    V.raw(f'j{i}.k', str(j['k_ar_diff']))
    V.raw(f'j{i}.rt', str(j['rank_trace']))
    V.raw(f'j{i}.rm', str(j['rank_maxeig']))
    for r in range(3):
        V.put(f'j{i}.t{r}', j['trace'][r], 1)
        V.put(f'j{i}.m{r}', j['maxeig'][r], 1)
        V.put(f'j{i}.e{r}', j['eig'][r], 4)
    JROWS.append((lab, i))
for r in range(3):
    V.put(f'jcv.t{r}', JO['US yields 1y, 5y, 10y']['trace_cv5'][r], 2)
    V.put(f'jcv.m{r}', JO['US yields 1y, 5y, 10y']['maxeig_cv5'][r], 2)
# worked example: trace from the eigenvalues of the yields system
lam = JO['US yields 1y, 5y, 10y']['eig']
Tn = JO['US yields 1y, 5y, 10y']['n']
V.put('wj.l0', -math.log(1 - lam[0]), 4)
V.put('wj.l1', -math.log(1 - lam[1]), 4)
V.put('wj.l2', -math.log(1 - lam[2]), 4)
V.put('wj.tr0', -Tn * sum(math.log(1 - x) for x in lam), 1)
V.put('wj.tr1', -Tn * sum(math.log(1 - x) for x in lam[1:]), 1)
V.put('wj.mx0', -Tn * math.log(1 - lam[0]), 1)
YD = JO['yields_det']
for d in ['-1', '0', '1']:
    for r in range(3):
        V.put(f'yd.{d}.t{r}', YD[d]['trace'][r], 1)
        V.put(f'yd.{d}.c{r}', YD[d]['trace_cv5'][r], 2)
    V.raw(f'yd.{d}.r', str(YD[d]['rank_trace']))

VE = N['vecm']
B = VE['beta']
V.put('ve.b1', -B[2][0], 3)
V.put('ve.b2', -B[2][1], 3)
V.put('ve.c1', VE['const_coint'][0], 2)
V.put('ve.c2', VE['const_coint'][1], 2)
for i in range(3):
    for j in range(2):
        V.put(f've.a{i}{j}', VE['alpha'][i][j], 3)
        V.put(f've.t{i}{j}', VE['alpha_t'][i][j], 2)
V.put('ve.eig0', VE['ec_eig'][0], 3)
V.put('ve.eig1', VE['ec_eig'][1], 3)
V.put('ve.h0', VE['ec_half'][0], 1)
V.put('ve.h1', VE['ec_half'][1], 1)
V.raw('ve.p0', pv(VE['ec_adf'][0]['p']))
V.raw('ve.p1', pv(VE['ec_adf'][1]['p']))
V.raw('ve.k', str(VE['k_ar_diff']))
V.raw('ve.first', month(VE['first']))
V.raw('ve.last', month(VE['last']))
V.int('ve.n', VE['n'])

IR = N['irf']
for i in range(3):
    V.put(f'ir.i{i}', IR['impact'][i], 2)
    V.put(f'ir.h{i}', IR['hH'][i], 2)
    V.put(f'ir.m{i}', IR['h12'][i], 2)
V.raw('ir.H', str(IR['H']))

FC = N['fc']
V.raw('fc.n', str(FC['n_origins']))
V.raw('fc.o0', month(FC['first_origin']))
V.raw('fc.o1', month(FC['last_origin']))
for h in ['1', '6', '12', '24', '36']:
    for j in range(3):
        V.put(f'fc.v{h}.{j}', FC['ratio_vecm'][h][j], 2)
        V.put(f'fc.d{h}.{j}', FC['ratio_dvar'][h][j], 2)
    V.put(f'fc.sv{h}', FC['spread_vecm'][h], 2)
    V.put(f'fc.sd{h}', FC['spread_dvar'][h], 2)

PP = N['ppp']
V.put('pp.s', PP['s_change'], 1)
V.put('pp.rel', PP['rel_change'], 1)
V.put('pp.q', PP['q_change'], 1)
V.put('pp.adf', PP['adf_q']['p'], 2)
V.put('pp.eg', PP['eg']['p'], 2)
V.put('pp.egb', PP['eg']['beta'][0], 2)
V.put('pp.jt0', PP['johansen']['trace'][0], 1)
V.raw('pp.jr', str(PP['johansen']['rank_trace']))
V.raw('pp.first', month(PP['first']))
V.raw('pp.last', month(PP['last']))

RE = N['roea']
V.put('re.full.p', RE['full']['p'], 2)
V.raw('re.full.pop', pv(RE['full']['po_p']))
V.put('re.10.p', RE['since2010']['p'], 3)
V.put('re.10.pop', RE['since2010']['po_p'], 3)
V.put('re.10.b', RE['since2010']['beta'][0], 2)
V.put('re.sp', RE['spread_mean10'], 2)
V.put('re.g', RE['ecm10']['gamma'], 3)
V.put('re.gt', RE['ecm10']['gamma_t'], 2)
V.put('re.h', RE['ecm10']['half'], 0)
V.raw('re.first', month(RE['first']))
V.raw('re.last', month(RE['last']))

CE = N['cee']
for c, k in [('EUR/RON', 'ron'), ('EUR/HUF', 'huf'), ('EUR/PLN', 'pln')]:
    V.put(f'ce.{k}', CE['change'][c], 1)
    V.put(f'ce.{k}.p', CE['adf'][c]['p'], 2)
V.put('ce.t0', CE['johansen']['trace'][0], 1)
V.put('ce.t12', CE['johansen12']['trace'][0], 1)

PA = N['pair']
V.put('pa.b', PA['beta'], 2)
V.put('pa.rho', PA['rho'], 3)
V.put('pa.h', PA['half'], 0)
V.put('pa.out', 100 * PA['share_out'], 1)

PB = N['pairs']
for k in ['bvb', 'us']:
    for g in ['gross', 'net']:
        V.put(f'pb.{k}.{g}.m', PB[k][g]['mean'], 2)
        V.put(f'pb.{k}.{g}.s', PB[k][g]['sharpe'], 2)
    V.raw(f'pb.{k}.w', str(PB[k]['windows']))
    V.raw(f'pb.{k}.wp', str(PB[k]['windows_with_pairs']))
    V.raw(f'pb.{k}.tr', str(PB[k]['trades']))
    V.put(f'pb.{k}.win', 100 * PB[k]['win_rate'], 0)
    V.put(f'pb.{k}.mt', 100 * PB[k]['mean_trade'], 2)
    V.put(f'pb.{k}.be', 100 * PB[k]['breakeven'], 2)
    V.put(f'pb.{k}.cost', 100 * PB[k]['cost'], 2)
    V.put(f'pb.{k}.sel', PB[k]['selected_mean'], 1)
V.put('pb.in.m', PB['insample']['mean'], 2)
V.put('pb.in.s', PB['insample']['sharpe'], 2)
V.raw('pb.tb.w', str(PB['tlvbrd_rolling']['windows']))
V.put('pb.fp', 0.05 * 28, 1)
from statsmodels.tsa.adfvalues import mackinnoncrit   # noqa: E402
V.put('sa.eg', mackinnoncrit(N=2, regression='c', nobs=300)[1], 2)
V.put('sa.df', mackinnoncrit(N=1, regression='c', nobs=300)[1], 2)
V.put('sa.hl', math.log(0.5) / math.log(0.9), 1)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: two prices both wander like random walks; can a combination of them be stable, and how do we model it?',
       '\\textbf{Întrebarea}: două prețuri evoluează amîndouă ca niște mersuri aleatoare; poate fi stabilă o combinație a lor și cum o modelăm?'),
     [T('if yes, the series share a common stochastic trend and a long-run equilibrium: regressions in levels are meaningful, not spurious',
        'dacă da, seriile au un trend stochastic comun și un echilibru pe termen lung: regresiile în niveluri au sens și nu sînt false')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('the idea of cointegration; the Engle--Granger two-step method with the right critical values; error correction models',
        'ideea de cointegrare; metoda Engle--Granger în doi pași, cu valorile critice potrivite; modelele cu corecția erorii'),
      T('the VECM and the Granger representation theorem; the Johansen tests; estimating and reading $\\beta$ and $\\alpha$; forecasting',
        'modelul VECM și teorema de reprezentare a lui Granger; testele Johansen; estimarea și interpretarea lui $\\beta$ și $\\alpha$; prognoza'),
      T('applications: purchasing power parity, interest rates, three Central European currencies, pairs trading on the Bucharest Stock Exchange',
        'aplicații: paritatea puterii de cumpărare, ratele dobînzii, trei monede central-europene, pairs trading la Bursa de Valori București')]),
    T('We build on Chapter 3 (unit roots, spurious regression) and Chapter 6 (VAR models); Seminar 7 comes before this lecture',
      'Pornim de la Capitolul 3 (rădăcini unitare, regresia falsă) și de la Capitolul 6 (modele VAR); Seminarul 7 are loc înaintea acestui curs')))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Define cointegration and explain it as a common stochastic trend', 'Definiți cointegrarea și explicați-o ca trend stochastic comun'),
    T('Run the Engle--Granger and Phillips--Ouliaris tests and compare them with the right (not the Dickey--Fuller) critical values',
      'Aplicați testele Engle--Granger și Phillips--Ouliaris și comparați statisticile cu valorile critice potrivite (nu cu cele Dickey--Fuller)'),
    T('Estimate an error correction model and interpret the speed of adjustment and the half-life', 'Estimați un model cu corecția erorii și interpretați viteza de ajustare și timpul de înjumătățire'),
    T('Write a VAR as a VECM, state the Granger representation theorem and choose the cointegration rank with the Johansen tests',
      'Scrieți un VAR ca VECM, enunțați teorema de reprezentare a lui Granger și alegeți rangul de cointegrare cu testele Johansen'),
    T('Interpret cointegrating vectors and adjustment coefficients, and decide when a VECM forecasts better than a VAR in differences',
      'Interpretați vectorii de cointegrare și coeficienții de ajustare și decideți cînd un VECM prognozează mai bine decît un VAR în diferențe'),
    T('Judge an application (parity conditions, pairs trading) out of sample and net of costs', 'Evaluați o aplicație (condiții de paritate, pairs trading) în afara eșantionului și după costuri')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refHP, Ch.~9 (nonstationarity and cointegration)', 'Manual: \\refHP, cap.~9 (nestaționaritate și cointegrare)'),
     [T('theory: \\refHamilton, Ch.~19--20; \\refLut, Ch.~6--8; the likelihood approach in \\refJohBook\\ and \\refJus', 'teorie: \\refHamilton, cap.~19--20; \\refLut, cap.~6--8; abordarea prin verosimilitate în \\refJohBook\\ și \\refJus')]),
    T('Landmark papers: \\refEG, \\refJohA, \\refJohB; Granger\'s Nobel lecture \\refGrangerN', 'Lucrări de referință: \\refEG, \\refJohA, \\refJohB; prelegerea Nobel a lui Granger, \\refGrangerN'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_07}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_07}'),
     [T('\\texttt{coint}, \\texttt{coint\\_johansen}, \\texttt{VECM} and \\texttt{VAR} from \\texttt{statsmodels}; \\texttt{phillips\\_ouliaris} from \\texttt{arch}',
        '\\texttt{coint}, \\texttt{coint\\_johansen}, \\texttt{VECM} și \\texttt{VAR} din \\texttt{statsmodels}; \\texttt{phillips\\_ouliaris} din \\texttt{arch}')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter7_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter7_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. IDEEA DE COINTEGRARE
# =============================================================================
D.section('The idea of cointegration', 'Ideea de cointegrare')

D.frame(T('From spurious regression to cointegration', 'De la regresia falsă la cointegrare'), items(
    (T('Chapter 3: regressing one $I(1)$ series on another, \\textbf{unrelated} $I(1)$ series gives large $t$, high $R^2$ and a Durbin--Watson statistic near 0 \\refGN',
       'Capitolul 3: regresia unei serii $I(1)$ pe o altă serie $I(1)$, \\textbf{fără legătură} cu ea, dă un $t$ mare, un $R^2$ mare și o statistică Durbin--Watson apropiată de 0 \\refGN'),
     [T('the error $y_t - a - b x_t$ is itself $I(1)$ for every $b$: nothing ties the two series together', 'eroarea $y_t - a - b x_t$ este ea însăși $I(1)$ pentru orice $b$: nimic nu leagă cele două serii')]),
    (T('The exception: for one value of $b$, the combination $u_t = y_t - a - b x_t$ is \\textbf{stationary}', 'Excepția: pentru o anumită valoare a lui $b$, combinația $u_t = y_t - a - b x_t$ este \\textbf{staționară}'),
     [T('then $y_t$ and $x_t$ are \\textbf{cointegrated} \\refGranger, \\refEG: the regression in levels estimates a real long-run relation',
        'atunci $y_t$ și $x_t$ sînt \\textbf{cointegrate} \\refGranger, \\refEG: regresia în niveluri estimează o relație reală pe termen lung'),
      T('the two series may drift apart for a while, but $u_t$ always pulls them back: an \\textbf{equilibrium}', 'cele două serii se pot îndepărta o vreme, dar $u_t$ le aduce mereu înapoi: un \\textbf{echilibru}')]),
    T('Today: how to recognise this case, how to test it, and how to model it', 'Azi: cum recunoaștem acest caz, cum îl testăm și cum îl modelăm')))

D.frame(T('A drunk and her dog', 'O femeie beată și cîinele ei'), cols(
    ph('dog', T('A walk with a dog: two paths that never drift far apart', 'O plimbare cu cîinele: două traiectorii care nu se îndepărtează mult una de alta'), h='0.40\\textheight'),
    items((T('\\refMurray: a drunk leaves the pub and wanders: her path $x_t$ is a random walk', '\\refMurray: o femeie beată iese din local și rătăcește: traiectoria ei $x_t$ este un mers aleator'),
           [T('her dog also wanders: $y_t$ is a random walk too', 'cîinele ei rătăcește și el: $y_t$ este tot un mers aleator'),
            T('the distance between two \\textbf{independent} walkers grows without limit, like $\\sqrt{t}$', 'distanța dintre doi trecători \\textbf{independenți} crește fără limită, ca $\\sqrt{t}$')]),
          (T('But the dog hears her call and runs back; she hears the dog and turns towards it', 'Dar cîinele îi aude strigătul și aleargă înapoi; ea aude cîinele și se întoarce spre el'),
           [T('each corrects part of the gap: $\\Delta x_t = c\\,(y_{t-1} - x_{t-1}) + u_t$, $\\Delta y_t = d\\,(x_{t-1} - y_{t-1}) + w_t$', 'fiecare corectează o parte din distanță: $\\Delta x_t = c\\,(y_{t-1} - x_{t-1}) + u_t$, $\\Delta y_t = d\\,(x_{t-1} - y_{t-1}) + w_t$'),
            T('both paths remain unpredictable, but the distance $y_t - x_t$ stays bounded: \\textbf{cointegration} with \\textbf{error correction}', 'ambele traiectorii rămîn imprevizibile, dar distanța $y_t - x_t$ rămîne mărginită: \\textbf{cointegrare} cu \\textbf{corecția erorii}')])),
    wl='0.38', wr='0.58'))

chart(T('The drunk and her dog by simulation', 'Femeia beată și cîinele ei prin simulare'), 'tsa_ch7_drunk_dog', 'TSA_ch7_common_trends', [
    T('$T = @{dr.T}$, Gaussian shocks; $c = @{dr.c}$, $d = @{dr.d}$; a stray dog $z_t$ is an independent random walk with the same shocks\' variance',
      '$T = @{dr.T}$, șocuri gaussiene; $c = @{dr.c}$, $d = @{dr.d}$; un cîine fără stăpîn $z_t$ este un mers aleator independent, cu aceeași varianță a șocurilor')], h='0.56\\textheight')

interp(('the simulation', 'simulării'), [
    (T('Each path is $I(1)$: ADF on $x_t$ gives p = @{dr.px}', 'Fiecare traiectorie este $I(1)$: ADF pe $x_t$ dă p = @{dr.px}'),
     [T('the distance $y_t - x_t$ is an AR(1) with $\\rho = 1 - c - d = @{dr.rho}$: ADF p @{dr.pd}, standard deviation @{dr.sdd}', 'distanța $y_t - x_t$ este un AR(1) cu $\\rho = 1 - c - d = @{dr.rho}$: ADF p @{dr.pd}, abaterea standard @{dr.sdd}'),
      T('the distance to the stray dog wanders: ADF p = @{dr.ps}, standard deviation @{dr.sds}', 'distanța față de cîinele fără stăpîn rătăcește: ADF p = @{dr.ps}, abaterea standard @{dr.sds}')]),
    (T('Half of a gap disappears in $\\ln 0.5/\\ln @{dr.rho} = @{dr.half}$ steps: the \\textbf{half-life} (Section 3)', 'Jumătate dintr-o distanță dispare în $\\ln 0{,}5/\\ln @{dr.rho} = @{dr.half}$ pași: \\textbf{timpul de înjumătățire} (secțiunea 3)'),
     [T('knowing where the drunk is tells us where to look for the dog, even though neither path can be forecast', 'știind unde este femeia, știm unde să căutăm cîinele, deși niciuna dintre traiectorii nu poate fi prognozată')])])

D.frame(T('Definition of cointegration', 'Definiția cointegrării'), items(
    (T('\\textbf{Definition} \\refEG: the components of $\\mathbf y_t = (y_{1t}, \\dots, y_{nt})^\\top$ are \\textbf{cointegrated of order (1,1)}, $\\mathbf y_t \\sim CI(1,1)$, if',
       '\\textbf{Definiție} \\refEG: componentele lui $\\mathbf y_t = (y_{1t}, \\dots, y_{nt})^\\top$ sînt \\textbf{cointegrate de ordinul (1,1)}, $\\mathbf y_t \\sim CI(1,1)$, dacă'),
     [T('every component is $I(1)$ (Chapter 3: stationary after one difference), and', 'fiecare componentă este $I(1)$ (Capitolul 3: staționară după o diferențiere) și'),
      T('some linear combination $\\beta^\\top \\mathbf y_t$, with $\\beta \\neq 0$, is $I(0)$: stationary', 'o anumită combinație liniară $\\beta^\\top \\mathbf y_t$, cu $\\beta \\neq 0$, este $I(0)$: staționară')]),
    (T('$\\beta$ is the \\textbf{cointegrating vector}; $u_t = \\beta^\\top \\mathbf y_t - \\mu$ is the \\textbf{equilibrium error}', '$\\beta$ este \\textbf{vectorul de cointegrare}; $u_t = \\beta^\\top \\mathbf y_t - \\mu$ este \\textbf{eroarea de echilibru}'),
     [T('$\\beta$ is unique only up to scale: $2\\beta$ works as well; we \\textbf{normalise} one coefficient to 1, e.g.\\ $\\beta = (1, -b)^\\top$', '$\\beta$ este unic doar pînă la o constantă multiplicativă: și $2\\beta$ este bun; \\textbf{normalizăm} un coeficient la 1, de exemplu $\\beta = (1, -b)^\\top$'),
      T('with $n$ variables there can be up to $r = n - 1$ independent cointegrating vectors; $r$ is the \\textbf{cointegration rank}', 'cu $n$ variabile pot exista cel mult $r = n - 1$ vectori de cointegrare independenți; $r$ este \\textbf{rangul de cointegrare}')]),
    T('Cointegration is a property of the \\textbf{levels}: differencing the data destroys the information in $\\beta^\\top \\mathbf y_t$', 'Cointegrarea este o proprietate a \\textbf{nivelurilor}: diferențierea datelor distruge informația din $\\beta^\\top \\mathbf y_t$')))

D.frame(T('Common stochastic trends', 'Trenduri stochastice comune'), items(
    (T('Write $x_t = x_{t-1} + \\varepsilon_t$ (a random walk) and $y_t = b\\,x_t + u_t$, with $u_t$ stationary', 'Fie $x_t = x_{t-1} + \\varepsilon_t$ (un mers aleator) și $y_t = b\\,x_t + u_t$, cu $u_t$ staționar'),
     [T('both series contain the same stochastic trend $\\sum_{s \\le t} \\varepsilon_s$; $y_t - b x_t = u_t$ removes it', 'ambele serii conțin același trend stochastic $\\sum_{s \\le t} \\varepsilon_s$; $y_t - b x_t = u_t$ îl elimină')]),
    (T('\\textbf{Common trends} representation \\refSW: $n$ variables with $r$ cointegrating vectors are driven by $n - r$ common stochastic trends',
       'Reprezentarea prin \\textbf{trenduri comune} \\refSW: $n$ variabile cu $r$ vectori de cointegrare sînt antrenate de $n - r$ trenduri stochastice comune'),
     [T('$r = 0$: $n$ separate trends, no cointegration; $r = n - 1$: one trend shared by all', '$r = 0$: $n$ trenduri separate, fără cointegrare; $r = n - 1$: un singur trend, comun tuturor'),
      T('example: three yields (1, 5 and 10 years) with $r = 2$ share one trend, the \\textbf{level} of interest rates (Section 5)', 'exemplu: trei randamente (la 1, 5 și 10 ani) cu $r = 2$ au un singur trend comun, \\textbf{nivelul} ratelor dobînzii (secțiunea 5)')]),
    (T('\\textbf{Worked example}: $x_t$ a random walk, $y_t = 2x_t + u_t$, $z_t = -x_t + v_t$ ($u_t$, $v_t$ stationary)', '\\textbf{Exemplu rezolvat}: $x_t$ mers aleator, $y_t = 2x_t + u_t$, $z_t = -x_t + v_t$ ($u_t$, $v_t$ staționare)'),
     [T('$n = 3$, one common trend, so $r = 2$: e.g.\\ $\\beta_1 = (1, -2, 0)^\\top$ and $\\beta_2 = (0, 1, 2)^\\top$ (since $y_t + 2z_t = u_t + 2v_t$)',
        '$n = 3$, un singur trend comun, deci $r = 2$: de exemplu $\\beta_1 = (1, -2, 0)^\\top$ și $\\beta_2 = (0, 1, 2)^\\top$ (deoarece $y_t + 2z_t = u_t + 2v_t$)')])))

D.frame(T('Where economics predicts cointegration', 'Cointegrarea în teoria economică'), items(
    (T('\\textbf{Purchasing power parity}: $s_t = p_t - p^*_t + q_t$, with the real exchange rate $q_t$ stationary (Section 8)', '\\textbf{Paritatea puterii de cumpărare}: $s_t = p_t - p^*_t + q_t$, cu cursul real $q_t$ staționar (secțiunea 8)'),
     [T('$s_t$: log exchange rate (lei per euro); $p_t$, $p^*_t$: log price levels at home and abroad', '$s_t$: logaritmul cursului (lei pentru un euro); $p_t$, $p^*_t$: logaritmii nivelurilor prețurilor în țară și în străinătate')]),
    T('\\textbf{Term structure of interest rates}: by the expectations hypothesis, the spread between a long and a short yield is stationary \\refCS',
      '\\textbf{Structura la termen a ratelor dobînzii}: conform ipotezei așteptărilor, diferența (spread-ul) dintre un randament pe termen lung și unul pe termen scurt este staționară \\refCS'),
    T('\\textbf{Consumption and income}: the ``great ratio\'\' $c_t - y_t$ (logs) is stable if consumption follows permanent income', '\\textbf{Consum și venit}: „marele raport” $c_t - y_t$ (în logaritmi) este stabil dacă consumul urmează venitul permanent'),
    T('\\textbf{Spot and futures prices}, an index and a fund that tracks it: arbitrage keeps the gap small', '\\textbf{Prețul spot și prețul futures}, un indice și un fond care îl replică: arbitrajul menține diferența mică'),
    T('\\textbf{Shares of similar firms}: the same sector shocks; the basis of pairs trading (Section 9)', '\\textbf{Acțiunile unor firme similare}: aceleași șocuri sectoriale; baza pairs trading (secțiunea 9)')))

chart(T('Four pairs of trending series', 'Patru perechi de serii cu trend'), 'tsa_ch7_examples', 'TSA_ch7_common_trends', [
    T('US Treasury yields (FRED, monthly, since 1960); US real GDP and consumption (statsmodels macrodata, 1959--2009); Banca Transilvania and BRD (adjusted daily prices); EUR/RON and EUR/HUF (BNR reference rates, month-end)',
      'Randamentele titlurilor de stat americane (FRED, lunar, din 1960); PIB-ul și consumul real al SUA (macrodata din statsmodels, 1959--2009); Banca Transilvania și BRD (prețuri zilnice ajustate); EUR/RON și EUR/HUF (cursuri de referință BNR, la sfîrșitul lunii)')], h='0.58\\textheight')

interp(('the four pairs', 'celor patru perechi'), [
    (T('Each series is $I(1)$ (Chapter 3); the question is whether the two series of a pair move \\textbf{together} in the long run', 'Fiecare serie este $I(1)$ (Capitolul 3); întrebarea este dacă cele două serii ale unei perechi evoluează \\textbf{împreună} pe termen lung'),
     [T('Engle--Granger test (Section 2), 5\\% critical value about $@{ex.rates.cv}$', 'testul Engle--Granger (secțiunea 2), valoarea critică de 5\\% circa $@{ex.rates.cv}$')]),
    T('Yields: $\\tau = @{ex.rates.tau}$, p = @{ex.rates.p}: borderline; consumption and income: $\\tau = @{ex.cons.tau}$, p = @{ex.cons.p}', 'Randamentele: $\\tau = @{ex.rates.tau}$, p = @{ex.rates.p}: la limită; consumul și venitul: $\\tau = @{ex.cons.tau}$, p = @{ex.cons.p}'),
    T('The two banks: $\\tau = @{ex.banks.tau}$, p = @{ex.banks.p}: cointegrated in this sample', 'Cele două bănci: $\\tau = @{ex.banks.tau}$, p = @{ex.banks.p}: cointegrate în acest eșantion'),
    (T('EUR/RON and EUR/HUF: $\\tau = @{ex.fx.tau}$, p = @{ex.fx.p}: \\textbf{not} cointegrated', 'EUR/RON și EUR/HUF: $\\tau = @{ex.fx.tau}$, p = @{ex.fx.p}: \\textbf{nu} sînt cointegrate'),
     [T('although they rose by similar amounts and $R^2 = @{ex.fx.r2}$ in levels: a common direction is not a common trend', 'deși au crescut cu valori apropiate, iar $R^2 = @{ex.fx.r2}$ în niveluri: o direcție comună nu este un trend comun')])])

D.frame(T('Question for the room', 'Întrebare pentru sală'), items(
    (T('\\textbf{Question}: two $I(1)$ series have a correlation of 0.95 in levels. Are they cointegrated?', '\\textbf{Întrebare}: două serii $I(1)$ au în niveluri o corelație de 0,95. Sînt ele cointegrate?'),
     [T('\\textbf{Answer}: we cannot tell: unrelated random walks often have high correlations (Chapter 3); only a test on the equilibrium error decides',
        '\\textbf{Răspuns}: nu putem spune: mersurile aleatoare fără legătură au adesea corelații mari (Capitolul 3); doar un test pe eroarea de echilibru decide')]),
    (T('\\textbf{Question}: two cointegrated series have a correlation of 0.2 between their daily changes. Is this a contradiction?', '\\textbf{Întrebare}: două serii cointegrate au o corelație de 0,2 între variațiile lor zilnice. Este o contradicție?'),
     [T('\\textbf{Answer}: no: cointegration is about the long run; in the short run the changes can be almost unrelated', '\\textbf{Răspuns}: nu: cointegrarea privește termenul lung; pe termen scurt variațiile pot fi aproape necorelate')])))

D.recap(('The idea of cointegration', 'ideea de cointegrare'), [
    T('$I(1)$ series are cointegrated if a combination $\\beta^\\top \\mathbf y_t$ is stationary: a long-run equilibrium', 'Seriile $I(1)$ sînt cointegrate dacă o combinație $\\beta^\\top \\mathbf y_t$ este staționară: un echilibru pe termen lung'),
    T('Equivalently, they share common stochastic trends: $n$ variables, $r$ vectors, $n - r$ trends', 'Echivalent, au trenduri stochastice comune: $n$ variabile, $r$ vectori, $n - r$ trenduri'),
    T('Theory suggests cointegration (parity conditions, spreads, great ratios), but the data must confirm it', 'Teoria sugerează cointegrarea (condiții de paritate, spread-uri, rapoarte mari), dar datele trebuie să o confirme')])

# =============================================================================
# 2. ENGLE-GRANGER
# =============================================================================
D.section('The Engle--Granger two-step method', 'Metoda Engle--Granger în doi pași')

D.frame(T('Two steps', 'Doi pași'), items(
    (T('\\textbf{Step 1}: estimate the long-run relation by OLS (ordinary least squares) in levels: $y_t = a + b\\,x_t + u_t$', '\\textbf{Pasul 1}: estimăm relația pe termen lung prin OLS (ordinary least squares, metoda celor mai mici pătrate) în niveluri: $y_t = a + b\\,x_t + u_t$'),
     [T('keep the residuals $\\hat u_t = y_t - \\hat a - \\hat b x_t$', 'păstrăm reziduurile $\\hat u_t = y_t - \\hat a - \\hat b x_t$'),
      T('if the series are cointegrated, $\\hat b$ is \\textbf{super-consistent}: its error shrinks like $1/T$, faster than the usual $1/\\sqrt{T}$', 'dacă seriile sînt cointegrate, $\\hat b$ este \\textbf{superconsistent}: eroarea lui scade ca $1/T$, mai repede decît de obicei ($1/\\sqrt{T}$)')]),
    (T('\\textbf{Step 2}: an ADF test (Chapter 3) on the residuals, without a constant: $\\Delta\\hat u_t = \\gamma\\,\\hat u_{t-1} + \\sum_{j=1}^{k}\\delta_j\\Delta\\hat u_{t-j} + e_t$',
       '\\textbf{Pasul 2}: un test ADF (Capitolul 3) pe reziduuri, fără constantă: $\\Delta\\hat u_t = \\gamma\\,\\hat u_{t-1} + \\sum_{j=1}^{k}\\delta_j\\Delta\\hat u_{t-j} + e_t$'),
     [T('$H_0$: $\\gamma = 0$, the residuals have a unit root: \\textbf{no cointegration}; $H_1$: $\\gamma < 0$: cointegration', '$H_0$: $\\gamma = 0$, reziduurile au o rădăcină unitară: \\textbf{fără cointegrare}; $H_1$: $\\gamma < 0$: cointegrare'),
      T('statistic $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$; reject $H_0$ when $\\tau$ is below the critical value', 'statistica $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$; respingem $H_0$ cînd $\\tau$ este sub valoarea critică')]),
    T('Pretest: each series must be $I(1)$; cointegration between an $I(0)$ and an $I(1)$ series is impossible', 'Testare prealabilă: fiecare serie trebuie să fie $I(1)$; o serie $I(0)$ și una $I(1)$ nu pot fi cointegrate')))

D.frame(T('Why the Dickey--Fuller table is wrong here', 'Motivul pentru care tabelul Dickey--Fuller este greșit aici'), items(
    (T('OLS chooses $\\hat b$ to make the residual variance as small as possible', 'OLS alege $\\hat b$ astfel încît varianța reziduurilor să fie cît mai mică'),
     [T('so $\\hat u_t$ looks more stationary than the true $u_t$, even when there is no cointegration', 'deci $\\hat u_t$ pare mai staționar decît adevăratul $u_t$, chiar și atunci cînd nu există cointegrare'),
      T('under $H_0$ the statistic $\\tau$ is shifted to the left of the Dickey--Fuller distribution', 'în ipoteza $H_0$, statistica $\\tau$ este deplasată la stînga distribuției Dickey--Fuller')]),
    (T('The critical values depend on the number of variables $n$ and on the deterministic terms of step 1', 'Valorile critice depind de numărul de variabile $n$ și de termenii determiniști din pasul 1'),
     [T('tables: \\refEG\\ (by simulation), then \\refMacKinnon\\ and \\refMacKinnonb, used by \\texttt{statsmodels} (\\texttt{coint})', 'tabele: \\refEG\\ (prin simulare), apoi \\refMacKinnon\\ și \\refMacKinnonb, folosite de \\texttt{statsmodels} (\\texttt{coint})')]),
    T('With $\\beta$ \\textbf{known} from theory (e.g.\\ a spread, $\\beta = (1, -1)^\\top$), nothing is estimated: the plain ADF test and its table apply',
      'Cu $\\beta$ \\textbf{cunoscut} din teorie (de exemplu un spread, $\\beta = (1, -1)^\\top$), nu se estimează nimic: se aplică testul ADF obișnuit, cu tabelul lui')))

chart(T('The null distribution by simulation', 'Distribuția în ipoteza nulă prin simulare'), 'tsa_ch7_eg_dist', 'TSA_ch7_engle_granger', [
    T('@{ed.R} samples of $T = @{ed.T}$ independent Gaussian random walks; step 1 with a constant, step 2 without lags', '@{ed.R} de eșantioane de $T = @{ed.T}$ mersuri aleatoare gaussiene independente; pasul 1 cu constantă, pasul 2 fără decalaje')], h='0.55\\textheight')

interp(('the simulation', 'simulării'), [
    (T('5\\% quantiles: $@{ed.1.q05}$ (one series, Dickey--Fuller), $@{ed.2.q05}$ (2 variables), $@{ed.3.q05}$ (3 variables)', 'Cuantilele de 5\\%: $@{ed.1.q05}$ (o serie, Dickey--Fuller), $@{ed.2.q05}$ (2 variabile), $@{ed.3.q05}$ (3 variabile)'),
     [T('MacKinnon\'s values for $T = @{ed.T}$: $@{ed.1.mk5}$; $@{ed.2.mk5}$; $@{ed.3.mk5}$: the simulation reproduces them', 'valorile MacKinnon pentru $T = @{ed.T}$: $@{ed.1.mk5}$; $@{ed.2.mk5}$; $@{ed.3.mk5}$: simularea le reproduce')]),
    (T('Using the Dickey--Fuller value $@{ed.1.mk5}$ for the residuals would reject a true $H_0$ in @{ed.2.size}\\% of samples with 2 variables and @{ed.3.size}\\% with 3', 'Folosind valoarea Dickey--Fuller $@{ed.1.mk5}$ pentru reziduuri, am respinge o ipoteză $H_0$ adevărată în @{ed.2.size}\\% din eșantioane cu 2 variabile și în @{ed.3.size}\\% cu 3'),
     [T('instead of 5\\%: we would ``find\'\' cointegration far too often', 'în loc de 5\\%: am „găsi” cointegrare mult prea des')]),
    T('More variables give OLS more freedom to fit, hence more negative critical values', 'Mai multe variabile îi dau metodei OLS mai multă libertate de ajustare, de unde valori critice mai negative')])

D.frame(T('Critical values of the Engle--Granger test', 'Valorile critice ale testului Engle--Granger'), table(
    'lcccccc', T('\\textbf{Variables $n$}', '\\textbf{Variabile $n$}') + ' & \\multicolumn{3}{c}{' + T('\\textbf{constant}', '\\textbf{constantă}') + '} & \\multicolumn{3}{c}{' + T('\\textbf{constant and trend}', '\\textbf{constantă și trend}') + '} \\\\\n & 1\\% & 5\\% & 10\\% & 1\\% & 5\\% & 10\\%',
    [(T('1 (Dickey--Fuller)', '1 (Dickey--Fuller)') if k == '1' else k) + ' & ' + ' & '.join(f'$@{{cv.c.{k}.{l}}}$' for l in ['1', '5', '10']) + ' & ' + ' & '.join(f'$@{{cv.ct.{k}.{l}}}$' for l in ['1', '5', '10'])
     for k in ['1', '2', '3', '4']], size='footnotesize') + items(
    T('Asymptotic values ($T \\to \\infty$) of \\refMacKinnonb; for finite $T$ the values are slightly more negative', 'Valori asimptotice ($T \\to \\infty$) din \\refMacKinnonb; pentru $T$ finit, valorile sînt puțin mai negative'),
    T('Read the row of the \\textbf{total} number of variables in the regression, $y_t$ included', 'Citiți rîndul cu numărul \\textbf{total} de variabile din regresie, inclusiv $y_t$'),
    T('Constant and trend in step 1: when the series have drifts that the relation does not cancel', 'Constantă și trend în pasul 1: cînd seriile au derive pe care relația nu le anulează')))

D.frame(T('The Phillips--Ouliaris test', 'Testul Phillips--Ouliaris'), items(
    (T('\\refPO: the same step 1 and the same $H_0$ (no cointegration)', '\\refPO: același pas 1 și aceeași ipoteză $H_0$ (fără cointegrare)'),
     [T('step 2: a Dickey--Fuller regression \\textbf{without} lagged differences; the $t$-statistic is corrected with the long-run variance of the residuals (as in the Phillips--Perron test of Chapter 3)',
        'pasul 2: o regresie Dickey--Fuller \\textbf{fără} diferențe decalate; statistica $t$ este corectată cu varianța pe termen lung a reziduurilor (ca în testul Phillips--Perron din Capitolul 3)'),
      T('two versions: $Z_t$ (corrected $t$-ratio) and $Z_\\alpha$ (corrected $T\\hat\\gamma$); the critical values of $Z_t$ are those of Engle--Granger', 'două versiuni: $Z_t$ (raportul $t$ corectat) și $Z_\\alpha$ ($T\\hat\\gamma$ corectat); valorile critice ale lui $Z_t$ sînt cele Engle--Granger')]),
    (T('Why two tests?', 'De ce două teste?'),
     [T('Engle--Granger depends on the number of lags $k$ chosen by AIC; Phillips--Ouliaris depends on the bandwidth of the long-run variance', 'Engle--Granger depinde de numărul de decalaje $k$ ales după AIC; Phillips--Ouliaris depinde de lățimea de bandă a varianței pe termen lung'),
      T('when they agree, the verdict is robust; when they disagree, look at the residual plot and at the sample', 'cînd concordă, verdictul este robust; cînd nu concordă, examinăm graficul reziduurilor și eșantionul')]),
    T('In Python: \\texttt{coint(y, x)} from \\texttt{statsmodels}; \\texttt{phillips\\_ouliaris(y, x, test\\_type="Zt")} from \\texttt{arch}', 'În Python: \\texttt{coint(y, x)} din \\texttt{statsmodels}; \\texttt{phillips\\_ouliaris(y, x, test\\_type="Zt")} din \\texttt{arch}')))

D.frame(T('Worked example: US consumption and income', 'Exemplu rezolvat: consumul și venitul în SUA'), items(
    (T('Data: $c_t$, $y_t$ = 100 $\\times$ log of real consumption and real GDP, quarterly, 1959Q1--2009Q3 (@{eg0.n} quarters)', 'Datele: $c_t$, $y_t$ = 100 $\\times$ logaritmul consumului real și al PIB-ului real, trimestrial, T1 1959--T3 2009 (@{eg0.n} de trimestre)'),
     [T('pretest: ADF with trend, p = @{ex.cc.p} for $c_t$ and @{ex.cy.p} for $y_t$: both $I(1)$', 'testare prealabilă: ADF cu trend, p = @{ex.cc.p} pentru $c_t$ și @{ex.cy.p} pentru $y_t$: ambele $I(1)$')]),
    (T('Step 1: $\\hat c_t = @{eg.cons.a} + @{eg.cons.b}\\,y_t$, $R^2 = @{eg.cons.r2}$, DW $= @{eg.cons.dw}$', 'Pasul 1: $\\hat c_t = @{eg.cons.a} + @{eg.cons.b}\\,y_t$, $R^2 = @{eg.cons.r2}$, DW $= @{eg.cons.dw}$'),
     [T('the slope is close to 1: a stable consumption share; but $R^2$ and DW alone prove nothing', 'panta este apropiată de 1: o pondere stabilă a consumului; dar $R^2$ și DW singure nu dovedesc nimic')]),
    (T('Step 2: $\\tau = @{eg0.tau}$ with @{eg0.k} lags; 5\\% critical value for $n = 2$, $T = @{eg0.n}$: $@{eg.cons.cv}$', 'Pasul 2: $\\tau = @{eg0.tau}$ cu @{eg0.k} decalaje; valoarea critică de 5\\% pentru $n = 2$, $T = @{eg0.n}$: $@{eg.cons.cv}$'),
     [T('$@{eg0.tau} < @{eg.cons.cv}$: reject ``no cointegration\'\' at 5\\% (p = @{eg0.p}); Phillips--Ouliaris: $Z_t = @{eg0.po}$, p = @{eg0.pop}', '$@{eg0.tau} < @{eg.cons.cv}$: respingem „fără cointegrare” la 5\\% (p = @{eg0.p}); Phillips--Ouliaris: $Z_t = @{eg0.po}$, p = @{eg0.pop}'),
      T('with the Dickey--Fuller value ($@{eg.cons.adfcv}$) the decision would be the same here, but for the wrong reason', 'cu valoarea Dickey--Fuller ($@{eg.cons.adfcv}$) decizia ar fi aceeași aici, dar din motive greșite')])))

chart(T('The step-1 residuals', 'Reziduurile din pasul 1'), 'tsa_ch7_eg_steps', 'TSA_ch7_engle_granger', [
    T('Left and centre: $\\hat u_t$ of consumption on income and of the 10-year on the 3-month Treasury yield; right: their sample ACF and that of a simulated random walk',
      'Stînga și centru: $\\hat u_t$ pentru consumul pe venit și pentru randamentul la 10 ani pe cel la 3 luni; dreapta: ACF de selecție a lor și a unui mers aleator simulat')], h='0.54\\textheight')

interp(('the residuals', 'reziduurilor'), [
    (T('Consumption: $\\hat\\rho(1) = @{es.c1}$ but $\\hat\\rho(8) = @{es.c8}$: deviations of 1--3\\% that shrink markedly within about two years', 'Consumul: $\\hat\\rho(1) = @{es.c1}$, dar $\\hat\\rho(8) = @{es.c8}$: abateri de 1--3\\% care se reduc mult în aproximativ doi ani'),
     []),
    (T('Yields: $\\hat\\rho(1) = @{es.r1}$, $\\hat\\rho(12) = @{es.r12}$: very persistent deviations; a borderline test ($\\tau = @{ex.rates.tau}$, p = @{ex.rates.p})', 'Randamentele: $\\hat\\rho(1) = @{es.r1}$, $\\hat\\rho(12) = @{es.r12}$: abateri foarte persistente; un test la limită ($\\tau = @{ex.rates.tau}$, p = @{ex.rates.p})'),
     [T('high persistence is the usual situation: the power of the test against $\\rho$ close to 1 is low (Chapter 3)', 'persistența ridicată este situația obișnuită: puterea testului împotriva unui $\\rho$ apropiat de 1 este mică (Capitolul 3)')]),
    T('A random walk decays much more slowly: the ACF alone cannot settle the question; the test can', 'Un mers aleator descrește mult mai lent: ACF singură nu poate tranșa întrebarea; testul poate')])

D.frame(T('Engle--Granger and Phillips--Ouliaris on real data', 'Engle--Granger și Phillips--Ouliaris pe date reale'), table(
    'llrrrrrr', T('\\textbf{Regression}', '\\textbf{Regresia}') + ' & ' + T('\\textbf{Freq.}', '\\textbf{Frecv.}') + ' & $T$ & $\\hat b$ & $\\tau$ (k) & p & $Z_t$ & p',
    EGR, size='scriptsize') + items(
    T('$\\tau$: Engle--Granger statistic ($k$ lags by AIC); p-values of \\refMacKinnonb\\ for 2 variables; $Z_t$: Phillips--Ouliaris', '$\\tau$: statistica Engle--Granger ($k$ decalaje după AIC); valori p din \\refMacKinnonb\\ pentru 2 variabile; $Z_t$: Phillips--Ouliaris'),
    T('Banks: adjusted prices from data/market; Romania: household consumption and GDP, Eurostat, chain-linked volumes, seasonally adjusted, since 1995',
      'Băncile: prețuri ajustate din data/market; România: consumul gospodăriilor și PIB-ul, Eurostat, volume înlănțuite, ajustate sezonier, din 1995')), 'small')

interp(('the table', 'tabelului'), [
    (T('\\textbf{Normalisation}: TLV on BRD and BRD on TLV give different $\\tau$ ($@{eg2.tau}$ and $@{eg3.tau}$) and slopes that are not exact inverses', '\\textbf{Normalizarea}: TLV pe BRD și BRD pe TLV dau valori $\\tau$ diferite ($@{eg2.tau}$ și $@{eg3.tau}$) și pante care nu sînt exact inverse una alteia'),
     [T('Engle--Granger depends on which variable is on the left; here the verdict is the same', 'Engle--Granger depinde de variabila aleasă în stînga; aici verdictul este același')]),
    (T('\\textbf{The sample}: since 2010 the same banks are not cointegrated (p = @{eg4.p}); since 2014 they are (p = @{eg2.p})', '\\textbf{Eșantionul}: din 2010 aceleași bănci nu sînt cointegrate (p = @{eg4.p}); din 2014 sînt (p = @{eg2.p})'),
     [T('choosing the start date after looking at the data is a form of data snooping', 'alegerea datei de început după examinarea datelor este o formă de data snooping (căutare în date pînă la găsirea unui rezultat)')]),
    (T('\\textbf{The two tests disagree} for Romania: Engle--Granger p = @{eg6.p} with @{eg.ro.k} lags, Phillips--Ouliaris p @{eg6.pop}', '\\textbf{Cele două teste nu concordă} pentru România: Engle--Granger p = @{eg6.p} cu @{eg.ro.k} decalaje, Phillips--Ouliaris p @{eg6.pop}'),
     [T('the slope @{eg6.b} is far from 1: the consumption share rose strongly; Seminar 7 (B2) discusses this case', 'panta @{eg6.b} este departe de 1: ponderea consumului a crescut puternic; Seminarul 7 (B2) discută acest caz')])])

D.frame(T('Limits of the Engle--Granger method', 'Limitele metodei Engle--Granger'), items(
    (T('\\textbf{At most one} cointegrating vector: with $n \\ge 3$ variables there may be several (Sections 4--5)', '\\textbf{Cel mult un} vector de cointegrare: cu $n \\ge 3$ variabile pot exista mai mulți (secțiunile 4--5)'), []),
    (T('\\textbf{Normalisation}: the results depend on the variable put on the left', '\\textbf{Normalizarea}: rezultatele depind de variabila pusă în stînga'), []),
    (T('\\textbf{Inference on $\\hat b$}: super-consistent, but biased in small samples, and its OLS $t$-statistics are \\textbf{not} valid', '\\textbf{Inferența asupra lui $\\hat b$}: superconsistent, dar deplasat în eșantioane mici, iar statisticile $t$ din OLS \\textbf{nu} sînt valide'),
     [T('remedy: dynamic OLS (DOLS) \\refSWd, adding leads and lags of $\\Delta x_t$, gives valid standard errors', 'remediu: OLS dinamic (DOLS) \\refSWd, cu valori anticipate și decalate ale lui $\\Delta x_t$, dă erori standard valide')]),
    (T('\\textbf{Two-step errors}: mistakes in step 1 carry over to step 2; the power is low against slow adjustment', '\\textbf{Erori în doi pași}: greșelile din pasul 1 se transmit în pasul 2; puterea este mică împotriva unei ajustări lente'),
     [T('the Johansen method (Section 5) estimates everything at once, by maximum likelihood', 'metoda Johansen (secțiunea 5) estimează totul deodată, prin verosimilitate maximă')])))

D.frame(T('Case study: Engle and Granger (1987)', 'Studiu de caz: Engle și Granger (1987)'), cols(
    ph('engle', T('Robert Engle (2022); Nobel Prize in Economics 2003', 'Robert Engle (2022); Premiul Nobel pentru economie 2003'), h='0.44\\textheight'),
    items((T('\\textbf{Question}: how to model $I(1)$ series that are tied in the long run, without spurious regressions and without differencing away the long run?',
             '\\textbf{Întrebarea}: cum modelăm serii $I(1)$ legate pe termen lung, fără regresii false și fără a elimina prin diferențiere informația pe termen lung?'),
           [T('\\refEG, written at the University of California, San Diego, one of the most cited papers in econometrics', '\\refEG, scrisă la Universitatea California din San Diego, una dintre cele mai citate lucrări din econometrie')]),
          (T('\\textbf{Contribution}', '\\textbf{Contribuția}'),
           [T('the definition of cointegration and the representation theorem (Section 4): cointegration $\\Leftrightarrow$ error correction', 'definiția cointegrării și teorema de reprezentare (secțiunea 4): cointegrare $\\Leftrightarrow$ corecția erorii'),
            T('the two-step estimator and residual-based tests with simulated critical values', 'estimatorul în doi pași și testele pe reziduuri, cu valori critice obținute prin simulare'),
            T('applications to US series such as consumption and income, prices and wages, short and long interest rates', 'aplicații pe serii din SUA, precum consumul și venitul, prețurile și salariile, ratele dobînzii pe termen scurt și lung')]),
          T('\\textbf{Recognition}: Nobel Prize 2003, Granger for cointegration, Engle for ARCH (Chapter 5)', '\\textbf{Recunoaștere}: Premiul Nobel 2003, Granger pentru cointegrare, Engle pentru ARCH (Capitolul 5)')),
    wl='0.30', wr='0.66'))

D.recap(('The Engle--Granger method', 'metoda Engle--Granger'), [
    T('Step 1: OLS in levels (super-consistent $\\hat b$); step 2: ADF on the residuals, $H_0$: no cointegration', 'Pasul 1: OLS în niveluri ($\\hat b$ superconsistent); pasul 2: ADF pe reziduuri, $H_0$: fără cointegrare'),
    T('Critical values of Engle--Granger (MacKinnon), more negative than Dickey--Fuller and dependent on $n$', 'Valorile critice Engle--Granger (MacKinnon), mai negative decît cele Dickey--Fuller și dependente de $n$'),
    T('Check with Phillips--Ouliaris, with the reverse normalisation and with another sample', 'Verificăm cu Phillips--Ouliaris, cu normalizarea inversă și cu un alt eșantion')])

# =============================================================================
# 3. MODELE CU CORECȚIA ERORII
# =============================================================================
D.section('Error correction models', 'Modele cu corecția erorii')

D.frame(T('The error correction model', 'Modelul cu corecția erorii'), items(
    (T('With $u_{t-1} = y_{t-1} - a - b x_{t-1}$ (last period\'s equilibrium error), the \\textbf{error correction model} (ECM) is', 'Cu $u_{t-1} = y_{t-1} - a - b x_{t-1}$ (eroarea de echilibru din perioada anterioară), \\textbf{modelul cu corecția erorii} (ECM) este'),
     [T('$\\Delta y_t = c + \\gamma\\,u_{t-1} + \\delta_0\\,\\Delta x_t + \\sum_{j \\ge 1}(\\phi_j\\Delta y_{t-j} + \\delta_j\\Delta x_{t-j}) + \\varepsilon_t$', '$\\Delta y_t = c + \\gamma\\,u_{t-1} + \\delta_0\\,\\Delta x_t + \\sum_{j \\ge 1}(\\phi_j\\Delta y_{t-j} + \\delta_j\\Delta x_{t-j}) + \\varepsilon_t$')]),
    (T('Every term is stationary: OLS and the usual $t$-tests are valid (with $\\hat u_{t-1}$ from step 1)', 'Toți termenii sînt staționari: OLS și testele $t$ obișnuite sînt valide (cu $\\hat u_{t-1}$ din pasul 1)'),
     [T('$b$: the \\textbf{long-run} effect of $x$ on $y$; $\\delta_0$: the \\textbf{short-run} (same-period) effect', '$b$: efectul \\textbf{pe termen lung} al lui $x$ asupra lui $y$; $\\delta_0$: efectul \\textbf{pe termen scurt} (în aceeași perioadă)'),
      T('$\\gamma$: the \\textbf{speed of adjustment}; error correction requires $-2 < \\gamma < 0$', '$\\gamma$: \\textbf{viteza de ajustare}; corecția erorii cere $-2 < \\gamma < 0$')]),
    T('Reading: if $y_{t-1}$ was above equilibrium ($u_{t-1} > 0$) and $\\gamma < 0$, then $y_t$ tends to fall: a share $|\\gamma|$ of the gap is closed each period',
      'Interpretare: dacă $y_{t-1}$ a fost peste echilibru ($u_{t-1} > 0$) și $\\gamma < 0$, atunci $y_t$ tinde să scadă: o fracțiune $|\\gamma|$ din distanță se închide în fiecare perioadă')))

D.frame(T('Speed of adjustment and half-life', 'Viteza de ajustare și timpul de înjumătățire'), items(
    (T('Without new shocks, the equilibrium error evolves as $u_t = (1 + \\gamma)\\,u_{t-1}$, so $u_h = (1 + \\gamma)^h u_0$', 'Fără șocuri noi, eroarea de echilibru evoluează după $u_t = (1 + \\gamma)\\,u_{t-1}$, deci $u_h = (1 + \\gamma)^h u_0$'),
     [T('(in a bivariate system where only $y$ adjusts)', '(într-un sistem cu două variabile în care se ajustează doar $y$)')]),
    (T('\\textbf{Half-life}: the number of periods after which half of a deviation is gone', '\\textbf{Timpul de înjumătățire}: numărul de perioade după care a dispărut jumătate dintr-o abatere'),
     [T('$(1 + \\gamma)^{h} = 0.5 \\;\\Rightarrow\\; h_{1/2} = \\ln 0.5 / \\ln(1 + \\gamma)$', '$(1 + \\gamma)^{h} = 0{,}5 \\;\\Rightarrow\\; h_{1/2} = \\ln 0{,}5 / \\ln(1 + \\gamma)$')]),
    (T('\\textbf{Worked example}: $\\gamma = -0.2$, $h_{1/2} = \\ln 0.5/\\ln 0.8 = @{wx.hl}$ periods', '\\textbf{Exemplu rezolvat}: $\\gamma = -0{,}2$, $h_{1/2} = \\ln 0{,}5/\\ln 0{,}8 = @{wx.hl}$ perioade'),
     [T('in months: about three months for half of a deviation; in quarters: three quarters', 'în luni: circa trei luni pentru jumătate dintr-o abatere; în trimestre: trei trimestre')]),
    T('Compare half-lives only in the same time unit: a monthly and a quarterly $\\gamma$ are not comparable', 'Comparăm timpii de înjumătățire doar în aceeași unitate de timp: un $\\gamma$ lunar și unul trimestrial nu sînt comparabile')))

chart(T('Half-lives: theory and estimates', 'Timpi de înjumătățire: teorie și estimări'), 'tsa_ch7_ecm', 'TSA_ch7_error_correction', [
    T('Left: $(1 + \\gamma)^h$ for three speeds; right: the decay implied by the estimated ECMs of US consumption (quarters) and of the 10-year yield (months)',
      'Stînga: $(1 + \\gamma)^h$ pentru trei viteze; dreapta: descreșterea implicată de modelele ECM estimate pentru consumul din SUA (trimestre) și pentru randamentul la 10 ani (luni)')], h='0.55\\textheight')

interp(('the estimated ECMs', 'modelelor ECM estimate'), [
    (T('Consumption: $\\hat\\gamma = @{ec.cons.g}$ ($t = @{ec.cons.t}$), half-life @{ec.cons.h} quarters; $\\hat\\delta_0 = @{ec.cons.d0}$', 'Consumul: $\\hat\\gamma = @{ec.cons.g}$ ($t = @{ec.cons.t}$), timp de înjumătățire @{ec.cons.h} trimestre; $\\hat\\delta_0 = @{ec.cons.d0}$'),
     [T('half of a rise in income reaches consumption in the same quarter; the rest comes slowly, and the adjustment term is only marginally significant', 'jumătate dintr-o creștere a venitului ajunge în consum în același trimestru; restul vine lent, iar termenul de ajustare este doar marginal semnificativ')]),
    (T('10-year yield: $\\hat\\gamma = @{ec.long.g}$ ($t = @{ec.long.t}$), half-life @{ec.long.h} months; $\\hat\\delta_0 = @{ec.long.d0}$', 'Randamentul la 10 ani: $\\hat\\gamma = @{ec.long.g}$ ($t = @{ec.long.t}$), timp de înjumătățire @{ec.long.h} luni; $\\hat\\delta_0 = @{ec.long.d0}$'),
     [T('the 3-month rate also reacts to the same gap: $\\hat\\gamma = @{ec.short.g}$ ($t = @{ec.short.t}$), with the opposite sign: both rates move towards each other', 'și rata la 3 luni reacționează la aceeași distanță: $\\hat\\gamma = @{ec.short.g}$ ($t = @{ec.short.t}$), cu semn opus: ambele rate se apropie una de alta')]),
    T('When both variables adjust, a single-equation ECM tells only half of the story: hence the VECM (Section 4)', 'Cînd ambele variabile se ajustează, un ECM cu o singură ecuație spune doar jumătate din poveste: de aici VECM (secțiunea 4)')])

D.frame(T('Worked example: one quarter of consumption', 'Exemplu rezolvat: un trimestru de consum'), items(
    (T('Estimated ECM (main terms): $\\Delta c_t = @{ec.cons.c} + @{ec.cons.d0}\\,\\Delta y_t @{ec.cons.g}\\,\\hat u_{t-1} + \\dots$, with $\\hat u_{t-1} = c_{t-1} - @{ec.cons.b}\\,y_{t-1} - \\hat a$',
       'ECM estimat (termenii principali): $\\Delta c_t = @{ec.cons.c} + @{ec.cons.d0}\\,\\Delta y_t @{ec.cons.g}\\,\\hat u_{t-1} + \\dots$, cu $\\hat u_{t-1} = c_{t-1} - @{ec.cons.b}\\,y_{t-1} - \\hat a$'),
     []),
    (T('Scenario: consumption is $@{wx.gap}\\%$ above its equilibrium ($\\hat u_{t-1} = @{wx.gap}$) and income grows by $@{wx.dy}\\%$ this quarter', 'Scenariu: consumul este cu $@{wx.gap}\\%$ peste echilibru ($\\hat u_{t-1} = @{wx.gap}$), iar venitul crește cu $@{wx.dy}\\%$ în acest trimestru'),
     [T('short-run effect: $@{ec.cons.d0} \\times @{wx.dy} = @{wx.short}$; error correction: $@{ec.cons.g} \\times @{wx.gap} = @{wx.corr}$', 'efectul pe termen scurt: $@{ec.cons.d0} \\times @{wx.dy} = @{wx.short}$; corecția erorii: $@{ec.cons.g} \\times @{wx.gap} = @{wx.corr}$'),
      T('predicted growth of consumption: about $@{wx.dc}\\%$ (the lagged differences are set to 0)', 'creșterea prognozată a consumului: circa $@{wx.dc}\\%$ (diferențele decalate sînt egale cu 0)')]),
    T('Without the error correction term, the model would ignore that consumption is already too high', 'Fără termenul de corecție a erorii, modelul ar ignora faptul că deja consumul este prea mare')))

D.frame(T('Case study: the consumption function of DHSY (1978)', 'Studiu de caz: funcția de consum DHSY (1978)'), items(
    (T('\\textbf{Question}: why did the consumption functions of the 1970s, estimated in levels or in differences, forecast UK consumers\' expenditure so badly?',
       '\\textbf{Întrebarea}: de ce prognozau atît de prost cheltuielile de consum din Marea Britanie funcțiile de consum din anii 1970, estimate în niveluri sau în diferențe?'),
     [T('\\refDHSY\\ (Davidson, Hendry, Srba and Yeo, ``DHSY\'\'): quarterly UK data on expenditure and income', '\\refDHSY\\ (Davidson, Hendry, Srba și Yeo, „DHSY”): date trimestriale pentru Marea Britanie, cheltuieli și venit')]),
    (T('\\textbf{Method}: a model in differences plus the lagged log ratio of expenditure to income, the error correction term', '\\textbf{Metoda}: un model în diferențe plus raportul logaritmic decalat dintre cheltuieli și venit, termenul de corecție a erorii'),
     [T('a general-to-specific search, starting from a large dynamic model', 'o căutare de la general la particular, pornind de la un model dinamic amplu')]),
    (T('\\textbf{Legacy}: the ECM form came first; \\refGranger\\ and \\refEG\\ then showed why it works: the two series are cointegrated', '\\textbf{Moștenirea}: forma ECM a apărut prima; \\refGranger\\ și \\refEG\\ au arătat apoi de ce funcționează: cele două serii sînt cointegrate'),
     [T('a test of cointegration can also use the $t$-statistic of $\\gamma$ in the ECM, with its own critical values \\refBDM', 'un test de cointegrare poate folosi și statistica $t$ a lui $\\gamma$ din ECM, cu valori critice proprii \\refBDM')])))

D.recap(('Error correction models', 'modele cu corecția erorii'), [
    T('ECM: $\\Delta y_t$ on $\\Delta x_t$, lagged differences and $\\hat u_{t-1}$; all terms stationary', 'ECM: $\\Delta y_t$ pe $\\Delta x_t$, diferențe decalate și $\\hat u_{t-1}$; toți termenii sînt staționari'),
    T('$\\gamma < 0$: speed of adjustment; half-life $\\ln 0.5/\\ln(1 + \\gamma)$, in the units of the data', '$\\gamma < 0$: viteza de ajustare; timpul de înjumătățire $\\ln 0{,}5/\\ln(1 + \\gamma)$, în unitățile datelor'),
    T('Short run ($\\delta_0$) and long run ($b$) in one equation; if several variables adjust, we need a system', 'Termenul scurt ($\\delta_0$) și termenul lung ($b$) într-o singură ecuație; dacă se ajustează mai multe variabile, avem nevoie de un sistem')])

# =============================================================================
# 4. VECM
# =============================================================================
D.section('The VECM and the Granger representation theorem', 'Modelul VECM și teorema de reprezentare a lui Granger')

D.frame(T('From a VAR to a VECM', 'De la VAR la VECM'), items(
    (T('Chapter 6: a VAR($p$) in levels, $\\mathbf y_t = \\mathbf c + A_1\\mathbf y_{t-1} + \\dots + A_p\\mathbf y_{t-p} + \\mathbf u_t$', 'Capitolul 6: un VAR($p$) în niveluri, $\\mathbf y_t = \\mathbf c + A_1\\mathbf y_{t-1} + \\dots + A_p\\mathbf y_{t-p} + \\mathbf u_t$'),
     [T('subtract $\\mathbf y_{t-1}$ and regroup the lags: the same model in \\textbf{vector error correction} form (VECM)', 'scădem $\\mathbf y_{t-1}$ și regrupăm decalajele: același model sub forma \\textbf{vectorială cu corecția erorii} (VECM)')]),
    T('$\\Delta\\mathbf y_t = \\mathbf c + \\Pi\\,\\mathbf y_{t-1} + \\sum_{i=1}^{p-1}\\Gamma_i\\,\\Delta\\mathbf y_{t-i} + \\mathbf u_t$, \\quad $\\Pi = A_1 + \\dots + A_p - I$, \\quad $\\Gamma_i = -(A_{i+1} + \\dots + A_p)$',
      '$\\Delta\\mathbf y_t = \\mathbf c + \\Pi\\,\\mathbf y_{t-1} + \\sum_{i=1}^{p-1}\\Gamma_i\\,\\Delta\\mathbf y_{t-i} + \\mathbf u_t$, \\quad $\\Pi = A_1 + \\dots + A_p - I$, \\quad $\\Gamma_i = -(A_{i+1} + \\dots + A_p)$'),
    (T('\\textbf{Derivation} for $p = 2$: $\\mathbf y_t - \\mathbf y_{t-1} = (A_1 - I)\\mathbf y_{t-1} + A_2\\mathbf y_{t-2} + \\mathbf u_t$; add and subtract $A_2\\mathbf y_{t-1}$:',
       '\\textbf{Deducere} pentru $p = 2$: $\\mathbf y_t - \\mathbf y_{t-1} = (A_1 - I)\\mathbf y_{t-1} + A_2\\mathbf y_{t-2} + \\mathbf u_t$; adunăm și scădem $A_2\\mathbf y_{t-1}$:'),
     [T('$\\Delta\\mathbf y_t = (A_1 + A_2 - I)\\,\\mathbf y_{t-1} - A_2\\,\\Delta\\mathbf y_{t-1} + \\mathbf u_t$: $\\Pi = A_1 + A_2 - I$, $\\Gamma_1 = -A_2$', '$\\Delta\\mathbf y_t = (A_1 + A_2 - I)\\,\\mathbf y_{t-1} - A_2\\,\\Delta\\mathbf y_{t-1} + \\mathbf u_t$: $\\Pi = A_1 + A_2 - I$, $\\Gamma_1 = -A_2$')]),
    T('The VECM is a rewriting, not a new model: all the information about the long run sits in the matrix $\\Pi$', 'VECM este o rescriere, nu un model nou: toată informația despre termenul lung se află în matricea $\\Pi$')))

D.frame(T('The rank of $\\Pi$', 'Rangul matricei $\\Pi$'), items(
    T('$\\Delta\\mathbf y_t$ is stationary; so $\\Pi\\mathbf y_{t-1}$ must be stationary too. Three cases, by $r = \\mathrm{rank}(\\Pi)$:', '$\\Delta\\mathbf y_t$ este staționar; deci și $\\Pi\\mathbf y_{t-1}$ trebuie să fie staționar. Trei cazuri, după $r = \\mathrm{rang}(\\Pi)$:'),
    (T('$r = 0$: $\\Pi = 0$; no cointegration; the right model is a \\textbf{VAR in differences}', '$r = 0$: $\\Pi = 0$; fără cointegrare; modelul potrivit este un \\textbf{VAR în diferențe}'), []),
    (T('$r = n$: $\\Pi$ has full rank; $\\mathbf y_t$ is already stationary; a \\textbf{VAR in levels} (Chapter 6)', '$r = n$: $\\Pi$ are rang maxim; $\\mathbf y_t$ este deja staționar; un \\textbf{VAR în niveluri} (Capitolul 6)'), []),
    (T('$0 < r < n$: \\textbf{cointegration} with $r$ vectors; $\\Pi = \\alpha\\beta^\\top$ with $\\alpha$, $\\beta$ of size $n \\times r$', '$0 < r < n$: \\textbf{cointegrare} cu $r$ vectori; $\\Pi = \\alpha\\beta^\\top$, cu $\\alpha$, $\\beta$ de dimensiune $n \\times r$'),
     [T('$\\beta^\\top\\mathbf y_{t-1}$: the $r$ equilibrium errors; $\\alpha$: the \\textbf{adjustment coefficients} (loadings), how strongly each variable reacts to each error',
        '$\\beta^\\top\\mathbf y_{t-1}$: cele $r$ erori de echilibru; $\\alpha$: \\textbf{coeficienții de ajustare} (loadings), cît de puternic reacționează fiecare variabilă la fiecare eroare'),
      T('only the product is unique: $\\alpha\\beta^\\top = (\\alpha Q)(\\beta Q^{-\\top})^\\top$ for any invertible $Q$; we normalise $\\beta$', 'doar produsul este unic: $\\alpha\\beta^\\top = (\\alpha Q)(\\beta Q^{-\\top})^\\top$ pentru orice $Q$ inversabilă; normalizăm $\\beta$')]),
    T('Testing for cointegration = testing the rank of $\\Pi$ (Section 5)', 'Testarea cointegrării = testarea rangului lui $\\Pi$ (secțiunea 5)')))

D.frame(T('The Granger representation theorem', 'Teorema de reprezentare a lui Granger'), items(
    (T('\\textbf{Theorem} \\refEG: if $\\mathbf y_t$ is $I(1)$ and cointegrated with rank $r$, then it has a VECM representation with $\\Pi = \\alpha\\beta^\\top$, and conversely',
       '\\textbf{Teorema} \\refEG: dacă $\\mathbf y_t$ este $I(1)$ și cointegrat cu rangul $r$, atunci are o reprezentare VECM cu $\\Pi = \\alpha\\beta^\\top$, și reciproc'),
     [T('cointegration $\\Leftrightarrow$ error correction: the equilibrium error must feed back into the changes', 'cointegrare $\\Leftrightarrow$ corecția erorii: eroarea de echilibru trebuie să influențeze variațiile')]),
    (T('\\textbf{Consequences}', '\\textbf{Consecințe}'),
     [T('a VAR in differences omits $\\alpha\\beta^\\top\\mathbf y_{t-1}$: it is \\textbf{misspecified} when the series are cointegrated', 'un VAR în diferențe omite $\\alpha\\beta^\\top\\mathbf y_{t-1}$: este \\textbf{greșit specificat} cînd seriile sînt cointegrate'),
      T('$\\alpha \\neq 0$: at least one variable adjusts, so there is Granger causality (Chapter 6) in at least one direction', '$\\alpha \\neq 0$: cel puțin o variabilă se ajustează, deci există cauzalitate Granger (Capitolul 6) în cel puțin o direcție'),
      T('the levels are driven by $n - r$ common stochastic trends; the effects of shocks on the levels do not die out', 'nivelurile sînt antrenate de $n - r$ trenduri stochastice comune; efectele șocurilor asupra nivelurilor nu se sting')]),
    T('A VAR in levels is not wrong either: it is consistent, but it ignores the restriction $\\mathrm{rank}(\\Pi) = r$ and its tests are non-standard', 'Un VAR în niveluri nu este nici el greșit: este consistent, dar ignoră restricția $\\mathrm{rang}(\\Pi) = r$, iar testele lui sînt nestandard')))

D.frame(T('2003: the Nobel Prize for cointegration', '2003: Premiul Nobel pentru cointegrare'), cols(
    ph('granger', T('Clive Granger (1934--2009)', 'Clive Granger (1934--2009)'), h='0.40\\textheight'),
    ph('nottingham', T('The Sir Clive Granger Building, University of Nottingham', 'Clădirea Sir Clive Granger, Universitatea din Nottingham'), h='0.30\\textheight') + '\n' +
    items(T('Granger studied and began his career at Nottingham; cointegration was developed at UC San Diego', 'Granger a studiat și și-a început cariera la Nottingham; cointegrarea a fost dezvoltată la UC San Diego'),
          T('Nobel lecture: \\refGrangerN: ``time series analysis, cointegration, and applications\'\'', 'Prelegerea Nobel: \\refGrangerN: „time series analysis, cointegration, and applications”')),
    wl='0.34', wr='0.62'))

D.frame(T('Worked example: a bivariate VECM', 'Exemplu rezolvat: un VECM cu două variabile'), items(
    (T('$\\Delta y_{1t} = -0.10\\,(y_{1,t-1} - y_{2,t-1}) + u_{1t}$, \\quad $\\Delta y_{2t} = 0.10\\,(y_{1,t-1} - y_{2,t-1}) + u_{2t}$', '$\\Delta y_{1t} = -0{,}10\\,(y_{1,t-1} - y_{2,t-1}) + u_{1t}$, \\quad $\\Delta y_{2t} = 0{,}10\\,(y_{1,t-1} - y_{2,t-1}) + u_{2t}$'),
     [T('$\\beta = (1, -1)^\\top$, $\\alpha = (-0.10, 0.10)^\\top$; $\\Pi = \\alpha\\beta^\\top = \\begin{pmatrix} -0.10 & 0.10 \\\\ 0.10 & -0.10 \\end{pmatrix}$, $\\det\\Pi = 0$, rank 1',
        '$\\beta = (1, -1)^\\top$, $\\alpha = (-0{,}10; 0{,}10)^\\top$; $\\Pi = \\alpha\\beta^\\top = \\begin{pmatrix} -0{,}10 & 0{,}10 \\\\ 0{,}10 & -0{,}10 \\end{pmatrix}$, $\\det\\Pi = 0$, rangul 1')]),
    (T('If $y_1$ is above $y_2$: $y_1$ falls ($\\alpha_1 < 0$) and $y_2$ rises ($\\alpha_2 > 0$): both close the gap', 'Dacă $y_1$ este peste $y_2$: $y_1$ scade ($\\alpha_1 < 0$), iar $y_2$ crește ($\\alpha_2 > 0$): ambele închid distanța'), []),
    (T('The gap $z_t = y_{1t} - y_{2t}$: $z_t = (1 + \\beta^\\top\\alpha)\\,z_{t-1} + (u_{1t} - u_{2t}) = 0.8\\,z_{t-1} + \\dots$', 'Distanța $z_t = y_{1t} - y_{2t}$: $z_t = (1 + \\beta^\\top\\alpha)\\,z_{t-1} + (u_{1t} - u_{2t}) = 0{,}8\\,z_{t-1} + \\dots$'),
     [T('a stationary AR(1); half-life $\\ln 0.5/\\ln 0.8 = @{wx.hl}$ periods; the condition is $-2 < \\beta^\\top\\alpha < 0$', 'un AR(1) staționar; timpul de înjumătățire $\\ln 0{,}5/\\ln 0{,}8 = @{wx.hl}$ perioade; condiția este $-2 < \\beta^\\top\\alpha < 0$')]),
    T('The average $(y_{1t} + y_{2t})/2$ changes by $(u_{1t} + u_{2t})/2$: a random walk, the common trend', 'Media $(y_{1t} + y_{2t})/2$ variază cu $(u_{1t} + u_{2t})/2$: un mers aleator, trendul comun')))

TB = '>{\\raggedright\\arraybackslash}'
D.frame(T('Deterministic terms: five cases', 'Termenii determiniști: cinci cazuri'), table(
    TB + 'p{0.6cm}' + TB + 'p{4.6cm}' + TB + 'p{4.4cm}' + TB + 'p{2.6cm}',
    T('\\textbf{Case}', '\\textbf{Cazul}') + ' & ' + T('\\textbf{In the VECM}', '\\textbf{În VECM}') + ' & ' + T('\\textbf{Implication for the data}', '\\textbf{Implicația pentru date}') + ' & \\texttt{statsmodels}',
    ['1 & ' + T('no constant, no trend', 'fără constantă, fără trend') + ' & ' + T('no drift; relations through the origin', 'fără derivă; relații prin origine') + ' & \\texttt{n}',
     '2 & ' + T('constant only inside $\\beta^\\top\\mathbf y_{t-1}$', 'constantă doar în $\\beta^\\top\\mathbf y_{t-1}$') + ' & ' + T('no drift; equilibrium errors with a non-zero mean (yields, spreads)', 'fără derivă; erori de echilibru cu medie nenulă (randamente, spread-uri)') + ' & \\texttt{ci}',
     '3 & ' + T('unrestricted constant', 'constantă nerestricționată') + ' & ' + T('linear trends in the levels, none in the relations (GDP, prices)', 'trenduri liniare în niveluri, nu și în relații (PIB, prețuri)') + ' & \\texttt{co}',
     '4 & ' + T('constant, and a trend inside $\\beta^\\top\\mathbf y_{t-1}$', 'constantă și trend în $\\beta^\\top\\mathbf y_{t-1}$') + ' & ' + T('trends in the levels and in the relations', 'trenduri în niveluri și în relații') + ' & \\texttt{coli}',
     '5 & ' + T('unrestricted constant and trend', 'constantă și trend nerestricționate') + ' & ' + T('quadratic trends in the levels; rarely plausible', 'trenduri pătratice în niveluri; rareori plauzibil') + ' & \\texttt{colo}'],
    size='scriptsize') + items(
    T('The critical values of the Johansen tests change with the case \\refJohBook; choose the case from the plot and from theory, before testing',
      'Valorile critice ale testelor Johansen se schimbă de la un caz la altul \\refJohBook; alegem cazul după grafic și după teorie, înainte de testare'),
    T('\\texttt{coint\\_johansen} of \\texttt{statsmodels} implements case 1 (\\texttt{det\\_order=-1}) and case 3 (\\texttt{det\\_order=0}), the case used in this chapter',
      '\\texttt{coint\\_johansen} din \\texttt{statsmodels} implementează cazul 1 (\\texttt{det\\_order=-1}) și cazul 3 (\\texttt{det\\_order=0}), cazul folosit în acest capitol')), 'small')

D.frame(T('Weak exogeneity', 'Exogenitatea slabă'), items(
    (T('If a row of $\\alpha$ is zero, $\\alpha_i = 0$, the variable $y_i$ does \\textbf{not} react to the equilibrium errors', 'Dacă un rînd din $\\alpha$ este nul, $\\alpha_i = 0$, variabila $y_i$ \\textbf{nu} reacționează la erorile de echilibru'),
     [T('$y_i$ is \\textbf{weakly exogenous} for $\\beta$: the other variables do all the adjusting', '$y_i$ este \\textbf{slab exogenă} pentru $\\beta$: celelalte variabile fac toată ajustarea'),
      T('$y_i$ pushes the common trend; e.g.\\ Euribor for Romanian interest rates (Seminar 7, B4)', '$y_i$ antrenează trendul comun; de exemplu Euribor pentru ratele dobînzii din România (Seminarul 7, B4)')]),
    (T('Test: a $t$-test on $\\alpha_i$ (one cointegrating vector) or a likelihood-ratio (LR) test of $\\alpha_i = 0$ \\refJohB', 'Testul: un test $t$ pentru $\\alpha_i$ (un vector de cointegrare) sau un test al raportului de verosimilitate (LR) pentru $\\alpha_i = 0$ \\refJohB'),
     [T('the statistic is asymptotically $\\chi^2$ with $r$ degrees of freedom, once $r$ is fixed', 'statistica are asimptotic distribuția $\\chi^2$ cu $r$ grade de libertate, după ce $r$ a fost fixat')]),
    T('Use: if $x_t$ is weakly exogenous, a single-equation ECM for $y_t$ (Section 3) loses no information about $\\beta$', 'Utilitate: dacă $x_t$ este slab exogenă, un ECM cu o singură ecuație pentru $y_t$ (secțiunea 3) nu pierde informație despre $\\beta$')))

D.recap(('The VECM', 'modelul VECM'), [
    T('$\\Delta\\mathbf y_t = \\alpha\\beta^\\top\\mathbf y_{t-1} + \\sum\\Gamma_i\\Delta\\mathbf y_{t-i} + \\mathbf u_t$: a VAR rewritten; $\\mathrm{rank}(\\Pi) = r$', '$\\Delta\\mathbf y_t = \\alpha\\beta^\\top\\mathbf y_{t-1} + \\sum\\Gamma_i\\Delta\\mathbf y_{t-i} + \\mathbf u_t$: un VAR rescris; $\\mathrm{rang}(\\Pi) = r$'),
    T('Granger representation: cointegration $\\Leftrightarrow$ error correction; a VAR in differences omits the long run', 'Reprezentarea Granger: cointegrare $\\Leftrightarrow$ corecția erorii; un VAR în diferențe omite termenul lung'),
    T('$\\beta$: equilibria; $\\alpha$: who adjusts; deterministic terms: five cases; $\\alpha_i = 0$: weak exogeneity', '$\\beta$: echilibre; $\\alpha$: cine se ajustează; termenii determiniști: cinci cazuri; $\\alpha_i = 0$: exogenitate slabă')])

# =============================================================================
# 5. JOHANSEN
# =============================================================================
D.section('The Johansen tests', 'Testele Johansen')

D.frame(T('The idea: reduced-rank regression', 'Ideea: regresia cu rang redus'), cols(
    ph('copenhagen', T('University of Copenhagen, where Søren Johansen and Katarina Juselius developed the method', 'Universitatea din Copenhaga, unde Søren Johansen și Katarina Juselius au dezvoltat metoda'), h='0.36\\textheight'),
    items((T('\\refJohA, \\refJohB: maximum likelihood for the VECM with Gaussian errors', '\\refJohA, \\refJohB: verosimilitate maximă pentru VECM, cu erori gaussiene'),
           [T('regress $\\Delta\\mathbf y_t$ and $\\mathbf y_{t-1}$ on the lagged differences; keep the residuals $R_{0t}$ and $R_{1t}$', 'regresăm $\\Delta\\mathbf y_t$ și $\\mathbf y_{t-1}$ pe diferențele decalate; păstrăm reziduurile $R_{0t}$ și $R_{1t}$'),
            T('find the combinations of $R_{1t}$ most correlated with $R_{0t}$: the \\textbf{canonical correlations}', 'căutăm combinațiile lui $R_{1t}$ cel mai puternic corelate cu $R_{0t}$: \\textbf{corelațiile canonice}')]),
          (T('Their squares are the eigenvalues $1 > \\hat\\lambda_1 \\ge \\dots \\ge \\hat\\lambda_n \\ge 0$', 'Pătratele lor sînt valorile proprii $1 > \\hat\\lambda_1 \\ge \\dots \\ge \\hat\\lambda_n \\ge 0$'),
           [T('a large $\\hat\\lambda_i$: a stationary combination that predicts the changes; $\\hat\\lambda_i \\approx 0$: a combination that is still $I(1)$', 'un $\\hat\\lambda_i$ mare: o combinație staționară care prezice variațiile; $\\hat\\lambda_i \\approx 0$: o combinație care rămîne $I(1)$'),
            T('the eigenvectors give $\\hat\\beta$; then $\\hat\\alpha$ follows by OLS', 'vectorii proprii dau $\\hat\\beta$; apoi $\\hat\\alpha$ rezultă prin OLS')])),
    wl='0.36', wr='0.60'))

D.frame(T('Trace and maximum-eigenvalue statistics', 'Statisticile urmei și a valorii proprii maxime'), items(
    (T('\\textbf{Trace test}: $H_0$: rank $\\le r$, against rank $= n$; \\quad $\\lambda_{\\mathrm{trace}}(r) = -T\\sum_{i=r+1}^{n}\\ln(1 - \\hat\\lambda_i)$',
       '\\textbf{Testul urmei}: $H_0$: rang $\\le r$, față de rang $= n$; \\quad $\\lambda_{\\mathrm{trace}}(r) = -T\\sum_{i=r+1}^{n}\\ln(1 - \\hat\\lambda_i)$'), []),
    (T('\\textbf{Maximum-eigenvalue test}: $H_0$: rank $= r$, against rank $= r + 1$; \\quad $\\lambda_{\\max}(r) = -T\\ln(1 - \\hat\\lambda_{r+1})$',
       '\\textbf{Testul valorii proprii maxime}: $H_0$: rang $= r$, față de rang $= r + 1$; \\quad $\\lambda_{\\max}(r) = -T\\ln(1 - \\hat\\lambda_{r+1})$'), []),
    (T('\\textbf{Sequential procedure}: test $r = 0$, then $r \\le 1$, \\dots; the rank is the first $r$ whose $H_0$ is \\textbf{not} rejected', '\\textbf{Procedura secvențială}: testăm $r = 0$, apoi $r \\le 1$ etc.; rangul este primul $r$ pentru care $H_0$ \\textbf{nu} este respinsă'),
     [T('the statistics are likelihood ratios, but their distributions are non-standard (functions of Brownian motions)', 'statisticile sînt rapoarte de verosimilitate, dar distribuțiile lor sînt nestandard (funcții de mișcări browniene)'),
      T('critical values depend on $n - r$ and on the deterministic case: \\refOL, \\refMHM', 'valorile critice depind de $n - r$ și de cazul termenilor determiniști: \\refOL, \\refMHM')]),
    T('The two tests usually agree; if not, the trace test is preferred in most applied work', 'Cele două teste concordă de obicei; dacă nu, în majoritatea lucrărilor aplicate se preferă testul urmei')))

D.frame(T('Worked example: from eigenvalues to statistics', 'Exemplu rezolvat: de la valori proprii la statistici'), items(
    (T('US yields at 1, 5 and 10 years, monthly, $T = @{j0.n}$: $\\hat\\lambda = (@{j0.e0}, @{j0.e1}, @{j0.e2})$', 'Randamentele din SUA la 1, 5 și 10 ani, lunar, $T = @{j0.n}$: $\\hat\\lambda = (@{j0.e0}, @{j0.e1}, @{j0.e2})$'),
     [T('$-\\ln(1 - \\hat\\lambda_i)$: @{wj.l0}; @{wj.l1}; @{wj.l2}', '$-\\ln(1 - \\hat\\lambda_i)$: @{wj.l0}; @{wj.l1}; @{wj.l2}')]),
    (T('$r = 0$: trace $= @{j0.n} \\times (@{wj.l0} + @{wj.l1} + @{wj.l2}) = @{wj.tr0} > @{jcv.t0}$: reject', '$r = 0$: urma $= @{j0.n} \\times (@{wj.l0} + @{wj.l1} + @{wj.l2}) = @{wj.tr0} > @{jcv.t0}$: respingem'),
     [T('max-eigenvalue: $@{j0.n} \\times @{wj.l0} = @{wj.mx0} > @{jcv.m0}$: reject', 'valoarea proprie maximă: $@{j0.n} \\times @{wj.l0} = @{wj.mx0} > @{jcv.m0}$: respingem')]),
    (T('$r \\le 1$: trace $= @{wj.tr1} > @{jcv.t1}$: reject; \\quad $r \\le 2$: trace $= @{j0.t2} < @{jcv.t2}$: do not reject', '$r \\le 1$: urma $= @{wj.tr1} > @{jcv.t1}$: respingem; \\quad $r \\le 2$: urma $= @{j0.t2} < @{jcv.t2}$: nu respingem'),
     [T('rank 2: two cointegrating vectors, one common trend', 'rangul 2: doi vectori de cointegrare, un singur trend comun')]),
    T('Small eigenvalues still matter: they are multiplied by $T$', 'Și valorile proprii mici contează: sînt înmulțite cu $T$')))

D.frame(T('Johansen tests on three systems', 'Testele Johansen pe trei sisteme'), table(
    'lrrcccccc', T('\\textbf{System}', '\\textbf{Sistemul}') + ' & $T$ & $k$ & ' + ' & '.join(['$\\lambda_{\\mathrm{trace}}(0)$', '$\\lambda_{\\mathrm{trace}}(1)$', '$\\lambda_{\\mathrm{trace}}(2)$', '$\\lambda_{\\max}(0)$', '$\\lambda_{\\max}(1)$']) + ' & ' + T('\\textbf{rank}', '\\textbf{rang}'),
    [f'{lab} & @{{j{i}.n}} & @{{j{i}.k}} & @{{j{i}.t0}} & @{{j{i}.t1}} & @{{j{i}.t2}} & @{{j{i}.m0}} & @{{j{i}.m1}} & @{{j{i}.rt}}' for lab, i in JROWS]
    + [T('5\\% critical value', 'Valoarea critică de 5\\%') + ' & & & @{jcv.t0} & @{jcv.t1} & @{jcv.t2} & @{jcv.m0} & @{jcv.m1} &'],
    size='scriptsize') + items(
    T('Case 3 (unrestricted constant); $k$: lagged differences chosen by BIC on the VAR in levels; ranks by the trace test (the maximum-eigenvalue test gives the same ranks here)',
      'Cazul 3 (constantă nerestricționată); $k$: numărul de diferențe decalate, ales după BIC pentru VAR-ul în niveluri; rangurile după testul urmei (testul valorii proprii maxime dă aici aceleași ranguri)'),
    T('Yields: FRED, 1960--2026; output, consumption and investment: 100 ln, US 1959--2009; currencies: BNR, month-end, 2005--2026',
      'Randamentele: FRED, 1960--2026; PIB, consum și investiții: 100 ln, SUA 1959--2009; monedele: BNR, sfîrșitul lunii, 2005--2026')), 'small')

interp(('the three systems', 'celor trei sisteme'), [
    (T('Yields: rank @{j0.rt}, so one common trend: the \\textbf{level} of rates moves all maturities; the two spreads are stationary', 'Randamentele: rangul @{j0.rt}, deci un singur trend comun: \\textbf{nivelul} ratelor mișcă toate scadențele; cele două spread-uri sînt staționare'),
     [T('the expectations hypothesis of the term structure \\refCS\\ in its weak form', 'ipoteza așteptărilor pentru structura la termen \\refCS, în forma ei slabă')]),
    (T('Output, consumption and investment: trace @{j1.t0} against @{jcv.t0}: rank @{j1.rt} at 5\\%, a near miss', 'PIB, consum și investiții: urma @{j1.t0} față de @{jcv.t0}: rangul @{j1.rt} la 5\\%, foarte aproape de respingere'),
     [T('balanced growth predicts rank 2 (stable shares of consumption and investment); with Engle--Granger, consumption and income were cointegrated (p = @{eg0.p}): low power, not proof of absence',
        'creșterea echilibrată prezice rangul 2 (ponderi stabile ale consumului și investițiilor); cu Engle--Granger, consumul și venitul erau cointegrate (p = @{eg0.p}): putere mică, nu dovada absenței')]),
    (T('Currencies: trace @{j2.t0}: rank @{j2.rt}; three separate stochastic trends (Section 8)', 'Monedele: urma @{j2.t0}: rangul @{j2.rt}; trei trenduri stochastice separate (secțiunea 8)'), [])])

D.frame(T('The deterministic case changes the answer', 'Cazul termenilor determiniști schimbă răspunsul'), table(
    'lcccccc', T('\\textbf{Deterministic terms}', '\\textbf{Termeni determiniști}') + ' & $\\lambda_{\\mathrm{trace}}(0)$ & 5\\% & $\\lambda_{\\mathrm{trace}}(1)$ & 5\\% & $\\lambda_{\\mathrm{trace}}(2)$ & 5\\%',
    [lab + ' & ' + ' & '.join(f'@{{yd.{d}.t{r}}} & @{{yd.{d}.c{r}}}' for r in range(3)) for lab, d in
     [(T('none (\\texttt{det\\_order=-1})', 'niciunul (\\texttt{det\\_order=-1})'), '-1'), (T('constant (\\texttt{det\\_order=0})', 'constantă (\\texttt{det\\_order=0})'), '0'),
      (T('linear trend (\\texttt{det\\_order=1})', 'trend liniar (\\texttt{det\\_order=1})'), '1')]],
    size='scriptsize') + items(
    T('US yields at 1, 5 and 10 years: ranks @{yd.-1.r}, @{yd.0.r} and @{yd.1.r}', 'Randamentele din SUA la 1, 5 și 10 ani: rangurile @{yd.-1.r}, @{yd.0.r} și @{yd.1.r}'),
    (T('With a linear trend, the test says that every yield is stationary around a trend: not plausible for interest rates over 66 years', 'Cu trend liniar, testul spune că fiecare randament este staționar în jurul unui trend: neplauzibil pentru ratele dobînzii pe 66 de ani'),
     [T('choose the deterministic terms from theory and from the plot, not from the result you like', 'alegem termenii determiniști după teorie și după grafic, nu după rezultatul care ne convine')]),
    T('Other practical issues: the lag length $k$; size distortions in small samples (correction factor $(T - nk)/T$, \\refRA); structural breaks', 'Alte probleme practice: numărul de decalaje $k$; distorsiunile nivelului de semnificație în eșantioane mici (factorul de corecție $(T - nk)/T$, \\refRA); rupturile structurale')), 'small')

D.frame(T('Case study: Johansen and Juselius (1990)', 'Studiu de caz: Johansen și Juselius (1990)'), items(
    (T('\\textbf{Question}: is there a stable demand for money in Denmark and Finland?', '\\textbf{Întrebarea}: există o cerere de bani stabilă în Danemarca și Finlanda?'),
     [T('\\refJJ: real money, real income, interest rates (and inflation); quarterly data', '\\refJJ: masa monetară reală, venitul real, ratele dobînzii (și inflația); date trimestriale')]),
    (T('\\textbf{Method}: the VECM by maximum likelihood; the rank by the trace and maximum-eigenvalue tests; then \\textbf{tests of restrictions} on $\\beta$ and $\\alpha$',
       '\\textbf{Metoda}: VECM prin verosimilitate maximă; rangul prin testele urmei și ale valorii proprii maxime; apoi \\textbf{teste ale restricțiilor} asupra lui $\\beta$ și $\\alpha$'),
     [T('e.g.\\ a unit income elasticity, $\\beta = (1, -1, \\cdot)$; weak exogeneity of income and interest rates', 'de exemplu o elasticitate unitară față de venit, $\\beta = (1, -1, \\cdot)$; exogenitatea slabă a venitului și a ratelor dobînzii')]),
    (T('\\textbf{Legacy}: the template of the ``cointegrated VAR\'\' methodology \\refJus: test the rank, then test economic hypotheses as restrictions', '\\textbf{Moștenirea}: modelul metodologiei „VAR cointegrat” \\refJus: testăm rangul, apoi testăm ipotezele economice ca restricții'),
     [T('one of the most cited papers in applied econometrics', 'una dintre cele mai citate lucrări din econometria aplicată')])))

D.recap(('The Johansen tests', 'testele Johansen'), [
    T('Eigenvalues of a reduced-rank regression; trace and maximum-eigenvalue statistics; sequential testing from $r = 0$', 'Valorile proprii ale unei regresii cu rang redus; statisticile urmei și a valorii proprii maxime; testare secvențială de la $r = 0$'),
    T('Critical values depend on $n - r$ and on the deterministic case', 'Valorile critice depind de $n - r$ și de cazul termenilor determiniști'),
    T('Yields: rank 2; macro aggregates: borderline; currencies: rank 0', 'Randamentele: rangul 2; agregatele macroeconomice: la limită; monedele: rangul 0')])

# =============================================================================
# 6. BETA, ALPHA
# =============================================================================
D.section('Estimating and interpreting $\\beta$ and $\\alpha$', 'Estimarea și interpretarea lui $\\beta$ și $\\alpha$')

chart(T('The term-structure VECM', 'Modelul VECM al structurii la termen'), 'tsa_ch7_rates', 'TSA_ch7_johansen_vecm', [
    T('US Treasury yields at 1, 5 and 10 years (FRED GS1, GS5, GS10), @{ve.first}--@{ve.last}; VECM with rank 2, @{ve.k} lagged differences, constant restricted to the relations (case 2)',
      'Randamentele titlurilor de stat americane la 1, 5 și 10 ani (FRED GS1, GS5, GS10), @{ve.first}--@{ve.last}; VECM cu rangul 2, @{ve.k} diferențe decalate, constantă restricționată la relații (cazul 2)')], h='0.55\\textheight')

interp(('the cointegrating vectors', 'vectorilor de cointegrare'), [
    (T('Normalisation on the 1-year and 5-year yields: $\\hat\\beta_1^\\top\\mathbf y_t = i^{(1)}_t - @{ve.b1}\\,i^{(10)}_t + @{ve.c1}$, $\\hat\\beta_2^\\top\\mathbf y_t = i^{(5)}_t - @{ve.b2}\\,i^{(10)}_t + @{ve.c2}$',
       'Normalizare pe randamentele la 1 și 5 ani: $\\hat\\beta_1^\\top\\mathbf y_t = i^{(1)}_t - @{ve.b1}\\,i^{(10)}_t + @{ve.c1}$, $\\hat\\beta_2^\\top\\mathbf y_t = i^{(5)}_t - @{ve.b2}\\,i^{(10)}_t + @{ve.c2}$'),
     [T('both slopes close to 1: the equilibria are almost the spreads $i^{(1)} - i^{(10)}$ and $i^{(5)} - i^{(10)}$, with mean term premia', 'ambele pante sînt apropiate de 1: echilibrele sînt aproape spread-urile $i^{(1)} - i^{(10)}$ și $i^{(5)} - i^{(10)}$, cu prime de termen medii')]),
    T('The equilibrium errors are stationary: ADF p @{ve.p0} and @{ve.p1}; they are large in the 1980s and around zero-rate periods', 'Erorile de echilibru sînt staționare: ADF p @{ve.p0} și @{ve.p1}; sînt mari în anii 1980 și în perioadele cu dobînzi aproape de zero'),
    T('A formal test of $\\beta = (1, -1)$ for each spread is a likelihood-ratio test \\refJohB; with super-consistent $\\hat\\beta$, values this close to 1 are informative', 'Un test formal pentru $\\beta = (1, -1)$ la fiecare spread este un test al raportului de verosimilitate \\refJohB; cu $\\hat\\beta$ superconsistent, valori atît de apropiate de 1 sînt informative')])

D.frame(T('The adjustment coefficients', 'Coeficienții de ajustare'), table(
    'lcccc', T('\\textbf{Equation}', '\\textbf{Ecuația}') + ' & $\\hat\\alpha_{\\cdot 1}$ & $t$ & $\\hat\\alpha_{\\cdot 2}$ & $t$',
    [T('$\\Delta i^{(1)}_t$ (1 year)', '$\\Delta i^{(1)}_t$ (1 an)') + ' & $@{ve.a00}$ & $@{ve.t00}$ & $@{ve.a01}$ & $@{ve.t01}$',
     T('$\\Delta i^{(5)}_t$ (5 years)', '$\\Delta i^{(5)}_t$ (5 ani)') + ' & $@{ve.a10}$ & $@{ve.t10}$ & $@{ve.a11}$ & $@{ve.t11}$',
     T('$\\Delta i^{(10)}_t$ (10 years)', '$\\Delta i^{(10)}_t$ (10 ani)') + ' & $@{ve.a20}$ & $@{ve.t20}$ & $@{ve.a21}$ & $@{ve.t21}$'],
    size='footnotesize') + items(
    (T('The 1-year yield does not react to either error ($|t| < 1$): it behaves as \\textbf{weakly exogenous}', 'Randamentul la 1 an nu reacționează la niciuna dintre erori ($|t| < 1$): se comportă ca \\textbf{slab exogen}'),
     [T('the short end follows monetary policy; the 5- and 10-year yields do the adjusting ($|t| \\approx 2$)', 'capătul scurt urmează politica monetară; randamentele la 5 și 10 ani fac ajustarea ($|t| \\approx 2$)')]),
    (T('The equilibrium errors follow $\\mathbf z_t = (I + \\hat\\beta^\\top\\hat\\alpha)\\,\\mathbf z_{t-1} + \\dots$; eigenvalues @{ve.eig0} and @{ve.eig1}', 'Erorile de echilibru urmează $\\mathbf z_t = (I + \\hat\\beta^\\top\\hat\\alpha)\\,\\mathbf z_{t-1} + \\dots$; valorile proprii @{ve.eig0} și @{ve.eig1}'),
     [T('half-lives of about @{ve.h0} and @{ve.h1} months: slow error correction, typical of interest rates', 'timpi de înjumătățire de circa @{ve.h0} și @{ve.h1} luni: corecție lentă a erorii, tipică pentru ratele dobînzii')])), 'small')

chart(T('Impulse responses of the VECM', 'Funcțiile de răspuns la impuls ale modelului VECM'), 'tsa_ch7_vecm_irf', 'TSA_ch7_johansen_vecm', [
    T('Orthogonalised responses (Cholesky, ordering 1y, 5y, 10y; Chapter 6) to a one-standard-deviation shock to the 1-year yield', 'Răspunsuri ortogonalizate (Cholesky, ordinea 1 an, 5 ani, 10 ani; Capitolul 6) la un șoc de o abatere standard asupra randamentului la 1 an')], h='0.50\\textheight')

interp(('the impulse responses', 'răspunsurilor la impuls'), [
    (T('On impact: +@{ir.i0}, +@{ir.i1} and +@{ir.i2} percentage points for 1, 5 and 10 years', 'La impact: +@{ir.i0}, +@{ir.i1} și +@{ir.i2} puncte procentuale pentru 1, 5 și 10 ani'),
     [T('after @{ir.H} months: +@{ir.h0}, +@{ir.h1} and +@{ir.h2}: the effects do \\textbf{not} die out and become almost equal', 'după @{ir.H} de luni: +@{ir.h0}, +@{ir.h1} și +@{ir.h2}: efectele \\textbf{nu} se sting și devin aproape egale')]),
    T('A permanent shift of the common trend (the level), with stationary spreads: the signature of $n - r = 1$ common trend', 'O deplasare permanentă a trendului comun (nivelul), cu spread-uri staționare: semnătura unui singur trend comun, $n - r = 1$'),
    T('In a stationary VAR (Chapter 6) every response returns to zero; in a VECM the long-run responses are generally different from zero', 'Într-un VAR staționar (Capitolul 6) orice răspuns revine la zero; într-un VECM, răspunsurile pe termen lung sînt în general diferite de zero')])

D.recap(('$\\beta$ and $\\alpha$', '$\\beta$ și $\\alpha$'), [
    T('$\\hat\\beta$: normalise, read as equilibria (spreads, ratios), test economic restrictions', '$\\hat\\beta$: normalizăm, interpretăm ca echilibre (spread-uri, rapoarte), testăm restricțiile economice'),
    T('$\\hat\\alpha$: who adjusts and how fast; a zero row means weak exogeneity', '$\\hat\\alpha$: cine se ajustează și cît de repede; un rînd nul înseamnă exogenitate slabă'),
    T('Shocks have permanent effects on cointegrated levels, but not on the equilibrium errors', 'Șocurile au efecte permanente asupra nivelurilor cointegrate, dar nu asupra erorilor de echilibru')])

# =============================================================================
# 7. PROGNOZA
# =============================================================================
D.section('Forecasting: VECM or VAR in differences?', 'Prognoza: VECM sau VAR în diferențe?')

D.frame(T('What theory says', 'Argumentele teoretice'), items(
    (T('VECM forecasts keep the equilibria: as the horizon grows, $\\hat\\beta^\\top\\hat{\\mathbf y}_{T+h}$ returns to its mean', 'Prognozele VECM păstrează echilibrele: pe măsură ce orizontul crește, $\\hat\\beta^\\top\\hat{\\mathbf y}_{T+h}$ revine la media sa'),
     [T('a VAR in differences forgets the levels: its forecasts of the spreads stay where they are', 'un VAR în diferențe uită nivelurile: prognozele lui pentru spread-uri rămîn unde se află')]),
    (T('\\refEY: imposing cointegration improves long-horizon forecasts in simulated systems', '\\refEY: impunerea cointegrării îmbunătățește prognozele pe orizonturi lungi în sisteme simulate'), []),
    (T('\\refCD: the gains are concentrated in the \\textbf{cointegrating combinations}', '\\refCD: cîștigurile sînt concentrate în \\textbf{combinațiile de cointegrare}'),
     [T('for the individual levels, the forecast error is dominated by the common trend, which no model can predict', 'pentru nivelurile individuale, eroarea de prognoză este dominată de trendul comun, pe care niciun model nu îl poate prezice'),
      T('estimation error in $\\alpha$, $\\beta$ and in the constants can make a VECM worse in practice', 'eroarea de estimare din $\\alpha$, $\\beta$ și din constante poate face un VECM mai slab în practică')]),
    T('Test it out of sample, with re-estimation at each forecast origin', 'Verificăm în afara eșantionului, cu reestimare la fiecare origine a prognozei')))

D.frame(T('The forecasting experiment', 'Experimentul de prognoză'), items(
    (T('Data: US yields at 1, 5 and 10 years, monthly, since 1960', 'Datele: randamentele din SUA la 1, 5 și 10 ani, lunar, din 1960'), []),
    (T('@{fc.n} forecast origins, every three months from @{fc.o0} to @{fc.o1}; models re-estimated on the data up to each origin', '@{fc.n} de origini ale prognozei, la fiecare trei luni, din @{fc.o0} pînă în @{fc.o1}; modelele sînt reestimate pe datele de pînă la fiecare origine'),
     [T('VECM with rank 2 and a restricted constant; VAR in differences without constant, with the same lags; random walk (no change)', 'VECM cu rangul 2 și constantă restricționată; VAR în diferențe fără constantă, cu aceleași decalaje; mersul aleator (fără schimbare)')]),
    (T('Horizons of 1 to 36 months; RMSE (root mean squared error) of each model divided by that of the random walk', 'Orizonturi de la 1 la 36 de luni; RMSE (root mean squared error, rădăcina erorii pătratice medii) a fiecărui model împărțită la cea a mersului aleator'),
     [T('ratio below 1: better than the random walk; also for the 10-year minus 1-year spread', 'raport sub 1: mai bun decît mersul aleator; și pentru spread-ul dintre randamentul la 10 ani și cel la 1 an')])))

chart(T('VECM against a VAR in differences, out of sample', 'VECM față de un VAR în diferențe, în afara eșantionului'), 'tsa_ch7_forecast', 'TSA_ch7_forecasting', [
    T('RMSE relative to the random walk by horizon; first three panels: the yields; last panel: the 10-year minus 1-year spread', 'RMSE relativ la mersul aleator, pe orizonturi; primele trei panouri: randamentele; ultimul panou: spread-ul 10 ani minus 1 an')], h='0.52\\textheight')

interp(('the forecast comparison', 'comparației prognozelor'), [
    (T('Levels: neither model beats the random walk clearly; the VECM is worse at short horizons (1 year at 6 months: @{fc.v6.0}; 5 years: @{fc.v6.1})', 'Nivelurile: niciun model nu bate clar mersul aleator; VECM este mai slab pe orizonturi scurte (1 an la 6 luni: @{fc.v6.0}; 5 ani: @{fc.v6.1})'),
     [T('the VAR in differences stays close to 1 at every horizon (1 year at 12 months: @{fc.d12.0})', 'VAR-ul în diferențe rămîne aproape de 1 la orice orizont (1 an la 12 luni: @{fc.d12.0})')]),
    (T('Spread: the VECM gains at long horizons: @{fc.sv24} at 24 months and @{fc.sv36} at 36 months; the VAR in differences: @{fc.sd36}', 'Spread-ul: VECM cîștigă pe orizonturi lungi: @{fc.sv24} la 24 de luni și @{fc.sv36} la 36 de luni; VAR-ul în diferențe: @{fc.sd36}'),
     [T('exactly the pattern of \\refCD: cointegration helps forecast the equilibrium errors, not the trend', 'exact tiparul din \\refCD: cointegrarea ajută la prognoza erorilor de echilibru, nu a trendului')]),
    T('Lesson: use a VECM when the object of interest is the relation (a spread, a ratio, a pair); for the levels alone, a random walk is hard to beat', 'Lecția: folosim un VECM cînd ne interesează relația (un spread, un raport, o pereche); pentru nivelurile singure, mersul aleator este greu de depășit')])

D.recap(('Forecasting', 'prognoza'), [
    T('The VECM pulls the forecasts of the equilibrium errors back to their means; the VAR in differences does not', 'VECM readuce prognozele erorilor de echilibru la mediile lor; VAR-ul în diferențe nu face acest lucru'),
    T('Out of sample: gains for the spread at long horizons, none for the levels', 'În afara eșantionului: cîștiguri pentru spread pe orizonturi lungi, niciunul pentru niveluri'),
    T('Always compare with the random walk, with re-estimation and without look-ahead', 'Comparăm întotdeauna cu mersul aleator, cu reestimare și fără a folosi date din viitor')])

# =============================================================================
# 8. APLICAȚII ECONOMICE
# =============================================================================
D.section('Economic examples: parity and interest rates', 'Exemple economice: paritate și rate ale dobînzii')

D.frame(T('Purchasing power parity', 'Paritatea puterii de cumpărare'), cols(
    ph('bnr', T('The National Bank of Romania, which publishes the reference rates', 'Banca Națională a României, care publică cursurile de referință'), h='0.34\\textheight'),
    items((T('\\textbf{Relative PPP} (purchasing power parity): the exchange rate offsets inflation differences', '\\textbf{Paritatea relativă a puterii de cumpărare} (PPP, purchasing power parity): cursul compensează diferențele de inflație'),
           [T('$s_t = 100\\ln(\\mathrm{EUR/RON})$; $p_t$, $p^*_t$: $100\\ln$ of the HICP (Harmonised Index of Consumer Prices) of Romania and of the euro area', '$s_t = 100\\ln(\\mathrm{EUR/RON})$; $p_t$, $p^*_t$: $100\\ln$ din IAPC (indicele armonizat al prețurilor de consum) al României și al zonei euro'),
            T('PPP holds in the long run if the real exchange rate $q_t = s_t - p_t + p^*_t$ is stationary: cointegration with $\\beta = (1, -1, 1)$', 'PPP este valabilă pe termen lung dacă cursul real $q_t = s_t - p_t + p^*_t$ este staționar: cointegrare cu $\\beta = (1, -1, 1)$')]),
          (T('Why it may fail', 'De ce poate să nu fie valabilă'),
           [T('\\refBalassa: fast productivity growth in tradables raises prices of non-tradables in a catching-up economy: a trend real appreciation', '\\refBalassa: creșterea rapidă a productivității în sectorul bunurilor comercializabile ridică prețurile serviciilor necomercializabile într-o economie în convergență: o apreciere reală în trend'),
            T('even where PPP holds, deviations decay slowly; the survey \\refTT\\ reports half-lives of several years', 'chiar acolo unde PPP este valabilă, abaterile se sting lent; sinteza \\refTT\\ raportează timpi de înjumătățire de mai mulți ani')])),
    wl='0.34', wr='0.62'))

chart(T('PPP for the leu', 'PPP pentru leu'), 'tsa_ch7_ppp', 'TSA_ch7_parity_conditions', [
    T('Monthly, @{pp.first}--@{pp.last}: EUR/RON (monthly mean of the BNR reference rate), HICP of Romania and of the euro area (Eurostat, 2015 = 100)',
      'Lunar, @{pp.first}--@{pp.last}: EUR/RON (media lunară a cursului de referință BNR), IAPC al României și al zonei euro (Eurostat, 2015 = 100)')], h='0.55\\textheight')

interp(('PPP for the leu', 'PPP pentru leu'), [
    (T('Since 2005 the leu lost @{pp.s} log points against the euro, while Romanian prices rose @{pp.rel} log points more than euro-area prices', 'Din 2005 leul a pierdut @{pp.s} puncte logaritmice față de euro, în timp ce prețurile din România au crescut cu @{pp.rel} puncte logaritmice mai mult decît cele din zona euro'),
     [T('the real exchange rate changed by $@{pp.q}$ log points: a real appreciation of the leu, concentrated after 2021', 'cursul real s-a modificat cu $@{pp.q}$ puncte logaritmice: o apreciere reală a leului, concentrată după 2021')]),
    T('ADF on $q_t$: p = @{pp.adf}; Engle--Granger of $s_t$ on $p_t - p^*_t$: slope @{pp.egb}, p = @{pp.eg}; Johansen on $(s_t, p_t, p^*_t)$: rank @{pp.jr}', 'ADF pe $q_t$: p = @{pp.adf}; Engle--Granger pentru $s_t$ pe $p_t - p^*_t$: panta @{pp.egb}, p = @{pp.eg}; Johansen pe $(s_t, p_t, p^*_t)$: rangul @{pp.jr}'),
    T('No evidence of PPP over 21 years, consistent with Balassa--Samuelson convergence; 21 years may also be too short for half-lives of several years', 'Nicio dovadă de PPP în 21 de ani, în acord cu convergența de tip Balassa--Samuelson; 21 de ani pot fi și prea puțini pentru timpi de înjumătățire de mai mulți ani')])

chart(T('ROBOR and Euribor', 'ROBOR și Euribor'), 'tsa_ch7_ro_ea_rates', 'TSA_ch7_parity_conditions', [
    T('Three-month money market rates (Eurostat), @{re.first}--@{re.last}: ROBOR 3M for Romania and Euribor 3M for the euro area', 'Ratele dobînzii pe piața monetară la trei luni (Eurostat), @{re.first}--@{re.last}: ROBOR 3M pentru România și Euribor 3M pentru zona euro')], h='0.55\\textheight')

interp(('ROBOR and Euribor', 'ROBOR și Euribor'), [
    (T('Full sample: Engle--Granger p = @{re.full.p}, Phillips--Ouliaris p @{re.full.pop}: the disinflation of the 1990s dominates and the tests disagree', 'Eșantionul complet: Engle--Granger p = @{re.full.p}, Phillips--Ouliaris p @{re.full.pop}: dezinflația din anii 1990 domină, iar testele nu concordă'), []),
    (T('Since 2010: p = @{re.10.p} (Engle--Granger) and @{re.10.pop} (Phillips--Ouliaris), slope @{re.10.b}, mean spread @{re.sp} percentage points', 'Din 2010: p = @{re.10.p} (Engle--Granger) și @{re.10.pop} (Phillips--Ouliaris), panta @{re.10.b}, spread-ul mediu @{re.sp} puncte procentuale'),
     [T('ECM for ROBOR: $\\hat\\gamma = @{re.g}$ ($t = @{re.gt}$), half-life about @{re.h} months: weak and slow adjustment', 'ECM pentru ROBOR: $\\hat\\gamma = @{re.g}$ ($t = @{re.gt}$), timp de înjumătățire de circa @{re.h} de luni: ajustare slabă și lentă')]),
    T('Romanian monetary policy follows its own inflation target; the link to Euribor is loose (Seminar 7, B4, adds the Romanian 10-year yield)', 'Politica monetară a României urmărește propria țintă de inflație; legătura cu Euribor este slabă (Seminarul 7, B4, adaugă randamentul la 10 ani al României)')])

chart(T('Three Central European currencies', 'Trei monede central-europene'), 'tsa_ch7_cee_fx', 'TSA_ch7_parity_conditions', [
    T('EUR/RON, EUR/HUF and EUR/PLN, cross rates from the BNR reference rates, month-end, 100 ln', 'EUR/RON, EUR/HUF și EUR/PLN, cursuri încrucișate calculate din cursurile de referință BNR, la sfîrșitul lunii, 100 ln')], h='0.55\\textheight')

interp(('the three currencies', 'celor trei monede'), [
    (T('Since July 2005: EUR/RON +@{ce.ron}, EUR/HUF +@{ce.huf}, EUR/PLN +@{ce.pln} log points; each rate is $I(1)$ (ADF p = @{ce.ron.p}; @{ce.huf.p}; @{ce.pln.p})', 'Din iulie 2005: EUR/RON +@{ce.ron}, EUR/HUF +@{ce.huf}, EUR/PLN +@{ce.pln} puncte logaritmice; fiecare curs este $I(1)$ (ADF p = @{ce.ron.p}; @{ce.huf.p}; @{ce.pln.p})'),
     []),
    (T('Johansen: trace @{ce.t0} for $r = 0$ (critical value @{jcv.t0}); also no cointegration since 2012 (trace @{ce.t12})', 'Johansen: urma @{ce.t0} pentru $r = 0$ (valoarea critică @{jcv.t0}); nici din 2012 nu există cointegrare (urma @{ce.t12})'),
     [T('a common anchor (the euro) and similar shocks do not create a common trend: each central bank and each inflation path adds its own', 'o ancoră comună (euro) și șocuri similare nu creează un trend comun: fiecare bancă centrală și fiecare evoluție a inflației adaugă propriul trend')]),
    T('The right model for these rates is a VAR in differences (Chapter 6); Seminar 7 (B3) repeats the analysis', 'Modelul potrivit pentru aceste cursuri este un VAR în diferențe (Capitolul 6); Seminarul 7 (B3) reia analiza')])

D.recap(('Economic examples', 'exemple economice'), [
    T('PPP for the leu is rejected over 2005--2026: real appreciation of a converging economy', 'PPP pentru leu este respinsă în 2005--2026: aprecierea reală a unei economii în convergență'),
    T('ROBOR and Euribor: a loose, slow link since 2010; the sample decides the verdict', 'ROBOR și Euribor: o legătură slabă și lentă din 2010; eșantionul decide verdictul'),
    T('Theory proposes $\\beta$; the tests, the sample and the deterministic terms decide whether the data agree', 'Teoria propune $\\beta$; testele, eșantionul și termenii determiniști decid dacă datele sînt de acord')])

# =============================================================================
# 9. PAIRS TRADING
# =============================================================================
D.section('An application: pairs trading', 'O aplicație: pairs trading')

D.frame(T('The idea of pairs trading', 'Ideea de pairs trading'), items(
    (T('Two shares exposed to the same shocks (two banks, two oil companies) may be cointegrated: their spread mean-reverts', 'Două acțiuni expuse acelorași șocuri (două bănci, două companii petroliere) pot fi cointegrate: spread-ul lor revine la medie'),
     [T('when the spread is unusually wide, sell the expensive share and buy the cheap one; close when the spread is back', 'cînd spread-ul este neobișnuit de mare, vindem acțiunea scumpă și cumpărăm acțiunea ieftină; închidem cînd spread-ul a revenit')]),
    (T('A \\textbf{relative-value} trade: the market direction cancels out; the profit comes from the error correction', 'O tranzacție de \\textbf{valoare relativă}: direcția pieței se anulează; profitul vine din corecția erorii'),
     [T('\\refGGR: the distance method on US shares, profitable over 1962--2002', '\\refGGR: metoda distanței pe acțiuni americane, profitabilă în perioada 1962--2002'),
      T('\\refDoFaff: the profits declined in later years', '\\refDoFaff: profiturile au scăzut în anii următori')]),
    T('A good test of a time-series idea: it must work \\textbf{out of sample}, \\textbf{after costs}, and on pairs chosen \\textbf{without looking ahead}',
      'Un bun test pentru o idee din seriile de timp: trebuie să funcționeze \\textbf{în afara eșantionului}, \\textbf{după costuri} și pe perechi alese \\textbf{fără a privi în viitor}')))

D.frame(T('A rule without look-ahead', 'O regulă fără informații din viitor'), items(
    (T('\\textbf{Formation} (the previous 252 trading days): test every pair with Engle--Granger; keep the pairs with p $<$ 0.05', '\\textbf{Formarea} (cele 252 de zile de tranzacționare anterioare): testăm fiecare pereche cu Engle--Granger; păstrăm perechile cu p $<$ 0,05'),
     [T('hedge ratio $\\hat b$ by OLS of $\\ln P_A$ on $\\ln P_B$; mean $\\bar s$ and standard deviation $\\hat\\sigma_s$ of the spread $s_t = \\ln P_{A,t} - \\hat a - \\hat b\\ln P_{B,t}$',
        'raportul de acoperire $\\hat b$ prin OLS pentru $\\ln P_A$ pe $\\ln P_B$; media $\\bar s$ și abaterea standard $\\hat\\sigma_s$ ale spread-ului $s_t = \\ln P_{A,t} - \\hat a - \\hat b\\ln P_{B,t}$')]),
    (T('\\textbf{Trading} (the next 126 days), with the formation parameters fixed: $z_t = (s_t - \\bar s)/\\hat\\sigma_s$', '\\textbf{Tranzacționarea} (următoarele 126 de zile), cu parametrii din formare ficși: $z_t = (s_t - \\bar s)/\\hat\\sigma_s$'),
     [T('$z_t > 2$: short A, long $\\hat b$ of B; $z_t < -2$: the opposite; close at $z_t = 0$ or at the end of the window', '$z_t > 2$: vindem A în lipsă, cumpărăm $\\hat b$ din B; $z_t < -2$: invers; închidem la $z_t = 0$ sau la sfîrșitul ferestrei'),
      T('daily return per unit of gross exposure: $\\pm(r_A - \\hat b\\,r_B)/(1 + |\\hat b|)$; equal capital per selected pair', 'randamentul zilnic pe unitatea de expunere brută: $\\pm(r_A - \\hat b\\,r_B)/(1 + |\\hat b|)$; capital egal pentru fiecare pereche selectată')]),
    T('\\textbf{Costs}: @{pb.bvb.cost}\\% (Bucharest) and @{pb.us.cost}\\% (US) of the gross exposure traded, at each opening and closing', '\\textbf{Costuri}: @{pb.bvb.cost}\\% (București) și @{pb.us.cost}\\% (SUA) din expunerea brută tranzacționată, la fiecare deschidere și închidere')))

chart(T('Banca Transilvania and BRD', 'Banca Transilvania și BRD'), 'tsa_ch7_pair', 'TSA_ch7_pairs_trading', [
    T('Adjusted daily prices since 2014; right: the spread of the full-sample regression, standardised, with the entry bands', 'Prețuri zilnice ajustate din 2014; dreapta: spread-ul regresiei pe întregul eșantion, standardizat, cu benzile de intrare')], h='0.55\\textheight')

interp(('the bank pair', 'perechii de bănci'), [
    (T('Full sample: $\\hat b = @{pa.b}$; the spread is an AR(1) with $\\hat\\rho = @{pa.rho}$: half-life about @{pa.h} trading days', 'Întregul eșantion: $\\hat b = @{pa.b}$; spread-ul este un AR(1) cu $\\hat\\rho = @{pa.rho}$: timp de înjumătățire de circa @{pa.h} de zile de tranzacționare'),
     [T('only @{pa.out}\\% of the days lie outside $\\pm 2$: few trading opportunities, each lasting months', 'doar @{pa.out}\\% din zile se află în afara benzii $\\pm 2$: puține ocazii de tranzacționare, fiecare durînd luni de zile')]),
    (T('Trading this picture with full-sample parameters gives @{pb.in.m}\\% per year (Sharpe ratio @{pb.in.s}), after costs', 'Aplicarea regulii pe acest grafic, cu parametrii din întregul eșantion, dă @{pb.in.m}\\% pe an (raportul Sharpe @{pb.in.s}), după costuri'),
     [T('an illusion: $\\hat b$, $\\bar s$ and $\\hat\\sigma_s$ use the future, and the pair was chosen because it looks cointegrated since 2014', 'o iluzie: $\\hat b$, $\\bar s$ și $\\hat\\sigma_s$ folosesc viitorul, iar perechea a fost aleasă pentru că pare cointegrată din 2014')]),
    T('With the rolling rule, the pair passes the formation test in none of its @{pb.tb.w} windows: one year of data is too short to detect such slow error correction',
      'Cu regula pe ferestre mobile, perechea nu trece testul de formare în niciuna dintre cele @{pb.tb.w} ferestre: un an de date este prea puțin pentru a detecta o corecție atît de lentă')])

chart(T('Pairs trading out of sample', 'Pairs trading în afara eșantionului'), 'tsa_ch7_pairs_backtest', 'TSA_ch7_pairs_trading', [
    T('Left: 8 liquid BVB shares (TLV, BRD, SNP, TGN, TEL, EL, SNG, SNN), since 2015; right: 6 US banks (JPM, BAC, C, WFC, GS, MS), since 2005', 'Stînga: 8 acțiuni lichide de la BVB (TLV, BRD, SNP, TGN, TEL, EL, SNG, SNN), din 2015; dreapta: 6 bănci americane (JPM, BAC, C, WFC, GS, MS), din 2005')], h='0.55\\textheight')

interp(('the backtest', 'testului istoric'), [
    (T('Bucharest: @{pb.bvb.w} windows, pairs found in @{pb.bvb.wp}, @{pb.bvb.tr} trades, @{pb.bvb.win}\\% of trades profitable, but the mean return per trade is $@{pb.bvb.mt}$\\%', 'București: @{pb.bvb.w} ferestre, perechi găsite în @{pb.bvb.wp}, @{pb.bvb.tr} de tranzacții, @{pb.bvb.win}\\% dintre ele profitabile, dar randamentul mediu pe tranzacție este $@{pb.bvb.mt}$\\%'),
     [T('gross @{pb.bvb.gross.m}\\% per year (Sharpe @{pb.bvb.gross.s}); net @{pb.bvb.net.m}\\% (Sharpe @{pb.bvb.net.s}); break-even cost @{pb.bvb.be}\\%', 'brut @{pb.bvb.gross.m}\\% pe an (Sharpe @{pb.bvb.gross.s}); net @{pb.bvb.net.m}\\% (Sharpe @{pb.bvb.net.s}); costul la care profitul net devine zero: @{pb.bvb.be}\\%')]),
    (T('US banks: gross @{pb.us.gross.m}\\% (Sharpe @{pb.us.gross.s}), net @{pb.us.net.m}\\% (Sharpe @{pb.us.net.s}), @{pb.us.tr} trades', 'Băncile americane: brut @{pb.us.gross.m}\\% (Sharpe @{pb.us.gross.s}), net @{pb.us.net.m}\\% (Sharpe @{pb.us.net.s}), @{pb.us.tr} de tranzacții'),
     [T('with a deep loss in the autumn of 2008: cointegration can break exactly when it matters', 'cu o pierdere mare în toamna lui 2008: cointegrarea se poate rupe exact cînd contează')]),
    T('Many small wins and a few large losses: the typical profile of a mean-reversion strategy', 'Multe cîștiguri mici și cîteva pierderi mari: profilul tipic al unei strategii de revenire la medie')])

D.frame(T('Pitfalls of pairs trading', 'Capcanele pairs trading'), items(
    (T('\\textbf{Multiple testing}: with 28 pairs and a 5\\% test, about @{pb.fp} pairs pass by chance in each window', '\\textbf{Testarea multiplă}: cu 28 de perechi și un test de 5\\%, circa @{pb.fp} perechi trec întîmplător în fiecare fereastră'), []),
    (T('\\textbf{Look-ahead}: choosing pairs, start dates or thresholds after seeing the whole sample', '\\textbf{Informațiile din viitor}: alegerea perechilor, a datelor de început sau a pragurilor după ce am văzut întregul eșantion'), []),
    (T('\\textbf{Breaks}: mergers, recapitalisations, regulation and crises change $\\beta$; the spread may never come back', '\\textbf{Rupturile}: fuziunile, recapitalizările, reglementările și crizele schimbă $\\beta$; spread-ul poate să nu mai revină'), []),
    (T('\\textbf{Market frictions}: short selling is limited on the BVB, borrowing shares costs a fee, bid-ask spreads are wide for small caps', '\\textbf{Fricțiunile pieței}: vînzarea în lipsă este limitată la BVB, împrumutul acțiunilor costă un comision, diferențele bid-ask sînt mari pentru companiile mici'), []),
    (T('\\textbf{Slow error correction}: half-lives of months mean capital tied up for months, and a long wait in a losing position', '\\textbf{Corecția lentă a erorii}: timpi de înjumătățire de luni înseamnă capital blocat luni de zile și o așteptare lungă într-o poziție pierzătoare'), [])))

D.recap(('Pairs trading', 'pairs trading'), [
    T('Pairs trading bets on error correction in the spread of two cointegrated shares', 'Pairs trading pariază pe corecția erorii în spread-ul a două acțiuni cointegrate'),
    T('In sample it looks attractive; out of sample, after costs, the profits on the BVB are close to zero', 'În eșantion pare atractiv; în afara eșantionului, după costuri, profiturile la BVB sînt aproape nule'),
    T('Report the selection, the costs, the number of trades and the worst episode, not only the Sharpe ratio', 'Raportăm selecția, costurile, numărul de tranzacții și cel mai rău episod, nu doar raportul Sharpe')])

# =============================================================================
# 10. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Code}: a first draft of a script that runs unit-root tests, Engle--Granger, Phillips--Ouliaris and Johansen on many series and tabulates the verdicts',
      '\\textbf{Cod}: o primă versiune a unui script care aplică teste de rădăcină unitară, Engle--Granger, Phillips--Ouliaris și Johansen pe multe serii și tabelează verdictele'),
    T('\\textbf{Explanation}: a second explanation of the Granger representation theorem, or of a \\texttt{VECM} output table', '\\textbf{Explicații}: o a doua explicație a teoremei de reprezentare a lui Granger sau a unui tabel de rezultate \\texttt{VECM}'),
    T('\\textbf{Exploration}: candidate pairs on the BVB, or parity conditions for several Central European countries', '\\textbf{Explorare}: perechi candidate la BVB sau condiții de paritate pentru mai multe țări din Europa Centrală'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that loads monthly US Treasury yields GS1, GS5 and GS10 from FRED, runs the Johansen trace test with a constant and BIC lags, fits a VECM with the chosen rank using statsmodels, and prints beta normalised on the first variables, alpha with t-statistics and the half-lives of the equilibrium errors.}',
        '\\aiprompt{Write Python code that loads monthly US Treasury yields GS1, GS5 and GS10 from FRED, runs the Johansen trace test with a constant and BIC lags, fits a VECM with the chosen rank using statsmodels, and prints beta normalised on the first variables, alpha with t-statistics and the half-lives of the equilibrium errors.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('The critical values: Engle--Granger (MacKinnon) for residuals, not Dickey--Fuller; Johansen for the right deterministic case', 'Valorile critice: Engle--Granger (MacKinnon) pentru reziduuri, nu Dickey--Fuller; Johansen pentru cazul potrivit al termenilor determiniști'),
    T('The direction of the null: ``no cointegration\'\' in Engle--Granger; ``rank $\\le r$\'\' in the trace test', 'Sensul ipotezei nule: „fără cointegrare” la Engle--Granger; „rang $\\le r$” la testul urmei'),
    T('That every series is $I(1)$ before testing, and that a high correlation of levels is not reported as cointegration', 'Că fiecare serie este $I(1)$ înainte de testare și că o corelație mare între niveluri nu este raportată drept cointegrare'),
    T('The sign of the adjustment coefficients and the half-lives, in the right time unit', 'Semnul coeficienților de ajustare și timpii de înjumătățire, în unitatea de timp potrivită'),
    T('Backtests: no look-ahead in pair selection, costs included, the number of pairs tested reported', 'Testele istorice: fără informații din viitor la alegerea perechilor, cu costuri incluse și cu numărul perechilor testate raportat'),
    T('Every cited reference: it must exist; check the DOI', 'Fiecare referință citată: trebuie să existe; verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Cointegrated $I(1)$ series share common stochastic trends; a combination $\\beta^\\top\\mathbf y_t$ is stationary', 'Seriile $I(1)$ cointegrate au trenduri stochastice comune; o combinație $\\beta^\\top\\mathbf y_t$ este staționară'),
    T('Engle--Granger: OLS in levels, then ADF on the residuals with MacKinnon critical values; check with Phillips--Ouliaris', 'Engle--Granger: OLS în niveluri, apoi ADF pe reziduuri, cu valorile critice MacKinnon; verificăm cu Phillips--Ouliaris'),
    T('ECM: changes respond to the lagged equilibrium error; $\\gamma$ gives the speed and the half-life', 'ECM: variațiile răspund la eroarea de echilibru decalată; $\\gamma$ dă viteza și timpul de înjumătățire'),
    T('VECM: $\\Pi = \\alpha\\beta^\\top$; Granger representation: cointegration $\\Leftrightarrow$ error correction; Johansen tests choose $r$', 'VECM: $\\Pi = \\alpha\\beta^\\top$; reprezentarea Granger: cointegrare $\\Leftrightarrow$ corecția erorii; testele Johansen aleg $r$'),
    T('Cointegration helps forecast relations (spreads), not levels; applications must be judged out of sample and after costs', 'Cointegrarea ajută la prognoza relațiilor (spread-uri), nu a nivelurilor; aplicațiile se evaluează în afara eșantionului și după costuri')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.4}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Cointegration', 'Cointegrarea') + ' & $\\mathbf y_t \\sim I(1)$, \\quad $\\beta^\\top\\mathbf y_t \\sim I(0)$',
     'Engle--Granger & $y_t = a + b x_t + u_t$; \\quad $\\Delta\\hat u_t = \\gamma\\hat u_{t-1} + \\sum_j\\delta_j\\Delta\\hat u_{t-j} + e_t$, \\quad $H_0$: $\\gamma = 0$',
     'ECM & $\\Delta y_t = c + \\gamma\\,\\hat u_{t-1} + \\delta_0\\Delta x_t + \\dots + \\varepsilon_t$, \\quad $h_{1/2} = \\ln 0.5/\\ln(1 + \\gamma)$',
     'VECM & $\\Delta\\mathbf y_t = \\alpha\\beta^\\top\\mathbf y_{t-1} + \\sum_{i=1}^{p-1}\\Gamma_i\\Delta\\mathbf y_{t-i} + \\mathbf u_t$, \\quad $\\Pi = \\sum_i A_i - I$, \\quad $\\Gamma_i = -\\sum_{j>i}A_j$',
     'Johansen & $\\lambda_{\\mathrm{trace}}(r) = -T\\sum_{i>r}\\ln(1 - \\hat\\lambda_i)$, \\quad $\\lambda_{\\max}(r) = -T\\ln(1 - \\hat\\lambda_{r+1})$',
     T('Equilibrium errors', 'Erorile de echilibru') + ' & $\\mathbf z_t = \\beta^\\top\\mathbf y_t$: \\quad $\\mathbf z_t = (I + \\beta^\\top\\alpha)\\mathbf z_{t-1} + \\dots$'],
    size='scriptsize') + '}')

D.frame(T('Self-assessment', 'Autoevaluare'), items(
    (T('\\textbf{Question}: Engle--Granger with two variables gives $\\tau = -3.1$ for $T = 300$. Do you reject ``no cointegration\'\' at 5\\%?', '\\textbf{Întrebare}: Engle--Granger cu două variabile dă $\\tau = -3{,}1$ pentru $T = 300$. Respingeți „fără cointegrare” la 5\\%?'),
     [T('\\textbf{Answer}: no: $-3.1 > @{sa.eg}$; with the Dickey--Fuller value ($@{sa.df}$) you would wrongly reject', '\\textbf{Răspuns}: nu: $-3{,}1 > @{sa.eg}$; cu valoarea Dickey--Fuller ($@{sa.df}$) ați respinge greșit')]),
    (T('\\textbf{Question}: an ECM gives $\\hat\\gamma = -0.1$ on monthly data. What is the half-life?', '\\textbf{Întrebare}: un ECM dă $\\hat\\gamma = -0{,}1$ pe date lunare. Cît este timpul de înjumătățire?'),
     [T('\\textbf{Answer}: $\\ln 0.5/\\ln 0.9 \\approx @{sa.hl}$ months', '\\textbf{Răspuns}: $\\ln 0{,}5/\\ln 0{,}9 \\approx @{sa.hl}$ luni')]),
    (T('\\textbf{Question}: four $I(1)$ variables have cointegration rank 3. How many common trends drive them?', '\\textbf{Întrebare}: patru variabile $I(1)$ au rangul de cointegrare 3. Cîte trenduri comune le antrenează?'),
     [T('\\textbf{Answer}: $n - r = 1$', '\\textbf{Răspuns}: $n - r = 1$')]),
    T('Next: Chapter 8, long memory: series between $I(0)$ and $I(1)$, fractional differencing and ARFIMA', 'Urmează: Capitolul 8, memoria lungă: serii între $I(0)$ și $I(1)$, diferențierea fracționară și ARFIMA')))

D.references(bib())

if __name__ == '__main__':
    finalize(D.write(V))
