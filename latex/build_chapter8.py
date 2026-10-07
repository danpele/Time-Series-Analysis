r"""
build_chapter8.py -- Capitolul 8 (Memorie lungă și ARFIMA), EN + RO dintr-o singură sursă
=========================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_08/ch8_numbers.json (generate_all_charts.py).
Nicio cifră nu este scrisă de mînă. Partea despre memoria lungă a vechiului capitol 8 (memorie lungă și ML) plus
materialul SFM, capitolul 11 (piețe fractale și memorie lungă), reformulate ca modelare de serii de timp.
Ieșire:
  EN/Courses/chapter8_long_memory_arfima.tex
  RO/Cursuri/capitol8_memorie_lunga_arfima.tex
Rulare:
  python3 Quantlets/Ch_08/generate_all_charts.py
  python3 latex/build_chapter8.py && python3 latex/tsa_build.py compile 8
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, photo   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch8_common import QLURL, REFS, T, bib, date, finalize, load, month   # noqa: E402


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


N = load()
V = Values()
D = Deck(8, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\quantlet{{{folder.replace("_", chr(92) + "_")}}}{{\\qlurl{{{folder}}}}}'


def chart(title, fig, folder, bullets, h='0.6\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def interp(title, bullets, size='small'):
    D.frame(T(f'Interpreting {title[0]}', f'Interpretarea {title[1]}'), items(*bullets), size)


PH = {
    'hurst': ('ch8_hurst_1953.jpg', C + 'Harold_Edwin_Hurst_in_1953.jpg',
              T('Photo', 'Foto') + ': Elliott \\& Fry (1953); ' + T('public domain', 'domeniu public') + '; Wikimedia Commons'),
    'aswan': ('ch8_aswan_low_dam.jpg', C + 'Aswan_Low_Dam_Egypt_1.jpg',
              T('Photo', 'Foto') + ': Karelj (2010); ' + T('public domain', 'domeniu public') + '; Wikimedia Commons'),
    'mandelbrot': ('ch8_mandelbrot_2006.jpg', C + 'Mandelbrot_p1130876.jpg',
                   T('Photo', 'Foto') + ': David Monniaux (2006); CC BY-SA 3.0; Wikimedia Commons'),
    'granger': ('ch3_clive_granger_2008.jpg', C + 'Clive_Granger_by_Olaf_Storbeck_(3x4_cropped).jpg',
                T('Photo', 'Foto') + ': Olaf Storbeck (2008); CC BY-SA 2.0; Wikimedia Commons'),
    'engle': ('ch5_robert_engle_2022.jpg', C + '0603-Kraneshares_KRBN-RobertEngle-JonDemske-16_(cropped).jpg',
              T('Photo', 'Foto') + ': Jon Demske (2022); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


def two(left, right, wl='0.38', wr='0.6'):
    return (f'\\begin{{columns}}[T]\n\\begin{{column}}{{{wl}\\textwidth}}\n{left}\n\\end{{column}}\n'
            f'\\begin{{column}}{{{wr}\\textwidth}}\n{right}\n\\end{{column}}\n\\end{{columns}}')


# =============================================================================
# CIFRE
# =============================================================================
P = V.put
EX = N['ex']
for i, w in enumerate(EX['w04']):
    P(f'w4.{i}', w, 4 if i > 2 else 2)
for i, r in enumerate(EX['r03']):
    P(f'r3.{i}', r, 3)
P('v03', EX['v03'], 3)
P('arsame', EX['ar_same'], 4)
RS = EX['rs']
P('rs.mean', RS['mean'], 3)
P('rs.R', RS['R'], 3)
P('rs.S', RS['S'], 3)
P('rs.RS', RS['RS'], 2)
P('rs.max', RS['max'], 3)
P('rs.min', RS['min'], 3)

DA = N['data']
P('nile.mean', DA['nile_mean'], 0)
P('i12.max', DA['i12_max'], 1)
V.raw('i12.maxd', month(DA['i12_max_d']))
P('i12.last', DA['i12_last'], 1)
V.raw('i12.lastd', month(DA['i12_last_d']))
V.raw('im.n', str(DA['im_n']))
V.raw('im.last', month(DA['im_last']))
P('u.max', DA['u_max'], 1)
V.raw('u.maxd', month(DA['u_max_d']))
P('u.last', DA['u_last'], 1)
V.raw('u.lastd', month(DA['u_last_d']))
P('rv.sp', DA['rv_max']['sp500'], 0)
V.raw('rv.spd', month(DA['rv_max_d']['sp500']))
P('rv.bet', DA['rv_max']['bet'], 0)
V.raw('rv.betd', month(DA['rv_max_d']['bet']))

A = N['acf']
for k in ('nile', 'infl', 'usinfl', 'sp500'):
    P(f'acf.{k}.r1', A[k]['r1'], 2)
    P(f'acf.{k}.r10', A[k]['r10'], 2)
    P(f'acf.{k}.ar10', A[k]['ar10'], 4)
    V.raw(f'acf.{k}.npos', str(A[k]['npos']))
    V.raw(f'acf.{k}.L', str(A[k]['L']))

DC = N['decay']
for k in ('phi', 'r10', 'r50', 'r100', 'r1000'):
    P(f'dc.{k}', DC[k], 3)
P('dc.a10', DC['a10'], 3)
V.raw('dc.a50', '⁅1.6⁆ \\cdot 10^{-9}')
P('dc.sum100', DC['sum100'], 1)
P('dc.sum1000', DC['sum1000'], 1)
P('dc.asum', DC['asum'], 1)
P('dc.C', DC['C'], 3)

W = N['weights']
P('wt.7.1', W['0.7']['pi'][1], 2)
P('wt.7.2', W['0.7']['pi'][2], 3)
for d in ('0.2', '0.4', '0.7'):
    k = d.replace('0.', '')
    P(f'wt.{k}.p10', W[d]['psi10'], 3)
    P(f'wt.{k}.p100', W[d]['psi100'], 3)
P('wt.ar10', W['ar10'], 3)
V.raw('wt.ar100', '⁅2.7⁆ \\cdot 10^{-5}')

FF = N['ffd']
P('ffd.crit', FF['crit'], 2)
P('ffd.dmin', FF['dmin'], 2)
P('ffd.cor', FF['cor_dmin'], 3)
V.int('ffd.w', FF['width_dmin'])
P('ffd.adf0', FF['adf0'], 2)
P('ffd.adf1', FF['adf1'], 1)
P('ffd.cor1', FF['cor1'], 2)
V.int('ffd.n', FF['n'])

PA = N['paths']
P('pa.p.r1', PA['0.4']['rho1'], 3)
P('pa.p.r10', PA['0.4']['rho10'], 3)
P('pa.p.r50', PA['0.4']['rho50'], 3)
P('pa.p.v', PA['0.4']['var_th'], 2)
P('pa.n.r1', PA['-0.4']['rho1'], 3)
P('pa.n.r2', PA['-0.4']['rho2'], 3)
P('pa.n.v', PA['-0.4']['var_th'], 2)
FB = N['fbm']
P('fb.3', FB['0.3']['acf1_th'], 2)
P('fb.7', FB['0.7']['acf1_th'], 2)
P('fb.3s', FB['0.3']['acf1'], 2)
P('fb.7s', FB['0.7']['acf1'], 2)

RD = N['rsdfa']
P('rd.ui.rs', RD['rs|US inflation, monthly'], 2)
P('rd.ui.dfa', RD['dfa|US inflation, monthly'], 2)
P('rd.r.rs', RD['rs|S&P 500 returns r'], 2)
P('rd.r.dfa', RD['dfa|S&P 500 returns r'], 2)
P('rd.a.rs', RD['rs|S&P 500 |r|'], 2)
P('rd.a.dfa', RD['dfa|S&P 500 |r|'], 2)

G = N['gph']
for k in ('nile', 'infl'):
    P(f'g.{k}.d', G[k]['d'], 2)
    P(f'g.{k}.se', G[k]['se'], 2)
    V.raw(f'g.{k}.m', str(G[k]['m']))
    V.raw(f'g.{k}.T', str(G[k]['T']))
    P(f'g.{k}.per', G[k]['period_m'], 1)
    P(f'g.{k}.lw', G[k]['lw'], 2)
    P(f'g.{k}.lwse', G[k]['lw_se'], 2)

BWD = N['bw']
for k in ('infl', 'usinfl'):
    b = BWD[k]
    P(f'bw.{k}.lo', min(b['lw']), 2)
    P(f'bw.{k}.hi', max(b['lw']), 2)
    P(f'bw.{k}.g.lo', min(b['gph']), 2)
    P(f'bw.{k}.g.hi', max(b['gph']), 2)
    V.raw(f'bw.{k}.m0', str(b['m'][0]))
    V.raw(f'bw.{k}.m1', str(b['m'][-1]))
    V.raw(f'bw.{k}.T', str(b['T']))

MC = N['mc']
V.raw('mc.T', str(MC['T']))
V.raw('mc.reps', str(MC['reps']))
for lab, key in [('white noise, d = 0', 'wn'), ('ARFIMA(0,0.3,0), d = 0.3', 'af'), ('AR(1), phi = 0.6, d = 0', 'ar')]:
    for e, ek in [('GPH', 'gph'), ('LW', 'lw'), ('Whittle', 'wh'), ('R/S', 'rs'), ('DFA', 'dfa')]:
        P(f'mc.{key}.{ek}.m', MC['res'][lab][e]['mean'], 2)
        P(f'mc.{key}.{ek}.s', MC['res'][lab][e]['sd'], 2)

FT = N['fit']
for r in FT['table']:
    k = r['name'].replace('(', '').replace(')', '').replace(',', '')
    P(f'ft.{k}.ll', r['loglik'], 1)
    P(f'ft.{k}.aic', r['aic'], 1)
    P(f'ft.{k}.bic', r['bic'], 1)
    P(f'ft.{k}.d', r['d'], 2)
    if r['phi']:
        P(f'ft.{k}.phi', r['phi'][0], 2)
    if r['theta']:
        P(f'ft.{k}.th', r['theta'][0], 2)
r0 = next(r for r in FT['table'] if r['name'] == 'ARFIMA(0,d,0)')
P('ft.se', r0['se_d'], 3)
V.raw('ft.T', str(FT['T']))
for k in ('r12', 'r24', 'd12', 'a12', 'd24', 'a24'):
    P(f'ft.{k}', FT[k], 2)

TB = N['table']
TROWS = [('Nile flow', 'Nile', 'Debitul Nilului'), ('RO inflation, monthly SA', 'RO inflation, monthly SA', 'Inflația RO, lunară, ajust.'),
         ('RO inflation, 12-month', 'RO inflation, 12-month', 'Inflația RO, anuală'),
         ('US inflation, monthly', 'US inflation, monthly', 'Inflația SUA, lunară'),
         ('US inflation, 1985-2019', 'US inflation, 1985--2019', 'Inflația SUA, 1985--2019'),
         ('US unemployment', 'US unemployment', 'Șomajul SUA'),
         ('S&P 500 r', 'S\\&P 500, $r_t$', 'S\\&P 500, $r_t$'), ('S&P 500 |r|', 'S\\&P 500, $|r_t|$', 'S\\&P 500, $|r_t|$'),
         ('BET r', 'BET, $r_t$', 'BET, $r_t$'), ('BET |r|', 'BET, $|r_t|$', 'BET, $|r_t|$'),
         ('EUR/RON r', 'EUR/RON, $r_t$', 'EUR/RON, $r_t$'), ('EUR/RON |r|', 'EUR/RON, $|r_t|$', 'EUR/RON, $|r_t|$')]
for i, (k, _, _) in enumerate(TROWS):
    e = TB[k]
    V.int(f'tb{i}.n', e['n'])
    for c in ('rs', 'dfa', 'gph', 'lw'):
        P(f'tb{i}.{c}', e[c], 2)
    P(f'tb{i}.se', e['lw_se'], 2)

FC = N['fc']
for k, R in (('ro', FC['ro']), ('us', FC['us'])):
    V.raw(f'fc.{k}.n', str(R['n']))
    V.raw(f'fc.{k}.f0', month(R['first']))
    V.raw(f'fc.{k}.f1', month(R['last']))
    for h in (1, 6, 12, 24):
        for m in ('ARFIMA', 'AR', 'RW', 'Mean'):
            P(f'fc.{k}.{m}.{h}', R['rmse'][m][h - 1], 2)
        P(f'fc.{k}.rel.{h}', R['rmse']['ARFIMA'][h - 1] / R['rmse']['AR'][h - 1], 3)
    P(f'fc.{k}.d', R['last_est']['d'], 2)
    V.raw(f'fc.{k}.pmed', str(int(R['p_median'])))
FP = N['fcpath']
for k in ('d', 'phi', 'mean', 'last', 'fa1', 'fa12', 'fa36', 'fr1', 'fr12', 'fr36', 'm12'):
    P(f'fp.{k}', FP[k], 2)
V.raw('fp.p', str(FP['p']))
V.raw('fp.lastd', month(FP['last_d']))
V.raw('fp.T', str(FP['T']))

VO = N['vol']
for k in ('sp500', 'bet'):
    v = VO[k]
    V.int(f'vo.{k}.n', v['n'])
    P(f'vo.{k}.band', v['band'], 3)
    for s, sk in (('returns', 'r'), ('|r|', 'a'), ('r^2', 's')):
        for lag in ('1', '10', '100', '250'):
            P(f'vo.{k}.{sk}.{lag}', v[s][lag], 2)
        V.raw(f'vo.{k}.{sk}.npos', str(v[s]['npos']))
    P(f'vo.{k}.alw', v['abs_lw'], 2)
    P(f'vo.{k}.shuf', v['abs_shuf_lw'], 2)
    P(f'vo.{k}.rlw', v['r_lw'], 2)

RV = N['rv']
for k in ('sp500', 'bet'):
    r = RV[k]
    V.raw(f'rv.{k}.T', str(r['T']))
    for c in ('ml', 'lw', 'gph', 'r1', 'r12', 'r24', 'f12', 'a12', 'ml_phi', 'a_phi', 'a_theta'):
        P(f'rv.{k}.{c}', r[c], 2)
    P(f'rv.{k}.bicf', r['bic_f'], 1)
    P(f'rv.{k}.bica', r['bic_a'], 1)

VM = N['volm']
for k in ('sp500', 'bet'):
    m = VM[k]
    P(f'vm.{k}.ab', m['garch']['alpha'] + m['garch']['beta'], 3)
    P(f'vm.{k}.a', m['garch']['alpha'], 3)
    P(f'vm.{k}.b', m['garch']['beta'], 3)
    P(f'vm.{k}.d', m['figarch']['d'], 2)
    P(f'vm.{k}.phi', m['figarch']['phi'], 2)
    P(f'vm.{k}.beta', m['figarch']['beta'], 2)
    P(f'vm.{k}.bicg', m['garch']['bic'], 0)
    P(f'vm.{k}.bicf', m['figarch']['bic'], 0)
    P(f'vm.{k}.dbic', m['garch']['bic'] - m['figarch']['bic'], 0)
    V.raw(f'vm.{k}.w100g', '⁅' + f"{m['w100_g'] * 1e7:.1f}" + '⁆ \\cdot 10^{-7}' if k == 'sp500' else '⁅' + f"{m['w100_g'] * 1e11:.1f}" + '⁆ \\cdot 10^{-11}')
    P(f'vm.{k}.w100f', m['w100_f'] * 1e4, 1)
H_ = VM['har']
for k in ('bd', 'bw', 'bm', 'w1', 'w2', 'w6', 'd', 'sum'):
    P(f'har.{k}', H_[k], 2)
P('har.w6', H_['w6'], 3)

SP = N['spurious']
V.raw('sp.T', str(SP['T']))
V.raw('sp.reps', str(SP['reps']))
for dl in ('0.2', '0.5', '1.0'):
    k = dl.replace('.', '')
    P(f'sp.b{k}.lw', SP['break'][dl]['lw'], 2)
    P(f'sp.b{k}.gph', SP['break'][dl]['gph'], 2)
for p in ('0.001', '0.01', '0.2'):
    k = p.replace('.', '')
    P(f'sp.s{k}.lw', SP['switch'][p]['lw'], 2)
    P(f'sp.s{k}.sw', SP['switch'][p]['switches'], 0)

NI = N['nile']
P('ni.pre', NI['pre'], 0)
P('ni.post', NI['post'], 0)
P('ni.drop', NI['drop'], 0)
for k in ('raw', 'adj'):
    for c in ('H', 'lw', 'gph', 'ml', 'ml_se', 'r1'):
        P(f'ni.{k}.{c}', NI[k][c], 2)

RO = N['rolling']
P('rl.us.max', RO['us']['max'], 2)
V.raw('rl.us.maxd', month(RO['us']['max_d']))
P('rl.us.min', RO['us']['min'], 2)
V.raw('rl.us.mind', month(RO['us']['min_d']))
P('rl.us.last', RO['us']['last'], 2)
P('rl.us.hi', RO['us']['band']['q975'], 2)
P('rl.us.lo', RO['us']['band']['q025'], 2)
P('rl.h.lo', RO['band_h']['q025'], 2)
P('rl.h.hi', RO['band_h']['q975'], 2)
for k in ('sp500', 'bet'):
    P(f'rl.{k}.min', RO[k]['min'], 2)
    P(f'rl.{k}.max', RO[k]['max'], 2)
    V.raw(f'rl.{k}.mind', month(RO[k]['min_d']))
    V.raw(f'rl.{k}.maxd', month(RO[k]['max_d']))
    P(f'rl.{k}.above', 100 * RO[k]['above'], 0)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how long does a shock stay in a series? ARMA models (Chapter 2) forget at an exponential rate; some series forget much more slowly',
       '\\textbf{Întrebarea}: cît timp rămîne un șoc într-o serie? Modelele ARMA (Capitolul 2) uită exponențial; unele serii uită mult mai lent'),
     [T('examples: the Nile floods, inflation, unemployment, the volatility of stock returns', 'exemple: viiturile Nilului, inflația, șomajul, volatilitatea randamentelor bursiere')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('short and long memory: exponential against hyperbolic decay of the ACF', 'memorie scurtă și memorie lungă: descreșterea exponențială și cea hiperbolică a ACF'),
      T('fractional differencing $(1-L)^d$ and the ARFIMA$(p,d,q)$ model', 'diferențierea fracționară $(1-L)^d$ și modelul ARFIMA$(p,d,q)$'),
      T('estimating $d$: R/S, DFA, GPH, local Whittle, maximum likelihood; forecasting', 'estimarea lui $d$: R/S, DFA, GPH, Whittle local, verosimilitate maximă; prognoza'),
      T('long memory in volatility; spurious long memory from breaks and regimes', 'memoria lungă a volatilității; memoria lungă aparentă, produsă de rupturi și regimuri')]),
    T('We build on Chapter 1 (ACF, spectrum), Chapter 3 (unit roots) and Chapter 5 (GARCH); Seminar 8 comes before this lecture',
      'Pornim de la Capitolul 1 (ACF, spectru), Capitolul 3 (rădăcini unitare) și Capitolul 5 (GARCH); Seminarul 8 are loc înaintea acestui curs')))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Distinguish short from long memory by the decay of the ACF and by the spectrum near frequency zero', 'Distingeți memoria scurtă de memoria lungă după descreșterea ACF și după spectrul din apropierea frecvenței zero'),
    T('Compute the weights of $(1-L)^d$ and the autocorrelations of ARFIMA$(0,d,0)$', 'Calculați ponderile lui $(1-L)^d$ și autocorelațiile unui ARFIMA$(0,d,0)$'),
    T('State the stationarity and invertibility ranges of $d$ and the link $H = d + 1/2$', 'Precizați intervalele de staționaritate și de inversabilitate ale lui $d$ și legătura $H = d + 1/2$'),
    T('Estimate $d$ with R/S, DFA, GPH, local Whittle and maximum likelihood, and judge their uncertainty', 'Estimați $d$ cu R/S, DFA, GPH, Whittle local și verosimilitate maximă și apreciați incertitudinea estimărilor'),
    T('Forecast with ARFIMA and compare it fairly with ARMA benchmarks', 'Faceți prognoze cu ARFIMA și comparați-le corect cu modele ARMA de referință'),
    T('Recognise spurious long memory caused by breaks and regime shifts', 'Recunoașteți memoria lungă aparentă, cauzată de rupturi și de schimbări de regim')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook background: \\refHP\\ (ARIMA and GARCH models, the building blocks of this chapter)', 'Manual: \\refHP\\ (modelele ARIMA și GARCH, pe care se sprijină acest capitol)'),
     [T('theory: \\refBDtm, Sec.~13.2 (long memory processes); \\refBeran; survey: \\refBaillie', 'teorie: \\refBDtm, secț.~13.2 (procese cu memorie lungă); \\refBeran; sinteză: \\refBaillie'),
      T('history: \\refGraves, ``A brief history of long memory: Hurst, Mandelbrot and the road to ARFIMA\'\'', 'istorie: \\refGraves, „A brief history of long memory: Hurst, Mandelbrot and the road to ARFIMA”')]),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_08}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_08}'),
     [T('the estimators are written out in the code (R/S, DFA, GPH, local Whittle, exact ML); \\texttt{statsmodels} has no ARFIMA class; FIGARCH comes from the \\texttt{arch} package',
        'estimatorii sînt scriși explicit în cod (R/S, DFA, GPH, Whittle local, ML exactă); \\texttt{statsmodels} nu are o clasă ARFIMA; FIGARCH provine din pachetul \\texttt{arch}')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter8_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter8_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. MEMORIE SCURTĂ ȘI MEMORIE LUNGĂ
# =============================================================================
D.section('Short and long memory', 'Memorie scurtă și memorie lungă')

D.frame(T('Harold Edwin Hurst and the Nile', 'Harold Edwin Hurst și Nilul'), two(
    ph('hurst', T('H. E. Hurst (1880--1978)', 'H. E. Hurst (1880--1978)'), h='0.38\\textheight'),
    items((T('Hurst spent more than 60 years in Egypt measuring the Nile, to size the reservoirs of the Aswan dams', 'Hurst a lucrat peste 60 de ani în Egipt, măsurînd Nilul pentru a dimensiona rezervoarele barajelor de la Aswan'),
           [T('a reservoir must cover the longest run of dry years: its size depends on the \\textbf{range} of cumulative inflows', 'un rezervor trebuie să acopere cel mai lung șir de ani secetoși: mărimea lui depinde de \\textbf{amplitudinea} intrărilor cumulate')]),
          (T('\\refHurst: the range grows like $n^{H}$ with $H \\approx 0.7$, not like $n^{0.5}$ as for independent years', '\\refHurst: amplitudinea crește ca $n^{H}$, cu $H \\approx 0{,}7$, nu ca $n^{0{,}5}$, cum ar fi pentru ani independenți'),
           [T('wet years cluster with wet years and dry with dry, over decades: the ``Joseph effect\'\' of \\refNoah', 'anii ploioși se grupează cu anii ploioși și cei secetoși cu cei secetoși, pe decenii: „efectul Iosif” din \\refNoah')]),
          T('This is \\textbf{long memory}: a dependence that fades too slowly for any low-order ARMA model', 'Aceasta este \\textbf{memoria lungă}: o dependență care se stinge prea lent pentru orice model ARMA de ordin mic'))
    + ph('aswan', T('The Aswan Low Dam (1902), on the Nile', 'Barajul Aswan Low (1902), pe Nil'), h='0.17\\textheight')), size='footnotesize')

chart(T('Four persistent series', 'Patru serii persistente'), 'tsa_ch8_memory_data', 'TSA_ch8_memory_data', [
    T('Nile flow at Aswan, 1871--1970 (statsmodels); Romanian HICP inflation, 12-month and monthly, seasonally adjusted (SA) with the monthly means (Eurostat); US unemployment rate (FRED UNRATE); monthly realised volatility of daily S\\&P 500 and BET returns (EODHD)',
      'Debitul Nilului la Aswan, 1871--1970 (statsmodels); inflația IAPC din România, anuală și lunară, ajustată sezonier (SA) cu mediile lunare (Eurostat); rata șomajului din SUA (FRED UNRATE); volatilitatea realizată lunară a randamentelor zilnice S\\&P 500 și BET (EODHD)')],
    h='0.7\\textheight')

interp(('the four series', 'celor patru serii'), [
    (T('Each series stays above or below its mean for long stretches', 'Fiecare serie stă mult timp deasupra sau sub media ei'),
     [T('Nile: mean @{nile.mean} ($10^8$ m$^3$), long dry spells after 1900; Romanian 12-month inflation peaked at @{i12.max}\\% in @{i12.maxd} and was @{i12.last}\\% in @{i12.lastd}',
        'Nilul: media @{nile.mean} ($10^8$ m$^3$), perioade secetoase lungi după 1900; inflația anuală din România a atins @{i12.max}\\% în @{i12.maxd} și era @{i12.last}\\% în @{i12.lastd}')]),
    (T('US unemployment rises fast in recessions (peak @{u.max}\\% in @{u.maxd}) and falls back slowly; latest @{u.last}\\% (@{u.lastd})',
       'Șomajul din SUA crește repede în recesiuni (maximum @{u.max}\\% în @{u.maxd}) și scade lent; ultima valoare @{u.last}\\% (@{u.lastd})'),
     [T('realised volatility: calm and turbulent years alternate; maxima @{rv.sp}\\% (S\\&P 500, @{rv.spd}) and @{rv.bet}\\% (BET, @{rv.betd})', 'volatilitatea realizată: anii calmi alternează cu anii agitați; maxime @{rv.sp}\\% (S\\&P 500, @{rv.spd}) și @{rv.bet}\\% (BET, @{rv.betd})')]),
    T('The question of the chapter: is this a slowly decaying (long) memory, a near unit root, or a sequence of regimes?', 'Întrebarea capitolului: este aceasta o memorie care se stinge lent (lungă), o rădăcină aproape unitară sau o succesiune de regimuri?')])

D.frame(T('Short memory', 'Memoria scurtă'), items(
    (T('Recall (Chapter 1): $\\rho(k) = \\gamma(k)/\\gamma(0)$, the autocorrelation at lag $k$ of a stationary series', 'Reamintim (Capitolul 1): $\\rho(k) = \\gamma(k)/\\gamma(0)$, autocorelația la lagul $k$ a unei serii staționare'),
     [T('\\textbf{short memory}: the autocorrelations are absolutely summable, $\\sum_{k=0}^{\\infty}|\\rho(k)| < \\infty$', '\\textbf{memorie scurtă}: autocorelațiile sînt absolut sumabile, $\\sum_{k=0}^{\\infty}|\\rho(k)| < \\infty$')]),
    (T('Stationary ARMA processes: $|\\rho(k)| \\le C r^k$ with $0 < r < 1$: \\textbf{exponential} decay', 'Procesele ARMA staționare: $|\\rho(k)| \\le C r^k$, cu $0 < r < 1$: descreștere \\textbf{exponențială}'),
     [T('$C > 0$: a constant; $r$: the decay rate per lag (for an AR(1), $r = |\\phi|$)', '$C > 0$: o constantă; $r$: rata de descreștere pe lag (pentru un AR(1), $r = |\\phi|$)'),
      T('AR(1): $\\rho(k) = \\phi^k$; with $\\phi = 0.9$, $\\rho(50) = 0.005$: after 50 periods the shock is forgotten', 'AR(1): $\\rho(k) = \\phi^k$; pentru $\\phi = 0{,}9$, $\\rho(50) = 0{,}005$: după 50 de perioade șocul este uitat')]),
    (T('Spectral view: the spectral density $f(\\lambda)$ is finite and positive at frequency $\\lambda = 0$', 'În domeniul frecvenței: densitatea spectrală $f(\\lambda)$ este finită și pozitivă la frecvența $\\lambda = 0$'),
     [T('$\\lambda \\in [0, \\pi]$: the frequency in radians per period (a cycle of length $2\\pi/\\lambda$ periods); $\\gamma(k)$: the autocovariance at lag $k$', '$\\lambda \\in [0, \\pi]$: frecvența, în radiani pe perioadă (un ciclu de lungime $2\\pi/\\lambda$ perioade); $\\gamma(k)$: autocovarianța la lagul $k$'),
      T('$f(0) = \\frac{1}{2\\pi}\\sum_k \\gamma(k)$: the long-run variance is finite; the variance of the sample mean falls like $1/T$', '$f(0) = \\frac{1}{2\\pi}\\sum_k \\gamma(k)$: varianța pe termen lung este finită; varianța mediei de selecție scade ca $1/T$')])))

D.frame(T('Long memory', 'Memoria lungă'), items(
    (T('\\textbf{Long memory} (long-range dependence): $\\rho(k) \\sim C\\,k^{2d-1}$ as $k \\to \\infty$, with $0 < d < 1/2$', '\\textbf{Memorie lungă} (dependență pe termen lung): $\\rho(k) \\sim C\\,k^{2d-1}$ cînd $k \\to \\infty$, cu $0 < d < 1/2$'),
     [T('a \\textbf{hyperbolic} (power-law) decay: the exponent $2d - 1$ lies in $(-1, 0)$', 'o descreștere \\textbf{hiperbolică} (de tip putere): exponentul $2d - 1$ se află în $(-1, 0)$'),
      T('the sum $\\sum_k \\rho(k)$ diverges: distant observations still matter together', 'suma $\\sum_k \\rho(k)$ diverge: observațiile îndepărtate contează încă, luate împreună'),
      T('$\\sim$: the ratio of the two sides tends to 1; $C, G, c > 0$: constants; $d$: the memory parameter', '$\\sim$: raportul celor doi membri tinde la 1; $C, G, c > 0$: constante; $d$: parametrul de memorie')]),
    (T('Spectral view: $f(\\lambda) \\sim G\\,\\lambda^{-2d}$ as $\\lambda \\to 0$: a \\textbf{pole} at frequency zero', 'În domeniul frecvenței: $f(\\lambda) \\sim G\\,\\lambda^{-2d}$ cînd $\\lambda \\to 0$: un \\textbf{pol} la frecvența zero'),
     [T('the low frequencies (slow cycles of all lengths) carry most of the variance', 'frecvențele joase (cicluri lente de toate lungimile) poartă cea mai mare parte a varianței')]),
    (T('Consequence: $\\Var(\\bar{x}_T) \\sim c\\,T^{2d-1}$ ($\\bar{x}_T$: the sample mean of $T$ observations), slower than $1/T$', 'Consecință: $\\Var(\\bar{x}_T) \\sim c\\,T^{2d-1}$ ($\\bar{x}_T$: media de selecție a $T$ observații), mai lent decît $1/T$'),
     [T('confidence intervals that assume independence are too narrow; the memory parameter $d$ measures how slowly the past fades', 'intervalele de încredere care presupun independența sînt prea înguste; parametrul de memorie $d$ măsoară cît de lent se estompează trecutul')])))

chart(T('Sample ACF against an AR(1)', 'ACF de selecție comparată cu un AR(1)'), 'tsa_ch8_memory_acf', 'TSA_ch8_memory_data', [
    T('Bars: sample ACF; dashed line: $\\hat\\rho(1)^k$, the ACF of an AR(1) with the same lag-1 autocorrelation; shaded: $\\pm 1.96/\\sqrt{T}$',
      'Bare: ACF de selecție; linia punctată: $\\hat\\rho(1)^k$, ACF a unui AR(1) cu aceeași autocorelație de ordinul 1; zona colorată: $\\pm 1{,}96/\\sqrt{T}$')],
    h='0.7\\textheight')

interp(('the sample ACF', 'ACF de selecție'), [
    (T('Nile: $\\hat\\rho(1) = @{acf.nile.r1}$; an AR(1) would give $\\rho(10) = @{acf.nile.ar10}$, the data give @{acf.nile.r10}', 'Nilul: $\\hat\\rho(1) = @{acf.nile.r1}$; un AR(1) ar da $\\rho(10) = @{acf.nile.ar10}$, datele dau @{acf.nile.r10}'),
     [T('@{acf.nile.npos} of the first @{acf.nile.L} autocorrelations lie above the white-noise band', '@{acf.nile.npos} dintre primele @{acf.nile.L} autocorelații sînt deasupra benzii zgomotului alb')]),
    (T('Romanian monthly inflation: $\\hat\\rho(1) = @{acf.infl.r1}$, $\\hat\\rho(10) = @{acf.infl.r10}$; @{acf.infl.npos} of @{acf.infl.L} lags above the band', 'Inflația lunară din România: $\\hat\\rho(1) = @{acf.infl.r1}$, $\\hat\\rho(10) = @{acf.infl.r10}$; @{acf.infl.npos} din @{acf.infl.L} de laguri deasupra benzii'),
     [T('US monthly inflation: @{acf.usinfl.npos} of @{acf.usinfl.L}; S\\&P 500 log realised volatility: @{acf.sp500.npos} of @{acf.sp500.L}', 'inflația lunară din SUA: @{acf.usinfl.npos} din @{acf.usinfl.L}; logaritmul volatilității realizate S\\&P 500: @{acf.sp500.npos} din @{acf.sp500.L}')]),
    T('A low first autocorrelation with a long, slowly decaying tail is the signature of long memory; an AR(1) cannot produce it', 'O primă autocorelație moderată, urmată de o coadă lungă care scade lent, este semnătura memoriei lungi; un AR(1) nu o poate produce')])

chart(T('Hyperbolic against exponential decay', 'Descreștere hiperbolică și descreștere exponențială'), 'tsa_ch8_decay', 'TSA_ch8_arfima_processes', [
    T('Theoretical ACF of ARFIMA$(0,d,0)$ with $d = 0.4$ (next sections) and of an AR(1) with the same $\\rho(1) = d/(1-d) = @{dc.phi}$; right: both on log--log axes, with the power law $C k^{2d-1}$',
      'ACF teoretică a unui ARFIMA$(0,d,0)$ cu $d = 0{,}4$ (secțiunile următoare) și a unui AR(1) cu același $\\rho(1) = d/(1-d) = @{dc.phi}$; dreapta: ambele pe axe log--log, cu legea de putere $C k^{2d-1}$')],
    h='0.7\\textheight')

interp(('the two decay laws', 'celor două legi de descreștere'), [
    (T('Same start, different tails: at lag 10, @{dc.r10} against @{dc.a10}; at lag 50, @{dc.r50} against $@{dc.a50}$', 'Același început, cozi diferite: la lagul 10, @{dc.r10} față de @{dc.a10}; la lagul 50, @{dc.r50} față de $@{dc.a50}$'),
     [T('the long-memory ACF is still @{dc.r1000} at lag 1000', 'ACF cu memorie lungă este încă @{dc.r1000} la lagul 1000')]),
    (T('On log--log axes the hyperbolic ACF becomes a straight line of slope $2d - 1 = -0.2$; the exponential ACF bends down', 'Pe axe log--log ACF hiperbolică devine o dreaptă de pantă $2d - 1 = -0{,}2$; ACF exponențială se curbează în jos'),
     [T('the slope of a log--log plot is the idea behind every estimator of this chapter', 'panta unui grafic log--log este ideea din spatele tuturor estimatorilor din acest capitol')]),
    T('Sums of autocorrelations: AR(1) $\\sum_{k\\ge1}\\phi^k = @{dc.asum}$; ARFIMA: @{dc.sum100} up to lag 100 and @{dc.sum1000} up to lag 1000, still growing', 'Sumele autocorelațiilor: AR(1) $\\sum_{k\\ge1}\\phi^k = @{dc.asum}$; ARFIMA: @{dc.sum100} pînă la lagul 100 și @{dc.sum1000} pînă la lagul 1000, în creștere')])

D.recap(('Short and long memory', 'memorie scurtă și memorie lungă'), [
    T('Short memory: summable ACF, exponential decay, finite spectrum at zero (ARMA)', 'Memorie scurtă: ACF sumabilă, descreștere exponențială, spectru finit la zero (ARMA)'),
    T('Long memory: $\\rho(k) \\sim Ck^{2d-1}$, non-summable ACF, spectral pole $f(\\lambda) \\sim G\\lambda^{-2d}$', 'Memorie lungă: $\\rho(k) \\sim Ck^{2d-1}$, ACF nesumabilă, pol spectral $f(\\lambda) \\sim G\\lambda^{-2d}$'),
    T('Hurst and the Nile: ranges grow like $n^H$, $H > 1/2$', 'Hurst și Nilul: amplitudinile cresc ca $n^H$, cu $H > 1/2$'),
    T('Inflation and volatility show long, slowly decaying ACF tails', 'Inflația și volatilitatea au cozi lungi ale ACF, care scad lent')])

# =============================================================================
# 2. DIFERENȚIEREA FRACȚIONARĂ
# =============================================================================
D.section('Fractional differencing', 'Diferențierea fracționară')

D.frame(T('From integer to fractional differences', 'De la diferențe întregi la diferențe fracționare'), items(
    (T('Lag operator: $Lx_t = x_{t-1}$; first difference $(1-L)x_t = x_t - x_{t-1}$ (Chapter 3)', 'Operatorul de lag: $Lx_t = x_{t-1}$; prima diferență $(1-L)x_t = x_t - x_{t-1}$ (Capitolul 3)'),
     [T('$d = 0$: the level; $d = 1$: the first difference; $d = 2$: $x_t - 2x_{t-1} + x_{t-2}$', '$d = 0$: nivelul; $d = 1$: prima diferență; $d = 2$: $x_t - 2x_{t-1} + x_{t-2}$')]),
    (T('\\textbf{Fractional difference} for any real $d$ (binomial series): $(1-L)^d = \\sum_{k=0}^{\\infty}\\binom{d}{k}(-L)^k = \\sum_{k=0}^{\\infty}\\pi_k L^k$',
       '\\textbf{Diferența fracționară}, pentru orice $d$ real (seria binomială): $(1-L)^d = \\sum_{k=0}^{\\infty}\\binom{d}{k}(-L)^k = \\sum_{k=0}^{\\infty}\\pi_k L^k$'),
     [T('$(1-L)^d = 1 - dL - \\frac{d(1-d)}{2!}L^2 - \\frac{d(1-d)(2-d)}{3!}L^3 - \\dots$', '$(1-L)^d = 1 - dL - \\frac{d(1-d)}{2!}L^2 - \\frac{d(1-d)(2-d)}{3!}L^3 - \\dots$'),
      T('recursion: $\\pi_0 = 1$, $\\pi_k = \\pi_{k-1}\\,\\dfrac{k - 1 - d}{k}$; for large $k$: $\\pi_k \\approx -\\dfrac{d}{\\Gamma(1-d)}\\,k^{-1-d}$', 'recurența: $\\pi_0 = 1$, $\\pi_k = \\pi_{k-1}\\,\\dfrac{k - 1 - d}{k}$; pentru $k$ mare: $\\pi_k \\approx -\\dfrac{d}{\\Gamma(1-d)}\\,k^{-1-d}$'),
      T('$\\binom{d}{k} = d(d-1)\\cdots(d-k+1)/k!$: the binomial coefficient for real $d$; $\\pi_k$: the weight of $x_{t-k}$; $\\Gamma(\\cdot)$: the gamma function, $\\Gamma(n) = (n-1)!$ for integers', '$\\binom{d}{k} = d(d-1)\\cdots(d-k+1)/k!$: coeficientul binomial pentru $d$ real; $\\pi_k$: ponderea lui $x_{t-k}$; $\\Gamma(\\cdot)$: funcția gamma, $\\Gamma(n) = (n-1)!$ pentru numere întregi')]),
    T('A fractional difference uses \\textbf{all} past values, with weights that decay hyperbolically: it removes part of the memory, not all of it', 'O diferență fracționară folosește \\textbf{toate} valorile trecute, cu ponderi care scad hiperbolic: elimină o parte din memorie, nu toată memoria')))

D.frame(T('Worked example: the weights of $(1-L)^{0.4}$', 'Exemplu rezolvat: ponderile lui $(1-L)^{0{,}4}$'), items(
    (T('Recursion $\\pi_k = \\pi_{k-1}(k - 1 - d)/k$ with $d = 0.4$:', 'Recurența $\\pi_k = \\pi_{k-1}(k - 1 - d)/k$ cu $d = 0{,}4$:'),
     [T('$\\pi_1 = 1 \\cdot (0 - 0.4)/1 = @{w4.1}$', '$\\pi_1 = 1 \\cdot (0 - 0{,}4)/1 = @{w4.1}$'),
      T('$\\pi_2 = @{w4.1} \\cdot (1 - 0.4)/2 = @{w4.2}$', '$\\pi_2 = @{w4.1} \\cdot (1 - 0{,}4)/2 = @{w4.2}$'),
      T('$\\pi_3 = @{w4.2} \\cdot (2 - 0.4)/3 = @{w4.3}$; \\quad $\\pi_4 = @{w4.3} \\cdot 3.6/4 = @{w4.4}$; \\quad $\\pi_5 = @{w4.5}$', '$\\pi_3 = @{w4.2} \\cdot (2 - 0{,}4)/3 = @{w4.3}$; \\quad $\\pi_4 = @{w4.3} \\cdot 3{,}6/4 = @{w4.4}$; \\quad $\\pi_5 = @{w4.5}$')]),
    (T('So $(1-L)^{0.4}x_t = x_t - 0.4x_{t-1} - 0.12x_{t-2} - 0.064x_{t-3} - 0.0416x_{t-4} - \\dots$', 'Deci $(1-L)^{0{,}4}x_t = x_t - 0{,}4x_{t-1} - 0{,}12x_{t-2} - 0{,}064x_{t-3} - 0{,}0416x_{t-4} - \\dots$'),
     [T('the weights sum to $(1-1)^{0.4} = 0$ over all lags: a constant level is removed, as with $d = 1$', 'ponderile însumează $(1-1)^{0{,}4} = 0$ pe toate lagurile: un nivel constant este eliminat, ca la $d = 1$')]),
    T('Compare $d = 1$: $\\pi_1 = -1$ and all other weights are 0; $d = 0.7$: $\\pi_1 = @{wt.7.1}$, $\\pi_2 = @{wt.7.2}$', 'Comparați cu $d = 1$: $\\pi_1 = -1$, iar toate celelalte ponderi sînt 0; $d = 0{,}7$: $\\pi_1 = @{wt.7.1}$, $\\pi_2 = @{wt.7.2}$')))

chart(T('Fractional differencing and fractional integration', 'Diferențiere fracționară și integrare fracționară'), 'tsa_ch8_weights', 'TSA_ch8_fractional_differencing', [
    T('Left: weights $\\pi_k$ of $(1-L)^d$; right: the impulse responses $\\psi_k$ of $(1-L)^{-d}$, i.e.\\ the effect after $k$ periods of a unit shock to a fractionally integrated series, against an AR(1) with $\\phi = 0.9$',
      'Stînga: ponderile $\\pi_k$ ale lui $(1-L)^d$; dreapta: răspunsurile la impuls $\\psi_k$ ale lui $(1-L)^{-d}$, adică efectul după $k$ perioade al unui șoc unitar asupra unei serii integrate fracționar, comparate cu un AR(1) cu $\\phi = 0{,}9$')],
    h='0.7\\textheight')

interp(('the weights', 'ponderilor'), [
    (T('Fractional integration: $x_t = (1-L)^{-d}\\varepsilon_t = \\sum_k\\psi_k\\varepsilon_{t-k}$, with $\\psi_k = \\psi_{k-1}(k-1+d)/k \\approx k^{d-1}/\\Gamma(d)$', 'Integrarea fracționară: $x_t = (1-L)^{-d}\\varepsilon_t = \\sum_k\\psi_k\\varepsilon_{t-k}$, cu $\\psi_k = \\psi_{k-1}(k-1+d)/k \\approx k^{d-1}/\\Gamma(d)$'),
     [T('after 100 periods a shock still has weight @{wt.4.p100} ($d = 0.4$) and @{wt.7.p100} ($d = 0.7$); the AR(1) keeps $@{wt.ar100}$', 'după 100 de perioade un șoc are încă ponderea @{wt.4.p100} ($d = 0{,}4$) și @{wt.7.p100} ($d = 0{,}7$); AR(1) păstrează $@{wt.ar100}$')]),
    (T('Early on the AR(1) remembers more (@{wt.ar10} at lag 10, against @{wt.4.p10} for $d = 0.4$); later it forgets much faster', 'La început AR(1) își amintește mai mult (@{wt.ar10} la lagul 10, față de @{wt.4.p10} pentru $d = 0{,}4$); apoi uită mult mai repede'),
     [T('$d = 1$ (random walk): $\\psi_k = 1$ forever, the shock never fades', '$d = 1$ (mers aleator): $\\psi_k = 1$ mereu, șocul nu se stinge niciodată')]),
    T('For $0 < d < 1$ the responses tend to zero: the series is \\textbf{mean-reverting}, even when it is non-stationary ($d \\ge 0.5$)', 'Pentru $0 < d < 1$ răspunsurile tind spre zero: seria \\textbf{revine la medie}, chiar și cînd este nestaționară ($d \\ge 0{,}5$)')])

chart(T('How much differencing does a price need?', 'Gradul de diferențiere necesar pentru un preț'), 'tsa_ch8_ffd', 'TSA_ch8_fractional_differencing', [
    T('Log S\\&P 500, daily, @{ffd.n} days since 2000; $(1-L)^d$ with a fixed window (weights below $10^{-4}$ dropped), $d = 0, 0.05, \\dots, 1$; left: ADF statistic (Chapter 3); right: correlation with the log level',
      'Logaritmul S\\&P 500, zilnic, @{ffd.n} de zile din 2000; $(1-L)^d$ cu fereastră fixă (ponderile sub $10^{-4}$ sînt eliminate), $d = 0;\\ 0{,}05;\\ \\dots;\\ 1$; stînga: statistica ADF (Capitolul 3); dreapta: corelația cu nivelul logaritmic')],
    h='0.7\\textheight')

interp(('fractional differencing of a price', 'diferențierii fracționare a unui preț'), [
    (T('The log level ($d = 0$) has a unit root (ADF @{ffd.adf0}); the return ($d = 1$) is stationary (ADF @{ffd.adf1}) but has correlation @{ffd.cor1} with the level', 'Nivelul logaritmic ($d = 0$) are rădăcină unitară (ADF @{ffd.adf0}); randamentul ($d = 1$) este staționar (ADF @{ffd.adf1}), dar are corelația @{ffd.cor1} cu nivelul'),
     [T('the first difference throws away all the information about the level', 'prima diferență elimină toată informația despre nivel')]),
    (T('The ADF statistic crosses the 5\\% critical value (@{ffd.crit}) already at $d = @{ffd.dmin}$, where the correlation with the level is still @{ffd.cor}', 'Statistica ADF trece de valoarea critică de 5\\% (@{ffd.crit}) deja la $d = @{ffd.dmin}$, unde corelația cu nivelul este încă @{ffd.cor}'),
     [T('the fixed window then has @{ffd.w} weights', 'fereastra fixă are atunci @{ffd.w} de ponderi')]),
    T('A fractional difference can make a series stationary while keeping most of its memory: a tool used for forecasting features in machine learning (Chapter 9)', 'O diferență fracționară poate face o serie staționară păstrîndu-i cea mai mare parte a memoriei: un instrument folosit pentru variabilele explicative din machine learning (Capitolul 9)')])

D.recap(('Fractional differencing', 'diferențierea fracționară'), [
    T('$(1-L)^d = \\sum_k\\pi_k L^k$, $\\pi_k = \\pi_{k-1}(k-1-d)/k$: hyperbolically decaying weights', '$(1-L)^d = \\sum_k\\pi_k L^k$, $\\pi_k = \\pi_{k-1}(k-1-d)/k$: ponderi care scad hiperbolic'),
    T('$(1-L)^{-d}$: impulse responses $\\psi_k \\approx k^{d-1}/\\Gamma(d)$; mean reversion for $d < 1$', '$(1-L)^{-d}$: răspunsuri la impuls $\\psi_k \\approx k^{d-1}/\\Gamma(d)$; revenire la medie pentru $d < 1$'),
    T('$d$ between 0 and 1 fills the gap between levels and first differences', '$d$ între 0 și 1 umple golul dintre niveluri și primele diferențe')])

# =============================================================================
# 3. MODELUL ARFIMA
# =============================================================================
D.section('The ARFIMA$(p,d,q)$ model', 'Modelul ARFIMA$(p,d,q)$')

D.frame(T('The ARFIMA$(p,d,q)$ model', 'Modelul ARFIMA$(p,d,q)$'), items(
    (T('\\textbf{Definition}: $\\phi(L)(1-L)^d(x_t - \\mu) = \\theta(L)\\varepsilon_t$, $\\varepsilon_t$ white noise with variance $\\sigma^2$', '\\textbf{Definiție}: $\\phi(L)(1-L)^d(x_t - \\mu) = \\theta(L)\\varepsilon_t$, $\\varepsilon_t$ zgomot alb cu varianța $\\sigma^2$'),
     [T('$\\phi(L) = 1 - \\phi_1L - \\dots - \\phi_pL^p$ and $\\theta(L) = 1 + \\theta_1L + \\dots + \\theta_qL^q$, with roots outside the unit circle (Chapter 2)', '$\\phi(L) = 1 - \\phi_1L - \\dots - \\phi_pL^p$ și $\\theta(L) = 1 + \\theta_1L + \\dots + \\theta_qL^q$, cu rădăcinile în afara cercului unitate (Capitolul 2)'),
      T('$\\mu$: the mean of $x_t$; ARFIMA: autoregressive fractionally integrated moving average', '$\\mu$: media lui $x_t$; ARFIMA: autoregresiv, fracționar integrat, cu medie mobilă')]),
    (T('Two parts with two jobs', 'Două părți cu două roluri'),
     [T('$d$: the \\textbf{long-run} memory, the hyperbolic tail of the ACF', '$d$: memoria \\textbf{pe termen lung}, coada hiperbolică a ACF'),
      T('$\\phi$, $\\theta$: the \\textbf{short-run} dynamics, the first few autocorrelations', '$\\phi$, $\\theta$: dinamica \\textbf{pe termen scurt}, primele cîteva autocorelații')]),
    T('Special cases: $d = 0$: ARMA$(p,q)$; $d = 1$: ARIMA$(p,1,q)$ (Chapter 3); $d$ between them: fractional integration, $x_t \\sim I(d)$', 'Cazuri particulare: $d = 0$: ARMA$(p,q)$; $d = 1$: ARIMA$(p,1,q)$ (Capitolul 3); $d$ între ele: integrare fracționară, $x_t \\sim I(d)$')))

D.frame(T('Granger, Joyeux and Hosking (1980--1981)', 'Granger, Joyeux și Hosking (1980--1981)'), two(
    ph('granger', T('Clive W. J. Granger (1934--2009), Nobel Prize 2003', 'Clive W. J. Granger (1934--2009), Premiul Nobel 2003'), h='0.48\\textheight'),
    items((T('\\refGJ\\ and \\refHosking\\ introduced fractional differencing independently', '\\refGJ\\ și \\refHosking\\ au introdus independent diferențierea fracționară'),
           [T('Granger and Joyeux: aggregating many AR(1) series with different persistence produces long memory', 'Granger și Joyeux: agregarea multor serii AR(1) cu persistențe diferite produce memorie lungă'),
            T('Hosking: the ACF, the stationarity and invertibility conditions, and the ARFIMA$(p,d,q)$ family', 'Hosking: ACF, condițiile de staționaritate și de inversabilitate și familia ARFIMA$(p,d,q)$')]),
          T('Their model connects the Hurst exponent of hydrology with the Box--Jenkins models of econometrics', 'Modelul lor leagă exponentul Hurst din hidrologie de modelele Box--Jenkins din econometrie'),
          T('Applications followed in inflation \\refHW, output, interest rates and volatility \\refDGE', 'Au urmat aplicații pentru inflație \\refHW, producție, dobînzi și volatilitate \\refDGE'))))

D.frame(T('The parameter $d$: what each range means', 'Parametrul $d$: semnificația fiecărui interval'), table(
    'lll', T('\\textbf{Range of $d$}', '\\textbf{Intervalul lui $d$}') + ' & ' + T('\\textbf{Properties}', '\\textbf{Proprietăți}') + ' & ' + T('\\textbf{ACF / shocks}', '\\textbf{ACF / șocuri}'),
    [T('$d \\le -0.5$', '$d \\le -0{,}5$') + ' & ' + T('not invertible (no AR($\\infty$) form)', 'neinversabil (fără formă AR($\\infty$))') + ' & --',
     T('$-0.5 < d < 0$', '$-0{,}5 < d < 0$') + ' & ' + T('stationary, invertible, anti-persistent', 'staționar, inversabil, antipersistent') + ' & ' + T('negative, summable to $-1/2$', 'negativă, cu suma $-1/2$'),
     '$d = 0$ & ' + T('ARMA, short memory', 'ARMA, memorie scurtă') + ' & ' + T('exponential decay', 'descreștere exponențială'),
     T('$0 < d < 0.5$', '$0 < d < 0{,}5$') + ' & ' + T('stationary, long memory', 'staționar, memorie lungă') + ' & ' + T('hyperbolic, not summable', 'hiperbolică, nesumabilă'),
     T('$0.5 \\le d < 1$', '$0{,}5 \\le d < 1$') + ' & ' + T('non-stationary, mean-reverting', 'nestaționar, cu revenire la medie') + ' & ' + T('shocks fade, slowly', 'șocurile se sting, lent'),
     '$d = 1$ & ' + T('unit root, ARIMA', 'rădăcină unitară, ARIMA') + ' & ' + T('shocks are permanent', 'șocurile sînt permanente')],
    size='footnotesize') + items(
    T('Stationarity needs $d < 1/2$, invertibility needs $d > -1/2$ (\\refHosking); a non-stationary series with $d < 1.5$ is handled by differencing once: $(1-L)x_t \\sim I(d-1)$',
      'Staționaritatea cere $d < 1/2$, inversabilitatea cere $d > -1/2$ (\\refHosking); o serie nestaționară cu $d < 1{,}5$ se tratează diferențiind o dată: $(1-L)x_t \\sim I(d-1)$'),
    T('Hurst exponent of the stationary increments: $H = d + 1/2$; $H > 1/2$ persistence, $H < 1/2$ anti-persistence', 'Exponentul Hurst al incrementelor staționare: $H = d + 1/2$; $H > 1/2$ persistență, $H < 1/2$ antipersistență')))

D.frame(T('Autocorrelations of ARFIMA$(0,d,0)$', 'Autocorelațiile unui ARFIMA$(0,d,0)$'), items(
    (T('For $x_t = (1-L)^{-d}\\varepsilon_t$, $-1/2 < d < 1/2$ (\\refHosking):', 'Pentru $x_t = (1-L)^{-d}\\varepsilon_t$, $-1/2 < d < 1/2$ (\\refHosking):'),
     [T('variance $\\gamma(0) = \\sigma^2\\,\\dfrac{\\Gamma(1-2d)}{\\Gamma(1-d)^2}$', 'varianța $\\gamma(0) = \\sigma^2\\,\\dfrac{\\Gamma(1-2d)}{\\Gamma(1-d)^2}$'),
      T('autocorrelations $\\rho(k) = \\dfrac{\\Gamma(1-d)\\,\\Gamma(k+d)}{\\Gamma(d)\\,\\Gamma(k+1-d)}$, computed with $\\rho(k) = \\rho(k-1)\\,\\dfrac{k-1+d}{k-d}$', 'autocorelațiile $\\rho(k) = \\dfrac{\\Gamma(1-d)\\,\\Gamma(k+d)}{\\Gamma(d)\\,\\Gamma(k+1-d)}$, calculate cu $\\rho(k) = \\rho(k-1)\\,\\dfrac{k-1+d}{k-d}$')]),
    (T('First lags: $\\rho(1) = \\dfrac{d}{1-d}$, \\quad $\\rho(2) = \\dfrac{d(1+d)}{(1-d)(2-d)}$', 'Primele laguri: $\\rho(1) = \\dfrac{d}{1-d}$, \\quad $\\rho(2) = \\dfrac{d(1+d)}{(1-d)(2-d)}$'),
     [T('large $k$: $\\rho(k) \\approx \\dfrac{\\Gamma(1-d)}{\\Gamma(d)}\\,k^{2d-1}$, the power law of Section 1 with $C = \\Gamma(1-d)/\\Gamma(d)$', '$k$ mare: $\\rho(k) \\approx \\dfrac{\\Gamma(1-d)}{\\Gamma(d)}\\,k^{2d-1}$, legea de putere din secțiunea 1, cu $C = \\Gamma(1-d)/\\Gamma(d)$')]),
    T('Spectral density ($i$: the imaginary unit): $f(\\lambda) = \\dfrac{\\sigma^2}{2\\pi}\\,|1 - e^{-i\\lambda}|^{-2d} = \\dfrac{\\sigma^2}{2\\pi}\\,\\bigl(2\\sin(\\lambda/2)\\bigr)^{-2d} \\approx \\dfrac{\\sigma^2}{2\\pi}\\lambda^{-2d}$ near 0',
      'Densitatea spectrală ($i$: unitatea imaginară): $f(\\lambda) = \\dfrac{\\sigma^2}{2\\pi}\\,|1 - e^{-i\\lambda}|^{-2d} = \\dfrac{\\sigma^2}{2\\pi}\\,\\bigl(2\\sin(\\lambda/2)\\bigr)^{-2d} \\approx \\dfrac{\\sigma^2}{2\\pi}\\lambda^{-2d}$ în apropierea lui 0')))

D.frame(T('Worked example: ARFIMA$(0, 0.3, 0)$', 'Exemplu rezolvat: ARFIMA$(0;\\ 0{,}3;\\ 0)$'), items(
    (T('$\\rho(1) = 0.3/0.7 = @{r3.1}$; \\quad $\\rho(2) = @{r3.1} \\cdot 1.3/1.7 = @{r3.2}$; \\quad $\\rho(3) = @{r3.2} \\cdot 2.3/2.7 = @{r3.3}$', '$\\rho(1) = 0{,}3/0{,}7 = @{r3.1}$; \\quad $\\rho(2) = @{r3.1} \\cdot 1{,}3/1{,}7 = @{r3.2}$; \\quad $\\rho(3) = @{r3.2} \\cdot 2{,}3/2{,}7 = @{r3.3}$'),
     [T('continuing the recursion: $\\rho(5) = @{r3.5}$, $\\rho(10) = @{r3.10}$', 'continuînd recurența: $\\rho(5) = @{r3.5}$, $\\rho(10) = @{r3.10}$')]),
    (T('Variance with $\\sigma^2 = 1$: $\\Gamma(0.4)/\\Gamma(0.7)^2 = @{v03}$, larger than the variance of the shocks', 'Varianța pentru $\\sigma^2 = 1$: $\\Gamma(0{,}4)/\\Gamma(0{,}7)^2 = @{v03}$, mai mare decît varianța șocurilor'),
     [T('the memory accumulates past shocks', 'memoria acumulează șocurile trecute')]),
    T('An AR(1) with the same $\\rho(1) = @{r3.1}$ has $\\rho(10) = @{r3.1}^{10} = @{arsame}$: three orders of magnitude below @{r3.10}', 'Un AR(1) cu același $\\rho(1) = @{r3.1}$ are $\\rho(10) = @{r3.1}^{10} = @{arsame}$: cu trei ordine de mărime sub @{r3.10}'),
    T('$H = d + 0.5 = 0.8$: a persistent series', '$H = d + 0{,}5 = 0{,}8$: o serie persistentă')))

chart(T('ARFIMA$(0,d,0)$ with $d = 0.4$ and $d = -0.4$', 'ARFIMA$(0,d,0)$ cu $d = 0{,}4$ și $d = -0{,}4$'), 'tsa_ch8_arfima_paths', 'TSA_ch8_arfima_processes', [
    T('Simulated paths (1000 observations, exact simulation by circulant embedding) with $\\varepsilon_t \\sim N(0, 1)$; right: sample ACF of 20\\,000 observations and the theoretical ACF',
      'Traiectorii simulate (1000 de observații, simulare exactă prin scufundare circulantă) cu $\\varepsilon_t \\sim N(0, 1)$; dreapta: ACF de selecție pentru 20\\,000 de observații și ACF teoretică')],
    h='0.7\\textheight')

interp(('the two ARFIMA paths', 'celor două traiectorii ARFIMA'), [
    (T('$d = 0.4$: long excursions above and below zero; $\\rho(1) = @{pa.p.r1}$, $\\rho(10) = @{pa.p.r10}$, $\\rho(50) = @{pa.p.r50}$; variance @{pa.p.v}', '$d = 0{,}4$: excursii lungi deasupra și sub zero; $\\rho(1) = @{pa.p.r1}$, $\\rho(10) = @{pa.p.r10}$, $\\rho(50) = @{pa.p.r50}$; varianța @{pa.p.v}'),
     [T('the path looks as if it had trends, although the mean is constant: a reason why long memory is mistaken for non-stationarity', 'traiectoria pare să aibă trenduri, deși media este constantă: un motiv pentru care memoria lungă este confundată cu nestaționaritatea')]),
    (T('$d = -0.4$: rapid zig-zag; $\\rho(1) = @{pa.n.r1}$, $\\rho(2) = @{pa.n.r2}$; variance @{pa.n.v}', '$d = -0{,}4$: zig-zag rapid; $\\rho(1) = @{pa.n.r1}$, $\\rho(2) = @{pa.n.r2}$; varianța @{pa.n.v}'),
     [T('anti-persistence: a rise tends to be followed by a fall; typical of over-differenced series, e.g.\\ the difference of a series with $d = 0.6$', 'antipersistență: o creștere tinde să fie urmată de o scădere; tipică seriilor diferențiate excesiv, de exemplu diferența unei serii cu $d = 0{,}6$')])])

D.frame(T('Mandelbrot: fractional Brownian motion and $H$', 'Mandelbrot: mișcarea browniană fracționară și $H$'), two(
    ph('mandelbrot', T('Benoît Mandelbrot (1924--2010)', 'Benoît Mandelbrot (1924--2010)'), h='0.42\\textheight'),
    items((T('\\textbf{Fractional Brownian motion} $B_H(t)$, $0 < H < 1$ \\refMVN: Gaussian, $B_H(0) = 0$, $\\Var(B_H(t)) = t^{2H}$', '\\textbf{Mișcarea browniană fracționară} $B_H(t)$, $0 < H < 1$ \\refMVN: gaussiană, $B_H(0) = 0$, $\\Var(B_H(t)) = t^{2H}$'),
           [T('self-similar: $B_H(ct)$ has the same distribution as $c^H B_H(t)$; $H = 1/2$: Brownian motion', 'autosimilară: $B_H(ct)$ are aceeași distribuție ca $c^H B_H(t)$; $H = 1/2$: mișcarea browniană')]),
          (T('Its increments, \\textbf{fractional Gaussian noise} (fGn), are stationary with $\\rho(k) = \\tfrac12(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})$', 'Incrementele ei, \\textbf{zgomotul gaussian fracționar} (fGn), sînt staționare, cu $\\rho(k) = \\tfrac12(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})$'),
           [T('$\\rho(k) \\approx H(2H-1)k^{2H-2}$: the same tail as ARFIMA with $d = H - 1/2$', '$\\rho(k) \\approx H(2H-1)k^{2H-2}$: aceeași coadă ca ARFIMA cu $d = H - 1/2$')]),
          T('Mandelbrot brought Hurst\'s finding into statistics: fGn is the continuous-time cousin of ARFIMA$(0,d,0)$', 'Mandelbrot a adus rezultatul lui Hurst în statistică: fGn este varianta în timp continuu a unui ARFIMA$(0,d,0)$'))))

chart(T('Fractional Brownian motion for three Hurst exponents', 'Mișcarea browniană fracționară pentru trei exponenți Hurst'), 'tsa_ch8_fbm', 'TSA_ch8_arfima_processes', [
    T('The same random numbers for $H = 0.3$, $0.5$ and $0.7$ (circulant embedding); right: the first 150 increments', 'Aceleași numere aleatoare pentru $H = 0{,}3$, $0{,}5$ și $0{,}7$ (scufundare circulantă); dreapta: primele 150 de incremente')],
    h='0.7\\textheight')

interp(('the fBm paths', 'traiectoriilor fBm'), [
    (T('$H = 0.7$: smooth, trending path; lag-1 autocorrelation of the increments $2^{2H-1} - 1 = @{fb.7}$ (sample @{fb.7s})', '$H = 0{,}7$: traiectorie netedă, cu aparență de trend; autocorelația de ordinul 1 a incrementelor $2^{2H-1} - 1 = @{fb.7}$ (în eșantion @{fb.7s})'),
     [T('$H = 0.3$: rough path that keeps turning back; $\\rho(1) = @{fb.3}$ (sample @{fb.3s})', '$H = 0{,}3$: traiectorie aspră, care se întoarce mereu; $\\rho(1) = @{fb.3}$ (în eșantion @{fb.3s})')]),
    T('The range of the path grows like $t^H$: the quantity Hurst measured on the Nile', 'Amplitudinea traiectoriei crește ca $t^H$: mărimea pe care a măsurat-o Hurst pe Nil'),
    T('In time-series modelling we prefer ARFIMA: it adds the short-run ARMA part that fGn lacks', 'În modelarea seriilor de timp preferăm ARFIMA: adaugă partea ARMA pe termen scurt, care lipsește din fGn')])

D.recap(('The ARFIMA model', 'modelul ARFIMA'), [
    T('$\\phi(L)(1-L)^d(x_t - \\mu) = \\theta(L)\\varepsilon_t$: $d$ for the long run, ARMA for the short run', '$\\phi(L)(1-L)^d(x_t - \\mu) = \\theta(L)\\varepsilon_t$: $d$ pentru termenul lung, ARMA pentru termenul scurt'),
    T('Stationary for $d < 1/2$, invertible for $d > -1/2$; mean-reverting for $d < 1$', 'Staționar pentru $d < 1/2$, inversabil pentru $d > -1/2$; cu revenire la medie pentru $d < 1$'),
    T('ARFIMA$(0,d,0)$: $\\rho(1) = d/(1-d)$, $\\rho(k) \\sim Ck^{2d-1}$; $H = d + 1/2$', 'ARFIMA$(0,d,0)$: $\\rho(1) = d/(1-d)$, $\\rho(k) \\sim Ck^{2d-1}$; $H = d + 1/2$'),
    T('fBm and fGn: the same long memory in Mandelbrot\'s continuous-time form', 'fBm și fGn: aceeași memorie lungă, în forma în timp continuu a lui Mandelbrot')])

# =============================================================================
# 4. ESTIMARE
# =============================================================================
D.section('Estimating $d$', 'Estimarea lui $d$')

TBc = '>{\\raggedright\\arraybackslash}'
D.frame(T('Five estimators', 'Cinci estimatori'), table(
    TBc + 'p{2.3cm}' + TBc + 'p{3.1cm}' + TBc + 'p{3.6cm}' + TBc + 'p{1.9cm}',
    T('\\textbf{Estimator}', '\\textbf{Estimatorul}') + ' & ' + T('\\textbf{Uses}', '\\textbf{Folosește}') + ' & ' + T('\\textbf{Estimate}', '\\textbf{Estimarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}'),
    ['R/S \\refHurst & ' + T('range of partial sums in blocks', 'amplitudinea sumelor parțiale pe blocuri') + ' & ' + T('slope of $\\log(R/S)_n$ on $\\log n$ = $H$', 'panta lui $\\log(R/S)_n$ față de $\\log n$ = $H$') + ' & ' + T('heuristic', 'euristic'),
     'DFA \\refPeng & ' + T('detrended fluctuations of the profile', 'fluctuațiile fără tendință ale profilului') + ' & ' + T('slope of $\\log F(n)$ on $\\log n$ = $H$', 'panta lui $\\log F(n)$ față de $\\log n$ = $H$') + ' & ' + T('heuristic', 'euristic'),
     'GPH \\refGPH & ' + T('lowest $m$ periodogram ordinates', 'primele $m$ valori ale periodogramei') + ' & ' + T('OLS slope of a log-periodogram regression', 'panta OLS a unei regresii pe log-periodogramă') + ' & ' + T('semiparametric', 'semiparametric'),
     T('Local Whittle', 'Whittle local') + ' \\refRob & ' + T('lowest $m$ periodogram ordinates', 'primele $m$ valori ale periodogramei') + ' & ' + T('maximises a local Gaussian likelihood', 'maximizează o verosimilitate gaussiană locală') + ' & ' + T('semiparametric', 'semiparametric'),
     T('Exact ML, Whittle', 'ML exactă, Whittle') + ' \\refSowell, \\refFT & ' + T('the whole series', 'toată seria') + ' & ' + T('$d$, $\\phi$, $\\theta$, $\\sigma^2$ jointly', '$d$, $\\phi$, $\\theta$, $\\sigma^2$ împreună') + ' & ' + T('parametric', 'parametric')],
    size='scriptsize') + items(
    T('Semiparametric: only the behaviour near frequency zero is modelled, the short-run part is left free; parametric: the whole ARFIMA model must be right', 'Semiparametric: se modelează doar comportamentul din apropierea frecvenței zero, partea pe termen scurt rămîne liberă; parametric: întregul model ARFIMA trebuie să fie corect'),
    T('ML: maximum likelihood; OLS: ordinary least squares; $d = H - 1/2$ converts between the two scales', 'ML: verosimilitate maximă; OLS: metoda celor mai mici pătrate; $d = H - 1/2$ face trecerea între cele două scale')))

D.frame(T('The R/S statistic, step by step', 'Statistica R/S, pas cu pas'), items(
    (T('Block $x_1, \\dots, x_n$: mean $\\bar x$, cumulative deviations $Y_k = \\sum_{i\\le k}(x_i - \\bar x)$', 'Blocul $x_1, \\dots, x_n$: media $\\bar x$, abaterile cumulate $Y_k = \\sum_{i\\le k}(x_i - \\bar x)$'),
     [T('range $R = \\max_k Y_k - \\min_k Y_k$, $S$: the standard deviation of the block; rescaled range $R/S$', 'amplitudinea $R = \\max_k Y_k - \\min_k Y_k$, $S$: abaterea standard a blocului; amplitudinea rescalată $R/S$')]),
    (T('Example: $x = (0.5, -0.3, 0.8, -0.2, 0.6, -0.1, 0.4, -0.7)$, $\\bar x = @{rs.mean}$', 'Exemplu: $x = (0{,}5;\\ -0{,}3;\\ 0{,}8;\\ -0{,}2;\\ 0{,}6;\\ -0{,}1;\\ 0{,}4;\\ -0{,}7)$, $\\bar x = @{rs.mean}$'),
     [T('$Y_k$: 0.375, $-$0.050, 0.625, 0.300, 0.775, 0.550, 0.825, 0; $R = @{rs.max} - (@{rs.min}) = @{rs.R}$', '$Y_k$: 0,375; $-$0,050; 0,625; 0,300; 0,775; 0,550; 0,825; 0; $R = @{rs.max} - (@{rs.min}) = @{rs.R}$'),
      T('$S = @{rs.S}$, so $R/S = @{rs.RS}$', '$S = @{rs.S}$, deci $R/S = @{rs.RS}$')]),
    (T('\\textbf{Hurst exponent}: average $(R/S)_n$ over blocks of size $n$; $E(R/S)_n \\approx c\\,n^H$, so $H$ is the slope of $\\log(R/S)_n$ on $\\log n$', '\\textbf{Exponentul Hurst}: media $(R/S)_n$ pe blocuri de mărime $n$; $E(R/S)_n \\approx c\\,n^H$, deci $H$ este panta lui $\\log(R/S)_n$ față de $\\log n$'),
     [T('pitfalls: biased upwards in small blocks \\refMW; short memory also raises it; \\refLo\\ corrects $S$ with autocovariances (modified R/S)', 'capcane: deplasat în sus pe blocuri mici \\refMW; și memoria scurtă îl crește; \\refLo\\ corectează $S$ cu autocovarianțe (R/S modificat)'),
      T('\\textbf{DFA} \\refPeng: fit a line to the profile $Y_k$ in each block; $F(n)$, the RMS (root mean square) deviation, grows like $n^H$', '\\textbf{DFA} \\refPeng: ajustăm o dreaptă profilului $Y_k$ în fiecare bloc; $F(n)$, abaterea medie pătratică (RMS), crește ca $n^H$')])), size='footnotesize')

chart(T('R/S and DFA on three series', 'R/S și DFA pentru trei serii'), 'tsa_ch8_rs_dfa', 'TSA_ch8_hurst_rs_dfa', [
    T('US monthly inflation (FRED CPIAUCSL, 1947--2026), daily S\\&P 500 returns $r_t$ and absolute returns $|r_t|$ (2000--2026); both axes logarithmic, curves normalised to 1 at $n = 10$',
      'Inflația lunară din SUA (FRED CPIAUCSL, 1947--2026), randamentele zilnice S\\&P 500 $r_t$ și randamentele absolute $|r_t|$ (2000--2026); ambele axe logaritmice, curbele normalizate la 1 pentru $n = 10$')],
    h='0.7\\textheight')

interp(('the R/S and DFA slopes', 'pantelor R/S și DFA'), [
    (T('S\\&P 500 returns: slopes @{rd.r.rs} (R/S) and @{rd.r.dfa} (DFA), close to 0.5: no memory in returns', 'Randamentele S\\&P 500: pantele @{rd.r.rs} (R/S) și @{rd.r.dfa} (DFA), apropiate de 0,5: fără memorie în randamente'),
     [T('the weak efficiency of Chapter 3 holds for the level of returns', 'eficiența slabă din Capitolul 3 este valabilă pentru randamente')]),
    (T('Absolute returns: @{rd.a.rs} and @{rd.a.dfa}; US inflation: @{rd.ui.rs} and @{rd.ui.dfa}', 'Randamentele absolute: @{rd.a.rs} și @{rd.a.dfa}; inflația din SUA: @{rd.ui.rs} și @{rd.ui.dfa}'),
     [T('both well above 0.5: persistent; a DFA slope near 1 hints at $d$ close to 0.5, the edge of stationarity', 'ambele mult peste 0,5: persistente; o pantă DFA apropiată de 1 sugerează un $d$ apropiat de 0,5, limita staționarității')]),
    T('These are graphical tools: they give no standard error and react to short memory and trends; the next estimators are built for inference', 'Acestea sînt instrumente grafice: nu dau o eroare standard și reacționează la memoria scurtă și la trenduri; estimatorii următori sînt construiți pentru inferență')])

D.frame(T('The periodogram and the GPH estimator', 'Periodograma și estimatorul GPH'), items(
    (T('\\textbf{Periodogram} at the Fourier frequencies $\\lambda_j = 2\\pi j/T$: $I(\\lambda_j) = \\frac{1}{2\\pi T}\\bigl|\\sum_t (x_t - \\bar x)e^{-i\\lambda_j t}\\bigr|^2$', '\\textbf{Periodograma} la frecvențele Fourier $\\lambda_j = 2\\pi j/T$: $I(\\lambda_j) = \\frac{1}{2\\pi T}\\bigl|\\sum_t (x_t - \\bar x)e^{-i\\lambda_j t}\\bigr|^2$'),
     [T('$i$: the imaginary unit; $|\\cdot|$: the modulus of a complex number; $j = 1, \\dots, \\lfloor T/2\\rfloor$', '$i$: unitatea imaginară; $|\\cdot|$: modulul unui număr complex; $j = 1, \\dots, \\lfloor T/2\\rfloor$'),
      T('a noisy estimate of the spectral density $f(\\lambda_j)$; $\\lambda_j$ near 0 = slow cycles of period $T/j$', 'o estimare zgomotoasă a densității spectrale $f(\\lambda_j)$; $\\lambda_j$ apropiat de 0 = cicluri lente, de perioadă $T/j$')]),
    (T('Near zero, $f(\\lambda) \\approx G\\,(4\\sin^2(\\lambda/2))^{-d}$; taking logs:', 'În apropierea lui zero, $f(\\lambda) \\approx G\\,(4\\sin^2(\\lambda/2))^{-d}$; logaritmînd:'),
     [T('\\textbf{GPH regression} \\refGPH: $\\log I(\\lambda_j) = c + d\\,\\bigl[-\\log(4\\sin^2(\\lambda_j/2))\\bigr] + u_j$, $j = 1, \\dots, m$', '\\textbf{Regresia GPH} \\refGPH: $\\log I(\\lambda_j) = c + d\\,\\bigl[-\\log(4\\sin^2(\\lambda_j/2))\\bigr] + u_j$, $j = 1, \\dots, m$'),
      T('$c$: constant; $u_j$: regression error; $\\hat d$ = OLS slope; standard error $\\pi/\\sqrt{24m}$', '$c$: termenul liber; $u_j$: eroarea regresiei; $\\hat d$ = panta OLS; eroarea standard $\\pi/\\sqrt{24m}$')]),
    (T('\\textbf{Bandwidth} $m$: how many low frequencies we use; here $m = \\lfloor T^{0.65}\\rfloor$', '\\textbf{Lățimea de bandă} $m$: cîte frecvențe joase folosim; aici $m = \\lfloor T^{0{,}65}\\rfloor$'),
     [T('small $m$: little bias from the short-run part, large variance; large $m$: small variance, but short memory leaks into $\\hat d$', '$m$ mic: puțină deplasare din partea pe termen scurt, varianță mare; $m$ mare: varianță mică, dar memoria scurtă se strecoară în $\\hat d$')])))

chart(T('GPH regressions: the Nile and Romanian inflation', 'Regresii GPH: Nilul și inflația din România'), 'tsa_ch8_gph', 'TSA_ch8_estimation', [
    T('Dots: log periodogram at the $m$ lowest Fourier frequencies; line: OLS fit. Nile: $T = @{g.nile.T}$ years; Romanian monthly inflation (SA): $T = @{g.infl.T}$ months, @{im.last} the last',
      'Puncte: logaritmul periodogramei la cele mai joase $m$ frecvențe Fourier; linia: ajustarea OLS. Nilul: $T = @{g.nile.T}$ de ani; inflația lunară din România (SA): $T = @{g.infl.T}$ de luni, ultima @{im.last}')],
    h='0.7\\textheight')

interp(('the GPH regressions', 'regresiilor GPH'), [
    (T('Nile: $\\hat d = @{g.nile.d}$ (SE @{g.nile.se}), $m = @{g.nile.m}$ frequencies, i.e.\\ cycles longer than @{g.nile.per} years', 'Nilul: $\\hat d = @{g.nile.d}$ (SE @{g.nile.se}), $m = @{g.nile.m}$ frecvențe, adică cicluri mai lungi de @{g.nile.per} ani'),
     [T('local Whittle (next slide): @{g.nile.lw} (SE @{g.nile.lwse}); both significantly above 0, inside the stationary range', 'Whittle local (slide-ul următor): @{g.nile.lw} (SE @{g.nile.lwse}); ambele semnificativ peste 0, în intervalul staționar')]),
    (T('Romanian inflation: $\\hat d = @{g.infl.d}$ (SE @{g.infl.se}); local Whittle @{g.infl.lw} (SE @{g.infl.lwse})', 'Inflația din România: $\\hat d = @{g.infl.d}$ (SE @{g.infl.se}); Whittle local @{g.infl.lw} (SE @{g.infl.lwse})'),
     [T('cycles longer than @{g.infl.per} months: a shock to monthly inflation fades over years, not months', 'cicluri mai lungi de @{g.infl.per} luni: un șoc al inflației lunare se stinge în ani, nu în luni')]),
    T('The cloud is wide: each periodogram ordinate is roughly an exponential variable around $f(\\lambda_j)$; the slope needs many points', 'Norul de puncte este larg: fiecare valoare a periodogramei este aproximativ o variabilă exponențială în jurul lui $f(\\lambda_j)$; panta are nevoie de multe puncte')])

D.frame(T('The local Whittle estimator', 'Estimatorul Whittle local'), items(
    (T('\\refRob: assume $f(\\lambda) \\approx G\\lambda^{-2d}$ for $\\lambda_1, \\dots, \\lambda_m$ and maximise the Gaussian (Whittle) likelihood of these ordinates', '\\refRob: presupunem $f(\\lambda) \\approx G\\lambda^{-2d}$ pentru $\\lambda_1, \\dots, \\lambda_m$ și maximizăm verosimilitatea gaussiană (Whittle) a acestor valori'),
     [T('after concentrating out the constant $G$ of $f(\\lambda) \\approx G\\lambda^{-2d}$: $\\hat d = \\arg\\min_d\\ \\log\\Bigl(\\frac1m\\sum_{j=1}^m \\lambda_j^{2d}I(\\lambda_j)\\Bigr) - \\frac{2d}{m}\\sum_{j=1}^m\\log\\lambda_j$', 'după eliminarea constantei $G$ din $f(\\lambda) \\approx G\\lambda^{-2d}$: $\\hat d = \\arg\\min_d\\ \\log\\Bigl(\\frac1m\\sum_{j=1}^m \\lambda_j^{2d}I(\\lambda_j)\\Bigr) - \\frac{2d}{m}\\sum_{j=1}^m\\log\\lambda_j$')]),
    (T('Standard error $1/(2\\sqrt m)$, smaller than GPH ($\\pi/\\sqrt{24m} \\approx 0.64/\\sqrt m$)', 'Eroarea standard $1/(2\\sqrt m)$, mai mică decît la GPH ($\\pi/\\sqrt{24m} \\approx 0{,}64/\\sqrt m$)'),
     [T('valid also for non-stationary series with $d$ up to about 1 \\refVelasco', 'valabil și pentru serii nestaționare cu $d$ pînă la aproximativ 1 \\refVelasco')]),
    T('Both GPH and local Whittle depend on $m$: always report $\\hat d$ for several bandwidths', 'Atît GPH, cît și Whittle local depind de $m$: raportați întotdeauna $\\hat d$ pentru mai multe lățimi de bandă')))

chart(T('Sensitivity to the bandwidth', 'Sensibilitatea la lățimea de bandă'), 'tsa_ch8_bandwidth', 'TSA_ch8_estimation', [
    T('$\\hat d$ for $m = \\lfloor T^a\\rfloor$, $a = 0.40, \\dots, 0.80$; shaded: 95\\% interval of local Whittle. Romanian monthly inflation ($T = @{bw.infl.T}$, $m$ from @{bw.infl.m0} to @{bw.infl.m1}) and US monthly inflation ($T = @{bw.usinfl.T}$)',
      '$\\hat d$ pentru $m = \\lfloor T^a\\rfloor$, $a = 0{,}40;\\ \\dots;\\ 0{,}80$; zona colorată: intervalul de 95\\% al estimatorului Whittle local. Inflația lunară din România ($T = @{bw.infl.T}$, $m$ de la @{bw.infl.m0} la @{bw.infl.m1}) și inflația lunară din SUA ($T = @{bw.usinfl.T}$)')],
    h='0.7\\textheight')

interp(('the bandwidth plot', 'graficului lățimii de bandă'), [
    (T('Romania: local Whittle between @{bw.infl.lo} and @{bw.infl.hi}, GPH between @{bw.infl.g.lo} and @{bw.infl.g.hi}', 'România: Whittle local între @{bw.infl.lo} și @{bw.infl.hi}, GPH între @{bw.infl.g.lo} și @{bw.infl.g.hi}'),
     [T('with few frequencies the estimate is near 0.5; with more it falls to about 0.3: the answer depends on the choice of $m$', 'cu puține frecvențe estimarea este aproape de 0,5; cu mai multe scade spre 0,3: răspunsul depinde de alegerea lui $m$')]),
    (T('United States: local Whittle between @{bw.usinfl.lo} and @{bw.usinfl.hi}, always well above 0', 'Statele Unite: Whittle local între @{bw.usinfl.lo} și @{bw.usinfl.hi}, mereu mult peste 0'),
     [T('the long sample (1947--2026) gives narrow intervals; the level of $d$ near 0.5 hints at regime changes (Section 7)', 'eșantionul lung (1947--2026) dă intervale înguste; un $d$ apropiat de 0,5 sugerează schimbări de regim (secțiunea 7)')]),
    T('Conclusion: inflation has long memory in both countries; a single number for $d$ hides an uncertainty of about $\\pm 0.15$', 'Concluzie: inflația are memorie lungă în ambele țări; un singur număr pentru $d$ ascunde o incertitudine de aproximativ $\\pm 0{,}15$')])

D.frame(T('Maximum likelihood for ARFIMA$(p,d,q)$', 'Verosimilitatea maximă pentru ARFIMA$(p,d,q)$'), items(
    (T('\\textbf{Exact ML} \\refSowell: for Gaussian $x = (x_1, \\dots, x_T)\'$ with covariance matrix $\\Gamma(\\vartheta)$, $\\vartheta = (d, \\phi, \\theta, \\sigma^2)$:', '\\textbf{ML exactă} \\refSowell: pentru $x = (x_1, \\dots, x_T)\'$ gaussian, cu matricea de covarianță $\\Gamma(\\vartheta)$, $\\vartheta = (d, \\phi, \\theta, \\sigma^2)$:'),
     [T('$\\ell(\\vartheta) = -\\frac{T}{2}\\log 2\\pi - \\frac12\\log\\det\\Gamma(\\vartheta) - \\frac12 (x - \\mu)\'\\Gamma(\\vartheta)^{-1}(x - \\mu)$', '$\\ell(\\vartheta) = -\\frac{T}{2}\\log 2\\pi - \\frac12\\log\\det\\Gamma(\\vartheta) - \\frac12 (x - \\mu)\'\\Gamma(\\vartheta)^{-1}(x - \\mu)$'),
      T('$\\ell$: the log-likelihood; $\\mu$: the vector of means; $\\Gamma(\\vartheta)$: the $T \\times T$ matrix of the ARFIMA autocovariances', '$\\ell$: logaritmul verosimilității; $\\mu$: vectorul mediilor; $\\Gamma(\\vartheta)$: matricea $T \\times T$ a autocovarianțelor ARFIMA'),
      T('$\\Gamma$ comes from the ARFIMA autocovariances; the Durbin--Levinson recursion evaluates $\\ell$ in $O(T^2)$ operations', '$\\Gamma$ provine din autocovarianțele ARFIMA; recurența Durbin--Levinson calculează $\\ell$ în $O(T^2)$ operații')]),
    (T('\\textbf{Whittle} (approximate) ML \\refFT: replace the likelihood by $-\\frac12\\sum_j\\bigl[\\log f(\\lambda_j;\\vartheta) + I(\\lambda_j)/f(\\lambda_j;\\vartheta)\\bigr]$ over all Fourier frequencies', '\\textbf{Whittle} (ML aproximativă) \\refFT: înlocuim verosimilitatea cu $-\\frac12\\sum_j\\bigl[\\log f(\\lambda_j;\\vartheta) + I(\\lambda_j)/f(\\lambda_j;\\vartheta)\\bigr]$ pe toate frecvențele Fourier'),
     [T('$f(\\lambda_j;\\vartheta)$: the spectral density implied by the parameters $\\vartheta$', '$f(\\lambda_j;\\vartheta)$: densitatea spectrală implicată de parametrii $\\vartheta$'),
      T('fast ($O(T\\log T)$ with the FFT, the fast Fourier transform), close to exact ML in large samples', 'rapidă ($O(T\\log T)$ cu FFT, transformata Fourier rapidă), apropiată de ML exactă în eșantioane mari')]),
    T('Parametric estimators are efficient (SE about $\\sqrt{6}/(\\pi\\sqrt T)$ for ARFIMA$(0,d,0)$) only if the ARMA orders are right; choose $p$, $q$ by AIC/BIC', 'Estimatorii parametrici sînt eficienți (SE în jur de $\\sqrt{6}/(\\pi\\sqrt T)$ pentru ARFIMA$(0,d,0)$) doar dacă ordinele ARMA sînt corecte; alegem $p$, $q$ după AIC/BIC')), size='footnotesize')

chart(T('Monte Carlo: five estimators of $d$', 'Monte Carlo: cinci estimatori ai lui $d$'), 'tsa_ch8_mc', 'TSA_ch8_monte_carlo', [
    T('@{mc.reps} samples of $T = @{mc.T}$ per design; GPH and local Whittle with $m = \\lfloor T^{0.65}\\rfloor$; Whittle ML of ARFIMA$(0,d,0)$; R/S and DFA converted with $d = H - 1/2$',
      '@{mc.reps} de eșantioane de $T = @{mc.T}$ pentru fiecare variantă; GPH și Whittle local cu $m = \\lfloor T^{0{,}65}\\rfloor$; ML Whittle pentru ARFIMA$(0,d,0)$; R/S și DFA convertite cu $d = H - 1/2$')],
    h='0.7\\textheight')

interp(('the Monte Carlo', 'simulării Monte Carlo'), [
    (T('Correct model: Whittle ML is the most precise (SD @{mc.af.wh.s} against @{mc.af.lw.s} for local Whittle and @{mc.af.gph.s} for GPH at $d = 0.3$)', 'Model corect: ML Whittle este cel mai precis (abaterea standard @{mc.af.wh.s}, față de @{mc.af.lw.s} pentru Whittle local și @{mc.af.gph.s} pentru GPH la $d = 0{,}3$)'),
     [T('R/S is biased upwards under white noise (mean @{mc.wn.rs.m}) \\refMW', 'R/S este deplasat în sus pentru zgomot alb (media @{mc.wn.rs.m}) \\refMW')]),
    (T('AR(1) with $\\phi = 0.6$ and no long memory: ARFIMA$(0,d,0)$ by Whittle gives @{mc.ar.wh.m}, GPH @{mc.ar.gph.m}, local Whittle @{mc.ar.lw.m}', 'AR(1) cu $\\phi = 0{,}6$, fără memorie lungă: ARFIMA$(0,d,0)$ prin Whittle dă @{mc.ar.wh.m}, GPH @{mc.ar.gph.m}, Whittle local @{mc.ar.lw.m}'),
     [T('a misspecified parametric model turns short memory into ``long memory\'\'; semiparametric estimators suffer less, but are not immune', 'un model parametric greșit specificat transformă memoria scurtă în „memorie lungă”; estimatorii semiparametrici sînt mai puțin afectați, dar nu sînt imuni')]),
    T('Practice: always fit ARFIMA$(p,d,q)$ with $p, q \\ge 1$ as alternatives, and check the semiparametric $\\hat d$ for several $m$', 'În practică: estimați întotdeauna și ARFIMA$(p,d,q)$ cu $p, q \\ge 1$ și verificați $\\hat d$ semiparametric pentru mai multe valori ale lui $m$')])

D.frame(T('Romanian inflation: ARMA or ARFIMA?', 'Inflația din România: ARMA sau ARFIMA?'), table(
    'lrrrrr', T('\\textbf{Model (exact ML)}', '\\textbf{Model (ML exactă)}') + ' & $\\hat d$ & ' + T('\\textbf{AR / MA}', '\\textbf{AR / MA}') + ' & $\\ell$ & AIC & BIC',
    ['ARMA(1,1) & 0 & @{ft.ARMA11.phi} / @{ft.ARMA11.th} & @{ft.ARMA11.ll} & @{ft.ARMA11.aic} & @{ft.ARMA11.bic}',
     'ARMA(2,0) & 0 & @{ft.ARMA20.phi}, \\dots & @{ft.ARMA20.ll} & @{ft.ARMA20.aic} & @{ft.ARMA20.bic}',
     'ARFIMA(0,$d$,0) & @{ft.ARFIMA0d0.d} & -- & @{ft.ARFIMA0d0.ll} & \\textbf{@{ft.ARFIMA0d0.aic}} & \\textbf{@{ft.ARFIMA0d0.bic}}',
     'ARFIMA(1,$d$,0) & @{ft.ARFIMA1d0.d} & @{ft.ARFIMA1d0.phi} / -- & @{ft.ARFIMA1d0.ll} & @{ft.ARFIMA1d0.aic} & @{ft.ARFIMA1d0.bic}',
     'ARFIMA(0,$d$,1) & @{ft.ARFIMA0d1.d} & -- / @{ft.ARFIMA0d1.th} & @{ft.ARFIMA0d1.ll} & @{ft.ARFIMA0d1.aic} & @{ft.ARFIMA0d1.bic}',
     'ARFIMA(1,$d$,1) & @{ft.ARFIMA1d1.d} & @{ft.ARFIMA1d1.phi} / @{ft.ARFIMA1d1.th} & @{ft.ARFIMA1d1.ll} & @{ft.ARFIMA1d1.aic} & @{ft.ARFIMA1d1.bic}'],
    size='scriptsize') + items(
    T('Romanian monthly HICP inflation, seasonally adjusted, $T = @{ft.T}$ months (2005--2026); $\\ell$: maximised log-likelihood; bold: the smallest criterion', 'Inflația lunară IAPC din România, ajustată sezonier, $T = @{ft.T}$ de luni (2005--2026); $\\ell$: logaritmul verosimilității maximizate; aldin: cel mai mic criteriu'),
    T('ARFIMA$(0,d,0)$ wins with one parameter: $\\hat d = @{ft.ARFIMA0d0.d}$ (SE @{ft.se}); adding $\\phi$ or $\\theta$ barely changes $\\ell$', 'ARFIMA$(0,d,0)$ are cele mai mici criterii, cu un singur parametru: $\\hat d = @{ft.ARFIMA0d0.d}$ (SE @{ft.se}); adăugarea lui $\\phi$ sau $\\theta$ abia schimbă $\\ell$'),
    T('The best ARMA needs $\\phi = @{ft.ARMA11.phi}$ close to 1 and a cancelling MA term to imitate the slow decay', 'Cel mai bun ARMA are nevoie de $\\phi = @{ft.ARMA11.phi}$, aproape de 1, și de un termen MA care îl compensează, pentru a imita descreșterea lentă')))

chart(T('Romanian inflation: fitted autocorrelations', 'Inflația din România: autocorelațiile modelelor estimate'), 'tsa_ch8_arfima_fit', 'TSA_ch8_estimation', [
    T('Bars: sample ACF; lines: the ACF implied by ARFIMA$(0,d,0)$ and by ARMA(1,1), both estimated by exact ML', 'Bare: ACF de selecție; linii: ACF implicată de ARFIMA$(0,d,0)$ și de ARMA(1,1), ambele estimate prin ML exactă')],
    h='0.7\\textheight')

interp(('the fitted ACF', 'ACF a modelelor estimate'), [
    (T('At lag 12: sample @{ft.r12}, ARFIMA @{ft.d12}, ARMA(1,1) @{ft.a12}; at lag 24: @{ft.r24}, @{ft.d24} and @{ft.a24}', 'La lagul 12: în eșantion @{ft.r12}, ARFIMA @{ft.d12}, ARMA(1,1) @{ft.a12}; la lagul 24: @{ft.r24}, @{ft.d24} și @{ft.a24}'),
     [T('the two models agree on the first year and disagree on the second: their long-horizon forecasts will differ', 'cele două modele sînt de acord pentru primul an și diferă pentru al doilea: prognozele lor pe termen lung vor fi diferite')]),
    T('The ARFIMA curve matches the slow tail with one parameter; the ARMA curve goes to zero exponentially', 'Curba ARFIMA reproduce coada lentă cu un singur parametru; curba ARMA tinde exponențial spre zero'),
    T('Seasonal adjustment matters: a seasonal peak near the lowest frequencies would distort $\\hat d$ (Chapter 4)', 'Ajustarea sezonieră contează: un vîrf sezonier aproape de cele mai joase frecvențe ar distorsiona $\\hat d$ (Capitolul 4)')])

rows = []
for i, (_, en, ro) in enumerate(TROWS):
    rows.append(T(en, ro) + f' & @{{tb{i}.n}} & @{{tb{i}.rs}} & @{{tb{i}.dfa}} & @{{tb{i}.gph}} & @{{tb{i}.lw}} (@{{tb{i}.se}})')
D.frame(T('Long memory in twelve series', 'Memoria lungă în douăsprezece serii'), table(
    'lrrrrr', T('\\textbf{Series}', '\\textbf{Seria}') + ' & $T$ & $H_{R/S}$ & $H_{DFA}$ & $d_{GPH}$ & $d_{LW}$ (SE)', rows, size='scriptsize') + items(
    T('$m = \\lfloor T^{0.65}\\rfloor$; data: statsmodels, Eurostat, FRED, EODHD, BNR; returns daily since 2000 (EUR/RON since 2005)', '$m = \\lfloor T^{0{,}65}\\rfloor$; date: statsmodels, Eurostat, FRED, EODHD, BNR; randamente zilnice din 2000 (EUR/RON din 2005)')), size='footnotesize')

interp(('the table', 'tabelului'), [
    (T('Returns: $d \\approx 0$ for the S\\&P 500 (@{tb6.lw}) and EUR/RON (@{tb10.lw}); small but positive for the BET (@{tb8.lw})', 'Randamente: $d \\approx 0$ pentru S\\&P 500 (@{tb6.lw}) și EUR/RON (@{tb10.lw}); mic, dar pozitiv pentru BET (@{tb8.lw})'),
     [T('absolute returns: @{tb7.lw}, @{tb9.lw} and @{tb11.lw}: long memory in volatility (Section 6)', 'randamentele absolute: @{tb7.lw}, @{tb9.lw} și @{tb11.lw}: memorie lungă în volatilitate (secțiunea 6)')]),
    (T('Romanian 12-month inflation: $d \\approx$ @{tb2.lw}, against @{tb1.lw} for the monthly rate: the 12-month rate sums 12 overlapping monthly changes', 'Inflația anuală din România: $d \\approx$ @{tb2.lw}, față de @{tb1.lw} pentru rata lunară: rata anuală însumează 12 variații lunare suprapuse'),
     [T('the overlap kills the spectrum at the frequencies used by the estimator and pushes $\\hat d$ towards 1; model the monthly rate', 'suprapunerea anulează spectrul la frecvențele folosite de estimator și împinge $\\hat d$ spre 1; modelați rata lunară')]),
    T('US unemployment: $d \\approx$ @{tb5.lw}: non-stationary but mean-reverting; US inflation 1985--2019: @{tb4.lw}, against @{tb3.lw} on 1947--2026', 'Șomajul din SUA: $d \\approx$ @{tb5.lw}: nestaționar, dar cu revenire la medie; inflația din SUA în 1985--2019: @{tb4.lw}, față de @{tb3.lw} în 1947--2026')])

D.recap(('Estimating $d$', 'estimarea lui $d$'), [
    T('R/S and DFA: slopes of log--log plots, graphical, biased in small samples', 'R/S și DFA: pantele unor grafice log--log, instrumente grafice, deplasate în eșantioane mici'),
    T('GPH and local Whittle: regressions or likelihoods on the $m$ lowest frequencies; report several $m$', 'GPH și Whittle local: regresii sau verosimilități pe cele mai joase $m$ frecvențe; raportați mai multe valori ale lui $m$'),
    T('Exact ML and Whittle: efficient if the ARMA part is right; short memory can masquerade as $d$', 'ML exactă și Whittle: eficienți dacă partea ARMA este corectă; memoria scurtă poate fi confundată cu $d$'),
    T('Romanian monthly inflation: ARFIMA$(0,d,0)$ with $\\hat d = @{ft.ARFIMA0d0.d}$ beats ARMA by AIC and BIC', 'Inflația lunară din România: ARFIMA$(0,d,0)$ cu $\\hat d = @{ft.ARFIMA0d0.d}$ este preferat modelelor ARMA după AIC și BIC')])

# =============================================================================
# 5. PROGNOZA
# =============================================================================
D.section('Forecasting with ARFIMA', 'Prognoza cu ARFIMA')

D.frame(T('Forecasts from the AR($\\infty$) form', 'Prognoze din forma AR($\\infty$)'), items(
    (T('An invertible ARFIMA can be written as $\\pi(L)(x_t - \\mu) = \\varepsilon_t$, with $\\pi(L) = \\phi(L)(1-L)^d/\\theta(L) = \\sum_k\\pi_kL^k$', 'Un ARFIMA inversabil se poate scrie $\\pi(L)(x_t - \\mu) = \\varepsilon_t$, cu $\\pi(L) = \\phi(L)(1-L)^d/\\theta(L) = \\sum_k\\pi_kL^k$'),
     [T('$\\pi_k$: the AR($\\infty$) weights (for ARFIMA$(0,d,0)$, the weights of $(1-L)^d$); $\\hat x_{T+h}$: the forecast made at $T$ for $T + h$', '$\\pi_k$: ponderile AR($\\infty$) (pentru ARFIMA$(0,d,0)$, ponderile lui $(1-L)^d$); $\\hat x_{T+h}$: prognoza făcută la $T$ pentru $T + h$'),
      T('one step: $\\hat x_{T+1} = \\mu - \\sum_{k=1}^{T}\\pi_k(x_{T+1-k} - \\mu)$; $h$ steps: the same recursion, with forecasts in place of unknown values', 'un pas: $\\hat x_{T+1} = \\mu - \\sum_{k=1}^{T}\\pi_k(x_{T+1-k} - \\mu)$; $h$ pași: aceeași recurență, cu prognoze în locul valorilor necunoscute')]),
    (T('Example, ARFIMA$(0,0.4,0)$: $-\\pi_k = 0.4,\\ 0.12,\\ 0.064,\\ 0.042, \\dots$', 'Exemplu, ARFIMA$(0;\\ 0{,}4;\\ 0)$: $-\\pi_k = 0{,}4;\\ 0{,}12;\\ 0{,}064;\\ 0{,}042;\\ \\dots$'),
     [T('$\\hat x_{T+1} - \\mu = 0.4(x_T - \\mu) + 0.12(x_{T-1} - \\mu) + 0.064(x_{T-2} - \\mu) + \\dots$: the whole history counts', '$\\hat x_{T+1} - \\mu = 0{,}4(x_T - \\mu) + 0{,}12(x_{T-1} - \\mu) + 0{,}064(x_{T-2} - \\mu) + \\dots$: contează toată istoria'),
      T('an AR(1) uses only $x_T$; after a long period above the mean the ARFIMA forecast stays higher, for longer', 'un AR(1) folosește doar $x_T$; după o perioadă lungă peste medie, prognoza ARFIMA rămîne mai sus, mai mult timp')]),
    T('Forecasts revert to $\\mu$ hyperbolically; the forecast error variance $\\sigma^2\\sum_{j<h}\\psi_j^2$ ($\\psi_j$: the MA($\\infty$) weights) grows slowly towards $\\gamma(0)$ \\refRay', 'Prognozele revin la $\\mu$ hiperbolic; varianța erorii de prognoză $\\sigma^2\\sum_{j<h}\\psi_j^2$ ($\\psi_j$: ponderile MA($\\infty$)) crește lent spre $\\gamma(0)$ \\refRay')))

chart(T('US inflation: ARFIMA and AR forecasts', 'Inflația din SUA: prognoze ARFIMA și AR'), 'tsa_ch8_forecast_path', 'TSA_ch8_forecasting', [
    T('US monthly CPI inflation, annualised, $T = @{fp.T}$ months up to @{fp.lastd}; ARFIMA(1,$d$,0) estimated by Whittle; AR($p$) with $p$ by AIC; 36 months ahead',
      'Inflația lunară IPC din SUA, anualizată, $T = @{fp.T}$ de luni pînă în @{fp.lastd}; ARFIMA(1,$d$,0) estimat prin Whittle; AR($p$) cu $p$ ales după AIC; 36 de luni înainte')],
    h='0.7\\textheight')

interp(('the two forecasts', 'celor două prognoze'), [
    (T('ARFIMA: $\\hat d = @{fp.d}$, $\\hat\\phi = @{fp.phi}$; AR: $p = @{fp.p}$ lags, the largest allowed', 'ARFIMA: $\\hat d = @{fp.d}$, $\\hat\\phi = @{fp.phi}$; AR: $p = @{fp.p}$ laguri, numărul maxim permis'),
     [T('the AR needs a year of lags to imitate what $d$ does with one parameter', 'AR are nevoie de un an de laguri pentru a imita ce face $d$ cu un singur parametru')]),
    (T('Last 12 months average @{fp.m12}\\%; the forecasts: @{fp.fa1}\\% and @{fp.fr1}\\% next month, @{fp.fa12}\\% and @{fp.fr12}\\% in a year, @{fp.fa36}\\% and @{fp.fr36}\\% in three years', 'Media ultimelor 12 luni @{fp.m12}\\%; prognozele: @{fp.fa1}\\% și @{fp.fr1}\\% luna viitoare, @{fp.fa12}\\% și @{fp.fr12}\\% peste un an, @{fp.fa36}\\% și @{fp.fr36}\\% peste trei ani'),
     [T('both approach the long-run mean @{fp.mean}\\%; the paths differ little because the AR(12) already carries a long memory', 'ambele se apropie de media pe termen lung @{fp.mean}\\%; traiectoriile diferă puțin, deoarece AR(12) poartă deja o memorie lungă')]),
    T('Whether the long-run mean of 1947--2026 is the right anchor today is a question about regimes, not about $d$', 'Dacă media pe termen lung din 1947--2026 este ancora potrivită azi este o întrebare despre regimuri, nu despre $d$')])

chart(T('Does long memory improve forecasts?', 'Contribuția memoriei lungi la prognoză'), 'tsa_ch8_forecast', 'TSA_ch8_forecasting', [
    T('Pseudo out-of-sample, expanding window: Romania @{fc.ro.n} origins (@{fc.ro.f0}--@{fc.ro.f1}), seasonal adjustment inside each window; United States @{fc.us.n} origins (@{fc.us.f0}--@{fc.us.f1}). RMSE relative to AR($p$); below 1: better',
      'În afara eșantionului, fereastră extinsă: România @{fc.ro.n} de origini (@{fc.ro.f0}--@{fc.ro.f1}), ajustare sezonieră în fiecare fereastră; Statele Unite @{fc.us.n} de origini (@{fc.us.f0}--@{fc.us.f1}). RMSE relativ la AR($p$); sub 1: mai bun')],
    h='0.7\\textheight')

interp(('the forecast comparison', 'comparației prognozelor'), [
    (T('Romania: ARFIMA beats the AR at every horizon, by 2--4\\% (ratio @{fc.ro.rel.1} at 1 month, @{fc.ro.rel.12} at 12, @{fc.ro.rel.24} at 24)', 'România: ARFIMA este mai precis decît AR la toate orizonturile, cu 2--4\\% (raportul @{fc.ro.rel.1} la o lună, @{fc.ro.rel.12} la 12, @{fc.ro.rel.24} la 24)'),
     [T('RMSE at 12 months: ARFIMA @{fc.ro.ARFIMA.12}, AR @{fc.ro.AR.12}, mean @{fc.ro.Mean.12}, random walk @{fc.ro.RW.12} (pp per month)', 'RMSE la 12 luni: ARFIMA @{fc.ro.ARFIMA.12}, AR @{fc.ro.AR.12}, media @{fc.ro.Mean.12}, mers aleator @{fc.ro.RW.12} (pp pe lună)')]),
    (T('United States: ratios between @{fc.us.rel.6} and @{fc.us.rel.12}: practically a tie with a rich AR($p$)', 'Statele Unite: rapoarte între @{fc.us.rel.6} și @{fc.us.rel.12}: practic aceeași precizie ca un AR($p$) cu multe laguri'),
     [T('the random walk is far worse at all horizons beyond one month', 'mersul aleator este mult mai slab la toate orizonturile de peste o lună')]),
    T('Long memory gives modest, steady gains; the large errors come from regime changes (2008, 2021--2022) that no linear model foresees; test the gains with Diebold--Mariano (Chapter 4)', 'Memoria lungă aduce cîștiguri modeste, dar constante; erorile mari provin din schimbări de regim (2008, 2021--2022) pe care niciun model liniar nu le anticipează; testați cîștigurile cu Diebold--Mariano (Capitolul 4)')])

D.recap(('Forecasting with ARFIMA', 'prognoza cu ARFIMA'), [
    T('Forecasts from the AR($\\infty$) weights $\\pi_k$: the whole history counts', 'Prognoze din ponderile AR($\\infty$) $\\pi_k$: contează toată istoria'),
    T('Hyperbolic return to the mean; slowly growing forecast uncertainty', 'Revenire hiperbolică la medie; incertitudinea prognozei crește lent'),
    T('Inflation: small gains over AR in Romania, a tie in the United States', 'Inflația: cîștiguri mici față de AR în România, aceeași precizie în Statele Unite')])

# =============================================================================
# 6. MEMORIA LUNGĂ A VOLATILITĂȚII
# =============================================================================
D.section('Long memory in volatility', 'Memoria lungă a volatilității')

D.frame(T('Case study: Ding, Granger and Engle (1993)', 'Studiu de caz: Ding, Granger și Engle (1993)'), two(
    ph('engle', T('Robert F. Engle, Nobel Prize 2003 with Clive Granger', 'Robert F. Engle, Premiul Nobel 2003, împreună cu Clive Granger'), h='0.44\\textheight'),
    items((T('\\refDGE: daily S\\&P 500 returns, 1928--1991', '\\refDGE: randamentele zilnice S\\&P 500, 1928--1991'),
           [T('returns are almost uncorrelated, but $|r_t|^\\delta$ is autocorrelated for thousands of lags, most strongly for $\\delta \\approx 1$', 'randamentele sînt aproape necorelate, dar $|r_t|^\\delta$ este autocorelat pe mii de laguri, cel mai puternic pentru $\\delta \\approx 1$'),
            T('the ``long memory property\'\' of stock market returns: memory lives in the size, not in the sign', '„proprietatea de memorie lungă” a randamentelor bursiere: memoria se află în mărimea randamentelor, nu în semnul lor')]),
          T('A GARCH(1,1) (Chapter 5) implies autocorrelations of $r_t^2$ that fall like $(\\alpha + \\beta)^k$: exponentially', 'Un GARCH(1,1) (Capitolul 5) implică autocorelații ale lui $r_t^2$ care scad ca $(\\alpha + \\beta)^k$: exponențial'),
          T('This led to FIGARCH \\refBBM\\ and, with high-frequency data, to realised volatility models \\refABDL, \\refCorsi', 'Rezultatul a dus la FIGARCH \\refBBM\\ și, cu date de înaltă frecvență, la modelele de volatilitate realizată \\refABDL, \\refCorsi'))))

chart(T('Returns, absolute and squared returns', 'Randamente, randamente absolute și pătrate'), 'tsa_ch8_vol_acf', 'TSA_ch8_volatility_memory', [
    T('Daily log returns of the S\\&P 500 ($T = @{vo.sp500.n}$) and of the BET ($T = @{vo.bet.n}$), 2000--2026; ACF up to lag 250 (one trading year); shaded: $\\pm 1.96/\\sqrt T$',
      'Randamentele logaritmice zilnice ale S\\&P 500 ($T = @{vo.sp500.n}$) și BET ($T = @{vo.bet.n}$), 2000--2026; ACF pînă la lagul 250 (un an bursier); zona colorată: $\\pm 1{,}96/\\sqrt T$')],
    h='0.7\\textheight')

interp(('the volatility ACF', 'ACF a volatilității'), [
    (T('S\\&P 500: $|r_t|$ has $\\hat\\rho = @{vo.sp500.a.1}$ at lag 1 and still @{vo.sp500.a.100} at lag 100; @{vo.sp500.a.npos} of 250 lags above the band', 'S\\&P 500: $|r_t|$ are $\\hat\\rho = @{vo.sp500.a.1}$ la lagul 1 și încă @{vo.sp500.a.100} la lagul 100; @{vo.sp500.a.npos} din 250 de laguri deasupra benzii'),
     [T('BET: @{vo.bet.a.1} at lag 1, @{vo.bet.a.250} at lag 250, @{vo.bet.a.npos} of 250 above the band', 'BET: @{vo.bet.a.1} la lagul 1, @{vo.bet.a.250} la lagul 250, @{vo.bet.a.npos} din 250 deasupra benzii')]),
    (T('$|r_t|$ is more persistent than $r_t^2$, as in \\refDGE; squares are dominated by a few crash days', '$|r_t|$ este mai persistent decît $r_t^2$, ca în \\refDGE; pătratele sînt dominate de cîteva zile de crah'),
     [T('local Whittle for $|r_t|$: @{vo.sp500.alw} (S\\&P 500), @{vo.bet.alw} (BET); after a random shuffle of the days: @{vo.sp500.shuf} and @{vo.bet.shuf}', 'Whittle local pentru $|r_t|$: @{vo.sp500.alw} (S\\&P 500), @{vo.bet.alw} (BET); după o permutare aleatoare a zilelor: @{vo.sp500.shuf} și @{vo.bet.shuf}')]),
    T('The shuffle keeps the distribution and destroys the order: the memory is in the timing of calm and turbulent days, not in fat tails', 'Permutarea păstrează distribuția și distruge ordinea: memoria se află în succesiunea zilelor calme și agitate, nu în cozile groase')])

D.frame(T('Realised volatility', 'Volatilitatea realizată'), items(
    (T('\\textbf{Realised variance} of month $m$: $RV_m = \\sum_{t \\in m} r_t^2$, the sum of the squared daily returns of the month \\refABDL', '\\textbf{Varianța realizată} a lunii $m$: $RV_m = \\sum_{t \\in m} r_t^2$, suma pătratelor randamentelor zilnice din lună \\refABDL'),
     [T('realised volatility $\\sqrt{RV_m}$ measures the volatility of month $m$ almost without a model; with intraday data the same idea gives a daily RV', 'volatilitatea realizată $\\sqrt{RV_m}$ măsoară volatilitatea lunii $m$ aproape fără model; cu date intrazilnice aceeași idee dă un RV zilnic')]),
    (T('We model $\\log\\sqrt{RV_m}$: the logarithm makes the distribution close to the Normal distribution and removes the positivity constraint', 'Modelăm $\\log\\sqrt{RV_m}$: logaritmul aduce distribuția aproape de distribuția Normală și elimină restricția de pozitivitate'),
     [T('the stylised fact of \\refABDL: log RV is well described by a long-memory model with $d \\approx 0.4$', 'faptul stilizat din \\refABDL: logaritmul RV este bine descris de un model cu memorie lungă, cu $d \\approx 0{,}4$')]),
    T('Unlike $|r_t|$, $\\log\\sqrt{RV_m}$ is a precise measure: the noise of single days averages out within the month', 'Spre deosebire de $|r_t|$, $\\log\\sqrt{RV_m}$ este o măsură precisă: zgomotul zilelor individuale se compensează în cadrul lunii')))

chart(T('Monthly realised volatility: ARFIMA or ARMA?', 'Volatilitatea realizată lunară: ARFIMA sau ARMA?'), 'tsa_ch8_rv', 'TSA_ch8_volatility_memory', [
    T('Log monthly realised volatility from daily returns, @{rv.sp500.T} months (2000--2026); lines: ACF of ARFIMA(1,$d$,0) and ARMA(1,1), both by exact ML', 'Logaritmul volatilității realizate lunare din randamentele zilnice, @{rv.sp500.T} de luni (2000--2026); linii: ACF a modelelor ARFIMA(1,$d$,0) și ARMA(1,1), ambele prin ML exactă')],
    h='0.7\\textheight')

interp(('realised volatility', 'volatilității realizate'), [
    (T('All estimators agree: $d$ between 0.36 and 0.47 for both markets (ML @{rv.sp500.ml} and @{rv.bet.ml})', 'Toți estimatorii sînt de acord: $d$ între 0,36 și 0,47 pentru ambele piețe (ML @{rv.sp500.ml} și @{rv.bet.ml})'),
     [T('BET: ARFIMA fits the slow tail (lag 24: sample @{rv.bet.r24}) and wins on BIC (@{rv.bet.bicf} against @{rv.bet.bica} for ARMA(1,1))', 'BET: ARFIMA reproduce coada lentă (lagul 24: în eșantion @{rv.bet.r24}) și are un BIC mai mic (@{rv.bet.bicf} față de @{rv.bet.bica} pentru ARMA(1,1))')]),
    (T('S\\&P 500: the sample ACF dies out after about 20 months; ARMA(1,1) with $\\phi = @{rv.sp500.a_phi}$ fits as well (BIC @{rv.sp500.bica} against @{rv.sp500.bicf})', 'S\\&P 500: ACF de selecție se stinge după aproximativ 20 de luni; ARMA(1,1) cu $\\phi = @{rv.sp500.a_phi}$ se potrivește la fel de bine (BIC @{rv.sp500.bica} față de @{rv.sp500.bicf})'),
     [T('with 27 years of data, long memory and a persistent ARMA are hard to tell apart', 'cu 27 de ani de date, memoria lungă și un ARMA persistent sînt greu de deosebit')]),
    T('The sample ACF of a long-memory series is biased downwards (the sample mean absorbs part of the slow component): judge models by likelihood, not by eye', 'ACF de selecție a unei serii cu memorie lungă este deplasată în jos (media de selecție absoarbe o parte din componenta lentă): judecați modelele după verosimilitate, nu vizual')])

D.frame(T('FIGARCH and HAR', 'FIGARCH și HAR'), items(
    (T('\\textbf{FIGARCH}$(1,d,1)$ \\refBBM: write GARCH(1,1) as an ARMA(1,1) in $\\varepsilon_t^2$ and apply $(1-L)^d$:', '\\textbf{FIGARCH}$(1,d,1)$ \\refBBM: scriem GARCH(1,1) ca ARMA(1,1) în $\\varepsilon_t^2$ și aplicăm $(1-L)^d$:'),
     [T('$\\sigma_t^2$: the conditional variance; $\\varepsilon_t$: the return shock; $\\omega > 0$, $\\beta$, $\\phi$: GARCH-type parameters (Chapter 5)', '$\\sigma_t^2$: varianța condiționată; $\\varepsilon_t$: șocul randamentului; $\\omega > 0$, $\\beta$, $\\phi$: parametri de tip GARCH (Capitolul 5)'),
      T('$\\sigma_t^2 = \\omega + \\beta\\sigma_{t-1}^2 + \\bigl[1 - \\beta L - (1 - \\phi L)(1-L)^d\\bigr]\\varepsilon_t^2 = \\omega^* + \\sum_{k\\ge1}\\lambda_k\\varepsilon_{t-k}^2$', '$\\sigma_t^2 = \\omega + \\beta\\sigma_{t-1}^2 + \\bigl[1 - \\beta L - (1 - \\phi L)(1-L)^d\\bigr]\\varepsilon_t^2 = \\omega^* + \\sum_{k\\ge1}\\lambda_k\\varepsilon_{t-k}^2$'),
      T('the ARCH($\\infty$) weights $\\lambda_k$ decay like $k^{-1-d}$; $d = 0$: GARCH, $d = 1$: IGARCH', 'ponderile ARCH($\\infty$) $\\lambda_k$ scad ca $k^{-1-d}$; $d = 0$: GARCH, $d = 1$: IGARCH')]),
    (T('\\textbf{HAR} (heterogeneous autoregressive model) \\refCorsi: $RV_{t+1} = c + \\beta_d RV_t + \\beta_w \\overline{RV}_{t-4:t} + \\beta_m \\overline{RV}_{t-21:t} + u_{t+1}$',
       '\\textbf{HAR} (model autoregresiv heterogen) \\refCorsi: $RV_{t+1} = c + \\beta_d RV_t + \\beta_w \\overline{RV}_{t-4:t} + \\beta_m \\overline{RV}_{t-21:t} + u_{t+1}$'),
     [T('$\\overline{RV}_{t-4:t}$, $\\overline{RV}_{t-21:t}$: averages over the last 5 and 22 days (weekly, monthly); $\\beta_d$, $\\beta_w$, $\\beta_m$: their weights; traders with three horizons; estimated by OLS', '$\\overline{RV}_{t-4:t}$, $\\overline{RV}_{t-21:t}$: mediile pe ultimele 5 și 22 de zile (săptămînală, lunară); $\\beta_d$, $\\beta_w$, $\\beta_m$: ponderile lor; participanți cu trei orizonturi; estimat prin OLS'),
      T('not a long-memory model, but three steps that imitate a power law over the horizons that matter', 'nu este un model cu memorie lungă, ci trei trepte care imită o lege de putere pe orizonturile relevante')]),
    T('Both answer the failure of GARCH from Chapter 5: volatility shocks die out too fast in GARCH', 'Ambele răspund unei slăbiciuni a modelului GARCH din Capitolul 5: în GARCH șocurile volatilității se sting prea repede')))

chart(T('FIGARCH against GARCH; HAR against $(1-L)^d$', 'FIGARCH comparat cu GARCH; HAR comparat cu $(1-L)^d$'), 'tsa_ch8_vol_models', 'TSA_ch8_volatility_memory', [
    T('Left: ARCH($\\infty$) weights of GARCH(1,1) and FIGARCH(1,$d$,1), Student $t$ errors, daily returns 2000--2026 (\\texttt{arch}). Right: HAR for the daily range-based volatility of the S\\&P 500 (Parkinson estimator from high and low prices)',
      'Stînga: ponderile ARCH($\\infty$) ale GARCH(1,1) și FIGARCH(1,$d$,1), erori Student $t$, randamente zilnice 2000--2026 (\\texttt{arch}). Dreapta: HAR pentru volatilitatea zilnică pe baza amplitudinii S\\&P 500 (estimatorul Parkinson, din prețurile maxime și minime)')],
    h='0.7\\textheight')

interp(('FIGARCH and HAR', 'modelelor FIGARCH și HAR'), [
    (T('FIGARCH: $\\hat d = @{vm.sp500.d}$ (S\\&P 500) and @{vm.bet.d} (BET); BIC falls by @{vm.sp500.dbic} and @{vm.bet.dbic} against GARCH', 'FIGARCH: $\\hat d = @{vm.sp500.d}$ (S\\&P 500) și @{vm.bet.d} (BET); BIC scade cu @{vm.sp500.dbic} și @{vm.bet.dbic} față de GARCH'),
     [T('weight of a shock 100 days ago in today\'s variance: GARCH $@{vm.sp500.w100g}$, FIGARCH $@{vm.sp500.w100f} \\cdot 10^{-4}$ (S\\&P 500): three orders of magnitude more', 'ponderea unui șoc de acum 100 de zile în varianța de azi: GARCH $@{vm.sp500.w100g}$, FIGARCH $@{vm.sp500.w100f} \\cdot 10^{-4}$ (S\\&P 500): cu trei ordine de mărime mai mult')]),
    (T('GARCH reaches persistence by $\\alpha + \\beta = @{vm.sp500.ab}$, near 1: the near-IGARCH result of Chapter 5 is a symptom of long memory', 'GARCH obține persistența prin $\\alpha + \\beta = @{vm.sp500.ab}$, aproape de 1: rezultatul aproape IGARCH din Capitolul 5 este un simptom al memoriei lungi'),
     [T('HAR: $\\hat\\beta_d = @{har.bd}$, $\\hat\\beta_w = @{har.bw}$, $\\hat\\beta_m = @{har.bm}$; the weekly component dominates', 'HAR: $\\hat\\beta_d = @{har.bd}$, $\\hat\\beta_w = @{har.bw}$, $\\hat\\beta_m = @{har.bm}$; componenta săptămînală domină')]),
    T('The HAR steps (@{har.w1} at lag 1, @{har.w2} at lags 2--5, @{har.w6} at lags 6--22) follow the hyperbolic weights of $(1-L)^d$ with $d = @{har.d}$ from local Whittle', 'Treptele HAR (@{har.w1} la lagul 1, @{har.w2} la lagurile 2--5, @{har.w6} la lagurile 6--22) urmează ponderile hiperbolice ale lui $(1-L)^d$, cu $d = @{har.d}$ din estimatorul Whittle local')])

D.recap(('Long memory in volatility', 'memoria lungă a volatilității'), [
    T('Returns: $d \\approx 0$; absolute returns and realised volatility: $d \\approx 0.4$', 'Randamente: $d \\approx 0$; randamente absolute și volatilitate realizată: $d \\approx 0{,}4$'),
    T('The shuffle test: memory in the order of the days, not in the distribution', 'Testul permutării: memoria se află în ordinea zilelor, nu în distribuție'),
    T('FIGARCH: hyperbolic ARCH($\\infty$) weights; HAR: a simple OLS approximation', 'FIGARCH: ponderi ARCH($\\infty$) hiperbolice; HAR: o aproximare simplă prin OLS')])

# =============================================================================
# 7. MEMORIE LUNGĂ APARENTĂ
# =============================================================================
D.section('Spurious long memory', 'Memoria lungă aparentă')

D.frame(T('Breaks and regimes look like long memory', 'Rupturile și regimurile seamănă cu memoria lungă'), items(
    (T('A level shift adds a slow, step-like component: low frequencies gain power and the sample ACF decays slowly', 'O schimbare de nivel adaugă o componentă lentă, în trepte: crește puterea spectrală la frecvențele joase, iar ACF de selecție scade lent'),
     [T('the same mechanism that makes a broken series look like a unit root (Perron, Chapter 3)', 'același mecanism care face ca o serie cu ruptură să pară cu rădăcină unitară (Perron, Capitolul 3)')]),
    (T('\\refDI: a Markov-switching mean with rare switches is ``observationally equivalent\'\' to long memory', '\\refDI: o medie cu schimbări de regim de tip Markov, cu schimbări rare, este „echivalentă observațional” cu memoria lungă'),
     [T('the fewer the switches in a sample of length $T$, the closer the series looks to $I(d)$', 'cu cît sînt mai puține schimbări într-un eșantion de lungime $T$, cu atît seria seamănă mai mult cu $I(d)$'),
      T('regime-switching models are the topic of Chapter 10', 'modelele cu schimbare de regim sînt tema Capitolului 10')]),
    T('\\refGH: occasional breaks explain much of the long memory of S\\&P 500 absolute returns', '\\refGH: rupturile ocazionale explică o mare parte din memoria lungă a randamentelor absolute S\\&P 500')))

chart(T('Short memory that looks long', 'Memorie scurtă care pare lungă'), 'tsa_ch8_spurious', 'TSA_ch8_spurious_memory', [
    T('@{sp.reps} samples of $T = @{sp.T}$, true $d = 0$. Left: $N(0,1)$ noise with one mean shift at $T/2$; right: $N(0,1)$ noise plus a mean that switches between 0 and 1 with probability $p$ each period',
      '@{sp.reps} de eșantioane de $T = @{sp.T}$, $d = 0$ în realitate. Stînga: zgomot $N(0,1)$ cu o singură schimbare a mediei la $T/2$; dreapta: zgomot $N(0,1)$ plus o medie care trece între 0 și 1 cu probabilitatea $p$ în fiecare perioadă')],
    h='0.7\\textheight')

interp(('the spurious memory', 'memoriei aparente'), [
    (T('One shift of half a standard deviation already gives $\\hat d = @{sp.b05.lw}$ (local Whittle); a shift of one standard deviation gives @{sp.b10.lw}', 'O singură schimbare de o jumătate de abatere standard dă deja $\\hat d = @{sp.b05.lw}$ (Whittle local); o schimbare de o abatere standard dă @{sp.b10.lw}'),
     [T('the break is hard to see by eye in noisy data, yet the estimator reports moderate long memory', 'ruptura se vede greu cu ochiul în date zgomotoase, dar estimatorul raportează o memorie lungă moderată')]),
    (T('Regime switching: with $p = 0.01$ (about @{sp.s001.sw} switches) $\\hat d = @{sp.s001.lw}$; with $p = 0.2$ (frequent switches) $\\hat d = @{sp.s02.lw}$', 'Schimbări de regim: cu $p = 0{,}01$ (aproximativ @{sp.s001.sw} schimbări) $\\hat d = @{sp.s001.lw}$; cu $p = 0{,}2$ (schimbări frecvente) $\\hat d = @{sp.s02.lw}$'),
     [T('rare regimes look like memory, frequent ones average out into short memory', 'regimurile rare imită memoria lungă, cele frecvente se compensează și dau memorie scurtă')]),
    T('A significant $\\hat d$ is consistent with long memory and with breaks: other evidence is needed to decide', 'Un $\\hat d$ semnificativ este compatibil atît cu memoria lungă, cît și cu rupturile: este nevoie de alte dovezi pentru a decide')])

chart(T('The Nile in 1898: memory or a break?', 'Nilul în 1898: memorie sau ruptură?'), 'tsa_ch8_nile_break', 'TSA_ch8_spurious_memory', [
    T('Left: the flow with the means before and after 1898, the change point of \\refCobb (the first Aswan dam was completed in 1902); right: R/S plot of the raw series and of the series with the two regime means removed',
      'Stînga: debitul, cu mediile dinainte și de după 1898, punctul de schimbare din \\refCobb (primul baraj de la Aswan a fost finalizat în 1902); dreapta: graficul R/S pentru seria brută și pentru seria fără cele două medii de regim')],
    h='0.7\\textheight')

interp(('the Nile break', 'rupturii Nilului'), [
    (T('The mean falls from @{ni.pre} to @{ni.post} ($-$@{ni.drop}\\%) after 1898', 'Media scade de la @{ni.pre} la @{ni.post} ($-$@{ni.drop}\\%) după 1898'),
     [T('raw series: $\\hat d = @{ni.raw.ml}$ (exact ML, SE @{ni.raw.ml_se}), local Whittle @{ni.raw.lw}, R/S $H = @{ni.raw.H}$', 'seria brută: $\\hat d = @{ni.raw.ml}$ (ML exactă, SE @{ni.raw.ml_se}), Whittle local @{ni.raw.lw}, R/S $H = @{ni.raw.H}$')]),
    (T('Regime means removed: $\\hat d = @{ni.adj.ml}$ (SE @{ni.adj.ml_se}), local Whittle @{ni.adj.lw}; $\\hat\\rho(1)$ falls from @{ni.raw.r1} to @{ni.adj.r1}', 'Fără mediile de regim: $\\hat d = @{ni.adj.ml}$ (SE @{ni.adj.ml_se}), Whittle local @{ni.adj.lw}; $\\hat\\rho(1)$ scade de la @{ni.raw.r1} la @{ni.adj.r1}'),
     [T('for these 100 years one break explains almost all of the memory', 'pentru acești 100 de ani o singură ruptură explică aproape toată memoria')]),
    T('Hurst\'s own evidence came from much longer records (the Roda gauge, from the 7th century); the lesson is to test for breaks before reading $d$', 'Dovezile lui Hurst proveneau din înregistrări mult mai lungi (nilometrul de la Roda, din secolul al VII-lea); lecția: testați rupturile înainte de a interpreta $d$')])

chart(T('Is memory stable over time?', 'Stabilitatea memoriei în timp'), 'tsa_ch8_rolling', 'TSA_ch8_spurious_memory', [
    T('Top: local Whittle $\\hat d$ of US monthly inflation in rolling 20-year windows (dated at the window end); bottom: DFA exponent of daily S\\&P 500 and BET returns in 1000-day windows moved by 21 days; shaded: 95\\% Monte Carlo bands for i.i.d. series \\refWeron',
      'Sus: $\\hat d$ Whittle local pentru inflația lunară din SUA, pe ferestre mobile de 20 de ani (datate la sfîrșitul ferestrei); jos: exponentul DFA al randamentelor zilnice S\\&P 500 și BET, pe ferestre de 1000 de zile mutate cu 21 de zile; zona colorată: benzi Monte Carlo de 95\\% pentru serii i.i.d. \\refWeron')],
    h='0.7\\textheight')

interp(('the rolling estimates', 'estimărilor pe ferestre mobile'), [
    (T('US inflation: $\\hat d$ peaks at @{rl.us.max} in the window ending @{rl.us.maxd} (the Great Inflation and the Volcker disinflation) and falls to @{rl.us.min} (@{rl.us.mind})', 'Inflația din SUA: $\\hat d$ atinge @{rl.us.max} în fereastra care se încheie în @{rl.us.maxd} (Marea Inflație și dezinflația Volcker) și coboară la @{rl.us.min} (@{rl.us.mind})'),
     [T('in the calm years of inflation targeting $d$ is inside the band of no memory [@{rl.us.lo}, @{rl.us.hi}]; the 2021--2022 surge brings it back to @{rl.us.last}', 'în anii calmi ai țintirii inflației $d$ se află în banda fără memorie [@{rl.us.lo}, @{rl.us.hi}]; creșterea bruscă din 2021--2022 îl readuce la @{rl.us.last}')]),
    (T('Returns: the S\\&P 500 exponent stays mostly below 0.5 (minimum @{rl.sp500.min}, @{rl.sp500.mind}); the BET exceeds the band in @{rl.bet.above}\\% of the windows', 'Randamentele: exponentul S\\&P 500 stă mai ales sub 0,5 (minimum @{rl.sp500.min}, @{rl.sp500.mind}); BET depășește banda în @{rl.bet.above}\\% din ferestre'),
     [T('a less liquid market shows some persistence in returns, fading in recent years', 'o piață mai puțin lichidă arată o anumită persistență a randamentelor, care slăbește în ultimii ani')]),
    T('The full-sample US estimate (0.55) mixes regimes: memory that comes and goes with monetary regimes is a sign of breaks, not of a constant $d$', 'Estimarea pe tot eșantionul pentru SUA (0,55) amestecă regimuri: o memorie care apare și dispare odată cu regimurile monetare semnalează rupturi, nu un $d$ constant')])

D.frame(T('Checks against spurious long memory', 'Verificări împotriva memoriei lungi aparente'), items(
    T('Plot the series and test for breaks (Chapter 3: Zivot--Andrews, Bai--Perron); re-estimate $d$ on subsamples and after removing regime means', 'Reprezentați seria grafic și testați rupturile (Capitolul 3: Zivot--Andrews, Bai--Perron); reestimați $d$ pe subeșantioane și după eliminarea mediilor de regim'),
    T('Report $\\hat d$ for several bandwidths $m$; a value that collapses when $m$ changes is fragile', 'Raportați $\\hat d$ pentru mai multe lățimi de bandă $m$; o valoare care se modifică puternic cînd $m$ se schimbă este fragilă'),
    T('Compare ARFIMA with ARMA and with regime-switching models by likelihood and out-of-sample forecasts (Chapter 10)', 'Comparați ARFIMA cu ARMA și cu modele cu schimbare de regim după verosimilitate și prin prognoze în afara eșantionului (Capitolul 10)'),
    T('Use the shuffle test and Monte Carlo bands built for the same sample length', 'Folosiți testul permutării și benzi Monte Carlo construite pentru aceeași lungime a eșantionului'),
    T('Avoid overlapping data (12-month inflation, rolling sums): they create artificial low-frequency power', 'Evitați datele suprapuse (inflația anuală, sumele pe ferestre mobile): ele creează putere artificială la frecvențele joase')))

D.recap(('Spurious long memory', 'memoria lungă aparentă'), [
    T('Breaks and rare regime switches produce $\\hat d > 0$ in short-memory series', 'Rupturile și schimbările rare de regim produc $\\hat d > 0$ în serii cu memorie scurtă'),
    T('The Nile: one break in 1898 explains most of the memory of 1871--1970', 'Nilul: o singură ruptură în 1898 explică cea mai mare parte a memoriei din 1871--1970'),
    T('US inflation: $d$ changes with the monetary regime', 'Inflația din SUA: $d$ se schimbă odată cu regimul monetar')])

# =============================================================================
# 8. COINTEGRARE FRACȚIONARĂ (TRIMITERE)
# =============================================================================
D.section('A pointer: fractional cointegration', 'O trimitere: cointegrarea fracționară')

D.frame(T('Fractional cointegration', 'Cointegrarea fracționară'), items(
    (T('Chapter 7: two $I(1)$ series are cointegrated if a combination $y_t - \\beta x_t$ is $I(0)$', 'Capitolul 7: două serii $I(1)$ sînt cointegrate dacă o combinație $y_t - \\beta x_t$ este $I(0)$'),
     [T('\\textbf{fractional cointegration}: $x_t, y_t \\sim I(d)$ and $y_t - \\beta x_t \\sim I(d - b)$ with $0 < b \\le d$', '\\textbf{cointegrare fracționară}: $x_t, y_t \\sim I(d)$ și $y_t - \\beta x_t \\sim I(d - b)$, cu $0 < b \\le d$'),
      T('the equilibrium error may itself have long memory: deviations from equilibrium are corrected, but slowly', 'eroarea de echilibru poate avea ea însăși memorie lungă: abaterile de la echilibru sînt corectate, dar lent')]),
    (T('Example: purchasing power parity \\refCL: the real exchange rate reverts to parity with $d$ between 0 and 1', 'Exemplu: paritatea puterii de cumpărare \\refCL: cursul real revine la paritate cu un $d$ între 0 și 1'),
     [T('the Engle--Granger and Johansen tests of Chapter 7 assume $b = 1$ and may miss such slow adjustment', 'testele Engle--Granger și Johansen din Capitolul 7 presupun $b = 1$ și pot să nu detecteze o astfel de ajustare lentă')]),
    T('Beyond this course; a natural project: estimate $d$ of the residual of a cointegrating regression with local Whittle', 'Dincolo de acest curs; un proiect natural: estimați $d$ al reziduului unei regresii de cointegrare cu estimatorul Whittle local')))

# =============================================================================
# 9. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Code}: a first draft of fractional differencing, of the GPH and local Whittle estimators, of an ARFIMA likelihood', '\\textbf{Cod}: o primă versiune a diferențierii fracționare, a estimatorilor GPH și Whittle local, a unei verosimilități ARFIMA'),
    T('\\textbf{Explanation}: a second explanation of the periodogram, of the bandwidth trade-off, of the link $H = d + 1/2$', '\\textbf{Explicații}: o a doua explicație a periodogramei, a compromisului lățimii de bandă, a legăturii $H = d + 1/2$'),
    T('\\textbf{Exploration}: $d$ for many countries\' inflation or many assets\' volatility, with subsamples and bandwidths', '\\textbf{Explorare}: $d$ pentru inflația multor țări sau volatilitatea multor active, cu subeșantioane și lățimi de bandă diferite'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that downloads Romanian monthly HICP (Eurostat prc\\_hicp\\_minr), computes the monthly inflation rate, removes the monthly means, estimates d with GPH and local Whittle for bandwidths T\\^{}0.5 to T\\^{}0.8, fits ARFIMA(p,d,q) by exact Gaussian maximum likelihood and compares it with ARMA(1,1) by BIC.}',
        '\\aiprompt{Write Python code that downloads Romanian monthly HICP (Eurostat prc\\_hicp\\_minr), computes the monthly inflation rate, removes the monthly means, estimates d with GPH and local Whittle for bandwidths T\\^{}0.5 to T\\^{}0.8, fits ARFIMA(p,d,q) by exact Gaussian maximum likelihood and compares it with ARMA(1,1) by BIC.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('The library: \\texttt{statsmodels} has no ARFIMA class; an AI answer that calls one has invented it', 'Biblioteca: \\texttt{statsmodels} nu are o clasă ARFIMA; un răspuns AI care folosește una a inventat-o'),
    T('The sign and the scale: GPH regresses on $-\\log(4\\sin^2(\\lambda/2))$, so the slope is $d$, not $-2d$; R/S and DFA give $H$, not $d$', 'Semnul și scala: GPH regresează pe $-\\log(4\\sin^2(\\lambda/2))$, deci panta este $d$, nu $-2d$; R/S și DFA dau $H$, nu $d$'),
    T('The range: a stationary ARFIMA likelihood needs $d < 0.5$; difference first if $\\hat d$ is near or above 0.5', 'Intervalul: verosimilitatea unui ARFIMA staționar cere $d < 0{,}5$; diferențiați întîi dacă $\\hat d$ este aproape de 0,5 sau peste'),
    T('Simulate a series with known $d$ and check that the code recovers it', 'Simulați o serie cu $d$ cunoscut și verificați că codul îl regăsește'),
    T('Breaks, seasonality and overlapping data before any claim of long memory', 'Rupturile, sezonalitatea și datele suprapuse, înainte de orice afirmație despre memoria lungă'),
    T('Every cited reference: it must exist; check the DOI', 'Fiecare referință citată: trebuie să existe; verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Long memory: hyperbolic ACF decay $\\rho(k) \\sim Ck^{2d-1}$ and a spectral pole at zero; ARMA memory is exponential', 'Memoria lungă: descreștere hiperbolică a ACF $\\rho(k) \\sim Ck^{2d-1}$ și un pol spectral la zero; memoria ARMA este exponențială'),
    T('$(1-L)^d$ with real $d$; ARFIMA$(p,d,q)$ separates long-run memory ($d$) from short-run dynamics (ARMA)', '$(1-L)^d$ cu $d$ real; ARFIMA$(p,d,q)$ separă memoria pe termen lung ($d$) de dinamica pe termen scurt (ARMA)'),
    T('Stationary for $d < 1/2$, invertible for $d > -1/2$, mean-reverting for $d < 1$; $H = d + 1/2$', 'Staționar pentru $d < 1/2$, inversabil pentru $d > -1/2$, cu revenire la medie pentru $d < 1$; $H = d + 1/2$'),
    T('Estimate $d$ with several methods and bandwidths; prefer likelihood-based comparisons with ARMA', 'Estimați $d$ cu mai multe metode și lățimi de bandă; preferați comparațiile cu ARMA pe baza verosimilității'),
    T('Inflation and volatility show long memory; returns do not; breaks and regimes can fake it', 'Inflația și volatilitatea au memorie lungă; randamentele nu; rupturile și regimurile o pot imita')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.35}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    ['ARFIMA$(p,d,q)$ & $\\phi(L)(1-L)^d(x_t - \\mu) = \\theta(L)\\varepsilon_t$',
     T('Weights of $(1-L)^d$', 'Ponderile lui $(1-L)^d$') + ' & $\\pi_0 = 1$, \\quad $\\pi_k = \\pi_{k-1}(k - 1 - d)/k$',
     T('ACF of ARFIMA$(0,d,0)$', 'ACF pentru ARFIMA$(0,d,0)$') + ' & $\\rho(1) = d/(1-d)$, \\quad $\\rho(k) = \\rho(k-1)(k-1+d)/(k-d) \\sim Ck^{2d-1}$',
     T('Spectrum near 0', 'Spectrul în apropierea lui 0') + ' & $f(\\lambda) \\sim G\\lambda^{-2d}$',
     'GPH & $\\log I(\\lambda_j) = c + d\\,[-\\log(4\\sin^2(\\lambda_j/2))] + u_j$, \\quad SE $= \\pi/\\sqrt{24m}$',
     T('Local Whittle', 'Whittle local') + ' & $\\min_d \\log\\bigl(\\tfrac1m\\sum_j\\lambda_j^{2d}I(\\lambda_j)\\bigr) - \\tfrac{2d}{m}\\sum_j\\log\\lambda_j$, \\quad SE $= 1/(2\\sqrt m)$',
     'R/S, DFA & $E(R/S)_n \\approx cn^H$, \\quad $F(n) \\approx cn^H$, \\quad $H = d + 1/2$',
     'FIGARCH & $\\sigma_t^2 = \\omega + \\beta\\sigma_{t-1}^2 + [1 - \\beta L - (1 - \\phi L)(1-L)^d]\\varepsilon_t^2$'],
    size='scriptsize') + '}')

D.frame(T('Self-assessment', 'Autoevaluare'), items(
    (T('\\textbf{Question}: what are $\\pi_1$ and $\\pi_2$ of $(1-L)^{0.2}$?', '\\textbf{Întrebare}: cît sînt $\\pi_1$ și $\\pi_2$ pentru $(1-L)^{0{,}2}$?'),
     [T('\\textbf{Answer}: $\\pi_1 = -0.2$, $\\pi_2 = -0.2 \\cdot 0.8/2 = -0.08$', '\\textbf{Răspuns}: $\\pi_1 = -0{,}2$, $\\pi_2 = -0{,}2 \\cdot 0{,}8/2 = -0{,}08$')]),
    (T('\\textbf{Question}: is ARFIMA$(0, 0.7, 0)$ stationary?', '\\textbf{Întrebare}: este staționar un ARFIMA$(0;\\ 0{,}7;\\ 0)$?'),
     [T('\\textbf{Answer}: no ($d \\ge 0.5$), but it is mean-reverting; its first difference is ARFIMA$(0, -0.3, 0)$, stationary and anti-persistent', '\\textbf{Răspuns}: nu ($d \\ge 0{,}5$), dar revine la medie; prima ei diferență este ARFIMA$(0;\\ -0{,}3;\\ 0)$, staționar și antipersistent')]),
    (T('\\textbf{Question}: GPH gives $\\hat d = 0.35$ on a series with a large level shift. Is this evidence of long memory?', '\\textbf{Întrebare}: GPH dă $\\hat d = 0{,}35$ pentru o serie cu o schimbare mare de nivel. Este aceasta o dovadă de memorie lungă?'),
     [T('\\textbf{Answer}: not by itself: a break produces the same low-frequency power; re-estimate after removing the shift and on subsamples', '\\textbf{Răspuns}: nu, prin ea însăși: o ruptură produce aceeași putere la frecvențele joase; reestimați după eliminarea schimbării și pe subeșantioane')]),
    T('Next: Chapter 9, machine learning for time series', 'Urmează: Capitolul 9, învățarea automată pentru serii de timp')))

D.references(bib())

if __name__ == '__main__':
    finalize(D.write(V))
