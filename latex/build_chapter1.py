r"""
build_chapter1.py -- Capitolul 1 (Procese stochastice și staționaritate), EN + RO dintr-o singură sursă
=====================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_01/ch1_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter1_stochastic_processes_stationarity.tex
  RO/Cursuri/capitol1_procese_stochastice_stationaritate.tex
Rulare:
  python3 Quantlets/Ch_01/generate_all_charts.py
  python3 latex/build_chapter1.py && python3 latex/tsa_build.py compile 1
"""

import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, cols, items, table, photo, block   # noqa: E402
from ch1_common import QLURL, REFS, T, bib, date, finalize, load, pv, quarter   # noqa: E402

N = load()
V = Values()
D = Deck(1, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\quantlet{{{folder.replace("_", chr(92) + "_")}}}{{\\qlurl{{{folder}}}}}'


def chart(title, fig, folder, bullets, h='0.66\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def interp(title, bullets, size='small'):
    D.frame(T(f'Interpreting {title[0]}', f'Interpretarea {title[1]}'), items(*bullets), size)


PH = {
    'wold': ('ch1_herman_wold_1969.jpg', C + 'Professor_Herman_Wold,_Uppsala,_1969_(cropped).jpg',
             T('Photo', 'Foto') + ': Uppsala-Bild (1969); CC BY 4.0; Wikimedia Commons'),
    'bvb': ('ch1_bvb_2024.jpg', C + 'Bursa_de_Valori_București.jpg',
            T('Photo', 'Foto') + ': Corina Chitu (2024); CC BY-SA 4.0; Wikimedia Commons'),
    'bnr': ('ch1_bnr_2018.jpg', C + 'National_Bank_of_Romania_(old_building),_Bucharest_by_nickispeaki_01.jpg',
            T('Photo', 'Foto') + ': Nickispeaki (2018); CC BY-SA 4.0; Wikimedia Commons'),
    'nilo': ('ch1_nilometer_cairo.jpg', C + 'Kairo_Nilometer_BW_1.jpg',
             T('Photo', 'Foto') + ': Berthold Werner (2010); CC BY-SA 3.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.50\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# CIFRE
# =============================================================================
F = N['four']
V.put('gdp.v0', F['gdp_v0'], 1)
V.put('gdp.v1', F['gdp_v1'], 1)
V.raw('gdp.q0', quarter(F['gdp_first']))
V.raw('gdp.q1', quarter(F['gdp_last']))
V.int('gdp.n', F['gdp_n'])
V.put('infl.max', F['infl_max'], 0)
V.raw('infl.maxd', date(F['infl_max_d'], day=False))
V.put('infl.min', F['infl_min'], 1)
V.raw('infl.mind', date(F['infl_min_d'], day=False))
V.put('infl.last', F['infl_lastv'], 1)
V.raw('infl.lastd', date(F['infl_last'], day=False))
V.put('fx.v0', F['fx_v0'], 2)
V.put('fx.v1', F['fx_v1'], 2)
V.int('bet.n', F['bet_n'])
V.put('bet.sd', F['bet_sd'], 2)
V.put('bet.min', F['bet_min'], 1)
V.raw('bet.mind', date(F['bet_min_d']))
V.put('bet.max', F['bet_max'], 1)
V.raw('bet.maxd', date(F['bet_max_d']))
V.put('bet.sd0', F['bet_sd_0'], 2)
V.put('bet.sd1', F['bet_sd_1'], 2)
V.put('bet.ratio', F['bet_sd_0'] / F['bet_sd_1'], 1)
V.raw('end', date('2026-09-18'))

E = N['ensemble']
V.put('ens.arvar', E['ar_var'], 2)
V.put('ens.arsd', E['ar_sd'], 2)
V.put('ens.band', 1.96 * E['ar_sd'], 2)
V.put('ens.rwvar', E['rw_var_sim'], 1)

Cx = N['counter']
V.put('cx.skewo', Cx['skew_o'], 2)
V.put('cx.skewe', Cx['skew_e'], 2)
V.put('cx.skewth', Cx['skew_th'], 2)
V.put('cx.vare', Cx['var_e'], 2)
V.put('cx.varo', Cx['var_o'], 2)

W = N['wn']
for k, key in [('Gaussian i.i.d.', 'g'), ('Laplace i.i.d.', 'l'), ('GARCH(1,1): weak WN', 'h')]:
    V.raw(f'wn.{key}.p', pv(W[k]['lb_x']['lb_p']))
    V.raw(f'wn.{key}.p2', pv(W[k]['lb_x2']['lb_p']))
    V.put(f'wn.{key}.q2', W[k]['lb_x2']['lb'], 1)
    V.put(f'wn.{key}.k', W[k]['kurt'], 1)

R = N['rw']
V.put('rw.var', R['var_end'], 1)
V.put('rw.corr', R['corr_100_110'], 3)
V.put('rw.corrth', R['corr_th'], 3)
V.put('rw.out', 100 * R['share_out'], 1)

S = N['spot']
V.raw('sp.real', S['real'])
V.raw('sp.d0', date(S['first']))
V.raw('sp.d1', date(S['last']))
V.put('sp.mu', S['mu'], 2)
V.put('sp.sd', S['sd'], 2)
V.put('sp.rho1', S['rho1'], 2)
V.raw('sp.p', pv(S['lb10_p']))

P = N['sp500']
V.int('spx.n', P['n'])
V.put('spx.acf1p', P['acf1_p'], 3)
V.put('spx.acf50p', P['acf50_p'], 3)
V.put('spx.acf1r', P['acf1_r'], 3)
V.put('spx.mean', P['mean_r'], 3)
V.put('spx.sd', P['sd_r'], 2)

Wo = N['wold']
V.put('wold.s2', Wo['sum_psi2_08'], 2)
V.put('wold.p5', Wo['psi5_08'], 3)
V.put('wold.p10', Wo['psi10_08'], 3)

G = N['ergo']
V.put('erg.amin', G['a_min'], 2)
V.put('erg.amax', G['a_max'], 2)
V.put('erg.zmin', G['z_min'], 2)
V.put('erg.zmax', G['z_max'], 2)
V.put('erg.bmin', G['b_min'], 2)
V.put('erg.bmax', G['b_max'], 2)
V.int('erg.n', G['n'])

A = N['acfm']
V.put('am.ar1', A['AR(1)']['r1'], 2)
V.put('am.ar2', A['AR(1)']['r2'], 2)
V.put('am.ar10', A['AR(1)']['r10'], 2)
V.put('am.ma1', A['MA(1)']['r1'], 2)
V.put('am.ma2', A['MA(1)']['r2'], 2)
V.put('am.math', A['th_ma1'], 2)
V.put('am.rw1', A['Random walk']['r1'], 2)
V.put('am.rw10', A['Random walk']['r10'], 2)
V.put('am.wn2', A['White noise']['r2'], 2)
V.put('am.band', 1.96 / math.sqrt(500), 3)

B = N['bartlett']
V.put('bt.sd', B['sd_r1'], 3)
V.put('bt.mean', B['mean_r1'], 3)
V.put('bt.pnone', 100 * B['p_none'], 0)
V.put('bt.pnoneth', 100 * B['p_none_th'], 0)
V.put('bt.pone', 100 * B['p_one_plus'], 0)
V.put('bt.pth', 100 * (1 - B['p_none_th']), 0)
V.put('bt.mout', B['mean_out'], 2)
V.put('bt.band', B['band'], 3)

AP = N['acfpacf']
V.put('ap.ar1.p1', AP['ar1']['p1'], 2)
V.put('ap.ar1.p2', AP['ar1']['p2'], 2)
V.put('ap.ar2.p1', AP['ar2']['p1'], 2)
V.put('ap.ar2.p2', AP['ar2']['p2'], 2)
V.put('ap.ar2.p3', AP['ar2']['p3'], 2)
V.put('ap.ar2.r1', AP['ar2']['r1'], 2)
V.put('ap.ar2.r2', AP['ar2']['r2'], 2)
V.put('ap.ar2.rho1', AP['ar2_rho1'], 3)
V.put('ap.ar2.rho2', AP['ar2_rho2'], 3)
V.put('ap.ma1.r1', AP['ma1']['r1'], 2)
V.put('ap.ma1.p2', AP['ma1']['p2'], 2)
V.put('ap.ma1.p3', AP['ma1']['p3'], 2)
r1, r2 = AP['ar2_rho1'], AP['ar2_rho2']
V.put('ap.phi22', (r2 - r1 ** 2) / (1 - r1 ** 2), 3)
V.put('ap.r1sq', r1 ** 2, 3)

BT = N['bet']
V.int('bt2.n', BT['n'])
V.put('bet.r1', BT['r']['r1'], 3)
V.put('bet.abs1', BT['abs']['r1'], 2)
V.put('bet.abs50', BT['abs']['r50'], 2)
V.put('bet.sq1', BT['sq']['r1'], 2)
V.put('bet.band', BT['band'], 3)
V.put('bet.q', BT['r']['lb10']['lb'], 0)
V.put('bet.qa', BT['abs']['lb10']['lb'], 0)
V.put('bet.r1sq', 100 * BT['r']['r1'] ** 2, 1)

LS = N['lbsize']
for k, v in LS.items():
    V.put(f'ls.{k}.bp', 100 * v['bp'], 1)
    V.put(f'ls.{k}.lb', 100 * v['lb'], 1)

RL = N['real']
ROWS = [('S&P 500 returns', T('S\\&P 500, daily returns', 'S\\&P 500, randamente zilnice')),
        ('S&P 500 squared returns', T('S\\&P 500, squared returns', 'S\\&P 500, randamente la pătrat')),
        ('BET returns', T('BET, daily returns', 'BET, randamente zilnice')),
        ('BET squared returns', T('BET, squared returns', 'BET, randamente la pătrat')),
        ('EUR/RON returns', T('EUR/RON, daily returns', 'EUR/RON, randamente zilnice')),
        ('GDP growth, q/q', T('GDP Romania, q/q growth', 'PIB România, creștere t/t')),
        ('GDP growth, y/y', T('GDP Romania, y/y growth', 'PIB România, creștere an/an')),
        ('HICP inflation, m/m', T('HICP Romania, monthly inflation', 'IAPC România, inflația lunară')),
        ('Nile flow', T('Nile, yearly flow', 'Nil, debitul anual')),
        ('Sunspots', T('Sunspots, yearly', 'Pete solare, anual'))]
for i, (k, _) in enumerate(ROWS):
    d = RL[k]
    V.int(f'rl{i}.n', d['n'])
    V.put(f'rl{i}.r1', d['r1'], 2)
    V.put(f'rl{i}.r4', d['r4'], 2)
    V.put(f'rl{i}.q', d['m10']['lb'], 1)
    V.raw(f'rl{i}.p', pv(d['m10']['lb_p']))

GD = N['gdp']
for k in ['acf1_level', 'acf8_level', 'acf1_d1', 'acf4_d1', 'acf2_d1', 'acf1_d4', 'acf4_d4']:
    V.put('gd.' + k, GD[k], 2)
V.put('gd.mean4', GD['mean_d4'], 1)
V.put('gd.sd4', GD['sd_d4'], 1)
V.put('gd.sd1', GD['sd_d1'], 1)
V.put('gd.covid', GD['covid_d4'], 1)
V.put('gd.last4', GD['last_d4'], 1)
V.put('gd.band', GD['band'], 2)

BX = N['boxcox']
V.put('bx.lam', BX['lambda'], 2)
V.put('bx.cv1', BX['cv_1'], 2)
V.put('bx.cv0', BX['cv_0'], 3)
V.put('bx.cvl', BX['cv_lam'], 3)
V.put('bx.v0', BX['v0'], 1)
V.put('bx.v1', BX['v1'], 1)
V.put('bx.amp0', 100 * BX['amp_rel_0'], 0)
V.put('bx.amp1', 100 * BX['amp_rel_1'], 0)

OD = N['overdiff']
V.put('od.r1', OD['r1_d'], 2)
V.put('od.re', OD['r1_e'], 2)
V.put('od.vd', OD['var_d'], 2)
V.put('od.ve', OD['var_e'], 2)

TB = N['textbook']
V.put('nile.m0', TB['m0'], 0)
V.put('nile.m1', TB['m1'], 0)
V.put('nile.r1', TB['r1_nile'], 2)
V.put('nile.rr', TB['r1_resid'], 2)
V.put('nile.q', TB['lb_nile']['lb'], 1)
V.raw('nile.p', pv(TB['lb_nile']['lb_p']))
V.put('nile.qr', TB['lb_resid']['lb'], 1)
V.raw('nile.pr', pv(TB['lb_resid']['lb_p']))
V.raw('sun.lag', str(TB['sun_peak_lag']))
V.put('sun.peak', TB['sun_peak'], 2)
V.raw('sun.minlag', str(TB['sun_min_lag']))
V.put('sun.min', TB['sun_min'], 2)
V.put('sun.r1', TB['sun_r1'], 2)

# ---- worked examples (computed here)
th = 0.6
V.put('ex.ma.g0', 1 + th ** 2, 2)
V.put('ex.ma.r1', th / (1 + th ** 2), 3)
ph_ = 0.8
V.put('ex.ar.var', 1 / (1 - ph_ ** 2), 2)
V.put('ex.ar.r5', ph_ ** 5, 3)
V.put('ex.ar.r10', ph_ ** 10, 3)
xs = np.array([3., 5., 4., 6., 7.])
dv = xs - xs.mean()
g0 = np.sum(dv ** 2) / 5
g1 = np.sum(dv[:-1] * dv[1:]) / 5
g2 = np.sum(dv[:-2] * dv[2:]) / 5
V.put('sx.g0', g0, 1)
V.put('sx.g1', g1, 1)
V.put('sx.g2', g2, 1)
V.put('sx.r1', g1 / g0, 2)
V.put('sx.r2', g2 / g0, 2)
V.put('sx.band', 1.96 / math.sqrt(5), 2)
rh = np.array([0.25, 0.12, -0.08])
Tn = 100
qbp = Tn * np.sum(rh ** 2)
qlb = Tn * (Tn + 2) * np.sum(rh ** 2 / (Tn - np.arange(1, 4)))
V.put('lbx.bp', qbp, 2)
V.put('lbx.lb', qlb, 2)
V.put('lbx.crit', stats.chi2.ppf(0.95, 3), 2)
V.raw('lbx.p', pv(stats.chi2.sf(qlb, 3)))
assert qbp > stats.chi2.ppf(0.95, 3) and qlb > stats.chi2.ppf(0.95, 3)   # the text says: reject
V.put('lbx.band', 1.96 / math.sqrt(Tn), 3)
V.put('lg.03', math.log(1.03), 4)
V.put('lg.30', math.log(1.30), 3)
V.put('lg.m30', math.log(0.70), 3)
V.put('bcx.05', (100 ** 0.5 - 1) / 0.5, 0)
V.put('bcx.0', math.log(100), 3)
V.put('bcx.05b', (400 ** 0.5 - 1) / 0.5, 0)
V.put('bcx.0b', math.log(400), 3)
V.put('chi10', stats.chi2.ppf(0.95, 10), 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: we observe one path of a series, a single history; what can we learn from it about the mechanism that produced it?',
       '\\textbf{Întrebarea}: observăm o singură traiectorie a unei serii, o singură istorie; ce putem afla din ea despre mecanismul care a generat-o?'),
     [T('the answer rests on two assumptions: \\textbf{stationarity} and \\textbf{ergodicity}', 'răspunsul se sprijină pe două ipoteze: \\textbf{staționaritatea} și \\textbf{ergodicitatea}'),
      T('the main tool: the \\textbf{autocorrelation function} (ACF)', 'instrumentul principal: \\textbf{funcția de autocorelație} (ACF)')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('stochastic processes and their moments; strict and weak stationarity', 'procese stochastice și momentele lor; staționaritate strictă și slabă'),
      T('white noise and the random walk; the lag operator, differencing, the Wold decomposition, ergodicity', 'zgomotul alb și mersul aleator; operatorul lag, diferențierea, descompunerea Wold, ergodicitatea'),
      T('sample ACF and PACF with confidence bands; the Box--Pierce and Ljung--Box tests', 'ACF și PACF de selecție cu benzi de încredere; testele Box--Pierce și Ljung--Box'),
      T('transformations (log, differencing, Box--Cox) and a first look at real series', 'transformări (logaritm, diferențiere, Box--Cox) și o primă analiză a unor serii reale')]),
    T('Seminar 1 comes before this lecture: its primer gives the definitions; here we derive and explain them',
      'Seminarul 1 are loc înaintea acestui curs: secțiunea „Noțiuni necesare azi” dă definițiile; aici le deducem și le explicăm')))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Define a stochastic process and its mean, autocovariance and autocorrelation functions', 'Definiți un proces stochastic și funcțiile lui de medie, autocovarianță și autocorelație'),
    T('Distinguish strict from weak stationarity, and check weak stationarity for simple processes', 'Deosebiți staționaritatea strictă de cea slabă și verificați staționaritatea slabă pentru procese simple'),
    T('Derive the moments of white noise, of a moving average and of a random walk', 'Deduceți momentele zgomotului alb, ale unei medii mobile și ale unui mers aleator'),
    T('Compute and read the sample ACF and PACF with their $\\pm 1.96/\\sqrt{T}$ bands', 'Calculați și interpretați ACF și PACF de selecție cu benzile $\\pm 1{,}96/\\sqrt{T}$'),
    T('Test for autocorrelation with the Box--Pierce and Ljung--Box statistics', 'Testați autocorelația cu statisticile Box--Pierce și Ljung--Box'),
    T('Choose a transformation (log, difference, Box--Cox) that makes a real series closer to stationary', 'Alegeți o transformare (logaritm, diferență, Box--Cox) care apropie o serie reală de staționaritate')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refHP, Ch.~1--2 (concepts, exploratory analysis)', 'Manual: \\refHP, cap.~1--2 (concepte, analiză exploratorie)'),
     [T('companion, free online: \\refFPP, Ch.~2 (time series graphics) and Ch.~3 (transformations)', 'manual însoțitor, gratuit online: \\refFPP, cap.~2 (grafice pentru serii de timp) și cap.~3 (transformări)')]),
    T('Theory: \\refBD, Ch.~1; \\refHamilton, Ch.~3', 'Teorie: \\refBD, cap.~1; \\refHamilton, cap.~3'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_01}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_01}'),
     [T('ACF, PACF and portmanteau tests with \\texttt{statsmodels}', 'ACF, PACF și teste portmanteau cu \\texttt{statsmodels}')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter1_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter1_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. PROCESE STOCHASTICE
# =============================================================================
D.section('Time series and stochastic processes', 'Serii de timp și procese stochastice')

chart(T('Four series of this course', 'Patru serii ale acestui curs'), 'tsa_ch1_four_series', 'TSA_ch1_real_series', [
    T('Romanian real GDP, @{gdp.q0}--@{gdp.q1} (Eurostat); HICP inflation (Eurostat); EUR/RON, the BNR reference rate; BET daily log returns since 2000',
      'PIB-ul real al României, @{gdp.q0}--@{gdp.q1} (Eurostat); inflația IAPC (Eurostat); cursul EUR/RON, cursul de referință BNR; randamentele logaritmice zilnice ale BET din 2000'),
    T('Four frequencies (quarterly, monthly, daily) and four kinds of behaviour', 'Patru frecvențe (trimestrială, lunară, zilnică) și patru tipuri de comportament')],
    h='0.66\\textheight')

interp(('the four series', 'celor patru serii'), [
    (T('\\textbf{GDP}: from @{gdp.v0} to @{gdp.v1} bn EUR per quarter; a trend and a seasonal pattern whose swing grows with the level',
       '\\textbf{PIB}: de la @{gdp.v0} la @{gdp.v1} mld. EUR pe trimestru; un trend și o sezonalitate a cărei amplitudine crește odată cu nivelul'),
     [T('the mean of the series changes with time', 'media seriei se schimbă în timp')]),
    (T('\\textbf{Inflation}: @{infl.max}\\% in @{infl.maxd}, @{infl.min}\\% in @{infl.mind}, @{infl.last}\\% in @{infl.lastd}',
       '\\textbf{Inflația}: @{infl.max}\\% în @{infl.maxd}, @{infl.min}\\% în @{infl.mind}, @{infl.last}\\% în @{infl.lastd}'),
     [T('a slow, persistent movement: this month resembles the previous one', 'o mișcare lentă, persistentă: luna aceasta seamănă cu luna trecută')]),
    (T('\\textbf{EUR/RON}: from @{fx.v0} to @{fx.v1} lei; it wanders without returning to a fixed level', '\\textbf{EUR/RON}: de la @{fx.v0} la @{fx.v1} lei; evoluează fără să revină la un nivel fix'),
     [T('a candidate for a random walk (Section 3)', 'un candidat pentru un mers aleator (secțiunea 3)')]),
    (T('\\textbf{BET returns}: they oscillate around a constant level close to 0; the extremes: $@{bet.min}\\%$ on @{bet.mind}, $+@{bet.max}\\%$ on @{bet.maxd}',
       '\\textbf{Randamentele BET}: oscilează în jurul unui nivel constant, apropiat de 0; extremele: $@{bet.min}\\%$ pe @{bet.mind}, $+@{bet.max}\\%$ pe @{bet.maxd}'),
     [T('but the amplitude changes: standard deviation @{bet.sd0}\\% in September 2008--March 2009, @{bet.sd1}\\% in 2017',
        'dar amplitudinea se schimbă: abaterea standard este @{bet.sd0}\\% în septembrie 2008--martie 2009 și @{bet.sd1}\\% în 2017')])])

D.frame(T('A time series as one realisation', 'Seria de timp ca o realizare'), items(
    (T('\\textbf{Time series}: observations $x_1, x_2, \\dots, x_T$ ordered in time, at equal intervals', '\\textbf{Serie de timp}: observații $x_1, x_2, \\dots, x_T$ ordonate în timp, la intervale egale'),
     [T('$T$ = sample size; $t$ = time index (day, month, quarter)', '$T$ = volumul eșantionului; $t$ = indicele de timp (zi, lună, trimestru)')]),
    (T('The key idea: each $x_t$ is the observed value of a \\textbf{random variable} $X_t$', 'Ideea de bază: fiecare $x_t$ este valoarea observată a unei \\textbf{variabile aleatoare} $X_t$'),
     [T('the BET return of tomorrow is unknown today: it has a distribution of possible values', 'randamentul BET de mîine este necunoscut azi: are o distribuție de valori posibile'),
      T('history ran only once: we see one value from each distribution', 'istoria s-a desfășurat o singură dată: vedem cîte o singură valoare din fiecare distribuție')]),
    (T('Consequence: we need assumptions that link the distributions at different dates', 'Consecința: avem nevoie de ipoteze care leagă distribuțiile de la date diferite'),
     [T('otherwise $T$ observations would estimate $T$ different means', 'altfel, $T$ observații ar estima $T$ medii diferite')])))

D.frame(T('Stochastic process', 'Procesul stochastic'), items(
    (T('\\textbf{Definition}: a \\textbf{stochastic process} is a family of random variables $\\{X_t : t \\in \\mathcal{T}\\}$ defined on the same probability space $(\\Omega, \\mathcal{F}, P)$',
       '\\textbf{Definiție}: un \\textbf{proces stochastic} este o familie de variabile aleatoare $\\{X_t : t \\in \\mathcal{T}\\}$ definite pe același spațiu de probabilitate $(\\Omega, \\mathcal{F}, P)$'),
     [T('$\\mathcal{T} = \\mathbb{Z}$ or $\\{1, \\dots, T\\}$: \\textbf{discrete time}, the case of this course', '$\\mathcal{T} = \\mathbb{Z}$ sau $\\{1, \\dots, T\\}$: \\textbf{timp discret}, cazul acestui curs'),
      T('$\\Omega$: the set of possible ``histories\'\' $\\omega$; $P$: their probabilities', '$\\Omega$: mulțimea „istoriilor” posibile $\\omega$; $P$: probabilitățile lor')]),
    (T('Two ways to look at $X_t(\\omega)$', 'Două moduri de a privi $X_t(\\omega)$'),
     [T('fix $t$: $X_t$ is a random variable (all values possible at date $t$)', 'fixăm $t$: $X_t$ este o variabilă aleatoare (toate valorile posibile la data $t$)'),
      T('fix $\\omega$: $t \\mapsto X_t(\\omega)$ is a \\textbf{path} (trajectory, realisation): what we observe', 'fixăm $\\omega$: $t \\mapsto X_t(\\omega)$ este o \\textbf{traiectorie} (realizare): ceea ce observăm')]),
    T('The process is described by its \\textbf{finite-dimensional distributions}: the joint distribution of $(X_{t_1}, \\dots, X_{t_k})$ for any dates $t_1, \\dots, t_k$',
      'Procesul este descris de \\textbf{distribuțiile finit-dimensionale}: distribuția comună a lui $(X_{t_1}, \\dots, X_{t_k})$ pentru orice date $t_1, \\dots, t_k$')))

chart(T('An ensemble of paths', 'Un ansamblu de traiectorii'), 'tsa_ch1_ensemble', 'TSA_ch1_processes', [
    T('Left: 40 paths of $X_t = 0.7X_{t-1} + \\varepsilon_t$, $\\varepsilon_t \\sim N(0, 1)$ i.i.d. (independent and identically distributed); right: 40 paths of $X_t = X_{t-1} + \\varepsilon_t$, $X_0 = 0$',
      'Stînga: 40 de traiectorii ale lui $X_t = 0{,}7X_{t-1} + \\varepsilon_t$, $\\varepsilon_t \\sim N(0, 1)$ i.i.d. (independente și identic distribuite); dreapta: 40 de traiectorii ale lui $X_t = X_{t-1} + \\varepsilon_t$, $X_0 = 0$'),
    T('The shaded band: $\\pm 1.96$ standard deviations of $X_t$ at each $t$', 'Banda colorată: $\\pm 1{,}96$ abateri standard ale lui $X_t$ la fiecare $t$')],
    h='0.64\\textheight')

interp(('the ensemble', 'ansamblului de traiectorii'), [
    (T('A vertical slice at date $t$ shows the \\textbf{distribution of $X_t$}; one path shows \\textbf{one history}', 'O secțiune verticală la data $t$ arată \\textbf{distribuția lui $X_t$}; o traiectorie arată \\textbf{o singură istorie}'),
     [T('the moments of the process (mean, variance) are averages across paths, at a fixed $t$', 'momentele procesului (media, varianța) sînt medii pe traiectorii, la $t$ fixat')]),
    (T('Left: the same band at every $t$: $\\mathrm{Var}(X_t) = 1/(1 - 0.7^2) = @{ens.arvar}$, band $\\pm @{ens.band}$', 'Stînga: aceeași bandă la orice $t$: $\\mathrm{Var}(X_t) = 1/(1 - 0{,}7^2) = @{ens.arvar}$, banda $\\pm @{ens.band}$'),
     [T('the distribution of $X_t$ does not depend on $t$: a first picture of \\textbf{stationarity}', 'distribuția lui $X_t$ nu depinde de $t$: o primă imagine a \\textbf{staționarității}')]),
    (T('Right: the band widens; across 5000 simulated paths, $\\mathrm{Var}(X_{100}) = @{ens.rwvar}$ (theory: 100)', 'Dreapta: banda se lărgește; pe 5000 de traiectorii simulate, $\\mathrm{Var}(X_{100}) = @{ens.rwvar}$ (teoretic: 100)'),
     [T('the distribution changes with $t$: the random walk is \\textbf{not stationary}', 'distribuția se schimbă cu $t$: mersul aleator \\textbf{nu este staționar}')])])

D.frame(T('Mean, autocovariance and autocorrelation functions', 'Funcțiile de medie, autocovarianță și autocorelație'), items(
    (T('For a process with $E[X_t^2] < \\infty$ for every $t$:', 'Pentru un proces cu $E[X_t^2] < \\infty$ pentru orice $t$:'),
     [T('\\textbf{mean function}: $\\mu_t = E[X_t]$', '\\textbf{funcția de medie}: $\\mu_t = E[X_t]$'),
      T('\\textbf{autocovariance function}: $\\gamma(t, s) = \\mathrm{Cov}(X_t, X_s) = E[(X_t - \\mu_t)(X_s - \\mu_s)]$', '\\textbf{funcția de autocovarianță}: $\\gamma(t, s) = \\mathrm{Cov}(X_t, X_s) = E[(X_t - \\mu_t)(X_s - \\mu_s)]$'),
      T('\\textbf{autocorrelation function}: $\\rho(t, s) = \\gamma(t, s)/\\sqrt{\\gamma(t, t)\\,\\gamma(s, s)}$', '\\textbf{funcția de autocorelație}: $\\rho(t, s) = \\gamma(t, s)/\\sqrt{\\gamma(t, t)\\,\\gamma(s, s)}$')]),
    (T('``Auto\'\': the covariance of the series with itself at two dates', '„Auto”: covarianța seriei cu ea însăși la două date'),
     [T('$\\gamma(t, t) = \\mathrm{Var}(X_t)$; $\\rho(t, t) = 1$; $|\\rho(t, s)| \\le 1$ (Cauchy--Schwarz inequality)', '$\\gamma(t, t) = \\mathrm{Var}(X_t)$; $\\rho(t, t) = 1$; $|\\rho(t, s)| \\le 1$ (inegalitatea Cauchy--Schwarz)')]),
    (T('\\textbf{Example}: $X_t = a + bt + \\varepsilon_t$, $\\varepsilon_t$ i.i.d. with mean 0 and variance $\\sigma^2$', '\\textbf{Exemplu}: $X_t = a + bt + \\varepsilon_t$, $\\varepsilon_t$ i.i.d. cu media 0 și varianța $\\sigma^2$'),
     [T('$\\mu_t = a + bt$ depends on $t$; $\\gamma(t, t) = \\sigma^2$, $\\gamma(t, s) = 0$ for $t \\ne s$', '$\\mu_t = a + bt$ depinde de $t$; $\\gamma(t, t) = \\sigma^2$, $\\gamma(t, s) = 0$ pentru $t \\ne s$')])))

D.recap(('Time series and stochastic processes', 'serii de timp și procese stochastice'), [
    T('A time series is one path of a stochastic process $\\{X_t\\}$', 'O serie de timp este o traiectorie a unui proces stochastic $\\{X_t\\}$'),
    T('Moments are averages across paths at fixed dates: $\\mu_t$, $\\gamma(t, s)$, $\\rho(t, s)$', 'Momentele sînt medii pe traiectorii, la date fixate: $\\mu_t$, $\\gamma(t, s)$, $\\rho(t, s)$'),
    T('With one path, we must assume these functions do not change with time', 'Cu o singură traiectorie, trebuie să presupunem că aceste funcții nu se schimbă în timp')])

# =============================================================================
# 2. STAȚIONARITATE
# =============================================================================
D.section('Stationarity', 'Staționaritate')

D.frame(T('Strict stationarity', 'Staționaritatea strictă'), items(
    (T('\\textbf{Definition}: $\\{X_t\\}$ is \\textbf{strictly stationary} if, for every $k \\ge 1$, all dates $t_1, \\dots, t_k$ and every shift $h$:',
       '\\textbf{Definiție}: $\\{X_t\\}$ este \\textbf{strict staționar} dacă, pentru orice $k \\ge 1$, orice date $t_1, \\dots, t_k$ și orice deplasare $h$:'),
     [T('$(X_{t_1}, \\dots, X_{t_k}) \\overset{d}{=} (X_{t_1 + h}, \\dots, X_{t_k + h})$, where $\\overset{d}{=}$ means equal joint distributions',
        '$(X_{t_1}, \\dots, X_{t_k}) \\overset{d}{=} (X_{t_1 + h}, \\dots, X_{t_k + h})$, unde $\\overset{d}{=}$ înseamnă distribuții comune egale')]),
    (T('Consequences', 'Consecințe'),
     [T('$k = 1$: all $X_t$ have the same distribution', '$k = 1$: toate variabilele $X_t$ au aceeași distribuție'),
      T('$k = 2$: the joint distribution of $(X_t, X_{t+h})$ depends only on the lag $h$', '$k = 2$: distribuția comună a lui $(X_t, X_{t+h})$ depinde doar de lagul $h$')]),
    (T('Example: an i.i.d. sequence is strictly stationary', 'Exemplu: un șir i.i.d. este strict staționar'),
     [T('the definition concerns entire distributions; the moments need not even exist (an i.i.d. Cauchy sequence)', 'definiția privește distribuții întregi; momentele nici nu trebuie să existe (un șir i.i.d. Cauchy)')]),
    T('Hard to check from data: we would need all joint distributions', 'Greu de verificat din date: ar trebui cunoscute toate distribuțiile comune')))

D.frame(T('Weak stationarity', 'Staționaritatea slabă'), items(
    (T('\\textbf{Definition} \\refKhinchin: $\\{X_t\\}$ is \\textbf{weakly stationary} (covariance stationary, second-order stationary) if',
       '\\textbf{Definiție} \\refKhinchin: $\\{X_t\\}$ este \\textbf{slab staționar} (staționar în covarianță, staționar de ordinul doi) dacă'),
     [T('(i) $E[X_t^2] < \\infty$ for every $t$', '(i) $E[X_t^2] < \\infty$ pentru orice $t$'),
      T('(ii) $E[X_t] = \\mu$, the same for every $t$', '(ii) $E[X_t] = \\mu$, aceeași pentru orice $t$'),
      T('(iii) $\\mathrm{Cov}(X_t, X_{t+h}) = \\gamma(h)$ depends only on the lag $h$, not on $t$', '(iii) $\\mathrm{Cov}(X_t, X_{t+h}) = \\gamma(h)$ depinde doar de lagul $h$, nu de $t$')]),
    (T('Then the functions have one argument, the \\textbf{lag} $h$:', 'Atunci funcțiile au un singur argument, \\textbf{lagul} $h$:'),
     [T('$\\gamma(h) = \\mathrm{Cov}(X_t, X_{t+h})$, $\\gamma(0) = \\mathrm{Var}(X_t)$', '$\\gamma(h) = \\mathrm{Cov}(X_t, X_{t+h})$, $\\gamma(0) = \\mathrm{Var}(X_t)$'),
      T('$\\rho(h) = \\gamma(h)/\\gamma(0)$: the \\textbf{ACF} (autocorrelation function)', '$\\rho(h) = \\gamma(h)/\\gamma(0)$: \\textbf{ACF} (funcția de autocorelație)')]),
    T('In this course, ``stationary\'\' means weakly stationary unless stated otherwise', 'În acest curs, „staționar” înseamnă slab staționar, dacă nu se precizează altfel')))

D.frame(T('Strict and weak stationarity compared', 'Staționaritatea strictă și cea slabă'), items(
    (T('\\textbf{Strict + finite variance $\\Rightarrow$ weak}', '\\textbf{Strictă + varianță finită $\\Rightarrow$ slabă}'),
     [T('equal distributions give equal means; equal joint distributions of $(X_t, X_{t+h})$ give equal covariances', 'distribuții egale dau medii egale; distribuții comune egale ale lui $(X_t, X_{t+h})$ dau covarianțe egale')]),
    (T('\\textbf{Weak $\\not\\Rightarrow$ strict}', '\\textbf{Slabă $\\not\\Rightarrow$ strictă}'),
     [T('weak stationarity fixes only the first two moments; the shape of the distribution may change (next slide)', 'staționaritatea slabă fixează doar primele două momente; forma distribuției se poate schimba (slide-ul următor)')]),
    (T('\\textbf{Strict $\\not\\Rightarrow$ weak} if the variance is infinite', '\\textbf{Strictă $\\not\\Rightarrow$ slabă} dacă varianța este infinită'),
     [T('i.i.d. Student-t with 2 degrees of freedom: strictly stationary, no finite variance', 'șir i.i.d. Student-t cu 2 grade de libertate: strict staționar, fără varianță finită')]),
    (T('\\textbf{Gaussian processes}: all finite-dimensional distributions are multivariate Normal', '\\textbf{Procese gaussiene}: toate distribuțiile finit-dimensionale sînt Normale multivariate'),
     [T('a Normal distribution is fixed by its means and covariances, so for them weak $\\Leftrightarrow$ strict', 'o distribuție Normală este determinată de medii și covarianțe, deci pentru ele slabă $\\Leftrightarrow$ strictă')])))

chart(T('Weakly but not strictly stationary', 'Slab, dar nu strict staționar'), 'tsa_ch1_counterexample', 'TSA_ch1_processes', [
    T('Independent $X_t$: $N(0, 1)$ for even $t$, $(\\chi^2_5 - 5)/\\sqrt{10}$ for odd $t$; both have mean 0 and variance 1',
      '$X_t$ independente: $N(0, 1)$ pentru $t$ par, $(\\chi^2_5 - 5)/\\sqrt{10}$ pentru $t$ impar; ambele au media 0 și varianța 1'),
    T('Sample of 4000: variance @{cx.vare} (even) and @{cx.varo} (odd); skewness $@{cx.skewe}$ and $@{cx.skewo}$ (theory: 0 and @{cx.skewth})',
      'Eșantion de 4000: varianța @{cx.vare} (par) și @{cx.varo} (impar); asimetria $@{cx.skewe}$ și $@{cx.skewo}$ (teoretic: 0 și @{cx.skewth})')],
    h='0.60\\textheight')

interp(('the counterexample', 'contraexemplului'), [
    (T('Weakly stationary: $E[X_t] = 0$, $\\gamma(0) = 1$, $\\gamma(h) = 0$ for $h \\ne 0$, at every $t$', 'Slab staționar: $E[X_t] = 0$, $\\gamma(0) = 1$, $\\gamma(h) = 0$ pentru $h \\ne 0$, la orice $t$'),
     [T('the ACF cannot tell this process from Gaussian white noise', 'ACF nu poate deosebi acest proces de zgomotul alb gaussian')]),
    (T('Not strictly stationary: the distribution of $X_t$ alternates between symmetric and right-skewed', 'Nu este strict staționar: distribuția lui $X_t$ alternează între una simetrică și una asimetrică la dreapta'),
     [T('$P(X_t < -1.6)$: about 5\\% for even $t$, 0 for odd $t$ ($X_t \\ge -5/\\sqrt{10} = -1.58$)', '$P(X_t < -1{,}6)$: circa 5\\% pentru $t$ par, 0 pentru $t$ impar ($X_t \\ge -5/\\sqrt{10} = -1{,}58$)')]),
    T('Lesson: second moments summarise dependence, but not the whole distribution; tail risk needs more (Chapter 5)', 'Lecția: momentele de ordinul doi rezumă dependența, dar nu întreaga distribuție; riscul din cozi cere mai mult (Capitolul 5)')])

D.frame(T('Properties of the autocovariance function', 'Proprietățile funcției de autocovarianță'), items(
    (T('For a weakly stationary process:', 'Pentru un proces slab staționar:'),
     [T('\\textbf{symmetry}: $\\gamma(-h) = \\gamma(h)$, since $\\mathrm{Cov}(X_t, X_{t-h}) = \\mathrm{Cov}(X_{t-h}, X_t)$', '\\textbf{simetrie}: $\\gamma(-h) = \\gamma(h)$, deoarece $\\mathrm{Cov}(X_t, X_{t-h}) = \\mathrm{Cov}(X_{t-h}, X_t)$'),
      T('\\textbf{bound}: $|\\gamma(h)| \\le \\gamma(0)$, so $|\\rho(h)| \\le 1$', '\\textbf{mărginire}: $|\\gamma(h)| \\le \\gamma(0)$, deci $|\\rho(h)| \\le 1$'),
      T('\\textbf{non-negative definiteness}: $\\sum_{i,j} a_i a_j \\gamma(i - j) \\ge 0$ for all real $a_1, \\dots, a_n$', '\\textbf{pozitiv semidefinire}: $\\sum_{i,j} a_i a_j \\gamma(i - j) \\ge 0$ pentru orice numere reale $a_1, \\dots, a_n$')]),
    (T('Proof of the third property', 'Demonstrația celei de-a treia proprietăți'),
     [T('$0 \\le \\mathrm{Var}\\big(\\sum_i a_i X_i\\big) = \\sum_{i,j} a_i a_j \\mathrm{Cov}(X_i, X_j) = \\sum_{i,j} a_i a_j \\gamma(i - j)$', '$0 \\le \\mathrm{Var}\\big(\\sum_i a_i X_i\\big) = \\sum_{i,j} a_i a_j \\mathrm{Cov}(X_i, X_j) = \\sum_{i,j} a_i a_j \\gamma(i - j)$')]),
    (T('Consequence: not every sequence is an ACF', 'Consecință: nu orice șir este o ACF'),
     [T('$\\rho(1) = 0.9$, $\\rho(2) = -0.9$ is impossible: with $a = (1, -1, -1)$, $\\mathrm{Var}(X_1 - X_2 - X_3) = \\gamma(0)(3 - 2\\rho(1) - 2\\rho(1) + 2\\rho(2)) < 0$',
        '$\\rho(1) = 0{,}9$, $\\rho(2) = -0{,}9$ este imposibil: cu $a = (1, -1, -1)$, $\\mathrm{Var}(X_1 - X_2 - X_3) = \\gamma(0)(3 - 2\\rho(1) - 2\\rho(1) + 2\\rho(2)) < 0$')])))

D.frame(T('Worked example: a moving average of order 1', 'Exemplu rezolvat: o medie mobilă de ordinul 1'), items(
    (T('$X_t = \\varepsilon_t + \\theta\\varepsilon_{t-1}$, $\\{\\varepsilon_t\\}$ uncorrelated, mean 0, variance $\\sigma^2$: the \\textbf{MA(1)} (moving average) process',
       '$X_t = \\varepsilon_t + \\theta\\varepsilon_{t-1}$, $\\{\\varepsilon_t\\}$ necorelate, media 0, varianța $\\sigma^2$: procesul \\textbf{MA(1)} (medie mobilă)'),
     [T('$\\theta$: the weight of the previous shock; $\\varepsilon_t$: the shock of period $t$', '$\\theta$: ponderea șocului din perioada anterioară; $\\varepsilon_t$: șocul din perioada $t$'),
      T('mean: $E[X_t] = 0$', 'media: $E[X_t] = 0$'),
      T('$\\gamma(0) = E[(\\varepsilon_t + \\theta\\varepsilon_{t-1})^2] = \\sigma^2(1 + \\theta^2)$ (the cross term has mean 0)', '$\\gamma(0) = E[(\\varepsilon_t + \\theta\\varepsilon_{t-1})^2] = \\sigma^2(1 + \\theta^2)$ (termenul încrucișat are media 0)'),
      T('$\\gamma(1) = E[(\\varepsilon_t + \\theta\\varepsilon_{t-1})(\\varepsilon_{t+1} + \\theta\\varepsilon_t)] = \\theta\\sigma^2$', '$\\gamma(1) = E[(\\varepsilon_t + \\theta\\varepsilon_{t-1})(\\varepsilon_{t+1} + \\theta\\varepsilon_t)] = \\theta\\sigma^2$'),
      T('$\\gamma(h) = 0$ for $|h| \\ge 2$: no common shock', '$\\gamma(h) = 0$ pentru $|h| \\ge 2$: nu există niciun șoc comun')]),
    (T('None of these depends on $t$: the MA(1) is weakly stationary for every $\\theta$', 'Niciuna nu depinde de $t$: MA(1) este slab staționar pentru orice $\\theta$'),
     [T('$\\rho(1) = \\theta/(1 + \\theta^2)$; with $\\theta = 0.6$, $\\sigma^2 = 1$: $\\gamma(0) = @{ex.ma.g0}$, $\\rho(1) = @{ex.ma.r1}$', '$\\rho(1) = \\theta/(1 + \\theta^2)$; cu $\\theta = 0{,}6$, $\\sigma^2 = 1$: $\\gamma(0) = @{ex.ma.g0}$, $\\rho(1) = @{ex.ma.r1}$'),
      T('$|\\rho(1)| \\le 0.5$ for any $\\theta$ (maximum at $\\theta = 1$)', '$|\\rho(1)| \\le 0{,}5$ pentru orice $\\theta$ (maximul la $\\theta = 1$)')])))

D.frame(T('Worked example: an autoregression of order 1', 'Exemplu rezolvat: o autoregresie de ordinul 1'), items(
    (T('$X_t = \\phi X_{t-1} + \\varepsilon_t$, $|\\phi| < 1$: the \\textbf{AR(1)} (autoregressive) process; assume it is stationary and solve for the moments',
       '$X_t = \\phi X_{t-1} + \\varepsilon_t$, $|\\phi| < 1$: procesul \\textbf{AR(1)} (autoregresiv); presupunem că este staționar și aflăm momentele'),
     [T('$\\phi$: the weight of the previous value; $\\varepsilon_t$: uncorrelated shocks with mean 0 and variance $\\sigma^2$', '$\\phi$: ponderea valorii din perioada anterioară; $\\varepsilon_t$: șocuri necorelate, cu media 0 și varianța $\\sigma^2$'),
      T('mean: $\\mu = \\phi\\mu + 0$, so $\\mu = 0$', 'media: $\\mu = \\phi\\mu + 0$, deci $\\mu = 0$'),
      T('variance: $\\gamma(0) = \\phi^2\\gamma(0) + \\sigma^2$, so $\\gamma(0) = \\sigma^2/(1 - \\phi^2)$', 'varianța: $\\gamma(0) = \\phi^2\\gamma(0) + \\sigma^2$, deci $\\gamma(0) = \\sigma^2/(1 - \\phi^2)$'),
      T('multiply by $X_{t-h}$ and take expectations: $\\gamma(h) = \\phi\\gamma(h - 1)$, so $\\rho(h) = \\phi^{|h|}$', 'înmulțim cu $X_{t-h}$ și aplicăm media: $\\gamma(h) = \\phi\\gamma(h - 1)$, deci $\\rho(h) = \\phi^{|h|}$')]),
    (T('Numbers: $\\phi = 0.8$, $\\sigma^2 = 1$', 'Valori: $\\phi = 0{,}8$, $\\sigma^2 = 1$'),
     [T('$\\gamma(0) = @{ex.ar.var}$; $\\rho(1) = 0.8$, $\\rho(5) = @{ex.ar.r5}$, $\\rho(10) = @{ex.ar.r10}$: geometric decay', '$\\gamma(0) = @{ex.ar.var}$; $\\rho(1) = 0{,}8$, $\\rho(5) = @{ex.ar.r5}$, $\\rho(10) = @{ex.ar.r10}$: descreștere geometrică')]),
    T('With $|\\phi| \\ge 1$ the variance formula fails: no stationary solution of this form (Chapters 2--3)', 'Cu $|\\phi| \\ge 1$ formula varianței nu mai funcționează: nu există o soluție staționară de această formă (Capitolele 2--3)')))

chart(T('Question for the room: which series are stationary?', 'Întrebare pentru sală: care serii sînt staționare?'), 'tsa_ch1_nonstationary', 'TSA_ch1_processes', [
    T('Four simulated series, 300 observations each; the same AR(1) noise with $\\phi = 0.5$ underlies all of them', 'Patru serii simulate, cu cîte 300 de observații; la baza tuturor stă același tip de zgomot AR(1) cu $\\phi = 0{,}5$'),
    T('Which of the conditions (ii) and (iii) of weak stationarity fails for each series?', 'Care dintre condițiile (ii) și (iii) ale staționarității slabe nu este îndeplinită pentru fiecare serie?')],
    h='0.64\\textheight')

D.frame(T('Answer: only series A is stationary', 'Răspuns: doar seria A este staționară'), items(
    T('\\textbf{A}: stationary AR(1): constant mean and variance, the ACF depends only on the lag', '\\textbf{A}: AR(1) staționar: media și varianța constante, ACF depinde doar de lag'),
    (T('\\textbf{B}: deterministic trend, $\\mu_t = 0.04\\,t$: condition (ii) fails', '\\textbf{B}: trend determinist, $\\mu_t = 0{,}04\\,t$: condiția (ii) nu este îndeplinită'),
     [T('a \\textbf{trend-stationary} series: stationary around a line, after subtracting the trend', 'o serie \\textbf{staționară în jurul trendului}: staționară în jurul unei drepte, după eliminarea trendului')]),
    T('\\textbf{C}: the standard deviation grows @{nonst.sd1} times along the sample: condition (iii) fails at $h = 0$', '\\textbf{C}: abaterea standard crește de @{nonst.sd1} ori de-a lungul eșantionului: condiția (iii) nu este îndeplinită la $h = 0$'),
    (T('\\textbf{D}: a level shift of $+@{nonst.shift}$ at $t = 150$: condition (ii) fails', '\\textbf{D}: un salt de nivel de $+@{nonst.shift}$ la $t = 150$: condiția (ii) nu este îndeplinită'),
     [T('a structural break; in real data: a change of regime or of policy (the Nile, Section 8)', 'o ruptură structurală; în date reale: o schimbare de regim sau de politică (Nilul, secțiunea 8)')]),
    T('In practice, a plot of the series is the first check; formal tests come in Chapter 3', 'În practică, graficul seriei este prima verificare; testele formale apar în Capitolul 3')))
V.put('nonst.sd0', 0.4, 1)
V.put('nonst.sd1', 3.0 / 0.4, 1)
V.put('nonst.shift', 4.0, 0)

D.recap(('Stationarity', 'staționaritate'), [
    T('Strict: all joint distributions are shift-invariant; weak: constant mean, $\\gamma$ depends only on the lag', 'Strictă: toate distribuțiile comune sînt invariante la deplasare; slabă: media constantă, $\\gamma$ depinde doar de lag'),
    T('Strict + finite variance $\\Rightarrow$ weak; for Gaussian processes they coincide', 'Strictă + varianță finită $\\Rightarrow$ slabă; pentru procesele gaussiene coincid'),
    T('MA(1) is always stationary; AR(1) is stationary for $|\\phi| < 1$, with $\\rho(h) = \\phi^{|h|}$', 'MA(1) este întotdeauna staționar; AR(1) este staționar pentru $|\\phi| < 1$, cu $\\rho(h) = \\phi^{|h|}$'),
    T('Trends, changing variance and breaks violate stationarity', 'Trendurile, varianța variabilă și rupturile încalcă staționaritatea')])

# =============================================================================
# 3. ZGOMOT ALB ȘI MERS ALEATOR
# =============================================================================
D.section('White noise and the random walk', 'Zgomotul alb și mersul aleator')

D.frame(T('White noise', 'Zgomotul alb'), items(
    (T('\\textbf{Definition}: $\\{\\varepsilon_t\\}$ is \\textbf{white noise}, $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$, if', '\\textbf{Definiție}: $\\{\\varepsilon_t\\}$ este \\textbf{zgomot alb}, $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$, dacă'),
     [T('$E[\\varepsilon_t] = 0$, $\\mathrm{Var}(\\varepsilon_t) = \\sigma^2$ and $\\mathrm{Cov}(\\varepsilon_t, \\varepsilon_s) = 0$ for $t \\ne s$', '$E[\\varepsilon_t] = 0$, $\\mathrm{Var}(\\varepsilon_t) = \\sigma^2$ și $\\mathrm{Cov}(\\varepsilon_t, \\varepsilon_s) = 0$ pentru $t \\ne s$'),
      T('so $\\rho(0) = 1$ and $\\rho(h) = 0$ for every $h \\ne 0$', 'deci $\\rho(0) = 1$ și $\\rho(h) = 0$ pentru orice $h \\ne 0$')]),
    (T('Three versions, from weakest to strongest', 'Trei variante, de la cea mai slabă la cea mai tare'),
     [T('\\textbf{weak} white noise: only uncorrelated; non-linear dependence is allowed (for example, in the squares)', 'zgomot alb \\textbf{slab}: doar necorelat; dependența neliniară este permisă (de exemplu, în pătrate)'),
      T('\\textbf{i.i.d.} (strong) white noise: independent and identically distributed', 'zgomot alb \\textbf{i.i.d.} (tare): independent și identic distribuit'),
      T('\\textbf{Gaussian} white noise: i.i.d. $N(0, \\sigma^2)$; for jointly Normal variables, uncorrelated means independent', 'zgomot alb \\textbf{gaussian}: i.i.d. $N(0, \\sigma^2)$; pentru variabile comun Normale, necorelat înseamnă independent')]),
    T('White noise is the building block of every model in this course: ARMA, ARIMA, GARCH, VAR', 'Zgomotul alb este piesa de bază a tuturor modelelor din acest curs: ARMA, ARIMA, GARCH, VAR')))

chart(T('Three kinds of white noise', 'Trei tipuri de zgomot alb'), 'tsa_ch1_white_noise', 'TSA_ch1_processes', [
    T('Top: 500 of 1000 simulated values, variance 1 in all three; bottom: the ACF of the squares $X_t^2$', 'Sus: 500 din cele 1000 de valori simulate, varianța 1 în toate trei; jos: ACF a pătratelor $X_t^2$'),
    T('GARCH(1,1) (Chapter 5): $\\sigma_t^2 = 0.05 + 0.10\\,\\varepsilon_{t-1}^2 + 0.85\\,\\sigma_{t-1}^2$, $\\varepsilon_t = \\sigma_t z_t$; $\\sigma_t^2$: the variance of day $t$ given the past; $z_t$: i.i.d. $N(0, 1)$', 'GARCH(1,1) (Capitolul 5): $\\sigma_t^2 = 0{,}05 + 0{,}10\\,\\varepsilon_{t-1}^2 + 0{,}85\\,\\sigma_{t-1}^2$, $\\varepsilon_t = \\sigma_t z_t$; $\\sigma_t^2$: varianța din ziua $t$, condiționată de trecut; $z_t$: i.i.d. $N(0, 1)$')],
    h='0.64\\textheight')

interp(('the three white noises', 'celor trei zgomote albe'), [
    (T('All three are white noise by construction; Ljung--Box $Q^*(10)$ (Section 6) on $X_t$: p = @{wn.g.p} (Gaussian), @{wn.l.p} (Laplace), @{wn.h.p} (GARCH)', 'Toate trei sînt zgomot alb prin construcție; testul Ljung--Box $Q^*(10)$ (secțiunea 6) pentru $X_t$: p = @{wn.g.p} (gaussian), @{wn.l.p} (Laplace), @{wn.h.p} (GARCH)'),
     [T('the GARCH series is close to rejection although it is uncorrelated: with changing variance the usual test rejects too often', 'seria GARCH este aproape de respingere, deși este necorelată: cînd varianța se schimbă, testul obișnuit respinge prea des')]),
    (T('Laplace: heavier tails (kurtosis @{wn.l.k} against @{wn.g.k}), but still i.i.d.: the squares are uncorrelated (p = @{wn.l.p2})',
       'Laplace: cozi mai groase (kurtosis-ul @{wn.l.k}, față de @{wn.g.k}), dar tot i.i.d.: pătratele sînt necorelate (p = @{wn.l.p2})'),
     [T('heavy tails alone do not create dependence', 'cozile groase singure nu creează dependență')]),
    (T('GARCH: the squares are strongly correlated, $Q^*(10) = @{wn.h.q2}$, p @{wn.h.p2}', 'GARCH: pătratele sînt puternic corelate, $Q^*(10) = @{wn.h.q2}$, p @{wn.h.p2}'),
     [T('a weak white noise: unpredictable sign and level, predictable size (volatility clustering)', 'un zgomot alb slab: semnul și nivelul nu pot fi anticipate, mărimea poate fi (volatility clustering)'),
      T('daily returns behave like this (BET, Section 5)', 'randamentele zilnice se comportă așa (BET, secțiunea 5)')])])

D.frame(T('The random walk', 'Mersul aleator'), items(
    (T('\\textbf{Definition}: $X_t = X_{t-1} + \\varepsilon_t$, $t \\ge 1$, $X_0 = 0$, $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$; by substitution, $X_t = \\sum_{i=1}^{t}\\varepsilon_i$',
       '\\textbf{Definiție}: $X_t = X_{t-1} + \\varepsilon_t$, $t \\ge 1$, $X_0 = 0$, $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$; prin substituție, $X_t = \\sum_{i=1}^{t}\\varepsilon_i$'),
     [T('every shock stays in the level forever: $\\partial X_{t+h}/\\partial\\varepsilon_t = 1$ for all $h \\ge 0$', 'fiecare șoc rămîne pentru totdeauna în nivel: $\\partial X_{t+h}/\\partial\\varepsilon_t = 1$ pentru orice $h \\ge 0$')]),
    (T('Moments (uncorrelated shocks are enough)', 'Momentele (sînt suficiente șocuri necorelate)'),
     [T('$E[X_t] = 0$; $\\mathrm{Var}(X_t) = \\sum_{i=1}^{t}\\mathrm{Var}(\\varepsilon_i) = t\\sigma^2$', '$E[X_t] = 0$; $\\mathrm{Var}(X_t) = \\sum_{i=1}^{t}\\mathrm{Var}(\\varepsilon_i) = t\\sigma^2$'),
      T('for $t \\le s$: $\\mathrm{Cov}(X_t, X_s) = \\mathrm{Cov}\\big(\\sum_{i \\le t}\\varepsilon_i, \\sum_{j \\le s}\\varepsilon_j\\big) = t\\sigma^2$', 'pentru $t \\le s$: $\\mathrm{Cov}(X_t, X_s) = \\mathrm{Cov}\\big(\\sum_{i \\le t}\\varepsilon_i, \\sum_{j \\le s}\\varepsilon_j\\big) = t\\sigma^2$'),
      T('$\\rho(t, s) = t/\\sqrt{ts} = \\sqrt{t/s}$', '$\\rho(t, s) = t/\\sqrt{ts} = \\sqrt{t/s}$')]),
    (T('\\textbf{Not stationary}: the variance grows with $t$, and the correlation depends on $t$, not only on $s - t$', '\\textbf{Nestaționar}: varianța crește cu $t$, iar corelația depinde de $t$, nu doar de $s - t$'),
     [T('example: $\\rho(X_{100}, X_{110}) = \\sqrt{100/110} = @{rw.corrth}$, but $\\rho(X_{10}, X_{20}) = \\sqrt{0.5} = 0.707$', 'exemplu: $\\rho(X_{100}, X_{110}) = \\sqrt{100/110} = @{rw.corrth}$, dar $\\rho(X_{10}, X_{20}) = \\sqrt{0{,}5} = 0{,}707$')])))

chart(T('Random walks and their variance', 'Mersuri aleatoare și varianța lor'), 'tsa_ch1_random_walk', 'TSA_ch1_processes', [
    T('Left: 60 paths with $\\sigma = 1$ and the band $\\pm 1.96\\sqrt{t}$; right: the variance of $X_t$ across 5000 paths and the line $t\\sigma^2$',
      'Stînga: 60 de traiectorii cu $\\sigma = 1$ și banda $\\pm 1{,}96\\sqrt{t}$; dreapta: varianța lui $X_t$ pe 5000 de traiectorii și dreapta $t\\sigma^2$'),
    T('At $t = 250$: simulated variance @{rw.var} (theory: 250); share of paths outside the band: @{rw.out}\\%', 'La $t = 250$: varianța simulată @{rw.var} (teoretic: 250); proporția traiectoriilor în afara benzii: @{rw.out}\\%')],
    h='0.62\\textheight')

interp(('the random walks', 'mersurilor aleatoare'), [
    (T('The spread grows like $\\sqrt{t}$: uncertainty about the level accumulates', 'Dispersia crește ca $\\sqrt{t}$: incertitudinea despre nivel se acumulează'),
     [T('a forecast of the level 100 days ahead has variance $100\\sigma^2$', 'o prognoză a nivelului peste 100 de zile are varianța $100\\sigma^2$')]),
    (T('Paths look like they have trends and cycles, although nothing but chance drives them', 'Traiectoriile par să aibă trenduri și cicluri, deși sînt generate exclusiv de șocuri aleatoare'),
     [T('the same lesson as \\refSlutzky: sums of random shocks can look like economic cycles', 'aceeași lecție ca la \\refSlutzky: sumele de șocuri aleatoare pot semăna cu ciclurile economice')]),
    (T('Simulated $\\rho(X_{100}, X_{110}) = @{rw.corr}$ against @{rw.corrth} in theory', 'Valoarea simulată $\\rho(X_{100}, X_{110}) = @{rw.corr}$, față de @{rw.corrth} teoretic'),
     [T('levels far apart in time are still highly correlated: very slow ACF decay', 'nivelurile depărtate în timp rămîn puternic corelate: o ACF care scade foarte încet')])])

D.frame(T('The random walk with drift', 'Mersul aleator cu derivă'), items(
    (T('$X_t = c + X_{t-1} + \\varepsilon_t$, $X_0 = 0$: $X_t = ct + \\sum_{i=1}^{t}\\varepsilon_i$; $c$ is the \\textbf{drift}', '$X_t = c + X_{t-1} + \\varepsilon_t$, $X_0 = 0$: $X_t = ct + \\sum_{i=1}^{t}\\varepsilon_i$; $c$ este \\textbf{deriva}'),
     [T('$E[X_t] = ct$: a linear trend in the mean; $\\mathrm{Var}(X_t) = t\\sigma^2$ as before', '$E[X_t] = ct$: un trend liniar în medie; $\\mathrm{Var}(X_t) = t\\sigma^2$, ca înainte')]),
    (T('The first difference is stationary: $\\Delta X_t = X_t - X_{t-1} = c + \\varepsilon_t$', 'Prima diferență este staționară: $\\Delta X_t = X_t - X_{t-1} = c + \\varepsilon_t$'),
     [T('a \\textbf{difference-stationary} series, with a \\textbf{stochastic trend}', 'o serie \\textbf{staționară în diferențe}, cu \\textbf{trend stochastic}')]),
    (T('Compare with the trend-stationary $Y_t = a + bt + u_t$, $u_t$ stationary; $a$: intercept, $b$: slope of the trend', 'Comparație cu seria staționară în jurul trendului $Y_t = a + bt + u_t$, $u_t$ staționar; $a$: termenul liber, $b$: panta trendului'),
     [T('both have a linear mean; in $Y_t$ shocks fade, in $X_t$ they last forever', 'ambele au o medie liniară; în $Y_t$ șocurile se sting, în $X_t$ rămîn pentru totdeauna'),
      T('telling the two apart is the unit-root problem of Chapter 3', 'deosebirea celor două este problema rădăcinii unitare din Capitolul 3')]),
    T('Log prices of stocks and indices are often close to a random walk with a small positive drift', 'Logaritmii prețurilor acțiunilor și ai indicilor sînt adesea apropiați de un mers aleator cu o derivă mică și pozitivă')))

D.frame(T('The Bucharest Stock Exchange', 'Bursa de Valori București'), cols(
    ph('bvb', T('The Bucharest Stock Exchange (BVB), 2024', 'Bursa de Valori București (BVB), 2024'), h='0.50\\textheight'),
    items(T('The BET index: the reference index of the BVB (Bucharest Stock Exchange), computed since 19 September 1997', 'Indicele BET: indicele de referință al BVB (Bursa de Valori București), calculat din 19 septembrie 1997'),
          T('Daily closes since 2000: @{bet.n} log returns $r_t = 100\\,\\Delta\\ln P_t$ ($P_t$: the closing level on day $t$); standard deviation @{bet.sd}\\% per day', 'Închideri zilnice din 2000: @{bet.n} randamente logaritmice $r_t = 100\\,\\Delta\\ln P_t$ ($P_t$: nivelul de închidere din ziua $t$); abaterea standard @{bet.sd}\\% pe zi'),
          T('Is the BET log price a random walk? A first, visual test on the next slide', 'Este logaritmul prețului BET un mers aleator? Un prim test, vizual, pe slide-ul următor')),
    wl='0.50', wr='0.46'))

chart(T('Question for the room: which one is real?', 'Întrebare pentru sală: care serie este cea reală?'), 'tsa_ch1_spot_real', 'TSA_ch1_processes', [
    T('One panel: the BET log price over its last 500 trading days; five panels: random walks with the same drift and the same daily standard deviation',
      'Un panou: logaritmul prețului BET în ultimele 500 de zile de tranzacționare; cinci panouri: mersuri aleatoare cu aceeași derivă și aceeași abatere standard zilnică'),
    T('Which panel is the BET?', 'Care panou este BET?')], h='0.64\\textheight')

D.frame(T('Answer: series @{sp.real}', 'Răspuns: seria @{sp.real}'), items(
    (T('Series @{sp.real} is the BET, @{sp.d0} to @{sp.d1}: mean daily change @{sp.mu}\\%, standard deviation @{sp.sd}\\%', 'Seria @{sp.real} este BET, @{sp.d0}--@{sp.d1}: variația zilnică medie @{sp.mu}\\%, abaterea standard @{sp.sd}\\%'),
     [T('by eye, it cannot be told apart from the random walks', 'cu ochiul liber, nu poate fi deosebită de mersurile aleatoare')]),
    (T('The data are not exactly a random walk: the first-order autocorrelation of the daily changes is @{sp.rho1}; Ljung--Box $Q^*(10)$ p-value @{sp.p}',
       'Datele nu sînt exact un mers aleator: autocorelația de ordinul 1 a variațiilor zilnice este @{sp.rho1}; p-value-ul testului Ljung--Box $Q^*(10)$ este @{sp.p}'),
     [T('small departures, invisible in the level, show up in the ACF of the differences', 'abaterile mici, invizibile în nivel, apar în ACF a diferențelor')]),
    T('Lesson: study the changes, not the level; the tools are the ACF and the portmanteau tests (Sections 5--6)', 'Lecția: studiem variațiile, nu nivelul; instrumentele sînt ACF și testele portmanteau (secțiunile 5--6)')))

D.recap(('White noise and the random walk', 'zgomotul alb și mersul aleator'), [
    T('White noise: mean 0, constant variance, no autocorrelation; weak, i.i.d. or Gaussian', 'Zgomotul alb: media 0, varianța constantă, fără autocorelație; slab, i.i.d. sau gaussian'),
    T('Uncorrelated is not independent: the squares of a weak white noise can be correlated', 'Necorelat nu înseamnă independent: pătratele unui zgomot alb slab pot fi corelate'),
    T('Random walk: $\\mathrm{Var}(X_t) = t\\sigma^2$, $\\rho(t, s) = \\sqrt{t/s}$: not stationary; its difference is white noise', 'Mersul aleator: $\\mathrm{Var}(X_t) = t\\sigma^2$, $\\rho(t, s) = \\sqrt{t/s}$: nestaționar; diferența lui este zgomot alb'),
    T('Trend-stationary and difference-stationary series look alike but react differently to shocks', 'Seriile staționare în jurul trendului și cele staționare în diferențe arată la fel, dar reacționează diferit la șocuri')])

# =============================================================================
# 4. OPERATORUL LAG, DIFERENȚIERE, WOLD, ERGODICITATE
# =============================================================================
D.section('The lag operator, differencing and the Wold decomposition', 'Operatorul lag, diferențierea și descompunerea Wold')

D.frame(T('The lag operator', 'Operatorul lag'), items(
    (T('\\textbf{Definition}: $LX_t = X_{t-1}$; powers: $L^kX_t = X_{t-k}$, $L^0 = 1$ (also written $B$, the backshift operator)', '\\textbf{Definiție}: $LX_t = X_{t-1}$; puteri: $L^kX_t = X_{t-k}$, $L^0 = 1$ (notat și $B$, de la \\emph{backshift})'),
     [T('a constant is unchanged: $Lc = c$', 'o constantă nu se schimbă: $Lc = c$')]),
    (T('\\textbf{Lag polynomials}: $\\phi(L) = 1 - \\phi_1L - \\dots - \\phi_pL^p$', '\\textbf{Polinoame în lag}: $\\phi(L) = 1 - \\phi_1L - \\dots - \\phi_pL^p$'),
     [T('$p$: the degree (the largest lag); $\\phi_1, \\dots, \\phi_p$: real coefficients; $\\phi(L)X_t = X_t - \\phi_1X_{t-1} - \\dots - \\phi_pX_{t-p}$', '$p$: gradul (lagul cel mai mare); $\\phi_1, \\dots, \\phi_p$: coeficienți reali; $\\phi(L)X_t = X_t - \\phi_1X_{t-1} - \\dots - \\phi_pX_{t-p}$'),
      T('AR(1): $(1 - \\phi L)X_t = \\varepsilon_t$; MA(1): $X_t = (1 + \\theta L)\\varepsilon_t$', 'AR(1): $(1 - \\phi L)X_t = \\varepsilon_t$; MA(1): $X_t = (1 + \\theta L)\\varepsilon_t$'),
      T('they multiply like ordinary polynomials: $(1 - L)(1 + L) = 1 - L^2$', 'se înmulțesc ca polinoamele obișnuite: $(1 - L)(1 + L) = 1 - L^2$')]),
    (T('\\textbf{Inversion}: for $|\\phi| < 1$, $(1 - \\phi L)^{-1} = 1 + \\phi L + \\phi^2L^2 + \\dots$ (a geometric series)', '\\textbf{Inversare}: pentru $|\\phi| < 1$, $(1 - \\phi L)^{-1} = 1 + \\phi L + \\phi^2L^2 + \\dots$ (o serie geometrică)'),
     [T('so the AR(1) is $X_t = \\sum_{j \\ge 0}\\phi^j\\varepsilon_{t-j}$: an infinite moving average', 'deci AR(1) se scrie $X_t = \\sum_{j \\ge 0}\\phi^j\\varepsilon_{t-j}$: o medie mobilă infinită')]),
    T('The language of Chapters 2--4: ARMA, ARIMA and SARIMA models are written with lag polynomials', 'Limbajul Capitolelor 2--4: modelele ARMA, ARIMA și SARIMA se scriu cu polinoame în lag')))

V.put('dx.d1a', 2, 0)
D.frame(T('Differencing', 'Diferențierea'), items(
    (T('\\textbf{First difference}: $\\Delta X_t = (1 - L)X_t = X_t - X_{t-1}$; \\textbf{second}: $\\Delta^2X_t = (1 - L)^2X_t = X_t - 2X_{t-1} + X_{t-2}$',
       '\\textbf{Prima diferență}: $\\Delta X_t = (1 - L)X_t = X_t - X_{t-1}$; \\textbf{a doua}: $\\Delta^2X_t = (1 - L)^2X_t = X_t - 2X_{t-1} + X_{t-2}$'),
     [T('\\textbf{seasonal difference} with period $s$: $\\Delta_sX_t = (1 - L^s)X_t = X_t - X_{t-s}$ ($s = 4$ for quarters, $s = 12$ for months)', '\\textbf{diferența sezonieră} cu perioada $s$: $\\Delta_sX_t = (1 - L^s)X_t = X_t - X_{t-s}$ ($s = 4$ pentru trimestre, $s = 12$ pentru luni)')]),
    (T('\\textbf{Worked example}: $x = (10, 12, 15, 14, 18)$', '\\textbf{Exemplu rezolvat}: $x = (10, 12, 15, 14, 18)$'),
     [T('$\\Delta x = (2, 3, -1, 4)$; $\\Delta^2x = (1, -4, 5)$; each difference loses one observation', '$\\Delta x = (2, 3, -1, 4)$; $\\Delta^2x = (1, -4, 5)$; fiecare diferențiere pierde o observație')]),
    (T('What differencing removes', 'Efectele diferențierii'),
     [T('a linear trend: $\\Delta(a + bt) = b$; a quadratic trend needs $\\Delta^2$', 'un trend liniar: $\\Delta(a + bt) = b$; un trend pătratic necesită $\\Delta^2$'),
      T('a random walk: $\\Delta X_t = \\varepsilon_t$; a seasonal pattern that repeats exactly: $\\Delta_s$', 'un mers aleator: $\\Delta X_t = \\varepsilon_t$; un tipar sezonier care se repetă exact: $\\Delta_s$')]),
    (T('\\textbf{Integrated process}: $X_t \\sim I(d)$ if $\\Delta^dX_t$ is stationary and $\\Delta^{d-1}X_t$ is not', '\\textbf{Proces integrat}: $X_t \\sim I(d)$ dacă $\\Delta^dX_t$ este staționar, iar $\\Delta^{d-1}X_t$ nu este'),
     [T('white noise is $I(0)$, the random walk is $I(1)$; testing $d$: Chapter 3', 'zgomotul alb este $I(0)$, mersul aleator este $I(1)$; testarea lui $d$: Capitolul 3')])))

chart(T('S\\&P 500: the level and its difference', 'S\\&P 500: nivelul și diferența lui'), 'tsa_ch1_sp500_diff', 'TSA_ch1_transformations', [
    T('Top: the log of the S\\&P 500 index since 2000; bottom: the daily log returns, @{spx.n} days to @{end}', 'Sus: logaritmul indicelui S\\&P 500 din 2000; jos: randamentele logaritmice zilnice, @{spx.n} zile pînă la @{end}')],
    h='0.66\\textheight')

interp(('the S\\&P 500 transformation', 'transformării S\\&P 500'), [
    (T('Log price: sample ACF @{spx.acf1p} at lag 1 and still @{spx.acf50p} at lag 50', 'Logaritmul prețului: ACF de selecție @{spx.acf1p} la lagul 1 și încă @{spx.acf50p} la lagul 50'),
     [T('the behaviour of a random walk: the level remembers everything', 'comportamentul unui mers aleator: nivelul rămîne corelat cu întregul său trecut')]),
    (T('Log returns: mean @{spx.mean}\\% per day, standard deviation @{spx.sd}\\%; ACF at lag 1: $@{spx.acf1r}$', 'Randamentele logaritmice: media @{spx.mean}\\% pe zi, abaterea standard @{spx.sd}\\%; ACF la lagul 1: $@{spx.acf1r}$'),
     [T('a constant level: one difference removed the stochastic trend', 'un nivel constant: o singură diferențiere a eliminat trendul stochastic')]),
    (T('Not yet white noise in the strict sense: quiet and turbulent periods alternate (2008, 2020)', 'Nu încă zgomot alb în sens strict: perioadele liniștite alternează cu cele agitate (2008, 2020)'),
     [T('the returns are close to a weak white noise; their variance is the subject of Chapter 5', 'randamentele sînt apropiate de un zgomot alb slab; varianța lor este subiectul Capitolului 5')])])

D.frame(T('The Wold decomposition', 'Descompunerea Wold'), cols(items(
    (T('\\textbf{Theorem} \\refWold: every weakly stationary process with mean 0 can be written as', '\\textbf{Teoremă} \\refWold: orice proces slab staționar cu media 0 se poate scrie'),
     [T('$X_t = \\sum_{j=0}^{\\infty}\\psi_j\\varepsilon_{t-j} + \\eta_t$, with $\\psi_0 = 1$, $\\sum_j\\psi_j^2 < \\infty$', '$X_t = \\sum_{j=0}^{\\infty}\\psi_j\\varepsilon_{t-j} + \\eta_t$, cu $\\psi_0 = 1$, $\\sum_j\\psi_j^2 < \\infty$'),
      T('$\\varepsilon_t$: white noise, the error of the best linear forecast of $X_t$ from its past (the \\textbf{innovation})', '$\\varepsilon_t$: zgomot alb, eroarea celei mai bune prognoze liniare a lui $X_t$ din trecutul său (\\textbf{inovația})'),
      T('$\\eta_t$: a deterministic part, perfectly predictable from the distant past (for example, a fixed cycle)', '$\\eta_t$: o parte deterministă, perfect previzibilă din trecutul îndepărtat (de exemplu, un ciclu fix)')]),
    (T('Meaning', 'Semnificația'),
     [T('any stationary series is a weighted sum of past shocks: MA($\\infty$)', 'orice serie staționară este o sumă ponderată de șocuri trecute: MA($\\infty$)'),
      T('the weights $\\psi_j$ = the effect of a shock after $j$ periods (impulse response)', 'ponderile $\\psi_j$ = efectul unui șoc după $j$ perioade (răspunsul la impuls)'),
      T('ARMA models (Chapter 2) approximate the infinite $\\psi_j$ with few parameters', 'modelele ARMA (Capitolul 2) aproximează șirul infinit $\\psi_j$ cu puțini parametri')])),
    ph('wold', T('Herman Wold, Uppsala, 1969', 'Herman Wold, Uppsala, 1969'), h='0.48\\textheight'), wl='0.66', wr='0.30'), 'footnotesize')

chart(T('Wold weights of four processes', 'Ponderile Wold ale a patru procese'), 'tsa_ch1_wold', 'TSA_ch1_wold_ergodicity', [
    T('$\\psi_j$, $j = 0, \\dots, 15$: AR(1) with $\\phi = 0.8$ and $\\phi = -0.6$, MA(1) with $\\theta = 0.6$, and the random walk', '$\\psi_j$, $j = 0, \\dots, 15$: AR(1) cu $\\phi = 0{,}8$ și $\\phi = -0{,}6$, MA(1) cu $\\theta = 0{,}6$ și mersul aleator'),
    T('AR(1), $\\phi = 0.8$: $\\psi_5 = @{wold.p5}$, $\\psi_{10} = @{wold.p10}$; $\\mathrm{Var}(X_t) = \\sigma^2\\sum_j\\psi_j^2 = @{wold.s2}\\,\\sigma^2$', 'AR(1), $\\phi = 0{,}8$: $\\psi_5 = @{wold.p5}$, $\\psi_{10} = @{wold.p10}$; $\\mathrm{Var}(X_t) = \\sigma^2\\sum_j\\psi_j^2 = @{wold.s2}\\,\\sigma^2$')],
    h='0.50\\textheight')

interp(('the Wold weights', 'ponderilor Wold'), [
    (T('Stationary processes: $\\psi_j \\to 0$, so shocks fade and the variance $\\sigma^2\\sum_j\\psi_j^2$ is finite', 'Procesele staționare: $\\psi_j \\to 0$, deci șocurile se sting, iar varianța $\\sigma^2\\sum_j\\psi_j^2$ este finită'),
     [T('$\\phi < 0$: the effect alternates in sign; MA(1): the effect stops after one period', '$\\phi < 0$: efectul își schimbă semnul alternativ; MA(1): efectul se oprește după o perioadă')]),
    (T('Random walk: $\\psi_j = 1$ for all $j$, $\\sum_j\\psi_j^2 = \\infty$: no Wold form, no stationarity', 'Mersul aleator: $\\psi_j = 1$ pentru orice $j$, $\\sum_j\\psi_j^2 = \\infty$: fără formă Wold, fără staționaritate'),
     [T('the economic meaning: a permanent shock (a technology shock to GDP, news in a price)', 'sensul economic: un șoc permanent (un șoc tehnologic asupra PIB-ului, o știre în preț)')]),
    T('The ACF is fixed by the weights: $\\gamma(h) = \\sigma^2\\sum_j\\psi_j\\psi_{j+h}$', 'ACF este determinată de ponderi: $\\gamma(h) = \\sigma^2\\sum_j\\psi_j\\psi_{j+h}$')])

D.frame(T('Ergodicity', 'Ergodicitatea'), items(
    (T('Stationarity says the moments are constant; we still need to estimate them from \\textbf{one} path', 'Staționaritatea spune că momentele sînt constante; mai trebuie să le estimăm dintr-o \\textbf{singură} traiectorie'),
     [T('ensemble mean: $\\mu = E[X_t]$ (across paths); time mean: $\\bar X_T = \\frac1T\\sum_{t=1}^{T}X_t$ (along one path)', 'media pe ansamblu: $\\mu = E[X_t]$ (pe traiectorii); media în timp: $\\bar X_T = \\frac1T\\sum_{t=1}^{T}X_t$ (de-a lungul unei traiectorii)')]),
    (T('\\textbf{Definition}: a stationary process is \\textbf{ergodic for the mean} if $\\bar X_T \\to \\mu$ (in mean square) as $T \\to \\infty$', '\\textbf{Definiție}: un proces staționar este \\textbf{ergodic pentru medie} dacă $\\bar X_T \\to \\mu$ (în medie pătratică) cînd $T \\to \\infty$'),
     [T('a sufficient condition: $\\gamma(h) \\to 0$ as $h \\to \\infty$ (correlation dies out)', 'o condiție suficientă: $\\gamma(h) \\to 0$ cînd $h \\to \\infty$ (corelația dispare)'),
      T('if $\\sum_h|\\gamma(h)| < \\infty$, then $T\\,\\mathrm{Var}(\\bar X_T) \\to \\sum_{h=-\\infty}^{\\infty}\\gamma(h)$', 'dacă $\\sum_h|\\gamma(h)| < \\infty$, atunci $T\\,\\mathrm{Var}(\\bar X_T) \\to \\sum_{h=-\\infty}^{\\infty}\\gamma(h)$')]),
    (T('\\textbf{Counterexample}: $X_t = Z + \\varepsilon_t$, $Z \\sim N(0, 1)$ drawn once, $\\varepsilon_t$ white noise', '\\textbf{Contraexemplu}: $X_t = Z + \\varepsilon_t$, $Z \\sim N(0, 1)$ extras o singură dată, $\\varepsilon_t$ zgomot alb'),
     [T('stationary, with $\\gamma(h) = 1$ for every $h \\ne 0$; $\\bar X_T \\to Z$, not $\\mu = 0$', 'staționar, cu $\\gamma(h) = 1$ pentru orice $h \\ne 0$; $\\bar X_T \\to Z$, nu $\\mu = 0$')])))

chart(T('Time averages: ergodic and non-ergodic', 'Mediile în timp: proces ergodic și neergodic'), 'tsa_ch1_ergodicity', 'TSA_ch1_wold_ergodicity', [
    T('Running mean $\\bar X_T$ along 6 paths, $T$ up to @{erg.n}: AR(1) with $\\phi = 0.7$ (left) and $X_t = Z + \\varepsilon_t$ (right)', 'Media cumulată $\\bar X_T$ pe 6 traiectorii, $T$ pînă la @{erg.n}: AR(1) cu $\\phi = 0{,}7$ (stînga) și $X_t = Z + \\varepsilon_t$ (dreapta)'),
    T('Final values: between $@{erg.amin}$ and $@{erg.amax}$ on the left; between $@{erg.bmin}$ and $@{erg.bmax}$ on the right, close to the drawn $Z$',
      'Valorile finale: între $@{erg.amin}$ și $@{erg.amax}$ în stînga; între $@{erg.bmin}$ și $@{erg.bmax}$ în dreapta, aproape de $Z$ extras')],
    h='0.58\\textheight')

interp(('ergodicity', 'ergodicității'), [
    (T('Left: every path ``learns\'\' the true mean 0; one long history is as good as many histories', 'Stînga: fiecare traiectorie „învață” media adevărată 0; o istorie lungă este la fel de bună ca multe istorii'),
     [T('convergence is slow when the correlation is strong: $\\mathrm{Var}(\\bar X_T) \\approx \\gamma(0)(1 + \\phi)/((1 - \\phi)T)$', 'convergența este lentă cînd corelația este puternică: $\\mathrm{Var}(\\bar X_T) \\approx \\gamma(0)(1 + \\phi)/((1 - \\phi)T)$')]),
    (T('Right: each path converges to its own $Z$; no amount of data from one path reveals $\\mu = 0$', 'Dreapta: fiecare traiectorie converge la propriul $Z$; oricîte date dintr-o singură traiectorie nu dezvăluie $\\mu = 0$'),
     [T('an economy with a permanent, unobserved ``type\'\' behaves like this', 'o economie cu un „tip” permanent și neobservat se comportă astfel')]),
    T('Ergodicity cannot be tested from one path; it is assumed whenever we estimate moments from a single series', 'Ergodicitatea nu poate fi testată dintr-o singură traiectorie; o presupunem ori de cîte ori estimăm momente dintr-o singură serie')])

D.recap(('The lag operator, differencing and Wold', 'operatorul lag, diferențierea și descompunerea Wold'), [
    T('$L^kX_t = X_{t-k}$; $\\Delta = 1 - L$; $\\Delta_s = 1 - L^s$; $I(d)$: stationary after $d$ differences', '$L^kX_t = X_{t-k}$; $\\Delta = 1 - L$; $\\Delta_s = 1 - L^s$; $I(d)$: staționar după $d$ diferențieri'),
    T('Wold: a stationary process is an MA($\\infty$) of its innovations, with $\\sum\\psi_j^2 < \\infty$', 'Wold: un proces staționar este un MA($\\infty$) al inovațiilor sale, cu $\\sum\\psi_j^2 < \\infty$'),
    T('Ergodicity lets time averages estimate ensemble moments; it needs correlations that die out', 'Ergodicitatea permite ca mediile în timp să estimeze momentele pe ansamblu; cere corelații care se sting')])

# =============================================================================
# 5. ACF ȘI PACF DE SELECȚIE
# =============================================================================
D.section('Sample ACF and PACF', 'ACF și PACF de selecție')

D.frame(T('Sample autocovariance and autocorrelation', 'Autocovarianța și autocorelația de selecție'), items(
    (T('From $x_1, \\dots, x_T$, with the sample mean $\\bar x = \\frac1T\\sum_t x_t$:', 'Din $x_1, \\dots, x_T$, cu media de selecție $\\bar x = \\frac1T\\sum_t x_t$:'),
     [T('$\\hat\\gamma(h) = \\frac1T\\sum_{t=1}^{T-h}(x_t - \\bar x)(x_{t+h} - \\bar x)$, $0 \\le h < T$', '$\\hat\\gamma(h) = \\frac1T\\sum_{t=1}^{T-h}(x_t - \\bar x)(x_{t+h} - \\bar x)$, $0 \\le h < T$'),
      T('$\\hat\\rho(h) = \\hat\\gamma(h)/\\hat\\gamma(0)$: the \\textbf{sample ACF}; the plot of $\\hat\\rho(h)$ against $h$ is the \\textbf{correlogram}', '$\\hat\\rho(h) = \\hat\\gamma(h)/\\hat\\gamma(0)$: \\textbf{ACF de selecție}; graficul lui $\\hat\\rho(h)$ în funcție de $h$ este \\textbf{corelograma}')]),
    (T('Why divide by $T$ and not by $T - h$?', 'Împărțitorul $T$ în locul lui $T - h$'),
     [T('the sequence $\\hat\\gamma(h)$ is then non-negative definite, like a true autocovariance \\refBD', 'șirul $\\hat\\gamma(h)$ este atunci pozitiv semidefinit, ca o autocovarianță adevărată \\refBD'),
      T('the price: a small bias towards 0 at large lags; use $h \\le T/4$', 'prețul: o mică deplasare spre 0 la laguri mari; folosim $h \\le T/4$')]),
    (T('\\textbf{Worked example}: $x = (3, 5, 4, 6, 7)$, $\\bar x = 5$, deviations $(-2, 0, -1, 1, 2)$', '\\textbf{Exemplu rezolvat}: $x = (3, 5, 4, 6, 7)$, $\\bar x = 5$, abaterile $(-2, 0, -1, 1, 2)$'),
     [T('$\\hat\\gamma(0) = 10/5 = @{sx.g0}$; $\\hat\\gamma(1) = (0 + 0 - 1 + 2)/5 = @{sx.g1}$; $\\hat\\gamma(2) = (2 + 0 - 2)/5 = @{sx.g2}$', '$\\hat\\gamma(0) = 10/5 = @{sx.g0}$; $\\hat\\gamma(1) = (0 + 0 - 1 + 2)/5 = @{sx.g1}$; $\\hat\\gamma(2) = (2 + 0 - 2)/5 = @{sx.g2}$'),
      T('$\\hat\\rho(1) = @{sx.r1}$, $\\hat\\rho(2) = @{sx.r2}$; with $T = 5$ the band is $\\pm @{sx.band}$: far too few data', '$\\hat\\rho(1) = @{sx.r1}$, $\\hat\\rho(2) = @{sx.r2}$; cu $T = 5$ banda este $\\pm @{sx.band}$: mult prea puține date')])))

D.frame(T('Confidence bands for the sample ACF', 'Benzi de încredere pentru ACF de selecție'), items(
    (T('\\textbf{Large-sample result} of \\refBartlett: if $X_t$ is i.i.d. with finite variance, for large $T$', '\\textbf{Rezultatul asimptotic} din \\refBartlett: dacă $X_t$ este i.i.d. cu varianță finită, pentru $T$ mare'),
     [T('$\\hat\\rho(1), \\dots, \\hat\\rho(m)$ are approximately independent $N(0, 1/T)$', '$\\hat\\rho(1), \\dots, \\hat\\rho(m)$ sînt aproximativ independente, $N(0, 1/T)$'),
      T('95\\% band: $\\pm 1.96/\\sqrt{T}$; a bar outside it rejects $\\rho(h) = 0$ at 5\\%, \\textbf{for that lag alone}', 'banda de 95\\%: $\\pm 1{,}96/\\sqrt{T}$; o bară în afara ei respinge $\\rho(h) = 0$ la 5\\%, \\textbf{doar pentru acel lag}')]),
    (T('For an MA($q$), at lags $h > q$: $\\mathrm{Var}(\\hat\\rho(h)) \\approx \\frac1T\\big(1 + 2\\sum_{k=1}^{q}\\rho(k)^2\\big)$', 'Pentru un MA($q$), la laguri $h > q$: $\\mathrm{Var}(\\hat\\rho(h)) \\approx \\frac1T\\big(1 + 2\\sum_{k=1}^{q}\\rho(k)^2\\big)$'),
     [T('the wider bands of \\texttt{statsmodels} \\texttt{plot\\_acf} (Chapter 2)', 'benzile mai largi din \\texttt{plot\\_acf} din \\texttt{statsmodels} (Capitolul 2)')]),
    (T('Two traps', 'Două capcane'),
     [T('with 20 lags and no correlation, about one bar is expected outside the band by chance', 'cu 20 de laguri și nicio corelație, ne așteptăm ca aproximativ o bară să iasă din bandă din întîmplare'),
      T('the band assumes i.i.d. data; for a weak white noise (GARCH returns) the true band is wider', 'banda presupune date i.i.d.; pentru un zgomot alb slab (randamente GARCH) banda adevărată este mai largă')])))

chart(T('Sampling distribution of the sample ACF', 'Distribuția de selecție a ACF'), 'tsa_ch1_bartlett', 'TSA_ch1_acf_pacf', [
    T('10000 samples of Gaussian white noise with $T = 100$: $\\hat\\rho(1)$ (left) and the number of lags, out of 20, outside $\\pm @{bt.band}$ (right)', '10000 de eșantioane de zgomot alb gaussian cu $T = 100$: $\\hat\\rho(1)$ (stînga) și numărul de laguri, din 20, în afara lui $\\pm @{bt.band}$ (dreapta)')],
    h='0.60\\textheight')

interp(('the simulation', 'simulării'), [
    (T('$\\hat\\rho(1)$: mean $@{bt.mean}$ (theory $-1/T = -0.01$), standard deviation @{bt.sd} (theory $1/\\sqrt{T} = 0.1$)', '$\\hat\\rho(1)$: media $@{bt.mean}$ (teoretic $-1/T = -0{,}01$), abaterea standard @{bt.sd} (teoretic $1/\\sqrt{T} = 0{,}1$)'),
     [T('Bartlett\'s approximation is already good at $T = 100$', 'aproximarea lui Bartlett este deja bună la $T = 100$')]),
    (T('At least one of 20 bars outside the band in @{bt.pone}\\% of the samples, although there is no correlation at all', 'Cel puțin una din 20 de bare iese din bandă în @{bt.pone}\\% dintre eșantioane, deși nu există nicio corelație'),
     [T('independent tests would give $1 - 0.95^{20} = @{bt.pth}\\%$; at distant lags $\\mathrm{Var}(\\hat\\rho(h)) < 1/T$, so slightly fewer bars cross', 'testele independente ar da $1 - 0{,}95^{20} = @{bt.pth}\\%$; la laguri mari $\\mathrm{Var}(\\hat\\rho(h)) < 1/T$, deci ies ceva mai puține bare')]),
    T('Do not over-read single bars: judge the pattern, or use a joint test (Section 6)', 'Nu supra-interpretați bare izolate: judecați tiparul sau folosiți un test comun (secțiunea 6)')])

chart(T('Theoretical and sample ACF of four processes', 'ACF teoretică și de selecție pentru patru procese'), 'tsa_ch1_acf_models', 'TSA_ch1_acf_pacf', [
    T('$T = 500$ simulated values; bars: sample ACF; dots: theoretical ACF; band $\\pm @{am.band}$', '$T = 500$ de valori simulate; bare: ACF de selecție; puncte: ACF teoretică; banda $\\pm @{am.band}$')],
    h='0.56\\textheight')

interp(('the four correlograms', 'celor patru corelograme'), [
    T('White noise: all bars small; the largest, $\\hat\\rho(2) = @{am.wn2}$, is noise', 'Zgomotul alb: toate barele sînt mici; cea mai mare, $\\hat\\rho(2) = @{am.wn2}$, este zgomot'),
    T('AR(1), $\\phi = 0.8$: geometric decay, $\\hat\\rho(1) = @{am.ar1}$, $\\hat\\rho(2) = @{am.ar2}$ (theory 0.80 and 0.64)', 'AR(1), $\\phi = 0{,}8$: descreștere geometrică, $\\hat\\rho(1) = @{am.ar1}$, $\\hat\\rho(2) = @{am.ar2}$ (teoretic 0,80 și 0,64)'),
    T('MA(1), $\\theta = 0.6$: one spike, $\\hat\\rho(1) = @{am.ma1}$ (theory @{am.math}), then nothing: the ACF \\textbf{cuts off} after lag 1', 'MA(1), $\\theta = 0{,}6$: un singur vîrf, $\\hat\\rho(1) = @{am.ma1}$ (teoretic @{am.math}), apoi nimic: ACF \\textbf{se anulează} după lagul 1'),
    (T('Random walk: $\\hat\\rho(1) = @{am.rw1}$, still @{am.rw10} at lag 10: the slow, almost linear decay of a non-stationary series', 'Mersul aleator: $\\hat\\rho(1) = @{am.rw1}$, încă @{am.rw10} la lagul 10: descreșterea lentă, aproape liniară, a unei serii nestaționare'),
     [T('the sample ACF of a random walk has no theoretical counterpart: $\\rho(h)$ is not defined', 'ACF de selecție a unui mers aleator nu are corespondent teoretic: $\\rho(h)$ nu este definit')])])

D.frame(T('Partial autocorrelation', 'Autocorelația parțială'), items(
    (T('\\textbf{Definition}: the \\textbf{PACF} at lag $h$, $\\phi_{hh}$, is the last coefficient of the best linear predictor of $X_t$ from $X_{t-1}, \\dots, X_{t-h}$:',
       '\\textbf{Definiție}: \\textbf{PACF} (funcția de autocorelație parțială) la lagul $h$, $\\phi_{hh}$, este ultimul coeficient al celei mai bune prognoze liniare a lui $X_t$ din $X_{t-1}, \\dots, X_{t-h}$:'),
     [T('$X_t = \\phi_{h1}X_{t-1} + \\dots + \\phi_{hh}X_{t-h} + e_t$; $\\phi_{h1}, \\dots, \\phi_{hh}$: the coefficients of the regression on $h$ lags; $e_t$: the prediction error', '$X_t = \\phi_{h1}X_{t-1} + \\dots + \\phi_{hh}X_{t-h} + e_t$; $\\phi_{h1}, \\dots, \\phi_{hh}$: coeficienții regresiei pe $h$ laguri; $e_t$: eroarea de prognoză'),
      T('the correlation of $X_t$ and $X_{t-h}$ after removing the linear effect of the lags in between', 'corelația dintre $X_t$ și $X_{t-h}$ după eliminarea efectului liniar al lagurilor intermediare')]),
    (T('First two values', 'Primele două valori'),
     [T('$\\phi_{11} = \\rho(1)$; $\\phi_{22} = \\dfrac{\\rho(2) - \\rho(1)^2}{1 - \\rho(1)^2}$', '$\\phi_{11} = \\rho(1)$; $\\phi_{22} = \\dfrac{\\rho(2) - \\rho(1)^2}{1 - \\rho(1)^2}$'),
      T('the general case: the Durbin--Levinson recursion; the sample PACF uses $\\hat\\rho(h)$ in place of $\\rho(h)$', 'cazul general: recursia Durbin--Levinson; PACF de selecție folosește $\\hat\\rho(h)$ în locul lui $\\rho(h)$')]),
    (T('\\textbf{Worked example}', '\\textbf{Exemplu rezolvat}'),
     [T('AR(1): $\\rho(2) = \\rho(1)^2$, so $\\phi_{22} = 0$: once $X_{t-1}$ is known, $X_{t-2}$ adds nothing', 'AR(1): $\\rho(2) = \\rho(1)^2$, deci $\\phi_{22} = 0$: odată cunoscut $X_{t-1}$, $X_{t-2}$ nu mai aduce nimic'),
      T('AR(2) with $\\phi_1 = 0.5$, $\\phi_2 = 0.3$: $\\rho(1) = @{ap.ar2.rho1}$, $\\rho(2) = @{ap.ar2.rho2}$, $\\phi_{22} = (@{ap.ar2.rho2} - @{ap.r1sq})/(1 - @{ap.r1sq}) = @{ap.phi22} = \\phi_2$',
        'AR(2) cu $\\phi_1 = 0{,}5$, $\\phi_2 = 0{,}3$: $\\rho(1) = @{ap.ar2.rho1}$, $\\rho(2) = @{ap.ar2.rho2}$, $\\phi_{22} = (@{ap.ar2.rho2} - @{ap.r1sq})/(1 - @{ap.r1sq}) = @{ap.phi22} = \\phi_2$')]),
    T('Same band as the ACF under white noise: $\\pm 1.96/\\sqrt{T}$', 'Aceeași bandă ca pentru ACF, în cazul zgomotului alb: $\\pm 1{,}96/\\sqrt{T}$')))

chart(T('ACF and PACF of three processes', 'ACF și PACF pentru trei procese'), 'tsa_ch1_acf_pacf', 'TSA_ch1_acf_pacf', [
    T('$T = 500$ simulated values of AR(1), AR(2) and MA(1); top: sample ACF; bottom: sample PACF', '$T = 500$ de valori simulate din AR(1), AR(2) și MA(1); sus: ACF de selecție; jos: PACF de selecție')],
    h='0.64\\textheight')

interp(('the ACF and PACF', 'ACF și PACF'), [
    (T('AR(1): the ACF decays; the PACF has one spike, $\\hat\\phi_{11} = @{ap.ar1.p1}$, then $\\hat\\phi_{22} = @{ap.ar1.p2}$', 'AR(1): ACF descrește; PACF are un singur vîrf, $\\hat\\phi_{11} = @{ap.ar1.p1}$, apoi $\\hat\\phi_{22} = @{ap.ar1.p2}$'),
     [T('AR(2): two PACF spikes, $\\hat\\phi_{22} = @{ap.ar2.p2}$ (theory 0.3), then $\\hat\\phi_{33} = @{ap.ar2.p3}$', 'AR(2): două vîrfuri în PACF, $\\hat\\phi_{22} = @{ap.ar2.p2}$ (teoretic 0,3), apoi $\\hat\\phi_{33} = @{ap.ar2.p3}$')]),
    T('MA(1): the mirror image; one ACF spike ($\\hat\\rho(1) = @{ap.ma1.r1}$), a PACF that alternates and decays ($@{ap.ma1.p2}$, $@{ap.ma1.p3}$, ...)', 'MA(1): imaginea în oglindă; un singur vîrf în ACF ($\\hat\\rho(1) = @{ap.ma1.r1}$), o PACF care alternează și descrește ($@{ap.ma1.p2}$, $@{ap.ma1.p3}$, ...)'),
    (T('The identification rule of Chapter 2 \\refHP:', 'Regula de identificare din Capitolul 2 \\refHP:'),
     [T('PACF cuts off after $p$: AR($p$); ACF cuts off after $q$: MA($q$); both decay: ARMA', 'PACF se anulează după $p$: AR($p$); ACF se anulează după $q$: MA($q$); ambele descresc: ARMA')])])

D.frame(T('Real data: three markets', 'Date reale: trei piețe'), cols(
    ph('bnr', T('The National Bank of Romania (BNR), publisher of the EUR/RON reference rate', 'Banca Națională a României (BNR), care publică cursul de referință EUR/RON'), h='0.48\\textheight'),
    items(T('Next: the ACF of BET daily log returns since 2000, of their absolute values and of their squares', 'Urmează: ACF a randamentelor logaritmice zilnice ale BET din 2000, a valorilor lor absolute și a pătratelor lor'),
          T('The same analysis for EUR/RON (the BNR reference rate) and the S\\&P 500: Seminar 1, task B2', 'Aceeași analiză pentru EUR/RON (cursul de referință BNR) și S\\&P 500: Seminarul 1, cerința B2'),
          T('Question: are BET returns white noise? Of which kind?', 'Întrebarea: sînt randamentele BET zgomot alb? De ce tip?')),
    wl='0.40', wr='0.56'))

chart(T('BET: ACF of returns, absolute and squared returns', 'BET: ACF a randamentelor, a valorilor absolute și a pătratelor'), 'tsa_ch1_bet_acf', 'TSA_ch1_acf_pacf', [
    T('Daily log returns, @{bt2.n} days, lags 1--50; band $\\pm @{bet.band}$', 'Randamente logaritmice zilnice, @{bt2.n} zile, lagurile 1--50; banda $\\pm @{bet.band}$')],
    h='0.58\\textheight')

interp(('the BET correlograms', 'corelogramelor BET'), [
    (T('Returns: $\\hat\\rho(1) = @{bet.r1}$, clearly outside the band; $Q^*(10) = @{bet.q}$', 'Randamentele: $\\hat\\rho(1) = @{bet.r1}$, clar în afara benzii; $Q^*(10) = @{bet.q}$'),
     [T('statistically significant but small: yesterday\'s return explains about $\\hat\\rho(1)^2 = @{bet.r1sq}\\%$ of today\'s variance', 'semnificativă statistic, dar mică: randamentul de ieri explică aproximativ $\\hat\\rho(1)^2 = @{bet.r1sq}\\%$ din varianța celui de azi'),
      T('a typical sign of a less liquid market (non-synchronous trading)', 'un semn tipic al unei piețe mai puțin lichide (tranzacționare nesincronă)')]),
    (T('Absolute returns: $\\hat\\rho(1) = @{bet.abs1}$ and still @{bet.abs50} at lag 50; squares: $\\hat\\rho(1) = @{bet.sq1}$', 'Valorile absolute: $\\hat\\rho(1) = @{bet.abs1}$ și încă @{bet.abs50} la lagul 50; pătratele: $\\hat\\rho(1) = @{bet.sq1}$'),
     [T('large moves follow large moves: volatility clustering, a long memory in the size of returns', 'mișcările mari urmează după mișcări mari: volatility clustering, o memorie lungă a mărimii randamentelor')]),
    T('Conclusion: BET returns are close to a \\textbf{weak} white noise; the mean is hardly predictable, the variance is (Chapter 5)', 'Concluzie: randamentele BET sînt apropiate de un zgomot alb \\textbf{slab}; media este greu de anticipat, varianța nu (Capitolul 5)')])

D.recap(('Sample ACF and PACF', 'ACF și PACF de selecție'), [
    T('$\\hat\\rho(h) = \\hat\\gamma(h)/\\hat\\gamma(0)$, with divisor $T$; use lags up to about $T/4$', '$\\hat\\rho(h) = \\hat\\gamma(h)/\\hat\\gamma(0)$, cu împărțitorul $T$; folosim laguri pînă la aproximativ $T/4$'),
    T('Under i.i.d. data: $\\hat\\rho(h) \\approx N(0, 1/T)$, band $\\pm 1.96/\\sqrt{T}$; 1 bar in 20 crosses by chance', 'Pentru date i.i.d.: $\\hat\\rho(h) \\approx N(0, 1/T)$, banda $\\pm 1{,}96/\\sqrt{T}$; o bară din 20 iese din întîmplare'),
    T('PACF: direct effect of lag $h$; AR($p$) cuts off in the PACF, MA($q$) in the ACF', 'PACF: efectul direct al lagului $h$; AR($p$) se anulează în PACF, MA($q$) în ACF'),
    T('Returns: small linear, strong non-linear dependence', 'Randamentele: dependență liniară slabă, dependență neliniară puternică')])

# =============================================================================
# 6. TESTE PORTMANTEAU
# =============================================================================
D.section('Portmanteau tests: Box--Pierce and Ljung--Box', 'Teste portmanteau: Box--Pierce și Ljung--Box')

D.frame(T('Testing many autocorrelations at once', 'Testarea simultană a mai multor autocorelații'), items(
    (T('$H_0$: $\\rho(1) = \\dots = \\rho(m) = 0$ (the series is white noise up to lag $m$); $H_1$: at least one $\\rho(h) \\ne 0$', '$H_0$: $\\rho(1) = \\dots = \\rho(m) = 0$ (seria este zgomot alb pînă la lagul $m$); $H_1$: cel puțin un $\\rho(h) \\ne 0$'),
     [T('a \\textbf{portmanteau} test: one statistic ``carries\'\' $m$ autocorrelations', 'un test \\textbf{portmanteau}: o singură statistică „poartă” $m$ autocorelații')]),
    (T('The \\textbf{Box--Pierce statistic}, from \\refBP: $Q(m) = T\\sum_{h=1}^{m}\\hat\\rho(h)^2$', '\\textbf{Statistica Box--Pierce}, din \\refBP: $Q(m) = T\\sum_{h=1}^{m}\\hat\\rho(h)^2$'),
     [T('under $H_0$, each $\\sqrt{T}\\hat\\rho(h) \\approx N(0, 1)$, independent: the sum of $m$ squares is $\\approx \\chi^2(m)$', 'în ipoteza $H_0$, fiecare $\\sqrt{T}\\hat\\rho(h) \\approx N(0, 1)$, independente: suma a $m$ pătrate este $\\approx \\chi^2(m)$')]),
    (T('The \\textbf{Ljung--Box statistic}, from \\refLB: $Q^*(m) = T(T + 2)\\sum_{h=1}^{m}\\dfrac{\\hat\\rho(h)^2}{T - h}$', '\\textbf{Statistica Ljung--Box}, din \\refLB: $Q^*(m) = T(T + 2)\\sum_{h=1}^{m}\\dfrac{\\hat\\rho(h)^2}{T - h}$'),
     [T('the weight $(T + 2)/(T - h)$ corrects the small variance of $\\hat\\rho(h)$ at distant lags; same $\\chi^2(m)$ limit', 'ponderea $(T + 2)/(T - h)$ corectează varianța mică a lui $\\hat\\rho(h)$ la laguri mari; aceeași limită $\\chi^2(m)$')]),
    (T('Reject $H_0$ at 5\\% if $Q^* > \\chi^2_{0.95}(m)$; for example $\\chi^2_{0.95}(10) = @{chi10}$', 'Respingem $H_0$ la 5\\% dacă $Q^* > \\chi^2_{0{,}95}(m)$; de exemplu, $\\chi^2_{0{,}95}(10) = @{chi10}$'),
     [T('$\\chi^2(m)$: the chi-square distribution with $m$ degrees of freedom; $\\chi^2_{0.95}(m)$: its 95\\% quantile', '$\\chi^2(m)$: distribuția hi-pătrat cu $m$ grade de libertate; $\\chi^2_{0{,}95}(m)$: cuantila ei de 95\\%'),
      T('p-value: the probability under $H_0$ of a statistic at least as large as the observed one; reject if it is below 0.05', 'p-value: probabilitatea, în ipoteza $H_0$, de a obține o statistică cel puțin la fel de mare ca cea observată; respingem dacă este sub 0,05'),
      T('on residuals of an ARMA($p, q$): $\\chi^2(m - p - q)$ (Chapter 2)', 'pentru reziduurile unui ARMA($p, q$): $\\chi^2(m - p - q)$ (Capitolul 2)')])))

D.frame(T('Worked example: the Ljung--Box statistic', 'Exemplu rezolvat: statistica Ljung--Box'), items(
    (T('$T = 100$; $\\hat\\rho(1) = 0.25$, $\\hat\\rho(2) = 0.12$, $\\hat\\rho(3) = -0.08$; band $\\pm @{lbx.band}$', '$T = 100$; $\\hat\\rho(1) = 0{,}25$, $\\hat\\rho(2) = 0{,}12$, $\\hat\\rho(3) = -0{,}08$; banda $\\pm @{lbx.band}$'),
     [T('only $\\hat\\rho(1)$ is outside the band', 'doar $\\hat\\rho(1)$ este în afara benzii')]),
    (T('Box--Pierce: $Q(3) = 100\\,(0.0625 + 0.0144 + 0.0064) = @{lbx.bp}$', 'Box--Pierce: $Q(3) = 100\\,(0{,}0625 + 0{,}0144 + 0{,}0064) = @{lbx.bp}$'),
     [T('Ljung--Box: $Q^*(3) = 100 \\cdot 102\\,\\big(\\frac{0.0625}{99} + \\frac{0.0144}{98} + \\frac{0.0064}{97}\\big) = @{lbx.lb}$', 'Ljung--Box: $Q^*(3) = 100 \\cdot 102\\,\\big(\\frac{0{,}0625}{99} + \\frac{0{,}0144}{98} + \\frac{0{,}0064}{97}\\big) = @{lbx.lb}$')]),
    (T('Critical value $\\chi^2_{0.95}(3) = @{lbx.crit}$: reject $H_0$; p-value @{lbx.p}', 'Valoarea critică $\\chi^2_{0{,}95}(3) = @{lbx.crit}$: respingem $H_0$; p-value-ul este @{lbx.p}'),
     [T('the series is not white noise; the dependence is concentrated at lag 1', 'seria nu este zgomot alb; dependența este concentrată la lagul 1')]),
    (T('Choice of $m$: too small misses distant lags; too large dilutes the evidence', 'Alegerea lui $m$: prea mic ratează lagurile îndepărtate; prea mare diluează dovezile'),
     [T('common choices: $m = 10$ for non-seasonal data, $m = 2s$ for seasonal data with period $s$ \\refFPP', 'alegeri uzuale: $m = 10$ pentru date nesezoniere, $m = 2s$ pentru date sezoniere cu perioada $s$ \\refFPP')])))

chart(T('Box--Pierce and Ljung--Box in small samples', 'Box--Pierce și Ljung--Box în eșantioane mici'), 'tsa_ch1_lb_size', 'TSA_ch1_portmanteau', [
    T('Share of 5000 Gaussian white-noise samples in which each test rejects a true $H_0$ at the 5\\% level', 'Proporția din 5000 de eșantioane de zgomot alb gaussian în care fiecare test respinge o ipoteză $H_0$ adevărată, la nivelul de 5\\%')],
    h='0.60\\textheight')

interp(('the size simulation', 'simulării mărimii testelor'), [
    (T('$T = 50$, $m = 20$: Box--Pierce rejects in @{ls.50_20.bp}\\% of samples, Ljung--Box in @{ls.50_20.lb}\\%', '$T = 50$, $m = 20$: Box--Pierce respinge în @{ls.50_20.bp}\\% dintre eșantioane, Ljung--Box în @{ls.50_20.lb}\\%'),
     [T('Box--Pierce is too lenient: it misses dependence; Ljung--Box errs the other way when $m$ is large relative to $T$', 'Box--Pierce respinge prea rar, deci poate rata dependența; Ljung--Box respinge prea des cînd $m$ este mare față de $T$')]),
    T('$T = 500$, $m = 20$: @{ls.500_20.bp}\\% and @{ls.500_20.lb}\\%: both close to 5\\%', '$T = 500$, $m = 20$: @{ls.500_20.bp}\\% și @{ls.500_20.lb}\\%: ambele aproape de 5\\%'),
    T('Practice: use Ljung--Box (the default in software), and keep $m$ well below $T$ (for example $m \\le T/5$)', 'În practică: folosim Ljung--Box (implicit în programe) și păstrăm $m$ mult sub $T$ (de exemplu, $m \\le T/5$)')])

chart(T('Ljung--Box statistics of real series', 'Statisticile Ljung--Box pentru serii reale'), 'tsa_ch1_lb_stats', 'TSA_ch1_portmanteau', [
    T('$Q^*(m)$ for $m = 1, \\dots, 30$ (log scale) against the 5\\% critical value $\\chi^2_{0.95}(m)$ (dashed)', '$Q^*(m)$ pentru $m = 1, \\dots, 30$ (scară logaritmică), față de valoarea critică de 5\\% $\\chi^2_{0{,}95}(m)$ (linie întreruptă)')],
    h='0.60\\textheight')

D.frame(T('Interpreting the Ljung--Box statistics', 'Interpretarea statisticilor Ljung--Box'), table(
    'lrrrrr', T('Series & $T$ & $\\hat\\rho(1)$ & $\\hat\\rho(4)$ & $Q^*(10)$ & p', 'Seria & $T$ & $\\hat\\rho(1)$ & $\\hat\\rho(4)$ & $Q^*(10)$ & p'),
    [f'{lab} & @{{rl{i}.n}} & $@{{rl{i}.r1}}$ & $@{{rl{i}.r4}}$ & @{{rl{i}.q}} & @{{rl{i}.p}}' for i, (_, lab) in enumerate(ROWS)],
    size='scriptsize') + items(
    T('Every series rejects white noise; for the squared returns the statistic is tens of times larger than for the returns', 'Toate seriile resping ipoteza de zgomot alb; pentru pătratele randamentelor statistica este de zeci de ori mai mare decît pentru randamente'),
    T('Rejection says only that \\textbf{some} correlation exists; the ACF shows where (GDP: lag 4, the season)', 'Respingerea spune doar că există \\textbf{o anumită} corelație; ACF arată unde (PIB: lagul 4, sezonul)')), 'footnotesize')

D.recap(('Portmanteau tests', 'teste portmanteau'), [
    T('$Q(m) = T\\sum\\hat\\rho(h)^2$, $Q^*(m) = T(T + 2)\\sum\\hat\\rho(h)^2/(T - h)$; both $\\approx \\chi^2(m)$ under white noise', '$Q(m) = T\\sum\\hat\\rho(h)^2$, $Q^*(m) = T(T + 2)\\sum\\hat\\rho(h)^2/(T - h)$; ambele $\\approx \\chi^2(m)$ pentru zgomot alb'),
    T('Ljung--Box has better size in small samples; keep $m$ well below $T$', 'Ljung--Box are o mărime mai bună în eșantioane mici; păstrăm $m$ mult sub $T$'),
    T('A rejection is a signal to model, not a model; under GARCH effects the test of returns rejects too often', 'O respingere este un semnal pentru modelare, nu un model; în prezența efectelor GARCH, testul randamentelor respinge prea des')])

# =============================================================================
# 7. TRANSFORMĂRI
# =============================================================================
D.section('Transformations', 'Transformări')

D.frame(T('Reasons to transform a series', 'Motivele transformării unei serii'), items(
    (T('Goals', 'Scopuri'),
     [T('stabilise the variance (seasonal swings that grow with the level)', 'stabilizarea varianței (oscilații sezoniere care cresc odată cu nivelul)'),
      T('turn exponential growth into a linear trend', 'transformarea creșterii exponențiale într-un trend liniar'),
      T('remove a stochastic trend or a seasonal pattern (differencing)', 'eliminarea unui trend stochastic sau a unui tipar sezonier (diferențiere)')]),
    (T('\\textbf{Logarithm}: $\\ln Y_t$; its difference is the growth rate in continuous terms', '\\textbf{Logaritmul}: $\\ln Y_t$; diferența lui este rata de creștere în termeni continui'),
     [T('$100\\,\\Delta\\ln Y_t \\approx$ percentage change for small changes: $\\ln 1.03 = @{lg.03}$, about 3\\%', '$100\\,\\Delta\\ln Y_t \\approx$ variația procentuală, pentru variații mici: $\\ln 1{,}03 = @{lg.03}$, circa 3\\%'),
      T('not for large ones: $\\ln 1.30 = @{lg.30}$, $\\ln 0.70 = @{lg.m30}$; log changes add up over time, percentages do not', 'nu și pentru cele mari: $\\ln 1{,}30 = @{lg.30}$, $\\ln 0{,}70 = @{lg.m30}$; variațiile logaritmice se adună în timp, procentele nu')]),
    (T('Order of operations for a trending, seasonal, positive series', 'Ordinea operațiilor pentru o serie pozitivă, cu trend și sezonalitate'),
     [T('first the log (variance), then the differences (trend, season); never the other way round', 'întîi logaritmul (varianța), apoi diferențele (trendul, sezonul); niciodată invers')])))

chart(T('Romanian real GDP: log, quarterly and annual growth', 'PIB-ul real al României: logaritm, creștere trimestrială și anuală'), 'tsa_ch1_gdp_transform', 'TSA_ch1_transformations', [
    T('Quarterly, not seasonally adjusted, 2010 prices (Eurostat), @{gdp.q0}--@{gdp.q1}; top: the series; bottom: their sample ACF',
      'Date trimestriale, neajustate sezonier, prețuri din 2010 (Eurostat), @{gdp.q0}--@{gdp.q1}; sus: seriile; jos: ACF de selecție')],
    h='0.64\\textheight')

interp(('the GDP transformations', 'transformărilor PIB'), [
    T('$\\ln Y_t$: ACF @{gd.acf1_level} at lag 1 and @{gd.acf8_level} at lag 8, with peaks every 4 quarters: trend and season', '$\\ln Y_t$: ACF @{gd.acf1_level} la lagul 1 și @{gd.acf8_level} la lagul 8, cu vîrfuri la fiecare 4 trimestre: trend și sezon'),
    (T('$\\Delta\\ln Y_t$: the trend is gone, the season dominates: $\\hat\\rho(4) = @{gd.acf4_d1}$, $\\hat\\rho(2) = @{gd.acf2_d1}$; standard deviation @{gd.sd1}\\%', '$\\Delta\\ln Y_t$: trendul a dispărut, sezonul domină: $\\hat\\rho(4) = @{gd.acf4_d1}$, $\\hat\\rho(2) = @{gd.acf2_d1}$; abaterea standard @{gd.sd1}\\%'),
     [T('a quarter-on-quarter rate of unadjusted data mostly measures the season', 'rata trimestru față de trimestru a datelor neajustate măsoară mai ales sezonul')]),
    (T('$\\Delta_4\\ln Y_t$ (annual growth): mean @{gd.mean4}\\%, the ACF dies out after 2--3 quarters ($\\hat\\rho(1) = @{gd.acf1_d4}$, $\\hat\\rho(4) = @{gd.acf4_d4}$)',
       '$\\Delta_4\\ln Y_t$ (creșterea anuală): media @{gd.mean4}\\%, ACF se stinge după 2--3 trimestre ($\\hat\\rho(1) = @{gd.acf1_d4}$, $\\hat\\rho(4) = @{gd.acf4_d4}$)'),
     [T('close to stationary, with business-cycle persistence; 2020Q2: $@{gd.covid}\\%$; last quarter: $@{gd.last4}\\%$', 'apropiată de staționaritate, cu persistența ciclului economic; T2 2020: $@{gd.covid}\\%$; ultimul trimestru: $@{gd.last4}\\%$')])])

D.frame(T('The Box--Cox transformation', 'Transformarea Box--Cox'), items(
    (T('\\textbf{Definition} \\refBC, for $y > 0$: $w = \\dfrac{y^\\lambda - 1}{\\lambda}$ for $\\lambda \\ne 0$, $w = \\ln y$ for $\\lambda = 0$', '\\textbf{Definiție} \\refBC, pentru $y > 0$: $w = \\dfrac{y^\\lambda - 1}{\\lambda}$ pentru $\\lambda \\ne 0$, $w = \\ln y$ pentru $\\lambda = 0$'),
     [T('$\\lambda = 1$: no change in shape; $\\lambda = 0.5$: square root; $\\lambda = 0$: log; $\\lambda = -1$: inverse', '$\\lambda = 1$: forma nu se schimbă; $\\lambda = 0{,}5$: radical; $\\lambda = 0$: logaritm; $\\lambda = -1$: inversă'),
      T('continuous in $\\lambda$: $(y^\\lambda - 1)/\\lambda \\to \\ln y$ as $\\lambda \\to 0$', 'continuă în $\\lambda$: $(y^\\lambda - 1)/\\lambda \\to \\ln y$ cînd $\\lambda \\to 0$')]),
    (T('\\textbf{Worked example}: $y = 100$ and $y = 400$', '\\textbf{Exemplu rezolvat}: $y = 100$ și $y = 400$'),
     [T('$\\lambda = 0.5$: $w = @{bcx.05}$ and $@{bcx.05b}$; $\\lambda = 0$: $w = @{bcx.0}$ and $@{bcx.0b}$', '$\\lambda = 0{,}5$: $w = @{bcx.05}$ și $@{bcx.05b}$; $\\lambda = 0$: $w = @{bcx.0}$ și $@{bcx.0b}$'),
      T('the smaller $\\lambda$, the more large values are compressed', 'cu cît $\\lambda$ este mai mic, cu atît valorile mari sînt comprimate mai mult')]),
    (T('Choosing $\\lambda$: \\refGuerrero, as in \\refFPP', 'Alegerea lui $\\lambda$: \\refGuerrero, ca în \\refFPP'),
     [T('split the series into years; choose $\\lambda$ so that $\\mathrm{sd}_i/\\mathrm{mean}_i^{1-\\lambda}$ is as constant as possible across years; $\\mathrm{sd}_i$, $\\mathrm{mean}_i$: the standard deviation and the mean of year $i$', 'împărțim seria pe ani; alegem $\\lambda$ astfel încît $\\mathrm{sd}_i/\\mathrm{medie}_i^{1-\\lambda}$ să fie cît mai constant de la un an la altul; $\\mathrm{sd}_i$, $\\mathrm{medie}_i$: abaterea standard și media din anul $i$')]),
    T('Forecasts are made for $w$ and transformed back: $y = (\\lambda w + 1)^{1/\\lambda}$; the back-transformed value is the median, not the mean', 'Prognozele se fac pentru $w$ și se transformă înapoi: $y = (\\lambda w + 1)^{1/\\lambda}$; valoarea obținută este mediana, nu media')))

chart(T('Box--Cox for Romanian nominal GDP', 'Box--Cox pentru PIB-ul nominal al României'), 'tsa_ch1_boxcox', 'TSA_ch1_transformations', [
    T('Quarterly nominal GDP, not seasonally adjusted, from @{bx.v0} to @{bx.v1} bn EUR; Guerrero\'s criterion on yearly blocks of 4 quarters', 'PIB-ul nominal trimestrial, neajustat sezonier, de la @{bx.v0} la @{bx.v1} mld. EUR; criteriul lui Guerrero pe blocuri anuale de 4 trimestre')],
    h='0.54\\textheight')

interp(('the Box--Cox choice', 'alegerii Box--Cox'), [
    (T('Guerrero\'s $\\hat\\lambda = @{bx.lam}$: practically the logarithm (criterion @{bx.cvl} at $\\hat\\lambda$, @{bx.cv0} at $\\lambda = 0$, @{bx.cv1} at $\\lambda = 1$)', '$\\hat\\lambda$ al lui Guerrero $= @{bx.lam}$: practic logaritmul (criteriul @{bx.cvl} la $\\hat\\lambda$, @{bx.cv0} la $\\lambda = 0$, @{bx.cv1} la $\\lambda = 1$)'),
     [T('choose the round, interpretable value: $\\lambda = 0$', 'alegem valoarea rotundă și ușor de interpretat: $\\lambda = 0$')]),
    (T('Why the log fits: the seasonal swing is a stable share of the level, about @{bx.amp0}\\% in the first five years and @{bx.amp1}\\% in the last five', 'Logaritmul se potrivește deoarece oscilația sezonieră este o proporție stabilă din nivel, circa @{bx.amp0}\\% în primii cinci ani și @{bx.amp1}\\% în ultimii cinci'),
     [T('multiplicative seasonality becomes additive after the log', 'sezonalitatea multiplicativă devine aditivă după logaritmare')]),
    T('Box--Cox fixes the variance only; the trend and the season still need differencing', 'Box--Cox corectează doar varianța; trendul și sezonul tot trebuie eliminate prin diferențiere')])

chart(T('Over-differencing', 'Supradiferențierea'), 'tsa_ch1_overdiff', 'TSA_ch1_transformations', [
    T('Sample ACF of 500 values of white noise (left) and of its first difference (right)', 'ACF de selecție pentru 500 de valori de zgomot alb (stînga) și pentru prima lui diferență (dreapta)')],
    h='0.50\\textheight')

interp(('over-differencing', 'supradiferențierii'), [
    (T('$\\Delta\\varepsilon_t = \\varepsilon_t - \\varepsilon_{t-1}$ is an MA(1) with $\\theta = -1$: $\\rho(1) = -1/2$; here $\\hat\\rho(1) = @{od.r1}$', '$\\Delta\\varepsilon_t = \\varepsilon_t - \\varepsilon_{t-1}$ este un MA(1) cu $\\theta = -1$: $\\rho(1) = -1/2$; aici $\\hat\\rho(1) = @{od.r1}$'),
     [T('the variance doubles: @{od.ve} becomes @{od.vd}', 'varianța se dublează: @{od.ve} devine @{od.vd}')]),
    (T('A strongly negative $\\hat\\rho(1)$, close to $-0.5$, after differencing: a sign that the series did not need it', 'Un $\\hat\\rho(1)$ puternic negativ, apropiat de $-0{,}5$, după diferențiere: un semn că seria nu avea nevoie de ea'),
     [T('the same happens to a trend-stationary series: $\\Delta(a + bt + \\varepsilon_t) = b + \\varepsilon_t - \\varepsilon_{t-1}$', 'la fel se întîmplă cu o serie staționară în jurul trendului: $\\Delta(a + bt + \\varepsilon_t) = b + \\varepsilon_t - \\varepsilon_{t-1}$')]),
    T('Difference as little as needed; Chapter 3 gives formal tests for the number of differences', 'Diferențiem cît este nevoie, nu mai mult; Capitolul 3 dă testele formale pentru numărul de diferențieri')])

D.recap(('Transformations', 'transformări'), [
    T('Log first for positive series whose swings grow with the level; then differences', 'Întîi logaritmul, pentru seriile pozitive ale căror oscilații cresc cu nivelul; apoi diferențele'),
    T('$100\\,\\Delta\\ln Y_t$: growth in \\%; $\\Delta_4$, $\\Delta_{12}$: annual growth of quarterly and monthly data', '$100\\,\\Delta\\ln Y_t$: creșterea în \\%; $\\Delta_4$, $\\Delta_{12}$: creșterea anuală pentru date trimestriale și lunare'),
    T('Box--Cox with Guerrero\'s $\\lambda$; for Romanian GDP $\\hat\\lambda \\approx 0$, the log', 'Box--Cox cu $\\lambda$ ales prin metoda Guerrero; pentru PIB-ul României $\\hat\\lambda \\approx 0$, logaritmul'),
    T('Over-differencing creates $\\rho(1) \\approx -0.5$ and inflates the variance', 'Supradiferențierea creează $\\rho(1) \\approx -0{,}5$ și mărește varianța')])

# =============================================================================
# 8. SERII CLASICE
# =============================================================================
D.section('Two classic series', 'Două serii clasice')

D.frame(T('Case study: Yule (1927) and Slutzky (1937)', 'Studiu de caz: Yule (1927) și Slutzky (1937)'), items(
    (T('\\textbf{Question} of the 1920s: where do the regular cycles of sunspots and of business activity come from?', '\\textbf{Întrebarea} anilor 1920: de unde provin ciclurile regulate ale petelor solare și ale activității economice?'),
     [T('the dominant view: hidden periodic components (sums of sine waves), estimated with the periodogram', 'viziunea dominantă: componente periodice ascunse (sume de sinusoide), estimate cu periodograma')]),
    (T('\\refYule: the sunspot number depends on its own last two values plus a random disturbance', '\\refYule: numărul de pete solare depinde de propriile două valori anterioare plus o perturbație aleatoare'),
     [T('$X_t = \\phi_1X_{t-1} + \\phi_2X_{t-2} + \\varepsilon_t$: the first autoregression; a damped pendulum hit by random shocks', '$X_t = \\phi_1X_{t-1} + \\phi_2X_{t-2} + \\varepsilon_t$: prima autoregresie; un pendul amortizat lovit de șocuri aleatoare'),
      T('a cycle without any sine wave: the shocks keep it alive, the dynamics give it its period', 'un ciclu fără nicio sinusoidă: șocurile îl întrețin, dinamica îi dă perioada')]),
    (T('\\refSlutzky: moving sums of purely random numbers (a lottery series) produce smooth, cycle-like waves', '\\refSlutzky: sumele mobile ale unor numere pur aleatoare (o serie de loterie) produc unde netede, asemănătoare ciclurilor'),
     [T('the moving average (MA) process, and a warning: cycles in data may be created by averaging', 'procesul de medie mobilă (MA) și un avertisment: ciclurile din date pot fi create prin mediere')]),
    T('Legacy: stationary processes built from white noise; \\refWold\\ unified the two views (Section 4); \\refHP, Ch.~3', 'Moștenirea: procese staționare construite din zgomot alb; \\refWold\\ a unificat cele două viziuni (secțiunea 4); \\refHP, cap.~3')))

D.frame(T('The Nile and the sunspots', 'Nilul și petele solare'), cols(
    ph('nilo', T('The Nilometer on Roda Island, Cairo: Nile levels have been recorded here for more than a thousand years', 'Nilometrul de pe insula Roda, Cairo: nivelurile Nilului se înregistrează aici de peste o mie de ani'), h='0.44\\textheight'),
    items(T('\\textbf{Nile}: yearly flow at Aswan, 1871--1970, 100 observations; a drop around 1898 \\refCobb', '\\textbf{Nilul}: debitul anual la Aswan, 1871--1970, 100 de observații; o scădere în jurul anului 1898 \\refCobb'),
          T('\\textbf{Sunspots}: yearly numbers, 1700--2008, 309 observations; the series that inspired the autoregression \\refYule', '\\textbf{Petele solare}: valori anuale, 1700--2008, 309 observații; seria care a inspirat autoregresia \\refYule'),
          T('Both ship with \\texttt{statsmodels}; both teach a lesson about reading an ACF', 'Ambele sînt incluse în \\texttt{statsmodels}; ambele oferă o lecție despre interpretarea ACF')),
    wl='0.46', wr='0.50'))

chart(T('The Nile and the sunspots: series and ACF', 'Nilul și petele solare: seriile și ACF'), 'tsa_ch1_textbook', 'TSA_ch1_textbook_series', [
    T('Left: the Nile flow with its mean before and after 1898, and two ACFs; right: sunspot numbers and their ACF up to lag 40', 'Stînga: debitul Nilului cu media înainte și după 1898, și două ACF; dreapta: numărul de pete solare și ACF pînă la lagul 40')],
    h='0.64\\textheight')

interp(('the two classic series', 'celor două serii clasice'), [
    (T('Nile: $\\hat\\rho(1) = @{nile.r1}$ and a slow decay; $Q^*(10) = @{nile.q}$ (p @{nile.p})', 'Nilul: $\\hat\\rho(1) = @{nile.r1}$ și o descreștere lentă; $Q^*(10) = @{nile.q}$ (p @{nile.p})'),
     [T('after removing the two means (@{nile.m0} and @{nile.m1}): $\\hat\\rho(1) = @{nile.rr}$, $Q^*(10) = @{nile.qr}$ (p = @{nile.pr})', 'după eliminarea celor două medii (@{nile.m0} și @{nile.m1}): $\\hat\\rho(1) = @{nile.rr}$, $Q^*(10) = @{nile.qr}$ (p = @{nile.pr})'),
      T('a slowly decaying ACF can come from a \\textbf{break}, not from persistence or a unit root', 'o ACF care scade lent poate proveni dintr-o \\textbf{ruptură}, nu din persistență sau dintr-o rădăcină unitară')]),
    (T('Sunspots: a damped wave, minimum $@{sun.min}$ at lag @{sun.minlag}, peak @{sun.peak} at lag @{sun.lag}', 'Petele solare: o undă amortizată, minim $@{sun.min}$ la lagul @{sun.minlag}, vîrf @{sun.peak} la lagul @{sun.lag}'),
     [T('the solar cycle of about 11 years; a stationary series with a cycle, modelled by \\refYule\\ with an AR(2)', 'ciclul solar de circa 11 ani; o serie staționară cu un ciclu, modelată de \\refYule\\ printr-un AR(2)')]),
    T('Read an ACF together with the plot of the series and with what you know about how the data were produced', 'Interpretăm ACF împreună cu graficul seriei și cu ce știm despre modul în care au fost produse datele')])

D.recap(('Two classic series', 'două serii clasice'), [
    T('Slow ACF decay has several causes: unit root, strong persistence, structural break', 'O ACF care scade lent are mai multe cauze posibile: rădăcina unitară, persistența puternică, ruptura structurală'),
    T('Cycles appear as damped waves in the ACF', 'Ciclurile apar ca unde amortizate în ACF')])

# =============================================================================
# 9. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Code}: a first draft of a script that downloads a series, plots it with its ACF and PACF, and runs Ljung--Box for several $m$', '\\textbf{Cod}: o primă versiune a unui script care descarcă o serie, o reprezintă grafic împreună cu ACF și PACF și aplică testul Ljung--Box pentru mai multe valori ale lui $m$'),
    T('\\textbf{Explanation}: a second explanation of a definition, or of why a correlogram looks as it does', '\\textbf{Explicații}: o a doua explicație a unei definiții sau a motivului pentru care o corelogramă arată într-un anumit fel'),
    T('\\textbf{Exploration}: the same diagnostics on many series at once (all BVB stocks, all EU inflation rates)', '\\textbf{Explorare}: aceleași diagnostice pe multe serii deodată (toate acțiunile BVB, toate ratele inflației din UE)'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that loads the Romanian monthly HICP from Eurostat, computes 100 times the monthly log difference since 2005, plots the series with its ACF and PACF up to lag 36, and reports the Ljung-Box statistic for m = 12 and m = 24.}',
        '\\aiprompt{Write Python code that loads the Romanian monthly HICP from Eurostat, computes 100 times the monthly log difference since 2005, plots the series with its ACF and PACF up to lag 36, and reports the Ljung-Box statistic for m = 12 and m = 24.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('The divisor of $\\hat\\gamma(h)$ ($T$, not $T - h$) and whether lag 0 is included in the plot', 'Împărțitorul lui $\\hat\\gamma(h)$ ($T$, nu $T - h$) și dacă lagul 0 este inclus în grafic'),
    T('Which bands are drawn: $\\pm 1.96/\\sqrt{T}$ or Bartlett\'s MA bands; they answer different questions', 'Tipul benzilor desenate: $\\pm 1{,}96/\\sqrt{T}$ sau benzile Bartlett pentru MA; ele răspund la întrebări diferite'),
    T('That Ljung--Box is applied to the stationary transformation, not to the level; and with $m - p - q$ degrees of freedom on residuals', 'Că testul Ljung--Box se aplică transformării staționare, nu nivelului; și cu $m - p - q$ grade de libertate pentru reziduuri'),
    T('That ``significant\'\' is not read as ``large\'\' or ``profitable\'\'', 'Că „semnificativ” nu este citit ca „mare” sau „profitabil”'),
    T('Dates, frequencies and units of the data (seasonally adjusted or not, index base, percent or decimals)', 'Datele, frecvențele și unitățile seriilor (ajustate sezonier sau nu, baza indicelui, procente sau zecimale)'),
    T('Every cited reference: it must exist; check the DOI', 'Fiecare referință citată: trebuie să existe; verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('A time series is one path of a stochastic process; inference from one path needs stationarity and ergodicity', 'O serie de timp este o traiectorie a unui proces stochastic; inferența dintr-o singură traiectorie cere staționaritate și ergodicitate'),
    T('Weak stationarity: constant mean and an autocovariance that depends only on the lag', 'Staționaritatea slabă: media constantă și o autocovarianță care depinde doar de lag'),
    T('White noise (weak, i.i.d., Gaussian) is the building block; the random walk is its cumulative sum and is not stationary', 'Zgomotul alb (slab, i.i.d., gaussian) este piesa de bază; mersul aleator este suma lui cumulată și nu este staționar'),
    T('Sample ACF and PACF with $\\pm 1.96/\\sqrt{T}$ bands; Ljung--Box tests many lags jointly', 'ACF și PACF de selecție cu benzile $\\pm 1{,}96/\\sqrt{T}$; testul Ljung--Box testează simultan mai multe laguri'),
    T('Real series need transformations: log or Box--Cox for the variance, differences for trend and season', 'Seriile reale au nevoie de transformări: logaritm sau Box--Cox pentru varianță, diferențe pentru trend și sezon'),
    T('Returns: nearly uncorrelated, but their squares are strongly correlated', 'Randamentele: aproape necorelate, dar pătratele lor sînt puternic corelate')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.4}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Autocovariance, ACF', 'Autocovarianța, ACF') + ' & $\\gamma(h) = \\mathrm{Cov}(X_t, X_{t+h})$, \\quad $\\rho(h) = \\gamma(h)/\\gamma(0)$',
     'MA(1) & $\\gamma(0) = \\sigma^2(1 + \\theta^2)$, \\quad $\\rho(1) = \\theta/(1 + \\theta^2)$, \\quad $\\rho(h) = 0$, $|h| \\ge 2$',
     'AR(1) & $\\gamma(0) = \\sigma^2/(1 - \\phi^2)$, \\quad $\\rho(h) = \\phi^{|h|}$',
     T('Random walk', 'Mersul aleator') + ' & $\\mathrm{Var}(X_t) = t\\sigma^2$, \\quad $\\mathrm{Cov}(X_t, X_s) = \\min(t, s)\\,\\sigma^2$',
     'Wold & $X_t = \\sum_{j \\ge 0}\\psi_j\\varepsilon_{t-j} + \\eta_t$, \\quad $\\sum_j\\psi_j^2 < \\infty$',
     T('Sample ACF', 'ACF de selecție') + ' & $\\hat\\rho(h) = \\sum_{t=1}^{T-h}(x_t - \\bar x)(x_{t+h} - \\bar x)/\\sum_{t=1}^{T}(x_t - \\bar x)^2$, \\quad $\\pm 1.96/\\sqrt{T}$',
     'PACF & $\\phi_{11} = \\rho(1)$, \\quad $\\phi_{22} = (\\rho(2) - \\rho(1)^2)/(1 - \\rho(1)^2)$',
     'Ljung--Box & $Q^*(m) = T(T + 2)\\sum_{h=1}^{m}\\hat\\rho(h)^2/(T - h) \\sim \\chi^2(m)$',
     'Box--Cox & $w = (y^\\lambda - 1)/\\lambda$, \\quad $w = \\ln y$ ($\\lambda = 0$)'],
    size='scriptsize') + '}')

D.frame(T('Self-assessment', 'Autoevaluare'), items(
    (T('\\textbf{Question}: $X_t = \\varepsilon_t - 0.5\\varepsilon_{t-1}$, $\\sigma^2 = 4$. What are $\\gamma(0)$ and $\\rho(1)$?', '\\textbf{Întrebare}: $X_t = \\varepsilon_t - 0{,}5\\varepsilon_{t-1}$, $\\sigma^2 = 4$. Cît sînt $\\gamma(0)$ și $\\rho(1)$?'),
     [T('\\textbf{Answer}: $\\gamma(0) = 4 \\times 1.25 = 5$; $\\rho(1) = -0.5/1.25 = -0.4$', '\\textbf{Răspuns}: $\\gamma(0) = 4 \\times 1{,}25 = 5$; $\\rho(1) = -0{,}5/1{,}25 = -0{,}4$')]),
    (T('\\textbf{Question}: a correlogram of 40 lags of a series with $T = 400$ shows two bars just outside $\\pm 0.098$. Is the series correlated?', '\\textbf{Întrebare}: o corelogramă cu 40 de laguri pentru o serie cu $T = 400$ arată două bare ușor în afara lui $\\pm 0{,}098$. Este seria corelată?'),
     [T('\\textbf{Answer}: not necessarily: about 2 of 40 bars cross by chance; run Ljung--Box', '\\textbf{Răspuns}: nu neapărat: aproximativ 2 din 40 de bare ies din întîmplare; aplicăm testul Ljung--Box')]),
    (T('\\textbf{Question}: after differencing, $\\hat\\rho(1) = -0.48$ and nothing else is significant. What does this suggest?', '\\textbf{Întrebare}: după diferențiere, $\\hat\\rho(1) = -0{,}48$ și nimic altceva nu este semnificativ. Ce sugerează acest lucru?'),
     [T('\\textbf{Answer}: over-differencing: the original series was probably already stationary', '\\textbf{Răspuns}: supradiferențiere: seria inițială era probabil deja staționară')]),
    T('Next: Chapter 2, ARMA models: identification with the ACF and PACF, estimation, diagnostics and forecasts', 'Urmează: Capitolul 2, modele ARMA: identificarea cu ACF și PACF, estimarea, diagnosticarea și prognoza')))

D.references(bib())

if __name__ == '__main__':
    for _k, _v in list(V.items()):   # a true minus sign for negative numbers, in text and in math
        if isinstance(_v, str) and _v.startswith('⁅-'):
            V[_k] = '⁅\\ensuremath{-}' + _v[2:]
    finalize(D.write(V))
