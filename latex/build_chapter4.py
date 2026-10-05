r"""
build_chapter4.py -- Capitolul 4 (Sezonalitate și prognoză: SARIMA, TBATS, Prophet), EN + RO dintr-o singură sursă
=================================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_04/ch4_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele rezolvate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter4_seasonality_forecasting.tex
  RO/Cursuri/capitol4_sezonalitate_prognoza.tex
Rulare:
  python3 Quantlets/Ch_04/generate_all_charts.py
  python3 latex/build_chapter4.py && python3 latex/tsa_build.py compile 4
"""

import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, cols, items, table, photo   # noqa: E402
from ch4_common import QLURL, REFS, T, bib, date, finalize, load, pv, quarter   # noqa: E402

N = load()
V = Values()
D = Deck(4, 'lecture', refs=REFS)
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
    'pan_am': ('ch4_pan_am_707_1958.jpg', C + 'Three_Pan_Am_Boeing_707_awaiting_delivery.jpg',
               T('Photo: unknown author (1958); public domain; Wikimedia Commons', 'Foto: autor necunoscut (1958); domeniu public; Wikimedia Commons')),
    'engle': ('ch4_robert_engle_2017.jpg', C + 'Robert_Engle_SantiagoWEAI2017.png',
              T('Photo', 'Foto') + ': Econterms (2017); CC BY-SA 4.0; Wikimedia Commons'),
    'census': ('ch4_census_bureau_2007.jpg', C + 'Census_Bureau_headquarters,_Suitland,_Maryland,_2007.jpg',
               T('Photo: United States Census Bureau (2007); public domain; Wikimedia Commons', 'Foto: United States Census Bureau (2007); domeniu public; Wikimedia Commons')),
    'eggs': ('ch4_romanian_easter_eggs.jpg', C + 'Painted_Romanian_Easter_Eggs.jpg',
             T('Photo', 'Foto') + ': jackmac34 (Pixabay); CC0; Wikimedia Commons'),
    'pdf': ('ch4_portile_de_fier_ii.jpg', C + 'Por\\%C8\\%9Bile_de_Fier_II_(01).jpg',
            T('Photo', 'Foto') + ': Nenea hartia (2016); CC BY-SA 4.0; Wikimedia Commons'),
    'mamaia': ('ch4_mamaia_2013.jpg', C + 'Mamaia_Beach_(September_2013).JPG',
               T('Photo', 'Foto') + ': Razvan Socol (2013); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.50\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# CIFRE
# =============================================================================
SE = N['series']
V.int('gdp.T', SE['gdp_T'])
V.raw('gdp.q1', quarter(SE['gdp_last']))
for i, v in enumerate(SE['gdp_q']):
    V.put(f'gdp.m{i + 1}', v, 1)
V.put('gdp.q4q1', 100 * SE['gdp_q4_q1'], 0)
V.put('tour.ratio', SE['tour_aug_jan'], 1)
V.put('infl.jun', SE['infl_month_mean'][5], 2)
V.put('infl.oct', SE['infl_month_mean'][9], 2)
V.int('load.T', SE['load_T'])
V.put('load.mean', SE['load_mean'], 2)
SP = N['splot']
V.put('sp.aug', SP['tour19_aug'], 2)
V.put('sp.jan', SP['tour19_jan'], 2)
V.put('sp.apr19', SP['tour19_apr'], 2)
V.put('sp.apr20', SP['tour20_apr'], 3)

GD = N['gdpdiff']
V.put('gd.d1.r4', GD['d1']['r'][3], 2)
V.put('gd.d1.r8', GD['d1']['r'][7], 2)
V.put('gd.d1.r1', GD['d1']['r'][0], 2)
V.put('gd.d4.r1', GD['d4']['r'][0], 2)
V.put('gd.d4.r4', GD['d4']['r'][3], 2)
V.put('gd.dd.r1', GD['dd']['r'][0], 2)
V.put('gd.dd.r4', GD['dd']['r'][3], 2)
V.put('gd.band', 1.96 / math.sqrt(GD['dd']['T']), 2)
V.put('gd.d1.sd', GD['d1']['sd'], 1)
V.put('gd.d4.sd', GD['d4']['sd'], 1)

TE = N['tests']
H = TE['hegy']
CV = TE['hegy_cv']
V.put('hegy.t1', H['t1'], 2)
V.put('hegy.t2', H['t2'], 2)
V.put('hegy.F', H['F34'], 2)
V.put('hegy.c1', CV['t1'], 2)
V.put('hegy.c2', CV['t2'], 2)
V.put('hegy.cF', CV['F34'], 2)
V.raw('hegy.n', str(H['n']))
V.raw('hegy.k', str(TE['hegy_k']))
V.int('hegy.nrep', CV['nrep'])
assert H['t1'] > CV['t1'] and H['t2'] > CV['t2'] and H['F34'] < CV['F34']   # the text: no seasonal root rejected
TT = {r['series']: r for r in TE['table']}
assert TT['GDP']['ocsb'] == 1 and TT['GDP']['ch'] == 0

TH = N['theory']
V.put('th.r1', TH['airline']['acf'][0], 3)
V.put('th.r11', TH['airline']['acf'][10], 3)
V.put('th.r12', TH['airline']['acf'][11], 3)
V.put('th.sar24', TH['sar']['acf'][23], 2)
V.put('th.sar36', TH['sar']['acf'][35], 3)
V.put('th.sma24', TH['sma']['pacf'][23], 3)

AI = N['air']
V.raw('air.T', str(AI['T']))
V.raw('air.Tw', str(AI['Tw']))
V.put('air.r1', AI['r1'], 2)
V.put('air.r3', AI['r3'], 2)
V.put('air.r12', AI['r12'], 2)
V.put('air.r13', AI['r13'], 2)
V.put('air.band', AI['band'], 2)
AF = N['airfc']
V.put('af.th', AF['theta'], 3)
V.put('af.ths', AF['theta_se'], 3)
V.put('af.Th', AF['Theta'], 3)
V.put('af.Ths', AF['Theta_se'], 3)
V.put('af.fth', AF['full_theta'], 3)
V.put('af.fths', AF['full_theta_se'], 3)
V.put('af.fTh', AF['full_Theta'], 3)
V.put('af.fThs', AF['full_Theta_se'], 3)
V.put('af.sig', 100 * AF['full_sigma'], 1)
V.put('af.lbp', AF['lb24']['lb_p'], 2)
V.put('af.mape', AF['mape_sarima'], 1)
V.put('af.mapesn', AF['mape_snaive'], 1)
V.put('af.f', AF['f_last'], 0)
V.put('af.lo', AF['lo_last'], 0)
V.put('af.hi', AF['hi_last'], 0)
V.put('af.act', AF['act_last'], 0)
V.put('af.w1', AF['width_1'], 0)
V.put('af.w24', AF['width_24'], 0)

GM = N['gdpm']
assert GM['best_aicc'] == '(1,1,1)(0,1,1)4' and GM['best_bic'] == '(0,1,0)(0,1,1)4'
GR = {r['model']: r for r in GM['rows']}
ROWS = ['(0,1,1)(0,1,1)4', '(0,1,0)(0,1,1)4', '(1,1,0)(0,1,1)4', '(1,1,1)(0,1,1)4', '(0,1,2)(0,1,1)4', '(0,1,1)(1,1,0)4',
        '(0,1,1)(1,1,1)4', '(0,1,1)(0,1,0)4']
for i, m in enumerate(ROWS):
    V.put(f'gm{i}.aicc', GR[m]['aicc'], 1)
    V.put(f'gm{i}.bic', GR[m]['bic'], 1)
    V.raw(f'gm{i}.p', pv(GR[m]['lb8_p']))
    V.put(f'gm{i}.sig', GR[m]['sigma'], 2)
A0 = GR['(0,1,1)(0,1,1)4']
V.put('gm.air.th', A0['params']['ma.L1'], 3)
V.put('gm.air.ths', A0['se']['ma.L1'], 3)
V.put('gm.air.Th', A0['params']['ma.S.L4'], 3)
A3 = GR['(1,1,1)(0,1,1)4']
V.put('gm.c.phi', A3['params']['ar.L1'], 2)
V.put('gm.c.th', A3['params']['ma.L1'], 2)
V.raw('gm.q0', quarter(GM['first']))
V.raw('gm.q1', quarter(GM['last']))
V.int('gm.T', GM['T'])
G2 = N['gdpdiag']
V.put('gdg.q8', G2['lb8']['lb'], 2)
V.raw('gdg.df', str(G2['lb8']['df']))
V.raw('gdg.p8', pv(G2['lb8']['lb_p']))
V.raw('gdg.p12', pv(G2['lb12']['lb_p']))
V.put('gdg.jb', G2['jb'], 1)
V.raw('gdg.jbp', pv(G2['jb_p']))
V.put('gdg.jbx', G2['jb_ex2020'], 1)
V.raw('gdg.outd', quarter(G2['out_d']))
V.put('gdg.outv', G2['out_v'], 1)
V.put('gdg.outz', G2['out_z'], 1)
V.put('gdg.sd', G2['sd'], 2)
GF = N['gdpfc']
V.put('gf.Th', GF['params']['ma.S.L4'], 3)
V.put('gf.Ths', GF['se']['ma.S.L4'], 3)
V.put('gf.sig', GF['sigma'], 2)
V.put('gf.al', 1 + GF['params']['ma.S.L4'], 2)
for i in range(4):
    V.put(f'gf.f{i + 1}', GF['f'][i], 1)
    V.put(f'gf.g{i + 1}', GF['g_f'][i], 2)
V.put('gf.g8', GF['g_f'][7], 2)
V.put('gf.lo1', GF['lo95'][0], 1)
V.put('gf.hi1', GF['hi95'][0], 1)
V.put('gf.lo8', GF['lo95'][7], 1)
V.put('gf.hi8', GF['hi95'][7], 1)
V.put('gf.last', GF['last_obs'], 1)
V.raw('gf.lastd', quarter(GF['last_obs_d']))
V.raw('gf.q0', quarter(GF['first_f']))
V.raw('gf.q1', quarter(GF['last_f']))

SA = N['sa']
for i, v in enumerate(SA['fq']):
    V.put(f'sa.f{i + 1}', v, 2)
V.put('sa.nsa', SA['qoq_nsa'][-2], 1)
V.put('sa.sca', SA['qoq_sca'][-2], 2)
V.raw('sa.q', quarter(SA['qoq_dates'][-2]))
V.put('sa.dec', SA['f_dec'], 2)
V.put('sa.jan', SA['f_jan'], 2)
V.put('sa.sum.n', SA['sum_nsa_2024'], 1)
V.put('sa.sum.s', SA['sum_sca_2024'], 1)

FO = N['fourier']
for K in (1, 2, 3, 6):
    V.put(f'fo.r{K}', FO[f'K{K}']['r2'], 3)

EA = N['easter']
FP, FS = EA['food']['params'], EA['food']['se']
TP, TS = EA['total']['params'], EA['total']['se']
V.put('ea.f.e', 100 * FP['easter'], 1)
V.put('ea.f.es', 100 * FS['easter'], 1)
V.put('ea.f.t', FP['easter'] / FS['easter'], 1)
V.put('ea.f.wd', 100 * FP['wd'], 2)
V.put('ea.f.wds', 100 * FS['wd'], 2)
V.put('ea.t.e', 100 * TP['easter'], 1)
V.put('ea.t.es', 100 * TS['easter'], 1)
V.put('ea.t.wd', 100 * TP['wd'], 2)
V.put('ea.t.wds', 100 * TS['wd'], 2)
V.put('ea.f.ao', 100 * FP['ao_2020_04'], 0)
V.put('ea.t.ao', 100 * TP['ao_2020_04'], 0)
V.put('ea.slope', EA['slope'], 1)
V.raw('ea.may', ', '.join(EA['months_may']))
V.raw('ea.may.ro', '; '.join(EA['months_may']))
V.raw('ea.d24', date(EA['easter_dates']['2024']))
V.raw('ea.d25', date(EA['easter_dates']['2025']))
V.raw('ea.d26', date(EA['easter_dates']['2026']))
V.raw('ea.d27', date(EA['easter_dates']['2027']))
V.raw('ea.f0', date(EA['food']['first'], day=False))
V.raw('ea.f1', date(EA['food']['last'], day=False))

HO = N['hourly']
V.put('ho.r24', HO['r24'], 2)
V.put('ho.r168', HO['r168'], 2)
V.put('ho.r12', HO['r12'], 2)
V.raw('ho.peak', str(HO['peak_h']))
V.raw('ho.trough', str(HO['trough_h']))
V.put('ho.pv', HO['peak_v'], 1)
V.put('ho.tv', HO['trough_v'], 1)
V.int('ho.T', HO['T_all'])
MS = N['mstl']
V.put('ms.s24', 100 * MS['share24'], 0)
V.put('ms.s168', 100 * MS['share168'], 0)
V.put('ms.sr', 100 * MS['share_r'], 0)
V.put('ms.a24', MS['amp24'], 1)
V.put('ms.a168', MS['amp168'], 1)
DY = N['daily']
V.put('dy.sun', -100 * DY['sun_wed'], 0)
V.put('dy.sat', -100 * DY['sat_wed'], 0)
V.put('dy.jan', 100 * DY['jan_jul'], 0)
V.put('dy.low', DY['low_vals'][0], 2)
V.raw('dy.lowd', date(DY['low_days'][0]))
V.put('dy.mean', DY['mean'], 2)
V.put('dy.eas', DY['easter_mean'], 2)
V.raw('dy.nhol', str(DY['n_hol']))
DH = N['dhr']
V.put('dh.eas', -DH['coef']['easter'], 2)
V.put('dh.xm', -DH['coef']['xmas'], 2)
V.put('dh.hol', -DH['coef']['holiday'], 2)
V.put('dh.hols', DH['se']['holiday'], 2)
V.put('dh.sat', DH['coef']['sat'], 2)
V.put('dh.wed', DH['coef']['wed'], 2)
V.put('dh.mon', DH['coef']['mon'], 2)
V.put('dh.ar', DH['ar'][0], 2)
V.put('dh.ma', DH['ma'][0], 2)
V.put('dh.sig', 1000 * DH['sigma'], 0)
V.raw('dh.lbp', pv(DH['lb14']['lb_p']))
V.raw('dh.order', f"({DH['order'][0]},{DH['order'][1]},{DH['order'][2]})")
assert DH['order'] == [1, 1, 1]
TB = N['tbats']
V.raw('tb.k7', str(TB['harmonics'][0]))
V.raw('tb.k365', str(TB['harmonics'][1]))
V.raw('tb.p', str(TB['p']))
V.raw('tb.q', str(TB['q']))
V.raw('tb.states', str(TB['n_states']))
PR = N['prophet']
V.put('pr.sun', PR['weekly'][6], 2)
V.put('pr.sat', PR['weekly'][5], 2)
V.put('pr.wed', PR['weekly'][2], 2)
V.put('pr.ymax', PR['yearly_max'], 2)
V.put('pr.ymin', PR['yearly_min'], 2)
V.raw('pr.ymaxd', date(PR['yearly_argmax']))
V.raw('pr.ymind', date(PR['yearly_argmin']))
V.put('pr.xmas', PR['hol']['Christmas'], 2)
V.put('pr.ny', PR['hol']['New Year'], 2)
V.put('pr.eas', PR['hol']['Easter'], 2)
V.put('pr.ch', PR['hol']["Children's Day"], 2)
V.put('pr.t0', PR['trend0'], 2)
V.put('pr.t1', PR['trend1'], 2)
V.raw('pr.ncp', str(PR['n_cp']))

CVN = N['cv']
TAB = CVN['tab']
MODELS = ['Seasonal naive', 'ETS', 'SARIMA', 'DHR', 'TBATS', 'Prophet', 'Combination']
KEYS = {'Seasonal naive': 'sn', 'ETS': 'ets', 'SARIMA': 'sar', 'DHR': 'dhr', 'TBATS': 'tb', 'Prophet': 'pro', 'Combination': 'comb'}
for m in MODELS:
    k = KEYS[m]
    V.put(f'cv.{k}.mae', 1000 * TAB[m]['MAE'], 0)
    V.put(f'cv.{k}.rmse', 1000 * TAB[m]['RMSE'], 0)
    V.put(f'cv.{k}.mase', TAB[m]['MASE'], 2)
assert CVN['best'] == 'DHR'
assert sorted(MODELS[:-1], key=lambda m: TAB[m]['MAE'])[:2] == ['DHR', 'Prophet']
V.put('cv.scale', 1000 * CVN['scale'], 0)
V.raw('cv.n', str(CVN['n_origins']))
V.raw('cv.o0', date(CVN['first_origin']))
V.raw('cv.o1', date(CVN['last_origin']))
o = CVN['orders']
V.raw('cv.sar', f"({o['sarima'][0]},{o['sarima'][1]},{o['sarima'][2]})")
DMv = CVN['dm']
for m in ['ETS', 'SARIMA', 'DHR', 'TBATS', 'Prophet', 'Combination']:
    k = KEYS[m]
    d = DMv[f'{m}|Seasonal naive|se']
    V.put(f'dm.{k}.sn', d['hln'], 2)
    V.raw(f'dm.{k}.snp', pv(d['p']))
for m in ['Seasonal naive', 'ETS', 'SARIMA', 'TBATS', 'Prophet', 'Combination']:
    k = KEYS[m]
    d = DMv[f'{m}|DHR|se']
    V.put(f'dm.{k}.dh', d['hln'], 2)
    V.raw(f'dm.{k}.dhp', pv(d['p']))
LF = N['loadfc']
V.raw('lf.o', date(LF['origin']))
for m in MODELS:
    V.put(f'lf.{KEYS[m]}', 1000 * LF['mae'][m], 0)
CB = N['comb']
V.put('cb.w0', CB['mae_w0'], 0)
V.put('cb.w1', CB['mae_w1'], 0)
V.put('cb.half', CB['mae_half'], 0)
V.put('cb.wb', CB['w_best'], 2)
V.put('cb.best', CB['mae_best'], 0)
V.put('cb.corr', CB['corr'], 2)
GC = N['gdpcv']
V.raw('gc.n', str(GC['n']))
V.raw('gc.o0', quarter(GC['first']))
V.raw('gc.o1', quarter(GC['last']))
V.raw('gc.nex', str(GC['n_ex']))
for m, k in [('SARIMA', 'sar'), ('ETS', 'ets'), ('Seasonal naive', 'sn'), ('Combination', 'comb')]:
    for h in (1, 4):
        V.put(f'gc.{k}.{h}', GC['tab'][m][str(h)]['mae'], 2)
        V.put(f'gc.{k}.x{h}', GC['tab_ex'][m][str(h)], 2)
for key in ['h1|ETS', 'h1|Seasonal naive', 'h4|ETS', 'h4|Seasonal naive']:
    kk = key.replace('|', '.').replace(' ', '')
    V.put(f'gc.dm.{kk}', GC['dm'][key]['hln'], 2)
    V.raw(f'gc.dmp.{kk}', pv(GC['dm'][key]['p']))

# ---- worked examples (computed here)
# (1) a fixed seasonal pattern with a linear trend: y_t = 10 + 0.5 t + S_q
Sq = [-3, 1, 0, 2]
yy = [10 + 0.5 * t + Sq[(t - 1) % 4] for t in range(1, 10)]
for t in range(1, 10):
    V.put(f'we1.y{t}', yy[t - 1], 1)
V.put('we1.d4', yy[8] - yy[4], 1)
V.put('we1.dd', (yy[8] - yy[4]) - (yy[7] - yy[3]), 1)
# (2) airline expansion, s = 12, theta = -0.4, Theta = -0.6
th, Th = -0.4, -0.6
V.put('we2.tT', th * Th, 2)
V.put('we2.g0', (1 + th ** 2) * (1 + Th ** 2), 4)
V.put('we2.r1', th / (1 + th ** 2), 3)
V.put('we2.r12', Th / (1 + Th ** 2), 3)
V.put('we2.r11', th / (1 + th ** 2) * Th / (1 + Th ** 2), 3)
# (3) forecasting a seasonal MA: y_t = y_{t-4} + e_t + Theta e_{t-4}, Theta = -0.6
yl, el = [100, 120, 130, 140], [2.0, -1.0, 0.0, 3.0]
f3 = [yl[i] + Th * el[i] for i in range(4)]
for i in range(4):
    V.put(f'we3.f{i + 1}', f3[i], 1)
V.put('we3.f5', f3[0], 1)
# (4) Diebold-Mariano by hand: window-average squared-error differences
d4 = np.array([-0.30, 0.10, -0.50, -0.20, -0.40])
db, s2 = d4.mean(), np.mean((d4 - d4.mean()) ** 2)
dmv = db / math.sqrt(s2 / len(d4))
V.put('we4.db', db, 2)
V.put('we4.s2', s2, 3)
V.put('we4.dm', dmv, 2)
V.put('we4.hln', math.sqrt((len(d4) - 1) / len(d4)) * dmv, 2)
V.put('we4.crit', stats.t.ppf(0.975, len(d4) - 1), 2)
V.put('we4.p', 2 * stats.t.sf(abs(math.sqrt((len(d4) - 1) / len(d4)) * dmv), len(d4) - 1), 3)
# (5) combination: sigma1 = 1, sigma2 = 1.2, rho = 0.75
s1, s2_, rho = 1.0, 1.2, 0.75
V.put('we5.half', (s1 ** 2 + s2_ ** 2 + 2 * rho * s1 * s2_) / 4, 3)
V.put('we5.w', (s2_ ** 2 - rho * s1 * s2_) / (s1 ** 2 + s2_ ** 2 - 2 * rho * s1 * s2_), 2)
wopt = (s2_ ** 2 - rho * s1 * s2_) / (s1 ** 2 + s2_ ** 2 - 2 * rho * s1 * s2_)
V.put('we5.opt', wopt ** 2 * s1 ** 2 + (1 - wopt) ** 2 * s2_ ** 2 + 2 * wopt * (1 - wopt) * rho * s1 * s2_, 3)
V.put('fourier.k', 2 * 5 + 2 * 4, 0)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: many economic series repeat a pattern every year, week or day; how do we model this pattern, forecast with it, and decide which forecast is better?',
       '\\textbf{Întrebarea}: multe serii economice repetă un tipar în fiecare an, săptămînă sau zi; cum modelăm acest tipar, cum prognozăm cu el și cum decidem care prognoză este mai bună?'),
     [T('three answers: \\textbf{difference} the pattern away (SARIMA), \\textbf{model} it with regressors (Fourier terms, TBATS, Prophet), or \\textbf{remove} it (seasonal adjustment)',
        'trei răspunsuri: \\textbf{eliminăm} tiparul prin diferențiere (SARIMA), îl \\textbf{modelăm} cu regresori (termeni Fourier, TBATS, Prophet) sau îl \\textbf{scoatem} din date (ajustarea sezonieră)')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('seasonal differences and seasonal unit roots (HEGY, Canova--Hansen, OCSB)', 'diferențe sezoniere și rădăcini unitare sezoniere (HEGY, Canova--Hansen, OCSB)'),
      T('SARIMA$(p,d,q)(P,D,Q)_s$: identification, estimation, diagnostics, forecasts; the airline model and Romanian GDP', 'SARIMA$(p,d,q)(P,D,Q)_s$: identificare, estimare, diagnosticare, prognoze; modelul airline și PIB-ul României'),
      T('seasonal adjustment in practice (X-13ARIMA-SEATS, Eurostat), Fourier terms and calendar effects (Orthodox Easter)', 'ajustarea sezonieră în practică (X-13ARIMA-SEATS, Eurostat), termeni Fourier și efecte de calendar (Paștele ortodox)'),
      T('multiple seasonality in electricity load: MSTL, dynamic harmonic regression, TBATS, Prophet', 'sezonalitatea multiplă a consumului de electricitate: MSTL, regresia armonică dinamică, TBATS, Prophet'),
      T('forecast evaluation (time-series cross-validation, Diebold--Mariano test) and forecast combination', 'evaluarea prognozelor (validare încrucișată pentru serii de timp, testul Diebold--Mariano) și combinarea prognozelor')]),
    T('It builds on Chapter 0 (components, Holt--Winters, MASE) and Chapters 2--3 (ARMA, unit roots, ARIMA): the same ideas, at the seasonal lag $s$',
      'Capitolul se sprijină pe Capitolul 0 (componente, Holt--Winters, MASE) și pe Capitolele 2--3 (ARMA, rădăcini unitare, ARIMA): aceleași idei, la decalajul sezonier $s$'),
    T('Seminar 4 comes before this lecture: it gives the formulas; here we derive them and use them on data',
      'Seminarul 4 are loc înaintea acestui curs: dă formulele; aici le deducem și le aplicăm pe date')), 'footnotesize')

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Recognise deterministic and stochastic seasonality and choose seasonal differencing with HEGY, Canova--Hansen and OCSB', 'Recunoașteți sezonalitatea deterministă și pe cea stochastică și alegeți diferențierea sezonieră cu testele HEGY, Canova--Hansen și OCSB'),
    T('Write, identify, estimate, check and forecast a SARIMA$(p,d,q)(P,D,Q)_s$ model, and expand its polynomials', 'Scrieți, identificați, estimați, verificați și prognozați un model SARIMA$(p,d,q)(P,D,Q)_s$ și dezvoltați polinoamele lui'),
    T('Read seasonally adjusted and unadjusted official series correctly, and explain what X-13ARIMA-SEATS does', 'Interpretați corect seriile oficiale ajustate sezonier și neajustate și explicați ce face X-13ARIMA-SEATS'),
    T('Model calendar effects (working days, Orthodox Easter) and multiple seasonality with Fourier terms, MSTL, TBATS and Prophet', 'Modelați efectele de calendar (zile lucrătoare, Paștele ortodox) și sezonalitatea multiplă cu termeni Fourier, MSTL, TBATS și Prophet'),
    T('Compare forecasts with time-series cross-validation, MASE and the Diebold--Mariano test, and combine them', 'Comparați prognozele cu validarea încrucișată pentru serii de timp, MASE și testul Diebold--Mariano și combinați-le')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refHP, Ch.~4--5 (ARIMA and nonstationary models)', 'Manual: \\refHP, cap.~4--5 (modele ARIMA și modele nestaționare)'),
     [T('companion, free online: \\refFPP, Ch.~9 (\\refFPPsar), Ch.~10 (\\refFPPdhr), Ch.~12 (\\refFPPcomplex) and Ch.~13', 'manual însoțitor, gratuit online: \\refFPP, cap.~9 (\\refFPPsar), cap.~10 (\\refFPPdhr), cap.~12 (\\refFPPcomplex) și cap.~13')]),
    T('The classic: \\refBJ, Ch.~9 (seasonal models); seasonal adjustment: \\refLadiray, \\refDagum, \\refESS', 'Lucrarea clasică: \\refBJ, cap.~9 (modele sezoniere); ajustarea sezonieră: \\refLadiray, \\refDagum, \\refESS'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_04}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_04}'),
     [T('\\texttt{statsmodels} (\\texttt{SARIMAX}, \\texttt{MSTL}, \\texttt{ExponentialSmoothing}); optional packages \\texttt{tbats} and \\texttt{prophet}', '\\texttt{statsmodels} (\\texttt{SARIMAX}, \\texttt{MSTL}, \\texttt{ExponentialSmoothing}); pachetele opționale \\texttt{tbats} și \\texttt{prophet}')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter4_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter4_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. SEZONALITATE
# =============================================================================
D.section('Seasonality in economic data', 'Sezonalitatea în datele economice')

D.frame(T('Seasonality: definition and sources', 'Sezonalitatea: definiție și surse'), items(
    (T('\\textbf{Seasonality}: a pattern that repeats with a fixed, known \\textbf{period} $s$ (observations per cycle)', '\\textbf{Sezonalitatea}: un tipar care se repetă cu o \\textbf{perioadă} $s$ fixă și cunoscută (observații pe ciclu)'),
     [T('$s = 4$ (quarters), $s = 12$ (months), $s = 7$ (days in a week), $s = 24$ and $s = 168$ (hours in a day and in a week)', '$s = 4$ (trimestre), $s = 12$ (luni), $s = 7$ (zile într-o săptămînă), $s = 24$ și $s = 168$ (ore într-o zi și într-o săptămînă)'),
      T('unlike a \\textbf{cycle}, whose length is irregular and unknown (Chapter 0)', 'spre deosebire de \\textbf{ciclu}, a cărui lungime este neregulată și necunoscută (Capitolul 0)')]),
    (T('\\textbf{Sources}', '\\textbf{Surse}'),
     [T('weather: heating in winter, the beach in August, harvests', 'vremea: încălzirea iarna, litoralul în august, recoltele'),
      T('the calendar: working days, public holidays, \\textbf{moving holidays} such as Orthodox Easter', 'calendarul: zilele lucrătoare, sărbătorile legale, \\textbf{sărbătorile mobile}, precum Paștele ortodox'),
      T('institutions and habits: the school year, tax deadlines, end-of-year bonuses, the working week', 'instituțiile și obiceiurile: anul școlar, termenele fiscale, primele de sfîrșit de an, săptămîna de lucru')]),
    T('\\textbf{Additive} ($y_t = T_t + S_t + R_t$) or \\textbf{multiplicative} ($y_t = T_tS_tR_t$, additive after logs): Chapter 0; here we work with logs when the swings grow with the level',
      '\\textbf{Aditivă} ($y_t = T_t + S_t + R_t$) sau \\textbf{multiplicativă} ($y_t = T_tS_tR_t$, aditivă după logaritmare): Capitolul 0; aici lucrăm cu logaritmi cînd amplitudinea crește cu nivelul')))

chart(T('Seasonality in six Romanian series', 'Sezonalitatea în șase serii din România'), 'tsa_ch4_series', 'TSA_ch4_seasonal_data', [
    T('Eurostat: GDP (quarterly, not adjusted), retail trade, industrial production, tourism nights, HICP; ENTSO-E: daily electricity load', 'Eurostat: PIB (trimestrial, neajustat), comerțul cu amănuntul, producția industrială, înnoptările turistice, IAPC; ENTSO-E: consumul zilnic de electricitate')],
    h='0.70\\textheight')

interp(('the six series', 'celor șase serii'), [
    (T('GDP: the mean of Q4 is @{gdp.q4q1}\\% above the mean of Q1 (2000--2026); the swing grows with the level: multiplicative, take logs', 'PIB: media trimestrului 4 este cu @{gdp.q4q1}\\% peste media trimestrului 1 (2000--2026); amplitudinea crește cu nivelul: multiplicativă, folosim logaritmi'),
     [T('quarter means (bn EUR, 2010 prices): @{gdp.m1}, @{gdp.m2}, @{gdp.m3}, @{gdp.m4}: construction and agriculture in Q2--Q3', 'mediile trimestriale (mld. EUR, prețuri 2010): @{gdp.m1}; @{gdp.m2}; @{gdp.m3}; @{gdp.m4}: construcțiile și agricultura în T2--T3')]),
    T('Tourism: August has @{tour.ratio} times the nights of January (2015--2019); the 2020 lockdown breaks the pattern', 'Turismul: august are de @{tour.ratio} ori mai multe înnoptări decît ianuarie (2015--2019); izolarea din 2020 rupe tiparul'),
    T('HICP: monthly inflation is lower in June (@{infl.jun}\\% on average, 2010--2019, fresh food) and higher in October (@{infl.oct}\\%)', 'IAPC: inflația lunară este mai mică în iunie (în medie @{infl.jun}\\%, 2010--2019, alimentele proaspete) și mai mare în octombrie (@{infl.oct}\\%)'),
    T('Electricity: two patterns at once, the week (dense oscillation) and the year (winter peaks): \\textbf{multiple seasonality}, Section 8', 'Electricitatea: două tipare simultan, săptămîna (oscilația deasă) și anul (vîrfurile de iarnă): \\textbf{sezonalitate multiplă}, secțiunea 8')], size='footnotesize')

chart(T('Seasonal plot and subseries plot', 'Graficul sezonier și graficul pe subserii'), 'tsa_ch4_seasonal_plot', 'TSA_ch4_seasonal_data', [
    T('Left: one line per year (seasonal plot); right: each quarter of GDP followed over the years, with its mean (subseries plot)', 'Stînga: o linie pentru fiecare an (graficul sezonier); dreapta: fiecare trimestru al PIB-ului urmărit de-a lungul anilor, cu media lui (graficul pe subserii)')],
    h='0.62\\textheight')

D.frame(T('Interpreting the seasonal plots', 'Interpretarea graficelor sezoniere'), cols(items(
    (T('Tourism: one shape every year, a summer peak and a winter trough', 'Turismul: aceeași formă în fiecare an, un vîrf vara și un minim iarna'),
     [T('2019: @{sp.aug} million nights in August, @{sp.jan} million in January', '2019: @{sp.aug} milioane de înnoptări în august, @{sp.jan} milioane în ianuarie'),
      T('April 2020: @{sp.apr20} million (@{sp.apr19} million in April 2019): an outlier, not a new season', 'aprilie 2020: @{sp.apr20} milioane (@{sp.apr19} milioane în aprilie 2019): o valoare aberantă, nu un sezon nou')]),
    (T('GDP subseries: all four quarters rise in parallel', 'Subseriile PIB: toate cele patru trimestre cresc paralel'),
     [T('a stable seasonal shape on top of a common trend', 'o formă sezonieră stabilă peste un trend comun'),
      T('if the lines crossed, the seasonal pattern would be changing: stochastic seasonality (next section)', 'dacă liniile s-ar intersecta, tiparul sezonier s-ar schimba: sezonalitate stochastică (secțiunea următoare)')])),
    ph('mamaia', T('Mamaia, on the Black Sea, September 2013', 'Mamaia, pe litoralul Mării Negre, septembrie 2013'), h='0.36\\textheight'),
    wl='0.58', wr='0.38'), 'footnotesize')

D.recap(('Seasonality in economic data', 'sezonalitatea în datele economice'), [
    T('Seasonality repeats with a known period $s$; a cycle does not', 'Sezonalitatea se repetă cu o perioadă cunoscută $s$; ciclul nu'),
    T('Weather, the calendar and institutions create it; moving holidays shift it between months', 'Vremea, calendarul și instituțiile o creează; sărbătorile mobile o mută între luni'),
    T('Romanian GDP, tourism, inflation and electricity load are all seasonal; electricity has several periods at once', 'PIB-ul, turismul, inflația și consumul de electricitate din România sînt toate sezoniere; electricitatea are mai multe perioade simultan'),
    T('Seasonal and subseries plots show whether the pattern is stable', 'Graficele sezoniere și pe subserii arată dacă tiparul este stabil')])

# =============================================================================
# 2. DIFERENȚIERE SEZONIERĂ
# =============================================================================
D.section('Seasonal differencing and seasonal unit roots', 'Diferențierea sezonieră și rădăcinile unitare sezoniere')

D.frame(T('The seasonal difference', 'Diferența sezonieră'), items(
    (T('\\textbf{Seasonal difference}: $\\Delta_s y_t = (1 - L^s)y_t = y_t - y_{t-s}$', '\\textbf{Diferența sezonieră}: $\\Delta_s y_t = (1 - L^s)y_t = y_t - y_{t-s}$'),
     [T('with logs: $100\\,\\Delta_4\\ln Y_t$ = growth over the same quarter of the previous year, in \\%', 'cu logaritmi: $100\\,\\Delta_4\\ln Y_t$ = creșterea față de același trimestru al anului anterior, în \\%')]),
    (T('A \\textbf{fixed} seasonal pattern $S_t = S_{t-s}$ plus a linear trend: $y_t = a + bt + S_t + \\varepsilon_t$', 'Un tipar sezonier \\textbf{fix} $S_t = S_{t-s}$ plus un trend liniar: $y_t = a + bt + S_t + \\varepsilon_t$'),
     [T('$\\Delta_s y_t = bs + \\varepsilon_t - \\varepsilon_{t-s}$: $\\Delta_s$ removes the pattern and turns the trend into a constant', '$\\Delta_s y_t = bs + \\varepsilon_t - \\varepsilon_{t-s}$: $\\Delta_s$ elimină tiparul și transformă trendul într-o constantă'),
      T('but it creates the non-invertible MA term $\\varepsilon_t - \\varepsilon_{t-s}$: \\textbf{seasonal over-differencing} (as in Chapter 3)', 'dar creează termenul MA neinvertibil $\\varepsilon_t - \\varepsilon_{t-s}$: \\textbf{supradiferențiere sezonieră} (ca în Capitolul 3)')]),
    (T('$\\Delta\\Delta_s = (1 - L)(1 - L^s) = 1 - L - L^s + L^{s+1}$: needed when the trend is stochastic as well', '$\\Delta\\Delta_s = (1 - L)(1 - L^s) = 1 - L - L^s + L^{s+1}$: necesar cînd și trendul este stochastic'),
     [T('the order of the two differences does not matter: the operators commute', 'ordinea celor două diferențe nu contează: operatorii comută')])))

D.frame(T('Deterministic and stochastic seasonality', 'Sezonalitatea deterministă și sezonalitatea stochastică'), items(
    (T('\\textbf{Deterministic}: $y_t = \\sum_{j=1}^{s}\\gamma_jD_{jt} + u_t$, with $D_{jt} = 1$ in season $j$ and $u_t$ stationary', '\\textbf{Deterministă}: $y_t = \\sum_{j=1}^{s}\\gamma_jD_{jt} + u_t$, cu $D_{jt} = 1$ în sezonul $j$ și $u_t$ staționar'),
     [T('the seasonal means $\\gamma_j$ never change; the right tool: seasonal dummies or Fourier terms (Section 7)', 'mediile sezoniere $\\gamma_j$ nu se schimbă niciodată; instrumentul potrivit: variabile dummy sezoniere sau termeni Fourier (secțiunea 7)')]),
    (T('\\textbf{Stochastic}: the \\textbf{seasonal random walk} $y_t = y_{t-s} + \\varepsilon_t$', '\\textbf{Stochastică}: \\textbf{mersul aleator sezonier} $y_t = y_{t-s} + \\varepsilon_t$'),
     [T('each season is its own random walk: $s$ separate walks interleaved', 'fiecare sezon este propriul mers aleator: $s$ mersuri aleatoare separate, intercalate'),
      T('$\\mathrm{Var}(y_t)$ grows without bound; the pattern drifts, and ``summer can become winter\'\'', '$\\mathrm{Var}(y_t)$ crește nelimitat; tiparul se deplasează, iar „vara poate deveni iarnă”'),
      T('the right tool: the seasonal difference $\\Delta_s$', 'instrumentul potrivit: diferența sezonieră $\\Delta_s$')]),
    T('The choice matters: differencing a deterministic pattern over-differences; modelling a stochastic one with dummies leaves a unit root in the residuals',
      'Alegerea contează: diferențierea unui tipar determinist duce la supradiferențiere; modelarea unuia stochastic cu variabile dummy lasă o rădăcină unitară în reziduuri')))

chart(T('The roots of $1 - z^s$: seasonal unit roots', 'Rădăcinile lui $1 - z^s$: rădăcini unitare sezoniere'), 'tsa_ch4_unit_circle', 'TSA_ch4_seasonal_roots', [
    T('$1 - z^4 = (1 - z)(1 + z)(1 + z^2)$: roots $1$, $-1$, $\\pm i$, all on the unit circle', '$1 - z^4 = (1 - z)(1 + z)(1 + z^2)$: rădăcinile $1$, $-1$, $\\pm i$, toate pe cercul unitate')],
    h='0.58\\textheight')

interp(('the seasonal roots', 'rădăcinilor sezoniere'), [
    (T('$\\Delta_s$ imposes $s$ unit roots at once, one for each \\textbf{frequency} $k/s$, $k = 0, \\dots, s - 1$', '$\\Delta_s$ impune simultan $s$ rădăcini unitare, cîte una pentru fiecare \\textbf{frecvență} $k/s$, $k = 0, \\dots, s - 1$'),
     [T('quarterly: $z = 1$ (frequency 0, the trend), $z = -1$ (half a cycle per quarter: period 2 quarters), $z = \\pm i$ (period 4 quarters: the annual cycle)',
        'trimestrial: $z = 1$ (frecvența 0, trendul), $z = -1$ (o jumătate de ciclu pe trimestru: perioada de 2 trimestre), $z = \\pm i$ (perioada de 4 trimestre: ciclul anual)')]),
    T('A series may have some of these roots and not others: then $\\Delta_4$ over-differences at the frequencies it does not need', 'O serie poate avea unele dintre aceste rădăcini și nu pe altele: atunci $\\Delta_4$ supradiferențiază la frecvențele de care nu are nevoie'),
    T('Seasonal unit-root tests ask, frequency by frequency, which roots are present: the HEGY test (next slides)', 'Testele de rădăcină unitară sezonieră întreabă, frecvență cu frecvență, ce rădăcini sînt prezente: testul HEGY (slide-urile următoare)')])

chart(T('Romanian GDP: regular and seasonal differences', 'PIB-ul României: diferențe obișnuite și sezoniere'), 'tsa_ch4_gdp_diff', 'TSA_ch4_seasonal_roots', [
    T('$\\ln$ of real GDP, not adjusted, @{gdp.T} quarters to @{gdp.q1}; top: three transformations; bottom: their sample ACF up to 16 quarters', '$\\ln$ din PIB-ul real neajustat, @{gdp.T} de trimestre pînă în @{gdp.q1}; sus: trei transformări; jos: ACF de selecție pînă la 16 trimestre')],
    h='0.66\\textheight')

interp(('the three transformations', 'celor trei transformări'), [
    (T('$\\Delta\\ln Y$: dominated by the season: ACF $@{gd.d1.r4}$ at lag 4 and $@{gd.d1.r8}$ at lag 8, decaying very slowly; sd @{gd.d1.sd}\\%', '$\\Delta\\ln Y$: dominată de sezon: ACF $@{gd.d1.r4}$ la decalajul 4 și $@{gd.d1.r8}$ la decalajul 8, cu o descreștere foarte lentă; abaterea standard @{gd.d1.sd}\\%'),
     [T('the seasonal analogue of the slow ACF decay of a random walk (Chapter 3)', 'analogul sezonier al descreșterii lente a ACF pentru un mers aleator (Capitolul 3)')]),
    T('$\\Delta_4\\ln Y$ (year-on-year growth): sd @{gd.d4.sd}\\%; ACF $@{gd.d4.r1}$ at lag 1, $@{gd.d4.r4}$ at lag 4: an AR-type pattern', '$\\Delta_4\\ln Y$ (creșterea anuală): abaterea standard @{gd.d4.sd}\\%; ACF $@{gd.d4.r1}$ la decalajul 1, $@{gd.d4.r4}$ la decalajul 4: un tipar de tip AR'),
    (T('$\\Delta\\Delta_4\\ln Y$: ACF $@{gd.dd.r1}$ at lag 1 and $@{gd.dd.r4}$ at lag 4 (band $\\pm @{gd.band}$)', '$\\Delta\\Delta_4\\ln Y$: ACF $@{gd.dd.r1}$ la decalajul 1 și $@{gd.dd.r4}$ la decalajul 4 (banda $\\pm @{gd.band}$)'),
     [T('one negative spike at the seasonal lag: a seasonal MA(1), the signature of the airline model (Section 3)', 'o singură valoare negativă semnificativă la decalajul sezonier: un MA(1) sezonier, semnătura modelului airline (secțiunea 3)')])], size='footnotesize')

D.frame(T('The HEGY test', 'Testul HEGY'), cols(items(
    (T('\\refHEGY: one regression tests every seasonal root of a quarterly series', '\\refHEGY: o singură regresie testează fiecare rădăcină sezonieră a unei serii trimestriale'),
     [T('filters: $y_{1t} = (1 + L + L^2 + L^3)y_t$, $y_{2t} = -(1 - L + L^2 - L^3)y_t$, $y_{3t} = -(1 - L^2)y_t$', 'filtre: $y_{1t} = (1 + L + L^2 + L^3)y_t$, $y_{2t} = -(1 - L + L^2 - L^3)y_t$, $y_{3t} = -(1 - L^2)y_t$')]),
    (T('$\\Delta_4y_t = \\pi_1y_{1,t-1} + \\pi_2y_{2,t-1} + \\pi_3y_{3,t-2} + \\pi_4y_{3,t-1} + \\text{deterministic terms} + \\text{lags of } \\Delta_4y_t + \\varepsilon_t$',
       '$\\Delta_4y_t = \\pi_1y_{1,t-1} + \\pi_2y_{2,t-1} + \\pi_3y_{3,t-2} + \\pi_4y_{3,t-1} + \\text{termeni determiniști} + \\text{decalaje ale lui } \\Delta_4y_t + \\varepsilon_t$'),
     [T('$H_0$: $\\pi_1 = 0$ (root $1$), $t$-test; $\\pi_2 = 0$ (root $-1$), $t$-test; $\\pi_3 = \\pi_4 = 0$ (roots $\\pm i$), $F$-test', '$H_0$: $\\pi_1 = 0$ (rădăcina $1$), test $t$; $\\pi_2 = 0$ (rădăcina $-1$), test $t$; $\\pi_3 = \\pi_4 = 0$ (rădăcinile $\\pm i$), test $F$'),
      T('non-standard distributions, as for Dickey--Fuller: critical values from tables or by simulation', 'distribuții nestandard, ca la Dickey--Fuller: valori critice din tabele sau prin simulare')]),
    T('Rejecting all three: no seasonal differencing needed; rejecting none: $\\Delta_4$ is justified', 'Respingerea tuturor celor trei: nu este nevoie de diferențiere sezonieră; nicio respingere: $\\Delta_4$ este justificată')),
    ph('engle', T('Robert Engle, a co-author of HEGY, 2017', 'Robert Engle, coautor al testului HEGY, 2017'), h='0.40\\textheight'),
    wl='0.64', wr='0.32'), 'footnotesize')

D.frame(T('Canova--Hansen and OCSB', 'Testele Canova--Hansen și OCSB'), items(
    (T('\\textbf{Canova--Hansen} (\\refCH): the null is \\textbf{stable} (deterministic) seasonality', '\\textbf{Canova--Hansen} (\\refCH): ipoteza nulă este sezonalitatea \\textbf{stabilă} (deterministă)'),
     [T('the seasonal analogue of KPSS (Chapter 3): rejecting means the pattern changes over time', 'analogul sezonier al testului KPSS (Capitolul 3): respingerea înseamnă că tiparul se schimbă în timp')]),
    (T('\\textbf{OCSB} (\\refOCSB): the null is a seasonal unit root; a regression of $\\Delta\\Delta_s y_t$ on $\\Delta_s y_{t-1}$ and $\\Delta y_{t-s}$', '\\textbf{OCSB} (\\refOCSB): ipoteza nulă este o rădăcină unitară sezonieră; o regresie a lui $\\Delta\\Delta_s y_t$ pe $\\Delta_s y_{t-1}$ și $\\Delta y_{t-s}$'),
     [T('the default seasonal test of \\texttt{auto\\_arima} in the Python package \\texttt{pmdarima} (\\texttt{nsdiffs})', 'testul sezonier implicit al funcției \\texttt{auto\\_arima} din pachetul Python \\texttt{pmdarima} (\\texttt{nsdiffs})')]),
    (T('As in Chapter 3: tests with opposite nulls answer different questions; use them together', 'Ca în Capitolul 3: testele cu ipoteze nule opuse răspund la întrebări diferite; folosiți-le împreună'),
     [T('and look at the data: seasonal plots, the ACF at lags $s, 2s, 3s$, and the size of $\\hat\\Theta$ after $\\Delta_s$ ($\\hat\\Theta \\approx -1$ signals over-differencing)', 'și priviți datele: graficele sezoniere, ACF la decalajele $s, 2s, 3s$ și mărimea lui $\\hat\\Theta$ după $\\Delta_s$ ($\\hat\\Theta \\approx -1$ semnalează supradiferențierea)')])))

TR = TE['table']
NAMES = {'GDP': T('GDP (quarterly)', 'PIB (trimestrial)'), 'Retail trade': T('Retail trade', 'Comerțul cu amănuntul'),
         'Industrial production': T('Industrial production', 'Producția industrială'), 'Tourism nights': T('Tourism nights', 'Înnoptări turistice'),
         'HICP': T('HICP', 'IAPC')}
D.frame(T('Seasonal unit-root tests on Romanian data', 'Teste de rădăcină unitară sezonieră pe date din România'), table(
    'lrrr', T('\\textbf{HEGY}, $\\ln$ GDP, 2000--2019', '\\textbf{HEGY}, $\\ln$ PIB, 2000--2019') + ' & ' + T('statistic', 'statistica') + ' & ' + T('5\\% critical value', 'valoarea critică 5\\%') + ' & ' + T('decision', 'decizia'),
    [T('$t(\\pi_1)$, root $1$', '$t(\\pi_1)$, rădăcina $1$') + ' & $@{hegy.t1}$ & $@{hegy.c1}$ & ' + T('not rejected', 'nu se respinge'),
     T('$t(\\pi_2)$, root $-1$', '$t(\\pi_2)$, rădăcina $-1$') + ' & $@{hegy.t2}$ & $@{hegy.c2}$ & ' + T('not rejected', 'nu se respinge'),
     T('$F(\\pi_3, \\pi_4)$, roots $\\pm i$', '$F(\\pi_3, \\pi_4)$, rădăcinile $\\pm i$') + ' & $@{hegy.F}$ & $@{hegy.cF}$ & ' + T('not rejected', 'nu se respinge')], size='scriptsize')
    + table('lccc', T('\\textbf{Series} (logs, 2005--2019)', '\\textbf{Seria} (logaritmi, 2005--2019)') + ' & ' + T('$D$ by OCSB', '$D$ după OCSB') + ' & ' + T('$D$ by Canova--Hansen', '$D$ după Canova--Hansen') + ' & ' + T('$d$ by KPSS', '$d$ după KPSS'),
            [NAMES[r['series']] + f" & {r['ocsb']} & {r['ch']} & {r['d_kpss']}" for r in TR], size='scriptsize')
    + items(T('HEGY: constant, seasonal dummies and trend, @{hegy.k} lag of $\\Delta_4y$ (AIC), $n = @{hegy.n}$; critical values from @{hegy.nrep} simulations of $\\Delta_4y_t = \\varepsilon_t$', 'HEGY: constantă, variabile dummy sezoniere și trend, @{hegy.k} decalaj al lui $\\Delta_4y$ (AIC), $n = @{hegy.n}$; valori critice din @{hegy.nrep} de simulări ale lui $\\Delta_4y_t = \\varepsilon_t$'),
            T('Interpretation: for GDP, HEGY and OCSB support $\\Delta_4$ while Canova--Hansen does not reject a stable pattern; with 20 years of quarters both views fit, so we use $\\Delta_4$ and let $\\hat\\Theta$ say how stable the pattern is (Section 5)', 'Interpretare: pentru PIB, HEGY și OCSB susțin $\\Delta_4$, în timp ce Canova--Hansen nu respinge un tipar stabil; cu 20 de ani de date trimestriale ambele perspective se potrivesc, deci folosim $\\Delta_4$ și lăsăm $\\hat\\Theta$ să arate cît de stabil este tiparul (secțiunea 5)'),
            T('Monthly series: both tests choose $D = 0$, a stable pattern; HICP needs $d = 2$ in logs: inflation itself is close to a unit root (Chapter 3)', 'Seriile lunare: ambele teste aleg $D = 0$, un tipar stabil; IAPC are nevoie de $d = 2$ în logaritmi: inflația însăși este aproape de o rădăcină unitară (Capitolul 3)')), 'scriptsize')

D.recap(('Seasonal differencing and seasonal unit roots', 'diferențierea sezonieră și rădăcinile unitare sezoniere'), [
    T('$\\Delta_s = 1 - L^s$ has $s$ unit roots on the circle, one per seasonal frequency', '$\\Delta_s = 1 - L^s$ are $s$ rădăcini unitare pe cerc, cîte una pentru fiecare frecvență sezonieră'),
    T('Deterministic seasonality: dummies or Fourier terms; stochastic seasonality: $\\Delta_s$', 'Sezonalitatea deterministă: variabile dummy sau termeni Fourier; sezonalitatea stochastică: $\\Delta_s$'),
    T('HEGY and OCSB test for seasonal unit roots; Canova--Hansen tests for a stable pattern', 'HEGY și OCSB testează rădăcinile unitare sezoniere; Canova--Hansen testează un tipar stabil'),
    T('Romanian GDP: $\\Delta\\Delta_4\\ln Y$ leaves one spike at lag 4, a seasonal MA', 'PIB-ul României: $\\Delta\\Delta_4\\ln Y$ lasă o singură valoare semnificativă la decalajul 4, un MA sezonier')])

# =============================================================================
# 3. SARIMA
# =============================================================================
D.section('SARIMA models', 'Modele SARIMA')

D.frame(T('The SARIMA$(p,d,q)(P,D,Q)_s$ model', 'Modelul SARIMA$(p,d,q)(P,D,Q)_s$'), items(
    (T('$\\phi(L)\\,\\Phi(L^s)\\,(1 - L)^d(1 - L^s)^D\\,y_t = c + \\theta(L)\\,\\Theta(L^s)\\,\\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$', '$\\phi(L)\\,\\Phi(L^s)\\,(1 - L)^d(1 - L^s)^D\\,y_t = c + \\theta(L)\\,\\Theta(L^s)\\,\\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$'),
     [T('regular polynomials $\\phi(L)$ (order $p$), $\\theta(L)$ (order $q$), as in Chapter 2', 'polinoamele obișnuite $\\phi(L)$ (ordinul $p$), $\\theta(L)$ (ordinul $q$), ca în Capitolul 2'),
      T('seasonal polynomials in $L^s$: $\\Phi(L^s) = 1 - \\Phi_1L^s - \\dots - \\Phi_PL^{Ps}$, $\\Theta(L^s) = 1 + \\Theta_1L^s + \\dots + \\Theta_QL^{Qs}$', 'polinoamele sezoniere în $L^s$: $\\Phi(L^s) = 1 - \\Phi_1L^s - \\dots - \\Phi_PL^{Ps}$, $\\Theta(L^s) = 1 + \\Theta_1L^s + \\dots + \\Theta_QL^{Qs}$'),
      T('$d$, $D$: numbers of regular and seasonal differences, usually 0 or 1', '$d$, $D$: numerele de diferențe obișnuite și sezoniere, de obicei 0 sau 1')]),
    (T('Why multiply? \\refBJ: in a monthly series, January depends on December (the regular part) and on last January (the seasonal part)', 'De ce înmulțim? \\refBJ: într-o serie lunară, ianuarie depinde de decembrie (partea obișnuită) și de ianuarie anul trecut (partea sezonieră)'),
     [T('a two-way table: months in the columns, years in the rows; one model along each direction', 'un tabel cu două intrări: lunile pe coloane, anii pe rînduri; cîte un model pe fiecare direcție')]),
    T('Stationarity: the roots of $\\phi(z)\\Phi(z^s)$ outside the unit circle; invertibility: the roots of $\\theta(z)\\Theta(z^s)$ outside; for a seasonal AR(1) this means $|\\Phi| < 1$',
      'Staționaritatea: rădăcinile lui $\\phi(z)\\Phi(z^s)$ în afara cercului unitate; invertibilitatea: rădăcinile lui $\\theta(z)\\Theta(z^s)$ în afara lui; pentru un AR(1) sezonier aceasta înseamnă $|\\Phi| < 1$')))

D.frame(T('Worked example: the airline model', 'Exemplu rezolvat: modelul airline'), items(
    (T('SARIMA$(0,1,1)(0,1,1)_{12}$ with $\\theta = -0.4$, $\\Theta = -0.6$: $(1 - L)(1 - L^{12})y_t = (1 - 0.4L)(1 - 0.6L^{12})\\varepsilon_t$', 'SARIMA$(0,1,1)(0,1,1)_{12}$ cu $\\theta = -0{,}4$, $\\Theta = -0{,}6$: $(1 - L)(1 - L^{12})y_t = (1 - 0{,}4L)(1 - 0{,}6L^{12})\\varepsilon_t$'),
     [T('left: $y_t - y_{t-1} - y_{t-12} + y_{t-13}$; right: $\\varepsilon_t - 0.4\\varepsilon_{t-1} - 0.6\\varepsilon_{t-12} + @{we2.tT}\\varepsilon_{t-13}$', 'stînga: $y_t - y_{t-1} - y_{t-12} + y_{t-13}$; dreapta: $\\varepsilon_t - 0{,}4\\varepsilon_{t-1} - 0{,}6\\varepsilon_{t-12} + @{we2.tT}\\varepsilon_{t-13}$'),
      T('the coefficient of $\\varepsilon_{t-13}$, $\\theta\\Theta = @{we2.tT}$, is not a new parameter: two parameters plus $\\sigma^2$', 'coeficientul lui $\\varepsilon_{t-13}$, $\\theta\\Theta = @{we2.tT}$, nu este un parametru nou: doi parametri plus $\\sigma^2$')]),
    (T('$w_t = \\Delta\\Delta_{12}y_t$ is an MA(13) with only four nonzero weights; its autocovariances (Chapter 2):', '$w_t = \\Delta\\Delta_{12}y_t$ este un MA(13) cu doar patru ponderi nenule; autocovarianțele lui (Capitolul 2):'),
     [T('$\\gamma(0) = \\sigma^2(1 + \\theta^2)(1 + \\Theta^2) = @{we2.g0}\\,\\sigma^2$', '$\\gamma(0) = \\sigma^2(1 + \\theta^2)(1 + \\Theta^2) = @{we2.g0}\\,\\sigma^2$'),
      T('$\\rho_1 = \\theta/(1 + \\theta^2) = @{we2.r1}$; $\\rho_{12} = \\Theta/(1 + \\Theta^2) = @{we2.r12}$; $\\rho_{11} = \\rho_{13} = \\rho_1\\rho_{12} = @{we2.r11}$', '$\\rho_1 = \\theta/(1 + \\theta^2) = @{we2.r1}$; $\\rho_{12} = \\Theta/(1 + \\Theta^2) = @{we2.r12}$; $\\rho_{11} = \\rho_{13} = \\rho_1\\rho_{12} = @{we2.r11}$'),
      T('all other autocorrelations are zero', 'toate celelalte autocorelații sînt zero')])))

chart(T('Theoretical ACF and PACF of three seasonal models', 'ACF și PACF teoretice pentru trei modele sezoniere'), 'tsa_ch4_theory_acf', 'TSA_ch4_sarima_theory', [
    T('Monthly, $s = 12$; the airline panel is the stationary part $w_t = \\Delta\\Delta_{12}y_t$', 'Lunar, $s = 12$; panoul airline arată partea staționară $w_t = \\Delta\\Delta_{12}y_t$')],
    h='0.66\\textheight')

interp(('the seasonal patterns', 'tiparelor sezoniere'), [
    (T('Seasonal AR(1): the ACF decays geometrically at lags 12, 24, 36 ($0.8$, $@{th.sar24}$, $@{th.sar36}$); the PACF has one spike at lag 12', 'AR(1) sezonier: ACF descrește geometric la decalajele 12, 24, 36 ($0{,}8$; $@{th.sar24}$; $@{th.sar36}$); PACF are o singură valoare nenulă la decalajul 12'),
     [T('the AR rule of Chapter 2, read only at multiples of $s$', 'regula AR din Capitolul 2, citită doar la multiplii lui $s$')]),
    T('Seasonal MA(1): the ACF cuts off after lag 12; the PACF decays at 12, 24, 36 (@{th.sma24} at lag 24)', 'MA(1) sezonier: ACF se anulează după decalajul 12; PACF descrește la 12, 24, 36 (@{th.sma24} la decalajul 24)'),
    (T('Airline: spikes at 1 (@{th.r1}) and 12 (@{th.r12}), with ``satellites\'\' at 11 and 13 (@{th.r11})', 'Airline: valori nenule la 1 (@{th.r1}) și 12 (@{th.r12}), cu „sateliți” la 11 și 13 (@{th.r11})'),
     [T('the satellites are the product $\\rho_1\\rho_{12}$: the fingerprint of the multiplicative structure', 'sateliții sînt produsul $\\rho_1\\rho_{12}$: amprenta structurii multiplicative')])])

D.frame(T('Identification of SARIMA models', 'Identificarea modelelor SARIMA'), table(
    'lll', T('\\textbf{After differencing}', '\\textbf{După diferențiere}') + ' & \\textbf{ACF} & \\textbf{PACF}',
    [T('seasonal AR($P$)', 'AR($P$) sezonier') + ' & ' + T('decays at $s, 2s, \\dots$', 'descrește la $s, 2s, \\dots$') + ' & ' + T('cuts off after lag $Ps$', 'se anulează după decalajul $Ps$'),
     T('seasonal MA($Q$)', 'MA($Q$) sezonier') + ' & ' + T('cuts off after lag $Qs$', 'se anulează după decalajul $Qs$') + ' & ' + T('decays at $s, 2s, \\dots$', 'descrește la $s, 2s, \\dots$'),
     T('regular part', 'partea obișnuită') + ' & ' + T('lags $1, 2, \\dots$ as in Chapter 2', 'decalajele $1, 2, \\dots$ ca în Capitolul 2') + ' & ' + T('lags $1, 2, \\dots$ as in Chapter 2', 'decalajele $1, 2, \\dots$ ca în Capitolul 2'),
     T('product of both', 'produsul celor două') + ' & ' + T('satellites at $ks \\pm j$', 'sateliți la $ks \\pm j$') + ' & ' + T('satellites at $ks \\pm j$', 'sateliți la $ks \\pm j$')], size='footnotesize') + items(
    (T('\\textbf{Steps}', '\\textbf{Pașii}'),
     [T('1. Stabilise the variance (logs or Box--Cox, \\refBoxCox); 2. choose $D$ (seasonal plot, HEGY, OCSB), then $d$ (Chapter 3)', '1. Stabilizați varianța (logaritmi sau Box--Cox, \\refBoxCox); 2. alegeți $D$ (graficul sezonier, HEGY, OCSB), apoi $d$ (Capitolul 3)'),
      T('3. Read $P, Q$ at the seasonal lags and $p, q$ at the first lags; 4. keep the orders small: $P, Q \\le 1$ almost always suffice', '3. Citiți $P, Q$ la decalajele sezoniere și $p, q$ la primele decalaje; 4. păstrați ordinele mici: $P, Q \\le 1$ sînt aproape întotdeauna suficiente')])), 'footnotesize')

D.frame(T('Estimation and model choice', 'Estimarea și alegerea modelului'), items(
    (T('\\textbf{Maximum likelihood} of the differenced series (state-space form, \\texttt{statsmodels} \\texttt{SARIMAX})', '\\textbf{Verosimilitatea maximă} pentru seria diferențiată (forma în spațiul stărilor, \\texttt{SARIMAX} din \\texttt{statsmodels})'),
     [T('$s + 1$ observations are lost to $\\Delta\\Delta_s$; the constant is dropped when $d + D \\ge 2$ (it would mean a quadratic trend)', 'se pierd $s + 1$ observații prin $\\Delta\\Delta_s$; constanta lipsește cînd $d + D \\ge 2$ (ar însemna un trend pătratic)')]),
    (T('\\textbf{Information criteria}: AICc $=$ AIC $+ \\frac{2k(k + 1)}{n - k - 1}$ and BIC, compared \\textbf{only} across models with the same $d$ and $D$', '\\textbf{Criteriile informaționale}: AICc $=$ AIC $+ \\frac{2k(k + 1)}{n - k - 1}$ și BIC, comparate \\textbf{doar} între modele cu aceleași $d$ și $D$'),
     [T('differencing changes the data on which the likelihood is computed', 'diferențierea schimbă datele pe care se calculează verosimilitatea')]),
    (T('\\textbf{Automatic SARIMA}: a stepwise search over $(p, q, P, Q)$ by AICc after choosing $d$ and $D$ by tests (\\refHK)', '\\textbf{SARIMA automat}: o căutare pas cu pas după $(p, q, P, Q)$ cu AICc, după alegerea lui $d$ și $D$ prin teste (\\refHK)'),
     [T('in Python: \\texttt{pmdarima.auto\\_arima(y, m=12)}; always check the residuals and compare with a benchmark', 'în Python: \\texttt{pmdarima.auto\\_arima(y, m=12)}; verificați întotdeauna reziduurile și comparați cu un reper')]),
    T('\\textbf{Diagnostics}: Ljung--Box with lags beyond $s$ (e.g. $m = 2s$) and $m - p - q - P - Q$ degrees of freedom (\\refLB)', '\\textbf{Diagnosticarea}: Ljung--Box cu decalaje dincolo de $s$ (de exemplu $m = 2s$) și $m - p - q - P - Q$ grade de libertate (\\refLB)')))

D.frame(T('Forecasting with SARIMA', 'Prognoza cu modele SARIMA'), items(
    (T('As for ARIMA (Chapter 3): write the model as an equation for $y_t$; replace future shocks by 0, future values by their forecasts, past shocks by residuals',
       'Ca pentru ARIMA (Capitolul 3): scriem modelul ca ecuație pentru $y_t$; înlocuim șocurile viitoare cu 0, valorile viitoare cu prognozele lor și șocurile trecute cu reziduuri'),
     [T('airline: $\\hat y_{T+h} = \\hat y_{T+h-1} + \\hat y_{T+h-s} - \\hat y_{T+h-s-1} + (\\text{MA terms while } h \\le s + 1)$', 'airline: $\\hat y_{T+h} = \\hat y_{T+h-1} + \\hat y_{T+h-s} - \\hat y_{T+h-s-1} + (\\text{termeni MA cît timp } h \\le s + 1)$')]),
    (T('\\textbf{Eventual forecast function} of the airline model: a fixed seasonal pattern on a straight line', '\\textbf{Funcția de prognoză pe termen lung} a modelului airline: un tipar sezonier fix pe o dreaptă'),
     [T('the pattern is an exponentially weighted average of past patterns, with weight $1 + \\Theta$ on the most recent year', 'tiparul este o medie ponderată exponențial a tiparelor trecute, cu ponderea $1 + \\Theta$ pentru anul cel mai recent'),
      T('$\\Theta \\to -1$: a fixed (deterministic) pattern; $\\Theta = 0$: last year\'s pattern repeated (seasonal naive)', '$\\Theta \\to -1$: un tipar fix (determinist); $\\Theta = 0$: tiparul de anul trecut repetat (sezonier naiv)')]),
    (T('Intervals widen with $h$ (two unit roots); for $\\ln y$, $\\exp(\\hat y_{T+h})$ is the forecast \\textbf{median}', 'Intervalele se lărgesc cu $h$ (două rădăcini unitare); pentru $\\ln y$, $\\exp(\\hat y_{T+h})$ este \\textbf{mediana} prognozei'),
     [T('the mean is $\\exp(\\hat y_{T+h} + \\sigma_h^2/2)$, slightly higher', 'media este $\\exp(\\hat y_{T+h} + \\sigma_h^2/2)$, puțin mai mare')])))

D.frame(T('Worked example: forecasting a seasonal MA', 'Exemplu rezolvat: prognoza unui MA sezonier'), items(
    (T('Quarterly SARIMA$(0,0,0)(0,1,1)_4$: $y_t = y_{t-4} + \\varepsilon_t + \\Theta\\varepsilon_{t-4}$, $\\Theta = -0.6$', 'SARIMA$(0,0,0)(0,1,1)_4$ trimestrial: $y_t = y_{t-4} + \\varepsilon_t + \\Theta\\varepsilon_{t-4}$, $\\Theta = -0{,}6$'),
     [T('last year: $y_{T-3}, \\dots, y_T = 100, 120, 130, 140$; residuals $\\hat\\varepsilon_{T-3}, \\dots, \\hat\\varepsilon_T = 2, -1, 0, 3$', 'ultimul an: $y_{T-3}, \\dots, y_T = 100; 120; 130; 140$; reziduurile $\\hat\\varepsilon_{T-3}, \\dots, \\hat\\varepsilon_T = 2; -1; 0; 3$')]),
    (T('$\\hat y_{T+1} = y_{T-3} + \\Theta\\hat\\varepsilon_{T-3} = 100 - 0.6 \\cdot 2 = @{we3.f1}$', '$\\hat y_{T+1} = y_{T-3} + \\Theta\\hat\\varepsilon_{T-3} = 100 - 0{,}6 \\cdot 2 = @{we3.f1}$'),
     [T('$\\hat y_{T+2} = @{we3.f2}$, $\\hat y_{T+3} = @{we3.f3}$, $\\hat y_{T+4} = @{we3.f4}$', '$\\hat y_{T+2} = @{we3.f2}$, $\\hat y_{T+3} = @{we3.f3}$, $\\hat y_{T+4} = @{we3.f4}$')]),
    (T('$\\hat y_{T+5} = \\hat y_{T+1} = @{we3.f5}$: after one year the MA term is gone and the forecast pattern repeats', '$\\hat y_{T+5} = \\hat y_{T+1} = @{we3.f5}$: după un an, termenul MA dispare și tiparul prognozei se repetă'),
     [T('a large positive surprise last season ($\\hat\\varepsilon_T = 3$) lowers the forecast for that season by $0.6 \\cdot 3$: partial correction', 'o surpriză pozitivă mare în ultimul sezon ($\\hat\\varepsilon_T = 3$) scade prognoza pentru acel sezon cu $0{,}6 \\cdot 3$: o corecție parțială')])))

D.recap(('SARIMA models', 'modele SARIMA'), [
    T('SARIMA multiplies regular and seasonal polynomials: $\\phi(L)\\Phi(L^s)\\Delta^d\\Delta_s^Dy_t = \\theta(L)\\Theta(L^s)\\varepsilon_t$', 'SARIMA înmulțește polinoame obișnuite și sezoniere: $\\phi(L)\\Phi(L^s)\\Delta^d\\Delta_s^Dy_t = \\theta(L)\\Theta(L^s)\\varepsilon_t$'),
    T('Identification: the Chapter 2 rules at lags $s, 2s, \\dots$, plus satellites at $ks \\pm j$', 'Identificarea: regulile din Capitolul 2 la decalajele $s, 2s, \\dots$, plus sateliții de la $ks \\pm j$'),
    T('Compare AICc only for the same $d$ and $D$; check Ljung--Box beyond lag $s$', 'Comparați AICc doar pentru aceleași $d$ și $D$; verificați Ljung--Box dincolo de decalajul $s$'),
    T('The airline forecast: a seasonal pattern on a line, updated with weight $1 + \\Theta$', 'Prognoza airline: un tipar sezonier pe o dreaptă, actualizat cu ponderea $1 + \\Theta$')])

# =============================================================================
# 4. STUDIU DE CAZ: AIRLINE
# =============================================================================
D.section('Case study: the airline model', 'Studiu de caz: modelul airline')

D.frame(T('Box and Jenkins and the airline passengers', 'Box, Jenkins și pasagerii companiilor aeriene'), cols(items(
    (T('\\refBJ (first edition 1970) modelled the monthly totals of international airline passengers, 1949--1960 (@{air.T} months)', '\\refBJ (prima ediție în 1970) au modelat totalurile lunare ale pasagerilor internaționali, 1949--1960 (@{air.T} de luni)'),
     [T('growth plus a summer peak whose size grows with the level', 'creștere plus un vîrf de vară a cărui mărime crește cu nivelul'),
      T('their model, SARIMA$(0,1,1)(0,1,1)_{12}$ for $\\ln y_t$, has carried the name ``airline model\'\' ever since', 'modelul lor, SARIMA$(0,1,1)(0,1,1)_{12}$ pentru $\\ln y_t$, poartă de atunci numele de „model airline”')]),
    (T('Why it lasted', 'Motivele longevității modelului'),
     [T('two parameters fit many seasonal series: trade, production, retail sales, energy', 'doi parametri se potrivesc multor serii sezoniere: comerț, producție, vînzări cu amănuntul, energie'),
      T('it is the reference model of the regARIMA step of seasonal-adjustment software (Section 6)', 'este modelul de referință al etapei regARIMA din programele de ajustare sezonieră (secțiunea 6)')])),
    ph('pan_am', T('The first Boeing 707s for Pan American, Seattle, 1958', 'Primele avioane Boeing 707 pentru Pan American, Seattle, 1958'), h='0.36\\textheight'),
    wl='0.56', wr='0.40'), 'footnotesize')

chart(T('Identification of the airline model', 'Identificarea modelului airline'), 'tsa_ch4_airline', 'TSA_ch4_airline', [
    T('Left: passengers and their log; middle and right: ACF and PACF of $w_t = \\Delta\\Delta_{12}\\ln y_t$, @{air.Tw} observations', 'Stînga: pasagerii și logaritmul lor; mijloc și dreapta: ACF și PACF pentru $w_t = \\Delta\\Delta_{12}\\ln y_t$, @{air.Tw} de observații')],
    h='0.58\\textheight')

interp(('the airline correlograms', 'corelogramelor airline'), [
    T('The log makes the summer swing constant: the multiplicative pattern becomes additive', 'Logaritmul face constantă amplitudinea verii: tiparul multiplicativ devine aditiv'),
    (T('ACF of $w_t$: $@{air.r1}$ at lag 1 and $@{air.r12}$ at lag 12, outside the band $\\pm @{air.band}$', 'ACF pentru $w_t$: $@{air.r1}$ la decalajul 1 și $@{air.r12}$ la decalajul 12, în afara benzii $\\pm @{air.band}$'),
     [T('a regular MA(1) and a seasonal MA(1); lag 3 ($@{air.r3}$) is borderline', 'un MA(1) obișnuit și un MA(1) sezonier; decalajul 3 ($@{air.r3}$) este la limită')]),
    T('The PACF decays at lags 1, 2, 3 and 12, 24: consistent with MA terms, not AR terms', 'PACF descrește la decalajele 1, 2, 3 și 12, 24: în acord cu termeni MA, nu cu termeni AR'),
    T('Conclusion: SARIMA$(0,1,1)(0,1,1)_{12}$, exactly the model of Box and Jenkins', 'Concluzia: SARIMA$(0,1,1)(0,1,1)_{12}$, exact modelul lui Box și Jenkins')])

chart(T('Airline model: forecasts out of sample', 'Modelul airline: prognoze în afara eșantionului'), 'tsa_ch4_airline_fc', 'TSA_ch4_airline', [
    T('Estimated on 1949--1958, forecasts for the 24 months of 1959--1960, back-transformed from $\\ln y$', 'Estimat pe 1949--1958, prognoze pentru cele 24 de luni din 1959--1960, transformate înapoi din $\\ln y$')],
    h='0.56\\textheight')

interp(('the airline forecasts', 'prognozelor airline'), [
    (T('1949--1958: $\\hat\\theta = @{af.th}$ (@{af.ths}), $\\hat\\Theta = @{af.Th}$ (@{af.Ths}); full sample: $\\hat\\theta = @{af.fth}$ (@{af.fths}), $\\hat\\Theta = @{af.fTh}$ (@{af.fThs})', '1949--1958: $\\hat\\theta = @{af.th}$ (@{af.ths}), $\\hat\\Theta = @{af.Th}$ (@{af.Ths}); întregul eșantion: $\\hat\\theta = @{af.fth}$ (@{af.fths}), $\\hat\\Theta = @{af.fTh}$ (@{af.fThs})'),
     [T('residual sd @{af.sig}\\% per month; Ljung--Box $Q^*(24)$ on 22 degrees of freedom, p = @{af.lbp}', 'abaterea standard a reziduurilor @{af.sig}\\% pe lună; Ljung--Box $Q^*(24)$ cu 22 de grade de libertate, p = @{af.lbp}')]),
    T('MAPE over 1959--1960: @{af.mape}\\% for the airline model, @{af.mapesn}\\% for the seasonal naive forecast (which ignores growth)', 'MAPE pe 1959--1960: @{af.mape}\\% pentru modelul airline, @{af.mapesn}\\% pentru prognoza sezonieră naivă (care ignoră creșterea)'),
    (T('December 1960: forecast @{af.f}, interval [@{af.lo}, @{af.hi}], actual @{af.act}: inside, but the model under-predicts the 1959--1960 growth', 'Decembrie 1960: prognoza @{af.f}, intervalul [@{af.lo}; @{af.hi}], valoarea reală @{af.act}: în interior, dar modelul subestimează creșterea din 1959--1960'),
     [T('the interval width grows from @{af.w1} (first month) to @{af.w24} (24th month): two unit roots', 'lățimea intervalului crește de la @{af.w1} (prima lună) la @{af.w24} (luna 24): două rădăcini unitare')])], size='footnotesize')

# =============================================================================
# 5. STUDIU DE CAZ: PIB
# =============================================================================
D.section('Case study: Romanian GDP, not adjusted', 'Studiu de caz: PIB-ul neajustat al României')

D.frame(T('SARIMA models for Romanian GDP', 'Modele SARIMA pentru PIB-ul României'), table(
    'lrrrr', T('\\textbf{Model}', '\\textbf{Modelul}') + ' & \\textbf{AICc} & \\textbf{BIC} & ' + T('\\textbf{LB(8), p}', '\\textbf{LB(8), p}') + ' & $\\hat\\sigma$ (\\%)',
    [f'$({m[1]},1,{m[5]})({m[8]},1,{m[12]})_4$' + f' & @{{gm{i}.aicc}} & @{{gm{i}.bic}} & @{{gm{i}.p}} & @{{gm{i}.sig}}' for i, m in enumerate(ROWS)],
    size='scriptsize').replace('@{gm1.bic}', '\\textbf{@{gm1.bic}}').replace('@{gm3.aicc}', '\\textbf{@{gm3.aicc}}') + items(
    T('$y_t = 100\\ln Y_t$, real GDP not adjusted, @{gm.q0}--@{gm.q1}, $T = @{gm.T}$; maximum likelihood; all with $d = D = 1$, so the criteria are comparable', '$y_t = 100\\ln Y_t$, PIB real neajustat, @{gm.q0}--@{gm.q1}, $T = @{gm.T}$; verosimilitate maximă; toate cu $d = D = 1$, deci criteriile sînt comparabile'),
    T('Ljung--Box on 8 lags with $8 - p - q - P - Q$ degrees of freedom', 'Ljung--Box pe 8 decalaje, cu $8 - p - q - P - Q$ grade de libertate')), 'footnotesize')

interp(('the model table', 'tabelului de modele'), [
    (T('BIC chooses $(0,1,0)(0,1,1)_4$: only the seasonal MA term; AICc chooses $(1,1,1)(0,1,1)_4$', 'BIC alege $(0,1,0)(0,1,1)_4$: doar termenul MA sezonier; AICc alege $(1,1,1)(0,1,1)_4$'),
     [T('the AICc model has $\\hat\\phi = @{gm.c.phi}$ and $\\hat\\theta = @{gm.c.th}$: almost a common factor (Chapter 2), little gain', 'modelul AICc are $\\hat\\phi = @{gm.c.phi}$ și $\\hat\\theta = @{gm.c.th}$: aproape un factor comun (Capitolul 2), cu un cîștig mic'),
      T('the airline model: $\\hat\\theta = @{gm.air.th}$ (@{gm.air.ths}), not significant; $\\hat\\Theta = @{gm.air.Th}$', 'modelul airline: $\\hat\\theta = @{gm.air.th}$ (@{gm.air.ths}), nesemnificativ; $\\hat\\Theta = @{gm.air.Th}$')]),
    T('A seasonal AR instead of the seasonal MA, or no seasonal term at all: much worse criteria and correlated residuals', 'Un AR sezonier în locul MA sezonier sau niciun termen sezonier: criterii mult mai slabe și reziduuri corelate'),
    T('We keep the BIC model: one parameter, $\\hat\\Theta = @{gf.Th}$ (@{gf.Ths}), $\\hat\\sigma = @{gf.sig}\\%$', 'Păstrăm modelul BIC: un singur parametru, $\\hat\\Theta = @{gf.Th}$ (@{gf.Ths}), $\\hat\\sigma = @{gf.sig}\\%$')])

chart(T('Diagnostics of the GDP model', 'Diagnosticarea modelului pentru PIB'), 'tsa_ch4_gdp_diag', 'TSA_ch4_gdp_case', [
    T('SARIMA$(0,1,0)(0,1,1)_4$ for $100\\ln Y_t$: residuals, their ACF and their distribution', 'SARIMA$(0,1,0)(0,1,1)_4$ pentru $100\\ln Y_t$: reziduurile, ACF a lor și distribuția lor')],
    h='0.56\\textheight')

interp(('the GDP diagnostics', 'diagnosticării PIB'), [
    (T('No autocorrelation left: $Q^*(8) = @{gdg.q8}$ on @{gdg.df} degrees of freedom, p = @{gdg.p8}; $Q^*(12)$: p = @{gdg.p12}', 'Nu rămîne autocorelație: $Q^*(8) = @{gdg.q8}$ cu @{gdg.df} grade de libertate, p = @{gdg.p8}; $Q^*(12)$: p = @{gdg.p12}'),
     [T('no spike at the seasonal lags 4, 8, 12, 16: the seasonal structure is captured', 'nicio valoare semnificativă la decalajele sezoniere 4, 8, 12, 16: structura sezonieră este surprinsă')]),
    (T('Not Normal: Jarque--Bera @{gdg.jb} (p @{gdg.jbp}); the largest residual is @{gdg.outv}\\% in @{gdg.outd} ($@{gdg.outz}$ standard deviations): the lockdown', 'Nu este distribuția Normală: Jarque--Bera @{gdg.jb} (p @{gdg.jbp}); cel mai mare reziduu este @{gdg.outv}\\% în @{gdg.outd} ($@{gdg.outz}$ abateri standard): izolarea'),
     [T('without 2020, Jarque--Bera falls to @{gdg.jbx}: an additive-outlier dummy would be the next step', 'fără 2020, Jarque--Bera scade la @{gdg.jbx}: o variabilă dummy pentru valoarea aberantă ar fi pasul următor')])])

chart(T('Forecasts of Romanian GDP', 'Prognozele PIB-ului României'), 'tsa_ch4_gdp_fc', 'TSA_ch4_gdp_case', [
    T('SARIMA$(0,1,0)(0,1,1)_4$, 8 quarters from @{gf.q0} to @{gf.q1}, 80\\% and 95\\% intervals; right: implied year-on-year growth', 'SARIMA$(0,1,0)(0,1,1)_4$, 8 trimestre, din @{gf.q0} pînă în @{gf.q1}, intervale de 80\\% și 95\\%; dreapta: creșterea anuală implicată')],
    h='0.56\\textheight')

interp(('the GDP forecasts', 'prognozelor PIB'), [
    (T('Last value: @{gf.last} bn EUR in @{gf.lastd}; forecasts @{gf.f1}, @{gf.f2}, @{gf.f3}, @{gf.f4} bn EUR for the next four quarters', 'Ultima valoare: @{gf.last} mld. EUR în @{gf.lastd}; prognozele @{gf.f1}; @{gf.f2}; @{gf.f3}; @{gf.f4} mld. EUR pentru următoarele patru trimestre'),
     [T('95\\% interval for the first quarter [@{gf.lo1}, @{gf.hi1}], for the eighth [@{gf.lo8}, @{gf.hi8}]', 'intervalul de 95\\% pentru primul trimestru [@{gf.lo1}; @{gf.hi1}], pentru al optulea [@{gf.lo8}; @{gf.hi8}]')]),
    (T('Implied year-on-year growth: @{gf.g1}\\%, @{gf.g2}\\%, @{gf.g3}\\%, @{gf.g4}\\%, then a constant @{gf.g8}\\%', 'Creșterea anuală implicată: @{gf.g1}\\%; @{gf.g2}\\%; @{gf.g3}\\%; @{gf.g4}\\%, apoi constant @{gf.g8}\\%'),
     [T('the model extrapolates the recent weak growth; it knows nothing about fiscal policy or EU funds', 'modelul extrapolează creșterea slabă recentă; nu știe nimic despre politica fiscală sau fondurile europene')]),
    T('The seasonal pattern of the forecast is a weighted average of past years, with weight $1 + \\hat\\Theta = @{gf.al}$ on the newest year', 'Tiparul sezonier al prognozei este o medie ponderată a anilor trecuți, cu ponderea $1 + \\hat\\Theta = @{gf.al}$ pentru anul cel mai recent')])

D.recap(('Airline passengers and Romanian GDP', 'pasagerii aerieni și PIB-ul României'), [
    T('Logs, then $\\Delta\\Delta_s$, then MA terms at lags 1 and $s$: the airline recipe works for both series', 'Logaritmi, apoi $\\Delta\\Delta_s$, apoi termeni MA la decalajele 1 și $s$: rețeta airline funcționează pentru ambele serii'),
    T('For GDP, BIC drops the regular MA term; AICc picks a near-redundant ARMA(1,1)', 'Pentru PIB, BIC renunță la termenul MA obișnuit; AICc alege un ARMA(1,1) aproape redundant'),
    T('Residuals are clean except for the 2020 outlier', 'Reziduurile sînt curate, cu excepția valorii aberante din 2020'),
    T('Forecasts repeat a smoothed seasonal pattern on a trend line, with widening intervals', 'Prognozele repetă un tipar sezonier netezit pe o linie de trend, cu intervale tot mai largi')])

# =============================================================================
# 6. AJUSTARE SEZONIERĂ
# =============================================================================
D.section('Seasonal adjustment in practice', 'Ajustarea sezonieră în practică')

D.frame(T('Why official statistics adjust for seasonality', 'Rostul ajustării sezoniere în statistica oficială'), items(
    (T('Users want to know whether the economy \\textbf{grew this quarter}, not whether Q3 is bigger than Q2', 'Utilizatorii vor să știe dacă economia \\textbf{a crescut în acest trimestru}, nu dacă T3 este mai mare decît T2'),
     [T('@{sa.q}: unadjusted GDP changed by @{sa.nsa}\\% (log points) from the previous quarter; the adjusted series by @{sa.sca}\\%', '@{sa.q}: PIB-ul neajustat s-a modificat cu @{sa.nsa}\\% (puncte logaritmice) față de trimestrul anterior; seria ajustată cu @{sa.sca}\\%'),
      T('the first number is pure season; only the second says something about the business cycle', 'prima cifră este doar sezon; doar a doua spune ceva despre ciclul economic')]),
    (T('\\textbf{Seasonal adjustment}: estimate the seasonal and calendar components and remove them: $y_t^{SA} = y_t - \\hat S_t - \\hat C_t$ (or $y_t/(\\hat S_t\\hat C_t)$)', '\\textbf{Ajustarea sezonieră}: estimăm componentele sezonieră și de calendar și le eliminăm: $y_t^{SA} = y_t - \\hat S_t - \\hat C_t$ (sau $y_t/(\\hat S_t\\hat C_t)$)'),
     [T('the classical decomposition and STL of Chapter 0 are simple versions of this', 'descompunerea clasică și STL din Capitolul 0 sînt variante simple ale acestei idei')]),
    T('For forecasting with SARIMA we use the \\textbf{unadjusted} series: adjustment filters change the autocorrelations and are revised when new data arrive',
      'Pentru prognoza cu SARIMA folosim seria \\textbf{neajustată}: filtrele de ajustare schimbă autocorelațiile și sînt revizuite cînd apar date noi')))

D.frame(T('X-11 and X-13ARIMA-SEATS', 'X-11 și X-13ARIMA-SEATS'), cols(items(
    (T('\\textbf{X-11} (US Census Bureau, 1960s; \\refLadiray): iterated moving averages', '\\textbf{X-11} (US Census Bureau, anii 1960; \\refLadiray): medii mobile aplicate iterativ'),
     [T('a centred $2 \\times 12$ moving average for the trend, then $3 \\times 3$ and $3 \\times 5$ moving averages of each month across years for the seasonal factors', 'o medie mobilă centrată $2 \\times 12$ pentru trend, apoi medii mobile $3 \\times 3$ și $3 \\times 5$ ale fiecărei luni de-a lungul anilor pentru factorii sezonieri')]),
    (T('\\textbf{X-12-ARIMA} (\\refFindley) and \\textbf{X-13ARIMA-SEATS} add a \\textbf{regARIMA} pre-adjustment:', '\\textbf{X-12-ARIMA} (\\refFindley) și \\textbf{X-13ARIMA-SEATS} adaugă o \\textbf{preajustare regARIMA}:'),
     [T('regression with SARIMA errors (often the airline model) for outliers, working days, Easter', 'regresie cu erori SARIMA (adesea modelul airline) pentru valori aberante, zile lucrătoare, Paște'),
      T('ARIMA forecasts extend the series, so the symmetric filters also work at the end: smaller revisions', 'prognozele ARIMA prelungesc seria, astfel încît filtrele simetrice funcționează și la capăt: revizuiri mai mici'),
      T('then X-11 filters or SEATS (model-based decomposition of the ARIMA model)', 'apoi filtrele X-11 sau SEATS (descompunerea pe baza modelului ARIMA)')]),
    T('EU: Eurostat and the national institutes follow \\refESS, with the JDemetra+ software (X-13 and TRAMO-SEATS)', 'UE: Eurostat și institutele naționale urmează \\refESS, cu programul JDemetra+ (X-13 și TRAMO-SEATS)')),
    ph('census', T('US Census Bureau, Suitland, Maryland: the home of X-11 and X-13', 'US Census Bureau, Suitland, Maryland: locul de naștere al X-11 și X-13'), h='0.28\\textheight'),
    wl='0.62', wr='0.34'), 'scriptsize')

D.frame(T('Eurostat series: NSA, CA, SA, SCA', 'Seriile Eurostat: NSA, CA, SA, SCA'), items(
    (T('Each series comes in several versions (dimension \\texttt{s\\_adj}):', 'Fiecare serie apare în mai multe variante (dimensiunea \\texttt{s\\_adj}):'),
     [T('\\textbf{NSA}: not seasonally adjusted (the raw data)', '\\textbf{NSA}: neajustată sezonier (datele brute)'),
      T('\\textbf{CA}: calendar adjusted only (working days, holidays)', '\\textbf{CA}: ajustată doar pentru efectele de calendar (zile lucrătoare, sărbători)'),
      T('\\textbf{SA}: seasonally adjusted; \\textbf{SCA}: seasonally and calendar adjusted', '\\textbf{SA}: ajustată sezonier; \\textbf{SCA}: ajustată sezonier și pentru efectele de calendar')]),
    (T('Which one for which question', 'Varianta potrivită pentru fiecare întrebare'),
     [T('quarter-on-quarter or month-on-month growth: SCA', 'creșterea față de trimestrul sau luna anterioară: SCA'),
      T('year-on-year growth: NSA or CA (the seasonal pattern cancels, the calendar does not)', 'creșterea față de anul anterior: NSA sau CA (tiparul sezonier se anulează, calendarul nu)'),
      T('SARIMA modelling and forecasting: NSA, with calendar regressors', 'modelarea și prognoza SARIMA: NSA, cu regresori de calendar')]),
    T('The annual totals of NSA and SCA are close (2024: @{sa.sum.n} and @{sa.sum.s} bn EUR): adjustment moves output between quarters, it does not create it',
      'Totalurile anuale NSA și SCA sînt apropiate (2024: @{sa.sum.n} și @{sa.sum.s} mld. EUR): ajustarea mută producția între trimestre, nu o creează')))

chart(T('Unadjusted and adjusted series', 'Serii neajustate și ajustate'), 'tsa_ch4_sa_nsa', 'TSA_ch4_seasonal_data', [
    T('Eurostat NSA and SCA: GDP (top left) and retail trade (top right); bottom: the implied seasonal and calendar factors NSA/SCA', 'Eurostat NSA și SCA: PIB (stînga sus) și comerțul cu amănuntul (dreapta sus); jos: factorii sezonieri și de calendar implicați, NSA/SCA')],
    h='0.66\\textheight')

interp(('the adjusted series', 'seriilor ajustate'), [
    (T('GDP factors 2015--2019: Q1 @{sa.f1}, Q2 @{sa.f2}, Q3 @{sa.f3}, Q4 @{sa.f4}: Q1 output is about a fifth below an average quarter', 'Factorii PIB 2015--2019: T1 @{sa.f1}, T2 @{sa.f2}, T3 @{sa.f3}, T4 @{sa.f4}: producția din T1 este cu aproximativ o cincime sub un trimestru mediu'),
     [T('the factors move slowly over the years: the stochastic seasonality that $\\hat\\Theta$ measures', 'factorii se mișcă lent de-a lungul anilor: sezonalitatea stochastică măsurată de $\\hat\\Theta$')]),
    T('Retail trade: December factor @{sa.dec}, January @{sa.jan} (2016--2019): Christmas shopping, then the January low', 'Comerțul cu amănuntul: factorul pentru decembrie @{sa.dec}, pentru ianuarie @{sa.jan} (2016--2019): cumpărăturile de Crăciun, apoi minimul din ianuarie'),
    T('The retail factor also jumps from month to month because of working days and Easter: the calendar part of SCA (Section 7)', 'Factorul pentru comerț sare și de la o lună la alta din cauza zilelor lucrătoare și a Paștelui: partea de calendar a SCA (secțiunea 7)'),
    T('The adjusted GDP shows the 2020 fall and the 2025--2026 stagnation that the raw series hides', 'PIB-ul ajustat arată scăderea din 2020 și stagnarea din 2025--2026 pe care seria brută le ascunde')], size='footnotesize')

D.recap(('Seasonal adjustment', 'ajustarea sezonieră'), [
    T('Seasonal adjustment removes $\\hat S_t$ and $\\hat C_t$ to show the trend and the cycle', 'Ajustarea sezonieră elimină $\\hat S_t$ și $\\hat C_t$ pentru a arăta trendul și ciclul'),
    T('X-13ARIMA-SEATS: regARIMA pre-adjustment (outliers, calendar, forecasts), then X-11 or SEATS; in the EU, JDemetra+', 'X-13ARIMA-SEATS: preajustare regARIMA (valori aberante, calendar, prognoze), apoi X-11 sau SEATS; în UE, JDemetra+'),
    T('Quarter-on-quarter growth: SCA; year-on-year growth: NSA or CA; SARIMA forecasting: NSA', 'Creșterea față de trimestrul anterior: SCA; creșterea anuală: NSA sau CA; prognoza SARIMA: NSA')])

# =============================================================================
# 7. FOURIER ȘI CALENDAR
# =============================================================================
D.section('Fourier terms and calendar effects', 'Termeni Fourier și efecte de calendar')

D.frame(T('Seasonal dummies and Fourier terms', 'Variabile dummy sezoniere și termeni Fourier'), items(
    (T('\\textbf{Dummies}: $y_t = \\beta_0 + \\sum_{j=2}^{s}\\gamma_jD_{jt} + u_t$: $s - 1$ coefficients, any shape', '\\textbf{Variabile dummy}: $y_t = \\beta_0 + \\sum_{j=2}^{s}\\gamma_jD_{jt} + u_t$: $s - 1$ coeficienți, orice formă'),
     [T('too many for $s = 52$ weeks, $s = 168$ hours or $s = 365$ days; impossible for $s = 365.25$', 'prea mulți pentru $s = 52$ de săptămîni, $s = 168$ ore sau $s = 365$ de zile; imposibil pentru $s = 365{,}25$')]),
    (T('\\textbf{Fourier terms} for a period $m$: $\\sum_{k=1}^{K}\\left[\\alpha_k\\sin\\frac{2\\pi kt}{m} + \\beta_k\\cos\\frac{2\\pi kt}{m}\\right]$', '\\textbf{Termeni Fourier} pentru o perioadă $m$: $\\sum_{k=1}^{K}\\left[\\alpha_k\\sin\\frac{2\\pi kt}{m} + \\beta_k\\cos\\frac{2\\pi kt}{m}\\right]$'),
     [T('$2K$ coefficients; $K = m/2$ reproduces the dummies exactly (for even $m$); a small $K$ gives a smooth pattern', '$2K$ coeficienți; $K = m/2$ reproduce exact variabilele dummy (pentru $m$ par); un $K$ mic dă un tipar neted'),
      T('$m$ may be non-integer, and several periods can be combined: e.g. $K = 5$ for $m = 24$ and $K = 4$ for $m = 168$ give @{fourier.k} regressors instead of $23 + 167$', '$m$ poate fi neîntreg și se pot combina mai multe perioade: de exemplu $K = 5$ pentru $m = 24$ și $K = 4$ pentru $m = 168$ dau @{fourier.k} regresori în locul a $23 + 167$')]),
    T('$K$ is chosen by AICc or by cross-validation; the same terms are inside TBATS and Prophet (Section 9)', '$K$ se alege cu AICc sau prin validare încrucișată; aceiași termeni se află în TBATS și Prophet (secțiunea 9)')))

chart(T('Fourier approximation of a seasonal pattern', 'Aproximarea Fourier a unui tipar sezonier'), 'tsa_ch4_fourier', 'TSA_ch4_calendar_fourier', [
    T('Monthly seasonal effect of $\\ln$(tourism nights), 2012--2019, after removing a 13-month centred mean; fits with $K = 1, 2, 3, 6$', 'Efectul sezonier lunar al $\\ln$(înnoptări turistice), 2012--2019, după eliminarea unei medii centrate pe 13 luni; ajustări cu $K = 1, 2, 3, 6$')],
    h='0.58\\textheight')

interp(('the Fourier fits', 'ajustărilor Fourier'), [
    (T('$K = 1$ (two coefficients) already gives $R^2 = @{fo.r1}$: the summer peak is close to a single wave', '$K = 1$ (doi coeficienți) dă deja $R^2 = @{fo.r1}$: vîrful de vară este aproape o singură undă'),
     [T('$K = 2$: @{fo.r2}; $K = 3$: @{fo.r3} with six coefficients instead of eleven', '$K = 2$: @{fo.r2}; $K = 3$: @{fo.r3} cu șase coeficienți în loc de unsprezece')]),
    T('$K = 6$ (11 coefficients) interpolates the 12 monthly points: the same as 11 dummies, $R^2 = @{fo.r6}$', '$K = 6$ (11 coeficienți) trece prin cele 12 puncte lunare: la fel ca 11 variabile dummy, $R^2 = @{fo.r6}$'),
    T('Higher harmonics add sharper detail: the August peak needs $K \\ge 3$', 'Armonicele superioare adaugă detalii mai fine: vîrful din august cere $K \\ge 3$')])

D.frame(T('Calendar effects and Orthodox Easter', 'Efecte de calendar și Paștele ortodox'), cols(items(
    (T('\\textbf{Working days}: a month with more Monday--Friday non-holiday days produces and sells more', '\\textbf{Zilele lucrătoare}: o lună cu mai multe zile de luni pînă vineri care nu sînt sărbători produce și vinde mai mult'),
     [T('known in advance: a regressor that can be used in forecasts', 'cunoscute dinainte: un regresor care poate fi folosit în prognoze')]),
    (T('\\textbf{Moving holidays}: Orthodox Easter falls between 4 April and 8 May', '\\textbf{Sărbătorile mobile}: Paștele ortodox cade între 4 aprilie și 8 mai'),
     [T('@{ea.d24}, @{ea.d25}, @{ea.d26}, @{ea.d27}: shopping shifts between March, April and May', '@{ea.d24}, @{ea.d25}, @{ea.d26}, @{ea.d27}: cumpărăturile se mută între martie, aprilie și mai'),
      T('in Python: \\texttt{dateutil.easter.easter(y, EASTER\\_ORTHODOX)}; Pentecost is 49 days later', 'în Python: \\texttt{dateutil.easter.easter(y, EASTER\\_ORTHODOX)}; Rusaliile sînt cu 49 de zile mai tîrziu')]),
    (T('\\textbf{Easter regressor} (X-13 \\texttt{easter[w]}): $E_t$ = share of the $w$ days before Easter Sunday that fall in month $t$', '\\textbf{Regresorul Paște} (\\texttt{easter[w]} din X-13): $E_t$ = partea din cele $w$ zile dinaintea Duminicii Paștelui care cade în luna $t$'),
     [T('$w = 10$: Easter on 20 April gives $E_{\\text{April}} = 1$; on 5 May, $E_{\\text{April}} = 0.6$ and $E_{\\text{May}} = 0.4$', '$w = 10$: Paștele pe 20 aprilie dă $E_{\\text{aprilie}} = 1$; pe 5 mai, $E_{\\text{aprilie}} = 0{,}6$ și $E_{\\text{mai}} = 0{,}4$')])),
    ph('eggs', T('Painted Easter eggs from Romania', 'Ouă încondeiate de Paște din România'), h='0.32\\textheight'),
    wl='0.62', wr='0.34'), 'scriptsize')

D.frame(T('Regression with SARIMA errors', 'Regresia cu erori SARIMA'), items(
    (T('$100\\ln y_t = \\beta_EE_t + \\beta_W\\,\\mathrm{WD}_t + \\text{outlier dummies} + u_t$, with $u_t \\sim$ SARIMA$(0,1,1)(0,1,1)_{12}$', '$100\\ln y_t = \\beta_EE_t + \\beta_W\\,\\mathrm{WD}_t + \\text{variabile dummy pentru valori aberante} + u_t$, cu $u_t \\sim$ SARIMA$(0,1,1)(0,1,1)_{12}$'),
     [T('the ``regARIMA\'\' model of X-13; estimated in one step by maximum likelihood (\\texttt{SARIMAX} with \\texttt{exog})', 'modelul „regARIMA” din X-13; estimat într-un singur pas prin verosimilitate maximă (\\texttt{SARIMAX} cu \\texttt{exog})'),
      T('ordinary least squares with SARIMA errors ignored would give wrong standard errors (Chapter 3, spurious regression)', 'metoda celor mai mici pătrate care ignoră erorile SARIMA ar da erori standard greșite (Capitolul 3, regresia falsă)')]),
    (T('Data: Eurostat retail trade volume, Romania, not adjusted, @{ea.f0}--@{ea.f1}: food (G47\\_FOOD) and all retail (G47)', 'Datele: volumul comerțului cu amănuntul, Eurostat, România, neajustat, @{ea.f0}--@{ea.f1}: alimente (G47\\_FOOD) și total (G47)'),
     [T('dummies for April and May 2020 (lockdown)', 'variabile dummy pentru aprilie și mai 2020 (izolarea)')])))

chart(T('Easter and Romanian retail trade', 'Paștele și comerțul cu amănuntul din România'), 'tsa_ch4_easter', 'TSA_ch4_calendar_fourier', [
    T('Left: Orthodox Easter Sunday as a day of April (above 30: May); right: year-on-year growth of food retail in March--May against the change in the Easter share (2020--2021 left out)', 'Stînga: Duminica Paștelui ortodox ca zi din aprilie (peste 30: mai); dreapta: creșterea anuală a vînzărilor de alimente în martie--mai față de modificarea ponderii Paștelui (fără 2020--2021)')],
    h='0.56\\textheight')

interp(('the Easter effect', 'efectului Paștelui'), [
    (T('Food retail: $\\hat\\beta_E = @{ea.f.e}\\%$ (se @{ea.f.es}, $t = @{ea.f.t}$): a month that holds the 10 days before Easter sells about @{ea.f.e}\\% more food', 'Alimente: $\\hat\\beta_E = @{ea.f.e}\\%$ (eroarea standard @{ea.f.es}, $t = @{ea.f.t}$): o lună care cuprinde cele 10 zile dinaintea Paștelui vinde cu aproximativ @{ea.f.e}\\% mai multe alimente'),
     [T('one working day more: $+@{ea.f.wd}\\%$ (se @{ea.f.wds}); April 2020: $@{ea.f.ao}\\%$', 'o zi lucrătoare în plus: $+@{ea.f.wd}\\%$ (eroarea standard @{ea.f.wds}); aprilie 2020: $@{ea.f.ao}\\%$')]),
    (T('All retail: Easter $@{ea.t.e}\\%$ (se @{ea.t.es}), not significant; working days $+@{ea.t.wd}\\%$ (se @{ea.t.wds})', 'Totalul comerțului: Paștele $@{ea.t.e}\\%$ (eroarea standard @{ea.t.es}), nesemnificativ; zilele lucrătoare $+@{ea.t.wd}\\%$ (eroarea standard @{ea.t.wds})'),
     [T('the Easter effect is in food, the working-day effect in everything', 'efectul Paștelui se află în alimente, efectul zilelor lucrătoare în toate vînzările')]),
    T('The raw scatter is noisy (slope @{ea.slope} percentage points): VAT changes and prices also move year-on-year growth; the regression separates the effects', 'Norul de puncte brut este zgomotos (panta @{ea.slope} puncte procentuale): modificările TVA și prețurile mișcă și ele creșterea anuală; regresia separă efectele'),
    T('Easter fell in May in @{ea.may}: a forecaster who ignores this mistakes April and May', 'Paștele a căzut în mai în @{ea.may.ro}: cine ignoră acest lucru greșește prognozele pentru aprilie și mai')], size='footnotesize')

D.recap(('Fourier terms and calendar effects', 'termeni Fourier și efecte de calendar'), [
    T('Fourier terms describe a seasonal pattern with $2K$ coefficients, for any period, even non-integer', 'Termenii Fourier descriu un tipar sezonier cu $2K$ coeficienți, pentru orice perioadă, chiar neîntreagă'),
    T('Working days and moving holidays are known in advance: model them as regressors', 'Zilele lucrătoare și sărbătorile mobile sînt cunoscute dinainte: le modelăm ca regresori'),
    T('Romanian food retail rises about @{ea.f.e}\\% in the Easter month; total retail follows working days', 'Vînzările de alimente din România cresc cu aproximativ @{ea.f.e}\\% în luna Paștelui; totalul comerțului urmează zilele lucrătoare'),
    T('Regression with SARIMA errors: estimate regressors and dynamics together', 'Regresia cu erori SARIMA: estimăm împreună regresorii și dinamica')])

# =============================================================================
# 8. SEZONALITATE MULTIPLĂ
# =============================================================================
D.section('Multiple seasonality: electricity load', 'Sezonalitate multiplă: consumul de electricitate')

D.frame(T('Electricity load in Romania', 'Consumul de electricitate din România'), cols(items(
    (T('\\textbf{Load}: the electricity consumed in the system, in MW (hourly averages); it must be matched by generation at every moment', '\\textbf{Consumul} (sarcina): energia electrică consumată în sistem, în MW (medii orare); trebuie egalat de producție în fiecare moment'),
     [T('forecasts are needed for every hour of the next day: day-ahead markets, reserves, grid planning', 'prognozele sînt necesare pentru fiecare oră a zilei următoare: piața pentru ziua următoare, rezerve, planificarea rețelei')]),
    (T('Data: ENTSO-E (the European network of transmission system operators), ``monthly hourly load values\'\', Romania', 'Datele: ENTSO-E (rețeaua europeană a operatorilor de transport și de sistem), „monthly hourly load values”, România'),
     [T('@{ho.T} hours, 2022--2026, Romanian winter time (UTC+2) to avoid the clock changes', '@{ho.T} de ore, 2022--2026, ora de iarnă a României (UTC+2), pentru a evita schimbările de oră')]),
    T('Load forecasting has its own competitions: GEFCom2014 (\\refGEF)', 'Prognoza consumului are propriile competiții: GEFCom2014 (\\refGEF)')),
    ph('pdf', T('Iron Gates II hydropower plant on the Danube', 'Hidrocentrala Porțile de Fier II, pe Dunăre'), h='0.30\\textheight'),
    wl='0.60', wr='0.36'), 'footnotesize')

chart(T('Hourly load: the day and the week', 'Consumul orar: ziua și săptămîna'), 'tsa_ch4_load_hourly', 'TSA_ch4_multiple_seasonality', [
    T('Left: three weeks of January 2025; right: sample ACF of all hours of 2025 up to lag 400', 'Stînga: trei săptămîni din ianuarie 2025; dreapta: ACF de selecție pentru toate orele din 2025, pînă la decalajul 400')],
    h='0.56\\textheight')

interp(('the hourly load', 'consumului orar'), [
    (T('Daily cycle: lowest around @{ho.trough}:00 (@{ho.tv} GW on average in 2025), highest around @{ho.peak}:00 (@{ho.pv} GW)', 'Ciclul zilnic: minimul în jurul orei @{ho.trough}:00 (în medie @{ho.tv} GW în 2025), maximul în jurul orei @{ho.peak}:00 (@{ho.pv} GW)'),
     [T('ACF $@{ho.r24}$ at lag 24, $@{ho.r12}$ at lag 12', 'ACF $@{ho.r24}$ la decalajul 24, $@{ho.r12}$ la decalajul 12')]),
    (T('Weekly cycle: weekends are lower; ACF $@{ho.r168}$ at lag 168 (one week), above the lag-24 value', 'Ciclul săptămînal: weekendurile sînt mai joase; ACF $@{ho.r168}$ la decalajul 168 (o săptămînă), peste valoarea de la decalajul 24'),
     [T('the same hour one week ago is the best single predictor: Monday is like last Monday, not like Sunday', 'aceeași oră de acum o săptămînă este cel mai bun predictor individual: lunea seamănă cu lunea trecută, nu cu duminica')]),
    T('And an annual cycle on top (heating and air conditioning): three periods, 24, 168 and about 8766 hours', 'Și un ciclu anual peste acestea (încălzire și aer condiționat): trei perioade, 24, 168 și aproximativ 8766 de ore'),
    (T('SARIMA is not enough: it has one integer period $s$, while here $s = 24$, $168$ and $8766$ act at once', 'SARIMA nu este suficient: are o singură perioadă întreagă $s$, în timp ce aici acționează simultan $s = 24$, $168$ și $8766$'),
     [T('a polynomial in $L^{168}$ is slow and unstable, 365.25 is not an integer, and holidays break the weekly pattern', 'un polinom în $L^{168}$ este lent și instabil, 365,25 nu este un număr întreg, iar sărbătorile rup tiparul săptămînal'),
      T('answers: MSTL, the dynamic harmonic regression (DHR), TBATS, Prophet', 'răspunsuri: MSTL, regresia armonică dinamică (DHR), TBATS, Prophet')])], size='footnotesize')

chart(T('MSTL: one seasonal component per period', 'MSTL: cîte o componentă sezonieră pentru fiecare perioadă'), 'tsa_ch4_mstl', 'TSA_ch4_multiple_seasonality', [
    T('MSTL (\\refMSTL) applies STL (Chapter 0) repeatedly, once for each period (24 and 168 hours), until the components settle', 'MSTL (\\refMSTL) aplică STL (Capitolul 0) în mod repetat, cîte o dată pentru fiecare perioadă (24 și 168 de ore), pînă cînd componentele se stabilizează')],
    h='0.64\\textheight')

interp(('the MSTL decomposition', 'descompunerii MSTL'), [
    (T('Of the variance around the trend: @{ms.s24}\\% daily component, @{ms.s168}\\% weekly component, @{ms.sr}\\% remainder', 'Din varianța în jurul trendului: @{ms.s24}\\% componenta zilnică, @{ms.s168}\\% componenta săptămînală, @{ms.sr}\\% componenta neregulată'),
     [T('range of the daily component in the first week: @{ms.a24} GW; of the weekly one: @{ms.a168} GW', 'amplitudinea componentei zilnice în prima săptămînă: @{ms.a24} GW; a celei săptămînale: @{ms.a168} GW')]),
    T('The weekly component is flat Monday to Friday and drops on Saturday and more on Sunday', 'Componenta săptămînală este plată de luni pînă vineri și scade sîmbăta și mai mult duminica'),
    T('The remainder is small but not white noise: cold spells and holidays; a forecasting model adds dynamics and regressors', 'Componenta neregulată este mică, dar nu este zgomot alb: valuri de frig și sărbători; un model de prognoză adaugă dinamică și regresori')])

chart(T('Daily load and public holidays', 'Consumul zilnic și sărbătorile legale'), 'tsa_ch4_load_daily', 'TSA_ch4_multiple_seasonality', [
    T('Daily mean load, 2022--2026, @{load.T} days; Romanian public holidays marked; right: mean by day of the week, holidays excluded', 'Consumul mediu zilnic, 2022--2026, @{load.T} de zile; sărbătorile legale marcate; dreapta: media pe zile ale săptămînii, fără sărbători')],
    h='0.56\\textheight')

interp(('the daily load', 'consumului zilnic'), [
    (T('Week: Saturday @{dy.sat}\\% and Sunday @{dy.sun}\\% below Wednesday', 'Săptămîna: sîmbăta cu @{dy.sat}\\%, iar duminica cu @{dy.sun}\\% sub miercuri'),
     [T('Year: January @{dy.jan}\\% above May; a smaller summer peak (air conditioning)', 'Anul: ianuarie cu @{dy.jan}\\% peste mai; un vîrf mai mic vara (aerul condiționat)')]),
    (T('Holidays: the lowest day of the sample is Easter Sunday, @{dy.lowd} (@{dy.low} GW, mean @{dy.mean} GW)', 'Sărbătorile: cea mai joasă zi din eșantion este Duminica Paștelui, @{dy.lowd} (@{dy.low} GW, media @{dy.mean} GW)'),
     [T('the four Easter days (Friday to Monday) average @{dy.eas} GW; Christmas to New Year is also low', 'cele patru zile de Paște (vineri--luni) au în medie @{dy.eas} GW; perioada Crăciun--Anul Nou este și ea joasă')]),
    T('The level falls over 2022--2026; possible causes: the price shock of 2022 and rooftop solar panels that cover part of the demand behind the meter', 'Nivelul scade în 2022--2026; cauze posibile: șocul de preț din 2022 și panourile solare de pe acoperișuri, care acoperă o parte din cerere înaintea contorului')])

D.frame(T('A dynamic harmonic regression for daily load', 'O regresie armonică dinamică pentru consumul zilnic'), items(
    (T('$y_t = \\sum_{k=1}^{4}\\left[\\alpha_k\\sin\\frac{2\\pi kt}{365.25} + \\beta_k\\cos\\frac{2\\pi kt}{365.25}\\right] + \\sum_{j}\\delta_jD^{\\mathrm{day}}_{jt} + \\lambda_1H_t + \\lambda_2E_t + \\lambda_3X_t + u_t$', '$y_t = \\sum_{k=1}^{4}\\left[\\alpha_k\\sin\\frac{2\\pi kt}{365.25} + \\beta_k\\cos\\frac{2\\pi kt}{365.25}\\right] + \\sum_{j}\\delta_jD^{\\mathrm{zi}}_{jt} + \\lambda_1H_t + \\lambda_2E_t + \\lambda_3X_t + u_t$'),
     [T('annual Fourier terms; weekday dummies (Sunday is the base); $H$ public holiday, $E$ Easter (Friday to Monday), $X$ 24 December -- 2 January', 'termeni Fourier anuali; variabile dummy pentru zilele săptămînii (duminica este baza); $H$ sărbătoare legală, $E$ Paște (vineri--luni), $X$ 24 decembrie -- 2 ianuarie'),
      T('$u_t \\sim$ ARIMA@{dh.order}, chosen by AIC: \\textbf{dynamic harmonic regression} (DHR, \\refFPPdhr)', '$u_t \\sim$ ARIMA@{dh.order}, ales cu AIC: \\textbf{regresia armonică dinamică} (DHR, \\refFPPdhr)')]),
    (T('Estimates, data to the first forecast origin (30 June 2025), in GW:', 'Estimările, cu datele pînă la prima origine de prognoză (30 iunie 2025), în GW:'),
     [T('Wednesday $+@{dh.wed}$, Monday $+@{dh.mon}$, Saturday $+@{dh.sat}$ relative to Sunday', 'miercuri $+@{dh.wed}$, luni $+@{dh.mon}$, sîmbătă $+@{dh.sat}$ față de duminică'),
      T('public holiday $-@{dh.hol}$ (se @{dh.hols}), Easter $-@{dh.eas}$ more, Christmas--New Year $-@{dh.xm}$', 'sărbătoare legală $-@{dh.hol}$ (eroarea standard @{dh.hols}), Paștele încă $-@{dh.eas}$, Crăciun--Anul Nou $-@{dh.xm}$'),
      T('errors: AR $@{dh.ar}$, MA $@{dh.ma}$, $\\hat\\sigma = @{dh.sig}$ MW; Ljung--Box(14) p @{dh.lbp}: some dynamics left', 'erori: AR $@{dh.ar}$, MA $@{dh.ma}$, $\\hat\\sigma = @{dh.sig}$ MW; Ljung--Box(14) p @{dh.lbp}: rămîne puțină dinamică')])), 'footnotesize')

D.recap(('Multiple seasonality', 'sezonalitatea multiplă'), [
    T('Hourly load has daily, weekly and annual cycles; daily load has weekly and annual cycles', 'Consumul orar are cicluri zilnice, săptămînale și anuale; consumul zilnic are cicluri săptămînale și anuale'),
    T('SARIMA handles one integer period; MSTL decomposes several', 'SARIMA tratează o singură perioadă întreagă; MSTL descompune mai multe'),
    T('Holidays, Easter most of all, are the largest deviations: they need regressors', 'Sărbătorile, mai ales Paștele, sînt cele mai mari abateri: au nevoie de regresori'),
    T('DHR = Fourier terms + calendar dummies + ARIMA errors: flexible and transparent', 'DHR = termeni Fourier + variabile dummy de calendar + erori ARIMA: flexibilă și transparentă')])

# =============================================================================
# 9. TBATS ȘI PROPHET
# =============================================================================
D.section('TBATS and Prophet', 'TBATS și Prophet')

D.frame(T('TBATS', 'TBATS'), items(
    (T('\\refTBATS: exponential smoothing (ETS, Chapter 0) for complex seasonality; the name lists its parts', '\\refTBATS: netezire exponențială (ETS, Capitolul 0) pentru sezonalitate complexă; numele enumeră părțile modelului'),
     [T('trigonometric seasonality (T), Box--Cox transformation (B), ARMA errors (A), trend (T), seasonal components (S)', 'sezonalitate trigonometrică (T), transformare Box--Cox (B), erori ARMA (A), trend (T), componente sezoniere (S)')]),
    (T('$y_t^{(\\omega)} = \\ell_{t-1} + \\phi b_{t-1} + \\sum_{i}s^{(i)}_{t-1} + d_t$, $d_t$ ARMA($p,q$); level and trend updated as in Holt\'s method', '$y_t^{(\\omega)} = \\ell_{t-1} + \\phi b_{t-1} + \\sum_{i}s^{(i)}_{t-1} + d_t$, $d_t$ ARMA($p,q$); nivelul și trendul se actualizează ca în metoda Holt'),
     [T('each period $m_i$ has $K_i$ harmonics $s^{(i)}_{j,t}$, each a rotating pair (cosine, sine) updated by the errors: a Fourier pattern that can evolve', 'fiecare perioadă $m_i$ are $K_i$ armonici $s^{(i)}_{j,t}$, fiecare o pereche rotitoare (cosinus, sinus) actualizată de erori: un tipar Fourier care poate evolua')]),
    (T('Automatic choices by AIC: Box--Cox or not, trend or damped trend or none, the $K_i$, the ARMA orders', 'Alegeri automate cu AIC: Box--Cox sau nu, trend, trend amortizat sau fără trend, valorile $K_i$, ordinele ARMA'),
     [T('daily load to June 2025, periods 7 and 365.25 (no Box--Cox, no trend): $K = @{tb.k7}$ weekly and $K = @{tb.k365}$ annual harmonics, ARMA(@{tb.p},@{tb.q}), @{tb.states} states', 'consumul zilnic pînă în iunie 2025, perioadele 7 și 365,25 (fără Box--Cox, fără trend): $K = @{tb.k7}$ armonici săptămînale și $K = @{tb.k365}$ anuale, ARMA(@{tb.p},@{tb.q}), @{tb.states} de stări')]),
    T('Limits: no regressors, hence no holidays or Easter; slow on long series; Python package \\texttt{tbats}', 'Limite: fără regresori, deci fără sărbători sau Paște; lent pe serii lungi; pachetul Python \\texttt{tbats}')), 'footnotesize')

D.frame(T('Prophet', 'Prophet'), items(
    (T('\\refProphet (Meta): a regression in time, $y(t) = g(t) + s(t) + h(t) + \\varepsilon_t$', '\\refProphet (Meta): o regresie în timp, $y(t) = g(t) + s(t) + h(t) + \\varepsilon_t$'),
     [T('$g(t)$: piecewise-linear (or logistic) trend whose slope may change at \\textbf{changepoints}; a Laplace prior shrinks most changes to zero', '$g(t)$: trend liniar pe porțiuni (sau logistic) a cărui pantă se poate schimba în \\textbf{puncte de schimbare}; o distribuție a priori Laplace aduce cele mai multe schimbări la zero'),
      T('$s(t)$: Fourier terms, by default $K = 10$ for the year and $K = 3$ for the week', '$s(t)$: termeni Fourier, implicit $K = 10$ pentru an și $K = 3$ pentru săptămînă'),
      T('$h(t)$: one effect per holiday (with optional windows around it), from a list of dates', '$h(t)$: un efect pentru fiecare sărbătoare (cu ferestre opționale în jurul ei), dintr-o listă de date')]),
    (T('Estimated by Stan (maximum a posteriori); intervals simulate future changepoints and the noise', 'Estimat cu Stan (maximul a posteriori); intervalele simulează punctele de schimbare viitoare și zgomotul'),
     [T('no autoregressive part: today\'s surprise does not move tomorrow\'s forecast', 'fără parte autoregresivă: surpriza de azi nu mută prognoza de mîine')]),
    T('Built for many business series with strong calendar effects and missing data; Python package \\texttt{prophet}', 'Construit pentru multe serii economice cu efecte puternice de calendar și date lipsă; pachetul Python \\texttt{prophet}')), 'footnotesize')

chart(T('Prophet components for daily load', 'Componentele Prophet pentru consumul zilnic'), 'tsa_ch4_prophet', 'TSA_ch4_tbats_prophet', [
    T('Prophet with Romanian public holidays, daily load 2022 -- June 2025', 'Prophet cu sărbătorile legale din România, consumul zilnic 2022 -- iunie 2025')],
    h='0.64\\textheight')

interp(('the Prophet components', 'componentelor Prophet'), [
    (T('Trend from @{pr.t0} to @{pr.t1} GW, flatter after 2023; @{pr.ncp} candidate changepoints, most shrunk to zero', 'Trendul de la @{pr.t0} la @{pr.t1} GW, mai plat după 2023; @{pr.ncp} de puncte de schimbare candidate, cele mai multe aduse la zero'),
     [T('weekly: Wednesday $+@{pr.wed}$, Saturday $@{pr.sat}$, Sunday $@{pr.sun}$ GW', 'săptămînal: miercuri $+@{pr.wed}$, sîmbătă $@{pr.sat}$, duminică $@{pr.sun}$ GW')]),
    T('Yearly: highest in early December ($+@{pr.ymax}$ GW, @{pr.ymaxd}), lowest in mid-September ($@{pr.ymin}$ GW, @{pr.ymind}), shown for 2024', 'Anual: maximul la începutul lui decembrie ($+@{pr.ymax}$ GW, @{pr.ymaxd}), minimul la mijlocul lui septembrie ($@{pr.ymin}$ GW, @{pr.ymind}), arătat pentru 2024'),
    T('Holidays: Christmas $@{pr.xmas}$, New Year $@{pr.ny}$, Easter $@{pr.eas}$ GW; Children\'s Day only $@{pr.ch}$ GW', 'Sărbătorile: Crăciunul $@{pr.xmas}$, Anul Nou $@{pr.ny}$, Paștele $@{pr.eas}$ GW; Ziua Copilului doar $@{pr.ch}$ GW'),
    T('Each component is readable and can be checked against what we know about the economy', 'Fiecare componentă poate fi citită și verificată cu ceea ce știm despre economie')])

D.frame(T('Recap: DHR, TBATS and Prophet side by side', 'Recapitulare: DHR, TBATS și Prophet comparate'), table(
    '>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{2.8cm}>{\\raggedright\\arraybackslash}p{2.8cm}>{\\raggedright\\arraybackslash}p{2.8cm}',
    ' & \\textbf{DHR} & \\textbf{TBATS} & \\textbf{Prophet}',
    [T('Seasonality', 'Sezonalitatea') + ' & ' + T('Fourier, fixed', 'Fourier, fixă') + ' & ' + T('Fourier, evolving', 'Fourier, evolutivă') + ' & ' + T('Fourier, fixed', 'Fourier, fixă'),
     T('Several periods', 'Mai multe perioade') + ' & ' + T('yes', 'da') + ' & ' + T('yes', 'da') + ' & ' + T('yes', 'da'),
     T('Holidays, regressors', 'Sărbători, regresori') + ' & ' + T('yes', 'da') + ' & ' + T('no', 'nu') + ' & ' + T('yes', 'da'),
     T('Short-run dynamics', 'Dinamica pe termen scurt') + ' & ' + T('ARMA errors', 'erori ARMA') + ' & ' + T('ARMA errors', 'erori ARMA') + ' & ' + T('none', 'niciuna'),
     T('Trend', 'Trendul') + ' & ' + T('through the errors (ARIMA)', 'prin erori (ARIMA)') + ' & ' + T('local (Holt)', 'local (Holt)') + ' & ' + T('piecewise linear', 'liniar pe porțiuni'),
     T('Speed', 'Viteza') + ' & ' + T('fast', 'rapidă') + ' & ' + T('slow', 'lentă') + ' & ' + T('fast', 'rapidă')],
    size='scriptsize') + items(
    T('All three are regressions on sines and cosines at heart; they differ in what else they model', 'În esență, toate trei sînt regresii pe sinusuri și cosinusuri; diferă prin ceea ce mai modelează'),
    T('Which forecasts best is an empirical question: the next section answers it out of sample', 'Care dintre ele prognozează mai bine este o întrebare empirică: secțiunea următoare îi răspunde în afara eșantionului')), 'footnotesize')

# =============================================================================
# 10. EVALUAREA PROGNOZELOR
# =============================================================================
D.section('Forecast evaluation', 'Evaluarea prognozelor')

chart(T('Time-series cross-validation', 'Validarea încrucișată pentru serii de timp'), 'tsa_ch4_cv_scheme', 'TSA_ch4_evaluation', [
    T('Rolling forecast origin (\\refTashman; \\refFPPcv): fit on the data up to the origin, forecast $h$ steps, move the origin, refit', 'Origine de prognoză mobilă (\\refTashman; \\refFPPcv): estimăm pe datele pînă la origine, prognozăm $h$ pași, mutăm originea, reestimăm'),
    T('Expanding window (shown) or sliding window of fixed length; never use future data to fit', 'Fereastră care se extinde (în figură) sau fereastră glisantă de lungime fixă; nu folosim niciodată date viitoare la estimare')],
    h='0.40\\textheight')

D.frame(T('Interpreting the design of our experiment', 'Interpretarea schemei de evaluare'), items(
    (T('Daily load: @{cv.n} origins every 14 days, from @{cv.o0} to @{cv.o1}; horizon $h = 14$ days', 'Consumul zilnic: @{cv.n} de origini la 14 zile, de la @{cv.o0} la @{cv.o1}; orizontul $h = 14$ zile'),
     [T('one year of out-of-sample forecasts, every season and every holiday, Easter 2026 included', 'un an de prognoze în afara eșantionului, toate anotimpurile și toate sărbătorile, inclusiv Paștele din 2026')]),
    (T('Seven methods, all refitted at each origin', 'Șapte metode, toate reestimate la fiecare origine'),
     [T('seasonal naive (period 7); ETS (Holt--Winters, damped trend, period 7); SARIMA@{cv.sar}$(0,1,1)_7$; DHR; TBATS; Prophet', 'sezonieră naivă (perioada 7); ETS (Holt--Winters, trend amortizat, perioada 7); SARIMA@{cv.sar}$(0,1,1)_7$; DHR; TBATS; Prophet'),
      T('Combination: the simple average of the five model forecasts (all except the seasonal naive)', 'Combinarea: media simplă a prognozelor celor cinci modele (toate, cu excepția celei sezoniere naive)')]),
    T('The orders of SARIMA and of the DHR errors are chosen once, by AIC, on the data before the first origin: no peeking at the test period', 'Ordinele SARIMA și ale erorilor DHR se aleg o singură dată, cu AIC, pe datele dinaintea primei origini: fără a privi perioada de test'),
    T('Cross-validation that ignores time order (random folds) is valid only for pure autoregressions with white-noise errors (\\refBHK)', 'Validarea încrucișată care ignoră ordinea în timp (grupuri aleatoare) este validă doar pentru autoregresii pure cu erori de tip zgomot alb (\\refBHK)')), 'footnotesize')

D.frame(T('The Diebold--Mariano test', 'Testul Diebold--Mariano'), items(
    (T('\\refDM: is the difference in accuracy between two forecasts larger than chance?', '\\refDM: este diferența de acuratețe dintre două prognoze mai mare decît hazardul?'),
     [T('loss differential $d_t = L(e_{1t}) - L(e_{2t})$, $L(e) = e^2$ or $|e|$; $H_0$: $E[d_t] = 0$', 'diferența de pierdere $d_t = L(e_{1t}) - L(e_{2t})$, $L(e) = e^2$ sau $|e|$; $H_0$: $E[d_t] = 0$')]),
    (T('$\\mathrm{DM} = \\dfrac{\\bar d}{\\sqrt{\\hat V/n}}$, $\\hat V = \\hat\\gamma_d(0) + 2\\sum_{k=1}^{h-1}\\hat\\gamma_d(k)$', '$\\mathrm{DM} = \\dfrac{\\bar d}{\\sqrt{\\hat V/n}}$, $\\hat V = \\hat\\gamma_d(0) + 2\\sum_{k=1}^{h-1}\\hat\\gamma_d(k)$'),
     [T('$h$-step errors overlap, so $d_t$ is autocorrelated up to lag $h - 1$', 'erorile la $h$ pași se suprapun, deci $d_t$ este autocorelat pînă la decalajul $h - 1$'),
      T('asymptotically $N(0, 1)$; small samples: $\\mathrm{HLN} = \\sqrt{\\frac{n + 1 - 2h + h(h - 1)/n}{n}}\\;\\mathrm{DM}$, compared with $t(n - 1)$ (\\refHLN)', 'asimptotic $N(0, 1)$; eșantioane mici: $\\mathrm{HLN} = \\sqrt{\\frac{n + 1 - 2h + h(h - 1)/n}{n}}\\;\\mathrm{DM}$, comparat cu $t(n - 1)$ (\\refHLN)')]),
    (T('Our design: one value per origin, $d_j$ = average loss difference over the 14-day window; the windows do not overlap, so $h = 1$ in the formula', 'Schema noastră: o valoare pe origine, $d_j$ = diferența medie de pierdere în fereastra de 14 zile; ferestrele nu se suprapun, deci $h = 1$ în formulă'),
     [T('the test compares forecasts, not models: re-estimation noise is part of what is tested', 'testul compară prognoze, nu modele: zgomotul reestimării face parte din ceea ce se testează')])), 'footnotesize')

D.frame(T('Worked example: a Diebold--Mariano test', 'Exemplu rezolvat: un test Diebold--Mariano'), items(
    (T('Five windows; window-average differences in squared error, method 1 minus method 2: $-0.30, 0.10, -0.50, -0.20, -0.40$', 'Cinci ferestre; diferențele medii ale erorilor pătratice, metoda 1 minus metoda 2: $-0{,}30; 0{,}10; -0{,}50; -0{,}20; -0{,}40$'),
     [T('$\\bar d = @{we4.db}$: method 1 has the smaller losses in four windows out of five', '$\\bar d = @{we4.db}$: metoda 1 are pierderi mai mici în patru ferestre din cinci')]),
    (T('$\\hat\\gamma_d(0) = \\frac15\\sum(d_j - \\bar d)^2 = @{we4.s2}$', '$\\hat\\gamma_d(0) = \\frac15\\sum(d_j - \\bar d)^2 = @{we4.s2}$'),
     [T('$\\mathrm{DM} = @{we4.db}/\\sqrt{@{we4.s2}/5} = @{we4.dm}$; $\\mathrm{HLN} = \\sqrt{4/5}\\cdot\\mathrm{DM} = @{we4.hln}$', '$\\mathrm{DM} = @{we4.db}/\\sqrt{@{we4.s2}/5} = @{we4.dm}$; $\\mathrm{HLN} = \\sqrt{4/5}\\cdot\\mathrm{DM} = @{we4.hln}$')]),
    (T('$|@{we4.hln}| < t_{0.975}(4) = @{we4.crit}$ (p = @{we4.p}): equal accuracy is not rejected at 5\\%', '$|@{we4.hln}| < t_{0{,}975}(4) = @{we4.crit}$ (p = @{we4.p}): acuratețea egală nu se respinge la 5\\%'),
     [T('the normal critical value 1.96 would have rejected: with few windows, the correction matters', 'valoarea critică a distribuției Normale, 1,96, ar fi dus la respingere: cu puține ferestre, corecția contează')])))

chart(T('Forecasts from one origin, across Easter 2026', 'Prognoze dintr-o singură origine, peste Paștele din 2026'), 'tsa_ch4_load_fc', 'TSA_ch4_evaluation', [
    T('Origin @{lf.o}, 14 days ahead; Orthodox Easter Sunday was @{ea.d26}', 'Originea @{lf.o}, 14 zile înainte; Duminica Paștelui ortodox a fost pe @{ea.d26}')],
    h='0.56\\textheight')

interp(('one forecast window', 'unei ferestre de prognoză'), [
    (T('MAE in this window (MW): ETS @{lf.ets}, TBATS @{lf.tb}, DHR @{lf.dhr}, Combination @{lf.comb}, SARIMA @{lf.sar}, Prophet @{lf.pro}, seasonal naive @{lf.sn}', 'MAE în această fereastră (MW): ETS @{lf.ets}, TBATS @{lf.tb}, DHR @{lf.dhr}, Combinarea @{lf.comb}, SARIMA @{lf.sar}, Prophet @{lf.pro}, sezonieră naivă @{lf.sn}'),
     [T('the seasonal naive method copies the previous normal week into Easter week', 'metoda sezonieră naivă copiază săptămîna normală anterioară în săptămîna Paștelui')]),
    T('Every model sees the Sunday dip; DHR and Prophet also lower Good Friday and Easter Monday, which the weekly models treat as working days', 'Toate modelele văd scăderea de duminică; DHR și Prophet coboară și Vinerea Mare și a doua zi de Paște, pe care modelele săptămînale le tratează ca zile lucrătoare'),
    T('After Easter the load stays below every forecast for the rest of the week; ETS and TBATS, whose level follows the latest days, lose least here', 'După Paște, consumul rămîne sub toate prognozele pînă la sfîrșitul săptămînii; ETS și TBATS, al căror nivel urmează ultimele zile, pierd cel mai puțin aici'),
    T('One window is anecdote; the next slides average over all @{cv.n} windows', 'O singură fereastră este o anecdotă; slide-urile următoare fac media pe toate cele @{cv.n} de ferestre')])

chart(T('Cross-validation results for daily load', 'Rezultatele validării încrucișate pentru consumul zilnic'), 'tsa_ch4_cv_results', 'TSA_ch4_evaluation', [
    T('@{cv.n} origins $\\times$ 14 horizons for each method; left: overall MAE; right: MAE by horizon', '@{cv.n} de origini $\\times$ 14 orizonturi pentru fiecare metodă; stînga: MAE totală; dreapta: MAE pe orizonturi')],
    h='0.56\\textheight')

D.frame(T('Interpreting the cross-validation results', 'Interpretarea rezultatelor validării încrucișate'), table(
    'lrrrrr', T('\\textbf{Method}', '\\textbf{Metoda}') + ' & \\textbf{MAE} & \\textbf{RMSE} & \\textbf{MASE} & ' + T('\\textbf{DM against s. naive}', '\\textbf{DM față de s. naivă}') + ' & ' + T('\\textbf{DM against DHR}', '\\textbf{DM față de DHR}'),
    [T('Seasonal naive', 'Sezonieră naivă') + ' & @{cv.sn.mae} & @{cv.sn.rmse} & @{cv.sn.mase} & -- & $@{dm.sn.dh}$ (@{dm.sn.dhp})',
     'ETS & @{cv.ets.mae} & @{cv.ets.rmse} & @{cv.ets.mase} & $@{dm.ets.sn}$ (@{dm.ets.snp}) & $@{dm.ets.dh}$ (@{dm.ets.dhp})',
     'SARIMA & @{cv.sar.mae} & @{cv.sar.rmse} & @{cv.sar.mase} & $@{dm.sar.sn}$ (@{dm.sar.snp}) & $@{dm.sar.dh}$ (@{dm.sar.dhp})',
     'DHR & \\textbf{@{cv.dhr.mae}} & \\textbf{@{cv.dhr.rmse}} & \\textbf{@{cv.dhr.mase}} & $@{dm.dhr.sn}$ (@{dm.dhr.snp}) & --',
     'TBATS & @{cv.tb.mae} & @{cv.tb.rmse} & @{cv.tb.mase} & $@{dm.tb.sn}$ (@{dm.tb.snp}) & $@{dm.tb.dh}$ (@{dm.tb.dhp})',
     'Prophet & @{cv.pro.mae} & @{cv.pro.rmse} & @{cv.pro.mase} & $@{dm.pro.sn}$ (@{dm.pro.snp}) & $@{dm.pro.dh}$ (@{dm.pro.dhp})',
     T('Combination', 'Combinarea') + ' & @{cv.comb.mae} & @{cv.comb.rmse} & @{cv.comb.mase} & $@{dm.comb.sn}$ (@{dm.comb.snp}) & $@{dm.comb.dh}$ (@{dm.comb.dhp})'],
    size='scriptsize') + items(
    T('MAE and RMSE in MW; MASE (\\refHKo) = MAE / @{cv.scale} MW, the in-sample MAE of the weekly seasonal naive method; DM: HLN statistic (p-value) on window-average squared errors, row method minus column method', 'MAE și RMSE în MW; MASE (\\refHKo) = MAE / @{cv.scale} MW, MAE în eșantion a metodei sezoniere naive săptămînale; DM: statistica HLN (valoarea p) pe media erorilor pătratice din fiecare fereastră, metoda de pe rînd minus metoda de pe coloană'),
    T('DHR wins; with the combination, it is the only method with MASE below 1; Prophet is the second single model: the two that know the holidays', 'DHR cîștigă; împreună cu combinarea, este singura metodă cu MASE sub 1; Prophet este al doilea model individual: cele două care cunosc sărbătorile'),
    T('ETS, SARIMA and TBATS are not significantly better than the seasonal naive method', 'ETS, SARIMA și TBATS nu sînt semnificativ mai bune decît metoda sezonieră naivă')), 'footnotesize')

D.frame(T('A second test: Romanian GDP', 'Un al doilea test: PIB-ul României'), table(
    'lrrrr', T('\\textbf{Method}', '\\textbf{Metoda}') + ' & ' + T('MAE, $h = 1$', 'MAE, $h = 1$') + ' & ' + T('MAE, $h = 4$', 'MAE, $h = 4$') + ' & ' + T('without 2019--2020, $h = 1$', 'fără 2019--2020, $h = 1$') + ' & ' + T('without 2019--2020, $h = 4$', 'fără 2019--2020, $h = 4$'),
    ['SARIMA (airline) & @{gc.sar.1} & @{gc.sar.4} & @{gc.sar.x1} & @{gc.sar.x4}',
     'ETS & @{gc.ets.1} & @{gc.ets.4} & @{gc.ets.x1} & @{gc.ets.x4}',
     T('Seasonal naive', 'Sezonieră naivă') + ' & @{gc.sn.1} & @{gc.sn.4} & @{gc.sn.x1} & @{gc.sn.x4}',
     T('Combination (SARIMA, ETS)', 'Combinarea (SARIMA, ETS)') + ' & @{gc.comb.1} & @{gc.comb.4} & @{gc.comb.x1} & @{gc.comb.x4}'], size='scriptsize') + items(
    T('$100\\ln Y$, errors in \\%; @{gc.n} quarterly origins, @{gc.o0}--@{gc.o1}, expanding window; @{gc.nex} origins without 2019--2020', '$100\\ln Y$, erori în \\%; @{gc.n} de origini trimestriale, @{gc.o0}--@{gc.o1}, fereastră care se extinde; @{gc.nex} de origini fără 2019--2020'),
    T('DM (HLN), SARIMA against ETS: $h = 1$: $@{gc.dm.h1.ETS}$ (p = @{gc.dmp.h1.ETS}); $h = 4$: $@{gc.dm.h4.ETS}$ (p = @{gc.dmp.h4.ETS})', 'DM (HLN), SARIMA față de ETS: $h = 1$: $@{gc.dm.h1.ETS}$ (p = @{gc.dmp.h1.ETS}); $h = 4$: $@{gc.dm.h4.ETS}$ (p = @{gc.dmp.h4.ETS})'),
    T('SARIMA against the seasonal naive method: $h = 1$: $@{gc.dm.h1.Seasonalnaive}$ (p = @{gc.dmp.h1.Seasonalnaive}); $h = 4$: $@{gc.dm.h4.Seasonalnaive}$ (p = @{gc.dmp.h4.Seasonalnaive})', 'SARIMA față de metoda sezonieră naivă: $h = 1$: $@{gc.dm.h1.Seasonalnaive}$ (p = @{gc.dmp.h1.Seasonalnaive}); $h = 4$: $@{gc.dm.h4.Seasonalnaive}$ (p = @{gc.dmp.h4.Seasonalnaive})'),
    T('One year ahead, the pandemic errors dominate and no difference is significant; without them, SARIMA has the smallest MAE at $h = 4$', 'La un an înainte, erorile din pandemie domină și nicio diferență nu este semnificativă; fără ele, SARIMA are cea mai mică MAE la $h = 4$')), 'footnotesize')

D.recap(('Forecast evaluation', 'evaluarea prognozelor'), [
    T('Evaluate out of sample with rolling origins; choose model orders before the test period', 'Evaluați în afara eșantionului, cu origini mobile; alegeți ordinele modelelor înaintea perioadei de test'),
    T('MASE compares with the seasonal naive method and across series', 'MASE compară cu metoda sezonieră naivă și între serii'),
    T('The DM test (with the HLN correction) says whether a gain is more than chance', 'Testul DM (cu corecția HLN) arată dacă un cîștig este mai mult decît hazard'),
    T('Daily load: the models that know the calendar (DHR, Prophet) win; for GDP, SARIMA and ETS are close', 'Consumul zilnic: modelele care cunosc calendarul (DHR, Prophet) cîștigă; pentru PIB, SARIMA și ETS sînt apropiate')])

# =============================================================================
# 11. COMBINAREA PROGNOZELOR
# =============================================================================
D.section('Forecast combination', 'Combinarea prognozelor')

D.frame(T('Why combining forecasts works', 'Avantajul combinării prognozelor'), items(
    (T('\\refBG: a weighted average of two unbiased forecasts, $f_c = wf_1 + (1 - w)f_2$', '\\refBG: o medie ponderată a două prognoze nedeplasate, $f_c = wf_1 + (1 - w)f_2$'),
     [T('$\\mathrm{MSE}(w) = w^2\\sigma_1^2 + (1 - w)^2\\sigma_2^2 + 2w(1 - w)\\rho\\sigma_1\\sigma_2$', '$\\mathrm{MSE}(w) = w^2\\sigma_1^2 + (1 - w)^2\\sigma_2^2 + 2w(1 - w)\\rho\\sigma_1\\sigma_2$'),
      T('best weight $w^* = \\dfrac{\\sigma_2^2 - \\rho\\sigma_1\\sigma_2}{\\sigma_1^2 + \\sigma_2^2 - 2\\rho\\sigma_1\\sigma_2}$; the gain is large when $\\rho$ is small', 'ponderea optimă $w^* = \\dfrac{\\sigma_2^2 - \\rho\\sigma_1\\sigma_2}{\\sigma_1^2 + \\sigma_2^2 - 2\\rho\\sigma_1\\sigma_2}$; cîștigul este mare cînd $\\rho$ este mic')]),
    (T('Example: $\\sigma_1 = 1$, $\\sigma_2 = 1.2$, $\\rho = 0.75$: $\\mathrm{MSE}(0.5) = @{we5.half}$; $w^* = @{we5.w}$, $\\mathrm{MSE}(w^*) = @{we5.opt}$', 'Exemplu: $\\sigma_1 = 1$, $\\sigma_2 = 1{,}2$, $\\rho = 0{,}75$: $\\mathrm{MSE}(0{,}5) = @{we5.half}$; $w^* = @{we5.w}$, $\\mathrm{MSE}(w^*) = @{we5.opt}$'),
     [T('highly correlated errors: the equal-weight average is worse than the better forecast ($\\sigma_1^2 = 1$), and even the best weight gains little', 'erori puternic corelate: media cu ponderi egale este mai slabă decît prognoza mai bună ($\\sigma_1^2 = 1$), iar chiar ponderea optimă cîștigă puțin')]),
    (T('The \\textbf{forecast combination puzzle} (\\refSW; \\refSmithWallis): estimated weights rarely beat equal weights', '\\textbf{Paradoxul combinării prognozelor} (\\refSW; \\refSmithWallis): ponderile estimate rareori bat ponderile egale'),
     [T('the weights must be estimated, and their sampling error eats the theoretical gain', 'ponderile trebuie estimate, iar eroarea lor de estimare consumă cîștigul teoretic')])), 'footnotesize')

chart(T('Combining DHR and Prophet', 'Combinarea DHR și Prophet'), 'tsa_ch4_combination', 'TSA_ch4_evaluation', [
    T('Cross-validation errors of the daily load; $w$ is the weight of DHR', 'Erorile din validarea încrucișată pentru consumul zilnic; $w$ este ponderea DHR')],
    h='0.52\\textheight')

interp(('the combination', 'combinării'), [
    (T('MAE: Prophet alone (w = 0) @{cb.w0} MW, DHR alone (w = 1) @{cb.w1} MW, equal weights @{cb.half} MW', 'MAE: doar Prophet (w = 0) @{cb.w0} MW, doar DHR (w = 1) @{cb.w1} MW, ponderi egale @{cb.half} MW'),
     [T('best in hindsight: $w = @{cb.wb}$, MAE @{cb.best} MW, a small gain chosen after seeing the test data', 'cea mai bună retrospectiv: $w = @{cb.wb}$, MAE @{cb.best} MW, un cîștig mic ales după ce am văzut datele de test')]),
    T('The errors of the two methods are strongly correlated ($\\rho = @{cb.corr}$): both miss the same cold spells', 'Erorile celor două metode sînt puternic corelate ($\\rho = @{cb.corr}$): ambele ratează aceleași valuri de frig'),
    T('The five-model average (MAE @{cv.comb.mae}) is second only to DHR and better than four of its five members: cheap insurance when the winner is not known in advance', 'Media celor cinci modele (MAE @{cv.comb.mae}) este a doua, după DHR, și mai bună decît patru dintre cele cinci componente: o asigurare ieftină cînd cîștigătorul nu este cunoscut dinainte'),
    T('In the M4 competition (\\refMFour), most of the best methods were combinations; review: \\refWangComb', 'În competiția M4 (\\refMFour), cele mai multe dintre metodele de top au fost combinații; o sinteză: \\refWangComb')], size='footnotesize')

# =============================================================================
# 12. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Code}: a first draft of a rolling-origin loop that refits SARIMA, DHR, TBATS and Prophet and stores the errors by horizon', '\\textbf{Cod}: o primă versiune a unei bucle cu origini mobile care reestimează SARIMA, DHR, TBATS și Prophet și păstrează erorile pe orizonturi'),
    T('\\textbf{Calendars}: lists of holidays and moving feasts for other countries, to be checked against official sources', '\\textbf{Calendare}: liste de sărbători și sărbători mobile pentru alte țări, care trebuie verificate cu sursele oficiale'),
    T('\\textbf{Explanation}: a second derivation of the ACF of the airline model, or of the eventual forecast function', '\\textbf{Explicații}: o a doua deducere a ACF pentru modelul airline sau a funcției de prognoză pe termen lung'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that forecasts daily electricity load 14 days ahead with a regression on annual Fourier terms, weekday dummies and Romanian public holidays (Orthodox Easter from dateutil) with ARMA errors, and evaluates it against SARIMA by rolling origins and a Diebold-Mariano test.}',
        '\\aiprompt{Write Python code that forecasts daily electricity load 14 days ahead with a regression on annual Fourier terms, weekday dummies and Romanian public holidays (Orthodox Easter from dateutil) with ARMA errors, and evaluates it against SARIMA by rolling origins and a Diebold-Mariano test.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('The Easter date: Orthodox, not Western (\\texttt{EASTER\\_ORTHODOX}); in 2024 the two differ by five weeks', 'Data Paștelui: ortodox, nu occidental (\\texttt{EASTER\\_ORTHODOX}); în 2024 cele două diferă cu cinci săptămîni'),
    T('No look-ahead: features, model orders and holiday lists must be available at each origin', 'Fără informații din viitor: variabilele, ordinele modelelor și listele de sărbători trebuie să fie disponibile la fiecare origine'),
    T('The sign conventions of $\\theta$ and $\\Theta$ in the software, and the degrees of freedom of Ljung--Box', 'Convențiile de semn pentru $\\theta$ și $\\Theta$ în program și gradele de libertate ale testului Ljung--Box'),
    T('Adjusted or unadjusted data: SARIMA on NSA; growth rates on the right version', 'Date ajustate sau neajustate: SARIMA pe NSA; ratele de creștere pe varianta potrivită'),
    T('That the DM test uses losses from the same windows for both methods, and the HLN correction for small $n$', 'Că testul DM folosește pierderile din aceleași ferestre pentru ambele metode și corecția HLN pentru $n$ mic'),
    T('Every cited reference: it must exist; check the DOI', 'Fiecare referință citată: trebuie să existe; verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Seasonality repeats with a known period; it can be deterministic (dummies, Fourier) or stochastic ($\\Delta_s$)', 'Sezonalitatea se repetă cu o perioadă cunoscută; poate fi deterministă (variabile dummy, Fourier) sau stochastică ($\\Delta_s$)'),
    T('SARIMA multiplies regular and seasonal polynomials; the airline model fits many series with two parameters', 'SARIMA înmulțește polinoame obișnuite și sezoniere; modelul airline se potrivește multor serii, cu doi parametri'),
    T('Official data: SCA for short-run growth, NSA with calendar regressors for SARIMA forecasting', 'Datele oficiale: SCA pentru creșterea pe termen scurt, NSA cu regresori de calendar pentru prognoza SARIMA'),
    T('Calendar effects (working days, Orthodox Easter, holidays) are known in advance and often matter more than the model class', 'Efectele de calendar (zile lucrătoare, Paștele ortodox, sărbători) sînt cunoscute dinainte și contează adesea mai mult decît clasa de modele'),
    T('Multiple seasonality: MSTL, DHR, TBATS, Prophet; on Romanian daily load, DHR wins out of sample', 'Sezonalitatea multiplă: MSTL, DHR, TBATS, Prophet; pentru consumul zilnic din România, DHR cîștigă în afara eșantionului'),
    T('Judge forecasts by rolling origins, MASE and the DM test; combine them when the winner is uncertain', 'Judecați prognozele cu origini mobile, MASE și testul DM; combinați-le cînd cîștigătorul este incert')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.35}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Seasonal difference', 'Diferența sezonieră') + ' & $\\Delta_sy_t = (1 - L^s)y_t$, \\quad $\\Delta\\Delta_s = 1 - L - L^s + L^{s+1}$',
     'SARIMA & $\\phi(L)\\Phi(L^s)(1 - L)^d(1 - L^s)^Dy_t = \\theta(L)\\Theta(L^s)\\varepsilon_t$',
     'Airline & $\\rho_1 = \\frac{\\theta}{1 + \\theta^2}$, \\quad $\\rho_s = \\frac{\\Theta}{1 + \\Theta^2}$, \\quad $\\rho_{s \\pm 1} = \\rho_1\\rho_s$',
     'Fourier & $\\sum_{k=1}^{K}[\\alpha_k\\sin(2\\pi kt/m) + \\beta_k\\cos(2\\pi kt/m)]$',
     'Prophet & $y(t) = g(t) + s(t) + h(t) + \\varepsilon_t$',
     'MASE & $\\frac1h\\sum|e_{T+j}| \\,/\\, \\frac{1}{T - m}\\sum|y_t - y_{t-m}|$',
     'DM & $\\bar d/\\sqrt{\\hat V/n}$, \\quad $d_t = L(e_{1t}) - L(e_{2t})$',
     T('Combination', 'Combinarea') + ' & $w^* = \\frac{\\sigma_2^2 - \\rho\\sigma_1\\sigma_2}{\\sigma_1^2 + \\sigma_2^2 - 2\\rho\\sigma_1\\sigma_2}$'],
    size='scriptsize') + '}')

D.frame(T('Self-assessment', 'Autoevaluare'), items(
    (T('\\textbf{Question}: how many parameters does SARIMA$(1,1,1)(0,1,1)_{12}$ have, and at which lags does $\\varepsilon$ appear?', '\\textbf{Întrebare}: cîți parametri are SARIMA$(1,1,1)(0,1,1)_{12}$ și la ce decalaje apare $\\varepsilon$?'),
     [T('\\textbf{Answer}: $\\phi, \\theta, \\Theta, \\sigma^2$, four; $\\varepsilon$ at lags 0, 1, 12, 13 (the last with coefficient $\\theta\\Theta$)', '\\textbf{Răspuns}: $\\phi, \\theta, \\Theta, \\sigma^2$, patru; $\\varepsilon$ la decalajele 0, 1, 12, 13 (ultimul cu coeficientul $\\theta\\Theta$)')]),
    (T('\\textbf{Question}: unadjusted GDP falls by 40\\% from Q4 to Q1. Is this a recession?', '\\textbf{Întrebare}: PIB-ul neajustat scade cu 40\\% din T4 în T1. Este o recesiune?'),
     [T('\\textbf{Answer}: no; it is the seasonal pattern; look at the SCA series or at year-on-year growth', '\\textbf{Răspuns}: nu; este tiparul sezonier; priviți seria SCA sau creșterea față de anul anterior')]),
    (T('\\textbf{Question}: a DM test with 26 windows gives HLN $= -1.93$. Is the first forecast significantly better at 5\\%?', '\\textbf{Întrebare}: un test DM cu 26 de ferestre dă HLN $= -1{,}93$. Este prima prognoză semnificativ mai bună la 5\\%?'),
     [T('\\textbf{Answer}: no: $|-1.93| < t_{0.975}(25) = 2.06$; it is better on average, but not significantly', '\\textbf{Răspuns}: nu: $|-1{,}93| < t_{0{,}975}(25) = 2{,}06$; este mai bună în medie, dar nu semnificativ')]),
    (T('\\textbf{Project idea}: forecast Romanian hourly electricity load with a DHR (daily, weekly and annual Fourier terms, holidays, temperature) against TBATS, Prophet and a combination, by rolling origins and DM tests', '\\textbf{Idee de proiect}: prognozați consumul orar de electricitate din România cu o DHR (termeni Fourier zilnici, săptămînali și anuali, sărbători, temperatură), comparată cu TBATS, Prophet și o combinare, prin origini mobile și teste DM'),
     [T('Next: Chapter 5, conditional volatility: ARCH and GARCH models', 'Urmează: Capitolul 5, volatilitatea condiționată: modelele ARCH și GARCH')])), 'footnotesize')

D.references(bib())

if __name__ == '__main__':
    finalize(D.write(V))
