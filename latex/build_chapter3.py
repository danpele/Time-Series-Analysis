r"""
build_chapter3.py -- Capitolul 3 (Rădăcini unitare și modele ARIMA), EN + RO dintr-o singură sursă
==================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_03/ch3_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter3_unit_roots_arima_models.tex
  RO/Cursuri/capitol3_radacini_unitare_modele_arima.tex
Rulare:
  python3 Quantlets/Ch_03/generate_all_charts.py
  python3 latex/build_chapter3.py && python3 latex/tsa_build.py compile 3
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, cols, table, photo   # noqa: E402
from tsa_build import items as _items   # noqa: E402


def items(*xs):
    """tsa_build.items, with (text, []) treated as a plain bullet."""
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])
from ch3_common import QLURL, REFS, T, bib, date, finalize, load, pv, qtr, month   # noqa: E402

N = load()
V = Values()
D = Deck(3, 'lecture', refs=REFS)
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
    'granger': ('ch3_clive_granger_2008.jpg', C + 'Clive_Granger_by_Olaf_Storbeck_(3x4_cropped).jpg',
                T('Photo', 'Foto') + ': Olaf Storbeck (2008); CC BY-SA 2.0; Wikimedia Commons'),
    'iowa': ('ch3_snedecor_hall_2023.jpg', C + 'Snedecor_Hall,_Iowa_State_University.tif',
             T('Photo', 'Foto') + ': Maitra (2023); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


def p3(p):
    """p-value with three decimals, or < 0.001."""
    return pv(p)


# =============================================================================
# CIFRE
# =============================================================================
F = N['four']
V.put('f.g', F['gdp_growth_q'], 2)
V.put('f.g4', 4 * F['gdp_growth_q'], 1)
V.put('f.dev', F['gdp_maxdev'], 0)
V.put('f.hicp', F['hicp_ratio'], 1)
V.put('f.fx0', F['fx_first'], 2)
V.put('f.fx1', F['fx_last'], 2)
V.put('f.bet', 100 * F['bet_logchg'], 0)
V.put('f.sp', 100 * F['sp_logchg'], 0)
V.raw('f.end', date(F['end']))

TD = N['tsds']
V.put('td.p10', TD['phi10'], 3)
V.put('td.p20', TD['phi20'], 3)
V.put('td.gds', TD['gap_ds_end'], 0)

DT = N['detrend']
V.put('dt.r2', DT['r2'], 2)
V.put('dt.t', DT['tstat'], 1)
V.put('dt.r1', DT['res_r1'], 2)
V.put('dt.rej', 100 * DT['mc_reject'], 0)
V.put('dt.r2m', DT['mc_r2_median'], 2)

SM = N['spur_mc']
for Tn in ['50', '100', '200', '1000']:
    V.put(f'sm.{Tn}.rej', 100 * SM[Tn]['levels']['reject'], 0)
    V.put(f'sm.{Tn}.drej', 100 * SM[Tn]['differences']['reject'], 1)
    V.put(f'sm.{Tn}.r2', SM[Tn]['levels']['r2'], 2)
    V.put(f'sm.{Tn}.dw', SM[Tn]['levels']['dw'], 2)
    V.put(f'sm.{Tn}.t', SM[Tn]['levels']['abs_t_median'], 1)

SR = N['spur_real']
V.put('sr.t', SR['levels']['t'], 1)
V.put('sr.r2', SR['levels']['r2'], 2)
V.put('sr.dw', SR['levels']['dw'], 2)
V.put('sr.b', SR['levels']['b'], 2)
V.put('sr.dt', SR['differences']['t'], 2)
V.put('sr.dr2', SR['differences']['r2'], 4)
V.put('sr.dp', SR['differences']['p'], 2)
V.int('sr.n', SR['levels']['n'])

DD = N['dfdist']
for reg in ['n', 'c', 'ct']:
    V.put(f'dd.{reg}.q05', DD[reg]['q05_sim'], 2)
    V.put(f'dd.{reg}.cv', DD[reg]['crit5'], 2)
    V.put(f'dd.{reg}.pn', 100 * DD[reg]['p_below_normal'], 0)
for Tn in ['50', '100', '250', 'inf']:
    for reg in ['n', 'c', 'ct']:
        for j, lev in enumerate(['1', '5', '10']):
            V.put(f'cv.{Tn}.{reg}.{lev}', DD['table'][Tn][reg][j], 2)

PW = N['power']
for Tn in [100, 250, 500]:
    for reg in ['c', 'ct']:
        for phi in [0.9, 0.95, 1.0]:
            V.put(f'pw.{Tn}.{reg}.{phi}', 100 * PW[f'{Tn}_{reg}_{phi}'], 0)

H = N['dfhand']
V.put('h.g', H['gamma'], 4)
V.put('h.se', H['se'], 4)
V.put('h.tau', H['tau'], 2)
V.put('h.cv', H['crit5'], 2)
V.put('h.phi', H['phi'], 3)
V.put('h.p', H['p'], 2)
V.raw('h.n', str(H['n']))
V.raw('h.maxlag', str(H['maxlag']))

KP = N['kpss']
V.put('kp.ar', KP['ar']['stat'], 2)
V.put('kp.rw', KP['rw']['stat'], 2)

U = N['urt']
ROWS = [('Romania real GDP, log', T('Romania, real GDP, $100\\ln Y$', 'România, PIB real, $100\\ln Y$')),
        ('Romania real GDP, growth', T('\\quad its growth $\\Delta$', '\\quad creșterea $\\Delta$')),
        ('Romania HICP, log', T('Romania, HICP, $100\\ln P$', 'România, IAPC, $100\\ln P$')),
        ('Romania inflation, 12 months', T('Romania, 12-month inflation $\\pi$', 'România, inflația anuală $\\pi$')),
        ('Romania inflation, change', T('\\quad its change $\\Delta\\pi$', '\\quad variația $\\Delta\\pi$')),
        ('US real GDP, log', T('US, real GDP, $100\\ln Y$', 'SUA, PIB real, $100\\ln Y$')),
        ('US real GDP, growth', T('\\quad its growth $\\Delta$', '\\quad creșterea $\\Delta$')),
        ('EUR/RON, log', 'EUR/RON, $100\\ln S$'),
        ('EUR/RON, returns', T('\\quad daily returns', '\\quad randamente zilnice')),
        ('BET, log price', T('BET, $100\\ln P$', 'BET, $100\\ln P$')),
        ('BET, returns', T('\\quad daily returns', '\\quad randamente zilnice')),
        ('S&P 500, log price', 'S\\&P 500, $100\\ln P$'),
        ('S&P 500, returns', T('\\quad daily returns', '\\quad randamente zilnice')),
        ('Nile flow', T('Nile, yearly flow', 'Nilul, debitul anual'))]
VERD = {'I(0)': '$I(0)$', 'I(1)': '$I(1)$', 'conflict': T('conflict', 'conflict'), 'inconclusive': T('inconclusive', 'neconcludent')}


def urow(i):
    k, lab = ROWS[i]
    u = U[k]
    reg = {'c': 'c', 'ct': 'c, t'}[u['reg']]
    V.int(f'u{i}.n', u['n'])
    V.put(f'u{i}.adf', u['adf']['stat'], 2)
    V.raw(f'u{i}.adfp', pv(u['adf']['p']))
    V.raw(f'u{i}.lags', str(u['adf']['lags']))
    V.put(f'u{i}.pp', u['pp']['stat'], 2)
    V.put(f'u{i}.kpss', u['kpss']['stat'], 3)
    V.put(f'u{i}.kcv', u['kpss']['crit5'], 3)
    return (f'{lab} & {reg} & @{{u{i}.n}} & $@{{u{i}.adf}}$ (@{{u{i}.adfp}}) & @{{u{i}.lags}} & $@{{u{i}.pp}}$ & '
            f'@{{u{i}.kpss}} & {VERD[u["verdict"]]}')


UROWS = [urow(i) for i in range(len(ROWS))]
for k in ['adf', 'pp']:
    pass
V.put('ug.adfp', U['Romania real GDP, log']['adf']['p'], 2)
V.put('ug.kpss', U['Romania real GDP, log']['kpss']['stat'], 3)
V.put('ui.kpss', U['Romania inflation, 12 months']['kpss']['stat'], 3)
V.put('ui.adfp', U['Romania inflation, 12 months']['adf']['p'], 2)
V.put('ub.ppr', U['BET, returns']['pp']['stat'], 1)
V.put('ub.adfr', U['BET, returns']['adf']['stat'], 1)

BS = N['brk_sim']
V.put('bs.adfp', BS['adf']['p'], 2)
V.put('bs.za', BS['za']['stat'], 2)
V.put('bs.zacv', BS['za']['crit5'], 2)
V.put('bs.rej', 100 * BS['rej_break'], 0)
V.put('bs.rej0', 100 * BS['rej_nobreak'], 0)
V.raw('bs.date', str(BS['best_date']))

BR = N['brk_real']
V.put('bn.za', BR['nile']['za']['stat'], 2)
V.raw('bn.y', BR['nile']['za']['break'][:4])
V.put('bn.m0', BR['nile']['m0'], 0)
V.put('bn.m1', BR['nile']['m1'], 0)
V.put('bn.adf', BR['nile']['adf']['stat'], 2)
V.raw('bn.adfp', pv(BR['nile']['adf']['p']))
V.put('bn.kpss', BR['nile']['kpss']['stat'], 2)
V.put('bf.za', BR['eurron']['za']['stat'], 2)
V.raw('bf.d', month(BR['eurron']['za']['break'][:7]))
V.put('bf.adfp', BR['eurron']['adf']['p'], 2)
V.put('bf.zat', BR['eurron']['za_t']['stat'], 2)
V.put('bf.zatcv', BR['eurron']['za_t']['crit5'], 2)
V.put('bf.zact', BR['eurron']['za_ct']['stat'], 2)
V.put('bf.zactcv', BR['eurron']['za_ct']['crit5'], 2)
V.put('bf.m0', math.exp(BR['eurron']['m0'] / 100), 2)
V.put('bf.m1', math.exp(BR['eurron']['m1'] / 100), 2)

GI = N['gdp_id']
V.int('gi.n', GI['n'])
V.raw('gi.q0', qtr(GI['first']))
V.raw('gi.q1', qtr(GI['last']))
V.put('gi.mean', GI['mean'], 2)
V.put('gi.mean4', 4 * GI['mean'], 1)
V.put('gi.sd', GI['sd'], 2)
V.put('gi.r1', GI['r1'], 2)
V.put('gi.r2', GI['r2'], 2)
V.put('gi.band', GI['band'], 2)
V.put('gi.q', GI['lb8']['lb'], 1)
V.put('gi.qp', GI['lb8']['lb_p'], 2)
V.raw('gi.w1d', qtr(GI['worst1'][0]))
V.put('gi.w1', GI['worst1'][1], 1)
V.raw('gi.w2d', qtr(GI['worst2'][0]))
V.put('gi.w2', GI['worst2'][1], 1)

GM = N['gdp_models']
gm = GM['main']['grid']
for k, v in gm.items():
    V.put(f'gm.{k}', v['aicc'], 1)
best = ''.join(map(str, GM['main']['best']))
V.raw('gm.best', f"ARIMA({GM['main']['best'][0]},1,{GM['main']['best'][1]})")
V.put('gm.gap', sorted(v['aicc'] for v in gm.values())[1] - gm[best]['aicc'], 1)
gf = GM['full']
V.raw('gf.best', f"ARIMA({gf['best_any'][0]},1,{gf['best_any'][1]})")
V.put('gf.aicc', gf['grid'][''.join(map(str, gf['best_any']))]['aicc'], 1)
V.put('gf.aicc0', gf['grid']['00']['aicc'], 1)
V.put('gf.ma', min(gf['best_any_maroots']), 4)
V.put('gf.ar', min(gf['best_any_arroots']), 3)
V.put('gf.ma1', gf['best_any_params']['ma.L1'], 2)
V.put('gf.ma1se', gf['best_any_se']['ma.L1'], 2)
V.put('gf.ma2', gf['best_any_params']['ma.L2'], 3)
V.put('gf.ma2se', gf['best_any_se']['ma.L2'], 2)
V.raw('gf.q0', qtr(gf['first']))

GD = N['gdp_diag']
V.raw('gd.order', f"ARIMA({GD['order'][0]},1,{GD['order'][2]})")
V.put('gd.drift', GD['params']['x1'], 2)
V.put('gd.dse', GD['se']['x1'], 2)
V.put('gd.dt', GD['params']['x1'] / GD['se']['x1'], 2)
V.put('gd.sig', GD['sigma'], 2)
V.put('gd.q8', GD['lb8']['lb'], 1)
V.put('gd.q8p', GD['lb8']['lb_p'], 2)
V.put('gd.q12p', GD['lb12']['lb_p'], 2)
V.put('gd.jb', GD['jb'], 0)
V.put('gd.k', GD['kurt'], 1)
V.raw('gd.mind', qtr(GD['min'][0]))
V.put('gd.min', GD['min'][1], 1)

GF = N['gdp_fc']
V.put('gf2.last', GF['last_level'], 1)
V.raw('gf2.lq', qtr(GF['last_q']))
V.raw('gf2.fq', qtr(GF['ds']['last_q']))
for lab in ['ds', 'ts']:
    for k in ['w1', 'w4', 'w8', 'w12']:
        V.put(f'gf2.{lab}.{k}', GF[lab][k], 1)
    V.put(f'gf2.{lab}.l12', GF[lab]['level12'], 1)
    V.put(f'gf2.{lab}.lo', GF[lab]['lo12'], 1)
    V.put(f'gf2.{lab}.hi', GF[lab]['hi12'], 1)
V.put('gf2.ts.phi', GF['ts']['params']['ar.L1'], 2)
V.put('gf2.ts.b', GF['ts']['params']['x1'], 2)
V.put('gf2.ratio', GF['ds']['w12'] / GF['ds']['w1'], 2)

OD = N['overdiff']
V.put('od.v1', OD['var1'], 2)
V.put('od.v2', OD['var2'], 2)
V.put('od.r1', OD['r1_d1'], 2)
V.put('od.r2', OD['r1_d2'], 2)
V.put('od.th', OD['theta'], 3)
V.put('od.thse', OD['theta_se'], 3)

W = N['width']
for i, k in enumerate(W):
    V.put(f'w{i}.4', W[k]['w4'], 2)
    V.put(f'w{i}.20', W[k]['w20'], 2)

HB = N['hand110']
for i in range(3):
    V.put(f'hb.f{i}', HB['f'][i], 2)
    V.put(f'hb.v{i}', HB['var'][i], 2)
    V.put(f'hb.h{i}', HB['half'][i], 2)
    V.put(f'hb.psi{i}', HB['psi'][i], 2)

IN = N['infl']
V.raw('in.last', month(IN['last']))
V.put('in.lastv', IN['lastv'], 1)
V.put('in.mean', IN['mean'], 1)
V.put('in.max', IN['max'], 1)
V.raw('in.maxd', month(IN['maxd']))
V.put('in.k0', IN['kpss0']['stat'], 3)
V.put('in.adf0p', IN['adf0']['p'], 2)
V.raw('in.dhk', str(IN['d_hk']))
for lab in ['d0', 'd1']:
    o = IN[lab]['order']
    V.raw(f'in.{lab}.o', f'ARIMA({o[0]},{o[1]},{o[2]})')
    V.put(f'in.{lab}.aicc', IN[lab]['aicc'], 1)
    V.put(f'in.{lab}.h24', IN[lab]['h24'], 1)
    V.put(f'in.{lab}.lo24', IN[lab]['lo24'], 1)
    V.put(f'in.{lab}.hi24', IN[lab]['hi24'], 1)
    V.put(f'in.{lab}.lo12', IN[lab]['lo12'], 1)
    V.put(f'in.{lab}.hi12', IN[lab]['hi12'], 1)
p0 = IN['d0']['params']
V.put('in.a1', p0['ar.L1'], 2)
V.put('in.a2', p0['ar.L2'], 2)
V.put('in.asum', p0['ar.L1'] + p0['ar.L2'], 3)
V.put('in.mu', p0['const'], 1)
V.put('in.b1', IN['d1']['params']['ar.L1'], 2)
best1 = min(IN['grid1'].items(), key=lambda kv: kv[1]['aicc'])
V.raw('in.bad', f'ARIMA({best1[0][0]},1,{best1[0][1]})')
V.put('in.badaicc', best1[1]['aicc'], 1)

FX = N['fx']
V.put('fx.phi', FX['phi'], 3)
V.put('fx.mu', FX['mu'], 4)
V.int('fx.ntr', FX['n_train'])
V.int('fx.nte', FX['n_test'])
for k in ['rw', 'drift', 'ar']:
    V.put(f'fx.r1.{k}', FX['rmse1'][k], 4)
    V.put(f'fx.r20.{k}', FX['rmse20'][k], 3)
V.put('fx.g1', 100 * (1 - FX['rmse1']['ar'] / FX['rmse1']['rw']), 1)
V.put('fx.g20', 100 * (1 - FX['rmse20']['ar'] / FX['rmse20']['rw']), 1)
V.put('fx.s0', FX['fx_split'], 4)
V.put('fx.s1', FX['fx_last'], 4)
V.raw('fx.n20', str(FX['n20']))

US = N['us']
V.put('us.ts', US['ts']['gap_end'], 0)
V.put('us.ds', US['ds']['gap_end'], 0)
V.put('us.ts09', US['ts']['gap_2009'], 0)
V.put('us.ds09', US['ds']['gap_2009'], 0)
V.put('us.tsh', US['ts']['half_end'], 0)
V.put('us.dsh', US['ds']['half_end'], 0)
V.raw('us.lq', qtr(US['last_q']))
V.put('us.adfp', U['US real GDP, log']['adf']['p'], 2)
V.put('us.kpss', U['US real GDP, log']['kpss']['stat'], 2)

# ---- worked examples (computed here)
V.put('ex.half', math.log(0.5) / math.log(H['phi']), 1)
V.put('ex.ses', 1 - 0.6, 1)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: when a series drifts upwards, is the shock of today temporary or permanent, and how should we forecast it?',
       '\\textbf{Întrebarea}: cînd o serie urcă în timp, este șocul de azi temporar sau permanent și cum trebuie prognozată seria?'),
     [T('the answer decides whether we detrend or difference, and how wide our forecast intervals are', 'răspunsul decide dacă eliminăm trendul sau diferențiem seria și cît de largi sînt intervalele de prognoză')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('deterministic and stochastic trends; integrated processes $I(d)$; spurious regression', 'trend determinist și trend stochastic; procese integrate $I(d)$; regresia falsă'),
      T('unit-root tests: Dickey--Fuller, ADF, Phillips--Perron; the KPSS stationarity test; structural breaks', 'teste de rădăcină unitară: Dickey--Fuller, ADF, Phillips--Perron; testul de staționaritate KPSS; rupturi structurale'),
      T('ARIMA$(p,d,q)$ models: identification, estimation, diagnostics, forecasts with growing intervals', 'modele ARIMA$(p,d,q)$: identificare, estimare, diagnosticare, prognoze cu intervale tot mai largi')]),
    T('We build on Chapter 1 (stationarity, random walk, differencing) and Chapter 2 (ARMA models); Seminar 3 comes before this lecture',
      'Pornim de la Capitolul 1 (staționaritate, mers aleator, diferențiere) și de la Capitolul 2 (modele ARMA); Seminarul 3 are loc înaintea acestui curs')))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Distinguish trend-stationary from difference-stationary series, and explain why the distinction matters', 'Deosebiți seriile staționare în jurul trendului de cele staționare în diferențe și explicați de ce contează deosebirea'),
    T('Recognise a spurious regression and explain why it happens', 'Recunoașteți o regresie falsă și explicați de ce apare'),
    T('Run and interpret the ADF, Phillips--Perron and KPSS tests, with the right deterministic terms and lags', 'Aplicați și interpretați testele ADF, Phillips--Perron și KPSS, cu termenii determiniști și numărul de laguri potrivite'),
    T('Explain how a structural break can mimic a unit root', 'Explicați cum poate o ruptură structurală să imite o rădăcină unitară'),
    T('Identify, estimate, check and forecast an ARIMA$(p,d,q)$ model, and explain why its intervals grow with the horizon', 'Identificați, estimați, verificați și folosiți pentru prognoză un model ARIMA$(p,d,q)$ și explicați de ce intervalele lui cresc cu orizontul'),
    T('Use automatic ARIMA selection critically', 'Folosiți critic selecția automată a modelelor ARIMA')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refHP, Ch.~4--5 (ARIMA models and non-stationarity)', 'Manual: \\refHP, cap.~4--5 (modele ARIMA și nestaționaritate)'),
     [T('companion, free online: \\refFPP, Ch.~9 (ARIMA models)', 'manual însoțitor, gratuit online: \\refFPP, cap.~9 (modele ARIMA)')]),
    T('Theory: \\refHamilton, Ch.~15--17 (trends, unit roots); \\refBJ\\ for the ARIMA methodology', 'Teorie: \\refHamilton, cap.~15--17 (trenduri, rădăcini unitare); \\refBJ\\ pentru metodologia ARIMA'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_03}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_03}'),
     [T('\\texttt{adfuller}, \\texttt{kpss}, \\texttt{zivot\\_andrews} and \\texttt{ARIMA} from \\texttt{statsmodels}; the Phillips--Perron test written out in a few lines',
        '\\texttt{adfuller}, \\texttt{kpss}, \\texttt{zivot\\_andrews} și \\texttt{ARIMA} din \\texttt{statsmodels}; testul Phillips--Perron scris în cîteva rînduri')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter3_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter3_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. TRENDURI
# =============================================================================
D.section('Deterministic and stochastic trends', 'Trend determinist și trend stochastic')

chart(T('Four series that drift', 'Patru serii care urcă în timp'), 'tsa_ch3_four_series', 'TSA_ch3_trends', [
    T('Romanian real GDP (Eurostat, seasonally adjusted) and HICP (Eurostat); EUR/RON, the BNR reference rate; BET and S\\&P 500 log prices, daily since 2000',
      'PIB-ul real (Eurostat, ajustat sezonier) și IAPC ale României (Eurostat); cursul EUR/RON, cursul de referință BNR; logaritmii prețurilor BET și S\\&P 500, zilnic din 2000'),
    T('None of them has a constant mean: Chapter 1 called them non-stationary', 'Niciuna nu are o medie constantă: în Capitolul 1 le-am numit nestaționare')], h='0.66\\textheight')

interp(('the four series', 'celor patru serii'), [
    (T('GDP grows by @{f.g}\\% per quarter on average (about @{f.g4}\\% per year), but it strays from the straight line by up to @{f.dev}\\%',
       'PIB-ul crește în medie cu @{f.g}\\% pe trimestru (circa @{f.g4}\\% pe an), dar se abate de la dreaptă cu pînă la @{f.dev}\\%'),
     [T('the 2009 recession: did GDP come back to the old line, or did it start a new one?', 'recesiunea din 2009: a revenit PIB-ul la vechea dreaptă sau a pornit pe una nouă?')]),
    T('Consumer prices: $\\times$@{f.hicp} since 2000; EUR/RON: from @{f.fx0} to @{f.fx1} lei; BET: $+@{f.bet}$ log points, S\\&P 500: $+@{f.sp}$ log points (to @{f.end})',
      'Prețurile de consum: $\\times$@{f.hicp} din 2000; EUR/RON: de la @{f.fx0} la @{f.fx1} lei; BET: $+@{f.bet}$ puncte logaritmice, S\\&P 500: $+@{f.sp}$ puncte logaritmice (pînă la @{f.end})'),
    (T('Two models can produce such pictures', 'Două modele pot produce asemenea grafice'),
     [T('a fixed line plus stationary noise (\\textbf{deterministic trend})', 'o dreaptă fixă plus zgomot staționar (\\textbf{trend determinist})'),
      T('a random walk with drift (\\textbf{stochastic trend})', 'un mers aleator cu derivă (\\textbf{trend stochastic})')])])

D.frame(T('Trend-stationary series', 'Serii staționare în jurul trendului'), items(
    (T('\\textbf{Definition}: $y_t = \\alpha + \\beta t + u_t$, with $u_t$ a stationary process (for example a stationary AR(1), Chapter 2)',
       '\\textbf{Definiție}: $y_t = \\alpha + \\beta t + u_t$, unde $u_t$ este un proces staționar (de exemplu un AR(1) staționar, Capitolul 2)'),
     [T('$\\alpha$: the intercept; $\\beta$: the slope, the average change per period; $t$: the time index', '$\\alpha$: termenul liber; $\\beta$: panta, adică variația medie pe perioadă; $t$: indicele de timp'),
      T('$y_t$ is \\textbf{trend-stationary} (TS): stationary after subtracting the line $\\alpha + \\beta t$', '$y_t$ este \\textbf{staționar în jurul trendului}: staționar după ce scădem dreapta $\\alpha + \\beta t$')]),
    (T('Properties', 'Proprietăți'),
     [T('$E[y_t] = \\alpha + \\beta t$; $\\mathrm{Var}(y_t) = \\mathrm{Var}(u_t)$, constant', '$E[y_t] = \\alpha + \\beta t$; $\\mathrm{Var}(y_t) = \\mathrm{Var}(u_t)$, constantă'),
      T('a shock moves $y_t$ away from the line only temporarily: with $u_t = \\phi u_{t-1} + \\varepsilon_t$, its effect after $h$ periods is $\\phi^h$',
        'un șoc îndepărtează $y_t$ de dreaptă doar temporar: cu $u_t = \\phi u_{t-1} + \\varepsilon_t$, efectul lui după $h$ perioade este $\\phi^h$'),
      T('long-run forecasts return to the line; their intervals have a bounded width', 'prognozele pe termen lung revin la dreaptă; intervalele lor au o lățime mărginită')]),
    T('Right treatment: estimate the trend by OLS (ordinary least squares) and model the residuals as ARMA', 'Tratamentul corect: estimăm trendul prin OLS (ordinary least squares, metoda celor mai mici pătrate) și modelăm reziduurile ca ARMA')))

D.frame(T('Difference-stationary series', 'Serii staționare în diferențe'), items(
    (T('\\textbf{Definition}: $y_t = \\beta + y_{t-1} + u_t$, $u_t$ stationary; by substitution, $y_t = y_0 + \\beta t + \\sum_{i=1}^{t} u_i$',
       '\\textbf{Definiție}: $y_t = \\beta + y_{t-1} + u_t$, $u_t$ staționar; prin substituție, $y_t = y_0 + \\beta t + \\sum_{i=1}^{t} u_i$'),
     [T('$\\beta$: the drift, the average change per period; $y_0$: the starting value', '$\\beta$: deriva, adică variația medie pe perioadă; $y_0$: valoarea inițială'),
      T('$y_t$ is \\textbf{difference-stationary} (DS): $\\Delta y_t = \\beta + u_t$ is stationary', '$y_t$ este \\textbf{staționar în diferențe}: $\\Delta y_t = \\beta + u_t$ este staționar'),
      T('$\\sum_{i \\le t} u_i$ is the \\textbf{stochastic trend}; $\\beta t$ is the drift', '$\\sum_{i \\le t} u_i$ este \\textbf{trendul stochastic}; $\\beta t$ provine din derivă')]),
    (T('Properties', 'Proprietăți'),
     [T('the same mean $y_0 + \\beta t$ as a TS series, but $\\mathrm{Var}(y_t)$ grows with $t$ (Chapter 1: $t\\sigma^2$ for a random walk)', 'aceeași medie $y_0 + \\beta t$ ca la o serie staționară în jurul trendului, dar $\\mathrm{Var}(y_t)$ crește cu $t$ (Capitolul 1: $t\\sigma^2$ pentru un mers aleator)'),
      T('every shock stays in the level forever: there is no line to return to', 'fiecare șoc rămîne pentru totdeauna în nivel: nu există o dreaptă la care seria să revină'),
      T('forecast intervals widen without limit as the horizon grows', 'intervalele de prognoză se lărgesc fără limită pe măsură ce orizontul crește')]),
    T('Right treatment: difference the series and model $\\Delta y_t$ as ARMA: the ARIMA model of this chapter', 'Tratamentul corect: diferențiem seria și modelăm $\\Delta y_t$ ca ARMA: modelul ARIMA din acest capitol')))

chart(T('Same drift, different memory', 'Aceeași derivă, memorie diferită'), 'tsa_ch3_ts_ds', 'TSA_ch3_trends', [
    T('Left and centre: the same shocks drive a TS series ($u_t$ AR(1), $\\phi = 0.8$) and a random walk with drift 0.2; the red paths add one shock of $+6$ at $t = 100$',
      'Stînga și centru: aceleași șocuri generează o serie staționară în jurul trendului ($u_t$ AR(1), $\\phi = 0{,}8$) și un mers aleator cu deriva 0,2; traiectoriile roșii adaugă un șoc de $+6$ la $t = 100$'),
    T('Right: the effect of a unit shock after $h$ periods', 'Dreapta: efectul unui șoc unitar după $h$ perioade')], h='0.54\\textheight')

interp(('the two trends', 'celor două trenduri'), [
    (T('TS series: the shock dies out; its effect is $0.8^{10} = @{td.p10}$ after 10 periods and $@{td.p20}$ after 20', 'Seria staționară în jurul trendului: șocul se stinge; efectul lui este $0{,}8^{10} = @{td.p10}$ după 10 perioade și $@{td.p20}$ după 20'),
     [T('at $t = 200$ the two red and blue paths coincide', 'la $t = 200$ traiectoriile roșie și albastră coincid')]),
    (T('Random walk: the whole path shifts up by @{td.gds} and stays there', 'Mersul aleator: întreaga traiectorie urcă cu @{td.gds} și rămîne acolo'),
     [T('a recession in a DS economy is a permanent loss of output, not a dip below a trend', 'o recesiune într-o economie de tip staționar în diferențe este o pierdere permanentă de producție, nu o scădere temporară sub trend')]),
    T('By eye, the two series without the shock look alike: we need a formal test (Sections 3--4)', 'Cu ochiul liber, cele două serii fără șoc arată la fel: avem nevoie de un test formal (secțiunile 3--4)')])

D.frame(T('Integrated processes', 'Procese integrate'), items(
    (T('\\textbf{Definition}: $y_t$ is \\textbf{integrated of order $d$}, $y_t \\sim I(d)$, if $\\Delta^d y_t$ is stationary and $\\Delta^{d-1}y_t$ is not',
       '\\textbf{Definiție}: $y_t$ este \\textbf{integrat de ordinul $d$}, $y_t \\sim I(d)$, dacă $\\Delta^d y_t$ este staționar, iar $\\Delta^{d-1}y_t$ nu este'),
     [T('$I(0)$: stationary (with an invertible ARMA representation, Chapter 2); $I(1)$: stationary after one difference; $I(2)$: after two',
        '$I(0)$: staționar (cu o reprezentare ARMA inversabilă, Capitolul 2); $I(1)$: staționar după o diferențiere; $I(2)$: după două'),
      T('$\\Delta^2 y_t = \\Delta y_t - \\Delta y_{t-1} = y_t - 2y_{t-1} + y_{t-2}$', '$\\Delta^2 y_t = \\Delta y_t - \\Delta y_{t-1} = y_t - 2y_{t-1} + y_{t-2}$')]),
    (T('Typical orders in economics and finance', 'Ordine tipice în economie și finanțe'),
     [T('log prices of stocks, exchange rates, log GDP: $I(1)$; returns and growth rates: $I(0)$', 'logaritmii prețurilor acțiunilor, cursurile de schimb, logaritmul PIB: $I(1)$; randamentele și ratele de creștere: $I(0)$'),
      T('price levels in high-inflation periods: possibly $I(2)$, so that inflation itself is $I(1)$', 'nivelul prețurilor în perioade cu inflație ridicată: posibil $I(2)$, deci inflația însăși ar fi $I(1)$')]),
    (T('Rules', 'Reguli'),
     [T('$I(0) + I(1) = I(1)$: the stochastic trend dominates; a TS series is $I(0)$ after removing the trend, not after differencing', '$I(0) + I(1) = I(1)$: trendul stochastic domină; o serie staționară în jurul trendului este $I(0)$ după eliminarea trendului, nu după diferențiere')])))

D.frame(T('The unit root', 'Rădăcina unitară'), items(
    (T('Write an AR($p$) with the lag operator $L$ (Chapter 1): $\\phi(L)y_t = \\varepsilon_t$, $\\phi(z) = 1 - \\phi_1 z - \\dots - \\phi_p z^p$',
       'Scriem un AR($p$) cu operatorul lag $L$ (Capitolul 1): $\\phi(L)y_t = \\varepsilon_t$, $\\phi(z) = 1 - \\phi_1 z - \\dots - \\phi_p z^p$'),
     [T('Chapter 2: stationary if all roots of $\\phi(z) = 0$ lie outside the unit circle, $|z| > 1$', 'Capitolul 2: staționar dacă toate rădăcinile ecuației $\\phi(z) = 0$ sînt în afara cercului unitate, $|z| > 1$')]),
    (T('\\textbf{Unit root}: one root equals $z = 1$, so $\\phi(z) = (1 - z)\\,\\phi^*(z)$ with the roots of $\\phi^*$ outside the circle',
       '\\textbf{Rădăcină unitară}: o rădăcină este $z = 1$, deci $\\phi(z) = (1 - z)\\,\\phi^*(z)$, iar rădăcinile lui $\\phi^*$ sînt în afara cercului'),
     [T('then $\\phi^*(L)\\,\\Delta y_t = \\varepsilon_t$: $\\Delta y_t$ is a stationary AR($p-1$), and $y_t \\sim I(1)$', 'atunci $\\phi^*(L)\\,\\Delta y_t = \\varepsilon_t$: $\\Delta y_t$ este un AR($p-1$) staționar, iar $y_t \\sim I(1)$'),
      T('AR(1): $\\phi(z) = 1 - \\phi z$ has the root $1/\\phi$, a unit root when $\\phi = 1$: the random walk', 'AR(1): $\\phi(z) = 1 - \\phi z$ are rădăcina $1/\\phi$, unitară cînd $\\phi = 1$: mersul aleator')]),
    (T('\\textbf{Example}: $y_t = 1.5y_{t-1} - 0.5y_{t-2} + \\varepsilon_t$; $1 - 1.5z + 0.5z^2 = (1 - z)(1 - 0.5z)$', '\\textbf{Exemplu}: $y_t = 1{,}5y_{t-1} - 0{,}5y_{t-2} + \\varepsilon_t$; $1 - 1{,}5z + 0{,}5z^2 = (1 - z)(1 - 0{,}5z)$'),
     [T('roots 1 and 2: $\\Delta y_t = 0.5\\,\\Delta y_{t-1} + \\varepsilon_t$, an AR(1) in differences', 'rădăcinile 1 și 2: $\\Delta y_t = 0{,}5\\,\\Delta y_{t-1} + \\varepsilon_t$, un AR(1) în diferențe')])))

D.frame(T('Case study: Nelson and Plosser (1982)', 'Studiu de caz: Nelson și Plosser (1982)'), items(
    (T('\\textbf{Question}: are US macroeconomic series stationary around a trend, as the business-cycle models of the 1970s assumed?',
       '\\textbf{Întrebarea}: sînt seriile macroeconomice ale SUA staționare în jurul unui trend, cum presupuneau modelele ciclului economic din anii 1970?'),
     [T('\\refNP: 14 long annual series (real GNP, employment, prices, wages, interest rates, stock prices), some since 1860', '\\refNP: 14 serii anuale lungi (PNB real, ocuparea, prețurile, salariile, ratele dobînzii, prețurile acțiunilor), unele din 1860')]),
    (T('\\textbf{Method}: the Dickey--Fuller test of Section 3, with a constant and a trend', '\\textbf{Metoda}: testul Dickey--Fuller din secțiunea 3, cu constantă și trend'),
     [T('the unit root was rejected for one series only, the unemployment rate', 'rădăcina unitară a fost respinsă pentru o singură serie, rata șomajului')]),
    (T('\\textbf{Consequence}: shocks to output look permanent, not transitory', '\\textbf{Consecința}: șocurile asupra producției par permanente, nu temporare'),
     [T('a large debate on the nature of recessions; and a warning against detrending by a straight line (next slide)', 'o dezbatere amplă despre natura recesiunilor și un avertisment împotriva eliminării trendului printr-o dreaptă (slide-ul următor)'),
      T('the answer was later challenged by structural breaks (Section 5)', 'concluzia a fost contestată ulterior prin rupturi structurale (secțiunea 5)')])))

chart(T('Detrending a random walk', 'Eliminarea trendului dintr-un mers aleator'), 'tsa_ch3_detrend', 'TSA_ch3_trends', [
    T('Left: a driftless random walk ($T = 200$) and the OLS line $\\hat a + \\hat b t$; centre: the residuals; right: their ACF',
      'Stînga: un mers aleator fără derivă ($T = 200$) și dreapta OLS $\\hat a + \\hat b t$; centru: reziduurile; dreapta: ACF a reziduurilor')], h='0.56\\textheight')

interp(('the detrended random walk', 'mersului aleator fără trend'), [
    (T('The line fits: $R^2 = @{dt.r2}$, $t = @{dt.t}$ for the slope, although the true slope is 0', 'Dreapta se potrivește: $R^2 = @{dt.r2}$, $t = @{dt.t}$ pentru pantă, deși panta adevărată este 0'),
     [T('over @{detr.R} random walks: $|t| > 1.96$ in @{dt.rej}\\% of cases, median $R^2 = @{dt.r2m}$', 'pe @{detr.R} mersuri aleatoare: $|t| > 1{,}96$ în @{dt.rej}\\% din cazuri, $R^2$ median $= @{dt.r2m}$')]),
    (T('The residuals look like long cycles: $\\hat\\rho(1) = @{dt.r1}$ and a slow decay', 'Reziduurile par cicluri lungi: $\\hat\\rho(1) = @{dt.r1}$ și o descreștere lentă'),
     [T('\\refNK: detrending a DS series creates \\textbf{spurious cycles}; a business cycle may be an artefact of the method', '\\refNK: eliminarea trendului dintr-o serie staționară în diferențe creează \\textbf{cicluri false}; un ciclu economic poate fi un artefact al metodei')]),
    T('The reverse error (differencing a TS series) is over-differencing (Chapter 1 and Section 6)', 'Eroarea inversă (diferențierea unei serii staționare în jurul trendului) este supradiferențierea (Capitolul 1 și secțiunea 6)')])
V.raw('detr.R', f"{DT['R']:,}".replace(',', '\\,'))

D.recap(('Deterministic and stochastic trends', 'trend determinist și trend stochastic'), [
    T('TS: $y_t = \\alpha + \\beta t + u_t$, shocks fade; DS: $\\Delta y_t = \\beta + u_t$, shocks are permanent', 'Staționară în jurul trendului: $y_t = \\alpha + \\beta t + u_t$, șocurile se sting; staționară în diferențe: $\\Delta y_t = \\beta + u_t$, șocurile sînt permanente'),
    T('$I(d)$: stationary after $d$ differences; a unit root in $\\phi(z)$ means $I(1)$', '$I(d)$: staționar după $d$ diferențieri; o rădăcină unitară a lui $\\phi(z)$ înseamnă $I(1)$'),
    T('The wrong treatment creates artefacts: spurious cycles (detrending a DS series) or over-differencing (differencing a TS series)', 'Tratamentul greșit creează artefacte: cicluri false (eliminarea trendului dintr-o serie staționară în diferențe) sau supradiferențiere (diferențierea unei serii staționare în jurul trendului)')])

# =============================================================================
# 2. REGRESIA FALSĂ
# =============================================================================
D.section('Spurious regression', 'Regresia falsă')

D.frame(T('1926: nonsense correlations', '1926: corelații fără sens'), items(
    (T('\\refYule\\ asked: \\textit{Why do we sometimes get nonsense-correlations between time-series?}', '\\refYule\\ a întrebat: \\textit{Why do we sometimes get nonsense-correlations between time-series?} (De ce obținem uneori corelații fără sens între serii de timp?)'),
     [T('his example: in England and Wales, 1866--1911, the share of Church of England marriages and the mortality rate had a correlation of about 0.95',
        'exemplul lui: în Anglia și Țara Galilor, 1866--1911, ponderea căsătoriilor oficiate de Biserica Angliei și rata mortalității aveau o corelație de circa 0,95'),
      T('no causal link: both series simply declined over the period', 'nicio legătură cauzală: ambele serii au scăzut în perioada respectivă')]),
    (T('His explanation: for series that wander like sums of random terms, the sample correlation does not settle near 0', 'Explicația lui: pentru seriile care evoluează ca sumele de termeni aleatori, corelația de selecție nu se stabilizează în jurul lui 0'),
     [T('large values occur often by chance; the usual standard error does not apply', 'valorile mari apar des din întîmplare; eroarea standard obișnuită nu se aplică')]),
    T('Fifty years later, the same problem reappeared in econometric models estimated on levels', 'Cincizeci de ani mai tîrziu, aceeași problemă a reapărut în modelele econometrice estimate pe niveluri')))

D.frame(T('1974: Granger and Newbold', '1974: Granger și Newbold'), cols(
    ph('granger', T('Clive Granger (1934--2009), Nobel Prize in Economics 2003', 'Clive Granger (1934--2009), Premiul Nobel pentru economie 2003'), h='0.42\\textheight'),
    items((T('\\refGN: regress one random walk on another, \\textbf{independent} random walk', '\\refGN: regresia unui mers aleator pe un alt mers aleator, \\textbf{independent}'),
           [T('$y_t = a + b x_t + e_t$ with $y_t$, $x_t$ independent $I(1)$ series', '$y_t = a + b x_t + e_t$, cu $y_t$, $x_t$ serii $I(1)$ independente'),
            T('the $t$-test of $b = 0$ rejects far too often; $R^2$ is high; the Durbin--Watson statistic is close to 0', 'testul $t$ pentru $b = 0$ respinge mult prea des; $R^2$ este mare; statistica Durbin--Watson este apropiată de 0')]),
          (T('Their rule of thumb: be suspicious when $R^2 > \\mathrm{DW}$', 'Regula lor practică: suspectați o regresie falsă cînd $R^2 > \\mathrm{DW}$'),
           [T('DW (Durbin--Watson) $\\approx 2(1 - \\hat\\rho_1)$ of the residuals: near 0 means $I(1)$-like residuals', 'DW (Durbin--Watson) $\\approx 2(1 - \\hat\\rho_1)$ al reziduurilor: aproape de 0 indică reziduuri de tip $I(1)$')]),
          T('Granger and Newbold were then at the University of Nottingham; Granger later received the Nobel Prize for cointegration (Chapter 7)', 'Granger și Newbold lucrau atunci la Universitatea din Nottingham; Granger a primit ulterior Premiul Nobel pentru cointegrare (Capitolul 7)')),
    wl='0.30', wr='0.66'))

chart(T('Spurious regression by simulation', 'Regresia falsă prin simulare'), 'tsa_ch3_spurious_mc', 'TSA_ch3_spurious_regression', [
    T('2\\,000 pairs of independent Gaussian random walks for each $T$; OLS of $y$ on a constant and $x$, in levels and in first differences',
      '2\\,000 de perechi de mersuri aleatoare gaussiene independente pentru fiecare $T$; OLS a lui $y$ pe o constantă și pe $x$, în niveluri și în diferențe'),
    T('Left: share of samples with $|t| > 1.96$; right: the distribution of $t$ for $T = 200$', 'Stînga: proporția eșantioanelor cu $|t| > 1{,}96$; dreapta: distribuția lui $t$ pentru $T = 200$')], h='0.55\\textheight')

interp(('the simulation', 'simulării'), [
    (T('Levels: a true null $b = 0$ is rejected in @{sm.50.rej}\\% of samples with $T = 50$ and in @{sm.1000.rej}\\% with $T = 1000$', 'Niveluri: ipoteza adevărată $b = 0$ este respinsă în @{sm.50.rej}\\% din eșantioane cu $T = 50$ și în @{sm.1000.rej}\\% cu $T = 1000$'),
     [T('more data makes it worse: the median $|t|$ grows from @{sm.50.t} ($T = 50$) to @{sm.1000.t} ($T = 1000$)', 'mai multe date înrăutățesc situația: $|t|$ median crește de la @{sm.50.t} ($T = 50$) la @{sm.1000.t} ($T = 1000$)'),
      T('median $R^2$ about @{sm.200.r2} for every $T$; median DW @{sm.200.dw} at $T = 200$', '$R^2$ median de circa @{sm.200.r2} pentru orice $T$; DW median @{sm.200.dw} la $T = 200$')]),
    (T('Differences: @{sm.200.drej}\\% at $T = 200$, close to the nominal 5\\%', 'Diferențe: @{sm.200.drej}\\% la $T = 200$, aproape de nivelul nominal de 5\\%'),
     [T('the $t$-test works again because $\\Delta y_t$ and $\\Delta x_t$ are stationary', 'testul $t$ funcționează din nou, deoarece $\\Delta y_t$ și $\\Delta x_t$ sînt staționare')])])

D.frame(T('Why the $t$-test fails', 'Motivul pentru care testul $t$ eșuează'), items(
    (T('Under $b = 0$ the error $e_t = y_t - a$ is itself a random walk: the OLS assumptions (stationary, uncorrelated errors) fail', 'În ipoteza $b = 0$, eroarea $e_t = y_t - a$ este ea însăși un mers aleator: ipotezele OLS (erori staționare, necorelate) nu sînt îndeplinite'),
     [T('the usual standard error is far too small', 'eroarea standard obișnuită este mult prea mică')]),
    (T('\\refPhillips\\ derived the large-sample behaviour', '\\refPhillips\\ a dedus comportamentul în eșantioane mari'),
     [T('$\\hat b$ does not converge to 0: it converges to a random variable', '$\\hat b$ nu converge la 0: converge la o variabilă aleatoare'),
      T('$t_{\\hat b}$ grows like $\\sqrt{T}$, so the rejection rate tends to 100\\%', '$t_{\\hat b}$ crește ca $\\sqrt{T}$, deci rata de respingere tinde la 100\\%'),
      T('$R^2$ has a non-degenerate limit distribution; $\\mathrm{DW} \\to 0$', '$R^2$ are o distribuție limită nedegenerată; $\\mathrm{DW} \\to 0$')]),
    T('Exception: if $y_t - b x_t$ is stationary for some $b$, the series are \\textbf{cointegrated} and the levels regression is meaningful \\refEG\\ (Chapter 7)',
      'Excepția: dacă $y_t - b x_t$ este staționar pentru un anumit $b$, seriile sînt \\textbf{cointegrate}, iar regresia în niveluri are sens \\refEG\\ (Capitolul 7)')))

chart(T('A spurious regression on real data', 'O regresie falsă pe date reale'), 'tsa_ch3_spurious_real', 'TSA_ch3_spurious_regression', [
    T('Romanian HICP (Eurostat) against the S\\&P 500 (month-end close), @{sr.n} months since January 2005, both as $100\\ln$',
      'IAPC al României (Eurostat) față de S\\&P 500 (închiderea de la sfîrșitul lunii), @{sr.n} luni din ianuarie 2005, ambele ca $100\\ln$'),
    T('Left: levels; right: monthly changes', 'Stînga: niveluri; dreapta: variații lunare')], h='0.55\\textheight')

interp(('the HICP regression', 'regresiei IAPC'), [
    (T('Levels: slope @{sr.b}, $t = @{sr.t}$, $R^2 = @{sr.r2}$, $\\mathrm{DW} = @{sr.dw}$', 'Niveluri: panta @{sr.b}, $t = @{sr.t}$, $R^2 = @{sr.r2}$, $\\mathrm{DW} = @{sr.dw}$'),
     [T('``US stocks explain Romanian consumer prices\'\': $R^2 > \\mathrm{DW}$, a textbook spurious regression', '„Acțiunile americane explică prețurile de consum din România”: $R^2 > \\mathrm{DW}$, o regresie falsă tipică')]),
    (T('Differences: $t = @{sr.dt}$ (p = @{sr.dp}), $R^2 = @{sr.dr2}$', 'Diferențe: $t = @{sr.dt}$ (p = @{sr.dp}), $R^2 = @{sr.dr2}$'),
     [T('monthly inflation and monthly stock returns are unrelated', 'inflația lunară și randamentul lunar al acțiunilor nu sînt legate')]),
    T('Both levels contain a trend; the regression only says that both went up', 'Ambele niveluri conțin un trend; regresia spune doar că ambele au crescut')])

D.recap(('Spurious regression', 'regresia falsă'), [
    T('Regressing one $I(1)$ series on an unrelated $I(1)$ series gives large $t$, high $R^2$ and DW near 0', 'Regresia unei serii $I(1)$ pe o altă serie $I(1)$ fără legătură cu ea dă un $t$ mare, un $R^2$ mare și un DW apropiat de 0'),
    T('More observations do not help: $t$ grows like $\\sqrt{T}$', 'Mai multe observații nu ajută: $t$ crește ca $\\sqrt{T}$'),
    T('Remedies: test the order of integration first; regress differences; or test for cointegration (Chapter 7)', 'Remedii: testăm întîi ordinul de integrare; estimăm regresia pe diferențe; sau testăm cointegrarea (Capitolul 7)')])

# =============================================================================
# 3. DICKEY-FULLER
# =============================================================================
D.section('The Dickey--Fuller test', 'Testul Dickey--Fuller')

D.frame(T('Testing $\\phi = 1$', 'Testarea ipotezei $\\phi = 1$'), items(
    (T('AR(1): $y_t = \\phi y_{t-1} + \\varepsilon_t$; subtract $y_{t-1}$: $\\Delta y_t = \\gamma y_{t-1} + \\varepsilon_t$, with $\\gamma = \\phi - 1$',
       'AR(1): $y_t = \\phi y_{t-1} + \\varepsilon_t$; scădem $y_{t-1}$: $\\Delta y_t = \\gamma y_{t-1} + \\varepsilon_t$, cu $\\gamma = \\phi - 1$'),
     [T('$H_0$: $\\gamma = 0$ (unit root, $I(1)$); $H_1$: $\\gamma < 0$ (stationary); a one-sided, left-tail test', '$H_0$: $\\gamma = 0$ (rădăcină unitară, $I(1)$); $H_1$: $\\gamma < 0$ (staționar); un test unilateral, la stînga'),
      T('statistic: $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$, the usual OLS $t$-ratio; $\\mathrm{SE}$: the standard error; a very negative $\\tau$ speaks against the unit root', 'statistica: $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$, raportul $t$ obișnuit din OLS; $\\mathrm{SE}$: eroarea standard; un $\\tau$ foarte negativ este un argument împotriva rădăcinii unitare')]),
    (T('Under $H_0$, $\\tau$ is \\textbf{not} Student or Normal \\refDF', 'În ipoteza $H_0$, $\\tau$ \\textbf{nu} urmează legea Student sau distribuția Normală \\refDF'),
     [T('$y_{t-1}$ is a random walk, so the regressor is not stationary; $\\hat\\phi$ converges at rate $T$, not $\\sqrt{T}$ (\\textbf{superconsistency})',
        '$y_{t-1}$ este un mers aleator, deci regresorul nu este staționar; $\\hat\\phi$ converge cu viteza $T$, nu $\\sqrt{T}$ (\\textbf{superconsistență})'),
      T('the limit of $\\tau$ is a functional of Brownian motion, skewed to the left \\refHamilton, Ch.~17', 'limita lui $\\tau$ este o funcțională a mișcării browniene, asimetrică la stînga \\refHamilton, cap.~17'),
      T('critical values come from simulation: \\refFuller; today from response surfaces \\refMacKinnonb', 'valorile critice provin din simulare: \\refFuller; astăzi, din suprafețe de răspuns \\refMacKinnonb')])))

D.frame(T('Iowa State: where the test was born', 'Iowa State: locul unde s-a născut testul'), cols(
    ph('iowa', T('Snedecor Hall, Iowa State University: the department of statistics and its Statistical Laboratory (1933)', 'Snedecor Hall, Iowa State University: departamentul de statistică și laboratorul său de statistică (1933)'), h='0.44\\textheight'),
    items(T('\\textbf{Wayne A. Fuller}: statistician at Iowa State, author of \\textit{Introduction to Statistical Time Series} \\refFuller', '\\textbf{Wayne A. Fuller}: statistician la Iowa State, autorul lucrării \\textit{Introduction to Statistical Time Series} \\refFuller'),
          T('\\textbf{David A. Dickey}: his doctoral student at Iowa State (1976), later at North Carolina State University', '\\textbf{David A. Dickey}: doctorandul lui la Iowa State (1976), apoi la North Carolina State University'),
          T('\\refDF: the distribution of $\\hat\\phi$ and $\\tau$ under a unit root; \\refDFb: joint tests of the unit root and the deterministic terms', '\\refDF: distribuția lui $\\hat\\phi$ și $\\tau$ în cazul unei rădăcini unitare; \\refDFb: teste comune pentru rădăcina unitară și termenii determiniști'),
          T('\\refSD: the augmented test for ARMA errors (this section)', '\\refSD: testul augmentat pentru erori ARMA (în această secțiune)')),
    wl='0.48', wr='0.48'))

D.frame(T('Three Dickey--Fuller regressions', 'Trei regresii Dickey--Fuller'), items(
    (T('\\textbf{No constant}: $\\Delta y_t = \\gamma y_{t-1} + \\varepsilon_t$', '\\textbf{Fără constantă}: $\\Delta y_t = \\gamma y_{t-1} + \\varepsilon_t$'),
     [T('$H_1$: stationary with mean 0; rarely realistic', '$H_1$: staționar cu media 0; rareori realist')]),
    (T('\\textbf{Constant}: $\\Delta y_t = c + \\gamma y_{t-1} + \\varepsilon_t$', '\\textbf{Constantă}: $\\Delta y_t = c + \\gamma y_{t-1} + \\varepsilon_t$'),
     [T('$H_0$: random walk (with drift if $c \\ne 0$); $H_1$: stationary around a non-zero mean', '$H_0$: mers aleator (cu derivă dacă $c \\ne 0$); $H_1$: staționar în jurul unei medii nenule'),
      T('for series without a trend: interest rates, inflation, exchange rates, returns', 'pentru serii fără trend: rate ale dobînzii, inflație, cursuri de schimb, randamente')]),
    (T('\\textbf{Constant and trend}: $\\Delta y_t = c + bt + \\gamma y_{t-1} + \\varepsilon_t$', '\\textbf{Constantă și trend}: $\\Delta y_t = c + bt + \\gamma y_{t-1} + \\varepsilon_t$'),
     [T('$c$: the constant; $bt$: a linear trend with slope $b$', '$c$: constanta; $bt$: un trend liniar cu panta $b$'),
      T('$H_0$: random walk with drift; $H_1$: trend-stationary: exactly the TS against DS question of Section 1', '$H_0$: mers aleator cu derivă; $H_1$: staționar în jurul trendului: exact întrebarea din secțiunea 1'),
      T('for trending series: log GDP, log prices, log price indices', 'pentru serii cu trend: logaritmul PIB, logaritmii prețurilor, logaritmii indicilor de preț')]),
    T('Each case has its own distribution of $\\tau$ and its own critical values', 'Fiecare caz are propria distribuție a lui $\\tau$ și propriile valori critice')))

chart(T('The Dickey--Fuller distributions', 'Distribuțiile Dickey--Fuller'), 'tsa_ch3_df_dist', 'TSA_ch3_dickey_fuller', [
    T('20\\,000 random walks with $T = 250$: the density of $\\tau$ in the three regressions, against $N(0, 1)$; dashed lines: 5\\% critical values \\refMacKinnonb',
      '20\\,000 de mersuri aleatoare cu $T = 250$: densitatea lui $\\tau$ în cele trei regresii, față de $N(0, 1)$; liniile întrerupte: valorile critice de 5\\% \\refMacKinnonb')], h='0.56\\textheight')

interp(('the three distributions', 'celor trei distribuții'), [
    (T('All three lie to the left of $N(0, 1)$; the more deterministic terms, the further left', 'Toate trei sînt la stînga lui $N(0, 1)$; cu cît sînt mai mulți termeni determiniști, cu atît mai la stînga'),
     [T('simulated 5\\% quantiles: $@{dd.n.q05}$ (no constant), $@{dd.c.q05}$ (constant) and $@{dd.ct.q05}$ (constant and trend); tabulated: $@{dd.n.cv}$, $@{dd.c.cv}$ and $@{dd.ct.cv}$', 'cuantilele de 5\\% simulate: $@{dd.n.q05}$ (fără constantă), $@{dd.c.q05}$ (constantă) și $@{dd.ct.q05}$ (constantă și trend); valorile din tabele: $@{dd.n.cv}$; $@{dd.c.cv}$ și $@{dd.ct.cv}$')]),
    (T('With the Normal critical value $-1.645$, a true unit root is ``rejected\'\' in @{dd.c.pn}\\% of samples (constant) and @{dd.ct.pn}\\% (constant and trend)',
       'Cu valoarea critică a distribuției Normale, $-1{,}645$, o rădăcină unitară adevărată este „respinsă” în @{dd.c.pn}\\% din eșantioane (constantă) și în @{dd.ct.pn}\\% (constantă și trend)'),
     [T('the wrong table turns random walks into ``stationary\'\' series', 'tabelul greșit transformă mersurile aleatoare în serii „staționare”')])])

D.frame(T('Critical values of the Dickey--Fuller test', 'Valorile critice ale testului Dickey--Fuller'), table(
    'lccc|ccc|ccc', T('& \\multicolumn{3}{c|}{\\textbf{no constant}} & \\multicolumn{3}{c|}{\\textbf{constant}} & \\multicolumn{3}{c}{\\textbf{constant and trend}} \\\\ $T$ & 1\\% & 5\\% & 10\\% & 1\\% & 5\\% & 10\\% & 1\\% & 5\\% & 10\\%',
                      '& \\multicolumn{3}{c|}{\\textbf{fără constantă}} & \\multicolumn{3}{c|}{\\textbf{constantă}} & \\multicolumn{3}{c}{\\textbf{constantă și trend}} \\\\ $T$ & 1\\% & 5\\% & 10\\% & 1\\% & 5\\% & 10\\% & 1\\% & 5\\% & 10\\%'),
    [f'{lab} & ' + ' & '.join(f'$@{{cv.{tn}.{reg}.{lev}}}$' for reg in ['n', 'c', 'ct'] for lev in ['1', '5', '10'])
     for tn, lab in [('50', '50'), ('100', '100'), ('250', '250'), ('inf', '$\\infty$')]], size='scriptsize') + items(
    T('Source: the response surfaces of \\refMacKinnonb, as used by \\texttt{statsmodels}; p-values from \\refMacKinnon', 'Sursa: suprafețele de răspuns din \\refMacKinnonb, folosite de \\texttt{statsmodels}; p-value-urile din \\refMacKinnon'),
    T('Decision: reject the unit root at 5\\% if $\\tau$ is \\textbf{below} (more negative than) the 5\\% value', 'Decizia: respingem rădăcina unitară la 5\\% dacă $\\tau$ este \\textbf{sub} (mai negativ decît) valoarea de 5\\%'),
    T('The values depend only weakly on $T$; they apply unchanged to the ADF test', 'Valorile depind puțin de $T$; se aplică neschimbate testului ADF')))

D.frame(T('Worked example: Romanian real GDP', 'Exemplu rezolvat: PIB-ul real al României'), items(
    (T('$y_t = 100\\ln$ real GDP, @{gi.q0}--@{gi.q1}; regression with constant and trend, no lags ($T = @{h.n}$ differences)',
       '$y_t = 100\\ln$ PIB real, @{gi.q0}--@{gi.q1}; regresia cu constantă și trend, fără laguri ($T = @{h.n}$ diferențe)'),
     [T('OLS: $\\hat\\gamma = @{h.g}$, $\\mathrm{SE}(\\hat\\gamma) = @{h.se}$, so $\\hat\\phi = 1 + \\hat\\gamma = @{h.phi}$', 'OLS: $\\hat\\gamma = @{h.g}$, $\\mathrm{SE}(\\hat\\gamma) = @{h.se}$, deci $\\hat\\phi = 1 + \\hat\\gamma = @{h.phi}$')]),
    (T('$\\tau = @{h.g}/@{h.se} = @{h.tau}$; 5\\% critical value with constant and trend: $@{h.cv}$', '$\\tau = @{h.g}/@{h.se} = @{h.tau}$; valoarea critică de 5\\% cu constantă și trend: $@{h.cv}$'),
     [T('$@{h.tau} > @{h.cv}$: do not reject the unit root (p = @{h.p})', '$@{h.tau} > @{h.cv}$: nu respingem rădăcina unitară (p = @{h.p})'),
      T('with the Normal value $-1.645$ we would have rejected it: the wrong table gives the wrong answer', 'cu valoarea distribuției Normale, $-1{,}645$, am fi respins-o: tabelul greșit dă răspunsul greșit')]),
    (T('Reading: ``not rejected\'\' is not ``proved\'\'', 'Interpretarea: „nerespins” nu înseamnă „demonstrat”'),
     [T('if $\\phi = @{h.phi}$ were true, a shock would halve in $\\ln 0.5/\\ln @{h.phi} \\approx @{ex.half}$ quarters; with 106 observations the test can hardly tell this from $\\phi = 1$',
        'dacă $\\phi = @{h.phi}$ ar fi adevărat, un șoc s-ar înjumătăți în $\\ln 0{,}5/\\ln @{h.phi} \\approx @{ex.half}$ trimestre; cu 106 observații, testul abia poate deosebi această situație de $\\phi = 1$')])))

D.frame(T('The augmented Dickey--Fuller test', 'Testul Dickey--Fuller augmentat'), items(
    (T('If $\\Delta y_t$ is autocorrelated, $\\varepsilon_t$ in the DF regression is not white noise and the size of the test is wrong', 'Dacă $\\Delta y_t$ este autocorelat, $\\varepsilon_t$ din regresia DF nu este zgomot alb, iar mărimea testului este greșită'),
     [T('the remedy: add lagged differences', 'remediul: adăugăm diferențele trecute $\\Delta y_{t-j}$')]),
    (T('\\textbf{ADF} (augmented Dickey--Fuller) regression:', 'Regresia \\textbf{ADF} (augmented Dickey--Fuller, Dickey--Fuller augmentat):'),
     [T('$\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum_{j=1}^{k} \\delta_j \\Delta y_{t-j} + \\varepsilon_t$; same $\\tau$, same critical values', '$\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum_{j=1}^{k} \\delta_j \\Delta y_{t-j} + \\varepsilon_t$; același $\\tau$, aceleași valori critice'),
      T('$\\delta_j$: the coefficients of the past differences; $k$: the number of lagged differences', '$\\delta_j$: coeficienții diferențelor trecute; $k$: numărul de diferențe trecute incluse (lagurile)'),
      T('an AR($p$) in levels becomes exactly this regression with $k = p - 1$ and $\\gamma = -\\phi(1)$', 'un AR($p$) în niveluri devine exact această regresie, cu $k = p - 1$ și $\\gamma = -\\phi(1)$')]),
    (T('\\refSD: with ARMA errors the test stays valid if $k$ grows slowly with $T$', '\\refSD: cu erori ARMA, testul rămîne valid dacă $k$ crește încet odată cu $T$'),
     [T('the lags approximate the MA part by a long AR', 'lagurile aproximează partea MA printr-un AR lung')]),
    T('The ADF test is the default unit-root test in software: \\texttt{adfuller} in \\texttt{statsmodels}', 'Testul ADF este testul implicit de rădăcină unitară în programe: \\texttt{adfuller} în \\texttt{statsmodels}')))

D.frame(T('Choosing the number of lags $k$', 'Alegerea numărului de laguri $k$'), items(
    (T('Maximum: $k_{\\max} = 12\\,(T/100)^{1/4}$ \\refSchwert; for $T = @{h.n}$, $k_{\\max} = @{h.maxlag}$', 'Maximul: $k_{\\max} = 12\\,(T/100)^{1/4}$ \\refSchwert; pentru $T = @{h.n}$, $k_{\\max} = @{h.maxlag}$'),
     [T('then choose $k \\le k_{\\max}$ by an information criterion, AIC or BIC (Chapter 2), on the same sample for every $k$', 'apoi alegem $k \\le k_{\\max}$ după un criteriu informațional, AIC sau BIC (Capitolul 2), pe același eșantion pentru fiecare $k$')]),
    (T('Too few lags: autocorrelated errors, wrong size (with a negative MA part, far too many rejections)', 'Prea puține laguri: erori autocorelate, mărime greșită (cu o componentă MA negativă, mult prea multe respingeri)'),
     [T('too many lags: lower power', 'prea multe laguri: putere mai mică')]),
    (T('\\refNgP: a modified AIC (MAIC) gives a better size; GLS detrending \\refERS\\ gives more power', '\\refNgP: un AIC modificat (MAIC) dă o mărime mai bună; eliminarea trendului prin GLS \\refERS\\ dă mai multă putere'),
     [T('MAIC: modified Akaike information criterion; GLS: generalised least squares; both appear in the DF-GLS test', 'MAIC: modified Akaike information criterion (criteriul Akaike modificat); GLS: generalised least squares (metoda generalizată a celor mai mici pătrate); ambele apar în testul DF-GLS')]),
    T('Check the result: Ljung--Box on the ADF residuals \\refLB; report $k$ together with $\\tau$', 'Verificăm rezultatul: testul Ljung--Box pe reziduurile ADF \\refLB; raportăm $k$ împreună cu $\\tau$')))

chart(T('Size and power of the Dickey--Fuller test', 'Mărimea și puterea testului Dickey--Fuller'), 'tsa_ch3_adf_power', 'TSA_ch3_dickey_fuller', [
    T('2\\,000 AR(1) paths $y_t = \\phi y_{t-1} + \\varepsilon_t$ for each $\\phi$; share of rejections at 5\\%; at $\\phi = 1$ this is the size',
      '2\\,000 de traiectorii AR(1) $y_t = \\phi y_{t-1} + \\varepsilon_t$ pentru fiecare $\\phi$; proporția respingerilor la 5\\%; la $\\phi = 1$ aceasta este mărimea testului')], h='0.55\\textheight')

interp(('the power curves', 'curbelor de putere'), [
    (T('Size: @{pw.100.c.1.0}\\%, @{pw.250.c.1.0}\\% and @{pw.500.c.1.0}\\% at $\\phi = 1$: the test holds its 5\\% level', 'Mărimea: @{pw.100.c.1.0}\\%, @{pw.250.c.1.0}\\% și @{pw.500.c.1.0}\\% la $\\phi = 1$: testul își respectă nivelul de 5\\%'),
     []),
    (T('Power at $\\phi = 0.95$: @{pw.100.c.0.95}\\% with $T = 100$, @{pw.250.c.0.95}\\% with $T = 250$, @{pw.500.c.0.95}\\% with $T = 500$', 'Puterea la $\\phi = 0{,}95$: @{pw.100.c.0.95}\\% cu $T = 100$, @{pw.250.c.0.95}\\% cu $T = 250$, @{pw.500.c.0.95}\\% cu $T = 500$'),
     [T('a quarterly series of 25 years ($T = 100$) almost never distinguishes $\\phi = 0.95$ from a unit root', 'o serie trimestrială de 25 de ani ($T = 100$) aproape niciodată nu deosebește $\\phi = 0{,}95$ de o rădăcină unitară'),
      T('what matters is the time span, not the frequency: daily data over two years do not help', 'contează lungimea perioadei, nu frecvența: datele zilnice pe doi ani nu ajută')]),
    T('An unnecessary trend costs power: at $T = 100$ and $\\phi = 0.9$, @{pw.100.c.0.9}\\% with a constant, @{pw.100.ct.0.9}\\% with constant and trend', 'Un trend inutil costă putere: la $T = 100$ și $\\phi = 0{,}9$, @{pw.100.c.0.9}\\% cu constantă, @{pw.100.ct.0.9}\\% cu constantă și trend')])

D.frame(T('Choosing the deterministic terms', 'Alegerea termenilor determiniști'), items(
    (T('The alternative must be a plausible description of the data', 'Ipoteza alternativă trebuie să fie o descriere plauzibilă a datelor'),
     [T('trending series (log GDP, log prices): constant and trend', 'serii cu trend (logaritmul PIB, logaritmii prețurilor): constantă și trend'),
      T('series that fluctuate around a level (interest rates, inflation, exchange rates, returns): constant', 'serii care fluctuează în jurul unui nivel (rate ale dobînzii, inflație, cursuri de schimb, randamente): constantă')]),
    (T('Omitting a needed trend: a TS series looks like a unit root; the test almost never rejects', 'Omiterea unui trend necesar: o serie staționară în jurul trendului pare să aibă rădăcină unitară; testul aproape niciodată nu respinge'),
     [T('adding an unneeded trend: lower power (previous chart)', 'adăugarea unui trend inutil: putere mai mică (graficul anterior)')]),
    T('Look at the plot first, then choose; \\refDFb\\ give joint $F$-type tests of the unit root and the trend', 'Ne uităm întîi la grafic, apoi alegem; \\refDFb\\ propun teste comune de tip $F$ pentru rădăcina unitară și trend')))

D.frame(T('How many differences: $I(1)$ or $I(2)$?', 'Cîte diferențieri: $I(1)$ sau $I(2)$?'), items(
    (T('\\refDP: test from the highest plausible order downwards', '\\refDP: testăm de la cel mai mare ordin plauzibil în jos'),
     [T('step 1: ADF on $\\Delta y_t$; if it does not reject, $y_t$ may be $I(2)$', 'pasul 1: ADF pe $\\Delta y_t$; dacă nu respinge, $y_t$ poate fi $I(2)$'),
      T('step 2: if it rejects, ADF on $y_t$; reject: $I(0)$; do not reject: $I(1)$', 'pasul 2: dacă respinge, ADF pe $y_t$; respingere: $I(0)$; nerespingere: $I(1)$')]),
    (T('Why downwards: if $y_t \\sim I(2)$, the test on $y_t$ assumes at most one unit root and is not valid', 'Motivul ordinii descendente: dacă $y_t \\sim I(2)$, testul pe $y_t$ presupune cel mult o rădăcină unitară și nu este valid'),
     []),
    (T('Example: Romanian consumer prices since 2005 (the table in Section 4)', 'Exemplu: prețurile de consum din România după 2005 (tabelul din secțiunea 4)'),
     [T('$\\ln P$: unit root; inflation $\\pi$: ADF p = @{ui.adfp}, KPSS @{ui.kpss}; change of inflation: clearly stationary', '$\\ln P$: rădăcină unitară; inflația $\\pi$: ADF p = @{ui.adfp}, KPSS @{ui.kpss}; variația inflației: clar staționară'),
      T('so $\\ln P$ is $I(1)$ or $I(2)$, depending on how persistent inflation is: Section 7 returns to this', 'deci $\\ln P$ este $I(1)$ sau $I(2)$, în funcție de cît de persistentă este inflația: secțiunea 7 revine asupra acestei întrebări')])))

D.recap(('The Dickey--Fuller test', 'testul Dickey--Fuller'), [
    T('$\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum_j\\delta_j\\Delta y_{t-j} + \\varepsilon_t$; $H_0$: $\\gamma = 0$; $\\tau$ against Dickey--Fuller critical values', '$\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum_j\\delta_j\\Delta y_{t-j} + \\varepsilon_t$; $H_0$: $\\gamma = 0$; $\\tau$ comparat cu valorile critice Dickey--Fuller'),
    T('Deterministic terms follow the plot; lags by AIC or BIC up to $12(T/100)^{1/4}$', 'Termenii determiniști urmează graficul; lagurile se aleg după AIC sau BIC, pînă la $12(T/100)^{1/4}$'),
    T('Low power near $\\phi = 1$: not rejecting is weak evidence for a unit root', 'Putere mică în apropierea lui $\\phi = 1$: nerespingerea este o dovadă slabă în favoarea rădăcinii unitare'),
    T('Order of integration: test from the top ($\\Delta y_t$ first)', 'Ordinul de integrare: testăm de sus în jos (întîi $\\Delta y_t$)')])

# =============================================================================
# 4. PP, KPSS
# =============================================================================
D.section('Phillips--Perron, KPSS and their joint use', 'Phillips--Perron, KPSS și folosirea lor împreună')

D.frame(T('The Phillips--Perron test', 'Testul Phillips--Perron'), items(
    (T('\\refPP: keep the simple DF regression (no lagged differences) and correct $\\tau$ instead', '\\refPP: păstrăm regresia DF simplă (fără diferențele trecute) și corectăm în schimb statistica $\\tau$'),
     [T('$Z_\\tau = \\sqrt{\\hat\\gamma_0/\\hat\\lambda^2}\\;\\tau - \\dfrac{(\\hat\\lambda^2 - \\hat\\gamma_0)\\,T\\,\\mathrm{SE}(\\hat\\gamma)}{2\\hat\\lambda\\,s}$', '$Z_\\tau = \\sqrt{\\hat\\gamma_0/\\hat\\lambda^2}\\;\\tau - \\dfrac{(\\hat\\lambda^2 - \\hat\\gamma_0)\\,T\\,\\mathrm{SE}(\\hat\\gamma)}{2\\hat\\lambda\\,s}$'),
      T('$\\hat\\gamma_0$: variance of the residuals; $s^2$: their OLS variance; $\\hat\\lambda^2$: their \\textbf{long-run variance}', '$\\hat\\gamma_0$: varianța reziduurilor; $s^2$: varianța lor OLS; $\\hat\\lambda^2$: \\textbf{varianța lor pe termen lung}')]),
    (T('Long-run variance (Newey--West, Bartlett weights): $\\hat\\lambda^2 = \\hat\\gamma_0 + 2\\sum_{j=1}^{L}(1 - \\frac{j}{L+1})\\hat\\gamma_j$', 'Varianța pe termen lung (Newey--West, ponderi Bartlett): $\\hat\\lambda^2 = \\hat\\gamma_0 + 2\\sum_{j=1}^{L}(1 - \\frac{j}{L+1})\\hat\\gamma_j$'),
     [T('$\\hat\\gamma_j$: autocovariances of the residuals; $L$: the number of autocovariances included (the bandwidth)', '$\\hat\\gamma_j$: autocovarianțele reziduurilor; $L$: numărul de autocovarianțe incluse (lățimea de bandă)'),
      T('if the residuals are white noise, $\\hat\\lambda^2 = \\hat\\gamma_0$ and $Z_\\tau = \\tau$: the correction only matters when they are autocorrelated', 'dacă reziduurile sînt zgomot alb, $\\hat\\lambda^2 = \\hat\\gamma_0$ și $Z_\\tau = \\tau$: corecția contează doar cînd ele sînt autocorelate')]),
    (T('Same $H_0$ and the same critical values as ADF; robust to heteroskedasticity; no lag choice, but a bandwidth $L$', 'Aceeași $H_0$ și aceleași valori critice ca ADF; robust la heteroscedasticitate; nu alegem laguri, ci o lățime de bandă $L$'),
     [T('weak point: large size distortions with a negative MA component \\refSchwert', 'punctul slab: distorsiuni mari ale mărimii cînd există o componentă MA negativă \\refSchwert')])))

D.frame(T('The KPSS test: stationarity as the null', 'Testul KPSS: staționaritatea ca ipoteză nulă'), items(
    (T('\\refKPSS\\ (KPSS: Kwiatkowski--Phillips--Schmidt--Shin): $y_t = \\xi t + r_t + \\varepsilon_t$, $r_t = r_{t-1} + u_t$, $u_t \\sim \\mathrm{WN}(0, \\sigma_u^2)$',
       '\\refKPSS\\ (KPSS: Kwiatkowski--Phillips--Schmidt--Shin): $y_t = \\xi t + r_t + \\varepsilon_t$, $r_t = r_{t-1} + u_t$, $u_t \\sim \\mathrm{WN}(0, \\sigma_u^2)$'),
     [T('$\\xi t$: a deterministic trend; $r_t$: a random walk, the moving level; $\\varepsilon_t$: stationary noise', '$\\xi t$: un trend determinist; $r_t$: un mers aleator, adică nivelul care se deplasează; $\\varepsilon_t$: zgomot staționar'),
      T('$H_0$: $\\sigma_u^2 = 0$ (the random walk is a constant: $y_t$ stationary around a level or a trend); $H_1$: $\\sigma_u^2 > 0$, a unit root',
        '$H_0$: $\\sigma_u^2 = 0$ (mersul aleator este o constantă: $y_t$ staționar în jurul unui nivel sau al unui trend); $H_1$: $\\sigma_u^2 > 0$, o rădăcină unitară')]),
    (T('Statistic: regress $y_t$ on a constant (and a trend); residuals $e_t$; partial sums $S_t = e_1 + \\dots + e_t$', 'Statistica: regresia lui $y_t$ pe o constantă (și un trend); reziduurile $e_t$; sumele parțiale $S_t = e_1 + \\dots + e_t$'),
     [T('$\\eta = \\dfrac{1}{T^2\\hat\\lambda^2}\\sum_{t=1}^{T} S_t^2$, with $\\hat\\lambda^2$ the long-run variance of $e_t$', '$\\eta = \\dfrac{1}{T^2\\hat\\lambda^2}\\sum_{t=1}^{T} S_t^2$, unde $\\hat\\lambda^2$ este varianța pe termen lung a lui $e_t$'),
      T('reject stationarity for \\textbf{large} $\\eta$: 5\\% critical values 0.463 (level) and 0.146 (trend); 1\\%: 0.739 and 0.216', 'respingem staționaritatea pentru valori \\textbf{mari} ale lui $\\eta$: valorile critice de 5\\% sînt 0,463 (nivel) și 0,146 (trend); la 1\\%: 0,739 și 0,216')]),
    T('A test of stationarity, not of a unit root: it reverses the burden of proof', 'Un test al staționarității, nu al rădăcinii unitare: inversează sarcina probei')))

chart(T('KPSS: why partial sums', 'KPSS: rolul sumelor parțiale'), 'tsa_ch3_kpss_sums', 'TSA_ch3_unit_root_tests', [
    T('A stationary AR(1) ($\\phi = 0.5$) and a random walk, $T = 250$: the demeaned series and their partial sums $S_t$',
      'Un AR(1) staționar ($\\phi = 0{,}5$) și un mers aleator, $T = 250$: seriile centrate și sumele lor parțiale $S_t$'),
    T('KPSS (level): @{kp.ar} for the AR(1), @{kp.rw} for the random walk; 5\\% critical value 0.463', 'KPSS (nivel): @{kp.ar} pentru AR(1), @{kp.rw} pentru mersul aleator; valoarea critică de 5\\% este 0,463')], h='0.52\\textheight')

interp(('the partial sums', 'sumelor parțiale'), [
    (T('Stationary series: deviations cancel out, $S_t$ stays close to 0', 'Seria staționară: abaterile se compensează, $S_t$ rămîne aproape de 0'),
     [T('$\\sum S_t^2$ grows like $T^2$, so $\\eta$ stays bounded', '$\\sum S_t^2$ crește ca $T^2$, deci $\\eta$ rămîne mărginit')]),
    (T('Random walk: long runs on one side of the mean, $S_t$ makes large excursions', 'Mersul aleator: perioade lungi de o parte a mediei, $S_t$ se abate mult de la 0'),
     [T('$\\sum S_t^2$ grows like $T^4$: $\\eta$ grows with $T$ and the test rejects', '$\\sum S_t^2$ crește ca $T^4$: $\\eta$ crește cu $T$, iar testul respinge')]),
    T('The same idea as the CUSUM charts of quality control: cumulated deviations reveal a drifting level', 'Aceeași idee ca în graficele CUSUM din controlul calității: abaterile cumulate dezvăluie un nivel care se deplasează')])

D.frame(T('ADF and KPSS together', 'ADF și KPSS împreună'), table(
    'lll', T('& \\textbf{KPSS does not reject} & \\textbf{KPSS rejects}', '& \\textbf{KPSS nu respinge} & \\textbf{KPSS respinge}'),
    [T('\\textbf{ADF rejects} & $I(0)$: both agree & conflict: break, long memory, outliers?',
       '\\textbf{ADF respinge} & $I(0)$: ambele concordă & conflict: ruptură, memorie lungă, valori extreme?'),
     T('\\textbf{ADF does not reject} & inconclusive: the data cannot decide & $I(1)$: both agree',
       '\\textbf{ADF nu respinge} & neconcludent: datele nu pot decide & $I(1)$: ambele concordă')], size='footnotesize') + items(
    T('The two tests have opposite null hypotheses: agreement is stronger evidence than either test alone', 'Cele două teste au ipoteze nule opuse: concordanța lor este o dovadă mai puternică decît oricare test singur'),
    (T('Inconclusive is common with short samples: low power on both sides', 'Situația neconcludentă este frecventă în eșantioane scurte: putere mică de ambele părți'),
     [T('then decide by the purpose: for forecasting, $d = 1$ is the cautious choice (wider intervals)', 'atunci decidem după scop: pentru prognoză, $d = 1$ este alegerea prudentă (intervale mai largi)')]),
    T('Conflict: look for breaks (Section 5) or fractional integration (Chapter 8)', 'Conflict: căutăm rupturi (secțiunea 5) sau integrare fracționară (Capitolul 8)')))

D.frame(T('Unit-root tests on real series', 'Teste de rădăcină unitară pe serii reale'), table(
    'llrrrrrl', T('\\textbf{Series} & \\textbf{det.} & $T$ & \\textbf{ADF} (p) & $k$ & \\textbf{PP} & \\textbf{KPSS} & \\textbf{ADF+KPSS}',
                  '\\textbf{Seria} & \\textbf{det.} & $T$ & \\textbf{ADF} (p) & $k$ & \\textbf{PP} & \\textbf{KPSS} & \\textbf{ADF+KPSS}'),
    UROWS, size='scriptsize') + items(
    T('det.: deterministic terms (c: constant; t: trend); $k$: ADF lags (AIC); 5\\% critical values: ADF and PP about $-2.86$ (c), $-3.41$ (c, t); KPSS 0.463 (c), 0.146 (c, t)',
      'det.: termenii determiniști (c: constantă; t: trend); $k$: lagurile ADF (AIC); valori critice de 5\\%: ADF și PP circa $-2{,}86$ (c), $-3{,}41$ (c, t); KPSS 0,463 (c), 0,146 (c, t)')) + ql('TSA_ch3_unit_root_tests'), size='scriptsize')

interp(('the unit-root table', 'tabelului testelor'), [
    (T('Log prices, the exchange rate, log GDP and log HICP: $I(1)$; their differences: $I(0)$', 'Logaritmii prețurilor, cursul de schimb, logaritmul PIB și al IAPC: $I(1)$; diferențele lor: $I(0)$'),
     [T('Romanian GDP: ADF p = @{ug.adfp} and KPSS @{ug.kpss} $> 0.146$: a unit root, as \\refNP\\ found for the US', 'PIB-ul României: ADF p = @{ug.adfp} și KPSS @{ug.kpss} $> 0{,}146$: rădăcină unitară, cum au găsit \\refNP\\ pentru SUA')]),
    (T('Romanian 12-month inflation: neither test rejects (KPSS @{ui.kpss} $< 0.463$): inconclusive', 'Inflația anuală din România: niciun test nu respinge (KPSS @{ui.kpss} $< 0{,}463$): neconcludent'),
     [T('a very persistent series, possibly with breaks; Sections 5 and 7', 'o serie foarte persistentă, posibil cu rupturi; secțiunile 5 și 7')]),
    (T('Nile: both reject: a conflict', 'Nilul: ambele resping: un conflict'),
     [T('Chapter 1 showed a level shift in 1898: the next section explains the conflict', 'Capitolul 1 a arătat o schimbare de nivel în 1898: secțiunea următoare explică acest conflict')]),
    T('Daily returns: PP gives far more negative values than ADF (BET: $@{ub.ppr}$ against $@{ub.adfr}$), but both reject; the verdict is the same', 'Randamentele zilnice: PP dă valori mult mai negative decît ADF (BET: $@{ub.ppr}$ față de $@{ub.adfr}$), dar ambele resping; verdictul este același')])

D.recap(('Phillips--Perron and KPSS', 'Phillips--Perron și KPSS'), [
    T('PP: DF regression plus a long-run-variance correction; same $H_0$ and critical values as ADF', 'PP: regresia DF plus o corecție prin varianța pe termen lung; aceeași $H_0$ și aceleași valori critice ca ADF'),
    T('KPSS: $H_0$ stationarity; large partial sums reject it', 'KPSS: $H_0$ este staționaritatea; sumele parțiale mari o resping'),
    T('Read ADF and KPSS together: agree, inconclusive, or conflict', 'Interpretăm ADF și KPSS împreună: concordanță, rezultat neconcludent sau conflict')])

# =============================================================================
# 5. RUPTURI STRUCTURALE
# =============================================================================
D.section('Structural breaks', 'Rupturi structurale')

D.frame(T('Perron (1989): a break can look like a unit root', 'Perron (1989): o ruptură poate semăna cu o rădăcină unitară'), items(
    (T('\\refPerron: a stationary series around a mean or a trend that \\textbf{shifts once} looks persistent', '\\refPerron: o serie staționară în jurul unei medii sau al unui trend care \\textbf{se schimbă o dată} pare persistentă'),
     [T('the DF regression explains the shift by $\\hat\\phi$ close to 1; the test loses power', 'regresia DF explică schimbarea printr-un $\\hat\\phi$ apropiat de 1; testul își pierde puterea')]),
    (T('He re-examined Nelson and Plosser with two known breaks: the 1929 crash (level) and the 1973 oil shock (slope)', 'A reexaminat seriile Nelson și Plosser cu două rupturi cunoscute: crahul din 1929 (nivel) și șocul petrolier din 1973 (pantă)'),
     [T('with dummy variables for the break, the unit root was rejected for most series', 'cu variabile dummy pentru ruptură, rădăcina unitară a fost respinsă pentru majoritatea seriilor'),
      T('reading: most shocks are transitory; only a few rare events are permanent', 'interpretarea: majoritatea șocurilor sînt temporare; doar cîteva evenimente rare sînt permanente')]),
    T('The critique: the break date was chosen after looking at the data', 'Critica: data rupturii a fost aleasă după ce s-au văzut datele')))

chart(T('A level shift misleads the ADF test', 'O schimbare de nivel înșală testul ADF'), 'tsa_ch3_breaks_sim', 'TSA_ch3_structural_breaks', [
    T('Left: AR(1) noise ($\\phi = 0.6$) around a mean that jumps from 0 to 5 at $t = 100$; right: the $t$-statistic on $y_{t-1}$ for every candidate break date, with a level dummy',
      'Stînga: zgomot AR(1) ($\\phi = 0{,}6$) în jurul unei medii care sare de la 0 la 5 la $t = 100$; dreapta: statistica $t$ a lui $y_{t-1}$ pentru fiecare dată posibilă a rupturii, cu o variabilă dummy de nivel')], h='0.54\\textheight')

interp(('the simulated break', 'rupturii simulate'), [
    (T('ADF with a constant: p = @{bs.adfp}: the stationary series ``has a unit root\'\'', 'ADF cu constantă: p = @{bs.adfp}: seria staționară „are rădăcină unitară”'),
     [T('over 2\\,000 such series, the DF test rejects in @{bs.rej}\\% of cases; without the break: @{bs.rej0}\\%', 'pe 2\\,000 de asemenea serii, testul DF respinge în @{bs.rej}\\% din cazuri; fără ruptură: @{bs.rej0}\\%')]),
    (T('With the break modelled, the statistic falls to $@{bs.za}$, far below the 5\\% value $@{bs.zacv}$', 'Cînd ruptura este modelată, statistica scade la $@{bs.za}$, mult sub valoarea de 5\\%, $@{bs.zacv}$'),
     [T('the minimum is reached at the true date $t = @{bs.date}$', 'minimul este atins la data adevărată, $t = @{bs.date}$')])])

D.frame(T('Zivot and Andrews (1992): an unknown break date', 'Zivot și Andrews (1992): data rupturii necunoscută'), items(
    (T('\\refZA: let the data choose the break date $T_B$', '\\refZA: lăsăm datele să aleagă data rupturii $T_B$'),
     [T('for each $T_B$ in the central 70\\% of the sample, run the ADF regression with a break dummy:', 'pentru fiecare $T_B$ din cele 70\\% centrale ale eșantionului, estimăm regresia ADF cu o variabilă dummy pentru ruptură:'),
      T('$\\Delta y_t = c + \\theta DU_t + bt + \\gamma y_{t-1} + \\sum_j\\delta_j\\Delta y_{t-j} + \\varepsilon_t$, $DU_t = 1$ for $t > T_B$', '$\\Delta y_t = c + \\theta DU_t + bt + \\gamma y_{t-1} + \\sum_j\\delta_j\\Delta y_{t-j} + \\varepsilon_t$, $DU_t = 1$ pentru $t > T_B$'),
      T('$DU_t$: a dummy variable, 0 before and 1 after the break; $\\theta$: the size of the level shift', '$DU_t$: o variabilă dummy, 0 înainte și 1 după ruptură; $\\theta$: mărimea saltului de nivel'),
      T('the statistic is the \\textbf{minimum} $t$-ratio of $\\gamma$ over all dates', 'statistica este raportul $t$ \\textbf{minim} al lui $\\gamma$ pe toate datele')]),
    (T('Three versions: break in the level, in the slope, in both', 'Trei variante: ruptură în nivel, în pantă sau în ambele'),
     [T('critical values more negative than ADF: about $-4.81$ (level) and $-5.07$ (both) at 5\\%', 'valori critice mai negative decît la ADF: circa $-4{,}81$ (nivel) și $-5{,}07$ (ambele) la 5\\%')]),
    (T('$H_0$: unit root without a break; $H_1$: stationary with one break', '$H_0$: rădăcină unitară fără ruptură; $H_1$: staționar cu o ruptură'),
     [T('several breaks: \\refBaiP; a rejection does not prove that the break is the right story', 'mai multe rupturi: \\refBaiP; o respingere nu demonstrează că ruptura este explicația corectă')])))

chart(T('Zivot--Andrews on two real series', 'Zivot--Andrews pe două serii reale'), 'tsa_ch3_breaks_real', 'TSA_ch3_structural_breaks', [
    T('Left: the Nile flow (statsmodels data set, 1871--1970); right: 100 ln EUR/RON, month-end BNR reference rate', 'Stînga: debitul Nilului (setul de date statsmodels, 1871--1970); dreapta: 100 ln EUR/RON, cursul de referință BNR de la sfîrșitul lunii'),
    T('Dotted line: the break date chosen by the test (break in the level)', 'Linia punctată: data rupturii aleasă de test (ruptură în nivel)')], h='0.54\\textheight')

interp(('the two breaks', 'celor două rupturi'), [
    (T('Nile: ZA = $@{bn.za}$, break in @{bn.y}, the mean falls from @{bn.m0} to @{bn.m1}', 'Nilul: ZA = $@{bn.za}$, ruptură în @{bn.y}, media scade de la @{bn.m0} la @{bn.m1}'),
     [T('the date matches the history: the first Aswan dam (built 1898--1902) and a drier climate; the ADF--KPSS conflict is explained', 'data corespunde istoriei: primul baraj de la Aswan (construit în 1898--1902) și un climat mai secetos; conflictul ADF--KPSS este explicat')]),
    (T('EUR/RON (month-end): ADF p = @{bf.adfp}, but ZA = $@{bf.za}$, break in @{bf.d}: the 2008 depreciation (mean @{bf.m0} lei before, @{bf.m1} after)', 'EUR/RON (sfîrșit de lună): ADF p = @{bf.adfp}, dar ZA = $@{bf.za}$, ruptură în @{bf.d}: deprecierea din 2008 (media @{bf.m0} lei înainte, @{bf.m1} după)'),
     [T('a managed exchange rate: long calm periods and rare adjustments, not a pure random walk', 'un curs în regim de managed float: perioade lungi de stabilitate și ajustări rare, nu un mers aleator pur'),
      T('but after 2008 the rate still drifts up: ``stationary around one break\'\' is also a simplification', 'dar după 2008 cursul continuă să urce: „staționar în jurul unei rupturi” este și el o simplificare')])])

D.recap(('Structural breaks', 'rupturi structurale'), [
    T('A one-time shift in the mean or trend biases unit-root tests towards non-rejection', 'O schimbare unică a mediei sau a trendului face ca testele de rădăcină unitară să respingă mai rar'),
    T('Zivot--Andrews searches the break date; its critical values are more negative', 'Zivot--Andrews caută data rupturii; valorile lui critice sînt mai negative'),
    T('Unit root or break: check the dates against known events', 'Rădăcină unitară sau ruptură: comparăm datele cu evenimente cunoscute')])

# =============================================================================
# 6. ARIMA
# =============================================================================
D.section('ARIMA$(p,d,q)$ models', 'Modele ARIMA$(p,d,q)$')

D.frame(T('The ARIMA model', 'Modelul ARIMA'), items(
    (T('\\textbf{Definition}: $y_t \\sim$ ARIMA$(p,d,q)$ (autoregressive integrated moving average) if $\\Delta^d y_t$ is a stationary, invertible ARMA$(p,q)$:',
       '\\textbf{Definiție}: $y_t \\sim$ ARIMA$(p,d,q)$ (autoregressive integrated moving average, model autoregresiv integrat cu medie mobilă) dacă $\\Delta^d y_t$ este un ARMA$(p,q)$ staționar și inversabil:'),
     [T('$\\phi(L)\\,(1 - L)^d\\,y_t = c + \\theta(L)\\,\\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$', '$\\phi(L)\\,(1 - L)^d\\,y_t = c + \\theta(L)\\,\\varepsilon_t$, $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$'),
      T('$p$: AR order; $d$: number of differences (unit roots); $q$: MA order; $\\phi(L)$, $\\theta(L)$ as in Chapter 2', '$p$: ordinul AR; $d$: numărul de diferențieri (rădăcini unitare); $q$: ordinul MA; $\\phi(L)$, $\\theta(L)$ ca în Capitolul 2')]),
    (T('The AR polynomial of the levels is $\\phi(z)(1 - z)^d$: $d$ roots exactly on the unit circle', 'Polinomul AR al nivelurilor este $\\phi(z)(1 - z)^d$: $d$ rădăcini exact pe cercul unitate'),
     [T('``integrated\'\': $y_t$ is a $d$-fold cumulative sum of an ARMA process', '„integrat”: $y_t$ este o sumă cumulată de $d$ ori a unui proces ARMA')]),
    T('\\refBJ\\ made the model popular; in practice $d \\le 2$, and $d = 1$ most often', '\\refBJ\\ au popularizat modelul; în practică $d \\le 2$, cel mai des $d = 1$')))

D.frame(T('Special cases and the constant', 'Cazuri particulare și constanta'), table(
    'll', T('\\textbf{Model} & \\textbf{Equation and meaning}', '\\textbf{Model} & \\textbf{Ecuația și semnificația}'),
    [T('ARIMA(0,1,0) & $\\Delta y_t = c + \\varepsilon_t$: random walk (with drift $c$)', 'ARIMA(0,1,0) & $\\Delta y_t = c + \\varepsilon_t$: mers aleator (cu deriva $c$)'),
     T('ARIMA(1,1,0) & $\\Delta y_t = c + \\phi\\Delta y_{t-1} + \\varepsilon_t$: growth with momentum', 'ARIMA(1,1,0) & $\\Delta y_t = c + \\phi\\Delta y_{t-1} + \\varepsilon_t$: creștere cu inerție'),
     T('ARIMA(0,1,1) & $\\Delta y_t = \\varepsilon_t + \\theta\\varepsilon_{t-1}$: simple exponential smoothing with $\\alpha = 1 + \\theta$ (Chapter 0)', 'ARIMA(0,1,1) & $\\Delta y_t = \\varepsilon_t + \\theta\\varepsilon_{t-1}$: netezirea exponențială simplă cu $\\alpha = 1 + \\theta$ (Capitolul 0)'),
     T('ARIMA(0,2,2) & $\\Delta^2 y_t = \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\theta_2\\varepsilon_{t-2}$: the model behind Holt\'s linear method', 'ARIMA(0,2,2) & $\\Delta^2 y_t = \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\theta_2\\varepsilon_{t-2}$: modelul din spatele metodei liniare Holt')],
    size='scriptsize') + items(
    (T('The constant $c$ changes the long-run forecast \\refFPP, Ch.~9.7', 'Constanta $c$ schimbă prognoza pe termen lung \\refFPP, cap.~9.7'),
     [T('$d = 0$: forecasts go to the mean; $d = 1$: $c \\ne 0$ gives a straight line (drift), $c = 0$ a flat line', '$d = 0$: prognozele tind spre medie; $d = 1$: $c \\ne 0$ dă o dreaptă (derivă), $c = 0$ o linie orizontală'),
      T('$d = 2$: $c \\ne 0$ gives a quadratic trend, rarely sensible; use $c = 0$', '$d = 2$: $c \\ne 0$ dă un trend pătratic, rareori rezonabil; folosim $c = 0$')])))

D.frame(T('The Box--Jenkins cycle with unit roots', 'Ciclul Box--Jenkins cu rădăcini unitare'), items(
    T('\\textbf{1. Transform}: logarithm or Box--Cox if the variance grows with the level (Chapter 1)', '\\textbf{1. Transformarea}: logaritm sau Box--Cox dacă varianța crește cu nivelul (Capitolul 1)'),
    T('\\textbf{2. Choose $d$}: plot, ACF, ADF and KPSS; test $\\Delta y_t$ first; stop at the first stationary difference', '\\textbf{2. Alegem $d$}: grafic, ACF, ADF și KPSS; testăm întîi $\\Delta y_t$; ne oprim la prima diferență staționară'),
    T('\\textbf{3. Choose $p$, $q$}: ACF and PACF of $\\Delta^d y_t$ (Chapter 2), then compare candidates by AICc on the same $d$', '\\textbf{3. Alegem $p$, $q$}: ACF și PACF ale lui $\\Delta^d y_t$ (Capitolul 2), apoi comparăm candidații după AICc, cu același $d$'),
    T('\\textbf{4. Estimate} by maximum likelihood; decide on the constant', '\\textbf{4. Estimăm} prin verosimilitate maximă; decidem asupra constantei'),
    T('\\textbf{5. Check} the residuals: ACF, Ljung--Box with $m - p - q$ degrees of freedom, outliers', '\\textbf{5. Verificăm} reziduurile: ACF, Ljung--Box cu $m - p - q$ grade de libertate, valori extreme'),
    T('\\textbf{6. Forecast} the levels with intervals; go back to step 2 or 3 if the checks fail', '\\textbf{6. Prognozăm} nivelurile cu intervale; revenim la pasul 2 sau 3 dacă verificările eșuează')))

chart(T('Romanian GDP: identification', 'PIB-ul României: identificarea'), 'tsa_ch3_gdp_ident', 'TSA_ch3_arima_gdp', [
    T('Real GDP, seasonally adjusted (Eurostat), @{gi.q0}--@{gi.q1}, $T = @{gi.n}$; top: $100\\ln Y_t$ and the quarterly growth; bottom: ACF and PACF of the growth',
      'PIB-ul real ajustat sezonier (Eurostat), @{gi.q0}--@{gi.q1}, $T = @{gi.n}$; sus: $100\\ln Y_t$ și creșterea trimestrială; jos: ACF și PACF ale creșterii')], h='0.66\\textheight')

interp(('the identification plots', 'graficelor de identificare'), [
    (T('$d = 1$: the level is $I(1)$ (Section 4); the growth rate is stationary, mean @{gi.mean}\\% per quarter (about @{gi.mean4}\\% per year)', '$d = 1$: nivelul este $I(1)$ (secțiunea 4); rata de creștere este staționară, cu media @{gi.mean}\\% pe trimestru (circa @{gi.mean4}\\% pe an)'),
     [T('two large falls: @{gi.w1}\\% in @{gi.w1d} (global financial crisis) and @{gi.w2}\\% in @{gi.w2d} (pandemic)', 'două scăderi mari: @{gi.w1}\\% în @{gi.w1d} (criza financiară globală) și @{gi.w2}\\% în @{gi.w2d} (pandemia)')]),
    (T('$p$ and $q$: no ACF or PACF value outside $\\pm @{gi.band}$; $\\hat\\rho(1) = @{gi.r1}$, $\\hat\\rho(2) = @{gi.r2}$', '$p$ și $q$: nicio valoare ACF sau PACF în afara benzii $\\pm @{gi.band}$; $\\hat\\rho(1) = @{gi.r1}$, $\\hat\\rho(2) = @{gi.r2}$'),
     [T('Ljung--Box $Q^*(8) = @{gi.q}$ (p = @{gi.qp}): the growth rate looks like white noise around its mean', 'Ljung--Box $Q^*(8) = @{gi.q}$ (p = @{gi.qp}): rata de creștere seamănă cu un zgomot alb în jurul mediei')]),
    T('The plots suggest ARIMA(0,1,0) with drift; the next slide checks the alternatives', 'Graficele sugerează ARIMA(0,1,0) cu derivă; slide-ul următor verifică alternativele')])

D.frame(T('Model choice by AICc', 'Alegerea modelului după AICc'), table(
    'lccc', T('& $q = 0$ & $q = 1$ & $q = 2$', '& $q = 0$ & $q = 1$ & $q = 2$'),
    [f'$p = {p}$ & ' + ' & '.join(f'@{{gm.{p}{q}}}' + ('$^\\dagger$' if not gm[f'{p}{q}']['converged'] else '') for q in range(3)) for p in range(3)],
    size='footnotesize') + items(
    T('ARIMA$(p,1,q)$ with drift for $100\\ln Y_t$, estimated by maximum likelihood; AICc: AIC with the small-sample correction (Chapter 2); $^\\dagger$: the optimiser did not converge',
      'ARIMA$(p,1,q)$ cu derivă pentru $100\\ln Y_t$, estimat prin verosimilitate maximă; AICc: AIC cu corecția pentru eșantioane mici (Capitolul 2); $^\\dagger$: optimizarea nu a convers'),
    (T('Lowest AICc: @{gm.best}; the next model is @{gm.gap} points worse', 'Cel mai mic AICc: @{gm.best}; următorul model este cu @{gm.gap} puncte mai slab'),
     [T('differences below 2 points are not meaningful: keep the simplest model', 'diferențele sub 2 puncte nu sînt relevante: păstrăm cel mai simplu model')]),
    T('Estimate: drift $\\hat c = @{gd.drift}$ (SE @{gd.dse}, $t = @{gd.dt}$), $\\hat\\sigma = @{gd.sig}$: $100\\,\\Delta\\ln Y_t = @{gd.drift} + \\hat\\varepsilon_t$',
      'Estimarea: deriva $\\hat c = @{gd.drift}$ (SE @{gd.dse}, $t = @{gd.dt}$), $\\hat\\sigma = @{gd.sig}$: $100\\,\\Delta\\ln Y_t = @{gd.drift} + \\hat\\varepsilon_t$')) + ql('TSA_ch3_arima_gdp'), size='small')

chart(T('Romanian GDP: residual checks', 'PIB-ul României: verificarea reziduurilor'), 'tsa_ch3_gdp_diag', 'TSA_ch3_arima_gdp', [
    T('Residuals of @{gd.order} with drift: time plot, ACF, Ljung--Box p-values for $m = 2, \\dots, 16$, histogram with a Normal density',
      'Reziduurile modelului @{gd.order} cu derivă: graficul în timp, ACF, p-value-urile Ljung--Box pentru $m = 2, \\dots, 16$, histograma cu densitatea Normală')], h='0.66\\textheight')

interp(('the residual checks', 'verificării reziduurilor'), [
    (T('No autocorrelation left: $Q^*(8) = @{gd.q8}$ (p = @{gd.q8p}), $Q^*(12)$ p = @{gd.q12p}; all p-values above 5\\%', 'Nu a rămas autocorelație: $Q^*(8) = @{gd.q8}$ (p = @{gd.q8p}), $Q^*(12)$ p = @{gd.q12p}; toate p-value-urile sînt peste 5\\%'),
     []),
    (T('Not Normal: kurtosis @{gd.k}, Jarque--Bera @{gd.jb}; the largest residual is $@{gd.min}$ in @{gd.mind}', 'Nu sînt normale: kurtosis-ul @{gd.k}, Jarque--Bera @{gd.jb}; cel mai mare reziduu este $@{gd.min}$ în @{gd.mind}'),
     [T('two crises dominate the tails; the Normal intervals of the next slides are too narrow in a crisis and slightly too wide in calm times', 'două crize domină cozile; intervalele normale din slide-urile următoare sînt prea înguste într-o criză și ușor prea largi în perioadele calme'),
      T('options: dummy variables for 2009 and 2020, or a bootstrap of the residuals for the intervals', 'opțiuni: variabile dummy pentru 2009 și 2020 sau un bootstrap al reziduurilor pentru intervale')])])

D.frame(T('Forecasting with ARIMA', 'Prognoza cu modele ARIMA'), items(
    (T('\\textbf{Point forecasts}: write the model for the levels and iterate it, with $\\hat\\varepsilon_{T+j} = 0$ for $j \\ge 1$', '\\textbf{Prognozele punctuale}: scriem modelul pentru niveluri și îl aplicăm recursiv, cu $\\hat\\varepsilon_{T+j} = 0$ pentru $j \\ge 1$'),
     [T('ARIMA(1,1,0): $y_t = (1 + \\phi)y_{t-1} - \\phi y_{t-2} + \\varepsilon_t$', 'ARIMA(1,1,0): $y_t = (1 + \\phi)y_{t-1} - \\phi y_{t-2} + \\varepsilon_t$')]),
    (T('\\textbf{Forecast error}: $y_{T+h} - \\hat y_{T+h} = \\sum_{j=0}^{h-1}\\psi_j\\varepsilon_{T+h-j}$, with $\\psi(z) = \\theta(z)/[\\phi(z)(1 - z)^d]$', '\\textbf{Eroarea de prognoză}: $y_{T+h} - \\hat y_{T+h} = \\sum_{j=0}^{h-1}\\psi_j\\varepsilon_{T+h-j}$, cu $\\psi(z) = \\theta(z)/[\\phi(z)(1 - z)^d]$'),
     [T('variance $\\sigma^2\\sum_{j=0}^{h-1}\\psi_j^2$; 95\\% interval $\\hat y_{T+h} \\pm 1.96\\,\\sigma\\sqrt{\\sum_{j<h}\\psi_j^2}$', 'varianța $\\sigma^2\\sum_{j=0}^{h-1}\\psi_j^2$; intervalul de 95\\%: $\\hat y_{T+h} \\pm 1{,}96\\,\\sigma\\sqrt{\\sum_{j<h}\\psi_j^2}$'),
      T('for $d \\ge 1$ the $\\psi_j$ do not decay to 0: the variance grows without bound', 'pentru $d \\ge 1$, $\\psi_j$ nu tind la 0: varianța crește nelimitat')]),
    (T('Two closed forms', 'Două formule explicite'),
     [T('random walk: $\\psi_j = 1$, variance $h\\sigma^2$: the interval grows like $\\sqrt{h}$', 'mersul aleator: $\\psi_j = 1$, varianța $h\\sigma^2$: intervalul crește ca $\\sqrt{h}$'),
      T('ARIMA(0,1,1): $\\psi_j = 1 + \\theta$ for $j \\ge 1$, variance $\\sigma^2[1 + (h - 1)(1 + \\theta)^2]$', 'ARIMA(0,1,1): $\\psi_j = 1 + \\theta$ pentru $j \\ge 1$, varianța $\\sigma^2[1 + (h - 1)(1 + \\theta)^2]$')])))

D.frame(T('Worked example: ARIMA(1,1,0) forecasts', 'Exemplu rezolvat: prognoze ARIMA(1,1,0)'), items(
    T('$\\Delta y_t = 0.6\\,\\Delta y_{t-1} + \\varepsilon_t$, $\\sigma = 2$; last data: $y_T = 108$, $\\Delta y_T = 5$', '$\\Delta y_t = 0{,}6\\,\\Delta y_{t-1} + \\varepsilon_t$, $\\sigma = 2$; ultimele date: $y_T = 108$, $\\Delta y_T = 5$'),
    (T('\\textbf{Point forecasts}: forecast the change, then add it', '\\textbf{Prognozele punctuale}: prognozăm variația, apoi o adunăm'),
     [T('$\\widehat{\\Delta y}_{T+1} = 0.6 \\cdot 5 = 3$, $\\hat y_{T+1} = @{hb.f0}$; $\\widehat{\\Delta y}_{T+2} = 1.8$, $\\hat y_{T+2} = @{hb.f1}$; $\\hat y_{T+3} = @{hb.f2}$', '$\\widehat{\\Delta y}_{T+1} = 0{,}6 \\cdot 5 = 3$, $\\hat y_{T+1} = @{hb.f0}$; $\\widehat{\\Delta y}_{T+2} = 1{,}8$, $\\hat y_{T+2} = @{hb.f1}$; $\\hat y_{T+3} = @{hb.f2}$')]),
    (T('\\textbf{$\\psi$ weights}: $\\psi_0 = 1$, $\\psi_1 = 1 + \\phi = @{hb.psi1}$, $\\psi_2 = 1 + \\phi + \\phi^2 = @{hb.psi2}$', '\\textbf{Ponderile $\\psi$}: $\\psi_0 = 1$, $\\psi_1 = 1 + \\phi = @{hb.psi1}$, $\\psi_2 = 1 + \\phi + \\phi^2 = @{hb.psi2}$'),
     [T('variances: $4$, $4(1 + @{hb.psi1}^2) = @{hb.v1}$, $4(1 + @{hb.psi1}^2 + @{hb.psi2}^2) = @{hb.v2}$', 'varianțele: $4$, $4(1 + @{hb.psi1}^2) = @{hb.v1}$, $4(1 + @{hb.psi1}^2 + @{hb.psi2}^2) = @{hb.v2}$'),
      T('95\\% half-widths: @{hb.h0}, @{hb.h1}, @{hb.h2}: the interval almost triples in three steps', 'semilățimile de 95\\%: @{hb.h0}, @{hb.h1}, @{hb.h2}: intervalul aproape se triplează în trei pași')]),
    T('The momentum $\\phi = 0.6$ makes the uncertainty grow faster than for a random walk ($\\sqrt{3} = 1.73$ times in three steps)', 'Inerția $\\phi = 0{,}6$ face ca incertitudinea să crească mai repede decît la un mers aleator ($\\sqrt{3} = 1{,}73$ ori în trei pași)')))

chart(T('How fast the intervals grow', 'Viteza de creștere a intervalelor'), 'tsa_ch3_interval_width', 'TSA_ch3_forecast_intervals', [
    T('Half-width of the 95\\% forecast interval, in units of $\\sigma$, for horizons 1 to 20, from the $\\psi$ weights of five models',
      'Semilățimea intervalului de prognoză de 95\\%, în unități $\\sigma$, pentru orizonturile 1--20, din ponderile $\\psi$ ale a cinci modele')], h='0.56\\textheight')

interp(('the interval widths', 'lățimii intervalelor'), [
    (T('$d = 0$, AR(1) with $\\phi = 0.8$: the width levels off at $1.96\\sigma/\\sqrt{1 - \\phi^2}$ (@{w0.20}$\\sigma$ at $h = 20$)', '$d = 0$, AR(1) cu $\\phi = 0{,}8$: lățimea se stabilizează la $1{,}96\\sigma/\\sqrt{1 - \\phi^2}$ (@{w0.20}$\\sigma$ la $h = 20$)'),
     []),
    (T('$d = 1$: growth like $\\sqrt{h}$; at $h = 20$: @{w1.20}$\\sigma$ (random walk), @{w2.20}$\\sigma$ (ARIMA(1,1,0)), @{w3.20}$\\sigma$ (ARIMA(0,1,1), $\\theta = -0.5$)',
       '$d = 1$: creștere ca $\\sqrt{h}$; la $h = 20$: @{w1.20}$\\sigma$ (mers aleator), @{w2.20}$\\sigma$ (ARIMA(1,1,0)), @{w3.20}$\\sigma$ (ARIMA(0,1,1), $\\theta = -0{,}5$)'),
     [T('a negative MA part offsets part of each shock: narrower intervals', 'o componentă MA negativă compensează o parte din fiecare șoc: intervale mai înguste')]),
    T('$d = 2$: growth like $h^{3/2}$ (@{w4.20}$\\sigma$ at $h = 20$): over-differencing makes long-run forecasts useless', '$d = 2$: creștere ca $h^{3/2}$ (@{w4.20}$\\sigma$ la $h = 20$): supradiferențierea face prognozele pe termen lung inutilizabile')])

chart(T('Romanian GDP: two forecasts', 'PIB-ul României: două prognoze'), 'tsa_ch3_gdp_forecast', 'TSA_ch3_arima_gdp', [
    T('@{gd.order} with drift (red) and a linear trend with AR(1) errors (blue), both estimated on @{gi.q0}--@{gi.q1}; forecasts to @{gf2.fq}, transformed back to bn EUR',
      '@{gd.order} cu derivă (roșu) și un trend liniar cu erori AR(1) (albastru), ambele estimate pe @{gi.q0}--@{gi.q1}; prognoze pînă în @{gf2.fq}, transformate înapoi în mld. EUR')], h='0.56\\textheight')

interp(('the GDP forecasts', 'prognozelor PIB'), [
    (T('The random walk with drift grows by @{gd.drift}\\% per quarter from the last value (@{gf2.last} bn EUR in @{gf2.lq}): @{gf2.ds.l12} bn after 12 quarters',
       'Mersul aleator cu derivă crește cu @{gd.drift}\\% pe trimestru de la ultima valoare (@{gf2.last} mld. EUR în @{gf2.lq}): @{gf2.ds.l12} mld. după 12 trimestre'),
     [T('95\\% half-width (in \\% of GDP): @{gf2.ds.w1} after 1 quarter, @{gf2.ds.w4} after 4, @{gf2.ds.w12} after 12, i.e.\\ $\\times$@{gf2.ratio} $= \\sqrt{12}$', 'semilățimea de 95\\% (în \\% din PIB): @{gf2.ds.w1} după 1 trimestru, @{gf2.ds.w4} după 4, @{gf2.ds.w12} după 12, adică $\\times$@{gf2.ratio} $= \\sqrt{12}$')]),
    (T('The TS model pulls GDP back towards its old line ($\\hat\\phi = @{gf2.ts.phi}$): @{gf2.ts.l12} bn after 12 quarters, half-width @{gf2.ts.w12}\\%', 'Modelul staționar în jurul trendului readuce PIB-ul spre vechea dreaptă ($\\hat\\phi = @{gf2.ts.phi}$): @{gf2.ts.l12} mld. după 12 trimestre, semilățimea @{gf2.ts.w12}\\%'),
     [T('more optimistic and more confident, because it assumes the recent slowdown is temporary', 'mai optimist și cu intervale mai înguste, pentru că presupune că încetinirea recentă este temporară')]),
    T('The tests could not settle TS against DS for this sample: the choice of $d$ is a forecasting decision with consequences', 'Testele nu au putut decide între cele două modele pe acest eșantion: alegerea lui $d$ este o decizie de prognoză cu consecințe')])

D.frame(T('Over-differencing', 'Supradiferențierea'), items(
    (T('Differencing an $I(0)$ series creates an MA unit root: $\\Delta\\varepsilon_t = \\varepsilon_t - \\varepsilon_{t-1}$, $\\theta = -1$ (Chapter 1)', 'Diferențierea unei serii $I(0)$ creează o rădăcină unitară MA: $\\Delta\\varepsilon_t = \\varepsilon_t - \\varepsilon_{t-1}$, $\\theta = -1$ (Capitolul 1)'),
     [T('the result is not invertible: no AR($\\infty$) representation, poor estimation \\refPS', 'rezultatul nu este inversabil: nu are reprezentare AR($\\infty$), iar estimarea este dificilă \\refPS')]),
    (T('Symptoms', 'Simptome'),
     [T('$\\hat\\rho(1)$ of the differenced series close to $-0.5$', '$\\hat\\rho(1)$ al seriei diferențiate apropiat de $-0{,}5$'),
      T('the variance increases after differencing instead of decreasing', 'varianța crește după diferențiere, în loc să scadă'),
      T('an estimated MA coefficient close to $-1$', 'un coeficient MA estimat apropiat de $-1$')]),
    T('Rule: difference only as long as the variance goes down and the tests ask for it', 'Regula: diferențiem doar cît timp varianța scade și testele o cer')))

chart(T('Over-differencing Romanian GDP', 'Supradiferențierea PIB-ului României'), 'tsa_ch3_overdiff_gdp', 'TSA_ch3_arima_gdp', [
    T('ACF of the quarterly growth $\\Delta\\,100\\ln Y_t$ ($d = 1$) and of its difference $\\Delta^2\\,100\\ln Y_t$ ($d = 2$)',
      'ACF a creșterii trimestriale $\\Delta\\,100\\ln Y_t$ ($d = 1$) și a diferenței ei $\\Delta^2\\,100\\ln Y_t$ ($d = 2$)')], h='0.48\\textheight')

interp(('over-differencing', 'supradiferențierii'), [
    T('$d = 1$: $\\hat\\rho(1) = @{od.r1}$, variance @{od.v1}; $d = 2$: $\\hat\\rho(1) = @{od.r2}$, variance @{od.v2}: all three symptoms', '$d = 1$: $\\hat\\rho(1) = @{od.r1}$, varianța @{od.v1}; $d = 2$: $\\hat\\rho(1) = @{od.r2}$, varianța @{od.v2}: toate cele trei simptome'),
    T('An MA(1) fitted to $\\Delta^2\\,100\\ln Y_t$ gives $\\hat\\theta = @{od.th}$ (SE @{od.thse}): practically $-1$, the extra difference is cancelled by the MA part', 'Un MA(1) estimat pe $\\Delta^2\\,100\\ln Y_t$ dă $\\hat\\theta = @{od.th}$ (SE @{od.thse}): practic $-1$, diferența în plus este anulată de partea MA'),
    T('Correct choice: $d = 1$', 'Alegerea corectă: $d = 1$')])

D.recap(('ARIMA models', 'modele ARIMA'), [
    T('ARIMA$(p,d,q)$: an ARMA$(p,q)$ for $\\Delta^d y_t$; the constant decides the long-run slope', 'ARIMA$(p,d,q)$: un ARMA$(p,q)$ pentru $\\Delta^d y_t$; constanta decide panta pe termen lung'),
    T('Romanian GDP since 2000: @{gm.best} with drift, @{gd.drift}\\% per quarter; residuals uncorrelated but heavy-tailed', 'PIB-ul României din 2000: @{gm.best} cu derivă, @{gd.drift}\\% pe trimestru; reziduuri necorelate, dar cu cozi groase'),
    T('Forecast variance $\\sigma^2\\sum_{j<h}\\psi_j^2$: bounded for $d = 0$, growing for $d \\ge 1$', 'Varianța prognozei $\\sigma^2\\sum_{j<h}\\psi_j^2$: mărginită pentru $d = 0$, crescătoare pentru $d \\ge 1$'),
    T('Over-differencing: $\\hat\\rho(1) \\approx -0.5$, larger variance, MA coefficient near $-1$', 'Supradiferențierea: $\\hat\\rho(1) \\approx -0{,}5$, varianță mai mare, coeficient MA apropiat de $-1$')])

# =============================================================================
# 7. ARIMA ÎN PRACTICĂ
# =============================================================================
D.section('ARIMA in practice', 'ARIMA în practică')

D.frame(T('Automatic ARIMA', 'Selecția automată ARIMA'), items(
    (T('\\refHK\\ (\\texttt{auto.arima} in R; \\texttt{pmdarima} and \\texttt{statsforecast} in Python):', '\\refHK\\ (\\texttt{auto.arima} în R; \\texttt{pmdarima} și \\texttt{statsforecast} în Python):'),
     [T('$d$: repeated KPSS tests at 5\\%: difference while KPSS rejects', '$d$: teste KPSS repetate la 5\\%: diferențiem cît timp KPSS respinge'),
      T('$p$, $q$: a stepwise search from a few starting models, by AICc, on the chosen $d$; a constant or drift if $d \\le 1$', '$p$, $q$: o căutare pas cu pas pornind de la cîteva modele inițiale, după AICc, cu $d$ ales; constantă sau derivă dacă $d \\le 1$'),
      T('rejects models with roots too close to the unit circle and fits that fail', 'elimină modelele cu rădăcini prea apropiate de cercul unitate și estimările care eșuează')]),
    (T('Useful for many series at once and as a starting point', 'Util pentru multe serii deodată și ca punct de plecare'),
     [T('evaluated by its authors on the 3\\,003 series of the M3 forecasting competition, it was competitive with the best methods there', 'evaluată de autori pe cele 3\\,003 serii ale competiției de prognoză M3, metoda a fost competitivă cu cele mai bune metode din competiție')]),
    T('But each step embeds a choice that you must be able to defend (next slide)', 'Dar fiecare pas conține o alegere pe care trebuie să o puteți justifica (slide-ul următor)')))

D.frame(T('Automatic ARIMA: checks', 'Selecția automată ARIMA: verificări'), items(
    (T('\\textbf{AICc is not comparable across $d$}: with $d = 1$ the likelihood is computed for $\\Delta y_t$, a different series', '\\textbf{AICc nu este comparabil între valori diferite ale lui $d$}: cu $d = 1$, verosimilitatea se calculează pentru $\\Delta y_t$, o altă serie'),
     [T('choose $d$ by tests and judgement, then compare $p$, $q$', 'alegem $d$ prin teste și judecată, apoi comparăm $p$ și $q$')]),
    (T('\\textbf{Near-cancelling roots}: Romanian GDP from @{gf.q0}: the lowest AICc, @{gf.aicc} (ARIMA(0,1,0): @{gf.aicc0}), belongs to @{gf.best}, whose fit did not converge',
       '\\textbf{Rădăcini care aproape se anulează}: PIB-ul României din @{gf.q0}: cel mai mic AICc, @{gf.aicc} (ARIMA(0,1,0): @{gf.aicc0}), aparține modelului @{gf.best}, a cărui estimare nu a convers'),
     [T('MA roots of modulus @{gf.ma}, AR roots @{gf.ar}: AR and MA almost cancel; $\\hat\\theta_2 = @{gf.ma2}$ with SE @{gf.ma2se}', 'rădăcini MA de modul @{gf.ma}, rădăcini AR @{gf.ar}: AR și MA aproape se anulează; $\\hat\\theta_2 = @{gf.ma2}$ cu SE @{gf.ma2se}'),
      T('a model fitted to the noise of the 1990s data, not a description of the economy', 'un model ajustat pe zgomotul datelor din anii 1990, nu o descriere a economiei')]),
    T('\\textbf{Always}: plot the data, check the convergence and the roots, run the residual tests, compare with a simple benchmark (random walk, Chapter 0)', '\\textbf{Întotdeauna}: reprezentăm grafic datele, verificăm convergența și rădăcinile, aplicăm testele pe reziduuri, comparăm cu o metodă simplă de referință (mersul aleator, Capitolul 0)')))

chart(T('Romanian inflation: $d = 0$ or $d = 1$?', 'Inflația din România: $d = 0$ sau $d = 1$?'), 'tsa_ch3_inflation_d', 'TSA_ch3_arima_in_practice', [
    T('12-month HICP inflation (Eurostat), January 2005--@{in.last}; best model by AICc for $d = 0$ (with a mean) and for $d = 1$, $p \\le 3$, $q \\le 2$; forecasts for 24 months',
      'Inflația anuală IAPC (Eurostat), ianuarie 2005--@{in.last}; cel mai bun model după AICc pentru $d = 0$ (cu medie) și pentru $d = 1$, $p \\le 3$, $q \\le 2$; prognoze pentru 24 de luni')], h='0.54\\textheight')

interp(('the two inflation models', 'celor două modele ale inflației'), [
    (T('$d = 0$: @{in.d0.o}, $\\hat\\phi_1 = @{in.a1}$, $\\hat\\phi_2 = @{in.a2}$, sum @{in.asum}: stationary, but barely; mean @{in.mu}\\%', '$d = 0$: @{in.d0.o}, $\\hat\\phi_1 = @{in.a1}$, $\\hat\\phi_2 = @{in.a2}$, suma @{in.asum}: staționar, dar la limită; media @{in.mu}\\%'),
     [T('$d = 1$: @{in.d1.o}, $\\hat\\phi = @{in.b1}$: almost the same short-run dynamics', '$d = 1$: @{in.d1.o}, $\\hat\\phi = @{in.b1}$: aproape aceeași dinamică pe termen scurt')]),
    (T('After 24 months: @{in.d0.h24}\\% in $[@{in.d0.lo24}, @{in.d0.hi24}]$ against @{in.d1.h24}\\% in $[@{in.d1.lo24}, @{in.d1.hi24}]$', 'După 24 de luni: @{in.d0.h24}\\% în $[@{in.d0.lo24}, @{in.d0.hi24}]$, față de @{in.d1.h24}\\% în $[@{in.d1.lo24}, @{in.d1.hi24}]$'),
     [T('similar points, different intervals: $d$ decides how much uncertainty we report', 'prognoze punctuale similare, intervale diferite: $d$ decide cîtă incertitudine raportăm')]),
    (T('KPSS (@{in.k0}) keeps $d = @{in.dhk}$, as automatic ARIMA would; ADF (p = @{in.adf0p}) suggests $d = 1$', 'KPSS (@{in.k0}) păstrează $d = @{in.dhk}$, ca selecția automată; ADF (p = @{in.adf0p}) sugerează $d = 1$'),
     [T('both models leave autocorrelated residuals: 12-month rates overlap and inherit the season; Chapter 4 models monthly inflation', 'ambele modele lasă reziduuri autocorelate: ratele anuale se suprapun și moștenesc sezonalitatea; Capitolul 4 modelează inflația lunară')])])

D.frame(T('Case study: Meese and Rogoff (1983)', 'Studiu de caz: Meese și Rogoff (1983)'), items(
    (T('\\textbf{Question}: can exchange-rate models forecast better than a random walk?', '\\textbf{Întrebarea}: pot modelele cursului de schimb să prognozeze mai bine decît un mers aleator?'),
     [T('\\refMR: monetary models of the 1970s, time series models (ARIMA, VAR) and the random walk; dollar rates, horizons of 1 to 12 months', '\\refMR: modelele monetare din anii 1970, modele de serii de timp (ARIMA, VAR) și mersul aleator; cursuri ale dolarului, orizonturi de 1--12 luni')]),
    (T('\\textbf{Finding}: out of sample, none beat the random walk in RMSE (root mean squared error)', '\\textbf{Rezultatul}: în afara eșantionului, niciun model nu a depășit mersul aleator ca RMSE (root mean squared error, rădăcina erorii pătratice medii)'),
     [T('even with the actual future values of the fundamentals plugged in', 'nici măcar atunci cînd au folosit valorile viitoare efective ale variabilelor fundamentale')]),
    (T('\\textbf{Legacy}: the ``Meese--Rogoff puzzle\'\' and the random walk as the benchmark for every exchange-rate forecast', '\\textbf{Moștenirea}: „enigma Meese--Rogoff” și mersul aleator ca reper pentru orice prognoză a cursului de schimb'),
     [T('next chart: the same comparison for EUR/RON', 'graficul următor: aceeași comparație pentru EUR/RON')])))

chart(T('EUR/RON: ARIMA against the random walk', 'EUR/RON: ARIMA față de mersul aleator'), 'tsa_ch3_eurron_forecast', 'TSA_ch3_arima_in_practice', [
    T('Parameters estimated on @{fx.ntr} daily changes until December 2023; @{fx.nte} one-day-ahead forecasts from January 2024; right: the cumulated difference of squared errors (below 0: the model beats the random walk)',
      'Parametrii estimați pe @{fx.ntr} variații zilnice pînă în decembrie 2023; @{fx.nte} prognoze cu o zi înainte din ianuarie 2024; dreapta: diferența cumulată a erorilor pătratice (sub 0: modelul este mai precis decît mersul aleator)')], h='0.52\\textheight')

interp(('the EUR/RON comparison', 'comparației EUR/RON'), [
    (T('RMSE of the daily change ($100\\ln$): random walk @{fx.r1.rw}, with drift @{fx.r1.drift}, ARIMA(1,1,0) @{fx.r1.ar} ($\\hat\\phi = @{fx.phi}$)', 'RMSE al variației zilnice ($100\\ln$): mers aleator @{fx.r1.rw}, cu derivă @{fx.r1.drift}, ARIMA(1,1,0) @{fx.r1.ar} ($\\hat\\phi = @{fx.phi}$)'),
     [T('a gain of @{fx.g1}\\%; over 20 days (@{fx.n20} non-overlapping windows): @{fx.r20.rw} against @{fx.r20.ar}, a gain of @{fx.g20}\\%', 'un cîștig de @{fx.g1}\\%; pe 20 de zile (@{fx.n20} ferestre fără suprapunere): @{fx.r20.rw} față de @{fx.r20.ar}, un cîștig de @{fx.g20}\\%')]),
    (T('The gain comes from a few days: two jumps of the rate (from @{fx.s0} to @{fx.s1} lei over the period) continued on the next day', 'Cîștigul provine din cîteva zile: două salturi ale cursului (de la @{fx.s0} la @{fx.s1} lei în perioada analizată) au continuat și a doua zi'),
     [T('on calm days the random walk is as good; a formal comparison needs the Diebold--Mariano test (Chapter 4)', 'în zilele calme, mersul aleator este la fel de bun; o comparație formală cere testul Diebold--Mariano (Capitolul 4)')]),
    T('Lesson of Meese and Rogoff: report the random walk next to any exchange-rate forecast', 'Lecția Meese--Rogoff: raportăm mersul aleator alături de orice prognoză a cursului de schimb')])

chart(T('A classic series: US real GDP after 2007', 'O serie clasică: PIB-ul real al SUA după 2007'), 'tsa_ch3_us_gdp', 'TSA_ch3_us_gdp', [
    T('FRED series GDPC1 since 1947; a TS model (linear trend, AR(2) errors) and a DS model (ARIMA(1,1,0) with drift) estimated up to 2007Q4 and forecast to @{us.lq}; 95\\% intervals',
      'Seria FRED GDPC1 din 1947; un model staționar în jurul trendului (trend liniar, erori AR(2)) și unul staționar în diferențe (ARIMA(1,1,0) cu derivă), estimate pînă în T4 2007 și prognozate pînă în @{us.lq}; intervale de 95\\%')], h='0.54\\textheight')

interp(('the US GDP forecasts', 'prognozelor PIB-ului SUA'), [
    (T('The TS model expected a return to the pre-2008 line; the gap kept widening: $@{us.ts}$ log points at the end (half-width of its interval: @{us.tsh})',
       'Modelul staționar în jurul trendului aștepta o revenire la dreapta de dinainte de 2008; abaterea s-a mărit continuu: $@{us.ts}$ puncte logaritmice la final (semilățimea intervalului: @{us.tsh})'),
     [T('the 2008--2009 loss was never recovered: evidence for a permanent shock, as the DS view of \\refNP\\ implies', 'pierderea din 2008--2009 nu a fost recuperată niciodată: o dovadă a unui șoc permanent, cum implică viziunea \\refNP')]),
    (T('The DS model also misses ($@{us.ds}$), but its interval (half-width @{us.dsh}) at least warns of large uncertainty', 'Și modelul staționar în diferențe greșește ($@{us.ds}$), dar intervalul lui (semilățimea @{us.dsh}) avertizează măcar asupra incertitudinii mari'),
     [T('both assume the same drift as before 2008; growth after 2008 was slower: a break in the drift (Section 5)', 'ambele presupun aceeași derivă ca înainte de 2008; creșterea de după 2008 a fost mai lentă: o ruptură în derivă (secțiunea 5)')]),
    T('Unit-root tests on the full series: ADF p = @{us.adfp}, KPSS @{us.kpss}: $I(1)$', 'Testele de rădăcină unitară pe întreaga serie: ADF p = @{us.adfp}, KPSS @{us.kpss}: $I(1)$')])

D.recap(('ARIMA in practice', 'ARIMA în practică'), [
    T('Automatic ARIMA: KPSS for $d$, AICc for $p$, $q$; check convergence, roots and residuals', 'Selecția automată ARIMA: KPSS pentru $d$, AICc pentru $p$ și $q$; verificăm convergența, rădăcinile și reziduurile'),
    T('AICc compares models with the same $d$ only', 'AICc compară doar modele cu același $d$'),
    T('Exchange rates: the random walk is a hard benchmark \\refMR', 'Cursurile de schimb: mersul aleator este un reper greu de depășit \\refMR'),
    T('TS or DS changes the long-run forecast and its interval: US GDP after 2008', 'Alegerea între staționaritate în jurul trendului și în diferențe schimbă prognoza pe termen lung și intervalul ei: PIB-ul SUA după 2008')])

# =============================================================================
# 8. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Code}: a first draft of a script that runs ADF, PP and KPSS on many series and tabulates the verdicts', '\\textbf{Cod}: o primă versiune a unui script care aplică ADF, PP și KPSS pe multe serii și tabelează verdictele'),
    T('\\textbf{Explanation}: a second explanation of why the Dickey--Fuller distribution is not Normal, or of a ZA output', '\\textbf{Explicații}: o a doua explicație a motivului pentru care distribuția Dickey--Fuller nu este Normală sau a unui rezultat ZA'),
    T('\\textbf{Exploration}: unit roots and breaks in all EU inflation rates, or ARIMA benchmarks for many Romanian series', '\\textbf{Explorare}: rădăcini unitare și rupturi în toate ratele inflației din UE sau modele ARIMA de referință pentru multe serii românești'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that loads Romanian quarterly real GDP (Eurostat namq\\_10\\_gdp, seasonally adjusted), runs ADF with constant and trend and KPSS on 100*log GDP and on its difference, fits ARIMA(p,1,q) with drift for p, q <= 2, and plots 8-quarter forecasts with 95\\% intervals.}',
        '\\aiprompt{Write Python code that loads Romanian quarterly real GDP (Eurostat namq\\_10\\_gdp, seasonally adjusted), runs ADF with constant and trend and KPSS on 100*log GDP and on its difference, fits ARIMA(p,1,q) with drift for p, q <= 2, and plots 8-quarter forecasts with 95\\% intervals.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('The direction of each test: ADF and PP have the unit root as $H_0$, KPSS has stationarity as $H_0$', 'Sensul fiecărui test: ADF și PP au rădăcina unitară ca $H_0$, KPSS are staționaritatea ca $H_0$'),
    T('The deterministic terms, the number of lags and the critical values actually used (Dickey--Fuller, not Normal)', 'Termenii determiniști, numărul de laguri și valorile critice folosite efectiv (Dickey--Fuller, nu ale distribuției Normale)'),
    T('That ``not rejected\'\' is not reported as ``proved\'\'', 'Că „nerespins” nu este raportat ca „demonstrat”'),
    T('That regressions between trending series are not presented as causal evidence', 'Că regresiile între serii cu trend nu sînt prezentate drept dovezi de cauzalitate'),
    T('Convergence warnings, roots near the unit circle, residual tests, and AICc compared only for the same $d$', 'Avertismentele de convergență, rădăcinile apropiate de cercul unitate, testele pe reziduuri și compararea AICc doar pentru același $d$'),
    T('Every cited reference: it must exist; check the DOI', 'Fiecare referință citată: trebuie să existe; verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Trending series are TS (shocks fade) or DS (shocks are permanent); the treatment differs: detrend or difference', 'Seriile cu trend sînt staționare în jurul trendului (șocurile se sting) sau staționare în diferențe (șocurile sînt permanente); tratamentul diferă: eliminăm trendul sau diferențiem'),
    T('Regressions between unrelated $I(1)$ series are spurious: high $t$ and $R^2$, DW near 0', 'Regresiile între serii $I(1)$ fără legătură sînt false: $t$ și $R^2$ mari, DW apropiat de 0'),
    T('ADF and PP test $H_0$: unit root with Dickey--Fuller critical values; KPSS tests $H_0$: stationarity; use them together', 'ADF și PP testează $H_0$: rădăcină unitară, cu valori critice Dickey--Fuller; KPSS testează $H_0$: staționaritate; le folosim împreună'),
    T('The tests have low power near $\\phi = 1$ and are misled by breaks; Zivot--Andrews allows one break', 'Testele au putere mică în apropierea lui $\\phi = 1$ și sînt înșelate de rupturi; Zivot--Andrews permite o ruptură'),
    T('ARIMA$(p,d,q)$ = ARMA on $\\Delta^d y_t$; its forecast intervals grow with the horizon when $d \\ge 1$', 'ARIMA$(p,d,q)$ = ARMA pe $\\Delta^d y_t$; intervalele de prognoză cresc cu orizontul cînd $d \\ge 1$'),
    T('Automatic selection is a starting point: check $d$, convergence, roots, residuals and a random-walk benchmark', 'Selecția automată este un punct de plecare: verificăm $d$, convergența, rădăcinile, reziduurile și comparația cu mersul aleator')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.4}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('TS and DS', 'Cele două tipuri de trend') + ' & $y_t = \\alpha + \\beta t + u_t$, \\quad $\\Delta y_t = \\beta + u_t$, \\quad $u_t \\sim I(0)$',
     'ADF & $\\Delta y_t = c + bt + \\gamma y_{t-1} + \\sum_{j=1}^{k}\\delta_j\\Delta y_{t-j} + \\varepsilon_t$, \\quad $H_0$: $\\gamma = 0$, \\quad $\\tau = \\hat\\gamma/\\mathrm{SE}(\\hat\\gamma)$',
     T('Lags', 'Laguri') + ' & $k_{\\max} = 12\\,(T/100)^{1/4}$, ' + T('then AIC or BIC', 'apoi AIC sau BIC'),
     'KPSS & $\\eta = \\sum_{t=1}^{T} S_t^2/(T^2\\hat\\lambda^2)$, \\quad $S_t = \\sum_{s \\le t} e_s$, \\quad 5\\%: ' + T('0.463 (c), 0.146 (c, t)', '0,463 (c); 0,146 (c, t)'),
     'ARIMA$(p,d,q)$ & $\\phi(L)(1 - L)^d y_t = c + \\theta(L)\\varepsilon_t$',
     T('Forecast variance', 'Varianța prognozei') + ' & $\\sigma^2\\sum_{j=0}^{h-1}\\psi_j^2$, \\quad $\\psi(z) = \\theta(z)/[\\phi(z)(1 - z)^d]$; ' + T('random walk', 'mers aleator') + ': $h\\sigma^2$',
     'ARIMA(0,1,1) & $\\sigma^2[1 + (h - 1)(1 + \\theta)^2]$, \\quad $\\alpha_{\\mathrm{SES}} = 1 + \\theta$'],
    size='scriptsize') + '}')

D.frame(T('Self-assessment', 'Autoevaluare'), items(
    (T('\\textbf{Question}: an ADF test with constant and trend gives $\\tau = -2.9$ with $T = 200$. Do you reject the unit root at 5\\%?', '\\textbf{Întrebare}: un test ADF cu constantă și trend dă $\\tau = -2{,}9$ pentru $T = 200$. Respingeți rădăcina unitară la 5\\%?'),
     [T('\\textbf{Answer}: no: $-2.9 > -3.43$; with the Normal table you would wrongly reject', '\\textbf{Răspuns}: nu: $-2{,}9 > -3{,}43$; cu tabelul distribuției Normale ați respinge greșit')]),
    (T('\\textbf{Question}: ADF does not reject and KPSS does not reject. What do you conclude?', '\\textbf{Întrebare}: ADF nu respinge, iar KPSS nu respinge. Ce concluzie trageți?'),
     [T('\\textbf{Answer}: inconclusive: the sample is too short to decide; report both models, or choose $d$ by the purpose', '\\textbf{Răspuns}: neconcludent: eșantionul este prea scurt pentru o decizie; raportăm ambele modele sau alegem $d$ după scop')]),
    (T('\\textbf{Question}: for ARIMA(0,1,1) with $\\theta = -0.6$ and $\\sigma = 1$, what is the variance of the 5-step forecast error?', '\\textbf{Întrebare}: pentru ARIMA(0,1,1) cu $\\theta = -0{,}6$ și $\\sigma = 1$, cît este varianța erorii de prognoză la 5 pași?'),
     [T('\\textbf{Answer}: $1 + 4 \\cdot 0.4^2 = 1.64$', '\\textbf{Răspuns}: $1 + 4 \\cdot 0{,}4^2 = 1{,}64$')]),
    T('Next: Chapter 4, seasonality: seasonal differences, SARIMA, TBATS and Prophet', 'Urmează: Capitolul 4, sezonalitatea: diferențe sezoniere, SARIMA, TBATS și Prophet')))

D.references(bib())

if __name__ == '__main__':
    for _k, _v in list(V.items()):   # a true minus sign for negative numbers, in text and in math
        if isinstance(_v, str) and _v.startswith('⁅-'):
            V[_k] = '⁅\\ensuremath{-}' + _v[2:]
    finalize(D.write(V))
