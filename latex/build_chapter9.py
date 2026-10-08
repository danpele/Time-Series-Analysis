r"""
build_chapter9.py -- Capitolul 9 (Învățare automată pentru serii de timp), EN + RO dintr-o singură sursă
======================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_09/ch9_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter9_machine_learning_time_series.tex
  RO/Cursuri/capitol9_invatare_automata_serii_timp.tex
Rulare:
  python3 Quantlets/Ch_09/generate_all_charts.py
  python3 latex/build_chapter9.py && python3 latex/tsa_build.py compile 9
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, photo   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch9_common import QLURL, REFS, T, bib, date, finalize, load, pv, month   # noqa: E402


def items(*xs):
    """tsa_build.items, with (text, []) treated as a plain bullet."""
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


N = load()
V = Values()
D = Deck(9, 'lecture', refs=REFS)
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
    'perceptron': ('ch9_mark1_perceptron_1960.png', C + 'Mark_I_Perceptron,_Figure_2_of_operator%27s_manual.png',
                   T('Image', 'Imagine') + ': J. C. Hay, A. E. Murray (1960), Mark I Perceptron operator\'s manual; public domain; Wikimedia Commons'),
    'hochreiter': ('ch9_sepp_hochreiter_2015.jpg', C + 'Sepp_Hochreiter.JPG',
                   T('Photo', 'Foto') + ': Eulenreich (2015); CC BY-SA 4.0; Wikimedia Commons'),
    'pylon': ('ch9_marasesti_750kv_2022.jpg', C + '750_kV_electricity_tower_at_M%C4%83r%C4%83%C8%99e%C8%99ti,_Romania.jpg',
              T('Photo', 'Foto') + ': TrainSimFan (2022); CC BY-SA 4.0; Wikimedia Commons'),
    'walmart': ('ch9_walmart_store.jpg', C + 'Walmart_store_exterior_5266815680.jpg',
                T('Photo', 'Foto') + ': Walmart Corporate; CC BY 2.0; Wikimedia Commons'),
    'chicago': ('ch9_harper_center_2013.jpg', C + 'University_of_Chicago_July_2013_01_(Charles_M._Harper_Center).jpg',
                T('Photo', 'Foto') + ': Michael Barera (2013); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url.replace('%', '\\%'), cred, h=h)


def twocol(left, right, wl='0.4', wr='0.58'):
    return (f'\\begin{{columns}}[T]\n\\begin{{column}}{{{wl}\\textwidth}}\n{left}\n\\end{{column}}\n'
            f'\\begin{{column}}{{{wr}\\textwidth}}\n{right}\n\\end{{column}}\n\\end{{columns}}')


# =============================================================================
# CIFRE
# =============================================================================
F = N['features']
V.int('f.n', F['n'])
V.raw('f.first', date(F['first']))
V.raw('f.last', date(F['last']))
V.put('f.mean', F['mean'], 2)
V.put('f.min', F['min'], 2)
V.raw('f.mind', date(F['min_d']))
V.put('f.max', F['max'], 2)
V.raw('f.maxd', date(F['max_d']))
V.put('f.c0', F['corr']['lag0'], 2)
V.put('f.c6', F['corr']['lag6'], 2)
V.put('f.c7', F['corr']['mean7'], 2)
V.int('f.rows', F['rows'])
V.raw('f.p', str(F['n_feats']))

MI = N['mini']['rows']
DOW_EN = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
DOW_RO = ['lu', 'ma', 'mi', 'jo', 'vi', 'sî', 'du']

L = N['leakage']
for s in ('sim', 'sp500'):
    for k in ('kfold', 'wf', 'gap'):
        V.put(f'lk.{s}.{k}', 100 * L[s][k], 1)
V.int('lk.n', L['sp500']['n'])

BV = N['bv']
V.raw('bv.best', str(BV['best_depth']))
for k in ('1', '3', '10'):
    V.put(f'bv.{k}.b', BV[f'd{k}']['bias2'], 3)
    V.put(f'bv.{k}.v', BV[f'd{k}']['var'], 3)
    V.put(f'bv.{k}.t', BV[f'd{k}']['test'], 3)
    V.put(f'bv.{k}.tr', BV[f'd{k}']['train'], 3)

SH = N['shrink']
V.raw('sh.kept', str(SH['n_kept']))
V.raw('sh.p', str(SH['p']))
V.int('sh.n', SH['n'])
V.put('sh.alpha', SH['alpha_cv'], 4)
V.put('sh.b0', SH['top'][0][1], 2)

TR = N['tree']
V.put('tr.thr', TR['root']['thr'], 2)
V.put('tr.root', TR['root']['value'], 2)
V.int('tr.n', TR['root']['n'])
V.put('tr.r2', TR['r2'], 2)
lv = TR['leaves']
for i, l_ in enumerate(lv):
    V.put(f'tr.l{i}', l_['value'], 2)
    V.raw(f'tr.n{i}', str(l_['n']))
V.put('tr.k0', TR['kids'][0]['thr'], 2)
V.put('tr.k1', TR['kids'][1]['thr'], 2)

EN_ = N['ens']
for k in ('tree_best', 'rf_best', 'naive', 'gb03_min', 'gb005_min', 'gb03_end', 'gb005_end'):
    V.put(f'en.{k}', EN_[k], 3)
for k in ('tree_best_d', 'rf_best_d', 'gb03_min_it', 'gb005_min_it', 'n_tr', 'n_va'):
    V.raw(f'en.{k}', str(EN_[k]))

EX = N['extra']
V.put('ex.max', EX['max_train'], 1)
V.put('ex.last', EX['last'], 1)
V.raw('ex.lastd', month(EX['last_d']))
V.put('ex.fl', EX['f_l_last'], 1)
V.put('ex.rl', EX['rmse_l'], 1)
V.put('ex.rc', EX['rmse_c'], 2)

IM = N['imp']
i1 = list(IM['1'].items())
i14 = list(IM['14'].items())
V.raw('im.1a', i1[0][0].replace('_', '\\_'))
V.put('im.1av', 1000 * i1[0][1], 0)
V.raw('im.1b', i1[1][0].replace('_', '\\_'))
V.put('im.1bv', 1000 * i1[1][1], 0)
V.raw('im.14a', i14[0][0].replace('_', '\\_'))
V.put('im.14av', 1000 * i14[0][1], 0)
V.raw('im.14b', i14[1][0].replace('_', '\\_'))
V.put('im.14bv', 1000 * i14[1][1], 0)

LD = N['load']
TB = LD['tab']
MODELS = ['Seasonal naive', 'ETS', 'SARIMA', 'DHR', 'Combination', 'Ridge', 'RF', 'GB direct', 'GB recursive', 'MLP', 'LSTM']
KEY = {m: m.replace(' ', '') for m in MODELS}
for m in MODELS:
    for c in ('MAE', 'RMSE', 'MASE'):
        V.put(f'ld.{KEY[m]}.{c}', TB[m][c], 3 if c != 'MASE' else 2)
    V.put(f'ld.{KEY[m]}.mw', 1000 * TB[m]['MAE'], 0)
for k, v in LD['dm'].items():
    a, b = k.split('|')
    V.raw(f'dm.{KEY[a]}.{KEY[b]}', pv(v['p']))
    V.put(f'dmd.{KEY[a]}.{KEY[b]}', 1000 * v['dbar'], 0)
V.raw('ld.no', str(LD['n_origins']))
V.raw('ld.first', date(LD['first']))
V.raw('ld.last', date(LD['last']))
V.put('ld.scale', 1000 * LD['scale'], 0)
V.raw('ld.n0', str(LD['n_train0']))
V.raw('ld.fco', date(LD['fc']['origin']))
for m in ('DHR', 'GB direct', 'GB recursive', 'Seasonal naive', 'LSTM', 'Ridge', 'MLP'):
    V.put(f'ld.fc.{KEY[m]}', 1000 * LD['fc']['mae'][m], 0)
V.put('ld.l0', LD['lstm_loss'][0], 2)
V.put('ld.l1', LD['lstm_loss'][1], 2)
BY = LD['byh']
V.put('ld.h1.GB', 1000 * BY['GB direct']['1'], 0)
V.put('ld.h1.DHR', 1000 * BY['DHR']['1'], 0)
V.put('ld.h14.GB', 1000 * BY['GB direct']['14'], 0)
V.put('ld.h14.DHR', 1000 * BY['DHR']['14'], 0)

IN = N['int']
for k in ('quantile', 'conformal'):
    V.put(f'in.{k}.c', 100 * IN[k]['cover'], 1)
    V.put(f'in.{k}.w', 1000 * IN[k]['width'], 0)
    V.put(f'in.{k}.lo', 100 * IN[k]['below'], 1)
    V.put(f'in.{k}.hi', 100 * IN[k]['above'], 1)
V.raw('in.n', str(IN['n']))
V.raw('in.cal', str(IN['n_cal']))

IF = N['inf']
for h in ('1', '3', '6', '12'):
    S = IF['sum'][h]
    for m in ('RW', 'AR', 'Lasso', 'RF', 'GB local', 'GB global'):
        V.put(f'if.{h}.{m.replace(" ", "")}', S['rmse'][m], 2)
        V.put(f'if.{h}.{m.replace(" ", "")}.r', S['rel'][m], 2)
        V.put(f'if.{h}.{m.replace(" ", "")}.s', S['rmse_2123'][m], 2)
        V.put(f'if.{h}.{m.replace(" ", "")}.q', S['rmse_1620'][m], 2)
    for m, d in S['dm_rw'].items():
        V.raw(f'if.{h}.dm.{m.replace(" ", "")}', pv(d['p']))
    V.raw(f'if.{h}.gl', pv(S['dm_gl']['p']))
    V.raw(f'if.{h}.n', str(S['n']))
V.raw('if.first', month(IF['sum']['1']['first']))
V.raw('if.last1', month(IF['sum']['1']['last']))
FG = IF['fig']
V.put('if.max', FG['max'], 1)
V.raw('if.maxd', month(FG['max_d']))
V.put('if.lastv', FG['last'], 1)
V.raw('if.lastd', month(FG['last_d']))
V.raw('if.nc', str(IF['n_countries']))

RV = N['rv']
for k in ('sp500', 'dax'):
    for m in ('HAR', 'Lasso', 'RF', 'GB', 'MLP', 'Mean'):
        V.put(f'rv.{k}.{m}.r2', 100 * RV[k][m]['r2_har'], 1, sign=True)
        V.put(f'rv.{k}.{m}.ql', RV[k][m]['qlike'], 3)
        if m != 'HAR':
            V.put(f'rv.{k}.{m}.t', RV[k][m]['dm_t'], 2)
    V.int(f'rv.{k}.n', RV[k]['n'])
SG = N['sign']
for k in ('sp500', 'bet'):
    V.put(f'sg.{k}.base', 100 * SG[k]['base_acc'], 1)
    V.put(f'sg.{k}.up', 100 * SG[k]['up'], 1)
    V.int(f'sg.{k}.n', SG[k]['n'])
    for m in ('Logit', 'RF', 'GB'):
        V.put(f'sg.{k}.{m}', 100 * SG[k][m]['acc'], 1)
        V.put(f'sg.{k}.{m}.auc', SG[k][m]['auc'], 3)
        V.raw(f'sg.{k}.{m}.p', pv(SG[k][m]['p']))

M4 = N['m4']
V.raw('m4.n', str(M4['n']))
V.put('m4.best', M4['best_owa'], 3)
V.put('m4.bests', M4['best_smape'], 2)
V.put('m4.second', M4['second_owa'], 3)
V.put('m4.comb', M4['comb'], 3)
V.put('m4.combs', M4['comb_smape'], 2)
V.put('m4.n2s', M4['naive2_smape'], 2)
V.put('m4.mlbest', M4['ml_best_owa'], 3)
V.raw('m4.mlrank', str(M4['ml_best_rank']))
V.raw('m4.nml', str(M4['n_ml']))
V.raw('m4.top17', str(M4['top17_comb']))
V.raw('m4.off', str(M4['n_off']))
V.put('m4.gain', 100 * (1 - M4['best_smape'] / M4['comb_smape']), 1)

# ---- worked examples (computed here)
yT = 5.0
r1 = 1.2 + 0.8 * yT
r2 = 1.2 + 0.8 * r1
d2 = 2.3 + 0.6 * yT
V.put('we.r1', r1, 2)
V.put('we.r2', r2, 2)
V.put('we.d2', d2, 2)
V.put('we.r2f', 1.2 + 0.8 * 1.2, 2)
V.put('we.r2b', 0.8 ** 2, 2)
Sxy, Sxx = 90.0, 100.0
V.put('rg.ols', Sxy / Sxx, 2)
V.put('rg.r50', Sxy / (Sxx + 50), 2)
V.put('rg.l40', (Sxy - 20) / Sxx, 2)
yb = np.array([5.0, 6.0, 8.0])
f0 = yb.mean()
res0 = yb - f0
V.put('gb.f0', f0, 2)
V.raw('gb.res', ', '.join('⁅' + f'{x:+.2f}' + '⁆' for x in res0))
V.put('gb.r3', res0[2], 2)
V.put('gb.f1', f0 + 0.1 * res0[2], 3)
V.put('gb.lr', 0.1, 1)
V.put('pb.a', 0.95 * 0.4, 3)
V.put('pb.b', 0.05 * 0.5, 3)
qk = math.ceil((IN['n_cal'] + 1) * 0.9)
V.raw('cf.k', str(qk))
V.put('mlp.p', 3 * 5 + 5 + 5 + 1, 0)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: can random forests, gradient boosting and neural networks forecast a time series better than ARIMA, ETS and the naive forecast, and how do we check it honestly?',
       '\\textbf{Întrebarea}: pot random forest, gradient boosting și rețelele neuronale prognoza o serie de timp mai bine decît ARIMA, ETS și prognoza naivă și cum verificăm corect acest lucru?'),
     [T('machine learning (ML): algorithms that learn a prediction rule from examples; for time series, the examples are built from the past of the series',
        'învățarea automată (machine learning, ML): algoritmi care învață o regulă de predicție din exemple; pentru serii de timp, exemplele se construiesc din trecutul seriei')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('forecasting as supervised learning: lags, rolling windows, calendar features; recursive, direct and multi-output strategies',
        'prognoza ca învățare supervizată: laguri, ferestre mobile, variabile de calendar; strategiile recursivă, directă și cu ieșiri multiple'),
      T('walk-forward validation and leakage; ridge and lasso; trees, random forest, gradient boosting; small neural networks and the LSTM',
        'validarea walk-forward și leakage-ul; ridge și lasso; arbori, random forest, gradient boosting; rețele neuronale mici și LSTM'),
      T('prediction intervals; local and global models; four applications and the M4, M5 competitions',
        'intervale de prognoză; modele locale și globale; patru aplicații și competițiile M4, M5')]),
    T('We build on Chapter 0 (accuracy measures, benchmarks) and Chapter 4 (time-series cross-validation, Diebold--Mariano); Seminar 9 comes before this lecture',
      'Pornim de la Capitolul 0 (măsuri de acuratețe, metode de referință) și de la Capitolul 4 (validare încrucișată pentru serii de timp, Diebold--Mariano); Seminarul 9 are loc înaintea acestui curs')))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Turn a time series into a supervised-learning table without using future information', 'Transformați o serie de timp într-un tabel de învățare supervizată fără să folosiți informație din viitor'),
    T('Produce multi-step forecasts with the recursive, direct and multi-output strategies', 'Construiți prognoze pe mai mulți pași cu strategiile recursivă, directă și cu ieșiri multiple'),
    T('Validate a model walk-forward and recognise leakage and look-ahead bias', 'Validați un model prin walk-forward și recunoașteți leakage-ul și look-ahead bias'),
    T('Explain ridge, lasso, regression trees, random forest, gradient boosting, the MLP and the LSTM, with their main tuning parameters', 'Explicați ridge, lasso, arborii de regresie, random forest, gradient boosting, MLP și LSTM, cu parametrii lor principali'),
    T('Build prediction intervals by quantile regression and conformal prediction', 'Construiți intervale de prognoză prin regresie cuantilică și predicție conformală'),
    T('Compare ML forecasts with statistical benchmarks on the same test period, with MASE and the Diebold--Mariano test', 'Comparați prognozele ML cu metodele statistice de referință pe aceeași perioadă de test, cu MASE și testul Diebold--Mariano')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refHP, Ch.~10 (machine learning for time series)', 'Manual: \\refHP, cap.~10 (învățare automată pentru serii de timp)'),
     [T('companion, free online: \\refFPP, Sec.~12.4 (neural network models); theory: \\refESL, Ch.~3, 9, 10, 11, 15',
        'manual însoțitor, gratuit online: \\refFPP, secț.~12.4 (rețele neuronale); teorie: \\refESL, cap.~3, 9, 10, 11, 15')]),
    T('Reviews: \\refHBB\\ (recurrent networks), \\refLZ\\ (deep learning), \\refMMH\\ (global models)', 'Sinteze: \\refHBB\\ (rețele recurente), \\refLZ\\ (deep learning), \\refMMH\\ (modele globale)'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_09}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_09}'),
     [T('\\texttt{scikit-learn}: \\texttt{RidgeCV}, \\texttt{LassoCV}, \\texttt{RandomForestRegressor}, \\texttt{HistGradientBoostingRegressor}, \\texttt{MLPRegressor}, \\texttt{TimeSeriesSplit}; a tiny LSTM in \\texttt{PyTorch}',
        '\\texttt{scikit-learn}: \\texttt{RidgeCV}, \\texttt{LassoCV}, \\texttt{RandomForestRegressor}, \\texttt{HistGradientBoostingRegressor}, \\texttt{MLPRegressor}, \\texttt{TimeSeriesSplit}; un LSTM mic în \\texttt{PyTorch}'),
      T('small models, fixed seeds, CPU only: every notebook runs in Google Colab in a few minutes', 'modele mici, semințe fixe, doar CPU: fiecare notebook rulează în Google Colab în cîteva minute')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter9_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter9_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. PROGNOZA CA ÎNVĂȚARE SUPERVIZATĂ
# =============================================================================
D.section('Forecasting as supervised learning', 'Prognoza ca învățare supervizată')

D.frame(T('Supervised learning in one slide', 'Învățarea supervizată pe un slide'), items(
    (T('\\textbf{Data}: pairs $(\\mathbf{x}_i, y_i)$, $i = 1, \\dots, n$; $\\mathbf{x}_i$: the \\textbf{features} (inputs), $y_i$: the \\textbf{target}',
       '\\textbf{Datele}: perechi $(\\mathbf{x}_i, y_i)$, $i = 1, \\dots, n$; $\\mathbf{x}_i$: \\textbf{variabilele explicative} (features, intrările), $y_i$: \\textbf{ținta} (target)'),
     [T('goal: a function $\\hat f$ with small expected loss $E[L(y, \\hat f(\\mathbf{x}))]$ on \\textbf{new} data', 'scopul: o funcție $\\hat f$ cu pierdere așteptată $E[L(y, \\hat f(\\mathbf{x}))]$ mică pe date \\textbf{noi}'),
      T('loss: squared error $L = (y - \\hat y)^2$ (forecasts the mean), absolute error $|y - \\hat y|$ (the median)', 'funcția de pierdere: eroarea pătratică $L = (y - \\hat y)^2$ (prognozează media), eroarea absolută $|y - \\hat y|$ (mediana)')]),
    (T('\\textbf{Training}: choose $\\hat f$ in a class of functions (linear, trees, networks) by minimising the average loss on the training data',
       '\\textbf{Antrenarea}: alegem $\\hat f$ dintr-o clasă de funcții (liniare, arbori, rețele) minimizînd pierderea medie pe datele de antrenare'),
     [T('\\textbf{hyperparameters}: settings fixed before training (tree depth, number of trees, penalty); chosen by validation', '\\textbf{hiperparametri}: setări fixate înaintea antrenării (adîncimea arborelui, numărul de arbori, penalizarea); se aleg prin validare')]),
    T('Standard ML assumes the pairs are independent; in a time series they are ordered and dependent, so the table and the validation must respect time',
      'ML standard presupune perechi independente; într-o serie de timp ele sînt ordonate și dependente, deci tabelul și validarea trebuie să respecte timpul')))

D.frame(T('From a series to a table', 'De la o serie la un tabel'), items(
    (T('\\textbf{Lag features}: $y_{t}, y_{t-1}, \\dots, y_{t-p+1}$, known at the forecast origin $t$; target $y_{t+h}$', '\\textbf{Laguri ca variabile explicative} (lag features): $y_{t}, y_{t-1}, \\dots, y_{t-p+1}$, cunoscute la originea prognozei $t$; ținta $y_{t+h}$'),
     [T('an AR$(p)$ (Chapter 2) is exactly a linear regression on this table; ML replaces the linear function by a flexible one', 'un AR$(p)$ (Capitolul 2) este exact o regresie liniară pe acest tabel; ML înlocuiește funcția liniară cu una flexibilă')]),
    (T('\\textbf{Rolling-window features}: means, standard deviations, minima over the last $w$ values',
       '\\textbf{Variabile pe ferestre mobile}: medii, abateri standard, minime pe ultimele $w$ valori'),
     [T('for example the 7-day mean $\\bar y_t^{(7)} = \\frac{1}{7}\\sum_{j=0}^{6} y_{t-j}$', 'de exemplu media pe 7 zile $\\bar y_t^{(7)} = \\frac{1}{7}\\sum_{j=0}^{6} y_{t-j}$'),
      T('the window must end at $t$, the origin: a window that contains $y_{t+1}$ leaks the target', 'fereastra trebuie să se termine la $t$, originea: o fereastră care conține $y_{t+1}$ introduce ținta în variabile (leakage)')]),
    (T('\\textbf{Calendar features} of the target day: day of week, month, Fourier terms $\\sin(2\\pi k\\,d/365.25)$, holidays',
       '\\textbf{Variabile de calendar} ale zilei-țintă: ziua săptămînii, luna, termeni Fourier $\\sin(2\\pi k\\,d/365{,}25)$, sărbători'),
     [T('$d$: the day of the year; $k = 1, 2, \\dots$: the harmonic (one annual cycle for $k = 1$)', '$d$: ziua din an; $k = 1, 2, \\dots$: armonica (un ciclu anual pentru $k = 1$)'),
      T('known in advance, so allowed (Chapter 4: Fourier terms and Orthodox Easter); exogenous variables (temperature) only if their values are known or forecast at $t$',
        'se cunosc dinainte, deci sînt permise (Capitolul 4: termeni Fourier și Paștele ortodox); variabilele exogene (temperatura) doar dacă valorile lor sînt cunoscute sau prognozate la momentul $t$')])))


def mini_rows():
    out = []
    ys = [r['y'] for r in MI]
    for i in range(2, len(MI) - 1):
        r = MI[i]
        out.append(f'⟦{DOW_EN[r["dow"]]}||{DOW_RO[r["dow"]]}⟧ & ' + ' & '.join('⁅' + f'{v:.2f}' + '⁆' for v in (ys[i], ys[i - 1], ys[i - 2], ys[i + 1])))
    return out


D.frame(T('Worked example: a lag table for the electricity load', 'Exemplu rezolvat: un tabel de laguri pentru consumul de energie electrică'), twocol(
    table('lcccc', T('\\textbf{Origin $t$}', '\\textbf{Originea $t$}') + ' & $y_t$ & $y_{t-1}$ & $y_{t-2}$ & ' + T('\\textbf{target} $y_{t+1}$', '\\textbf{ținta} $y_{t+1}$'),
          mini_rows(), size='footnotesize'),
    items((T('Daily mean load of Romania (GW), the last days before ' + '@{ld.first}', 'Consumul mediu zilnic al României (GW), ultimele zile dinaintea datei de ' + '@{ld.first}'),
           [T('each row: what we knew at $t$, and the value we want for $t+1$', 'fiecare rînd: ce știam la $t$ și valoarea dorită pentru $t+1$')]),
          (T('With $p$ lags, the first $p - 1$ days have no complete row', 'Cu $p$ laguri, primele $p - 1$ zile nu au un rînd complet'),
           [T('a series of length $n$ gives $n - p - h + 1$ rows for horizon $h$', 'o serie de lungime $n$ dă $n - p - h + 1$ rînduri pentru orizontul $h$')]),
          T('Weekday column: the Sunday dips are visible; a model needs the weekday of the \\textbf{target} day', 'Coloana zilei: scăderile de duminică se văd; modelul are nevoie de ziua săptămînii a zilei-\\textbf{țintă}')),
    wl='0.5', wr='0.48'), 'footnotesize')

chart(T('Electricity load: the series and three features', 'Consumul de energie electrică: seria și trei variabile explicative'), 'tsa_ch9_features', 'TSA_ch9_supervised_learning', [
    T('Daily mean load of Romania, ENTSO-E (the European Network of Transmission System Operators for Electricity), @{f.first}--@{f.last}, @{f.n} days; bottom: tomorrow\'s load against three features',
      'Consumul mediu zilnic al României, ENTSO-E (rețeaua europeană a operatorilor de transport și de sistem pentru energie electrică), @{f.first}--@{f.last}, @{f.n} de zile; jos: consumul de mîine față de trei variabile explicative')],
    h='0.56\\textheight')

interp(('the load features', 'variabilelor pentru consum'), [
    (T('Mean @{f.mean} GW; minimum @{f.min} GW on @{f.mind} (Orthodox Easter), maximum @{f.max} GW on @{f.maxd} (a winter week)',
       'Media @{f.mean} GW; minimul @{f.min} GW pe @{f.mind} (Paștele ortodox), maximul @{f.max} GW pe @{f.maxd} (o săptămînă de iarnă)'),
     [T('weekly cycle, annual cycle (winter heating, summer cooling) and deep holiday dips: the multiple seasonality of Chapter 4', 'ciclul săptămînal, ciclul anual (încălzire iarna, răcire vara) și scăderi mari de sărbători: sezonalitatea multiplă din Capitolul 4')]),
    (T('Correlation with tomorrow\'s load: $y_t$ @{f.c0}; the same weekday last week @{f.c6}; the 7-day mean @{f.c7}', 'Corelația cu consumul de mîine: $y_t$ @{f.c0}; aceeași zi a săptămînii trecute @{f.c6}; media pe 7 zile @{f.c7}'),
     [T('with $y_t$ alone the Sundays form a separate cloud: a single linear lag cannot represent the weekly pattern', 'doar cu $y_t$, duminicile formează un nor separat: un singur lag liniar nu poate reprezenta tiparul săptămînal'),
      T('the table used below has @{f.rows} rows and @{f.p} features (14 lags, two means, the same weekday, calendar)', 'tabelul folosit mai departe are @{f.rows} de rînduri și @{f.p} de variabile explicative (14 laguri, două medii, aceeași zi a săptămînii, calendar)')])])

D.frame(T('Multi-step forecasts: three strategies', 'Prognoze pe mai mulți pași: trei strategii'), items(
    (T('\\textbf{Recursive} (iterated): one model for $h = 1$; its forecast replaces the unknown $y_{T+1}$ and the model is applied again', '\\textbf{Recursivă} (iterată): un model pentru $h = 1$; prognoza lui înlocuiește valoarea necunoscută $y_{T+1}$ și modelul se aplică din nou'),
     [T('this is how ARIMA forecasts (Chapter 3); errors accumulate when the one-step model is wrong', 'așa prognozează ARIMA (Capitolul 3); erorile se acumulează cînd modelul pe un pas este greșit')]),
    (T('\\textbf{Direct}: a separate model $\\hat f_h$ for each horizon $h = 1, \\dots, H$, trained on the target $y_{t+h}$', '\\textbf{Directă}: un model separat $\\hat f_h$ pentru fiecare orizont $h = 1, \\dots, H$, antrenat pe ținta $y_{t+h}$'),
     [T('no error accumulation, but $H$ models and fewer rows for long horizons; the forecasts of different $h$ need not be coherent', 'erorile nu se acumulează, dar avem $H$ modele și mai puține rînduri pentru orizonturi lungi; prognozele pentru $h$ diferite pot fi incoerente')]),
    (T('\\textbf{Multi-output} (MIMO, multiple-input multiple-output): one model returns the vector $(\\hat y_{T+1}, \\dots, \\hat y_{T+H})$', '\\textbf{Cu ieșiri multiple} (MIMO, multiple-input multiple-output): un model întoarce vectorul $(\\hat y_{T+1}, \\dots, \\hat y_{T+H})$'),
     [T('natural for neural networks; review and comparison: \\refBenTaieb', 'natural pentru rețelele neuronale; sinteză și comparație: \\refBenTaieb')])))

chart(T('Recursive, direct and multi-output forecasts', 'Prognoze recursive, directe și cu ieșiri multiple'), 'tsa_ch9_strategies', 'TSA_ch9_supervised_learning', [
    T('Blue: the inputs known at the origin $T$; green: the fitted models; red: the forecasts; orange: a forecast fed back as an input',
      'Albastru: intrările cunoscute la originea $T$; verde: modelele estimate; roșu: prognozele; portocaliu: o prognoză folosită din nou ca intrare')],
    h='0.45\\textheight')

D.frame(T('Worked example: recursive and direct two-step forecasts', 'Exemplu rezolvat: prognoze recursive și directe pe doi pași'), items(
    (T('One-step model estimated on the table: $\\hat y_{t+1} = 1.2 + 0.8\\,y_t$; last value $y_T = 5$', 'Modelul pe un pas estimat pe tabel: $\\hat y_{t+1} = 1{,}2 + 0{,}8\\,y_t$; ultima valoare $y_T = 5$'),
     [T('recursive: $\\hat y_{T+1} = 1.2 + 0.8 \\cdot 5 = @{we.r1}$; \\quad $\\hat y_{T+2} = 1.2 + 0.8 \\cdot @{we.r1} = @{we.r2}$', 'recursiv: $\\hat y_{T+1} = 1{,}2 + 0{,}8 \\cdot 5 = @{we.r1}$; \\quad $\\hat y_{T+2} = 1{,}2 + 0{,}8 \\cdot @{we.r1} = @{we.r2}$'),
      T('implied two-step rule: $\\hat y_{T+2} = @{we.r2f} + @{we.r2b}\\,y_T$', 'regula implicită pe doi pași: $\\hat y_{T+2} = @{we.r2f} + @{we.r2b}\\,y_T$')]),
    (T('Direct model estimated on the target $y_{t+2}$: $\\hat y_{t+2} = 2.3 + 0.6\\,y_t$; \\quad $\\hat y_{T+2} = @{we.d2}$', 'Modelul direct estimat pe ținta $y_{t+2}$: $\\hat y_{t+2} = 2{,}3 + 0{,}6\\,y_t$; \\quad $\\hat y_{T+2} = @{we.d2}$'),
     [T('if the AR(1) were the true model, the direct coefficients would estimate the same $@{we.r2f}$ and $@{we.r2b}$', 'dacă AR(1) ar fi modelul adevărat, coeficienții direcți ar estima aceleași valori, $@{we.r2f}$ și $@{we.r2b}$'),
      T('they differ: the one-step model is misspecified at two steps (for example, a weekly pattern), and the direct model adapts to the horizon', 'diferă: modelul pe un pas este greșit specificat la doi pași (de exemplu, un tipar săptămînal), iar modelul direct se adaptează orizontului')]),
    T('Neither strategy wins always: try both on the validation data', 'Nicio strategie nu este întotdeauna superioară: le încercăm pe amîndouă pe datele de validare')))

D.recap(('Forecasting as supervised learning', 'prognoza ca învățare supervizată'), [
    T('A forecast is a prediction of $y_{t+h}$ from features known at $t$: lags, rolling statistics, calendar', 'O prognoză este o predicție a lui $y_{t+h}$ din variabile cunoscute la $t$: laguri, statistici pe ferestre mobile, calendar'),
    T('An AR$(p)$ is linear regression on lags; ML changes the function, not the table', 'Un AR$(p)$ este o regresie liniară pe laguri; ML schimbă funcția, nu tabelul'),
    T('Multi-step: recursive (one model, iterated), direct (one model per $h$), MIMO (one model, $H$ outputs)', 'Pe mai mulți pași: recursiv (un model, iterat), direct (un model pentru fiecare $h$), MIMO (un model, $H$ ieșiri)')])

# =============================================================================
# 2. VALIDARE
# =============================================================================
D.section('Validation without leakage', 'Validare fără leakage')

D.frame(T('Training, validation and test samples', 'Eșantioanele de antrenare, validare și test'), items(
    (T('\\textbf{Training} sample: estimates the parameters; \\textbf{validation}: chooses the hyperparameters; \\textbf{test}: measures the final accuracy, used once',
       'Eșantionul de \\textbf{antrenare}: estimează parametrii; de \\textbf{validare}: alege hiperparametrii; de \\textbf{test}: măsoară acuratețea finală, folosit o singură dată'),
     [T('for time series the three samples are consecutive blocks, in this order', 'pentru serii de timp cele trei eșantioane sînt blocuri consecutive, în această ordine')]),
    (T('\\textbf{Walk-forward} validation (time-series cross-validation, Chapter 4): train on data up to an origin, forecast the next block, move the origin forward',
       'Validarea \\textbf{walk-forward} (validarea încrucișată pentru serii de timp, Capitolul 4): antrenăm pe datele de pînă la o origine, prognozăm blocul următor, mutăm originea înainte'),
     [T('\\textbf{expanding} window: all past data; \\textbf{rolling} window: the last $W$ observations (adapts to change, uses less data)', 'fereastră \\textbf{extinsă}: toate datele trecute; fereastră \\textbf{mobilă}: ultimele $W$ observații (se adaptează schimbărilor, folosește mai puține date)'),
      T('the errors of all origins are pooled into MAE, RMSE, MASE (Chapter 0) and compared with the Diebold--Mariano test', 'erorile tuturor originilor se combină în MAE, RMSE, MASE (Capitolul 0) și se compară cu testul Diebold--Mariano')]),
    T('Random $K$-fold cross-validation (random folds) is valid only for pure autoregressions with white-noise errors \\refBHK', 'Validarea încrucișată cu $K$ grupuri aleatoare este validă doar pentru autoregresii pure cu erori de tip zgomot alb \\refBHK')))

chart(T('Four validation schemes', 'Patru scheme de validare'), 'tsa_ch9_cv_schemes', 'TSA_ch9_validation', [
    T('Random $K$-fold mixes past and future; walk-forward keeps the order; a gap removes the training rows whose targets overlap the test block',
      '$K$-fold aleator amestecă trecutul și viitorul; walk-forward păstrează ordinea; un spațiu (gap) elimină rîndurile de antrenare ale căror ținte se suprapun cu blocul de test')],
    h='0.56\\textheight')

D.frame(T('Look-ahead bias and leakage', 'Look-ahead bias și leakage-ul'), items(
    (T('\\textbf{Leakage}: information that would not be available at the forecast origin enters training or the features \\refKaufman', '\\textbf{Leakage-ul} (scurgerea de informație): informație care nu ar fi disponibilă la originea prognozei intră în antrenare sau în variabile \\refKaufman'),
     [T('result: excellent validation scores that disappear in real use; \\textbf{look-ahead bias} is its name in finance', 'rezultat: scoruri excelente la validare, care dispar la utilizarea reală; în finanțe se numește \\textbf{look-ahead bias}')]),
    (T('Typical sources', 'Surse tipice'),
     [T('a rolling mean or a difference that includes $y_{t+1}$; centred moving averages and two-sided filters (Chapter 0)', 'o medie mobilă sau o diferență care include $y_{t+1}$; mediile mobile centrate și filtrele bilaterale (Capitolul 0)'),
      T('scaling, imputation or feature selection fitted on the whole sample before the split', 'scalarea, completarea valorilor lipsă sau selecția variabilelor estimate pe tot eșantionul înainte de împărțire'),
      T('overlapping targets (sums over the next $h$ days) with random folds or without a gap', 'ținte suprapuse (sume pe următoarele $h$ zile) cu grupuri aleatoare sau fără spațiu (gap)'),
      T('revised macro data: today\'s GDP vintage was not known at the origin; actual weather instead of the weather forecast', 'date macroeconomice revizuite: versiunea de azi a PIB-ului nu era cunoscută la origine; vremea efectivă în locul prognozei meteo')]),
    T('Rule: write the feature code as a function of the data up to $t$ only, and refit every preprocessing step inside each training window', 'Regula: scrieți codul variabilelor ca funcție doar de datele de pînă la $t$ și reestimați fiecare pas de preprocesare în fiecare fereastră de antrenare')))

chart(T('Leakage in practice: random folds on overlapping targets', 'Leakage-ul în practică: grupuri aleatoare pe ținte suprapuse'), 'tsa_ch9_leakage', 'TSA_ch9_validation', [
    T('Random forest for the sum of the next 21 daily returns; features: past sums over 5, 21, 63 days and volatilities; left: a simulated random walk; right: the S\\&P 500, @{lk.n} days',
      'Random forest pentru suma următoarelor 21 de randamente zilnice; variabile: sume trecute pe 5, 21, 63 de zile și volatilități; stînga: un mers aleator simulat; dreapta: S\\&P 500, @{lk.n} de zile')],
    h='0.5\\textheight')

interp(('the leakage experiment', 'experimentului de leakage'), [
    (T('Random 5-fold reports $R^2$ = @{lk.sim.kfold}\\% for a random walk, where nothing is predictable, and @{lk.sp500.kfold}\\% for the S\\&P 500',
       '5-fold aleator raportează $R^2$ = @{lk.sim.kfold}\\% pentru un mers aleator, unde nimic nu este previzibil, și @{lk.sp500.kfold}\\% pentru S\\&P 500'),
     [T('neighbouring targets share 20 of 21 returns: a test row has its near-copies in the training folds', 'țintele vecine au în comun 20 din 21 de randamente: un rînd de test are aproape-copii în grupurile de antrenare')]),
    (T('Walk-forward: @{lk.sim.wf}\\% and @{lk.sp500.wf}\\%; with a 21-day gap: @{lk.sim.gap}\\% and @{lk.sp500.gap}\\%', 'Walk-forward: @{lk.sim.wf}\\% și @{lk.sp500.wf}\\%; cu un spațiu de 21 de zile: @{lk.sim.gap}\\% și @{lk.sp500.gap}\\%'),
     [T('negative out-of-sample $R^2$: the forest is worse than the historical mean, as expected for returns (Chapter 1)', '$R^2$ negativ în afara eșantionului: random forest este mai slab decît media istorică, cum ne așteptăm pentru randamente (Capitolul 1)')]),
    T('A model is only as good as its validation; a spectacular score is a reason to look for a leak', 'Un model este atît de bun cît este validarea lui; un scor spectaculos este un motiv să căutăm un leakage')])

D.frame(T('Overfitting and the bias--variance trade-off', 'Overfitting și compromisul deplasare--varianță'), items(
    (T('Model $y = f(x) + \\varepsilon$, $\\Var(\\varepsilon) = \\sigma^2$; $\\hat f$ estimated on a random training sample',
       'Modelul $y = f(x) + \\varepsilon$, $\\Var(\\varepsilon) = \\sigma^2$; $\\hat f$ estimat pe un eșantion de antrenare aleator'),
     [T('the expected test error at a point $x_0$:', 'eroarea de test așteptată într-un punct $x_0$:'),
      T('$E[(y_0 - \\hat f(x_0))^2] = \\underbrace{(E[\\hat f(x_0)] - f(x_0))^2}_{\\text{bias}^2} + \\underbrace{\\Var(\\hat f(x_0))}_{\\text{variance}} + \\sigma^2$',
        '$E[(y_0 - \\hat f(x_0))^2] = \\underbrace{(E[\\hat f(x_0)] - f(x_0))^2}_{\\text{deplasare}^2} + \\underbrace{\\Var(\\hat f(x_0))}_{\\text{varianță}} + \\sigma^2$'),
      T('$f$: the true function; $y_0$: a new observation at $x_0$; the expectation is over training samples and noise; $\\sigma^2$: the noise that no model can remove', '$f$: funcția adevărată; $y_0$: o observație nouă în $x_0$; media se ia după eșantioanele de antrenare și zgomot; $\\sigma^2$: zgomotul pe care niciun model nu îl poate elimina')]),
    (T('\\textbf{Overfitting}: a flexible model learns the noise; small training error, large test error', '\\textbf{Overfitting}: un model flexibil învață zgomotul; eroare de antrenare mică, eroare de test mare'),
     [T('\\textbf{underfitting}: a rigid model misses the signal (large bias)', '\\textbf{underfitting}: un model rigid ratează semnalul (deplasare mare)')]),
    T('Every ML method has a complexity knob (depth, number of trees, penalty, hidden units); validation sets it', 'Fiecare metodă ML are un parametru de complexitate (adîncime, număr de arbori, penalizare, neuroni ascunși); validarea îl fixează')))

chart(T('Bias and variance of regression trees', 'Deplasarea și varianța arborilor de regresie'), 'tsa_ch9_bias_variance', 'TSA_ch9_validation', [
    T('Simulation: $y = \\sin(2\\pi x) + \\varepsilon$, $\\sigma = 0.3$, 80 points, 300 training samples; trees of depth 1 to 10',
      'Simulare: $y = \\sin(2\\pi x) + \\varepsilon$, $\\sigma = 0{,}3$, 80 de puncte, 300 de eșantioane de antrenare; arbori de adîncime 1 pînă la 10')],
    h='0.52\\textheight')

interp(('the U-shaped test error', 'erorii de test în formă de U'), [
    (T('Depth 1: bias$^2$ @{bv.1.b}, variance @{bv.1.v}; depth 10: bias$^2$ @{bv.10.b}, variance @{bv.10.v}', 'Adîncimea 1: deplasare$^2$ @{bv.1.b}, varianță @{bv.1.v}; adîncimea 10: deplasare$^2$ @{bv.10.b}, varianță @{bv.10.v}'),
     [T('the expected test error is smallest at depth @{bv.best} (@{bv.3.t}), above the noise variance 0.09', 'eroarea de test așteptată este minimă la adîncimea @{bv.best} (@{bv.3.t}), peste varianța zgomotului, 0,09')]),
    T('The training error falls to @{bv.10.tr} at depth 10: it cannot choose the complexity', 'Eroarea de antrenare scade la @{bv.10.tr} la adîncimea 10: nu poate alege complexitatea'),
    T('Ensembles (random forest, boosting) and penalties (ridge, lasso) reduce the variance without losing much bias', 'Ansamblurile (random forest, boosting) și penalizările (ridge, lasso) reduc varianța fără să piardă mult în deplasare')])

D.recap(('Validation without leakage', 'validare fără leakage'), [
    T('Walk-forward: train on the past, test on the next block, move on; random folds only for pure autoregressions', 'Walk-forward: antrenăm pe trecut, testăm pe blocul următor, mergem mai departe; grupuri aleatoare doar pentru autoregresii pure'),
    T('Leakage: future values in features, preprocessing on the full sample, overlapping targets, revised data', 'Leakage-ul: valori viitoare în variabile, preprocesare pe tot eșantionul, ținte suprapuse, date revizuite'),
    T('Test error = bias$^2$ + variance + noise; validation sets the complexity', 'Eroarea de test = deplasare$^2$ + varianță + zgomot; validarea fixează complexitatea')])

# =============================================================================
# 3. RIDGE ȘI LASSO
# =============================================================================
D.section('Regularised linear models: ridge and lasso', 'Modele liniare regularizate: ridge și lasso')

D.frame(T('Many lags, unstable least squares', 'Multe laguri, metoda celor mai mici pătrate instabilă'), items(
    (T('Linear model on $p$ standardised features: $\\hat y = \\beta_0 + \\sum_j \\beta_j x_j$; OLS (ordinary least squares) minimises $\\sum_t (y_t - \\hat y_t)^2$',
       'Model liniar pe $p$ variabile standardizate: $\\hat y = \\beta_0 + \\sum_j \\beta_j x_j$; OLS (ordinary least squares, metoda celor mai mici pătrate) minimizează $\\sum_t (y_t - \\hat y_t)^2$'),
     [T('neighbouring lags are highly correlated: OLS coefficients become large, of opposite signs and unstable from one window to the next', 'lagurile vecine sînt puternic corelate: coeficienții OLS devin mari, de semne opuse și instabili de la o fereastră la alta')]),
    (T('\\textbf{Standardise} first: $x_j \\to (x_j - \\bar x_j)/s_j$, with $\\bar x_j$, $s_j$ from the training window only', '\\textbf{Standardizăm} întîi: $x_j \\to (x_j - \\bar x_j)/s_j$, cu $\\bar x_j$, $s_j$ doar din fereastra de antrenare'),
     [T('otherwise the penalty treats features measured in GW and in dummies unequally', 'altfel penalizarea tratează inegal variabilele măsurate în GW și variabilele binare')]),
    T('Idea of regularisation: accept a little bias in exchange for a large fall in variance', 'Ideea regularizării: acceptăm puțină deplasare în schimbul unei scăderi mari a varianței')))

D.frame(T('Ridge and lasso', 'Ridge și lasso'), items(
    (T('\\textbf{Ridge} \\refHoerlK: $\\min_{\\beta} \\sum_t (y_t - \\beta_0 - \\mathbf{x}_t\'\\beta)^2 + \\lambda\\sum_j \\beta_j^2$', '\\textbf{Ridge} \\refHoerlK: $\\min_{\\beta} \\sum_t (y_t - \\beta_0 - \\mathbf{x}_t\'\\beta)^2 + \\lambda\\sum_j \\beta_j^2$'),
     [T('$\\lambda \\ge 0$: the penalty ($\\lambda = 0$: OLS); $\\mathbf{X}$: the matrix of standardised features; $\\mathbf{y}$: the targets; $\\mathbf{I}$: the identity matrix', '$\\lambda \\ge 0$: penalizarea ($\\lambda = 0$: OLS); $\\mathbf{X}$: matricea variabilelor standardizate; $\\mathbf{y}$: țintele; $\\mathbf{I}$: matricea identitate'),
      T('closed form $\\hat\\beta = (\\mathbf{X}\'\\mathbf{X} + \\lambda\\mathbf{I})^{-1}\\mathbf{X}\'\\mathbf{y}$; all coefficients shrink towards 0, none becomes exactly 0', 'formă închisă $\\hat\\beta = (\\mathbf{X}\'\\mathbf{X} + \\lambda\\mathbf{I})^{-1}\\mathbf{X}\'\\mathbf{y}$; toți coeficienții se contractă spre 0, niciunul nu devine exact 0')]),
    (T('\\textbf{Lasso} \\refTib\\ (least absolute shrinkage and selection operator): penalty $\\lambda\\sum_j |\\beta_j|$', '\\textbf{Lasso} \\refTib\\ (least absolute shrinkage and selection operator): penalizarea $\\lambda\\sum_j |\\beta_j|$'),
     [T('sets some coefficients exactly to 0: it selects lags', 'face unii coeficienți exact 0: selectează lagurile')]),
    (T('\\textbf{Worked example}, one standardised feature, $S_{xy} = \\sum x_ty_t = 90$, $S_{xx} = \\sum x_t^2 = 100$', '\\textbf{Exemplu rezolvat}, o singură variabilă standardizată, $S_{xy} = \\sum x_ty_t = 90$, $S_{xx} = \\sum x_t^2 = 100$'),
     [T('OLS $90/100 = @{rg.ols}$; ridge, $\\lambda = 50$: $90/150 = @{rg.r50}$; lasso, $\\lambda = 40$: $(90 - 20)/100 = @{rg.l40}$; lasso, $\\lambda \\ge 180$: exactly 0',
        'OLS $90/100 = @{rg.ols}$; ridge, $\\lambda = 50$: $90/150 = @{rg.r50}$; lasso, $\\lambda = 40$: $(90 - 20)/100 = @{rg.l40}$; lasso, $\\lambda \\ge 180$: exact 0'),
      T('lasso in one dimension: $\\hat\\beta = \\mathrm{sign}(S_{xy})\\max(|S_{xy}| - \\lambda/2, 0)/S_{xx}$ (soft thresholding)', 'lasso într-o dimensiune: $\\hat\\beta = \\mathrm{sign}(S_{xy})\\max(|S_{xy}| - \\lambda/2, 0)/S_{xx}$ (prag moale, soft thresholding)')]),
    T('$\\lambda$ is chosen by walk-forward validation (\\texttt{TimeSeriesSplit}), never by random folds', '$\\lambda$ se alege prin validare walk-forward (\\texttt{TimeSeriesSplit}), niciodată prin grupuri aleatoare')), 'footnotesize')

chart(T('Ridge and lasso paths on the load features', 'Traiectoriile ridge și lasso pentru variabilele consumului'), 'tsa_ch9_shrinkage', 'TSA_ch9_regularisation_trees', [
    T('Target: tomorrow\'s load; @{sh.p} standardised features; data up to @{ld.first} (@{sh.n} rows); dashed: the lasso penalty chosen by walk-forward cross-validation',
      'Ținta: consumul de mîine; @{sh.p} de variabile standardizate; date pînă la @{ld.first} (@{sh.n} de rînduri); linia întreruptă: penalizarea lasso aleasă prin validare walk-forward')],
    h='0.54\\textheight')

interp(('the coefficient paths', 'traiectoriilor coeficienților'), [
    (T('With a large penalty all coefficients are near 0; as $\\lambda$ falls, $y_t$ (lag0) enters first and dominates, with coefficient @{sh.b0} GW per standard deviation',
       'Cu o penalizare mare toți coeficienții sînt aproape de 0; cînd $\\lambda$ scade, $y_t$ (lag0) intră primul și domină, cu coeficientul @{sh.b0} GW la o abatere standard'),
     [T('then the weekday dummies and the holiday flags: the calendar carries most of the remaining signal', 'urmează variabilele binare ale zilelor și sărbătorile: calendarul poartă cea mai mare parte din semnalul rămas')]),
    T('At the chosen penalty the lasso keeps @{sh.kept} of @{sh.p} features: many lags are redundant once $y_t$, the weekday and the holidays are in', 'La penalizarea aleasă, lasso păstrează @{sh.kept} din @{sh.p} de variabile: multe laguri sînt redundante cînd $y_t$, ziua săptămînii și sărbătorile sînt incluse'),
    T('Ridge keeps everything but shrinks it smoothly; on the test origins of Section 6 ridge is the best ML model', 'Ridge păstrează totul, dar contractă lin; pe originile de test din secțiunea 6, ridge este cel mai bun model ML')])

D.recap(('Ridge and lasso', 'ridge și lasso'), [
    T('Penalised least squares: ridge $\\lambda\\sum\\beta_j^2$ shrinks; lasso $\\lambda\\sum|\\beta_j|$ shrinks and selects', 'Cele mai mici pătrate penalizate: ridge $\\lambda\\sum\\beta_j^2$ contractă; lasso $\\lambda\\sum|\\beta_j|$ contractă și selectează'),
    T('Standardise inside each training window; choose $\\lambda$ walk-forward', 'Standardizăm în fiecare fereastră de antrenare; alegem $\\lambda$ prin walk-forward'),
    T('A regularised AR with calendar dummies is a strong ML baseline', 'Un AR regularizat cu variabile de calendar este un reper ML puternic')])

# =============================================================================
# 4. ARBORI
# =============================================================================
D.section('Trees, random forest and gradient boosting', 'Arbori, random forest și gradient boosting')

D.frame(T('Regression trees', 'Arborii de regresie'), items(
    (T('A \\textbf{regression tree} splits the feature space with questions ``$x_j \\le c$?\'\' and predicts, in each final region (leaf), the mean target of its training rows',
       'Un \\textbf{arbore de regresie} împarte spațiul variabilelor prin întrebări „$x_j \\le c$?” și prognozează, în fiecare regiune finală (frunză), media țintei din rîndurile de antrenare ale acesteia'),
     [T('the split is chosen greedily: the $(j, c)$ with the smallest sum of squared errors of the two children', 'împărțirea se alege lacom (greedy): perechea $(j, c)$ cu cea mai mică sumă a pătratelor erorilor celor doi descendenți')]),
    (T('Strengths: nonlinearity and interactions (weekday $\\times$ season) without specifying them; no scaling needed', 'Puncte tari: neliniaritate și interacțiuni (ziua săptămînii $\\times$ sezon) fără să le specificăm; nu necesită scalare'),
     [T('weaknesses: a single tree is unstable (high variance) and piecewise constant', 'puncte slabe: un singur arbore este instabil (varianță mare) și constant pe porțiuni')]),
    (T('\\textbf{A tree cannot extrapolate}: its forecast is always a mean of training targets', '\\textbf{Un arbore nu poate extrapola}: prognoza lui este întotdeauna o medie a unor ținte din antrenare'),
     [T('for trending series: model differences, growth rates or the deviation from a trend, not levels', 'pentru serii cu trend: modelăm diferențe, rate de creștere sau abaterea de la trend, nu niveluri')])))

chart(T('A depth-2 tree for tomorrow\'s load', 'Un arbore de adîncime 2 pentru consumul de mîine'), 'tsa_ch9_tree', 'TSA_ch9_regularisation_trees', [
    T('Features: $y_t$ (lag0), the same weekday last week (same\\_dow), weekend and holiday flags; data up to @{ld.first}; at least 20 days per leaf',
      'Variabile: $y_t$ (lag0), aceeași zi a săptămînii trecute (same\\_dow), indicatori de weekend și de sărbătoare; date pînă la @{ld.first}; cel puțin 20 de zile în fiecare frunză')],
    h='0.5\\textheight')

interp(('the tree', 'arborelui'), [
    (T('First question: was the same weekday last week below @{tr.thr} GW? It separates weekends and holidays from working days', 'Prima întrebare: a fost aceeași zi a săptămînii trecute sub @{tr.thr} GW? Separă weekendurile și sărbătorile de zilele lucrătoare'),
     [T('four leaves: @{tr.l0}, @{tr.l1}, @{tr.l2} and @{tr.l3} GW, with @{tr.n0}, @{tr.n1}, @{tr.n2} and @{tr.n3} days', 'patru frunze: @{tr.l0}, @{tr.l1}, @{tr.l2} și @{tr.l3} GW, cu @{tr.n0}, @{tr.n1}, @{tr.n2} și @{tr.n3} zile')]),
    T('Four numbers already explain $R^2$ = @{tr.r2} of the variance in training; deeper trees add detail and variance', 'Patru numere explică deja $R^2$ = @{tr.r2} din varianță la antrenare; arborii mai adînci adaugă detalii și varianță'),
    T('The forecast for any day is one of these four means: a tree never forecasts outside the range of the training targets', 'Prognoza pentru orice zi este una dintre aceste patru medii: un arbore nu prognozează niciodată în afara intervalului țintelor de antrenare')])

D.frame(T('Random forest', 'Random forest'), items(
    (T('\\textbf{Bagging} (bootstrap aggregation): grow $B$ deep trees on bootstrap samples of the rows and average them', '\\textbf{Bagging} (bootstrap aggregation): creștem $B$ arbori adînci pe eșantioane bootstrap ale rîndurilor și facem media lor'),
     [T('averaging $B$ trees with correlation $\\rho$ and variance $\\sigma^2$ gives variance $\\rho\\sigma^2 + (1 - \\rho)\\sigma^2/B$', 'media a $B$ arbori cu corelația $\\rho$ și varianța $\\sigma^2$ are varianța $\\rho\\sigma^2 + (1 - \\rho)\\sigma^2/B$')]),
    (T('\\textbf{Random forest} \\refBreiman: at each split, only a random subset of features is tried (\\texttt{max\\_features})', '\\textbf{Random forest} \\refBreiman: la fiecare împărțire se încearcă doar o submulțime aleatoare de variabile (\\texttt{max\\_features})'),
     [T('this decorrelates the trees ($\\rho$ falls), so the average has a lower variance', 'astfel arborii se decorelează ($\\rho$ scade), iar media are o varianță mai mică')]),
    (T('Hyperparameters: number of trees (more is never worse, only slower), \\texttt{max\\_features}, minimum leaf size', 'Hiperparametri: numărul de arbori (un număr mai mare nu înrăutățește rezultatul, doar mărește timpul de calcul), \\texttt{max\\_features}, dimensiunea minimă a frunzei'),
     [T('the bootstrap ignores time order inside the training window; that is fine, because the validation is walk-forward', 'bootstrap-ul ignoră ordinea în timp în fereastra de antrenare; nu este o problemă, deoarece validarea este walk-forward')])))

D.frame(T('Gradient boosting', 'Gradient boosting'), items(
    (T('\\textbf{Boosting} \\refFriedman: add small trees one at a time, each fitted to the residuals (the negative gradient of the loss) of the current model',
       '\\textbf{Boosting} \\refFriedman: adăugăm cîte un arbore mic, fiecare estimat pe reziduurile (gradientul negativ al funcției de pierdere) modelului curent'),
     [T('$F_m(\\mathbf{x}) = F_{m-1}(\\mathbf{x}) + \\nu\\,g_m(\\mathbf{x})$; $F_m$: the model after $m$ trees; $g_m$: the $m$-th small tree; $\\nu$: the \\textbf{learning rate} (0.01--0.3)', '$F_m(\\mathbf{x}) = F_{m-1}(\\mathbf{x}) + \\nu\\,g_m(\\mathbf{x})$; $F_m$: modelul după $m$ arbori; $g_m$: al $m$-lea arbore mic; $\\nu$: \\textbf{rata de învățare} (0,01--0,3)')]),
    (T('\\textbf{Worked example}: targets $5, 6, 8$; start with the mean $F_0 = @{gb.f0}$; residuals @{gb.res}', '\\textbf{Exemplu rezolvat}: țintele $5, 6, 8$; pornim de la medie, $F_0 = @{gb.f0}$; reziduurile @{gb.res}'),
     [T('a stump that isolates the third point predicts $@{gb.r3}$ there; with $\\nu = @{gb.lr}$: $F_1 = @{gb.f0} + @{gb.lr} \\cdot @{gb.r3} = @{gb.f1}$', 'un arbore cu o singură împărțire care izolează al treilea punct prognozează acolo $@{gb.r3}$; cu $\\nu = @{gb.lr}$: $F_1 = @{gb.f0} + @{gb.lr} \\cdot @{gb.r3} = @{gb.f1}$')]),
    (T('Fast implementations bin the features into histograms: XGBoost \\refXGB, LightGBM \\refLGBM, \\texttt{HistGradientBoostingRegressor} in scikit-learn',
       'Implementările rapide grupează variabilele în histograme: XGBoost \\refXGB, LightGBM \\refLGBM, \\texttt{HistGradientBoostingRegressor} în scikit-learn'),
     [T('below we write GB for gradient boosting; the winners of M5 (Section 10) used LightGBM', 'mai departe notăm GB pentru gradient boosting; cîștigătorii M5 (secțiunea 10) au folosit LightGBM'),
      T('hyperparameters: number of trees with \\textbf{early stopping} on validation data, tree size, $\\nu$, minimum leaf size', 'hiperparametri: numărul de arbori cu \\textbf{oprire timpurie} pe datele de validare, dimensiunea arborilor, $\\nu$, dimensiunea minimă a frunzei')])), 'footnotesize')

chart(T('One tree, a forest and boosting on the load', 'Un arbore, un random forest și boosting pe datele de consum'), 'tsa_ch9_ensembles', 'TSA_ch9_regularisation_trees', [
    T('Tomorrow\'s load; training up to 31 December 2024 (@{en.n_tr} days), validation January--June 2025 (@{en.n_va} days); dotted: the seasonal naive forecast',
      'Consumul de mîine; antrenare pînă la 31 decembrie 2024 (@{en.n_tr} de zile), validare ianuarie--iunie 2025 (@{en.n_va} de zile); linia punctată: prognoza naivă sezonieră')],
    h='0.5\\textheight')

interp(('the ensembles', 'ansamblurilor'), [
    (T('Single tree: best validation MSE (mean squared error) @{en.tree_best} at depth @{en.tree_best_d}; random forest: @{en.rf_best} at depth @{en.rf_best_d}',
       'Un singur arbore: cea mai bună MSE (mean squared error, eroarea pătratică medie) la validare, @{en.tree_best}, la adîncimea @{en.tree_best_d}; random forest: @{en.rf_best} la adîncimea @{en.rf_best_d}'),
     [T('averaging removes most of the variance; the seasonal naive forecast has @{en.naive}', 'media elimină cea mai mare parte a varianței; prognoza naivă sezonieră are @{en.naive}')]),
    (T('Boosting with $\\nu = 0.3$: minimum @{en.gb03_min} after @{en.gb03_min_it} trees; with $\\nu = 0.05$: @{en.gb005_min} after @{en.gb005_min_it} trees', 'Boosting cu $\\nu = 0{,}3$: minimul @{en.gb03_min} după @{en.gb03_min_it} de arbori; cu $\\nu = 0{,}05$: @{en.gb005_min} după @{en.gb005_min_it} de arbori'),
     [T('a small learning rate needs more trees and usually ends slightly lower; early stopping picks the number of trees', 'o rată de învățare mică are nevoie de mai mulți arbori și ajunge de obicei puțin mai jos; oprirea timpurie alege numărul de arbori')])])

chart(T('Trees cannot extrapolate: the Romanian price level', 'Arborii nu pot extrapola: nivelul prețurilor în România'), 'tsa_ch9_extrapolation', 'TSA_ch9_regularisation_trees', [
    T('HICP (harmonised index of consumer prices) of Romania, Eurostat, 2015 = 100; one-month-ahead forecasts from random forests trained up to December 2020, on lagged levels or on monthly log changes',
      'IAPC (indicele armonizat al prețurilor de consum) al României, Eurostat, 2015 = 100; prognoze cu o lună înainte din random forest antrenate pînă în decembrie 2020, pe laguri ale nivelurilor sau pe variațiile lunare logaritmice')],
    h='0.5\\textheight')

interp(('the extrapolation failure', 'eșecului de extrapolare'), [
    (T('The highest index in training was @{ex.max}; by @{ex.lastd} the index reached @{ex.last}', 'Cel mai mare indice din antrenare a fost @{ex.max}; în @{ex.lastd} indicele a ajuns la @{ex.last}'),
     [T('the forest on levels stays near @{ex.fl} for five years: RMSE @{ex.rl} index points', 'random forest pe niveluri rămîne în jurul valorii @{ex.fl} timp de cinci ani: RMSE @{ex.rl} puncte de indice')]),
    T('The same forest on monthly changes, then cumulated, follows the index: RMSE @{ex.rc}', 'Același random forest pe variațiile lunare, apoi cumulate, urmărește indicele: RMSE @{ex.rc}'),
    T('Chapter 3 again: difference or transform non-stationary series before a tree sees them', 'Din nou Capitolul 3: diferențiem sau transformăm seriile nestaționare înainte ca un arbore să le vadă')])

D.frame(T('The features a model uses', 'Variabilele folosite de model'), items(
    (T('\\textbf{Permutation importance}: shuffle one feature in the test data and measure how much the error grows', '\\textbf{Importanța prin permutare}: amestecăm o variabilă în datele de test și măsurăm cît crește eroarea'),
     [T('model-agnostic; measured out of sample; correlated features share (and hide) their importance', 'nu depinde de tipul modelului; se măsoară în afara eșantionului; variabilele corelate își împart (și își ascund) importanța')]),
    T('Impurity-based importance of trees (MDI, mean decrease impurity) is computed in training and favours continuous features: prefer permutation', 'Importanța calculată din impuritatea arborilor (MDI, mean decrease impurity) se obține la antrenare și favorizează variabilele continue: preferăm permutarea'),
    T('Importance is not causality: it describes the model, not the electricity system', 'Importanța nu înseamnă cauzalitate: descrie modelul, nu sistemul energetic')))

chart(T('Permutation importance of gradient boosting', 'Importanța prin permutare pentru gradient boosting'), 'tsa_ch9_importance', 'TSA_ch9_load_forecasting', [
    T('Direct GB models for horizons 1 and 14 days; trained up to 31 December 2024, evaluated on January--June 2025; 20 random permutations per feature',
      'Modele GB directe pentru orizonturile de 1 și 14 zile; antrenate pînă la 31 decembrie 2024, evaluate pe ianuarie--iunie 2025; 20 de permutări aleatoare pentru fiecare variabilă')],
    h='0.5\\textheight')

interp(('the importances', 'importanțelor'), [
    (T('One day ahead: @{im.1a} (+@{im.1av} MW of MAE when shuffled) and @{im.1b} (+@{im.1bv} MW) dominate', 'Cu o zi înainte: @{im.1a} (+@{im.1av} MW de MAE cînd este amestecată) și @{im.1b} (+@{im.1bv} MW) domină'),
     [T('yesterday and the same weekday last week: the persistence and the weekly cycle', 'ziua de ieri și aceeași zi a săptămînii trecute: persistența și ciclul săptămînal')]),
    (T('Fourteen days ahead: @{im.14a} (+@{im.14av} MW) and @{im.14b} (+@{im.14bv} MW)', 'Cu paisprezece zile înainte: @{im.14a} (+@{im.14av} MW) și @{im.14b} (+@{im.14bv} MW)'),
     [T('the annual cycle (cos1, the Fourier term) gains weight as the recent values become less informative', 'ciclul anual (cos1, termenul Fourier) cîștigă importanță pe măsură ce valorile recente devin mai puțin informative')])])

D.recap(('Trees and ensembles', 'arbori și ansambluri'), [
    T('A tree is a piecewise-constant function found by greedy splits; it cannot extrapolate', 'Un arbore este o funcție constantă pe porțiuni, găsită prin împărțiri lacome; nu poate extrapola'),
    T('Random forest averages decorrelated deep trees (less variance); boosting adds small trees on residuals (less bias)', 'Random forest face media unor arbori adînci decorelați (mai puțină varianță); boosting adaugă arbori mici pe reziduuri (mai puțină deplasare)'),
    T('GB (gradient boosting, LightGBM, XGBoost, \\texttt{HistGradientBoosting}): learning rate, number of trees, early stopping', 'GB (gradient boosting, LightGBM, XGBoost, \\texttt{HistGradientBoosting}): rata de învățare, numărul de arbori, oprirea timpurie'),
    T('Permutation importance measures what the model uses, out of sample', 'Importanța prin permutare măsoară ce folosește modelul, în afara eșantionului')])

# =============================================================================
# 5. REȚELE NEURONALE
# =============================================================================
D.section('Neural networks', 'Rețele neuronale')

D.frame(T('The multilayer perceptron', 'Perceptronul multistrat'), twocol(
    ph('perceptron', T('Mark I Perceptron, Cornell (1960): the first trainable neural network, built in hardware', 'Mark I Perceptron, Cornell (1960): prima rețea neuronală antrenabilă, construită fizic'), h='0.5\\textheight'),
    items((T('\\textbf{MLP} (multilayer perceptron), one hidden layer with $q$ neurons:', '\\textbf{MLP} (multilayer perceptron, perceptron multistrat), un strat ascuns cu $q$ neuroni:'),
           [T('$h_j = g(b_j + \\mathbf{w}_j\'\\mathbf{x})$, $j = 1, \\dots, q$; \\quad $\\hat y = c + \\sum_j v_jh_j$', '$h_j = g(b_j + \\mathbf{w}_j\'\\mathbf{x})$, $j = 1, \\dots, q$; \\quad $\\hat y = c + \\sum_j v_jh_j$'),
            T('$\\mathbf{x}$: the inputs; $\\mathbf{w}_j$, $b_j$: the weights and the bias of neuron $j$; $h_j$: its output; $v_j$, $c$: the output weights and constant', '$\\mathbf{x}$: intrările; $\\mathbf{w}_j$, $b_j$: ponderile și termenul liber ale neuronului $j$; $h_j$: ieșirea lui; $v_j$, $c$: ponderile și termenul liber ale ieșirii'),
            T('$g$: activation function (ReLU $\\max(0, z)$, tanh, logistic); with $g(z) = z$ the network is linear again', '$g$: funcția de activare (ReLU $\\max(0, z)$, tanh, logistică); cu $g(z) = z$ rețeaua redevine liniară')]),
          (T('Universal approximation \\refHornik: enough hidden neurons approximate any continuous function', 'Aproximarea universală \\refHornik: suficienți neuroni ascunși aproximează orice funcție continuă'),
           [T('parameters: 3 inputs and 5 neurons already give $3 \\cdot 5 + 5 + 5 + 1 = @{mlp.p}$ weights', 'parametri: 3 intrări și 5 neuroni dau deja $3 \\cdot 5 + 5 + 5 + 1 = @{mlp.p}$ de ponderi')])),
    wl='0.36', wr='0.62'), 'footnotesize')

chart(T('A small network and its activation functions', 'O rețea mică și funcțiile ei de activare'), 'tsa_ch9_mlp', 'TSA_ch9_neural_networks', [
    T('Left: 3 inputs (yesterday, the same weekday last week, a holiday flag), 5 hidden neurons, one output; right: three activation functions',
      'Stînga: 3 intrări (ziua de ieri, aceeași zi a săptămînii trecute, indicatorul de sărbătoare), 5 neuroni ascunși, o ieșire; dreapta: trei funcții de activare')],
    h='0.48\\textheight')

D.frame(T('Training a network', 'Antrenarea unei rețele'), items(
    (T('Minimise the training loss by \\textbf{gradient descent}; the gradients come from back-propagation \\refRumelhart', 'Minimizăm pierderea la antrenare prin \\textbf{coborîre pe gradient}; gradienții se obțin prin retropropagare (back-propagation) \\refRumelhart'),
     [T('SGD (stochastic gradient descent) and Adam use small random batches; one pass through the data is an epoch', 'SGD (stochastic gradient descent) și Adam folosesc loturi mici aleatoare; o trecere prin date este o epocă')]),
    (T('The loss surface has many local minima: results depend on the random starting weights', 'Suprafața pierderii are multe minime locale: rezultatele depind de ponderile inițiale aleatoare'),
     [T('fix the seed, and average a few networks with different seeds (an ensemble)', 'fixăm sămînța și facem media cîtorva rețele cu semințe diferite (un ansamblu)')]),
    (T('Regularisation: an L2 penalty on the weights, early stopping, dropout (randomly switching off neurons)', 'Regularizare: o penalizare L2 pe ponderi, oprirea timpurie, dropout (dezactivarea aleatoare a unor neuroni)'),
     [T('inputs and targets must be standardised with training-window statistics', 'intrările și țintele trebuie standardizate cu statisticile ferestrei de antrenare')]),
    T('Small data (a few thousand rows) favour small networks; deep networks need many series or long high-frequency data', 'Datele puține (cîteva mii de rînduri) favorizează rețelele mici; rețelele adînci au nevoie de multe serii sau de date lungi, de frecvență înaltă')))

D.frame(T('Recurrent networks and the LSTM (1/2): the recurrent network', 'Rețele recurente și LSTM (1/2): rețeaua recurentă'), items(
    (T('\\textbf{RNN} (recurrent neural network): a hidden state carries the past forward, with the same weights at every step',
       '\\textbf{RNN} (recurrent neural network, rețea neuronală recurentă): o stare ascunsă duce trecutul mai departe, cu aceleași ponderi la fiecare pas'),
     [T('$h_t = \\tanh(Wh_{t-1} + Ux_t + b)$: the new state mixes the previous state and the new input', '$h_t = \\tanh(Wh_{t-1} + Ux_t + b)$: noua stare combină starea anterioară și noua intrare'),
      T('$x_t$: the input at $t$; $h_t$: the hidden state; $W$, $U$, $b$: weights and bias, shared by all steps; $\\tanh$ keeps each component in $(-1, 1)$', '$x_t$: intrarea la momentul $t$; $h_t$: starea ascunsă; $W$, $U$, $b$: ponderile și termenul liber, comune tuturor pașilor; $\\tanh$ ține fiecare componentă în $(-1, 1)$'),
      T('the forecast is read from the last state, for example $\\hat y_{t+1} = v\'h_t + c$', 'prognoza se citește din ultima stare, de exemplu $\\hat y_{t+1} = v\'h_t + c$')]),
    (T('The difficulty: training through many steps multiplies many derivatives', 'Dificultatea: antrenarea prin mulți pași înmulțește multe derivate'),
     [T('the gradients vanish or explode: a plain RNN forgets what happened more than a few steps back', 'gradienții dispar sau explodează: un RNN simplu uită ce s-a întîmplat cu mai mult de cîțiva pași în urmă')]),
    (T('Honest summary \\refHBB', 'Concluzia sintezei \\refHBB'),
     [T('on single, short series, recurrent networks rarely beat ETS or ARIMA', 'pe serii unice și scurte, rețelele recurente sînt rareori mai precise decît ETS sau ARIMA'),
      T('they shine with many related series and enough data (global models, \\refDeepAR)', 'sînt utile mai ales cu multe serii înrudite și suficiente date (modele globale, \\refDeepAR)')])), 'footnotesize')

D.frame(T('Recurrent networks and the LSTM (2/2): the LSTM cell', 'Rețele recurente și LSTM (2/2): celula LSTM'), items(
    (T('\\textbf{LSTM} (long short-term memory) \\refLSTM: a cell state $c_t$ updated additively, controlled by three gates', '\\textbf{LSTM} (long short-term memory) \\refLSTM: o stare a celulei $c_t$ actualizată aditiv, controlată de trei porți'),
     [T('the gates: $f_t = \\sigma(W_f[h_{t-1}, x_t] + b_f)$ (forget), $i_t = \\sigma(W_i[h_{t-1}, x_t] + b_i)$ (input), $o_t = \\sigma(W_o[h_{t-1}, x_t] + b_o)$ (output)', 'porțile: $f_t = \\sigma(W_f[h_{t-1}, x_t] + b_f)$ (uitare), $i_t = \\sigma(W_i[h_{t-1}, x_t] + b_i)$ (intrare), $o_t = \\sigma(W_o[h_{t-1}, x_t] + b_o)$ (ieșire)'),
      T('the candidate new content: $\\tilde c_t = \\tanh(W_c[h_{t-1}, x_t] + b_c)$', 'conținutul nou propus: $\\tilde c_t = \\tanh(W_c[h_{t-1}, x_t] + b_c)$'),
      T('$\\sigma(z) = 1/(1 + e^{-z}) \\in (0, 1)$: the logistic function; $[h_{t-1}, x_t]$: the two vectors stacked; $W_f, W_i, W_o, W_c$ and $b_f, b_i, b_o, b_c$: weights and biases', '$\\sigma(z) = 1/(1 + e^{-z}) \\in (0, 1)$: funcția logistică; $[h_{t-1}, x_t]$: cei doi vectori puși unul sub altul; $W_f, W_i, W_o, W_c$ și $b_f, b_i, b_o, b_c$: ponderile și termenii liberi')]),
    (T('The updates of the cell and of the output', 'Actualizarea celulei și a ieșirii'),
     [T('$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde c_t$: with $f_t$ close to 1 the old memory survives many steps; $i_t$ sets how much new content enters', '$c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde c_t$: cu $f_t$ apropiat de 1 memoria veche se păstrează pe mulți pași; $i_t$ stabilește cît conținut nou intră'),
      T('$h_t = o_t \\odot \\tanh(c_t)$: the output gate decides which part of the memory is passed on; $\\odot$: element-wise product', '$h_t = o_t \\odot \\tanh(c_t)$: poarta de ieșire decide ce parte a memoriei este transmisă mai departe; $\\odot$: produsul element cu element')]),
    T('Each gate is a small logistic layer: an LSTM with 32 units and 8 inputs has about 5\\,400 weights', 'Fiecare poartă este un mic strat logistic: un LSTM cu 32 de unități și 8 intrări are circa 5\\,400 de ponderi')), 'footnotesize')

chart(T('A recurrent network and the LSTM cell', 'O rețea recurentă și celula LSTM'), 'tsa_ch9_rnn', 'TSA_ch9_neural_networks', [
    T('Left: the RNN unrolled over three steps; right: the LSTM cell state (red) and its three gates; $\\odot$: element-wise product',
      'Stînga: RNN desfășurată pe trei pași; dreapta: starea celulei LSTM (roșu) și cele trei porți; $\\odot$: produsul element cu element')],
    h='0.48\\textheight')

D.frame(T('Sepp Hochreiter and the LSTM', 'Sepp Hochreiter și LSTM'), twocol(
    ph('hochreiter', T('Sepp Hochreiter (2015), Johannes Kepler University Linz', 'Sepp Hochreiter (2015), Universitatea Johannes Kepler din Linz'), h='0.55\\textheight'),
    items(T('In his 1991 diploma thesis in Munich, Hochreiter analysed why gradients vanish in recurrent networks', 'În lucrarea de diplomă din 1991, la München, Hochreiter a analizat de ce gradienții dispar în rețelele recurente'),
          (T('With Jürgen Schmidhuber he proposed the LSTM \\refLSTM', 'Împreună cu Jürgen Schmidhuber a propus LSTM \\refLSTM'),
           [T('one of the most cited papers in neural computing', 'una dintre cele mai citate lucrări despre rețele neuronale')]),
          T('LSTMs powered speech recognition and machine translation before transformers (Chapter 11)', 'LSTM a stat la baza recunoașterii vorbirii și a traducerii automate înainte de transformer (Capitolul 11)')),
    wl='0.38', wr='0.6'))

D.recap(('Neural networks', 'rețele neuronale'), [
    T('MLP: layers of $g(b + \\mathbf{w}\'\\mathbf{x})$; trained by gradient descent; seed, scaling, L2 penalty, early stopping', 'MLP: straturi de $g(b + \\mathbf{w}\'\\mathbf{x})$; antrenat prin coborîre pe gradient; sămînța, scalarea, penalizarea L2, oprirea timpurie'),
    T('RNN and LSTM process sequences; the LSTM cell state avoids vanishing gradients', 'RNN și LSTM procesează secvențe; starea celulei LSTM evită dispariția gradienților'),
    T('Small data favour small models; neural networks need many series or long data', 'Datele puține favorizează modelele mici; rețelele neuronale au nevoie de multe serii sau de date lungi')])

# =============================================================================
# 6. APLICAȚIE: CONSUMUL DE ENERGIE ELECTRICĂ
# =============================================================================
D.section('Application: electricity load against Chapter 4', 'Aplicație: consumul de energie electrică și Capitolul 4')

D.frame(T('The forecasting exercise', 'Exercițiul de prognoză'), twocol(
    ph('pylon', T('750 kV line at Mărășești: the Romanian transmission grid', 'Linia de 750 kV de la Mărășești: rețeaua de transport a României'), h='0.45\\textheight'),
    items((T('The evaluation design of Chapter 4: @{ld.no} forecast origins every 14 days, @{ld.first} to @{ld.last}', 'Schema de evaluare din Capitolul 4: @{ld.no} de origini ale prognozei, la fiecare 14 zile, de la @{ld.first} la @{ld.last}'),
           [T('horizon 1--14 days; expanding window from 1 January 2022', 'orizontul 1--14 zile; fereastră extinsă de la 1 ianuarie 2022'),
            T('statistical models (Chapter 4): seasonal naive, ETS, SARIMA, DHR (dynamic harmonic regression), their combination', 'modele statistice (Capitolul 4): naivă sezonieră, ETS, SARIMA, DHR (regresie armonică dinamică), combinația lor')]),
          (T('ML models, refitted at each origin:', 'Modele ML, reestimate la fiecare origine:'),
           [T('direct: ridge, random forest (200 trees), GB (300 trees, $\\nu = 0.05$), one model per horizon', 'directe: ridge, random forest (200 de arbori), GB (300 de arbori, $\\nu = 0{,}05$), un model pentru fiecare orizont'),
            T('recursive: GB on the one-step table; MIMO: an MLP (3 networks, 32 neurons) and an LSTM (56 days in, 14 out)', 'recursiv: GB pe tabelul pe un pas; MIMO: un MLP (3 rețele, 32 de neuroni) și un LSTM (56 de zile la intrare, 14 la ieșire)')]),
          T('MASE scale: in-sample MAE of the weekly seasonal naive method, @{ld.scale} MW', 'Scala MASE: MAE în eșantion a metodei naive sezoniere săptămînale, @{ld.scale} MW')),
    wl='0.34', wr='0.64'), 'footnotesize')

D.frame(T('Results: 26 origins, 14 horizons', 'Rezultatele: 26 de origini, 14 orizonturi'), table(
    'lrrrr', T('\\textbf{Model}', '\\textbf{Model}') + ' & MAE (GW) & RMSE (GW) & MASE & ' + T('DM p, against DHR', 'DM p, față de DHR'),
    [T(m, {'Seasonal naive': 'Naivă sezonieră', 'Combination': 'Combinația', 'GB direct': 'GB direct', 'GB recursive': 'GB recursiv'}.get(m, m))
     + f' & @{{ld.{KEY[m]}.MAE}} & @{{ld.{KEY[m]}.RMSE}} & @{{ld.{KEY[m]}.MASE}} & ' + ('--' if m == 'DHR' else f'@{{dm.{KEY[m]}.DHR}}')
     for m in MODELS], size='scriptsize') + items(
    T('DM: Diebold--Mariano test on the origin-average absolute errors (@{ld.no} values), with the Harvey--Leybourne--Newbold correction \\refHLN', 'DM: testul Diebold--Mariano pe erorile absolute medii pe origine (@{ld.no} de valori), cu corecția Harvey--Leybourne--Newbold \\refHLN')), 'footnotesize')

chart(T('MASE and accuracy by horizon', 'MASE și acuratețea pe orizonturi'), 'tsa_ch9_load_results', 'TSA_ch9_load_forecasting', [
    T('Left: MASE of the eleven models (blue: Chapter 4; red: ML); right: MAE by horizon for six models', 'Stînga: MASE pentru cele unsprezece modele (albastru: Capitolul 4; roșu: ML); dreapta: MAE pe orizonturi pentru șase modele')],
    h='0.76\\textheight')

interp(('the load results', 'rezultatelor pentru consum'), [
    (T('DHR (MASE @{ld.DHR.MASE}) and ridge (@{ld.Ridge.MASE}) are the most accurate; the difference is not significant (p @{dm.Ridge.DHR})', 'DHR (MASE @{ld.DHR.MASE}) și ridge (@{ld.Ridge.MASE}) sînt cele mai precise; diferența nu este semnificativă (p @{dm.Ridge.DHR})'),
     [T('both are linear models with the right calendar features; the trees and networks do not beat them', 'ambele sînt modele liniare cu variabilele de calendar potrivite; arborii și rețelele nu le depășesc')]),
    (T('GB direct (@{ld.GBdirect.MASE}), MLP (@{ld.MLP.MASE}), random forest (@{ld.RF.MASE}) beat the seasonal naive method clearly (p @{dm.GBdirect.Seasonalnaive}) and SARIMA, ETS',
       'GB direct (@{ld.GBdirect.MASE}), MLP (@{ld.MLP.MASE}), random forest (@{ld.RF.MASE}) depășesc clar metoda naivă sezonieră (p @{dm.GBdirect.Seasonalnaive}), SARIMA și ETS'),
     [T('direct and recursive GB are close (@{ld.GBdirect.MASE} and @{ld.GBrecursive.MASE}); the LSTM (@{ld.LSTM.MASE}) is worse than every model except the naive, SARIMA and ETS', 'GB direct și recursiv sînt apropiate (@{ld.GBdirect.MASE} și @{ld.GBrecursive.MASE}); LSTM (@{ld.LSTM.MASE}) este mai slab decît toate modelele, cu excepția celui naiv, SARIMA și ETS')]),
    T('Lesson: features matter more than the algorithm; with four years of daily data, a well-specified linear model is hard to beat', 'Lecția: variabilele contează mai mult decît algoritmul; cu patru ani de date zilnice, un model liniar bine specificat este greu de depășit')])

chart(T('Forecasts around Orthodox Easter 2026', 'Prognozele în jurul Paștelui ortodox din 2026'), 'tsa_ch9_load_forecasts', 'TSA_ch9_load_forecasting', [
    T('Origin @{ld.fco}; 14-day forecasts; MAE in this window: DHR @{ld.fc.DHR} MW, GB direct @{ld.fc.GBdirect} MW, GB recursive @{ld.fc.GBrecursive} MW, LSTM @{ld.fc.LSTM} MW',
      'Originea @{ld.fco}; prognoze pe 14 zile; MAE în această fereastră: DHR @{ld.fc.DHR} MW, GB direct @{ld.fc.GBdirect} MW, GB recursiv @{ld.fc.GBrecursive} MW, LSTM @{ld.fc.LSTM} MW')],
    h='0.5\\textheight')

interp(('the Easter window', 'ferestrei de Paște'), [
    (T('All models see the Easter flag, but the trees have only four Easters in training: they predict a dip that is too shallow', 'Toate modelele văd indicatorul de Paște, dar arborii au doar patru sărbători de Paște la antrenare: prognozează o scădere prea mică'),
     [T('DHR estimates one Easter coefficient from the same four episodes and pools the information more efficiently', 'DHR estimează un singur coeficient pentru Paște din aceleași patru episoade și folosește informația mai eficient')]),
    T('The LSTM sees no holiday flag: it repeats the weekly pattern and misses the holiday entirely', 'LSTM nu vede indicatorul de sărbătoare: repetă tiparul săptămînal și nu surprinde deloc sărbătoarea'),
    T('Rare events need either structure (a regression coefficient) or many series that share the event (a global model)', 'Evenimentele rare cer fie structură (un coeficient de regresie), fie multe serii care au în comun evenimentul (un model global)')])

D.recap(('Electricity load', 'consumul de energie electrică'), [
    T('Same origins, same horizon, same errors: ML models enter the comparison of Chapter 4', 'Aceleași origini, același orizont, aceleași erori: modelele ML intră în comparația din Capitolul 4'),
    T('Ridge equals DHR; GB, random forest and the MLP beat SARIMA and ETS; the LSTM does not', 'Ridge are aceeași precizie ca DHR; GB, random forest și MLP depășesc SARIMA și ETS; LSTM nu'),
    T('Calendar features and holidays decide the ranking', 'Variabilele de calendar și sărbătorile decid clasamentul')])

# =============================================================================
# 7. INTERVALE DE PROGNOZĂ
# =============================================================================
D.section('Prediction intervals', 'Intervale de prognoză')

D.frame(T('Quantile regression and the pinball loss', 'Regresia cuantilică și funcția de pierdere pinball'), items(
    (T('A point forecast is not enough: a grid operator needs the load it will not exceed with 95\\% probability', 'O prognoză punctuală nu ajunge: operatorul rețelei are nevoie de consumul pe care nu îl va depăși cu probabilitatea de 95\\%'),
     [T('the $\\tau$-quantile $q_\\tau$ of $y_{t+h}$: $P(y_{t+h} \\le q_\\tau) = \\tau$; a 90\\% interval: $[q_{0.05}, q_{0.95}]$', 'cuantila $\\tau$, $q_\\tau$, a lui $y_{t+h}$: $P(y_{t+h} \\le q_\\tau) = \\tau$; un interval de 90\\%: $[q_{0.05}, q_{0.95}]$')]),
    (T('\\textbf{Pinball loss}: $L_\\tau(y, q) = \\tau(y - q)$ if $y \\ge q$, and $(1 - \\tau)(q - y)$ if $y < q$ \\refKB', '\\textbf{Funcția de pierdere pinball}: $L_\\tau(y, q) = \\tau(y - q)$ dacă $y \\ge q$ și $(1 - \\tau)(q - y)$ dacă $y < q$ \\refKB'),
     [T('its expected value is minimised by the true quantile; with $\\tau = 0.5$ it is half the absolute error', 'valoarea ei așteptată este minimă în cuantila adevărată; pentru $\\tau = 0{,}5$ este jumătate din eroarea absolută'),
      T('example, $\\tau = 0.95$, $q = 7.0$ GW: $y = 7.4$ costs $0.95 \\cdot 0.4 = @{pb.a}$; $y = 6.5$ costs $0.05 \\cdot 0.5 = @{pb.b}$', 'exemplu, $\\tau = 0{,}95$, $q = 7{,}0$ GW: $y = 7{,}4$ costă $0{,}95 \\cdot 0{,}4 = @{pb.a}$; $y = 6{,}5$ costă $0{,}05 \\cdot 0{,}5 = @{pb.b}$')]),
    T('Any learner can minimise the pinball loss: linear quantile regression, quantile GB (\\texttt{loss=\'quantile\'}), networks', 'Orice algoritm poate minimiza funcția pinball: regresia cuantilică liniară, GB cuantilic (\\texttt{loss=\'quantile\'}), rețelele')), 'footnotesize')

D.frame(T('Split conformal prediction', 'Predicția conformală prin împărțire'), items(
    (T('\\textbf{Idea} \\refLei: use the errors of the model on data it has not seen to set the width of the interval', '\\textbf{Ideea} \\refLei: folosim erorile modelului pe date pe care nu le-a văzut pentru a fixa lățimea intervalului'),
     [T('1. fit the model on the training data except the last $n$ points (the calibration set)', '1. estimăm modelul pe datele de antrenare fără ultimele $n$ puncte (setul de calibrare)'),
      T('2. compute the absolute errors $|e_i|$ on the calibration set and sort them', '2. calculăm erorile absolute $|e_i|$ pe setul de calibrare și le ordonăm'),
      T('3. $\\hat q$ = the $\\lceil (n+1)(1 - \\alpha) \\rceil$-th smallest; interval $\\hat y \\pm \\hat q$', '3. $\\hat q$ = a $\\lceil (n+1)(1 - \\alpha) \\rceil$-a cea mai mică valoare; intervalul $\\hat y \\pm \\hat q$'),
      T('$\\alpha$: the target miscoverage ($\\alpha = 0.1$ for a 90\\% interval); $\\lceil\\cdot\\rceil$: rounding up', '$\\alpha$: proporția admisă de valori în afara intervalului ($\\alpha = 0{,}1$ pentru un interval de 90\\%); $\\lceil\\cdot\\rceil$: rotunjirea în sus')]),
    (T('Guarantee: coverage at least $1 - \\alpha$ if the errors are exchangeable (their order does not matter)', 'Garanția: acoperire de cel puțin $1 - \\alpha$ dacă erorile sînt interschimbabile (ordinea lor nu contează)'),
     [T('time series are not exchangeable: the guarantee is approximate; use recent calibration data and recheck the coverage walk-forward', 'seriile de timp nu sînt interschimbabile: garanția este aproximativă; folosim date de calibrare recente și reverificăm acoperirea prin walk-forward')]),
    T('Here: $n = @{in.cal}$ days, $\\alpha = 0.1$, $\\hat q$ = the @{cf.k}-th smallest absolute error, one calibration set per horizon and origin', 'Aici: $n = @{in.cal}$ de zile, $\\alpha = 0{,}1$, $\\hat q$ = a @{cf.k}-a cea mai mică eroare absolută, cîte un set de calibrare pentru fiecare orizont și origine')), 'footnotesize')

chart(T('90\\% intervals for the load', 'Intervale de 90\\% pentru consum'), 'tsa_ch9_intervals', 'TSA_ch9_load_forecasting', [
    T('Direct GB, the @{in.n} forecasts of Section 6 (all origins and horizons); left: intervals and outcomes; right: coverage by horizon',
      'GB direct, cele @{in.n} de prognoze din secțiunea 6 (toate originile și orizonturile); stînga: intervale și valori realizate; dreapta: acoperirea pe orizonturi')],
    h='0.5\\textheight')

interp(('the intervals', 'intervalelor'), [
    (T('Quantile GB covers @{in.quantile.c}\\% with mean width @{in.quantile.w} MW; split conformal covers @{in.conformal.c}\\% with width @{in.conformal.w} MW', 'GB cuantilic acoperă @{in.quantile.c}\\%, cu lățimea medie de @{in.quantile.w} MW; predicția conformală acoperă @{in.conformal.c}\\%, cu lățimea de @{in.conformal.w} MW'),
     [T('quantile GB: @{in.quantile.lo}\\% of outcomes below and @{in.quantile.hi}\\% above the interval (nominal 5\\% and 5\\%)', 'GB cuantilic: @{in.quantile.lo}\\% dintre valori sub interval și @{in.quantile.hi}\\% deasupra (nominal 5\\% și 5\\%)')]),
    T('Quantiles estimated by trees are noisy in the tails and too narrow; conformal calibration corrects the width with real out-of-sample errors', 'Cuantilele estimate prin arbori sînt zgomotoase în cozi și prea înguste; calibrarea conformală corectează lățimea cu erori reale din afara eșantionului'),
    T('Always report the empirical coverage next to an interval', 'Raportați întotdeauna acoperirea empirică alături de un interval')])

D.recap(('Prediction intervals', 'intervale de prognoză'), [
    T('Quantile forecasts minimise the pinball loss; an interval is a pair of quantiles', 'Prognozele cuantilice minimizează funcția pinball; un interval este o pereche de cuantile'),
    T('Split conformal: $\\hat y \\pm$ an empirical quantile of calibration errors', 'Predicția conformală: $\\hat y \\pm$ o cuantilă empirică a erorilor de calibrare'),
    T('Check the coverage walk-forward', 'Verificăm acoperirea prin walk-forward')])

# =============================================================================
# 8. MODELE GLOBALE: INFLAȚIA
# =============================================================================
D.section('Local and global models: Romanian inflation', 'Modele locale și globale: inflația din România')

D.frame(T('Local and global models', 'Modele locale și globale'), items(
    (T('\\textbf{Local} model: one model per series, estimated on that series only (ARIMA, ETS, the models so far)', 'Model \\textbf{local}: un model pentru fiecare serie, estimat doar pe acea serie (ARIMA, ETS, modelele de pînă acum)'),
     [T('few observations per series limit the complexity a local model can afford', 'puținele observații ale unei serii limitează complexitatea pe care și-o poate permite un model local')]),
    (T('\\textbf{Global} model: one model estimated on the pooled rows of many related series \\refMMH, \\refJanus', 'Model \\textbf{global}: un singur model estimat pe rîndurile combinate ale mai multor serii înrudite \\refMMH, \\refJanus'),
     [T('the same features for every series (scaled lags, calendar); optionally a series identifier', 'aceleași variabile pentru fiecare serie (laguri scalate, calendar); opțional, un identificator al seriei'),
      T('more data per parameter: complex learners (GB, networks) become usable; the winners of M4 and M5 were global', 'mai multe date pentru fiecare parametru: algoritmii complecși (GB, rețele) devin utilizabili; cîștigătorii M4 și M5 au fost globali')]),
    T('Risk: the series must share dynamics; a global model may average away what is special about one series', 'Riscul: seriile trebuie să aibă dinamici comune; un model global poate face să dispară prin mediere ceea ce este specific unei serii')))

D.frame(T('The inflation exercise', 'Exercițiul pentru inflație'), items(
    (T('Target: 12-month HICP inflation of Romania (Eurostat), $\\pi_t = 100(P_t/P_{t-12} - 1)$, $P_t$: the HICP index, at horizons $h = 1, 3, 6, 12$ months', 'Ținta: inflația anuală IAPC a României (Eurostat), $\\pi_t = 100(P_t/P_{t-12} - 1)$, $P_t$: indicele IAPC, la orizonturile $h = 1, 3, 6, 12$ luni'),
     [T('direct strategy; the models forecast the change $\\pi_{t+h} - \\pi_t$, so trees never extrapolate the level', 'strategie directă; modelele prognozează variația $\\pi_{t+h} - \\pi_t$, deci arborii nu extrapolează niciodată nivelul')]),
    (T('Features at $t$: $\\pi_t, \\pi_{t-1}, \\pi_{t-2}, \\pi_{t-3}, \\pi_{t-6}, \\pi_{t-12}$, the last three monthly log changes and their 3- and 6-month sums, the calendar month', 'Variabile la $t$: $\\pi_t, \\pi_{t-1}, \\pi_{t-2}, \\pi_{t-3}, \\pi_{t-6}, \\pi_{t-12}$, ultimele trei variații lunare logaritmice și sumele lor pe 3 și 6 luni, luna calendaristică'),
     [T('models: random walk (RW, $\\hat\\pi_{t+h} = \\pi_t$), AR (OLS on the same features), lasso, random forest, GB local (Romania only), GB global (@{if.nc} EU countries pooled)',
        'modele: mers aleator (RW, $\\hat\\pi_{t+h} = \\pi_t$), AR (OLS pe aceleași variabile), lasso, random forest, GB local (doar România), GB global (@{if.nc} de țări UE combinate)')]),
    T('Walk-forward: refit every January on all rows whose target month is already observed (from 2006); test origins @{if.first}--@{if.last1}', 'Walk-forward: reestimare în fiecare ianuarie pe toate rîndurile a căror lună-țintă este deja observată (din 2006); originile de test @{if.first}--@{if.last1}'),
    T('Related evidence: \\refMedeiros\\ find that random forests beat the benchmarks for US inflation with more than 100 predictors', 'Rezultate înrudite: \\refMedeiros\\ arată că random forest depășește metodele de referință pentru inflația SUA cu peste 100 de predictori')), 'footnotesize')

chart(T('Romanian inflation: forecasts and relative RMSE', 'Inflația din România: prognoze și RMSE relativ'), 'tsa_ch9_inflation', 'TSA_ch9_inflation_global', [
    T('Left: inflation and the 12-month-ahead forecasts plotted at the target month; right: RMSE relative to the random walk',
      'Stînga: inflația și prognozele cu 12 luni înainte, reprezentate la luna-țintă; dreapta: RMSE relativ la mersul aleator')],
    h='0.52\\textheight')

D.frame(T('Inflation results', 'Rezultatele pentru inflație'), table(
    'lrrrr', T('\\textbf{RMSE (pp), relative to RW}', '\\textbf{RMSE (pp), relativ la RW}') + ' & $h = 1$ & $h = 3$ & $h = 6$ & $h = 12$',
    [T(m, {'RW': 'RW (mers aleator)'}.get(m, m)) + ' & ' + ' & '.join(f'@{{if.{h}.{m.replace(" ", "")}}} (@{{if.{h}.{m.replace(" ", "")}.r}})' for h in ('1', '3', '6', '12'))
     for m in ('RW', 'AR', 'Lasso', 'RF', 'GB local', 'GB global')], size='scriptsize') + items(
    T('DM p against RW, AR: @{if.1.dm.AR} ($h = 1$), @{if.3.dm.AR} ($h = 3$), @{if.6.dm.AR} ($h = 6$); GB global against GB local: @{if.1.gl}, @{if.3.gl}, @{if.6.gl}, @{if.12.gl}',
      'DM p față de RW, AR: @{if.1.dm.AR} ($h = 1$), @{if.3.dm.AR} ($h = 3$), @{if.6.dm.AR} ($h = 6$); GB global față de GB local: @{if.1.gl}, @{if.3.gl}, @{if.6.gl}, @{if.12.gl}'),
    T('Test origins: @{if.1.n} ($h = 1$) to @{if.12.n} ($h = 12$) months; DM with squared errors and $h - 1$ autocovariances', 'Origini de test: de la @{if.1.n} ($h = 1$) la @{if.12.n} ($h = 12$) luni; DM cu erori pătratice și $h - 1$ autocovarianțe')), 'footnotesize')

interp(('the inflation results', 'rezultatelor pentru inflație'), [
    (T('The linear AR is the best model up to 6 months (about 10\\% below the random walk); no ML model beats the random walk significantly', 'AR liniar este cel mai bun model pînă la 6 luni (cu circa 10\\% sub mersul aleator); niciun model ML nu depășește semnificativ mersul aleator'),
     [T('at 12 months nothing beats the random walk: the 2022 peak (@{if.max}\\% in @{if.maxd}) came from energy prices, not from the past of inflation', 'la 12 luni niciun model nu depășește mersul aleator: vîrful din 2022 (@{if.max}\\% în @{if.maxd}) a venit din prețurile energiei, nu din trecutul inflației')]),
    (T('The global GB is better than the local one at short horizons (RMSE @{if.3.GBglobal} against @{if.3.GBlocal} at $h = 3$), but not significantly (p @{if.3.gl})', 'GB global este mai bun decît cel local la orizonturi scurte (RMSE @{if.3.GBglobal} față de @{if.3.GBlocal} la $h = 3$), dar nesemnificativ (p @{if.3.gl})'),
     [T('pooling 27 countries gives the trees enough rows; the local GB, with 120--240 rows per refit, overfits', 'combinarea a 27 de țări dă arborilor suficiente rînduri; GB local, cu 120--240 de rînduri la fiecare reestimare, face overfitting')]),
    T('About 120 monthly test errors cannot separate models whose RMSEs differ by 5\\%: the honest conclusion is ``no evidence\'\'', 'Circa 120 de erori lunare de test nu pot separa modele ale căror RMSE diferă cu 5\\%: concluzia onestă este „nu avem dovezi”')])

D.recap(('Local and global models', 'modele locale și globale'), [
    T('A global model pools many related series and lets flexible learners use more data', 'Un model global combină multe serii înrudite și permite algoritmilor flexibili să folosească mai multe date'),
    T('Romanian inflation: AR is best; global GB beats local GB; nothing beats the random walk at 12 months', 'Inflația din România: AR este cel mai bun; GB global depășește GB local; niciun model nu depășește mersul aleator la 12 luni'),
    T('Small samples: report DM p-values, not only RMSE rankings', 'Eșantioane mici: raportăm p-value-urile DM, nu doar clasamentele RMSE')])

# =============================================================================
# 9. VOLATILITATE ȘI SEMNUL RANDAMENTULUI
# =============================================================================
D.section('Applications in finance: volatility and the sign of returns', 'Aplicații în finanțe: volatilitatea și semnul randamentului')

D.frame(T('Realised volatility: HAR against ML (1/2)', 'Volatilitatea realizată: HAR și ML (1/2)'), items(
    (T('Target: the log of the mean daily variance over the next 5 days; daily variance proxy from open, high, low and close prices \\refGK', 'Ținta: logaritmul varianței zilnice medii pe următoarele 5 zile; aproximarea varianței zilnice din prețurile de deschidere, maxim, minim și închidere \\refGK'),
     [T('$\\hat\\sigma_t^2 = [\\ln(O_t/C_{t-1})]^2 + \\frac12[\\ln(H_t/L_t)]^2 - (2\\ln 2 - 1)[\\ln(C_t/O_t)]^2$: the overnight return squared plus the Garman--Klass term', '$\\hat\\sigma_t^2 = [\\ln(O_t/C_{t-1})]^2 + \\frac12[\\ln(H_t/L_t)]^2 - (2\\ln 2 - 1)[\\ln(C_t/O_t)]^2$: pătratul randamentului de peste noapte plus termenul Garman--Klass'),
      T('$O_t$, $H_t$, $L_t$, $C_t$: the open, high, low and close of day $t$', '$O_t$, $H_t$, $L_t$, $C_t$: prețurile de deschidere, maxim, minim și închidere din ziua $t$'),
      T('volatility is persistent and predictable (Chapter 5): the right ground for ML', 'volatilitatea este persistentă și previzibilă (Capitolul 5): terenul potrivit pentru ML')]),
    (T('\\textbf{HAR} (heterogeneous autoregressive) model \\refCorsi: OLS on the log variance of the last day, week and month', 'Modelul \\textbf{HAR} (heterogeneous autoregressive) \\refCorsi: OLS pe logaritmul varianței din ultima zi, săptămînă și lună'),
     [T('$\\ln \\bar\\sigma^2_{t+1:t+5} = c + \\beta_d \\ln\\hat\\sigma_t^2 + \\beta_w \\ln\\bar\\sigma^2_{t-4:t} + \\beta_m \\ln\\bar\\sigma^2_{t-21:t} + u_t$; $\\bar\\sigma^2_{a:b}$: the mean of $\\hat\\sigma^2$ over days $a$ to $b$; $u_t$: the error', '$\\ln \\bar\\sigma^2_{t+1:t+5} = c + \\beta_d \\ln\\hat\\sigma_t^2 + \\beta_w \\ln\\bar\\sigma^2_{t-4:t} + \\beta_m \\ln\\bar\\sigma^2_{t-21:t} + u_t$; $\\bar\\sigma^2_{a:b}$: media lui $\\hat\\sigma^2$ în zilele de la $a$ la $b$; $u_t$: eroarea'),
      T('ML models add 8 features: lags of the daily variance, the quarterly mean, the return, its negative part and its absolute value', 'modelele ML adaugă 8 variabile: laguri ale varianței zilnice, media trimestrială, randamentul, partea lui negativă și valoarea lui absolută')])), 'footnotesize')

D.frame(T('Realised volatility: HAR against ML (2/2)', 'Volatilitatea realizată: HAR și ML (2/2)'), items(
    T('Yearly walk-forward from 2013 with a 5-day gap between training and test', 'Walk-forward anual din 2013, cu un spațiu de 5 zile între antrenare și test'),
    (T('Out-of-sample $R^2$ against HAR: $R^2_{\\text{HAR}} = 1 - \\sum_t (y_t - \\hat y_t)^2 / \\sum_t (y_t - \\hat y_t^{\\text{HAR}})^2$', '$R^2$ în afara eșantionului față de HAR: $R^2_{\\text{HAR}} = 1 - \\sum_t (y_t - \\hat y_t)^2 / \\sum_t (y_t - \\hat y_t^{\\text{HAR}})^2$'),
     [T('$y_t$: the realised log variance; $\\hat y_t$, $\\hat y_t^{\\text{HAR}}$: the forecasts of the model and of HAR; positive: better than HAR', '$y_t$: logaritmul varianței realizate; $\\hat y_t$, $\\hat y_t^{\\text{HAR}}$: prognozele modelului și ale HAR; pozitiv: mai precis decît HAR')]),
    (T('QLIKE \\refPatton: $L(\\sigma^2, h) = \\sigma^2/h + \\ln h$, the quasi-likelihood loss for variances (Chapter 5); lower is better', 'QLIKE \\refPatton: $L(\\sigma^2, h) = \\sigma^2/h + \\ln h$, funcția de pierdere quasi-likelihood pentru varianță (Capitolul 5); o valoare mai mică este mai bună'),
     [T('$\\sigma^2$: the realised variance; $h$: the variance forecast; for a given $\\sigma^2$ the loss is smallest at $h = \\sigma^2$ and penalises forecasts that are too low more', '$\\sigma^2$: varianța realizată; $h$: prognoza varianței; pentru un $\\sigma^2$ dat, pierderea este minimă la $h = \\sigma^2$ și penalizează mai mult prognozele prea mici'),
      T('robust to the noise of the variance proxy: it ranks forecasts as the true variance would', 'robustă la zgomotul aproximării varianței: ordonează prognozele la fel ca varianța adevărată')]),
    T('Diebold--Mariano test (Chapter 4) on the daily QLIKE differences', 'Testul Diebold--Mariano (Capitolul 4) pe diferențele zilnice ale QLIKE')), 'footnotesize')

D.frame(T('Results: S\\&P 500 and DAX, 2013--2026', 'Rezultatele: S\\&P 500 și DAX, 2013--2026'), table(
    'lrrrrrr', T('\\textbf{Model}', '\\textbf{Model}') + ' & ' + T('S\\&P: $R^2$ vs HAR (\\%)', 'S\\&P: $R^2$ față de HAR (\\%)') + ' & QLIKE & DM $t$ & ' + T('DAX: $R^2$ vs HAR (\\%)', 'DAX: $R^2$ față de HAR (\\%)') + ' & QLIKE & DM $t$',
    [f'{m} & @{{rv.sp500.{m}.r2}} & @{{rv.sp500.{m}.ql}} & ' + ('--' if m == 'HAR' else f'@{{rv.sp500.{m}.t}}')
     + f' & @{{rv.dax.{m}.r2}} & @{{rv.dax.{m}.ql}} & ' + ('--' if m == 'HAR' else f'@{{rv.dax.{m}.t}}')
     for m in ('HAR', 'Lasso', 'RF', 'GB', 'MLP')], size='scriptsize') + items(
    T('@{rv.sp500.n} and @{rv.dax.n} test days; DM $t < -1.96$: the model has a significantly lower QLIKE than HAR', '@{rv.sp500.n} și @{rv.dax.n} zile de test; DM $t < -1{,}96$: modelul are un QLIKE semnificativ mai mic decît HAR'),
    T('The lasso and the MLP gain 7--9\\% of $R^2$ over HAR and lower QLIKE significantly on the S\\&P 500; the random forest and GB do not beat HAR', 'Lasso și MLP obțin un $R^2$ față de HAR de 7--9\\% și reduc semnificativ QLIKE pe S\\&P 500; random forest și GB nu depășesc HAR'),
    T('The gain comes from the extra features (the negative return: the leverage effect of Chapter 5), used smoothly; trees cut a smooth relation into steps', 'Cîștigul vine din variabilele suplimentare (randamentul negativ: efectul de levier din Capitolul 5), folosite lin; arborii taie o relație netedă în trepte')), 'footnotesize')

chart(T('Volatility forecasts in two storms', 'Prognozele volatilității în două episoade de criză'), 'tsa_ch9_rv', 'TSA_ch9_finance', [
    T('S\\&P 500: annualised realised volatility of the next 5 days and the HAR, random forest and GB forecasts, 2020 and 2025',
      'S\\&P 500: volatilitatea realizată anualizată a următoarelor 5 zile și prognozele HAR, random forest și GB, 2020 și 2025')],
    h='0.52\\textheight')

D.frame(T('The sign of tomorrow\'s return: the right baseline', 'Semnul randamentului de mîine: reperul corect'), items(
    (T('Classification: target $1\\{r_{t+1} > 0\\}$ (1 if tomorrow\'s return is positive, 0 otherwise); features: the last five returns, sums over 5, 21, 63 days, volatility over 21 and 63 days', 'Clasificare: ținta $1\\{r_{t+1} > 0\\}$ (1 dacă randamentul de mîine este pozitiv, altfel 0); variabile: ultimele cinci randamente, sume pe 5, 21, 63 de zile, volatilitatea pe 21 și 63 de zile'),
     [T('models: logistic regression (logit), random forest, GB; yearly walk-forward from 2008', 'modele: regresia logistică (logit), random forest, GB; walk-forward anual din 2008')]),
    (T('\\textbf{The right baseline} is not 50\\%: markets rise on more days than they fall', '\\textbf{Reperul corect} nu este 50\\%: piețele cresc în mai multe zile decît scad'),
     [T('``always up\'\' (the majority class of the training data) is right on @{sg.sp500.base}\\% of S\\&P 500 days and @{sg.bet.base}\\% of BET days', '„întotdeauna în creștere” (clasa majoritară din antrenare) are dreptate în @{sg.sp500.base}\\% din zilele S\\&P 500 și în @{sg.bet.base}\\% din zilele BET')]),
    T('Accuracy, S\\&P 500: logit @{sg.sp500.Logit}\\%, random forest @{sg.sp500.RF}\\%, GB @{sg.sp500.GB}\\%; BET: @{sg.bet.Logit}\\%, @{sg.bet.RF}\\%, @{sg.bet.GB}\\%', 'Acuratețea, S\\&P 500: logit @{sg.sp500.Logit}\\%, random forest @{sg.sp500.RF}\\%, GB @{sg.sp500.GB}\\%; BET: @{sg.bet.Logit}\\%, @{sg.bet.RF}\\%, @{sg.bet.GB}\\%'),
    (T('No model beats the baseline; AUC (area under the ROC curve) @{sg.sp500.Logit.auc}--@{sg.bet.Logit.auc}, close to 0.5: no ranking skill either', 'Niciun model nu depășește reperul; AUC (aria de sub curba ROC) @{sg.sp500.Logit.auc}--@{sg.bet.Logit.auc}, aproape de 0,5: nici capacitate de ordonare'),
     [T('AUC: the probability that a randomly chosen up day gets a higher model score than a randomly chosen down day; 0.5: no skill, 1: perfect ranking', 'AUC: probabilitatea ca o zi de creștere aleasă la întîmplare să primească un scor mai mare decît o zi de scădere aleasă la întîmplare; 0,5: fără putere, 1: ordonare perfectă')])), 'footnotesize')

chart(T('Accuracy against the baseline', 'Acuratețea față de reper'), 'tsa_ch9_sign', 'TSA_ch9_finance', [
    T('Out-of-sample accuracy 2008--2026: @{sg.sp500.n} S\\&P 500 days and @{sg.bet.n} BET days', 'Acuratețea în afara eșantionului, 2008--2026: @{sg.sp500.n} de zile S\\&P 500 și @{sg.bet.n} de zile BET')],
    h='0.48\\textheight')

interp(('the sign experiment', 'experimentului privind semnul'), [
    (T('A model with ``53\\% accuracy\'\' sounds skilful; against the @{sg.sp500.base}\\% of ``always up\'\' it is worse than doing nothing', 'Un model cu „53\\% acuratețe” pare performant; față de @{sg.sp500.base}\\% pentru „întotdeauna în creștere”, este mai slab decît a nu face nimic'),
     [T('GB on the S\\&P 500 is significantly worse than the baseline (p @{sg.sp500.GB.p}): flexible models fit noise', 'GB pe S\\&P 500 este semnificativ mai slab decît reperul (p @{sg.sp500.GB.p}): modelele flexibile ajustează zgomotul')]),
    T('Weak-form efficiency (Chapter 1): past returns contain almost no information about the sign of the next one', 'Eficiența în formă slabă (Capitolul 1): randamentele trecute nu conțin aproape nicio informație despre semnul următorului randament'),
    T('Volatility is predictable, the direction is not: ML confirms what the statistical models of Chapters 1 and 5 showed', 'Volatilitatea este previzibilă, direcția nu: ML confirmă ce au arătat modelele statistice din Capitolele 1 și 5')])

D.recap(('Volatility and the sign of returns', 'volatilitatea și semnul randamentului'), [
    T('Realised volatility: lasso and MLP with extra features beat HAR; trees do not', 'Volatilitatea realizată: lasso și MLP cu variabile suplimentare sînt mai precise decît HAR; arborii nu'),
    T('Sign of returns: compare with the majority class, not with 50\\%; no model beats it', 'Semnul randamentului: comparăm cu clasa majoritară, nu cu 50\\%; niciun model nu o depășește')])

# =============================================================================
# 10. STUDII DE CAZ
# =============================================================================
D.section('Landmark case studies: M4, M5 and Gu, Kelly and Xiu', 'Studii de caz de referință: M4, M5 și Gu, Kelly și Xiu')

D.frame(T('The M competitions', 'Competițiile M'), items(
    (T('Since 1982 Spyros Makridakis has organised open forecasting competitions: all methods forecast the same series, evaluated by the organisers on hidden data', 'Din 1982, Spyros Makridakis organizează competiții deschise de prognoză: toate metodele prognozează aceleași serii, evaluate de organizatori pe date ascunse'),
     [T('the honest version of the walk-forward test: no participant can tune on the test data', 'versiunea onestă a testului walk-forward: niciun participant nu își poate ajusta metoda pe datele de test')]),
    (T('\\refMSAplos: on 1\\,045 monthly series of M3, popular ML methods were less accurate than eight statistical methods at every horizon, and much slower',
       '\\refMSAplos: pe 1\\,045 de serii lunare din M3, metodele ML populare au fost mai puțin precise decît opt metode statistice la toate orizonturile și mult mai lente'),
     []),
    (T('M4 (2018), \\refMfour: 100\\,000 series of six frequencies, 61 methods, ranked by OWA (overall weighted average)', 'M4 (2018), \\refMfour: 100\\,000 de serii cu șase frecvențe, 61 de metode, clasate după OWA (overall weighted average)'),
     [T('$\\text{OWA} = \\frac12\\bigl(\\text{sMAPE}/\\text{sMAPE}_{N2} + \\text{MASE}/\\text{MASE}_{N2}\\bigr)$; $N2$: Naive2, the naive forecast of the seasonally adjusted series; below 1: better than Naive2', '$\\text{OWA} = \\frac12\\bigl(\\text{sMAPE}/\\text{sMAPE}_{N2} + \\text{MASE}/\\text{MASE}_{N2}\\bigr)$; $N2$: Naive2, prognoza naivă a seriei desezonalizate; sub 1: mai precis decît Naive2'),
      T('sMAPE (symmetric mean absolute percentage error) $= \\frac{200}{H}\\sum_{h=1}^{H} |y_{T+h} - \\hat y_{T+h}| / (|y_{T+h}| + |\\hat y_{T+h}|)$, in \\%', 'sMAPE (symmetric mean absolute percentage error, eroarea procentuală absolută medie simetrică) $= \\frac{200}{H}\\sum_{h=1}^{H} |y_{T+h} - \\hat y_{T+h}| / (|y_{T+h}| + |\\hat y_{T+h}|)$, în \\%')]),
    T('M5 (2020), \\refMfive: 42\\,840 hierarchical daily series of Walmart unit sales, 28-day horizon', 'M5 (2020), \\refMfive: 42\\,840 de serii zilnice ierarhice de vînzări Walmart, orizontul de 28 de zile')), 'footnotesize')

chart(T('M4: the accuracy of all methods', 'M4: acuratețea tuturor metodelor'), 'tsa_ch9_m4', 'TSA_ch9_case_studies', [
    T('Official evaluation file of M4 (github.com/Mcompetitions/M4-methods): OWA of the @{m4.n} ranked methods, submissions and benchmarks, coloured by type; @{m4.off} methods with OWA above 1.7 are off the chart',
      'Fișierul oficial de evaluare M4 (github.com/Mcompetitions/M4-methods): OWA pentru cele @{m4.n} de metode clasate, propuneri și metode de referință, colorate după tip; @{m4.off} metode cu OWA peste 1,7 nu apar pe grafic')],
    h='0.5\\textheight')

interp(('M4', 'M4'), [
    (T('Winner: the hybrid ES-RNN of \\refSmyl, exponential smoothing inside an LSTM trained on all series (a global model): OWA @{m4.best}, sMAPE @{m4.bests}\\%', 'Cîștigătorul: modelul hibrid ES-RNN al lui \\refSmyl, netezire exponențială în interiorul unui LSTM antrenat pe toate seriile (un model global): OWA @{m4.best}, sMAPE @{m4.bests}\\%'),
     [T('Comb, the average of SES, Holt and damped trend: OWA @{m4.comb}, sMAPE @{m4.combs}\\%; Naive2: sMAPE @{m4.n2s}\\%', 'Comb, media metodelor SES, Holt și trend amortizat: OWA @{m4.comb}, sMAPE @{m4.combs}\\%; Naive2: sMAPE @{m4.n2s}\\%')]),
    (T('The @{m4.nml} pure ML methods: the best ranked @{m4.mlrank} (OWA @{m4.mlbest}); none beat Comb, only one beat Naive2 \\refMfourA', 'Cele @{m4.nml} metode ML pure: cea mai bună pe locul @{m4.mlrank} (OWA @{m4.mlbest}); niciuna nu a depășit Comb, doar una a depășit Naive2 \\refMfourA'),
     [T('@{m4.top17} of the 17 best methods were combinations', '@{m4.top17} dintre cele mai bune 17 metode au fost combinații')]),
    T('Lesson: on short, heterogeneous series, combine statistical models, or let ML learn across series; pure local ML loses', 'Lecția: pe serii scurte și eterogene, combinăm modele statistice sau lăsăm ML să învețe de la toate seriile; ML pur, aplicat local, este mai slab')])

D.frame(T('M5: gradient boosting wins on retail data', 'M5: gradient boosting, cel mai precis pe datele din comerț'), twocol(
    ph('walmart', T('A Walmart store: the M5 data are the daily unit sales of 3\\,049 products in 10 stores', 'Un magazin Walmart: datele M5 sînt vînzările zilnice a 3\\,049 de produse în 10 magazine'), h='0.42\\textheight'),
    items((T('5\\,507 teams from 101 countries; error measure WRMSSE (weighted root mean squared scaled error)', '5\\,507 echipe din 101 țări; măsura erorii WRMSSE (weighted root mean squared scaled error)'),
           [T('RMSSE: the RMSE of the forecasts over the in-sample RMSE of the one-step naive forecast', 'RMSSE: RMSE a prognozelor raportată la RMSE în eșantion a prognozei naive la un pas'),
            T('WRMSSE: the average RMSSE, weighted by sales value', 'WRMSSE: media RMSSE, ponderată cu valoarea vînzărilor'),
            T('the winner (WRMSSE 0.520) was 22.4\\% more accurate than the best benchmark, bottom-up exponential smoothing (0.671)', 'cîștigătorul (WRMSSE 0,520) a fost cu 22,4\\% mai precis decît cea mai bună metodă de referință, netezirea exponențială de jos în sus (0,671)')]),
          (T('Most top methods used LightGBM \\refLGBM', 'Cele mai multe metode de top au folosit LightGBM \\refLGBM'),
           [T('the winner averaged LightGBM models pooled by store, category and department, recursive and direct', 'cîștigătorul a făcut media unor modele LightGBM combinate pe magazin, categorie și departament, recursive și directe'),
            T('global models, rich calendar and price features, many related series: the conditions where ML wins', 'modele globale, variabile bogate de calendar și preț, multe serii înrudite: condițiile în care ML este superior')])),
    wl='0.36', wr='0.62'), 'footnotesize')

D.frame(T('Gu, Kelly and Xiu (2020): machine learning for stock returns', 'Gu, Kelly și Xiu (2020): învățare automată pentru randamentele acțiunilor'), twocol(
    ph('chicago', T('Charles M. Harper Center, University of Chicago Booth School of Business', 'Charles M. Harper Center, University of Chicago Booth School of Business'), h='0.36\\textheight'),
    items((T('\\refGKX: monthly returns of about 30\\,000 US stocks, 1957--2016, 920 predictors', '\\refGKX: randamentele lunare a circa 30\\,000 de acțiuni americane, 1957--2016, 920 de predictori'),
           [T('12 methods, from OLS to neural networks with five layers', '12 metode, de la OLS la rețele neuronale cu cinci straturi'),
            T('walk-forward: 18 years of training, 12 of validation, 30 years of test, refitted yearly', 'walk-forward: 18 ani de antrenare, 12 de validare, 30 de ani de test, reestimare anuală')]),
          T('Out-of-sample monthly $R^2$: at most 0.40\\% (three-layer network); OLS with all predictors: $-3.46$\\%', '$R^2$ lunar în afara eșantionului: cel mult 0,40\\% (rețea cu trei straturi); OLS cu toți predictorii: $-3{,}46$\\%'),
          (T('Small $R^2$, large value: a long-short portfolio sorted on the forecasts reaches an annual Sharpe ratio of 1.35 (four layers), against 0.61 for OLS-3', '$R^2$ mic, valoare mare: un portofoliu long-short ordonat după prognoze atinge un raport Sharpe anual de 1,35 (patru straturi), față de 0,61 pentru OLS-3'),
           [T('long-short: buy the decile with the highest forecasts, sell the decile with the lowest', 'long-short: cumpărăm decila cu cele mai mari prognoze și vindem decila cu cele mai mici'),
            T('Sharpe ratio: the mean annual return divided by its annual standard deviation', 'raportul Sharpe: randamentul mediu anual împărțit la abaterea lui standard anuală')])),
    wl='0.34', wr='0.64'), 'footnotesize')

chart(T('Gu, Kelly and Xiu (2020): small $R^2$, large economic value', 'Gu, Kelly și Xiu (2020): $R^2$ mic, valoare economică mare'), 'tsa_ch9_gkx', 'TSA_ch9_case_studies', [
    T('Published numbers: Table 1 (monthly out-of-sample $R^2$, all stocks) and Table 7 (annualised Sharpe ratio of the value-weighted long-short decile portfolio); +H: Huber loss',
      'Cifrele publicate: Tabelul 1 ($R^2$ lunar în afara eșantionului, toate acțiunile) și Tabelul 7 (raportul Sharpe anualizat al portofoliului long-short pe decile, ponderat cu valoarea); +H: funcția de pierdere Huber'),
    T('OLS-3: OLS on size, book-to-market and momentum; PLS, PCR: partial least squares, principal component regression; ENet: elastic net',
      'OLS-3: OLS pe mărime, raportul valoare contabilă/valoare de piață și momentum; PLS, PCR: partial least squares, regresie pe componente principale; ENet: elastic net'),
    T('GLM: generalised linear model with splines; RF: random forest; GBRT: gradient-boosted regression trees; NN1--NN5: networks with 1--5 hidden layers',
      'GLM: model liniar generalizat cu spline; RF: random forest; GBRT: arbori de regresie cu gradient boosting; NN1--NN5: rețele cu 1--5 straturi ascunse')],
    h='0.4\\textheight')

D.recap(('Case studies', 'studii de caz'), [
    T('M4: combinations and a hybrid global model win; pure local ML loses to simple benchmarks', 'M4: cele mai precise sînt combinațiile și un model hibrid global; ML pur local este mai slab decît metodele simple'),
    T('M5: global LightGBM models with calendar and price features win clearly', 'M5: modelele globale LightGBM cu variabile de calendar și preț sînt clar cele mai precise'),
    T('Gu, Kelly and Xiu: tiny but real predictability in the cross-section of stocks; nonlinear models add value', 'Gu, Kelly și Xiu: predictibilitate mică, dar reală, în secțiunea transversală a acțiunilor; modelele neliniare aduc valoare')])

# =============================================================================
# 11. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Foundation models: a pointer to Chapter 11', 'Foundation models: trimitere la Capitolul 11'), items(
    (T('A \\textbf{foundation model} for time series is a large network (a transformer) pre-trained on millions of series and used without retraining (zero-shot)', 'Un \\textbf{foundation model} pentru serii de timp este o rețea mare (un transformer) pre-antrenată pe milioane de serii și folosită fără reantrenare (zero-shot)'),
     [T('the extreme global model: Chronos, TimesFM, Moirai, Lag-Llama (self-study, Chapter 11)', 'modelul global extrem: Chronos, TimesFM, Moirai, Lag-Llama (studiu individual, Capitolul 11)')]),
    T('The same rules apply: walk-forward on data the model has not seen in pre-training, the same benchmarks, MASE and DM', 'Se aplică aceleași reguli: walk-forward pe date pe care modelul nu le-a văzut la pre-antrenare, aceleași metode de referință, MASE și DM')))

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Code}: a first draft of the feature table, the walk-forward loop and the evaluation table', '\\textbf{Cod}: o primă versiune a tabelului de variabile, a buclei walk-forward și a tabelului de evaluare'),
    T('\\textbf{Explanation}: a second explanation of boosting, of the LSTM gates or of conformal prediction', '\\textbf{Explicații}: o a doua explicație a boosting-ului, a porților LSTM sau a predicției conformale'),
    T('\\textbf{Exploration}: many feature sets and hyperparameters at once (on validation data only)', '\\textbf{Explorare}: multe seturi de variabile și hiperparametri deodată (doar pe datele de validare)'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that builds lag, rolling-mean and calendar features for daily electricity load, trains HistGradientBoostingRegressor with the direct strategy for horizons 1-14, evaluates it walk-forward against the seasonal naive forecast with MASE and the Diebold-Mariano test, and refits every preprocessing step inside each training window.}',
        '\\aiprompt{Write Python code that builds lag, rolling-mean and calendar features for daily electricity load, trains HistGradientBoostingRegressor with the direct strategy for horizons 1-14, evaluates it walk-forward against the seasonal naive forecast with MASE and the Diebold-Mariano test, and refits every preprocessing step inside each training window.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('Every feature uses data up to the origin only: check the shifts of lags and rolling windows on a printed table', 'Fiecare variabilă folosește doar date pînă la origine: verificați deplasarea lagurilor și a ferestrelor mobile pe un tabel tipărit'),
    T('No \\texttt{KFold(shuffle=True)}, no scaler or feature selection fitted on the full sample', 'Fără \\texttt{KFold(shuffle=True)}, fără scalare sau selecție de variabile estimate pe tot eșantionul'),
    T('The benchmarks are there: naive or seasonal naive, ETS or ARIMA, on the same origins', 'Metodele de referință sînt prezente: naivă sau naivă sezonieră, ETS sau ARIMA, pe aceleași origini'),
    T('The comparison has a DM test; the classification has the majority-class baseline', 'Comparația are un test DM; clasificarea are reperul clasei majoritare'),
    T('Seeds are fixed and the result survives another seed', 'Semințele sînt fixate, iar rezultatul se menține cu o altă sămînță'),
    T('Every cited reference exists: check the DOI', 'Fiecare referință citată există: verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('ML forecasting = supervised learning on a table of lags, rolling statistics and calendar features known at the origin', 'Prognoza ML = învățare supervizată pe un tabel de laguri, statistici pe ferestre mobile și variabile de calendar cunoscute la origine'),
    T('Walk-forward validation only; leakage produces spectacular and false scores', 'Doar validare walk-forward; leakage-ul produce scoruri spectaculoase și false'),
    T('Ridge and lasso, random forest, GB, MLP, LSTM: each has a complexity knob set by validation', 'Ridge și lasso, random forest, GB, MLP, LSTM: fiecare are un parametru de complexitate fixat prin validare'),
    T('Trees cannot extrapolate: model changes, not levels', 'Arborii nu pot extrapola: modelăm variații, nu niveluri'),
    T('Our data: linear models with good features match or beat ML on load and inflation; ML helps for volatility; nothing predicts the sign of returns', 'Datele noastre: modelele liniare cu variabile bune au cel puțin aceeași precizie ca ML pentru consum și inflație; ML ajută pentru volatilitate; niciun model nu prognozează semnul randamentelor'),
    T('ML wins with many related series (global models, M4 hybrid, M5 LightGBM); always compare with benchmarks, MASE and DM', 'ML este superior cînd există multe serii înrudite (modele globale, hibridul din M4, LightGBM în M5); comparăm întotdeauna cu metode de referință, MASE și DM')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.35}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Direct forecast', 'Prognoza directă') + ' & $\\hat y_{T+h} = \\hat f_h(y_T, \\dots, y_{T-p+1}, \\mathbf{z}_{T+h})$, ' + T('$\\mathbf{z}_{T+h}$: calendar features', '$\\mathbf{z}_{T+h}$: variabilele de calendar'),
     T('Ridge, lasso', 'Ridge, lasso') + ' & $\\min \\sum_t (y_t - \\mathbf{x}_t\'\\beta)^2 + \\lambda\\sum_j\\beta_j^2$; \\quad $+ \\lambda\\sum_j|\\beta_j|$',
     T('Bias--variance', 'Deplasare--varianță') + ' & $E[(y_0 - \\hat f(x_0))^2] = ' + T('\\text{bias}', '\\text{deplasare}') + '^2 + \\Var(\\hat f(x_0)) + \\sigma^2$',
     T('Boosting', 'Boosting') + ' & $F_m(\\mathbf{x}) = F_{m-1}(\\mathbf{x}) + \\nu\\,g_m(\\mathbf{x})$',
     'MLP & $\\hat y = c + \\sum_j v_j\\,g(b_j + \\mathbf{w}_j\'\\mathbf{x})$',
     'LSTM & $c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde c_t$, \\quad $h_t = o_t \\odot \\tanh(c_t)$',
     T('Pinball loss', 'Funcția pinball') + ' & $L_\\tau(y, q) = \\max(\\tau(y - q), (\\tau - 1)(y - q))$',
     T('Split conformal', 'Predicția conformală') + ' & $\\hat y \\pm \\hat q$, \\quad $\\hat q = |e|_{(\\lceil (n+1)(1-\\alpha) \\rceil)}$',
     'MASE & $\\text{MAE} / \\frac{1}{n-m}\\sum_{t>m}|y_t - y_{t-m}|$'],
    size='footnotesize') + '}')

D.frame(T('Self-assessment', 'Autoevaluare'), items(
    (T('\\textbf{Question}: a 7-day moving average centred on $t$ is used as a feature for $y_{t+1}$. Is that allowed?', '\\textbf{Întrebare}: o medie mobilă pe 7 zile centrată în $t$ este folosită ca variabilă pentru $y_{t+1}$. Este permis?'),
     [T('\\textbf{Answer}: no: it contains $y_{t+1}, y_{t+2}, y_{t+3}$; use the trailing mean of $y_{t-6}, \\dots, y_t$', '\\textbf{Răspuns}: nu: conține $y_{t+1}, y_{t+2}, y_{t+3}$; folosiți media $y_{t-6}, \\dots, y_t$')]),
    (T('\\textbf{Question}: a random forest trained on GDP levels up to 2019 forecasts 2024. What goes wrong?', '\\textbf{Întrebare}: un random forest antrenat pe nivelurile PIB-ului pînă în 2019 prognozează anul 2024. Ce nu funcționează?'),
     [T('\\textbf{Answer}: its forecast cannot exceed the largest level seen in training; model growth rates', '\\textbf{Răspuns}: prognoza lui nu poate depăși cel mai mare nivel văzut la antrenare; modelați ratele de creștere')]),
    (T('\\textbf{Question}: a classifier predicts the sign of daily returns with 53\\% accuracy. Is it useful?', '\\textbf{Întrebare}: un clasificator prognozează semnul randamentelor zilnice cu acuratețea de 53\\%. Este util?'),
     [T('\\textbf{Answer}: only if it beats the share of up days in the test period (about 54\\% for the S\\&P 500), significantly', '\\textbf{Răspuns}: doar dacă depășește semnificativ ponderea zilelor de creștere în perioada de test (circa 54\\% pentru S\\&P 500)')]),
    T('Next: Chapter 10, state space models, the Kalman filter and Markov switching', 'Urmează: Capitolul 10, modele în spațiul stărilor, filtrul Kalman și modele Markov switching')))

D.references(bib(), per=12)

if __name__ == '__main__':
    finalize(D.write(V))
