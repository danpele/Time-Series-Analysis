r"""
build_seminar9.py -- Seminarul 9 (Învățare automată pentru serii de timp), EN + RO dintr-o singură sursă
======================================================================================================
Seminarul are loc ÎNAINTEA cursului 9: secțiunea „Noțiuni necesare azi” dă tot ce folosesc cerințele.
Formatul A/B/C: A calcule pe hîrtie, B date reale cu o întrebare de interpretare, C o întrebare deschisă și
critica unui răspuns AI. [Rezolvat]: rezolvarea vizibilă pentru toți; [Propus]: rezolvarea doar în versiunea
profesorului (*_solutions.tex, exclusă din git). Studenții nu predau nimic.
Cifrele @{cheie} vin din Quantlets/Ch_09/sem9_results.json (seminar9.py).
Ieșire:
  EN/Seminars/seminar9_machine_learning_time_series.tex          (+ _solutions.tex)
  RO/Seminarii/seminar9_invatare_automata_serii_timp_ro.tex      (+ _solutions.tex)
Rulare:
  python3 Quantlets/Ch_09/seminar9.py && python3 latex/build_seminar9.py && python3 latex/tsa_build.py compile 9
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, fig   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch9_common import REFS, T, bib, date, finalize, load_sem, pv, month   # noqa: E402

S = load_sem()
V = Values()
D = Deck(9, 'seminar', refs=REFS)


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


def qlsem():
    return '\\quantlet{TSA\\_ch9\\_seminar}{\\qlurl{TSA_ch9_seminar}}'


def num(x, d=2):
    return '⁅' + f'{x:.{d}f}' + '⁆'


# =============================================================================
# CIFRE
# =============================================================================
A1 = S['A1']
for i, r in enumerate(A1['rec']):
    V.put(f'a1.r{i + 1}', r, 2)
V.put('a1.d2', A1['direct2'], 2)
V.raw('a1.n1', str(A1['rows_h1']))
V.raw('a1.n2', str(A1['rows_h2']))
A1ROWS = [f'{r["t"]} & {num(r["y_t-1"], 0)} & {num(r["y_t-2"], 0)} & {num(r["y_t"], 0)}' for r in A1['table']]
A2 = S['A2']
V.put('a2.c', A2['correct'], 2)
V.put('a2.l', A2['leaky'], 2)
A3 = S['A3']
V.put('a3.sst', A3['sst'], 3)
V.put('a3.mean', A3['mean'], 2)
V.put('a3.thr', A3['best']['thr'], 1)
V.put('a3.lm', A3['best']['left_mean'], 2)
V.put('a3.rm', A3['best']['right_mean'], 2)
V.put('a3.sse', A3['best']['sse'], 3)
A3ROWS = [f'{num(r["thr"], 1)} & {num(r["left_mean"], 2)} & {num(r["right_mean"], 2)} & {num(r["sse"], 3)}' for r in A3['rows']]
A4 = S['A4']
V.put('a4.ols', A4['ols'], 2)
V.put('a4.r10', A4['ridge']['10.0'], 3)
V.put('a4.r50', A4['ridge']['50.0'], 2)
V.put('a4.l20', A4['lasso']['20.0'], 2)
V.put('a4.l100', A4['lasso']['100.0'], 2)
A5 = S['A5']
V.raw('a5.sorted', '; '.join(num(x) for x in A5['sorted']) if False else ', '.join(num(x) for x in A5['sorted']))
V.raw('a5.k', str(A5['k']))
V.put('a5.half', A5['half'], 2)
V.put('a5.lo', A5['lo'], 2)
V.put('a5.hi', A5['hi'], 2)
V.put('a5.p1', A5['pin']['6.0'], 2)
V.put('a5.p2', A5['pin']['6.6'], 2)
A6 = S['A6']
for k in ('mae_a', 'mae_b', 'mase_a', 'mase_b', 'dbar', 's', 't', 'crit'):
    V.put(f'a6.{k}', A6[k], 3 if k in ('mae_a', 'mae_b', 'dbar', 's') else 2)
V.raw('a6.p', pv(A6['p']))
V.raw('a6.d', ', '.join(num(x) for x in A6['d']))

B1 = S['B1']
V.raw('b1.no', str(B1['n_origins']))
V.put('b1.scale', 1000 * B1['scale'], 0)
for m, k in (('GB direct', 'gb'), ('Seasonal naive', 'sn')):
    V.put(f'b1.{k}.mae', 1000 * B1[m]['mae'], 0)
    V.put(f'b1.{k}.rmse', 1000 * B1[m]['rmse'], 0)
    V.put(f'b1.{k}.mase', B1[m]['mase'], 2)
V.put('b1.dm', B1['dm']['hln'], 2)
V.raw('b1.p', pv(B1['dm']['p']))
V.put('b1.h1g', 1000 * B1['byh']['GB direct'][0], 0)
V.put('b1.h1s', 1000 * B1['byh']['Seasonal naive'][0], 0)
V.put('b1.h7g', 1000 * B1['byh']['GB direct'][-1], 0)
V.put('b1.h7s', 1000 * B1['byh']['Seasonal naive'][-1], 0)
V.raw('b1.wo', date(B1['worst']['origin']))
V.put('b1.wm', 1000 * B1['worst']['mae'], 0)
B2 = S['B2']
for m, k in (('direct', 'd'), ('recursive', 'r'), ('direct, no holidays', 'n')):
    V.put(f'b2.{k}', 1000 * B2[m], 0)
    V.put(f'b2.{k}.h', 1000 * B2['on_holidays'][m], 0)
    V.put(f'b2.{k}.o', 1000 * B2['other_days'][m], 0)
V.raw('b2.nh', str(B2['holiday_days']))
B3 = S['B3']
for m in ('RW', 'AR', 'Lasso', 'GB local', 'GB global', 'RF'):
    k = m.replace(' ', '')
    V.put(f'b3.{k}', B3['rmse'][m], 2)
    V.put(f'b3.{k}.r', B3['rel'][m], 2)
    V.put(f'b3.{k}.s', B3['rmse_surge'][m], 2)
V.raw('b3.n', str(B3['n']))
V.raw('b3.first', month(B3['first']))
V.raw('b3.last', month(B3['last']))
for k in ('dm_ar_rw', 'dm_gl_ar', 'dm_gl_loc'):
    V.put(f'b3.{k}', B3[k]['hln'], 2)
    V.raw(f'b3.{k}.p', pv(B3[k]['p']))
B4 = S['B4']
V.int('b4.n', B4['n'])
V.put('b4.up', 100 * B4['up'], 1)
V.put('b4.base', 100 * B4['base_acc'], 1)
for m in ('Logit', 'GB'):
    V.put(f'b4.{m}', 100 * B4[m]['acc'], 1)
    V.put(f'b4.{m}.z', B4[m]['z'], 2)
    V.raw(f'b4.{m}.p', pv(B4[m]['p']))
    V.put(f'b4.{m}.auc', B4[m]['auc'], 3)
    V.put(f'b4.{m}.up', 100 * B4[m]['share_up_pred'], 1)
C2 = S['C2']
V.put('c2.kf', 100 * C2['kfold'], 1)
V.put('c2.wf', 100 * C2['wf'], 1)
V.put('c2.la', 100 * C2['logit_acc'], 1)
V.put('c2.ba', 100 * C2['base_acc'], 1)
V.put('c2.gb', C2['gb_mase'], 2)
V.put('c2.dhr', C2['dhr_mase'], 2)
V.put('c2.lstm', C2['lstm_mase'], 2)
V.put('c2.gain', 100 * (1 - C2['gb_mase']), 0)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: how do we turn a time series into a machine-learning problem, and how do we know whether the result beats a simple benchmark?',
       '\\textbf{Întrebarea}: cum transformăm o serie de timp într-o problemă de învățare automată și cum știm dacă rezultatul depășește o metodă simplă de referință?'),
     [T('this seminar comes \\textbf{before} Lecture 9: the section ``What you need today\'\' gives every definition the tasks use',
        'seminarul are loc \\textbf{înaintea} Cursului 9: secțiunea „Noțiuni necesare azi” dă toate definițiile folosite în cerințe')]),
    (T('Route', 'Traseul'),
     [T('Part A: a lag table and multi-step forecasts; leakage; one split of a tree; ridge and lasso in one dimension; a conformal interval; MASE and a Diebold--Mariano statistic',
        'Partea A: un tabel de decalaje și prognoze pe mai mulți pași; scurgerea de informație; o împărțire a unui arbore; ridge și lasso într-o dimensiune; un interval conformal; MASE și o statistică Diebold--Mariano'),
      T('Part B: gradient boosting for the Romanian electricity load; recursive, direct and the value of holidays; Romanian inflation three months ahead; the sign of BET returns',
        'Partea B: gradient boosting pentru consumul de energie electrică al României; strategiile recursivă și directă și valoarea sărbătorilor; inflația din România cu trei luni înainte; semnul randamentelor BET'),
      T('Part C: a project idea and an AI answer to audit', 'Partea C: o idee de proiect și un răspuns AI de verificat')]),
    T('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}; each task names its notebook section',
      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}; fiecare cerință indică secțiunea din notebook')))

TB = '>{\\raggedright\\arraybackslash}'
SP = T('Solved, Proposed', 'Rezolvat, Propus')
D.frame(T('Exercise map', 'Harta exercițiilor'), table(
    TB + 'p{1.1cm}' + TB + 'p{7.5cm}' + TB + 'p{1.9cm}' + TB + 'p{1.4cm}',
    T('\\textbf{Task}', '\\textbf{Cerința}') + ' & ' + T('\\textbf{Question}', '\\textbf{Întrebarea}') + ' & ' + T('\\textbf{Type}', '\\textbf{Tipul}') + ' & \\textbf{Model}',
    ['A1, A2 & ' + T('a lag table, recursive and direct forecasts; leakage in a rolling mean', 'un tabel de decalaje, prognoze recursive și directe; scurgerea de informație într-o medie mobilă') + ' & ' + SP + ' & A1',
     'A3, A4 & ' + T('one split of a regression tree; ridge and lasso in one dimension', 'o împărțire a unui arbore de regresie; ridge și lasso într-o dimensiune') + ' & ' + SP + ' & A3',
     'A5, A6 & ' + T('a conformal interval and the pinball loss; MASE and a DM statistic', 'un interval conformal și funcția pinball; MASE și o statistică DM') + ' & ' + SP + ' & A5',
     'B1, B2 & ' + T('electricity load: boosting against the seasonal naive; recursive, direct, holidays', 'consumul de energie: boosting față de metoda naivă sezonieră; recursiv, direct, sărbători') + ' & ' + SP + ' & B1',
     'B3, B4 & ' + T('Romanian inflation; the sign of BET returns', 'inflația din România; semnul randamentelor BET') + ' & ' + SP + ' & B3',
     'C1, C2 & ' + T('a global model as a project; what is wrong in an AI answer?', 'un model global ca proiect; ce este greșit într-un răspuns AI?') + ' & ' + T('Proposed', 'Propus') + ' & B1, B3'],
    size='footnotesize') + items(
    T('\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, following the model',
      '\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați dumneavoastră, după model')))

D.frame(T('Data used', 'Datele folosite'), table(
    'llll', T('\\textbf{Series}', '\\textbf{Seria}') + ' & ' + T('\\textbf{Source}', '\\textbf{Sursa}') + ' & ' + T('\\textbf{Frequency}', '\\textbf{Frecvența}') + ' & ' + T('\\textbf{Period}', '\\textbf{Perioada}'),
    [T('Electricity load, Romania', 'Consumul de energie electrică, România') + ' & ENTSO-E & ' + T('hourly, averaged per day (GW)', 'orar, mediat pe zi (GW)') + ' & 2022--2026',
     T('HICP, 27 EU countries', 'IAPC, 27 de țări UE') + ' & Eurostat (prc\\_hicp\\_minr) & ' + T('monthly, 2015 = 100', 'lunar, 2015 = 100') + ' & 2005--2026',
     'BET & EODHD & ' + T('daily, close', 'zilnic, închidere') + ' & 2000--2026'],
    size='footnotesize') + items(
    T('ENTSO-E: the European Network of Transmission System Operators for Electricity; the same load file as in Chapter 4', 'ENTSO-E: rețeaua europeană a operatorilor de transport și de sistem pentru energie electrică; același fișier de consum ca în Capitolul 4'),
    T('In the notebook: \\texttt{direct\\_frame}, \\texttt{forecast\\_direct}, \\texttt{forecast\\_recursive}, \\texttt{dm\\_raw}, \\texttt{inflation\\_forecasts}, \\texttt{sign\\_frame}; scikit-learn models with fixed seeds; no account or key is needed',
      'În notebook: \\texttt{direct\\_frame}, \\texttt{forecast\\_direct}, \\texttt{forecast\\_recursive}, \\texttt{dm\\_raw}, \\texttt{inflation\\_forecasts}, \\texttt{sign\\_frame}; modele scikit-learn cu semințe fixe; nu este nevoie de cont sau de cheie')))

# =============================================================================
# NOȚIUNI NECESARE AZI
# =============================================================================
D.section('What you need today', 'Noțiuni necesare azi')

D.frame(T('What you need today (1/4): from a series to a table', 'Noțiuni necesare azi (1/4): de la o serie la un tabel'), items(
    (T('\\textbf{Supervised learning}: learn $\\hat f$ from pairs (features $\\mathbf{x}_i$, target $y_i$) so that $\\hat f(\\mathbf{x})$ is close to $y$ on \\textbf{new} data',
       '\\textbf{Învățarea supervizată}: învățăm $\\hat f$ din perechi (variabile explicative $\\mathbf{x}_i$, ținta $y_i$) astfel încît $\\hat f(\\mathbf{x})$ să fie aproape de $y$ pe date \\textbf{noi}'),
     [T('for a series: at the origin $t$, features = lags $y_t, y_{t-1}, \\dots$, rolling means of past values, calendar of the target day; target $y_{t+h}$',
        'pentru o serie: la originea $t$, variabilele = decalajele $y_t, y_{t-1}, \\dots$, medii mobile ale valorilor trecute, calendarul zilei-țintă; ținta $y_{t+h}$')]),
    (T('\\textbf{Multi-step strategies}', '\\textbf{Strategii pe mai mulți pași}'),
     [T('recursive: one model for $h = 1$, applied again with its own forecasts as inputs', 'recursivă: un model pentru $h = 1$, aplicat din nou cu propriile prognoze ca intrări'),
      T('direct: one model for each horizon $h$, trained on the target $y_{t+h}$', 'directă: cîte un model pentru fiecare orizont $h$, antrenat pe ținta $y_{t+h}$'),
      T('MIMO (multiple-input multiple-output): one model with $H$ outputs', 'MIMO (multiple-input multiple-output): un model cu $H$ ieșiri')]),
    T('A series of length $n$ with $p$ lags gives $n - p - h + 1$ complete rows for horizon $h$', 'O serie de lungime $n$ cu $p$ decalaje dă $n - p - h + 1$ rînduri complete pentru orizontul $h$')))

D.frame(T('What you need today (2/4): validation and leakage', 'Noțiuni necesare azi (2/4): validare și scurgere de informație'), items(
    (T('\\textbf{Walk-forward} validation: train on data up to the origin, forecast the next block, move the origin; errors pooled over all origins',
       'Validarea \\textbf{walk-forward}: antrenăm pe datele de pînă la origine, prognozăm blocul următor, mutăm originea; erorile se combină pe toate originile'),
     [T('never random $K$-fold for forecasting models with overlapping targets or exogenous features', 'niciodată $K$-fold aleator pentru modele de prognoză cu ținte suprapuse sau cu variabile exogene')]),
    (T('\\textbf{Leakage}: information unavailable at the origin enters the features or the training', '\\textbf{Scurgerea de informație} (leakage): informație indisponibilă la origine intră în variabile sau în antrenare'),
     [T('a rolling mean that contains $y_{t+1}$; scaling fitted on the full sample; overlapping targets without a gap; actual instead of forecast weather', 'o medie mobilă care conține $y_{t+1}$; scalare estimată pe tot eșantionul; ținte suprapuse fără spațiu (gap); vremea efectivă în locul celei prognozate'),
      T('calendar variables (weekday, holidays) are known in advance: no leak', 'variabilele de calendar (ziua săptămînii, sărbătorile) se cunosc dinainte: nu există scurgere')]),
    T('Benchmarks (Chapter 0): naive $\\hat y_{T+h} = y_T$, seasonal naive $\\hat y_{T+h} = y_{T+h-m}$ (with $m = 7$ for daily data: same weekday last week)',
      'Metode de referință (Capitolul 0): naivă $\\hat y_{T+h} = y_T$, naivă sezonieră $\\hat y_{T+h} = y_{T+h-m}$ (cu $m = 7$ pentru date zilnice: aceeași zi a săptămînii trecute)')))

D.frame(T('What you need today (3/4): regularisation and trees', 'Noțiuni necesare azi (3/4): regularizare și arbori'), items(
    (T('\\textbf{Ridge}: $\\min \\sum_t (y_t - \\mathbf{x}_t\'\\beta)^2 + \\lambda\\sum_j\\beta_j^2$; \\textbf{lasso}: penalty $\\lambda\\sum_j|\\beta_j|$ (some $\\hat\\beta_j = 0$)',
       '\\textbf{Ridge}: $\\min \\sum_t (y_t - \\mathbf{x}_t\'\\beta)^2 + \\lambda\\sum_j\\beta_j^2$; \\textbf{lasso}: penalizarea $\\lambda\\sum_j|\\beta_j|$ (unii $\\hat\\beta_j = 0$)'),
     [T('one standardised feature, $S_{xy} = \\sum x_ty_t$, $S_{xx} = \\sum x_t^2$: OLS $S_{xy}/S_{xx}$; ridge $S_{xy}/(S_{xx} + \\lambda)$; lasso $\\mathrm{sign}(S_{xy})\\max(|S_{xy}| - \\lambda/2, 0)/S_{xx}$',
        'o variabilă standardizată, $S_{xy} = \\sum x_ty_t$, $S_{xx} = \\sum x_t^2$: OLS $S_{xy}/S_{xx}$; ridge $S_{xy}/(S_{xx} + \\lambda)$; lasso $\\mathrm{sign}(S_{xy})\\max(|S_{xy}| - \\lambda/2, 0)/S_{xx}$')]),
    (T('\\textbf{Regression tree}: split ``$x \\le c$?\'\'; each side predicts the mean of its targets; the best $c$ minimises the total SSE (sum of squared errors)',
       '\\textbf{Arborele de regresie}: împărțirea „$x \\le c$?”; fiecare parte prognozează media țintelor ei; cel mai bun $c$ minimizează SSE totală (suma pătratelor erorilor)'),
     [T('candidates: midpoints between consecutive sorted values of $x$; a tree cannot forecast outside the range of the training targets', 'candidații: mijloacele dintre valorile consecutive ordonate ale lui $x$; un arbore nu poate prognoza în afara intervalului țintelor de antrenare')]),
    (T('\\textbf{Random forest}: average of many deep trees on bootstrap samples; \\textbf{gradient boosting} (GB): small trees added one by one on the residuals, with a learning rate $\\nu$',
       '\\textbf{Random forest}: media multor arbori adînci pe eșantioane bootstrap; \\textbf{gradient boosting} (GB): arbori mici adăugați unul cîte unul pe reziduuri, cu o rată de învățare $\\nu$'),
     [T('scikit-learn: \\texttt{HistGradientBoostingRegressor}; the same family as LightGBM and XGBoost', 'scikit-learn: \\texttt{HistGradientBoostingRegressor}; aceeași familie ca LightGBM și XGBoost')])), 'footnotesize')

D.frame(T('What you need today (4/4): evaluation and intervals', 'Noțiuni necesare azi (4/4): evaluare și intervale'), items(
    (T('\\textbf{MASE} = MAE / (in-sample MAE of the seasonal naive method); below 1: better than that benchmark', '\\textbf{MASE} = MAE / (MAE în eșantion a metodei naive sezoniere); sub 1: mai bună decît acea metodă de referință'),
     [T('\\textbf{Diebold--Mariano} (Chapter 4): $d_t = L(e_{1t}) - L(e_{2t})$, $DM = \\bar d / \\sqrt{\\hat s^2/n}$; $|DM|$ above the critical value: different accuracy',
        '\\textbf{Diebold--Mariano} (Capitolul 4): $d_t = L(e_{1t}) - L(e_{2t})$, $DM = \\bar d / \\sqrt{\\hat s^2/n}$; $|DM|$ peste valoarea critică: acuratețe diferită')]),
    (T('\\textbf{Pinball loss} of a $\\tau$-quantile forecast $q$: $\\tau(y - q)$ if $y \\ge q$, else $(1 - \\tau)(q - y)$', '\\textbf{Funcția pinball} pentru o prognoză cuantilică $q$ de nivel $\\tau$: $\\tau(y - q)$ dacă $y \\ge q$, altfel $(1 - \\tau)(q - y)$'),
     [T('\\textbf{split conformal} interval: $\\hat y \\pm \\hat q$, $\\hat q$ = the $\\lceil (n+1)(1 - \\alpha) \\rceil$-th smallest of $n$ absolute calibration errors', 'intervalul \\textbf{conformal prin împărțire}: $\\hat y \\pm \\hat q$, $\\hat q$ = a $\\lceil (n+1)(1 - \\alpha) \\rceil$-a cea mai mică dintre $n$ erori absolute de calibrare')]),
    (T('\\textbf{Classification} of the sign: the baseline is the majority class of the training data, not 50\\%', '\\textbf{Clasificarea} semnului: reperul este clasa majoritară din datele de antrenare, nu 50\\%'),
     [T('$z = (\\text{acc} - \\text{acc}_0)/\\sqrt{\\text{acc}_0(1 - \\text{acc}_0)/n}$ compares the accuracy with the baseline accuracy $\\text{acc}_0$', '$z = (\\text{acc} - \\text{acc}_0)/\\sqrt{\\text{acc}_0(1 - \\text{acc}_0)/n}$ compară acuratețea cu acuratețea reperului, $\\text{acc}_0$')])), 'footnotesize')

# =============================================================================
# PARTEA A
# =============================================================================
D.section('Part A: computations on paper', 'Partea A: calcule pe hîrtie')

D.solved(T('A1: a lag table and two strategies', 'A1: un tabel de decalaje și două strategii'),
         items(T('Series $y_1, \\dots, y_8$: 10, 12, 11, 13, 14, 13, 15, 16.', 'Seria $y_1, \\dots, y_8$: 10, 12, 11, 13, 14, 13, 15, 16.'),
               T('1. Build the table with the features $y_{t-1}$, $y_{t-2}$ and the target $y_t$; how many rows does it have?', '1. Construiți tabelul cu variabilele $y_{t-1}$, $y_{t-2}$ și ținta $y_t$; cîte rînduri are?'),
               T('2. With the fitted model $\\hat y_t = 2 + 0.6\\,y_{t-1} + 0.3\\,y_{t-2}$, compute the recursive forecasts $\\hat y_9$, $\\hat y_{10}$, $\\hat y_{11}$.', '2. Cu modelul estimat $\\hat y_t = 2 + 0{,}6\\,y_{t-1} + 0{,}3\\,y_{t-2}$, calculați prognozele recursive $\\hat y_9$, $\\hat y_{10}$, $\\hat y_{11}$.'),
               T('3. A direct two-step model is $\\hat y_{t+2} = 3 + 0.5\\,y_t + 0.3\\,y_{t-1}$: compute $\\hat y_{10}$ and the number of rows of its table.', '3. Un model direct pe doi pași este $\\hat y_{t+2} = 3 + 0{,}5\\,y_t + 0{,}3\\,y_{t-1}$: calculați $\\hat y_{10}$ și numărul de rînduri al tabelului lui.'),
               T('Report: the table, three recursive forecasts, one direct forecast, two row counts.', 'Raportați: tabelul, trei prognoze recursive, o prognoză directă, două numere de rînduri.')),
         table('cccc', '$t$ & $y_{t-1}$ & $y_{t-2}$ & $y_t$', A1ROWS, size='scriptsize') + items(
             T('1. @{a1.n1} rows ($t = 3, \\dots, 8$)', '1. @{a1.n1} rînduri ($t = 3, \\dots, 8$)'),
             T('2. $\\hat y_9 = 2 + 0.6 \\cdot 16 + 0.3 \\cdot 15 = @{a1.r1}$; $\\hat y_{10} = 2 + 0.6 \\cdot @{a1.r1} + 0.3 \\cdot 16 = @{a1.r2}$; $\\hat y_{11} = @{a1.r3}$', '2. $\\hat y_9 = 2 + 0{,}6 \\cdot 16 + 0{,}3 \\cdot 15 = @{a1.r1}$; $\\hat y_{10} = 2 + 0{,}6 \\cdot @{a1.r1} + 0{,}3 \\cdot 16 = @{a1.r2}$; $\\hat y_{11} = @{a1.r3}$'),
             T('3. $\\hat y_{10} = 3 + 0.5 \\cdot 16 + 0.3 \\cdot 15 = @{a1.d2}$; @{a1.n2} rows (origins $t = 2, \\dots, 6$)', '3. $\\hat y_{10} = 3 + 0{,}5 \\cdot 16 + 0{,}3 \\cdot 15 = @{a1.d2}$; @{a1.n2} rînduri (originile $t = 2, \\dots, 6$)')),
         size='scriptsize')

D.proposed(T('A2: spot the leak', 'A2: găsiți scurgerea de informație'),
           items(T('Daily series 10, 12, 11, 13, 14; the target is the last value, 14. Model: A1.', 'Seria zilnică 10, 12, 11, 13, 14; ținta este ultima valoare, 14. Model: A1.'),
                 T('1. Compute the correct 3-day mean feature for this target, and the mean that a careless \\texttt{rolling(3).mean()} without \\texttt{shift(1)} would give.', '1. Calculați media corectă pe 3 zile pentru această țintă și media pe care ar da-o un \\texttt{rolling(3).mean()} scris neglijent, fără \\texttt{shift(1)}.'),
                 T('2. For each choice, say whether it leaks: (a) a scaler fitted on 2022--2026 before the split; (b) holiday dummies of the target day; (c) the temperature measured on the target day; (d) 5-fold shuffled CV for a 21-day target.',
                   '2. Pentru fiecare alegere, spuneți dacă produce scurgere: (a) o scalare estimată pe 2022--2026 înainte de împărțire; (b) indicatorii de sărbătoare ai zilei-țintă; (c) temperatura măsurată în ziua-țintă; (d) validare cu 5 grupuri amestecate pentru o țintă pe 21 de zile.'),
                 T('Report: two means and four verdicts with one reason each.', 'Raportați: două medii și patru verdicte, fiecare cu un motiv.')),
           items(T('1. Correct: $(11 + 13 + 12)/3 = @{a2.c}$; leaky: $(11 + 13 + 14)/3 = @{a2.l}$, it contains the target', '1. Corect: $(11 + 13 + 12)/3 = @{a2.c}$; cu scurgere: $(11 + 13 + 14)/3 = @{a2.l}$, conține ținta'),
                 T('2. (a) leak: test-period means and variances enter training; (b) no leak: known in advance; (c) leak, unless a weather forecast made at the origin is used; (d) leak: overlapping targets fall in train and test',
                   '2. (a) scurgere: mediile și varianțele din perioada de test intră în antrenare; (b) fără scurgere: se cunosc dinainte; (c) scurgere, dacă nu se folosește prognoza meteo făcută la origine; (d) scurgere: țintele suprapuse cad și în antrenare, și în test')),
           size='scriptsize')

D.solved(T('A3: one split of a regression tree', 'A3: o împărțire a unui arbore de regresie'),
         items(T('Six days: $x$ = load on the same weekday last week, $y$ = load today (GW): $(4.8, 5.0)$, $(5.0, 5.2)$, $(5.6, 5.9)$, $(6.0, 6.0)$, $(6.4, 6.6)$, $(6.9, 7.0)$.',
                 'Șase zile: $x$ = consumul din aceeași zi a săptămînii trecute, $y$ = consumul de azi (GW): $(4{,}8;\\ 5{,}0)$, $(5{,}0;\\ 5{,}2)$, $(5{,}6;\\ 5{,}9)$, $(6{,}0;\\ 6{,}0)$, $(6{,}4;\\ 6{,}6)$, $(6{,}9;\\ 7{,}0)$.'),
               T('1. List the five candidate thresholds.', '1. Enumerați cele cinci praguri candidate.'),
               T('2. For each, compute the two leaf means and the total SSE.', '2. Pentru fiecare, calculați cele două medii ale frunzelor și SSE totală.'),
               T('3. Choose the split and give the forecast for $x = 6.2$.', '3. Alegeți împărțirea și dați prognoza pentru $x = 6{,}2$.'),
               T('Report: a table of five rows, the chosen threshold, one forecast.', 'Raportați: un tabel cu cinci rînduri, pragul ales, o prognoză.')),
         table('cccc', T('threshold', 'prag') + ' & ' + T('left mean', 'media stînga') + ' & ' + T('right mean', 'media dreapta') + ' & SSE', A3ROWS, size='scriptsize') + items(
             T('Without a split: mean @{a3.mean}, SSE @{a3.sst}', 'Fără împărțire: media @{a3.mean}, SSE @{a3.sst}'),
             T('Best: $x \\le @{a3.thr}$, SSE @{a3.sse}; leaves @{a3.lm} and @{a3.rm} GW', 'Cea mai bună: $x \\le @{a3.thr}$, SSE @{a3.sse}; frunzele @{a3.lm} și @{a3.rm} GW'),
             T('$x = 6.2 > @{a3.thr}$: forecast @{a3.rm} GW', '$x = 6{,}2 > @{a3.thr}$: prognoza @{a3.rm} GW')),
         size='scriptsize')

D.proposed(T('A4: ridge and lasso in one dimension', 'A4: ridge și lasso într-o dimensiune'),
           items(T('One standardised feature with $S_{xy} = 40$ and $S_{xx} = 50$. Model: A3 and the formulas of ``What you need today (3/4)\'\'.', 'O variabilă standardizată cu $S_{xy} = 40$ și $S_{xx} = 50$. Model: A3 și formulele din „Noțiuni necesare azi (3/4)”.'),
                 T('1. Compute the OLS coefficient.', '1. Calculați coeficientul OLS.'),
                 T('2. Compute the ridge coefficient for $\\lambda = 10$ and $\\lambda = 50$.', '2. Calculați coeficientul ridge pentru $\\lambda = 10$ și $\\lambda = 50$.'),
                 T('3. Compute the lasso coefficient for $\\lambda = 20$ and $\\lambda = 100$.', '3. Calculați coeficientul lasso pentru $\\lambda = 20$ și $\\lambda = 100$.'),
                 T('4. Explain in one sentence which method can drop a lag from the model.', '4. Explicați într-o frază ce metodă poate elimina un decalaj din model.'),
                 T('Report: five coefficients and one sentence.', 'Raportați: cinci coeficienți și o frază.')),
           items(T('1. OLS $40/50 = @{a4.ols}$', '1. OLS $40/50 = @{a4.ols}$'),
                 T('2. Ridge: $40/60 = @{a4.r10}$; $40/100 = @{a4.r50}$', '2. Ridge: $40/60 = @{a4.r10}$; $40/100 = @{a4.r50}$'),
                 T('3. Lasso: $(40 - 10)/50 = @{a4.l20}$; $\\lambda/2 = 50 > 40$: exactly @{a4.l100}', '3. Lasso: $(40 - 10)/50 = @{a4.l20}$; $\\lambda/2 = 50 > 40$: exact @{a4.l100}'),
                 T('4. Only the lasso sets coefficients exactly to zero (soft thresholding); ridge only shrinks them', '4. Doar lasso face coeficienții exact zero (prag moale); ridge doar îi contractă')),
           size='scriptsize')

D.solved(T('A5: a conformal interval and the pinball loss', 'A5: un interval conformal și funcția pinball'),
         items(T('A model forecasts tomorrow\'s load $\\hat y = 6.10$ GW. On 9 calibration days its absolute errors were 0.12, 0.30, 0.05, 0.22, 0.41, 0.18, 0.09, 0.27, 0.35 GW.',
                 'Un model prognozează consumul de mîine, $\\hat y = 6{,}10$ GW. În 9 zile de calibrare, erorile lui absolute au fost 0,12; 0,30; 0,05; 0,22; 0,41; 0,18; 0,09; 0,27; 0,35 GW.'),
               T('1. Build the 80\\% split conformal interval.', '1. Construiți intervalul conformal de 80\\%.'),
               T('2. A 90\\% quantile forecast is $q = 6.4$ GW: compute its pinball loss if $y = 6.0$ and if $y = 6.6$.', '2. O prognoză cuantilică de 90\\% este $q = 6{,}4$ GW: calculați funcția pinball dacă $y = 6{,}0$ și dacă $y = 6{,}6$.'),
               T('Report: $k$, $\\hat q$, the interval, two losses.', 'Raportați: $k$, $\\hat q$, intervalul, două valori ale pierderii.')),
         items(T('1. Sorted: @{a5.sorted}; $k = \\lceil 10 \\cdot 0.8 \\rceil = @{a5.k}$; $\\hat q = @{a5.half}$', '1. Ordonate: @{a5.sorted}; $k = \\lceil 10 \\cdot 0{,}8 \\rceil = @{a5.k}$; $\\hat q = @{a5.half}$'),
               T('interval $6.10 \\pm @{a5.half} = [@{a5.lo}, @{a5.hi}]$ GW', 'intervalul $6{,}10 \\pm @{a5.half} = [@{a5.lo}, @{a5.hi}]$ GW'),
               T('2. $y = 6.0 < q$: $(1 - 0.9)(6.4 - 6.0) = @{a5.p1}$; $y = 6.6 > q$: $0.9 \\cdot 0.2 = @{a5.p2}$', '2. $y = 6{,}0 < q$: $(1 - 0{,}9)(6{,}4 - 6{,}0) = @{a5.p1}$; $y = 6{,}6 > q$: $0{,}9 \\cdot 0{,}2 = @{a5.p2}$'),
               T('an outcome above a high quantile costs nine times more per GW: the loss pushes $q$ up until only 10\\% of outcomes exceed it', 'o valoare peste o cuantilă mare costă de nouă ori mai mult pe GW: pierderea împinge $q$ în sus pînă cînd doar 10\\% dintre valori o depășesc')),
         size='scriptsize')

D.proposed(T('A6: MASE and a Diebold--Mariano statistic', 'A6: MASE și o statistică Diebold--Mariano'),
           items(T('Average absolute errors (GW) of two models at 6 origins: A = 0.30, 0.25, 0.40, 0.28, 0.35, 0.22; B = 0.32, 0.30, 0.38, 0.35, 0.41, 0.30; MASE scale 0.31 GW. Model: A5 and ``What you need today (4/4)\'\'.',
                   'Erorile absolute medii (GW) a două modele la 6 origini: A = 0,30; 0,25; 0,40; 0,28; 0,35; 0,22; B = 0,32; 0,30; 0,38; 0,35; 0,41; 0,30; scala MASE 0,31 GW. Model: A5 și „Noțiuni necesare azi (4/4)”.'),
                 T('1. Compute the MAE and the MASE of each model.', '1. Calculați MAE și MASE pentru fiecare model.'),
                 T('2. Compute $d_t = A_t - B_t$, its mean and standard deviation, and $DM = \\bar d/(s/\\sqrt{6})$.', '2. Calculați $d_t = A_t - B_t$, media și abaterea standard a lui și $DM = \\bar d/(s/\\sqrt{6})$.'),
                 T('3. Compare $|DM|$ with the 5\\% critical value of $t(5)$ and conclude.', '3. Comparați $|DM|$ cu valoarea critică de 5\\% a distribuției $t(5)$ și formulați concluzia.'),
                 T('Report: four accuracy numbers, $\\bar d$, $DM$ and a decision.', 'Raportați: patru valori de acuratețe, $\\bar d$, $DM$ și o decizie.')),
           items(T('1. MAE: A @{a6.mae_a}, B @{a6.mae_b}; MASE: @{a6.mase_a} and @{a6.mase_b}', '1. MAE: A @{a6.mae_a}, B @{a6.mae_b}; MASE: @{a6.mase_a} și @{a6.mase_b}'),
                 T('2. $d$ = @{a6.d}; $\\bar d = @{a6.dbar}$, $s = @{a6.s}$, $DM = @{a6.t}$', '2. $d$ = @{a6.d}; $\\bar d = @{a6.dbar}$, $s = @{a6.s}$, $DM = @{a6.t}$'),
                 T('3. $|DM| > @{a6.crit}$ (p @{a6.p}): A is significantly more accurate, although only 6 origins are available', '3. $|DM| > @{a6.crit}$ (p @{a6.p}): A este semnificativ mai precis, deși avem doar 6 origini')),
           size='scriptsize')

# =============================================================================
# PARTEA B
# =============================================================================
D.section('Part B: real data and interpretation', 'Partea B: date reale și interpretare')

D.task(T('B1: gradient boosting for the electricity load [Solved]', 'B1: gradient boosting pentru consumul de energie electrică [Rezolvat]'),
       T('does a gradient-boosting model forecast next week\'s daily load better than ``the same weekday last week\'\'?', 'prognozează un model gradient boosting consumul zilnic din săptămîna următoare mai bine decît „aceeași zi a săptămînii trecute”?'),
       T('daily mean load of Romania (GW), 2022--2026; weekly origins from 5 January to 22 June 2026; horizons 1--7 days', 'consumul mediu zilnic al României (GW), 2022--2026; origini săptămînale de la 5 ianuarie la 22 iunie 2026; orizonturile 1--7 zile'),
       [T('Build the direct tables (14 lags, 7- and 28-day means, the same weekday, calendar of the target day) with \\texttt{direct\\_frame}.', 'Construiți tabelele directe (14 decalaje, medii pe 7 și 28 de zile, aceeași zi a săptămînii, calendarul zilei-țintă) cu \\texttt{direct\\_frame}.'),
        T('At each origin, train one \\texttt{HistGradientBoostingRegressor} per horizon on the data before the origin and forecast the next 7 days.', 'La fiecare origine, antrenați cîte un \\texttt{HistGradientBoostingRegressor} pentru fiecare orizont pe datele de dinaintea originii și prognozați următoarele 7 zile.'),
        T('Compute the MAE, RMSE and MASE of GB and of the seasonal naive forecast.', 'Calculați MAE, RMSE și MASE pentru GB și pentru prognoza naivă sezonieră.'),
        T('Run the Diebold--Mariano test on the origin-average absolute errors.', 'Aplicați testul Diebold--Mariano pe erorile absolute medii pe origine.'),
        T('Interpretation: at which horizons does GB gain most, and why?', 'Interpretare: la ce orizonturi cîștigă GB cel mai mult și de ce?')],
       T('six accuracy numbers, a DM statistic with its p-value, one sentence', 'șase valori de acuratețe, o statistică DM cu valoarea p, o frază'), size='footnotesize', nb='B1')

D.frame(T('B1: solution [Solved]', 'B1: rezolvare [Rezolvat]'), fig('ch9_sem_b1', h='0.34') + items(
    T('@{b1.no} origins, 7 horizons; MAE: GB @{b1.gb.mae} MW, seasonal naive @{b1.sn.mae} MW; RMSE @{b1.gb.rmse} and @{b1.sn.rmse} MW; MASE @{b1.gb.mase} and @{b1.sn.mase} (scale @{b1.scale} MW)',
      '@{b1.no} de origini, 7 orizonturi; MAE: GB @{b1.gb.mae} MW, naivă sezonieră @{b1.sn.mae} MW; RMSE @{b1.gb.rmse} și @{b1.sn.rmse} MW; MASE @{b1.gb.mase} și @{b1.sn.mase} (scala @{b1.scale} MW)'),
    T('DM (with the HLN correction) $= @{b1.dm}$, p @{b1.p}; the worst origin for GB: @{b1.wo} (MAE @{b1.wm} MW)', 'DM (cu corecția HLN) $= @{b1.dm}$, p @{b1.p}; cea mai proastă origine pentru GB: @{b1.wo} (MAE @{b1.wm} MW)'),
    T('Interpretation: one day ahead GB has @{b1.h1g} MW against @{b1.h1s} MW: it uses yesterday\'s level, which the seasonal naive ignores; at 7 days (@{b1.h7g} and @{b1.h7s} MW) both rely on last week\'s pattern',
      'Interpretare: cu o zi înainte, GB are @{b1.h1g} MW față de @{b1.h1s} MW: folosește nivelul de ieri, pe care metoda naivă sezonieră îl ignoră; la 7 zile (@{b1.h7g} și @{b1.h7s} MW) ambele se bazează pe tiparul săptămînii trecute')) + qlsem(),
    'scriptsize')

D.task(T('B2: recursive, direct and the value of holidays [Proposed]', 'B2: recursiv, direct și valoarea sărbătorilor [Propus]'),
       T('does the recursive strategy forecast as well as the direct one, and how much do the holiday features help?', 'prognozează strategia recursivă la fel de bine ca strategia directă și cît ajută variabilele de sărbătoare?'),
       T('the data and origins of B1; model: B1', 'datele și originile din B1; model: B1'),
       [T('Repeat B1 with the recursive strategy (\\texttt{forecast\\_recursive}).', 'Repetați B1 cu strategia recursivă (\\texttt{forecast\\_recursive}).'),
        T('Repeat the direct model without the three holiday features (easter, xmas, holiday).', 'Repetați modelul direct fără cele trei variabile de sărbătoare (easter, xmas, holiday).'),
        T('Report the MAE of the three models on all days, on the public-holiday days and on the other days.', 'Raportați MAE pentru cele trei modele pe toate zilele, pe zilele de sărbătoare legală și pe celelalte zile.'),
        T('Interpretation: which feature group explains most of the difference, and on which days?', 'Interpretare: ce grup de variabile explică cea mai mare parte a diferenței și în ce zile?')],
       T('nine MAE values and one sentence', 'nouă valori MAE și o frază'), size='footnotesize', nb='B2')

D.frame(T('B2: solution [Proposed]', 'B2: rezolvare [Propus]'), fig('ch9_sem_b2', h='0.34') + items(
    T('MAE (MW): direct @{b2.d}, recursive @{b2.r}, direct without holidays @{b2.n}', 'MAE (MW): direct @{b2.d}, recursiv @{b2.r}, direct fără sărbători @{b2.n}'),
    T('On the @{b2.nh} holiday days: @{b2.d.h}, @{b2.r.h} and @{b2.n.h} MW; on the other days: @{b2.d.o}, @{b2.r.o} and @{b2.n.o} MW', 'În cele @{b2.nh} zile de sărbătoare: @{b2.d.h}, @{b2.r.h} și @{b2.n.h} MW; în celelalte zile: @{b2.d.o}, @{b2.r.o} și @{b2.n.o} MW'),
    T('Interpretation: removing the holiday flags raises the error on holidays (@{b2.d.h} to @{b2.n.h} MW) and leaves the other days almost unchanged; holidays stay the hardest days for every model; recursive and direct differ little',
      'Interpretare: eliminarea indicatorilor de sărbătoare crește eroarea în zilele de sărbătoare (de la @{b2.d.h} la @{b2.n.h} MW) și lasă celelalte zile aproape neschimbate; sărbătorile rămîn cele mai grele zile pentru toate modelele; strategiile recursivă și directă diferă puțin')) + qlsem(),
    'scriptsize', instructor_only=True)

D.task(T('B3: Romanian inflation three months ahead [Solved]', 'B3: inflația din România cu trei luni înainte [Rezolvat]'),
       T('can machine learning forecast Romanian inflation three months ahead better than the random walk and a linear AR model?', 'poate învățarea automată prognoza inflația din România cu trei luni înainte mai bine decît mersul aleator și un model AR liniar?'),
       T('12-month HICP inflation of 27 EU countries (Eurostat), monthly, 2005--2026', 'inflația anuală IAPC a 27 de țări UE (Eurostat), lunar, 2005--2026'),
       [T('Build the features of each country (inflation lags, monthly changes, calendar month) and the target $\\pi_{t+3} - \\pi_t$ with \\texttt{inflation\\_frame}.', 'Construiți variabilele fiecărei țări (decalajele inflației, variațiile lunare, luna calendaristică) și ținta $\\pi_{t+3} - \\pi_t$ cu \\texttt{inflation\\_frame}.'),
        T('Every January from 2016, refit AR, lasso, random forest and GB on Romania only, and a global GB on all countries; forecast every month of the year.', 'În fiecare ianuarie din 2016, reestimați AR, lasso, random forest și GB doar pe România, precum și un GB global pe toate țările; prognozați fiecare lună a anului.'),
        T('Compute the RMSE of each model and its ratio to the random walk.', 'Calculați RMSE pentru fiecare model și raportul lui față de mersul aleator.'),
        T('Run the DM test for AR against RW and for the global against the local GB.', 'Aplicați testul DM pentru AR față de RW și pentru GB global față de GB local.'),
        T('Interpretation: what happened to all models in 2021--2023?', 'Interpretare: ce s-a întîmplat cu toate modelele în 2021--2023?')],
       T('six RMSEs, two DM tests and one sentence', 'șase valori RMSE, două teste DM și o frază'), size='footnotesize', nb='B3')

D.frame(T('B3: solution [Solved]', 'B3: rezolvare [Rezolvat]'), fig('ch9_sem_b3', h='0.32') + items(
    T('@{b3.n} origins, @{b3.first}--@{b3.last}; RMSE (pp): RW @{b3.RW}, AR @{b3.AR} (@{b3.AR.r}), lasso @{b3.Lasso} (@{b3.Lasso.r}), RF @{b3.RF} (@{b3.RF.r}), GB local @{b3.GBlocal} (@{b3.GBlocal.r}), GB global @{b3.GBglobal} (@{b3.GBglobal.r})',
      '@{b3.n} de origini, @{b3.first}--@{b3.last}; RMSE (pp): RW @{b3.RW}, AR @{b3.AR} (@{b3.AR.r}), lasso @{b3.Lasso} (@{b3.Lasso.r}), RF @{b3.RF} (@{b3.RF.r}), GB local @{b3.GBlocal} (@{b3.GBlocal.r}), GB global @{b3.GBglobal} (@{b3.GBglobal.r})'),
    T('DM: AR against RW $@{b3.dm_ar_rw}$ (p @{b3.dm_ar_rw.p}); GB global against GB local $@{b3.dm_gl_loc}$ (p @{b3.dm_gl_loc.p})', 'DM: AR față de RW $@{b3.dm_ar_rw}$ (p @{b3.dm_ar_rw.p}); GB global față de GB local $@{b3.dm_gl_loc}$ (p @{b3.dm_gl_loc.p})'),
    T('Interpretation: from mid-2021 to mid-2023 every RMSE jumps (RW @{b3.RW.s}, AR @{b3.AR.s}, GB global @{b3.GBglobal.s} pp): the energy shock was not in the past of inflation; pooling 27 countries helps the trees, but no model is significantly better than AR',
      'Interpretare: de la jumătatea lui 2021 pînă la jumătatea lui 2023, toate valorile RMSE cresc (RW @{b3.RW.s}, AR @{b3.AR.s}, GB global @{b3.GBglobal.s} pp): șocul energetic nu se afla în trecutul inflației; combinarea a 27 de țări ajută arborii, dar niciun model nu este semnificativ mai bun decît AR')) + qlsem(),
    'scriptsize')

D.task(T('B4: the sign of BET returns [Proposed]', 'B4: semnul randamentelor BET [Propus]'),
       T('can a classifier predict whether the BET rises tomorrow better than ``it always rises\'\'?', 'poate un clasificator prognoza dacă BET crește mîine mai bine decît „crește întotdeauna”?'),
       T('daily BET log returns since 2000; test years 2015--2026; model: B3 (yearly refits)', 'randamentele logaritmice zilnice ale BET din 2000; anii de test 2015--2026; model: B3 (reestimări anuale)'),
       [T('Build the features of \\texttt{sign\\_frame} (five lagged returns, sums over 5, 21, 63 days, volatility over 21 and 63 days) and the target $1\\{r_{t+1} > 0\\}$.', 'Construiți variabilele din \\texttt{sign\\_frame} (cinci randamente întîrziate, sume pe 5, 21, 63 de zile, volatilitatea pe 21 și 63 de zile) și ținta $1\\{r_{t+1} > 0\\}$.'),
        T('Each January, fit a logistic regression and a gradient-boosting classifier on all earlier days; the baseline predicts the majority class of the training data.', 'În fiecare ianuarie, estimați o regresie logistică și un clasificator gradient boosting pe toate zilele anterioare; reperul prognozează clasa majoritară din datele de antrenare.'),
        T('Report the accuracy of the three, the $z$ statistic against the baseline and the AUC.', 'Raportați acuratețea celor trei, statistica $z$ față de reper și AUC.'),
        T('Interpretation: what does the result say about the weak-form efficiency of the Bucharest market?', 'Interpretare: ce spune rezultatul despre eficiența în formă slabă a pieței de la București?')],
       T('three accuracies, two $z$ statistics, two AUCs and one sentence', 'trei valori ale acurateței, două statistici $z$, două valori AUC și o frază'), size='footnotesize', nb='B4')

D.frame(T('B4: solution [Proposed]', 'B4: rezolvare [Propus]'), fig('ch9_sem_b4', h='0.32') + items(
    T('@{b4.n} test days; up days @{b4.up}\\%; accuracy: baseline @{b4.base}\\%, logit @{b4.Logit}\\% ($z = @{b4.Logit.z}$, p @{b4.Logit.p}), GB @{b4.GB}\\% ($z = @{b4.GB.z}$, p @{b4.GB.p})',
      '@{b4.n} de zile de test; zile de creștere @{b4.up}\\%; acuratețea: reper @{b4.base}\\%, logit @{b4.Logit}\\% ($z = @{b4.Logit.z}$, p @{b4.Logit.p}), GB @{b4.GB}\\% ($z = @{b4.GB.z}$, p @{b4.GB.p})'),
    T('AUC: logit @{b4.Logit.auc}, GB @{b4.GB.auc}; the logit predicts ``up\'\' on @{b4.Logit.up}\\% of the days', 'AUC: logit @{b4.Logit.auc}, GB @{b4.GB.auc}; modelul logit prognozează „creștere” în @{b4.Logit.up}\\% din zile'),
    T('Interpretation: no gain over the baseline; the past returns carry almost no information about the next sign, as weak-form efficiency says',
      'Interpretare: niciun cîștig față de reper; randamentele trecute nu conțin aproape nicio informație despre semnul următor, cum spune eficiența în formă slabă')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# PARTEA C
# =============================================================================
D.section('Part C: open questions and AI critique', 'Partea C: întrebări deschise și critica unui răspuns AI')

D.frame(T('C1: a global model for European electricity load [Proposed]', 'C1: un model global pentru consumul de energie din Europa [Propus]'), items(
    (T('\\textbf{Idea}: ENTSO-E publishes the hourly load of every European country; one global GB model trained on 20 countries may forecast Romania better than a local one',
       '\\textbf{Ideea}: ENTSO-E publică consumul orar al fiecărei țări europene; un model GB global antrenat pe 20 de țări poate prognoza România mai bine decît unul local'),
     [T('features: lags scaled by each country\'s mean, weekday, national holidays, a country identifier', 'variabile: decalaje scalate cu media fiecărei țări, ziua săptămînii, sărbătorile naționale, un identificator al țării')]),
    (T('Questions to answer', 'Întrebări la care trebuie să răspundeți'),
     [T('1. Does the global model beat the local GB and DHR on the 26 origins of Chapter 4?', '1. Depășește modelul global GB local și DHR pe cele 26 de origini din Capitolul 4?'),
      T('2. Does it forecast Orthodox Easter better, although most countries celebrate Western Easter?', '2. Prognozează mai bine Paștele ortodox, deși majoritatea țărilor sărbătoresc Paștele occidental?'),
      T('3. Which countries help Romania most (leave-one-country-out)?', '3. Ce țări ajută cel mai mult România (eliminînd pe rînd cîte o țară)?')]),
    T('A possible team project: data, code, DM tests and a two-page report; model: B1, B3', 'Un posibil proiect de echipă: date, cod, teste DM și un raport de două pagini; model: B1, B3')), 'footnotesize')

D.frame(T('C2: audit an AI answer [Proposed]', 'C2: verificați un răspuns AI [Propus]'), items(
    T('A student asked an AI assistant whether machine learning can forecast markets and electricity load. The answer:', 'Un student a întrebat un asistent AI dacă învățarea automată poate prognoza piețele și consumul de energie. Răspunsul:'),
    T('\\aiprompt{(a) A random forest with 5-fold cross-validation explains @{c2.kf}\\% of the next month\'s S\\&P 500 return, so the market is predictable.}', '\\aiprompt{(a) Un random forest validat cu 5 grupuri explică @{c2.kf}\\% din randamentul S\\&P 500 din luna următoare, deci piața este previzibilă.}'),
    T('\\aiprompt{(b) A logistic model predicts the daily sign with @{c2.la}\\% accuracy, well above 50\\%: a profitable signal.}', '\\aiprompt{(b) Un model logistic prognozează semnul zilnic cu acuratețea de @{c2.la}\\%, mult peste 50\\%: un semnal profitabil.}'),
    T('\\aiprompt{(c) For the load, GB has MASE @{c2.gb}, so it is @{c2.gain}\\% more accurate than DHR.}', '\\aiprompt{(c) Pentru consum, GB are MASE @{c2.gb}, deci este cu @{c2.gain}\\% mai precis decît DHR.}'),
    T('\\aiprompt{(d) Standardise the whole series first, then split it into training and test.}', '\\aiprompt{(d) Standardizați întîi toată seria, apoi împărțiți-o în antrenare și test.}'),
    T('\\aiprompt{(e) An LSTM always beats ARIMA because it has long memory.}', '\\aiprompt{(e) Un LSTM bate întotdeauna ARIMA, pentru că are memorie lungă.}'),
    T('\\aiprompt{(f) Calendar features of the target day, such as holidays, are allowed because they are known in advance.}', '\\aiprompt{(f) Variabilele de calendar ale zilei-țintă, precum sărbătorile, sînt permise, deoarece se cunosc dinainte.}'),
    (T('Tasks', 'Cerințe'),
     [T('1. For each statement, say whether it is correct; if not, give the correct statement and, where possible, the correct number from the lecture notebook (sections 2, 6, 9).',
        '1. Pentru fiecare afirmație, precizați dacă este corectă; dacă nu este, formulați afirmația corectă și, acolo unde se poate, dați valoarea corectă din notebook-ul cursului (secțiunile 2, 6, 9).'),
      T('2. Report: a list of six verdicts with one line of justification each.', '2. Raportați: o listă de șase verdicte, fiecare cu un rînd de justificare.')])),
    'scriptsize')

D.frame(T('C2: solution [Proposed]', 'C2: rezolvare [Propus]'), items(
    T('(a) Wrong: overlapping 21-day targets leak across shuffled folds; walk-forward gives $R^2$ = @{c2.wf}\\%', '(a) Greșit: țintele suprapuse pe 21 de zile se scurg între grupurile amestecate; walk-forward dă $R^2$ = @{c2.wf}\\%'),
    T('(b) Wrong: the baseline ``always up\'\' has @{c2.ba}\\%; the model is below it', '(b) Greșit: reperul „întotdeauna în creștere” are @{c2.ba}\\%; modelul este sub el'),
    T('(c) Wrong: MASE compares with the in-sample seasonal naive method, not with DHR; DHR has MASE @{c2.dhr}, so GB is less accurate than DHR', '(c) Greșit: MASE compară cu metoda naivă sezonieră în eșantion, nu cu DHR; DHR are MASE @{c2.dhr}, deci GB este mai puțin precis decît DHR'),
    T('(d) Wrong: leakage; fit the scaler on each training window only', '(d) Greșit: scurgere de informație; scalarea se estimează doar pe fiecare fereastră de antrenare'),
    T('(e) Wrong: on our load data the LSTM has MASE @{c2.lstm}, worse than DHR, GB and ridge; M4: pure ML methods lost to simple combinations', '(e) Greșit: pe datele noastre de consum, LSTM are MASE @{c2.lstm}, mai slab decît DHR, GB și ridge; M4: metodele ML pure au pierdut în fața combinațiilor simple'),
    T('(f) Correct', '(f) Corect')) + qlsem(),
    'scriptsize', instructor_only=True)

# =============================================================================
# ÎNCHEIERE
# =============================================================================
D.section('Wrap-up', 'Încheiere')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('A forecast is a prediction from features known at the origin: lags, past rolling means, calendar', 'O prognoză este o predicție din variabile cunoscute la origine: decalaje, medii mobile trecute, calendar'),
    T('Recursive and direct strategies: compare them on the same origins', 'Strategiile recursivă și directă: le comparăm pe aceleași origini'),
    T('Leakage: any value from after the origin, including through scaling or shuffled folds', 'Scurgerea de informație: orice valoare de după origine, inclusiv prin scalare sau prin grupuri amestecate'),
    T('GB beats the seasonal naive forecast for the load; nothing beats ``always up\'\' for the sign of BET returns', 'GB depășește prognoza naivă sezonieră pentru consum; nimic nu depășește „întotdeauna în creștere” pentru semnul randamentelor BET'),
    T('An AI answer is a draft: check the validation scheme and the benchmark behind every claimed accuracy', 'Un răspuns AI este o ciornă: verificați schema de validare și reperul din spatele fiecărei acurateți declarate')))

D.frame(T('After the seminar', 'După seminar'), items(
    T('Lecture 9 develops each topic of today: features and strategies, walk-forward validation, ridge and lasso, trees and boosting, neural networks and the LSTM, prediction intervals, global models, the M4 and M5 competitions',
      'Cursul 9 dezvoltă fiecare temă de azi: variabile și strategii, validarea walk-forward, ridge și lasso, arbori și boosting, rețele neuronale și LSTM, intervale de prognoză, modele globale, competițiile M4 și M5'),
    T('Try the [Proposed] tasks in the notebook', 'Încercați cerințele [Propus] în notebook'),
    T('C1 can grow into a team project: a global model for European electricity load', 'C1 poate deveni un proiect de echipă: un model global pentru consumul de energie din Europa'),
    T('Reading: \\refHP, Ch.~10; \\refFPP, Sec.~12.4; \\refMMH', 'Lectură: \\refHP, cap.~10; \\refFPP, secț.~12.4; \\refMMH'),
    T('\\textbf{The seminar is for practice and is not graded; the solutions of [Proposed] tasks are discussed in class}',
      '\\textbf{Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar}')))

D.references(bib(['DM', 'FPP', 'HK', 'HP', 'KB', 'Lei', 'MMH', 'Kaufman']), per=16)

if __name__ == '__main__':
    finalize(D.write(V))
