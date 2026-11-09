"""Build the interactive pages of Chapter 9 (machine learning for time series): interactive/ch9.html (EN) and interactive/ch9_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its interactive version on
the right (the diagrams of the strategies, of the validation schemes, of the perceptron and of the LSTM get a live block too).
The result tables of the lecture (the load results with the DM tests, the lasso entry order, the inflation results, the
volatility results, accuracy and AUC of the sign forecasts) and its worked examples (the lag table, recursive and direct
forecasts, the leakage experiment, ridge and lasso, gradient boosting, MASE, the Diebold-Mariano statistic, the pinball loss,
split conformal prediction) get blocks of their own (EXTRA), placed by slide; on the web lecture each of them travels inside a
chart block and ch9.js moves it to its own frame. Logic in interactive/ch9.js, inputs in interactive/ch9_data.js
(Quantlets/Ch_09/export_interactive.py), slide images in interactive/slides/.
"""
import json
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 9

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_09'
TEX = {'ro': 'RO/Cursuri/capitol9_invatare_automata_serii_timp', 'en': 'EN/Courses/chapter9_machine_learning_time_series'}

# (anchor, chart file, section key); order of the slides
BLOCKS = [('features', 'tsa_ch9_features', 'sup'), ('strategies', 'tsa_ch9_strategies', 'sup'),
          ('cv-schemes', 'tsa_ch9_cv_schemes', 'val'), ('leakage', 'tsa_ch9_leakage', 'val'), ('bias-variance', 'tsa_ch9_bias_variance', 'val'),
          ('shrinkage', 'tsa_ch9_shrinkage', 'reg'),
          ('tree', 'tsa_ch9_tree', 'tree'), ('ensembles', 'tsa_ch9_ensembles', 'tree'), ('extrapolation', 'tsa_ch9_extrapolation', 'tree'),
          ('importance', 'tsa_ch9_importance', 'tree'),
          ('mlp', 'tsa_ch9_mlp', 'nn'), ('rnn', 'tsa_ch9_rnn', 'nn'),
          ('load-results', 'tsa_ch9_load_results', 'load'), ('load-forecasts', 'tsa_ch9_load_forecasts', 'load'),
          ('intervals', 'tsa_ch9_intervals', 'pi'),
          ('inflation', 'tsa_ch9_inflation', 'inf'),
          ('rv', 'tsa_ch9_rv', 'fin'), ('sign', 'tsa_ch9_sign', 'fin'),
          ('m4', 'tsa_ch9_m4', 'case'), ('gkx', 'tsa_ch9_gkx', 'case')]

# result tables and worked examples: (anchor, frame title EN, frame title RO, section key, host chart block)
EXTRA = [('wx-lag', 'Worked example: a lag table for the electricity load', 'Exemplu rezolvat: un tabel de laguri pentru consumul de energie electrică', 'sup', 'features'),
         ('wx-strat', 'Worked example: recursive and direct two-step forecasts', 'Exemplu rezolvat: prognoze recursive și directe pe doi pași', 'sup', 'strategies'),
         ('wx-leak', 'Interpreting the leakage experiment', 'Interpretarea experimentului de leakage', 'val', 'leakage'),
         ('wx-shrink', 'Ridge and lasso', 'Ridge și lasso', 'reg', 'shrinkage'),
         ('lasso-table', 'Interpreting the coefficient paths', 'Interpretarea traiectoriilor coeficienților', 'reg', 'shrinkage'),
         ('wx-boost', 'Gradient boosting', 'Gradient boosting', 'tree', 'ensembles'),
         ('wx-mase', 'The forecasting exercise', 'Exercițiul de prognoză', 'load', 'load-results'),
         ('load-table', 'Results: 26 origins, 14 horizons', 'Rezultatele: 26 de origini, 14 orizonturi', 'load', 'load-results'),
         ('wx-dm', 'Interpreting the load results', 'Interpretarea rezultatelor pentru consum', 'load', 'load-results'),
         ('wx-pinball', 'Quantile regression and the pinball loss', 'Regresia cuantilică și funcția de pierdere pinball', 'pi', 'intervals'),
         ('wx-conformal', 'Split conformal prediction', 'Predicția conformală prin împărțire', 'pi', 'intervals'),
         ('inf-table', 'Inflation results', 'Rezultatele pentru inflație', 'inf', 'inflation'),
         ('rv-table', 'Results: S&P 500 and DAX, 2013–2026', 'Rezultatele: S&P 500 și DAX, 2013–2026', 'fin', 'rv'),
         ('sign-table', 'Forecasting the sign of the next-day return: choice of the baseline', 'Prognoza semnului randamentului din ziua următoare: alegerea reperului', 'fin', 'sign')]

PFX = {'features': 'fe', 'strategies': 'st', 'cv-schemes': 'cv', 'leakage': 'lk', 'bias-variance': 'bv', 'shrinkage': 'sh', 'tree': 'tr', 'ensembles': 'en',
       'extrapolation': 'ex', 'importance': 'im', 'mlp': 'ml', 'rnn': 'rn', 'load-results': 'lr', 'load-forecasts': 'lf', 'intervals': 'iv', 'inflation': 'in',
       'rv': 'rv', 'sign': 'si', 'm4': 'm4', 'gkx': 'gk', 'wx-lag': 'wl', 'wx-strat': 'ws', 'wx-leak': 'wk', 'wx-shrink': 'wr', 'lasso-table': 'lt',
       'wx-boost': 'wb', 'wx-mase': 'wm', 'load-table': 'ld', 'wx-dm': 'wd', 'wx-pinball': 'wp', 'wx-conformal': 'wc', 'inf-table': 'it', 'rv-table': 'rt',
       'sign-table': 'sb'}

SECTIONS = {
 'ro': dict(sup='Prognoza ca învățare supervizată', val='Validare fără leakage', reg='Modele liniare regularizate: ridge și lasso',
            tree='Arbori, random forest și gradient boosting', nn='Rețele neuronale', load='Aplicație: consumul de energie electrică față de Capitolul 4',
            pi='Intervale de prognoză', inf='Modele locale și globale: inflația din România', fin='Aplicații în finanțe: volatilitatea și semnul randamentelor',
            case='Studii de caz de referință: M4, M5 și Gu, Kelly și Xiu'),
 'en': dict(sup='Forecasting as supervised learning', val='Validation without leakage', reg='Regularised linear models: ridge and lasso',
            tree='Trees, random forest and gradient boosting', nn='Neural networks', load='Application: electricity load against Chapter 4',
            pi='Prediction intervals', inf='Local and global models: Romanian inflation', fin='Applications in finance: volatility and the sign of returns',
            case='Landmark case studies: M4, M5 and Gu, Kelly and Xiu'),
}

L = {
 'ro': dict(file='ch9_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 9 · Învățare automată pentru serii de timp', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'casetele „Exemplu rezolvat” refac calculul cu cifrele capitolului.',
   src='Consumul mediu zilnic de energie electrică al României (ENTSO-E, extrasul din Capitolul 4, 1 ianuarie 2022 – 30 iunie 2026). Prețurile zilnice S&P 500, DAX și BET '
       '(data/market, pînă la 18 septembrie 2026). Indicele IAPC al României (Eurostat prc_hicp_minr), citit cu funcția capitolului pe 4 noiembrie 2026 și tăiat la august 2026, '
       'ultima lună din slide-uri; reproduce exact cifrele din curs. Fișierul oficial de evaluare M4 (ch9_m4_owa.csv) și cifrele publicate de Gu, Kelly și Xiu (2020). '
       'Prognozele walk-forward ale modelelor (ch9_load_ml.csv, ch9_inflation.csv, ch4_load_cv.csv) și estimările care cer reantrenarea modelelor (random forest, gradient boosting, '
       'MLP, LSTM, lasso) sînt cele ale codului capitolului, cu aceeași sămînță; în browser se recalculează tabelele de laguri, corelațiile, arborii de regresie, MASE, testele '
       'Diebold–Mariano, acoperirea intervalelor, RMSE, QLIKE, acuratețea și AUC. Variantele care nu apar în curs (alt orizont, altă sămînță a eșantionului, alt sfîrșit al '
       'eșantionului, alt nivel al intervalului, alt prag) sînt rulări suplimentare ale codului capitolului, cu aceleași date și aceeași sămînță; modelele AR(1) ale blocului despre '
       'strategii și aproximarea lui tanh prin neuroni ReLU sînt calcule suplimentare, făcute în browser.'),
 'en': dict(file='ch9.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 9 · Machine Learning for Time Series', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; '
        'the “Worked example” boxes redo the calculation with the numbers of the chapter.',
   src='Daily mean electricity load of Romania (ENTSO-E, the extract of Chapter 4, 1 January 2022 – 30 June 2026). Daily S&P 500, DAX and BET prices (data/market, up to '
       '18 September 2026). The Romanian HICP index (Eurostat prc_hicp_minr), read with the chapter’s loader on 4 November 2026 and cut at August 2026, the last month of the slides; '
       'it reproduces the numbers of the lecture exactly. The official M4 evaluation file (ch9_m4_owa.csv) and the published numbers of Gu, Kelly and Xiu (2020). The walk-forward '
       'forecasts of the models (ch9_load_ml.csv, ch9_inflation.csv, ch4_load_cv.csv) and the estimates that need the models to be retrained (random forest, gradient boosting, MLP, '
       'LSTM, lasso) are those of the chapter code, with the same seed; the browser recomputes the lag tables, the correlations, the regression trees, MASE, the Diebold–Mariano '
       'tests, the coverage of the intervals, RMSE, QLIKE, accuracy and AUC. Variants that do not appear in the lecture (another horizon, another noise level, another sample end, '
       'another interval level, another threshold) are extra runs of the chapter code with the same data and seed; the AR(1) models of the strategies block and the approximation '
       'of tanh by ReLU neurons are extra computations made in the browser.'),
}

CSS = ('<style>.c9x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c9x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c9x .c9f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c9x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c9x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c9x .wctl .ctl{max-width:360px;margin:.15rem 0 .2rem}'
       '.c9t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c9t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c9t th,.c9t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c9t thead tr:last-child th{border-bottom:1px solid #1A3A6E}'
       '.c9t tbody tr:nth-child(even){background:#F5F7FB}.c9t tbody tr.hi{background:#E3F4F7}.c9t .r{text-align:right}.c9cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}</style>')


def tbl(i):
    return f'<div class="tbl c9t" id="{i}"></div>'


def wx(i, title, formula, ctl=''):
    """Worked-example box; `ctl` holds the controls that only drive the box (they stay out of the chart's controls bar)."""
    return f'<div class="c9x"><h4>{title}</h4>' + (f'<div class="wctl">{ctl}</div>' if ctl else '') + f'<div class="c9f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def box(ctl, inner):
    """A box with controls that only drive the table inside it."""
    return f'<div class="c9x"><div class="wctl">{ctl}</div>{inner}</div>'


MON = {'ro': ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'],
       'en': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']}
ORIGINS = ['2025-06-30', '2025-07-14', '2025-07-28', '2025-08-11', '2025-08-25', '2025-09-08', '2025-09-22', '2025-10-06', '2025-10-20', '2025-11-03', '2025-11-17',
           '2025-12-01', '2025-12-15', '2025-12-29', '2026-01-12', '2026-01-26', '2026-02-09', '2026-02-23', '2026-03-09', '2026-03-23', '2026-04-06', '2026-04-20',
           '2026-05-04', '2026-05-18', '2026-06-01', '2026-06-15']
SHRINK_ENDS = ('2023-06-30', '2023-12-31', '2024-06-30', '2024-12-31', '2025-06-30', '2025-12-31', '2026-06-29')
ML_MODELS = ['Ridge', 'RF', 'GB direct', 'GB recursive', 'MLP', 'LSTM']


def widgets_all(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    lec = s(' (curs)', ' (lecture)')
    W = s('Exemplu rezolvat', 'Worked example')
    dmy = lambda iso: f'{int(iso[8:10])} {MON[lang][int(iso[5:7]) - 1]} {iso[:4]}'   # noqa: E731
    orig = lambda i, d: select(i, s('Originea prognozei', 'Forecast origin'), [(o, dmy(o) + (lec if o == d else '')) for o in ORIGINS], d)   # noqa: E731
    ends = lambda i: select(i, s('Eșantionul se termină pe', 'Sample ends on'), [(e, dmy(e) + (lec if e == '2025-06-30' else '')) for e in SHRINK_ENDS], '2025-06-30')   # noqa: E731
    mdl = {'Seasonal naive': s('naivă sezonieră', 'seasonal naive'), 'Combination': s('combinația', 'combination'), 'GB recursive': s('GB recursiv', 'GB recursive')}
    mname = lambda m: mdl.get(m, m)   # noqa: E731
    return {
     'features': (slider('fe-h', s('Orizontul țintei h (zile; curs: 1)', 'Horizon of the target h (days; lecture: 1)'), 1, 14, 1, 1) + play('fe'), CSS + charts('fe-chart')),
     'strategies': (orig('st-o', '2025-06-30') + play('st'), charts('st-chart')),
     'cv-schemes': (slider('cv-k', s('Numărul de grupuri și de pași K', 'Number of folds and steps K'), 2, 6, 1, 4)
                    + slider('cv-g', s('Spațiul h (observații; curs: 3)', 'Gap h (observations; lecture: 3)'), 0, 6, 1, 3) + play('cv'), charts('cv-chart')),
     'leakage': (select('lk-h', s('Ținta: suma următoarelor h randamente', 'Target: sum of the next h returns'), [(str(h), f'h = {h}' + (lec if h == 21 else '')) for h in (1, 5, 10, 21, 42, 63)], '21')
                 + play('lk'), charts('lk-chart')),
     'bias-variance': (slider('bv-s', s('Abaterea standard a zgomotului σ (curs: 0,3)', 'Noise standard deviation σ (lecture: 0.3)'), 0.1, 0.6, 0.05, 0.3)
                       + slider('bv-d', s('Adîncimea arborelui roșu', 'Depth of the red tree'), 1, 10, 1, 10) + play('bv'), charts('bv-chart')),
     'shrinkage': (ends('sh-e') + play('sh'), charts('sh-chart')),
     'tree': (select('tr-e', s('Antrenare pînă la', 'Training up to'), [(e, dmy(e) + (lec if e == '2025-06-30' else '')) for e in ('2024-06-30', '2024-12-31', '2025-06-30', '2025-12-31', '2026-06-29')], '2025-06-30')
              + slider('tr-d', s('Adîncimea maximă', 'Maximum depth'), 1, 4, 1, 2) + slider('tr-m', s('Cel puțin atîtea zile într-o frunză', 'At least this many days per leaf'), 10, 300, 10, 20) + play('tr'),
              charts('tr-chart')),
     'ensembles': (select('en-s', s('Antrenare pînă la (validare pînă la 30 iunie 2025)', 'Training up to (validation up to 30 June 2025)'),
                          [(e, dmy(e) + (lec if e == '2024-12-31' else '')) for e in ('2023-12-31', '2024-06-30', '2024-12-31')], '2024-12-31') + play('en'), charts('en-chart')),
     'extrapolation': (select('ex-s', s('Antrenare pînă la', 'Training up to'), [(e, f'{MON[lang][11]} {e[:4]}' + (lec if e == '2020-12' else '')) for e in ('2018-12', '2019-12', '2020-12', '2021-12', '2022-12')], '2020-12')
                       + play('ex'), charts('ex-chart')),
     'importance': (slider('im-h', s('Orizonturile h și 15 − h (curs: 1 și 14)', 'Horizons h and 15 − h (lecture: 1 and 14)'), 1, 7, 1, 1) + play('im'), charts('im-chart')),
     'mlp': (slider('ml-q', s('Neuroni ascunși q (curs: 5)', 'Hidden neurons q (lecture: 5)'), 1, 10, 1, 5) + play('ml'), charts('ml-chart')),
     'rnn': (orig('rn-o', '2025-06-30') + play('rn'), charts('rn-chart')),
     'load-results': (slider('lr-n', s('Primele n origini folosite (curs: 26)', 'First n origins used (lecture: 26)'), 6, 26, 2, 26) + play('lr'), charts('lr-chart')),
     'load-forecasts': (orig('lf-o', '2026-04-06') + play('lf'), charts('lf-chart')),
     'intervals': (slider('iv-l', s('Nivelul intervalului conformal', 'Level of the conformal interval'), 0.5, 0.95, 0.05, 0.9) + play('iv'), charts('iv-chart')),
     'inflation': (slider('in-y', s('Originile de test încep în ianuarie', 'Test origins start in January'), 2016, 2022, 1, 2016)
                   + select('in-h', s('Orizontul din stînga (luni)', 'Horizon on the left (months)'), [(h, h + (lec if h == '12' else '')) for h in ('1', '3', '6', '12')], '12') + play('in'),
                   charts('in-chart')),
     'rv': (select('rv-k', s('Indicele', 'Index'), [('sp500', 'S&P 500' + lec), ('dax', 'DAX')], 'sp500')
            + select('rv-m', s('Modelele ML afișate lîngă HAR', 'ML models shown next to HAR'), [('RF,GB', 'RF, GB' + lec), ('Lasso,MLP', 'Lasso, MLP'), ('RF,Lasso', 'RF, Lasso'), ('GB,MLP', 'GB, MLP')], 'RF,GB')
            + select('rv-a', s('Anul din stînga', 'Year on the left'), [(str(y), str(y) + (lec if y == 2020 else '')) for y in range(2013, 2027)], '2020')
            + select('rv-b', s('Anul din dreapta', 'Year on the right'), [(str(y), str(y) + (lec if y == 2025 else '')) for y in range(2013, 2027)], '2025') + play('rv'),
            charts('rv-chart')),
     'sign': (slider('si-t', s('Pragul de decizie: „creștere” dacă probabilitatea ≥ prag', 'Decision threshold: “up” if the probability ≥ threshold'), 0.4, 0.6, 0.01, 0.5) + play('si'),
              charts('si-chart')),
     'm4': (slider('m4-w', s('Ponderea sMAPE în OWA (curs: 0,5)', 'Weight of sMAPE in OWA (lecture: 0.5)'), 0, 1, 0.1, 0.5) + play('m4'), charts('m4-chart')),
     'gkx': (select('gk-m', s('Modelul marcat', 'Model marked'), [(m, m) for m in ('OLS-3+H', 'PLS', 'PCR', 'ENet+H', 'GLM+H', 'RF', 'GBRT+H', 'NN1', 'NN2', 'NN3', 'NN4', 'NN5')], 'NN4')
             + play('gk'), charts('gk-chart')),
     # ---------------------------------------------------------------- EXTRA
     'wx-lag': (slider('wl-e', s('Ultima zi a tabelului', 'Last day of the table'), -14, 14, 1, 0) + play('wl'),
                charts('wl-chart') + box(slider('wl-p', s('Laguri p (curs: 3)', 'Lags p (lecture: 3)'), 1, 5, 1, 3) + slider('wl-h', s('Orizontul h (curs: 1)', 'Horizon h (lecture: 1)'), 1, 3, 1, 1),
                                         tbl('wl-tb') + f'<h4>{W}</h4><ol id="wl-ws"></ol>')),
     'wx-strat': (slider('ws-y', s('Ultima valoare y<sub>T</sub> (curs: 5)', 'Last value y<sub>T</sub> (lecture: 5)'), 3, 8, 0.1, 5)
                  + slider('ws-b', s('Panta modelului pe un pas (curs: 0,8)', 'Slope of the one-step model (lecture: 0.8)'), 0.5, 0.95, 0.05, 0.8) + play('ws'),
                  charts('ws-chart') + wx('ws-ws', W, r'\hat y_{T+1}=a+b\,y_T,\quad \hat y_{T+2}=a+b\,\hat y_{T+1}=a(1+b)+b^2y_T;\qquad \text{direct: }\hat y_{T+2}=2.3+0.6\,y_T'.replace('2.3', s('2{,}3', '2.3')).replace('0.6', s('0{,}6', '0.6')))),
     'wx-leak': (select('wk-s', s('Seria', 'Series'), [('sp500', 'S&P 500' + lec), ('sim', s('mersul aleator simulat', 'simulated random walk'))], 'sp500')
                 + select('wk-h', s('Ținta: suma următoarelor h randamente', 'Target: sum of the next h returns'), [(str(h), f'h = {h}' + (lec if h == 21 else '')) for h in (1, 5, 10, 21, 42, 63)], '21')
                 + play('wk'),
                 charts('wk-chart') + wx('wk-ws', W, r'\rho(Y_t,Y_{t+k})=\max\!\Big(0,1-\frac{k}{h}\Big),\qquad R^2=1-\frac{\sum_{\text{folds}}\sum_t(y_t-\hat y_t)^2}{\sum_{\text{folds}}\sum_t(y_t-\bar y_{\text{train}})^2}',
                                         select('wk-c', s('Schema din calcul', 'Scheme of the calculation'), [('kfold', s('5-fold aleator', 'random 5-fold') + lec), ('wf', 'walk-forward'), ('gap', s('walk-forward cu spațiu', 'walk-forward with a gap'))], 'kfold'))),
     'wx-shrink': (slider('wr-l', s('λ pentru lasso (curs: 40)', 'λ of the lasso (lecture: 40)'), 0, 200, 5, 40) + slider('wr-r', s('λ pentru ridge (curs: 50)', 'λ of ridge (lecture: 50)'), 0, 200, 5, 50)
                   + slider('wr-s', s('S<sub>xy</sub> (curs: 90; S<sub>xx</sub> = 100)', 'S<sub>xy</sub> (lecture: 90; S<sub>xx</sub> = 100)'), -120, 120, 5, 90) + play('wr'),
                   charts('wr-chart') + wx('wr-ws1', W + ': ridge', r'\hat\beta_{\text{ridge}}=\frac{S_{xy}}{S_{xx}+\lambda}=\frac{S_{xx}}{S_{xx}+\lambda}\,\hat\beta_{\text{OLS}}')
                   + wx('wr-ws2', W + ': lasso', r'\hat\beta_{\text{lasso}}=\operatorname{sign}(S_{xy})\,\frac{\max(|S_{xy}|-\lambda/2,\,0)}{S_{xx}}')),
     'lasso-table': (ends('lt-e') + slider('lt-k', s('Penalizarea: pași pe grilă față de cea aleasă prin validare (negativ: λ mai mare)', 'Penalty: grid steps from the one chosen by validation (negative: larger λ)'), -15, 15, 1, 0) + play('lt'),
                     charts('lt-chart') + box(slider('lt-n', s('Rînduri afișate', 'Rows shown'), 5, 30, 1, 10), tbl('lt-tb'))),
     'wx-boost': (slider('wb-n', s('Rata de învățare ν (curs: 0,1)', 'Learning rate ν (lecture: 0.1)'), 0.05, 1, 0.05, 0.1) + play('wb'),
                  charts('wb-chart') + wx('wb-ws', W, r'F_0=\bar y,\qquad r_i=y_i-F_{m-1}(x_i),\qquad F_m(x)=F_{m-1}(x)+\nu\,g_m(x)',
                                          slider('wb-m', s('Arborii m din calcul', 'Trees m in the calculation'), 1, 30, 1, 1))),
     'wx-mase': (slider('wm-m', s('Perioada sezonieră m a scalei (curs: 7)', 'Seasonal period m of the scale (lecture: 7)'), 1, 14, 1, 7) + play('wm'),
                 charts('wm-chart') + wx('wm-ws', W, r'\text{MASE}=\frac{\text{MAE}}{\frac{1}{n-m}\sum_{t=m+1}^{n}|y_t-y_{t-m}|}',
                                         select('wm-x', s('Modelul din calcul', 'Model of the calculation'), [(m, mname(m)) for m in ['Seasonal naive', 'ETS', 'SARIMA', 'DHR', 'Combination'] + ML_MODELS], 'DHR'))),
     'load-table': (slider('ld-k', s('Primele origini eliminate (curs: 0)', 'First origins dropped (lecture: 0)'), 0, 20, 2, 0) + play('ld'),
                    charts('ld-chart') + box(select('ld-r', s('Referința testului DM', 'Reference of the DM test'), [('DHR', 'DHR' + lec), ('Seasonal naive', mname('Seasonal naive'))], 'DHR')
                                             + select('ld-h', s('Orizonturile', 'Horizons'), [('1-14', '1–14' + lec), ('1-7', '1–7'), ('8-14', '8–14')], '1-14'), tbl('ld-tb'))),
     'wx-dm': (select('wd-a', s('Modelul comparat', 'Model compared'), [(m, mname(m) + (lec if m == 'Ridge' else '')) for m in ML_MODELS + ['Combination', 'SARIMA', 'ETS']], 'Ridge')
               + slider('wd-L', s('Autocovarianțe incluse, h − 1 (curs: 0)', 'Autocovariances included, h − 1 (lecture: 0)'), 0, 5, 1, 0) + play('wd'),
               charts('wd-chart') + wx('wd-ws', W, r'd_t=\overline{|e|}_{t}^{\,A}-\overline{|e|}_{t}^{\,B},\quad \hat V=\frac{1}{n}\Big(\hat\gamma_0+2\sum_{k=1}^{h-1}\hat\gamma_k\Big),\quad DM=\frac{\bar d}{\sqrt{\hat V}},\quad DM^{*}=c_h\,DM\sim t_{n-1}',
                                       select('wd-r', s('Referința B', 'Reference B'), [('DHR', 'DHR' + lec), ('Seasonal naive', mname('Seasonal naive'))], 'DHR'))),
     'wx-pinball': (slider('wp-t', 'τ' + s(' (curs: 0,95)', ' (lecture: 0.95)'), 0.05, 0.95, 0.05, 0.95) + slider('wp-q', s('Prognoza q (GW; curs: 7,0)', 'Forecast q (GW; lecture: 7.0)'), 5, 8, 0.1, 7) + play('wp'),
                    charts('wp-chart') + wx('wp-ws', W, r'L_\tau(y,q)=\begin{cases}\tau\,(y-q), & y\ge q\\(1-\tau)\,(q-y), & y<q\end{cases}')),
     'wx-conformal': (slider('wc-a', s('Nivelul 1 − α (curs: 90%)', 'Level 1 − α (lecture: 90%)'), 0.5, 0.95, 0.05, 0.9)
                      + select('wc-h', s('Orizontul setului de calibrare afișat', 'Horizon of the calibration set shown'), [(str(h), str(h)) for h in range(1, 15)], '1') + play('wc'),
                      charts('wc-chart') + wx('wc-ws', W, r'k=\lceil (n+1)(1-\alpha)\rceil,\qquad \hat q=|e|_{(k)},\qquad \hat y\pm\hat q')),
     'inf-table': (select('it-p', s('Originile de test', 'Test origins'), [('all', s('ianuarie 2016 – iulie 2026', 'January 2016 – July 2026') + lec), ('p1620', '2016–2020'), ('p2123', '2021–2023'), ('p2426', '2024–2026')], 'all')
                   + play('it'),
                   charts('it-chart') + box(select('it-r', s('Referința testului DM', 'Reference of the DM test'), [('RW', 'RW' + lec), ('AR', 'AR'), ('GB local', 'GB local' + s(' (pentru GB global, curs)', ' (for GB global, lecture)'))], 'RW'),
                                            tbl('it-tb') + tbl('it-dm'))),
     'rv-table': (slider('rt-y', s('Perioada de test începe în', 'Test period starts in'), 2013, 2024, 1, 2013) + play('rt'), charts('rt-chart') + tbl('rt-tb')),
     'sign-table': (slider('sb-y', s('Perioada de test începe în', 'Test period starts in'), 2008, 2020, 1, 2008) + play('sb'), charts('sb-chart') + tbl('sb-tb')),
    }


def _frame_page(lang, title):
    from interactive_common import ROOT, frames_of, latex_text
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    assert len(frames) == len(pages)
    k = [latex_text(t) for t, _ in frames].index(title)
    return int(pages[k][0]), latex_text(frames[k][0])


def _extra_pages(lang):
    """Page and title of the frames of EXTRA, from the .tex and the beamer .nav file of each language."""
    return {a: _frame_page(lang, ten if lang == 'en' else tro) for a, ten, tro, _, _ in EXTRA}


def _extra_live(anchor, lang, W):
    t = L[lang]
    ctl, ch = W[anchor]
    q, a = QA[lang][anchor]
    live = f'<div class="live"><div class="ctls">{ctl}</div>{ch}<ul class="out" id="{PFX[anchor]}-out"></ul></div>'
    tail = f'<div class="ask"><span class="ask-tag">{t["ask"]}</span> {q}</div>\n  <details><summary>{t["answer"]}</summary><p>{a}</p></details>'
    return live, tail


def widgets(lang):
    """Chart blocks for build_web_lecture.py: each host carries the web version of its EXTRA blocks (moved by ch9.js)."""
    W = widgets_all(lang)
    xp = _extra_pages(lang)
    res = {}
    for a, _, _ in BLOCKS:
        ctl, ch = W[a]
        held = ''
        for x, _, _, _, host in EXTRA:
            if host == a:
                live, tail = _extra_live(x, lang, W)
                held += f'<section class="blk wlive xblk" id="{x}" data-frame="{xp[x][0]}">{live}{tail}</section>'
        res[a] = (ctl, ch + (f'<div class="xhold">{held}</div>' if held else ''))
    return res


MATHJAX = ('<script>window.MathJax = { tex: { inlineMath: [[\'\\\\(\', \'\\\\)\']], displayMath: [[\'\\\\[\', \'\\\\]\']] }, '
           'startup: { pageReady: () => MathJax.startup.defaultPageReady().then(() => document.body.setAttribute(\'data-mj\', \'done\')) } };</script>\n'
           '<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>\n')


def _post(lang):
    """Insert the EXTRA blocks in slide order (with their section headings) and MathJax; render their slide images."""
    import pymupdf
    import interactive_common as ic
    sp = sys.modules['__main__']
    t, W = L[lang], widgets_all(lang)
    path = os.path.join(ic.OUT, t['file'])
    html = open(path, encoding='utf-8').read()
    viewer = '../pdfjs/web/viewer.html?file=../../' + TEX[lang] + '.pdf'
    chart_pages = ic.slide_pages(sp, lang)
    blocks = {m.group(1): m.group(0) for m in re.finditer(r'\n<section class="blk" id="([^"]+)">.*?</section>', html, flags=re.S)}
    items = [(chart_pages[c][0], 0, a, k, blocks[a]) for a, c, k in BLOCKS]
    xp = _extra_pages(lang)
    doc = pymupdf.open(os.path.join(ic.ROOT, TEX[lang] + '.pdf'))
    for j, (anchor, _, _, skey, _) in enumerate(EXTRA):
        pg, title = xp[anchor]
        img = f'ch9x_{anchor}_{lang}.jpg'
        doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', img), jpg_quality=82)
        live, tail = _extra_live(anchor, lang, W)
        items.append((pg, j + 1, anchor, skey, f'''
<section class="blk" id="{anchor}">
  <h3>{title}</h3>
  <div class="pair">
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/{img}" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    {live}
  </div>
  {tail}
</section>'''))
    items.sort(key=lambda x: (x[0], x[1]))
    body, nav, sec = [], [], None
    for _, _, anchor, skey, h in items:
        if skey != sec:
            sec = skey
            body.append(f'<h2 class="sec" id="s-{skey}">{SECTIONS[lang][skey]}</h2>')
            nav.append(f'<a href="#s-{skey}">{SECTIONS[lang][skey]}</a>')
        body.append(h)
    i0, i1 = html.index('<p class="toc">'), html.index('<p class="src foot">')
    html = html[:i0] + f'<p class="toc">{" · ".join(nav)}</p>\n' + ''.join(body) + '\n' + html[i1:]
    if 'MathJax-script' not in html:
        html = html.replace('</head>', MATHJAX + '</head>', 1)
    open(path, 'w', encoding='utf-8').write(html)
    print('extra blocks', lang, len(EXTRA))



# ================================================================= readouts
TEXT_JS = {
 'ro': r"""(function () {
const MON = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
const de = n => { const r = Math.abs(n) % 100; return (r === 0 && n !== 0) || r >= 20 ? ' de' : ''; };
const zile = n => `${n}${de(n)} ${n === 1 ? 'zi' : 'zile'}`;
window.CH9_TEXT = {
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 sep: '; ', lt: '&lt; ',
 wd: ['luni', 'marți', 'miercuri', 'joi', 'vineri', 'sîmbătă', 'duminică'],
 model: { 'Seasonal naive': 'Naivă sezonieră', Combination: 'Combinația', 'GB recursive': 'GB recursiv', Mean: 'Media', RW: 'RW (mers aleator)' },
 per: { all: 'ianuarie 2016 – iulie 2026', p1620: '2016–2020', p2123: '2021–2023', p2426: '2024–2026' },
 scheme: { kfold: '5-fold aleator', wf: 'walk-forward', gap: 'walk-forward cu spațiu' },
 series: { sim: 'mersul aleator simulat', sp500: 'S&P 500' },
 L: { origin: 'Originea t', target: 'ținta', rank: 'Ordinea', feature: 'Variabila', entry: 'λ la intrare', coef: 'Coeficient la λ ales', model: 'Modelul',
   dmp: r => `DM p, față de ${r}`, rmseRel: 'RMSE (pp), relativ la RW', acc: 'Acuratețea', vsBase: 'Față de reper (pp)', upShare: 'Zile prognozate „creștere”' },
 fe1: (m, mn, mnd, mx, mxd, n) => `Consumul mediu zilnic: media ${m} GW pe ${n}${de(n)} zile; minimul ${mn} GW pe ${mnd}, maximul ${mx} GW pe ${mxd}.`,
 fe2: (h, k, n, c0, c6, cm) => `Ținta y<sub>t+${h}</sub>, ${n}${de(n)} rînduri: corelația cu y<sub>t</sub> este ${c0}, cu aceeași zi a săptămînii de acum ${k === 1 ? 'o săptămînă' : k + ' săptămîni'} ${c6}, cu media pe 7 zile ${cm}.`,
 wlCap: (a, b) => `Consumul mediu zilnic (GW), ${a} – ${b}; un rînd pentru fiecare origine cu toate lagurile și ținta cunoscute.`,
 wls1: (n, p, h, r) => `Din n = ${n} zile, cu p = ${p} laguri și orizontul h = ${h}: primele p − 1 = ${p - 1} zile nu au toate lagurile, ultimele h = ${h} nu au țintă; rămîn n − p − h + 1 = <b>${r}</b> rînduri.`,
 wls2: (n, h, r) => `Pe toată seria (n = ${n}), media pe 28 de zile cere 27 de zile anterioare: tabelul pentru h = ${h} are ${n} − 28 − ${h} + 1 = <b>${r}</b> rînduri.`,
 wls3: w => `Ziua primei ținte: ${w}. Modelul are nevoie de ziua săptămînii a țintei, nu a originii.`,
 wl1: (r, p, h, d, y, t) => `${r}${de(r)} rînduri cu p = ${p} și h = ${h}; primul rînd: originea ${d}, y<sub>t</sub> = ${y} GW, ținta ${t} GW.`,
 st1: (o, a, b, b2, d2, b7, d7) => `AR(1) estimat pe datele pînă la ${o}: ŷ<sub>t+1</sub> = ${a} + ${b}·y<sub>t</sub>. Coeficientul lui y<sub>T</sub> la h = 2: ${b2} implicit (b²), ${d2} estimat direct; la h = 7: ${b7} față de ${d7}.`,
 st2: (r, d) => `MAE pe cele 14 zile: ${r} MW recursiv, ${d} MW direct (rulare suplimentară: un AR(1) fără calendar; slide-ul este o diagramă).`,
 wss1: (y, b, f1, f2) => `Recursiv: ŷ<sub>T+1</sub> = 1,2 + ${b}·${y} = ${f1}; ŷ<sub>T+2</sub> = 1,2 + ${b}·${f1} = <b>${f2}</b>.`,
 wss2: (b, a2, b2) => `Regula implicită pe doi pași: a(1 + b) + b²y<sub>T</sub> = 1,2·(1 + ${b}) + ${b}²·y<sub>T</sub> = ${a2} + ${b2}·y<sub>T</sub>.`,
 wss3: (y, fd, d) => `Direct: ŷ<sub>T+2</sub> = 2,3 + 0,6·${y} = <b>${fd}</b>; diferența față de prognoza recursivă este ${d}.`,
 ws1: (y, r, d) => `y<sub>T</sub> = ${y}: prognoza pe doi pași este ${r} recursiv și ${d} direct.`,
 cv1: (k, s, h) => `K = ${k}: blocuri de test de ${s} observații în walk-forward; spațiul elimină ultimele ${h} rînduri de antrenare dinaintea fiecărui bloc de test.`,
 cv2: (h, a, b, c) => h > 1 ? `Cu ținte care se suprapun pe ${h} observații, rîndurile de test care au un rînd de antrenare la mai puțin de ${h} pași: ${a} în K-fold aleator, ${b} în walk-forward fără spațiu, ${c} cu spațiu.` : `Cu h = ${h}, țintele nu se suprapun: niciun rînd de antrenare nu împarte ținta cu un rînd de test.`,
 lk1: (h, a, b, c, d, e, f) => `h = ${h}. Mers aleator simulat: R² = ${a} cu 5-fold aleator, ${b} walk-forward, ${c} cu spațiu. S&P 500: ${d}, ${e} și ${f}.`,
 lk2: (h, k, r, n) => `Țintele vecine au în comun ${k} din ${h}${de(h)} randamente (corelația teoretică ${r}); ${n}${de(n)} zile pentru S&P 500.`,
 wks1: (h, k, r, a, n) => `Țintele de pe zile vecine au în comun ${k} din ${h}${de(h)} randamente: corelația lor este ${k}/${h} = ${r} în teorie și ${a} în eșantion (${n}${de(n)} rînduri).`,
 wks2: (s, f) => `${s}, pe grupuri (rînduri de test: Σ(y − ŷ)² / Σ(y − ȳ)²): ${f}.`,
 wks3: (a, b, r) => `R² comun = 1 − ${a}/${b} = <b>${r}</b>.`,
 wk1: (s, h, a, b, c) => `${s}, h = ${h}: R² = ${a} (5-fold aleator), ${b} (walk-forward), ${c} (cu spațiu).`,
 bv1: (s, d, t, n) => `σ = ${s}: eroarea de test așteptată este minimă la adîncimea ${d} (${t}), peste varianța zgomotului ${n}.`,
 bv2: (b1, v1, b10, v10, t10) => `Adîncimea 1: deplasare² ${b1}, varianță ${v1}; adîncimea 10: deplasare² ${b10}, varianță ${v10}, eroarea de antrenare ${t10}.`,
 wrs1: (s, x, o) => `OLS: β̂ = S<sub>xy</sub>/S<sub>xx</sub> = ${s}/${x} = <b>${o}</b>.`,
 wrs2: (l, s, d, b, f) => `Ridge, λ = ${l}: β̂ = ${s}/${d} = <b>${b}</b>; factorul de contracție S<sub>xx</sub>/(S<sub>xx</sub> + λ) = ${f}, niciodată 0.`,
 wrs3: (l, l2, a, r, b) => `Lasso, λ = ${l}: pragul λ/2 = ${l2}; |S<sub>xy</sub>| − λ/2 = ${a} − ${l2} → ${r}; β̂ = ${r}/100 = <b>${b}</b>.`,
 wrs4: z => `Pentru λ ≥ 2|S<sub>xy</sub>| = ${z}, coeficientul lasso este exact 0.`,
 wr1: (o, lr, br, ll, bl, z) => `OLS ${o}; ridge cu λ = ${lr}: ${br}; lasso cu λ = ${ll}: ${bl} (exact 0 de la λ = ${z}).`,
 sh1: (e, n, f, k, c0, cm) => `Datele pînă la ${e} (${n}${de(n)} rînduri): primele variabile care intră sînt ${f}; la penalizarea aleasă, lasso păstrează ${k} din 30, cu lag0 ${c0} și luni ${cm} (GW la o abatere standard).`,
 sh2: a => `Penalizarea aleasă prin validare walk-forward (TimeSeriesSplit, 5 grupuri): λ = ${a}.`,
 ltCap: (e, n) => `Datele pînă la ${e}, ${n}${de(n)} rînduri; ordinea de intrare pe traiectoria lasso (λ descrescător); rîndurile evidențiate: primele trei.`,
 lt1: (f, l, k) => `Ordinea de intrare: ${f}. La λ = ${l} rămîn ${k}${de(k)} variabile nenule din 30.`,
 tr0: n => `Niciun split nu respectă dimensiunea minimă a frunzei (${n}${de(n)} zile): prognoza este media.`,
 tr1: (f, t, v, n, N) => `Prima întrebare: ${f} ≤ ${t} GW? Frunzele: ${v} GW, cu ${n} zile (${N}${de(N)} zile de antrenare).`,
 tr2: (r, k) => `R² la antrenare: ${r}, cu ${k} valori posibile ale prognozei.`,
 wbs1: (f0, r) => `F<sub>0</sub> = media țintelor 5, 6, 8 = ${f0}; reziduurile: ${r}.`,
 wbs2: (l, r, nu, f0, r2, f1, f1a) => `Arborele cu o împărțire care izolează al treilea punct prognozează ${l} pentru primele două și ${r} pentru al treilea; cu ν = ${nu}: F<sub>1</sub>(x<sub>3</sub>) = ${f0} + ${nu}·${r2} = <b>${f1}</b>, F<sub>1</sub>(x<sub>1</sub>) = ${f1a}.`,
 wbs3: (m, v, e) => `După m = ${m}${de(m)} ${m === 1 ? 'arbore' : 'arbori'}: F<sub>m</sub> = ${v}; MSE la antrenare ${e}.`,
 wb1: (nu, v) => `ν = ${nu}: prognoza pentru al treilea punct (ținta 8) este ${v} după 5, 20 și 60 de arbori.`,
 en1: (s, a, b, t, td, r, rd, n) => `Antrenare pînă la ${s} (${a}${de(a)} zile), validare ${b}${de(b)} zile: un arbore ${t} la adîncimea ${td}, random forest ${r} la adîncimea ${rd}; naiv sezonier ${n} (MSE, GW²).`,
 en2: (a, ia, b, ib) => `Boosting: minimul ${a} după ${ia}${de(ia)} ${ia === 1 ? 'arbore' : 'arbori'} cu ν = 0,3; ${b} după ${ib}${de(ib)} ${ib === 1 ? 'arbore' : 'arbori'} cu ν = 0,05.`,
 ex1: (s, mx, l, ld, f, dl) => `Antrenare pînă în ${s}: cel mai mare indice văzut ${mx}; în ${ld} indicele ajunge la ${l} (Eurostat, descărcat pe ${dl}); random forest pe niveluri prognozează ${f}.`,
 ex2: (a, b, n) => `RMSE pe ${n}${de(n)} luni: ${a} puncte de indice pe niveluri, ${b} pe variațiile lunare cumulate.`,
 im1: (h, a, va, b, vb) => `Orizontul ${h} ${h === 1 ? 'zi' : 'zile'}: ${a} (+${va} MW de MAE cînd este amestecată) și ${b} (+${vb} MW).`,
 ml1: (q, a, b, c, n) => `q = ${q}: ${a} ponderi de intrare, ${b} constante, ${c} ponderi de ieșire și o constantă: ${n}${de(n)} parametri.`,
 ml2: (q, e) => `Cu ${q} ${q === 1 ? 'neuron' : 'neuroni'} ReLU, cea mai bună funcție liniară pe porțiuni din figură aproximează tanh pe [−4; 4] cu eroarea maximă ${e} (calcul suplimentar).`,
 rn1: (o, a, b, n) => `LSTM-ul reantrenat la ${o}: pierderea scade de la ${a} la ${b} în ${n}${de(n)} epoci (aceeași sămînță; prognozele sînt cele din ch9_load_ml.csv).`,
 rn2: (a, b) => `MAE pe cele 14 zile: LSTM ${a} MW, DHR ${b} MW.`,
 ldCap: (n, o, h) => `${n}${de(n)} origini, de la ${o}; orizonturile ${h.replace('-', '–')}; scala MASE 309 MW; DM pe erorile absolute medii pe origine, cu corecția HLN.`,
 ld1: (n, d, r, p, ref) => `${n}${de(n)} origini: MASE ${d} pentru DHR și ${r} pentru ridge; DM p pentru ridge față de ${ref}: ${p}.`,
 lr1: (n, o, a, ma, b, mb, l) => `Primele ${n}${de(n)} origini (pînă la ${o}): cele mai precise sînt ${a} (MASE ${ma}) și ${b} (${mb}); LSTM ${l}.`,
 lr2: (a, b, c, d) => `MAE la 1 și la 14 zile: DHR ${a} și ${b} MW, GB direct ${c} și ${d} MW.`,
 wms1: (a, b, n, m, k, s) => `Primul eșantion de antrenare, ${a} – ${b}: n = ${n} zile, n − m = ${k} diferențe |y<sub>t</sub> − y<sub>t−${m}</sub>|; media lor, scala, este <b>${s} MW</b>.`,
 wms2: (md, a, s, q) => `${md}: MAE pe cele 364 de prognoze = ${a} MW; MASE = ${a}/${s} = <b>${q}</b>.`,
 wms3: (m, s, q) => m === 7 ? `Sub 1: modelul este mai precis în afara eșantionului decît metoda naivă sezonieră în eșantion.` : `Cu scala săptămînală din curs (m = 7, ${s} MW), MASE ar fi ${q}: scala schimbă nivelul, nu ordinea modelelor.`,
 wm1: (m, s, md, q) => `m = ${m}: scala ${s} MW; MASE pentru ${md} ${q}.`,
 wds1: (a, b, n, d) => `d<sub>t</sub> = MAE pe origine al lui ${a} minus cel al lui ${b}, pentru n = ${n}${de(n)} origini; d̄ = <b>${d} MW</b>.`,
 wds2: (L, g, v) => `Cu ${L} ${L === 1 ? 'autocovarianță' : 'autocovarianțe'} (h = ${L + 1}): ${g} (MW²); V̂ = ${v} MW².`,
 wds3: (d, s, dm, f, h) => `DM = d̄/√V̂ = ${d}/${s} = ${dm}; corecția HLN c<sub>h</sub> = √((n + 1 − 2h + h(h − 1)/n)/n) = ${f}: DM* = c<sub>h</sub>·DM = <b>${h}</b>.`,
 wds4: (df, p) => `p = 2·P(t<sub>${df}</sub> > |DM*|) = <b>${p}</b>.`,
 wd1: (a, b, L, h, p) => `${a} față de ${b}, ${L} ${L === 1 ? 'autocovarianță' : 'autocovarianțe'}: DM* = ${h}, p = ${p}.`,
 lf1: (o, m) => `Originea ${o}, MAE pe cele 14 zile: ${m}.`,
 lf2: (n, b, v) => `${n} ${n === 1 ? 'zi de sărbătoare' : 'zile de sărbătoare'} în fereastră; cel mai mic MAE: ${b}, ${v} MW.`,
 wps1: (t, q, a, b) => `τ = ${t}, q = ${q} GW: y = 7,4 costă ${a}; y = 6,5 costă ${b}.`,
 wps2: (t, q, e) => `Media pierderii pe cele 364 de valori observate este minimă la q = ${q} GW, cuantila empirică de ordin ${t} fiind ${e} GW.`,
 wps3: () => `Cu τ = 0,5 pierderea este jumătate din eroarea absolută, iar minimul este mediana.`,
 wp1: (t, q, a, b) => `τ = ${t}, q = ${q}: pierderea este ${a} pentru y = 7,4 și ${b} pentru y = 6,5.`,
 wcs1: (n, l, a, n1, k) => `n = ${n}, nivel ${l} (α = ${a}): k = ⌈${n1}·(1 − ${a})⌉ = <b>${k}</b>.`,
 wcs2: (k, q) => `q̂ = a ${k}-a cea mai mică eroare absolută = <b>${q} MW</b>.`,
 wcs3: (o, h, y, a, b, v, ok) => `Originea ${o}, h = ${h}: ŷ = ${y} GW, intervalul [${a}; ${b}] GW; valoarea observată ${v} GW este ${ok ? 'în interval' : 'în afara intervalului'}.`,
 wc1: (l, c, w) => `Nivel ${l}: acoperire ${c} pe cele 364 de prognoze, lățimea medie ${w} MW.`,
 iv1: (c, w, a, b) => `GB cuantilic (90%): acoperire ${c}, lățimea medie ${w} MW; ${a} dintre valori sub interval și ${b} deasupra.`,
 iv2: (l, c, w, a, b) => `Predicția conformală la ${l}: acoperire ${c}, lățimea medie ${w} MW; ${a} sub și ${b} deasupra.`,
 in1: (mx, mxd, mn, mnd, l, ld, dl) => `Inflația anuală din România: maximul ${mx}% în ${mxd}, minimul ${mn}% în ${mnd}, ${l}% în ${ld} (Eurostat, descărcat pe ${dl}).`,
 in2: (y, a, g, l) => `Origini din ${y}: RMSE relativ al AR ${a}; la 12 luni GB global ${g}, GB local ${l}.`,
 itCap: (p, n) => `Originile de test: ${p}; numărul de origini pentru h = 1, 3, 6, 12: ${n}. În paranteză: RMSE relativ la RW.`,
 itNote: 'DM pe erorile pătratice, cu h − 1 autocovarianțe și corecția HLN; p &lt; 0,05 îngroșat.',
 it1: (p, a, g) => `${p}: RMSE relativ al AR ${a} (h = 1, 3, 6, 12); GB global ${g}.`,
 rtCap: (y, a, b) => `Perioada de test ${y}–2026: ${a}${de(a)} zile S&P 500 și ${b}${de(b)} zile DAX; DM t &lt; −1,96: QLIKE semnificativ mai mic decît HAR.`,
 rt1: (y, a, b, l, m, tl, tm) => `Din ${y}: S&P 500 (${a}${de(a)} zile), lasso ${l}% și MLP ${m}% R² față de HAR, DM t ${tl} și ${tm}; DAX: ${b}${de(b)} zile.`,
 rv1: (k, y, q) => `${k}, ${y}: QLIKE ${q}.`,
 baseRow: u => `reper; ${u} zile de creștere`,
 sbCap: (y, a, b) => `Perioada de test ${y}–2026: ${a}${de(a)} zile S&P 500 și ${b}${de(b)} zile BET; reperul: clasa majoritară a eșantionului de antrenare.`,
 sb1: (y, a, b, mn, mx) => `Din ${y}: zile de creștere ${a} (S&P 500) și ${b} (BET); AUC între ${mn} și ${mx}.`,
 si1: (t, a, ma, b, mb) => `Prag ${t}. S&P 500: reperul ${a}; ${ma}. BET: reperul ${b}; ${mb}.`,
 si2: (u, p) => `GB pe S&P 500 prognozează „creștere” în ${u} din zile; p față de reper ${p}.`,
 m41: (w, m, ty, o, c, r) => `Ponderea sMAPE ${w}: primul este metoda ${m} (${ty}), OWA ${o}; Comb ${c}, locul ${r}.`,
 m42: (n, r, o, a, b, k) => `Cele ${n} metode ML pure: cea mai bună pe locul ${r} (OWA ${o}); ${a} ${a === 1 ? 'depășește' : 'depășesc'} Comb, ${b} ${b === 1 ? 'depășește' : 'depășesc'} Naive2; ${k === 1 ? 'o metodă are' : k + ' metode au'} OWA peste 1,7.`,
 gk1: (m, r, s, q) => `${m}: R² lunar ${r}%, raportul Sharpe ${s} (de ${q} ori cel al OLS-3+H).`,
 gk2: (m, t) => `Pe cei 30 de ani de test, un raport Sharpe anual S corespunde unei statistici t ≈ S·√30 pentru media randamentului: ${t} pentru ${m} (calcul suplimentar).`,
};
})();
""",
 'en': r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
window.CH9_TEXT = {
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 sep: ', ', lt: '&lt; ',
 wd: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
 model: { RW: 'RW (random walk)' },
 per: { all: 'January 2016 – July 2026', p1620: '2016–2020', p2123: '2021–2023', p2426: '2024–2026' },
 scheme: { kfold: 'Random 5-fold', wf: 'Walk-forward', gap: 'Walk-forward with a gap' },
 series: { sim: 'Simulated random walk', sp500: 'S&P 500' },
 L: { origin: 'Origin t', target: 'target', rank: 'Order', feature: 'Feature', entry: 'λ at entry', coef: 'Coefficient at the chosen λ', model: 'Model',
   dmp: r => `DM p, against ${r}`, rmseRel: 'RMSE (pp), relative to RW', acc: 'Accuracy', vsBase: 'Against the baseline (pp)', upShare: 'Days forecast “up”' },
 fe1: (m, mn, mnd, mx, mxd, n) => `Daily mean load: mean ${m} GW over ${n} days; minimum ${mn} GW on ${mnd}, maximum ${mx} GW on ${mxd}.`,
 fe2: (h, k, n, c0, c6, cm) => `Target y<sub>t+${h}</sub>, ${n} rows: the correlation with y<sub>t</sub> is ${c0}, with the same weekday ${k === 1 ? 'one week' : k + ' weeks'} earlier ${c6}, with the 7-day mean ${cm}.`,
 wlCap: (a, b) => `Daily mean load (GW), ${a} – ${b}; one row per origin with all lags and the target known.`,
 wls1: (n, p, h, r) => `From n = ${n} days, with p = ${p} lags and horizon h = ${h}: the first p − 1 = ${p - 1} days lack lags, the last h = ${h} lack a target; n − p − h + 1 = <b>${r}</b> rows remain.`,
 wls2: (n, h, r) => `On the whole series (n = ${n}), the 28-day mean needs 27 earlier days: the table for h = ${h} has ${n} − 28 − ${h} + 1 = <b>${r}</b> rows.`,
 wls3: w => `The first target falls on a ${w}: the model needs the weekday of the target, not of the origin.`,
 wl1: (r, p, h, d, y, t) => `${r} rows with p = ${p} and h = ${h}; first row: origin ${d}, y<sub>t</sub> = ${y} GW, target ${t} GW.`,
 st1: (o, a, b, b2, d2, b7, d7) => `AR(1) estimated on the data up to ${o}: ŷ<sub>t+1</sub> = ${a} + ${b}·y<sub>t</sub>. The coefficient on y<sub>T</sub> at h = 2: ${b2} implied (b²), ${d2} estimated directly; at h = 7: ${b7} against ${d7}.`,
 st2: (r, d) => `MAE over the 14 days: ${r} MW recursive, ${d} MW direct (extra run: an AR(1) without calendar; the slide is a diagram).`,
 wss1: (y, b, f1, f2) => `Recursive: ŷ<sub>T+1</sub> = 1.2 + ${b}·${y} = ${f1}; ŷ<sub>T+2</sub> = 1.2 + ${b}·${f1} = <b>${f2}</b>.`,
 wss2: (b, a2, b2) => `Implied two-step rule: a(1 + b) + b²y<sub>T</sub> = 1.2·(1 + ${b}) + ${b}²·y<sub>T</sub> = ${a2} + ${b2}·y<sub>T</sub>.`,
 wss3: (y, fd, d) => `Direct: ŷ<sub>T+2</sub> = 2.3 + 0.6·${y} = <b>${fd}</b>; the difference from the recursive forecast is ${d}.`,
 ws1: (y, r, d) => `y<sub>T</sub> = ${y}: the two-step forecast is ${r} recursive and ${d} direct.`,
 cv1: (k, s, h) => `K = ${k}: test blocks of ${s} observations in walk-forward; the gap removes the last ${h} training rows before each test block.`,
 cv2: (h, a, b, c) => h > 1 ? `With targets that overlap over ${h} observations, the test rows with a training row less than ${h} steps away: ${a} in random K-fold, ${b} in walk-forward without a gap, ${c} with a gap.` : `With h = ${h} the targets do not overlap: no training row shares its target with a test row.`,
 lk1: (h, a, b, c, d, e, f) => `h = ${h}. Simulated random walk: R² = ${a} with random 5-fold, ${b} walk-forward, ${c} with a gap. S&P 500: ${d}, ${e} and ${f}.`,
 lk2: (h, k, r, n) => `Neighbouring targets share ${k} of ${h} returns (theoretical correlation ${r}); ${n} days for the S&P 500.`,
 wks1: (h, k, r, a, n) => `Targets on neighbouring days share ${k} of ${h} returns: their correlation is ${k}/${h} = ${r} in theory and ${a} in the sample (${n} rows).`,
 wks2: (s, f) => `${s}, by fold (test rows: Σ(y − ŷ)² / Σ(y − ȳ)²): ${f}.`,
 wks3: (a, b, r) => `Pooled R² = 1 − ${a}/${b} = <b>${r}</b>.`,
 wk1: (s, h, a, b, c) => `${s}, h = ${h}: R² = ${a} (random 5-fold), ${b} (walk-forward), ${c} (with a gap).`,
 bv1: (s, d, t, n) => `σ = ${s}: the expected test error is smallest at depth ${d} (${t}), above the noise variance ${n}.`,
 bv2: (b1, v1, b10, v10, t10) => `Depth 1: bias² ${b1}, variance ${v1}; depth 10: bias² ${b10}, variance ${v10}, training error ${t10}.`,
 wrs1: (s, x, o) => `OLS: β̂ = S<sub>xy</sub>/S<sub>xx</sub> = ${s}/${x} = <b>${o}</b>.`,
 wrs2: (l, s, d, b, f) => `Ridge, λ = ${l}: β̂ = ${s}/${d} = <b>${b}</b>; shrinkage factor S<sub>xx</sub>/(S<sub>xx</sub> + λ) = ${f}, never 0.`,
 wrs3: (l, l2, a, r, b) => `Lasso, λ = ${l}: threshold λ/2 = ${l2}; |S<sub>xy</sub>| − λ/2 = ${a} − ${l2} → ${r}; β̂ = ${r}/100 = <b>${b}</b>.`,
 wrs4: z => `For λ ≥ 2|S<sub>xy</sub>| = ${z} the lasso coefficient is exactly 0.`,
 wr1: (o, lr, br, ll, bl, z) => `OLS ${o}; ridge with λ = ${lr}: ${br}; lasso with λ = ${ll}: ${bl} (exactly 0 from λ = ${z}).`,
 sh1: (e, n, f, k, c0, cm) => `Data up to ${e} (${n} rows): the first features to enter are ${f}; at the chosen penalty the lasso keeps ${k} of 30, with lag0 ${c0} and Monday ${cm} (GW per standard deviation).`,
 sh2: a => `Penalty chosen by walk-forward validation (TimeSeriesSplit, 5 folds): λ = ${a}.`,
 ltCap: (e, n) => `Data up to ${e}, ${n} rows; order of entry along the lasso path (decreasing λ); highlighted rows: the first three.`,
 lt1: (f, l, k) => `Order of entry: ${f}. At λ = ${l}, ${k} of 30 features are non-zero.`,
 tr0: n => `No split respects the minimum leaf size (${n} days): the forecast is the mean.`,
 tr1: (f, t, v, n, N) => `First question: ${f} ≤ ${t} GW? Leaves: ${v} GW, with ${n} days (${N} training days).`,
 tr2: (r, k) => `Training R²: ${r}, with ${k} possible forecast values.`,
 wbs1: (f0, r) => `F<sub>0</sub> = mean of the targets 5, 6, 8 = ${f0}; residuals: ${r}.`,
 wbs2: (l, r, nu, f0, r2, f1, f1a) => `The stump that isolates the third point predicts ${l} for the first two and ${r} for the third; with ν = ${nu}: F<sub>1</sub>(x<sub>3</sub>) = ${f0} + ${nu}·${r2} = <b>${f1}</b>, F<sub>1</sub>(x<sub>1</sub>) = ${f1a}.`,
 wbs3: (m, v, e) => `After m = ${m} tree${m === 1 ? '' : 's'}: F<sub>m</sub> = ${v}; training MSE ${e}.`,
 wb1: (nu, v) => `ν = ${nu}: the forecast for the third point (target 8) is ${v} after 5, 20 and 60 trees.`,
 en1: (s, a, b, t, td, r, rd, n) => `Training up to ${s} (${a} days), validation ${b} days: a single tree ${t} at depth ${td}, random forest ${r} at depth ${rd}; seasonal naive ${n} (MSE, GW²).`,
 en2: (a, ia, b, ib) => `Boosting: minimum ${a} after ${ia} tree${ia === 1 ? '' : 's'} with ν = 0.3; ${b} after ${ib} tree${ib === 1 ? '' : 's'} with ν = 0.05.`,
 ex1: (s, mx, l, ld, f, dl) => `Training up to ${s}: the largest index seen ${mx}; by ${ld} the index reaches ${l} (Eurostat, downloaded on ${dl}); the forest on levels forecasts ${f}.`,
 ex2: (a, b, n) => `RMSE over ${n} months: ${a} index points on levels, ${b} on cumulated monthly changes.`,
 im1: (h, a, va, b, vb) => `Horizon ${h} day${h === 1 ? '' : 's'}: ${a} (+${va} MW of MAE when shuffled) and ${b} (+${vb} MW).`,
 ml1: (q, a, b, c, n) => `q = ${q}: ${a} input weights, ${b} biases, ${c} output weights and one constant: ${n} parameters.`,
 ml2: (q, e) => `With ${q} ReLU neuron${q === 1 ? '' : 's'}, the piecewise-linear function in the figure approximates tanh on [−4, 4] with maximum error ${e} (extra computation).`,
 rn1: (o, a, b, n) => `The LSTM retrained at ${o}: the loss falls from ${a} to ${b} in ${n} epochs (same seed; the forecasts are those of ch9_load_ml.csv).`,
 rn2: (a, b) => `MAE over the 14 days: LSTM ${a} MW, DHR ${b} MW.`,
 ldCap: (n, o, h) => `${n} origins, from ${o}; horizons ${h.replace('-', '–')}; MASE scale 309 MW; DM on the origin-average absolute errors, with the HLN correction.`,
 ld1: (n, d, r, p, ref) => `${n} origins: MASE ${d} for DHR and ${r} for ridge; DM p of ridge against ${ref}: ${p}.`,
 lr1: (n, o, a, ma, b, mb, l) => `First ${n} origins (up to ${o}): the most accurate are ${a} (MASE ${ma}) and ${b} (${mb}); LSTM ${l}.`,
 lr2: (a, b, c, d) => `MAE at 1 and at 14 days: DHR ${a} and ${b} MW, GB direct ${c} and ${d} MW.`,
 wms1: (a, b, n, m, k, s) => `First training sample, ${a} – ${b}: n = ${n} days, n − m = ${k} differences |y<sub>t</sub> − y<sub>t−${m}</sub>|; their mean, the scale, is <b>${s} MW</b>.`,
 wms2: (md, a, s, q) => `${md}: MAE over the 364 forecasts = ${a} MW; MASE = ${a}/${s} = <b>${q}</b>.`,
 wms3: (m, s, q) => m === 7 ? `Below 1: the model is more accurate out of sample than the seasonal naive method in sample.` : `With the weekly scale of the lecture (m = 7, ${s} MW), MASE would be ${q}: the scale changes the level, not the ranking of the models.`,
 wm1: (m, s, md, q) => `m = ${m}: scale ${s} MW; MASE of ${md} ${q}.`,
 wds1: (a, b, n, d) => `d<sub>t</sub> = origin MAE of ${a} minus that of ${b}, for n = ${n} origins; d̄ = <b>${d} MW</b>.`,
 wds2: (L, g, v) => `With ${L} autocovariance${L === 1 ? '' : 's'} (h = ${L + 1}): ${g} (MW²); V̂ = ${v} MW².`,
 wds3: (d, s, dm, f, h) => `DM = d̄/√V̂ = ${d}/${s} = ${dm}; HLN correction c<sub>h</sub> = √((n + 1 − 2h + h(h − 1)/n)/n) = ${f}: DM* = c<sub>h</sub>·DM = <b>${h}</b>.`,
 wds4: (df, p) => `p = 2·P(t<sub>${df}</sub> > |DM*|) = <b>${p}</b>.`,
 wd1: (a, b, L, h, p) => `${a} against ${b}, ${L} autocovariance${L === 1 ? '' : 's'}: DM* = ${h}, p = ${p}.`,
 lf1: (o, m) => `Origin ${o}, MAE over the 14 days: ${m}.`,
 lf2: (n, b, v) => `${n} public holiday${n === 1 ? '' : 's'} in the window; the lowest MAE: ${b}, ${v} MW.`,
 wps1: (t, q, a, b) => `τ = ${t}, q = ${q} GW: y = 7.4 costs ${a}; y = 6.5 costs ${b}.`,
 wps2: (t, q, e) => `The mean loss over the 364 observed values is smallest at q = ${q} GW; the empirical quantile of order ${t} is ${e} GW.`,
 wps3: () => `With τ = 0.5 the loss is half the absolute error, and the minimum is the median.`,
 wp1: (t, q, a, b) => `τ = ${t}, q = ${q}: the loss is ${a} for y = 7.4 and ${b} for y = 6.5.`,
 wcs1: (n, l, a, n1, k) => `n = ${n}, level ${l} (α = ${a}): k = ⌈${n1}·(1 − ${a})⌉ = <b>${k}</b>.`,
 wcs2: (k, q) => `q̂ = the absolute error of rank ${k} in increasing order = <b>${q} MW</b>.`,
 wcs3: (o, h, y, a, b, v, ok) => `Origin ${o}, h = ${h}: ŷ = ${y} GW, the interval [${a}, ${b}] GW; the observed value ${v} GW is ${ok ? 'inside' : 'outside'} the interval.`,
 wc1: (l, c, w) => `Level ${l}: coverage ${c} of the 364 forecasts, mean width ${w} MW.`,
 iv1: (c, w, a, b) => `Quantile GB (90%): coverage ${c}, mean width ${w} MW; ${a} of the outcomes below and ${b} above the interval.`,
 iv2: (l, c, w, a, b) => `Split conformal at ${l}: coverage ${c}, mean width ${w} MW; ${a} below and ${b} above.`,
 in1: (mx, mxd, mn, mnd, l, ld, dl) => `Romanian 12-month inflation: maximum ${mx}% in ${mxd}, minimum ${mn}% in ${mnd}, ${l}% in ${ld} (Eurostat, downloaded on ${dl}).`,
 in2: (y, a, g, l) => `Origins from ${y}: relative RMSE of the AR ${a}; at 12 months GB global ${g}, GB local ${l}.`,
 itCap: (p, n) => `Test origins: ${p}; number of origins for h = 1, 3, 6, 12: ${n}. In brackets: RMSE relative to RW.`,
 itNote: 'DM on squared errors, with h − 1 autocovariances and the HLN correction; p &lt; 0.05 in bold.',
 it1: (p, a, g) => `${p}: relative RMSE of the AR ${a} (h = 1, 3, 6, 12); GB global ${g}.`,
 rtCap: (y, a, b) => `Test period ${y}–2026: ${a} S&P 500 days and ${b} DAX days; DM t &lt; −1.96: QLIKE significantly lower than HAR.`,
 rt1: (y, a, b, l, m, tl, tm) => `From ${y}: S&P 500 (${a} days), lasso ${l}% and MLP ${m}% R² against HAR, DM t ${tl} and ${tm}; DAX: ${b} days.`,
 rv1: (k, y, q) => `${k}, ${y}: QLIKE ${q}.`,
 baseRow: u => `baseline; ${u} up days`,
 sbCap: (y, a, b) => `Test period ${y}–2026: ${a} S&P 500 days and ${b} BET days; the baseline: the majority class of the training sample.`,
 sb1: (y, a, b, mn, mx) => `From ${y}: up days ${a} (S&P 500) and ${b} (BET); AUC between ${mn} and ${mx}.`,
 si1: (t, a, ma, b, mb) => `Threshold ${t}. S&P 500: baseline ${a}; ${ma}. BET: baseline ${b}; ${mb}.`,
 si2: (u, p) => `GB on the S&P 500 forecasts “up” on ${u} of the days; p against the baseline ${p}.`,
 m41: (w, m, ty, o, c, r) => `Weight of sMAPE ${w}: first is method ${m} (${ty}), OWA ${o}; Comb ${c}, rank ${r}.`,
 m42: (n, r, o, a, b, k) => `The ${n} pure ML methods: the best ranks ${r} (OWA ${o}); ${a} beat${a === 1 ? 's' : ''} Comb, ${b} beat${b === 1 ? 's' : ''} Naive2; ${k} method${k === 1 ? ' has' : 's have'} OWA above 1.7.`,
 gk1: (m, r, s, q) => `${m}: monthly R² ${r}%, Sharpe ratio ${s} (${q} times that of OLS-3+H).`,
 gk2: (m, t) => `Over the 30 test years, an annual Sharpe ratio S corresponds to a t-statistic t ≈ S·√30 for the mean return: ${t} for ${m} (extra computation).`,
};
})();
""",
}

# Question for the room and answer of every block; RO and EN say the same, with the numbers of the page (verified by a node script).
QA = {
 'ro': {
  'features': ('Duceți orizontul la h = 7. Cît devine corelația țintei cu y<sub>t</sub> și cu media pe 7 zile?',
               '0,81 cu y<sub>t</sub> (0,77 la h = 1) și 0,60 cu media pe 7 zile (0,74 la h = 1). La h = 7 ziua de azi este aceeași zi a săptămînii ca ținta, deci y<sub>t</sub> preia rolul lagului săptămînal, iar media pe 7 zile nu mai poartă ziua săptămînii.'),
  'wx-lag': ('Cu p = 5 laguri și h = 3, cîte rînduri complete dau cele 9 zile?',
             '9 − 5 − 3 + 1 = 2 rînduri, cu originile 25 și 26 iunie 2025. Pe toată seria, tabelul pentru h = 3 are 1642 − 28 − 3 + 1 = 1612 rînduri.'),
  'strategies': ('Alegeți originea 6 aprilie 2026, cu Paștele în fereastră. Care strategie are MAE mai mic pe cele 14 zile?',
                 'Cea directă: 668 MW, față de 898 MW pentru cea recursivă. La h = 7 coeficientul direct al lui y<sub>T</sub> este 0,81, iar cel implicit b⁷ doar 0,15: iterarea unui AR(1) pierde tiparul săptămînal, pe care modelul direct îl păstrează.'),
  'wx-strat': ('Fixați y<sub>T</sub> = 7. Care prognoză pe doi pași este mai mare?',
               'Cea recursivă: 2,16 + 0,64·7 = 6,64, față de 2,3 + 0,6·7 = 6,50 pentru cea directă. Cele două reguli se intersectează la y<sub>T</sub> = 3,5; peste acest prag regula recursivă, cu panta mai mare, dă mereu mai mult.'),
  'cv-schemes': ('Duceți spațiul la h = 6. Ce parte din rîndurile de test din walk-forward fără spațiu are un rînd de antrenare la mai puțin de 6 pași?',
                 '63% (25% la h = 3); în K-fold aleator 100%, cu spațiu 0%. Primele h − 1 = 5 dintre cele 8 rînduri ale fiecărui bloc de test au ținte care se suprapun cu ultimele rînduri de antrenare.'),
  'leakage': ('Alegeți h = 1: ținta este un singur randament, fără suprapunere. Ce R² raportează 5-fold aleator pentru mersul aleator simulat?',
              '−4,5%, aproape de walk-forward (−6,3%) și de walk-forward cu spațiu (−6,1%). Fără ținte suprapuse, un rînd de test nu mai are aproape-copii în grupurile de antrenare, iar R² de 34,5% dispare.'),
  'wx-leak': ('Alegeți h = 63. Ce R² dau 5-fold aleator și walk-forward cu spațiu pentru S&P 500?',
              '43,4% și −43,6%. Țintele vecine au în comun 62 din 63 de randamente (corelația de selecție 0,976), deci grupurile aleatoare scurg și mai multă informație. Fără spațiu, walk-forward dă −36,2%: ultimele rînduri de antrenare au ținte care se suprapun cu primele rînduri de test, un leakage mic pe care spațiul îl elimină.'),
  'bias-variance': ('Ridicați zgomotul la σ = 0,6. La ce adîncime este minimă eroarea de test așteptată?',
                    'La adîncimea 2 (0,465; varianța zgomotului este 0,36). Cu σ = 0,1 minimul trece la adîncimea 6. Mai mult zgomot crește varianța arborilor adînci, deci complexitatea potrivită scade.'),
  'wx-shrink': ('Puneți S<sub>xy</sub> = 30. De la ce λ devine coeficientul lasso exact 0 și cît dă ridge cu λ = 50?',
                'De la λ = 2·30 = 60; ridge dă 30/150 = 0,200, nenul. Pragul lasso crește cu |S<sub>xy</sub>|: variabilele cu o legătură slabă cu ținta ies primele.'),
  'shrinkage': ('Tăiați eșantionul la 31 decembrie 2023. Cîte variabile păstrează lasso la penalizarea aleasă?',
                '23 din 30 (22 pe datele pînă la 30 iunie 2025), pe 702 rînduri, cu λ = 0,00486; primele variabile care intră sînt tot lag6, lag0 și lag13.'),
  'lasso-table': ('Mutați penalizarea 6 pași spre un λ mai mare (−6). Cîte variabile rămîn nenule?',
                  '19 din 30, la λ = 0,00441 (22 la λ ales, 0,00173). Ordinea de intrare nu depinde de λ ales: lag6, lag0, lag13.'),
  'tree': ('Antrenați arborele pe datele pînă la 29 iunie 2026. Se schimbă prima întrebare?',
           'Variabila rămîne same_dow, iar pragul coboară de la 6,08 la 6,04 GW; frunzele devin 5,27; 5,86; 6,37 și 7,17 GW, cu 1613 zile, iar R² rămîne 0,64.'),
  'wx-boost': ('Puneți ν = 1. Cît este F<sub>1</sub> în cele trei puncte?',
               '5,50; 5,50 și 8,00: primul arbore corectează complet reziduul punctului izolat, iar MSE la antrenare scade de la 1,56 la 0,167 după un singur arbore. O rată de învățare mare ajustează repede datele de antrenare, inclusiv zgomotul lor.'),
  'ensembles': ('Antrenați pînă la 31 decembrie 2023 (validare de 546 de zile). După cîți arbori atinge boosting-ul cu ν = 0,3 minimul?',
                'După 24 de arbori (MSE 0,093), față de 72 în curs; cu ν = 0,05 minimul vine după 222 de arbori (0,092). Pe un eșantion de antrenare mai scurt, boosting-ul rapid începe mai devreme să ajusteze zgomotul.'),
  'extrapolation': ('Antrenați pădurile pînă în decembrie 2022. Cît este RMSE pe niveluri?',
                    '22,9 puncte de indice (36,9 cu antrenarea pînă în 2020): cel mai mare nivel văzut este acum 135,6, dar prognoza rămîne plafonată la 134,1, cînd indicele ajunge la 173,4. Pe variațiile lunare RMSE este 0,68.'),
  'importance': ('Duceți orizonturile la 7 și 8 zile. Ce variabilă domină la h = 8?',
                 'cos1 (+98 MW), urmată de lag6 (+84 MW). La h = 7 domină încă lag0 (+317 MW), pentru că la 7 zile y<sub>t</sub> este aceeași zi a săptămînii ca ținta.'),
  'mlp': ('Cîți parametri are rețeaua cu q = 10 neuroni ascunși?',
          '3·10 + 10 + 10 + 1 = 51. Cu 10 neuroni ReLU, funcția liniară pe porțiuni aproximează tanh cu eroarea maximă 0,048, față de 0,102 cu 5.'),
  'rnn': ('Alegeți originea 6 aprilie 2026. Cît este MAE al LSTM, față de DHR?',
          '661 MW față de 344 MW. LSTM nu vede indicatorul de sărbătoare și repetă tiparul săptămînal peste Paște, deși pierderea la antrenare scade de la 0,951 la 0,210.'),
  'load-results': ('Folosiți doar primele 10 origini (pînă la 3 noiembrie 2025). Care model are cel mai mic MASE?',
                   'Tot DHR (0,83), urmat de ridge (0,86). LSTM are aici 0,95, dar pe toate cele 26 de origini urcă la 1,16.'),
  'load-forecasts': ('Alegeți originea 20 aprilie 2026, după Paște. Care model are cel mai mic MAE?',
                     'MLP, cu 255 MW; GB recursiv are 260, GB direct 276, naivul sezonier 283, DHR 286 și LSTM 398 MW. Fără Paște în fereastră, diferențele dintre modele sînt mici.'),
  'wx-mase': ('Schimbați perioada scalei la m = 1. Cît devin scala și MASE pentru DHR?',
              'Scala devine 371 MW și MASE 0,744 (309 MW și 0,894 cu m = 7). Diferențele de la o zi la alta sînt mai mari decît cele de la o săptămînă la alta, pentru că trecerea la weekend schimbă consumul; scala mută nivelul, nu ordinea modelelor.'),
  'load-table': ('Eliminați primele 10 origini. Rămîne diferența dintre ridge și DHR nesemnificativă?',
                 'Da: pe 16 origini ridge are MASE 0,92 și DHR 0,93, iar DM p = 0,791. Nesemnificativ înseamnă că testul nu separă diferența de zgomot, nu că modelele sînt egale.'),
  'wx-dm': ('Includeți 2 autocovarianțe (h = 3) pentru ridge față de DHR. Cît devine p?',
            '0,953 (DM* = 0,060), față de 0,942 fără autocovarianțe: d̄ = 0,9 MW este mic față de abaterea sa standard, oricum ar fi estimată varianța.'),
  'wx-pinball': ('Puneți τ = 0,5. Cît costă y = 7,4 și y = 6,5 pentru q = 7,0?',
                 '0,200 și 0,250: cu τ = 0,5 pierderea este jumătate din eroarea absolută (0,4/2 și 0,5/2), iar minimul pierderii medii se mută la mediana celor 364 de valori.'),
  'wx-conformal': ('Coborîți nivelul la 50%. Cît este k și ce acoperire obține intervalul?',
                   'k = ⌈183·0,5⌉ = 92 și q̂ = 175 MW pentru h = 1; acoperirea pe cele 364 de prognoze este 49,5%, cu lățimea medie 509 MW.'),
  'intervals': ('Coborîți nivelul conformal la 80%. Ce acoperire obține?',
                '77,5%, cu lățimea medie 1000 MW: 13,7% dintre valori sînt sub interval și 8,8% deasupra. GB cuantilic rămîne la 66,5% pentru un nivel nominal de 90%.'),
  'inflation': ('Începeți originile de test în 2022. Cît este RMSE relativ al GB local la 12 luni?',
                '0,64 (1,21 pe toate originile), iar GB global are 1,19. Pe puținele origini ale perioadei de dezinflație de după vîrful din 2022, clasamentul de la 12 luni se inversează: o comparație pe cîteva zeci de erori este fragilă.'),
  'inf-table': ('Alegeți originile din 2021–2023. Cît este RMSE relativ al AR la h = 6?',
                '0,83 (0,88; 0,86; 0,83 și 0,98 la h = 1, 3, 6, 12): în anii cu inflație mare AR rămîne sub mersul aleator pînă la 6 luni, iar GB global are 1,10 la h = 6.'),
  'rv-table': ('Începeți perioada de test în 2020. Rămîne lasso semnificativ mai precis decît HAR pe S&P 500?',
               'Da: R² față de HAR +7,4% și DM t = −3,42 pe 1682 de zile; MLP are +7,9% și t = −2,28.'),
  'rv': ('Afișați lasso și MLP pentru S&P 500 în 2022 (anul din stînga). Bate vreunul HAR după QLIKE în acel an?',
         'Da, amîndouă: QLIKE 1,536 (lasso) și 1,541 (MLP), față de 1,553 pentru HAR. În 2025 HAR are 0,990, sub amîndouă (0,997 și 1,002): un singur an nu decide clasamentul.'),
  'sign-table': ('Începeți perioada de test în 2016. Cît este cel mai mare AUC?',
                 '0,541, cel mai mic 0,507; zilele de creștere sînt 54,6% (S&P 500) și 55,7% (BET). AUC rămîne aproape de 0,5.'),
  'sign': ('Ridicați pragul la 0,55. Ce acuratețe are GB pe S&P 500?',
           '49,3%, sub reperul de 54,4%: cu pragul mai sus modelul prognozează „creștere” doar în 40,4% din zile, deși piața crește în 54,4% din ele.'),
  'm4': ('Puneți ponderea sMAPE la 1 (OWA calculat doar din sMAPE). Pe ce loc ajunge Comb?',
         'Pe locul 23 (OWA 0,926), față de 19; cîștigătorul rămîne metoda 118, hibridă, cu 0,839, iar cea mai bună metodă ML pură coboară pe locul 27.'),
  'gkx': ('De cîte ori este raportul Sharpe al NN4 mai mare decît cel al OLS-3+H?',
          '1,35/0,61 = 2,21 ori, cu un R² lunar de doar 0,39%: o predictibilitate mică, folosită pe mii de acțiuni și luni, susține un portofoliu valoros.'),
 },
 'en': {
  'features': ('Move the horizon to h = 7. What do the correlations of the target with y<sub>t</sub> and with the 7-day mean become?',
               '0.81 with y<sub>t</sub> (0.77 at h = 1) and 0.60 with the 7-day mean (0.74 at h = 1). At h = 7 today is the same weekday as the target, so y<sub>t</sub> takes the role of the weekly lag, while the 7-day mean carries no weekday.'),
  'wx-lag': ('With p = 5 lags and h = 3, how many complete rows do the 9 days give?',
             '9 − 5 − 3 + 1 = 2 rows, with origins 25 and 26 June 2025. On the whole series the table for h = 3 has 1642 − 28 − 3 + 1 = 1612 rows.'),
  'strategies': ('Choose the origin 6 April 2026, with Easter in the window. Which strategy has the lower MAE over the 14 days?',
                 'The direct one: 668 MW against 898 MW for the recursive one. At h = 7 the direct coefficient on y<sub>T</sub> is 0.81, the implied b⁷ only 0.15: iterating an AR(1) loses the weekly pattern, which the direct model keeps.'),
  'wx-strat': ('Set y<sub>T</sub> = 7. Which two-step forecast is larger?',
               'The recursive one: 2.16 + 0.64·7 = 6.64, against 2.3 + 0.6·7 = 6.50 for the direct one. The two rules cross at y<sub>T</sub> = 3.5; above it the recursive rule, with the steeper slope, always gives more.'),
  'cv-schemes': ('Move the gap to h = 6. What share of the test rows of walk-forward without a gap has a training row less than 6 steps away?',
                 '63% (25% at h = 3); 100% in random K-fold, 0% with a gap. The first h − 1 = 5 of the 8 rows of each test block have targets that overlap the last training rows.'),
  'leakage': ('Choose h = 1: the target is one return, without overlap. What R² does random 5-fold report for the simulated random walk?',
              '−4.5%, close to walk-forward (−6.3%) and to walk-forward with a gap (−6.1%). Without overlapping targets a test row has no near-copies in the training folds, and the 34.5% disappears.'),
  'wx-leak': ('Choose h = 63. What R² do random 5-fold and walk-forward with a gap give for the S&P 500?',
              '43.4% and −43.6%. Neighbouring targets share 62 of 63 returns (sample correlation 0.976), so random folds leak even more. Without a gap walk-forward gives −36.2%: the last training rows have targets that overlap the first test rows, a small leak that the gap removes.'),
  'bias-variance': ('Raise the noise to σ = 0.6. At what depth is the expected test error smallest?',
                    'At depth 2 (0.465; the noise variance is 0.36). With σ = 0.1 the minimum moves to depth 6. More noise raises the variance of deep trees, so the right complexity falls.'),
  'wx-shrink': ('Set S<sub>xy</sub> = 30. From what λ is the lasso coefficient exactly 0, and what does ridge give with λ = 50?',
                'From λ = 2·30 = 60; ridge gives 30/150 = 0.200, not zero. The lasso threshold grows with |S<sub>xy</sub>|: features weakly related to the target leave first.'),
  'shrinkage': ('Cut the sample at 31 December 2023. How many features does the lasso keep at the chosen penalty?',
                '23 of 30 (22 on the data up to 30 June 2025), on 702 rows, with λ = 0.00486; the first features to enter are still lag6, lag0 and lag13.'),
  'lasso-table': ('Move the penalty 6 grid steps towards a larger λ (−6). How many features remain non-zero?',
                  '19 of 30, at λ = 0.00441 (22 at the chosen λ, 0.00173). The order of entry does not depend on the chosen λ: lag6, lag0, lag13.'),
  'tree': ('Train the tree on the data up to 29 June 2026. Does the first question change?',
           'The feature stays same_dow and the threshold falls from 6.08 to 6.04 GW; the leaves become 5.27, 5.86, 6.37 and 7.17 GW, with 1613 days, and R² stays 0.64.'),
  'wx-boost': ('Set ν = 1. What is F<sub>1</sub> at the three points?',
               '5.50, 5.50 and 8.00: the first tree fully corrects the residual of the isolated point, and the training MSE falls from 1.56 to 0.167 after one tree. A large learning rate fits the training data fast, noise included.'),
  'ensembles': ('Train up to 31 December 2023 (546 validation days). After how many trees does boosting with ν = 0.3 reach its minimum?',
                'After 24 trees (MSE 0.093), against 72 in the lecture; with ν = 0.05 the minimum comes after 222 trees (0.092). On a shorter training sample fast boosting starts fitting noise earlier.'),
  'extrapolation': ('Train the forests up to December 2022. What is the RMSE on levels?',
                    '22.9 index points (36.9 with training up to 2020): the largest level seen is now 135.6, but the forecast stays capped at 134.1 while the index reaches 173.4. On monthly changes the RMSE is 0.68.'),
  'importance': ('Move the horizons to 7 and 8 days. Which feature dominates at h = 8?',
                 'cos1 (+98 MW), followed by lag6 (+84 MW). At h = 7 lag0 still dominates (+317 MW), because at 7 days y<sub>t</sub> is the same weekday as the target.'),
  'mlp': ('How many parameters does the network with q = 10 hidden neurons have?',
          '3·10 + 10 + 10 + 1 = 51. With 10 ReLU neurons the piecewise-linear function approximates tanh with maximum error 0.048, against 0.102 with 5.'),
  'rnn': ('Choose the origin 6 April 2026. What is the MAE of the LSTM, against DHR?',
          '661 MW against 344 MW. The LSTM sees no holiday flag and repeats the weekly pattern over Easter, although its training loss falls from 0.951 to 0.210.'),
  'load-results': ('Use only the first 10 origins (up to 3 November 2025). Which model has the lowest MASE?',
                   'Still DHR (0.83), followed by ridge (0.86). The LSTM has 0.95 here, but over all 26 origins it rises to 1.16.'),
  'load-forecasts': ('Choose the origin 20 April 2026, after Easter. Which model has the lowest MAE?',
                     'The MLP, with 255 MW; GB recursive has 260, GB direct 276, the seasonal naive 283, DHR 286 and the LSTM 398 MW. Without Easter in the window the differences between the models are small.'),
  'wx-mase': ('Change the period of the scale to m = 1. What do the scale and the MASE of DHR become?',
              'The scale becomes 371 MW and the MASE 0.744 (309 MW and 0.894 with m = 7). Day-to-day differences are larger than week-to-week ones, because the step into the weekend changes the load; the scale moves the level, not the ranking of the models.'),
  'load-table': ('Drop the first 10 origins. Does the difference between ridge and DHR stay not significant?',
                 'Yes: on 16 origins ridge has MASE 0.92 and DHR 0.93, and DM p = 0.791. Not significant means that the test does not separate the difference from noise, not that the models are equal.'),
  'wx-dm': ('Include 2 autocovariances (h = 3) for ridge against DHR. What does p become?',
            '0.953 (DM* = 0.060), against 0.942 without autocovariances: d̄ = 0.9 MW is small relative to its standard error, however the variance is estimated.'),
  'wx-pinball': ('Set τ = 0.5. What do y = 7.4 and y = 6.5 cost for q = 7.0?',
                 '0.200 and 0.250: with τ = 0.5 the loss is half the absolute error (0.4/2 and 0.5/2), and the minimum of the mean loss moves to the median of the 364 values.'),
  'wx-conformal': ('Lower the level to 50%. What is k and what coverage does the interval reach?',
                   'k = ⌈183·0.5⌉ = 92 and q̂ = 175 MW for h = 1; the coverage over the 364 forecasts is 49.5%, with mean width 509 MW.'),
  'intervals': ('Lower the conformal level to 80%. What coverage does it reach?',
                '77.5%, with mean width 1000 MW: 13.7% of the outcomes fall below the interval and 8.8% above. Quantile GB stays at 66.5% for a nominal 90%.'),
  'inflation': ('Start the test origins in 2022. What is the relative RMSE of GB local at 12 months?',
                '0.64 (1.21 over all origins), while GB global has 1.19. On the few origins of the disinflation after the 2022 peak the 12-month ranking reverses: a comparison on a few dozen errors is fragile.'),
  'inf-table': ('Choose the origins of 2021–2023. What is the relative RMSE of the AR at h = 6?',
                '0.83 (0.88, 0.86, 0.83 and 0.98 at h = 1, 3, 6, 12): in the high-inflation years the AR stays below the random walk up to 6 months, while GB global has 1.10 at h = 6.'),
  'rv-table': ('Start the test period in 2020. Is the lasso still significantly more accurate than HAR on the S&P 500?',
               'Yes: R² against HAR +7.4% and DM t = −3.42 on 1682 days; the MLP has +7.9% and t = −2.28.'),
  'rv': ('Show the lasso and the MLP for the S&P 500 in 2022 (left year). Does either beat HAR by QLIKE in that year?',
         'Yes, both: QLIKE 1.536 (lasso) and 1.541 (MLP), against 1.553 for HAR. In 2025 HAR has 0.990, below both (0.997 and 1.002): one year does not decide the ranking.'),
  'sign-table': ('Start the test period in 2016. What is the largest AUC?',
                 '0.541, the smallest 0.507; the up days are 54.6% (S&P 500) and 55.7% (BET). AUC stays close to 0.5.'),
  'sign': ('Raise the threshold to 0.55. What accuracy does GB reach on the S&P 500?',
           '49.3%, below the 54.4% baseline: with the higher threshold the model forecasts “up” on only 40.4% of the days, while the market rises on 54.4% of them.'),
  'm4': ('Set the weight of sMAPE to 1 (OWA from sMAPE only). Where does Comb rank?',
         '23rd (OWA 0.926), against 19th; the winner is still method 118, a hybrid, with 0.839, and the best pure ML method falls to rank 27.'),
  'gkx': ('How many times larger is the Sharpe ratio of NN4 than that of OLS-3+H?',
          '1.35/0.61 = 2.21 times, with a monthly R² of only 0.39%: a small predictability, used on thousands of stocks and months, supports a valuable portfolio.'),
 },
}

# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
PREDICT = {
 'features': (1, [('Amîndouă scad: informația se pierde cu orizontul.', 'Both fall: information is lost with the horizon.'), ('0,81 cu yₜ și 0,60 cu media pe 7 zile.', '0.81 with yₜ and 0.60 with the 7-day mean.'),
                  ('Amîndouă cresc peste 0,9: ținta de peste o săptămînă este mai netedă.', 'Both rise above 0.9: the target a week ahead is smoother.')]),
 'wx-lag': (1, [('6, ca în curs: lagurile nu consumă rînduri.', '6, as in the lecture: lags do not use up rows.'), ('2.', '2.'), ('6: se pierd doar cele h = 3 zile de la sfîrșit.', '6: only the h = 3 days at the end are lost.')]),
 'strategies': (1, [('Cea recursivă: un singur model, estimat pe mai multe rînduri.', 'The recursive one: one model, estimated on more rows.'), ('Cea directă: 668 MW față de 898 MW.', 'The direct one: 668 MW against 898 MW.'),
                    ('Sînt egale: ambele folosesc același AR(1).', 'They are equal: both use the same AR(1).')]),
 'wx-strat': (0, [('Cea recursivă: 6,64 față de 6,50.', 'The recursive one: 6.64 against 6.50.'), ('Sînt egale: ambele estimează aceeași regulă pe doi pași.', 'They are equal: both estimate the same two-step rule.'),
                  ('Cea directă: modelul direct se adaptează orizontului, deci prognozează mai sus.', 'The direct one: the direct model adapts to the horizon, so it forecasts higher.')]),
 'cv-schemes': (1, [('0%: walk-forward păstrează ordinea timpului.', '0%: walk-forward keeps the time order.'), ('63%.', '63%.'), ('100%, ca în K-fold aleator.', '100%, as in random K-fold.')]),
 'leakage': (1, [('Tot circa 34,5%: grupurile aleatoare au mereu acest avantaj.', 'Still about 34.5%: random folds always have this advantage.'), ('−4,5%, aproape de walk-forward.', '−4.5%, close to walk-forward.'),
                 ('Aproape 100%: cu o singură zi, ținta este ușor de copiat.', 'Close to 100%: with one day the target is easy to copy.')]),
 'wx-leak': (0, [('43,4% și −43,6%.', '43.4% and −43.6%.'), ('Ambele în jur de −40%: pe ținte lungi leakage-ul dispare.', 'Both about −40%: on long targets the leak disappears.'),
                 ('Ambele peste 40%: spațiul nu ajută pe ținte lungi.', 'Both above 40%: the gap does not help on long targets.')]),
 'bias-variance': (1, [('Tot la 3: adîncimea optimă nu depinde de zgomot.', 'Still 3: the best depth does not depend on the noise.'), ('La adîncimea 2.', 'At depth 2.'),
                       ('La adîncimea 10: un zgomot mai mare cere un model mai flexibil.', 'At depth 10: more noise needs a more flexible model.')]),
 'wx-shrink': (1, [('Tot de la 180: pragul nu depinde de legătura cu ținta.', 'Still from 180: the threshold does not depend on the link with the target.'), ('De la 60; ridge 0,200.', 'From 60; ridge 0.200.'),
                   ('De la 60; ridge tot 0, ca lasso.', 'From 60; ridge 0 as well, like the lasso.')]),
 'shrinkage': (0, [('23 din 30.', '23 of 30.'), ('Sub 10: pe mai puține date lasso selectează sever.', 'Below 10: on fewer data the lasso selects severely.'),
                   ('Toate 30: pe un eșantion scurt lasso nu elimină nimic.', 'All 30: on a short sample the lasso removes nothing.')]),
 'lasso-table': (1, [('Tot 22: numărul de variabile este fixat de validare.', 'Still 22: the number of features is fixed by the validation.'), ('19 din 30.', '19 of 30.'),
                     ('Doar primele trei, lag6, lag0 și lag13.', 'Only the first three, lag6, lag0 and lag13.')]),
 'tree': (0, [('Nu: tot same_dow, cu pragul 6,04 GW.', 'No: still same_dow, with threshold 6.04 GW.'), ('Da: cu 2026 în date, lag0 devine prima întrebare.', 'Yes: with 2026 in the data, lag0 becomes the first question.'),
              ('Da: arborele se schimbă complet, arborii sînt instabili.', 'Yes: the tree changes completely, trees are unstable.')]),
 'wx-boost': (1, [('6,50 pentru al treilea punct, ca la ν = 0,1.', '6.50 for the third point, as with ν = 0.1.'), ('5,50; 5,50; 8,00.', '5.50, 5.50, 8.00.'),
                  ('5; 6; 8: cu ν = 1 un arbore potrivește toate punctele.', '5, 6, 8: with ν = 1 one tree fits every point.')]),
 'ensembles': (1, [('Tot după 72: numărul de arbori depinde doar de ν.', 'Still after 72: the number of trees depends on ν only.'), ('După 24 de arbori.', 'After 24 trees.'),
                   ('Niciodată: MSE scade pînă la 600 de arbori.', 'Never: the MSE falls up to 600 trees.')]),
 'extrapolation': (1, [('Circa 0,7, ca pe variații: cu mai multe date pădurea învață trendul.', 'About 0.7, as on changes: with more data the forest learns the trend.'), ('22,9 puncte de indice.', '22.9 index points.'),
                       ('Peste 36,9: mai puține luni de test dau o eroare mai mare.', 'Above 36.9: fewer test months give a larger error.')]),
 'importance': (1, [('lag0, la ambele orizonturi.', 'lag0, at both horizons.'), ('cos1.', 'cos1.'), ('holiday: sărbătorile contează mai mult la orizonturi lungi.', 'holiday: holidays matter more at long horizons.')]),
 'mlp': (1, [('26, ca în curs.', '26, as in the lecture.'), ('51.', '51.'), ('100 = 10².', '100 = 10².')]),
 'rnn': (1, [('LSTM, pentru că pierderea lui la antrenare este mică.', 'The LSTM, because its training loss is small.'), ('661 MW față de 344 MW.', '661 MW against 344 MW.'),
             ('Aproape egale: ambele modele văd același istoric.', 'About equal: both models see the same history.')]),
 'load-results': (1, [('Ridge.', 'Ridge.'), ('DHR, cu 0,83.', 'DHR, with 0.83.'), ('LSTM: pe origini de toamnă rețeaua cîștigă.', 'The LSTM: on autumn origins the network wins.')]),
 'load-forecasts': (1, [('DHR, ca la Paște.', 'DHR, as at Easter.'), ('MLP, cu 255 MW.', 'The MLP, with 255 MW.'), ('LSTM.', 'The LSTM.')]),
 'wx-mase': (1, [('309 MW și 0,894: scala nu depinde de m.', '309 MW and 0.894: the scale does not depend on m.'), ('371 MW și 0,744.', '371 MW and 0.744.'),
                 ('Sub 309 MW: ziua de ieri seamănă mai mult cu azi decît săptămîna trecută.', 'Below 309 MW: yesterday is closer to today than last week.')]),
 'load-table': (1, [('Nu: ridge devine semnificativ mai bun.', 'No: ridge becomes significantly better.'), ('Da, p = 0,791.', 'Yes, p = 0.791.'), ('Nu se mai poate testa cu 16 origini.', 'It can no longer be tested with 16 origins.')]),
 'wx-dm': (1, [('Sub 0,05: corecția HAC face diferența semnificativă.', 'Below 0.05: the HAC correction makes the difference significant.'), ('0,953.', '0.953.'),
               ('Tot 0,942: autocovarianțele nu intră în statistică.', 'Still 0.942: the autocovariances do not enter the statistic.')]),
 'wx-pinball': (1, [('Tot 0,380 și 0,025.', 'Still 0.380 and 0.025.'), ('0,200 și 0,250.', '0.200 and 0.250.'), ('0,4 și 0,5, erorile absolute.', '0.4 and 0.5, the absolute errors.')]),
 'wx-conformal': (1, [('k = 165, ca la 90%.', 'k = 165, as at 90%.'), ('k = 92, acoperire 49,5%.', 'k = 92, coverage 49.5%.'), ('k = 92, acoperire tot circa 90%.', 'k = 92, coverage still about 90%.')]),
 'intervals': (1, [('Exact 80%: garanția conformală este exactă.', 'Exactly 80%: the conformal guarantee is exact.'), ('77,5%.', '77.5%.'), ('66,5%, ca GB cuantilic.', '66.5%, like quantile GB.')]),
 'inflation': (1, [('Tot 1,21.', 'Still 1.21.'), ('0,64.', '0.64.'), ('Peste 2: pe puține date GB local face overfitting.', 'Above 2: on few data GB local overfits.')]),
 'inf-table': (1, [('0,90, ca pe toată perioada.', '0.90, as over the whole period.'), ('0,83.', '0.83.'), ('Peste 1: în 2021–2023 nimic nu bate mersul aleator.', 'Above 1: in 2021–2023 nothing beats the random walk.')]),
 'rv-table': (1, [('Nu: anul 2020 anulează cîștigul.', 'No: 2020 cancels the gain.'), ('Da: t = −3,42.', 'Yes: t = −3.42.'), ('Doar MLP rămîne semnificativ.', 'Only the MLP stays significant.')]),
 'rv': (1, [('Nu: HAR are cel mai mic QLIKE în fiecare an.', 'No: HAR has the lowest QLIKE in every year.'), ('Da, amîndouă.', 'Yes, both.'), ('Doar MLP.', 'Only the MLP.')]),
 'sign-table': (1, [('Peste 0,6: pe date mai recente modelele ordonează mai bine.', 'Above 0.6: on more recent data the models rank better.'), ('0,541.', '0.541.'), ('Sub 0,5 pentru toate modelele.', 'Below 0.5 for every model.')]),
 'sign': (1, [('Peste 54,4%: un prag mai strict filtrează semnalele slabe.', 'Above 54.4%: a stricter threshold filters weak signals.'), ('49,3%.', '49.3%.'), ('Tot 52,5%: pragul nu schimbă acuratețea.', 'Still 52.5%: the threshold does not change accuracy.')]),
 'm4': (1, [('Tot pe locul 19.', 'Still 19th.'), ('Pe locul 23.', '23rd.'), ('Pe primul loc.', 'First.')]),
 'gkx': (1, [('Circa 1: un R² atît de mic nu are valoare economică.', 'About 1: such a small R² has no economic value.'), ('2,21 ori.', '2.21 times.'),
             ('2,44 ori, raportul R² (0,39/0,16).', '2.44 times, the ratio of the R² (0.39/0.16).')]),
}
VEIL_CHARTS = {'gkx', 'wx-strat', 'mlp'}


# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slide „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('O medie mobilă pe 7 zile centrată în t este folosită ca variabilă pentru yₜ₊₁. Este permis?',
         'A 7-day moving average centred on t is used as a feature for yₜ₊₁. Is that allowed?'), key=1, opts=[
  (('Da: media netezește zgomotul și nu folosește ținta direct.', 'Yes: the mean smooths the noise and does not use the target directly.'),
   ('Media centrată în t conține yₜ₊₁, yₜ₊₂, yₜ₊₃: ținta intră în variabilă.', 'The mean centred on t contains yₜ₊₁, yₜ₊₂, yₜ₊₃: the target enters the feature.')),
  (('Nu: conține yₜ₊₁, yₜ₊₂, yₜ₊₃; folosiți media yₜ₋₆, …, yₜ.', 'No: it contains yₜ₊₁, yₜ₊₂, yₜ₊₃; use the trailing mean of yₜ₋₆, …, yₜ.'),
   ('O fereastră care se termină în origine folosește doar ce era cunoscut la t.', 'A window that ends at the origin uses only what was known at t.')),
  (('Da, dacă validarea este walk-forward.', 'Yes, if the validation is walk-forward.'),
   ('Walk-forward păstrează ordinea rîndurilor, dar nu repară o variabilă care conține deja viitorul.', 'Walk-forward keeps the order of the rows, but it does not repair a feature that already contains the future.'))]),
 dict(q=('Un random forest antrenat pe nivelurile PIB-ului pînă în 2019 prognozează anul 2024. Ce nu funcționează?',
         'A random forest trained on GDP levels up to 2019 forecasts 2024. What goes wrong?'), key=0, opts=[
  (('Prognoza lui nu poate depăși cel mai mare nivel văzut la antrenare; modelați ratele de creștere.', 'Its forecast cannot exceed the largest level seen in training; model growth rates.'),
   ('Fiecare frunză prognozează o medie a țintelor de antrenare: pe IAPC, random forest pe niveluri rămîne la 110,8 cînd indicele ajunge la 173,4.', 'Each leaf forecasts a mean of training targets: on the HICP the forest on levels stays at 110.8 while the index reaches 173.4.')),
  (('Nimic, dacă pădurea are destui arbori.', 'Nothing, if the forest has enough trees.'),
   ('Mai mulți arbori reduc varianța mediei, dar fiecare arbore rămîne mărginit de țintele de antrenare.', 'More trees reduce the variance of the average, but every tree stays bounded by the training targets.')),
  (('Prognoza crește prea repede, pentru că arborii extrapolează trendul.', 'The forecast rises too fast, because the trees extrapolate the trend.'),
   ('Un arbore nu are pantă: nu poate continua un trend dincolo de datele de antrenare.', 'A tree has no slope: it cannot continue a trend beyond the training data.'))]),
 dict(q=('Un clasificator prognozează semnul randamentelor zilnice cu acuratețea de 53%. Este util?',
         'A classifier predicts the sign of daily returns with 53% accuracy. Is it useful?'), key=2, opts=[
  (('Da: peste 50% înseamnă mai bine decît întîmplarea.', 'Yes: above 50% means better than chance.'),
   ('Reperul nu este 50%: „întotdeauna în creștere” are dreptate în 54,4% din zilele S&P 500.', 'The baseline is not 50%: “always up” is right on 54.4% of S&P 500 days.')),
  (('Nu, niciodată: semnul randamentelor nu poate fi prognozat.', 'No, never: the sign of returns cannot be forecast.'),
   ('Testul nu dovedește imposibilitatea; arată doar că aceste modele nu depășesc reperul pe aceste date.', 'The test does not prove impossibility; it shows only that these models do not beat the baseline on these data.')),
  (('Doar dacă depășește semnificativ ponderea zilelor de creștere din perioada de test (circa 54% pentru S&P 500).', 'Only if it beats the share of up days in the test period (about 54% for the S&P 500), significantly.'),
   ('Cu 53% față de 54,4%, modelul este mai slab decît a nu face nimic.', 'With 53% against 54.4%, the model is worse than doing nothing.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {})) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch9-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH9_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH9_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n')


_predict_js()


if __name__ == '__main__':
    import interactive_common
    page_spec = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA', 'TEXT_JS', 'PREDICT', 'TICKET')})
    page_spec.widgets = lambda lang: {a: v for a, v in widgets_all(lang).items() if a in {b for b, _, _ in BLOCKS}}
    interactive_common.build(page_spec)
    for lg in ('en', 'ro'):
        _post(lg)
