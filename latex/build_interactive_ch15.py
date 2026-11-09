"""Build the interactive pages of Chapter 15 (course review and exam preparation): interactive/ch15.html (EN) and interactive/ch15_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its interactive version on
the right. The result tables of the Box-Jenkins case (unit-root tests, the candidates by BIC, the chosen model, the residual
checks, the forecast accuracy), the slides that give only numbers (the data, the correlogram) and the eight review problems
of the exam section get blocks of their own (EXTRA), placed by slide, each with a chart; on the web lecture each of them
travels inside a chart block and ch15.js moves it to its own frame. The exam format, grading and final-grade frames get no
block. Logic in interactive/ch15.js, inputs in interactive/ch15_data.js (Quantlets/Ch_15/export_interactive.py), slide
images in interactive/slides/.
"""
import json
import math
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 15

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_15'
TEX = {'ro': 'RO/Cursuri/capitol15_recapitulare', 'en': 'EN/Courses/chapter15_review'}

_D = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'interactive', 'ch15_data.js'), encoding='utf-8').read()
_D = json.loads(_D[_D.index('{'):_D.rindex('}') + 1])
PH_ID, TH_ID = _D['numbers']['diag']['ident']['params']['ar.L1'], _D['numbers']['diag']['ident']['params']['ma.S.L12']
PH3, PPH3 = _D['numbers']['p3']['phi'], _D['numbers']['p3']['Phi']
LNT = math.log(_D['fit']['neff'])
P11, P22, D8 = _D['p8']['p11'], _D['p8']['p22'], _D['p8']['d']
TESTS = list(_D['tests'])
ORG = list(_D['fc']['origins'])
P2E = list(_D['p2']['ends'])
Q4 = []
for _i in range(len(_D['p4']['g'])):
    _m = int(_D['p4']['q0'][5:7]) - 1 + 3 * _i
    Q4.append(f"{int(_D['p4']['q0'][:4]) + _m // 12}Q{_m % 12 // 3 + 1}")
P4E = [q for q in Q4 if (q.endswith('Q4') and int(q[:4]) >= 2015) or q == Q4[-1]]
TAB5 = [_D['numbers']['grid']['ident']['model']] + [r['model'] for r in _D['numbers']['grid']['rows'] if r['model'] != _D['numbers']['grid']['ident']['model']][:4]
CHOSEN = _D['numbers']['grid']['chosen']
del _D

# (anchor, chart file, section key); order of the slides
BLOCKS = [('data', 'tsa_ch15_bj_data', 'bj'), ('acf', 'tsa_ch15_bj_acf', 'bj'), ('diag', 'tsa_ch15_bj_diag', 'bj'), ('fc', 'tsa_ch15_bj_forecast', 'bj')]

# result tables, slides with numbers only and the review problems: (anchor, frame title EN, frame title RO, section key, host chart block)
EXTRA = [('t-data', 'Interpreting the data', 'Interpretarea datelor', 'bj', 'data'),
         ('t-tests', 'Step 2: unit-root and stationarity tests', 'Pasul 2: teste de rădăcină unitară și de staționaritate', 'bj', 'data'),
         ('t-acf', 'Interpreting the correlogram', 'Interpretarea corelogramei', 'bj', 'acf'),
         ('t-grid', 'Step 4: estimating the candidates', 'Pasul 4: estimarea candidaților', 'bj', 'acf'),
         ('t-chosen', 'Step 4: the chosen model', 'Pasul 4: modelul ales', 'bj', 'acf'),
         ('t-diag', 'Interpreting the residual checks', 'Interpretarea verificării reziduurilor', 'bj', 'diag'),
         ('t-fc', 'Interpreting the forecasts', 'Interpretarea prognozelor', 'bj', 'fc')] + \
        [(f'p{k}', f'Problem {k}: solution', f'Problema {k}: rezolvare', 'ex', 'fc') for k in range(1, 9)]

PFX = {'data': 'da', 'acf': 'ac', 'diag': 'dg', 'fc': 'fc', 't-data': 'td', 't-tests': 'tt', 't-acf': 'ta', 't-grid': 'tg', 't-chosen': 'tc', 't-diag': 'tx',
       't-fc': 'tf', **{f'p{k}': f'q{k}' for k in range(1, 9)}}

SECTIONS = {
 'ro': dict(bj='Box–Jenkins de la un capăt la altul: inflația din România', ex='Examenul'),
 'en': dict(bj='Box–Jenkins from start to finish: Romanian inflation', ex='The exam'),
}

L = {
 'ro': dict(file='ch15_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 15 · Recapitulare și pregătire pentru examen', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'casetele „Exemplu rezolvat” refac rezolvările problemelor de examen cu cifrele capitolului.',
   src='IAPC al României (Eurostat prc_hicp_minr, 2015 = 100) este citit cu funcția capitolului pe 4 noiembrie 2026 și tăiat la august 2026, ultima lună din curs; '
       'valorile din ianuarie 2005 – august 2026 nu au fost revizuite și reproduc exact cifrele din curs. Estimările celor 36 de modele SARIMA, reziduurile lor și '
       'prognozele provin din codul capitolului (statsmodels SARIMAX). Problema 2 folosește PIB-ul real al României în versiunea din curs (Capitolul 3, citit din '
       'graficul vectorial al capitolului, deoarece Eurostat a revizuit seria), Problema 4 creșterea PIB, inflația și ROBOR 3M ale VAR-ului din Capitolul 6, Problema 5 '
       'randamentele titlurilor de stat ale SUA (FRED, pînă în septembrie 2026), Problema 6 estimările GARCH(1,1)-t ale BET din Capitolul 5. Browserul calculează '
       'corelogramele, testele Ljung–Box și Jarque–Bera, măsurile de acuratețe, testul Diebold–Mariano, regresiile OLS ale Problemei 4 și calculele de mînă. Variantele '
       'care nu apar în curs (alte capete ale eșantionului, alte origini ale prognozei, alte penalizări ale criteriului, alte laguri, alți parametri ai problemelor) sînt '
       'rulări suplimentare ale codului capitolului pe aceleași date; graficele din „Interpretarea datelor”, „Pasul 2”, „Interpretarea corelogramei”, „Pasul 4”, '
       '„Interpretarea verificării reziduurilor”, „Interpretarea prognozelor” și din rezolvările problemelor sînt grafice suplimentare pentru slide-uri care dau doar cifre.'),
 'en': dict(file='ch15.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 15 · Course Review and Exam Preparation', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; '
        'the “Worked example” boxes redo the solutions of the exam problems with the numbers of the chapter.',
   src='The Romanian HICP (Eurostat prc_hicp_minr, 2015 = 100) is read with the chapter’s loader on 4 November 2026 and cut at August 2026, the last month of the lecture; '
       'the January 2005 – August 2026 values have not been revised and reproduce the numbers of the lecture exactly. The estimates of the 36 SARIMA models, their '
       'residuals and the forecasts come from the chapter’s code (statsmodels SARIMAX). Problem 2 uses Romanian real GDP in the version of the lecture (Chapter 3, read '
       'from the chapter’s vector chart because Eurostat has revised the series), Problem 4 the GDP growth, inflation and ROBOR 3M of the VAR of Chapter 6, Problem 5 the '
       'US Treasury yields (FRED, to September 2026), Problem 6 the GARCH(1,1)-t estimates of the BET from Chapter 5. The browser computes the correlograms, the '
       'Ljung–Box and Jarque–Bera tests, the accuracy measures, the Diebold–Mariano test, the OLS regressions of Problem 4 and the hand computations. Variants that do '
       'not appear in the lecture (other sample ends, other forecast origins, other penalties of the criterion, other lags, other parameters of the problems) are extra '
       'runs of the chapter’s code on the same data; the charts of “Interpreting the data”, “Step 2”, “Interpreting the correlogram”, “Step 4”, “Interpreting the '
       'residual checks”, “Interpreting the forecasts” and of the problem solutions are extra charts for slides that give only numbers.'),
}

CSS = ('<style>.c15x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c15x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c15x .c15f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c15x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c15x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c15x .wctl .ctl{max-width:360px;margin:.15rem 0 .2rem}'
       '.c15t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c15t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c15t th,.c15t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c15t thead tr:last-child th{border-bottom:1px solid #1A3A6E}'
       '.c15t tbody tr:nth-child(even){background:#F5F7FB}.c15t tbody tr.hi{background:#E3F4F7}.c15t .r{text-align:right}.c15t .c{text-align:center}'
       '.c15cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}</style>')


def tbl(i):
    return f'<div class="tbl c15t" id="{i}"></div>'


def wx(i, title, formula, ctl='', pre=''):
    """Worked-example box; `ctl` holds the controls that only drive the box (they stay out of the chart's controls bar)."""
    return f'<div class="c15x"><h4>{title}</h4>' + (f'<div class="wctl">{ctl}</div>' if ctl else '') + pre + f'<div class="c15f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def box(ctl, inner):
    """A box with controls that only drive the table and the chart of the block."""
    return f'<div class="c15x"><div class="wctl">{ctl}</div>{inner}</div>'


def slider_any(i, label, lo, hi, val):
    """A range input with step="any", so that the initial value is the exact estimate of the lecture."""
    return f'<div class="ctl"><label for="{i}">{label}: <b id="{i}-v"></b></label><input type="range" id="{i}" min="{lo}" max="{hi}" step="any" value="{val}"></div>'


def mod(m):
    return m + '<sub>12</sub>'


def widgets_all(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    W = s('Exemplu rezolvat', 'Worked example')
    c = lambda x: str(x).replace('.', ',') if ro else str(x)   # noqa: E731
    cf = lambda x: str(x).replace('.', '{,}') if ro else str(x)   # noqa: E731  (decimal comma in the formulas)
    my = lambda d: d  # noqa: E731
    MONR = ['ian.', 'febr.', 'mart.', 'apr.', 'mai', 'iun.', 'iul.', 'aug.', 'sept.', 'oct.', 'nov.', 'dec.']
    MONE = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ym = lambda k: f'{(MONR if ro else MONE)[int(k[5:7]) - 1]} {k[:4]}'   # noqa: E731
    del my
    sn = s('naiv sezonier', 'seasonal naive')
    return {
     'data': (slider('da-e', s('Ultima lună din grafic (august al anului; curs: august 2026)', 'Last month in the chart (August of the year; lecture: August 2026)'), 2010, 2026, 1, 2026)
              + play('da'), CSS + charts('da-chart')),
     'acf': (select('ac-s', s('Seria', 'Series'), [('m', 'Δ ln P'), ('a', 'Δ₁₂ ln P'), ('z', s('z = ΔΔ₁₂ ln P (curs)', 'z = ΔΔ₁₂ ln P (lecture)'))], 'z')
             + slider('ac-k', s('Numărul de laguri (curs: 36)', 'Number of lags (lecture: 36)'), 12, 48, 6, 36) + play('ac'), charts('ac-chart')),
     'diag': (select('dg-m', s('Modelul ale cărui reziduuri apar în grafic', 'Model whose residuals are shown'), [(m, m + '₁₂' + (s(' (ales, curs)', ' (chosen, lecture)') if m == CHOSEN else s(' (identificat)', ' (identified)') if m == TAB5[0] else '')) for m in TAB5], CHOSEN)
              + slider('dg-k', s('Laguri în ACF (curs: 24)', 'Lags in the ACF (lecture: 24)'), 12, 36, 6, 24) + play('dg'), charts('dg-chart')),
     'fc': (slider('fc-o', s('Originea prognozei (curs: august 2026)', 'Forecast origin (lecture: August 2026)'), 0, len(ORG) - 1, 1, len(ORG) - 1)
            + select('fc-b', s('Reperul din stînga', 'Benchmark on the left'), [('sn', s('naiv sezonier (curs)', 'seasonal naive (lecture)')), ('nv', s('naiv', 'naive'))], 'sn')
            + select('fc-l', s('Nivelul intervalului', 'Interval level'), [('80', '80%'), ('95', s('95% (curs)', '95% (lecture)')), ('99', '99%')], '95') + play('fc'), charts('fc-chart')),
     # ---------------------------------------------------------------- EXTRA
     't-data': (slider('td-s', s('Primul an al eșantionului pentru mediile lunare (curs: 2005)', 'First year of the sample for the monthly means (lecture: 2005)'), 2005, 2020, 1, 2005)
                + play('td'), charts('td-chart')),
     't-tests': (select('tt-e', s('Ultima lună a eșantionului (curs: august 2026)', 'Last month of the sample (lecture: August 2026)'),
                        [(k, ym(k) + (s(' (curs)', ' (lecture)') if k == TESTS[-1] else '')) for k in TESTS], TESTS[-1]) + play('tt'),
                 charts('tt-chart') + tbl('tt-tb')),
     't-acf': (slider_any('ta-p', s(f'φ al modelului (curs: estimarea {c(round(PH_ID, 3))})', f'φ of the model (lecture: the estimate {round(PH_ID, 3)})'), -0.9, 0.95, repr(PH_ID))
               + slider_any('ta-q', s(f'Θ al modelului (curs: estimarea {c(round(TH_ID, 3))})', f'Θ of the model (lecture: the estimate {round(TH_ID, 3)})'), -0.99, 0.5, repr(TH_ID)) + play('ta'),
               charts('ta-chart')),
     't-grid': (slider_any('tg-c', s(f'Penalizarea c pe parametru (curs: BIC, c = ln 223 = {c(round(LNT, 2))})', f'Penalty c per parameter (lecture: BIC, c = ln 223 = {round(LNT, 2)})'), 0, 10, repr(LNT))
                + play('tg'),
                charts('tg-chart') + box(select('tg-a', s('Nivelul testelor Ljung–Box pentru modelul ales', 'Level of the Ljung–Box tests for the chosen model'),
                                                [('0.01', '1%'), ('0.05', s('5% (curs)', '5% (lecture)')), ('0.1', '10%')], '0.05'), tbl('tg-tb'))),
     't-chosen': (select('tc-m', s('Modelul', 'Model'), [(m, m + '₁₂' + (s(' (ales, curs)', ' (chosen, lecture)') if m == CHOSEN else '')) for m in TAB5], CHOSEN) + play('tc'),
                  charts('tc-chart') + tbl('tc-tb')),
     't-diag': (slider('tx-m', s('Numărul de laguri m al testului Ljung–Box (curs: 12 și 24)', 'Number of lags m of the Ljung–Box test (lecture: 12 and 24)'), 6, 36, 1, 12) + play('tx'),
                charts('tx-chart') + tbl('tx-tb')),
     't-fc': (slider('tf-n', s('Lunile de test incluse, de la septembrie 2024 (curs: 24)', 'Test months included, from September 2024 (lecture: 24)'), 6, 24, 1, 24)
              + select('tf-l', s('Pierderea din testul Diebold–Mariano', 'Loss in the Diebold–Mariano test'), [('sq', s('erori pătratice (curs)', 'squared errors (lecture)')), ('abs', s('erori absolute', 'absolute errors'))], 'sq')
              + play('tf'), charts('tf-chart') + tbl('tf-tb')),
     'p1': (slider('q1-f', s('Coeficientul φ al erorilor AR(1) (curs: 0,6)', 'Coefficient φ of the AR(1) errors (lecture: 0.6)'), -0.9, 0.95, 0.05, 0.6) + play('q1'),
            charts('q1-chart') + wx('q1-ws', W, rf'\begin{{gathered}}Y_t = 10 + {cf(0.5)}\,t + u_t,\quad u_t = \phi u_{{t-1}} + \varepsilon_t\\ \hat Y_{{T+h}} = 10 + {cf(0.5)}\,(T+h) + \phi^h u_T,\quad \mathrm{{Var}}(e_{{T+h}}) = \frac{{1-\phi^{{2h}}}}{{1-\phi^2}}\end{{gathered}}',
                                    slider('q1-y', s('Valoarea observată Y<sub>T</sub> (curs: 62)', 'Observed value Y<sub>T</sub> (lecture: 62)'), 58, 66, 0.5, 62))),
     'p2': (select('q2-e', s('Ultimul trimestru al eșantionului (curs: 2026Q2)', 'Last quarter of the sample (lecture: 2026Q2)'), [(q, q + (s(' (curs)', ' (lecture)') if q == P2E[-1] else '')) for q in P2E], P2E[-1])
            + play('q2'), charts('q2-chart') + tbl('q2-tb') + wx('q2-ws', W, r'\Delta y_t = c + bt + \gamma y_{t-1} + \sum_j \delta_j \Delta y_{t-j} + \varepsilon_t,\quad H_0:\ \gamma = 0;\qquad Q(m) = T(T+2)\sum_{k=1}^{m}\frac{\hat\rho_k^2}{T-k}')),
     'p3': (slider('q3-o', s('Originea prognozei (curs: august 2026; celelalte: rulări suplimentare)', 'Forecast origin (lecture: August 2026; the others: extra runs)'), 0, 23, 1, 23) + play('q3'),
            charts('q3-chart') + wx('q3-ws', W, r'z_t = \phi z_{t-1} + \Phi z_{t-12} - \phi\Phi z_{t-13} + \varepsilon_t,\qquad \Delta_{12}\ln P_{T+1} = \Delta_{12}\ln P_T + \hat z_{T+1}',
                                    slider_any('q3-p', s(f'φ (curs: {c(round(PH3, 3))})', f'φ (lecture: {round(PH3, 3)})'), -0.9, 0.9, repr(PH3))
                                    + slider_any('q3-q', s(f'Φ (curs: {c(round(PPH3, 3))})', f'Φ (lecture: {round(PPH3, 3)})'), -0.95, 0.5, repr(PPH3)))),
     'p4': (select('q4-e', s('Ultimul trimestru al eșantionului (curs: 2026Q2)', 'Last quarter of the sample (lecture: 2026Q2)'), [(q, q + (s(' (curs)', ' (lecture)') if q == P4E[-1] else '')) for q in P4E], P4E[-1])
            + play('q4'), charts('q4-chart') + tbl('q4-tb') + tbl('q4-tb2')
            + wx('q4-ws', W, r'\frac{\partial i}{\partial \pi} = \frac{b_{\pi,1} + b_{\pi,2}}{1 - b_{i,1} - b_{i,2}},\qquad F = \frac{(RSS_R - RSS_U)/p}{RSS_U/(T - Kp - 1)}')),
     'p5': (slider('q5-k', s('Multiplicatorul coeficienților de ajustare α (curs: 1)', 'Multiplier of the adjustment coefficients α (lecture: 1)'), 0.25, 2, 0.05, 1) + play('q5'),
            charts('q5-chart') + wx('q5-ws', W, r'\Delta \mathbf{y}_t = \boldsymbol\alpha\,\boldsymbol\beta^\top \mathbf{y}_{t-1} + \dots,\qquad \mathbf{ec}_t \approx (\mathbf{I} + \boldsymbol\beta^\top\boldsymbol\alpha)\,\mathbf{ec}_{t-1},\qquad h_{1/2} = \frac{\ln 0{.}5}{\ln|\lambda|}'.replace('0{.}5', cf(0.5)),
                                    select('q5-y', s('Ecuația scrisă la pasul 2 (curs: Δy<sup>(10)</sup>)', 'Equation written in step 2 (lecture: Δy<sup>(10)</sup>)'),
                                           [('0', 'Δy<sup>(1)</sup>'), ('1', 'Δy<sup>(5)</sup>'), ('2', 'Δy<sup>(10)</sup>')], '2'))),
     'p6': (slider('q6-e', s('Șocul de azi ε<sub>t</sub> (curs: −3,0)', 'Today’s shock ε<sub>t</sub> (lecture: −3.0)'), -6, 6, 0.5, -3)
            + slider('q6-s', s('Varianța de azi σ<sub>t</sub><sup>2</sup> (curs: 1,0)', 'Today’s variance σ<sub>t</sub><sup>2</sup> (lecture: 1.0)'), 0.5, 9, 0.5, 1) + play('q6'),
            charts('q6-chart') + wx('q6-ws', W, r'\begin{gathered}\sigma^2_{t+1} = \omega + \alpha\varepsilon_t^2 + \beta\sigma_t^2,\qquad \bar\sigma^2 = \frac{\omega}{1-\alpha-\beta}\\ \mathrm{VaR}_{1\%} = -\left(\mu + \sigma_{t+1}\, t_\nu^{-1}(0{.}01)\sqrt{(\nu-2)/\nu}\right)\end{gathered}'.replace('0{.}01', cf(0.01)),
                                    slider('q6-n', s('Gradele de libertate ν ale cuantilei (curs: 5, estimarea 5,011 rotunjită)', 'Degrees of freedom ν of the quantile (lecture: 5, the estimate 5.011 rounded)'), 3, 30, 1, 5))),
     'p7': (slider('q7-l', s('Factorul care înmulțește erorile metodei naive sezoniere (curs: 1)', 'Factor that multiplies the errors of the seasonal naive method (lecture: 1)'), 0.5, 2, 0.05, 1) + play('q7'),
            charts('q7-chart') + wx('q7-ws', W, r'\begin{gathered}\mathrm{RMSE} = \sqrt{\tfrac1n\sum e_t^2},\quad \mathrm{MASE} = \frac{\mathrm{MAE}}{\mathrm{MAE}_{\text{in}}}\\ d_t = e_{A,t}^2 - e_{B,t}^2,\quad DM = \frac{\bar d}{s_d/\sqrt n}\end{gathered}',
                                    slider('q7-s', s('MAE în eșantion al metodei naive sezoniere (curs: 0,6)', 'In-sample MAE of the seasonal naive method (lecture: 0.6)'), 0.3, 1.2, 0.05, 0.6))),
     'p8': (slider('q8-q', s('Varianța nivelului σ<sub>η</sub><sup>2</sup>, cu σ<sub>ε</sub><sup>2</sup> = 4 (curs: 1)', 'Level variance σ<sub>η</sub><sup>2</sup>, with σ<sub>ε</sub><sup>2</sup> = 4 (lecture: 1)'), 0.1, 8, 0.05, 1) + play('q8'),
            charts('q8-chart') + wx('q8-ws', W, r'\begin{gathered}K_t = \frac{P_t}{P_t + \sigma^2_\varepsilon},\quad \frac{\bar P}{\sigma^2_\varepsilon} = \frac{q + \sqrt{q^2 + 4q}}{2}\\ E(D_i) = \frac{1}{1 - p_{ii}},\quad \pi_1 = \frac{1 - p_{22}}{2 - p_{11} - p_{22}},\quad H = d + \tfrac12\end{gathered}',
                                    slider('q8-y', s('Observația y<sub>t</sub> (curs: 13)', 'Observation y<sub>t</sub> (lecture: 13)'), 6, 16, 0.5, 13)
                                    + slider_any('q8-a', s(f'p<sub>11</sub> (curs: {c(round(P11, 3))})', f'p<sub>11</sub> (lecture: {round(P11, 3)})'), 0.5, 0.99, repr(P11))
                                    + slider_any('q8-b', s(f'p<sub>22</sub> (curs: {c(round(P22, 3))})', f'p<sub>22</sub> (lecture: {round(P22, 3)})'), 0.5, 0.99, repr(P22))
                                    + slider_any('q8-d', s(f'd al modelului ARFIMA (curs: {c(round(D8, 2))})', f'd of the ARFIMA model (lecture: {round(D8, 2)})'), -0.45, 0.95, repr(D8)))),
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
    """Chart blocks for build_web_lecture.py: each host carries the web version of its EXTRA blocks (moved by ch15.js)."""
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
        img = f'ch15x_{anchor}_{lang}.jpg'
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
const MONE = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const de = n => { const r = Math.abs(Math.round(n)) % 100; return Number.isInteger(+n) && ((r === 0 && +n !== 0) || r >= 20) ? ' de' : ''; };
const mod = m => m + '<sub>12</sub>';
const rej = b => (b ? 'se respinge' : 'nu se respinge');
window.CH15_TEXT = {
 plt: '&lt; 0,001',
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 myEn: d => `${MONE[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 mon: MON,
 months: n => `${n}${de(n)} ${n === 1 ? 'lună' : 'luni'}`,
 mod, sub: s => s.replace(/12$/, '<sub>12</sub>'), identTag: ', identificat',
 sn: 'naiv sezonier', nv: 'naiv', const: 'constanta', c: 'const.', ct: 'const., trend',
 extra: 'Rulare suplimentară a codului capitolului pe aceleași date; cifrele din curs corespund valorilor marcate „curs”.',
 extraP: 'Variantă a problemei (rulare suplimentară); rezolvarea din curs corespunde valorilor marcate „curs”.',
 da1: (n, e, idx, mm) => `Ianuarie 2005 – ${e}: ${n}${de(n)} luni; indicele ajunge la ${idx} (2015 = 100); inflația lunară are media ${mm}%.`,
 da2: (mn, dmn, mx, dmx, last) => `Inflația anuală variază între ${mn}% (${dmn}) și ${mx}% (${dmx}); ultima valoare: ${last}%.`,
 td1: (y, jan, jun, hm, hv, lm, lv) => `Media inflației lunare, ${y}–2026: ianuarie ${jan}%, iunie ${jun}%; cea mai mare în ${hm} (${hv}%), cea mai mică în ${lm} (${lv}%).`,
 td2: (a7, a8, m8) => `Șocul din iulie–august 2025 (încetarea plafonării prețului la electricitate și creșterea TVA): inflația anuală urcă de la ${a7}% la ${a8}%, inflația lunară din august 2025 este ${m8}%.`,
 serT: { y: '100 ln P<sub>t</sub>', m: 'Δ ln P<sub>t</sub>', a: 'Δ<sub>12</sub> ln P<sub>t</sub>', z: 'z<sub>t</sub> = ΔΔ<sub>12</sub> ln P<sub>t</sub>' },
 verd: { ur: 'rădăcină unitară', sts: 'staționară, sezonieră', st: 'staționară', inc: 'neconcludent', conf: 'contradictoriu: ambele resping' },
 ttHead: ['Seria', 'Termeni', 'ADF τ', 'p', 'KPSS η (5%: val. critică)', 'Verdict'],
 ttNote: (n, cvt, cvc) => `ADF: H<sub>0</sub> rădăcină unitară, laguri după AIC; KPSS: H<sub>0</sub> staționaritate. Valorile critice ADF la 5%: ${cvt} (cu trend), ${cvc} (cu constantă); z are ${n}${de(n)} observații.`,
 tt1: (e, v, aa, ka, az, kz) => `Eșantion pînă în ${e}: inflația anuală Δ<sub>12</sub> ln P are verdictul „${v}” (ADF ${aa}, KPSS ${ka}); z: ADF ${az}, KPSS ${kz}.`,
 ac1: (nm, n, bd, r1, r2, r12, r24) => `${nm}: T = ${n}, banda ±${bd}; ρ̂<sub>1</sub> = ${r1}, ρ̂<sub>2</sub> = ${r2}, ρ̂<sub>12</sub> = ${r12}, ρ̂<sub>24</sub> = ${r24}.`,
 ac2: (p1, p12, p24, p36) => `PACF: φ̂<sub>11</sub> = ${p1}; la lagurile 12, 24, 36: ${p12}, ${p24}, ${p36}.`,
 ac3: l => `Lagurile ACF în afara benzii: ${l || 'niciunul'}.`,
 ta1: (r1, r12, pr, r11, r13) => `Eșantionul: ρ̂<sub>1</sub> = ${r1}, ρ̂<sub>12</sub> = ${r12}, produsul ρ̂<sub>1</sub>ρ̂<sub>12</sub> = ${pr}; sateliții ρ̂<sub>11</sub> = ${r11}, ρ̂<sub>13</sub> = ${r13}.`,
 ta2: (ph, th, m1, m12, m11, m13) => `Modelul (1 ${ph}L)z<sub>t</sub> = (1 ${th}L<sup>12</sup>)ε<sub>t</sub>: ρ<sub>1</sub> = ${m1}, ρ<sub>12</sub> = ${m12}, ρ<sub>11</sub> = ${m11}, ρ<sub>13</sub> = ${m13}.`,
 tgHead: c => ['Modelul', 'param.', 'AICc', 'BIC', 'LB(12) p', 'LB(24) p', `−2 ln L + ${c}·k`],
 tgNote: (n, ne, ln_, al) => `Toate cele 36 de modele SARIMA(p,1,q)(P,1,Q)<sub>12</sub> cu p, q ≤ 2, P, Q ≤ 1, pe eșantionul de antrenare pînă în august 2024 (T = ${n}, ${ne}${de(ne)} observații după diferențiere); primul rînd: modelul identificat, apoi celelalte patru cu cel mai mic criteriu. BIC = −2 ln L + k ln ${ne}, cu ln ${ne} = ${ln_}; Ljung–Box cu m − k grade de libertate. Rîndul colorat: modelul ales (cel mai mic criteriu dintre modelele care trec ambele teste la ${al}).`,
 tg1: (c, b, ch, al) => `Cu c = ${c}: cel mai mic criteriu îl are ${b}; modelul ales (trece Ljung–Box la 12 și 24, nivelul ${al}) este ${ch}.`,
 tg2: (ln_, ne) => `c = 2 este AIC fără corecție; c = ln ${ne} = ${ln_} este BIC: penalizarea crește cu lungimea eșantionului și favorizează modelele mai mici.`,
 tcHead: ['Parametrul', 'Estimarea', 'SE', 'z', 'p'],
 tcNote: m => `SARIMA${m} pe eșantionul de antrenare; (1 − φ<sub>1</sub>L − φ<sub>2</sub>L<sup>2</sup>)(1 − Φ<sub>1</sub>L<sup>12</sup>) z<sub>t</sub> = (1 + θ<sub>1</sub>L + θ<sub>2</sub>L<sup>2</sup>)(1 + Θ<sub>1</sub>L<sup>12</sup>) ε<sub>t</sub>, z<sub>t</sub> = ΔΔ<sub>12</sub> 100 ln P<sub>t</sub>.`,
 tc1: (m, a, b) => `${m}: cea mai mică rădăcină AR are modulul ${a}, cea mai mică rădăcină MA ${b} (în afara cercului unitate: staționar și invertibil).`,
 tc2: (a11, a12, a24, up) => `Un șoc de 1 pp în inflația lunară ridică inflația anuală cu ${a11} pp după 11 luni; după 12 luni efectul este ${a12} pp, după 24 de luni ${a24} pp.${up ? ' Efectul trece de 1 pp pentru că rădăcina AR aproape de 1 menține ridicate și lunile următoare;' : ''} Θ<sub>1</sub> aproape de −1 scoate șocul inițial din fereastra de 12 luni.`,
 dg1: (m, q12, d12, p12, q24, d24, p24) => `${m}: Ljung–Box Q(12) = ${q12} (${d12} grade de libertate, p = ${p12}), Q(24) = ${q24} (${d24} grade de libertate, p = ${p24}).`,
 dg2: (p12, p24) => `Modelul identificat (1,1,0)(0,1,1)<sub>12</sub>: p = ${p12} și ${p24}.`,
 dg3: (jb, k, d1, v1, d2, v2) => `Jarque–Bera ${jb}, kurtosis-ul ${k}; cele mai mari reziduuri: ${d1} (${v1}σ) și ${d2} (${v2}σ).`,
 txHead: m => ['Modelul', `Q(${m}) (gl)`, `p, m = ${m}`, 'LB(12) p', 'LB(24) p', 'Jarque–Bera', 'Kurtosis', 'Cel mai mare reziduu'],
 txNote: 'Reziduurile celor cinci modele din tabelul pasului 4, eșantionul de antrenare; Ljung–Box cu m − k grade de libertate, k = parametrii ARMA. Rîndul colorat: modelul ales.',
 tx1: (m, ch, q, df, p, pi) => `m = ${m}: modelul ales ${ch} are Q = ${q} (${df} grade de libertate, p = ${p}); modelul identificat are p = ${pi}.`,
 fc1: (o, aT, m1, a1, l1, h1, m12, a12, l12, h12, lv) => `Originea ${o} (inflația anuală ${aT}%): prognoza pentru ${m1} ${a1}% [${l1}; ${h1}], pentru ${m12} ${a12}% [${l12}; ${h12}], intervale de ${lv}%.`,
 fc2: (n, rs, rn, rv) => `Pe ${n}${de(n)} luni de test: RMSE ${rs} (SARIMA), ${rn} (naiv sezonier), ${rv} (naiv).`,
 tfHead: ['Inflația lunară, eșantion de test', 'RMSE', 'MAE', 'MASE'],
 tfNote: (n, sc) => `Primele ${n}${de(n)} luni de test (din septembrie 2024); MASE = MAE / ${sc}, MAE în eșantion al metodei naive sezoniere.`,
 tf1: (abs, hs, ps, hn, pn) => `Diebold–Mariano (HLN, erori ${abs ? 'absolute' : 'pătratice'}): ${hs} (p = ${ps}) față de naivul sezonier, ${hn} (p = ${pn}) față de metoda naivă.`,
 tf2: (n, d1, a1, f1, d2, a2, f2) => `Cele mai mari erori în primele ${n}${de(n)} luni: ${d1} (${a1}%, față de ${f1}%) și ${d2} (${a2}%, față de ${f2}%).`,
 q1s1: (ph, ph2, v) => `E(Y<sub>t</sub>) = 10 + 0,5t depinde de t: nestaționar; Var(Y<sub>t</sub>) = 1/(1 − ${ph2}) = ${v} este constantă (φ = ${ph}): staționar în jurul trendului.`,
 q1s2: (ph, g0, g1, r1) => `ΔY<sub>t</sub> = 0,5 + Δu<sub>t</sub> și (1 − ${ph}L)Δu<sub>t</sub> = (1 − L)ε<sub>t</sub>: ARMA(1,1), θ = −1; γ(0) = ${g0}, γ(1) = ${g1}, ρ(1) = ${r1}.`,
 q1s3: (y, u, ph, f1, f10, v) => `u<sub>T</sub> = ${y} − 10 − 50 = ${u}; Ŷ<sub>T+1</sub> = 10 + 50,5 + ${ph} · ${u} = ${f1}; Ŷ<sub>T+10</sub> = 10 + 55 + ${ph}<sup>10</sup> · ${u} = ${f10}; varianța pe 10 pași ${v}.`,
 q1s4: 'Rădăcina MA este 1: ΔY<sub>t</sub> nu este invertibil; ρ̂<sub>1</sub> &lt; 0 și un coeficient MA aproape de −1 sînt semnele supradiferențierii.',
 q1o: (ph, p10, v) => `Cu φ = ${ph}, după 10 perioade rămîne ${p10} din abatere; varianța erorii crește spre ${v} și rămîne mărginită.`,
 q2Head: ['Seria', 'Termeni', 'ADF τ', 'p', 'val. critice 1%; 5%; 10%', 'KPSS', 'val. critică 5%'],
 q2s1: (al, cv, ra, ag, cvg, rg) => `ADF, H<sub>0</sub>: γ = 0 (rădăcină unitară). ln Y<sub>t</sub>: ${al} ${ra ? '&lt;' : '&gt;'} ${cv}: ${rej(ra)}; g<sub>t</sub>: ${ag} ${rg ? '&lt;' : '&gt;'} ${cvg}: ${rej(rg)}.`,
 q2s2: (kl, kc, rl, kg, kcg, rg) => `KPSS (H<sub>0</sub>: staționaritate): nivel ${kl} ${rl ? '&gt;' : '&lt;'} ${kc}: ${rej(rl)}; creștere ${kg} ${rg ? '&gt;' : '&lt;'} ${kcg}: ${rej(rg)}${rl && !rg ? ': ln Y<sub>t</sub> ~ I(1)' : ''}.`,
 q2s3: (r1, r2, p1, p2, bd, q, c8, p, ok) => `ACF ${r1}; ${r2}, PACF ${p1}; ${p2} (banda ±${bd}); Q(8) = ${q} ${ok ? '&lt;' : '&gt;'} χ<sup>2</sup><sub>0,95</sub>(8) = ${c8} (p = ${p}): ${ok ? 'creșterea este zgomot alb în jurul mediei' : 'rămîne autocorelație'}.`,
 q2s4: (c, se, s, t, a) => `ARIMA(0,1,0) cu derivă: 100 ln Y<sub>t</sub> = 100 ln Y<sub>t−1</sub> + ${c} + ε<sub>t</sub>, σ̂ = ${s}; t = ${c}/${se} = ${t}; aproximativ ${a}% pe an.`,
 q2o: (e, n, m, s) => `2000Q2–${e}: ${n} trimestre de creștere, media ${m}% pe trimestru, σ̂ = ${s}.`,
 q3s1: '(1 − φL)(1 − ΦL<sup>12</sup>)z<sub>t</sub> = ε<sub>t</sub>, deci z<sub>t</sub> = φz<sub>t−1</sub> + Φz<sub>t−12</sub> − φΦz<sub>t−13</sub> + ε<sub>t</sub>.',
 q3s2: (ph, Ph, i1, i12, st) => `φ̂ = ${ph}: o surpriză lunară continuă și luna următoare; Φ̂ = ${Ph}: o surpriză se inversează parțial în aceeași lună a anului următor; rădăcinile AR inversate ${i1} și ${i12}${st ? ' &lt; 1: staționar' : ': nu sînt ambele sub 1'}.`,
 q3s3: (q, df, p) => `Ljung–Box Q(12) = ${q} cu ${df} grade de libertate, p = ${p} (estimările din curs): ipoteza de zgomot alb se respinge la 5%.`,
 q3s4: (ph, zT, Ph, z11, pP, z12, t1, t2, t3, zn, mN, mT, m11, m12) => `ẑ (${mN}) = ${ph} · ${zT} + ${Ph} · ${z11} − ${pP} · ${z12} = ${t1} + ${t2} + ${t3} = ${zn} (z din ${mT}, ${m11}, ${m12}).`,
 q3s5: (aT, zn, an, mN) => `Inflația anuală în ${mN}: Δ<sub>12</sub> ln P<sub>T+1</sub> = ${aT} + ${zn} = ${an}%.`,
 q3o: (mN, an, aa, za) => `Prognoza de mînă pentru ${mN}: ${an}%${aa === null ? '.' : `; valoarea realizată: ${aa}% (z = ${za}).`}`,
 q4Head: ['Ecuația ROBOR', 'coef. [t]', '', 'coef. [t]'],
 q4Head2: ['Ipoteza nulă', 'F', 'p'],
 q4h: (x, y) => `${x} nu cauzează Granger ${y}`,
 q4s1: (pg, pp, pi, hg, hp, hi) => `x cauzează Granger y dacă lagurile lui x îmbunătățesc prognoza lui y, dată fiind istoria lui y (test F comun): creșterea ${hg ? 'ajută' : 'nu ajută'} la prognoza ROBOR (p ${pg}), inflația ${hp ? 'ajută' : 'nu ajută'} (p ${pp}); ROBOR ${hi ? 'ajută' : 'nu ajută'} la prognoza inflației (p ${pi}).`,
 q4s2: (c, g1, g2, p1, p2, i1, i2) => `i<sub>t</sub> = ${c} + ${g1}g<sub>t−1</sub> + ${g2}g<sub>t−2</sub> + ${p1}π<sub>t−1</sub> + ${p2}π<sub>t−2</sub> + ${i1}i<sub>t−1</sub> + ${i2}i<sub>t−2</sub> + u<sub>t</sub>.`,
 q4s3: (b1, b5, b3, b6, num, den, r) => `Pe termen lung (i<sub>t</sub> = i<sub>t−1</sub>, π<sub>t</sub> = π<sub>t−1</sub>): ∂i/∂π = (${b1} + ${b5})/(1 − ${b3} − ${b6}) = ${num}/${den} = ${r} pp.`,
 q4s4: (rr, ru, df, F, cr, rj, Fr, dif) => `F = ((${rr} − ${ru})/2)/(${ru}/${df}) = ${F} ${rj ? '&gt;' : '&lt;'} F<sub>0,95</sub>(2, ${df}) = ${cr}: H<sub>0</sub> ${rj ? 'se respinge' : 'nu se respinge'}${dif ? ` (cu RSS rotunjite la trei zecimale se obține ${Fr}; diferența vine doar din rotunjire)` : ''}.`,
 q4o: (e, n, lr, den) => `Eșantion pînă în ${e} (T = ${n}): efectul pe termen lung ${lr} pp; numitorul 1 − b<sub>i,1</sub> − b<sub>i,2</sub> = ${den}, aproape de zero, face efectul imprecis.`,
 q5s1: 'Seriile I(1) sînt cointegrate dacă o combinație liniară a lor este I(0); urma: 54,60 &gt; 29,80, 21,57 &gt; 15,49, 3,12 &lt; 3,84: rangul r = 2, un singur trend stochastic comun.',
 q5s2: (nm, a1, a2, t1, t2, u1, u2) => `Δ${nm}<sub>t</sub> = ${a1} EC1<sub>t−1</sub> + ${a2} EC2<sub>t−1</sub> + … (t din curs: ${t1} și ${t2}); randamentul la 1 an are |t| &lt; 1 pe ambele relații (${u1}, ${u2}): dobînda scurtă nu se ajustează (exogenitate slabă).`,
 q5s3: (b1, b2, e1, e2, h1, h2) => `β ≈ 1 (${b1} și ${b2}): spread-urile y<sup>(1)</sup> − y<sup>(10)</sup> și y<sup>(5)</sup> − y<sup>(10)</sup> sînt staționare; valorile proprii ale lui I + β′α: ${e1} și ${e2}; timpii de înjumătățire ai erorilor de echilibru: ${h1} și ${h2} luni.`,
 q5o: (k, h1, h2, e1, e2) => `Cu α înmulțit cu ${k}: timpii de înjumătățire sînt ${h1} și ${h2} luni; în septembrie 2026 EC1 = ${e1}, EC2 = ${e2}.`,
 q6s1: (ps, hl, om, lv, an) => `α + β = ${ps}; h<sub>1/2</sub> = ln 0,5/ln ${ps} = ${hl}${de(+hl)} zile; σ̄<sup>2</sup> = ${om}/(1 − ${ps}) = ${lv}, √(250 × ${lv}) = ${an}% pe an.`,
 q6s2: (om, al, e2, be, s2, t1, t2, s2n, sn) => `σ<sup>2</sup><sub>t+1</sub> = ${om} + ${al} × ${e2} + ${be} × ${s2} = ${om} + ${t1} + ${t2} = ${s2n}, σ<sub>t+1</sub> = ${sn}%.`,
 q6s3: (nu, tq, fr, sc, q, mu, sn, v, vn) => `q<sub>0,01</sub>(z) = ${tq} × √(${fr}) = ${tq} × ${sc} = ${q} (t<sub>${nu}</sub>); VaR 1% = −(${mu} + ${sn} × (${q})) = ${v}%; Normal: ${vn}%.`,
 q6o: (v, vn, vs) => `VaR 1% pe o zi: ${v}% cu cuantila t standardizată, ${vn}% cu distribuția Normală; volatilitatea de selecție a BET: ${vs}% pe an.`,
 q7s1: (rA, mA, qA, rB, mB, qB, sc) => `SARIMA: RMSE ${rA}, MAE ${mA}, MASE ${qA}; naiv sezonier: RMSE ${rB}, MAE ${mB}, MASE ${qB} (MAE în eșantion ${sc}).`,
 q7s2: (dl, db, sd, dm, cr, rj) => `d<sub>t</sub>: ${dl}; d̄ = ${db}, s<sub>d</sub> = ${sd}; DM = ${dm}, |${dm}| ${rj ? '&gt;' : '&lt;'} ${cr}: acuratețea egală ${rj ? 'se respinge' : 'nu se respinge'}.`,
 q7s3: (kf, wf) => `Grupurile aleatoare pun zile din viitor în setul de antrenare: leakage; validarea walk-forward dă R<sup>2</sup> = ${wf} pentru același model, față de ${kf} raportat.`,
 q7o: (l, dm, ls) => `Cu erorile naivului sezonier înmulțite cu ${l}: DM = ${dm}; acuratețea egală se respinge la 5% abia de la un factor de aproximativ ${ls}.`,
 q8s1: (K, y, v, af, Pf, s2n, Pn) => `F<sub>t</sub> = 2 + 4 = 6, K<sub>t</sub> = 2/6 = ${K}; a<sub>t|t</sub> = 10 + ${K} × ${v} = ${af}, P<sub>t|t</sub> = 2(1 − ${K}) = ${Pf}; a<sub>t+1</sub> = ${af}, P<sub>t+1</sub> = ${Pf} + ${s2n} = ${Pn} (y<sub>t</sub> = ${y}).`,
 q8s2: (q, Pb, Kb) => `q = σ<sub>η</sub><sup>2</sup>/σ<sub>ε</sub><sup>2</sup> = ${q}: P̄/σ<sub>ε</sub><sup>2</sup> = (q + √(q<sup>2</sup> + 4q))/2 = ${Pb}, K̄ = ${Pb}/(1 + ${Pb}) = ${Kb} = α<sub>SES</sub>.`,
 q8s3: (p11, p22, d1, d2, nu, dn, pi) => `E(D<sub>1</sub>) = 1/(1 − ${p11}) = ${d1} trimestre, E(D<sub>2</sub>) = 1/(1 − ${p22}) = ${d2} trimestre; π<sub>1</sub> = ${nu}/${dn} = ${pi}.`,
 q8s4: (d, H, r1, k) => `H = d + 0,5 = ${H}; ρ(1) = d/(1 − d) = ${r1}; ${k === 'st' ? '0 &lt; d &lt; 0,5: staționar cu memorie lungă, cu revenire la medie' : k === 'nst' ? '0,5 ≤ d &lt; 1: nestaționar, dar cu revenire la medie' : k === 'anti' ? '−0,5 &lt; d ≤ 0: staționar, cu memorie intermediară (antipersistent)' : 'în afara intervalului −0,5 &lt; d &lt; 1'} (d = ${d}).`,
 q8o: (s, K, p) => `Cu σ<sub>η</sub><sup>2</sup> = ${s}, cîștigul de echilibru este ${K}; regimul volatil are o pondere pe termen lung de ${p}%.`,
};
})();
""",
 'en': r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const mod = m => m + '<sub>12</sub>';
const rej = b => (b ? 'rejected' : 'not rejected');
window.CH15_TEXT = {
 plt: '&lt; 0.001',
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 myEn: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 mon: MON,
 months: n => `${n} ${n === 1 ? 'month' : 'months'}`,
 mod, sub: s => s.replace(/12$/, '<sub>12</sub>'), identTag: ', identified',
 sn: 'seasonal naive', nv: 'naive', const: 'constant', c: 'const.', ct: 'const., trend',
 extra: 'Extra run of the chapter’s code on the same data; the numbers of the lecture correspond to the values marked “lecture”.',
 extraP: 'A variant of the problem (extra run); the lecture’s solution corresponds to the values marked “lecture”.',
 da1: (n, e, idx, mm) => `January 2005 – ${e}: ${n} months; the index reaches ${idx} (2015 = 100); monthly inflation averages ${mm}%.`,
 da2: (mn, dmn, mx, dmx, last) => `Annual inflation ranges from ${mn}% (${dmn}) to ${mx}% (${dmx}); last value: ${last}%.`,
 td1: (y, jan, jun, hm, hv, lm, lv) => `Mean monthly inflation, ${y}–2026: January ${jan}%, June ${jun}%; highest in ${hm} (${hv}%), lowest in ${lm} (${lv}%).`,
 td2: (a7, a8, m8) => `The July–August 2025 shock (end of the electricity price caps and a higher VAT): annual inflation rises from ${a7}% to ${a8}%, monthly inflation in August 2025 is ${m8}%.`,
 serT: { y: '100 ln P<sub>t</sub>', m: 'Δ ln P<sub>t</sub>', a: 'Δ<sub>12</sub> ln P<sub>t</sub>', z: 'z<sub>t</sub> = ΔΔ<sub>12</sub> ln P<sub>t</sub>' },
 verd: { ur: 'unit root', sts: 'stationary, seasonal', st: 'stationary', inc: 'inconclusive', conf: 'conflicting: both reject' },
 ttHead: ['Series', 'Terms', 'ADF τ', 'p', 'KPSS η (5%: crit.)', 'Verdict'],
 ttNote: (n, cvt, cvc) => `ADF: H<sub>0</sub> unit root, lags by AIC; KPSS: H<sub>0</sub> stationarity. 5% ADF critical values: ${cvt} (with trend), ${cvc} (constant); z has ${n} observations.`,
 tt1: (e, v, aa, ka, az, kz) => `Sample to ${e}: annual inflation Δ<sub>12</sub> ln P gets the verdict “${v}” (ADF ${aa}, KPSS ${ka}); z: ADF ${az}, KPSS ${kz}.`,
 ac1: (nm, n, bd, r1, r2, r12, r24) => `${nm}: T = ${n}, band ±${bd}; ρ̂<sub>1</sub> = ${r1}, ρ̂<sub>2</sub> = ${r2}, ρ̂<sub>12</sub> = ${r12}, ρ̂<sub>24</sub> = ${r24}.`,
 ac2: (p1, p12, p24, p36) => `PACF: φ̂<sub>11</sub> = ${p1}; at lags 12, 24, 36: ${p12}, ${p24}, ${p36}.`,
 ac3: l => `ACF lags outside the band: ${l || 'none'}.`,
 ta1: (r1, r12, pr, r11, r13) => `Sample: ρ̂<sub>1</sub> = ${r1}, ρ̂<sub>12</sub> = ${r12}, the product ρ̂<sub>1</sub>ρ̂<sub>12</sub> = ${pr}; the satellites ρ̂<sub>11</sub> = ${r11}, ρ̂<sub>13</sub> = ${r13}.`,
 ta2: (ph, th, m1, m12, m11, m13) => `Model (1 ${ph}L)z<sub>t</sub> = (1 ${th}L<sup>12</sup>)ε<sub>t</sub>: ρ<sub>1</sub> = ${m1}, ρ<sub>12</sub> = ${m12}, ρ<sub>11</sub> = ${m11}, ρ<sub>13</sub> = ${m13}.`,
 tgHead: c => ['Model', 'param.', 'AICc', 'BIC', 'LB(12) p', 'LB(24) p', `−2 ln L + ${c}·k`],
 tgNote: (n, ne, ln_, al) => `All 36 SARIMA(p,1,q)(P,1,Q)<sub>12</sub> with p, q ≤ 2, P, Q ≤ 1, on the training sample to August 2024 (T = ${n}, ${ne} observations after differencing); first row: the identified model, then the four others with the lowest criterion. BIC = −2 ln L + k ln ${ne}, with ln ${ne} = ${ln_}; Ljung–Box with m − k degrees of freedom. Highlighted row: the chosen model (the lowest criterion among the models that pass both tests at ${al}).`,
 tg1: (c, b, ch, al) => `With c = ${c}: the lowest criterion belongs to ${b}; the chosen model (passes Ljung–Box at 12 and 24, level ${al}) is ${ch}.`,
 tg2: (ln_, ne) => `c = 2 is AIC without the correction; c = ln ${ne} = ${ln_} is BIC: the penalty grows with the sample length and favours smaller models.`,
 tcHead: ['Parameter', 'Estimate', 'SE', 'z', 'p'],
 tcNote: m => `SARIMA${m} on the training sample; (1 − φ<sub>1</sub>L − φ<sub>2</sub>L<sup>2</sup>)(1 − Φ<sub>1</sub>L<sup>12</sup>) z<sub>t</sub> = (1 + θ<sub>1</sub>L + θ<sub>2</sub>L<sup>2</sup>)(1 + Θ<sub>1</sub>L<sup>12</sup>) ε<sub>t</sub>, z<sub>t</sub> = ΔΔ<sub>12</sub> 100 ln P<sub>t</sub>.`,
 tc1: (m, a, b) => `${m}: the smallest AR root has modulus ${a}, the smallest MA root ${b} (outside the unit circle: stationary and invertible).`,
 tc2: (a11, a12, a24, up) => `A 1 pp shock to monthly inflation raises annual inflation by ${a11} pp after 11 months; after 12 months the effect is ${a12} pp, after 24 months ${a24} pp.${up ? ' The effect exceeds 1 pp because the AR root close to 1 keeps the following months higher too;' : ''} Θ<sub>1</sub> close to −1 takes the initial shock out of the 12-month window.`,
 dg1: (m, q12, d12, p12, q24, d24, p24) => `${m}: Ljung–Box Q(12) = ${q12} (${d12} df, p = ${p12}), Q(24) = ${q24} (${d24} df, p = ${p24}).`,
 dg2: (p12, p24) => `The identified model (1,1,0)(0,1,1)<sub>12</sub>: p = ${p12} and ${p24}.`,
 dg3: (jb, k, d1, v1, d2, v2) => `Jarque–Bera ${jb}, kurtosis ${k}; the largest residuals: ${d1} (${v1}σ) and ${d2} (${v2}σ).`,
 txHead: m => ['Model', `Q(${m}) (df)`, `p, m = ${m}`, 'LB(12) p', 'LB(24) p', 'Jarque–Bera', 'Kurtosis', 'Largest residual'],
 txNote: 'Residuals of the five models of the table of step 4, training sample; Ljung–Box with m − k degrees of freedom, k = the ARMA parameters. Highlighted row: the chosen model.',
 tx1: (m, ch, q, df, p, pi) => `m = ${m}: the chosen model ${ch} has Q = ${q} (${df} df, p = ${p}); the identified model has p = ${pi}.`,
 fc1: (o, aT, m1, a1, l1, h1, m12, a12, l12, h12, lv) => `Origin ${o} (annual inflation ${aT}%): forecast for ${m1} ${a1}% [${l1}, ${h1}], for ${m12} ${a12}% [${l12}, ${h12}], ${lv}% intervals.`,
 fc2: (n, rs, rn, rv) => `Over ${n} test months: RMSE ${rs} (SARIMA), ${rn} (seasonal naive), ${rv} (naive).`,
 tfHead: ['Monthly inflation, test sample', 'RMSE', 'MAE', 'MASE'],
 tfNote: (n, sc) => `The first ${n} test months (from September 2024); MASE = MAE / ${sc}, the in-sample MAE of the seasonal naive method.`,
 tf1: (abs, hs, ps, hn, pn) => `Diebold–Mariano (HLN, ${abs ? 'absolute' : 'squared'} errors): ${hs} (p = ${ps}) against the seasonal naive, ${hn} (p = ${pn}) against the naive method.`,
 tf2: (n, d1, a1, f1, d2, a2, f2) => `The largest misses in the first ${n} months: ${d1} (${a1}% against ${f1}%) and ${d2} (${a2}% against ${f2}%).`,
 q1s1: (ph, ph2, v) => `E(Y<sub>t</sub>) = 10 + 0.5t depends on t: not stationary; Var(Y<sub>t</sub>) = 1/(1 − ${ph2}) = ${v} is constant (φ = ${ph}): trend-stationary.`,
 q1s2: (ph, g0, g1, r1) => `ΔY<sub>t</sub> = 0.5 + Δu<sub>t</sub> and (1 − ${ph}L)Δu<sub>t</sub> = (1 − L)ε<sub>t</sub>: ARMA(1,1), θ = −1; γ(0) = ${g0}, γ(1) = ${g1}, ρ(1) = ${r1}.`,
 q1s3: (y, u, ph, f1, f10, v) => `u<sub>T</sub> = ${y} − 10 − 50 = ${u}; Ŷ<sub>T+1</sub> = 10 + 50.5 + ${ph} · ${u} = ${f1}; Ŷ<sub>T+10</sub> = 10 + 55 + ${ph}<sup>10</sup> · ${u} = ${f10}; 10-step variance ${v}.`,
 q1s4: 'The MA root is 1: ΔY<sub>t</sub> is not invertible; ρ̂<sub>1</sub> &lt; 0 and an MA coefficient near −1 are the signs of over-differencing.',
 q1o: (ph, p10, v) => `With φ = ${ph}, ${p10} of the deviation is left after 10 periods; the error variance rises toward ${v} and stays bounded.`,
 q2Head: ['Series', 'Terms', 'ADF τ', 'p', 'crit. 1%; 5%; 10%', 'KPSS', 'crit. 5%'],
 q2s1: (al, cv, ra, ag, cvg, rg) => `ADF, H<sub>0</sub>: γ = 0 (unit root). ln Y<sub>t</sub>: ${al} ${ra ? '&lt;' : '&gt;'} ${cv}: ${rej(ra)}; g<sub>t</sub>: ${ag} ${rg ? '&lt;' : '&gt;'} ${cvg}: ${rej(rg)}.`,
 q2s2: (kl, kc, rl, kg, kcg, rg) => `KPSS (H<sub>0</sub>: stationarity): level ${kl} ${rl ? '&gt;' : '&lt;'} ${kc}: ${rej(rl)}; growth ${kg} ${rg ? '&gt;' : '&lt;'} ${kcg}: ${rej(rg)}${rl && !rg ? ': ln Y<sub>t</sub> ~ I(1)' : ''}.`,
 q2s3: (r1, r2, p1, p2, bd, q, c8, p, ok) => `ACF ${r1}, ${r2}, PACF ${p1}, ${p2} (band ±${bd}); Q(8) = ${q} ${ok ? '&lt;' : '&gt;'} χ<sup>2</sup><sub>0.95</sub>(8) = ${c8} (p = ${p}): ${ok ? 'growth is white noise around its mean' : 'autocorrelation is left'}.`,
 q2s4: (c, se, s, t, a) => `ARIMA(0,1,0) with drift: 100 ln Y<sub>t</sub> = 100 ln Y<sub>t−1</sub> + ${c} + ε<sub>t</sub>, σ̂ = ${s}; t = ${c}/${se} = ${t}; about ${a}% a year.`,
 q2o: (e, n, m, s) => `2000Q2–${e}: ${n} quarters of growth, mean ${m}% a quarter, σ̂ = ${s}.`,
 q3s1: '(1 − φL)(1 − ΦL<sup>12</sup>)z<sub>t</sub> = ε<sub>t</sub>, so z<sub>t</sub> = φz<sub>t−1</sub> + Φz<sub>t−12</sub> − φΦz<sub>t−13</sub> + ε<sub>t</sub>.',
 q3s2: (ph, Ph, i1, i12, st) => `φ̂ = ${ph}: a monthly surprise continues one month later; Φ̂ = ${Ph}: a surprise is partly reversed in the same month a year later; inverted AR roots ${i1} and ${i12}${st ? ' &lt; 1: stationary' : ': not both below 1'}.`,
 q3s3: (q, df, p) => `Ljung–Box Q(12) = ${q} on ${df} df, p = ${p} (the lecture’s estimates): white noise is rejected at 5%.`,
 q3s4: (ph, zT, Ph, z11, pP, z12, t1, t2, t3, zn, mN, mT, m11, m12) => `ẑ (${mN}) = ${ph} · ${zT} + ${Ph} · ${z11} − ${pP} · ${z12} = ${t1} + ${t2} + ${t3} = ${zn} (z of ${mT}, ${m11}, ${m12}).`,
 q3s5: (aT, zn, an, mN) => `Annual inflation in ${mN}: Δ<sub>12</sub> ln P<sub>T+1</sub> = ${aT} + ${zn} = ${an}%.`,
 q3o: (mN, an, aa, za) => `Hand forecast for ${mN}: ${an}%${aa === null ? '.' : `; the realised value: ${aa}% (z = ${za}).`}`,
 q4Head: ['ROBOR equation', 'coef. [t]', '', 'coef. [t]'],
 q4Head2: ['Null hypothesis', 'F', 'p'],
 q4h: (x, y) => `${x} does not Granger-cause ${y}`,
 q4s1: (pg, pp, pi, hg, hp, hi) => `x Granger-causes y if the lags of x improve the forecast of y given the lags of y (joint F test): growth ${hg ? 'helps' : 'does not help'} to forecast ROBOR (p ${pg}), inflation ${hp ? 'helps' : 'does not help'} (p ${pp}); ROBOR ${hi ? 'helps' : 'does not help'} to forecast inflation (p ${pi}).`,
 q4s2: (c, g1, g2, p1, p2, i1, i2) => `i<sub>t</sub> = ${c} + ${g1}g<sub>t−1</sub> + ${g2}g<sub>t−2</sub> + ${p1}π<sub>t−1</sub> + ${p2}π<sub>t−2</sub> + ${i1}i<sub>t−1</sub> + ${i2}i<sub>t−2</sub> + u<sub>t</sub>.`,
 q4s3: (b1, b5, b3, b6, num, den, r) => `Long run (i<sub>t</sub> = i<sub>t−1</sub>, π<sub>t</sub> = π<sub>t−1</sub>): ∂i/∂π = (${b1} + ${b5})/(1 − ${b3} − ${b6}) = ${num}/${den} = ${r} pp.`,
 q4s4: (rr, ru, df, F, cr, rj, Fr, dif) => `F = ((${rr} − ${ru})/2)/(${ru}/${df}) = ${F} ${rj ? '&gt;' : '&lt;'} F<sub>0.95</sub>(2, ${df}) = ${cr}: H<sub>0</sub> ${rj ? 'rejected' : 'not rejected'}${dif ? ` (with the RSS rounded to three decimals one gets ${Fr}; the gap comes only from rounding)` : ''}.`,
 q4o: (e, n, lr, den) => `Sample to ${e} (T = ${n}): long-run effect ${lr} pp; the denominator 1 − b<sub>i,1</sub> − b<sub>i,2</sub> = ${den}, close to zero, makes the effect imprecise.`,
 q5s1: 'I(1) series are cointegrated if a linear combination is I(0); trace: 54.60 &gt; 29.80, 21.57 &gt; 15.49, 3.12 &lt; 3.84: rank r = 2, one common stochastic trend.',
 q5s2: (nm, a1, a2, t1, t2, u1, u2) => `Δ${nm}<sub>t</sub> = ${a1} EC1<sub>t−1</sub> + ${a2} EC2<sub>t−1</sub> + … (t of the lecture: ${t1} and ${t2}); the 1-year yield has |t| &lt; 1 on both relations (${u1}, ${u2}): the short rate does not adjust (weakly exogenous).`,
 q5s3: (b1, b2, e1, e2, h1, h2) => `β ≈ 1 (${b1} and ${b2}): the spreads y<sup>(1)</sup> − y<sup>(10)</sup> and y<sup>(5)</sup> − y<sup>(10)</sup> are stationary; eigenvalues of I + β′α: ${e1} and ${e2}; half-lives of the equilibrium errors: ${h1} and ${h2} months.`,
 q5o: (k, h1, h2, e1, e2) => `With α multiplied by ${k}: the half-lives are ${h1} and ${h2} months; in September 2026 EC1 = ${e1}, EC2 = ${e2}.`,
 q6s1: (ps, hl, om, lv, an) => `α + β = ${ps}; h<sub>1/2</sub> = ln 0.5/ln ${ps} = ${hl} days; σ̄<sup>2</sup> = ${om}/(1 − ${ps}) = ${lv}, √(250 × ${lv}) = ${an}% a year.`,
 q6s2: (om, al, e2, be, s2, t1, t2, s2n, sn) => `σ<sup>2</sup><sub>t+1</sub> = ${om} + ${al} × ${e2} + ${be} × ${s2} = ${om} + ${t1} + ${t2} = ${s2n}, σ<sub>t+1</sub> = ${sn}%.`,
 q6s3: (nu, tq, fr, sc, q, mu, sn, v, vn) => `q<sub>0.01</sub>(z) = ${tq} × √(${fr}) = ${tq} × ${sc} = ${q} (t<sub>${nu}</sub>); VaR 1% = −(${mu} + ${sn} × (${q})) = ${v}%; Normal: ${vn}%.`,
 q6o: (v, vn, vs) => `One-day VaR 1%: ${v}% with the standardised t quantile, ${vn}% with the Normal distribution; the sample volatility of the BET: ${vs}% a year.`,
 q7s1: (rA, mA, qA, rB, mB, qB, sc) => `SARIMA: RMSE ${rA}, MAE ${mA}, MASE ${qA}; seasonal naive: RMSE ${rB}, MAE ${mB}, MASE ${qB} (in-sample MAE ${sc}).`,
 q7s2: (dl, db, sd, dm, cr, rj) => `d<sub>t</sub>: ${dl}; d̄ = ${db}, s<sub>d</sub> = ${sd}; DM = ${dm}, |${dm}| ${rj ? '&gt;' : '&lt;'} ${cr}: equal accuracy is ${rj ? 'rejected' : 'not rejected'}.`,
 q7s3: (kf, wf) => `Random folds put future days in the training set: leakage; walk-forward validation gives R<sup>2</sup> = ${wf} for the same model, against the ${kf} reported.`,
 q7o: (l, dm, ls) => `With the seasonal naive errors multiplied by ${l}: DM = ${dm}; equal accuracy is rejected at 5% only from a factor of about ${ls}.`,
 q8s1: (K, y, v, af, Pf, s2n, Pn) => `F<sub>t</sub> = 2 + 4 = 6, K<sub>t</sub> = 2/6 = ${K}; a<sub>t|t</sub> = 10 + ${K} × ${v} = ${af}, P<sub>t|t</sub> = 2(1 − ${K}) = ${Pf}; a<sub>t+1</sub> = ${af}, P<sub>t+1</sub> = ${Pf} + ${s2n} = ${Pn} (y<sub>t</sub> = ${y}).`,
 q8s2: (q, Pb, Kb) => `q = σ<sub>η</sub><sup>2</sup>/σ<sub>ε</sub><sup>2</sup> = ${q}: P̄/σ<sub>ε</sub><sup>2</sup> = (q + √(q<sup>2</sup> + 4q))/2 = ${Pb}, K̄ = ${Pb}/(1 + ${Pb}) = ${Kb} = α<sub>SES</sub>.`,
 q8s3: (p11, p22, d1, d2, nu, dn, pi) => `E(D<sub>1</sub>) = 1/(1 − ${p11}) = ${d1} quarters, E(D<sub>2</sub>) = 1/(1 − ${p22}) = ${d2} quarters; π<sub>1</sub> = ${nu}/${dn} = ${pi}.`,
 q8s4: (d, H, r1, k) => `H = d + 0.5 = ${H}; ρ(1) = d/(1 − d) = ${r1}; ${k === 'st' ? '0 &lt; d &lt; 0.5: stationary with long memory, mean-reverting' : k === 'nst' ? '0.5 ≤ d &lt; 1: not stationary, but mean-reverting' : k === 'anti' ? '−0.5 &lt; d ≤ 0: stationary, intermediate memory (antipersistent)' : 'outside −0.5 &lt; d &lt; 1'} (d = ${d}).`,
 q8o: (s, K, p) => `With σ<sub>η</sub><sup>2</sup> = ${s}, the steady-state gain is ${K}; the volatile regime has a long-run share of ${p}%.`,
};
})();
""",
}

QA = {
 'ro': {
  'data': ('▶ mută sfîrșitul eșantionului din august 2010 în august 2026. Cum se schimbă media inflației lunare?',
           'Scade de la 0,52% (2005–2010) la 0,31% (pînă în august 2020), apoi urcă la 0,39% pe tot eșantionul: valul de inflație din 2021–2023 și șocul fiscal din 2025 ridică media. '
           'Minimul inflației anuale, −3,0% în mai 2016, cuprinde în fereastra de 12 luni reducerea TVA la alimente din iunie 2015; maximul, 13,6%, este în noiembrie 2022.'),
  't-data': ('▶ mută începutul eșantionului pînă în 2020. Rămîne inflația medie din iunie negativă?',
             'Nu: din 2020 iunie are în medie +0,28%, iar ianuarie rămîne luna cea mai mare (0,94%). Tiparul sezonier (ianuarie sus, iunie jos) se păstrează, dar nivelul crește '
             'odată cu inflația generală din 2021–2023; de aceea modelul lucrează cu Δ<sub>12</sub>, care elimină și nivelul, și sezonalitatea fixă.'),
  't-tests': ('▶ trece prin eșantioanele care se termină în decembrie 2012, …, 2025. Ce verdict primește inflația anuală Δ<sub>12</sub> ln P pentru eșantioanele 2013–2021?',
              'Rădăcină unitară: ADF nu respinge (p între 0,66 și 0,97), iar KPSS respinge (η între 0,59 și 1,48, peste 0,463). Rezultatul neconcludent din curs apare numai după 2022, '
              'cînd valul inflației adaugă variație. În aceiași ani Δ ln P primește „contradictoriu”: ambele teste resping, pentru că sezonalitatea și schimbările de nivel ale inflației '
              'lunare încalcă ipotezele ambelor teste.'),
  'acf': ('▶ trece la Δ ln P (fără diferența sezonieră). Ce arată ACF la lagurile 12, 24 și 36?',
          '0,25, 0,22 și 0,18, toate peste banda ±0,12 și descrescînd încet: tiparul sezonier nu a fost eliminat, deci este nevoie de Δ<sub>12</sub>. Pentru Δ<sub>12</sub> ln P ACF '
          'începe la 0,98 și scade foarte încet (o rădăcină aproape unitară); abia z = ΔΔ<sub>12</sub> ln P are o singură valoare sezonieră mare, −0,45 la lagul 12.'),
  't-acf': ('▶ trece prin φ = 0. Cît este atunci ρ<sub>12</sub> teoretic pentru Θ = −0,867?',
            '−0,49 = Θ/(1 + Θ<sup>2</sup>) = −0,867/1,752. Un MA(1) sezonier are o singură autocorelație sezonieră nenulă, la lagul 12, iar mărimea ei nu poate depăși 0,5 în modul; '
            'cu φ = 0 sateliții de la 11 și 13 dispar. Cu φ = 0,416 ei sînt aproximativ φ · ρ<sub>12</sub>, iar în eșantion ρ̂<sub>11</sub> = −0,12 și ρ̂<sub>13</sub> = −0,15, aproape de produsul ρ̂<sub>1</sub>ρ̂<sub>12</sub> = −0,15.'),
  't-grid': ('▶ mută penalizarea c de la 2 la 9. Ce model are criteriul cel mai mic la c = 2 (AIC fără corecție)?',
             '(2,1,1)(0,1,1)<sub>12</sub>, chiar modelul ales. Cu c = ln 223 = 5,41 (BIC) cîștigă (1,1,1)(0,1,1)<sub>12</sub>, care are un parametru mai puțin, dar pică testul Ljung–Box la lagul 12 '
             '(p = 0,029); de la c ≈ 5 în sus el rămîne primul. Modelul ales nu se schimbă pentru niciun c din grafic, pentru că este cel mai bun dintre modelele care trec verificarea.'),
  't-chosen': ('▶ trece prin cele cinci modele din tabel. Care are cea mai mică rădăcină AR cea mai apropiată de 1?',
               '(2,1,1)(0,1,1)<sub>12</sub>, cu modulul 1,034 (inversul 0,967), urmat de (1,1,2)(0,1,1)<sub>12</sub> cu 1,035. Modelul identificat are o singură rădăcină AR, 2,40: '
               'un AR(1) cu φ = 0,42 nu poate reproduce persistența inflației. Rădăcinile MA sezoniere au toate modulul aproape 1 (1,006): Θ<sub>1</sub> ≈ −1 aproape anulează Δ<sub>12</sub>.'),
  'diag': ('▶ trece prin cele cinci modele din tabelul pasului 4. La care dintre ele scade kurtosis-ul reziduurilor sub 5?',
           'La niciunul: kurtosis-ul rămîne între 9,5 și 10,0, iar Jarque–Bera între 390 și 454. Termenii AR și MA în plus elimină autocorelația, nu valorile extreme: reziduurile din '
           'iunie 2015 (−6,1σ) și iulie 2010 (5,3σ) vin din modificările TVA, pe care niciun model univariat nu le poate anticipa.'),
  't-diag': ('▶ mută m de la 6 la 36. Trece modelul ales (2,1,1)(0,1,1)<sub>12</sub> testul Ljung–Box la 5% pentru fiecare m?',
             'Nu: la m = 7 p = 0,025 și la m = 8 p = 0,048; la toate celelalte laguri, inclusiv 12 și 24 (cele din curs), p &gt; 0,05. Modelul identificat pică la fiecare m între 6 și 36. '
             'Un test la mai multe laguri nu este independent: alegerea lui m schimbă verdictul la limită, de aceea cursul raportează două laguri fixate dinainte.'),
  'fc': ('▶ mută originea prognozei din august 2025 în august 2026. Cu originea în august 2025, cît prognozează modelul pentru inflația anuală din august 2026 (realizată: 6,1%)?',
         '9,7%, cu intervalul de 95% [3,9; 15,5], care conține valoarea realizată. Modelul tratează o parte din salturile din iulie–august 2025 ca persistente (rădăcina AR 0,967) și nu '
         'știe că acestea au fost decizii fiscale unice; la originile următoare prognoza pe 12 luni coboară, cu oscilații, pînă la 6,3% din august 2026.'),
  't-fc': ('▶ mărește numărul de luni de test de la 6 la 24. Respinge testul Diebold–Mariano acuratețea egală față de naivul sezonier pentru vreo lungime?',
           'Nu: cel mai mic p este 0,174, chiar la 24 de luni. Cu 11 luni statistica devine chiar pozitivă (0,10), pentru că erorile mari din iulie–august 2025 cad pe SARIMA. '
           'RMSE mai mic nu înseamnă o diferență semnificativă: pe un eșantion scurt varianța lui d<sub>t</sub> este mare.'),
  'p1': ('▶ mută φ pînă la 0,95. Cît este ρ<sub>ΔY</sub>(1) pentru φ = 0,95?',
         '−0,025 = −(1 − φ)/2. Diferențierea unei serii staționare în jurul trendului dă un ARMA(1,1) cu θ = −1; pentru φ aproape de 1 rădăcina AR și rădăcina MA aproape se anulează '
         'și ACF a lui ΔY pare zgomot alb, deși ΔY rămîne neinvertibil. Cu φ = 0 corelația este −0,5, valoarea unui MA(1) cu θ = −1.'),
  'p2': ('▶ scurtează eșantionul pînă în 2019Q4 (fără pandemie). Trece creșterea testul Ljung–Box Q(8)?',
         'Da: Q(8) = 4,18, p = 0,841, iar deriva este 1,00% pe trimestru (σ̂ = 2,49). Pe tot eșantionul, cu 2020 inclus, Q(8) = 7,28 (p = 0,506) și deriva 0,81%: căderea din 2020Q2 '
         'scade media, dar nu creează autocorelație. ADF pentru nivel rămîne nerespins (−2,08 în 2019Q4).'),
  'p3': ('▶ mută originea în august 2025 (prognoza pentru septembrie 2025, chiar după saltul TVA). Cum se compară prognoza de mînă cu valoarea realizată?',
         'Prognoza este z = 0,753 și inflația anuală 8,93%; s-a realizat z = 0,090 și 8,26%. Termenul φz<sub>T</sub> duce saltul din august în septembrie, deși saltul a fost o decizie unică. '
         'Cu originea în iulie 2026 prognoza pentru august 2026 este 6,83%, față de 6,09% realizat: ieșirea saltului din fereastra de 12 luni este subestimată, pentru că |Φ̂| = 0,49 &lt; 1.'),
  'p4': ('▶ trece prin eșantioanele care se termină în 2015Q4, …, 2026Q2. Cum se schimbă efectul pe termen lung al inflației asupra ROBOR?',
         'Pînă în 2022Q4 stă între 1,06 și 1,32 (1,27 în 2019Q4), apoi coboară la 0,91 în 2023Q4 și 0,90 în 2026Q2. Numitorul 1 − b<sub>i,1</sub> − b<sub>i,2</sub> rămîne între 0,03 și 0,09, deci mici schimbări ale coeficienților '
         'mută mult raportul. Testul Granger pentru g → i respinge pe toate eșantioanele (p ≤ 0,008).'),
  'p5': ('▶ mărește multiplicatorul lui α de la 0,25 la 2. Cît devine timpul de înjumătățire lent (22,1 luni în curs) cu α dublat?',
         '10,9 luni, aproximativ jumătate: pentru valori proprii aproape de 1, ln λ ≈ −(1 − λ), iar 1 − λ este proporțional cu α. Componenta rapidă scade de la 9,0 la 4,3 luni. '
         'Cu α împărțit la 4 timpii devin 89,6 și 37,2 luni.'),
  'p6': ('Puneți ε<sub>t</sub> = +3 în loc de −3. Cît este σ<sup>2</sup><sub>t+1</sub>?',
         'Tot 2,5685: GARCH(1,1) depinde de ε<sub>t</sub><sup>2</sup>, deci o creștere de 3% și o scădere de 3% au același efect. Efectul de levier cere un model asimetric (GJR sau EGARCH, Capitolul 5). '
         'Cu ε<sub>t</sub> = 0, σ<sup>2</sup><sub>t+1</sub> = 0,844, sub varianța de azi.'),
  'p7': ('▶ înmulțește erorile metodei naive sezoniere cu un factor de la 0,5 la 2. De la ce factor respinge testul DM acuratețea egală la 5%?',
         'De la aproximativ 1,35: DM = −2,33 la 1,2 și −2,63 la 1,4, față de valoarea critică 2,571 cu 5 grade de libertate. Cu șase luni chiar o metodă cu erori cu 40% mai mari '
         'abia se distinge; MASE sub 1 nu este o dovadă.'),
  'p8': ('▶ mută σ<sub>η</sub><sup>2</sup> pînă la 4 (q = 1). Cît este cîștigul de echilibru?',
         '0,618 = (1 + √5)/2 / (1 + (1 + √5)/2). Cu cît nivelul se mișcă mai mult față de zgomot, cu atît filtrul crede mai mult observația nouă. Primul pas rămîne K<sub>t</sub> = 2/6 = 0,333, '
         'pentru că depinde doar de P<sub>t</sub> = 2 și σ<sub>ε</sub><sup>2</sup> = 4; cîștigul converge apoi la valoarea de echilibru în cîțiva pași.'),
 },
 'en': {
  'data': ('▶ moves the end of the sample from August 2010 to August 2026. How does the mean monthly inflation change?',
           'It falls from 0.52% (2005–2010) to 0.31% (to August 2020), then rises to 0.39% over the whole sample: the 2021–2023 inflation wave and the 2025 tax shock raise the mean. '
           'The minimum of annual inflation, −3.0% in May 2016, has the June 2015 VAT cut on food inside its 12-month window; the maximum, 13.6%, is in November 2022.'),
  't-data': ('▶ moves the start of the sample up to 2020. Does mean June inflation stay negative?',
             'No: since 2020 June averages +0.28%, and January remains the highest month (0.94%). The seasonal pattern (January high, June low) survives, but its level rises with the '
             'general inflation of 2021–2023; this is why the model works with Δ<sub>12</sub>, which removes both the level and the fixed seasonality.'),
  't-tests': ('▶ steps through the samples ending in December 2012, …, 2025. What verdict does annual inflation Δ<sub>12</sub> ln P get for the 2013–2021 samples?',
              'Unit root: ADF does not reject (p between 0.66 and 0.97) and KPSS rejects (η between 0.59 and 1.48, above 0.463). The inconclusive result of the lecture appears only '
              'after 2022, when the inflation wave adds variation. In the same years Δ ln P gets “conflicting”: both tests reject, because the seasonality and the level shifts of '
              'monthly inflation violate the assumptions of both tests.'),
  'acf': ('▶ switches to Δ ln P (no seasonal difference). What does the ACF show at lags 12, 24 and 36?',
          '0.25, 0.22 and 0.18, all above the ±0.12 band and decaying slowly: the seasonal pattern has not been removed, so Δ<sub>12</sub> is needed. For Δ<sub>12</sub> ln P the ACF '
          'starts at 0.98 and decays very slowly (a near unit root); only z = ΔΔ<sub>12</sub> ln P has a single large seasonal value, −0.45 at lag 12.'),
  't-acf': ('▶ passes through φ = 0. What is the theoretical ρ<sub>12</sub> then, for Θ = −0.867?',
            '−0.49 = Θ/(1 + Θ<sup>2</sup>) = −0.867/1.752. A seasonal MA(1) has a single nonzero seasonal autocorrelation, at lag 12, and its size cannot exceed 0.5 in absolute value; '
            'with φ = 0 the satellites at 11 and 13 vanish. With φ = 0.416 they are about φ · ρ<sub>12</sub>, and in the sample ρ̂<sub>11</sub> = −0.12 and ρ̂<sub>13</sub> = −0.15, close to the product ρ̂<sub>1</sub>ρ̂<sub>12</sub> = −0.15.'),
  't-grid': ('▶ moves the penalty c from 2 to 9. Which model has the lowest criterion at c = 2 (AIC without the correction)?',
             '(2,1,1)(0,1,1)<sub>12</sub>, the chosen model itself. With c = ln 223 = 5.41 (BIC) the winner is (1,1,1)(0,1,1)<sub>12</sub>, which has one parameter less but fails the '
             'Ljung–Box test at lag 12 (p = 0.029); from c ≈ 5 upward it stays first. The chosen model does not change for any c in the chart, because it is the best of the models that pass the checks.'),
  't-chosen': ('▶ steps through the five models of the table. Which one has its smallest AR root closest to 1?',
               '(2,1,1)(0,1,1)<sub>12</sub>, with modulus 1.034 (inverse 0.967), followed by (1,1,2)(0,1,1)<sub>12</sub> with 1.035. The identified model has a single AR root, 2.40: an '
               'AR(1) with φ = 0.42 cannot reproduce the persistence of inflation. The seasonal MA roots all have modulus close to 1 (1.006): Θ<sub>1</sub> ≈ −1 nearly cancels Δ<sub>12</sub>.'),
  'diag': ('▶ steps through the five models of the table of step 4. For which of them does the kurtosis of the residuals fall below 5?',
           'None: the kurtosis stays between 9.5 and 10.0 and Jarque–Bera between 390 and 454. Extra AR and MA terms remove autocorrelation, not outliers: the residuals of June 2015 '
           '(−6.1σ) and July 2010 (5.3σ) come from VAT changes that no univariate model can foresee.'),
  't-diag': ('▶ moves m from 6 to 36. Does the chosen model (2,1,1)(0,1,1)<sub>12</sub> pass the Ljung–Box test at 5% for every m?',
             'No: p = 0.025 at m = 7 and p = 0.048 at m = 8; at every other lag, 12 and 24 (those of the lecture) included, p &gt; 0.05. The identified model fails at every m from 6 to 36. '
             'Tests at several lags are not independent: the choice of m flips a borderline verdict, which is why the lecture reports two lags fixed in advance.'),
  'fc': ('▶ moves the forecast origin from August 2025 to August 2026. With the origin in August 2025, what does the model forecast for annual inflation in August 2026 (realised: 6.1%)?',
         '9.7%, with the 95% interval [3.9, 15.5], which contains the realised value. The model treats part of the July–August 2025 jumps as persistent (AR root 0.967) and cannot '
         'know they were one-off tax decisions; over the later origins the 12-month forecast comes down, with ups and downs, to 6.3% from August 2026.'),
  't-fc': ('▶ lengthens the test sample from 6 to 24 months. Does the Diebold–Mariano test reject equal accuracy against the seasonal naive for any length?',
           'No: the smallest p is 0.174, at 24 months. With 11 months the statistic even turns positive (0.10), because the large misses of July–August 2025 fall on SARIMA. A smaller '
           'RMSE is not a significant difference: on a short sample the variance of d<sub>t</sub> is large.'),
  'p1': ('▶ moves φ up to 0.95. What is ρ<sub>ΔY</sub>(1) for φ = 0.95?',
         '−0.025 = −(1 − φ)/2. Differencing a trend-stationary series gives an ARMA(1,1) with θ = −1; for φ close to 1 the AR root and the MA root nearly cancel and the ACF of ΔY looks like '
         'white noise, although ΔY stays non-invertible. With φ = 0 the correlation is −0.5, the value of an MA(1) with θ = −1.'),
  'p2': ('▶ shortens the sample to 2019Q4 (no pandemic). Does growth pass the Ljung–Box Q(8) test?',
         'Yes: Q(8) = 4.18, p = 0.841, and the drift is 1.00% a quarter (σ̂ = 2.49). Over the full sample, with 2020 included, Q(8) = 7.28 (p = 0.506) and the drift 0.81%: the 2020Q2 fall '
         'lowers the mean but creates no autocorrelation. The ADF test for the level is still not rejected (−2.08 in 2019Q4).'),
  'p3': ('▶ moves the origin to August 2025 (the forecast for September 2025, right after the VAT jump). How does the hand forecast compare with the realised value?',
         'The forecast is z = 0.753 and annual inflation 8.93%; the realised values are z = 0.090 and 8.26%. The term φz<sub>T</sub> carries the August jump into September, although the jump '
         'was a one-off decision. With the origin in July 2026 the forecast for August 2026 is 6.83%, against 6.09% realised: the exit of the jump from the 12-month window is underestimated, because |Φ̂| = 0.49 &lt; 1.'),
  'p4': ('▶ steps through the samples ending in 2015Q4, …, 2026Q2. How does the long-run effect of inflation on ROBOR change?',
         'Up to 2022Q4 it stays between 1.06 and 1.32 (1.27 in 2019Q4), then falls to 0.91 in 2023Q4 and 0.90 in 2026Q2. The denominator 1 − b<sub>i,1</sub> − b<sub>i,2</sub> stays between 0.03 and 0.09, so small changes in the '
         'coefficients move the ratio a lot. The Granger test for g → i rejects on every sample (p ≤ 0.008).'),
  'p5': ('▶ raises the multiplier of α from 0.25 to 2. What does the slow half-life (22.1 months in the lecture) become with α doubled?',
         '10.9 months, about half: for eigenvalues close to 1, ln λ ≈ −(1 − λ), and 1 − λ is proportional to α. The fast component falls from 9.0 to 4.3 months. With α divided by 4 the half-lives are 89.6 and 37.2 months.'),
  'p6': ('Set ε<sub>t</sub> = +3 instead of −3. What is σ<sup>2</sup><sub>t+1</sub>?',
         'Still 2.5685: GARCH(1,1) depends on ε<sub>t</sub><sup>2</sup>, so a 3% rise and a 3% fall have the same effect. The leverage effect needs an asymmetric model (GJR or EGARCH, Chapter 5). '
         'With ε<sub>t</sub> = 0, σ<sup>2</sup><sub>t+1</sub> = 0.844, below today’s variance.'),
  'p7': ('▶ multiplies the errors of the seasonal naive method by a factor from 0.5 to 2. From which factor does the DM test reject equal accuracy at 5%?',
         'From about 1.35: DM = −2.33 at 1.2 and −2.63 at 1.4, against the critical value 2.571 with 5 degrees of freedom. With six months even a method with 40% larger errors is barely '
         'distinguishable; a MASE below 1 is not proof.'),
  'p8': ('▶ moves σ<sub>η</sub><sup>2</sup> up to 4 (q = 1). What is the steady-state gain?',
         '0.618 = (1 + √5)/2 / (1 + (1 + √5)/2). The more the level moves relative to the noise, the more the filter trusts the new observation. The first step stays K<sub>t</sub> = 2/6 = 0.333, '
         'because it depends only on P<sub>t</sub> = 2 and σ<sub>ε</sub><sup>2</sup> = 4; the gain then converges to the steady state in a few steps.'),
 },
}

# Ce credeți? / What do you think?: (index of the correct option, [(RO, EN), ...]); keys checked against the page (node)
PREDICT = {
 'data': (0, [('Scade la aproximativ 0,31% pînă în 2020, apoi urcă la 0,39%.', 'It falls to about 0.31% by 2020, then rises to 0.39%.'),
              ('Rămîne aproximativ 0,39%: media unei serii staționare nu depinde de eșantion.', 'It stays about 0.39%: the mean of a stationary series does not depend on the sample.'),
              ('Crește continuu: prețurile cresc aproape în fiecare lună.', 'It rises steadily: prices rise almost every month.')]),
 't-data': (0, [('Nu: iunie are +0,28%, iar ianuarie rămîne luna cea mai mare (0,94%).', 'No: June has +0.28%, and January remains the highest month (0.94%).'),
                ('Da: tiparul sezonier este fix, deci iunie rămîne −0,03%.', 'Yes: the seasonal pattern is fixed, so June stays at −0.03%.'),
                ('Nu, iar vîrful se mută din ianuarie în octombrie.', 'No, and the peak moves from January to October.')]),
 't-tests': (0, [('Rădăcină unitară: ADF nu respinge, KPSS respinge.', 'Unit root: ADF does not reject, KPSS rejects.'),
                 ('Staționară: ADF nu respinge ipoteza de staționaritate.', 'Stationary: ADF does not reject stationarity.'),
                 ('Neconcludent, ca în curs: verdictul nu depinde de eșantion.', 'Inconclusive, as in the lecture: the verdict does not depend on the sample.')]),
 'acf': (0, [('Valori pozitive peste bandă, care scad încet: 0,25, 0,22, 0,18.', 'Positive values above the band, decaying slowly: 0.25, 0.22, 0.18.'),
             ('O singură valoare negativă mare la lagul 12, ca la z.', 'A single large negative value at lag 12, as for z.'),
             ('Toate în bandă: diferența de ordinul 1 elimină sezonalitatea.', 'All inside the band: the first difference removes the seasonality.')]),
 't-acf': (0, [('−0,49 = Θ/(1 + Θ²).', '−0.49 = Θ/(1 + Θ²).'), ('−0,87: ρ₁₂ este chiar Θ.', '−0.87: ρ₁₂ is Θ itself.'),
               ('0: fără parte AR nu există autocorelație.', '0: without an AR part there is no autocorrelation.')]),
 't-grid': (0, [('(2,1,1)(0,1,1)₁₂, modelul ales.', '(2,1,1)(0,1,1)₁₂, the chosen model.'),
                ('(1,1,1)(0,1,1)₁₂, ca la BIC: penalizarea nu schimbă ordinea.', '(1,1,1)(0,1,1)₁₂, as with BIC: the penalty does not change the ranking.'),
                ('(2,1,2)(1,1,1)₁₂, cel mai mare model din grilă.', '(2,1,2)(1,1,1)₁₂, the largest model of the grid.')]),
 't-chosen': (0, [('(2,1,1)(0,1,1)₁₂: modulul 1,034.', '(2,1,1)(0,1,1)₁₂: modulus 1.034.'),
                  ('(1,1,0)(0,1,1)₁₂, modelul identificat: un singur termen AR este cel mai persistent.', '(1,1,0)(0,1,1)₁₂, the identified model: a single AR term is the most persistent.'),
                  ('(1,1,1)(1,1,1)₁₂: termenul AR sezonier pune rădăcini pe cercul unitate.', '(1,1,1)(1,1,1)₁₂: the seasonal AR term puts roots on the unit circle.')]),
 'diag': (0, [('La niciunul: kurtosis-ul rămîne între 9,5 și 10,0.', 'None: the kurtosis stays between 9.5 and 10.0.'),
              ('La modelele care trec Ljung–Box: fără autocorelație reziduurile devin Normale.', 'The models that pass Ljung–Box: without autocorrelation the residuals become Normal.'),
              ('La modelul cu AR sezonier, (1,1,1)(1,1,1)₁₂.', 'The model with a seasonal AR, (1,1,1)(1,1,1)₁₂.')]),
 't-diag': (0, [('Nu: la m = 7 și m = 8 p este sub 0,05.', 'No: at m = 7 and m = 8 p is below 0.05.'),
                ('Da: trece la 12 și la 24, deci trece la orice m.', 'Yes: it passes at 12 and 24, so it passes at every m.'),
                ('Nu: pică la toate lagurile de peste 24.', 'No: it fails at every lag above 24.')]),
 'fc': (0, [('Aproximativ 9,7%, cu un interval de 95% care conține 6,1%.', 'About 9.7%, with a 95% interval that contains 6.1%.'),
            ('Aproximativ 6,1%: modelul știe că saltul TVA iese din fereastra de 12 luni.', 'About 6.1%: the model knows the VAT jump leaves the 12-month window.'),
            ('Aproximativ 2,5%, ținta BNR: prognozele pe 12 luni revin la medie.', 'About 2.5%, the BNR target: 12-month forecasts return to the mean.')]),
 't-fc': (0, [('Nu: cel mai mic p este 0,174.', 'No: the smallest p is 0.174.'),
              ('Da, la 24 de luni: SARIMA are RMSE mai mic.', 'Yes, at 24 months: SARIMA has the smaller RMSE.'),
              ('Da, pe eșantioanele scurte, unde SARIMA are MASE 0,24.', 'Yes, on the short samples, where SARIMA has a MASE of 0.24.')]),
 'p1': (0, [('−0,025.', '−0.025.'), ('−0,5: θ = −1 dă mereu ρ(1) = −0,5.', '−0.5: θ = −1 always gives ρ(1) = −0.5.'), ('0,95: ΔY moștenește φ.', '0.95: ΔY inherits φ.')]),
 'p2': (0, [('Da: Q(8) = 4,18, p = 0,84.', 'Yes: Q(8) = 4.18, p = 0.84.'),
            ('Nu: fără 2020 creșterea devine autocorelată.', 'No: without 2020 growth becomes autocorrelated.'),
            ('Nu se poate testa: ADF respinge rădăcina unitară pentru nivel.', 'It cannot be tested: ADF rejects the unit root for the level.')]),
 'p3': (0, [('Prognoza 8,93% depășește valoarea realizată de 8,26%.', 'The forecast of 8.93% exceeds the realised 8.26%.'),
            ('Prognoza este exactă: modelul are AR sezonier.', 'The forecast is exact: the model has a seasonal AR.'),
            ('Prognoza este sub valoarea realizată: Φ̂ < 0 inversează saltul chiar în luna următoare.', 'The forecast is below the realised value: Φ̂ < 0 reverses the jump in the very next month.')]),
 'p4': (0, [('Scade de la aproximativ 1,1–1,3 la 0,90.', 'It falls from about 1.1–1.3 to 0.90.'),
            ('Rămîne 0,90: coeficienții VAR sînt stabili.', 'It stays 0.90: the VAR coefficients are stable.'),
            ('Crește peste 2: eșantioanele scurte au rădăcini unitare.', 'It rises above 2: short samples have unit roots.')]),
 'p5': (0, [('Aproximativ 11 luni, jumătate.', 'About 11 months, half.'), ('Aproximativ 44 de luni: o ajustare mai mare înseamnă mai multă inerție.', 'About 44 months: stronger adjustment means more inertia.'),
            ('22,1 luni: timpul de înjumătățire depinde doar de β.', '22.1 months: the half-life depends only on β.')]),
 'p6': (0, [('Tot 2,5685.', 'Still 2.5685.'), ('Mai mic, 0,84: un șoc pozitiv reduce volatilitatea.', 'Smaller, 0.84: a positive shock lowers volatility.'),
            ('Mai mare: GARCH dă un efect de levier.', 'Larger: GARCH gives a leverage effect.')]),
 'p7': (0, [('De la aproximativ 1,35.', 'From about 1.35.'), ('De la 1: MASE-ul naivului este deja 0,94.', 'From 1: the naive MASE is already 0.94.'),
            ('De la 2,571, valoarea critică.', 'From 2.571, the critical value.')]),
 'p8': (0, [('0,618.', '0.618.'), ('0,5 = q/(1 + q).', '0.5 = q/(1 + q).'), ('0,333, ca la primul pas.', '0.333, as at the first step.')]),
}
VEIL_CHARTS = {'p6'}

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slide „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('ADF nu respinge, iar KPSS respinge. Ce concluzie trageți?', 'ADF does not reject and KPSS rejects. What do you conclude?'), key=0, opts=[
  (('Ambele indică o rădăcină unitară: diferențiem seria.', 'Both point to a unit root: difference the series.'),
   ('H₀ ADF este rădăcina unitară (nerespinsă), H₀ KPSS este staționaritatea (respinsă): cele două teste spun același lucru.', 'The ADF H₀ is a unit root (not rejected), the KPSS H₀ is stationarity (rejected): the two tests agree.')),
  (('Seria este staționară: ADF nu găsește o rădăcină unitară.', 'The series is stationary: ADF finds no unit root.'),
   ('Inversează ipoteza nulă a ADF: nerespingerea înseamnă că rădăcina unitară nu poate fi exclusă.', 'It inverts the ADF null: not rejecting means the unit root cannot be excluded.')),
  (('Rezultat neconcludent: testele se contrazic.', 'Inconclusive: the tests contradict each other.'),
   ('Neconcludent este cazul în care niciun test nu respinge; aici ambele indică aceeași concluzie.', 'Inconclusive is when neither test rejects; here both point the same way.'))]),
 dict(q=('Reziduurile unui ARMA(1,1) dau Ljung–Box Q(10) = 16. Este modelul adecvat la 5%?', 'The residuals of an ARMA(1,1) give Ljung–Box Q(10) = 16. Is the model adequate at 5%?'), key=1, opts=[
  (('Da: valoarea critică χ²₀,₉₅(10) = 18,31 este peste 16.', 'Yes: the critical value χ²₀.₉₅(10) = 18.31 is above 16.'),
   ('Pe reziduuri gradele de libertate sînt m − k = 10 − 2 = 8, nu 10.', 'On residuals the degrees of freedom are m − k = 10 − 2 = 8, not 10.')),
  (('Nu: cu 8 grade de libertate valoarea critică este 15,51 < 16.', 'No: with 8 degrees of freedom the critical value is 15.51 < 16.'),
   ('Rămîne autocorelație: modelul trebuie extins.', 'Autocorrelation is left: the model must be extended.')),
  (('Nu se poate decide fără p-value.', 'It cannot be decided without the p-value.'),
   ('Valoarea critică χ²₀,₉₅(8) = 15,51 este suficientă pentru decizie.', 'The critical value χ²₀.₉₅(8) = 15.51 is enough for the decision.'))]),
 dict(q=('Două serii I(1) sînt cointegrate. De ce nu un VAR în diferențe?', 'Two I(1) series are cointegrated. Why not a VAR in differences?'), key=0, opts=[
  (('Omite termenul de corecție a erorii și pierde informația pe termen lung.', 'It omits the error-correction term and loses the long-run information.'),
   ('VECM = VAR în diferențe plus αβ′yₜ₋₁; fără acest termen modelul este greșit specificat.', 'VECM = VAR in differences plus αβ′yₜ₋₁; without that term the model is misspecified.')),
  (('Pentru că diferențele seriilor cointegrate nu sînt staționare.', 'Because the differences of cointegrated series are not stationary.'),
   ('Diferențele sînt staționare; problema este termenul omis, nu staționaritatea.', 'The differences are stationary; the problem is the omitted term, not stationarity.')),
  (('Nu există niciun motiv: VAR în diferențe și VECM dau aceleași prognoze.', 'There is no reason: a VAR in differences and a VECM give the same forecasts.'),
   ('Prognozele pe termen lung diferă: VECM readuce seriile la relația de echilibru.', 'Long-horizon forecasts differ: the VECM pulls the series back to the equilibrium relation.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {})) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch15-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] = TEXT_JS[lang].replace('\n};\n})();\n', '\n};\n') + ('window.CH15_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                                                                        + 'window.CH15_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n})();\n')


_predict_js()


if __name__ == '__main__':
    import interactive_common
    page_spec = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA', 'TEXT_JS', 'PREDICT', 'TICKET')})
    page_spec.widgets = lambda lang: {a: v for a, v in widgets_all(lang).items() if a in {b for b, _, _ in BLOCKS}}
    interactive_common.build(page_spec)
    for lg in ('en', 'ro'):
        _post(lg)
