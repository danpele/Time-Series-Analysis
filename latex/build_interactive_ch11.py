"""Build the interactive pages of Chapter 11 (foundation models for time series): interactive/ch11.html (EN) and
interactive/ch11_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its interactive version on
the right. The result tables of the lecture (the families side by side, the WQL ratios of the backtest), its worked examples
(attention weights, scaling and tokens, the weighted quantile loss, pinball losses, the MASE with the seasonal-naive scale,
coverage and relative scores, scoring one hour) and the extra charts where the lecture gives only numbers (inputs and
attention comparisons per context length, the scores as a heat map) get blocks of their own (EXTRA), placed by slide; on the
web lecture each of them travels inside a chart block and ch11.js moves it to its own frame. Logic in interactive/ch11.js,
inputs in interactive/ch11_data.js (Quantlets/Ch_11/export_interactive.py), slide images in interactive/slides/.
No foundation model runs in the browser: the page uses the forecasts, quantiles and scores the chapter's code saved.
"""
import json
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 11

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_11'
TEX = {'ro': 'RO/Cursuri/capitol11_foundation_models_serii_timp', 'en': 'EN/Courses/chapter11_foundation_models_time_series'}

# (anchor, chart file, section key); order of the slides
BLOCKS = [('tokens', 'tsa_ch11_tokens', 'tok'), ('patching', 'tsa_ch11_patching', 'tok'),
          ('scores', 'tsa_ch11_scores', 'sc'),
          ('series', 'tsa_ch11_series', 'zs'), ('fan-load', 'tsa_ch11_fan_load', 'zs'), ('fan-ip', 'tsa_ch11_fan_ip', 'zs'),
          ('fan-eurron', 'tsa_ch11_fan_eurron', 'zs'),
          ('benchmark', 'tsa_ch11_benchmark', 'ev'), ('coverage', 'tsa_ch11_coverage', 'ev'), ('horizon', 'tsa_ch11_horizon', 'ev'),
          ('context', 'tsa_ch11_context', 'ev'), ('size', 'tsa_ch11_size', 'ev'),
          ('prepost', 'tsa_ch11_prepost', 'lk')]

# result tables, worked examples and extra charts: (anchor, frame title EN, frame title RO, section key, host chart block)
EXTRA = [('wx-att', 'Worked example: attention weights', 'Exemplu rezolvat: ponderile atenției', 'tr', 'tokens'),
         ('wx-tok', 'Worked example: scaling and tokens', 'Exemplu rezolvat: scalarea și tokenii', 'tok', 'tokens'),
         ('fam-table', 'The families side by side', 'Comparația familiilor de modele', 'fam', 'patching'),
         ('wx-wql', 'The weighted quantile loss (WQL)', 'Pierderea cuantilă ponderată (WQL)', 'sc', 'scores'),
         ('wx-pin', 'Worked example: pinball losses', 'Exemplu rezolvat: pierderile pinball', 'sc', 'scores'),
         ('wx-mase', 'Point accuracy: the MASE', 'Precizia punctuală: MASE', 'sc', 'scores'),
         ('wx-cov', 'Coverage and relative scores', 'Acoperirea și scorurile relative', 'sc', 'scores'),
         ('wx-hour', 'Worked example: scoring one hour', 'Exemplu rezolvat: evaluarea unei ore', 'zs', 'fan-load'),
         ('res-table', 'Results: WQL relative to the seasonal naive', 'Rezultate: WQL relativ la prognoza sezonieră naivă', 'ev', 'benchmark')]

PFX = {'tokens': 'tk', 'patching': 'pa', 'scores': 'sc', 'series': 'sr', 'fan-load': 'fl', 'fan-ip': 'fi', 'fan-eurron': 'fe', 'benchmark': 'bm',
       'coverage': 'cv', 'horizon': 'hz', 'context': 'cx', 'size': 'sz', 'prepost': 'pp', 'wx-att': 'wa', 'wx-tok': 'wt', 'fam-table': 'ft',
       'wx-wql': 'ww', 'wx-pin': 'wp', 'wx-mase': 'wm', 'wx-cov': 'wc', 'wx-hour': 'wh', 'res-table': 'rt'}

SECTIONS = {
 'ro': dict(tr='Transformer-ul, pe scurt', tok='De la numere la tokeni', fam='Principalele familii de modele', sc='Prognozele probabiliste și scorurile lor',
            zs='Prognoze zero-shot pe date reale', ev='Evaluarea cu origine mobilă pe șapte serii', lk='Leakage-ul și contaminarea'),
 'en': dict(tr='Transformers in brief', tok='From numbers to tokens', fam='The main model families', sc='Probabilistic forecasts and their scores',
            zs='Zero-shot forecasts on real data', ev='Rolling-origin evaluation on seven series', lk='Leakage and contamination'),
}

L = {
 'ro': dict(file='ch11_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 11 · Foundation models pentru serii de timp', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'casetele „Exemplu rezolvat” refac calculul cu cifrele capitolului. Niciun foundation model nu rulează în browser: pagina folosește prognozele, cuantilele '
        'și scorurile salvate de codul capitolului.',
   src='Rezultatele backtestului (Quantlets/Ch_11: ch11_backtest.csv, ch11_abs_err.json, ch11_numbers.json): MASE, părțile WQL, acoperirea intervalului de 80% și '
       'eroarea absolută a medianei la fiecare pas, pentru fiecare origine și fiecare model; reproduc exact cifrele din curs. Cuantilele zero-shot ale Chronos-Bolt '
       'small din cele trei fan charts (nivelurile 0,1; 0,3; 0,5; 0,7; 0,9) sînt citite din graficele vectoriale ale slide-urilor (charts/tsa_ch11_fan_load.pdf, '
       'tsa_ch11_fan_ip.pdf, tsa_ch11_fan_eurron.pdf), în care fiecare pas al prognozei este un vîrf al benzii și al medianei; reproduc cifrele din curs pînă la a '
       'cincea cifră semnificativă. Seriile, citite cu funcțiile capitolului pe 4 noiembrie 2026 și tăiate la ultima perioadă din slide-uri: consumul orar al României '
       '(extrasul ENTSO-E din Capitolul 4), inflația și rata șomajului (Eurostat), cursul EUR/RON (BNR) nu au fost revizuite și reproduc exact cifrele din curs. '
       'Producția industrială (Eurostat sts_inpr_m): valoarea din iulie 2026 a fost revizuită după realizarea slide-urilor (94,5 în slide, 94,6 cu datele din 4 noiembrie '
       '2026); cele 96 de luni ale graficului tokenilor și cele 60 de luni ale fan chart-ului sînt citite din graficele slide-urilor (fiecare lună este un vîrf al liniei). '
       'PIB-ul real al României (Eurostat namq_10_gdp) a fost revizuit: graficul seriilor folosește linia din charts/tsa_ch11_series.pdf (toate trimestrele 2005–2026). '
       'Vînzările cu amănuntul din SUA (FRED RSXFSN): iulie și august 2026 au fost revizuite; graficul folosește datele din 4 noiembrie 2026, iar valorile din slide sînt '
       'date alături. Variantele care nu apar în curs (alte origini, alte orizonturi, alt context, altă dată de tăiere, alt sezon m, alte intervale de cuantizare) sînt '
       'rulări suplimentare ale codului capitolului pe rezultatele salvate; graficele din „Comparația familiilor de modele”, „Pierderea cuantilă ponderată (WQL)”, '
       '„Acoperirea și scorurile relative”, „Precizia punctuală: MASE” și „Rezultate: WQL relativ la prognoza sezonieră naivă” sînt grafice suplimentare pentru '
       'slide-uri care dau doar cifre.'),
 'en': dict(file='ch11.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 11 · Foundation Models for Time Series', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; '
        'the “Worked example” boxes redo the calculation with the numbers of the chapter. No foundation model runs in the browser: the page uses the forecasts, quantiles '
        'and scores saved by the chapter’s code.',
   src='Results of the backtest (Quantlets/Ch_11: ch11_backtest.csv, ch11_abs_err.json, ch11_numbers.json): MASE, the WQL parts, the coverage of the 80% interval and the '
       'absolute error of the median at every step, for every origin and model; they reproduce the numbers of the lecture exactly. The zero-shot quantiles of Chronos-Bolt '
       'small in the three fan charts (levels 0.1, 0.3, 0.5, 0.7, 0.9) are read from the vector charts of the slides (charts/tsa_ch11_fan_load.pdf, tsa_ch11_fan_ip.pdf, '
       'tsa_ch11_fan_eurron.pdf), in which every forecast step is a vertex of the band and of the median; they reproduce the numbers of the lecture to the fifth '
       'significant digit. The series, read with the chapter’s loaders on 4 November 2026 and cut at the last period of the slides: the hourly Romanian load (the ENTSO-E '
       'extract of Chapter 4), inflation and the unemployment rate (Eurostat) and EUR/RON (BNR) have not been revised and reproduce the numbers of the lecture exactly. '
       'Industrial production (Eurostat sts_inpr_m): the July 2026 value has been revised since the slides were made (94.5 on the slide, 94.6 with the data of 4 November '
       '2026); the 96 months of the token chart and the 60 months of the fan chart are read from the charts of the slides (every month is a vertex of the line). '
       'Romanian real GDP (Eurostat namq_10_gdp) has been revised: the series chart uses the line of charts/tsa_ch11_series.pdf (every quarter 2005–2026). US retail '
       'sales (FRED RSXFSN): July and August 2026 have been revised; the chart uses the data of 4 November 2026 and the slide values are given next to them. Variants that '
       'do not appear in the lecture (other origins, horizons, contexts, cut dates, season m, quantisation grids) are extra runs of the chapter’s code on the saved '
       'results; the charts of “The families side by side”, “The weighted quantile loss (WQL)”, “Coverage and relative scores”, “Point accuracy: the MASE” and '
       '“Results: WQL relative to the seasonal naive” are extra charts for slides that give only numbers.'),
}

CSS = ('<style>.c11x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c11x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c11x .c11f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c11x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c11x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c11x .wctl .ctl{max-width:360px;margin:.15rem 0 .2rem}'
       '.c11t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c11t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c11t th,.c11t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c11t thead tr:last-child th{border-bottom:1px solid #1A3A6E}'
       '.c11t tbody tr:nth-child(even){background:#F5F7FB}.c11t tbody tr.hi{background:#E3F4F7}.c11t .r{text-align:right}.c11cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}</style>')


def tbl(i):
    return f'<div class="tbl c11t" id="{i}"></div>'


def wx(i, title, formula, ctl=''):
    """Worked-example box; `ctl` holds the controls that only drive the box (they stay out of the chart's controls bar)."""
    return f'<div class="c11x"><h4>{title}</h4>' + (f'<div class="wctl">{ctl}</div>' if ctl else '') + f'<div class="c11f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def box(ctl, inner):
    """A box with controls that only drive the table inside it."""
    return f'<div class="c11x"><div class="wctl">{ctl}</div>{inner}</div>'


SER = ['load', 'ip', 'gdp', 'infl', 'unemp', 'eurron', 'retail']
MODELS = ['Seasonal naive', 'ETS', 'ARIMA', 'Chronos-Bolt tiny', 'Chronos-Bolt small', 'Chronos-2']
SNAME = {'ro': dict(load='Consumul RO (orar)', ip='Producția industrială RO', gdp='PIB-ul real RO', infl='Inflația RO', unemp='Șomajul RO', eurron='EUR/RON (zilnic)',
                    retail='Vînzările cu amănuntul SUA'),
         'en': dict(load='RO load (hourly)', ip='RO industrial production', gdp='RO real GDP', infl='RO inflation', unemp='RO unemployment', eurron='EUR/RON (daily)',
                    retail='US retail sales')}
UMON = {'ro': ['martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august'], 'en': ['March', 'April', 'May', 'June', 'July', 'August']}


def widgets_all(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    lec = s(' (curs)', ' (lecture)')
    W = s('Exemplu rezolvat', 'Worked example')
    sn = SNAME[lang]
    frm = lambda i: select(i, s('Originile incluse', 'Origins included'), [('', s('toate', 'all') + lec)] + [(f'{y}-01-01', s(f'din {y}', f'from {y}')) for y in (2018, 2020, 2022, 2024)], '')  # noqa: E731
    return {
     'tokens': (slider('tk-n', s('Lungimea contextului (luni; curs: 96)', 'Context length (months; lecture: 96)'), 12, 96, 12, 96)
                + select('tk-b', s('Grila de cuantizare', 'Quantisation grid'), [('21', s('21 de intervale în [−3, 3]', '21 bins in [−3, 3]') + lec), ('41', s('41 de intervale în [−3, 3]', '41 bins in [−3, 3]')),
                                                                               ('101', s('101 intervale în [−3, 3]', '101 bins in [−3, 3]')), ('4093', s('4093 de intervale în [−15, 15] (Chronos)', '4093 bins in [−15, 15] (Chronos)'))], '21')
                + play('tk'), CSS + charts('tk-chart')),
     'patching': (slider('pa-P', s('Lungimea patch-ului P (valori; curs: 16)', 'Patch length P (values; lecture: 16)'), 4, 48, 4, 16)
                  + slider('pa-n', s('Ore afișate, din 1 iunie 2026 (curs: 336)', 'Hours shown, from 1 June 2026 (lecture: 336)'), 168, 672, 168, 336) + play('pa'), charts('pa-chart')),
     'scores': (slider('sc-y', s('Valoarea observată y (curs: 1)', 'Outcome y (lecture: 1)'), -3, 3, 0.25, 1)
                + slider('sc-s', s('σ al prognozei N(0, σ²) (curs: 1)', 'σ of the forecast N(0, σ²) (lecture: 1)'), 0.5, 3, 0.25, 1) + play('sc'), charts('sc-chart')),
     'series': (select('sr-v', s('Afișați', 'Show'), [('lev', s('nivelurile', 'the levels') + lec), ('sd', s('diferențele sezoniere y<sub>t</sub> − y<sub>t−m</sub> (m din backtest)', 'seasonal differences y<sub>t</sub> − y<sub>t−m</sub> (m of the backtest)')),
                                                    ('d1', s('primele diferențe y<sub>t</sub> − y<sub>t−1</sub>', 'first differences y<sub>t</sub> − y<sub>t−1</sub>'))], 'lev') + play('sr'), charts('sr-chart')),
     'fan-load': (slider('fl-H', s('Orele evaluate (curs: 48)', 'Hours scored (lecture: 48)'), 6, 48, 6, 48) + play('fl'), charts('fl-chart')),
     'fan-ip': (slider('fi-H', s('Lunile evaluate (curs: 12)', 'Months scored (lecture: 12)'), 1, 12, 1, 12) + play('fi'), charts('fi-chart')),
     'fan-eurron': (slider('fe-H', s('Zilele evaluate (curs: 20)', 'Days scored (lecture: 20)'), 1, 20, 1, 20) + play('fe'), charts('fe-chart')),
     'benchmark': (frm('bm-f') + play('bm'), charts('bm-chart')),
     'coverage': (frm('cv-f') + play('cv'), charts('cv-chart')),
     'horizon': (select('hz-s', s('Seria', 'Series'), [(k, sn[k] + (lec if k == 'load' else '')) for k in SER], 'load')
                 + select('hz-w', s('Originile', 'Origins'), [('all', s('toate', 'all') + lec), ('a', s('prima jumătate', 'first half')), ('b', s('a doua jumătate', 'second half'))], 'all')
                 + play('hz'), charts('hz-chart')),
     'context': (slider('cx-c', s('Contextul evidențiat (ore)', 'Context highlighted (hours)'), 0, 5, 1, 0)
                 + select('cx-r', s('Axa verticală', 'Vertical axis'), [('abs', 'MASE' + lec), ('rel', s('MASE / prognoza sezonieră naivă', 'MASE / seasonal naive'))], 'abs') + play('cx'), charts('cx-chart')),
     'size': (frm('sz-f') + select('sz-m', s('Scorul', 'Score'), [('rel_wql', s('WQL / prognoza sezonieră naivă', 'WQL / seasonal naive') + lec), ('rel_mase', s('MASE / prognoza sezonieră naivă', 'MASE / seasonal naive'))], 'rel_wql')
              + play('sz'), charts('sz-chart')),
     'prepost': (select('pp-c', s('Data de tăiere', 'Cut date'), [('-12', s('lansarea − 12 luni (placebo)', 'release − 12 months (placebo)')), ('-6', s('lansarea − 6 luni (placebo)', 'release − 6 months (placebo)')),
                                                                ('0', s('data lansării ponderilor', 'release of the weights') + lec), ('6', s('lansarea + 6 luni (placebo)', 'release + 6 months (placebo)'))], '0')
                 + play('pp'), charts('pp-chart')),
     # ---------------------------------------------------------------- EXTRA
     'wx-att': (slider('wa-c', s('Multiplicatorul c al scorurilor (curs: 1)', 'Multiplier c of the scores (lecture: 1)'), 0, 3, 0.25, 1) + play('wa'),
                charts('wa-chart') + wx('wa-ws', W, r's=(2c,\ c,\ 0),\qquad w_j=\frac{\exp(s_j)}{\sum_l \exp(s_l)},\qquad \text{out}=\sum_j w_jv_j,\quad v=(10,\ 20,\ 30)')),
     'wx-tok': (select('wt-b', s('Numărul de intervale B în [−15, 15]', 'Number of bins B in [−15, 15]'), [('4093', '4093' + lec), ('1001', '1001'), ('101', '101'), ('21', '21')], '4093') + play('wt'),
                charts('wt-chart') + wx('wt-ws', W, r's=\frac1C\sum_{t=1}^{C}|x_t|,\qquad \tilde x_t=\frac{x_t}{s},\qquad \text{token}_t=\operatorname{round}\Big(\frac{\tilde x_t+15}{w}\Big),\quad w=\frac{30}{B-1}',
                                        select('wt-m', s('Luna din calcul', 'Month of the calculation'), [(str(i), UMON[lang][i] + ' 2026') for i in range(6)], '0'))),
     'fam-table': (slider('ft-c', s('Lungimea contextului C (valori; slide-ul Patching: 2048)', 'Context length C (values; slide Patching: 2048)'), 256, 4096, 256, 2048) + play('ft'),
                   charts('ft-chart') + box(select('ft-h', s('Evidențiați', 'Highlight'), [('none', s('nimic', 'nothing')), ('apache', s('licența Apache-2.0', 'the Apache-2.0 licence')),
                                                                                         ('patch', s('intrarea prin patch-uri', 'patch inputs')), ('dec', 'decoder-only')], 'none'), tbl('ft-tb'))),
     'wx-wql': (select('ww-s', s('Seria', 'Series'), [(k, sn[k]) for k in SER], 'load') + select('ww-m', s('Modelul', 'Model'), [(m, m) for m in MODELS[1:]], 'Chronos-2') + play('ww'),
                charts('ww-chart') + wx('ww-ws', W, r'\text{WQL}=\frac{1}{9}\sum_{\tau}\frac{2\sum_t L_\tau(y_t,\hat q_{\tau,t})}{\sum_t|y_t|}=\frac{\sum_{\text{origins}}\text{num}}{\sum_{\text{origins}}\text{den}},\quad \text{num}=\sum_\tau\sum_t 2L_\tau,\ \ \text{den}=9\sum_t|y_t|',
                                        slider('ww-o', s('Originea din calcul', 'Origin of the calculation'), 1, 52, 1, 1))),
     'wx-pin': (slider('wp-y', s('Valoarea observată y (%; curs: 7,2)', 'Outcome y (%; lecture: 7.2)'), 6.0, 8.4, 0.1, 7.2) + play('wp'),
                charts('wp-chart') + wx('wp-ws', W, r'L_\tau(y,\hat q_\tau)=\begin{cases}\tau\,(y-\hat q_\tau), & y\ge \hat q_\tau\\(1-\tau)(\hat q_\tau-y), & y<\hat q_\tau\end{cases},\qquad \text{CRPS}\approx\tfrac{2}{3}\sum_{\tau\in\{0.1,0.5,0.9\}}L_\tau')),
     'wx-mase': (select('wm-m', s('Sezonul m din scală', 'Season m of the scale'), [('168', s('168 de ore', '168 hours') + lec), ('24', s('24 de ore', '24 hours')), ('1', s('1 oră', '1 hour'))], '168') + play('wm'),
                 charts('wm-chart') + wx('wm-ws', W, r'\text{MASE}=\frac{\frac1h\sum_{j=1}^{h}|y_{T+j}-\hat y_{T+j}|}{\frac{1}{T-m}\sum_{t=m+1}^{T}|y_t-y_{t-m}|}',
                                         slider('wm-a', s('MAE al prognozei (curs: 0,30)', 'MAE of the forecast (lecture: 0.30)'), 0.05, 1, 0.05, 0.3)
                                         + slider('wm-b', s('MAE naiv în eșantion (curs: 0,25)', 'In-sample naive MAE (lecture: 0.25)'), 0.05, 1, 0.05, 0.25))),
     'wx-cov': (select('wc-m', s('Modelul', 'Model'), [(m, m) for m in MODELS], 'Chronos-2') + select('wc-s', s('Seria din stînga', 'Series on the left'), [(k, sn[k]) for k in SER], 'retail') + play('wc'),
                charts('wc-chart') + wx('wc-ws', W, r'\text{Cov}=\frac1n\sum_{i=1}^{n}\mathbf 1\{\hat q^{(i)}_{0.1}\le y_i\le \hat q^{(i)}_{0.9}\},\qquad \bar r=(r_1r_2\cdots r_S)^{1/S}',
                                        slider('wc-k', s('Valori observate în interval (curs: 7)', 'Outcomes inside (lecture: 7)'), 0, 20, 1, 7)
                                        + slider('wc-n', s('Numărul de prognoze n (curs: 10)', 'Number of forecasts n (lecture: 10)'), 1, 20, 1, 10)
                                        + slider('wc-r', s('Raportul r₁ al primei serii (curs: 2; r₂ = 0,5)', 'Ratio r₁ of the first series (lecture: 2; r₂ = 0.5)'), 0.25, 4, 0.25, 2))),
     'wx-hour': (slider('wh-h', s('Ora h a prognozei (curs: 24)', 'Hour h of the forecast (lecture: 24)'), 1, 48, 1, 24) + play('wh'),
                 charts('wh-chart') + wx('wh-ws', W, r'L_\tau(y,\hat q_\tau)=\max\{\tau(y-\hat q_\tau),\ (\tau-1)(y-\hat q_\tau)\},\qquad \mathbf 1\{\hat q_{0.1}\le y\le \hat q_{0.9}\}')),
     'res-table': (select('rt-k', s('Scorul din harta de culori', 'Score in the heat map'), [('rel_wql', s('WQL / prognoza sezonieră naivă', 'WQL / seasonal naive') + lec), ('rel_mase', s('MASE / prognoza sezonieră naivă', 'MASE / seasonal naive')),
                                                                                       ('cov', s('acoperirea intervalului de 80%', 'coverage of the 80% interval'))], 'rel_wql') + play('rt'),
                   charts('rt-chart') + box(select('rt-v', s('Valorile din tabel', 'Values in the table'), [('rel', s('raportul față de prognoza sezonieră naivă', 'ratio to the seasonal naive') + lec), ('abs', s('WQL absolut', 'absolute WQL'))], 'rel'), tbl('rt-tb'))),
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
    """Chart blocks for build_web_lecture.py: each host carries the web version of its EXTRA blocks (moved by ch11.js)."""
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
        img = f'ch11x_{anchor}_{lang}.jpg'
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
def _js_common(lang):
    ro = lang == 'ro'
    fam = {'ro': [['Chronos (2024)', 'encoder–decoder (T5)', 'tokeni (intervale)', 'traiectorii eșantionate', 'Apache-2.0'],
                  ['Chronos-Bolt (2024)', 'encoder–decoder', 'patch-uri de 16', '9 cuantile', 'Apache-2.0'],
                  ['Chronos-2 (2025)', 'encoder', 'patch-uri + covariate', '21 de cuantile', 'Apache-2.0'],
                  ['TimesFM (2024)', 'decoder', 'patch-uri de 32', 'patch-uri de 128', 'Apache-2.0'],
                  ['Moirai (2024)', 'encoder', 'patch-uri, orice variabile', 'amestec', 'CC BY-NC 4.0'],
                  ['Lag-Llama (2023)', 'decoder (Llama)', 'valori la laguri fixe', 'Student-<i>t</i>', 'Apache-2.0'],
                  ['TimeGPT (2023)', 'encoder–decoder (puține detalii)', '–', 'puncte, intervale', 'API cu plată']],
           'en': [['Chronos (2024)', 'encoder–decoder (T5)', 'tokens (bins)', 'sampled paths', 'Apache-2.0'],
                  ['Chronos-Bolt (2024)', 'encoder–decoder', 'patches of 16', '9 quantiles', 'Apache-2.0'],
                  ['Chronos-2 (2025)', 'encoder', 'patches + covariates', '21 quantiles', 'Apache-2.0'],
                  ['TimesFM (2024)', 'decoder', 'patches of 32', 'patches of 128', 'Apache-2.0'],
                  ['Moirai (2024)', 'encoder', 'patches, any variables', 'mixture', 'CC BY-NC 4.0'],
                  ['Lag-Llama (2023)', 'decoder (Llama)', 'lagged values', 'Student-<i>t</i>', 'Apache-2.0'],
                  ['TimeGPT (2023)', 'encoder–decoder (few details)', '–', 'points, intervals', 'paid API']]}[lang]
    head = ['Model', 'Arhitectura', 'Intrarea', 'Ieșirea', 'Acces'] if ro else ['Model', 'Architecture', 'Input', 'Output', 'Access']
    return (f' fam: {json.dumps(fam, ensure_ascii=False)},\n famHead: {json.dumps(head, ensure_ascii=False)},\n'
            f' sname: {json.dumps(SNAME[lang], ensure_ascii=False)},\n')


TEXT_JS = {
 'ro': r"""(function () {
const MON = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
const de = n => { const r = Math.abs(n) % 100; return (r === 0 && n !== 0) || r >= 20 ? ' de' : ''; };
const big = n => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
const deS = s => de(+String(s).replace(/[^\d]/g, ''));
window.CH11_TEXT = {
 sep: '; ', semi: '; ', days: 'zile', mil: n => `${n}${de(+n)} milioane`, big,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 hm: t => `${+t.slice(8, 10)} ${MON[+t.slice(5, 7) - 1]} ${t.slice(0, 4)}, ora ${t.slice(11, 16)}`,
@@COMMON@@ mn: { 'Seasonal naive': 'prognoza sezonieră naivă' },
 L: { gm: 'Media geometrică', series: 'Seria', origins: 'Origini' },
 score: { rel_wql: 'raportul WQL față de prognoza sezonieră naivă', rel_mase: 'raportul MASE față de prognoza sezonieră naivă', cov: 'acoperirea intervalului de 80%' },
 famNote: 'O altă direcție dă cifrele unor LLM-uri generale (Gruver et al., 2023); Tan et al. (2024) arată că partea de model de limbaj adaugă cost, dar nu și precizie.',
 famSel: (h, n) => `${n} din cele 7 modele: ${({ apache: 'ponderi deschise sub licența Apache-2.0', patch: 'intrarea prin patch-uri', dec: 'decoder-only' })[h]}.`,
 was1: (a, b, c) => `Scorurile q<sup>⊤</sup>k<sub>j</sub>/√d: ${a}; ${b}; ${c}.`,
 was2: (a, b, c, S) => `Pasul 1, exponențiala: ${a}; ${b}; ${c}; suma ${S}.`,
 was3: (a, b, c) => `Pasul 2, împărțirea la sumă: ponderile ${a}; ${b}; ${c} (au suma 1).`,
 was4: (a, b, c, o) => `Pasul 3, ieșirea: ${a}·10 + ${b}·20 + ${c}·30 = <b>${o}</b>.`,
 wa1: (c, w, o) => `c = ${c}: cheia cu scorul cel mai mare primește ${w} din pondere; ieșirea ${o} (cu ponderi egale ar fi 20, media valorilor).`,
 tk1: (n, a, b, s, lo, hi) => `Contextul: ultimele ${n}${de(n)} luni (${a} – ${b}); scala s = ${s}; valorile scalate sînt între ${lo} și ${hi}.`,
 tk2: (nb, w, lo, hi, k) => `${nb}${de(nb)} intervale de lățime ${w} în [${lo}, ${hi}]: contextul folosește ${k === 1 ? 'un singur token' : k + (k % 100 >= 20 || k % 100 === 0 ? ' de' : '') + ' tokeni distincți'}.`,
 tk3: (t, c) => `Aprilie 2020 (carantina): tokenul ${t}, ${c === 1 ? 'folosit doar de această lună' : `comun cu încă ${c - 1}${de(c - 1)} ${c - 1 === 1 ? 'lună' : 'luni'}`}.`,
 tk3n: 'Aprilie 2020 nu este în acest context.',
 tk4: (m, a, b, dl) => `Valoarea din ${m} a fost revizuită: ${a} în slide, ${b} cu datele Eurostat din ${dl}; graficul folosește valorile din slide.`,
 wts1: (l, s) => `Pasul 1: s = media lui |x<sub>t</sub>| pentru ${l} = <b>${s}</b>.`,
 wts2: (m, x, s, z) => `Pasul 2 (${m}): x<sub>t</sub>/s = ${x}/${s} = <b>${z}</b>.`,
 wts3: (nb, w, z, r, t, c) => `Pasul 3: cu ${nb}${de(nb)} intervale, w = 30/(${nb} − 1) = ${w}; (${z} + 15)/w = ${r}, deci tokenul <b>${t}</b>, al cărui centru este ${c}.`,
 wts4: l => `Tokenii celor șase luni: ${l}.`,
 wt1: (nb, k, r) => `Cu ${nb}${de(nb)} intervale cele șase luni dau ${k === 1 ? 'un singur token' : k + ' tokeni distincți'}; o variație de 0,1 puncte a ratei mută valoarea scalată cu ${r} intervale.`,
 pa1: (n, np, P, r) => `${n}${de(n)} valori orare = ${np}${de(np)} patch-uri de cîte ${P}${r ? ` și ${r}${de(r)} valori rămase` : ''}.`,
 pa2: (a, b, r) => `Comparațiile atenției într-un strat: ${a} cu valori individuale, ${b} cu patch-uri (de ${r} ori mai puține).`,
 pa3: (P, f, wk, c, c2) => `Un patch de ${P}${de(P)} ore acoperă ${f} dintr-o zi; o săptămînă (168 de ore) cere ${wk}${de(wk)} patch-uri; un context de 2048 de valori dă ${c}${de(c)} patch-uri și ${c2}${deS(c2)} comparații.`,
 ft1: (C, n1, c1, n16, c16, n32, c32) => `C = ${C}: Chronos primește ${n1}${de(n1)} tokeni (${c1}${deS(c1)} comparații într-un strat), Chronos-Bolt și Chronos-2 ${n16}${de(n16)} patch-uri (${c16}), TimesFM ${n32}${de(n32)} patch-uri (${c32}).`,
 sc1: (y, s2, c) => `Prognoza N(0, ${s2}), valoarea observată y = ${y}: CRPS = <b>${c}</b>.`,
 sc2: (q9, a, b, c, w) => `Pierderile pinball la cuantilele prognozei (q̂<sub>0,9</sub> = ${q9}): τ = 0,1: ${a}; τ = 0,5: ${b}; τ = 0,9: ${c}; aproximarea CRPS cu cele 9 niveluri 0,1, …, 0,9 (2 × media pierderilor): ${w}.`,
 sc3: (c0, s2, cw, s4, y) => `Pentru comparație: aceeași prognoză N(0, ${s2}) dă ${c0} dacă y = 0; o bandă de două ori mai largă, N(0, ${s4}), dă ${cw} în y = ${y}.`,
 wws1: (i, o, a, b, w) => `Originea ${i} (${o}): num = Σ<sub>τ</sub>Σ<sub>t</sub> 2L<sub>τ</sub> = ${a}, den = 9 Σ<sub>t</sub>|y<sub>t</sub>| = ${b}, WQL al originii = ${w}.`,
 wws2: (n, a, b, w) => `Pe toate cele ${n}${de(n)} origini: Σ num = ${a}, Σ den = ${b}, deci WQL = <b>${w}</b>.`,
 wws3: (wn, wm, r) => `Prognoza sezonieră naivă pe aceleași origini: WQL = ${wn}; raportul ${wm}/${wn} = <b>${r}</b> (valoarea din tabelul rezultatelor).`,
 ww1: (m, k, w, r, n, sh) => `${m}, ${k}: WQL = ${w}, raportul față de prognoza sezonieră naivă ${r}; WQL-ul originii este mai mic decît cel naiv în ${sh} din cele ${n}${de(n)} origini.`,
 wps1: (t, y, q, ab, L) => ab ? `τ = ${t}: y = ${y} este deasupra lui q̂ = ${q}, pierderea ${t}·(${y} − ${q}) = <b>${L}</b>.` : `τ = ${t}: y = ${y} este sub q̂ = ${q}, pierderea (1 − ${t})·(${q} − ${y}) = <b>${L}</b>.`,
 wps2: (a, b, c, C) => `CRPS aproximat cu cele trei niveluri: (2/3)·(${a} + ${b} + ${c}) = <b>${C}</b> puncte procentuale.`,
 wps3: c => c ? 'Valoarea observată este în intervalul de 80% [6,5; 7,6]: în scorul de acoperire contează ca acoperită.' : 'Valoarea observată este în afara intervalului de 80% [6,5; 7,6]: indicatorul de acoperire este 0.',
 wp1: (y, C, t) => `y = ${y}: CRPS aproximat ${C}; cea mai mare pierdere vine de la nivelul ${t}.`,
 wms1: (a, b, r) => `Exemplul din curs: MAE ${a}, MAE naiv în eșantion ${b}, deci MASE = ${a}/${b} = <b>${r}</b>.`,
 wms2: (m, s) => `Consumul, contextul de 2048 de ore dinaintea originii de 22 iunie 2026: MAE al prognozei y<sub>t−${m}</sub> în context = ${s} GW (numitorul).`,
 wms3: (mb, s, b, mn, n) => `Chronos-Bolt small pe cele 48 de ore: MAE ${mb} GW, MASE = ${mb}/${s} = <b>${b}</b>; prognoza sezonieră naivă: MAE ${mn} GW, MASE = <b>${n}</b>.`,
 wm1: (m, b, n) => `m = ${m}: MASE ${b} pentru Chronos-Bolt small și ${n} pentru prognoza sezonieră naivă; prognozele sînt aceleași, se schimbă doar scala.`,
 wcs1: (k, n, p, kind) => `${k} din ${n} valori observate în interval: Cov = ${p}${kind === 'narrow' ? ', sub ținta de 80%: intervale prea înguste' : kind === 'wide' ? ', peste ținta de 80%: intervale prea largi' : ': exact ținta'}.`,
 wcs2: (r, g, a) => `Rapoartele ${r} și 0,5: media geometrică √(${r}·0,5) = <b>${g}</b>; media aritmetică ar fi ${a}.`,
 wc1: (m, k, c, sh, n) => `${m}, ${k}: acoperirea medie ${c} pe ${n}${de(n)} origini; ${sh} din origini au acoperirea sub 80%.`,
 wc2: (g, a) => `Pe cele șapte serii: media geometrică a rapoartelor WQL ${g}, media aritmetică ${a}.`,
 sr1: (g, l, e) => `Lungimi: de la ${g} trimestre (PIB) la ${l} ore (consumul); EUR/RON are ${e} de zile.`,
 sr2l: 'Nivelurile, ca în slide: sezonalitate puternică la consum, producția industrială, PIB și vînzările din SUA; EUR/RON și rata șomajului sînt aproape de un mers aleator.',
 sr2: (sd, l) => `Media lui |${sd ? 'y<sub>t</sub> − y<sub>t−m</sub>' : 'y<sub>t</sub> − y<sub>t−1</sub>'}| în fereastra afișată${sd ? ' (numitorul MASE)' : ''}: ${l}.`,
 sr3: (a, b, c, d, e, f, dl) => `Revizuiri după realizarea slide-urilor (datele din ${dl}): vînzările din SUA în iulie 2026 ${b} (slide ${a}), în august ${d} (slide ${c}) mld. USD; producția industrială în iulie 2026 ${f} (slide ${e}).`,
 fl1: (H, b, n, s) => `Orele 1–${H}: MASE ${b} pentru Chronos-Bolt small, ${n} pentru prognoza sezonieră naivă (scala: MAE al prognozei sezoniere naive în contextul de 2048 de ore, ${s} GW).`,
 fl2: (k, H, p, k5, ab) => `${k} din ${H}${de(H)} ore cad în banda de 80% (${p}), ${k5} în banda de 50%; valoarea observată este peste mediană în ${ab} din ${H}${de(H)} ore.`,
 whs1: (t, a, b, c, y) => `${t}: q̂<sub>0,1</sub> = ${a}, q̂<sub>0,5</sub> = ${b}, q̂<sub>0,9</sub> = ${c} GW; valoarea observată ${y} GW.`,
 whs2: (t, ab, y, q, L) => ab ? `τ = ${t}: valoarea este peste cuantilă, τ(y − q̂) = ${t}·(${y} − ${q}) = <b>${L}</b> GW.` : `τ = ${t}: valoarea este sub cuantilă, (1 − τ)(q̂ − y) = (1 − ${t})·(${q} − ${y}) = <b>${L}</b> GW.`,
 whs3: c => c ? 'Valoarea observată este în [q̂<sub>0,1</sub>, q̂<sub>0,9</sub>]: indicatorul de acoperire este 1 pentru această oră.' : 'Valoarea observată este în afara lui [q̂<sub>0,1</sub>, q̂<sub>0,9</sub>]: indicatorul de acoperire este 0 pentru această oră.',
 wh1: (h, a, b, c, i) => `Ora ${h}: pierderile ${a}; ${b}; ${c} GW la τ = 0,1; 0,5; 0,9; acoperire ${i ? 1 : 0}.`,
 fi1: (H, m, b, e, n) => `${H === 1 ? 'Luna 1 (august 2025)' : `Lunile 1–${H} (august 2025 – ${m})`}: MASE ${b} pentru Chronos-Bolt small, ${e} pentru ETS, ${n} pentru prognoza sezonieră naivă.`,
 fi2: (cb, ce, wb, we) => `Banda de 80% acoperă ${cb} din luni (Chronos-Bolt) și ${ce} (ETS); lățimea ei în ultima lună evaluată: ${wb} (Chronos-Bolt) și ${we} (ETS) puncte de indice.`,
 fi3: (a, b, dl) => `Iulie 2026: ${a} în slide; valoarea revizuită este ${b} (Eurostat, ${dl}).`,
 fe1: (H, d, m, l) => `Ziua ${H} (${d}): mediana Chronos-Bolt ${m}, ultima valoare dinaintea originii ${l}.`,
 fe2: (b, r) => `Lățimea benzii de 80%: ${b} (Chronos-Bolt) și ${r} (mersul aleator).`,
 fe3: (H, b, r) => `MAE ${H === 1 ? 'în prima zi' : `pe primele ${H}${de(H)} zile`}: ${b} (Chronos-Bolt) față de ${r} (mersul aleator).`,
 rt1: (s, l) => s === 'cov' ? `Media aritmetică pe cele șapte serii a acoperirii intervalului de 80%: ${l}.` : `Media geometrică pe cele șapte serii a raportului ${s === 'rel_wql' ? 'WQL' : 'MASE'} față de prognoza sezonieră naivă: ${l}.`,
 rt2: (w, m) => `EUR/RON, Chronos-2: raportul WQL ${w}, raportul MASE ${m}: benzile mai înguste îmbunătățesc WQL, deși mediana nu este mai bună decît prognoza naivă.`,
 rtNote: 'Sub 1: mai bun decît prognoza sezonieră naivă; fiecare valoare este WQL al modelului împărțit la cel al prognozei sezoniere naive pe aceleași origini; cel mai mic raport din fiecare rînd este îngroșat.',
 rtNoteAbs: l => `WQL absolut (fără unități, împărțit la Σ|y|); prognoza sezonieră naivă: ${l}.`,
 bm1: (f, n, k) => `${f ? 'Originile din ' + f : 'Toate originile'}: ${n}${de(n)} origini pe ${k} serii.`,
 bm2: l => `Media geometrică a raportului WQL: ${l}.`,
 bm3: (m, w, k) => `${m} este cel mai bun pe ${w} din ${k} serii.`,
 bm4: (w, m) => `EUR/RON, Chronos-2: WQL ${w}, MASE ${m}.`,
 cv1: (f, l) => `${f ? 'Originile din ' + f : 'Toate originile'}, acoperirea medie pe serii: ${l}.`,
 cv2: (a, p, b, q, c, r) => `Vînzările din SUA: de la ${p} (${a}) și ${q} (${b}) pînă la ${r} (${c}).`,
 cv3: (c, w) => `Chronos-2: acoperire medie ${c}, deși are cea mai mică medie geometrică a raportului WQL (${w}): intervale fine, dar prea înguste.`,
 hz1: (k, n, w, a, b) => `${k}: ${n}${de(n)} origini (${w === 'all' ? 'toate' : w === 'a' ? 'prima jumătate' : 'a doua jumătate'}, ${a} – ${b}).`,
 hz2: (j, a, c, n) => `Pasul ${j}: ARIMA ${a}; Chronos-2 ${c}; prognoza sezonieră naivă ${n}.`,
 hz3: j => `ETS are cea mai mare eroare dintre cele șase modele începînd cu pasul ${j}.`,
 hz3n: 'ETS nu are la niciun pas cea mai mare eroare.',
 cx1: (C, d, full) => `Context de ${C} de ore (${d} zile), ${full ? 'deci cel puțin o săptămînă întreagă: ciclul săptămînal este în context' : 'mai scurt decît o săptămînă: ciclul săptămînal lipsește din context'}.`,
 cx2: (l, n, rel) => `${rel ? 'MASE relativ la prognoza sezonieră naivă' : 'MASE'}: ${l}; prognoza sezonieră naivă ${n} (26 de origini).`,
 cx3: (m, k) => `Cel mai bun: ${m}; ${k === 0 ? 'niciun model nu bate' : k === 1 ? 'un model bate' : k + ' modele bat'} prognoza sezonieră naivă.`,
 sz1: (f, met, l, a, e) => `${f ? 'Originile din ' + f : 'Toate originile'}, media geometrică a raportului ${met}: ${l}; ARIMA ${a}; ETS ${e}.`,
 sz2: (t, s, c, a, e) => `Timpul pentru o prognoză a consumului pe procesorul unui laptop: Bolt tiny ${t} ms, Bolt small ${s} ms, Chronos-2 ${c} ms, ARIMA ${a} ms, ETS ${e} ms.`,
 pp0: (sh, a, b) => sh ? `Tăiere placebo, cu ${Math.abs(sh)} luni ${sh < 0 ? 'înainte de' : 'după'} lansare: ${a} (Chronos-Bolt small) și ${b} (Chronos-2) (rulare suplimentară).` : `Tăierea la lansarea ponderilor: ${a} (Chronos-Bolt small) și ${b} (Chronos-2).`,
 pp1: (b, nb, a, na) => `Chronos-2 pe EUR/RON: ${b} înainte (${nb}${de(nb)} origini), ${a} după (${na}${de(na)} ${na === 1 ? 'origine' : 'origini'}).`,
 pp2: (b, nb, a, na) => `Chronos-2 pe consum: ${b} înainte (${nb}${de(nb)} origini), ${a} după (${na}${de(na)} ${na === 1 ? 'origine' : 'origini'}).`,
 pp3: (b, a, eb, ea, nb, na) => `Chronos-Bolt small, vînzările din SUA: ${b} înainte, ${a} după (${nb} și ${na} origini); ETS ${eb} și ${ea}.`,
 ppn: k => `${k}: nu are origini de ambele părți ale acestei date.`,
};
})();
""",
 'en': r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const big = n => String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
const pl = (n, a, b) => (n === 1 ? a : b);
window.CH11_TEXT = {
 sep: '; ', semi: ', ', days: 'days', mil: n => `${n} million`, big,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 hm: t => `${+t.slice(8, 10)} ${MON[+t.slice(5, 7) - 1]} ${t.slice(0, 4)}, ${t.slice(11, 16)}`,
@@COMMON@@ mn: { 'Seasonal naive': 'seasonal naive' },
 L: { gm: 'Geometric mean', series: 'Series', origins: 'Origins' },
 score: { rel_wql: 'WQL ratio to the seasonal naive', rel_mase: 'MASE ratio to the seasonal naive', cov: 'coverage of the 80% interval' },
 famNote: 'A further line of work feeds the digits to general LLMs (Gruver et al., 2023); Tan et al. (2024) show that the language-model part adds cost but no accuracy.',
 famSel: (h, n) => `${n} of the 7 models: ${({ apache: 'open weights under the Apache-2.0 licence', patch: 'patch inputs', dec: 'decoder-only' })[h]}.`,
 was1: (a, b, c) => `Scores q<sup>⊤</sup>k<sub>j</sub>/√d: ${a}, ${b}, ${c}.`,
 was2: (a, b, c, S) => `Step 1, exponentiate: ${a}, ${b}, ${c}; sum ${S}.`,
 was3: (a, b, c) => `Step 2, divide by the sum: weights ${a}, ${b}, ${c} (they sum to 1).`,
 was4: (a, b, c, o) => `Step 3, output: ${a}·10 + ${b}·20 + ${c}·30 = <b>${o}</b>.`,
 wa1: (c, w, o) => `c = ${c}: the key with the highest score gets ${w} of the weight; output ${o} (with equal weights it would be 20, the mean of the values).`,
 tk1: (n, a, b, s, lo, hi) => `Context: the last ${n} months (${a} – ${b}); scale s = ${s}; the scaled values lie between ${lo} and ${hi}.`,
 tk2: (nb, w, lo, hi, k) => `${nb} bins of width ${w} in [${lo}, ${hi}]: the context uses ${k === 1 ? 'a single token' : k + ' distinct tokens'}.`,
 tk3: (t, c) => `April 2020 (the lockdown): token ${t}, ${c === 1 ? 'used by this month only' : `shared with ${c - 1} other ${pl(c - 1, 'month', 'months')}`}.`,
 tk3n: 'April 2020 is not in this context.',
 tk4: (m, a, b, dl) => `The value of ${m} has been revised: ${a} on the slide, ${b} with the Eurostat data of ${dl}; the chart uses the values of the slide.`,
 wts1: (l, s) => `Step 1: s = mean of |x<sub>t</sub>| for ${l} = <b>${s}</b>.`,
 wts2: (m, x, s, z) => `Step 2 (${m}): x<sub>t</sub>/s = ${x}/${s} = <b>${z}</b>.`,
 wts3: (nb, w, z, r, t, c) => `Step 3: with ${nb} bins, w = 30/(${nb} − 1) = ${w}; (${z} + 15)/w = ${r}, so token <b>${t}</b>, whose centre is ${c}.`,
 wts4: l => `Tokens of the six months: ${l}.`,
 wt1: (nb, k, r) => `With ${nb} bins the six months give ${k === 1 ? 'a single token' : k + ' distinct tokens'}; a change of 0.1 points in the rate moves the scaled value by ${r} bins.`,
 pa1: (n, np, P, r) => `${n} hourly values = ${np} patches of ${P}${r ? ` and ${r} values left over` : ''}.`,
 pa2: (a, b, r) => `Attention comparisons in one layer: ${a} with single values, ${b} with patches (${r} times fewer).`,
 pa3: (P, f, wk, c, c2) => `A patch of ${P} hours covers ${f} of a day; one week (168 hours) needs ${wk} patches; a context of 2048 values gives ${c} patches and ${c2} comparisons.`,
 ft1: (C, n1, c1, n16, c16, n32, c32) => `C = ${C}: Chronos receives ${n1} tokens (${c1} comparisons in one layer), Chronos-Bolt and Chronos-2 ${n16} patches (${c16}), TimesFM ${n32} patches (${c32}).`,
 sc1: (y, s2, c) => `Forecast N(0, ${s2}), outcome y = ${y}: CRPS = <b>${c}</b>.`,
 sc2: (q9, a, b, c, w) => `Pinball losses at the quantiles of the forecast (q̂<sub>0.9</sub> = ${q9}): τ = 0.1: ${a}; τ = 0.5: ${b}; τ = 0.9: ${c}; CRPS approximated with the 9 levels 0.1, …, 0.9 (2 × the mean loss): ${w}.`,
 sc3: (c0, s2, cw, s4, y) => `For comparison: the same forecast N(0, ${s2}) gives ${c0} if y = 0; a band twice as wide, N(0, ${s4}), gives ${cw} at y = ${y}.`,
 wws1: (i, o, a, b, w) => `Origin ${i} (${o}): num = Σ<sub>τ</sub>Σ<sub>t</sub> 2L<sub>τ</sub> = ${a}, den = 9 Σ<sub>t</sub>|y<sub>t</sub>| = ${b}, WQL of the origin = ${w}.`,
 wws2: (n, a, b, w) => `Over all ${n} origins: Σ num = ${a}, Σ den = ${b}, so WQL = <b>${w}</b>.`,
 wws3: (wn, wm, r) => `Seasonal naive on the same origins: WQL = ${wn}; the ratio ${wm}/${wn} = <b>${r}</b> (the value of the results table).`,
 ww1: (m, k, w, r, n, sh) => `${m}, ${k}: WQL = ${w}, ratio to the seasonal naive ${r}; the WQL of the origin is below that of the naive forecast at ${sh} of the ${n} origins.`,
 wps1: (t, y, q, ab, L) => ab ? `τ = ${t}: y = ${y} is above q̂ = ${q}, loss ${t}·(${y} − ${q}) = <b>${L}</b>.` : `τ = ${t}: y = ${y} is below q̂ = ${q}, loss (1 − ${t})·(${q} − ${y}) = <b>${L}</b>.`,
 wps2: (a, b, c, C) => `Approximate CRPS with these three levels: (2/3)·(${a} + ${b} + ${c}) = <b>${C}</b> percentage points.`,
 wps3: c => c ? 'The outcome lies inside the 80% interval [6.5; 7.6]: it counts as covered in the coverage score.' : 'The outcome lies outside the 80% interval [6.5; 7.6]: the coverage indicator is 0.',
 wp1: (y, C, t) => `y = ${y}: approximate CRPS ${C}; the largest loss comes from level ${t}.`,
 wms1: (a, b, r) => `The example of the lecture: MAE ${a}, in-sample naive MAE ${b}, so MASE = ${a}/${b} = <b>${r}</b>.`,
 wms2: (m, s) => `Load, the 2048-hour context before the origin of 22 June 2026: MAE of the forecast y<sub>t−${m}</sub> inside the context = ${s} GW (the denominator).`,
 wms3: (mb, s, b, mn, n) => `Chronos-Bolt small over the 48 hours: MAE ${mb} GW, MASE = ${mb}/${s} = <b>${b}</b>; seasonal naive: MAE ${mn} GW, MASE = <b>${n}</b>.`,
 wm1: (m, b, n) => `m = ${m}: MASE ${b} for Chronos-Bolt small and ${n} for the seasonal naive; the forecasts are the same, only the scale changes.`,
 wcs1: (k, n, p, kind) => `${k} of ${n} outcomes inside: Cov = ${p}${kind === 'narrow' ? ', below the 80% target: intervals too narrow' : kind === 'wide' ? ', above the 80% target: intervals too wide' : ': exactly the target'}.`,
 wcs2: (r, g, a) => `Ratios ${r} and 0.5: geometric mean √(${r}·0.5) = <b>${g}</b>; the arithmetic mean would be ${a}.`,
 wc1: (m, k, c, sh, n) => `${m}, ${k}: mean coverage ${c} over ${n} origins; ${sh} of the origins have coverage below 80%.`,
 wc2: (g, a) => `Over the seven series: geometric mean of the WQL ratios ${g}, arithmetic mean ${a}.`,
 sr1: (g, l, e) => `Lengths from ${g} quarters (GDP) to ${l} hours (load); EUR/RON has ${e} days.`,
 sr2l: 'The levels, as on the slide: strong seasonality in load, industrial production, GDP and US retail sales; EUR/RON and the unemployment rate are close to a random walk.',
 sr2: (sd, l) => `Mean of |${sd ? 'y<sub>t</sub> − y<sub>t−m</sub>' : 'y<sub>t</sub> − y<sub>t−1</sub>'}| over the window shown${sd ? ' (the denominator of the MASE)' : ''}: ${l}.`,
 sr3: (a, b, c, d, e, f, dl) => `Revisions since the slides were made (data of ${dl}): US retail sales in July 2026 ${b} (slide ${a}), in August ${d} (slide ${c}) bn USD; industrial production in July 2026 ${f} (slide ${e}).`,
 fl1: (H, b, n, s) => `Hours 1–${H}: MASE ${b} for Chronos-Bolt small, ${n} for the seasonal naive (scale: the MAE of the seasonal naive inside the 2048-hour context, ${s} GW).`,
 fl2: (k, H, p, k5, ab) => `${k} of ${H} hours fall in the 80% band (${p}), ${k5} in the 50% band; the outcome is above the median in ${ab} of ${H} hours.`,
 whs1: (t, a, b, c, y) => `${t}: q̂<sub>0.1</sub> = ${a}, q̂<sub>0.5</sub> = ${b}, q̂<sub>0.9</sub> = ${c} GW; outcome ${y} GW.`,
 whs2: (t, ab, y, q, L) => ab ? `τ = ${t}: the outcome is above the quantile, τ(y − q̂) = ${t}·(${y} − ${q}) = <b>${L}</b> GW.` : `τ = ${t}: the outcome is below the quantile, (1 − τ)(q̂ − y) = (1 − ${t})·(${q} − ${y}) = <b>${L}</b> GW.`,
 whs3: c => c ? 'The outcome lies in [q̂<sub>0.1</sub>, q̂<sub>0.9</sub>]: coverage indicator 1 for this hour.' : 'The outcome lies outside [q̂<sub>0.1</sub>, q̂<sub>0.9</sub>]: coverage indicator 0 for this hour.',
 wh1: (h, a, b, c, i) => `Hour ${h}: losses ${a}, ${b}, ${c} GW at τ = 0.1, 0.5, 0.9; coverage ${i ? 1 : 0}.`,
 fi1: (H, m, b, e, n) => `${H === 1 ? 'Month 1 (August 2025)' : `Months 1–${H} (August 2025 – ${m})`}: MASE ${b} for Chronos-Bolt small, ${e} for ETS, ${n} for the seasonal naive.`,
 fi2: (cb, ce, wb, we) => `The 80% band covers ${cb} of the months (Chronos-Bolt) and ${ce} (ETS); its width in the last month scored: ${wb} (Chronos-Bolt) and ${we} (ETS) index points.`,
 fi3: (a, b, dl) => `July 2026: ${a} on the slide; the revised value is ${b} (Eurostat, ${dl}).`,
 fe1: (H, d, m, l) => `Day ${H} (${d}): median of Chronos-Bolt ${m}, last value before the origin ${l}.`,
 fe2: (b, r) => `Width of the 80% band: ${b} (Chronos-Bolt) and ${r} (random walk).`,
 fe3: (H, b, r) => `MAE ${H === 1 ? 'on the first day' : `over the first ${H} days`}: ${b} (Chronos-Bolt) against ${r} (random walk).`,
 rt1: (s, l) => s === 'cov' ? `Mean over the seven series of the coverage of the 80% interval: ${l}.` : `Geometric mean over the seven series of the ${s === 'rel_wql' ? 'WQL' : 'MASE'} ratio to the seasonal naive: ${l}.`,
 rt2: (w, m) => `EUR/RON, Chronos-2: WQL ratio ${w}, MASE ratio ${m}: the narrower bands improve the WQL although the median is no better than the naive forecast.`,
 rtNote: 'Below 1: better than the seasonal naive; each value is the WQL of the model divided by that of the seasonal naive on the same origins; the smallest ratio of each row is in bold.',
 rtNoteAbs: l => `Absolute WQL (unit-free, divided by Σ|y|); seasonal naive: ${l}.`,
 bm1: (f, n, k) => `${f ? 'Origins from ' + f : 'All origins'}: ${n} origins on ${k} series.`,
 bm2: l => `Geometric mean of the WQL ratio: ${l}.`,
 bm3: (m, w, k) => `${m} is best on ${w} of ${k} series.`,
 bm4: (w, m) => `EUR/RON, Chronos-2: WQL ${w}, MASE ${m}.`,
 cv1: (f, l) => `${f ? 'Origins from ' + f : 'All origins'}, mean coverage over the series: ${l}.`,
 cv2: (a, p, b, q, c, r) => `US retail sales: from ${p} (${a}) and ${q} (${b}) to ${r} (${c}).`,
 cv3: (c, w) => `Chronos-2: mean coverage ${c}, although it has the smallest geometric mean of the WQL ratio (${w}): sharp intervals, but too narrow.`,
 hz1: (k, n, w, a, b) => `${k}: ${n} origins (${w === 'all' ? 'all' : w === 'a' ? 'first half' : 'second half'}, ${a} – ${b}).`,
 hz2: (j, a, c, n) => `Step ${j}: ARIMA ${a}, Chronos-2 ${c}, seasonal naive ${n}.`,
 hz3: j => `ETS has the largest error of the six models from step ${j} on.`,
 hz3n: 'ETS never has the largest error.',
 cx1: (C, d, full) => `Context of ${C} hours (${d} days), ${full ? 'so at least one full week: the weekly cycle is in the context' : 'shorter than a week: the weekly cycle is missing from the context'}.`,
 cx2: (l, n, rel) => `${rel ? 'MASE relative to the seasonal naive' : 'MASE'}: ${l}; seasonal naive ${n} (26 origins).`,
 cx3: (m, k) => `Best: ${m}; ${k === 0 ? 'no model beats' : k === 1 ? 'one model beats' : k + ' models beat'} the seasonal naive.`,
 sz1: (f, met, l, a, e) => `${f ? 'Origins from ' + f : 'All origins'}, geometric mean of the ${met} ratio: ${l}; ARIMA ${a}, ETS ${e}.`,
 sz2: (t, s, c, a, e) => `Time per load forecast on a laptop CPU: Bolt tiny ${t} ms, Bolt small ${s} ms, Chronos-2 ${c} ms, ARIMA ${a} ms, ETS ${e} ms.`,
 pp0: (sh, a, b) => sh ? `Placebo cut, ${Math.abs(sh)} months ${sh < 0 ? 'before' : 'after'} the release: ${a} (Chronos-Bolt small) and ${b} (Chronos-2) (extra run).` : `Cut at the release of the weights: ${a} (Chronos-Bolt small) and ${b} (Chronos-2).`,
 pp1: (b, nb, a, na) => `Chronos-2 on EUR/RON: ${b} before (${nb} origins), ${a} after (${na} ${pl(na, 'origin', 'origins')}).`,
 pp2: (b, nb, a, na) => `Chronos-2 on load: ${b} before (${nb} origins), ${a} after (${na} ${pl(na, 'origin', 'origins')}).`,
 pp3: (b, a, eb, ea, nb, na) => `Chronos-Bolt small, US retail sales: ${b} before, ${a} after (${nb} and ${na} origins); ETS ${eb} and ${ea}.`,
 ppn: k => `${k}: no origins on both sides of this date.`,
};
})();
""",
}
for _lg in ('ro', 'en'):
    TEXT_JS[_lg] = TEXT_JS[_lg].replace('@@COMMON@@', _js_common(_lg))

QA = {
 'ro': {
  'tokens': ('▶ scurtează contextul la 24 de luni (august 2024 – iulie 2026). Cîți tokeni distincți folosește contextul pe grila grosieră de 21 de intervale?',
             'Doi (față de trei pentru 96 de luni): fără aprilie 2020, valorile scalate rămîn între 0,87 și 1,11, adică în două intervale de lățime 0,30. Scala se adaptează contextului (s = 94,1 în loc de 97,8), deci tokenii descriu forma relativă la nivelul recent, nu nivelul.'),
  'patching': ('Puneți P = 24 (un patch pe zi). Cîte patch-uri cere o săptămînă (168 de ore)?',
               '7 = 168/24, față de 11 pentru P = 16 (168/16 = 10,5, rotunjit în sus). Cu P = 24 patch-urile se aliniază cu ciclul zilnic, iar cele 336 de ore dau 14 patch-uri; comparațiile atenției scad de la 441 la 196.'),
  'scores': ('Mutați valoarea observată la y = 0, mediana prognozei N(0, 1). Cît este CRPS?',
             '0,234, nu 0: CRPS evaluează toată distribuția, iar o prognoză cu abaterea standard 1 rămîne incertă chiar dacă mediana nimerește valoarea. Doar o prognoză care pune toată masa în y are CRPS 0; în y = 1 scorul este 0,602.'),
  'series': ('▶ trece la diferențele sezoniere și apoi la primele diferențe. Pentru consumul orar din iunie 2026, media lui |y<sub>t</sub> − y<sub>t−168</sub>| este mai mare sau mai mică decît media lui |y<sub>t</sub> − y<sub>t−1</sub>|?',
             'Mai mare: 0,43 GW față de 0,24 GW. Consumul variază puțin de la o oră la alta, iar într-o săptămînă se schimbă vremea și nivelul; de aceea numitorul MASE cu m = 168 este mai mare decît eroarea prognozei de o oră. La producția industrială este invers (4,86 cu m = 12, 6,37 cu m = 1), pentru că sezonalitatea anuală domină.'),
  'fan-load': ('▶ evaluează doar primele 6 ore. Ce acoperire are banda de 80% a Chronos-Bolt în orele 1–6?',
               '100% (6 din 6 ore), față de 29% pe 48 de ore; MASE este 0,11 în primele 6 ore. Prognoza pornește de la ultimele valori observate, iar abaterea consumului efectiv se acumulează în cursul zilei: acoperirea unei singure origini depinde mult de orizont.'),
  'fan-ip': ('Puneți H = 3 (august–octombrie 2025). Care prognoză are cel mai mic MASE?',
             'Prognoza sezonieră naivă: 0,43, față de 0,79 pentru ETS și 0,95 pentru Chronos-Bolt. Pentru această origine, anul precedent prognozează cel mai bine lunile următoare; pe 12 luni prognoza naivă rămîne prima (0,67), iar ETS (1,20) și Chronos-Bolt (1,04) își schimbă locurile.'),
  'fan-eurron': ('Puneți H = 5. La ziua 5, banda de 80% a Chronos-Bolt este mai largă sau mai îngustă decît cea a mersului aleator?',
                 'Mai largă: 0,044 față de 0,031. Banda mersului aleator crește cu √h, de la 0,014 în prima zi, în timp ce banda Chronos-Bolt pornește de la 0,035; cele două ajung aproape egale abia la 20 de zile (0,064 și 0,063).'),
  'benchmark': ('Păstrați doar originile din 2024. Care model are cea mai mică medie geometrică a raportului WQL?',
                'Tot Chronos-2 (0,78), urmat de ARIMA (0,92); Chronos-Bolt small trece la 1,06, mai slab decît prognoza sezonieră naivă, iar ETS la 1,01. Cu 175 de origini în loc de 708, ordinea celorlalte modele se schimbă: un singur clasament nu descrie toate perioadele.'),
  'coverage': ('Păstrați doar originile din 2024. Cît este acoperirea medie a prognozei sezoniere naive?',
               '94%: intervalele naive devin prea largi, pentru că abaterea lor standard este estimată pe tot contextul, care include șocurile din 2020–2022. Chronos-2 ajunge la 83%: acoperirea depinde de perioada testată, nu doar de model.'),
  'horizon': ('Alegeți a doua jumătate a originilor (ianuarie–iunie 2026). De la ce oră ETS are, pînă la ora 48, cea mai mare eroare dintre cele șase modele?',
              'De la ora 41, față de ora 4 pe toate originile. În a doua jumătate erorile tuturor modelelor sînt mai mari (prognoza naivă 0,32 GW la ora 24, față de 0,19 în prima jumătate), iar Chronos-2 rămîne la 0,15 GW.'),
  'context': ('▶ trece la 192 de ore (8 zile). Cîte dintre cele trei foundation models bat acum prognoza sezonieră naivă (MASE 1,16)?',
              'Toate trei: Chronos-Bolt tiny 1,09, Chronos-Bolt small 0,98, Chronos-2 1,04. Cu 96 de ore niciunul nu reușea; contextul de 8 zile conține primul ciclu săptămînal complet. Chronos-2, cel mai bun cu 2048 de ore (0,63), este aici abia al doilea: el are nevoie de cel mai lung context.'),
  'size': ('Păstrați doar originile din 2022. Chronos-Bolt small (48 de milioane de parametri) mai bate prognoza sezonieră naivă?',
           'Nu: media geometrică a raportului WQL devine 1,04 (0,90 pe toate originile), în timp ce Chronos-2 rămîne la 0,83 și ARIMA la 0,91. Numărul de parametri nu garantează un cîștig; Chronos-2 diferă și prin arhitectură și date de antrenare.'),
  'prepost': ('Alegeți tăierea placebo cu 6 luni înainte de lansare (25 mai 2024 pentru Chronos-Bolt small). Cît devine raportul WQL al Chronos-Bolt small pe vînzările din SUA după tăiere?',
              '1,08, ca după lansarea reală (0,81 înainte): saltul apare și la o dată la care nu s-a întîmplat nimic. Diferența dintre perioade, nu neapărat contaminarea, poate produce un astfel de salt; de aceea o scădere este un semnal de alarmă, nu o dovadă.'),
  'wx-att': ('▶ crește c pînă la 3 (scorurile 6, 3, 0). Ce se întîmplă cu ieșirea?',
             'Scade la 10,5, aproape de valoarea primei chei (10): ponderea ei crește la 95%. Cu c = 0 toate ponderile sînt 1/3 și ieșirea este media, 20; împărțirea la √d ține scorurile moderate, astfel încît și celelalte poziții contribuie.'),
  'wx-tok': ('Alegeți 21 de intervale în [−15, 15]. Cîți tokeni distincți dau cele șase luni?',
             'Unul singur: intervalele au lățimea 1,5, iar toate valorile scalate (de la 0,970 la 1,030) cad în același interval. Cu 4093 de intervale (lățimea 0,0073) sînt 4 tokeni; o variație de 0,1 puncte a ratei mută valoarea scalată cu 2,07 intervale.'),
  'fam-table': ('Puneți C = 512. Cîte patch-uri de 16 primește Chronos-Bolt?',
                '32 = 512/16, cu 32² = 1024 de comparații într-un strat de atenție, față de 512² = 262 144 pentru un token pe valoare (Chronos). TimesFM, cu patch-uri de 32, primește 16.'),
  'wx-wql': ('▶ trece la EUR/RON. Cît este raportul WQL al Chronos-2 față de prognoza sezonieră naivă?',
             '0,87: WQL 0,0019 față de 0,0022 pentru prognoza naivă, deși raportul MASE al medianei este 1,05. WQL evaluează toate cuantilele, iar benzile mai înguste ale Chronos-2 cîștigă chiar cînd mediana nu este mai bună.'),
  'wx-pin': ('Mutați y la 8,0, peste q̂<sub>0,9</sub> = 7,6. Care nivel are cea mai mare pierdere?',
             'Nivelul 0,5: 0,5·(8,0 − 7,0) = 0,50, față de 0,9·(8,0 − 7,6) = 0,36 și 0,1·(8,0 − 6,5) = 0,15. Depășirea cuantilei de 0,9 costă mult pe unitate, dar distanța pînă la ea este mică; CRPS aproximat crește la 0,673.'),
  'wx-mase': ('Alegeți m = 1 (prognoza naivă de o oră în numitor). Cît devine MASE pentru Chronos-Bolt small?',
              '2,13 (1,31 cu m = 168): numitorul scade de la 0,421 GW la 0,260 GW, eroarea medie a prognozei de o oră în context, iar prognozele rămîn aceleași. Valoarea MASE și pragul 1 au sens doar împreună cu m-ul declarat.'),
  'wx-cov': ('▶ trece prin modele. Ce acoperire are Chronos-Bolt tiny pe vînzările din SUA?',
             '75,2%, cea mai mare dintre cele șase modele, dar tot sub 80%; Chronos-2 are 65,0%, prognoza sezonieră naivă 64,7%. Șocul din 2020 și saltul din 2021 ies din benzile tuturor modelelor.'),
  'wx-hour': ('Mutați ora la 1 (22 iunie 2026, ora 00:00). Cît este indicatorul de acoperire?',
              '1: valoarea observată este în banda de 80%, iar pierderile sînt mici (0,017; 0,010; 0,012 GW). Prognoza pornește de la ultimele valori; abia în cursul zilei consumul efectiv iese peste bandă, ca la ora 24 (0,062; 0,163; 0,025 GW).'),
  'res-table': ('Treceți harta la raportul MASE. Pe ce serie raportul MASE al Chronos-2 este peste 1?',
                'EUR/RON: 1,05, deși raportul WQL este 0,87. Pe celelalte șase serii raportul MASE al Chronos-2 este sub 1, de la 0,59 (vînzările din SUA) pînă la 0,98 (șomajul).'),
 },
 'en': {
  'tokens': ('▶ shortens the context to 24 months (August 2024 – July 2026). How many distinct tokens does the context use on the coarse grid of 21 bins?',
             'Two (three for 96 months): without April 2020 the scaled values stay between 0.87 and 1.11, that is in two bins of width 0.30. The scale adapts to the context (s = 94.1 instead of 97.8), so the tokens describe the shape relative to the recent level, not the level.'),
  'patching': ('Set P = 24 (one patch per day). How many patches does one week (168 hours) need?',
               '7 = 168/24, against 11 for P = 16 (168/16 = 10.5, rounded up). With P = 24 the patches align with the daily cycle and the 336 hours give 14 patches; the attention comparisons fall from 441 to 196.'),
  'scores': ('Move the outcome to y = 0, the median of the forecast N(0, 1). What is the CRPS?',
             '0.234, not 0: the CRPS scores the whole distribution, and a forecast with standard deviation 1 remains uncertain even when its median hits the outcome. Only a forecast that puts all its mass on y has CRPS 0; at y = 1 the score is 0.602.'),
  'series': ('▶ switches to seasonal and then to first differences. For the hourly load of June 2026, is the mean of |y<sub>t</sub> − y<sub>t−168</sub>| larger or smaller than the mean of |y<sub>t</sub> − y<sub>t−1</sub>|?',
             'Larger: 0.43 GW against 0.24 GW. Load changes little from one hour to the next, while within a week the weather and the level change; so the denominator of the MASE with m = 168 is larger than the error of the one-hour forecast. For industrial production it is the other way round (4.86 with m = 12, 6.37 with m = 1), because the yearly seasonality dominates.'),
  'fan-load': ('▶ scores only the first 6 hours. What is the coverage of the 80% band of Chronos-Bolt in hours 1–6?',
               '100% (6 of 6 hours), against 29% over 48 hours; the MASE is 0.11 in the first 6 hours. The forecast starts from the last observed values and the gap to the actual load builds up during the day: the coverage of a single origin depends strongly on the horizon.'),
  'fan-ip': ('Set H = 3 (August–October 2025). Which forecast has the lowest MASE?',
             'The seasonal naive: 0.43, against 0.79 for ETS and 0.95 for Chronos-Bolt. For this origin the previous year forecasts the next months best; over 12 months the naive forecast stays first (0.67), while ETS (1.20) and Chronos-Bolt (1.04) swap places.'),
  'fan-eurron': ('Set H = 5. At day 5, is the 80% band of Chronos-Bolt wider or narrower than that of the random walk?',
                 'Wider: 0.044 against 0.031. The band of the random walk grows with √h, from 0.014 on the first day, while the band of Chronos-Bolt starts at 0.035; the two become almost equal only at 20 days (0.064 and 0.063).'),
  'benchmark': ('Keep only the origins from 2024. Which model has the smallest geometric mean of the WQL ratio?',
                'Still Chronos-2 (0.78), followed by ARIMA (0.92); Chronos-Bolt small moves to 1.06, worse than the seasonal naive, and ETS to 1.01. With 175 origins instead of 708 the order of the other models changes: one ranking does not describe every period.'),
  'coverage': ('Keep only the origins from 2024. What is the mean coverage of the seasonal naive?',
               '94%: the naive intervals become too wide, because their standard deviation is estimated on the whole context, which contains the shocks of 2020–2022. Chronos-2 reaches 83%: coverage depends on the test period, not only on the model.'),
  'horizon': ('Choose the second half of the origins (January–June 2026). From which hour on, up to hour 48, does ETS have the largest error of the six models?',
              'From hour 41, against hour 4 over all origins. In the second half every model has larger errors (the naive forecast 0.32 GW at hour 24, against 0.19 in the first half), while Chronos-2 stays at 0.15 GW.'),
  'context': ('▶ moves to 192 hours (8 days). How many of the three foundation models now beat the seasonal naive (MASE 1.16)?',
              'All three: Chronos-Bolt tiny 1.09, Chronos-Bolt small 0.98, Chronos-2 1.04. With 96 hours none of them did; the context of 8 days contains the first full weekly cycle. Chronos-2, the best with 2048 hours (0.63), is only second here: it needs the longest context.'),
  'size': ('Keep only the origins from 2022. Does Chronos-Bolt small (48 million parameters) still beat the seasonal naive?',
           'No: the geometric mean of the WQL ratio becomes 1.04 (0.90 over all origins), while Chronos-2 stays at 0.83 and ARIMA at 0.91. The number of parameters does not guarantee a gain; Chronos-2 also differs in architecture and training data.'),
  'prepost': ('Choose the placebo cut 6 months before the release (25 May 2024 for Chronos-Bolt small). What does the WQL ratio of Chronos-Bolt small on US retail sales become after the cut?',
              '1.08, as after the real release (0.81 before): the jump also appears at a date at which nothing happened. The difference between the periods, not necessarily contamination, can produce such a jump; this is why a drop is a warning, not a proof.'),
  'wx-att': ('▶ raises c to 3 (scores 6, 3, 0). What happens to the output?',
             'It falls to 10.5, close to the value of the first key (10): its weight rises to 95%. With c = 0 all weights are 1/3 and the output is the mean, 20; dividing by √d keeps the scores moderate, so that the other positions also contribute.'),
  'wx-tok': ('Choose 21 bins in [−15, 15]. How many distinct tokens do the six months give?',
             'A single one: the bins have width 1.5 and all scaled values (from 0.970 to 1.030) fall in the same bin. With 4093 bins (width 0.0073) there are 4 tokens; a change of 0.1 points in the rate moves the scaled value by 2.07 bins.'),
  'fam-table': ('Set C = 512. How many patches of 16 does Chronos-Bolt receive?',
                '32 = 512/16, with 32² = 1,024 comparisons in one attention layer, against 512² = 262,144 for one token per value (Chronos). TimesFM, with patches of 32, receives 16.'),
  'wx-wql': ('▶ switches to EUR/RON. What is the WQL ratio of Chronos-2 to the seasonal naive?',
             '0.87: WQL 0.0019 against 0.0022 for the naive forecast, although the MASE ratio of the median is 1.05. The WQL scores all quantiles, and the narrower bands of Chronos-2 win even when the median is no better.'),
  'wx-pin': ('Move y to 8.0, above q̂<sub>0.9</sub> = 7.6. Which level has the largest loss?',
             'Level 0.5: 0.5·(8.0 − 7.0) = 0.50, against 0.9·(8.0 − 7.6) = 0.36 and 0.1·(8.0 − 6.5) = 0.15. Exceeding the 0.9 quantile costs a lot per unit, but the distance to it is small; the approximate CRPS rises to 0.673.'),
  'wx-mase': ('Choose m = 1 (the one-hour naive forecast in the denominator). What does the MASE of Chronos-Bolt small become?',
              '2.13 (1.31 with m = 168): the denominator falls from 0.421 GW to 0.260 GW, the mean error of the one-hour forecast inside the context, while the forecasts stay the same. The value of the MASE and the threshold 1 make sense only together with the stated m.'),
  'wx-cov': ('▶ cycles through the models. What is the coverage of Chronos-Bolt tiny on US retail sales?',
             '75.2%, the highest of the six models, but still below 80%; Chronos-2 has 65.0%, the seasonal naive 64.7%. The 2020 shock and the 2021 jump fall outside the bands of every model.'),
  'wx-hour': ('Move to hour 1 (22 June 2026, 00:00). What is the coverage indicator?',
              '1: the outcome lies in the 80% band and the losses are small (0.017, 0.010, 0.012 GW). The forecast starts from the last values; only during the day does the actual load rise above the band, as at hour 24 (0.062, 0.163, 0.025 GW).'),
  'res-table': ('Switch the heat map to the MASE ratio. On which series is the MASE ratio of Chronos-2 above 1?',
                'EUR/RON: 1.05, although the WQL ratio is 0.87. On the other six series the MASE ratio of Chronos-2 is below 1, from 0.59 (US retail sales) to 0.98 (unemployment).'),
 },
}

# Ce credeți? / What do you think?: (index of the correct option, [(RO, EN), ...]); keys checked against the page (node)
PREDICT = {
 'tokens': (0, [('2: fără aprilie 2020 valorile scalate ocupă doar două intervale.', '2: without April 2020 the scaled values span only two bins.'),
                ('3, ca pentru 96 de luni: grila nu depinde de context.', '3, as for 96 months: the grid does not depend on the context.'), ('24: un token pentru fiecare lună.', '24: one token per month.')]),
 'patching': (0, [('7.', '7.'), ('11, ca pentru P = 16: săptămîna are tot 168 de ore.', '11, as for P = 16: the week still has 168 hours.'), ('168: un patch pentru fiecare oră.', '168: one patch per hour.')]),
 'scores': (0, [('0,234.', '0.234.'), ('0: mediana nimerește valoarea observată.', '0: the median hits the outcome.'), ('0,602: CRPS nu depinde de valoarea observată.', '0.602: the CRPS does not depend on the outcome.')]),
 'series': (0, [('Mai mare: 0,43 față de 0,24 GW.', 'Larger: 0.43 against 0.24 GW.'), ('Mai mică: ciclul săptămînal se repetă, deci diferența sezonieră este aproape zero.', 'Smaller: the weekly cycle repeats, so the seasonal difference is close to zero.'),
                ('Egale: ambele sînt diferențe ale aceleiași serii.', 'Equal: both are differences of the same series.')]),
 'fan-load': (0, [('100%.', '100%.'), ('29%, ca pe 48 de ore: acoperirea nu depinde de orizont.', '29%, as over 48 hours: coverage does not depend on the horizon.'), ('80%: banda este construită pentru 80%.', '80%: the band is built for 80%.')]),
 'fan-ip': (0, [('Prognoza sezonieră naivă.', 'The seasonal naive.'), ('Chronos-Bolt: este pre-antrenat pe multe serii.', 'Chronos-Bolt: it is pretrained on many series.'), ('ETS: este estimat pe această serie.', 'ETS: it is estimated on this series.')]),
 'fan-eurron': (0, [('Mai largă: 0,044 față de 0,031.', 'Wider: 0.044 against 0.031.'), ('Aproape aceeași, ca la 20 de zile.', 'Almost the same, as at 20 days.'), ('Mai îngustă: modelul a învățat din context.', 'Narrower: the model has learnt from the context.')]),
 'benchmark': (0, [('Chronos-2.', 'Chronos-2.'), ('Chronos-Bolt small: originile recente sînt după lansarea lui.', 'Chronos-Bolt small: the recent origins come after its release.'),
                   ('ARIMA: modelele reestimate profită de datele recente.', 'ARIMA: the re-estimated models profit from the recent data.')]),
 'coverage': (0, [('94%.', '94%.'), ('83%, ca pe toate originile.', '83%, as over all origins.'), ('76%, ca la Chronos-2.', '76%, as for Chronos-2.')]),
 'horizon': (0, [('De la ora 41.', 'From hour 41.'), ('De la ora 4, ca pe toate originile.', 'From hour 4, as over all origins.'), ('De la ora 1: ETS are doar sezonul zilnic.', 'From hour 1: ETS has a daily season only.')]),
 'context': (0, [('Toate trei.', 'All three.'), ('Niciunul, ca la 96 de ore: e nevoie de un context mult mai lung.', 'None, as at 96 hours: a much longer context is needed.'), ('Unul: doar Chronos-2, cel mai mare model.', 'One: only Chronos-2, the largest model.')]),
 'size': (0, [('Nu: 1,04.', 'No: 1.04.'), ('Da: 0,90, ca pe toate originile.', 'Yes: 0.90, as over all origins.'), ('Da, și este mai bun decît ARIMA: modelele mai mari cîștigă mereu.', 'Yes, and it beats ARIMA: larger models always win.')]),
 'prepost': (0, [('1,08: saltul apare și la data placebo.', '1.08: the jump appears at the placebo date too.'), ('0,82, ca înainte de lansare: fără lansare nu există salt.', '0.82, as before the release: without a release there is no jump.'),
                 ('Nu se poate calcula: nu există origini după această dată.', 'It cannot be computed: there are no origins after this date.')]),
 'wx-att': (0, [('Scade spre 10, valoarea cheii cu scorul cel mai mare.', 'It falls towards 10, the value of the key with the highest score.'), ('Rămîne 14,2: softmax normalizează ponderile.', 'It stays at 14.2: softmax normalises the weights.'),
                ('Crește spre 20.', 'It rises towards 20.')]),
 'wx-tok': (0, [('1.', '1.'), ('4, ca la 4093 de intervale.', '4, as with 4093 bins.'), ('6: un token pentru fiecare lună.', '6: one token per month.')]),
 'fam-table': (0, [('32.', '32.'), ('128, ca pentru 2048 de valori.', '128, as for 2048 values.'), ('512: un patch pentru fiecare valoare.', '512: one patch per value.')]),
 'wx-wql': (0, [('0,87.', '0.87.'), ('1,05: același raport ca la MASE.', '1.05: the same ratio as for the MASE.'), ('0,58, ca la consum.', '0.58, as for load.')]),
 'wx-pin': (0, [('Nivelul 0,5: pierderea 0,50.', 'Level 0.5: loss 0.50.'), ('Nivelul 0,9: valoarea a depășit cuantila de 0,9.', 'Level 0.9: the outcome exceeded the 0.9 quantile.'), ('Nivelul 0,1: cuantila cea mai îndepărtată.', 'Level 0.1: the most distant quantile.')]),
 'wx-mase': (0, [('2,13.', '2.13.'), ('1,31: prognozele nu se schimbă.', '1.31: the forecasts do not change.'), ('Sub 1: o scală mai fină favorizează modelul.', 'Below 1: a finer scale favours the model.')]),
 'wx-cov': (0, [('75,2%.', '75.2%.'), ('65,0%, ca la Chronos-2.', '65.0%, as for Chronos-2.'), ('80%: modelele zero-shot sînt calibrate.', '80%: zero-shot models are calibrated.')]),
 'wx-hour': (0, [('1.', '1.'), ('0, ca la ora 24: consumul este peste prognoză toată ziua.', '0, as at hour 24: load is above the forecast all day.'), ('Nu se poate calcula pentru o singură oră.', 'It cannot be computed for a single hour.')]),
 'res-table': (0, [('EUR/RON.', 'EUR/RON.'), ('Șomajul României: toate modelele sînt aproape de 1.', 'Romanian unemployment: all models are close to 1.'), ('Niciuna: Chronos-2 are cel mai bun WQL.', 'None: Chronos-2 has the best WQL.')]),
}
VEIL_CHARTS = set()

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slides „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('Care este diferența dintre prognoza zero-shot și fine-tuning?', 'What is the difference between zero-shot forecasting and fine-tuning?'), key=0, opts=[
  (('Zero-shot păstrează fixe ponderile pre-antrenate și folosește doar contextul; fine-tuning-ul antrenează în continuare ponderile pe datele-țintă.',
    'Zero-shot keeps the pretrained weights fixed and uses only the context; fine-tuning trains the weights further on the target data.'),
   ('Așa le definește cursul: zero-shot nu estimează niciun parametru pe seria-țintă.', 'This is how the lecture defines them: zero-shot estimates no parameter on the target series.')),
  (('Zero-shot nu folosește deloc datele seriei-țintă.', 'Zero-shot does not use the data of the target series at all.'),
   ('Zero-shot folosește contextul seriei-țintă; doar ponderile rămîn fixe.', 'Zero-shot uses the context of the target series; only the weights stay fixed.')),
  (('Fine-tuning-ul schimbă doar lungimea contextului.', 'Fine-tuning only changes the context length.'),
   ('Fine-tuning-ul modifică ponderile modelului, nu contextul.', 'Fine-tuning changes the weights of the model, not the context.'))]),
 dict(q=('Un context este 10, 20, 30. Cît sînt s și valorile scalate în Chronos?', 'A context is 10, 20, 30. What are s and the scaled values in Chronos?'), key=1, opts=[
  (('s = 60; valorile scalate 0,17; 0,33; 0,5.', 's = 60; scaled values 0.17, 0.33, 0.5.'),
   ('60 este suma valorilor absolute; scala este media lor.', '60 is the sum of the absolute values; the scale is their mean.')),
  (('s = 20; valorile scalate 0,5; 1; 1,5.', 's = 20; scaled values 0.5, 1, 1.5.'),
   ('s = (10 + 20 + 30)/3 = 20, iar fiecare valoare se împarte la s.', 's = (10 + 20 + 30)/3 = 20, and each value is divided by s.')),
  (('s = 30; valorile scalate 0,33; 0,67; 1.', 's = 30; scaled values 0.33, 0.67, 1.'),
   ('Împărțirea la maximum nu este scalarea prin medie a Chronos.', 'Dividing by the maximum is not the mean scaling of Chronos.'))]),
 dict(q=('q̂₀,₉ = 5 și y = 6. Cît este pierderea pinball?', 'q̂₀.₉ = 5 and y = 6. What is the pinball loss?'), key=2, opts=[
  (('0,1·(6 − 5) = 0,1.', '0.1·(6 − 5) = 0.1.'),
   ('Ponderea 1 − τ se aplică valorilor de sub cuantilă; aici y este deasupra.', 'The weight 1 − τ applies to outcomes below the quantile; here y is above.')),
  (('|6 − 5| = 1.', '|6 − 5| = 1.'),
   ('Eroarea absolută neponderată nu este pierderea pinball.', 'The unweighted absolute error is not the pinball loss.')),
  (('0,9·(6 − 5) = 0,9.', '0.9·(6 − 5) = 0.9.'),
   ('y este peste cuantilă, deci pierderea este τ(y − q̂).', 'y is above the quantile, so the loss is τ(y − q̂).'))]),
 dict(q=('Cîte patch-uri de 16 dă un context de 512 valori?', 'How many patches of 16 does a context of 512 values give?'), key=0, opts=[
  (('32 = 512/16.', '32 = 512/16.'), ('Fiecare patch grupează 16 valori consecutive.', 'Each patch groups 16 consecutive values.')),
  (('128.', '128.'), ('128 este numărul de patch-uri pentru 2048 de valori.', '128 is the number of patches for 2048 values.')),
  (('8192 = 512·16.', '8192 = 512·16.'), ('Patch-urile reduc numărul de intrări; nu îl înmulțesc.', 'Patches reduce the number of inputs; they do not multiply it.'))]),
 dict(q=('Un interval de 80% acoperă 95% din valorile observate. Este un rezultat bun?', 'An 80% interval covers 95% of the outcomes. Is this good?'), key=1, opts=[
  (('Da: o acoperire mai mare este întotdeauna mai bună.', 'Yes: a higher coverage is always better.'),
   ('Un interval de 80% trebuie să conțină circa 80% din valori; 95% arată benzi prea largi.', 'An 80% interval should contain about 80% of the outcomes; 95% shows bands that are too wide.')),
  (('Nu: intervalele sînt prea largi (lipsește finețea); CRPS penalizează acest lucru.', 'No: the intervals are too wide (not sharp); the CRPS penalises this.'),
   ('Calibrarea cere circa 80%; dintre prognozele calibrate, cea mai îngustă este mai bună.', 'Calibration requires about 80%; among calibrated forecasts, the narrower one is better.')),
  (('Da, dacă mediana este bună.', 'Yes, if the median is good.'),
   ('Acoperirea evaluează intervalul, nu mediana.', 'Coverage scores the interval, not the median.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {})) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch11-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] = TEXT_JS[lang].replace('\n};\n})();\n', '\n};\n') + ('window.CH11_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                                                                        + 'window.CH11_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n})();\n')


_predict_js()


if __name__ == '__main__':
    import interactive_common
    page_spec = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA', 'TEXT_JS', 'PREDICT', 'TICKET')})
    page_spec.widgets = lambda lang: {a: v for a, v in widgets_all(lang).items() if a in {b for b, _, _ in BLOCKS}}
    interactive_common.build(page_spec)
    for lg in ('en', 'ro'):
        _post(lg)
