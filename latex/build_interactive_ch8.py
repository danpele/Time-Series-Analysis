"""Build the interactive pages of Chapter 8 (long memory and ARFIMA): interactive/ch8.html (EN) and interactive/ch8_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its interactive version on
the right. The result tables of the lecture (GPH and local Whittle with intervals, the ARMA and ARFIMA fits, d of twelve series,
the forecast RMSE and ratios, d of realised volatility, HAR against (1 - L)^d, the ranges of d) and its worked examples (the two
decay laws, the weights of (1 - L)^d, the ACF of ARFIMA(0,d,0), the R/S statistic, the GPH regression, the local Whittle
estimator, the HAR regression) get blocks of their own (EXTRA), placed by slide; on the web lecture each of them travels inside
a chart block and ch8.js moves it to its own frame. Logic in interactive/ch8.js, inputs in interactive/ch8_data.js
(Quantlets/Ch_08/export_interactive.py), slide images in interactive/slides/.
"""
import json
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 8

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_08'
TEX = {'ro': 'RO/Cursuri/capitol8_memorie_lunga_arfima', 'en': 'EN/Courses/chapter8_long_memory_arfima'}

# (anchor, chart file, section key); order of the slides
BLOCKS = [('memory-data', 'tsa_ch8_memory_data', 'short'), ('memory-acf', 'tsa_ch8_memory_acf', 'short'), ('decay', 'tsa_ch8_decay', 'short'),
          ('weights', 'tsa_ch8_weights', 'frac'), ('ffd', 'tsa_ch8_ffd', 'frac'),
          ('arfima-paths', 'tsa_ch8_arfima_paths', 'arfima'), ('fbm', 'tsa_ch8_fbm', 'arfima'),
          ('rs-dfa', 'tsa_ch8_rs_dfa', 'est'), ('gph', 'tsa_ch8_gph', 'est'), ('bandwidth', 'tsa_ch8_bandwidth', 'est'), ('mc', 'tsa_ch8_mc', 'est'),
          ('arfima-fit', 'tsa_ch8_arfima_fit', 'est'),
          ('forecast-path', 'tsa_ch8_forecast_path', 'fc'), ('forecast', 'tsa_ch8_forecast', 'fc'),
          ('vol-acf', 'tsa_ch8_vol_acf', 'vol'), ('rv', 'tsa_ch8_rv', 'vol'), ('vol-models', 'tsa_ch8_vol_models', 'vol'),
          ('spurious', 'tsa_ch8_spurious', 'spur'), ('nile-break', 'tsa_ch8_nile_break', 'spur'), ('rolling', 'tsa_ch8_rolling', 'spur')]

# result tables and worked examples: (anchor, frame title EN (the RO frame is the one with the same position), section key, host chart block)
EXTRA = [('wx-decay', 'Interpreting the two decay laws', 'short', 'decay'),
         ('wx-weights', 'Worked example: the weights of (1-L)^0.4', 'frac', 'weights'),
         ('d-range', 'The parameter d: the meaning of each range', 'arfima', 'arfima-paths'),
         ('wx-acf', 'Worked example: ARFIMA(0, 0.3, 0)', 'arfima', 'arfima-paths'),
         ('wx-rs', 'The R/S statistic, step by step', 'est', 'rs-dfa'),
         ('wx-gph', 'The periodogram and the GPH estimator', 'est', 'gph'),
         ('gph-table', 'Interpreting the GPH regressions', 'est', 'gph'),
         ('wx-lw', 'The local Whittle estimator', 'est', 'bandwidth'),
         ('fit-table', 'Romanian inflation: ARMA and ARFIMA compared', 'est', 'arfima-fit'),
         ('memory-table', 'Long memory in twelve series', 'est', 'arfima-fit'),
         ('ratio-table', 'Interpreting the forecast comparison', 'fc', 'forecast'),
         ('rv-table', 'Interpreting realised volatility', 'vol', 'rv'),
         ('wx-har', 'FIGARCH and HAR', 'vol', 'vol-models'),
         ('har-table', 'Interpreting FIGARCH and HAR', 'vol', 'vol-models')]

PFX = {'memory-data': 'md', 'memory-acf': 'ma', 'decay': 'dc', 'weights': 'wt', 'ffd': 'ff', 'arfima-paths': 'ap', 'fbm': 'fb', 'rs-dfa': 'rd', 'gph': 'gp',
       'bandwidth': 'bw', 'mc': 'mc', 'arfima-fit': 'af', 'forecast-path': 'fp', 'forecast': 'fo', 'vol-acf': 'va', 'rv': 'rv', 'vol-models': 'vm', 'spurious': 'sp',
       'nile-break': 'nb', 'rolling': 'rl', 'wx-decay': 'wd', 'wx-weights': 'ww', 'd-range': 'dr', 'wx-acf': 'wa', 'wx-rs': 'wr', 'wx-gph': 'wg', 'gph-table': 'gt',
       'wx-lw': 'wl', 'fit-table': 'ft', 'memory-table': 'mt', 'ratio-table': 'rt', 'rv-table': 'rq', 'wx-har': 'wh', 'har-table': 'hb'}

SECTIONS = {
 'ro': dict(short='Memorie scurtă și memorie lungă', frac='Diferențierea fracționară', arfima='Modelul ARFIMA(p,d,q)', est='Estimarea lui d',
            fc='Prognoza cu ARFIMA', vol='Memoria lungă a volatilității', spur='Memoria lungă aparentă'),
 'en': dict(short='Short and long memory', frac='Fractional differencing', arfima='The ARFIMA(p,d,q) model', est='Estimating d',
            fc='Forecasting with ARFIMA', vol='Long memory in volatility', spur='Spurious long memory'),
}

L = {
 'ro': dict(file='ch8_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 8 · Memorie lungă și modele ARFIMA', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'casetele „Exemplu rezolvat” refac calculul cu cifrele capitolului.',
   src='Debitul Nilului la Assuan, 1871–1970 (statsmodels). Inflația IAPC a României, lunară și anuală (Eurostat prc_hicp_minr, ianuarie 2005 – august 2026). Inflația IPC a SUA '
       '(FRED CPIAUCSL, februarie 1947 – august 2026) și rata șomajului (FRED UNRATE, ianuarie 1948 – septembrie 2026). Cursul EUR/RON de referință (BNR) și prețurile zilnice S&P 500 și BET '
       '(data/market, pînă la 18 septembrie 2026). Seriile FRED, Eurostat și BNR sînt citite cu funcțiile capitolului (descărcate pe 4 noiembrie 2026) și tăiate la datele de sfîrșit ale '
       'slide-urilor; reproduc exact cifrele din curs. Estimatorii R/S, DFA, GPH, Whittle local, verosimilitatea exactă, regresia HAR și diferențierea fracționară se recalculează în browser; '
       'simulările folosesc extragerile seminței capitolului, iar estimările Monte Carlo, erorile prognozelor în afara eșantionului, estimările Whittle ARFIMA(1,d,0) și tabelele de modele '
       'sînt cele ale codului capitolului. Variantele care nu apar în curs (alt eșantion, altă lățime de bandă, altă origine a prognozei, alte ferestre, alt d) sînt rulări suplimentare '
       'ale codului capitolului, cu aceleași date și aceeași sămînță.'),
 'en': dict(file='ch8.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 8 · Long Memory and ARFIMA', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; '
        'the “Worked example” boxes redo the calculation with the numbers of the chapter.',
   src='Nile flow at Aswan, 1871–1970 (statsmodels). Romanian HICP inflation, monthly and 12-month (Eurostat prc_hicp_minr, January 2005 – August 2026). US CPI inflation '
       '(FRED CPIAUCSL, February 1947 – August 2026) and the unemployment rate (FRED UNRATE, January 1948 – September 2026). The BNR EUR/RON reference rate and the daily S&P 500 and BET prices '
       '(data/market, up to 18 September 2026). The FRED, Eurostat and BNR series are read with the chapter’s loaders (downloaded on 4 November 2026) and cut at the end dates of the slides; '
       'they reproduce the numbers of the lecture exactly. The R/S, DFA, GPH and local Whittle estimators, the exact likelihood, the HAR regression and fractional differencing are recomputed in '
       'the browser; the simulations use the draws of the chapter’s seed, and the Monte Carlo estimates, the out-of-sample forecast errors, the Whittle ARFIMA(1,d,0) fits and the model tables '
       'are those of the chapter code. Variants that do not appear in the lecture (another sample, another bandwidth, another forecast origin, other windows, another d) are extra runs of the '
       'chapter code with the same data and seed.'),
}

CSS = ('<style>.c8x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c8x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c8x .c8f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c8x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c8x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c8x .wctl .ctl{max-width:360px;margin:.15rem 0 .2rem}'
       '.c8t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c8t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c8t th,.c8t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c8t thead tr:last-child th{border-bottom:1px solid #1A3A6E}'
       '.c8t tbody tr:nth-child(even){background:#F5F7FB}.c8t tbody tr.hi{background:#E3F4F7}.c8t .r{text-align:right}.c8cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}</style>')


def tbl(i):
    return f'<div class="tbl c8t" id="{i}"></div>'


def wx(i, title, formula, ctl=''):
    """Worked-example box; `ctl` holds the controls that only drive the box (they stay out of the chart's controls bar)."""
    return f'<div class="c8x"><h4>{title}</h4>' + (f'<div class="wctl">{ctl}</div>' if ctl else '') + f'<div class="c8f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def box(ctl, inner):
    """A box with controls that only drive the table inside it."""
    return f'<div class="c8x"><div class="wctl">{ctl}</div>{inner}</div>'


FC_ORIGINS = ('2019-12', '2020-12', '2021-06', '2021-12', '2022-06', '2022-12', '2023-12', '2024-12', '2025-08', '2026-08')
FIT_ENDS = ('2014-12', '2016-12', '2018-12', '2020-12', '2022-12', '2024-12', '2026-08')
MON = {'ro': ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'],
       'en': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']}


def widgets_all(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    lec = s(' (curs)', ' (lecture)')
    W = s('Exemplu rezolvat', 'Worked example')
    c = (lambda x: x.replace('.', ',')) if ro else (lambda x: x)
    my = lambda iso: f'{MON[lang][int(iso[5:7]) - 1]} {iso[:4]}'   # noqa: E731
    bwa = lambda i, lo=0.4, hi=0.8, v=0.65: slider(i, s('Exponentul lățimii de bandă a, m = ⌊T<sup>a</sup>⌋', 'Bandwidth exponent a, m = ⌊T<sup>a</sup>⌋'), lo, hi, 0.05, v)   # noqa: E731
    drop = lambda i, lab: slider(i, lab, 0, 80, 10, 0)   # noqa: E731
    return {
     'memory-data': (slider('md-k', s('Media mobilă a ultimelor k observații (k = 1: seria)', 'Moving average of the last k observations (k = 1: the series)'), 1, 36, 1, 1) + play('md'),
                     CSS + charts('md-chart')),
     'memory-acf': (slider('ma-k', s('Partea eliminată de la începutul fiecărui eșantion', 'Share dropped at the start of each sample'), 0, 60, 5, 0) + play('ma'), charts('ma-chart')),
     'decay': (slider('dc-d', 'd', 0.05, 0.45, 0.05, 0.4) + play('dc'), charts('dc-chart')),
     'weights': (slider('wt-d', s('d al curbei albastre (curs: 0,4)', 'd of the blue curve (lecture: 0.4)'), 0.05, 0.95, 0.05, 0.4)
                 + slider('wt-p', s('φ al AR(1) de comparație', 'φ of the AR(1) for comparison'), 0.5, 0.99, 0.01, 0.9) + play('wt'), charts('wt-chart')),
     'ffd': (slider('ff-y', s('Eșantionul începe în', 'Sample starts in'), 2000, 2018, 1, 2000)
             + select('ff-t', s('Pragul ponderilor păstrate', 'Threshold of the weights kept'), [('0.001', '10⁻³'), ('0.0001', '10⁻⁴' + lec), ('0.00001', '10⁻⁵')], '0.0001') + play('ff'),
             charts('ff-chart')),
     'arfima-paths': (slider('ap-d', s('d (rîndul de jos: −d)', 'd (bottom row: −d)'), 0.05, 0.45, 0.05, 0.4) + play('ap'), charts('ap-chart')),
     'fbm': (slider('fb-h', s('Distanța lui H față de 1/2 (H = 0,5 ± Δ)', 'Distance of H from 1/2 (H = 0.5 ± Δ)'), 0.05, 0.45, 0.05, 0.2) + play('fb'), charts('fb-chart')),
     'rs-dfa': (slider('rd-n', s('Cel mai mic bloc n<sub>min</sub>', 'Smallest block n<sub>min</sub>'), 6, 60, 2, 10) + play('rd'), charts('rd-chart')),
     'gph': (bwa('gp-a') + play('gp'), charts('gp-chart')),
     'bandwidth': (slider('bw-e', s('Eșantioanele se termină în august', 'Samples end in August'), 2014, 2026, 1, 2026) + play('bw'), charts('bw-chart')),
     'mc': (slider('mc-r', s('Replicări folosite', 'Replications used'), 30, 300, 30, 300) + play('mc'), charts('mc-chart')),
     'arfima-fit': (slider('af-d', s('d − d̂ (ML) al curbei ARFIMA', 'd − d̂ (ML) of the ARFIMA curve'), -0.3, 0.18, 0.02, 0)
                    + slider('af-L', s('Laguri afișate', 'Lags shown'), 24, 60, 4, 36) + play('af'), charts('af-chart')),
     'forecast-path': (select('fp-o', s('Originea prognozei', 'Forecast origin'), [(o, my(o + '-01') + (lec if o == '2026-08' else '')) for o in FC_ORIGINS], '2026-08') + play('fp'),
                       charts('fp-chart')),
     'forecast': (drop('fo-k', s('Partea eliminată din primele origini', 'Share of the first origins dropped'))
                  + select('fo-b', s('Reperul (= 1)', 'Benchmark (= 1)'), [('AR', 'AR(p)' + lec), ('Mean', s('media eșantionului', 'sample mean')), ('RW', s('mersul aleator', 'random walk'))], 'AR')
                  + play('fo'), charts('fo-chart')),
     'vol-acf': (slider('va-p', s('Puterea δ în |r<sub>t</sub>|<sup>δ</sup>', 'Power δ in |r<sub>t</sub>|<sup>δ</sup>'), 0.25, 3, 0.25, 1)
                 + select('va-s', s('Ordinea zilelor', 'Order of the days'), [('0', s('cea observată', 'as observed') + lec), ('1', s('amestecată aleator', 'randomly shuffled'))], '0') + play('va'),
                 charts('va-chart')),
     'rv': (slider('rv-d', s('d − d̂ (ML), cu φ̂ fixat', 'd − d̂ (ML), with φ̂ fixed'), -0.3, 0.08, 0.02, 0) + play('rv'), charts('rv-chart')),
     'vol-models': (slider('vm-k', s('Lagul comparat k (zile)', 'Lag compared k (days)'), 0, 10, 1, 8)
                    + slider('vm-d', s('d − d̂ (Whittle local) al lui (1−L)<sup>d</sup>', 'd − d̂ (local Whittle) of (1−L)<sup>d</sup>'), -0.3, 0.3, 0.02, 0) + play('vm'),
                    charts('vm-chart')),
     'spurious': (slider('sp-r', s('Eșantioane folosite', 'Samples used'), 20, 200, 20, 200)
                  + select('sp-b', s('Banda', 'Band'), [('0', s('fără', 'none') + lec), ('1', s('5%–95% pentru Whittle local', '5%–95% of local Whittle'))], '0') + play('sp'),
                  charts('sp-chart')),
     'nile-break': (slider('nb-y', s('Ultimul an înainte de ruptură', 'Last year before the break'), 1880, 1960, 1, 1898) + play('nb'), charts('nb-chart')),
     'rolling': (slider('rl-w', s('Lungimea ferestrei, multiplu al celei din curs', 'Window length, multiple of the lecture’s'), 0.5, 1.5, 0.25, 1) + play('rl'), charts('rl-chart')),
     # ---------------------------------------------------------------- EXTRA
     'wx-decay': (slider('wd-d', 'd', 0.05, 0.45, 0.05, 0.4) + play('wd'),
                  charts('wd-chart') + wx('wd-ws', W, r'\rho(k)=\frac{\Gamma(k+d)\,\Gamma(1-d)}{\Gamma(k-d+1)\,\Gamma(d)}\approx C\,k^{2d-1},\quad C=\frac{\Gamma(1-d)}{\Gamma(d)};\qquad \rho_{AR}(k)=\phi^k,\ \ \phi=\frac{d}{1-d}',
                                         select('wd-k', s('Lagul k din calcul', 'Lag k of the calculation'), [('10', '10'), ('50', '50'), ('100', '100'), ('1000', '1000')], '10'))),
     'wx-weights': (slider('ww-d', 'd', 0.1, 1.5, 0.1, 0.4) + play('ww'),
                    charts('ww-chart') + wx('ww-ws', W, c(r'\pi_0=1,\qquad \pi_k=\pi_{k-1}\,\frac{k-1-d}{k},\qquad (1-L)^d x_t=\sum_{k\ge0}\pi_k x_{t-k},\qquad \sum_{k\ge0}\pi_k=(1-1)^d=0'))),
     'd-range': (slider('dr-d', 'd', -0.6, 1.2, 0.1, 0.4) + play('dr'), charts('dr-chart') + tbl('dr-tb')),
     'wx-acf': (slider('wa-d', 'd', 0.05, 0.45, 0.05, 0.3) + play('wa'),
                charts('wa-chart') + wx('wa-ws', W, r'\rho(1)=\frac{d}{1-d},\quad \rho(k)=\rho(k-1)\,\frac{k-1+d}{k-d},\quad \gamma(0)=\sigma^2\,\frac{\Gamma(1-2d)}{\Gamma(1-d)^2},\quad H=d+\tfrac12')),
     'wx-rs': (slider('wr-x', s('Ultima valoare x<sub>8</sub> (curs: −0,7)', 'The last value x<sub>8</sub> (lecture: −0.7)'), -1.5, 1.5, 0.1, -0.7) + play('wr'),
               charts('wr-chart') + wx('wr-ws', W, r'Y_k=\sum_{i\le k}(x_i-\bar x),\qquad R=\max_k Y_k-\min_k Y_k,\qquad S=\sqrt{\tfrac1n\sum_i(x_i-\bar x)^2},\qquad R/S')),
     'wx-gph': (select('wg-s', s('Seria', 'Series'), [('nile', s('Nilul', 'the Nile')), ('infl', s('inflația lunară din România (SA)', 'Romanian monthly inflation (SA)'))], 'nile')
                + bwa('wg-a', 0.3, 0.9) + play('wg'),
                charts('wg-chart') + wx('wg-ws', W, r'\log I(\lambda_j)=c+d\,X_j+u_j,\quad X_j=-\log\!\big(4\sin^2(\lambda_j/2)\big),\quad \hat d=\frac{\sum_j(X_j-\bar X)(Y_j-\bar Y)}{\sum_j(X_j-\bar X)^2},\quad \mathrm{SE}=\frac{\pi}{\sqrt{24m}}')),
     'gph-table': (bwa('gt-a') + play('gt'), charts('gt-chart') + tbl('gt-tb')),
     'wx-lw': (select('wl-s', s('Seria', 'Series'), [('nile', s('Nilul', 'the Nile')), ('infl', s('inflația lunară din România (SA)', 'Romanian monthly inflation (SA)')),
                                                    ('usinfl', s('inflația lunară din SUA', 'US monthly inflation')), ('unrate', s('șomajul din SUA', 'US unemployment'))], 'nile')
               + bwa('wl-a') + play('wl'),
               charts('wl-chart') + wx('wl-ws', W, r'R(d)=\log\Big(\frac1m\sum_{j=1}^m\lambda_j^{2d}I(\lambda_j)\Big)-\frac{2d}{m}\sum_{j=1}^m\log\lambda_j,\qquad \hat d=\arg\min_d R(d),\qquad \mathrm{SE}=\frac{1}{2\sqrt m}')),
     'fit-table': (select('ft-e', s('Eșantionul se termină în', 'Sample ends in'), [(e, my(e + '-01') + (lec if e == '2026-08' else '')) for e in FIT_ENDS], '2026-08') + play('ft'),
                   charts('ft-chart') + tbl('ft-tb')),
     'memory-table': (bwa('mt-a', 0.5, 0.8) + play('mt'), charts('mt-chart') + tbl('mt-tb')),
     'ratio-table': (drop('rt-k', s('Partea eliminată din primele origini', 'Share of the first origins dropped')) + play('rt'),
                     charts('rt-chart') + box(select('rt-h', s('Orizontul din tabel (luni)', 'Horizon in the table (months)'), [(str(h), str(h) + (lec if h == 12 else '')) for h in (1, 3, 6, 12, 18, 24)], '12'), tbl('rt-tb'))),
     'rv-table': (bwa('rq-a') + play('rq'), charts('rq-chart') + tbl('rq-tb')),
     'wx-har': (slider('wh-e', s('Eșantionul se termină în', 'Sample ends in'), 2006, 2026, 1, 2026) + play('wh'),
                charts('wh-chart') + wx('wh-ws', W, r'RV_{t+1}=c+\beta_d RV_t+\beta_w \overline{RV}_{t-4:t}+\beta_m \overline{RV}_{t-21:t}+u_{t+1},\quad w_1=\beta_d+\tfrac{\beta_w}{5}+\tfrac{\beta_m}{22},\ \ w_{2..5}=\tfrac{\beta_w}{5}+\tfrac{\beta_m}{22},\ \ w_{6..22}=\tfrac{\beta_m}{22}')),
     'har-table': (slider('hb-d', s('d − d̂ (Whittle local)', 'd − d̂ (local Whittle)'), -0.3, 0.3, 0.05, 0) + play('hb'), charts('hb-chart') + tbl('hb-tb')),
    }


def _frame_index(title):
    from interactive_common import ROOT, frames_of, latex_text
    frames = frames_of(open(os.path.join(ROOT, TEX['en']) + '.tex', encoding='utf-8').read())
    return [latex_text(t) for t, _ in frames].index(title)


def _extra_pages(lang):
    """Page and title of the frames of EXTRA, from the .tex and the beamer .nav file (the RO frame has the position of the EN one)."""
    from interactive_common import ROOT, frames_of, latex_text
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    assert len(frames) == len(pages)
    res = {}
    for anchor, ten, _, _ in EXTRA:
        k = _frame_index(ten)
        res[anchor] = (int(pages[k][0]), latex_text(frames[k][0]))
    return res


def _extra_live(anchor, lang, W):
    t = L[lang]
    ctl, ch = W[anchor]
    q, a = QA[lang][anchor]
    live = f'<div class="live"><div class="ctls">{ctl}</div>{ch}<ul class="out" id="{PFX[anchor]}-out"></ul></div>'
    tail = f'<div class="ask"><span class="ask-tag">{t["ask"]}</span> {q}</div>\n  <details><summary>{t["answer"]}</summary><p>{a}</p></details>'
    return live, tail


def widgets(lang):
    """Chart blocks for build_web_lecture.py: each host carries the web version of its EXTRA blocks (moved by ch8.js)."""
    W = widgets_all(lang)
    xp = _extra_pages(lang)
    res = {}
    for a, _, _ in BLOCKS:
        ctl, ch = W[a]
        held = ''
        for x, _, _, host in EXTRA:
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
    items = [(chart_pages[c][0], a, k, blocks[a]) for a, c, k in BLOCKS]
    xp = _extra_pages(lang)
    doc = pymupdf.open(os.path.join(ic.ROOT, TEX[lang] + '.pdf'))
    for anchor, _, skey, _ in EXTRA:
        pg, title = xp[anchor]
        img = f'ch8x_{anchor}_{lang}.jpg'
        doc[pg - 1].get_pixmap(dpi=130).save(os.path.join(ic.OUT, 'slides', img), jpg_quality=82)
        live, tail = _extra_live(anchor, lang, W)
        items.append((pg, anchor, skey, f'''
<section class="blk" id="{anchor}">
  <h3>{title}</h3>
  <div class="pair">
    <figure class="slide"><a href="{viewer}#page={pg}" target="_blank" rel="noopener"><img src="slides/{img}" alt="{t["slide"]} {pg - 1}: {title}" loading="lazy"></a>
      <figcaption>{t["slide"]} {pg - 1}</figcaption></figure>
    {live}
  </div>
  {tail}
</section>'''))
    items.sort(key=lambda x: x[0])
    body, nav, sec = [], [], None
    for _, anchor, skey, h in items:
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
const de = n => { const r = n % 100; return (r === 0 && n > 0) || r >= 20 ? ' de' : ''; };
const SER = { nile: 'Nilul', infl: 'inflația lunară din România', usinfl: 'inflația lunară din SUA', sp500: 'volatilitatea realizată S&P 500', unrate: 'șomajul din SUA', bet: 'BET' };
const MKT = { sp500: 'S&P 500', bet: 'BET' };
window.CH8_TEXT = {
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 sep: '; ', yrs: 'ani', dys: 'zile', mos: 'luni', usA: 'SUA', sep26: '18 septembrie 2026', yes: b => (b ? 'da' : 'nu'),
 L: { series: 'Seria', cyc: 'Cicluri mai lungi de', ci95: 'Interval 95%', has05: 'Conține 0,5 (GPH / LW)', model: 'Modelul (ML exactă)', arma: 'AR / MA', est: 'Estimatorul',
   lag: 'Lagul', ratioAR: 'Raport față de AR', drR: 'Intervalul lui d', drP: 'Proprietăți', drA: 'ACF / șocuri' },
 md1: (k, r) => (k > 1 ? `Media mobilă a ultimelor ${k} observații: ` : 'Seria însăși: ') + `cel mai lung șir de valori de aceeași parte a mediei are ${r[0]}${de(r[0])} ani pentru Nil, ${r[1]}${de(r[1])} luni pentru inflația anuală din România, ${r[2]}${de(r[2])} luni pentru șomajul din SUA și ${r[3]}${de(r[3])} luni pentru volatilitatea realizată S&P 500.`,
 md2: (nm, im, imd, il, ild, um, umd, ul, uld, rs, rsd, rb, rbd) => `Nil: media ${nm} (10<sup>8</sup> m<sup>3</sup>). Inflația anuală din România: maximum ${im}% în ${imd}, ${il}% în ${ild}. Șomajul din SUA: maximum ${um}% în ${umd}, ${ul}% în ${uld}. Volatilitatea realizată lunară: maximum ${rs}% (S&P 500, ${rsd}) și ${rb}% (BET, ${rbd}).`,
 ma1: (k, rows) => (k ? `Fără primele ${k}% din fiecare eșantion. ` : '') + rows.map(([key, n, r1, r10, a10, np, Lm]) => `${SER[key]} (T = ${n}): ρ̂(1) = ${r1}, ρ̂(10) = ${r10} față de ${a10} pentru AR(1); ${np} din ${Lm}${de(Lm)} laguri peste bandă`).join('; ') + '.',
 dc1: (d, phi, r10, a10, r50, a50, r1000) => `d = ${d}, AR(1) cu φ = ρ(1) = ${phi}: la lagul 10, ${r10} față de ${a10}; la lagul 50, ${r50} față de ${a50}; la lagul 1000, ARFIMA are încă ${r1000}.`,
 dc2: (sl, s100, s1000, asum) => `Panta log–log a ACF hiperbolice este 2d − 1 = ${sl}. Suma autocorelațiilor: ARFIMA ${s100} pînă la lagul 100 și ${s1000} pînă la 1000; AR(1): ${asum} în total.`,
 wt1: (d, p1, p2, s10, s100, ph, a10, a100) => `Curba albastră, d = ${d}: π<sub>1</sub> = ${p1}, π<sub>2</sub> = ${p2}; un șoc unitar are ponderea ${s10} după 10 perioade și ${s100} după 100. AR(1) cu φ = ${ph}: ${a10} și ${a100}.`,
 ff1: (d0, n, dm, cor, K, crit) => `De la ${d0} (${n}${de(n)} zile): statistica ADF trece de valoarea critică de 5% (${crit}) de la d = ${dm}, cu fereastra de ${K} ponderi; corelația cu nivelul este încă ${cor}.`,
 ff2: (a0, a1, c1) => `Nivelul (d = 0): ADF ${a0}. Randamentul (d = 1): ADF ${a1}, corelația cu nivelul ${c1}.`,
 ap1: (d, r1, r2, v, sv) => `d = ${d}: ρ(1) = ${r1}, ρ(2) = ${r2}; varianța teoretică ${v} (σ² = 1), varianța traiectoriei de 1000 de observații ${sv}.`,
 fb1: rows => 'Autocorelația de lag 1 a incrementelor (eșantion / teorie 2<sup>2H−1</sup> − 1): ' + rows.map(([H, a, t]) => `H = ${H}: ${a} / ${t}`).join('; ') + '.',
 fb2: rows => 'Amplitudinea traiectoriei pe 1000 de pași (max − min): ' + rows.map(([H, R]) => `H = ${H}: ${R}`).join('; ') + '.',
 rd1: (n, rows) => `Blocuri de la n = ${n}. Pantele R/S și DFA: ` + rows.map(([nm, a, b]) => `${({ 'US inflation, monthly': 'inflația lunară din SUA', 'S&P 500 returns r': 'randamentele S&P 500', 'S&P 500 |r|': '|r| S&P 500' })[nm]} ${a} și ${b}`).join('; ') + '.',
 gp1: (k, d, se, m, per, c, lw, lse, lc) => `${k === 'nile' ? 'Nilul' : 'Inflația din România'}: d̂ = ${d} (SE ${se}), m = ${m}${de(m)} frecvențe, adică cicluri mai lungi de ${per} ${k === 'nile' ? 'ani' : 'luni'}; interval 95% ${c}. Whittle local: ${lw} (SE ${lse}), ${lc}.`,
 bw1: (k, n, m0, m1, a, b, c, d) => `${k === 'infl' ? 'România' : 'Statele Unite'} (T = ${n}, m de la ${m0} la ${m1}): Whittle local între ${a} și ${b}, GPH între ${c} și ${d}.`,
 mc1: (R, w, l, g, rs) => `${R}${de(R)} replicări, d = 0,3: abaterea standard ${w} (Whittle ML), ${l} (Whittle local), ${g} (GPH). Zgomot alb: R/S dă în medie ${rs}.`,
 mc2: (w, g, l) => `AR(1) cu φ = 0,6, fără memorie lungă: Whittle ML pentru ARFIMA(0,d,0) dă în medie ${w}, GPH ${g}, Whittle local ${l}.`,
 af1: (r12, d12, a12, r24, d24, a24) => `Lagul 12: eșantion ${r12}, ARFIMA ${d12}, ARMA(1,1) ${a12}. Lagul 24: ${r24}, ${d24} și ${a24}.`,
 af2: (d, ll, l0, lr) => `Log-verosimilitatea exactă la d = ${d}: ${ll}, față de ${l0} la d̂; statistica raportului de verosimilitate 2Δℓ = ${lr} (valoarea critică de 5% este 3,84).`,
 fp1: (o, d, phi, p, mu, m12) => `Originea ${o}: d̂ = ${d}, φ̂ = ${phi} (Whittle); AR(${p}). Media de lungă durată ${mu}%, media ultimelor 12 luni ${m12}%.`,
 fp2: (a1, r1, a12, r12, a36, r36) => `Prognozele ARFIMA și AR: ${a1}% și ${r1}% luna următoare, ${a12}% și ${r12}% peste un an, ${a36}% și ${r36}% peste trei ani.`,
 fp3: (n, a, r) => `Pe cele ${n}${de(n)} valori observate după origine (asociate după luna calendaristică): RMSE ${a} pp (ARFIMA) și ${r} pp (AR).`,
 fo1: (n, o, r1, r12, r24, a, b) => `România: ${n}${de(n)} origini, de la ${o}; raportul RMSE ARFIMA/AR ${r1} la o lună, ${r12} la 12 și ${r24} la 24; cîștig între ${a}% și ${b}%.`,
 fo2: (n, o, a, b) => `Statele Unite: ${n}${de(n)} origini, de la ${o}; rapoarte între ${a} și ${b}.`,
 va1: (p, sh, rows) => `ACF a lui |r<sub>t</sub>|${p ? `<sup>${p}</sup>` : ''}${sh ? ', zile amestecate' : ''}: ` + rows.map(([k, a1, a100, a250, np]) => `${MKT[k]} ${a1} la lagul 1, ${a100} la 100, ${a250} la 250, ${np} din 250 de laguri peste bandă`).join('; ') + '.',
 va2: (p, sh, a, b) => `Whittle local pentru |r<sub>t</sub>|${p ? `<sup>${p}</sup>` : ''}${sh ? ' (zile amestecate)' : ''}: ${a} (S&P 500) și ${b} (BET).`,
 rv1: (k, d, phi, r24, m24, dll) => `${MKT[k]}: ARFIMA(1,d,0) cu d = ${d}, φ = ${phi}; ACF la lagul 24: eșantion ${r24}, model ${m24}; log-verosimilitatea se schimbă cu ${dll} față de estimarea ML.`,
 vm1: (k, ws) => `Ponderea lui r²<sub>t−${k}</sub> în σ²<sub>t</sub>: S&P 500 GARCH ${ws[0][0]}, FIGARCH ${ws[0][1]}; BET GARCH ${ws[1][0]}, FIGARCH ${ws[1][1]}.`,
 vm2: (k, h, p, d) => `Lagul ${k}: HAR ${h}, (1−L)<sup>d</sup> cu d = ${d}: ${p}.`,
 sp1: (R, a, b, g) => `${R}${de(R)} eșantioane de T = 1000, d adevărat 0. O singură schimbare de medie de o jumătate de abatere standard: Whittle local ${a} (GPH ${g}); de o abatere standard: ${b}.`,
 sp2: (a, sa, b, sb) => `Schimbare de regim: p = 0,01 (în medie ${sa} schimbări) dă ${a}; p = 0,2 (${sb} schimbări) dă ${b}.`,
 sp3: (q1, q0) => `Cu o schimbare de o abatere standard, cuantila de 5% a lui d̂ (Whittle local) este ${q1}; fără schimbare, cuantila de 95% este ${q0}.`,
 nb1: (B, a, b, dr, na, nb) => `Ruptura după ${B}: media scade de la ${a} (${na}${de(na)} ani) la ${b} (${nb}${de(nb)} ani), cu ${dr}%.`,
 nb2: (m0, l0, h0, r0, m1, l1, h1, r1) => `Seria brută: d̂ = ${m0} (ML exactă), Whittle local ${l0}, R/S H = ${h0}, ρ̂(1) = ${r0}. Fără mediile regimurilor: d̂ = ${m1}, Whittle local ${l1}, H = ${h1}, ρ̂(1) = ${r1}.`,
 rl1: (y, mx, mxd, mn, mnd, last, lo, hi, hi4, up) => `Inflația din SUA, ferestre de ${y} ani: d̂ maxim ${mx} (fereastra încheiată în ${mxd}), minim ${mn} (${mnd}); ultima fereastră ${last}, ${up ? 'peste' : 'sub'} marginea de sus a benzii fără memorie [${lo}; ${hi}] (${hi4}).`,
 rl2: (w, mn, mnd, ab, lo, hi) => `Ferestre de ${w} de zile: exponentul S&P 500 coboară pînă la ${mn} (${mnd}); BET depășește banda [${lo}; ${hi}] în ${ab}% din ferestre.`,
 wds1: (d, phi) => `ρ(1) = d/(1 − d) = ${d}/(1 − ${d}) = <b>${phi}</b>; AR(1) de comparație are φ = ${phi}.`,
 wds2: (k, d, r, a) => `La lagul k = ${k}: ρ(${k}) = Γ(${k} + ${d})Γ(1 − ${d})/(Γ(${k} − ${d} + 1)Γ(${d})) = <b>${r}</b>; AR(1): φ<sup>${k}</sup> = <b>${a}</b>.`,
 wds3: (C, k, e, ap, ra) => `C = Γ(1 − d)/Γ(d) = ${C}; C·${k}<sup>${e}</sup> = ${ap}; raportul ρ(k)/(C k<sup>2d−1</sup>) = <b>${ra}</b>.`,
 wds4: (s100, s1000, phi, asum) => `Sumele: ARFIMA ${s100} pînă la lagul 100 și ${s1000} pînă la 1000 (încă în creștere); AR(1): φ/(1 − φ) = ${phi}/(1 − ${phi}) = <b>${asum}</b>.`,
 wd1: (d, r) => `Cu d = ${d}, suma primelor 1000 de autocorelații ARFIMA este de ${r} ori suma completă a AR(1).`,
 wws1: (a, b, c, d) => `Deci (1 − L)<sup>d</sup>x<sub>t</sub> = x<sub>t</sub> − ${a}x<sub>t−1</sub> − ${b}x<sub>t−2</sub> − ${c}x<sub>t−3</sub> − ${d}x<sub>t−4</sub> − … (semnele se schimbă cînd d > 1).`,
 wws2: (a, b, c) => `Suma ponderilor 1 + π<sub>1</sub> + … + π<sub>K</sub>: ${a} pentru K = 10, ${b} pentru K = 100, ${c} pentru K = 1000; tinde la 0 încet, ca puterea K<sup>−d</sup>.`,
 ww1: (d, p1, p2, ap, p5) => `d = ${d}: π<sub>1</sub> = ${p1}, π<sub>2</sub> = ${p2}; aproximarea −d/Γ(1 − d)·k<sup>−1−d</sup> dă ${ap} la k = 5, față de valoarea exactă ${p5}.`,
 drRows: [['d ≤ −0,5', 'neinversabil (fără formă AR(∞))', '–'], ['−0,5 < d < 0', 'staționar, inversabil, antipersistent', 'negativă, cu suma −1/2'],
   ['d = 0', 'ARMA, memorie scurtă', 'descreștere exponențială'], ['0 < d < 0,5', 'staționar, memorie lungă', 'hiperbolică, nesumabilă'],
   ['0,5 ≤ d < 1', 'nestaționar, cu revenire la medie', 'șocurile se sting, încet'], ['d = 1', 'rădăcină unitară, ARIMA', 'șocurile sînt permanente']],
 drCap: d => `Rîndul evidențiat corespunde lui d = ${d}.`,
 dr1: (d, j, H, psi, r1, ds, dif) => `d = ${d}: ` + (j < 6 ? ['neinversabil', 'antipersistent', 'memorie scurtă', 'staționar, cu memorie lungă', 'nestaționar, dar cu revenire la medie', 'rădăcină unitară'][j] : 'în afara tabelului: răspunsurile ψ<sub>k</sub> cresc ca k<sup>d−1</sup>')
   + `; H = d + 1/2 = ${H}; ψ<sub>100</sub> = ${psi}` + (dif ? `; prima diferență este ARFIMA(0; ${ds}; 0)` : '') + (r1 !== null ? `, cu ρ(1) = ${r1}.` : '.'),
 was1: (d, r1, r2, r3, r5, r10) => `ρ(1) = ${d}/(1 − ${d}) = <b>${r1}</b>; ρ(2) = ${r1}·(1 + ${d})/(2 − ${d}) = <b>${r2}</b>; ρ(3) = ${r2}·(2 + ${d})/(3 − ${d}) = <b>${r3}</b>; ρ(5) = ${r5}, ρ(10) = <b>${r10}</b>.`,
 was2: (d, g) => `Aceeași valoare din formula cu funcția Gamma: ρ(10) = Γ(10 + ${d})Γ(1 − ${d})/(Γ(11 − ${d})Γ(${d})) = ${g}.`,
 was3: (a, b, v) => `Varianța cu σ² = 1: Γ(${a})/Γ(${b})² = <b>${v}</b>, mai mare decît varianța șocurilor: memoria acumulează șocurile trecute.`,
 was4: (phi, a10, r10, ra) => `AR(1) cu același ρ(1) = ${phi}: ρ(10) = ${phi}<sup>10</sup> = ${a10}, de ${ra} ori sub ${r10}.`,
 was5: H => `H = d + 0,5 = <b>${H}</b>.`,
 wa1: (d, r10, a10, v) => `d = ${d}: ρ(10) = ${r10} față de ${a10} pentru AR(1); varianța ${v}.`,
 wrs1: (s, m) => `Media blocului: x̄ = ${s}/8 = <b>${m}</b>.`,
 wrs2: Y => `Abaterile cumulate Y<sub>k</sub>: ${Y}.`,
 wrs3: (mx, mn, R) => `R = max Y<sub>k</sub> − min Y<sub>k</sub> = ${mx} − (${mn}) = <b>${R}</b>.`,
 wrs4: (S, rs) => `S = ${S} (abaterea standard a blocului), deci R/S = <b>${rs}</b>.`,
 wr1: (x8, R, S, rs) => `Cu x<sub>8</sub> = ${x8}: R = ${R}, S = ${S}, R/S = ${rs}.`,
 wgs1: (T, a, m, l1, lm, per, J) => `T = ${T}, m = ⌊${T}<sup>${a}</sup>⌋ = <b>${m}</b> din cele ${J}${de(J)} frecvențe Fourier; λ<sub>1</sub> = 2π/T = ${l1}, λ<sub>m</sub> = ${lm}: cicluri mai lungi de ${per} perioade.`,
 wgs2: (x1, y1, mx, my) => `Primul punct: X<sub>1</sub> = ${x1}, Y<sub>1</sub> = log I(λ<sub>1</sub>) = ${y1}; mediile X̄ = ${mx}, Ȳ = ${my}.`,
 wgs3: (sxy, sxx, d) => `d̂ = ${sxy}/${sxx} = <b>${d}</b>.`,
 wgs4: (m, se, c) => `SE = π/√(24·${m}) = <b>${se}</b>; interval 95% ${c}.`,
 wg1: (k, m, d, se) => `${k === 'nile' ? 'Nilul' : 'Inflația din România'}, m = ${m}: d̂ = ${d} (SE ${se}).`,
 gpName: { nile: 'Nilul', infl: 'Inflația din România' },
 gtCap: a => `m = ⌊T<sup>${a}</sup>⌋; intervalele: d̂ ± 1,96 SE, cu SE GPH = π/√(24m) și SE Whittle local = 1/(2√m).`,
 gt1: rows => rows.map(([k, g, w, pos]) => `${k === 'nile' ? 'Nilul' : 'Inflația din România'}: GPH ${g}, Whittle local ${w}` + (pos ? ' (exclude 0)' : '')).join('; ') + '.',
 wls1: (T, a, m, lm) => `T = ${T}, m = ⌊T<sup>${a}</sup>⌋ = <b>${m}</b>, λ<sub>m</sub> = ${lm}.`,
 wls2: (d, G, R) => `Minimul funcției R(d) este la d̂ = <b>${d}</b>, unde G(d̂) = ${G} și R(d̂) = ${R}.`,
 wls3: (m, se, gse, r) => `SE = 1/(2√${m}) = <b>${se}</b>, mai mică decît SE GPH π/√(24m) = ${gse} (raportul π/√6 = ${r}).`,
 wls4: (c, ok) => `Interval 95%: ${c}. ` + (ok ? 'd̂ este în (−1/2, 3/4): eroarea standard este valabilă.' : 'd̂ este peste 3/4: aproximarea normală cu această eroare standard nu mai este valabilă.'),
 wl1: (k, d, se, c, ok) => `${({ nile: 'Nilul', infl: 'Inflația din România', usinfl: 'Inflația din SUA', unrate: 'Șomajul din SUA' })[k]}: d̂ = ${d} (SE ${se}), ${c}` + (ok ? '.' : '; SE nu este valabilă (d̂ > 3/4).'),
 ftCap: (e, T) => `Inflația lunară IAPC a României, ajustată sezonier, T = ${T} luni pînă în ${e}; ℓ: log-verosimilitatea maximă; cu aldine: criteriul cel mai mic.`,
 ft1: (e, a, b, d, ll, la, phi) => `Pînă în ${e}: AIC alege ${a}, BIC alege ${b}; ARFIMA(0,d,0) are d̂ = ${d} și ℓ = ${ll}, față de ${la} pentru ARMA(1,1) cu φ̂ = ${phi}.`,
 mtName: { 'Nile': 'Nilul', 'RO inflation, monthly SA': 'Inflația RO, lunară SA', 'RO inflation, 12-month': 'Inflația RO, anuală', 'US inflation, monthly': 'Inflația SUA, lunară',
   'US inflation, 1985-2019': 'Inflația SUA, 1985–2019', 'US unemployment': 'Șomajul SUA', 'S&P 500, r_t': 'S&P 500, r<sub>t</sub>', 'S&P 500, |r_t|': 'S&P 500, |r<sub>t</sub>|',
   'BET, r_t': 'BET, r<sub>t</sub>', 'BET, |r_t|': 'BET, |r<sub>t</sub>|', 'EUR/RON, r_t': 'EUR/RON, r<sub>t</sub>', 'EUR/RON, |r_t|': 'EUR/RON, |r<sub>t</sub>|' },
 mtCap: a => `m = ⌊T<sup>${a}</sup>⌋; LW: Whittle local, SE 1/(2√m); intervalul: d̂ ± 1,96 SE.`,
 mtNote: '* d̂ ≥ 3/4: eroarea standard 1/(2√m) nu este valabilă.',
 mt1: (d, c, has1, inv) => `Șomajul din SUA: d̂ = ${d}, interval ${c}` + (has1 ? ': conține 1, deci revenirea la medie nu este demonstrată' : ': exclude 1') + (inv ? ' (iar peste d = 3/4 SE nu este valabilă).' : '.'),
 mt2: (a, b, c, d) => `Inflația anuală din România: ${a}, față de ${b} pentru rata lunară; randamentele S&P 500: ${c}, randamentele absolute: ${d}.`,
 fcName: { ARFIMA: 'ARFIMA(1,d,0)', AR: 'AR(p)', Mean: 'media eșantionului', RW: 'mersul aleator' },
 rtCap: (a, b, c, d) => `România: ${a}${de(a)} origini de la ${b} (pp pe lună); SUA: ${c}${de(c)} origini de la ${d} (pp, anualizat). Cu aldine: raport sub 1.`,
 rt1: (a, b, c, d) => `România, RMSE la 12 luni: ARFIMA ${a}, AR ${b}, media ${c}, mers aleator ${d} (pp pe lună).`,
 rqName: ['Whittle local', 'GPH', 'ML exactă, ARFIMA(1,d,0)'],
 rqBic: 'BIC: ARFIMA(1,d,0) / ARMA(1,1)',
 rqCap: T => `Log-volatilitatea realizată lunară, T = ${T} luni (2000–2026); ML exactă din codul capitolului.`,
 rq1: (a, b) => `Cele șase estimări sînt între ${a} și ${b}.`,
 whs1: (a, b, n) => `Eșantionul: ${a} – ${b}, n = ${n} zile (primele 22 de zile dau mediile).`,
 whs2: (c, bd, bw, bm, s) => `MCO: ĉ = ${c}, β̂<sub>d</sub> = <b>${bd}</b>, β̂<sub>w</sub> = <b>${bw}</b>, β̂<sub>m</sub> = <b>${bm}</b>; suma ${s}.`,
 whs3: (bd, bw, bm, w1) => `Ponderea lagului 1: ${bd} + ${bw}/5 + ${bm}/22 = <b>${w1}</b>.`,
 whs4: (bw, bm, w2, w6) => `Lagurile 2–5: ${bw}/5 + ${bm}/22 = <b>${w2}</b>; lagurile 6–22: ${bm}/22 = <b>${w6}</b>.`,
 wh1: (e, bd, bw, bm, s) => `Pînă în ${e}: β̂<sub>d</sub> = ${bd}, β̂<sub>w</sub> = ${bw}, β̂<sub>m</sub> = ${bm} (suma ${s}).`,
 hbHar: 'HAR', hbFd: d => `−π<sub>k</sub>, (1−L)<sup>d</sup>, d = ${d}`, hbCap: 'Volatilitatea zilnică pe baza amplitudinii (Parkinson), S&P 500.',
 hb1: (d, r) => `d = ${d}: la lagurile 1, 2, 6 și 22, HAR ${r.map(x => x[0]).join('; ')}; (1−L)<sup>d</sup> ${r.map(x => x[1]).join('; ')}.`,
 hb2: g => `Ponderea unui șoc de acum 100 de zile: S&P 500 GARCH ${g[0][0]}, FIGARCH ${g[0][1]}; BET ${g[1][0]} și ${g[1][1]}. BIC scade cu ${g[0][2]} și ${g[1][2]} față de GARCH (α + β = ${g[0][3]} și ${g[1][3]}).`,
};
})();
""",
 'en': r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const SER = { nile: 'Nile', infl: 'Romanian monthly inflation', usinfl: 'US monthly inflation', sp500: 'S&P 500 realised volatility', unrate: 'US unemployment', bet: 'BET' };
const MKT = { sp500: 'S&P 500', bet: 'BET' };
window.CH8_TEXT = {
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 sep: ', ', yrs: 'years', dys: 'days', mos: 'months', usA: 'US', sep26: '18 September 2026', yes: b => (b ? 'yes' : 'no'),
 L: { series: 'Series', cyc: 'Cycles longer than', ci95: '95% interval', has05: 'Contains 0.5 (GPH / LW)', model: 'Model (exact ML)', arma: 'AR / MA', est: 'Estimator',
   lag: 'Lag', ratioAR: 'Ratio to AR', drR: 'Range of d', drP: 'Properties', drA: 'ACF / shocks' },
 md1: (k, r) => (k > 1 ? `Moving average of the last ${k} observations: ` : 'The series itself: ') + `the longest run on one side of the mean lasts ${r[0]} years for the Nile, ${r[1]} months for Romanian 12-month inflation, ${r[2]} months for US unemployment and ${r[3]} months for S&P 500 realised volatility.`,
 md2: (nm, im, imd, il, ild, um, umd, ul, uld, rs, rsd, rb, rbd) => `Nile: mean ${nm} (10<sup>8</sup> m<sup>3</sup>). Romanian 12-month inflation: peak ${im}% in ${imd}, ${il}% in ${ild}. US unemployment: peak ${um}% in ${umd}, ${ul}% in ${uld}. Monthly realised volatility: maxima ${rs}% (S&P 500, ${rsd}) and ${rb}% (BET, ${rbd}).`,
 ma1: (k, rows) => (k ? `Without the first ${k}% of each sample. ` : '') + rows.map(([key, n, r1, r10, a10, np, Lm]) => `${SER[key]} (T = ${n}): ρ̂(1) = ${r1}, ρ̂(10) = ${r10} against ${a10} for the AR(1); ${np} of ${Lm} lags above the band`).join('; ') + '.',
 dc1: (d, phi, r10, a10, r50, a50, r1000) => `d = ${d}, AR(1) with φ = ρ(1) = ${phi}: at lag 10, ${r10} against ${a10}; at lag 50, ${r50} against ${a50}; at lag 1000 the ARFIMA still has ${r1000}.`,
 dc2: (sl, s100, s1000, asum) => `The log–log slope of the hyperbolic ACF is 2d − 1 = ${sl}. Sums of autocorrelations: ARFIMA ${s100} up to lag 100 and ${s1000} up to 1000; AR(1): ${asum} in total.`,
 wt1: (d, p1, p2, s10, s100, ph, a10, a100) => `Blue curve, d = ${d}: π<sub>1</sub> = ${p1}, π<sub>2</sub> = ${p2}; a unit shock has weight ${s10} after 10 periods and ${s100} after 100. AR(1) with φ = ${ph}: ${a10} and ${a100}.`,
 ff1: (d0, n, dm, cor, K, crit) => `From ${d0} (${n} days): the ADF statistic passes the 5% critical value (${crit}) from d = ${dm}, with a window of ${K} weights; the correlation with the level is still ${cor}.`,
 ff2: (a0, a1, c1) => `The level (d = 0): ADF ${a0}. The return (d = 1): ADF ${a1}, correlation with the level ${c1}.`,
 ap1: (d, r1, r2, v, sv) => `d = ${d}: ρ(1) = ${r1}, ρ(2) = ${r2}; theoretical variance ${v} (σ² = 1), variance of the 1000-observation path ${sv}.`,
 fb1: rows => 'Lag-1 autocorrelation of the increments (sample / theory 2<sup>2H−1</sup> − 1): ' + rows.map(([H, a, t]) => `H = ${H}: ${a} / ${t}`).join('; ') + '.',
 fb2: rows => 'Range of the path over 1000 steps (max − min): ' + rows.map(([H, R]) => `H = ${H}: ${R}`).join('; ') + '.',
 rd1: (n, rows) => `Blocks from n = ${n}. R/S and DFA slopes: ` + rows.map(([nm, a, b]) => `${({ 'US inflation, monthly': 'US monthly inflation', 'S&P 500 returns r': 'S&P 500 returns', 'S&P 500 |r|': 'S&P 500 |r|' })[nm]} ${a} and ${b}`).join('; ') + '.',
 gp1: (k, d, se, m, per, c, lw, lse, lc) => `${k === 'nile' ? 'Nile' : 'Romanian inflation'}: d̂ = ${d} (SE ${se}), m = ${m} frequencies, i.e. cycles longer than ${per} ${k === 'nile' ? 'years' : 'months'}; 95% interval ${c}. Local Whittle: ${lw} (SE ${lse}), ${lc}.`,
 bw1: (k, n, m0, m1, a, b, c, d) => `${k === 'infl' ? 'Romania' : 'United States'} (T = ${n}, m from ${m0} to ${m1}): local Whittle between ${a} and ${b}, GPH between ${c} and ${d}.`,
 mc1: (R, w, l, g, rs) => `${R} replications, d = 0.3: standard deviation ${w} (Whittle ML), ${l} (local Whittle), ${g} (GPH). White noise: R/S gives ${rs} on average.`,
 mc2: (w, g, l) => `AR(1) with φ = 0.6 and no long memory: Whittle ML of ARFIMA(0,d,0) gives ${w} on average, GPH ${g}, local Whittle ${l}.`,
 af1: (r12, d12, a12, r24, d24, a24) => `Lag 12: sample ${r12}, ARFIMA ${d12}, ARMA(1,1) ${a12}. Lag 24: ${r24}, ${d24} and ${a24}.`,
 af2: (d, ll, l0, lr) => `Exact log-likelihood at d = ${d}: ${ll}, against ${l0} at d̂; likelihood-ratio statistic 2Δℓ = ${lr} (the 5% critical value is 3.84).`,
 fp1: (o, d, phi, p, mu, m12) => `Origin ${o}: d̂ = ${d}, φ̂ = ${phi} (Whittle); AR(${p}). Long-run mean ${mu}%, mean of the last 12 months ${m12}%.`,
 fp2: (a1, r1, a12, r12, a36, r36) => `ARFIMA and AR forecasts: ${a1}% and ${r1}% next month, ${a12}% and ${r12}% in a year, ${a36}% and ${r36}% in three years.`,
 fp3: (n, a, r) => `Over the ${n} values observed after the origin (matched by calendar month): RMSE ${a} pp (ARFIMA) and ${r} pp (AR).`,
 fo1: (n, o, r1, r12, r24, a, b) => `Romania: ${n} origins from ${o}; RMSE ratio ARFIMA/AR ${r1} at 1 month, ${r12} at 12 and ${r24} at 24; gain between ${a}% and ${b}%.`,
 fo2: (n, o, a, b) => `United States: ${n} origins from ${o}; ratios between ${a} and ${b}.`,
 va1: (p, sh, rows) => `ACF of |r<sub>t</sub>|${p ? `<sup>${p}</sup>` : ''}${sh ? ', days shuffled' : ''}: ` + rows.map(([k, a1, a100, a250, np]) => `${MKT[k]} ${a1} at lag 1, ${a100} at 100, ${a250} at 250, ${np} of 250 lags above the band`).join('; ') + '.',
 va2: (p, sh, a, b) => `Local Whittle for |r<sub>t</sub>|${p ? `<sup>${p}</sup>` : ''}${sh ? ' (days shuffled)' : ''}: ${a} (S&P 500) and ${b} (BET).`,
 rv1: (k, d, phi, r24, m24, dll) => `${MKT[k]}: ARFIMA(1,d,0) with d = ${d}, φ = ${phi}; ACF at lag 24: sample ${r24}, model ${m24}; the log-likelihood changes by ${dll} from the ML estimate.`,
 vm1: (k, ws) => `Weight of r²<sub>t−${k}</sub> in σ²<sub>t</sub>: S&P 500 GARCH ${ws[0][0]}, FIGARCH ${ws[0][1]}; BET GARCH ${ws[1][0]}, FIGARCH ${ws[1][1]}.`,
 vm2: (k, h, p, d) => `Lag ${k}: HAR ${h}, (1−L)<sup>d</sup> with d = ${d}: ${p}.`,
 sp1: (R, a, b, g) => `${R} samples of T = 1000, true d = 0. One mean shift of half a standard deviation: local Whittle ${a} (GPH ${g}); of one standard deviation: ${b}.`,
 sp2: (a, sa, b, sb) => `Regime switching: p = 0.01 (${sa} switches on average) gives ${a}; p = 0.2 (${sb} switches) gives ${b}.`,
 sp3: (q1, q0) => `With a shift of one standard deviation the 5% quantile of d̂ (local Whittle) is ${q1}; without a shift the 95% quantile is ${q0}.`,
 nb1: (B, a, b, dr, na, nb) => `Break after ${B}: the mean falls from ${a} (${na} years) to ${b} (${nb} years), by ${dr}%.`,
 nb2: (m0, l0, h0, r0, m1, l1, h1, r1) => `Raw series: d̂ = ${m0} (exact ML), local Whittle ${l0}, R/S H = ${h0}, ρ̂(1) = ${r0}. Regime means removed: d̂ = ${m1}, local Whittle ${l1}, H = ${h1}, ρ̂(1) = ${r1}.`,
 rl1: (y, mx, mxd, mn, mnd, last, lo, hi, hi4, up) => `US inflation, ${y}-year windows: highest d̂ ${mx} (window ending ${mxd}), lowest ${mn} (${mnd}); last window ${last}, ${up ? 'above' : 'below'} the upper edge of the band of no memory [${lo}, ${hi}] (${hi4}).`,
 rl2: (w, mn, mnd, ab, lo, hi) => `${w}-day windows: the S&P 500 exponent falls to ${mn} (${mnd}); the BET exceeds the band [${lo}, ${hi}] in ${ab}% of the windows.`,
 wds1: (d, phi) => `ρ(1) = d/(1 − d) = ${d}/(1 − ${d}) = <b>${phi}</b>; the AR(1) for comparison has φ = ${phi}.`,
 wds2: (k, d, r, a) => `At lag k = ${k}: ρ(${k}) = Γ(${k} + ${d})Γ(1 − ${d})/(Γ(${k} − ${d} + 1)Γ(${d})) = <b>${r}</b>; AR(1): φ<sup>${k}</sup> = <b>${a}</b>.`,
 wds3: (C, k, e, ap, ra) => `C = Γ(1 − d)/Γ(d) = ${C}; C·${k}<sup>${e}</sup> = ${ap}; the ratio ρ(k)/(C k<sup>2d−1</sup>) = <b>${ra}</b>.`,
 wds4: (s100, s1000, phi, asum) => `Sums: ARFIMA ${s100} up to lag 100 and ${s1000} up to 1000 (still growing); AR(1): φ/(1 − φ) = ${phi}/(1 − ${phi}) = <b>${asum}</b>.`,
 wd1: (d, r) => `With d = ${d}, the sum of the first 1000 ARFIMA autocorrelations is ${r} times the full sum of the AR(1).`,
 wws1: (a, b, c, d) => `So (1 − L)<sup>d</sup>x<sub>t</sub> = x<sub>t</sub> − ${a}x<sub>t−1</sub> − ${b}x<sub>t−2</sub> − ${c}x<sub>t−3</sub> − ${d}x<sub>t−4</sub> − … (the signs change when d > 1).`,
 wws2: (a, b, c) => `Sum of the weights 1 + π<sub>1</sub> + … + π<sub>K</sub>: ${a} for K = 10, ${b} for K = 100, ${c} for K = 1000; it tends to 0 slowly, like the power K<sup>−d</sup>.`,
 ww1: (d, p1, p2, ap, p5) => `d = ${d}: π<sub>1</sub> = ${p1}, π<sub>2</sub> = ${p2}; the approximation −d/Γ(1 − d)·k<sup>−1−d</sup> gives ${ap} at k = 5, against the exact ${p5}.`,
 drRows: [['d ≤ −0.5', 'not invertible (no AR(∞) form)', '–'], ['−0.5 < d < 0', 'stationary, invertible, anti-persistent', 'negative, summable to −1/2'],
   ['d = 0', 'ARMA, short memory', 'exponential decay'], ['0 < d < 0.5', 'stationary, long memory', 'hyperbolic, not summable'],
   ['0.5 ≤ d < 1', 'non-stationary, mean-reverting', 'shocks fade, slowly'], ['d = 1', 'unit root, ARIMA', 'shocks are permanent']],
 drCap: d => `The highlighted row is the one of d = ${d}.`,
 dr1: (d, j, H, psi, r1, ds, dif) => `d = ${d}: ` + (j < 6 ? ['not invertible', 'anti-persistent', 'short memory', 'stationary with long memory', 'non-stationary but mean-reverting', 'unit root'][j] : 'outside the table: the responses ψ<sub>k</sub> grow like k<sup>d−1</sup>')
   + `; H = d + 1/2 = ${H}; ψ<sub>100</sub> = ${psi}` + (dif ? `; the first difference is ARFIMA(0, ${ds}, 0)` : '') + (r1 !== null ? `, with ρ(1) = ${r1}.` : '.'),
 was1: (d, r1, r2, r3, r5, r10) => `ρ(1) = ${d}/(1 − ${d}) = <b>${r1}</b>; ρ(2) = ${r1}·(1 + ${d})/(2 − ${d}) = <b>${r2}</b>; ρ(3) = ${r2}·(2 + ${d})/(3 − ${d}) = <b>${r3}</b>; ρ(5) = ${r5}, ρ(10) = <b>${r10}</b>.`,
 was2: (d, g) => `The same value from the Gamma-function formula: ρ(10) = Γ(10 + ${d})Γ(1 − ${d})/(Γ(11 − ${d})Γ(${d})) = ${g}.`,
 was3: (a, b, v) => `Variance with σ² = 1: Γ(${a})/Γ(${b})² = <b>${v}</b>, larger than the variance of the shocks: the memory accumulates past shocks.`,
 was4: (phi, a10, r10, ra) => `An AR(1) with the same ρ(1) = ${phi}: ρ(10) = ${phi}<sup>10</sup> = ${a10}, ${ra} times below ${r10}.`,
 was5: H => `H = d + 0.5 = <b>${H}</b>.`,
 wa1: (d, r10, a10, v) => `d = ${d}: ρ(10) = ${r10} against ${a10} for the AR(1); variance ${v}.`,
 wrs1: (s, m) => `Mean of the block: x̄ = ${s}/8 = <b>${m}</b>.`,
 wrs2: Y => `Cumulative deviations Y<sub>k</sub>: ${Y}.`,
 wrs3: (mx, mn, R) => `R = max Y<sub>k</sub> − min Y<sub>k</sub> = ${mx} − (${mn}) = <b>${R}</b>.`,
 wrs4: (S, rs) => `S = ${S} (the standard deviation of the block), so R/S = <b>${rs}</b>.`,
 wr1: (x8, R, S, rs) => `With x<sub>8</sub> = ${x8}: R = ${R}, S = ${S}, R/S = ${rs}.`,
 wgs1: (T, a, m, l1, lm, per, J) => `T = ${T}, m = ⌊${T}<sup>${a}</sup>⌋ = <b>${m}</b> of the ${J} Fourier frequencies; λ<sub>1</sub> = 2π/T = ${l1}, λ<sub>m</sub> = ${lm}: cycles longer than ${per} periods.`,
 wgs2: (x1, y1, mx, my) => `First point: X<sub>1</sub> = ${x1}, Y<sub>1</sub> = log I(λ<sub>1</sub>) = ${y1}; means X̄ = ${mx}, Ȳ = ${my}.`,
 wgs3: (sxy, sxx, d) => `d̂ = ${sxy}/${sxx} = <b>${d}</b>.`,
 wgs4: (m, se, c) => `SE = π/√(24·${m}) = <b>${se}</b>; 95% interval ${c}.`,
 wg1: (k, m, d, se) => `${k === 'nile' ? 'Nile' : 'Romanian inflation'}, m = ${m}: d̂ = ${d} (SE ${se}).`,
 gpName: { nile: 'Nile', infl: 'Romanian inflation' },
 gtCap: a => `m = ⌊T<sup>${a}</sup>⌋; intervals: d̂ ± 1.96 SE, with GPH SE = π/√(24m) and local Whittle SE = 1/(2√m).`,
 gt1: rows => rows.map(([k, g, w, pos]) => `${k === 'nile' ? 'Nile' : 'Romanian inflation'}: GPH ${g}, local Whittle ${w}` + (pos ? ' (excludes 0)' : '')).join('; ') + '.',
 wls1: (T, a, m, lm) => `T = ${T}, m = ⌊T<sup>${a}</sup>⌋ = <b>${m}</b>, λ<sub>m</sub> = ${lm}.`,
 wls2: (d, G, R) => `The minimum of R(d) is at d̂ = <b>${d}</b>, where G(d̂) = ${G} and R(d̂) = ${R}.`,
 wls3: (m, se, gse, r) => `SE = 1/(2√${m}) = <b>${se}</b>, smaller than the GPH SE π/√(24m) = ${gse} (ratio π/√6 = ${r}).`,
 wls4: (c, ok) => `95% interval: ${c}. ` + (ok ? 'd̂ lies in (−1/2, 3/4): the standard error is valid.' : 'd̂ is above 3/4: the normal approximation with this standard error is no longer valid.'),
 wl1: (k, d, se, c, ok) => `${({ nile: 'Nile', infl: 'Romanian inflation', usinfl: 'US inflation', unrate: 'US unemployment' })[k]}: d̂ = ${d} (SE ${se}), ${c}` + (ok ? '.' : '; the SE is not valid (d̂ > 3/4).'),
 ftCap: (e, T) => `Romanian monthly HICP inflation, seasonally adjusted, T = ${T} months up to ${e}; ℓ: maximised log-likelihood; bold: the smallest criterion.`,
 ft1: (e, a, b, d, ll, la, phi) => `Up to ${e}: AIC chooses ${a}, BIC chooses ${b}; ARFIMA(0,d,0) has d̂ = ${d} and ℓ = ${ll}, against ${la} for ARMA(1,1) with φ̂ = ${phi}.`,
 mtName: { 'Nile': 'Nile', 'RO inflation, monthly SA': 'RO inflation, monthly SA', 'RO inflation, 12-month': 'RO inflation, 12-month', 'US inflation, monthly': 'US inflation, monthly',
   'US inflation, 1985-2019': 'US inflation, 1985–2019', 'US unemployment': 'US unemployment', 'S&P 500, r_t': 'S&P 500, r<sub>t</sub>', 'S&P 500, |r_t|': 'S&P 500, |r<sub>t</sub>|',
   'BET, r_t': 'BET, r<sub>t</sub>', 'BET, |r_t|': 'BET, |r<sub>t</sub>|', 'EUR/RON, r_t': 'EUR/RON, r<sub>t</sub>', 'EUR/RON, |r_t|': 'EUR/RON, |r<sub>t</sub>|' },
 mtCap: a => `m = ⌊T<sup>${a}</sup>⌋; LW: local Whittle, SE 1/(2√m); interval: d̂ ± 1.96 SE.`,
 mtNote: '* d̂ ≥ 3/4: the standard error 1/(2√m) is not valid.',
 mt1: (d, c, has1, inv) => `US unemployment: d̂ = ${d}, interval ${c}` + (has1 ? ': it contains 1, so mean reversion is not established' : ': it excludes 1') + (inv ? ' (and above d = 3/4 the SE is not valid).' : '.'),
 mt2: (a, b, c, d) => `Romanian 12-month inflation: ${a}, against ${b} for the monthly rate; S&P 500 returns: ${c}, absolute returns: ${d}.`,
 fcName: { ARFIMA: 'ARFIMA(1,d,0)', AR: 'AR(p)', Mean: 'sample mean', RW: 'random walk' },
 rtCap: (a, b, c, d) => `Romania: ${a} origins from ${b} (pp per month); US: ${c} origins from ${d} (pp, annualised). Bold: ratio below 1.`,
 rt1: (a, b, c, d) => `Romania, RMSE at 12 months: ARFIMA ${a}, AR ${b}, mean ${c}, random walk ${d} (pp per month).`,
 rqName: ['Local Whittle', 'GPH', 'Exact ML, ARFIMA(1,d,0)'],
 rqBic: 'BIC: ARFIMA(1,d,0) / ARMA(1,1)',
 rqCap: T => `Log monthly realised volatility, T = ${T} months (2000–2026); exact ML from the chapter code.`,
 rq1: (a, b) => `The six estimates lie between ${a} and ${b}.`,
 whs1: (a, b, n) => `Sample: ${a} – ${b}, n = ${n} days (the first 22 days give the averages).`,
 whs2: (c, bd, bw, bm, s) => `OLS: ĉ = ${c}, β̂<sub>d</sub> = <b>${bd}</b>, β̂<sub>w</sub> = <b>${bw}</b>, β̂<sub>m</sub> = <b>${bm}</b>; sum ${s}.`,
 whs3: (bd, bw, bm, w1) => `Weight of lag 1: ${bd} + ${bw}/5 + ${bm}/22 = <b>${w1}</b>.`,
 whs4: (bw, bm, w2, w6) => `Lags 2–5: ${bw}/5 + ${bm}/22 = <b>${w2}</b>; lags 6–22: ${bm}/22 = <b>${w6}</b>.`,
 wh1: (e, bd, bw, bm, s) => `Up to ${e}: β̂<sub>d</sub> = ${bd}, β̂<sub>w</sub> = ${bw}, β̂<sub>m</sub> = ${bm} (sum ${s}).`,
 hbHar: 'HAR', hbFd: d => `−π<sub>k</sub>, (1−L)<sup>d</sup>, d = ${d}`, hbCap: 'Daily range-based (Parkinson) volatility of the S&P 500.',
 hb1: (d, r) => `d = ${d}: at lags 1, 2, 6 and 22, HAR ${r.map(x => x[0]).join(', ')}; (1−L)<sup>d</sup> ${r.map(x => x[1]).join(', ')}.`,
 hb2: g => `Weight of a shock 100 days ago: S&P 500 GARCH ${g[0][0]}, FIGARCH ${g[0][1]}; BET ${g[1][0]} and ${g[1][1]}. BIC falls by ${g[0][2]} and ${g[1][2]} against GARCH (α + β = ${g[0][3]} and ${g[1][3]}).`,
};
})();
""",
}

QA = {
 'ro': {
  'memory-data': ('Pe cîte luni consecutive, cel mult, rămîne șomajul din SUA de aceeași parte a mediei sale din 1948–2026?',
                  '106 luni, aproape nouă ani; Nilul rămîne 11 ani de aceeași parte a mediei, inflația anuală din România 103 luni. Pentru o serie fără memorie, șirurile lungi de acest fel ar fi foarte rare.'),
  'memory-acf': ('Eliminați primele 50% din fiecare eșantion (inflația din SUA începe atunci în noiembrie 1986). Cîte dintre cele 48 de autocorelații ale inflației lunare din SUA rămîn peste bandă?',
                 '11 din 48, față de 48 din 48 pe 1947–2026; ρ̂(10) scade de la 0,38 la 0,16. O mare parte din memoria seriei complete vine din anii 1970 și de la începutul anilor 1980.'),
  'decay': ('Fixați d = 0,2. Cît este ρ(10) pentru ARFIMA(0; 0,2; 0) și pentru AR(1) cu același ρ(1) = 0,25?',
            '0,064 față de 9,5 · 10⁻⁷. Cu d mic, coada hiperbolică este joasă, dar tot cu cinci ordine de mărime peste cea exponențială.'),
  'weights': ('Duceți curba albastră la d = 0,9. Ce parte dintr-un șoc unitar mai rămîne după 100 de perioade?',
              'ψ₁₀₀ = 0,59 (0,74 după 10 perioade), față de 0,9¹⁰⁰ = 2,7 · 10⁻⁵ pentru AR(1) cu φ = 0,9. Seria revine la medie, dar foarte încet.'),
  'ffd': ('Alegeți pragul 10⁻³ (fereastră mai scurtă). De la ce d trece statistica ADF de valoarea critică de 5% și cu cîte ponderi?',
          'De la d = 0,35, cu 61 de ponderi, față de 0,25 cu 445. Ponderile trunchiate nu mai au suma 0, deci filtrul păstrează o parte din nivel și trebuie un d mai mare pentru staționaritate.'),
  'arfima-paths': ('Fixați d = 0,45. Cît este varianța teoretică a seriei (σ² = 1)?',
                   'Γ(0,1)/Γ(0,55)² = 3,64. Varianța traiectoriei de 1000 de observații este doar 2,31: media eșantionului absoarbe o parte din componenta lentă.'),
  'fbm': ('Duceți H la 0,05 și 0,95 (Δ = 0,45). Cît este autocorelația de lag 1 a incrementelor pentru H = 0,95, în teorie și în eșantion?',
          'În teorie 2<sup>0,9</sup> − 1 = 0,87, în eșantion 0,72. Cu o persistență atît de puternică, media celor 1000 de incremente absoarbe o parte din dependență, așa că ACF de selecție este deplasată în jos.'),
  'rs-dfa': ('Ridicați cel mai mic bloc la n = 60. Ce pantă R/S obțin randamentele absolute S&P 500?',
             '0,92, față de 0,86 cu n de la 10; randamentele rămîn la 0,53. La blocuri mici domină zgomotul de la o zi la alta al lui |r<sub>t</sub>|; partea persistentă se vede la blocuri mari.'),
  'gph': ('Fixați a = 0,8. Ce dau GPH și Whittle local pentru inflația lunară din România (m = 85)?',
          'GPH 0,62 (SE 0,07), Whittle local 0,29 (SE 0,05). GPH lucrează cu log I, iar valorile aproape nule de la frecvențele sezoniere eliminate prin ajustare îi trag panta în sus; Whittle local folosește I însăși și nu este afectat.'),
  'bandwidth': ('Încheiați ambele eșantioane în august 2019. Între ce valori se mișcă Whittle local pentru SUA și pentru România (T = 176)?',
                'SUA: 0,35–0,54; România: 0,24–0,59. Concluzia memoriei lungi rămîne, iar incertitudinea de circa ±0,15 se vede și fără anii 2020–2026.'),
  'mc': ('Folosiți doar primele 30 de replicări. Care estimator este cel mai precis la d = 0,3?',
         'Whittle ML: abaterea standard 0,033, față de 0,076 (Whittle local) și 0,086 (GPH), aceeași ordine ca la 300 de replicări (0,04; 0,08; 0,10).'),
  'arfima-fit': ('Ridicați d cu 0,14, la 0,45. Cu cît scade log-verosimilitatea și trece 2Δℓ de 3,84?',
                 'ℓ scade de la −135,9 la −139,8; 2Δℓ = 7,8 > 3,84, deci d = 0,45 este respins la 5%. Curba ARFIMA devine prea înaltă la lagurile 12 și 24 (0,65 și 0,60).'),
  'forecast-path': ('Mutați originea în iunie 2022. Pe cele 36 de luni observate după ea, care prognoză are RMSE mai mic?',
                    'ARFIMA: 3,37 pp, față de 4,02 pp pentru AR(12). Ambele porneau de sus, dar AR(12) a prelungit vîrful din 2022 (6,2% peste un an, față de 5,4%), iar inflația a scăzut mai repede.'),
  'forecast': ('Eliminați primele 50% din origini (România de la mai 2019). Mai este ARFIMA mai precis decît AR la toate orizonturile în România?',
               'Nu: raportul este 1,038 la o lună (AR mai precis), 0,999 la 12 și 0,976 la 24. Cîștigul rămîne doar la orizonturile lungi.'),
  'vol-acf': ('Alegeți zilele amestecate. Cîte dintre cele 250 de autocorelații ale lui |r<sub>t</sub>| S&P 500 rămîn peste bandă și cît dă Whittle local?',
              '6 din 250 (față de 227), iar Whittle local 0,00. Amestecarea păstrează distribuția și distruge ordinea: memoria stă în succesiunea zilelor calme și agitate.'),
  'rv': ('Scădeți d cu 0,30 (cu φ fixat). Cu cît scade log-verosimilitatea pentru BET și pentru S&P 500?',
         'Cu 43,8 (BET) și 33,4 (S&P 500): ambele valori sînt respinse clar. La BET, ACF a modelului la lagul 24 ar fi 0,004, față de 0,28 în eșantion.'),
  'vol-models': ('Duceți k la 500 de zile. Cît sînt ponderile FIGARCH și GARCH pentru S&P 500?',
                 'FIGARCH 2,9 · 10⁻⁵, GARCH 2,8 · 10⁻³¹. Chiar cu α + β = 0,995, ponderile GARCH scad exponențial; cele FIGARCH scad ca o putere.'),
  'spurious': ('Afișați banda 5%–95%. Cu o schimbare de o abatere standard, cît este cuantila de 5% a lui d̂ (Whittle local)?',
               '0,31, peste cuantila de 95% fără schimbare (0,07): distribuțiile nu se suprapun, deci fiecare dintre cele 200 de eșantioane cu ruptură arată „memorie lungă”.'),
  'nile-break': ('Mutați ruptura în 1930. Ce d (ML exactă) rămîne după eliminarea celor două medii?',
                 '0,35, aproape cît în seria brută (0,36). O ruptură pusă în alt an nu explică memoria; doar ruptura din 1898 o reduce la 0,05.'),
  'rolling': ('Înjumătățiți ferestrele (10 ani, 500 de zile). Cînd și cît de sus ajunge d pentru inflația din SUA?',
              '0,64 în fereastra încheiată în iulie 1974, față de 0,71 în iulie 1980 cu ferestre de 20 de ani; banda fără memorie se lărgește la [−0,33; 0,25].'),
  'wx-decay': ('Fixați d = 0,1. De cîte ori este suma primelor 1000 de autocorelații ARFIMA mai mare decît suma completă a AR(1)?',
               'De 14 ori: 1,74 față de 0,125. Chiar o memorie lungă slabă acumulează mult peste o memorie scurtă cu același ρ(1).'),
  'wx-weights': ('Fixați d = 1,5. Cît sînt π₁ și π₂ și ce semn are π₂?',
                 'π₁ = −1,5 și π₂ = −1,5 · (1 − 1,5)/2 = +0,375. Pentru d > 1 factorul (k − 1 − d) este negativ la k = 2, deci ponderile își schimbă semnul după primul lag.'),
  'd-range': ('Fixați d = 0,6. Este seria staționară și ce este prima ei diferență?',
              'Nu este staționară (d ≥ 0,5), dar revine la medie: ψ₁₀₀ = 0,11. Prima diferență este ARFIMA(0; −0,4; 0), antipersistentă, cu ρ(1) = −0,29.'),
  'wx-acf': ('Fixați d = 0,45. Cît sînt ρ(1) și ρ(10)?',
             'ρ(1) = 0,818 și ρ(10) = 0,652; AR(1) cu același ρ(1) are ρ(10) = 0,134. Varianța crește la 3,64.'),
  'wx-rs': ('Fixați x₈ = 0,5. Cît devine R/S?',
            'Media devine 0,275, R = 0,575, S = 0,386, deci R/S = 1,49 (față de 1,79). O singură valoare schimbă media, deci toate abaterile cumulate.'),
  'wx-gph': ('Inflația din România cu a = 0,5 (m = 16): cît sînt d̂ și SE?',
             'd̂ = 0,56, SE 0,16. Cu mai puține frecvențe estimarea urcă spre 0,5, iar eroarea standard crește ca 1/√m.'),
  'gph-table': ('Fixați a = 0,5. Mai exclude 0 intervalul GPH pentru Nil?',
                'Nu: d̂ = 0,39, SE 0,20, intervalul [−0,01; 0,79]. Cu m = 10 GPH nu mai separă d de 0; Whittle local [0,15; 0,77] încă îl separă.'),
  'wx-lw': ('Alegeți șomajul din SUA (a = 0,65). Este valabilă eroarea standard 1/(2√m)?',
            'Nu: d̂ = 0,91 > 3/4, în afara intervalului aproximării normale. Intervalul [0,80; 1,01] conține oricum 1, deci revenirea la medie nu este demonstrată.'),
  'fit-table': ('Încheiați eșantionul în decembrie 2018 (T = 168). Care model are BIC cel mai mic și cît este d̂?',
                'ARFIMA(0,d,0), cu d̂ = 0,24: BIC 191,2 față de 195,5 pentru ARMA(1,1). Concluzia din curs se păstrează și pe eșantionul mai scurt.'),
  'memory-table': ('Fixați a = 0,5 (m = 30 pentru șomajul din SUA). Mai conține 1 intervalul Whittle local?',
                   'Nu: d̂ = 0,64, intervalul [0,46; 0,82], iar SE este valabilă (d̂ < 3/4). Cu a = 0,65 (m = 85): 0,91, [0,80; 1,01]. Verdictul depinde de lățimea de bandă.'),
  'ratio-table': ('Eliminați primele 50% din origini (România de la mai 2019). Cît sînt RMSE la 12 luni pentru ARFIMA și pentru mersul aleator în România?',
                  '0,57 și 0,64 pp pe lună (față de 0,53 și 0,61 pe toate originile): mersul aleator rămîne cel mai slab.'),
  'rv-table': ('Fixați a = 0,5 (m = 17). Între ce valori sînt cele șase estimări?',
               'Între 0,36 (ML, BET) și 0,74 (GPH, BET), față de 0,36–0,44 în curs. Cu puține frecvențe estimările semiparametrice se împrăștie; ML nu depinde de m.'),
  'wx-har': ('Încheiați eșantionul în 2008. Care componentă are coeficientul cel mai mare și cît este β̂<sub>d</sub>?',
             'Componenta săptămînală: β̂<sub>w</sub> = 0,62; β̂<sub>d</sub> = 0,10, jumătate din 0,20 pe 2000–2026.'),
  'har-table': ('Ridicați d cu 0,3 (la 0,88). Cum se compară ponderile lui (1 − L)<sup>d</sup> la lagurile 1 și 22 cu HAR?',
                'Lagul 1: 0,88 față de 0,31; lagul 22: 0,0003 față de 0,008. Un d mai mare pune mai multă pondere pe primul lag și mai puțină pe coadă.'),
 },
 'en': {
  'memory-data': ('For how many consecutive months, at most, does US unemployment stay on one side of its 1948–2026 mean?',
                  '106 months, almost nine years; the Nile stays 11 years on one side of its mean, Romanian 12-month inflation 103 months. For a series without memory such long runs would be very rare.'),
  'memory-acf': ('Drop the first 50% of each sample (US inflation then starts in November 1986). How many of the 48 autocorrelations of US monthly inflation stay above the band?',
                 '11 of 48, against 48 of 48 on 1947–2026; ρ̂(10) falls from 0.38 to 0.16. Much of the memory of the full series comes from the 1970s and early 1980s.'),
  'decay': ('Set d = 0.2. What is ρ(10) for ARFIMA(0, 0.2, 0) and for the AR(1) with the same ρ(1) = 0.25?',
            '0.064 against 9.5 · 10⁻⁷. With a small d the hyperbolic tail is low, but still five orders of magnitude above the exponential one.'),
  'weights': ('Move the blue curve to d = 0.9. What share of a unit shock is left after 100 periods?',
              'ψ₁₀₀ = 0.59 (0.74 after 10 periods), against 0.9¹⁰⁰ = 2.7 · 10⁻⁵ for the AR(1) with φ = 0.9. The series reverts to its mean, but very slowly.'),
  'ffd': ('Choose the threshold 10⁻³ (a shorter window). From which d does the ADF statistic pass the 5% critical value, and with how many weights?',
          'From d = 0.35, with 61 weights, against 0.25 with 445. The truncated weights no longer sum to 0, so the filter keeps part of the level and a larger d is needed for stationarity.'),
  'arfima-paths': ('Set d = 0.45. What is the theoretical variance of the series (σ² = 1)?',
                   'Γ(0.1)/Γ(0.55)² = 3.64. The variance of the 1000-observation path is only 2.31: the sample mean absorbs part of the slow component.'),
  'fbm': ('Move H to 0.05 and 0.95 (Δ = 0.45). What is the lag-1 autocorrelation of the increments for H = 0.95, in theory and in the sample?',
          'In theory 2<sup>0.9</sup> − 1 = 0.87, in the sample 0.72. With such strong persistence the mean of the 1000 increments absorbs part of the dependence, so the sample ACF is biased downwards.'),
  'rs-dfa': ('Raise the smallest block to n = 60. What R/S slope do the S&P 500 absolute returns get?',
             '0.92, against 0.86 with n from 10; returns stay at 0.53. In small blocks the day-to-day noise of |r<sub>t</sub>| dominates; the persistent part shows in large blocks.'),
  'gph': ('Set a = 0.8. What do GPH and local Whittle give for Romanian monthly inflation (m = 85)?',
          'GPH 0.62 (SE 0.07), local Whittle 0.29 (SE 0.05). GPH works with log I, and the near-zero ordinates at the seasonal frequencies removed by the adjustment pull its slope up; local Whittle uses I itself and is not affected.'),
  'bandwidth': ('End both samples in August 2019. Between which values does local Whittle move for the US and for Romania (T = 176)?',
                'US: 0.35–0.54; Romania: 0.24–0.59. The conclusion of long memory stays, and the uncertainty of about ±0.15 is there without the years 2020–2026 too.'),
  'mc': ('Use only the first 30 replications. Which estimator is the most precise at d = 0.3?',
         'Whittle ML: standard deviation 0.033, against 0.076 (local Whittle) and 0.086 (GPH), the same order as with 300 replications (0.04, 0.08, 0.10).'),
  'arfima-fit': ('Raise d by 0.14, to 0.45. How much does the log-likelihood fall, and is 2Δℓ above 3.84?',
                 'ℓ falls from −135.9 to −139.8; 2Δℓ = 7.8 > 3.84, so d = 0.45 is rejected at 5%. The ARFIMA curve becomes too high at lags 12 and 24 (0.65 and 0.60).'),
  'forecast-path': ('Move the origin to June 2022. Over the 36 months observed after it, which forecast has the smaller RMSE?',
                    'ARFIMA: 3.37 pp, against 4.02 pp for the AR(12). Both started high, but the AR(12) extended the 2022 peak (6.2% a year ahead, against 5.4%), and inflation fell faster.'),
  'forecast': ('Drop the first 50% of the origins (Romania from May 2019). Does ARFIMA still beat the AR at every horizon in Romania?',
               'No: the ratio is 1.038 at 1 month (the AR is better), 0.999 at 12 and 0.976 at 24. The gain remains at the long horizons only.'),
  'vol-acf': ('Choose shuffled days. How many of the 250 autocorrelations of S&P 500 |r<sub>t</sub>| stay above the band, and what does local Whittle give?',
              '6 of 250 (against 227), and local Whittle 0.00. The shuffle keeps the distribution and destroys the order: the memory lies in the sequence of calm and turbulent days.'),
  'rv': ('Lower d by 0.30 (with φ fixed). By how much does the log-likelihood fall for the BET and for the S&P 500?',
         'By 43.8 (BET) and 33.4 (S&P 500): both values are clearly rejected. For the BET the model ACF at lag 24 would be 0.004, against 0.28 in the sample.'),
  'vol-models': ('Move k to 500 days. What are the FIGARCH and GARCH weights of the S&P 500?',
                 'FIGARCH 2.9 · 10⁻⁵, GARCH 2.8 · 10⁻³¹. Even with α + β = 0.995 the GARCH weights fall exponentially; the FIGARCH weights fall like a power.'),
  'spurious': ('Show the 5%–95% band. With a shift of one standard deviation, what is the 5% quantile of d̂ (local Whittle)?',
               '0.31, above the 95% quantile without a shift (0.07): the distributions do not overlap, so every one of the 200 samples with a break shows “long memory”.'),
  'nile-break': ('Move the break to 1930. What d (exact ML) is left after removing the two means?',
                 '0.35, almost as in the raw series (0.36). A break placed in another year does not explain the memory; only the 1898 break brings it down to 0.05.'),
  'rolling': ('Halve the windows (10 years, 500 days). When and how high does d of US inflation peak?',
              '0.64 in the window ending July 1974, against 0.71 in July 1980 with 20-year windows; the band of no memory widens to [−0.33, 0.25].'),
  'wx-decay': ('Set d = 0.1. How many times larger is the sum of the first 1000 ARFIMA autocorrelations than the full sum of the AR(1)?',
               '14 times: 1.74 against 0.125. Even a weak long memory accumulates far more than a short memory with the same ρ(1).'),
  'wx-weights': ('Set d = 1.5. What are π₁ and π₂, and what sign has π₂?',
                 'π₁ = −1.5 and π₂ = −1.5 · (1 − 1.5)/2 = +0.375. For d > 1 the factor (k − 1 − d) is negative at k = 2, so the weights change sign after the first lag.'),
  'd-range': ('Set d = 0.6. Is the series stationary, and what is its first difference?',
              'It is not stationary (d ≥ 0.5) but it reverts to its mean: ψ₁₀₀ = 0.11. The first difference is ARFIMA(0, −0.4, 0), anti-persistent with ρ(1) = −0.29.'),
  'wx-acf': ('Set d = 0.45. What are ρ(1) and ρ(10)?',
             'ρ(1) = 0.818 and ρ(10) = 0.652; an AR(1) with the same ρ(1) has ρ(10) = 0.134. The variance rises to 3.64.'),
  'wx-rs': ('Set x₈ = 0.5. What does R/S become?',
            'The mean becomes 0.275, R = 0.575, S = 0.386, so R/S = 1.49 (against 1.79). One value changes the mean, hence all cumulative deviations.'),
  'wx-gph': ('Romanian inflation with a = 0.5 (m = 16): what are d̂ and its SE?',
             'd̂ = 0.56, SE 0.16. With fewer frequencies the estimate moves towards 0.5, and the standard error grows like 1/√m.'),
  'gph-table': ('Set a = 0.5. Does the GPH interval of the Nile still exclude 0?',
                'No: d̂ = 0.39, SE 0.20, interval [−0.01, 0.79]. With m = 10 GPH no longer separates d from 0; local Whittle [0.15, 0.77] still does.'),
  'wx-lw': ('Choose US unemployment (a = 0.65). Is the standard error 1/(2√m) valid?',
            'No: d̂ = 0.91 > 3/4, outside the range of the normal approximation. The interval [0.80, 1.01] contains 1 anyway, so mean reversion is not established.'),
  'fit-table': ('End the sample in December 2018 (T = 168). Which model has the smallest BIC, and what is d̂?',
                'ARFIMA(0,d,0), with d̂ = 0.24: BIC 191.2 against 195.5 for ARMA(1,1). The conclusion of the lecture holds on the shorter sample too.'),
  'memory-table': ('Set a = 0.5 (m = 30 for US unemployment). Does the local Whittle interval still contain 1?',
                   'No: d̂ = 0.64, interval [0.46, 0.82], and the SE is valid (d̂ < 3/4). With a = 0.65 (m = 85): 0.91, [0.80, 1.01]. The verdict depends on the bandwidth.'),
  'ratio-table': ('Drop the first 50% of the origins (Romania from May 2019). What are the 12-month RMSE of ARFIMA and of the random walk for Romania?',
                  '0.57 and 0.64 pp per month (against 0.53 and 0.61 on all origins): the random walk stays the worst.'),
  'rv-table': ('Set a = 0.5 (m = 17). Between which values do the six estimates lie?',
               'Between 0.36 (ML, BET) and 0.74 (GPH, BET), against 0.36–0.44 in the lecture. With few frequencies the semiparametric estimates scatter; ML does not depend on m.'),
  'wx-har': ('End the sample in 2008. Which component has the largest coefficient, and what is β̂<sub>d</sub>?',
             'The weekly one: β̂<sub>w</sub> = 0.62; β̂<sub>d</sub> = 0.10, half of 0.20 on 2000–2026.'),
  'har-table': ('Raise d by 0.3 (to 0.88). How do the weights of (1 − L)<sup>d</sup> at lags 1 and 22 compare with HAR?',
                'Lag 1: 0.88 against 0.31; lag 22: 0.0003 against 0.008. A larger d puts more weight on the first lag and less on the tail.'),
 },
}

# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with the readout of the block at the state the question names (verified by a node script).
PREDICT = {
 'memory-data': (1, [('Circa 12 luni: un ciclu economic.', 'About 12 months: one business cycle.'), ('106 luni, aproape nouă ani.', '106 months, almost nine years.'),
                     ('Cel mult 3 luni: datele lunare trec mereu prin medie.', 'At most 3 months: monthly data cross the mean all the time.')]),
 'memory-acf': (1, [('Tot 48: memoria este o proprietate a seriei, nu a eșantionului.', 'Still 48: memory is a property of the series, not of the sample.'), ('11 din 48.', '11 of 48.'),
                    ('Niciuna: după 1986 inflația este zgomot alb.', 'None: after 1986 inflation is white noise.')]),
 'decay': (0, [('0,064 și 9,5 · 10⁻⁷.', '0.064 and 9.5 · 10⁻⁷.'), ('Amîndouă circa 0,06: cu d mic cele două legi coincid.', 'Both about 0.06: with a small d the two laws coincide.'),
               ('0,25¹⁰ pentru amîndouă: ρ(10) = ρ(1)¹⁰ în orice model staționar.', '0.25¹⁰ for both: ρ(10) = ρ(1)¹⁰ in any stationary model.')]),
 'weights': (0, [('0,59.', '0.59.'), ('0,9¹⁰⁰ ≈ 2,7 · 10⁻⁵, ca la AR(1) cu φ = 0,9.', '0.9¹⁰⁰ ≈ 2.7 · 10⁻⁵, as for an AR(1) with φ = 0.9.'),
                 ('1: cu d aproape de 1 șocul este permanent.', '1: with d close to 1 the shock is permanent.')]),
 'ffd': (1, [('Tot 0,25: lungimea ferestrei nu contează.', 'Still 0.25: the length of the window does not matter.'), ('0,35, cu 61 de ponderi.', '0.35, with 61 weights.'),
             ('0,05: mai puține ponderi fac seria staționară mai devreme.', '0.05: fewer weights make the series stationary sooner.')]),
 'arfima-paths': (1, [('1, varianța șocurilor.', '1, the variance of the shocks.'), ('3,64.', '3.64.'), ('2,07, ca pentru d = 0,4.', '2.07, as for d = 0.4.')]),
 'fbm': (0, [('0,87 în teorie, 0,72 în eșantion.', '0.87 in theory, 0.72 in the sample.'), ('0,95: ρ(1) este egal cu H.', '0.95: ρ(1) equals H.'),
             ('0: incrementele mișcării browniene sînt independente.', '0: the increments of Brownian motion are independent.')]),
 'rs-dfa': (0, [('0,92.', '0.92.'), ('Tot 0,86: panta nu depinde de blocuri.', 'Still 0.86: the slope does not depend on the blocks.'),
                ('0,5: blocurile mari elimină memoria.', '0.5: large blocks remove the memory.')]),
 'gph': (1, [('Amîndoi circa 0,3.', 'Both about 0.3.'), ('GPH 0,62, Whittle local 0,29.', 'GPH 0.62, local Whittle 0.29.'),
             ('Amîndoi peste 0,5: mai multe frecvențe ridică mereu d.', 'Both above 0.5: more frequencies always raise d.')]),
 'bandwidth': (0, [('SUA 0,35–0,54, România 0,24–0,59.', 'US 0.35–0.54, Romania 0.24–0.59.'), ('Amîndouă aproape de 0: înainte de 2020 inflația nu avea memorie.', 'Both near 0: before 2020 inflation had no memory.'),
                   ('Amîndouă aproape de 1: eșantioanele mai scurte arată ca rădăcini unitare.', 'Both near 1: shorter samples look like unit roots.')]),
 'mc': (0, [('Whittle ML (0,033).', 'Whittle ML (0.033).'), ('GPH: cu puține eșantioane estimatorii semiparametrici sînt mai preciși.', 'GPH: with few samples the semiparametric estimators are more precise.'),
            ('Cu 30 de eșantioane nu se poate face nicio ordonare.', 'No ranking is possible with 30 samples.')]),
 'arfima-fit': (1, [('Abia: 2Δℓ < 1, verosimilitatea este plată în d.', 'Barely: 2Δℓ < 1, the likelihood is flat in d.'), ('2Δℓ = 7,8: respins la 5%.', '2Δℓ = 7.8: rejected at 5%.'),
                    ('Crește: un d mai mare potrivește mai bine coada lentă.', 'It rises: a larger d fits the slow tail better.')]),
 'forecast-path': (0, [('ARFIMA: 3,37 față de 4,02.', 'ARFIMA: 3.37 against 4.02.'), ('AR(12), pentru că are mai mulți parametri.', 'The AR(12), because it has more parameters.'),
                       ('Sînt egale: ambele revin la aceeași medie.', 'They are equal: both revert to the same mean.')]),
 'forecast': (1, [('Da, la toate orizonturile.', 'Yes, at every horizon.'), ('Nu: 1,038 la o lună, 0,976 la 24.', 'No: 1.038 at 1 month, 0.976 at 24.'),
                  ('Nu: AR este mai precis la toate orizonturile.', 'No: the AR is better at every horizon.')]),
 'vol-acf': (1, [('Tot circa 227: cozile groase ale lui |rₜ| creează memoria.', 'Still about 227: the fat tails of |rₜ| create the memory.'), ('6 din 250, Whittle local 0,00.', '6 of 250, local Whittle 0.00.'),
                 ('Niciuna, iar Whittle local devine −0,5.', 'None, and local Whittle becomes −0.5.')]),
 'rv': (1, [('Cu mai puțin de 2: verosimilitatea este plată în d.', 'By less than 2: the likelihood is flat in d.'), ('Cu 43,8 (BET) și 33,4 (S&P 500).', 'By 43.8 (BET) and 33.4 (S&P 500).'),
            ('Crește la S&P 500: ACF de selecție dispare după 20 de luni, deci un d mai mic se potrivește mai bine.', 'It rises for the S&P 500: the sample ACF dies out after 20 months, so a lower d fits better.')]),
 'vol-models': (1, [('Amîndouă sînt zero.', 'Both are zero.'), ('FIGARCH 2,9 · 10⁻⁵, GARCH 2,8 · 10⁻³¹.', 'FIGARCH 2.9 · 10⁻⁵, GARCH 2.8 · 10⁻³¹.'),
                    ('GARCH este mai mare: α + β = 0,995 este aproape de 1.', 'GARCH is larger: α + β = 0.995 is close to 1.')]),
 'spurious': (1, [('Aproape 0: majoritatea eșantioanelor cu ruptură nu arată memorie.', 'Close to 0: most samples with a break show no memory.'), ('0,31, peste cuantila de 95% fără ruptură (0,07).', '0.31, above the 95% quantile without a break (0.07).'),
                  ('Peste 1: o ruptură arată ca o rădăcină unitară.', 'Above 1: a break looks like a unit root.')]),
 'nile-break': (1, [('0,05, ca în 1898: orice ruptură elimină memoria.', '0.05, as with 1898: any break removes the memory.'), ('0,35.', '0.35.'),
                    ('Negativ: eliminarea a două medii supradiferențiază seria.', 'Negative: removing two means over-differences the series.')]),
 'rolling': (1, [('0,71 în iulie 1980, ca la 20 de ani.', '0.71 in July 1980, as with 20 years.'), ('0,64 în iulie 1974.', '0.64 in July 1974.'),
                 ('Peste 1: ferestrele scurte dau mereu un d mai mare.', 'Above 1: short windows always give a higher d.')]),
 'wx-decay': (1, [('Cam la fel: cu d mic memoria este scurtă.', 'About the same: with a small d the memory is short.'), ('De 14 ori (1,74 față de 0,125).', '14 times (1.74 against 0.125).'),
                  ('De 105 ori, ca pentru d = 0,4.', '105 times, as for d = 0.4.')]),
 'wx-weights': (1, [('−1,5 și −0,375.', '−1.5 and −0.375.'), ('−1,5 și +0,375.', '−1.5 and +0.375.'), ('−1 și 0, ca pentru d = 1.', '−1 and 0, as for d = 1.')]),
 'd-range': (1, [('Staționară: d < 1.', 'Stationary: d < 1.'), ('Nestaționară, dar revine la medie; diferența este ARFIMA(0; −0,4; 0).', 'Non-stationary but mean-reverting; the difference is ARFIMA(0, −0.4, 0).'),
                 ('O rădăcină unitară: șocurile sînt permanente.', 'A unit root: shocks are permanent.')]),
 'wx-acf': (0, [('0,818 și 0,652.', '0.818 and 0.652.'), ('0,818 și 0,818¹⁰ = 0,134.', '0.818 and 0.818¹⁰ = 0.134.'), ('0,45 și 0,045.', '0.45 and 0.045.')]),
 'wx-rs': (0, [('1,49.', '1.49.'), ('Tot 1,79: o singură valoare nu schimbă amplitudinea.', 'Still 1.79: one value cannot change the range.'), ('0: abaterile se anulează.', '0: the deviations cancel.')]),
 'wx-gph': (1, [('0,37 (SE 0,11), ca la m = 37.', '0.37 (SE 0.11), as with m = 37.'), ('0,56 (SE 0,16).', '0.56 (SE 0.16).'),
                ('0,56 (SE 0,08): mai puține puncte dau o eroare mai mică.', '0.56 (SE 0.08): fewer points give a smaller error.')]),
 'gph-table': (1, [('Da: d̂ este tot 0,39.', 'Yes: d̂ is still 0.39.'), ('Nu: [−0,01; 0,79] conține 0.', 'No: [−0.01, 0.79] contains 0.'),
                   ('Da, și acum exclude și 0,5.', 'Yes, and it now excludes 0.5 too.')]),
 'wx-lw': (1, [('Da: SE depinde doar de m.', 'Yes: the SE depends on m only.'), ('Nu: d̂ = 0,91 > 3/4.', 'No: d̂ = 0.91 > 3/4.'), ('Da, iar d < 1 este demonstrat.', 'Yes, and d < 1 is established.')]),
 'fit-table': (1, [('ARMA(1,1): pe un eșantion mai scurt cîștigă ARMA.', 'ARMA(1,1): on a shorter sample the ARMA wins.'), ('ARFIMA(0,d,0), d̂ = 0,24.', 'ARFIMA(0,d,0), d̂ = 0.24.'),
                   ('ARFIMA(1,d,1): cîștigă modelul cel mai general.', 'ARFIMA(1,d,1): the most general model wins.')]),
 'memory-table': (1, [('Tot 0,91; intervalul conține 1.', 'Still 0.91; the interval contains 1.'), ('0,64, [0,46; 0,82]: exclude 1.', '0.64, [0.46, 0.82]: it excludes 1.'),
                      ('Peste 1: mai puține frecvențe arată rădăcina unitară.', 'Above 1: fewer frequencies reveal the unit root.')]),
 'ratio-table': (1, [('0,53 și 0,61, ca pe toate originile.', '0.53 and 0.61, as on all origins.'), ('0,57 și 0,64.', '0.57 and 0.64.'),
                     ('Mersul aleator cîștigă: după 2019 inflația se comportă ca un mers aleator.', 'The random walk wins: after 2019 inflation behaves like a random walk.')]),
 'rv-table': (1, [('0,36–0,44, ca în curs.', '0.36–0.44, as in the lecture.'), ('0,36–0,74.', '0.36–0.74.'), ('Toate peste 0,75.', 'All above 0.75.')]),
 'wx-har': (1, [('Componenta zilnică, cu coeficientul 0,20.', 'The daily one, with coefficient 0.20.'), ('Componenta săptămînală, cu 0,62; cea zilnică are 0,10.', 'The weekly one, with 0.62; the daily one has 0.10.'),
                ('Componenta lunară: în anii calmi domină orizontul lung.', 'The monthly one: in calm years the long horizon dominates.')]),
 'har-table': (1, [('Mai mari la toate lagurile.', 'Larger at every lag.'), ('0,88 la lagul 1, 0,0003 la lagul 22.', '0.88 at lag 1, 0.0003 at lag 22.'),
                   ('Egale cu HAR: d nu schimbă ponderile.', 'Equal to HAR: d does not change the weights.')]),
}

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slide „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('Cît sînt π₁ și π₂ pentru (1 − L)^0,2?', 'What are π₁ and π₂ of (1 − L)^0.2?'), key=0, opts=[
  (('π₁ = −0,2, π₂ = −0,2 · 0,8/2 = −0,08.', 'π₁ = −0.2, π₂ = −0.2 · 0.8/2 = −0.08.'),
   ('Recursia πₖ = πₖ₋₁(k − 1 − d)/k cu d = 0,2 dă exact aceste valori.', 'The recursion πₖ = πₖ₋₁(k − 1 − d)/k with d = 0.2 gives exactly these values.')),
  (('π₁ = −0,2, π₂ = −0,04 (adică 0,2²).', 'π₁ = −0.2, π₂ = −0.04 (that is 0.2²).'),
   ('Ponderile nu sînt puteri ale lui d: π₂ = −d(1 − d)/2, nu −d².', 'The weights are not powers of d: π₂ = −d(1 − d)/2, not −d².')),
  (('π₁ = −0,2, π₂ = 0.', 'π₁ = −0.2, π₂ = 0.'),
   ('Doar pentru d întreg (d = 1) ponderile se opresc după primul lag; pentru d fracționar toate sînt nenule.', 'Only for an integer d (d = 1) do the weights stop after the first lag; for a fractional d all are non-zero.'))]),
 dict(q=('Este staționar un ARFIMA(0; 0,7; 0)?', 'Is ARFIMA(0, 0.7, 0) stationary?'), key=1, opts=[
  (('Da: d < 1.', 'Yes: d < 1.'), ('Staționaritatea cere d < 1/2; d < 1 dă doar revenirea la medie.', 'Stationarity needs d < 1/2; d < 1 gives mean reversion only.')),
  (('Nu, dar revine la medie; prima diferență este ARFIMA(0; −0,3; 0), staționară și antipersistentă.', 'No, but it is mean-reverting; its first difference is ARFIMA(0, −0.3, 0), stationary and anti-persistent.'),
   ('d ≥ 0,5 înseamnă nestaționar; răspunsurile la șocuri tind totuși la 0, iar (1 − L)x are d − 1 = −0,3.', 'd ≥ 0.5 means non-stationary; the responses to shocks still tend to 0, and (1 − L)x has d − 1 = −0.3.')),
  (('Nu, și șocurile sînt permanente, ca la o rădăcină unitară.', 'No, and shocks are permanent, as with a unit root.'),
   ('Șocurile sînt permanente doar pentru d = 1; pentru d = 0,7 ponderea după 100 de perioade este 0,19.', 'Shocks are permanent only for d = 1; for d = 0.7 the weight after 100 periods is 0.19.'))]),
 dict(q=('GPH dă d̂ = 0,35 pentru o serie cu o schimbare mare de nivel. Este aceasta o dovadă de memorie lungă?',
         'GPH gives d̂ = 0.35 on a series with a large level shift. Is this evidence of long memory?'), key=1, opts=[
  (('Da: d̂ este semnificativ peste 0.', 'Yes: d̂ is significantly above 0.'),
   ('Și o serie fără memorie, cu o singură ruptură, dă un d̂ semnificativ: testul nu separă cele două explicații.', 'A series without memory and with a single break also gives a significant d̂: the test does not separate the two explanations.')),
  (('Nu, prin ea însăși: o ruptură produce aceeași putere la frecvențele joase; reestimați după eliminarea schimbării și pe subeșantioane.',
    'Not by itself: a break produces the same low-frequency power; re-estimate after removing the shift and on subsamples.'),
   ('Pe Nil, eliminarea rupturii din 1898 coboară d̂ de la 0,36 la 0,05.', 'On the Nile, removing the 1898 break lowers d̂ from 0.36 to 0.05.')),
  (('Nu: GPH nu se poate calcula pe o serie cu ruptură.', 'No: GPH cannot be computed on a series with a break.'),
   ('Estimatorul se calculează pe orice serie; problema este interpretarea rezultatului.', 'The estimator can be computed on any series; the problem is the interpretation of the result.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts]) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch8-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH8_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH8_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n')


_predict_js()


if __name__ == '__main__':
    import interactive_common
    page_spec = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA', 'TEXT_JS', 'PREDICT', 'TICKET')})
    page_spec.widgets = lambda lang: {a: v for a, v in widgets_all(lang).items() if a in {b for b, _, _ in BLOCKS}}
    interactive_common.build(page_spec)
    for lg in ('en', 'ro'):
        _post(lg)
