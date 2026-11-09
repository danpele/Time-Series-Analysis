"""Build the interactive pages of Chapter 14 (multivariate GARCH models): interactive/ch14.html (EN) and interactive/ch14_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its interactive version on
the right. The worked examples of the lecture (correlation and portfolio risk, a positive definite H, one step of a diagonal
BEKK, one step of the DCC recursion, the half-life of correlation shocks, a two-asset VaR 1% from H = DRD, a hedge ratio) and
its result tables (asynchronous trading, the parameter counts, the two GARCH fits, the estimated DCC, the crises, the five-asset
DCC, the backtest, the hedging effectiveness) get blocks of their own (EXTRA), placed by slide, each with a chart; on the web
lecture each of them travels inside a chart block and ch14.js moves it to its own frame. Logic in interactive/ch14.js, inputs
in interactive/ch14_data.js (Quantlets/Ch_14/export_interactive.py), slide images in interactive/slides/.
"""
import json
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 14

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_14'
TEX = {'ro': 'RO/Cursuri/capitol14_modele_garch_multivariate', 'en': 'EN/Courses/chapter14_multivariate_garch_models'}

_D = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'interactive', 'ch14_data.js'), encoding='utf-8').read()
_D = json.loads(_D[_D.index('{'):_D.rindex('}') + 1])
A_DCC, B_DCC = _D['numbers']['dcc']['a'], _D['numbers']['dcc']['b']
A_PAN = _D['numbers']['panel']['a']
A_HED = _D['numbers']['hedge']['a']
del _D

# (anchor, chart file, section key); order of the slides
BLOCKS = [('divers', 'tsa_ch14_diversification', 'wm'), ('rolling', 'tsa_ch14_rolling_corr', 'wm'),
          ('params', 'tsa_ch14_param_count', 'vb'),
          ('step1', 'tsa_ch14_garch_step1', 'sd'), ('dcc', 'tsa_ch14_dcc_sp_dax', 'sd'), ('zoom', 'tsa_ch14_crisis_zoom', 'sd'),
          ('panel', 'tsa_ch14_panel_corr', 'fm'), ('heat', 'tsa_ch14_panel_heatmap', 'fm'),
          ('var', 'tsa_ch14_var_backtest', 'pv'), ('hedge', 'tsa_ch14_hedge', 'hr')]

# worked examples and result tables: (anchor, frame title EN, frame title RO, section key, host chart block)
EXTRA = [('wx-div', 'Worked example: correlation and portfolio risk', 'Exemplu rezolvat: corelația și riscul portofoliului', 'wm', 'divers'),
         ('t-async', 'Asynchronous trading and the bias of daily correlations', 'Tranzacționarea asincronă și deplasarea corelațiilor zilnice', 'wm', 'rolling'),
         ('wx-pd', 'Two conditions on Hₜ', 'Două condiții pentru Hₜ', 'fw', 'params'),
         ('wx-bekk', 'Worked example: one step of a diagonal BEKK', 'Exemplu rezolvat: un pas al unui BEKK diagonal', 'vb', 'params'),
         ('t-dim', 'The curse of dimensionality', 'Problema dimensionalității', 'vb', 'params'),
         ('wx-dcc', 'Worked example: one step of the DCC recursion', 'Exemplu rezolvat: un pas al recurenței DCC', 'cd', 'step1'),
         ('wx-hl', 'Reading a and b', 'Interpretarea parametrilor a și b', 'cd', 'step1'),
         ('t-step1', 'Step 1: two univariate GARCH models', 'Pasul 1: două modele GARCH univariate', 'sd', 'step1'),
         ('t-dcc', 'Step 2: the estimated DCC', 'Pasul 2: modelul DCC estimat', 'sd', 'dcc'),
         ('t-crisis', 'Correlations in crises', 'Corelațiile în crize', 'sd', 'zoom'),
         ('t-panel', 'A five-asset DCC', 'Un model DCC cu cinci active', 'fm', 'panel'),
         ('wx-var', 'Worked example: VaR 1% of two assets', 'Exemplu rezolvat: VaR 1% pentru două active', 'pv', 'var'),
         ('t-var', 'Interpreting the backtest', 'Interpretarea backtesting-ului', 'pv', 'var'),
         ('wx-hedge', 'Worked example: a hedge ratio', 'Exemplu rezolvat: un raport de acoperire', 'hr', 'hedge'),
         ('t-hedge', 'Interpreting the hedge ratios', 'Interpretarea rapoartelor de acoperire', 'hr', 'hedge')]

PFX = {'divers': 'dv', 'rolling': 'ro', 'params': 'pc', 'step1': 's1', 'dcc': 'dc', 'zoom': 'zm', 'panel': 'pn', 'heat': 'ht', 'var': 'vb', 'hedge': 'hg',
       'wx-div': 'xd', 't-async': 'ta', 'wx-pd': 'xp', 'wx-bekk': 'xb', 't-dim': 'tm', 'wx-dcc': 'xc', 'wx-hl': 'xh', 't-step1': 't1', 't-dcc': 't2',
       't-crisis': 'tc', 't-panel': 'tp', 'wx-var': 'xv', 't-var': 'tv', 'wx-hedge': 'xg', 't-hedge': 'th'}

SECTIONS = {
 'ro': dict(wm='Rolul volatilității multivariate', fw='Cadrul GARCH multivariat', vb='Modelele VEC și BEKK', cd='Corelații condiționate constante și dinamice',
            sd='DCC pentru S&P 500 și DAX', fm='Cinci piețe: S&P 500, DAX, BET, EUR/RON, Bitcoin', pv='VaR 1% al unui portofoliu cu DCC', hr='Rapoarte dinamice de acoperire'),
 'en': dict(wm='Why multivariate volatility', fw='The multivariate GARCH framework', vb='VEC and BEKK models', cd='Constant and dynamic conditional correlations',
            sd='DCC for the S&P 500 and the DAX', fm='Five markets: S&P 500, DAX, BET, EUR/RON, Bitcoin', pv='Portfolio VaR 1% with DCC', hr='Dynamic hedge ratios'),
}

L = {
 'ro': dict(file='ch14_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 14 · Modele GARCH multivariate', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'casetele „Exemplu rezolvat” refac calculul cu cifrele capitolului.',
   src='Randamentele logaritmice zilnice (în %) ale eșantioanelor capitolului, în zilele comune de tranzacționare (S&P 500, DAX, BET, EUR/RON, Bitcoin; data/market, EODHD), '
       'nu sînt revizuite și reproduc exact cifrele din curs. Estimările provin din codul capitolului: modelele GARCH(1,1) ale pasului 1 (pachetul arch) și parametrii (a, b) '
       'ai pasului 2, cu erorile lor standard, pentru S&P 500 și DAX (2000–2026), pentru panelul cu cinci active (din 2015), pentru eșantionul de estimare 2000–2014 al VaR-ului '
       'și pentru eșantionul 2005–2014 al acoperirii. Browserul rulează din nou recurențele: filtrul de varianță GARCH cu parametrii dați, recurența DCC, corelațiile și pantele '
       'pe ferestre mobile, verosimilitatea de corelație, VaR-ul și testele Kupiec și Christoffersen. Variantele care nu apar în curs (alte ferestre, alte valori ale lui a și b, '
       'alt α al pasului 1 cu aceeași persistență, alte niveluri ale VaR-ului, randamente pe k zile, prognozele fără șocuri noi, mediile trimestriale) sînt rulări suplimentare '
       'ale codului capitolului pe aceleași date; graficele din casetele „Exemplu rezolvat”, „Tranzacționarea asincronă și deplasarea corelațiilor zilnice”, „Două condiții pentru Hₜ”, „Problema '
       'dimensionalității”, „Interpretarea parametrilor a și b”, „Pasul 1: două modele GARCH univariate”, „Pasul 2: modelul DCC estimat”, „Corelațiile în crize”, „Un model DCC '
       'cu cinci active”, „Interpretarea backtesting-ului” și „Interpretarea rapoartelor de acoperire” sînt grafice suplimentare pentru slide-uri care dau doar cifre.'),
 'en': dict(file='ch14.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 14 · Multivariate GARCH Models', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; '
        'the “Worked example” boxes redo the calculation with the numbers of the chapter.',
   src='The daily log returns (in %) of the chapter’s samples, on the common trading days (S&P 500, DAX, BET, EUR/RON, Bitcoin; data/market, EODHD), are not revised and '
       'reproduce the numbers of the lecture exactly. The estimates come from the chapter’s code: the GARCH(1,1) models of step 1 (arch package) and the parameters (a, b) of '
       'step 2 with their standard errors, for the S&P 500 and the DAX (2000–2026), for the five-asset panel (since 2015), for the 2000–2014 estimation sample of the VaR and '
       'for the 2005–2014 sample of the hedge. The browser runs the recursions again: the GARCH variance filter with the given parameters, the DCC recursion, the rolling '
       'correlations and slopes, the correlation log-likelihood, the VaR and the Kupiec and Christoffersen tests. Variants that do not appear in the lecture (other windows, '
       'other values of a and b, another α of step 1 with the same persistence, other VaR levels, k-day returns, the forecasts without new shocks, the quarterly means) are '
       'extra runs of the chapter’s code on the same data; the charts of the “Worked example” boxes, “Asynchronous trading and the bias of daily correlations”, “Two conditions on Hₜ”, “The curse of '
       'dimensionality”, “Reading a and b”, “Step 1: two univariate GARCH models”, “Step 2: the estimated DCC”, “Correlations in crises”, “A five-asset DCC”, “Interpreting '
       'the backtest” and “Interpreting the hedge ratios” are extra charts for slides that give only numbers.'),
}

CSS = ('<style>.c14x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c14x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c14x .c14f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c14x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c14x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c14x .wctl .ctl{max-width:360px;margin:.15rem 0 .2rem}'
       '.c14t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c14t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c14t th,.c14t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c14t thead tr:last-child th{border-bottom:1px solid #1A3A6E}'
       '.c14t tbody tr:nth-child(even){background:#F5F7FB}.c14t tbody tr.hi{background:#E3F4F7}.c14t .r{text-align:right}.c14t .c{text-align:center}'
       '.c14cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}</style>')


def tbl(i):
    return f'<div class="tbl c14t" id="{i}"></div>'


def wx(i, title, formula, ctl=''):
    """Worked-example box; `ctl` holds the controls that only drive the box (they stay out of the chart's controls bar)."""
    return f'<div class="c14x"><h4>{title}</h4>' + (f'<div class="wctl">{ctl}</div>' if ctl else '') + f'<div class="c14f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def box(ctl, inner):
    """A box with controls that only drive the table and the chart of the block."""
    return f'<div class="c14x"><div class="wctl">{ctl}</div>{inner}</div>'


def slider_any(i, label, lo, hi, val):
    """A range input with step="any", so that the initial value is the exact estimate of the lecture."""
    return f'<div class="ctl"><label for="{i}">{label}: <b id="{i}-v"></b></label><input type="range" id="{i}" min="{lo}" max="{hi}" step="any" value="{val}"></div>'


def widgets_all(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    W = s('Exemplu rezolvat', 'Worked example')
    c = lambda x: str(x).replace('.', ',') if ro else str(x)   # noqa: E731
    win = lambda i: slider(i, s('Fereastra mobilă (zile de tranzacționare; curs: 250)', 'Rolling window (trading days; lecture: 250)'), 60, 500, 10, 250)  # noqa: E731
    a_dcc = lambda i: slider_any(i, s(f'Parametrul a, cu b = {c(round(B_DCC, 4))} fixat (curs: estimarea {c(round(A_DCC, 4))})',  # noqa: E731
                                      f'Parameter a, with b = {round(B_DCC, 4)} fixed (lecture: the estimate {round(A_DCC, 4)})'), 0, 0.023, repr(A_DCC))
    a_pan = lambda i: slider_any(i, s(f'Parametrul a, cu b fixat (curs: estimarea {c(round(A_PAN, 4))})', f'Parameter a, with b fixed (lecture: the estimate {round(A_PAN, 4)})'),  # noqa: E731
                                 0, 0.0215, repr(A_PAN))
    mult = lambda i: slider(i, s('Multiplicatorul lui α din pasul 1, cu α + β păstrat (curs: 1)', 'Multiplier of α in step 1, with α + β kept (lecture: 1)'), 0.25, 2, 0.05, 1)  # noqa: E731
    lev = lambda i: select(i, s('Nivelul VaR-ului', 'Level of the VaR'), [('0.005', '0,5%' if ro else '0.5%'), ('0.01', s('1% (curs)', '1% (lecture)')),  # noqa: E731
                                                                          ('0.025', '2,5%' if ro else '2.5%'), ('0.05', '5%')], '0.01')
    assets = [(k, n) for k, n in (('sp500', 'S&P 500'), ('dax', 'DAX'), ('bet', 'BET'), ('eurron', 'EUR/RON'), ('btc', 'Bitcoin'))]
    return {
     'divers': (slider('dv-w', s('Ponderea primului activ (curs: 50%)', 'Weight of the first asset (lecture: 50%)'), 0, 1, 0.05, 0.5)
                + slider('dv-s', s('Volatilitatea celui de-al doilea activ (curs: 20%)', 'Volatility of the second asset (lecture: 20%)'), 10, 40, 1, 20) + play('dv'),
                CSS + charts('dv-chart')),
     'rolling': (win('ro-w') + play('ro'), charts('ro-chart')),
     'params': (slider('pc-n', s('Cel mai mare N de pe axă (curs: 50)', 'Largest N on the axis (lecture: 50)'), 5, 100, 5, 50) + play('pc'), charts('pc-chart')),
     'step1': (mult('s1-m') + play('s1'), charts('s1-chart')),
     'dcc': (a_dcc('dc-a') + win('dc-w') + play('dc'), charts('dc-chart')),
     'zoom': (mult('zm-m') + a_dcc('zm-a') + play('zm'), charts('zm-chart')),
     'panel': (a_pan('pn-a') + play('pn'), charts('pn-chart')),
     'heat': (a_pan('ht-a') + play('ht'), charts('ht-chart')),
     'var': (lev('vb-l') + select('vb-m', s('Modelul comparat cu matricea statică', 'Model compared with the static matrix'),
                                  [('DCC-FHS', s('DCC, FHS (curs)', 'DCC, FHS (lecture)')), ('DCC', s('DCC, distribuția Normală', 'DCC, Normal')), ('CCC', s('CCC, distribuția Normală', 'CCC, Normal'))], 'DCC-FHS')
             + play('vb'), charts('vb-chart')),
     'hedge': (win('hg-w') + slider_any('hg-a', s(f'Parametrul a al DCC (curs: estimarea {c(round(A_HED, 4))})', f'Parameter a of the DCC (lecture: the estimate {round(A_HED, 4)})'), 0, 0.012, repr(A_HED))
               + play('hg'), charts('hg-chart')),
     # ---------------------------------------------------------------- EXTRA
     'wx-div': (slider('xd-r', s('Corelația ρ (curs: 0,2 și 0,8)', 'Correlation ρ (lecture: 0.2 and 0.8)'), -1, 1, 0.05, 0.2) + play('xd'),
                charts('xd-chart') + wx('xd-ws', W, r'\sigma_p^2=w_1^2\sigma_1^2+w_2^2\sigma_2^2+2w_1w_2\rho\,\sigma_1\sigma_2,\qquad w_1=w_2=0.5',
                                        slider('xd-s', s('Volatilitatea fiecărui activ (curs: 20%)', 'Volatility of each asset (lecture: 20%)'), 5, 40, 1, 20))),
     't-async': (slider('ta-k', s('Orizontul k al randamentelor (zile de tranzacționare; curs: 1, adică zilnic)', 'Horizon k of the returns (trading days; lecture: 1, i.e. daily)'), 1, 10, 1, 1) + play('ta'),
                 charts('ta-chart') + box(select('ta-p', s('Perechile din tabel', 'Pairs in the table'), [('lec', s('cele trei din curs', 'the three of the lecture')), ('all', s('toate cele zece', 'all ten'))], 'lec'),
                                          tbl('ta-tb') + tbl('ta-tb2'))),
     'wx-pd': (slider('xp-h', s('Covarianța h<sub>12</sub> (curs: 2,5)', 'Covariance h<sub>12</sub> (lecture: 2.5)'), -3, 3, 0.1, 2.5) + play('xp'),
               charts('xp-chart') + wx('xp-ws', W, r'\mathbf{w}^\top\mathbf{H}\mathbf{w}>0\ \ \forall\,\mathbf{w}\neq\mathbf{0}\iff h_{11}>0,\ \ h_{11}h_{22}-h_{12}^2>0,\qquad \rho_{12}=\frac{h_{12}}{\sqrt{h_{11}h_{22}}}',
                                       slider('xp-a', s('Varianța h<sub>11</sub> (curs: 4; h<sub>22</sub> = 1)', 'Variance h<sub>11</sub> (lecture: 4; h<sub>22</sub> = 1)'), 0.5, 8, 0.5, 4))),
     'wx-bekk': (slider('xb-e', s('Șocul de ieri ε<sub>2,t−1</sub> (curs: −1,5)', 'Yesterday’s shock ε<sub>2,t−1</sub> (lecture: −1.5)'), -3, 3, 0.1, -1.5) + play('xb'),
                 charts('xb-chart') + wx('xb-ws', W, r'\begin{gathered}h_{12,t}=c_{12}^*+a_{11}a_{22}\,\varepsilon_{1,t-1}\varepsilon_{2,t-1}+b_{11}b_{22}\,h_{12,t-1}\\ c_{12}^*=0.02,\ a_{11}=0.30,\ a_{22}=0.25,\ b_{11}=0.94,\ b_{22}=0.95,\ h_{12,t-1}=0.5\end{gathered}',
                                         slider('xb-f', s('Șocul de ieri ε<sub>1,t−1</sub> (curs: −2)', 'Yesterday’s shock ε<sub>1,t−1</sub> (lecture: −2)'), -3, 3, 0.1, -2))),
     't-dim': (slider('tm-n', s('Numărul de active N din ultima coloană și din grafic', 'Number of assets N in the last column and in the chart'), 2, 50, 1, 5) + play('tm'),
               charts('tm-chart') + tbl('tm-tb')),
     'wx-dcc': (slider('xc-z', s('Reziduul de ieri z<sub>2,t−1</sub> (curs: −2,5)', 'Yesterday’s residual z<sub>2,t−1</sub> (lecture: −2.5)'), -3, 3, 0.1, -2.5) + play('xc'),
                charts('xc-chart') + wx('xc-ws', W, r'\begin{gathered}\mathbf{Q}_t=(1-a-b)\,\bar{\mathbf{Q}}+a\,\mathbf{z}_{t-1}\mathbf{z}_{t-1}^\top+b\,\mathbf{Q}_{t-1},\qquad \rho_{12,t}=\frac{q_{12,t}}{\sqrt{q_{11,t}\,q_{22,t}}}\\ \bar q_{12}=0.5,\ \ \mathbf{Q}_{t-1}=\begin{pmatrix}1&0.5\\0.5&1\end{pmatrix},\ \ z_{1,t-1}=-2\end{gathered}',
                                        slider('xc-a', s('Parametrul a (curs: 0,05)', 'Parameter a (lecture: 0.05)'), 0, 0.06, 0.01, 0.05)
                                        + slider('xc-b', s('Parametrul b (curs: 0,93)', 'Parameter b (lecture: 0.93)'), 0.8, 0.93, 0.01, 0.93))),
     'wx-hl': (slider('xh-p', s('Persistența a + b (curs: 0,98 și 0,99)', 'Persistence a + b (lecture: 0.98 and 0.99)'), 0.9, 0.995, 0.005, 0.98) + play('xh'),
               charts('xh-chart') + wx('xh-ws', W, r'q_{ij,t+k}-\bar q_{ij}\approx(a+b)^k\,(q_{ij,t}-\bar q_{ij}),\qquad \text{half-life}=\frac{\ln 0.5}{\ln(a+b)}')),
     't-step1': (select('t1-s', s('Seria din grafic', 'Series in the chart'), [('sp500', 'S&P 500'), ('dax', 'DAX')], 'sp500') + play('t1'),
                 charts('t1-chart') + tbl('t1-tb')),
     't-dcc': (slider_any('t2-b', s(f'Parametrul b fixat pe grafic (curs: estimarea {c(round(B_DCC, 4))})', f'Parameter b held fixed in the chart (lecture: the estimate {round(B_DCC, 4)})'), 0.9, 0.99, repr(B_DCC))
               + play('t2'), charts('t2-chart') + tbl('t2-tb')),
     't-crisis': (a_dcc('tc-a') + play('tc'),
                  charts('tc-chart') + box(select('tc-r', s('Corelația din tabel', 'Correlation in the table'), [('dcc', s('media corelației DCC (curs)', 'mean DCC correlation (lecture)')),
                                                                                                                ('roll', s('media corelației pe 250 de zile', 'mean 250-day rolling correlation')),
                                                                                                                ('sample', s('corelația randamentelor din perioadă', 'correlation of the returns within the period'))], 'dcc'), tbl('tc-tb'))),
     't-panel': (select('tp-s', s('Activul din grafic', 'Asset in the chart'), assets, 'sp500') + play('tp'), charts('tp-chart') + tbl('tp-tb')),
     'wx-var': (slider('xv-r', s('Corelația prognozată ρ<sub>t</sub> (curs: 0,3 și 0,9)', 'Forecast correlation ρ<sub>t</sub> (lecture: 0.3 and 0.9)'), -0.5, 1, 0.05, 0.3) + play('xv'),
                charts('xv-chart') + wx('xv-ws', W, r'\mathbf{H}_t=\mathbf{D}_t\mathbf{R}_t\mathbf{D}_t,\quad \sigma_{p,t}=\sqrt{\mathbf{w}^\top\mathbf{H}_t\mathbf{w}},\quad \mathrm{VaR}_{0.01,t}=-(\mu_p+z_{0.01}\,\sigma_{p,t}),\ \ z_{0.01}=-2.326',
                                        slider('xv-a', s('σ<sub>1</sub> zilnic (curs: 1,5%)', 'Daily σ<sub>1</sub> (lecture: 1.5%)'), 0.5, 4, 0.1, 1.5)
                                        + slider('xv-b', s('σ<sub>2</sub> zilnic (curs: 1,2%)', 'Daily σ<sub>2</sub> (lecture: 1.2%)'), 0.5, 4, 0.1, 1.2)
                                        + slider('xv-v', s('Valoarea portofoliului (milioane de lei; curs: 1)', 'Value of the portfolio (million lei; lecture: 1)'), 0.5, 10, 0.5, 1))),
     't-var': (lev('tv-l') + play('tv'), charts('tv-chart') + tbl('tv-tb')),
     'wx-hedge': (slider('xg-r', s('Corelația ρ (curs: 0,8)', 'Correlation ρ (lecture: 0.8)'), -1, 1, 0.05, 0.8) + play('xg'),
                  charts('xg-chart') + wx('xg-ws', W, r'h^*=\frac{\sigma_{sf}}{\sigma_f^2}=\rho\,\frac{\sigma_s}{\sigma_f},\qquad HE=1-\frac{\mathrm{Var}(r_s-h^*r_f)}{\mathrm{Var}(r_s)}=\rho^2',
                                          slider('xg-s', s('σ<sub>s</sub> zilnic (curs: 1,2%)', 'Daily σ<sub>s</sub> (lecture: 1.2%)'), 0.5, 3, 0.1, 1.2)
                                          + slider('xg-f', s('σ<sub>f</sub> zilnic (curs: 1,0%)', 'Daily σ<sub>f</sub> (lecture: 1.0%)'), 0.5, 3, 0.1, 1.0))),
     't-hedge': (win('th-w') + play('th'), charts('th-chart') + tbl('th-tb')),
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
    """Chart blocks for build_web_lecture.py: each host carries the web version of its EXTRA blocks (moved by ch14.js)."""
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
        img = f'ch14x_{anchor}_{lang}.jpg'
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
const de = n => { const r = Math.abs(Math.round(n)) % 100; return Number.isInteger(+n) && ((r === 0 && +n !== 0) || r >= 20) ? ' de' : ''; };
const big = n => String(Math.round(n)).replace('-', '−').replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
const dz = s => de(+String(s).replace(/[^\d]/g, ''));
const nz = (n, w1, wn) => `${big(n)}${de(n)} ${n === 1 ? w1 : wn}`;
const MOD = { VEC: 'VEC(1,1)', DVEC: 'VEC diagonal', BEKK: 'BEKK(1,1)', DBEKK: 'BEKK diagonal', SBEKK: 'BEKK scalar', CCC: 'CCC', DCC: 'DCC' };
window.CH14_TEXT = {
 big, plt: '&lt; 0,001',
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 days: n => nz(n, 'zi', 'zile'),
 mil: v => (v === 1 ? '1 milion de lei' : `${String(v).replace('.', ',')}${de(v)} milioane de lei`),
 model: { VEC: 'VEC(1,1)', DVEC: 'Diagonal VEC', BEKK: 'BEKK(1,1)', DBEKK: 'Diagonal BEKK', SBEKK: 'Scalar BEKK', CCC: 'CCC', DCC: 'DCC' },
 modelT: MOD,
 per: { calm: 'Calmă, 2017–2019', 2008: 'sept. 2008–martie 2009', 2020: 'febr.–aprilie 2020' },
 perEn: { calm: 'calm, 2017-2019', 2008: 'Sept 2008 - March 2009', 2020: 'Feb - April 2020' },
 vmod: { DCC: 'DCC, Normală', 'DCC-FHS': 'DCC, FHS', CCC: 'CCC, Normală', Static: 'Statică, Normală' },
 hm: { static: 'static OLS', rolling: 'rolling OLS', dcc: 'DCC' }, hmEn: { static: 'static OLS', rolling: 'rolling OLS', dcc: 'DCC' },
 hmT: { static: 'OLS static (2005–2014)', rolling: 'OLS pe ferestre mobile', dcc: 'DCC' },
 dv1: (w, s2, a, b, inc) => `Ponderi ${w} și restul, volatilități 20% și ${s2}%: σ<sub>p</sub> = ${a}% pentru ρ = 0,2 și ${b}% pentru ρ = 0,8 (cu ${inc} mai mult).`,
 dv2: (r1, r0, rm) => `ρ = 1: σ<sub>p</sub> = ${r1}%; ρ = 0: ${r0}%; ρ = −1: ${rm}%.`,
 xds1: (q, r, s2, t3, v) => `σ<sub>p</sub><sup>2</sup> = ${q} + ${q} + 2 · 0,25 · ${r} · ${s2} = ${q} + ${q} + ${t3} = ${v}.`,
 xds2: (v, s) => `σ<sub>p</sub> = √${v} = ${s}%.`,
 xds3: (ref, s, inc) => `Față de cazul ρ = 0,2 (σ<sub>p</sub> = ${ref}%), volatilitatea portofoliului este ${s}%, o variație relativă de ${inc}; activele și ponderile sînt aceleași.`,
 xd1: (r, s, v) => `Cu ρ = ${r} și volatilități de ${v}%, portofoliul 50/50 are σ<sub>p</sub> = ${s}%.`,
 ro1: (n, a, da, b, db, f) => `${n}: între ${a} (${da}) și ${b} (${db}); pe tot eșantionul ${f}.`,
 ro2: (w, s) => `Fereastra de ${w}; abaterea standard a corelației S&P 500 – DAX pe ferestre mobile: ${s}.`,
 taHead: ['Perechea', 'Zilnic', 'Săptămînal', 'Diferența'],
 taNote: (n, nw) => `Corelațiile randamentelor zilnice (${n}${dz(n)} zile comune de tranzacționare din 2015) și ale randamentelor săptămînale (${nw}${dz(nw)} săptămîni, închiderile de vineri).`,
 taHead2: ['Ziua', 'S&P 500', 'DAX'],
 ta1: (a, b, c, d) => `9 aprilie 2025: S&P 500 ${a}%, DAX ${b}%; 10 aprilie: S&P 500 ${c}%, DAX ${d}%: aceeași știre, în două zile diferite.`,
 ta2: (k, a, b, c, x) => `Randamente pe ${k}${x ? ' (rulare suplimentară)' : ''}: S&P 500 – BET ${a}, S&P 500 – DAX ${b}, BET – DAX ${c}.`,
 xps1: (h11, h12, d) => `Determinantul: h<sub>11</sub>h<sub>22</sub> − h<sub>12</sub><sup>2</sup> = ${h11} · 1 − ${h12}<sup>2</sup> = ${d}.`,
 xps2: (h12, h11, r) => `„Corelația”: ${h12}/√(${h11} · 1) = ${r}.`,
 xps3: (l1, l2, e1, e2, neg) => `Valorile proprii ale lui H: ${l1} și ${l2}; portofoliul cu ponderile (${e1}; ${e2}) are varianța ${l1}${neg ? ', negativă: H nu este o matrice de covarianță' : ''}.`,
 xp1: (pd, r, a, b) => (pd ? `H este pozitiv definită: ρ = ${r}, iar orice portofoliu are varianță pozitivă.` : `H nu este pozitiv definită (ρ = ${r})${a ? `: pe intervalul din grafic, portofoliul (w, 1 − w) are varianță negativă pentru w între ${a} și ${b}` : ''}.`),
 xbs1: (aa, bb) => `Pasul 1: a<sub>11</sub>a<sub>22</sub> = ${aa}, b<sub>11</sub>b<sub>22</sub> = ${bb}.`,
 xbs2: (e1, e2, p) => `Pasul 2: ε<sub>1,t−1</sub>ε<sub>2,t−1</sub> = (${e1}) · (${e2}) = ${p}.`,
 xbs3: (c, aa, p, bb, t2, t3, h) => `Pasul 3: h<sub>12,t</sub> = ${c} + ${aa} · ${p} + ${bb} · 0,5 = ${c} + ${t2} + ${t3} = ${h}.`,
 xbs4: (h, up) => `Covarianța ${up ? 'crește' : 'scade'} de la 0,5 la ${h}; un BEKK diagonal nu are transmitere a volatilității (a<sub>12</sub> = a<sub>21</sub> = 0).`,
 xb1: (h, lr) => `h<sub>12,t</sub> = ${h}; fără șocuri noi, prognoza revine spre nivelul de lungă durată c<sub>12</sub><sup>*</sup>/(1 − a<sub>11</sub>a<sub>22</sub> − b<sub>11</sub>b<sub>22</sub>) = ${lr} (rulare suplimentară).`,
 tmHead: n => ['Model', 'N = 2', 'N = 5', 'N = 10', 'N = 50', `N = ${n}`],
 tmNote: 'Numărul de parametri ai ecuației de varianță (mediile nu sînt numărate); CCC și DCC: 3 parametri GARCH pe activ, plus cele N(N − 1)/2 corelații, iar DCC încă 2.',
 tm1: (n, v, b, d, k) => `N = ${n}: ${k} elemente distincte în H<sub>t</sub>; VEC are ${v} parametri, BEKK ${b}, DCC ${d}.`,
 pc1: (n, v, b, db, d) => `N = ${n}: VEC(1,1) ${v} parametri, BEKK(1,1) ${b}, BEKK diagonal ${db}, DCC ${d}.`,
 pc2: r => `VEC are de ${r} ori mai mulți parametri decît DCC; optimizarea DCC lucrează doar cu a și b.`,
 xcs1: (a, b, c) => `Pasul 1: 1 − a − b = 1 − ${a} − ${b} = ${c}.`,
 xcs2: (c, a, z1, z2, b, t1, t2, t3, q) => `Pasul 2: q<sub>12,t</sub> = ${c} · 0,5 + ${a} · (${z1}) · (${z2}) + ${b} · 0,5 = ${t1} + ${t2} + ${t3} = ${q}.`,
 xcs3: (c, a, z1s, b, q11, z2s, q22) => `Pasul 3: q<sub>11,t</sub> = ${c} + ${a} · ${z1s} + ${b} = ${q11}; q<sub>22,t</sub> = ${c} + ${a} · ${z2s} + ${b} = ${q22}.`,
 xcs4: (q12, q11, q22, r) => `Pasul 4: ρ<sub>12,t</sub> = ${q12}/√(${q11} · ${q22}) = ${r}.`,
 xc1: (r, q, h) => `Corelația de mîine: ${r} (q<sub>12,t</sub> = ${q} nu este o corelație înainte de rescalare)${h ? `; o abatere se înjumătățește în ${h} zile` : ''}.`,
 xhs1: (p, l, h) => `ln 0,5/ln ${p} = −0,69315/(${l}) = ${h} zile.`,
 xhs2: (a, b) => `a + b = 0,98 dă ${a}${dz(a)} zile, a + b = 0,99 dă ${b}${dz(b)} zile.`,
 xhs3: (a, h, ap, hp) => `S&P 500 și DAX: a + b = ${a}, ${h}${dz(h)} zile; panelul cu cinci active: a + b = ${ap}, ${hp}${dz(hp)} zile.`,
 xh1: (p, h, s) => `a + b = ${p}: timpul de înjumătățire ${h} zile; după 20 de zile rămîne ${s} din abatere.`,
 t1Head: ['Indice', 'μ', 'ω', 'α', 'β', 'α + β', 'kurtosis r<sub>t</sub>', 'kurtosis z<sub>t</sub>'],
 t1Note: (n, a, b, rc, zc) => `Randamente logaritmice zilnice în %, ${n}${dz(n)} zile comune de tranzacționare, ${a} – ${b}; GARCH(1,1) cu medie constantă, QML gaussian; kurtosis: excesul de kurtosis. Corelația randamentelor ${rc}; corelația reziduurilor q̄<sub>12</sub> = ${zc}, estimația CCC.`,
 t11: (n, a, b, s) => `${n}: excesul de kurtosis scade de la ${a} (randamente) la ${b} (reziduuri standardizate, abaterea standard ${s}); cozile lui z<sub>t</sub> rămîn mai groase decît la distribuția Normală.`,
 s11: (a, da, b, db) => `Maximul volatilității: S&P 500 ${a}% pe ${da}, DAX ${b}% pe ${db}.`,
 s12: (lec, a1, b1, a2, b2, p1, p2) => `${lec ? 'Estimările din curs' : 'Rulare suplimentară a filtrului'}: S&P 500 α = ${a1}, β = ${b1}; DAX α = ${a2}, β = ${b2}; persistența rămîne ${p1} și ${p2}.`,
 t2Head: ['Mărimea', 'Estimație', 'SE'],
 t2hl: 'Timpul de înjumătățire al unui șoc de corelație (zile)', t2lr: 'LR față de CCC', t2crit: 'Valoarea critică χ<sup>2</sup>(2), 5%',
 t2Note: 'SE: erori standard din hessiana pasului 2 (prea mici, ignoră pasul 1). l<sub>C</sub>: partea de corelație a verosimilității, calculată din nou în browser.',
 t21: (b, a, lr, lrE) => `Cu b = ${b}, cel mai bun a din grilă este ${a} și dă LR = ${lr}; la estimarea (a, b) LR = ${lrE}.`,
 dc1: (a, ab, mn, dmn, mx, dmx, m) => `a = ${a} (a + b = ${ab}): corelația DCC variază între ${mn} (${dmn}) și ${mx} (${dmx}), în jurul mediei ${m}.`,
 dc2: (s, w, sr, lr) => `Abaterea standard: DCC ${s}, fereastra de ${w} ${sr}; LR față de CCC ${lr}.`,
 tcHead: k => ['Perioada', k === 'dcc' ? 'ρ̄<sup>DCC</sup>' : k === 'roll' ? 'ρ̄ pe 250 de zile' : 'corelația randamentelor', 'σ̄<sup>S&P</sup> (%)', 'Zile'],
 tcNote: k => (k === 'sample' ? 'Corelația de selecție a randamentelor crește mecanic cu volatilitatea (Forbes și Rigobon); DCC lucrează cu reziduurile standardizate z<sub>t</sub>.' : 'Media corelației și a volatilității anualizate a S&P 500 în fiecare perioadă.'),
 tc1: (a, b, c, va, vb, vc) => `Corelația DCC medie: perioada calmă ${a}, 2008–2009 ${b}, 2020 ${c}; volatilitatea medie: ${va}%, ${vb}%, ${vc}%.`,
 tc2: r => `Pe trimestre, corelația dintre logaritmul volatilității și corelația DCC medie este ${r} (rulare suplimentară).`,
 zm1: (v, dv, r0, r, dr) => `2008: volatilitatea atinge ${v}% pe ${dv}; corelația urcă de la ${r0} la ${r} pe ${dr}.`,
 zm2: (r, dr, v, dv, bf) => `2020: corelația atinge ${r} pe ${dr}, ${bf ? 'înaintea' : 'după'} maximului volatilității (${v}% pe ${dv}).`,
 zm3: lec => (lec ? 'Estimările din curs.' : 'Rulare suplimentară: pasul 1 și pasul 2 cu alți parametri decît estimările, aceleași date.'),
 tpHead: ['Activ', 'μ', 'ω', 'α', 'β', 'α + β'],
 tpNote: (n, d0, a, sa, b, sb, h, lr) => `Pasul 1: GARCH(1,1) pe randamentele zilnice din cele ${n}${dz(n)} zile comune de tranzacționare, din ${d0}. Pasul 2: â = ${a} (SE ${sa}), b̂ = ${b} (SE ${sb}); timpul de înjumătățire ${h}${dz(h)} zile; LR față de CCC ${lr}.`,
 tp1: (n, p, v, dv, lr) => `${n}: persistența ${p}; volatilitatea anualizată atinge ${v}% pe ${dv}; nivelul de lungă durată √(252 ω/(1 − α − β)) = ${lr}%.`,
 pn1: (sd, bd, mn, mx, dmx) => `S&P 500 și DAX: media ${sd}; BET și DAX: media ${bd}, între ${mn} și ${mx} (${dmx}).`,
 pn2: (mn, mx, dmx, m, be) => `S&P 500 și Bitcoin: de la ${mn} la ${mx} (${dmx}), media ${m}; BET și EUR/RON: media ${be}.`,
 pn3: (a, ab, h) => `a = ${a}, a + b = ${ab}: timpul de înjumătățire ${h} zile, același pentru toate cele 10 perechi.`,
 ht1: (a, b, n) => `Corelația medie dintre perechi: ${a} în perioada calmă, ${b} în crahul din 2020 (${n}${dz(n)} zile).`,
 ht2: (a, b, c, d, e, f) => `BET și DAX: ${a} → ${b}; S&P 500 și BET: ${c} → ${d}; DAX și Bitcoin: ${e} → ${f}.`,
 ht3: (a, b) => `BET și EUR/RON: ${a} → ${b}.`,
 xvs1: (s1, s2, r, a, b, h) => `H<sub>t</sub> = D<sub>t</sub>R<sub>t</sub>D<sub>t</sub>: h<sub>11</sub> = ${s1}<sup>2</sup> = ${a}, h<sub>22</sub> = ${s2}<sup>2</sup> = ${b}, h<sub>12</sub> = ${r} · ${s1} · ${s2} = ${h}.`,
 xvs2: (a, b, h, v2, v) => `σ<sub>p</sub><sup>2</sup> = 0,25 · ${a} + 0,25 · ${b} + 2 · 0,25 · ${h} = ${v2}, deci σ<sub>p</sub> = ${v}%.`,
 xvs3: (z, v, x) => `VaR 1% = ${z} · ${v} = ${x}% (media 0).`,
 xvs4: (V, k) => `Pentru ${V}: VaR 1% de aproximativ ${k} mii de lei.`,
 xvs5: (a, b, inc) => `Cu aceleași volatilități: ρ = 0,3 dă ${a}%, ρ = 0,9 dă ${b}% (cu ${inc} mai mult).`,
 xv1: (r, v, x, k) => `ρ<sub>t</sub> = ${r}: σ<sub>p</sub> = ${v}%, VaR 1% = ${x}%, adică ${k} mii de lei.`,
 vb1: (l, m, x, xs, e, n) => `VaR ${l}: ${m} are ${x}${dz(x)} încălcări, matricea statică ${xs}; așteptat ${e} în ${n}${dz(n)} zile.`,
 vb2: (m, r, lr, p, rs, lrs, ps) => `Kupiec: ${m} rata ${r}, LR<sub>uc</sub> = ${lr}, p ${p}; statică rata ${rs}, LR<sub>uc</sub> = ${lrs}, p ${ps}.`,
 vb3: (l, ms, m, mx, q) => `VaR ${l} mediu: static ${ms}% în fiecare zi, dinamic ${m}% în medie și pînă la ${mx}% într-un crah${q ? `; cuantila FHS q = ${q}` : ''}.`,
 tvHead: ['Model', 'Încălcări', 'Rata', 'LR<sub>uc</sub>', 'p', 'p indep.', 'febr.–apr. 2020'],
 tvNote: (l, e, n, ni, q) => `VaR ${l}; așteptat: aproximativ ${e}${dz(e)} încălcări în ${n}${dz(n)} zile. Parametrii sînt estimați pe 2000–2014 (${ni}${dz(ni)} zile); cuantila FHS q = ${q}.`,
 tv1: (l, n11, x, p11, c) => `VaR ${l} static: ${x}${dz(x)} încălcări, ${n11} dintre ele în ziua de după o încălcare (π̂<sub>11</sub> = ${p11}); ${c} în februarie–aprilie 2020.`,
 tv2: (f, d, q) => `Rata DCC cu FHS ${f} față de ${d} cu cuantila distribuției Normale; q = ${q}.`,
 xgs1: (r, s, f, c) => `Pasul 1: σ<sub>sf</sub> = ρ σ<sub>s</sub> σ<sub>f</sub> = ${r} · ${s} · ${f} = ${c}.`,
 xgs2: (c, f2, h) => `Pasul 2: h* = σ<sub>sf</sub>/σ<sub>f</sub><sup>2</sup> = ${c}/${f2} = ${h} unități din f pentru fiecare unitate din s.`,
 xgs3: (r, he, s, sh) => `Pasul 3: HE = ρ<sup>2</sup> = ${r}<sup>2</sup> = ${he}; volatilitatea poziției acoperite σ<sub>s</sub>√(1 − ρ<sup>2</sup>) = ${sh}% în loc de ${s}%.`,
 xgs4: (s, h) => `Dacă mîine ρ<sub>t</sub> = 0,5 și σ<sub>f,t</sub> = 2%: h = 0,5 · ${s}/2 = ${h}.`,
 xg1: (h, he, sh, s) => `h* = ${h}; acoperirea elimină ${he} din varianță: ${sh}% în loc de ${s}%.`,
 hg1: (s, m, mn, mx, dmx) => `Raportul static ${s}; după 2015 raportul DCC are media ${m}, între ${mn} și ${mx} (${dmx}).`,
 hg2: (s, w, r, d, a) => `Eficiența acoperirii, 2015–2026: statică ${s}, OLS pe ${w} ${r}, DCC (a = ${a}) ${d}.`,
 thHead: ['Metoda', 'Raportul mediu', 'Var(r<sub>s</sub> − h r<sub>f</sub>)', 'HE'],
 thNote: (n, v) => `În afara eșantionului: ${n}${dz(n)} zile din 2015; Var(r<sub>BET</sub>) = ${v}. HE = 1 − Var(acoperit)/Var(neacoperit).`,
 th1: (s, r, d, w) => `HE: statică ${s}, OLS pe ${w} ${r}, DCC ${d}.`,
};
})();
""",
 'en': r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const big = n => String(Math.round(n)).replace('-', '−').replace(/\B(?=(\d{3})+(?!\d))/g, ',');
const MOD = { VEC: 'VEC(1,1)', DVEC: 'Diagonal VEC', BEKK: 'BEKK(1,1)', DBEKK: 'Diagonal BEKK', SBEKK: 'Scalar BEKK', CCC: 'CCC', DCC: 'DCC' };
window.CH14_TEXT = {
 big, plt: '&lt; 0.001',
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 days: n => `${big(n)} ${n === 1 ? 'day' : 'days'}`,
 mil: v => `${v} million lei`,
 model: MOD, modelT: MOD,
 per: { calm: 'Calm, 2017–2019', 2008: 'Sept 2008–March 2009', 2020: 'Feb–April 2020' },
 perEn: { calm: 'calm, 2017-2019', 2008: 'Sept 2008 - March 2009', 2020: 'Feb - April 2020' },
 vmod: { DCC: 'DCC, Normal', 'DCC-FHS': 'DCC, FHS', CCC: 'CCC, Normal', Static: 'Static, Normal' },
 hm: { static: 'static OLS', rolling: 'rolling OLS', dcc: 'DCC' }, hmEn: { static: 'static OLS', rolling: 'rolling OLS', dcc: 'DCC' },
 hmT: { static: 'Static OLS (2005–2014)', rolling: 'Rolling OLS', dcc: 'DCC' },
 dv1: (w, s2, a, b, inc) => `Weights ${w} and the rest, volatilities 20% and ${s2}%: σ<sub>p</sub> = ${a}% for ρ = 0.2 and ${b}% for ρ = 0.8 (${inc} higher).`,
 dv2: (r1, r0, rm) => `ρ = 1: σ<sub>p</sub> = ${r1}%; ρ = 0: ${r0}%; ρ = −1: ${rm}%.`,
 xds1: (q, r, s2, t3, v) => `σ<sub>p</sub><sup>2</sup> = ${q} + ${q} + 2 · 0.25 · ${r} · ${s2} = ${q} + ${q} + ${t3} = ${v}.`,
 xds2: (v, s) => `σ<sub>p</sub> = √${v} = ${s}%.`,
 xds3: (ref, s, inc) => `Against the case ρ = 0.2 (σ<sub>p</sub> = ${ref}%), the portfolio volatility is ${s}%, a relative change of ${inc}; the assets and the weights are the same.`,
 xd1: (r, s, v) => `With ρ = ${r} and volatilities of ${v}%, the 50/50 portfolio has σ<sub>p</sub> = ${s}%.`,
 ro1: (n, a, da, b, db, f) => `${n}: between ${a} (${da}) and ${b} (${db}); full sample ${f}.`,
 ro2: (w, s) => `Window of ${w}; standard deviation of the rolling S&P 500 – DAX correlation: ${s}.`,
 taHead: ['Pair', 'Daily', 'Weekly', 'Difference'],
 taNote: (n, nw) => `Correlations of the daily returns (${n} common trading days since 2015) and of the weekly returns (${nw} weeks, Friday closes).`,
 taHead2: ['Day', 'S&P 500', 'DAX'],
 ta1: (a, b, c, d) => `9 April 2025: S&P 500 ${a}%, DAX ${b}%; 10 April: S&P 500 ${c}%, DAX ${d}%: the same news, on two different days.`,
 ta2: (k, a, b, c, x) => `Returns over ${k}${x ? ' (extra run)' : ''}: S&P 500 – BET ${a}, S&P 500 – DAX ${b}, BET – DAX ${c}.`,
 xps1: (h11, h12, d) => `Determinant: h<sub>11</sub>h<sub>22</sub> − h<sub>12</sub><sup>2</sup> = ${h11} · 1 − ${h12}<sup>2</sup> = ${d}.`,
 xps2: (h12, h11, r) => `“Correlation”: ${h12}/√(${h11} · 1) = ${r}.`,
 xps3: (l1, l2, e1, e2, neg) => `Eigenvalues of H: ${l1} and ${l2}; the portfolio with weights (${e1}, ${e2}) has variance ${l1}${neg ? ', negative: H is not a covariance matrix' : ''}.`,
 xp1: (pd, r, a, b) => (pd ? `H is positive definite: ρ = ${r}, and every portfolio has a positive variance.` : `H is not positive definite (ρ = ${r})${a ? `: on the plotted range, the portfolio (w, 1 − w) has a negative variance for w between ${a} and ${b}` : ''}.`),
 xbs1: (aa, bb) => `Step 1: a<sub>11</sub>a<sub>22</sub> = ${aa}, b<sub>11</sub>b<sub>22</sub> = ${bb}.`,
 xbs2: (e1, e2, p) => `Step 2: ε<sub>1,t−1</sub>ε<sub>2,t−1</sub> = (${e1}) · (${e2}) = ${p}.`,
 xbs3: (c, aa, p, bb, t2, t3, h) => `Step 3: h<sub>12,t</sub> = ${c} + ${aa} · ${p} + ${bb} · 0.5 = ${c} + ${t2} + ${t3} = ${h}.`,
 xbs4: (h, up) => `The covariance ${up ? 'rises' : 'falls'} from 0.5 to ${h}; a diagonal BEKK has no volatility spillovers (a<sub>12</sub> = a<sub>21</sub> = 0).`,
 xb1: (h, lr) => `h<sub>12,t</sub> = ${h}; without new shocks the forecast goes back to the long-run level c<sub>12</sub><sup>*</sup>/(1 − a<sub>11</sub>a<sub>22</sub> − b<sub>11</sub>b<sub>22</sub>) = ${lr} (extra run).`,
 tmHead: n => ['Model', 'N = 2', 'N = 5', 'N = 10', 'N = 50', `N = ${n}`],
 tmNote: 'Number of parameters of the variance equation (the means are not counted); CCC and DCC: 3 GARCH parameters per asset plus the N(N − 1)/2 correlations, and 2 more for DCC.',
 tm1: (n, v, b, d, k) => `N = ${n}: ${k} distinct elements in H<sub>t</sub>; VEC has ${v} parameters, BEKK ${b}, DCC ${d}.`,
 pc1: (n, v, b, db, d) => `N = ${n}: VEC(1,1) ${v} parameters, BEKK(1,1) ${b}, diagonal BEKK ${db}, DCC ${d}.`,
 pc2: r => `VEC has ${r} times as many parameters as DCC; the DCC optimiser works only on a and b.`,
 xcs1: (a, b, c) => `Step 1: 1 − a − b = 1 − ${a} − ${b} = ${c}.`,
 xcs2: (c, a, z1, z2, b, t1, t2, t3, q) => `Step 2: q<sub>12,t</sub> = ${c} · 0.5 + ${a} · (${z1}) · (${z2}) + ${b} · 0.5 = ${t1} + ${t2} + ${t3} = ${q}.`,
 xcs3: (c, a, z1s, b, q11, z2s, q22) => `Step 3: q<sub>11,t</sub> = ${c} + ${a} · ${z1s} + ${b} = ${q11}; q<sub>22,t</sub> = ${c} + ${a} · ${z2s} + ${b} = ${q22}.`,
 xcs4: (q12, q11, q22, r) => `Step 4: ρ<sub>12,t</sub> = ${q12}/√(${q11} · ${q22}) = ${r}.`,
 xc1: (r, q, h) => `Tomorrow’s correlation: ${r} (q<sub>12,t</sub> = ${q} is not a correlation before the rescaling)${h ? `; a deviation halves in ${h} days` : ''}.`,
 xhs1: (p, l, h) => `ln 0.5/ln ${p} = −0.69315/(${l}) = ${h} days.`,
 xhs2: (a, b) => `a + b = 0.98 gives ${a} days, a + b = 0.99 gives ${b} days.`,
 xhs3: (a, h, ap, hp) => `S&P 500 and DAX: a + b = ${a}, ${h} days; five-asset panel: a + b = ${ap}, ${hp} days.`,
 xh1: (p, h, s) => `a + b = ${p}: half-life ${h} days; after 20 days ${s} of the deviation is left.`,
 t1Head: ['Index', 'μ', 'ω', 'α', 'β', 'α + β', 'kurt. r<sub>t</sub>', 'kurt. z<sub>t</sub>'],
 t1Note: (n, a, b, rc, zc) => `Daily log returns in %, ${n} common trading days, ${a} – ${b}; GARCH(1,1) with a constant mean, Gaussian QML; kurt.: excess kurtosis. Correlation of the returns ${rc}; correlation of the residuals q̄<sub>12</sub> = ${zc}, the CCC estimate.`,
 t11: (n, a, b, s) => `${n}: the excess kurtosis falls from ${a} (returns) to ${b} (standardised residuals, standard deviation ${s}); the tails of z<sub>t</sub> are still fatter than Normal.`,
 s11: (a, da, b, db) => `Peak volatility: S&P 500 ${a}% on ${da}, DAX ${b}% on ${db}.`,
 s12: (lec, a1, b1, a2, b2, p1, p2) => `${lec ? 'The estimates of the lecture' : 'Extra run of the filter'}: S&P 500 α = ${a1}, β = ${b1}; DAX α = ${a2}, β = ${b2}; the persistence stays ${p1} and ${p2}.`,
 t2Head: ['Quantity', 'Estimate', 'SE'],
 t2hl: 'Half-life of a correlation shock (days)', t2lr: 'LR against CCC', t2crit: 'χ<sup>2</sup>(2) critical value, 5%',
 t2Note: 'SE: standard errors from the step-2 Hessian (too small, they ignore step 1). l<sub>C</sub>: the correlation part of the log-likelihood, computed again in the browser.',
 t21: (b, a, lr, lrE) => `With b = ${b}, the best a on the grid is ${a} and gives LR = ${lr}; at the estimate (a, b) LR = ${lrE}.`,
 dc1: (a, ab, mn, dmn, mx, dmx, m) => `a = ${a} (a + b = ${ab}): the DCC correlation ranges from ${mn} (${dmn}) to ${mx} (${dmx}), around a mean of ${m}.`,
 dc2: (s, w, sr, lr) => `Standard deviation: DCC ${s}, window of ${w} ${sr}; LR against CCC ${lr}.`,
 tcHead: k => ['Period', k === 'dcc' ? 'ρ̄<sup>DCC</sup>' : k === 'roll' ? 'ρ̄ over 250 days' : 'correlation of the returns', 'σ̄<sup>S&P</sup> (%)', 'Days'],
 tcNote: k => (k === 'sample' ? 'The sample correlation of the returns rises mechanically with the volatility (Forbes and Rigobon); DCC works with the standardised residuals z<sub>t</sub>.' : 'Mean correlation and mean annualised S&P 500 volatility in each period.'),
 tc1: (a, b, c, va, vb, vc) => `Mean DCC correlation: calm ${a}, 2008–2009 ${b}, 2020 ${c}; mean volatility: ${va}%, ${vb}%, ${vc}%.`,
 tc2: r => `Over the quarters, the correlation between the log volatility and the mean DCC correlation is ${r} (extra run).`,
 zm1: (v, dv, r0, r, dr) => `2008: volatility peaks at ${v}% on ${dv}; the correlation drifts up from ${r0} to ${r} on ${dr}.`,
 zm2: (r, dr, v, dv, bf) => `2020: the correlation peaks at ${r} on ${dr}, ${bf ? 'before' : 'after'} the volatility peak (${v}% on ${dv}).`,
 zm3: lec => (lec ? 'The estimates of the lecture.' : 'Extra run: step 1 and step 2 with parameters other than the estimates, the same data.'),
 tpHead: ['Asset', 'μ', 'ω', 'α', 'β', 'α + β'],
 tpNote: (n, d0, a, sa, b, sb, h, lr) => `Step 1: GARCH(1,1) on the daily returns of the ${n} common trading days since ${d0}. Step 2: â = ${a} (SE ${sa}), b̂ = ${b} (SE ${sb}); half-life ${h} days; LR against CCC ${lr}.`,
 tp1: (n, p, v, dv, lr) => `${n}: persistence ${p}; the annualised volatility peaks at ${v}% on ${dv}; long-run level √(252 ω/(1 − α − β)) = ${lr}%.`,
 pn1: (sd, bd, mn, mx, dmx) => `S&P 500 and DAX: mean ${sd}; BET and DAX: mean ${bd}, between ${mn} and ${mx} (${dmx}).`,
 pn2: (mn, mx, dmx, m, be) => `S&P 500 and Bitcoin: from ${mn} to ${mx} (${dmx}), mean ${m}; BET and EUR/RON: mean ${be}.`,
 pn3: (a, ab, h) => `a = ${a}, a + b = ${ab}: half-life ${h} days, the same for all 10 pairs.`,
 ht1: (a, b, n) => `Average pairwise correlation: ${a} in the calm period, ${b} in the 2020 crash (${n} days).`,
 ht2: (a, b, c, d, e, f) => `BET and DAX: ${a} → ${b}; S&P 500 and BET: ${c} → ${d}; DAX and Bitcoin: ${e} → ${f}.`,
 ht3: (a, b) => `BET and EUR/RON: ${a} → ${b}.`,
 xvs1: (s1, s2, r, a, b, h) => `H<sub>t</sub> = D<sub>t</sub>R<sub>t</sub>D<sub>t</sub>: h<sub>11</sub> = ${s1}<sup>2</sup> = ${a}, h<sub>22</sub> = ${s2}<sup>2</sup> = ${b}, h<sub>12</sub> = ${r} · ${s1} · ${s2} = ${h}.`,
 xvs2: (a, b, h, v2, v) => `σ<sub>p</sub><sup>2</sup> = 0.25 · ${a} + 0.25 · ${b} + 2 · 0.25 · ${h} = ${v2}, so σ<sub>p</sub> = ${v}%.`,
 xvs3: (z, v, x) => `VaR 1% = ${z} · ${v} = ${x}% (mean 0).`,
 xvs4: (V, k) => `For ${V}: VaR 1% of about ${k} thousand lei.`,
 xvs5: (a, b, inc) => `With the same volatilities: ρ = 0.3 gives ${a}%, ρ = 0.9 gives ${b}% (${inc} higher).`,
 xv1: (r, v, x, k) => `ρ<sub>t</sub> = ${r}: σ<sub>p</sub> = ${v}%, VaR 1% = ${x}%, i.e. ${k} thousand lei.`,
 vb1: (l, m, x, xs, e, n) => `VaR ${l}: ${m} has ${x} violations, the static matrix ${xs}; expected ${e} in ${n} days.`,
 vb2: (m, r, lr, p, rs, lrs, ps) => `Kupiec: ${m} rate ${r}, LR<sub>uc</sub> = ${lr}, p ${p}; static rate ${rs}, LR<sub>uc</sub> = ${lrs}, p ${ps}.`,
 vb3: (l, ms, m, mx, q) => `Mean VaR ${l}: static ${ms}% every day, dynamic ${m}% on average and up to ${mx}% in a crash${q ? `; FHS quantile q = ${q}` : ''}.`,
 tvHead: ['Model', 'Violations', 'Rate', 'LR<sub>uc</sub>', 'p', 'p indep.', 'Feb–Apr 2020'],
 tvNote: (l, e, n, ni, q) => `VaR ${l}; expected: about ${e} violations in ${n} days. Parameters estimated on 2000–2014 (${ni} days); FHS quantile q = ${q}.`,
 tv1: (l, n11, x, p11, c) => `Static VaR ${l}: ${x} violations, ${n11} of them on the day after a violation (π̂<sub>11</sub> = ${p11}); ${c} in February–April 2020.`,
 tv2: (f, d, q) => `Rate of DCC with FHS ${f} against ${d} with the Normal quantile; q = ${q}.`,
 xgs1: (r, s, f, c) => `Step 1: σ<sub>sf</sub> = ρ σ<sub>s</sub> σ<sub>f</sub> = ${r} · ${s} · ${f} = ${c}.`,
 xgs2: (c, f2, h) => `Step 2: h* = σ<sub>sf</sub>/σ<sub>f</sub><sup>2</sup> = ${c}/${f2} = ${h} units of f per unit of s.`,
 xgs3: (r, he, s, sh) => `Step 3: HE = ρ<sup>2</sup> = ${r}<sup>2</sup> = ${he}; the hedged volatility σ<sub>s</sub>√(1 − ρ<sup>2</sup>) = ${sh}% instead of ${s}%.`,
 xgs4: (s, h) => `If tomorrow ρ<sub>t</sub> = 0.5 and σ<sub>f,t</sub> = 2%: h = 0.5 · ${s}/2 = ${h}.`,
 xg1: (h, he, sh, s) => `h* = ${h}; the hedge removes ${he} of the variance: ${sh}% instead of ${s}%.`,
 hg1: (s, m, mn, mx, dmx) => `Static ratio ${s}; after 2015 the DCC ratio averages ${m}, from ${mn} to ${mx} (${dmx}).`,
 hg2: (s, w, r, d, a) => `Hedging effectiveness, 2015–2026: static ${s}, OLS over ${w} ${r}, DCC (a = ${a}) ${d}.`,
 thHead: ['Method', 'Mean ratio', 'Var(r<sub>s</sub> − h r<sub>f</sub>)', 'HE'],
 thNote: (n, v) => `Out of sample: ${n} days since 2015; Var(r<sub>BET</sub>) = ${v}. HE = 1 − Var(hedged)/Var(unhedged).`,
 th1: (s, r, d, w) => `HE: static ${s}, OLS over ${w} ${r}, DCC ${d}.`,
};
})();
""",
}

QA = {
 'ro': {
  'wx-div': ('▶ crește ρ de la 0,3 la 1, cu pasul 0,1. Crește σ<sub>p</sub> cu aceeași mărime la fiecare pas?',
             'Nu: +0,63 puncte de la ρ = 0,2 la 0,3 (de la 15,49% la 16,12%), dar doar +0,51 puncte de la 0,9 la 1 (de la 19,49% la 20%). σ<sub>p</sub><sup>2</sup> este liniar în ρ, iar σ<sub>p</sub> este rădăcina lui, deci curba este concavă. Cu ρ = −0,2 volatilitatea este 12,6%.'),
  'divers': ('▶ mută ponderea primului activ de la 60% la 90% (ambele active cu 20%). Ce se întîmplă cu σ<sub>p</sub> pentru ρ = 0,8?',
             'Crește: 19,02% la 60% și 19,64% la 90%, față de 18,97% la 50/50. Cu volatilități egale, portofoliul 50/50 are cel mai mic risc oricare ar fi ρ. Cu al doilea activ la 40% și ρ = −1, σ<sub>p</sub> = |0,5 · 20 − 0,5 · 40| = 10%: riscul dispare doar cu w = 2/3.'),
  'rolling': ('▶ lungește fereastra pînă la 500 de zile. Ce se întîmplă cu minimul corelației S&P 500 – DAX (0,21 la 250 de zile)?',
              'Crește la 0,30 (noiembrie 2025): într-o fereastră lungă zilele asincrone din aprilie 2025 cîntăresc mai puțin. Cu 60 de zile corelația coboară pînă la −0,27 (iulie 2025) și urcă pînă la 0,86 (ianuarie 2012), cu abaterea standard 0,147 față de 0,122 la 250 de zile.'),
  't-async': ('▶ adună randamentele zilnice pe k = 2, …, 10 zile. Ce se întîmplă cu corelația S&P 500 – BET?',
              'Crește: 0,32 zilnic, 0,40 pe 2 zile și 0,50 pe 10 zile, peste corelația săptămînală de 0,46: o știre care ajunge la București a doua zi intră în același randament pe k zile. Dintre cele zece perechi, doar S&P 500 și Bitcoin au o corelație săptămînală mai mică decît cea zilnică (0,25 → 0,19): randamentul săptămînal al lui Bitcoin include și weekendul, iar cele 610 săptămîni dau o estimare mai zgomotoasă decît cele 2 815 zile.'),
  'wx-pd': ('▶ coboară h<sub>12</sub> de la 2,75 la −3 (h<sub>11</sub> = 4, h<sub>22</sub> = 1). Pentru ce valori este H pozitiv definită?',
            'Doar pentru |h<sub>12</sub>| &lt; 2, adică |ρ| &lt; 1. La h<sub>12</sub> = ±2 determinantul și o valoare proprie sînt 0 (o combinație fără risc); pentru |h<sub>12</sub>| &gt; 2 există portofolii cu varianță negativă, de exemplu −0,416 pentru h<sub>12</sub> = ±2,5. Semnul covarianței nu contează, doar mărimea ei față de √(h<sub>11</sub>h<sub>22</sub>).'),
  'wx-bekk': ('Puneți ε<sub>2,t−1</sub> = +1,5 (semn opus lui ε<sub>1,t−1</sub> = −2). Cît este h<sub>12,t</sub>?',
              '0,2415 = 0,020 + 0,075 · (−3) + 0,4465: șocurile de semne opuse scad covarianța. Cu ε<sub>2,t−1</sub> = 0, h<sub>12,t</sub> = 0,4665, tot sub 0,5, deși nivelul de lungă durată este 0,625: acel nivel ține cont că, în medie, produsul șocurilor este egal cu h<sub>12</sub>, iar o zi fără șoc comun aduce doar constanta și persistența.'),
  't-dim': ('Pentru N = 30, BEKK(1,1) are 2 265 de parametri. Cîți are DCC?',
            '527: 90 de parametri GARCH estimați serie cu serie, 435 de corelații de selecție și doar 2 parametri, a și b, găsiți prin optimizare numerică. VEC(1,1) are 432 915.'),
  'params': ('▶ mărește cel mai mare N pînă la 100. De cîte ori crește numărul de parametri ai VEC(1,1) de la N = 50 la N = 100?',
             'De 15,7 ori (de la 3 252 525 la 51 010 050), ca N<sup>4</sup>: 2<sup>4</sup> = 16. DCC crește de 3,8 ori (de la 1 377 la 5 252), BEKK(1,1) de 4 ori (de la 6 275 la 25 050).'),
  'wx-dcc': ('▶ mută z<sub>2,t−1</sub> pînă la +3, cu z<sub>1,t−1</sub> = −2. Cît este ρ<sub>12,t</sub> pentru z<sub>2,t−1</sub> = +2,5?',
             '0,187: q<sub>12,t</sub> = 0,010 − 0,250 + 0,465 = 0,225, iar q<sub>11,t</sub> = 1,150 și q<sub>22,t</sub> = 1,2625 rămîn aceleași. Cu z<sub>2,t−1</sub> = 0 corelația este 0,454, tot sub 0,5: mișcarea unei singure piețe mărește o varianță de la numitor fără să adauge nimic la covarianță.'),
  'wx-hl': ('▶ crește a + b pînă la 0,995. Cum se schimbă timpul de înjumătățire de la a + b = 0,99 la 0,995?',
            'Se dublează, de la 69 la 138 de zile: ln(a + b) ≈ −(1 − a − b), deci timpul de înjumătățire este aproximativ 0,69/(1 − a − b). Cu a + b = 0,90 el este 6,6 zile, iar după 20 de zile rămîne 12% din abatere.'),
  't-step1': ('Care indice pierde o pondere mai mare din excesul de kurtosis după standardizare?',
              'S&P 500: 1 − 1,8/10,5 = 83%, față de DAX: 1 − 1,4/6,2 = 78%. Volatilitatea variabilă explică cea mai mare parte a cozilor groase, dar reziduurile z<sub>t</sub> păstrează un exces de kurtosis de 1,4–1,8; de aceea VaR-ul cu cuantila distribuției Normale va avea prea multe încălcări.'),
  'step1': ('▶ crește multiplicatorul lui α pînă la 2 (aceeași persistență). Ce se întîmplă cu maximul volatilității S&P 500 din 2020?',
            'Crește la 139% (17 martie 2020), de la 109%: un α mai mare reacționează mai mult la șocul fiecărei zile. Cu multiplicatorul 0,25 maximul scade la 64% și se mută pe 25 martie 2020. Sînt rulări suplimentare ale filtrului, nu estimări.'),
  't-dcc': ('▶ fixează b la 0,98 și apoi la 0,99. Cu b = 0,99, mai este respinsă corelația constantă?',
            'Da: cel mai bun a din grilă, 0,0074, dă LR = 121,0, mult peste 5,99. Cu b = 0,90 cel mai bun a este 0,0375 și LR = 102,1: verosimilitatea are o creastă pe care a și b se înlocuiesc parțial; maximul, 131,8, este la estimarea (0,0152; 0,9766).'),
  'dcc': ('▶ crește a pînă la 0,022 (b fixat). Ce se întîmplă cu abaterea standard a corelației DCC (0,083 la estimare)?',
          'Crește la 0,120, aproape cît corelația pe 250 de zile (0,122), iar corelația variază între 0,19 și 0,82. LR scade la 100,1: variația în plus este zgomot pe care verosimilitatea îl penalizează. Cu a = 0 corelația este constantă, 0,575, iar LR = 0.'),
  't-crisis': ('▶ crește a pînă la 0,022. Ce se întîmplă cu diferența dintre corelația DCC medie din 2020 și cea din perioada calmă (0,70 − 0,56 = 0,14)?',
               'Crește la 0,18 (0,73 față de 0,55); cu a = 0 toate trei mediile sînt 0,575. Corelația de selecție a randamentelor (alegeți-o în casetă) dă 0,74 și 0,52: crește mecanic cu volatilitatea (Forbes și Rigobon), iar DCC lucrează cu reziduurile standardizate.'),
  'zoom': ('▶ crește multiplicatorul lui α pînă la 2. Mai atinge corelația din 2020 maximul înaintea volatilității?',
           'Nu: cu α dublat maximul corelației, 0,71, este pe 12 iunie 2020, după maximul volatilității (139% pe 17 martie 2020). Volatilitățile mai mari din martie împart șocurile acelor zile, deci reziduurile standardizate care alimentează recurența DCC sînt mai mici. Cu multiplicatorul 0,25, volatilitatea din 2008 atinge maximul abia pe 2 decembrie 2008 (60%).'),
  't-panel': ('▶ arată pe rînd fiecare activ. Care are cel mai mare maxim al volatilității și cînd?',
              'Bitcoin: 281% pe 13 martie 2020. S&P 500 atinge 121% pe 17 martie 2020, DAX 84% pe 13 martie 2020, BET 98% pe 20 decembrie 2018 (ziua în care raportul de acoperire DCC atinge 1,00), EUR/RON 13% pe 21 mai 2025.'),
  'panel': ('▶ crește a pînă la 0,02. Ce se întîmplă cu intervalul corelației S&P 500 – Bitcoin (−0,05 … 0,42)?',
            'Se lărgește: de la −0,33 la 0,62, cu media tot 0,21. Cu a = 0 cele patru corelații devin constante, egale cu Q̄: 0,55, 0,32, 0,21 și −0,07; acesta este modelul CCC, respins de LR = 88,5.'),
  'heat': ('▶ crește a pînă la 0,02. Ce se întîmplă cu corelația medie din crahul din 2020 (0,19)?',
           'Crește la 0,23, iar media din perioada calmă scade la 0,10. Cu a = 0 cele două matrice sînt identice (media 0,135): cu corelații constante nu se vede nicio creștere în criză.'),
  'wx-var': ('▶ crește ρ<sub>t</sub> pînă la 1. Cît este VaR 1% pentru ρ<sub>t</sub> = 1?',
             '3,14%: σ<sub>p</sub> = 0,5 · 1,5 + 0,5 · 1,2 = 1,35%, cea mai mare valoare posibilă cu aceste volatilități. Cu ρ<sub>t</sub> = −0,5 VaR 1% este 1,60%, aproximativ 16,0 mii de lei pentru 1 milion de lei.'),
  'var': ('▶ trece la nivelurile 2,5% și 5%. La 5%, cît este rata încălcărilor VaR-ului static?',
          '2,83% (82 de încălcări față de 145 așteptate): prea puține, LR<sub>uc</sub> = 33,6, p &lt; 0,001. La 1% VaR-ul static trecea testul (0,90%, p = 0,577): este prea mare în anii calmi, ceea ce la 1% compensează încălcările din crize, iar la 5% nu. DCC cu FHS are la 5% rata 4,60% (p = 0,314).'),
  't-var': ('▶ trece la nivelurile 2,5% și 5%. La 5%, trece vreun model testul de independență Christoffersen?',
            'Nu: toate patru au p ≤ 0,0004; încălcările vin grupate în zilele agitate. La 2,5% doar DCC cu FHS trece testul Kupiec (79 de încălcări, rata 2,73%, p = 0,433).'),
  'wx-hedge': ('Puneți ρ = −0,8. Cît sînt h* și HE?',
               'h* = −0,96: cumpărați 0,96 unități din f pentru fiecare unitate din s; HE = ρ<sup>2</sup> = 0,64, ca pentru ρ = 0,8. Cu ρ = 1 volatilitatea poziției acoperite este 0 (h* = 1,2).'),
  'hedge': ('▶ lungește fereastra OLS pînă la 500 de zile. Elimină atunci OLS pe ferestre mobile mai mult risc decît DCC (19,1%)?',
            'Nu: între 17,0% (300 de zile) și 18,0% (500 de zile). Cu a = 0 în DCC eficiența scade la 15,5%, tot peste raportul static (13,6%): corelația este atunci constantă, dar raportul q̄ σ<sub>BET,t</sub>/σ<sub>DAX,t</sub> urmează încă volatilitățile GARCH.'),
  't-hedge': ('Există o fereastră (60–500 de zile) pentru care OLS pe ferestre mobile elimină mai mult risc decît DCC?',
              'Nu: cel mai bun rezultat este 18,6%, cu 80 de zile, sub 19,1% pentru DCC; cu 300 de zile se obține doar 17,0%. Alegerea ferestrei schimbă rezultatul cu peste un punct procentual, iar DCC nu cere această alegere.'),
 },
 'en': {
  'wx-div': ('▶ raises ρ from 0.3 to 1 in steps of 0.1. Does σ<sub>p</sub> rise by the same amount at every step?',
             'No: +0.63 points from ρ = 0.2 to 0.3 (from 15.49% to 16.12%), but only +0.51 points from 0.9 to 1 (from 19.49% to 20%). σ<sub>p</sub><sup>2</sup> is linear in ρ and σ<sub>p</sub> is its square root, so the curve is concave. With ρ = −0.2 the volatility is 12.6%.'),
  'divers': ('▶ moves the weight of the first asset from 60% to 90% (both assets at 20%). What happens to σ<sub>p</sub> for ρ = 0.8?',
             'It rises: 19.02% at 60% and 19.64% at 90%, against 18.97% at 50/50. With equal volatilities the 50/50 portfolio has the lowest risk whatever ρ. With the second asset at 40% and ρ = −1, σ<sub>p</sub> = |0.5 · 20 − 0.5 · 40| = 10%: the risk disappears only with w = 2/3.'),
  'rolling': ('▶ lengthens the window up to 500 days. What happens to the minimum of the S&P 500 – DAX correlation (0.21 at 250 days)?',
              'It rises to 0.30 (November 2025): in a long window the asynchronous days of April 2025 weigh less. With 60 days the correlation goes down to −0.27 (July 2025) and up to 0.86 (January 2012), with a standard deviation of 0.147 against 0.122 at 250 days.'),
  't-async': ('▶ sums the daily returns over k = 2, …, 10 days. What happens to the S&P 500 – BET correlation?',
              'It rises: 0.32 daily, 0.40 over 2 days and 0.50 over 10 days, above the weekly correlation of 0.46: news that reaches Bucharest a day later falls into the same k-day return. Of the ten pairs, only the S&P 500 and Bitcoin have a weekly correlation below the daily one (0.25 → 0.19): the weekly Bitcoin return also contains the weekend, and the 610 weeks give a noisier estimate than the 2,815 days.'),
  'wx-pd': ('▶ lowers h<sub>12</sub> from 2.75 to −3 (h<sub>11</sub> = 4, h<sub>22</sub> = 1). For which values is H positive definite?',
            'Only for |h<sub>12</sub>| &lt; 2, i.e. |ρ| &lt; 1. At h<sub>12</sub> = ±2 the determinant and one eigenvalue are 0 (a riskless combination); for |h<sub>12</sub>| &gt; 2 some portfolios have a negative variance, for example −0.416 for h<sub>12</sub> = ±2.5. The sign of the covariance does not matter, only its size against √(h<sub>11</sub>h<sub>22</sub>).'),
  'wx-bekk': ('Set ε<sub>2,t−1</sub> = +1.5 (the opposite sign of ε<sub>1,t−1</sub> = −2). What is h<sub>12,t</sub>?',
              '0.2415 = 0.020 + 0.075 · (−3) + 0.4465: shocks of opposite signs lower the covariance. With ε<sub>2,t−1</sub> = 0, h<sub>12,t</sub> = 0.4665, still below 0.5, although the long-run level is 0.625: that level accounts for the cross-product of the shocks being equal to h<sub>12</sub> on average, and a day without a joint shock brings only the constant and the persistence.'),
  't-dim': ('For N = 30, BEKK(1,1) has 2,265 parameters. How many does DCC have?',
            '527: 90 GARCH parameters estimated series by series, 435 sample correlations and only 2 parameters, a and b, found by numerical optimisation. VEC(1,1) has 432,915.'),
  'params': ('▶ raises the largest N up to 100. By what factor does the number of VEC(1,1) parameters grow from N = 50 to N = 100?',
             '15.7 times (from 3,252,525 to 51,010,050), like N<sup>4</sup>: 2<sup>4</sup> = 16. DCC grows 3.8 times (from 1,377 to 5,252), BEKK(1,1) 4 times (from 6,275 to 25,050).'),
  'wx-dcc': ('▶ moves z<sub>2,t−1</sub> up to +3, with z<sub>1,t−1</sub> = −2. What is ρ<sub>12,t</sub> for z<sub>2,t−1</sub> = +2.5?',
             '0.187: q<sub>12,t</sub> = 0.010 − 0.250 + 0.465 = 0.225, while q<sub>11,t</sub> = 1.150 and q<sub>22,t</sub> = 1.2625 stay the same. With z<sub>2,t−1</sub> = 0 the correlation is 0.454, still below 0.5: a move of one market alone raises a variance in the denominator without adding anything to the covariance.'),
  'wx-hl': ('▶ raises a + b up to 0.995. How does the half-life change from a + b = 0.99 to 0.995?',
            'It doubles, from 69 to 138 days: ln(a + b) ≈ −(1 − a − b), so the half-life is about 0.69/(1 − a − b). With a + b = 0.90 it is 6.6 days, and after 20 days 12% of the deviation is left.'),
  't-step1': ('Which index loses the larger share of its excess kurtosis after the standardisation?',
              'The S&P 500: 1 − 1.8/10.5 = 83%, against the DAX: 1 − 1.4/6.2 = 78%. Time-varying volatility explains most of the fat tails, but the residuals z<sub>t</sub> keep an excess kurtosis of 1.4–1.8; this is why a VaR with the Normal quantile will have too many violations.'),
  'step1': ('▶ raises the multiplier of α up to 2 (same persistence). What happens to the 2020 peak of the S&P 500 volatility?',
            'It rises to 139% (17 March 2020), from 109%: a larger α reacts more to each day’s shock. With the multiplier 0.25 the peak falls to 64% and moves to 25 March 2020. These are extra runs of the filter, not estimates.'),
  't-dcc': ('▶ holds b at 0.98 and then at 0.99. With b = 0.99, is constant correlation still rejected?',
            'Yes: the best a on the grid, 0.0074, gives LR = 121.0, far above 5.99. With b = 0.90 the best a is 0.0375 and LR = 102.1: the likelihood has a ridge along which a and b partly replace each other; the maximum, 131.8, is at the estimate (0.0152, 0.9766).'),
  'dcc': ('▶ raises a up to 0.022 (b fixed). What happens to the standard deviation of the DCC correlation (0.083 at the estimate)?',
          'It rises to 0.120, close to the 250-day rolling correlation (0.122), and the correlation ranges from 0.19 to 0.82. The LR falls to 100.1: the extra variation is noise that the likelihood penalises. With a = 0 the correlation is constant, 0.575, and LR = 0.'),
  't-crisis': ('▶ raises a up to 0.022. What happens to the gap between the mean DCC correlation of 2020 and that of the calm period (0.70 − 0.56 = 0.14)?',
               'It widens to 0.18 (0.73 against 0.55); with a = 0 all three means are 0.575. The sample correlation of the returns (choose it in the box) gives 0.74 and 0.52: it rises mechanically with the volatility (Forbes and Rigobon), while DCC works with the standardised residuals.'),
  'zoom': ('▶ raises the multiplier of α up to 2. Does the 2020 correlation still peak before the volatility?',
           'No: with α doubled the correlation peak, 0.71, is on 12 June 2020, after the volatility peak (139% on 17 March 2020). The larger volatilities of March divide the shocks of those days, so the standardised residuals that drive the DCC recursion are smaller. With the multiplier 0.25 the 2008 volatility peaks only on 2 December 2008 (60%).'),
  't-panel': ('▶ shows each asset in turn. Which one has the highest volatility peak, and when?',
              'Bitcoin: 281% on 13 March 2020. The S&P 500 reaches 121% on 17 March 2020, the DAX 84% on 13 March 2020, the BET 98% on 20 December 2018 (the day on which the DCC hedge ratio reaches 1.00), EUR/RON 13% on 21 May 2025.'),
  'panel': ('▶ raises a up to 0.02. What happens to the range of the S&P 500 – Bitcoin correlation (−0.05 … 0.42)?',
            'It widens: from −0.33 to 0.62, with the mean still 0.21. With a = 0 the four correlations become constant, equal to Q̄: 0.55, 0.32, 0.21 and −0.07; this is the CCC model, rejected by LR = 88.5.'),
  'heat': ('▶ raises a up to 0.02. What happens to the average correlation in the 2020 crash (0.19)?',
           'It rises to 0.23, while the calm-period average falls to 0.10. With a = 0 the two matrices are identical (average 0.135): with constant correlations no rise in the crisis can be seen.'),
  'wx-var': ('▶ raises ρ<sub>t</sub> up to 1. What is the VaR 1% for ρ<sub>t</sub> = 1?',
             '3.14%: σ<sub>p</sub> = 0.5 · 1.5 + 0.5 · 1.2 = 1.35%, the largest possible value with these volatilities. With ρ<sub>t</sub> = −0.5 the VaR 1% is 1.60%, about 16.0 thousand lei for 1 million lei.'),
  'var': ('▶ moves to the levels 2.5% and 5%. At 5%, what is the violation rate of the static VaR?',
          '2.83% (82 violations against 145 expected): too few, LR<sub>uc</sub> = 33.6, p &lt; 0.001. At 1% the static VaR passed the test (0.90%, p = 0.577): it is too high in calm years, which at 1% offsets the violations of the crises and at 5% does not. DCC with FHS has a rate of 4.60% at 5% (p = 0.314).'),
  't-var': ('▶ moves to the levels 2.5% and 5%. At 5%, does any model pass the Christoffersen independence test?',
            'No: all four have p ≤ 0.0004; the violations come in clusters on the turbulent days. At 2.5% only DCC with FHS passes the Kupiec test (79 violations, a rate of 2.73%, p = 0.433).'),
  'wx-hedge': ('Set ρ = −0.8. What are h* and HE?',
               'h* = −0.96: buy 0.96 units of f per unit of s; HE = ρ<sup>2</sup> = 0.64, as for ρ = 0.8. With ρ = 1 the volatility of the hedged position is 0 (h* = 1.2).'),
  'hedge': ('▶ lengthens the OLS window up to 500 days. Does rolling OLS then remove more risk than DCC (19.1%)?',
            'No: between 17.0% (300 days) and 18.0% (500 days). With a = 0 in the DCC the effectiveness falls to 15.5%, still above the static ratio (13.6%): the correlation is then constant, but the ratio q̄ σ<sub>BET,t</sub>/σ<sub>DAX,t</sub> still follows the GARCH volatilities.'),
  't-hedge': ('Is there a window (60–500 days) for which rolling OLS removes more risk than DCC?',
              'No: the best result is 18.6%, with 80 days, below 19.1% for DCC; 300 days give only 17.0%. The choice of the window moves the result by more than a percentage point, and DCC needs no such choice.'),
 },
}

# Ce credeți? / What do you think?: (index of the correct option, [(RO, EN), ...]); keys checked against the page (node)
PREDICT = {
 'wx-div': (0, [('Nu: pașii se micșorează, de la +0,6 puncte (0,2 → 0,3) la +0,5 puncte (0,9 → 1).', 'No: the steps shrink, from +0.6 points (0.2 → 0.3) to +0.5 points (0.9 → 1).'),
                ('Da: σₚ² este liniar în ρ, deci și σₚ.', 'Yes: σₚ² is linear in ρ, so σₚ is too.'),
                ('Nu: pașii cresc, corelația afectează mai mult riscul la valori mari.', 'No: the steps grow, the correlation hurts more at high values.')]),
 'divers': (0, [('Crește, de la 19,0% la aproximativ 19,6%.', 'It rises, from 19.0% to about 19.6%.'),
                ('Scade: concentrarea pe un activ evită corelația.', 'It falls: concentrating on one asset avoids the correlation.'),
                ('Rămîne 19,0%: cu volatilități egale ponderile nu contează.', 'It stays 19.0%: with equal volatilities the weights do not matter.')]),
 'rolling': (0, [('Crește la aproximativ 0,30.', 'It rises to about 0.30.'), ('Rămîne 0,21: lungimea ferestrei nu schimbă minimul.', 'It stays 0.21: the window length does not change the minimum.'),
                 ('Scade sub 0: o fereastră lungă păstrează mai mult zilele din aprilie 2025.', 'It falls below 0: a long window keeps the days of April 2025 longer.')]),
 't-async': (0, [('Crește: 0,40 pe 2 zile și 0,50 pe 10 zile.', 'It rises: 0.40 over 2 days and 0.50 over 10 days.'),
                 ('Rămîne 0,32: suma randamentelor nu schimbă corelația.', 'It stays 0.32: summing the returns does not change the correlation.'),
                 ('Scade spre 0: în sume șocurile se compensează.', 'It falls towards 0: in sums the shocks cancel out.')]),
 'wx-pd': (0, [('Doar pentru −2 < h₁₂ < 2.', 'Only for −2 < h₁₂ < 2.'), ('Pentru orice h₁₂ < 2,5: doar covarianțele pozitive prea mari sînt o problemă.', 'For every h₁₂ < 2.5: only too large positive covariances are a problem.'),
               ('Doar pentru 0 ≤ h₁₂ < 2: o covarianță negativă nu este permisă.', 'Only for 0 ≤ h₁₂ < 2: a negative covariance is not allowed.')]),
 'wx-bekk': (0, [('0,2415.', '0.2415.'), ('0,6915: contează doar mărimea șocurilor.', '0.6915: only the size of the shocks matters.'), ('0,4665: un șoc pozitiv nu schimbă covarianța.', '0.4665: a positive shock does not change the covariance.')]),
 't-dim': (0, [('527.', '527.'), ('Aproximativ 2 265, ca BEKK: ambele cresc ca N².', 'About 2,265, as BEKK: both grow like N².'), ('2: DCC are doar a și b.', '2: DCC has only a and b.')]),
 'params': (0, [('De aproximativ 16 ori.', 'About 16 times.'), ('De 2 ori: numărul de active se dublează.', '2 times: the number of assets doubles.'), ('De aproximativ 4 ori, ca DCC.', 'About 4 times, as DCC.')]),
 'wx-dcc': (0, [('Scade la aproximativ 0,19.', 'It falls to about 0.19.'), ('0,602, ca înainte: contează doar mărimea șocurilor.', '0.602, as before: only the size of the shocks matters.'), ('Rămîne 0,5.', 'It stays 0.5.')]),
 'wx-hl': (0, [('Se dublează: de la 69 la aproximativ 138 de zile.', 'It doubles: from 69 to about 138 days.'), ('Crește cu 0,5%, cît a + b.', 'It rises by 0.5%, as a + b.'), ('Se înjumătățește.', 'It halves.')]),
 't-step1': (0, [('S&P 500: 83% față de 78%.', 'S&P 500: 83% against 78%.'), ('DAX: kurtosis-ul lui z este mai mic (1,4).', 'DAX: its kurtosis of z is lower (1.4).'), ('Aceeași pondere pentru ambii indici.', 'The same share for both indices.')]),
 'step1': (0, [('Crește la aproximativ 139%.', 'It rises to about 139%.'), ('Rămîne 109%: persistența este aceeași.', 'It stays 109%: the persistence is the same.'), ('Scade: un α mai mare înseamnă un β mai mic.', 'It falls: a larger α means a smaller β.')]),
 't-dcc': (0, [('Da: LR ≈ 121.', 'Yes: LR ≈ 121.'), ('Nu: cu b departe de estimare, LR scade sub 5,99.', 'No: with b far from the estimate, LR falls below 5.99.'),
               ('Da, cu LR peste 131,8: o persistență mai mare se potrivește mai bine.', 'Yes, with LR above 131.8: a higher persistence fits better.')]),
 'dcc': (0, [('Crește la aproximativ 0,12, aproape cît corelația pe ferestre mobile.', 'It rises to about 0.12, close to the rolling correlation.'),
             ('Rămîne 0,083: a schimbă doar viteza, nu amplitudinea.', 'It stays 0.083: a changes only the speed, not the amplitude.'), ('Scade.', 'It falls.')]),
 't-crisis': (0, [('Crește la aproximativ 0,18.', 'It widens to about 0.18.'), ('Rămîne 0,14: a nu schimbă mediile.', 'It stays 0.14: a does not change the means.'), ('Scade la 0.', 'It shrinks to 0.')]),
 'zoom': (0, [('Nu: maximul corelației trece în iunie 2020, după maximul volatilității.', 'No: the correlation peak moves to June 2020, after the volatility peak.'),
              ('Da, tot pe 28 februarie: pasul 1 nu influențează corelația.', 'Yes, still on 28 February: step 1 does not affect the correlation.'), ('Da, și mai sus decît 0,73.', 'Yes, and higher than 0.73.')]),
 't-panel': (0, [('Bitcoin: 281% pe 13 martie 2020.', 'Bitcoin: 281% on 13 March 2020.'), ('BET: cea mai mică persistență dă cele mai mari salturi.', 'BET: the lowest persistence gives the largest spikes.'),
                 ('EUR/RON: cea mai mare persistență.', 'EUR/RON: the highest persistence.')]),
 'panel': (0, [('Se lărgește: de la −0,33 la 0,62.', 'It widens: from −0.33 to 0.62.'), ('Rămîne între −0,05 și 0,42: a schimbă doar media.', 'It stays between −0.05 and 0.42: a changes only the mean.'),
               ('Se îngustează spre media 0,21.', 'It narrows towards the mean 0.21.')]),
 'heat': (0, [('Crește la aproximativ 0,23, iar media din perioada calmă scade la 0,10.', 'It rises to about 0.23, and the calm-period average falls to 0.10.'),
              ('Rămîne 0,19: a schimbă doar viteza.', 'It stays 0.19: a changes only the speed.'), ('Scade la 0,12, cît în perioada calmă.', 'It falls to 0.12, as in the calm period.')]),
 'wx-var': (0, [('3,14%.', '3.14%.'), ('6,28% = 2,326 · (1,5 + 1,2).', '6.28% = 2.326 · (1.5 + 1.2).'), ('3,06%, ca la ρ = 0,9: VaR-ul nu mai crește.', '3.06%, as at ρ = 0.9: the VaR no longer rises.')]),
 'var': (0, [('2,8%: prea puține încălcări, respins de testul Kupiec.', '2.8%: too few violations, rejected by the Kupiec test.'),
             ('Aproximativ 5%: a trecut testul la 1%, deci trece la orice nivel.', 'About 5%: it passed the test at 1%, so it passes at every level.'), ('Aproximativ 10%.', 'About 10%.')]),
 't-var': (0, [('Nu: toate patru resping independența.', 'No: all four reject independence.'), ('Doar modelele dinamice trec.', 'Only the dynamic models pass.'),
               ('Toate trec: cu mai multe încălcări gruparea este mai greu de observat.', 'All pass: with more violations the clustering is harder to detect.')]),
 'wx-hedge': (0, [('h* = −0,96 (cumpărați f) și HE = 0,64.', 'h* = −0.96 (buy f) and HE = 0.64.'), ('h* = 0,96 și HE = 0,64: semnul nu contează.', 'h* = 0.96 and HE = 0.64: the sign does not matter.'),
                  ('h* = −0,96 și HE = −0,64: acoperirea crește riscul.', 'h* = −0.96 and HE = −0.64: the hedge adds risk.')]),
 'hedge': (0, [('Rămîne sub DCC: între 17,0% și 18,0%.', 'It stays below DCC: between 17.0% and 18.0%.'), ('Depășește DCC: o fereastră lungă este mai stabilă.', 'It beats DCC: a long window is more stable.'),
               ('Coboară la nivelul static, 13,6%.', 'It falls to the static level, 13.6%.')]),
 't-hedge': (0, [('Nu: cel mai bun este 18,6%, cu 80 de zile.', 'No: the best is 18.6%, with 80 days.'), ('Da: fereastra de 500 de zile.', 'Yes: the 500-day window.'), ('Da: orice fereastră peste 250 de zile.', 'Yes: any window above 250 days.')]),
}
VEIL_CHARTS = {'wx-div'}

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slides „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('Cîți parametri are un BEKK(1,1) diagonal pentru N = 3?', 'How many parameters does a diagonal BEKK(1,1) have for N = 3?'), key=0, opts=[
  (('12.', '12.'), ('N(N + 1)/2 + 2N = 6 + 6 = 12.', 'N(N + 1)/2 + 2N = 6 + 6 = 12.')),
  (('24.', '24.'), ('24 este BEKK(1,1) complet: N(N + 1)/2 + 2N² = 6 + 18.', '24 is the full BEKK(1,1): N(N + 1)/2 + 2N² = 6 + 18.')),
  (('8.', '8.'), ('8 este BEKK scalar: N(N + 1)/2 + 2.', '8 is the scalar BEKK: N(N + 1)/2 + 2.'))]),
 dict(q=('DCC cu a = 0,02, b = 0,97, q̄₁₂ = 0,4, q₁₁ = q₂₂ = 1, q₁₂ = 0,4 ieri și z = (1, −1). Cît este q₁₂,ₜ?',
         'DCC with a = 0.02, b = 0.97, q̄₁₂ = 0.4, q₁₁ = q₂₂ = 1, q₁₂ = 0.4 yesterday and z = (1, −1). What is q₁₂,ₜ?'), key=1, opts=[
  (('0,412.', '0.412.'), ('Termenul a · z₁z₂ are semnul produsului: (1)(−1) = −1, nu +1.', 'The term a · z₁z₂ has the sign of the product: (1)(−1) = −1, not +1.')),
  (('0,372.', '0.372.'), ('0,01 · 0,4 + 0,02 · (−1) + 0,97 · 0,4 = 0,372: mișcările în sens opus reduc corelația.', '0.01 · 0.4 + 0.02 · (−1) + 0.97 · 0.4 = 0.372: opposite moves lower the correlation.')),
  (('0,4.', '0.4.'), ('Recurența revine spre q̄ doar fără șocuri; aici șocul comun este −1.', 'The recursion returns to q̄ only without shocks; here the joint shock is −1.'))]),
 dict(q=('Cît este timpul de înjumătățire al șocurilor de corelație cînd a + b = 0,95?', 'What is the half-life of correlation shocks when a + b = 0.95?'), key=0, opts=[
  (('Aproximativ 13,5 zile.', 'About 13.5 days.'), ('ln 0,5/ln 0,95 = 13,5.', 'ln 0.5/ln 0.95 = 13.5.')),
  (('10 zile = 0,5/0,05.', '10 days = 0.5/0.05.'), ('Abaterea scade geometric, cu factorul 0,95 pe zi, nu liniar.', 'The deviation shrinks geometrically, by the factor 0.95 a day, not linearly.')),
  (('20 de zile = 1/(1 − 0,95).', '20 days = 1/(1 − 0.95).'), ('1/(1 − a − b) este o durată medie, nu timpul după care rămîne jumătate.', '1/(1 − a − b) is a mean duration, not the time after which half is left.'))]),
 dict(q=('σs = 2%, σf = 1%, ρ = 0,5. Cît sînt raportul de acoperire și eficiența acoperirii?', 'σs = 2%, σf = 1%, ρ = 0.5. What are the hedge ratio and the hedging effectiveness?'), key=2, opts=[
  (('h* = 0,5 și HE = 0,5.', 'h* = 0.5 and HE = 0.5.'), ('h* = ρ σs/σf, nu ρ; HE = ρ², nu ρ.', 'h* = ρ σs/σf, not ρ; HE = ρ², not ρ.')),
  (('h* = 0,25 și HE = 0,25.', 'h* = 0.25 and HE = 0.25.'), ('Raportul este σsf/σf² = 0,5 · 2 · 1/1 = 1.', 'The ratio is σsf/σf² = 0.5 · 2 · 1/1 = 1.')),
  (('h* = 1 și HE = 0,25.', 'h* = 1 and HE = 0.25.'), ('h* = 0,5 · 2/1 = 1; HE = ρ² = 0,25.', 'h* = 0.5 · 2/1 = 1; HE = ρ² = 0.25.'))]),
 dict(q=('Un VaR 1% are 26 de încălcări în 2 500 de zile, dintre care 10 într-o singură lună. Este un VaR bun?',
         'A VaR 1% has 26 violations in 2,500 days, 10 of them in one month. Is it a good VaR?'), key=1, opts=[
  (('Da: 26 este aproape de cele 25 așteptate.', 'Yes: 26 is close to the expected 25.'), ('Testul Kupiec privește doar frecvența; gruparea încălcărilor este o altă condiție.', 'The Kupiec test looks only at the frequency; the clustering of violations is another condition.')),
  (('Nu: frecvența este corectă, dar încălcările se grupează.', 'No: the frequency is right, but the violations cluster.'), ('Nu trece testul de independență și nu se adaptează crizelor, ca VaR-ul static din curs.', 'It fails the independence test and does not adapt to crises, like the static VaR of the lecture.')),
  (('Nu: are prea multe încălcări.', 'No: it has too many violations.'), ('26 față de 25 nu este o diferență semnificativă.', '26 against 25 is not a significant difference.'))]),
 dict(q=('Dovedește o corelație mai mare într-o criză existența contagiunii?', 'Does a higher correlation in a crisis prove contagion?'), key=0, opts=[
  (('Nu, prin ea însăși: volatilitatea mai mare crește mecanic corelația de selecție.', 'Not by itself: higher volatility raises the sample correlation mechanically.'),
   ('Forbes și Rigobon; comparați corelațiile reziduurilor standardizate, ca în DCC.', 'Forbes and Rigobon; compare the correlations of standardised residuals, as DCC does.')),
  (('Da: piețele se mișcă împreună exact cînd riscul este mare.', 'Yes: markets move together exactly when risk is high.'), ('O legătură neschimbată dă și ea o corelație de selecție mai mare cînd volatilitatea crește.', 'An unchanged link also gives a higher sample correlation when the volatility rises.')),
  (('Da, dacă testul LR respinge CCC.', 'Yes, if the LR test rejects CCC.'), ('LR arată că corelația variază, nu că variația vine din contagiune.', 'The LR shows that the correlation varies, not that the variation comes from contagion.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {})) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch14-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] = TEXT_JS[lang].replace('\n};\n})();\n', '\n};\n') + ('window.CH14_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                                                                        + 'window.CH14_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n})();\n')


_predict_js()


if __name__ == '__main__':
    import interactive_common
    page_spec = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA', 'TEXT_JS', 'PREDICT', 'TICKET')})
    page_spec.widgets = lambda lang: {a: v for a, v in widgets_all(lang).items() if a in {b for b, _, _ in BLOCKS}}
    interactive_common.build(page_spec)
    for lg in ('en', 'ro'):
        _post(lg)
