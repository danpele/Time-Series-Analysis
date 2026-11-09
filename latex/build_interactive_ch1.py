"""Build the interactive pages of Chapter 1 (stochastic processes and stationarity): interactive/ch1.html (EN) and
interactive/ch1_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its
interactive version on the right. Logic in interactive/ch1.js, inputs in interactive/ch1_data.js
(Quantlets/Ch_01/export_interactive.py), slide images in interactive/slides/.
The worked examples and result tables of the lecture (properties of the autocovariance, the MA(1) and AR(1) examples,
differencing, the sample ACF, the confidence bands, the Ljung-Box example and table, Box-Cox) get blocks of their own (EXTRA),
placed by slide; several chart blocks also carry live tables and step-by-step calculations. On the web lecture
(build_web_lecture.py) the EXTRA blocks travel inside a chart block of the same section and ch1.js moves each one to its frame.
"""
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 1

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_01'
TEX = {'ro': 'RO/Cursuri/capitol1_procese_stochastice_stationaritate', 'en': 'EN/Courses/chapter1_stochastic_processes_stationarity'}

# (anchor, chart file, section key); order of the slides
BLOCKS = [('four-series', 'tsa_ch1_four_series', 'ts'), ('ensemble', 'tsa_ch1_ensemble', 'ts'),
          ('counterexample', 'tsa_ch1_counterexample', 'st'), ('nonstationary', 'tsa_ch1_nonstationary', 'st'),
          ('white-noise', 'tsa_ch1_white_noise', 'wn'), ('random-walk', 'tsa_ch1_random_walk', 'wn'), ('spot-real', 'tsa_ch1_spot_real', 'wn'),
          ('sp500-diff', 'tsa_ch1_sp500_diff', 'lag'), ('wold', 'tsa_ch1_wold', 'lag'), ('ergodicity', 'tsa_ch1_ergodicity', 'lag'),
          ('bartlett', 'tsa_ch1_bartlett', 'acf'), ('acf-models', 'tsa_ch1_acf_models', 'acf'), ('acf-pacf', 'tsa_ch1_acf_pacf', 'acf'),
          ('bet-acf', 'tsa_ch1_bet_acf', 'acf'),
          ('lb-size', 'tsa_ch1_lb_size', 'pm'), ('lb-stats', 'tsa_ch1_lb_stats', 'pm'),
          ('gdp-transform', 'tsa_ch1_gdp_transform', 'tr'), ('boxcox', 'tsa_ch1_boxcox', 'tr'), ('overdiff', 'tsa_ch1_overdiff', 'tr'),
          ('textbook', 'tsa_ch1_textbook', 'cl')]

# blocks of the worked examples and result tables: (anchor, frame title RO, frame title EN, section key, host chart block on the web lecture)
EXTRA = [('acov-props', 'Proprietățile funcției de autocovarianță', 'Properties of the autocovariance function', 'st', 'counterexample'),
         ('wx-ma1', 'Exemplu rezolvat: o medie mobilă de ordinul 1', 'Worked example: a moving average of order 1', 'st', 'counterexample'),
         ('wx-ar1', 'Exemplu rezolvat: o autoregresie de ordinul 1', 'Worked example: an autoregression of order 1', 'st', 'counterexample'),
         ('wx-diff', 'Diferențierea', 'Differencing', 'lag', 'sp500-diff'),
         ('wx-sacf', 'Autocovarianța și autocorelația de selecție', 'Sample autocovariance and autocorrelation', 'acf', 'bartlett'),
         ('bands', 'Benzi de încredere pentru ACF de selecție', 'Confidence bands for the sample ACF', 'acf', 'bartlett'),
         ('wx-lb', 'Exemplu rezolvat: statistica Ljung–Box', 'Worked example: the Ljung–Box statistic', 'pm', 'lb-size'),
         ('lb-table', 'Interpretarea statisticilor Ljung–Box', 'Interpreting the Ljung–Box statistics', 'pm', 'lb-stats'),
         ('wx-boxcox', 'Transformarea Box–Cox', 'The Box–Cox transformation', 'tr', 'boxcox')]

PFX = {'four-series': 'fs', 'ensemble': 'en', 'counterexample': 'ce', 'nonstationary': 'ns', 'white-noise': 'wn', 'random-walk': 'rw',
       'spot-real': 'sr', 'sp500-diff': 'sp', 'wold': 'wd', 'ergodicity': 'eg', 'bartlett': 'bt', 'acf-models': 'am', 'acf-pacf': 'ap',
       'bet-acf': 'ba', 'lb-size': 'ls', 'lb-stats': 'lq', 'gdp-transform': 'gt', 'boxcox': 'bc', 'overdiff': 'od', 'textbook': 'tb',
       'acov-props': 'cp', 'wx-ma1': 'xm', 'wx-ar1': 'xa', 'wx-diff': 'xd', 'wx-sacf': 'xs', 'bands': 'xb', 'wx-lb': 'xl', 'lb-table': 'lt', 'wx-boxcox': 'xc'}

SECTIONS = {
 'ro': dict(ts='Serii de timp și procese stochastice', st='Staționaritate', wn='Zgomotul alb și mersul aleator',
            lag='Operatorul lag, diferențierea și descompunerea Wold', acf='ACF și PACF de selecție',
            pm='Teste portmanteau: Box–Pierce și Ljung–Box', tr='Transformări', cl='Două serii clasice'),
 'en': dict(ts='Time series and stochastic processes', st='Stationarity', wn='White noise and the random walk',
            lag='The lag operator, differencing and the Wold decomposition', acf='Sample ACF and PACF',
            pm='Portmanteau tests: Box–Pierce and Ljung–Box', tr='Transformations', cl='Two classic series'),
}

L = {
 'ro': dict(file='ch1_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 1 · Procese stochastice și staționaritate', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'casetele „Exemplu rezolvat” refac calculul cu cifrele capitolului.',
   src='Simulări: șocuri generate cu seed-urile codului capitolului; cînd parametrii diferă de cei din curs, seria se recalculează din aceleași șocuri sau se simulează în browser. '
       'PIB real și nominal: Eurostat, trimestrial, neajustat sezonier, în versiunea folosită în curs. IAPC: Eurostat, pînă în august 2026. '
       'EUR/RON: cursul de referință BNR. BET și S&P 500: datele de piață ale cursului (EODHD), pînă la 18 septembrie 2026. Nilul și petele solare: seturile de date din statsmodels.'),
 'en': dict(file='ch1.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 1 · Stochastic Processes and Stationarity', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; '
        'the “Worked example” boxes redo the calculation with the numbers of the chapter.',
   src='Simulations: shocks drawn with the seeds of the chapter code; when the parameters differ from those of the lecture, the series is recomputed from the same shocks or simulated in the browser. '
       'Real and nominal GDP: Eurostat, quarterly, not seasonally adjusted, in the version used in the lecture. HICP: Eurostat, up to August 2026. '
       'EUR/RON: BNR reference rate. BET and S&P 500: the course market data (EODHD), up to 18 September 2026. The Nile and the sunspots: the statsmodels data sets.'),
}


CSS = ('<style>.c1x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c1x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c1x .c1f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c1x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c1x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c1t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c1t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c1t th,.c1t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c1t thead th{border-bottom:1px solid #1A3A6E}'
       '.c1veil{filter:blur(5px);user-select:none}.c1t tbody tr:nth-child(even){background:#F5F7FB}.c1t tbody tr.hi{background:#FDF0E1}.c1t .r{text-align:right}.c1cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}.c1x .ctl{max-width:360px;margin:.15rem 0 .2rem}</style>')


def tbl(i):
    return f'<div class="tbl c1t" id="{i}"></div>'


def wx(i, title, formula, ctl=''):
    """Worked-example box; `ctl` holds the controls that only drive the example (they stay out of the chart's controls bar)."""
    return f'<div class="c1x"><h4>{title}</h4>{ctl}<div class="c1f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def _chart_widgets(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    return {
     'four-series': (slider('fs-y0', s('Fereastra începe în', 'Window starts in'), 1995, 2025, 1, 1995)
                     + select('fs-len', s('Lungimea ferestrei', 'Window length'), [('1', s('1 an', '1 year')), ('2', s('2 ani', '2 years')), ('5', s('5 ani', '5 years')),
                                                                                 ('10', s('10 ani', '10 years')), ('32', s('tot eșantionul', 'full sample'))], '32') + play('fs'), charts('fs-chart')),
     'ensemble': (slider('en-phi', s('φ al AR(1)', 'AR(1) φ'), 0, 0.99, 0.01, 0.7) + slider('en-t', s('Secțiunea la data t', 'Cross-section at t'), 1, 100, 1, 100) + play('en'), charts('en-chart')),
     'counterexample': (slider('ce-k', s('Grade de libertate k ale χ² la t impar', 'Degrees of freedom k of the χ² at odd t'), 1, 60, 1, 5) + play('ce'), charts('ce-chart')),
     'nonstationary': (slider('ns-b', s('Seria B: panta trendului', 'Series B: slope of the trend'), 0, 0.1, 0.005, 0.04)
                       + slider('ns-s', s('Seria C: scala finală (inițial 0,4)', 'Series C: final scale (initial 0.4)'), 0.4, 5, 0.1, 3)
                       + slider('ns-d', s('Seria D: saltul de nivel la t = 150', 'Series D: level shift at t = 150'), 0, 8, 0.5, 4)
                       + slider('ns-t', s('Fereastra mobilă de 50 de observații se termină la t', 'The rolling window of 50 observations ends at t'), 50, 300, 10, 300) + play('ns'), charts('ns-chart')),
     'white-noise': (slider('wn-a', s('α al GARCH (β = 0,85, ω = 1 − α − β)', 'GARCH α (β = 0.85, ω = 1 − α − β)'), 0, 0.14, 0.01, 0.1)
                     + slider('wn-n', s('Numărul de observații T (primele T din 1000)', 'Number of observations T (the first T of 1000)'), 100, 1000, 50, 1000) + play('wn'), charts('wn-chart')),
     'random-walk': (slider('rw-t', s('Data t', 'Date t'), 1, 250, 1, 250) + slider('rw-c', s('Deriva c', 'Drift c'), -0.3, 0.3, 0.02, 0) + play('rw'), charts('rw-chart')),
     'spot-real': (slider('sr-n', s('Numărul de zile', 'Number of days'), 100, 500, 50, 500)
                   + select('sr-v', s('Afișează', 'Show'), [('lev', s('nivelul', 'the level')), ('acf', s('ACF a variațiilor zilnice', 'ACF of the daily changes'))], 'lev') + play('sr'), charts('sr-chart')),
     'sp500-diff': (slider('sp-y0', s('Eșantionul începe în', 'Sample starts in'), 2000, 2024, 1, 2000)
                    + select('sp-d', s('Panoul de jos', 'Bottom panel'), [('1', s('prima diferență', 'first difference')), ('2', s('a doua diferență', 'second difference'))], '1') + play('sp'), charts('sp-chart')),
     'wold': (slider('wd-a', s('φ, primul panou', 'φ, first panel'), -1, 1, 0.05, 0.8) + slider('wd-b', s('φ, al doilea panou', 'φ, second panel'), -1, 1, 0.05, -0.6)
              + slider('wd-th', s('θ, al treilea panou', 'θ, third panel'), -2, 2, 0.05, 0.6) + play('wd'), charts('wd-chart')),
     'ergodicity': (slider('eg-phi', s('φ al AR(1), stînga', 'AR(1) φ, left'), 0, 0.99, 0.01, 0.7) + slider('eg-z', s('Abaterea standard a lui Z, dreapta', 'Standard deviation of Z, right'), 0, 2, 0.1, 1)
                    + slider('eg-T', s('Orizontul T al mediilor', 'Horizon T of the averages'), 10, 2000, 10, 2000)
                    + play('eg'), charts('eg-chart')),
     'bartlett': (slider('bt-m', s('Numărul de laguri m', 'Number of lags m'), 1, 20, 1, 20)
                  + select('bt-T', s('Lungimea eșantionului T', 'Sample length T'), [(t, t) for t in ('50', '100', '200', '500')], '100') + play('bt'), charts('bt-chart')),
     'acf-models': (slider('am-phi', 'φ (AR(1))', -0.95, 0.99, 0.01, 0.8) + slider('am-th', 'θ (MA(1))', -2, 2, 0.05, 0.6)
                    + slider('am-T', s('Lungimea eșantionului T', 'Sample length T'), 50, 500, 10, 500) + play('am'), charts('am-chart')),
     'acf-pacf': (slider('ap-a', 'φ (AR(1))', -0.95, 0.95, 0.05, 0.7) + slider('ap-b', s('φ₂ al AR(2) (φ₁ = 0,5)', 'AR(2) φ₂ (φ₁ = 0.5)'), -0.9, 0.49, 0.01, 0.3)
                  + slider('ap-c', 'θ (MA(1))', -0.95, 0.95, 0.05, 0.7) + play('ap'), charts('ap-chart')),
     'bet-acf': (select('ba-s', s('Seria', 'Series'), [('bet', 'BET'), ('sp', 'S&amp;P 500'), ('fx', 'EUR/RON')], 'bet')
                 + slider('ba-y0', s('Eșantionul începe în', 'Sample starts in'), 2000, 2024, 1, 2000) + play('ba'), charts('ba-chart')),
     'lb-size': (slider('ls-m', s('Numărul de laguri m (dreapta)', 'Number of lags m (right)'), 1, 20, 1, 20)
                 + select('ls-T', s('T în panoul din dreapta', 'T in the right panel'), [(t, t) for t in ('50', '100', '500')], '50')
                 + select('ls-a', s('Nivelul de semnificație', 'Significance level'), [('0.01', '1%'), ('0.05', '5%'), ('0.1', '10%')], '0.05') + play('ls'), charts('ls-chart')),
     'lb-stats': (slider('lq-m', s('Numărul de laguri m', 'Number of lags m'), 1, 30, 1, 10) + slider('lq-y0', s('Eșantionul începe în', 'Sample starts in'), 1995, 2025, 1, 1995)
                  + play('lq'), charts('lq-chart')),
     'gdp-transform': (slider('gt-k', s('Lagul diferenței din dreapta, k', 'Lag of the difference on the right, k'), 1, 8, 1, 4) + play('gt'), charts('gt-chart')),
     'boxcox': (slider('bc-l', 'λ', -1, 2, 0.01, 0.02) + play('bc'), charts('bc-chart')),
     'overdiff': (slider('od-phi', s('φ al seriei diferențiate (0: zgomot alb, 1: mers aleator)', 'φ of the differenced series (0: white noise, 1: random walk)'), 0, 1, 0.02, 0) + play('od'), charts('od-chart')),
     'textbook': (slider('tb-y', s('Nilul: ultimul an al primului regim', 'Nile: last year of the first regime'), 1875, 1965, 1, 1898) + play('tb'), charts('tb-chart')),
    }


def widgets_all(lang):
    """Widgets of every block of the interactive page: the chart blocks (with their live tables and worked examples) and EXTRA."""
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    W = s('Exemplu rezolvat', 'Worked example')
    d = (lambda x: x.replace('1.96', '1{,}96')) if ro else (lambda x: x)   # decimal comma in the RO formulas
    w = _chart_widgets(lang)
    rw = wx('rw-ws', W, r'X_t=\sum_{i=1}^{t}\varepsilon_i,\quad \mathrm{Var}(X_t)=t\sigma^2,\quad \mathrm{Cov}(X_t,X_s)=\min(t,s)\,\sigma^2,\quad \rho(X_t,X_s)=\sqrt{t/s}\ \ (t\le s)')
    add = {'four-series': (CSS, tbl('fs-tb')), 'nonstationary': ('', tbl('ns-tb')), 'white-noise': ('', tbl('wn-tb')), 'random-walk': ('', rw + tbl('rw-tb')),
           'bet-acf': ('', tbl('ba-tb')), 'gdp-transform': ('', tbl('gt-tb')), 'textbook': ('', tbl('tb-tb'))}
    for a, (pre, post) in add.items():
        ctl, ch = w[a]
        w[a] = (ctl, pre + ch + post)
    s10 = [('sp', 'S&amp;P 500, ' + s('randamente zilnice', 'daily returns')), ('sp2', 'S&amp;P 500, ' + s('randamente la pătrat', 'squared returns')),
           ('bet', 'BET, ' + s('randamente zilnice', 'daily returns')), ('bet2', 'BET, ' + s('randamente la pătrat', 'squared returns')),
           ('fx', 'EUR/RON, ' + s('randamente zilnice', 'daily returns')), ('gq', s('PIB România, creștere t/t', 'GDP Romania, q/q growth')),
           ('gy', s('PIB România, creștere an/an', 'GDP Romania, y/y growth')), ('hm', s('IAPC România, inflația lunară', 'HICP Romania, monthly inflation')),
           ('nile', s('Nil, debitul anual', 'Nile, yearly flow')), ('sun', s('Pete solare, anual', 'Sunspots, yearly'))]
    sig = s('Seria', 'Series')
    w.update({
     'acov-props': (slider('cp-r1', 'ρ(1)', -1, 1, 0.05, 0.9) + slider('cp-r2', 'ρ(2)', -1, 1, 0.05, -0.9) + slider('cp-a2', 'a₂', -1, 1, 0.5, -1) + slider('cp-a3', 'a₃', -1, 1, 0.5, 1) + play('cp'),
                    wx('cp-ws', W, r'\begin{gathered}\mathrm{Var}\Big(\sum_{i=1}^{3}a_iX_i\Big)=\sum_{i,j}a_ia_j\gamma(i-j)\\ =\gamma(0)\big[(a_1^2+a_2^2+a_3^2)+2(a_1a_2+a_2a_3)\rho(1)+2a_1a_3\rho(2)\big]\ge 0\end{gathered}') + charts('cp-chart')),
     'wx-ma1': (slider('xm-th', 'θ', -2, 2, 0.05, 0.6) + slider('xm-s2', 'σ²', 0.25, 4, 0.25, 1) + play('xm'),
                wx('xm-ws', W, r'X_t=\varepsilon_t+\theta\varepsilon_{t-1}:\quad \gamma(0)=\sigma^2(1+\theta^2),\quad \gamma(1)=\theta\sigma^2,\quad \rho(1)=\frac{\theta}{1+\theta^2},\quad \gamma(h)=0\ \ (|h|\ge 2)')
                + tbl('xm-tb') + charts('xm-chart')),
     'wx-ar1': (slider('xa-phi', 'φ', -0.95, 0.99, 0.01, 0.8) + slider('xa-s2', 'σ²', 0.25, 4, 0.25, 1) + slider('xa-h', s('Lagul h', 'Lag h'), 1, 20, 1, 5) + play('xa'),
                wx('xa-ws', W, r'\begin{gathered}X_t=\phi X_{t-1}+\varepsilon_t:\quad \gamma(0)=\frac{\sigma^2}{1-\phi^2},\quad \rho(h)=\phi^{|h|},\\ X_t=\sum_{j\ge 0}\phi^j\varepsilon_{t-j},\quad \psi_j=\phi^j,\quad \sum_{j\ge0}\psi_j^2=\frac{1}{1-\phi^2}\end{gathered}')
                + charts('xa-chart') + tbl('xa-tb')),
     'wx-diff': (select('xd-s', s('Datele', 'Data'), [('slide', s('x = (10, 12, 15, 14, 18), slide-ul', 'x = (10, 12, 15, 14, 18), the slide')), ('gdp', s('PIB-ul real al României', 'Romanian real GDP'))], 'slide')
                 + slider('xd-e', s('Trimestrul t (PIB)', 'Quarter t (GDP)'), 8, 125, 1, 125) + play('xd'),
                 wx('xd-ws', W, r'\begin{gathered}\Delta X_t=X_t-X_{t-1},\quad \Delta^2X_t=X_t-2X_{t-1}+X_{t-2},\\ \Delta_sX_t=X_t-X_{t-s},\quad 100\,\Delta\ln Y_t\approx 100\Big(\frac{Y_t}{Y_{t-1}}-1\Big)\end{gathered}')
                 + tbl('xd-tb') + charts('xd-chart')),
     'wx-sacf': (select('xs-s', s('Datele', 'Data'), [('slide', s('x = (3, 5, 4, 6, 7), slide-ul', 'x = (3, 5, 4, 6, 7), the slide')), ('nile', s('Nilul, primii T ani din 1871', 'The Nile, the first T years from 1871')),
                                                     ('gdp', s('Creșterea anuală a PIB-ului, ultimele T trimestre', 'Annual GDP growth, the last T quarters'))], 'slide')
                 + slider('xs-T', s('Lungimea eșantionului T', 'Sample length T'), 5, 12, 1, 8) + slider('xs-k', s('Lagul k', 'Lag k'), 1, 4, 1, 1) + play('xs'),
                 wx('xs-ws', W, d(r'\begin{gathered}\bar x=\frac1T\sum_{t=1}^{T}x_t,\quad \hat\gamma(k)=\frac1T\sum_{t=1}^{T-k}(x_t-\bar x)(x_{t+k}-\bar x),\\ \hat\rho(k)=\frac{\hat\gamma(k)}{\hat\gamma(0)},\quad \pm\frac{1.96}{\sqrt T}\end{gathered}'))
                 + tbl('xs-tb') + charts('xs-chart')),
     'bands': (select('xb-s', sig, s10, 'bet') + slider('xb-m', s('Numărul de laguri m', 'Number of lags m'), 5, 30, 1, 20) + slider('xb-h', s('Lagul h din calcul', 'Lag h in the calculation'), 1, 30, 1, 5) + play('xb'),
               wx('xb-ws', W, d(r'\pm\frac{1.96}{\sqrt T}\ \ (\text{i.i.d.}),\qquad \pm 1.96\sqrt{\frac{1}{T}\Big(1+2\sum_{k=1}^{h-1}\hat\rho(k)^2\Big)}\ \ (\text{Bartlett, MA}(h-1))'))
               + charts('xb-chart') + tbl('xb-tb')),
     'wx-lb': (select('xl-s', s('Datele', 'Data'), [('slide', s('T = 100, ρ̂ = (0,25; 0,12; −0,08), slide-ul', 'T = 100, ρ̂ = (0.25, 0.12, −0.08), the slide'))] + s10, 'slide')
               + slider('xl-m', s('Numărul de laguri m', 'Number of lags m'), 1, 3, 1, 3) + play('xl'),
               wx('xl-ws', W, r'Q(m)=T\sum_{h=1}^{m}\hat\rho(h)^2,\qquad Q^*(m)=T(T+2)\sum_{h=1}^{m}\frac{\hat\rho(h)^2}{T-h}\ \sim\ \chi^2(m)\ \text{' + s('sub', 'under') + r' }H_0')
               + tbl('xl-tb') + charts('xl-chart')),
     'lb-table': (slider('lt-m', s('Numărul de laguri m', 'Number of lags m'), 1, 30, 1, 10)
                  + select('lt-q', s('Statistica', 'Statistic'), [('lb', 'Ljung–Box Q*(m)'), ('bp', 'Box–Pierce Q(m)')], 'lb') + play('lt'),
                  tbl('lt-tb') + charts('lt-chart')),
     'wx-boxcox': (slider('xc-l', 'λ', -1, 2, 0.05, 0.5)
                   + select('xc-s', s('Seria pentru criteriul lui Guerrero', "Series for Guerrero's criterion"), [('nom', s('PIB nominal (slide-ul următor)', 'Nominal GDP (next slide)')), ('gdp', s('PIB real', 'Real GDP')), ('hicp', s('Indicele IAPC', 'HICP index'))], 'nom') + play('xc'),
                   wx('xc-ws', W, r'w=\frac{y^\lambda-1}{\lambda}\ (\lambda\ne0),\quad w=\ln y\ (\lambda=0),\quad y=(\lambda w+1)^{1/\lambda},\quad \mathrm{CV}\Big(\frac{\mathrm{sd}_i}{\mathrm{' + s('media', 'mean') + r'}_i^{\,1-\lambda}}\Big)\to\min_\lambda',
                      slider('xc-y', s('A doua valoare y₂ (y₁ = 100)', 'Second value y₂ (y₁ = 100)'), 200, 1600, 50, 400))
                   + tbl('xc-tb') + tbl('xc-tb2') + charts('xc-chart')),
    })
    return w


def _extra_pages(lang):
    """Page and title of the frames of EXTRA, from the .tex and the beamer .nav file."""
    from interactive_common import ROOT, frames_of, latex_text
    tex = os.path.join(ROOT, TEX[lang])
    frames = frames_of(open(tex + '.tex', encoding='utf-8').read())
    pages = re.findall(r'\\beamer@framepages \{(\d+)\}\{(\d+)\}', open(tex + '.nav', encoding='utf-8').read())
    titles = [latex_text(t) for t, _ in frames]
    res = {}
    for anchor, tro, ten, _, _ in EXTRA:
        k = titles.index(tro if lang == 'ro' else ten)
        res[anchor] = (int(pages[k][0]), titles[k])
    return res


def _extra_live(anchor, lang, W):
    t = L[lang]
    ctl, ch = W[anchor]
    q, a = QA[lang][anchor]
    live = f'<div class="live"><div class="ctls">{ctl}</div>{ch}<ul class="out" id="{PFX[anchor]}-out"></ul></div>'
    tail = f'<div class="ask"><span class="ask-tag">{t["ask"]}</span> {q}</div>\n  <details><summary>{t["answer"]}</summary><p>{a}</p></details>'
    return live, tail


def widgets(lang):
    """Chart blocks for build_web_lecture.py: each host carries the web version of the EXTRA blocks of its section (moved by ch1.js)."""
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


# readouts: JavaScript functions, one per block
TEXT_JS = {
 'ro': r"""window.CH1_TEXT = {
 fs0: (a, b, full) => full ? 'Tot eșantionul.' : `Fereastra ${a}–${b}.`,
 fs: (j, m, s, n) => { const nm = ['PIB-ul real', 'Inflația IAPC', 'EUR/RON', 'Randamentele BET'][j], u = [' mld. EUR', '%', ' lei', '%'][j];
   return `${nm}: media ${m}${u}, abaterea standard ${s}${u} (${n} observații).`; },
 fsNone: j => `${['PIB-ul real', 'Inflația IAPC', 'EUR/RON', 'Randamentele BET'][j]}: nicio observație în fereastră.`,
 en1: (p, v, b, t, va) => `AR(1), φ = ${p}: Var(X<sub>t</sub>) = 1/(1 − φ²) = ${v}, banda ±${b}; varianța celor 40 de traiectorii la t = ${t}: ${va}.`,
 en2: (t, b, v) => `Mersul aleator: Var(X<sub>t</sub>) = t = ${t}, banda ±${b}; varianța celor 40 de traiectorii la t = ${t}: ${v}.`,
 en3: v => `Pe 5000 de traiectorii simulate, Var(X<sub>100</sub>) = ${v}.`,
 ce1: (k, me, mo, ve, vo) => `La t impar, (χ²<sub>k</sub> − k)/√(2k) cu k = ${k}. Media: ${me} (t par) și ${mo} (t impar); varianța: ${ve} și ${vo}.`,
 ce2: (se, so, st) => `Asimetria: ${se} (t par) și ${so} (t impar); teoretic 0 și √(8/k) = ${st}.`,
 ce3: (pe, po, lb) => `P(X<sub>t</sub> < −1,6): ${pe} pentru t par, ${po} pentru t impar; la t impar, X<sub>t</sub> ≥ −√(k/2) = ${lb}.`,
 ns: (l, m1, m2, s1, s2) => `Seria ${l}: media ${m1} în prima jumătate și ${m2} în a doua; abaterea standard ${s1} și ${s2}.`,
 ns2: (a, b, m, sd) => `Fereastra mobilă t = ${a}–${b}: media ${m.join('; ')} și abaterea standard ${sd.join('; ')} (seriile A; B; C; D).`,
 wn0: (n, b) => `Primele T = ${n} observații; banda ACF ±1,96/√T = ±${b}.`,
 wn: (j, a, p1, q2, p2, k) => `${['Zgomotul gaussian i.i.d.', 'Zgomotul Laplace i.i.d.', `GARCH(1,1), α = ${a}`][j]}: Ljung–Box Q*(10) pentru X<sub>t</sub>: ${p1}; pentru X<sub>t</sub><sup>2</sup>: Q*(10) = ${q2}, ${p2}; kurtosis-ul ${k}.`,
 rw1: (t, v, vt, sh) => `t = ${t}: varianța pe 5000 de traiectorii ${v} (teoretic tσ² = ${vt}); ${sh} dintre traiectorii sînt în afara benzii ±1,96√t.`,
 rw2: (t, s, c, th) => `Corelația dintre X<sub>${t}</sub> și X<sub>${s}</sub> pe cele 5000 de traiectorii: ${c} (teoretic √(t/s) = ${th}).`,
 rw3: (c, t, ct) => `Cu deriva c = ${c}: E[X<sub>${t}</sub>] = ct = ${ct}; varianța rămîne tσ².`,
 sr: (l, r1, p) => `Seria ${l}: ρ̂(1) al variațiilor zilnice ${r1}; Ljung–Box Q*(10): ${p}.`,
 sr0: (l, a, b, n) => `BET este seria ${l}: ${a} – ${b}, ${n} zile de tranzacționare.`,
 sp1: (d, n, a1, a50) => `De la ${d}: ${n} zile; ACF a lui ln P<sub>t</sub>: ${a1} la lagul 1, ${a50} la lagul 50.`,
 sp2: (n, m, s, r1) => `Randamentele (${n}): media ${m}% pe zi, abaterea standard ${s}%, ρ̂(1) = ${r1}.`,
 sp3: (r1, s) => `A doua diferență: ρ̂(1) = ${r1}, abaterea standard ${s}.`,
 wd: (k, v, p5, p10, s2) => k === 'ma' ? `MA(1), θ = ${v}: ψ<sub>j</sub> = 0 pentru j ≥ 2; Σψ<sub>j</sub>² = 1 + θ² = ${s2}.`
   : `AR(1), φ = ${v}: ψ<sub>5</sub> = ${p5}, ψ<sub>10</sub> = ${p10}; ` + (s2 === null ? 'Σψ<sub>j</sub>² = ∞: ponderile nu se sting.' : `Var(X<sub>t</sub>) = σ²Σψ<sub>j</sub>² = ${s2}σ².`),
 wdRW: () => 'Mersul aleator: ψ<sub>j</sub> = 1 pentru orice j, Σψ<sub>j</sub>² = ∞.',
 eg1: (p, T, lo, hi, sd) => `AR(1), φ = ${p}: X̄<sub>${T}</sub> este între ${lo} și ${hi} pe cele 6 traiectorii; abaterea standard a lui X̄<sub>T</sub> ≈ 1/((1 − φ)√T) = ${sd}.`,
 eg2: (sz, T, lo, hi, zl, zh) => `X<sub>t</sub> = Z + ε<sub>t</sub>, sd(Z) = ${sz}: X̄<sub>${T}</sub> este între ${lo} și ${hi}; cei 6 Z extrași sînt între ${zl} și ${zh}.`,
 bt1: (T, ns, l, m, mt, sd, st, b) => `T = ${T}, ${ns} de eșantioane${l ? ' (cele din curs)' : ' (simulare în browser)'}: media lui ρ̂(1) ${m} (teoretic −1/T = ${mt}), abaterea standard ${sd} (1/√T = ${st}); banda ±${b}.`,
 bt2: (m, p, pt, mo, mt) => `${m} ${m === 1 ? 'lag' : 'laguri'}: cel puțin o bară în afara benzii în ${p} dintre eșantioane (cu teste independente, 1 − 0,95<sup>m</sup> = ${pt}); în medie ${mo} bare în afara benzii (0,05m = ${mt}).`,
 am: (j, v, r1, r2, r10, th) => `${['Zgomotul alb', `AR(1), φ = ${v}`, `MA(1), θ = ${v}`, 'Mersul aleator'][j]}: ρ̂(1) = ${r1}, ρ̂(2) = ${r2}, ρ̂(10) = ${r10}` + (th ? ` (teoretic ${th[0]}; ${th[1]}; ${th[2]}).` : '; ρ(h) nu este definit.'),
 amB: (T, b) => `T = ${T}: banda ±${b}.`,
 ap: (j, v, r1, r2, p1, p2, p3) => `${['AR(1), φ = ' + v, 'AR(2), φ = (0,5; ' + v + ')', 'MA(1), θ = ' + v][j]}: ρ̂(1) = ${r1}, ρ̂(2) = ${r2}; φ̂<sub>11</sub> = ${p1}, φ̂<sub>22</sub> = ${p2}, φ̂<sub>33</sub> = ${p3}.`,
 apT: (f, r1, r2) => `AR(2) teoretic, cu φ₂ = ${f}: ρ(1) = φ₁/(1 − φ₂) = ${r1}, ρ(2) = φ₁ρ(1) + φ₂ = ${r2}, φ<sub>22</sub> = φ₂.`,
 ba1: (nm, d, n, b) => `${nm}, de la ${d}: ${n} randamente zilnice; banda ±${b}.`,
 ba2: (r1, q, p) => `Randamentele: ρ̂(1) = ${r1}; Ljung–Box Q*(10) = ${q}, ${p}.`,
 ba3: (a1, a50, s1, qs) => `Valorile absolute: ρ̂(1) = ${a1}, ρ̂(50) = ${a50}; pătratele: ρ̂(1) = ${s1}, Q*(10) = ${qs}.`,
 ls: (T, m, a, bp, lb, ns) => `T = ${T}, m = ${m}, nivelul ${a}%: Box–Pierce respinge ipoteza adevărată în ${bp}% dintre cele ${ns} de eșantioane, Ljung–Box în ${lb}%.`,
 lq0: (y, m, c) => `De la ${y}, m = ${m}: valoarea critică χ²<sub>0,95</sub>(${m}) = ${c}.`,
 lq: (j, n, m, q, p) => `${['S&P 500, randamente zilnice', 'BET, randamente zilnice', 'EUR/RON, randamente zilnice', 'S&P 500, randamente la pătrat', 'BET, randamente la pătrat', 'PIB România, creștere t/t'][j]} (T = ${n}): Q*(${m}) = ${q}, ${p}.`,
 lqFew: (j, n) => `${['S&P 500, randamente zilnice', 'BET, randamente zilnice', 'EUR/RON, randamente zilnice', 'S&P 500, randamente la pătrat', 'BET, randamente la pătrat', 'PIB România, creștere t/t'][j]}: doar ${n} observații, prea puține pentru m pînă la 30.`,
 gt1: (a1, a8) => `ln Y<sub>t</sub>: ACF ${a1} la lagul 1 și ${a8} la lagul 8.`,
 gt2: (r4, r2, sd) => `100 Δ ln Y<sub>t</sub>: ρ̂(4) = ${r4}, ρ̂(2) = ${r2}; abaterea standard ${sd}%.`,
 gt3: (k, m, s, r1, r4, l) => `100 Δ<sub>${k}</sub> ln Y<sub>t</sub>: media ${m}%, abaterea standard ${s}%; ρ̂(1) = ${r1}, ρ̂(4) = ${r4}; ultimul trimestru ${l}%.`,
 gt4: c => `T2 2020: ${c}%.`,
 bc1: (l, cv, lh, cm, c0, c1) => `λ = ${l}: criteriul lui Guerrero ${cv}; minimul, ${cm}, la λ̂ = ${lh}; la λ = 0: ${c0}, la λ = 1: ${c1}.`,
 bc2: (a0, a1, r) => `Amplitudinea sezonieră a seriei transformate (maximul minus minimul din fiecare an): ${a0} în primii cinci ani, ${a1} în ultimii cinci (raportul ${r}).`,
 od1: (p, r1, v) => `Seria, φ = ${p}: ρ̂(1) = ${r1}, varianța ${v}.`,
 od2: (r1, th, v) => `Prima diferență: ρ̂(1) = ${r1} (teoretic −(1 − φ)/2 = ${th}), varianța ${v}.`,
 tb1: (b, m0, m1) => `Media debitului: ${m0} pînă în ${b}, ${m1} după.`,
 tb2: (r1, q, p) => `Debitul: ρ̂(1) = ${r1}; Ljung–Box Q*(10) = ${q}, ${p}.`,
 tb3: (r1, q, p) => `După eliminarea celor două medii: ρ̂(1) = ${r1}; Q*(10) = ${q}, ${p}.`,
 tb4: (mn, lm, pk, lp) => `Petele solare: ACF are minimul ${mn} la lagul ${lm} și vîrful ${pk} la lagul ${lp}.`,
 tb5: (b, m0, m1, r1, q1, pk, qk, lp) => `Petele solare fără mediile de pînă în ${b} (${m0}) și de după (${m1}): ρ̂(1) = ${q1} (înainte ${r1}), ρ̂(${lp}) = ${qk} (înainte ${pk}).`
};
""",
 'en': r"""window.CH1_TEXT = {
 fs0: (a, b, full) => full ? 'Full sample.' : `Window ${a}–${b}.`,
 fs: (j, m, s, n) => { const nm = ['Real GDP', 'HICP inflation', 'EUR/RON', 'BET returns'][j], u = [' bn EUR', '%', ' RON', '%'][j];
   return `${nm}: mean ${m}${u}, standard deviation ${s}${u} (${n} observations).`; },
 fsNone: j => `${['Real GDP', 'HICP inflation', 'EUR/RON', 'BET returns'][j]}: no observation in the window.`,
 en1: (p, v, b, t, va) => `AR(1), φ = ${p}: Var(X<sub>t</sub>) = 1/(1 − φ²) = ${v}, band ±${b}; variance of the 40 paths at t = ${t}: ${va}.`,
 en2: (t, b, v) => `Random walk: Var(X<sub>t</sub>) = t = ${t}, band ±${b}; variance of the 40 paths at t = ${t}: ${v}.`,
 en3: v => `Over 5000 simulated paths, Var(X<sub>100</sub>) = ${v}.`,
 ce1: (k, me, mo, ve, vo) => `At odd t, (χ²<sub>k</sub> − k)/√(2k) with k = ${k}. Mean: ${me} (even t) and ${mo} (odd t); variance: ${ve} and ${vo}.`,
 ce2: (se, so, st) => `Skewness: ${se} (even t) and ${so} (odd t); theoretical 0 and √(8/k) = ${st}.`,
 ce3: (pe, po, lb) => `P(X<sub>t</sub> < −1.6): ${pe} for even t, ${po} for odd t; at odd t, X<sub>t</sub> ≥ −√(k/2) = ${lb}.`,
 ns: (l, m1, m2, s1, s2) => `Series ${l}: mean ${m1} in the first half and ${m2} in the second; standard deviation ${s1} and ${s2}.`,
 ns2: (a, b, m, sd) => `Rolling window t = ${a}–${b}: mean ${m.join(', ')} and standard deviation ${sd.join(', ')} (series A, B, C, D).`,
 wn0: (n, b) => `The first T = ${n} observations; ACF band ±1.96/√T = ±${b}.`,
 wn: (j, a, p1, q2, p2, k) => `${['Gaussian i.i.d. noise', 'Laplace i.i.d. noise', `GARCH(1,1), α = ${a}`][j]}: Ljung–Box Q*(10) for X<sub>t</sub>: ${p1}; for X<sub>t</sub><sup>2</sup>: Q*(10) = ${q2}, ${p2}; kurtosis ${k}.`,
 rw1: (t, v, vt, sh) => `t = ${t}: variance over 5000 paths ${v} (theoretical tσ² = ${vt}); ${sh} of the paths lie outside the ±1.96√t band.`,
 rw2: (t, s, c, th) => `Correlation between X<sub>${t}</sub> and X<sub>${s}</sub> over the 5000 paths: ${c} (theoretical √(t/s) = ${th}).`,
 rw3: (c, t, ct) => `With drift c = ${c}: E[X<sub>${t}</sub>] = ct = ${ct}; the variance stays tσ².`,
 sr: (l, r1, p) => `Series ${l}: ρ̂(1) of the daily changes ${r1}; Ljung–Box Q*(10): ${p}.`,
 sr0: (l, a, b, n) => `The BET is series ${l}: ${a} – ${b}, ${n} trading days.`,
 sp1: (d, n, a1, a50) => `From ${d}: ${n} days; ACF of ln P<sub>t</sub>: ${a1} at lag 1, ${a50} at lag 50.`,
 sp2: (n, m, s, r1) => `Returns (${n}): mean ${m}% a day, standard deviation ${s}%, ρ̂(1) = ${r1}.`,
 sp3: (r1, s) => `Second difference: ρ̂(1) = ${r1}, standard deviation ${s}.`,
 wd: (k, v, p5, p10, s2) => k === 'ma' ? `MA(1), θ = ${v}: ψ<sub>j</sub> = 0 for j ≥ 2; Σψ<sub>j</sub>² = 1 + θ² = ${s2}.`
   : `AR(1), φ = ${v}: ψ<sub>5</sub> = ${p5}, ψ<sub>10</sub> = ${p10}; ` + (s2 === null ? 'Σψ<sub>j</sub>² = ∞: the weights do not die out.' : `Var(X<sub>t</sub>) = σ²Σψ<sub>j</sub>² = ${s2}σ².`),
 wdRW: () => 'Random walk: ψ<sub>j</sub> = 1 for every j, Σψ<sub>j</sub>² = ∞.',
 eg1: (p, T, lo, hi, sd) => `AR(1), φ = ${p}: X̄<sub>${T}</sub> lies between ${lo} and ${hi} on the 6 paths; standard deviation of X̄<sub>T</sub> ≈ 1/((1 − φ)√T) = ${sd}.`,
 eg2: (sz, T, lo, hi, zl, zh) => `X<sub>t</sub> = Z + ε<sub>t</sub>, sd(Z) = ${sz}: X̄<sub>${T}</sub> lies between ${lo} and ${hi}; the 6 drawn values of Z lie between ${zl} and ${zh}.`,
 bt1: (T, ns, l, m, mt, sd, st, b) => `T = ${T}, ${ns} samples${l ? ' (those of the lecture)' : ' (browser simulation)'}: mean of ρ̂(1) ${m} (theoretical −1/T = ${mt}), standard deviation ${sd} (1/√T = ${st}); band ±${b}.`,
 bt2: (m, p, pt, mo, mt) => `${m} ${m === 1 ? 'lag' : 'lags'}: at least one bar outside the band in ${p} of the samples (with independent tests, 1 − 0.95<sup>m</sup> = ${pt}); on average ${mo} bars outside the band (0.05m = ${mt}).`,
 am: (j, v, r1, r2, r10, th) => `${['White noise', `AR(1), φ = ${v}`, `MA(1), θ = ${v}`, 'Random walk'][j]}: ρ̂(1) = ${r1}, ρ̂(2) = ${r2}, ρ̂(10) = ${r10}` + (th ? ` (theoretical ${th[0]}, ${th[1]}, ${th[2]}).` : '; ρ(h) is not defined.'),
 amB: (T, b) => `T = ${T}: band ±${b}.`,
 ap: (j, v, r1, r2, p1, p2, p3) => `${['AR(1), φ = ' + v, 'AR(2), φ = (0.5, ' + v + ')', 'MA(1), θ = ' + v][j]}: ρ̂(1) = ${r1}, ρ̂(2) = ${r2}; φ̂<sub>11</sub> = ${p1}, φ̂<sub>22</sub> = ${p2}, φ̂<sub>33</sub> = ${p3}.`,
 apT: (f, r1, r2) => `Theoretical AR(2) with φ₂ = ${f}: ρ(1) = φ₁/(1 − φ₂) = ${r1}, ρ(2) = φ₁ρ(1) + φ₂ = ${r2}, φ<sub>22</sub> = φ₂.`,
 ba1: (nm, d, n, b) => `${nm}, from ${d}: ${n} daily returns; band ±${b}.`,
 ba2: (r1, q, p) => `Returns: ρ̂(1) = ${r1}; Ljung–Box Q*(10) = ${q}, ${p}.`,
 ba3: (a1, a50, s1, qs) => `Absolute values: ρ̂(1) = ${a1}, ρ̂(50) = ${a50}; squares: ρ̂(1) = ${s1}, Q*(10) = ${qs}.`,
 ls: (T, m, a, bp, lb, ns) => `T = ${T}, m = ${m}, level ${a}%: Box–Pierce rejects the true hypothesis in ${bp}% of the ${ns} samples, Ljung–Box in ${lb}%.`,
 lq0: (y, m, c) => `From ${y}, m = ${m}: critical value χ²<sub>0.95</sub>(${m}) = ${c}.`,
 lq: (j, n, m, q, p) => `${['S&P 500, daily returns', 'BET, daily returns', 'EUR/RON, daily returns', 'S&P 500, squared returns', 'BET, squared returns', 'Romanian GDP, q/q growth'][j]} (T = ${n}): Q*(${m}) = ${q}, ${p}.`,
 lqFew: (j, n) => `${['S&P 500, daily returns', 'BET, daily returns', 'EUR/RON, daily returns', 'S&P 500, squared returns', 'BET, squared returns', 'Romanian GDP, q/q growth'][j]}: only ${n} observations, too few for m up to 30.`,
 gt1: (a1, a8) => `ln Y<sub>t</sub>: ACF ${a1} at lag 1 and ${a8} at lag 8.`,
 gt2: (r4, r2, sd) => `100 Δ ln Y<sub>t</sub>: ρ̂(4) = ${r4}, ρ̂(2) = ${r2}; standard deviation ${sd}%.`,
 gt3: (k, m, s, r1, r4, l) => `100 Δ<sub>${k}</sub> ln Y<sub>t</sub>: mean ${m}%, standard deviation ${s}%; ρ̂(1) = ${r1}, ρ̂(4) = ${r4}; last quarter ${l}%.`,
 gt4: c => `2020Q2: ${c}%.`,
 bc1: (l, cv, lh, cm, c0, c1) => `λ = ${l}: Guerrero's criterion ${cv}; the minimum, ${cm}, at λ̂ = ${lh}; at λ = 0: ${c0}, at λ = 1: ${c1}.`,
 bc2: (a0, a1, r) => `Seasonal amplitude of the transformed series (maximum minus minimum within each year): ${a0} in the first five years, ${a1} in the last five (ratio ${r}).`,
 od1: (p, r1, v) => `Series, φ = ${p}: ρ̂(1) = ${r1}, variance ${v}.`,
 od2: (r1, th, v) => `First difference: ρ̂(1) = ${r1} (theoretical −(1 − φ)/2 = ${th}), variance ${v}.`,
 tb1: (b, m0, m1) => `Mean flow: ${m0} up to ${b}, ${m1} after.`,
 tb2: (r1, q, p) => `Flow: ρ̂(1) = ${r1}; Ljung–Box Q*(10) = ${q}, ${p}.`,
 tb3: (r1, q, p) => `After removing the two means: ρ̂(1) = ${r1}; Q*(10) = ${q}, ${p}.`,
 tb4: (mn, lm, pk, lp) => `Sunspots: the ACF has its minimum ${mn} at lag ${lm} and its peak ${pk} at lag ${lp}.`,
 tb5: (b, m0, m1, r1, q1, pk, qk, lp) => `Sunspots without the means up to ${b} (${m0}) and after (${m1}): ρ̂(1) = ${q1} (before ${r1}), ρ̂(${lp}) = ${qk} (before ${pk}).`
};
""",
}


S10EN = ("{sp: 'S&P 500, daily returns', sp2: 'S&P 500, squared returns', bet: 'BET, daily returns', bet2: 'BET, squared returns', fx: 'EUR/RON, daily returns', "
         "gq: 'GDP Romania, q/q growth', gy: 'GDP Romania, y/y growth', hm: 'HICP Romania, monthly inflation', nile: 'Nile, yearly flow', sun: 'Sunspots, yearly'}")

TEXT_ADD = {
 'ro': r"""Object.assign(window.CH1_TEXT, {
 q: (y, k) => `T${k} ${y}`,
 x: { s10: {sp: 'S&P 500, randamente zilnice', sp2: 'S&P 500, randamente la pătrat', bet: 'BET, randamente zilnice', bet2: 'BET, randamente la pătrat', fx: 'EUR/RON, randamente zilnice',
        gq: 'PIB România, creștere t/t', gy: 'PIB România, creștere an/an', hm: 'IAPC România, inflația lunară', nile: 'Nil, debitul anual', sun: 'Pete solare, anual'},
   s10en: """ + S10EN + r""",
   fsN: ['PIB real (mld. EUR)', 'Inflația IAPC (%)', 'EUR/RON (lei)', 'Randamentele BET (%)'],
   fsH: ['Seria', 'n', 'prima valoare', 'ultima valoare', 'media', 'abaterea standard', 'minimul (data)', 'maximul (data)'],
   nsH: ['Seria', 'media, prima jumătate', 'media, a doua jumătate', 'ab. std., prima jumătate', 'ab. std., a doua jumătate', 'raportul ab. std.', 'condiția încălcată'],
   nsB: b => `(ii): media μ<sub>t</sub> = ${b}t depinde de t`, nsC: r => `(iii) la h = 0: scala crește de ${r} ori`, nsD: d => `(ii): media crește cu ${d} la t = 150`, nsOK: 'niciuna',
   nsCap: 'Cele 300 de observații, împărțite în două jumătăți; ultima coloană vine din construcția seriei.',
   rej: 'respinge', norej: 'nu respinge',
   wnH: ['Seria', 'kurtosis', 'Q*(10), X<sub>t</sub>', 'p', 'la 5%', 'Q*(10), X<sub>t</sub><sup>2</sup>', 'p', 'la 5%'],
   wnCap: n => `Testul Ljung–Box cu m = 10 pentru serie și pentru pătratele ei, T = ${n}; ipoteza nulă: zgomot alb.`,
   rwH: ['t', 'tσ²', 'varianța pe traiectorii', '±1,96√t', 'în afara benzii', '√(t/(t+10))', 'corelația pe traiectorii'],
   rwCap: n => `Mersul aleator cu σ = 1: valorile teoretice și cele calculate pe ${n} de traiectorii simulate.`,
   baCap: (n, b) => `T = ${n} zile; bold: în afara benzii ±${b}.`,
   series: 'Seria', mean: 'media', sd: 'abaterea standard',
   gtCap: 'PIB-ul real al României, trimestrial, neajustat sezonier; Q*(8): testul Ljung–Box cu 8 laguri.',
   tbN: ['Nil, debitul', b => `Nil, fără cele două medii (ruptura după ${b})`, 'Pete solare'],
   share: 'din Σψ<sub>j</sub><sup>2</sup>', quarter: 'Trimestrul',
   xdCap: 'PIB-ul real în milioane EUR (prețurile din 2010); creșterile în %.', obs: 'observația',
   xsSrc: { slide: 'Exemplul din slide', nile: 'Nilul', gdp: 'Creșterea anuală a PIB-ului' },
   bartlett: 'banda Bartlett', outA: 'ρ̂ în afara ±1,96/√T', outB: 'ρ̂ în afara benzii Bartlett', outP: 'φ̂ în afara ±1,96/√T',
   xbCap: (nm, n) => `${nm}, T = ${n}. Banda Bartlett la lagul h presupune un MA(h − 1); bold: valori în afara benzii ±1,96/√T.`,
   slideEx: 'Exemplul din slide (T = 100)',
   blocks: 'blocuri', critAt: l => `criteriul la ${l}`, block: 'Blocul', meanI: 'media<sub>i</sub>', sdI: 'sd<sub>i</sub>',
   xcName: { nom: 'PIB nominal (trimestrial)', gdp: 'PIB real (trimestrial)', hicp: 'Indicele IAPC (lunar)' },
   xcCap: 'Criteriul lui Guerrero: coeficientul de variație al raportului sd<sub>i</sub>/media<sub>i</sub><sup>1−λ</sup> pe blocuri de un an (4 trimestre sau 12 luni); λ̂ îl minimizează pe grila −1, −0,99, …, 2.',
   xcCap2: (nm, nb) => `${nm}: primele și ultimele patru dintre cele ${nb} blocuri.`,
   xcNote: (m, s, cv) => `Pe toate blocurile: media raportului ${m}, abaterea standard ${s}, coeficientul de variație ${cv}.` },
 cps1: (a2, a3, r1, r2, b2, b3, q2) => `Cu a = (1; ${a2}; ${a3}), ρ(1) = ${r1} și ρ(2) = ${r2}: Var(X<sub>1</sub> + a<sub>2</sub>X<sub>2</sub> + a<sub>3</sub>X<sub>3</sub>) = γ(0)[(1 + ${b2}² + ${b3}²) + 2(${b2} + ${b2} × ${b3}) × ${r1} + 2 × ${b3} × ${q2}].`,
 cps2: (t, v) => `= γ(0)(${t}) = ${v} γ(0).`,
 cps3n: v => `O varianță nu poate fi negativă: ${v} γ(0) &lt; 0, deci perechea (ρ(1), ρ(2)) nu este ACF-ul niciunui proces staționar.`,
 cps3p: v => `${v} γ(0) ≥ 0: acest vector a nu arată o contradicție; perechea poate fi o ACF numai dacă varianța este nenegativă pentru orice a.`,
 cps4: (l, b, ok) => `Condiția pentru orice a: matricea de autocorelație 3 × 3 trebuie să fie pozitiv semidefinită; cea mai mică valoare proprie a ei este ${l}, iar ρ(2) ≥ 2ρ(1)² − 1 = ${b} ${ok ? 'este' : 'nu este'} îndeplinită.`,
 cp: (r1, r2, v, ok) => `ρ(1) = ${r1}, ρ(2) = ${r2}: varianța combinației este ${v} γ(0); perechea ${ok ? 'este admisibilă' : 'nu este o ACF'}.`,
 xms1: (s2, th, g0) => `γ(0) = σ²(1 + θ²) = ${s2} × (1 + ${th}²) = ${g0}.`,
 xms2: (th, s2, g1) => `γ(1) = θσ² = ${th} × ${s2} = ${g1}; γ(h) = 0 pentru |h| ≥ 2, pentru că X<sub>t</sub> și X<sub>t+h</sub> nu au niciun șoc comun.`,
 xms3: (g1, g0, r1) => `ρ(1) = γ(1)/γ(0) = ${g1}/${g0} = ${r1}; σ² se simplifică, deci ρ(1) = θ/(1 + θ²).`,
 xms4: a => `|ρ(1)| = ${a} ≤ 0,5 pentru orice θ; maximul, 0,5, se atinge la θ = 1.`,
 xm: (th, g0, r1) => `MA(1) cu θ = ${th}: γ(0) = ${g0}, ρ(1) = ${r1}, ρ(h) = 0 pentru h ≥ 2.`,
 xas1: (s2, ph, d, g0) => `Media: μ = φμ, deci μ = 0. Varianța: γ(0) = σ²/(1 − φ²) = ${s2}/(1 − ${ph}²) = ${s2}/${d} = ${g0}.`,
 xas2: (ph, h, r1, rh, r10, gh) => `ρ(h) = φ<sup>h</sup>: ρ(1) = ${r1}, ρ(${h}) = ${ph}<sup>${h}</sup> = ${rh}, ρ(10) = ${r10}; γ(${h}) = γ(0)ρ(${h}) = ${gh}.`,
 xas3: (ph, S, J, c, sh) => `Forma Wold: X<sub>t</sub> = Σ φ<sup>j</sup>ε<sub>t−j</sub>, deci ψ<sub>j</sub> = ${ph}<sup>j</sup> și Σψ<sub>j</sub><sup>2</sup> = 1/(1 − φ²) = ${S} = γ(0)/σ²; primele ${J + 1} ponderi (j = 0, …, ${J}) dau ${c}, adică ${sh}% din sumă.`,
 xas4: hl => `Un șoc își pierde jumătate din efect după ln 0,5/ln φ = ${hl} perioade.`,
 xas4n: () => 'Cu φ ≤ 0, ponderile ψ<sub>j</sub> își schimbă semnul de la o perioadă la alta (sau sînt nule pentru j ≥ 1).',
 xa: (ph, g0, h, rh) => `AR(1) cu φ = ${ph}: γ(0) = ${g0}, ρ(${h}) = ${rh}.`,
 xds1: s => `Δx<sub>t</sub> = x<sub>t</sub> − x<sub>t−1</sub>: ${s}.`,
 xds2: (s, c) => `Δ²x<sub>t</sub> = Δx<sub>t</sub> − Δx<sub>t−1</sub>: ${s}; direct, Δ²x<sub>3</sub> = x<sub>3</sub> − 2x<sub>2</sub> + x<sub>1</sub> = ${c}.`,
 xds3: (a, b, c) => `${a} observații, apoi ${b} diferențe de ordinul întîi și ${c} de ordinul al doilea: fiecare diferențiere pierde o observație.`,
 xd0: (a, b) => `Δx = (${a}); Δ²x = (${b}).`,
 xdg1: (q, y, y1, d) => `${q}: ΔY<sub>t</sub> = Y<sub>t</sub> − Y<sub>t−1</sub> = ${y} − ${y1} = ${d} mil. EUR.`,
 xdg2: (l, l1, g, s) => `100 Δ ln Y<sub>t</sub> = 100(${l} − ${l1}) = ${g}%; creșterea simplă 100(Y<sub>t</sub>/Y<sub>t−1</sub> − 1) = ${s}%.`,
 xdg3: (q4, y4, l4, g4) => `Diferența sezonieră, față de ${q4} (Y<sub>t−4</sub> = ${y4}, ln Y<sub>t−4</sub> = ${l4}): 100 Δ<sub>4</sub> ln Y<sub>t</sub> = ${g4}%.`,
 xdg4: (a, b, c) => `A doua diferență a logaritmului: 100 Δ² ln Y<sub>t</sub> = ${a} − ${b} = ${c} puncte procentuale.`,
 xd1: (q, g1, s, g4) => `${q}: creșterea trimestrială ${g1}% (logaritmică) și ${s}% (simplă), creșterea anuală ${g4}%.`,
 xss1: (sm, n, s, m, dv) => `x̄ = (${sm})/${n} = ${s}/${n} = ${m}; abaterile x<sub>t</sub> − x̄: (${dv}).`,
 xss2: (s, n, g0) => `γ̂(0) = Σ(x<sub>t</sub> − x̄)²/T = ${s}/${n} = ${g0}.`,
 xss3: (k, pr, s, n, gk) => `γ̂(${k}) = [${pr}]/T = ${s}/${n} = ${gk}.`,
 xss4: (k, gk, g0, rk, n, b, out) => `ρ̂(${k}) = γ̂(${k})/γ̂(0) = ${gk}/${g0} = ${rk}; cu T = ${n}, banda este ±1,96/√${n} = ±${b}: ρ̂(${k}) este ${out ? 'în afara benzii' : 'în bandă'}.`,
 xs: (src, n, k, gk, rk, b) => `${src}, T = ${n}: γ̂(${k}) = ${gk}, ρ̂(${k}) = ${rk}, banda ±${b}.`,
 xbs1: (n, sq, b) => `T = ${n}, √T = ${sq}: banda pentru zgomotul alb i.i.d. este ±1,96/√T = ±${b}.`,
 xbs2: (h, r, out) => `La lagul h = ${h}: ρ̂(${h}) = ${r}, ${out ? 'în afara benzii' : 'în bandă'}.`,
 xbs3: (h, h1, S, n, bb, out) => `Banda Bartlett, dacă seria ar fi un MA(${h1}): ±1,96 √((1 + 2Σ<sub>k=1</sub><sup>${h1}</sup> ρ̂(k)²)/T) = ±1,96 √((1 + 2 × ${S})/${n}) = ±${bb}; ρ̂(${h}) este ${out ? 'în afara ei' : 'în interiorul ei'}.`,
 xbs4: (m, no, e, nb, np) => `Din ${m} laguri, ${no} bare ale ACF sînt în afara benzii ±1,96/√T (fără nicio corelație ne-am aștepta la 0,05m = ${e}), ${nb} în afara benzii Bartlett; PACF: ${np}.`,
 xb: (nm, n, b, m, no, e, nb) => `${nm}, T = ${n}: banda ±${b}; ${no} din ${m} autocorelații în afara ei (așteptat sub zgomot alb: ${e}), ${nb} în afara benzii Bartlett.`,
 xls1: (n, m, r, b, k) => `T = ${n}, m = ${m}: ρ̂(1), ρ̂(2), … = ${r}; banda ±1,96/√T = ±${b}, cu ${k} ${k === 1 ? 'autocorelație' : 'autocorelații'} în afara ei.`,
 xls2: (m, n, sq, s, q) => `Box–Pierce: Q(${m}) = T Σρ̂(h)² = ${n} × (${sq}) = ${n} × ${s} = ${q}.`,
 xls3: (m, n, n2, fr, s, q) => `Ljung–Box: Q*(${m}) = T(T + 2) Σρ̂(h)²/(T − h) = ${n} × ${n2} × (${fr}) = ${n * n2} × ${s} = ${q}.`,
 xls4: (m, cr, q, rej, p, pb) => `Valoarea critică χ²<sub>0,95</sub>(${m}) = ${cr}: Q*(${m}) = ${q} ${rej ? '> ' + cr + ', respingem H<sub>0</sub>' : '≤ ' + cr + ', nu respingem H<sub>0</sub>'}; p-value-ul este ${p} (Box–Pierce: ${pb}).`,
 xl: (src, m, q, qs, p, rej) => `${src}, m = ${m}: Q(${m}) = ${q}, Q*(${m}) = ${qs}, ${p}: ${rej ? 'respingem' : 'nu respingem'} ipoteza de zgomot alb la 5%.`,
 lt: (st, m, cr, nr, nm, r) => `${st}, m = ${m}, valoarea critică ${cr}: ${nr} din 10 serii resping ipoteza de zgomot alb la 5%; cel mai mic raport statistică/valoare critică: ${nm}, ${r}.`,
 xcs1: (l, y1, y2, a, b, w1, w2) => `λ = ${l}: w = (y<sup>λ</sup> − 1)/λ; y = ${y1}: (${a} − 1)/${l} = ${w1}; y = ${y2}: (${b} − 1)/${l} = ${w2}.`,
 xcs1L: (y1, y2, w1, w2) => `λ = 0: w = ln y; ln ${y1} = ${w1}, ln ${y2} = ${w2}.`,
 xcs2: (dy, dw, ry, rw) => `Distanța dintre cele două valori: ${dy} pe scara lui y, ${dw} pe scara lui w; raportul y<sub>2</sub>/y<sub>1</sub> = ${ry}, raportul w<sub>2</sub>/w<sub>1</sub> = ${rw}.`,
 xcs3: (l, w, y) => `Înapoi: y = (λw + 1)<sup>1/λ</sup> = (${l} × ${w} + 1)<sup>1/${l}</sup> = ${y}.`,
 xcs3L: (w, y) => `Înapoi: y = e<sup>w</sup> = e<sup>${w}</sup> = ${y}.`,
 xcs4: (nm, nb, p, b0, s0, m0, e, q0, b1, q1, cv, lh, cm) => `${nm}, ${nb} blocuri de ${p}: în blocul ${b0}, sd<sub>i</sub>/media<sub>i</sub><sup>${e}</sup> = ${s0}/${m0}<sup>${e}</sup> = ${q0}; în ultimul bloc, ${b1}: ${q1}. Coeficientul de variație al acestor rapoarte este ${cv}; minimul, ${cm}, la λ̂ = ${lh}.`,
 xc: (l, y2, w1, w2, nm, cv, lh) => `λ = ${l}: w = ${w1} pentru y = 100 și ${w2} pentru y = ${y2}. ${nm}: criteriul lui Guerrero ${cv} (λ̂ = ${lh}).`,
 rws1: (t, v) => `Var(X<sub>${t}</sub>) = Σ<sub>i=1</sub><sup>${t}</sup> Var(ε<sub>i</sub>) = ${t} × 1 = ${t}; pe cele 5000 de traiectorii: ${v}.`,
 rws2: (t, s, th, c) => `Cov(X<sub>${t}</sub>, X<sub>${s}</sub>) = min(${t}, ${s})σ² = ${t}; ρ = ${t}/√(${t} × ${s}) = √(${t}/${s}) = ${th}; pe traiectorii: ${c}.`,
 rws2n: t => `Pentru t = ${t} nu există X<sub>t+10</sub> în cele 250 de perioade simulate.`,
 rws3: (t, b, sh) => `Banda de 95%: ±1,96√${t} = ±${b}; ${sh} dintre traiectorii sînt în afara ei (teoretic 5%).`,
 rws4: (c, t, ct) => `Cu deriva c = ${c}: E[X<sub>${t}</sub>] = ct = ${ct}, iar varianța rămîne tσ².`,
 rws4n: () => 'Fără derivă, E[X<sub>t</sub>] = 0 la orice t: doar varianța depinde de t.'
});
""",
 'en': r"""Object.assign(window.CH1_TEXT, {
 q: (y, k) => `${y}Q${k}`,
 x: { s10: """ + S10EN + r""",
   s10en: """ + S10EN + r""",
   fsN: ['Real GDP (bn EUR)', 'HICP inflation (%)', 'EUR/RON (RON)', 'BET returns (%)'],
   fsH: ['Series', 'n', 'first value', 'last value', 'mean', 'standard deviation', 'minimum (date)', 'maximum (date)'],
   nsH: ['Series', 'mean, first half', 'mean, second half', 'st. dev., first half', 'st. dev., second half', 'ratio of st. dev.', 'condition violated'],
   nsB: b => `(ii): the mean μ<sub>t</sub> = ${b}t depends on t`, nsC: r => `(iii) at h = 0: the scale grows ${r} times`, nsD: d => `(ii): the mean rises by ${d} at t = 150`, nsOK: 'none',
   nsCap: 'The 300 observations, split into two halves; the last column comes from the construction of the series.',
   rej: 'rejects', norej: 'does not reject',
   wnH: ['Series', 'kurtosis', 'Q*(10), X<sub>t</sub>', 'p', 'at 5%', 'Q*(10), X<sub>t</sub><sup>2</sup>', 'p', 'at 5%'],
   wnCap: n => `The Ljung–Box test with m = 10 for the series and for its squares, T = ${n}; null hypothesis: white noise.`,
   rwH: ['t', 'tσ²', 'variance across paths', '±1.96√t', 'outside the band', '√(t/(t+10))', 'correlation across paths'],
   rwCap: n => `The random walk with σ = 1: theoretical values and values computed over ${n} simulated paths.`,
   baCap: (n, b) => `T = ${n} days; bold: outside the band ±${b}.`,
   series: 'Series', mean: 'mean', sd: 'standard deviation',
   gtCap: 'Romanian real GDP, quarterly, not seasonally adjusted; Q*(8): the Ljung–Box test with 8 lags.',
   tbN: ['Nile, flow', b => `Nile, the two means removed (break after ${b})`, 'Sunspots'],
   share: 'of Σψ<sub>j</sub><sup>2</sup>', quarter: 'Quarter',
   xdCap: 'Real GDP in million EUR (2010 prices); growth rates in %.', obs: 'observation',
   xsSrc: { slide: 'The example of the slide', nile: 'The Nile', gdp: 'Annual GDP growth' },
   bartlett: 'Bartlett band', outA: 'ρ̂ outside ±1.96/√T', outB: 'ρ̂ outside the Bartlett band', outP: 'φ̂ outside ±1.96/√T',
   xbCap: (nm, n) => `${nm}, T = ${n}. The Bartlett band at lag h assumes an MA(h − 1); bold: values outside the band ±1.96/√T.`,
   slideEx: 'The example of the slide (T = 100)',
   blocks: 'blocks', critAt: l => `criterion at ${l}`, block: 'Block', meanI: 'mean<sub>i</sub>', sdI: 'sd<sub>i</sub>',
   xcName: { nom: 'Nominal GDP (quarterly)', gdp: 'Real GDP (quarterly)', hicp: 'HICP index (monthly)' },
   xcCap: "Guerrero's criterion: the coefficient of variation of the ratio sd<sub>i</sub>/mean<sub>i</sub><sup>1−λ</sup> over blocks of one year (4 quarters or 12 months); λ̂ minimises it on the grid −1, −0.99, …, 2.",
   xcCap2: (nm, nb) => `${nm}: the first and the last four of the ${nb} blocks.`,
   xcNote: (m, s, cv) => `Over all the blocks: mean of the ratio ${m}, standard deviation ${s}, coefficient of variation ${cv}.` },
 cps1: (a2, a3, r1, r2, b2, b3, q2) => `With a = (1, ${a2}, ${a3}), ρ(1) = ${r1} and ρ(2) = ${r2}: Var(X<sub>1</sub> + a<sub>2</sub>X<sub>2</sub> + a<sub>3</sub>X<sub>3</sub>) = γ(0)[(1 + ${b2}² + ${b3}²) + 2(${b2} + ${b2} × ${b3}) × ${r1} + 2 × ${b3} × ${q2}].`,
 cps2: (t, v) => `= γ(0)(${t}) = ${v} γ(0).`,
 cps3n: v => `A variance cannot be negative: ${v} γ(0) &lt; 0, so the pair (ρ(1), ρ(2)) is not the ACF of any stationary process.`,
 cps3p: v => `${v} γ(0) ≥ 0: this vector a shows no contradiction; the pair can be an ACF only if the variance is non-negative for every a.`,
 cps4: (l, b, ok) => `The condition for every a: the 3 × 3 autocorrelation matrix must be non-negative definite; its smallest eigenvalue is ${l}, and ρ(2) ≥ 2ρ(1)² − 1 = ${b} ${ok ? 'holds' : 'fails'}.`,
 cp: (r1, r2, v, ok) => `ρ(1) = ${r1}, ρ(2) = ${r2}: the variance of the combination is ${v} γ(0); the pair ${ok ? 'is admissible' : 'is not an ACF'}.`,
 xms1: (s2, th, g0) => `γ(0) = σ²(1 + θ²) = ${s2} × (1 + ${th}²) = ${g0}.`,
 xms2: (th, s2, g1) => `γ(1) = θσ² = ${th} × ${s2} = ${g1}; γ(h) = 0 for |h| ≥ 2, because X<sub>t</sub> and X<sub>t+h</sub> share no shock.`,
 xms3: (g1, g0, r1) => `ρ(1) = γ(1)/γ(0) = ${g1}/${g0} = ${r1}; σ² cancels, so ρ(1) = θ/(1 + θ²).`,
 xms4: a => `|ρ(1)| = ${a} ≤ 0.5 for every θ; the maximum, 0.5, is reached at θ = 1.`,
 xm: (th, g0, r1) => `MA(1) with θ = ${th}: γ(0) = ${g0}, ρ(1) = ${r1}, ρ(h) = 0 for h ≥ 2.`,
 xas1: (s2, ph, d, g0) => `Mean: μ = φμ, so μ = 0. Variance: γ(0) = σ²/(1 − φ²) = ${s2}/(1 − ${ph}²) = ${s2}/${d} = ${g0}.`,
 xas2: (ph, h, r1, rh, r10, gh) => `ρ(h) = φ<sup>h</sup>: ρ(1) = ${r1}, ρ(${h}) = ${ph}<sup>${h}</sup> = ${rh}, ρ(10) = ${r10}; γ(${h}) = γ(0)ρ(${h}) = ${gh}.`,
 xas3: (ph, S, J, c, sh) => `Wold form: X<sub>t</sub> = Σ φ<sup>j</sup>ε<sub>t−j</sub>, so ψ<sub>j</sub> = ${ph}<sup>j</sup> and Σψ<sub>j</sub><sup>2</sup> = 1/(1 − φ²) = ${S} = γ(0)/σ²; the first ${J + 1} weights (j = 0, …, ${J}) give ${c}, that is ${sh}% of the sum.`,
 xas4: hl => `A shock loses half of its effect after ln 0.5/ln φ = ${hl} periods.`,
 xas4n: () => 'With φ ≤ 0, the weights ψ<sub>j</sub> change sign from one period to the next (or are zero for j ≥ 1).',
 xa: (ph, g0, h, rh) => `AR(1) with φ = ${ph}: γ(0) = ${g0}, ρ(${h}) = ${rh}.`,
 xds1: s => `Δx<sub>t</sub> = x<sub>t</sub> − x<sub>t−1</sub>: ${s}.`,
 xds2: (s, c) => `Δ²x<sub>t</sub> = Δx<sub>t</sub> − Δx<sub>t−1</sub>: ${s}; directly, Δ²x<sub>3</sub> = x<sub>3</sub> − 2x<sub>2</sub> + x<sub>1</sub> = ${c}.`,
 xds3: (a, b, c) => `${a} observations, then ${b} first differences and ${c} second differences: each difference loses one observation.`,
 xd0: (a, b) => `Δx = (${a}); Δ²x = (${b}).`,
 xdg1: (q, y, y1, d) => `${q}: ΔY<sub>t</sub> = Y<sub>t</sub> − Y<sub>t−1</sub> = ${y} − ${y1} = ${d} million EUR.`,
 xdg2: (l, l1, g, s) => `100 Δ ln Y<sub>t</sub> = 100(${l} − ${l1}) = ${g}%; the simple growth rate 100(Y<sub>t</sub>/Y<sub>t−1</sub> − 1) = ${s}%.`,
 xdg3: (q4, y4, l4, g4) => `The seasonal difference, against ${q4} (Y<sub>t−4</sub> = ${y4}, ln Y<sub>t−4</sub> = ${l4}): 100 Δ<sub>4</sub> ln Y<sub>t</sub> = ${g4}%.`,
 xdg4: (a, b, c) => `The second difference of the log: 100 Δ² ln Y<sub>t</sub> = ${a} − ${b} = ${c} percentage points.`,
 xd1: (q, g1, s, g4) => `${q}: quarterly growth ${g1}% (log) and ${s}% (simple), annual growth ${g4}%.`,
 xss1: (sm, n, s, m, dv) => `x̄ = (${sm})/${n} = ${s}/${n} = ${m}; the deviations x<sub>t</sub> − x̄: (${dv}).`,
 xss2: (s, n, g0) => `γ̂(0) = Σ(x<sub>t</sub> − x̄)²/T = ${s}/${n} = ${g0}.`,
 xss3: (k, pr, s, n, gk) => `γ̂(${k}) = [${pr}]/T = ${s}/${n} = ${gk}.`,
 xss4: (k, gk, g0, rk, n, b, out) => `ρ̂(${k}) = γ̂(${k})/γ̂(0) = ${gk}/${g0} = ${rk}; with T = ${n} the band is ±1.96/√${n} = ±${b}: ρ̂(${k}) is ${out ? 'outside the band' : 'inside the band'}.`,
 xs: (src, n, k, gk, rk, b) => `${src}, T = ${n}: γ̂(${k}) = ${gk}, ρ̂(${k}) = ${rk}, band ±${b}.`,
 xbs1: (n, sq, b) => `T = ${n}, √T = ${sq}: the band for i.i.d. white noise is ±1.96/√T = ±${b}.`,
 xbs2: (h, r, out) => `At lag h = ${h}: ρ̂(${h}) = ${r}, ${out ? 'outside the band' : 'inside the band'}.`,
 xbs3: (h, h1, S, n, bb, out) => `The Bartlett band, if the series were an MA(${h1}): ±1.96 √((1 + 2Σ<sub>k=1</sub><sup>${h1}</sup> ρ̂(k)²)/T) = ±1.96 √((1 + 2 × ${S})/${n}) = ±${bb}; ρ̂(${h}) is ${out ? 'outside it' : 'inside it'}.`,
 xbs4: (m, no, e, nb, np) => `Of ${m} lags, ${no} ACF bars are outside the band ±1.96/√T (with no correlation at all we would expect 0.05m = ${e}), ${nb} outside the Bartlett band; PACF: ${np}.`,
 xb: (nm, n, b, m, no, e, nb) => `${nm}, T = ${n}: band ±${b}; ${no} of ${m} autocorrelations outside it (expected under white noise: ${e}), ${nb} outside the Bartlett band.`,
 xls1: (n, m, r, b, k) => `T = ${n}, m = ${m}: ρ̂(1), ρ̂(2), … = ${r}; band ±1.96/√T = ±${b}, with ${k} ${k === 1 ? 'autocorrelation' : 'autocorrelations'} outside it.`,
 xls2: (m, n, sq, s, q) => `Box–Pierce: Q(${m}) = T Σρ̂(h)² = ${n} × (${sq}) = ${n} × ${s} = ${q}.`,
 xls3: (m, n, n2, fr, s, q) => `Ljung–Box: Q*(${m}) = T(T + 2) Σρ̂(h)²/(T − h) = ${n} × ${n2} × (${fr}) = ${n * n2} × ${s} = ${q}.`,
 xls4: (m, cr, q, rej, p, pb) => `Critical value χ²<sub>0.95</sub>(${m}) = ${cr}: Q*(${m}) = ${q} ${rej ? '> ' + cr + ', we reject H<sub>0</sub>' : '≤ ' + cr + ', we do not reject H<sub>0</sub>'}; the p-value is ${p} (Box–Pierce: ${pb}).`,
 xl: (src, m, q, qs, p, rej) => `${src}, m = ${m}: Q(${m}) = ${q}, Q*(${m}) = ${qs}, ${p}: we ${rej ? 'reject' : 'do not reject'} white noise at 5%.`,
 lt: (st, m, cr, nr, nm, r) => `${st}, m = ${m}, critical value ${cr}: ${nr} of the 10 series reject white noise at 5%; the smallest ratio of statistic to critical value: ${nm}, ${r}.`,
 xcs1: (l, y1, y2, a, b, w1, w2) => `λ = ${l}: w = (y<sup>λ</sup> − 1)/λ; y = ${y1}: (${a} − 1)/${l} = ${w1}; y = ${y2}: (${b} − 1)/${l} = ${w2}.`,
 xcs1L: (y1, y2, w1, w2) => `λ = 0: w = ln y; ln ${y1} = ${w1}, ln ${y2} = ${w2}.`,
 xcs2: (dy, dw, ry, rw) => `Distance between the two values: ${dy} on the scale of y, ${dw} on the scale of w; the ratio y<sub>2</sub>/y<sub>1</sub> = ${ry}, the ratio w<sub>2</sub>/w<sub>1</sub> = ${rw}.`,
 xcs3: (l, w, y) => `Back: y = (λw + 1)<sup>1/λ</sup> = (${l} × ${w} + 1)<sup>1/${l}</sup> = ${y}.`,
 xcs3L: (w, y) => `Back: y = e<sup>w</sup> = e<sup>${w}</sup> = ${y}.`,
 xcs4: (nm, nb, p, b0, s0, m0, e, q0, b1, q1, cv, lh, cm) => `${nm}, ${nb} blocks of ${p}: in block ${b0}, sd<sub>i</sub>/mean<sub>i</sub><sup>${e}</sup> = ${s0}/${m0}<sup>${e}</sup> = ${q0}; in the last block, ${b1}: ${q1}. The coefficient of variation of these ratios is ${cv}; the minimum, ${cm}, at λ̂ = ${lh}.`,
 xc: (l, y2, w1, w2, nm, cv, lh) => `λ = ${l}: w = ${w1} for y = 100 and ${w2} for y = ${y2}. ${nm}: Guerrero's criterion ${cv} (λ̂ = ${lh}).`,
 rws1: (t, v) => `Var(X<sub>${t}</sub>) = Σ<sub>i=1</sub><sup>${t}</sup> Var(ε<sub>i</sub>) = ${t} × 1 = ${t}; over the 5000 paths: ${v}.`,
 rws2: (t, s, th, c) => `Cov(X<sub>${t}</sub>, X<sub>${s}</sub>) = min(${t}, ${s})σ² = ${t}; ρ = ${t}/√(${t} × ${s}) = √(${t}/${s}) = ${th}; over the paths: ${c}.`,
 rws2n: t => `For t = ${t} there is no X<sub>t+10</sub> within the 250 simulated periods.`,
 rws3: (t, b, sh) => `The 95% band: ±1.96√${t} = ±${b}; ${sh} of the paths lie outside it (5% in theory).`,
 rws4: (c, t, ct) => `With drift c = ${c}: E[X<sub>${t}</sub>] = ct = ${ct}, and the variance stays tσ².`,
 rws4n: () => 'Without drift, E[X<sub>t</sub>] = 0 at every t: only the variance depends on t.'
});
""",
}

for _lg in TEXT_JS:
    TEXT_JS[_lg] += TEXT_ADD[_lg]


QA = {
 'ro': {
  'four-series': ('Butonul ▶ mută o fereastră de 5 ani de la 1995–1999 la 2021–2025. Care dintre cele patru serii are media apropiată de aceeași valoare în toate ferestrele?',
                  'Randamentele BET: media este 0,18% în 2000–2004, −0,05% în 2008–2012, 0,03% în 2013–2017 și 0,07% în 2021–2025, toate apropiate de 0. Abaterea standard se schimbă însă: 1,48%, 2,04%, 0,78% și 0,88%. '
                  'Media PIB-ului real crește de la 21,3 mld. EUR în 1995–1999 la 48,1 mld. EUR în 2021–2025, iar media inflației scade de la 59,3% în 1995–1999 la 0,82% în 2013–2017: pentru aceste serii, media depinde de perioadă.'),
  'ensemble': ('Butonul ▶ mută data t a secțiunii de la 2 la 100 (φ = 0,7). Cum se schimbă împrăștierea celor 40 de valori de la data t, în stînga și în dreapta?',
               'În stînga rămîne aceeași: Var(X<sub>t</sub>) = 1/(1 − φ²) = 1,96 la orice t, iar varianța celor 40 de valori oscilează în jurul ei, fără tendință (1,17 la t = 10, 2,44 la t = 50, 2,04 la t = 100): procesul este staționar. În dreapta crește cu t: la mersul aleator Var(X<sub>t</sub>) = t, banda ±1,96√t trece de la ±6,20 la t = 10 la ±19,60 la t = 100, iar varianța celor 40 de valori urcă de la 14,94 la 116,93. Cu φ mai mare, banda din stînga se lărgește (±13,89 la φ = 0,99), dar rămîne aceeași la orice t.'),
  'counterexample': ('Butonul ▶ crește k, gradele de libertate ale χ² de la t impar, de la 1 la 60. Ce se schimbă: media și varianța sau forma distribuției?',
                     'Doar forma. Pentru orice k, (χ²<sub>k</sub> − k)/√(2k) are media 0 și varianța 1, deci procesul rămîne slab staționar. Asimetria teoretică √(8/k) scade de la 2,83 la k = 1 la 1,26 la k = 5 și la 0,37 la k = 60 (în eșantion: 2,80; 1,30; 0,35). '
                     'P(X<sub>t</sub> < −1,6) la t impar este 0 pînă la k = 5 și 4,2% la k = 60, față de 5,9% la t par. Pentru orice k finit, distribuțiile de la t par și impar diferă, deci procesul nu este strict staționar.'),
  'nonstationary': ('Care dintre condițiile (ii) și (iii) ale staționarității slabe nu este îndeplinită pentru seriile B, C și D?',
                    'Doar seria A este staționară. B are trendul determinist μ<sub>t</sub> = 0,04t: condiția (ii) nu este îndeplinită, iar media este 2,84 în prima jumătate și 9,05 în a doua. '
                    'C are abaterea standard de 7,5 ori mai mare la final decît la început (1,09 și 2,74 pe cele două jumătăți): condiția (iii) nu este îndeplinită la h = 0. D are un salt de nivel de +4 la t = 150, care mută media de la −0,07 la 4,19: condiția (ii). '
                    'Butonul ▶ trece o fereastră mobilă de 50 de observații peste serii: la A, media ferestrei rămîne între −0,13 și 0,39 (ferestrele care se termină la t = 49, 149, 199 și 299), la B urcă de la 0,95 la 11,10, '
                    'la C abaterea standard crește de la 0,67 la 3,70, iar la D media sare de la 0,19 (t = 100–149) la 3,99 (t = 150–199).'),
  'white-noise': ('Butonul ▶ crește numărul de observații T de la 100 la 1000 (α = 0,1). Ce se întîmplă cu statistica Ljung–Box Q*(10) a pătratelor pentru cele trei serii?',
                  'Pentru GARCH crește odată cu T: 17,8 (p = 0,059) la T = 100, 42,5 la T = 500 și 73,2 la T = 1000 (p < 0,001); de la T = 150 testul respinge la fiecare pas. Pentru zgomotele i.i.d. statistica nu crește cu T: rămîne între 2,0 și 21,4 pentru seria gaussiană și între 3,2 și 8,0 pentru seria Laplace. Seria gaussiană respinge totuși la T = 150 și 200 (p = 0,039 și 0,018): o respingere falsă, care apare în circa 5% dintre eșantioane. Pătratele GARCH sînt corelate, deși seria rămîne necorelată; cu T = 1000, Q*(10) este 9,3 la α = 0 și 838,9 la α = 0,14.'),
  'random-walk': ('Butonul ▶ mută data t de la 10 la 250. Depinde corelația dintre X<sub>t</sub> și X<sub>t+10</sub> doar de distanța de 10 perioade?',
                  'Nu. Pe cele 5000 de traiectorii, corelația este 0,708 între X<sub>10</sub> și X<sub>20</sub> și 0,954 între X<sub>100</sub> și X<sub>110</sub>, ca valorile teoretice √(t/s) = 0,707 și 0,953. Pentru un proces staționar, corelația ar depinde doar de lag. '
                  'Varianța crește și ea cu t: 10,1 la t = 10, 98,1 la t = 100 și 249,1 la t = 250 (teoretic tσ²).'),
  'spot-real': ('Un panou este logaritmul prețului BET în ultimele 500 de zile de tranzacționare, celelalte cinci sînt mersuri aleatoare cu aceeași derivă și aceeași abatere standard zilnică. Care panou este BET?',
                'Seria D: BET, 11 septembrie 2024 – 18 septembrie 2026. Cu ochiul liber nu poate fi deosebită de mersurile aleatoare. Variațiile zilnice o deosebesc: ρ̂(1) = 0,123, iar p-value-ul testului Ljung–Box Q*(10) este 0,018, singurul sub 5% dintre cele șase serii; '
                'la mersurile aleatoare simulate, p-value-urile sînt între 0,377 și 0,911.'),
  'sp500-diff': ('Panoul de jos arată randamentele, Δ ln P<sub>t</sub>. Ce ρ̂(1) are a doua diferență, Δ² ln P<sub>t</sub>?',
                 '−0,543. Randamentele sînt deja apropiate de un zgomot alb (ρ̂(1) = −0,099), iar diferența unui zgomot alb este un MA(1) cu ρ(1) = −0,5: o diferențiere în plus creează autocorelație negativă și crește abaterea standard de la 1,21% la 1,80%. '
                 'Nivelul ln P<sub>t</sub> are ACF 0,999 la lagul 1 și 0,973 la lagul 50; la lagul 1, ACF rămîne peste 0,99 pentru orice început al eșantionului între 2000 și 2024.'),
  'wold': ('Butonul ▶ crește φ din primul panou de la 0 la 1; al doilea panou primește −φ, iar al treilea θ = φ. Ce se întîmplă cu Σψ<sub>j</sub>²?',
           'Crește: Σψ<sub>j</sub>² = 1/(1 − φ²) este 1 la φ = 0, 2,78 la φ = 0,8 și 10,26 la φ = 0,95, cînd ψ<sub>10</sub> = 0,599. La φ = 1, ψ<sub>j</sub> = 1 pentru orice j, ca la mersul aleator (punctele din ultimul panou ajung pe bare), iar suma este infinită: procesul nu mai are formă Wold și nu mai este staționar. Al doilea panou, cu −φ, are aceeași sumă, pentru că ψ<sub>j</sub>² = φ<sup>2j</sup>; la MA(1), Σψ<sub>j</sub>² = 1 + θ² rămîne cel mult 2.'),
  'ergodicity': ('Butonul ▶ mărește orizontul T al mediilor de la 10 la 2000. Spre ce se apropie X̄<sub>T</sub> pe fiecare traiectorie, în stînga și în dreapta?',
                 'În stînga, spre μ = 0 pe toate traiectoriile: X̄<sub>T</sub> este între −1,95 și 0,77 la T = 10, între −0,33 și 0,63 la T = 100 și între −0,13 și 0,09 la T = 2000; AR(1) este ergodic pentru medie. În dreapta, fiecare medie se apropie de propriul Z: la T = 2000 mediile sînt între −0,95 și 1,16, cît valorile Z extrase (între −0,99 și 1,15), iar împrăștierea nu scade cu T. Cu φ mai aproape de 1, convergența din stînga este mai lentă: abaterea standard a lui X̄<sub>2000</sub> este aproximativ 1/((1 − φ)√T), adică 0,075 la φ = 0,7 și 2,24 la φ = 0,99.'),
  'bartlett': ('Butonul ▶ crește numărul de laguri m de la 1 la 20, pe aceleași 10000 de eșantioane de zgomot alb. Cît de des iese cel puțin o bară din bandă?',
               'Din ce în ce mai des: în 4,5% dintre eșantioane la m = 1, 20,3% la m = 5 și 50,3% la m = 20, deși nu există nicio corelație. Cu teste independente ar fi 1 − 0,95<sup>m</sup>: 5,0%, 22,6% și 64,2%; '
               'la laguri mari Var(ρ̂(h)) < 1/T, deci ies ceva mai puține bare. La m = 20 ies în medie 0,73 bare. '
               'În stînga, eșantioanele cu cel puțin o bară în afara benzii (roșu) au adesea ρ̂(1) aproape de 0: alarma vine de la celelalte laguri.'),
  'acf-models': ('Butonul ▶ crește T de la 50 la 500. Ce se întîmplă cu ACF de selecție a mersului aleator la lagul 10?',
                 'Crește: ρ̂(10) este −0,08 la T = 50, 0,37 la T = 100 și 0,55 la T = 500. Mersul aleator nu are o ACF teoretică, iar descreșterea ACF de selecție depinde de lungimea eșantionului. '
                 'Pentru AR(1) cu φ = 0,8, ρ̂(10) se apropie de valoarea teoretică 0,11: −0,14 la T = 50, 0,11 la T = 100 și 0,09 la T = 500, iar banda se îngustează de la ±0,277 la ±0,088.'),
  'acf-pacf': ('Butonul ▶ mută de la −0,4 la 0,45 ultimul coeficient al fiecărui proces: φ al AR(1), φ₂ al AR(2) (φ₁ = 0,5) și θ al MA(1). Ce valoare urmărește PACF de selecție la lagul 2 al AR(2)?',
               'Coeficientul φ₂: φ̂<sub>22</sub> = −0,40 pentru φ₂ = −0,4, −0,03 pentru φ₂ = 0 și 0,38 pentru φ₂ = 0,45 (teoretic, φ<sub>22</sub> = φ₂). φ̂<sub>33</sub> rămîne în bandă (0,05; 0,04; −0,01): PACF se anulează după lagul 2. ρ̂(2) nu urmărește φ₂: la φ₂ = 0 este 0,28, aproape de ρ(2) = φ₁² = 0,25. În același timp, φ̂<sub>11</sub> al AR(1) urmărește φ (−0,38 la φ = −0,4 și 0,48 la φ = 0,45), iar ρ̂(1) al MA(1) urmărește θ/(1 + θ²) (−0,30 și 0,37; teoretic −0,345 și 0,374).'),
  'bet-acf': ('Selectați S&P 500. Au randamentele aceeași autocorelație de ordinul 1 ca BET? Dar pătratele lor?',
              'Pentru randamente, nu: ρ̂(1) = −0,099 pentru S&P 500 și 0,109 pentru BET (0,170 pentru EUR/RON). Pentru pătrate, da: ρ̂(1) = 0,31 pentru S&P 500 și 0,33 pentru BET, cu Q*(10) = 6133 și 2440. '
              'Semnul autocorelației randamentelor diferă de la o piață la alta, iar volatility clustering apare pe toate trei. Dacă eșantionul BET începe în 2013, ρ̂(1) scade la 0,054 pentru randamente și la 0,18 pentru pătrate.'),
  'lb-size': ('Cu T = 50, butonul ▶ crește m de la 1 la 20. Cum se îndepărtează cele două teste de nivelul de 5%?',
              'În direcții opuse. Box–Pierce respinge ipoteza adevărată în 4,5% dintre eșantioane la m = 1, în 4,0% la m = 10 și în 2,4% la m = 20; Ljung–Box în 5,2%, 7,4% și 9,4%. '
              'Cu T = 500 și m = 20, ratele sînt 5,0% și 5,9%. De aceea păstrăm m mult sub T (de exemplu, m ≤ T/5).'),
  'lb-stats': ('Dacă eșantionul începe în 2025 (circa 425 de zile), mai respinge testul Ljung–Box cu m = 10 ipoteza de zgomot alb pentru randamentele BET?',
               'Nu: Q*(10) = 14,3, p = 0,161, sub valoarea critică 18,31. Pe tot eșantionul, cu 6670 de zile, Q*(10) = 108,0, p < 0,001. Autocorelația mică a randamentelor BET (ρ̂(1) = 0,11) are nevoie de multe observații ca să fie separată de zgomot; '
               'nerespingerea nu arată că seria este zgomot alb. Pătratele resping și pe eșantionul scurt: Q*(10) = 68,6 pentru BET și 103,0 pentru S&P 500.'),
  'gdp-transform': ('Butonul ▶ crește lagul k al diferenței din dreapta de la 1 la 8. Pentru ce k dispare vîrful sezonier al ACF la lagul 4?',
                    'Pentru k = 4: ρ̂(4) = 0,03. Pentru k = 1, 2, 3 și 5, ρ̂(4) rămîne între 0,94 și 0,96: o diferență care nu acoperă un an întreg păstrează sezonul. '
                    'k = 8 elimină și el sezonul, dar ρ̂(4) = 0,52, pentru că două creșteri pe doi ani aflate la un an distanță au un an în comun. Creșterea anuală, k = 4, are media 2,73% și ρ̂(1) = 0,63.'),
  'boxcox': ('Cu λ = 1 (fără transformare), de cîte ori este mai mare amplitudinea sezonieră din ultimii cinci ani decît din primii cinci?',
             'De 10,5 ori: 3059 de milioane EUR în primii cinci ani și 32155 în ultimii cinci, pentru că oscilația sezonieră este o proporție stabilă din nivel. Cu λ = 0, logaritmul, raportul este 1,07 (0,392 și 0,420), '
             'iar criteriul lui Guerrero, 0,243, este practic minimul atins la λ̂ = 0,02; la λ = 1 criteriul este 0,677.'),
  'overdiff': ('Butonul ▶ crește φ al seriei diferențiate de la 0 (zgomot alb) la 1 (mers aleator). Cum se schimbă ρ̂(1) al primei diferențe?',
               'Urcă de la −0,54 spre 0: −0,28 la φ = 0,5, −0,10 la φ = 0,9 și −0,06 la φ = 1, conform valorii teoretice −(1 − φ)/2. Diferențierea este potrivită numai pentru mersul aleator, cînd ΔX<sub>t</sub> = ε<sub>t</sub>; '
               'pentru o serie staționară, un ρ̂(1) negativ, apropiat de −0,5, semnalează supradiferențierea. Varianța primei diferențe scade de la 2,01 la 0,95.'),
  'textbook': ('Butonul ▶ mută anul rupturii de la 1875 la 1965; media de pînă la ruptură și media de după sînt scăzute din debitul Nilului și din petele solare. Ce se întîmplă cu ρ̂(1) al Nilului după eliminarea celor două medii?',
               'Este cel mai mic în jurul lui 1898: 0,13 pentru o ruptură după 1897 și 0,16 după 1898, cu Q*(10) = 13,0 (p = 0,225). Cu ruptura după 1880 sau după 1920, ρ̂(1) este 0,40 și 0,42, iar după 1950 este 0,50, cît are debitul însuși. O mare parte din autocorelația debitului vine din schimbarea mediei în jurul lui 1898. La petele solare, aceeași operație aproape nu schimbă ACF: cu ruptura după 1898, ρ̂(1) = 0,81 (0,82 înainte), iar vîrful de la lagul 10 este 0,64 (0,66 înainte), pentru că autocorelația lor vine din ciclul de circa 11 ani, nu dintr-o ruptură.'),
 },
 'en': {
  'four-series': ('The ▶ button moves a 5-year window from 1995–1999 to 2021–2025. Which of the four series has a mean close to the same value in every window?',
                  'The BET returns: the mean is 0.18% in 2000–2004, −0.05% in 2008–2012, 0.03% in 2013–2017 and 0.07% in 2021–2025, all close to 0. The standard deviation changes, however: 1.48%, 2.04%, 0.78% and 0.88%. '
                  'The mean of real GDP rises from 21.3 bn EUR in 1995–1999 to 48.1 bn EUR in 2021–2025, and the mean of inflation falls from 59.3% in 1995–1999 to 0.82% in 2013–2017: for these series the mean depends on the period.'),
  'ensemble': ('The ▶ button moves the date t of the cross-section from 2 to 100 (φ = 0.7). How does the spread of the 40 values at date t change, on the left and on the right?',
               'On the left it stays the same: Var(X<sub>t</sub>) = 1/(1 − φ²) = 1.96 at every t, and the variance of the 40 values fluctuates around it with no trend (1.17 at t = 10, 2.44 at t = 50, 2.04 at t = 100): the process is stationary. On the right it grows with t: for the random walk Var(X<sub>t</sub>) = t, the band ±1.96√t goes from ±6.20 at t = 10 to ±19.60 at t = 100, and the variance of the 40 values rises from 14.94 to 116.93. With a larger φ the band on the left widens (±13.89 at φ = 0.99), but it stays the same at every t.'),
  'counterexample': ('The ▶ button raises k, the degrees of freedom of the χ² at odd t, from 1 to 60. What changes: the mean and the variance, or the shape of the distribution?',
                     'Only the shape. For every k, (χ²<sub>k</sub> − k)/√(2k) has mean 0 and variance 1, so the process stays weakly stationary. The theoretical skewness √(8/k) falls from 2.83 at k = 1 to 1.26 at k = 5 and 0.37 at k = 60 (in the sample: 2.80, 1.30, 0.35). '
                     'P(X<sub>t</sub> < −1.6) at odd t is 0 up to k = 5 and 4.2% at k = 60, against 5.9% at even t. For every finite k, the distributions at even and odd t differ, so the process is not strictly stationary.'),
  'nonstationary': ('Which of conditions (ii) and (iii) of weak stationarity fails for series B, C and D?',
                    'Only series A is stationary. B has the deterministic trend μ<sub>t</sub> = 0.04t: condition (ii) fails, and the mean is 2.84 in the first half and 9.05 in the second. '
                    'C has a standard deviation 7.5 times larger at the end than at the start (1.09 and 2.74 over the two halves): condition (iii) fails at h = 0. D has a level shift of +4 at t = 150, which moves the mean from −0.07 to 4.19: condition (ii). '
                    'The ▶ button runs a rolling window of 50 observations over the series: for A the window mean stays between −0.13 and 0.39 (windows ending at t = 49, 149, 199 and 299), for B it rises from 0.95 to 11.10, '
                    'for C the standard deviation grows from 0.67 to 3.70, and for D the mean jumps from 0.19 (t = 100–149) to 3.99 (t = 150–199).'),
  'white-noise': ('The ▶ button raises the number of observations T from 100 to 1000 (α = 0.1). What happens to the Ljung–Box statistic Q*(10) of the squares for the three series?',
                  'For the GARCH it grows with T: 17.8 (p = 0.059) at T = 100, 42.5 at T = 500 and 73.2 at T = 1000 (p < 0.001); from T = 150 on the test rejects at every step. For the i.i.d. noises the statistic does not grow with T: it stays between 2.0 and 21.4 for the Gaussian series and between 3.2 and 8.0 for the Laplace series. The Gaussian series nevertheless rejects at T = 150 and 200 (p = 0.039 and 0.018): a false rejection, which happens in about 5% of the samples. The GARCH squares are correlated although the series stays uncorrelated; with T = 1000, Q*(10) is 9.3 at α = 0 and 838.9 at α = 0.14.'),
  'random-walk': ('The ▶ button moves the date t from 10 to 250. Does the correlation between X<sub>t</sub> and X<sub>t+10</sub> depend only on the distance of 10 periods?',
                  'No. Over the 5000 paths, the correlation is 0.708 between X<sub>10</sub> and X<sub>20</sub> and 0.954 between X<sub>100</sub> and X<sub>110</sub>, like the theoretical values √(t/s) = 0.707 and 0.953. For a stationary process the correlation would depend only on the lag. '
                  'The variance also grows with t: 10.1 at t = 10, 98.1 at t = 100 and 249.1 at t = 250 (theoretical tσ²).'),
  'spot-real': ('One panel is the log price of the BET over its last 500 trading days, the other five are random walks with the same drift and the same daily standard deviation. Which panel is the BET?',
                'Series D: the BET, 11 September 2024 – 18 September 2026. By eye it cannot be told apart from the random walks. Its daily changes set it apart: ρ̂(1) = 0.123, and the p-value of the Ljung–Box test Q*(10) is 0.018, the only one below 5% among the six series; '
                'for the simulated random walks the p-values lie between 0.377 and 0.911.'),
  'sp500-diff': ('The bottom panel shows the returns, Δ ln P<sub>t</sub>. What ρ̂(1) does the second difference, Δ² ln P<sub>t</sub>, have?',
                 '−0.543. The returns are already close to white noise (ρ̂(1) = −0.099), and the difference of white noise is an MA(1) with ρ(1) = −0.5: one more difference creates negative autocorrelation and raises the standard deviation from 1.21% to 1.80%. '
                 'The level ln P<sub>t</sub> has an ACF of 0.999 at lag 1 and 0.973 at lag 50; at lag 1 the ACF stays above 0.99 for every start of the sample between 2000 and 2024.'),
  'wold': ('The ▶ button raises φ in the first panel from 0 to 1; the second panel gets −φ and the third θ = φ. What happens to Σψ<sub>j</sub>²?',
           'It rises: Σψ<sub>j</sub>² = 1/(1 − φ²) is 1 at φ = 0, 2.78 at φ = 0.8 and 10.26 at φ = 0.95, where ψ<sub>10</sub> = 0.599. At φ = 1, ψ<sub>j</sub> = 1 for every j, as for the random walk (the dots in the last panel reach the bars), and the sum is infinite: the process no longer has a Wold form and is no longer stationary. The second panel, with −φ, has the same sum, because ψ<sub>j</sub>² = φ<sup>2j</sup>; for the MA(1), Σψ<sub>j</sub>² = 1 + θ² stays at most 2.'),
  'ergodicity': ('The ▶ button extends the horizon T of the averages from 10 to 2000. What does X̄<sub>T</sub> approach on each path, on the left and on the right?',
                 'On the left, μ = 0 on every path: X̄<sub>T</sub> lies between −1.95 and 0.77 at T = 10, between −0.33 and 0.63 at T = 100 and between −0.13 and 0.09 at T = 2000; the AR(1) is ergodic for the mean. On the right, each average approaches its own Z: at T = 2000 the averages lie between −0.95 and 1.16, like the drawn values of Z (between −0.99 and 1.15), and the spread does not shrink with T. With φ closer to 1, convergence on the left is slower: the standard deviation of X̄<sub>2000</sub> is about 1/((1 − φ)√T), that is 0.075 at φ = 0.7 and 2.24 at φ = 0.99.'),
  'bartlett': ('The ▶ button raises the number of lags m from 1 to 20, on the same 10000 samples of white noise. How often is at least one bar outside the band?',
               'More and more often: in 4.5% of the samples at m = 1, 20.3% at m = 5 and 50.3% at m = 20, although there is no correlation at all. With independent tests it would be 1 − 0.95<sup>m</sup>: 5.0%, 22.6% and 64.2%; '
               'at large lags Var(ρ̂(h)) < 1/T, so slightly fewer bars fall outside. At m = 20, on average 0.73 bars fall outside. '
               'On the left, the samples with at least one bar outside the band (red) often have ρ̂(1) close to 0: the alarm comes from the other lags.'),
  'acf-models': ('The ▶ button raises T from 50 to 500. What happens to the sample ACF of the random walk at lag 10?',
                 'It rises: ρ̂(10) is −0.08 at T = 50, 0.37 at T = 100 and 0.55 at T = 500. The random walk has no theoretical ACF, and the decay of its sample ACF depends on the length of the sample. '
                 'For the AR(1) with φ = 0.8, ρ̂(10) approaches the theoretical value 0.11: −0.14 at T = 50, 0.11 at T = 100 and 0.09 at T = 500, while the band narrows from ±0.277 to ±0.088.'),
  'acf-pacf': ('The ▶ button moves the last coefficient of each process from −0.4 to 0.45: φ of the AR(1), φ₂ of the AR(2) (φ₁ = 0.5) and θ of the MA(1). Which value does the sample PACF of the AR(2) at lag 2 follow?',
               'The coefficient φ₂: φ̂<sub>22</sub> = −0.40 for φ₂ = −0.4, −0.03 for φ₂ = 0 and 0.38 for φ₂ = 0.45 (in theory, φ<sub>22</sub> = φ₂). φ̂<sub>33</sub> stays inside the band (0.05, 0.04, −0.01): the PACF cuts off after lag 2. ρ̂(2) does not follow φ₂: at φ₂ = 0 it is 0.28, close to ρ(2) = φ₁² = 0.25. At the same time φ̂<sub>11</sub> of the AR(1) follows φ (−0.38 at φ = −0.4 and 0.48 at φ = 0.45), and ρ̂(1) of the MA(1) follows θ/(1 + θ²) (−0.30 and 0.37; in theory −0.345 and 0.374).'),
  'bet-acf': ('Select the S&P 500. Do its returns have the same first-order autocorrelation as the BET? And their squares?',
              'For the returns, no: ρ̂(1) = −0.099 for the S&P 500 and 0.109 for the BET (0.170 for EUR/RON). For the squares, yes: ρ̂(1) = 0.31 for the S&P 500 and 0.33 for the BET, with Q*(10) = 6133 and 2440. '
              'The sign of the autocorrelation of returns differs from one market to another, while volatility clustering appears in all three. If the BET sample starts in 2013, ρ̂(1) falls to 0.054 for the returns and to 0.18 for the squares.'),
  'lb-size': ('With T = 50, the ▶ button raises m from 1 to 20. How do the two tests move away from the 5% level?',
              'In opposite directions. Box–Pierce rejects the true hypothesis in 4.5% of the samples at m = 1, in 4.0% at m = 10 and in 2.4% at m = 20; Ljung–Box in 5.2%, 7.4% and 9.4%. '
              'With T = 500 and m = 20 the rates are 5.0% and 5.9%. This is why we keep m well below T (for example, m ≤ T/5).'),
  'lb-stats': ('If the sample starts in 2025 (about 425 days), does the Ljung–Box test with m = 10 still reject white noise for the BET returns?',
               'No: Q*(10) = 14.3, p = 0.161, below the critical value 18.31. On the full sample, with 6670 days, Q*(10) = 108.0, p < 0.001. The small autocorrelation of the BET returns (ρ̂(1) = 0.11) needs many observations to be separated from noise; '
               'the non-rejection does not show that the series is white noise. The squares reject on the short sample too: Q*(10) = 68.6 for the BET and 103.0 for the S&P 500.'),
  'gdp-transform': ('The ▶ button raises the lag k of the difference on the right from 1 to 8. For which k does the seasonal spike of the ACF at lag 4 disappear?',
                    'For k = 4: ρ̂(4) = 0.03. For k = 1, 2, 3 and 5, ρ̂(4) stays between 0.94 and 0.96: a difference that does not cover a whole year keeps the season. '
                    'k = 8 removes the season too, but ρ̂(4) = 0.52, because two two-year growth rates one year apart share a year. The annual growth rate, k = 4, has a mean of 2.73% and ρ̂(1) = 0.63.'),
  'boxcox': ('With λ = 1 (no transformation), how many times larger is the seasonal amplitude in the last five years than in the first five?',
             '10.5 times: 3059 million EUR in the first five years and 32155 in the last five, because the seasonal swing is a stable share of the level. With λ = 0, the logarithm, the ratio is 1.07 (0.392 and 0.420), '
             "and Guerrero's criterion, 0.243, is practically the minimum reached at λ̂ = 0.02; at λ = 1 the criterion is 0.677."),
  'overdiff': ('The ▶ button raises φ of the differenced series from 0 (white noise) to 1 (random walk). How does ρ̂(1) of the first difference change?',
               'It rises from −0.54 towards 0: −0.28 at φ = 0.5, −0.10 at φ = 0.9 and −0.06 at φ = 1, in line with the theoretical value −(1 − φ)/2. Differencing is appropriate only for the random walk, where ΔX<sub>t</sub> = ε<sub>t</sub>; '
               'for a stationary series, a negative ρ̂(1) close to −0.5 signals over-differencing. The variance of the first difference falls from 2.01 to 0.95.'),
  'textbook': ('The ▶ button moves the year of the break from 1875 to 1965; the mean up to the break and the mean after it are removed from the Nile flow and from the sunspots. What happens to ρ̂(1) of the Nile after removing the two means?',
               'It is smallest around 1898: 0.13 for a break after 1897 and 0.16 after 1898, with Q*(10) = 13.0 (p = 0.225). With the break after 1880 or after 1920, ρ̂(1) is 0.40 and 0.42, and after 1950 it is 0.50, the value of the flow itself. A large part of the autocorrelation of the flow comes from the change in the mean around 1898. For the sunspots the same operation hardly changes the ACF: with the break after 1898, ρ̂(1) = 0.81 (0.82 before) and the peak at lag 10 is 0.64 (0.66 before), because their autocorrelation comes from the cycle of about 11 years, not from a break.'),
 },
}

# Questions of the EXTRA blocks
QX = {
 'ro': {
  'acov-props': ('Cu ρ(1) = 0,9, butonul ▶ crește ρ(2) de la −1 la 1. De la ce valoare a lui ρ(2) este perechea (ρ(1), ρ(2)) o ACF? Ajunge vectorul a = (1, −1, 1) pentru a decide?',
                 'De la ρ(2) = 2 × 0,9² − 1 = 0,62. Cu a = (1, −1, 1), varianța este γ(0)(3 − 4ρ(1) + 2ρ(2)): −2,4γ(0) la ρ(2) = −0,9 și 0 la ρ(2) = 0,30. Între 0,30 și 0,62, acest vector nu mai arată nicio contradicție, '
                 'dar cea mai mică valoare proprie a matricei de autocorelație este încă negativă (−0,13 la ρ(2) = 0,30, −0,01 la 0,60): un alt vector a dă o varianță negativă. Cu a = (1, −1, −1) în schimb, și ρ(1) = 0,9, ρ(2) = −0,9, varianța este 4,8γ(0) > 0, '
                 'deci contradicția apare numai cu vectorul (1, −1, 1).'),
  'wx-ma1': ('Butonul ▶ mută θ de la −2 la 2. Mai există un θ, în afară de 0,6, cu ρ(1) = 0,441?',
             'Da: θ = 1/0,6 = 1,67, pentru că θ/(1 + θ²) nu se schimbă cînd θ este înlocuit cu 1/θ; pe slider, θ = 1,65 dă ρ(1) = 0,443 și θ = 1,70 dă 0,437. Cele două procese au aceeași ACF, dar alt γ(0): 1,36 la θ = 0,6 și 3,72 la θ = 1,65 (σ² = 1). '
             'ρ(1) nu identifică singur θ; |ρ(1)| atinge maximul 0,5 la θ = 1.'),
  'wx-ar1': ('Butonul ▶ crește φ de la 0 la 0,99. Ce parte din Σψ<sub>j</sub><sup>2</sup> dau primele 11 ponderi Wold (j = 0, …, 10)?',
             'O parte tot mai mică: 99,3% la φ = 0,8, 67,6% la φ = 0,95 și 19,8% la φ = 0,99, pentru că partea este 1 − φ<sup>22</sup>. Suma însăși, γ(0)/σ² = 1/(1 − φ²), crește de la 2,78 la 10,26 și 50,25, '
             'iar un șoc își pierde jumătate din efect după 3,1, 13,5 și 69,0 perioade. Cu cît φ este mai aproape de 1, cu atît contează mai mult șocurile îndepărtate.'),
  'wx-diff': ('Alegeți PIB-ul real și trimestrul T2 2020. Ce arată creșterea trimestrială și ce arată creșterea anuală?',
              'Creșterea trimestrială este pozitivă, 7,76% (logaritmică) și 8,07% (simplă), pentru că PIB-ul neajustat sezonier al trimestrului al doilea este mereu peste cel al primului. Creșterea anuală, 100 Δ<sub>4</sub> ln Y<sub>t</sub> = −9,87%, '
              'compară trimestrul cu același trimestru din 2019 și arată căderea din pandemie. Pe datele neajustate sezonier, Δ amestecă sezonul cu economia, iar Δ<sub>4</sub> le separă.'),
  'wx-sacf': ('Cu datele din slide, butonul ▶ mută lagul k de la 1 la 4. La ce lag este |ρ̂(k)| cel mai mare și este acesta semnificativ?',
              'La k = 3 și k = 4: ρ̂(3) = −0,20 și ρ̂(4) = −0,40, față de ρ̂(1) = 0,10 și ρ̂(2) = 0,00. ρ̂(4) vine dintr-un singur produs, (−2) × 2 = −4, împărțit la T = 5, și rămîne în banda ±0,88. '
              'Cu T = 5 nicio autocorelație nu poate fi separată de zgomot; de aceea folosim h ≤ T/4.'),
  'bands': ('Alegeți creșterea trimestrială a PIB-ului (T = 125) și lagul h = 8. Cît de largă este banda Bartlett față de ±1,96/√T?',
            'De două ori mai largă: ±0,361 față de ±0,175, pentru că ρ̂(1), …, ρ̂(7) ale acestei serii sezoniere sînt mari (Σρ̂(k)² = 1,62). ρ̂(8) = 0,93 rămîne în afara ambelor benzi. '
            'Din 20 de laguri, toate cele 20 de bare ies din banda ±1,96/√T și 10 din banda Bartlett. Pentru randamentele BET, banda Bartlett la h = 5 este ±0,0243, aproape cît ±0,0240: acolo cele două benzi dau același răspuns.'),
  'wx-lb': ('Cu cifrele din slide, cum se schimbă p-value-ul testului Ljung–Box de la m = 1 la m = 3?',
            'Crește: 0,011 la m = 1 (Q*(1) = 6,44), 0,019 la m = 2 (Q*(2) = 7,94) și 0,035 la m = 3 (Q*(3) = 8,61). Statistica crește puțin, pentru că ρ̂(2) și ρ̂(3) sînt mici, iar valoarea critică crește cu un grad de libertate la fiecare lag: '
            'un m prea mare diluează dovezile. Box–Pierce dă Q(3) = 8,33, cu p = 0,040.'),
  'lb-table': ('Butonul ▶ crește m de la 1 la 30. Pentru ce serie sînt dovezile cele mai slabe la m = 1 și ce se întîmplă la m = 4?',
               'Creșterea trimestrială a PIB-ului: Q*(1) = 7,7, p = 0,006, doar de 2,0 ori valoarea critică 3,84. La m = 4, Q*(4) = 168,1 față de 9,49, pentru că ρ̂(4) = 0,96: sezonul. '
               'Toate cele 10 serii resping ipoteza de zgomot alb pentru orice m între 1 și 30; la m = 10, cel mai mic raport îl are creșterea anuală a PIB-ului, 4,1.'),
  'wx-boxcox': ('Alegeți PIB-ul real. Rămîne logaritmul alegerea criteriului lui Guerrero?',
                'Nu: pentru PIB-ul real, λ̂ = 0,39, cu criteriul 0,110, față de 0,165 la λ = 0 și 0,218 la λ = 1. Pentru PIB-ul nominal, λ̂ = 0,02, iar criteriul este 0,243 atît la λ̂, cît și la λ = 0, deci alegem logaritmul. '
                'Valoarea lui λ̂ depinde de serie; la indicele IAPC, λ̂ = 0,87.'),
 },
 'en': {
  'acov-props': ('With ρ(1) = 0.9, the ▶ button raises ρ(2) from −1 to 1. From which value of ρ(2) is the pair (ρ(1), ρ(2)) an ACF? Is the vector a = (1, −1, 1) enough to decide?',
                 'From ρ(2) = 2 × 0.9² − 1 = 0.62. With a = (1, −1, 1) the variance is γ(0)(3 − 4ρ(1) + 2ρ(2)): −2.4γ(0) at ρ(2) = −0.9 and 0 at ρ(2) = 0.30. Between 0.30 and 0.62 this vector shows no contradiction, '
                 'but the smallest eigenvalue of the autocorrelation matrix is still negative (−0.13 at ρ(2) = 0.30, −0.01 at 0.60): another vector a gives a negative variance. With a = (1, −1, −1) instead, and ρ(1) = 0.9, ρ(2) = −0.9, the variance is 4.8γ(0) > 0, '
                 'so the contradiction appears only with the vector (1, −1, 1).'),
  'wx-ma1': ('The ▶ button moves θ from −2 to 2. Is there another θ, besides 0.6, with ρ(1) = 0.441?',
             'Yes: θ = 1/0.6 = 1.67, because θ/(1 + θ²) does not change when θ is replaced by 1/θ; on the slider, θ = 1.65 gives ρ(1) = 0.443 and θ = 1.70 gives 0.437. The two processes have the same ACF but a different γ(0): 1.36 at θ = 0.6 and 3.72 at θ = 1.65 (σ² = 1). '
             'ρ(1) alone does not identify θ; |ρ(1)| reaches its maximum 0.5 at θ = 1.'),
  'wx-ar1': ('The ▶ button raises φ from 0 to 0.99. What share of Σψ<sub>j</sub><sup>2</sup> do the first 11 Wold weights (j = 0, …, 10) give?',
             'A smaller and smaller share: 99.3% at φ = 0.8, 67.6% at φ = 0.95 and 19.8% at φ = 0.99, because the share is 1 − φ<sup>22</sup>. The sum itself, γ(0)/σ² = 1/(1 − φ²), rises from 2.78 to 10.26 and 50.25, '
             'and a shock loses half of its effect after 3.1, 13.5 and 69.0 periods. The closer φ is to 1, the more the distant shocks matter.'),
  'wx-diff': ('Select real GDP and the quarter 2020Q2. What does the quarterly growth rate show, and what does the annual one show?',
              'The quarterly growth rate is positive, 7.76% (log) and 8.07% (simple), because the not seasonally adjusted GDP of the second quarter is always above that of the first. The annual growth rate, 100 Δ<sub>4</sub> ln Y<sub>t</sub> = −9.87%, '
              'compares the quarter with the same quarter of 2019 and shows the pandemic fall. On not seasonally adjusted data, Δ mixes the season with the economy, and Δ<sub>4</sub> separates them.'),
  'wx-sacf': ('With the data of the slide, the ▶ button moves the lag k from 1 to 4. At which lag is |ρ̂(k)| largest, and is it significant?',
              'At k = 3 and k = 4: ρ̂(3) = −0.20 and ρ̂(4) = −0.40, against ρ̂(1) = 0.10 and ρ̂(2) = 0.00. ρ̂(4) comes from a single product, (−2) × 2 = −4, divided by T = 5, and stays inside the band ±0.88. '
              'With T = 5 no autocorrelation can be separated from noise; this is why we use h ≤ T/4.'),
  'bands': ('Select the quarterly GDP growth (T = 125) and the lag h = 8. How wide is the Bartlett band compared with ±1.96/√T?',
            'Twice as wide: ±0.361 against ±0.175, because ρ̂(1), …, ρ̂(7) of this seasonal series are large (Σρ̂(k)² = 1.62). ρ̂(8) = 0.93 stays outside both bands. '
            'Of 20 lags, all 20 bars fall outside ±1.96/√T and 10 outside the Bartlett band. For the BET returns, the Bartlett band at h = 5 is ±0.0243, almost the same as ±0.0240: there the two bands give the same answer.'),
  'wx-lb': ('With the numbers of the slide, how does the p-value of the Ljung–Box test change from m = 1 to m = 3?',
            'It rises: 0.011 at m = 1 (Q*(1) = 6.44), 0.019 at m = 2 (Q*(2) = 7.94) and 0.035 at m = 3 (Q*(3) = 8.61). The statistic grows a little, because ρ̂(2) and ρ̂(3) are small, while the critical value gains one degree of freedom at each lag: '
            'too large an m dilutes the evidence. Box–Pierce gives Q(3) = 8.33, with p = 0.040.'),
  'lb-table': ('The ▶ button raises m from 1 to 30. For which series is the evidence weakest at m = 1, and what happens at m = 4?',
               'The quarterly GDP growth: Q*(1) = 7.7, p = 0.006, only 2.0 times the critical value 3.84. At m = 4, Q*(4) = 168.1 against 9.49, because ρ̂(4) = 0.96: the season. '
               'All 10 series reject white noise for every m between 1 and 30; at m = 10 the smallest ratio belongs to the annual GDP growth, 4.1.'),
  'wx-boxcox': ("Select real GDP. Is the logarithm still the choice of Guerrero's criterion?",
                'No: for real GDP, λ̂ = 0.39, with criterion 0.110, against 0.165 at λ = 0 and 0.218 at λ = 1. For nominal GDP, λ̂ = 0.02 and the criterion is 0.243 both at λ̂ and at λ = 0, so we choose the logarithm. '
                'The value of λ̂ depends on the series; for the HICP index, λ̂ = 0.87.'),
 },
}
for _lg in QA:
    QA[_lg].update(QX[_lg])

# „Ce credeți?” / "What do you think?": options of the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with what the page computes (initial state, or the end of ▶). Each wrong option encodes one misconception.
PREDICT = {
 'four-series': (1, [('PIB-ul real: crește lent și regulat, deci media lui este stabilă.', 'Real GDP: it grows slowly and smoothly, so its mean is stable.'),
                     ('Randamentele BET; abaterea lor standard se schimbă însă de la o fereastră la alta.', 'The BET returns; their standard deviation, however, changes from one window to the next.'),
                     ('Cursul EUR/RON: variază puțin de la o zi la alta, deci și media lui variază puțin.', 'EUR/RON: it moves little from one day to the next, so its mean moves little too.')]),
 'ensemble': (0, [('În stînga rămîne aceeași la orice t; în dreapta crește ca √t, pînă la ±19,6 la t = 100.', 'On the left it stays the same at every t; on the right it grows like √t, up to ±19.6 at t = 100.'),
                  ('Crește în ambele panouri: și la AR(1) șocurile se adună în timp.', 'It grows in both panels: the shocks add up over time for the AR(1) too.'),
                  ('Rămîne aceeași în ambele: fiecare pas are aceeași varianță a șocului, 1.', 'It stays the same in both: every step has the same shock variance, 1.')]),
 'counterexample': (1, [('Media și varianța de la t impar se apropie de 0 și 1 abia pentru k mare.', 'The mean and variance at odd t approach 0 and 1 only for large k.'),
                        ('Media rămîne 0 și varianța 1 pentru orice k; scade doar asimetria, de la 2,83 la 0,37.', 'The mean stays 0 and the variance 1 for every k; only the skewness falls, from 2.83 to 0.37.'),
                        ('Nimic: procesul este slab staționar, deci distribuția lui Xₜ nu se poate schimba.', 'Nothing: the process is weakly stationary, so the distribution of Xₜ cannot change.')]),
 'nonstationary': (0, [('B și D încalcă (ii), media constantă; C încalcă (iii) la h = 0, varianța constantă.', 'B and D violate (ii), the constant mean; C violates (iii) at h = 0, the constant variance.'),
                       ('Doar B: un salt unic, ca în D, și o varianță variabilă, ca în C, nu contează.', 'Only B: a single jump, as in D, and a changing variance, as in C, do not matter.'),
                       ('C încalcă (ii), pentru că valorile ei extreme cresc; B și D sînt staționare în jurul unei drepte.', 'C violates (ii), because its extreme values grow; B and D are stationary around a line.')]),
 'white-noise': (2, [('Crește pentru toate trei: cu mai multe observații, orice test respinge.', 'It grows for all three: with more observations every test rejects.'),
                     ('Crește și pentru seria Laplace: cozile groase fac pătratele corelate.', 'It grows for the Laplace series too: fat tails make the squares correlated.'),
                     ('Crește doar pentru GARCH, de la 17,8 la 73,2; la seriile i.i.d. nu crește cu T.', 'It grows only for the GARCH, from 17.8 to 73.2; for the i.i.d. series it does not grow with T.')]),
 'random-walk': (0, [('Nu: 0,71 la t = 10 și 0,95 la t = 100.', 'No: 0.71 at t = 10 and 0.95 at t = 100.'),
                     ('Da: distanța este aceeași, deci și corelația este aceeași.', 'Yes: the distance is the same, so the correlation is the same.'),
                     ('Nu: corelația scade spre 0 pe măsură ce t crește.', 'No: the correlation falls towards 0 as t grows.')]),
 'spot-real': (1, [('Seria cu trendul cel mai clar: un indice bursier crește în timp.', 'The series with the clearest trend: a stock index rises over time.'),
                   ('Seria D; nivelul nu o deosebește, dar variațiile ei zilnice au ρ̂(1) = 0,12 (p = 0,018).', 'Series D; the level does not set it apart, but its daily changes have ρ̂(1) = 0.12 (p = 0.018).'),
                   ('Seria cu cele mai mari oscilații: piața reală este mai volatilă decît o simulare.', 'The series with the largest swings: the real market is more volatile than a simulation.')]),
 'sp500-diff': (1, [('Aproape 0: diferențierea elimină orice autocorelație.', 'Close to 0: differencing removes any autocorrelation.'),
                    ('În jur de −0,5: diferența unui zgomot alb este un MA(1) cu ρ(1) = −0,5.', 'Around −0.5: the difference of white noise is an MA(1) with ρ(1) = −0.5.'),
                    ('Aproape 1, ca ACF a nivelului.', 'Close to 1, like the ACF of the level.')]),
 'wold': (2, [('Rămîne finită pentru orice φ ≤ 1, pentru că fiecare ψⱼ este cel mult 1.', 'It stays finite for every φ ≤ 1, because every ψⱼ is at most 1.'),
              ('Scade, pentru că ponderile se sting mai repede.', 'It falls, because the weights die out faster.'),
              ('Crește la 10,26 pentru φ = 0,95 și devine infinită la φ = 1.', 'It rises to 10.26 at φ = 0.95 and becomes infinite at φ = 1.')]),
 'ergodicity': (0, [('În stînga spre 0 pe toate traiectoriile; în dreapta fiecare spre propriul Z, iar împrăștierea rămîne.', 'On the left, 0 on every path; on the right, each its own Z, and the spread remains.'),
                    ('Spre 0 în ambele panouri; în dreapta doar mai lent.', 'Towards 0 in both panels; on the right only more slowly.'),
                    ('În stînga fiecare spre propria medie, pentru că φ = 0,7 este persistent; în dreapta spre 0.', 'On the left each towards its own mean, because φ = 0.7 is persistent; on the right towards 0.')]),
 'bartlett': (1, [('În circa 5% dintre eșantioane, oricare ar fi m: banda corespunde nivelului de 5%.', 'In about 5% of the samples, whatever m: the band corresponds to the 5% level.'),
                  ('Din ce în ce mai des: 4,5% la m = 1 și 50,3% la m = 20.', 'More and more often: 4.5% at m = 1 and 50.3% at m = 20.'),
                  ('Niciodată: datele sînt zgomot alb.', 'Never: the data are white noise.')]),
 'acf-models': (2, [('Rămîne aproape aceeași: ρ̂(10) estimează ρ(10) al mersului aleator.', 'It stays about the same: ρ̂(10) estimates ρ(10) of the random walk.'),
                    ('Scade spre 0, ca la AR(1).', 'It falls towards 0, as for the AR(1).'),
                    ('Crește de la −0,08 la 0,55: ACF de selecție a mersului aleator depinde de T.', 'It rises from −0.08 to 0.55: the sample ACF of the random walk depends on T.')]),
 'acf-pacf': (0, [('Coeficientul φ₂: de la −0,40 la 0,38.', 'The coefficient φ₂: from −0.40 to 0.38.'),
                  ('ρ(2), autocorelația de la lagul 2.', 'ρ(2), the autocorrelation at lag 2.'),
                  ('φ₁ = 0,5, pentru că primul coeficient domină.', 'φ₁ = 0.5, because the first coefficient dominates.')]),
 'bet-acf': (1, [('Da: ambele au ρ̂(1) pozitiv, în jur de 0,1, ca orice piață.', 'Yes: both have a positive ρ̂(1), around 0.1, like any market.'),
                 ('Nu: S&P 500 are ρ̂(1) = −0,099; pătratele au însă ρ̂(1) ≈ 0,3 pe ambele piețe.', 'No: the S&P 500 has ρ̂(1) = −0.099; the squares, however, have ρ̂(1) ≈ 0.3 on both markets.'),
                 ('Nu: S&P 500 este un zgomot alb i.i.d., cu pătrate necorelate.', 'No: the S&P 500 is i.i.d. white noise, with uncorrelated squares.')]),
 'lb-size': (2, [('Ambele rămîn la 5%: nivelul este fixat de valoarea critică χ².', 'Both stay at 5%: the level is fixed by the χ² critical value.'),
                 ('Ambele resping prea des, cu aceeași rată.', 'Both reject too often, at the same rate.'),
                 ('Box–Pierce coboară la 2,4%, Ljung–Box urcă la 9,4%.', 'Box–Pierce falls to 2.4%, Ljung–Box rises to 9.4%.')]),
 'lb-stats': (0, [('Nu: Q*(10) = 14,3, p = 0,161; eșantionul scurt nu separă corelația mică de zgomot.', 'No: Q*(10) = 14.3, p = 0.161; the short sample does not separate the small correlation from noise.'),
                  ('Da: o serie care respinge pe tot eșantionul respinge pe orice subeșantion.', 'Yes: a series that rejects on the full sample rejects on every subsample.'),
                  ('Nu, deci randamentele BET au devenit zgomot alb după 2025.', 'No, so the BET returns have become white noise after 2025.')]),
 'gdp-transform': (1, [('k = 1: prima diferență elimină și trendul, și sezonul.', 'k = 1: the first difference removes both the trend and the season.'),
                       ('k = 4, perioada sezonului: ρ̂(4) = 0,03.', 'k = 4, the period of the season: ρ̂(4) = 0.03.'),
                       ('Pentru orice k ≥ 2.', 'For every k ≥ 2.')]),
 'boxcox': (2, [('Aproape la fel: sezonalitatea nu depinde de nivel.', 'About the same: the seasonality does not depend on the level.'),
                ('De circa 15 ori, cît a crescut nivelul de la primul la ultimul trimestru.', 'About 15 times, as much as the level grew from the first to the last quarter.'),
                ('De 10,5 ori; după logaritmare, raportul scade la 1,07.', '10.5 times; after taking logs, the ratio falls to 1.07.')]),
 'overdiff': (1, [('Rămîne în jur de −0,5 pentru orice φ: diferențierea creează mereu un MA(1) cu θ = −1.', 'It stays around −0.5 for every φ: differencing always creates an MA(1) with θ = −1.'),
                  ('Urcă spre 0: −0,28 la φ = 0,5 și −0,06 la φ = 1.', 'It rises towards 0: −0.28 at φ = 0.5 and −0.06 at φ = 1.'),
                  ('Coboară spre −1.', 'It falls towards −1.')]),
 'textbook': (0, [('Este cel mai mic în jurul lui 1898 (0,13–0,16) și crește spre 0,50 departe de el.', 'It is smallest around 1898 (0.13–0.16) and rises towards 0.50 away from it.'),
                  ('Rămîne aproape de 0 pentru orice an: eliminarea a două medii elimină mereu corelația.', 'It stays close to 0 for every year: removing two means always removes the correlation.'),
                  ('Rămîne 0,50: o ruptură nu influențează ACF.', 'It stays at 0.50: a break does not affect the ACF.')]),
}

# blocks whose initial charts show the answer: the live charts are veiled too until „Verifică” / "Check" or a change of the controls
VEIL_CHARTS = ('bartlett', 'lb-size', 'gdp-transform', 'boxcox')

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slide „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('Xₜ = εₜ − 0,5εₜ₋₁, cu σ² = 4. Cît sînt γ(0) și ρ(1)?', 'Xₜ = εₜ − 0.5εₜ₋₁, with σ² = 4. What are γ(0) and ρ(1)?'),
      key=0, opts=[
  (('γ(0) = 5, ρ(1) = −0,4.', 'γ(0) = 5, ρ(1) = −0.4.'),
   ('γ(0) = σ²(1 + θ²) = 4 × 1,25 = 5; ρ(1) = θ/(1 + θ²) = −0,5/1,25 = −0,4.', 'γ(0) = σ²(1 + θ²) = 4 × 1.25 = 5; ρ(1) = θ/(1 + θ²) = −0.5/1.25 = −0.4.')),
  (('γ(0) = 4, ρ(1) = −0,5.', 'γ(0) = 4, ρ(1) = −0.5.'),
   ('γ(0) nu este σ²: termenul θεₜ₋₁ adaugă θ²σ² = 1. Iar ρ(1) nu este θ.', 'γ(0) is not σ²: the term θεₜ₋₁ adds θ²σ² = 1. And ρ(1) is not θ.')),
  (('γ(0) = 5, ρ(1) = −0,5.', 'γ(0) = 5, ρ(1) = −0.5.'),
   ('γ(0) este corect, dar ρ(1) = θ/(1 + θ²), nu θ; |ρ(1)| = 0,5 se atinge doar la |θ| = 1.', 'γ(0) is right, but ρ(1) = θ/(1 + θ²), not θ; |ρ(1)| = 0.5 is reached only at |θ| = 1.'))]),
 dict(q=('O corelogramă cu 40 de laguri pentru o serie cu T = 400 arată două bare ușor în afara lui ±0,098. Este seria corelată?',
         'A correlogram with 40 lags for a series with T = 400 shows two bars slightly outside ±0.098. Is the series correlated?'),
      key=1, opts=[
  (('Da: două bare în afara benzii resping zgomotul alb la 5%.', 'Yes: two bars outside the band reject white noise at 5%.'),
   ('Banda este de 5% pentru fiecare lag luat separat; din 40 de laguri, aproximativ 2 bare ies din întîmplare.', 'The band is 5% for each lag on its own; out of 40 lags, about 2 bars fall outside by chance.')),
  (('Nu neapărat: aproximativ 2 din 40 de bare ies din întîmplare; aplicăm testul Ljung–Box.', 'Not necessarily: about 2 of 40 bars fall outside by chance; we apply the Ljung–Box test.'),
   ('40 × 0,05 = 2 bare sînt de așteptat în afara benzii fără nicio corelație; un test comun pentru toate lagurile decide.', '40 × 0.05 = 2 bars are expected outside the band with no correlation at all; a joint test for all the lags decides.')),
  (('Nu: seria este sigur zgomot alb.', 'No: the series is certainly white noise.'),
   ('Corelograma nu dovedește absența corelației; barele nesemnificative arată doar că testul nu o separă de zgomot.', 'The correlogram does not prove the absence of correlation; non-significant bars only show that the test does not separate it from noise.'))]),
 dict(q=('După diferențiere, ρ̂(1) = −0,48 și nimic altceva nu este semnificativ. Ce sugerează acest lucru?',
         'After differencing, ρ̂(1) = −0.48 and nothing else is significant. What does this suggest?'),
      key=2, opts=[
  (('Seria are nevoie de încă o diferențiere.', 'The series needs one more difference.'),
   ('O nouă diferențiere ar adăuga încă un factor (1 − L) și ar face autocorelația negativă și mai puternică.', 'Another difference would add one more factor (1 − L) and make the negative autocorrelation even stronger.')),
  (('Un AR(1) cu φ = −0,48 în seria diferențiată.', 'An AR(1) with φ = −0.48 in the differenced series.'),
   ('Un singur vîrf în ACF, apoi nimic, este tiparul unui MA(1), nu al unui AR(1), a cărui ACF descrește geometric.', 'A single spike in the ACF, then nothing, is the pattern of an MA(1), not of an AR(1), whose ACF decays geometrically.')),
  (('Supradiferențiere: seria inițială era probabil deja staționară.', 'Over-differencing: the original series was probably already stationary.'),
   ('Diferența unui zgomot alb este un MA(1) cu θ = −1 și ρ(1) = −0,5; un ρ̂(1) apropiat de −0,5 este semnul ei.', 'The difference of white noise is an MA(1) with θ = −1 and ρ(1) = −0.5; a ρ̂(1) close to −0.5 is its signature.'))]),
]


def _predict_js():
    import json
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {}))
              for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch1-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH1_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH1_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n')


_predict_js()


MATHJAX = ('<script>window.MathJax = { tex: { inlineMath: [[\'\\\\(\', \'\\\\)\']], displayMath: [[\'\\\\[\', \'\\\\]\']] }, '
           'startup: { pageReady: () => MathJax.startup.defaultPageReady().then(() => document.body.setAttribute(\'data-mj\', \'done\')) } };</script>\n'
           '<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>\n')


def _post(lang):
    """Insert the EXTRA blocks in slide order (with their section headings) and MathJax; render their slide images."""
    import pymupdf
    import interactive_common as ic
    sp = sys.modules[__name__]
    t, W = L[lang], widgets_all(lang)
    path = os.path.join(ic.OUT, t['file'])
    html = open(path, encoding='utf-8').read()
    viewer = '../pdfjs/web/viewer.html?file=../../' + TEX[lang] + '.pdf'
    chart_pages = ic.slide_pages(sp, lang)
    blocks = {m.group(1): m.group(0) for m in re.finditer(r'\n<section class="blk" id="([^"]+)">.*?</section>', html, flags=re.S)}
    items = [(chart_pages[c][0], a, k, blocks[a]) for a, c, k in BLOCKS]
    xp = _extra_pages(lang)
    doc = pymupdf.open(os.path.join(ic.ROOT, TEX[lang] + '.pdf'))
    for anchor, _, _, skey, _ in EXTRA:
        pg, title = xp[anchor]
        img = f'ch1x_{anchor}_{lang}.jpg'
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
    html = html.replace('</head>', MATHJAX + '</head>', 1)
    open(path, 'w', encoding='utf-8').write(html)
    print('extra blocks', lang, len(EXTRA))


if __name__ == '__main__':
    import interactive_common
    page_spec = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA', 'TEXT_JS')})
    page_spec.widgets = lambda lang: {a: v for a, v in widgets_all(lang).items() if a in {b for b, _, _ in BLOCKS}}
    interactive_common.build(page_spec)
    for lg in ('en', 'ro'):
        _post(lg)
