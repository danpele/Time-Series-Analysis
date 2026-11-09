"""Build the interactive pages of Chapter 13 (bubbles and LPPL models): interactive/ch13.html (EN) and interactive/ch13_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its interactive version on
the right. The worked examples of the lecture (a Blanchard-Watson bubble, SADF and GSADF as a sup over windows, a BSADF
date-stamp crossing, the right-tailed ADF on one Nasdaq 100 window, the log-periodic ratio lambda, an LPPL value by hand, the
filter conditions, the two-step Shanghai fit), its result tables (the six episodes, the Nasdaq 100 tests and episodes, the
three markets, the six fits with their search bounds, the windows, the evaluation) get blocks of their own (EXTRA), placed by
slide, each with a chart; on the web lecture each of them travels inside a chart block and ch13.js moves it to its own
frame. Logic in interactive/ch13.js, inputs in interactive/ch13_data.js (Quantlets/Ch_13/export_interactive.py), slide
images in interactive/slides/.
"""
import json
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 13

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_13'
TEX = {'ro': 'RO/Cursuri/capitol13_bule_lppl', 'en': 'EN/Courses/chapter13_bubbles_lppl'}

_D = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'interactive', 'ch13_data.js'), encoding='utf-8').read()
_D = json.loads(_D[_D.index('{'):_D.rindex('}') + 1])
_END = _D['psy']['ndx']['dates'][1:]
NN = len(_END)                                           # 782 weekly changes of the Nasdaq 100
W0 = int((0.01 + 1.8 / NN ** 0.5) * NN)                  # minimum window, 58 weeks
I_PEAK, I_CONF, I_A, I_B = _END.index('2000-03-24'), _END.index('1998-12-04'), _END.index('1998-06-05'), _END.index('2001-06-29')
F_SSEC = _D['fits']['ssec']['rows']['30']
del _D

# (anchor, chart file, section key); order of the slides
BLOCKS = [('episodes', 'tsa_ch13_episodes', 'hi'),
          ('rational', 'tsa_ch13_rational', 'rb'), ('psy-ndx', 'tsa_ch13_psy_ndx', 'rb'), ('psy-panel', 'tsa_ch13_psy_panel', 'rb'),
          ('growth', 'tsa_ch13_growth', 'lp'), ('components', 'tsa_ch13_lppl_components', 'lp'),
          ('cost', 'tsa_ch13_cost', 'es'), ('fits', 'tsa_ch13_fits', 'es'),
          ('windows', 'tsa_ch13_windows', 'cf'), ('ci', 'tsa_ch13_ci', 'cf'), ('ci-b', 'tsa_ch13_ci_b', 'cf'), ('tc-path', 'tsa_ch13_tc_path', 'cf'),
          ('eval', 'tsa_ch13_eval', 'ev'), ('eval-btc', 'tsa_ch13_eval_btc', 'ev')]

# worked examples and result tables: (anchor, frame title EN, frame title RO, section key, host chart block)
EXTRA = [('t-ep', 'Interpreting the six episodes', 'Interpretarea celor șase episoade', 'hi', 'episodes'),
         ('wx-bw', 'Worked example: the Blanchard–Watson bubble', 'Exemplu rezolvat: bula Blanchard–Watson', 'rb', 'rational'),
         ('wx-sadf', 'SADF, GSADF and BSADF (1/2)', 'SADF, GSADF și BSADF (1/2)', 'rb', 'psy-ndx'),
         ('wx-bsadf', 'SADF, GSADF and BSADF (2/2)', 'SADF, GSADF și BSADF (2/2)', 'rb', 'psy-ndx'),
         ('wx-adf', 'Worked example: one window of the Nasdaq 100', 'Exemplu rezolvat: o fereastră din Nasdaq 100', 'rb', 'psy-ndx'),
         ('t-ndx', 'Interpreting the Nasdaq 100 tests', 'Interpretarea testelor pentru Nasdaq 100', 'rb', 'psy-ndx'),
         ('t-psy', 'Interpreting the three markets', 'Interpretarea celor trei piețe', 'rb', 'psy-panel'),
         ('wx-lam', 'Log-periodic oscillations', 'Oscilațiile log-periodice', 'lp', 'components'),
         ('wx-lppl', 'Worked example: an LPPL value by hand', 'Exemplu rezolvat: o valoare LPPL calculată de mînă', 'lp', 'components'),
         ('wx-filter', 'Parameter constraints: the filter', 'Restricțiile parametrilor: filtrul', 'lp', 'components'),
         ('wx-ssec', 'Worked example: the Shanghai fit at the end of the window', 'Exemplu rezolvat: ajustarea pentru Shanghai la sfîrșitul ferestrei', 'es', 'fits'),
         ('t-fits', 'Interpreting the six fits', 'Interpretarea celor șase ajustări', 'es', 'fits'),
         ('t-win', 'Interpreting the windows', 'Interpretarea ferestrelor', 'cf', 'windows'),
         ('t-eval', 'Interpreting the evaluation', 'Interpretarea evaluării', 'ev', 'eval-btc')]

PFX = {'episodes': 'ep', 'rational': 'ra', 'psy-ndx': 'pn', 'psy-panel': 'pp', 'growth': 'gr', 'components': 'cm', 'cost': 'co', 'fits': 'fi',
       'windows': 'wi', 'ci': 'c1', 'ci-b': 'c2', 'tc-path': 'tp', 'eval': 'e1', 'eval-btc': 'e2',
       't-ep': 'te', 'wx-bw': 'wb', 'wx-sadf': 'ws', 'wx-bsadf': 'wd', 'wx-adf': 'wa', 't-ndx': 'tn', 't-psy': 'ts', 'wx-lam': 'wl', 'wx-lppl': 'wp',
       'wx-filter': 'wf', 'wx-ssec': 'wx', 't-fits': 'tf', 't-win': 'tw', 't-eval': 'tv'}

SECTIONS = {
 'ro': dict(hi='Bule în istorie', rb='Bule raționale și rădăcini explozive', lp='Modelul LPPL', es='Estimarea', cf='Estimarea pe mai multe ferestre',
            ev='Evaluarea prognozelor de crah'),
 'en': dict(hi='Bubbles in history', rb='Rational bubbles and explosive roots', lp='The LPPL model', es='Estimation', cf='Estimation over many windows',
            ev='Evaluation of crash predictions'),
}

L = {
 'ro': dict(file='ch13_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 13 · Bule speculative și modele LPPL', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'casetele „Exemplu rezolvat” refac calculul cu cifrele capitolului.',
   src='Închiderile zilnice ale celor șase episoade și ale evaluării (S&P 500, Nasdaq 100, BET, Shanghai Composite, Bitcoin; data/market, EODHD) nu sînt revizuite și '
       'reproduc exact cifrele din curs; browserul găsește din nou minimul, maximul, scăderile, ratele de reușită și grupurile de alarme. Simulările capitolului: '
       'valoarea fundamentală și șocurile celor trei AR(1) (sămînța 2026), extragerile Normale și uniforme ale bulei Blanchard–Watson (sămînța 13), în ordinea în care '
       'le face codul capitolului. Testele ADF, SADF, GSADF și BSADF sînt calculate în browser din prețurile logaritmice săptămînale; valorile lor critice sînt cele '
       'Monte Carlo ale capitolului (1000 de mersuri aleatoare cu driftul 1/T, sămînța 2026), exportate cu cuantilele de 90%, 95% și 99%. Ajustările LPPL, cu '
       'testele Lomb și de rădăcină unitară ale reziduurilor, peisajul funcției de cost și seriile indicatorului (fișierele ch13_ci_*.csv) provin din codul capitolului; '
       'pasul liniar (OLS) al exemplului Shanghai este refăcut în browser. Variantele care nu apar în curs (alte orizonturi, alte valori ale lui r și π, alte niveluri '
       'și lungimi minime, alte date de sfîrșit ale ferestrei LPPL, alte limite ale filtrului și ale lui ω, alte praguri, alte definiții ale crahului) sînt rulări '
       'suplimentare ale codului capitolului pe aceleași date; graficele din „Interpretarea celor șase episoade”, „SADF, GSADF și BSADF”, „Exemplu rezolvat”, '
       '„Interpretarea testelor pentru Nasdaq 100”, „Interpretarea celor trei piețe”, „Oscilațiile log-periodice”, „Restricțiile parametrilor: filtrul”, '
       '„Interpretarea celor șase ajustări”, „Interpretarea ferestrelor” și „Interpretarea evaluării” sînt grafice suplimentare pentru slide-uri care dau doar cifre.'),
 'en': dict(file='ch13.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 13 · Speculative Bubbles and LPPL Models', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; '
        'the “Worked example” boxes redo the calculation with the numbers of the chapter.',
   src='The daily closes of the six episodes and of the evaluation (S&P 500, Nasdaq 100, BET, Shanghai Composite, Bitcoin; data/market, EODHD) are not revised and '
       'reproduce the numbers of the lecture exactly; the browser finds the low, the peak, the falls, the hit rates and the alarm clusters again. Simulations of the '
       'chapter: the fundamental value and the shocks of the three AR(1) paths (seed 2026), the Normal and uniform draws of the Blanchard–Watson bubble (seed 13), in '
       'the order in which the chapter’s code draws them. The ADF, SADF, GSADF and BSADF tests are computed in the browser from the weekly log prices; their critical '
       'values are the chapter’s Monte Carlo values (1000 random walks with drift 1/T, seed 2026), exported with the 90%, 95% and 99% quantiles. The LPPL fits, with the '
       'Lomb and unit-root tests of their residuals, the cost landscape and the indicator series (the files ch13_ci_*.csv) come from the chapter’s code; the linear step '
       '(OLS) of the Shanghai example is redone in the browser. Variants that do not appear in the lecture (other horizons, other values of r and π, other levels and '
       'minimum lengths, other end dates of the LPPL window, other filter bounds and limits of ω, other thresholds, other crash definitions) are extra runs of the '
       'chapter’s code on the same data; the charts of “Interpreting the six episodes”, “SADF, GSADF and BSADF”, “Worked example”, “Interpreting the Nasdaq 100 tests”, '
       '“Interpreting the three markets”, “Log-periodic oscillations”, “Parameter constraints: the filter”, “Interpreting the six fits”, “Interpreting the windows” and '
       '“Interpreting the evaluation” are extra charts for slides that give only numbers.'),
}

CSS = ('<style>.c13x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c13x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c13x .c13f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c13x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c13x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c13x .wctl .ctl{max-width:360px;margin:.15rem 0 .2rem}'
       '.c13t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c13t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c13t th,.c13t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c13t thead tr:last-child th{border-bottom:1px solid #1A3A6E}'
       '.c13t tbody tr:nth-child(even){background:#F5F7FB}.c13t tbody tr.hi{background:#E3F4F7}.c13t .r{text-align:right}.c13t .c{text-align:center}'
       '.c13cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}</style>')


def tbl(i):
    return f'<div class="tbl c13t" id="{i}"></div>'


def wx(i, title, formula, ctl=''):
    """Worked-example box; `ctl` holds the controls that only drive the box (they stay out of the chart's controls bar)."""
    return f'<div class="c13x"><h4>{title}</h4>' + (f'<div class="wctl">{ctl}</div>' if ctl else '') + f'<div class="c13f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def box(ctl, inner):
    """A box with controls that only drive the table and the chart of the block."""
    return f'<div class="c13x"><div class="wctl">{ctl}</div>{inner}</div>'


def slider_any(i, label, lo, hi, val):
    """A range input with step="any", so that the initial value is the exact estimate of the lecture."""
    return f'<div class="ctl"><label for="{i}">{label}: <b id="{i}-v"></b></label><input type="range" id="{i}" min="{lo}" max="{hi}" step="any" value="{val}"></div>'


def widgets_all(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    lec = s(' (curs)', ' (lecture)')
    W = s('Exemplu rezolvat', 'Worked example')
    hor = lambda i: select(i, s('Orizontul scăderii după maximum', 'Horizon of the fall after the peak'),  # noqa: E731
                           [(str(m), s(f'{m} luni', f'{m} months') + (lec if m == 12 else '')) for m in (3, 6, 9, 12, 18)], '12')
    lev = lambda i: select(i, s('Nivelul valorilor critice', 'Level of the critical values'), [('95', '95%' + lec), ('90', '90%'), ('99', '99%')], '95')  # noqa: E731
    mlen = lambda i, v: slider(i, s(f'Lungimea minimă a unui episod (săptămîni; curs: {v})', f'Minimum length of an episode (weeks; lecture: {v})'), 1, 20, 1, v)  # noqa: E731
    eps = [(k, lab) for k, lab in (('sp500', 'S&P 500, 2000'), ('ndx', 'Nasdaq 100, 2000'), ('bet', 'BET, 2007'), ('ssec', 'Shanghai Composite, 2015'), ('btc17', 'Bitcoin, 2017'), ('btc21', 'Bitcoin, 2021'))]
    lags = [(str(v), s(f'{v} de zile', f'{v} days') if v not in (0, 15) else s(f'{v} zile', f'{v} days') + ('' if v else s(' (la maximum)', ' (at the peak)'))) for v in (90, 75, 60, 45, 30, 15, 0)]
    lags = [(v, t + (lec if v == '30' else '')) for v, t in lags]
    thr = lambda i: slider(i, s('Pragul alarmei c (curs: 0,2)', 'Alarm threshold c (lecture: 0.2)'), 0.05, 0.6, 0.05, 0.2)  # noqa: E731
    ev = lambda p: (slider(f'{p}-c', s('Pragul alarmei c (curs: 0,2)', 'Alarm threshold c (lecture: 0.2)'), 0.05, 0.5, 0.05, 0.2)  # noqa: E731
                    + slider(f'{p}-x', s('Scăderea care definește un crah (curs: 20%)', 'Fall that defines a crash (lecture: 20%)'), 10, 40, 5, 20)
                    + select(f'{p}-h', s('Orizontul scăderii', 'Horizon of the fall'), [('182', s('182 de zile', '182 days') + lec), ('91', s('91 de zile', '91 days')), ('365', s('365 de zile', '365 days'))], '182')
                    + play(p))
    return {
     'episodes': (hor('ep-h') + play('ep'), CSS + charts('ep-chart')),
     'rational': (slider('ra-r', s('Rata r; AR(1) exploziv din dreapta are ρ = 1 + r (curs: 2%)', 'Rate r; the explosive AR(1) on the right has ρ = 1 + r (lecture: 2%)'), 0, 0.04, 0.0025, 0.02)
                  + slider('ra-p', s('Probabilitatea de supraviețuire π (curs: 0,98)', 'Survival probability π (lecture: 0.98)'), 0.9, 0.99, 0.01, 0.98) + play('ra'), charts('ra-chart')),
     'psy-ndx': (mlen('pn-l', 7) + lev('pn-c') + play('pn'), charts('pn-chart')),
     'psy-panel': (slider('pp-s', s('Partea din eșantion observată pînă acum (curs: 100%)', 'Share of the sample observed so far (lecture: 100%)'), 0.3, 1, 0.05, 1)
                   + mlen('pp-l', 6) + lev('pp-c') + play('pp'), charts('pp-chart')),
     'growth': (slider('gr-m', s('Exponentul m (curs: 0,5)', 'Exponent m (lecture: 0.5)'), 0.1, 0.9, 0.05, 0.5)
                + slider('gr-w', s('Frecvența ω a curbei LPPL (curs: 8)', 'Frequency ω of the LPPL curve (lecture: 8)'), 4, 16, 1, 8) + play('gr'), charts('gr-chart')),
     'components': (slider('cm-m', s('Exponentul m (curs: 0,5 în mijloc și în dreapta)', 'Exponent m (lecture: 0.5 in the middle and on the right)'), 0.1, 0.9, 0.05, 0.5)
                    + slider('cm-w', s('Frecvența ω a curbei din dreapta (curs: 8)', 'Frequency ω of the curve on the right (lecture: 8)'), 2, 20, 1, 8)
                    + slider('cm-c', s('Raportul C/|B| (curs: 0,08)', 'Ratio C/|B| (lecture: 0.08)'), 0, 0.2, 0.01, 0.08) + play('cm'), charts('cm-chart')),
     'cost': (slider('co-w', s('Limita superioară a lui ω în minimizare (curs: 25)', 'Upper limit of ω in the minimisation (lecture: 25)'), 3, 25, 2, 25)
              + slider('co-m', s('Limita superioară a lui m (curs: 0,99)', 'Upper limit of m (lecture: 0.99)'), 0.29, 0.99, 0.02, 0.99) + play('co'), charts('co-chart')),
     'fits': (select('fi-l', s('Sfîrșitul ferestrei: zile înainte de maximum', 'End of the window: days before the peak'), lags, '30') + play('fi'), charts('fi-chart')),
     'windows': (slider('wi-f', s('Limita condiției asupra lui t<sub>c</sub>: t<sub>2</sub> + f (t<sub>2</sub> − t<sub>1</sub>), f (curs: 0,2)', 'Bound of the t<sub>c</sub> condition: t<sub>2</sub> + f (t<sub>2</sub> − t<sub>1</sub>), f (lecture: 0.2)'), 0.05, 0.33, 0.01, 0.2)
                 + slider('wi-m', s('Limita superioară a lui m în filtru (curs: 0,99)', 'Upper limit of m in the filter (lecture: 0.99)'), 0.5, 0.99, 0.01, 0.99) + play('wi'), charts('wi-chart')),
     'ci': (thr('c1-c') + play('c1'), charts('c1-chart')),
     'ci-b': (thr('c2-c') + play('c2'), charts('c2-chart')),
     'tc-path': (slider('tp-k', s('Data de sfîrșit aleasă (a k-a dintre cele 38 Nasdaq / 39 Bitcoin)', 'Chosen end date (the k-th of the 38 Nasdaq / 39 Bitcoin dates)'), 1, 39, 1, 39) + play('tp'), charts('tp-chart')),
     'eval': (ev('e1'), charts('e1-chart')),
     'eval-btc': (ev('e2'), charts('e2-chart')),
     # ---------------------------------------------------------------- EXTRA
     't-ep': (hor('te-h') + play('te'),
              charts('te-chart') + box(select('te-f', s('Scăderea', 'The fall'), [('min', s('cea mai mică închidere din orizont', 'the lowest close within the horizon') + lec),
                                                                                  ('end', s('închiderea de la sfîrșitul orizontului', 'the close at the end of the horizon'))], 'min'), tbl('te-tb'))),
     'wx-bw': (slider('wb-r', s('Rata r (curs: 2%)', 'Rate r (lecture: 2%)'), 0, 0.05, 0.005, 0.02) + slider('wb-p', s('Probabilitatea de supraviețuire π (curs: 0,98)', 'Survival probability π (lecture: 0.98)'), 0.9, 0.99, 0.01, 0.98) + play('wb'),
               charts('wb-chart') + wx('wb-ws', W, r'\begin{gathered}E_t[B_{t+1}]=\pi\cdot\frac{1+r}{\pi}B_t+(1-\pi)\cdot 0=(1+r)B_t\\ E[\text{lifetime}]=\frac{1}{1-\pi},\qquad P(\text{survives }k\text{ periods})=\pi^k\end{gathered}',
                                       slider('wb-k', s('Numărul de perioade k din calcul', 'Number of periods k in the calculation'), 1, 100, 1, 34))),
     'wx-sadf': (slider('ws-e', s('Sfîrșitul r<sub>2</sub> al ferestrei (curs: întreg eșantionul)', 'End r<sub>2</sub> of the window (lecture: the whole sample)'), W0 - 1, NN - 1, 1, NN - 1)
                 + select('ws-m', s('Statistica', 'Statistic'), [('s', s('SADF: începutul fixat', 'SADF: start fixed')), ('g', s('GSADF: începutul se mișcă', 'GSADF: start moves'))], 's') + play('ws'),
                 charts('ws-chart') + wx('ws-ws', W, r'\begin{gathered}\mathrm{SADF}=\sup_{r_2\in[r_0,1]}\mathrm{ADF}_0^{r_2},\qquad \mathrm{GSADF}=\sup_{r_2\in[r_0,1]}\ \sup_{r_1\in[0,r_2-r_0]}\mathrm{ADF}_{r_1}^{r_2}\\ r_0=0.01+1.8/\sqrt T\end{gathered}')),
     'wx-bsadf': (slider('wd-e', s('Săptămîna curentă (datele pînă la ea)', 'Current week (the data up to it)'), I_A, I_B, 1, I_CONF) + lev('wd-c') + play('wd'),
                  charts('wd-chart') + wx('wd-ws', W, r'\begin{gathered}\mathrm{BSADF}(r_2)=\sup_{r_1\in[0,r_2-r_0]}\mathrm{ADF}_{r_1}^{r_2}\\ \text{episode: }\mathrm{BSADF}(r_2)>cv(r_2)\ \text{for at least }\log T\text{ observations}\end{gathered}',
                                          mlen('wd-l', 7))),
     'wx-adf': (slider('wa-e', s('Sfîrșitul ferestrei r<sub>2</sub> (curs: 24 martie 2000)', 'End of the window r<sub>2</sub> (lecture: 24 March 2000)'), W0 - 1, NN - 1, 1, I_PEAK) + play('wa'),
                charts('wa-chart') + wx('wa-ws', W, r'\Delta y_t=a+\delta\,y_{t-1}+\varepsilon_t,\qquad t=\frac{\hat\delta}{\mathrm{SE}(\hat\delta)},\qquad H_0:\ \delta=0\ \text{against}\ H_1:\ \delta>0',
                                        select('wa-s', s('Începutul ferestrei', 'Start of the window'), [('best', s('cel cu statistica cea mai mare', 'the one with the largest statistic') + lec),
                                                                                                         ('first', s('prima săptămînă (fereastra SADF)', 'the first week (the SADF window)')), ('short', s('fereastra minimă, 58 de săptămîni', 'the minimum window, 58 weeks'))], 'best'))),
     't-ndx': (lev('tn-c') + play('tn'), charts('tn-chart') + box(mlen('tn-l', 7), tbl('tn-tb') + tbl('tn-tb2'))),
     't-psy': (lev('ts-c') + play('ts'), charts('ts-chart') + box(mlen('ts-l', 6), tbl('ts-tb'))),
     'wx-lam': (slider('wl-w', s('Frecvența ω (curs: 8)', 'Frequency ω (lecture: 8)'), 2, 25, 0.5, 8) + play('wl'),
                charts('wl-chart') + wx('wl-ws', W, r'\frac{t_c-t_n}{t_c-t_{n+1}}=\lambda=e^{2\pi/\omega},\qquad t_c-t_{n+k}=\frac{t_c-t_n}{\lambda^{k}}',
                                        slider('wl-d', s('Prima corecție: zile înainte de t<sub>c</sub> (curs: 200)', 'First correction: days before t<sub>c</sub> (lecture: 200)'), 50, 400, 10, 200))),
     'wx-lppl': (slider('wp-m', s('Exponentul m (curs: 0,5)', 'Exponent m (lecture: 0.5)'), 0.1, 0.9, 0.05, 0.5) + slider('wp-w', s('Frecvența ω (curs: 8)', 'Frequency ω (lecture: 8)'), 2, 20, 1, 8) + play('wp'),
                 charts('wp-chart') + wx('wp-ws', W, r'\begin{gathered}\ln P(t)=A+B\,(t_c-t)^m+C\,(t_c-t)^m\cos\bigl(\omega\ln(t_c-t)-\phi\bigr)\\ A=8,\ B=-1.2,\ C=0.08,\ \phi=0,\ t_c=2\end{gathered}',
                                         select('wp-t', s('Momentul t', 'Time t'), [('1', 't = 1' + lec), ('1.75', 't = 1,75' if ro else 't = 1.75'), ('1.99', 't = 1,99' if ro else 't = 1.99'), ('0', 't = 0')], '1'))),
     'wx-filter': (select('wf-e', s('Ajustarea (30 de zile înainte de maximum)', 'The fit (30 days before the peak)'), eps, 'ssec') + play('wf'),
                   charts('wf-chart') + wx('wf-ws', W, r'\begin{gathered}B<0,\quad 0.01\le m\le 0.99,\quad 2\le\omega\le 25,\quad t_2\le t_c\le t_2+0.2\,(t_2-t_1)\\ \frac{\omega}{\pi}\ln\frac{t_c-t_1}{t_c-t_2}\ge 2.5,\qquad \frac{m|B|}{\omega|C|}\ge 1\end{gathered}')),
     'wx-ssec': (slider('wx-s', s('t<sub>c</sub> minus estimarea (zile; curs: 0)', 't<sub>c</sub> minus the estimate (days; lecture: 0)'), -30, 60, 2, 0)
                 + slider_any('wx-m', s('Exponentul m (curs: estimarea 0,392)', 'Exponent m (lecture: the estimate 0.392)'), 0.2, 0.8, repr(F_SSEC['m']))
                 + slider_any('wx-w', s('Frecvența ω (curs: estimarea 5,50)', 'Frequency ω (lecture: the estimate 5.50)'), 2, 12, repr(F_SSEC['w'])) + play('wx'),
                 charts('wx-chart') + wx('wx-ws', W, r'\begin{gathered}\ln P(t)=A+B f_t+C_1 g_t+C_2 h_t,\qquad \tau=t_c-t\\ f_t=\tau^m,\quad g_t=\tau^m\cos(\omega\ln\tau),\quad h_t=\tau^m\sin(\omega\ln\tau)\end{gathered}',
                                         select('wx-t', s('Data calculului', 'Date of the calculation'), [('t2', s('sfîrșitul ferestrei t<sub>2</sub>', 'the end of the window t<sub>2</sub>') + lec),
                                                                                                          ('mid', s('mijlocul ferestrei', 'the middle of the window')), ('t1', s('începutul ferestrei t<sub>1</sub>', 'the start of the window t<sub>1</sub>'))], 't2'))),
     't-fits': (select('tf-l', s('Sfîrșitul ferestrei: zile înainte de maximum', 'End of the window: days before the peak'), lags, '30') + play('tf'),
                charts('tf-chart') + box(select('tf-v', s('Coloanele tabelului', 'Columns of the table'), [('p', s('estimările', 'the estimates') + lec), ('c', s('fiecare condiție a filtrului', 'each condition of the filter'))], 'p'), tbl('tf-tb'))),
     't-win': (slider('tw-L', s('Cea mai scurtă fereastră inclusă (observații; curs: 50)', 'Shortest window included (observations; lecture: 50)'), 50, 500, 25, 50) + play('tw'),
               charts('tw-chart') + box(select('tw-q', s('Cuantilele lui t<sub>c</sub> din tabel', 'Quantiles of t<sub>c</sub> in the table'), [('q', s('ajustările care trec de condițiile asupra parametrilor', 'the fits that pass the parameter conditions') + lec),
                                                                                                                                       ('all', s('toate ajustările', 'all the fits'))], 'q'), tbl('tw-tb'))),
     't-eval': (slider('tv-x', s('Scăderea care definește un crah (curs: 20%)', 'Fall that defines a crash (lecture: 20%)'), 10, 40, 5, 20) + play('tv'),
                charts('tv-chart') + box(select('tv-v', s('Versiunea indicatorului', 'Version of the indicator'), [('param', s('condițiile asupra parametrilor', 'parameter conditions') + lec), ('full', s('toate condițiile', 'all conditions'))], 'param'), tbl('tv-tb'))),
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
    """Chart blocks for build_web_lecture.py: each host carries the web version of its EXTRA blocks (moved by ch13.js)."""
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
        img = f'ch13x_{anchor}_{lang}.jpg'
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
const sapt = n => `${n}${de(n)} ${n === 1 ? 'săptămînă' : 'săptămîni'}`;
const nep = n => (n === 0 ? 'niciun episod' : n === 1 ? 'un episod' : `${n}${de(n)} episoade`);
const COND = { B: 'B &lt; 0', m: '0,01 ≤ m ≤ 0,99', w: '2 ≤ ω ≤ 25', tc: 'condiția asupra lui t<sub>c</sub>', osc: 'cel puțin 2,5 oscilații', damping: 'amortizarea ≥ 1', rel_err: 'eroarea ≤ 15%' };
const EPN = { sp500: 'S&P 500, 2000', ndx: 'Nasdaq 100, 2000', bet: 'BET, 2007', ssec: 'Shanghai Composite, 2015', btc17: 'Bitcoin, 2017', btc21: 'Bitcoin, 2021' };
window.CH13_TEXT = {
 semi: '; ', big, yes: 'da', no: 'nu', pass: 'trece', fail: 'nu trece', na: 'nu se calculează', base: 'frecvența de bază', atBound: ' (limita atinsă)',
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 epName: k => EPN[k],
 ep1: (a, b) => `Creșterile de la minim la maximum: între ${a} (S&P 500) și ${b} (Bitcoin, 2017).`,
 ep2: (mo, a, b, l) => `Cea mai mică închidere în ${mo} luni după maximum, față de maximum: scăderi între ${a} și ${b}: ${l}.`,
 teHead: (mo, e) => ['Episodul', 'Minim', 'Maxim', 'Creștere', 'Ritm anual', e ? `Scădere la ${mo} luni` : `Scădere în ${mo} luni`, e ? 'Data' : 'Cea mai mică închidere'],
 teNote: e => `Ritmul anual: ln(P<sub>maxim</sub>/P<sub>minim</sub>) împărțit la durata creșterii în ani; scăderea: ${e ? 'închiderea de la sfîrșitul orizontului (ultima zi de tranzacționare din orizont)' : 'cea mai mică închidere din orizont'}, față de maximum. 12 luni = 365 de zile, ca în curs.`,
 te1: (x, y, s) => `Bitcoin și-a înmulțit prețul de ${x} ori în ${y} ani (2017); S&P 500 a crescut cu ${s}.`,
 te2: (mo, e, a, b) => `${e ? `Închiderea de la ${mo} luni după maximum` : `Cea mai mică închidere în ${mo} luni după maximum`}: scăderi între ${a} și ${b}.`,
 ra1: (mx, n, r, pi, g) => `Bula (r = ${r}, π = ${pi}) atinge ${mx} și se sparge de ${n} ${n === 1 ? 'dată' : 'ori'} după ce a depășit 4; cît supraviețuiește, crește cu factorul (1 + r)/π = ${g} pe perioadă.`,
 ra2: (neg, m) => neg ? `Cel mai mic preț al traiectoriei este ${m}: o bulă negativă duce prețul sub zero, imposibil pentru un activ pe care îl poți arunca fără cost (free disposal); de aceea o bulă rațională trebuie să fie nenegativă.` : `Cel mai mic preț al traiectoriei este ${m}.`,
 ra3: (rho, g, d, a, b, c) => `AR(1) cu ρ = ${rho}: un șoc se înmulțește cu ${g} în 100 de perioade și se dublează în ${d}${de(d)} perioade. După 200 de perioade: ρ = 0,9 dă ${a}, ρ = 1 dă ${b}, ρ = ${rho} dă ${c}.`,
 wbs1: (g, pi, g2) => `Media condiționată: π · (${g}/${pi}) · B<sub>t</sub> + (1 − π) · 0 = ${g2} · B<sub>t</sub>: bula crește în medie cu rata r.`,
 wbs2: (g, pi, x, r) => `Cît supraviețuiește: ${g}/${pi} − 1 = ${x} pe perioadă, mai mult decît r = ${r}; diferența plătește riscul spargerii.`,
 wbs3: (pi, d, h) => `Durata medie de viață 1/(1 − ${pi}) = ${d} perioade; supraviețuiește peste ${h} perioade cu probabilitatea 1/2 (π<sup>k</sup> = 0,5).`,
 wbs4: (k, pi, p, g, e) => `După k = ${k}: probabilitatea de supraviețuire ${pi}<sup>${k}</sup> = ${p}; o bulă care a supraviețuit este de ${g} ori mai mare, media tuturor traiectoriilor de ${e} ori.`,
 wb1: (g, d, h) => `Creștere cît supraviețuiește ${g} pe perioadă; durata medie ${d} perioade; jumătate dintre bule trăiesc peste ${h} perioade.`,
 pn1: (a, ca, s, cs, g, cg, lev) => `ADF pe tot eșantionul ${a} (valoarea critică de ${lev}%: ${ca}); SADF ${s} (${cs}); GSADF ${g} (${cg}).`,
 pn2: (n, L, a, b, len, f) => n ? `BSADF datează ${nep(n)} de cel puțin ${sapt(L)}; cel mai lung: ${a} – ${b} (${sapt(len)}); primul începe în ${f}.` : `BSADF nu datează niciun episod de cel puțin ${sapt(L)}.`,
 wss1: (g, a, b, n) => `Fereastra ${g ? 'cu statistica cea mai mare' : 'care începe în prima săptămînă'}: ${a} – ${b}, n = ${n}${de(n)} variații săptămînale.`,
 wss2: (d, s, t) => `OLS: δ̂ = ${d}, SE(δ̂) = ${s}, t = ${t}.`,
 wss3: (g, s) => `${g ? 'GSADF' : 'SADF'} al datelor pînă la această săptămînă (supremumul statisticilor de pînă acum): ${s}.`,
 wss4: (g, f, c, T, w0) => `Pe tot eșantionul (T = ${T}, fereastra minimă ${sapt(w0)}): ${g ? 'GSADF' : 'SADF'} = ${f} față de valoarea critică de 95% ${c}.`,
 ws1: (s, cs, g, cg, a, ca) => `SADF ${s} (valoarea critică de 95%: ${cs}) și GSADF ${g} (${cg}) resping rădăcina unitară; ADF pe tot eșantionul ${a} (${ca}) nu o respinge.`,
 wds1: (T, l, Ll, L) => `T = ${T}: log T = ${l}, rotunjit la ${Ll}; aici lungimea minimă este ${sapt(L)}.`,
 wds2: (d, b, c, lev, ab) => `Săptămîna ${d}: BSADF = ${b}, valoarea critică de ${lev}% = ${c}: ${ab ? 'peste' : 'sub'} ea.`,
 wds3: (r, f, L, c) => r ? `Peste valoarea critică de ${sapt(r)} la rînd, din ${f}; ${c ? `episodul este datat din ${c}, a ${L}-a săptămînă la rînd` : `încă nu ajunge la ${sapt(L)}`}.` : 'Statistica nu este peste valoarea critică în această săptămînă.',
 wd1: (n, d, L, lev) => `Cu datele pînă la ${d}: ${nep(n)} de cel puțin ${sapt(L)} (nivelul ${lev}%).`,
 was1: (a, b, n, m) => `Fereastra ${a} – ${b} (${m === 'best' ? 'începutul cu statistica cea mai mare' : m === 'first' ? 'începutul eșantionului' : 'fereastra minimă'}): n = ${n}${de(n)} variații săptămînale.`,
 was2: (a, d) => `OLS: Δŷ<sub>t</sub> = ${a} + ${d} · y<sub>t−1</sub>.`,
 was3: (d, s, t) => `t = δ̂ / SE(δ̂) = ${d} / ${s} = ${t}.`,
 was4: (r, t, c, ab) => `ρ̂ = 1 + δ̂ = ${r}; t = ${t} ${ab ? '>' : '≤'} ${c}, valoarea critică de 95% a BSADF la această dată: ${ab ? 'respingem rădăcina unitară în favoarea unei rădăcini explozive' : 'nu respingem rădăcina unitară'}.`,
 wa1: (e, b, s, c, ab) => `BSADF la ${e} = ${b} (cea mai mare statistică, din începutul ${s}); valoarea critică de 95% ${c}: ${ab ? 'fază explozivă' : 'nicio dovadă de fază explozivă'}.`,
 tnHead: ['Statistica', 'Valoarea', 'Critică 90%', 'Critică 95%', 'Critică 99%', 'Respinge la nivelul ales'],
 tnNote: (T, w0, r0, lev) => `Nasdaq 100, închideri săptămînale logaritmice, T = ${T}; r<sub>0</sub> = ${r0}, fereastra minimă ${sapt(w0)}; valori critice Monte Carlo (1000 de mersuri aleatoare cu driftul 1/T); nivelul ales: ${lev}%.`,
 tnHead2: ['Nr.', 'Începutul', 'Sfîrșitul', 'Săptămîni'],
 tnNote2: (L, Ll) => `Episoade datate: BSADF peste valoarea critică cel puțin ${sapt(L)} la rînd (în curs: ${Ll} = log T rotunjit).`,
 tn1: (n, L, lev) => `La ${lev}% și cel puțin ${sapt(L)}: ${nep(n)}.`,
 pp1: (s, l) => `Cu ${s} din fiecare eșantion: ${l}.`,
 pp1m: (k, d, n) => `${k} (date pînă la ${d}) ${nep(n)}`,
 pp2: (l, lev) => `GSADF pe tot eșantionul, față de valoarea critică de ${lev}%: ${l}.`,
 pp2m: (k, g, c) => `${k} ${g} (${c})`,
 tsHead: lev => ['Piața', 'Eșantionul', 'T (săptămîni)', 'SADF', `Critică ${lev}%`, 'GSADF', `Critică ${lev}%`, 'Episoade datate'],
 tsNote: (L, lev) => `Episoade de cel puțin ${sapt(L)} (în curs: 6 = log T rotunjit), valoarea critică de ${lev}%. Semnalul BET din septembrie 2008 – aprilie 2009 este prăbușirea: o scădere care se accelerează dă tot δ̂ > 0.`,
 ts1: (lev, l) => `Valorile critice de ${lev}% ale GSADF: ${l}.`,
 gr1: (m, a, b, c) => `Singularitatea cu m = ${m}: ritmul de creștere este ${a} la început, ${b} la jumătatea drumului și ${c} chiar înainte de t<sub>c</sub>; ritmul exponențial rămîne 0,8.`,
 gr2: (m, e) => `Ritmul −Bm(t<sub>c</sub> − t)<sup>m−1</sup> crește fără limită cînd t → t<sub>c</sub>, pentru că exponentul m − 1 = −${e} este negativ, în timp ce ln p(t<sub>c</sub>) = A rămîne finit.`,
 cm1: (w, l, a, b, c) => `ω = ${w}: λ = e<sup>2π/ω</sup> = ${l}; o corecție cu 200 de zile înainte de t<sub>c</sub> este urmată de altele cu aproximativ ${a}, ${b} și ${c} zile înainte.`,
 cm2: (m, c, d) => `Amortizarea m|B|/(ω|C|) a curbei din dreapta: ${m}/(ω · ${c}) = ${d}${+d.replace(',', '.') >= 1 ? ' ≥ 1: trece de condiție' : ' &lt; 1: nu trece de condiția de amortizare'}.`,
 wls1: (w, a, l) => `ω = ${w}: un ciclu complet cînd ln(t<sub>c</sub> − t) se schimbă cu 2π/ω = ${a}; λ = e<sup>${a}</sup> = ${l}.`,
 wls2: (d, l, a, b, c) => `Corecția de la ${d}${de(d)} zile înainte de t<sub>c</sub> este urmată de una la ${d}/${l} ≈ ${a}${de(a)} zile, apoi ${b}, apoi ${c}.`,
 wls3: (a, b) => `ω = 6,28 dă λ = ${a}; ω = 10 dă λ = ${b}; literatura empirică raportează λ în jur de 2.`,
 wl1: (w, l, n) => `ω = ${w}, λ = ${l}: ${n}${de(n)} maxime între 400 de zile și o zi înainte de t<sub>c</sub>.`,
 wps1: (t, d, m, f) => `t = ${t}: t<sub>c</sub> − t = ${d}; (t<sub>c</sub> − t)<sup>m</sup> = ${d}<sup>${m}</sup> = ${f}.`,
 wps2: (d, l, w, c) => `ln ${d} = ${l}; cos(${w} · ${l}) = ${c}.`,
 wps3: (f, c, l, p) => `ln P = 8 − 1,2 · ${f} + 0,08 · ${f} · ${c} = ${l}; P = e<sup>${l}</sup> = ${p}.`,
 wp1: (a, b, c, s) => `P crește de la ${a} (t = 1) la ${b} (t = 1,99) și ar atinge e<sup>A</sup> = ${c} la t<sub>c</sub>; ${s} din creșterea dintre t = 0 și t = 1,99 vine după t = 1,75.`,
 wfs1: (k, a, b, n, tc, td) => `${k}: fereastra ${a} – ${b} (${n}${de(n)} zile de tranzacționare); t̂<sub>c</sub> = ${td}.`,
 wfs2: (B, oB, m, om, w, ow) => `B = ${B}: ${oB}; m = ${m}: ${om}; ω = ${w}: ${ow}.`,
 wfs3: (p, ot, o, oo, d, od) => `(t<sub>c</sub> − t<sub>2</sub>)/(t<sub>2</sub> − t<sub>1</sub>) = ${p} (limita 0,2): ${ot}; oscilații în fereastră ${o} (cel puțin 2,5): ${oo}; amortizarea ${d}: ${od}.`,
 wfs4: (e, oe, l, ol, oa) => `Eroarea maximă ${e} (cel mult 15%): ${oe}; testul Lomb${l === null ? '' : `, valoarea p ${l}`}: ${ol}; reziduuri staționare (Dickey–Fuller și Phillips–Perron, nivel 10%): ${oa}.`,
 wfs5: (p, f) => `Condițiile asupra parametrilor: ${p ? 'trece' : 'nu trece'}; tot filtrul: ${f ? 'trece, o ajustare calificată' : 'nu trece'}.`,
 wf1: (l, n) => `Trec de condițiile asupra parametrilor: ${l || 'niciuna'}; de tot filtrul: ${n === 1 ? 'o singură ajustare' : n + ' ajustări'}.`,
 co1: (w, m, n, r) => `ω în [2, ${w}], m ≤ ${m}: profilul după t<sub>c</sub> are ${n} ${n === 1 ? 'minim local' : 'minime locale'}; cel mai prost și cel mai bun t<sub>c</sub> diferă cu ${r} în SSR.`,
 co2: (tc, m, p, g) => `Minimul global: t<sub>c</sub> = ${tc}, m = ${m}; maximul real ${p}; grila întreagă are ${g} de puncte.`,
 wxs1: (td, tc, m, w, A, B, C1, C2, C) => `t<sub>c</sub> = ${tc} (${td}), m = ${m}, ω = ${w}; pasul 1 (OLS) dă Â = ${A}, B̂ = ${B}, Ĉ<sub>1</sub> = ${C1}, Ĉ<sub>2</sub> = ${C2}, deci Ĉ = ${C}.`,
 wxs2: (d, t, a, f, l) => `La ${d} (t = ${t} ani): τ = t<sub>c</sub> − t = ${a}, τ<sup>m</sup> = ${f}, ln τ = ${l}.`,
 wxs3: (c, s, A, B, o, l, y) => `cos(ω ln τ) = ${c}, sin(ω ln τ) = ${s}; ln P̂ = ${A} + (${B}) + (${o}) = ${l}; observat ln P = ${y}.`,
 wxs4: (h, p, e) => `În niveluri: ajustat ${h} față de observat ${p}, o eroare de ${e}.`,
 wx1: (s, a, b, r) => `t<sub>c</sub> mutat cu ${s} zile: SSR = ${a}, față de ${b} la estimare (${r}).`,
 fi1: (l, x) => `Fereastra se încheie cu ${l}${de(l)} zile înainte de maximum. t̂<sub>c</sub> minus maximul real (zile): ${x}.`,
 fi2: (p, f) => `Trec de condițiile asupra parametrilor: ${p || 'niciuna'}; de tot filtrul: ${f || 'niciuna'}.`,
 tfHeadP: ['Episodul', 't<sub>2</sub>', 't̂<sub>c</sub>', 'Zile față de maxim', 'm̂', 'ω̂', 'Limita căutării t<sub>2</sub> + (t<sub>2</sub> − t<sub>1</sub>)/3', 'Param.', 'Toate'],
 tfNoteP: l => `Fereastra: de la minimul dinaintea creșterii pînă la ${l}${de(l)} zile calendaristice înainte de maximum (curs: 30). Spațiul de căutare: t<sub>c</sub> ∈ [t<sub>2</sub>, t<sub>2</sub> + (t<sub>2</sub> − t<sub>1</sub>)/3], m ∈ [0, 1], ω ∈ [1, 50]; Param.: trece de condițiile asupra parametrilor; Toate: și de cele asupra reziduurilor.`,
 tfHeadC: ['Episodul', 'B &lt; 0', 'm', 'ω', 't<sub>c</sub>', 'Osc.', 'Amort.', 'Eroare', 'Lomb', 'Staț.'],
 tfNoteC: '✓: trece; ✗: nu trece; –: testul Lomb și testul de staționaritate a reziduurilor se aplică doar ajustărilor care trec de toate celelalte condiții.',
 tf1: (l, b, m) => `La ${l}${de(l)} zile înainte de maximum, t̂<sub>c</sub> stă la limita superioară a căutării pentru: ${b || 'niciun episod'}; m̂ atinge limita m = 1 pentru: ${m || 'niciun episod'}.`,
 wi1: (f, m, a, b, c, d) => `Cu limita t<sub>2</sub> + ${f} (t<sub>2</sub> − t<sub>1</sub>) și m ≤ ${m}: Nasdaq 100 ${a} din 29 de ferestre trec de condițiile asupra parametrilor (${b} de tot filtrul); Shanghai ${c} din 29 (${d}).`,
 wi2: (a, b, c, d) => a ? `t̂<sub>c</sub> calificat la Nasdaq 100: mediana ${a}, intervalul 10%–90% ${b} – ${c}; maximul real 27 martie 2000. Shanghai: mediana ${d || '–'}, maximul real 12 iunie 2015.` : 'Nicio fereastră Nasdaq 100 nu este calificată.',
 twHead: a => ['Episodul', 'Sfîrșitul t<sub>2</sub>', 'Ferestre', 'Param.', 'Toate', 'Indicatorul', a ? 't̂<sub>c</sub> 10% (toate)' : 't̂<sub>c</sub> 10%', 'Mediana', '90%', 'Maximul real'],
 twNote: L => `Ferestrele [t<sub>2</sub> − L, t<sub>2</sub>] cu L ≥ ${L} (curs: toate cele 29, L = 750, 725, …, 50); indicatorul = ponderea ferestrelor care trec de condițiile asupra parametrilor.`,
 tw1: (L, s, n, k) => `Cu L ≥ ${L}: condiția 0,01 ≤ m ≤ 0,99 trece la ${s} din ferestrele Shanghai și la ${n} din cele Nasdaq 100; la Shanghai cade cel mai des ${COND[k]}.`,
 ci1: (k, a, b, c, d) => `${k}: versiunea pe parametri atinge ${a} la ${b}; versiunea completă ${c} la ${d}.`,
 ci2: (c, l) => `Date cu indicatorul ≥ ${c}: ${l}.`,
 ciFirst: d => `, prima la ${d}`,
 tp1: (k, e, tc, d, q) => `${k}, fereastra pînă la ${e}: t̂<sub>c</sub> = ${tc}, ${d >= 0 ? d + ' zile după' : -d + ' zile înainte de'} maximul real; ${q ? 'trece' : 'nu trece'} de condițiile asupra parametrilor.`,
 tp2: (n, a, b, s, x, nb, c, d, sb, md) => `Nasdaq 100: pe ${n}${de(n)} date de sfîrșit, t̂<sub>c</sub> variază de la ${a} la ${b}; trec ${s}; ${x}${de(x)} estimări stau la limita superioară a căutării. Bitcoin 2017 (${nb}${de(nb)} date): ${c} – ${d}; trec ${sb}; eroarea mediană ${md} de zile.`,
 ev1: (k, b, x, h, c, s, n, r) => `${k}: o scădere de cel puțin ${x} în ${h}${de(h)} zile urmează după ${b} dintre toate datele; cu pragul ${c}, alarmele acoperă ${s} dintre date (${n}), iar rata de reușită este ${r === null ? 'nedefinită (nicio alarmă)' : r}.`,
 ev2: (n, c, k) => `${n}${de(n)} grupuri de alarme (pauză de peste 90 de zile între ele); ${c} urmate de crah, din ${k} cu orizontul observat.`,
 ev3: (n, x, b, y) => `${n}${de(n)} scăderi de cel puțin ${x} de la un maxim; ${b} au avut o alarmă în orizontul dinaintea maximului${y ? ' (' + y + ')' : ''}.`,
 tvHead: ['Piața', 'Pragul', 'Ponderea datelor', 'Date', 'Rata de reușită'],
 tvNote: (x, c) => `Crah: o scădere de cel puțin ${x} în 182 de zile; versiunea ${c === 'param' ? 'pe parametri' : 'completă'} a indicatorului; primul rînd al fiecărei piețe: frecvența necondiționată.`,
 tv1: (x, a, b, c, d) => `Crah de ${x}: S&P 500 frecvența de bază ${a}, după o alarmă (c = 0,2) ${b}; Bitcoin ${c}, respectiv ${d}.`,
 tv2: (n1, c1, n2, c2, f1, b1, f2, b2) => `Grupuri de alarme (c = 0,2): S&P 500 ${n1}, dintre care ${c1} urmate de crah; Bitcoin ${n2}, dintre care ${c2}. Scăderi: S&P 500 ${f1}, ${b1} cu alarmă înainte de maximum; Bitcoin ${f2}, ${b2} cu alarmă înainte.`,
};
})();
""",
 'en': r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const big = n => String(Math.round(n)).replace('-', '−').replace(/\B(?=(\d{3})+(?!\d))/g, ',');
const wk = n => `${n} ${n === 1 ? 'week' : 'weeks'}`;
const nep = n => (n === 0 ? 'no episode' : n === 1 ? 'one episode' : `${n} episodes`);
const COND = { B: 'B &lt; 0', m: '0.01 ≤ m ≤ 0.99', w: '2 ≤ ω ≤ 25', tc: 'the condition on t<sub>c</sub>', osc: 'at least 2.5 oscillations', damping: 'damping ≥ 1', rel_err: 'error ≤ 15%' };
const EPN = { sp500: 'S&P 500, 2000', ndx: 'Nasdaq 100, 2000', bet: 'BET, 2007', ssec: 'Shanghai Composite, 2015', btc17: 'Bitcoin, 2017', btc21: 'Bitcoin, 2021' };
window.CH13_TEXT = {
 semi: '; ', big, yes: 'yes', no: 'no', pass: 'passes', fail: 'fails', na: 'not computed', base: 'base rate', atBound: ' (bound reached)',
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 epName: k => EPN[k],
 ep1: (a, b) => `Run-ups from the low to the peak: between ${a} (S&P 500) and ${b} (Bitcoin, 2017).`,
 ep2: (mo, a, b, l) => `Lowest close within ${mo} months after the peak, relative to the peak: falls between ${a} and ${b}: ${l}.`,
 teHead: (mo, e) => ['Episode', 'Low', 'Peak', 'Rise', 'Growth p.a.', e ? `Fall at ${mo} months` : `Fall within ${mo} months`, e ? 'Date' : 'Lowest close'],
 teNote: e => `Growth p.a.: ln(P<sub>peak</sub>/P<sub>low</sub>) divided by the length of the run-up in years; fall: ${e ? 'the close at the end of the horizon (the last trading day within it)' : 'the lowest close within the horizon'}, relative to the peak. 12 months = 365 days, as in the lecture.`,
 te1: (x, y, s) => `Bitcoin multiplied its price ${x} times in ${y} years (2017); the S&P 500 rose by ${s}.`,
 te2: (mo, e, a, b) => `${e ? `The close ${mo} months after the peak` : `The lowest close within ${mo} months after the peak`}: falls between ${a} and ${b}.`,
 ra1: (mx, n, r, pi, g) => `The bubble (r = ${r}, π = ${pi}) reaches ${mx} and bursts ${n === 1 ? 'once' : n === 2 ? 'twice' : n + ' times'} after exceeding 4; while it survives it grows by the factor (1 + r)/π = ${g} per period.`,
 ra2: (neg, m) => neg ? `The lowest price of the path is ${m}: a negative bubble takes the price below zero, impossible for an asset that can be thrown away at no cost (free disposal); so a rational bubble must be non-negative.` : `The lowest price of the path is ${m}.`,
 ra3: (rho, g, d, a, b, c) => `AR(1) with ρ = ${rho}: a shock is multiplied by ${g} in 100 periods and doubles in ${d} periods. After 200 periods: ρ = 0.9 gives ${a}, ρ = 1 gives ${b}, ρ = ${rho} gives ${c}.`,
 wbs1: (g, pi, g2) => `Conditional mean: π · (${g}/${pi}) · B<sub>t</sub> + (1 − π) · 0 = ${g2} · B<sub>t</sub>: on average the bubble grows at the rate r.`,
 wbs2: (g, pi, x, r) => `While it survives: ${g}/${pi} − 1 = ${x} per period, more than r = ${r}; the difference pays for the risk of bursting.`,
 wbs3: (pi, d, h) => `Expected lifetime 1/(1 − ${pi}) = ${d} periods; it survives beyond ${h} periods with probability 1/2 (π<sup>k</sup> = 0.5).`,
 wbs4: (k, pi, p, g, e) => `After k = ${k}: survival probability ${pi}<sup>${k}</sup> = ${p}; a bubble that survived is ${g} times larger, the mean over all paths ${e} times.`,
 wb1: (g, d, h) => `Growth while it survives ${g} per period; expected lifetime ${d} periods; half of the bubbles live beyond ${h} periods.`,
 pn1: (a, ca, s, cs, g, cg, lev) => `ADF on the whole sample ${a} (${lev}% critical value: ${ca}); SADF ${s} (${cs}); GSADF ${g} (${cg}).`,
 pn2: (n, L, a, b, len, f) => n ? `BSADF dates ${nep(n)} of at least ${wk(L)}; the longest: ${a} – ${b} (${wk(len)}); the first starts in ${f}.` : `BSADF dates no episode of at least ${wk(L)}.`,
 wss1: (g, a, b, n) => `The window ${g ? 'with the largest statistic' : 'starting in the first week'}: ${a} – ${b}, n = ${n} weekly changes.`,
 wss2: (d, s, t) => `OLS: δ̂ = ${d}, SE(δ̂) = ${s}, t = ${t}.`,
 wss3: (g, s) => `${g ? 'GSADF' : 'SADF'} of the data up to this week (the supremum of the statistics so far): ${s}.`,
 wss4: (g, f, c, T, w0) => `On the whole sample (T = ${T}, minimum window ${wk(w0)}): ${g ? 'GSADF' : 'SADF'} = ${f} against the 95% critical value ${c}.`,
 ws1: (s, cs, g, cg, a, ca) => `SADF ${s} (95% critical value: ${cs}) and GSADF ${g} (${cg}) reject the unit root; the whole-sample ADF ${a} (${ca}) does not.`,
 wds1: (T, l, Ll, L) => `T = ${T}: log T = ${l}, rounded to ${Ll}; here the minimum length is ${wk(L)}.`,
 wds2: (d, b, c, lev, ab) => `Week of ${d}: BSADF = ${b}, ${lev}% critical value = ${c}: ${ab ? 'above' : 'below'} it.`,
 wds3: (r, f, L, c) => r ? `Above the critical value for ${wk(r)} in a row, since ${f}; ${c ? `the episode is dated from ${c}, the ${L}th week in a row` : `not yet ${wk(L)}`}.` : 'The statistic is not above the critical value this week.',
 wd1: (n, d, L, lev) => `With the data up to ${d}: ${nep(n)} of at least ${wk(L)} (${lev}% level).`,
 was1: (a, b, n, m) => `Window ${a} – ${b} (${m === 'best' ? 'the start with the largest statistic' : m === 'first' ? 'the start of the sample' : 'the minimum window'}): n = ${n} weekly changes.`,
 was2: (a, d) => `OLS: Δŷ<sub>t</sub> = ${a} + ${d} · y<sub>t−1</sub>.`,
 was3: (d, s, t) => `t = δ̂ / SE(δ̂) = ${d} / ${s} = ${t}.`,
 was4: (r, t, c, ab) => `ρ̂ = 1 + δ̂ = ${r}; t = ${t} ${ab ? '>' : '≤'} ${c}, the 95% critical value of BSADF at this date: ${ab ? 'we reject the unit root in favour of an explosive root' : 'we do not reject the unit root'}.`,
 wa1: (e, b, s, c, ab) => `BSADF on ${e} = ${b} (the largest statistic, from the start ${s}); 95% critical value ${c}: ${ab ? 'an explosive phase' : 'no evidence of an explosive phase'}.`,
 tnHead: ['Statistic', 'Value', 'Critical 90%', 'Critical 95%', 'Critical 99%', 'Rejects at the chosen level'],
 tnNote: (T, w0, r0, lev) => `Nasdaq 100, weekly log closes, T = ${T}; r<sub>0</sub> = ${r0}, minimum window ${wk(w0)}; Monte Carlo critical values (1000 random walks with drift 1/T); chosen level: ${lev}%.`,
 tnHead2: ['No.', 'Start', 'End', 'Weeks'],
 tnNote2: (L, Ll) => `Dated episodes: BSADF above its critical value for at least ${wk(L)} in a row (lecture: ${Ll} = log T rounded).`,
 tn1: (n, L, lev) => `At ${lev}% and at least ${wk(L)}: ${nep(n)}.`,
 pp1: (s, l) => `With ${s} of each sample: ${l}.`,
 pp1m: (k, d, n) => `${k} (data up to ${d}) ${nep(n)}`,
 pp2: (l, lev) => `GSADF on the whole sample, against the ${lev}% critical value: ${l}.`,
 pp2m: (k, g, c) => `${k} ${g} (${c})`,
 tsHead: lev => ['Market', 'Sample', 'T (weeks)', 'SADF', `Critical ${lev}%`, 'GSADF', `Critical ${lev}%`, 'Dated episodes'],
 tsNote: (L, lev) => `Episodes of at least ${wk(L)} (lecture: 6 = log T rounded), ${lev}% critical value. The BET signal of September 2008 – April 2009 is the collapse: an accelerating fall also gives δ̂ > 0.`,
 ts1: (lev, l) => `${lev}% critical values of GSADF: ${l}.`,
 gr1: (m, a, b, c) => `The singularity with m = ${m}: the growth rate is ${a} at the start, ${b} halfway and ${c} just before t<sub>c</sub>; the exponential rate stays at 0.8.`,
 gr2: (m, e) => `The rate −Bm(t<sub>c</sub> − t)<sup>m−1</sup> grows without bound as t → t<sub>c</sub>, because the exponent m − 1 = −${e} is negative, while ln p(t<sub>c</sub>) = A stays finite.`,
 cm1: (w, l, a, b, c) => `ω = ${w}: λ = e<sup>2π/ω</sup> = ${l}; a correction 200 days before t<sub>c</sub> is followed by others about ${a}, ${b} and ${c} days before it.`,
 cm2: (m, c, d) => `Damping m|B|/(ω|C|) of the curve on the right: ${m}/(ω · ${c}) = ${d}${+d >= 1 ? ' ≥ 1: it passes the condition' : ' &lt; 1: it fails the damping condition'}.`,
 wls1: (w, a, l) => `ω = ${w}: one full cycle each time ln(t<sub>c</sub> − t) changes by 2π/ω = ${a}; λ = e<sup>${a}</sup> = ${l}.`,
 wls2: (d, l, a, b, c) => `The correction ${d} days before t<sub>c</sub> is followed by one at ${d}/${l} ≈ ${a} days, then ${b}, then ${c}.`,
 wls3: (a, b) => `ω = 6.28 gives λ = ${a}; ω = 10 gives λ = ${b}; the empirical literature reports λ around 2.`,
 wl1: (w, l, n) => `ω = ${w}, λ = ${l}: ${n} maxima between 400 days and one day before t<sub>c</sub>.`,
 wps1: (t, d, m, f) => `t = ${t}: t<sub>c</sub> − t = ${d}; (t<sub>c</sub> − t)<sup>m</sup> = ${d}<sup>${m}</sup> = ${f}.`,
 wps2: (d, l, w, c) => `ln ${d} = ${l}; cos(${w} · ${l}) = ${c}.`,
 wps3: (f, c, l, p) => `ln P = 8 − 1.2 · ${f} + 0.08 · ${f} · ${c} = ${l}; P = e<sup>${l}</sup> = ${p}.`,
 wp1: (a, b, c, s) => `P rises from ${a} (t = 1) to ${b} (t = 1.99) and would reach e<sup>A</sup> = ${c} at t<sub>c</sub>; ${s} of the rise between t = 0 and t = 1.99 comes after t = 1.75.`,
 wfs1: (k, a, b, n, tc, td) => `${k}: window ${a} – ${b} (${n} trading days); t̂<sub>c</sub> = ${td}.`,
 wfs2: (B, oB, m, om, w, ow) => `B = ${B}: ${oB}; m = ${m}: ${om}; ω = ${w}: ${ow}.`,
 wfs3: (p, ot, o, oo, d, od) => `(t<sub>c</sub> − t<sub>2</sub>)/(t<sub>2</sub> − t<sub>1</sub>) = ${p} (bound 0.2): ${ot}; oscillations in the window ${o} (at least 2.5): ${oo}; damping ${d}: ${od}.`,
 wfs4: (e, oe, l, ol, oa) => `Largest error ${e} (at most 15%): ${oe}; Lomb test${l === null ? '' : `, p-value ${l}`}: ${ol}; stationary residuals (Dickey–Fuller and Phillips–Perron, 10% level): ${oa}.`,
 wfs5: (p, f) => `Parameter conditions: ${p ? 'passes' : 'fails'}; the whole filter: ${f ? 'passes, a qualified fit' : 'fails'}.`,
 wf1: (l, n) => `Pass the parameter conditions: ${l || 'none'}; pass the whole filter: ${n === 1 ? 'one fit' : n + ' fits'}.`,
 co1: (w, m, n, r) => `ω in [2, ${w}], m ≤ ${m}: the profile over t<sub>c</sub> has ${n} local ${n === 1 ? 'minimum' : 'minima'}; the worst and the best t<sub>c</sub> differ by ${r} in SSR.`,
 co2: (tc, m, p, g) => `Global minimum: t<sub>c</sub> = ${tc}, m = ${m}; actual peak ${p}; the whole grid has ${g} points.`,
 wxs1: (td, tc, m, w, A, B, C1, C2, C) => `t<sub>c</sub> = ${tc} (${td}), m = ${m}, ω = ${w}; step 1 (OLS) gives Â = ${A}, B̂ = ${B}, Ĉ<sub>1</sub> = ${C1}, Ĉ<sub>2</sub> = ${C2}, so Ĉ = ${C}.`,
 wxs2: (d, t, a, f, l) => `On ${d} (t = ${t} years): τ = t<sub>c</sub> − t = ${a}, τ<sup>m</sup> = ${f}, ln τ = ${l}.`,
 wxs3: (c, s, A, B, o, l, y) => `cos(ω ln τ) = ${c}, sin(ω ln τ) = ${s}; ln P̂ = ${A} + (${B}) + (${o}) = ${l}; observed ln P = ${y}.`,
 wxs4: (h, p, e) => `In levels: fitted ${h} against observed ${p}, an error of ${e}.`,
 wx1: (s, a, b, r) => `t<sub>c</sub> moved by ${s} days: SSR = ${a}, against ${b} at the estimate (${r}).`,
 fi1: (l, x) => `The window ends ${l} days before the peak. Estimated t<sub>c</sub> minus the actual peak (days): ${x}.`,
 fi2: (p, f) => `Pass the parameter conditions: ${p || 'none'}; pass the whole filter: ${f || 'none'}.`,
 tfHeadP: ['Episode', 't<sub>2</sub>', 't̂<sub>c</sub>', 'Days vs peak', 'm̂', 'ω̂', 'Search bound t<sub>2</sub> + (t<sub>2</sub> − t<sub>1</sub>)/3', 'Param.', 'All'],
 tfNoteP: l => `Window: from the low before the run-up to ${l} calendar days before the peak (lecture: 30). Search space: t<sub>c</sub> ∈ [t<sub>2</sub>, t<sub>2</sub> + (t<sub>2</sub> − t<sub>1</sub>)/3], m ∈ [0, 1], ω ∈ [1, 50]; Param.: passes the parameter conditions; All: also the conditions on the residuals.`,
 tfHeadC: ['Episode', 'B &lt; 0', 'm', 'ω', 't<sub>c</sub>', 'Osc.', 'Damp.', 'Error', 'Lomb', 'Stat.'],
 tfNoteC: '✓: passes; ✗: fails; –: the Lomb test and the stationarity test of the residuals apply only to fits that pass all the other conditions.',
 tf1: (l, b, m) => `${l} days before the peak, the estimated t<sub>c</sub> sits at the upper bound of the search for: ${b || 'no episode'}; m̂ reaches the bound m = 1 for: ${m || 'no episode'}.`,
 wi1: (f, m, a, b, c, d) => `With the bound t<sub>2</sub> + ${f} (t<sub>2</sub> − t<sub>1</sub>) and m ≤ ${m}: Nasdaq 100 ${a} of 29 windows pass the parameter conditions (${b} the whole filter); Shanghai ${c} of 29 (${d}).`,
 wi2: (a, b, c, d) => a ? `Qualified t̂<sub>c</sub> of the Nasdaq 100: median ${a}, 10%–90% range ${b} – ${c}; actual peak 27 March 2000. Shanghai: median ${d || '–'}, actual peak 12 June 2015.` : 'No Nasdaq 100 window is qualified.',
 twHead: a => ['Episode', 'End t<sub>2</sub>', 'Windows', 'Param.', 'All', 'Indicator', a ? 't̂<sub>c</sub> 10% (all fits)' : 't̂<sub>c</sub> 10%', 'Median', '90%', 'Actual peak'],
 twNote: L => `Windows [t<sub>2</sub> − L, t<sub>2</sub>] with L ≥ ${L} (lecture: all 29, L = 750, 725, …, 50); indicator = share of the windows that pass the parameter conditions.`,
 tw1: (L, s, n, k) => `With L ≥ ${L}: the condition 0.01 ≤ m ≤ 0.99 holds in ${s} of the Shanghai windows and ${n} of the Nasdaq 100 ones; for Shanghai the condition that fails most often is ${COND[k]}.`,
 ci1: (k, a, b, c, d) => `${k}: the parameter version reaches ${a} on ${b}; the full version ${c} on ${d}.`,
 ci2: (c, l) => `Dates with the indicator ≥ ${c}: ${l}.`,
 ciFirst: d => `, the first on ${d}`,
 tp1: (k, e, tc, d, q) => `${k}, window up to ${e}: estimated t<sub>c</sub> = ${tc}, ${d >= 0 ? d + ' days after' : -d + ' days before'} the actual peak; it ${q ? 'passes' : 'fails'} the parameter conditions.`,
 tp2: (n, a, b, s, x, nb, c, d, sb, md) => `Nasdaq 100: over ${n} end dates, the estimated t<sub>c</sub> ranges from ${a} to ${b}; ${s} pass; ${x} estimates sit at the upper bound of the search. Bitcoin 2017 (${nb} dates): ${c} – ${d}; ${sb} pass; median error ${md} days.`,
 ev1: (k, b, x, h, c, s, n, r) => `${k}: a fall of at least ${x} within ${h} days follows ${b} of all dates; with the threshold ${c}, alarms cover ${s} of the dates (${n}), and the hit rate is ${r === null ? 'undefined (no alarm)' : r}.`,
 ev2: (n, c, k) => `${n} alarm clusters (gaps of more than 90 days between them); ${c} followed by a crash, out of ${k} with the horizon observed.`,
 ev3: (n, x, b, y) => `${n} falls of at least ${x} from a peak; ${b} had an alarm within the horizon before the peak${y ? ' (' + y + ')' : ''}.`,
 tvHead: ['Market', 'Threshold', 'Share of dates', 'Dates', 'Hit rate'],
 tvNote: (x, c) => `Crash: a fall of at least ${x} within 182 days; ${c === 'param' ? 'parameter' : 'full'} version of the indicator; first row of each market: the unconditional frequency.`,
 tv1: (x, a, b, c, d) => `Crash of ${x}: S&P 500 base rate ${a}, after an alarm (c = 0.2) ${b}; Bitcoin ${c} and ${d}.`,
 tv2: (n1, c1, n2, c2, f1, b1, f2, b2) => `Alarm clusters (c = 0.2): S&P 500 ${n1}, of which ${c1} followed by a crash; Bitcoin ${n2}, of which ${c2}. Falls: S&P 500 ${f1}, ${b1} with an alarm before the peak; Bitcoin ${f2}, ${b2} with an alarm before.`,
};
})();
""",
}

QA = {
 'ro': {
  'episodes': ('Puneți orizontul la 3 luni. Care episod are cea mai mare scădere în primele 3 luni după maximum?',
               'Bitcoin, 2017: cea mai mică închidere din primele 91 de zile este cu 64% sub maximul din 16 decembrie 2017; urmează Bitcoin 2021 (50%), Shanghai (43%) și Nasdaq 100 (36%), iar S&P 500 scade doar cu 11%. La 12 luni ordinea rămîne aproape aceeași, dar S&P 500 ajunge la 27% și Nasdaq 100 la 66%: o parte din cădere vine încet.'),
  't-ep': ('Alegeți „închiderea de la sfîrșitul orizontului” și 18 luni. Cît de jos este atunci Shanghai Composite față de maximul din 12 iunie 2015?',
           'Cu 37% sub maximum, mai puțin decît scăderea maximă de 49% din aceleași 18 luni: după minimul din ianuarie 2016 prețul a revenit parțial. Cele două măsuri răspund la întrebări diferite: cît de mare a fost căderea și unde a rămas prețul.'),
  'wx-bw': ('Puneți π = 0,90 (cu r = 2%). Cu cît crește bula într-o perioadă în care supraviețuiește?',
            'Cu 1,02/0,90 − 1 = 13,3% pe perioadă, față de 4,08% pentru π = 0,98: cu cît spargerea este mai probabilă, cu atît creșterea pe perioadele de supraviețuire trebuie să fie mai mare, ca media să rămînă 1 + r. Durata medie de viață scade la 1/(1 − 0,90) = 10 perioade.'),
  'rational': ('▶ crește r. Cu r = 4%, AR(1) exploziv are ρ = 1,04. De cîte ori crește un șoc după 100 de perioade?',
               'De 1,04¹⁰⁰ = 50,5 ori, față de 1,02¹⁰⁰ = 7,24: factorul crește exponențial cu numărul de perioade, deci dublarea ratei îl ridică la puterea a doua (7,24² ≈ 52, aproape de 50,5). Un șoc se dublează acum în 17,7 perioade, nu în 35.'),
  'wx-sadf': ('▶ mută sfîrșitul r₂. Cînd depășește statistica ADF pe fereastra [0, r₂] prima dată valoarea critică de 95% a SADF (1,48)?',
              'În săptămîna 8 ianuarie 1999, cu peste un an înainte de maximul din martie 2000; SADF ajunge la 2,72. Testul ADF pe tot eșantionul (−1,27) nu respinge, pentru că prăbușirea din 2000–2002 trage statistica întregului eșantion în jos; supremumul pe ferestre evită această problemă.'),
  'wx-bsadf': ('Episodul din 1998 începe pe 23 octombrie 1998. În ce săptămînă devine el un episod datat, cu 7 săptămîni peste valoarea critică?',
               'Pe 4 decembrie 1998, a șaptea săptămînă la rînd peste valoarea critică: în timp real, regula lungimii minime confirmă un episod abia după log T ≈ 7 săptămîni, deși data lui de început rămîne 23 octombrie 1998. Sfîrșitul episodului (3 noiembrie 2000) se cunoaște și mai tîrziu.'),
  'wx-adf': ('Mutați sfîrșitul ferestrei la 29 decembrie 2000, după prăbușire. Este BSADF încă peste valoarea critică de 95%?',
             'Nu: BSADF este −0,20, sub valoarea critică 0,53; cea mai mare statistică provine acum din începutul eșantionului (5 ianuarie 1990). Odată ce fereastra include scăderea, prețurile mari nu mai sînt urmate de creșteri mari, iar δ̂ devine negativ pentru orice început.'),
  'psy-ndx': ('▶ crește lungimea minimă a unui episod. La 20 de săptămîni, cîte episoade rămîn?',
              '3: cele din septembrie 1996 – martie 1997 (26 de săptămîni), mai – decembrie 1997 (32) și octombrie 1998 – noiembrie 2000 (107). Episoadele scurte din 1995 și 1998 dispar; cu lungimea 1 BSADF ar data 17 episoade, multe de o săptămînă.'),
  't-ndx': ('Treceți la nivelul de 99%. Cîte episoade de cel puțin 7 săptămîni datează BSADF?',
            '3: iulie – septembrie 1997, noiembrie 1998 – iulie 1999 și august 1999 – aprilie 2000. Cu valori critice mai mari, episodul lung din 1998–2000 se rupe în două și se termină în aprilie 2000, la o lună după maximum, nu în noiembrie 2000. SADF (2,72) și GSADF (2,92) rămîn peste valorile critice de 99% (1,92 și 2,86).'),
  'psy-panel': ('Cu primele 60% din eșantionul Shanghai (date pînă la 22 mai 2015, înainte de maximul din 12 iunie), este episodul 2014–2015 deja datat?',
                'Da: BSADF este peste valoarea critică din 28 noiembrie 2014, de 26 de săptămîni la rînd, deci episodul este datat și încă în desfășurare. BSADF folosește doar datele pînă la fiecare dată, deci datarea începutului este posibilă în timp real; sfîrșitul (26 iunie 2015) apare doar după maximum.'),
  't-psy': ('Treceți la 99%. Respinge GSADF rădăcina unitară pentru toate cele trei piețe?',
            'Da: Bitcoin 3,85 față de 2,63, BET 3,53 față de 2,86, Shanghai 3,46 față de 2,60. Episoadele datate se schimbă totuși: la BET rămîn cinci episoade scurte, iar la Bitcoin doar faza din 2017, ruptă în două.'),
  'growth': ('▶ schimbă m. Cu m = 0,2, cît este ritmul de creștere al singularității chiar înainte de t<sub>c</sub>?',
             '5,52, față de 3,98 pentru m = 0,5, deși la început ritmul este doar 0,20 (față de 0,50): un m mic ține creșterea plată multă vreme și o concentrează aproape de t<sub>c</sub>. Ritmul exponențial rămîne 0,8 pe tot drumul.'),
  'wx-lam': ('Puneți ω = 4. Cu ce factor λ se micșorează timpul rămas pînă la t<sub>c</sub> de la o corecție la următoarea?',
             'λ = e<sup>2π/4</sup> = 4,81: o corecție cu 200 de zile înainte de t<sub>c</sub> este urmată de una cu 42 de zile înainte, apoi cu 9. Un ω mic dă cicluri puține și rare; pentru ω = 8 factorul este 2,19.'),
  'components': ('Trece curba din dreapta (m = 0,5, ω = 8, C/|B| = 0,08) de condiția de amortizare m|B|/(ω|C|) ≥ 1 a filtrului?',
                 'Nu: 0,5/(8 · 0,08) = 0,78 &lt; 1. Deși |C| este mult mai mic decît |B|, termenul oscilant se înmulțește cu ω în derivată, iar cu ω = 8 oscilațiile ar face rata de hazard negativă în unele momente; condiția trece doar pentru C/|B| ≤ 0,0625 la m = 0,5 și ω = 8.'),
  'wx-lppl': ('Puneți m = 0,3 (restul ca în exemplu). Cît este P la t = 1,99?',
              'Aproximativ 2234, față de 2658 pentru m = 0,5: un m mai mic lasă o parte mai mare din creștere pentru ultimele momente dinaintea lui t<sub>c</sub>, unde prețul atinge e<sup>A</sup> = 2981 pentru orice m. La t = 1 prețul este 973 pentru orice m, pentru că acolo t<sub>c</sub> − t = 1.'),
  'wx-filter': ('▶ trece prin cele șase ajustări. Cîte au ω în intervalul filtrului, 2 ≤ ω ≤ 25?',
                '4: S&P 500 și BET se opresc la ω = 1,0, marginea de jos a spațiului de căutare [1, 50]. Spațiul de căutare este mai larg decît filtrul: o ajustare poate avea cel mai mic SSR la o valoare pe care filtrul o respinge.'),
  'cost': ('Limitați ω la intervalul [2, 5]. Unde este minimul global al lui t<sub>c</sub>?',
           'La 21 iunie 2000, tot după maximul real din 27 martie 2000, cu m = 0,99 și 4 minime locale. Doar la ω ≤ 3 minimul se mută la marginea grilei (25 februarie 2000), cu un raport SSR de 1,49; pentru ω ≤ 7 rezultatul este deja cel din curs (19 iulie 2000).'),
  'wx-ssec': ('Mutați t<sub>c</sub> cu 30 de zile mai tîrziu, cu m și ω fixate. Cu cît crește SSR?',
              'Cu aproximativ 55%. Cu m și ω fixate, SSR are o vale îngustă în jurul estimării (+11% la 10 zile); peisajul din „Peisajul funcției de cost” este plat pentru că acolo m și ω se reestimează pentru fiecare t<sub>c</sub> și compensează mutarea.'),
  'fits': ('Mutați sfîrșitul ferestrei la maximum (0 zile). Care ajustări trec de tot filtrul?',
           'Doar Shanghai, cu t̂<sub>c</sub> la 11 zile după maximum. S&P 500 și Nasdaq 100 rămîn la m = 1 și la limita superioară a căutării (190 și 178 de zile după maximum); nici datele pînă la maximum nu fac din creșterea lor una superexponențială.'),
  't-fits': ('La 90 de zile înainte de maximum, cîte dintre cele șase ajustări trec de condițiile asupra parametrilor?',
             'Una, Bitcoin 2017; Shanghai, singura calificată la 30 de zile, nu trece cu 90 de zile înainte. Calificarea unei ajustări depinde de data de sfîrșit a ferestrei, de aceea indicatorul folosește multe ferestre.'),
  'windows': ('Ridicați limita condiției asupra lui t<sub>c</sub> la 0,33, marginea spațiului de căutare. Cîte ferestre Nasdaq 100 trec de condițiile asupra parametrilor?',
              '12 din 29, față de 11 cu limita 0,2; la Shanghai 7 față de 5. Pentru Nasdaq 100 condiția care elimină cele mai multe ferestre este cea asupra lui m (doar 69% o îndeplinesc), nu cea asupra lui t<sub>c</sub>.'),
  't-win': ('Păstrați doar ferestrele de cel puțin 400 de observații. Ce pondere din ferestrele Shanghai trec de condiția asupra lui m?',
            '13% (2 din 15), față de 41% pe toate cele 29: în 13 dintre cele 15 ferestre lungi, ajustarea Shanghai pune m la o margine a spațiului de căutare (0,001 sau 0,999). La Nasdaq 100 toate ferestrele lungi trec de această condiție.'),
  'ci': ('Ridicați pragul alarmei la 0,3. Dă Shanghai vreo alarmă înainte de maximul din 12 iunie 2015?',
         'Nu: versiunea pe parametri atinge cel mult 0,28 (22 aprilie 2015). Cu pragul 0,2 apar 4 date cu alarmă înainte de maximum, din 15 aprilie 2015; pragul fixat dinainte decide dacă un semnal slab devine alarmă.'),
  'ci-b': ('Cu pragul 0,5, cînd apare prima alarmă Bitcoin înainte de maximul din 16 decembrie 2017?',
           'Pe 3 noiembrie 2017, cu șase săptămîni înainte de maximum (încă o dată pe 8 decembrie). Valoarea cea mai mare, 0,66, vine pe 29 decembrie 2017, după maximum: ea descrie creșterea deja încheiată.'),
  'tc-path': ('Cîte dintre cele 38 de estimări Nasdaq 100 ale lui t<sub>c</sub> stau la limita superioară a căutării, t₂ + (t₂ − t₁)/3?',
              '36 din 38: estimarea ar vrea un t<sub>c</sub> mai tîrziu decît permite căutarea, deci t̂<sub>c</sub> se mută odată cu t₂ doar pentru că limita se mută. La Bitcoin 2017 doar 3 din 39 ating limita; acolo 4 estimări stau exact pe dreapta t<sub>c</sub> = t₂, iar 11 la cel mult 20 de zile de ea.'),
  'eval': ('Ridicați pragul crahului la 30%. Cît devine frecvența necondiționată pentru S&P 500?',
           '2,3% dintre date, față de 6,8% pentru 20%; rata de reușită după alarme rămîne 0%. Frecvența de bază depinde de definiția crahului, de aceea definiția trebuie fixată înainte de a compara alarmele cu ea.'),
  'eval-btc': ('Prelungiți orizontul la 365 de zile. Cît devine frecvența de bază a unei scăderi de 20% la Bitcoin?',
               '51%, față de 47% la 182 de zile; după o alarmă (c = 0,2) rata este 55%. Scăderile de 20% sînt atît de frecvente la Bitcoin încît aproape orice regulă pare să le prindă: comparația relevantă este cu frecvența de bază.'),
  't-eval': ('Coborîți pragul crahului la 10%. Cum se compară rata de reușită a S&P 500 la c = 0,2 cu frecvența de bază?',
             'Rămîne sub ea: 16,9% față de 23,1%. Alarmele S&P 500 apar în creșteri calme și constante, după care scăderile, chiar și cele de 10%, sînt mai rare decît într-o zi oarecare.'),
 },
 'en': {
  'episodes': ('Set the horizon to 3 months. Which episode has the largest fall within the first 3 months after the peak?',
               'Bitcoin, 2017: the lowest close within the first 91 days is 64% below the peak of 16 December 2017; next come Bitcoin 2021 (50%), Shanghai (43%) and the Nasdaq 100 (36%), while the S&P 500 falls only 11%. At 12 months the order is almost the same, but the S&P 500 reaches 27% and the Nasdaq 100 66%: part of the fall comes slowly.'),
  't-ep': ('Choose “the close at the end of the horizon” and 18 months. How far below its peak of 12 June 2015 is the Shanghai Composite then?',
           '37% below the peak, less than the largest fall of 49% within the same 18 months: after the low of January 2016 the price recovered in part. The two measures answer different questions: how large the fall was and where the price stayed.'),
  'wx-bw': ('Set π = 0.90 (with r = 2%). By how much does the bubble grow in a period in which it survives?',
            'By 1.02/0.90 − 1 = 13.3% per period, against 4.08% for π = 0.98: the more likely a burst, the faster the bubble must grow while it survives, so that the mean stays at 1 + r. The expected lifetime falls to 1/(1 − 0.90) = 10 periods.'),
  'rational': ('▶ raises r. With r = 4%, the explosive AR(1) has ρ = 1.04. By how much is a shock multiplied after 100 periods?',
               'By 1.04¹⁰⁰ = 50.5, against 1.02¹⁰⁰ = 7.24: the factor grows exponentially with the number of periods, so doubling the rate roughly squares it (7.24² ≈ 52, close to 50.5). A shock now doubles in 17.7 periods instead of 35.'),
  'wx-sadf': ('▶ moves the end r₂. When does the ADF statistic over the window [0, r₂] first exceed the 95% critical value of SADF (1.48)?',
              'In the week of 8 January 1999, more than a year before the peak of March 2000; SADF reaches 2.72. The whole-sample ADF (−1.27) does not reject because the collapse of 2000–2002 pulls the statistic of the whole sample down; the supremum over windows avoids this.'),
  'wx-bsadf': ('The 1998 episode starts on 23 October 1998. In which week does it become a dated episode, with 7 weeks above the critical value?',
               'On 4 December 1998, the seventh week in a row above the critical value: in real time the minimum-length rule confirms an episode only after log T ≈ 7 weeks, although its start date remains 23 October 1998. The end of the episode (3 November 2000) is known even later.'),
  'wx-adf': ('Move the end of the window to 29 December 2000, after the collapse. Is BSADF still above its 95% critical value?',
             'No: BSADF is −0.20, below the critical value 0.53; the largest statistic now comes from the start of the sample (5 January 1990). Once the window contains the fall, high prices are no longer followed by large rises, and δ̂ turns negative for every start.'),
  'psy-ndx': ('▶ raises the minimum length of an episode. At 20 weeks, how many episodes remain?',
              '3: September 1996 – March 1997 (26 weeks), May – December 1997 (32) and October 1998 – November 2000 (107). The short episodes of 1995 and 1998 disappear; with length 1 BSADF would date 17 episodes, many of one week.'),
  't-ndx': ('Switch to the 99% level. How many episodes of at least 7 weeks does BSADF date?',
            '3: July – September 1997, November 1998 – July 1999 and August 1999 – April 2000. With higher critical values the long 1998–2000 episode splits in two and ends in April 2000, a month after the peak, not in November 2000. SADF (2.72) and GSADF (2.92) stay above their 99% critical values (1.92 and 2.86).'),
  'psy-panel': ('With the first 60% of the Shanghai sample (data up to 22 May 2015, before the peak of 12 June), is the 2014–2015 episode already dated?',
                'Yes: BSADF has been above its critical value since 28 November 2014, for 26 weeks in a row, so the episode is dated and still running. BSADF uses only the data up to each date, so dating the start is possible in real time; the end (26 June 2015) appears only after the peak.'),
  't-psy': ('Switch to 99%. Does GSADF reject the unit root for all three markets?',
            'Yes: Bitcoin 3.85 against 2.63, BET 3.53 against 2.86, Shanghai 3.46 against 2.60. The dated episodes change, though: BET keeps five short episodes, and Bitcoin only its 2017 phase, split in two.'),
  'growth': ('▶ changes m. With m = 0.2, what is the growth rate of the singularity just before t<sub>c</sub>?',
             '5.52, against 3.98 for m = 0.5, although at the start the rate is only 0.20 (against 0.50): a small m keeps the growth flat for a long time and concentrates it near t<sub>c</sub>. The exponential rate stays at 0.8 throughout.'),
  'wx-lam': ('Set ω = 4. By what factor λ does the time left to t<sub>c</sub> shrink from one correction to the next?',
             'λ = e<sup>2π/4</sup> = 4.81: a correction 200 days before t<sub>c</sub> is followed by one 42 days before it, then 9. A small ω gives few, sparse cycles; for ω = 8 the factor is 2.19.'),
  'components': ('Does the curve on the right (m = 0.5, ω = 8, C/|B| = 0.08) pass the damping condition m|B|/(ω|C|) ≥ 1 of the filter?',
                 'No: 0.5/(8 · 0.08) = 0.78 &lt; 1. Although |C| is much smaller than |B|, the oscillating term is multiplied by ω in the derivative, and with ω = 8 the oscillations would make the hazard rate negative at times; the condition holds only for C/|B| ≤ 0.0625 at m = 0.5 and ω = 8.'),
  'wx-lppl': ('Set m = 0.3 (the rest as in the example). What is P at t = 1.99?',
              'About 2234, against 2658 for m = 0.5: a smaller m leaves a larger part of the rise for the last moments before t<sub>c</sub>, where the price reaches e<sup>A</sup> = 2981 for any m. At t = 1 the price is 973 for any m, because there t<sub>c</sub> − t = 1.'),
  'wx-filter': ('▶ goes through the six fits. How many have ω inside the filter range 2 ≤ ω ≤ 25?',
                '4: the S&P 500 and BET stop at ω = 1.0, the lower edge of the search space [1, 50]. The search space is wider than the filter: a fit can have its smallest SSR at a value the filter rejects.'),
  'cost': ('Limit ω to [2, 5]. Where is the global minimum over t<sub>c</sub>?',
           'On 21 June 2000, still after the actual peak of 27 March 2000, with m = 0.99 and 4 local minima. Only for ω ≤ 3 does the minimum move to the edge of the grid (25 February 2000), with an SSR ratio of 1.49; for ω ≤ 7 the result is already the one of the lecture (19 July 2000).'),
  'wx-ssec': ('Move t<sub>c</sub> 30 days later, with m and ω fixed. By how much does the SSR rise?',
              'By about 55%. With m and ω fixed, the SSR has a narrow valley around the estimate (+11% at 10 days); the landscape of “The cost landscape” is flat because there m and ω are re-estimated for every t<sub>c</sub> and compensate for the move.'),
  'fits': ('Move the end of the window to the peak (0 days). Which fits pass the whole filter?',
           'Only Shanghai, with the estimated t<sub>c</sub> 11 days after the peak. The S&P 500 and the Nasdaq 100 stay at m = 1 and at the upper bound of the search (190 and 178 days after the peak); even the data up to the peak do not make their run-ups super-exponential.'),
  't-fits': ('90 days before the peak, how many of the six fits pass the parameter conditions?',
             'One, Bitcoin 2017; Shanghai, the only qualified fit at 30 days, fails 90 days before. Whether a fit qualifies depends on the end date of its window, which is why the indicator uses many windows.'),
  'windows': ('Raise the bound of the t<sub>c</sub> condition to 0.33, the edge of the search space. How many Nasdaq 100 windows pass the parameter conditions?',
              '12 of 29, against 11 with the bound 0.2; for Shanghai 7 against 5. For the Nasdaq 100 the condition that removes most windows is the one on m (only 69% meet it), not the one on t<sub>c</sub>.'),
  't-win': ('Keep only the windows of at least 400 observations. What share of the Shanghai windows pass the condition on m?',
            '13% (2 of 15), against 41% over all 29: in 13 of the 15 long windows the Shanghai fit puts m at an edge of the search space (0.001 or 0.999). For the Nasdaq 100 all the long windows pass this condition.'),
  'ci': ('Raise the alarm threshold to 0.3. Does Shanghai give any alarm before the peak of 12 June 2015?',
         'No: the parameter version reaches at most 0.28 (22 April 2015). With the threshold 0.2 there are 4 alarm dates before the peak, from 15 April 2015; the threshold fixed in advance decides whether a weak signal becomes an alarm.'),
  'ci-b': ('With the threshold 0.5, when is the first Bitcoin alarm before the peak of 16 December 2017?',
           'On 3 November 2017, six weeks before the peak (once more on 8 December). The largest value, 0.66, comes on 29 December 2017, after the peak: it describes the run-up that has already ended.'),
  'tc-path': ('How many of the 38 Nasdaq 100 estimates of t<sub>c</sub> sit at the upper bound of the search, t₂ + (t₂ − t₁)/3?',
              '36 of 38: the fit would like a later t<sub>c</sub> than the search allows, so the estimated t<sub>c</sub> moves with t₂ only because the bound moves. For Bitcoin 2017 only 3 of 39 reach the bound; there 4 estimates sit exactly on the line t<sub>c</sub> = t₂ and 11 within 20 days of it.'),
  'eval': ('Raise the crash size to 30%. What does the unconditional frequency become for the S&P 500?',
           '2.3% of the dates, against 6.8% for 20%; the hit rate after alarms stays at 0%. The base rate depends on the crash definition, which is why the definition must be fixed before the alarms are compared with it.'),
  'eval-btc': ('Lengthen the horizon to 365 days. What does the base rate of a 20% fall become for Bitcoin?',
               '51%, against 47% at 182 days; after an alarm (c = 0.2) the rate is 55%. Falls of 20% are so frequent for Bitcoin that almost any rule seems to catch them: the relevant comparison is with the base rate.'),
  't-eval': ('Lower the crash size to 10%. How does the S&P 500 hit rate at c = 0.2 compare with the base rate?',
             'It stays below it: 16.9% against 23.1%. S&P 500 alarms come in calm, steady run-ups, after which falls, even of 10%, are rarer than on an ordinary day.'),
 },
}

# Ce credeți? / What do you think?: (index of the correct option, [(RO, EN), ...]); keys checked against the page (node)
PREDICT = {
 'episodes': (0, [('Bitcoin, 2017.', 'Bitcoin, 2017.'), ('Nasdaq 100, 2000: cea mai mare creștere dintre acțiuni cade cel mai repede.', 'Nasdaq 100, 2000: the largest run-up among the stock indices falls fastest.'),
                  ('S&P 500, 2000: crahul unui indice larg vine imediat după maximum.', 'S&P 500, 2000: the crash of a broad index comes right after the peak.')]),
 't-ep': (0, [('Cu aproximativ 37%.', 'About 37%.'), ('Cu 49%, cît scăderea maximă: prețul nu își revine deloc.', '49%, as the largest fall: the price does not recover at all.'),
              ('Cu 0%: după 18 luni indicele revine la maximum.', '0%: after 18 months the index is back at its peak.')]),
 'wx-bw': (0, [('Cu 13,3%.', 'By 13.3%.'), ('Cu 2%, cît r: riscul spargerii nu schimbă creșterea.', 'By 2%, as r: the risk of bursting does not change the growth.'),
               ('Cu 11,1% = 1/0,90 − 1.', 'By 11.1% = 1/0.90 − 1.')]),
 'rational': (0, [('De aproximativ 50 de ori.', 'About 50 times.'), ('De aproximativ 14 ori, dublul lui 7,24: rata se dublează, deci și factorul.', 'About 14 times, twice 7.24: the rate doubles, so does the factor.'),
                  ('De 5 ori = 1 + 100 · 0,04.', '5 times = 1 + 100 · 0.04.')]),
 'wx-sadf': (0, [('La începutul lui 1999, cu peste un an înainte de maximum.', 'In early 1999, more than a year before the peak.'), ('Abia la maximul din martie 2000.', 'Only at the peak of March 2000.'),
                 ('Niciodată: ADF pe tot eșantionul este −1,27.', 'Never: the whole-sample ADF is −1.27.')]),
 'wx-bsadf': (0, [('Pe 4 decembrie 1998.', 'On 4 December 1998.'), ('Pe 23 octombrie 1998: prima săptămînă peste valoarea critică ajunge.', 'On 23 October 1998: the first week above the critical value is enough.'),
                  ('Pe 3 noiembrie 2000, cînd episodul se termină.', 'On 3 November 2000, when the episode ends.')]),
 'wx-adf': (0, [('Nu.', 'No.'), ('Da: o fereastră care începe în 1992 conține toată creșterea, deci rămîne explozivă.', 'Yes: a window starting in 1992 contains the whole run-up, so it stays explosive.'),
                ('Da: o scădere care se accelerează dă tot δ̂ > 0.', 'Yes: an accelerating fall also gives δ̂ > 0.')]),
 'psy-ndx': (0, [('3.', '3.'), ('6: lungimea minimă schimbă doar capetele episoadelor.', '6: the minimum length changes only the ends of the episodes.'),
                 ('1: doar bula din 1998–2000 este destul de lungă.', '1: only the 1998–2000 bubble is long enough.')]),
 't-ndx': (0, [('3.', '3.'), ('6, ca la 95%: nivelul schimbă doar testul pe tot eșantionul.', '6, as at 95%: the level changes only the whole-sample test.'),
               ('Niciunul: la 99% nicio săptămînă nu trece.', 'None: at 99% no week passes.')]),
 'psy-panel': (0, [('Da, datat din noiembrie 2014 și încă în desfășurare.', 'Yes, dated from November 2014 and still running.'), ('Nu: un episod se datează doar după ce se termină.', 'No: an episode is dated only after it ends.'),
                   ('Nu: valorile critice cer tot eșantionul.', 'No: the critical values need the whole sample.')]),
 't-psy': (0, [('Da, pentru toate trei.', 'Yes, for all three.'), ('Doar pentru Bitcoin: la 99% rămîne doar bula cea mai mare.', 'Only for Bitcoin: at 99% only the largest bubble remains.'),
               ('Pentru niciuna: 99% cere o statistică de peste 4.', 'For none: 99% needs a statistic above 4.')]),
 'growth': (0, [('Mai mare: aproximativ 5,5.', 'Higher: about 5.5.'), ('Mai mic: 0,2, cît m.', 'Lower: 0.2, as m.'), ('Același, 3,98: m schimbă doar nivelul.', 'The same, 3.98: m changes only the level.')]),
 'wx-lam': (0, [('4,81.', '4.81.'), ('1,57 = 2π/4.', '1.57 = 2π/4.'), ('2,19, ca pentru ω = 8: λ nu depinde de ω.', '2.19, as for ω = 8: λ does not depend on ω.')]),
 'components': (0, [('Nu: 0,78 < 1.', 'No: 0.78 < 1.'), ('Da: |C| < |B|, deci oscilația este mică.', 'Yes: |C| < |B|, so the oscillation is small.'),
                    ('Da: amortizarea privește doar exponentul m.', 'Yes: damping concerns only the exponent m.')]),
 'wx-lppl': (0, [('Aproximativ 2 234.', 'About 2,234.'), ('Aproximativ 2 900: un m mai mic urcă mai repede peste tot.', 'About 2,900: a smaller m rises faster everywhere.'),
                 ('2 658, ca pentru m = 0,5: m nu schimbă prețul.', '2,658, as for m = 0.5: m does not change the price.')]),
 'wx-filter': (0, [('4.', '4.'), ('6: căutarea are deja ω între 1 și 50.', '6: the search already keeps ω between 1 and 50.'), ('1: doar Shanghai, singura ajustare calificată.', '1: only Shanghai, the only qualified fit.')]),
 'cost': (0, [('Tot după maximum: 21 iunie 2000.', 'Still after the peak: 21 June 2000.'), ('La maximul real, 27 martie 2000: oscilațiile lente fixează timpul critic.', 'At the actual peak, 27 March 2000: slow oscillations pin the critical time down.'),
              ('La 25 februarie 2000, sfîrșitul ferestrei.', 'On 25 February 2000, the end of the window.')]),
 'wx-ssec': (0, [('Cu aproximativ 55%.', 'By about 55%.'), ('Cu sub 1%: peisajul funcției de cost este plat.', 'By less than 1%: the cost landscape is flat.'),
                 ('Scade: un timp critic mai tîrziu este mai aproape de maximul real din 12 iunie.', 'It falls: a later critical time is closer to the actual peak of 12 June.')]),
 'fits': (0, [('Doar Shanghai.', 'Only Shanghai.'), ('Toate șase: cu datele pînă la maximum, bula este completă.', 'All six: with the data up to the peak, the bubble is complete.'),
              ('Niciuna.', 'None.')]),
 't-fits': (0, [('Una: Bitcoin, 2017.', 'One: Bitcoin, 2017.'), ('Una: Shanghai Composite, ca la 30 de zile.', 'One: the Shanghai Composite, as at 30 days.'),
                ('Toate șase: cu mai mult timp pînă la maximum, timpul critic intră ușor în limita filtrului.', 'All six: with more time to the peak, the critical time easily fits the bound of the filter.')]),
 'windows': (0, [('12.', '12.'), ('29: fără limita pentru timpul critic, toate trec.', '29: without the bound on the critical time, all pass.'), ('11, ca înainte: limita pentru timpul critic nu contează.', '11, as before: the bound on the critical time does not matter.')]),
 't-win': (0, [('13%.', '13%.'), ('41%, ca pe toate ferestrele.', '41%, as over all the windows.'), ('100%, ca la Nasdaq 100: ferestrele lungi stabilizează m.', '100%, as for the Nasdaq 100: long windows stabilise m.')]),
 'ci': (0, [('Nu.', 'No.'), ('Da, pe 22 aprilie 2015.', 'Yes, on 22 April 2015.'), ('Da, cu un an și jumătate înainte.', 'Yes, a year and a half before.')]),
 'ci-b': (0, [('Pe 3 noiembrie 2017.', 'On 3 November 2017.'), ('Pe 29 decembrie 2017, cînd indicatorul atinge 0,66.', 'On 29 December 2017, when the indicator reaches 0.66.'),
              ('Niciodată: 0,5 nu este atins.', 'Never: 0.5 is not reached.')]),
 'tc-path': (0, [('36.', '36.'), ('Niciuna: estimările sînt libere în interiorul spațiului de căutare.', 'None: the estimates are free inside the search space.'), ('19, jumătate.', '19, half.')]),
 'eval': (0, [('Aproximativ 2,3%.', 'About 2.3%.'), ('6,8%: frecvența de bază nu depinde de definiția crahului.', '6.8%: the base rate does not depend on the crash definition.'),
              ('0%: o scădere de 30% în 182 de zile nu apare niciodată.', '0%: a 30% fall within 182 days never happens.')]),
 'eval-btc': (0, [('Aproximativ 51%.', 'About 51%.'), ('Aproximativ 94% = 2 · 47%: un orizont dublu dublează frecvența.', 'About 94% = 2 · 47%: a double horizon doubles the frequency.'),
                  ('47%: orizontul nu contează.', '47%: the horizon does not matter.')]),
 't-eval': (0, [('Sub ea: 17% față de 23%.', 'Below it: 17% against 23%.'), ('Peste ea: scăderile mici sînt prinse ușor.', 'Above it: small falls are easy to catch.'),
                ('Egală cu ea: alarmele nu spun nimic.', 'Equal to it: the alarms say nothing.')]),
}
VEIL_CHARTS = {'tc-path'}

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slides „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('Într-o bulă Blanchard–Watson cu r = 1% și π = 0,95, cu cît crește bula într-o perioadă în care supraviețuiește?',
         'In a Blanchard–Watson bubble with r = 1% and π = 0.95, by how much does the bubble grow in a period in which it survives?'), key=0, opts=[
  (('Cu aproximativ 6,3%.', 'By about 6.3%.'), ('1,01/0,95 − 1 ≈ 6,3%; durata medie de viață 1/(1 − 0,95) = 20 de perioade.', '1.01/0.95 − 1 ≈ 6.3%; expected lifetime 1/(1 − 0.95) = 20 periods.')),
  (('Cu 1%, cît r.', 'By 1%, as r.'), ('1% este creșterea medie, inclusiv perioadele în care bula se sparge; cît supraviețuiește, crește mai repede.', '1% is the mean growth, including the periods in which the bubble bursts; while it survives it grows faster.')),
  (('Cu 5,3% = 1/0,95 − 1.', 'By 5.3% = 1/0.95 − 1.'), ('Factorul este (1 + r)/π, nu 1/π: lipsește r.', 'The factor is (1 + r)/π, not 1/π: r is missing.'))]),
 dict(q=('Un ADF la dreapta pe o fereastră dă δ̂ = 0,004 cu SE 0,002. Este fereastra explozivă la 5%, dacă valoarea critică este 1,5?',
         'A right-tailed ADF on a window gives δ̂ = 0.004 with SE 0.002. Is the window explosive at 5% if the critical value is 1.5?'), key=1, opts=[
  (('Nu: δ̂ = 0,004 este practic zero.', 'No: δ̂ = 0.004 is practically zero.'), ('Decide statistica t, nu mărimea lui δ̂: t = 0,004/0,002 = 2.', 'The t statistic decides, not the size of δ̂: t = 0.004/0.002 = 2.')),
  (('Da: t = 2 > 1,5.', 'Yes: t = 2 > 1.5.'), ('Respingem rădăcina unitară în favoarea unei rădăcini explozive.', 'We reject the unit root in favour of an explosive root.')),
  (('Nu: testul Dickey–Fuller respinge doar pentru t mare negativ.', 'No: the Dickey–Fuller test rejects only for a large negative t.'), ('Aceea este coada din stînga din Capitolul 3; aici alternativa este δ > 0.', 'That is the left tail of Chapter 3; here the alternative is δ > 0.'))]),
 dict(q=('De ce folosim coada din dreapta și nu coada din stînga din Capitolul 3?', 'Why do we use the right tail and not the left tail of Chapter 3?'), key=0, opts=[
  (('Alternativa este ρ > 1 (exploziv), nu ρ < 1 (staționar).', 'The alternative is ρ > 1 (explosive), not ρ < 1 (stationary).'), ('Dovada pentru o rădăcină explozivă este un t pozitiv mare.', 'Evidence for an explosive root is a large positive t.')),
  (('Pentru că prețurile sînt pozitive.', 'Because prices are positive.'), ('Semnul prețului nu are legătură cu coada testului; contează alternativa.', 'The sign of the price has nothing to do with the tail of the test; the alternative does.')),
  (('Pentru că valorile critice Dickey–Fuller sînt pozitive.', 'Because the Dickey–Fuller critical values are positive.'), ('Valorile critice de stînga sînt negative; SADF și GSADF au propriile valori critice simulate.', 'The left-tail critical values are negative; SADF and GSADF have their own simulated critical values.'))]),
 dict(q=('O ajustare LPPL dă ω = 6,28. Cu ce factor se micșorează timpul rămas pînă la timpul critic între două corecții?',
         'An LPPL fit gives ω = 6.28. By what factor does the time left to the critical time shrink between two corrections?'), key=2, opts=[
  (('1,00 = 2π/6,28.', '1.00 = 2π/6.28.'), ('2π/ω este schimbarea logaritmului timpului rămas într-un ciclu; factorul este exponențiala ei.', '2π/ω is the change of the log of the time left over one cycle; the factor is its exponential.')),
  (('6,28, cît ω.', '6.28, as ω.'), ('ω este frecvența logaritmică, nu raportul de scală.', 'ω is the log-frequency, not the scaling ratio.')),
  (('λ = exp(2π/6,28) ≈ e ≈ 2,72.', 'λ = exp(2π/6.28) ≈ e ≈ 2.72.'), ('λ = exp(2π/ω); pentru ω = 6,28 exponentul este 1.', 'λ = exp(2π/ω); for ω = 6.28 the exponent is 1.'))]),
 dict(q=('O ajustare are m = 1,0, la marginea spațiului de căutare. Este o ajustare calificată de bulă?',
         'A fit has m = 1.0, at the edge of the search space. Is it a qualified bubble fit?'), key=1, opts=[
  (('Da: m este în intervalul de căutare [0, 1].', 'Yes: m is inside the search range [0, 1].'), ('Spațiul de căutare este mai larg decît filtrul, care cere m ≤ 0,99.', 'The search space is wider than the filter, which needs m ≤ 0.99.')),
  (('Nu: filtrul cere m ≤ 0,99; m = 1 este un trend liniar în t.', 'No: the filter needs m ≤ 0.99; m = 1 is a linear trend in t.'), ('Cu m = 1 creșterea nu se accelerează; S&P 500 și Nasdaq 100 sînt în această situație cu 30 de zile înainte de maximum.', 'With m = 1 the growth does not accelerate; the S&P 500 and the Nasdaq 100 are in this case 30 days before the peak.')),
  (('Da, dacă celelalte condiții trec.', 'Yes, if the other conditions pass.'), ('Fiecare condiție trebuie îndeplinită; m = 1 nu trece de condiția asupra lui m.', 'Every condition must hold; m = 1 fails the condition on m.'))]),
 dict(q=('Un indicator dă alarme înainte de 8 din 10 crahuri. Este un predictor bun?', 'An indicator gives alarms before 8 of 10 crashes. Is it a good predictor?'), key=2, opts=[
  (('Da: prinde 80% dintre crahuri.', 'Yes: it catches 80% of the crashes.'), ('Fără alarmele false nu știm cît costă aceste reușite: alarme aproape zilnice prind ușor 8 crahuri.', 'Without the false alarms we do not know the cost of these hits: near-daily alarms easily catch 8 crashes.')),
  (('Nu: un predictor bun prinde toate cele 10.', 'No: a good predictor catches all 10.'), ('Nici 10 din 10 nu ajung dacă alarmele sînt aproape permanente.', 'Even 10 of 10 is not enough if the alarms are almost always on.')),
  (('Încă nu: avem nevoie de alarmele false și de frecvența de bază.', 'Not yet: we also need the false alarms and the base rate.'), ('Comparația este P(crah | alarmă) față de P(crah).', 'The comparison is P(crash | alarm) against P(crash).'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {})) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch13-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] = TEXT_JS[lang].replace('\n};\n})();\n', '\n};\n') + ('window.CH13_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                                                                        + 'window.CH13_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n})();\n')


_predict_js()


if __name__ == '__main__':
    import interactive_common
    page_spec = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA', 'TEXT_JS', 'PREDICT', 'TICKET')})
    page_spec.widgets = lambda lang: {a: v for a, v in widgets_all(lang).items() if a in {b for b, _, _ in BLOCKS}}
    interactive_common.build(page_spec)
    for lg in ('en', 'ro'):
        _post(lg)
