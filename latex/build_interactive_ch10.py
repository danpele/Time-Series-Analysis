"""Build the interactive pages of Chapter 10 (state space models, the Kalman filter and Markov switching): interactive/ch10.html
(EN) and interactive/ch10_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its interactive version on
the right. The result tables of the lecture (the three periods by hand, the maximum-likelihood estimates of the Nile model,
the two maxima of Hamilton's MS-AR(4), the parameters, transition probabilities and durations of the chapter's
Markov-switching models), its worked examples (the AR(2) in state space form, one Kalman step, the steady-state gain, the
smoother step, the prediction-error decomposition, the HP filter as a smooth-trend model, the durations and the ergodic
probabilities, one step of the Hamilton filter) and one extra chart where the lecture gives only numbers (which regimes the
model finds) get blocks of their own (EXTRA), placed by slide; on the web lecture each of them travels inside a chart block and
ch10.js moves it to its own frame. Logic in interactive/ch10.js, inputs in interactive/ch10_data.js
(Quantlets/Ch_10/export_interactive.py), slide images in interactive/slides/.
"""
import json
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 10

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_10'
TEX = {'ro': 'RO/Cursuri/capitol10_spatiul_starilor_kalman_markov_switching', 'en': 'EN/Courses/chapter10_state_space_kalman_markov_switching'}

# (anchor, chart file, section key); order of the slides
BLOCKS = [('ss-examples', 'tsa_ch10_ss_examples', 'ssf'),
          ('nile-filter', 'tsa_ch10_nile_filter', 'kf'), ('gain-ses', 'tsa_ch10_gain_ses', 'kf'),
          ('nile-smooth', 'tsa_ch10_nile_smooth', 'sm'), ('likelihood', 'tsa_ch10_likelihood', 'sm'), ('diagnostics', 'tsa_ch10_diagnostics', 'sm'),
          ('missing', 'tsa_ch10_missing', 'sm'),
          ('ro-trend', 'tsa_ch10_ro_trend', 'tc'), ('output-gap', 'tsa_ch10_output_gap', 'tc'), ('realtime', 'tsa_ch10_realtime', 'tc'),
          ('tvp-beta', 'tsa_ch10_tvp_beta', 'tc'), ('dfm', 'tsa_ch10_dfm', 'tc'),
          ('ms-us', 'tsa_ch10_ms_us', 'ms'), ('ms-filtered', 'tsa_ch10_ms_filtered', 'ms'), ('ms-ro', 'tsa_ch10_ms_ro', 'ms'),
          ('vol-regimes', 'tsa_ch10_vol_regimes', 'ms'), ('regimes-memory', 'tsa_ch10_regimes_memory', 'ms')]

# result tables, worked examples and extra charts: (anchor, frame title EN, frame title RO, section key, host chart block)
EXTRA = [('wx-arma', 'Worked example: AR(2) for US GDP growth', 'Exemplu rezolvat: AR(2) pentru creșterea PIB din SUA', 'ssf', 'ss-examples'),
         ('wx-kf', 'Worked example: three periods by hand (1/2)', 'Exemplu rezolvat: trei perioade de mînă (1/2)', 'kf', 'nile-filter'),
         ('kf-table', 'Worked example: three periods by hand (2/2)', 'Exemplu rezolvat: trei perioade de mînă (2/2)', 'kf', 'nile-filter'),
         ('wx-ss', 'Steady state and exponential smoothing', 'Starea de echilibru și netezirea exponențială', 'kf', 'nile-filter'),
         ('nile-ml', 'Interpreting the Nile filter', 'Interpretarea filtrului pentru Nil', 'kf', 'nile-filter'),
         ('wx-rts', 'Smoothing with the full sample', 'Netezirea pe întregul eșantion', 'sm', 'nile-smooth'),
         ('wx-lik', 'The likelihood: prediction-error decomposition', 'Verosimilitatea: descompunerea erorilor de predicție', 'sm', 'likelihood'),
         ('wx-hp', 'The HP filter as a state space smoother', 'Filtrul HP ca netezitor în spațiul stărilor', 'tc', 'ro-trend'),
         ('wx-dur', 'Transition matrix, durations and ergodic probabilities', 'Matricea de tranziție, durate și probabilități ergodice', 'ms', 'ms-us'),
         ('wx-ham', 'The Hamilton filter', 'Filtrul Hamilton', 'ms', 'ms-us'),
         ('ms-table', "Hamilton's specification on today's data", 'Specificația lui Hamilton pe datele de azi', 'ms', 'ms-us'),
         ('ms-checks', 'Checks: the regimes identified by the model', 'Verificări: regimurile identificate de model', 'ms', 'ms-filtered'),
         ('ms-params', 'Recap: Markov switching', 'Recapitulare: modelele Markov switching', 'ms', 'regimes-memory')]

PFX = {'ss-examples': 'se', 'wx-arma': 'wa', 'wx-kf': 'wk', 'kf-table': 'kt', 'wx-ss': 'ws', 'nile-filter': 'nf', 'nile-ml': 'nm', 'gain-ses': 'gs',
       'wx-rts': 'wr', 'nile-smooth': 'ns', 'wx-lik': 'wl', 'likelihood': 'lk', 'diagnostics': 'dg', 'missing': 'mi', 'wx-hp': 'wh', 'ro-trend': 'rt',
       'output-gap': 'og', 'realtime': 'rl', 'tvp-beta': 'tv', 'dfm': 'df', 'wx-dur': 'wd', 'wx-ham': 'wf', 'ms-table': 'mt', 'ms-us': 'mu',
       'ms-filtered': 'mf', 'ms-checks': 'mc', 'ms-ro': 'mr', 'vol-regimes': 'vr', 'regimes-memory': 'rm', 'ms-params': 'mp'}

SECTIONS = {
 'ro': dict(ssf='Forma în spațiul stărilor', kf='Filtrul Kalman', sm='Netezire, verosimilitate și valori lipsă', tc='Trend, ciclu și factori comuni',
            ms='Modele Markov switching'),
 'en': dict(ssf='The state space form', kf='The Kalman filter', sm='Smoothing, likelihood and missing data', tc='Trend, cycle and common factors',
            ms='Markov switching'),
}

L = {
 'ro': dict(file='ch10_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 10 · Modele în spațiul stărilor, filtrul Kalman și modele Markov switching', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'casetele „Exemplu rezolvat” refac calculul cu cifrele capitolului.',
   src='Debitul anual al Nilului la Aswan, 1871–1970 (setul de date din statsmodels). Prețurile săptămînale S&P 500, BET și Euro Stoxx 50 (data/market, pînă la '
       '18 septembrie 2026). Creșterea PIB real al SUA și datările NBER (FRED GDPC1, USREC), citite cu funcția capitolului pe 4 noiembrie 2026 și tăiate la T2 2026, '
       'ultimul trimestru din slide-uri; reproduc exact cifrele din curs. Cei patru indicatori coincidenți (FRED INDPRO, PAYEMS, W875RX1, CMRMTSPL), citiți tot pe '
       '4 noiembrie 2026 și tăiați la iulie 2026: au fost revizuiți după realizarea slide-urilor, iar parametrii modelului factorial (estimat pe 1967–2019) rămîn cei '
       'din curs, însă ultimele luni ale factorului diferă puțin (valoarea din iulie 2026: −0,87 cu datele de azi, −0,9 în slide). PIB-ul real al României '
       '(Eurostat namq_10_gdp) a fost revizuit după realizarea slide-urilor; pagina folosește seria din slide-uri, citită din linia vectorială a graficului din curs '
       '(charts/tsa_ch10_ro_trend.pdf, toate cele 126 de trimestre), care reproduce cifrele din curs pînă la a șasea cifră semnificativă. Componentele modelului UC, '
       'factorul comun, probabilitățile celor două maxime ale MS-AR(4), volatilitatea GARCH(1,1) și traiectoriile simulate ale regimurilor provin din codul capitolului, '
       'cu aceeași sămînță; în browser se recalculează filtrul și netezitorul Kalman, verosimilitatea, diagnosticarea, filtrul HP, filtrul Hamilton (2018), deviațiile '
       'în timp real, beta variabil, filtrul Hamilton (1989) și netezitorul Kim. Variantele care nu apar în curs (alt q, alt λ, alt h, alt sfîrșit al eșantionului, '
       'alt p₁₁, alt număr de luni lipsă) sînt rulări suplimentare ale codului capitolului, cu aceleași date; graficul „Verificări: regimurile identificate de model” este '
       'o rulare suplimentară pentru un slide care dă doar cifre.'),
 'en': dict(file='ch10.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 10 · State Space Models, the Kalman Filter and Markov Switching', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; '
        'the “Worked example” boxes redo the calculation with the numbers of the chapter.',
   src='Annual flow of the Nile at Aswan, 1871–1970 (the statsmodels data set). Weekly S&P 500, BET and Euro Stoxx 50 prices (data/market, up to 18 September 2026). '
       'US real GDP growth and the NBER dates (FRED GDPC1, USREC), read with the chapter’s loader on 4 November 2026 and cut at 2026Q2, the last quarter of the slides; '
       'they reproduce the numbers of the lecture exactly. The four coincident indicators (FRED INDPRO, PAYEMS, W875RX1, CMRMTSPL), also read on 4 November 2026 and '
       'cut at July 2026: they have been revised since the slides were made; the parameters of the factor model (estimated on 1967–2019) are those of the lecture, '
       'but the latest months of the factor differ slightly (the July 2026 value: −0.87 with today’s data, −0.9 on the slide). Romanian real GDP (Eurostat '
       'namq_10_gdp) has been revised since the slides were made; the page uses the series of the slides, read from the vector line of the lecture chart '
       '(charts/tsa_ch10_ro_trend.pdf, all 126 quarters), which reproduces the numbers of the lecture to the sixth significant digit. The components of the UC model, '
       'the common factor, the probabilities of the two maxima of the MS-AR(4), the GARCH(1,1) volatility and the simulated regime paths come from the chapter’s '
       'code, with the same seed; the browser recomputes the Kalman filter and smoother, the likelihood, the diagnostics, the HP filter, the Hamilton (2018) filter, '
       'the real-time gaps, the time-varying beta, the Hamilton (1989) filter and the Kim smoother. Variants that do not appear in the lecture (another q, λ, h, '
       'sample end, p₁₁ or number of missing months) are extra runs of the chapter’s code with the same data; the chart “Checks: the regimes identified by the model” '
       'is an extra run for a slide that gives only numbers.'),
}

CSS = ('<style>.c10x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c10x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c10x .c10f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c10x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c10x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c10x .wctl .ctl{max-width:360px;margin:.15rem 0 .2rem}'
       '.c10t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c10t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c10t th,.c10t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c10t thead tr:last-child th{border-bottom:1px solid #1A3A6E}'
       '.c10t tbody tr:nth-child(even){background:#F5F7FB}.c10t tbody tr.hi{background:#E3F4F7}.c10t .r{text-align:right}.c10cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}</style>')


def tbl(i):
    return f'<div class="tbl c10t" id="{i}"></div>'


def wx(i, title, formula, ctl=''):
    """Worked-example box; `ctl` holds the controls that only drive the box (they stay out of the chart's controls bar)."""
    return f'<div class="c10x"><h4>{title}</h4>' + (f'<div class="wctl">{ctl}</div>' if ctl else '') + f'<div class="c10f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def box(ctl, inner):
    """A box with controls that only drive the table inside it."""
    return f'<div class="c10x"><div class="wctl">{ctl}</div>{inner}</div>'


RO_ENDS = ['2008-07-01'] + [f'{y}-04-01' for y in range(2009, 2027)]
SV_ENDS = ['2019-10-01', '2020-04-01', '2020-10-01', '2021-10-01', '2023-10-01', '2026-04-01']
MF_WIN = [('2005-01-01|2012-10-01', '2005–2012'), ('1973-01-01|1976-10-01', '1973–1976'), ('1979-01-01|1983-10-01', '1979–1983'),
          ('1989-01-01|1992-10-01', '1989–1992'), ('2000-01-01|2002-10-01', '2000–2002'), ('2019-01-01|2021-10-01', '2019–2021')]


def widgets_all(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    lec = s(' (curs)', ' (lecture)')
    W = s('Exemplu rezolvat', 'Worked example')
    ql = lambda d: f'T{(int(d[5:7]) + 2) // 3} {d[:4]}' if ro else f'{d[:4]}Q{(int(d[5:7]) + 2) // 3}'   # noqa: E731
    qsl = lambda i: slider(i, s('Raportul semnal–zgomot q = σ²<sub>η</sub>/σ²<sub>ε</sub> (σ²<sub>ε</sub> = 15100; curs: 0,097, ML)',   # noqa: E731
                                'Signal-to-noise ratio q = σ²<sub>η</sub>/σ²<sub>ε</sub> (σ²<sub>ε</sub> = 15100; lecture: 0.097, ML)'), 0, 8, 1, 4)
    lam = lambda i, lab: slider(i, lab, 0, 7, 1, 4)   # noqa: E731
    return {
     'ss-examples': (slider('se-T', s('Perioade afișate T (curs: 200)', 'Periods shown T (lecture: 200)'), 20, 200, 20, 200)
                     + slider('se-q', s('q al traiectoriei roșii (curs: 1)', 'q of the red path (lecture: 1)'), 0.1, 3, 0.1, 1)
                     + slider('se-z', s('σ<sub>ζ</sub> al pantei (curs: 0,02)', 'σ<sub>ζ</sub> of the slope (lecture: 0.02)'), 0.005, 0.05, 0.005, 0.02) + play('se'),
                     CSS + charts('se-chart')),
     'nile-filter': (qsl('nf-q') + play('nf'), charts('nf-chart')),
     'gain-ses': (qsl('gs-q') + select('gs-a', s('Ponderea SES din dreapta', 'SES weight on the right'), [('K', 'α = K̄' + lec), ('ls', s('α̂ = 0,246, cele mai mici pătrate', 'α̂ = 0.246, least squares'))], 'K')
                  + play('gs'), charts('gs-chart')),
     'nile-smooth': (qsl('ns-q') + play('ns'), charts('ns-chart')),
     'likelihood': (slider('lk-q', s('q ales în dreapta, între 0,001 și 1 (curs: 0,097, ML)', 'q chosen on the right, between 0.001 and 1 (lecture: 0.097, ML)'), 0, 8, 1, 4) + play('lk'),
                    charts('lk-chart')),
     'diagnostics': (qsl('dg-q') + play('dg'), charts('dg-chart')),
     'missing': (qsl('mi-q') + slider('mi-g', s('Lungimea fiecărui gol (ani; curs: 20)', 'Length of each gap (years; lecture: 20)'), 0, 20, 5, 20)
                 + slider('mi-h', s('Orizontul prognozei (ani; curs: 30)', 'Forecast horizon (years; lecture: 30)'), 10, 50, 10, 30) + play('mi'), charts('mi-chart')),
     'ro-trend': (lam('rt-l', s('λ al filtrului HP (curs: 1600)', 'λ of the HP filter (lecture: 1600)')) + play('rt'), charts('rt-chart')),
     'output-gap': (slider('og-h', s('Orizontul h al filtrului Hamilton (trimestre; curs: 8)', 'Horizon h of the Hamilton filter (quarters; lecture: 8)'), 4, 12, 1, 8)
                    + slider('og-p', s('Laguri p (curs: 4)', 'Lags p (lecture: 4)'), 1, 6, 1, 4) + lam('og-l', s('λ al filtrului HP (curs: 1600)', 'λ of the HP filter (lecture: 1600)'))
                    + play('og'), charts('og-chart')),
     'realtime': (select('rl-e', s('Datele pînă în', 'Data up to'), [(e, ql(e) + (lec if e == '2026-04-01' else '')) for e in RO_ENDS], '2026-04-01')
                  + lam('rl-l', s('λ al filtrului HP (curs: 1600)', 'λ of the HP filter (lecture: 1600)')) + play('rl'), charts('rl-chart')),
     'tvp-beta': (slider('tv-s', s('Varianța șocurilor lui beta, multiplu al lui σ̂²<sub>β</sub> (curs: 1 × ML)', 'Variance of the beta shocks, multiple of σ̂²<sub>β</sub> (lecture: 1 × ML)'), 0, 5, 1, 3)
                  + slider('tv-w', s('Fereastra OLS mobilă (săptămîni; curs: 52)', 'Rolling OLS window (weeks; lecture: 52)'), 26, 104, 13, 52) + play('tv'), charts('tv-chart')),
     'dfm': (select('df-k', s('Luni lipsă pentru venit și vînzări la capătul eșantionului', 'Missing months of income and sales at the end of the sample'),
                    [(str(k), str(k) + (lec if k == 2 else '')) for k in range(7)], '2') + play('df'), charts('df-chart')),
     'ms-us': (slider('mu-p', s('p<sub>11</sub> al regimului de recesiune (curs: 0,685, ML)', 'p<sub>11</sub> of the recession regime (lecture: 0.685, ML)'), 0, 9, 1, 4) + play('mu'),
               charts('mu-chart')),
     'ms-filtered': (select('mf-w', s('Fereastra', 'Window'), [(v, t + (lec if v.startswith('2005') else '') + (s(' (parametrii din 1947–2019)', ' (parameters of 1947–2019)') if v.startswith('2019') else ''))
                                                             for v, t in MF_WIN], MF_WIN[0][0]) + play('mf'), charts('mf-chart')),
     'ms-ro': (slider('mr-p', s('p<sub>11</sub> al regimului volatil (curs: 0,911, ML)', 'p<sub>11</sub> of the volatile regime (lecture: 0.911, ML)'), 0, 8, 1, 6) + play('mr'),
               charts('mr-chart')),
     'vol-regimes': (slider('vr-p', s('p<sub>11</sub> al regimului agitat (curs: 0,930, ML)', 'p<sub>11</sub> of the turbulent regime (lecture: 0.930, ML)'), 0, 6, 1, 4) + play('vr'),
                     charts('vr-chart')),
     'regimes-memory': (slider('rm-m', s('Traiectorii simulate în medie (curs: 40)', 'Simulated paths averaged (lecture: 40)'), 1, 40, 1, 40)
                        + slider('rm-l', s('Lagul maxim (săptămîni; curs: 100)', 'Largest lag (weeks; lecture: 100)'), 20, 100, 10, 100) + play('rm'), charts('rm-chart')),
     # ---------------------------------------------------------------- EXTRA
     'wx-arma': (slider('wa-t', s('Trimestrul t din calcul', 'Quarter t of the calculation'), 1, 12, 1, 1) + play('wa'),
                 charts('wa-chart') + wx('wa-ws', W, r'\mathrm{vec}(P_1)=(I-T\otimes T)^{-1}\mathrm{vec}(Q),\quad v_t=y_t-\hat\mu-Za_t,\quad F_t=ZP_tZ^{\prime},\quad '
                                                 r'\log L=-\tfrac12\sum_t\Big(\log 2\pi+\log F_t+\frac{v_t^2}{F_t}\Big)')),
     'wx-kf': (slider('wk-e', s('σ²<sub>ε</sub> (curs: 4)', 'σ²<sub>ε</sub> (lecture: 4)'), 0.5, 10, 0.5, 4) + slider('wk-n', s('σ²<sub>η</sub> (curs: 1)', 'σ²<sub>η</sub> (lecture: 1)'), 0, 4, 0.25, 1)
               + slider('wk-p', s('P<sub>1</sub> (curs: 4)', 'P<sub>1</sub> (lecture: 4)'), 0.5, 20, 0.5, 4) + play('wk'),
               charts('wk-chart') + wx('wk-ws', W, r'F_t=P_t+\sigma^2_\varepsilon,\quad K_t=\frac{P_t}{F_t},\quad a_{t|t}=a_t+K_tv_t,\quad P_{t|t}=P_t(1-K_t),\quad P_{t+1}=P_{t|t}+\sigma^2_\eta',
                                       select('wk-s', s('Pasul din calcul', 'Step of the calculation'), [('1', 't = 1' + lec), ('2', 't = 2' + lec), ('3', 't = 3' + s(' (slide-ul 2/2)', ' (slide 2/2)'))], '1'))),
     'kf-table': (slider('kt-y', s('y<sub>3</sub> (curs: 14)', 'y<sub>3</sub> (lecture: 14)'), 8, 20, 0.5, 14) + slider('kt-a', s('a<sub>1</sub>, media a priori (curs: 10)', 'a<sub>1</sub>, the prior mean (lecture: 10)'), 6, 14, 0.5, 10)
                  + play('kt'),
                  charts('kt-chart') + box(select('kt-r', s('Rînduri', 'Rows'), [('3', '3' + lec), ('6', s('6: t = 4–6 fără date', '6: t = 4–6 without data'))], '3'), tbl('kt-tb'))),
     'wx-ss': (slider('ws-q', s('q = σ²<sub>η</sub>/σ²<sub>ε</sub> (curs: 0,25)', 'q = σ²<sub>η</sub>/σ²<sub>ε</sub> (lecture: 0.25)'), 0, 9, 1, 4) + play('ws'),
               charts('ws-chart') + wx('ws-ws', W, r'\bar P=\sigma^2_\varepsilon\,\frac{q+\sqrt{q^2+4q}}{2},\qquad \bar K=\frac{\bar P}{\bar P+\sigma^2_\varepsilon}=\alpha_{SES},\qquad q=\frac{\alpha^2}{1-\alpha}',
                                       slider('ws-a', s('Ponderea SES α pentru relația inversă (autoevaluare: 0,5)', 'SES weight α for the inverse map (self-assessment: 0.5)'), 0.05, 0.95, 0.05, 0.5))),
     'nile-ml': (slider('nm-e', s('Ultimul an al eșantionului (curs: 1970)', 'Last year of the sample (lecture: 1970)'), 1920, 1970, 5, 1970) + play('nm'),
                 charts('nm-chart') + box(select('nm-u', s('Afișați', 'Show'), [('var', s('varianțele', 'variances') + lec), ('sd', s('abaterile standard', 'standard deviations'))], 'var'), tbl('nm-tb'))),
     'wx-rts': (slider('wr-y', s('Anul t din calcul (curs: 1897)', 'Year t of the calculation (lecture: 1897)'), 1872, 1969, 1, 1897) + play('wr'),
                charts('wr-chart') + wx('wr-ws', W, r'J_t=\frac{P_{t|t}}{P_{t+1}},\qquad \hat\alpha_t=a_{t|t}+J_t(\hat\alpha_{t+1}-a_{t+1}),\qquad V_t=P_{t|t}+J_t^2(V_{t+1}-P_{t+1})')),
     'wx-lik': (qsl('wl-q') + play('wl'),
                charts('wl-chart') + wx('wl-ws', W, r'\log L=\sum_{t=2}^{n}\ell_t,\qquad \ell_t=-\tfrac12\Big(\log 2\pi+\log F_t+\frac{v_t^2}{F_t}\Big)',
                                        slider('wl-y', s('Anul termenului din calcul', 'Year of the term in the calculation'), 1872, 1970, 1, 1899))),
     'wx-hp': (lam('wh-l', s('λ (curs: 1600)', 'λ (lecture: 1600)')) + play('wh'),
               charts('wh-chart') + wx('wh-ws', W, r'\min_{\tau}\sum_t(y_t-\tau_t)^2+\lambda\sum_t(\Delta^2\tau_t)^2\ \Longleftrightarrow\ y_t=\mu_t+\varepsilon_t,\ \ \Delta^2\mu_{t+1}=\zeta_{t-1},\ \ \lambda=\frac{\sigma^2_\varepsilon}{\sigma^2_\zeta}')),
     'wx-dur': (slider('wd-p', s('p<sub>11</sub> (curs: 0,75)', 'p<sub>11</sub> (lecture: 0.75)'), 0.05, 0.95, 0.05, 0.75) + slider('wd-q', s('p<sub>22</sub> (curs: 0,95)', 'p<sub>22</sub> (lecture: 0.95)'), 0.5, 0.99, 0.01, 0.95)
                + play('wd'),
                charts('wd-chart') + wx('wd-ws', W, r'E(D_i)=\frac{1}{1-p_{ii}},\qquad \pi_1=\frac{1-p_{22}}{2-p_{11}-p_{22}},\qquad \Pr(S_{t+h}=1\mid S_t=1)=\pi_1+(1-\pi_1)(p_{11}+p_{22}-1)^h',
                                        slider('wd-h', s('Orizontul h al lui P<sup>h</sup>', 'Horizon h of P<sup>h</sup>'), 1, 20, 1, 4))),
     'wx-ham': (slider('wf-t', s('Trimestrul (curs: T3 2008)', 'Quarter (lecture: 2008Q3)'), 0, 15, 1, 6) + play('wf'),
                charts('wf-chart') + wx('wf-ws', W, r'\Pr(S_t=j\mid Y_{t-1})=\sum_i p_{ij}\Pr(S_{t-1}=i\mid Y_{t-1}),\qquad \Pr(S_t=j\mid Y_t)=\frac{\Pr(S_t=j\mid Y_{t-1})\,f_j(y_t)}{\sum_i\Pr(S_t=i\mid Y_{t-1})\,f_i(y_t)}')),
     'ms-table': (select('mt-m', s('Maximul din grafic', 'Maximum in the chart'), [('A', 'A' + lec), ('B', 'B')], 'A') + play('mt'),
                  charts('mt-chart') + box(slider('mt-c', s('Pragul probabilității netezite (curs: 0,5)', 'Threshold of the smoothed probability (lecture: 0.5)'), 0.3, 0.7, 0.05, 0.5), tbl('mt-tb'))),
     'ms-checks': (select('mc-e', s('Eșantionul 1947–… se termină în', 'The sample 1947–… ends in'),
                          [(e, ql(e) + (s(' (curs: varianța cu schimbare de regim)', ' (lecture: switching variance)') if e == SV_ENDS[0] else s(' (curs: includerea anului 2020)', ' (lecture: including 2020)') if e == SV_ENDS[-1] else ''))
                           for e in SV_ENDS], SV_ENDS[0]) + play('mc'), charts('mc-chart')),
     'ms-params': (slider('mp-h', s('Durata d (perioade)', 'Duration d (periods)'), 1, 40, 1, 4) + play('mp'),
                   charts('mp-chart') + box(select('mp-u', s('Duratele în', 'Durations in'), [('p', s('perioade', 'periods') + lec), ('y', s('ani', 'years'))], 'p'), tbl('mp-tb'))),
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
    """Chart blocks for build_web_lecture.py: each host carries the web version of its EXTRA blocks (moved by ch10.js)."""
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
        img = f'ch10x_{anchor}_{lang}.jpg'
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
const q = d => `T${(+d.slice(5, 7) + 2) / 3} ${d.slice(0, 4)}`;
window.CH10_TEXT = {
 q, qq: (a, b) => `${q(a)}–${q(b)}`,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 ml: ' (ML)', sep: '; ', lt: '&lt; ',
 tvS: (m, ml) => `${m} × σ̂²<sub>β</sub>${ml ? ' (ML)' : ''}`,
 L: { nobs: 'Observații', par: 'Parametrul', own: 'Filtrul propriu (curs)', sm: 'statsmodels (curs)', conc: e => `Profilul concentrat, 1871–${e}`,
   means: 'Media, regimul 1 / regimul 2 (% pe trimestru)', dur: 'Durata așteptată, regimul 1 / 2 (trimestre)', found: c => `Trimestre de recesiune NBER găsite (prob. netezită > ${c})`,
   falseA: c => `Alarme false (prob. netezită > ${c})`, msar: 'MS-AR(4), creșterea PIB real al SUA, T2 1951–T4 1984', maxA: 'Maximul A', maxB: 'Maximul B',
   reg1: 'Regimul 1', mu12: 'Media, regimul 1 / 2', sd12: 'Abaterea standard, regimul 1 / 2', dur12: u => `Durata așteptată, regimul 1 / 2 (${u === 'y' ? 'ani' : 'perioade'})`, model: 'Modelul' },
 mname: { us: 'Creșterea PIB, SUA, 1947–2019', sv: 'PIB SUA, varianță cu schimbare de regim', ro: 'Creșterea PIB, România', sp: 'Randamente săptămînale S&P 500' },
 reg1: { us: 'recesiune', sv: 'varianță mică', ro: 'volatil', sp: 'agitat' },
 se1: (n, l) => `Primele ${n}${de(n)} perioade; autocorelația de ordinul 1 a lui Δy<sub>t</sub> (teoretic −1/(q + 2)) și abaterea standard a lui Δy<sub>t</sub>: ${l}.`,
 seq: (q_, a, th, sd) => `q = ${q_}: ${a} (teoretic ${th}), abaterea standard ${sd}`,
 se2: (sz, s) => `Local linear trend cu σ<sub>ζ</sub> = ${sz}: panta ajunge la β = ${s} în ultima perioadă afișată (pornește de la 0,05).`,
 was1: (p1, p2, s2, g0, g1) => `Pornirea staționară: vec(P<sub>1</sub>) = (I − T ⊗ T)<sup>−1</sup> vec(Q), cu φ̂<sub>1</sub> = ${p1}, φ̂<sub>2</sub> = ${p2}, σ̂² = ${s2}, dă P<sub>1</sub> cu diagonala ${g0} și elementul din afara diagonalei ${g1}: varianța γ(0) și autocovarianța γ(1) ale AR(2).`,
 was2: (t, q_, y, f, v, F, k1, k2) => `Trimestrul t = ${t} (${q_}): y<sub>t</sub> = ${y}%, prognoza μ̂ + Za<sub>t</sub> = ${f}%, deci v<sub>t</sub> = ${v}; F<sub>t</sub> = ZP<sub>t</sub>Z′ = ${F}; K<sub>t</sub> = P<sub>t</sub>Z′/F<sub>t</sub> = (${k1}; ${k2})′.`,
 was3: (t, l, ll, ls) => `Contribuția trimestrului ${t}: −½(log 2π + log F<sub>t</sub> + v<sub>t</sub>²/F<sub>t</sub>) = ${l}. Însumat pe cele 291 de trimestre: log L = <b>${ll}</b>; SARIMAX: ${ls}.`,
 wa1: (t, q_, y, f, F) => `Trimestrul ${t} (${q_}): creșterea ${y}%, prognoza la un pas ${f}%, F<sub>t</sub> = ${F}.`,
 wa2: (ll, ls, n) => `Log-verosimilitatea Kalman pe ${n}${de(n)} trimestre: ${ll}; SARIMAX: ${ls}.`,
 wks1: (t, P, e, F, K) => `t = ${t}: F<sub>t</sub> = P<sub>t</sub> + σ²<sub>ε</sub> = ${P} + ${e} = ${F}; K<sub>t</sub> = P<sub>t</sub>/F<sub>t</sub> = <b>${K}</b>.`,
 wks2: (t, y, a, v) => `v<sub>t</sub> = y<sub>t</sub> − a<sub>t</sub> = ${y} − ${a} = ${v}.`,
 wks3: (t, a, K, v, af, P, Pf) => `a<sub>t|t</sub> = a<sub>t</sub> + K<sub>t</sub>v<sub>t</sub> = ${a} + ${K}·${v} = <b>${af}</b>; P<sub>t|t</sub> = P<sub>t</sub>(1 − K<sub>t</sub>) = ${P}·(1 − ${K}) = ${Pf}.`,
 wks4: (t1, af, Pf, n, P) => `Predicția: a<sub>${t1}</sub> = a<sub>t|t</sub> = ${af}, P<sub>${t1}</sub> = P<sub>t|t</sub> + σ²<sub>η</sub> = ${Pf} + ${n} = ${P}.`,
 wk1: (k1, k2, k3, a) => `Cîștigurile K<sub>1</sub>, K<sub>2</sub>, K<sub>3</sub>: ${k1}, ${k2}, ${k3}; nivelul filtrat la t = 3: ${a}.`,
 wk2: (q_, K, P) => `q = σ²<sub>η</sub>/σ²<sub>ε</sub> = ${q_}: starea de echilibru K̄ = ${K}, P̄ = ${P}.`,
 ktCap: (a1, nr) => `Local level cu σ²<sub>ε</sub> = 4, σ²<sub>η</sub> = 1, a<sub>1</sub> = ${a1}, P<sub>1</sub> = 4; ${nr} rînduri${nr > 3 ? ' (t = 4–6 fără date: pasul de actualizare lipsește)' : ''}.`,
 kt1: (y, v, K, m, a) => `y<sub>3</sub> = ${y}: surpriza v<sub>3</sub> = ${v}; nivelul se mută cu K<sub>3</sub>v<sub>3</sub> = ${K}·${v} = ${m}, pînă la a<sub>3|3</sub> = ${a}.`,
 kt2: (a, P, F) => `Prognoza pentru t = 4: a<sub>4</sub> = ${a}, P<sub>4</sub> = ${P}; prognoza lui y<sub>4</sub> are varianța P<sub>4</sub> + σ²<sub>ε</sub> = ${F}.`,
 wss1: (q_, d, p, P) => `q = ${q_}: P̄/σ²<sub>ε</sub> = (q + √(q² + 4q))/2 = (${q_} + √${d})/2 = ${p}; cu σ²<sub>ε</sub> = 4, P̄ = ${P}.`,
 wss2: (P, K) => `K̄ = P̄/(P̄ + σ²<sub>ε</sub>) = ${P}/(${P} + 4) = <b>${K}</b>: ponderea α a SES.`,
 wss3: (a, q_) => `Relația inversă: ponderea SES α = ${a} corespunde lui q = α²/(1 − α) = <b>${q_}</b>.`,
 ws1: (q_, P, K, K3) => `q = ${q_}: P̄ = ${P}, K̄ = ${K} (cu σ²<sub>ε</sub> = 4); pe drum, K<sub>3</sub> = ${K3}.`,
 nf1: (q_, n, K) => `q = ${q_}: σ²<sub>η</sub> = ${n}, cu σ²<sub>ε</sub> = 15100; cîștigul de echilibru K̄ = ${K}.`,
 nf2: (y, ap, v, K, af, a5) => `1899: debitul ${y}, nivelul prezis ${ap}, surpriza v = ${v}; cu K = ${K} nivelul coboară la ${af} (${a5} în 1905).`,
 nf3: (a, sd, m) => `1970: nivelul ${a}, cu eroarea standard ${sd}; media eșantionului ${m}.`,
 nmCap: 'Modelul local level pentru Nil: verosimilitate maximă cu pornire difuză (filtrul propriu, curs), statsmodels UnobservedComponents (pornire difuză exactă, curs) și verosimilitatea profil concentrată, maximizată în browser.',
 nmNote: 'AIC = −2 log L + 2·2, BIC = −2 log L + 2 log n; prima observație nu intră în log L (pornire difuză). Un alt sfîrșit al eșantionului este o rulare suplimentară a codului capitolului.',
 nm1: (e, q_, se, sn, ll) => `1871–${e}: q̂ = ${q_}, σ̂²<sub>ε</sub> = ${se}, σ̂²<sub>η</sub> = ${sn}, log L = ${ll}.`,
 nm2: K => `Cîștigul de echilibru al acestei estimări: K̄ = ${K}.`,
 gs1: (k2, k5, nc, K) => `K<sub>t</sub> pornește de la ${k2} (1872, după pornirea difuză), este ${k5} în anul 5 și rămîne la mai puțin de 0,001 de K̄ = ${K} începînd din anul ${nc}.`,
 gs2: (a, g5, g20) => `SES cu α = ${a}: prognozele diferă de cele Kalman cu ${g5} la a 5-a prognoză și cu cel mult ${g20} începînd cu a 20-a.`,
 gs3: (a, q_) => `SES estimată prin cele mai mici pătrate: α̂ = ${a}, adică q = α²/(1 − α) = ${q_}.`,
 wrs1: (yr, Pf, yr1, Pp, J) => `J<sub>${yr}</sub> = P<sub>t|t</sub>/P<sub>t+1</sub> = ${Pf}/${Pp} = <b>${J}</b>.`,
 wrs2: (yr, af, J, as1, ap1, as) => `α̂<sub>${yr}</sub> = a<sub>t|t</sub> + J<sub>t</sub>(α̂<sub>t+1</sub> − a<sub>t+1</sub>) = ${af} + ${J}·(${as1} − ${ap1}) = <b>${as}</b>.`,
 wrs3: (Pf, J, Ps1, Pp1, Ps, sds, sdf) => `V<sub>t</sub> = P<sub>t|t</sub> + J<sub>t</sub>²(V<sub>t+1</sub> − P<sub>t+1</sub>) = ${Pf} + ${J}²·(${Ps1} − ${Pp1}) = ${Ps}: eroarea standard ${sds}, față de ${sdf} pentru nivelul filtrat.`,
 wr1: (yr, af, as, J) => `${yr}: filtrat ${af}, netezit ${as}, J<sub>t</sub> = ${J}.`,
 ns1: (q_, s97, f97, s00, f00) => `q = ${q_}. 1897: netezit ${s97}, filtrat ${f97}; 1900: netezit ${s00}, filtrat ${f00}.`,
 ns2: (s20, f20, sN, fN) => `Eroarea standard în 1920: ${s20} netezit, ${f20} filtrat; în 1970: ${sN} netezit, ${fN} filtrat (la t = n coincid).`,
 wls1: (yr, y, a, v, F) => `${yr}: y<sub>t</sub> = ${y}, a<sub>t</sub> = ${a}, v<sub>t</sub> = ${v}, F<sub>t</sub> = ${F}.`,
 wls2: (yr, c, lf, r, l) => `ℓ<sub>t</sub> = −½(log 2π + log F<sub>t</sub> + v<sub>t</sub>²/F<sub>t</sub>) = −½(${c} + ${lf} + ${r}) = <b>${l}</b>.`,
 wls3: (n, ll, F1) => `Suma pe ${n}${de(n)} ani (1871 nu intră: F<sub>1</sub> = ${F1}, pornirea difuză): log L = <b>${ll}</b>.`,
 wl1: (q_, ll, n) => `q = ${q_}: log L = ${ll}, din ${n}${de(n)} erori de predicție.`,
 wl2: (yr, l) => `Cea mai mică contribuție: ${yr} (${l}).`,
 lk1: (q_, s2, ll) => `Maximul la q̂ = ${q_}, cu σ̂²<sub>ε</sub> = ${s2} din forma închisă; log-verosimilitatea profil ${ll}.`,
 lk2: (q_, ll, lr, s2) => `q = ${q_}: log-verosimilitatea profil ${ll}, σ̂²<sub>ε</sub>(q) = ${s2}; statistica raportului de verosimilitate față de maxim: ${lr}.`,
 dg1: (q_, lb, p, jb, pj, k, n) => `q = ${q_}: Ljung–Box Q(10) = ${lb} (p = ${p}), Jarque–Bera ${jb} (p = ${pj}); ${k} din ${n} erori în afara intervalului ±1,96.`,
 dg2: (yl, l, yo, o) => `Minimul reziduului nivelului: ${yl} (${l}); minimul reziduului observației: ${yo} (${o}).`,
 mi0: s => `Niciun an eliminat: eroarea standard a nivelului netezit în 1900 este ${s}.`,
 mi1: (L_, y, s, sf, n) => `${n}${de(n)} ani eliminați (două goluri de cîte ${L_} ani): eroarea standard a nivelului netezit în ${y} este ${s}, față de ${sf} cu toate datele.`,
 mi2: (q_, f, s1, H, sH) => `q = ${q_}: prognoza ${f} pentru orice orizont; eroarea standard a lui y este ${s1} la un an și ${sH} după ${H}${de(H)} ani.`,
 whs1: (lam, z) => `λ = σ²<sub>ε</sub>/σ²<sub>ζ</sub> = ${lam}: cu σ²<sub>ε</sub> = 1, σ²<sub>ζ</sub> = 1/λ = ${z}.`,
 whs2: () => `Trend neted: μ<sub>t+1</sub> = μ<sub>t</sub> + β<sub>t</sub>, β<sub>t+1</sub> = β<sub>t</sub> + ζ<sub>t</sub>, deci Δ²μ<sub>t+1</sub> = β<sub>t</sub> − β<sub>t−1</sub> = ζ<sub>t−1</sub>. De −2 ori logaritmul densității comune, înmulțit cu σ²<sub>ε</sub>, este Σ(y<sub>t</sub> − μ<sub>t</sub>)² + (σ²<sub>ε</sub>/σ²<sub>ζ</sub>)Σ(Δ²μ<sub>t</sub>)²: criteriul HP cu λ = σ²<sub>ε</sub>/σ²<sub>ζ</sub>.`,
 whs3: d => `Netezitorul Kalman (pornire difuză) față de trendul prin cele mai mici pătrate penalizate: cea mai mare diferență absolută ${d}.`,
 wh1: (lam, sd, mx, q_) => `λ = ${lam}: abaterea standard a deviației HP ${sd}; cea mai mare deviație ${mx}% în ${q_}.`,
 rt1: (g, gx, qx, gn, qn) => `Creștere medie de ${g}% pe an; creșterea anualizată a trendului UC între ${gn}% (${qn}) și ${gx}% (${qx}).`,
 rt2: (sz, p1, p2, sk, r) => `Estimările UC: σ̂²<sub>ζ</sub> = ${sz} (panta trendului), ciclul AR(2) cu φ̂<sub>1</sub> = ${p1}, φ̂<sub>2</sub> = ${p2}, σ̂²<sub>κ</sub> = ${sk}; raportul varianțelor șocurilor σ̂²<sub>κ</sub>/σ̂²<sub>ζ</sub> = ${r}.`,
 rt3: (lam, d) => `HP cu λ = ${lam} (raportul σ²<sub>ε</sub>/σ²<sub>ζ</sub> impus): cea mai mare distanță față de trendul UC este ${d} puncte.`,
 og1: (pu, qu, ph, pm, u, h, m) => `Maxime în 2007–2009: UC ${pu}% (${qu}), HP ${ph}%, Hamilton ${pm}%. T2 2020: UC ${u}%, HP ${h}%, Hamilton ${m}%.`,
 og2: (a, b, c) => `Corelații: UC–HP ${a}, UC–Hamilton ${b}, HP–Hamilton ${c}.`,
 og3: (su, sh, sm, p, bs, lost) => `Abateri standard: UC ${su}, HP ${sh}, Hamilton ${sm}; ${p === 1 ? 'coeficientul' : `cei ${p} coeficienți`} ai regresiei ${p === 1 ? 'este' : 'au suma'} ${bs}; primele ${lost} trimestre se pierd.`,
 og4: (q_, u, h, m) => `Ultima deviație (${q_}): UC ${u}%, HP ${h}%, Hamilton ${m}%.`,
 rl1: (q_, rt, fin, uf, us) => `Datele pînă în ${q_}. T3 2008: HP ${rt}% în timp real, ${fin}% bilateral; UC ${uf}% filtrat, ${us}% netezit.`,
 rl2: (q_, rh, ru, ch, cu) => `T4 2004–${q_}: revizuirea medie pătratică HP ${rh} puncte, UC ${ru} puncte; corelația timp real–final HP ${ch}, UC ${cu}.`,
 tv1: (o, mx, dmx, mn, dmn) => `Beta OLS constant ${o}; beta netezit: maximul ${mx} (${dmx}), minimul ${mn} (${dmn}).`,
 tv2: (lr, sd, ml) => ml ? `Raportul de verosimilitate față de un beta constant: ${lr}; abaterea standard săptămînală a șocurilor lui beta ${sd}.`
   : `σ<sub>β</sub> = ${sd} (σ²<sub>ε</sub> la valoarea ML): raportul de verosimilitate față de beta constant ${lr} (rulare suplimentară).`,
 tv3: (b, sd, w, sr, sk) => `Ultimul beta ${b}, cu eroarea standard ${sd}. Variația săptămînală: abaterea standard ${sr} pentru OLS pe ${w}${de(w)} săptămîni, ${sk} pentru beta Kalman.`,
 df1: (e, ip, s, inc, p1, p2) => `Încărcări: salariați ${e}, producția industrială ${ip}, vînzări ${s}, venit ${inc}; factorul AR(2) cu φ̂<sub>1</sub> = ${p1}, φ̂<sub>2</sub> = ${p2}.`,
 df2: (mr, me, sh, mn, dmn) => `Media factorului: ${mr} în lunile de recesiune NBER, ${me} în expansiuni; negativ în ${sh} din lunile de recesiune; minimul ${mn} (${dmn}).`,
 df3: (k, m, v, sl, td, dl) => `Valoarea din ${m}, cu ${k === 1 ? 'o lună lipsă' : k + ' luni lipsă'}: ${v}. Datele FRED descărcate pe ${dl} au fost revizuite după realizarea slide-urilor: cu două luni lipsă valoarea este ${td}, iar slide-ul arată ${sl}.`,
 wds1: (p, q_, d1, p22, d2) => `E(D<sub>1</sub>) = 1/(1 − p<sub>11</sub>) = 1/${q_} = <b>${d1}</b> perioade; E(D<sub>2</sub>) = 1/(1 − ${p22}) = <b>${d2}</b> perioade.`,
 wds2: (a, b, pi) => `π<sub>1</sub> = (1 − p<sub>22</sub>)/(2 − p<sub>11</sub> − p<sub>22</sub>) = ${a}/${b} = <b>${pi}</b>.`,
 wds3: (h, a, b, c, d, l) => `P<sup>${h}</sup>: din regimul 1, Pr(1) = ${a}, Pr(2) = ${b}; din regimul 2, Pr(1) = ${c}, Pr(2) = ${d}; distanța față de π<sub>1</sub> se micșorează cu factorul p<sub>11</sub> + p<sub>22</sub> − 1 = ${l} în fiecare perioadă.`,
 wd1: (p11, d1, p22, d2, pi) => `p<sub>11</sub> = ${p11}: regimul 1 durează în medie ${d1} perioade; p<sub>22</sub> = ${p22}: ${d2} perioade; probabilitatea ergodică π<sub>1</sub> = ${pi}.`,
 wfs1: (qp, a, b, p11, p21, pr) => `Predicția: Pr(S<sub>t</sub> = 1 | Y<sub>t−1</sub>) = p<sub>11</sub>·${a} + (1 − p<sub>22</sub>)·${b} = ${p11}·${a} + ${p21}·${b} = <b>${pr}</b> (probabilitățile filtrate din ${qp}).`,
 wfs2: (y, m1, m2, s, d1, d2) => `Densitățile în y<sub>t</sub> = ${y}: f<sub>1</sub> = φ((y<sub>t</sub> − (${m1}))/${s})/${s} = ${d1}, f<sub>2</sub> = φ((y<sub>t</sub> − ${m2})/${s})/${s} = ${d2}.`,
 wfs3: (j1, j2, fl) => `Actualizarea: Pr(S<sub>t</sub> = 1 | Y<sub>t</sub>) = ${j1}/(${j1} + ${j2}) = <b>${fl}</b>.`,
 wf1: (q_, y, pr, fl, s) => `${q_}: creșterea ${y}%; probabilitatea recesiunii: prezisă ${pr}, filtrată ${fl}, netezită ${s}.`,
 mtCap: 'MS-AR(4) cu medie variabilă, creșterea PIB real al SUA, T2 1951–T4 1984 (primele patru trimestre sînt laguri); două maxime ale verosimilității, atinse din două valori de pornire.',
 mt1: (m, k, n, h, nr) => `Maximul ${m}: ${k} din ${n} trimestre au probabilitatea netezită peste 0,5; găsește ${h} din cele ${nr} trimestre de recesiune NBER.`,
 mu1: (m1, m2, s, p11, p22) => `Mediile regimurilor: ${m1}% (recesiune) și ${m2}% (expansiune), σ̂ = ${s}; p<sub>11</sub> = ${p11}, p<sub>22</sub> = ${p22}.`,
 mu2: (d1, d2, pi) => `Durate așteptate: ${d1} trimestre de recesiune, ${d2} trimestre de expansiune; ponderea ergodică a recesiunii ${pi}.`,
 mu3: (c, h, f) => `Concordanța cu NBER, 1947–2019: ${c} din trimestre; ${h} din trimestrele de recesiune detectate, ${f} alarme false.`,
 mu4: (p, mx) => `T2 2020: probabilitatea ${p}; după 2021 cel mult ${mx}.`,
 mf0: 'Nicio recesiune NBER în această fereastră.',
 mf1: (a, b, r, f, s) => `${a}–${b}: prima recesiune NBER ${r ? 'începe în ' + r : 'lipsește'}; probabilitatea filtrată trece de 0,5 ${f ? 'prima dată în ' + f : 'niciodată'}, cea netezită ${s ? 'în ' + s : 'niciodată'}.`,
 mf2: (q_, f, s) => `În ${q_}: filtrată ${f}, netezită ${s}.`,
 mf3: (c, h, late) => `Probabilitățile filtrate față de NBER, 1947–2019: ${c} din trimestre, ${h} din trimestrele de recesiune.${late ? ' După 2019: parametrii din 1947–2019 (rulare suplimentară).' : ''}`,
 mc1: (e, s2, s1, d2, d1, sw, c, f) => `Varianță cu schimbare de regim, T2 1947–${e}: abateri standard ${s2} și ${s1}; durate de ${d2} și ${d1} trimestre; primul trimestru calm ${sw}; concordanța cu NBER ${c} (${f} alarme false).`,
 mc2: (m1, m2, d1, d2, n1, q_) => `Medie cu schimbare de regim: media regimului 1 ${m1}%, a regimului 2 ${m2}%; durate de ${d1} și ${d2} trimestre; ${n1 === 1 ? 'un trimestru' : n1 + (n1 % 100 >= 20 || (n1 % 100 === 0 && n1) ? ' de' : '') + ' trimestre'} în regimul 1${q_ ? ': ' + q_ : ''}.`,
 mr1: (m2, s2, m1, s1) => `Regimul stabil: media ${m2}% pe trimestru, abaterea standard ${s2}; regimul volatil: media ${m1}%, abaterea standard ${s1}.`,
 mr2: (p, d2, d1, sh, pi) => `p<sub>11</sub> = ${p}: durate așteptate de ${d2} trimestre stabile și ${d1} trimestre volatile; ${sh} din trimestre sînt volatile (ergodic ${pi}).`,
 mr3: s => `Episoade volatile: ${s}.`,
 mr4: p => `Ultima probabilitate a regimului volatil: ${p}.`,
 vr1: (v2, m2, v1, m1) => `Regimul calm: volatilitatea ${v2}% pe an, media ${m2}% pe săptămînă; regimul agitat: ${v1}% pe an, media ${m1}% pe săptămînă.`,
 vr2: (p, d2, d1, sh, pi) => `p<sub>11</sub> = ${p}: durate de ${d2} săptămîni calme și ${d1} săptămîni agitate; agitat în ${sh} din săptămîni (ergodic ${pi}).`,
 vr3: s => `Cei mai agitați ani: ${s}.`,
 vr4: (c, a, b) => `Corelația cu volatilitatea GARCH(1,1): ${c}; GARCH: α̂ = ${a}, β̂ = ${b}.`,
 rm1: (a1, a26, M, s1, s26) => `Datele: ACF a lui |r<sub>t</sub>| ${a1} la lagul 1, ${a26} la 26 de săptămîni; media a ${M} ${M === 1 ? 'traiectorie simulată' : 'traiectorii simulate'}: ${s1}${s26 ? ' și ' + s26 : ''}.`,
 rm2: (d, M, ds, sd, di) => `d̂ local Whittle al lui |r<sub>t</sub>|: ${d} în date, ${ds} (abaterea standard ${sd}) pe ${M} ${M === 1 ? 'traiectorie simulată' : 'traiectorii simulate'}, ${di} în ordine aleatoare.`,
 rm3: (k, ab, mn) => `GARCH(1,1) pe primele ${k} traiectorii simulate: α + β = ${ab} în medie (minimum ${mn}), deși traiectoriile nu conțin GARCH.`,
 mpCap: 'Modelele cu două regimuri din capitol, estimate cu statsmodels (MarkovRegression); regimul 1 ca în curs. Duratele: E(D) = 1/(1 − p<sub>ii</sub>).',
 mp1: (h, s) => `Probabilitatea ca un episod din regimul 1 să dureze mai mult de ${h} perioade: ${s}.`,
};
})();
""",
 'en': r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const q = d => `${d.slice(0, 4)}Q${(+d.slice(5, 7) + 2) / 3}`;
window.CH10_TEXT = {
 q, qq: (a, b) => `${q(a)}–${q(b)}`,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 ml: ' (ML)', sep: '; ', lt: '&lt; ',
 tvS: (m, ml) => `${m} × σ̂²<sub>β</sub>${ml ? ' (ML)' : ''}`,
 L: { nobs: 'Observations', par: 'Parameter', own: 'Own filter (lecture)', sm: 'statsmodels (lecture)', conc: e => `Concentrated profile, 1871–${e}`,
   means: 'Mean, regime 1 / regime 2 (% per quarter)', dur: 'Expected duration, regime 1 / 2 (quarters)', found: c => `NBER recession quarters found (smoothed prob. > ${c})`,
   falseA: c => `False alarms (smoothed prob. > ${c})`, msar: 'MS-AR(4), US real GDP growth, 1951Q2–1984Q4', maxA: 'Maximum A', maxB: 'Maximum B',
   reg1: 'Regime 1', mu12: 'Mean, regime 1 / 2', sd12: 'Standard deviation, regime 1 / 2', dur12: u => `Expected duration, regime 1 / 2 (${u === 'y' ? 'years' : 'periods'})`, model: 'Model' },
 mname: { us: 'US GDP growth, 1947–2019', sv: 'US GDP, switching variance', ro: 'Romanian GDP growth', sp: 'S&P 500 weekly returns' },
 reg1: { us: 'recession', sv: 'low variance', ro: 'volatile', sp: 'turbulent' },
 se1: (n, l) => `First ${n} periods; lag-1 autocorrelation of Δy<sub>t</sub> (theory −1/(q + 2)) and standard deviation of Δy<sub>t</sub>: ${l}.`,
 seq: (q_, a, th, sd) => `q = ${q_}: ${a} (theory ${th}), standard deviation ${sd}`,
 se2: (sz, s) => `Local linear trend with σ<sub>ζ</sub> = ${sz}: the slope reaches β = ${s} in the last period shown (it starts at 0.05).`,
 was1: (p1, p2, s2, g0, g1) => `Stationary start: vec(P<sub>1</sub>) = (I − T ⊗ T)<sup>−1</sup> vec(Q), with φ̂<sub>1</sub> = ${p1}, φ̂<sub>2</sub> = ${p2}, σ̂² = ${s2}, gives P<sub>1</sub> with diagonal ${g0} and off-diagonal ${g1}: the variance γ(0) and the autocovariance γ(1) of the AR(2).`,
 was2: (t, q_, y, f, v, F, k1, k2) => `Quarter t = ${t} (${q_}): y<sub>t</sub> = ${y}%, forecast μ̂ + Za<sub>t</sub> = ${f}%, so v<sub>t</sub> = ${v}; F<sub>t</sub> = ZP<sub>t</sub>Z′ = ${F}; K<sub>t</sub> = P<sub>t</sub>Z′/F<sub>t</sub> = (${k1}, ${k2})′.`,
 was3: (t, l, ll, ls) => `Contribution of quarter ${t}: −½(log 2π + log F<sub>t</sub> + v<sub>t</sub>²/F<sub>t</sub>) = ${l}. Summed over the 291 quarters: log L = <b>${ll}</b>; SARIMAX: ${ls}.`,
 wa1: (t, q_, y, f, F) => `Quarter ${t} (${q_}): growth ${y}%, one-step forecast ${f}%, F<sub>t</sub> = ${F}.`,
 wa2: (ll, ls, n) => `Kalman log-likelihood over ${n} quarters: ${ll}; SARIMAX: ${ls}.`,
 wks1: (t, P, e, F, K) => `t = ${t}: F<sub>t</sub> = P<sub>t</sub> + σ²<sub>ε</sub> = ${P} + ${e} = ${F}; K<sub>t</sub> = P<sub>t</sub>/F<sub>t</sub> = <b>${K}</b>.`,
 wks2: (t, y, a, v) => `v<sub>t</sub> = y<sub>t</sub> − a<sub>t</sub> = ${y} − ${a} = ${v}.`,
 wks3: (t, a, K, v, af, P, Pf) => `a<sub>t|t</sub> = a<sub>t</sub> + K<sub>t</sub>v<sub>t</sub> = ${a} + ${K}·${v} = <b>${af}</b>; P<sub>t|t</sub> = P<sub>t</sub>(1 − K<sub>t</sub>) = ${P}·(1 − ${K}) = ${Pf}.`,
 wks4: (t1, af, Pf, n, P) => `Prediction: a<sub>${t1}</sub> = a<sub>t|t</sub> = ${af}, P<sub>${t1}</sub> = P<sub>t|t</sub> + σ²<sub>η</sub> = ${Pf} + ${n} = ${P}.`,
 wk1: (k1, k2, k3, a) => `Gains K<sub>1</sub>, K<sub>2</sub>, K<sub>3</sub>: ${k1}, ${k2}, ${k3}; filtered level at t = 3: ${a}.`,
 wk2: (q_, K, P) => `q = σ²<sub>η</sub>/σ²<sub>ε</sub> = ${q_}: steady state K̄ = ${K}, P̄ = ${P}.`,
 ktCap: (a1, nr) => `Local level with σ²<sub>ε</sub> = 4, σ²<sub>η</sub> = 1, a<sub>1</sub> = ${a1}, P<sub>1</sub> = 4; ${nr} rows${nr > 3 ? ' (t = 4–6 without data: the update step is skipped)' : ''}.`,
 kt1: (y, v, K, m, a) => `y<sub>3</sub> = ${y}: surprise v<sub>3</sub> = ${v}; the level moves by K<sub>3</sub>v<sub>3</sub> = ${K}·${v} = ${m}, to a<sub>3|3</sub> = ${a}.`,
 kt2: (a, P, F) => `Forecast for t = 4: a<sub>4</sub> = ${a}, P<sub>4</sub> = ${P}; the forecast of y<sub>4</sub> has variance P<sub>4</sub> + σ²<sub>ε</sub> = ${F}.`,
 wss1: (q_, d, p, P) => `q = ${q_}: P̄/σ²<sub>ε</sub> = (q + √(q² + 4q))/2 = (${q_} + √${d})/2 = ${p}; with σ²<sub>ε</sub> = 4, P̄ = ${P}.`,
 wss2: (P, K) => `K̄ = P̄/(P̄ + σ²<sub>ε</sub>) = ${P}/(${P} + 4) = <b>${K}</b>: the weight α of SES.`,
 wss3: (a, q_) => `Inverse map: the SES weight α = ${a} corresponds to q = α²/(1 − α) = <b>${q_}</b>.`,
 ws1: (q_, P, K, K3) => `q = ${q_}: P̄ = ${P}, K̄ = ${K} (with σ²<sub>ε</sub> = 4); on the way, K<sub>3</sub> = ${K3}.`,
 nf1: (q_, n, K) => `q = ${q_}: σ²<sub>η</sub> = ${n}, with σ²<sub>ε</sub> = 15100; steady-state gain K̄ = ${K}.`,
 nf2: (y, ap, v, K, af, a5) => `1899: flow ${y}, predicted level ${ap}, surprise v = ${v}; with K = ${K} the level falls to ${af} (${a5} in 1905).`,
 nf3: (a, sd, m) => `1970: level ${a} with standard error ${sd}; sample mean ${m}.`,
 nmCap: 'Local level model of the Nile: maximum likelihood with a diffuse start (own filter, lecture), statsmodels UnobservedComponents (exact diffuse start, lecture) and the concentrated profile likelihood, maximised in the browser.',
 nmNote: 'AIC = −2 log L + 2·2, BIC = −2 log L + 2 log n; the first observation is left out of log L (diffuse start). Another sample end is an extra run of the chapter’s code.',
 nm1: (e, q_, se, sn, ll) => `1871–${e}: q̂ = ${q_}, σ̂²<sub>ε</sub> = ${se}, σ̂²<sub>η</sub> = ${sn}, log L = ${ll}.`,
 nm2: K => `Steady-state gain of this fit: K̄ = ${K}.`,
 gs1: (k2, k5, nc, K) => `K<sub>t</sub> starts at ${k2} (1872, after the diffuse start), is ${k5} in year 5 and stays within 0.001 of K̄ = ${K} from year ${nc} on.`,
 gs2: (a, g5, g20) => `SES with α = ${a}: the forecasts differ from the Kalman forecasts by ${g5} at the 5th forecast and by at most ${g20} from the 20th on.`,
 gs3: (a, q_) => `SES fitted by least squares: α̂ = ${a}, i.e. q = α²/(1 − α) = ${q_}.`,
 wrs1: (yr, Pf, yr1, Pp, J) => `J<sub>${yr}</sub> = P<sub>t|t</sub>/P<sub>t+1</sub> = ${Pf}/${Pp} = <b>${J}</b>.`,
 wrs2: (yr, af, J, as1, ap1, as) => `α̂<sub>${yr}</sub> = a<sub>t|t</sub> + J<sub>t</sub>(α̂<sub>t+1</sub> − a<sub>t+1</sub>) = ${af} + ${J}·(${as1} − ${ap1}) = <b>${as}</b>.`,
 wrs3: (Pf, J, Ps1, Pp1, Ps, sds, sdf) => `V<sub>t</sub> = P<sub>t|t</sub> + J<sub>t</sub>²(V<sub>t+1</sub> − P<sub>t+1</sub>) = ${Pf} + ${J}²·(${Ps1} − ${Pp1}) = ${Ps}: standard error ${sds}, against ${sdf} for the filtered level.`,
 wr1: (yr, af, as, J) => `${yr}: filtered ${af}, smoothed ${as}, J<sub>t</sub> = ${J}.`,
 ns1: (q_, s97, f97, s00, f00) => `q = ${q_}. 1897: smoothed ${s97}, filtered ${f97}; 1900: smoothed ${s00}, filtered ${f00}.`,
 ns2: (s20, f20, sN, fN) => `Standard error in 1920: ${s20} smoothed, ${f20} filtered; in 1970: ${sN} smoothed, ${fN} filtered (at t = n they coincide).`,
 wls1: (yr, y, a, v, F) => `${yr}: y<sub>t</sub> = ${y}, a<sub>t</sub> = ${a}, v<sub>t</sub> = ${v}, F<sub>t</sub> = ${F}.`,
 wls2: (yr, c, lf, r, l) => `ℓ<sub>t</sub> = −½(log 2π + log F<sub>t</sub> + v<sub>t</sub>²/F<sub>t</sub>) = −½(${c} + ${lf} + ${r}) = <b>${l}</b>.`,
 wls3: (n, ll, F1) => `Sum over ${n} years (1871 is left out: F<sub>1</sub> = ${F1}, the diffuse start): log L = <b>${ll}</b>.`,
 wl1: (q_, ll, n) => `q = ${q_}: log L = ${ll}, from ${n} prediction errors.`,
 wl2: (yr, l) => `Smallest contribution: ${yr} (${l}).`,
 lk1: (q_, s2, ll) => `Maximum at q̂ = ${q_}, with σ̂²<sub>ε</sub> = ${s2} from the closed form; profile log-likelihood ${ll}.`,
 lk2: (q_, ll, lr, s2) => `q = ${q_}: profile log-likelihood ${ll}, σ̂²<sub>ε</sub>(q) = ${s2}; likelihood-ratio statistic against the maximum: ${lr}.`,
 dg1: (q_, lb, p, jb, pj, k, n) => `q = ${q_}: Ljung–Box Q(10) = ${lb} (p = ${p}), Jarque–Bera ${jb} (p = ${pj}); ${k} of ${n} errors outside ±1.96.`,
 dg2: (yl, l, yo, o) => `Minimum of the level residual: ${yl} (${l}); minimum of the observation residual: ${yo} (${o}).`,
 mi0: s => `No year removed: the standard error of the smoothed level in 1900 is ${s}.`,
 mi1: (L_, y, s, sf, n) => `${n} years removed (two gaps of ${L_} years): the standard error of the smoothed level in ${y} is ${s}, against ${sf} with all data.`,
 mi2: (q_, f, s1, H, sH) => `q = ${q_}: forecast ${f} for every horizon; the standard error of y is ${s1} one year ahead and ${sH} after ${H} years.`,
 whs1: (lam, z) => `λ = σ²<sub>ε</sub>/σ²<sub>ζ</sub> = ${lam}: with σ²<sub>ε</sub> = 1, σ²<sub>ζ</sub> = 1/λ = ${z}.`,
 whs2: () => `Smooth trend: μ<sub>t+1</sub> = μ<sub>t</sub> + β<sub>t</sub>, β<sub>t+1</sub> = β<sub>t</sub> + ζ<sub>t</sub>, so Δ²μ<sub>t+1</sub> = β<sub>t</sub> − β<sub>t−1</sub> = ζ<sub>t−1</sub>. Minus twice the log of the joint density, times σ²<sub>ε</sub>, is Σ(y<sub>t</sub> − μ<sub>t</sub>)² + (σ²<sub>ε</sub>/σ²<sub>ζ</sub>)Σ(Δ²μ<sub>t</sub>)²: the HP criterion with λ = σ²<sub>ε</sub>/σ²<sub>ζ</sub>.`,
 whs3: d => `Kalman smoother (diffuse start) against the penalised least-squares trend: largest absolute difference ${d}.`,
 wh1: (lam, sd, mx, q_) => `λ = ${lam}: standard deviation of the HP gap ${sd}; largest gap ${mx}% in ${q_}.`,
 rt1: (g, gx, qx, gn, qn) => `Average growth ${g}% per year; annualised growth of the UC trend between ${gn}% (${qn}) and ${gx}% (${qx}).`,
 rt2: (sz, p1, p2, sk, r) => `UC estimates: σ̂²<sub>ζ</sub> = ${sz} (trend slope), cycle AR(2) with φ̂<sub>1</sub> = ${p1}, φ̂<sub>2</sub> = ${p2}, σ̂²<sub>κ</sub> = ${sk}; ratio of the shock variances σ̂²<sub>κ</sub>/σ̂²<sub>ζ</sub> = ${r}.`,
 rt3: (lam, d) => `HP with λ = ${lam} (the ratio σ²<sub>ε</sub>/σ²<sub>ζ</sub> imposed): the largest distance from the UC trend is ${d} points.`,
 og1: (pu, qu, ph, pm, u, h, m) => `Peaks in 2007–2009: UC ${pu}% (${qu}), HP ${ph}%, Hamilton ${pm}%. 2020Q2: UC ${u}%, HP ${h}%, Hamilton ${m}%.`,
 og2: (a, b, c) => `Correlations: UC–HP ${a}, UC–Hamilton ${b}, HP–Hamilton ${c}.`,
 og3: (su, sh, sm, p, bs, lost) => `Standard deviations: UC ${su}, HP ${sh}, Hamilton ${sm}; the ${p === 1 ? 'regression coefficient is' : p + ' regression coefficients sum to'} ${bs}; the first ${lost} quarters are lost.`,
 og4: (q_, u, h, m) => `Latest gap (${q_}): UC ${u}%, HP ${h}%, Hamilton ${m}%.`,
 rl1: (q_, rt, fin, uf, us) => `Data up to ${q_}. 2008Q3: HP ${rt}% in real time, ${fin}% two-sided; UC ${uf}% filtered, ${us}% smoothed.`,
 rl2: (q_, rh, ru, ch, cu) => `2004Q4–${q_}: root mean square revision HP ${rh} points, UC ${ru} points; correlation real time–final HP ${ch}, UC ${cu}.`,
 tv1: (o, mx, dmx, mn, dmn) => `Constant OLS beta ${o}; smoothed beta: maximum ${mx} (${dmx}), minimum ${mn} (${dmn}).`,
 tv2: (lr, sd, ml) => ml ? `Likelihood ratio against a constant beta: ${lr}; weekly standard deviation of the beta shocks ${sd}.`
   : `σ<sub>β</sub> = ${sd} (σ²<sub>ε</sub> at its ML value): likelihood ratio against the constant beta ${lr} (extra run).`,
 tv3: (b, sd, w, sr, sk) => `Latest beta ${b} with standard error ${sd}. Weekly change: standard deviation ${sr} for the ${w}-week rolling OLS, ${sk} for the Kalman beta.`,
 df1: (e, ip, s, inc, p1, p2) => `Loadings: employment ${e}, industrial production ${ip}, sales ${s}, income ${inc}; factor AR(2) with φ̂<sub>1</sub> = ${p1}, φ̂<sub>2</sub> = ${p2}.`,
 df2: (mr, me, sh, mn, dmn) => `Mean of the factor: ${mr} in NBER recession months, ${me} in expansions; negative in ${sh} of recession months; minimum ${mn} (${dmn}).`,
 df3: (k, m, v, sl, td, dl) => `Value for ${m} with ${k} missing month${k === 1 ? '' : 's'}: ${v}. The FRED data downloaded on ${dl} have been revised since the slides were made: with two missing months the value is ${td}, the slide shows ${sl}.`,
 wds1: (p, q_, d1, p22, d2) => `E(D<sub>1</sub>) = 1/(1 − p<sub>11</sub>) = 1/${q_} = <b>${d1}</b> periods; E(D<sub>2</sub>) = 1/(1 − ${p22}) = <b>${d2}</b> periods.`,
 wds2: (a, b, pi) => `π<sub>1</sub> = (1 − p<sub>22</sub>)/(2 − p<sub>11</sub> − p<sub>22</sub>) = ${a}/${b} = <b>${pi}</b>.`,
 wds3: (h, a, b, c, d, l) => `P<sup>${h}</sup>: from regime 1, Pr(1) = ${a}, Pr(2) = ${b}; from regime 2, Pr(1) = ${c}, Pr(2) = ${d}; the distance from π<sub>1</sub> shrinks by the factor p<sub>11</sub> + p<sub>22</sub> − 1 = ${l} each period.`,
 wd1: (p11, d1, p22, d2, pi) => `p<sub>11</sub> = ${p11}: regime 1 lasts ${d1} periods on average; p<sub>22</sub> = ${p22}: ${d2} periods; ergodic probability π<sub>1</sub> = ${pi}.`,
 wfs1: (qp, a, b, p11, p21, pr) => `Prediction: Pr(S<sub>t</sub> = 1 | Y<sub>t−1</sub>) = p<sub>11</sub>·${a} + (1 − p<sub>22</sub>)·${b} = ${p11}·${a} + ${p21}·${b} = <b>${pr}</b> (the filtered probabilities of ${qp}).`,
 wfs2: (y, m1, m2, s, d1, d2) => `Densities at y<sub>t</sub> = ${y}: f<sub>1</sub> = φ((y<sub>t</sub> − (${m1}))/${s})/${s} = ${d1}, f<sub>2</sub> = φ((y<sub>t</sub> − ${m2})/${s})/${s} = ${d2}.`,
 wfs3: (j1, j2, fl) => `Update: Pr(S<sub>t</sub> = 1 | Y<sub>t</sub>) = ${j1}/(${j1} + ${j2}) = <b>${fl}</b>.`,
 wf1: (q_, y, pr, fl, s) => `${q_}: growth ${y}%; probability of recession: predicted ${pr}, filtered ${fl}, smoothed ${s}.`,
 mtCap: 'MS-AR(4) with switching mean, US real GDP growth, 1951Q2–1984Q4 (the first four quarters are lags); two maxima of the likelihood, reached from two starting values.',
 mt1: (m, k, n, h, nr) => `Maximum ${m}: ${k} of ${n} quarters have a smoothed probability above 0.5; it finds ${h} of the ${nr} NBER recession quarters.`,
 mu1: (m1, m2, s, p11, p22) => `Regime means: ${m1}% (recession) and ${m2}% (expansion), σ̂ = ${s}; p<sub>11</sub> = ${p11}, p<sub>22</sub> = ${p22}.`,
 mu2: (d1, d2, pi) => `Expected durations: ${d1} quarters of recession, ${d2} quarters of expansion; ergodic share of recession ${pi}.`,
 mu3: (c, h, f) => `Agreement with the NBER, 1947–2019: ${c} of quarters; ${h} of recession quarters detected, ${f} false alarms.`,
 mu4: (p, mx) => `2020Q2: probability ${p}; after 2021 at most ${mx}.`,
 mf0: 'No NBER recession in this window.',
 mf1: (a, b, r, f, s) => `${a}–${b}: the first NBER recession ${r ? 'starts in ' + r : 'is absent'}; the filtered probability crosses 0.5 ${f ? 'first in ' + f : 'never'}, the smoothed one ${s ? 'in ' + s : 'never'}.`,
 mf2: (q_, f, s) => `In ${q_}: filtered ${f}, smoothed ${s}.`,
 mf3: (c, h, late) => `Filtered probabilities against the NBER, 1947–2019: ${c} of quarters, ${h} of recession quarters.${late ? ' After 2019: the parameters of 1947–2019 (extra run).' : ''}`,
 mc1: (e, s2, s1, d2, d1, sw, c, f) => `Switching variance, 1947Q2–${e}: standard deviations ${s2} and ${s1}; durations ${d2} and ${d1} quarters; first calm quarter ${sw}; agreement with the NBER ${c} (${f} false alarms).`,
 mc2: (m1, m2, d1, d2, n1, q_) => `Switching mean: regime-1 mean ${m1}%, regime 2 ${m2}%; durations ${d1} and ${d2} quarters; ${n1} quarter${n1 === 1 ? '' : 's'} in regime 1${q_ ? ': ' + q_ : ''}.`,
 mr1: (m2, s2, m1, s1) => `Stable regime: mean ${m2}% per quarter, standard deviation ${s2}; volatile regime: mean ${m1}%, standard deviation ${s1}.`,
 mr2: (p, d2, d1, sh, pi) => `p<sub>11</sub> = ${p}: expected durations ${d2} quarters stable and ${d1} quarters volatile; ${sh} of the quarters are volatile (ergodic ${pi}).`,
 mr3: s => `Volatile spells: ${s}.`,
 mr4: p => `Latest probability of the volatile regime: ${p}.`,
 vr1: (v2, m2, v1, m1) => `Calm regime: volatility ${v2}% per year, mean ${m2}% per week; turbulent regime: ${v1}% per year, mean ${m1}% per week.`,
 vr2: (p, d2, d1, sh, pi) => `p<sub>11</sub> = ${p}: durations ${d2} weeks calm and ${d1} weeks turbulent; turbulent in ${sh} of the weeks (ergodic ${pi}).`,
 vr3: s => `Most turbulent years: ${s}.`,
 vr4: (c, a, b) => `Correlation with the GARCH(1,1) volatility: ${c}; GARCH: α̂ = ${a}, β̂ = ${b}.`,
 rm1: (a1, a26, M, s1, s26) => `Data: ACF of |r<sub>t</sub>| ${a1} at lag 1, ${a26} at 26 weeks; mean of ${M} simulated path${M === 1 ? '' : 's'}: ${s1}${s26 ? ' and ' + s26 : ''}.`,
 rm2: (d, M, ds, sd, di) => `Local Whittle d̂ of |r<sub>t</sub>|: ${d} in the data, ${ds} (standard deviation ${sd}) over ${M} simulated path${M === 1 ? '' : 's'}, ${di} in random order.`,
 rm3: (k, ab, mn) => `GARCH(1,1) on the first ${k} simulated paths: α + β = ${ab} on average (minimum ${mn}), although the paths contain no GARCH.`,
 mpCap: 'Two-regime models of the chapter, estimated with statsmodels (MarkovRegression); regime 1 as in the lecture. Durations: E(D) = 1/(1 − p<sub>ii</sub>).',
 mp1: (h, s) => `Probability that a spell of regime 1 lasts more than ${h} periods: ${s}.`,
};
})();
""",
}

# Question for the room and answer of every block; RO and EN say the same, with the numbers of the page (verified by a node script).
QA = {
 'ro': {
  'ss-examples': ('Cu T = 200, care traiectorie are autocorelația de ordinul 1 a lui Δy<sub>t</sub> cea mai apropiată de −0,5?',
                  'Cea cu q = 0,01: −0,51 (teoretic −1/2,01 = −0,50); pentru q = 1 autocorelația este −0,28 (teoretic −1/3). Cînd zgomotul domină, Δy<sub>t</sub> ≈ ε<sub>t</sub> − ε<sub>t−1</sub>, un MA(1) cu ρ(1) = −1/2; prima diferență a unui mers aleator fără zgomot ar fi necorelată.'),
  'wx-arma': ('Cît este F<sub>1</sub>, varianța primei erori de predicție a filtrului pentru AR(2)?',
              '0,858 = γ(0), varianța staționară a AR(2), elementul (1,1) al lui P<sub>1</sub>. F<sub>2</sub> = 0,748, iar din t = 3, cu două laguri cunoscute, F<sub>t</sub> = σ̂² = 0,739. Un model staționar nu are nevoie de pornire difuză.'),
  'wx-kf': ('▶ crește σ²<sub>ε</sub>. La σ²<sub>ε</sub> = 8 (cu P<sub>1</sub> = 4), cît este K<sub>1</sub>?',
            'K<sub>1</sub> = 4/(4 + 8) = 0,333: cu măsurători mai zgomotoase filtrul se bazează mai puțin pe observație, iar nivelul filtrat rămîne mai aproape de predicție (a<sub>1|1</sub> = 10,67 în loc de 11).'),
  'kf-table': ('Puneți y<sub>3</sub> = 18 (surpriza devine 7 în loc de 3). Cu cît se mută nivelul la t = 3?',
               'Cu K<sub>3</sub>v<sub>3</sub> = 0,404·7 = 2,83, pînă la a<sub>3|3</sub> = 13,83. Cîștigul K<sub>3</sub> = 2,71/6,71 = 0,404 nu depinde de date, ci doar de P<sub>3</sub> și σ²<sub>ε</sub>: o surpriză mai mare mută nivelul proporțional, nu pînă la observație.'),
  'wx-ss': ('▶ duce q de la 0,01 la 10. Cît este K̄ la q = 10?',
            'P̄/σ²<sub>ε</sub> = (10 + √140)/2 = 10,92, deci K̄ = 10,92/11,92 = 0,916. K̄ rămîne sub 1 pentru orice q finit; q/(q + 1) = 0,909 nu este starea de echilibru.'),
  'nile-filter': ('▶ duce q la 1. Unde ajunge nivelul filtrat în 1899, anul cu debitul 774?',
                  'La 899, față de 1037 la q̂ = 0,097: cu q = 1, K̄ = 0,618, iar nivelul preia aproape două treimi din surpriză (757 în 1905). Nivelul se adaptează mai repede la baraj, dar urmărește și anii izolați.'),
  'nile-ml': ('Tăiați eșantionul la 1920. q̂ crește sau scade față de eșantionul complet?',
              'Crește, la 0,163 (0,097 pe 1871–1970), cu σ̂²<sub>ε</sub> = 19103. Pe 50 de ani ruptura din 1899 cîntărește mai mult, iar verosimilitatea o atribuie mișcării nivelului.'),
  'gain-ses': ('▶ duce q la 1. Din ce an se află K<sub>t</sub> la mai puțin de 0,001 de K̄?',
               'Din anul 5 (din anul 11 la q̂ = 0,097 și din anul 17 la q = 0,03). Cu q mare, P<sub>t</sub> se stabilizează repede: fiecare observație nouă domină incertitudinea rămasă de la pornirea difuză.'),
  'wx-rts': ('Mutați anul la 1898. Cît este diferența dintre nivelul filtrat și cel netezit?',
             'Filtrat 1133, netezit 1000: o diferență de 134, mai mare decît în 1897 (1145 față de 1038, adică 107). Netezitorul folosește anii slabi de după 1899, deci coboară nivelul înainte ca scăderea să apară în date.'),
  'nile-smooth': ('Puneți q = 0,001 (un nivel aproape constant). Cum arată nivelul netezit în jurul anului 1899?',
                  'Aproape o dreaptă, care coboară lent de la 956 în 1897 la 881 în 1950: treapta dispare. Cu q mic modelul atribuie scăderea zgomotului, iar netezitorul nu poate arăta o ruptură pe care modelul o exclude.'),
  'wx-lik': ('La q̂ = 0,097, care an aduce termenul cel mai negativ în log L?',
             '1913 (ℓ<sub>t</sub> = −9,78), un singur an foarte secetos; urmează 1916 (−9,18) și 1899 (−9,02). Anul 1871 nu intră în sumă: pornirea difuză îi dă un F<sub>t</sub> uriaș.'),
  'likelihood': ('Duceți q-ul ales la 10. Cît este statistica raportului de verosimilitate față de maxim?',
                 '23,3, la fel de departe de date ca q = 0,001 (23,1); q = 1 dă 7,2. Un nivel care urmărește fiecare viitură prognozează la un pas la fel de prost ca un nivel aproape constant.'),
  'diagnostics': ('Puneți q = 0,001. Mai trece testul Ljung–Box Q(10)?',
                  'Nu: Q(10) = 20,9, p = 0,022. Un nivel aproape constant lasă scăderea de după 1899 în erorile de predicție, care devin autocorelate; la q̂ = 0,097, p = 0,21.'),
  'missing': ('Goluri de 10 ani (1891–1900 și 1931–1940): cît este eroarea standard a nivelului netezit în 1895?',
              '78, față de 99 în 1900 cu goluri de 20 de ani și 48 cu toate datele. Banda crește cu distanța pînă la cel mai apropiat an observat.'),
  'wx-hp': ('▶ duce λ la 100000. Cît devine abaterea standard a deviației HP?',
            '5,5, față de 2,9 la λ = 1600 și 1,3 la λ = 1: un trend mai rigid lasă mai mult din mișcarea PIB-ului în deviație.'),
  'ro-trend': ('Care λ din grilă aduce trendul HP cel mai aproape de trendul UC?',
               'λ = 400: cea mai mare distanță este 0,63 puncte (1,01 la λ = 100 și 2,65 la λ = 1600). Trendul UC este mai flexibil decît trendul impus de λ = 1600, de aceea un λ mai mic îl aproximează mai bine.'),
  'output-gap': ('▶ duce h la 4. Cum se schimbă abaterea standard a deviației Hamilton?',
                 'Scade la 4,6 (6,5 la h = 8 și 7,7 la h = 12): reziduul regresiei conține surprizele unui orizont mai scurt.'),
  'realtime': ('Alegeți datele pînă în T3 2008. Cît este deviația HP bilaterală în T3 2008?',
               '2,2%, exact valoarea în timp real: la capătul eșantionului filtrul bilateral devine unilateral. Cu datele pînă în T2 2026, deviația din T3 2008 este 9,5%.'),
  'tvp-beta': ('Puneți varianța șocurilor lui beta la 0. Ce devine beta Kalman?',
               'O dreaptă orizontală la 0,58, beta OLS: cu σ²<sub>β</sub> = 0 filtrul Kalman calculează recursiv estimarea OLS, iar netezitorul dă fiecărei săptămîni estimarea pe tot eșantionul.'),
  'dfm': ('Fără luni lipsă (k = 0), cît este valoarea factorului din iulie 2026?',
          '−0,78 cu datele FRED de azi, față de −0,87 cu două luni lipsă (slide-ul, cu datele de atunci: −0,9). Factorul folosește ce a sosit: lipsa venitului și a vînzărilor mută ultima valoare, nu o face imposibil de calculat.'),
  'wx-dur': ('Puneți p<sub>11</sub> = 0,5. Cît durează în medie regimul 1?',
             '1/(1 − 0,5) = 2 perioade. p<sub>11</sub>/(1 − p<sub>11</sub>) = 1 numără doar perioadele de după prima, iar 1/p<sub>11</sub> nu este o durată.'),
  'wx-ham': ('▶ ajunge la T4 2008 (creștere −2,21%). Cît este probabilitatea filtrată a recesiunii?',
             '0,993: predicția era 0,349, dar f<sub>1</sub>(−2,21) este mult mai mare decît f<sub>2</sub>(−2,21), iar regula lui Bayes duce probabilitatea aproape la 1.'),
  'ms-table': ('Ridicați pragul la 0,7. Ce pondere din trimestrele de recesiune NBER mai găsește maximul A?',
               '90% (94% la pragul 0,5); maximul B rămîne la 10%. Probabilitățile netezite ale maximului A sînt aproape 0 sau 1, deci pragul schimbă puțin.'),
  'ms-us': ('Coborîți p<sub>11</sub> la 0,3. Ce devin durata recesiunii și ponderea trimestrelor NBER detectate?',
            '1,4 trimestre și 39,6% (3,2 trimestre și 62,5% la p̂<sub>11</sub> = 0,685). Un regim mai puțin persistent cere date mai slabe pentru fiecare trimestru de recesiune, deci recesiunile scurte și ușoare nu mai sînt găsite; T2 2020 rămîne la probabilitatea 1,00.'),
  'ms-filtered': ('Alegeți fereastra 2000–2002 (recesiunea ușoară din 2001). Trece probabilitatea filtrată de 0,5?',
                  'Nu: maximul ei este 0,38, iar nici cea netezită nu trece de 0,5. Recesiunea din 2001 a fost prea ușoară pentru media de −0,42% pe trimestru a regimului de recesiune.'),
  'ms-checks': ('Încheiați eșantionul în T2 2020. Cîte trimestre formează regimul cu media scăzută?',
                'Unul singur, T2 2020 (media −8,2%); pe 1947–2019 regimul cuprinde 30 de trimestre. O singură observație extremă formează singură un regim.'),
  'ms-ro': ('Coborîți p<sub>11</sub> al regimului volatil la 0,5. Ce pondere din trimestre rămîne volatilă?',
            '29% (40% la p̂<sub>11</sub> = 0,911): un regim volatil mai puțin persistent păstrează doar trimestrele cu creșteri sau scăderi mari.'),
  'vol-regimes': ('Puneți p<sub>11</sub> = 0,5 pentru regimul agitat. Ce pondere din săptămîni rămîne agitată?',
                  '12% (26% la p̂<sub>11</sub> = 0,930). p<sub>11</sub> = 0,5 nu înseamnă o clasificare la întîmplare: un regim agitat care durează în medie două săptămîni rămîne doar pentru săptămînile extreme.'),
  'regimes-memory': ('▶ adaugă traiectorii simulate, de la 1 la 40. Unde se stabilizează media ACF la lagul 1?',
                     'În jur de 0,21, sub 0,31 din date: regimurile explică o mare parte din clustering-ul pe termen scurt, nu tot. Prima traiectorie dă 0,26, a doua 0,19: media pe 40 de traiectorii elimină variația de eșantionare.'),
  'ms-params': ('Care regim durează mai mult în timp calendaristic: regimul calm al S&P 500 (36,8 perioade) sau expansiunea din SUA (19,7 perioade)?',
                'Expansiunea din SUA: 19,7 trimestre = 4,9 ani, față de 36,8 săptămîni = 0,7 ani. Durata 1/(1 − p<sub>ii</sub>) este în unitatea de timp a datelor.'),
 },
 'en': {
  'ss-examples': ('With T = 200, which path has the lag-1 autocorrelation of Δy<sub>t</sub> closest to −0.5?',
                  'The one with q = 0.01: −0.51 (theory −1/2.01 = −0.50); for q = 1 the autocorrelation is −0.28 (theory −1/3). When the noise dominates, Δy<sub>t</sub> ≈ ε<sub>t</sub> − ε<sub>t−1</sub>, an MA(1) with ρ(1) = −1/2; the first difference of a random walk without noise would be uncorrelated.'),
  'wx-arma': ('What is F<sub>1</sub>, the variance of the first prediction error of the AR(2) filter?',
              '0.858 = γ(0), the stationary variance of the AR(2), the (1,1) element of P<sub>1</sub>. F<sub>2</sub> = 0.748, and from t = 3 on, with two lags known, F<sub>t</sub> = σ̂² = 0.739. A stationary model needs no diffuse start.'),
  'wx-kf': ('▶ raises σ²<sub>ε</sub>. At σ²<sub>ε</sub> = 8 (with P<sub>1</sub> = 4), what is K<sub>1</sub>?',
            'K<sub>1</sub> = 4/(4 + 8) = 0.333: with noisier measurements the filter relies less on the observation, and the filtered level stays closer to the prediction (a<sub>1|1</sub> = 10.67 instead of 11).'),
  'kf-table': ('Set y<sub>3</sub> = 18 (the surprise becomes 7 instead of 3). How far does the level move at t = 3?',
               'By K<sub>3</sub>v<sub>3</sub> = 0.404·7 = 2.83, to a<sub>3|3</sub> = 13.83. The gain K<sub>3</sub> = 2.71/6.71 = 0.404 does not depend on the data, only on P<sub>3</sub> and σ²<sub>ε</sub>: a larger surprise moves the level proportionally, not to the observation.'),
  'wx-ss': ('▶ moves q from 0.01 to 10. What is K̄ at q = 10?',
            'P̄/σ²<sub>ε</sub> = (10 + √140)/2 = 10.92, so K̄ = 10.92/11.92 = 0.916. K̄ stays below 1 for every finite q; q/(q + 1) = 0.909 is not the steady state.'),
  'nile-filter': ('▶ takes q to 1. Where is the filtered level in 1899, the year with flow 774?',
                  'At 899, against 1037 at q̂ = 0.097: with q = 1, K̄ = 0.618, and the level takes almost two thirds of the surprise (757 in 1905). The level adapts to the dam faster, but it also follows isolated years.'),
  'nile-ml': ('Cut the sample at 1920. Does q̂ rise or fall compared to the full sample?',
              'It rises, to 0.163 (0.097 on 1871–1970), with σ̂²<sub>ε</sub> = 19103. Over 50 years the 1899 break weighs more, and the likelihood attributes it to movement of the level.'),
  'gain-ses': ('▶ takes q to 1. From which year on is K<sub>t</sub> within 0.001 of K̄?',
               'From year 5 (from year 11 at q̂ = 0.097 and from year 17 at q = 0.03). With a large q, P<sub>t</sub> settles quickly: every new observation dominates the uncertainty left by the diffuse start.'),
  'wx-rts': ('Move the year to 1898. What is the difference between the filtered and the smoothed level?',
             'Filtered 1133, smoothed 1000: a difference of 134, larger than in 1897 (1145 against 1038, i.e. 107). The smoother uses the low years after 1899, so it lowers the level before the drop appears in the data.'),
  'nile-smooth': ('Set q = 0.001 (an almost constant level). What does the smoothed level look like around 1899?',
                  'Almost a straight line, falling slowly from 956 in 1897 to 881 in 1950: the step disappears. With a small q the model attributes the drop to noise, and the smoother cannot show a break that the model rules out.'),
  'wx-lik': ('At q̂ = 0.097, which year contributes the most negative term to log L?',
             '1913 (ℓ<sub>t</sub> = −9.78), a single very dry year; then 1916 (−9.18) and 1899 (−9.02). The year 1871 is not in the sum: the diffuse start gives it a huge F<sub>t</sub>.'),
  'likelihood': ('Take the chosen q to 10. What is the likelihood-ratio statistic against the maximum?',
                 '23.3, as far from the data as q = 0.001 (23.1); q = 1 gives 7.2. A level that chases every flood predicts one step ahead as badly as an almost constant level.'),
  'diagnostics': ('Set q = 0.001. Is the Ljung–Box Q(10) test still passed?',
                  'No: Q(10) = 20.9, p = 0.022. An almost constant level leaves the drop after 1899 in the prediction errors, which become autocorrelated; at q̂ = 0.097, p = 0.21.'),
  'missing': ('Gaps of 10 years (1891–1900 and 1931–1940): what is the standard error of the smoothed level in 1895?',
              '78, against 99 in 1900 with 20-year gaps and 48 with all data. The band grows with the distance to the nearest observed year.'),
  'wx-hp': ('▶ takes λ to 100000. What becomes of the standard deviation of the HP gap?',
            '5.5, against 2.9 at λ = 1600 and 1.3 at λ = 1: a stiffer trend leaves more of the movement of GDP in the gap.'),
  'ro-trend': ('Which λ of the grid brings the HP trend closest to the UC trend?',
               'λ = 400: the largest distance is 0.63 points (1.01 at λ = 100 and 2.65 at λ = 1600). The UC trend is more flexible than the trend imposed by λ = 1600, so a smaller λ approximates it better.'),
  'output-gap': ('▶ takes h to 4. How does the standard deviation of the Hamilton gap change?',
                 'It falls to 4.6 (6.5 at h = 8 and 7.7 at h = 12): the residual of the regression contains the surprises of a shorter horizon.'),
  'realtime': ('Choose the data up to 2008Q3. What is the two-sided HP gap in 2008Q3?',
               '2.2%, exactly the real-time value: at the end of the sample the two-sided filter becomes one-sided. With the data up to 2026Q2, the gap of 2008Q3 is 9.5%.'),
  'tvp-beta': ('Set the variance of the beta shocks to 0. What becomes of the Kalman beta?',
               'A flat line at 0.58, the OLS beta: with σ²<sub>β</sub> = 0 the Kalman filter computes the OLS estimate recursively, and the smoother gives every week the full-sample estimate.'),
  'dfm': ('Without missing months (k = 0), what is the value of the factor for July 2026?',
          '−0.78 with today’s FRED data, against −0.87 with two missing months (the slide, with the data of that time: −0.9). The factor uses whatever has arrived: missing income and sales move the latest value, they do not make it impossible to compute.'),
  'wx-dur': ('Set p<sub>11</sub> = 0.5. How long does regime 1 last on average?',
             '1/(1 − 0.5) = 2 periods. p<sub>11</sub>/(1 − p<sub>11</sub>) = 1 counts only the periods after the first, and 1/p<sub>11</sub> is not a duration.'),
  'wx-ham': ('▶ reaches 2008Q4 (growth −2.21%). What is the filtered probability of recession?',
             '0.993: the prediction was 0.349, but f<sub>1</sub>(−2.21) is much larger than f<sub>2</sub>(−2.21), and Bayes’ rule takes the probability close to 1.'),
  'ms-table': ('Raise the threshold to 0.7. Which share of the NBER recession quarters does maximum A still find?',
               '90% (94% at the threshold 0.5); maximum B stays at 10%. The smoothed probabilities of maximum A are close to 0 or 1, so the threshold changes little.'),
  'ms-us': ('Lower p<sub>11</sub> to 0.3. What become the duration of a recession and the share of NBER quarters detected?',
            '1.4 quarters and 39.6% (3.2 quarters and 62.5% at p̂<sub>11</sub> = 0.685). A less persistent regime needs weaker data for every recession quarter, so short and mild recessions are no longer found; 2020Q2 keeps the probability 1.00.'),
  'ms-filtered': ('Choose the window 2000–2002 (the mild recession of 2001). Does the filtered probability cross 0.5?',
                  'No: its maximum is 0.38, and the smoothed one does not cross 0.5 either. The 2001 recession was too mild for the mean of −0.42% per quarter of the recession regime.'),
  'ms-checks': ('End the sample in 2020Q2. How many quarters form the low-mean regime?',
                'One, 2020Q2 (mean −8.2%); on 1947–2019 the regime holds 30 quarters. A single extreme observation forms a regime on its own.'),
  'ms-ro': ('Lower p<sub>11</sub> of the volatile regime to 0.5. Which share of the quarters stays volatile?',
            '29% (40% at p̂<sub>11</sub> = 0.911): a less persistent volatile regime keeps only the quarters with large rises or falls.'),
  'vol-regimes': ('Set p<sub>11</sub> = 0.5 for the turbulent regime. Which share of the weeks stays turbulent?',
                  '12% (26% at p̂<sub>11</sub> = 0.930). p<sub>11</sub> = 0.5 does not mean a random classification: a turbulent regime that lasts two weeks on average remains only for the extreme weeks.'),
  'regimes-memory': ('▶ adds simulated paths, from 1 to 40. Where does the mean ACF at lag 1 settle?',
                     'Around 0.21, below the 0.31 of the data: the regimes explain much of the short-run clustering, not all of it. The first path gives 0.26, the second 0.19: the mean over 40 paths removes the sampling variation.'),
  'ms-params': ('Which regime lasts longer in calendar time: the calm regime of the S&P 500 (36.8 periods) or the US expansion (19.7 periods)?',
                'The US expansion: 19.7 quarters = 4.9 years, against 36.8 weeks = 0.7 years. The duration 1/(1 − p<sub>ii</sub>) is in the time unit of the data.'),
 },
}

# Predictions „Ce credeți?” / "What do you think?": (index of the correct option, [(RO, EN), ...]); every wrong option encodes one misconception.
PREDICT = {
 'ss-examples': (0, [('Cea cu q = 0,01: −0,51.', 'The one with q = 0.01: −0.51.'), ('Cea cu q = 1: nivelul se mișcă mai mult, deci diferențele sînt mai corelate.', 'The one with q = 1: its level moves more, so the differences are more correlated.'),
                     ('Niciuna: diferențele unui mers aleator sînt necorelate.', 'None: the differences of a random walk are uncorrelated.')]),
 'wx-arma': (1, [('0,739 = σ̂², varianța șocului.', '0.739 = σ̂², the shock variance.'), ('0,858 = γ(0), varianța AR(2).', '0.858 = γ(0), the variance of the AR(2).'),
                 ('10⁷: o pornire difuză, ca la Nil.', '10⁷: a diffuse start, as for the Nile.')]),
 'wx-kf': (0, [('0,333 = 4/12.', '0.333 = 4/12.'), ('0,667: datele mai zgomotoase cer o corecție mai mare.', '0.667: noisier data need a larger correction.'),
               ('Tot 0,5: cîștigul depinde doar de P₁.', 'Still 0.5: the gain depends on P₁ only.')]),
 'kf-table': (2, [('Cu 7, pînă la 18: filtrul urmează noua observație.', 'By 7, to 18: the filter follows the new observation.'), ('Tot cu 1,21: cîștigul fixează mărimea mutării.', 'Still by 1.21: the gain fixes the size of the move.'),
                  ('Cu 2,83, pînă la 13,83.', 'By 2.83, to 13.83.')]),
 'wx-ss': (1, [('1: cu q = 10 filtrul copiază datele.', '1: with q = 10 the filter copies the data.'), ('0,916.', '0.916.'), ('0,909 = q/(q + 1).', '0.909 = q/(q + 1).')]),
 'nile-filter': (2, [('Tot la 1037: surpriza din 1899 este aceeași.', 'Still 1037: the surprise of 1899 is the same.'), ('La 774: cu q = 1 nivelul este chiar observația.', 'At 774: with q = 1 the level is the observation itself.'),
                     ('La circa 899.', 'At about 899.')]),
 'nile-ml': (0, [('Crește, la circa 0,16.', 'It rises, to about 0.16.'), ('Scade: mai puțini ani dau un nivel mai neted.', 'It falls: fewer years give a smoother level.'),
                 ('Rămîne 0,097: q este o proprietate a fluviului.', 'It stays at 0.097: q is a property of the river.')]),
 'gain-ses': (1, [('Tot din anul 11: convergența nu depinde de q.', 'Still from year 11: the convergence does not depend on q.'), ('Din anul 5.', 'From year 5.'),
                  ('După anul 11: un cîștig mai mare converge mai încet.', 'After year 11: a larger gain converges more slowly.')]),
 'wx-rts': (0, [('Netezitul este cu circa 134 sub filtrat, mai mult decît în 1897.', 'The smoothed level is about 134 below the filtered one, more than in 1897.'),
                ('Coincid: scăderea nu a avut loc încă în 1898.', 'They coincide: the drop has not happened yet in 1898.'), ('Netezitul este peste filtrat.', 'The smoothed level is above the filtered one.')]),
 'nile-smooth': (0, [('Aproape o dreaptă, de la 956 în 1897 la 881 în 1950: fără treaptă.', 'Almost a straight line, from 956 in 1897 to 881 in 1950: no step.'),
                     ('O treaptă mai clară: cu mai puțin zgomot de nivel netezitorul vede mai bine ruptura.', 'A sharper step: with less level noise the smoother sees the break better.'),
                     ('La fel ca nivelul filtrat.', 'The same as the filtered level.')]),
 'wx-lik': (2, [('1899, anul rupturii.', '1899, the year of the break.'), ('1871, primul an.', '1871, the first year.'), ('1913.', '1913.')]),
 'likelihood': (0, [('23,3: la fel de departe ca q = 0,001.', '23.3: as far as q = 0.001.'), ('7,2, ca la q = 1.', '7.2, as for q = 1.'),
                    ('Peste 100: q = 10 face nivelul să copieze fiecare viitură.', 'Above 100: q = 10 makes the level copy every flood.')]),
 'diagnostics': (2, [('Da, p tot 0,21: testul nu depinde de q.', 'Yes, p still 0.21: the test does not depend on q.'), ('Da, p peste 0,5: un nivel mai neted dă erori mai curate.', 'Yes, p above 0.5: a smoother level gives cleaner errors.'),
                     ('Nu: p = 0,022.', 'No: p = 0.022.')]),
 'missing': (1, [('99, ca la golurile de 20 de ani.', '99, as with 20-year gaps.'), ('78.', '78.'), ('48, ca la datele complete: netezitorul umple golul.', '48, as with all data: the smoother fills the gap.')]),
 'wx-hp': (0, [('5,5: un trend mai rigid lasă mai mult în deviație.', '5.5: a stiffer trend leaves more in the gap.'), ('Aproape 0: un λ mare elimină ciclul.', 'Close to 0: a large λ removes the cycle.'),
               ('Tot 2,9, ca la λ = 1600.', 'Still 2.9, as at λ = 1600.')]),
 'ro-trend': (1, [('λ = 1600, convenția pentru date trimestriale.', 'λ = 1600, the convention for quarterly data.'), ('λ = 400.', 'λ = 400.'),
                  ('λ = 100000: cel mai neted trend.', 'λ = 100000: the smoothest trend.')]),
 'output-gap': (2, [('Crește: un orizont mai scurt dă o deviație mai zgomotoasă.', 'It rises: a shorter horizon gives a noisier gap.'), ('Rămîne 6,5: coeficienții regresiei se adaptează.', 'It stays at 6.5: the regression coefficients adapt.'),
                    ('Scade la 4,6.', 'It falls to 4.6.')]),
 'realtime': (0, [('2,2%, valoarea în timp real.', '2.2%, the real-time value.'), ('9,5%, ca în curs.', '9.5%, as in the lecture.'), ('0: deviația HP este zero la capătul eșantionului.', '0: the HP gap is zero at the end of the sample.')]),
 'tvp-beta': (1, [('Beta OLS pe ferestre de 52 de săptămîni.', 'The 52-week rolling OLS beta.'), ('O dreaptă la 0,58, beta OLS.', 'A flat line at 0.58, the OLS beta.'),
                  ('Zero: fără șocuri, beta dispare.', 'Zero: without shocks, the beta disappears.')]),
 'dfm': (2, [('−0,87, ca la două luni lipsă: lunile lipsă nu schimbă factorul.', '−0.87, as with two missing months: missing months do not change the factor.'),
             ('Nu se poate calcula pînă nu sosesc toate cele patru serii.', 'It cannot be computed until all four series have arrived.'), ('−0,78.', '−0.78.')]),
 'wx-dur': (0, [('2 perioade.', '2 periods.'), ('1 = p₁₁/(1 − p₁₁).', '1 = p₁₁/(1 − p₁₁).'), ('0,5 perioade.', '0.5 periods.')]),
 'wx-ham': (1, [('Circa 0,35, predicția: un singur trimestru nu o mută mult.', 'About 0.35, the prediction: one quarter cannot move it much.'), ('0,993.', '0.993.'),
                ('0,47, ca în T3 2008.', '0.47, as in 2008Q3.')]),
 'ms-table': (0, [('90%.', '90%.'), ('Tot 94%: pragul nu contează.', 'Still 94%: the threshold does not matter.'), ('Sub 50%: un prag mai strict pierde majoritatea recesiunilor.', 'Below 50%: a stricter threshold loses most recessions.')]),
 'ms-us': (2, [('3,2 trimestre și 62,5%: p₁₁ schimbă doar duratele, nu probabilitățile.', '3.2 quarters and 62.5%: p₁₁ changes only the durations, not the probabilities.'),
               ('1,4 trimestre, tot 62,5%.', '1.4 quarters, still 62.5%.'), ('1,4 trimestre și 39,6%.', '1.4 quarters and 39.6%.')]),
 'ms-filtered': (0, [('Nu: maximul este 0,38.', 'No: its maximum is 0.38.'), ('Da, în T2 2001, primul trimestru NBER.', 'Yes, in 2001Q2, the first NBER quarter.'),
                     ('Doar cea netezită trece de 0,5.', 'Only the smoothed one crosses 0.5.')]),
 'ms-checks': (1, [('Circa 30, ca pe 1947–2019.', 'About 30, as on 1947–2019.'), ('Unul singur, T2 2020.', 'One, 2020Q2.'), ('Circa 50: anul 2020 adaugă o recesiune lungă.', 'About 50: 2020 adds a long recession.')]),
 'ms-ro': (2, [('Tot 40%: p₁₁ nu schimbă clasificarea.', 'Still 40%: p₁₁ does not change the classification.'), ('0%: cu p₁₁ = 0,5 regimul volatil dispare.', '0%: with p₁₁ = 0.5 the volatile regime disappears.'),
               ('29%.', '29%.')]),
 'vol-regimes': (1, [('50%: p₁₁ = 0,5 înseamnă o clasificare la întîmplare.', '50%: p₁₁ = 0.5 means a random classification.'), ('12%.', '12%.'), ('Tot 26%.', 'Still 26%.')]),
 'regimes-memory': (0, [('În jur de 0,21, sub 0,31 din date.', 'Around 0.21, below the 0.31 of the data.'), ('Urcă la 0,31: cu destule traiectorii modelul atinge datele.', 'It climbs to 0.31: with enough paths the model reaches the data.'),
                        ('Coboară spre 0: media anulează clustering-ul.', 'It falls towards 0: averaging cancels the clustering.')]),
 'ms-params': (1, [('Regimul calm al S&P 500: 36,8 perioade sînt mai multe decît 19,7.', 'The calm S&P 500 regime: 36.8 periods are more than 19.7.'), ('Expansiunea din SUA: 4,9 ani față de 0,7 ani.', 'The US expansion: 4.9 years against 0.7 years.'),
                   ('Sînt egale: ambele sînt 1/(1 − p₂₂).', 'They are equal: both are 1/(1 − p₂₂).')]),
}
VEIL_CHARTS = {'wx-arma', 'wx-ss', 'wx-lik', 'wx-rts', 'likelihood'}


# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slide „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('Modelul local level cu Pₜ = 3 și varianța zgomotului de măsurare 1. Cît este cîștigul Kalman?',
         'Local level model with Pₜ = 3 and a measurement-noise variance of 1. What is the Kalman gain?'), key=1, opts=[
  (('0,25 = 1/(3 + 1).', '0.25 = 1/(3 + 1).'),
   ('Aceasta este ponderea predicției, 1 − Kₜ, nu a observației.', 'That is the weight of the prediction, 1 − Kₜ, not of the observation.')),
  (('0,75 = 3/(3 + 1): observația nouă primește ponderea 0,75.', '0.75 = 3/(3 + 1): the new observation gets the weight 0.75.'),
   ('Kₜ = Pₜ/(Pₜ + σ²) = 3/4: starea este incertă în raport cu zgomotul, deci datele cîntăresc mult.', 'Kₜ = Pₜ/(Pₜ + σ²) = 3/4: the state is uncertain relative to the noise, so the data weigh a lot.')),
  (('3 = Pₜ/σ².', '3 = Pₜ/σ².'),
   ('Un raport de varianțe nu este o pondere: cîștigul unui model local level este între 0 și 1.', 'A ratio of variances is not a weight: the gain of a local level model lies between 0 and 1.'))]),
 dict(q=('Un model SES are α = 0,5. Ce model local level se află în spatele lui?',
         'An SES model has α = 0.5. Which local level model is behind it?'), key=2, opts=[
  (('q = α² = 0,25.', 'q = α² = 0.25.'),
   ('Lipsește numitorul: q = α²/(1 − α).', 'The denominator is missing: q = α²/(1 − α).')),
  (('q = α/(1 − α) = 1.', 'q = α/(1 − α) = 1.'),
   ('Raportul α/(1 − α) nu este relația inversă a stării de echilibru.', 'The ratio α/(1 − α) is not the inverse map of the steady state.')),
  (('q = α²/(1 − α) = 0,25/0,5 = 0,5.', 'q = α²/(1 − α) = 0.25/0.5 = 0.5.'),
   ('Relația inversă a lui K̄(q): cu q = 0,5, K̄ = 0,5.', 'The inverse of K̄(q): with q = 0.5, K̄ = 0.5.'))]),
 dict(q=('Într-un model cu două regimuri p₁₁ = 0,9. Cît durează în medie regimul 1?',
         'In a two-regime model p₁₁ = 0.9. How long does regime 1 last on average?'), key=0, opts=[
  (('1/(1 − 0,9) = 10 perioade.', '1/(1 − 0.9) = 10 periods.'),
   ('Durata este geometrică cu probabilitatea de ieșire 0,1: media 1/0,1 = 10.', 'The duration is geometric with exit probability 0.1: mean 1/0.1 = 10.')),
  (('0,9/(1 − 0,9) = 9 perioade.', '0.9/(1 − 0.9) = 9 periods.'),
   ('p/(1 − p) numără doar perioadele de după prima; slide-ul „Verificări necesare” avertizează asupra acestei greșeli.', 'p/(1 − p) counts only the periods after the first; the slide “Necessary checks” warns about this mistake.')),
  (('1/0,9 = 1,1 perioade.', '1/0.9 = 1.1 periods.'),
   ('1/p₁₁ ar însemna că un regim foarte persistent durează puțin.', '1/p₁₁ would mean that a very persistent regime is short.'))]),
]


def _predict_js():
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {})) for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch10-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH10_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH10_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n')


_predict_js()


if __name__ == '__main__':
    import interactive_common
    page_spec = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA', 'TEXT_JS', 'PREDICT', 'TICKET')})
    page_spec.widgets = lambda lang: {a: v for a, v in widgets_all(lang).items() if a in {b for b, _, _ in BLOCKS}}
    interactive_common.build(page_spec)
    for lg in ('en', 'ro'):
        _post(lg)
