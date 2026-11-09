"""Build the interactive pages of Chapter 7 (cointegration and VECM): interactive/ch7.html (EN) and interactive/ch7_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its interactive version on
the right. The result tables of the lecture (critical values, Engle-Granger and Phillips-Ouliaris tests, Johansen tests, the
deterministic cases, the adjustment coefficients) and its worked examples (common trends, the two Engle-Granger steps, the
half-life, one quarter of an ECM, a bivariate VECM, the Johansen statistics) get blocks of their own (EXTRA), placed by slide;
on the web lecture each of them travels inside a chart block and ch7.js moves it to its own frame. Logic in interactive/ch7.js,
inputs in interactive/ch7_data.js (Quantlets/Ch_07/export_interactive.py), slide images in interactive/slides/.
"""
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 7

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_07'
TEX = {'ro': 'RO/Cursuri/capitol7_cointegrare_vecm', 'en': 'EN/Courses/chapter7_cointegration_vecm'}

# (anchor, chart file, section key); order of the slides
BLOCKS = [('drunk', 'tsa_ch7_drunk_dog', 'idea'), ('examples', 'tsa_ch7_examples', 'idea'),
          ('eg-dist', 'tsa_ch7_eg_dist', 'eg'), ('eg-steps', 'tsa_ch7_eg_steps', 'eg'),
          ('ecm', 'tsa_ch7_ecm', 'ecm'),
          ('rates', 'tsa_ch7_rates', 'beta'), ('vecm-irf', 'tsa_ch7_vecm_irf', 'beta'),
          ('forecast', 'tsa_ch7_forecast', 'fc'),
          ('ppp', 'tsa_ch7_ppp', 'app'), ('ro-ea', 'tsa_ch7_ro_ea_rates', 'app'), ('cee-fx', 'tsa_ch7_cee_fx', 'app'),
          ('pair', 'tsa_ch7_pair', 'pairs'), ('pairs-bt', 'tsa_ch7_pairs_backtest', 'pairs')]

# result tables and worked examples: (anchor, frame title RO, frame title EN, section key, host chart block on the web lecture)
EXTRA = [('wx-trends', 'Trenduri stochastice comune', 'Common stochastic trends', 'idea', 'examples'),
         ('eg-crit', 'Valorile critice ale testului Engle–Granger', 'Critical values of the Engle–Granger test', 'eg', 'eg-dist'),
         ('wx-eg', 'Exemplu rezolvat: consumul și venitul în SUA', 'Worked example: US consumption and income', 'eg', 'eg-steps'),
         ('eg-table', 'Engle–Granger și Phillips–Ouliaris pe date reale', 'Engle–Granger and Phillips–Ouliaris on real data', 'eg', 'eg-steps'),
         ('wx-half', 'Viteza de ajustare și timpul de înjumătățire', 'Speed of adjustment and half-life', 'ecm', 'ecm'),
         ('wx-ecm', 'Exemplu rezolvat: un trimestru de consum', 'Worked example: one quarter of consumption', 'ecm', 'ecm'),
         ('wx-vecm', 'Exemplu rezolvat: un VECM cu două variabile', 'Worked example: a bivariate VECM', 'vecm', 'ecm'),
         ('wx-joh', 'Exemplu rezolvat: de la valori proprii la statistici', 'Worked example: from eigenvalues to statistics', 'joh', 'rates'),
         ('joh-table', 'Testele Johansen pe trei sisteme', 'Johansen tests on three systems', 'joh', 'rates'),
         ('joh-det', 'Sensibilitatea testelor Johansen la termenii determiniști', 'Sensitivity of the Johansen tests to the deterministic terms', 'joh', 'rates'),
         ('alpha-table', 'Coeficienții de ajustare', 'The adjustment coefficients', 'beta', 'rates')]

PFX = {'drunk': 'dd', 'examples': 'ex', 'eg-dist': 'ed', 'eg-steps': 'es', 'ecm': 'em', 'rates': 'ra', 'vecm-irf': 'ir', 'forecast': 'fc', 'ppp': 'pp',
       'ro-ea': 're', 'cee-fx': 'cx', 'pair': 'pr', 'pairs-bt': 'pb', 'wx-trends': 'ct', 'eg-crit': 'ec', 'wx-eg': 'we', 'eg-table': 'et', 'wx-half': 'wh',
       'wx-ecm': 'wc', 'wx-vecm': 'wv', 'wx-joh': 'wj', 'joh-table': 'jt', 'joh-det': 'jd', 'alpha-table': 'at'}

SECTIONS = {
 'ro': dict(idea='Ideea de cointegrare', eg='Metoda Engle–Granger în doi pași', ecm='Modele cu corecția erorii',
            vecm='Modelul VECM și teorema de reprezentare a lui Granger', joh='Testele Johansen', beta='Estimarea și interpretarea lui β și α',
            fc='Prognoza cu VECM și cu VAR în diferențe', app='Exemple economice: paritate și rate ale dobînzii', pairs='O aplicație: pairs trading'),
 'en': dict(idea='The idea of cointegration', eg='The Engle–Granger two-step method', ecm='Error correction models',
            vecm='The VECM and the Granger representation theorem', joh='The Johansen tests', beta='Estimating and interpreting β and α',
            fc='Forecasting with a VECM and with a VAR in differences', app='Economic examples: parity and interest rates', pairs='An application: pairs trading'),
}

L = {
 'ro': dict(file='ch7_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 7 · Cointegrare și modele VECM', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură sau același tabel, calculat în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas; '
        'casetele „Exemplu rezolvat” refac calculul cu cifrele capitolului.',
   src='Randamentele titlurilor de stat americane GS1, GS5, GS10 și TB3MS (FRED, lunar, ianuarie 1960 – septembrie 2026), cursurile EUR/RON, EUR/HUF și EUR/PLN (BNR, sfîrșit de lună, '
       'iulie 2005 – septembrie 2026), media lunară EUR/RON și IAPC (Eurostat, iulie 2005 – august 2026), ROBOR 3M și Euribor 3M (Eurostat, august 1995 – august 2026), consumul '
       'gospodăriilor și PIB-ul României (Eurostat, T1 1995 – T2 2026, versiunea din curs, citită de pe linia graficului din seminar, fără puncte omise). Randamentele, cursurile BNR, IAPC, ROBOR și Euribor sînt citite cu funcțiile capitolului (FRED, BNR, Eurostat, descărcate pe 4 noiembrie 2026) și tăiate la datele de sfîrșit ale slide-urilor; reproduc exact cifrele din curs. Consumul, venitul și investițiile SUA: macrodata din statsmodels '
       '(T1 1959 – T3 2009). Prețurile ajustate ale acțiunilor: data/market. Testele Engle–Granger, ADF și Johansen se recalculează în browser; statisticile Phillips–Ouliaris, '
       'modelele VECM, prognozele în afara eșantionului și testele istorice pairs trading sînt cele ale codului capitolului. Simulările folosesc extragerile seminței capitolului. '
       'Variantele care nu apar în curs (alt eșantion, alt număr de laguri, alte praguri, alte costuri, alți parametri ai simulărilor) sînt rulări suplimentare ale codului capitolului, '
       'cu aceleași date și aceeași sămînță.'),
 'en': dict(file='ch7.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 7 · Cointegration and VECM', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure or table on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step; '
        'the “Worked example” boxes redo the calculation with the numbers of the chapter.',
   src='US Treasury yields GS1, GS5, GS10 and TB3MS (FRED, monthly, January 1960 – September 2026), EUR/RON, EUR/HUF and EUR/PLN (BNR, month-end, July 2005 – September 2026), '
       'the monthly mean of EUR/RON and the HICP (Eurostat, July 2005 – August 2026), ROBOR 3M and Euribor 3M (Eurostat, August 1995 – August 2026), Romanian household consumption '
       'and GDP (Eurostat, 1995Q1 – 2026Q2, the vintage of the lecture, read from the line of the seminar chart, with no point dropped). The yields, BNR rates, HICP, ROBOR and Euribor are read with the chapter’s loaders (FRED, BNR, Eurostat, downloaded on 4 November 2026) and cut at the end dates of the slides; they reproduce the numbers of the lecture exactly. US consumption, income and investment: macrodata of statsmodels '
       '(1959Q1 – 2009Q3). Adjusted share prices: data/market. The Engle–Granger, ADF and Johansen tests are recomputed in the browser; the Phillips–Ouliaris statistics, the VECMs, '
       'the out-of-sample forecasts and the pairs-trading backtests are those of the chapter code. The simulations use the draws of the chapter’s seed. Variants that do not '
       'appear in the lecture (another sample, another number of lags, other thresholds, other costs, other simulation parameters) are extra runs of the chapter code with the '
       'same data and seed.'),
}

CSS = ('<style>.c7x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c7x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c7x .c7f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c7x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c7x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c7x .wctl .ctl{max-width:360px;margin:.15rem 0 .2rem}'
       '.c7t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c7t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c7t th,.c7t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c7t thead tr:last-child th{border-bottom:1px solid #1A3A6E}'
       '.c7t tbody tr:nth-child(even){background:#F5F7FB}.c7t .r{text-align:right}.c7cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}</style>')


def tbl(i):
    return f'<div class="tbl c7t" id="{i}"></div>'


def wx(i, title, formula, ctl=''):
    """Worked-example box; `ctl` holds the controls that only drive the box (they stay out of the chart's controls bar)."""
    return f'<div class="c7x"><h4>{title}</h4>' + (f'<div class="wctl">{ctl}</div>' if ctl else '') + f'<div class="c7f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def box(ctl, inner):
    """A box with controls that only drive the table inside it."""
    return f'<div class="c7x"><div class="wctl">{ctl}</div>{inner}</div>'


def widgets_all(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    lec = s(' (curs)', ' (lecture)')
    W = s('Exemplu rezolvat', 'Worked example')
    d = (lambda x: x.replace('0.5', '0{,}5')) if ro else (lambda x: x)
    ends = [('2000-12', s('decembrie 2000', 'December 2000')), ('2005-12', s('decembrie 2005', 'December 2005')), ('2010-12', s('decembrie 2010', 'December 2010')),
            ('2015-12', s('decembrie 2015', 'December 2015')), ('2019-12', s('decembrie 2019', 'December 2019')), ('2026-09', s('septembrie 2026', 'September 2026') + lec)]
    kk = lambda i, v=2: select(i, s('Diferențe întîrziate k', 'Lagged differences k'), [(str(k), str(k) + (lec if k == v else '')) for k in (1, 2, 3)], str(v))   # noqa: E731
    drop = lambda i, lab=None: slider(i, lab or s('Partea eliminată de la începutul fiecărui eșantion', 'Share dropped at the start of each sample'), 0, 60, 5, 0)   # noqa: E731
    rows = [('cons', s('consumul SUA pe venit', 'US consumption on income')), ('rates', s('SUA: 10 ani pe 3 luni', 'US: 10-year on 3-month')), ('tlv', s('TLV pe BRD (din 2014)', 'TLV on BRD (since 2014)')),
            ('brd', s('BRD pe TLV (din 2014)', 'BRD on TLV (since 2014)')), ('tlv10', s('TLV pe BRD (din 2010)', 'TLV on BRD (since 2010)')), ('fx', 'EUR/RON on EUR/HUF' if not ro else 'EUR/RON pe EUR/HUF'),
            ('ro', s('România: consumul pe PIB', 'Romania: consumption on GDP'))]
    return {
     'drunk': (slider('dd-c', s('c: partea din distanță recuperată de femeie', 'c: share of the gap closed by the drunk'), 0, 0.5, 0.05, 0.15)
               + slider('dd-d', s('d: partea din distanță recuperată de cîine', 'd: share of the gap closed by the dog'), 0, 0.5, 0.05, 0.25) + play('dd'),
               CSS + charts('dd-chart') + tbl('dd-tb')),
     'examples': (drop('ex-k') + play('ex'), charts('ex-chart') + tbl('ex-tb')),
     'eg-dist': (select('ed-T', s('Lungimea eșantionului T', 'Sample length T'), [('50', '50'), ('100', '100'), ('200', '200' + lec), ('500', '500')], '200')
                 + select('ed-a', s('Nivelul de semnificație (cuantilele punctate)', 'Significance level (dotted quantiles)'), [('1', '1%'), ('5', '5%' + lec), ('10', '10%')], '5') + play('ed'),
                 charts('ed-chart') + tbl('ed-tb')),
     'eg-steps': (drop('es-k') + slider('es-L', s('Laguri în ACF', 'Lags in the ACF'), 8, 48, 4, 24) + play('es'), charts('es-chart')),
     'ecm': (slider('em-e', s('Eșantioanele se termină în (consumul: cel mult 2009)', 'Samples end in (consumption: 2009 at most)'), 1980, 2026, 1, 2026)
             + slider('em-p', s('Laguri ale diferențelor în ECM', 'Lagged differences in the ECM'), 0, 4, 1, 1) + play('em'), charts('em-chart')),
     'rates': (select('ra-e', s('Eșantionul se termină în', 'Sample ends in'), ends, '2026-09') + kk('ra-k') + play('ra'), charts('ra-chart')),
     'vecm-irf': (select('ir-s', s('Șocul', 'Shock'), [('0', s('randamentul la 1 an', 'the 1-year yield') + lec), ('1', s('randamentul la 5 ani', 'the 5-year yield')), ('2', s('randamentul la 10 ani', 'the 10-year yield'))], '0')
                  + select('ir-e', s('Eșantionul se termină în', 'Sample ends in'), ends, '2026-09') + kk('ir-k') + play('ir'), charts('ir-chart')),
     'forecast': (slider('fc-y', s('Evaluarea începe în', 'Evaluation starts in'), 1990, 2015, 1, 1990) + play('fc'),
                  charts('fc-chart') + box(slider('fc-h', s('Orizontul din tabel (luni)', 'Horizon in the table (months)'), 1, 36, 1, 36), tbl('fc-tb'))),
     'ppp': (slider('pp-y', s('Eșantionul începe în', 'Sample starts in'), 2005, 2016, 1, 2005) + play('pp'), charts('pp-chart') + tbl('pp-tb')),
     'ro-ea': (slider('re-y', s('Al doilea eșantion începe în', 'Second sample starts in'), 2005, 2018, 1, 2010) + play('re'), charts('re-chart') + tbl('re-tb')),
     'cee-fx': (slider('cx-y', s('Eșantionul începe în', 'Sample starts in'), 2005, 2016, 1, 2005) + slider('cx-k', s('Diferențe întîrziate k', 'Lagged differences k'), 1, 4, 1, 1) + play('cx'),
                charts('cx-chart')),
     'pair': (slider('pr-y', s('Eșantionul începe în', 'Sample starts in'), 2010, 2020, 1, 2014) + slider('pr-z', s('Pragul de intrare |z|', 'Entry threshold |z|'), 1, 3, 0.25, 2) + play('pr'),
              charts('pr-chart')),
     'pairs-bt': (select('pb-e', s('Pragul de intrare |z|', 'Entry threshold |z|'), [('1.5', s('1,5', '1.5')), ('2.0', '2' + lec), ('2.5', s('2,5', '2.5'))], '2.0')
                  + slider('pb-c', s('Costurile, ca multiplu al celor din curs (0,20% și 0,05%)', 'Costs, as a multiple of the lecture’s (0.20% and 0.05%)'), 0, 3, 0.25, 1) + play('pb'),
                  charts('pb-chart')),
     'wx-trends': (slider('ct-a', s('a în y<sub>t</sub> = a x<sub>t</sub> + u<sub>t</sub>', 'a in y<sub>t</sub> = a x<sub>t</sub> + u<sub>t</sub>'), 0.5, 3, 0.5, 2)
                   + select('ct-c', s('c în z<sub>t</sub> = c x<sub>t</sub> + v<sub>t</sub>', 'c in z<sub>t</sub> = c x<sub>t</sub> + v<sub>t</sub>'), [(v, v.replace('.', ',') if ro else v) for v in ('-2', '-1.5', '-1', '-0.5', '0.5', '1', '1.5', '2')], '-1')
                   + play('ct'),
                   charts('ct-chart') + wx('ct-ws', W, r'\beta_1^\top\mathbf y_t=y_t-a\,x_t=u_t,\qquad \beta_2^\top\mathbf y_t=y_t-\tfrac{a}{c}\,z_t=u_t-\tfrac{a}{c}\,v_t,\qquad n-r=3-2=1')),
     'eg-crit': (select('ec-T', s('Numărul de observații T', 'Number of observations T'), [('inf', '∞' + lec), ('1000', '1000'), ('500', '500'), ('200', '200'), ('100', '100'), ('50', '50')], 'inf') + play('ec'),
                 charts('ec-chart') + tbl('ec-tb')),
     'wx-eg': (slider('we-e', s('Eșantionul se termină în', 'Sample ends in'), 1975, 2009, 1, 2009)
               + select('we-k', s('Laguri în pasul 2', 'Lags in step 2'), [('aic', s('alese după AIC', 'chosen by AIC') + lec), ('0', '0'), ('1', '1'), ('2', '2'), ('4', '4')], 'aic') + play('we'),
               charts('we-chart') + wx('we-ws', W, r'c_t=a+b\,y_t+u_t\ \text{(OLS)},\qquad \Delta\hat u_t=\gamma\,\hat u_{t-1}+\textstyle\sum_{j=1}^{k}\delta_j\Delta\hat u_{t-j}+e_t,\qquad \tau=\hat\gamma/\mathrm{SE}(\hat\gamma)')),
     'eg-table': (select('et-k', s('Partea eliminată de la începutul fiecărui eșantion', 'Share dropped at the start of each sample'), [(str(k), f'{k}%' + (lec if k == 0 else '')) for k in (0, 10, 20, 30, 40, 50)], '0') + play('et'),
                  charts('et-chart') + tbl('et-tb') + wx('et-ws', W, r'\Delta\hat u_t=\gamma\,\hat u_{t-1}+\textstyle\sum_{j=1}^{k}\delta_j\Delta\hat u_{t-j}+e_t,\qquad \tau=\dfrac{\hat\gamma}{\mathrm{SE}(\hat\gamma)}\ \text{'
                                                            + s('față de valoarea critică Engle–Granger', 'against the Engle–Granger critical value') + r'}',
                                                            select('et-r', s('Regresia din calcul', 'Regression of the calculation'), rows, 'cons'))),
     'wx-half': (slider('wh-g', s('Viteza de ajustare γ', 'Speed of adjustment γ'), -0.9, -0.05, 0.01, -0.2)
                 + select('wh-u', s('Unitatea de timp', 'Time unit'), [('m', s('luni', 'months')), ('q', s('trimestre', 'quarters'))], 'm') + play('wh'),
                 charts('wh-chart') + wx('wh-ws', W, d(r'u_h=(1+\gamma)^h u_0,\qquad (1+\gamma)^{h_{1/2}}=0.5\ \Rightarrow\ h_{1/2}=\frac{\ln 0.5}{\ln(1+\gamma)}'))),
     'wx-ecm': (slider('wc-u', s('Abaterea de la echilibru û<sub>t−1</sub> (%)', 'Deviation from equilibrium û<sub>t−1</sub> (%)'), -4, 4, 0.5, 2)
                + slider('wc-g', s('Creșterea venitului Δy<sub>t</sub> (%)', 'Income growth Δy<sub>t</sub> (%)'), -2, 3, 0.5, 1) + play('wc'),
                charts('wc-chart') + wx('wc-ws', W, r'\Delta c_t=c+\delta_0\,\Delta y_t+\gamma\,\hat u_{t-1}+\dots,\qquad \hat u_{t-1}=c_{t-1}-\hat a-\hat b\,y_{t-1}')),
     'wx-vecm': (slider('wv-a1', 'α₁', -0.5, 0.3, 0.05, -0.1) + slider('wv-a2', 'α₂', -0.3, 0.5, 0.05, 0.1) + play('wv'),
                 charts('wv-chart') + wx('wv-ws', W, r'\Delta\mathbf y_t=\alpha\beta^\top\mathbf y_{t-1}+\mathbf u_t,\quad \beta=(1,-1)^\top,\quad z_t=\beta^\top\mathbf y_t=(1+\beta^\top\alpha)\,z_{t-1}+u_{1t}-u_{2t}')),
     'wx-joh': (slider('wj-k', s('Diferențe întîrziate k', 'Lagged differences k'), 1, 4, 1, 2) + slider('wj-e', s('Eșantionul se termină în', 'Sample ends in'), 1990, 2026, 1, 2026) + play('wj'),
                charts('wj-chart') + wx('wj-ws', W, r'\lambda_{\mathrm{trace}}(r)=-T\sum_{i=r+1}^{n}\ln(1-\hat\lambda_i),\qquad \lambda_{\max}(r)=-T\ln(1-\hat\lambda_{r+1}),\qquad T=n_{\mathrm{obs}}-k-1')),
     'joh-table': (select('jt-k', s('Diferențe întîrziate k', 'Lagged differences k'), [('bic', s('alese după BIC: 2, 1, 1', 'chosen by BIC: 2, 1, 1') + lec), ('1', '1'), ('2', '2'), ('3', '3'), ('4', '4')], 'bic') + play('jt'),
                   charts('jt-chart') + tbl('jt-tb')),
     'joh-det': (slider('jd-y', s('Eșantionul începe în', 'Sample starts in'), 1960, 2000, 1, 1960) + slider('jd-k', s('Diferențe întîrziate k', 'Lagged differences k'), 1, 4, 1, 2) + play('jd'),
                 charts('jd-chart') + tbl('jd-tb')),
     'alpha-table': (select('at-e', s('Eșantionul se termină în', 'Sample ends in'), ends, '2026-09') + kk('at-k') + play('at'),
                     charts('at-chart') + tbl('at-tb') + wx('at-ws', W, r'\mathbf z_t=\hat\beta^\top\mathbf y_t,\qquad \mathbf z_t=(I+\hat\beta^\top\hat\alpha)\,\mathbf z_{t-1}+\dots,\qquad h_{1/2}=\frac{\ln 0.5}{\ln|\lambda|}'.replace('0.5', '0{,}5' if ro else '0.5'))),
    }


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
    """Chart blocks for build_web_lecture.py: each host carries the web version of its EXTRA blocks (moved by ch7.js)."""
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
    items = [(chart_pages[c][0], a, k, blocks[a]) for a, c, k in BLOCKS]
    xp = _extra_pages(lang)
    doc = pymupdf.open(os.path.join(ic.ROOT, TEX[lang] + '.pdf'))
    for anchor, _, _, skey, _ in EXTRA:
        pg, title = xp[anchor]
        img = f'ch7x_{anchor}_{lang}.jpg'
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
const yes = b => (b ? 'da' : 'nu');
window.CH7_TEXT = {
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 q: d => `T${Math.floor((+d.slice(5, 7) - 1) / 3) + 1} ${d.slice(0, 4)}`,
 L: { series: 'Seria', lags: 'Laguri', sd: 'Abaterea standard', pair: 'Perechea', crit5: 'Valoarea critică 5%', vars: 'Variabile n', regression: 'Regresia', freq: 'Frecvența',
   sample: 'Eșantionul', test: 'Testul', stat: 'Statistica', verdict: 'Concluzia la 5%', eq: 'Ecuația', system: 'Sistemul', rank: 'Rangul', cv5: 'Valoarea critică 5%', det: 'Termenii determiniști' },
 rej: 'se respinge', norej: 'nu se respinge',
 dd1: (c, d, r, h) => `c = ${c}, d = ${d}: distanța y − x este un AR(1) cu ρ = 1 − c − d = ${r}; timpul de înjumătățire ln 0,5/ln ρ = ${h} pași.`,
 dd2: (px, pd, sd, ps, ss) => `ADF: p = ${px} pentru x<sub>t</sub>, p = ${pd} pentru distanța y − x (abaterea standard ${sd}), p = ${ps} pentru distanța față de cîinele fără stăpîn (abaterea standard ${ss}).`,
 ddCap: (t, p, b) => `ADF cu constantă, laguri alese după AIC. Engle–Granger pentru y pe x: τ = ${t}, p = ${p}, panta ${b}.`,
 pairName: { rates: 'randamentele SUA', cons: 'consumul și PIB-ul SUA', banks: 'TLV și BRD', fx: 'EUR/RON și EUR/HUF' },
 exCap: k => `Engle–Granger (pasul 1 cu constantă, laguri după AIC, p-valori MacKinnon pentru 2 variabile)` + (k ? `, fără primele ${k}% din fiecare eșantion.` : ', eșantioanele din curs.'),
 ex1: rows => rows.map(([n, t, p, r]) => `${n}: τ = ${t}, p = ${p}${r ? ' (se respinge „fără cointegrare” la 5%)' : ''}`).join('; ') + '.',
 edQ: a => `Cuantila ${a}% simulată`, edMk: a => `MacKinnon ${a}%`, edSize: a => `Respingeri cu valoarea Dickey–Fuller ${a}%`, edOne: '1 (Dickey–Fuller)',
 edCap: T => `10 000 de eșantioane de T = ${T} mersuri aleatoare Gaussiene independente; pasul 1 cu constantă, pasul 2 fără laguri.`,
 ed1: (T, a, q1, q2, q3, m1, m2, m3) => `T = ${T}, cuantilele ${a}%: ${q1} (o serie), ${q2} (2 variabile), ${q3} (3 variabile); MacKinnon: ${m1}; ${m2}; ${m3}.`,
 ed2: (a, m, s2, s3) => `Cu valoarea Dickey–Fuller ${m} testul ar respinge o ipoteză nulă adevărată în ${s2}% din eșantioane cu 2 variabile și în ${s3}% cu 3, în loc de ${a}%.`,
 es1: (q0, r1, r8, sd) => `Consumul pe venit, de la ${q0}: ρ̂(1) = ${r1}, ρ̂(8) = ${r8}; abaterea standard a reziduurilor ${sd}.`,
 es2: (m0, r1, r12, sd, rw) => `Randamentele, de la ${m0}: ρ̂(1) = ${r1}, ρ̂(12) = ${r12}; abaterea standard ${sd}. Mersul aleator: ρ̂(12) = ${rw}.`,
 em1: (q, g, t, h, d0) => `Consumul (pînă în ${q}): γ̂ = ${g} (t = ${t}), timpul de înjumătățire ${h} trimestre; δ̂₀ = ${d0}.`,
 em2: (m, g, t, h, d0, gs, ts) => `Randamentul la 10 ani (pînă în ${m}): γ̂ = ${g} (t = ${t}), timpul de înjumătățire ${h} luni; δ̂₀ = ${d0}. Rata la 3 luni pe aceeași abatere: γ̂ = ${gs} (t = ${ts}).`,
 ra1: (e, k, n, b1, c1, b2, c2) => `VECM cu rangul 2, k = ${k}, pînă în ${e} (${n} observații): β̂₁ᵀy = i⁽¹⁾ ${b1.startsWith('−') ? '' : '+ '}${b1} i⁽¹⁰⁾ + ${c1}, β̂₂ᵀy = i⁽⁵⁾ ${b2.startsWith('−') ? '' : '+ '}${b2} i⁽¹⁰⁾ + ${c2}.`,
 ra2: (p1, p2, e1, e2, h1, h2) => `ADF pe cele două erori de echilibru: p = ${p1} și ${p2}. Valorile proprii ale lui I + β̂ᵀα̂: ${e1} și ${e2}, timpi de înjumătățire ${h1} și ${h2} luni.`,
 shock: ['randamentul la 1 an', 'randamentul la 5 ani', 'randamentul la 10 ani'],
 ir1: (e, k, s, a, b, c) => `Șoc de o abatere standard pe ${s} (pînă în ${e}, k = ${k}), în puncte procentuale pentru 1, 5 și 10 ani: la impact ${a.join(', ')}; după 12 luni ${b.join(', ')}; după 60 de luni ${c.join(', ')}.`,
 fc1: (n, f, a, b, c) => `${n}${de(n)} origini, de la ${f}. RMSE relativ la mersul aleator la 6 luni: VECM ${a} (1 an) și ${b} (5 ani); VAR în diferențe la 12 luni (1 an): ${c}.`,
 fc2: (a, b, c, d) => `Spread-ul 10 ani − 1 an: VECM ${a} la 24 de luni și ${b} la 36 de luni; VAR în diferențe ${c} și ${d}.`,
 fcName: ['1 an', '5 ani', '10 ani', 'spread 10 ani − 1 an'],
 fcCap: (h, n) => `RMSE relativ la mersul aleator la ${h} ${h === 1 ? 'lună' : 'luni'}, ${n}${de(n)} origini (cu aldine: modelul mai precis); RMSE RW în puncte procentuale.`,
 pp1: (m, s, r, q) => `De la ${m}: leul a pierdut ${s} puncte logaritmice față de euro, prețurile românești au crescut cu ${r} puncte mai mult decît cele din zona euro; cursul real s-a modificat cu ${q} puncte.`,
 pp2: (a, b, c, p) => `ADF pe q<sub>t</sub>: p = ${a} (cu constantă), p = ${b} (cu trend). Engle–Granger s<sub>t</sub> pe p<sub>t</sub> − p*<sub>t</sub>: panta ${c}, p = ${p}.`,
 ppT: { adf: 'ADF pe q<sub>t</sub>, cu constantă', adft: 'ADF pe q<sub>t</sub>, cu constantă și trend', eg: b => `Engle–Granger s<sub>t</sub> pe p<sub>t</sub> − p*<sub>t</sub> (panta ${b})`, joh: k => `Johansen pe (s<sub>t</sub>, p<sub>t</sub>, p*<sub>t</sub>), k = ${k}, λ<sub>trace</sub>(0)` },
 ppJoh: (cv, r) => `rangul ${r} (valoarea critică ${cv})`,
 ppCap: (m, n) => `Eșantionul ${m} – august 2026, ${n}${de(n)} luni.`,
 re1: (pf, pof, y, pw, pow, b, sp) => `Tot eșantionul: Engle–Granger p = ${pf}, Phillips–Ouliaris p = ${pof}. De la ${y}: p = ${pw} și ${pow}, panta ${b}, spread mediu ${sp} puncte procentuale.`,
 re2: (g, t, h) => `ECM pentru ROBOR: γ̂ = ${g} (t = ${t}), timpul de înjumătățire circa ${h} luni.`,
 reFull: 'august 1995 – august 2026', reWin: y => `ianuarie ${y} – august 2026`, reCap: 'ROBOR 3M pe Euribor 3M; Z<sub>t</sub>: Phillips–Ouliaris (codul capitolului).',
 cx1: (m, c, p) => `De la ${m}: EUR/RON ${c[0]}, EUR/HUF ${c[1]}, EUR/PLN ${c[2]} puncte logaritmice; ADF p = ${p.join('; ')}.`,
 cx2: (k, T, t0, cv, t1, r) => `Johansen (constantă, k = ${k}, T = ${T}): λ<sub>trace</sub>(0) = ${t0} față de valoarea critică ${cv}, λ<sub>trace</sub>(1) = ${t1}; rangul ${r}.`,
 pr1: (d0, n, b, rho, h, o, z, cr) => `De la ${d0} (${n}${de(n)} zile): b̂ = ${b}, spread-ul este un AR(1) cu ρ̂ = ${rho}, timpul de înjumătățire circa ${h} zile; ${o}% din zile în afara ±${z}; ${cr} treceri prin zero.`,
 pr2: (z, m, s) => `Regula aplicată cu parametrii întregului eșantion (prag ${z}, cost 0,20%): ${m}% pe an, raportul Sharpe ${s}; o iluzie, pentru că b̂, media și abaterea standard folosesc viitorul.`,
 pb1: (en, c, w, ww, gm, gs, nm, ns, tr, wr, mt, be) => `București (prag ${en}, cost ${c}%): ${ww} din ${w} ferestre cu perechi; brut ${gm}% pe an (Sharpe ${gs}), net ${nm}% (Sharpe ${ns}); ${tr} tranzacții, ${wr}% profitabile, media pe tranzacție ${mt}%; costul de echilibru ${be}%.`,
 pb2: (c, gm, gs, nm, ns, tr, wr, mt, be) => `Băncile americane (cost ${c}%): brut ${gm}% (Sharpe ${gs}), net ${nm}% (Sharpe ${ns}); ${tr} tranzacții, ${wr}% profitabile, media ${mt}%; costul de echilibru ${be}%.`,
 cts1: (a, c, ma, b2) => `Cu a = ${a} și c = ${c}: β₁ = (${ma}, 1, 0)ᵀ elimină trendul din y<sub>t</sub> − a x<sub>t</sub>; β₂ = (0, 1, ${b2})ᵀ îl elimină din y<sub>t</sub> − (a/c) z<sub>t</sub>.`,
 cts2: (y, x, z, ma, z1, b2, z2) => `La t = 399: y = ${y}, x = ${x}, z = ${z}; β₁ᵀy = ${y} + (${ma})·${x} = <b>${z1}</b>; β₂ᵀy = ${y} + (${b2})·${z} = <b>${z2}</b>.`,
 cts3: (s1, s2, s3, p1, p2, p3, one) => `Abaterile standard: ${s1} și ${s2} pentru cele două combinații, ${s3} pentru y − x; ADF p = ${p1}, ${p2}, ${p3}.` + (one ? ' Cu a = 1, y − x este chiar β₁ᵀy.' : ''),
 ct1: (ma, b2, p1, p2, p3) => `Vectorii de cointegrare: β₁ = (${ma}, 1, 0)ᵀ, β₂ = (0, 1, ${b2})ᵀ; ADF p = ${p1} și ${p2}; pentru y − x, p = ${p3}.`,
 cstOnly: 'constantă', cstTrend: 'constantă și trend',
 ecCap: T => `Valorile critice MacKinnon (2010), T = ${T}; rîndul cu numărul total de variabile din regresie, y<sub>t</sub> inclus.`,
 ec1: (T, a, b, c, d) => `T = ${T}, cu constantă, 5%: ${a} (Dickey–Fuller), ${b} (2 variabile), ${c} (3), ${d} (4).`,
 wes1: (q0, q1, n, a, b, r2, dw) => `Pasul 1, ${q0} – ${q1} (${n}${de(n)} trimestre): ĉ<sub>t</sub> = ${a} + ${b} y<sub>t</sub>, R² = ${r2}, DW = ${dw}.`,
 wes2: (k, g, se, t) => `Pasul 2 (${k} laguri): γ̂ = ${g}, SE(γ̂) = ${se}, τ = ${g}/${se} = <b>${t}</b>.`,
 wes3: (t, c, df, p, rej, rejdf, po) => `Valoarea critică Engle–Granger 5% pentru n = 2: ${c}; ${t} ${rej ? '<' : '>'} ${c}: ${rej ? '<b>se respinge</b>' : '<b>nu se respinge</b>'} „fără cointegrare” (p = ${p}). Cu valoarea Dickey–Fuller (${df}) decizia ar fi „${rejdf ? 'se respinge' : 'nu se respinge'}”.` + (po ? ` Phillips–Ouliaris: Z<sub>t</sub> = ${po[0]}, p = ${po[1]}.` : ''),
 we1: (q, b, t, k, c, p) => `Pînă în ${q}: b̂ = ${b}, τ = ${t} (${k} laguri), valoarea critică ${c}, p = ${p}.`,
 etShort: { cons: 'cons./venit', rates: '10a/3l', tlv: 'TLV/BRD 14', brd: 'BRD/TLV 14', tlv10: 'TLV/BRD 10', fx: 'RON/HUF', ro: 'RO cons./PIB' },
 etName: { cons: 'Consumul SUA pe venit', rates: 'SUA: 10 ani pe 3 luni', tlv: 'TLV pe BRD (din 2014)', brd: 'BRD pe TLV (din 2014)', tlv10: 'TLV pe BRD (din 2010)', fx: 'EUR/RON pe EUR/HUF', ro: 'România: consumul pe PIB' },
 etFreq: { cons: 'trimestrial', rates: 'lunar', tlv: 'zilnic', brd: 'zilnic', tlv10: 'zilnic', fx: 'lunar', ro: 'trimestrial' },
 etCap: k => `τ: statistica Engle–Granger (k laguri după AIC); p-valori MacKinnon pentru 2 variabile; Z<sub>t</sub>: Phillips–Ouliaris` + (k ? `; fără primele ${k}% din fiecare eșantion.` : '; eșantioanele din curs.'),
 et1: (k, ns, t, p, tr, pr, po) => `${k ? `Fără primele ${k}%: ` : ''}${ns} din 7 regresii resping „fără cointegrare” la 5%. TLV pe BRD din 2010: τ = ${t}, p = ${p}. România: Engle–Granger τ = ${tr}, p = ${pr}, Phillips–Ouliaris p = ${po}.`,
 ets1: (nm, n, b, k) => `${nm}: T = ${n}, panta b̂ = ${b}; pasul 2 pe reziduuri, fără constantă, cu k = ${k} laguri alese după AIC.`,
 ets2: (g, se, t) => `γ̂ = ${g}, SE(γ̂) = ${se}; τ = ${g}/${se} = <b>${t}</b>.`,
 ets3: (t, c, df, p, rej, rejdf) => `Valoarea critică Engle–Granger 5% (n = 2): ${c}; Dickey–Fuller (n = 1): ${df}. ${t} ${rej ? '<' : '>'} ${c}: <b>${rej ? 'se respinge' : 'nu se respinge'}</b> „fără cointegrare”, p = ${p}.` + (rej !== rejdf ? ` Cu valoarea Dickey–Fuller decizia ar fi fost „${rejdf ? 'se respinge' : 'nu se respinge'}”.` : ''),
 unit: { m: 'luni', q: 'trimestre' },
 whs1: (g, r, l, h) => `γ = ${g}: 1 + γ = ${r}, ln(1 + γ) = ${l}; h<sub>1/2</sub> = ln 0,5/ln(1 + γ) = −0,6931/(${l}) = <b>${h}</b> perioade.`,
 whs2: (h, u, y) => `În ${u}: ${h} ${u}, adică ${y} ani. Timpii de înjumătățire se compară doar în aceeași unitate de timp.`,
 whs3: (a, b, c) => `Din abaterea inițială rămîn ${a} după o perioadă, ${b} după 4 și ${c} după 12.`,
 wh1: (g, h, u, y) => `γ = ${g}: jumătate din abatere dispare în ${h} ${u} (${y} ani).`,
 wcs1: (c, d0, g) => `ECM estimat (T1 1959 – T3 2009, un lag): ĉ = ${c}, δ̂₀ = ${d0}, γ̂ = ${g}.`,
 wcs2: (d0, dy, sr, g, u, ec) => `Efectul pe termen scurt: ${d0} × ${dy} = ${sr}; corecția erorii: ${g} × ${u} = ${ec}.`,
 wcs3: (c, sr, ec, tot, no) => `Δĉ<sub>t</sub> = ${c} + ${sr} + (${ec}) = <b>${tot}%</b> (diferențele întîrziate puse la 0); fără termenul de corecție: ${no}%.`,
 wc1: (u, g, t, no) => `Abaterea ${u}% și venitul +${g}%: creșterea prognozată a consumului ${t}% (fără corecția erorii ${no}%).`,
 wvs1: (a1, a2, p11, p12, p21, p22) => `α = (${a1}, ${a2})ᵀ, β = (1, −1)ᵀ: Π = αβᵀ = [${p11} ${p12}; ${p21} ${p22}], det Π = 0, rangul 1.`,
 wvs2: (a1, a2, d, r, st) => `1 + βᵀα = 1 + (${a1}) − (${a2}) = 1 + (${d}) = <b>${r}</b>: ${st ? 'decalajul este un AR(1) staționar' : 'decalajul nu este staționar (condiția −2 < βᵀα < 0 nu este îndeplinită)'}.`,
 wvs3: (st, h, both) => (st ? `Timpul de înjumătățire al decalajului: ${h} perioade.` : 'Fără corecția erorii, cele două serii se pot îndepărta oricît.') + (both ? ' Ambele variabile închid decalajul (α₁ ≤ 0 ≤ α₂).' : ''),
 wv1: (r, st, h, s) => `1 + βᵀα = ${r}: ${st ? `decalaj staționar, timpul de înjumătățire ${h} perioade` : 'decalajul nu revine'}; abaterea standard a decalajului simulat ${s}.`,
 wjs1: (n, k, T, l, m) => `${n}${de(n)} luni, k = ${k}: T efectiv = ${n} − ${k} − 1 = ${T}; λ̂ = (${l.join('; ')}); −ln(1 − λ̂<sub>i</sub>) = ${m.join('; ')}.`,
 wjs2: (T, m, t0, c0, x0, cm0) => `r = 0: urma = ${T} × (${m.join(' + ')}) = <b>${t0}</b> față de ${c0}; valoarea proprie maximă: ${T} × ${m[0]} = <b>${x0}</b> față de ${cm0}.`,
 wjs3: (t1, c1, t2, c2, r) => `r ≤ 1: urma = ${t1} față de ${c1}; r ≤ 2: ${t2} față de ${c2}. Rangul ales de testul urmei: <b>${r}</b>.`,
 wj1: (e, k, T, a, b, c, rt, rm) => `Pînă în ${e}, k = ${k}, T = ${T}: λ<sub>trace</sub> = ${a}, ${b}, ${c}; rangul ${rt} (urma), ${rm} (valoarea proprie maximă).`,
 jtTitle: { yields: 'Randamentele SUA 1, 5, 10 ani', macro: 'SUA: producție, consum, investiții', fx: 'EUR/RON, EUR/HUF, EUR/PLN' },
 jtCap: 'Cazul 3 (constantă nerestricționată); cu aldine: statisticile peste valoarea critică de 5%; rangul după testul urmei.',
 jt1: rows => rows.map(([n, k, t, r]) => `${n}: k = ${k}, λ<sub>trace</sub>(0) = ${t}, rangul ${r}`).join('; ') + '.',
 jdTitle: ['Case 1: none', 'Case 3: constant', 'Case 5: constant and trend'],
 jdName: ['niciunul, cazul 1 (det_order = −1)', 'constantă, cazul 3 (det_order = 0)', 'constantă și trend, cazul 5 (det_order = 1)'],
 jdCap: (m, k, T) => `Randamentele SUA la 1, 5 și 10 ani, ${m} – septembrie 2026, k = ${k}, T = ${T}; cu aldine: peste valoarea critică de 5%.`,
 jd1: (m, k, r, t, c) => `De la ${m}, k = ${k}: rangurile ${r.join(', ')} pentru cazurile 1, 3 și 5; în cazul 5, λ<sub>trace</sub>(2) = ${t} față de ${c}.`,
 atEq: ['Δi⁽¹⁾<sub>t</sub> (1 an)', 'Δi⁽⁵⁾<sub>t</sub> (5 ani)', 'Δi⁽¹⁰⁾<sub>t</sub> (10 ani)'],
 atCap: (e, k, n, b1, b2) => `VECM cu rangul 2, k = ${k}, pînă în ${e} (${n} observații); β̂ normalizat pe 1 an și 5 ani: pantele 10 ani ${b1} și ${b2}; cu aldine: |t| ≥ 1,96.`,
 ats1: M => `I + β̂ᵀα̂ = [${M[0].join(' ')}; ${M[1].join(' ')}].`,
 ats2: (a, b) => `Valorile proprii (în modul): <b>${a}</b> și <b>${b}</b>.`,
 ats3: (a, h1, b, h2) => `Timpii de înjumătățire: ln 0,5/ln ${a} = <b>${h1}</b> luni și ln 0,5/ln ${b} = <b>${h2}</b> luni.`,
 at1: (e, k, t1, lo, hi) => `Pînă în ${e}, k = ${k}: randamentul la 1 an are |t| cel mult ${t1}; la 5 și 10 ani |t| este între ${lo} și ${hi}.`,
};
})();
""",
 'en': r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
window.CH7_TEXT = {
 dmy: d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 my: d => `${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`,
 q: d => `${d.slice(0, 4)}Q${Math.floor((+d.slice(5, 7) - 1) / 3) + 1}`,
 L: { series: 'Series', lags: 'Lags', sd: 'Standard deviation', pair: 'Pair', crit5: '5% critical value', vars: 'Variables n', regression: 'Regression', freq: 'Freq.',
   sample: 'Sample', test: 'Test', stat: 'Statistic', verdict: 'Verdict at 5%', eq: 'Equation', system: 'System', rank: 'Rank', cv5: '5% critical value', det: 'Deterministic terms' },
 rej: 'rejected', norej: 'not rejected',
 dd1: (c, d, r, h) => `c = ${c}, d = ${d}: the distance y − x is an AR(1) with ρ = 1 − c − d = ${r}; half-life ln 0.5/ln ρ = ${h} steps.`,
 dd2: (px, pd, sd, ps, ss) => `ADF: p = ${px} for x<sub>t</sub>, p = ${pd} for the distance y − x (standard deviation ${sd}), p = ${ps} for the distance to the stray dog (standard deviation ${ss}).`,
 ddCap: (t, p, b) => `ADF with a constant, lags by AIC. Engle–Granger of y on x: τ = ${t}, p = ${p}, slope ${b}.`,
 pairName: { rates: 'US yields', cons: 'US consumption and GDP', banks: 'TLV and BRD', fx: 'EUR/RON and EUR/HUF' },
 exCap: k => `Engle–Granger (step 1 with a constant, lags by AIC, MacKinnon p-values for 2 variables)` + (k ? `, without the first ${k}% of each sample.` : ', the samples of the lecture.'),
 ex1: rows => rows.map(([n, t, p, r]) => `${n}: τ = ${t}, p = ${p}${r ? ' (“no cointegration” rejected at 5%)' : ''}`).join('; ') + '.',
 edQ: a => `Simulated ${a}% quantile`, edMk: a => `MacKinnon ${a}%`, edSize: a => `Rejections with the Dickey–Fuller ${a}% value`, edOne: '1 (Dickey–Fuller)',
 edCap: T => `10,000 samples of T = ${T} independent Gaussian random walks; step 1 with a constant, step 2 without lags.`,
 ed1: (T, a, q1, q2, q3, m1, m2, m3) => `T = ${T}, ${a}% quantiles: ${q1} (one series), ${q2} (2 variables), ${q3} (3 variables); MacKinnon: ${m1}; ${m2}; ${m3}.`,
 ed2: (a, m, s2, s3) => `With the Dickey–Fuller value ${m} the test would reject a true null in ${s2}% of the samples with 2 variables and in ${s3}% with 3, instead of ${a}%.`,
 es1: (q0, r1, r8, sd) => `Consumption on income, from ${q0}: ρ̂(1) = ${r1}, ρ̂(8) = ${r8}; standard deviation of the residuals ${sd}.`,
 es2: (m0, r1, r12, sd, rw) => `Yields, from ${m0}: ρ̂(1) = ${r1}, ρ̂(12) = ${r12}; standard deviation ${sd}. The random walk: ρ̂(12) = ${rw}.`,
 em1: (q, g, t, h, d0) => `Consumption (up to ${q}): γ̂ = ${g} (t = ${t}), half-life ${h} quarters; δ̂₀ = ${d0}.`,
 em2: (m, g, t, h, d0, gs, ts) => `10-year yield (up to ${m}): γ̂ = ${g} (t = ${t}), half-life ${h} months; δ̂₀ = ${d0}. The 3-month rate on the same gap: γ̂ = ${gs} (t = ${ts}).`,
 ra1: (e, k, n, b1, c1, b2, c2) => `VECM with rank 2, k = ${k}, up to ${e} (${n} observations): β̂₁ᵀy = i⁽¹⁾ ${b1.startsWith('−') ? '' : '+ '}${b1} i⁽¹⁰⁾ + ${c1}, β̂₂ᵀy = i⁽⁵⁾ ${b2.startsWith('−') ? '' : '+ '}${b2} i⁽¹⁰⁾ + ${c2}.`,
 ra2: (p1, p2, e1, e2, h1, h2) => `ADF on the two equilibrium errors: p = ${p1} and ${p2}. Eigenvalues of I + β̂ᵀα̂: ${e1} and ${e2}, half-lives ${h1} and ${h2} months.`,
 shock: ['the 1-year yield', 'the 5-year yield', 'the 10-year yield'],
 ir1: (e, k, s, a, b, c) => `One-standard-deviation shock to ${s} (up to ${e}, k = ${k}), in percentage points for 1, 5 and 10 years: on impact ${a.join(', ')}; after 12 months ${b.join(', ')}; after 60 months ${c.join(', ')}.`,
 fc1: (n, f, a, b, c) => `${n} origins, from ${f}. RMSE relative to the random walk at 6 months: VECM ${a} (1 year) and ${b} (5 years); VAR in differences at 12 months (1 year): ${c}.`,
 fc2: (a, b, c, d) => `The 10-year minus 1-year spread: VECM ${a} at 24 months and ${b} at 36 months; VAR in differences ${c} and ${d}.`,
 fcName: ['1 year', '5 years', '10 years', '10-year minus 1-year spread'],
 fcCap: (h, n) => `RMSE relative to the random walk at ${h} month${h === 1 ? '' : 's'}, ${n} origins (bold: the more accurate model); RMSE RW in percentage points.`,
 pp1: (m, s, r, q) => `Since ${m}: the leu lost ${s} log points against the euro, Romanian prices rose ${r} log points more than euro-area prices; the real exchange rate changed by ${q} log points.`,
 pp2: (a, b, c, p) => `ADF on q<sub>t</sub>: p = ${a} (constant), p = ${b} (trend). Engle–Granger of s<sub>t</sub> on p<sub>t</sub> − p*<sub>t</sub>: slope ${c}, p = ${p}.`,
 ppT: { adf: 'ADF on q<sub>t</sub>, constant', adft: 'ADF on q<sub>t</sub>, constant and trend', eg: b => `Engle–Granger of s<sub>t</sub> on p<sub>t</sub> − p*<sub>t</sub> (slope ${b})`, joh: k => `Johansen on (s<sub>t</sub>, p<sub>t</sub>, p*<sub>t</sub>), k = ${k}, λ<sub>trace</sub>(0)` },
 ppJoh: (cv, r) => `rank ${r} (critical value ${cv})`,
 ppCap: (m, n) => `Sample ${m} – August 2026, ${n} months.`,
 re1: (pf, pof, y, pw, pow, b, sp) => `Full sample: Engle–Granger p = ${pf}, Phillips–Ouliaris p = ${pof}. Since ${y}: p = ${pw} and ${pow}, slope ${b}, mean spread ${sp} percentage points.`,
 re2: (g, t, h) => `ECM for ROBOR: γ̂ = ${g} (t = ${t}), half-life about ${h} months.`,
 reFull: 'August 1995 – August 2026', reWin: y => `January ${y} – August 2026`, reCap: 'ROBOR 3M on Euribor 3M; Z<sub>t</sub>: Phillips–Ouliaris (the chapter code).',
 cx1: (m, c, p) => `Since ${m}: EUR/RON ${c[0]}, EUR/HUF ${c[1]}, EUR/PLN ${c[2]} log points; ADF p = ${p.join('; ')}.`,
 cx2: (k, T, t0, cv, t1, r) => `Johansen (constant, k = ${k}, T = ${T}): λ<sub>trace</sub>(0) = ${t0} against the critical value ${cv}, λ<sub>trace</sub>(1) = ${t1}; rank ${r}.`,
 pr1: (d0, n, b, rho, h, o, z, cr) => `Since ${d0} (${n} days): b̂ = ${b}, the spread is an AR(1) with ρ̂ = ${rho}, half-life about ${h} days; ${o}% of the days outside ±${z}; ${cr} zero crossings.`,
 pr2: (z, m, s) => `The rule run with full-sample parameters (threshold ${z}, cost 0.20%): ${m}% per year, Sharpe ratio ${s}; an illusion, since b̂, the mean and the standard deviation use the future.`,
 pb1: (en, c, w, ww, gm, gs, nm, ns, tr, wr, mt, be) => `Bucharest (threshold ${en}, cost ${c}%): ${ww} of ${w} windows with pairs; gross ${gm}% per year (Sharpe ${gs}), net ${nm}% (Sharpe ${ns}); ${tr} trades, ${wr}% profitable, mean per trade ${mt}%; break-even cost ${be}%.`,
 pb2: (c, gm, gs, nm, ns, tr, wr, mt, be) => `US banks (cost ${c}%): gross ${gm}% (Sharpe ${gs}), net ${nm}% (Sharpe ${ns}); ${tr} trades, ${wr}% profitable, mean ${mt}%; break-even cost ${be}%.`,
 cts1: (a, c, ma, b2) => `With a = ${a} and c = ${c}: β₁ = (${ma}, 1, 0)ᵀ removes the trend in y<sub>t</sub> − a x<sub>t</sub>; β₂ = (0, 1, ${b2})ᵀ removes it in y<sub>t</sub> − (a/c) z<sub>t</sub>.`,
 cts2: (y, x, z, ma, z1, b2, z2) => `At t = 399: y = ${y}, x = ${x}, z = ${z}; β₁ᵀy = ${y} + (${ma})·${x} = <b>${z1}</b>; β₂ᵀy = ${y} + (${b2})·${z} = <b>${z2}</b>.`,
 cts3: (s1, s2, s3, p1, p2, p3, one) => `Standard deviations: ${s1} and ${s2} for the two combinations, ${s3} for y − x; ADF p = ${p1}, ${p2}, ${p3}.` + (one ? ' With a = 1, y − x is β₁ᵀy itself.' : ''),
 ct1: (ma, b2, p1, p2, p3) => `Cointegrating vectors: β₁ = (${ma}, 1, 0)ᵀ, β₂ = (0, 1, ${b2})ᵀ; ADF p = ${p1} and ${p2}; for y − x, p = ${p3}.`,
 cstOnly: 'constant', cstTrend: 'constant and trend',
 ecCap: T => `MacKinnon (2010) critical values, T = ${T}; the row of the total number of variables in the regression, y<sub>t</sub> included.`,
 ec1: (T, a, b, c, d) => `T = ${T}, constant, 5%: ${a} (Dickey–Fuller), ${b} (2 variables), ${c} (3), ${d} (4).`,
 wes1: (q0, q1, n, a, b, r2, dw) => `Step 1, ${q0} – ${q1} (${n} quarters): ĉ<sub>t</sub> = ${a} + ${b} y<sub>t</sub>, R² = ${r2}, DW = ${dw}.`,
 wes2: (k, g, se, t) => `Step 2 (${k} lags): γ̂ = ${g}, SE(γ̂) = ${se}, τ = ${g}/${se} = <b>${t}</b>.`,
 wes3: (t, c, df, p, rej, rejdf, po) => `Engle–Granger 5% critical value for n = 2: ${c}; ${t} ${rej ? '<' : '>'} ${c}: “no cointegration” <b>${rej ? 'rejected' : 'not rejected'}</b> (p = ${p}). With the Dickey–Fuller value (${df}) the decision would be “${rejdf ? 'rejected' : 'not rejected'}”.` + (po ? ` Phillips–Ouliaris: Z<sub>t</sub> = ${po[0]}, p = ${po[1]}.` : ''),
 we1: (q, b, t, k, c, p) => `Up to ${q}: b̂ = ${b}, τ = ${t} (${k} lags), critical value ${c}, p = ${p}.`,
 etShort: { cons: 'cons./income', rates: '10y/3m', tlv: 'TLV/BRD 14', brd: 'BRD/TLV 14', tlv10: 'TLV/BRD 10', fx: 'RON/HUF', ro: 'RO cons./GDP' },
 etName: { cons: 'US consumption on income', rates: 'US: 10-year on 3-month yield', tlv: 'TLV on BRD (since 2014)', brd: 'BRD on TLV (since 2014)', tlv10: 'TLV on BRD (since 2010)', fx: 'EUR/RON on EUR/HUF', ro: 'Romania: consumption on GDP' },
 etFreq: { cons: 'quarterly', rates: 'monthly', tlv: 'daily', brd: 'daily', tlv10: 'daily', fx: 'monthly', ro: 'quarterly' },
 etCap: k => `τ: Engle–Granger statistic (k lags by AIC); MacKinnon p-values for 2 variables; Z<sub>t</sub>: Phillips–Ouliaris` + (k ? `; without the first ${k}% of each sample.` : '; the samples of the lecture.'),
 et1: (k, ns, t, p, tr, pr, po) => `${k ? `Without the first ${k}%: ` : ''}${ns} of the 7 regressions reject “no cointegration” at 5%. TLV on BRD since 2010: τ = ${t}, p = ${p}. Romania: Engle–Granger τ = ${tr}, p = ${pr}, Phillips–Ouliaris p = ${po}.`,
 ets1: (nm, n, b, k) => `${nm}: T = ${n}, slope b̂ = ${b}; step 2 on the residuals, without a constant, with k = ${k} lags chosen by AIC.`,
 ets2: (g, se, t) => `γ̂ = ${g}, SE(γ̂) = ${se}; τ = ${g}/${se} = <b>${t}</b>.`,
 ets3: (t, c, df, p, rej, rejdf) => `Engle–Granger 5% critical value (n = 2): ${c}; Dickey–Fuller (n = 1): ${df}. ${t} ${rej ? '<' : '>'} ${c}: “no cointegration” <b>${rej ? 'rejected' : 'not rejected'}</b>, p = ${p}.` + (rej !== rejdf ? ` With the Dickey–Fuller value the decision would have been “${rejdf ? 'rejected' : 'not rejected'}”.` : ''),
 unit: { m: 'months', q: 'quarters' },
 whs1: (g, r, l, h) => `γ = ${g}: 1 + γ = ${r}, ln(1 + γ) = ${l}; h<sub>1/2</sub> = ln 0.5/ln(1 + γ) = −0.6931/(${l}) = <b>${h}</b> periods.`,
 whs2: (h, u, y) => `In ${u}: ${h} ${u}, that is ${y} years. Half-lives are comparable only in the same time unit.`,
 whs3: (a, b, c) => `Of the initial deviation, ${a} is left after one period, ${b} after 4 and ${c} after 12.`,
 wh1: (g, h, u, y) => `γ = ${g}: half of a deviation is gone in ${h} ${u} (${y} years).`,
 wcs1: (c, d0, g) => `Estimated ECM (1959Q1 – 2009Q3, one lag): ĉ = ${c}, δ̂₀ = ${d0}, γ̂ = ${g}.`,
 wcs2: (d0, dy, sr, g, u, ec) => `Short-run effect: ${d0} × ${dy} = ${sr}; error correction: ${g} × ${u} = ${ec}.`,
 wcs3: (c, sr, ec, tot, no) => `Δĉ<sub>t</sub> = ${c} + ${sr} + (${ec}) = <b>${tot}%</b> (the lagged differences set to 0); without the error correction term: ${no}%.`,
 wc1: (u, g, t, no) => `Deviation ${u}% and income +${g}%: predicted growth of consumption ${t}% (without error correction ${no}%).`,
 wvs1: (a1, a2, p11, p12, p21, p22) => `α = (${a1}, ${a2})ᵀ, β = (1, −1)ᵀ: Π = αβᵀ = [${p11} ${p12}; ${p21} ${p22}], det Π = 0, rank 1.`,
 wvs2: (a1, a2, d, r, st) => `1 + βᵀα = 1 + (${a1}) − (${a2}) = 1 + (${d}) = <b>${r}</b>: ${st ? 'the gap is a stationary AR(1)' : 'the gap is not stationary (the condition −2 < βᵀα < 0 fails)'}.`,
 wvs3: (st, h, both) => (st ? `Half-life of the gap: ${h} periods.` : 'Without error correction the two series can drift apart without limit.') + (both ? ' Both variables close the gap (α₁ ≤ 0 ≤ α₂).' : ''),
 wv1: (r, st, h, s) => `1 + βᵀα = ${r}: ${st ? `stationary gap, half-life ${h} periods` : 'the gap does not return'}; standard deviation of the simulated gap ${s}.`,
 wjs1: (n, k, T, l, m) => `${n} months, k = ${k}: effective T = ${n} − ${k} − 1 = ${T}; λ̂ = (${l.join('; ')}); −ln(1 − λ̂<sub>i</sub>) = ${m.join('; ')}.`,
 wjs2: (T, m, t0, c0, x0, cm0) => `r = 0: trace = ${T} × (${m.join(' + ')}) = <b>${t0}</b> against ${c0}; maximum eigenvalue: ${T} × ${m[0]} = <b>${x0}</b> against ${cm0}.`,
 wjs3: (t1, c1, t2, c2, r) => `r ≤ 1: trace = ${t1} against ${c1}; r ≤ 2: ${t2} against ${c2}. Rank chosen by the trace test: <b>${r}</b>.`,
 wj1: (e, k, T, a, b, c, rt, rm) => `Up to ${e}, k = ${k}, T = ${T}: λ<sub>trace</sub> = ${a}, ${b}, ${c}; rank ${rt} (trace), ${rm} (maximum eigenvalue).`,
 jtTitle: { yields: 'US yields 1y, 5y, 10y', macro: 'US output, consumption, investment', fx: 'EUR/RON, EUR/HUF, EUR/PLN' },
 jtCap: 'Case 3 (unrestricted constant); bold: statistics above the 5% critical value; rank by the trace test.',
 jt1: rows => rows.map(([n, k, t, r]) => `${n}: k = ${k}, λ<sub>trace</sub>(0) = ${t}, rank ${r}`).join('; ') + '.',
 jdTitle: ['Case 1: none', 'Case 3: constant', 'Case 5: constant and trend'],
 jdName: ['none, case 1 (det_order = −1)', 'constant, case 3 (det_order = 0)', 'constant and trend, case 5 (det_order = 1)'],
 jdCap: (m, k, T) => `US yields at 1, 5 and 10 years, ${m} – September 2026, k = ${k}, T = ${T}; bold: above the 5% critical value.`,
 jd1: (m, k, r, t, c) => `Since ${m}, k = ${k}: ranks ${r.join(', ')} for cases 1, 3 and 5; in case 5, λ<sub>trace</sub>(2) = ${t} against ${c}.`,
 atEq: ['Δi⁽¹⁾<sub>t</sub> (1 year)', 'Δi⁽⁵⁾<sub>t</sub> (5 years)', 'Δi⁽¹⁰⁾<sub>t</sub> (10 years)'],
 atCap: (e, k, n, b1, b2) => `VECM with rank 2, k = ${k}, up to ${e} (${n} observations); β̂ normalised on the 1-year and 5-year yields: 10-year slopes ${b1} and ${b2}; bold: |t| ≥ 1.96.`,
 ats1: M => `I + β̂ᵀα̂ = [${M[0].join(' ')}; ${M[1].join(' ')}].`,
 ats2: (a, b) => `Eigenvalues (in modulus): <b>${a}</b> and <b>${b}</b>.`,
 ats3: (a, h1, b, h2) => `Half-lives: ln 0.5/ln ${a} = <b>${h1}</b> months and ln 0.5/ln ${b} = <b>${h2}</b> months.`,
 at1: (e, k, t1, lo, hi) => `Up to ${e}, k = ${k}: the 1-year yield has |t| at most ${t1}; for 5 and 10 years |t| is between ${lo} and ${hi}.`,
};
})();
""",
}

# QA, PREDICT and TICKET: see below
QA = {
 'ro': {
  'drunk': ('Setați c = d = 0: nici femeia, nici cîinele nu mai corectează distanța. Ce p-value dă ADF pentru distanța y − x?',
            '0,261: rădăcina unitară nu se mai respinge. Fără corecție, y − x este diferența a două mersuri aleatoare independente, deci tot un mers aleator; abaterea standard a distanței crește de la 1,86 la 7,07. '
            'Cu c = 0,15 și d = 0,25, ρ = 0,6 și p < 0,001: corecția erorii, nu faptul că ambele serii pornesc din același loc, ține distanța mărginită.'),
  'examples': ('Eliminați primele 50% din fiecare eșantion. Mai respinge Engle–Granger „fără cointegrare” la 5% pentru consumul și PIB-ul SUA?',
               'Nu: τ = −1,95, p = 0,553 pe ultimele 102 trimestre, față de τ = −3,54, p = 0,029 pe 203. Cu jumătate din date și abateri de durata unor ani, testul are puțină putere; '
               'p mare înseamnă că relația nu se separă de zgomot pe acest eșantion, nu că nu există. Pentru bănci testul rămîne sub 5% (p = 0,044).'),
  'eg-dist': ('Treceți la T = 50. Cu valoarea critică Dickey–Fuller de 5%, ce proporție din eșantioanele cu 2 variabile ar respinge o ipoteză nulă adevărată?',
              '14,2%, aproape ca la T = 200 (14,5%), și 30,0% cu 3 variabile. Distorsiunea vine din faptul că OLS alege b̂ pentru a face reziduurile cît mai mici, nu din lungimea eșantionului; '
              'la T = 50 valoarea critică corectă pentru 2 variabile este −3,46 (cuantila simulată −3,41), față de −2,92 în tabelul Dickey–Fuller.'),
  'eg-steps': ('Eliminați primele 50% din fiecare eșantion. Scade autocorelația de ordinul 8 a reziduurilor consumului?',
               'Nu: crește de la 0,31 la 0,46, iar pentru randamente ρ̂(12) crește de la 0,58 la 0,62. În a doua jumătate a eșantioanelor abaterile de la relația pe termen lung durează mai mult; '
               'persistența reziduurilor depinde de perioada aleasă, de aceea decizia o dă testul, nu forma ACF.'),
  'ecm': ('Terminați eșantioanele în 1990. Cît devine timpul de înjumătățire al ECM-ului pentru consum?',
          '27 de trimestre: γ̂ = −0,025, cu t = −0,66, față de 14,3 trimestre și t = −1,75 pînă în 2009. Pe 1959–1990 termenul de corecție nu se separă de zgomot, iar un γ̂ mic în modul dă un timp de înjumătățire lung și imprecis. '
          'Pentru randamentul la 10 ani, pînă în 1990: γ̂ = −0,048 (t = −4,58), 13,9 luni.'),
  'rates': ('Terminați eșantionul în decembrie 2000. Cît devine panta randamentului la 10 ani în prima relație de echilibru?',
            '−0,954, față de −1,046 pe 1960–2026; în a doua relație −0,988 față de −1,045. Ambele rămîn aproape de −1, deci echilibrele sînt tot aproape spread-urile i⁽¹⁾ − i⁽¹⁰⁾ și i⁽⁵⁾ − i⁽¹⁰⁾. '
            'Cu β̂ super-consistent, o diferență de 0,09 pe 40 de ani de date este o schimbare reală a eșantionului, nu doar zgomot de estimare.'),
  'vecm-irf': ('Terminați eșantionul în decembrie 2005. Revin la zero, după 60 de luni, răspunsurile la un șoc pe randamentul la 1 an?',
               'Nu: +0,30, +0,30 și +0,29 puncte procentuale pentru 1, 5 și 10 ani (la impact 0,41, 0,27, 0,21). Cu un singur trend comun, șocul mută nivelul tuturor randamentelor permanent, '
               'iar răspunsurile devin aproape egale pentru că spread-urile revin la medie; aceeași formă apare și pe 1960–2026 (0,37, 0,36, 0,33).'),
  'forecast': ('Porniți evaluarea în 2005. Mai cîștigă VECM-ul la spread-ul 10 ani − 1 an la orizontul de 36 de luni?',
               'Da: RMSE relativ 0,83 (75 de origini), față de 0,74 de la 1990, iar VAR-ul în diferențe rămîne la 1,00. Pentru nivelul randamentului la 1 an, VECM-ul nu bate mersul aleator (1,05 la 36 de luni). '
               'Cointegrarea ajută la prognoza relației, nu a trendului comun.'),
  'ppp': ('Porniți eșantionul în 2010. Respinge ADF rădăcina unitară în cursul real q<sub>t</sub>?',
          'Nu: p = 0,924 (cu trend 0,957), mai departe de respingere decît pe 2005–2026 (0,24). După 2010 leul s-a depreciat cu 23,7 puncte logaritmice, iar prețurile relative au crescut cu 33,1: cursul real a continuat să scadă. '
          'Un eșantion mai scurt nu poate arăta ajustări cu timpi de înjumătățire de cîțiva ani.'),
  'ro-ea': ('Porniți al doilea eșantion în 2015. Rămîne Engle–Granger semnificativ la 5%?',
            'Nu: τ = −3,24, p = 0,063, iar Phillips–Ouliaris dă p = 0,338, față de 0,047 și 0,029 de la 2010. În plus, γ̂ din ECM-ul pentru ROBOR devine pozitiv (0,011): după 2015 ROBOR nu se mai întoarce spre relația cu Euribor. '
            'Cu o politică monetară proprie, legătura dintre cele două dobînzi depinde mult de perioada aleasă.'),
  'cee-fx': ('Porniți eșantionul în 2015. Cît devine statistica urmei pentru r = 0?',
             '11,1, față de valoarea critică 29,80: rangul rămîne 0, ca de la 2005 (24,2) și de la 2012 (12,7). Pe un eșantion mai scurt testul are și mai puțină putere, '
             'iar ADF nu respinge rădăcina unitară pentru niciun curs (p = 0,958; 0,626; 0,335). Nicio perioadă nu arată un trend comun al celor trei monede.'),
  'pair': ('Porniți eșantionul în 2010. Cît devine timpul de înjumătățire al spread-ului TLV − b̂ BRD?',
           'Circa 444 de zile de tranzacționare, cu ρ̂ = 0,998 și b̂ = 1,68, față de 69 de zile din 2014. Pe 2010–2026 testul Engle–Granger nu respinge „fără cointegrare” (p = 0,622): '
           'spread-ul se comportă aproape ca un mers aleator, iar o poziție deschisă ar putea aștepta ani. Rezultatul favorabil din 2014 depinde de data de început aleasă.'),
  'pairs-bt': ('Dublați costurile (2×: 0,40% la București). Ce randament net anual are strategia la București?',
               '−0,35% pe an (Sharpe −0,05), față de 0,41% cu costurile din curs; brut rămîne 1,17%. Costul de echilibru este 0,31%, deci orice cost mai mare face strategia neprofitabilă. '
               'La băncile americane, cu un cost de 0,10%, randamentul net scade doar de la 2,57% la 2,37%.'),
  'wx-trends': ('Setați a = 1 (c = −1). Care este al doilea vector de cointegrare?',
                'β₂ = (0, 1, 1)ᵀ: y<sub>t</sub> + z<sub>t</sub> = u<sub>t</sub> + v<sub>t</sub>, fără trend. Vectorul depinde de coeficienți: în general β₂ = (0, 1, −a/c)ᵀ, deci (0, 1, 2)ᵀ pentru a = 2 și c = −1. '
                'Rangul rămîne 2, pentru că trei serii au tot un singur trend comun.'),
  'eg-crit': ('Treceți la T = 50. Cît este valoarea critică de 5% pentru n = 2, cu constantă?',
              '−3,46, față de −3,34 asimptotic și −3,37 la T = 200. La eșantioane mici distribuția statisticii este mai deplasată spre stînga, deci valorile critice sînt mai negative; '
              'tabelul Dickey–Fuller (−2,92 la T = 50) ar fi și mai departe de cea corectă.'),
  'wx-eg': ('Terminați eșantionul în 1985. Se respinge „fără cointegrare” la 5%?',
            'Nu: τ = −3,00 față de valoarea critică Engle–Granger −3,39 (p = 0,111), pe 108 trimestre. Cu valoarea Dickey–Fuller (−2,89) s-ar respinge, greșit. '
            'R² rămîne peste 0,99 și panta 1,072, dar nici R² mare, nici o pantă apropiată de 1 nu dovedesc cointegrarea.'),
  'eg-table': ('Eliminați primele 30% din fiecare eșantion. Mai lipsește dovada de cointegrare pentru TLV pe BRD din 2010?',
               'Nu: τ = −3,98, p = 0,008 (Phillips–Ouliaris p = 0,001), față de p = 0,622 pe tot eșantionul din 2010. Fără anii 2010–2014 relația dintre cele două bănci apare; '
               'aceeași pereche dă verdicte opuse după data de început, de aceea alegerea eșantionului după ce ați văzut datele este data snooping.'),
  'wx-half': ('Setați γ = −0,5. Cît este timpul de înjumătățire?',
              'O perioadă: ln 0,5/ln 0,5 = 1. Jumătate din abatere dispare la fiecare pas, deci după o perioadă rămîne exact jumătate. Regula 1/|γ| = 2 ar presupune o ajustare liniară, nu geometrică.'),
  'wx-ecm': ('Setați abaterea û<sub>t−1</sub> = −2 (consumul sub echilibru), cu venitul +1%. Cît devine creșterea prognozată a consumului?',
             '1,05%: 0,418 + 0,537 × 1 + (−0,047) × (−2) = 0,418 + 0,537 + 0,095. Cînd consumul este sub echilibru, termenul de corecție îl împinge în sus, cu aceeași mărime cu care îl frîna în exemplul din curs (0,86%).'),
  'wx-vecm': ('Setați α₂ = −0,1 (cu α₁ = −0,1). Mai revine decalajul y₁ − y₂ la zero?',
              'Nu: 1 + βᵀα = 1 + α₁ − α₂ = 1 − 0,1 + 0,1 = 1, deci z<sub>t</sub> = z<sub>t−1</sub> + u₁<sub>t</sub> − u₂<sub>t</sub> este un mers aleator. Ambele variabile scad cînd y₁ > y₂, cu aceeași viteză, '
              'deci decalajul nu se închide: contează βᵀα, nu semnul fiecărui α.'),
  'wx-joh': ('Terminați eșantionul în 2010. Ce rang dă testul urmei?',
             'Rangul 1: λ<sub>trace</sub>(0) = 45,5 > 29,80, dar λ<sub>trace</sub>(1) = 14,5 < 15,49, pe T = 609. Pe 1960–2026 rangul este 2. Cu mai puține observații a doua relație, foarte persistentă, nu se mai separă de o rădăcină unitară; '
             'rangul ales de test depinde și de puterea lui, nu doar de structura datelor.'),
  'joh-table': ('Cu k = 4 diferențe întîrziate în toate sistemele, ce rang dă testul urmei pentru producția, consumul și investițiile SUA?',
                'Rangul 1: λ<sub>trace</sub>(0) = 37,5 > 29,80, iar λ<sub>trace</sub>(1) = 15,3 < 15,49, față de rangul 0 cu k = 1 (28,9, un rezultat la limită). Numărul de laguri schimbă verdictul; '
                'pentru cele trei monede rangul rămîne 0 (28,1).'),
  'joh-det': ('Porniți eșantionul în 1990. Ce rang dă cazul 5 (constantă și trend)?',
              'Rangul 1: λ<sub>trace</sub> = 41,4, apoi 11,1 < 18,40, față de rangul 3 pe 1960–2026. Pe tot eșantionul trendul liniar absoarbe scăderea randamentelor după 1981, iar testul ajunge să declare fiecare randament staționar în jurul unui trend; '
              'pe eșantionul de după 1990 toate cele trei cazuri dau rangul 1.'),
  'alpha-table': ('Terminați eșantionul în decembrie 2000. Mai pare randamentul la 1 an slab exogen?',
                  'Nu: t = 2,10 și −3,10 pentru cele două erori de echilibru, față de −0,82 și 0,31 pe 1960–2026. Înainte de 2001 și randamentul scurt reacționa la abaterile de la echilibru; '
                  'exogenitatea slabă a capătului scurt apare pe eșantionul care include politica monetară cu dobînzi aproape de zero.'),
 },
 'en': {
  'drunk': ('Set c = d = 0: neither the drunk nor the dog corrects the distance any more. What p-value does ADF give for the distance y − x?',
            '0.261: the unit root is no longer rejected. Without correction, y − x is the difference of two independent random walks, hence a random walk itself; the standard deviation of the distance rises from 1.86 to 7.07. '
            'With c = 0.15 and d = 0.25, ρ = 0.6 and p < 0.001: error correction, not the common starting point, keeps the distance bounded.'),
  'examples': ('Drop the first 50% of each sample. Does Engle–Granger still reject “no cointegration” at 5% for US consumption and GDP?',
               'No: τ = −1.95, p = 0.553 on the last 102 quarters, against τ = −3.54, p = 0.029 on 203. With half the data and deviations that last years, the test has little power; '
               'a large p means that the relation is not separated from noise in this sample, not that it does not exist. For the banks the test stays below 5% (p = 0.044).'),
  'eg-dist': ('Switch to T = 50. With the 5% Dickey–Fuller critical value, what share of the samples with 2 variables would reject a true null?',
              '14.2%, about as at T = 200 (14.5%), and 30.0% with 3 variables. The distortion comes from OLS choosing b̂ to make the residuals as small as possible, not from the sample length; '
              'at T = 50 the right critical value for 2 variables is −3.46 (simulated quantile −3.41), against −2.92 in the Dickey–Fuller table.'),
  'eg-steps': ('Drop the first 50% of each sample. Does the lag-8 autocorrelation of the consumption residuals fall?',
               'No: it rises from 0.31 to 0.46, and for the yields ρ̂(12) rises from 0.58 to 0.62. In the second half of the samples the deviations from the long-run relation last longer; '
               'the persistence of the residuals depends on the period, which is why the test, not the shape of the ACF, decides.'),
  'ecm': ('End the samples in 1990. What does the half-life of the consumption ECM become?',
          '27 quarters: γ̂ = −0.025, with t = −0.66, against 14.3 quarters and t = −1.75 up to 2009. On 1959–1990 the error correction term is not separated from noise, and a γ̂ small in modulus gives a long and imprecise half-life. '
          'For the 10-year yield, up to 1990: γ̂ = −0.048 (t = −4.58), 13.9 months.'),
  'rates': ('End the sample in December 2000. What does the slope of the 10-year yield in the first equilibrium relation become?',
            '−0.954, against −1.046 over 1960–2026; in the second relation −0.988 against −1.045. Both stay close to −1, so the equilibria are still almost the spreads i⁽¹⁾ − i⁽¹⁰⁾ and i⁽⁵⁾ − i⁽¹⁰⁾. '
            'With a super-consistent β̂, a difference of 0.09 on 40 years of data reflects a change of sample, not only estimation noise.'),
  'vecm-irf': ('End the sample in December 2005. Do the responses to a shock to the 1-year yield return to zero after 60 months?',
               'No: +0.30, +0.30 and +0.29 percentage points for 1, 5 and 10 years (on impact 0.41, 0.27, 0.21). With one common trend, the shock moves the level of all yields permanently, '
               'and the responses become almost equal because the spreads return to their means; the same shape appears over 1960–2026 (0.37, 0.36, 0.33).'),
  'forecast': ('Start the evaluation in 2005. Does the VECM still gain for the 10-year minus 1-year spread at the 36-month horizon?',
               'Yes: relative RMSE 0.83 (75 origins), against 0.74 from 1990, and the VAR in differences stays at 1.00. For the level of the 1-year yield the VECM does not beat the random walk (1.05 at 36 months). '
               'Cointegration helps to forecast the relation, not the common trend.'),
  'ppp': ('Start the sample in 2010. Does ADF reject a unit root in the real exchange rate q<sub>t</sub>?',
          'No: p = 0.924 (with trend 0.957), further from rejection than over 2005–2026 (0.24). Since 2010 the leu depreciated by 23.7 log points while relative prices rose by 33.1: the real exchange rate kept falling. '
          'A shorter sample cannot show adjustments with half-lives of several years.'),
  'ro-ea': ('Start the second sample in 2015. Is Engle–Granger still significant at 5%?',
            'No: τ = −3.24, p = 0.063, and Phillips–Ouliaris gives p = 0.338, against 0.047 and 0.029 since 2010. Moreover, γ̂ of the ROBOR ECM turns positive (0.011): after 2015 ROBOR no longer returns towards the relation with Euribor. '
            'With its own monetary policy, the link between the two rates depends strongly on the chosen period.'),
  'cee-fx': ('Start the sample in 2015. What does the trace statistic for r = 0 become?',
             '11.1, against the critical value 29.80: the rank stays 0, as since 2005 (24.2) and since 2012 (12.7). On a shorter sample the test has even less power, '
             'and ADF does not reject a unit root for any rate (p = 0.958; 0.626; 0.335). No period shows a common trend of the three currencies.'),
  'pair': ('Start the sample in 2010. What does the half-life of the spread TLV − b̂ BRD become?',
           'About 444 trading days, with ρ̂ = 0.998 and b̂ = 1.68, against 69 days since 2014. Over 2010–2026 the Engle–Granger test does not reject “no cointegration” (p = 0.622): '
           'the spread behaves almost like a random walk, and an open position could wait for years. The favourable result since 2014 depends on the chosen start date.'),
  'pairs-bt': ('Double the costs (2×: 0.40% in Bucharest). What net annual return does the strategy earn in Bucharest?',
               '−0.35% per year (Sharpe −0.05), against 0.41% with the lecture’s costs; gross it stays 1.17%. The break-even cost is 0.31%, so any higher cost makes the strategy unprofitable. '
               'For the US banks, with a cost of 0.10%, the net return falls only from 2.57% to 2.37%.'),
  'wx-trends': ('Set a = 1 (c = −1). What is the second cointegrating vector?',
                'β₂ = (0, 1, 1)ᵀ: y<sub>t</sub> + z<sub>t</sub> = u<sub>t</sub> + v<sub>t</sub>, with no trend. The vector depends on the coefficients: in general β₂ = (0, 1, −a/c)ᵀ, hence (0, 1, 2)ᵀ for a = 2 and c = −1. '
                'The rank stays 2, because three series still share one common trend.'),
  'eg-crit': ('Switch to T = 50. What is the 5% critical value for n = 2, with a constant?',
              '−3.46, against −3.34 asymptotically and −3.37 at T = 200. In small samples the distribution of the statistic is shifted further to the left, so the critical values are more negative; '
              'the Dickey–Fuller table (−2.92 at T = 50) would be even further from the right one.'),
  'wx-eg': ('End the sample in 1985. Is “no cointegration” rejected at 5%?',
            'No: τ = −3.00 against the Engle–Granger critical value −3.39 (p = 0.111), on 108 quarters. With the Dickey–Fuller value (−2.89) it would be rejected, wrongly. '
            'R² stays above 0.99 and the slope is 1.072, but neither a high R² nor a slope close to 1 proves cointegration.'),
  'eg-table': ('Drop the first 30% of each sample. Is there still no evidence of cointegration for TLV on BRD since 2010?',
               'There is now: τ = −3.98, p = 0.008 (Phillips–Ouliaris p = 0.001), against p = 0.622 on the whole sample since 2010. Without 2010–2014 the relation between the two banks appears; '
               'the same pair gives opposite verdicts depending on the start date, which is why choosing the sample after seeing the data is data snooping.'),
  'wx-half': ('Set γ = −0.5. What is the half-life?',
              'One period: ln 0.5/ln 0.5 = 1. Half of the deviation disappears at each step, so after one period exactly half is left. The rule 1/|γ| = 2 would assume a linear, not a geometric, adjustment.'),
  'wx-ecm': ('Set the deviation û<sub>t−1</sub> = −2 (consumption below equilibrium), with income +1%. What does the predicted growth of consumption become?',
             '1.05%: 0.418 + 0.537 × 1 + (−0.047) × (−2) = 0.418 + 0.537 + 0.095. When consumption is below equilibrium, the correction term pushes it up, by the same amount by which it held it back in the lecture’s example (0.86%).'),
  'wx-vecm': ('Set α₂ = −0.1 (with α₁ = −0.1). Does the gap y₁ − y₂ still return to zero?',
              'No: 1 + βᵀα = 1 + α₁ − α₂ = 1 − 0.1 + 0.1 = 1, so z<sub>t</sub> = z<sub>t−1</sub> + u₁<sub>t</sub> − u₂<sub>t</sub> is a random walk. Both variables fall when y₁ > y₂, at the same speed, '
              'so the gap does not close: what matters is βᵀα, not the sign of each α.'),
  'wx-joh': ('End the sample in 2010. What rank does the trace test give?',
             'Rank 1: λ<sub>trace</sub>(0) = 45.5 > 29.80, but λ<sub>trace</sub>(1) = 14.5 < 15.49, with T = 609. Over 1960–2026 the rank is 2. With fewer observations the second, very persistent relation is no longer separated from a unit root; '
             'the rank chosen by the test depends on its power too, not only on the structure of the data.'),
  'joh-table': ('With k = 4 lagged differences in every system, what rank does the trace test give for US output, consumption and investment?',
                'Rank 1: λ<sub>trace</sub>(0) = 37.5 > 29.80 and λ<sub>trace</sub>(1) = 15.3 < 15.49, against rank 0 with k = 1 (28.9, a near miss). The number of lags changes the verdict; '
                'for the three currencies the rank stays 0 (28.1).'),
  'joh-det': ('Start the sample in 1990. What rank does case 5 (constant and trend) give?',
              'Rank 1: λ<sub>trace</sub> = 41.4, then 11.1 < 18.40, against rank 3 over 1960–2026. On the whole sample the linear trend absorbs the fall of the yields after 1981, and the test ends up calling every yield stationary around a trend; '
              'on the sample since 1990 all three cases give rank 1.'),
  'alpha-table': ('End the sample in December 2000. Does the 1-year yield still look weakly exogenous?',
                  'No: t = 2.10 and −3.10 for the two equilibrium errors, against −0.82 and 0.31 over 1960–2026. Before 2001 the short yield also reacted to the deviations from equilibrium; '
                  'the weak exogeneity of the short end appears on the sample that includes the periods of near-zero policy rates.'),
 },
}

# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with the readout of the block at the state the question names (verified by a node script).
PREDICT = {
 'drunk': (1, [('Tot sub 0,001: distanța rămîne staționară, pentru că ambele trasee pornesc aproape unul de altul.', 'Still below 0.001: the distance stays stationary, because both paths start close to each other.'),
               ('Peste 0,05 (0,261): distanța devine un mers aleator.', 'Above 0.05 (0.261): the distance becomes a random walk.'),
               ('ADF nu se mai poate calcula: fără corecție distanța este constantă.', 'ADF can no longer be computed: without correction the distance is constant.')]),
 'examples': (1, [('Da: cointegrarea este o proprietate a seriilor, nu a eșantionului.', 'Yes: cointegration is a property of the series, not of the sample.'),
                  ('Nu: τ = −1,95, p = 0,553.', 'No: τ = −1.95, p = 0.553.'), ('Da, și mai clar: datele recente sînt mai precise.', 'Yes, and more clearly: recent data are more precise.')]),
 'eg-dist': (1, [('5%: pe eșantioane mici distribuțiile Engle–Granger și Dickey–Fuller coincid.', '5%: in small samples the Engle–Granger and Dickey–Fuller distributions coincide.'),
                 ('14,2%.', '14.2%.'), ('Peste 50%: la T = 50 testul respinge aproape mereu.', 'Above 50%: at T = 50 the test rejects almost always.')]),
 'eg-steps': (1, [('Da, sub 0,1: reziduurile pe eșantioane scurte sînt mai puțin persistente.', 'Yes, below 0.1: residuals in short samples are less persistent.'),
                  ('Nu: crește la 0,46.', 'No: it rises to 0.46.'), ('Rămîne 0,31: ACF-ul reziduurilor nu depinde de eșantion.', 'It stays 0.31: the ACF of the residuals does not depend on the sample.')]),
 'ecm': (1, [('Tot 14,3 trimestre: viteza de ajustare este o constantă a economiei.', 'Still 14.3 quarters: the speed of adjustment is a constant of the economy.'),
             ('27 de trimestre, cu γ̂ = −0,025 (t = −0,66).', '27 quarters, with γ̂ = −0.025 (t = −0.66).'), ('Sub 5 trimestre: eșantioanele scurte ajustează mai repede.', 'Below 5 quarters: short samples adjust faster.')]),
 'rates': (1, [('−1,046: β̂ este super-consistent, deci nu se schimbă cu eșantionul.', '−1.046: β̂ is super-consistent, so it does not change with the sample.'),
               ('−0,954: tot aproape de −1.', '−0.954: still close to −1.'), ('Aproape 0: înainte de 2001 randamentele nu erau legate.', 'Close to 0: before 2001 the yields were not linked.')]),
 'vecm-irf': (1, [('Da: după 60 de luni orice șoc dispare.', 'Yes: after 60 months every shock dies out.'), ('Nu: +0,30, +0,30 și +0,29 pp.', 'No: +0.30, +0.30 and +0.29 pp.'),
                  ('Devin negative: randamentele revin peste nivelul inițial.', 'They turn negative: the yields overshoot their initial level.')]),
 'forecast': (1, [('Nu: avantajul VECM vine doar din anii 1990.', 'No: the VECM’s advantage comes from the 1990s only.'), ('Da: 0,83.', 'Yes: 0.83.'),
                  ('Da, și pentru niveluri: 1 an la 36 de luni sub 0,8.', 'Yes, and for the levels too: 1 year at 36 months below 0.8.')]),
 'ppp': (1, [('Da: pe un eșantion recent PPP se verifică.', 'Yes: on a recent sample PPP holds.'), ('Nu: p = 0,924.', 'No: p = 0.924.'), ('Testul nu se poate calcula pe mai puțin de 20 de ani.', 'The test cannot be computed on less than 20 years.')]),
 'ro-ea': (1, [('Da: fără anii de criză legătura este mai strînsă.', 'Yes: without the crisis years the link is tighter.'), ('Nu: p = 0,063.', 'No: p = 0.063.'),
               ('Da, și Phillips–Ouliaris coboară sub 0,01.', 'Yes, and Phillips–Ouliaris falls below 0.01.')]),
 'cee-fx': (1, [('Peste 29,80: după 2015 cursurile au fost stabile, deci cointegrate.', 'Above 29.80: after 2015 the rates were stable, hence cointegrated.'), ('11,1: rangul rămîne 0.', '11.1: the rank stays 0.'),
                ('Aproximativ 24,2, ca pe tot eșantionul: statistica nu depinde de eșantion.', 'About 24.2, as on the whole sample: the statistic does not depend on the sample.')]),
 'pair': (1, [('Tot circa 69 de zile.', 'Still about 69 days.'), ('Circa 444 de zile.', 'About 444 days.'), ('Sub 10 zile: un eșantion mai lung dă o ajustare mai rapidă.', 'Below 10 days: a longer sample gives faster adjustment.')]),
 'pairs-bt': (1, [('Tot pozitiv, circa 0,3%: cu 73 de tranzacții costurile contează puțin.', 'Still positive, about 0.3%: with 73 trades costs matter little.'), ('−0,35% pe an.', '−0.35% per year.'),
                  ('0,82%: randamentul net crește cu costurile.', '0.82%: the net return rises with the costs.')]),
 'wx-trends': (1, [('(0, 1, 2)ᵀ, ca în curs: β₂ nu depinde de a.', '(0, 1, 2)ᵀ, as in the lecture: β₂ does not depend on a.'), ('(0, 1, 1)ᵀ.', '(0, 1, 1)ᵀ.'),
                   ('Nu mai există: cu a = 1 rangul scade la 1.', 'There is none: with a = 1 the rank falls to 1.')]),
 'eg-crit': (1, [('−3,34, ca asimptotic: valorile critice nu depind de T.', '−3.34, as asymptotically: the critical values do not depend on T.'), ('−3,46.', '−3.46.'),
                 ('−2,92: la T = 50 se folosește tabelul Dickey–Fuller.', '−2.92: at T = 50 one uses the Dickey–Fuller table.')]),
 'wx-eg': (1, [('Da: τ = −3,00 este sub −2,89.', 'Yes: τ = −3.00 is below −2.89.'), ('Nu: −3,00 > −3,39, p = 0,111.', 'No: −3.00 > −3.39, p = 0.111.'), ('Da: R² este peste 0,99.', 'Yes: R² is above 0.99.')]),
 'eg-table': (1, [('Da: o relație care lipsește pe tot eșantionul lipsește și pe o parte a lui.', 'Yes: a relation absent from the whole sample is absent from any part of it.'),
                  ('Nu: τ = −3,98, p = 0,008.', 'No: τ = −3.98, p = 0.008.'), ('Testul nu se mai poate face pe 70% din date.', 'The test can no longer be run on 70% of the data.')]),
 'wx-half': (0, [('O perioadă.', 'One period.'), ('Două perioade: 1/|γ|.', 'Two periods: 1/|γ|.'), ('Patru perioade: (1 + γ)⁴ = 0,0625.', 'Four periods: (1 + γ)⁴ = 0.0625.')]),
 'wx-ecm': (1, [('0,86%, ca în curs: semnul abaterii nu contează.', '0.86%, as in the lecture: the sign of the deviation does not matter.'), ('1,05%.', '1.05%.'),
                ('0,67%: corecția erorii scade mereu consumul.', '0.67%: error correction always lowers consumption.')]),
 'wx-vecm': (1, [('Da: ambele α sînt negative, deci ambele ajustează.', 'Yes: both α are negative, so both adjust.'), ('Nu: 1 + βᵀα = 1, decalajul este un mers aleator.', 'No: 1 + βᵀα = 1, the gap is a random walk.'),
                 ('Da, mai repede: două ajustări în loc de una.', 'Yes, faster: two adjustments instead of one.')]),
 'wx-joh': (1, [('2, ca pe tot eșantionul.', '2, as on the whole sample.'), ('1: urma pentru r ≤ 1 este 14,5 < 15,49.', '1: the trace for r ≤ 1 is 14.5 < 15.49.'),
                ('3: un eșantion mai scurt face randamentele staționare.', '3: a shorter sample makes the yields stationary.')]),
 'joh-table': (1, [('0, ca în curs: rangul nu depinde de k.', '0, as in the lecture: the rank does not depend on k.'), ('1.', '1.'), ('3: mai multe laguri dau mereu rangul complet.', '3: more lags always give full rank.')]),
 'joh-det': (1, [('3, ca pe tot eșantionul.', '3, as on the whole sample.'), ('1.', '1.'), ('0: după 1990 randamentele nu mai sînt legate.', '0: after 1990 the yields are no longer linked.')]),
 'alpha-table': (1, [('Da: |t| rămîne sub 1.', 'Yes: |t| stays below 1.'), ('Nu: t = 2,10 și −3,10.', 'No: t = 2.10 and −3.10.'), ('Nu se poate estima un VECM pe 40 de ani.', 'A VECM cannot be estimated on 40 years.')]),
}
VEIL_CHARTS = ()

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slide „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('Engle–Granger cu două variabile dă τ = −3,1 pentru T = 300. Respingeți „fără cointegrare” la 5%?',
         'Engle–Granger with two variables gives τ = −3.1 for T = 300. Do you reject “no cointegration” at 5%?'), key=1, opts=[
  (('Da: −3,1 este sub valoarea Dickey–Fuller −2,87.', 'Yes: −3.1 is below the Dickey–Fuller value −2.87.'),
   ('Reziduurile vin dintr-o regresie estimată: tabelul corect este cel Engle–Granger pentru n = 2, nu cel Dickey–Fuller.', 'The residuals come from an estimated regression: the right table is the Engle–Granger one for n = 2, not the Dickey–Fuller one.')),
  (('Nu: −3,1 > −3,36, valoarea critică Engle–Granger.', 'No: −3.1 > −3.36, the Engle–Granger critical value.'),
   ('Cu n = 2 și T = 300 valoarea critică de 5% este −3,36; τ nu trece de ea, deci testul nu separă reziduurile de o rădăcină unitară.', 'With n = 2 and T = 300 the 5% critical value is −3.36; τ does not pass it, so the test does not separate the residuals from a unit root.')),
  (('Da: orice τ negativ arată cointegrare.', 'Yes: any negative τ shows cointegration.'),
   ('Statistica este negativă și sub ipoteza nulă; decizia cere comparația cu valoarea critică.', 'The statistic is negative under the null too; the decision needs the comparison with the critical value.'))]),
 dict(q=('Un ECM dă γ̂ = −0,1 pe date lunare. Cît este timpul de înjumătățire?', 'An ECM gives γ̂ = −0.1 on monthly data. What is the half-life?'), key=0, opts=[
  (('ln 0,5/ln 0,9 ≈ 6,6 luni.', 'ln 0.5/ln 0.9 ≈ 6.6 months.'),
   ('Abaterea scade cu factorul 1 + γ = 0,9 pe lună; 0,9^6,6 ≈ 0,5.','The deviation shrinks by the factor 1 + γ = 0.9 a month; 0.9^6.6 ≈ 0.5.')),
  (('10 luni: 1/|γ|.', '10 months: 1/|γ|.'),
   ('1/|γ| este durata medie aproximativă a unei ajustări liniare, nu timpul în care dispare jumătate din abatere.', '1/|γ| is the approximate mean lag of a linear adjustment, not the time in which half of the deviation is gone.')),
  (('5 luni: 0,5/|γ|.', '5 months: 0.5/|γ|.'),
   ('Ajustarea este geometrică, nu liniară: în fiecare lună se închide 10% din abaterea rămasă.', 'The adjustment is geometric, not linear: each month 10% of the remaining deviation is closed.'))]),
 dict(q=('Patru variabile I(1) au rangul de cointegrare 3. Cîte trenduri comune le antrenează?', 'Four I(1) variables have cointegration rank 3. How many common trends drive them?'), key=0, opts=[
  (('Unul: n − r = 1.', 'One: n − r = 1.'), ('Fiecare vector de cointegrare elimină un trend; 4 variabile cu 3 vectori păstrează un singur trend comun.', 'Each cointegrating vector removes one trend; 4 variables with 3 vectors keep a single common trend.')),
  (('Trei: cîte unul pentru fiecare vector.', 'Three: one for each vector.'), ('Vectorii de cointegrare dau combinații staționare, nu trenduri; trendurile sînt n − r.', 'The cointegrating vectors give stationary combinations, not trends; the trends are n − r.')),
  (('Niciunul: cu rangul 3 seriile sînt staționare.', 'None: with rank 3 the series are stationary.'), ('Seriile sînt staționare doar cu rangul complet, r = n = 4.', 'The series are stationary only with full rank, r = n = 4.'))]),
]


def _predict_js():
    import json
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {}))
              for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch7-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH7_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH7_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n')


_predict_js()


if __name__ == '__main__':
    import interactive_common
    page_spec = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA', 'TEXT_JS', 'PREDICT', 'TICKET')})
    page_spec.widgets = lambda lang: {a: v for a, v in widgets_all(lang).items() if a in {b for b, _, _ in BLOCKS}}
    interactive_common.build(page_spec)
    for lg in ('en', 'ro'):
        _post(lg)
