"""Build the interactive pages of Chapter 3 (unit roots and ARIMA models): interactive/ch3.html (EN) and interactive/ch3_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its
interactive version on the right. Logic in interactive/ch3.js, inputs in interactive/ch3_data.js
(Quantlets/Ch_03/export_interactive.py), slide images in interactive/slides/.
The worked examples and result tables of the lecture (the HICP regression, the Dickey-Fuller critical values and the GDP
example, lag choice, size and power, KPSS from the partial sums, ADF and KPSS together, the unit-root table, the two
Zivot-Andrews tables, the AICc grid, the residual checks, the forecast variance of DS and TS models, the ARIMA(1,1,0)
example, the interval widths and the GDP forecasts) get blocks of their own (EXTRA), placed by slide; slide images
interactive/slides/ch3x_*.jpg. On the web lecture (build_web_lecture.py) the EXTRA blocks travel inside a chart block of the
same section and ch3.js moves each one to its own frame.
"""
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 3

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_03'
TEX = {'ro': 'RO/Cursuri/capitol3_radacini_unitare_modele_arima', 'en': 'EN/Courses/chapter3_unit_roots_arima_models'}

# (anchor, chart file, section key); order of the slides
BLOCKS = [('four-series', 'tsa_ch3_four_series', 'trend'), ('ts-ds', 'tsa_ch3_ts_ds', 'trend'), ('detrend', 'tsa_ch3_detrend', 'trend'),
          ('spurious-mc', 'tsa_ch3_spurious_mc', 'spur'), ('spurious-real', 'tsa_ch3_spurious_real', 'spur'),
          ('df-dist', 'tsa_ch3_df_dist', 'df'), ('adf-power', 'tsa_ch3_adf_power', 'df'),
          ('kpss-sums', 'tsa_ch3_kpss_sums', 'kpss'),
          ('breaks-sim', 'tsa_ch3_breaks_sim', 'brk'), ('breaks-real', 'tsa_ch3_breaks_real', 'brk'),
          ('gdp-ident', 'tsa_ch3_gdp_ident', 'arima'), ('gdp-diag', 'tsa_ch3_gdp_diag', 'arima'), ('interval-width', 'tsa_ch3_interval_width', 'arima'),
          ('gdp-forecast', 'tsa_ch3_gdp_forecast', 'arima'), ('overdiff', 'tsa_ch3_overdiff_gdp', 'arima'),
          ('inflation', 'tsa_ch3_inflation_d', 'prac'), ('eurron', 'tsa_ch3_eurron_forecast', 'prac'), ('us-gdp', 'tsa_ch3_us_gdp', 'prac')]

PFX = {'four-series': 'fs', 'ts-ds': 'td', 'detrend': 'dt', 'spurious-mc': 'sm', 'spurious-real': 'sr', 'df-dist': 'dd', 'adf-power': 'ap',
       'kpss-sums': 'kp', 'breaks-sim': 'bs', 'breaks-real': 'br', 'gdp-ident': 'gi', 'gdp-diag': 'gd', 'interval-width': 'iw',
       'gdp-forecast': 'gf', 'overdiff': 'od', 'inflation': 'in', 'eurron': 'fx', 'us-gdp': 'us'}

SECTIONS = {
 'ro': dict(trend='Trend determinist și trend stochastic', spur='Regresia falsă', df='Testul Dickey–Fuller',
            kpss='Phillips–Perron, KPSS și folosirea lor împreună', brk='Rupturi structurale', arima='Modele ARIMA(p,d,q)',
            prac='ARIMA în practică'),
 'en': dict(trend='Deterministic and stochastic trends', spur='Spurious regression', df='The Dickey–Fuller test',
            kpss='Phillips–Perron, KPSS and their joint use', brk='Structural breaks', arima='ARIMA(p,d,q) models',
            prac='ARIMA in practice'),
}

L = {
 'ro': dict(file='ch3_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 3 · Rădăcini unitare și modele ARIMA', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, calculată în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas.',
   src='Simulări: șocuri din distribuția Normală, generate cu seed-urile codului capitolului; rezultatele Monte Carlo sînt cele din curs. '
       'PIB-ul României: Eurostat, volume înlănțuite (prețuri 2010), serie ajustată sezonier, în versiunea folosită în curs. IAPC: Eurostat, pînă în august 2026. '
       'EUR/RON: cursul de referință BNR; BET și S&amp;P 500: datele de piață ale cursului, pînă la 18 septembrie 2026. PIB-ul SUA: FRED GDPC1, pînă în T2 2026. '
       'Nilul: setul de date din statsmodels, 1871–1970.'),
 'en': dict(file='ch3.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 3 · Unit Roots and ARIMA Models', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step.',
   src='Simulations: shocks from the Normal distribution, drawn with the seeds of the chapter code; the Monte Carlo results are those of the lecture. '
       'Romanian GDP: Eurostat, chain-linked volumes (2010 prices), seasonally adjusted, in the version used in the lecture. HICP: Eurostat, up to August 2026. '
       'EUR/RON: BNR reference rate; BET and S&amp;P 500: the course market data, up to 18 September 2026. US GDP: FRED GDPC1, up to 2026Q2. '
       'Nile: the statsmodels data set, 1871–1970.'),
}

PQ = ['00', '01', '02', '10', '11', '12', '20', '21', '22']
PQ3 = PQ + ['30', '31', '32']


def _chart_widgets(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    return {
     'four-series': (slider('fs-end', s('Dreptele estimate pînă în', 'Straight lines fitted up to'), 19, 105, 1, 105) + play('fs'), charts('fs-chart')),
     'ts-ds': (slider('td-phi', s('φ al seriei staționare în jurul trendului', 'φ of the trend-stationary series'), 0, 1, 0.01, 0.8)
               + slider('td-sh', s('Mărimea șocului la t = 100', 'Size of the shock at t = 100'), 0, 10, 0.5, 6) + play('td'), charts('td-chart')),
     'detrend': (slider('dt-T', s('Lungimea eșantionului T', 'Sample length T'), 50, 200, 10, 200) + play('dt'), charts('dt-chart')),
     'spurious-mc': (select('sm-T', s('T în histograma din dreapta', 'T in the right histogram'), [(t, t) for t in ('25', '50', '100', '200', '500', '1000')], '200')
                     + play('sm'), charts('sm-chart')),
     'spurious-real': (slider('sr-y0', s('De la anul', 'From year'), 2005, 2022, 1, 2005) + slider('sr-y1', s('Pînă în anul', 'To year'), 2008, 2026, 1, 2026)
                       + play('sr'), charts('sr-chart')),
     'df-dist': (slider('dd-c', s('Valoarea critică folosită', 'Critical value used'), -4, -1, 0.005, -1.645) + play('dd'), charts('dd-chart')),
     'adf-power': (select('ap-T', s('Lungimea eșantionului T', 'Sample length T'), [(t, t) for t in ('50', '100', '250', '500', '1000')], '100') + play('ap'), charts('ap-chart')),
     'kpss-sums': (slider('kp-phi', s('φ al seriei AR(1)', 'φ of the AR(1) series'), 0, 1, 0.05, 0.5) + play('kp'), charts('kp-chart')),
     'breaks-sim': (slider('bs-d', s('Saltul mediei la t = 100', 'Jump of the mean at t = 100'), 0, 10, 0.5, 5) + play('bs'), charts('bs-chart')),
     'breaks-real': (select('br-reg', s('Ruptura permisă de testul Zivot–Andrews', 'Break allowed by the Zivot–Andrews test'),
                            [('c', s('în nivel', 'in the level')), ('t', s('în pantă', 'in the trend')), ('ct', s('în nivel și pantă', 'in the level and the trend'))], 'c')
                     + play('br'), charts('br-chart')),
     'gdp-ident': (slider('gi-end', s('Eșantionul se termină în', 'Sample ends in'), 20, 105, 1, 105) + play('gi'), charts('gi-chart')),
     'gdp-diag': (select('gd-mod', s('Modelul ARIMA(p,1,q) cu derivă', 'ARIMA(p,1,q) model with drift'),
                         [(k, f'ARIMA({k[0]},1,{k[1]})' + (s(' · cel mai mic AICc', ' · lowest AICc') if k == '00' else '') + (s(' · nu a convers', ' · not converged') if k == '22' else '')) for k in PQ], '00')
                  + play('gd'), charts('gd-chart')),
     'interval-width': (slider('iw-f0', s('φ al modelului AR(1)', 'φ of the AR(1)'), 0, 0.99, 0.01, 0.8) + slider('iw-f1', s('φ al modelului ARIMA(1,1,0)', 'φ of the ARIMA(1,1,0)'), -0.9, 0.95, 0.05, 0.5)
                        + slider('iw-th', s('θ al modelului ARIMA(0,1,1)', 'θ of the ARIMA(0,1,1)'), -0.95, 0.95, 0.05, -0.5) + play('iw'), charts('iw-chart')),
     'gdp-forecast': (slider('gf-o', s('Originea prognozei', 'Forecast origin'), 0, 42, 1, 42) + play('gf'), charts('gf-chart')),
     'overdiff': (select('od-s', s('Seria', 'Series'), [('ro', s('PIB-ul real al României', 'Romanian real GDP')), ('us', s('PIB-ul real al SUA', 'US real GDP'))], 'ro')
                  + select('od-d', s('Ordinul de diferențiere din dreapta', 'Order of differencing on the right'), [('2', 'd = 2'), ('3', 'd = 3')], '2') + play('od'), charts('od-chart')),
     'inflation': (select('in-m0', s('Modelul cu d = 0', 'Model with d = 0'), [(k, f'ARIMA({k[0]},0,{k[1]})' + (s(' · cel mai mic AICc', ' · lowest AICc') if k == '20' else '')) for k in PQ3], '20')
                   + select('in-m1', s('Modelul cu d = 1', 'Model with d = 1'), [(k, f'ARIMA({k[0]},1,{k[1]})' + (s(' · cel mai mic AICc', ' · lowest AICc') if k == '10' else '')
                                                                                   + (s(' · nu a convers', ' · not converged') if k in ('20', '32') else '')) for k in PQ3], '10')
                   + play('in'), charts('in-chart')),
     'eurron': (slider('fx-y', s('Perioada de test începe în ianuarie', 'Test period starts in January'), 2010, 2025, 1, 2024) + play('fx'), charts('fx-chart')),
     'us-gdp': (slider('us-y', s('Originea prognozei', 'Forecast origin'), 1990, 2019, 1, 2007) + play('us'), charts('us-chart')),
    }


QA = {
 'ro': {
  'four-series': ('Butonul ▶ mută sfîrșitul eșantionului pe care se estimează dreptele întrerupte din toate cele patru panouri. Unde se află PIB-ul din T2 2026 față de o dreaptă estimată pe datele de pînă în T3 2008?',
                  'Mult sub ea: dreapta estimată pînă în T3 2008 are panta 1,56% pe trimestru, iar PIB-ul din T2 2026 este cu 76 de puncte logaritmice (100 × diferența logaritmilor) sub prelungirea ei. '
                  'Pe tot eșantionul panta este 0,80% pe trimestru (circa 3,2% pe an), iar cea mai mare abatere, 17,5 puncte logaritmice, apare în T3 2008. '
                  'După recesiunea din 2009, PIB-ul nu a revenit la dreapta de dinainte: un argument pentru un trend stochastic, nu pentru o dreaptă fixă plus zgomot.'),
  'ts-ds': ('Butonul ▶ crește φ de la 0 la 1. Ce rămîne, la sfîrșitul eșantionului, din șocul de +6 în seria staționară în jurul trendului?',
            'Pentru φ < 1, aproape nimic: 6φ⁹⁹ este 1,5 · 10⁻⁹ la φ = 0,8 și 0,04 la φ = 0,95; abia la φ = 0,99 rămîn 2,2. La φ = 1 rămîne întregul șoc, 6, ca la mersul aleator: '
            'zgomotul AR(1) devine el însuși un mers aleator, iar cele două modele coincid. Cu φ = 0,8, efectul unui șoc unitar este 0,107 după 10 perioade, iar timpul de înjumătățire este 3,1 perioade.'),
  'detrend': ('Panta adevărată a mersului aleator este 0. Pe 2 000 de mersuri aleatoare cu T = 200, cît de des respinge testul t ipoteza „panta este 0” la 5%?',
              'În 90,4% din cazuri, nu în 5%; R² median este 0,44. Pe traiectoria din slide, R² = 0,61 și t = −17,7. Reziduurile au ρ̂(1) = 0,96 și rămîn pozitiv autocorelate pînă la lagul 36: '
              'cicluri lungi create de eliminarea trendului (Nelson și Kang). Cu T = 50, testul respinge tot în 82,0% din cazuri: eroarea nu dispare cu un eșantion mai scurt sau mai lung.'),
  'spurious-mc': ('În niveluri, testul t respinge b = 0 în 83,6% din eșantioane pentru T = 200. Ce se întîmplă cu rata de respingere cînd T crește la 1 000?',
                  'Crește, la 92,3%: |t| median urcă de la 6,3 la 14,2, aproximativ ca √T; R² median rămîne în jur de 0,17, iar DW median scade la 0,015. Mai multe date nu corectează o regresie falsă. '
                  'În diferențe, rata rămîne aproape de nivelul nominal: 4,65% la T = 200 și 4,5% la T = 1 000.'),
  'spurious-real': ('Pe 2005–2026, regresia IAPC pe S&amp;P 500 în niveluri dă t = 34,6 și R² = 0,82. Rămîne legătura dacă folosim doar 2005–2012?',
                    'Nu: pe 2005–2012, t = −0,72 și R² sub 0,01 în niveluri; pe 2013–2026, t = 33,2 și R² = 0,87. În 2005–2012 acțiunile au scăzut și au revenit, în timp ce prețurile au crescut; după 2013 ambele cresc. '
                    'În diferențe, pe tot eșantionul, t = −0,21 (p = 0,83) și R² = 0,0002. Regula lui Granger și Newbold semnalează problema: R² = 0,82 > DW = 0,03.'),
  'df-dist': ('Dacă folosim valoarea critică a distribuției Normale, −1,645, cît de des „respingem” o rădăcină unitară adevărată în regresia cu constantă și trend?',
              'În 77,4% din eșantioane (46,1% cu constantă, 9,8% fără constantă), față de 5% sub N(0, 1). Ca să respingem în 5% din cazuri, valoarea critică trebuie coborîtă la cuantila de 5% a distribuției Dickey–Fuller: '
              '−1,95, −2,86 și −3,42 în simulare; −1,94, −2,87 și −3,43 în tabelele MacKinnon. Cu −3,42, proporția pentru constantă și trend este 4,99%.'),
  'adf-power': ('Cu o serie trimestrială de 25 de ani (T = 100) și φ = 0,95, cît de des respinge testul Dickey–Fuller cu constantă rădăcina unitară la 5%?',
                'În 12,5% din cazuri: testul aproape niciodată nu deosebește φ = 0,95 de φ = 1. Cu T = 250, 47,4%; cu T = 500, 97,25%. La φ = 1, proporțiile sînt mărimea testului: 5,3%, 5,2% și 4,65% cu constantă. '
                'Un trend inutil costă putere: la T = 100 și φ = 0,9, 35,1% cu constantă și 19,4% cu constantă și trend.'),
  'kpss-sums': ('Butonul ▶ crește φ al seriei AR(1) de la 0 la 1. De la ce valoare respinge KPSS staționaritatea la 5%?',
                'Abia de la φ = 0,95: KPSS = 0,565 > 0,463 (p = 0,027). La φ = 0,9 statistica este 0,326 și testul nu respinge, deși seria este foarte persistentă; la φ = 1 urcă la 1,213. '
                'Pentru mersul aleator din panou, KPSS = 0,733 (p = 0,011). Sumele parțiale ale unei serii persistente se îndepărtează mult de 0, iar η crește.'),
  'breaks-sim': ('Butonul ▶ crește saltul mediei de la 0 la 10. De la ce mărime a saltului nu mai respinge ADF rădăcina unitară la 5%, pe această serie staționară?',
                 'De la un salt de 3,5 (p = 0,056); la 3, p = 0,030. La saltul din curs, 5, p = 0,17, iar pe 2 000 de serii cu același salt testul DF respinge în 71,5% din cazuri (fără salt: 100%); la saltul 7, doar în 0,55%. '
                 'Zivot–Andrews include ruptura și respinge pentru orice salt: la saltul 5, ZA = −7,47 (un lag ales după AIC), mult sub −4,81, cu nivelul schimbat de la t = 100. '
                 'Curba din dreapta este calculată fără laguri: minimul ei, −8,37, este atins la t = 100; −7,47 este statistica testului, nu minimul curbei.'),
  'breaks-real': ('Pentru EUR/RON, ZA cu ruptură în nivel dă −6,89. Ce dă varianta cu ruptură doar în pantă?',
                  'ZA = −4,40, cu valoarea critică de 5% −4,41 (p = 0,051): rădăcina unitară nu mai este respinsă la 5%, iar data aleasă se mută în iulie 2009. Cu ruptură în nivel și pantă: −6,83, sub −5,07, în octombrie 2008. '
                  'Pentru Nil, toate cele trei variante resping (−6,86; −5,68; −6,84); rupturile în nivel sînt datate în 1898, cea în pantă în 1914. Rezultatul depinde de forma presupusă a rupturii.'),
  'gdp-ident': ('Pe eșantionul complet, nicio autocorelație a creșterii nu iese din bandă. Rămîne așa dacă eșantionul se oprește în T4 2008, înainte de criză?',
                'Nu întru totul: cu 35 de rate de creștere (T2 2000–T4 2008), media este 1,57% pe trimestru, ρ̂(1) = −0,27, iar o valoare ACF iese din banda ±0,33; Ljung–Box Q*(8) = 11,1 (p = 0,20) nu respinge însă zgomotul alb. '
                'Pe tot eșantionul: media 0,81%, ρ̂(1) = −0,05, ρ̂(2) = −0,14, Q*(8) = 7,3 (p = 0,51); pînă în T4 2019, Q*(8) = 4,2 (p = 0,84).'),
  'gdp-diag': ('ARIMA(2,1,2) are cel mai mic Q*(8) dintre cele nouă modele, 3,28. Este acesta un argument pentru el?',
               'Nu: optimizarea nu a convers, iar AICc este 495,4, față de 491,9 pentru ARIMA(0,1,0) cu derivă. ARIMA(0,1,0) lasă deja reziduuri necorelate: Q*(8) = 7,3 (p = 0,51), iar Q*(12) are p = 0,54. '
               'Parametrii în plus reduc mecanic autocorelația reziduurilor. Ce rămîne în toate modelele sînt cozile groase: pentru ARIMA(0,1,0), kurtosis-ul 7,9, Jarque–Bera 145 și reziduul −11,0 în T1 2009.'),
  'interval-width': ('Butonul ▶ mișcă θ al modelului ARIMA(0,1,1) de la −0,95 la 0,95. Cum se schimbă semilățimea intervalului la h = 20?',
                     'Crește cu θ: 2,14σ la θ = −0,9, 4,70σ la θ = −0,5, 8,77σ la θ = 0 (mersul aleator) și 12,96σ la θ = 0,5. Varianța este σ²[1 + (h − 1)(1 + θ)²]: un θ negativ compensează o parte din fiecare șoc. '
                     'Pentru ARIMA(1,1,0), semilățimea la h = 20 este 16,78σ cu φ = 0,5 (graficul) și 20,53σ cu φ = 0,6 (exemplul rezolvat, unde la h = 3 este 5,33σ, adică 10,66 cu σ = 2). AR(1) cu φ = 0,8 se oprește la 3,27σ.'),
  'gdp-forecast': ('Cu originea în T4 2019, cele două modele prognozează PIB-ul pe 12 trimestre. Care a avut eroarea mai mică pe trimestrele observate?',
                   'Modelul staționar în jurul trendului, dar la mică distanță: RMSE 7,2 puncte logaritmice, față de 7,6 pentru ARIMA(0,1,0) cu derivă; ambele au ratat scăderea din 2020. Din T4 2015 diferența este mai mare (2,6 față de 4,7), '
                   'iar din T2 2023 ordinea se inversează (6,5 față de 5,6). Din T2 2026, ARIMA(0,1,0) dă 53,8 mld. EUR (prețuri 2010) după 12 trimestre, cu semilățimea de 95% de 16,8 puncte logaritmice; modelul cu trend dă 56,3 mld. și 11,2. '
                   'ADF și KPSS clasifică nivelul PIB ca I(1) (p = 0,43; 0,198 > 0,146), dar puterea mică a testelor nu exclude modelul cu trend: alegerea lui d schimbă prognoza și intervalul.'),
  'overdiff': ('Pentru PIB-ul SUA, ce simptome de supradiferențiere apar la a doua diferențiere?',
               'Toate trei: ρ̂(1) scade de la 0,13 la −0,49, varianța crește de la 1,22 la 2,12, iar un MA(1) estimat pe Δ² dă θ̂ = −0,986 (SE 0,008). Pentru România: ρ̂(1) de la −0,05 la −0,43, varianța de la 6,16 la 12,80, θ̂ = −0,987 (SE 0,080). '
               'O a treia diferențiere agravează simptomele: pentru România, varianța urcă la 35,6 și ρ̂(1) la −0,57.'),
  'inflation': ('Cel mai mic AICc pentru d = 1, 451,2 (ARIMA(1,1,0)), este sub cel pentru d = 0, 455,6 (ARIMA(2,0,0)). Arată aceasta că d = 1 este alegerea mai bună?',
                'Nu: AICc nu compară modele cu d diferit, deoarece cu d = 1 verosimilitatea este calculată pentru Δπ, o altă serie. Alegem d prin teste și judecată: KPSS (0,404 < 0,463) păstrează d = 0, ADF (p = 0,32) sugerează d = 1. '
                'Prognozele după 24 de luni sînt apropiate, 5,1% și 5,2%, dar intervalele de 95% diferă: [−0,6; 10,9] și [−2,9; 13,3]. ARIMA(3,1,2) are AICc 441,1, dar optimizarea nu a convers.'),
  'eurron': ('Cu începutul perioadei de test în ianuarie 2015, mai bate ARIMA(1,1,0) mersul aleator la o zi?',
             'Nu: RMSE 0,1307, față de 0,1304 pentru mersul aleator, o pierdere de 0,19%. Cu testul din 2019, 2020, 2022 sau 2024, cîștigul este 2,1–2,3%; în curs (din 2024), 0,1300 față de 0,1330, iar pe 20 de zile 0,638 față de 0,667 (4,3%). '
             'Cîștigul depinde de perioada aleasă și provine din cîteva zile cu salturi ale cursului; o comparație formală cere testul Diebold–Mariano.'),
  'us-gdp': ('Cu originea în T4 2007, modelul staționar în jurul trendului greșește cu −30 de puncte logaritmice la final. Este eroarea specifică lui 2007? Mutați originea în T4 2000.',
             'Nu: din T4 2000, abaterea la final este −34,7 pentru modelul cu trend (semilățimea intervalului de 95%: 7,9) și −34,5 pentru ARIMA(1,1,0) cu derivă (28,1). Din T4 2007: −30,1 (8,2) și −25,4 (22,9); din T4 2019: −8,0 (11,1) și −5,0 (13,2). '
             'Din 2000 și din 2007, niciun interval nu acoperă rezultatul; din 2019, ambele îl acoperă. Ambele modele prelungesc deriva de dinaintea originii, iar creșterea de după 2008 a fost mai lentă.'),
 },
 'en': {
  'four-series': ('The ▶ button moves the end of the sample on which the dashed straight lines of all four panels are fitted. Where is GDP in 2026Q2 relative to a line fitted on the data up to 2008Q3?',
                  'Far below it: the line fitted up to 2008Q3 has a slope of 1.56% per quarter, and GDP in 2026Q2 is 76 log points (100 × the difference of logs) below its extension. '
                  'On the full sample the slope is 0.80% per quarter (about 3.2% a year), and the largest deviation, 17.5 log points, occurs in 2008Q3. '
                  'After the 2009 recession GDP did not return to the earlier line: an argument for a stochastic trend, not for a fixed line plus noise.'),
  'ts-ds': ('The ▶ button raises φ from 0 to 1. What is left, at the end of the sample, of the +6 shock in the trend-stationary series?',
            'For φ < 1, almost nothing: 6φ⁹⁹ is 1.5 · 10⁻⁹ at φ = 0.8 and 0.04 at φ = 0.95; only at φ = 0.99 does 2.2 remain. At φ = 1 the whole shock, 6, remains, as for the random walk: '
            'the AR(1) noise becomes a random walk itself, and the two models coincide. With φ = 0.8, the effect of a unit shock is 0.107 after 10 periods, and the half-life is 3.1 periods.'),
  'detrend': ('The true slope of the random walk is 0. On 2,000 random walks with T = 200, how often does the t-test reject “the slope is 0” at 5%?',
              'In 90.4% of the cases, not in 5%; the median R² is 0.44. On the path of the slide, R² = 0.61 and t = −17.7. The residuals have ρ̂(1) = 0.96 and stay positively autocorrelated up to lag 36: '
              'long cycles created by detrending (Nelson and Kang). With T = 50 the test still rejects in 82.0% of the cases: the error does not go away with a shorter or a longer sample.'),
  'spurious-mc': ('In levels, the t-test rejects b = 0 in 83.6% of the samples for T = 200. What happens to the rejection rate when T rises to 1,000?',
                  'It rises, to 92.3%: the median |t| goes from 6.3 to 14.2, roughly like √T; the median R² stays around 0.17, and the median DW falls to 0.015. More data do not correct a spurious regression. '
                  'In differences, the rate stays close to the nominal level: 4.65% at T = 200 and 4.5% at T = 1,000.'),
  'spurious-real': ('On 2005–2026, the regression of the HICP on the S&amp;P 500 in levels gives t = 34.6 and R² = 0.82. Does the relation remain if we use only 2005–2012?',
                    'No: on 2005–2012, t = −0.72 and R² is below 0.01 in levels; on 2013–2026, t = 33.2 and R² = 0.87. In 2005–2012 stocks fell and recovered while prices rose; after 2013 both rise. '
                    'In differences, on the full sample, t = −0.21 (p = 0.83) and R² = 0.0002. The Granger–Newbold rule flags the problem: R² = 0.82 > DW = 0.03.'),
  'df-dist': ('If we use the critical value of the Normal distribution, −1.645, how often do we “reject” a true unit root in the regression with constant and trend?',
              'In 77.4% of the samples (46.1% with a constant, 9.8% without a constant), against 5% under N(0, 1). To reject in 5% of the cases, the critical value must move down to the 5% quantile of the Dickey–Fuller distribution: '
              '−1.95, −2.86 and −3.42 in the simulation; −1.94, −2.87 and −3.43 in the MacKinnon tables. With −3.42, the share for constant and trend is 4.99%.'),
  'adf-power': ('With a quarterly series of 25 years (T = 100) and φ = 0.95, how often does the Dickey–Fuller test with a constant reject the unit root at 5%?',
                'In 12.5% of the cases: the test almost never tells φ = 0.95 from φ = 1. With T = 250, 47.4%; with T = 500, 97.25%. At φ = 1 the shares are the size of the test: 5.3%, 5.2% and 4.65% with a constant. '
                'An unnecessary trend costs power: at T = 100 and φ = 0.9, 35.1% with a constant and 19.4% with constant and trend.'),
  'kpss-sums': ('The ▶ button raises φ of the AR(1) series from 0 to 1. From which value does KPSS reject stationarity at 5%?',
                'Only from φ = 0.95: KPSS = 0.565 > 0.463 (p = 0.027). At φ = 0.9 the statistic is 0.326 and the test does not reject, although the series is very persistent; at φ = 1 it rises to 1.213. '
                'For the random walk of the panel, KPSS = 0.733 (p = 0.011). The partial sums of a persistent series wander far from 0, and η grows.'),
  'breaks-sim': ('The ▶ button raises the jump of the mean from 0 to 10. From which size of the jump does ADF stop rejecting the unit root at 5%, on this stationary series?',
                 'From a jump of 3.5 (p = 0.056); at 3, p = 0.030. At the jump of the lecture, 5, p = 0.17, and on 2,000 series with the same jump the DF test rejects in 71.5% of the cases (without the jump: 100%); at a jump of 7, only in 0.55%. '
                 'Zivot–Andrews includes the break and rejects for every jump: at a jump of 5, ZA = −7.47 (one lag chosen by AIC), far below −4.81, with the level changing from t = 100. '
                 'The curve on the right is computed without lags: its minimum, −8.37, is reached at t = 100; −7.47 is the test statistic, not the minimum of the curve.'),
  'breaks-real': ('For EUR/RON, ZA with a break in the level gives −6.89. What does the variant with a break in the trend only give?',
                  'ZA = −4.40, with a 5% critical value of −4.41 (p = 0.051): the unit root is no longer rejected at 5%, and the chosen date moves to July 2009. With a break in the level and the trend: −6.83, below −5.07, in October 2008. '
                  'For the Nile, all three variants reject (−6.86, −5.68, −6.84); the level breaks are dated 1898, the trend break 1914. The result depends on the assumed form of the break.'),
  'gdp-ident': ('On the full sample, no autocorrelation of growth leaves the band. Does this still hold if the sample stops in 2008Q4, before the crisis?',
                'Not entirely: with 35 growth rates (2000Q2–2008Q4), the mean is 1.57% per quarter, ρ̂(1) = −0.27, and one ACF value leaves the ±0.33 band; the Ljung–Box Q*(8) = 11.1 (p = 0.20) does not reject white noise, though. '
                'On the full sample: mean 0.81%, ρ̂(1) = −0.05, ρ̂(2) = −0.14, Q*(8) = 7.3 (p = 0.51); up to 2019Q4, Q*(8) = 4.2 (p = 0.84).'),
  'gdp-diag': ('ARIMA(2,1,2) has the smallest Q*(8) of the nine models, 3.28. Is this an argument for it?',
               'No: the optimisation did not converge, and its AICc is 495.4, against 491.9 for ARIMA(0,1,0) with drift. ARIMA(0,1,0) already leaves uncorrelated residuals: Q*(8) = 7.3 (p = 0.51), and Q*(12) has p = 0.54. '
               'Extra parameters reduce the residual autocorrelation mechanically. What remains in every model are the fat tails: for ARIMA(0,1,0), kurtosis 7.9, Jarque–Bera 145 and the residual of −11.0 in 2009Q1.'),
  'interval-width': ('The ▶ button moves θ of the ARIMA(0,1,1) from −0.95 to 0.95. How does the half-width of the interval at h = 20 change?',
                     'It rises with θ: 2.14σ at θ = −0.9, 4.70σ at θ = −0.5, 8.77σ at θ = 0 (the random walk) and 12.96σ at θ = 0.5. The variance is σ²[1 + (h − 1)(1 + θ)²]: a negative θ offsets part of every shock. '
                     'For the ARIMA(1,1,0), the half-width at h = 20 is 16.78σ with φ = 0.5 (the chart) and 20.53σ with φ = 0.6 (the worked example, where at h = 3 it is 5.33σ, i.e. 10.66 with σ = 2). The AR(1) with φ = 0.8 levels off at 3.27σ.'),
  'gdp-forecast': ('With the origin in 2019Q4, the two models forecast GDP for 12 quarters. Which one had the smaller error on the observed quarters?',
                   'The trend-stationary model, but only just: RMSE 7.2 log points, against 7.6 for ARIMA(0,1,0) with drift; both missed the 2020 drop. From 2015Q4 the gap is larger (2.6 against 4.7), '
                   'and from 2023Q2 the ranking reverses (6.5 against 5.6). From 2026Q2, ARIMA(0,1,0) gives 53.8 bn EUR (2010 prices) after 12 quarters, with a 95% half-width of 16.8 log points; the trend model gives 56.3 bn and 11.2. '
                   'ADF and KPSS classify the level of GDP as I(1) (p = 0.43; 0.198 > 0.146), but the low power of the tests does not rule out the trend model: the choice of d changes the forecast and its interval.'),
  'overdiff': ('For US GDP, which symptoms of over-differencing appear with a second difference?',
               'All three: ρ̂(1) falls from 0.13 to −0.49, the variance rises from 1.22 to 2.12, and an MA(1) fitted to Δ² gives θ̂ = −0.986 (SE 0.008). For Romania: ρ̂(1) from −0.05 to −0.43, the variance from 6.16 to 12.80, θ̂ = −0.987 (SE 0.080). '
               'A third difference makes the symptoms worse: for Romania the variance rises to 35.6 and ρ̂(1) to −0.57.'),
  'inflation': ('The lowest AICc for d = 1, 451.2 (ARIMA(1,1,0)), is below the one for d = 0, 455.6 (ARIMA(2,0,0)). Does this show that d = 1 is the better choice?',
                'No: AICc does not compare models with different d, because with d = 1 the likelihood is computed for Δπ, another series. We choose d by tests and judgement: KPSS (0.404 < 0.463) keeps d = 0, ADF (p = 0.32) suggests d = 1. '
                'The forecasts after 24 months are close, 5.1% and 5.2%, but the 95% intervals differ: [−0.6, 10.9] and [−2.9, 13.3]. ARIMA(3,1,2) has an AICc of 441.1, but its optimisation did not converge.'),
  'eurron': ('With the test period starting in January 2015, does the ARIMA(1,1,0) still beat the random walk one day ahead?',
             'No: RMSE 0.1307, against 0.1304 for the random walk, a loss of 0.19%. With the test period from 2019, 2020, 2022 or 2024, the gain is 2.1–2.3%; in the lecture (from 2024), 0.1300 against 0.1330, and over 20 days 0.638 against 0.667 (4.3%). '
             'The gain depends on the period chosen and comes from a few days with jumps of the rate; a formal comparison needs the Diebold–Mariano test.'),
  'us-gdp': ('With the origin in 2007Q4, the trend-stationary model misses by −30 log points at the end. Is the error specific to 2007? Move the origin to 2000Q4.',
             'No: from 2000Q4, the gap at the end is −34.7 for the trend model (half-width of the 95% interval: 7.9) and −34.5 for ARIMA(1,1,0) with drift (28.1). From 2007Q4: −30.1 (8.2) and −25.4 (22.9); from 2019Q4: −8.0 (11.1) and −5.0 (13.2). '
             'From 2000 and from 2007 neither interval covers the outcome; from 2019 both do. Both models extend the drift from before the origin, and growth after 2008 was slower.'),
 },
}

# readouts: JavaScript functions, one per block
TEXT_JS = {
 'ro': r"""window.CH3_TEXT = {
 play: 'Pornește', pause: 'Pauză', wm: ' cu medie',
 q: s => `T${s[5]} ${s.slice(0, 4)}`,
 m: d => ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'][+d.slice(5, 7) - 1] + ' ' + d.slice(0, 4),
 dmy: d => `${+d.slice(8, 10)} ` + ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'][+d.slice(5, 7) - 1] + ' ' + d.slice(0, 4),
 fs1: (a, b, sl, yr) => `Dreapta este estimată pe ${a} – ${b}: panta ${sl}% pe trimestru (circa ${yr}% pe an).`,
 fs2: (q, d, mx, qm) => `Abaterea PIB-ului de la dreaptă în ${q}: ${d} puncte logaritmice (100 × diferența logaritmilor); cea mai mare abatere în valoare absolută: ${mx}, în ${qm}.`,
 fs3: (h, f0, f1, b, s) => `IAPC: ×${h} din 2000; EUR/RON: de la ${f0} la ${f1} lei; BET: +${b} puncte logaritmice, S&P 500: +${s} (pînă la 18 septembrie 2026).`,
 td1: (p, a, b) => `φ = ${p}: la sfîrșitul eșantionului (t = 199), traiectoria cu șoc se află cu ${a} peste cea fără șoc pentru seria staționară în jurul trendului și cu ${b} pentru mersul aleator.`,
 td2: (a, b, h) => `Efectul unui șoc unitar: ${a} după 10 perioade și ${b} după 20` + (h === null ? '; la φ = 1 efectul rămîne 1, iar cele două modele coincid.' : `; timpul de înjumătățire ${h} perioade.`),
 dt1: (n, r2, t) => `T = ${n}: R² = ${r2}, statistica t a pantei ${t}, deși panta adevărată este 0.`,
 dt2: (r1, k) => `Reziduurile: ρ̂(1) = ${r1}; ` + (k ? `prima autocorelație negativă apare la lagul ${k}.` : 'nicio autocorelație negativă pînă la lagul 60.'),
 dt3: (r, m) => `Pe 2 000 de mersuri aleatoare de aceeași lungime: |t| > 1,96 în ${r}% din cazuri, R² median ${m}.`,
 sm1: (T, r, t, r2, dw) => `T = ${T}, niveluri: b = 0 respins în ${r}% din eșantioane; |t| median ${t}, R² median ${r2}, DW median ${dw}.`,
 sm2: (r, t, r2, dw) => `Diferențe: ${r}% respingeri; |t| median ${t}, R² median ${r2}, DW median ${dw}.`,
 sr1: (a, b, n) => `${a} – ${b}: ${n} luni.`,
 sr2: (b, t, r2, dw, f) => `Niveluri: panta ${b}, t = ${t}, R² = ${r2}, DW = ${dw}` + (f ? ': R² > DW, semnul unei regresii false.' : '.'),
 sr3: (b, t, p, r2, dw) => `Diferențe: panta ${b}, t = ${t} (${p}), R² = ${r2}, DW = ${dw}.`,
 dd1: (c, a, b, d, nr) => `Cu valoarea critică ${c}, o rădăcină unitară adevărată este „respinsă” în ${a} din eșantioane fără constantă, ${b} cu constantă și ${d} cu constantă și trend; sub N(0, 1), proporția ar fi ${nr}.`,
 dd2: (a, b, c) => `Valorile critice de 5% (MacKinnon, T = 250): ${a} fără constantă, ${b} cu constantă, ${c} cu constantă și trend.`,
 ap0: (T, l, a, b) => `T = ${T}` + (l ? ' (rezultatele din curs)' : ' (o rulare suplimentară a codului capitolului, cu aceeași sămînță)') + `; valorile critice de 5%: ${a} cu constantă, ${b} cu constantă și trend.`,
 ap1: (a, b) => `Mărimea testului (φ = 1): ${a} cu constantă, ${b} cu constantă și trend.`,
 ap2: (a, b, c, d) => `Puterea la φ = 0,95: ${a} cu constantă, ${b} cu constantă și trend; la φ = 0,9: ${c} și ${d}.`,
 kp: (p, e, L, pv, r) => `AR(1) cu φ = ${p}: KPSS = ${e} (lățimea de bandă ${L}, ${pv}): ${r ? 'respinge' : 'nu respinge'} staționaritatea la 5% (valoarea critică 0,463).`,
 kpR: (e, L, pv) => `Mersul aleator: KPSS = ${e} (lățimea de bandă ${L}, ${pv}).`,
 bs1: (d, a, k, p, c, r) => `Saltul ${d}: ADF cu constantă τ = ${a} (numărul de laguri ales după AIC: ${k}), ${p}; valoarea critică de 5% ${c}. Pe 2 000 de serii cu același salt, testul DF respinge în ${r} din cazuri.`,
 bs2: (z, k, b, c, m, a) => `Zivot–Andrews (ruptură în nivel, laguri alese după AIC: ${k}): ZA = ${z}, cu nivelul schimbat de la t = ${b}; valoarea critică de 5% ${c}. Curba din dreapta, fără laguri, are minimul ${m} la data ${a}.`,
 br0: r => 'Ruptura permisă: ' + { c: 'în nivel', t: 'în pantă', ct: 'în nivel și pantă' }[r] + '.',
 brN: (z, c, p, y, m0, m1, ap, k) => `Nilul: ZA = ${z} (valoarea critică de 5% ${c}, ${p}), ruptură în ${y}; media ${m0} înainte, ${m1} după. Fără ruptură: ADF p = ${ap}, KPSS ${k}: ambele resping.`,
 brE: (z, c, p, d, m0, m1, ap) => `EUR/RON: ZA = ${z} (valoarea critică de 5% ${c}, ${p}), ruptură în ${d}; media ${m0} lei înainte, ${m1} lei după. ADF fără ruptură: p = ${ap}.`,
 gi1: (a, b, n, m, y, s) => `${a} – ${b}: ${n} trimestre; creșterea medie ${m}% pe trimestru (circa ${y}% pe an), abaterea standard ${s}.`,
 gi2: (r1, r2, p2, b, k) => `ρ̂(1) = ${r1}, ρ̂(2) = ${r2}, PACF la lagul 2: ${p2}; banda ±${b}; valori ACF în afara benzii: ${k}.`,
 gi3: (q, p) => `Ljung–Box pentru creștere: Q*(8) = ${q}, ${p}.`,
 gi4: (a, qa, b, qb) => `Cele mai mari scăderi: ${a}% în ${qa} și ${b}% în ${qb}.`,
 gd1: (p, q, a, c, s) => `ARIMA(${p},1,${q}) cu derivă: AICc ${a}${c ? '' : ' (optimizarea nu a convers)'}, σ̂ = ${s}.`,
 gd2: (q, df, p, p12, all) => `Ljung–Box: Q*(8) = ${q} cu ${df} grade de libertate, ${p}; Q*(12): ${p12}; ${all ? 'toate p-value-urile din grafic sînt peste 5%' : 'cel puțin un p-value din grafic este sub 5%'}.`,
 gd3: (k, jb, mn, q) => `Kurtosis-ul ${k}, Jarque–Bera ${jb}; cel mai mare reziduu negativ: ${mn} în ${q}.`,
 iw1: (f, a, b, c, l) => `AR(1), φ = ${f}: semilățimea ${a}σ la h = 1, ${b}σ la h = 4, ${c}σ la h = 20` + (l === null ? '.' : `; limita 1,96σ/√(1 − φ²) = ${l}σ.`),
 iw2: (a, b, c) => `Mersul aleator: ${a}σ, ${b}σ și ${c}σ (crește ca √h).`,
 iw3: (f, a, b, c) => `ARIMA(1,1,0), φ = ${f}: ${a}σ, ${b}σ și ${c}σ.`,
 iw4: (t, a, b, c) => `ARIMA(0,1,1), θ = ${t}: ${a}σ, ${b}σ și ${c}σ.`,
 iw5: (a, b, c) => `ARIMA(0,2,0): ${a}σ, ${b}σ și ${c}σ (crește ca h<sup>3/2</sup>).`,
 gf0: (o, v) => `Originea ${o}; ultima valoare ${v} mld. EUR (prețuri 2010).`,
 gf: (k, l, lo, hi, a, b, c, rm, n) => (k === 'ds' ? 'ARIMA(0,1,0) cu derivă' : 'Trend liniar cu erori AR(1)') + `: ${l} mld. EUR după 12 trimestre, intervalul de 95% [${lo}; ${hi}]; semilățimea de 95%: ${a}, ${b} și ${c} puncte logaritmice la h = 1, 4 și 12.`
   + (rm === null ? '' : ` Pe cele ${n} trimestre observate după origine: RMSE ${rm} puncte logaritmice.`),
 od1: (s, r1, v1, d, r2, v2) => (s === 'ro' ? 'România' : 'SUA') + `: d = 1: ρ̂(1) = ${r1}, varianța ${v1}; d = ${d}: ρ̂(1) = ${r2}, varianța ${v2}.`,
 od2: (d, t, se) => `MA(1) estimat pe diferența de ordinul ${d}: θ̂ = ${t} (SE ${se}).`,
 inm: (n, a, c, h12, h24, lo, hi, m) => `${n}: AICc ${a}${c ? '' : ' (optimizarea nu a convers)'}; prognoza ${h12}% după 12 luni și ${h24}% în ${m}, intervalul de 95% [${lo}; ${hi}].`,
 inmu: m => ` Media estimată: ${m}%.`,
 inb: (a, b) => `Cel mai mic AICc între modelele care au convers: ${a} pentru d = 0, ${b} pentru d = 1. AICc nu compară modele cu d diferit: cu d = 1 verosimilitatea este calculată pentru Δπ.`,
 fx0: (n, d, m, p) => `Parametrii sînt estimați pe ${n} variații zilnice, pînă la ${d}; ${m} prognoze cu o zi înainte; φ̂ = ${p}.`,
 fx1: (a, b, c, g) => `RMSE la o zi (100 ln): mers aleator ${a}, cu derivă ${b}, ARIMA(1,1,0) ${c}; cîștigul ARIMA față de mersul aleator: ${g}%.`,
 fx2: (n, a, b, c, g) => `La 20 de zile (${n} ferestre fără suprapunere): ${a}, ${b} și ${c}; cîștigul ARIMA: ${g}%.`,
 us0: (o, a, b) => `Prognoze din ${o}; abaterea rezultatului de la prognoză în ${a} și în ${b}:`,
 us: (k, a, b, h, c) => (k === 'ts' ? 'modelul staționar în jurul trendului' : 'ARIMA(1,1,0) cu derivă') + `: ${a} și ${b} puncte logaritmice; semilățimea intervalului de 95% la final ${h}: rezultatul ${c ? 'este în interval' : 'este în afara intervalului'}.`
};
""",
 'en': r"""window.CH3_TEXT = {
 play: 'Play', pause: 'Pause', wm: ' with mean',
 q: s => s,
 m: d => ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][+d.slice(5, 7) - 1] + ' ' + d.slice(0, 4),
 dmy: d => `${+d.slice(8, 10)} ` + ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][+d.slice(5, 7) - 1] + ' ' + d.slice(0, 4),
 fs1: (a, b, sl, yr) => `The line is fitted on ${a} – ${b}: slope ${sl}% per quarter (about ${yr}% a year).`,
 fs2: (q, d, mx, qm) => `Deviation of GDP from the line in ${q}: ${d} log points (100 × the difference of logs); largest absolute deviation: ${mx}, in ${qm}.`,
 fs3: (h, f0, f1, b, s) => `HICP: ×${h} since 2000; EUR/RON: from ${f0} to ${f1} lei; BET: +${b} log points, S&P 500: +${s} (up to 18 September 2026).`,
 td1: (p, a, b) => `φ = ${p}: at the end of the sample (t = 199), the path with the shock lies ${a} above the path without it for the trend-stationary series and ${b} for the random walk.`,
 td2: (a, b, h) => `Effect of a unit shock: ${a} after 10 periods and ${b} after 20` + (h === null ? '; at φ = 1 the effect stays 1, and the two models coincide.' : `; half-life ${h} periods.`),
 dt1: (n, r2, t) => `T = ${n}: R² = ${r2}, t-statistic of the slope ${t}, although the true slope is 0.`,
 dt2: (r1, k) => `Residuals: ρ̂(1) = ${r1}; ` + (k ? `the first negative autocorrelation is at lag ${k}.` : 'no negative autocorrelation up to lag 60.'),
 dt3: (r, m) => `On 2,000 random walks of the same length: |t| > 1.96 in ${r}% of the cases, median R² ${m}.`,
 sm1: (T, r, t, r2, dw) => `T = ${T}, levels: b = 0 rejected in ${r}% of the samples; median |t| ${t}, median R² ${r2}, median DW ${dw}.`,
 sm2: (r, t, r2, dw) => `Differences: ${r}% rejections; median |t| ${t}, median R² ${r2}, median DW ${dw}.`,
 sr1: (a, b, n) => `${a} – ${b}: ${n} months.`,
 sr2: (b, t, r2, dw, f) => `Levels: slope ${b}, t = ${t}, R² = ${r2}, DW = ${dw}` + (f ? ': R² > DW, the sign of a spurious regression.' : '.'),
 sr3: (b, t, p, r2, dw) => `Differences: slope ${b}, t = ${t} (${p}), R² = ${r2}, DW = ${dw}.`,
 dd1: (c, a, b, d, nr) => `With the critical value ${c}, a true unit root is “rejected” in ${a} of the samples without a constant, ${b} with a constant and ${d} with constant and trend; under N(0, 1) the share would be ${nr}.`,
 dd2: (a, b, c) => `5% critical values (MacKinnon, T = 250): ${a} without a constant, ${b} with a constant, ${c} with constant and trend.`,
 ap0: (T, l, a, b) => `T = ${T}` + (l ? ' (the results of the lecture)' : ' (an extra run of the chapter code, with the same seed)') + `; 5% critical values: ${a} with a constant, ${b} with constant and trend.`,
 ap1: (a, b) => `Size of the test (φ = 1): ${a} with a constant, ${b} with constant and trend.`,
 ap2: (a, b, c, d) => `Power at φ = 0.95: ${a} with a constant, ${b} with constant and trend; at φ = 0.9: ${c} and ${d}.`,
 kp: (p, e, L, pv, r) => `AR(1) with φ = ${p}: KPSS = ${e} (bandwidth ${L}, ${pv}): ${r ? 'rejects' : 'does not reject'} stationarity at 5% (critical value 0.463).`,
 kpR: (e, L, pv) => `Random walk: KPSS = ${e} (bandwidth ${L}, ${pv}).`,
 bs1: (d, a, k, p, c, r) => `Jump ${d}: ADF with a constant τ = ${a} (number of lags chosen by AIC: ${k}), ${p}; 5% critical value ${c}. On 2,000 series with the same jump, the DF test rejects in ${r} of the cases.`,
 bs2: (z, k, b, c, m, a) => `Zivot–Andrews (break in the level, lags chosen by AIC: ${k}): ZA = ${z}, with the level changing from t = ${b}; 5% critical value ${c}. The curve on the right, without lags, has its minimum ${m} at date ${a}.`,
 br0: r => 'Break allowed: ' + { c: 'in the level', t: 'in the trend', ct: 'in the level and the trend' }[r] + '.',
 brN: (z, c, p, y, m0, m1, ap, k) => `Nile: ZA = ${z} (5% critical value ${c}, ${p}), break in ${y}; mean ${m0} before, ${m1} after. Without a break: ADF p = ${ap}, KPSS ${k}: both reject.`,
 brE: (z, c, p, d, m0, m1, ap) => `EUR/RON: ZA = ${z} (5% critical value ${c}, ${p}), break in ${d}; mean ${m0} lei before, ${m1} lei after. ADF without a break: p = ${ap}.`,
 gi1: (a, b, n, m, y, s) => `${a} – ${b}: ${n} quarters; mean growth ${m}% per quarter (about ${y}% a year), standard deviation ${s}.`,
 gi2: (r1, r2, p2, b, k) => `ρ̂(1) = ${r1}, ρ̂(2) = ${r2}, PACF at lag 2: ${p2}; band ±${b}; ACF values outside the band: ${k}.`,
 gi3: (q, p) => `Ljung–Box for growth: Q*(8) = ${q}, ${p}.`,
 gi4: (a, qa, b, qb) => `Largest drops: ${a}% in ${qa} and ${b}% in ${qb}.`,
 gd1: (p, q, a, c, s) => `ARIMA(${p},1,${q}) with drift: AICc ${a}${c ? '' : ' (the optimisation did not converge)'}, σ̂ = ${s}.`,
 gd2: (q, df, p, p12, all) => `Ljung–Box: Q*(8) = ${q} with ${df} degrees of freedom, ${p}; Q*(12): ${p12}; ${all ? 'all p-values in the chart are above 5%' : 'at least one p-value in the chart is below 5%'}.`,
 gd3: (k, jb, mn, q) => `Kurtosis ${k}, Jarque–Bera ${jb}; largest negative residual: ${mn} in ${q}.`,
 iw1: (f, a, b, c, l) => `AR(1), φ = ${f}: half-width ${a}σ at h = 1, ${b}σ at h = 4, ${c}σ at h = 20` + (l === null ? '.' : `; limit 1.96σ/√(1 − φ²) = ${l}σ.`),
 iw2: (a, b, c) => `Random walk: ${a}σ, ${b}σ and ${c}σ (grows like √h).`,
 iw3: (f, a, b, c) => `ARIMA(1,1,0), φ = ${f}: ${a}σ, ${b}σ and ${c}σ.`,
 iw4: (t, a, b, c) => `ARIMA(0,1,1), θ = ${t}: ${a}σ, ${b}σ and ${c}σ.`,
 iw5: (a, b, c) => `ARIMA(0,2,0): ${a}σ, ${b}σ and ${c}σ (grows like h<sup>3/2</sup>).`,
 gf0: (o, v) => `Origin ${o}; last value ${v} bn EUR (2010 prices).`,
 gf: (k, l, lo, hi, a, b, c, rm, n) => (k === 'ds' ? 'ARIMA(0,1,0) with drift' : 'Linear trend with AR(1) errors') + `: ${l} bn EUR after 12 quarters, 95% interval [${lo}, ${hi}]; 95% half-width: ${a}, ${b} and ${c} log points at h = 1, 4 and 12.`
   + (rm === null ? '' : ` Over the ${n} quarters observed after the origin: RMSE ${rm} log points.`),
 od1: (s, r1, v1, d, r2, v2) => (s === 'ro' ? 'Romania' : 'US') + `: d = 1: ρ̂(1) = ${r1}, variance ${v1}; d = ${d}: ρ̂(1) = ${r2}, variance ${v2}.`,
 od2: (d, t, se) => `MA(1) fitted to the difference of order ${d}: θ̂ = ${t} (SE ${se}).`,
 inm: (n, a, c, h12, h24, lo, hi, m) => `${n}: AICc ${a}${c ? '' : ' (the optimisation did not converge)'}; forecast ${h12}% after 12 months and ${h24}% in ${m}, 95% interval [${lo}, ${hi}].`,
 inmu: m => ` Estimated mean: ${m}%.`,
 inb: (a, b) => `Lowest AICc among the models that converged: ${a} for d = 0, ${b} for d = 1. AICc does not compare models with different d: with d = 1 the likelihood is computed for Δπ.`,
 fx0: (n, d, m, p) => `The parameters are estimated on ${n} daily changes, up to ${d}; ${m} one-day-ahead forecasts; φ̂ = ${p}.`,
 fx1: (a, b, c, g) => `One-day RMSE (100 ln): random walk ${a}, with drift ${b}, ARIMA(1,1,0) ${c}; gain of the ARIMA over the random walk: ${g}%.`,
 fx2: (n, a, b, c, g) => `At 20 days (${n} non-overlapping windows): ${a}, ${b} and ${c}; gain of the ARIMA: ${g}%.`,
 us0: (o, a, b) => `Forecasts from ${o}; outcome minus forecast in ${a} and in ${b}:`,
 us: (k, a, b, h, c) => (k === 'ts' ? 'the trend-stationary model' : 'ARIMA(1,1,0) with drift') + `: ${a} and ${b} log points; half-width of the 95% interval at the end ${h}: the outcome ${c ? 'lies inside the interval' : 'lies outside the interval'}.`
};
""",
}



# ================================================================= worked examples and result tables (blocks of their own frames)
# (anchor, frame title RO, frame title EN, section key, host chart block on the web lecture)
EXTRA = [('wx-spur', 'Interpretarea regresiei IAPC', 'Interpreting the HICP regression', 'spur', 'spurious-real'),
         ('cv-table', 'Valorile critice ale testului Dickey–Fuller', 'Critical values of the Dickey–Fuller test', 'df', 'df-dist'),
         ('wx-df', 'Exemplu rezolvat: PIB-ul real al României', 'Worked example: Romanian real GDP', 'df', 'df-dist'),
         ('lag-table', 'Alegerea numărului de laguri k', 'Choosing the number of lags k', 'df', 'adf-power'),
         ('power-table', 'Interpretarea curbelor de putere', 'Interpreting the power curves', 'df', 'adf-power'),
         ('wx-kpss', 'Testul KPSS: staționaritatea ca ipoteză nulă', 'The KPSS test: stationarity as the null', 'kpss', 'kpss-sums'),
         ('adf-kpss', 'ADF și KPSS împreună', 'ADF and KPSS together', 'kpss', 'kpss-sums'),
         ('urt-table', 'Teste de rădăcină unitară pe serii reale', 'Unit-root tests on real series', 'kpss', 'kpss-sums'),
         ('za-sim', 'Interpretarea rupturii simulate', 'Interpreting the simulated break', 'brk', 'breaks-sim'),
         ('za-table', 'Interpretarea celor două rupturi', 'Interpreting the two breaks', 'brk', 'breaks-real'),
         ('aicc-table', 'Alegerea modelului după AICc', 'Model choice by AICc', 'arima', 'gdp-ident'),
         ('diag-table', 'Interpretarea verificării reziduurilor', 'Interpreting the residual checks', 'arima', 'gdp-diag'),
         ('wx-fvar', 'Prognoza cu modele ARIMA', 'Forecasting with ARIMA', 'arima', 'interval-width'),
         ('wx-110', 'Exemplu rezolvat: prognoze ARIMA(1,1,0)', 'Worked example: ARIMA(1,1,0) forecasts', 'arima', 'interval-width'),
         ('wx-width', 'Interpretarea lățimii intervalelor', 'Interpreting the interval widths', 'arima', 'interval-width'),
         ('fc-table', 'Interpretarea prognozelor PIB', 'Interpreting the GDP forecasts', 'arima', 'gdp-forecast')]
PFX.update({'wx-spur': 'xs', 'cv-table': 'cv', 'wx-df': 'xd', 'lag-table': 'lg', 'power-table': 'pw', 'wx-kpss': 'xk', 'adf-kpss': 'ak', 'urt-table': 'ut',
            'za-sim': 'zs', 'za-table': 'zt', 'aicc-table': 'ac', 'diag-table': 'dg', 'wx-fvar': 'fv', 'wx-110': 'x1', 'wx-width': 'xw', 'fc-table': 'ft'})

CSS = ('<style>.c3x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c3x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c3x .c3f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c3x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c3x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c3t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c3t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c3t th,.c3t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c3t thead tr:last-child th{border-bottom:1px solid #1A3A6E}'
       '.c3t tbody tr:nth-child(even){background:#F5F7FB}.c3t tbody tr.hi{background:#FDF0E1}.c3t .r{text-align:right}.c3cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}'
       '#ak-tb td{white-space:normal;min-width:9rem}</style>')


def tbl(i):
    return f'<div class="tbl c3t" id="{i}"></div>'


def wx(i, title, formula):
    return f'<div class="c3x"><h4>{title}</h4><div class="c3f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


SERK = ['gdp', 'dgdp', 'hicp', 'infl', 'dinfl', 'us', 'dus', 'nile']
SERN = {'ro': ['PIB-ul real al României, 100 ln Y', 'creșterea PIB-ului României', 'IAPC România, 100 ln P', 'inflația anuală (12 luni) π',
               'variația inflației Δπ', 'PIB-ul real al SUA, 100 ln Y', 'creșterea PIB-ului SUA', 'debitul Nilului'],
        'en': ['Romanian real GDP, 100 ln Y', 'Romanian GDP growth', 'Romanian HICP, 100 ln P', '12-month inflation π',
               'change of inflation Δπ', 'US real GDP, 100 ln Y', 'US GDP growth', 'Nile flow']}


def widgets_all(lang):
    """Widgets of every block of the interactive page: the chart blocks and EXTRA (worked examples and live tables)."""
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    W = s('Exemplu rezolvat', 'Worked example')
    d = (lambda x: x.replace('0.5', '0{,}5').replace('1.96', '1{,}96').replace('0.6', '0{,}6')) if ro else (lambda x: x)
    w = _chart_widgets(lang)
    ctl, ch = w['four-series']
    w['four-series'] = (ctl, CSS + ch)
    cases = [('n', s('fără constantă', 'no constant')), ('c', s('constantă', 'constant')), ('ct', s('constantă și trend', 'constant and trend'))]
    ics = [('AIC', 'AIC'), ('BIC', 'BIC')]
    lvs = [('0.01', '1%'), ('0.05', '5%'), ('0.1', '10%')]
    det = [('slide', s('ca pe slide', 'as on the slide')), ('c', s('constantă pentru toate', 'constant for all')), ('ct', s('constantă și trend pentru toate', 'constant and trend for all')),
           ('n', s('fără termeni determiniști', 'no deterministic terms'))]
    ser = [(k, n) for k, n in zip(SERK, SERN[lang])]
    end = s('Eșantionul se termină în', 'Sample ends in')
    mods = [(k, f'ARIMA({k[0]},1,{k[1]})' + (s(' · nu a convers', ' · not converged') if k == '22' else '')) for k in PQ]
    w.update({
     'wx-spur': (slider('xs-y0', s('De la anul', 'From year'), 2005, 2022, 1, 2005) + slider('xs-y1', s('Pînă în anul', 'To year'), 2008, 2026, 1, 2026)
                 + select('xs-k', s('Regresia detaliată', 'Regression worked out'), [('lev', s('în niveluri', 'in levels')), ('dif', s('în diferențe', 'in differences'))], 'lev') + play('xs'),
                 wx('xs-ws', W, r'\hat b=\frac{\sum_t(x_t-\bar x)(y_t-\bar y)}{\sum_t(x_t-\bar x)^2},\quad t=\frac{\hat b}{\sqrt{s^2/\sum_t(x_t-\bar x)^2}},\quad R^2=\frac{t^2}{t^2+n-2},\quad \mathrm{DW}=\frac{\sum_t(e_t-e_{t-1})^2}{\sum_t e_t^2}')
                 + tbl('xs-tb')),
     'cv-table': (slider('cv-T', s('Lungimea eșantionului T', 'Sample length T'), 20, 1000, 5, 250) + play('cv'), tbl('cv-tb') + charts('cv-chart')),
     'wx-df': (slider('xd-end', end, 39, 105, 1, 105) + select('xd-reg', s('Termenii determiniști', 'Deterministic terms'), cases, 'ct')
               + select('xd-k', s('Laguri k', 'Lags k'), [('AIC', s('alese după AIC', 'chosen by AIC')), ('BIC', s('alese după BIC', 'chosen by BIC'))] + [(str(k), str(k)) for k in range(14)], '0') + play('xd'),
               wx('xd-ws', W, r'\Delta y_t=c+bt+\gamma y_{t-1}+\sum_{j=1}^{k}\delta_j\Delta y_{t-j}+\varepsilon_t,\qquad \tau=\frac{\hat\gamma}{\mathrm{SE}(\hat\gamma)},\qquad \hat\phi=1+\hat\gamma')
               + tbl('xd-tb') + charts('xd-chart')),
     'lag-table': (select('lg-s', s('Seria', 'Series'), ser, 'gdp') + select('lg-reg', s('Termenii determiniști', 'Deterministic terms'), cases, 'ct')
                   + select('lg-ic', s('Criteriul', 'Criterion'), [('aic', 'AIC'), ('bic', 'BIC')], 'aic') + play('lg'),
                   wx('lg-ws', W, r'k_{\max}=\Big\lceil 12\Big(\frac{T}{100}\Big)^{1/4}\Big\rceil,\qquad \mathrm{AIC}=-2\ln L+2m,\qquad \mathrm{BIC}=-2\ln L+m\ln N')
                   + tbl('lg-tb')),
     'power-table': (select('pw-reg', s('Regresia testului', 'Test regression'), cases[1:], 'c')
                     + select('pw-phi', s('φ adevărat (rîndul evidențiat)', 'True φ (highlighted row)'), [(str(f), f.replace('.', ',') if ro else f) for f in ('0.8', '0.85', '0.9', '0.925', '0.95', '0.975', '0.99', '1')], '0.95')
                     + play('pw'), tbl('pw-tb') + charts('pw-chart')),
     'wx-kpss': (select('xk-s', s('Seria', 'Series'), ser, 'gdp') + select('xk-reg', s('Staționaritate în jurul', 'Stationarity around'), [('c', s('unui nivel (c)', 'a level (c)')), ('ct', s('unui trend (c, t)', 'a trend (c, t)'))], 'ct')
                 + select('xk-L', s('Lățimea de bandă L', 'Bandwidth L'), [('auto', s('automată', 'automatic'))] + [(str(k), str(k)) for k in range(21)], 'auto') + play('xk'),
                 wx('xk-ws', W, r'S_t=\sum_{s\le t}e_s,\qquad \hat\lambda^2=\hat\gamma_0+2\sum_{j=1}^{L}\Big(1-\frac{j}{L+1}\Big)\hat\gamma_j,\qquad \eta=\frac{\sum_{t=1}^{T}S_t^2}{T^2\hat\lambda^2}')
                 + tbl('xk-tb') + charts('xk-chart')),
     'adf-kpss': (select('ak-det', s('Termenii determiniști', 'Deterministic terms'), det[:3], 'slide') + select('ak-ic', s('Lagurile ADF', 'ADF lags'), ics, 'AIC')
                  + select('ak-lv', s('Nivelul de semnificație', 'Significance level'), lvs, '0.05') + play('ak'), tbl('ak-tb')),
     'urt-table': (select('ut-det', s('Termenii determiniști', 'Deterministic terms'), det, 'slide') + select('ut-ic', s('Lagurile ADF și DF-GLS', 'ADF and DF-GLS lags'), ics, 'AIC')
                   + select('ut-lv', s('Nivelul de semnificație', 'Significance level'), lvs, '0.05') + play('ut'), tbl('ut-tb') + charts('ut-chart')),
     'za-sim': (slider('zs-d', s('Saltul mediei la t = 100 (rîndul evidențiat)', 'Jump of the mean at t = 100 (highlighted row)'), 0, 10, 0.5, 5) + play('zs'), tbl('zs-tb')),
     'za-table': (select('zt-reg', s('Ruptura permisă', 'Break allowed'), [('c', s('în nivel', 'in the level')), ('t', s('în pantă', 'in the trend')), ('ct', s('în nivel și pantă', 'in the level and the trend'))], 'c')
                  + play('zt'), tbl('zt-tb') + charts('zt-chart')),
     'aicc-table': (slider('ac-end', end, 0, 42, 1, 42) + select('ac-c', s('Criteriul', 'Criterion'), [('aicc', 'AICc'), ('aic', 'AIC'), ('bic', 'BIC')], 'aicc') + play('ac'), tbl('ac-tb')),
     'diag-table': (select('dg-mod', s('Modelul ARIMA(p,1,q) cu derivă', 'ARIMA(p,1,q) model with drift'), mods, '00')
                    + slider('dg-m', s('Numărul de laguri m (rîndul evidențiat)', 'Number of lags m (highlighted row)'), 1, 16, 1, 8) + play('dg'),
                    wx('dg-ws', W, r'Q^*(m)=T(T+2)\sum_{h=1}^{m}\frac{\hat\rho_{\hat\varepsilon}(h)^2}{T-h}\approx\chi^2(m-p-q),\qquad \mathrm{JB}=\frac{T}{6}\Big(S^2+\frac{(K-3)^2}{4}\Big)\approx\chi^2(2)').replace('<ol id="dg-ws"></ol>', '')
                    + tbl('dg-tb') + tbl('dg-tb2')),
     'wx-fvar': (slider('fv-h', s('Orizontul h (trimestre)', 'Horizon h (quarters)'), 1, 40, 1, 12) + slider('fv-phi', s('φ al modelului staționar în jurul trendului', 'φ of the trend-stationary model'), 0.5, 0.99, 0.0001, 0.92) + play('fv'),
                 wx('fv-ws', W, d(r'\begin{gathered}\text{DS}:\ \Delta y_t=c+\varepsilon_t\ \Rightarrow\ \mathrm{Var}(e_{T+h})=h\sigma^2;\qquad \text{TS}:\ y_t=\alpha+\beta t+u_t,\ u_t=\phi u_{t-1}+\varepsilon_t\\ \Rightarrow\ \mathrm{Var}(e_{T+h})=\sigma^2\frac{1-\phi^{2h}}{1-\phi^2}\ \to\ \frac{\sigma^2}{1-\phi^2},\qquad \text{' + s('semilățimea', 'half-width') + r'}=1.96\sqrt{\mathrm{Var}(e_{T+h})}\end{gathered}'))
                 + tbl('fv-tb') + charts('fv-chart')),
     'wx-110': (slider('x1-phi', 'φ', -0.9, 0.95, 0.05, 0.6) + slider('x1-s', 'σ', 0.5, 5, 0.5, 2) + slider('x1-y', 'y<sub>T</sub>', 90, 130, 1, 108)
                + slider('x1-dy', 'Δy<sub>T</sub>', -10, 10, 0.5, 5) + slider('x1-H', s('Orizontul H', 'Horizon H'), 1, 12, 1, 3) + play('x1'),
                wx('x1-ws', W, d(r'\begin{gathered}\widehat{\Delta y}_{T+h}=\phi\,\widehat{\Delta y}_{T+h-1}=\phi^h\Delta y_T,\qquad \hat y_{T+h}=\hat y_{T+h-1}+\widehat{\Delta y}_{T+h},\\ \psi_j=1+\phi+\dots+\phi^j,\qquad \mathrm{Var}(e_{T+h})=\sigma^2\sum_{j=0}^{h-1}\psi_j^2,\qquad \hat y_{T+h}\pm1.96\sqrt{\mathrm{Var}(e_{T+h})}\end{gathered}'))
                + tbl('x1-tb') + charts('x1-chart')),
     'wx-width': (slider('xw-h', s('Orizontul h', 'Horizon h'), 1, 40, 1, 20) + slider('xw-f0', s('φ al modelului AR(1)', 'φ of the AR(1)'), 0, 0.99, 0.01, 0.8)
                  + slider('xw-f1', s('φ al modelului ARIMA(1,1,0)', 'φ of the ARIMA(1,1,0)'), 0.05, 0.95, 0.05, 0.5) + slider('xw-th', s('θ al modelului ARIMA(0,1,1)', 'θ of the ARIMA(0,1,1)'), -0.95, 0.95, 0.05, -0.5) + play('xw'),
                  wx('xw-ws', W, d(r'\frac{\text{' + s('semilățimea', 'half-width') + r'}}{\sigma}=1.96\sqrt{\sum_{j=0}^{h-1}\psi_j^2},\qquad \psi(z)=\frac{\theta(z)}{\phi(z)(1-z)^d}'))
                  + tbl('xw-tb')),
     'fc-table': (slider('ft-o', s('Originea prognozei', 'Forecast origin'), 0, 42, 1, 42) + select('ft-lv', s('Nivelul intervalului', 'Interval level'), [('0.8', '80%'), ('0.9', '90%'), ('0.95', '95%'), ('0.99', '99%')], '0.95')
                  + play('ft'), tbl('ft-tb')),
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
    """Chart blocks for build_web_lecture.py: each host carries the web version of the EXTRA blocks of its section (moved by ch3.js)."""
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
        img = f'ch3x_{anchor}_{lang}.jpg'
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


# readouts of the EXTRA blocks
TEXT_ADD = {
 'ro': r"""Object.assign(window.CH3_TEXT, { x: {
 case: { n: 'fără constantă', c: 'cu constantă', ct: 'cu constantă și trend' },
 ser: { gdp: 'PIB-ul real al României (100 ln Y)', dgdp: 'creșterea PIB-ului României', hicp: 'IAPC România (100 ln P)', infl: 'inflația anuală π', dinfl: 'variația inflației Δπ',
   us: 'PIB-ul real al SUA (100 ln Y)', dus: 'creșterea PIB-ului SUA', nile: 'debitul Nilului' },
 coef: 'Coeficient', est: 'Estimația', rej: 'respingem', norej: 'nu respingem', df: 'grade de libertate', at5: 'decizia la 5%', mom: 'Statistica', val: 'Valoarea',
 skew: 'Asimetria', kurt: 'Kurtosis', jbp: 'p-value Jarque–Bera', minr: 'Cel mai mic reziduu', maxr: 'Cel mai mare reziduu', var: 'Varianța erorii', half: 'Semilățimea 95%',
 rwHalf: 'Semilățimea 95%, mers aleator', model: 'Modelul', vsRw: 'față de mersul aleator', limit: 'limita (h → ∞)', rw: 'mers aleator, ARIMA(0,1,0)', wex: 'exemplul rezolvat',
 quarter: 'trimestrul', fc: 'prognoza', intv: l => `intervalul ${l}%`, halfp: 'semilățimea (%)', obs: 'observat', ftDS: 'ARIMA(0,1,0) cu derivă (DS)', ftTS: 'trend liniar + AR(1) (TS)',
 levels: 'niveluri', diffs: 'diferențe', slope: 'panta b̂', wgt: 'ponderea 1 − j/(L + 1)', contrib: 'contribuția la λ̂²', fvD: 'DS: 1,96σ√h', fvT: 'TS: AR(1) în jurul trendului', ratio: 'raportul DS/TS',
 verd: { I0: 'I(0)', I1: 'I(1)', inc: 'neconcludent', conflict: 'conflict' },
 kNo: 'KPSS nu respinge', kYes: 'KPSS respinge', aYes: 'ADF respinge', aNo: 'ADF nu respinge', conf: 'conflict', inc: 'neconcludent',
 urtName: ['România, PIB real, 100 ln Y', '  creșterea Δ', 'România, IAPC, 100 ln P', 'România, inflația anuală π', '  variația Δπ', 'EUR/RON, 100 ln S', '  randamente zilnice',
   'BET, 100 ln P', '  randamente zilnice', 'S&amp;P 500, 100 ln P', '  randamente zilnice', 'SUA, PIB real, 100 ln Y', '  creșterea Δ', 'Nil, debitul anual'].map(v => v.replace(/^  /, '&nbsp;&nbsp;&nbsp;')),
 urtShort: ['PIB RO', 'creșterea PIB RO', 'IAPC', 'π', 'Δπ', 'EUR/RON', 'randamente EUR/RON', 'BET', 'randamente BET', 'S&amp;P 500', 'randamente S&amp;P 500', 'PIB SUA', 'creșterea PIB SUA', 'Nil'],
 utH: ['Seria', 'det.', 'T', 'ADF (p)', 'k', 'PP', 'KPSS', 'DF-GLS (k)', 'ADF+KPSS'],
 zsH: ['salt δ', 'DF respinge (%)', 'ADF τ (k)', 'ADF p', 'ZA (k)', 'ruptura ZA, t', 'min. t fără laguri', 'la data'],
 ztH: ['Seria', 'ruptura', 'ZA', '5%', 'p', 'data', 'laguri', 'media înainte', 'media după'],
 zaSer: { nile: 'Nil', eurron: 'EUR/RON (lei)' }, zaReg: { c: 'în nivel', t: 'în pantă', ct: 'în nivel și pantă' },
 cvCap: 'Valori critice MacKinnon (2010) pentru τ, după numărul de observații T al regresiei; rîndul evidențiat este T ales.',
 cv1: (t, n, c, ct) => `T = ${t}: valorile critice de 5% sînt ${n} (fără constantă), ${c} (constantă) și ${ct} (constantă și trend); respingem rădăcina unitară dacă τ este mai mic.`,
 cv2: (n, c, ct) => `Cuantilele de 5% ale celor 20 000 de statistici simulate cu T = 250: ${n}, ${c} și ${ct}; valoarea Normală −1,645 este mult la dreapta lor.`,
 xd1: (a, b, n, k, ic, km) => `Eșantionul ${a} – ${b}: T = ${n} diferențe în regresie, k = ${k} laguri` + (ic ? ` (alese după ${ic} din cel mult ${km})` : ` (cel mult ${km} după regula lui Schwert)`) + '.',
 xd2: (r, g, se, p) => `OLS (${({ n: 'fără constantă', c: 'cu constantă', ct: 'cu constantă și trend' })[r]}): γ̂ = ${g}, SE(γ̂) = ${se}, deci φ̂ = 1 + γ̂ = ${p}.`,
 xd3: (g, se, t) => `τ = γ̂/SE(γ̂) = ${g}/${se} = ${t}.`,
 xd4: (r, n, a, b, c, p) => `Valorile critice MacKinnon pentru cazul ${({ n: 'fără constantă', c: 'cu constantă', ct: 'cu constantă și trend' })[r]} și T = ${n}: ${a} (1%), ${b} (5%), ${c} (10%); p-value = ${p}.`,
 xd5: (t, c, rej, rn) => (rej ? `${t} < ${c}: respingem rădăcina unitară la 5%.` : `${t} > ${c}: nu respingem rădăcina unitară la 5%.`) + (rn && !rej ? ' Cu valoarea Normală −1,645 am fi respins-o: tabelul greșit dă concluzia greșită.' : !rn ? ' Nici cu valoarea Normală −1,645 nu am respinge-o.' : ''),
 xd6: (p, h) => `Dacă φ = ${p} ar fi adevărat, un șoc s-ar înjumătăți în ln 0,5/ln ${p} = ${h} perioade; „nerespins” nu înseamnă „demonstrat”.`,
 xdCap: (n, s) => `Regresia Dickey–Fuller pe ${n} observații; abaterea standard a reziduurilor ${s}. Rîndul evidențiat este γ.`,
 xd: (t, c, p, rej) => `τ = ${t}, valoarea critică de 5% ${c}, p = ${p}: ` + (rej ? 'rădăcina unitară este respinsă.' : 'rădăcina unitară nu este respinsă.'),
 lgCap: (n, k) => `Toate regresiile folosesc aceleași ${n} observații (cele rămase după k<sub>max</sub> = ${k} laguri), ca să fie comparabile; rîndul evidențiat este k ales.`,
 lg1: (n, v, k) => `T = ${n} observații: 12(T/100)<sup>1/4</sup> = ${v}, deci k<sub>max</sub> = ${k}.`,
 lg2: (k, n, ic, b) => `Estimăm regresia ADF pentru k = 0, …, ${k} pe aceleași ${n} observații; ${ic} minim la k = ${b}.`,
 lg3: (k, n, t, p, c) => `Reestimăm cu k = ${k} pe toate cele ${n} observații disponibile: τ = ${t}, p = ${p} (valoarea critică de 5% ${c}).`,
 lg: (s, c, ic, k, t, p) => `${s}, ${c}, ${ic}: k = ${k}, τ = ${t}, p = ${p}.`,
 pwCap: c => `Procentul de respingeri ale rădăcinii unitare la 5% (regresia ${c}), din 2 000 de serii AR(1) pentru fiecare celulă; rîndul φ = 1 este mărimea testului.`,
 pwNote: '* T = 50 și T = 1 000: rulări suplimentare cu același cod și același seed; T = 100, 250 și 500 sînt valorile din curs.',
 pw1: (c, a, b, d) => `Mărimea (φ = 1, regresia ${c}): ${a}% cu T = 100, ${b}% cu T = 250 și ${d}% cu T = 500, toate aproape de 5%.`,
 pw2: (f, a, b, c, d, e) => `Puterea la φ = ${f}: ${a}% (T = 50), ${b}% (T = 100), ${c}% (T = 250), ${d}% (T = 500), ${e}% (T = 1 000).`,
 xk1: (r, n, s) => `${s}: regresia pe ${r === 'c' ? 'o constantă' : 'o constantă și un trend'}, T = ${n} observații; reziduurile e<sub>t</sub> au suma 0.`,
 xk2: (m, d, ss) => `Sumele parțiale S<sub>t</sub> = e<sub>1</sub> + … + e<sub>t</sub>; cea mai mare în valoare absolută: ${m} (${d}); ΣS<sub>t</sub>² = ${ss}.`,
 xk3: (g0, L, a, lr) => `Varianța pe termen lung: γ̂<sub>0</sub> = ${g0}, L = ${L} ${a ? '(lățimea de bandă automată din statsmodels)' : '(aleasă)'}, ponderi Bartlett: λ̂² = ${lr}.`,
 xk4: (ss, n, lr, e) => `η = ΣS<sub>t</sub>²/(T²λ̂²) = ${ss}/(${n}² × ${lr}) = ${e}.`,
 xk5: (e, c, rej, p) => `${e} ${rej ? '>' : '<'} ${c} (valoarea critică de 5%): ${rej ? 'respingem staționaritatea' : 'nu respingem staționaritatea'}; ${p}.`,
 xkCap: L => `Autocovarianțele reziduurilor și ponderile Bartlett pînă la L = ${L}.`,
 xk: (s, r, e, L, c, rej) => `${s}, ${r === 'c' ? 'nivel' : 'trend'}: η = ${e} cu L = ${L}; valoarea critică de 5% ${c}: ${rej ? 'staționaritatea este respinsă' : 'staționaritatea nu este respinsă'}.`,
 utCap: l => `Bold: test care respinge la ${l}. ADF, PP și DF-GLS: H₀ rădăcină unitară; KPSS: H₀ staționaritate. p-value-urile MacKinnon; ADF+KPSS combină cele două decizii la ${l}.`,
 utNote: 'det.: termenii determiniști (c: constantă; t: trend); k: lagurile ADF; PP: Phillips–Perron cu lățimea de bandă 12(T/100)<sup>1/4</sup>; DF-GLS: testul Elliott–Rothenberg–Stock (ERS), cu valorile critice din pachetul arch. Cu lagurile alese după AIC (18–35 de laguri), DF-GLS își pierde puterea și nu respinge rădăcina unitară pentru randamentele BET și S&amp;P 500; cu lagurile după BIC (0 laguri) o respinge clar (τ = −15,4 și −18,5). Randamentele sînt staționare: e un efect al prea multor laguri, nu o rădăcină unitară.',
 ut1: (d, ic, l, a, b, c, e, f) => (f ? `Fără termeni determiniști, KPSS și DF-GLS nu sînt definite, deci nici verdictul combinat. ` : '') + `La ${l} (laguri după ${ic}): I(1) ${a} serii, I(0) ${b}, neconcludent ${c}, conflict ${e}.`,
 ut2: (t, p, k, v) => `PIB-ul României: ADF τ = ${t} (p = ${p}), KPSS ${k}: ` + ({ I1: 'I(1).', I0: 'I(0).', inc: 'neconcludent: niciun test nu respinge.', conflict: 'conflict.' }[v] || '—'),
 akCap: l => `Cele 14 serii ale tabelului, plasate după deciziile ADF și KPSS la ${l}.`,
 ak: (l, a, b, c, d) => `La ${l}: I(0) ${a} serii, I(1) ${b}, neconcludent ${c}, conflict ${d}.`,
 zsCap: (a, z) => `Seria simulată din curs (AR(1), φ = 0,6) cu saltul δ la t = 100; valoarea critică de 5%: ${a} (ADF), ${z} (ZA). ZA din statsmodels alege lagurile după AIC; curba din graficul anterior nu folosește laguri.`,
 zs1: (d, r, p) => `Saltul ${d}: pe 2 000 de serii testul DF respinge în ${r} din cazuri; pe seria din grafic ADF are p = ${p}.`,
 zs2: (z, k, b, m, a, c) => `Statistica Zivot–Andrews este ${z} (laguri: ${k}, ruptura de la t = ${b}); minimul curbei fără laguri este ${m}, la t = ${a}. Ambele sînt sub ${c}: sînt două statistici diferite, nu două citiri ale aceleiași curbe.`,
 ztCap: (a, b, c, d) => `Bold: ZA sub valoarea critică de 5%. Nil: ADF p = ${a}, KPSS ${b}; EUR/RON (sfîrșit de lună, 100 ln): ADF p = ${c}, KPSS ${d}. Mediile EUR/RON sînt în lei.`,
 zt1: (r, z, c, y, a, b) => `Ruptura ${({ c: 'în nivel', t: 'în pantă', ct: 'în nivel și pantă' })[r]}. Nil: ZA = ${z} (5%: ${c}), ruptura în ${y}, media de la ${a} la ${b}.`,
 zt2: (z, c, d, a, b) => `EUR/RON: ZA = ${z} (5%: ${c}), ruptura în ${d}; media de la ${a} la ${b} lei.`,
 acCap: (c, a, b, n) => `${c} pentru ARIMA(p,1,q) cu derivă, 100 ln PIB, ${a} – ${b} (${n} diferențe); bold: cel mai mic dintre modelele care au convers.`,
 acNote: '† optimizatorul nu a convers.',
 ac1: (p, q, c, g, p2, q2) => `Cel mai mic ${c}: ARIMA(${p},1,${q}); următorul model, ARIMA(${p2},1,${q2}), este mai slab cu ${g} puncte.`,
 ac2: (p, q, c, se, t, s, n) => `ARIMA(${p},1,${q}): deriva ĉ = ${c} (SE ${se}, t = ${t}), σ̂ = ${s}.`,
 dgCap: (p, q, n) => `Reziduurile modelului ARIMA(${p},1,${q}) cu derivă (${n} trimestre); gradele de libertate sînt m − p − q; rîndul evidențiat este m ales.`,
 dg1: (m, Q, df, p) => `Q*(${m}) = ${Q} cu ${df} grade de libertate: p = ${p}.`,
 dg0: (m, k) => `Pentru m = ${m} ≤ p + q = ${k} testul nu are grade de libertate.`,
 dg2: (k, j) => `Kurtosis ${k} și Jarque–Bera ${j}: reziduurile nu provin din distribuția Normală; cozile vin din 2009 și 2020.`,
 fv1: (s2, h, v, w) => `DS: σ̂² = ${s2}; Var(e<sub>T+${h}</sub>) = ${h} × ${s2} = ${v}; semilățimea 95% = 1,96√${v} = ${w}% din PIB.`,
 fv2: (s2, p, h, v, w) => `TS: σ̂² = ${s2}, φ̂ = ${p}; Var(e<sub>T+${h}</sub>) = σ̂²(1 − φ̂<sup>${2 * h}</sup>)/(1 − φ̂²) = ${v}; semilățimea = ${w}% din PIB.`,
 fv3: l => `Pentru h → ∞, semilățimea TS se oprește la 1,96σ̂/√(1 − φ̂²) = ${l}; cea DS crește ca √h fără limită.`,
 fv4: (h, r) => `La h = ${h}, intervalul DS este de ${r} ori mai lat decît cel TS.`,
 fvCap: 'Semilățimea intervalului de 95% (în % din PIB, adică puncte de 100 ln) pentru cele două modele estimate pînă în T2 2026.',
 fv: (h, a, b) => `h = ${h}: semilățimea ${a}% pentru modelul DS și ${b}% pentru modelul TS.`,
 x11: (p, d, d1, y, y1) => `Pasul 1, prognoza variației: Δŷ<sub>T+1</sub> = φΔy<sub>T</sub> = ${p} · ${d} = ${d1}; apoi ŷ<sub>T+1</sub> = ${y} + ${d1} = ${y1}.`,
 x12: (p, d1, d2, y1, y2, d3, y3) => `Pasul 2: Δŷ<sub>T+2</sub> = ${p} · ${d1} = ${d2}, ŷ<sub>T+2</sub> = ${y1} + ${d2} = ${y2}; pasul 3: Δŷ<sub>T+3</sub> = ${d3}, ŷ<sub>T+3</sub> = ${y3}.`,
 x13: (p, a, b) => `Ponderile ψ: ψ<sub>0</sub> = 1, ψ<sub>1</sub> = 1 + φ = ${a}, ψ<sub>2</sub> = 1 + φ + φ² = ${b}.`,
 x14: (s2, a, b, c) => `Varianțele erorilor: ${a}, σ²(1 + ψ<sub>1</sub>²) = ${b}, σ²(1 + ψ<sub>1</sub>² + ψ<sub>2</sub>²) = ${c} (σ² = ${s2}).`,
 x15: (a, b, c, r) => `Semilățimile 95%: ${a}, ${b}, ${c}; în trei pași intervalul crește de ${r} ori (mersul aleator: √3 = 1,73).`,
 x1Cap: 'Prognozele modelului Δy<sub>t</sub> = φΔy<sub>t−1</sub> + ε<sub>t</sub> pentru nivel și variație; ultima coloană: mersul aleator cu același σ.',
 x1: (a, b, c, d, e, f) => `ŷ<sub>T+1</sub> = ${a}, ŷ<sub>T+2</sub> = ${b}, ŷ<sub>T+3</sub> = ${c}; semilățimi 95%: ${d}, ${e}, ${f}.`,
 xwCap: 'Semilățimea intervalului de 95% în unități de σ; coloana „față de mersul aleator” este raportul la orizontul h.',
 xw1: (f, l, h, w) => `AR(1), φ = ${f}: ψ<sub>j</sub> = φ<sup>j</sup>; la h = ${h} semilățimea este ${w}σ` + (l ? `, iar limita 1,96/√(1 − φ²) = ${l}σ.` : '.'),
 xw2: (h, r, w) => `Mersul aleator: ψ<sub>j</sub> = 1, deci 1,96√h σ = 1,96 · ${r} = ${w}σ la h = ${h}.`,
 xw3: (f, p, h, w) => `ARIMA(1,1,0), φ = ${f}: ψ<sub>j</sub> = (1 − φ<sup>j+1</sup>)/(1 − φ) → ${p}; la h = ${h}: ${w}σ.`,
 xw4: (h, a, b) => `Graficul din curs folosește φ = 0,5 (${a}σ la h = ${h}); exemplul rezolvat folosește φ = 0,6, care dă ${b}σ la același orizont.`,
 xw5: (t, p, h, w) => `ARIMA(0,1,1), θ = ${t}: ψ<sub>j</sub> = 1 + θ = ${p} pentru j ≥ 1; la h = ${h}: ${w}σ.`,
 xw6: (h, w) => `ARIMA(0,2,0): ψ<sub>j</sub> = j + 1, semilățimea crește ca h<sup>3/2</sup>: ${w}σ la h = ${h}.`,
 xw: (h, a, b, c, d, e) => `La h = ${h}: ${a}σ (AR(1)), ${b}σ (mers aleator), ${c}σ (ARIMA(1,1,0)), ${d}σ (ARIMA(0,1,1)), ${e}σ (ARIMA(0,2,0)).`,
 ftCap: (o, v) => `Prognoze din ${o}; PIB-ul real în mld. EUR (prețuri 2010), ultima valoare ${v}; semilățimea în % din PIB (puncte de 100 ln). Rîndurile evidențiate: h = 1, 4 și 12.`,
 ft1: (l, v, a, b, c, r) => `DS: ${v} mld. EUR după 12 trimestre; semilățimea ${l}%: ${a} (h = 1), ${b} (h = 4), ${c} (h = 12), adică de ${r} ori cea de la h = 1 (√12 = 3,46).`,
 ft2: (v, c) => `TS: ${v} mld. EUR după 12 trimestre, cu semilățimea ${c} la h = 12.`,
 xsCap: 'Regresia 100 ln IAPC România pe 100 ln S&amp;P 500 (sfîrșit de lună), în niveluri și în diferențe lunare; rîndurile evidențiate: t și R².',
 xs1: (l, n, mx, my) => `${l ? 'Niveluri' : 'Diferențe'}: n = ${n} luni, x̄ = ${mx}, ȳ = ${my}.`,
 xs2: (a, b, c) => `b̂ = Σ(x − x̄)(y − ȳ)/Σ(x − x̄)² = ${a}/${b} = ${c}.`,
 xs3: (s2, se, t) => `s² = ${s2}, SE(b̂) = √(s²/Σ(x − x̄)²) = ${se}, t = ${t}.`,
 xs4: (t, df, r) => `R² = t²/(t² + n − 2) = ${t}²/(${t}² + ${df}) = ${r}.`,
 xs5: (d, g) => `DW = ${d}` + (g ? ': R² > DW, semnul clasic al unei regresii false.' : '.'),
 xs: (a, b, c, d, e) => `Niveluri: t = ${a}, R² = ${b}, DW = ${c}; diferențe: t = ${d}, R² = ${e}.`
} });
""",
 'en': r"""Object.assign(window.CH3_TEXT, { x: {
 case: { n: 'no constant', c: 'with a constant', ct: 'with constant and trend' },
 ser: { gdp: 'Romanian real GDP (100 ln Y)', dgdp: 'Romanian GDP growth', hicp: 'Romanian HICP (100 ln P)', infl: '12-month inflation π', dinfl: 'change of inflation Δπ',
   us: 'US real GDP (100 ln Y)', dus: 'US GDP growth', nile: 'Nile flow' },
 coef: 'Coefficient', est: 'Estimate', rej: 'reject', norej: 'do not reject', df: 'degrees of freedom', at5: 'decision at 5%', mom: 'Statistic', val: 'Value',
 skew: 'Skewness', kurt: 'Kurtosis', jbp: 'Jarque–Bera p-value', minr: 'Smallest residual', maxr: 'Largest residual', var: 'Error variance', half: '95% half-width',
 rwHalf: '95% half-width, random walk', model: 'Model', vsRw: 'relative to the random walk', limit: 'limit (h → ∞)', rw: 'random walk, ARIMA(0,1,0)', wex: 'worked example',
 quarter: 'quarter', fc: 'forecast', intv: l => `${l}% interval`, halfp: 'half-width (%)', obs: 'observed', ftDS: 'ARIMA(0,1,0) with drift (DS)', ftTS: 'linear trend + AR(1) (TS)',
 levels: 'levels', diffs: 'differences', slope: 'slope b̂', wgt: 'weight 1 − j/(L + 1)', contrib: 'contribution to λ̂²', fvD: 'DS: 1.96σ√h', fvT: 'TS: AR(1) around a trend', ratio: 'ratio DS/TS',
 verd: { I0: 'I(0)', I1: 'I(1)', inc: 'inconclusive', conflict: 'conflict' },
 kNo: 'KPSS does not reject', kYes: 'KPSS rejects', aYes: 'ADF rejects', aNo: 'ADF does not reject', conf: 'conflict', inc: 'inconclusive',
 urtName: ['Romania, real GDP, 100 ln Y', '  its growth Δ', 'Romania, HICP, 100 ln P', 'Romania, 12-month inflation π', '  its change Δπ', 'EUR/RON, 100 ln S', '  daily returns',
   'BET, 100 ln P', '  daily returns', 'S&amp;P 500, 100 ln P', '  daily returns', 'US, real GDP, 100 ln Y', '  its growth Δ', 'Nile, yearly flow'].map(v => v.replace(/^  /, '&nbsp;&nbsp;&nbsp;')),
 urtShort: ['RO GDP', 'RO GDP growth', 'HICP', 'π', 'Δπ', 'EUR/RON', 'EUR/RON returns', 'BET', 'BET returns', 'S&amp;P 500', 'S&amp;P 500 returns', 'US GDP', 'US GDP growth', 'Nile'],
 utH: ['Series', 'det.', 'T', 'ADF (p)', 'k', 'PP', 'KPSS', 'DF-GLS (k)', 'ADF+KPSS'],
 zsH: ['jump δ', 'DF rejects (%)', 'ADF τ (k)', 'ADF p', 'ZA (k)', 'ZA break, t', 'min. t without lags', 'at date'],
 ztH: ['Series', 'break', 'ZA', '5%', 'p', 'date', 'lags', 'mean before', 'mean after'],
 zaSer: { nile: 'Nile', eurron: 'EUR/RON (lei)' }, zaReg: { c: 'in the level', t: 'in the trend', ct: 'in level and trend' },
 cvCap: 'MacKinnon (2010) critical values of τ, by the number of observations T of the regression; the highlighted row is the chosen T.',
 cv1: (t, n, c, ct) => `T = ${t}: the 5% critical values are ${n} (no constant), ${c} (constant) and ${ct} (constant and trend); we reject the unit root if τ is lower.`,
 cv2: (n, c, ct) => `The 5% quantiles of the 20,000 statistics simulated with T = 250: ${n}, ${c} and ${ct}; the Normal value −1.645 lies far to their right.`,
 xd1: (a, b, n, k, ic, km) => `Sample ${a} – ${b}: T = ${n} differences in the regression, k = ${k} lags` + (ic ? ` (chosen by ${ic} out of at most ${km})` : ` (at most ${km} by Schwert's rule)`) + '.',
 xd2: (r, g, se, p) => `OLS (${({ n: 'no constant', c: 'with a constant', ct: 'with constant and trend' })[r]}): γ̂ = ${g}, SE(γ̂) = ${se}, so φ̂ = 1 + γ̂ = ${p}.`,
 xd3: (g, se, t) => `τ = γ̂/SE(γ̂) = ${g}/${se} = ${t}.`,
 xd4: (r, n, a, b, c, p) => `MacKinnon critical values for the case ${({ n: 'without a constant', c: 'with a constant', ct: 'with constant and trend' })[r]} and T = ${n}: ${a} (1%), ${b} (5%), ${c} (10%); p-value = ${p}.`,
 xd5: (t, c, rej, rn) => (rej ? `${t} < ${c}: reject the unit root at 5%.` : `${t} > ${c}: do not reject the unit root at 5%.`) + (rn && !rej ? ' With the Normal value −1.645 we would have rejected it: the wrong table gives the wrong answer.' : !rn ? ' Not even the Normal value −1.645 would reject it.' : ''),
 xd6: (p, h) => `If φ = ${p} were true, a shock would halve in ln 0.5/ln ${p} = ${h} periods; "not rejected" is not "proved".`,
 xdCap: (n, s) => `Dickey–Fuller regression on ${n} observations; standard deviation of the residuals ${s}. The highlighted row is γ.`,
 xd: (t, c, p, rej) => `τ = ${t}, 5% critical value ${c}, p = ${p}: ` + (rej ? 'the unit root is rejected.' : 'the unit root is not rejected.'),
 lgCap: (n, k) => `Every regression uses the same ${n} observations (those left after k<sub>max</sub> = ${k} lags), so that they are comparable; the highlighted row is the chosen k.`,
 lg1: (n, v, k) => `T = ${n} observations: 12(T/100)<sup>1/4</sup> = ${v}, so k<sub>max</sub> = ${k}.`,
 lg2: (k, n, ic, b) => `Estimate the ADF regression for k = 0, …, ${k} on the same ${n} observations; ${ic} is lowest at k = ${b}.`,
 lg3: (k, n, t, p, c) => `Re-estimate with k = ${k} on all ${n} available observations: τ = ${t}, p = ${p} (5% critical value ${c}).`,
 lg: (s, c, ic, k, t, p) => `${s}, ${c}, ${ic}: k = ${k}, τ = ${t}, p = ${p}.`,
 pwCap: c => `Percentage of rejections of the unit root at 5% (regression ${c}), from 2,000 AR(1) series in each cell; the row φ = 1 is the size of the test.`,
 pwNote: '* T = 50 and T = 1,000: extra runs with the same code and the same seed; T = 100, 250 and 500 are the values of the lecture.',
 pw1: (c, a, b, d) => `Size (φ = 1, regression ${c}): ${a}% with T = 100, ${b}% with T = 250 and ${d}% with T = 500, all close to 5%.`,
 pw2: (f, a, b, c, d, e) => `Power at φ = ${f}: ${a}% (T = 50), ${b}% (T = 100), ${c}% (T = 250), ${d}% (T = 500), ${e}% (T = 1,000).`,
 xk1: (r, n, s) => `${s}: regression on ${r === 'c' ? 'a constant' : 'a constant and a trend'}, T = ${n} observations; the residuals e<sub>t</sub> sum to 0.`,
 xk2: (m, d, ss) => `Partial sums S<sub>t</sub> = e<sub>1</sub> + … + e<sub>t</sub>; the largest in absolute value: ${m} (${d}); ΣS<sub>t</sub>² = ${ss}.`,
 xk3: (g0, L, a, lr) => `Long-run variance: γ̂<sub>0</sub> = ${g0}, L = ${L} ${a ? '(the automatic bandwidth of statsmodels)' : '(chosen)'}, Bartlett weights: λ̂² = ${lr}.`,
 xk4: (ss, n, lr, e) => `η = ΣS<sub>t</sub>²/(T²λ̂²) = ${ss}/(${n}² × ${lr}) = ${e}.`,
 xk5: (e, c, rej, p) => `${e} ${rej ? '>' : '<'} ${c} (5% critical value): ${rej ? 'reject stationarity' : 'do not reject stationarity'}; ${p}.`,
 xkCap: L => `Autocovariances of the residuals and Bartlett weights up to L = ${L}.`,
 xk: (s, r, e, L, c, rej) => `${s}, ${r === 'c' ? 'level' : 'trend'}: η = ${e} with L = ${L}; 5% critical value ${c}: ${rej ? 'stationarity is rejected' : 'stationarity is not rejected'}.`,
 utCap: l => `Bold: a test that rejects at ${l}. ADF, PP and DF-GLS: H₀ unit root; KPSS: H₀ stationarity. MacKinnon p-values; ADF+KPSS combines the two decisions at ${l}.`,
 utNote: 'det.: deterministic terms (c: constant; t: trend); k: ADF lags; PP: Phillips–Perron with bandwidth 12(T/100)<sup>1/4</sup>; DF-GLS: the Elliott–Rothenberg–Stock (ERS) test, with the critical values of the arch package. With lags chosen by AIC (18–35 lags), DF-GLS loses power and does not reject the unit root for BET and S&amp;P 500 returns; with lags chosen by BIC (0 lags) it rejects clearly (τ = −15.4 and −18.5). The returns are stationary: this is an effect of too many lags, not a unit root.',
 ut1: (d, ic, l, a, b, c, e, f) => (f ? 'Without deterministic terms, KPSS and DF-GLS are not defined, and neither is the joint verdict. ' : '') + `At ${l} (lags by ${ic}): I(1) ${a} series, I(0) ${b}, inconclusive ${c}, conflict ${e}.`,
 ut2: (t, p, k, v) => `Romanian GDP: ADF τ = ${t} (p = ${p}), KPSS ${k}: ` + ({ I1: 'I(1).', I0: 'I(0).', inc: 'inconclusive: neither test rejects.', conflict: 'conflict.' }[v] || '—'),
 akCap: l => `The 14 series of the table, placed by the ADF and KPSS decisions at ${l}.`,
 ak: (l, a, b, c, d) => `At ${l}: I(0) ${a} series, I(1) ${b}, inconclusive ${c}, conflict ${d}.`,
 zsCap: (a, z) => `The simulated series of the lecture (AR(1), φ = 0.6) with the jump δ at t = 100; 5% critical value: ${a} (ADF), ${z} (ZA). The statsmodels ZA test chooses the lags by AIC; the curve of the previous chart uses no lags.`,
 zs1: (d, r, p) => `Jump ${d}: on 2,000 series the DF test rejects in ${r} of the cases; on the series of the chart ADF has p = ${p}.`,
 zs2: (z, k, b, m, a, c) => `The Zivot–Andrews statistic is ${z} (lags: ${k}, break from t = ${b}); the minimum of the curve without lags is ${m}, at t = ${a}. Both are below ${c}: they are two different statistics, not two readings of the same curve.`,
 ztCap: (a, b, c, d) => `Bold: ZA below the 5% critical value. Nile: ADF p = ${a}, KPSS ${b}; EUR/RON (month-end, 100 ln): ADF p = ${c}, KPSS ${d}. EUR/RON means in lei.`,
 zt1: (r, z, c, y, a, b) => `Break ${({ c: 'in the level', t: 'in the trend', ct: 'in the level and the trend' })[r]}. Nile: ZA = ${z} (5%: ${c}), break in ${y}, mean from ${a} to ${b}.`,
 zt2: (z, c, d, a, b) => `EUR/RON: ZA = ${z} (5%: ${c}), break in ${d}; mean from ${a} to ${b} lei.`,
 acCap: (c, a, b, n) => `${c} of ARIMA(p,1,q) with drift, 100 ln GDP, ${a} – ${b} (${n} differences); bold: the lowest among the models that converged.`,
 acNote: '† the optimiser did not converge.',
 ac1: (p, q, c, g, p2, q2) => `Lowest ${c}: ARIMA(${p},1,${q}); the next model, ARIMA(${p2},1,${q2}), is ${g} points worse.`,
 ac2: (p, q, c, se, t, s, n) => `ARIMA(${p},1,${q}): drift ĉ = ${c} (SE ${se}, t = ${t}), σ̂ = ${s}.`,
 dgCap: (p, q, n) => `Residuals of ARIMA(${p},1,${q}) with drift (${n} quarters); the degrees of freedom are m − p − q; the highlighted row is the chosen m.`,
 dg1: (m, Q, df, p) => `Q*(${m}) = ${Q} with ${df} degrees of freedom: p = ${p}.`,
 dg0: (m, k) => `For m = ${m} ≤ p + q = ${k} the test has no degrees of freedom.`,
 dg2: (k, j) => `Kurtosis ${k} and Jarque–Bera ${j}: the residuals do not come from the Normal distribution; the tails come from 2009 and 2020.`,
 fv1: (s2, h, v, w) => `DS: σ̂² = ${s2}; Var(e<sub>T+${h}</sub>) = ${h} × ${s2} = ${v}; 95% half-width = 1.96√${v} = ${w}% of GDP.`,
 fv2: (s2, p, h, v, w) => `TS: σ̂² = ${s2}, φ̂ = ${p}; Var(e<sub>T+${h}</sub>) = σ̂²(1 − φ̂<sup>${2 * h}</sup>)/(1 − φ̂²) = ${v}; half-width = ${w}% of GDP.`,
 fv3: l => `As h → ∞, the TS half-width stops at 1.96σ̂/√(1 − φ̂²) = ${l}; the DS one grows like √h without a limit.`,
 fv4: (h, r) => `At h = ${h}, the DS interval is ${r} times as wide as the TS one.`,
 fvCap: 'Half-width of the 95% interval (in % of GDP, i.e. points of 100 ln) for the two models estimated up to 2026Q2.',
 fv: (h, a, b) => `h = ${h}: half-width ${a}% for the DS model and ${b}% for the TS model.`,
 x11: (p, d, d1, y, y1) => `Step 1, forecast the change: Δŷ<sub>T+1</sub> = φΔy<sub>T</sub> = ${p} · ${d} = ${d1}; then ŷ<sub>T+1</sub> = ${y} + ${d1} = ${y1}.`,
 x12: (p, d1, d2, y1, y2, d3, y3) => `Step 2: Δŷ<sub>T+2</sub> = ${p} · ${d1} = ${d2}, ŷ<sub>T+2</sub> = ${y1} + ${d2} = ${y2}; step 3: Δŷ<sub>T+3</sub> = ${d3}, ŷ<sub>T+3</sub> = ${y3}.`,
 x13: (p, a, b) => `ψ weights: ψ<sub>0</sub> = 1, ψ<sub>1</sub> = 1 + φ = ${a}, ψ<sub>2</sub> = 1 + φ + φ² = ${b}.`,
 x14: (s2, a, b, c) => `Error variances: ${a}, σ²(1 + ψ<sub>1</sub>²) = ${b}, σ²(1 + ψ<sub>1</sub>² + ψ<sub>2</sub>²) = ${c} (σ² = ${s2}).`,
 x15: (a, b, c, r) => `95% half-widths: ${a}, ${b}, ${c}; in three steps the interval grows ${r} times (random walk: √3 = 1.73).`,
 x1Cap: 'Forecasts of the model Δy<sub>t</sub> = φΔy<sub>t−1</sub> + ε<sub>t</sub> for the level and the change; last column: the random walk with the same σ.',
 x1: (a, b, c, d, e, f) => `ŷ<sub>T+1</sub> = ${a}, ŷ<sub>T+2</sub> = ${b}, ŷ<sub>T+3</sub> = ${c}; 95% half-widths: ${d}, ${e}, ${f}.`,
 xwCap: 'Half-width of the 95% interval in units of σ; the column "relative to the random walk" is the ratio at horizon h.',
 xw1: (f, l, h, w) => `AR(1), φ = ${f}: ψ<sub>j</sub> = φ<sup>j</sup>; at h = ${h} the half-width is ${w}σ` + (l ? `, and the limit 1.96/√(1 − φ²) = ${l}σ.` : '.'),
 xw2: (h, r, w) => `Random walk: ψ<sub>j</sub> = 1, so 1.96√h σ = 1.96 · ${r} = ${w}σ at h = ${h}.`,
 xw3: (f, p, h, w) => `ARIMA(1,1,0), φ = ${f}: ψ<sub>j</sub> = (1 − φ<sup>j+1</sup>)/(1 − φ) → ${p}; at h = ${h}: ${w}σ.`,
 xw4: (h, a, b) => `The chart of the lecture uses φ = 0.5 (${a}σ at h = ${h}); the worked example uses φ = 0.6, which gives ${b}σ at the same horizon.`,
 xw5: (t, p, h, w) => `ARIMA(0,1,1), θ = ${t}: ψ<sub>j</sub> = 1 + θ = ${p} for j ≥ 1; at h = ${h}: ${w}σ.`,
 xw6: (h, w) => `ARIMA(0,2,0): ψ<sub>j</sub> = j + 1, the half-width grows like h<sup>3/2</sup>: ${w}σ at h = ${h}.`,
 xw: (h, a, b, c, d, e) => `At h = ${h}: ${a}σ (AR(1)), ${b}σ (random walk), ${c}σ (ARIMA(1,1,0)), ${d}σ (ARIMA(0,1,1)), ${e}σ (ARIMA(0,2,0)).`,
 ftCap: (o, v) => `Forecasts from ${o}; real GDP in bn EUR (2010 prices), last value ${v}; half-width in % of GDP (points of 100 ln). Highlighted rows: h = 1, 4 and 12.`,
 ft1: (l, v, a, b, c, r) => `DS: ${v} bn EUR after 12 quarters; ${l}% half-width: ${a} (h = 1), ${b} (h = 4), ${c} (h = 12), i.e. ${r} times the one at h = 1 (√12 = 3.46).`,
 ft2: (v, c) => `TS: ${v} bn EUR after 12 quarters, with half-width ${c} at h = 12.`,
 xsCap: 'Regression of 100 ln Romanian HICP on 100 ln S&amp;P 500 (month-end), in levels and in monthly differences; highlighted rows: t and R².',
 xs1: (l, n, mx, my) => `${l ? 'Levels' : 'Differences'}: n = ${n} months, x̄ = ${mx}, ȳ = ${my}.`,
 xs2: (a, b, c) => `b̂ = Σ(x − x̄)(y − ȳ)/Σ(x − x̄)² = ${a}/${b} = ${c}.`,
 xs3: (s2, se, t) => `s² = ${s2}, SE(b̂) = √(s²/Σ(x − x̄)²) = ${se}, t = ${t}.`,
 xs4: (t, df, r) => `R² = t²/(t² + n − 2) = ${t}²/(${t}² + ${df}) = ${r}.`,
 xs5: (d, g) => `DW = ${d}` + (g ? ': R² > DW, the textbook sign of a spurious regression.' : '.'),
 xs: (a, b, c, d, e) => `Levels: t = ${a}, R² = ${b}, DW = ${c}; differences: t = ${d}, R² = ${e}.`
} });
""",
}
for _lg in TEXT_ADD:
    TEXT_JS[_lg] += TEXT_ADD[_lg]


# Questions of the EXTRA blocks
QX = {
 'ro': {
  'wx-spur': ('În niveluri, regresia IAPC pe S&amp;P 500 are t = 34,6 și R² = 0,82. Este R² o informație nouă față de t?',
              'Nu: cu o singură variabilă explicativă, R² = t²/(t² + n − 2) = 34,6²/(34,6² + 258) = 0,82. Ambele vin din aceeași eroare standard, prea mică pentru că eroarea este ea însăși un mers aleator; '
              'DW = 0,03 o arată. În diferențe lunare, t = −0,21 (p = 0,83) și R² = 0,0002: inflația lunară și randamentele bursiere nu sînt legate.'),
  'cv-table': ('Cu T = 50 și o regresie cu constantă, τ = −2,90. Respingem rădăcina unitară la 5%?',
               'Nu: valoarea critică de 5% pentru T = 50 este −2,92, iar −2,90 > −2,92. Cu valoarea asimptotică −2,86 am fi respins-o: valorile critice depind puțin de T, dar contează cînd τ este aproape de prag. '
               'Cu valoarea Normală −1,645 respingerea ar fi fost „clară” și greșită.'),
  'wx-df': ('Ce se întîmplă cu τ pentru PIB dacă scoatem trendul din regresie (doar constantă, k = 0)?',
            'τ devine −1,80 (γ̂ = −0,0173, SE 0,0096), cu valoarea critică −2,89 și p = 0,38: tot nerespingere. Fără constantă, τ = 3,29 și p ≈ 1. Pentru o serie cu trend, alternativa corectă este cea cu trend: '
            'slide-ul are τ = −2,31 față de −3,45 (p = 0,43). Concluzia nu depinde aici de termenii determiniști, dar valoarea critică se schimbă cu fiecare caz.'),
  'lag-table': ('Pentru IAPC (100 ln P), AIC alege k = 6. Ce alege BIC și se schimbă concluzia?',
                'BIC alege k = 1, pentru că penalizează fiecare lag cu ln N în loc de 2: τ = −0,38 (p = 0,988), față de τ = −0,96 (p = 0,949) cu k = 6. Concluzia rămîne aceeași: rădăcina unitară nu este respinsă. '
                'Lagurile schimbă τ, de aceea k se raportează împreună cu τ.'),
  'power-table': ('Cu T = 100 trimestre (25 de ani) și φ = 0,95, cît de des respinge testul DF (cu constantă) rădăcina unitară?',
                  'În 12,5% din cazuri; cu constantă și trend, în 8,9%. Ca să ajungă la 97,3% este nevoie de T = 500. La φ = 1 testul respinge în 5,3%, 5,2% și 4,65% din cazuri (T = 100, 250, 500): mărimea este corectă, '
                  'puterea pe eșantioane scurte este mică.'),
  'wx-kpss': ('Pentru PIB-ul României, η = 0,198 în jurul unui trend. Cît este η în jurul unui nivel și de ce?',
              'η = 1,533 (L = 6), mult peste 0,463. În jurul unei constante, reziduurile conțin tot trendul: sînt negative la început și pozitive la sfîrșit, iar sumele parțiale fac excursii uriașe. '
              'Pentru o serie cu trend se folosește versiunea cu trend, cu valoarea critică 0,146; acolo 0,198 > 0,146 respinge staționaritatea la 5%, dar nu la 1% (0,216).'),
  'adf-kpss': ('La 1% în loc de 5%, în ce celulă ajunge PIB-ul României?',
               'În „neconcludent”: ADF tot nu respinge (p = 0,43), iar KPSS = 0,198 este sub valoarea critică de 1%, 0,216. La 10%, inflația anuală trece în I(1) (KPSS 0,404 > 0,347), iar randamentele BET, S&amp;P 500 '
               'și creșterea PIB-ului SUA trec în „conflict”: la nivelul de 10%, KPSS respinge mai ușor.'),
  'urt-table': ('Dacă toate seriile sînt testate cu constantă și trend, ce se întîmplă cu inflația anuală?',
                'Devine I(1): KPSS în jurul unui trend este 0,399 > 0,146, iar ADF tot nu respinge (τ = −1,99, p = 0,61). Randamentele BET trec în „conflict” (KPSS 0,250 > 0,146). '
                'Verdictul depinde de alternativa aleasă; pentru seriile fără trend, constanta este alegerea din curs.'),
  'za-sim': ('Slide-ul dă ZA = −7,47 și spune că minimul este atins la t = 100. Este −7,47 minimul curbei din graficul anterior?',
             'Nu. Curba din grafic, fără laguri, are minimul −8,37 la t = 100. Statistica Zivot–Andrews din statsmodels alege lagurile după AIC (aici 1 lag) și dă −7,47, cu ruptura tot la t = 100. '
             'Ambele sînt mult sub −4,81, deci concluzia nu se schimbă: seria este staționară în jurul unei medii cu o ruptură.'),
  'za-table': ('Pentru EUR/RON, respinge Zivot–Andrews și dacă ruptura este permisă doar în pantă?',
               'Abia nu: ZA = −4,40, iar valoarea critică de 5% este −4,41 (p = 0,051), cu ruptura în iulie 2009. Cu ruptura în nivel (−6,89) sau în nivel și pantă (−6,83, valoarea critică −5,07) respinge, '
               'cu ruptura în octombrie 2008. Pentru Nil toate cele trei versiuni resping.'),
  'aicc-table': ('ARIMA(0,1,2) are un AICc mai mare cu 1,7 puncte. Este aceasta o dovadă împotriva lui?',
                 'Nu: diferențele sub 2 puncte nu sînt informative, așa că păstrăm modelul cel mai simplu. Pentru orice sfîrșit de eșantion din T4 2015 pînă în T2 2026, cel mai mic AICc și cel mai mic BIC aparțin '
                 'lui ARIMA(0,1,0); deriva estimată pe tot eșantionul este 0,81 (SE 0,30, t = 2,72).'),
  'diag-table': ('Pentru ARIMA(1,1,1), cu cîte grade de libertate se compară Q*(8)?',
                 'Cu m − p − q = 8 − 2 = 6: Q*(8) = 5,81 și p = 0,44. Pentru ARIMA(0,1,0), Q*(8) = 7,28 cu 8 grade de libertate și p = 0,51. Fără corecție, p-value-urile modelelor cu parametri ARMA ar fi prea mari.'),
  'wx-fvar': ('La h = 40 trimestre (10 ani), de cîte ori este mai lat intervalul DS decît cel TS?',
              'De 2,53 ori: 30,6% față de 12,1% din PIB. Intervalul TS se oprește la 1,96σ̂/√(1 − φ̂²) = 12,09, pe cînd cel DS crește ca √h. La h = 12, semilățimile sînt 16,8% și 11,2%, ca pe slide.'),
  'wx-110': ('Cu φ = 0 (mers aleator fără derivă), cît devin prognozele și semilățimile pentru h = 1, 2, 3?',
             'Prognozele sînt toate 108, iar semilățimile 3,92, 5,54 și 6,79 = 1,96 · 2 · √h. Cu φ = 0,6 prognozele urcă la 111,00, 112,80 și 113,88, iar semilățimile la 3,92, 7,40 și 10,66: '
             'impulsul face ca fiecare șoc să se transmită și în variațiile următoare.'),
  'wx-width': ('Slide-ul de interpretare dă 16,78σ la h = 20 pentru ARIMA(1,1,0); exemplul rezolvat folosește φ = 0,6. Ce φ a produs 16,78σ?',
               'φ = 0,5, valoarea din grafic. Cu φ = 0,6, semilățimea la h = 20 este 20,53σ. La h = 1 ambele dau 1,96σ, iar la h = 4: 6,14σ (φ = 0,5) și 6,83σ (φ = 0,6). Mersul aleator are 8,77σ la h = 20.'),
  'fc-table': ('Slide-ul spune că testele favorizează DS la 5%, dar sînt neconcludente la 1%, iar tabelul testelor clasifică PIB-ul ca I(1). Cum se împacă?',
               'La 5%, ADF nu respinge (p = 0,43) și KPSS respinge (0,198 > 0,146): I(1). Dovada este însă slabă: la 1% KPSS nu mai respinge (0,198 < 0,216), iar cu T ≈ 100 puterea ADF la φ = 0,9 este 19%. '
               'Alegerea lui d contează: după 12 trimestre, 53,8 mld. EUR cu semilățimea 16,8% (DS) față de 56,3 mld. EUR cu 11,2% (TS).'),
 },
 'en': {
  'wx-spur': ('In levels, the regression of the HICP on the S&amp;P 500 has t = 34.6 and R² = 0.82. Is R² new information beyond t?',
              'No: with one regressor, R² = t²/(t² + n − 2) = 34.6²/(34.6² + 258) = 0.82. Both come from the same standard error, which is far too small because the error is itself a random walk; '
              'DW = 0.03 shows it. In monthly differences, t = −0.21 (p = 0.83) and R² = 0.0002: monthly inflation and stock returns are unrelated.'),
  'cv-table': ('With T = 50 and a regression with a constant, τ = −2.90. Do we reject the unit root at 5%?',
               'No: the 5% critical value for T = 50 is −2.92, and −2.90 > −2.92. With the asymptotic value −2.86 we would have rejected it: the critical values depend only weakly on T, but it matters when τ is close to the threshold. '
               'With the Normal value −1.645 the rejection would have been "clear" and wrong.'),
  'wx-df': ('What happens to τ for GDP if we drop the trend from the regression (constant only, k = 0)?',
            'τ becomes −1.80 (γ̂ = −0.0173, SE 0.0096), with critical value −2.89 and p = 0.38: still no rejection. Without a constant, τ = 3.29 and p ≈ 1. For a trending series the right alternative is the one with a trend: '
            'the slide has τ = −2.31 against −3.45 (p = 0.43). Here the conclusion does not depend on the deterministic terms, but the critical value changes with each case.'),
  'lag-table': ('For the HICP (100 ln P), AIC chooses k = 6. What does BIC choose, and does the conclusion change?',
                'BIC chooses k = 1, because it penalises each lag by ln N instead of 2: τ = −0.38 (p = 0.988), against τ = −0.96 (p = 0.949) with k = 6. The conclusion is the same: the unit root is not rejected. '
                'The lags change τ, which is why k is reported together with τ.'),
  'power-table': ('With T = 100 quarters (25 years) and φ = 0.95, how often does the DF test (constant) reject the unit root?',
                  'In 12.5% of the cases; with constant and trend, in 8.9%. To reach 97.3% it needs T = 500. At φ = 1 the test rejects in 5.3%, 5.2% and 4.65% of the cases (T = 100, 250, 500): the size is right, '
                  'the power in short samples is low.'),
  'wx-kpss': ('For Romanian GDP, η = 0.198 around a trend. What is η around a level, and why?',
              'η = 1.533 (L = 6), far above 0.463. Around a constant, the residuals contain the whole trend: negative at the start and positive at the end, and the partial sums make huge excursions. '
              'A trending series needs the trend version, with critical value 0.146; there 0.198 > 0.146 rejects stationarity at 5%, but not at 1% (0.216).'),
  'adf-kpss': ('At 1% instead of 5%, in which cell does Romanian GDP land?',
               'In "inconclusive": ADF still does not reject (p = 0.43), and KPSS = 0.198 is below the 1% critical value, 0.216. At 10%, 12-month inflation moves to I(1) (KPSS 0.404 > 0.347), and the BET and S&amp;P 500 returns '
               'and US GDP growth move to "conflict": at the 10% level KPSS rejects more easily.'),
  'urt-table': ('If every series is tested with constant and trend, what happens to 12-month inflation?',
                'It becomes I(1): KPSS around a trend is 0.399 > 0.146, while ADF still does not reject (τ = −1.99, p = 0.61). The BET returns move to "conflict" (KPSS 0.250 > 0.146). '
                'The verdict depends on the alternative chosen; for series without a trend, the constant is the choice of the lecture.'),
  'za-sim': ('The slide gives ZA = −7.47 and says that the minimum is reached at t = 100. Is −7.47 the minimum of the curve in the previous chart?',
             'No. The curve of the chart, without lags, has its minimum −8.37 at t = 100. The statsmodels Zivot–Andrews statistic chooses the lags by AIC (here 1 lag) and gives −7.47, with the break also at t = 100. '
             'Both are far below −4.81, so the conclusion does not change: the series is stationary around a mean with one break.'),
  'za-table': ('For EUR/RON, does Zivot–Andrews still reject if the break is allowed only in the trend?',
               'Just not: ZA = −4.40 and the 5% critical value is −4.41 (p = 0.051), with the break in July 2009. With the break in the level (−6.89) or in both (−6.83, critical value −5.07) it rejects, '
               'with the break in October 2008. For the Nile all three versions reject.'),
  'aicc-table': ('ARIMA(0,1,2) has an AICc 1.7 points higher. Is that evidence against it?',
                 'No: differences below 2 points are not informative, so we keep the simplest model. For every sample end from 2015Q4 to 2026Q2, the lowest AICc and the lowest BIC belong to '
                 'ARIMA(0,1,0); the drift estimated on the full sample is 0.81 (SE 0.30, t = 2.72).'),
  'diag-table': ('For ARIMA(1,1,1), with how many degrees of freedom is Q*(8) compared?',
                 'With m − p − q = 8 − 2 = 6: Q*(8) = 5.81 and p = 0.44. For ARIMA(0,1,0), Q*(8) = 7.28 with 8 degrees of freedom and p = 0.51. Without the correction, the p-values of models with ARMA parameters would be too large.'),
  'wx-fvar': ('At h = 40 quarters (10 years), how many times wider is the DS interval than the TS one?',
              '2.53 times: 30.6% against 12.1% of GDP. The TS interval stops at 1.96σ̂/√(1 − φ̂²) = 12.09, while the DS one grows like √h. At h = 12 the half-widths are 16.8% and 11.2%, as on the slide.'),
  'wx-110': ('With φ = 0 (a random walk without drift), what do the forecasts and the half-widths become for h = 1, 2, 3?',
             'The forecasts are all 108, and the half-widths 3.92, 5.54 and 6.79 = 1.96 · 2 · √h. With φ = 0.6 the forecasts rise to 111.00, 112.80 and 113.88, and the half-widths to 3.92, 7.40 and 10.66: '
             'the momentum passes each shock on to the following changes as well.'),
  'wx-width': ('The interpretation slide gives 16.78σ at h = 20 for ARIMA(1,1,0); the worked example uses φ = 0.6. Which φ produced 16.78σ?',
               'φ = 0.5, the value of the chart. With φ = 0.6 the half-width at h = 20 is 20.53σ. At h = 1 both give 1.96σ, and at h = 4: 6.14σ (φ = 0.5) and 6.83σ (φ = 0.6). The random walk has 8.77σ at h = 20.'),
  'fc-table': ('The slide says that the tests favour DS at 5% but are inconclusive at 1%, while the unit-root table classifies GDP as I(1). How do the two fit?',
               'At 5%, ADF does not reject (p = 0.43) and KPSS rejects (0.198 > 0.146): I(1). The evidence is weak, though: at 1% KPSS no longer rejects (0.198 < 0.216), and with T ≈ 100 the power of ADF at φ = 0.9 is 19%. '
               'The choice of d matters: after 12 quarters, 53.8 bn EUR with half-width 16.8% (DS) against 56.3 bn EUR with 11.2% (TS).'),
 },
}
for _lg in QA:
    QA[_lg].update(QX[_lg])


# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with the readout of the block at its initial (slide) state, or at the state the question names.
# Each wrong option encodes one identifiable misconception.
PREDICT = {
 'four-series': (1, [('Aproape pe dreaptă: după recesiune, PIB-ul revine la trendul de dinainte.', 'Close to the line: after a recession, GDP returns to its earlier trend.'),
                     ('Cu 76 de puncte logaritmice sub dreaptă.', '76 log points below the line.'),
                     ('Deasupra dreptei: o recesiune este urmată de o creștere care o compensează.', 'Above the line: a recession is followed by growth that makes up for it.')]),
 'ts-ds': (1, [('Întregul șoc, pentru orice φ: cele două serii au același trend.', 'The whole shock, for every φ: the two series have the same trend.'),
               ('Aproape nimic pentru φ < 1 (2,2 abia la φ = 0,99); la φ = 1, întregul șoc.', 'Almost nothing for φ < 1 (2.2 only at φ = 0.99); at φ = 1, the whole shock.'),
               ('Nimic, pentru orice φ, inclusiv φ = 1: un trend determinist absoarbe orice șoc.', 'Nothing, for every φ including φ = 1: a deterministic trend absorbs any shock.')]),
 'detrend': (2, [('În circa 5% din cazuri, nivelul testului.', 'In about 5% of the cases, the level of the test.'),
                 ('Rar: cînd panta adevărată este 0, panta estimată este aproape de 0.', 'Rarely: when the true slope is 0, the estimated slope is close to 0.'),
                 ('În 90,4% din cazuri.', 'In 90.4% of the cases.')]),
 'spurious-mc': (1, [('Scade spre 5%: cu mai multe date, panta estimată se apropie de 0.', 'It falls towards 5%: with more data, the estimated slope approaches 0.'),
                     ('Crește, la 92,3%.', 'It rises, to 92.3%.'),
                     ('Rămîne 83,6%: rata de respingere nu depinde de T.', 'It stays at 83.6%: the rejection rate does not depend on T.')]),
 'spurious-real': (1, [('Da: o legătură cu t = 34,6 nu poate dispărea pe un subeșantion.', 'Yes: a relation with t = 34.6 cannot vanish on a subsample.'),
                       ('Nu: pe 2005–2012, t = −0,72 și R² sub 0,01.', 'No: on 2005–2012, t = −0.72 and R² below 0.01.'),
                       ('Da, și mai puternică: un eșantion scurt are mai puțin zgomot.', 'Yes, and stronger: a short sample has less noise.')]),
 'df-dist': (1, [('În 5% din eșantioane, ca la orice test unilateral la 5%.', 'In 5% of the samples, as for any one-sided test at 5%.'),
                 ('În 77,4% din eșantioane.', 'In 77.4% of the samples.'),
                 ('În 46,1%, la fel pentru orice specificație a termenilor determiniști.', 'In 46.1%, the same for every specification of the deterministic terms.')]),
 'adf-power': (1, [('În circa 95%: φ = 0,95 este staționar, deci testul îl recunoaște.', 'In about 95%: φ = 0.95 is stationary, so the test recognises it.'),
                   ('În 12,5% din cazuri.', 'In 12.5% of the cases.'),
                   ('În exact 5%: puterea unui test este nivelul lui.', 'In exactly 5%: the power of a test is its level.')]),
 'kpss-sums': (1, [('De la orice φ > 0: orice autocorelație contrazice staționaritatea.', 'From any φ > 0: any autocorrelation contradicts stationarity.'),
                   ('Abia de la φ = 0,95.', 'Only from φ = 0.95.'),
                   ('Doar la φ = 1: KPSS respinge numai o rădăcină unitară exactă.', 'Only at φ = 1: KPSS rejects only an exact unit root.')]),
 'breaks-sim': (1, [('Niciodată: seria este staționară, deci ADF o recunoaște pentru orice salt.', 'Never: the series is stationary, so ADF recognises it for any jump.'),
                    ('De la un salt de 3,5 (p = 0,056).', 'From a jump of 3.5 (p = 0.056).'),
                    ('De la cel mai mic salt, 0,5: orice ruptură face seria nestaționară.', 'From the smallest jump, 0.5: any break makes the series non-stationary.')]),
 'breaks-real': (1, [('Tot o respingere clară: ruptura din 2008 apare în orice variantă.', 'Still a clear rejection: the 2008 break shows up in every variant.'),
                     ('ZA = −4,40, la limita valorii critice −4,41: nu respinge la 5%.', 'ZA = −4.40, at the edge of the critical value −4.41: no rejection at 5%.'),
                     ('Același rezultat: testul alege singur și forma rupturii.', 'The same result: the test also chooses the form of the break by itself.')]),
 'gdp-ident': (1, [('Da: creșterea este zgomot alb pe orice subeșantion.', 'Yes: growth is white noise on every subsample.'),
                   ('Nu întru totul: o valoare ACF iese din bandă, dar Ljung–Box Q*(8) = 11,1 (p = 0,20) nu respinge.', 'Not entirely: one ACF value leaves the band, but the Ljung–Box Q*(8) = 11.1 (p = 0.20) does not reject.'),
                   ('Nu: Ljung–Box respinge clar zgomotul alb pe 2000–2008.', 'No: the Ljung–Box test clearly rejects white noise on 2000–2008.')]),
 'gdp-diag': (1, [('Da: cel mai mic Q*(8) indică cel mai bun model.', 'Yes: the smallest Q*(8) points to the best model.'),
                  ('Nu: estimarea nu a convers, iar AICc este mai mare decît pentru ARIMA(0,1,0).', 'No: the estimation did not converge, and its AICc is higher than for ARIMA(0,1,0).'),
                  ('Da, pentru că este singurul model cu p-value Ljung–Box peste 5%.', 'Yes, because it is the only model with a Ljung–Box p-value above 5%.')]),
 'interval-width': (1, [('Nu se schimbă: cu d = 1 intervalul crește ca √h, oricare ar fi θ.', 'It does not change: with d = 1 the interval grows like √h, whatever θ.'),
                        ('Crește cu θ: de la 2,14σ la θ = −0,9 la 12,96σ la θ = 0,5.', 'It rises with θ: from 2.14σ at θ = −0.9 to 12.96σ at θ = 0.5.'),
                        ('Scade cînd |θ| crește: orice componentă MA îngustează intervalul.', 'It falls as |θ| rises: any MA component narrows the interval.')]),
 'gdp-forecast': (1, [('ARIMA(0,1,0) cu derivă, pentru că testele clasifică PIB-ul ca I(1).', 'ARIMA(0,1,0) with drift, because the tests classify GDP as I(1).'),
                      ('Modelul staționar în jurul trendului, dar la mică distanță: 7,2 față de 7,6.', 'The trend-stationary model, but only just: 7.2 against 7.6.'),
                      ('Cele două modele au aceeași eroare: prognozele lor punctuale coincid.', 'The two models have the same error: their point forecasts coincide.')]),
 'overdiff': (1, [('Niciunul: creșterea SUA are ρ̂(1) > 0, deci a doua diferențiere este utilă.', 'None: US growth has ρ̂(1) > 0, so a second difference is useful.'),
                  ('Toate trei: ρ̂(1) = −0,49, varianța crește, θ̂ = −0,986.', 'All three: ρ̂(1) = −0.49, the variance rises, θ̂ = −0.986.'),
                  ('Doar o varianță mai mică: diferențierea elimină mereu variație.', 'Only a smaller variance: differencing always removes variation.')]),
 'inflation': (1, [('Da: un AICc mai mic înseamnă întotdeauna un model mai bun.', 'Yes: a lower AICc always means a better model.'),
                   ('Nu: AICc nu compară modele cu d diferit.', 'No: AICc does not compare models with different d.'),
                   ('Da, pentru că ADF nu respinge rădăcina unitară a inflației.', 'Yes, because ADF does not reject a unit root in inflation.')]),
 'eurron': (1, [('Da: un model care cîștigă pe 2024–2026 cîștigă pe orice perioadă.', 'Yes: a model that wins on 2024–2026 wins on any period.'),
                ('Nu: RMSE 0,1307 față de 0,1304.', 'No: RMSE 0.1307 against 0.1304.'),
                ('Da, cu mai mult: cu mai multe zile de test, cîștigul crește.', 'Yes, by more: with more test days, the gain grows.')]),
 'us-gdp': (1, [('Da: 2007 a fost un vîrf; din alte origini, modelul cu trend nimerește.', 'Yes: 2007 was a peak; from other origins the trend model gets it right.'),
                ('Nu: din T4 2000, abaterea la final este tot mare, −34,7.', 'No: from 2000Q4 the gap at the end is still large, −34.7.'),
                ('Nu, dar din T4 2000 ARIMA(1,1,0) cu derivă are o abatere apropiată de 0.', 'No, but from 2000Q4 the ARIMA(1,1,0) with drift has a gap close to 0.')]),
}

# blocks whose initial charts show the answer: the live charts are veiled too until „Verifică” / "Check" or a change of the controls
VEIL_CHARTS = ('spurious-mc', 'df-dist', 'adf-power')

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slide „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('Un test ADF cu constantă și trend dă τ = −2,9 pentru T = 200. Respingeți rădăcina unitară la 5%?',
         'An ADF test with constant and trend gives τ = −2.9 with T = 200. Do you reject the unit root at 5%?'),
      key=0, opts=[
  (('Nu: −2,9 > −3,43, valoarea critică de 5% cu constantă și trend.', 'No: −2.9 > −3.43, the 5% critical value with constant and trend.'),
   ('τ trebuie comparat cu valoarea Dickey–Fuller a regresiei cu constantă și trend; −2,9 nu este sub −3,43.', 'τ must be compared with the Dickey–Fuller value of the regression with constant and trend; −2.9 is not below −3.43.')),
  (('Da: −2,9 < −1,645.', 'Yes: −2.9 < −1.645.'),
   ('−1,645 este valoarea distribuției Normale; sub rădăcina unitară, τ nu urmează distribuția Normală, iar această valoare respinge greșit în 77% din cazuri.', '−1.645 is the value of the Normal distribution; under a unit root τ does not follow the Normal distribution, and this value wrongly rejects in 77% of the cases.')),
  (('Da: −2,9 < −2,87.', 'Yes: −2.9 < −2.87.'),
   ('−2,87 este valoarea critică a regresiei cu constantă, fără trend; cu trend, distribuția este mai la stînga.', '−2.87 is the critical value of the regression with a constant and no trend; with a trend the distribution lies further left.'))]),
 dict(q=('ADF nu respinge, iar KPSS nu respinge. Ce concluzie trageți?', 'ADF does not reject and KPSS does not reject. What do you conclude?'),
      key=1, opts=[
  (('Seria este I(1): ADF nu a respins rădăcina unitară.', 'The series is I(1): ADF did not reject the unit root.'),
   ('„Nerespins” nu înseamnă „demonstrat”; KPSS nu a respins nici staționaritatea.', '“Not rejected” does not mean “proven”; KPSS did not reject stationarity either.')),
  (('Neconcludent: eșantionul nu permite o decizie; raportăm ambele modele sau alegem d după scop.', 'Inconclusive: the sample does not allow a decision; report both models, or choose d by the purpose.'),
   ('Cele două teste au ipoteze nule opuse și niciunul nu respinge: puterea este mică de ambele părți.', 'The two tests have opposite null hypotheses and neither rejects: power is low on both sides.')),
  (('Seria este I(0): KPSS nu a respins staționaritatea.', 'The series is I(0): KPSS did not reject stationarity.'),
   ('Aceeași greșeală, în sens invers: nerespingerea KPSS nu demonstrează staționaritatea.', 'The same mistake, the other way round: a KPSS non-rejection does not prove stationarity.'))]),
 dict(q=('Pentru ARIMA(0,1,1) cu θ = −0,6 și σ = 1, cît este varianța erorii de prognoză la 5 pași?',
         'For ARIMA(0,1,1) with θ = −0.6 and σ = 1, what is the variance of the 5-step forecast error?'),
      key=2, opts=[
  (('5, ca la mersul aleator: hσ².', '5, as for the random walk: hσ².'),
   ('Mersul aleator are ψⱼ = 1; aici ψⱼ = 1 + θ = 0,4 pentru j ≥ 1.', 'The random walk has ψⱼ = 1; here ψⱼ = 1 + θ = 0.4 for j ≥ 1.')),
  (('1 + 4 · 0,6² = 2,44.', '1 + 4 · 0.6² = 2.44.'),
   ('Ponderile sînt 1 + θ = 0,4, nu |θ| = 0,6.', 'The weights are 1 + θ = 0.4, not |θ| = 0.6.')),
  (('1 + 4 · 0,4² = 1,64.', '1 + 4 · 0.4² = 1.64.'),
   ('σ²[1 + (h − 1)(1 + θ)²] = 1 + 4 · 0,16 = 1,64.', 'σ²[1 + (h − 1)(1 + θ)²] = 1 + 4 · 0.16 = 1.64.'))]),
]


def _predict_js():
    import json
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {}))
              for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch3-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH3_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH3_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n')


_predict_js()


if __name__ == '__main__':
    import interactive_common
    page_spec = types.SimpleNamespace(**{k: globals()[k] for k in ('N', 'TEX', 'QL', 'BLOCKS', 'PFX', 'SECTIONS', 'L', 'QA', 'TEXT_JS')})
    page_spec.widgets = lambda lang: {a: v for a, v in widgets_all(lang).items() if a in {b for b, _, _ in BLOCKS}}
    for _k in ('PREDICT', 'TICKET'):
        if _k in globals():
            setattr(page_spec, _k, globals()[_k])
    interactive_common.build(page_spec)
    for lg in ('en', 'ro'):
        _post(lg)
