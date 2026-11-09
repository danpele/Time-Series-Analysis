"""Build the interactive pages of Chapter 4 (seasonality and forecasting): interactive/ch4.html (EN) and interactive/ch4_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its
interactive version on the right. Logic in interactive/ch4.js, inputs in interactive/ch4_data.js
(Quantlets/Ch_04/export_interactive.py), slide images in interactive/slides/.
The result tables and worked examples of the lecture (seasonal differencing, the roots of 1 - z^s, HEGY, OCSB and Canova-Hansen,
the airline ACF and forecast recursion, the airline estimates by origin, the GDP model table with AICc, the GDP forecast table,
the adjustment factors, a Fourier pair, the Fourier fits, the Easter regressor and its coefficients, the DHR coefficients, the
Diebold-Mariano test by horizon and worked out, the cross-validation table with MASE, the GDP comparison, the combination
formula and the combination weights) get blocks of their own (EXTRA), placed by slide; slide images
interactive/slides/ch4x_*.jpg. On the web lecture (build_web_lecture.py) the EXTRA blocks travel inside a chart block of the
same section and ch4.js moves each one to its own frame.
"""
import os
import re
import sys
import types

from interactive_common import slider, select, play, charts

N = 4

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_04'
TEX = {'ro': 'RO/Cursuri/capitol4_sezonalitate_prognoza', 'en': 'EN/Courses/chapter4_seasonality_forecasting'}

# (anchor, chart file, section key); order of the slides
BLOCKS = [('series', 'tsa_ch4_series', 'seas'), ('seasonal-plot', 'tsa_ch4_seasonal_plot', 'seas'),
          ('unit-circle', 'tsa_ch4_unit_circle', 'roots'), ('gdp-diff', 'tsa_ch4_gdp_diff', 'roots'),
          ('theory-acf', 'tsa_ch4_theory_acf', 'sarima'),
          ('airline', 'tsa_ch4_airline', 'air'), ('airline-fc', 'tsa_ch4_airline_fc', 'air'),
          ('gdp-diag', 'tsa_ch4_gdp_diag', 'gdp'), ('gdp-fc', 'tsa_ch4_gdp_fc', 'gdp'),
          ('sa-nsa', 'tsa_ch4_sa_nsa', 'sa'),
          ('fourier', 'tsa_ch4_fourier', 'cal'), ('easter', 'tsa_ch4_easter', 'cal'),
          ('load-hourly', 'tsa_ch4_load_hourly', 'mult'), ('mstl', 'tsa_ch4_mstl', 'mult'), ('load-daily', 'tsa_ch4_load_daily', 'mult'),
          ('prophet', 'tsa_ch4_prophet', 'tp'),
          ('cv-scheme', 'tsa_ch4_cv_scheme', 'ev'), ('load-fc', 'tsa_ch4_load_fc', 'ev'), ('cv-results', 'tsa_ch4_cv_results', 'ev'),
          ('combination', 'tsa_ch4_combination', 'comb')]

PFX = {'series': 'se', 'seasonal-plot': 'sp', 'unit-circle': 'uc', 'gdp-diff': 'df', 'theory-acf': 'th', 'airline': 'ai', 'airline-fc': 'af',
       'gdp-diag': 'dg', 'gdp-fc': 'gf', 'sa-nsa': 'sa', 'fourier': 'fo', 'easter': 'ea', 'load-hourly': 'lh', 'mstl': 'ms', 'load-daily': 'ld',
       'prophet': 'pr', 'cv-scheme': 'cs', 'load-fc': 'lf', 'cv-results': 'cr', 'combination': 'cb'}

SECTIONS = {
 'ro': dict(seas='Sezonalitatea în datele economice', roots='Diferențierea sezonieră și rădăcinile unitare sezoniere', sarima='Modele SARIMA',
            air='Studiu de caz: modelul airline', gdp='Studiu de caz: PIB-ul neajustat al României', sa='Ajustarea sezonieră în practică',
            cal='Termeni Fourier și efecte de calendar', mult='Sezonalitate multiplă: consumul de electricitate', tp='TBATS și Prophet',
            ev='Evaluarea prognozelor', comb='Combinarea prognozelor'),
 'en': dict(seas='Seasonality in economic data', roots='Seasonal differencing and seasonal unit roots', sarima='SARIMA models',
            air='Case study: the airline model', gdp='Case study: seasonally unadjusted Romanian GDP', sa='Seasonal adjustment in practice',
            cal='Fourier terms and calendar effects', mult='Multiple seasonality: electricity load', tp='TBATS and Prophet',
            ev='Forecast evaluation', comb='Forecast combination'),
}

L = {
 'ro': dict(file='ch4_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 4 · Sezonalitate și prognoză', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, calculată în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas.',
   src='PIB-ul României: Eurostat, volume înlănțuite (prețuri 2010), serie neajustată (NSA) și ajustată sezonier și pentru efectele de calendar (SCA), în versiunea folosită în curs. '
       'Comerțul cu amănuntul, producția industrială și înnoptările turistice: Eurostat, pînă în iulie 2026; IAPC: Eurostat, pînă în august 2026. '
       'Consumul de electricitate: ENTSO-E, „monthly hourly load values”, România, 1 ianuarie 2022 – 30 iunie 2026, ora de iarnă a României. '
       'Pasagerii aerieni: setul de date AirPassengers (Box și Jenkins), 1949–1960. Prognozele validării încrucișate sînt cele din curs; '
       'variantele care nu apar în curs sînt rulări suplimentare ale codului capitolului, cu aceleași date.'),
 'en': dict(file='ch4.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 4 · Seasonality and Forecasting', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step.',
   src='Romanian GDP: Eurostat, chain-linked volumes (2010 prices), not adjusted (NSA) and seasonally and calendar adjusted (SCA), in the version used in the lecture. '
       'Retail trade, industrial production and tourism nights: Eurostat, up to July 2026; HICP: Eurostat, up to August 2026. '
       'Electricity load: ENTSO-E, “monthly hourly load values”, Romania, 1 January 2022 – 30 June 2026, Romanian winter time. '
       'Airline passengers: the AirPassengers data set (Box and Jenkins), 1949–1960. The cross-validation forecasts are those of the lecture; '
       'variants that do not appear in the lecture are extra runs of the chapter code, on the same data.'),
}

MODELS = ['(0,1,1)(0,1,1)4', '(1,1,0)(0,1,1)4', '(1,1,1)(0,1,1)4', '(0,1,2)(0,1,1)4', '(2,1,0)(0,1,1)4', '(0,1,1)(1,1,0)4', '(1,1,0)(1,1,0)4',
          '(0,1,1)(1,1,1)4', '(0,1,0)(0,1,1)4', '(0,1,1)(0,1,0)4']
CVM = ['Seasonal naive', 'ETS', 'SARIMA', 'DHR', 'TBATS', 'Prophet']


def _chart_widgets(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    sub4 = lambda m: m.replace(')4', ')₄')   # noqa: E731
    return {
     'series': (select('se-s', s('Tiparul sezonier al seriei', 'Seasonal pattern of the series'),
                       [('gdp', s('PIB (mediile trimestriale)', 'GDP (quarter means)')), ('tour', s('Înnoptări turistice (mediile lunare)', 'Tourism nights (month means)')),
                        ('infl', s('Inflația IAPC (mediile lunare)', 'HICP inflation (month means)'))], 'gdp')
                + slider('se-y0', s('De la anul', 'From year'), 2000, 2026, 1, 2000) + slider('se-y1', s('Pînă în anul', 'To year'), 2000, 2026, 1, 2026)
                + play('se'), charts('se-chart')),
     'seasonal-plot': (slider('sp-y', s('Anul evidențiat', 'Highlighted year'), 2012, 2025, 1, 2019) + play('sp'), charts('sp-chart')),
     'unit-circle': (select('uc-s', s('Perioada s din dreapta', 'Period s on the right'), [(v, v) for v in ('2', '3', '4', '6', '7', '12', '24')], '12') + play('uc'), charts('uc-chart')),
     'gdp-diff': (slider('df-end', s('Eșantionul se termină în', 'Sample ends in'), 19, 105, 1, 105) + play('df'), charts('df-chart')),
     'theory-acf': (slider('th-Phi', 'Φ (SAR)', -0.95, 0.95, 0.05, 0.8) + slider('th-Th', 'Θ (SMA, airline)', -0.95, 0.95, 0.05, -0.6)
                    + slider('th-th', 'θ (airline)', -0.9, 0.9, 0.05, -0.4) + play('th'), charts('th-chart')),
     'airline': (select('ai-tr', s('Transformarea', 'Transformation'),
                        [('ddl', 'ΔΔ₁₂ ln y' + s(' (curs)', ' (lecture)')), ('d12l', 'Δ₁₂ ln y'), ('d1l', 'Δ ln y'), ('ddy', s('ΔΔ₁₂ y, fără logaritm', 'ΔΔ₁₂ y, no log'))], 'ddl')
                 + play('ai'), charts('ai-chart')),
     'airline-fc': (slider('af-o', s('Originea prognozei', 'Forecast origin'), 1952, 1958, 1, 1958) + play('af'), charts('af-chart')),
     'gdp-diag': (select('dg-m', s('Modelul SARIMA', 'SARIMA model'), [(m, sub4(m)) for m in MODELS], '(0,1,0)(0,1,1)4') + play('dg'), charts('dg-chart')),
     'gdp-fc': (slider('gf-Th', s('Θ al MA sezonier (fixat)', 'Θ of the seasonal MA (fixed)'), -0.99, 0, 0.0001, -0.5765) + play('gf'), charts('gf-chart')),
     'sa-nsa': (slider('sa-q', s('Trimestrul', 'Quarter'), 41, 105, 1, 104) + play('sa'), charts('sa-chart')),
     'fourier': (slider('fo-K', s('Numărul de armonici K', 'Number of harmonics K'), 1, 6, 1, 3) + play('fo'), charts('fo-chart')),
     'easter': (slider('ea-w', s('Fereastra w (zile înainte de Paște)', 'Window w (days before Easter)'), 1, 20, 1, 10) + play('ea'), charts('ea-chart')),
     'load-hourly': (slider('lh-wk', s('Fereastra de trei săptămîni începe luni', 'The three-week window starts on Monday'), 0, 230, 1, 158)
                     + select('lh-yr', s('Anul pentru ACF', 'Year of the ACF'), [(v, v) for v in ('2022', '2023', '2024', '2025')], '2025') + play('lh'), charts('lh-chart')),
     'mstl': (select('ms-w', s('Fereastra de patru săptămîni', 'Four-week window'), [('2025-02-03', '2025-02-03')], '2025-02-03') + play('ms'), charts('ms-chart')),
     'load-daily': (select('ld-yr', s('Mediile pe zile, pentru', 'Means by day, for'), [('all', s('2022–2026', '2022–2026'))] + [(v, v) for v in ('2022', '2023', '2024', '2025', '2026')], 'all')
                    + play('ld'), charts('ld-chart')),
     'prophet': (select('pr-K', s('Armonici anuale K', 'Yearly harmonics K'), [(v, v + (s(' (implicit, curs)', ' (default, lecture)') if v == '10' else '')) for v in ('3', '5', '10', '20')], '10')
                 + select('pr-h', s('Sărbătorile legale', 'Public holidays'), [('1', s('incluse (curs)', 'included (lecture)')), ('0', s('neincluse', 'not included'))], '1')
                 + play('pr'), charts('pr-chart')),
     'cv-scheme': (slider('cs-h', s('Orizontul h', 'Horizon h'), 1, 6, 1, 3) + slider('cs-st', s('Pasul dintre origini', 'Step between origins'), 1, 4, 1, 2) + play('cs'), charts('cs-chart')),
     'load-fc': (select('lf-o', s('Originea prognozei', 'Forecast origin'), [('2026-04-06', '2026-04-06')], '2026-04-06') + play('lf'), charts('lf-chart')),
     'cv-results': (select('cr-m', s('Măsura', 'Measure'), [('MAE', 'MAE'), ('RMSE', 'RMSE')], 'MAE')
                    + select('cr-w', s('Ferestrele', 'Windows'), [('all', s('toate cele 26', 'all 26')), ('hol', s('cu sărbători', 'with holidays')), ('nohol', s('fără sărbători', 'without holidays'))], 'all')
                    + play('cr'), charts('cr-chart')),
     'combination': (select('cb-1', s('Prognoza 1 (ponderea w)', 'Forecast 1 (weight w)'), [(m, m) for m in CVM], 'DHR')
                     + select('cb-2', s('Prognoza 2 (ponderea 1 − w)', 'Forecast 2 (weight 1 − w)'), [(m, m) for m in CVM], 'Prophet') + play('cb'), charts('cb-chart')),
    }


QA = {
 'ro': {
  'series': ('În 2010–2019, în ce lună este cea mai mare inflația lunară medie a IAPC?',
             'În ianuarie: 0,575% în medie, urmată de octombrie, 0,487%; cea mai mică este în iunie, −0,368% (alimentele proaspete). Ordinea depinde de perioadă: pe 2015–2019, octombrie are 0,575% și mai 0,457%. '
             'Pentru PIB, media trimestrului 4 este cu 57% peste media trimestrului 1 (27,8 și 43,6 mld. EUR, 2000–2026); pentru turism, august are de 3,7 ori mai multe înnoptări decît ianuarie (2015–2019).'),
  'seasonal-plot': ('Butonul ▶ trece prin anii 2012–2025. În ce an este cel mai mare raportul dintre înnoptările din august și cele din ianuarie?',
                    'În 2021: 5,3, față de 3,4 în 2019 și 2,2 în 2020. În ianuarie 2021 restricțiile țineau încă turismul jos, iar vara a revenit; în 2020 a fost afectată și vara, iar aprilie 2020 a avut doar 0,045 milioane de înnoptări. '
                    'Forma anului rămîne aceeași: august este luna de vîrf în fiecare an. La PIB, cele patru subserii păstrează ordinea T1 < T2 < T3 < T4 în toți cei 21 de ani din 2005–2025.'),
  'unit-circle': ('Pentru date zilnice cu tipar săptămînal, s = 7. Cîte rădăcini reale are 1 − z⁷?',
                  'Una singură, z = 1. Rădăcina z = −1 (perioada de 2) există doar cînd s este par; pentru s = 7 celelalte șase rădăcini formează trei perechi complex conjugate, cu perioadele 7, 3,5 și 2,33 zile. '
                  'Pentru s = 4: 1, −1 și ±i; pentru s = 12: 1, −1 și cinci perechi.'),
  'gdp-diff': ('Pe tot eșantionul, ACF pentru ΔΔ₄ ln Y este −0,40 la lagul 4. Rămîne semnificativă dacă eșantionul se oprește în T4 2019, înainte de pandemie?',
               'Da: −0,34, în afara benzii ±0,23 (T = 75). Valoarea negativă la lagul sezonier nu vine din 2020, ci din tiparul sezonier care se modifică lent: semnătura unui MA sezonier. '
               'La Δ ln Y, ACF rămîne 0,94 la lagul 4 și pe eșantionul scurt; la Δ₄ ln Y, ACF la lagul 1 este 0,68.'),
  'theory-acf': ('Cu θ = −0,4 și Θ = −0,6, sateliții de la lagurile 11 și 13 au 0,152. Ce se întîmplă cu ei dacă θ devine +0,4?',
                 'Își schimbă semnul: −0,152. Sateliții sînt produsul ρ₁ρ₁₂, iar ρ₁ = θ/(1 + θ²) trece de la −0,345 la 0,345, în timp ce ρ₁₂ = −0,441 rămîne. '
                 'La SAR(1)₁₂ cu Φ = 0,8, ACF descrește 0,8; 0,64; 0,512 la lagurile 12, 24, 36; la SMA(1)₁₂ cu Θ = −0,6, PACF este −0,242 la lagul 24.'),
  'airline': ('Fără diferența obișnuită, doar cu Δ₁₂ ln y, cum arată ACF?',
              'Autocorelații pozitive care scad încet: 0,71 la lagul 1, 0,48 la lagul 3, încă 0,19 la lagul 8. Trendul nu este eliminat, iar seria rămîne nestaționară. '
              'Cu ΔΔ₁₂ ln y rămîn valori nenule la lagurile 1 (−0,34) și 12 (−0,39); în afara benzii ±0,17 mai ies lagurile 3, 9, 23 și 32. '
              'PACF are −0,341, −0,013 și −0,193 la lagurile 1, 2 și 3: valoarea de la lagul 2 este practic zero.'),
  'airline-fc': ('Cu originea în decembrie 1958, intervalul de 95% pentru decembrie 1960 are lățimea 239. Ce lățime are dacă originea este decembrie 1954, cu 72 de luni înainte?',
                 'Mult mai mare: 852 (intervalul [201; 1053]), deoarece cele două rădăcini unitare fac varianța erorii să crească cu orizontul. Valoarea reală, 432, rămîne în interval de la orice origine. '
                 'Din 1958: prognoza 388, intervalul [286; 526], MAPE 8,5% față de 15,5% pentru metoda sezonieră naivă.'),
  'gdp-diag': ('SARIMA(0,1,1)(1,1,0)₄ înlocuiește MA-ul sezonier cu un AR sezonier. Sînt reziduurile lui necorelate după testul Ljung–Box?',
               'Nu: Q*(8) = 19,5 cu 6 grade de libertate, p = 0,003, iar AICc crește la 529,9, față de 519,7. Pentru SARIMA(0,1,0)(0,1,1)₄: Q*(8) = 8,55 cu 7 grade de libertate, p = 0,286, și Q*(12) cu p = 0,526. '
               'În toate modelele reziduurile nu urmează distribuția Normală: pentru modelul BIC, Jarque–Bera 22,0 și reziduul −11,4% în T2 2020.'),
  'gdp-fc': ('Butonul ▶ mișcă Θ de la −0,95 la 0. Ce devine creșterea anuală prognozată cînd Θ = 0?',
             'Aceeași în fiecare trimestru: −1,96%, creșterea anuală din T2 2026. Cu Θ = 0, modelul este Δ₄y_t = Δ₄y_{t−1} + ε_t, iar prognoza repetă ultima creștere anuală. '
             'Cu Θ̂ = −0,577, tiparul are ponderea 1 + Θ̂ = 0,42 pentru anul cel mai recent: creșterea implicată −0,81%; 0,53%; 0,83%; 0,55%, apoi 0,55%. Cu Θ = −0,95, tiparul este aproape fix, iar creșterea implicată urcă la 3,99%.'),
  'sa-nsa': ('În T1 2026, PIB-ul neajustat scade cu 42,1% (puncte logaritmice) față de trimestrul anterior. Cît s-a modificat seria SCA?',
             'Cu −0,06%: practic deloc. Scăderea seriei neajustate este tiparul sezonier, cu factorul NSA/SCA 0,786 în T1 și 1,197 în T4 2025. '
             'Creșterea anuală spune același lucru în ambele serii: −1,50% (NSA) și −1,14% (SCA). Totalurile anuale coincid: 196,4 mld. EUR în 2024.'),
  'fourier': ('Cîte armonici sînt necesare pentru R² ≥ 0,98?',
              'Trei: K = 3, șase coeficienți, R² = 0,989. K = 1 dă 0,925, K = 2 dă 0,966. K = 5, cu zece coeficienți, dă 0,999. Pentru K = 6, sin(2π6t/12) este zero în fiecare lună și este omis: '
              'cu cos(2π6t/12), cei 11 coeficienți trec prin toate cele 12 puncte (R² = 1), ca 11 variabile dummy.'),
  'easter': ('Cu fereastra w = 3 (doar cele trei zile dinaintea Paștelui), efectul estimat al Paștelui asupra vînzărilor de alimente crește sau scade?',
             'Scade la 1,6% (eroarea standard 1,6, t = 1,0) și nu se mai separă de zgomot. Cu w = 10 (curs): 5,9% (2,3; t = 2,6); cu w = 12: 6,1% (t = 2,9). '
             'Cumpărăturile de Paște se întind pe mai multe zile; un regresor prea îngust pune o parte din ele în luna greșită. Zilele lucrătoare rămîn în jur de +0,6% pentru alimente și +1,0% pentru total, pentru orice w.'),
  'load-hourly': ('Pentru 2025, care autocorelație este mai mare: la lagul 24 (aceeași oră ieri) sau la lagul 168 (aceeași oră acum o săptămînă)?',
                  'La lagul 168: 0,817 față de 0,811. Diferența este mai mare în 2023 (0,841 și 0,800) și 2024 (0,846 și 0,804). La lagul 84 (o jumătate de săptămînă) ACF este doar 0,22. '
                  'Ciclul zilnic: minimul la ora 3:00 (5,3 GW), maximul la ora 19:00 (7,3 GW).'),
  'mstl': ('În fereastra din februarie 2025, componenta neregulată are 2% din varianța din jurul trendului. Cît are în fereastra 7 aprilie – 4 mai 2025, care cuprinde Paștele?',
           '13%: Paștele (20 aprilie) și 1 Mai nu urmează tiparul zilnic sau săptămînal, deci ajung în componenta neregulată. În februarie: 70% componenta zilnică, 29% cea săptămînală. '
           'Ponderea componentei săptămînale variază între 20% (august) și 43% (iunie).'),
  'load-daily': ('Care este ziua cu cel mai mic consum din 2023?',
                 'Duminica Rusaliilor, 4 iunie 2023: 4,37 GW, sub Duminica Paștelui, 16 aprilie 2023 (4,48 GW). În 2022, 2024 și 2025 cea mai joasă zi este Duminica Paștelui; '
                 'pe tot eșantionul, 20 aprilie 2025 (3,79 GW, media 6,18 GW). Sîmbăta este cu 9%, duminica cu 17% sub miercuri, iar ianuarie cu 25% peste mai.'),
  'prophet': ('Dacă Prophet este estimat fără lista sărbătorilor, ce se întîmplă cu eroarea medie absolută în eșantion?',
              'Crește de la 197 la 226 MW. Fără sărbători, zilele de Crăciun și Paște rămîn în reziduuri, iar componenta anuală se deformează: maximul ei trece de la 7 decembrie la 24 ianuarie 2024. '
              'Cu sărbătorile, efectele sînt −1,24 GW (Crăciunul), −1,10 GW (Anul Nou) și −1,04 GW (Paștele).'),
  'cv-scheme': ('Cu pasul 1 între origini și orizontul h = 3, în cîte ferestre de prognoză apare aceeași dată?',
                'În trei: ferestrele consecutive se suprapun pe h − 1 = 2 perioade, deci erorile sînt corelate pînă la lagul h − 1, iar testul Diebold–Mariano folosește h − 1 autocovarianțe. '
                'În schema cursului originile sînt la 14 zile, iar h = 14: ferestrele nu se suprapun, deci testul are o valoare pe fereastră și h = 1 în formulă.'),
  'load-fc': ('În fereastra Paștelui (originea 6 aprilie 2026), care model are cea mai mică MAE?',
              'ETS, cu 299 MW, urmat de TBATS (309) și DHR (344); metoda sezonieră naivă are 686, Prophet 616. ETS nu are o linie în graficul din slide, deși are cea mai mică eroare în această fereastră. '
              'După Paște consumul rămîne sub toate prognozele, iar ETS și TBATS urmează nivelul ultimelor zile.'),
  'cv-results': ('În cele 8 ferestre care cuprind o sărbătoare, mai are DHR cea mai mică MAE?',
                 'Nu: TBATS 317 MW și DHR 318 MW sînt la egalitate, iar testul DM nu le separă (HLN 0,83, p = 0,431). În cele 18 ferestre fără sărbători, DHR are 258 MW, înaintea combinării (290). '
                 'Pe toate cele 26 de ferestre: DHR 276, combinarea 305, Prophet 328 MW; ETS, SARIMA și TBATS nu sînt separate de metoda sezonieră naivă (p = 0,431; 0,231; 0,145).'),
  'combination': ('Combinați ETS (MAE 381 MW) și Prophet (328 MW), ale căror erori au corelația 0,36. Ce MAE are media cu ponderi egale?',
                  '295 MW: sub ambele componente. Cu erori slab corelate, greșelile se compensează parțial; ponderea cea mai bună retrospectiv este w = 0,37 (MAE 286). '
                  'Pentru DHR și Prophet (ρ = 0,75): 283 MW cu ponderi egale, față de 276 pentru DHR singur; pentru ETS și TBATS (ρ = 0,98), media cu ponderi egale (370) este mai slabă decît TBATS singur (363).'),
 },
 'en': {
  'series': ('On 2010–2019, in which month is the mean monthly HICP inflation highest?',
             'In January: 0.575% on average, followed by October, 0.487%; the lowest is in June, −0.368% (fresh food). The ranking depends on the period: on 2015–2019 October has 0.575% and May 0.457%. '
             'For GDP, the mean of quarter 4 is 57% above the mean of quarter 1 (27.8 and 43.6 bn EUR, 2000–2026); for tourism, August has 3.7 times as many nights as January (2015–2019).'),
  'seasonal-plot': ('The ▶ button goes through the years 2012–2025. In which year is the ratio of August to January tourism nights largest?',
                    'In 2021: 5.3, against 3.4 in 2019 and 2.2 in 2020. In January 2021 restrictions still kept tourism low, and the summer recovered; in 2020 the summer was hit as well, and April 2020 had only 0.045 million nights. '
                    'The shape of the year stays the same: August is the peak month in every year. For GDP, the four subseries keep the order Q1 < Q2 < Q3 < Q4 in all 21 years of 2005–2025.'),
  'unit-circle': ('For daily data with a weekly pattern, s = 7. How many real roots does 1 − z⁷ have?',
                  'Only one, z = 1. The root z = −1 (period 2) exists only when s is even; for s = 7 the other six roots form three complex-conjugate pairs, with periods 7, 3.5 and 2.33 days. '
                  'For s = 4: 1, −1 and ±i; for s = 12: 1, −1 and five pairs.'),
  'gdp-diff': ('On the full sample, the ACF of ΔΔ₄ ln Y is −0.40 at lag 4. Is it still significant if the sample stops in 2019Q4, before the pandemic?',
               'Yes: −0.34, outside the ±0.23 band (T = 75). The negative value at the seasonal lag does not come from 2020 but from a seasonal pattern that changes slowly: the signature of a seasonal MA. '
               'For Δ ln Y the ACF stays at 0.94 at lag 4 on the short sample as well; for Δ₄ ln Y the ACF at lag 1 is 0.68.'),
  'theory-acf': ('With θ = −0.4 and Θ = −0.6, the satellites at lags 11 and 13 are 0.152. What happens to them if θ becomes +0.4?',
                 'They change sign: −0.152. The satellites are the product ρ₁ρ₁₂, and ρ₁ = θ/(1 + θ²) goes from −0.345 to 0.345, while ρ₁₂ = −0.441 stays. '
                 'For SAR(1)₁₂ with Φ = 0.8 the ACF decays 0.8, 0.64, 0.512 at lags 12, 24, 36; for SMA(1)₁₂ with Θ = −0.6 the PACF is −0.242 at lag 24.'),
  'airline': ('Without the regular difference, with Δ₁₂ ln y only, what does the ACF look like?',
              'Positive autocorrelations that decay slowly: 0.71 at lag 1, 0.48 at lag 3, still 0.19 at lag 8. The trend is not removed, and the series stays non-stationary. '
              'With ΔΔ₁₂ ln y, non-zero values remain at lags 1 (−0.34) and 12 (−0.39); lags 3, 9, 23 and 32 also leave the ±0.17 band. '
              'The PACF is −0.341, −0.013 and −0.193 at lags 1, 2 and 3: the value at lag 2 is practically zero.'),
  'airline-fc': ('With the origin in December 1958, the 95% interval for December 1960 has a width of 239. What width does it have if the origin is December 1954, 72 months ahead?',
                 'Much larger: 852 (the interval [201, 1053]), because the two unit roots make the forecast-error variance grow with the horizon. The actual value, 432, stays inside the interval from every origin. '
                 'From 1958: forecast 388, interval [286, 526], MAPE 8.5% against 15.5% for the seasonal naive method.'),
  'gdp-diag': ('SARIMA(0,1,1)(1,1,0)₄ replaces the seasonal MA by a seasonal AR. Are its residuals uncorrelated by the Ljung–Box test?',
               'No: Q*(8) = 19.5 with 6 degrees of freedom, p = 0.003, and the AICc rises to 529.9, against 519.7. For SARIMA(0,1,0)(0,1,1)₄: Q*(8) = 8.55 with 7 degrees of freedom, p = 0.286, and Q*(12) has p = 0.526. '
               'In every model the residuals do not follow the Normal distribution: for the BIC model, Jarque–Bera 22.0 and the residual of −11.4% in 2020Q2.'),
  'gdp-fc': ('The ▶ button moves Θ from −0.95 to 0. What does the forecast year-on-year growth become when Θ = 0?',
             'The same in every quarter: −1.96%, the year-on-year growth of 2026Q2. With Θ = 0 the model is Δ₄y_t = Δ₄y_{t−1} + ε_t, and the forecast repeats the last year-on-year growth. '
             'With Θ̂ = −0.577 the pattern gives weight 1 + Θ̂ = 0.42 to the most recent year: implied growth −0.81%, 0.53%, 0.83%, 0.55%, then 0.55%. With Θ = −0.95 the pattern is almost fixed, and the implied growth rises to 3.99%.'),
  'sa-nsa': ('In 2026Q1, unadjusted GDP falls by 42.1% (log points) from the previous quarter. How much did the SCA series change?',
             'By −0.06%: practically not at all. The fall of the unadjusted series is the seasonal pattern, with an NSA/SCA factor of 0.786 in Q1 and 1.197 in 2025Q4. '
             'Year-on-year growth says the same in both series: −1.50% (NSA) and −1.14% (SCA). The annual totals coincide: 196.4 bn EUR in 2024.'),
  'fourier': ('How many harmonics are needed for R² ≥ 0.98?',
              'Three: K = 3, six coefficients, R² = 0.989. K = 1 gives 0.925, K = 2 gives 0.966. K = 5, with ten coefficients, gives 0.999. For K = 6, sin(2π6t/12) is zero in every month and is left out: '
              'with cos(2π6t/12), the 11 coefficients pass through all 12 points (R² = 1), like 11 dummy variables.'),
  'easter': ('With the window w = 3 (only the three days before Easter), does the estimated Easter effect on food sales rise or fall?',
             'It falls to 1.6% (standard error 1.6, t = 1.0) and is no longer separated from noise. With w = 10 (lecture): 5.9% (2.3; t = 2.6); with w = 12: 6.1% (t = 2.9). '
             'Easter shopping spreads over several days; a regressor that is too narrow puts part of it in the wrong month. Working days stay around +0.6% for food and +1.0% for the total, for every w.'),
  'load-hourly': ('For 2025, which autocorrelation is larger: at lag 24 (the same hour yesterday) or at lag 168 (the same hour a week ago)?',
                  'At lag 168: 0.817 against 0.811. The gap is larger in 2023 (0.841 and 0.800) and 2024 (0.846 and 0.804). At lag 84 (half a week) the ACF is only 0.22. '
                  'Daily cycle: the minimum at 3:00 (5.3 GW), the maximum at 19:00 (7.3 GW).'),
  'mstl': ('In the February 2025 window, the remainder has 2% of the variance around the trend. How much does it have in the window 7 April – 4 May 2025, which contains Easter?',
           '13%: Easter (20 April) and 1 May follow neither the daily nor the weekly pattern, so they end up in the remainder. In February: 70% the daily component, 29% the weekly one. '
           'The share of the weekly component ranges from 20% (August) to 43% (June).'),
  'load-daily': ('Which is the day with the lowest load in 2023?',
                 'Pentecost Sunday, 4 June 2023: 4.37 GW, below Easter Sunday, 16 April 2023 (4.48 GW). In 2022, 2024 and 2025 the lowest day is Easter Sunday; '
                 'over the whole sample, 20 April 2025 (3.79 GW, mean 6.18 GW). Saturday is 9% and Sunday 17% below Wednesday, and January is 25% above May.'),
  'prophet': ('If Prophet is estimated without the list of holidays, what happens to the in-sample mean absolute error?',
              'It rises from 197 to 226 MW. Without holidays, the Christmas and Easter days stay in the residuals and the yearly component is distorted: its maximum moves from 7 December to 24 January 2024. '
              'With the holidays, the effects are −1.24 GW (Christmas), −1.10 GW (New Year) and −1.04 GW (Easter).'),
  'cv-scheme': ('With a step of 1 between origins and horizon h = 3, in how many forecast windows does the same date appear?',
                'In three: consecutive windows overlap by h − 1 = 2 periods, so the errors are correlated up to lag h − 1, and the Diebold–Mariano test uses h − 1 autocovariances. '
                'In the design of the lecture the origins are 14 days apart and h = 14: the windows do not overlap, so the test has one value per window and h = 1 in the formula.'),
  'load-fc': ('In the Easter window (origin 6 April 2026), which model has the lowest MAE?',
              'ETS, with 299 MW, followed by TBATS (309) and DHR (344); the seasonal naive method has 686, Prophet 616. ETS has no line in the chart of the slide, although it has the lowest error in this window. '
              'After Easter the load stays below all forecasts, and ETS and TBATS follow the level of the last days.'),
  'cv-results': ('In the 8 windows that contain a holiday, does DHR still have the lowest MAE?',
                 'No: TBATS 317 MW and DHR 318 MW are level, and the DM test does not separate them (HLN 0.83, p = 0.431). In the 18 windows without holidays, DHR has 258 MW, ahead of the combination (290). '
                 'Over all 26 windows: DHR 276, the combination 305, Prophet 328 MW; ETS, SARIMA and TBATS are not separated from the seasonal naive method (p = 0.431, 0.231, 0.145).'),
  'combination': ('Combine ETS (MAE 381 MW) and Prophet (328 MW), whose errors have a correlation of 0.36. What MAE does the equal-weight average have?',
                  '295 MW: below both components. With weakly correlated errors the mistakes partly offset each other; the best weight in hindsight is w = 0.37 (MAE 286). '
                  'For DHR and Prophet (ρ = 0.75): 283 MW with equal weights, against 276 for DHR alone; for ETS and TBATS (ρ = 0.98) the equal-weight average (370) is worse than TBATS alone (363).'),
 },
}

TEXT_JS = {
 'ro': r"""(function () {
const MON = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
const H = { 'New Year': 'Anul Nou', 'Union Day': 'Ziua Unirii', Easter: 'Paștele', 'Labour Day': 'Ziua Muncii', "Children's Day": 'Ziua Copilului', Pentecost: 'Rusaliile',
  Assumption: 'Adormirea Maicii Domnului', "St Andrew's Day": 'Sfîntul Andrei', 'National Day': 'Ziua Națională', Christmas: 'Crăciunul', Epiphany: 'Boboteaza și Sfîntul Ioan' };
const M = { 'Seasonal naive': 'sezonieră naivă', Combination: 'combinarea' };
const Q = s => `T${s[5]} ${s.slice(0, 4)}`, dmy = d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;
const hn = x => H[x] || x, mn = x => M[x] || x, j = a => a.join('; '), de = n => { const r = n % 100; return (r === 0 && n > 0) || r >= 20 ? ' de' : ''; };
window.CH4_TEXT = {
 mon: MON, q: Q, dmy, dec: y => `decembrie ${y}`, lect: ' (curs)', bestBic: ' · cel mai mic BIC', bestAicc: ' · cel mai mic AICc',
 dmyEn: d => `${+d.slice(8, 10)} ` + ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][+d.slice(5, 7) - 1] + ' ' + d.slice(0, 4),
 selYear: y => `selected year: ${y}`,
 se_g: (a, b, m, p) => `PIB, ${a}–${b}: mediile trimestriale ${j(m)} mld. EUR (prețuri 2010); media T4 este cu ${p}% peste media T1.`,
 se_t: (a, b, r, mx, mn0, au, ja) => `Înnoptări turistice, ${a}–${b}: august ${au}, ianuarie ${ja} milioane în medie, de ${r} ori mai multe în august; luna cu cele mai multe înnoptări: ${mx}, cu cele mai puține: ${mn0}.`,
 se_i: (a, b, m1, v1, m2, v2, m0, v0) => `Inflația IAPC, ${a}–${b}: cea mai mare medie lunară în ${m1} (${v1}%), apoi în ${m2} (${v2}%); cea mai mică în ${m0} (${v0}%).`,
 sp1: (y, au, ja, r, ap, mx, mn0) => `${y}: ${au} milioane de înnoptări în august, ${ja} în ianuarie (raportul ${r}); aprilie ${ap}. Luna de vîrf: ${mx}; minimul: ${mn0}.`,
 sp2: (m, k, n) => `PIB, 2005–2026: mediile trimestriale ${j(m)} mld. EUR; ordinea T1 < T2 < T3 < T4 se păstrează în ${k} din ${n}${de(n)} ani (2005–2025).`,
 uc1: (s, nr, np) => `1 − z<sup>${s}</sup> are ${s}${de(s)} rădăcini pe cercul unitate: ${nr === 2 ? 'două reale, z = 1 și z = −1' : 'una reală, z = 1 (z = −1 este rădăcină doar pentru s par)'}, și ${np} ${np === 1 ? 'pereche' : 'perechi'} complex conjugate.`,
 uc2: (s, L) => `Frecvențele k/${s} și perioadele ${s}/k: ` + L.map(([k, p]) => `k = ${k}: perioada ${(Math.round(p * 100) / 100).toString().replace('.', ',')}`).join('; ') + '; k = 0: frecvența 0 (trendul).',
 df0: (q, n) => `Eșantionul T1 2000 – ${q}: ${n} trimestre.`,
 df1: (k, n, sd, r1, r4, r8, b) => ['Δ ln Y', 'Δ₄ ln Y', 'ΔΔ₄ ln Y'][k] + `: T = ${n}, abaterea standard ${sd}%; ACF ${r1} la lagul 1, ${r4} la lagul 4, ${r8} la lagul 8 (banda ±${b}).`,
 th1: (P, a, b, c, p) => `SAR(1)<sub>12</sub>, Φ = ${P}: ACF ${a}; ${b}; ${c} la lagurile 12, 24, 36; PACF ${p} la lagul 12 și 0 în rest.`,
 th2: (t, a, p1, p2, p3) => `SMA(1)<sub>12</sub>, Θ = ${t}: ACF ${a} la lagul 12 și 0 în rest; PACF ${p1}; ${p2}; ${p3} la lagurile 12, 24, 36.`,
 th3: (a, b, r1, r12, r11, r13) => `Airline, θ = ${a}, Θ = ${b}: ρ₁ = ${r1}, ρ₁₂ = ${r12}; sateliții ρ₁₁ = ${r11} și ρ₁₃ = ${r13} sînt produsul ρ₁ρ₁₂.`,
 ai1: (nm, n, b) => `${nm}: ${n} observații, banda ±${b}.`,
 ai2: (a, b, c, d, l) => `ACF: ${a} la lagul 1, ${b} la lagul 3, ${c} la lagul 12, ${d} la lagul 24; lagurile în afara benzii: ${l}.`,
 ai3: (a, b, c, d, e, l) => `PACF: ${a}; ${b}; ${c} la lagurile 1, 2, 3; ${d} la lagul 12, ${e} la lagul 24; lagurile în afara benzii: ${l}.`,
 af0: (y, l, h) => `Estimat pe 1949–${y}, ${h}${de(h)} prognoze pentru ianuarie ${y + 1} – decembrie 1960` + (l ? ' (curs).' : ' (o rulare suplimentară a codului capitolului).'),
 af1: (t, ts, T0, Ts, s) => `θ̂ = ${t} (${ts}), Θ̂ = ${T0} (${Ts}); abaterea standard a reziduurilor ${s}% pe lună.`,
 af2: (a, n) => `MAPE: ${a}% pentru modelul airline, ${n}% pentru metoda sezonieră naivă.`,
 af3: (f, lo, hi, a, ok) => `Decembrie 1960: prognoza ${f}, intervalul de 95% [${lo}; ${hi}], valoarea reală ${a}: ${ok ? 'în interval' : 'în afara intervalului'}.`,
 af4: (a, b, h) => `Lățimea intervalului de 95%: ${a} în prima lună, ${b} în luna ${h}.`,
 dg1: (m, k, a, b, s, ib, ia) => `SARIMA${m.replace(/\)4$/, ')₄')}: ${k} ${k === 1 ? 'parametru' : 'parametri'} ARMA, AICc ${a}, BIC ${b}` + (ib ? ' (cel mai mic BIC)' : '') + (ia ? ' (cel mai mic AICc)' : '') + `; σ̂ = ${s}%.`,
 dg2: p => `Estimările (erorile standard): ${p}.`,
 dg3: (q, df, p8, p12) => `Ljung–Box: Q*(8) = ${q} cu ${df} grade de libertate, ${p8}; Q*(12): ${p12}.`,
 dg4: (jb, p, q, v, z) => `Jarque–Bera ${jb} (${p}); cel mai mare reziduu: ${v}% în ${q} (${z} abateri standard).`,
 gf0: (t, w, l, s) => `Θ = ${t}` + (l ? ' (estimarea din curs)' : ' (fixat; σ rescalat după reziduuri)') + `: ponderea anului cel mai recent în tiparul prognozei 1 + Θ = ${w}; σ = ${s}%.`,
 gf1: (q, f) => `Prognozele pentru ${q} și următoarele trei trimestre: ${j(f)} mld. EUR.`,
 gf2: (a, b, c, d) => `Intervalul de 95%: [${a}; ${b}] pentru primul trimestru, [${c}; ${d}] pentru al optulea.`,
 gf3: g => `Creșterea anuală implicată: ${j(g)}%.`,
 sa1: (q, a, b) => `${q}, față de trimestrul anterior: NSA ${a}%, SCA ${b}% (puncte logaritmice).`,
 sa2: (a, b) => `Față de același trimestru al anului anterior: NSA ${a}%, SCA ${b}%.`,
 sa3: f => `Factorul sezonier și de calendar NSA/SCA: ${f}.`,
 sa4: (y, a, b) => `Totalul anului ${y}: NSA ${a}, SCA ${b} mld. EUR.`,
 fo1: (K, n, r) => `K = ${K}: ${n} coeficienți, R² = ${r}.`,
 fo2: a => `Amplitudinile armonicelor k = 1, 2, …: ${j(a)} (log).`,
 fo3: (a, b) => `sin(2π6t/12) este zero în fiecare lună și este omis: cu cos(2π6t/12), cei 11 coeficienți trec prin toate cele 12 puncte (R² = ${b}); K = 5, cu 10 coeficienți, dă R² = ${a}.`,
 ea1: (w, l, s, n) => `w = ${w} zile` + (l ? ' (curs)' : ' (o rulare suplimentară a codului capitolului)') + `: panta dreptei MCMMP (OLS) ${s} puncte procentuale, ${n}${de(n)} luni (martie–mai, fără 2020–2021).`,
 ea2: (b, se, t, w, ws) => `Alimente, regresie cu erori SARIMA(0,1,1)(0,1,1)<sub>12</sub>: Paștele ${b}% (eroarea standard ${se}, t = ${t}); o zi lucrătoare în plus ${w}% (${ws}).`,
 ea3: (b, se, t, w, ws) => `Totalul comerțului: Paștele ${b}% (${se}, t = ${t}); zilele lucrătoare ${w}% (${ws}).`,
 ea4: y => `Paștele în mai: ${y}.`,
 lh1: (a, b, m, lo, hi) => `${a} – ${b}: media ${m} GW, minimul ${lo}, maximul ${hi} GW.`,
 lh2: (y, n, a, b, c, d) => `ACF pentru ${n}${de(n)} ore din ${y}: ${a} la lagul 12, ${b} la lagul 24, ${c} la lagul 84, ${d} la lagul 168.`,
 lh3: (y, p, v, t, w) => `Profilul zilnic în ${y}: maximul la ora ${p}:00 (${v} GW), minimul la ora ${t}:00 (${w} GW).`,
 ms1: (a, b, l) => `Fereastra ${a} – ${b}` + (l ? ' (curs).' : ' (o rulare suplimentară a codului capitolului).'),
 ms2: (a, b, c) => `Din varianța în jurul trendului: ${a}% componenta zilnică, ${b}% componenta săptămînală, ${c}% componenta neregulată.`,
 ms3: (a, b) => `Amplitudinea în prima săptămînă: ${a} GW pentru componenta zilnică, ${b} GW pentru cea săptămînală.`,
 ld1: (p, n) => (p === 'all' ? 'Tot eșantionul, 2022–2026' : `Anul ${p}`) + `: ${n}${de(n)} zile.`,
 ld2: (a, b) => `Fără sărbători: sîmbăta ${a}%, duminica ${b}% față de miercuri.`,
 ld3: a => `Ianuarie față de mai: +${a}%.`,
 ld4: (L, m, e) => 'Cele mai joase zile: ' + L.map(([d, h, v]) => `${d}${h ? ' (' + hn(h) + ')' : ''}, ${v} GW`).join('; ') + `; media ${m} GW` + (e ? `; zilele de Paște (vineri–luni) ${e} GW.` : '.'),
 pr1: (K, h, l) => `Prophet cu K = ${K} armonici anuale, ${h ? 'cu' : 'fără'} sărbătorile legale` + (l ? ' (curs).' : ' (o rulare suplimentară a codului capitolului).'),
 pr2: (a, b, n) => `Trendul de la ${a} la ${b} GW; puncte de schimbare folosite: ${n} din 25.`,
 pr3: (a, b, c) => `Săptămînal: miercuri ${a}, sîmbătă ${b}, duminică ${c} GW.`,
 pr4: (a, da, b, db) => `Anual (2024): maximul ${a} GW pe ${da}, minimul ${b} GW pe ${db}.`,
 pr5: L => 'Cele mai mari efecte ale sărbătorilor: ' + L.map(([h, v]) => `${hn(h)} ${v}`).join('; ') + ' GW.',
 pr6: a => `Eroarea medie absolută în eșantion: ${a} MW.`,
 cs1: (h, s, o, c) => `h = ${h}, pasul ${s}: ` + (o ? `ferestrele consecutive se suprapun pe ${h - s} ${h - s === 1 ? 'perioadă' : 'perioade'}; o dată apare în pînă la ${c} ferestre, iar erorile sînt corelate pînă la lagul ${h - 1}.` : 'ferestrele nu se suprapun; fiecare dată apare într-o singură fereastră.'),
 cs2: () => 'În curs: 26 de origini la 14 zile, h = 14, fereastră care se extinde; ferestrele nu se suprapun.',
 lf1: (o, a, b, h, l) => `Originea ${o}` + (l ? ' (curs)' : '') + `; fereastra ${a} – ${b}` + (h.length ? '; sărbători: ' + h.map(hn).join(', ') + '.' : '; fără sărbători.'),
 lf2: L => 'MAE în fereastră (MW): ' + L.map(([m, v]) => `${mn(m)} ${v}`).join(', ') + '.',
 cr1: (w, n) => ({ all: 'Toate ferestrele', hol: 'Ferestrele cu o sărbătoare legală sau cu zilele 24 decembrie – 2 ianuarie', nohol: 'Ferestrele fără sărbători' })[w] + `: ${n}${de(n)} origini × 14 orizonturi.`,
 cr2: (k, L) => `${k} (MW), în ordine: ` + L.map(([m, v]) => `${mn(m)} ${v}`).join(', ') + '.',
 cr3: L => (L.length ? 'MASE sub 1 (MAE / 309 MW): ' + L.map(([m, v]) => `${mn(m)} ${v}`).join(', ') + '.' : 'Nicio metodă nu are MASE sub 1.'),
 cr4: L => 'DM (HLN, erori pătratice medii pe fereastră) față de metoda sezonieră naivă: ' + L.map(([m, v, p]) => `${mn(m)} ${v} (${p})`).join(', ') + '.',
 cr5: (v, p) => `Combinarea față de DHR: HLN ${v} (${p}).`,
 cb1: (a, b, m1, m0, mh, r1, r0, rh) => `MAE: doar ${mn(a)} (w = 1) ${m1} MW, doar ${mn(b)} (w = 0) ${m0} MW, ponderi egale ${mh} MW; RMSE: ${r1}, ${r0} și ${rh} MW.`,
 cb2: (w, m) => `Cea mai mică MAE, aleasă retrospectiv: w = ${w}, ${m} MW.`,
 cb3: (r, a, b, w) => `Corelația erorilor ρ = ${r}; σ₁ = ${a}, σ₂ = ${b} MW; ponderea optimă Bates–Granger w* = ${w}.`
};
})();
""",
 'en': r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const Q = s => s, dmy = d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`, j = a => a.join(', ');
const mn = x => (x === 'Seasonal naive' ? 'seasonal naive' : x === 'Combination' ? 'combination' : x);
window.CH4_TEXT = {
 mon: MON, q: Q, dmy, dec: y => `December ${y}`, lect: ' (lecture)', bestBic: ' · lowest BIC', bestAicc: ' · lowest AICc', dmyEn: dmy,
 selYear: y => `selected year: ${y}`,
 se_g: (a, b, m, p) => `GDP, ${a}–${b}: quarter means ${j(m)} bn EUR (2010 prices); the Q4 mean is ${p}% above the Q1 mean.`,
 se_t: (a, b, r, mx, mn0, au, ja) => `Tourism nights, ${a}–${b}: August ${au}, January ${ja} million on average, ${r} times as many in August; month with the most nights: ${mx}, with the fewest: ${mn0}.`,
 se_i: (a, b, m1, v1, m2, v2, m0, v0) => `HICP inflation, ${a}–${b}: highest monthly mean in ${m1} (${v1}%), then in ${m2} (${v2}%); lowest in ${m0} (${v0}%).`,
 sp1: (y, au, ja, r, ap, mx, mn0) => `${y}: ${au} million nights in August, ${ja} in January (ratio ${r}); April ${ap}. Peak month: ${mx}; minimum: ${mn0}.`,
 sp2: (m, k, n) => `GDP, 2005–2026: quarter means ${j(m)} bn EUR; the order Q1 < Q2 < Q3 < Q4 holds in ${k} of ${n} years (2005–2025).`,
 uc1: (s, nr, np) => `1 − z<sup>${s}</sup> has ${s} roots on the unit circle: ${nr === 2 ? 'two real ones, z = 1 and z = −1' : 'one real one, z = 1 (z = −1 is a root only for even s)'}, and ${np} complex-conjugate ${np === 1 ? 'pair' : 'pairs'}.`,
 uc2: (s, L) => `Frequencies k/${s} and periods ${s}/k: ` + L.map(([k, p]) => `k = ${k}: period ${Math.round(p * 100) / 100}`).join('; ') + '; k = 0: frequency 0 (the trend).',
 df0: (q, n) => `Sample 2000Q1 – ${q}: ${n} quarters.`,
 df1: (k, n, sd, r1, r4, r8, b) => ['Δ ln Y', 'Δ₄ ln Y', 'ΔΔ₄ ln Y'][k] + `: T = ${n}, standard deviation ${sd}%; ACF ${r1} at lag 1, ${r4} at lag 4, ${r8} at lag 8 (band ±${b}).`,
 th1: (P, a, b, c, p) => `SAR(1)<sub>12</sub>, Φ = ${P}: ACF ${a}, ${b}, ${c} at lags 12, 24, 36; PACF ${p} at lag 12 and 0 elsewhere.`,
 th2: (t, a, p1, p2, p3) => `SMA(1)<sub>12</sub>, Θ = ${t}: ACF ${a} at lag 12 and 0 elsewhere; PACF ${p1}, ${p2}, ${p3} at lags 12, 24, 36.`,
 th3: (a, b, r1, r12, r11, r13) => `Airline, θ = ${a}, Θ = ${b}: ρ₁ = ${r1}, ρ₁₂ = ${r12}; the satellites ρ₁₁ = ${r11} and ρ₁₃ = ${r13} are the product ρ₁ρ₁₂.`,
 ai1: (nm, n, b) => `${nm}: ${n} observations, band ±${b}.`,
 ai2: (a, b, c, d, l) => `ACF: ${a} at lag 1, ${b} at lag 3, ${c} at lag 12, ${d} at lag 24; lags outside the band: ${l}.`,
 ai3: (a, b, c, d, e, l) => `PACF: ${a}, ${b}, ${c} at lags 1, 2, 3; ${d} at lag 12, ${e} at lag 24; lags outside the band: ${l}.`,
 af0: (y, l, h) => `Estimated on 1949–${y}, ${h} forecasts for January ${y + 1} – December 1960` + (l ? ' (lecture).' : ' (an extra run of the chapter code).'),
 af1: (t, ts, T0, Ts, s) => `θ̂ = ${t} (${ts}), Θ̂ = ${T0} (${Ts}); residual standard deviation ${s}% a month.`,
 af2: (a, n) => `MAPE: ${a}% for the airline model, ${n}% for the seasonal naive method.`,
 af3: (f, lo, hi, a, ok) => `December 1960: forecast ${f}, 95% interval [${lo}, ${hi}], actual value ${a}: ${ok ? 'inside the interval' : 'outside the interval'}.`,
 af4: (a, b, h) => `Width of the 95% interval: ${a} in the first month, ${b} in month ${h}.`,
 dg1: (m, k, a, b, s, ib, ia) => `SARIMA${m.replace(/\)4$/, ')₄')}: ${k} ARMA ${k === 1 ? 'parameter' : 'parameters'}, AICc ${a}, BIC ${b}` + (ib ? ' (lowest BIC)' : '') + (ia ? ' (lowest AICc)' : '') + `; σ̂ = ${s}%.`,
 dg2: p => `Estimates (standard errors): ${p}.`,
 dg3: (q, df, p8, p12) => `Ljung–Box: Q*(8) = ${q} with ${df} degrees of freedom, ${p8}; Q*(12): ${p12}.`,
 dg4: (jb, p, q, v, z) => `Jarque–Bera ${jb} (${p}); largest residual: ${v}% in ${q} (${z} standard deviations).`,
 gf0: (t, w, l, s) => `Θ = ${t}` + (l ? ' (the estimate of the lecture)' : ' (fixed; σ rescaled by the residuals)') + `: weight of the most recent year in the forecast pattern 1 + Θ = ${w}; σ = ${s}%.`,
 gf1: (q, f) => `Forecasts for ${q} and the next three quarters: ${j(f)} bn EUR.`,
 gf2: (a, b, c, d) => `95% interval: [${a}, ${b}] for the first quarter, [${c}, ${d}] for the eighth.`,
 gf3: g => `Implied year-on-year growth: ${j(g)}%.`,
 sa1: (q, a, b) => `${q}, on the previous quarter: NSA ${a}%, SCA ${b}% (log points).`,
 sa2: (a, b) => `On the same quarter of the previous year: NSA ${a}%, SCA ${b}%.`,
 sa3: f => `Seasonal and calendar factor NSA/SCA: ${f}.`,
 sa4: (y, a, b) => `Total of ${y}: NSA ${a}, SCA ${b} bn EUR.`,
 fo1: (K, n, r) => `K = ${K}: ${n} coefficients, R² = ${r}.`,
 fo2: a => `Amplitudes of the harmonics k = 1, 2, …: ${j(a)} (log).`,
 fo3: (a, b) => `sin(2π6t/12) is zero in every month and is left out: with cos(2π6t/12), the 11 coefficients pass through all 12 points (R² = ${b}); K = 5, with 10 coefficients, gives R² = ${a}.`,
 ea1: (w, l, s, n) => `w = ${w} days` + (l ? ' (lecture)' : ' (an extra run of the chapter code)') + `: slope of the OLS line ${s} percentage points, ${n} months (March–May, without 2020–2021).`,
 ea2: (b, se, t, w, ws) => `Food, regression with SARIMA(0,1,1)(0,1,1)<sub>12</sub> errors: Easter ${b}% (standard error ${se}, t = ${t}); one more working day ${w}% (${ws}).`,
 ea3: (b, se, t, w, ws) => `Total retail: Easter ${b}% (${se}, t = ${t}); working days ${w}% (${ws}).`,
 ea4: y => `Easter in May: ${y}.`,
 lh1: (a, b, m, lo, hi) => `${a} – ${b}: mean ${m} GW, minimum ${lo}, maximum ${hi} GW.`,
 lh2: (y, n, a, b, c, d) => `ACF of ${n} hours of ${y}: ${a} at lag 12, ${b} at lag 24, ${c} at lag 84, ${d} at lag 168.`,
 lh3: (y, p, v, t, w) => `Daily profile in ${y}: maximum at ${p}:00 (${v} GW), minimum at ${t}:00 (${w} GW).`,
 ms1: (a, b, l) => `Window ${a} – ${b}` + (l ? ' (lecture).' : ' (an extra run of the chapter code).'),
 ms2: (a, b, c) => `Of the variance around the trend: ${a}% the daily component, ${b}% the weekly component, ${c}% the remainder.`,
 ms3: (a, b) => `Amplitude in the first week: ${a} GW for the daily component, ${b} GW for the weekly one.`,
 ld1: (p, n) => (p === 'all' ? 'Whole sample, 2022–2026' : `Year ${p}`) + `: ${n} days.`,
 ld2: (a, b) => `Without holidays: Saturday ${a}%, Sunday ${b}% relative to Wednesday.`,
 ld3: a => `January relative to May: +${a}%.`,
 ld4: (L, m, e) => 'Lowest days: ' + L.map(([d, h, v]) => `${d}${h ? ' (' + h + ')' : ''}, ${v} GW`).join('; ') + `; mean ${m} GW` + (e ? `; Easter days (Friday to Monday) ${e} GW.` : '.'),
 pr1: (K, h, l) => `Prophet with K = ${K} yearly harmonics, ${h ? 'with' : 'without'} the public holidays` + (l ? ' (lecture).' : ' (an extra run of the chapter code).'),
 pr2: (a, b, n) => `Trend from ${a} to ${b} GW; changepoints used: ${n} of 25.`,
 pr3: (a, b, c) => `Weekly: Wednesday ${a}, Saturday ${b}, Sunday ${c} GW.`,
 pr4: (a, da, b, db) => `Yearly (2024): maximum ${a} GW on ${da}, minimum ${b} GW on ${db}.`,
 pr5: L => 'Largest holiday effects: ' + L.map(([h, v]) => `${h} ${v}`).join(', ') + ' GW.',
 pr6: a => `In-sample mean absolute error: ${a} MW.`,
 cs1: (h, s, o, c) => `h = ${h}, step ${s}: ` + (o ? `consecutive windows overlap by ${h - s} ${h - s === 1 ? 'period' : 'periods'}; a date appears in up to ${c} windows, and the errors are correlated up to lag ${h - 1}.` : 'the windows do not overlap; every date appears in one window only.'),
 cs2: () => 'In the lecture: 26 origins 14 days apart, h = 14, expanding window; the windows do not overlap.',
 lf1: (o, a, b, h, l) => `Origin ${o}` + (l ? ' (lecture)' : '') + `; window ${a} – ${b}` + (h.length ? '; holidays: ' + h.join(', ') + '.' : '; no holidays.'),
 lf2: L => 'MAE in the window (MW): ' + L.map(([m, v]) => `${mn(m)} ${v}`).join(', ') + '.',
 cr1: (w, n) => ({ all: 'All windows', hol: 'Windows with a public holiday or the days 24 December – 2 January', nohol: 'Windows without holidays' })[w] + `: ${n} origins × 14 horizons.`,
 cr2: (k, L) => `${k} (MW), in order: ` + L.map(([m, v]) => `${mn(m)} ${v}`).join(', ') + '.',
 cr3: L => (L.length ? 'MASE below 1 (MAE / 309 MW): ' + L.map(([m, v]) => `${mn(m)} ${v}`).join(', ') + '.' : 'No method has a MASE below 1.'),
 cr4: L => 'DM (HLN, mean squared errors per window) against the seasonal naive method: ' + L.map(([m, v, p]) => `${mn(m)} ${v} (${p})`).join(', ') + '.',
 cr5: (v, p) => `The combination against DHR: HLN ${v} (${p}).`,
 cb1: (a, b, m1, m0, mh, r1, r0, rh) => `MAE: ${mn(a)} only (w = 1) ${m1} MW, ${mn(b)} only (w = 0) ${m0} MW, equal weights ${mh} MW; RMSE: ${r1}, ${r0} and ${rh} MW.`,
 cb2: (w, m) => `Lowest MAE, chosen in hindsight: w = ${w}, ${m} MW.`,
 cb3: (r, a, b, w) => `Correlation of the errors ρ = ${r}; σ₁ = ${a}, σ₂ = ${b} MW; Bates–Granger optimal weight w* = ${w}.`
};
})();
""",
}


# ================================================================= result tables and worked examples (blocks of their own frames)
# (anchor, frame title RO, frame title EN, section key, host chart block on the web lecture)
EXTRA = [('wx-sdiff', 'Diferența sezonieră', 'The seasonal difference', 'roots', 'unit-circle'),
         ('wx-roots', 'Interpretarea rădăcinilor sezoniere', 'Interpreting the seasonal roots', 'roots', 'unit-circle'),
         ('hegy-table', 'Testul HEGY', 'The HEGY test', 'roots', 'gdp-diff'),
         ('sur-table', 'Teste de rădăcină unitară sezonieră pe date din România', 'Seasonal unit-root tests on Romanian data', 'roots', 'gdp-diff'),
         ('wx-air', 'Exemplu rezolvat: modelul airline', 'Worked example: the airline model', 'sarima', 'theory-acf'),
         ('wx-fc', 'Prognoza cu modele SARIMA', 'Forecasting with SARIMA', 'sarima', 'theory-acf'),
         ('air-table', 'Interpretarea prognozelor airline', 'Interpreting the airline forecasts', 'air', 'airline-fc'),
         ('gdpm-table', 'Modele SARIMA pentru PIB-ul României', 'SARIMA models for Romanian GDP', 'gdp', 'gdp-diag'),
         ('gdpfc-table', 'Interpretarea prognozelor PIB', 'Interpreting the GDP forecasts', 'gdp', 'gdp-fc'),
         ('sa-table', 'Interpretarea seriilor ajustate', 'Interpreting the adjusted series', 'sa', 'sa-nsa'),
         ('wx-fourier', 'Variabile dummy sezoniere și termeni Fourier', 'Seasonal dummies and Fourier terms', 'cal', 'fourier'),
         ('fourier-table', 'Interpretarea ajustărilor Fourier', 'Interpreting the Fourier fits', 'cal', 'fourier'),
         ('wx-easter', 'Efecte de calendar și Paștele ortodox', 'Calendar effects and Orthodox Easter', 'cal', 'easter'),
         ('easter-table', 'Interpretarea efectului Paștelui', 'Interpreting the Easter effect', 'cal', 'easter'),
         ('dhr-table', 'O regresie armonică dinamică pentru consumul zilnic', 'A dynamic harmonic regression for daily load', 'mult', 'load-daily'),
         ('dm-h', 'Testul Diebold–Mariano', 'The Diebold–Mariano test', 'ev', 'cv-scheme'),
         ('wx-dm', 'Exemplu rezolvat: un test Diebold–Mariano', 'Worked example: a Diebold–Mariano test', 'ev', 'cv-scheme'),
         ('cv-table', 'Interpretarea rezultatelor validării încrucișate', 'Interpreting the cross-validation results', 'ev', 'cv-results'),
         ('gdpcv-table', 'Un al doilea test: PIB-ul României', 'A second test: Romanian GDP', 'ev', 'cv-results'),
         ('wx-comb', 'Avantajul combinării prognozelor', 'Why combining forecasts works', 'comb', 'combination'),
         ('comb-table', 'Interpretarea combinării', 'Interpreting the combination', 'comb', 'combination')]
PFX.update({'wx-sdiff': 'xs', 'wx-roots': 'xr', 'hegy-table': 'hg', 'sur-table': 'su', 'wx-air': 'xa', 'wx-fc': 'xf', 'air-table': 'at', 'gdpm-table': 'gm',
            'gdpfc-table': 'gt', 'sa-table': 'st', 'wx-fourier': 'xo', 'fourier-table': 'fk', 'wx-easter': 'xe', 'easter-table': 'et', 'dhr-table': 'dr',
            'dm-h': 'dz', 'wx-dm': 'xd', 'cv-table': 'cv', 'gdpcv-table': 'gc', 'wx-comb': 'xc', 'comb-table': 'cw'})

CSS = ('<style>.c4x{margin:.6rem 0 .3rem;padding:.55rem .8rem;border:1px solid #D5DEEC;border-left:4px solid #17A2B8;border-radius:6px;background:#F7FAFD;min-width:0}'
       '.c4x h4{margin:0 0 .2rem;font-size:.95rem;color:#1A3A6E}.c4x .c4f{overflow-x:auto;overflow-y:hidden;max-width:100%;font-size:.92rem}'
       '.c4x ol{margin:.3rem 0 0 1.25rem;padding:0;font-size:.92rem}.c4x li{margin:.18rem 0;overflow-wrap:anywhere}'
       '.c4t{max-width:100%;overflow-x:auto;margin:.55rem 0 .3rem}.c4t table{border-collapse:collapse;font-size:.86rem;margin:0;border-top:2px solid #1A3A6E;border-bottom:2px solid #1A3A6E}'
       '.c4t th,.c4t td{padding:.24rem .55rem;text-align:left;white-space:nowrap;vertical-align:top}.c4t thead tr:last-child th{border-bottom:1px solid #1A3A6E}'
       '.c4t tbody tr:nth-child(even){background:#F5F7FB}.c4t tbody tr.hi{background:#FDF0E1}.c4t .r{text-align:right}.c4cap{font-size:.84rem;margin:.15rem 0 .3rem;color:#1F2A44;white-space:normal}.c4tc .ctl{max-width:360px;margin:.15rem 0 .2rem}.c4tc{margin:.55rem 0 0}</style>')


def tbl(i):
    return f'<div class="tbl c4t" id="{i}"></div>'


def wx(i, title, formula):
    return f'<div class="c4x"><h4>{title}</h4><div class="c4f">\\[{formula}\\]</div><ol id="{i}"></ol></div>'


def widgets_all(lang):
    """Widgets of every block of the interactive page: the chart blocks and EXTRA (worked examples and live tables)."""
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    W = s('Exemplu rezolvat', 'Worked example')
    d = (lambda x: x.replace('365.25', '365{,}25')) if ro else (lambda x: x)
    mn = lambda m: s({'Seasonal naive': 'Sezonieră naivă', 'Combination': 'Combinarea'}.get(m, m), m)   # noqa: E731
    M7 = CVM + ['Combination']
    meth = lambda i, lab, opts, v: select(i, lab, [(m, mn(m)) for m in opts], v)   # noqa: E731
    lossS = lambda i: select(i, s('Pierderea', 'Loss'), [('se', s('erori pătratice (curs)', 'squared errors (lecture)')), ('ae', s('erori absolute', 'absolute errors'))], 'se')   # noqa: E731
    sub4 = lambda m: m.replace(')4', ')₄')   # noqa: E731
    GM = ['(0,1,1)(0,1,1)4', '(0,1,0)(0,1,1)4', '(1,1,0)(0,1,1)4', '(1,1,1)(0,1,1)4', '(0,1,2)(0,1,1)4', '(0,1,1)(1,1,0)4', '(0,1,1)(1,1,1)4', '(0,1,1)(0,1,0)4',
          '(2,1,0)(0,1,1)4', '(1,1,0)(1,1,0)4']
    w = _chart_widgets(lang)
    ctl, ch = w['series']
    w['series'] = (ctl, CSS + ch)
    w.update({
     'wx-sdiff': (slider('xs-q', s('Trimestrul t', 'Quarter t'), 5, 105, 1, 105) + play('xs'),
                  wx('xs-ws', W, r'y_t=100\ln Y_t,\qquad \Delta_4y_t=(1-L^4)y_t=y_t-y_{t-4},\qquad \Delta\Delta_4y_t=(1-L)(1-L^4)y_t=y_t-y_{t-1}-y_{t-4}+y_{t-5}') + tbl('xs-tb')),
     'wx-roots': (select('xr-s', s('Perioada s', 'Period s'), [(v, v) for v in ('2', '3', '4', '6', '7', '12', '24')], '4') + slider('xr-k', s('Rădăcina k', 'Root k'), 0, 3, 1, 1) + play('xr'),
                  wx('xr-ws', W, r'1-z^s=0\iff z_k=e^{2\pi ik/s}=\cos\frac{2\pi k}{s}+i\sin\frac{2\pi k}{s},\ \ k=0,1,\dots,s-1;\qquad (1-z_kz)(1-\bar z_kz)=1-2\cos\frac{2\pi k}{s}\,z+z^2') + tbl('xr-tb')),
     'hegy-table': (slider('hg-end', s('Eșantionul se termină în', 'Sample ends in'), 0, 18, 1, 11)
                    + select('hg-k', s('Laguri ale lui Δ₄y', 'Lags of Δ₄y'), [('AIC', s('alese după AIC (curs)', 'chosen by AIC (lecture)'))] + [(str(k), str(k)) for k in range(5)], 'AIC')
                    + select('hg-det', s('Termenii determiniști', 'Deterministic terms'), [('t', s('constantă, dummy sezoniere și trend (curs)', 'constant, seasonal dummies and trend (lecture)')),
                                                                                           ('c', s('constantă și dummy sezoniere', 'constant and seasonal dummies'))], 't') + play('hg'),
                    wx('hg-ws', W, r'\begin{gathered}y_{1t}=(1+L+L^2+L^3)y_t,\quad y_{2t}=-(1-L+L^2-L^3)y_t,\quad y_{3t}=-(1-L^2)y_t\\ \Delta_4y_t=\pi_1y_{1,t-1}+\pi_2y_{2,t-1}+\pi_3y_{3,t-2}+\pi_4y_{3,t-1}+\text{'
                       + s('termeni determiniști', 'deterministic terms') + r'}+\textstyle\sum_{j=1}^{k}a_j\Delta_4y_{t-j}+\varepsilon_t\end{gathered}')
                    + tbl('hg-tb') + tbl('hg-tb2') + charts('hg-chart')),
     'sur-table': (slider('su-end', s('Eșantionul 2005 – sfîrșitul anului', 'Sample 2005 – end of'), 2012, 2025, 1, 2019) + play('su'), tbl('su-tb')),
     'wx-air': (slider('xa-th', 'θ', -0.9, 0.9, 0.05, -0.4) + slider('xa-Th', 'Θ', -0.95, 0.95, 0.05, -0.6) + play('xa'),
                wx('xa-ws', W, r'(1-L)(1-L^{12})y_t=(1+\theta L)(1+\Theta L^{12})\varepsilon_t,\quad \gamma(0)=\sigma^2(1+\theta^2)(1+\Theta^2),\quad \rho_1=\frac{\theta}{1+\theta^2},\ \rho_{12}=\frac{\Theta}{1+\Theta^2},\ \rho_{11}=\rho_{13}=\rho_1\rho_{12}')
                + tbl('xa-tb')),
     'wx-fc': (select('xf-o', s('Originea prognozei', 'Forecast origin'), [(str(y), s(f'decembrie {y}', f'December {y}') + s(' (curs)', ' (lecture)') * (y == 1958)) for y in range(1955, 1959)], '1958')
               + slider('xf-h', s('Orizontul h (luni)', 'Horizon h (months)'), 1, 24, 1, 1) + play('xf'),
               wx('xf-ws', W, r'\hat y_{T+h}=\hat y_{T+h-1}+\hat y_{T+h-12}-\hat y_{T+h-13}+\hat\theta\hat\varepsilon_{T+h-1}+\hat\Theta\hat\varepsilon_{T+h-12}+\hat\theta\hat\Theta\hat\varepsilon_{T+h-13},\qquad \hat\varepsilon_{T+j}=0\ (j\ge1)')
               + tbl('xf-tb')),
     'air-table': (slider('at-o', s('Originea prognozei: decembrie', 'Forecast origin: December'), 1952, 1958, 1, 1958) + play('at'), tbl('at-tb') + charts('at-chart')),
     'gdpm-table': (select('gm-m', s('Modelul detaliat (rîndul evidențiat)', 'Model worked out (highlighted row)'), [(m, sub4(m)) for m in GM], '(0,1,0)(0,1,1)4')
                    + slider('gm-L', s('Laguri Ljung–Box m', 'Ljung–Box lags m'), 5, 16, 1, 8) + play('gm'),
                    tbl('gm-tb') + wx('gm-ws', W, r'\mathrm{AIC}=-2\ln L+2k,\qquad \mathrm{AICc}=\mathrm{AIC}+\frac{2k(k+1)}{n-k-1},\qquad \mathrm{BIC}=-2\ln L+k\ln n') + tbl('gm-tb2') + charts('gm-chart')),
     'gdpfc-table': (slider('gt-Th', s('Θ al MA sezonier (fixat)', 'Θ of the seasonal MA (fixed)'), -0.99, 0, 0.0001, -0.5765)
                     + select('gt-lv', s('Nivelul intervalului', 'Interval level'), [('0.8', '80%'), ('0.9', '90%'), ('0.95', '95%'), ('0.99', '99%')], '0.95') + play('gt'), tbl('gt-tb')),
     'sa-table': (slider('st-y0', s('Factorii de la anul', 'Factors from'), 2010, 2025, 1, 2015) + slider('st-y1', s('pînă în anul', 'to'), 2010, 2026, 1, 2019)
                  + slider('st-q', s('Trimestrul pentru creșterea trimestrială', 'Quarter of the quarter-on-quarter growth'), 41, 105, 1, 104) + play('st'),
                  wx('st-ws', W, r'100(\ln Y_t-\ln Y_{t-1})\ \ \text{(NSA)},\qquad 100(\ln Y^{SCA}_t-\ln Y^{SCA}_{t-1})\ \ \text{(SCA)},\qquad \frac{Y_t}{Y^{SCA}_t}=\hat S_t\hat C_t')
                  + tbl('st-tb') + tbl('st-tb2')),
     'wx-fourier': (select('xo-c', s('Seria', 'Series'), [('tour', s('Înnoptări turistice, m = 12', 'Tourism nights, m = 12')), ('dhr', s('Consumul zilnic (DHR), m = 365,25', 'Daily load (DHR), m = 365.25'))], 'tour')
                    + slider('xo-k', s('Armonica k', 'Harmonic k'), 1, 3, 1, 1) + slider('xo-t', s('Momentul t', 'Time t'), 1, 12, 1, 8) + play('xo'),
                    wx('xo-ws', W, d(r'\sum_{k=1}^{K}\Big[\alpha_k\sin\frac{2\pi kt}{m}+\beta_k\cos\frac{2\pi kt}{m}\Big],\qquad \text{' + s('amplitudinea', 'amplitude') + r'}_k=\sqrt{\alpha_k^2+\beta_k^2}')) + tbl('xo-tb')),
     'fourier-table': (slider('fk-K', s('K evidențiat', 'Highlighted K'), 1, 6, 1, 3) + play('fk'), tbl('fk-tb') + charts('fk-chart')),
     'wx-easter': (slider('xe-y', s('Anul', 'Year'), 2000, 2030, 1, 2025) + slider('xe-w', s('Fereastra w (zile)', 'Window w (days)'), 1, 20, 1, 10) + play('xe'),
                   wx('xe-ws', W, r'E_t=\frac{1}{w}\,\#\{d\in[\text{' + s('Paște', 'Easter') + r'}-w,\ \text{' + s('Paște', 'Easter') + r'}-1]:\ d\ \text{' + s('în luna', 'in month') + r'}\ t\}') + tbl('xe-tb')),
     'easter-table': (slider('et-w', s('Fereastra w (zile)', 'Window w (days)'), 1, 20, 1, 10) + play('et'), tbl('et-tb') + charts('et-chart')),
     'dhr-table': (slider('dr-d', s('Ziua din 2025', 'Day of 2025'), 0, 364, 1, 0) + slider('dr-K', s('Armonici anuale K', 'Annual harmonics K'), 1, 4, 1, 4) + play('dr'),
                   tbl('dr-tb') + wx('dr-ws', W, d(r'y_t=\sum_{k=1}^{4}\Big[\alpha_k\sin\frac{2\pi kt}{365.25}+\beta_k\cos\frac{2\pi kt}{365.25}\Big]+\sum_j\delta_jD^{\mathrm{day}}_{jt}+\lambda_1H_t+\lambda_2E_t+\lambda_3X_t+u_t'))
                   + charts('dr-chart')),
     'dm-h': (meth('dz-1', s('Metoda 1', 'Method 1'), M7, 'DHR') + meth('dz-2', s('Metoda 2', 'Method 2'), M7, 'Seasonal naive') + lossS('dz-l') + play('dz'),
              charts('dz-chart') + tbl('dz-tb')),
     'wx-dm': (select('xd-src', s('Datele', 'Data'), [('slide', s('cele cinci ferestre din slide', 'the five windows of the slide')), ('real', s('validarea încrucișată, 26 de ferestre', 'cross-validation, 26 windows'))], 'slide')
               + meth('xd-1', s('Metoda 1', 'Method 1'), M7, 'DHR') + meth('xd-2', s('Metoda 2', 'Method 2'), M7, 'Seasonal naive') + lossS('xd-l') + play('xd'),
               wx('xd-ws', W, r'\bar d=\frac1n\sum_j d_j,\quad \hat\gamma_d(0)=\frac1n\sum_j(d_j-\bar d)^2,\quad \mathrm{DM}=\frac{\bar d}{\sqrt{\hat\gamma_d(0)/n}},\quad \mathrm{HLN}=\sqrt{\frac{n+1-2h+h(h-1)/n}{n}}\;\mathrm{DM}')
               + tbl('xd-tb')),
     'cv-table': (select('cv-w', s('Ferestrele', 'Windows'), [('all', s('toate cele 26 (curs)', 'all 26 (lecture)')), ('hol', s('cu sărbători', 'with holidays')), ('nohol', s('fără sărbători', 'without holidays'))], 'all')
                  + select('cv-h', s('Orizonturile', 'Horizons'), [('all', '1–14'), ('1-7', '1–7'), ('8-14', '8–14')], 'all') + lossS('cv-l')
                  + meth('cv-r', s('A doua coloană DM', 'Second DM column'), M7, 'DHR') + meth('cv-m', s('Metoda pentru MASE (rîndul evidențiat)', 'Method for MASE (highlighted row)'), M7, 'DHR') + play('cv'),
                  tbl('cv-tb') + wx('cv-ws', W, r'\mathrm{MASE}=\frac{\mathrm{MAE}}{\frac{1}{T-7}\sum_{t=8}^{T}|y_t-y_{t-7}|}')),
     'gdpcv-table': (select('gc-s', s('Originile', 'Origins'), [('all', s('toate cele 51', 'all 51')), ('ex', s('fără 2019–2020 (43)', 'without 2019–2020 (43)'))], 'all')
                     + select('gc-m', s('Măsura', 'Measure'), [('MAE', 'MAE'), ('RMSE', 'RMSE')], 'MAE') + play('gc'), tbl('gc-tb') + '<div class="c4tc">' + lossS('gc-l') + '</div>' + tbl('gc-tb2') + charts('gc-chart')),
     'wx-comb': (select('xc-src', s('Datele', 'Data'), [('slide', s('exemplul din slide', 'the slide example')), ('real', s('erorile validării încrucișate', 'the cross-validation errors'))], 'slide')
                 + meth('xc-1', s('Prognoza 1 (ponderea w)', 'Forecast 1 (weight w)'), CVM, 'DHR') + meth('xc-2', s('Prognoza 2 (ponderea 1 − w)', 'Forecast 2 (weight 1 − w)'), CVM, 'Prophet')
                 + slider('xc-w', s('Ponderea w', 'Weight w'), 0, 1, 0.01, 0.5) + play('xc'),
                 wx('xc-ws', W, r'\mathrm{MSE}(w)=w^2\sigma_1^2+(1-w)^2\sigma_2^2+2w(1-w)\rho\sigma_1\sigma_2,\qquad w^*=\frac{\sigma_2^2-\rho\sigma_1\sigma_2}{\sigma_1^2+\sigma_2^2-2\rho\sigma_1\sigma_2}')),
     'comb-table': (select('cw-s', s('Componentele', 'Members'), [('five', s('cele cinci modele (curs)', 'the five models (lecture)')), ('dp', 'DHR, Prophet'), ('es', 'ETS, SARIMA'), ('et', 'ETS, TBATS')], 'five')
                    + play('cw'), tbl('cw-tb')),
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
    """Chart blocks for build_web_lecture.py: each host carries the web version of the EXTRA blocks of its section (moved by ch4.js)."""
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
        img = f'ch4x_{anchor}_{lang}.jpg'
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
 'ro': r"""Object.assign(window.CH4_TEXT, { x: {
 mn: m => ({ 'Seasonal naive': 'sezonieră naivă', Combination: 'combinarea' })[m] || m,
 monName: ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'],
 wdName: ['luni', 'marți', 'miercuri', 'joi', 'vineri', 'sîmbătă', 'duminică'],
 hol: h => ({ 'New Year': 'Anul Nou', 'Union Day': 'Ziua Unirii', Easter: 'Paștele', 'Labour Day': 'Ziua Muncii', "Children's Day": 'Ziua Copilului', Pentecost: 'Rusaliile',
   Assumption: 'Adormirea Maicii Domnului', "St Andrew's Day": 'Sfîntul Andrei', 'National Day': 'Ziua Națională', Christmas: 'Crăciunul', Epiphany: 'Boboteaza și Sfîntul Ioan' })[h] || h,
 quarter: 'Trimestrul', quarter0: 'Trimestrul', yBn: 'Y<sub>t</sub> (mld. EUR)', coef: 'Coeficient', est: 'Estimația', test: 'Testul', stat: 'statistica', cv5: 'valoarea critică de 5%', decision: 'Decizia la 5%',
 rej: 'respinsă', norej: 'nerespinsă', root: 'rădăcina', roots: 'rădăcinile', lag: 'Lagul', model: 'Modelul', method: 'Metoda', sample: 'Eșantionul de estimare', year: 'Anul', month: 'Luna', month0: 'Luna',
 actual: 'Valoarea reală', origin: 'Originea', series0: 'Seria', ser0: 'Seria (logaritmi, din 2005)', food: 'Comerț cu alimente', total: 'Tot comerțul cu amănuntul', scheme: 'Ponderile', annual: 'pereche Fourier anuală',
 // diferența sezonieră
 s0: (q, d1, d4, dd) => `${q}: Δy = ${d1} (oscilația sezonieră față de trimestrul anterior), Δ<sub>4</sub>y = ${d4} (creșterea față de același trimestru al anului anterior, în puncte logaritmice), ΔΔ<sub>4</sub>y = ${dd}.`,
 s1: (q, Y, y, q4, Y4, y4) => `Datele: Y = ${Y} mld. EUR în ${q}, deci y<sub>t</sub> = 100 ln Y<sub>t</sub> = ${y}; Y = ${Y4} mld. EUR în ${q4}, y<sub>t−4</sub> = ${y4}.`,
 s2: (y, y4, d, q, q4) => `Δ<sub>4</sub>y<sub>t</sub> = y<sub>t</sub> − y<sub>t−4</sub> = ${y} − ${y4} = ${d}: PIB-ul din ${q} față de ${q4}, în puncte logaritmice (≈ %).`,
 s3: (y, y1, d) => `Δy<sub>t</sub> = y<sub>t</sub> − y<sub>t−1</sub> = ${y} − ${y1} = ${d}: în cea mai mare parte tiparul sezonier, nu creștere.`,
 s4: (y, y1, y4, y5, d) => `ΔΔ<sub>4</sub>y<sub>t</sub> = y<sub>t</sub> − y<sub>t−1</sub> − y<sub>t−4</sub> + y<sub>t−5</sub> = ${y} − ${y1} − ${y4} + ${y5} = ${d}.`,
 s5: (a, b, c, d, e, f) => `Ordinea nu contează: Δ(Δ<sub>4</sub>y)<sub>t</sub> = ${a} − ${b} = ${c} și Δ<sub>4</sub>(Δy)<sub>t</sub> = ${d} − ${e} = ${f}.`,
 // rădăcinile sezoniere
 unit: { 4: 'trimestre', 12: 'luni', 7: 'zile', 24: 'ore' }, unit0: 'perioade', freq: 'Frecvența k/s (cicluri pe perioadă)', period: u => `Perioada s/k (${u})`, factor: 'Factorul real al lui 1 − z<sup>s</sup>', same: k => `perechea lui k = ${k}`,
 r0: (s, nr, np) => `1 − z<sup>${s}</sup> are ${s}${s >= 20 ? ' de' : ''} rădăcini pe cercul unitate: ${nr === 2 ? 'două reale, 1 și −1' : 'una reală, 1'}, și ${np} ${np === 1 ? 'pereche' : 'perechi'} complex conjugate.`,
 r1: (k, s, a, z) => `z<sub>${k}</sub> = cos(2π·${k}/${s}) + i sin(2π·${k}/${s}), cu unghiul 2π·${k}/${s} = ${a} rad: z<sub>${k}</sub> = ${z}.`,
 r2: (s, k, m) => `z<sub>${k}</sub><sup>${s}</sup> = cos(2π·${k}) + i sin(2π·${k}) = 1, deci 1 − z<sub>${k}</sub><sup>${s}</sup> = 0; modulul |z<sub>${k}</sub>| = ${m}: rădăcina se află pe cercul unitate.`,
 r3: (k, s, f, p, u) => `Frecvența ${k}/${s} = ${f} cicluri pe perioadă, perioada ${s}/${k} = ${p} ${u}.`,
 r3a: s => `k = 0: z = 1, frecvența 0: rădăcina diferenței obișnuite (trendul), prezentă pentru orice s = ${s}.`,
 r4: (k2, z, f) => `Conjugata z<sub>${k2}</sub> = ${z} are aceeași perioadă; împreună dau factorul real ${f}.`,
 r4a: (f, z) => `O rădăcină reală (z = ${z}) dă factorul real ${f}.`,
 r5: (s, f) => `1 − z<sup>${s}</sup> = ${f}: cîte un factor pentru fiecare frecvență; un test de rădăcină unitară sezonieră verifică fiecare factor separat.`,
 // HEGY
 h0: (q, n, k, t1, t2, F, nr) => `Eșantionul T1 2000 – ${q}, n = ${n}, ${k} ${k === 1 ? 'lag' : 'laguri'} ale lui Δ<sub>4</sub>y: t(π₁) = ${t1}, t(π₂) = ${t2}, F(π₃, π₄) = ${F}; ` + (nr ? `${nr} din cele 3 ipoteze nule sînt respinse la 5%.` : 'niciuna dintre cele trei ipoteze nule nu este respinsă la 5%: Δ<sub>4</sub> este justificată.'),
 hNote: () => 'Faptul că nu respingem o rădăcină unitară nu dovedește că ea există: cu 20 de ani de trimestre testul are putere mică, iar decizia se schimbă cu numărul de laguri și cu sfîrșitul eșantionului.',
 h1: (q, a, b, c, d, r) => `Filtrele la t = ${q} (y = ln Y): y<sub>1t</sub> = (1 + L + L² + L³)y<sub>t</sub> = ${a} + ${b} + ${c} + ${d} = ${r} (păstrează frecvența 0).`,
 h2: (a, b, c, d, r) => `y<sub>2t</sub> = −(1 − L + L² − L³)y<sub>t</sub> = −(${a} − ${b} + ${c} − ${d}) = ${r} (păstrează rădăcina −1, perioada de 2 trimestre).`,
 h3: (a, c, r, d4) => `y<sub>3t</sub> = −(1 − L²)y<sub>t</sub> = −(${a} − ${c}) = ${r} (păstrează ±i, perioada de 4 trimestre); variabila dependentă Δ<sub>4</sub>y<sub>t</sub> = ${d4}.`,
 h4: (n, k, aic, tr) => `Regresia lui Δ<sub>4</sub>y<sub>t</sub> pe y<sub>1,t−1</sub>, y<sub>2,t−1</sub>, y<sub>3,t−2</sub>, y<sub>3,t−1</sub>, o constantă, trei variabile dummy sezoniere${tr ? ', un trend' : ''} și ${k} ${k === 1 ? 'lag' : 'laguri'} ale lui Δ<sub>4</sub>y${aic ? ' (ales după AIC dintre 0–4)' : ''}: n = ${n}.`,
 hCap1: 'Estimațiile celor patru coeficienți π',
 hCap2: (a, b, n) => `Testele, ${a} – ${b}, n = ${n}; valorile critice de 5% din 4 000 de simulări ale lui Δ<sub>4</sub>y<sub>t</sub> = ε<sub>t</sub> cu același n`,
 // OCSB, CH, KPSS
 ser: { GDP: 'PIB (trimestrial)', 'Retail trade': 'Comerțul cu amănuntul', 'Industrial production': 'Producția industrială', 'Tourism nights': 'Înnoptări turistice', HICP: 'IAPC' },
 suCap: y => `Logaritmii seriilor, 2005 – sfîrșitul lui ${y}`,
 suNote: 'OCSB: D = 1 cînd statistica este peste valoarea critică (rădăcina unitară sezonieră nu este respinsă). Canova–Hansen: D = 1 cînd statistica este peste valoarea critică (sezonalitatea stabilă este respinsă). d: diferențele obișnuite alese de KPSS la 5%.',
 su0: (y, o, c, d, k) => `PIB pînă în ${y}: D = ${o} după OCSB, D = ${c} după Canova–Hansen, d = ${d} după KPSS. Serii lunare cu D = 0 la ambele teste: ${k} din 4.`,
 // exemplul airline
 maw: 'Ponderea MA', rhoTh: 'ρ, exemplul rezolvat', rhoFit: 'ρ implicat de θ̂, Θ̂ (1949–1960)', rhoSample: 'ACF de selecție a lui w<sub>t</sub>',
 aCap: (a, b, n, bd) => `Datele airline: θ̂ = ${a}, Θ̂ = ${b}; ACF de selecție a lui w<sub>t</sub> = ΔΔ<sub>12</sub> ln y<sub>t</sub>, ${n} de observații, banda ±${bd}`,
 a0: (g, r1, r12, r11) => `γ(0) = ${g}σ²; ρ₁ = ${r1}, ρ₁₂ = ${r12}, ρ₁₁ = ρ₁₃ = ${r11}; toate celelalte autocorelații sînt 0.`,
 a1: (t, T0, p) => `(1 + θL)(1 + ΘL¹²) = 1 + θL + ΘL¹² + θΘL¹³ = 1 + ${t}L + ${T0}L¹² + ${p}L¹³: patru ponderi nenule, doi parametri.`,
 a2: (t, T0, g) => `γ(0) = σ²(1 + θ²)(1 + Θ²) = σ²(1 + (${t})²)(1 + (${T0})²) = ${g}σ².`,
 a3: (t, a, r1, T0, b, r12, x, y, r11) => `ρ₁ = θ/(1 + θ²) = ${t}/${a} = ${r1}; ρ₁₂ = Θ/(1 + Θ²) = ${T0}/${b} = ${r12}; ρ₁₁ = ρ₁₃ = ρ₁ρ₁₂ = ${x}·${y} = ${r11}.`,
 // pasul de prognoză airline
 hh: 'h', maPart: 'Termenii MA', fcLev: 'Prognoza (mii)', obsT: 'observat', fcT: 'prognoză',
 f0: (m, f, a, e) => `${m}: prognoza ${f} mii de pasageri, valoarea reală ${a}; eroarea este ${e}% din valoarea reală.`,
 f1: (y, t, T0, p) => `Modelul estimat pe 1949–${y}: y<sub>t</sub> = y<sub>t−1</sub> + y<sub>t−12</sub> − y<sub>t−13</sub> + ε<sub>t</sub> + θε<sub>t−1</sub> + Θε<sub>t−12</sub> + θΘε<sub>t−13</sub>, θ̂ = ${t}, Θ̂ = ${T0}, θ̂Θ̂ = ${p}; y = ln(pasageri).`,
 f2: (h, m, m1, t1, a, m12, t12, b, m13, t13, c) => `h = ${h} (${m}): ŷ<sub>T+h−1</sub> = ${a} (${m1}, ${t1}), ŷ<sub>T+h−12</sub> = ${b} (${m12}, ${t12}), ŷ<sub>T+h−13</sub> = ${c} (${m13}, ${t13}).`,
 f3: (t, e1, T0, e12, p, e13, m) => `Termenii MA, cu reziduurile estimării (șocurile viitoare sînt 0): θ̂ε̂<sub>T+h−1</sub> + Θ̂ε̂<sub>T+h−12</sub> + θ̂Θ̂ε̂<sub>T+h−13</sub> = ${t}·${e1} + ${T0}·${e12} + ${p}·${e13} = ${m}.`,
 f3b: h => `La h = ${h} > 13 toate șocurile din termenii MA sînt în viitor, deci termenii sînt 0: prognoza doar repetă tiparul sezonier pe dreaptă.`,
 f4: (a, b, c, m, f, lv, chart, act) => `ŷ<sub>T+h</sub> = ${a} + ${b} − ${c} + ${m} = ${f}; exp(${f}) = ${lv} mii (graficul: ${chart}; valoarea reală: ${act}).`,
 // estimațiile airline pe origini
 mapeA: 'MAPE, airline (%)', mapeS: 'MAPE, sezonieră naivă (%)', dec60: 'Decembrie 1960: prognoza [95%]',
 atCap: 'Modelul airline SARIMA(0,1,1)(0,1,1)<sub>12</sub> pentru ln y; Q*(24): Ljung–Box cu 22 de grade de libertate; MAPE pe lunile de la origine pînă în decembrie 1960',
 at0: (y, t, T0, p, a, s) => `Estimat pe 1949–${y}: θ̂ = ${t}, Θ̂ = ${T0}, Ljung–Box p = ${p}; MAPE pînă în decembrie 1960: ${a}% pentru modelul airline, ${s}% pentru prognoza sezonieră naivă.`,
 at1: (t, T0, p) => `Eșantionul complet 1949–1960: θ̂ = ${t}, Θ̂ = ${T0}, Ljung–Box p = ${p}: p = 0,37 din slide este testul pe eșantionul complet.`,
 // modelele PIB
 kPar: 'Parametri ARMA', gmCap: '100 ln PIB, neajustat, T1 2000 – T2 2026, d = D = 1; n = 101 după diferențiere; * nu apare în slide',
 gmNote: m => `LB(${m}): Ljung–Box pe ${m} laguri ale reziduurilor, cu ${m} minus numărul de parametri ARMA grade de libertate. Îngroșat: cel mai mic AICc și cel mai mic BIC.`,
 gmCap2: m => `Estimațiile modelului SARIMA${m}`,
 g1: (m, k0, k, n) => `${m}: ${k0} ${k0 === 1 ? 'parametru' : 'parametri'} ARMA plus σ², deci k = ${k}; n = 106 − 5 = ${n} observații ale lui ΔΔ<sub>4</sub>y.`,
 g2: (ll, k, a) => `AIC = −2 ln L + 2k = −2·${ll} + 2·${k} = ${a}.`,
 g3: (a, k, k1, d, c, r) => `AICc = AIC + 2k(k + 1)/(n − k − 1) = ${a} + 2·${k}·${k1}/${d} = ${a} + ${c} = ${r}.`,
 g4: (ll, k, l, b) => `BIC = −2 ln L + k ln n = −2·${ll} + ${k}·${l} = ${b}.`,
 gm0: (m, a, b, da, db, lm, p) => `${m}: AICc ${a} (cu ${da} peste cel mai mic), BIC ${b} (cu ${db} peste cel mai mic); LB(${lm}) ${p}.`,
 // prognozele PIB
 fcBn: 'Prognoza (mld. EUR)', lower: l => `Limita inferioară ${l}%`, upper: l => `Limita superioară ${l}%`, halfPct: 'Limita superioară / prognoză − 1 (%)', yoy: 'Creșterea anuală (%)', lastYear: 'Același trimestru, cu un an înainte (mld. EUR)',
 gtCap: (T0, s) => `SARIMA(0,1,0)(0,1,1)<sub>4</sub>, Θ = ${T0}, σ̂ = ${s}%; prognozele lui 100 ln Y sînt transformate înapoi în mld. EUR (prețuri 2010)`,
 gt0: (l, f, lv, a, b, c, d) => `Ultima valoare, T2 2026: ${l} mld. EUR; prognozele ${f.join('; ')} mld. EUR pentru T3 2026 – T2 2027; intervalul de ${lv}% ${a}–${b} pentru primul trimestru și ${c}–${d} pentru al optulea.`,
 gt1: (w, T0) => `Ponderea celui mai recent an în tiparul sezonier al prognozei: 1 + Θ = ${w} (Θ = ${T0}).`,
 // ajustarea sezonieră
 facMean: (a, b) => `Factorul mediu NSA/SCA, ${a}–${b}`, facMin: 'Cel mai mic', facMax: 'Cel mai mare', diffPct: 'SCA / NSA − 1 (%)',
 stCap: 'PIB: factorul sezonier și de calendar implicat NSA/SCA, pe trimestre', stCap2: 'Totalurile anuale, mld. EUR (prețuri 2010)',
 q1: (q, q0, a, b, d) => `Neajustat: 100(ln Y<sub>${q}</sub> − ln Y<sub>${q0}</sub>) = 100(ln ${a} − ln ${b}) = ${d}%.`,
 q2: (a, b, d) => `Ajustat (SCA): 100(ln ${a} − ln ${b}) = ${d}%.`,
 q3: (d, f1, f0, df) => `Diferența, ${d}, este variația factorului sezonier și de calendar: 100(ln ${f1} − ln ${f0}) = ${df}.`,
 st0: (a, b, f) => `Factorii PIB, ${a}–${b}: T1 ${f[0]}, T2 ${f[1]}, T3 ${f[2]}, T4 ${f[3]}.`,
 st1: (a, b, d, j) => `Comerțul cu amănuntul, ${a}–${b}: factorul din decembrie ${d}, din ianuarie ${j}.`,
 // perechea Fourier
 contrib: 'α<sub>k</sub> sin + β<sub>k</sub> cos', amp: 'Amplitudinea √(α² + β²)',
 oCapT: 'Înnoptări turistice: coeficienții ajustării cu K = 3 a efectului sezonier lunar al lui ln(înnoptări), 2012–2019', oCapD: 'Consumul zilnic: coeficienții Fourier anuali ai DHR (GW), date pînă la 30 iunie 2025; t = zile de la 1 ianuarie 2000',
 o1t: (t, m) => `Perioada m = 12 luni; t = ${t} (${m}).`,
 o1d: (d, t) => `Perioada m = 365,25 zile; ${d} înseamnă t = ${t} zile după 1 ianuarie 2000.`,
 o2: (k, t, m, a, c) => `Unghiul armonicii k = ${k}: 2π·${k}·${t}/${m} = ${a} rad, adică ${c} dintr-un ciclu întreg.`,
 o3: (s, c) => `sin = ${s}, cos = ${c}.`,
 o4: (k, a, s, b, c, v, u, am) => `α<sub>${k}</sub> sin + β<sub>${k}</sub> cos = ${a}·${s} + ${b}·${c} = ${v}${u}; amplitudinea perechii ${am}${u}.`,
 o5t: (K, v, p, e) => `Suma pentru k = 1..${K}: ${v} (efectul lunar ajustat: ${p}); exp(${v}) = ${e}: circa ${e} din media mobilă pe 13 luni.`,
 o5d: (K, v, mw) => `Suma pentru k = 1..${K}: ${v} GW, adică ${mw} MW peste sau sub media anuală a părții deterministe.`,
 o6: (K, n, tr) => `${K} perechi = ${n} coeficienți, față de ${tr ? '11 variabile dummy lunare' : '364 de variabile dummy zilnice, imposibil pentru o perioadă de 365,25 zile'}.`,
 o0t: (m, v, p) => `${m}: ajustarea Fourier cu K = 3 dă ${v}, efectul sezonier este ${p} (în logaritmi).`,
 o0d: (d, mw) => `${d}: componenta Fourier anuală a DHR, ${mw} MW.`,
 // tabelul ajustărilor Fourier
 ncoef: 'Coeficienți', resSd: 'Abaterea standard a reziduurilor', aug: 'Ajustarea, august', jan: 'Ajustarea, ianuarie',
 fkCap: (a, j) => `Efectul sezonier lunar al lui ln(înnoptări turistice), 2012–2019 (august ${a}, ianuarie ${j}); AICc = 12 ln(SSR/12) + 2k + 2k(k + 1)/(12 − k − 1), k = coeficienți + 1`,
 fkNote: (a, b) => `K = 6: sin(2π·6t/12) = sin(πt) este 0 în fiecare lună și este omis, deci 11 coeficienți reproduc cele 12 puncte (R² = ${a}); K = 5 dă R² = ${b}.`,
 fk0: (K, n, r2, a, p) => `K = ${K}: ${n} coeficienți, R² = ${r2}; august ajustat ${a}, observat ${p}.`,
 fk1: (K, a) => `AICc este minim la K = ${K} (${a}): cu 12 puncte penalizarea 2k(k + 1)/(12 − k − 1) este deja 28 la K = 3, așa că alege o singură undă, deși vîrful din august cere K ≥ 3.`,
 // ponderea Paștelui
 daysIn: 'Zile ale ferestrei în lună', effFood: 'Comerț cu alimente: β̂<sub>E</sub>·E<sub>t</sub> (%)', xeCap: w => `Regresorul Paștelui cu w = ${w}; β̂<sub>E</sub> pentru alimente, estimat cu același w`,
 e0: (y, d, L) => `${y}: Paștele ortodox pe ${d}; E = ` + L.map(([m, v]) => `${v} în ${m}`).join('; ') + '.',
 e1: (y, d, w, a, b) => `${y}: duminica Paștelui pe ${d}; cele w = ${w} zile dinainte merg de la ${a} la ${b}.`,
 e2: L => L.map(([m, n, w, v]) => `${m}: ${n} din cele ${w} zile, E = ${n}/${w} = ${v}`).join('; ') + '; în celelalte luni E = 0.',
 e3: (w, b, L) => `Comerț cu alimente, β̂<sub>E</sub> = ${b}% (w = ${w}): ` + L.map(([m, v]) => `${m} +${v}%`).join('; ') + '.',
 // coeficienții Paștelui
 easterCol: w => `Paștele (w = ${w})`, wdCol: 'Zile lucrătoare', etCap: 'Regresie cu erori SARIMA(0,1,1)(0,1,1)<sub>12</sub> pentru 100 ln(volumul vînzărilor), ianuarie 2010 – iulie 2026; p din N(0, 1)',
 etNote: (a, b) => `w = 10 este estimarea din slide. Variabila dummy a carantinei, aprilie 2020: ${a}% (alimente), ${b}% (tot comerțul).`,
 et0: (w, l, f, fs, ft, t, ts, tw, fw) => `w = ${w}${l ? ' (curs)' : ''}: comerț cu alimente β̂<sub>E</sub> = ${f}% (SE ${fs}, t = ${ft}); tot comerțul ${t}% (SE ${ts}); o zi lucrătoare în plus: +${tw}% tot comerțul, +${fw}% alimente.`,
 // DHR
 dhrLab: { mon: 'Luni', tue: 'Marți', wed: 'Miercuri', thu: 'Joi', fri: 'Vineri', sat: 'Sîmbătă', holiday: 'Sărbătoare legală (H)', easter: 'Paștele, de vineri pînă luni (E)', xmas: '24 decembrie – 2 ianuarie (X)' },
 drCap: (q, df, p) => `Consumul zilnic, 1 ianuarie 2022 – 30 iunie 2025, erori ARIMA(1,1,1); efectele zilelor față de duminică; Ljung–Box(14) = ${q} cu ${df} grade de libertate, ${p}`,
 d1: (d, w, v) => `${d} este ${w}: efectul zilei față de duminică ${v} GW.`,
 d2: (K, v) => `Partea Fourier anuală cu K = ${K}: Σ<sub>k</sub>[α<sub>k</sub> sin(2πkt/365,25) + β<sub>k</sub> cos(2πkt/365,25)] = ${v} GW.`,
 d3: (h, ea, xm, ho, v) => (h ? `Sărbătoare legală (${h}): ` : 'Nicio zi specială: ') + (ea ? 'variabila dummy a Paștelui' : xm ? 'variabila dummy Crăciun – Anul Nou' : ho ? 'variabila dummy a sărbătorilor legale' : 'toate variabilele dummy ale zilelor speciale sînt 0') + `, efectul ${v} GW.`,
 d4: (a, b, c, s) => `Partea deterministă față de o duminică la media anuală: ${a} + ${b} + ${c} = ${s} GW; nivelul este purtat de erorile ARIMA(1,1,1).`,
 dr0: (w, m, s, h, e, x) => `Față de duminică: miercuri +${w}, luni +${m}, sîmbătă +${s} GW; sărbătoare legală ${h}, Paștele încă ${e}, Crăciun – Anul Nou ${x} GW.`,
 dr1: (a, va, b, vb, s, d) => `Componenta anuală (K = 4) este maximă pe ${a} (+${va} GW) și minimă pe ${b} (${vb} GW); partea deterministă pe ${d}: ${s} GW.`,
 // DM pe orizonturi
 dzCap: 'O valoare pentru fiecare origine la fiecare orizont (26 de valori); ferestrele nu se suprapun, deci h = 1 în formulă; rîndurile evidențiate: |HLN| peste t<sub>0,975</sub>(25)',
 dz0: (a, b, l, h, p) => `${({ 'Seasonal naive': 'sezoniera naivă', Combination: 'combinarea' })[a] || a} minus ${({ 'Seasonal naive': 'sezoniera naivă', Combination: 'combinarea' })[b] || b}, erori ${l === 'se' ? 'pătratice' : 'absolute'} mediate pe fiecare fereastră de 14 zile: HLN = ${h}, ${p}.`,
 dz1: (L, H, c) => (L.length ? `Orizont cu orizont, |HLN| > ${c} la h = ${L.join(', ')}` : `Orizont cu orizont, |HLN| rămîne sub ${c} la toate cele ${H} orizonturi`) + ': o singură zi pentru fiecare origine are mai mult zgomot decît media pe 14 zile folosită de testul din slide.',
 // exemplul DM
 slide5: 'cele cinci ferestre din slide',
 xdCapS: 'Cele cinci diferențe de pierdere mediate pe ferestre, din slide', xdCapR: l => `Erorile ${l === 'se' ? 'pătratice' : 'absolute'} mediate pe ferestre (GW${l === 'se' ? '²' : ''}), validarea încrucișată a consumului zilnic`,
 w1s: d => `d<sub>j</sub> = ${d}.`,
 w1r: (a, b, l, n, d) => `d<sub>j</sub> = eroarea ${l === 'se' ? 'pătratică' : 'absolută'} medie a metodei ${({ 'Seasonal naive': 'sezoniere naive', Combination: 'combinării' })[a] || a} minus cea a metodei ${({ 'Seasonal naive': 'sezoniere naive', Combination: 'combinării' })[b] || b} în fereastra j, n = ${n}: ${d}.`,
 w2: (n, s, m) => `d̄ = Σd<sub>j</sub>/n = ${s}/${n} = ${m}.`,
 w3: (n, g) => `γ̂<sub>d</sub>(0) = Σ(d<sub>j</sub> − d̄)²/n = ${g} (n = ${n}).`,
 w4: (m, g, n, dm) => `DM = d̄/√(γ̂<sub>d</sub>(0)/n) = ${m}/√(${g}/${n}) = ${dm}.`,
 w5: (n, c, dm, h) => `HLN = √((n + 1 − 2h + h(h − 1)/n)/n)·DM cu h = 1: √((n − 1)/n) = ${c}, deci HLN = ${c}·${dm} = ${h}.`,
 w6: (a, r, df, c, p, diff) => `|HLN| = ${a} ${r ? '>' : '<'} t<sub>0,975</sub>(${df}) = ${c}, ${p}: acuratețea egală ${r ? 'este respinsă' : 'nu este respinsă'} la 5%` + (diff ? '; valoarea critică a distribuției Normale, 1,96, ar fi dat răspunsul opus.' : '.'),
 xd0: (w, dm, h, p) => `${w}: DM = ${dm}, HLN = ${h}, ${p}.`,
 // tabelul validării încrucișate și MASE
 dmVs: m => `DM față de ${m === 'Seasonal naive' ? 's. naivă' : m === 'Combination' ? 'combinare' : m}`,
 cvCap: (k, j, l) => `MAE și RMSE în MW; ${k} origini × ${j} orizonturi; DM: statistica HLN (p-value) pe erorile ${l === 'se' ? 'pătratice' : 'absolute'} mediate pe ferestre, metoda din rînd minus metoda din coloană`,
 m1: (a, b, n) => `Scala MASE: eșantionul de antrenare al primei origini, ${a} – ${b}, ${n} de zile.`,
 m2: (d7, y7, d0, y0, a0, d8, y8, d1, y1, a1) => `Erorile în eșantion ale metodei sezoniere naive săptămînale: |y(${d7}) − y(${d0})| = |${y7} − ${y0}| = ${a0} GW; |y(${d8}) − y(${d1})| = |${y8} − ${y1}| = ${a1} GW; …`,
 m3: (n, s, q) => `Media lor: ${s}/${n} GW = ${q} MW.`,
 m4: (m, n, a, a2, q, r, lt) => `MAE pentru ${({ 'Seasonal naive': 'metoda sezonieră naivă', Combination: 'combinare' })[m] || m} pe ${n} erori în afara eșantionului: ${a} MW; MASE = ${a2}/${q} = ${r}: ${lt ? 'mai mică' : 'mai mare'} decît eroarea în eșantion a metodei sezoniere naive.`,
 cv0: (k, j, b, L) => `${k} origini × ${j} orizonturi: cel mai mic MAE pentru ${({ 'Seasonal naive': 'metoda sezonieră naivă', Combination: 'combinare' })[b] || b}; MASE sub 1: ${L.length ? L.map(m => ({ 'Seasonal naive': 'sezoniera naivă', Combination: 'combinarea' })[m] || m).join(', ') : 'niciuna'}.`,
 // validarea pentru PIB
 gcName: { SARIMA: 'SARIMA (airline)', ETS: 'ETS', 'Seasonal naive': 'Sezonieră naivă', Combination: 'Combinarea (SARIMA, ETS)' },
 gcCap: (n, ex, m) => `100 ln PIB, erori în %; ${n} de origini trimestriale din T4 2012${ex ? ', fără 2019–2020' : ''}; fereastră extinsă; îngroșat: cel mai mic ${m} la fiecare orizont`,
 gcDm: 'SARIMA față de', gcVs: r => (r === 'ETS' ? 'ETS' : 'Sezoniera naivă'),
 gcCap2: l => `HLN (p) pe erorile ${l === 'se' ? 'pătratice' : 'absolute'}; la orizontul h erorile se suprapun, deci intră h − 1 autocovarianțe`,
 gc0: (n, ex, m, s1, e1, s4, e4, n4) => `${n} de origini${ex ? ' (fără 2019–2020)' : ''}: ${m} la h = 1: SARIMA ${s1}, ETS ${e1}; la h = 4: SARIMA ${s4}, ETS ${e4}, sezoniera naivă ${n4}.`,
 gc1: (a, pa, b, pb) => `SARIMA față de ETS: HLN ${a} (${pa}) la h = 1, ${b} (${pb}) la h = 4.`,
 // combinarea
 c1: (m, s1, s2, r, real) => (real ? `Erorile pentru ${m} pe 26 × 14 zile (MW): ` : 'Exemplul din slide: ') + `σ₁ = ${s1}, σ₂ = ${s2}, ρ = ${r}.`,
 c2: (w, s1, w2, s2, r, m) => `MSE(${w}) = ${w}²·${s1}² + ${w2}²·${s2}² + 2·${w}·${w2}·${r}·${s1}·${s2} = ${m}.`,
 c3: (a, b, c, w, m) => `w* = (σ₂² − ρσ₁σ₂)/(σ₁² + σ₂² − 2ρσ₁σ₂) = (${a} − ${b})/${c} = ${w}; MSE(w*) = ${m}.`,
 c4: (h, b, lt) => `Ponderi egale: MSE(0,5) = ${h}, ${lt ? 'sub' : 'nu sub'} varianța erorii celei mai bune prognoze singure, ${b}.`,
 c5: (d, y, f1, f2, w, w2, fc, ec, e1, e2) => `O zi, ${d}: consumul ${y} MW; prognozele ${f1} și ${f2}; combinată ${w}·${f1} + ${w2}·${f2} = ${fc}; eroarea ${ec} MW (cele două singure: ${e1} și ${e2}).`,
 c6: (v, b, m, r, a) => `Pe cele 364 de zile: varianța erorii combinate după formulă, ${v}, plus pătratul erorii medii (${b})², dă MSE ${m}: RMSE ${r} MW, MAE ${a} MW.`,
 c0: (real, w, m, ws, mw) => `${real ? 'Erorile validării încrucișate' : 'Exemplul din slide'}: MSE(${w}) = ${m}; ponderea optimă w* = ${ws}, MSE(w*) = ${mw}.`,
 // ponderile combinării
 eqW: 'Ponderi egale', invPast: 'Inversul MSE din ferestrele anterioare (ponderi medii)', invHind: 'Inversul MSE din toate cele 26 de ferestre (a posteriori)', bestHind: 'Cea mai bună pondere pe o grilă de 0,01 (a posteriori)',
 alone: m => `${({ 'Seasonal naive': 'Sezoniera naivă' })[m] || m} singură`,
 cwCap: n => `Consumul zilnic, 26 de origini × 14 zile; ${n} componente; ponderile inverse MSE din ferestrele anterioare sînt reestimate la fiecare origine (ponderi egale la prima)`,
 cw0: (n, e, p, h, b) => `${n} componente: ponderi egale, MAE ${e} MW; ponderi din ferestrele anterioare ${p}; ponderi alese a posteriori ${h}; cea mai bună componentă singură ${b}.`
} });
""",
 'en': r"""Object.assign(window.CH4_TEXT, { x: {
 mn: m => m, monName: ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
 wdName: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], hol: h => h,
 quarter: 'Quarter', quarter0: 'Quarter', yBn: 'Y<sub>t</sub> (bn EUR)', coef: 'Coefficient', est: 'Estimate', test: 'Test', stat: 'statistic', cv5: '5% critical value', decision: 'Decision at 5%',
 rej: 'rejected', norej: 'not rejected', root: 'root', roots: 'roots', lag: 'Lag', model: 'Model', method: 'Method', sample: 'Estimation sample', year: 'Year', month: 'Month', month0: 'Month',
 actual: 'Actual', origin: 'Origin', series0: 'Series', ser0: 'Series (logs, from 2005)', food: 'Food retail', total: 'All retail', scheme: 'Weights', annual: 'annual Fourier pair',
 // the seasonal difference
 s0: (q, d1, d4, dd) => `${q}: Δy = ${d1} (the seasonal swing from the previous quarter), Δ<sub>4</sub>y = ${d4} (growth over the same quarter one year earlier, in log points), ΔΔ<sub>4</sub>y = ${dd}.`,
 s1: (q, Y, y, q4, Y4, y4) => `Data: Y = ${Y} bn EUR in ${q}, so y<sub>t</sub> = 100 ln Y<sub>t</sub> = ${y}; Y = ${Y4} bn EUR in ${q4}, y<sub>t−4</sub> = ${y4}.`,
 s2: (y, y4, d, q, q4) => `Δ<sub>4</sub>y<sub>t</sub> = y<sub>t</sub> − y<sub>t−4</sub> = ${y} − ${y4} = ${d}: GDP in ${q} compared with ${q4}, in log points (≈ %).`,
 s3: (y, y1, d) => `Δy<sub>t</sub> = y<sub>t</sub> − y<sub>t−1</sub> = ${y} − ${y1} = ${d}: mostly the seasonal pattern, not growth.`,
 s4: (y, y1, y4, y5, d) => `ΔΔ<sub>4</sub>y<sub>t</sub> = y<sub>t</sub> − y<sub>t−1</sub> − y<sub>t−4</sub> + y<sub>t−5</sub> = ${y} − ${y1} − ${y4} + ${y5} = ${d}.`,
 s5: (a, b, c, d, e, f) => `The order does not matter: Δ(Δ<sub>4</sub>y)<sub>t</sub> = ${a} − ${b} = ${c} and Δ<sub>4</sub>(Δy)<sub>t</sub> = ${d} − ${e} = ${f}.`,
 // seasonal roots
 unit: { 4: 'quarters', 12: 'months', 7: 'days', 24: 'hours' }, unit0: 'periods', freq: 'Frequency k/s (cycles per period)', period: u => `Period s/k (${u})`, factor: 'Real factor of 1 − z<sup>s</sup>', same: k => `pair of k = ${k}`,
 r0: (s, nr, np) => `1 − z<sup>${s}</sup> has ${s} roots on the unit circle: ${nr === 2 ? 'two real ones, 1 and −1' : 'one real one, 1'}, and ${np} complex-conjugate pair${np === 1 ? '' : 's'}.`,
 r1: (k, s, a, z) => `z<sub>${k}</sub> = cos(2π·${k}/${s}) + i sin(2π·${k}/${s}), with the angle 2π·${k}/${s} = ${a} rad: z<sub>${k}</sub> = ${z}.`,
 r2: (s, k, m) => `z<sub>${k}</sub><sup>${s}</sup> = cos(2π·${k}) + i sin(2π·${k}) = 1, so 1 − z<sub>${k}</sub><sup>${s}</sup> = 0; the modulus |z<sub>${k}</sub>| = ${m}: the root lies on the unit circle.`,
 r3: (k, s, f, p, u) => `Frequency ${k}/${s} = ${f} cycles per period, period ${s}/${k} = ${p} ${u}.`,
 r3a: s => `k = 0: z = 1, frequency 0: the root of the regular difference (the trend), present for every s = ${s}.`,
 r4: (k2, z, f) => `Its conjugate z<sub>${k2}</sub> = ${z} has the same period; together they give the real factor ${f}.`,
 r4a: (f, z) => `A real root (z = ${z}) gives the real factor ${f}.`,
 r5: (s, f) => `1 − z<sup>${s}</sup> = ${f}: one factor per frequency; a seasonal unit-root test checks each factor separately.`,
 // HEGY
 h0: (q, n, k, t1, t2, F, nr) => `Sample 2000Q1–${q}, n = ${n}, ${k} lag${k === 1 ? '' : 's'} of Δ<sub>4</sub>y: t(π₁) = ${t1}, t(π₂) = ${t2}, F(π₃, π₄) = ${F}; ` + (nr ? `${nr} of the 3 nulls rejected at 5%.` : 'none of the three nulls is rejected at 5%: Δ<sub>4</sub> is justified.'),
 hNote: () => 'Not rejecting a unit root is not evidence that it exists: with 20 years of quarters the test has little power, and the decision moves with the number of lags and the sample end.',
 h1: (q, a, b, c, d, r) => `Filters at t = ${q} (y = ln Y): y<sub>1t</sub> = (1 + L + L² + L³)y<sub>t</sub> = ${a} + ${b} + ${c} + ${d} = ${r} (keeps frequency 0).`,
 h2: (a, b, c, d, r) => `y<sub>2t</sub> = −(1 − L + L² − L³)y<sub>t</sub> = −(${a} − ${b} + ${c} − ${d}) = ${r} (keeps the root −1, period 2 quarters).`,
 h3: (a, c, r, d4) => `y<sub>3t</sub> = −(1 − L²)y<sub>t</sub> = −(${a} − ${c}) = ${r} (keeps ±i, period 4 quarters); the dependent variable Δ<sub>4</sub>y<sub>t</sub> = ${d4}.`,
 h4: (n, k, aic, tr) => `Regression of Δ<sub>4</sub>y<sub>t</sub> on y<sub>1,t−1</sub>, y<sub>2,t−1</sub>, y<sub>3,t−2</sub>, y<sub>3,t−1</sub>, a constant, three seasonal dummies${tr ? ', a trend' : ''} and ${k} lag${k === 1 ? '' : 's'} of Δ<sub>4</sub>y${aic ? ' (chosen by AIC among 0–4)' : ''}: n = ${n}.`,
 hCap1: 'Estimates of the four π coefficients',
 hCap2: (a, b, n) => `Tests, ${a}–${b}, n = ${n}; 5% critical values from 4,000 simulations of Δ<sub>4</sub>y<sub>t</sub> = ε<sub>t</sub> with the same n`,
 // OCSB, CH, KPSS
 ser: { GDP: 'GDP (quarterly)', 'Retail trade': 'Retail trade', 'Industrial production': 'Industrial production', 'Tourism nights': 'Tourism nights', HICP: 'HICP' },
 suCap: y => `Logs of the series, 2005 – end of ${y}`,
 suNote: 'OCSB: D = 1 when its statistic is above the critical value (the seasonal unit root is not rejected). Canova–Hansen: D = 1 when its statistic is above the critical value (stable seasonality rejected). d: regular differences chosen by KPSS at 5%.',
 su0: (y, o, c, d, k) => `GDP to ${y}: D = ${o} by OCSB, D = ${c} by Canova–Hansen, d = ${d} by KPSS. Monthly series with D = 0 by both tests: ${k} of 4.`,
 // airline worked example
 maw: 'MA weight', rhoTh: 'ρ, worked example', rhoFit: 'ρ implied by θ̂, Θ̂ (1949–1960)', rhoSample: 'Sample ACF of w<sub>t</sub>',
 aCap: (a, b, n, bd) => `Airline data: θ̂ = ${a}, Θ̂ = ${b}; sample ACF of w<sub>t</sub> = ΔΔ<sub>12</sub> ln y<sub>t</sub>, ${n} observations, band ±${bd}`,
 a0: (g, r1, r12, r11) => `γ(0) = ${g}σ²; ρ₁ = ${r1}, ρ₁₂ = ${r12}, ρ₁₁ = ρ₁₃ = ${r11}; every other autocorrelation is 0.`,
 a1: (t, T0, p) => `(1 + θL)(1 + ΘL¹²) = 1 + θL + ΘL¹² + θΘL¹³ = 1 + ${t}L + ${T0}L¹² + ${p}L¹³: four nonzero weights, two parameters.`,
 a2: (t, T0, g) => `γ(0) = σ²(1 + θ²)(1 + Θ²) = σ²(1 + (${t})²)(1 + (${T0})²) = ${g}σ².`,
 a3: (t, a, r1, T0, b, r12, x, y, r11) => `ρ₁ = θ/(1 + θ²) = ${t}/${a} = ${r1}; ρ₁₂ = Θ/(1 + Θ²) = ${T0}/${b} = ${r12}; ρ₁₁ = ρ₁₃ = ρ₁ρ₁₂ = ${x}·${y} = ${r11}.`,
 // airline forecast step
 hh: 'h', maPart: 'MA terms', fcLev: 'Forecast (thousands)', obsT: 'observed', fcT: 'forecast',
 f0: (m, f, a, e) => `${m}: forecast ${f} thousand passengers, actual ${a}; the error is ${e}% of the actual value.`,
 f1: (y, t, T0, p) => `Model estimated on 1949–${y}: y<sub>t</sub> = y<sub>t−1</sub> + y<sub>t−12</sub> − y<sub>t−13</sub> + ε<sub>t</sub> + θε<sub>t−1</sub> + Θε<sub>t−12</sub> + θΘε<sub>t−13</sub>, θ̂ = ${t}, Θ̂ = ${T0}, θ̂Θ̂ = ${p}; y = ln(passengers).`,
 f2: (h, m, m1, t1, a, m12, t12, b, m13, t13, c) => `h = ${h} (${m}): ŷ<sub>T+h−1</sub> = ${a} (${m1}, ${t1}), ŷ<sub>T+h−12</sub> = ${b} (${m12}, ${t12}), ŷ<sub>T+h−13</sub> = ${c} (${m13}, ${t13}).`,
 f3: (t, e1, T0, e12, p, e13, m) => `MA terms with the residuals of the fit (future shocks are 0): θ̂ε̂<sub>T+h−1</sub> + Θ̂ε̂<sub>T+h−12</sub> + θ̂Θ̂ε̂<sub>T+h−13</sub> = ${t}·${e1} + ${T0}·${e12} + ${p}·${e13} = ${m}.`,
 f3b: h => `At h = ${h} > 13 every shock in the MA terms lies in the future, so they are 0: the forecast only repeats the seasonal pattern on the line.`,
 f4: (a, b, c, m, f, lv, chart, act) => `ŷ<sub>T+h</sub> = ${a} + ${b} − ${c} + ${m} = ${f}; exp(${f}) = ${lv} thousand (the chart: ${chart}; actual: ${act}).`,
 // airline estimates by origin
 mapeA: 'MAPE, airline (%)', mapeS: 'MAPE, seasonal naive (%)', dec60: 'December 1960: forecast [95%]',
 atCap: 'Airline model SARIMA(0,1,1)(0,1,1)<sub>12</sub> for ln y; Q*(24): Ljung–Box on 22 degrees of freedom; MAPE over the months from the origin to December 1960',
 at0: (y, t, T0, p, a, s) => `Estimated on 1949–${y}: θ̂ = ${t}, Θ̂ = ${T0}, Ljung–Box p = ${p}; MAPE to December 1960: ${a}% for the airline model, ${s}% for the seasonal naive forecast.`,
 at1: (t, T0, p) => `Full sample 1949–1960: θ̂ = ${t}, Θ̂ = ${T0}, Ljung–Box p = ${p}: the p = 0.37 of the slide is this full-sample test.`,
 // GDP models
 kPar: 'ARMA parameters', gmCap: '100 ln GDP, not adjusted, 2000Q1–2026Q2, d = D = 1; n = 101 after differencing; * not on the slide',
 gmNote: m => `LB(${m}): Ljung–Box on ${m} lags of the residuals with ${m} minus the number of ARMA parameters degrees of freedom. Bold: lowest AICc and lowest BIC.`,
 gmCap2: m => `Estimates of SARIMA${m}`,
 g1: (m, k0, k, n) => `${m}: ${k0} ARMA parameter${k0 === 1 ? '' : 's'} plus σ², so k = ${k}; n = 106 − 5 = ${n} observations of ΔΔ<sub>4</sub>y.`,
 g2: (ll, k, a) => `AIC = −2 ln L + 2k = −2·${ll} + 2·${k} = ${a}.`,
 g3: (a, k, k1, d, c, r) => `AICc = AIC + 2k(k + 1)/(n − k − 1) = ${a} + 2·${k}·${k1}/${d} = ${a} + ${c} = ${r}.`,
 g4: (ll, k, l, b) => `BIC = −2 ln L + k ln n = −2·${ll} + ${k}·${l} = ${b}.`,
 gm0: (m, a, b, da, db, lm, p) => `${m}: AICc ${a} (${da} above the lowest), BIC ${b} (${db} above the lowest); LB(${lm}) ${p}.`,
 // GDP forecasts
 fcBn: 'Forecast (bn EUR)', lower: l => `${l}% lower`, upper: l => `${l}% upper`, halfPct: 'Upper bound / forecast − 1 (%)', yoy: 'y/y growth (%)', lastYear: 'Same quarter, one year earlier (bn EUR)',
 gtCap: (T0, s) => `SARIMA(0,1,0)(0,1,1)<sub>4</sub>, Θ = ${T0}, σ̂ = ${s}%; the forecasts of 100 ln Y are turned back into bn EUR (2010 prices)`,
 gt0: (l, f, lv, a, b, c, d) => `Last value 2026Q2: ${l} bn EUR; forecasts ${f.join(', ')} bn EUR for 2026Q3–2027Q2; ${lv}% interval ${a}–${b} for the first quarter and ${c}–${d} for the eighth.`,
 gt1: (w, T0) => `Weight of the newest year in the forecast seasonal pattern: 1 + Θ = ${w} (Θ = ${T0}).`,
 // seasonal adjustment
 facMean: (a, b) => `Mean factor NSA/SCA, ${a}–${b}`, facMin: 'Lowest', facMax: 'Highest', diffPct: 'SCA / NSA − 1 (%)',
 stCap: 'GDP: implied seasonal and calendar factor NSA/SCA by quarter', stCap2: 'Annual totals, bn EUR (2010 prices)',
 q1: (q, q0, a, b, d) => `Unadjusted: 100(ln Y<sub>${q}</sub> − ln Y<sub>${q0}</sub>) = 100(ln ${a} − ln ${b}) = ${d}%.`,
 q2: (a, b, d) => `Adjusted (SCA): 100(ln ${a} − ln ${b}) = ${d}%.`,
 q3: (d, f1, f0, df) => `The gap ${d} is the change of the seasonal and calendar factor: 100(ln ${f1} − ln ${f0}) = ${df}.`,
 st0: (a, b, f) => `GDP factors, ${a}–${b}: Q1 ${f[0]}, Q2 ${f[1]}, Q3 ${f[2]}, Q4 ${f[3]}.`,
 st1: (a, b, d, j) => `Retail trade, ${a}–${b}: December factor ${d}, January ${j}.`,
 // Fourier pair
 contrib: 'α<sub>k</sub> sin + β<sub>k</sub> cos', amp: 'Amplitude √(α² + β²)',
 oCapT: 'Tourism nights: coefficients of the K = 3 fit to the monthly seasonal effect of ln(nights), 2012–2019', oCapD: 'Daily load: annual Fourier coefficients of the DHR (GW), data to 30 June 2025; t = days since 1 January 2000',
 o1t: (t, m) => `Period m = 12 months; t = ${t} (${m}).`,
 o1d: (d, t) => `Period m = 365.25 days; ${d} is t = ${t} days after 1 January 2000.`,
 o2: (k, t, m, a, c) => `Angle of harmonic k = ${k}: 2π·${k}·${t}/${m} = ${a} rad, i.e. ${c} of a full cycle.`,
 o3: (s, c) => `sin = ${s}, cos = ${c}.`,
 o4: (k, a, s, b, c, v, u, am) => `α<sub>${k}</sub> sin + β<sub>${k}</sub> cos = ${a}·${s} + ${b}·${c} = ${v}${u}; amplitude of the pair ${am}${u}.`,
 o5t: (K, v, p, e) => `Sum over k = 1..${K}: ${v} (the monthly effect it fits: ${p}); exp(${v}) = ${e}: about ${e} times the 13-month moving average.`,
 o5d: (K, v, mw) => `Sum over k = 1..${K}: ${v} GW, i.e. ${mw} MW above or below the annual mean of the deterministic part.`,
 o6: (K, n, tr) => `${K} pairs = ${n} coefficients, against ${tr ? '11 monthly dummies' : '364 daily dummies, impossible for a period of 365.25 days'}.`,
 o0t: (m, v, p) => `${m}: the Fourier fit with K = 3 gives ${v}, the seasonal effect is ${p} (in logs).`,
 o0d: (d, mw) => `${d}: annual Fourier component of the DHR ${mw} MW.`,
 // Fourier fits table
 ncoef: 'Coefficients', resSd: 'Residual sd', aug: 'Fit, August', jan: 'Fit, January',
 fkCap: (a, j) => `Monthly seasonal effect of ln(tourism nights), 2012–2019 (August ${a}, January ${j}); AICc = 12 ln(SSR/12) + 2k + 2k(k + 1)/(12 − k − 1), k = coefficients + 1`,
 fkNote: (a, b) => `K = 6: sin(2π·6t/12) = sin(πt) is 0 every month and is left out, so 11 coefficients reproduce the 12 points (R² = ${a}); K = 5 gives R² = ${b}.`,
 fk0: (K, n, r2, a, p) => `K = ${K}: ${n} coefficients, R² = ${r2}; August fitted ${a}, observed ${p}.`,
 fk1: (K, a) => `AICc is lowest at K = ${K} (${a}): with 12 points its penalty 2k(k + 1)/(12 − k − 1) is already 28 at K = 3, so it prefers the single wave, although the August peak needs K ≥ 3.`,
 // Easter share
 daysIn: 'Days of the window in the month', effFood: 'Food retail: β̂<sub>E</sub>·E<sub>t</sub> (%)', xeCap: w => `Easter regressor with w = ${w}; β̂<sub>E</sub> of food retail estimated with the same w`,
 e0: (y, d, L) => `${y}: Orthodox Easter on ${d}; E = ` + L.map(([m, v]) => `${v} in ${m}`).join(', ') + '.',
 e1: (y, d, w, a, b) => `${y}: Easter Sunday ${d}; the w = ${w} days before it run from ${a} to ${b}.`,
 e2: L => L.map(([m, n, w, v]) => `${m}: ${n} of the ${w} days, E = ${n}/${w} = ${v}`).join('; ') + '; every other month: E = 0.',
 e3: (w, b, L) => `Food retail, β̂<sub>E</sub> = ${b}% (w = ${w}): ` + L.map(([m, v]) => `${m} +${v}%`).join(', ') + '.',
 // Easter coefficients
 easterCol: w => `Easter (w = ${w})`, wdCol: 'Working days', etCap: 'Regression with SARIMA(0,1,1)(0,1,1)<sub>12</sub> errors for 100 ln(retail volume), January 2010 – July 2026; p from N(0, 1)',
 etNote: (a, b) => `w = 10 is the fit of the slide. Lockdown dummy, April 2020: ${a}% (food), ${b}% (all retail).`,
 et0: (w, l, f, fs, ft, t, ts, tw, fw) => `w = ${w}${l ? ' (lecture)' : ''}: food retail β̂<sub>E</sub> = ${f}% (SE ${fs}, t = ${ft}); all retail ${t}% (SE ${ts}); one working day more: +${tw}% all retail, +${fw}% food.`,
 // DHR
 dhrLab: { mon: 'Monday', tue: 'Tuesday', wed: 'Wednesday', thu: 'Thursday', fri: 'Friday', sat: 'Saturday', holiday: 'Public holiday (H)', easter: 'Easter, Friday to Monday (E)', xmas: '24 December – 2 January (X)' },
 drCap: (q, df, p) => `Daily load, 1 January 2022 – 30 June 2025, ARIMA(1,1,1) errors; weekday effects relative to Sunday; Ljung–Box(14) = ${q} on ${df} degrees of freedom, ${p}`,
 d1: (d, w, v) => `${d} is a ${w}: weekday effect relative to Sunday ${v} GW.`,
 d2: (K, v) => `Annual Fourier part with K = ${K}: Σ<sub>k</sub>[α<sub>k</sub> sin(2πkt/365.25) + β<sub>k</sub> cos(2πkt/365.25)] = ${v} GW.`,
 d3: (h, ea, xm, ho, v) => (h ? `Public holiday (${h}): ` : 'No special day: ') + (ea ? 'Easter dummy' : xm ? 'Christmas – New Year dummy' : ho ? 'public-holiday dummy' : 'all special-day dummies are 0') + `, effect ${v} GW.`,
 d4: (a, b, c, s) => `Deterministic part relative to a Sunday at the annual mean: ${a} + ${b} + ${c} = ${s} GW; the ARIMA(1,1,1) errors carry the level.`,
 dr0: (w, m, s, h, e, x) => `Relative to Sunday: Wednesday +${w}, Monday +${m}, Saturday +${s} GW; public holiday ${h}, Easter ${e} more, Christmas – New Year ${x} GW.`,
 dr1: (a, va, b, vb, s, d) => `Annual component (K = 4) highest on ${a} (+${va} GW) and lowest on ${b} (${vb} GW); deterministic part on ${d}: ${s} GW.`,
 // DM by horizon
 dzCap: 'One value per origin at each horizon (26 values); the windows do not overlap, so h = 1 in the formula; highlighted rows: |HLN| above t<sub>0.975</sub>(25)',
 dz0: (a, b, l, h, p) => `${a} minus ${b}, ${l === 'se' ? 'squared' : 'absolute'} errors averaged over each 14-day window: HLN = ${h}, ${p}.`,
 dz1: (L, H, c) => (L.length ? `Horizon by horizon, |HLN| > ${c} at h = ${L.join(', ')}` : `Horizon by horizon, |HLN| stays below ${c} at all ${H} horizons`) + ': a single day per origin is noisier than the 14-day average used by the test of the slide.',
 // DM worked example
 slide5: 'the five windows of the slide',
 xdCapS: 'The five window-average loss differentials of the slide', xdCapR: l => `Window-average ${l === 'se' ? 'squared' : 'absolute'} errors (GW${l === 'se' ? '²' : ''}), cross-validation of the daily load`,
 w1s: d => `d<sub>j</sub> = ${d}.`,
 w1r: (a, b, l, n, d) => `d<sub>j</sub> = mean ${l === 'se' ? 'squared' : 'absolute'} error of ${a} minus that of ${b} in window j, n = ${n}: ${d}.`,
 w2: (n, s, m) => `d̄ = Σd<sub>j</sub>/n = ${s}/${n} = ${m}.`,
 w3: (n, g) => `γ̂<sub>d</sub>(0) = Σ(d<sub>j</sub> − d̄)²/n = ${g} (n = ${n}).`,
 w4: (m, g, n, dm) => `DM = d̄/√(γ̂<sub>d</sub>(0)/n) = ${m}/√(${g}/${n}) = ${dm}.`,
 w5: (n, c, dm, h) => `HLN = √((n + 1 − 2h + h(h − 1)/n)/n)·DM with h = 1: √((n − 1)/n) = ${c}, so HLN = ${c}·${dm} = ${h}.`,
 w6: (a, r, df, c, p, diff) => `|HLN| = ${a} ${r ? '>' : '<'} t<sub>0.975</sub>(${df}) = ${c}, ${p}: equal accuracy ${r ? 'rejected' : 'not rejected'} at 5%` + (diff ? '; the Normal value 1.96 would have given the opposite answer.' : '.'),
 xd0: (w, dm, h, p) => `${w}: DM = ${dm}, HLN = ${h}, ${p}.`,
 // cross-validation table and MASE
 dmVs: m => `DM against ${m === 'Seasonal naive' ? 's. naive' : m}`,
 cvCap: (k, j, l) => `MAE and RMSE in MW; ${k} origins × ${j} horizons; DM: HLN statistic (p-value) on window-average ${l === 'se' ? 'squared' : 'absolute'} errors, row method minus column method`,
 m1: (a, b, n) => `Scale of MASE: the training sample of the first origin, ${a} – ${b}, ${n} days.`,
 m2: (d7, y7, d0, y0, a0, d8, y8, d1, y1, a1) => `In-sample errors of the weekly seasonal naive method: |y(${d7}) − y(${d0})| = |${y7} − ${y0}| = ${a0} GW; |y(${d8}) − y(${d1})| = |${y8} − ${y1}| = ${a1} GW; …`,
 m3: (n, s, q) => `Their mean: ${s}/${n} GW = ${q} MW.`,
 m4: (m, n, a, a2, q, r, lt) => `MAE of ${m} over ${n} out-of-sample errors: ${a} MW; MASE = ${a2}/${q} = ${r}: ${lt ? 'smaller' : 'larger'} than the in-sample error of the seasonal naive method.`,
 cv0: (k, j, b, L) => `${k} origins × ${j} horizons: lowest MAE ${b}; MASE below 1: ${L.length ? L.join(', ') : 'none'}.`,
 // GDP cross-validation
 gcName: { SARIMA: 'SARIMA (airline)', ETS: 'ETS', 'Seasonal naive': 'Seasonal naive', Combination: 'Combination (SARIMA, ETS)' },
 gcCap: (n, ex, m) => `100 ln GDP, errors in %; ${n} quarterly origins from 2012Q4${ex ? ', 2019–2020 left out' : ''}; expanding window; bold: lowest ${m} at each horizon`,
 gcDm: 'SARIMA against', gcVs: r => (r === 'ETS' ? 'ETS' : 'Seasonal naive'),
 gcCap2: l => `HLN (p) on ${l === 'se' ? 'squared' : 'absolute'} errors; at horizon h the errors overlap, so h − 1 autocovariances enter`,
 gc0: (n, ex, m, s1, e1, s4, e4, n4) => `${n} origins${ex ? ' (without 2019–2020)' : ''}: ${m} at h = 1: SARIMA ${s1}, ETS ${e1}; at h = 4: SARIMA ${s4}, ETS ${e4}, seasonal naive ${n4}.`,
 gc1: (a, pa, b, pb) => `SARIMA against ETS: HLN ${a} (${pa}) at h = 1, ${b} (${pb}) at h = 4.`,
 // combination
 c1: (m, s1, s2, r, real) => (real ? `Errors of ${m} over 26 × 14 days (MW): ` : 'Slide example: ') + `σ₁ = ${s1}, σ₂ = ${s2}, ρ = ${r}.`,
 c2: (w, s1, w2, s2, r, m) => `MSE(${w}) = ${w}²·${s1}² + ${w2}²·${s2}² + 2·${w}·${w2}·${r}·${s1}·${s2} = ${m}.`,
 c3: (a, b, c, w, m) => `w* = (σ₂² − ρσ₁σ₂)/(σ₁² + σ₂² − 2ρσ₁σ₂) = (${a} − ${b})/${c} = ${w}; MSE(w*) = ${m}.`,
 c4: (h, b, lt) => `Equal weights: MSE(0.5) = ${h}, ${lt ? 'below' : 'not below'} the error variance of the better forecast alone, ${b}.`,
 c5: (d, y, f1, f2, w, w2, fc, ec, e1, e2) => `One day, ${d}: load ${y} MW; forecasts ${f1} and ${f2}; combined ${w}·${f1} + ${w2}·${f2} = ${fc}; error ${ec} MW (the two alone: ${e1} and ${e2}).`,
 c6: (v, b, m, r, a) => `On the 364 days: the variance of the combined error by the formula, ${v}, plus the squared mean error (${b})², gives the MSE ${m}: RMSE ${r} MW, MAE ${a} MW.`,
 c0: (real, w, m, ws, mw) => `${real ? 'Cross-validation errors' : 'Slide example'}: MSE(${w}) = ${m}; best weight w* = ${ws}, MSE(w*) = ${mw}.`,
 // combination weights
 eqW: 'Equal weights', invPast: 'Inverse MSE of the earlier windows (mean weights)', invHind: 'Inverse MSE of all 26 windows (hindsight)', bestHind: 'Best weight on a 0.01 grid (hindsight)', alone: m => `${m} alone`,
 cwCap: n => `Daily load, 26 origins × 14 days; ${n} members; the inverse-MSE weights of the earlier windows are re-estimated at each origin (equal weights at the first)`,
 cw0: (n, e, p, h, b) => `${n} members: equal weights MAE ${e} MW; weights from the earlier windows ${p}; weights chosen with hindsight ${h}; best single member ${b}.`
} });
""",
}
for _lg in TEXT_ADD:
    TEXT_JS[_lg] += TEXT_ADD[_lg]

# Questions of the EXTRA blocks
QX = {
 'ro': {
  'wx-sdiff': ('În T2 2026, Δy = 17,9, iar Δ₄y = −1,96. A crescut PIB-ul?',
               'Nu: 17,9 este saltul sezonier de la T1 la T2 (factorii NSA/SCA ai PIB sînt 0,78 în T1 și 0,93 în T2). Comparația cu același trimestru elimină sezonul: PIB-ul din T2 2026, 45,88 mld. EUR, este cu 1,96 puncte logaritmice sub cel din T2 2025, 46,79 mld. EUR. '
               'ΔΔ₄y = −0,46 arată că scăderea anuală s-a adîncit, de la −1,50 în T1 2026 la −1,96.'),
  'wx-roots': ('Pentru date lunare (s = 12), ce perioadă are rădăcina k = 1 și ce factor real dă împreună cu conjugata ei?',
               'Perioada 12/1 = 12 luni, ciclul anual: z₁ = 0,866 + 0,500i, iar perechea (k = 1, k = 11) dă factorul 1 − 1,732z + z². Pentru s = 12 există 12 rădăcini: două reale (1 și −1, perioada de 2 luni) și cinci perechi complex conjugate, cu perioadele 12, 6, 4, 3 și 2,4 luni.'),
  'hegy-table': ('Cu 0 laguri ale lui Δ₄y în loc de 1 (ales după AIC), pe 2000–2019, mai rămîn nerespinse toate cele trei ipoteze nule?',
                 'Nu: cu k = 0, t(π₂) = −3,34 < −2,77 și F(π₃, π₄) = 17,25 > 6,64, deci rădăcinile −1 și ±i sînt respinse; rămîne doar rădăcina 1 (t(π₁) = −1,46). Cu k = 1, ales după AIC, nicio ipoteză nu este respinsă (−1,97; −2,19; 6,12). '
                 'Autocorelația lăsată în reziduuri de regresia fără laguri schimbă decizia. Și sfîrșitul eșantionului contează: pînă în T4 2025, cu k = 1, t(π₂) = −2,88 și F = 8,38 resping două ipoteze.'),
  'sur-table': ('Pentru ce serii se schimbă decizia cînd mutăm sfîrșitul eșantionului între 2012 și 2025?',
                'Pentru PIB nu se schimbă: OCSB dă D = 1, Canova–Hansen D = 0 și KPSS d = 1 pentru orice sfîrșit. Se schimbă la înnoptările turistice (OCSB dă D = 1 pentru 2012 și 2020, iar d trece între 0 și 1), la producția industrială (D = 1 după OCSB pentru 2012 și 2014) '
                'și la IAPC (d = 1 pentru 2012, apoi d = 2). Canova–Hansen nu respinge sezonalitatea stabilă pentru nicio serie și niciun sfîrșit.'),
  'wx-air': ('Modelul airline dă aceeași autocorelație la lagurile 11 și 13. Arată datele pasagerilor aerieni același lucru?',
             'Nu exact: estimațiile pe 1949–1960, θ̂ = −0,402 și Θ̂ = −0,557, implică ρ₁₁ = ρ₁₃ = 0,147, iar ACF de selecție a lui w_t este 0,064 la lagul 11 și 0,152 la lagul 13. Ambele valori sînt în banda ±0,17, deci testul nu le separă de valoarea modelului. '
             'La lagurile 1 și 12 modelul implică −0,346 și −0,425, iar datele dau −0,341 și −0,387.'),
  'wx-fc': ('Din decembrie 1958, cît adaugă termenii MA la prognoza pentru ianuarie 1959 și de la ce orizont dispar?',
            '+0,0220 în logaritmi, adică circa 2,2%: prognoza este 348,6 mii, față de 341,0 fără termenii MA; valoarea reală a fost 360. Termenii MA folosesc reziduuri cel mult pînă la lagul 13, deci de la h = 14 sînt 0 și prognoza repetă tiparul pe dreaptă '
            '(februarie 1960: 354,6 față de 391 real).'),
  'air-table': ('Slide-ul dă Ljung–Box p = 0,37. Este acesta testul modelului estimat pe 1949–1958?',
                'Nu: pentru 1949–1958, p = 0,71; p = 0,37 este testul pe eșantionul complet, 1949–1960. Estimațiile depind de lungimea eșantionului: pe 1949–1952, Θ̂ = −0,904 cu SE 3,39 (aproape neidentificat), pe 1949–1958, Θ̂ = −0,541 cu SE 0,105. '
                'Cu originea în decembrie 1952, MAPE pînă în decembrie 1960 este 16,8% pentru modelul airline și 38,2% pentru metoda sezonieră naivă.'),
  'gdpm-table': ('Care modele nu trec testul Ljung–Box la 5% cu m = 8 și trece modelul ales după BIC cu m = 12?',
                 'Nu trec cele trei fără MA sezonier: (0,1,1)(1,1,0)₄ cu p = 0,003, (1,1,0)(1,1,0)₄ cu p = 0,004 și (0,1,1)(0,1,0)₄ cu p = 0,005. Modelul ales după BIC, (0,1,0)(0,1,1)₄, are p = 0,286 cu m = 8 și p = 0,526 cu m = 12. '
                 'AICc diferă cu doar 1,4 între acest model și (1,1,1)(0,1,1)₄; BIC penalizează mai mult parametrul în plus.'),
  'gdpfc-table': ('Θ = 0 înseamnă că tiparul de anul trecut se repetă. Cum se schimbă prognozele și intervalul de 95% pentru T2 2028?',
                  'Prognozele scad puțin: 52,5; 57,3; 37,6 și 45,0 mld. EUR, față de 53,1; 58,7; 38,7 și 46,1 cu Θ = −0,577; intervalul pentru T2 2028 devine 32,4–60,0, față de 37,6–57,2. '
                  'Cu Θ = −0,95 (tipar aproape fix) intervalul este 37,3–66,1: abaterea standard a reziduurilor crește cînd Θ se depărtează de estimația de verosimilitate maximă.'),
  'sa-table': ('S-au schimbat factorii sezonieri ai PIB între 2010–2014 și 2020–2025?',
               'Foarte puțin: T1 0,77 și 0,79, T2 0,93 și 0,93, T3 1,09 și 1,08, T4 1,20 și 1,20. Tiparul se mișcă încet, cum indică Θ̂ = −0,577. La comerțul cu amănuntul, factorul din ianuarie urcă de la 0,84 (2015–2019) la 0,86 (2020–2025), iar cel din decembrie rămîne 1,10.'),
  'wx-fourier': ('Pentru consumul zilnic, în ce zi din 2025 este maximă componenta Fourier anuală a DHR și cît este?',
                 'Pe 1 februarie 2025: +0,64 GW (644 MW), cu toate cele patru perechi; minimul este pe 13 mai 2025, −0,60 GW. Prima pereche are amplitudinea 0,499 GW. La turism, ajustarea cu K = 3 dă pentru august 0,765, față de efectul observat 0,847.'),
  'fourier-table': ('Ce K alege AICc pentru tiparul lunar al turismului?',
                    'K = 1 (AICc −41,5, față de −40,1 la K = 2 și −32,0 la K = 3), deși R² crește de la 0,925 la 0,989 și vîrful din august cere K ≥ 3. Cu doar 12 puncte, corecția 2k(k + 1)/(12 − k − 1) crește repede: 28 la K = 3; la K = 5 și K = 6 AICc nu mai este definit. '
                    'K = 5 dă R² = 0,999; K = 6 (11 coeficienți) trece prin toate punctele, R² = 1.'),
  'wx-easter': ('În 2027 Paștele este pe 2 mai. Cum se împarte regresorul cu w = 10?',
                'Fereastra merge de la 22 aprilie la 1 mai: 9 zile în aprilie (E = 0,9) și 1 zi în mai (E = 0,1). Cu β̂_E = 5,95% pentru alimente, efectul este +5,35% în aprilie și +0,59% în mai. În 2024 (5 mai): 0,6 în aprilie și 0,4 în mai; în 2025 (20 aprilie): 1 în aprilie.'),
  'easter-table': ('Depinde efectul Paștelui asupra comerțului cu alimente de fereastra w?',
                   'Da: cu w = 5, β̂_E = 3,0% (SE 1,8, t = 1,7), nesemnificativ la 5%; cu w = 10, 5,9% (SE 2,3, t = 2,6); cu w = 20, 5,4% (SE 1,4, t = 3,8). Pentru tot comerțul, efectul rămîne mic (−0,1%; 0,8%; 1,5%), în timp ce zilele lucrătoare contează la orice w (circa +1% pe zi).'),
  'dhr-table': ('Ce dă partea deterministă a DHR pentru Lunea Paștelui, 21 aprilie 2025, față de o duminică la media anuală?',
                '−0,145 GW: efectul zilei de luni, +0,884, plus componenta anuală, −0,448, plus variabila dummy a Paștelui, −0,581. O luni obișnuită din aceeași perioadă ar avea circa +0,44 GW. Nivelul seriei vine din erorile ARIMA(1,1,1), cu AR 0,80 și MA −0,98.'),
  'dm-h': ('Global, DHR bate metoda sezonieră naivă (HLN −4,42). La ce orizonturi respinge testul făcut orizont cu orizont?',
           'Doar la h = 8 și h = 10–14 (|HLN| > 2,06); la h = 1–7 și h = 9 nu. O singură zi pentru fiecare origine are mai mult zgomot decît media pe 14 zile. DHR față de Prophet: respins la h = 1, 2 și 12–14.'),
  'wx-dm': ('Cu cele 26 de ferestre reale, DHR față de metoda sezonieră naivă: cît sînt DM și HLN și de ce diferă puțin?',
            'DM = −4,51, HLN = −4,42, p < 0,001; DHR are pierderea mai mică în 22 din 26 de ferestre (d̄ = −0,157 GW²). Corecția HLN este √(25/26) = 0,981, deci contează puțin la n = 26; în exemplul cu cinci ferestre este √(4/5) = 0,894, '
            'iar valoarea critică t(4) = 2,78 în loc de 1,96 schimbă decizia.'),
  'cv-table': ('Fără cele 8 ferestre cu sărbători, rămîne DHR cea mai bună și ce se întîmplă în ferestrele cu sărbători?',
               'Fără sărbători (18 ferestre), DHR are MAE 258 MW și MASE 0,83, iar combinarea 0,94. În cele 8 ferestre cu sărbători nicio metodă nu are MASE sub 1: TBATS 317 MW, DHR 318 MW, combinarea 340 MW. '
               'Pe orizonturile 8–14 nici DHR nu mai are MASE sub 1 (1,02).'),
  'gdpcv-table': ('Fără 2019–2020, SARIMA are MAE 1,89 la h = 4, față de 2,44 pentru ETS. Este diferența semnificativă?',
                  'Nu: HLN = −1,11 (p = 0,275) la h = 4 și −1,47 (p = 0,148) la h = 1, cu 43 de origini. Cu toate cele 51 de origini, la h = 4 SARIMA are chiar MAE 2,70 față de 2,84, iar HLN = 0,50 (p = 0,620): pierderea pătratică a SARIMA este mai mare în anii pandemiei.'),
  'wx-comb': ('Cu erorile reale ale DHR și Prophet, bate media cu ponderi egale DHR singur?',
              'Nu: σ₁ = 348 MW, σ₂ = 411 MW, ρ = 0,75; varianța combinării egale este 126 036 MW², peste σ₁² = 120 854. Ponderea optimă este w* = 0,82 (MSE 118 370), aproape de DHR singur. În MAE, combinarea egală are 283 MW, DHR singur 276. '
              'Erorile puternic corelate lasă puțin loc pentru cîștig, ca în exemplul din slide (1,060 față de 1).'),
  'comb-table': ('Bat ponderile estimate din ferestrele anterioare ponderile egale pentru cele cinci modele?',
                 'Puțin: MAE 299 MW, față de 305 cu ponderi egale; ponderile alese a posteriori dau 296, iar DHR singur 276. Pentru DHR și Prophet: 283 cu ponderi egale, 281 din ferestrele anterioare, 274 cu cea mai bună pondere a posteriori (w = 0,86). '
                 'Cîștigul ponderilor estimate este mic, cum spune „forecast combination puzzle”.'),
 },
 'en': {
  'wx-sdiff': ('In 2026Q2, Δy = 17.9 while Δ₄y = −1.96. Did GDP grow?',
               'No: 17.9 is the seasonal jump from Q1 to Q2 (the GDP factors NSA/SCA are 0.78 in Q1 and 0.93 in Q2). The comparison with the same quarter removes the season: GDP in 2026Q2, 45.88 bn EUR, is 1.96 log points below 2025Q2, 46.79 bn EUR. '
               'ΔΔ₄y = −0.46 shows that the annual fall deepened, from −1.50 in 2026Q1 to −1.96.'),
  'wx-roots': ('For monthly data (s = 12), what is the period of the root k = 1, and which real factor does it give with its conjugate?',
               'Period 12/1 = 12 months, the annual cycle: z₁ = 0.866 + 0.500i, and the pair (k = 1, k = 11) gives the factor 1 − 1.732z + z². For s = 12 there are 12 roots: two real ones (1 and −1, period 2 months) and five complex-conjugate pairs, with periods 12, 6, 4, 3 and 2.4 months.'),
  'hegy-table': ('With 0 lags of Δ₄y instead of 1 (chosen by AIC), on 2000–2019, are all three nulls still not rejected?',
                 'No: with k = 0, t(π₂) = −3.34 < −2.77 and F(π₃, π₄) = 17.25 > 6.64, so the roots −1 and ±i are rejected; only the root 1 remains (t(π₁) = −1.46). With k = 1, chosen by AIC, no null is rejected (−1.97, −2.19, 6.12). '
                 'The autocorrelation left in the residuals of the regression without lags changes the decision. The sample end matters too: up to 2025Q4, with k = 1, t(π₂) = −2.88 and F = 8.38 reject two nulls.'),
  'sur-table': ('For which series does the decision change when the sample end moves between 2012 and 2025?',
                'Not for GDP: OCSB gives D = 1, Canova–Hansen D = 0 and KPSS d = 1 for every end. It changes for tourism nights (OCSB gives D = 1 for 2012 and 2020, and d moves between 0 and 1), for industrial production (D = 1 by OCSB for 2012 and 2014) '
                'and for the HICP (d = 1 for 2012, then d = 2). Canova–Hansen rejects stable seasonality for no series and no end.'),
  'wx-air': ('The airline model gives the same autocorrelation at lags 11 and 13. Do the airline passenger data show this?',
             'Not exactly: the 1949–1960 estimates, θ̂ = −0.402 and Θ̂ = −0.557, imply ρ₁₁ = ρ₁₃ = 0.147, while the sample ACF of w_t is 0.064 at lag 11 and 0.152 at lag 13. Both values lie inside the band ±0.17, so the test does not separate them from the model value. '
             'At lags 1 and 12 the model implies −0.346 and −0.425, and the data give −0.341 and −0.387.'),
  'wx-fc': ('From December 1958, how much do the MA terms add to the January 1959 forecast, and from which horizon do they vanish?',
            '+0.0220 in logs, about 2.2%: the forecast is 348.6 thousand, against 341.0 without the MA terms; the actual value was 360. The MA terms use residuals up to lag 13 at most, so from h = 14 they are 0 and the forecast repeats the pattern on the line '
            '(February 1960: 354.6 against 391 actual).'),
  'air-table': ('The slide gives Ljung–Box p = 0.37. Is this the test of the model estimated on 1949–1958?',
                'No: for 1949–1958, p = 0.71; p = 0.37 is the test on the full sample, 1949–1960. The estimates depend on the sample length: on 1949–1952, Θ̂ = −0.904 with SE 3.39 (almost unidentified); on 1949–1958, Θ̂ = −0.541 with SE 0.105. '
                'With the origin in December 1952, the MAPE to December 1960 is 16.8% for the airline model and 38.2% for the seasonal naive method.'),
  'gdpm-table': ('Which models fail the Ljung–Box test at 5% with m = 8, and does the BIC model pass with m = 12?',
                 'The three without a seasonal MA fail: (0,1,1)(1,1,0)₄ with p = 0.003, (1,1,0)(1,1,0)₄ with p = 0.004 and (0,1,1)(0,1,0)₄ with p = 0.005. The BIC model, (0,1,0)(0,1,1)₄, has p = 0.286 with m = 8 and p = 0.526 with m = 12. '
                 'AICc differs by only 1.4 between this model and (1,1,1)(0,1,1)₄; BIC penalises the extra parameter more.'),
  'gdpfc-table': ('Θ = 0 means that last year\'s pattern repeats. How do the forecasts and the 95% interval for 2028Q2 change?',
                  'The forecasts fall a little: 52.5, 57.3, 37.6 and 45.0 bn EUR, against 53.1, 58.7, 38.7 and 46.1 with Θ = −0.577; the 2028Q2 interval becomes 32.4–60.0, against 37.6–57.2. '
                  'With Θ = −0.95 (an almost fixed pattern) the interval is 37.3–66.1: the residual standard deviation rises when Θ moves away from the maximum-likelihood estimate.'),
  'sa-table': ('Did the GDP seasonal factors change between 2010–2014 and 2020–2025?',
               'Very little: Q1 0.77 and 0.79, Q2 0.93 and 0.93, Q3 1.09 and 1.08, Q4 1.20 and 1.20. The pattern moves slowly, as Θ̂ = −0.577 indicates. For retail trade, the January factor rises from 0.84 (2015–2019) to 0.86 (2020–2025), and the December one stays at 1.10.'),
  'wx-fourier': ('For daily load, on which day of 2025 is the annual Fourier component of the DHR highest, and how large is it?',
                 'On 1 February 2025: +0.64 GW (644 MW), with all four pairs; the lowest is on 13 May 2025, −0.60 GW. The first pair has an amplitude of 0.499 GW. For tourism, the K = 3 fit gives 0.765 for August, against the observed effect of 0.847.'),
  'fourier-table': ('Which K does AICc choose for the monthly tourism pattern?',
                    'K = 1 (AICc −41.5, against −40.1 at K = 2 and −32.0 at K = 3), although R² rises from 0.925 to 0.989 and the August peak needs K ≥ 3. With only 12 points the correction 2k(k + 1)/(12 − k − 1) grows fast: 28 at K = 3; at K = 5 and K = 6 AICc is no longer defined. '
                    'K = 5 gives R² = 0.999; K = 6 (11 coefficients) passes through every point, R² = 1.'),
  'wx-easter': ('In 2027 Easter falls on 2 May. How is the regressor split with w = 10?',
                'The window runs from 22 April to 1 May: 9 days in April (E = 0.9) and 1 day in May (E = 0.1). With β̂_E = 5.95% for food, the effect is +5.35% in April and +0.59% in May. In 2024 (5 May): 0.6 in April and 0.4 in May; in 2025 (20 April): 1 in April.'),
  'easter-table': ('Does the Easter effect on food retail depend on the window w?',
                   'Yes: with w = 5, β̂_E = 3.0% (SE 1.8, t = 1.7), not significant at 5%; with w = 10, 5.9% (SE 2.3, t = 2.6); with w = 20, 5.4% (SE 1.4, t = 3.8). For all retail the effect stays small (−0.1%, 0.8%, 1.5%), while working days matter at every w (about +1% per day).'),
  'dhr-table': ('What does the deterministic part of the DHR give for Easter Monday, 21 April 2025, relative to a Sunday at the annual mean?',
                '−0.145 GW: the Monday effect, +0.884, plus the annual component, −0.448, plus the Easter dummy, −0.581. An ordinary Monday at the same time of year would have about +0.44 GW. The level of the series comes from the ARIMA(1,1,1) errors, with AR 0.80 and MA −0.98.'),
  'dm-h': ('Overall, DHR beats the seasonal naive method (HLN −4.42). At which horizons does the horizon-by-horizon test reject?',
           'Only at h = 8 and h = 10–14 (|HLN| > 2.06); not at h = 1–7 and h = 9. A single day per origin is noisier than the 14-day average. DHR against Prophet: rejected at h = 1, 2 and 12–14.'),
  'wx-dm': ('With the 26 real windows, DHR against the seasonal naive method: what are DM and HLN, and why do they differ only a little?',
            'DM = −4.51, HLN = −4.42, p < 0.001; DHR has the smaller loss in 22 of the 26 windows (d̄ = −0.157 GW²). The HLN factor is √(25/26) = 0.981, so it matters little at n = 26; in the five-window example it is √(4/5) = 0.894, '
            'and the critical value t(4) = 2.78 instead of 1.96 changes the decision.'),
  'cv-table': ('Without the 8 windows that contain holidays, is DHR still the best, and what happens in the holiday windows?',
               'Without holidays (18 windows), DHR has MAE 258 MW and MASE 0.83, and the combination 0.94. In the 8 holiday windows no method has MASE below 1: TBATS 317 MW, DHR 318 MW, the combination 340 MW. '
               'Over horizons 8–14 not even DHR has MASE below 1 (1.02).'),
  'gdpcv-table': ('Without 2019–2020, SARIMA has MAE 1.89 at h = 4, against 2.44 for ETS. Is the difference significant?',
                  'No: HLN = −1.11 (p = 0.275) at h = 4 and −1.47 (p = 0.148) at h = 1, with 43 origins. With all 51 origins, at h = 4 SARIMA even has MAE 2.70 against 2.84, and HLN = 0.50 (p = 0.620): the squared loss of SARIMA is larger in the pandemic years.'),
  'wx-comb': ('With the real errors of DHR and Prophet, does the equal-weight average beat DHR alone?',
              'No: σ₁ = 348 MW, σ₂ = 411 MW, ρ = 0.75; the variance of the equal combination is 126,036 MW², above σ₁² = 120,854. The best weight is w* = 0.82 (MSE 118,370), close to DHR alone. In MAE, the equal combination has 283 MW, DHR alone 276. '
              'Strongly correlated errors leave little room for a gain, as in the slide example (1.060 against 1).'),
  'comb-table': ('Do the weights estimated on the earlier windows beat equal weights for the five models?',
                 'A little: MAE 299 MW, against 305 with equal weights; weights chosen with hindsight give 296, and DHR alone 276. For DHR and Prophet: 283 with equal weights, 281 from the earlier windows, 274 with the best weight in hindsight (w = 0.86). '
                 'The gain of estimated weights is small, as the forecast combination puzzle says.'),
 },
}
for _lg in QA:
    QA[_lg].update(QX[_lg])


# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with the readout of the block at the state the question names (verified by a node script).
PREDICT = {
 'series': (1, [('În octombrie: 0,49%.', 'In October: 0.49%.'), ('În ianuarie: 0,575%.', 'In January: 0.575%.'), ('În iunie, luna alimentelor proaspete.', 'In June, the month of fresh food.')]),
 'seasonal-plot': (2, [('În 2019, anul cu cele mai multe înnoptări în august.', 'In 2019, the year with the most nights in August.'),
                       ('În 2020, anul restricțiilor.', 'In 2020, the year of the restrictions.'), ('În 2021: 5,3.', 'In 2021: 5.3.')]),
 'unit-circle': (0, [('Una: z = 1.', 'One: z = 1.'), ('Două: z = 1 și z = −1, ca pentru s = 4 și s = 12.', 'Two: z = 1 and z = −1, as for s = 4 and s = 12.'), ('Șapte: toate rădăcinile unei diferențe sînt reale.', 'Seven: all roots of a difference are real.')]),
 'gdp-diff': (1, [('Nu: valoarea negativă vine din scăderea din 2020.', 'No: the negative value comes from the 2020 drop.'), ('Da: −0,34, în afara benzii ±0,23.', 'Yes: −0.34, outside the ±0.23 band.'),
                  ('Devine pozitivă: fără pandemie, tiparul sezonier este fix.', 'It turns positive: without the pandemic the seasonal pattern is fixed.')]),
 'theory-acf': (2, [('Rămîn 0,152: sateliții depind doar de Θ.', 'They stay at 0.152: the satellites depend on Θ only.'), ('Dispar: sateliții apar doar pentru θ negativ.', 'They vanish: satellites appear only for negative θ.'),
                    ('Își schimbă semnul: −0,152.', 'They change sign: −0.152.')]),
 'airline': (1, [('La fel ca pentru ΔΔ₁₂ ln y: valori la lagurile 1 și 12.', 'The same as for ΔΔ₁₂ ln y: values at lags 1 and 12.'), ('Autocorelații pozitive care scad încet: 0,71 la lagul 1.', 'Positive autocorrelations that decay slowly: 0.71 at lag 1.'),
                 ('O singură valoare, la lagul 12: diferența sezonieră lasă doar partea sezonieră.', 'A single value, at lag 12: the seasonal difference leaves only the seasonal part.')]),
 'airline-fc': (2, [('Aproximativ aceeași, 239: modelul are aceiași doi parametri.', 'About the same, 239: the model has the same two parameters.'),
                    ('Mai mică: estimarea pe un eșantion scurt este mai prudentă.', 'Smaller: estimation on a short sample is more cautious.'), ('Mult mai mare: 852.', 'Much larger: 852.')]),
 'gdp-diag': (1, [('Da: orice termen sezonier elimină autocorelația de la lagul 4.', 'Yes: any seasonal term removes the autocorrelation at lag 4.'), ('Nu: Q*(8) = 19,5, p = 0,003.', 'No: Q*(8) = 19.5, p = 0.003.'),
                  ('Da, pentru că AICc este mai mic decît pentru modelul BIC.', 'Yes, because its AICc is lower than that of the BIC model.')]),
 'gdp-fc': (0, [('Aceeași în fiecare trimestru: −1,96%, ultima creștere anuală observată.', 'The same in every quarter: −1.96%, the last observed year-on-year growth.'),
                ('Zero: fără termen MA, prognoza repetă nivelul de anul trecut.', 'Zero: without the MA term the forecast repeats last year’s level.'),
                ('Media creșterilor anuale din 2000–2026.', 'The mean of the year-on-year growth rates of 2000–2026.')]),
 'sa-nsa': (1, [('A scăzut și ea cu aproape 40%: o recesiune.', 'It fell by almost 40% too: a recession.'), ('Cu −0,06%: practic deloc.', 'By −0.06%: practically not at all.'),
                ('A crescut cu 42%: ajustarea inversează semnul sezonului.', 'It rose by 42%: adjustment reverses the sign of the season.')]),
 'fourier': (1, [('Unsprezece coeficienți: altfel tiparul nu este reprodus.', 'Eleven coefficients: otherwise the pattern is not reproduced.'), ('Trei armonici, șase coeficienți: R² = 0,989.', 'Three harmonics, six coefficients: R² = 0.989.'),
                 ('O singură armonică: R² = 0,925 ajunge.', 'One harmonic: R² = 0.925 is enough.')]),
 'easter': (1, [('Crește: cumpărăturile se concentrează în ultimele zile.', 'It rises: shopping concentrates in the last days.'), ('Scade la 1,6% și nu se mai separă de zgomot.', 'It falls to 1.6% and is no longer separated from noise.'),
                ('Rămîne 5,9%: efectul nu depinde de fereastră.', 'It stays at 5.9%: the effect does not depend on the window.')]),
 'load-hourly': (1, [('La lagul 24: ieri este mai aproape în timp decît săptămîna trecută.', 'At lag 24: yesterday is closer in time than last week.'), ('La lagul 168: 0,817 față de 0,811.', 'At lag 168: 0.817 against 0.811.'),
                     ('Sînt egale: ambele laguri sînt multipli de 24.', 'They are equal: both lags are multiples of 24.')]),
 'mstl': (2, [('Tot 2%: componenta neregulată nu depinde de fereastră.', 'Still 2%: the remainder does not depend on the window.'), ('Zero: MSTL include sărbătorile în componenta săptămînală.', 'Zero: MSTL puts the holidays in the weekly component.'),
              ('13%.', '13%.')]),
 'load-daily': (1, [('Duminica Paștelui, 16 aprilie 2023.', 'Easter Sunday, 16 April 2023.'), ('Duminica Rusaliilor, 4 iunie 2023.', 'Pentecost Sunday, 4 June 2023.'), ('Ziua de Crăciun, 25 decembrie 2023.', 'Christmas Day, 25 December 2023.')]),
 'prophet': (2, [('Rămîne aceeași: sărbătorile sînt doar cîteva zile pe an.', 'It stays the same: holidays are only a few days a year.'), ('Scade: un model cu mai puțini parametri are erori mai mici.', 'It falls: a model with fewer parameters has smaller errors.'),
                 ('Crește de la 197 la 226 MW.', 'It rises from 197 to 226 MW.')]),
 'cv-scheme': (1, [('Într-una singură: originile sînt diferite.', 'In one only: the origins are different.'), ('În trei.', 'In three.'), ('În șase: în toate ferestrele.', 'In six: in all windows.')]),
 'load-fc': (1, [('DHR, singurul cu o variabilă dummy pentru Paște.', 'DHR, the only one with an Easter dummy.'), ('ETS: 299 MW.', 'ETS: 299 MW.'), ('Metoda sezonieră naivă: copiază săptămîna anterioară.', 'The seasonal naive method: it copies the previous week.')]),
 'cv-results': (2, [('Da, cu un avantaj și mai mare: sărbătorile sînt punctul ei forte.', 'Yes, by an even wider margin: holidays are its strength.'), ('Nu: metoda sezonieră naivă este cea mai bună în aceste ferestre.', 'No: the seasonal naive method is best in these windows.'),
                    ('Nu: TBATS 317 și DHR 318 MW sînt la egalitate.', 'No: TBATS 317 and DHR 318 MW are level.')]),
 'combination': (0, [('295 MW: sub ambele.', '295 MW: below both.'), ('Aproximativ 355 MW: media celor două MAE.', 'About 355 MW: the mean of the two MAEs.'), ('Peste 381 MW: combinarea moștenește erorile ambelor.', 'Above 381 MW: the combination inherits the errors of both.')]),
}

# blocks whose initial charts show the answer: the live charts are veiled too until „Verifică” / "Check" or a change of the controls
VEIL_CHARTS = ('seasonal-plot', 'sa-nsa', 'fourier', 'load-hourly', 'load-fc')

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slide „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('Cîți parametri are SARIMA(1,1,1)(0,1,1)₁₂ și la ce laguri apare ε?', 'How many parameters does SARIMA(1,1,1)(0,1,1)₁₂ have, and at which lags does ε appear?'),
      key=1, opts=[
  (('Trei (φ, θ, Θ); ε la lagurile 0, 1 și 12.', 'Three (φ, θ, Θ); ε at lags 0, 1 and 12.'),
   ('Lipsesc σ² și termenul de la lagul 13: produsul (1 + θL)(1 + ΘL¹²) conține θΘL¹³.', 'σ² and the lag-13 term are missing: the product (1 + θL)(1 + ΘL¹²) contains θΘL¹³.')),
  (('Patru (φ, θ, Θ, σ²); ε la lagurile 0, 1, 12 și 13.', 'Four (φ, θ, Θ, σ²); ε at lags 0, 1, 12 and 13.'),
   ('Coeficientul de la lagul 13 este θΘ, produsul a doi parametri deja estimați.', 'The coefficient at lag 13 is θΘ, the product of two parameters already estimated.')),
  (('Cinci: θΘ este un parametru separat.', 'Five: θΘ is a separate parameter.'),
   ('θΘ nu se estimează separat: structura multiplicativă îl fixează.', 'θΘ is not estimated separately: the multiplicative structure fixes it.'))]),
 dict(q=('PIB-ul neajustat scade cu 40% din T4 în T1. Este o recesiune?', 'Unadjusted GDP falls by 40% from Q4 to Q1. Is it a recession?'),
      key=1, opts=[
  (('Da: o scădere de 40% într-un trimestru este o recesiune profundă.', 'Yes: a 40% fall in one quarter is a deep recession.'),
   ('Scăderea din T1 se repetă în fiecare an: în T1 2026, −42,1% neajustat, dar −0,06% în seria SCA.', 'The Q1 fall repeats every year: in 2026Q1, −42.1% unadjusted but −0.06% in the SCA series.')),
  (('Nu: este tiparul sezonier; priviți seria SCA sau creșterea față de anul anterior.', 'No: it is the seasonal pattern; look at the SCA series or the year-on-year growth.'),
   ('Ambele elimină sezonul: SCA prin ajustare, creșterea anuală prin compararea aceluiași trimestru.', 'Both remove the season: SCA by adjustment, year-on-year growth by comparing the same quarter.')),
  (('Da, dacă ajustarea sezonieră doar netezește scăderea.', 'Yes, if seasonal adjustment only smooths the fall.'),
   ('Ajustarea elimină componenta sezonieră, nu o netezește: factorul T1 este 0,79, deci scăderea dispare în SCA.', 'Adjustment removes the seasonal component rather than smoothing it: the Q1 factor is 0.79, so the fall disappears in SCA.'))]),
 dict(q=('Un test DM cu 26 de ferestre dă HLN = −1,93. Este prima prognoză semnificativ mai bună la 5%?', 'A DM test with 26 windows gives HLN = −1.93. Is the first forecast significantly better at 5%?'),
      key=2, opts=[
  (('Da: |−1,93| > 1,645.', 'Yes: |−1.93| > 1.645.'),
   ('1,645 este cuantila de 95% a distribuției Normale, pentru un test unilateral; testul DM este bilateral, iar HLN se compară cu t(25).', '1.645 is the 95% quantile of the Normal distribution, for a one-sided test; the DM test is two-sided, and HLN is compared with t(25).')),
  (('Da: statistica este negativă, deci prima prognoză are pierderi mai mici.', 'Yes: the statistic is negative, so the first forecast has smaller losses.'),
   ('Semnul arată direcția, nu semnificația: o diferență medie negativă poate fi zgomot.', 'The sign gives the direction, not the significance: a negative mean difference can be noise.')),
  (('Nu: |−1,93| < t₀,₉₇₅(25) = 2,06.', 'No: |−1.93| < t₀.₉₇₅(25) = 2.06.'),
   ('Prima prognoză este mai bună în medie, dar diferența nu se separă de zgomot la 5% (p ≈ 0,065).', 'The first forecast is better on average, but the difference is not separated from noise at 5% (p ≈ 0.065).'))]),
]


def _predict_js():
    import json
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {}))
              for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch4-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH4_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH4_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n')


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
