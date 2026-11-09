"""Build the interactive pages of Chapter 5 (conditional volatility: ARCH and GARCH): interactive/ch5.html (EN) and interactive/ch5_ro.html (RO).

One block per chart slide of the lecture body, in the order of the slides: the slide on the left, its
interactive version on the right. Logic in interactive/ch5.js, inputs in interactive/ch5_data.js
(Quantlets/Ch_05/export_interactive.py), slide images in interactive/slides/.
"""
from interactive_common import slider, select, play, charts

N = 5

QL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_05'
TEX = {'ro': 'RO/Cursuri/capitol5_volatilitate_conditionata_garch', 'en': 'EN/Courses/chapter5_conditional_volatility_garch'}

# (anchor, chart file, section key); order of the slides
BLOCKS = [('returns', 'tsa_ch5_returns', 'sty'), ('acf-squares', 'tsa_ch5_acf_squares', 'sty'),
          ('arma-resid', 'tsa_ch5_arma_resid', 'cm'),
          ('simulated', 'tsa_ch5_simulated', 'arch'),
          ('ewma-garch', 'tsa_ch5_ewma_garch', 'garch'),
          ('lik-arch1', 'tsa_ch5_lik_arch1', 'ml'), ('lik-garch', 'tsa_ch5_lik_garch', 'ml'), ('qq', 'tsa_ch5_qq', 'ml'),
          ('vol-sp500-bet', 'tsa_ch5_vol_sp500_bet', 'mk'), ('vol-eurron-btc', 'tsa_ch5_vol_eurron_btc', 'mk'),
          ('persistence', 'tsa_ch5_persistence', 'mk'), ('bands', 'tsa_ch5_bands', 'mk'),
          ('nic', 'tsa_ch5_nic', 'asym'),
          ('acf-diag', 'tsa_ch5_acf_diag', 'diag'),
          ('term-structure', 'tsa_ch5_term_structure', 'fc'), ('forecast-eval', 'tsa_ch5_forecast_eval', 'fc'),
          ('var', 'tsa_ch5_var', 'var')]

PFX = {'returns': 're', 'acf-squares': 'aq', 'arma-resid': 'ar', 'simulated': 'si', 'ewma-garch': 'ew', 'lik-arch1': 'l1', 'lik-garch': 'lg',
       'qq': 'qq', 'vol-sp500-bet': 'v1', 'vol-eurron-btc': 'v2', 'persistence': 'pe', 'bands': 'bd', 'nic': 'ni', 'acf-diag': 'dg',
       'term-structure': 'ts', 'forecast-eval': 'fe', 'var': 'va'}

SECTIONS = {
 'ro': dict(sty='Faptele stilizate ale randamentelor financiare', cm='Media condiționată și varianța condiționată', arch='Modelul ARCH',
            garch='Modelul GARCH(1,1)', ml='Estimarea prin verosimilitate maximă', mk='GARCH pe patru piețe și modele ARMA-GARCH',
            asym='Asimetria și efectul de levier', diag='Diagnosticare și alegerea modelului', fc='Prognoza varianței și evaluarea ei',
            var='O aplicație: VaR 1% dintr-un model GARCH'),
 'en': dict(sty='Stylised facts of financial returns', cm='Conditional mean and conditional variance', arch='The ARCH model',
            garch='The GARCH(1,1) model', ml='Estimation by maximum likelihood', mk='GARCH in four markets and ARMA-GARCH models',
            asym='Asymmetry and the leverage effect', diag='Diagnostics and model choice', fc='Variance forecasts and their evaluation',
            var='An application: VaR 1% from a GARCH model'),
}

L = {
 'ro': dict(file='ch5_ro.html', html_lang='ro', site='../index_ro.html#chapters', course='Serii de timp',
   h1='Capitolul 5 · Volatilitate condiționată: modelele ARCH și GARCH', back='Site-ul cursului', pdf_btn='PDF-ul cursului', ql='Quantlets',
   slide='Slide-ul', ask='Întrebare pentru sală', answer='Răspuns',
   note='În stînga este slide-ul din curs, în dreapta aceeași figură, calculată în browser din aceleași date. Butonul ▶ modifică parametrul pas cu pas.',
   src='Randamente logaritmice zilnice, în %, pînă la 18 septembrie 2026: S&P 500, BET și Bitcoin din datele de piață ale cursului (EODHD), '
       'EUR/RON din cursul de referință al BNR. Estimările GARCH, GJR și EGARCH sînt cele din curs; volatilitățile condiționate se recalculează în browser '
       'din parametrii estimați. Variantele cu alți parametri decît cei estimați (α, β, λ, nivelul VaR, acoperirea intervalelor) folosesc parametri '
       'fixați, fără reestimare. Seriile simulate folosesc extragerile Normale ale seminței capitolului; variantele care nu apar în curs '
       '(alte valori ale lui α, β sau n) sînt rulări suplimentare ale codului capitolului, cu aceeași sămînță.'),
 'en': dict(file='ch5.html', html_lang='en', site='../index.html#chapters', course='Time Series Analysis',
   h1='Chapter 5 · Conditional Volatility: ARCH and GARCH Models', back='Course site', pdf_btn='Lecture PDF', ql='Quantlets',
   slide='Slide', ask='Question for the room', answer='Answer',
   note='The slide of the lecture is on the left, the same figure on the right, computed in the browser from the same data. The ▶ button moves the parameter step by step.',
   src='Daily log returns in %, up to 18 September 2026: S&P 500, BET and Bitcoin from the market data of the course (EODHD), '
       'EUR/RON from the BNR reference rate. The GARCH, GJR and EGARCH estimates are those of the lecture; the conditional volatilities are recomputed in '
       'the browser from the estimated parameters. Variants with parameters other than the estimates (α, β, λ, the VaR level, the interval coverage) '
       'use fixed parameters, not re-estimated. The simulated series use the Normal draws of the chapter’s seed; variants that do not appear in the '
       'lecture (other values of α, β or n) are extra runs of the chapter code with the same seed.'),
}


def widgets(lang):
    ro = lang == 'ro'
    s = lambda a, b: a if ro else b   # noqa: E731
    lec = s(' (curs)', ' (lecture)')
    keep = [('keep', s('inclusă', 'included') + lec), ('drop', s('exclusă', 'excluded'))]
    markets = [('sp500', 'S&P 500'), ('bet', 'BET'), ('eurron', 'EUR/RON'), ('btc', 'Bitcoin')]
    return {
     'returns': (slider('re-y', s('Anul evidențiat', 'Highlighted year'), 2000, 2026, 1, 2017) + play('re'), charts('re-chart')),
     'acf-squares': (select('aq-f', s('A doua serie', 'Second series'), [('sq', 'r²' + lec), ('abs', '|r|')], 'sq')
                     + slider('aq-y', s('Eșantionul începe în', 'Sample starts in'), 2000, 2020, 1, 2000) + play('aq'), charts('aq-chart')),
     'arma-resid': (slider('ar-p', s('Ordinul p al AR pentru medie (ambele panouri)', 'Order p of the AR for the mean (both panels)'), 0, 5, 1, 1) + play('ar'), charts('ar-chart')),
     'simulated': (slider('si-a1', s('α al ARCH(1), ω = 1 − α', 'α of ARCH(1), ω = 1 − α'), 0.05, 0.9, 0.05, 0.5)
                   + slider('si-a', s('α al GARCH(1,1)', 'α of GARCH(1,1)'), 0.02, 0.3, 0.02, 0.1)
                   + slider('si-b', s('β al GARCH(1,1), ω = 1 − α − β', 'β of GARCH(1,1), ω = 1 − α − β'), 0, 0.96, 0.04, 0.88) + play('si'), charts('si-chart')),
     'ewma-garch': (slider('ew-l', s('λ al EWMA', 'λ of EWMA'), 0.8, 0.99, 0.01, 0.94)
                    + slider('ew-w', s('Fereastra mobilă (zile)', 'Rolling window (days)'), 21, 252, 21, 63) + play('ew'), charts('ew-chart')),
     'lik-arch1': (slider('l1-a', s('α adevărat (ω = 1 − α)', 'True α (ω = 1 − α)'), 0.1, 0.9, 0.05, 0.5)
                   + select('l1-n', s('A doua selecție', 'Second sample'), [(v, 'n = ' + v + (lec if v == '1000' else '')) for v in ('250', '500', '1000', '2000', '5000')], '1000')
                   + play('l1'), charts('l1-chart')),
     'lik-garch': (slider('lg-n', s('n în panoul din dreapta', 'n in the right panel'), 250, 5000, 250, 2000) + play('lg'), charts('lg-chart')),
     'qq': (slider('qq-nu', s('ν al distribuției t de referință (ambele panouri)', 'ν of the reference t distribution (both panels)'), 3, 30, 0.01, 6.26) + play('qq'), charts('qq-chart')),
     'vol-sp500-bet': (slider('v1-c', s('α = c·α̂ și β = α̂ + β̂ − α; c', 'α = c·α̂ and β = α̂ + β̂ − α; c'), 0.25, 2, 0.05, 1) + play('v1'), charts('v1-chart')),
     'vol-eurron-btc': (slider('v2-c', s('α = c·α̂ și β = α̂ + β̂ − α; c', 'α = c·α̂ and β = α̂ + β̂ − α; c'), 0.25, 2, 0.05, 1) + play('v2'), charts('v2-chart')),
     'persistence': (select('pe-d', s('α + β în timpul de înjumătățire', 'α + β in the half-life'),
                            [('x', s('nerotunjit', 'not rounded') + lec), ('3', s('rotunjit la 3 zecimale', 'rounded to 3 decimals')), ('2', s('rotunjit la 2 zecimale', 'rounded to 2 decimals'))], 'x')
                     + slider('pe-p', s('α + β al curbei întrerupte', 'α + β of the dashed curve'), 0.9, 0.999, 0.001, 0.98) + play('pe'), charts('pe-chart')),
     'bands': (slider('bd-c', s('Acoperirea intervalelor', 'Coverage of the intervals'), 80, 99, 1, 95) + play('bd'), charts('bd-chart')),
     'nic': (select('ni-k', s('Panoul din dreapta', 'Right panel'), [('btc', 'Bitcoin' + lec), ('bet', 'BET')], 'btc')
             + slider('ni-x', s('Mărimea șocului |ε|', 'Size of the shock |ε|'), 1, 4, 0.5, 2)
             + slider('ni-h', s('Banda kernel, în abateri standard', 'Kernel bandwidth, in standard deviations'), 0.25, 1.5, 0.05, 0.5) + play('ni'), charts('ni-chart')),
     'acf-diag': (select('dg-k', s('Seria', 'Series'), [(k, t + (lec if k == 'sp500' else '')) for k, t in markets], 'sp500')
                  + select('dg-o', s('Ziua cu cel mai mare |ẑ|', 'The day with the largest |ẑ|'), keep, 'keep') + play('dg'), charts('dg-chart')),
     'term-structure': (slider('ts-o', s('Originea celei de-a treia prognoze', 'Origin of the third forecast'), 0, 1, 1, 1) + play('ts'), charts('ts-chart')),
     'forecast-eval': (select('fe-m', s('Comparația', 'Comparison'), [('ge', 'EWMA − GARCH-t' + lec), ('je', 'EWMA − GJR-t'), ('gj', 'GARCH-t − GJR-t')], 'ge')
                       + select('fe-x', s('Ziua cu cea mai mare diferență', 'The day with the largest difference'), keep, 'keep') + play('fe'), charts('fe-chart')),
     'var': (select('va-k', s('Seria', 'Series'), [(k, t + (lec if k == 'sp500' else '')) for k, t in markets], 'sp500')
             + select('va-a', s('Nivelul VaR', 'VaR level'), [('0.5', s('0,5%', '0.5%')), ('1', '1%' + lec), ('2.5', s('2,5%', '2.5%')), ('5', '5%')], '1')
             + select('va-w', s('Perioada', 'Period'), [('covid', s('iulie 2019 – iunie 2021', 'July 2019 – June 2021') + lec), ('all', '2015–2026')], 'covid')
             + play('va'), charts('va-chart')),
    }


QA = {
 'ro': {
  'returns': ('În ce an a fost cea mai mică abaterea standard zilnică a randamentelor S&P 500?',
              'În 2017: 0,42%, urmat de 2006 (0,63%) și 2005 (0,65%). Între septembrie 2008 și martie 2009 abaterea standard a fost 3,52%, de 8,3 ori mai mare, '
              'iar în februarie–mai 2020, 3,70% (de 8,8 ori). La EUR/RON cel mai calm an a fost 2024 (0,025%), cel mai agitat 2008 (0,72%): '
              'BNR a ținut leul aproape fix în 2020, cînd acțiunile treceau prin cea mai mare furtună.'),
  'acf-squares': ('Înlocuiți r² cu |r|. Cîte dintre cele 50 de autocorelații ale lui |r| pentru S&P 500 ies din bandă, și cît este autocorelația la lagul 50?',
                  'Toate 50 ies din bandă, iar la lagul 50 autocorelația este 0,127, față de 0,053 pentru r²; la lagul 1: 0,289 față de 0,313. '
                  'Valoarea absolută este mai puțin dominată de cîteva zile extreme decît pătratul, deci memoria mărimii randamentelor apare și mai clar. '
                  'Semnul (r) rămîne aproape imprevizibil: 15 autocorelații din 50 ies din bandă, cele mai multe mici.'),
  'arma-resid': ('Creșteți ordinul AR al mediei de la 1 la 5. Dispar efectele ARCH din reziduuri?',
                 'Nu. Q(10) al reziduurilor scade de la 29,7 la 15,4 (p = 0,009), dar Q(10) al pătratelor crește de la 2496 la 2562, iar ARCH-LM(5) rămîne în jur de 1020 '
                 '(pentru AR(1): n R² = 6664 × 0,1526 = 1017). Un model mai bogat pentru medie elimină autocorelația nivelului, nu pe cea a pătratelor: '
                 'varianța condiționată cere un model separat. În dreapta, aceleași reziduuri împărțite la volatilitatea GARCH(1,1) nu mai au efecte ARCH pentru niciun p: '
                 'Q(10) al lui ẑ² rămîne între 7,0 și 8,3 (p > 0,6).'),
  'simulated': ('Duceți α al ARCH(1) la 0,7 (ω = 0,3). Cît este kurtosisul teoretic și cît cel de selecție?',
                'Teoretic, momentul de ordinul patru nu există (3α² = 1,47 ≥ 1, pragul este α = 1/√3 = 0,577), deci kurtosisul este infinit. '
                'Kurtosisul celor 1000 de valori simulate este 11,24, iar la α = 0,9 este 34,9: o selecție finită are mereu un kurtosis finit, '
                'deci valoarea de selecție nu poate arăta că momentul nu există. La α = 0,5 teoria dă 9,0, selecția 4,91: momentul de ordinul patru converge încet.'),
  'ewma-garch': ('Creșteți λ la 0,97. Este vîrful EWMA din martie 2020 mai mare sau mai mic decît 85%?',
                 'Mai mic: 67%, pe 27 martie 2020, față de 85% pe 25 martie pentru λ = 0,94. Cu λ mai mare, ponderea randamentului de ieri, 1 − λ, scade de la 0,06 la 0,03, '
                 'iar timpul de înjumătățire al ponderilor crește de la 11,2 la 22,8 zile: estimarea reacționează mai încet și coboară mai încet (37% pe 30 iunie 2020, față de 30%). '
                 'Cu λ = 0,90 vîrful este 103%, aproape de cel al GARCH (111%), al cărui α = 0,123 este mai mare decît 1 − λ = 0,06.'),
  'lik-arch1': ('Treceți a doua selecție la n = 5000. Cît devine eroarea standard a lui α̂?',
                '0,015 (α̂ = 0,516), față de 0,035 pentru n = 1000: de 2,3 ori mai mică, aproape de √5 = 2,24. Eroarea standard scade cu √n, '
                'pentru că log-verosimilitatea este o sumă de n termeni și curbura ei la maxim crește proporțional cu n.'),
  'lik-garch': ('Cu n = 1000 în panoul din dreapta, unde este maximul pe grilă?',
                'În (0,045; 0,90): α este tot subestimat și β supraestimat, de-a lungul aceleiași creste pe care α + β se schimbă puțin. '
                'Cu n = 1500 maximul este în (0,075; 0,82), cu n = 3000 în (0,10; 0,81), cu n = 5000 în (0,10; 0,82). '
                'Între α și β există un compromis: date diferite cu aceeași persistență dau aproape aceeași verosimilitate, de aceea GARCH cere eșantioane lungi.'),
  'qq': ('Pentru ce ν cuantila teoretică de 0,1% a distribuției t standardizate devine egală cu cea de selecție, −4,79?',
         'Pentru ν ≈ 4,5 (−4,79), mai mic decît ν̂ = 6,26. Dar distribuția t este simetrică: cu ν = 4,5 cuantila de 99,9% este +4,79, '
         'în timp ce cea de selecție este doar 2,93. Coada stîngă a reziduurilor este mai groasă decît cea dreaptă; o singură valoare ν nu le potrivește pe amîndouă, '
         'de aceea cursul compară și distribuția t asimetrică (skewed t).'),
  'vol-sp500-bet': ('Dublați α (c = 2) păstrînd α + β. Ce se întîmplă cu vîrful S&P 500 din martie 2020 și cu mediana volatilității?',
                    'Vîrful crește de la 111% la 140%, iar mediana scade de la 14,3% la 13,3%: cu α = 0,245 și β = 0,750, volatilitatea reacționează mai puternic la fiecare șoc '
                    'și revine mai repede. Pentru BET vîrful din 2008 crește de la 117% la 138%. Persistența, deci și nivelul pe termen lung, rămîne aceeași; '
                    'se schimbă forma traiectoriei. Valorile estimate (c = 1) sînt cele care maximizează verosimilitatea.'),
  'vol-eurron-btc': ('Înjumătățiți α pentru Bitcoin (c = 0,5). Scade vîrful din 13 martie 2020 sub 250%?',
                     'Da: 218%, față de 301% cu α estimat; mediana crește de la 58,7% la 66,6%. Și la un IGARCH, α decide cît de mult urcă volatilitatea după un șoc; '
                     'α + β = 1 decide doar că șocul nu se stinge. Pentru EUR/RON, cu c = 0,5, vîrful din 2008 scade de la 22% la 19%.'),
  'persistence': ('Cu α + β rotunjit la 0,995, timpul de înjumătățire al S&P 500 iese 138 de zile. Tabelul din curs dă 0,9947 și 131. Care este corect?',
                  '131: persistența estimată, nerotunjită, este 0,99471, iar ln 0,5 / ln 0,99471 = 130,7 zile. Rotunjirea la trei zecimale adaugă 7 zile; pentru BET, 0,99071 dă 74,3 zile, '
                  'rotunjit la 0,991 dă 76,7. Rotunjit la 0,99, ambele dau 69,0 zile. Aproape de 1, timpul de înjumătățire depinde puternic de a treia zecimală: '
                  'se calculează din valoarea nerotunjită.'),
  'bands': ('Cu acoperire 99%, ce proporție din zilele din martie–aprilie 2020 ratează intervalul constant și ce proporție ratează intervalul GARCH?',
            'Intervalul constant ratează 28,6% din zile, intervalul GARCH 2,4%. Pe tot eșantionul: 2,9% și 1,0%, iar în 2017: 0,4% și 0,4%. '
            'Lățimea intervalului constant este 7,2 puncte în fiecare zi; a celui GARCH variază între 3,4 și 35,1 puncte în fereastra din grafic. '
            'Intervalul constant greșește tocmai în crize, cînd contează cel mai mult.'),
  'nic': ('Pentru un șoc de 4%, de cîte ori este mai mare varianța de mîine a S&P 500 după o scădere decît după o creștere, în GJR?',
          'De 3,50 ori, față de 1,62 pentru un șoc de 2%. Raportul crește cu mărimea șocului: termenul γε² crește cu pătratul șocului, iar βσ² nu depinde de el. '
          'EGARCH dă 2,97; estimarea kernel doar 1,41, pentru că există puține zile cu scăderi de 4%. La Bitcoin raportul GJR rămîne aproape 1 (0,98).'),
  'acf-diag': ('Pentru EUR/RON, Q(10) al lui ẑ² este 0,004 (p = 1,000). Este modelul GARCH-t perfect?',
               'Nu. Valoarea vine dintr-o singură zi, 6 mai 2025, cu ẑ = 153,7, al cărei pătrat domină toate autocorelațiile. Fără această zi, Q(10) = 19,2 (p = 0,038): '
               'rămîn efecte ARCH, pe care testul cu toate zilele nu le separă de zgomot. Pentru S&P 500 excluderea celei mai mari |ẑ| (27 februarie 2007) schimbă puțin '
               'rezultatul (13,6 → 14,2).'),
  'term-structure': ('Mutați originea pe 16 martie 2020. Este volatilitatea pe 10 zile din regula rădăcinii pătrate mai mare sau mai mică decît cea din suma prognozelor?',
                     'Mai mare: √(10 × 49,03) = 22,14% față de √479,49 = 21,90%. Pe 18 septembrie 2026 era invers: 2,26% față de 2,38%. '
                     'Într-o criză varianța prognozată coboară spre nivelul pe termen lung, deci regula rădăcinii pătrate supraestimează; într-o perioadă calmă o subestimează.'),
  'forecast-eval': ('Eliminați ziua cu cea mai mare diferență de pierdere. Devine semnificativă la 5% diferența dintre GARCH-t și EWMA pentru EUR/RON?',
                    'Nu: t = −1,32 (p = 0,187), față de −1,02 (p = 0,310) cu toate zilele. Ziua de 6 mai 2025 aducea 98% din diferența totală, iar fără ea mediile QLIKE ale celor două '
                    'modele sînt apropiate. Pentru BET, fără 19 decembrie 2018 avantajul GARCH-t devine mai clar (t = −2,79, p = 0,005).'),
  'var': ('La VaR 5%, în fereastra iulie 2019 – iunie 2021, are EWMA-Normal tot mai multe depășiri decît GARCH-t?',
          'Nu: EWMA-Normal are 28, GARCH-t 32, față de 25,3 așteptate. Cuantila de 5% a distribuției t standardizate cu ν = 6,26 este −1,59, mai aproape de zero decît −1,645 a distribuției Normale: '
          'cozile groase cer cuantile mai mari doar departe în coadă. La VaR 1%: 13 depășiri GARCH-t și 17 EWMA-Normal (5 așteptate); la 0,5%: 7 și 16.'),
 },
 'en': {
  'returns': ('In which year was the daily standard deviation of the S&P 500 returns lowest?',
              'In 2017: 0.42%, followed by 2006 (0.63%) and 2005 (0.65%). From September 2008 to March 2009 the standard deviation was 3.52%, 8.3 times larger, '
              'and in February–May 2020 3.70% (8.8 times). For the EUR/RON the calmest year was 2024 (0.025%) and the most agitated 2008 (0.72%): '
              'the BNR kept the leu almost fixed in 2020, when equities went through their largest storm.'),
  'acf-squares': ('Replace r² by |r|. How many of the 50 autocorrelations of |r| for the S&P 500 lie outside the band, and what is the autocorrelation at lag 50?',
                  'All 50 lie outside the band, and at lag 50 the autocorrelation is 0.127, against 0.053 for r²; at lag 1: 0.289 against 0.313. '
                  'The absolute value is less dominated by a few extreme days than the square, so the memory of the size of returns shows even more clearly. '
                  'The sign (r) stays almost unpredictable: 15 of 50 autocorrelations are outside the band, most of them small.'),
  'arma-resid': ('Raise the AR order of the mean from 1 to 5. Do the ARCH effects of the residuals disappear?',
                 'No. Q(10) of the residuals falls from 29.7 to 15.4 (p = 0.009), but Q(10) of the squares rises from 2496 to 2562, and ARCH-LM(5) stays around 1020 '
                 '(for AR(1): n R² = 6664 × 0.1526 = 1017). A richer model for the mean removes the autocorrelation of the level, not that of the squares: '
                 'the conditional variance needs a model of its own. On the right, the same residuals divided by the GARCH(1,1) volatility lose their ARCH effects for every p: '
                 'Q(10) of ẑ² stays between 7.0 and 8.3 (p > 0.6).'),
  'simulated': ('Move α of ARCH(1) to 0.7 (ω = 0.3). What are the theoretical and the sample kurtosis?',
                'In theory the fourth moment does not exist (3α² = 1.47 ≥ 1, the threshold is α = 1/√3 = 0.577), so the kurtosis is infinite. '
                'The kurtosis of the 1000 simulated values is 11.24, and 34.9 at α = 0.9: a finite sample always has a finite kurtosis, '
                'so the sample value cannot show that the moment does not exist. At α = 0.5 theory gives 9.0 and the sample 4.91: the fourth moment converges slowly.'),
  'ewma-garch': ('Raise λ to 0.97. Is the EWMA peak of March 2020 higher or lower than 85%?',
                 'Lower: 67%, on 27 March 2020, against 85% on 25 March for λ = 0.94. With a larger λ the weight of yesterday’s return, 1 − λ, falls from 0.06 to 0.03, '
                 'and the half-life of the weights rises from 11.2 to 22.8 days: the estimate reacts more slowly and comes down more slowly (37% on 30 June 2020, against 30%). '
                 'With λ = 0.90 the peak is 103%, close to that of GARCH (111%), whose α = 0.123 is larger than 1 − λ = 0.06.'),
  'lik-arch1': ('Move the second sample to n = 5000. What does the standard error of α̂ become?',
                '0.015 (α̂ = 0.516), against 0.035 for n = 1000: 2.3 times smaller, close to √5 = 2.24. The standard error falls with √n, '
                'because the log-likelihood is a sum of n terms and its curvature at the maximum grows in proportion to n.'),
  'lik-garch': ('With n = 1000 in the right panel, where is the maximum on the grid?',
                'At (0.045, 0.90): α is still underestimated and β overestimated, along the same ridge on which α + β changes little. '
                'With n = 1500 the maximum is at (0.075, 0.82), with n = 3000 at (0.10, 0.81), with n = 5000 at (0.10, 0.82). '
                'α and β trade off: different pairs with the same persistence give almost the same likelihood, which is why GARCH needs long samples.'),
  'qq': ('For which ν does the theoretical 0.1% quantile of the standardised t equal the sample one, −4.79?',
         'For ν ≈ 4.5 (−4.79), smaller than ν̂ = 6.26. But the t distribution is symmetric: with ν = 4.5 its 99.9% quantile is +4.79, '
         'while the sample one is only 2.93. The left tail of the residuals is heavier than the right one; a single ν cannot fit both, '
         'which is why the lecture also compares the skewed t distribution.'),
  'vol-sp500-bet': ('Double α (c = 2) keeping α + β. What happens to the S&P 500 peak of March 2020 and to the median volatility?',
                    'The peak rises from 111% to 140% and the median falls from 14.3% to 13.3%: with α = 0.245 and β = 0.750 the volatility reacts more strongly to every shock '
                    'and comes back faster. For the BET the 2008 peak rises from 117% to 138%. The persistence, and so the long-run level, stays the same; '
                    'the shape of the path changes. The estimated values (c = 1) are those that maximise the likelihood.'),
  'vol-eurron-btc': ('Halve α for Bitcoin (c = 0.5). Does the peak of 13 March 2020 fall below 250%?',
                     'Yes: 218%, against 301% with the estimated α; the median rises from 58.7% to 66.6%. In an IGARCH too, α decides how far volatility jumps after a shock; '
                     'α + β = 1 only decides that the shock does not die out. For the EUR/RON, with c = 0.5, the 2008 peak falls from 22% to 19%.'),
  'persistence': ('With α + β rounded to 0.995, the half-life of the S&P 500 comes out at 138 days. The table of the lecture gives 0.9947 and 131. Which is right?',
                  '131: the unrounded estimated persistence is 0.99471, and ln 0.5 / ln 0.99471 = 130.7 days. Rounding to three decimals adds 7 days; for the BET, 0.99071 gives 74.3 days, '
                  'rounded to 0.991 it gives 76.7. Rounded to 0.99, both give 69.0 days. Close to 1 the half-life depends strongly on the third decimal: '
                  'compute it from the unrounded value.'),
  'bands': ('With 99% coverage, what share of the days of March–April 2020 does the constant interval miss, and what share does the GARCH interval miss?',
            'The constant interval misses 28.6% of the days, the GARCH interval 2.4%. Over the whole sample: 2.9% and 1.0%, and in 2017: 0.4% and 0.4%. '
            'The width of the constant interval is 7.2 points every day; that of the GARCH interval varies between 3.4 and 35.1 points in the window of the chart. '
            'The constant interval fails precisely in crises, when it matters most.'),
  'nic': ('For a shock of 4%, how many times larger is tomorrow’s variance of the S&P 500 after a fall than after a rise, in GJR?',
          '3.50 times, against 1.62 for a shock of 2%. The ratio grows with the size of the shock: the term γε² grows with the square of the shock, while βσ² does not depend on it. '
          'EGARCH gives 2.97; the kernel estimate only 1.41, because there are few days with falls of 4%. For Bitcoin the GJR ratio stays close to 1 (0.98).'),
  'acf-diag': ('For the EUR/RON, Q(10) of ẑ² is 0.004 (p = 1.000). Is the GARCH-t model perfect?',
               'No. The value comes from a single day, 6 May 2025, with ẑ = 153.7, whose square dominates all autocorrelations. Without that day, Q(10) = 19.2 (p = 0.038): '
               'ARCH effects remain, which the test with all days does not separate from noise. For the S&P 500, excluding the largest |ẑ| (27 February 2007) changes the result '
               'little (13.6 → 14.2).'),
  'term-structure': ('Move the origin to 16 March 2020. Is the 10-day volatility of the square-root-of-time rule larger or smaller than the one from the sum of the forecasts?',
                     'Larger: √(10 × 49.03) = 22.14% against √479.49 = 21.90%. On 18 September 2026 it was the other way round: 2.26% against 2.38%. '
                     'In a crisis the forecast variance comes down towards the long-run level, so the square-root-of-time rule overstates; in a calm period it understates.'),
  'forecast-eval': ('Remove the day with the largest loss difference. Does the difference between GARCH-t and EWMA for the EUR/RON become significant at 5%?',
                    'No: t = −1.32 (p = 0.187), against −1.02 (p = 0.310) with all days. The day of 6 May 2025 brought 98% of the total difference, and without it the mean QLIKE of the '
                    'two models are close. For the BET, without 19 December 2018 the advantage of GARCH-t becomes clearer (t = −2.79, p = 0.005).'),
  'var': ('At VaR 5%, in the window July 2019 – June 2021, does EWMA-Normal still have more exceedances than GARCH-t?',
          'No: EWMA-Normal has 28, GARCH-t 32, against 25.3 expected. The 5% quantile of the standardised t with ν = 6.26 is −1.59, closer to zero than the −1.645 of the Normal distribution: '
          'heavy tails require larger quantiles only far in the tail. At VaR 1%: 13 GARCH-t and 17 EWMA-Normal exceedances (5 expected); at 0.5%: 7 and 16.'),
 },
}

TEXT_JS = {
 'ro': r"""(function () {
const MON = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie', 'octombrie', 'noiembrie', 'decembrie'];
const dmy = d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;
const deN = n => { const r = n % 100; return (r === 0 && n > 0) || r >= 20 ? ' de' : ''; };
window.CH5_TEXT = {
 dmy,
 re1: (n, y, s, e8, e0, r8, r0) => `${n}: abaterea standard zilnică ${s === null ? `nu este definită în ${y} (seria începe mai tîrziu)` : `${s}% în ${y}`}; `
   + (e8 === null ? '' : `${e8}% în septembrie 2008 – martie 2009${r8 === null ? '' : ` (de ${r8} ori)`}, `) + `${e0}% în februarie–mai 2020${r0 === null ? '' : ` (de ${r0} ori)`}.`,
 aq1: (n, d0, T, f, r1, a1, a50, o2, o1, b) => `${n}, de la ${dmy(d0)} (T = ${T}): ACF a lui r la lagul 1 ${r1}; ACF a lui ${f === 'abs' ? '|r|' : 'r²'} ${a1} la lagul 1 și ${a50} la lagul 50; `
   + `în afara benzii ±${b}: ${o2} din 50 pentru ${f === 'abs' ? '|r|' : 'r²'}, ${o1} din 50 pentru r.`,
 ar1: (p, q, df, pq, r1, q2, pq2, r2, n, lm, plm) => `${p ? `AR(${p})` : 'Media constantă'}, varianță constantă: Q(10) al reziduurilor ${q} (χ²(${df}), ${pq}); ACF a pătratelor la lagul 1 ${r1}, Q(10) al pătratelor ${q2} (${pq2}); `
   + `ARCH-LM(5): R² = ${r2}, LM = n R² = ${n} × ${r2} = ${lm} (${plm}).`,
 ar2: (k, r1, q2, p2, lm, plm, q, p) => `${k ? `AR(${k})` : 'Media constantă'}-GARCH(1,1)-t${k === 1 ? '' : ' (parametrii GARCH ai cursului, nereestimați)'}: ACF a lui ẑ² la lagul 1 ${r1}, Q(10) al lui ẑ² ${q2} (${p2}), ARCH-LM(5) ${lm} (${plm}); Q(10) al lui ẑ ${q} (${p}): rămîne autocorelație în nivel.`,
 si1: (k, r1) => `i.i.d. Normal: kurtosis de selecție ${k} (teoretic 3); ACF a pătratelor la lagul 1 ${r1}.`,
 si2: (a, k, kt, r1, mx) => `ARCH(1), α = ${a}: kurtosis de selecție ${k}, teoretic ${kt}${kt === '∞' ? ' (3α² ≥ 1: momentul de ordinul patru nu există)' : ''}; ACF a pătratelor la lagul 1 ${r1}; cea mai mare valoare absolută ${mx}.`,
 si3: (a, b, k, kt, r1, hl) => `GARCH(1,1), α = ${a}, β = ${b}: kurtosis de selecție ${k}, teoretic ${kt}; ACF a pătratelor la lagul 1 ${r1}` + (hl === null ? '.' : `; timpul de înjumătățire al unui șoc de varianță ${hl} zile.`),
 ew1: (g, dg, e, de, w, r, dr) => `Vîrfuri: GARCH ${g}% (${dg}), EWMA ${e}% (${de}), fereastra de ${w}${deN(w)} zile ${r}% (${dr}).`,
 ew2: (g, e, r) => `Pe 30 iunie 2020: GARCH ${g}%, EWMA ${e}%, fereastra ${r}%.`,
 ew3: (l, w, hl, a) => `EWMA cu λ = ${l}: ponderea pătratului de ieri 1 − λ = ${w}, timpul de înjumătățire al ponderilor ${hl} zile; α al GARCH-t este ${a}.`,
 l11: (n, a, se, lo, hi) => `n = ${n}: α̂ = ${a}, eroarea standard ${se} (din curbura log-verosimilității); interval aproximativ de 95%: ${lo} – ${hi}.`,
 l12: (a0, r, s) => `α adevărat = ${a0}; eroarea standard pentru n = 100 este de ${r} ori mai mare decît pentru a doua selecție (√(n/100) = ${s}).`,
 lg1: (n, a, b, p) => `n = ${n}: maximul pe grilă în α = ${a}, β = ${b} (α + β = ${p}); valorile adevărate α = 0,1, β = 0,8.`,
 qq1: (q, t, q9, t9) => `GARCH-N față de Normală: cuantila de 0,1% a lui ẑ ${q}, teoretic ${t}; cuantila de 99,9% ${q9}, teoretic ${t9}.`,
 qq2: (nu, q, t, q9, t9, k) => `GARCH-t față de t standardizată cu ν = ${nu}: cuantila de 0,1% ${q}, teoretic ${t}; cuantila de 99,9% ${q9}, teoretic ${t9}` + (k === null ? ' (kurtosis infinit pentru ν ≤ 4).' : `; kurtosisul distribuției de referință ${k}.`),
 vo1: (n, a, b, p8, p0, med, last, s) => `${n}: α = ${a}, β = ${b}; ` + (p8 ? `vîrful din 2008–2009 ${p8[0]}% (${p8[1]}), ` : '') + (p0 ? `vîrful din 2020 ${p0[0]}% (${p0[1]}); ` : '')
   + `mediana ${med}%, pe 18 septembrie 2026 ${last}%; volatilitatea de selecție ${s}%.`,
 vo2: r => `Mediana Bitcoin este de ${r} ori mediana S&P 500 (cu același c).`,
 pe1: (n, p, hl) => `${n}: α + β = ${p}, timpul de înjumătățire ln 0,5 / ln(α + β) = ${hl} zile.`,
 pe2: (p, hl) => `Curba întreruptă: α + β = ${p}, timpul de înjumătățire ${hl} zile. EUR/RON și Bitcoin: α + β = 1 (IGARCH), fără timp de înjumătățire.`,
 bd1: (c, wc, w0, w1) => `Intervale de ${c}%: lățimea intervalului constant ${wc} puncte în fiecare zi; a intervalului GARCH între ${w0} și ${w1} puncte în fereastra iulie 2019 – iunie 2021.`,
 bd2: (c, oc, og, e) => `Zile în afara intervalului, tot eșantionul: constant ${oc}%, GARCH ${og}% (nominal ${e}%).`,
 bd3: (c7, g7, cc, gc) => `În 2017: constant ${c7}%, GARCH ${g7}%; în martie–aprilie 2020: constant ${cc}%, GARCH ${gc}%.`,
 ni1: (n, x, g, e, k, a, gm) => `${n}: după o scădere de ${x}% varianța de mîine este de ${g} ori cea de după o creștere de ${x}% în GJR (α = ${a}, γ = ${gm}), de ${e} ori în EGARCH, de ${k} ori în estimarea kernel.`,
 ni2: (c, h, n2, h2) => `Banda kernel: ${c} abateri standard ale randamentelor, adică ${h} pentru S&P 500 și ${h2} pentru ${n2}.`,
 dg1: (n, a1, a50, z1, zm, b) => `${n}: ACF a lui r² ${a1} la lagul 1 și ${a50} la lagul 50; ACF a lui ẑ² ${z1} la lagul 1, cel mult ${zm} în valoare absolută; banda ±${b}.`,
 dg2: (r2, z2, p2, z, p) => `Q(10): al lui r² ${r2}; al lui ẑ² ${z2} (${p2}); al lui ẑ ${z} (${p}).`,
 dg3: (d, z, drop) => `Cea mai mare |ẑ|: ${d}, ẑ = ${z}` + (drop ? '; ziua este exclusă din ẑ.' : '.'),
 ts1: (d, h1, a10, a250) => `Prognoză făcută pe ${d}: ${h1}% pentru ziua următoare, în medie ${a10}% pe 10 zile și ${a250}% pe 250 de zile (anualizat).`,
 ts2: lr => `Volatilitatea pe termen lung: ${lr}%.`,
 ts3: (d, s, sm, v, r) => `${d}: σ²(t+1) = ${s}; varianța pe 10 zile, suma celor zece prognoze, ${sm}, deci volatilitatea pe 10 zile ${v}%; regula rădăcinii pătrate √(10 σ²(t+1)) dă ${r}%.`,
 fe1: (n, T, mb, qb, ma, qa, t, p, d, v, sh) => `${n} (${T}${deN(T)} zile): QLIKE medie ${mb} ${qb}, ${ma} ${qa}; t al testului DM ${t} (${p}); cea mai mare diferență zilnică ${v}, pe ${d}`
   + (sh === null ? '.' : `, ${sh}% din diferența totală.`),
 va1: (n, d0, d1, T, a, e, g, pg, w, pw) => `${n}, ${d0} – ${d1} (${T}${deN(T)} zile), VaR ${a}%: depășiri așteptate ${e}; GARCH-t ${g} (${pg}%), EWMA-Normal ${w} (${pw}%).`,
};
})();
""",
 'en': r"""(function () {
const MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const dmy = d => `${+d.slice(8, 10)} ${MON[+d.slice(5, 7) - 1]} ${d.slice(0, 4)}`;
window.CH5_TEXT = {
 dmy,
 re1: (n, y, s, e8, e0, r8, r0) => `${n}: daily standard deviation ${s === null ? `not defined in ${y} (the series starts later)` : `${s}% in ${y}`}; `
   + (e8 === null ? '' : `${e8}% in September 2008 – March 2009${r8 === null ? '' : ` (${r8} times)`}, `) + `${e0}% in February–May 2020${r0 === null ? '' : ` (${r0} times)`}.`,
 aq1: (n, d0, T, f, r1, a1, a50, o2, o1, b) => `${n}, from ${dmy(d0)} (T = ${T}): ACF of r at lag 1 ${r1}; ACF of ${f === 'abs' ? '|r|' : 'r²'} ${a1} at lag 1 and ${a50} at lag 50; `
   + `outside the band ±${b}: ${o2} of 50 for ${f === 'abs' ? '|r|' : 'r²'}, ${o1} of 50 for r.`,
 ar1: (p, q, df, pq, r1, q2, pq2, r2, n, lm, plm) => `${p ? `AR(${p})` : 'Constant mean'}, constant variance: Q(10) of the residuals ${q} (χ²(${df}), ${pq}); ACF of the squares at lag 1 ${r1}, Q(10) of the squares ${q2} (${pq2}); `
   + `ARCH-LM(5): R² = ${r2}, LM = n R² = ${n} × ${r2} = ${lm} (${plm}).`,
 ar2: (k, r1, q2, p2, lm, plm, q, p) => `${k ? `AR(${k})` : 'Constant mean'}-GARCH(1,1)-t${k === 1 ? '' : ' (GARCH parameters of the lecture, not re-estimated)'}: ACF of ẑ² at lag 1 ${r1}, Q(10) of ẑ² ${q2} (${p2}), ARCH-LM(5) ${lm} (${plm}); Q(10) of ẑ ${q} (${p}): some autocorrelation remains in the level.`,
 si1: (k, r1) => `i.i.d. Normal: sample kurtosis ${k} (theory 3); ACF of the squares at lag 1 ${r1}.`,
 si2: (a, k, kt, r1, mx) => `ARCH(1), α = ${a}: sample kurtosis ${k}, theory ${kt}${kt === '∞' ? ' (3α² ≥ 1: the fourth moment does not exist)' : ''}; ACF of the squares at lag 1 ${r1}; largest absolute value ${mx}.`,
 si3: (a, b, k, kt, r1, hl) => `GARCH(1,1), α = ${a}, β = ${b}: sample kurtosis ${k}, theory ${kt}; ACF of the squares at lag 1 ${r1}` + (hl === null ? '.' : `; half-life of a variance shock ${hl} days.`),
 ew1: (g, dg, e, de, w, r, dr) => `Peaks: GARCH ${g}% (${dg}), EWMA ${e}% (${de}), ${w}-day window ${r}% (${dr}).`,
 ew2: (g, e, r) => `On 30 June 2020: GARCH ${g}%, EWMA ${e}%, window ${r}%.`,
 ew3: (l, w, hl, a) => `EWMA with λ = ${l}: weight of yesterday’s square 1 − λ = ${w}, half-life of the weights ${hl} days; α of the GARCH-t is ${a}.`,
 l11: (n, a, se, lo, hi) => `n = ${n}: α̂ = ${a}, standard error ${se} (from the curvature of the log-likelihood); approximate 95% interval: ${lo} – ${hi}.`,
 l12: (a0, r, s) => `True α = ${a0}; the standard error for n = 100 is ${r} times that of the second sample (√(n/100) = ${s}).`,
 lg1: (n, a, b, p) => `n = ${n}: maximum on the grid at α = ${a}, β = ${b} (α + β = ${p}); true values α = 0.1, β = 0.8.`,
 qq1: (q, t, q9, t9) => `GARCH-N against the Normal: 0.1% quantile of ẑ ${q}, theory ${t}; 99.9% quantile ${q9}, theory ${t9}.`,
 qq2: (nu, q, t, q9, t9, k) => `GARCH-t against the standardised t with ν = ${nu}: 0.1% quantile ${q}, theory ${t}; 99.9% quantile ${q9}, theory ${t9}` + (k === null ? ' (infinite kurtosis for ν ≤ 4).' : `; kurtosis of the reference distribution ${k}.`),
 vo1: (n, a, b, p8, p0, med, last, s) => `${n}: α = ${a}, β = ${b}; ` + (p8 ? `2008–2009 peak ${p8[0]}% (${p8[1]}), ` : '') + (p0 ? `2020 peak ${p0[0]}% (${p0[1]}); ` : '')
   + `median ${med}%, on 18 September 2026 ${last}%; sample volatility ${s}%.`,
 vo2: r => `The Bitcoin median is ${r} times the S&P 500 median (with the same c).`,
 pe1: (n, p, hl) => `${n}: α + β = ${p}, half-life ln 0.5 / ln(α + β) = ${hl} days.`,
 pe2: (p, hl) => `Dashed curve: α + β = ${p}, half-life ${hl} days. EUR/RON and Bitcoin: α + β = 1 (IGARCH), no half-life.`,
 bd1: (c, wc, w0, w1) => `${c}% intervals: width of the constant interval ${wc} points every day; of the GARCH interval between ${w0} and ${w1} points in the window July 2019 – June 2021.`,
 bd2: (c, oc, og, e) => `Days outside the interval, whole sample: constant ${oc}%, GARCH ${og}% (nominal ${e}%).`,
 bd3: (c7, g7, cc, gc) => `In 2017: constant ${c7}%, GARCH ${g7}%; in March–April 2020: constant ${cc}%, GARCH ${gc}%.`,
 ni1: (n, x, g, e, k, a, gm) => `${n}: after a fall of ${x}% tomorrow’s variance is ${g} times that after a rise of ${x}% in GJR (α = ${a}, γ = ${gm}), ${e} times in EGARCH, ${k} times in the kernel estimate.`,
 ni2: (c, h, n2, h2) => `Kernel bandwidth: ${c} standard deviations of the returns, that is ${h} for the S&P 500 and ${h2} for ${n2}.`,
 dg1: (n, a1, a50, z1, zm, b) => `${n}: ACF of r² ${a1} at lag 1 and ${a50} at lag 50; ACF of ẑ² ${z1} at lag 1, at most ${zm} in absolute value; band ±${b}.`,
 dg2: (r2, z2, p2, z, p) => `Q(10): of r² ${r2}; of ẑ² ${z2} (${p2}); of ẑ ${z} (${p}).`,
 dg3: (d, z, drop) => `Largest |ẑ|: ${d}, ẑ = ${z}` + (drop ? '; the day is excluded from ẑ.' : '.'),
 ts1: (d, h1, a10, a250) => `Forecast made on ${d}: ${h1}% for the next day, ${a10}% on average over 10 days and ${a250}% over 250 days (annualised).`,
 ts2: lr => `Long-run volatility: ${lr}%.`,
 ts3: (d, s, sm, v, r) => `${d}: σ²(t+1) = ${s}; 10-day variance, the sum of the ten forecasts, ${sm}, so the 10-day volatility is ${v}%; the square-root-of-time rule √(10 σ²(t+1)) gives ${r}%.`,
 fe1: (n, T, mb, qb, ma, qa, t, p, d, v, sh) => `${n} (${T} days): mean QLIKE ${mb} ${qb}, ${ma} ${qa}; DM t statistic ${t} (${p}); largest daily difference ${v}, on ${d}`
   + (sh === null ? '.' : `, ${sh}% of the total difference.`),
 va1: (n, d0, d1, T, a, e, g, pg, w, pw) => `${n}, ${d0} – ${d1} (${T} days), VaR ${a}%: expected exceedances ${e}; GARCH-t ${g} (${pg}%), EWMA-Normal ${w} (${pw}%).`,
};
})();
""",
}

# Predict, then check: options for the question of each block, (RO, EN) pairs; key = index of the correct option.
# The correct option agrees with the readout of the block at the state the question names (verified by a node script).
PREDICT = {
 'returns': (0, [('În 2017: 0,42%.', 'In 2017: 0.42%.'), ('În 2006, anul calm dinaintea crizei din 2008.', 'In 2006, the calm year before the 2008 crisis.'),
                 ('În 2024: ultimii ani sînt cei mai liniștiți.', 'In 2024: the last years are the calmest.')]),
 'acf-squares': (0, [('Toate 50; la lagul 50: 0,127, mai mult decît pentru r².', 'All 50; at lag 50: 0.127, more than for r².'),
                     ('Niciuna: valoarea absolută elimină semnul, deci și dependența.', 'None: the absolute value removes the sign, hence the dependence.'),
                     ('Mai puține decît pentru r²: |r| pierde informația despre mărime.', 'Fewer than for r²: |r| loses the information about size.')]),
 'arma-resid': (1, [('Da: un AR de ordin mai mare face reziduurile zgomot alb.', 'Yes: a higher-order AR makes the residuals white noise.'),
                    ('Nu: Q(10) al pătratelor rămîne peste 2500, ARCH-LM în jur de 1020.', 'No: Q(10) of the squares stays above 2500, ARCH-LM around 1020.'),
                    ('Scad la jumătate: o parte din efectul ARCH venea din medie.', 'They halve: part of the ARCH effect came from the mean.')]),
 'simulated': (2, [('Aproximativ 9, ca pentru α = 0,5: kurtosisul nu depinde de α.', 'About 9, as for α = 0.5: the kurtosis does not depend on α.'),
                   ('Sub 3: o varianță mai mare aplatizează distribuția.', 'Below 3: a larger variance flattens the distribution.'),
                   ('Teoretic infinit; în selecție 11,24.', 'Infinite in theory; 11.24 in the sample.')]),
 'ewma-garch': (0, [('Mai mic: 67%, pe 27 martie 2020.', 'Lower: 67%, on 27 March 2020.'), ('Mai mare: o pondere mai mare a trecutului amplifică șocul.', 'Higher: a larger weight on the past amplifies the shock.'),
                    ('Același, 85%: λ schimbă doar viteza revenirii.', 'The same, 85%: λ only changes the speed of the return.')]),
 'lik-arch1': (1, [('Aproximativ 0,007: eroarea standard scade proporțional cu n.', 'About 0.007: the standard error falls in proportion to n.'),
                   ('Aproximativ 0,015: eroarea standard scade cu √n.', 'About 0.015: the standard error falls with √n.'),
                   ('Rămîne 0,035: curbura depinde doar de α.', 'It stays 0.035: the curvature depends on α only.')]),
 'lik-garch': (2, [('În (0,10; 0,80): de la n = 1000 maximul este la valorile adevărate.', 'At (0.10, 0.80): from n = 1000 on the maximum is at the true values.'),
                   ('În jur de (0,07; 0,70): la mijloc între n = 500 și n = 2000.', 'Around (0.07, 0.70): halfway between n = 500 and n = 2000.'),
                   ('În (0,045; 0,90): tot departe, de-a lungul crestei.', 'At (0.045, 0.90): still far, along the ridge.')]),
 'qq': (0, [('ν ≈ 4,5; dar atunci coada dreaptă nu se mai potrivește.', 'ν ≈ 4.5; but then the right tail no longer fits.'),
            ('ν̂ = 6,26: estimarea de verosimilitate maximă potrivește exact cozile.', 'ν̂ = 6.26: the maximum likelihood estimate fits the tails exactly.'),
            ('ν ≈ 10: cozi mai groase cer mai multe grade de libertate.', 'ν ≈ 10: heavier tails need more degrees of freedom.')]),
 'vol-sp500-bet': (1, [('Nimic: cu α + β neschimbat traiectoria rămîne aceeași.', 'Nothing: with α + β unchanged the path stays the same.'),
                       ('Vîrful crește la 140%, mediana scade la 13,3%.', 'The peak rises to 140%, the median falls to 13.3%.'),
                       ('Vîrful scade: un β mai mic înseamnă mai puțină memorie.', 'The peak falls: a smaller β means less memory.')]),
 'vol-eurron-btc': (0, [('Da: 218%.', 'Yes: 218%.'), ('Nu: la un IGARCH șocurile nu se sting, deci vîrful nu depinde de α.', 'No: in an IGARCH shocks do not die out, so the peak does not depend on α.'),
                        ('Nu: vîrful vine din randamentul acelei zile, nu din parametri.', 'No: the peak comes from that day’s return, not from the parameters.')]),
 'persistence': (2, [('138: persistența este 0,995.', '138: the persistence is 0.995.'),
                     ('Amîndouă: 7 zile sînt o diferență de rotunjire fără importanță.', 'Both: 7 days are an unimportant rounding difference.'),
                     ('131: se calculează din persistența nerotunjită, 0,99471.', '131: it is computed from the unrounded persistence, 0.99471.')]),
 'bands': (1, [('Cîte 1%: un interval de 99% ratează 1% din zile în orice perioadă.', '1% each: a 99% interval misses 1% of the days in any period.'),
               ('28,6% față de 2,4%.', '28.6% against 2.4%.'), ('În jur de 10% fiecare: în criză toate intervalele cedează la fel.', 'About 10% each: in a crisis all intervals fail alike.')]),
 'nic': (2, [('1,62, ca pentru 2%: asimetria este un raport fix.', '1.62, as for 2%: the asymmetry is a fixed ratio.'),
             ('3,24, dublul lui 1,62: raportul crește proporțional cu șocul.', '3.24, twice 1.62: the ratio grows in proportion to the shock.'), ('3,50.', '3.50.')]),
 'acf-diag': (1, [('Da: p = 1 înseamnă că nu există efecte ARCH.', 'Yes: p = 1 means there are no ARCH effects.'),
                  ('Nu: fără 6 mai 2025, Q(10) = 19,2 (p = 0,038).', 'No: without 6 May 2025, Q(10) = 19.2 (p = 0.038).'),
                  ('Da: distribuția t, cu cozi groase, absoarbe ziua extremă.', 'Yes: the heavy-tailed t distribution absorbs the extreme day.')]),
 'term-structure': (0, [('Mai mare: 22,14% față de 21,90%.', 'Larger: 22.14% against 21.90%.'), ('Mai mică, ca pe 18 septembrie 2026.', 'Smaller, as on 18 September 2026.'),
                        ('Egale: randamentele sînt necorelate, deci varianța pe 10 zile este 10 σ²(t+1).', 'Equal: returns are uncorrelated, so the 10-day variance is 10 σ²(t+1).')]),
 'forecast-eval': (0, [('Nu: t = −1,32, p = 0,187.', 'No: t = −1.32, p = 0.187.'), ('Da: rămîne avantajul de 98% al GARCH-t.', 'Yes: the 98% advantage of GARCH-t remains.'),
                       ('Semnul se inversează și EWMA devine semnificativ mai bun.', 'The sign flips and EWMA becomes significantly better.')]),
 'var': (1, [('Da: distribuția Normală subestimează riscul la orice nivel.', 'Yes: the Normal distribution understates the risk at every level.'),
             ('Nu: 28 pentru EWMA-Normal, 32 pentru GARCH-t.', 'No: 28 for EWMA-Normal, 32 for GARCH-t.'), ('Da, cu aceeași diferență de 4 depășiri ca la VaR 1%.', 'Yes, by the same 4 exceedances as at VaR 1%.')]),
}

# blocks whose initial charts show the answer: the live charts are veiled too until „Verifică” / "Check" or a change of the controls
VEIL_CHARTS = ('returns', 'lik-garch')

# Exit ticket „Verifică-te” / "Check Yourself": the questions of the slide „Autoevaluare” / "Self-assessment" and their answers.
TICKET = [
 dict(q=('ω = 0,05, α = 0,05, β = 0,90. Cît sînt varianța pe termen lung și timpul de înjumătățire?', 'ω = 0.05, α = 0.05, β = 0.90. What are the long-run variance and the half-life?'),
      key=0, opts=[
  (('1 și 13,5 zile.', '1 and 13.5 days.'),
   ('σ̄² = ω/(1 − α − β) = 0,05/0,05 = 1; h = ln 0,5 / ln 0,95 = 13,5 zile.', 'σ̄² = ω/(1 − α − β) = 0.05/0.05 = 1; h = ln 0.5 / ln 0.95 = 13.5 days.')),
  (('0,5 și 6,6 zile.', '0.5 and 6.6 days.'),
   ('Aici s-a folosit doar β: ω/(1 − β) și ln 0,5 / ln 0,90. Persistența este α + β = 0,95.', 'This uses β only: ω/(1 − β) and ln 0.5 / ln 0.90. The persistence is α + β = 0.95.')),
  (('1 și 20 de zile.', '1 and 20 days.'),
   ('20 = 1/(1 − 0,95) este suma ponderilor (α + β)ʰ pe toate orizonturile, nu timpul după care rămîne jumătate din șoc.', '20 = 1/(1 − 0.95) is the sum of the weights (α + β)ʰ over all horizons, not the time after which half of the shock is left.'))]),
 dict(q=('O regresie ARCH-LM cu q = 5 pe 1000 de reziduuri dă R² = 0,004. Există efecte ARCH la 5%?',
         'An ARCH-LM regression with q = 5 on 1000 residuals gives R² = 0.004. Are there ARCH effects at 5%?'),
      key=1, opts=[
  (('Da: R² este pozitiv, deci pătratele trecute explică o parte din pătratul de azi.', 'Yes: R² is positive, so past squares explain part of today’s square.'),
   ('Orice regresie are R² > 0 în selecție; testul compară LM = n R² cu χ²(5).', 'Every regression has R² > 0 in a sample; the test compares LM = n R² with χ²(5).')),
  (('Nu există dovezi de efecte ARCH la 5%: LM = 1000 × 0,004 = 4 < 11,07.', 'There is no evidence of ARCH effects at 5%: LM = 1000 × 0.004 = 4 < 11.07.'),
   ('H₀ nu se respinge: efectele ARCH, dacă există, nu se separă de zgomot cu acest test; asta nu dovedește că varianța este constantă.',
    'H₀ is not rejected: ARCH effects, if any, are not separated from noise by this test; this does not prove that the variance is constant.')),
  (('Nu: LM = 4 dovedește că varianța condiționată este constantă.', 'No: LM = 4 proves that the conditional variance is constant.'),
   ('O nerespingere nu dovedește ipoteza nulă; un test cu q = 5 poate rata, de exemplu, efecte la laguri mai mari.', 'A non-rejection does not prove the null; a test with q = 5 can miss, for example, effects at longer lags.'))]),
 dict(q=('GJR dă α̂ = 0 și γ̂ = 0,2. Ce se întîmplă cu varianța după o creștere de 3%?', 'GJR gives α̂ = 0 and γ̂ = 0.2. What happens to the variance after a rise of 3%?'),
      key=2, opts=[
  (('Crește cu 0,2 × 9 = 1,8.', 'It rises by 0.2 × 9 = 1.8.'),
   ('γ înmulțește ε² doar cînd ε < 0; pentru o creștere termenul lipsește.', 'γ multiplies ε² only when ε < 0; for a rise the term is absent.')),
  (('Scade sub nivelul de ieri: creșterile reduc volatilitatea.', 'It falls below yesterday’s level: rises reduce volatility.'),
   ('Modelul nu are un termen negativ pentru creșteri; varianța doar se atenuează cu β, ca după o zi fără șoc.', 'The model has no negative term for rises; the variance only decays with β, as after a day without a shock.')),
  (('Nimic peste atenuarea βσ²: doar șocurile negative cresc varianța.', 'Nothing beyond the decay βσ²: only negative shocks raise the variance.'),
   ('σ²(t+1) = ω + (α + γ·1[ε < 0]) ε² + βσ² = ω + βσ² pentru ε = +3.', 'σ²(t+1) = ω + (α + γ·1[ε < 0]) ε² + βσ² = ω + βσ² for ε = +3.'))]),
]


def _predict_js():
    import json
    for i, lang in enumerate(('ro', 'en')):
        pr = {a: dict(key=k, opts=[o[i] for o in opts], **({'veil': {'out': True, 'charts': True}} if a in VEIL_CHARTS else {}))
              for a, (k, opts) in PREDICT.items()}
        tk = dict(title='Verifică-te' if lang == 'ro' else 'Check Yourself', id='check', store=f'tsa-ch5-check-{lang}',
                  qs=[dict(q=t['q'][i], key=t['key'], opts=[dict(t=o[i], why=w[i]) for o, w in t['opts']]) for t in TICKET])
        TEXT_JS[lang] += ('window.CH5_TEXT.predict = ' + json.dumps(pr, ensure_ascii=False) + ';\n'
                          + 'window.CH5_TEXT.ticket = ' + json.dumps(tk, ensure_ascii=False) + ';\n')


_predict_js()


if __name__ == '__main__':
    import sys
    import interactive_common
    interactive_common.build(sys.modules[__name__])
