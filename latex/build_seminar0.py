r"""
build_seminar0.py -- Seminarul 0 (Introducere: primii pași cu seriile de timp), EN + RO
=======================================================================================
Seminarul are loc ÎNAINTEA cursului 0: este autonom („Noțiuni necesare azi” la început). Format A/B/C:
  A1-A3 [Rezolvat], A4 [Propus]       calcule pe hîrtie: rate de creștere, r_1 de mînă, prognoze naive și erori
  B1-B2 [Rezolvat], B3-B4 [Propus]    date reale: PIB (Eurostat), EUR/RON (BNR), IAPC (Eurostat), electricitate
  C1 (idee de proiect), C2 [Propus]   critica unui răspuns generat de AI (trei greșeli introduse intenționat)
Studenții nu predau nimic. Rezolvările problemelor propuse apar doar în versiunea profesorului
(*_solutions.tex, \solutionstrue). Cifrele vin din Quantlets/Ch_00/sem0_results.json (Quantlets/Ch_00/seminar0.py,
fișier al profesorului, exclus din git).
Ieșire:
  EN/Seminars/seminar0_introduction.tex (+ _solutions.tex)
  RO/Seminarii/seminar0_introducere_ro.tex (+ _solutions.tex)
Rulare:  python3 Quantlets/Ch_00/seminar0.py && python3 latex/build_seminar0.py && python3 latex/tsa_build.py compile 0
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import MARK, ROOT, Deck, Values, items, cols, table   # noqa: E402

VAL = json.load(open(os.path.join(ROOT, 'Quantlets', 'Ch_00', 'sem0_results.json')))
MONTHS = {'en': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                 'November', 'December'],
          'ro': ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie',
                 'octombrie', 'noiembrie', 'decembrie']}


def dt(s, day=True):
    y, m, d = s.split('-')
    en, ro = MONTHS['en'][int(m) - 1], MONTHS['ro'][int(m) - 1]
    return f'⟦{int(d)} {en} {y}||{int(d)} {ro} {y}⟧' if day else f'⟦{en} {y}||{ro} {y}⟧'


def quarter(s):
    y, q = s.split(' ')
    return f'⟦{y} {q}||T{q[1]} {y}⟧'


class LangValues(Values):
    lang = 'en'

    def __getitem__(self, key):
        v = super().__getitem__(key)
        return MARK.sub(lambda m: m.group(1) if self.lang == 'en' else m.group(2), v) if isinstance(v, str) else v


class LangDeck(Deck):
    def head(self, lang):
        V.lang = lang
        return super().head(lang)


V = LangValues()
for k, x in VAL.items():
    if isinstance(x, list):
        for i, v in enumerate(x):
            if v is not None and not isinstance(v, list):
                V.put(f'{k}{i}', v, 1)
        continue
    if isinstance(x, str):
        V.raw(k, dt(x) if len(x) == 10 and x[4] == '-' else (quarter(x) if ' Q' in x else x))
    elif isinstance(x, int):
        V.int(k, x)
    elif k.startswith(('a2_', 'a4_r1', 'a4_band', 'b2_acf', 'b2_band', 'b3_acf', 'b3_band', 'b4_', 'c2_ok_rmse')):
        V.put(k, x, 3 if k.startswith('b2_acf_ret') or k == 'b2_band' else 2)
    else:
        V.put(k, x, 2 if k.startswith(('a3_', 'a4_', 'b2_ret_sd')) or k == 'b1_last_y' else 1)
V.put('a4_yoy', VAL['a4_yoy'], 1)
V.put('b2_acf_ret_2', VAL['b2_acf_ret_2'], 4)
V.put('c2_ai_growth', VAL['c2_ai_growth'], 2)
V.raw('b1_neg_years', ', '.join(str(y) for y in VAL['b1_neg_yoy_years']))
V.raw('b3_month_max_m', f'⟦{MONTHS["en"][VAL["b3_month_max"] - 1]}||{MONTHS["ro"][VAL["b3_month_max"] - 1]}⟧')
V.raw('b3_month_min_m', f'⟦{MONTHS["en"][VAL["b3_month_min"] - 1]}||{MONTHS["ro"][VAL["b3_month_min"] - 1]}⟧')
V.raw('b4_test_first_m', dt(VAL['b4_test_first'], day=False))
V.raw('b4_test_last_m', dt(VAL['b4_test_last'], day=False))
V.raw('b3_last_m', dt(VAL['b3_last'], day=False))
V.raw('b3_max_m', dt(VAL['b3_max_yoy_date'], day=False))

SOLVED, PROP = '⟦[Solved]||[Rezolvat]⟧', '⟦[Proposed]||[Propus]⟧'
QL = '\\quantlet{TSA\\_ch0\\_seminar}{\\qlurl{TSA_ch0_seminar}}'
NB_EN, NB_RO = '\\href{\\nb}{seminar notebook}', '\\href{\\nb}{notebook-ul seminarului}'
CLOSING = ('⟦The seminar is practice and is not graded; the solutions of the [Proposed] tasks are discussed in class||'
           'Seminarul are rol de exercițiu și nu se notează; rezolvările cerințelor [Propus] se discută la seminar⟧')


def fig(name, h='0.5\\textheight'):
    return (f'\\begin{{center}}\n\\includegraphics[width=0.94\\textwidth,height={h},keepaspectratio]{{{name}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-2mm}}\n')


def clean(tex):
    return tex.replace('    \\begin{itemize}\n    \\end{itemize}\n', '')


D = LangDeck(0, 'seminar')

# ===============================================================================================================
D.section('Route and Primer', 'Traseu și noțiuni necesare')
# ===============================================================================================================
D.frame('⟦Today\'s Questions and Route||Întrebările de azi și traseul⟧', clean(items(
    '⟦First seminar of the course, before the Chapter 0 lecture: everything you need is on the next slides||Primul seminar, înaintea cursului din Capitolul 0: noțiunile necesare se găsesc pe slide-urile următoare⟧',
    ('⟦\\textbf{Questions of the seminar}||\\textbf{Întrebările seminarului}⟧',
     ['⟦how do we open the course notebooks and load a series?||cum deschidem notebook-urile cursului și cum încărcăm o serie?⟧',
      '⟦which growth rate tells us how the economy is doing?||ce rată de creștere arată starea economiei?⟧',
      '⟦does today\'s value tell us anything about tomorrow\'s?||spune valoarea de azi ceva despre cea de mîine?⟧',
      '⟦how good is the simplest possible forecast?||cît de bună este cea mai simplă prognoză posibilă?⟧']),
    ('⟦\\textbf{Part I}: setup, calculations on paper (A1--A3), the same ideas on real data (B1, B2)||\\textbf{Partea I}: pregătirea mediului de lucru, calcule pe hîrtie (A1--A3), aceleași idei pe date reale (B1, B2)⟧', []),
    ('⟦\\textbf{Part II}: your turn (A4, B3, B4), a project idea (C1) and the critique of an AI answer (C2)||\\textbf{Partea a II-a}: rîndul dumneavoastră (A4, B3, B4), o idee de proiect (C1) și critica unui răspuns generat de AI (C2)⟧', []),
    f'⟦Open the {NB_EN} in Google Colab; nothing is handed in||Deschideți {NB_RO} în Google Colab; nu se predă nimic⟧')))

MAP = [
    'A1 & ⟦growth rates of a quarterly series||rate de creștere ale unei serii trimestriale⟧ & ⟦Solved||Rezolvat⟧ & --',
    'A2 & ⟦a sample autocorrelation by hand||o autocorelație de selecție calculată de mînă⟧ & ⟦Solved||Rezolvat⟧ & --',
    'A3 & ⟦naive forecasts and their errors||prognoze naive și erorile lor⟧ & ⟦Solved||Rezolvat⟧ & --',
    'B1 & ⟦Romanian GDP: which growth rate?||PIB-ul României: ce rată de creștere?⟧ & ⟦Solved||Rezolvat⟧ & --',
    'B2 & ⟦EUR/RON: memory of levels and of changes||EUR/RON: memoria nivelurilor și a variațiilor⟧ & ⟦Solved||Rezolvat⟧ & --',
    'A4 & ⟦the same calculations on new numbers||aceleași calcule pe alte cifre⟧ & ⟦Proposed||Propus⟧ & A1--A3',
    'B3 & ⟦Romanian inflation: monthly and annual||inflația din România: lunară și anuală⟧ & ⟦Proposed||Propus⟧ & B1, B2',
    'B4 & ⟦electricity: naive or seasonal naive?||electricitatea: naivă sau naivă sezonieră?⟧ & ⟦Proposed||Propus⟧ & A3',
    'C1 & ⟦a project idea||o idee de proiect⟧ & ⟦Open||Deschis⟧ & --',
    'C2 & ⟦find the errors in an AI answer||greșelile dintr-un răspuns generat de AI⟧ & ⟦Proposed||Propus⟧ & A1, A3, B1',
]
D.frame('⟦Exercise Map||Harta exercițiilor⟧', table(
    'l>{\\raggedright\\arraybackslash}p{6.4cm}ll', '\\textbf{⟦Exercise||Exercițiu⟧} & \\textbf{⟦Question||Întrebare⟧} & \\textbf{⟦Type||Tip⟧} & \\textbf{Model}',
    MAP, size='footnotesize') + items(
    '⟦\\textbf{[Solved]}: full solution in the slides and the notebook, a model to follow; \\textbf{[Proposed]}: you solve it, the solution is discussed in class||\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat; \\textbf{[Propus]}: îl rezolvați dumneavoastră, rezolvarea se discută la seminar⟧',
    '⟦Part A: on paper; Part B: real data, each task ending with an interpretation question; Part C: open||Partea A: pe hîrtie; Partea B: date reale, fiecare cerință încheindu-se cu o întrebare de interpretare; Partea C: deschisă⟧'),
    size='footnotesize')

D.frame('⟦Needed Today (1/2): Series and Growth Rates||Noțiuni necesare azi (1/2): serii și rate de creștere⟧', items(
    ('⟦\\textbf{Time series} $y_1, \\ldots, y_T$: one variable observed at successive moments||\\textbf{Serie de timp} $y_1, \\ldots, y_T$: o variabilă observată în momente succesive⟧',
     ['⟦frequency: quarterly ($m = 4$ quarters a year), monthly ($m = 12$), daily||frecvența: trimestrială ($m = 4$ trimestre pe an), lunară ($m = 12$), zilnică⟧',
      '⟦unadjusted series contain \\textbf{seasonality}: a pattern that repeats every $m$ periods||seriile neajustate conțin \\textbf{sezonalitate}: un tipar care se repetă la fiecare $m$ perioade⟧']),
    ('⟦\\textbf{Growth over one period}: $100\\,(y_t / y_{t-1} - 1)$, in \\%||\\textbf{Creșterea pe o perioadă}: $100\\,(y_t / y_{t-1} - 1)$, în \\%⟧',
     ['⟦for quarterly data: quarter on quarter (q/q)||pentru date trimestriale: față de trimestrul anterior⟧']),
    ('⟦\\textbf{Growth over a year}: $100\\,(y_t / y_{t-m} - 1)$, year on year (y/y)||\\textbf{Creșterea pe un an}: $100\\,(y_t / y_{t-m} - 1)$, față de aceeași perioadă a anului anterior⟧',
     ['⟦compares the same season, so the seasonal pattern cancels out||compară același sezon, deci tiparul sezonier se anulează⟧']),
    ('⟦\\textbf{Log difference}: $100\\,(\\ln y_t - \\ln y_{t-1}) \\approx 100\\,(y_t / y_{t-1} - 1)$ for small changes||\\textbf{Diferența logaritmilor}: $100\\,(\\ln y_t - \\ln y_{t-1}) \\approx 100\\,(y_t / y_{t-1} - 1)$ pentru variații mici⟧',
     ['⟦for a price, it is the \\textbf{log return}||pentru un preț, este \\textbf{randamentul logaritmic}⟧'])))

D.frame('⟦Needed Today (2/2): Autocorrelation and Naive Forecasts||Noțiuni necesare azi (2/2): autocorelație și prognoze naive⟧', items(
    ('⟦\\textbf{Sample autocorrelation} at lag $k$||\\textbf{Autocorelația de selecție} la decalajul $k$⟧',
     ['$r_k = \\dfrac{\\sum_{t=k+1}^{T} (y_t - \\bar y)(y_{t-k} - \\bar y)}{\\sum_{t=1}^{T} (y_t - \\bar y)^2}$, ⟦between $-1$ and $1$||între $-1$ și $1$⟧',
      '⟦ACF (autocorrelation function): $r_1, r_2, \\ldots$ against $k$; band $\\pm 1.96/\\sqrt T$ for independent noise||ACF (autocorrelation function, funcția de autocorelație): $r_1, r_2, \\ldots$ în funcție de $k$; banda $\\pm 1.96/\\sqrt T$ pentru un zgomot independent⟧']),
    ('⟦\\textbf{Naive forecast}: $\\hat y_{T+h} = y_T$; \\textbf{seasonal naive}: $\\hat y_{T+h} = y_{T+h-m}$ ($h \\le m$)||\\textbf{Prognoza naivă}: $\\hat y_{T+h} = y_T$; \\textbf{naivă sezonieră}: $\\hat y_{T+h} = y_{T+h-m}$ ($h \\le m$)⟧', []),
    ('⟦\\textbf{Training set and test set}: split by time||\\textbf{Setul de antrenare și setul de test}: împărțire în timp⟧',
     ['⟦forecasts are built on the training set and compared with the test set they have not seen||prognozele se construiesc pe setul de antrenare și se compară cu setul de test, pe care nu l-au văzut⟧']),
    ('⟦\\textbf{Errors} $e = y - \\hat y$ on the test set||\\textbf{Erorile} $e = y - \\hat y$ pe setul de test⟧',
     ['⟦MAE (mean absolute error) $= \\frac1h \\sum |e|$; RMSE (root mean squared error) $= \\sqrt{\\frac1h \\sum e^2}$||MAE (mean absolute error, eroarea absolută medie) $= \\frac1h \\sum |e|$; RMSE (root mean squared error, rădăcina erorii pătratice medii) $= \\sqrt{\\frac1h \\sum e^2}$⟧']))
    .replace('    \\begin{itemize}\n    \\end{itemize}\n', ''), size='footnotesize')

DATA = [
    '⟦Romanian real GDP||PIB-ul real al României⟧ & ⟦Eurostat, unadjusted||Eurostat, neajustat⟧ & ⟦billion EUR, 2010 prices||miliarde EUR, prețurile din 2010⟧ & ⟦quarterly||trimestrial⟧ & B1, C2',
    'EUR/RON & ⟦BNR reference rate||cursul de referință BNR⟧ & ⟦lei per euro||lei pentru un euro⟧ & ⟦daily, from 2015||zilnic, din 2015⟧ & B2',
    '⟦Romanian HICP||IAPC, România⟧ & Eurostat & ⟦index, 2015 = 100||indice, 2015 = 100⟧ & ⟦monthly, from 2015||lunar, din 2015⟧ & B3',
    '⟦Electricity generation||Producția de electricitate⟧ & Eurostat & ⟦TWh per month||TWh pe lună⟧ & ⟦monthly, from 2008||lunar, din 2008⟧ & B4',
]
D.frame('⟦Data Used Today||Datele folosite azi⟧', table(
    '>{\\raggedright\\arraybackslash}p{2.8cm}>{\\raggedright\\arraybackslash}p{2.8cm}>{\\raggedright\\arraybackslash}p{2.9cm}>{\\raggedright\\arraybackslash}p{2.4cm}l',
    '\\textbf{⟦Series||Serie⟧} & \\textbf{⟦Source||Sursă⟧} & \\textbf{⟦Unit||Unitate⟧} & \\textbf{⟦Frequency||Frecvență⟧} & \\textbf{⟦Exercise||Exercițiu⟧}',
    DATA, size='footnotesize') + items(
    '⟦HICP: Harmonised Index of Consumer Prices; BNR: the National Bank of Romania; TWh: terawatt-hour||IAPC: indicele armonizat al prețurilor de consum (HICP); BNR: Banca Națională a României; TWh: terawatt-oră⟧',
    '⟦Public sources, read online without a key; the notebook defines one function per source||Surse publice, citite online, fără cheie; notebook-ul definește cîte o funcție pentru fiecare sursă⟧'),
    size='footnotesize')

D.task('⟦Setup (1/2): Open the Notebook and Save a Copy||Pregătire (1/2): deschiderea notebook-ului și salvarea unei copii⟧',
       '⟦can you run Python on the course data in the browser?||puteți rula Python pe datele cursului, în browser?⟧',
       '⟦Google Colab: Google\'s free notebook service; Python runs on Google\'s servers, a Google account is enough||Google Colab: serviciul gratuit de notebook-uri al Google; Python rulează pe serverele Google, este suficient un cont Google⟧',
       [f'⟦Open the {NB_EN} in Colab.||Deschideți {NB_RO} în Colab.⟧',
        '⟦Choose \\emph{File $\\to$ Save a copy in Drive}, so that your changes are kept.||Alegeți \\emph{File $\\to$ Save a copy in Drive}, ca să vă păstrați modificările.⟧',
        '⟦Run the cells of the section \\emph{Setup} in order with Shift + Enter; they import the libraries and define the data functions.||Rulați în ordine celulele secțiunii \\emph{Setup}, cu Shift + Enter; ele importă bibliotecile și definesc funcțiile de citire a datelor.⟧',
        '⟦If a cell fails, choose \\emph{Runtime $\\to$ Restart session} and run again from the top.||Dacă o celulă dă eroare, alegeți \\emph{Runtime $\\to$ Restart session} și rulați din nou de la început.⟧'],
       '⟦nothing; you are ready when the check on the next slide prints the expected numbers||nimic; sînteți gata atunci cînd verificarea de pe slide-ul următor afișează rezultatele așteptate⟧')

D.frame('⟦Setup (2/2): Load a Series||Pregătire (2/2): încărcarea unei serii⟧', cols(
    r"""\begin{lstlisting}
# Romanian real GDP, quarterly, unadjusted
g = read_eurostat('namq_10_gdp',
                  'Q.CLV10_MEUR.NSA.B1GQ.RO') / 1000
print(len(g), g.index[0].date(), g.index[-1].date())
print(g.tail(2).round(1))
g.plot()          # a first chart
\end{lstlisting}""" + '\n' + items('⟦\\texttt{read\\_eurostat(dataset, key)} reads one Eurostat series; the key lists the dimensions (frequency, unit, adjustment, indicator, country)||\\texttt{read\\_eurostat(dataset, cheie)} citește o serie Eurostat; cheia enumeră dimensiunile (frecvență, unitate, ajustare, indicator, țară)⟧'),
    items(('⟦\\textbf{Expected output}||\\textbf{Rezultatul așteptat}⟧',
           ['⟦@{b1_n} quarters, from @{b1_first} to @{b1_last}||@{b1_n} de trimestre, de la @{b1_first} la @{b1_last}⟧',
            '⟦last value: @{b1_last_y} billion EUR||ultima valoare: @{b1_last_y} miliarde EUR⟧']),
          '⟦Eurostat revises its data: a slightly different last value is normal||Eurostat revizuiește datele: o ultimă valoare ușor diferită este normală⟧',
          '⟦If the count differs a lot, check the key before going on||Dacă numărul de observații diferă mult, verificați cheia înainte de a continua⟧'),
    '0.56', '0.41'), opts='fragile', size='footnotesize')

# ===============================================================================================================
D.section('Part A: On Paper', 'Partea A: pe hîrtie')
# ===============================================================================================================
D.task(f'A1 {SOLVED}: ⟦Growth Rates --- Task||Rate de creștere --- cerință⟧',
       '⟦did the Romanian economy collapse in the first quarter of 2025?||s-a prăbușit economia României în primul trimestru din 2025?⟧',
       '⟦real GDP, billion EUR: @{a1_y0}, @{a1_y1}, @{a1_y2}, @{a1_y3} (2024 Q1--Q4), @{a1_y4}, @{a1_y5} (2025 Q1--Q2)||PIB real, miliarde EUR: @{a1_y0}, @{a1_y1}, @{a1_y2}, @{a1_y3} (T1--T4 2024), @{a1_y4}, @{a1_y5} (T1--T2 2025)⟧',
       ['⟦Compute the quarter-on-quarter growth rates for 2024 Q2 to 2025 Q2.||Calculați ratele de creștere față de trimestrul anterior, pentru T2 2024 -- T2 2025.⟧',
        '⟦Compute the year-on-year growth rates for 2025 Q1 and 2025 Q2.||Calculați ratele de creștere față de același trimestru al anului anterior, pentru T1 2025 și T2 2025.⟧',
        '⟦Compute the log difference for 2025 Q1 and compare it with the quarter-on-quarter rate.||Calculați diferența logaritmilor pentru T1 2025 și comparați-o cu rata față de trimestrul anterior.⟧'],
       '⟦the two kinds of growth rates, and one sentence that answers the question||cele două tipuri de rate de creștere și o propoziție care răspunde la întrebare⟧', nb='A1')

D.frame(f'A1 {SOLVED}: ⟦Solution||Rezolvare⟧', cols(
    table('lrrr', '⟦Quarter||Trimestru⟧ & $y_t$ & ⟦q/q, \\%||(1), \\%⟧ & ⟦y/y, \\%||(2), \\%⟧',
          ['⟦2024 Q1||T1 2024⟧ & @{a1_y0} & -- & --', '⟦2024 Q2||T2 2024⟧ & @{a1_y1} & $+$@{a1_pop1} & --',
           '⟦2024 Q3||T3 2024⟧ & @{a1_y2} & $+$@{a1_pop2} & --', '⟦2024 Q4||T4 2024⟧ & @{a1_y3} & $+$@{a1_pop3} & --',
           '⟦2025 Q1||T1 2025⟧ & @{a1_y4} & @{a1_pop4} & $+$@{a1_yoy4}', '⟦2025 Q2||T2 2025⟧ & @{a1_y5} & $+$@{a1_pop5} & $+$@{a1_yoy5}'],
          size='footnotesize') + items('⟦q/q: quarter on quarter; y/y: year on year||(1): față de trimestrul anterior; (2): față de același trimestru al anului anterior⟧'),
    items(('⟦\\textbf{Step by step}||\\textbf{Pas cu pas}⟧',
           ['⟦2025 Q1, q/q: $100\\,(@{a1_y4}/@{a1_y3} - 1) = $ @{a1_pop4}\\%||T1 2025, față de trimestrul anterior: $100\\,(@{a1_y4}/@{a1_y3} - 1) = $ @{a1_pop4}\\%⟧',
            '⟦2025 Q1, y/y: $100\\,(@{a1_y4}/@{a1_y0} - 1) = +$@{a1_yoy4}\\%||T1 2025, față de T1 2024: $100\\,(@{a1_y4}/@{a1_y0} - 1) = +$@{a1_yoy4}\\%⟧',
            '⟦log difference: $100\\,(\\ln @{a1_y4} - \\ln @{a1_y3}) = $ @{a1_dlog4}\\%, far from @{a1_pop4}\\%: the approximation fails for large changes||diferența logaritmilor: $100\\,(\\ln @{a1_y4} - \\ln @{a1_y3}) = $ @{a1_dlog4}\\%, departe de @{a1_pop4}\\%: aproximarea nu funcționează pentru variații mari⟧']),
          ('⟦\\textbf{Interpretation}||\\textbf{Interpretare}⟧',
           ['⟦no collapse: the fall of a third from Q4 to Q1 happens every winter; compared with a year earlier, GDP grew by @{a1_yoy4}\\%||nu a fost o prăbușire: scăderea de o treime din T4 în T1 apare în fiecare iarnă; față de anul anterior, PIB a crescut cu @{a1_yoy4}\\%⟧'])),
    '0.42', '0.55') + '\n' + QL, size='footnotesize')

D.task(f'A2 {SOLVED}: ⟦An Autocorrelation by Hand --- Task||O autocorelație calculată de mînă --- cerință⟧',
       '⟦is a high value usually followed by a high value in this short series?||este, de obicei, o valoare mare urmată de o valoare mare în această serie scurtă?⟧',
       '$y = 2, 4, 3, 5, 4, 6$ ($T = 6$)',
       ['⟦Compute the mean $\\bar y$ and the deviations $y_t - \\bar y$.||Calculați media $\\bar y$ și abaterile $y_t - \\bar y$.⟧',
        '⟦Compute the sum of squared deviations and the sum of products of consecutive deviations.||Calculați suma pătratelor abaterilor și suma produselor abaterilor consecutive.⟧',
        '⟦Compute $r_1$ and $r_2$, and compare them with the band $\\pm 1.96/\\sqrt T$.||Calculați $r_1$ și $r_2$ și comparați-le cu banda $\\pm 1.96/\\sqrt T$.⟧'],
       '⟦$\\bar y$, $r_1$, $r_2$, the band, and one sentence on what can be concluded from 6 observations||$\\bar y$, $r_1$, $r_2$, banda și o propoziție despre ce se poate concluziona din 6 observații⟧', nb='A2')

D.frame(f'A2 {SOLVED}: ⟦Solution||Rezolvare⟧', cols(
    table('rrrr', '$t$ & $y_t$ & $d_t = y_t - \\bar y$ & $d_t d_{t-1}$',
          ['1 & 2 & $-2$ & --', '2 & 4 & $0$ & $0$', '3 & 3 & $-1$ & $0$', '4 & 5 & $1$ & $-1$',
           '5 & 4 & $0$ & $0$', '6 & 6 & $2$ & $0$', '\\midrule ⟦sum||sumă⟧ & 24 & 0 & $-1$'], size='footnotesize'),
    items('$\\bar y = 24/6 = 4$; $\\sum d_t^2 = 4 + 0 + 1 + 1 + 0 + 4 = 10$',
          '$r_1 = -1/10 = $ @{a2_r1}',
          '⟦lag 2: $\\sum d_t d_{t-2} = (-1)(-2) + (1)(0) + (0)(-1) + (2)(1) = 4$, so $r_2 = $ @{a2_r2}||decalajul 2: $\\sum d_t d_{t-2} = (-1)(-2) + (1)(0) + (0)(-1) + (2)(1) = 4$, deci $r_2 = $ @{a2_r2}⟧',
          '⟦band: $\\pm 1.96/\\sqrt 6 = \\pm$@{a2_band}||banda: $\\pm 1.96/\\sqrt 6 = \\pm$@{a2_band}⟧',
          ('⟦\\textbf{Interpretation}||\\textbf{Interpretare}⟧',
           ['⟦both values lie well inside the band: 6 observations are far too few to detect memory||ambele valori sînt mult în interiorul benzii: 6 observații sînt mult prea puține pentru a detecta memoria⟧'])),
    '0.48', '0.48') + '\n' + QL, size='footnotesize')

D.task(f'A3 {SOLVED}: ⟦Naive Forecasts and Their Errors --- Task||Prognoze naive și erorile lor --- cerință⟧',
       '⟦for a seasonal series, which simple forecast is better: the last value or the same quarter last year?||pentru o serie sezonieră, ce prognoză simplă este mai bună: ultima valoare sau același trimestru de anul trecut?⟧',
       '⟦quarterly sales; training set: 10, 14, 18, 12 (year 1) and 11, 15, 20, 13 (year 2); test set: 12, 16, 21, 14 (year 3)||vînzări trimestriale; setul de antrenare: 10, 14, 18, 12 (anul 1) și 11, 15, 20, 13 (anul 2); setul de test: 12, 16, 21, 14 (anul 3)⟧',
       ['⟦Write the naive forecasts and the seasonal naive forecasts ($m = 4$) for the four quarters of year 3.||Scrieți prognozele naive și prognozele naive sezoniere ($m = 4$) pentru cele patru trimestre ale anului 3.⟧',
        '⟦Compute the errors $e = y - \\hat y$ of each method.||Calculați erorile $e = y - \\hat y$ ale fiecărei metode.⟧',
        '⟦Compute the MAE and the RMSE of each method.||Calculați MAE și RMSE pentru fiecare metodă.⟧'],
       '⟦a table with the forecasts and errors, the four error measures, and one sentence naming the better method||un tabel cu prognozele și erorile, cele patru măsuri ale erorii și o propoziție care numește metoda mai bună⟧', nb='A3')

D.frame(f'A3 {SOLVED}: ⟦Solution||Rezolvare⟧', cols(
    table('lrrrr', '& ⟦Q1||T1⟧ & ⟦Q2||T2⟧ & ⟦Q3||T3⟧ & ⟦Q4||T4⟧',
          ['⟦actual (year 3)||observat (anul 3)⟧ & 12 & 16 & 21 & 14', '⟦naive||naivă⟧ & 13 & 13 & 13 & 13',
           '⟦error||eroare⟧ & $-1$ & 3 & 8 & 1', '\\midrule ⟦seasonal naive||naivă sezonieră⟧ & 11 & 15 & 20 & 13',
           '⟦error||eroare⟧ & 1 & 1 & 1 & 1'], size='footnotesize'),
    items(('⟦\\textbf{Naive}: every forecast is the last value, 13||\\textbf{Naivă}: fiecare prognoză este ultima valoare, 13⟧',
           ['MAE $= (1 + 3 + 8 + 1)/4 = $ @{a3_naive_mae}; RMSE $= \\sqrt{(1 + 9 + 64 + 1)/4} = $ @{a3_naive_rmse}']),
          ('⟦\\textbf{Seasonal naive}: each quarter of year 2||\\textbf{Naivă sezonieră}: fiecare trimestru din anul 2⟧',
           ['MAE $=$ @{a3_snaive_mae}; RMSE $=$ @{a3_snaive_rmse}']),
          ('⟦\\textbf{Interpretation}||\\textbf{Interpretare}⟧',
           ['⟦the seasonal naive forecast is better: it keeps the seasonal pattern; the large Q3 error inflates the naive RMSE||prognoza naivă sezonieră este mai bună: păstrează tiparul sezonier; eroarea mare din T3 crește RMSE-ul prognozei naive⟧'])),
    '0.47', '0.50') + '\n' + QL, size='footnotesize')

# ===============================================================================================================
D.section('Part B: Real Data', 'Partea B: date reale')
# ===============================================================================================================
D.task(f'B1 {SOLVED}: ⟦Romanian GDP --- Task||PIB-ul României --- cerință⟧',
       '⟦which growth rate should be reported for an unadjusted quarterly series?||ce rată de creștere trebuie raportată pentru o serie trimestrială neajustată?⟧',
       '⟦Romanian real GDP, unadjusted, @{b1_first} -- @{b1_last} (Eurostat)||PIB-ul real al României, neajustat, @{b1_first} -- @{b1_last} (Eurostat)⟧',
       ['⟦Compute the quarter-on-quarter and the year-on-year growth rates of the whole series.||Calculați ratele de creștere față de trimestrul anterior și față de același trimestru al anului anterior, pentru întreaga serie.⟧',
        '⟦Plot the two growth rates since 2010 on the same chart.||Reprezentați cele două rate de creștere din 2010 pe același grafic.⟧',
        '⟦Count the negative values of each rate, and compute the average q/q growth of the first quarters.||Numărați valorile negative ale fiecărei rate și calculați media creșterii față de trimestrul anterior pentru trimestrele I.⟧',
        '⟦Interpretation: which of the two rates shows the recessions?||Interpretare: care dintre cele două rate arată recesiunile?⟧'],
       '⟦the counts of negative values, the average Q1 growth, the latest two rates, and the answer to the interpretation question||numărul valorilor negative, creșterea medie a trimestrelor I, ultimele două rate și răspunsul la întrebarea de interpretare⟧',
       size='footnotesize', nb='B1')

D.frame(f'B1 {SOLVED}: ⟦Solution (1/2) --- Step by Step||Rezolvare (1/2) --- pas cu pas⟧', items(
    ('⟦\\textbf{Quarter on quarter}: @{b1_n_pop} values, @{b1_neg_pop} negative||\\textbf{Față de trimestrul anterior}: @{b1_n_pop} de valori, dintre care @{b1_neg_pop} negative⟧',
     ['⟦the first quarters average @{b1_q1_pop_mean}\\%: a fall of a third every winter||trimestrele I au în medie @{b1_q1_pop_mean}\\%: o scădere de o treime în fiecare iarnă⟧',
      '⟦standard deviation @{b1_sd_pop} percentage points: dominated by the season||abaterea standard: @{b1_sd_pop} puncte procentuale, dominată de sezonalitate⟧']),
    ('⟦\\textbf{Year on year}: @{b1_n_yoy} values, @{b1_neg_yoy} negative||\\textbf{Față de anul anterior}: @{b1_n_yoy} de valori, dintre care @{b1_neg_yoy} negative⟧',
     ['⟦negative in the years @{b1_neg_years}||negative în anii @{b1_neg_years}⟧',
      '⟦average @{b1_yoy_mean}\\%, standard deviation @{b1_sd_yoy} percentage points||media @{b1_yoy_mean}\\%, abaterea standard @{b1_sd_yoy} puncte procentuale⟧']),
    ('⟦\\textbf{Latest quarter}, @{b1_last}||\\textbf{Ultimul trimestru}, @{b1_last}⟧',
     ['⟦q/q: $+$@{b1_last_pop}\\% (spring after winter); y/y: @{b1_last_yoy}\\%||față de trimestrul anterior: $+$@{b1_last_pop}\\% (primăvara după iarnă); față de anul anterior: @{b1_last_yoy}\\%⟧'])) + '\n' + QL,
    size='footnotesize')

D.frame(f'B1 {SOLVED}: ⟦Solution (2/2) --- Chart and Interpretation||Rezolvare (2/2) --- grafic și interpretare⟧',
        fig('ch0_sem_b1_growth', '0.5\\textheight') + items(
            '⟦Blue: q/q swings between about $-35\\%$ and $+20\\%$ every year; red: y/y moves slowly, around the business cycle||Albastru: rata față de trimestrul anterior oscilează în fiecare an între aproximativ $-35\\%$ și $+20\\%$; roșu: rata față de anul anterior se mișcă lent, odată cu ciclul economic⟧',
            '⟦\\textbf{Interpretation}: only the y/y rate shows the recessions (2009--2010, 2020) and the slowdown of 2025--2026; for unadjusted data, report y/y growth||\\textbf{Interpretare}: doar rata față de anul anterior arată recesiunile (2009--2010, 2020) și încetinirea din 2025--2026; pentru datele neajustate, raportați creșterea față de anul anterior⟧') + '\n' + QL,
        size='footnotesize')

D.task(f'B2 {SOLVED}: ⟦EUR/RON: Memory of Levels and Changes --- Task||EUR/RON: memoria nivelurilor și a variațiilor --- cerință⟧',
       '⟦does today\'s exchange rate help to forecast tomorrow\'s, and does today\'s change help to forecast tomorrow\'s change?||ajută cursul de azi la prognoza cursului de mîine? Dar variația de azi la prognoza variației de mîine?⟧',
       '⟦EUR/RON reference rate of the BNR, @{b2_first} -- @{b2_last} (@{b2_n} days)||cursul de referință EUR/RON al BNR, @{b2_first} -- @{b2_last} (@{b2_n} de zile)⟧',
       ['⟦Compute the daily log returns $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$.||Calculați randamentele logaritmice zilnice $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$.⟧',
        '⟦Compute the sample ACF of the level and of the returns for lags 1--20.||Calculați ACF de selecție a nivelului și a randamentelor pentru decalajele 1--20.⟧',
        '⟦Compare $r_1$ of the returns with the band $\\pm 1.96/\\sqrt T$, and count the lags outside the band.||Comparați $r_1$ al randamentelor cu banda $\\pm 1.96/\\sqrt T$ și numărați decalajele din afara benzii.⟧',
        '⟦Interpretation: is the naive forecast a sensible benchmark for the exchange rate?||Interpretare: este prognoza naivă un reper potrivit pentru cursul de schimb?⟧'],
       '⟦$r_1$ and $r_{20}$ of the level, $r_1$ and $r_2$ of the returns, the band, the number of lags outside it, and the answer to the interpretation question||$r_1$ și $r_{20}$ ale nivelului, $r_1$ și $r_2$ ale randamentelor, banda, numărul decalajelor din afara ei și răspunsul la întrebarea de interpretare⟧',
       size='footnotesize', nb='B2')

D.frame(f'B2 {SOLVED}: ⟦Solution (1/2) --- Step by Step||Rezolvare (1/2) --- pas cu pas⟧', items(
    ('⟦\\textbf{Level}: $r_1 = $ @{b2_acf_level_1}, $r_{20} = $ @{b2_acf_level_20}||\\textbf{Nivelul}: $r_1 = $ @{b2_acf_level_1}, $r_{20} = $ @{b2_acf_level_20}⟧',
     ['⟦the level remembers itself for weeks: a slow decay, the signature of a trend||nivelul își amintește valorile pe parcursul a mai multe săptămîni: o descreștere lentă, semnătura unui trend⟧']),
    ('⟦\\textbf{Returns}: standard deviation @{b2_ret_sd}\\% a day; $r_1 = $ @{b2_acf_ret_1}, $r_2 = $ @{b2_acf_ret_2}||\\textbf{Randamentele}: abaterea standard @{b2_ret_sd}\\% pe zi; $r_1 = $ @{b2_acf_ret_1}, $r_2 = $ @{b2_acf_ret_2}⟧',
     ['⟦band $\\pm$@{b2_band}; @{b2_ret_out} of the 20 lags lie outside it||banda $\\pm$@{b2_band}; @{b2_ret_out} din cele 20 de decalaje se află în afara ei⟧',
      '⟦$r_1$ is significant but small: yesterday\'s change explains less than 1\\% of the variance of today\'s change ($r_1^2$)||$r_1$ este semnificativ, dar mic: variația de ieri explică mai puțin de 1\\% din varianța variației de azi ($r_1^2$)⟧'])) + '\n' + QL,
    size='footnotesize')

D.frame(f'B2 {SOLVED}: ⟦Solution (2/2) --- Chart and Interpretation||Rezolvare (2/2) --- grafic și interpretare⟧',
        fig('ch0_sem_b2_acf', '0.46\\textheight') + items(
            '⟦Left: the ACF of the level stays close to 1; right: the ACF of the returns is almost flat; shaded: the band $\\pm 1.96/\\sqrt T$||Stînga: ACF a nivelului rămîne aproape de 1; dreapta: ACF a randamentelor este aproape plată; zona colorată: banda $\\pm 1.96/\\sqrt T$⟧',
            '⟦\\textbf{Interpretation}: yes; the level of today is the best simple forecast of tomorrow\'s level, while past changes add very little: the naive forecast is the benchmark to beat||\\textbf{Interpretare}: da; nivelul de azi este cea mai bună prognoză simplă a nivelului de mîine, iar variațiile trecute adaugă foarte puțin: prognoza naivă este reperul de depășit⟧') + '\n' + QL,
        size='footnotesize')

# ===============================================================================================================
D.section('Your Turn', 'Rîndul dumneavoastră')
# ===============================================================================================================
D.task(f'A4 {PROP}: ⟦New Numbers --- Task||Alte cifre --- cerință⟧',
       '⟦can you repeat A1--A3 without the solution in front of you?||puteți repeta A1--A3 fără rezolvarea în față?⟧',
       '⟦a quarterly series 120, 150, 132, 168, 126 (Q1 of year 1 to Q1 of year 2); a series 5, 7, 6, 8, 9; a series with $m = 3$: training 20, 30, 25, 22, 32, 27, test 23, 33, 29||o serie trimestrială 120, 150, 132, 168, 126 (T1 din anul 1 -- T1 din anul 2); o serie 5, 7, 6, 8, 9; o serie cu $m = 3$: antrenare 20, 30, 25, 22, 32, 27, test 23, 33, 29⟧',
       ['⟦Model: A1, A2 and A3 [Solved].||Model: A1, A2 și A3 [Rezolvat].⟧',
        '⟦Compute the period-on-period growth rates, the log differences and the year-on-year rate of the last quarter.||Calculați ratele de creștere față de perioada anterioară, diferențele logaritmilor și rata față de anul anterior a ultimului trimestru.⟧',
        '⟦Compute $r_1$ of the series 5, 7, 6, 8, 9 and the band $\\pm 1.96/\\sqrt T$.||Calculați $r_1$ pentru seria 5, 7, 6, 8, 9 și banda $\\pm 1.96/\\sqrt T$.⟧',
        '⟦Compute the MAE and the RMSE of the naive and seasonal naive forecasts of the test values.||Calculați MAE și RMSE ale prognozelor naive și naive sezoniere pentru valorile de test.⟧'],
       '⟦the growth table, $r_1$ and the band, the four error measures, and one sentence naming the better forecast||tabelul ratelor de creștere, $r_1$ și banda, cele patru măsuri ale erorii și o propoziție care numește prognoza mai bună⟧',
       size='footnotesize', nb='A4')

D.frame('A4: ⟦Solution (Instructor)||Rezolvare (profesor)⟧', items(
    ('⟦\\textbf{Growth}||\\textbf{Creștere}⟧',
     ['⟦period on period, \\%: $+$@{a4_pop1}, @{a4_pop2}, $+$@{a4_pop3}, @{a4_pop4}; log differences: $+$@{a4_dlog1}, @{a4_dlog2}, $+$@{a4_dlog3}, @{a4_dlog4}||față de perioada anterioară, \\%: $+$@{a4_pop1}, @{a4_pop2}, $+$@{a4_pop3}, @{a4_pop4}; diferențele logaritmilor: $+$@{a4_dlog1}, @{a4_dlog2}, $+$@{a4_dlog3}, @{a4_dlog4}⟧',
      '⟦year on year (Q1 of year 2): $100\\,(126/120 - 1) = +$@{a4_yoy}\\%||față de anul anterior (T1 din anul 2): $100\\,(126/120 - 1) = +$@{a4_yoy}\\%⟧']),
    ('⟦\\textbf{Autocorrelation}: $\\bar y = 7$, $\\sum d^2 = 10$, $\\sum d_t d_{t-1} = 1$||\\textbf{Autocorelația}: $\\bar y = 7$, $\\sum d^2 = 10$, $\\sum d_t d_{t-1} = 1$⟧',
     ['⟦$r_1 = $ @{a4_r1}; band $\\pm$@{a4_band}: not distinguishable from 0||$r_1 = $ @{a4_r1}; banda $\\pm$@{a4_band}: nu se deosebește de 0⟧']),
    ('⟦\\textbf{Forecasts} (naive: 27, 27, 27; seasonal naive: 22, 32, 27)||\\textbf{Prognoze} (naivă: 27, 27, 27; naivă sezonieră: 22, 32, 27)⟧',
     ['⟦naive: MAE @{a4_naive_mae}, RMSE @{a4_naive_rmse}; seasonal naive: MAE @{a4_snaive_mae}, RMSE @{a4_snaive_rmse}; the seasonal naive wins||naivă: MAE @{a4_naive_mae}, RMSE @{a4_naive_rmse}; naivă sezonieră: MAE @{a4_snaive_mae}, RMSE @{a4_snaive_rmse}; cîștigă prognoza naivă sezonieră⟧'])),
    instructor_only=True, size='footnotesize')

D.task(f'B3 {PROP}: ⟦Romanian Inflation --- Task||Inflația din România --- cerință⟧',
       '⟦does monthly inflation have a seasonal pattern that the annual rate hides?||are inflația lunară un tipar sezonier pe care rata anuală îl ascunde?⟧',
       '⟦Romanian HICP, 2015 = 100, monthly, @{b3_first} -- @{b3_last} (Eurostat)||IAPC al României, 2015 = 100, lunar, @{b3_first} -- @{b3_last} (Eurostat)⟧',
       ['⟦Model: B1 and B2 [Solved].||Model: B1 și B2 [Rezolvat].⟧',
        '⟦Compute the monthly (m/m) and the annual (y/y) inflation rates in \\%.||Calculați rata lunară și rata anuală a inflației, în \\%.⟧',
        '⟦Compute the average monthly inflation for each calendar month.||Calculați inflația lunară medie pentru fiecare lună calendaristică.⟧',
        '⟦Compute the ACF of the monthly rate for lags 1--24, and of the annual rate at lag 1.||Calculați ACF a ratei lunare pentru decalajele 1--24 și pe cea a ratei anuale la decalajul 1.⟧',
        '⟦Interpretation: why is the annual rate so persistent?||Interpretare: de ce este rata anuală atît de persistentă?⟧'],
       '⟦the latest monthly and annual rates, the months with the highest and lowest average, $r_1$ and $r_{12}$ of the monthly rate, $r_1$ of the annual rate, and the answer to the interpretation question||ultimele rate lunară și anuală, lunile cu media cea mai mare și cea mai mică, $r_1$ și $r_{12}$ ale ratei lunare, $r_1$ al ratei anuale și răspunsul la întrebarea de interpretare⟧',
       size='footnotesize', nb='B3')

D.frame('B3: ⟦Solution (Instructor)||Rezolvare (profesor)⟧', cols(
    fig('ch0_sem_b3_acf', '0.42\\textheight'),
    items('⟦@{b3_last_m}: monthly $+$@{b3_last_pop}\\%, annual @{b3_last_yoy}\\%; annual peak @{b3_max_yoy}\\% in @{b3_max_m}||@{b3_last_m}: lunar $+$@{b3_last_pop}\\%, anual @{b3_last_yoy}\\%; vîrful ratei anuale: @{b3_max_yoy}\\%, în @{b3_max_m}⟧',
          '⟦highest average month: @{b3_month_max_m} (@{b3_month_max_v}\\%); lowest: @{b3_month_min_m} (@{b3_month_min_v}\\%)||luna cu media cea mai mare: @{b3_month_max_m} (@{b3_month_max_v}\\%); cea mai mică: @{b3_month_min_m} (@{b3_month_min_v}\\%)⟧',
          '⟦monthly rate: $r_1 = $ @{b3_acf_mm_1}, $r_{12} = $ @{b3_acf_mm_12} (band $\\pm$@{b3_band}): memory and a seasonal peak||rata lunară: $r_1 = $ @{b3_acf_mm_1}, $r_{12} = $ @{b3_acf_mm_12} (banda $\\pm$@{b3_band}): memorie și un vîrf sezonier⟧',
          '⟦annual rate: $r_1 = $ @{b3_acf_yy_1}: two consecutive annual rates share 11 of their 12 monthly changes (as in Slutsky\'s moving sums)||rata anuală: $r_1 = $ @{b3_acf_yy_1}: două rate anuale consecutive au în comun 11 din cele 12 variații lunare (ca în sumele mobile ale lui Slutsky)⟧'),
    '0.5', '0.47'), instructor_only=True, size='footnotesize')

D.task(f'B4 {PROP}: ⟦Electricity: Naive or Seasonal Naive? --- Task||Electricitatea: naivă sau naivă sezonieră? --- cerință⟧',
       '⟦for monthly electricity generation, does the seasonal naive forecast beat the naive one?||pentru producția lunară de electricitate, este prognoza naivă sezonieră mai bună decît cea naivă?⟧',
       '⟦net electricity generation in Romania, TWh per month (Eurostat); test set: @{b4_test_first_m} -- @{b4_test_last_m} (24 months)||producția netă de electricitate a României, TWh pe lună (Eurostat); setul de test: @{b4_test_first_m} -- @{b4_test_last_m} (24 de luni)⟧',
       ['⟦Model: A3 [Solved].||Model: A3 [Rezolvat].⟧',
        '⟦Split the series by time: the last 24 months form the test set.||Împărțiți seria în timp: ultimele 24 de luni formează setul de test.⟧',
        '⟦Forecast the test set with the naive and with the seasonal naive method ($m = 12$), using only the training set.||Prognozați setul de test cu metoda naivă și cu metoda naivă sezonieră ($m = 12$), folosind doar setul de antrenare.⟧',
        '⟦Compute the MAE and the RMSE of each method, and plot both forecasts against the test data.||Calculați MAE și RMSE pentru fiecare metodă și reprezentați grafic ambele prognoze, alături de datele de test.⟧',
        '⟦Interpretation: why does the seasonal naive method not win clearly here?||Interpretare: de ce metoda naivă sezonieră nu cîștigă clar aici?⟧'],
       '⟦the four error measures, the chart, and the answer to the interpretation question||cele patru măsuri ale erorii, graficul și răspunsul la întrebarea de interpretare⟧',
       size='footnotesize', nb='B4')

D.frame('B4: ⟦Solution (Instructor)||Rezolvare (profesor)⟧', cols(
    fig('ch0_sem_b4_forecast', '0.42\\textheight'),
    items('⟦naive (all forecasts @{b4_last_train} TWh): MAE @{b4_naive_mae}, RMSE @{b4_naive_rmse}||naivă (toate prognozele @{b4_last_train} TWh): MAE @{b4_naive_mae}, RMSE @{b4_naive_rmse}⟧',
          '⟦seasonal naive: MAE @{b4_snaive_mae}, RMSE @{b4_snaive_rmse}||naivă sezonieră: MAE @{b4_snaive_mae}, RMSE @{b4_snaive_rmse}⟧',
          '⟦the two are almost equal: the seasonal naive copies the strong winter of the training year into a lower test period, so its winter errors are large||cele două sînt aproape egale: metoda naivă sezonieră copiază iarna puternică din anul de antrenare peste o perioadă de test cu nivel mai scăzut, deci erorile ei de iarnă sînt mari⟧',
          '⟦a method that combines season and level (Holt--Winters, in the lecture) does better||o metodă care combină sezonalitatea și nivelul (Holt--Winters, la curs) se descurcă mai bine⟧'),
    '0.5', '0.47'), instructor_only=True, size='footnotesize')

# ===============================================================================================================
D.section('Part C: Open Questions', 'Partea C: întrebări deschise')
# ===============================================================================================================
D.frame('C1: ⟦A Project Idea||O idee de proiect⟧', items(
    '⟦\\textbf{Question}: how much of the variation of a Romanian monthly series is seasonal, and is the seasonal pattern changing?||\\textbf{Întrebarea}: cît din variația unei serii lunare românești este sezonieră și se schimbă tiparul sezonier în timp?⟧',
    ('⟦\\textbf{Candidate series} (Eurostat or INS)||\\textbf{Serii posibile} (Eurostat sau INS)⟧',
     ['⟦electricity generation, tourist arrivals, retail trade, industrial production||producția de electricitate, sosirile de turiști, comerțul cu amănuntul, producția industrială⟧']),
    ('⟦\\textbf{Steps}||\\textbf{Pași}⟧',
     ['⟦plot the series and its year-on-year growth||reprezentați grafic seria și creșterea ei față de anul anterior⟧',
      '⟦compare the average of each calendar month in the first and in the last five years||comparați media fiecărei luni calendaristice în primii și în ultimii cinci ani⟧',
      '⟦after the lecture: decompose the series with STL and compare the seasonal amplitude over time||după curs: descompuneți seria cu STL și comparați amplitudinea sezonieră în timp⟧']),
    '⟦A good first step for the team project of Chapters 0 and 4||Un prim pas bun pentru proiectul de echipă, legat de Capitolele 0 și 4⟧'))

D.frame(f'C2 {PROP}: ⟦Critique an AI Answer --- the Answer||Critica unui răspuns generat de AI --- răspunsul⟧', items(
    '⟦\\textbf{Prompt} sent to an AI assistant: ``From the quarterly Romanian real GDP, compute the average annual growth rate and the RMSE of a naive forecast.\'\'||\\textbf{Cererea} trimisă unui asistent AI: „Din PIB-ul real trimestrial al României, calculează rata medie anuală de creștere și RMSE pentru o prognoză naivă.”⟧') + r"""
\begin{lstlisting}
g = read_eurostat('namq_10_gdp', 'Q.CLV10_MEUR.NSA.B1GQ.RO') / 1000
growth = g.diff()                                # annual growth rate, %
print('average annual growth:', growth.mean())
test = g.sample(frac=0.2, random_state=1).sort_index()   # test set
train = g.drop(test.index)                               # training set
fc = train.reindex(g.index).ffill().shift(1).reindex(test.index)  # naive
rmse = ((test - fc) ** 2).mean()
print('RMSE of the naive forecast:', rmse)
\end{lstlisting}
""" + items('⟦\\textbf{Conclusion of the AI}: ``Romanian GDP grows by only @{c2_ai_growth}\\% a year, and the naive forecast has an RMSE of @{c2_ai_rmse} billion EUR, larger than GDP itself: forecasting GDP is hopeless.\'\'||\\textbf{Concluzia AI}: „PIB-ul României crește cu doar @{c2_ai_growth}\\% pe an, iar prognoza naivă are un RMSE de @{c2_ai_rmse} miliarde EUR, mai mare decît PIB-ul însuși: prognoza PIB este fără speranță.”⟧'),
    opts='fragile', size='footnotesize')

D.task(f'C2 {PROP}: ⟦Critique an AI Answer --- Task||Critica unui răspuns generat de AI --- cerință⟧',
       '⟦can you trust code that runs without an error message?||puteți avea încredere într-un cod care rulează fără niciun mesaj de eroare?⟧',
       '⟦the code and the conclusion on the previous slide; the definitions of the primer||codul și concluzia de pe slide-ul anterior; definițiile din noțiunile necesare⟧',
       ['⟦Model: A1, A3 and B1 [Solved].||Model: A1, A3 și B1 [Rezolvat].⟧',
        '⟦Find the three errors planted in the code.||Găsiți cele trei greșeli introduse intenționat în cod.⟧',
        '⟦For each error, say how you would detect it: a check on the output, the primer, or a solved exercise.||Pentru fiecare greșeală, precizați cum ați detecta-o: o verificare a rezultatului, noțiunile necesare sau un exercițiu rezolvat.⟧',
        '⟦Write the corrected code, with the last 8 quarters as the test set, and report the corrected numbers.||Scrieți codul corectat, cu ultimele 8 trimestre ca set de test, și raportați rezultatele corectate.⟧',
        '⟦Say whether the conclusion ``forecasting GDP is hopeless\'\' survives.||Precizați dacă mai rămîne valabilă concluzia „prognoza PIB este fără speranță”.⟧'],
       '⟦one line per error in the format: request, answer, error, how found, correction; the corrected growth and RMSE||cîte o linie pentru fiecare greșeală, în formatul: cerere, răspuns, greșeală, cum a fost găsită, corectare; creșterea și RMSE corectate⟧',
       nb='C2')

D.frame('C2: ⟦Solution (Instructor)||Rezolvare (profesor)⟧', items(
    ('⟦\\textbf{Error 1}: \\texttt{g.diff()} is a change in billion EUR over one quarter, not an annual growth rate in \\%||\\textbf{Greșeala 1}: \\texttt{g.diff()} este o variație în miliarde EUR pe un trimestru, nu o rată anuală de creștere în \\%⟧',
     ['⟦detect: units (A1, B1); correct: \\texttt{100 * (g / g.shift(4) - 1)}, average @{c2_ok_growth}\\% a year||detectare: unitățile de măsură (A1, B1); corectare: \\texttt{100 * (g / g.shift(4) - 1)}, media @{c2_ok_growth}\\% pe an⟧']),
    ('⟦\\textbf{Error 2}: a random test set; earlier test quarters are forecast with later data||\\textbf{Greșeala 2}: un set de test aleator; trimestrele de test mai vechi sînt prognozate cu date ulterioare⟧',
     ['⟦detect: the primer (split by time); correct: \\texttt{train, test = g.iloc[:-8], g.iloc[-8:]}||detectare: noțiunile necesare (împărțire în timp); corectare: \\texttt{train, test = g.iloc[:-8], g.iloc[-8:]}⟧']),
    ('⟦\\textbf{Error 3}: the mean squared error without the square root, in (billion EUR)$^2$||\\textbf{Greșeala 3}: eroarea pătratică medie fără rădăcina pătrată, în (miliarde EUR)$^2$⟧',
     ['⟦detect: an RMSE larger than the series itself (A3); correct: \\texttt{np.sqrt(...)}||detectare: un RMSE mai mare decît seria însăși (A3); corectare: \\texttt{np.sqrt(...)}⟧']),
    '⟦Corrected, last 8 quarters: RMSE @{c2_ok_rmse_naive} billion EUR (naive), @{c2_ok_rmse_snaive} (seasonal naive): the conclusion fails; a seasonal benchmark forecasts GDP within about 1\\%||Corectat, ultimele 8 trimestre: RMSE @{c2_ok_rmse_naive} miliarde EUR (naivă), @{c2_ok_rmse_snaive} (naivă sezonieră): concluzia nu se susține; un reper sezonier prognozează PIB-ul cu o eroare de aproximativ 1\\%⟧'),
    instructor_only=True, size='footnotesize')

# ===============================================================================================================
D.section('Wrap-up', 'Încheiere')
# ===============================================================================================================
D.frame('⟦Practice at Home||Exerciții pentru acasă⟧', items(
    ('⟦\\textbf{Finish the proposed exercises}: A4, B3, B4 and C2||\\textbf{Terminați exercițiile propuse}: A4, B3, B4 și C2⟧',
     ['⟦each names its model, a solved exercise to follow||fiecare indică exercițiul rezolvat care servește drept model⟧']),
    ('⟦\\textbf{Try one more series}||\\textbf{Încercați încă o serie}⟧',
     ['⟦repeat B1 for the euro area GDP (key \\texttt{Q.CLV10\\_MEUR.NSA.B1GQ.EA20}) and compare the seasonal swings with Romania||repetați B1 pentru PIB-ul zonei euro (cheia \\texttt{Q.CLV10\\_MEUR.NSA.B1GQ.EA20}) și comparați oscilațiile sezoniere cu cele ale României⟧']),
    '⟦\\textbf{Take the Chapter 0 quiz} on the course website: 20 questions, each answer explained||\\textbf{Rezolvați quiz-ul Capitolului 0} pe site-ul cursului: 20 de întrebări, cu explicație pentru fiecare răspuns⟧'))

D.frame('⟦Recap, and Next: the Chapter 0 Lecture||Recapitulare; urmează cursul din Capitolul 0⟧', cols(
    items(('⟦\\textbf{Today}||\\textbf{Azi}⟧',
           ['⟦for unadjusted data, report growth over a year, not over one quarter||pentru datele neajustate, raportați creșterea pe un an, nu pe un trimestru⟧',
            '⟦$r_k$: how much a series remembers its value $k$ periods earlier||$r_k$: cît își amintește o serie valoarea de acum $k$ perioade⟧',
            '⟦levels with a trend remember a lot; daily changes of prices remember little||nivelurile cu trend au memorie lungă; variațiile zilnice ale prețurilor au memorie redusă⟧',
            '⟦the naive and seasonal naive forecasts are the benchmarks; split by time, then measure MAE and RMSE||prognozele naivă și naivă sezonieră sînt reperele; împărțiți în timp, apoi măsurați MAE și RMSE⟧',
            '⟦AI code can run and still be wrong||codul generat de AI poate rula și totuși să fie greșit⟧'])),
    items(('⟦\\textbf{In the lecture}||\\textbf{La curs}⟧',
           ['⟦How is the course organised and graded?||Cum este organizat și evaluat cursul?⟧',
            '⟦Which components make up a time series?||Ce componente alcătuiesc o serie de timp?⟧',
            '⟦How does exponential smoothing forecast?||Cum prognozează netezirea exponențială?⟧',
            '⟦Which forecasts win on real Romanian data?||Ce prognoze cîștigă pe datele reale ale României?⟧']),
          CLOSING),
    '0.52', '0.45'), size='footnotesize')

if __name__ == '__main__':
    D.write(V)
