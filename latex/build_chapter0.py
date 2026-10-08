r"""
build_chapter0.py -- Capitolul 0 (Introducere: componente și netezire exponențială), EN + RO dintr-o singură sursă
=================================================================================================================
Generator bilingv (Deck din latex/tsa_build.py, text ⟦EN||RO⟧). Cifrele vin din Quantlets/Ch_00/ch0_values.json
(Quantlets/Ch_00/generate_all_charts.py); graficele din charts/tsa_ch0_*.pdf; fotografiile din photos/ch0_*.jpg
(licențe verificate prin API-ul Wikimedia Commons; vezi photos/CREDITS.md).
Înlocuiește conversia deck-urilor din 2025/2026 (EN/Courses/chapter0_fundamentals.tex rămîne neschimbat).
Ieșire:
  EN/Courses/chapter0_introduction.tex
  RO/Cursuri/capitol0_introducere.tex
Rulare:  python3 Quantlets/Ch_00/generate_all_charts.py
         python3 latex/build_chapter0.py   apoi   python3 latex/tsa_build.py compile 0
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import MARK, ROOT, Deck, Values, items, cols, table, photo   # noqa: E402
from tsa_chapters import TITLES, SELF_STUDY                               # noqa: E402

VAL = json.load(open(os.path.join(ROOT, 'Quantlets', 'Ch_00', 'ch0_values.json')))

MONTHS = {'en': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                 'November', 'December'],
          'ro': ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie',
                 'octombrie', 'noiembrie', 'decembrie']}


def dt(s, day=True):
    """Data ISO -> ⟦9 March 2009||9 martie 2009⟧ (day=False: ⟦March 2009||martie 2009⟧)."""
    y, m, d = s.split('-')
    en, ro = MONTHS['en'][int(m) - 1], MONTHS['ro'][int(m) - 1]
    return f'⟦{int(d)} {en} {y}||{int(d)} {ro} {y}⟧' if day else f'⟦{en} {y}||{ro} {y}⟧'


def month(i):
    return f'⟦{MONTHS["en"][i - 1]}||{MONTHS["ro"][i - 1]}⟧'


def quarter(s):
    """'2024 Q1' -> ⟦2024 Q1||T1 2024⟧."""
    y, q = s.split(' ')
    return f'⟦{y} {q}||T{q[1]} {y}⟧'


class LangValues(Values):
    """Values whose entries may hold ⟦EN||RO⟧ text (dates), resolved for the language being rendered."""
    lang = 'en'

    def __getitem__(self, key):
        v = super().__getitem__(key)
        return MARK.sub(lambda m: m.group(1) if self.lang == 'en' else m.group(2), v) if isinstance(v, str) else v


class LangDeck(Deck):
    def frame(self, title, body, *a, **k):
        # '\\pause' given as a list element: a pause between two bullets, not an empty bullet
        super().frame(title, body.replace('    \\item \\pause\n', '    \\pause\n'), *a, **k)

    def head(self, lang):
        V.lang = lang
        return super().head(lang)


V = LangValues()
DEC = {  # decimals of each number shown on the slides
    'gdp_last_nsa': 1, 'gdp_last_sca': 1, 'gdp_mult': 1, 'gdp_growth_ann': 1, 'gdp_fall_2009': 1, 'gdp_fall_2020': 1,
    'hicp_last': 1, 'infl_last': 1, 'infl_first': 0, 'infl_max': 0, 'infl_min10': 1, 'infl_peak22': 1,
    'eurron_first': 4, 'eurron_last': 4, 'eurron_min': 4, 'eurron_maxmove': 2, 'eurron_sd': 2,
    'bet_mult': 0, 'sp_mult': 1, 'elec_mean': 2, 'elec_max_mean': 2, 'elec_min_mean': 2, 'elec_ratio': 2,
    'elec_2008': 1, 'elec_2025': 1, 'co2_first': 1, 'co2_last': 1, 'co2_slope': 2,
    'gdp_S1': 3, 'gdp_S2': 3, 'gdp_S3': 3, 'gdp_S4': 3, 'gdp_S1_pct': 1, 'gdp_S2_pct': 1, 'gdp_S3_pct': 1,
    'gdp_S4_pct': 1, 'gdp_rem_sd': 1, 'ma_2x4': 2, 'co2_amp_first': 1, 'co2_amp_last': 1, 'co2_resid_sd': 2,
    'bet_sd': 2, 'bet_mean': 3, 'bet_min': 1, 'bet_max': 1, 'acf_co2_1': 3, 'acf_co2_36': 2, 'acf_el_1': 2,
    'acf_el_6': 2, 'acf_el_12': 2, 'acf_bet_1': 3, 'acf_bet_2': 3, 'band_bet': 3, 'band_el': 2, 'acf_absbet_1': 2,
    'ses_alpha': 2, 'ses_last_y': 4,
}
for k, x in VAL.items():
    if isinstance(x, list):
        continue
    if isinstance(x, str):
        if len(x) == 10 and x[4] == '-':
            V.raw(k, dt(x))
        elif ' Q' in x:
            V.raw(k, quarter(x))
        else:
            V.raw(k, x)
    elif isinstance(x, int):
        V.raw(k, str(x)) if k.startswith('gdp_year') else V.int(k, x)
    elif k.startswith(('fc_', 'bm_')):
        V.put(k, x, 1 if k.endswith('mape') else 2)
    else:
        V.put(k, x, DEC.get(k, 2))
for i, (yv, dq) in enumerate(zip(VAL['ma_y'], VAL['ma_dates'])):
    V.put(f'ma_y{i}', yv, 1)
    V.raw(f'ma_d{i}', quarter(dq))
V.raw('elec_max_m', month(VAL['elec_max_month']))
V.raw('elec_min_m', month(VAL['elec_min_month']))
V.raw('co2_max_m', month(VAL['co2_seas_max_month']))
V.raw('co2_min_m', month(VAL['co2_seas_min_month']))
for k in ('hicp_last_date', 'infl_max_date', 'infl_min10_date', 'infl_peak22_date', 'fc_test_first', 'fc_test_last',
          'elec_first', 'elec_last', 'co2_first_date', 'co2_last_date', 'ses_first', 'ses_last'):
    V.raw(k + '_m', dt(VAL[k], day=False))
V.put('gdp_fall_2009_abs', -VAL['gdp_fall_2009'], 1)
V.put('gdp_fall_2020_abs', -VAL['gdp_fall_2020'], 1)
V.put('bet_sd_ann', VAL['bet_sd'] * 252 ** 0.5, 0)
V.put('gdp_S1_abs', -VAL['gdp_S1_pct'], 1)
V.put('gdp_q1_rel', 100 * (VAL['gdp_S4'] / VAL['gdp_S1'] - 1), 0)

# ---------------------------------------------------------------------------------------------------------------
# Citări (DOI verificate prin Crossref; pagini oficiale cu HTTP 200)
# ---------------------------------------------------------------------------------------------------------------
REFS = r"""
\newcommand{\refHP}{\href{https://doi.org/10.1007/978-3-031-13584-2}{Huang \& Petukhina (2022)}}
\newcommand{\refFPP}{\href{https://otexts.com/fpp3/}{Hyndman \& Athanasopoulos (2021)}}
\newcommand{\refBD}{\href{https://doi.org/10.1007/978-3-319-29854-2}{Brockwell \& Davis (2016)}}
\newcommand{\refHamilton}{\href{https://doi.org/10.2307/j.ctv14jx6sm}{Hamilton (1994)}}
\newcommand{\refYuleNonsense}{\href{https://doi.org/10.2307/2341482}{Yule (1926)}}
\newcommand{\refYule}{\href{https://doi.org/10.1098/rsta.1927.0007}{Yule (1927)}}
\newcommand{\refSlutsky}{\href{https://doi.org/10.2307/1907241}{Slutzky (1937)}}
\newcommand{\refKolmogorov}{\href{https://doi.org/10.1007/978-94-011-2260-3_28}{Kolmogorov (1941)}}
\newcommand{\refWiener}{\href{https://doi.org/10.7551/mitpress/2946.001.0001}{Wiener (1949)}}
\newcommand{\refBJ}{\href{https://doi.org/10.1002/9781118619193}{Box \& Jenkins (1970)}}
\newcommand{\refHolt}{\href{https://doi.org/10.1016/j.ijforecast.2003.09.015}{Holt (1957)}}
\newcommand{\refWinters}{\href{https://doi.org/10.1287/mnsc.6.3.324}{Winters (1960)}}
\newcommand{\refGardner}{\href{https://doi.org/10.1002/for.3980040103}{Gardner (1985)}}
\newcommand{\refHKSG}{\href{https://doi.org/10.1016/S0169-2070(01)00110-8}{Hyndman, Koehler, Snyder \& Grose (2002)}}
\newcommand{\refHK}{\href{https://doi.org/10.1016/j.ijforecast.2006.03.001}{Hyndman \& Koehler (2006)}}
\newcommand{\refMthree}{\href{https://doi.org/10.1016/S0169-2070(00)00057-1}{Makridakis \& Hibon (2000)}}
\newcommand{\refMfour}{\href{https://doi.org/10.1016/j.ijforecast.2019.04.014}{Makridakis, Spiliotis \& Assimakopoulos (2020)}}
\newcommand{\refSTL}{\href{https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/stl-a-seasonal-trend-decomposition-procedure-based-on-loess.pdf}{Cleveland et al.\ (1990)}}
\newcommand{\refKeeling}{\href{https://doi.org/10.3402/tellusa.v12i2.9366}{Keeling (1960)}}
\newcommand{\refEngle}{\href{https://doi.org/10.2307/1912773}{Engle (1982)}}
\newcommand{\refSims}{\href{https://doi.org/10.2307/1912017}{Sims (1980)}}
\newcommand{\refEG}{\href{https://doi.org/10.2307/1913236}{Engle \& Granger (1987)}}
\newcommand{\refStatsmodels}{\href{https://doi.org/10.25080/Majora-92bf1922-011}{Seabold \& Perktold (2010)}}
\newcommand{\refPandas}{\href{https://doi.org/10.25080/Majora-92bf1922-00a}{McKinney (2010)}}
\newcommand{\refNumpy}{\href{https://doi.org/10.1038/s41586-020-2649-2}{Harris et al.\ (2020)}}
"""

REFERENCES = [
    r"Box, G.E.P., Jenkins, G.M. (1970). \emph{Time Series Analysis: Forecasting and Control}. Holden-Day; 4th ed.\ with G.C.\ Reinsel: \href{https://doi.org/10.1002/9781118619193}{Wiley (2008)}.",
    r"Brockwell, P.J., Davis, R.A. (2016). \href{https://doi.org/10.1007/978-3-319-29854-2}{\emph{Introduction to Time Series and Forecasting}} (3rd ed.). Springer.",
    r"Cleveland, R.B., Cleveland, W.S., McRae, J.E., Terpenning, I. (1990). \href{https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/stl-a-seasonal-trend-decomposition-procedure-based-on-loess.pdf}{STL: a seasonal-trend decomposition procedure based on loess}. \emph{Journal of Official Statistics}, 6(1), 3--73.",
    r"Engle, R.F. (1982). \href{https://doi.org/10.2307/1912773}{Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation}. \emph{Econometrica}, 50(4), 987--1007.",
    r"Engle, R.F., Granger, C.W.J. (1987). \href{https://doi.org/10.2307/1913236}{Co-integration and error correction: representation, estimation, and testing}. \emph{Econometrica}, 55(2), 251--276.",
    r"Gardner, E.S. (1985). \href{https://doi.org/10.1002/for.3980040103}{Exponential smoothing: the state of the art}. \emph{Journal of Forecasting}, 4(1), 1--28.",
    r"Hamilton, J.D. (1994). \href{https://doi.org/10.2307/j.ctv14jx6sm}{\emph{Time Series Analysis}}. Princeton University Press.",
    r"Harris, C.R., Millman, K.J., van der Walt, S.J., et al.\ (2020). \href{https://doi.org/10.1038/s41586-020-2649-2}{Array programming with NumPy}. \emph{Nature}, 585, 357--362.",
    r"Holt, C.C. (1957; reprinted 2004). \href{https://doi.org/10.1016/j.ijforecast.2003.09.015}{Forecasting seasonals and trends by exponentially weighted moving averages}. \emph{International Journal of Forecasting}, 20(1), 5--10.",
    r"Huang, C., Petukhina, A. (2022). \href{https://doi.org/10.1007/978-3-031-13584-2}{\emph{Applied Time Series Analysis and Forecasting with Python}}. Springer.",
    r"Hyndman, R.J., Athanasopoulos, G. (2021). \href{https://otexts.com/fpp3/}{\emph{Forecasting: Principles and Practice}} (3rd ed.). OTexts.",
    r"Hyndman, R.J., Koehler, A.B. (2006). \href{https://doi.org/10.1016/j.ijforecast.2006.03.001}{Another look at measures of forecast accuracy}. \emph{International Journal of Forecasting}, 22(4), 679--688.",
    r"Hyndman, R.J., Koehler, A.B., Snyder, R.D., Grose, S. (2002). \href{https://doi.org/10.1016/S0169-2070(01)00110-8}{A state space framework for automatic forecasting using exponential smoothing methods}. \emph{International Journal of Forecasting}, 18(3), 439--454.",
    r"Keeling, C.D. (1960). \href{https://doi.org/10.3402/tellusa.v12i2.9366}{The concentration and isotopic abundances of carbon dioxide in the atmosphere}. \emph{Tellus}, 12(2), 200--203.",
    r"Kolmogorov, A.N. (1941). Interpolation and extrapolation of stationary random sequences. English translation in \href{https://doi.org/10.1007/978-94-011-2260-3_28}{\emph{Selected Works of A.N.\ Kolmogorov}, Vol.\ II}, Springer (1992), 272--280.",
    r"Makridakis, S., Hibon, M. (2000). \href{https://doi.org/10.1016/S0169-2070(00)00057-1}{The M3-Competition: results, conclusions and implications}. \emph{International Journal of Forecasting}, 16(4), 451--476.",
    r"Makridakis, S., Spiliotis, E., Assimakopoulos, V. (2020). \href{https://doi.org/10.1016/j.ijforecast.2019.04.014}{The M4 Competition: 100,000 time series and 61 forecasting methods}. \emph{International Journal of Forecasting}, 36(1), 54--74.",
    r"McKinney, W. (2010). \href{https://doi.org/10.25080/Majora-92bf1922-00a}{Data structures for statistical computing in Python}. \emph{Proceedings of the 9th Python in Science Conference}, 56--61.",
    r"Seabold, S., Perktold, J. (2010). \href{https://doi.org/10.25080/Majora-92bf1922-011}{Statsmodels: econometric and statistical modeling with Python}. \emph{Proceedings of the 9th Python in Science Conference}, 92--96.",
    r"Sims, C.A. (1980). \href{https://doi.org/10.2307/1912017}{Macroeconomics and reality}. \emph{Econometrica}, 48(1), 1--48.",
    r"Slutzky, E. (1937). \href{https://doi.org/10.2307/1907241}{The summation of random causes as the source of cyclic processes}. \emph{Econometrica}, 5(2), 105--146.",
    r"Wiener, N. (1949). \href{https://doi.org/10.7551/mitpress/2946.001.0001}{\emph{Extrapolation, Interpolation, and Smoothing of Stationary Time Series}}. MIT Press.",
    r"Winters, P.R. (1960). \href{https://doi.org/10.1287/mnsc.6.3.324}{Forecasting sales by exponentially weighted moving averages}. \emph{Management Science}, 6(3), 324--342.",
    r"Yule, G.U. (1926). \href{https://doi.org/10.2307/2341482}{Why do we sometimes get nonsense-correlations between time-series?} \emph{Journal of the Royal Statistical Society}, 89(1), 1--63.",
    r"Yule, G.U. (1927). \href{https://doi.org/10.1098/rsta.1927.0007}{On a method of investigating periodicities in disturbed series, with special reference to Wolfer's sunspot numbers}. \emph{Philosophical Transactions of the Royal Society A}, 226, 267--298.",
]

# ---------------------------------------------------------------------------------------------------------------
# Fotografii (Wikimedia Commons; licența verificată prin API)
# ---------------------------------------------------------------------------------------------------------------
C = 'https://commons.wikimedia.org/wiki/File:'
PHOTOS = {
    'ase': ('ch0_ase_2014.jpg', C + 'Bucharest_-_Academie_de_Studii_Economice_01.jpg',
            '⟦Photo||Foto⟧: Joe Mabel (2014); CC BY 3.0; Wikimedia Commons'),
    'wolf': ('ch0_wolf.jpg', C + 'ETH-BIB-Wolf,_Johann_Rudolf_(1816-1893)-Portrait-Portr_12033-RE.tif_(cropped).jpg',
             '⟦Photo||Foto⟧: Emil Gassler (1888--1893); ⟦public domain||domeniu public⟧; ETH-Bibliothek, Wikimedia Commons'),
    'sunspots': ('ch0_sunspots_1859.jpg', C + 'Carrington_Richard_drawing_of_1859_sunspots.jpeg',
                 '⟦Drawing||Desen⟧: Richard Carrington (1859); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'slutsky': ('ch0_slutsky.jpg', C + '\\%D0\\%A1\\%D0\\%BB\\%D1\\%83\\%D1\\%86\\%D1\\%8C\\%D0\\%BA\\%D0\\%B8\\%D0\\%B9_\\%D0\\%84\\%D0\\%B2\\%D0\\%B3\\%D0\\%B5\\%D0\\%BD.jpg',
                '⟦Photo||Foto⟧: ⟦unknown author||autor necunoscut⟧; ⟦public domain||domeniu public⟧; Wikimedia Commons'),
    'kolmogorov': ('ch0_kolmogorov.jpg', C + 'Andrej_Nikolajewitsch_Kolmogorov.jpg',
                   '⟦Photo||Foto⟧: Konrad Jacobs; CC BY-SA 2.0 de; Oberwolfach, Wikimedia Commons'),
    'wiener': ('ch0_wiener.jpg', C + 'Norbert_wiener.jpg',
               '⟦Photo||Foto⟧: Konrad Jacobs; CC BY-SA 2.0 de; Oberwolfach, Wikimedia Commons'),
    'box': ('ch0_box.jpg', C + 'GeorgeEPBox_(cropped).jpg',
            '⟦Photo||Foto⟧: DavidMCEddy (2011); CC BY-SA 3.0; Wikimedia Commons'),
    'keeling': ('ch0_keeling_2001.jpg', C + 'Charles_David_Keeling_2001.jpg',
                '⟦Photo||Foto⟧: National Science Foundation (2001); ⟦public domain||domeniu public⟧; Wikimedia Commons'),
}


def ph(key, cap, h='0.5\\textheight'):
    f, url, cred = PHOTOS[key]
    return photo(f, cap, url, cred, h=h)


def ql(folder):
    return f'\\quantlet{{{folder.replace("_", chr(92) + "_")}}}{{\\qlurl{{{folder}}}}}'


def chart(D, title, fig, folder, bullets, h='0.6\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.94\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-2mm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def clean(tex):
    """Drop the empty sub-lists left by items((text, []))."""
    return tex.replace('    \\begin{itemize}\n    \\end{itemize}\n', '')


QUESTION = '⟦Question for the room||Întrebare pentru sală⟧'
ANSWER = '⟦Answer||Răspuns⟧'

D = LangDeck(0, 'lecture', refs=REFS)

# ===============================================================================================================
D.section('Course organisation', 'Organizarea cursului')
# ===============================================================================================================
D.frame('⟦Route of this chapter||Traseul acestui capitol⟧', cols(
    items(('⟦\\textbf{Guiding questions}||\\textbf{Întrebări de pornire}⟧',
           ['⟦what is a time series, and where do we meet one?||ce este o serie de timp și unde o întîlnim?⟧',
            '⟦which patterns repeat in economic, financial, energy and climate data?||ce tipare se repetă în datele economice, financiare, energetice și climatice?⟧',
            '⟦how do we forecast a series, and how do we know the forecast is good?||cum prognozăm o serie și de unde știm că prognoza este bună?⟧']),
          ('⟦\\textbf{Part I}||\\textbf{Partea I}⟧',
           ['⟦how the course works||organizarea cursului⟧',
            '⟦time series on real data and a short history||serii de timp pe date reale și o scurtă istorie⟧',
            '⟦components: trend, seasonality, cycle, noise||componentele: trend, sezonalitate, ciclu, zgomot⟧']),
          ('⟦\\textbf{Part II}||\\textbf{Partea a II-a}⟧',
           ['⟦notation, transformations and a first look at the ACF||notație, transformări și o primă privire asupra ACF⟧',
            '⟦exponential smoothing; forecasts and their evaluation||netezirea exponențială; prognoze și evaluarea lor⟧',
            '⟦tools, and the possible contribution of AI||instrumente și contribuția posibilă a AI⟧'])),
    items(('⟦\\textbf{Learning outcomes}: after this chapter you can||\\textbf{Rezultatele învățării}: după acest capitol puteți⟧',
           ['⟦explain how the course is organised and graded||explica felul în care este organizat și evaluat cursul⟧',
            '⟦recognise trend, seasonality, cycle and noise in a chart||recunoaște într-un grafic trendul, sezonalitatea, ciclul și zgomotul⟧',
            '⟦compute growth rates, a moving average and a sample autocorrelation||calcula rate de creștere, o medie mobilă și o autocorelație de selecție⟧',
            '⟦produce naive and exponential smoothing forecasts||construi prognoze naive și prognoze prin netezire exponențială⟧',
            '⟦evaluate forecasts on a test set with MAE, RMSE and MASE||evalua prognozele pe un set de test, cu MAE, RMSE și MASE⟧']),
          '⟦Seminar 0 comes \\textbf{before} this lecture and practises the first tools on data||Seminarul 0 are loc \\textbf{înaintea} acestui curs și exersează primele instrumente pe date⟧'),
    '0.47', '0.50'), size='footnotesize')

D.frame('⟦The course at a glance||Cursul pe scurt⟧', cols(
    items(('⟦\\textbf{Programme}||\\textbf{Program}⟧',
           ["⟦Bachelor's programmes Economic Informatics and Economic Cybernetics, year 3, semester 2||Licență, programele Informatică economică și Cibernetică economică, anul III, semestrul 2⟧",
            '⟦Faculty of Cybernetics, Statistics and Economic Informatics (CSIE), ASE Bucharest||Facultatea de Cibernetică, Statistică și Informatică Economică (CSIE), ASE București⟧',
            '⟦4 ECTS credits; 2 hours of lecture and 1 hour of seminar a week; academic year 2026/2027||4 credite ECTS; 2 ore de curs și 1 oră de seminar pe săptămînă; anul universitar 2026/2027⟧']),
          ('⟦\\textbf{Lecture}||\\textbf{Curs}⟧',
           ['Prof.\\ Daniel Traian Pele, \\href{mailto:danpele@ase.ro}{danpele@ase.ro}']),
          ('⟦\\textbf{Prerequisites}||\\textbf{Cunoștințe necesare}⟧',
           ['⟦probability and statistics, econometrics (linear regression), Python||probabilități și statistică, econometrie (regresia liniară), Python⟧']),
          ('⟦\\textbf{Course website}||\\textbf{Site-ul cursului}⟧',
           ['\\href{https://danpele.github.io/Time-Series-Analysis/}{danpele.github.io/Time-Series-Analysis}: ⟦slides, seminars, notebooks, Quantlets, quizzes||slide-uri, seminarii, notebook-uri, Quantlets, quiz-uri⟧'])),
    ph('ase', '⟦The Bucharest University of Economic Studies (ASE)||Academia de Studii Economice din București (ASE)⟧', h='0.46\\textheight'),
    '0.56', '0.40'), size='footnotesize')

D.frame('⟦Evaluation||Evaluare⟧', items(
    ('⟦\\textbf{Written exam: 70\\%}||\\textbf{Examen scris: 70\\%}⟧',
     ['⟦reading and interpreting software output, short derivations, choosing a model for a given series||citirea și interpretarea rezultatelor obținute în Python, derivări scurte, alegerea modelului potrivit pentru o serie dată⟧',
      '⟦covers the lectures and seminars of Chapters 0--10 and 15||acoperă cursurile și seminariile Capitolelor 0--10 și 15⟧']),
    ('⟦\\textbf{Team project: 20\\%}||\\textbf{Proiect de echipă: 20\\%}⟧',
     ['⟦2--4 students analyse and forecast real time series with the methods of the course||2--4 studenți analizează și prognozează serii de timp reale cu metodele cursului⟧',
      '⟦a GitHub repository, a short report, a presentation and an oral defence||un repository GitHub, un raport scurt, o prezentare și o susținere orală⟧']),
    ('⟦\\textbf{Attendance: 10\\%}||\\textbf{Prezență: 10\\%}⟧',
     ['⟦lectures and seminars||cursuri și seminarii⟧']),
    ('⟦\\textbf{Self-assessment quizzes}||\\textbf{Quiz-uri de autoevaluare}⟧',
     ['⟦one per chapter on the website: 20 questions drawn from a bank of 24, each answer explained||cîte unul pentru fiecare capitol, pe site: 20 de întrebări extrase dintr-o bancă de 24, cu explicație pentru fiecare răspuns⟧'])))

D.frame('⟦Textbooks||Manuale⟧', items(
    ('⟦\\textbf{Main textbook}||\\textbf{Manualul de bază}⟧',
     ['\\refHP, \\emph{Applied Time Series Analysis and Forecasting with Python}, Springer',
      '⟦the chapters of the course follow its order: components and smoothing, stationarity, ARMA, ARIMA, seasonality, volatility, multivariate models||capitolele cursului urmează ordinea manualului: componente și netezire, staționaritate, ARMA, ARIMA, sezonalitate, volatilitate, modele multivariate⟧']),
    ('⟦\\textbf{Free companion}||\\textbf{Manual însoțitor, gratuit}⟧',
     ['\\refFPP, \\emph{Forecasting: Principles and Practice} (⟦3rd edition, online||ediția a 3-a, online⟧)',
      '⟦the reference for decomposition, exponential smoothing and forecast evaluation (Chapter 0)||referința pentru descompunere, netezire exponențială și evaluarea prognozei (Capitolul 0)⟧']),
    ('⟦\\textbf{Theory}||\\textbf{Teorie}⟧',
     ['\\refBD, \\emph{Introduction to Time Series and Forecasting}',
      '\\refHamilton, \\emph{Time Series Analysis}: ⟦the standard reference of time series econometrics||referința standard a econometriei seriilor de timp⟧'])))


def ch_rows(rng):
    return [f'{n} & ⟦{TITLES[n][0]}||{TITLES[n][1]}⟧' + ('$^*$' if n in SELF_STUDY else '') for n in rng]


D.frame('⟦Course map: chapters 0--15||Harta cursului: Capitolele 0--15⟧', cols(
    table('r>{\\raggedright\\arraybackslash}p{5.6cm}', '\\textbf{⟦No.||Nr.⟧} & \\textbf{⟦Chapter||Capitol⟧}', ch_rows(range(0, 8)), size='footnotesize'),
    table('r>{\\raggedright\\arraybackslash}p{5.6cm}', '\\textbf{⟦No.||Nr.⟧} & \\textbf{⟦Chapter||Capitol⟧}', ch_rows(range(8, 16)), size='footnotesize'),
    '0.49', '0.49') + '\n' + items(
    '⟦Univariate models (0--5), several series (6, 7), extensions (8--10), review (15)||Modele univariate (0--5), mai multe serii (6, 7), extensii (8--10), recapitulare (15)⟧',
    '⟦Self-study, marked $^*$: Chapters 11--14 (shorter materials and a quiz, no compulsory seminar)||Studiu individual, marcat cu $^*$: Capitolele 11--14 (materiale mai scurte și un quiz, fără seminar obligatoriu)⟧'),
    size='footnotesize')

D.frame('⟦Materials and tools||Materiale și instrumente⟧', items(
    ('⟦\\textbf{For every chapter}||\\textbf{Pentru fiecare capitol}⟧',
     ['⟦lecture and seminar slides, in Romanian and in English||slide-urile de curs și de seminar, în română și în engleză⟧',
      '⟦a lecture notebook and a seminar notebook (Python), opened in Google Colab with one click||un notebook de curs și unul de seminar (Python), care se deschid în Google Colab cu un singur clic⟧',
      '⟦a quiz of 20 questions||un quiz de 20 de întrebări⟧']),
    ('⟦\\textbf{Quantlets}: every chart has a public folder with its code, a description file (\\texttt{Metainfo.txt}) and the chart||\\textbf{Quantlets}: fiecare grafic are un folder public cu codul, un fișier de descriere (\\texttt{Metainfo.txt}) și graficul⟧',
     ['⟦the icon under each chart opens its Quantlet on GitHub||pictograma de sub fiecare grafic deschide Quantlet-ul corespunzător pe GitHub⟧']),
    ('⟦\\textbf{Data}: public sources, read directly in the code||\\textbf{Date}: surse publice, citite direct în cod⟧',
     ['⟦Eurostat, INS (National Institute of Statistics), BNR (National Bank of Romania), FRED (Federal Reserve Bank of St.\\ Louis)||Eurostat, INS, BNR, FRED (Federal Reserve Bank of St.\\ Louis)⟧',
      '⟦daily market data from EODHD (EOD Historical Data), saved in the course repository until 18 September 2026||date zilnice de piață de la EODHD (EOD Historical Data), salvate în repository-ul cursului pînă la 18 septembrie 2026⟧'])))

D.frame('⟦Seminars||Seminarii⟧', items(
    ('⟦\\textbf{Each seminar comes before its lecture}||\\textbf{Fiecare seminar are loc înaintea cursului său}⟧',
     ['⟦it starts with a short primer, so it can be followed without the lecture||începe cu noțiunile necesare, deci poate fi urmat fără curs⟧',
      '⟦the lecture then explains why the tools work and where they fail||cursul explică apoi de ce funcționează instrumentele și unde dau greș⟧']),
    ('⟦\\textbf{Three parts}||\\textbf{Trei părți}⟧',
     ['⟦A: short calculations on paper; B: real data, each task ending with an interpretation question; C: an open idea and the critique of an AI answer||A: calcule scurte pe hîrtie; B: date reale, fiecare cerință încheindu-se cu o întrebare de interpretare; C: o idee deschisă și critica unui răspuns generat de AI⟧']),
    ('⟦\\textbf{Two kinds of exercises}||\\textbf{Două tipuri de exerciții}⟧',
     ['⟦\\textbf{[Solved]}: full solution in the slides and in the notebook, a model to follow||\\textbf{[Rezolvat]}: rezolvarea completă în slide-uri și în notebook, un model de urmat⟧',
      '⟦\\textbf{[Proposed]}: you solve it; the solution is discussed in class||\\textbf{[Propus]}: îl rezolvați dumneavoastră; rezolvarea se discută la seminar⟧']),
    '⟦\\textbf{Nothing is handed in}: the seminar is practice for the exam and for the project||\\textbf{Seminarul nu se notează}: are rol de exercițiu pentru examen și pentru proiect⟧'))

D.frame('⟦The team project||Proiectul de echipă⟧', items(
    '⟦\\textbf{Question}: one forecasting or modelling question on real time series, chosen by the team||\\textbf{Întrebarea}: o întrebare de prognoză sau de modelare pe serii de timp reale, aleasă de echipă⟧',
    ('⟦\\textbf{Methods}: those of the course||\\textbf{Metode}: cele din curs⟧',
     ['⟦decomposition and smoothing, ARIMA and SARIMA, GARCH, VAR or VECM, Granger causality, impulse responses, forecast evaluation||descompunere și netezire, ARIMA și SARIMA, GARCH, VAR sau VECM, cauzalitate Granger, funcții de răspuns la impuls, evaluarea prognozei⟧',
      '⟦Romanian data (INS, BNR, Eurostat, BVB) are encouraged||sînt recomandate datele românești (INS, BNR, Eurostat, BVB)⟧']),
    ('⟦\\textbf{Deliverables}||\\textbf{Livrabile}⟧',
     ['⟦a GitHub repository whose code reproduces every number and every chart||un repository GitHub al cărui cod reproduce fiecare rezultat numeric și fiecare grafic⟧',
      '⟦a short report, a presentation and the file \\texttt{AI\\_USE.md}||un raport scurt, o prezentare și fișierul \\texttt{AI\\_USE.md}⟧']),
    ('⟦\\textbf{Grading}||\\textbf{Criterii de evaluare}⟧',
     ['⟦a clear question, correct methods and diagnostics, an honest out-of-sample evaluation||o întrebare clară, metode și diagnosticări corecte, o evaluare onestă în afara eșantionului⟧',
      '⟦\\textbf{oral defence}: each member explains the code and the results||\\textbf{susținere orală}: fiecare membru explică codul și rezultatele⟧'])))

D.frame('⟦AI policy||Politica privind AI⟧', items(
    ('⟦\\textbf{AI tools are allowed and must be declared}||\\textbf{Instrumentele AI sînt permise și trebuie declarate}⟧',
     ['⟦AI: artificial intelligence; here, assistants based on an LLM (large language model) that write text and code||AI (artificial intelligence, inteligență artificială): aici, asistenți construiți pe un LLM (model lingvistic de mari dimensiuni), care scriu text și cod⟧',
      '⟦every use goes into \\texttt{AI\\_USE.md}: the tool, the prompt, what was kept, what was corrected||fiecare utilizare se consemnează în \\texttt{AI\\_USE.md}: instrumentul, promptul, ce s-a păstrat, ce s-a corectat⟧']),
    ('⟦\\textbf{You are responsible for every number and every reference}||\\textbf{Răspundeți pentru fiecare rezultat numeric și pentru fiecare referință}⟧',
     ['⟦typical AI errors: invented references, wrong formulas, code that runs but computes something else||greșeli tipice ale AI: referințe inventate, formule greșite, cod care rulează, dar calculează altceva⟧',
      '⟦every reference: open its DOI (Digital Object Identifier) and check the title||fiecare referință: deschideți DOI-ul (Digital Object Identifier) și verificați titlul⟧',
      '⟦every number: recompute it with your own code||fiecare rezultat numeric: recalculați-l cu propriul cod⟧']),
    ('⟦\\textbf{Oral defence}||\\textbf{Susținerea orală}⟧',
     ['⟦a line of code you cannot explain does not count as your work||o linie de cod pe care nu o puteți explica nu este considerată muncă proprie⟧'])))

D.recap(('Organisation', 'organizarea cursului'), [
    '⟦Grade: 70\\% written exam, 20\\% team project, 10\\% attendance||Nota: 70\\% examen scris, 20\\% proiect de echipă, 10\\% prezență⟧',
    '⟦Textbook: \\refHP; free companion: \\refFPP||Manual: \\refHP; manual însoțitor gratuit: \\refFPP⟧',
    '⟦Chapters 0--15; Chapters 11--14 are self-study||Capitolele 0--15; Capitolele 11--14 sînt de studiu individual⟧',
    '⟦Seminars come before lectures; nothing is handed in||Seminariile preced cursurile și nu se notează⟧',
    '⟦AI is allowed, declared in \\texttt{AI\\_USE.md}, and checked at the oral defence||AI este permis, declarat în \\texttt{AI\\_USE.md} și verificat la susținerea orală⟧'])

# ===============================================================================================================
D.section('What a time series is', 'Seria de timp')
# ===============================================================================================================
D.frame('⟦Definition||Definiție⟧', items(
    '⟦\\textbf{Time series}: observations of one variable, recorded at successive moments and ordered in time||\\textbf{Serie de timp}: observații ale unei variabile, înregistrate în momente succesive și ordonate în timp⟧',
    ('⟦\\textbf{Notation}: $y_1, y_2, \\ldots, y_T$||\\textbf{Notație}: $y_1, y_2, \\ldots, y_T$⟧',
     ['⟦$t$: the time index (quarter, month, day); $T$: the number of observations||$t$: indicele de timp (trimestru, lună, zi); $T$: numărul de observații⟧',
      '⟦\\textbf{frequency}: the number of observations per year (4 quarterly, 12 monthly, about 252 for trading days)||\\textbf{frecvența}: numărul de observații pe an (4 trimestriale, 12 lunare, aproximativ 252 pentru zilele de tranzacționare)⟧']),
    ('⟦\\textbf{Example}: Romanian real GDP, quarterly, billion EUR at 2010 prices||\\textbf{Exemplu}: PIB-ul real al României, trimestrial, miliarde EUR în prețurile anului 2010⟧',
     ['@{ma_d0}: @{ma_y0}; @{ma_d1}: @{ma_y1}; @{ma_d2}: @{ma_y2}; @{ma_d3}: @{ma_y3}; @{ma_d4}: @{ma_y4}',
      '⟦the order matters: shuffling the five numbers destroys the information||ordinea contează: dacă amestecăm cele cinci valori, pierdem informația⟧']),
    '⟦GDP: gross domestic product; Q1--Q4: the four quarters of a year||PIB: produsul intern brut; T1--T4: cele patru trimestre ale unui an⟧'))

D.frame('⟦Three kinds of data||Trei tipuri de date⟧', table(
    '>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{4.2cm}>{\\raggedright\\arraybackslash}p{5.4cm}',
    '\\textbf{⟦Type||Tip⟧} & \\textbf{⟦Structure||Structură⟧} & \\textbf{⟦Example||Exemplu⟧}',
    ['⟦Cross-section||Date transversale⟧ & ⟦many units, one moment||multe unități, un singur moment⟧ & ⟦GDP of the 27 EU countries in 2025||PIB-ul celor 27 de țări UE în 2025⟧',
     '⟦Time series||Serie de timp⟧ & ⟦one unit, many moments||o unitate, multe momente⟧ & ⟦Romanian GDP, every quarter since 1995||PIB-ul României, în fiecare trimestru din 1995⟧',
     '⟦Panel||Date panel⟧ & ⟦many units, many moments||multe unități, multe momente⟧ & ⟦GDP of the 27 EU countries, every quarter||PIB-ul celor 27 de țări UE, în fiecare trimestru⟧'],
    size='footnotesize') + items(
    ('⟦\\textbf{What changes in a time series}||\\textbf{Particularitatea seriilor de timp}⟧',
     ['⟦observations are \\textbf{not independent}: this quarter resembles the previous one||observațiile \\textbf{nu sînt independente}: trimestrul acesta seamănă cu cel anterior⟧',
      '⟦this dependence is the information we use to forecast||această dependență este chiar informația pe care o folosim pentru prognoză⟧',
      '⟦the classical formulas for independent samples (standard errors, tests) must be adapted||formulele clasice pentru eșantioane independente (erori standard, teste) trebuie adaptate⟧']),
    '⟦EU: European Union||UE: Uniunea Europeană⟧'), size='footnotesize')

D.frame('⟦Why time series matter||Importanța seriilor de timp⟧', table(
    '>{\\raggedright\\arraybackslash}p{1.9cm}>{\\raggedright\\arraybackslash}p{4.3cm}>{\\raggedright\\arraybackslash}p{6.2cm}',
    '\\textbf{⟦Field||Domeniu⟧} & \\textbf{⟦Series||Serie⟧} & \\textbf{⟦Typical question||Întrebare tipică⟧}',
    ['⟦Economics||Economie⟧ & ⟦GDP, inflation, unemployment||PIB, inflație, șomaj⟧ & ⟦Where will inflation be in a year?||Unde va fi inflația peste un an?⟧',
     '⟦Finance||Finanțe⟧ & ⟦exchange rates, stock indices, interest rates||cursuri de schimb, indici bursieri, dobînzi⟧ & ⟦How risky is the BET tomorrow?||Cît de riscant este BET mîine?⟧',
     '⟦Energy||Energie⟧ & ⟦electricity production and consumption, prices||producția și consumul de electricitate, prețuri⟧ & ⟦How much electricity is needed next January?||De cîtă electricitate este nevoie în ianuarie viitor?⟧',
     '⟦Climate||Climă⟧ & ⟦CO2 concentration, temperatures||concentrația de CO2, temperaturi⟧ & ⟦How fast does CO2 rise, beyond the seasonal swing?||Cît de repede crește CO2, dincolo de oscilația sezonieră?⟧'],
    size='footnotesize') + items(
    '⟦Decisions depend on forecasts: the central bank sets the interest rate on its inflation forecast; the grid operator plans production on its demand forecast||Deciziile depind de prognoze: banca centrală stabilește dobînda pe baza prognozei inflației; operatorul de rețea planifică producția pe baza prognozei cererii⟧'),
    size='footnotesize')

D.frame('⟦Four goals of time series analysis||Patru obiective ale analizei seriilor de timp⟧', items(
    ('⟦\\textbf{Describe}: trend, seasonality, cycles, unusual observations||\\textbf{Descriere}: trend, sezonalitate, cicluri, observații neobișnuite⟧',
     ['⟦charts, decomposition, autocorrelation (this chapter and Chapter 1)||grafice, descompunere, autocorelație (acest capitol și Capitolul 1)⟧']),
    ('⟦\\textbf{Model}: a probability model that could have produced the data||\\textbf{Modelare}: un model probabilist care ar fi putut genera datele⟧',
     ['⟦ARMA, ARIMA, GARCH (Chapters 2--5)||ARMA, ARIMA, GARCH (Capitolele 2--5)⟧']),
    ('⟦\\textbf{Forecast}: values not yet observed, with their uncertainty||\\textbf{Prognoză}: valori încă neobservate, împreună cu incertitudinea lor⟧',
     ['⟦the goal that runs through the whole course||obiectivul care străbate întregul curs⟧']),
    ('⟦\\textbf{Explain and control}: how one series responds to another||\\textbf{Explicare și control}: cum răspunde o serie la modificarea alteia⟧',
     ['⟦VAR models, Granger causality, cointegration (Chapters 6 and 7)||modele VAR, cauzalitate Granger, cointegrare (Capitolele 6 și 7)⟧']),
    '⟦\\refBJ\\ framed the same goals: identify, estimate, check, forecast||\\refBJ\\ au formulat aceleași obiective: identificare, estimare, verificare, prognoză⟧'))

D.frame('⟦Frequency, flows and stocks, adjusted data||Frecvență, fluxuri și stocuri, date ajustate⟧', items(
    ('⟦\\textbf{Flow}: measured over a period; \\textbf{stock}: measured at a moment||\\textbf{Flux}: măsurat pe o perioadă; \\textbf{stoc}: măsurat la un moment⟧',
     ['⟦flows: GDP of a quarter, electricity produced in a month; they add up over time||fluxuri: PIB-ul unui trimestru, electricitatea produsă într-o lună; se adună în timp⟧',
      '⟦stocks: exchange rate on a day, CO2 concentration, an index level; we average them, we do not add them||stocuri: cursul de schimb dintr-o zi, concentrația de CO2, nivelul unui indice; le mediem, nu le adunăm⟧']),
    ('⟦\\textbf{Changing the frequency}||\\textbf{Schimbarea frecvenței}⟧',
     ['⟦monthly to quarterly: the sum for flows, the average or the last value for stocks||din lunar în trimestrial: suma pentru fluxuri, media sau ultima valoare pentru stocuri⟧']),
    ('⟦\\textbf{Seasonally adjusted data}||\\textbf{Date ajustate sezonier}⟧',
     ['⟦the statistical office removes the regular within-year pattern and the calendar effects (number of working days)||institutul de statistică elimină tiparul regulat din interiorul anului și efectele de calendar (numărul de zile lucrătoare)⟧',
      '⟦the unadjusted series is what was measured; the adjusted one is an estimate, revised when new data arrive||seria neajustată este cea măsurată; seria ajustată este o estimare, revizuită cînd apar date noi⟧'])))

DATA_ROWS = [
    '⟦Romanian real GDP||PIB-ul real al României⟧ & Eurostat & ⟦quarterly||trimestrial⟧ & @{gdp_first} -- @{gdp_last_q}',
    '⟦Romanian HICP (consumer prices)||IAPC, prețurile de consum⟧ & Eurostat & ⟦monthly||lunar⟧ & 1996 -- @{hicp_last_date_m}',
    '⟦Electricity generation, Romania||Producția de electricitate, România⟧ & Eurostat & ⟦monthly||lunar⟧ & @{elec_first_m} -- @{elec_last_m}',
    'EUR/RON & ⟦BNR reference rate||cursul de referință BNR⟧ & ⟦daily||zilnic⟧ & 2005 -- 2026',
    'BET, S\\&P 500 & EODHD & ⟦daily||zilnic⟧ & 2000 -- 2026',
    '⟦CO2 at Mauna Loa||CO2 la Mauna Loa⟧ & ⟦statsmodels data set||setul de date statsmodels⟧ & ⟦weekly, averaged monthly||săptămînal, mediat lunar⟧ & 1958 -- 2001',
]
D.frame('⟦Data used in this chapter||Datele folosite în acest capitol⟧', table(
    '>{\\raggedright\\arraybackslash}p{4.0cm}>{\\raggedright\\arraybackslash}p{3.0cm}>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{2.8cm}',
    '\\textbf{⟦Series||Serie⟧} & \\textbf{⟦Source||Sursă⟧} & \\textbf{⟦Frequency||Frecvență⟧} & \\textbf{⟦Sample||Eșantion⟧}',
    DATA_ROWS, size='footnotesize') + items(
    '⟦HICP: Harmonised Index of Consumer Prices, computed with the same method in every EU country||IAPC: indicele armonizat al prețurilor de consum (HICP, Harmonised Index of Consumer Prices), calculat cu aceeași metodă în toate țările UE⟧',
    '⟦BET: the reference index of the BVB (Bucharest Stock Exchange); S\\&P 500: 500 large US companies||BET: indicele de referință al BVB; S\\&P 500: 500 de companii mari din SUA⟧',
    '⟦statsmodels: the Python library for statistical models used throughout the course||statsmodels: biblioteca Python de modele statistice folosită pe tot parcursul cursului⟧'),
    size='footnotesize')

D.recap(('What a time series is', 'seria de timp'), [
    '⟦A time series is one variable observed over time; the order of the observations carries information||O serie de timp este o variabilă observată în timp; ordinea observațiilor conține informație⟧',
    '⟦Consecutive observations depend on each other: this is the basis of forecasting||Observațiile consecutive depind unele de altele: pe acest fapt se sprijină prognoza⟧',
    '⟦Goals: describe, model, forecast, explain||Obiective: descriere, modelare, prognoză, explicare⟧',
    '⟦Flows add up, stocks are averaged; adjusted series are estimates||Fluxurile se adună, stocurile se mediază; seriile ajustate sînt estimări⟧'])

# ===============================================================================================================
D.section('Time series on real data', 'Serii de timp pe date reale')
# ===============================================================================================================
chart(D, '⟦Romanian real GDP||PIB-ul real al României⟧', 'tsa_ch0_gdp', 'TSA_ch0_examples', [
    '⟦Quarterly, @{gdp_first} -- @{gdp_last_q}, chain-linked volumes at 2010 prices (billion EUR); blue: as measured; red: seasonally and calendar adjusted (Eurostat)||Trimestrial, @{gdp_first} -- @{gdp_last_q}, volume înlănțuite în prețurile anului 2010 (miliarde EUR); albastru: seria măsurată; roșu: seria ajustată sezonier și cu efectele de calendar (Eurostat)⟧'],
    h='0.6\\textheight')

D.frame('⟦Interpretation: GDP||Interpretarea: PIB⟧', items(
    ('⟦\\textbf{Trend}: the economy grew about @{gdp_mult} times from @{gdp_year_first} to @{gdp_year_last}||\\textbf{Trend}: economia a crescut de aproximativ @{gdp_mult} ori din @{gdp_year_first} pînă în @{gdp_year_last}⟧',
     ['⟦an average real growth of @{gdp_growth_ann}\\% a year||o creștere reală medie de @{gdp_growth_ann}\\% pe an⟧']),
    ('⟦\\textbf{Seasonality}: every year, Q1 is far below Q4||\\textbf{Sezonalitate}: în fiecare an, trimestrul I este mult sub trimestrul IV⟧',
     ['⟦winter: less construction and agriculture; autumn: the harvest and year-end activity||iarna: mai puține construcții și lucrări agricole; toamna: recolta și activitatea de la sfîrșit de an⟧',
      '⟦the swing grows with the level of GDP: a multiplicative pattern||oscilația crește odată cu nivelul PIB: un tipar multiplicativ⟧']),
    ('⟦\\textbf{Cycle}: recessions interrupt the trend||\\textbf{Ciclu}: recesiunile întrerup trendul⟧',
     ['⟦annual GDP fell by @{gdp_fall_2009_abs}\\% in 2009 (global financial crisis) and by @{gdp_fall_2020_abs}\\% in 2020 (COVID-19 pandemic)||PIB-ul anual a scăzut cu @{gdp_fall_2009_abs}\\% în 2009 (criza financiară globală) și cu @{gdp_fall_2020_abs}\\% în 2020 (pandemia de COVID-19)⟧']),
    '⟦Last value, @{gdp_last_q}: @{gdp_last_nsa} billion EUR unadjusted, @{gdp_last_sca} adjusted||Ultima valoare, @{gdp_last_q}: @{gdp_last_nsa} miliarde EUR neajustat, @{gdp_last_sca} ajustat⟧'))

chart(D, '⟦Romanian consumer prices and inflation||Prețurile de consum și inflația în România⟧', 'tsa_ch0_hicp', 'TSA_ch0_examples', [
    '⟦Left: HICP, 2015 = 100, log scale; right: annual inflation rate $100\\,(P_t/P_{t-12} - 1)$ since 2005, against the BNR target of 2.5\\%; $P_t$: the HICP in month $t$||Stînga: IAPC, 2015 = 100, scară logaritmică; dreapta: rata anuală a inflației $100\\,(P_t/P_{t-12} - 1)$ din 2005, față de ținta BNR de 2,5\\%; $P_t$: IAPC în luna $t$⟧'],
    h='0.58\\textheight')

D.frame('⟦Interpretation: inflation||Interpretarea: inflația⟧', items(
    ('⟦\\textbf{Level and rate tell different stories}||\\textbf{Nivelul și rata oferă informații diferite}⟧',
     ['⟦the price index almost never falls: its trend is the accumulated inflation||indicele prețurilor aproape nu scade niciodată: trendul lui este inflația acumulată⟧',
      '⟦the inflation rate rises and falls: it is the series the central bank targets||rata inflației crește și scade: este seria pe care o țintește banca centrală⟧']),
    ('⟦\\textbf{Episodes}||\\textbf{Episoade}⟧',
     ['⟦@{infl_max_date_m}: @{infl_max}\\%, after the price liberalisation of 1997||@{infl_max_date_m}: @{infl_max}\\%, după liberalizarea prețurilor din 1997⟧',
      '⟦@{infl_min10_date_m}: @{infl_min10}\\%, after the VAT cuts of 2015--2016||@{infl_min10_date_m}: @{infl_min10}\\%, după reducerile de TVA din 2015--2016⟧',
      '⟦@{infl_peak22_date_m}: @{infl_peak22}\\%, the energy price shock||@{infl_peak22_date_m}: @{infl_peak22}\\%, șocul prețurilor la energie⟧',
      '⟦@{hicp_last_date_m}: @{infl_last}\\%||@{hicp_last_date_m}: @{infl_last}\\%⟧']),
    '⟦A rate over 12 months removes the seasonal pattern but reacts with a delay: it reflects the whole past year||O rată calculată pe 12 luni elimină tiparul sezonier, dar reacționează cu întîrziere: reflectă întregul an trecut⟧',
    '⟦VAT: value added tax||TVA: taxa pe valoarea adăugată⟧'))

chart(D, '⟦EUR/RON, BNR reference rate||EUR/RON, cursul de referință BNR⟧', 'tsa_ch0_eurron', 'TSA_ch0_examples', [
    '⟦Lei per euro, every working day from @{eurron_first_date} to 18 September 2026 (@{eurron_n} observations)||Lei pentru un euro, în fiecare zi lucrătoare de la @{eurron_first_date} la 18 septembrie 2026 (@{eurron_n} de observații)⟧'],
    h='0.6\\textheight')

D.frame('⟦Interpretation: EUR/RON||Interpretarea: EUR/RON⟧', items(
    ('⟦\\textbf{A trend without a fixed shape}||\\textbf{Un trend fără o formă fixă}⟧',
     ['⟦from @{eurron_first} to @{eurron_last} lei; lowest value @{eurron_min} on @{eurron_min_date}||de la @{eurron_first} la @{eurron_last} lei; valoarea minimă @{eurron_min} pe @{eurron_min_date}⟧',
      '⟦long calm periods, interrupted by jumps (autumn 2008, 2012, 2025)||perioade lungi de calm, întrerupte de salturi (toamna lui 2008, 2012, 2025)⟧']),
    ('⟦\\textbf{No seasonality}||\\textbf{Fără sezonalitate}⟧',
     ['⟦if the euro were predictably cheaper every spring, traders would buy it in spring and the pattern would disappear||dacă euro ar fi, previzibil, mai ieftin în fiecare primăvară, participanții la piață l-ar cumpăra primăvara și tiparul ar dispărea⟧']),
    ('⟦\\textbf{Daily changes}||\\textbf{Variațiile zilnice}⟧',
     ['⟦standard deviation of the daily log change: @{eurron_sd}\\%; the largest move: @{eurron_maxmove}\\% on @{eurron_maxmove_date}||abaterea standard a variației logaritmice zilnice: @{eurron_sd}\\%; cea mai mare variație: @{eurron_maxmove}\\% pe @{eurron_maxmove_date}⟧']),
    '⟦A trend like this one, built from accumulated shocks, is called a \\textbf{stochastic trend} (Chapters 1 and 3)||Un astfel de trend, format din șocuri acumulate, se numește \\textbf{trend stochastic} (Capitolele 1 și 3)⟧'))

chart(D, '⟦BET and S\\&P 500 since 2000||BET și S\\&P 500 din 2000⟧', 'tsa_ch0_markets', 'TSA_ch0_examples', [
    '⟦Value of 100 invested on @{mk_first}, closing prices, log scale: equal vertical distances are equal percentage changes||Valoarea a 100 de unități investite pe @{mk_first}, prețuri de închidere, scară logaritmică: distanțele verticale egale sînt modificări procentuale egale⟧'],
    h='0.6\\textheight')

D.frame('⟦Interpretation: stock indices||Interpretarea: indicii bursieri⟧', items(
    ('⟦\\textbf{Long-run growth with deep falls}||\\textbf{Creștere pe termen lung, cu scăderi adînci}⟧',
     ['⟦100 invested became about @{bet_mult} times more in the BET and @{sp_mult} times more in the S\\&P 500 (price indices, without dividends)||100 de unități investite s-au înmulțit de aproximativ @{bet_mult} de ori în BET și de @{sp_mult} ori în S\\&P 500 (indici de preț, fără dividende)⟧',
      '⟦2008--2009: the BET lost more than three quarters of its value; it took years to recover||2008--2009: BET a pierdut mai mult de trei sferturi din valoare; revenirea a durat ani⟧']),
    ('⟦\\textbf{Why the log scale}||\\textbf{Rolul scării logaritmice}⟧',
     ['⟦on a linear scale, the early years look flat and recent moves look huge||pe o scară liniară, primii ani par plați, iar variațiile recente par uriașe⟧',
      '⟦the logarithm turns constant percentage growth into a straight line||logaritmul transformă o creștere procentuală constantă într-o dreaptă⟧']),
    '⟦Prices of traded assets are hard to forecast; their \\textbf{risk} is easier to forecast (Chapter 5)||Prețurile activelor tranzacționate se prognozează greu; \\textbf{riscul} lor se prognozează mai ușor (Capitolul 5)⟧'))

chart(D, '⟦Electricity generation in Romania||Producția de electricitate în România⟧', 'tsa_ch0_electricity', 'TSA_ch0_examples', [
    '⟦Net electricity generation, TWh per month, @{elec_first_m} -- @{elec_last_m} (Eurostat); dots: January of each year||Producția netă de electricitate, TWh pe lună, @{elec_first_m} -- @{elec_last_m} (Eurostat); puncte: luna ianuarie a fiecărui an⟧'],
    h='0.58\\textheight')

D.frame('⟦Interpretation: electricity||Interpretarea: electricitatea⟧', items(
    ('⟦\\textbf{Strong seasonality}||\\textbf{Sezonalitate puternică}⟧',
     ['⟦highest average month: @{elec_max_m} (@{elec_max_mean} TWh); lowest: @{elec_min_m} (@{elec_min_mean} TWh)||luna cu media cea mai mare: @{elec_max_m} (@{elec_max_mean} TWh); cea mai mică: @{elec_min_m} (@{elec_min_mean} TWh)⟧',
      '⟦winter demand for heating and lighting; the ratio of the two averages is @{elec_ratio}||cererea de iarnă pentru încălzire și iluminat; raportul celor două medii este @{elec_ratio}⟧']),
    ('⟦\\textbf{A falling level}||\\textbf{Un nivel în scădere}⟧',
     ['⟦@{elec_2008} TWh in 2008, @{elec_2025} TWh in 2025: more imports, less industrial demand||@{elec_2008} TWh în 2008, @{elec_2025} TWh în 2025: mai multe importuri, o cerere industrială mai mică⟧']),
    ('⟦\\textbf{Irregular years}||\\textbf{Ani atipici}⟧',
     ['⟦dry years reduce hydropower; mild winters reduce demand||anii secetoși reduc producția hidro; iernile blînde reduc cererea⟧']),
    '⟦TWh: terawatt-hour, $10^{12}$ watt-hours; GWh: gigawatt-hour, $10^{9}$ watt-hours||TWh: terawatt-oră, $10^{12}$ wați-oră; GWh: gigawatt-oră, $10^{9}$ wați-oră⟧'))

D.frame('⟦The Keeling curve||Curba Keeling⟧', cols(
    items(('⟦\\textbf{1958}: Charles David Keeling starts measuring CO2 at Mauna Loa, Hawaii, far from local sources (\\refKeeling)||\\textbf{1958}: Charles David Keeling începe măsurarea CO2 la Mauna Loa, Hawaii, departe de surse locale (\\refKeeling)⟧',
           ['⟦the longest continuous record of atmospheric CO2||cea mai lungă serie continuă de măsurători ale CO2 din atmosferă⟧']),
          ('⟦\\textbf{Two patterns in one series}||\\textbf{Două tipare într-o singură serie}⟧',
           ['⟦a rising trend: fossil fuel emissions accumulate||un trend crescător: emisiile din combustibili fosili se acumulează⟧',
            '⟦a yearly wave: plants of the Northern Hemisphere absorb CO2 in summer and release it in winter||o undă anuală: plantele din emisfera nordică absorb CO2 vara și îl eliberează iarna⟧']),
          '⟦ppm: parts per million, molecules of CO2 per million molecules of dry air||ppm (parts per million): molecule de CO2 la un milion de molecule de aer uscat⟧',
          '⟦CO2: carbon dioxide||CO2: dioxid de carbon⟧'),
    ph('keeling', '⟦Charles David Keeling (1928--2005), 2001||Charles David Keeling (1928--2005), 2001⟧', h='0.5\\textheight'),
    '0.6', '0.36'), size='footnotesize')

chart(D, '⟦Atmospheric CO2 at Mauna Loa||CO2 atmosferic la Mauna Loa⟧', 'tsa_ch0_co2', 'TSA_ch0_examples', [
    '⟦Monthly averages of the weekly measurements, @{co2_first_date_m} -- @{co2_last_date_m} (@{co2_n} months; the classic data set of statsmodels)||Mediile lunare ale măsurătorilor săptămînale, @{co2_first_date_m} -- @{co2_last_date_m} (@{co2_n} de luni; setul de date clasic din statsmodels)⟧'],
    h='0.58\\textheight')

D.frame('⟦Interpretation: CO2||Interpretarea: CO2⟧', items(
    ('⟦\\textbf{Trend}: from @{co2_first} to @{co2_last} ppm||\\textbf{Trend}: de la @{co2_first} la @{co2_last} ppm⟧',
     ['⟦on average @{co2_slope} ppm a year; the slope itself increases over time||în medie @{co2_slope} ppm pe an; panta însăși crește în timp⟧']),
    ('⟦\\textbf{Seasonality}: a regular yearly wave of a few ppm||\\textbf{Sezonalitate}: o undă anuală regulată, de cîțiva ppm⟧',
     ['⟦the wave keeps roughly its size while the level rises: an additive pattern||unda își păstrează aproximativ mărimea în timp ce nivelul crește: un tipar aditiv⟧']),
    ('⟦\\textbf{Almost no noise}||\\textbf{Aproape fără zgomot}⟧',
     ['⟦a careful physical measurement, unlike GDP or exchange rates||o măsurătoare fizică atentă, spre deosebire de PIB sau de cursul de schimb⟧']),
    '⟦A smooth series with a clear trend and season: the easiest case for decomposition and smoothing||O serie netedă, cu trend și sezonalitate clare: cazul cel mai simplu pentru descompunere și netezire⟧'))

D.frame('⟦Patterns in six series||Tipare în șase serii⟧', table(
    'lllll',
    '\\textbf{⟦Series||Serie⟧} & \\textbf{Trend} & \\textbf{⟦Season||Sezonalitate⟧} & \\textbf{⟦Cycle||Ciclu⟧} & \\textbf{⟦Noise||Zgomot⟧}',
    ['⟦GDP (unadjusted)||PIB (neajustat)⟧ & ⟦up||crescător⟧ & ⟦strong, multiplicative||puternică, multiplicativă⟧ & ⟦recessions||recesiuni⟧ & ⟦small||mic⟧',
     '⟦Inflation rate||Rata inflației⟧ & ⟦down after 1997||descrescător după 1997⟧ & ⟦removed by the 12-month rate||eliminată de rata pe 12 luni⟧ & ⟦shocks||șocuri⟧ & ⟦medium||mediu⟧',
     'EUR/RON & ⟦stochastic||stochastic⟧ & -- & -- & ⟦medium||mediu⟧',
     'BET & ⟦up, with crashes||crescător, cu prăbușiri⟧ & -- & ⟦booms and busts||avînturi și prăbușiri⟧ & ⟦large||mare⟧',
     '⟦Electricity||Electricitate⟧ & ⟦down||descrescător⟧ & ⟦strong, winter peak||puternică, vîrf iarna⟧ & -- & ⟦medium||mediu⟧',
     'CO2 & ⟦up, accelerating||crescător, accelerat⟧ & ⟦regular, additive||regulată, aditivă⟧ & -- & ⟦very small||foarte mic⟧'],
    size='scriptsize') + items(
    f'\\textbf{{{QUESTION}}}: ⟦which of these series would you forecast most accurately one year ahead?||care dintre aceste serii ați prognoza-o cel mai precis cu un an înainte?⟧',
    '\\pause',
    f'\\textbf{{{ANSWER}}}: ⟦CO2: a smooth trend and a stable season, little noise; the BET is the hardest: no season and large noise||CO2: un trend neted și o sezonalitate stabilă, puțin zgomot; BET este cea mai grea: fără sezonalitate și cu zgomot mare⟧'),
    size='footnotesize')

D.recap(('Time series on real data', 'serii de timp pe date reale'), [
    '⟦GDP and electricity: trend plus a strong seasonal pattern; GDP also shows recessions||PIB și electricitate: trend și un tipar sezonier puternic; PIB arată și recesiunile⟧',
    '⟦Inflation: the 12-month rate removes the season but reacts late||Inflația: rata pe 12 luni elimină sezonalitatea, dar reacționează tîrziu⟧',
    '⟦EUR/RON and the BET: trends built from shocks, no seasonality, hard to forecast||EUR/RON și BET: trenduri formate din șocuri, fără sezonalitate, greu de prognozat⟧',
    '⟦CO2: a smooth trend and an additive yearly wave||CO2: un trend neted și o undă anuală aditivă⟧',
    '⟦Look at the chart first: it decides the model||Priviți întîi graficul: el orientează alegerea modelului⟧'])

# ===============================================================================================================
D.section('A short history', 'O scurtă istorie')
# ===============================================================================================================
D.frame('⟦Sunspots: the first famous time series||Petele solare: prima serie de timp celebră⟧', cols(
    ph('wolf', 'Rudolf Wolf (1816--1893)', h='0.42\\textheight'),
    items('⟦\\textbf{1848}: the Swiss astronomer Rudolf Wolf defines a daily index of solar activity from the number of sunspots||\\textbf{1848}: astronomul elvețian Rudolf Wolf definește un indice zilnic al activității solare, pe baza numărului de pete solare⟧',
          '⟦he reconstructs it back to 1749: the Wolf (later Wolfer) sunspot numbers||îl reconstituie pînă în 1749: numerele de pete solare Wolf (ulterior Wolfer)⟧',
          '⟦the series rises and falls in waves of about 11 years, but neither the length nor the height of the waves is constant||seria crește și scade în valuri de aproximativ 11 ani, dar nici lungimea, nici înălțimea valurilor nu sînt constante⟧',
          '⟦the question: a hidden periodic signal plus noise, or something else?||întrebarea: un semnal periodic ascuns, peste care se suprapune zgomot, sau altceva?⟧')
    + '\n' + ph('sunspots', '⟦Sunspots drawn by Richard Carrington, 1 September 1859||Pete solare desenate de Richard Carrington, 1 septembrie 1859⟧', h='0.26\\textheight'),
    '0.30', '0.66'), size='footnotesize')

D.frame('⟦Yule (1926, 1927): dependence is the model||Yule (1926, 1927): dependența este modelul⟧', items(
    ('⟦\\textbf{1926}: \\emph{nonsense correlations} between time series (\\refYuleNonsense)||\\textbf{1926}: \\emph{corelațiile fără sens} dintre seriile de timp (\\refYuleNonsense)⟧',
     ['⟦two unrelated series with trends can show a correlation close to 1||două serii fără nicio legătură, dar cu trend, pot avea o corelație apropiată de 1⟧',
      '⟦his example: the share of Church of England marriages and the mortality rate in England and Wales||exemplul lui: ponderea căsătoriilor oficiate de Biserica Angliei și rata mortalității în Anglia și Țara Galilor⟧',
      '⟦today: spurious regression (Chapter 3) and cointegration (Chapter 7)||astăzi: regresia falsă (Capitolul 3) și cointegrarea (Capitolul 7)⟧']),
    ('⟦\\textbf{1927}: Wolfer\'s sunspot numbers explained by their own past (\\refYule)||\\textbf{1927}: numerele lui Wolfer explicate prin propriul trecut (\\refYule)⟧',
     ['⟦instead of hidden sine waves: $y_t = \\phi_1 y_{t-1} + \\phi_2 y_{t-2} + \\varepsilon_t$, with random disturbances $\\varepsilon_t$||în locul unor sinusoide ascunse: $y_t = \\phi_1 y_{t-1} + \\phi_2 y_{t-2} + \\varepsilon_t$, cu perturbații aleatoare $\\varepsilon_t$⟧',
      '⟦$y_t$: the sunspot number in year $t$; $\\phi_1, \\phi_2$: coefficients estimated from the data, which set the length and the damping of the waves||$y_t$: numărul de pete solare din anul $t$; $\\phi_1, \\phi_2$: coeficienți estimați din date, care fixează lungimea și amortizarea valurilor⟧',
      '⟦the first \\textbf{autoregressive} model: AR(2) (Chapter 2)||primul model \\textbf{autoregresiv}: AR(2) (Capitolul 2)⟧',
      '⟦disturbances shift the waves, so the period drifts, as in the data||perturbațiile deplasează valurile, deci perioada variază, ca în date⟧'])))

D.frame('⟦Slutsky (1937): cycles from random shocks||Slutsky (1937): cicluri din șocuri aleatoare⟧', cols(
    ph('slutsky', 'Eugen Slutsky (1880--1948)', h='0.42\\textheight'),
    items('⟦\\textbf{1927} (Moscow), \\textbf{1937} in English: \\emph{The summation of random causes as the source of cyclic processes} (\\refSlutsky)||\\textbf{1927} (Moscova), \\textbf{1937} în engleză: \\emph{Însumarea cauzelor aleatoare ca sursă a proceselor ciclice} (\\refSlutsky)⟧',
          ('⟦\\textbf{The experiment}||\\textbf{Experimentul}⟧',
           ['⟦take independent random numbers $\\varepsilon_1, \\varepsilon_2, \\ldots$ (the digits of a lottery draw)||se iau numere aleatoare independente $\\varepsilon_1, \\varepsilon_2, \\ldots$ (cifrele unei extrageri la loterie)⟧',
            '⟦replace each one by the sum of the last 10: $y_t = \\varepsilon_t + \\varepsilon_{t-1} + \\cdots + \\varepsilon_{t-9}$||fiecare număr se înlocuiește cu suma ultimelor 10: $y_t = \\varepsilon_t + \\varepsilon_{t-1} + \\cdots + \\varepsilon_{t-9}$⟧',
            '⟦the result looks like business cycles||rezultatul seamănă cu ciclurile economice⟧']),
          '⟦Today this is a \\textbf{moving-average} process: MA(9) (Chapter 2)||Astăzi, acesta este un proces de \\textbf{medie mobilă}: MA(9) (Capitolul 2)⟧'),
    '0.28', '0.68'), size='footnotesize')

chart(D, '⟦Slutsky\'s experiment, simulated||Experimentul lui Slutsky, simulat⟧', 'tsa_ch0_slutsky', 'TSA_ch0_slutsky', [
    '⟦Top: 240 independent shocks with the standard Normal distribution; bottom: the moving sum of 10 consecutive shocks||Sus: 240 de șocuri independente, cu distribuția Normală standard; jos: suma mobilă a cîte 10 șocuri consecutive⟧'],
    h='0.58\\textheight')

D.frame('⟦Interpretation: Slutsky\'s experiment||Interpretarea: experimentul lui Slutsky⟧', items(
    ('⟦\\textbf{No cycle was put in, yet waves appear}||\\textbf{Nu am introdus niciun ciclu, totuși apar valuri}⟧',
     ['⟦the moving sum crosses zero upwards @{slutsky_waves} times in 240 periods: waves of irregular length||suma mobilă traversează zero în sus de @{slutsky_waves} ori în 240 de perioade: valuri de lungime neregulată⟧',
      '⟦neighbouring values share 9 of their 10 shocks, so they are strongly correlated||valorile vecine au în comun 9 din cele 10 șocuri, deci sînt puternic corelate⟧']),
    ('⟦\\textbf{Two lessons}||\\textbf{Două lecții}⟧',
     ['⟦a cycle in a chart is not proof of a periodic cause||un ciclu într-un grafic nu dovedește existența unei cauze periodice⟧',
      '⟦smoothing (moving averages) can \\emph{create} apparent cycles: be careful with smoothed data||netezirea (mediile mobile) poate \\emph{crea} cicluri aparente: atenție la datele netezite⟧']),
    '⟦Yule (AR) and Slutsky (MA) gave the two building blocks of the ARMA models||Yule (AR) și Slutsky (MA) au oferit cele două componente de bază ale modelelor ARMA⟧'))

D.frame('⟦Kolmogorov and Wiener: the best linear forecast||Kolmogorov și Wiener: cea mai bună prognoză liniară⟧', cols(
    ph('kolmogorov', 'Andrei Kolmogorov (1903--1987)', h='0.36\\textheight')
    + '\\\\[1mm]\n\\raggedright\n' + items(
        '⟦\\textbf{1941}: the theory of interpolation and extrapolation of stationary sequences (\\refKolmogorov)||\\textbf{1941}: teoria interpolării și extrapolării șirurilor staționare (\\refKolmogorov)⟧'),
    ph('wiener', 'Norbert Wiener (1894--1964)', h='0.36\\textheight')
    + '\\\\[1mm]\n\\raggedright\n' + items(
        '⟦\\textbf{1942} (classified report), \\textbf{1949}: the Wiener filter, developed for anti-aircraft fire control (\\refWiener)||\\textbf{1942} (raport secret), \\textbf{1949}: filtrul Wiener, dezvoltat pentru conducerea tirului antiaerian (\\refWiener)⟧'),
    '0.48', '0.48') + '\n' + items(
    '⟦Both answer: which weighted sum of past values forecasts the future with the smallest mean squared error? The ancestor of the Kalman filter (Chapter 10)||Amîndoi răspund la întrebarea: ce sumă ponderată a valorilor trecute prognozează viitorul cu cea mai mică eroare pătratică medie? Strămoșul filtrului Kalman (Capitolul 10)⟧'),
    size='footnotesize')

D.frame('⟦Box and Jenkins (1970): a method for practice||Box și Jenkins (1970): o metodă pentru practică⟧', cols(
    items('⟦\\textbf{1970}: George Box and Gwilym Jenkins publish \\emph{Time Series Analysis: Forecasting and Control} (\\refBJ)||\\textbf{1970}: George Box și Gwilym Jenkins publică \\emph{Time Series Analysis: Forecasting and Control} (\\refBJ)⟧',
          ('⟦\\textbf{The Box--Jenkins cycle}||\\textbf{Ciclul Box--Jenkins}⟧',
           ['⟦\\textbf{identify}: choose a model from charts and the ACF||\\textbf{identificare}: alegerea modelului din grafice și din ACF⟧',
            '⟦\\textbf{estimate}: fit its parameters||\\textbf{estimare}: estimarea parametrilor⟧',
            '⟦\\textbf{check}: are the residuals unpredictable noise?||\\textbf{verificare}: sînt reziduurile un zgomot imprevizibil?⟧',
            '⟦\\textbf{forecast}: only when the checks pass||\\textbf{prognoză}: doar după ce verificările au trecut⟧']),
          '⟦ARIMA models and their seasonal version SARIMA come from this book (Chapters 2--4)||Modelele ARIMA și versiunea lor sezonieră, SARIMA, provin din această carte (Capitolele 2--4)⟧',
          '⟦Box: \\emph{all models are wrong, but some are useful}||Box: \\emph{toate modelele sînt greșite, dar unele sînt utile}⟧'),
    ph('box', 'George E.P.\\ Box (1919--2013)', h='0.52\\textheight'),
    '0.62', '0.34'), size='footnotesize')

TIMELINE = r"""\centering
\begin{tikzpicture}[x=0.118cm, y=0.62cm, font=\tiny]
\draw[thick, MainBlue, -{Latex}] (0,0) -- (101,0);
\foreach \x/\y in {0/1920, 20/1940, 40/1960, 60/1980, 80/2000, 100/2020} { \draw[MainBlue] (\x,0.1) -- (\x,-0.1); \node[below, text=MainBlue] at (\x,-0.1) {\y}; }
\foreach \x/\l/\h in {7/{Yule: AR}/1.5, 17/{Slutsky: MA}/0.75, 21/{Kolmogorov}/2.25, 29/{Wiener}/1.5, 37/{Holt}/0.75, 40/{Winters}/2.25, 50/{Box--Jenkins}/1.5, 60/{Sims: VAR}/0.75, 62/{Engle: ARCH}/2.25, 67/{Engle--Granger}/1.5, 80/{M3}/0.75, 98/{M4}/1.5} {
  \draw[IDAred] (\x,0) -- (\x,\h-0.12); \fill[IDAred] (\x,0) circle (0.6pt);
  \node[above, text=DarkGray, inner sep=1pt] at (\x,\h-0.12) {\l};
}
\end{tikzpicture}"""
D.frame('⟦A century in one line||Un secol pe o singură axă⟧', TIMELINE + '\n' + items(
    '⟦1927 Yule, 1937 Slutsky, 1941 Kolmogorov, 1949 Wiener: the foundations (Chapters 1 and 2)||1927 Yule, 1937 Slutsky, 1941 Kolmogorov, 1949 Wiener: fundamentele (Capitolele 1 și 2)⟧',
    '⟦1957 \\refHolt, 1960 \\refWinters: exponential smoothing, born in industry for sales and inventories (this chapter)||1957 \\refHolt, 1960 \\refWinters: netezirea exponențială, apărută în industrie, pentru vînzări și stocuri (acest capitol)⟧',
    '⟦1970 Box--Jenkins (Chapters 2--4); 1980 \\refSims\\ (Chapter 6); 1982 \\refEngle\\ (Chapter 5); 1987 \\refEG\\ (Chapter 7)||1970 Box--Jenkins (Capitolele 2--4); 1980 \\refSims\\ (Capitolul 6); 1982 \\refEngle\\ (Capitolul 5); 1987 \\refEG\\ (Capitolul 7)⟧',
    '⟦2000 and 2020: the M3 and M4 forecasting competitions compare methods on thousands of real series (\\refMthree; \\refMfour)||2000 și 2020: competițiile de prognoză M3 și M4 compară metodele pe mii de serii reale (\\refMthree; \\refMfour)⟧'),
    size='scriptsize')

D.recap(('a short history', 'o scurtă istorie'), [
    '⟦Sunspots (Wolf, 1848) posed the first question: hidden periods or random dependence?||Petele solare (Wolf, 1848) au ridicat prima întrebare: perioade ascunse sau dependență aleatoare?⟧',
    '⟦Yule: a series explained by its own past (AR); Slutsky: sums of shocks look like cycles (MA)||Yule: o serie explicată prin propriul trecut (AR); Slutsky: sumele de șocuri seamănă cu ciclurile (MA)⟧',
    '⟦Kolmogorov and Wiener: the optimal linear forecast||Kolmogorov și Wiener: prognoza liniară optimă⟧',
    '⟦Box and Jenkins: identify, estimate, check, forecast||Box și Jenkins: identificare, estimare, verificare, prognoză⟧',
    '⟦Forecasting competitions: simple methods are hard to beat||Competițiile de prognoză: metodele simple sînt greu de depășit⟧'])

# ===============================================================================================================
D.section('Components and decomposition', 'Componente și descompunere')
# ===============================================================================================================
D.frame('⟦Four components||Patru componente⟧', items(
    ('⟦\\textbf{Trend} $T_t$: the long-run movement of the level||\\textbf{Trendul} $T_t$: evoluția pe termen lung a nivelului⟧',
     ['⟦GDP growth, the rise of CO2; it can change direction slowly||creșterea PIB, creșterea CO2; își poate schimba lent direcția⟧']),
    ('⟦\\textbf{Seasonality} $S_t$: a pattern that repeats with a \\textbf{fixed, known period} $m$||\\textbf{Sezonalitatea} $S_t$: un tipar care se repetă cu o \\textbf{perioadă fixă și cunoscută} $m$⟧',
     ['⟦$m = 4$ for quarterly data, $m = 12$ for monthly data, $m = 7$ for daily data with a weekly pattern||$m = 4$ pentru date trimestriale, $m = 12$ pentru date lunare, $m = 7$ pentru date zilnice cu tipar săptămînal⟧']),
    ('⟦\\textbf{Cycle} $C_t$: rises and falls \\textbf{without a fixed period}, usually longer than a year||\\textbf{Ciclul} $C_t$: creșteri și scăderi \\textbf{fără perioadă fixă}, de obicei mai lungi de un an⟧',
     ['⟦business cycles: expansions and recessions of 2--10 years; usually merged with the trend into a \\textbf{trend-cycle}||ciclurile economice: expansiuni și recesiuni de 2--10 ani; de obicei sînt unite cu trendul într-o componentă \\textbf{trend-ciclu}⟧']),
    ('⟦\\textbf{Remainder} $R_t$ (irregular component, noise): what is left||\\textbf{Componenta neregulată} $R_t$ (reziduală, zgomot): ce rămîne⟧',
     ['⟦it should look unpredictable; if it does not, the decomposition missed something||ar trebui să pară imprevizibilă; dacă nu, descompunerea a omis ceva⟧'])))

D.frame('⟦Cycle or seasonality?||Ciclu sau sezonalitate?⟧', items(
    (f'\\textbf{{{QUESTION}}}: ⟦electricity generation is high every January||producția de electricitate este mare în fiecare ianuarie⟧',
     ['⟦is this a cycle or seasonality?||este un ciclu sau sezonalitate?⟧']),
    '\\pause',
    (f'\\textbf{{{ANSWER}}}: ⟦seasonality: the period is fixed (12 months) and known in advance||sezonalitate: perioada este fixă (12 luni) și cunoscută dinainte⟧',
     ['⟦a recession arrives at an unknown moment and lasts an unknown time: that is a cycle||o recesiune apare într-un moment necunoscut și durează un timp necunoscut: acesta este un ciclu⟧']),
    ('⟦\\textbf{Why the difference matters}||\\textbf{Importanța diferenței}⟧',
     ['⟦seasonality can be forecast years ahead: next January will again be high||sezonalitatea poate fi prognozată cu ani înainte: și luna ianuarie viitoare va avea valori mari⟧',
      '⟦cycles are hard to forecast: their timing changes (Slutsky)||ciclurile se prognozează greu: momentul lor se schimbă (Slutsky)⟧',
      '⟦sunspots: an 11-year \\emph{cycle}, not a season, because its length varies||petele solare: un \\emph{ciclu} de 11 ani, nu o sezonalitate, pentru că durata lui variază⟧'])))

D.frame('⟦Additive and multiplicative models||Modelul aditiv și modelul multiplicativ⟧', items(
    ('⟦\\textbf{Additive}: $y_t = T_t + S_t + R_t$||\\textbf{Aditiv}: $y_t = T_t + S_t + R_t$⟧',
     ['⟦the seasonal swing has a constant size in units of $y$ (CO2: a few ppm every year)||oscilația sezonieră are o mărime constantă, în unitățile lui $y$ (CO2: cîțiva ppm în fiecare an)⟧']),
    ('⟦\\textbf{Multiplicative}: $y_t = T_t \\times S_t \\times R_t$||\\textbf{Multiplicativ}: $y_t = T_t \\times S_t \\times R_t$⟧',
     ['⟦the seasonal swing is a constant \\textbf{percentage} of the level (GDP: Q1 about a quarter below the trend)||oscilația sezonieră este un \\textbf{procent} constant din nivel (PIB: trimestrul I cu aproximativ un sfert sub trend)⟧',
      '⟦$S_t$ and $R_t$ are factors around 1: $S_t = 0.8$ means 20\\% below the trend||$S_t$ și $R_t$ sînt factori în jurul lui 1: $S_t = 0.8$ înseamnă 20\\% sub trend⟧']),
    ('⟦\\textbf{The logarithm links them}: $\\ln y_t = \\ln T_t + \\ln S_t + \\ln R_t$||\\textbf{Logaritmul le leagă}: $\\ln y_t = \\ln T_t + \\ln S_t + \\ln R_t$⟧',
     ['⟦a multiplicative series becomes additive after taking logs||o serie multiplicativă devine aditivă după logaritmare⟧']),
    ('⟦\\textbf{How to choose}: look at the chart||\\textbf{Alegerea}: priviți graficul⟧',
     ['⟦does the seasonal swing grow with the level? Multiplicative (or logs); otherwise additive||oscilația sezonieră crește odată cu nivelul? Multiplicativ (sau logaritmi); altfel, aditiv⟧'])))

D.frame('⟦Centred moving averages||Medii mobile centrate⟧', items(
    ('⟦\\textbf{Moving average of order} $k$ (odd): the mean of $k$ neighbouring values||\\textbf{Media mobilă de ordin} $k$ (impar): media a $k$ valori vecine⟧',
     ['⟦$\\hat T_t = \\frac{1}{k} \\sum_{j=-q}^{q} y_{t+j}$, with $k = 2q + 1$; it smooths out the noise||$\\hat T_t = \\frac{1}{k} \\sum_{j=-q}^{q} y_{t+j}$, cu $k = 2q + 1$; netezește zgomotul⟧',
      '⟦$q$: the number of values on each side of $t$; $j$: the position relative to $t$; the hat marks an estimate||$q$: numărul de valori de fiecare parte a lui $t$; $j$: poziția față de $t$; notația $\hat{\ }$ indică o estimare⟧']),
    ('⟦\\textbf{The key property}: a moving average over one full season removes the seasonal pattern||\\textbf{Proprietatea esențială}: o medie mobilă pe un sezon complet elimină tiparul sezonier⟧',
     ['⟦every quarter enters once, so the seasonal effects cancel out||fiecare trimestru intră o singură dată, deci efectele sezoniere se compensează⟧']),
    ('⟦\\textbf{Even period} $m = 4$: a $2 \\times 4$ moving average, to stay centred on a quarter||\\textbf{Perioadă pară} $m = 4$: o medie mobilă $2 \\times 4$, pentru a rămîne centrată pe un trimestru⟧',
     ['$\\hat T_t = \\frac{1}{8} y_{t-2} + \\frac{1}{4} y_{t-1} + \\frac{1}{4} y_t + \\frac{1}{4} y_{t+1} + \\frac{1}{8} y_{t+2}$',
      '⟦five quarters, the two ends with half weight; for monthly data, $2 \\times 12$||cinci trimestre, cele două capete cu pondere înjumătățită; pentru date lunare, $2 \\times 12$⟧']),
    '⟦The first and the last $m/2$ values have no centred average: the trend estimate is missing at both ends||Primele și ultimele $m/2$ valori nu au medie centrată: estimarea trendului lipsește la ambele capete⟧'))

D.frame('⟦Worked example: a $2 \\times 4$ moving average||Exemplu rezolvat: o medie mobilă $2 \\times 4$⟧', cols(
    table('lr', '\\textbf{⟦Quarter||Trimestru⟧} & \\textbf{$y_t$}',
          ['@{ma_d0} & @{ma_y0}', '@{ma_d1} & @{ma_y1}', '@{ma_d2} & @{ma_y2}', '@{ma_d3} & @{ma_y3}', '@{ma_d4} & @{ma_y4}'],
          size='footnotesize'),
    items('⟦Romanian real GDP, billion EUR (Eurostat, unadjusted)||PIB-ul real al României, miliarde EUR (Eurostat, neajustat)⟧',
          ('⟦\\textbf{Trend in @{ma_d2}}||\\textbf{Trendul în @{ma_d2}}⟧',
           ['$\\hat T_t = \\frac{1}{4}\\left(\\frac{@{ma_y0}}{2} + @{ma_y1} + @{ma_y2} + @{ma_y3} + \\frac{@{ma_y4}}{2}\\right) = @{ma_2x4}$']),
          ('⟦\\textbf{Seasonal ratio}||\\textbf{Raportul sezonier}⟧',
           ['$y_t / \\hat T_t = @{ma_y2} / @{ma_2x4}$: ⟦the third quarter is above the trend||trimestrul III este peste trend⟧']),
          '⟦Averaging such ratios over all years gives the seasonal factor of each quarter||Media acestor rapoarte pe toți anii oferă factorul sezonier al fiecărui trimestru⟧'),
    '0.32', '0.64'), size='footnotesize')

D.frame('⟦Classical decomposition, step by step||Descompunerea clasică, pas cu pas⟧', clean(items(
    ('⟦\\textbf{Step 1}: trend-cycle $\\hat T_t$ by a centred $2 \\times m$ moving average||\\textbf{Pasul 1}: trend-ciclul $\\hat T_t$, printr-o medie mobilă centrată $2 \\times m$⟧', []),
    ('⟦\\textbf{Step 2}: detrend||\\textbf{Pasul 2}: eliminarea trendului⟧',
     ['⟦multiplicative: $y_t / \\hat T_t$; additive: $y_t - \\hat T_t$||multiplicativ: $y_t / \\hat T_t$; aditiv: $y_t - \\hat T_t$⟧']),
    ('⟦\\textbf{Step 3}: seasonal factors $\\hat S_1, \\ldots, \\hat S_m$||\\textbf{Pasul 3}: factorii sezonieri $\\hat S_1, \\ldots, \\hat S_m$⟧',
     ['⟦average the detrended values of each season over all years||media valorilor fără trend din fiecare sezon, pe toți anii⟧',
      '⟦normalise: they average to 1 (multiplicative) or to 0 (additive)||normalizare: media lor este 1 (multiplicativ) sau 0 (aditiv)⟧']),
    ('⟦\\textbf{Step 4}: remainder $\\hat R_t = y_t / (\\hat T_t \\hat S_t)$ or $y_t - \\hat T_t - \\hat S_t$||\\textbf{Pasul 4}: componenta neregulată $\\hat R_t = y_t / (\\hat T_t \\hat S_t)$ sau $y_t - \\hat T_t - \\hat S_t$⟧', []),
    ('⟦\\textbf{Seasonally adjusted series}: $y_t / \\hat S_t$ or $y_t - \\hat S_t$||\\textbf{Seria ajustată sezonier}: $y_t / \\hat S_t$ sau $y_t - \\hat S_t$⟧',
     ['⟦limits: the same season every year, no trend at the ends, sensitive to outliers||limite: același tipar sezonier în fiecare an, fără trend la capete, sensibilitate la valori extreme⟧']))))

chart(D, '⟦Decomposition of the Romanian GDP||Descompunerea PIB-ului României⟧', 'tsa_ch0_components', 'TSA_ch0_decomposition', [
    '⟦Classical multiplicative decomposition, $m = 4$: observed series and trend-cycle, trend-cycle, seasonal factors, remainder||Descompunere clasică multiplicativă, $m = 4$: seria observată și trend-ciclul, trend-ciclul, factorii sezonieri, componenta neregulată⟧'],
    h='0.72\\textheight')

D.frame('⟦Interpretation: seasonal factors of GDP||Interpretarea: factorii sezonieri ai PIB⟧', cols(
    table('lrr', '\\textbf{⟦Quarter||Trimestru⟧} & $\\hat S_q$ & ⟦relative to trend||față de trend⟧',
          ['⟦Q1||T1⟧ & @{gdp_S1} & $@{gdp_S1_pct}$\\%', '⟦Q2||T2⟧ & @{gdp_S2} & $@{gdp_S2_pct}$\\%',
           '⟦Q3||T3⟧ & @{gdp_S3} & $+$@{gdp_S3_pct}\\%', '⟦Q4||T4⟧ & @{gdp_S4} & $+$@{gdp_S4_pct}\\%'], size='footnotesize'),
    items(('⟦\\textbf{Reading the factors}||\\textbf{Citirea factorilor}⟧',
           ['⟦$\\hat S_q$: the estimated seasonal factor of quarter $q$||$\\hat S_q$: factorul sezonier estimat al trimestrului $q$⟧',
            '⟦a typical first quarter is @{gdp_S1_abs}\\% below the trend, a typical fourth quarter @{gdp_S4_pct}\\% above it||un trimestru I tipic se află cu @{gdp_S1_abs}\\% sub trend, un trimestru IV tipic cu @{gdp_S4_pct}\\% peste trend⟧',
            '⟦Q4 is about @{gdp_q1_rel}\\% larger than Q1 for purely seasonal reasons||trimestrul IV este cu aproximativ @{gdp_q1_rel}\\% mai mare decît trimestrul I, din motive pur sezoniere⟧']),
          ('⟦\\textbf{Remainder}||\\textbf{Componenta neregulată}⟧',
           ['⟦standard deviation about @{gdp_rem_sd}\\%; larger in the 1990s and in 2020||abaterea standard este de aproximativ @{gdp_rem_sd}\\%; mai mare în anii 1990 și în 2020⟧']),
          '⟦Consequence: compare a quarter with the same quarter of the previous year, never with the previous quarter of the raw series||Consecință: comparați un trimestru cu același trimestru din anul anterior, niciodată cu trimestrul precedent al seriei neajustate⟧'),
    '0.36', '0.60'), size='footnotesize')

D.frame('⟦STL: a flexible decomposition||STL: o descompunere flexibilă⟧', items(
    ('⟦\\textbf{STL}: Seasonal and Trend decomposition using LOESS (\\refSTL)||\\textbf{STL}: Seasonal and Trend decomposition using LOESS, descompunere în sezonalitate și trend cu LOESS (\\refSTL)⟧',
     ['⟦LOESS (locally estimated scatterplot smoothing): a weighted regression fitted around each point||LOESS (locally estimated scatterplot smoothing): o regresie ponderată, estimată în jurul fiecărui punct⟧']),
    ('⟦\\textbf{Advantages over the classical method}||\\textbf{Avantaje față de metoda clasică}⟧',
     ['⟦the seasonal pattern may change slowly from year to year||tiparul sezonier se poate schimba lent de la un an la altul⟧',
      '⟦a trend estimate up to the last observation||o estimare a trendului pînă la ultima observație⟧',
      '⟦a robust version: outliers go into the remainder and do not distort the season||o variantă robustă: valorile extreme ajung în componenta neregulată și nu deformează sezonalitatea⟧']),
    ('⟦\\textbf{Limits}||\\textbf{Limite}⟧',
     ['⟦additive only: use it on $\\ln y_t$ for multiplicative series||doar aditivă: pentru serii multiplicative se aplică lui $\\ln y_t$⟧',
      '⟦no calendar effects (working days, Easter)||nu tratează efectele de calendar (zile lucrătoare, Paște)⟧']),
    '⟦Statistical offices use X-13ARIMA-SEATS or TRAMO-SEATS, which build on ARIMA models (Chapter 4)||Institutele de statistică folosesc X-13ARIMA-SEATS sau TRAMO-SEATS, metode construite pe modele ARIMA (Capitolul 4)⟧'))

chart(D, '⟦STL decomposition of CO2||Descompunerea STL a seriei CO2⟧', 'tsa_ch0_stl', 'TSA_ch0_decomposition', [
    '⟦Robust STL with period 12 on the monthly CO2 series: observed, trend, seasonal, remainder (ppm)||STL robustă cu perioada 12, pe seria lunară CO2: seria observată, trendul, sezonalitatea, componenta neregulată (ppm)⟧'],
    h='0.72\\textheight')

D.frame('⟦Interpretation: STL of CO2||Interpretarea: STL pentru CO2⟧', clean(items(
    ('⟦\\textbf{Trend}: smooth, slightly convex: the rise accelerates||\\textbf{Trendul}: neted, ușor convex: creșterea se accelerează⟧', []),
    ('⟦\\textbf{Seasonal}: maximum in @{co2_max_m}, minimum in @{co2_min_m}||\\textbf{Sezonalitatea}: maximul în @{co2_max_m}, minimul în @{co2_min_m}⟧',
     ['⟦winter and spring: decay of fallen leaves releases CO2; summer: photosynthesis absorbs it||iarna și primăvara: descompunerea frunzelor căzute eliberează CO2; vara: fotosinteza îl absoarbe⟧',
      '⟦the yearly range grows from @{co2_amp_first} ppm (1959) to @{co2_amp_last} ppm (2001): STL lets the season change||amplitudinea anuală crește de la @{co2_amp_first} ppm (1959) la @{co2_amp_last} ppm (2001): STL permite modificarea sezonalității⟧']),
    ('⟦\\textbf{Remainder}: standard deviation @{co2_resid_sd} ppm||\\textbf{Componenta neregulată}: abaterea standard @{co2_resid_sd} ppm⟧',
     ['⟦small compared with the season: trend and season explain almost everything||mică în raport cu sezonalitatea: trendul și sezonalitatea explică aproape totul⟧',
      '⟦check the larger spikes against the history of the station before calling them noise||verificați vîrfurile mai mari în istoricul stației înainte de a le considera zgomot⟧']))))

D.recap(('components and decomposition', 'componente și descompunere'), [
    '⟦Trend, seasonality (fixed period), cycle (no fixed period), remainder||Trend, sezonalitate (perioadă fixă), ciclu (fără perioadă fixă), componenta neregulată⟧',
    '⟦Additive if the seasonal swing is constant, multiplicative if it grows with the level; logs turn one into the other||Aditiv dacă oscilația sezonieră este constantă, multiplicativ dacă ea crește odată cu nivelul; logaritmul transformă un model în celălalt⟧',
    '⟦A centred $2 \\times m$ moving average removes the season and estimates the trend-cycle||O medie mobilă centrată $2 \\times m$ elimină sezonalitatea și estimează trend-ciclul⟧',
    '⟦Romanian GDP: Q1 about @{gdp_S1_abs}\\% below the trend, Q4 about @{gdp_S4_pct}\\% above it||PIB-ul României: trimestrul I cu aproximativ @{gdp_S1_abs}\\% sub trend, trimestrul IV cu aproximativ @{gdp_S4_pct}\\% peste trend⟧',
    '⟦STL lets the seasonal pattern change and is robust to outliers||STL permite modificarea tiparului sezonier și este robustă la valori extreme⟧'])

# ===============================================================================================================
D.section('Notation, transformations and the ACF', 'Notație, transformări și ACF')
# ===============================================================================================================
D.frame('⟦Notation used in the course||Notația folosită în curs⟧', table(
    '>{\\raggedright\\arraybackslash}p{3.2cm}>{\\raggedright\\arraybackslash}p{8.8cm}',
    '\\textbf{⟦Symbol||Simbol⟧} & \\textbf{⟦Meaning||Semnificație⟧}',
    ['$y_t$, $t = 1, \\ldots, T$ & ⟦the observation at time $t$; $T$ observations in the sample||observația din momentul $t$; $T$ observații în eșantion⟧',
     '$\\bar y = \\frac1T \\sum_t y_t$ & ⟦the sample mean||media de selecție⟧',
     '$L y_t = y_{t-1}$ & ⟦the \\textbf{lag operator}: shifts the series back one period; $L^k y_t = y_{t-k}$||\\textbf{operatorul lag}: deplasează seria cu o perioadă înapoi; $L^k y_t = y_{t-k}$⟧',
     '$\\Delta y_t = y_t - y_{t-1}$ & ⟦the \\textbf{first difference}, $\\Delta = 1 - L$||\\textbf{diferența de ordinul întîi}, $\\Delta = 1 - L$⟧',
     '$\\Delta_m y_t = y_t - y_{t-m}$ & ⟦the \\textbf{seasonal difference}, $\\Delta_m = 1 - L^m$||\\textbf{diferența sezonieră}, $\\Delta_m = 1 - L^m$⟧',
     '$\\hat y_{T+h|T}$ & ⟦the forecast of $y_{T+h}$ made with the data up to $T$; $h$ is the \\textbf{horizon}||prognoza lui $y_{T+h}$ făcută cu datele pînă la $T$; $h$ este \\textbf{orizontul}⟧',
     '$e_t = y_t - \\hat y_t$ & ⟦the forecast error||eroarea de prognoză⟧'],
    size='footnotesize') + items(
    '⟦Hats ($\\hat{\\ }$) mark estimates and forecasts; Greek letters ($\\phi, \\theta, \\alpha$) mark parameters||Notația $\\hat{\\ }$ indică estimările și prognozele; literele grecești ($\\phi, \\theta, \\alpha$) marchează parametrii⟧'),
    size='footnotesize')

D.frame('⟦Transformations: logs, differences, growth rates||Transformări: logaritmi, diferențe, rate de creștere⟧', clean(items(
    ('⟦\\textbf{Logarithm} $\\ln y_t$: stabilises a variance that grows with the level||\\textbf{Logaritmul} $\\ln y_t$: stabilizează o varianță care crește odată cu nivelul⟧', []),
    ('⟦\\textbf{Growth rate over one period}: $g_t = 100\\,(y_t / y_{t-1} - 1)$||\\textbf{Rata de creștere pe o perioadă}: $g_t = 100\\,(y_t / y_{t-1} - 1)$⟧',
     ['⟦log approximation: $100\\,\\Delta \\ln y_t \\approx g_t$ for small changes||aproximarea logaritmică: $100\\,\\Delta \\ln y_t \\approx g_t$ pentru variații mici⟧']),
    ('⟦\\textbf{Annual growth rate} of a quarterly series: $100\\,(y_t / y_{t-4} - 1)$||\\textbf{Rata anuală de creștere} a unei serii trimestriale: $100\\,(y_t / y_{t-4} - 1)$⟧',
     ['⟦it compares the same quarters, so the season cancels out||compară aceleași trimestre, deci sezonalitatea se anulează⟧']),
    ('⟦\\textbf{Worked example}: Romanian GDP, @{ma_d0} = @{ma_y0}, @{ma_d3} = @{ma_y3}, @{ma_d4} = @{ma_y4}||\\textbf{Exemplu rezolvat}: PIB-ul României, @{ma_d0} = @{ma_y0}, @{ma_d3} = @{ma_y3}, @{ma_d4} = @{ma_y4}⟧',
     ['⟦over one quarter: $100\\,(@{ma_y4} / @{ma_y3} - 1) \\approx -34\\%$: not a collapse, only winter||pe un trimestru: $100\\,(@{ma_y4} / @{ma_y3} - 1) \\approx -34\\%$: nu o prăbușire, ci doar iarna⟧',
      '⟦over one year: $100\\,(@{ma_y4} / @{ma_y0} - 1) \\approx +0.5\\%$: the economy barely grew||pe un an: $100\\,(@{ma_y4} / @{ma_y0} - 1) \\approx +0.5\\%$: economia aproape nu a crescut⟧']),
    ('⟦\\textbf{Log return} of a price: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$, in \\%; $P_t$: the price on day $t$||\\textbf{Randamentul logaritmic} al unui preț: $r_t = 100\\,(\\ln P_t - \\ln P_{t-1})$, în \\%; $P_t$: prețul din ziua $t$⟧', []))),
    size='footnotesize')

chart(D, '⟦From prices to returns: the BET||De la prețuri la randamente: BET⟧', 'tsa_ch0_returns', 'TSA_ch0_acf', [
    '⟦Top: closing value of the BET since 2000; bottom: daily log returns in \\% (@{bet_n} trading days)||Sus: valoarea de închidere a indicelui BET din 2000; jos: randamentele logaritmice zilnice, în \\% (@{bet_n} de zile de tranzacționare)⟧'],
    h='0.6\\textheight')

D.frame('⟦Interpretation: prices and returns||Interpretarea: prețuri și randamente⟧', items(
    ('⟦\\textbf{The price has a trend; the returns do not}||\\textbf{Prețul are trend; randamentele nu au}⟧',
     ['⟦returns fluctuate around a mean of @{bet_mean}\\% a day, with a standard deviation of @{bet_sd}\\% (about @{bet_sd_ann}\\% a year)||randamentele fluctuează în jurul unei medii de @{bet_mean}\\% pe zi, cu abaterea standard de @{bet_sd}\\% (aproximativ @{bet_sd_ann}\\% pe an)⟧']),
    ('⟦\\textbf{Extreme days}||\\textbf{Zile extreme}⟧',
     ['⟦worst: @{bet_min}\\% on @{bet_min_date}; best: +@{bet_max}\\% on @{bet_max_date}||cea mai slabă: @{bet_min}\\% pe @{bet_min_date}; cea mai bună: +@{bet_max}\\% pe @{bet_max_date}⟧']),
    ('⟦\\textbf{Calm and turbulent periods alternate}||\\textbf{Perioadele calme alternează cu cele agitate}⟧',
     ['⟦large moves cluster in 2008--2010 and in March 2020: \\emph{volatility clustering} (Chapter 5)||variațiile mari apar grupat în 2008--2010 și în martie 2020: \\emph{volatility clustering} (Capitolul 5)⟧']),
    '⟦Differencing the log price removes the trend: most models of the course work on such transformed series||Diferențierea logaritmului prețului elimină trendul: majoritatea modelelor din curs se aplică unor astfel de serii transformate⟧'))

D.frame('⟦The sample autocorrelation function||Funcția de autocorelație de selecție⟧', clean(items(
    ('⟦\\textbf{Question}: how strongly is $y_t$ related to its own value $k$ periods earlier?||\\textbf{Întrebarea}: cît de puternic este legat $y_t$ de propria valoare de acum $k$ perioade?⟧', []),
    ('⟦\\textbf{Sample autocovariance} at lag $k$||\\textbf{Autocovarianța de selecție} la lagul $k$⟧',
     ['$c_k = \\frac{1}{T} \\sum_{t=k+1}^{T} (y_t - \\bar y)(y_{t-k} - \\bar y)$',
      '⟦$k$: the lag, in periods; $\\bar y$: the sample mean; the product is positive when $y_t$ and $y_{t-k}$ lie on the same side of the mean||$k$: lagul, în perioade; $\\bar y$: media de selecție; produsul este pozitiv cînd $y_t$ și $y_{t-k}$ se află de aceeași parte a mediei⟧']),
    ('⟦\\textbf{Sample autocorrelation}: $r_k = c_k / c_0$, between $-1$ and $1$||\\textbf{Autocorelația de selecție}: $r_k = c_k / c_0$, între $-1$ și $1$⟧',
     ['⟦$c_0$: the sample variance (the autocovariance at lag 0)||$c_0$: varianța de selecție (autocovarianța la lagul 0)⟧',
      '⟦the \\textbf{ACF} (autocorrelation function) is the sequence $r_1, r_2, \\ldots$ plotted against $k$ (the \\textbf{correlogram})||\\textbf{ACF} (autocorrelation function, funcția de autocorelație) este șirul $r_1, r_2, \\ldots$ reprezentat în funcție de $k$ (\\textbf{corelograma})⟧']),
    ('⟦\\textbf{Reference band} $\\pm 1.96 / \\sqrt{T}$||\\textbf{Banda de referință} $\\pm 1.96 / \\sqrt{T}$⟧',
     ['⟦if the series were independent noise, about 95\\% of the $r_k$ would fall inside it; 1.96: the 97.5\\% quantile of the standard Normal distribution||dacă seria ar fi un zgomot independent, aproximativ 95\\% dintre valorile $r_k$ ar cădea în interiorul ei; 1,96: cuantila de 97,5\\% a distribuției Normale standard⟧',
      '⟦the theory (stationarity, white noise, the Ljung--Box test) comes in Chapter 1||teoria (staționaritate, zgomot alb, testul Ljung--Box) urmează în Capitolul 1⟧']))))

D.frame('⟦Worked example: $r_1$ by hand||Exemplu rezolvat: $r_1$ calculat de mînă⟧', cols(
    table('rrrr', '$t$ & $y_t$ & $y_t - \\bar y$ & $(y_t - \\bar y)(y_{t-1} - \\bar y)$',
          ['1 & 2 & $-2$ & --', '2 & 4 & $0$ & $0$', '3 & 3 & $-1$ & $0$', '4 & 5 & $1$ & $-1$',
           '5 & 4 & $0$ & $0$', '6 & 6 & $2$ & $0$', '\\midrule ⟦sum||sumă⟧ & 24 & 0 & $-1$'], size='footnotesize'),
    items('$T = 6$, $\\bar y = 24 / 6 = 4$',
          '⟦sum of squared deviations: $4 + 0 + 1 + 1 + 0 + 4 = 10$||suma pătratelor abaterilor: $4 + 0 + 1 + 1 + 0 + 4 = 10$⟧',
          '$r_1 = \\dfrac{-1/6}{10/6} = -0.1$',
          '⟦at lag 2: products $(-1)(-2) + (1)(0) + (0)(-1) + (2)(1) = 4$, so $r_2 = 0.4$||la lagul 2: produsele $(-1)(-2) + (1)(0) + (0)(-1) + (2)(1) = 4$, deci $r_2 = 0.4$⟧',
          ('⟦\\textbf{Interpretation}||\\textbf{Interpretare}⟧',
           ['⟦band $\\pm 1.96/\\sqrt 6 = \\pm 0.80$: with 6 observations, neither value is distinguishable from 0||banda $\\pm 1.96/\\sqrt 6 = \\pm 0.80$: cu 6 observații, nicio valoare nu se deosebește de 0⟧'])),
    '0.48', '0.48'), size='footnotesize')

chart(D, '⟦Three correlograms||Trei corelograme⟧', 'tsa_ch0_acf', 'TSA_ch0_acf', [
    '⟦Sample ACF for lags 1--36; shaded: the band $\\pm 1.96/\\sqrt{T}$. Left: CO2 level; centre: monthly electricity generation; right: BET daily log returns||ACF de selecție pentru lagurile 1--36; zona colorată: banda $\\pm 1.96/\\sqrt{T}$. Stînga: nivelul CO2; centru: producția lunară de electricitate; dreapta: randamentele logaritmice zilnice ale BET⟧'],
    h='0.5\\textheight')

D.frame('⟦Interpretation: three correlograms||Interpretarea: trei corelograme⟧', items(
    ('⟦\\textbf{Trend}: CO2, $r_1 = $ @{acf_co2_1}, still @{acf_co2_36} at lag 36||\\textbf{Trend}: CO2, $r_1 = $ @{acf_co2_1}, încă @{acf_co2_36} la lagul 36⟧',
     ['⟦a slow, almost linear decay is the signature of a trend: the level remembers its past for years||o descreștere lentă, aproape liniară, este semnătura unui trend: nivelul rămîne corelat cu valorile de acum cîțiva ani⟧']),
    ('⟦\\textbf{Seasonality}: electricity, $r_6 = $ @{acf_el_6}, $r_{12} = $ @{acf_el_12}||\\textbf{Sezonalitate}: electricitatea, $r_6 = $ @{acf_el_6}, $r_{12} = $ @{acf_el_12}⟧',
     ['⟦peaks at 12, 24 and 36 months: this January resembles last January||vîrfuri la 12, 24 și 36 de luni: luna ianuarie de anul acesta seamănă cu ianuarie de anul trecut⟧']),
    ('⟦\\textbf{Almost no memory}: BET returns, $r_1 = $ @{acf_bet_1}, $r_2 = $ @{acf_bet_2}, band $\\pm$@{band_bet}||\\textbf{Aproape fără memorie}: randamentele BET, $r_1 = $ @{acf_bet_1}, $r_2 = $ @{acf_bet_2}, banda $\\pm$@{band_bet}⟧',
     ['⟦small but significant autocorrelations: hard to exploit after trading costs||autocorelații mici, dar semnificative: greu de exploatat după costurile de tranzacționare⟧',
      '⟦the \\emph{absolute} returns have $r_1 = $ @{acf_absbet_1}: the size of moves is predictable, their sign much less (Chapter 5)||randamentele în valoare \\emph{absolută} au $r_1 = $ @{acf_absbet_1}: mărimea variațiilor este previzibilă, semnul lor mult mai puțin (Capitolul 5)⟧']),
    '⟦The ACF of a level with a trend says little: transform first, then look at the ACF||ACF a unei serii cu trend este puțin informativă: întîi transformați seria, apoi examinați ACF⟧'))

D.recap(('notation, transformations and the ACF', 'notație, transformări și ACF'), [
    '⟦$L y_t = y_{t-1}$, $\\Delta = 1 - L$, $\\Delta_m = 1 - L^m$; $\\hat y_{T+h|T}$ is the forecast for horizon $h$||$L y_t = y_{t-1}$, $\\Delta = 1 - L$, $\\Delta_m = 1 - L^m$; $\\hat y_{T+h|T}$ este prognoza pentru orizontul $h$⟧',
    '⟦Logs stabilise the variance; differences remove the trend; annual rates remove the season||Logaritmii stabilizează varianța; diferențele elimină trendul; ratele anuale elimină sezonalitatea⟧',
    '⟦$r_k = c_k / c_0$, compared with the band $\\pm 1.96/\\sqrt T$||$r_k = c_k / c_0$, comparat cu banda $\\pm 1.96/\\sqrt T$⟧',
    '⟦Slow decay: trend; peaks at $m, 2m, \\ldots$: seasonality; all close to 0: little linear memory||Descreștere lentă: trend; vîrfuri la $m, 2m, \\ldots$: sezonalitate; toate aproape de 0: memorie liniară redusă⟧'])

# ===============================================================================================================
D.section('Exponential smoothing', 'Netezirea exponențială')
# ===============================================================================================================
D.frame('⟦Simple exponential smoothing (SES)||Netezirea exponențială simplă (SES)⟧', clean(items(
    ('⟦\\textbf{Idea}: forecast with a weighted average of the past, recent values weighing more||\\textbf{Ideea}: prognoza este o medie ponderată a trecutului, în care valorile recente au ponderi mai mari⟧', []),
    ('⟦\\textbf{Level equation}: $\\ell_t = \\alpha y_t + (1 - \\alpha)\\, \\ell_{t-1}$, with $0 < \\alpha \\le 1$||\\textbf{Ecuația nivelului}: $\\ell_t = \\alpha y_t + (1 - \\alpha)\\, \\ell_{t-1}$, cu $0 < \\alpha \\le 1$⟧',
     ['⟦$\\ell_t$: the smoothed level at time $t$; $\\alpha$: the smoothing constant; $\\ell_t$ moves from $\\ell_{t-1}$ towards $y_t$ by the fraction $\\alpha$||$\\ell_t$: nivelul netezit în momentul $t$; $\\alpha$: constanta de netezire; $\\ell_t$ se deplasează de la $\\ell_{t-1}$ spre $y_t$ cu fracțiunea $\\alpha$⟧',
      '⟦forecast for every horizon: $\\hat y_{T+h|T} = \\ell_T$, a flat line||prognoza pentru orice orizont: $\\hat y_{T+h|T} = \\ell_T$, o dreaptă orizontală⟧']),
    ('⟦\\textbf{Why ``exponential\'\'}: substituting repeatedly||\\textbf{Originea denumirii „exponențială”}: substituiri repetate⟧',
     ['$\\ell_T = \\alpha y_T + \\alpha(1-\\alpha) y_{T-1} + \\alpha(1-\\alpha)^2 y_{T-2} + \\cdots$',
      '⟦the weights fall geometrically; with $\\alpha = 0.5$: 0.5, 0.25, 0.125, \\ldots||ponderile scad geometric; cu $\\alpha = 0.5$: 0,5; 0,25; 0,125; \\ldots⟧']),
    ('⟦\\textbf{The smoothing constant} $\\alpha$||\\textbf{Constanta de netezire} $\\alpha$⟧',
     ['⟦small $\\alpha$: long memory, smooth forecasts, slow reaction; $\\alpha = 1$: the naive forecast $\\hat y_{T+1} = y_T$||$\\alpha$ mic: memorie lungă, prognoze netede, reacție lentă; $\\alpha = 1$: prognoza naivă $\\hat y_{T+1} = y_T$⟧',
      '⟦estimated by minimising the sum of squared one-step forecast errors||se estimează prin minimizarea sumei pătratelor erorilor de prognoză cu un pas înainte⟧']))))

D.frame('⟦Worked example: SES with $\\alpha = 0.5$||Exemplu rezolvat: SES cu $\\alpha = 0.5$⟧', cols(
    table('rrrr', '$t$ & $y_t$ & $\\hat y_{t|t-1} = \\ell_{t-1}$ & $\\ell_t = 0.5\\,y_t + 0.5\\,\\ell_{t-1}$',
          ['0 & -- & -- & $\\ell_0 = 10$', '1 & 10 & 10 & 10', '2 & 12 & 10 & 11', '3 & 11 & 11 & 11', '4 & 13 & 11 & 12',
           '5 & ? & \\textbf{12} & --'], size='footnotesize'),
    items('⟦initial level $\\ell_0 = 10$ (the first observation)||nivelul inițial $\\ell_0 = 10$ (prima observație)⟧',
          '⟦$t = 2$: $0.5 \\times 12 + 0.5 \\times 10 = 11$||$t = 2$: $0.5 \\times 12 + 0.5 \\times 10 = 11$⟧',
          '⟦$t = 4$: $0.5 \\times 13 + 0.5 \\times 11 = 12$||$t = 4$: $0.5 \\times 13 + 0.5 \\times 11 = 12$⟧',
          ('⟦\\textbf{Interpretation}||\\textbf{Interpretare}⟧',
           ['⟦the forecast for $t = 5$ and every later period is 12: SES does not extrapolate trends||prognoza pentru $t = 5$ și pentru toate perioadele următoare este 12: SES nu extrapolează trendul⟧',
            '⟦one-step errors: 0, 2, 0, 2: the forecasts lag behind the rising data||erorile cu un pas: 0, 2, 0, 2: prognozele rămîn în urma datelor crescătoare⟧'])),
    '0.52', '0.44'), size='footnotesize')

chart(D, '⟦SES on the monthly EUR/RON||SES pe cursul EUR/RON lunar⟧', 'tsa_ch0_ses', 'TSA_ch0_smoothing', [
    '⟦Dots: monthly averages of the BNR reference rate, @{ses_first_m} -- @{ses_last_m}; lines: one-step SES forecasts $\\hat y_{t|t-1}$ with $\\alpha = 0.1$ and $\\alpha = 0.7$||Puncte: mediile lunare ale cursului de referință BNR, @{ses_first_m} -- @{ses_last_m}; linii: prognozele SES cu un pas, $\\hat y_{t|t-1}$, cu $\\alpha = 0.1$ și $\\alpha = 0.7$⟧'],
    h='0.58\\textheight')

D.frame('⟦Interpretation: the choice of $\\alpha$||Interpretarea: alegerea lui $\\alpha$⟧', clean(items(
    ('⟦\\textbf{$\\alpha = 0.1$}: smooth, but always below the rising rate||\\textbf{$\\alpha = 0.1$}: netedă, dar mereu sub cursul crescător⟧',
     ['⟦it remembers several years: after the jump of 2025 it needs many months to catch up||ponderează observații din mai mulți ani: după saltul din 2025 are nevoie de multe luni pentru a ajunge la noul nivel⟧']),
    ('⟦\\textbf{$\\alpha = 0.7$}: follows the data closely, one month late||\\textbf{$\\alpha = 0.7$}: urmărește îndeaproape datele, cu o lună întîrziere⟧', []),
    ('⟦\\textbf{Estimated} $\\hat\\alpha = $ @{ses_alpha}||\\textbf{Valoarea estimată} $\\hat\\alpha = $ @{ses_alpha}⟧',
     ['⟦the best SES forecast of next month is this month\'s average: SES reduces to the naive forecast||cea mai bună prognoză SES pentru luna viitoare este media lunii curente: SES se reduce la prognoza naivă⟧',
      '⟦typical of exchange rates and prices: the last value contains almost all the information (a random walk, Chapter 1)||tipic pentru cursurile de schimb și prețuri: ultima valoare conține aproape toată informația (un mers aleator, Capitolul 1)⟧']),
    '⟦An estimated $\\alpha$ near 1 is a finding about the series, not a failure of the method||O valoare estimată a lui $\\alpha$ apropiată de 1 este o informație despre serie, nu un eșec al metodei⟧')))

D.frame('⟦Holt and Holt--Winters (1/2)||Metodele Holt și Holt--Winters (1/2)⟧', items(
    ('⟦\\textbf{Holt (1957)}: level plus a \\textbf{trend} $b_t$ (\\refHolt)||\\textbf{Holt (1957)}: nivel și \\textbf{trend} $b_t$ (\\refHolt)⟧',
     ['$\\ell_t = \\alpha y_t + (1-\\alpha)(\\ell_{t-1} + b_{t-1})$, \\ $b_t = \\beta (\\ell_t - \\ell_{t-1}) + (1-\\beta) b_{t-1}$',
      '⟦the level is updated towards $y_t$ from the previous level plus the previous slope||nivelul se actualizează spre $y_t$, pornind de la nivelul anterior plus panta anterioară⟧',
      '⟦$b_t$: the estimated slope, the change of the level per period; it is updated towards the latest change $\\ell_t - \\ell_{t-1}$||$b_t$: panta estimată, adică variația nivelului pe perioadă; se actualizează spre ultima variație $\\ell_t - \\ell_{t-1}$⟧',
      '⟦$\\beta \\in (0, 1]$: the smoothing constant of the slope; a small $\\beta$ gives a slope that changes slowly||$\\beta \\in (0, 1]$: constanta de netezire a pantei; un $\\beta$ mic dă o pantă care se schimbă lent⟧']),
    ('⟦\\textbf{Forecast}: $\\hat y_{T+h|T} = \\ell_T + h\\, b_T$, a straight line||\\textbf{Prognoza}: $\\hat y_{T+h|T} = \\ell_T + h\\, b_T$, o dreaptă⟧',
     ['⟦the last level plus $h$ times the last slope||ultimul nivel plus de $h$ ori ultima pantă⟧',
      '⟦a \\textbf{damped} trend multiplies the slope by $\\phi^j$, $0 < \\phi < 1$, at step $j$: the line flattens for long horizons||un trend \\textbf{amortizat} înmulțește panta cu $\\phi^j$, $0 < \\phi < 1$, la pasul $j$: dreapta se aplatizează pentru orizonturi lungi⟧'])), size='footnotesize')

D.frame('⟦Holt and Holt--Winters (2/2)||Metodele Holt și Holt--Winters (2/2)⟧', items(
    ('⟦\\textbf{Holt--Winters (1960)}: adds seasonal factors $s_t$ with period $m$ (\\refWinters)||\\textbf{Holt--Winters (1960)}: adaugă factori sezonieri $s_t$ cu perioada $m$ (\\refWinters)⟧',
     ['⟦multiplicative form: $\\hat y_{T+h|T} = (\\ell_T + h\\, b_T)\\, s_{T+h-m}$||forma multiplicativă: $\\hat y_{T+h|T} = (\\ell_T + h\\, b_T)\\, s_{T+h-m}$⟧',
      '⟦$s_{T+h-m}$: the latest seasonal factor of the same season (for $h \\le m$); $s = 1.1$ means 10\\% above the trend line||$s_{T+h-m}$: ultimul factor sezonier al aceluiași sezon (pentru $h \\le m$); $s = 1.1$ înseamnă 10\\% peste linia trendului⟧',
      '⟦each component has its own smoothing constant: $\\alpha$ (level), $\\beta$ (slope), $\\gamma$ (season)||fiecare componentă are propria constantă de netezire: $\\alpha$ (nivel), $\\beta$ (pantă), $\\gamma$ (sezonalitate)⟧']),
    ('⟦\\textbf{ETS}: Error, Trend, Seasonal (\\refHKSG)||\\textbf{ETS}: Error, Trend, Seasonal, adică eroare, trend, sezonalitate (\\refHKSG)⟧',
     ['⟦each method is a statistical model: N (none), A (additive), M (multiplicative), A$_d$ (damped)||fiecare metodă este un model statistic: N (none, absent), A (aditiv), M (multiplicativ), A$_d$ (amortizat)⟧',
      '⟦e.g.\\ ETS(M,A$_d$,M): multiplicative errors, damped trend, multiplicative season; it gives forecast intervals and an information criterion (Chapter 2)||de exemplu, ETS(M,A$_d$,M): erori multiplicative, trend amortizat, sezonalitate multiplicativă; oferă intervale de prognoză și un criteriu informațional (Capitolul 2)⟧']),
    '⟦A review of 25 years of exponential smoothing: \\refGardner||O sinteză a 25 de ani de netezire exponențială: \\refGardner⟧'), size='footnotesize')

D.recap(('exponential smoothing', 'netezirea exponențială'), [
    '⟦SES: $\\ell_t = \\alpha y_t + (1-\\alpha)\\ell_{t-1}$; weights decline geometrically; flat forecasts||SES: $\\ell_t = \\alpha y_t + (1-\\alpha)\\ell_{t-1}$; ponderile scad geometric; prognoze orizontale⟧',
    '⟦Small $\\alpha$: smooth and slow; $\\alpha = 1$: the naive forecast||$\\alpha$ mic: netezire puternică și reacție lentă; $\\alpha = 1$: prognoza naivă⟧',
    '⟦EUR/RON: $\\hat\\alpha = $ @{ses_alpha}, the last value is the best SES forecast||EUR/RON: $\\hat\\alpha = $ @{ses_alpha}, ultima valoare este cea mai bună prognoză SES⟧',
    '⟦Holt adds a trend, Holt--Winters a season; ETS turns them into statistical models||Holt adaugă un trend, Holt--Winters o sezonalitate; ETS le transformă în modele statistice⟧'])

# ===============================================================================================================
D.section('Forecasts and their evaluation', 'Prognoza și evaluarea ei')
# ===============================================================================================================
D.frame('⟦Forecasting: the setting||Prognoza: cadrul general⟧', items(
    ('⟦\\textbf{Forecast} $\\hat y_{T+h|T}$: a statement about $y_{T+h}$ made at time $T$||\\textbf{Prognoza} $\\hat y_{T+h|T}$: o afirmație despre $y_{T+h}$, făcută în momentul $T$⟧',
     ['⟦it may use only the information available at $T$: the \\textbf{information set}||poate folosi doar informația disponibilă în momentul $T$: \\textbf{mulțimea de informație}⟧',
      '⟦$h = 1, 2, \\ldots$: the \\textbf{horizon}; uncertainty grows with $h$||$h = 1, 2, \\ldots$: \\textbf{orizontul}; incertitudinea crește odată cu $h$⟧']),
    ('⟦\\textbf{Point and interval forecasts}||\\textbf{Prognoze punctuale și prognoze pe interval}⟧',
     ['⟦point: one number; interval: a range that contains $y_{T+h}$ with a stated probability (e.g.\\ 95\\%)||punctuală: un singur număr; pe interval: un interval care conține $y_{T+h}$ cu o probabilitate stabilită (de exemplu, 95\\%)⟧']),
    ('⟦\\textbf{What makes a series forecastable}||\\textbf{Condițiile previzibilității}⟧',
     ['⟦a stable pattern (CO2) rather than one driven by news (exchange rates)||un tipar stabil (CO2), nu unul determinat de știri (cursurile de schimb)⟧',
      '⟦a forecast that does not change the outcome: a price forecast published to millions of traders changes the price||o prognoză care nu modifică rezultatul: o prognoză a prețului publicată pentru milioane de participanți la piață schimbă prețul⟧'])))

D.frame('⟦Four benchmark methods||Patru metode de referință⟧', items(
    ('⟦\\textbf{Mean}: $\\hat y_{T+h|T} = \\bar y$||\\textbf{Media}: $\\hat y_{T+h|T} = \\bar y$⟧',
     ['⟦the future looks like the average past||viitorul seamănă cu trecutul mediu⟧']),
    ('⟦\\textbf{Naive}: $\\hat y_{T+h|T} = y_T$||\\textbf{Naivă}: $\\hat y_{T+h|T} = y_T$⟧',
     ['⟦the last value; optimal for a random walk (exchange rates, prices)||ultima valoare; optimă pentru un mers aleator (cursuri de schimb, prețuri)⟧']),
    ('⟦\\textbf{Seasonal naive}: $\\hat y_{T+h|T} = y_{T+h-m}$ (for $h \\le m$)||\\textbf{Naivă sezonieră}: $\\hat y_{T+h|T} = y_{T+h-m}$ (pentru $h \\le m$)⟧',
     ['⟦next January = last January; strong for highly seasonal series||ianuarie viitor = ianuarie trecut; puternică pentru seriile cu sezonalitate pronunțată⟧']),
    ('⟦\\textbf{Drift}: $\\hat y_{T+h|T} = y_T + h\\, \\dfrac{y_T - y_1}{T - 1}$||\\textbf{Cu derivă}: $\\hat y_{T+h|T} = y_T + h\\, \\dfrac{y_T - y_1}{T - 1}$⟧',
     ['⟦the last value plus the average past change per period||ultima valoare plus variația medie pe perioadă din trecut⟧']),
    '⟦A new model is worth using only if it beats these benchmarks on data it has not seen||Un model nou merită folosit doar dacă depășește aceste metode de referință pe date pe care nu le-a văzut⟧'))

SPLIT = r"""\centering
\begin{tikzpicture}[x=0.5cm, y=0.42cm, font=\scriptsize]
\foreach \r/\n in {0/1, 1/2, 2/3} {
  \pgfmathsetmacro{\e}{14 + 2*\r}
  \fill[MainBlue!70] (0, -\r*1.4) rectangle (\e, -\r*1.4 + 0.9);
  \fill[IDAred!75] (\e, -\r*1.4) rectangle (\e + 2, -\r*1.4 + 0.9);
  \node[left] at (0, -\r*1.4 + 0.45) {⟦origin||originea⟧ \n};
}
\node[text=MainBlue] at (6, 1.6) {⟦training set||set de antrenare⟧};
\node[text=IDAred] at (17, 1.6) {⟦test set||set de test⟧};
\draw[-{Latex}, thick] (0, -3.9) -- (21, -3.9) node[right] {$t$};
\end{tikzpicture}"""
D.frame('⟦Training set and test set||Setul de antrenare și setul de test⟧', SPLIT + '\n' + items(
    ('⟦\\textbf{Split by time, never at random}||\\textbf{Împărțirea se face în timp, niciodată aleator}⟧',
     ['⟦training set: the first observations, used to choose and fit the method||setul de antrenare: primele observații, folosite pentru alegerea și estimarea metodei⟧',
      '⟦test set: the last $h$ observations, kept hidden and used only to measure the errors||setul de test: ultimele $h$ observații, ascunse și folosite doar pentru măsurarea erorilor⟧',
      '⟦a random split lets the model see the future: the errors look too small||o împărțire aleatoare îi permite modelului să vadă viitorul: erorile par prea mici⟧']),
    ('⟦\\textbf{Rolling origin} (time series cross-validation): repeat with the origin moved forward||\\textbf{Origine mobilă} (validare încrucișată pentru serii de timp): se repetă, mutînd originea înainte⟧',
     ['⟦average the errors over origins: a more stable estimate of accuracy||mediați erorile pe toate originile: o estimare mai stabilă a acurateței⟧'])),
    size='footnotesize')

D.frame('⟦Measures of forecast error||Măsuri ale erorii de prognoză⟧', items(
    '⟦Errors on the test set: $e_{T+j} = y_{T+j} - \\hat y_{T+j|T}$, $j = 1, \\ldots, h$||Erorile pe setul de test: $e_{T+j} = y_{T+j} - \\hat y_{T+j|T}$, $j = 1, \\ldots, h$⟧',
    ('⟦\\textbf{MAE} (mean absolute error) $= \\frac1h \\sum_j |e_{T+j}|$||\\textbf{MAE} (mean absolute error, eroarea absolută medie) $= \\frac1h \\sum_j |e_{T+j}|$⟧',
     ['⟦in the units of the series; easy to explain||în unitățile seriei; ușor de explicat⟧']),
    ('⟦\\textbf{RMSE} (root mean squared error) $= \\sqrt{\\frac1h \\sum_j e_{T+j}^2}$||\\textbf{RMSE} (root mean squared error, rădăcina erorii pătratice medii) $= \\sqrt{\\frac1h \\sum_j e_{T+j}^2}$⟧',
     ['⟦penalises large errors more; always $\\ge$ MAE||penalizează mai mult erorile mari; întotdeauna $\\ge$ MAE⟧']),
    ('⟦\\textbf{MAPE} (mean absolute percentage error) $= \\frac{100}{h} \\sum_j |e_{T+j} / y_{T+j}|$||\\textbf{MAPE} (mean absolute percentage error, eroarea procentuală absolută medie) $= \\frac{100}{h} \\sum_j |e_{T+j} / y_{T+j}|$⟧',
     ['⟦without units, but useless when $y$ is close to 0 (inflation, returns)||fără unități de măsură, dar inutilizabilă cînd $y$ este aproape de 0 (inflație, randamente)⟧']),
    ('⟦\\textbf{MASE} (mean absolute scaled error) $=$ MAE $/$ in-sample MAE of the seasonal naive method (\\refHK)||\\textbf{MASE} (mean absolute scaled error, eroarea absolută medie scalată) $=$ MAE $/$ MAE în eșantion a metodei naive sezoniere (\\refHK)⟧',
     ['⟦below 1: better than the seasonal naive method on the training data; comparable across series||sub 1: mai bună decît metoda naivă sezonieră pe datele de antrenare; comparabilă între serii⟧'])), size='footnotesize')

D.frame('⟦Worked example: errors by hand||Exemplu rezolvat: erorile calculate de mînă⟧', cols(
    table('lrrrr', '& ⟦Q1||T1⟧ & ⟦Q2||T2⟧ & ⟦Q3||T3⟧ & ⟦Q4||T4⟧',
          ['⟦year 1||anul 1⟧ & 10 & 14 & 18 & 12', '⟦year 2||anul 2⟧ & 11 & 15 & 20 & 13', '\\midrule ⟦year 3 (test)||anul 3 (test)⟧ & 12 & 16 & 21 & 14',
           '⟦naive||naivă⟧ & 13 & 13 & 13 & 13', '⟦seasonal naive||naivă sezonieră⟧ & 11 & 15 & 20 & 13'], size='footnotesize'),
    items(('⟦\\textbf{Naive}: errors $-1, 3, 8, 1$||\\textbf{Naivă}: erorile $-1, 3, 8, 1$⟧',
           ['MAE $= (1 + 3 + 8 + 1)/4 = 3.25$', 'RMSE $= \\sqrt{(1 + 9 + 64 + 1)/4} = \\sqrt{18.75} = 4.33$']),
          ('⟦\\textbf{Seasonal naive}: errors $1, 1, 1, 1$||\\textbf{Naivă sezonieră}: erorile $1, 1, 1, 1$⟧',
           ['MAE $=$ RMSE $= 1$']),
          ('⟦\\textbf{Interpretation}||\\textbf{Interpretare}⟧',
           ['⟦the naive forecast misses the season: its large Q3 error dominates the RMSE||prognoza naivă ignoră sezonalitatea: eroarea ei mare din trimestrul III domină RMSE⟧',
            '⟦on seasonal data, the seasonal naive method is the benchmark to beat||pentru datele sezoniere, metoda naivă sezonieră este reperul de depășit⟧'])),
    '0.46', '0.50'), size='footnotesize')

chart(D, '⟦Forecasting electricity generation||Prognoza producției de electricitate⟧', 'tsa_ch0_forecast', 'TSA_ch0_forecast', [
    '⟦Test set: @{fc_test_first_m} -- @{fc_test_last_m} (24 months); training data: all earlier months; dashed: forecasts made once, at the end of the training set||Setul de test: @{fc_test_first_m} -- @{fc_test_last_m} (24 de luni); datele de antrenare: toate lunile anterioare; linii întrerupte: prognoze făcute o singură dată, la sfîrșitul setului de antrenare⟧'],
    h='0.58\\textheight')

D.frame('⟦Interpretation: which forecast wins?||Interpretarea: metoda cîștigătoare⟧', cols(
    table('lrrrr', '\\textbf{⟦Method||Metodă⟧} & MAE & RMSE & MAPE & MASE',
          ['⟦Naive||Naivă⟧ & @{fc_nv_mae} & @{fc_nv_rmse} & @{fc_nv_mape}\\% & @{fc_nv_mase}',
           '⟦Seasonal naive||Naivă sezonieră⟧ & @{fc_snv_mae} & @{fc_snv_rmse} & @{fc_snv_mape}\\% & @{fc_snv_mase}',
           'SES & @{fc_ses_mae} & @{fc_ses_rmse} & @{fc_ses_mape}\\% & @{fc_ses_mase}',
           'Holt--Winters & @{fc_hw_mae} & @{fc_hw_rmse} & @{fc_hw_mape}\\% & @{fc_hw_mase}'], size='scriptsize')
    + items('⟦MAE and RMSE in TWh per month||MAE și RMSE în TWh pe lună⟧'),
    items(('⟦\\textbf{Holt--Winters} has the smallest error on every measure||\\textbf{Holt--Winters} are cea mai mică eroare după fiecare măsură⟧',
           ['⟦it combines a falling level with the winter peak||combină nivelul în scădere cu vîrful de iarnă⟧']),
          ('⟦\\textbf{The seasonal naive method is not better than the naive one here}||\\textbf{Aici, metoda naivă sezonieră nu este mai bună decît cea naivă}⟧',
           ['⟦it copies the high winter of 2023--2024 into a lower 2024--2026||copiază iarna puternică din 2023--2024 peste perioada 2024--2026, cu un nivel mai scăzut⟧']),
          '⟦One test period of 24 months is one sample: a rolling origin gives a firmer answer||O singură perioadă de test de 24 de luni este un singur eșantion: o origine mobilă oferă un răspuns mai solid⟧'),
    '0.50', '0.46'), size='footnotesize')

D.frame('⟦Case study: the M4 competition||Studiu de caz: competiția M4⟧', items(
    ('⟦\\textbf{The question}: which forecasting methods work best on real data?||\\textbf{Întrebarea}: ce metode de prognoză funcționează cel mai bine pe date reale?⟧',
     ['⟦the M competitions, started by Spyros Makridakis in 1982: forecasts submitted blind, scored on hidden test data||competițiile M, inițiate de Spyros Makridakis în 1982: prognozele sînt trimise fără acces la datele de test și sînt evaluate pe acestea⟧']),
    ('⟦\\textbf{M4 (2018)}: 100\\,000 series (yearly to hourly), 61 methods (\\refMfour)||\\textbf{M4 (2018)}: 100\\,000 de serii (de la anuale la orare), 61 de metode (\\refMfour)⟧',
     ['⟦12 of the 17 most accurate methods were \\textbf{combinations} of several methods||12 dintre cele mai precise 17 metode au fost \\textbf{combinații} de metode⟧',
      '⟦the winner was a hybrid of exponential smoothing and a neural network||cîștigătoarea a fost o metodă hibridă, care combină netezirea exponențială cu o rețea neuronală⟧',
      '⟦pure machine learning methods did poorly: none beat a simple combination of exponential smoothing methods||metodele care folosesc exclusiv învățarea automată au avut rezultate slabe: niciuna nu a depășit o combinație simplă de metode de netezire exponențială⟧']),
    ('⟦\\textbf{Why it matters for this course}||\\textbf{Relevanța pentru acest curs}⟧',
     ['⟦benchmarks first; accuracy measured out of sample, on many series rather than one||întîi metodele de referință; acuratețea se măsoară în afara eșantionului, pe multe serii, nu pe una singură⟧',
      '⟦the same lesson appeared in M3 (\\refMthree): complex is not automatically better||aceeași lecție apăruse și în M3 (\\refMthree): o metodă complexă nu este automat mai bună⟧'])), size='footnotesize')

chart(D, '⟦A small M-Competition on Romanian data||O mică competiție M pe date românești⟧', 'tsa_ch0_benchmarks', 'TSA_ch0_forecast', [
    '⟦MASE on the test set (log scale) of four methods on four seasonal series; test sets: the last 8 quarters (GDP) or 24 months; dashed line: MASE = 1||MASE pe setul de test (scară logaritmică) pentru patru metode și patru serii sezoniere; seturi de test: ultimele 8 trimestre (PIB) sau 24 de luni; linia întreruptă: MASE = 1⟧'],
    h='0.56\\textheight')

D.frame('⟦Interpretation: the mini-competition||Interpretarea: mini-competiția⟧', items(
    ('⟦\\textbf{Holt--Winters wins on @{bm_hw_wins} of 4 series}||\\textbf{Holt--Winters cîștigă pe @{bm_hw_wins} din 4 serii}⟧',
     ['⟦CO2: MASE @{bm_co2_hw} against @{bm_co2_snv} for the seasonal naive method; HICP: @{bm_hicp_hw} against @{bm_hicp_nv} for the naive one||CO2: MASE @{bm_co2_hw}, față de @{bm_co2_snv} pentru metoda naivă sezonieră; IAPC: @{bm_hicp_hw}, față de @{bm_hicp_nv} pentru cea naivă⟧']),
    ('⟦\\textbf{GDP: the seasonal naive method wins}, MASE @{bm_gdp_snv} against @{bm_gdp_hw}||\\textbf{PIB: cîștigă metoda naivă sezonieră}, MASE @{bm_gdp_snv} față de @{bm_gdp_hw}⟧',
     ['⟦growth stalled in 2025--2026: the trend learnt by Holt--Winters overshoots||creșterea s-a oprit în 2025--2026: trendul estimat de Holt--Winters duce la prognoze prea mari⟧']),
    ('⟦\\textbf{Naive and SES fail on seasonal series}||\\textbf{Metodele naivă și SES eșuează pe seriile sezoniere}⟧',
     ['⟦GDP: MASE @{bm_gdp_nv} (naive) and @{bm_gdp_ses} (SES): they ignore the season||PIB: MASE @{bm_gdp_nv} (naivă) și @{bm_gdp_ses} (SES): ignoră sezonalitatea⟧']),
    '⟦As in M4: no method wins everywhere, and a simple benchmark can beat a better-looking model||Ca în M4: nicio metodă nu cîștigă peste tot, iar o metodă de referință simplă poate depăși un model aparent mai bun⟧'))

D.recap(('forecasts and their evaluation', 'prognoza și evaluarea ei'), [
    '⟦$\\hat y_{T+h|T}$ uses only the information up to $T$; uncertainty grows with $h$||$\\hat y_{T+h|T}$ folosește doar informația pînă la $T$; incertitudinea crește odată cu $h$⟧',
    '⟦Benchmarks: mean, naive, seasonal naive, drift||Metode de referință: media, naivă, naivă sezonieră, cu derivă⟧',
    '⟦Split by time; measure MAE, RMSE, MAPE and MASE on the test set only||Împărțiți datele în timp; măsurați MAE, RMSE, MAPE și MASE doar pe setul de test⟧',
    '⟦M4 and our mini-competition: simple methods are strong; combinations and Holt--Winters do well||M4 și mini-competiția noastră: metodele simple sînt puternice; combinațiile și Holt--Winters au rezultate bune⟧'])

# ===============================================================================================================
D.section('Tools', 'Instrumente')
# ===============================================================================================================
D.frame('⟦Python for time series||Python pentru serii de timp⟧', items(
    ('⟦\\textbf{The libraries of the course}||\\textbf{Bibliotecile cursului}⟧',
     ['⟦\\textbf{NumPy}: arrays and linear algebra (\\refNumpy)||\\textbf{NumPy}: tablouri și algebră liniară (\\refNumpy)⟧',
      '⟦\\textbf{pandas}: tables indexed by date, resampling, lags, differences (\\refPandas)||\\textbf{pandas}: tabele indexate după dată, schimbarea frecvenței, laguri, diferențe (\\refPandas)⟧',
      '⟦\\textbf{matplotlib}: charts||\\textbf{matplotlib}: grafice⟧',
      '⟦\\textbf{statsmodels}: decomposition, exponential smoothing, ARIMA, VAR, unit-root tests (\\refStatsmodels)||\\textbf{statsmodels}: descompunere, netezire exponențială, ARIMA, VAR, teste de rădăcină unitară (\\refStatsmodels)⟧',
      '⟦\\textbf{arch}: GARCH models (Chapter 5)||\\textbf{arch}: modele GARCH (Capitolul 5)⟧']),
    ('⟦\\textbf{Google Colab}: notebooks in the browser, nothing to install||\\textbf{Google Colab}: notebook-uri în browser, fără instalare⟧',
     ['⟦open a course notebook with one click; \\emph{File $\\to$ Save a copy in Drive} keeps your changes||deschideți un notebook al cursului cu un singur clic; \\emph{File $\\to$ Save a copy in Drive} vă păstrează modificările⟧',
      '⟦every chart of the slides has a Quantlet: the same code, runnable in Colab||fiecare grafic din slide-uri are un Quantlet: același cod, care poate fi rulat în Colab⟧'])))

D.frame('⟦A first script||Primul script⟧', cols(
    r"""\begin{lstlisting}
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Romanian real GDP, quarterly (Eurostat)
url = ('https://ec.europa.eu/eurostat/api/dissemination/'
       'sdmx/2.1/data/namq_10_gdp/Q.CLV10_MEUR.NSA.B1GQ.RO'
       '?format=SDMX-CSV')
t = pd.read_csv(url)
y = pd.Series(t['OBS_VALUE'].values / 1000,
              index=pd.PeriodIndex(t['TIME_PERIOD'], freq='Q'))
y = y.to_timestamp()

dec = seasonal_decompose(y, model='multiplicative', period=4)
print(dec.seasonal.iloc[:4])            # seasonal factors
yoy = 100 * (y / y.shift(4) - 1)         # annual growth, %
hw = ExponentialSmoothing(y, trend='add', seasonal='mul',
                          seasonal_periods=4).fit()
print(hw.forecast(4))                    # next four quarters
\end{lstlisting}""",
    items('⟦\\textbf{Line by line}||\\textbf{Linie cu linie}⟧',
          '⟦read the series directly from the Eurostat API (no key)||citirea seriei direct din API-ul Eurostat (fără cheie)⟧',
          '⟦\\texttt{PeriodIndex}: quarters as the time index||\\texttt{PeriodIndex}: trimestrele ca indice de timp⟧',
          '⟦classical multiplicative decomposition with $m = 4$||descompunere clasică multiplicativă, cu $m = 4$⟧',
          '⟦\\texttt{shift(4)}: the value of the same quarter a year earlier||\\texttt{shift(4)}: valoarea din același trimestru al anului anterior⟧',
          '⟦Holt--Winters with additive trend and multiplicative season||Holt--Winters cu trend aditiv și sezonalitate multiplicativă⟧',
          '⟦API: application programming interface||API (application programming interface): interfață de programare a aplicațiilor⟧'),
    '0.62', '0.35'), opts='fragile', size='scriptsize')

# ===============================================================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')
# ===============================================================================================================
D.frame('⟦Possible contribution of AI||Contribuția posibilă a AI⟧', items(
    ('⟦\\textbf{Where an AI assistant helps}||\\textbf{Unde ajută un asistent AI}⟧',
     ['⟦a first draft of the code that reads a Eurostat or BNR series and plots it||o primă variantă a codului care citește și reprezintă grafic o serie Eurostat sau BNR⟧',
      '⟦explaining an error message or a statsmodels function||explicarea unui mesaj de eroare sau a unei funcții statsmodels⟧',
      '⟦listing candidate explanations of a pattern (a drop in GDP, a jump in EUR/RON)||enumerarea explicațiilor posibile ale unui tipar (o scădere a PIB, un salt al cursului EUR/RON)⟧']),
    ('⟦\\textbf{Time-series errors that AI code makes often}||\\textbf{Greșeli frecvente în codul generat de AI pentru serii de timp}⟧',
     ['⟦a random train/test split, which lets the model see the future||împărțirea aleatoare în set de antrenare și set de test, care îi permite modelului să vadă viitorul⟧',
      '⟦growth over one quarter called ``annual growth\'\'; \\texttt{shift(1)} instead of \\texttt{shift(4)}||creșterea pe un trimestru numită „creștere anuală”; \\texttt{shift(1)} în loc de \\texttt{shift(4)}⟧',
      '⟦an RMSE without the square root; a MAPE on a series that crosses zero||RMSE fără rădăcina pătrată; MAPE pe o serie care trece prin zero⟧']),
    '⟦Seminar 0, exercise C2: find three such errors in an AI answer||Seminarul 0, exercițiul C2: găsiți trei astfel de greșeli într-un răspuns generat de AI⟧'))

D.frame('⟦Checks before trusting a result||Verificări necesare⟧', items(
    ('⟦\\textbf{The data}||\\textbf{Datele}⟧',
     ['⟦plot the series first; check the frequency, the units, the first and the last date||reprezentați seria grafic înainte de orice calcul; verificați frecvența, unitățile de măsură, prima și ultima dată⟧',
      '⟦adjusted or unadjusted? A model for the season needs the unadjusted series||ajustată sau neajustată? Un model pentru sezonalitate are nevoie de seria neajustată⟧']),
    ('⟦\\textbf{The evaluation}||\\textbf{Evaluarea}⟧',
     ['⟦the test set comes after the training set; it is used once, at the end||setul de test urmează după setul de antrenare; se folosește o singură dată, la final⟧',
      '⟦compare with the naive and seasonal naive forecasts before claiming success||comparați cu prognozele naivă și naivă sezonieră înainte de a declara un succes⟧']),
    ('⟦\\textbf{The claims}||\\textbf{Afirmațiile}⟧',
     ['⟦every reference: open the DOI; every number: recompute it||fiecare referință: deschideți DOI-ul; fiecare rezultat numeric: recalculați-l⟧',
      '⟦declare the use of AI in \\texttt{AI\\_USE.md}||declarați utilizarea AI în \\texttt{AI\\_USE.md}⟧'])))

# ===============================================================================================================
D.section('Conclusions', 'Concluzii')
# ===============================================================================================================
D.frame('⟦Key takeaways||Idei de reținut⟧', items(
    '⟦Grade: 70\\% exam, 20\\% project, 10\\% attendance; AI allowed and declared; seminars before lectures||Nota: 70\\% examen, 20\\% proiect, 10\\% prezență; AI permis și declarat; seminariile înaintea cursurilor⟧',
    '⟦A time series is ordered in time; its dependence on the past is what makes forecasting possible||O serie de timp este ordonată în timp; dependența de trecut face posibilă prognoza⟧',
    '⟦Components: trend, seasonality (fixed period), cycle (no fixed period), remainder; additive or multiplicative||Componente: trend, sezonalitate (perioadă fixă), ciclu (fără perioadă fixă), componenta neregulată; aditiv sau multiplicativ⟧',
    '⟦Yule and Slutsky: dependence and sums of shocks, the roots of AR and MA models||Yule și Slutsky: dependența și sumele de șocuri, originile modelelor AR și MA⟧',
    '⟦The ACF reveals trend (slow decay), season (peaks at $m$) and memory||ACF dezvăluie trendul (descreștere lentă), sezonalitatea (vîrfuri la $m$) și memoria⟧',
    '⟦Exponential smoothing weighs the recent past more; evaluate every forecast against naive benchmarks on a test set||Netezirea exponențială acordă ponderi mai mari trecutului recent; evaluați orice prognoză față de metodele naive, pe un set de test⟧'))

D.frame('⟦Self-assessment, project idea and next: Chapter 1||Autoevaluare, idee de proiect; urmează Capitolul 1⟧', cols(
    items(('⟦\\textbf{Self-assessment}||\\textbf{Autoevaluare}⟧',
           ['⟦Why is the quarter-on-quarter growth of unadjusted GDP misleading?||De ce este înșelătoare creșterea față de trimestrul anterior a PIB-ului neajustat?⟧',
            '⟦What does a seasonal factor of 0.76 for Q1 mean?||Ce înseamnă un factor sezonier de 0,76 pentru trimestrul I?⟧',
            '⟦What does an estimated $\\alpha$ close to 1 say about a series?||Ce spune despre o serie o valoare estimată a lui $\\alpha$ apropiată de 1?⟧',
            '⟦Why must the test set come after the training set?||De ce trebuie ca setul de test să urmeze după setul de antrenare?⟧']),
          ('⟦\\textbf{Project idea}||\\textbf{Idee de proiect}⟧',
           ['⟦forecast monthly electricity generation in Romania one year ahead; compare the seasonal naive method, Holt--Winters, and STL followed by SES, with a rolling origin||prognozați producția lunară de electricitate a României cu un an înainte; comparați metoda naivă sezonieră, Holt--Winters și STL urmată de SES, cu origine mobilă⟧'])),
    items(('⟦\\textbf{Chapter 1: stochastic processes and stationarity}||\\textbf{Capitolul 1: procese stochastice și staționaritate}⟧',
           ['⟦What is a stochastic process?||Ce este un proces stochastic?⟧',
            '⟦When does a series have a stable mean and variance?||Cînd are o serie medie și varianță stabile?⟧',
            '⟦What are white noise and a random walk?||Ce sînt zgomotul alb și mersul aleator?⟧',
            '⟦How do we test that autocorrelations are zero?||Cum testăm că autocorelațiile sînt nule?⟧'])),
    '0.52', '0.44'), size='footnotesize')

D.references(REFERENCES, per=13)

if __name__ == '__main__':
    for _k, _v in list(V.items()):   # a true minus sign for negative numbers, in text and in math
        if isinstance(_v, str) and _v.startswith('⁅-'):
            V[_k] = '⁅\\ensuremath{-}' + _v[2:]
    D.write(V)
