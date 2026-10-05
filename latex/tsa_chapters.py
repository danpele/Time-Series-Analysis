r"""
tsa_chapters.py -- registrul capitolelor TSA si schema de nume a fisierelor noi
================================================================================
Titlurile vin din assets/course-data.js (capitolele 0-15). Din ele se deriveaza "slug"-urile fisierelor:

  curs EN       EN/Courses/chapterN_<slug_en>.tex/.pdf
  curs RO       RO/Cursuri/capitolN_<slug_ro>.tex/.pdf
  seminar EN    EN/Seminars/seminarN_<slug_en>.tex/.pdf          (+ seminarN_<slug_en>_solutions.tex, privat)
  seminar RO    RO/Seminarii/seminarN_<slug_ro>_ro.tex/.pdf      (+ seminarN_<slug_ro>_ro_solutions.tex, privat)
  notebook-uri  notebooks/EN/chapterN_lecture_notebook.ipynb, notebooks/EN/chapterN_seminar_notebook.ipynb
  Quantlet-uri  Quantlets/Ch_NN/TSA_chN_<nume>/
  site          https://danpele.github.io/Time-Series-Analysis/<cale>.pdf

Slug = titlul fara diacritice, cu litere mici, fara cuvintele de legatura (and, of, the, si, ...), cuvintele unite
cu "_"; pentru titlurile lungi se foloseste o forma scurta (SHORT). Deck-urile vechi (EN/Courses/chapterN_*.tex cu
alt nume, RO/Courses/chapterN_*_ro.tex, RO/Seminars/chapterN_seminar_ro.tex) raman unde sint pina la reconstruirea capitolului.

Un deck face parte din noul flux daca exista la calea de mai sus SI contine "\input{../../latex/preamble}"
(is_new_pipeline). Doar aceste deck-uri primesc glosar (acronyms.py) si legaturi intre capitole (appendix_links.py).

Verificare:  python3 latex/tsa_chapters.py        (tabelul numelor; avertizeaza daca titlurile din site s-au schimbat)
"""

import os
import re
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SITE = 'https://danpele.github.io/Time-Series-Analysis/'
REPO = 'https://github.com/danpele/Time-Series-Analysis'
COLAB = 'https://colab.research.google.com/github/danpele/Time-Series-Analysis/blob/main/'

COURSE = {'en': 'Time Series Analysis', 'ro': 'Serii de timp'}
PRESET_MARK = r'\input{../../latex/preamble}'

# N -> (titlu EN, titlu RO), ca in assets/course-data.js (capitolele 11-14: studiu individual)
TITLES = {
    0: ('Introduction: components and exponential smoothing', 'Introducere: componente și netezire exponențială'),
    1: ('Stochastic processes and stationarity', 'Procese stochastice și staționaritate'),
    2: ('ARMA models', 'Modele ARMA'),
    3: ('Unit roots and ARIMA models', 'Rădăcini unitare și modele ARIMA'),
    4: ('Seasonality and forecasting: SARIMA, TBATS, Prophet', 'Sezonalitate și prognoză: SARIMA, TBATS, Prophet'),
    5: ('Conditional volatility: ARCH and GARCH', 'Volatilitate condiționată: ARCH și GARCH'),
    6: ('VAR models and Granger causality', 'Modele VAR și cauzalitate Granger'),
    7: ('Cointegration and VECM', 'Cointegrare și VECM'),
    8: ('Long memory and ARFIMA', 'Memorie lungă și ARFIMA'),
    9: ('Machine learning for time series', 'Învățare automată pentru serii de timp'),
    10: ('State space models, Kalman filter and Markov switching', 'Modele în spațiul stărilor, filtrul Kalman și modele Markov switching'),
    11: ('Foundation models for time series', 'Foundation models pentru serii de timp'),
    12: ('Spectral analysis', 'Analiză spectrală'),
    13: ('Speculative bubbles: LPPL models', 'Bule speculative: modele LPPL'),
    14: ('Multivariate GARCH models', 'Modele GARCH multivariate'),
    15: ('Review and exam preparation', 'Recapitulare și pregătire pentru examen'),
}
SELF_STUDY = {11, 12, 13, 14}

# forme scurte pentru titlurile lungi (aceleasi cuvinte-cheie ca titlul)
SHORT = {
    0: ('introduction', 'introducere'),
    4: ('seasonality_forecasting', 'sezonalitate_prognoza'),
    5: ('conditional_volatility_garch', 'volatilitate_conditionata_garch'),
    10: ('state_space_kalman_markov_switching', 'spatiul_starilor_kalman_markov_switching'),
    13: ('bubbles_lppl', 'bule_lppl'),
    15: ('review', 'recapitulare'),
}

STOP = {'and', 'of', 'the', 'a', 'an', 'in', 'for', 'si', 'de', 'ale', 'al', 'a', 'in', 'pentru', 'testele',
        'ipoteza', 'hypothesis', 'tests'}


def slugify(title):
    t = title.replace('α', 'alpha ').replace('&', ' ')
    t = unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode().lower()
    t = t.replace('variance-ratio', 'vr')
    words = [w for w in re.findall(r'[a-z0-9]+', t) if w not in STOP]
    words = ['alfa' if w == 'alpha' and any(x in title for x in 'ăâîșțȘȚ') else w for w in words]
    return '_'.join(words)


def slugs(n):
    if n in SHORT:
        return SHORT[n]
    en, ro = TITLES[n]
    s_ro = slugify(ro)
    if 'α' in ro:
        s_ro = s_ro.replace('alpha', 'alfa')
    return slugify(en), s_ro


def paths(n):
    """Caile relative la radacina depozitului (fara extensie pentru deck-uri)."""
    en, ro = slugs(n)
    return {
        'lecture_en': f'EN/Courses/chapter{n}_{en}',
        'lecture_ro': f'RO/Cursuri/capitol{n}_{ro}',
        'seminar_en': f'EN/Seminars/seminar{n}_{en}',
        'seminar_ro': f'RO/Seminarii/seminar{n}_{ro}_ro',
        'nb_lecture': f'notebooks/EN/chapter{n}_lecture_notebook.ipynb',
        'nb_seminar': f'notebooks/EN/chapter{n}_seminar_notebook.ipynb',
        'quantlets': f'Quantlets/Ch_{n:02d}',
    }


def deck(n, kind, lang):
    """Calea relativa a unui deck (.tex): kind in {'lecture', 'seminar'}, lang in {'en', 'ro'}."""
    return paths(n)[f'{kind}_{lang}'] + '.tex'


def is_new_pipeline(path):
    """True daca fisierul .tex exista si foloseste preambulul comun latex/preamble.tex."""
    if not os.path.exists(path):
        return False
    with open(path, encoding='utf-8') as f:
        return PRESET_MARK in f.read(20000)


def pdf_url(n, kind, lang):
    return SITE + deck(n, kind, lang).replace('.tex', '.pdf')


def new_decks(kinds=('lecture', 'seminar'), chapters=None):
    """(cale absoluta, limba, kind, N) pentru deck-urile existente in noul flux."""
    out = []
    for n in sorted(TITLES):
        if chapters and str(n) not in chapters and n not in chapters:
            continue
        for kind in kinds:
            for lang in ('en', 'ro'):
                p = os.path.join(ROOT, deck(n, kind, lang))
                if is_new_pipeline(p):
                    out.append((p, lang, kind, n))
    return out


def site_titles():
    """Titlurile din assets/course-data.js (pentru verificare)."""
    p = os.path.join(ROOT, 'assets', 'course-data.js')
    if not os.path.exists(p):
        return {}
    s = open(p, encoding='utf-8').read()
    out = {}
    for m in re.finditer(r"num:\s*(\d+),\s*\n\s*title:\s*\{\s*en:\s*'([^']*)',\s*ro:\s*'([^']*)'", s):
        out[int(m.group(1))] = (m.group(2), m.group(3))
    return out


if __name__ == '__main__':
    st = site_titles()
    for n in sorted(TITLES):
        p = paths(n)
        flag = '' if st.get(n, TITLES[n]) == TITLES[n] else '   <-- title differs from assets/course-data.js: ' + str(st.get(n))
        built = [k for k in ('lecture_en', 'lecture_ro', 'seminar_en', 'seminar_ro')
                 if is_new_pipeline(os.path.join(ROOT, p[k] + '.tex'))]
        print(f'{n:2d}  {p["lecture_en"]}.pdf | {p["lecture_ro"]}.pdf | {p["seminar_en"]}.pdf | {p["seminar_ro"]}.pdf'
              + (f'   [built: {", ".join(built)}]' if built else '') + flag)
