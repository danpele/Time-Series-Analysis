r"""
build_chapter0.py -- Capitolul 0 (Introducere: componente și netezire exponențială), EN + RO, în noul flux TSA
=============================================================================================================
Etapa de infrastructură: conținutul este cel existent (deck-urile din primăvara 2026), convertit la preambulul
comun (latex/preamble.tex), cu titlul standard, numele noi și glosarul de acronime. Cînd capitolul va fi rescris,
acest fișier devine un generator bilingv complet (Deck din latex/tsa_build.py, text ⟦EN||RO⟧).
Surse (deck-urile vechi rămîn neschimbate):
  curs     EN  EN/Courses/chapter0_fundamentals.tex        RO  RO/Courses/chapter0_fundamentals_ro.tex
  seminar  EN  EN/Seminars/chapter0_seminar.tex            RO  RO/Seminars/chapter0_seminar_ro.tex
Ieșire (python3 latex/tsa_chapters.py):
  EN/Courses/chapter0_introduction.tex        RO/Cursuri/capitol0_introducere.tex
  EN/Seminars/seminar0_introduction.tex       RO/Seminarii/seminar0_introducere_ro.tex  (+ wrapper-ele *_solutions.tex)
Rulare:  python3 latex/build_chapter0.py   apoi   python3 latex/tsa_build.py compile 0
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import ROOT, convert_legacy, run_acronyms   # noqa: E402

SRC = {
    ('lecture', 'en'): 'EN/Courses/chapter0_fundamentals.tex',
    ('lecture', 'ro'): 'RO/Courses/chapter0_fundamentals_ro.tex',
    ('seminar', 'en'): 'EN/Seminars/chapter0_seminar.tex',
    ('seminar', 'ro'): 'RO/Seminars/chapter0_seminar_ro.tex',
}
# small text replacements only (the content is rewritten when the chapter is rebuilt)
REPLACE = {
    'en': [('Time Series Analysis and Forecasting', 'Time Series Analysis'),
           ('\\begin{frame}{Quiz 1: Answer}', '\\begin{frame}[shrink=5]{Quiz 1: Answer}')],   # overfull vbox (seminar)
    'ro': [('Analiza și Prognoza Seriilor de Timp', 'Serii de timp'),
           # overfull vbox in the RO deck (longer text than EN): smaller chart and font only
           ('height=0.42\\textheight, keepaspectratio]{cross_validation_forecast.pdf}',
            'height=0.34\\textheight, keepaspectratio]{cross_validation_forecast.pdf}'),
           ('\\vspace{-0.2cm}\n        {\\small\n        \\begin{block}{Context}',
            '\\vspace{-0.2cm}\n        {\\footnotesize\n        \\begin{block}{Context}'),
           ('{\\small\n        \\begin{block}{Scop}', '{\\footnotesize\n        \\begin{block}{Scop}'),
           ('{\\small\n        \\begin{block}{Metode bootstrap}', '{\\footnotesize\n        \\begin{block}{Metode bootstrap}'),
           ('\\begin{frame}{Proprietăți asimptotice vs.\\ eșantion mic}', '\\begin{frame}[shrink=10]{Proprietăți asimptotice vs.\\ eșantion mic}'),
           ('\\begin{frame}{Simulare Monte Carlo: Principii}', '\\begin{frame}[shrink=5]{Simulare Monte Carlo: Principii}'),
           ('\\begin{frame}{Corecții pentru eșantion mic}', '\\begin{frame}[shrink=8]{Corecții pentru eșantion mic}')],
}

if __name__ == '__main__':
    for (kind, lang), rel in SRC.items():
        convert_legacy(os.path.join(ROOT, rel), 0, lang, kind=kind, replace=REPLACE[lang])
    run_acronyms(0)
