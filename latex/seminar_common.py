r"""
seminar_common.py -- glossary and chapter links for the seminar decks of one chapter only
========================================================================================
`acronyms.py N` also rewrites the lecture decks of chapter N. The seminar generators of the self-study chapters
(build_seminar13.py, build_seminar14.py) use this helper instead, so that building a seminar never touches the
lecture files: acronym glossary (acronyms.process) and the chapter links of appendix_links.py, on the seminar
decks (EN and RO) of the chapter.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_chapters import new_decks   # noqa: E402
from tsa_build import ROOT           # noqa: E402


def seminar_glossary(n):
    import acronyms
    import appendix_links as al
    for p, lang, kind, _ in new_decks(kinds=('seminar',), chapters={str(n)}):
        keys, missing = acronyms.process(p, lang)
        print(f'{os.path.relpath(p, ROOT)}: {len(keys)} acronyms' + (f'; NOT IN DICTIONARY: {missing}' if missing else ''))
        t = al.strip_old(open(p, encoding='utf-8').read())
        t, _ = al.chapter_links(t, lang)
        d = t.find('\\begin{document}')
        open(p, 'w', encoding='utf-8').write(t[:d] + al.MACROS + t[d:])
