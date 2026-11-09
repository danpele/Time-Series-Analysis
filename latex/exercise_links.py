r"""
exercise_links.py -- clickable exercise codes and back buttons in the seminar decks (as in MFM seminar 1)
=======================================================================================================
  * every exercise code (A1, B3, C2, ...) in the text of a seminar deck links to the exercise's task frame
    (the first frame whose title starts with the code);
  * the task frame gets \hypertarget{ex-<code>}{}\exback{} inline after its first \item: \exback draws two
    buttons in the bottom right corner, „Înapoi”/„Back” (the viewer's GoBack) and
    „Lista exercițiilor”/„Exercise list” (the structure slide (2/3), target ex-list);
  * codes that refer to another seminar („Seminar 5, A1 and A4”, „Seminarul 7, B1 și A3”) stay plain text.

Called by tsa_build.Deck.write for seminars, after the acronym glossary; idempotent.
    python3 latex/exercise_links.py N [N ...]      # the seminar decks of chapters N (EN + RO)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ID = r'[ABC]\d{1,2}'
PREAMBLE = '\\input{../../latex/preamble}'
TASK_TITLE = re.compile(r'^\\begin\{frame\}(?:\[[^\]]*\])?\{(' + ID + r')(?=[: ])')
STRUCT = re.compile(r"^\\begin\{frame\}\{(?:Today's Question and Seminar Structure|"
                    r'Întrebarea de azi și structura seminarului) \(2/\d\)\}$')
REF = re.compile(r'(?<![A-Za-z0-9_\\{-])(' + ID + r')(?![0-9A-Za-z_])')
# a code (or a chain of codes) that belongs to another seminar
OTHER = re.compile(r'Seminar(?:ul|ului)?(?:~|\s)+\d+(?:,|\s)+(?:(?:exercise|task|exercițiul|cerința)\s+)?'
                   + ID + r'(?:(?:\s*,\s*|\s+(?:and|și)\s+)' + ID + r')*')
# spans never touched: math, \texttt, link arguments
PROTECT = re.compile(r'(\$[^$]*\$|\\texttt\{[^}]*\}|\\href\{[^}]*\}|\\[A-Za-z]*link[A-Za-z]*\{[^}]*\}|'
                     + OTHER.pattern + ')')


def macros(ro):
    back, lst = ('Înapoi', 'Lista exercițiilor') if ro else ('Back', 'Exercise list')
    return ('\\providecommand{\\exback}{\\begin{tikzpicture}[remember picture,overlay]\\node[anchor=south east] at '
            '([xshift=-3mm,yshift=7mm]current page.south east){\\Acrobatmenu{GoBack}{\\beamerreturnbutton{%s}}\\ '
            '\\hyperlink{ex-list}{\\beamergotobutton{%s}}};\\end{tikzpicture}}\n'
            '\\providecommand{\\exlink}[2]{\\hyperlink{#1}{\\textcolor{MainBlue}{#2}}}' % (back, lst))


def first_item(lines, i):
    """Index of the first line with '\\item ' in the frame that starts at line i (None if there is none)."""
    for j in range(i + 1, len(lines)):
        if '\\end{frame}' in lines[j]:
            return None
        if '\\item ' in lines[j]:
            return j
    return None


def inline_after_item(line, text):
    k = line.index('\\item ') + len('\\item ')
    return line[:k] + text + line[k:]


def process(path):
    s = open(path, encoding='utf-8').read()
    if '\\providecommand{\\exlink}' in s:      # already processed
        return 0, 0
    if PREAMBLE not in s:
        return 0, 0
    ro = path.endswith('_ro.tex') or '/RO/' in path
    s = s.replace(PREAMBLE, PREAMBLE + '\n' + macros(ro), 1)
    lines = s.split('\n')
    start = next(i for i, l in enumerate(lines) if l.startswith('\\begin{document}'))

    # targets: the first frame of each code
    targets, marks = {}, {}
    for i in range(start, len(lines)):
        m = TASK_TITLE.match(lines[i])
        if m and m.group(1) not in targets:
            key = 'ex-' + m.group(1)
            targets[m.group(1)] = key
            j = first_item(lines, i)
            if j is None:
                raise SystemExit(f'{path}: no \\item in the task frame of {m.group(1)}')
            marks[j] = '\\hypertarget{%s}{}\\exback{}' % key
        elif STRUCT.match(lines[i]) and 'ex-list' not in marks.values():
            j = first_item(lines, i)
            if j is not None:
                marks[j] = '\\hypertarget{ex-list}{}'

    def rep(m):
        t = targets.get(m.group(1))
        return '\\exlink{%s}{%s}' % (t, m.group(1)) if t else m.group(1)

    out, glossary = [], False
    for i, l in enumerate(lines):
        st = l.lstrip()
        if 'BEGIN-ACRONYMS' in l:
            glossary = True
        if 'END-ACRONYMS' in l:
            glossary = False
        if not (i < start or glossary or st.startswith(('\\begin{frame}', '\\section', '%'))
                or '\\includegraphics' in l or '\\quantlet' in l):
            parts = PROTECT.split(l)
            l = ''.join(p if (p and PROTECT.fullmatch(p)) else REF.sub(rep, p) for p in parts if p is not None)
        if i in marks:
            l = inline_after_item(l, marks[i])
        out.append(l)
    s = '\n'.join(out)
    if '\\hypertarget{ex-list}' not in s:
        raise SystemExit(f'{path}: no structure slide (2/n) for the ex-list target')
    open(path, 'w', encoding='utf-8').write(s)
    return len(targets), s.count('\\exlink{')


def seminar_decks(chapters):
    sys.path.insert(0, HERE)
    from tsa_chapters import deck
    for n in chapters:
        for lang in ('en', 'ro'):
            yield os.path.join(ROOT, deck(int(n), 'seminar', lang))


if __name__ == '__main__':
    for p in seminar_decks(sys.argv[1:]):
        if os.path.exists(p):
            t, k = process(p)
            print(f'{os.path.relpath(p, ROOT)}: {t} exercise targets, {k} links')
