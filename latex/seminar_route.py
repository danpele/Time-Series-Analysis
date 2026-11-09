r"""
seminar_route.py -- the three opening frames of a seminar, EN + RO (as in MFM seminar 1)
=========================================================================================
  (1/3) the question of the seminar; the three exercise types A/B/C; the markers [Rezolvat]/[Propus]
        (no remark that the seminar comes before the lecture: instructor's rule)
  (2/3) today's exercises in order, each as „cod: titlu scurt” (titles from the exercise frames), and, if any,
        the parts that need a concept of the lecture
  (3/3) the results to obtain, the notebook, the seminar instructor

`lecture` is the chapter number (kept for reference; it is not printed).
"""

from tsa_build import items

INSTRUCTOR = 'prof.\\ Daniel Traian Pele, \\href{mailto:danpele@ase.ro}{danpele@ase.ro}'


def L(en, ro):
    return f'⟦{en}||{ro}⟧'


def route(D, lecture, question, today, after=None, outputs=None, notebook=None, extra=None,
          size='small', title=None):
    """question: (en, ro), or a list of (en, ro) for several questions.
    today: list of (code, en, ro), or list of (heading_en, heading_ro, [(code, en, ro), ...]) groups.
    after: (heading_en, heading_ro, [(code, en, ro), ...]); a code may be a pair (en, ro), e.g. ('C1, step 3', 'C1, pasul 3') for the parts that need the lecture.
    outputs: list of (en, ro); default: A answers, B figure and table, C2 corrected answer, one interpretation.
    extra: list of bilingual items added at the end of (3/3)."""
    t_en, t_ro = title or ("Today's Question and Seminar Structure", 'Întrebarea de azi și structura seminarului')

    def ft(k):
        return L(f'{t_en} ({k}/3)', f'{t_ro} ({k}/3)')

    if isinstance(question, tuple):
        q = (L('The question of the seminar', 'Întrebarea seminarului'),
             [L(f'\\textbf{{{question[0]}}}', f'\\textbf{{{question[1]}}}')])
    else:
        q = (L('The questions of the seminar', 'Întrebările seminarului'), [L(e, r) for e, r in question])
    D.frame(ft(1), items(
        q,
        (L('Three types of exercises', 'Trei tipuri de exerciții'),
         [L('\\textbf{A}: calculations, on paper', '\\textbf{A}: exerciții de calcul, pe hîrtie'),
          L('\\textbf{B}: real data, solved in the notebook', '\\textbf{B}: exerciții pe date reale, rezolvate în notebook'),
          L('\\textbf{C}: open questions, for example correcting an answer given by an AI tool',
            '\\textbf{C}: întrebări deschise, de exemplu corectarea unui răspuns dat de un instrument AI')]),
        (L('Each exercise is marked', 'Fiecare exercițiu este marcat'),
         [L('\\textbf{[Solved]}: the full solution is presented in the seminar',
            '\\textbf{[Rezolvat]}: rezolvarea completă este prezentată la seminar'),
          L('\\textbf{[Proposed]}: you solve it; the solution is discussed in the seminar',
            '\\textbf{[Propus]}: îl rezolvați dumneavoastră; soluția se discută la seminar')])), size)

    def ex(xs):
        return [L(f'{c[0] if isinstance(c, tuple) else c}: {e}', f'{c[1] if isinstance(c, tuple) else c}: {r}')
                for c, e, r in xs]

    if today and len(today[0]) == 3 and isinstance(today[0][2], list):
        body = [(L(he, hr), ex(xs)) for he, hr, xs in today]
    else:
        body = [(L('Today\'s exercises, in order', 'Exercițiile de azi, în ordine'), ex(today))]
    if after:
        body.append((L(after[0], after[1]), ex(after[2])))
    D.frame(ft(2), items(*body), size)

    outs = outputs or [('the answers to the A exercises', 'răspunsurile la exercițiile A'),
                       ('a figure and a table for each B exercise', 'cîte o figură și un tabel pentru fiecare exercițiu B'),
                       ('the corrected AI answer of C2', 'răspunsul AI corectat din C2'),
                       ('one sentence of interpretation for each exercise', 'o frază de interpretare pentru fiecare exercițiu')]
    nb = notebook or ('Notebook for today: \\href{\\nb}{open the seminar notebook in Google Colab}',
                      'Notebook-ul de azi: \\href{\\nb}{deschideți notebook-ul seminarului în Google Colab}')
    D.frame(ft(3), items(
        (L('Results to obtain (nothing is handed in)', 'Rezultate de obținut (nu se predă nimic)'), [L(e, r) for e, r in outs]),
        (L(*nb), [L('each task names its notebook section', 'fiecare cerință indică secțiunea din notebook')]),
        L(f'Seminar: {INSTRUCTOR}', f'Seminar: {INSTRUCTOR}'),
        *(extra or [])), size)
