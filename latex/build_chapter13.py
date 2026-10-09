r"""
build_chapter13.py -- Capitolul 13 (Bule speculative: modele LPPL), EN + RO dintr-o singură sursă
================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_13/ch13_numbers.json (generate_all_charts.py).
Nicio cifră nu este scrisă de mînă. Capitol de studiu individual: fără seminar; exemple rezolvate, slide-uri de
interpretare, recapitulări și autoevaluare cu răspunsuri. Sursa: vechiul capitol 13 TSA (LPPL), scurtat și rescris,
plus materialul MFM, capitolul 17 (bule și crahuri).
Ieșire:
  EN/Courses/chapter13_bubbles_lppl.tex
  RO/Cursuri/capitol13_bule_lppl.tex
Rulare:
  python3 Quantlets/Ch_13/generate_all_charts.py
  python3 latex/build_chapter13.py && python3 latex/tsa_build.py compile 13
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, photo   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch13_common import QLURL, REFS, T, V2, bib, date, finalize, load, month   # noqa: E402


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


N = load()
V = Values()
D = Deck(13, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\quantlet{{{folder.replace("_", chr(92) + "_")}}}{{\\qlurl{{{folder}}}}}'


def chart(title, fig, folder, bullets, h='0.6\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def interp(title, bullets, size='small'):
    D.frame(T(f'Interpreting {title[0]}', f'Interpretarea {title[1]}'), items(*bullets), size)


PH = {
    'tulip': ('ch13_tulip_mania_1637.jpg', C + "Flora's_Wagon_of_Fools_(Flora's_Mallewagen)_tulipomania,_Hendrik_Gerritsz_Pot_c1637.jpg",
              T('Painting', 'Pictură') + ': Hendrik Gerritsz Pot (c.~1637); ' + T('public domain', 'domeniu public') + '; Wikimedia Commons'),
    'nyse': ('ch13_nyse_crowd_1929.jpg', C + 'Crowd_outside_nyse.jpg',
             T('Photo', 'Foto') + ': US government (1929); ' + T('public domain', 'domeniu public') + '; Wikimedia Commons'),
    'amsterdam': ('ch13_amsterdam_crash_1987.jpg', C + 'Effectenbeurs_Amsterdam_na_koersval,_Bestanddeelnr_934-1094.jpg',
                  T('Photo', 'Foto') + ': Bart Molendijk / Anefo, Nationaal Archief (1987); CC0; Wikimedia Commons'),
    'sornette': ('ch13_sornette_2012.jpg', C + 'Didier_Sornette.jpg',
                 T('Photo', 'Foto') + ': Didier Sornette (2012); CC BY-SA 3.0 DE; Wikimedia Commons'),
    'petscom': ('ch13_petscom_puppet_2013.jpg', C + 'Pets.com_Sock_Puppet_(12935451763).jpg',
                T('Photo', 'Foto') + ': Atomic Taco (2013); CC BY-SA 2.0; Wikimedia Commons'),
    'shiller': ('ch13_shiller_2017.jpg', C + 'Robert_J._Shiller_2017.jpg',
                T('Photo', 'Foto') + ': Presidential Office Building, Taiwan (2017); CC BY 2.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


def two(left, right, wl='0.38', wr='0.6'):
    return (f'\\begin{{columns}}[T]\n\\begin{{column}}{{{wl}\\textwidth}}\n{left}\n\\end{{column}}\n'
            f'\\begin{{column}}{{{wr}\\textwidth}}\n{right}\n\\end{{column}}\n\\end{{columns}}')


def yes(b):
    return T('yes', 'da') if b else T('no', 'nu')


# =============================================================================
# CIFRE
# =============================================================================
P = V.put
LAB = {'sp500': ('S\\&P 500, 2000', 'S\\&P 500, 2000'), 'ndx': ('Nasdaq 100, 2000', 'Nasdaq 100, 2000'),
       'bet': ('BET, 2007', 'BET, 2007'), 'ssec': ('Shanghai Composite, 2015', 'Shanghai Composite, 2015'),
       'btc17': ('Bitcoin, 2017', 'Bitcoin, 2017'), 'btc21': ('Bitcoin, 2021', 'Bitcoin, 2021')}

E = N['episodes']
for k, e in E.items():
    V.raw(f'ep.{k}.low', date(e['low']))
    V.raw(f'ep.{k}.peak', date(e['peak']))
    V.raw(f'ep.{k}.trough', date(e['trough']))
    P(f'ep.{k}.runup', e['runup'], 0, pct=True)
    P(f'ep.{k}.years', e['years'], 1)
    P(f'ep.{k}.growth', e['growth'], 0, pct=True)
    P(f'ep.{k}.fall', -e['fall1y'], 0, pct=True)
P('ep.btc17.mult', E['btc17']['p_peak'] / E['btc17']['p_low'], 0)
GR = N['growth']
P('gr.start', GR['g_start'], 2)
P('gr.mid', GR['g_mid'], 2)
P('gr.end', GR['g_end'], 2)

RA = N['rational']
P('ra.max', RA['bw_max'], 0)
V.raw('ra.bursts', str(RA['bw_bursts']))
P('ra.g100', RA['g100'], 2)
P('ra.g35', RA['g35'], 0)
EX = N['ex']
P('bw.g', EX['bw']['g'], 2, pct=True)
P('bw.dur', EX['bw']['dur'], 0)
P('bw.half', EX['bw']['half'], 0)
for t, key in (('1.0', 'a'), ('1.75', 'b'), ('1.99', 'c')):
    x = EX[t]
    P(f'lp.{key}.dt', x['dt'], 2)
    P(f'lp.{key}.fm', x['fm'], 3)
    P(f'lp.{key}.lg', x['lg'], 3)
    P(f'lp.{key}.cos', x['cos'], 3)
    P(f'lp.{key}.lnp', x['lnp'], 3)
    P(f'lp.{key}.p', EX['p'][t], 0)

PN = N['psy_ndx']
V.raw('pn.T', str(PN['T']))
V.raw('pn.w0', str(PN['w0']))
P('pn.r0', PN['r0'], 3)
P('pn.gsadf', PN['gsadf'], 2)
P('pn.gsadf95', PN['cv']['gsadf']['95'], 2)
P('pn.sadf', PN['sadf'], 2)
P('pn.sadf95', PN['cv']['sadf']['95'], 2)
P('pn.adf', PN['adf'], 2)
P('pn.adf95', PN['cv']['adf']['95'], 2)
V.raw('pn.minlen', str(PN['min_len']))
V.raw('pn.nep', str(len(PN['episodes'])))
main = max(PN['episodes'], key=lambda e: e[2])
V.raw('pn.ep.a', date(main[0]))
V.raw('pn.ep.b', date(main[1]))
V.raw('pn.ep.n', str(main[2]))
V.raw('pn.ep0.a', month(PN['episodes'][0][0]))
XE = PN['example']
V.raw('xe.start', date(XE['start']))
V.raw('xe.end', date(XE['end']))
V.raw('xe.n', str(XE['n']))
P('xe.delta', XE['delta'], 4)
P('xe.se', XE['se'], 4)
P('xe.t', XE['t'], 2)
P('xe.cv', XE['cv'], 2)
P('xe.a', XE['a'], 4)

PS = N['psy']
for k in ('btc', 'bet', 'ssec'):
    P(f'ps.{k}.gsadf', PS[k]['gsadf'], 2)
    P(f'ps.{k}.cv', PS[k]['cv95'], 2)
    V.raw(f'ps.{k}.T', str(PS[k]['T']))
    for i, (a, b, n_) in enumerate(PS[k]['episodes']):
        V.raw(f'ps.{k}.e{i}.a', month(a))
        V.raw(f'ps.{k}.e{i}.b', month(b))

CO = N['comp']
for w in ('6.28', '8', '10'):
    P(f'lam.{w}', CO['lam'][w], 2)
for i in (1, 2, 3):
    P(f'lam.d{i}', 200 / CO['lam']['8'] ** i, 0)
P('lp.eA', __import__('math').exp(8), 0)

CS = N['cost']
V.int('cost.n', CS['n_grid'])
V.raw('cost.loc', str(CS['locmin']))
V.raw('cost.tc', date(CS['tc_best']))
P('cost.ratio', 100 * (CS['ssr_ratio'] - 1), 1)

FI = N['fits']
for k in LAB:
    f = FI[k]
    V.raw(f'fi.{k}.t2', date(f['t2']))
    V.raw(f'fi.{k}.tc', date(f['tc']))
    V.raw(f'fi.{k}.err', ('+' if f['tc_err'] > 0 else '$-$') + str(abs(f['tc_err'])))
    P(f'fi.{k}.m', f['m'], 2)
    P(f'fi.{k}.w', f['w'], 1)
    P(f'fi.{k}.B', f['B'], 2)
    V.raw(f'fi.{k}.qp', yes(f['q_param']))
    V.raw(f'fi.{k}.qf', yes(f['q_full']))
    V.raw(f'fi.{k}.n', str(f['n']))
FX = FI['example']
for k in ('dt', 'fm', 'lg', 'cos', 'sin', 'lnp_hat', 'lnp'):
    P(f'fx.{k}', FX[k], 3)
for k in ('A', 'B', 'C1', 'C2'):
    P(f'fx.{k}', FX[k], 3)
P('fx.m', FX['m'], 3)
P('fx.w', FX['w'], 2)
P('fx.t2', FX['t2'], 3)
P('fx.tc', FX['tc'], 3)
P('fx.p', FX['p'], 0)
P('fx.phat', FX['p_hat'], 0)
P('fx.err', 100 * (FX['p_hat'] / FX['p'] - 1), 1)
P('fx.C', (FX['C1'] ** 2 + FX['C2'] ** 2) ** 0.5, 3)
P('fx.Bfm', FX['B'] * FX['fm'], 3)
P('fx.osc', FX['fm'] * (FX['C1'] * FX['cos'] + FX['C2'] * FX['sin']), 3)

TP = N['tcpath']
for k in ('ndx', 'btc17'):
    V.raw(f'tp.{k}.n', str(TP[k]['n']))
    P(f'tp.{k}.q', TP[k]['share_q'], 0, pct=True)
    V.raw(f'tp.{k}.min', date(TP[k]['tc_min']))
    V.raw(f'tp.{k}.max', date(TP[k]['tc_max']))
    V.raw(f'tp.{k}.range', str(TP[k]['range_days']))
    V.raw(f'tp.{k}.err', str(int(TP[k]['err_med'])))

WI = N['windows']
for k in ('ndx', 'ssec'):
    w = WI[k]
    V.raw(f'wi.{k}.np', str(w['n_param']))
    V.raw(f'wi.{k}.nf', str(w['n_full']))
    V.raw(f'wi.{k}.n', str(w['n']))
    V.raw(f'wi.{k}.t2', date(w['t2']))
    V.raw(f'wi.{k}.peak', date(w['peak']))
    for q in ('q10', 'q50', 'q90'):
        V.raw(f'wi.{k}.{q}', date(w['tc_' + q]))
    for c in ('m', 'tc', 'rel_err', 'w'):
        P(f'wi.{k}.s.{c}', w['shares'][c], 0, pct=True)
    P(f'wi.{k}.cip', w['ci_param'], 2)
    P(f'wi.{k}.cif', w['ci_full'], 2)

CI = N['ci']
for k in ('ndx', 'ssec', 'bet', 'btc17'):
    c = CI[k]
    P(f'ci.{k}.maxp', c['max_param'], 2)
    V.raw(f'ci.{k}.maxpd', date(c['max_param_d']))
    P(f'ci.{k}.maxf', c['max_full'], 2)
    V.raw(f'ci.{k}.maxfd', date(c['max_full_d']))
    V.raw(f'ci.{k}.first', date(c['first_alarm']) if c['first_alarm'] else '--')

EV = N['eval']
for k in ('sp500', 'btc'):
    e = EV[k]
    P(f'ev.{k}.base', e['hit']['base'], 0, pct=True)
    for c in ('0.1', '0.2', '0.3'):
        h = e['hit'][c]
        P(f'ev.{k}.hit.{c}', h['hit'] if h['hit'] == h['hit'] else 0, 0, pct=True)
        P(f'ev.{k}.share.{c}', h['share'], 0, pct=True)
        hf = e['hit_full'][c]
        P(f'ev.{k}.hitf.{c}', hf['hit'] if hf['hit'] == hf['hit'] else 0, 0, pct=True)
        P(f'ev.{k}.sharef.{c}', hf['share'], 0, pct=True)
    V.raw(f'ev.{k}.nep', str(e['n_ep']))
    V.raw(f'ev.{k}.nepc', str(e['n_ep_crash']))
    V.raw(f'ev.{k}.nepk', str(e['n_ep_known']))
    V.raw(f'ev.{k}.nfall', str(len(e['falls'])))
    V.raw(f'ev.{k}.nfa', str(sum(f['alarm_before'] for f in e['falls'])))
    yrs_ = [f['peak'][:4] for f in e['falls'] if f['alarm_before']]
    V.raw(f'ev.{k}.fay', V2(', '.join(yrs_[:-1]) + ' and ' + yrs_[-1], ', '.join(yrs_[:-1]) + ' și ' + yrs_[-1]) if len(yrs_) > 1 else ''.join(yrs_))
    V.raw(f'ev.{k}.start', month(e['start']))
    V.raw(f'ev.{k}.end', month(e['end']))
    V.raw(f'ev.{k}.n', str(e['hit']['n']))

def add_de(values):
    """RO: „de” after integer values whose last two digits are >= 20 or 00 (@{key.de}); nothing in EN."""
    for k, v in list(values.items()):
        d = re.sub(r'\\,|[⁅⁆]', '', str(v))
        if re.fullmatch(r'\d+', d):
            x = int(d)
            values[k + '.de'] = V2('', ' de') if (x >= 20 and (x % 100 >= 20 or x % 100 == 0)) else ''



add_de(V)

# =============================================================================
# 0. MOTIVAȚIE
# =============================================================================
D.section('Motivation', 'Motivație')

D.frame(T('The question of the chapter', 'Întrebarea capitolului'), items(
    (T('\\textbf{Question}: can a speculative bubble be recognised in a price series before it bursts?', '\\textbf{Întrebarea}: poate fi recunoscută o bulă speculativă într-o serie de prețuri înainte să se spargă?'),
     [T('a \\textbf{speculative bubble}: a price that rises far above the value justified by the expected cash flows of the asset, sustained by the expectation of selling it later at a higher price',
        'o \\textbf{bulă speculativă}: un preț care crește mult peste valoarea justificată de fluxurile de numerar așteptate ale activului, susținut de așteptarea de a-l vinde mai tîrziu la un preț mai mare'),
      T('a \\textbf{crash}: a large fall of the price in a short time (here: at least 20\\% from the peak)', 'un \\textbf{crah}: o scădere mare a prețului într-un timp scurt (aici: cel puțin 20\\% față de maximum)')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('bubbles in history and the statistical signatures of a bubble', 'bulele din istorie și semnăturile statistice ale unei bule'),
      T('rational bubbles and explosive roots: the right-tailed unit-root tests of Phillips, Shi and Yu (link with Chapter 3)', 'bulele raționale și rădăcinile explozive: testele de rădăcină unitară la dreapta ale lui Phillips, Shi și Yu (legătura cu Capitolul 3)'),
      T('the LPPL model of Johansen, Ledoit and Sornette: super-exponential growth, log-periodic oscillations, the critical time', 'modelul LPPL al lui Johansen, Ledoit și Sornette: creștere superexponențială, oscilații log-periodice, timpul critic'),
      T('estimation, confidence from many windows and an honest evaluation of crash prediction', 'estimarea, încrederea obținută din multe ferestre și o evaluare riguroasă a prognozei crahurilor')])))

D.frame(T('Self-study guide', 'Ghid de studiu individual'), items(
    (T('This chapter is for \\textbf{self-study}', 'Acest capitol este pentru \\textbf{studiu individual}'),
         [T('there is no seminar', 'nu are seminar'),
          T('each section ends with a recap', 'fiecare secțiune se încheie cu o recapitulare'),
          T('read a section, then redo its worked example on paper before you look at the solution', 'citiți o secțiune, apoi refaceți pe hîrtie exemplul rezolvat, înainte să vă uitați la soluție'),
          T('after each chart, read the interpretation slide and compare it with your own reading of the chart', 'după fiecare grafic, citiți slide-ul de interpretare și comparați-l cu propria lectură a graficului')]),
    (T('Run the lecture notebook: every chart and number of the slides is produced there', 'Rulați notebook-ul cursului: fiecare grafic și fiecare cifră din slide-uri sînt produse acolo'),
     [T('change one setting (the window, the filter, the threshold) and see how the conclusion changes', 'modificați o setare (fereastra, filtrul, pragul) și urmăriți cum se schimbă concluzia')]),
    T('Check yourself with the self-assessment at the end and with the quiz of Chapter 13 on the course site', 'Verificați-vă cu autoevaluarea de la final și cu quiz-ul Capitolului 13 de pe site-ul cursului'),
    T('Prerequisites: log returns and log prices (Chapter 1), the Dickey--Fuller test (Chapter 3), OLS regression', 'Cunoștințe necesare: randamentele și prețurile logaritmice (Capitolul 1), testul Dickey--Fuller (Capitolul 3), regresia OLS')))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Describe a bubble by its fundamental value and by its statistical signatures', 'Descrieți o bulă prin valoarea fundamentală și prin semnăturile ei statistice'),
    T('Explain why a rational bubble must grow at an explosive rate and test for explosive roots with SADF and GSADF', 'Explicați de ce o bulă rațională trebuie să crească exploziv și testați rădăcinile explozive cu SADF și GSADF'),
    T('Write the LPPL equation and interpret each of its seven parameters', 'Scrieți ecuația LPPL și interpretați fiecare dintre cei șapte parametri'),
    T('Estimate LPPL with the two-step method of Filimonov and Sornette and check the filter conditions', 'Estimați LPPL cu metoda în doi pași a lui Filimonov și Sornette și verificați condițiile de filtrare'),
    T('Build a confidence indicator from many windows and evaluate its alarms honestly, with false alarms and base rates', 'Construiți un indicator de încredere din multe ferestre și evaluați riguros alarmele lui, cu alarme false și frecvențe de bază')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook background: \\refHP\\ (unit roots and nonlinear least squares, the tools of this chapter)', 'Manual: \\refHP\\ (rădăcini unitare și cele mai mici pătrate neliniare, instrumentele acestui capitol)'),
     [T('the LPPL model: \\refJLS; \\refSornette; calibration: \\refFS', 'modelul LPPL: \\refJLS; \\refSornette; calibrarea: \\refFS'),
      T('explosive roots: \\refPWY; \\refPSYa; survey: \\refPS', 'rădăcini explozive: \\refPWY; \\refPSYa; sinteză: \\refPS'),
      T('history of bubbles: \\refKA; \\refShiller; \\refGarber', 'istoria bulelor: \\refKA; \\refShiller; \\refGarber')]),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_13}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_13}'),
     [T('the SADF/GSADF statistics and the LPPL calibration are written out in the code (no special package)', 'statisticile SADF/GSADF și calibrarea LPPL sînt scrise explicit în cod (fără un pachet special)')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter13_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter13_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Data: daily closes of the S\\&P 500, Nasdaq 100, BET, Shanghai Composite and Bitcoin (EODHD)', 'Date: închiderile zilnice ale indicilor S\\&P 500, Nasdaq 100, BET, Shanghai Composite și ale Bitcoin (EODHD)')))

# =============================================================================
# 1. BULE ÎN ISTORIE
# =============================================================================
D.section('Bubbles in history', 'Bule în istorie')

D.frame(T('Four centuries of manias', 'Patru secole de manii'), two(
    ph('tulip', T('Flora\'s wagon of fools, c.~1637', 'Carul nebunilor al Florei, c.~1637'), h='0.36\\textheight'),
    items((T('\\textbf{Tulip mania}, Netherlands, 1636--1637', '\\textbf{Mania lalelelor}, Olanda, 1636--1637'),
               [T('contracts on rare bulbs traded at extreme prices, then collapsed in February 1637', 'contracte pe bulbi rari tranzacționate la prețuri extreme, apoi prăbușite în februarie 1637'),
                T('\\refGarber: part of the story is a myth; rare bulbs were expensive for good reasons, common bulbs show the mania', '\\refGarber: o parte din poveste este mit; bulbii rari erau scumpi din motive întemeiate, bulbii obișnuiți arată mania')]),
          T('\\textbf{South Sea} (1720), \\textbf{Wall Street} (1929), \\textbf{Japan} (1989), \\textbf{dot-com} (2000), \\textbf{housing} (2007), \\textbf{crypto} (2017, 2021)', '\\textbf{South Sea} (1720), \\textbf{Wall Street} (1929), \\textbf{Japonia} (1989), \\textbf{dot-com} (2000), \\textbf{imobiliare} (2007), \\textbf{cripto} (2017, 2021)'),
          T('The common pattern \\refKA: a new story, easy credit, rising prices that attract new buyers, then panic', 'Tiparul comun \\refKA: o poveste nouă, credit ieftin, prețuri în creștere care atrag noi cumpărători, apoi panică'))),
    size='footnotesize')

D.frame(T('1929 and 1987', '1929 și 1987'), two(
    ph('nyse', T('Crowd outside the New York Stock Exchange, 29 October 1929', 'Mulțime în fața Bursei din New York, 29 octombrie 1929'), h='0.25\\textheight')
    + '\n\\par\\vspace{2mm}\n'
    + ph('amsterdam', T('Amsterdam Stock Exchange, 21 October 1987', 'Bursa din Amsterdam, 21 octombrie 1987'), h='0.16\\textheight'),
    items((T('\\textbf{October 1929}', '\\textbf{Octombrie 1929}'),
               [T('the end of the boom of the 1920s and the start of the Great Depression', 'sfîrșitul boom-ului anilor 1920 și începutul Marii Crize')]),
          (T('\\textbf{19 October 1987} (Black Monday)', '\\textbf{19 octombrie 1987} (Black Monday)'),
               [T('the largest one-day fall of the Dow Jones index, without any major news that day', 'cea mai mare scădere într-o singură zi a indicelui Dow Jones, fără vreo știre majoră în acea zi'),
                T('\\refSJB\\ found accelerating, log-periodic oscillations in the years before 1987; this observation started the LPPL literature', '\\refSJB\\ au găsit oscilații log-periodice, tot mai rapide, în anii dinaintea lui 1987; această observație a pornit literatura LPPL'),
                T('the daily data of this course start in 1990: we study 1987 through the literature and later bubbles with our own data', 'datele zilnice ale cursului încep în 1990: studiem 1987 prin literatură, iar bulele de mai tîrziu cu datele noastre')]),
          T('A crash without news suggests an \\textbf{endogenous} cause: the market itself became fragile', 'Un crah fără știri sugerează o cauză \\textbf{endogenă}: piața însăși devenise fragilă'))),
    size='footnotesize')

D.frame(T('Fundamental value and bubble', 'Valoarea fundamentală și bula'), items(
    (T('\\textbf{Fundamental value} $F_t$', '\\textbf{Valoarea fundamentală} $F_t$'),
         [T('the present value of the expected future cash flows (dividends $D$), discounted at the rate $r$', 'valoarea actualizată a fluxurilor de numerar viitoare așteptate (dividendele $D$), actualizate cu rata $r$'),
          T('$F_t = \\sum_{k=1}^{\\infty} \\dfrac{E_t[D_{t+k}]}{(1+r)^k}$', '$F_t = \\sum_{k=1}^{\\infty} \\dfrac{E_t[D_{t+k}]}{(1+r)^k}$'),
          T('$D_{t+k}$: the dividend paid $k$ periods from now; $E_t[\\cdot]$: the expectation given the information available at $t$; $r$: the discount rate per period', '$D_{t+k}$: dividendul plătit peste $k$ perioade; $E_t[\\cdot]$: media condiționată de informația disponibilă la momentul $t$; $r$: rata de actualizare pe perioadă')]),
    (T('\\textbf{Bubble component}', '\\textbf{Componenta de bulă}'),
         [T('$B_t = P_t - F_t$, the part of the price $P_t$ not explained by fundamentals', '$B_t = P_t - F_t$, partea din prețul $P_t$ neexplicată de fundamente'),
          T('$F_t$ is not observed: we can only test the \\textbf{implications} of a bubble in the price path', '$F_t$ nu este observată: putem testa doar \\textbf{implicațiile} unei bule asupra traiectoriei prețului')]),
    (T('Two views', 'Două puncte de vedere'),
     [T('efficient markets: prices reflect information; ``bubbles\'\' are visible only with hindsight \\refFama', 'piețe eficiente: prețurile reflectă informația; „bulele” se văd doar retrospectiv \\refFama'),
      T('behavioural finance: herding and extrapolation push prices away from value \\refShiller; large run-ups do raise the probability of a crash \\refGSY', 'finanțe comportamentale: comportamentul de turmă și extrapolarea îndepărtează prețurile de valoare \\refShiller; creșterile mari ridică probabilitatea unui crah \\refGSY')])), size='footnotesize')

chart(T('Six run-ups and crashes in the course data', 'Șase creșteri și crahuri în datele cursului'), 'tsa_ch13_episodes', 'TSA_ch13_episodes', [
    T('Daily closes, log scale, from six months before the low of the run-up to eighteen months after the peak; shaded: the run-up from the low to the peak',
      'Închideri zilnice, scară logaritmică, de la șase luni înainte de minimul creșterii pînă la optsprezece luni după maximum; zona colorată: creșterea de la minim la maximum')],
    h='0.66\\textheight')

D.frame(T('Interpreting the six episodes', 'Interpretarea celor șase episoade'), '\\setlength{\\tabcolsep}{4pt}' + table(
    'lllrrr', T('\\textbf{Episode}', '\\textbf{Episodul}') + ' & ' + T('\\textbf{Low}', '\\textbf{Minim}') + ' & ' + T('\\textbf{Peak}', '\\textbf{Maxim}')
    + ' & ' + T('\\textbf{Rise}', '\\textbf{Creștere}') + ' & ' + T('\\textbf{Growth p.a.}', '\\textbf{Ritm anual}') + ' & ' + T('\\textbf{Fall in 1 year}', '\\textbf{Scădere în 1 an}'),
    [f'{T(*LAB[k])} & @{{ep.{k}.low}} & @{{ep.{k}.peak}} & @{{ep.{k}.runup}}\\% & @{{ep.{k}.growth}}\\% & @{{ep.{k}.fall}}\\%' for k in LAB],
    size='scriptsize') + items(
    T('Growth p.a.: $\\ln(P_{\\text{peak}}/P_{\\text{low}})$ divided by the length of the run-up in years; fall: the lowest close in the year after the peak, relative to the peak',
      'Ritmul anual: $\\ln(P_{\\text{maxim}}/P_{\\text{minim}})$ împărțit la durata creșterii în ani; scăderea: cea mai mică închidere din anul de după maximum, față de maximum'),
    T('Bitcoin multiplied its price @{ep.btc17.mult} times in about a year; the S\\&P 500 rose much less: not every peak is a bubble of the same strength',
      'Bitcoin și-a înmulțit prețul de @{ep.btc17.mult} ori în aproximativ un an; S\\&P 500 a crescut mult mai puțin: nu orice maximum este o bulă de aceeași intensitate')),
    size='footnotesize')

D.recap(('Bubbles in history', 'bule în istorie'), [
    T('A bubble is a price above the fundamental value $F_t$; $F_t$ is not observed', 'O bulă este un preț peste valoarea fundamentală $F_t$; $F_t$ nu este observată'),
    T('Six episodes in our data: run-ups of @{ep.sp500.runup}\\% to @{ep.btc17.runup}\\%, followed by falls of @{ep.sp500.fall}\\% to @{ep.btc17.fall}\\% within a year',
      'Șase episoade în datele noastre: creșteri între @{ep.sp500.runup}\\% și @{ep.btc17.runup}\\%, urmate de scăderi între @{ep.sp500.fall}\\% și @{ep.btc17.fall}\\% într-un an'),
    T('Signatures: explosive and super-exponential growth, accelerating oscillations, then an end', 'Semnături: creștere explozivă și superexponențială, oscilații tot mai rapide, apoi un sfîrșit')])

# =============================================================================
# 2. BULE RAȚIONALE ȘI RĂDĂCINI EXPLOZIVE
# =============================================================================
D.section('Rational bubbles and explosive roots', 'Bule raționale și rădăcini explozive')

D.frame(T('Why a rational bubble must explode', 'Bula rațională și explozia ei inevitabilă'), items(
    (T('No-arbitrage pricing: $P_t = \\dfrac{E_t[P_{t+1} + D_{t+1}]}{1+r}$; the fundamental value $F_t$ is one solution', 'Evaluarea fără arbitraj: $P_t = \\dfrac{E_t[P_{t+1} + D_{t+1}]}{1+r}$; valoarea fundamentală $F_t$ este o soluție'),
     [T('in words: today\'s price is the discounted expected value of tomorrow\'s price plus tomorrow\'s dividend', 'în cuvinte: prețul de azi este valoarea actualizată a sumei așteptate dintre prețul și dividendul de mîine'),
      T('every $P_t = F_t + B_t$ with $E_t[B_{t+1}] = (1+r)\\,B_t$ is also a solution', 'orice $P_t = F_t + B_t$ cu $E_t[B_{t+1}] = (1+r)\\,B_t$ este tot o soluție')]),
    (T('The bubble must grow, in expectation, at the rate $r$: $E_t[B_{t+k}] = (1+r)^k B_t$', 'Bula trebuie să crească, în medie, cu rata $r$: $E_t[B_{t+k}] = (1+r)^k B_t$'),
     [T('an investor holds an asset above its value only if the overvaluation is expected to grow', 'un investitor deține un activ peste valoarea lui doar dacă se așteaptă ca supraevaluarea să crească'),
      T('$1+r > 1$: the bubble component is an \\textbf{explosive} process', '$1+r > 1$: componenta de bulă este un proces \\textbf{exploziv}')]),
    T('Consequence: if dividends are $I(1)$ (Chapter 3), the price is $I(1)$ without a bubble and explosive with one', 'Consecință: dacă dividendele sînt $I(1)$ (Capitolul 3), prețul este $I(1)$ fără bulă și exploziv cu bulă')), size='footnotesize')

D.frame(T('Worked example: a bubble that can burst', 'Exemplu rezolvat: o bulă care se poate sparge'), items(
    (T('\\refBW: each period the bubble survives with probability $\\pi$ and then grows to $\\frac{1+r}{\\pi}B_t$; otherwise it bursts ($B_{t+1} \\approx 0$)', '\\refBW: în fiecare perioadă bula supraviețuiește cu probabilitatea $\\pi$ și crește atunci la $\\frac{1+r}{\\pi}B_t$; altfel se sparge ($B_{t+1} \\approx 0$)'),
     [T('check: $E_t[B_{t+1}] = \\pi \\cdot \\frac{1+r}{\\pi} B_t + (1-\\pi) \\cdot 0 = (1+r)B_t$', 'verificare: $E_t[B_{t+1}] = \\pi \\cdot \\frac{1+r}{\\pi} B_t + (1-\\pi) \\cdot 0 = (1+r)B_t$')]),
    (T('With $r = 2\\%$ and $\\pi = 0.98$ per period:', 'Cu $r = 2\\%$ și $\\pi = 0{,}98$ pe perioadă:'),
     [T('while it survives the bubble grows by $1.02/0.98 - 1 = @{bw.g}\\%$ per period, more than $r$: the extra growth pays for the risk of a burst', 'cît supraviețuiește, bula crește cu $1{,}02/0{,}98 - 1 = @{bw.g}\\%$ pe perioadă, mai mult decît $r$: creșterea suplimentară plătește riscul spargerii'),
      T('expected lifetime $1/(1-\\pi) = @{bw.dur}$ periods; it survives more than @{bw.half} periods with probability $1/2$ ($0.98^{k} = 0.5$)', 'durata medie de viață $1/(1-\\pi) = @{bw.dur}$@{bw.dur.de} perioade; supraviețuiește peste @{bw.half}@{bw.half.de} perioade cu probabilitatea $1/2$ ($0{,}98^{k} = 0{,}5$)')]),
    T('The longer the bubble lasts, the faster it must grow and the larger the fall when it bursts', 'Cu cît bula durează mai mult, cu atît trebuie să crească mai repede și cu atît este mai mare căderea la spargere')))

chart(T('A simulated rational bubble and three AR(1) roots', 'O bulă rațională simulată și trei rădăcini AR(1)'), 'tsa_ch13_rational', 'TSA_ch13_rational_bubble', [
    T('Left: fundamental value (random walk) plus a Blanchard--Watson bubble ($r = 2\\%$, $\\pi = 0.98$), 400 periods; right: $y_t = \\rho y_{t-1} + \\varepsilon_t$ with the same shocks and $\\rho = 0.9$, $1$, $1.02$',
      'Stînga: valoarea fundamentală (mers aleator) plus o bulă Blanchard--Watson ($r = 2\\%$, $\\pi = 0{,}98$), 400 de perioade; dreapta: $y_t = \\rho y_{t-1} + \\varepsilon_t$ cu aceleași șocuri și $\\rho = 0{,}9$, $1$, $1{,}02$')],
    h='0.6\\textheight')

interp(('the simulation', 'simulării'), [
    (T('The bubble grows, bursts, restarts: in this path it reaches @{ra.max} and bursts @{ra.bursts} times after a large size', 'Bula crește, se sparge, reîncepe: pe această traiectorie atinge @{ra.max} și se sparge de @{ra.bursts} ori după ce a devenit mare'),
     [T('the price inherits the explosive episodes; between them it follows the fundamental value', 'prețul preia episoadele explozive; între ele urmează valoarea fundamentală')]),
    (T('AR(1) roots (Chapter 3): $\\rho = 0.9$ reverts to the mean, $\\rho = 1$ wanders (unit root), $\\rho = 1.02$ explodes', 'Rădăcinile AR(1) (Capitolul 3): $\\rho = 0{,}9$ revine la medie, $\\rho = 1$ rătăcește (rădăcină unitară), $\\rho = 1{,}02$ explodează'),
     [T('with $\\rho = 1.02$ a shock doubles in about @{ra.g35} periods and is multiplied by @{ra.g100} after 100', 'cu $\\rho = 1{,}02$ un șoc se dublează în aproximativ @{ra.g35}@{ra.g35.de} perioade și se înmulțește cu @{ra.g100} după 100')]),
    T('Idea of the tests: an explosive root ($\\rho > 1$) in the log price during a run-up is the statistical trace of a rational bubble', 'Ideea testelor: o rădăcină explozivă ($\\rho > 1$) în prețul logaritmic în timpul unei creșteri este urma statistică a unei bule raționale')])

D.frame(T('Right-tailed unit-root tests', 'Teste de rădăcină unitară la dreapta'), items(
    (T('The regression of Chapter 3 on a window of the log price $y_t = \\ln P_t$: $\\Delta y_t = a + \\delta\\, y_{t-1} + \\varepsilon_t$, with $\\delta = \\rho - 1$', 'Regresia din Capitolul 3 pe o fereastră a prețului logaritmic $y_t = \\ln P_t$: $\\Delta y_t = a + \\delta\\, y_{t-1} + \\varepsilon_t$, cu $\\delta = \\rho - 1$'),
     [T('Chapter 3 (\\refDF): $H_0$: $\\delta = 0$ against $H_1$: $\\delta < 0$ (stationary), the \\textbf{left} tail', 'Capitolul 3 (\\refDF): $H_0$: $\\delta = 0$ față de $H_1$: $\\delta < 0$ (staționar), coada din \\textbf{stînga}'),
      T('bubbles: $H_0$: $\\delta = 0$ (unit root) against $H_1$: $\\delta > 0$ (explosive), the \\textbf{right} tail: reject for large $t$-statistics', 'bule: $H_0$: $\\delta = 0$ (rădăcină unitară) față de $H_1$: $\\delta > 0$ (exploziv), coada din \\textbf{dreapta}: respingem pentru statistici $t$ mari')]),
    (T('Problem: a bubble occupies only a part of the sample, and the crash after it looks like mean reversion', 'Problema: o bulă ocupă doar o parte din eșantion, iar crahul de după ea seamănă cu revenirea la medie'),
     [T('one ADF on the whole sample has almost no power: we must compute the test on many \\textbf{windows}', 'un singur test ADF pe tot eșantionul nu are aproape nicio putere: trebuie să calculăm testul pe multe \\textbf{ferestre}')]),
    (T('Notation', 'Notațiile'),
     [T('$\\Delta y_t = y_t - y_{t-1}$; $a$: the intercept; $\\varepsilon_t$: the error; $\\rho$: the AR(1) coefficient of $y_t$', '$\\Delta y_t = y_t - y_{t-1}$; $a$: termenul liber; $\\varepsilon_t$: eroarea; $\\rho$: coeficientul AR(1) al lui $y_t$'),
      T('$r_1, r_2 \\in [0,1]$: fractions of the sample; $\\mathrm{ADF}_{r_1}^{r_2}$: the $t$-statistic of $\\hat\\delta$ on the window from $r_1T$ to $r_2T$; $r_0$: the smallest window', '$r_1, r_2 \\in [0,1]$: fracțiuni din eșantion; $\\mathrm{ADF}_{r_1}^{r_2}$: statistica $t$ a lui $\\hat\\delta$ pe fereastra de la $r_1T$ la $r_2T$; $r_0$: cea mai mică fereastră')])), size='footnotesize')

D.frame(T('SADF, GSADF and BSADF (1/2)', 'SADF, GSADF și BSADF (1/2)'), items(
    (T('$\\sup$: the largest value of the ADF statistic over all the windows considered', '$\\sup$: cea mai mare valoare a statisticii ADF pe toate ferestrele considerate'),
     [T('a single large value, in any window, is enough to signal an explosive episode', 'o singură valoare mare, în oricare fereastră, este suficientă pentru a semnala un episod exploziv')]),
    (T('\\textbf{SADF} \\refPWY', '\\textbf{SADF} \\refPWY'),
         [T('start fixed at 0, the end $r_2$ moves forward', 'începutul fixat la 0, sfîrșitul $r_2$ înaintează'),
          T('$\\mathrm{SADF} = \\sup_{r_2 \\in [r_0, 1]} \\mathrm{ADF}_{0}^{r_2}$', '$\\mathrm{SADF} = \\sup_{r_2 \\in [r_0, 1]} \\mathrm{ADF}_{0}^{r_2}$'),
          T('good for one bubble; after a first crash it loses power', 'bun pentru o singură bulă; după un prim crah își pierde puterea')]),
    (T('\\textbf{GSADF} \\refPSYa', '\\textbf{GSADF} \\refPSYa'),
         [T('both the start $r_1$ and the end $r_2$ move', 'se mișcă și începutul $r_1$, și sfîrșitul $r_2$'),
          T('$\\mathrm{GSADF} = \\sup_{r_2 \\in [r_0,1]}\\ \\sup_{r_1 \\in [0, r_2 - r_0]} \\mathrm{ADF}_{r_1}^{r_2}$', '$\\mathrm{GSADF} = \\sup_{r_2 \\in [r_0,1]}\\ \\sup_{r_1 \\in [0, r_2 - r_0]} \\mathrm{ADF}_{r_1}^{r_2}$'),
          T('detects several bubbles in one sample; smallest window $r_0 = 0.01 + 1.8/\\sqrt{T}$', 'detectează mai multe bule într-un eșantion; fereastra minimă $r_0 = 0{,}01 + 1{,}8/\\sqrt{T}$')])))

D.frame(T('SADF, GSADF and BSADF (2/2)', 'SADF, GSADF și BSADF (2/2)'), items(
    (T('\\textbf{BSADF} (date-stamping)', '\\textbf{BSADF} (datarea)'),
         [T('one statistic for each date $r_2$, using only data up to $r_2$', 'o statistică pentru fiecare dată $r_2$, folosind doar datele pînă la $r_2$'),
          T('$\\mathrm{BSADF}(r_2) = \\sup_{r_1} \\mathrm{ADF}_{r_1}^{r_2}$: the largest statistic over all starts $r_1$ of windows ending at $r_2$', '$\\mathrm{BSADF}(r_2) = \\sup_{r_1} \\mathrm{ADF}_{r_1}^{r_2}$: cea mai mare statistică pe toate începuturile $r_1$ ale ferestrelor care se încheie la $r_2$'),
          T('an episode starts when BSADF crosses its critical value and lasts at least $\\log T$ observations \\refPSYb ($\\log$: the natural logarithm)', 'un episod începe cînd BSADF depășește valoarea critică și durează cel puțin $\\log T$ observații \\refPSYb ($\\log$: logaritmul natural)')]),
    (T('The distributions are not standard: critical values by Monte Carlo', 'Distribuțiile nu sînt standard: valorile critice se obțin prin Monte Carlo'),
     [T('simulated under a random walk with a weak drift, $y_t = T^{-1} + y_{t-1} + \\varepsilon_t$', 'simulate sub un mers aleator cu drift slab, $y_t = T^{-1} + y_{t-1} + \\varepsilon_t$'),
      T('the drift $T^{-1}$ is very small: a unit root, not an explosive root, under $H_0$', 'driftul $T^{-1}$ este foarte mic: sub $H_0$ există o rădăcină unitară, nu una explozivă')]),
    T('Reading: an ADF, SADF or GSADF statistic above its 95\\% critical value rejects the unit root in favour of an explosive root', 'Interpretarea: o statistică ADF, SADF sau GSADF peste valoarea critică de 95\\% respinge rădăcina unitară în favoarea unei rădăcini explozive')))

D.frame(T('Worked example: one window of the Nasdaq 100', 'Exemplu rezolvat: o fereastră din Nasdaq 100'), items(
    (T('Weekly log closes of the Nasdaq 100, $T = @{pn.T}$ weeks; $r_0 = @{pn.r0}$, i.e.\\ a smallest window of @{pn.w0} weeks', 'Închideri săptămînale logaritmice ale Nasdaq 100, $T = @{pn.T}$@{pn.T.de} săptămîni; $r_0 = @{pn.r0}$, adică o fereastră minimă de @{pn.w0}@{pn.w0.de} săptămîni'),
     [T('end of the window: @{xe.end}, the week of the peak; among all start points the largest statistic comes from the start @{xe.start}', 'sfîrșitul ferestrei: @{xe.end}, săptămîna maximului; dintre toate punctele de început, cea mai mare statistică provine din începutul @{xe.start}')]),
    (T('OLS on this window ($n = @{xe.n}$ weekly changes): $\\widehat{\\Delta y_t} = @{xe.a} + @{xe.delta}\\, y_{t-1}$', 'OLS pe această fereastră ($n = @{xe.n}$@{xe.n.de} variații săptămînale): $\\widehat{\\Delta y_t} = @{xe.a} + @{xe.delta}\\, y_{t-1}$'),
     [T('$t = \\hat\\delta / \\mathrm{SE}(\\hat\\delta) = @{xe.delta} / @{xe.se} = @{xe.t}$', '$t = \\hat\\delta / \\mathrm{SE}(\\hat\\delta) = @{xe.delta} / @{xe.se} = @{xe.t}$'),
      T('so BSADF at the peak is @{xe.t}, above its 95\\% critical value of @{xe.cv}', 'deci BSADF la maximum este @{xe.t}, peste valoarea critică de 95\\%, egală cu @{xe.cv}')]),
    T('Reading: $\\hat\\delta > 0$ means $\\hat\\rho = 1 + \\hat\\delta > 1$: the higher the price, the larger the next weekly rise, the mark of an explosive phase', 'Interpretarea: $\\hat\\delta > 0$ înseamnă $\\hat\\rho = 1 + \\hat\\delta > 1$: cu cît prețul este mai mare, cu atît creșterea săptămînală următoare este mai mare, semnul unei faze explozive')))

chart(T('Explosive episodes of the Nasdaq 100, 1990--2004', 'Episoadele explozive ale Nasdaq 100, 1990--2004'), 'tsa_ch13_psy_ndx', 'TSA_ch13_explosive_roots', [
    T('Top: weekly close, log scale; bottom: BSADF and its 95\\% critical values (Monte Carlo, 1000 random walks); shaded: dated episodes (at least @{pn.minlen} weeks above the critical value)',
      'Sus: închiderea săptămînală, scară logaritmică; jos: BSADF și valorile critice de 95\\% (Monte Carlo, 1000 de mersuri aleatoare); zone colorate: episoadele datate (cel puțin @{pn.minlen} săptămîni peste valoarea critică)')],
    h='0.62\\textheight')

interp(('the Nasdaq 100 tests', 'testelor pentru Nasdaq 100'), [
    (T('Whole-sample ADF $@{pn.adf}$ (95\\% critical value $@{pn.adf95}$): no evidence', 'ADF pe tot eșantionul $@{pn.adf}$ (valoarea critică de 95\\%: $@{pn.adf95}$): nicio dovadă'),
     [T('SADF @{pn.sadf} (critical value @{pn.sadf95}) and GSADF @{pn.gsadf} (critical value @{pn.gsadf95}): explosive behaviour', 'SADF @{pn.sadf} (valoarea critică @{pn.sadf95}) și GSADF @{pn.gsadf} (valoarea critică @{pn.gsadf95}): comportament exploziv'),
      T('the crash of 2000--2002 hides the bubble from a single test on the whole sample', 'crahul din 2000--2002 ascunde bula de un test unic pe tot eșantionul')]),
    (T('BSADF dates @{pn.nep} episodes; the longest runs from @{pn.ep.a} to @{pn.ep.b} (@{pn.ep.n} weeks)', 'BSADF datează @{pn.nep} episoade; cel mai lung ține de la @{pn.ep.a} pînă la @{pn.ep.b} (@{pn.ep.n}@{pn.ep.n.de} săptămîni)'),
     [T('short explosive bursts already from @{pn.ep0.a}: the 1990s boom came in waves, as \\refPWY\\ found', 'episoade explozive scurte încă din @{pn.ep0.a}: boom-ul anilor 1990 a venit în valuri, cum au găsit \\refPWY')]),
    T('The end date is late: the statistic stays high for months after the peak; the test dates exuberance, it does not announce the crash', 'Data de sfîrșit este tîrzie: statistica rămîne mare luni de zile după maximum; testul datează exuberanța, nu anunță crahul')])

chart(T('Bitcoin, BET and Shanghai: dated episodes', 'Bitcoin, BET și Shanghai: episoade datate'), 'tsa_ch13_psy_panel', 'TSA_ch13_explosive_roots', [
    T('Weekly log closes; top: price with the dated episodes; bottom: BSADF and its 95\\% critical values; Bitcoin 2014--2023, BET 2000--2012, Shanghai Composite 2010--2018',
      'Închideri săptămînale logaritmice; sus: prețul cu episoadele datate; jos: BSADF și valorile critice de 95\\%; Bitcoin 2014--2023, BET 2000--2012, Shanghai Composite 2010--2018')],
    h='0.62\\textheight')

interp(('the three markets', 'celor trei piețe'), [
    (T('GSADF: Bitcoin @{ps.btc.gsadf}, BET @{ps.bet.gsadf}, Shanghai @{ps.ssec.gsadf}; all above their 95\\% critical values (about @{ps.btc.cv})', 'GSADF: Bitcoin @{ps.btc.gsadf}, BET @{ps.bet.gsadf}, Shanghai @{ps.ssec.gsadf}; toate peste valorile critice de 95\\% (aproximativ @{ps.btc.cv})'),
     [T('Bitcoin: @{ps.btc.e0.a} -- @{ps.btc.e0.b} and @{ps.btc.e1.a} -- @{ps.btc.e1.b}; Shanghai: @{ps.ssec.e0.a} -- @{ps.ssec.e0.b}', 'Bitcoin: @{ps.btc.e0.a} -- @{ps.btc.e0.b} și @{ps.btc.e1.a} -- @{ps.btc.e1.b}; Shanghai: @{ps.ssec.e0.a} -- @{ps.ssec.e0.b}')]),
    (T('BET: the explosive phase is @{ps.bet.e1.a} -- @{ps.bet.e1.b}, not the run-up to July 2007, which was strong but not explosive', 'BET: faza explozivă este @{ps.bet.e1.a} -- @{ps.bet.e1.b}, nu creșterea pînă în iulie 2007, care a fost puternică, dar nu explozivă'),
     [T('the BET flag of @{ps.bet.e2.a} -- @{ps.bet.e2.b} is the collapse: an \\textbf{accelerating fall} also gives $\\hat\\delta > 0$; always read the dates against the price chart', 'semnalul BET din @{ps.bet.e2.a} -- @{ps.bet.e2.b} este prăbușirea: o \\textbf{scădere care se accelerează} dă tot $\\hat\\delta > 0$; citiți întotdeauna datele alături de graficul prețului')]),
    T('The Bitcoin 2017 episode lasts until @{ps.btc.e0.b}, again well after the peak', 'Episodul Bitcoin din 2017 ține pînă în @{ps.btc.e0.b}, din nou mult după maximum')])

D.frame(T('Limits of the explosive-root tests', 'Limitele testelor de rădăcini explozive'), items(
    (T('\\textbf{Fundamentals}', '\\textbf{Fundamentele}'),
         [T('an explosive price is a bubble only if the fundamentals are not explosive', 'un preț exploziv este bulă doar dacă fundamentele nu sînt explozive'),
          T('better: test the price--dividend ratio (\\refPSYa\\ for the S\\&P 500); for Bitcoin there are no dividends at all', 'mai bine: testați raportul preț--dividend (\\refPSYa\\ pentru S\\&P 500); pentru Bitcoin nu există deloc dividende')]),
    (T('\\textbf{Volatility}', '\\textbf{Volatilitatea}'),
         [T('changing volatility distorts the size of the test', 'volatilitatea variabilă distorsionează nivelul testului'),
          T('a wild bootstrap gives robust critical values \\refPS', 'un wild bootstrap dă valori critice robuste \\refPS')]),
    (T('\\textbf{Timing}', '\\textbf{Momentul}'),
         [T('BSADF uses only past data, so it can run in real time', 'BSADF folosește doar date trecute, deci poate rula în timp real'),
          T('but it signals a bubble that is already under way and says nothing about when it ends', 'dar semnalează o bulă deja în desfășurare și nu spune nimic despre momentul în care se termină'),
          T('the LPPL model of the next section tries to answer exactly this: when?', 'modelul LPPL din secțiunea următoare încearcă să răspundă exact la această întrebare: cînd?')])))

D.recap(('Rational bubbles and explosive roots', 'bule raționale și rădăcini explozive'), [
    T('A rational bubble grows in expectation at the rate $r$: $E_t[B_{t+1}] = (1+r)B_t$, an explosive process', 'O bulă rațională crește în medie cu rata $r$: $E_t[B_{t+1}] = (1+r)B_t$, un proces exploziv'),
    T('Right-tailed ADF: $H_1$: $\\delta > 0$; SADF and GSADF take the supremum over windows; BSADF dates the episodes', 'ADF la dreapta: $H_1$: $\\delta > 0$; SADF și GSADF iau supremumul pe ferestre; BSADF datează episoadele'),
    T('Nasdaq 100, Bitcoin, BET and Shanghai all show explosive episodes; the tests date the run-up, not the crash', 'Nasdaq 100, Bitcoin, BET și Shanghai au episoade explozive; testele datează creșterea, nu crahul')])

# =============================================================================
# 3. MODELUL LPPL
# =============================================================================
D.section('The LPPL model', 'Modelul LPPL')

D.frame(T('Crashes as critical points', 'Crahurile ca puncte critice'), two(
    ph('sornette', T('Didier Sornette (born 1957)', 'Didier Sornette (n.~1957)'), h='0.36\\textheight'),
    items((T('Physicist, ETH Zürich; with Anders Johansen and Olivier Ledoit he proposed the \\textbf{JLS model} \\refJLS', 'Fizician, ETH Zürich; împreună cu Anders Johansen și Olivier Ledoit a propus \\textbf{modelul JLS} \\refJLS'),
           []),
          (T('Idea: traders imitate their neighbours; imitation creates positive feedback', 'Ideea: investitorii își imită vecinii; imitația creează o reacție pozitivă (positive feedback)'),
           [T('rising prices attract buyers, who push prices higher: growth accelerates', 'prețurile în creștere atrag cumpărători, care urcă prețurile și mai mult: creșterea se accelerează'),
            T('the market becomes more and more fragile, like a physical system near a \\textbf{critical point} (water near boiling)', 'piața devine tot mai fragilă, ca un sistem fizic aproape de un \\textbf{punct critic} (apa aproape de fierbere)')]),
          (T('The \\textbf{critical time} $t_c$', '\\textbf{Timpul critic} $t_c$'),
               [T('the most probable moment of the end of the bubble', 'momentul cel mai probabil al sfîrșitului bulei'),
                T('a crash is likely near $t_c$ but not certain: the bubble may also end with a slow decline', 'un crah este probabil aproape de $t_c$, dar nu sigur: bula se poate încheia și cu o scădere lentă')]))),
    size='footnotesize')

D.frame(T('Super-exponential growth', 'Creșterea superexponențială'), items(
    (T('\\textbf{Exponential} growth', 'Creșterea \\textbf{exponențială}'),
         [T('constant growth rate, $\\ln P(t) = a + g\\,t$', 'ritm de creștere constant, $\\ln P(t) = a + g\\,t$'),
          T('$d \\ln P/dt = g$', '$d \\ln P/dt = g$'),
          T('$a$: the initial log price; $g$: the constant growth rate; $d\\ln P/dt$: the growth rate of the price at time $t$', '$a$: prețul logaritmic inițial; $g$: ritmul constant de creștere; $d\\ln P/dt$: ritmul de creștere al prețului la momentul $t$'),
          T('a straight line on a log scale', 'o dreaptă pe scară logaritmică')]),
    (T('\\textbf{Super-exponential} growth', 'Creșterea \\textbf{superexponențială}'),
         [T('the growth rate increases with time', 'ritmul de creștere crește în timp'),
          T('on a log scale the curve bends upward', 'pe scară logaritmică curba se îndoaie în sus'),
          T('power-law singularity: $\\ln P(t) = A + B\\,(t_c - t)^m$, with $B < 0$ and $0 < m < 1$', 'singularitate de tip lege de putere: $\\ln P(t) = A + B\\,(t_c - t)^m$, cu $B < 0$ și $0 < m < 1$'),
          T('$t_c$: the critical time; $t_c - t$: the time left until it; $A$: the log price reached at $t_c$; $B$: the size of the growth; $m$: the exponent', '$t_c$: timpul critic; $t_c - t$: timpul rămas pînă la el; $A$: prețul logaritmic atins la $t_c$; $B$: mărimea creșterii; $m$: exponentul'),
          T('growth rate $d \\ln P/dt = -Bm\\,(t_c - t)^{m-1} \\to \\infty$ as $t \\to t_c$, while $\\ln P(t_c) = A$ stays finite', 'ritmul de creștere $d \\ln P/dt = -Bm\\,(t_c - t)^{m-1} \\to \\infty$ cînd $t \\to t_c$, în timp ce $\\ln P(t_c) = A$ rămîne finit')]),
    T('Such growth cannot last: it must end at or before $t_c$, a \\textbf{finite-time singularity}', 'O astfel de creștere nu poate dura: trebuie să se încheie la $t_c$ sau înainte, o \\textbf{singularitate în timp finit}')), size='footnotesize')

chart(T('Exponential and super-exponential growth', 'Creștere exponențială și creștere superexponențială'), 'tsa_ch13_growth', 'TSA_ch13_lppl_model', [
    T('Left: log price; right: its growth rate $d \\ln P/dt$; exponential ($g = 0.8$), power-law singularity ($A = 1$, $B = -1$, $m = 0.5$, $t_c = 1$) and the same with log-periodic oscillations',
      'Stînga: prețul logaritmic; dreapta: ritmul lui de creștere $d \\ln P/dt$; exponențială ($g = 0{,}8$), singularitate de tip lege de putere ($A = 1$, $B = -1$, $m = 0{,}5$, $t_c = 1$) și aceeași cu oscilații log-periodice')],
    h='0.58\\textheight')

interp(('the growth paths', 'traiectoriilor de creștere'), [
    T('All three start with similar growth rates: early in a bubble the paths are hard to tell apart', 'Toate trei încep cu ritmuri de creștere asemănătoare: la începutul unei bule traiectoriile se disting greu'),
    T('The exponential rate stays at 0.8; the singular rate rises from @{gr.start} to @{gr.mid} halfway and to @{gr.end} just before $t_c$', 'Ritmul exponențial rămîne 0,8; ritmul singular crește de la @{gr.start} la @{gr.mid} la jumătatea drumului și la @{gr.end} chiar înainte de $t_c$'),
    T('The oscillations ride on the accelerating trend and become faster near $t_c$: the next slides explain why', 'Oscilațiile se suprapun peste trendul accelerat și devin mai rapide aproape de $t_c$: slide-urile următoare explică de ce')])

D.frame(T('From a crash hazard to the LPPL equation (1/2)', 'De la riscul de crah la ecuația LPPL (1/2)'), items(
    (T('JLS: the relative price change before the crash has three parts', 'JLS: variația relativă a prețului înainte de crah are trei componente'),
     [T('$dP/P = \\mu(t)\\,dt + \\sigma\\,dW - \\kappa\\, dj$', '$dP/P = \\mu(t)\\,dt + \\sigma\\,dW - \\kappa\\, dj$'),
      T('$\\mu(t)\\,dt$: the expected growth over a short interval $dt$; $\\mu(t)$: the drift, which may change in time', '$\\mu(t)\\,dt$: creșterea așteptată pe un interval scurt $dt$; $\\mu(t)$: driftul, care se poate schimba în timp'),
      T('$\\sigma\\,dW$: the ordinary noise; $W$: a Brownian motion (a continuous random walk); $\\sigma$: the volatility', '$\\sigma\\,dW$: zgomotul obișnuit; $W$: o mișcare browniană (un mers aleator în timp continuu); $\\sigma$: volatilitatea'),
      T('$\\kappa\\, dj$: the crash; $j$ jumps from 0 to 1 at the crash, which removes a fraction $\\kappa$ of the price', '$\\kappa\\, dj$: crahul; $j$ sare de la 0 la 1 la crah, care elimină o fracțiune $\\kappa$ din preț')]),
    (T('\\textbf{Hazard rate} $h(t)$', '\\textbf{Rata de hazard} $h(t)$'),
         [T('the probability per unit of time that the crash happens now, given that it has not happened yet', 'probabilitatea pe unitatea de timp ca crahul să aibă loc acum, știind că nu a avut loc pînă acum')])), size='footnotesize')

D.frame(T('From a crash hazard to the LPPL equation (2/2)', 'De la riscul de crah la ecuația LPPL (2/2)'), items(
    (T('No arbitrage: $E[dP] = 0$ $\\Rightarrow$ $\\mu(t) = \\kappa\\, h(t)$', 'Fără arbitraj: $E[dP] = 0$ $\\Rightarrow$ $\\mu(t) = \\kappa\\, h(t)$'),
     [T('$E[dj] = h(t)\\,dt$, so the expected change is zero only if the drift pays for the expected crash loss', '$E[dj] = h(t)\\,dt$, deci variația așteptată este zero doar dacă driftul compensează pierderea așteptată din crah'),
      T('the higher the hazard, the faster the price must grow: risk and growth increase together', 'cu cît hazardul este mai mare, cu atît prețul trebuie să crească mai repede: riscul și creșterea cresc împreună')]),
    (T('Imitation near a critical point gives a hazard that rises towards $t_c$ with oscillations', 'Imitația aproape de un punct critic dă un hazard care crește spre $t_c$, cu oscilații'),
     [T('$h(t) \\approx \\alpha (t_c - t)^{m-1}[1 + \\beta\\cos(\\omega \\ln(t_c - t) - \\phi\')]$', '$h(t) \\approx \\alpha (t_c - t)^{m-1}[1 + \\beta\\cos(\\omega \\ln(t_c - t) - \\phi\')]$'),
      T('$\\alpha > 0$: the level of the hazard; $(t_c - t)^{m-1}$ grows without bound as $t \\to t_c$, since $m < 1$', '$\\alpha > 0$: nivelul hazardului; $(t_c - t)^{m-1}$ crește nelimitat cînd $t \\to t_c$, deoarece $m < 1$'),
      T('$\\beta$: the relative size of the oscillations; $\\omega$: their log-frequency; $\\phi\'$: their phase', '$\\beta$: mărimea relativă a oscilațiilor; $\\omega$: frecvența lor logaritmică; $\\phi\'$: faza lor')]),
    T('Integrating $\\mu(t) = \\kappa h(t)$ over time gives the log price on the next slide', 'Integrînd $\\mu(t) = \\kappa h(t)$ în timp obținem prețul logaritmic de pe slide-ul următor')), size='footnotesize')

D.frame(T('The LPPL equation', 'Ecuația LPPL'), items(
    (T('\\textbf{Log-periodic power law} (LPPL), for $t < t_c$', '\\textbf{Legea de putere log-periodică} (LPPL), pentru $t < t_c$'),
         [T('$\\quad \\ln P(t) = A + B\\,(t_c - t)^m + C\\,(t_c - t)^m \\cos\\bigl(\\omega \\ln(t_c - t) - \\phi\\bigr)$', '$\\quad \\ln P(t) = A + B\\,(t_c - t)^m + C\\,(t_c - t)^m \\cos\\bigl(\\omega \\ln(t_c - t) - \\phi\\bigr)$')])) + table(
    'lll', T('\\textbf{Parameter}', '\\textbf{Parametrul}') + ' & ' + T('\\textbf{Meaning}', '\\textbf{Semnificația}') + ' & ' + T('\\textbf{Usual range}', '\\textbf{Interval uzual}'),
    ['$t_c$ & ' + T('critical time: most probable end of the bubble', 'timpul critic: sfîrșitul cel mai probabil al bulei') + ' & ' + T('after the last observation', 'după ultima observație'),
     '$A$ & ' + T('log price at $t_c$ (if the bubble reached $t_c$)', 'prețul logaritmic la $t_c$ (dacă bula ar atinge $t_c$)') + ' & $A > 0$',
     '$B$ & ' + T('size of the power-law growth', 'mărimea creșterii de tip lege de putere') + ' & ' + T('$B < 0$ (rising price)', '$B < 0$ (preț în creștere)'),
     '$m$ & ' + T('exponent: how fast the growth accelerates', 'exponentul: cît de repede se accelerează creșterea') + ' & $0 < m < 1$',
     '$C$ & ' + T('relative size of the oscillations', 'mărimea relativă a oscilațiilor') + ' & $|C| < |B|$',
     '$\\omega$ & ' + T('angular log-frequency of the oscillations', 'frecvența unghiulară logaritmică a oscilațiilor') + ' & $2 \\le \\omega \\le 25$',
     '$\\phi$ & ' + T('phase of the oscillations', 'faza oscilațiilor') + ' & $[0, 2\\pi)$'],
    size='scriptsize') + items(
    T('Seven parameters: three of them ($t_c$, $m$, $\\omega$) enter nonlinearly; $\\phi$ disappears in the estimation (Section 4)', 'Șapte parametri: trei dintre ei ($t_c$, $m$, $\\omega$) intră neliniar; $\\phi$ dispare la estimare (Secțiunea 4)')),
    size='footnotesize')

D.frame(T('Log-periodic oscillations', 'Oscilațiile log-periodice'), items(
    (T('$\\cos(\\omega \\ln(t_c - t) - \\phi)$ is periodic in $\\ln(t_c - t)$, not in $t$: one full cycle each time $\\ln(t_c - t)$ changes by $2\\pi/\\omega$', '$\\cos(\\omega \\ln(t_c - t) - \\phi)$ este periodic în $\\ln(t_c - t)$, nu în $t$: un ciclu complet de fiecare dată cînd $\\ln(t_c - t)$ se schimbă cu $2\\pi/\\omega$'),
     [T('successive peaks $t_n$ satisfy $\\dfrac{t_c - t_n}{t_c - t_{n+1}} = \\lambda = e^{2\\pi/\\omega}$: the time left to $t_c$ shrinks by the same factor at each cycle', 'maximele succesive $t_n$ verifică $\\dfrac{t_c - t_n}{t_c - t_{n+1}} = \\lambda = e^{2\\pi/\\omega}$: timpul rămas pînă la $t_c$ se micșorează cu același factor la fiecare ciclu')]),
    (T('\\textbf{Discrete scale invariance}', '\\textbf{Invarianța discretă la scală}'),
         [T('the pattern looks the same when time to $t_c$ is rescaled by $\\lambda$ (as in a hierarchy of traders, groups, institutions)', 'tiparul arată la fel cînd timpul pînă la $t_c$ se rescalează cu $\\lambda$ (ca într-o ierarhie de investitori, grupuri, instituții)')]),
    (T('Worked example: with $\\omega = 8$, $\\lambda = e^{2\\pi/8} = @{lam.8}$', 'Exemplu rezolvat: pentru $\\omega = 8$, $\\lambda = e^{2\\pi/8} = @{lam.8}$'),
     [T('if a correction happens 200 days before $t_c$, the next one comes about $200/@{lam.8} \\approx @{lam.d1}$ days before $t_c$, then about @{lam.d2}, then @{lam.d3}', 'dacă o corecție are loc cu 200 de zile înainte de $t_c$, următoarea vine cu aproximativ $200/@{lam.8} \\approx @{lam.d1}$@{lam.d1.de} zile înainte de $t_c$, apoi cu aproximativ @{lam.d2}, apoi @{lam.d3}'),
      T('$\\omega = 6.28$ gives $\\lambda = @{lam.6.28}$; $\\omega = 10$ gives $\\lambda = @{lam.10}$; the empirical literature reports $\\lambda$ near 2 \\refSornette', '$\\omega = 6{,}28$ dă $\\lambda = @{lam.6.28}$; $\\omega = 10$ dă $\\lambda = @{lam.10}$; literatura empirică raportează $\\lambda$ în jur de 2 \\refSornette')])))

chart(T('The pieces of the LPPL equation', 'Componentele ecuației LPPL'), 'tsa_ch13_lppl_components', 'TSA_ch13_lppl_model', [
    T('Left: $A + B(t_c - t)^m$ for $m = 0.3, 0.5, 0.8$; middle: the oscillation $(t_c - t)^m \\cos(\\omega \\ln(t_c - t))$ for $\\omega = 6$ and $10$; right: the full LPPL path; $t_c = 1$',
      'Stînga: $A + B(t_c - t)^m$ pentru $m = 0{,}3;\\ 0{,}5;\\ 0{,}8$; mijloc: oscilația $(t_c - t)^m \\cos(\\omega \\ln(t_c - t))$ pentru $\\omega = 6$ și $10$; dreapta: traiectoria LPPL completă; $t_c = 1$')],
    h='0.55\\textheight')

interp(('the components', 'componentelor'), [
    (T('A smaller $m$ concentrates the growth close to $t_c$: the curve is flat for a long time, then shoots up', 'Un $m$ mai mic concentrează creșterea aproape de $t_c$: curba este plată multă vreme, apoi urcă brusc'),
     [T('$m$ close to 1 is almost a straight line: hardly a bubble; $m$ close to 0 is a jump at $t_c$', '$m$ apropiat de 1 este aproape o dreaptă: cu greu o bulă; $m$ apropiat de 0 este un salt la $t_c$')]),
    (T('The oscillation term shrinks with $(t_c - t)^m$ and speeds up: the cycles get shorter by the factor $\\lambda$', 'Termenul oscilant se micșorează cu $(t_c - t)^m$ și se accelerează: ciclurile se scurtează cu factorul $\\lambda$'),
     [T('a larger $\\omega$ gives more cycles before $t_c$', 'un $\\omega$ mai mare dă mai multe cicluri înainte de $t_c$')]),
    T('Real run-ups (Section 4) show this mixture: an accelerating trend with corrections that come more and more often', 'Creșterile reale (Secțiunea 4) arată acest amestec: un trend accelerat cu corecții tot mai dese')])

D.frame(T('Worked example: an LPPL value by hand', 'Exemplu rezolvat: o valoare LPPL calculată de mînă'), items(
    (T('Parameters: $A = 8$, $B = -1.2$, $C = 0.08$, $m = 0.5$, $\\omega = 8$, $\\phi = 0$, $t_c = 2$ (years); compute $\\ln P(t)$', 'Parametri: $A = 8$, $B = -1{,}2$, $C = 0{,}08$, $m = 0{,}5$, $\\omega = 8$, $\\phi = 0$, $t_c = 2$ (ani); calculați $\\ln P(t)$'),
     [T('$t = 1$: $t_c - t = 1$, $(t_c - t)^m = 1$, $\\ln 1 = 0$, $\\cos 0 = 1$: $\\ln P = 8 - 1.2 + 0.08 = @{lp.a.lnp}$, $P = @{lp.a.p}$', '$t = 1$: $t_c - t = 1$, $(t_c - t)^m = 1$, $\\ln 1 = 0$, $\\cos 0 = 1$: $\\ln P = 8 - 1{,}2 + 0{,}08 = @{lp.a.lnp}$, $P = @{lp.a.p}$'),
      T('$t = 1.75$: $(0.25)^{0.5} = @{lp.b.fm}$, $\\ln 0.25 = @{lp.b.lg}$, $\\cos(8 \\cdot (@{lp.b.lg})) = @{lp.b.cos}$: $\\ln P = 8 - 1.2 \\cdot @{lp.b.fm} + 0.08 \\cdot @{lp.b.fm} \\cdot @{lp.b.cos} = @{lp.b.lnp}$, $P = @{lp.b.p}$',
        '$t = 1{,}75$: $(0{,}25)^{0{,}5} = @{lp.b.fm}$, $\\ln 0{,}25 = @{lp.b.lg}$, $\\cos(8 \\cdot (@{lp.b.lg})) = @{lp.b.cos}$: $\\ln P = 8 - 1{,}2 \\cdot @{lp.b.fm} + 0{,}08 \\cdot @{lp.b.fm} \\cdot @{lp.b.cos} = @{lp.b.lnp}$, $P = @{lp.b.p}$'),
      T('$t = 1.99$: $(0.01)^{0.5} = @{lp.c.fm}$, $\\cos(8 \\cdot (@{lp.c.lg})) = @{lp.c.cos}$: $\\ln P = @{lp.c.lnp}$, $P = @{lp.c.p}$', '$t = 1{,}99$: $(0{,}01)^{0{,}5} = @{lp.c.fm}$, $\\cos(8 \\cdot (@{lp.c.lg})) = @{lp.c.cos}$: $\\ln P = @{lp.c.lnp}$, $P = @{lp.c.p}$')]),
    T('The price rises from @{lp.a.p} to @{lp.c.p} as $t$ goes from 1 to 1.99 and would reach $e^A = @{lp.eA}$ at $t_c$; most of the rise comes in the last months', 'Prețul crește de la @{lp.a.p} la @{lp.c.p} cînd $t$ trece de la 1 la 1,99 și ar atinge $e^A = @{lp.eA}$ la $t_c$; cea mai mare parte a creșterii vine în ultimele luni')),
    size='footnotesize')

D.frame(T('Parameter constraints: the filter', 'Restricțiile parametrilor: filtrul'), items(
    (T('A fit is \\textbf{qualified} only if it describes a credible bubble (\\refSZ, following \\refShanghai):', 'O ajustare este \\textbf{calificată} doar dacă descrie o bulă credibilă (\\refSZ, după \\refShanghai):'),
     [T('$B < 0$: rising price; $0.01 \\le m \\le 0.99$: accelerating growth with a finite price at $t_c$', '$B < 0$: preț în creștere; $0{,}01 \\le m \\le 0{,}99$: creștere accelerată, cu preț finit la $t_c$'),
      T('$2 \\le \\omega \\le 25$: neither too slow nor too fast oscillations; at least 2.5 oscillations in the window, $\\frac{\\omega}{\\pi}\\ln\\frac{t_c - t_1}{t_c - t_2} \\ge 2.5$', '$2 \\le \\omega \\le 25$: oscilații nici prea lente, nici prea rapide; cel puțin 2,5 oscilații în fereastră, $\\frac{\\omega}{\\pi}\\ln\\frac{t_c - t_1}{t_c - t_2} \\ge 2{,}5$'),
      T('$t_2 \\le t_c \\le t_2 + 0.2\\,(t_2 - t_1)$: the end is close, relative to the length of the window $[t_1, t_2]$', '$t_2 \\le t_c \\le t_2 + 0{,}2\\,(t_2 - t_1)$: sfîrșitul este aproape, raportat la lungimea ferestrei $[t_1, t_2]$'),
      T('damping $\\dfrac{m|B|}{\\omega|C|} \\ge 1$: the hazard rate stays positive', 'amortizarea $\\dfrac{m|B|}{\\omega|C|} \\ge 1$: rata de hazard rămîne pozitivă')]),
    (T('Conditions on the residuals', 'Condiții asupra reziduurilor'),
     [T('every fitted price within 15\\% of the observed price; the Lomb periodogram \\refLomb\\ confirms the log-periodic cycle (level 10\\%)', 'fiecare preț ajustat la cel mult 15\\% de prețul observat; periodograma Lomb \\refLomb\\ confirmă ciclul log-periodic (nivel 10\\%)'),
      T('Lomb periodogram: a periodogram for unevenly spaced points, here of the residuals as a function of $\\ln(t_c - t)$', 'periodograma Lomb: o periodogramă pentru puncte neuniform distanțate, aici a reziduurilor în funcție de $\\ln(t_c - t)$'),
      T('the residuals are stationary: the Dickey--Fuller and Phillips--Perron tests of Chapter 3 reject a unit root (level 10\\%)', 'reziduurile sînt staționare: testele Dickey--Fuller și Phillips--Perron din Capitolul 3 resping rădăcina unitară (nivel 10\\%)')])), size='footnotesize')

D.recap(('The LPPL model', 'modelul LPPL'), [
    T('Imitation creates a hazard rate that grows towards $t_c$; no arbitrage turns it into super-exponential growth', 'Imitația creează o rată de hazard care crește spre $t_c$; lipsa arbitrajului o transformă în creștere superexponențială'),
    T('$\\ln P(t) = A + B(t_c - t)^m + C(t_c - t)^m\\cos(\\omega\\ln(t_c - t) - \\phi)$, seven parameters', '$\\ln P(t) = A + B(t_c - t)^m + C(t_c - t)^m\\cos(\\omega\\ln(t_c - t) - \\phi)$, șapte parametri'),
    T('Log-periodic: cycles shrink by $\\lambda = e^{2\\pi/\\omega}$; a fit counts only if it passes the filter', 'Log-periodic: ciclurile se scurtează cu $\\lambda = e^{2\\pi/\\omega}$; o ajustare contează doar dacă trece de filtru')])

# =============================================================================
# 4. ESTIMAREA
# =============================================================================
D.section('Estimation', 'Estimarea')

D.frame(T('The two-step method of Filimonov and Sornette (1/2)', 'Metoda în doi pași a lui Filimonov și Sornette (1/2)'), items(
    (T('Least squares over seven parameters has many local minima; \\refFS\\ reduce the search to three parameters', 'Cele mai mici pătrate după șapte parametri au multe minime locale; \\refFS\\ reduc căutarea la trei parametri'),
     [T('$\\min \\mathrm{SSR} = \\sum_{t=t_1}^{t_2} [\\ln P_t - \\mathrm{LPPL}(t)]^2$; SSR: the sum of squared residuals over the window $[t_1, t_2]$', '$\\min \\mathrm{SSR} = \\sum_{t=t_1}^{t_2} [\\ln P_t - \\mathrm{LPPL}(t)]^2$; SSR: suma pătratelor reziduurilor pe fereastra $[t_1, t_2]$'),
      T('$\\mathrm{LPPL}(t)$: the right-hand side of the LPPL equation', '$\\mathrm{LPPL}(t)$: membrul drept al ecuației LPPL')]),
    (T('Expand the cosine, with $\\tau = t_c - t$', 'Desfacem cosinusul, cu $\\tau = t_c - t$'),
     [T('$C\\cos(\\omega\\ln\\tau - \\phi) = C_1\\cos(\\omega\\ln\\tau) + C_2\\sin(\\omega\\ln\\tau)$, where $C_1 = C\\cos\\phi$, $C_2 = C\\sin\\phi$', '$C\\cos(\\omega\\ln\\tau - \\phi) = C_1\\cos(\\omega\\ln\\tau) + C_2\\sin(\\omega\\ln\\tau)$, unde $C_1 = C\\cos\\phi$, $C_2 = C\\sin\\phi$')]),
    (T('The LPPL equation becomes linear in $A$, $B$, $C_1$, $C_2$', 'Ecuația LPPL devine liniară în $A$, $B$, $C_1$, $C_2$'),
     [T('$\\ln P(t) = A + B\\,f_t + C_1\\,g_t + C_2\\,h_t$', '$\\ln P(t) = A + B\\,f_t + C_1\\,g_t + C_2\\,h_t$'),
      T('$f_t = \\tau^m$, $g_t = \\tau^m\\cos(\\omega\\ln\\tau)$, $h_t = \\tau^m\\sin(\\omega\\ln\\tau)$: known numbers once $t_c$, $m$, $\\omega$ are fixed', '$f_t = \\tau^m$, $g_t = \\tau^m\\cos(\\omega\\ln\\tau)$, $h_t = \\tau^m\\sin(\\omega\\ln\\tau)$: numere cunoscute odată ce $t_c$, $m$, $\\omega$ sînt fixați')])), size='footnotesize')

D.frame(T('The two-step method of Filimonov and Sornette (2/2)', 'Metoda în doi pași a lui Filimonov și Sornette (2/2)'), items(
    (T('\\textbf{Step 1} (linear)', '\\textbf{Pasul 1} (liniar)'),
         [T('for given $(t_c, m, \\omega)$, regress $\\ln P_t$ on $(1, f_t, g_t, h_t)$ by OLS', 'pentru $(t_c, m, \\omega)$ dați, regresăm $\\ln P_t$ pe $(1, f_t, g_t, h_t)$ prin OLS'),
          T('$(\\hat A, \\hat B, \\hat C_1, \\hat C_2) = (X\'X)^{-1}X\'y$; $X$: the matrix with columns $1, f_t, g_t, h_t$; $y$: the vector of $\\ln P_t$', '$(\\hat A, \\hat B, \\hat C_1, \\hat C_2) = (X\'X)^{-1}X\'y$; $X$: matricea cu coloanele $1, f_t, g_t, h_t$; $y$: vectorul valorilor $\\ln P_t$'),
          T('this gives the concentrated cost $\\mathrm{SSR}(t_c, m, \\omega)$, a function of three parameters only', 'obținem astfel costul concentrat $\\mathrm{SSR}(t_c, m, \\omega)$, o funcție de doar trei parametri')]),
    (T('\\textbf{Step 2} (nonlinear)', '\\textbf{Pasul 2} (neliniar)'),
         [T('minimise $\\mathrm{SSR}(t_c, m, \\omega)$ over the search space', 'minimizăm $\\mathrm{SSR}(t_c, m, \\omega)$ în spațiul de căutare'),
          T('$t_c \\in [t_2, t_2 + (t_2 - t_1)/3]$, $m \\in [0, 1]$, $\\omega \\in [1, 50]$', '$t_c \\in [t_2, t_2 + (t_2 - t_1)/3]$, $m \\in [0, 1]$, $\\omega \\in [1, 50]$'),
          T('in the code: a grid of $8 \\times 8 \\times 16$ points, then the Nelder--Mead simplex from the best grid point', 'în cod: o grilă de $8 \\times 8 \\times 16$ puncte, apoi simplexul Nelder--Mead pornind din cel mai bun punct al grilei')]),
    (T('At the end: $C = \\sqrt{C_1^2 + C_2^2}$ and $\\phi = \\operatorname{atan2}(C_2, C_1)$', 'La final: $C = \\sqrt{C_1^2 + C_2^2}$ și $\\phi = \\operatorname{atan2}(C_2, C_1)$'),
     [T('$\\operatorname{atan2}(C_2, C_1)$: the angle of the point $(C_1, C_2)$, in $(-\\pi, \\pi]$', '$\\operatorname{atan2}(C_2, C_1)$: unghiul punctului $(C_1, C_2)$, în $(-\\pi, \\pi]$')])), size='footnotesize')

chart(T('The cost landscape', 'Peisajul funcției de cost'), 'tsa_ch13_cost', 'TSA_ch13_estimation', [
    (T('Nasdaq 100, window from the low of October 1998 to 30 days before the peak', 'Nasdaq 100, fereastra de la minimul din octombrie 1998 pînă la 30 de zile înainte de maximum'),
     [T('left: $\\log \\mathrm{SSR}$ over $(t_c, m)$, minimised over $\\omega \\in [2, 25]$ (@{cost.n} grid points, with the linear step solved at each point)', 'stînga: $\\log \\mathrm{SSR}$ după $(t_c, m)$, minimizat după $\\omega \\in [2, 25]$ (@{cost.n}@{cost.n.de} puncte de grilă, cu pasul liniar rezolvat în fiecare punct)'),
      T('right: the minimum over $m$ and $\\omega$ as a function of $t_c$', 'dreapta: minimul după $m$ și $\\omega$, ca funcție de $t_c$')])],
    h='0.56\\textheight')

interp(('the cost landscape', 'peisajului funcției de cost'), [
    (T('The profile over $t_c$ has @{cost.loc} local minima: an optimiser started at a different $t_c$ can end in a different valley', 'Profilul după $t_c$ are @{cost.loc} minime locale: un optimizator pornit de la alt $t_c$ poate ajunge în altă vale'),
     [T('the waves come from the oscillation term: shifting $t_c$ shifts the phase of the cycles', 'valurile provin din termenul oscilant: mutarea lui $t_c$ mută faza ciclurilor')]),
    (T('The landscape is flat: the worst and the best $t_c$ differ by only @{cost.ratio}\\% in SSR', 'Peisajul este plat: cel mai prost și cel mai bun $t_c$ diferă cu doar @{cost.ratio}\\% în SSR'),
     [T('the global minimum is at $m$ near 1 and $t_c$ = @{cost.tc}, far after the actual peak: the data barely identify $t_c$', 'minimul global este la $m$ aproape de 1 și $t_c$ = @{cost.tc}, mult după maximul real: datele abia identifică $t_c$')]),
    T('Lesson: one LPPL fit is never enough; we need many windows and the filter (Section 5)', 'Lecția: o singură ajustare LPPL nu este niciodată suficientă; avem nevoie de multe ferestre și de filtru (Secțiunea 5)')])

D.frame(T('Worked example: the Shanghai fit at the end of the window', 'Exemplu rezolvat: ajustarea pentru Shanghai la sfîrșitul ferestrei'), items(
    (T('Shanghai Composite, window @{ep.ssec.low} -- @{fi.ssec.t2} (@{fi.ssec.n} days); estimates: $t_c = @{fx.tc}$ (@{fi.ssec.tc}), $m = @{fx.m}$, $\\omega = @{fx.w}$', 'Shanghai Composite, fereastra @{ep.ssec.low} -- @{fi.ssec.t2} (@{fi.ssec.n}@{fi.ssec.n.de} zile); estimări: $t_c = @{fx.tc}$ (@{fi.ssec.tc}), $m = @{fx.m}$, $\\omega = @{fx.w}$'),
     [T('step 1 gives $\\hat A = @{fx.A}$, $\\hat B = @{fx.B}$, $\\hat C_1 = @{fx.C1}$, $\\hat C_2 = @{fx.C2}$, so $\\hat C = @{fx.C}$', 'pasul 1 dă $\\hat A = @{fx.A}$, $\\hat B = @{fx.B}$, $\\hat C_1 = @{fx.C1}$, $\\hat C_2 = @{fx.C2}$, deci $\\hat C = @{fx.C}$')]),
    (T('At $t_2 = @{fx.t2}$ (years): $\\tau = t_c - t_2 = @{fx.dt}$, $\\tau^m = @{fx.fm}$, $\\ln\\tau = @{fx.lg}$', 'La $t_2 = @{fx.t2}$ (ani): $\\tau = t_c - t_2 = @{fx.dt}$, $\\tau^m = @{fx.fm}$, $\\ln\\tau = @{fx.lg}$'),
     [T('$\\cos(\\omega\\ln\\tau) = @{fx.cos}$, $\\sin(\\omega\\ln\\tau) = @{fx.sin}$', '$\\cos(\\omega\\ln\\tau) = @{fx.cos}$, $\\sin(\\omega\\ln\\tau) = @{fx.sin}$'),
      T('$\\widehat{\\ln P} = @{fx.A} + (@{fx.Bfm}) + (@{fx.osc}) = @{fx.lnp_hat}$; observed $\\ln P = @{fx.lnp}$', '$\\widehat{\\ln P} = @{fx.A} + (@{fx.Bfm}) + (@{fx.osc}) = @{fx.lnp_hat}$; observat $\\ln P = @{fx.lnp}$')]),
    T('In levels: fitted @{fx.phat} against observed @{fx.p}, an error of @{fx.err}\\%; the three terms are the level $A$, the power-law growth and the oscillation', 'În niveluri: ajustat @{fx.phat} față de observat @{fx.p}, o eroare de @{fx.err}\\%; cei trei termeni sînt nivelul $A$, creșterea de tip lege de putere și oscilația')),
    size='footnotesize')

chart(T('LPPL fits 30 days before the peak', 'Ajustări LPPL cu 30 de zile înainte de maximum'), 'tsa_ch13_fits', 'TSA_ch13_estimation', [
    T('Window: from the low before the run-up to 30 calendar days before the peak; the fitted path is extended to the estimated $t_c$; data after the window are shown only for comparison',
      'Fereastra: de la minimul dinaintea creșterii pînă la 30 de zile calendaristice înainte de maximum; traiectoria ajustată este prelungită pînă la $t_c$ estimat; datele de după fereastră sînt arătate doar pentru comparație')],
    h='0.64\\textheight')

D.frame(T('Interpreting the six fits', 'Interpretarea celor șase ajustări'), table(
    'lllrrrll', T('\\textbf{Episode}', '\\textbf{Episodul}') + ' & $t_2$ & $\\hat t_c$ & ' + T('\\textbf{Days to peak}', '\\textbf{Zile față de maxim}')
    + ' & $\\hat m$ & $\\hat\\omega$ & ' + T('\\textbf{Param.}', '\\textbf{Param.}') + ' & ' + T('\\textbf{All}', '\\textbf{Toate}'),
    [f'{T(*LAB[k])} & @{{fi.{k}.t2}} & @{{fi.{k}.tc}} & @{{fi.{k}.err}} & @{{fi.{k}.m}} & @{{fi.{k}.w}} & @{{fi.{k}.qp}} & @{{fi.{k}.qf}}' for k in LAB],
    size='scriptsize') + items(
    T('Days to peak: $\\hat t_c$ minus the actual peak; Param.: the fit passes the parameter conditions; All: it also passes the residual conditions', 'Zile față de maxim: $\\hat t_c$ minus maximul real; Param.: ajustarea trece de condițiile asupra parametrilor; Toate: trece și de condițiile asupra reziduurilor'),
    T('Only Shanghai passes the whole filter, with $\\hat t_c$ within a week of the peak; S\\&P 500 and Nasdaq 100 hit the boundary $m = 1$: no super-exponential growth in these windows', 'Doar Shanghai trece de tot filtrul, cu $\\hat t_c$ la mai puțin de o săptămînă de maximum; S\\&P 500 și Nasdaq 100 ating limita $m = 1$: nicio creștere superexponențială în aceste ferestre'),
    T('Bitcoin fails the 15\\% error condition: its daily swings are too large for one smooth path', 'Bitcoin nu trece de condiția erorii de 15\\%: variațiile lui zilnice sînt prea mari pentru o singură traiectorie netedă')),
    size='footnotesize')

D.recap(('Estimation', 'estimarea'), [
    T('Four linear parameters by OLS, three nonlinear ones ($t_c$, $m$, $\\omega$) by search: the method of \\refFS', 'Patru parametri liniari prin OLS, trei neliniari ($t_c$, $m$, $\\omega$) prin căutare: metoda \\refFS'),
    T('The cost surface is rugged and flat in $t_c$: the critical time is weakly identified', 'Suprafața de cost este accidentată și plată după $t_c$: timpul critic este slab identificat'),
    T('One window, one fit: only one of six episodes gives a fully qualified fit 30 days before the peak', 'O fereastră, o ajustare: doar unul dintre cele șase episoade dă o ajustare complet calificată cu 30 de zile înainte de maximum')])

# =============================================================================
# 5. ÎNCREDERE DIN MAI MULTE FERESTRE
# =============================================================================
D.section('Confidence from many windows', 'Încrederea obținută din multe ferestre')

D.frame(T('The LPPLS confidence indicator', 'Indicatorul de încredere LPPLS'), items(
    (T('So far $t_1$ was the low before the run-up, a date known only \\textbf{after} the bubble; in real time the start is unknown', 'Pînă acum $t_1$ a fost minimul dinaintea creșterii, o dată cunoscută doar \\textbf{după} bulă; în timp real începutul nu este cunoscut'),
     [T('remedy: for each end date $t_2$ (today), fit LPPL on 29 windows $[t_2 - L, t_2]$, $L = 750, 725, \\dots, 50$ trading days \\refSZ', 'remediul: pentru fiecare dată de sfîrșit $t_2$ (azi), ajustăm LPPL pe 29 de ferestre $[t_2 - L, t_2]$, $L = 750, 725, \\dots, 50$ de zile de tranzacționare \\refSZ')]),
    (T('Indicator $=$ share of the windows ending at $t_2$ whose fit is qualified \\refShanghai; 0: no window sees a bubble, 1: all do', 'Indicatorul $=$ ponderea ferestrelor care se încheie la $t_2$ și au o ajustare calificată \\refShanghai; 0: nicio fereastră nu vede o bulă, 1: toate văd'),
     [T('LPPLS (LPPL singularity) is the name used for this indicator; the equation is the same', 'LPPLS (singularitatea LPPL) este numele folosit pentru acest indicator; ecuația este aceeași')]),
    (T('Two versions in this chapter', 'Două versiuni în acest capitol'),
     [T('\\textbf{parameter conditions}: $B$, $m$, $\\omega$, $t_c$, oscillations, damping', '\\textbf{condițiile asupra parametrilor}: $B$, $m$, $\\omega$, $t_c$, oscilații, amortizare'),
      T('\\textbf{all conditions}: also the 15\\% error, the Lomb test and stationary residuals (\\refSZ)', '\\textbf{toate condițiile}: și eroarea de 15\\%, testul Lomb și reziduurile staționare (\\refSZ)')]),
    T('Computed only with data up to $t_2$: a real-time indicator; an alarm when it exceeds a threshold fixed in advance (here 0.2)', 'Se calculează doar cu date pînă la $t_2$: un indicator în timp real; o alarmă cînd depășește un prag fixat dinainte (aici 0,2)')))

chart(T('Many windows, one end date', 'Multe ferestre, o singură dată de sfîrșit'), 'tsa_ch13_windows', 'TSA_ch13_confidence', [
    T('Estimated $t_c$ for each of the 29 windows ending 30 days before the peak; green dots: the fit passes the parameter conditions; red crosses: it fails them',
      '$t_c$ estimat pentru fiecare dintre cele 29 de ferestre care se încheie cu 30 de zile înainte de maximum; puncte verzi: ajustarea trece de condițiile asupra parametrilor; cruci roșii: nu trece')],
    h='0.6\\textheight')

interp(('the windows', 'ferestrelor'), [
    (T('Nasdaq 100 (end @{wi.ndx.t2}): @{wi.ndx.np} of @{wi.ndx.n} windows pass the parameter conditions, @{wi.ndx.nf} the whole filter', 'Nasdaq 100 (sfîrșit @{wi.ndx.t2}): @{wi.ndx.np} din @{wi.ndx.n} ferestre trec de condițiile asupra parametrilor, @{wi.ndx.nf} de tot filtrul'),
     [T('qualified $\\hat t_c$: median @{wi.ndx.q50}, 10\\%--90\\% range @{wi.ndx.q10} -- @{wi.ndx.q90}; actual peak @{wi.ndx.peak}', '$\\hat t_c$ calificat: mediana @{wi.ndx.q50}, intervalul 10\\%--90\\% @{wi.ndx.q10} -- @{wi.ndx.q90}; maximul real @{wi.ndx.peak}')]),
    (T('Shanghai (end @{wi.ssec.t2}): @{wi.ssec.np} of @{wi.ssec.n} pass the parameter conditions; median $\\hat t_c$ @{wi.ssec.q50}, actual peak @{wi.ssec.peak}', 'Shanghai (sfîrșit @{wi.ssec.t2}): @{wi.ssec.np} din @{wi.ssec.n} trec de condițiile asupra parametrilor; mediana $\\hat t_c$ @{wi.ssec.q50}, maximul real @{wi.ssec.peak}'),
     [T('the condition on $m$ fails most often: only @{wi.ssec.s.m}\\% of the Shanghai windows have $0.01 \\le m \\le 0.99$', 'condiția asupra lui $m$ cade cel mai des: doar @{wi.ssec.s.m}\\% dintre ferestrele Shanghai au $0{,}01 \\le m \\le 0{,}99$')]),
    T('Short windows put $t_c$ right after $t_2$; long windows spread it out: the window length is a hidden modelling choice', 'Ferestrele scurte pun $t_c$ imediat după $t_2$; ferestrele lungi îl împrăștie: lungimea ferestrei este o alegere de modelare ascunsă')])

chart(T('The indicator around four peaks (1/2)', 'Indicatorul în jurul a patru maxime (1/2)'), 'tsa_ch13_ci', 'TSA_ch13_confidence', [
    (T('Log price (left axis) and the indicator (right axis) every 5 trading days, from 18 months before to 4 months after the peak', 'Prețul logaritmic (axa din stînga) și indicatorul (axa din dreapta) la fiecare 5 zile de tranzacționare, de la 18 luni înainte pînă la 4 luni după maximum'),
     [T('green area: parameter conditions; purple line: all conditions', 'zona verde: condițiile asupra parametrilor; linia mov: toate condițiile')])],
    h='0.64\\textheight')

chart(T('The indicator around four peaks (2/2)', 'Indicatorul în jurul a patru maxime (2/2)'), 'tsa_ch13_ci_b', 'TSA_ch13_confidence', [
    T('The same layout and colours as in (1/2), for the BET (2007) and Bitcoin (2017); Bitcoin: one value every 7 days', 'Aceeași structură și aceleași culori ca în (1/2), pentru BET (2007) și Bitcoin (2017); Bitcoin: o valoare la fiecare 7 zile')],
    h='0.64\\textheight')

interp(('the indicator', 'indicatorului'), [
    (T('Nasdaq 100: the parameter version reaches @{ci.ndx.maxp} on @{ci.ndx.maxpd}, a year before the peak, and rises again in the last months; the full version stays below @{ci.ndx.maxf}', 'Nasdaq 100: versiunea pe parametri atinge @{ci.ndx.maxp} la @{ci.ndx.maxpd}, cu un an înainte de maximum, și crește din nou în ultimele luni; versiunea completă rămîne sub @{ci.ndx.maxf}'),
     [T('Shanghai: both versions rise in the two months before the peak (maximum @{ci.ssec.maxf} on @{ci.ssec.maxfd})', 'Shanghai: ambele versiuni cresc în cele două luni dinaintea maximului (maximum @{ci.ssec.maxf} la @{ci.ssec.maxfd})')]),
    (T('BET: the strongest full-filter signal, @{ci.bet.maxf}, comes on @{ci.bet.maxfd}, after the peak: the fits then describe the last part of the run-up, not the future', 'BET: cel mai puternic semnal cu filtrul complet, @{ci.bet.maxf}, vine la @{ci.bet.maxfd}, după maximum: ajustările descriu atunci ultima parte a creșterii, nu viitorul'),
     [T('Bitcoin 2017: the parameter version peaks at @{ci.btc17.maxp} on @{ci.btc17.maxpd}; the full version is almost always 0 (the 15\\% error condition)', 'Bitcoin 2017: versiunea pe parametri ajunge la @{ci.btc17.maxp} la @{ci.btc17.maxpd}; versiunea completă este aproape mereu 0 (condiția erorii de 15\\%)')]),
    T('The indicator does light up in bubbles, but early, late and with gaps; the choice of filter changes the picture', 'Indicatorul se aprinde în bule, dar devreme, tîrziu și cu întreruperi; alegerea filtrului schimbă imaginea')])

chart(T('The critical time moves with the data', 'Timpul critic se mută odată cu datele'), 'tsa_ch13_tc_path', 'TSA_ch13_confidence', [
    T('Window start fixed at the low before the run-up; the end $t_2$ moves from 8 months before to 1 month after the peak (weekly); each point: the $\\hat t_c$ of the fit ending at $t_2$',
      'Începutul ferestrei fixat la minimul dinaintea creșterii; sfîrșitul $t_2$ se mută de la 8 luni înainte pînă la o lună după maximum (săptămînal); fiecare punct: $\\hat t_c$ al ajustării care se încheie la $t_2$')],
    h='0.6\\textheight')

interp(('the moving critical time', 'timpului critic care se mută'), [
    (T('Nasdaq 100: over @{tp.ndx.n} end dates $\\hat t_c$ ranges from @{tp.ndx.min} to @{tp.ndx.max} (@{tp.ndx.range} days); none of these fits passes the parameter conditions', 'Nasdaq 100: pe @{tp.ndx.n}@{tp.ndx.n.de} date de sfîrșit, $\\hat t_c$ variază de la @{tp.ndx.min} la @{tp.ndx.max} (@{tp.ndx.range}@{tp.ndx.range.de} zile); niciuna dintre aceste ajustări nu trece de condițiile asupra parametrilor'),
     [T('Bitcoin 2017: range @{tp.btc17.min} -- @{tp.btc17.max}; @{tp.btc17.q}\\% of the fits pass; median error @{tp.btc17.err} days', 'Bitcoin 2017: intervalul @{tp.btc17.min} -- @{tp.btc17.max}; @{tp.btc17.q}\\% dintre ajustări trec; eroarea mediană @{tp.btc17.err}@{tp.btc17.err.de} zile')]),
    (T('Many points lie close to the line $t_c = t_2$: the fit often says ``the end is now\'\', week after week', 'Multe puncte stau aproape de dreapta $t_c = t_2$: ajustarea spune adesea „sfîrșitul este acum”, săptămînă după săptămînă'),
     [T('a forecast that keeps moving with the data is a description of the past, not a prediction', 'o prognoză care se tot mută odată cu datele este o descriere a trecutului, nu o predicție')]),
    T('Picking, after the crash, the one window whose $\\hat t_c$ was right is \\textbf{look-ahead bias}', 'Alegerea, după crah, a singurei ferestre al cărei $\\hat t_c$ a fost corect înseamnă \\textbf{look-ahead bias} (folosirea informației din viitor)')])

D.recap(('Confidence from many windows', 'încrederea obținută din multe ferestre'), [
    T('The start of a bubble is unknown: fit 29 windows for each end date', 'Începutul unei bule nu este cunoscut: ajustăm 29 de ferestre pentru fiecare dată de sfîrșit'),
    T('Indicator = share of qualified windows; it depends strongly on the filter', 'Indicatorul = ponderea ferestrelor calificate; depinde puternic de filtru'),
    T('$\\hat t_c$ moves with the end of the data: report its spread over windows, never one value', '$\\hat t_c$ se mută odată cu sfîrșitul datelor: raportați dispersia lui pe ferestre, niciodată o singură valoare')])

# =============================================================================
# 6. EVALUARE ONESTĂ
# =============================================================================
D.section('An honest evaluation of crash prediction', 'O evaluare riguroasă a prognozei crahurilor')

D.frame(T('Traps in evaluating crash predictions', 'Capcane în evaluarea prognozelor de crah'), items(
    (T('\\textbf{Look-ahead bias}', '\\textbf{Look-ahead bias}'),
         [T('using information from after $t_2$', 'folosirea informației de după $t_2$'),
          T('choosing $t_1$ at the low of the run-up, the episodes after the crash, or the filter after seeing the results', 'alegerea lui $t_1$ la minimul creșterii, a episoadelor după crah sau a filtrului după ce am văzut rezultatele')]),
    (T('\\textbf{Selection}', '\\textbf{Selecția}'),
         [T('studying only the bubbles that burst', 'studierea doar a bulelor care s-au spart'),
          T('the run-ups that did not end in a crash are forgotten', 'creșterile care nu s-au încheiat cu un crah sînt uitate')]),
    (T('\\textbf{False alarms and base rates}', '\\textbf{Alarmele false și frecvențele de bază}'),
         [T('an alarm is useful only if a crash is \\textbf{more likely after an alarm} than on an ordinary day', 'o alarmă este utilă doar dacă un crah este \\textbf{mai probabil după o alarmă} decît într-o zi obișnuită'),
          T('the comparison: $P(\\text{crash} \\mid \\text{alarm})$ against $P(\\text{crash})$, the unconditional frequency', 'comparația: $P(\\text{crah} \\mid \\text{alarmă})$ față de $P(\\text{crah})$, frecvența necondiționată')]),
    T('A fair test fixes everything in advance (windows, filter, threshold, the definition of a crash) and runs over the whole sample', 'Un test corect fixează totul dinainte (ferestre, filtru, prag, definiția crahului) și rulează pe tot eșantionul')))

chart(T('Alarms over the whole sample (1/2)', 'Alarmele pe tot eșantionul (1/2)'), 'tsa_ch13_eval', 'TSA_ch13_evaluation', [
    (T('S\\&P 500: indicator every 5 trading days, @{ev.sp500.start} -- @{ev.sp500.end}', 'S\\&P 500: indicatorul la fiecare 5 zile de tranzacționare, @{ev.sp500.start} -- @{ev.sp500.end}'),
     [T('a crash: a fall of at least 20\\% within 182 days after the alarm; right: hit rate by threshold', 'un crah: o scădere de cel puțin 20\\% în cele 182 de zile de după alarmă; dreapta: rata de reușită în funcție de prag')])],
    h='0.62\\textheight')

chart(T('Alarms over the whole sample (2/2)', 'Alarmele pe tot eșantionul (2/2)'), 'tsa_ch13_eval_btc', 'TSA_ch13_evaluation', [
    (T('Bitcoin: indicator every 7 days, @{ev.btc.start} -- @{ev.btc.end}', 'Bitcoin: indicatorul la fiecare 7 zile, @{ev.btc.start} -- @{ev.btc.end}'),
     [T('hit rate: the share of the alarm dates followed by a crash; dashed line: the unconditional frequency', 'rata de reușită: ponderea datelor cu alarmă urmate de un crah; linia punctată: frecvența necondiționată')])],
    h='0.62\\textheight')

interp(('the evaluation', 'evaluării'), [
    (T('S\\&P 500: a fall of 20\\% within six months follows @{ev.sp500.base}\\% of all dates, but @{ev.sp500.hit.0.2}\\% of the alarm dates (threshold 0.2, parameter conditions)', 'S\\&P 500: o scădere de 20\\% în șase luni urmează după @{ev.sp500.base}\\% dintre toate datele, dar după @{ev.sp500.hit.0.2}\\% dintre datele cu alarmă (pragul 0,2, condițiile asupra parametrilor)'),
     [T('@{ev.sp500.nep} alarm clusters, none followed by such a fall within six months; the alarms come in calm, steady uptrends', '@{ev.sp500.nep}@{ev.sp500.nep.de} grupuri de alarme, niciunul urmat de o astfel de scădere în șase luni; alarmele apar în creșteri calme și constante'),
      T('of the @{ev.sp500.nfall} drawdowns of 20\\% or more since @{ev.sp500.start}, @{ev.sp500.nfa} had an alarm in the six months before their peak (@{ev.sp500.fay}), but the falls came later', 'dintre cele @{ev.sp500.nfall} scăderi de 20\\% sau mai mult din @{ev.sp500.start}, @{ev.sp500.nfa} au avut o alarmă în cele șase luni dinaintea maximului (@{ev.sp500.fay}), dar scăderile au venit mai tîrziu')]),
    (T('Bitcoin: base rate @{ev.btc.base}\\%, after an alarm @{ev.btc.hit.0.2}\\%: almost no gain; @{ev.btc.nfa} of its @{ev.btc.nfall} drawdowns of 20\\% had an alarm before the peak', 'Bitcoin: frecvența de bază @{ev.btc.base}\\%, după o alarmă @{ev.btc.hit.0.2}\\%: aproape niciun cîștig; @{ev.btc.nfa} dintre cele @{ev.btc.nfall} scăderi de 20\\% au avut o alarmă înainte de maximum'),
     [T('falls of 20\\% are so frequent in Bitcoin that almost any alarm looks ``right\'\'; higher thresholds help a little, on very few dates (@{ev.btc.hit.0.3}\\% at 0.3, on @{ev.btc.share.0.3}\\% of the dates)', 'scăderile de 20\\% sînt atît de frecvente la Bitcoin încît aproape orice alarmă pare „corectă”; pragurile mai mari ajută puțin, pe foarte puține date (@{ev.btc.hit.0.3}\\% la 0,3, pe @{ev.btc.share.0.3}\\% dintre date)')]),
    T('The indicator says ``this looks like a bubble regime\'\', not ``the crash comes within six months\'\'', 'Indicatorul spune „aceasta seamănă cu un regim de bulă”, nu „crahul vine în următoarele șase luni”')])

D.frame(T('The literature on crash prediction', 'Literatura despre prognoza crahurilor'), items(
    (T('Successes reported by the authors of the method', 'Reușite raportate de autorii metodei'),
     [T('\\refJiang: the Chinese bubbles of 2007 and 2009 diagnosed in advance; \\refShanghai: the Shanghai 2015 peak announced before it happened', '\\refJiang: bulele chinezești din 2007 și 2009 diagnosticate dinainte; \\refShanghai: maximul Shanghai din 2015 anunțat înainte să aibă loc'),
      T('\\refGerlach: Bitcoin\'s bubbles of 2012--2018 dissected with the confidence indicator', '\\refGerlach: bulele Bitcoin din 2012--2018 analizate cu indicatorul de încredere')]),
    (T('Critical evaluations', 'Evaluări critice'),
     [T('\\refFeig: the log-periodic term adds little once the estimation error is taken into account', '\\refFeig: termenul log-periodic adaugă puțin odată ce se ține cont de eroarea de estimare'),
      T('\\refBJ: on many crashes, the LPPL conditions are rarely all met before the crash; parameters are unstable', '\\refBJ: pe multe crahuri, condițiile LPPL sînt rareori toate îndeplinite înainte de crah; parametrii sînt instabili')]),
    T('Our own numbers point the same way: the indicator reacts to strong run-ups, but its timing is loose and its alarms are often false', 'Propriile noastre cifre arată același lucru: indicatorul reacționează la creșterile puternice, dar momentul lui este imprecis și alarmele lui sînt adesea false')))

D.frame(T('Strengths and limits of LPPL', 'Posibilitățile și limitele modelului LPPL'), items(
    (T('\\textbf{Can}', '\\textbf{Poate}'),
         [T('describe a run-up with accelerating growth and shrinking corrections in a few interpretable parameters', 'descrie o creștere cu ritm accelerat și corecții tot mai scurte prin cîțiva parametri interpretabili'),
          T('measure ``how bubble-like\'\' a market is today, across many windows (the indicator)', 'măsura cît de mult seamănă azi o piață cu o bulă, pe multe ferestre (indicatorul)')]),
    (T('\\textbf{Cannot}', '\\textbf{Nu poate}'),
         [T('give the date of a crash: $t_c$ marks the end of the regime', 'da data unui crah: $t_c$ marchează sfîrșitul regimului'),
          T('guarantee a crash: the regime may end with a slow decline', 'garanta un crah: regimul se poate încheia și cu o scădere lentă'),
          T('tell whether the price is above the fundamental value: that needs fundamentals', 'spune dacă prețul este peste valoarea fundamentală: pentru aceasta sînt necesare fundamentele')]),
    T('Use it with the explosive-root tests, with fundamentals (valuation ratios) and with risk measures such as VaR 1\\% (Chapter 5), not alone', 'Folosiți-l împreună cu testele de rădăcini explozive, cu fundamentele (rapoarte de evaluare) și cu măsuri de risc precum VaR 1\\% (Capitolul 5), nu singur')))

D.recap(('An honest evaluation', 'o evaluare riguroasă'), [
    T('Fix windows, filter, threshold and the definition of a crash before looking at the outcomes', 'Fixați ferestrele, filtrul, pragul și definiția crahului înainte de a vă uita la rezultate'),
    T('Compare the hit rate after alarms with the base rate; count the false alarms', 'Comparați rata de reușită după alarme cu frecvența de bază; numărați alarmele false'),
    T('LPPL describes bubbles well; as a crash-timing tool its record is weak', 'LPPL descrie bine bulele; ca instrument de datare a crahurilor, rezultatele lui sînt slabe')])

# =============================================================================
# 7. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    (T('\\textbf{Code}', '\\textbf{Cod}'),
         [T('a first version of the BSADF recursion, of the Filimonov--Sornette two-step fit, of the indicator over many windows', 'o primă versiune a recursiei BSADF, a ajustării în doi pași Filimonov--Sornette, a indicatorului pe multe ferestre')]),
    (T('\\textbf{Explanation}', '\\textbf{Explicații}'),
         [T('a second explanation of the hazard-rate argument or of log-periodicity, with your own numbers', 'o a doua explicație a argumentului ratei de hazard sau a log-periodicității, cu cifrele dumneavoastră')]),
    (T('\\textbf{Exploration}', '\\textbf{Explorare}'),
         [T('the indicator on many markets and periods, with different filters, under a protocol fixed in advance', 'indicatorul pe multe piețe și perioade, cu filtre diferite, după un protocol fixat dinainte')]),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that fits the LPPL model to the daily log price of the Shanghai Composite between two dates with the Filimonov-Sornette method (OLS for A, B, C1, C2; grid search plus Nelder-Mead for tc, m, omega), checks the filter conditions of Shu and Zhu (2020) and plots the fit.}',
        '\\aiprompt{Write Python code that fits the LPPL model to the daily log price of the Shanghai Composite between two dates with the Filimonov-Sornette method (OLS for A, B, C1, C2; grid search plus Nelder-Mead for tc, m, omega), checks the filter conditions of Shu and Zhu (2020) and plots the fit.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('Simulate an LPPL path with known parameters plus noise and check that the code recovers $t_c$, $m$ and $\\omega$', 'Simulați o traiectorie LPPL cu parametri cunoscuți plus zgomot și verificați că codul regăsește $t_c$, $m$ și $\\omega$'),
    T('The sign conventions: $B < 0$ for a rising bubble; the test of explosive roots uses the right tail, not the left tail of Chapter 3', 'Convențiile de semn: $B < 0$ pentru o bulă în creștere; testul de rădăcini explozive folosește coada din dreapta, nu coada din stînga din Capitolul 3'),
    T('Critical values: SADF and GSADF need their own simulated critical values, not the Dickey--Fuller table', 'Valorile critice: SADF și GSADF au nevoie de propriile valori critice simulate, nu de tabelul Dickey--Fuller'),
    T('No look-ahead: every quantity at date $t_2$ must use only data up to $t_2$', 'Fără look-ahead: orice mărime la data $t_2$ trebuie să folosească doar date pînă la $t_2$'),
    T('A crash ``predicted\'\' by an AI answer after the fact is not a prediction; ask for the full list of alarms and false alarms', 'Un crah „prezis” de un răspuns AI după ce a avut loc nu este o predicție; cereți lista completă a alarmelor și a alarmelor false'),
    T('Every cited reference: it must exist; check the DOI', 'Fiecare referință citată: trebuie să existe; verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('A bubble is a price above its fundamental value; a rational bubble must grow explosively', 'O bulă este un preț peste valoarea fundamentală; o bulă rațională trebuie să crească exploziv'),
    T('Right-tailed ADF tests (SADF, GSADF, BSADF) detect and date explosive episodes with data up to each date', 'Testele ADF la dreapta (SADF, GSADF, BSADF) detectează și datează episoadele explozive cu datele disponibile la fiecare dată'),
    T('LPPL: super-exponential growth with log-periodic oscillations ending at a critical time $t_c$', 'LPPL: creștere superexponențială cu oscilații log-periodice, care se încheie la un timp critic $t_c$'),
    T('Estimate in two steps (OLS + search over $t_c$, $m$, $\\omega$), filter the fits, use many windows', 'Estimați în doi pași (OLS + căutare după $t_c$, $m$, $\\omega$), filtrați ajustările, folosiți multe ferestre'),
    T('Judge predictions by hit rates against base rates, with all the false alarms: the timing of crashes remains hard', 'Judecați prognozele prin rata de reușită față de frecvența de bază, cu toate alarmele false: datarea crahurilor rămîne dificilă')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.35}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Rational bubble', 'Bulă rațională') + ' & $P_t = F_t + B_t$, \\quad $E_t[B_{t+1}] = (1+r)B_t$',
     T('Right-tailed ADF', 'ADF la dreapta') + ' & $\\Delta y_t = a + \\delta y_{t-1} + \\varepsilon_t$, \\quad $H_1$: $\\delta > 0$',
     'GSADF, BSADF & $\\sup_{r_2}\\sup_{r_1}\\mathrm{ADF}_{r_1}^{r_2}$, \\quad $\\mathrm{BSADF}(r_2) = \\sup_{r_1}\\mathrm{ADF}_{r_1}^{r_2}$, \\quad $r_0 = 0.01 + 1.8/\\sqrt T$',
     'LPPL & $\\ln P(t) = A + B(t_c - t)^m + C(t_c - t)^m\\cos(\\omega\\ln(t_c - t) - \\phi)$',
     T('Linear form', 'Forma liniară') + ' & $\\ln P(t) = A + Bf_t + C_1g_t + C_2h_t$, \\quad $C = \\sqrt{C_1^2 + C_2^2}$',
     T('Scaling ratio', 'Raportul de scală') + ' & $\\lambda = e^{2\\pi/\\omega}$',
     T('Damping', 'Amortizarea') + ' & $m|B|/(\\omega|C|) \\ge 1$',
     T('Indicator', 'Indicatorul') + ' & ' + T('share of qualified windows ending at $t_2$', 'ponderea ferestrelor calificate care se încheie la $t_2$')],
    size='footnotesize') + '}')

D.frame(T('Self-assessment (1/2)', 'Autoevaluare (1/2)'), items(
    (T('\\textbf{Question}: in a Blanchard--Watson bubble with $r = 1\\%$ and $\\pi = 0.95$, by how much does the bubble grow in a period in which it survives?', '\\textbf{Întrebare}: într-o bulă Blanchard--Watson cu $r = 1\\%$ și $\\pi = 0{,}95$, cu cît crește bula într-o perioadă în care supraviețuiește?'),
     [T('\\textbf{Answer}: by $1.01/0.95 - 1 \\approx 6.3\\%$; expected lifetime $1/(1 - 0.95) = 20$ periods', '\\textbf{Răspuns}: cu $1{,}01/0{,}95 - 1 \\approx 6{,}3\\%$; durata medie de viață $1/(1 - 0{,}95) = 20$ de perioade')]),
    (T('\\textbf{Question}: a right-tailed ADF on a window gives $\\hat\\delta = 0.004$ with SE 0.002. Is the window explosive at 5\\% if the critical value is 1.5?', '\\textbf{Întrebare}: un ADF la dreapta pe o fereastră dă $\\hat\\delta = 0{,}004$ cu SE 0,002. Este fereastra explozivă la 5\\%, dacă valoarea critică este 1,5?'),
     [T('\\textbf{Answer}: yes: $t = 0.004/0.002 = 2 > 1.5$; we reject the unit root in favour of an explosive root', '\\textbf{Răspuns}: da: $t = 0{,}004/0{,}002 = 2 > 1{,}5$; respingem rădăcina unitară în favoarea unei rădăcini explozive')]),
    (T('\\textbf{Question}: why do we use the right tail and not the left tail of Chapter 3?', '\\textbf{Întrebare}: de ce folosim coada din dreapta și nu coada din stînga din Capitolul 3?'),
     [T('\\textbf{Answer}: the alternative is $\\rho > 1$ (explosive), not $\\rho < 1$ (stationary); evidence for it is a large positive $t$', '\\textbf{Răspuns}: alternativa este $\\rho > 1$ (exploziv), nu $\\rho < 1$ (staționar); dovada pentru ea este un $t$ pozitiv mare')])))

D.frame(T('Self-assessment (2/2)', 'Autoevaluare (2/2)'), items(
    (T('\\textbf{Question}: an LPPL fit gives $\\omega = 6.28$. By what factor does the time left to $t_c$ shrink between two corrections?', '\\textbf{Întrebare}: o ajustare LPPL dă $\\omega = 6{,}28$. Cu ce factor se micșorează timpul rămas pînă la $t_c$ între două corecții?'),
     [T('\\textbf{Answer}: $\\lambda = e^{2\\pi/6.28} \\approx e \\approx 2.72$', '\\textbf{Răspuns}: $\\lambda = e^{2\\pi/6{,}28} \\approx e \\approx 2{,}72$')]),
    (T('\\textbf{Question}: a fit has $m = 1.0$ at the edge of the search space. Is it a qualified bubble fit?', '\\textbf{Întrebare}: o ajustare are $m = 1{,}0$, la marginea spațiului de căutare. Este o ajustare calificată de bulă?'),
     [T('\\textbf{Answer}: no: the filter needs $m \\le 0.99$; $m = 1$ is a linear trend in $t$, not an accelerating one', '\\textbf{Răspuns}: nu: filtrul cere $m \\le 0{,}99$; $m = 1$ este un trend liniar în $t$, nu unul accelerat')]),
    (T('\\textbf{Question}: an indicator gives alarms before 8 of 10 crashes. Is it a good predictor?', '\\textbf{Întrebare}: un indicator dă alarme înainte de 8 din 10 crahuri. Este un predictor bun?'),
     [T('\\textbf{Answer}: not yet: we also need the false alarms and the base rate; if alarms are on almost every day, catching 8 crashes is easy', '\\textbf{Răspuns}: încă nu: avem nevoie și de alarmele false și de frecvența de bază; dacă alarmele apar aproape în fiecare zi, este ușor să prinzi 8 crahuri')])))

D.frame(T('Project idea', 'Idee de proiect'), items(
    (T('\\textbf{Is the BET in a bubble today?} A real-time protocol, fixed before looking at the result', '\\textbf{Este BET azi într-o bulă?} Un protocol în timp real, fixat înainte de a vedea rezultatul'),
     [T('BSADF on the weekly BET and BET-TR; the indicator of this chapter on the daily BET', 'BSADF pe BET și BET-TR săptămînal; indicatorul din acest capitol pe BET zilnic'),
      T('the evaluation of Section 6 on the BET since 2000: alarms, false alarms, base rate', 'evaluarea din Secțiunea 6 pe BET din 2000: alarme, alarme false, frecvența de bază')]),
    (T('Extensions', 'Extensii'),
     [T('the price--earnings ratio instead of the price; Bitcoin and Ethereum together', 'raportul preț--profit în locul prețului; Bitcoin și Ethereum împreună'),
      T('compare with the regime-switching models of Chapter 10', 'comparați cu modelele cu schimbare de regim din Capitolul 10')]),
    T('Further reading on the Bucharest Stock Exchange: \\refPeleCrash', 'Lecturi suplimentare despre Bursa de Valori București: \\refPeleCrash'),
    T('Next: Chapter 14, multivariate GARCH models', 'Urmează: Capitolul 14, modele GARCH multivariate')))

D.references(bib())

if __name__ == '__main__':
    finalize(D.write(V))
