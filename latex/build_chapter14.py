r"""
build_chapter14.py -- Capitolul 14 (Modele GARCH multivariate), EN + RO dintr-o singură sursă
==========================================================================================
Capitol de studiu individual. Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_14/ch14_numbers.json
(generate_all_charts.py). Nicio cifră nu este scrisă de mînă. Materialul vechiului capitol 5b (GARCH multivariat)
și al capitolului 6 din MFM, reformulat pentru licență.
Ieșire:
  EN/Courses/chapter14_multivariate_garch_models.tex
  RO/Cursuri/capitol14_modele_garch_multivariate.tex
Rulare:
  python3 Quantlets/Ch_14/generate_all_charts.py
  python3 latex/build_chapter14.py && python3 latex/tsa_build.py compile 14
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, table, photo   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch14_common import QLURL, REFS, T, bib, date, finalize, load, month, pv   # noqa: E402


def items(*xs):
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


N = load()
V = Values()
D = Deck(14, 'lecture', refs=REFS)
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
    'engle': ('ch5_robert_engle_2022.jpg', C + '0603-Kraneshares_KRBN-RobertEngle-JonDemske-16_(cropped).jpg',
              T('Photo', 'Foto') + ': Jon Demske (2022); CC BY-SA 4.0; Wikimedia Commons'),
    'lehman': ('ch5_lehman_2008.jpg', C + 'Lehman_Brothers-NYC-20080915.jpg',
               T('Photo', 'Foto') + ': Robert Scoble (2008); CC BY 2.0; Wikimedia Commons'),
    'bvb': ('ch1_bvb_2024.jpg', C + 'Bursa_de_Valori_București.jpg',
            T('Photo', 'Foto') + ': Corina Chitu (2024); CC BY-SA 4.0; Wikimedia Commons'),
    'frankfurt': ('ch6_frankfurt_exchange_2015.jpg', C + 'Frankfurt_Stock_Exchange_(Ank_Kumar)_01.jpg',
                  T('Photo', 'Foto') + ': Ank Kumar (2015); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


def two(left, right, wl='0.38', wr='0.6'):
    return (f'\\begin{{columns}}[T]\n\\begin{{column}}{{{wl}\\textwidth}}\n{left}\n\\end{{column}}\n'
            f'\\begin{{column}}{{{wr}\\textwidth}}\n{right}\n\\end{{column}}\n\\end{{columns}}')


def de(x):
    """RO: „de” after a numeral that needs it (20, 21, ..., 100, ...), nothing in EN."""
    x = abs(int(x))
    return '⟦|| de⟧' if x >= 20 and (x % 100 >= 20 or x % 100 == 0) else ''


# =============================================================================
# CIFRE
# =============================================================================
P = V.put
EX = N['ex']
P('ex.div2', EX['div']['0.2'], 1)
P('ex.div8', EX['div']['0.8'], 1)
for k in ('c', 'q11', 'q22', 'q12', 't1', 't2', 't3'):
    P(f'ex.{k}', EX['dcc'][k], 4 if k in ('q22',) else 3)
P('ex.rho', EX['dcc']['rho'], 3)
P('ex.z', -EX['var']['z'], 3)
for r in ('0.3', '0.9'):
    k = r[-1]
    P(f'ex.sp{k}', EX['var'][r]['sp'], 3)
    P(f'ex.var{k}', EX['var'][r]['var'], 2)
    P(f'ex.vark{k}', 10 * EX['var'][r]['var'], 1)          # thousand lei for a portfolio of 1 million lei
P('ex.h', EX['hedge']['h'], 2)
P('ex.he', EX['hedge']['he'], 2)
P('ex.sdh', EX['hedge']['sd_hedged'], 2)
for k in ('aa', 'bb', 'ee', 't1', 't2', 't3', 'h12'):
    P(f'bk.{k}', EX['bekk'][k], 4 if k in ('t3', 'h12') else 3)

RO_ = N['roll']
for k, s in (('sd', 'sp500_dax'), ('bd', 'bet_dax'), ('bc', 'sp500_btc')):
    for f in ('full', 'min', 'max', 'last'):
        P(f'{k}.{f}', RO_[s][f], 2)
    V.raw(f'{k}.dmin', month(RO_[s]['dmin']))
    V.raw(f'{k}.dmax', month(RO_[s]['dmax']))
A = RO_['apr2025']
P('apr9.sp', A['2025-04-09'][0], 1, sign=True)
P('apr9.dax', A['2025-04-09'][1], 1, sign=True)
P('apr10.sp', A['2025-04-10'][0], 1, sign=True)
P('apr10.dax', A['2025-04-10'][1], 1, sign=True)

PR = N['params']
for nn in ('2', '5', '10', '50'):
    for m in ('VEC', 'DVEC', 'BEKK', 'DBEKK', 'SBEKK', 'CCC', 'DCC'):
        V.int(f'np.{nn}.{m}', PR[nn][m])

S1 = N['step1']
V.int('s1.n', S1['n'])
V.raw('s1.first', date(S1['first']))
V.raw('s1.last', date(S1['last']))
for k in ('sp500', 'dax'):
    e = S1[k]
    P(f's1.{k}.mu', e['mu'], 3)
    P(f's1.{k}.om', e['omega'], 3)
    P(f's1.{k}.al', e['alpha'], 3)
    P(f's1.{k}.be', e['beta'], 3)
    P(f's1.{k}.pers', e['pers'], 3)
    P(f's1.{k}.vmax', e['vmax'], 0)
    V.raw(f's1.{k}.dmax', date(e['dmax']))
    P(f's1.{k}.rk', e['rkurt'], 1)
    P(f's1.{k}.zk', e['zkurt'], 1)
P('s1.zcorr', S1['zcorr'], 3)
P('s1.rcorr', S1['rcorr'], 3)

DC = N['dcc']
for k in ('a', 'b', 'se_a', 'se_b'):
    P(f'dc.{k}', DC[k], 4)
P('dc.ab', DC['ab'], 4)
P('dc.hl', DC['hl'], 0)
P('dc.lr', DC['lr'], 1)
P('dc.crit', DC['crit'], 2)
P('dc.qbar', DC['qbar'], 3)
for k in ('min', 'max', 'mean', 'sd_dcc', 'sd_roll', 'mean_2008', 'mean_2020', 'mean_calm'):
    P(f'dc.{k}', DC[k], 2 if k.startswith(('min', 'max', 'mean')) else 3)
for k in ('vol_2008', 'vol_2020', 'vol_calm'):
    P(f'dc.{k}', DC[k], 0)
V.raw('dc.dmin', month(DC['dmin']))
V.raw('dc.dmax', month(DC['dmax']))

Z = N['zoom']
for c in ('2008', '2020'):
    P(f'z{c}.rmax', Z[c]['rmax'], 2)
    P(f'z{c}.r0', Z[c]['r0'], 2)
    P(f'z{c}.vmax', Z[c]['vmax'], 0)
    V.raw(f'z{c}.rdmax', date(Z[c]['rdmax']))
    V.raw(f'z{c}.vdmax', date(Z[c]['vdmax']))

PA = N['panel']
for k in ('a', 'b', 'se_a', 'se_b'):
    P(f'pa.{k}', PA[k], 4)
P('pa.ab', PA['ab'], 4)
P('pa.hl', PA['hl'], 0)
P('pa.lr', PA['lr'], 1)
V.int('pa.n', PA['n'])
V.raw('pa.first', date(PA['first']))
for k in ('sp500', 'dax', 'bet', 'eurron', 'btc'):
    P(f'pa.{k}.al', PA['garch'][k]['alpha'], 2)
    P(f'pa.{k}.be', PA['garch'][k]['beta'], 2)
    P(f'pa.{k}.pers', PA['garch'][k]['pers'], 3)
for k in ('sp500_bet', 'sp500_dax', 'bet_dax'):
    P(f'as.{k}.d', PA[f'async_{k}']['daily'], 2)
    P(f'as.{k}.w', PA[f'async_{k}']['weekly'], 2)
for k in ('bet_dax', 'sp500_dax', 'sp500_btc', 'bet_eurron'):
    for f in ('mean', 'min', 'max', 'last'):
        P(f'pa.{k}.{f}', PA[k][f], 2)
    V.raw(f'pa.{k}.dmax', month(PA[k]['dmax']))

HT = N['heat']
P('ht.calm', HT['calm_avg'], 2)
P('ht.cris', HT['crisis_avg'], 2)
V.raw('ht.nc', str(HT['nc']))
for k in ('bet_dax', 'sp500_dax', 'sp500_btc', 'bet_eurron', 'dax_btc', 'sp500_bet'):
    P(f'ht.{k}.c', HT[k]['calm'], 2)
    P(f'ht.{k}.k', HT[k]['crisis'], 2)

VR = N['var']
P('vr.qfhs', -VR['q_fhs'], 2)
V.int('vr.nin', VR['n_in'])
V.int('vr.n', VR['DCC']['n'])
P('vr.exp', 0.01 * VR['DCC']['n'], 0)
for k, kk in (('DCC', 'dcc'), ('DCC-FHS', 'fhs'), ('CCC', 'ccc'), ('Static', 'sta')):
    e = VR[k]
    V.raw(f'vr.{kk}.x', str(e['x']) + de(e['x']))
    V.raw(f'vr.{kk}.xn', str(e['x']))
    P(f'vr.{kk}.rate', e['rate'], 2, pct=True)
    P(f'vr.{kk}.lr', e['lr'], 1)
    V.raw(f'vr.{kk}.p', pv(e['p']))
    V.raw(f'vr.{kk}.ip', pv(e['ind']['p']))
    P(f'vr.{kk}.p11', e['ind']['p11'], 1, pct=True)
    V.raw(f'vr.{kk}.v20', str(e['viol2020']))
    P(f'vr.{kk}.mv', e['mean_var'], 2)
    P(f'vr.{kk}.maxv', e['max_var'], 1)

HG = N['hedge']
P('hg.sta', HG['static'], 2)
P('hg.a', HG['a'], 4)
P('hg.b', HG['b'], 4)
for k in ('dcc_min', 'dcc_max', 'dcc_mean'):
    P(f'hg.{k}', HG[k], 2)
V.raw('hg.dmax', date(HG['dcc_dmax']))
V.int('hg.n', HG['n_oos'])
for k in ('static', 'rolling', 'dcc'):
    P(f'hg.he.{k}', HG[f'he_{k}'], 1, pct=True)
    P(f'hg.v.{k}', HG[f'var_{k}'], 3)
P('hg.v.un', HG['var_un'], 3)

# negative numbers used in running text: a typographic minus instead of a hyphen
for _k, _v in list(V.items()):
    if str(_v).startswith('⁅-'):
        V[_k] = '$-$' + _v.replace('⁅-', '⁅', 1)

# =============================================================================
# INTRODUCERE
# =============================================================================
D.frame(T('Why this chapter', 'Motivația capitolului'), items(
    (T('\\textbf{Question}: how do the risks of several assets move \\textbf{together}? Chapter 5 modelled the volatility of one series at a time',
       '\\textbf{Întrebarea}: cum se mișcă \\textbf{împreună} riscurile mai multor active? Capitolul 5 a modelat volatilitatea cîte unei singure serii'),
     [T('the risk of a portfolio, a hedge or the spread of a crisis depends on \\textbf{covariances and correlations}, and these change in time',
        'riscul unui portofoliu, al unei acoperiri sau al propagării unei crize depinde de \\textbf{covarianțe și corelații}, care se schimbă în timp')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('the conditional covariance matrix $\\mathbf{H}_t$ and its conditions', 'matricea de covarianță condiționată $\\mathbf{H}_t$ și condițiile ei'),
      T('the VEC and BEKK models; the number of parameters', 'modelele VEC și BEKK; numărul de parametri'),
      T('CCC and DCC: volatilities and correlations estimated in two steps', 'CCC și DCC: volatilități și corelații estimate în doi pași'),
      T('applications: correlations in crises, portfolio VaR 1\\%, hedge ratios', 'aplicații: corelațiile în crize, VaR 1\\% al unui portofoliu, rapoarte de acoperire')]),
    (T('\\textbf{Self-study chapter}: read the slides in order, redo the worked examples, run the notebook, then answer the self-assessment and the quiz',
       '\\textbf{Capitol de studiu individual}: parcurgeți slide-urile în ordine, refaceți exemplele rezolvate, rulați notebook-ul, apoi răspundeți la autoevaluare și la quiz'),
     [T('prerequisites: Chapter 5 (GARCH), Chapter 6 (VAR models), Chapter 7 (cointegration)', 'cunoștințe necesare: Capitolul 5 (GARCH), Capitolul 6 (modele VAR), Capitolul 7 (cointegrare)')])))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Explain why portfolio risk, hedging and contagion need a model for time-varying covariances', 'Explicați de ce riscul unui portofoliu, acoperirea riscului și contagiunea cer un model pentru covarianțe variabile în timp'),
    T('Write the VEC, BEKK, CCC and DCC models and count their parameters', 'Scrieți modelele VEC, BEKK, CCC și DCC și numărați-le parametrii'),
    T('Compute one step of the DCC recursion by hand and interpret $a$, $b$ and $a+b$', 'Calculați de mînă un pas al recurenței DCC și interpretați $a$, $b$ și $a+b$'),
    T('Estimate a DCC model in two steps and test it against constant correlations', 'Estimați un model DCC în doi pași și testați-l față de corelațiile constante'),
    T('Compute and backtest a portfolio VaR 1\\% from $\\mathbf{H}_t$', 'Calculați VaR 1\\% al unui portofoliu pe baza lui $\\mathbf{H}_t$ și verificați-l prin backtesting'),
    T('Compute dynamic minimum-variance hedge ratios and their hedging effectiveness', 'Calculați rapoarte dinamice de acoperire cu varianță minimă și eficiența acoperirii')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook background: \\refHP\\ (GARCH models); surveys: \\refBLR, \\refST', 'Manual: \\refHP\\ (modelele GARCH); sinteze: \\refBLR, \\refST'),
     [T('the original papers: \\refBEW\\ (VEC), \\refEK\\ (BEKK), \\refBoll\\ (CCC), \\refEngle\\ (DCC)', 'lucrările originale: \\refBEW\\ (VEC), \\refEK\\ (BEKK), \\refBoll\\ (CCC), \\refEngle\\ (DCC)')]),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_14}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_14}'),
     [T('step 1 (univariate GARCH) uses the \\texttt{arch} package; step 2 (DCC) is written out in about 40 lines of \\texttt{numpy}, because \\texttt{arch} has no DCC',
        'pasul 1 (GARCH univariat) folosește pachetul \\texttt{arch}; pasul 2 (DCC) este scris explicit în circa 40 de linii \\texttt{numpy}, deoarece \\texttt{arch} nu are DCC')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter14_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter14_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. DE CE VOLATILITATE MULTIVARIATĂ
# =============================================================================
D.section('Why multivariate volatility', 'De ce volatilitate multivariată')

D.frame(T('Three questions that need covariances', 'Trei întrebări care cer covarianțe'), items(
    (T('\\textbf{Portfolio risk}: the variance of a portfolio with weights $\\mathbf{w}$ is $\\sigma_p^2 = \\mathbf{w}^\\top\\mathbf{H}\\mathbf{w}$ \\refMark',
       '\\textbf{Riscul portofoliului}: varianța unui portofoliu cu ponderile $\\mathbf{w}$ este $\\sigma_p^2 = \\mathbf{w}^\\top\\mathbf{H}\\mathbf{w}$ \\refMark'),
     [T('two assets: $\\sigma_p^2 = w_1^2\\sigma_1^2 + w_2^2\\sigma_2^2 + 2w_1w_2\\rho\\,\\sigma_1\\sigma_2$; the correlation $\\rho$ decides how much diversification helps',
        'două active: $\\sigma_p^2 = w_1^2\\sigma_1^2 + w_2^2\\sigma_2^2 + 2w_1w_2\\rho\\,\\sigma_1\\sigma_2$; corelația $\\rho$ decide cît ajută diversificarea')]),
    (T('\\textbf{Hedging}: how many units of an asset to sell to offset the risk of another', '\\textbf{Acoperirea riscului} (hedging): cîte unități dintr-un activ trebuie vîndute pentru a compensa riscul altuia'),
     [T('the answer is a ratio of a covariance to a variance (Section 7)', 'răspunsul este raportul dintre o covarianță și o varianță (secțiunea 7)')]),
    (T('\\textbf{Contagion}: do markets move more closely together in a crisis?', '\\textbf{Contagiunea}: se mișcă piețele mai strîns împreună într-o criză?'),
     [T('if correlations rise exactly when volatility rises, diversification fails when it is needed most \\refLS', 'dacă corelațiile cresc tocmai cînd crește volatilitatea, diversificarea dispare cînd este mai necesară \\refLS')]),
    T('A univariate GARCH (Chapter 5) gives $\\sigma_{1,t}$ and $\\sigma_{2,t}$; this chapter adds the time-varying $\\rho_t$', 'Un GARCH univariat (Capitolul 5) dă $\\sigma_{1,t}$ și $\\sigma_{2,t}$; acest capitol adaugă corelația variabilă $\\rho_t$')))

D.frame(T('Worked example: correlation and portfolio risk', 'Exemplu rezolvat: corelația și riscul portofoliului'), items(
    T('Two assets with annual volatility 20\\% each, weights 50\\% and 50\\%', 'Două active, fiecare cu volatilitatea anuală de 20\\%, ponderi 50\\% și 50\\%'),
    (T('$\\rho = 0.2$: $\\sigma_p^2 = 0.25 \\cdot 400 + 0.25 \\cdot 400 + 2 \\cdot 0.25 \\cdot 0.2 \\cdot 400 = 240$, so $\\sigma_p = @{ex.div2}\\%$',
       '$\\rho = 0.2$: $\\sigma_p^2 = 0.25 \\cdot 400 + 0.25 \\cdot 400 + 2 \\cdot 0.25 \\cdot 0.2 \\cdot 400 = 240$, deci $\\sigma_p = @{ex.div2}\\%$'), []),
    (T('$\\rho = 0.8$: $\\sigma_p^2 = 200 + 2 \\cdot 0.25 \\cdot 0.8 \\cdot 400 = 360$, so $\\sigma_p = @{ex.div8}\\%$',
       '$\\rho = 0.8$: $\\sigma_p^2 = 200 + 2 \\cdot 0.25 \\cdot 0.8 \\cdot 400 = 360$, deci $\\sigma_p = @{ex.div8}\\%$'), []),
    (T('The same assets and weights: the risk of the portfolio rises by more than a fifth only because the correlation rose',
       'Aceleași active și aceleași ponderi: riscul portofoliului crește cu mai mult de o cincime doar pentru că a crescut corelația'),
     [T('a risk model that keeps $\\rho$ fixed at its calm-period value underestimates the risk in a crisis', 'un model de risc care ține $\\rho$ fix la valoarea din perioadele calme subestimează riscul într-o criză')])))

chart(T('Portfolio volatility against the correlation', 'Volatilitatea portofoliului în funcție de corelație'), 'tsa_ch14_diversification', 'TSA_ch14_correlations', [
    T('A 50/50 portfolio of two assets with 20\\% volatility each; the points are the two cases of the worked example',
      'Un portofoliu 50/50 din două active cu volatilitatea de 20\\% fiecare; punctele sînt cele două cazuri din exemplul rezolvat')],
    h='0.6\\textheight')

interp(('the diversification curve', 'curbei de diversificare'), [
    (T('$\\rho = 1$: no diversification, $\\sigma_p = 20\\%$; $\\rho = 0$: $\\sigma_p = 20/\\sqrt{2} \\approx 14.1\\%$; $\\rho = -1$: the risk disappears',
       '$\\rho = 1$: nicio diversificare, $\\sigma_p = 20\\%$; $\\rho = 0$: $\\sigma_p = 20/\\sqrt{2} \\approx 14{,}1\\%$; $\\rho = -1$: riscul dispare'), []),
    (T('The curve is concave: a rise of $\\rho$ from 0.2 to 0.8 costs @{ex.div2}\\% $\\to$ @{ex.div8}\\% of volatility',
       'Curba este concavă: o creștere a lui $\\rho$ de la 0,2 la 0,8 duce volatilitatea de la @{ex.div2}\\% la @{ex.div8}\\%'), []),
    T('The question of the chapter: does $\\rho$ stay put in real markets, or does it move with the state of the market?', 'Întrebarea capitolului: rămîne $\\rho$ constant pe piețele reale sau se mișcă odată cu starea pieței?')])

chart(T('Correlations change over time', 'Corelațiile se schimbă în timp'), 'tsa_ch14_rolling_corr', 'TSA_ch14_correlations', [
    T('Correlation of daily log returns over rolling windows of 250 trading days (about one year), dated at the window end; returns on the days on which both markets trade; shaded: September 2008--March 2009 and February--April 2020',
      'Corelația randamentelor logaritmice zilnice pe ferestre mobile de 250 de zile de tranzacționare (aproximativ un an), datată la sfîrșitul ferestrei; randamente în zilele în care se tranzacționează ambele piețe; zonele colorate: septembrie 2008--martie 2009 și februarie--aprilie 2020')],
    h='0.62\\textheight')

interp(('the rolling correlations', 'corelațiilor pe ferestre mobile'), [
    (T('S\\&P 500 and DAX: between @{sd.min} (@{sd.dmin}) and @{sd.max} (@{sd.dmax}); full-sample value @{sd.full}', 'S\\&P 500 și DAX: între @{sd.min} (@{sd.dmin}) și @{sd.max} (@{sd.dmax}); valoarea pe tot eșantionul @{sd.full}'),
     [T('the euro-area debt crisis of 2010--2012 tied the two markets even more closely than 2008', 'criza datoriilor din zona euro din 2010--2012 a legat cele două piețe chiar mai strîns decît 2008')]),
    (T('BET and DAX: close to zero before 2007 (minimum @{bd.min}), up to @{bd.max} in @{bd.dmax}: the Romanian market became integrated with Europe after EU accession',
       'BET și DAX: aproape de zero înainte de 2007 (minimum @{bd.min}), pînă la @{bd.max} în @{bd.dmax}: piața românească s-a integrat cu Europa după aderarea la UE'), []),
    (T('S\\&P 500 and Bitcoin: from @{bc.min} (@{bc.dmin}) to @{bc.max} (@{bc.dmax}); Bitcoin became a risk asset correlated with equities after 2020',
       'S\\&P 500 și Bitcoin: de la @{bc.min} (@{bc.dmin}) la @{bc.max} (@{bc.dmax}); Bitcoin a devenit după 2020 un activ riscant, corelat cu acțiunile'), []),
    T('Correlations wander far from any constant value: a model must let them move', 'Corelațiile se îndepărtează mult de orice valoare constantă: modelul trebuie să le permită să se miște')])

D.frame(T('A trap: asynchronous trading', 'O capcană: tranzacționarea asincronă'), items(
    (T('Markets close at different hours: Bucharest at 16:45 CET (17:45 local time), Frankfurt at 17:30 CET, New York at 22:00 CET; Bitcoin trades all day, every day',
       'Piețele se închid la ore diferite: București la 16:45 CET (17:45, ora locală), Frankfurt la 17:30 CET, New York la 22:00 CET; Bitcoin se tranzacționează non-stop'),
     [T('news that arrives after the European close shows up in the European prices only the next day', 'o știre care apare după închiderea piețelor europene intră în prețurile europene abia a doua zi')]),
    (T('Example: 9 April 2025, the pause of the US tariffs announced in the afternoon in New York', 'Exemplu: 9 aprilie 2025, suspendarea tarifelor vamale ale SUA, anunțată după-amiaza la New York'),
     [T('9 April: S\\&P 500 @{apr9.sp}\\%, DAX @{apr9.dax}\\%; 10 April: S\\&P 500 @{apr10.sp}\\%, DAX @{apr10.dax}\\%', '9 aprilie: S\\&P 500 @{apr9.sp}\\%, DAX @{apr9.dax}\\%; 10 aprilie: S\\&P 500 @{apr10.sp}\\%, DAX @{apr10.dax}\\%'),
      T('the same news, on two different days: the daily correlation is biased towards zero', 'aceeași știre, în două zile diferite: corelația zilnică este deplasată spre zero')]),
    (T('Correlations of daily and weekly returns since 2015: S\\&P 500 and BET @{as.sp500_bet.d} against @{as.sp500_bet.w}; S\\&P 500 and DAX @{as.sp500_dax.d} against @{as.sp500_dax.w}; BET and DAX @{as.bet_dax.d} against @{as.bet_dax.w}',
       'Corelațiile randamentelor zilnice și săptămînale din 2015: S\\&P 500 și BET @{as.sp500_bet.d} față de @{as.sp500_bet.w}; S\\&P 500 și DAX @{as.sp500_dax.d} față de @{as.sp500_dax.w}; BET și DAX @{as.bet_dax.d} față de @{as.bet_dax.w}'),
     [T('the gap is small for BET and DAX, which close almost together; remedies: weekly returns or returns between synchronous times',
        'diferența este mică pentru BET și DAX, care se închid aproape simultan; remedii: randamente săptămînale sau randamente între momente sincrone')])), size='footnotesize')

D.recap(('Why multivariate volatility', 'de ce volatilitate multivariată'), [
    T('Portfolio variance $\\mathbf{w}^\\top\\mathbf{H}\\mathbf{w}$, hedge ratios and contagion all depend on covariances', 'Varianța portofoliului $\\mathbf{w}^\\top\\mathbf{H}\\mathbf{w}$, rapoartele de acoperire și contagiunea depind toate de covarianțe'),
    T('Rolling correlations of real markets move a lot: integration, crises, new asset classes', 'Corelațiile pe ferestre mobile se mișcă mult pe piețele reale: integrare, crize, clase noi de active'),
    T('Check the trading hours: asynchronous closes bias daily correlations towards zero', 'Verificați orele de tranzacționare: închiderile asincrone deplasează corelațiile zilnice spre zero')])

# =============================================================================
# 2. CADRUL MULTIVARIAT
# =============================================================================
D.section('The multivariate GARCH framework', 'Cadrul GARCH multivariat')

D.frame(T('The model for $N$ return series', 'Modelul pentru $N$ serii de randamente'), items(
    (T('$\\mathbf{r}_t = \\boldsymbol{\\mu} + \\boldsymbol{\\varepsilon}_t$, \\quad $\\boldsymbol{\\varepsilon}_t = \\mathbf{H}_t^{1/2}\\boldsymbol{\\eta}_t$, \\quad $\\boldsymbol{\\eta}_t$ i.i.d. with mean $\\mathbf{0}$ and covariance $\\mathbf{I}_N$',
       '$\\mathbf{r}_t = \\boldsymbol{\\mu} + \\boldsymbol{\\varepsilon}_t$, \\quad $\\boldsymbol{\\varepsilon}_t = \\mathbf{H}_t^{1/2}\\boldsymbol{\\eta}_t$, \\quad $\\boldsymbol{\\eta}_t$ i.i.d. cu media $\\mathbf{0}$ și covarianța $\\mathbf{I}_N$'),
     [T('$\\mathbf{r}_t$: the vector of $N$ returns on day $t$; $\\boldsymbol{\\mu}$: the means (or a VAR model for the mean, Chapter 6)', '$\\mathbf{r}_t$: vectorul celor $N$ randamente din ziua $t$; $\\boldsymbol{\\mu}$: mediile (sau un model VAR pentru medie, Capitolul 6)'),
      T('$\\mathbf{H}_t^{1/2}$: a square root of $\\mathbf{H}_t$, e.g. the Cholesky factor $\\mathbf{L}_t$ with $\\mathbf{L}_t\\mathbf{L}_t^\\top = \\mathbf{H}_t$', '$\\mathbf{H}_t^{1/2}$: o rădăcină pătrată a lui $\\mathbf{H}_t$, de exemplu factorul Cholesky $\\mathbf{L}_t$, cu $\\mathbf{L}_t\\mathbf{L}_t^\\top = \\mathbf{H}_t$')]),
    (T('$\\mathbf{H}_t = \\Var(\\mathbf{r}_t \\mid \\mathcal{F}_{t-1})$: the \\textbf{conditional covariance matrix}, given the information $\\mathcal{F}_{t-1}$ up to day $t-1$',
       '$\\mathbf{H}_t = \\Var(\\mathbf{r}_t \\mid \\mathcal{F}_{t-1})$: \\textbf{matricea de covarianță condiționată}, dată fiind informația $\\mathcal{F}_{t-1}$ pînă în ziua $t-1$'),
     [T('diagonal: the conditional variances $h_{ii,t} = \\sigma_{i,t}^2$ (Chapter 5)', 'diagonala: varianțele condiționate $h_{ii,t} = \\sigma_{i,t}^2$ (Capitolul 5)'),
      T('off-diagonal: the conditional covariances $h_{ij,t}$; the conditional correlation is $\\rho_{ij,t} = h_{ij,t}/\\sqrt{h_{ii,t}h_{jj,t}}$', 'în afara diagonalei: covarianțele condiționate $h_{ij,t}$; corelația condiționată este $\\rho_{ij,t} = h_{ij,t}/\\sqrt{h_{ii,t}h_{jj,t}}$')]),
    T('A \\textbf{multivariate GARCH} (MGARCH) model is a rule that computes $\\mathbf{H}_t$ from past shocks $\\boldsymbol{\\varepsilon}_{t-1}$ and past $\\mathbf{H}_{t-1}$',
      'Un model \\textbf{GARCH multivariat} (MGARCH) este o regulă care calculează $\\mathbf{H}_t$ din șocurile trecute $\\boldsymbol{\\varepsilon}_{t-1}$ și din $\\mathbf{H}_{t-1}$')))

D.frame(T('Two conditions on $\\mathbf{H}_t$', 'Două condiții pentru $\\mathbf{H}_t$'), items(
    (T('\\textbf{Symmetry}: $h_{ij,t} = h_{ji,t}$, so only $N(N+1)/2$ elements are distinct', '\\textbf{Simetria}: $h_{ij,t} = h_{ji,t}$, deci doar $N(N+1)/2$ elemente sînt distincte'),
     [T('the operator $\\mathrm{vech}$ stacks the lower triangle: $\\mathrm{vech}\\begin{pmatrix} h_{11} & h_{12}\\\\ h_{12} & h_{22}\\end{pmatrix} = (h_{11}, h_{12}, h_{22})^\\top$',
        'operatorul $\\mathrm{vech}$ așază triunghiul inferior într-un vector: $\\mathrm{vech}\\begin{pmatrix} h_{11} & h_{12}\\\\ h_{12} & h_{22}\\end{pmatrix} = (h_{11}, h_{12}, h_{22})^\\top$'),
      T('$N = 2$: 3 elements; $N = 10$: 55; $N = 50$: 1\\,275', '$N = 2$: 3 elemente; $N = 10$: 55; $N = 50$: 1\\,275')]),
    (T('\\textbf{Positive definiteness}: $\\mathbf{w}^\\top\\mathbf{H}_t\\mathbf{w} > 0$ for every $\\mathbf{w} \\neq \\mathbf{0}$, i.e. every portfolio has a positive variance',
       '\\textbf{Pozitiv definirea}: $\\mathbf{w}^\\top\\mathbf{H}_t\\mathbf{w} > 0$ pentru orice $\\mathbf{w} \\neq \\mathbf{0}$, adică orice portofoliu are varianță pozitivă'),
     [T('$N = 2$: $h_{11,t} > 0$ and $h_{11,t}h_{22,t} - h_{12,t}^2 > 0$, i.e. $|\\rho_{12,t}| < 1$', '$N = 2$: $h_{11,t} > 0$ și $h_{11,t}h_{22,t} - h_{12,t}^2 > 0$, adică $|\\rho_{12,t}| < 1$'),
      T('example: $h_{11} = 4$, $h_{22} = 1$, $h_{12} = 2.5$ gives $4 - 6.25 < 0$: an impossible ``correlation\'\' of 1.25', 'exemplu: $h_{11} = 4$, $h_{22} = 1$, $h_{12} = 2{,}5$ dau $4 - 6{,}25 < 0$: o „corelație” imposibilă, de 1,25')]),
    T('A good MGARCH model guarantees both conditions for \\textbf{every} $t$, with few parameters', 'Un model MGARCH bun garantează ambele condiții pentru \\textbf{orice} $t$, cu puțini parametri')))

D.frame(T('Estimation by maximum likelihood', 'Estimarea prin verosimilitate maximă'), items(
    (T('If $\\boldsymbol{\\eta}_t$ follows the multivariate Normal distribution, the log-likelihood is',
       'Dacă $\\boldsymbol{\\eta}_t$ urmează distribuția Normală multivariată, logaritmul verosimilității este'),
     [T('$\\ell(\\theta) = -\\frac12\\sum_{t=1}^{T}\\left(N\\log 2\\pi + \\log|\\mathbf{H}_t| + \\boldsymbol{\\varepsilon}_t^\\top\\mathbf{H}_t^{-1}\\boldsymbol{\\varepsilon}_t\\right)$',
        '$\\ell(\\theta) = -\\frac12\\sum_{t=1}^{T}\\left(N\\log 2\\pi + \\log|\\mathbf{H}_t| + \\boldsymbol{\\varepsilon}_t^\\top\\mathbf{H}_t^{-1}\\boldsymbol{\\varepsilon}_t\\right)$'),
      T('$|\\mathbf{H}_t|$ is the determinant; for $N = 1$ this is the GARCH likelihood of Chapter 5', '$|\\mathbf{H}_t|$ este determinantul; pentru $N = 1$ regăsim verosimilitatea GARCH din Capitolul 5')]),
    (T('Returns have fat tails, so the Normal likelihood is used as \\textbf{quasi-maximum likelihood} (QML): the estimates stay consistent, the standard errors need a robust (sandwich) correction',
       'Randamentele au cozi groase, deci verosimilitatea Normală este folosită ca \\textbf{cvasi-verosimilitate maximă} (QML): estimatorii rămîn consistenți, iar erorile standard au nevoie de o corecție robustă (sandwich)'), []),
    (T('Every evaluation of $\\ell(\\theta)$ needs $\\mathbf{H}_t^{-1}$ and $|\\mathbf{H}_t|$ for all $t$: the cost grows fast with $N$ and with the number of parameters',
       'Fiecare evaluare a lui $\\ell(\\theta)$ cere $\\mathbf{H}_t^{-1}$ și $|\\mathbf{H}_t|$ pentru toți $t$: costul crește repede cu $N$ și cu numărul de parametri'), [])))

D.recap(('The multivariate GARCH framework', 'cadrul GARCH multivariat'), [
    T('$\\mathbf{H}_t$: conditional variances on the diagonal, covariances off the diagonal; $\\rho_{ij,t} = h_{ij,t}/\\sqrt{h_{ii,t}h_{jj,t}}$', '$\\mathbf{H}_t$: varianțele condiționate pe diagonală, covarianțele în afara ei; $\\rho_{ij,t} = h_{ij,t}/\\sqrt{h_{ii,t}h_{jj,t}}$'),
    T('Conditions: symmetry ($N(N+1)/2$ distinct elements) and positive definiteness', 'Condiții: simetria ($N(N+1)/2$ elemente distincte) și pozitiv definirea'),
    T('Estimation by (quasi-)maximum likelihood with the multivariate Normal density', 'Estimare prin (cvasi-)verosimilitate maximă cu densitatea Normală multivariată')])

# =============================================================================
# 3. VEC ȘI BEKK
# =============================================================================
D.section('VEC and BEKK models', 'Modelele VEC și BEKK')

D.frame(T('The VEC model', 'Modelul VEC'), items(
    (T('\\refBEW: every element of $\\mathbf{H}_t$ depends on every past squared shock, cross-product and element of $\\mathbf{H}_{t-1}$',
       '\\refBEW: fiecare element al lui $\\mathbf{H}_t$ depinde de toate pătratele și produsele încrucișate ale șocurilor trecute și de toate elementele lui $\\mathbf{H}_{t-1}$'),
     [T('VEC(1,1): $\\mathrm{vech}(\\mathbf{H}_t) = \\mathbf{c} + \\mathbf{A}\\,\\mathrm{vech}(\\boldsymbol{\\varepsilon}_{t-1}\\boldsymbol{\\varepsilon}_{t-1}^\\top) + \\mathbf{B}\\,\\mathrm{vech}(\\mathbf{H}_{t-1})$',
        'VEC(1,1): $\\mathrm{vech}(\\mathbf{H}_t) = \\mathbf{c} + \\mathbf{A}\\,\\mathrm{vech}(\\boldsymbol{\\varepsilon}_{t-1}\\boldsymbol{\\varepsilon}_{t-1}^\\top) + \\mathbf{B}\\,\\mathrm{vech}(\\mathbf{H}_{t-1})$'),
      T('with $k = N(N+1)/2$: $\\mathbf{c}$ has $k$ elements, $\\mathbf{A}$ and $\\mathbf{B}$ are $k \\times k$: in total $k + 2k^2$ parameters', 'cu $k = N(N+1)/2$: $\\mathbf{c}$ are $k$ elemente, $\\mathbf{A}$ și $\\mathbf{B}$ sînt $k \\times k$: în total $k + 2k^2$ parametri')]),
    (T('$N = 2$: $k = 3$ and @{np.2.VEC} parameters; $N = 10$: @{np.10.VEC}', '$N = 2$: $k = 3$ și @{np.2.VEC} de parametri; $N = 10$: @{np.10.VEC}'),
     [T('and positive definiteness of $\\mathbf{H}_t$ is \\textbf{not} guaranteed: it needs complicated restrictions', 'iar pozitiv definirea lui $\\mathbf{H}_t$ \\textbf{nu} este garantată: cere restricții complicate')]),
    (T('\\textbf{Diagonal VEC}: $h_{ij,t} = c_{ij} + a_{ij}\\varepsilon_{i,t-1}\\varepsilon_{j,t-1} + b_{ij}h_{ij,t-1}$', '\\textbf{VEC diagonal}: $h_{ij,t} = c_{ij} + a_{ij}\\varepsilon_{i,t-1}\\varepsilon_{j,t-1} + b_{ij}h_{ij,t-1}$'),
     [T('each element is a GARCH(1,1) of its own; $3k$ parameters (@{np.2.DVEC} for $N = 2$); still no guarantee of positive definiteness', 'fiecare element este un GARCH(1,1) separat; $3k$ parametri (@{np.2.DVEC} pentru $N = 2$); tot fără garanția pozitiv definirii')])))

D.frame(T('The BEKK model', 'Modelul BEKK'), items(
    (T('\\refEK: $\\mathbf{H}_t = \\mathbf{C}\\mathbf{C}^\\top + \\mathbf{A}^\\top\\boldsymbol{\\varepsilon}_{t-1}\\boldsymbol{\\varepsilon}_{t-1}^\\top\\mathbf{A} + \\mathbf{B}^\\top\\mathbf{H}_{t-1}\\mathbf{B}$',
       '\\refEK: $\\mathbf{H}_t = \\mathbf{C}\\mathbf{C}^\\top + \\mathbf{A}^\\top\\boldsymbol{\\varepsilon}_{t-1}\\boldsymbol{\\varepsilon}_{t-1}^\\top\\mathbf{A} + \\mathbf{B}^\\top\\mathbf{H}_{t-1}\\mathbf{B}$'),
     [T('BEKK: Baba, Engle, Kraft and Kroner; $\\mathbf{C}$ lower triangular, $\\mathbf{A}$ and $\\mathbf{B}$ are $N \\times N$', 'BEKK: Baba, Engle, Kraft și Kroner; $\\mathbf{C}$ inferior triunghiulară, $\\mathbf{A}$ și $\\mathbf{B}$ de dimensiune $N \\times N$')]),
    (T('\\textbf{Positive definite by construction}: each term is a ``square\'\' $\\mathbf{X}^\\top\\mathbf{M}\\mathbf{X}$ of a positive (semi)definite matrix',
       '\\textbf{Pozitiv definită prin construcție}: fiecare termen este un „pătrat” $\\mathbf{X}^\\top\\mathbf{M}\\mathbf{X}$ al unei matrice pozitiv (semi)definite'),
     [T('parameters: $N(N+1)/2 + 2N^2$, i.e. @{np.2.BEKK} for $N = 2$ and @{np.10.BEKK} for $N = 10$', 'parametri: $N(N+1)/2 + 2N^2$, adică @{np.2.BEKK} pentru $N = 2$ și @{np.10.BEKK} pentru $N = 10$')]),
    (T('\\textbf{Diagonal BEKK} ($\\mathbf{A}$, $\\mathbf{B}$ diagonal): $h_{12,t} = c_{12}^* + a_{11}a_{22}\\,\\varepsilon_{1,t-1}\\varepsilon_{2,t-1} + b_{11}b_{22}\\,h_{12,t-1}$',
       '\\textbf{BEKK diagonal} ($\\mathbf{A}$, $\\mathbf{B}$ diagonale): $h_{12,t} = c_{12}^* + a_{11}a_{22}\\,\\varepsilon_{1,t-1}\\varepsilon_{2,t-1} + b_{11}b_{22}\\,h_{12,t-1}$'),
     [T('\\textbf{scalar BEKK}: $\\mathbf{A} = a\\mathbf{I}$, $\\mathbf{B} = b\\mathbf{I}$, the same dynamics for all elements', '\\textbf{BEKK scalar}: $\\mathbf{A} = a\\mathbf{I}$, $\\mathbf{B} = b\\mathbf{I}$, aceeași dinamică pentru toate elementele')]),
    T('The off-diagonal elements of a full $\\mathbf{A}$ measure \\textbf{volatility spillovers}: a shock to asset 1 raises the variance of asset 2', 'Elementele din afara diagonalei ale unei matrice $\\mathbf{A}$ complete măsoară \\textbf{transmiterea volatilității}: un șoc al activului 1 crește varianța activului 2')),
    size='footnotesize')

D.frame(T('Worked example: one step of a diagonal BEKK', 'Exemplu rezolvat: un pas al unui BEKK diagonal'), items(
    (T('Given: $c_{12}^* = 0.02$, $a_{11} = 0.30$, $a_{22} = 0.25$, $b_{11} = 0.94$, $b_{22} = 0.95$, $h_{12,t-1} = 0.5$; yesterday\'s shocks $\\varepsilon_{1,t-1} = -2$, $\\varepsilon_{2,t-1} = -1.5$',
       'Date: $c_{12}^* = 0{,}02$, $a_{11} = 0{,}30$, $a_{22} = 0{,}25$, $b_{11} = 0{,}94$, $b_{22} = 0{,}95$, $h_{12,t-1} = 0{,}5$; șocurile de ieri $\\varepsilon_{1,t-1} = -2$, $\\varepsilon_{2,t-1} = -1{,}5$'), []),
    (T('Step 1: the coefficients $a_{11}a_{22} = @{bk.aa}$ and $b_{11}b_{22} = @{bk.bb}$', 'Pasul 1: coeficienții $a_{11}a_{22} = @{bk.aa}$ și $b_{11}b_{22} = @{bk.bb}$'), []),
    (T('Step 2: the cross-product of shocks $\\varepsilon_{1,t-1}\\varepsilon_{2,t-1} = @{bk.ee}$ (both negative: a joint fall)', 'Pasul 2: produsul șocurilor $\\varepsilon_{1,t-1}\\varepsilon_{2,t-1} = @{bk.ee}$ (ambele negative: o scădere comună)'), []),
    (T('Step 3: $h_{12,t} = @{bk.t1} + @{bk.aa} \\cdot @{bk.ee} + @{bk.bb} \\cdot 0.5 = @{bk.t1} + @{bk.t2} + @{bk.t3} = @{bk.h12}$',
       'Pasul 3: $h_{12,t} = @{bk.t1} + @{bk.aa} \\cdot @{bk.ee} + @{bk.bb} \\cdot 0{,}5 = @{bk.t1} + @{bk.t2} + @{bk.t3} = @{bk.h12}$'), []),
    (T('A joint fall raises the covariance from 0.5 to @{bk.h12}; a shock of opposite signs would have lowered it', 'O scădere comună crește covarianța de la 0,5 la @{bk.h12}; un șoc cu semne opuse ar fi scăzut-o'),
     [T('note: a diagonal BEKK has no spillovers, because $a_{12} = a_{21} = 0$', 'observație: un BEKK diagonal nu are transmitere a volatilității, deoarece $a_{12} = a_{21} = 0$')])))

D.frame(T('The curse of dimensionality', 'Problema dimensionalității'), table(
    'lrrrr', T('\\textbf{Model}', '\\textbf{Model}') + ' & $N = 2$ & $N = 5$ & $N = 10$ & $N = 50$',
    ['VEC(1,1) & @{np.2.VEC} & @{np.5.VEC} & @{np.10.VEC} & @{np.50.VEC}',
     T('Diagonal VEC', 'VEC diagonal') + ' & @{np.2.DVEC} & @{np.5.DVEC} & @{np.10.DVEC} & @{np.50.DVEC}',
     'BEKK(1,1) & @{np.2.BEKK} & @{np.5.BEKK} & @{np.10.BEKK} & @{np.50.BEKK}',
     T('Diagonal BEKK', 'BEKK diagonal') + ' & @{np.2.DBEKK} & @{np.5.DBEKK} & @{np.10.DBEKK} & @{np.50.DBEKK}',
     T('Scalar BEKK', 'BEKK scalar') + ' & @{np.2.SBEKK} & @{np.5.SBEKK} & @{np.10.SBEKK} & @{np.50.SBEKK}',
     'CCC & @{np.2.CCC} & @{np.5.CCC} & @{np.10.CCC} & @{np.50.CCC}',
     'DCC & @{np.2.DCC} & @{np.5.DCC} & @{np.10.DCC} & @{np.50.DCC}'], size='small')
    + items(T('Number of parameters of the variance equation (the means $\\boldsymbol{\\mu}$ are not counted); CCC and DCC: 3 GARCH parameters per asset plus the $N(N-1)/2$ correlations, and 2 more for DCC',
              'Numărul de parametri ai ecuației de varianță (mediile $\\boldsymbol{\\mu}$ nu sînt numărate); CCC și DCC: 3 parametri GARCH pe activ, plus cele $N(N-1)/2$ corelații, iar DCC încă 2'),
            T('A full VEC for 50 assets has more parameters than any sample has observations', 'Un VEC complet pentru 50 de active are mai mulți parametri decît numărul de observații al oricărui eșantion')),
    size='footnotesize')

chart(T('Parameters against the number of assets', 'Numărul de parametri în funcție de numărul de active'), 'tsa_ch14_param_count', 'TSA_ch14_parameters', [
    T('Number of parameters of the variance equation for $N = 2, \\dots, 50$ assets, log scale', 'Numărul de parametri ai ecuației de varianță pentru $N = 2, \\dots, 50$ de active, scară logaritmică')],
    h='0.62\\textheight')

interp(('the parameter counts', 'numărului de parametri'), [
    (T('VEC grows like $N^4/2$, full BEKK like $2N^2$: both become impossible to estimate beyond a handful of assets', 'VEC crește ca $N^4/2$, BEKK complet ca $2N^2$: ambele devin imposibil de estimat dincolo de cîteva active'), []),
    (T('CCC and DCC grow like $N^2/2$, but the correlations are \\textbf{not} found by numerical optimisation: they are sample correlations (Section 4)',
       'CCC și DCC cresc ca $N^2/2$, dar corelațiile \\textbf{nu} se obțin prin optimizare numerică: ele sînt corelații de selecție (secțiunea 4)'),
     [T('the DCC optimiser works on only 2 parameters, $a$ and $b$, whatever $N$', 'optimizarea DCC lucrează doar cu 2 parametri, $a$ și $b$, oricare ar fi $N$')]),
    T('The price of parsimony: all pairs share the same correlation dynamics', 'Prețul parcimoniei: toate perechile au aceeași dinamică a corelației')])

D.recap(('VEC and BEKK', 'VEC și BEKK'), [
    T('VEC: the most general linear model; $k + 2k^2$ parameters, no positive-definiteness guarantee', 'VEC: cel mai general model liniar; $k + 2k^2$ parametri, fără garanția pozitiv definirii'),
    T('BEKK: positive definite by construction; off-diagonal $a_{ij}$ measure volatility spillovers', 'BEKK: pozitiv definit prin construcție; elementele $a_{ij}$ din afara diagonalei măsoară transmiterea volatilității'),
    T('Diagonal and scalar versions cut the parameters but remove the spillovers', 'Versiunile diagonală și scalară reduc numărul de parametri, dar elimină transmiterea volatilității')])

# =============================================================================
# 4. CCC ȘI DCC
# =============================================================================
D.section('Constant and dynamic conditional correlations', 'Corelații condiționate constante și dinamice')

D.frame(T('Splitting volatilities and correlations', 'Separarea volatilităților de corelații'), items(
    (T('Decomposition: $\\mathbf{H}_t = \\mathbf{D}_t\\mathbf{R}_t\\mathbf{D}_t$, \\quad $\\mathbf{D}_t = \\mathrm{diag}(\\sigma_{1,t}, \\dots, \\sigma_{N,t})$',
       'Descompunerea: $\\mathbf{H}_t = \\mathbf{D}_t\\mathbf{R}_t\\mathbf{D}_t$, \\quad $\\mathbf{D}_t = \\mathrm{diag}(\\sigma_{1,t}, \\dots, \\sigma_{N,t})$'),
     [T('$\\mathbf{R}_t$: the conditional correlation matrix (ones on the diagonal); element by element: $h_{ij,t} = \\rho_{ij,t}\\,\\sigma_{i,t}\\sigma_{j,t}$', '$\\mathbf{R}_t$: matricea corelațiilor condiționate (cu 1 pe diagonală); element cu element: $h_{ij,t} = \\rho_{ij,t}\\,\\sigma_{i,t}\\sigma_{j,t}$'),
      T('each $\\sigma_{i,t}$ is a univariate GARCH(1,1) \\refBollG: $\\sigma_{i,t}^2 = \\omega_i + \\alpha_i\\varepsilon_{i,t-1}^2 + \\beta_i\\sigma_{i,t-1}^2$', 'fiecare $\\sigma_{i,t}$ este un GARCH(1,1) univariat \\refBollG: $\\sigma_{i,t}^2 = \\omega_i + \\alpha_i\\varepsilon_{i,t-1}^2 + \\beta_i\\sigma_{i,t-1}^2$')]),
    (T('\\textbf{Standardised residuals}: $z_{i,t} = \\varepsilon_{i,t}/\\sigma_{i,t}$; their conditional covariance matrix is exactly $\\mathbf{R}_t$', '\\textbf{Reziduurile standardizate}: $z_{i,t} = \\varepsilon_{i,t}/\\sigma_{i,t}$; matricea lor de covarianță condiționată este chiar $\\mathbf{R}_t$'), []),
    (T('\\textbf{CCC} (constant conditional correlation, \\refBoll): $\\mathbf{R}_t = \\mathbf{R}$ for all $t$', '\\textbf{CCC} (corelație condiționată constantă, \\refBoll): $\\mathbf{R}_t = \\mathbf{R}$ pentru orice $t$'),
     [T('$\\mathbf{H}_t$ is positive definite whenever $\\mathbf{R}$ is; $\\mathbf{R}$ is estimated by the sample correlation of the $z_{i,t}$', '$\\mathbf{H}_t$ este pozitiv definită ori de cîte ori $\\mathbf{R}$ este; $\\mathbf{R}$ se estimează prin corelația de selecție a lui $z_{i,t}$'),
      T('covariances still move, but only through the volatilities; the rolling correlations of Section 1 reject this', 'covarianțele se mișcă totuși, dar numai prin volatilități; corelațiile pe ferestre mobile din secțiunea 1 contrazic această ipoteză')])))

D.frame(T('Robert Engle and the DCC model', 'Robert Engle și modelul DCC'), two(
    ph('engle', T('Robert F. Engle (b.\\ 1942), Nobel Prize 2003', 'Robert F. Engle (n.\\ 1942), Premiul Nobel 2003'), h='0.42\\textheight'),
    items((T('\\refEngle: let the correlations follow a GARCH-like recursion', '\\refEngle: corelațiile urmează o recurență de tip GARCH'),
           [T('$\\mathbf{Q}_t = (1 - a - b)\\,\\bar{\\mathbf{Q}} + a\\,\\mathbf{z}_{t-1}\\mathbf{z}_{t-1}^\\top + b\\,\\mathbf{Q}_{t-1}$', '$\\mathbf{Q}_t = (1 - a - b)\\,\\bar{\\mathbf{Q}} + a\\,\\mathbf{z}_{t-1}\\mathbf{z}_{t-1}^\\top + b\\,\\mathbf{Q}_{t-1}$'),
            T('$\\rho_{ij,t} = q_{ij,t}/\\sqrt{q_{ii,t}\\,q_{jj,t}}$: the rescaling puts ones on the diagonal', '$\\rho_{ij,t} = q_{ij,t}/\\sqrt{q_{ii,t}\\,q_{jj,t}}$: rescalarea pune 1 pe diagonală')]),
          (T('$\\bar{\\mathbf{Q}}$: the unconditional correlation of $\\mathbf{z}_t$, set to its sample value (\\textbf{correlation targeting})', '$\\bar{\\mathbf{Q}}$: corelația necondiționată a lui $\\mathbf{z}_t$, fixată la valoarea de selecție (\\textbf{țintirea corelației})'),
           [T('$a \\ge 0$: reaction to yesterday\'s joint shock; $b \\ge 0$: persistence; $a + b < 1$: mean reversion to $\\bar{\\mathbf{Q}}$', '$a \\ge 0$: reacția la șocul comun de ieri; $b \\ge 0$: persistența; $a + b < 1$: revenire la $\\bar{\\mathbf{Q}}$')]),
          T('$a = b = 0$ gives back CCC', '$a = b = 0$ readuce modelul CCC')), wl='0.33', wr='0.65'), size='footnotesize')

D.frame(T('Two-step estimation', 'Estimarea în doi pași'), items(
    (T('The log-likelihood splits into a volatility part and a correlation part: $\\ell = \\ell_V(\\theta_1) + \\ell_C(\\theta_1, a, b)$', 'Logaritmul verosimilității se descompune într-o parte de volatilitate și o parte de corelație: $\\ell = \\ell_V(\\theta_1) + \\ell_C(\\theta_1, a, b)$'), []),
    (T('\\textbf{Step 1}: fit a GARCH(1,1) to each series separately (Chapter 5); keep $\\hat\\sigma_{i,t}$ and $\\hat z_{i,t} = \\hat\\varepsilon_{i,t}/\\hat\\sigma_{i,t}$',
       '\\textbf{Pasul 1}: estimați cîte un GARCH(1,1) pentru fiecare serie separat (Capitolul 5); păstrați $\\hat\\sigma_{i,t}$ și $\\hat z_{i,t} = \\hat\\varepsilon_{i,t}/\\hat\\sigma_{i,t}$'), []),
    (T('\\textbf{Step 2}: set $\\bar{\\mathbf{Q}}$ to the sample correlation of $\\hat{\\mathbf{z}}_t$ and maximise over $(a, b)$ only', '\\textbf{Pasul 2}: fixați $\\bar{\\mathbf{Q}}$ la corelația de selecție a lui $\\hat{\\mathbf{z}}_t$ și maximizați doar după $(a, b)$'),
     [T('$\\ell_C(a, b) = -\\frac12\\sum_t\\left(\\log|\\mathbf{R}_t| + \\hat{\\mathbf{z}}_t^\\top\\mathbf{R}_t^{-1}\\hat{\\mathbf{z}}_t - \\hat{\\mathbf{z}}_t^\\top\\hat{\\mathbf{z}}_t\\right)$', '$\\ell_C(a, b) = -\\frac12\\sum_t\\left(\\log|\\mathbf{R}_t| + \\hat{\\mathbf{z}}_t^\\top\\mathbf{R}_t^{-1}\\hat{\\mathbf{z}}_t - \\hat{\\mathbf{z}}_t^\\top\\hat{\\mathbf{z}}_t\\right)$')]),
    (T('Consistent, fast, works for large $N$; the step-2 standard errors ignore the estimation error of step 1 (they are too small)', 'Estimatorul este consistent, rapid și funcționează pentru $N$ mare; erorile standard din pasul 2 ignoră eroarea de estimare din pasul 1 (sînt prea mici)'), []),
    (T('\\textbf{Test against CCC}: likelihood ratio $LR = 2(\\ell_C^{DCC} - \\ell_C^{CCC})$, compared with $\\chi^2(2)$ (5\\% critical value @{dc.crit}); also \\refTse',
       '\\textbf{Test față de CCC}: raportul de verosimilitate $LR = 2(\\ell_C^{DCC} - \\ell_C^{CCC})$, comparat cu $\\chi^2(2)$ (valoarea critică la 5\\%: @{dc.crit}); vezi și \\refTse'),
     [T('$a = b = 0$ lies on the boundary of the parameter space, so the $\\chi^2(2)$ critical value is conservative', '$a = b = 0$ se află pe frontiera spațiului parametrilor, deci valoarea critică $\\chi^2(2)$ este conservatoare')])), size='footnotesize')

D.frame(T('Worked example: one step of the DCC recursion', 'Exemplu rezolvat: un pas al recurenței DCC'), items(
    (T('Given: $a = 0.05$, $b = 0.93$, $\\bar q_{12} = 0.5$ (and $\\bar q_{11} = \\bar q_{22} = 1$), $\\mathbf{Q}_{t-1} = \\begin{pmatrix}1 & 0.5\\\\ 0.5 & 1\\end{pmatrix}$, yesterday $z_{1,t-1} = -2$, $z_{2,t-1} = -2.5$',
       'Date: $a = 0{,}05$, $b = 0{,}93$, $\\bar q_{12} = 0{,}5$ (și $\\bar q_{11} = \\bar q_{22} = 1$), $\\mathbf{Q}_{t-1} = \\begin{pmatrix}1 & 0{,}5\\\\ 0{,}5 & 1\\end{pmatrix}$, ieri $z_{1,t-1} = -2$, $z_{2,t-1} = -2{,}5$'), []),
    (T('Step 1: $1 - a - b = @{ex.c}$', 'Pasul 1: $1 - a - b = @{ex.c}$'), []),
    (T('Step 2: $q_{12,t} = @{ex.c} \\cdot 0.5 + 0.05 \\cdot (-2)(-2.5) + 0.93 \\cdot 0.5 = @{ex.t1} + @{ex.t2} + @{ex.t3} = @{ex.q12}$',
       'Pasul 2: $q_{12,t} = @{ex.c} \\cdot 0{,}5 + 0{,}05 \\cdot (-2)(-2{,}5) + 0{,}93 \\cdot 0{,}5 = @{ex.t1} + @{ex.t2} + @{ex.t3} = @{ex.q12}$'), []),
    (T('Step 3: $q_{11,t} = @{ex.c} + 0.05 \\cdot 4 + 0.93 = @{ex.q11}$; \\quad $q_{22,t} = @{ex.c} + 0.05 \\cdot 6.25 + 0.93 = @{ex.q22}$',
       'Pasul 3: $q_{11,t} = @{ex.c} + 0{,}05 \\cdot 4 + 0{,}93 = @{ex.q11}$; \\quad $q_{22,t} = @{ex.c} + 0{,}05 \\cdot 6{,}25 + 0{,}93 = @{ex.q22}$'), []),
    (T('Step 4: $\\rho_{12,t} = @{ex.q12}/\\sqrt{@{ex.q11} \\cdot @{ex.q22}} = @{ex.rho}$', 'Pasul 4: $\\rho_{12,t} = @{ex.q12}/\\sqrt{@{ex.q11} \\cdot @{ex.q22}} = @{ex.rho}$'),
     [T('a large joint fall raises tomorrow\'s correlation from 0.50 to @{ex.rho}; without step 3 the ``correlation\'\' $q_{12,t}$ would be @{ex.q12}', 'o scădere comună mare crește corelația de mîine de la 0,50 la @{ex.rho}; fără pasul 3, „corelația” $q_{12,t}$ ar fi @{ex.q12}')])), size='footnotesize')

D.frame(T('Reading $a$ and $b$', 'Interpretarea parametrilor $a$ și $b$'), items(
    (T('$a$ is the \\textbf{news} coefficient: how much one day\'s joint shock moves the correlation; typical daily values 0.01--0.05', '$a$ este coeficientul de \\textbf{știri}: cît mișcă un șoc comun de o zi corelația; valori zilnice tipice: 0,01--0,05'), []),
    (T('$a + b$ is the \\textbf{persistence}: a deviation of $q_{ij,t}$ from $\\bar q_{ij}$ shrinks by the factor $a + b$ each day', '$a + b$ este \\textbf{persistența}: o abatere a lui $q_{ij,t}$ de la $\\bar q_{ij}$ se reduce zilnic cu factorul $a + b$'),
     [T('half-life: $\\ln 0.5/\\ln(a + b)$ days; $a + b = 0.98$ gives 34 days, $0.99$ gives 69 days', 'timpul de înjumătățire: $\\ln 0{,}5/\\ln(a + b)$ zile; $a + b = 0{,}98$ dă 34 de zile, $0{,}99$ dă 69 de zile')]),
    (T('The same reading as $\\alpha$ and $\\alpha + \\beta$ in a GARCH(1,1) (Chapter 5)', 'Aceeași interpretare ca pentru $\\alpha$ și $\\alpha + \\beta$ într-un GARCH(1,1) (Capitolul 5)'), []),
    (T('Variants: asymmetric DCC (joint falls move correlations more than joint rises) \\refCES; corrected DCC \\refAielli', 'Variante: DCC asimetric (scăderile comune mișcă mai mult corelațiile decît creșterile comune) \\refCES; DCC corectat \\refAielli'), [])))

D.recap(('CCC and DCC', 'CCC și DCC'), [
    T('$\\mathbf{H}_t = \\mathbf{D}_t\\mathbf{R}_t\\mathbf{D}_t$: GARCH volatilities times a correlation matrix', '$\\mathbf{H}_t = \\mathbf{D}_t\\mathbf{R}_t\\mathbf{D}_t$: volatilități GARCH combinate cu o matrice de corelații'),
    T('CCC: $\\mathbf{R}$ constant; DCC: $\\mathbf{Q}_t$ follows a GARCH-like recursion with parameters $a$, $b$', 'CCC: $\\mathbf{R}$ constantă; DCC: $\\mathbf{Q}_t$ urmează o recurență de tip GARCH, cu parametrii $a$, $b$'),
    T('Two steps: univariate GARCH, then $(a, b)$ with correlation targeting; LR test against CCC', 'Doi pași: GARCH univariat, apoi $(a, b)$ cu țintirea corelației; testul LR față de CCC')])

# =============================================================================
# 5. S&P 500 ȘI DAX
# =============================================================================
D.section('DCC for the S\\&P 500 and the DAX', 'DCC pentru S\\&P 500 și DAX')

D.frame(T('Step 1: two univariate GARCH models', 'Pasul 1: două modele GARCH univariate'), table(
    'lrrrrrrr', T('\\textbf{Index}', '\\textbf{Indice}') + ' & $\\mu$ & $\\omega$ & $\\alpha$ & $\\beta$ & $\\alpha + \\beta$ & ' + T('kurt.\\ $r_t$', 'boltire $r_t$') + ' & ' + T('kurt.\\ $z_t$', 'boltire $z_t$'),
    ['S\\&P 500 & @{s1.sp500.mu} & @{s1.sp500.om} & @{s1.sp500.al} & @{s1.sp500.be} & @{s1.sp500.pers} & @{s1.sp500.rk} & @{s1.sp500.zk}',
     'DAX & @{s1.dax.mu} & @{s1.dax.om} & @{s1.dax.al} & @{s1.dax.be} & @{s1.dax.pers} & @{s1.dax.rk} & @{s1.dax.zk}'], size='small')
    + items(T('Daily log returns in \\%, @{s1.n} common trading days, @{s1.first}--@{s1.last} (EODHD); GARCH(1,1) with a constant mean, Gaussian QML; kurt.: excess kurtosis',
              'Randamente logaritmice zilnice în \\%, @{s1.n} zile comune de tranzacționare, @{s1.first}--@{s1.last} (EODHD); GARCH(1,1) cu medie constantă, QML gaussian; boltire: excesul de boltire'),
            T('Standardising by $\\hat\\sigma_{i,t}$ removes most of the excess kurtosis, but not all: the tails of $z_t$ are still fatter than Normal',
              'Standardizarea cu $\\hat\\sigma_{i,t}$ elimină cea mai mare parte a excesului de boltire, dar nu tot: cozile lui $z_t$ rămîn mai groase decît la distribuția Normală'),
            T('Correlation of the returns @{s1.rcorr}; correlation of the residuals $\\bar q_{12} = @{s1.zcorr}$, the CCC estimate', 'Corelația randamentelor @{s1.rcorr}; corelația reziduurilor $\\bar q_{12} = @{s1.zcorr}$, estimația CCC')),
    size='footnotesize')

chart(T('Step 1: conditional volatilities', 'Pasul 1: volatilitățile condiționate'), 'tsa_ch14_garch_step1', 'TSA_ch14_dcc_estimation', [
    T('One-step-ahead GARCH(1,1) volatility $\\hat\\sigma_{i,t}\\sqrt{252}$, annualised, in \\%; shaded: the crises of 2008 and 2020', 'Volatilitatea GARCH(1,1) prognozată cu un pas înainte, $\\hat\\sigma_{i,t}\\sqrt{252}$, anualizată, în \\%; zonele colorate: crizele din 2008 și 2020')],
    h='0.62\\textheight')

interp(('the volatilities', 'volatilităților'), [
    (T('Both volatilities peak in the same weeks: S\\&P 500 @{s1.sp500.vmax}\\% on @{s1.sp500.dmax}, DAX @{s1.dax.vmax}\\% on @{s1.dax.dmax}', 'Ambele volatilități ating maximul în aceleași săptămîni: S\\&P 500 @{s1.sp500.vmax}\\% pe @{s1.sp500.dmax}, DAX @{s1.dax.vmax}\\% pe @{s1.dax.dmax}'),
     [T('volatility clustering happens \\textbf{jointly}: the covariance rises even with a constant correlation', 'volatility clustering apare \\textbf{simultan}: covarianța crește chiar și cu o corelație constantă')]),
    (T('Persistence @{s1.sp500.pers} and @{s1.dax.pers}: volatility shocks fade slowly, as in Chapter 5', 'Persistența @{s1.sp500.pers} și @{s1.dax.pers}: șocurile de volatilitate se sting lent, ca în Capitolul 5'), []),
    T('Step 2 asks a different question: once the volatilities are removed, is the remaining correlation constant?', 'Pasul 2 pune o altă întrebare: după eliminarea volatilităților, este corelația rămasă constantă?')])

D.frame(T('Step 2: the estimated DCC', 'Pasul 2: modelul DCC estimat'), table(
    'lrr', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & \\textbf{' + T('Estimate', 'Estimație') + '} & \\textbf{SE}',
    ['$a$ & @{dc.a} & @{dc.se_a}', '$b$ & @{dc.b} & @{dc.se_b}',
     '$a + b$ & @{dc.ab} & ',
     T('Half-life of a correlation shock (days)', 'Timpul de înjumătățire al unui șoc de corelație (zile)') + ' & @{dc.hl} & ',
     T('LR against CCC', 'LR față de CCC') + ' & @{dc.lr} & ',
     T('$\\chi^2(2)$ critical value, 5\\%', 'Valoarea critică $\\chi^2(2)$, 5\\%') + ' & @{dc.crit} & '], size='small')
    + items(T('$a$ is small but well determined; $a + b$ is close to 1: correlation shocks last for months', '$a$ este mic, dar bine determinat; $a + b$ este aproape de 1: șocurile de corelație durează luni'),
            T('LR = @{dc.lr} is far above @{dc.crit}: constant correlation is rejected', 'LR = @{dc.lr} este mult peste @{dc.crit}: corelația constantă este respinsă'),
            T('SE: standard errors from the step-2 Hessian (too small, they ignore step 1)', 'SE: erori standard din hessiana pasului 2 (prea mici, ignoră pasul 1)')),
    size='footnotesize')

chart(T('DCC correlation of the S\\&P 500 and the DAX', 'Corelația DCC dintre S\\&P 500 și DAX'), 'tsa_ch14_dcc_sp_dax', 'TSA_ch14_dcc_estimation', [
    T('One-step-ahead DCC correlation $\\hat\\rho_{12,t}$, the 250-day rolling correlation and the CCC value $\\bar q_{12}$; daily, @{s1.first}--@{s1.last}', 'Corelația DCC prognozată cu un pas înainte, $\\hat\\rho_{12,t}$, corelația pe ferestre mobile de 250 de zile și valoarea CCC $\\bar q_{12}$; date zilnice, @{s1.first}--@{s1.last}')],
    h='0.62\\textheight')

interp(('the DCC correlation', 'corelației DCC'), [
    (T('The DCC correlation ranges from @{dc.min} (@{dc.dmin}) to @{dc.max} (@{dc.dmax}), around a mean of @{dc.mean}', 'Corelația DCC variază între @{dc.min} (@{dc.dmin}) și @{dc.max} (@{dc.dmax}), în jurul unei medii de @{dc.mean}'), []),
    (T('It follows the rolling correlation, but reacts within days instead of a year and has no window to choose', 'Urmează corelația pe ferestre mobile, dar reacționează în cîteva zile, nu într-un an, și nu cere alegerea unei ferestre'),
     [T('the rolling estimate is more erratic (standard deviation @{dc.sd_roll} against @{dc.sd_dcc}) and jumps when an extreme day enters or leaves the window', 'estimarea pe ferestre mobile este mai instabilă (abaterea standard @{dc.sd_roll} față de @{dc.sd_dcc}) și sare cînd o zi extremă intră în fereastră sau iese din ea')]),
    T('The rolling estimate falls sharply in 2025--2026: the asynchronous days of April 2025 (Section 1) pull it down for exactly 250 days', 'Estimarea pe ferestre mobile scade brusc în 2025--2026: zilele asincrone din aprilie 2025 (secțiunea 1) o trag în jos exact 250 de zile')])

D.frame(T('Correlations in crises', 'Corelațiile în crize'), two(
    ph('lehman', T('Lehman Brothers, New York, 15 September 2008', 'Lehman Brothers, New York, 15 septembrie 2008'), h='0.36\\textheight'),
    table('lrr', T('\\textbf{Period}', '\\textbf{Perioada}') + ' & $\\bar\\rho^{DCC}$ & $\\bar\\sigma^{S\\&P}$ (\\%)',
          [T('Calm, 2017--2019', 'Calmă, 2017--2019') + ' & @{dc.mean_calm} & @{dc.vol_calm}',
           T('Sept 2008--March 2009', 'sept.\\ 2008--martie 2009') + ' & @{dc.mean_2008} & @{dc.vol_2008}',
           T('Feb--April 2020', 'febr.--aprilie 2020') + ' & @{dc.mean_2020} & @{dc.vol_2020}'], size='footnotesize')
    + items(T('Average DCC correlation and average annualised S\\&P 500 volatility in each period', 'Corelația DCC medie și volatilitatea anualizată medie a S\\&P 500 în fiecare perioadă'),
            T('Correlations are higher in crises, but volatility rises far more', 'Corelațiile sînt mai mari în crize, dar volatilitatea crește mult mai mult'),
            T('Caution \\refFR: the sample correlation rises mechanically when volatility rises, even if the link is unchanged; DCC reduces this effect because it works with $z_t$, not with $r_t$',
              'Atenție \\refFR: corelația de selecție crește mecanic cînd crește volatilitatea, chiar dacă legătura nu s-a schimbat; DCC reduce acest efect, deoarece lucrează cu $z_t$, nu cu $r_t$')),
    wl='0.36', wr='0.62'), size='footnotesize')

chart(T('Inside the crises of 2008 and 2020', 'În interiorul crizelor din 2008 și 2020'), 'tsa_ch14_crisis_zoom', 'TSA_ch14_dcc_estimation', [
    T('DCC correlation of the S\\&P 500 and the DAX (left axis) and the GARCH volatility of the S\\&P 500 (right axis)', 'Corelația DCC dintre S\\&P 500 și DAX (axa din stînga) și volatilitatea GARCH a S\\&P 500 (axa din dreapta)')],
    h='0.6\\textheight')

interp(('the two crises', 'celor două crize'), [
    (T('2008: volatility peaks at @{z2008.vmax}\\% on @{z2008.vdmax}, while the correlation drifts up from @{z2008.r0} to @{z2008.rmax} only by @{z2008.rdmax}', '2008: volatilitatea atinge @{z2008.vmax}\\% pe @{z2008.vdmax}, iar corelația urcă lent de la @{z2008.r0} la @{z2008.rmax} abia pe @{z2008.rdmax}'), []),
    (T('2020: the correlation jumps to @{z2020.rmax} on @{z2020.rdmax}, before the volatility peak (@{z2020.vmax}\\% on @{z2020.vdmax})', '2020: corelația sare la @{z2020.rmax} pe @{z2020.rdmax}, înaintea maximului volatilității (@{z2020.vmax}\\% pe @{z2020.vdmax})'),
     [T('a global shock hit both markets on the same days: joint falls drive the DCC recursion', 'un șoc global a lovit ambele piețe în aceleași zile: scăderile comune alimentează recurența DCC')]),
    T('Correlations and volatilities need not peak together: modelling them separately, as DCC does, shows the difference', 'Corelațiile și volatilitățile nu ating neapărat maximul în același timp: modelarea lor separată, ca în DCC, arată diferența')])

D.recap(('S\\&P 500 and DAX', 'S\\&P 500 și DAX'), [
    T('Step 1: persistent GARCH volatilities that peak together', 'Pasul 1: volatilități GARCH persistente, cu maxime simultane'),
    T('Step 2: $a + b = @{dc.ab}$; LR @{dc.lr}: the correlation is not constant', 'Pasul 2: $a + b = @{dc.ab}$; LR @{dc.lr}: corelația nu este constantă'),
    T('Higher correlations in 2008 and 2020, but the largest change is in volatility', 'Corelații mai mari în 2008 și 2020, dar cea mai mare schimbare apare în volatilitate')])

# =============================================================================
# 6. CINCI PIEȚE
# =============================================================================
D.section('Five markets: S\\&P 500, DAX, BET, EUR/RON, Bitcoin', 'Cinci piețe: S\\&P 500, DAX, BET, EUR/RON, Bitcoin')

D.frame(T('A five-asset DCC', 'Un model DCC cu cinci active'), two(
    ph('bvb', T('The Bucharest Stock Exchange', 'Bursa de Valori București'), h='0.3\\textheight'),
    items((T('Daily log returns on the @{pa.n} days on which all five markets trade, from @{pa.first}', 'Randamente logaritmice zilnice în cele @{pa.n} zile în care se tranzacționează toate cele cinci piețe, din @{pa.first}'),
           [T('EUR/RON: the BNR reference rate (fixed at 13:00 Bucharest time); BET: Bucharest Stock Exchange; Bitcoin: close of the day (UTC)', 'EUR/RON: cursul de referință BNR (stabilit la ora 13:00, ora Bucureștiului); BET: Bursa de Valori București; Bitcoin: închiderea zilei (UTC)')]),
          (T('Step 1: GARCH persistence S\\&P 500 @{pa.sp500.pers}, DAX @{pa.dax.pers}, BET @{pa.bet.pers}, EUR/RON @{pa.eurron.pers}, Bitcoin @{pa.btc.pers}', 'Pasul 1: persistența GARCH S\\&P 500 @{pa.sp500.pers}, DAX @{pa.dax.pers}, BET @{pa.bet.pers}, EUR/RON @{pa.eurron.pers}, Bitcoin @{pa.btc.pers}'), []),
          (T('Step 2: $\\hat a = @{pa.a}$ (SE @{pa.se_a}), $\\hat b = @{pa.b}$ (SE @{pa.se_b}); half-life @{pa.hl} days; LR against CCC @{pa.lr}', 'Pasul 2: $\\hat a = @{pa.a}$ (SE @{pa.se_a}), $\\hat b = @{pa.b}$ (SE @{pa.se_b}); timpul de înjumătățire @{pa.hl} zile; LR față de CCC @{pa.lr}'),
           [T('10 correlation pairs, one pair of dynamics parameters $(a, b)$ for all of them', '10 perechi de corelații, o singură pereche de parametri de dinamică $(a, b)$ pentru toate')]))), size='footnotesize')

chart(T('Dynamic correlations of four pairs', 'Corelațiile dinamice a patru perechi'), 'tsa_ch14_panel_corr', 'TSA_ch14_dcc_panel', [
    T('One-step-ahead DCC correlations from the five-asset model; shaded: the COVID-19 crash, mid-February--June 2020', 'Corelațiile DCC prognozate cu un pas înainte din modelul cu cinci active; zona colorată: crahul COVID-19, jumătatea lui februarie--iunie 2020')],
    h='0.62\\textheight')

interp(('the four pairs', 'celor patru perechi'), [
    (T('S\\&P 500 and DAX: the strongest link, mean @{pa.sp500_dax.mean}; BET and DAX: mean @{pa.bet_dax.mean}, between @{pa.bet_dax.min} and @{pa.bet_dax.max} (@{pa.bet_dax.dmax})', 'S\\&P 500 și DAX: legătura cea mai puternică, media @{pa.sp500_dax.mean}; BET și DAX: media @{pa.bet_dax.mean}, între @{pa.bet_dax.min} și @{pa.bet_dax.max} (@{pa.bet_dax.dmax})'), []),
    (T('S\\&P 500 and Bitcoin: from @{pa.sp500_btc.min} to @{pa.sp500_btc.max} (@{pa.sp500_btc.dmax}); mean @{pa.sp500_btc.mean}', 'S\\&P 500 și Bitcoin: de la @{pa.sp500_btc.min} la @{pa.sp500_btc.max} (@{pa.sp500_btc.dmax}); media @{pa.sp500_btc.mean}'),
     [T('the diversification benefit of Bitcoin shrank after 2020', 'beneficiul de diversificare al lui Bitcoin s-a redus după 2020')]),
    (T('BET and EUR/RON: close to zero (mean @{pa.bet_eurron.mean}); a weaker leu tends to come with a weaker BET, a small negative link', 'BET și EUR/RON: aproape de zero (media @{pa.bet_eurron.mean}); un leu mai slab tinde să însoțească un BET mai slab, o legătură negativă mică'), []),
    T('One $(a, b)$ for all pairs: pairs with different speeds are forced to the same rhythm, the price of DCC\'s parsimony', 'Un singur $(a, b)$ pentru toate perechile: perechi cu viteze diferite sînt forțate la același ritm, prețul parcimoniei DCC')])

chart(T('Average correlations: calm against crisis', 'Corelațiile medii: perioadă calmă și criză'), 'tsa_ch14_panel_heatmap', 'TSA_ch14_dcc_panel', [
    T('Average one-step-ahead DCC correlation matrix: 2017--2019 and mid-February--June 2020 (@{ht.nc} common trading days)', 'Matricea medie a corelațiilor DCC prognozate cu un pas înainte: 2017--2019 și jumătatea lui februarie--iunie 2020 (@{ht.nc} de zile comune de tranzacționare)')],
    h='0.6\\textheight')

interp(('the two matrices', 'celor două matrice'), [
    (T('The average pairwise correlation rises from @{ht.calm} to @{ht.cris}', 'Corelația medie dintre perechi crește de la @{ht.calm} la @{ht.cris}'),
     [T('BET and DAX: @{ht.bet_dax.c} $\\to$ @{ht.bet_dax.k}; S\\&P 500 and BET: @{ht.sp500_bet.c} $\\to$ @{ht.sp500_bet.k}; DAX and Bitcoin: @{ht.dax_btc.c} $\\to$ @{ht.dax_btc.k}', 'BET și DAX: @{ht.bet_dax.c} $\\to$ @{ht.bet_dax.k}; S\\&P 500 și BET: @{ht.sp500_bet.c} $\\to$ @{ht.sp500_bet.k}; DAX și Bitcoin: @{ht.dax_btc.c} $\\to$ @{ht.dax_btc.k}')]),
    (T('The largest rises are for the BET and for Bitcoin, which are weakly linked to the large markets in calm times: exactly the assets that were supposed to diversify', 'Cele mai mari creșteri apar pentru BET și pentru Bitcoin, slab legate de piețele mari în perioadele calme: tocmai activele care ar fi trebuit să diversifice riscul'), []),
    T('EUR/RON stays near zero or slightly negative with all equity markets: the only diversifier in this set', 'EUR/RON rămîne aproape de zero sau ușor negativ față de toate piețele de acțiuni: singurul element de diversificare din acest set')])

D.recap(('Five markets', 'cinci piețe'), [
    T('One DCC for 5 assets: 2 dynamics parameters for 10 correlations', 'Un singur DCC pentru 5 active: 2 parametri de dinamică pentru 10 corelații'),
    T('Correlations of the Romanian market with Europe and of Bitcoin with equities move a lot', 'Corelațiile pieței românești cu Europa și ale lui Bitcoin cu acțiunile se mișcă mult'),
    T('In the 2020 crash the weak links strengthened most: diversification weakens in crises', 'În crahul din 2020 legăturile slabe s-au întărit cel mai mult: diversificarea slăbește în crize')])

# =============================================================================
# 7. VaR 1%
# =============================================================================
D.section('Portfolio VaR 1\\% with DCC', 'VaR 1\\% al unui portofoliu cu DCC')

D.frame(T('From $\\mathbf{H}_t$ to the VaR of a portfolio', 'De la $\\mathbf{H}_t$ la VaR-ul unui portofoliu'), items(
    (T('Portfolio return $r_{p,t} = \\mathbf{w}^\\top\\mathbf{r}_t$; conditional mean $\\mu_p = \\mathbf{w}^\\top\\boldsymbol{\\mu}$ and volatility $\\sigma_{p,t} = \\sqrt{\\mathbf{w}^\\top\\mathbf{H}_t\\mathbf{w}}$',
       'Randamentul portofoliului $r_{p,t} = \\mathbf{w}^\\top\\mathbf{r}_t$; media condiționată $\\mu_p = \\mathbf{w}^\\top\\boldsymbol{\\mu}$ și volatilitatea $\\sigma_{p,t} = \\sqrt{\\mathbf{w}^\\top\\mathbf{H}_t\\mathbf{w}}$'), []),
    (T('\\textbf{VaR 1\\%}: the loss exceeded with probability 1\\%, $\\mathrm{VaR}_{0.01,t} = -q_{0.01}(r_{p,t}) = -(\\mu_p + q\\,\\sigma_{p,t})$',
       '\\textbf{VaR 1\\%}: pierderea depășită cu probabilitatea 1\\%, $\\mathrm{VaR}_{0{,}01,t} = -q_{0{,}01}(r_{p,t}) = -(\\mu_p + q\\,\\sigma_{p,t})$'),
     [T('Normal: $q = z_{0.01} = -@{ex.z}$', 'distribuția Normală: $q = z_{0{,}01} = -@{ex.z}$'),
      T('\\textbf{filtered historical simulation} (FHS): $q$ = the empirical 1\\% quantile of the past standardised portfolio returns $(r_{p,t} - \\mu_p)/\\sigma_{p,t}$', '\\textbf{simularea istorică filtrată} (FHS): $q$ = cuantila empirică de 1\\% a randamentelor standardizate trecute ale portofoliului, $(r_{p,t} - \\mu_p)/\\sigma_{p,t}$')]),
    (T('Three choices of $\\mathbf{H}_t$: DCC; CCC (GARCH volatilities with a constant correlation); \\textbf{static} (one sample covariance matrix for all days)',
       'Trei variante pentru $\\mathbf{H}_t$: DCC; CCC (volatilități GARCH cu o corelație constantă); \\textbf{statică} (o singură matrice de covarianță de selecție pentru toate zilele)'), [])))

D.frame(T('Worked example: VaR 1\\% of two assets', 'Exemplu rezolvat: VaR 1\\% pentru două active'), items(
    (T('Today\'s forecasts: $\\sigma_1 = 1.5\\%$, $\\sigma_2 = 1.2\\%$ (daily), weights 50\\% and 50\\%, mean 0', 'Prognozele de azi: $\\sigma_1 = 1{,}5\\%$, $\\sigma_2 = 1{,}2\\%$ (zilnic), ponderi 50\\% și 50\\%, media 0'), []),
    (T('Calm, $\\rho_t = 0.3$: $\\sigma_p^2 = 0.25 \\cdot 2.25 + 0.25 \\cdot 1.44 + 2 \\cdot 0.25 \\cdot 0.3 \\cdot 1.8$, so $\\sigma_p = @{ex.sp3}\\%$ and VaR 1\\% $= @{ex.z} \\cdot @{ex.sp3} = @{ex.var3}\\%$',
       'Calm, $\\rho_t = 0{,}3$: $\\sigma_p^2 = 0{,}25 \\cdot 2{,}25 + 0{,}25 \\cdot 1{,}44 + 2 \\cdot 0{,}25 \\cdot 0{,}3 \\cdot 1{,}8$, deci $\\sigma_p = @{ex.sp3}\\%$ și VaR 1\\% $= @{ex.z} \\cdot @{ex.sp3} = @{ex.var3}\\%$'), []),
    (T('Crisis, $\\rho_t = 0.9$ (the same volatilities): $\\sigma_p = @{ex.sp9}\\%$ and VaR 1\\% $= @{ex.var9}\\%$', 'Criză, $\\rho_t = 0{,}9$ (aceleași volatilități): $\\sigma_p = @{ex.sp9}\\%$ și VaR 1\\% $= @{ex.var9}\\%$'),
     [T('the jump of the correlation alone adds a fifth to the VaR; in a real crisis the volatilities rise too', 'doar saltul corelației adaugă o cincime la VaR; într-o criză reală cresc și volatilitățile')]),
    (T('For a portfolio of 1 million lei: VaR 1\\% of about @{ex.vark3} thousand lei in calm times and @{ex.vark9} thousand lei in the crisis case', 'Pentru un portofoliu de 1 milion de lei: VaR 1\\% de aproximativ @{ex.vark3} mii de lei în perioada calmă și @{ex.vark9} mii de lei în cazul de criză'), [])))

D.frame(T('Backtesting a VaR', 'Backtesting pentru VaR'), items(
    (T('A \\textbf{violation} on day $t$: $r_{p,t} < -\\mathrm{VaR}_{0.01,t}$; a correct VaR 1\\% has violations on 1\\% of the days, at random times', 'O \\textbf{încălcare} în ziua $t$: $r_{p,t} < -\\mathrm{VaR}_{0{,}01,t}$; un VaR 1\\% corect are încălcări în 1\\% din zile, la momente aleatoare'), []),
    (T('\\textbf{Kupiec test} \\refKupiec\\ (frequency): $x$ violations in $n$ days, $\\hat p = x/n$', '\\textbf{Testul Kupiec} \\refKupiec\\ (frecvența): $x$ încălcări în $n$ zile, $\\hat p = x/n$'),
     [T('$LR_{uc} = -2\\log\\dfrac{(1-0.01)^{n-x}\\,0.01^{x}}{(1-\\hat p)^{n-x}\\,\\hat p^{x}} \\sim \\chi^2(1)$; reject at 5\\% if $LR_{uc} > 3.84$', '$LR_{uc} = -2\\log\\dfrac{(1-0{,}01)^{n-x}\\,0{,}01^{x}}{(1-\\hat p)^{n-x}\\,\\hat p^{x}} \\sim \\chi^2(1)$; respingem la 5\\% dacă $LR_{uc} > 3{,}84$')]),
    (T('\\textbf{Christoffersen test} \\refChr\\ (independence): is a violation more likely the day after a violation?', '\\textbf{Testul Christoffersen} \\refChr\\ (independența): este o încălcare mai probabilă în ziua de după o încălcare?'),
     [T('compares $\\pi_{11} = P(\\text{violation} \\mid \\text{violation yesterday})$ with $\\pi_{01} = P(\\text{violation} \\mid \\text{no violation yesterday})$', 'compară $\\pi_{11} = P(\\text{încălcare} \\mid \\text{încălcare ieri})$ cu $\\pi_{01} = P(\\text{încălcare} \\mid \\text{fără încălcare ieri})$')]),
    (T('\\textbf{Honest design}: estimate all parameters on 2000--2014 (@{vr.nin} days), then run the filters forward on 2015--2026 (@{vr.n} days) without re-estimating',
       '\\textbf{Un design corect}: estimați toți parametrii pe 2000--2014 (@{vr.nin} de zile), apoi rulați filtrele mai departe pe 2015--2026 (@{vr.n} de zile), fără reestimare'), [])), size='footnotesize')

chart(T('VaR 1\\% of an S\\&P 500 and DAX portfolio, 2015--2026', 'VaR 1\\% al unui portofoliu S\\&P 500 și DAX, 2015--2026'), 'tsa_ch14_var_backtest', 'TSA_ch14_portfolio_var', [
    T('Daily return of a 50/50 portfolio and minus VaR 1\\% from DCC with filtered historical simulation and from a static covariance matrix; triangles: violations', 'Randamentul zilnic al unui portofoliu 50/50 și minus VaR 1\\% din DCC cu simulare istorică filtrată și dintr-o matrice de covarianță statică; triunghiuri: încălcări')],
    h='0.62\\textheight')

D.frame(T('Interpreting the backtest', 'Interpretarea backtesting-ului'), table(
    'lrrrrrr', T('\\textbf{Model}', '\\textbf{Model}') + ' & ' + T('Violations', 'Încălcări') + ' & ' + T('Rate', 'Rata') + ' & $LR_{uc}$ & $p$ & ' + T('$p$ indep.', '$p$ indep.') + ' & ' + T('Feb--Apr 2020', 'febr.--apr.\\ 2020'),
    ['DCC, ' + T('Normal', 'Normală') + ' & @{vr.dcc.xn} & @{vr.dcc.rate}\\% & @{vr.dcc.lr} & @{vr.dcc.p} & @{vr.dcc.ip} & @{vr.dcc.v20}',
     'DCC, FHS & @{vr.fhs.xn} & @{vr.fhs.rate}\\% & @{vr.fhs.lr} & @{vr.fhs.p} & @{vr.fhs.ip} & @{vr.fhs.v20}',
     'CCC, ' + T('Normal', 'Normală') + ' & @{vr.ccc.xn} & @{vr.ccc.rate}\\% & @{vr.ccc.lr} & @{vr.ccc.p} & @{vr.ccc.ip} & @{vr.ccc.v20}',
     T('Static, Normal', 'Statică, Normală') + ' & @{vr.sta.xn} & @{vr.sta.rate}\\% & @{vr.sta.lr} & @{vr.sta.p} & @{vr.sta.ip} & @{vr.sta.v20}'], size='footnotesize')
    + items(T('Expected: about @{vr.exp} violations in @{vr.n} days', 'Așteptat: aproximativ @{vr.exp} de încălcări în @{vr.n} de zile'),
            T('Dynamic models with Normal quantiles have too many violations: the residuals have fat tails (Chapter 5); FHS ($q = -@{vr.qfhs}$) brings the rate down to @{vr.fhs.rate}\\%',
              'Modelele dinamice cu cuantilele distribuției Normale au prea multe încălcări: reziduurile au cozi groase (Capitolul 5); FHS ($q = -@{vr.qfhs}$) reduce rata la @{vr.fhs.rate}\\%'),
            T('The static VaR passes the frequency test, but its violations cluster: @{vr.sta.v20} in February--April 2020 alone, and after a violation the next day is a violation with probability @{vr.sta.p11}\\%',
              'VaR-ul static trece testul de frecvență, dar încălcările lui se grupează: @{vr.sta.v20} doar în februarie--aprilie 2020, iar după o încălcare ziua următoare este tot o încălcare cu probabilitatea @{vr.sta.p11}\\%'),
            T('The static VaR is too high in calm years (@{vr.sta.mv}\\% every day) and far too low in a crash; a dynamic VaR adapts in both directions',
              'VaR-ul static este prea mare în anii calmi (@{vr.sta.mv}\\% în fiecare zi) și mult prea mic într-un crah; un VaR dinamic se adaptează în ambele sensuri')),
    size='footnotesize')

D.recap(('Portfolio VaR', 'VaR-ul portofoliului'), [
    T('$\\sigma_{p,t} = \\sqrt{\\mathbf{w}^\\top\\mathbf{H}_t\\mathbf{w}}$; VaR 1\\% $= -(\\mu_p + q\\,\\sigma_{p,t})$', '$\\sigma_{p,t} = \\sqrt{\\mathbf{w}^\\top\\mathbf{H}_t\\mathbf{w}}$; VaR 1\\% $= -(\\mu_p + q\\,\\sigma_{p,t})$'),
    T('Backtest out of sample: Kupiec (frequency) and Christoffersen (independence)', 'Backtesting în afara eșantionului: Kupiec (frecvența) și Christoffersen (independența)'),
    T('Dynamic covariances fix the clustering of violations; fat tails need a better quantile than the Normal', 'Covarianțele dinamice corectează gruparea încălcărilor; cozile groase cer o cuantilă mai bună decît cea a distribuției Normale')])

# =============================================================================
# 8. ACOPERIREA RISCULUI
# =============================================================================
D.section('Dynamic hedge ratios', 'Rapoarte dinamice de acoperire')

D.frame(T('The minimum-variance hedge ratio', 'Raportul de acoperire cu varianță minimă'), items(
    (T('Hold one unit of asset $s$ (e.g. Romanian shares, the BET) and sell $h$ units of a hedging asset $f$ (e.g. a DAX futures contract)',
       'Dețineți o unitate din activul $s$ (de exemplu acțiuni românești, BET) și vindeți $h$ unități dintr-un activ de acoperire $f$ (de exemplu un contract futures pe DAX)'),
     [T('hedged return $r_{s,t} - h\\,r_{f,t}$, with variance $\\sigma_s^2 - 2h\\,\\sigma_{sf} + h^2\\sigma_f^2$', 'randamentul acoperit $r_{s,t} - h\\,r_{f,t}$, cu varianța $\\sigma_s^2 - 2h\\,\\sigma_{sf} + h^2\\sigma_f^2$')]),
    (T('Setting the derivative in $h$ to zero: $h^* = \\dfrac{\\sigma_{sf}}{\\sigma_f^2} = \\rho\\,\\dfrac{\\sigma_s}{\\sigma_f}$ \\refEd', 'Anulînd derivata în raport cu $h$: $h^* = \\dfrac{\\sigma_{sf}}{\\sigma_f^2} = \\rho\\,\\dfrac{\\sigma_s}{\\sigma_f}$ \\refEd'),
     [T('static $h^*$: the OLS slope of $r_s$ on $r_f$; \\textbf{dynamic}: $h_t^* = h_{sf,t}/h_{ff,t}$ from an MGARCH model \\refKS', '$h^*$ static: panta OLS a regresiei lui $r_s$ pe $r_f$; \\textbf{dinamic}: $h_t^* = h_{sf,t}/h_{ff,t}$ dintr-un model MGARCH \\refKS')]),
    (T('\\textbf{Hedging effectiveness}: $HE = 1 - \\Var(r_s - h\\,r_f)/\\Var(r_s)$, the share of the variance removed', '\\textbf{Eficiența acoperirii}: $HE = 1 - \\Var(r_s - h\\,r_f)/\\Var(r_s)$, ponderea varianței eliminate'),
     [T('with $h = h^*$ constant: $HE = \\rho^2$; a hedge is only as good as the correlation', 'cu $h = h^*$ constant: $HE = \\rho^2$; o acoperire este atît de bună cît este corelația')])))

D.frame(T('Worked example: a hedge ratio', 'Exemplu rezolvat: un raport de acoperire'), items(
    (T('Given: $\\sigma_s = 1.2\\%$, $\\sigma_f = 1.0\\%$ (daily), $\\rho = 0.8$', 'Date: $\\sigma_s = 1{,}2\\%$, $\\sigma_f = 1{,}0\\%$ (zilnic), $\\rho = 0{,}8$'), []),
    (T('Step 1: $\\sigma_{sf} = \\rho\\,\\sigma_s\\sigma_f = 0.8 \\cdot 1.2 \\cdot 1.0 = @{ex.h}$', 'Pasul 1: $\\sigma_{sf} = \\rho\\,\\sigma_s\\sigma_f = 0{,}8 \\cdot 1{,}2 \\cdot 1{,}0 = @{ex.h}$'), []),
    (T('Step 2: $h^* = \\sigma_{sf}/\\sigma_f^2 = @{ex.h}/1 = @{ex.h}$: sell @{ex.h} units of $f$ per unit of $s$', 'Pasul 2: $h^* = \\sigma_{sf}/\\sigma_f^2 = @{ex.h}/1 = @{ex.h}$: vindeți @{ex.h} unități din $f$ pentru fiecare unitate din $s$'), []),
    (T('Step 3: $HE = \\rho^2 = @{ex.he}$; the hedged volatility is $\\sigma_s\\sqrt{1 - \\rho^2} = @{ex.sdh}\\%$ instead of 1.2\\%', 'Pasul 3: $HE = \\rho^2 = @{ex.he}$; volatilitatea poziției acoperite este $\\sigma_s\\sqrt{1 - \\rho^2} = @{ex.sdh}\\%$ în loc de 1,2\\%'), []),
    (T('If tomorrow DCC forecasts $\\rho_t = 0.5$ and $\\sigma_{f,t} = 2\\%$, the hedge ratio becomes $0.5 \\cdot 1.2/2 = 0.3$: the position must be adjusted',
       'Dacă mîine DCC prognozează $\\rho_t = 0{,}5$ și $\\sigma_{f,t} = 2\\%$, raportul de acoperire devine $0{,}5 \\cdot 1{,}2/2 = 0{,}3$: poziția trebuie ajustată'), [])))

chart(T('Hedging the BET with the DAX, 2015--2026', 'Acoperirea BET cu DAX, 2015--2026'), 'tsa_ch14_hedge', 'TSA_ch14_hedging', [
    T('Daily returns on common trading days since 2005; all parameters estimated on 2005--2014; DCC: $h_t = \\hat\\rho_t\\hat\\sigma_{BET,t}/\\hat\\sigma_{DAX,t}$; rolling OLS: the slope over the previous 250 days',
      'Randamente zilnice în zilele comune de tranzacționare din 2005; toți parametrii estimați pe 2005--2014; DCC: $h_t = \\hat\\rho_t\\hat\\sigma_{BET,t}/\\hat\\sigma_{DAX,t}$; OLS pe ferestre mobile: panta din ultimele 250 de zile')],
    h='0.62\\textheight')

interp(('the hedge ratios', 'rapoartelor de acoperire'), [
    (T('The static ratio @{hg.sta} reflects 2005--2014, which includes the 2008 crisis; after 2015 the DCC ratio averages @{hg.dcc_mean}, from @{hg.dcc_min} to @{hg.dcc_max} (@{hg.dmax})',
       'Raportul static @{hg.sta} reflectă perioada 2005--2014, care include criza din 2008; după 2015 raportul DCC are media @{hg.dcc_mean}, între @{hg.dcc_min} și @{hg.dcc_max} (@{hg.dmax})'), []),
    (T('Hedging effectiveness out of sample (@{hg.n} days): static @{hg.he.static}\\%, rolling OLS @{hg.he.rolling}\\%, DCC @{hg.he.dcc}\\%', 'Eficiența acoperirii în afara eșantionului (@{hg.n} de zile): statică @{hg.he.static}\\%, OLS pe ferestre mobile @{hg.he.rolling}\\%, DCC @{hg.he.dcc}\\%'),
     [T('the dynamic ratios remove more risk, and DCC does so without choosing a window', 'rapoartele dinamice elimină mai mult risc, iar DCC face acest lucru fără alegerea unei ferestre')]),
    (T('All effectiveness values are modest: the BET--DAX DCC correlation averages @{pa.bet_dax.mean} since 2015, so $\\rho^2$ is small', 'Toate valorile eficienței sînt modeste: corelația DCC dintre BET și DAX are media @{pa.bet_dax.mean} din 2015, deci $\\rho^2$ este mic'),
     [T('a daily DCC ratio also implies daily trading costs, which are not counted here', 'un raport DCC zilnic implică și costuri zilnice de tranzacționare, care nu sînt incluse aici')])])

D.recap(('Hedging', 'acoperirea riscului'), [
    T('$h_t^* = h_{sf,t}/h_{ff,t} = \\rho_t\\sigma_{s,t}/\\sigma_{f,t}$; $HE = 1 - \\Var(\\text{hedged})/\\Var(\\text{unhedged})$', '$h_t^* = h_{sf,t}/h_{ff,t} = \\rho_t\\sigma_{s,t}/\\sigma_{f,t}$; $HE = 1 - \\Var(\\text{acoperit})/\\Var(\\text{neacoperit})$'),
    T('BET with DAX: DCC @{hg.he.dcc}\\% against static @{hg.he.static}\\% out of sample', 'BET cu DAX: DCC @{hg.he.dcc}\\% față de static @{hg.he.static}\\% în afara eșantionului'),
    T('Judge a hedge out of sample and after trading costs', 'Evaluați o acoperire în afara eșantionului și după costurile de tranzacționare')])

# =============================================================================
# 9. LEGĂTURI ȘI LIMITE
# =============================================================================
D.section('Links with other chapters and limits', 'Legături cu alte capitole și limite')

D.frame(T('Links with Chapters 5, 6 and 7', 'Legături cu Capitolele 5, 6 și 7'), items(
    (T('\\textbf{Chapter 5}: every MGARCH model contains univariate GARCH models; in DCC they are literally step 1', '\\textbf{Capitolul 5}: orice model MGARCH conține modele GARCH univariate; în DCC ele sînt chiar pasul 1'), []),
    (T('\\textbf{Chapter 6}: a VAR models spillovers in the \\textbf{mean} (e.g.\\ whether yesterday\'s DAX return helps to forecast today\'s BET return); BEKK with a full $\\mathbf{A}$ models spillovers in the \\textbf{variance}',
       '\\textbf{Capitolul 6}: un VAR modelează transmiterea în \\textbf{medie} (de exemplu, dacă randamentul DAX de ieri ajută la prognoza randamentului BET de azi); BEKK cu $\\mathbf{A}$ complet modelează transmiterea în \\textbf{varianță}'),
     [T('the two combine: a VAR for $\\boldsymbol{\\mu}_t$ and an MGARCH for $\\mathbf{H}_t$; spillover indices from variance decompositions \\refDY', 'cele două se combină: un VAR pentru $\\boldsymbol{\\mu}_t$ și un MGARCH pentru $\\mathbf{H}_t$; indicii de transmitere din descompunerea varianței \\refDY')]),
    (T('\\textbf{Chapter 7}: cointegration is a \\textbf{long-run} link between price levels; correlation is a \\textbf{short-run} link between returns', '\\textbf{Capitolul 7}: cointegrarea este o legătură pe \\textbf{termen lung} între nivelurile prețurilor; corelația este o legătură pe \\textbf{termen scurt} între randamente'),
     [T('two prices can be highly correlated day by day and drift apart for ever, or cointegrated with low daily correlation', 'două prețuri pot fi puternic corelate de la o zi la alta și să se îndepărteze definitiv, sau pot fi cointegrate cu o corelație zilnică mică'),
      T('a VECM can carry a DCC error term: equilibrium in the mean, dynamic correlation in the shocks', 'un VECM poate avea erori de tip DCC: echilibru în medie, corelație dinamică în șocuri')])))

D.frame(T('Limits and extensions', 'Limite și extensii'), two(
    ph('frankfurt', T('Frankfurt Stock Exchange', 'Bursa din Frankfurt'), h='0.3\\textheight'),
    items(T('One $(a, b)$ for all pairs; no spillovers between volatilities in DCC', 'Un singur $(a, b)$ pentru toate perechile; DCC nu are transmitere între volatilități'),
          T('Normal QML underestimates the tails: use Student $t$ innovations or FHS for risk', 'QML cu distribuția Normală subestimează cozile: pentru risc folosiți inovații Student $t$ sau FHS'),
          T('Asymmetry: correlations react more to joint falls (asymmetric DCC, \\refCES)', 'Asimetria: corelațiile reacționează mai mult la scăderile comune (DCC asimetric, \\refCES)'),
          T('Hundreds of assets: factor models, composite likelihood and shrinkage of $\\bar{\\mathbf{Q}}$', 'Sute de active: modele factoriale, verosimilitate compusă și contracția (shrinkage) lui $\\bar{\\mathbf{Q}}$'),
          T('Correlation is linear dependence; for joint extremes see copulas and tail dependence \\refLS', 'Corelația măsoară dependența liniară; pentru extreme comune vedeți copulele și dependența în cozi \\refLS')),
    wl='0.34', wr='0.64'), size='footnotesize')

# =============================================================================
# 10. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Code}: a first draft of the DCC likelihood, of a BEKK filter, of a Kupiec test', '\\textbf{Cod}: o primă versiune a verosimilității DCC, a unui filtru BEKK, a testului Kupiec'),
    T('\\textbf{Explanation}: a second explanation of $\\mathbf{D}_t\\mathbf{R}_t\\mathbf{D}_t$, of correlation targeting, of the hedge-ratio formula', '\\textbf{Explicații}: o a doua explicație a descompunerii $\\mathbf{D}_t\\mathbf{R}_t\\mathbf{D}_t$, a țintirii corelației, a formulei raportului de acoperire'),
    T('\\textbf{Exploration}: DCC for many pairs (CEE markets, sectors, currencies) and their behaviour in crises', '\\textbf{Explorare}: DCC pentru multe perechi (piețe din Europa Centrală și de Est, sectoare, valute) și comportamentul lor în crize'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that fits a GARCH(1,1) to daily BET and DAX log returns on common trading days, estimates a DCC(1,1) model by two-step maximum likelihood with correlation targeting, plots the dynamic hedge ratio and compares its out-of-sample hedging effectiveness with static OLS.}',
        '\\aiprompt{Write Python code that fits a GARCH(1,1) to daily BET and DAX log returns on common trading days, estimates a DCC(1,1) model by two-step maximum likelihood with correlation targeting, plots the dynamic hedge ratio and compares its out-of-sample hedging effectiveness with static OLS.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('The library: \\texttt{arch} has no DCC model; an AI answer that calls \\texttt{arch\\_model(..., vol=\'DCC\')} has invented it', 'Biblioteca: \\texttt{arch} nu are model DCC; un răspuns AI care apelează \\texttt{arch\\_model(..., vol=\'DCC\')} l-a inventat'),
    T('The timing: $\\mathbf{Q}_t$ must use $\\mathbf{z}_{t-1}$, not $\\mathbf{z}_t$; otherwise the VaR and the hedge use tomorrow\'s information', 'Momentul: $\\mathbf{Q}_t$ trebuie să folosească $\\mathbf{z}_{t-1}$, nu $\\mathbf{z}_t$; altfel VaR-ul și acoperirea folosesc informația de mîine'),
    T('The rescaling: $\\rho_{ij,t} = q_{ij,t}/\\sqrt{q_{ii,t}q_{jj,t}}$, and every $\\mathbf{R}_t$ must have ones on the diagonal and $|\\rho| < 1$', 'Rescalarea: $\\rho_{ij,t} = q_{ij,t}/\\sqrt{q_{ii,t}q_{jj,t}}$, iar fiecare $\\mathbf{R}_t$ trebuie să aibă 1 pe diagonală și $|\\rho| < 1$'),
    T('The calendars: align prices on common trading days before computing returns; watch asynchronous closes', 'Calendarele: aliniați prețurile pe zilele comune de tranzacționare înainte de a calcula randamentele; atenție la închiderile asincrone'),
    T('The evaluation: estimate on one period and test on another; simulate a DCC with known $(a, b)$ and check that the code recovers it', 'Evaluarea: estimați pe o perioadă și testați pe alta; simulați un DCC cu $(a, b)$ cunoscuți și verificați că programul îi regăsește'),
    T('Every cited reference: it must exist; check the DOI', 'Fiecare referință citată: trebuie să existe; verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Portfolio risk, hedging and contagion depend on time-varying covariances $\\mathbf{H}_t$', 'Riscul portofoliului, acoperirea și contagiunea depind de covarianțele variabile în timp $\\mathbf{H}_t$'),
    T('VEC is general but has too many parameters; BEKK is positive definite by construction and measures spillovers', 'VEC este general, dar are prea mulți parametri; BEKK este pozitiv definit prin construcție și măsoară transmiterea volatilității'),
    T('CCC and DCC split $\\mathbf{H}_t = \\mathbf{D}_t\\mathbf{R}_t\\mathbf{D}_t$ and are estimated in two steps; DCC needs only $(a, b)$ in step 2', 'CCC și DCC descompun $\\mathbf{H}_t = \\mathbf{D}_t\\mathbf{R}_t\\mathbf{D}_t$ și se estimează în doi pași; DCC are nevoie în pasul 2 doar de $(a, b)$'),
    T('Real correlations move: integration, crises, new assets; constant correlation is rejected', 'Corelațiile reale se mișcă: integrare, crize, active noi; corelația constantă este respinsă'),
    T('Applications: portfolio VaR 1\\% with backtesting; dynamic hedge ratios judged out of sample', 'Aplicații: VaR 1\\% al portofoliului, cu backtesting; rapoarte dinamice de acoperire evaluate în afara eșantionului')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.35}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    [T('Correlation', 'Corelația') + ' & $\\rho_{ij,t} = h_{ij,t}/\\sqrt{h_{ii,t}h_{jj,t}}$',
     'BEKK(1,1) & $\\mathbf{H}_t = \\mathbf{C}\\mathbf{C}^\\top + \\mathbf{A}^\\top\\boldsymbol{\\varepsilon}_{t-1}\\boldsymbol{\\varepsilon}_{t-1}^\\top\\mathbf{A} + \\mathbf{B}^\\top\\mathbf{H}_{t-1}\\mathbf{B}$',
     'CCC, DCC & $\\mathbf{H}_t = \\mathbf{D}_t\\mathbf{R}_t\\mathbf{D}_t$, \\quad $\\mathbf{D}_t = \\mathrm{diag}(\\sigma_{i,t})$',
     'DCC & $\\mathbf{Q}_t = (1 - a - b)\\bar{\\mathbf{Q}} + a\\,\\mathbf{z}_{t-1}\\mathbf{z}_{t-1}^\\top + b\\,\\mathbf{Q}_{t-1}$, \\quad $\\rho_{ij,t} = q_{ij,t}/\\sqrt{q_{ii,t}q_{jj,t}}$',
     T('Half-life', 'Timpul de înjumătățire') + ' & $\\ln 0.5/\\ln(a + b)$',
     T('Portfolio VaR 1\\%', 'VaR 1\\% al portofoliului') + ' & $-(\\mathbf{w}^\\top\\boldsymbol{\\mu} + q_{0.01}\\sqrt{\\mathbf{w}^\\top\\mathbf{H}_t\\mathbf{w}})$',
     T('Hedge ratio', 'Raportul de acoperire') + ' & $h_t^* = h_{sf,t}/h_{ff,t}$, \\quad $HE = 1 - \\Var(r_s - h r_f)/\\Var(r_s)$'],
    size='scriptsize') + '}')

D.frame(T('Self-assessment (1/2)', 'Autoevaluare (1/2)'), items(
    (T('\\textbf{Question}: how many parameters does a diagonal BEKK(1,1) have for $N = 3$?', '\\textbf{Întrebare}: cîți parametri are un BEKK(1,1) diagonal pentru $N = 3$?'),
     [T('\\textbf{Answer}: $N(N+1)/2 + 2N = 6 + 6 = 12$', '\\textbf{Răspuns}: $N(N+1)/2 + 2N = 6 + 6 = 12$')]),
    (T('\\textbf{Question}: $h_{11} = 1$, $h_{22} = 4$, $h_{12} = 1.2$. What is the correlation, and is the matrix positive definite?', '\\textbf{Întrebare}: $h_{11} = 1$, $h_{22} = 4$, $h_{12} = 1{,}2$. Cît este corelația și este matricea pozitiv definită?'),
     [T('\\textbf{Answer}: $\\rho = 1.2/\\sqrt{4} = 0.6$; yes, $1 \\cdot 4 - 1.44 > 0$', '\\textbf{Răspuns}: $\\rho = 1{,}2/\\sqrt{4} = 0{,}6$; da, $1 \\cdot 4 - 1{,}44 > 0$')]),
    (T('\\textbf{Question}: DCC with $a = 0.02$, $b = 0.97$, $\\bar q_{12} = 0.4$, $q_{11,t-1} = q_{22,t-1} = 1$, $q_{12,t-1} = 0.4$ and $z_{t-1} = (1, -1)$. What is $q_{12,t}$?', '\\textbf{Întrebare}: DCC cu $a = 0{,}02$, $b = 0{,}97$, $\\bar q_{12} = 0{,}4$, $q_{11,t-1} = q_{22,t-1} = 1$, $q_{12,t-1} = 0{,}4$ și $z_{t-1} = (1, -1)$. Cît este $q_{12,t}$?'),
     [T('\\textbf{Answer}: $0.01 \\cdot 0.4 + 0.02 \\cdot (-1) + 0.97 \\cdot 0.4 = 0.372$; opposite moves lower the correlation', '\\textbf{Răspuns}: $0{,}01 \\cdot 0{,}4 + 0{,}02 \\cdot (-1) + 0{,}97 \\cdot 0{,}4 = 0{,}372$; mișcările în sens opus reduc corelația')]),
    (T('\\textbf{Question}: what is the half-life of correlation shocks when $a + b = 0.95$?', '\\textbf{Întrebare}: cît este timpul de înjumătățire al șocurilor de corelație cînd $a + b = 0{,}95$?'),
     [T('\\textbf{Answer}: $\\ln 0.5/\\ln 0.95 \\approx 13.5$ days', '\\textbf{Răspuns}: $\\ln 0{,}5/\\ln 0{,}95 \\approx 13{,}5$ zile')])), size='footnotesize')

D.frame(T('Self-assessment (2/2)', 'Autoevaluare (2/2)'), items(
    (T('\\textbf{Question}: $\\sigma_s = 2\\%$, $\\sigma_f = 1\\%$, $\\rho = 0.5$. What are the hedge ratio and the hedging effectiveness?', '\\textbf{Întrebare}: $\\sigma_s = 2\\%$, $\\sigma_f = 1\\%$, $\\rho = 0{,}5$. Cît sînt raportul de acoperire și eficiența acoperirii?'),
     [T('\\textbf{Answer}: $h^* = 0.5 \\cdot 2/1 = 1$; $HE = 0.25$', '\\textbf{Răspuns}: $h^* = 0{,}5 \\cdot 2/1 = 1$; $HE = 0{,}25$')]),
    (T('\\textbf{Question}: a VaR 1\\% has 26 violations in 2\\,500 days, 10 of them in one month. Is it a good VaR?', '\\textbf{Întrebare}: un VaR 1\\% are 26 de încălcări în 2\\,500 de zile, dintre care 10 într-o singură lună. Este un VaR bun?'),
     [T('\\textbf{Answer}: the frequency is fine (expected 25), but the violations cluster: it fails the independence test and does not adapt to crises', '\\textbf{Răspuns}: frecvența este corectă (25 așteptate), dar încălcările se grupează: VaR-ul nu trece testul de independență și nu se adaptează crizelor')]),
    (T('\\textbf{Question}: why can the daily correlation of the S\\&P 500 and the BET be lower than the weekly one?', '\\textbf{Întrebare}: de ce poate fi corelația zilnică dintre S\\&P 500 și BET mai mică decît cea săptămînală?'),
     [T('\\textbf{Answer}: asynchronous trading: news from New York reaches Bucharest the next day, so daily returns of the same day do not match', '\\textbf{Răspuns}: tranzacționarea asincronă: știrile de la New York ajung la București a doua zi, deci randamentele zilnice din aceeași zi nu se potrivesc')]),
    (T('\\textbf{Question}: does a higher correlation in a crisis prove contagion?', '\\textbf{Întrebare}: dovedește o corelație mai mare într-o criză existența contagiunii?'),
     [T('\\textbf{Answer}: not by itself: higher volatility raises the sample correlation mechanically \\refFR; compare correlations of standardised residuals', '\\textbf{Răspuns}: nu, prin ea însăși: volatilitatea mai mare crește mecanic corelația de selecție \\refFR; comparați corelațiile reziduurilor standardizate')]),
    T('Next: Chapter 15, review and exam preparation; check yourself with the quiz of this chapter', 'Urmează: Capitolul 15, recapitulare și pregătire pentru examen; verificați-vă cu quiz-ul acestui capitol')), size='footnotesize')

D.references(bib(), per=10)

if __name__ == '__main__':
    finalize(D.write(V))
