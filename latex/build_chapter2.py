r"""
build_chapter2.py -- Capitolul 2 (Modele ARMA), EN + RO dintr-o singură sursă
==============================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_02/ch2_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter2_arma_models.tex
  RO/Cursuri/capitol2_modele_arma.tex
Rulare:
  python3 Quantlets/Ch_02/generate_all_charts.py
  python3 latex/build_chapter2.py && python3 latex/tsa_build.py compile 2
"""

import math
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, Values, cols, items, table, photo   # noqa: E402
from ch2_common import QLURL, REFS, T, bib, date, finalize, load, pv, quarter   # noqa: E402

N = load()
V = Values()
D = Deck(2, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\quantlet{{{folder.replace("_", chr(92) + "_")}}}{{\\qlurl{{{folder}}}}}'


def chart(title, fig, folder, bullets, h='0.62\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def interp(title, bullets, size='small'):
    D.frame(T(f'Interpreting {title[0]}', f'Interpretarea {title[1]}'), items(*bullets), size)


PH = {
    'box': ('ch2_george_box.jpg', C + 'GeorgeEPBox.jpg',
            T('Photo', 'Foto') + ': DavidMCEddy; CC BY-SA 3.0; Wikimedia Commons'),
    'walker': ('ch2_gilbert_walker.jpg', C + 'Gilbert_Walker.jpg',
               T('Photo: unknown author (1925); public domain; Wikimedia Commons', 'Foto: autor necunoscut (1925); domeniu public; Wikimedia Commons')),
    'akaike': ('ch2_hirotugu_akaike.jpg', C + 'Akaike.jpg',
               T('Photo', 'Foto') + ': The Institute of Statistical Mathematics (2017); CC BY-SA 4.0; Wikimedia Commons'),
    'ins': ('ch2_ins_bucharest_2009.jpg', C + 'Institutul_Na\\%C8\\%9Bional_de_Statistic\\%C4\\%83.jpg',
            T('Photo', 'Foto') + ': Dan Mihai Pitea (2009); CC BY-SA 3.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.50\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# CIFRE
# =============================================================================
A1 = N['ar1']
for k, key in [('0.9', 'a'), ('0.3', 'b'), ('-0.7', 'c')]:
    V.put(f'ar1.{key}.r1', A1[k]['r1'], 2)
    V.put(f'ar1.{key}.var', A1[k]['var'], 2)
    V.put(f'ar1.{key}.vth', A1[k]['var_th'], 2)
V.put('ar1.c.r2', A1['-0.7']['r2'], 2)

A2 = N['ar2']
V.put('ar2.a.r1', A2['A']['rho1'], 3)
V.put('ar2.a.r2', A2['A']['rho2'], 3)
V.put('ar2.a.z1', A2['A']['roots'][0][0], 2)
V.put('ar2.a.z2', A2['A']['roots'][1][0], 2)
V.put('ar2.b.re', A2['B']['roots'][0][0], 3)
V.put('ar2.b.im', abs(A2['B']['roots'][0][1]), 3)
V.put('ar2.b.mod', 1 / A2['B']['inv_roots_mod'][0], 3)
V.put('ar2.b.imod', A2['B']['inv_roots_mod'][0], 3)
V.put('ar2.b.per', A2['B']['period'], 1)
V.put('ar2.c.z1', A2['C']['roots'][1][0], 3)

M1 = N['ma1']
V.put('ma1.p.r1', M1['0.8']['r1'], 2)
V.put('ma1.p.rth', M1['0.8']['rho1_th'], 3)
V.put('ma1.p.p2', M1['0.8']['p2'], 2)
V.put('ma1.m.r1', M1['-0.8']['r1'], 2)

PT = N['patterns']
V.put('pt.ar2.p2', PT['AR(2)']['pt'][1], 2)
V.put('pt.ar2.r1', PT['AR(2)']['rt'][0], 3)
V.put('pt.ma2.r1', PT['MA(2)']['rt'][0], 3)
V.put('pt.ma2.r2', PT['MA(2)']['rt'][1], 3)
V.put('pt.arma.r1', PT['ARMA(1,1)']['rt'][0], 3)
V.put('pt.arma.r2', PT['ARMA(1,1)']['rt'][1], 3)

PS = N['psi']
V.put('psi.ar2.3', PS['AR(2)'][3], 2)
V.put('psi.ar2.4', PS['AR(2)'][4], 2)
V.put('psi.arma.1', PS['ARMA(1'][1], 2)
V.put('psi.arma.2', PS['ARMA(1'][2], 2)

E = N['est']
for k in ['yw', 'cls', 'ml']:
    V.put(f'est.{k}.m', E[f'ar_{k}_mean'], 3)
    V.put(f'est.{k}.s', E[f'ar_{k}_sd'], 3)
V.put('est.ma.cls', E['ma_cls_mean'], 3)
V.put('est.ma.ml', E['ma_ml_mean'], 3)
V.put('est.ma.clss', E['ma_cls_sd'], 3)
V.put('est.ma.mls', E['ma_ml_sd'], 3)
V.put('est.ar.asy', E['ar_se_asy'], 3)
V.put('est.ma.asy', E['ma_se_asy'], 3)
V.put('est.bias', -(1 + 3 * 0.9) / 100, 3)
V.int('est.nrep', E['nrep'])

IC = N['ic']
for t in ['100', '1000']:
    for k in ['aic_true', 'bic_true', 'aic_over', 'bic_over', 'aic_under', 'bic_under']:
        V.put(f'ic.{t}.{k}', 100 * IC[t][k], 1)

LD = N['lbdf']
V.put('lbdf.ar.m', LD['AR(1)'][0], 1)
V.put('lbdf.ar.k', LD['AR(1)'][1], 1)
V.put('lbdf.arma.m', LD['ARMA(1,1)'][0], 1)
V.put('lbdf.arma.k', LD['ARMA(1,1)'][1], 1)
V.int('lbdf.nrep', LD['nrep'])

FT = N['fct']
V.put('ft.9.f5', FT['\\phi = 0.9']['f5'], 2)
V.put('ft.9.se5', FT['\\phi = 0.9']['se5'], 2)
V.put('ft.9.inf', FT['\\phi = 0.9']['se_inf'], 2)
V.put('ft.5.f5', FT['\\phi = 0.5']['f5'], 3)
V.put('ft.5.inf', FT['\\phi = 0.5']['se_inf'], 2)
V.put('ft.ma.inf', FT['ma']['se_inf'], 2)

G = N['gdp']
V.int('gdp.T', G['T'])
V.raw('gdp.q0', quarter(G['first']))
V.raw('gdp.q1', quarter(G['last']))
V.put('gdp.mean', G['mean'], 2)
V.put('gdp.sd', G['sd'], 2)
for i in range(6):
    V.put(f'gdp.r{i + 1}', G['r'][i], 2)
    V.put(f'gdp.p{i + 1}', G['p'][i], 2)
V.put('gdp.band', G['band'], 2)
V.put('gdp.min', G['min'], 1)
V.raw('gdp.mind', quarter(G['min_d']))
V.put('gdp.max', G['max'], 1)
V.raw('gdp.maxd', quarter(G['max_d']))
V.put('gdp.last', G['last_v'], 1)
V.put('gdp.lb8', G['lb8']['lb'], 1)

GT = N['gdptab']
ROWS_GDP = ['00', '10', '20', '11', '03', '13', '23']
best_a, best_b = GT['aic_best'], GT['bic_best']
assert best_b == [0, 3] and best_a == [1, 3]          # the text below describes this outcome
for r in ROWS_GDP:
    d = GT['rows'][r]
    V.put(f'gt.{r}.ll', d['loglik'], 2)
    V.put(f'gt.{r}.aic', d['aic'], 2)
    V.put(f'gt.{r}.bic', d['bic'], 2)
    V.raw(f'gt.{r}.lbp', pv(d['lb8_p']))
    V.raw(f'gt.{r}.k', str(int(d['k'])))
    V.raw(f'gt.{r}.df', str(int(d['lb8_df'])))
    V.put(f'gt.{r}.sig', d['sigma'], 2)
P3, S3 = GT['bic_params'], GT['bic_se']
V.put('ma3.mu', P3['const'], 2)
V.put('ma3.mus', S3['const'], 2)
for j in (1, 2, 3):
    V.put(f'ma3.t{j}', P3[f'ma.L{j}'], 3)
    V.put(f'ma3.s{j}', S3[f'ma.L{j}'], 3)
V.put('ma3.sig', math.sqrt(P3['sigma2']), 2)
V.put('ma3.rmin', min(GT['bic_ma_root_mod']), 2)
V.put('arma13.t1', GT['aic_params']['ma.L1'], 3)
V.put('arma13.phi', GT['aic_params']['ar.L1'], 3)
V.put('arma13.phis', GT['aic_se']['ar.L1'], 3)
V.put('arma13.rmin', min(GT['aic_ma_root_mod']), 2)
V.put('gt.dbic', GT['rows']['13']['bic'] - GT['rows']['03']['bic'], 2)
V.put('gt.daic', GT['rows']['03']['aic'] - GT['rows']['13']['aic'], 2)

GD = N['gdpdiag']
V.put('gd.q8', GD['lb8']['lb'], 2)
V.raw('gd.p8', pv(GD['lb8']['lb_p']))
V.raw('gd.p8w', pv(GD['lb8_wrong_p']))
V.put('gd.q16', GD['lb16']['lb'], 2)
V.raw('gd.p16', pv(GD['lb16']['lb_p']))
V.put('gd.jb', GD['jb'], 1)
V.raw('gd.jbp', pv(GD['jb_p']))
V.put('gd.skew', GD['skew'], 2)
V.put('gd.kurt', GD['kurt'], 2)
V.raw('gd.outd', quarter(GD['out_d']))
V.put('gd.outv', GD['out_v'], 1)
V.put('gd.outz', GD['out_z'], 2)
V.put('gd.minp', GD['min_p'], 2)
V.raw('gd.psq', pv(GD['lbsq8']['lb_p']))

GF = N['gdpfc']
F3, F1 = GF['ARMA(0,3)'], GF['ARMA(1,0)']
for h in range(4):
    V.put(f'gf.f{h + 1}', F3['f'][h], 2)
V.put('gf.lo1', F3['lo'][0], 1)
V.put('gf.hi1', F3['hi'][0], 1)
V.put('gf.lo4', F3['lo'][3], 1)
V.put('gf.hi4', F3['hi'][3], 1)
V.put('gf.ar8', F1['f'][7], 2)
V.put('gf.armu', F1['mu'], 2)
V.put('gf.hw1', (F3['hi'][0] - F3['lo'][0]) / 2, 1)
V.put('gf.hw4', (F3['hi'][3] - F3['lo'][3]) / 2, 1)
V.raw('gf.q0', quarter(GF['first']))
V.raw('gf.q1', quarter(GF['last']))

IN = N['infl']
assert IN['p'] == 2
V.put('in.mu', IN['mu'], 2)
V.put('in.mus', IN['se'][0], 2)
V.put('in.p1', IN['params'][1], 3)
V.put('in.p2', IN['params'][2], 3)
V.put('in.s1', IN['se'][1], 3)
V.put('in.s2', IN['se'][2], 3)
V.put('in.sum', IN['sum_phi'], 3)
V.put('in.mod', max(IN['inv_mod']), 3)
V.int('in.T', IN['T'])
V.raw('in.last', date(IN['last'], day=False))
V.raw('in.flast', date(IN['f_last'], day=False))
V.put('in.lastv', IN['last_v'], 2)
V.put('in.f1', IN['f'][0], 2)
V.put('in.f12', IN['f'][11], 2)
V.put('in.f24', IN['f'][23], 2)
V.put('in.lo24', IN['lo95'][23], 1)
V.put('in.hi24', IN['hi95'][23], 1)
V.put('in.max', IN['max'], 1)
V.raw('in.maxd', date(IN['max_d'], day=False))
V.put('in.q12', IN['lb12']['lb'], 1)
V.raw('in.p12', pv(IN['lb12']['lb_p']))
V.put('in.bic1', IN['bic'][0], 1)
V.put('in.bic2', IN['bic'][1], 1)
V.put('in.bic3', IN['bic'][2], 1)

RT = N['ret']
for k in ['bet', 'eurron']:
    d = RT[k]
    V.int(f'{k}.T', d['T'])
    V.put(f'{k}.phi', d['phi'], 3)
    V.put(f'{k}.se', d['se'], 3)
    V.put(f'{k}.t', d['t'], 1)
    V.put(f'{k}.r2', 100 * d['r2'], 1)
    V.put(f'{k}.q', d['lb10']['lb'], 1)
    V.raw(f'{k}.qp', pv(d['lb10']['lb_p']))
    V.put(f'{k}.q2', d['lbsq10']['lb'], 0)
    V.put(f'{k}.jb', d['jb'], 0)
    V.put(f'{k}.kurt', d['kurt'], 1)
    V.raw(f'{k}.bic', f"ARMA({d['bic'][0]},{d['bic'][1]})")
    V.raw(f'{k}.aic', f"ARMA({d['aic'][0]},{d['aic'][1]})")
    V.put(f'{k}.band', 1.96 / math.sqrt(d['T']), 3)

SU = N['sun']
V.put('sun.p1', SU['phi1'], 3)
V.put('sun.p2', SU['phi2'], 3)
V.put('sun.per', SU['period'], 2)
V.put('sun.mod', SU['mod'], 3)
V.raw('sun.n', str(SU['n_yule']))
V.raw('sun.aic', str(SU['aic_p']))
V.raw('sun.bic', str(SU['bic_p']))
V.put('sun.f1', SU['full_phi'][0], 3)
V.put('sun.f2', SU['full_phi'][1], 3)
V.put('sun.r2', 100 * SU['r2'], 0)
V.raw('sun.N', str(SU['n']))

# ---- worked examples (computed here)
c, ph1, s2 = 2.0, 0.8, 1.0
V.put('we.ar.mu', c / (1 - ph1), 0)
V.put('we.ar.g0', s2 / (1 - ph1 ** 2), 2)
V.put('we.ar.r3', ph1 ** 3, 3)
V.put('we.ar.hl', math.log(0.5) / math.log(ph1), 2)
xT = 12.0
mu = c / (1 - ph1)
V.put('we.fc.f1', mu + ph1 * (xT - mu), 2)
V.put('we.fc.f2', mu + ph1 ** 2 * (xT - mu), 2)
V.put('we.fc.f10', mu + ph1 ** 10 * (xT - mu), 2)
V.put('we.fc.v2', s2 * (1 + ph1 ** 2), 2)
V.put('we.fc.lo1', mu + ph1 * (xT - mu) - 1.96, 2)
V.put('we.fc.hi1', mu + ph1 * (xT - mu) + 1.96, 2)
V.put('we.fc.lo2', mu + ph1 ** 2 * (xT - mu) - 1.96 * math.sqrt(1 + ph1 ** 2), 2)
V.put('we.fc.hi2', mu + ph1 ** 2 * (xT - mu) + 1.96 * math.sqrt(1 + ph1 ** 2), 2)
V.put('we.fc.sdinf', math.sqrt(s2 / (1 - ph1 ** 2)), 2)
# Yule-Walker AR(2) on the GDP sample autocorrelations
r1, r2 = G['r'][0], G['r'][1]
f1 = r1 * (1 - r2) / (1 - r1 ** 2)
f2 = (r2 - r1 ** 2) / (1 - r1 ** 2)
V.put('yw.f1', f1, 3)
V.put('yw.f2', f2, 3)
V.put('yw.s2', G['sd'] ** 2 * (1 - f1 * r1 - f2 * r2), 2)
V.put('yw.g0', G['sd'] ** 2, 2)
V.put('yw.det', 1 - r1 ** 2, 3)
# MA(1): theta and 1/theta
V.put('ma.r05', 0.5 / 1.25, 1)
# ARMA(1,1) of the pattern chart
V.put('arma.g0', (1 + 2 * 0.7 * 0.4 + 0.16) / (1 - 0.49), 2)
V.put('chi8', stats.chi2.ppf(0.95, 8), 2)
V.put('chi5', stats.chi2.ppf(0.95, 5), 2)
V.put('aicc.k', 2 * 5 * 6 / (G['T'] - 5 - 1), 2)
V.put('ln106', math.log(G['T']), 2)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: a stationary series is correlated with its own past; which small model captures this memory and turns it into forecasts with honest uncertainty?',
       '\\textbf{Întrebarea}: o serie staționară este corelată cu propriul trecut; ce model mic surprinde această memorie și o transformă în prognoze cu o incertitudine corect măsurată?'),
     [T('the answer of Box and Jenkins: \\textbf{ARMA($p,q$)} models, a few parameters instead of infinitely many Wold weights',
        'răspunsul lui Box și Jenkins: modelele \\textbf{ARMA($p,q$)}, cîțiva parametri în locul infinității de ponderi Wold')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('AR($p$) and MA($q$): stationarity, invertibility, characteristic roots, ACF and PACF', 'AR($p$) și MA($q$): staționaritate, invertibilitate, rădăcini caracteristice, ACF și PACF'),
      T('ARMA($p,q$): the causal MA($\\infty$) form, the impulse responses, identification', 'ARMA($p,q$): forma cauzală MA($\\infty$), răspunsurile la impuls, identificarea'),
      T('estimation (Yule--Walker, conditional least squares, maximum likelihood), AIC and BIC, residual diagnostics', 'estimarea (Yule--Walker, cele mai mici pătrate condiționate, verosimilitate maximă), AIC și BIC, diagnosticarea reziduurilor'),
      T('forecasting and the Box--Jenkins method on Romanian GDP, inflation, BET, EUR/RON and the sunspots', 'prognoza și metoda Box--Jenkins pe PIB-ul și inflația României, BET, EUR/RON și petele solare')]),
    T('Seminar 2 comes before this lecture: it gives the formulas; here we derive them and use them on data',
      'Seminarul 2 are loc înaintea acestui curs: dă formulele; aici le deducem și le aplicăm pe date')))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('Check the stationarity of an AR($p$) and the invertibility of an MA($q$) from the roots of their polynomials', 'Verificați staționaritatea unui AR($p$) și invertibilitatea unui MA($q$) din rădăcinile polinoamelor lor'),
    T('Derive the mean, the autocovariances and the $\\psi$ weights of AR(1), AR(2), MA($q$) and ARMA(1,1)', 'Deduceți media, autocovarianțele și ponderile $\\psi$ pentru AR(1), AR(2), MA($q$) și ARMA(1,1)'),
    T('Propose candidate orders from the ACF and the PACF, and choose among them with AIC and BIC', 'Propuneți ordine candidate din ACF și PACF și alegeți între ele cu AIC și BIC'),
    T('Estimate ARMA models by Yule--Walker, conditional least squares and maximum likelihood, and read the output', 'Estimați modele ARMA prin Yule--Walker, cele mai mici pătrate condiționate și verosimilitate maximă și interpretați rezultatele'),
    T('Test the residuals: Ljung--Box with $m - p - q$ degrees of freedom, Jarque--Bera', 'Testați reziduurile: Ljung--Box cu $m - p - q$ grade de libertate, Jarque--Bera'),
    T('Compute ARMA forecasts and their intervals, and explain mean reversion', 'Calculați prognozele ARMA și intervalele lor și explicați revenirea la medie')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refHP, Ch.~3--4 (stationary models, ARMA)', 'Manual: \\refHP, cap.~3--4 (modele staționare, ARMA)'),
     [T('companion, free online: \\refFPP, Ch.~9 (ARIMA models)', 'manual însoțitor, gratuit online: \\refFPP, cap.~9 (modele ARIMA)')]),
    T('Theory: \\refBD, Ch.~3 (ARMA) and Ch.~5 (estimation); \\refHamilton, Ch.~3--5; the classic: \\refBJ', 'Teorie: \\refBD, cap.~3 (ARMA) și cap.~5 (estimare); \\refHamilton, cap.~3--5; lucrarea clasică: \\refBJ'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_02}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_02}'),
     [T('ARMA estimation with \\texttt{statsmodels} (\\texttt{ARIMA}, \\texttt{ArmaProcess}, \\texttt{acorr\\_ljungbox})', 'estimarea ARMA cu \\texttt{statsmodels} (\\texttt{ARIMA}, \\texttt{ArmaProcess}, \\texttt{acorr\\_ljungbox})')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter2_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter2_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. DE LA CAPITOLUL 1 LA ARMA
# =============================================================================
D.section('From Chapter 1 to ARMA models', 'De la Capitolul 1 la modelele ARMA')

D.frame(T('Tools from Chapter 1', 'Instrumente din Capitolul 1'), items(
    (T('\\textbf{Weak stationarity}: constant mean $\\mu$ and autocovariance $\\gamma(h) = \\mathrm{Cov}(X_t, X_{t+h})$ that depends only on the lag $h$',
       '\\textbf{Staționaritatea slabă}: media constantă $\\mu$ și autocovarianța $\\gamma(h) = \\mathrm{Cov}(X_t, X_{t+h})$, care depinde doar de lagul $h$'),
     [T('ACF $\\rho(h) = \\gamma(h)/\\gamma(0)$; PACF $\\phi_{hh}$ = correlation of $X_t$ and $X_{t+h}$ after removing $X_{t+1}, \\dots, X_{t+h-1}$',
        'ACF $\\rho(h) = \\gamma(h)/\\gamma(0)$; PACF $\\phi_{hh}$ = corelația dintre $X_t$ și $X_{t+h}$ după eliminarea efectului lui $X_{t+1}, \\dots, X_{t+h-1}$')]),
    (T('\\textbf{White noise} $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$: mean 0, variance $\\sigma^2$, no autocorrelation', '\\textbf{Zgomotul alb} $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$: media 0, varianța $\\sigma^2$, fără autocorelație'),
     [T('the \\textbf{lag operator}: $LX_t = X_{t-1}$, $L^kX_t = X_{t-k}$; polynomials in $L$ act on series', '\\textbf{operatorul lag}: $LX_t = X_{t-1}$, $L^kX_t = X_{t-k}$; polinoamele în $L$ acționează asupra seriilor')]),
    (T('\\textbf{Wold decomposition} (\\refWold): every stationary, purely nondeterministic process is $X_t = \\mu + \\sum_{j \\ge 0}\\psi_j\\varepsilon_{t-j}$, $\\psi_0 = 1$, $\\sum_j\\psi_j^2 < \\infty$',
       '\\textbf{Descompunerea Wold} (\\refWold): orice proces staționar pur nedeterminist este $X_t = \\mu + \\sum_{j \\ge 0}\\psi_j\\varepsilon_{t-j}$, $\\psi_0 = 1$, $\\sum_j\\psi_j^2 < \\infty$'),
     [T('details and charts: Chapter 1, sections on the lag operator, Wold and the sample ACF and PACF', 'detalii și grafice: Capitolul 1, secțiunile despre operatorul lag, descompunerea Wold și ACF și PACF de selecție')])))

D.frame(T('The problem: infinitely many weights', 'Problema: o infinitate de ponderi'), items(
    (T('Wold says \\textbf{what} a stationary process looks like, not \\textbf{how to estimate it}', 'Teorema Wold spune \\textbf{cum arată} un proces staționar, nu \\textbf{cum îl estimăm}'),
     [T('$\\psi_1, \\psi_2, \\dots$: infinitely many unknowns, and only $T$ observations', '$\\psi_1, \\psi_2, \\dots$: o infinitate de necunoscute și doar $T$ observații')]),
    (T('\\textbf{Idea}: write $\\psi(L) = \\sum_j\\psi_jL^j$ as a \\textbf{ratio of two short polynomials}', '\\textbf{Ideea}: scriem $\\psi(L) = \\sum_j\\psi_jL^j$ ca \\textbf{raport a două polinoame scurte}'),
     [T('$\\psi(L) = \\theta(L)/\\phi(L)$ with $\\phi(L) = 1 - \\phi_1L - \\dots - \\phi_pL^p$, $\\theta(L) = 1 + \\theta_1L + \\dots + \\theta_qL^q$',
        '$\\psi(L) = \\theta(L)/\\phi(L)$, cu $\\phi(L) = 1 - \\phi_1L - \\dots - \\phi_pL^p$, $\\theta(L) = 1 + \\theta_1L + \\dots + \\theta_qL^q$'),
      T('then $\\phi(L)(X_t - \\mu) = \\theta(L)\\varepsilon_t$: the \\textbf{ARMA($p,q$)} model with $p + q + 2$ parameters ($\\phi$, $\\theta$, $\\mu$, $\\sigma^2$)',
        'atunci $\\phi(L)(X_t - \\mu) = \\theta(L)\\varepsilon_t$: modelul \\textbf{ARMA($p,q$)}, cu $p + q + 2$ parametri ($\\phi$, $\\theta$, $\\mu$, $\\sigma^2$)')]),
    (T('A rational function approximates almost any smooth $\\psi(L)$ well with small $p$ and $q$', 'O funcție rațională aproximează bine aproape orice $\\psi(L)$ netedă, cu $p$ și $q$ mici'),
     [T('the principle of \\textbf{parsimony}: the smallest model that leaves white-noise residuals', 'principiul \\textbf{parcimoniei}: cel mai mic model care lasă reziduuri de tip zgomot alb')])))

D.frame(T('Three building blocks', 'Trei piese de bază'), items(
    (T('\\textbf{AR($p$)}, autoregressive: today depends on the last $p$ values of the series', '\\textbf{AR($p$)}, autoregresiv: valoarea de azi depinde de ultimele $p$ valori ale seriei'),
     [T('$X_t = c + \\phi_1X_{t-1} + \\dots + \\phi_pX_{t-p} + \\varepsilon_t$; origin: \\refYule, sunspots', '$X_t = c + \\phi_1X_{t-1} + \\dots + \\phi_pX_{t-p} + \\varepsilon_t$; originea: \\refYule, petele solare')]),
    (T('\\textbf{MA($q$)}, moving average: today depends on the last $q$ shocks', '\\textbf{MA($q$)}, medie mobilă: valoarea de azi depinde de ultimele $q$ șocuri'),
     [T('$X_t = \\mu + \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\dots + \\theta_q\\varepsilon_{t-q}$; origin: \\refSlutzky, sums of random causes', '$X_t = \\mu + \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\dots + \\theta_q\\varepsilon_{t-q}$; originea: \\refSlutzky, sume de cauze aleatoare')]),
    (T('\\textbf{ARMA($p,q$)}: both parts; popularised as a complete method by \\refBJ (first edition 1970)', '\\textbf{ARMA($p,q$)}: ambele părți; transformat într-o metodă completă de \\refBJ (prima ediție în 1970)'),
     [T('$c$ is the \\textbf{intercept}; the mean is $\\mu = c/(1 - \\phi_1 - \\dots - \\phi_p)$, not $c$', '$c$ este \\textbf{termenul liber}; media este $\\mu = c/(1 - \\phi_1 - \\dots - \\phi_p)$, nu $c$')]),
    T('Throughout: $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$, and for likelihood and intervals $\\varepsilon_t$ i.i.d. $N(0, \\sigma^2)$ (independent and identically distributed)',
      'Peste tot: $\\varepsilon_t \\sim \\mathrm{WN}(0, \\sigma^2)$, iar pentru verosimilitate și intervale $\\varepsilon_t$ i.i.d. $N(0, \\sigma^2)$ (independente și identic distribuite)')))

# =============================================================================
# 2. AR
# =============================================================================
D.section('Autoregressive models', 'Modele autoregresive')

D.frame(T('The AR(1) model', 'Modelul AR(1)'), items(
    (T('$X_t = c + \\phi X_{t-1} + \\varepsilon_t$, or $(1 - \\phi L)X_t = c + \\varepsilon_t$', '$X_t = c + \\phi X_{t-1} + \\varepsilon_t$, sau $(1 - \\phi L)X_t = c + \\varepsilon_t$'),
     [T('$\\phi$: the share of today\'s deviation from the mean that survives until tomorrow', '$\\phi$: partea din abaterea de azi față de medie care se păstrează și mîine')]),
    (T('\\textbf{Back substitution} $k$ times:', '\\textbf{Substituind înapoi} de $k$ ori:'),
     [T('$X_t = c(1 + \\phi + \\dots + \\phi^{k-1}) + \\phi^kX_{t-k} + \\sum_{j=0}^{k-1}\\phi^j\\varepsilon_{t-j}$', '$X_t = c(1 + \\phi + \\dots + \\phi^{k-1}) + \\phi^kX_{t-k} + \\sum_{j=0}^{k-1}\\phi^j\\varepsilon_{t-j}$'),
      T('if $|\\phi| < 1$: $\\phi^k \\to 0$ and $X_t = \\dfrac{c}{1 - \\phi} + \\sum_{j=0}^{\\infty}\\phi^j\\varepsilon_{t-j}$', 'dacă $|\\phi| < 1$: $\\phi^k \\to 0$ și $X_t = \\dfrac{c}{1 - \\phi} + \\sum_{j=0}^{\\infty}\\phi^j\\varepsilon_{t-j}$')]),
    (T('This is the Wold form with $\\psi_j = \\phi^j$: the \\textbf{causal} (MA($\\infty$)) representation', 'Aceasta este forma Wold cu $\\psi_j = \\phi^j$: reprezentarea \\textbf{cauzală} (MA($\\infty$))'),
     [T('causal = $X_t$ depends only on present and past shocks', 'cauzală = $X_t$ depinde doar de șocurile prezente și trecute'),
      T('$\\phi = 1$: random walk (Chapter 1), no stationary solution; $|\\phi| > 1$: explosive, or a solution that depends on future shocks, excluded', '$\\phi = 1$: mers aleator (Capitolul 1), fără soluție staționară; $|\\phi| > 1$: proces exploziv sau o soluție care depinde de șocuri viitoare, exclusă')])))

D.frame(T('Moments of a stationary AR(1)', 'Momentele unui AR(1) staționar'), items(
    (T('\\textbf{Mean}: take expectations, $\\mu = c + \\phi\\mu$, so $\\mu = c/(1 - \\phi)$', '\\textbf{Media}: aplicăm media, $\\mu = c + \\phi\\mu$, deci $\\mu = c/(1 - \\phi)$'),
     [T('stationarity is used here: $E[X_t] = E[X_{t-1}]$', 'aici folosim staționaritatea: $E[X_t] = E[X_{t-1}]$')]),
    (T('\\textbf{Variance}: $\\gamma(0) = \\phi^2\\gamma(0) + \\sigma^2$, since $\\varepsilon_t$ is uncorrelated with $X_{t-1}$', '\\textbf{Varianța}: $\\gamma(0) = \\phi^2\\gamma(0) + \\sigma^2$, deoarece $\\varepsilon_t$ este necorelat cu $X_{t-1}$'),
     [T('$\\gamma(0) = \\sigma^2/(1 - \\phi^2)$: grows without bound as $|\\phi| \\to 1$', '$\\gamma(0) = \\sigma^2/(1 - \\phi^2)$: crește nelimitat cînd $|\\phi| \\to 1$')]),
    (T('\\textbf{Autocovariances}: multiply $X_t - \\mu = \\phi(X_{t-1} - \\mu) + \\varepsilon_t$ by $X_{t-h} - \\mu$, $h \\ge 1$, and take expectations', '\\textbf{Autocovarianțele}: înmulțim $X_t - \\mu = \\phi(X_{t-1} - \\mu) + \\varepsilon_t$ cu $X_{t-h} - \\mu$, $h \\ge 1$, și aplicăm media'),
     [T('$\\gamma(h) = \\phi\\gamma(h - 1)$, so $\\gamma(h) = \\phi^h\\gamma(0)$ and $\\rho(h) = \\phi^h$', '$\\gamma(h) = \\phi\\gamma(h - 1)$, deci $\\gamma(h) = \\phi^h\\gamma(0)$ și $\\rho(h) = \\phi^h$'),
      T('geometric decay; alternating signs if $\\phi < 0$', 'descreștere geometrică; semne alternante dacă $\\phi < 0$')]),
    T('\\textbf{Half-life} of a shock: the $h$ with $\\phi^h = 1/2$, i.e. $h = \\ln 0.5/\\ln\\phi$', '\\textbf{Timpul de înjumătățire} al unui șoc: $h$ pentru care $\\phi^h = 1/2$, adică $h = \\ln 0{,}5/\\ln\\phi$')))

D.frame(T('Worked example: an AR(1)', 'Exemplu rezolvat: un AR(1)'), items(
    T('$X_t = 2 + 0.8X_{t-1} + \\varepsilon_t$, $\\sigma^2 = 1$', '$X_t = 2 + 0{,}8X_{t-1} + \\varepsilon_t$, $\\sigma^2 = 1$'),
    (T('Stationary? $|\\phi| = 0.8 < 1$: yes', 'Este staționar? $|\\phi| = 0{,}8 < 1$: da'),
     [T('mean $\\mu = 2/(1 - 0.8) = @{we.ar.mu}$; the intercept is 2, the mean is @{we.ar.mu}', 'media $\\mu = 2/(1 - 0{,}8) = @{we.ar.mu}$; termenul liber este 2, media este @{we.ar.mu}')]),
    (T('Variance and ACF', 'Varianța și ACF'),
     [T('$\\gamma(0) = 1/(1 - 0.64) = @{we.ar.g0}$', '$\\gamma(0) = 1/(1 - 0{,}64) = @{we.ar.g0}$'),
      T('$\\rho(3) = 0.8^3 = @{we.ar.r3}$', '$\\rho(3) = 0{,}8^3 = @{we.ar.r3}$')]),
    (T('Half-life $= \\ln 0.5/\\ln 0.8 = @{we.ar.hl}$ periods', 'Timpul de înjumătățire $= \\ln 0{,}5/\\ln 0{,}8 = @{we.ar.hl}$ perioade'),
     [T('after about 3 periods, half of a shock is gone', 'după aproximativ 3 perioade, jumătate din șoc a dispărut')]),
    T('$\\psi$ weights: $1, 0.8, 0.64, 0.512, \\dots$: a shock of 1 today moves $X_{t+2}$ by $0.64$', 'Ponderile $\\psi$: $1; 0{,}8; 0{,}64; 0{,}512; \\dots$: un șoc de 1 azi îl mută pe $X_{t+2}$ cu $0{,}64$')))

chart(T('AR(1) for three values of $\\phi$', 'AR(1) pentru trei valori ale lui $\\phi$'), 'tsa_ch2_ar1', 'TSA_ch2_ar_models', [
    T('The same 200 shocks, $\\phi = 0.9$, $0.3$, $-0.7$; bottom: sample ACF (bars) and $\\rho(h) = \\phi^h$ (dots)', 'Aceleași 200 de șocuri, $\\phi = 0{,}9$; $0{,}3$; $-0{,}7$; jos: ACF de selecție (bare) și $\\rho(h) = \\phi^h$ (puncte)')],
    h='0.72\\textheight')

interp(('the AR(1) paths', 'traiectoriilor AR(1)'), [
    (T('$\\phi = 0.9$: long swings away from 0; sample $\\hat\\rho(1) = @{ar1.a.r1}$, variance @{ar1.a.var} (theory @{ar1.a.vth})', '$\\phi = 0{,}9$: abateri lungi de la 0; $\\hat\\rho(1) = @{ar1.a.r1}$ de selecție, varianța @{ar1.a.var} (teoretic @{ar1.a.vth})'),
     [T('strong persistence looks like a trend over short windows', 'persistența puternică seamănă cu un trend pe ferestre scurte')]),
    (T('$\\phi = 0.3$: close to white noise; only $\\hat\\rho(1) = @{ar1.b.r1}$ is clearly outside the band', '$\\phi = 0{,}3$: aproape de zgomot alb; doar $\\hat\\rho(1) = @{ar1.b.r1}$ iese clar din bandă'),
     [T('variance @{ar1.b.var} (theory @{ar1.b.vth})', 'varianța @{ar1.b.var} (teoretic @{ar1.b.vth})')]),
    (T('$\\phi = -0.7$: zig-zag; $\\hat\\rho(1) = @{ar1.c.r1}$, $\\hat\\rho(2) = @{ar1.c.r2}$, alternating signs', '$\\phi = -0{,}7$: zigzag; $\\hat\\rho(1) = @{ar1.c.r1}$, $\\hat\\rho(2) = @{ar1.c.r2}$, semne alternante'),
     [T('in economics: overshooting and correction, or a measurement error that is corrected next period', 'în economie: depășire și corecție sau o eroare de măsurare corectată în perioada următoare')]),
    T('With $T = 200$ the sample ACF is below the theoretical one at long lags: a small-sample bias that also affects the estimators (Section 5)', 'Cu $T = 200$, ACF de selecție este sub cea teoretică la laguri mari: o deplasare de eșantion mic, care afectează și estimatorii (secțiunea 5)')])

D.frame(T('The AR($p$) model and its characteristic polynomial', 'Modelul AR($p$) și polinomul lui caracteristic'), items(
    (T('$\\phi(L)X_t = c + \\varepsilon_t$, with $\\phi(z) = 1 - \\phi_1z - \\dots - \\phi_pz^p$, a polynomial in a complex number $z$', '$\\phi(L)X_t = c + \\varepsilon_t$, cu $\\phi(z) = 1 - \\phi_1z - \\dots - \\phi_pz^p$, un polinom într-un număr complex $z$'),
     [T('factorise: $\\phi(z) = (1 - \\lambda_1z)\\cdots(1 - \\lambda_pz)$, where $z_i = 1/\\lambda_i$ are the roots', 'factorizăm: $\\phi(z) = (1 - \\lambda_1z)\\cdots(1 - \\lambda_pz)$, unde $z_i = 1/\\lambda_i$ sînt rădăcinile')]),
    (T('\\textbf{Stationarity (causality) condition}: all roots of $\\phi(z) = 0$ lie \\textbf{outside the unit circle}, $|z_i| > 1$', '\\textbf{Condiția de staționaritate (cauzalitate)}: toate rădăcinile lui $\\phi(z) = 0$ sînt \\textbf{în afara cercului unitate}, $|z_i| > 1$'),
     [T('equivalently: all \\textbf{inverse roots} $\\lambda_i$ lie inside, $|\\lambda_i| < 1$ (the form shown by software)', 'echivalent: toate \\textbf{rădăcinile inverse} $\\lambda_i$ sînt în interior, $|\\lambda_i| < 1$ (forma afișată de programe)'),
      T('reason: $1/(1 - \\lambda L) = \\sum_j\\lambda^jL^j$ converges only if $|\\lambda| < 1$; each factor is an AR(1)', 'motivul: $1/(1 - \\lambda L) = \\sum_j\\lambda^jL^j$ converge doar dacă $|\\lambda| < 1$; fiecare factor este un AR(1)')]),
    (T('A root on the circle ($|z| = 1$, e.g. $z = 1$) is a \\textbf{unit root}: non-stationarity, the topic of Chapter 3', 'O rădăcină pe cerc ($|z| = 1$, de exemplu $z = 1$) este o \\textbf{rădăcină unitară}: nestaționaritate, subiectul Capitolului 3'),
     [T('quick necessary check: $\\phi_1 + \\dots + \\phi_p < 1$ (i.e. $\\phi(1) > 0$)', 'verificare rapidă, necesară: $\\phi_1 + \\dots + \\phi_p < 1$ (adică $\\phi(1) > 0$)')])))

D.frame(T('AR(2): the stationarity triangle', 'AR(2): triunghiul de staționaritate'), items(
    (T('$X_t = \\phi_1X_{t-1} + \\phi_2X_{t-2} + \\varepsilon_t$ is stationary if and only if', '$X_t = \\phi_1X_{t-1} + \\phi_2X_{t-2} + \\varepsilon_t$ este staționar dacă și numai dacă'),
     [T('$\\phi_1 + \\phi_2 < 1$, \\quad $\\phi_2 - \\phi_1 < 1$, \\quad $|\\phi_2| < 1$', '$\\phi_1 + \\phi_2 < 1$, \\quad $\\phi_2 - \\phi_1 < 1$, \\quad $|\\phi_2| < 1$')]),
    (T('The roots are complex when $\\phi_1^2 + 4\\phi_2 < 0$', 'Rădăcinile sînt complexe cînd $\\phi_1^2 + 4\\phi_2 < 0$'),
     [T('then the ACF is a \\textbf{damped cosine}: pseudo-cycles of average length $2\\pi/\\arccos\\big(\\phi_1/(2\\sqrt{-\\phi_2})\\big)$', 'atunci ACF este un \\textbf{cosinus amortizat}: pseudo-cicluri cu lungimea medie $2\\pi/\\arccos\\big(\\phi_1/(2\\sqrt{-\\phi_2})\\big)$'),
      T('damping factor per period: $\\sqrt{-\\phi_2}$, the modulus of the inverse roots', 'factorul de amortizare pe perioadă: $\\sqrt{-\\phi_2}$, modulul rădăcinilor inverse')]),
    (T('\\textbf{Yule--Walker equations} for the ACF: multiply by $X_{t-h}$, take expectations, divide by $\\gamma(0)$', '\\textbf{Ecuațiile Yule--Walker} pentru ACF: înmulțim cu $X_{t-h}$, aplicăm media, împărțim la $\\gamma(0)$'),
     [T('$\\rho(1) = \\phi_1 + \\phi_2\\rho(1)$, so $\\rho(1) = \\phi_1/(1 - \\phi_2)$', '$\\rho(1) = \\phi_1 + \\phi_2\\rho(1)$, deci $\\rho(1) = \\phi_1/(1 - \\phi_2)$'),
      T('$\\rho(h) = \\phi_1\\rho(h - 1) + \\phi_2\\rho(h - 2)$ for $h \\ge 2$: the ACF obeys the same recursion as the process', '$\\rho(h) = \\phi_1\\rho(h - 1) + \\phi_2\\rho(h - 2)$ pentru $h \\ge 2$: ACF respectă aceeași recurență ca procesul')])))

chart(T('Three AR(2) processes and the sunspots', 'Trei procese AR(2) și petele solare'), 'tsa_ch2_ar2', 'TSA_ch2_ar_models', [
    T('A: $\\phi = (0.5, 0.3)$; B: $(1.0, -0.6)$; C: $(0.6, 0.5)$; Sunspots: Yule\'s AR(2), estimated in Section 10', 'A: $\\phi = (0{,}5; 0{,}3)$; B: $(1{,}0; -0{,}6)$; C: $(0{,}6; 0{,}5)$; Sunspots: AR(2)-ul lui Yule, estimat în secțiunea 10'),
    T('Middle: inverse roots $\\lambda_i$; stationary means inside the circle', 'Mijloc: rădăcinile inverse $\\lambda_i$; staționar înseamnă în interiorul cercului')], h='0.72\\textheight')

interp(('the AR(2) examples', 'exemplelor AR(2)'), [
    (T('\\textbf{A}: real roots $@{ar2.a.z1}$ and $@{ar2.a.z2}$, both outside the circle: stationary', '\\textbf{A}: rădăcini reale $@{ar2.a.z1}$ și $@{ar2.a.z2}$, ambele în afara cercului: staționar'),
     [T('$\\rho(1) = 0.5/0.7 = @{ar2.a.r1}$, $\\rho(2) = 0.5\\rho(1) + 0.3 = @{ar2.a.r2}$: slow, smooth decay', '$\\rho(1) = 0{,}5/0{,}7 = @{ar2.a.r1}$, $\\rho(2) = 0{,}5\\rho(1) + 0{,}3 = @{ar2.a.r2}$: descreștere lentă și netedă')]),
    (T('\\textbf{B}: complex roots $@{ar2.b.re} \\pm @{ar2.b.im}\\,i$, modulus $@{ar2.b.mod} > 1$: stationary', '\\textbf{B}: rădăcini complexe $@{ar2.b.re} \\pm @{ar2.b.im}\\,i$, modul $@{ar2.b.mod} > 1$: staționar'),
     [T('ACF: a wave with period @{ar2.b.per} lags, damped by $@{ar2.b.imod}$ per lag: a \\textbf{stochastic cycle}', 'ACF: o undă cu perioada de @{ar2.b.per} laguri, amortizată cu $@{ar2.b.imod}$ pe lag: un \\textbf{ciclu stochastic}')]),
    (T('\\textbf{C}: $\\phi_1 + \\phi_2 = 1.1 > 1$; one root $@{ar2.c.z1} < 1$: \\textbf{not} stationary (explosive)', '\\textbf{C}: $\\phi_1 + \\phi_2 = 1{,}1 > 1$; o rădăcină $@{ar2.c.z1} < 1$: proces \\textbf{nestaționar} (exploziv)'),
     [T('each coefficient is below 1, yet the process explodes: always check the roots, not the coefficients one by one', 'fiecare coeficient este sub 1, totuși procesul explodează: verificați întotdeauna rădăcinile, nu coeficienții unul cîte unul')]),
    T('Complex roots are how linear models produce business cycles and the 11-year solar cycle', 'Rădăcinile complexe sînt mecanismul prin care modelele liniare produc ciclurile economice și ciclul solar de 11 ani')])

D.frame(T('The PACF of an AR($p$) cuts off after lag $p$', 'PACF a unui AR($p$) se anulează după lagul $p$'), items(
    (T('$\\phi_{hh}$ = the last coefficient in the best linear prediction of $X_t$ from $X_{t-1}, \\dots, X_{t-h}$ (Chapter 1)', '$\\phi_{hh}$ = ultimul coeficient din cea mai bună predicție liniară a lui $X_t$ din $X_{t-1}, \\dots, X_{t-h}$ (Capitolul 1)'),
     [T('computed from the ACF by the Durbin--Levinson recursion (\\refDurbin)', 'calculat din ACF prin recursia Durbin--Levinson (\\refDurbin)')]),
    (T('For an AR($p$) and $h > p$, the best predictor is the model itself: $\\phi_1X_{t-1} + \\dots + \\phi_pX_{t-p}$', 'Pentru un AR($p$) și $h > p$, cel mai bun predictor este chiar modelul: $\\phi_1X_{t-1} + \\dots + \\phi_pX_{t-p}$'),
     [T('so $\\phi_{hh} = 0$ for $h > p$, and $\\phi_{pp} = \\phi_p$', 'deci $\\phi_{hh} = 0$ pentru $h > p$, iar $\\phi_{pp} = \\phi_p$')]),
    (T('\\textbf{Identification rule}: ACF decays gradually, PACF has $p$ spikes and then stays inside the band: AR($p$)', '\\textbf{Regula de identificare}: ACF descrește treptat, PACF are $p$ valori semnificative și apoi rămîne în bandă: AR($p$)'),
     [T('band for the sample PACF of an AR($p$) at lags $h > p$: $\\pm 1.96/\\sqrt{T}$', 'banda pentru PACF de selecție a unui AR($p$) la lagurile $h > p$: $\\pm 1{,}96/\\sqrt{T}$')])))

D.recap(('Autoregressive models', 'modele autoregresive'), [
    T('AR($p$): $\\phi(L)X_t = c + \\varepsilon_t$; stationary iff all roots of $\\phi(z)$ lie outside the unit circle', 'AR($p$): $\\phi(L)X_t = c + \\varepsilon_t$; staționar dacă și numai dacă toate rădăcinile lui $\\phi(z)$ sînt în afara cercului unitate'),
    T('Mean $\\mu = c/\\phi(1)$; ACF from the Yule--Walker recursion, a mix of geometric decays and damped waves', 'Media $\\mu = c/\\phi(1)$; ACF din recurența Yule--Walker, o combinație de descreșteri geometrice și unde amortizate'),
    T('AR(1): $\\rho(h) = \\phi^h$, $\\psi_j = \\phi^j$, half-life $\\ln 0.5/\\ln\\phi$', 'AR(1): $\\rho(h) = \\phi^h$, $\\psi_j = \\phi^j$, timpul de înjumătățire $\\ln 0{,}5/\\ln\\phi$'),
    T('PACF cuts off after lag $p$: the signature of an AR($p$)', 'PACF se anulează după lagul $p$: semnătura unui AR($p$)')])

# =============================================================================
# 3. MA
# =============================================================================
D.section('Moving-average models', 'Modele de medie mobilă')

D.frame(T('The MA($q$) model', 'Modelul MA($q$)'), items(
    (T('$X_t = \\mu + \\theta(L)\\varepsilon_t = \\mu + \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\dots + \\theta_q\\varepsilon_{t-q}$', '$X_t = \\mu + \\theta(L)\\varepsilon_t = \\mu + \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\dots + \\theta_q\\varepsilon_{t-q}$'),
     [T('a finite Wold sum: \\textbf{always stationary}, for any $\\theta$', 'o sumă Wold finită: \\textbf{întotdeauna staționar}, pentru orice $\\theta$')]),
    (T('Moments (derived in Seminar 1 and Chapter 1): with $\\theta_0 = 1$', 'Momentele (deduse în Seminarul 1 și în Capitolul 1): cu $\\theta_0 = 1$'),
     [T('$E[X_t] = \\mu$; $\\gamma(h) = \\sigma^2\\sum_{j=0}^{q-h}\\theta_j\\theta_{j+h}$ for $0 \\le h \\le q$; $\\gamma(h) = 0$ for $h > q$', '$E[X_t] = \\mu$; $\\gamma(h) = \\sigma^2\\sum_{j=0}^{q-h}\\theta_j\\theta_{j+h}$ pentru $0 \\le h \\le q$; $\\gamma(h) = 0$ pentru $h > q$')]),
    (T('\\textbf{Identification rule}: the ACF \\textbf{cuts off} after lag $q$; the PACF decays gradually', '\\textbf{Regula de identificare}: ACF \\textbf{se anulează} după lagul $q$; PACF descrește treptat'),
     [T('the mirror image of the AR($p$) rule', 'imaginea în oglindă a regulii pentru AR($p$)'),
      T('band for the sample ACF at lags $h > q$ (Bartlett): $\\pm 1.96\\sqrt{(1 + 2\\sum_{j=1}^{q}\\hat\\rho(j)^2)/T}$, wider than $\\pm 1.96/\\sqrt{T}$', 'banda pentru ACF de selecție la lagurile $h > q$ (Bartlett): $\\pm 1{,}96\\sqrt{(1 + 2\\sum_{j=1}^{q}\\hat\\rho(j)^2)/T}$, mai largă decît $\\pm 1{,}96/\\sqrt{T}$')])))

chart(T('MA(1): $\\theta = 0.8$ and $\\theta = -0.8$', 'MA(1): $\\theta = 0{,}8$ și $\\theta = -0{,}8$'), 'tsa_ch2_ma1', 'TSA_ch2_ma_models', [
    T('$T = 300$; the ACF has one spike, the PACF decays (with alternating signs when $\\theta > 0$)', '$T = 300$; ACF are o singură valoare semnificativă, PACF descrește (cu semne alternante cînd $\\theta > 0$)')],
    h='0.72\\textheight')

interp(('the MA(1) correlograms', 'corelogramelor MA(1)'), [
    (T('Theory: $\\rho(1) = \\theta/(1 + \\theta^2) = \\pm @{ma1.p.rth}$, $\\rho(h) = 0$ for $h \\ge 2$', 'Teoretic: $\\rho(1) = \\theta/(1 + \\theta^2) = \\pm @{ma1.p.rth}$, $\\rho(h) = 0$ pentru $h \\ge 2$'),
     [T('sample: $\\hat\\rho(1) = @{ma1.p.r1}$ and $@{ma1.m.r1}$', 'de selecție: $\\hat\\rho(1) = @{ma1.p.r1}$ și $@{ma1.m.r1}$')]),
    (T('The PACF does not cut off: $\\hat\\phi_{22} = @{ma1.p.p2}$ for $\\theta = 0.8$', 'PACF nu se anulează: $\\hat\\phi_{22} = @{ma1.p.p2}$ pentru $\\theta = 0{,}8$'),
     [T('an MA(1) is an AR($\\infty$) (next slides), so every past value helps a little', 'un MA(1) este un AR($\\infty$) (slide-urile următoare), deci fiecare valoare trecută ajută puțin')]),
    T('An MA(1) remembers exactly one period: a shock today affects tomorrow and is then forgotten', 'Un MA(1) are o memorie de exact o perioadă: un șoc de azi afectează valoarea de mîine, apoi efectul lui dispare'),
    T('Economic examples: overlapping observations (Section 9), a measurement error, a bid--ask bounce in prices', 'Exemple economice: observații care se suprapun (secțiunea 9), o eroare de măsurare, oscilația bid--ask a prețurilor')])

chart(T('The largest autocorrelation of an MA(1)', 'Cea mai mare autocorelație a unui MA(1)'), 'tsa_ch2_ma1_rho', 'TSA_ch2_ma_models', [
    T('$\\rho(1) = \\theta/(1 + \\theta^2)$ as a function of $\\theta$', '$\\rho(1) = \\theta/(1 + \\theta^2)$ ca funcție de $\\theta$')], h='0.72\\textheight')

interp(('the curve $\\rho(1)$', 'curbei $\\rho(1)$'), [
    (T('Maximum $|\\rho(1)| = 0.5$ at $\\theta = \\pm 1$: an MA(1) can never produce $|\\rho(1)| > 0.5$', 'Maximul $|\\rho(1)| = 0{,}5$ în $\\theta = \\pm 1$: un MA(1) nu poate produce niciodată $|\\rho(1)| > 0{,}5$'),
     [T('if $\\hat\\rho(1) = 0.7$ and then nothing: not an MA(1); try an AR or an MA($q$) with more terms', 'dacă $\\hat\\rho(1) = 0{,}7$ și apoi nimic: nu este un MA(1); încercați un AR sau un MA($q$) cu mai mulți termeni')]),
    (T('$\\theta = 0.5$ and $\\theta = 2$ give the same $\\rho(1) = @{ma.r05}$', '$\\theta = 0{,}5$ și $\\theta = 2$ dau același $\\rho(1) = @{ma.r05}$'),
     [T('in general $\\theta$ and $1/\\theta$ give the same ACF (with $\\sigma^2$ rescaled by $\\theta^2$)', 'în general, $\\theta$ și $1/\\theta$ dau aceeași ACF (cu $\\sigma^2$ înmulțit cu $\\theta^2$)'),
      T('the data cannot tell them apart: an \\textbf{identification problem}', 'datele nu le pot deosebi: o \\textbf{problemă de identificare}')]),
    T('The convention that removes the ambiguity: invertibility, $|\\theta| < 1$ (next slide)', 'Convenția care elimină ambiguitatea: invertibilitatea, $|\\theta| < 1$ (slide-ul următor)')])

D.frame(T('Invertibility', 'Invertibilitatea'), items(
    (T('MA(1): $\\varepsilon_t = X_t - \\theta\\varepsilon_{t-1}$; substitute repeatedly ($\\mu = 0$):', 'MA(1): $\\varepsilon_t = X_t - \\theta\\varepsilon_{t-1}$; substituim repetat ($\\mu = 0$):'),
     [T('$\\varepsilon_t = X_t - \\theta X_{t-1} + \\theta^2X_{t-2} - \\dots = \\sum_{j \\ge 0}(-\\theta)^jX_{t-j}$, if $|\\theta| < 1$', '$\\varepsilon_t = X_t - \\theta X_{t-1} + \\theta^2X_{t-2} - \\dots = \\sum_{j \\ge 0}(-\\theta)^jX_{t-j}$, dacă $|\\theta| < 1$'),
      T('an \\textbf{AR($\\infty$) representation}: $X_t = \\theta X_{t-1} - \\theta^2X_{t-2} + \\dots + \\varepsilon_t$', 'o \\textbf{reprezentare AR($\\infty$)}: $X_t = \\theta X_{t-1} - \\theta^2X_{t-2} + \\dots + \\varepsilon_t$')]),
    (T('\\textbf{Invertible}: the shocks can be recovered from the past of $X$; condition: all roots of $\\theta(z) = 0$ outside the unit circle', '\\textbf{Invertibil}: șocurile pot fi recuperate din trecutul lui $X$; condiția: toate rădăcinile lui $\\theta(z) = 0$ în afara cercului unitate'),
     [T('MA(1): $|\\theta| < 1$; the same algebra as AR stationarity, applied to $\\theta(z)$', 'MA(1): $|\\theta| < 1$; aceeași algebră ca la staționaritatea AR, aplicată lui $\\theta(z)$')]),
    (T('Why we want it', 'Motivația'),
     [T('uniqueness: among $\\theta$ and $1/\\theta$ only one is invertible', 'unicitate: dintre $\\theta$ și $1/\\theta$ doar unul este invertibil'),
      T('forecasting: the shocks $\\varepsilon_t$ are computed from the data, so forecasts can use them', 'prognoză: șocurile $\\varepsilon_t$ se calculează din date, deci prognozele le pot folosi'),
      T('software reports invertible estimates; a root near the circle signals over-differencing (Chapter 1)', 'programele raportează estimări invertibile; o rădăcină aproape de cerc semnalează supradiferențierea (Capitolul 1)')])))

D.frame(T('Worked example: $\\theta = 2$ or $\\theta = 0.5$?', 'Exemplu rezolvat: $\\theta = 2$ sau $\\theta = 0{,}5$?'), items(
    T('$X_t = \\varepsilon_t + 2\\varepsilon_{t-1}$, $\\sigma^2 = 1$; and $Y_t = u_t + 0.5u_{t-1}$, $\\mathrm{Var}(u_t) = 4$', '$X_t = \\varepsilon_t + 2\\varepsilon_{t-1}$, $\\sigma^2 = 1$; și $Y_t = u_t + 0{,}5u_{t-1}$, $\\mathrm{Var}(u_t) = 4$'),
    (T('$X$: $\\gamma(0) = 1 \\cdot (1 + 4) = 5$, $\\gamma(1) = 2 \\cdot 1 = 2$', '$X$: $\\gamma(0) = 1 \\cdot (1 + 4) = 5$, $\\gamma(1) = 2 \\cdot 1 = 2$'),
     [T('$Y$: $\\gamma(0) = 4 \\cdot (1 + 0.25) = 5$, $\\gamma(1) = 0.5 \\cdot 4 = 2$', '$Y$: $\\gamma(0) = 4 \\cdot (1 + 0{,}25) = 5$, $\\gamma(1) = 0{,}5 \\cdot 4 = 2$')]),
    (T('Same mean, same autocovariances: with Gaussian shocks, the \\textbf{same distribution}', 'Aceeași medie, aceleași autocovarianțe: cu șocuri gaussiene, \\textbf{aceeași distribuție}'),
     [T('no amount of data separates them', 'oricît de multe date am avea, nu le putem separa')]),
    (T('Only $Y$ is invertible: $u_t = Y_t - 0.5Y_{t-1} + 0.25Y_{t-2} - \\dots$', 'Doar $Y$ este invertibil: $u_t = Y_t - 0{,}5Y_{t-1} + 0{,}25Y_{t-2} - \\dots$'),
     [T('for $X$, the same sum with weights $(-2)^j$ explodes', 'pentru $X$, aceeași sumă cu ponderile $(-2)^j$ explodează')]),
    T('Rule: report the invertible version ($\\theta = 0.5$, $\\sigma^2 = 4$)', 'Regula: raportăm versiunea invertibilă ($\\theta = 0{,}5$, $\\sigma^2 = 4$)')))

D.recap(('Moving-average models', 'modele de medie mobilă'), [
    T('MA($q$) is always stationary; its ACF cuts off after lag $q$, its PACF decays', 'MA($q$) este întotdeauna staționar; ACF lui se anulează după lagul $q$, PACF descrește'),
    T('MA(1): $\\rho(1) = \\theta/(1 + \\theta^2)$, at most $0.5$ in absolute value', 'MA(1): $\\rho(1) = \\theta/(1 + \\theta^2)$, cel mult $0{,}5$ în valoare absolută'),
    T('Invertible iff the roots of $\\theta(z)$ lie outside the unit circle; then MA = AR($\\infty$)', 'Invertibil dacă și numai dacă rădăcinile lui $\\theta(z)$ sînt în afara cercului unitate; atunci MA = AR($\\infty$)'),
    T('$\\theta$ and $1/\\theta$ are observationally equivalent; we keep the invertible one', '$\\theta$ și $1/\\theta$ sînt echivalente observațional; îl păstrăm pe cel invertibil')])

# =============================================================================
# 4. ARMA
# =============================================================================
D.section('ARMA($p,q$) models', 'Modele ARMA($p,q$)')

D.frame(T('The ARMA($p,q$) model', 'Modelul ARMA($p,q$)'), items(
    (T('$\\phi(L)(X_t - \\mu) = \\theta(L)\\varepsilon_t$, i.e.', '$\\phi(L)(X_t - \\mu) = \\theta(L)\\varepsilon_t$, adică'),
     [T('$X_t = c + \\phi_1X_{t-1} + \\dots + \\phi_pX_{t-p} + \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\dots + \\theta_q\\varepsilon_{t-q}$, $c = \\mu\\,\\phi(1)$', '$X_t = c + \\phi_1X_{t-1} + \\dots + \\phi_pX_{t-p} + \\varepsilon_t + \\theta_1\\varepsilon_{t-1} + \\dots + \\theta_q\\varepsilon_{t-q}$, $c = \\mu\\,\\phi(1)$')]),
    (T('Three conditions (\\refBD, Ch.~3)', 'Trei condiții (\\refBD, cap.~3)'),
     [T('\\textbf{stationary and causal}: roots of $\\phi(z)$ outside the unit circle; then $X_t - \\mu = \\psi(L)\\varepsilon_t$, $\\psi(L) = \\theta(L)/\\phi(L)$', '\\textbf{staționar și cauzal}: rădăcinile lui $\\phi(z)$ în afara cercului unitate; atunci $X_t - \\mu = \\psi(L)\\varepsilon_t$, $\\psi(L) = \\theta(L)/\\phi(L)$'),
      T('\\textbf{invertible}: roots of $\\theta(z)$ outside the unit circle; then $\\pi(L)(X_t - \\mu) = \\varepsilon_t$, $\\pi(L) = \\phi(L)/\\theta(L)$', '\\textbf{invertibil}: rădăcinile lui $\\theta(z)$ în afara cercului unitate; atunci $\\pi(L)(X_t - \\mu) = \\varepsilon_t$, $\\pi(L) = \\phi(L)/\\theta(L)$'),
      T('\\textbf{no common factors}: $\\phi(z)$ and $\\theta(z)$ share no root', '\\textbf{fără factori comuni}: $\\phi(z)$ și $\\theta(z)$ nu au nicio rădăcină comună')]),
    T('ARMA(1,1) can match a slowly decaying ACF that starts below $\\phi$; a pure AR or MA would need many terms for it', 'ARMA(1,1) poate reproduce o ACF care descrește lent și pornește sub $\\phi$; un AR sau un MA pur ar avea nevoie de mulți termeni')))

D.frame(T('The $\\psi$ weights and the impulse response', 'Ponderile $\\psi$ și răspunsul la impuls'), items(
    (T('From $\\phi(L)\\psi(L) = \\theta(L)$, match the coefficients of $L^j$:', 'Din $\\phi(L)\\psi(L) = \\theta(L)$, egalăm coeficienții lui $L^j$:'),
     [T('$\\psi_0 = 1$, \\quad $\\psi_j = \\theta_j + \\sum_{i=1}^{\\min(j,p)}\\phi_i\\psi_{j-i}$ \\quad ($\\theta_j = 0$ for $j > q$)', '$\\psi_0 = 1$, \\quad $\\psi_j = \\theta_j + \\sum_{i=1}^{\\min(j,p)}\\phi_i\\psi_{j-i}$ \\quad ($\\theta_j = 0$ pentru $j > q$)')]),
    (T('ARMA(1,1): $\\psi_1 = \\phi + \\theta$, $\\psi_j = \\phi^{j-1}(\\phi + \\theta)$ for $j \\ge 1$', 'ARMA(1,1): $\\psi_1 = \\phi + \\theta$, $\\psi_j = \\phi^{j-1}(\\phi + \\theta)$ pentru $j \\ge 1$'),
     [T('example $\\phi = 0.7$, $\\theta = 0.4$: $\\psi_1 = @{psi.arma.1}$, $\\psi_2 = @{psi.arma.2}$', 'exemplu $\\phi = 0{,}7$, $\\theta = 0{,}4$: $\\psi_1 = @{psi.arma.1}$, $\\psi_2 = @{psi.arma.2}$')]),
    (T('\\textbf{Impulse response}: $\\psi_j = \\partial X_{t+j}/\\partial\\varepsilon_t$, the effect of a unit shock after $j$ periods', '\\textbf{Răspunsul la impuls}: $\\psi_j = \\partial X_{t+j}/\\partial\\varepsilon_t$, efectul unui șoc unitar după $j$ perioade'),
     [T('the same object drives the forecast errors (Section 7) and the impulse responses of VAR models (Chapter 6)', 'același obiect determină erorile de prognoză (secțiunea 7) și răspunsurile la impuls din modelele VAR (Capitolul 6)')])))

chart(T('Impulse responses of four models', 'Răspunsurile la impuls ale a patru modele'), 'tsa_ch2_psi', 'TSA_ch2_arma_acf', [
    T('$\\psi_j$, $j = 0, \\dots, 15$, from the recursion above', '$\\psi_j$, $j = 0, \\dots, 15$, din recurența de mai sus')], h='0.72\\textheight')

interp(('the impulse responses', 'răspunsurilor la impuls'), [
    T('AR(1): geometric decay $0.8^j$; the shock never vanishes completely, but fades fast enough for $\\sum\\psi_j^2 < \\infty$', 'AR(1): descreștere geometrică $0{,}8^j$; șocul nu dispare niciodată complet, dar se stinge suficient de repede pentru ca $\\sum\\psi_j^2 < \\infty$'),
    T('AR(2) with complex roots: the response changes sign, $\\psi_3 = @{psi.ar2.3}$, $\\psi_4 = @{psi.ar2.4}$: a shock creates a cycle', 'AR(2) cu rădăcini complexe: răspunsul își schimbă semnul, $\\psi_3 = @{psi.ar2.3}$, $\\psi_4 = @{psi.ar2.4}$: un șoc creează un ciclu'),
    T('MA(2): exactly zero after two periods: finite memory', 'MA(2): exact zero după două perioade: memorie finită'),
    T('ARMA(1,1): the MA term lifts the first response to $\\psi_1 = @{psi.arma.1} > 1$; afterwards the AR part takes over', 'ARMA(1,1): termenul MA ridică primul răspuns la $\\psi_1 = @{psi.arma.1} > 1$; după aceea preia partea AR')])

D.frame(T('Moments of the ARMA(1,1)', 'Momentele procesului ARMA(1,1)'), items(
    T('$X_t = \\phi X_{t-1} + \\varepsilon_t + \\theta\\varepsilon_{t-1}$, $|\\phi| < 1$, $|\\theta| < 1$, $\\phi + \\theta \\ne 0$', '$X_t = \\phi X_{t-1} + \\varepsilon_t + \\theta\\varepsilon_{t-1}$, $|\\phi| < 1$, $|\\theta| < 1$, $\\phi + \\theta \\ne 0$'),
    (T('Variance, from $\\gamma(0) = \\sigma^2\\sum_j\\psi_j^2$:', 'Varianța, din $\\gamma(0) = \\sigma^2\\sum_j\\psi_j^2$:'),
     [T('$\\gamma(0) = \\sigma^2\\,\\dfrac{1 + 2\\phi\\theta + \\theta^2}{1 - \\phi^2}$; for $\\phi = 0.7$, $\\theta = 0.4$, $\\sigma^2 = 1$: $@{arma.g0}$', '$\\gamma(0) = \\sigma^2\\,\\dfrac{1 + 2\\phi\\theta + \\theta^2}{1 - \\phi^2}$; pentru $\\phi = 0{,}7$, $\\theta = 0{,}4$, $\\sigma^2 = 1$: $@{arma.g0}$')]),
    (T('Autocorrelations:', 'Autocorelațiile:'),
     [T('$\\rho(1) = \\dfrac{(1 + \\phi\\theta)(\\phi + \\theta)}{1 + 2\\phi\\theta + \\theta^2}$, \\quad $\\rho(h) = \\phi\\,\\rho(h - 1)$ for $h \\ge 2$', '$\\rho(1) = \\dfrac{(1 + \\phi\\theta)(\\phi + \\theta)}{1 + 2\\phi\\theta + \\theta^2}$, \\quad $\\rho(h) = \\phi\\,\\rho(h - 1)$ pentru $h \\ge 2$'),
      T('example: $\\rho(1) = @{pt.arma.r1}$, $\\rho(2) = @{pt.arma.r2}$', 'exemplu: $\\rho(1) = @{pt.arma.r1}$, $\\rho(2) = @{pt.arma.r2}$')]),
    T('The ACF decays like an AR(1) from lag 1 onwards, but $\\rho(1) \\ne \\phi$: the MA part only moves the starting point', 'ACF descrește ca la un AR(1) începînd cu lagul 1, dar $\\rho(1) \\ne \\phi$: partea MA mută doar punctul de pornire')))

D.frame(T('Common factors: a model that is too large', 'Factori comuni: un model prea mare'), items(
    (T('$X_t = 0.5X_{t-1} + \\varepsilon_t - 0.5\\varepsilon_{t-1}$', '$X_t = 0{,}5X_{t-1} + \\varepsilon_t - 0{,}5\\varepsilon_{t-1}$'),
     [T('$(1 - 0.5L)X_t = (1 - 0.5L)\\varepsilon_t$: cancel the factor, $X_t = \\varepsilon_t$, white noise', '$(1 - 0{,}5L)X_t = (1 - 0{,}5L)\\varepsilon_t$: simplificăm factorul, $X_t = \\varepsilon_t$, zgomot alb')]),
    (T('In general $\\phi = -\\theta$ in an ARMA(1,1) gives white noise, for any value', 'În general, $\\phi = -\\theta$ într-un ARMA(1,1) dă zgomot alb, pentru orice valoare'),
     [T('the likelihood is flat along the line $\\phi = -\\theta$: the parameters are not identified', 'verosimilitatea este plată de-a lungul dreptei $\\phi = -\\theta$: parametrii nu sînt identificați')]),
    (T('Symptoms in software output', 'Simptome în rezultatele programelor'),
     [T('large and opposite estimates, e.g. $\\hat\\phi = 0.9$, $\\hat\\theta = -0.88$, with huge standard errors', 'estimări mari și de semne opuse, de exemplu $\\hat\\phi = 0{,}9$, $\\hat\\theta = -0{,}88$, cu erori standard foarte mari'),
      T('a log-likelihood no better than the smaller model', 'o log-verosimilitate care nu este mai bună decît a modelului mai mic')]),
    T('Remedy: reduce $p$ and $q$ together; prefer the smallest model with white-noise residuals', 'Remediul: reducem $p$ și $q$ împreună; preferăm cel mai mic model cu reziduuri de tip zgomot alb')))

D.frame(T('Identification: the ACF--PACF table', 'Identificarea: tabelul ACF--PACF'), table(
    'lll', T('\\textbf{Model}', '\\textbf{Modelul}') + ' & \\textbf{ACF} & \\textbf{PACF}',
    [T('White noise', 'Zgomot alb') + ' & ' + T('no significant lag', 'niciun lag semnificativ') + ' & ' + T('no significant lag', 'niciun lag semnificativ'),
     'AR($p$) & ' + T('decays (geometric or damped wave)', 'descrește (geometric sau undă amortizată)') + ' & ' + T('\\textbf{cuts off} after lag $p$', '\\textbf{se anulează} după lagul $p$'),
     'MA($q$) & ' + T('\\textbf{cuts off} after lag $q$', '\\textbf{se anulează} după lagul $q$') + ' & ' + T('decays', 'descrește'),
     'ARMA($p,q$) & ' + T('decays after lag $q - p$', 'descrește după lagul $q - p$') + ' & ' + T('decays after lag $p - q$', 'descrește după lagul $p - q$'),
     T('Unit root (Ch.~3)', 'Rădăcină unitară (cap.~3)') + ' & ' + T('very slow, almost linear decay', 'descreștere foarte lentă, aproape liniară') + ' & ' + T('$\\hat\\phi_{11} \\approx 1$', '$\\hat\\phi_{11} \\approx 1$')],
    size='footnotesize') + items(
    T('The table proposes \\textbf{candidates}; the information criteria and the diagnostics decide (Sections 5--6)', 'Tabelul propune \\textbf{candidați}; criteriile informaționale și diagnosticarea decid (secțiunile 5--6)'),
    T('Mixed ARMA models are hard to spot by eye: both functions just decay', 'Modelele ARMA mixte sînt greu de recunoscut din ochi: ambele funcții doar descresc')))

chart(T('Theoretical and sample ACF and PACF', 'ACF și PACF teoretice și de selecție'), 'tsa_ch2_patterns', 'TSA_ch2_arma_acf', [
    T('AR(2) $\\phi = (1.0, -0.6)$, MA(2) $\\theta = (0.6, 0.3)$, ARMA(1,1) $\\phi = 0.7$, $\\theta = 0.4$; one simulated path of $T = 500$ each', 'AR(2) $\\phi = (1{,}0; -0{,}6)$, MA(2) $\\theta = (0{,}6; 0{,}3)$, ARMA(1,1) $\\phi = 0{,}7$, $\\theta = 0{,}4$; cîte o traiectorie simulată cu $T = 500$')],
    h='0.72\\textheight')

interp(('the three patterns', 'celor trei tipare'), [
    (T('AR(2): the PACF has two spikes, $\\phi_{22} = \\phi_2 = @{pt.ar2.p2}$, then zero; the ACF is a damped wave', 'AR(2): PACF are două valori semnificative, $\\phi_{22} = \\phi_2 = @{pt.ar2.p2}$, apoi zero; ACF este o undă amortizată'),
     [T('identification is easy: read $p$ from the PACF', 'identificarea este ușoară: citim $p$ din PACF')]),
    (T('MA(2): $\\rho(1) = @{pt.ma2.r1}$, $\\rho(2) = @{pt.ma2.r2}$, then zero; the PACF decays slowly', 'MA(2): $\\rho(1) = @{pt.ma2.r1}$, $\\rho(2) = @{pt.ma2.r2}$, apoi zero; PACF descrește lent'),
     [T('the sample ACF at lag 2 is only just outside the band: with real data, MA(1) and MA(2) can look alike', 'ACF de selecție la lagul 2 iese abia ușor din bandă: pe date reale, MA(1) și MA(2) pot semăna')]),
    (T('ARMA(1,1): both decay; the ACF looks like an AR(1), the PACF has a second, smaller spike', 'ARMA(1,1): ambele descresc; ACF seamănă cu un AR(1), PACF are o a doua valoare, mai mică'),
     [T('an analyst who reads only the PACF would choose AR(2) or AR(3): close, but more parameters', 'un analist care citește doar PACF ar alege AR(2) sau AR(3): apropiat, dar cu mai mulți parametri')]),
    T('Sample correlograms are noisy even with $T = 500$; with 100 quarterly observations, much more so', 'Corelogramele de selecție sînt zgomotoase chiar cu $T = 500$; cu 100 de observații trimestriale, mult mai mult')])

D.recap(('ARMA models', 'modele ARMA'), [
    T('ARMA($p,q$): $\\phi(L)(X_t - \\mu) = \\theta(L)\\varepsilon_t$; causal if $\\phi(z)$ has roots outside the circle, invertible if $\\theta(z)$ has', 'ARMA($p,q$): $\\phi(L)(X_t - \\mu) = \\theta(L)\\varepsilon_t$; cauzal dacă $\\phi(z)$ are rădăcinile în afara cercului, invertibil dacă $\\theta(z)$ le are'),
    T('$\\psi$ weights by the recursion $\\psi_j = \\theta_j + \\sum\\phi_i\\psi_{j-i}$: the impulse response', 'Ponderile $\\psi$ prin recurența $\\psi_j = \\theta_j + \\sum\\phi_i\\psi_{j-i}$: răspunsul la impuls'),
    T('No common factors; a flat likelihood and opposite estimates signal one', 'Fără factori comuni; o verosimilitate plată și estimări opuse semnalează unul'),
    T('ACF cuts off: MA; PACF cuts off: AR; both decay: ARMA', 'ACF se anulează: MA; PACF se anulează: AR; ambele descresc: ARMA')])

# =============================================================================
# 5. ESTIMARE
# =============================================================================
D.section('Estimation', 'Estimarea')

D.frame(T('Three estimators', 'Trei estimatori'), items(
    (T('\\textbf{Yule--Walker} (method of moments, \\refYule; \\refWalker): replace $\\rho(h)$ by $\\hat\\rho(h)$ in the Yule--Walker equations', '\\textbf{Yule--Walker} (metoda momentelor, \\refYule; \\refWalker): înlocuim $\\rho(h)$ cu $\\hat\\rho(h)$ în ecuațiile Yule--Walker'),
     [T('pure AR only; fast; always gives a stationary model; less precise when roots are near the circle', 'doar pentru AR pur; rapid; dă întotdeauna un model staționar; mai puțin precis cînd rădăcinile sînt aproape de cerc')]),
    (T('\\textbf{Conditional least squares}: minimise $\\sum_t\\varepsilon_t^2$, treating the first values as fixed', '\\textbf{Cele mai mici pătrate condiționate}: minimizăm $\\sum_t\\varepsilon_t^2$, tratînd primele valori ca fixe'),
     [T('AR($p$): ordinary least squares (OLS) of $X_t$ on $1, X_{t-1}, \\dots, X_{t-p}$', 'AR($p$): metoda celor mai mici pătrate (OLS) pentru $X_t$ pe $1, X_{t-1}, \\dots, X_{t-p}$'),
      T('MA and ARMA: the residuals $\\varepsilon_t = X_t - \\mu - \\theta\\varepsilon_{t-1} - \\dots$ are computed recursively from $\\varepsilon_0 = 0$; numerical minimisation', 'MA și ARMA: reziduurile $\\varepsilon_t = X_t - \\mu - \\theta\\varepsilon_{t-1} - \\dots$ se calculează recursiv, pornind de la $\\varepsilon_0 = 0$; minimizare numerică')]),
    (T('\\textbf{Exact Gaussian maximum likelihood} (MLE): maximise the joint density of $(X_1, \\dots, X_T)$', '\\textbf{Verosimilitatea maximă exactă gaussiană} (MLE): maximizăm densitatea comună a lui $(X_1, \\dots, X_T)$'),
     [T('uses the first observations too; the default of \\texttt{statsmodels} \\texttt{ARIMA} (Kalman filter, Chapter 10)', 'folosește și primele observații; metoda implicită a lui \\texttt{ARIMA} din \\texttt{statsmodels} (filtrul Kalman, Capitolul 10)')]),
    T('In large samples all three agree for pure AR models; MLE is the reference for MA and ARMA', 'În eșantioane mari, cele trei coincid pentru modelele AR pure; MLE este referința pentru MA și ARMA')))

D.frame(T('Yule--Walker for an AR($p$)', 'Yule--Walker pentru un AR($p$)'), cols(items(
    (T('For $h = 1, \\dots, p$: $\\rho(h) = \\phi_1\\rho(h - 1) + \\dots + \\phi_p\\rho(h - p)$', 'Pentru $h = 1, \\dots, p$: $\\rho(h) = \\phi_1\\rho(h - 1) + \\dots + \\phi_p\\rho(h - p)$'),
     [T('in matrix form $R\\,\\phi = \\rho$, $R_{ij} = \\rho(|i - j|)$; $\\phi = (\\phi_1, \\dots, \\phi_p)^\\top$, $\\rho = (\\rho(1), \\dots, \\rho(p))^\\top$', 'matricial, $R\\,\\phi = \\rho$, $R_{ij} = \\rho(|i - j|)$; $\\phi = (\\phi_1, \\dots, \\phi_p)^\\top$, $\\rho = (\\rho(1), \\dots, \\rho(p))^\\top$'),
      T('$R$: the $p \\times p$ matrix of autocorrelations; the hat marks the sample version', '$R$: matricea $p \\times p$ a autocorelațiilor; notația $\\hat{\\ }$ indică varianta de selecție'),
      T('estimator $\\hat\\phi = \\hat R^{-1}\\hat\\rho$; noise variance $\\hat\\sigma^2 = \\hat\\gamma(0)(1 - \\hat\\phi^\\top\\hat\\rho)$', 'estimatorul $\\hat\\phi = \\hat R^{-1}\\hat\\rho$; varianța zgomotului $\\hat\\sigma^2 = \\hat\\gamma(0)(1 - \\hat\\phi^\\top\\hat\\rho)$')]),
    (T('AR(2) by Cramer\'s rule:', 'AR(2) prin regula lui Cramer:'),
     [T('$\\hat\\phi_1 = \\dfrac{\\hat\\rho_1(1 - \\hat\\rho_2)}{1 - \\hat\\rho_1^2}$, \\quad $\\hat\\phi_2 = \\dfrac{\\hat\\rho_2 - \\hat\\rho_1^2}{1 - \\hat\\rho_1^2}$', '$\\hat\\phi_1 = \\dfrac{\\hat\\rho_1(1 - \\hat\\rho_2)}{1 - \\hat\\rho_1^2}$, \\quad $\\hat\\phi_2 = \\dfrac{\\hat\\rho_2 - \\hat\\rho_1^2}{1 - \\hat\\rho_1^2}$'),
      T('$\\hat\\phi_2$ is the sample PACF at lag 2 (Chapter 1)', '$\\hat\\phi_2$ este PACF de selecție la lagul 2 (Capitolul 1)')])),
    ph('walker', T('Gilbert Walker (1868--1958), who extended Yule\'s method in 1931', 'Gilbert Walker (1868--1958), care a extins metoda lui Yule în 1931'), h='0.46\\textheight'),
    wl='0.64', wr='0.32'), 'footnotesize')

D.frame(T('Worked example: Yule--Walker on Romanian GDP growth', 'Exemplu rezolvat: Yule--Walker pentru creșterea PIB-ului României'), items(
    T('Annual growth of real GDP, @{gdp.q0}--@{gdp.q1} (Section 9): $\\hat\\rho(1) = @{gdp.r1}$, $\\hat\\rho(2) = @{gdp.r2}$, $\\hat\\gamma(0) = @{yw.g0}$', 'Creșterea anuală a PIB-ului real, @{gdp.q0}--@{gdp.q1} (secțiunea 9): $\\hat\\rho(1) = @{gdp.r1}$, $\\hat\\rho(2) = @{gdp.r2}$, $\\hat\\gamma(0) = @{yw.g0}$'),
    (T('Determinant $1 - \\hat\\rho_1^2 = @{yw.det}$', 'Determinantul $1 - \\hat\\rho_1^2 = @{yw.det}$'),
     [T('$\\hat\\phi_1 = @{gdp.r1}\\,(1 - @{gdp.r2})/@{yw.det} = @{yw.f1}$', '$\\hat\\phi_1 = @{gdp.r1}\\,(1 - @{gdp.r2})/@{yw.det} = @{yw.f1}$'),
      T('$\\hat\\phi_2 = (@{gdp.r2} - @{gdp.r1}^2)/@{yw.det} = @{yw.f2}$', '$\\hat\\phi_2 = (@{gdp.r2} - @{gdp.r1}^2)/@{yw.det} = @{yw.f2}$')]),
    T('$\\hat\\sigma^2 = @{yw.g0}\\,(1 - \\hat\\phi_1\\hat\\rho_1 - \\hat\\phi_2\\hat\\rho_2) = @{yw.s2}$', '$\\hat\\sigma^2 = @{yw.g0}\\,(1 - \\hat\\phi_1\\hat\\rho_1 - \\hat\\phi_2\\hat\\rho_2) = @{yw.s2}$'),
    (T('$\\hat\\phi_2 \\approx 0$: the second lag adds nothing, AR(1) would do', '$\\hat\\phi_2 \\approx 0$: al doilea lag nu aduce nimic, AR(1) ar fi suficient'),
     [T('but is an AR model the right family at all? Section 9 shows that an MA(3) fits better', 'dar este familia AR cea potrivită? Secțiunea 9 arată că un MA(3) se potrivește mai bine')])), 'footnotesize')

D.frame(T('Maximum likelihood', 'Verosimilitatea maximă'), items(
    (T('Gaussian likelihood by the \\textbf{prediction-error decomposition}', 'Verosimilitatea gaussiană prin \\textbf{descompunerea erorilor de predicție}'),
     [T('$f(x_1, \\dots, x_T) = \\prod_t f(x_t \\mid x_{t-1}, \\dots, x_1)$; each factor is $N(\\hat x_{t|t-1}, v_t)$', '$f(x_1, \\dots, x_T) = \\prod_t f(x_t \\mid x_{t-1}, \\dots, x_1)$; fiecare factor este $N(\\hat x_{t|t-1}, v_t)$'),
      T('$\\ln L = -\\tfrac12\\sum_{t=1}^{T}\\big[\\ln(2\\pi v_t) + (x_t - \\hat x_{t|t-1})^2/v_t\\big]$', '$\\ln L = -\\tfrac12\\sum_{t=1}^{T}\\big[\\ln(2\\pi v_t) + (x_t - \\hat x_{t|t-1})^2/v_t\\big]$'),
      T('$\\hat x_{t|t-1}$: one-step prediction; $v_t$: its error variance, larger for the first observations', '$\\hat x_{t|t-1}$: predicția cu un pas; $v_t$: varianța erorii ei, mai mare pentru primele observații')]),
    (T('AR(1) by hand: $X_1 \\sim N(\\mu, \\sigma^2/(1 - \\phi^2))$, then $X_t \\mid X_{t-1} \\sim N(c + \\phi X_{t-1}, \\sigma^2)$', 'AR(1) de mînă: $X_1 \\sim N(\\mu, \\sigma^2/(1 - \\phi^2))$, apoi $X_t \\mid X_{t-1} \\sim N(c + \\phi X_{t-1}, \\sigma^2)$'),
     [T('dropping the first factor gives conditional least squares', 'dacă renunțăm la primul factor, obținem cele mai mici pătrate condiționate')]),
    (T('Standard errors (se) from the curvature of $\\ln L$; asymptotically', 'Erorile standard (se, standard error) din curbura lui $\\ln L$; asimptotic'),
     [T('AR(1): $\\mathrm{se}(\\hat\\phi) \\approx \\sqrt{(1 - \\phi^2)/T}$; MA(1): $\\mathrm{se}(\\hat\\theta) \\approx \\sqrt{(1 - \\theta^2)/T}$', 'AR(1): $\\mathrm{se}(\\hat\\phi) \\approx \\sqrt{(1 - \\phi^2)/T}$; MA(1): $\\mathrm{se}(\\hat\\theta) \\approx \\sqrt{(1 - \\theta^2)/T}$'),
      T('$t$-ratios $\\hat\\phi/\\mathrm{se}$ compared with $\\pm 1.96$, as in regression', 'rapoartele $t$, $\\hat\\phi/\\mathrm{se}$, se compară cu $\\pm 1{,}96$, ca în regresie')])))

chart(T('Sampling distributions of the estimators', 'Distribuțiile de selecție ale estimatorilor'), 'tsa_ch2_estimators', 'TSA_ch2_estimation', [
    T('@{est.nrep} simulated samples of $T = 100$ each; left: AR(1) with $\\phi = 0.9$; right: MA(1) with $\\theta = 0.5$', '@{est.nrep} de eșantioane simulate, fiecare cu $T = 100$; stînga: AR(1) cu $\\phi = 0{,}9$; dreapta: MA(1) cu $\\theta = 0{,}5$')],
    h='0.72\\textheight')

interp(('the estimator comparison', 'comparației estimatorilor'), [
    (T('AR(1), $\\phi = 0.9$: means @{est.yw.m} (Yule--Walker), @{est.cls.m} (conditional least squares), @{est.ml.m} (MLE)', 'AR(1), $\\phi = 0{,}9$: mediile @{est.yw.m} (Yule--Walker), @{est.cls.m} (cele mai mici pătrate condiționate), @{est.ml.m} (MLE)'),
     [T('all three are biased downwards; the textbook approximation of the least squares bias is $-(1 + 3\\phi)/T = @{est.bias}$', 'toți trei sînt deplasați în jos; aproximarea de manual a deplasării celor mai mici pătrate este $-(1 + 3\\phi)/T = @{est.bias}$'),
      T('Yule--Walker is the most biased near the unit circle', 'Yule--Walker este cel mai deplasat aproape de cercul unitate')]),
    (T('Standard deviations @{est.yw.s}, @{est.cls.s}, @{est.ml.s}, above the asymptotic @{est.ar.asy}', 'Abaterile standard @{est.yw.s}; @{est.cls.s}; @{est.ml.s}, peste valoarea asimptotică @{est.ar.asy}'),
     [T('the distribution is skewed to the left: asymptotic intervals are optimistic near $\\phi = 1$', 'distribuția este asimetrică la stînga: intervalele asimptotice sînt optimiste aproape de $\\phi = 1$')]),
    T('MA(1), $\\theta = 0.5$: conditional least squares @{est.ma.cls} (sd @{est.ma.clss}), MLE @{est.ma.ml} (sd @{est.ma.mls}); asymptotic sd @{est.ma.asy}', 'MA(1), $\\theta = 0{,}5$: cele mai mici pătrate condiționate @{est.ma.cls} (abaterea standard @{est.ma.clss}), MLE @{est.ma.ml} (@{est.ma.mls}); valoarea asimptotică @{est.ma.asy}'),
    T('Practical rule: use MLE; Yule--Walker only for quick starting values or for teaching', 'Regula practică: folosiți MLE; Yule--Walker doar pentru valori de pornire rapide sau în scop didactic')])

D.recap(('Estimation', 'estimarea'), [
    T('Yule--Walker: moments, pure AR, $\\hat\\phi = \\hat R^{-1}\\hat\\rho$', 'Yule--Walker: momente, AR pur, $\\hat\\phi = \\hat R^{-1}\\hat\\rho$'),
    T('Conditional least squares: OLS for AR, a recursion for MA', 'Cele mai mici pătrate condiționate: OLS pentru AR, o recurență pentru MA'),
    T('Exact Gaussian MLE: the prediction-error decomposition; the standard in software', 'MLE exactă gaussiană: descompunerea erorilor de predicție; standardul programelor'),
    T('Persistent AR coefficients are biased downwards in small samples', 'Coeficienții AR persistenți sînt deplasați în jos în eșantioane mici')])

# =============================================================================
# 6. SELECTIA MODELULUI SI DIAGNOSTICAREA
# =============================================================================
D.section('Model selection and diagnostics', 'Selecția modelului și diagnosticarea')

D.frame(T('Information criteria', 'Criterii informaționale'), cols(items(
    (T('Fit improves with every parameter; the criteria charge a \\textbf{penalty} per parameter', 'Ajustarea se îmbunătățește cu fiecare parametru; criteriile impun o \\textbf{penalizare} pe parametru'),
     [T('$k$ = number of estimated parameters ($\\phi$, $\\theta$, $\\mu$, $\\sigma^2$), $\\ln L$ = maximised log-likelihood', '$k$ = numărul parametrilor estimați ($\\phi$, $\\theta$, $\\mu$, $\\sigma^2$), $\\ln L$ = log-verosimilitatea maximizată')]),
    (T('\\textbf{AIC} $= -2\\ln L + 2k$ (\\refAkaike)', '\\textbf{AIC} $= -2\\ln L + 2k$ (\\refAkaike)'),
     [T('\\textbf{AICc} $= \\mathrm{AIC} + 2k(k + 1)/(T - k - 1)$, for small $T$ (\\refHT)', '\\textbf{AICc} $= \\mathrm{AIC} + 2k(k + 1)/(T - k - 1)$, pentru $T$ mic (\\refHT)')]),
    (T('\\textbf{BIC} $= -2\\ln L + k\\ln T$ (\\refSchwarz)', '\\textbf{BIC} $= -2\\ln L + k\\ln T$ (\\refSchwarz)'),
     [T('$\\ln T > 2$ once $T \\ge 8$: BIC prefers smaller models', '$\\ln T > 2$ de îndată ce $T \\ge 8$: BIC preferă modele mai mici')]),
    T('Choose the model with the \\textbf{smallest} value; compare only models fitted to the same observations', 'Alegem modelul cu valoarea \\textbf{cea mai mică}; comparăm doar modele estimate pe aceleași observații')),
    ph('akaike', T('Hirotugu Akaike (1927--2009), Institute of Statistical Mathematics, Tokyo', 'Hirotugu Akaike (1927--2009), Institutul de Matematică Statistică, Tokyo'), h='0.44\\textheight'),
    wl='0.64', wr='0.32'), 'footnotesize')

D.frame(T('AIC or BIC?', 'AIC sau BIC?'), items(
    (T('\\textbf{BIC is consistent}: if the true model is among the candidates, BIC finds it with probability $\\to 1$', '\\textbf{BIC este consistent}: dacă modelul adevărat se află printre candidați, BIC îl găsește cu probabilitate $\\to 1$'),
     [T('but it may choose a model that is too small when $T$ is small', 'dar poate alege un model prea mic cînd $T$ este mic')]),
    (T('\\textbf{AIC targets forecasting}: it estimates the expected out-of-sample fit (Kullback--Leibler distance)', '\\textbf{AIC este orientat spre prognoză}: estimează ajustarea așteptată în afara eșantionului (distanța Kullback--Leibler)'),
     [T('it overfits with positive probability, even with large $T$', 'supraparametrizează cu probabilitate pozitivă, chiar pentru $T$ mare')]),
    (T('Practice (\\refFPP, Ch.~9; \\refHK)', 'Practica (\\refFPP, cap.~9; \\refHK)'),
     [T('automatic searches (\\texttt{auto.arima}, \\texttt{pmdarima}) minimise AICc over a grid of $(p, q)$', 'căutările automate (\\texttt{auto.arima}, \\texttt{pmdarima}) minimizează AICc pe o grilă de valori $(p, q)$'),
      T('when AIC and BIC disagree, keep both candidates and let the diagnostics and the forecasts decide', 'cînd AIC și BIC nu coincid, păstrăm ambii candidați și lăsăm diagnosticarea și prognozele să decidă')])))

chart(T('How often AIC and BIC find the true order', 'Cît de des găsesc AIC și BIC ordinul adevărat'), 'tsa_ch2_ic', 'TSA_ch2_selection', [
    T('True model AR(2), $\\phi = (0.5, 0.25)$; AR($p$), $p = 0, \\dots, 6$, fitted by OLS on the same sample; 500 samples per panel', 'Modelul adevărat AR(2), $\\phi = (0{,}5; 0{,}25)$; AR($p$), $p = 0, \\dots, 6$, estimate prin OLS pe același eșantion; 500 de eșantioane pe panou')],
    h='0.72\\textheight')

interp(('the selection experiment', 'experimentului de selecție'), [
    (T('$T = 100$: AIC finds $p = 2$ in @{ic.100.aic_true}\\% of samples, BIC in @{ic.100.bic_true}\\%', '$T = 100$: AIC găsește $p = 2$ în @{ic.100.aic_true}\\% din eșantioane, BIC în @{ic.100.bic_true}\\%'),
     [T('BIC chooses too small a model in @{ic.100.bic_under}\\% (the small $\\phi_2 = 0.25$ is hard to detect); AIC too large in @{ic.100.aic_over}\\%', 'BIC alege un model prea mic în @{ic.100.bic_under}\\% din cazuri (valoarea mică $\\phi_2 = 0{,}25$ este greu de detectat); AIC alege unul prea mare în @{ic.100.aic_over}\\%')]),
    (T('$T = 1000$: BIC @{ic.1000.bic_true}\\%, AIC @{ic.1000.aic_true}\\%', '$T = 1000$: BIC @{ic.1000.bic_true}\\%, AIC @{ic.1000.aic_true}\\%'),
     [T('BIC is consistent; AIC still overfits in @{ic.1000.aic_over}\\% of samples', 'BIC este consistent; AIC încă supraparametrizează în @{ic.1000.aic_over}\\% din eșantioane')]),
    T('Neither criterion is a proof: with quarterly macro data ($T \\approx 100$) the order is uncertain', 'Niciun criteriu nu este o dovadă: cu date macroeconomice trimestriale ($T \\approx 100$), ordinul este incert'),
    T('Overfitting costs little in forecasts; underfitting leaves correlation in the residuals, which the diagnostics catch', 'Supraparametrizarea costă puțin în prognoză; un model prea mic lasă corelație în reziduuri, pe care o detectează diagnosticarea')])

D.frame(T('Residual diagnostics', 'Diagnosticarea reziduurilor'), items(
    (T('If the model is right, the residuals $\\hat\\varepsilon_t$ behave like white noise', 'Dacă modelul este corect, reziduurile $\\hat\\varepsilon_t$ se comportă ca un zgomot alb'),
     [T('plot them: no pattern, no change of variance, no isolated huge values', 'le reprezentăm grafic: fără tipare, fără schimbări de varianță, fără valori izolate foarte mari'),
      T('residual ACF inside $\\pm 1.96/\\sqrt{T}$, except about 1 lag in 20', 'ACF a reziduurilor în banda $\\pm 1{,}96/\\sqrt{T}$, cu excepția a aproximativ 1 lag din 20')]),
    (T('\\textbf{Ljung--Box} on residuals (\\refLB): $Q^*(m) = T(T + 2)\\sum_{h=1}^{m}\\hat\\rho_{\\hat\\varepsilon}(h)^2/(T - h)$', '\\textbf{Ljung--Box} pentru reziduuri (\\refLB): $Q^*(m) = T(T + 2)\\sum_{h=1}^{m}\\hat\\rho_{\\hat\\varepsilon}(h)^2/(T - h)$'),
     [T('under $H_0$ (correct model) $Q^*(m) \\approx \\chi^2(m - p - q)$ (\\refBP): estimating $p + q$ coefficients uses up $p + q$ degrees of freedom', 'în ipoteza $H_0$ (model corect), $Q^*(m) \\approx \\chi^2(m - p - q)$ (\\refBP): estimarea a $p + q$ coeficienți consumă $p + q$ grade de libertate'),
      T('choose $m$ well above $p + q$: $m = 10$ for annual or quarterly data, $m = 2s$ for seasonal data (\\refFPP)', 'alegem $m$ mult peste $p + q$: $m = 10$ pentru date anuale sau trimestriale, $m = 2s$ pentru date sezoniere (\\refFPP)')]),
    (T('\\textbf{Normality}: Jarque--Bera (\\refJB) $\\mathrm{JB} = \\frac{T}{6}\\big(S^2 + (K - 3)^2/4\\big) \\approx \\chi^2(2)$; QQ plot', '\\textbf{Normalitatea}: Jarque--Bera (\\refJB) $\\mathrm{JB} = \\frac{T}{6}\\big(S^2 + (K - 3)^2/4\\big) \\approx \\chi^2(2)$; graficul QQ'),
     [T('$S$ = skewness, $K$ = kurtosis (0 and 3 for the Normal distribution); reject normality at 5\\% if $\\mathrm{JB} > 5.99$, the 95\\% quantile of $\\chi^2(2)$', '$S$ = asimetria, $K$ = kurtosis-ul (0 și 3 pentru distribuția Normală); respingem normalitatea la 5\\% dacă $\\mathrm{JB} > 5{,}99$, cuantila de 95\\% a lui $\\chi^2(2)$'),
      T('non-Normal residuals do not bias $\\hat\\phi$, but they make Normal intervals wrong', 'reziduurile ne-normale nu deplasează $\\hat\\phi$, dar fac greșite intervalele construite cu distribuția Normală')]),
    T('\\textbf{Squared residuals}: Ljung--Box on $\\hat\\varepsilon_t^2$ detects volatility clustering (Chapter 5)', '\\textbf{Pătratele reziduurilor}: testul Ljung--Box pentru $\\hat\\varepsilon_t^2$ detectează volatility clustering (Capitolul 5)')))

chart(T('Why $m - p - q$ degrees of freedom', 'Justificarea celor $m - p - q$ grade de libertate'), 'tsa_ch2_lb_df', 'TSA_ch2_selection', [
    T('Ljung--Box at 5\\% on the residuals of the \\textbf{true} model; @{lbdf.nrep} samples of AR(1), half as many of ARMA(1,1)', 'Testul Ljung--Box la 5\\% pentru reziduurile modelului \\textbf{adevărat}; @{lbdf.nrep} de eșantioane AR(1), jumătate pentru ARMA(1,1)')],
    h='0.72\\textheight')

interp(('the degrees-of-freedom experiment', 'experimentului cu gradele de libertate'), [
    (T('With $m = 10$ degrees of freedom the test rejects too rarely: @{lbdf.ar.m}\\% (AR(1)) and @{lbdf.arma.m}\\% (ARMA(1,1)) instead of 5\\%', 'Cu $m = 10$ grade de libertate testul respinge prea rar: @{lbdf.ar.m}\\% (AR(1)) și @{lbdf.arma.m}\\% (ARMA(1,1)) în loc de 5\\%'),
     [T('residuals are fitted to look uncorrelated at the first lags: their ACF is smaller than that of true white noise', 'reziduurile sînt ajustate astfel încît să pară necorelate la primele laguri: ACF lor este mai mică decît a unui zgomot alb adevărat')]),
    (T('With $m - p - q$: @{lbdf.ar.k}\\% and @{lbdf.arma.k}\\%, close to the nominal 5\\%', 'Cu $m - p - q$: @{lbdf.ar.k}\\% și @{lbdf.arma.k}\\%, aproape de nivelul nominal de 5\\%'),
     [T('the correction matters most when $p + q$ is large relative to $m$', 'corecția contează cel mai mult cînd $p + q$ este mare în raport cu $m$')]),
    T('Software (\\texttt{acorr\\_ljungbox}) uses $m$ by default: pass \\texttt{model\\_df = p + q}', 'Programele (\\texttt{acorr\\_ljungbox}) folosesc implicit $m$: transmiteți \\texttt{model\\_df = p + q}'),
    T('Wrong degrees of freedom make a bad model look acceptable', 'Gradele de libertate greșite fac ca un model prost să pară acceptabil')])

D.recap(('Model selection and diagnostics', 'selecția modelului și diagnosticarea'), [
    T('AIC $= -2\\ln L + 2k$, BIC $= -2\\ln L + k\\ln T$; smallest wins; BIC picks smaller models', 'AIC $= -2\\ln L + 2k$, BIC $= -2\\ln L + k\\ln T$; cîștigă valoarea cea mai mică; BIC alege modele mai mici'),
    T('BIC is consistent, AIC aims at forecasting and overfits sometimes', 'BIC este consistent, AIC este orientat spre prognoză și uneori supraparametrizează'),
    T('Residuals: plot, ACF, Ljung--Box with $m - p - q$ degrees of freedom, Jarque--Bera, Ljung--Box on squares', 'Reziduurile: grafic, ACF, Ljung--Box cu $m - p - q$ grade de libertate, Jarque--Bera, Ljung--Box pe pătrate'),
    T('A model passes when its residuals are white noise; normality matters for the intervals', 'Un model este acceptat cînd reziduurile lui sînt zgomot alb; normalitatea contează pentru intervale')])

# =============================================================================
# 7. PROGNOZA
# =============================================================================
D.section('Forecasting with ARMA models', 'Prognoza cu modele ARMA')

D.frame(T('The optimal forecast', 'Prognoza optimă'), items(
    (T('Forecast of $X_{T+h}$ made at time $T$: $\\hat X_{T+h|T} = E[X_{T+h} \\mid X_T, X_{T-1}, \\dots]$', 'Prognoza lui $X_{T+h}$ făcută la momentul $T$: $\\hat X_{T+h|T} = E[X_{T+h} \\mid X_T, X_{T-1}, \\dots]$'),
     [T('it minimises the mean squared error (MSE) among all functions of the past', 'minimizează eroarea pătratică medie (MSE) între toate funcțiile de trecut')]),
    (T('Recipe: write the model at $T + h$ and replace', 'Rețeta: scriem modelul la $T + h$ și înlocuim'),
     [T('future shocks $\\varepsilon_{T+j}$, $j \\ge 1$, by 0', 'șocurile viitoare $\\varepsilon_{T+j}$, $j \\ge 1$, cu 0'),
      T('future values $X_{T+j}$ by their forecasts; past values and past residuals stay as they are', 'valorile viitoare $X_{T+j}$ cu prognozele lor; valorile trecute și reziduurile trecute rămîn neschimbate')]),
    (T('AR(1): $\\hat X_{T+h|T} = \\mu + \\phi^h(X_T - \\mu)$: \\textbf{mean reversion} at the speed $\\phi$', 'AR(1): $\\hat X_{T+h|T} = \\mu + \\phi^h(X_T - \\mu)$: \\textbf{revenirea la medie} cu viteza $\\phi$'),
     [T('MA($q$): $\\hat X_{T+h|T} = \\mu$ for $h > q$: after $q$ steps the model knows nothing beyond the mean', 'MA($q$): $\\hat X_{T+h|T} = \\mu$ pentru $h > q$: după $q$ pași modelul nu mai conține altă informație decît media')])))

D.frame(T('Forecast errors and intervals', 'Erorile de prognoză și intervalele'), items(
    (T('From the causal form $X_{T+h} = \\mu + \\sum_j\\psi_j\\varepsilon_{T+h-j}$:', 'Din forma cauzală $X_{T+h} = \\mu + \\sum_j\\psi_j\\varepsilon_{T+h-j}$:'),
     [T('error $e_{T+h} = X_{T+h} - \\hat X_{T+h|T} = \\varepsilon_{T+h} + \\psi_1\\varepsilon_{T+h-1} + \\dots + \\psi_{h-1}\\varepsilon_{T+1}$', 'eroarea $e_{T+h} = X_{T+h} - \\hat X_{T+h|T} = \\varepsilon_{T+h} + \\psi_1\\varepsilon_{T+h-1} + \\dots + \\psi_{h-1}\\varepsilon_{T+1}$'),
      T('variance $\\sigma_h^2 = \\sigma^2(1 + \\psi_1^2 + \\dots + \\psi_{h-1}^2)$, increasing in $h$', 'varianța $\\sigma_h^2 = \\sigma^2(1 + \\psi_1^2 + \\dots + \\psi_{h-1}^2)$, crescătoare în $h$')]),
    (T('\\textbf{95\\% interval}: $\\hat X_{T+h|T} \\pm 1.96\\,\\sigma_h$ (Gaussian shocks)', '\\textbf{Intervalul de 95\\%}: $\\hat X_{T+h|T} \\pm 1{,}96\\,\\sigma_h$ (șocuri gaussiene)'),
     [T('$h = 1$: $\\pm 1.96\\,\\sigma$; $h \\to \\infty$: $\\sigma_h^2 \\to \\gamma(0)$, the unconditional variance', '$h = 1$: $\\pm 1{,}96\\,\\sigma$; $h \\to \\infty$: $\\sigma_h^2 \\to \\gamma(0)$, varianța necondiționată')]),
    (T('What the interval leaves out', 'Limitele intervalului'),
     [T('parameter uncertainty ($\\hat\\phi$ instead of $\\phi$), model uncertainty, fat tails: real coverage is usually below 95\\%', 'incertitudinea parametrilor ($\\hat\\phi$ în loc de $\\phi$), incertitudinea modelului, cozile groase: acoperirea reală este de obicei sub 95\\%')]),
    T('Errors of forecasts made at the same time for horizons $h$ and $h + 1$ are correlated: they share shocks', 'Erorile prognozelor făcute în același moment pentru orizonturile $h$ și $h + 1$ sînt corelate: au șocuri comune')))

D.frame(T('Worked example: forecasting an AR(1)', 'Exemplu rezolvat: prognoza unui AR(1)'), items(
    T('$X_t = 2 + 0.8X_{t-1} + \\varepsilon_t$, $\\sigma^2 = 1$, $\\mu = @{we.ar.mu}$; last value $X_T = 12$', '$X_t = 2 + 0{,}8X_{t-1} + \\varepsilon_t$, $\\sigma^2 = 1$, $\\mu = @{we.ar.mu}$; ultima valoare $X_T = 12$'),
    (T('Point forecasts', 'Prognozele punctuale'),
     [T('$\\hat X_{T+1} = 10 + 0.8 \\cdot 2 = @{we.fc.f1}$; $\\hat X_{T+2} = 10 + 0.64 \\cdot 2 = @{we.fc.f2}$; $\\hat X_{T+10} = @{we.fc.f10}$', '$\\hat X_{T+1} = 10 + 0{,}8 \\cdot 2 = @{we.fc.f1}$; $\\hat X_{T+2} = 10 + 0{,}64 \\cdot 2 = @{we.fc.f2}$; $\\hat X_{T+10} = @{we.fc.f10}$')]),
    (T('Error variances: $\\sigma_1^2 = 1$, $\\sigma_2^2 = 1 + 0.8^2 = @{we.fc.v2}$', 'Varianțele erorilor: $\\sigma_1^2 = 1$, $\\sigma_2^2 = 1 + 0{,}8^2 = @{we.fc.v2}$'),
     [T('95\\% intervals: $h = 1$: $[@{we.fc.lo1}, @{we.fc.hi1}]$; $h = 2$: $[@{we.fc.lo2}, @{we.fc.hi2}]$', 'intervalele de 95\\%: $h = 1$: $[@{we.fc.lo1}; @{we.fc.hi1}]$; $h = 2$: $[@{we.fc.lo2}; @{we.fc.hi2}]$')]),
    T('Long run: forecast $\\to 10$, standard deviation $\\to \\sqrt{\\gamma(0)} = @{we.fc.sdinf}$', 'Pe termen lung: prognoza $\\to 10$, abaterea standard $\\to \\sqrt{\\gamma(0)} = @{we.fc.sdinf}$')))

chart(T('Mean reversion in forecasts', 'Revenirea la medie în prognoze'), 'tsa_ch2_forecast_theory', 'TSA_ch2_forecasting', [
    T('Start at $X_T = 3$, $\\mu = 0$, $\\sigma = 1$; dashed: the unconditional band $\\mu \\pm 1.96\\,\\sigma_X$', 'Pornire din $X_T = 3$, $\\mu = 0$, $\\sigma = 1$; linie întreruptă: banda necondiționată $\\mu \\pm 1{,}96\\,\\sigma_X$')],
    h='0.72\\textheight')

interp(('the forecast paths', 'traiectoriilor de prognoză'), [
    (T('$\\phi = 0.9$: slow reversion; after 5 periods the forecast is still @{ft.9.f5}, standard error @{ft.9.se5}', '$\\phi = 0{,}9$: revenire lentă; după 5 perioade prognoza este încă @{ft.9.f5}, eroarea standard @{ft.9.se5}'),
     [T('the interval approaches $\\pm 1.96 \\times @{ft.9.inf}$ only after many periods', 'intervalul se apropie de $\\pm 1{,}96 \\times @{ft.9.inf}$ abia după multe perioade')]),
    T('$\\phi = 0.5$: after 5 periods the forecast is @{ft.5.f5}, practically the mean', '$\\phi = 0{,}5$: după 5 perioade prognoza este @{ft.5.f5}, practic media'),
    T('MA(1): one informative step ($\\theta\\varepsilon_T = 1.2$), then the mean; the interval jumps to its long-run width at $h = 2$', 'MA(1): un singur pas informativ ($\\theta\\varepsilon_T = 1{,}2$), apoi media; intervalul atinge lățimea de lungă durată la $h = 2$'),
    T('A stationary model always forecasts a return to the mean: useful for spreads and growth rates, misleading if the mean shifts', 'Un model staționar prognozează întotdeauna revenirea la medie: util pentru spread-uri și rate de creștere, înșelător dacă media se schimbă')])

D.recap(('Forecasting', 'prognoza'), [
    T('$\\hat X_{T+h|T}$: replace future shocks by 0 and future values by forecasts', '$\\hat X_{T+h|T}$: înlocuim șocurile viitoare cu 0 și valorile viitoare cu prognoze'),
    T('Error variance $\\sigma^2\\sum_{j<h}\\psi_j^2$; intervals widen up to $\\pm 1.96\\sqrt{\\gamma(0)}$', 'Varianța erorii $\\sigma^2\\sum_{j<h}\\psi_j^2$; intervalele se lărgesc pînă la $\\pm 1{,}96\\sqrt{\\gamma(0)}$'),
    T('Forecasts revert to $\\mu$; MA($q$) forecasts equal $\\mu$ beyond $q$ steps', 'Prognozele revin la $\\mu$; prognozele MA($q$) sînt egale cu $\\mu$ după $q$ pași'),
    T('Intervals ignore parameter and model uncertainty', 'Intervalele ignoră incertitudinea parametrilor și a modelului')])

# =============================================================================
# 8. BOX-JENKINS
# =============================================================================
D.section('The Box--Jenkins method', 'Metoda Box--Jenkins')

D.frame(T('George Box and Gwilym Jenkins', 'George Box și Gwilym Jenkins'), cols(items(
    (T('\\textbf{George E. P. Box} (1919--2013), statistician, University of Wisconsin--Madison', '\\textbf{George E. P. Box} (1919--2013), statistician, Universitatea Wisconsin--Madison'),
     [T('``All models are wrong, but some are useful\'\'', '„Toate modelele sînt greșite, dar unele sînt utile”')]),
    (T('\\textbf{Gwilym M. Jenkins} (1932--1982), statistician, Lancaster University', '\\textbf{Gwilym M. Jenkins} (1932--1982), statistician, Universitatea Lancaster'),
     [T('\\textit{Time Series Analysis: Forecasting and Control}, 1970; fourth edition with Reinsel: \\refBJ', '\\textit{Time Series Analysis: Forecasting and Control}, 1970; a patra ediție, cu Reinsel: \\refBJ')]),
    (T('Their contribution: not new models, but a \\textbf{method}', 'Contribuția lor: nu modele noi, ci o \\textbf{metodă}'),
     [T('a loop of identification, estimation and checking that anyone can apply', 'o buclă de identificare, estimare și verificare pe care o poate aplica oricine'),
      T('Box--Pierce (1970) and Ljung--Box (1978) come from this programme', 'testele Box--Pierce (1970) și Ljung--Box (1978) provin din acest program')])),
    ph('box', T('George E. P. Box', 'George E. P. Box'), h='0.50\\textheight'), wl='0.62', wr='0.34'), 'footnotesize')

BJ_TIKZ = r"""
\begin{center}
\begin{tikzpicture}[node distance=0.55cm and 0.5cm, font=\footnotesize,
  bx/.style={draw=MainBlue, rounded corners, fill=MainBlue!8, align=center, minimum height=0.95cm, text width=2.35cm},
  ar/.style={-{Stealth[length=2mm]}, thick, MainBlue}]
\node[bx] (a) {⟦0. Prepare||0. Pregătire⟧\\{\scriptsize ⟦plot, transform, make stationary||grafic, transformare, staționarizare⟧}};
\node[bx, right=of a] (b) {⟦1. Identify||1. Identificare⟧\\{\scriptsize ⟦ACF, PACF: candidates $(p,q)$||ACF, PACF: candidați $(p,q)$⟧}};
\node[bx, right=of b] (c) {⟦2. Estimate||2. Estimare⟧\\{\scriptsize ⟦MLE; AIC, BIC||MLE; AIC, BIC⟧}};
\node[bx, right=of c] (d) {⟦3. Check||3. Verificare⟧\\{\scriptsize ⟦residuals, Ljung--Box, roots||reziduuri, Ljung--Box, rădăcini⟧}};
\node[bx, below=0.9cm of d, fill=Forest!10, draw=Forest] (e) {⟦4. Forecast||4. Prognoză⟧\\{\scriptsize ⟦points and intervals||puncte și intervale⟧}};
\draw[ar] (a) -- (b); \draw[ar] (b) -- (c); \draw[ar] (c) -- (d);
\draw[ar, Forest] (d) -- node[right, font=\scriptsize, text=Forest] {⟦passes||acceptat⟧} (e);
\draw[ar, IDAred] (d.north) -- ++(0, 0.5) -| node[pos=0.25, above, font=\scriptsize, text=IDAred] {⟦fails: back to identification||respins: înapoi la identificare⟧} (b.north);
\end{tikzpicture}
\end{center}
"""

D.frame(T('The Box--Jenkins loop', 'Bucla Box--Jenkins'), BJ_TIKZ + items(
    T('Step 0 is Chapters 1 and 3: logs, differences, unit-root tests; ARMA needs a stationary series', 'Pasul 0 ține de Capitolele 1 și 3: logaritmi, diferențe, teste de rădăcină unitară; ARMA cere o serie staționară'),
    T('The loop ends with the smallest model whose residuals are white noise and whose roots are well inside the admissible region', 'Bucla se încheie cu cel mai mic model ale cărui reziduuri sînt zgomot alb și ale cărui rădăcini sînt bine în interiorul regiunii admise'),
    T('Out-of-sample checks (Chapter 0: training and test samples) complete the method', 'Verificările în afara eșantionului (Capitolul 0: eșantioane de antrenare și de test) completează metoda')), 'footnotesize')

# =============================================================================
# 9. STUDIU DE CAZ: PIB
# =============================================================================
D.section('Case study: Romanian GDP growth', 'Studiu de caz: creșterea PIB-ului României')

D.frame(T('The data', 'Datele'), cols(items(
    (T('Romanian real GDP, chain-linked volumes (2010 prices), quarterly, not seasonally adjusted; Eurostat, \\texttt{namq\\_10\\_gdp}', 'PIB-ul real al României, volume înlănțuite (prețurile din 2010), trimestrial, neajustat sezonier; Eurostat, \\texttt{namq\\_10\\_gdp}'),
     [T('the national source: the National Institute of Statistics (INS)', 'sursa națională: Institutul Național de Statistică (INS)')]),
    (T('Annual growth $y_t = 100\\,(\\ln Y_t - \\ln Y_{t-4})$, in \\%: removes the season (Chapter 1)', 'Creșterea anuală $y_t = 100\\,(\\ln Y_t - \\ln Y_{t-4})$, în \\%: elimină sezonalitatea (Capitolul 1)'),
     [T('@{gdp.q0}--@{gdp.q1}: $T = @{gdp.T}$ quarters, mean @{gdp.mean}\\%, standard deviation @{gdp.sd}\\%', '@{gdp.q0}--@{gdp.q1}: $T = @{gdp.T}$ trimestre, media @{gdp.mean}\\%, abaterea standard @{gdp.sd}\\%'),
      T('extremes: $@{gdp.min}\\%$ in @{gdp.mind}, $+@{gdp.max}\\%$ in @{gdp.maxd}; last value $@{gdp.last}\\%$', 'extremele: $@{gdp.min}\\%$ în @{gdp.mind}, $+@{gdp.max}\\%$ în @{gdp.maxd}; ultima valoare $@{gdp.last}\\%$')]),
    T('Stationary? No trend in the growth rate; a formal test is in Chapter 3', 'Este staționară? Rata de creștere nu are trend; testul formal este în Capitolul 3')),
    ph('ins', T('The National Institute of Statistics, Bucharest', 'Institutul Național de Statistică, București'), h='0.42\\textheight'),
    wl='0.60', wr='0.36'), 'footnotesize')

chart(T('Step 1: identification', 'Pasul 1: identificarea'), 'tsa_ch2_gdp_ident', 'TSA_ch2_gdp_case', [
    T('Annual growth of Romanian real GDP, its ACF and PACF with $\\pm 1.96/\\sqrt{T} = \\pm @{gdp.band}$', 'Creșterea anuală a PIB-ului real al României, ACF și PACF cu $\\pm 1{,}96/\\sqrt{T} = \\pm @{gdp.band}$')],
    h='0.72\\textheight')

interp(('the correlograms of GDP growth', 'corelogramelor creșterii PIB'), [
    (T('ACF: $@{gdp.r1}$, $@{gdp.r2}$, $@{gdp.r3}$, then $@{gdp.r4}$ at lag 4: \\textbf{cuts off after lag 3}', 'ACF: $@{gdp.r1}$; $@{gdp.r2}$; $@{gdp.r3}$, apoi $@{gdp.r4}$ la lagul 4: \\textbf{se anulează după lagul 3}'),
     [T('candidate: MA(3)', 'candidat: MA(3)')]),
    (T('PACF: $@{gdp.p1}$ at lag 1, then small, except $@{gdp.p5}$ at lag 5', 'PACF: $@{gdp.p1}$ la lagul 1, apoi valori mici, cu excepția lui $@{gdp.p5}$ la lagul 5'),
     [T('candidate: AR(1); lag 5 may be chance or a remnant of the season', 'candidat: AR(1); lagul 5 poate fi întîmplător sau o urmă a sezonalității')]),
    (T('Why MA(3)? $y_t$ is the sum of four quarterly growth rates, $y_t \\approx g_t + g_{t-1} + g_{t-2} + g_{t-3}$', 'Motivul pentru MA(3): $y_t$ este suma a patru rate trimestriale de creștere, $y_t \\approx g_t + g_{t-1} + g_{t-2} + g_{t-3}$'),
     [T('if $g_t$ is close to white noise (Seminar 2, B2), consecutive annual rates share three quarters: an MA(3) by construction', 'dacă $g_t$ este aproape de un zgomot alb (Seminarul 2, B2), ratele anuale consecutive au trei trimestre comune: un MA(3) prin construcție')]),
    T('Ljung--Box $Q^*(8) = @{gdp.lb8}$ on the series itself: strongly autocorrelated, so a model is needed', 'Ljung--Box $Q^*(8) = @{gdp.lb8}$ pentru seria însăși: puternic autocorelată, deci este nevoie de un model')])

D.frame(T('Step 2: estimation and information criteria', 'Pasul 2: estimarea și criteriile informaționale'), table(
    'lrrrrrr', T('\\textbf{Model}', '\\textbf{Modelul}') + ' & $k$ & $\\ln L$ & \\textbf{AIC} & \\textbf{BIC} & $\\hat\\sigma$ & ' + T('\\textbf{LB p-value}', '\\textbf{LB, p-value}'),
    [f'ARMA({r[0]},{r[1]}) & @{{gt.{r}.k}} & $@{{gt.{r}.ll}}$ & @{{gt.{r}.aic}} & @{{gt.{r}.bic}} & @{{gt.{r}.sig}} & @{{gt.{r}.lbp}}' for r in ROWS_GDP],
    size='footnotesize') + items(
    T('Exact Gaussian MLE on the same @{gdp.T} quarters; LB = Ljung--Box $Q^*(8)$ on the residuals, $8 - p - q$ degrees of freedom; full grid $p, q \\le 3$ in the Quantlet', 'MLE exactă gaussiană pe aceleași @{gdp.T} trimestre; LB = Ljung--Box $Q^*(8)$ pentru reziduuri, cu $8 - p - q$ grade de libertate; grila completă $p, q \\le 3$ în Quantlet'),
    T('\\textbf{BIC} chooses MA(3); \\textbf{AIC} chooses ARMA(1,3), lower by only @{gt.daic}', '\\textbf{BIC} alege MA(3); \\textbf{AIC} alege ARMA(1,3), mai mic cu doar @{gt.daic}'),
    T('AR(1), the PACF candidate, leaves autocorrelation: LB p-value @{gt.10.lbp}', 'AR(1), candidatul sugerat de PACF, lasă autocorelație: p-value-ul testului LB este @{gt.10.lbp}')), 'footnotesize')

D.frame(T('Interpreting the estimated MA(3)', 'Interpretarea modelului MA(3) estimat'), items(
    (T('$\\hat y_t = @{ma3.mu} + \\hat\\varepsilon_t + @{ma3.t1}\\,\\hat\\varepsilon_{t-1} + @{ma3.t2}\\,\\hat\\varepsilon_{t-2} + @{ma3.t3}\\,\\hat\\varepsilon_{t-3}$, $\\hat\\sigma = @{ma3.sig}$', '$\\hat y_t = @{ma3.mu} + \\hat\\varepsilon_t + @{ma3.t1}\\,\\hat\\varepsilon_{t-1} + @{ma3.t2}\\,\\hat\\varepsilon_{t-2} + @{ma3.t3}\\,\\hat\\varepsilon_{t-3}$, $\\hat\\sigma = @{ma3.sig}$'),
     [T('standard errors: @{ma3.mus} (mean), @{ma3.s1}, @{ma3.s2}, @{ma3.s3}: all three MA terms are significant', 'erorile standard: @{ma3.mus} (media), @{ma3.s1}; @{ma3.s2}; @{ma3.s3}: toți cei trei termeni MA sînt semnificativi')]),
    (T('The $\\hat\\theta_j$ decline from @{ma3.t1} to @{ma3.t3}: close to, but below, the overlap pattern $(1, 1, 1)$', 'Coeficienții $\\hat\\theta_j$ scad de la @{ma3.t1} la @{ma3.t3}: aproape de tiparul suprapunerii $(1, 1, 1)$, dar sub el'),
     [T('$(1, 1, 1)$ would put roots on the unit circle; the smallest root modulus here is @{ma3.rmin} $> 1$: invertible', '$(1, 1, 1)$ ar pune rădăcini pe cercul unitate; cel mai mic modul al rădăcinilor este aici @{ma3.rmin} $> 1$: invertibil')]),
    (T('ARMA(1,3): $\\hat\\theta_1 = @{arma13.t1} > 1$, yet the roots have modulus at least @{arma13.rmin}: still invertible', 'ARMA(1,3): $\\hat\\theta_1 = @{arma13.t1} > 1$, dar rădăcinile au modulul cel puțin @{arma13.rmin}: tot invertibil'),
     [T('for $q > 1$, invertibility is read from the roots, not from single coefficients', 'pentru $q > 1$, invertibilitatea se citește din rădăcini, nu din coeficienți luați separat'),
      T('$\\hat\\phi = @{arma13.phi}$ (se @{arma13.phis}): one extra parameter for little gain; BIC drops it', '$\\hat\\phi = @{arma13.phi}$ (eroarea standard @{arma13.phis}): un parametru în plus pentru un cîștig mic; BIC renunță la el')]),
    T('Mean growth @{ma3.mu}\\% per year, but with a standard error of @{ma3.mus}: overlapping data carry less information than $T$ suggests', 'Creșterea medie este de @{ma3.mu}\\% pe an, dar cu o eroare standard de @{ma3.mus}: datele care se suprapun conțin mai puțină informație decît sugerează $T$')))

chart(T('Step 3: diagnostics of the MA(3)', 'Pasul 3: diagnosticarea modelului MA(3)'), 'tsa_ch2_gdp_diag', 'TSA_ch2_gdp_case', [
    T('Residuals, their ACF, Ljung--Box p-values for $m = 4, \\dots, 16$ with $m - 3$ degrees of freedom, Normal QQ plot', 'Reziduurile, ACF lor, p-value-urile testului Ljung--Box pentru $m = 4, \\dots, 16$, cu $m - 3$ grade de libertate, graficul QQ față de distribuția Normală')],
    h='0.72\\textheight')

interp(('the MA(3) diagnostics', 'diagnosticării MA(3)'), [
    (T('No residual autocorrelation: $Q^*(8) = @{gd.q8}$ on 5 degrees of freedom, p = @{gd.p8}; $Q^*(16)$: p = @{gd.p16}', 'Fără autocorelație în reziduuri: $Q^*(8) = @{gd.q8}$ cu 5 grade de libertate, p = @{gd.p8}; $Q^*(16)$: p = @{gd.p16}'),
     [T('every p-value in the chart is above @{gd.minp}; with 8 instead of 5 degrees of freedom, p would be @{gd.p8w}, too reassuring', 'toate p-value-urile din grafic sînt peste @{gd.minp}; cu 8 în loc de 5 grade de libertate, p ar fi @{gd.p8w}, prea liniștitor')]),
    (T('Not Normal: Jarque--Bera @{gd.jb}, p @{gd.jbp}; skewness @{gd.skew}, kurtosis @{gd.kurt}', 'Nu este Normal: Jarque--Bera @{gd.jb}, p @{gd.jbp}; asimetria @{gd.skew}, kurtosis-ul @{gd.kurt}'),
     [T('the largest residual: $@{gd.outv}$ in @{gd.outd}, @{gd.outz} standard deviations: the pandemic lockdown', 'cel mai mare reziduu: $@{gd.outv}$ în @{gd.outd}, @{gd.outz} abateri standard: perioada de lockdown din pandemie')]),
    T('Squared residuals: Ljung--Box p = @{gd.psq}, no volatility clustering at this frequency', 'Pătratele reziduurilor: testul Ljung--Box dă p = @{gd.psq}, fără volatility clustering la această frecvență'),
    T('Verdict: the MA(3) captures the dependence; the Normal intervals understate the risk of recessions', 'Verdictul: MA(3) surprinde dependența; intervalele construite cu distribuția Normală subestimează riscul recesiunilor')])

chart(T('Step 4: forecasts', 'Pasul 4: prognozele'), 'tsa_ch2_gdp_forecast', 'TSA_ch2_gdp_case', [
    T('MA(3) (BIC) and AR(1) forecasts for @{gf.q0}--@{gf.q1}, with 95\\% intervals', 'Prognozele MA(3) (BIC) și AR(1) pentru @{gf.q0}--@{gf.q1}, cu intervale de 95\\%')], h='0.72\\textheight')

interp(('the GDP forecasts', 'prognozelor PIB'), [
    (T('MA(3): $@{gf.f1}$, $@{gf.f2}$, $@{gf.f3}$, then the mean $@{gf.f4}$\\% from the fourth quarter on', 'MA(3): $@{gf.f1}$; $@{gf.f2}$; $@{gf.f3}$, apoi media de $@{gf.f4}\\%$ începînd cu al patrulea trimestru'),
     [T('exactly the MA($q$) rule: after $q = 3$ steps the model only knows the mean', 'exact regula MA($q$): după $q = 3$ pași modelul cunoaște doar media')]),
    (T('AR(1): a smooth return to its mean, @{gf.ar8}\\% after 8 quarters (mean @{gf.armu}\\%)', 'AR(1): o revenire netedă la medie, @{gf.ar8}\\% după 8 trimestre (media @{gf.armu}\\%)'),
     [T('the two models agree in the short run and differ in how fast they forget the recent slowdown', 'cele două modele coincid pe termen scurt și diferă prin viteza cu care se estompează efectul încetinirii recente')]),
    T('Intervals: $[@{gf.lo1}, @{gf.hi1}]$ one quarter ahead, $[@{gf.lo4}, @{gf.hi4}]$ from four quarters on: $\\pm @{gf.hw1}$ and $\\pm @{gf.hw4}$ percentage points', 'Intervalele: $[@{gf.lo1}; @{gf.hi1}]$ pentru un trimestru, $[@{gf.lo4}; @{gf.hi4}]$ începînd cu patru trimestre: $\\pm @{gf.hw1}$ și $\\pm @{gf.hw4}$ puncte procentuale'),
    T('A univariate model cannot anticipate a turning point; it tells us how unusual the next value would be', 'Un model univariat nu poate anticipa un punct de întoarcere; ne spune cît de neobișnuită ar fi următoarea valoare')])

D.recap(('Romanian GDP growth', 'creșterea PIB-ului României'), [
    T('Annual growth of quarterly GDP: ACF cuts off after lag 3, MA(3) by BIC, ARMA(1,3) by AIC', 'Creșterea anuală a PIB-ului trimestrial: ACF se anulează după lagul 3, MA(3) după BIC, ARMA(1,3) după AIC'),
    T('The MA(3) is the overlap of four quarters: the data structure explains the model', 'MA(3) reflectă suprapunerea a patru trimestre: structura datelor explică modelul'),
    T('White-noise residuals but fat tails (2020): intervals are too narrow in crises', 'Reziduuri de tip zgomot alb, dar cozi groase (2020): intervalele sînt prea înguste în crize'),
    T('Forecasts reach the mean after three quarters', 'Prognozele ajung la medie după trei trimestre')])

# =============================================================================
# 10. ALTE SERII REALE
# =============================================================================
D.section('More real series', 'Alte serii reale')

chart(T('Romanian inflation: an AR(2) forecast', 'Inflația în România: o prognoză AR(2)'), 'tsa_ch2_inflation', 'TSA_ch2_real_series', [
    T('HICP (harmonised index of consumer prices, Eurostat), 12-month rate in \\%, @{in.T} months to @{in.last}; AR order chosen by BIC among $p \\le 6$', 'IAPC (indicele armonizat al prețurilor de consum, Eurostat), rata pe 12 luni în \\%, @{in.T} luni pînă în @{in.last}; ordinul AR ales prin BIC dintre $p \\le 6$')],
    h='0.72\\textheight')

interp(('the inflation model', 'modelului pentru inflație'), [
    (T('BIC: @{in.bic1} for AR(1), @{in.bic2} for AR(2), @{in.bic3} for AR(3): AR(2)', 'BIC: @{in.bic1} pentru AR(1), @{in.bic2} pentru AR(2), @{in.bic3} pentru AR(3): AR(2)'),
     [T('$\\hat\\phi_1 = @{in.p1}$ (@{in.s1}), $\\hat\\phi_2 = @{in.p2}$ (@{in.s2}); $\\hat\\phi_1 + \\hat\\phi_2 = @{in.sum}$, largest inverse root @{in.mod}', '$\\hat\\phi_1 = @{in.p1}$ (@{in.s1}), $\\hat\\phi_2 = @{in.p2}$ (@{in.s2}); $\\hat\\phi_1 + \\hat\\phi_2 = @{in.sum}$, cea mai mare rădăcină inversă @{in.mod}')]),
    (T('Very close to a unit root: is inflation stationary at all? The test is in Chapter 3', 'Foarte aproape de o rădăcină unitară: este inflația staționară? Testul este în Capitolul 3'),
     [T('the estimated mean, $@{in.mu}\\%$ (se @{in.mus}), is the average of 2005--2026, above the BNR target of 2.5\\% $\\pm$ 1 p.p.', 'media estimată, $@{in.mu}\\%$ (eroarea standard @{in.mus}), este media perioadei 2005--2026, peste ținta BNR de 2,5\\% $\\pm$ 1 p.p.')]),
    (T('Forecast: from @{in.lastv}\\% to @{in.f1}\\% next month and @{in.f24}\\% in @{in.flast}; 95\\% interval $[@{in.lo24}, @{in.hi24}]$', 'Prognoza: de la @{in.lastv}\\% la @{in.f1}\\% luna viitoare și @{in.f24}\\% în @{in.flast}; intervalul de 95\\%: $[@{in.lo24}; @{in.hi24}]$'),
     [T('the model reverts to its historical mean, not to the target: it knows nothing about monetary policy', 'modelul revine la media istorică, nu la țintă: nu conține informații despre politica monetară')]),
    T('Diagnostics fail: Ljung--Box $Q^*(12) = @{in.q12}$, p @{in.p12}: correlation at lag 12 (base effects of the 12-month rate), a job for Chapter 4', 'Diagnosticarea eșuează: Ljung--Box $Q^*(12) = @{in.q12}$, p @{in.p12}: corelație la lagul 12 (efectele de bază ale ratei pe 12 luni), o temă pentru Capitolul 4')], size='footnotesize')

chart(T('Daily returns: BET and EUR/RON', 'Randamente zilnice: BET și EUR/RON'), 'tsa_ch2_returns', 'TSA_ch2_real_series', [
    T('Left: ACF of daily log returns; right: ACF of the squared residuals of an AR(1)', 'Stînga: ACF a randamentelor logaritmice zilnice; dreapta: ACF a pătratelor reziduurilor unui AR(1)')],
    h='0.72\\textheight')

D.frame(T('Interpreting the AR(1) for daily returns', 'Interpretarea modelului AR(1) pentru randamentele zilnice'), table(
    'lrr', ' & \\textbf{BET} & \\textbf{EUR/RON}',
    [T('Observations', 'Observații') + ' & @{bet.T} & @{eurron.T}',
     '$\\hat\\phi$ (se) & $@{bet.phi}$ (@{bet.se}) & $@{eurron.phi}$ (@{eurron.se})',
     T('$t$-ratio', 'Raportul $t$') + ' & @{bet.t} & @{eurron.t}',
     T('$R^2$ (\\%)', '$R^2$ (\\%)') + ' & @{bet.r2} & @{eurron.r2}',
     T('LB $Q^*(10)$ residuals, 9 df (p)', 'LB $Q^*(10)$ reziduuri, 9 g.l. (p)') + ' & @{bet.q} (@{bet.qp}) & @{eurron.q} (@{eurron.qp})',
     T('LB $Q^*(10)$ squared residuals', 'LB $Q^*(10)$ pătratele reziduurilor') + ' & @{bet.q2} & @{eurron.q2}',
     T('Kurtosis of residuals', 'Kurtosis-ul reziduurilor') + ' & @{bet.kurt} & @{eurron.kurt}',
     T('BIC choice, $p, q \\le 2$', 'Alegerea BIC, $p, q \\le 2$') + ' & @{bet.bic} & @{eurron.bic}'],
    size='footnotesize') + items(
    T('Highly significant $\\hat\\phi$, but $R^2$ of 1--3\\%: \\textbf{statistically} real, \\textbf{economically} small after trading costs', '$\\hat\\phi$ foarte semnificativ, dar un $R^2$ de 1--3\\%: real din punct de vedere \\textbf{statistic}, mic din punct de vedere \\textbf{economic} după costurile de tranzacționare'),
    T('Positive $\\hat\\phi$: slow price adjustment (BET), a managed exchange rate (EUR/RON)', '$\\hat\\phi$ pozitiv: ajustarea lentă a prețurilor (BET), un curs de schimb în regim de managed float (EUR/RON)'),
    T('Squared residuals and kurtosis: the mean is modelled, the variance is not; ARMA + GARCH in Chapter 5', 'Pătratele reziduurilor și kurtosis-ul: media este modelată, varianța nu; ARMA + GARCH în Capitolul 5')), 'footnotesize')

D.frame(T('Case study: Yule (1927) and the sunspots', 'Studiu de caz: Yule (1927) și petele solare'), items(
    (T('\\refYule fitted the first autoregression: Wolfer\'s yearly sunspot numbers, 1749--1924', '\\refYule a estimat prima autoregresie: numerele anuale ale petelor solare ale lui Wolfer, 1749--1924'),
     [T('until then, cycles were modelled as sums of sine waves (periodogram)', 'pînă atunci, ciclurile erau modelate ca sume de sinusoide (periodograma)'),
      T('Yule: a ``pendulum bombarded by peas\'\': a damped oscillator kicked by random shocks', 'Yule: un „pendul bombardat cu mazăre”: un oscilator amortizat lovit de șocuri aleatoare')]),
    (T('Our least squares AR(2) on the same years (@{sun.n} observations, statsmodels data):', 'AR(2)-ul nostru, estimat prin cele mai mici pătrate pe aceiași ani (@{sun.n} observații, datele din statsmodels):'),
     [T('$\\hat\\phi_1 = @{sun.p1}$, $\\hat\\phi_2 = @{sun.p2}$: complex roots, modulus of the inverse roots @{sun.mod}', '$\\hat\\phi_1 = @{sun.p1}$, $\\hat\\phi_2 = @{sun.p2}$: rădăcini complexe, modulul rădăcinilor inverse @{sun.mod}'),
      T('implied period $2\\pi/\\arccos(\\hat\\phi_1/(2\\sqrt{-\\hat\\phi_2})) = @{sun.per}$ years: the solar cycle, from two coefficients', 'perioada implicită $2\\pi/\\arccos(\\hat\\phi_1/(2\\sqrt{-\\hat\\phi_2})) = @{sun.per}$ ani: ciclul solar, din doar doi coeficienți')]),
    (T('\\refWalker generalised the equations to AR($p$): the Yule--Walker equations', '\\refWalker a generalizat ecuațiile la AR($p$): ecuațiile Yule--Walker'),
     [T('the start of parametric time series analysis', 'începutul analizei parametrice a seriilor de timp')])), 'footnotesize')

chart(T('The sunspots: AR(2) and the information criteria', 'Petele solare: AR(2) și criteriile informaționale'), 'tsa_ch2_sunspots', 'TSA_ch2_real_series', [
    T('Yearly sunspot numbers, 1700--2008 (@{sun.N} years); AR($p$) by MLE on the full sample', 'Numerele anuale ale petelor solare, 1700--2008 (@{sun.N} ani); AR($p$) prin MLE pe întregul eșantion')],
    h='0.72\\textheight')

interp(('the sunspot models', 'modelelor pentru petele solare'), [
    (T('Full-sample AR(2): $\\hat\\phi = (@{sun.f1}, @{sun.f2})$, close to Yule\'s; it explains @{sun.r2}\\% of the variance one year ahead', 'AR(2) pe întregul eșantion: $\\hat\\phi = (@{sun.f1}; @{sun.f2})$, apropiat de cel al lui Yule; explică @{sun.r2}\\% din varianță cu un an înainte'),
     [T('the one-step predictions follow the cycle, but lag behind at the peaks', 'predicțiile cu un pas urmează ciclul, dar întîrzie la vîrfuri')]),
    (T('AIC and BIC both choose $p = @{sun.aic}$ over $p \\le 10$', 'AIC și BIC aleg amîndouă $p = @{sun.aic}$ dintre $p \\le 10$'),
     [T('the cycle is asymmetric (fast rise, slow decay), which a linear AR(2) cannot reproduce', 'ciclul este asimetric (creștere rapidă, scădere lentă), ceea ce un AR(2) liniar nu poate reproduce')]),
    T('Lesson: two parameters capture the essence; more lags capture details; nonlinearity is beyond ARMA', 'Lecția: doi parametri surprind esențialul; mai multe laguri surprind detaliile; neliniaritatea depășește cadrul ARMA')])

D.frame(T('Case study: forecasting competitions', 'Studiu de caz: competițiile de prognoză'), items(
    (T('\\refMH: the M3 competition, 3003 series (yearly, quarterly, monthly, other), 24 methods', '\\refMH: competiția M3, 3003 serii (anuale, trimestriale, lunare, altele), 24 de metode'),
     [T('forecasts compared out of sample, on several accuracy measures and horizons', 'prognoze comparate în afara eșantionului, după mai multe măsuri de acuratețe și orizonturi')]),
    (T('Conclusions confirmed from the earlier M-competitions', 'Concluzii confirmate față de competițiile M anterioare'),
     [T('statistically sophisticated methods do not necessarily forecast better than simple ones', 'metodele sofisticate statistic nu prognozează neapărat mai bine decît cele simple'),
      T('the ranking depends on the accuracy measure and on the horizon; combinations of methods do well', 'clasamentul depinde de măsura de acuratețe și de orizont; combinațiile de metode au rezultate bune')]),
    (T('Consequences for ARMA', 'Consecințe pentru ARMA'),
     [T('the Box--Jenkins loop must end with an out-of-sample comparison against simple benchmarks (mean, last value, exponential smoothing from Chapter 0)', 'bucla Box--Jenkins trebuie să se încheie cu o comparație în afara eșantionului cu repere simple (media, ultima valoare, netezirea exponențială din Capitolul 0)'),
      T('Seminar 2, C1 runs this comparison for Romanian inflation', 'Seminarul 2, C1 face această comparație pentru inflația din România')])))

D.recap(('More real series', 'alte serii reale'), [
    T('Inflation: AR(2) near a unit root; reverts to the historical mean; lag-12 residual correlation', 'Inflația: AR(2) aproape de o rădăcină unitară; revine la media istorică; corelație reziduală la lagul 12'),
    T('Daily returns: significant but tiny AR effects; the variance needs GARCH', 'Randamentele zilnice: efecte AR semnificative, dar foarte mici; varianța cere GARCH'),
    T('Sunspots: Yule\'s AR(2) gives the 11-year cycle from complex roots', 'Petele solare: AR(2)-ul lui Yule dă ciclul de 11 ani din rădăcini complexe'),
    T('Forecasting competitions: always compare with simple benchmarks', 'Competițiile de prognoză: comparați întotdeauna cu repere simple')])

# =============================================================================
# 11. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Code}: a first draft of a loop that fits ARMA($p,q$) on a grid, tabulates AIC and BIC, and runs the residual tests', '\\textbf{Cod}: o primă versiune a unei bucle care estimează ARMA($p,q$) pe o grilă, tabelează AIC și BIC și aplică testele pe reziduuri'),
    T('\\textbf{Explanation}: a second derivation of the ACF of an ARMA(1,1), or of why the PACF of an MA decays', '\\textbf{Explicații}: o a doua deducere a ACF pentru un ARMA(1,1) sau a motivului pentru care PACF a unui MA descrește'),
    T('\\textbf{Exploration}: the same Box--Jenkins loop on many series (GDP growth of all EU countries) to compare persistence', '\\textbf{Explorare}: aceeași buclă Box--Jenkins pe multe serii (creșterea PIB a tuturor țărilor UE), pentru a compara persistența'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code with statsmodels that fits ARMA(p,q), p,q <= 3, to Romanian annual GDP growth from Eurostat, reports AIC, BIC and the Ljung-Box p-value with model\\_df = p + q, and forecasts 8 quarters from the BIC model.}',
        '\\aiprompt{Write Python code with statsmodels that fits ARMA(p,q), p,q <= 3, to Romanian annual GDP growth from Eurostat, reports AIC, BIC and the Ljung-Box p-value with model\\_df = p + q, and forecasts 8 quarters from the BIC model.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('The sign convention of $\\theta$ and $\\phi$ in the software ($\\theta(L) = 1 + \\theta L$ in statsmodels, $1 - \\theta L$ in some books)', 'Convenția de semn pentru $\\theta$ și $\\phi$ în program ($\\theta(L) = 1 + \\theta L$ în statsmodels, $1 - \\theta L$ în unele manuale)'),
    T('Intercept or mean: the \\texttt{const} of \\texttt{ARIMA} is the mean $\\mu$, not $c$', 'Termenul liber sau media: \\texttt{const} din \\texttt{ARIMA} este media $\\mu$, nu $c$'),
    T('Ljung--Box degrees of freedom on residuals ($m - p - q$), and that all models are compared on the same sample', 'Gradele de libertate ale testului Ljung--Box pentru reziduuri ($m - p - q$) și faptul că toate modelele sînt comparate pe același eșantion'),
    T('Roots of both polynomials, not the size of single coefficients', 'Rădăcinile ambelor polinoame, nu mărimea coeficienților luați separat'),
    T('That the series is stationary before an ARMA is fitted (Chapter 3), and that forecasts are compared with simple benchmarks', 'Că seria este staționară înainte de estimarea unui ARMA (Capitolul 3) și că prognozele sînt comparate cu repere simple'),
    T('Every cited reference: it must exist; check the DOI', 'Fiecare referință citată: trebuie să existe; verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('ARMA($p,q$) replaces the infinite Wold sum by a ratio of two short polynomials, $\\theta(L)/\\phi(L)$', 'ARMA($p,q$) înlocuiește suma Wold infinită cu raportul a două polinoame scurte, $\\theta(L)/\\phi(L)$'),
    T('Stationarity: roots of $\\phi(z)$ outside the unit circle; invertibility: roots of $\\theta(z)$ outside; no common factors', 'Staționaritatea: rădăcinile lui $\\phi(z)$ în afara cercului unitate; invertibilitatea: rădăcinile lui $\\theta(z)$ în afara lui; fără factori comuni'),
    T('ACF cuts off: MA; PACF cuts off: AR; both decay: ARMA', 'ACF se anulează: MA; PACF se anulează: AR; ambele descresc: ARMA'),
    T('Estimate by MLE, choose by AIC/BIC, check residuals with Ljung--Box on $m - p - q$ degrees of freedom', 'Estimați prin MLE, alegeți cu AIC/BIC, verificați reziduurile cu Ljung--Box pe $m - p - q$ grade de libertate'),
    T('Forecasts revert to the mean; intervals widen to $\\pm 1.96\\sqrt{\\gamma(0)}$ and ignore parameter uncertainty', 'Prognozele revin la medie; intervalele se lărgesc pînă la $\\pm 1{,}96\\sqrt{\\gamma(0)}$ și ignoră incertitudinea parametrilor'),
    T('Real data: GDP growth is an MA(3) by construction; inflation is near a unit root; returns need GARCH', 'Datele reale: creșterea PIB este un MA(3) prin construcție; inflația este aproape de o rădăcină unitară; randamentele cer GARCH')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.35}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    ['ARMA($p,q$) & $\\phi(L)(X_t - \\mu) = \\theta(L)\\varepsilon_t$, \\quad $\\psi(L) = \\theta(L)/\\phi(L)$',
     'AR(1) & $\\mu = c/(1 - \\phi)$, \\quad $\\gamma(0) = \\sigma^2/(1 - \\phi^2)$, \\quad $\\rho(h) = \\phi^h$',
     'AR(2) & $\\rho(1) = \\phi_1/(1 - \\phi_2)$, \\quad $\\rho(h) = \\phi_1\\rho(h - 1) + \\phi_2\\rho(h - 2)$',
     'MA(1) & $\\rho(1) = \\theta/(1 + \\theta^2)$, \\quad $\\rho(h) = 0$, $h \\ge 2$; ' + T('invertible', 'invertibil') + ' $|\\theta| < 1$',
     'ARMA(1,1) & $\\psi_j = \\phi^{j-1}(\\phi + \\theta)$, \\quad $\\rho(h) = \\phi\\rho(h - 1)$, $h \\ge 2$',
     'Yule--Walker & $\\hat\\phi = \\hat R^{-1}\\hat\\rho$, \\quad $\\hat\\sigma^2 = \\hat\\gamma(0)(1 - \\hat\\phi^\\top\\hat\\rho)$',
     T('Criteria', 'Criterii') + ' & $\\mathrm{AIC} = -2\\ln L + 2k$, \\quad $\\mathrm{BIC} = -2\\ln L + k\\ln T$',
     T('Residuals', 'Reziduuri') + ' & $Q^*(m) \\sim \\chi^2(m - p - q)$',
     T('Forecast', 'Prognoză') + ' & $\\hat X_{T+h|T} \\pm 1.96\\,\\sigma\\sqrt{\\textstyle\\sum_{j=0}^{h-1}\\psi_j^2}$'],
    size='scriptsize') + '}')

D.frame(T('Self-assessment', 'Autoevaluare'), items(
    (T('\\textbf{Question}: is $X_t = 1.2X_{t-1} - 0.32X_{t-2} + \\varepsilon_t$ stationary?', '\\textbf{Întrebare}: este $X_t = 1{,}2X_{t-1} - 0{,}32X_{t-2} + \\varepsilon_t$ staționar?'),
     [T('\\textbf{Answer}: $1 - 1.2z + 0.32z^2 = (1 - 0.8z)(1 - 0.4z)$, roots 1.25 and 2.5, both outside the circle: yes', '\\textbf{Răspuns}: $1 - 1{,}2z + 0{,}32z^2 = (1 - 0{,}8z)(1 - 0{,}4z)$, rădăcinile 1,25 și 2,5, ambele în afara cercului: da')]),
    (T('\\textbf{Question}: the ACF of a series has one spike $\\hat\\rho(1) = -0.45$ and the PACF decays. Which model?', '\\textbf{Întrebare}: ACF a unei serii are o singură valoare semnificativă, $\\hat\\rho(1) = -0{,}45$, iar PACF descrește. Ce model propuneți?'),
     [T('\\textbf{Answer}: MA(1) with $\\theta < 0$; solve $\\theta/(1 + \\theta^2) = -0.45$ and keep the invertible root', '\\textbf{Răspuns}: MA(1) cu $\\theta < 0$; rezolvăm $\\theta/(1 + \\theta^2) = -0{,}45$ și păstrăm rădăcina invertibilă')]),
    (T('\\textbf{Question}: an ARMA(2,1) leaves $Q^*(10) = 15.2$ in its residuals. Reject at 5\\%?', '\\textbf{Întrebare}: un ARMA(2,1) lasă în reziduuri $Q^*(10) = 15{,}2$. Respingem la 5\\%?'),
     [T('\\textbf{Answer}: degrees of freedom $10 - 3 = 7$, $\\chi^2_{0.95}(7) = 14.07 < 15.2$: reject; with 10 degrees of freedom (critical 18.31) one would wrongly accept', '\\textbf{Răspuns}: gradele de libertate $10 - 3 = 7$, $\\chi^2_{0{,}95}(7) = 14{,}07 < 15{,}2$: respingem; cu 10 grade de libertate (valoarea critică 18,31) am accepta greșit')]),
    T('Next: Chapter 3, unit roots and ARIMA models: what to do when a root sits on the unit circle', 'Urmează: Capitolul 3, rădăcini unitare și modele ARIMA: ce facem cînd o rădăcină se află pe cercul unitate')))

D.references(bib())

if __name__ == '__main__':
    for _k, _v in list(V.items()):   # a true minus sign for negative numbers, in text and in math
        if isinstance(_v, str) and _v.startswith('⁅-'):
            V[_k] = '⁅\\ensuremath{-}' + _v[2:]
    finalize(D.write(V))
