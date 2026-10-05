r"""
build_chapter5.py -- Capitolul 5 (Volatilitate condiționată: ARCH și GARCH), EN + RO dintr-o singură sursă
==========================================================================================================
Text ⟦english||română⟧; cifrele @{cheie} vin din Quantlets/Ch_05/ch5_numbers.json (generate_all_charts.py) sau
sînt calculate aici, în Python, pentru exemplele lucrate. Nicio cifră nu este scrisă de mînă.
Ieșire:
  EN/Courses/chapter5_conditional_volatility_garch.tex
  RO/Cursuri/capitol5_volatilitate_conditionata_garch.tex
Rulare:
  python3 Quantlets/Ch_05/generate_all_charts.py
  python3 latex/build_chapter5.py && python3 latex/tsa_build.py compile 5
"""

import math
import os
import sys

from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsa_build import Deck, cols, table, photo   # noqa: E402
from tsa_build import items as _items   # noqa: E402
from ch5_common import ASSETS, NAMES, QLURL, REFS, T, bib, finalize, load, values   # noqa: E402


def items(*xs):
    """tsa_build.items, with (text, []) treated as a plain bullet."""
    return _items(*[x[0] if isinstance(x, tuple) and not x[1] else x for x in xs])


N = load()
V = values(N)
D = Deck(5, 'lecture', refs=REFS)
C = 'https://commons.wikimedia.org/wiki/File:'


def ql(folder):
    return f'\\quantlet{{{folder.replace("_", chr(92) + "_")}}}{{\\qlurl{{{folder}}}}}'


def chart(title, fig, folder, bullets, h='0.60\\textheight', size='footnotesize'):
    body = (f'\\begin{{center}}\n\\includegraphics[width=0.97\\textwidth,height={h},keepaspectratio]{{{fig}.pdf}}\n'
            f'\\end{{center}}\n\\vspace{{-0.25cm}}\n' + items(*bullets) + '\n' + ql(folder))
    D.frame(title, body, size)


def interp(title, bullets, size='small'):
    D.frame(T(f'Interpreting {title[0]}', f'Interpretarea {title[1]}'), items(*bullets), size)


PH = {
    'engle': ('ch5_robert_engle_2022.jpg', C + '0603-Kraneshares_KRBN-RobertEngle-JonDemske-16_(cropped).jpg',
              T('Photo', 'Foto') + ': Jon Demske (2022); CC BY-SA 4.0; Wikimedia Commons'),
    'stockholm': ('ch5_stockholm_concert_hall.jpg', C + 'Konserthuset_Stockholm_(Stockholm_Concert_Hall).jpg',
                  T('Photo', 'Foto') + ': Karen Zhou (2025); CC BY-SA 4.0; Wikimedia Commons'),
    'ucsd': ('ch5_ucsd_geisel_2010.jpg', C + 'Geisel_Library,_UC_San_Diego.jpg',
             T('Photo', 'Foto') + ': Stephen Bay (2010); CC BY 4.0; Wikimedia Commons'),
    'lehman': ('ch5_lehman_2008.jpg', C + 'Lehman_Brothers-NYC-20080915.jpg',
               T('Photo', 'Foto') + ': Robert Scoble (2008); CC BY 2.0; Wikimedia Commons'),
    'fidi': ('ch5_fidi_2020.jpg', C + 'Subdued_FiDi_(50063555551).jpg',
             T('Photo', 'Foto') + ': Billie Grace Ward (2020); CC BY 2.0; Wikimedia Commons'),
    'bnr': ('ch1_bnr_2018.jpg', C + 'National_Bank_of_Romania_(old_building),_Bucharest_by_nickispeaki_01.jpg',
            T('Photo', 'Foto') + ': Nickispeaki (2018); CC BY-SA 4.0; Wikimedia Commons'),
}


def ph(key, cap, h='0.46\\textheight'):
    f, url, cred = PH[key]
    return photo(f, cap, url, cred, h=h)


# =============================================================================
# EXEMPLE LUCRATE (calculate aici)
# =============================================================================
w1, a1 = 0.5, 0.5                                   # ARCH(1)
V.put('ex.a1.uv', w1 / (1 - a1), 2)
V.put('ex.a1.k', 3 * (1 - a1 ** 2) / (1 - 3 * a1 ** 2), 1)
V.put('ex.a1.s2', w1 + a1 * 2.0 ** 2, 2)
V.put('ex.a1.s', math.sqrt(w1 + a1 * 2.0 ** 2), 2)
V.put('ex.a1.lim', 1 / math.sqrt(3), 3)
w, a, b = 0.02, 0.10, 0.88                          # GARCH(1,1), daily returns in %
V.put('ex.g.pers', a + b, 2)
V.put('ex.g.uv', w / (1 - a - b), 2)
V.put('ex.g.vol', math.sqrt(252 * w / (1 - a - b)), 1)
V.put('ex.g.hl', math.log(0.5) / math.log(a + b), 1)
V.put('ex.g.s2', w + a * 3.0 ** 2 + b * 1.2, 3)
V.put('ex.g.s', math.sqrt(w + a * 3.0 ** 2 + b * 1.2), 2)
V.put('ex.g.k', 3 * (1 - (a + b) ** 2) / (1 - (a + b) ** 2 - 2 * a ** 2), 2)
V.put('ex.g.cond', (a + b) ** 2 + 2 * a ** 2, 4)
for pp_ in (0.90, 0.98, 0.99):
    V.put(f'hl.{int(round(100 * pp_))}', math.log(0.5) / math.log(pp_), 1)
V.put('ltv.v', 0.5 * 0.5 + 0.5 * 4.5, 2)            # law of total variance
V.put('ltv.s', math.sqrt(0.5 * 0.5 + 0.5 * 4.5), 2)
V.put('ltv.calm', math.sqrt(0.5), 2)
V.put('ltv.storm', math.sqrt(4.5), 2)
V.put('ltv.k', 3 * (0.5 * 0.5 ** 2 + 0.5 * 4.5 ** 2) / (0.5 * 0.5 + 0.5 * 4.5) ** 2, 2)
AR = N['arma']['archlm_e']                          # ARCH-LM on the BET AR(1) residuals
V.put('lm.chi', AR['crit'], 2)
TS = N['term']                                      # multi-step forecast, last day
last = sorted(k for k in TS if k[:2] == '20')[-1]
pp = TS['params']
pers = pp['alpha[1]'] + pp['beta[1]']
V.put('fc.pers', pers, 4)
V.put('fc.pers9', pers ** 9, 3)
V.put('fc.s2n', TS[last]['s2next'], 3)
V.put('fc.s2bar', TS['s2bar'], 3)
V.put('fc.s210', TS['s2bar'] + pers ** 9 * (TS[last]['s2next'] - TS['s2bar']), 3)
V.put('fc.sum10', TS[last]['sum10'], 2)
V.put('fc.vol10', math.sqrt(TS[last]['sum10']), 2)
V.put('fc.vol10sq', math.sqrt(10 * TS[last]['s2next']), 2)
nu = pp['nu']                                       # VaR 1%
qt = stats.t.ppf(0.01, nu) * math.sqrt((nu - 2) / nu)
V.put('var.nu', nu, 2)
V.put('var.t', stats.t.ppf(0.01, nu), 3)
V.put('var.sc', math.sqrt((nu - 2) / nu), 3)
V.put('var.qt', qt, 3)
V.put('var.qa', -qt, 3)
V.put('var.qn', stats.norm.ppf(0.01), 3)
V.put('var.mu', pp['mu'], 3)
V.put('var.sig', math.sqrt(TS[last]['s2next']), 3)
V.put('var.v1', -(pp['mu'] + math.sqrt(TS[last]['s2next']) * qt), 2)
V.put('var.vn', -(pp['mu'] + math.sqrt(TS[last]['s2next']) * stats.norm.ppf(0.01)), 2)
V.put('var.v10', -qt * math.sqrt(TS[last]['sum10']), 2)
FE = N['fe']
V.put('fe.eg.min', min(100 * FE[k]['exc_garch'] for k in FE), 1)
V.put('fe.eg.max', max(100 * FE[k]['exc_garch'] for k in FE), 1)
V.put('fe.ee.min', min(100 * FE[k]['exc_ewma'] for k in FE), 1)
V.put('fe.ee.max', max(100 * FE[k]['exc_ewma'] for k in FE), 1)
V.put('v.ratio', N['vol2']['btc']['median'] / N['vol1']['sp500']['median'], 1)
V.put('mg.lam', N['markets']['eurron']['params']['beta[1]'], 2)
V.put('mg.lamb', N['markets']['btc']['params']['beta[1]'], 2)
SY = N['sty']
qr = [SY[k]['lb_r2'][0] / SY[k]['lb_r'][0] for k in ASSETS]
V.put('sty.qmin', min(qr), 0)
V.put('sty.qmax', max(qr), 0)
V.put('sty.kmin', min(SY[k]['kurt'] for k in ASSETS), 1)
V.put('sty.kmax', max(SY[k]['kurt'] for k in ASSETS), 1)
V.put('m.nu.min', min(N['markets'][k]['params']['nu'] for k in ASSETS), 2)
V.put('m.nu.max', max(N['markets'][k]['params']['nu'] for k in ASSETS), 2)
V.put('yrs', SY['sp500']['n'] / SY['sp500']['ppy'], 0)

# =============================================================================
# DESCHIDERE
# =============================================================================
D.frame(T("Today's question and route", 'Întrebarea de azi și traseul'), items(
    (T('\\textbf{Question}: if tomorrow\'s return cannot be predicted, can tomorrow\'s \\textbf{risk} be predicted?',
       '\\textbf{Întrebarea}: dacă randamentul de mîine nu poate fi anticipat, poate fi anticipat \\textbf{riscul} de mîine?'),
     [T('Chapters 1--3 modelled the conditional \\textbf{mean} of a series; this chapter models its conditional \\textbf{variance}',
        'Capitolele 1--3 au modelat \\textbf{media} condiționată a unei serii; acest capitol modelează \\textbf{varianța} ei condiționată')]),
    (T('\\textbf{Route} of the chapter', '\\textbf{Traseul} capitolului'),
     [T('stylised facts of returns; the ACF of squared returns and the ARCH-LM test', 'faptele stilizate ale randamentelor; ACF a pătratelor randamentelor și testul ARCH-LM'),
      T('conditional mean and conditional variance; ARCH($q$) and GARCH(1,1); persistence, IGARCH and EWMA', 'media și varianța condiționată; ARCH($q$) și GARCH(1,1); persistența, IGARCH și EWMA'),
      T('estimation by maximum likelihood with the \\texttt{arch} package; Student-t innovations; ARMA-GARCH models', 'estimarea prin verosimilitate maximă cu pachetul \\texttt{arch}; inovații Student-t; modele ARMA-GARCH'),
      T('asymmetry (GJR, EGARCH), diagnostics, variance forecasts and their evaluation, VaR 1\\%', 'asimetria (GJR, EGARCH), diagnosticarea, prognoza varianței și evaluarea ei, VaR 1\\%')]),
    T('Data: S\\&P 500, BET, EUR/RON (BNR reference rate) and Bitcoin, daily; Seminar 5 comes before this lecture',
      'Date: S\\&P 500, BET, EUR/RON (cursul de referință BNR) și Bitcoin, zilnic; Seminarul 5 are loc înaintea acestui curs')))

D.frame(T('Learning outcomes', 'Rezultatele învățării'), items(
    T('List the stylised facts of financial returns and test for ARCH effects with the ACF of squared returns and the ARCH-LM test',
      'Enumerați faptele stilizate ale randamentelor financiare și testați efectele ARCH cu ACF a pătratelor și testul ARCH-LM'),
    T('Distinguish the conditional from the unconditional variance, and explain why returns can be uncorrelated yet dependent',
      'Deosebiți varianța condiționată de cea necondiționată și explicați de ce randamentele pot fi necorelate, dar dependente'),
    T('Write ARCH($q$), GARCH(1,1) and ARMA-GARCH models; compute the persistence, the long-run variance and the half-life',
      'Scrieți modelele ARCH($q$), GARCH(1,1) și ARMA-GARCH; calculați persistența, varianța de lungă durată și timpul de înjumătățire'),
    T('Estimate GARCH models by maximum likelihood in Python, choose the innovation distribution and check the standardised residuals',
      'Estimați modele GARCH prin verosimilitate maximă în Python, alegeți distribuția inovațiilor și verificați reziduurile standardizate'),
    T('Measure asymmetry with GJR-GARCH, EGARCH and the news impact curve', 'Măsurați asimetria cu GJR-GARCH, EGARCH și curba de impact a știrilor'),
    T('Forecast the variance over several horizons, evaluate the forecasts with QLIKE and compute a one-day VaR 1\\%',
      'Prognozați varianța pe mai multe orizonturi, evaluați prognozele cu QLIKE și calculați un VaR 1\\% pe o zi')))

D.frame(T('Reading and tools', 'Bibliografie și instrumente'), items(
    (T('Textbook: \\refHP, Ch.~6 (financial time series: ARCH and GARCH models)', 'Manual: \\refHP, cap.~6 (serii de timp financiare: modele ARCH și GARCH)'),
     [T('further reading: \\refTsay, Ch.~3; \\refFHH, Ch.~13', 'lectură suplimentară: \\refTsay, cap.~3; \\refFHH, cap.~13')]),
    T('Theory: \\refHamilton, Ch.~21 (time series models of heteroskedasticity)', 'Teorie: \\refHamilton, cap.~21 (modele de serii de timp cu heteroscedasticitate)'),
    (T('Python Quantlets of this chapter: \\href{' + QLURL + '}{Quantlets/Ch\\_05}', 'Quantlet-urile Python ale capitolului: \\href{' + QLURL + '}{Quantlets/Ch\\_05}'),
     [T('estimation with the package \\texttt{arch} (install it in Colab with \\texttt{pip install arch}); tests with \\texttt{statsmodels}',
        'estimarea cu pachetul \\texttt{arch} (în Colab se instalează cu \\texttt{pip install arch}); testele cu \\texttt{statsmodels}')]),
    T('Lecture notebook: \\href{\\colaburl{notebooks/EN/chapter5_lecture_notebook.ipynb}}{open in Google Colab}',
      'Notebook-ul cursului: \\href{\\colaburl{notebooks/EN/chapter5_lecture_notebook.ipynb}}{deschideți în Google Colab}'),
    T('Video course: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}',
      'Curs video: \\quantinar{Applied Time Series Analysis with Python}{https://quantinar.com/course/137/applied-time-series-analysis-with-python}')))

# =============================================================================
# 1. FAPTELE STILIZATE
# =============================================================================
D.section('Stylised facts of financial returns', 'Faptele stilizate ale randamentelor financiare')

chart(T('Four markets, one pattern', 'Patru piețe, același tipar'), 'tsa_ch5_returns', 'TSA_ch5_stylised_facts', [
    T('Daily log returns $r_t = 100(\\ln P_t - \\ln P_{t-1})$, in \\%, to @{end}: S\\&P 500 and BET since 2000, EUR/RON (BNR reference rate) since July 2005, Bitcoin since September 2014',
      'Randamente logaritmice zilnice $r_t = 100(\\ln P_t - \\ln P_{t-1})$, în \\%, pînă la @{end}: S\\&P 500 și BET din 2000, EUR/RON (cursul de referință BNR) din iulie 2005, Bitcoin din septembrie 2014'),
    T('Shaded: the global financial crisis (September 2008 -- March 2009) and the COVID-19 crash (February -- May 2020)',
      'Zonele colorate: criza financiară globală (septembrie 2008 -- martie 2009) și crahul COVID-19 (februarie -- mai 2020)')], h='0.64\\textheight')

interp(('the four series', 'celor patru serii'), [
    (T('Calm years alternate with storms: large changes follow large changes, of either sign', 'Anii liniștiți alternează cu furtunile: variațiile mari urmează după variații mari, de orice semn'),
     [T('this is \\textbf{volatility clustering}, first described by \\refMandelbrot', 'este fenomenul de \\textbf{volatility clustering}, descris prima dată de \\refMandelbrot')]),
    T('S\\&P 500: daily standard deviation @{ret.sp500.sd2017}\\% in 2017, @{ret.sp500.sd2008}\\% from September 2008 to March 2009 ($\\times$@{ret.sp500.r08}) and @{ret.sp500.sd2020}\\% in February--May 2020 ($\\times$@{ret.sp500.r20})',
      'S\\&P 500: abaterea standard zilnică @{ret.sp500.sd2017}\\% în 2017, @{ret.sp500.sd2008}\\% din septembrie 2008 pînă în martie 2009 ($\\times$@{ret.sp500.r08}) și @{ret.sp500.sd2020}\\% în februarie--mai 2020 ($\\times$@{ret.sp500.r20})'),
    T('EUR/RON: @{ret.eurron.sd2008}\\% in the 2008 crisis, only @{ret.eurron.sd2020}\\% in 2020: the BNR kept the leu almost fixed', 'EUR/RON: @{ret.eurron.sd2008}\\% în criza din 2008, doar @{ret.eurron.sd2020}\\% în 2020: BNR a menținut leul aproape fix'),
    T('A model with a constant variance (white noise, Chapter 1) describes none of these series', 'Un model cu varianță constantă (zgomot alb, Capitolul 1) nu descrie niciuna dintre aceste serii')])

D.frame(T('Two storms: 2008 and 2020', 'Două furtuni: 2008 și 2020'), cols(
    ph('lehman', T('Lehman Brothers headquarters, New York, 15 September 2008', 'Sediul Lehman Brothers, New York, 15 septembrie 2008'), h='0.42\\textheight'),
    ph('fidi', T('The Financial District of New York, 25 March 2020', 'Districtul financiar din New York, 25 martie 2020'), h='0.42\\textheight'),
    wl='0.48', wr='0.48') + items(
    T('15 September 2008: Lehman Brothers files for bankruptcy; March 2020: the COVID-19 pandemic closes economies',
      '15 septembrie 2008: Lehman Brothers intră în faliment; martie 2020: pandemia COVID-19 închide economiile'),
    T('In both episodes calm returned only after months: a shock to volatility is persistent', 'În ambele episoade calmul a revenit abia după cîteva luni: un șoc de volatilitate este persistent')), 'footnotesize')


def strow(k):
    return (f'{NAMES[k]} & @{{sty.{k}.n}} & $@{{sty.{k}.sd}}$ & $@{{sty.{k}.skew}}$ & $@{{sty.{k}.kurt}}$ & $@{{sty.{k}.rr}}$ & '
            f'$@{{sty.{k}.rr2}}$ & @{{sty.{k}.q}} & @{{sty.{k}.q2}} & @{{sty.{k}.lm}}')


D.frame(T('Stylised facts in numbers', 'Faptele stilizate în cifre'), table(
    'lrrrrrrrrr', T('& $T$ & s.d. & skew. & kurt. & $\\hat\\rho_1(r)$ & $\\hat\\rho_1(r^2)$ & $Q(10)$, $r$ & $Q(10)$, $r^2$ & LM(5)',
                    '& $T$ & ab. std. & asim. & boltire & $\\hat\\rho_1(r)$ & $\\hat\\rho_1(r^2)$ & $Q(10)$, $r$ & $Q(10)$, $r^2$ & LM(5)'),
    [strow(k) for k in ASSETS], size='scriptsize') + items(
    (T('\\textbf{Heavy tails}: kurtosis between @{sty.kmin} and @{sty.kmax}, far above 3, the value of the Normal distribution',
       '\\textbf{Cozi groase}: coeficientul de boltire între @{sty.kmin} și @{sty.kmax}, mult peste 3, valoarea distribuției Normale'),
     [T('kurtosis $K = E[(r - \\mu)^4]/\\sigma^4$; skewness $E[(r - \\mu)^3]/\\sigma^3$ (Chapter 1)', 'coeficientul de boltire $K = E[(r - \\mu)^4]/\\sigma^4$; asimetria $E[(r - \\mu)^3]/\\sigma^3$ (Capitolul 1)')]),
    T('\\textbf{Little autocorrelation in $r_t$}, strong autocorrelation in $r_t^2$: $Q(10)$ of the squares is @{sty.qmin}--@{sty.qmax} times larger',
      '\\textbf{Autocorelație mică în $r_t$}, autocorelație puternică în $r_t^2$: $Q(10)$ al pătratelor este de @{sty.qmin}--@{sty.qmax} de ori mai mare'),
    T('$Q(10)$: the Ljung--Box statistic \\refLB\\ (Chapter 1), $\\chi^2(10)$, 5\\% value 18.31; LM(5): the ARCH-LM test (two slides ahead), $\\chi^2(5)$, 5\\% value @{lm.chi}',
      '$Q(10)$: statistica Ljung--Box \\refLB\\ (Capitolul 1), $\\chi^2(10)$, valoarea de 5\\%: 18,31; LM(5): testul ARCH-LM (peste două slide-uri), $\\chi^2(5)$, valoarea de 5\\%: @{lm.chi}')) + ql('TSA_ch5_stylised_facts'), 'footnotesize')

chart(T('The sign is unpredictable, the size is not', 'Semnul este imprevizibil, mărimea nu'), 'tsa_ch5_acf_squares', 'TSA_ch5_stylised_facts', [
    T('Sample ACF (autocorrelation function, Chapter 1) of $r_t$ and of $r_t^2$, lags 1--50, with the band $\\pm 1.96/\\sqrt{T}$ of an i.i.d.\\ (independent and identically distributed) series',
      'ACF de selecție (autocorrelation function, funcția de autocorelație, Capitolul 1) a lui $r_t$ și a lui $r_t^2$, decalajele 1--50, cu banda $\\pm 1.96/\\sqrt{T}$ a unei serii i.i.d.\\ (independente și identic distribuite)')],
    h='0.56\\textheight')

interp(('the two ACFs', 'celor două ACF'), [
    (T('Returns: almost all autocorrelations inside the band (BET: $\\hat\\rho_1 = @{sty.bet.rr}$, a thin market; Chapter 2)', 'Randamentele: aproape toate autocorelațiile sînt în bandă (BET: $\\hat\\rho_1 = @{sty.bet.rr}$, o piață mai puțin lichidă; Capitolul 2)'),
     [T('the direction of tomorrow\'s move is almost unpredictable', 'direcția variației de mîine este aproape imprevizibilă')]),
    (T('Squared returns: S\\&P 500 @{aq.sp500.r1} at lag 1 and still @{aq.sp500.r50} at lag 50; @{aq.sp500.out} of 50 lags outside the band',
       'Pătratele randamentelor: S\\&P 500 @{aq.sp500.r1} la decalajul 1 și încă @{aq.sp500.r50} la decalajul 50; @{aq.sp500.out} din 50 de decalaje în afara benzii'),
     [T('the size of tomorrow\'s move is predictable: a large move today makes a large move tomorrow more likely', 'mărimea variației de mîine este previzibilă: o variație mare azi face mai probabilă o variație mare mîine')]),
    T('Uncorrelated is not the same as independent: $r_t$ is close to white noise but not i.i.d. (Chapter 1)', 'Necorelat nu înseamnă independent: $r_t$ este apropiat de zgomotul alb, dar nu este i.i.d. (Capitolul 1)'),
    T('Bitcoin: weaker but significant ($\\hat\\rho_1(r_t^2) = @{aq.btc.r1}$): a few huge days dominate the squares', 'Bitcoin: mai slab, dar semnificativ ($\\hat\\rho_1(r_t^2) = @{aq.btc.r1}$): cîteva zile extreme domină pătratele')])

D.frame(T('Testing for ARCH effects', 'Testarea efectelor ARCH'), items(
    T('\\textbf{ARCH effects}: the conditional variance depends on the past (the name comes from the ARCH model, Section 3)', '\\textbf{Efecte ARCH}: varianța condiționată depinde de trecut (numele vine de la modelul ARCH, secțiunea 3)'),
    (T('\\textbf{Ljung--Box on squares} (the McLeod--Li test \\refML): $Q(m) = T(T+2)\\sum_{k=1}^{m}\\hat\\rho_k^2(\\hat\\varepsilon^2)/(T-k)$', '\\textbf{Ljung--Box pe pătrate} (testul McLeod--Li \\refML): $Q(m) = T(T+2)\\sum_{k=1}^{m}\\hat\\rho_k^2(\\hat\\varepsilon^2)/(T-k)$'),
     [T('$\\hat\\varepsilon_t$: residuals of the mean model (for returns, $r_t - \\bar r$); under $H_0$ (no ARCH effects), $Q(m) \\sim \\chi^2(m)$', '$\\hat\\varepsilon_t$: reziduurile modelului pentru medie (pentru randamente, $r_t - \\bar r$); în ipoteza $H_0$ (fără efecte ARCH), $Q(m) \\sim \\chi^2(m)$')]),
    (T('\\textbf{ARCH-LM test} \\refEngle: LM (Lagrange multiplier) test from the auxiliary regression', '\\textbf{Testul ARCH-LM} \\refEngle: test LM (Lagrange multiplier, multiplicatorul Lagrange) din regresia auxiliară'),
     [T('$\\hat\\varepsilon_t^2 = b_0 + b_1\\hat\\varepsilon_{t-1}^2 + \\dots + b_q\\hat\\varepsilon_{t-q}^2 + u_t$, estimated by OLS (ordinary least squares)', '$\\hat\\varepsilon_t^2 = b_0 + b_1\\hat\\varepsilon_{t-1}^2 + \\dots + b_q\\hat\\varepsilon_{t-q}^2 + u_t$, estimată prin OLS (ordinary least squares, metoda celor mai mici pătrate)'),
      T('$H_0$: $b_1 = \\dots = b_q = 0$; $\\mathrm{LM} = n R^2 \\sim \\chi^2(q)$, $n$ = number of observations in the regression', '$H_0$: $b_1 = \\dots = b_q = 0$; $\\mathrm{LM} = n R^2 \\sim \\chi^2(q)$, $n$ = numărul de observații din regresie')]),
    T('Both tests are run on the residuals of the mean model, before and after fitting a volatility model', 'Ambele teste se aplică reziduurilor modelului pentru medie, înainte și după estimarea unui model de volatilitate')))

D.frame(T('Worked example: ARCH-LM on the BET', 'Exemplu rezolvat: ARCH-LM pentru BET'), items(
    T('Mean model: AR(1) for the BET returns (Chapter 2), $\\hat\\phi = @{ar.phi}$; residuals $\\hat\\varepsilon_t$; $q = 5$', 'Modelul pentru medie: AR(1) pentru randamentele BET (Capitolul 2), $\\hat\\phi = @{ar.phi}$; reziduurile $\\hat\\varepsilon_t$; $q = 5$'),
    (T('Auxiliary regression (OLS)', 'Regresia auxiliară (OLS)'),
     [T('$\\hat\\varepsilon_t^2 = @{ar.b0} + @{ar.b1}\\,\\hat\\varepsilon_{t-1}^2 + @{ar.b2}\\,\\hat\\varepsilon_{t-2}^2 + @{ar.b3}\\,\\hat\\varepsilon_{t-3}^2 + (@{ar.b4})\\,\\hat\\varepsilon_{t-4}^2 + @{ar.b5}\\,\\hat\\varepsilon_{t-5}^2$',
        '$\\hat\\varepsilon_t^2 = @{ar.b0} + @{ar.b1}\\,\\hat\\varepsilon_{t-1}^2 + @{ar.b2}\\,\\hat\\varepsilon_{t-2}^2 + @{ar.b3}\\,\\hat\\varepsilon_{t-3}^2 + (@{ar.b4})\\,\\hat\\varepsilon_{t-4}^2 + @{ar.b5}\\,\\hat\\varepsilon_{t-5}^2$'),
      T('$R^2 = @{ar.lmr2}$, $n = @{ar.lmn}$', '$R^2 = @{ar.lmr2}$, $n = @{ar.lmn}$')]),
    (T('Step by step: $\\mathrm{LM} = n R^2 = @{ar.lmn} \\times @{ar.lmr2} = @{ar.lm}$; 5\\% critical value $\\chi^2_{0.95}(5) = @{lm.chi}$', 'Pas cu pas: $\\mathrm{LM} = n R^2 = @{ar.lmn} \\times @{ar.lmr2} = @{ar.lm}$; valoarea critică de 5\\% $\\chi^2_{0.95}(5) = @{lm.chi}$'),
     [T('$@{ar.lm} \\gg @{lm.chi}$: reject $H_0$; the variance of the BET depends on the last five squared shocks', '$@{ar.lm} \\gg @{lm.chi}$: respingem $H_0$; varianța BET depinde de pătratele ultimelor cinci șocuri')]),
    T('Interpretation: $R^2 = @{ar.lmr2}$ is small, but for squared returns it is a strong signal; the model of the variance is the next step',
      'Interpretare: $R^2 = @{ar.lmr2}$ este mic, dar pentru pătratele randamentelor este un semnal puternic; următorul pas este un model pentru varianță')) + ql('TSA_ch5_mean_model'), 'footnotesize')

D.recap(('stylised facts', 'faptele stilizate'), [
    T('Heavy tails, volatility clustering, little autocorrelation in $r_t$, strong autocorrelation in $r_t^2$', 'Cozi groase, volatility clustering, autocorelație mică în $r_t$, autocorelație puternică în $r_t^2$'),
    T('ARCH effects: Ljung--Box on squared residuals, ARCH-LM $= nR^2 \\sim \\chi^2(q)$', 'Efectele ARCH: Ljung--Box pe pătratele reziduurilor, ARCH-LM $= nR^2 \\sim \\chi^2(q)$'),
    T('All four series reject $H_0$ by a wide margin', 'Toate cele patru serii resping $H_0$ la mare distanță de pragul critic')])

# =============================================================================
# 2. MEDIA ȘI VARIANȚA CONDIȚIONATĂ
# =============================================================================
D.section('Conditional mean and conditional variance', 'Media condiționată și varianța condiționată')

D.frame(T('Conditioning on the past', 'Condiționarea pe trecut'), items(
    (T('$\\mathcal{F}_{t-1}$: the information available at the end of day $t-1$ (all past values)', '$\\mathcal{F}_{t-1}$: informația disponibilă la sfîrșitul zilei $t-1$ (toate valorile trecute)'),
     [T('conditional expectation $E[X \\mid \\mathcal{F}_{t-1}]$: the best forecast of $X$ given the past (Chapter 2)', 'speranța condiționată $E[X \\mid \\mathcal{F}_{t-1}]$: cea mai bună prognoză a lui $X$, dat fiind trecutul (Capitolul 2)')]),
    (T('\\textbf{Conditional mean}: $\\mu_t = E[r_t \\mid \\mathcal{F}_{t-1}]$', '\\textbf{Media condiționată}: $\\mu_t = E[r_t \\mid \\mathcal{F}_{t-1}]$'),
     [T('an ARMA model (Chapter 2) is a model for $\\mu_t$; for daily returns $\\mu_t$ is small and almost constant', 'un model ARMA (Capitolul 2) este un model pentru $\\mu_t$; pentru randamentele zilnice, $\\mu_t$ este mică și aproape constantă')]),
    (T('\\textbf{Conditional variance}: $\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1}) = E[(r_t - \\mu_t)^2 \\mid \\mathcal{F}_{t-1}]$',
       '\\textbf{Varianța condiționată}: $\\sigma_t^2 = \\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1}) = E[(r_t - \\mu_t)^2 \\mid \\mathcal{F}_{t-1}]$'),
     [T('$\\sigma_t$ is the \\textbf{conditional volatility}: known at $t-1$, it changes from day to day', '$\\sigma_t$ este \\textbf{volatilitatea condiționată}: cunoscută la $t-1$, se schimbă de la o zi la alta'),
      T('\\textbf{conditional heteroskedasticity}: a conditional variance that is not constant', '\\textbf{heteroscedasticitate condiționată}: o varianță condiționată care nu este constantă')]),
    T('\\textbf{Unconditional variance} $\\mathrm{Var}(r_t)$: one number for the whole sample; it can be constant while $\\sigma_t^2$ moves', '\\textbf{Varianța necondiționată} $\\mathrm{Var}(r_t)$: un singur număr pentru tot eșantionul; poate fi constantă în timp ce $\\sigma_t^2$ se schimbă')))

D.frame(T('The basic decomposition', 'Descompunerea de bază'), items(
    (T('Every model of this chapter: $r_t = \\mu_t + \\varepsilon_t$, \\quad $\\varepsilon_t = \\sigma_t z_t$', 'Toate modelele din acest capitol: $r_t = \\mu_t + \\varepsilon_t$, \\quad $\\varepsilon_t = \\sigma_t z_t$'),
     [T('$z_t$: the \\textbf{innovations}, i.i.d.\\ with mean 0 and variance 1 (Normal, Student-t, \\dots)', '$z_t$: \\textbf{inovațiile}, i.i.d., cu media 0 și varianța 1 (Normale, Student-t, \\dots)'),
      T('$\\varepsilon_t$: the \\textbf{shock}; $\\mu_t$ and $\\sigma_t$ depend only on the past', '$\\varepsilon_t$: \\textbf{șocul}; $\\mu_t$ și $\\sigma_t$ depind doar de trecut')]),
    (T('Consequences', 'Consecințe'),
     [T('$E[\\varepsilon_t \\mid \\mathcal{F}_{t-1}] = \\sigma_t E[z_t] = 0$: the shocks are a \\textbf{martingale difference}, hence uncorrelated (white noise)', '$E[\\varepsilon_t \\mid \\mathcal{F}_{t-1}] = \\sigma_t E[z_t] = 0$: șocurile sînt o \\textbf{diferență de martingal}, deci sînt necorelate (zgomot alb)'),
      T('$\\mathrm{Var}(\\varepsilon_t \\mid \\mathcal{F}_{t-1}) = \\sigma_t^2$: the variance is predictable', '$\\mathrm{Var}(\\varepsilon_t \\mid \\mathcal{F}_{t-1}) = \\sigma_t^2$: varianța este previzibilă'),
      T('the $\\varepsilon_t$ are uncorrelated but not independent: $\\varepsilon_t^2$ is correlated with $\\varepsilon_{t-1}^2$', '$\\varepsilon_t$ sînt necorelate, dar nu independente: $\\varepsilon_t^2$ este corelat cu $\\varepsilon_{t-1}^2$')]),
    T('Two models, one for each moment: ARMA for $\\mu_t$ (Chapter 2), GARCH for $\\sigma_t^2$ (this chapter); together: \\textbf{ARMA-GARCH}', 'Două modele, cîte unul pentru fiecare moment: ARMA pentru $\\mu_t$ (Capitolul 2), GARCH pentru $\\sigma_t^2$ (acest capitol); împreună: \\textbf{ARMA-GARCH}')))

D.frame(T('Worked example: conditional and unconditional variance', 'Exemplu rezolvat: varianța condiționată și varianța necondiționată'), items(
    (T('A market has calm days ($\\sigma_t^2 = 0.5$) and stormy days ($\\sigma_t^2 = 4.5$), each with probability $1/2$; the mean is 0',
       'O piață are zile liniștite ($\\sigma_t^2 = 0{,}5$) și zile agitate ($\\sigma_t^2 = 4{,}5$), fiecare cu probabilitatea $1/2$; media este 0'),
     [T('law of total variance: $\\mathrm{Var}(r_t) = E[\\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1})] + \\mathrm{Var}(E[r_t \\mid \\mathcal{F}_{t-1}])$',
        'legea varianței totale: $\\mathrm{Var}(r_t) = E[\\mathrm{Var}(r_t \\mid \\mathcal{F}_{t-1})] + \\mathrm{Var}(E[r_t \\mid \\mathcal{F}_{t-1}])$'),
      T('step by step: $0.5 \\times 0.5 + 0.5 \\times 4.5 + 0 = @{ltv.v}$; unconditional standard deviation $@{ltv.s}\\%$',
        'pas cu pas: $0.5 \\times 0.5 + 0.5 \\times 4.5 + 0 = @{ltv.v}$; abaterea standard necondiționată $@{ltv.s}\\%$')]),
    (T('The unconditional variance is an average; the conditional variance tells which day we are in', 'Varianța necondiționată este o medie; varianța condiționată ne spune în ce fel de zi ne aflăm'),
     [T('a calm day: $\\sigma_t = @{ltv.calm}\\%$; a stormy day: $\\sigma_t = @{ltv.storm}\\%$; the average $@{ltv.s}\\%$ fits neither', 'o zi liniștită: $\\sigma_t = @{ltv.calm}\\%$; o zi agitată: $\\sigma_t = @{ltv.storm}\\%$; media de $@{ltv.s}\\%$ nu descrie niciuna dintre ele')]),
    T('With Normal $z_t$, the mixture has kurtosis $3E[\\sigma_t^4]/(E[\\sigma_t^2])^2 = @{ltv.k}$: a time-varying variance creates heavy tails',
      'Cu $z_t$ Normale, amestecul are coeficientul de boltire $3E[\\sigma_t^4]/(E[\\sigma_t^2])^2 = @{ltv.k}$: o varianță variabilă în timp creează cozi groase')))

chart(T('The mean model leaves ARCH effects', 'Modelul pentru medie lasă efecte ARCH'), 'tsa_ch5_arma_resid', 'TSA_ch5_mean_model', [
    T('BET: left, an AR(1) for the mean (order chosen by BIC among AR(0)--AR(5): $p = @{ar.p}$), constant variance; right, the same AR(1) with a GARCH(1,1)-t variance (Section 6)',
      'BET: stînga, un AR(1) pentru medie (ordinul ales după BIC dintre AR(0)--AR(5): $p = @{ar.p}$), cu varianță constantă; dreapta, același AR(1) cu o varianță GARCH(1,1)-t (secțiunea 6)')], h='0.48\\textheight')

interp(('the residual ACFs', 'ACF ale reziduurilor'), [
    (T('Left: the residuals look almost like white noise, the Box--Jenkins check of Chapter 2 would be nearly passed', 'Stînga: reziduurile arată aproape ca un zgomot alb; verificarea Box--Jenkins din Capitolul 2 ar fi aproape trecută'),
     [T('but their squares: $\\hat\\rho_1 = @{ar.e21}$, $Q(10) = @{ar.qe2}$: strong ARCH effects', 'dar pătratele lor: $\\hat\\rho_1 = @{ar.e21}$, $Q(10) = @{ar.qe2}$: efecte ARCH puternice')]),
    (T('Right: after dividing by $\\hat\\sigma_t$, the squares are clean: $\\hat\\rho_1 = @{ar.z21}$, $Q(10) = @{ar.qz2}$ (p = @{ar.qz2p}), ARCH-LM @{ar.lmz} (p = @{ar.lmzp})',
       'Dreapta: după împărțirea la $\\hat\\sigma_t$, pătratele sînt curate: $\\hat\\rho_1 = @{ar.z21}$, $Q(10) = @{ar.qz2}$ (p = @{ar.qz2p}), ARCH-LM @{ar.lmz} (p = @{ar.lmzp})'),
     [T('some autocorrelation in the level remains ($Q(10) = @{ar.qz}$): the BET mean has more structure than an AR(1)', 'rămîne puțină autocorelație în nivel ($Q(10) = @{ar.qz}$): media BET are mai multă structură decît un AR(1)')]),
    T('Lesson: check both $\\hat\\varepsilon_t$ and $\\hat\\varepsilon_t^2$; a white-noise check on $\\hat\\varepsilon_t$ alone misses the variance', 'Lecția: verificăm atît $\\hat\\varepsilon_t$, cît și $\\hat\\varepsilon_t^2$; o verificare de zgomot alb doar pe $\\hat\\varepsilon_t$ nu vede varianța')])

D.recap(('conditional mean and variance', 'media și varianța condiționată'), [
    T('$r_t = \\mu_t + \\sigma_t z_t$: $\\mu_t$ from an ARMA model, $\\sigma_t^2$ from a volatility model', '$r_t = \\mu_t + \\sigma_t z_t$: $\\mu_t$ dintr-un model ARMA, $\\sigma_t^2$ dintr-un model de volatilitate'),
    T('Shocks are uncorrelated but dependent through their squares', 'Șocurile sînt necorelate, dar dependente prin pătratele lor'),
    T('Unconditional variance = average of the conditional variances; the mixture has heavy tails', 'Varianța necondiționată = media varianțelor condiționate; amestecul are cozi groase')])

# =============================================================================
# 3. ARCH
# =============================================================================
D.section('The ARCH model', 'Modelul ARCH')

D.frame(T('1982: Robert Engle and ARCH', '1982: Robert Engle și modelul ARCH'), cols(items(
    (T('\\refEngle: \\textbf{ARCH} (autoregressive conditional heteroskedasticity): today\'s variance depends on yesterday\'s squared shocks',
       '\\refEngle: \\textbf{ARCH} (autoregressive conditional heteroskedasticity, heteroscedasticitate condiționată autoregresivă): varianța de azi depinde de pătratele șocurilor de ieri'),
     [T('first application: the variance of inflation in the United Kingdom, not a financial market', 'prima aplicație: varianța inflației din Regatul Unit, nu o piață financiară')]),
    (T('\\href{https://www.nobelprize.org/prizes/economic-sciences/2003/summary/}{Nobel Prize 2003}: ``for methods of analyzing economic time series with time-varying volatility (ARCH)\'\'',
       '\\href{https://www.nobelprize.org/prizes/economic-sciences/2003/summary/}{Premiul Nobel 2003}: „pentru metode de analiză a seriilor de timp economice cu volatilitate variabilă în timp (ARCH)”'),
     [T('shared with Clive Granger (Chapter 3: spurious regression; Chapter 7: cointegration); Nobel lecture: \\refEngleN',
        'împărțit cu Clive Granger (Capitolul 3: regresia falsă; Capitolul 7: cointegrarea); prelegerea Nobel: \\refEngleN')])),
    ph('engle', T('Robert F. Engle (2022)', 'Robert F. Engle (2022)'), h='0.25\\textheight') + '\\\\[1mm]\n' +
    ph('stockholm', T('Stockholm Concert Hall, venue of the Nobel Prize ceremony', 'Sala de concerte din Stockholm, unde se decernează Premiile Nobel'), h='0.16\\textheight'),
    wl='0.60', wr='0.36'), 'footnotesize')

D.frame(T('ARCH(1): definition', 'ARCH(1): definiție'), items(
    (T('$\\varepsilon_t = \\sigma_t z_t$, \\quad $\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2$, \\quad $\\omega > 0$, $\\alpha \\ge 0$',
       '$\\varepsilon_t = \\sigma_t z_t$, \\quad $\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2$, \\quad $\\omega > 0$, $\\alpha \\ge 0$'),
     [T('$\\omega > 0$ and $\\alpha \\ge 0$ keep the variance positive', '$\\omega > 0$ și $\\alpha \\ge 0$ asigură o varianță pozitivă'),
      T('a large shock yesterday, of either sign, raises today\'s variance', 'un șoc mare ieri, indiferent de semn, crește varianța de azi')]),
    (T('``Autoregressive\'\': $\\varepsilon_t^2$ follows an AR(1) model (Chapter 2)', '„Autoregresiv”: $\\varepsilon_t^2$ urmează un model AR(1) (Capitolul 2)'),
     [T('write $\\varepsilon_t^2 = \\sigma_t^2 + v_t$, with $v_t = \\sigma_t^2(z_t^2 - 1)$ and $E[v_t \\mid \\mathcal{F}_{t-1}] = 0$', 'scriem $\\varepsilon_t^2 = \\sigma_t^2 + v_t$, cu $v_t = \\sigma_t^2(z_t^2 - 1)$ și $E[v_t \\mid \\mathcal{F}_{t-1}] = 0$'),
      T('then $\\varepsilon_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2 + v_t$: an AR(1) for the squared shocks, with white-noise errors $v_t$', 'atunci $\\varepsilon_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2 + v_t$: un AR(1) pentru pătratele șocurilor, cu erori $v_t$ de tip zgomot alb')]),
    T('Hence the ACF of $\\varepsilon_t^2$ is $\\rho_k = \\alpha^k$ (when the fourth moment exists): clustering, but a fast decay',
      'De aici, ACF a lui $\\varepsilon_t^2$ este $\\rho_k = \\alpha^k$ (cînd momentul de ordinul patru există): clustering, dar o scădere rapidă')))

D.frame(T('ARCH(1): properties', 'ARCH(1): proprietăți'), items(
    T('$E[\\varepsilon_t] = 0$ and $\\mathrm{Cov}(\\varepsilon_t, \\varepsilon_{t-k}) = 0$ for $k \\ne 0$: white noise', '$E[\\varepsilon_t] = 0$ și $\\mathrm{Cov}(\\varepsilon_t, \\varepsilon_{t-k}) = 0$ pentru $k \\ne 0$: zgomot alb'),
    (T('\\textbf{Unconditional variance} (stationary if $\\alpha < 1$): take expectations in $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2$',
       '\\textbf{Varianța necondiționată} (staționar dacă $\\alpha < 1$): aplicăm media în $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2$'),
     [T('$\\bar\\sigma^2 = E[\\varepsilon_t^2] = \\omega + \\alpha\\bar\\sigma^2$, so $\\bar\\sigma^2 = \\omega/(1 - \\alpha)$', '$\\bar\\sigma^2 = E[\\varepsilon_t^2] = \\omega + \\alpha\\bar\\sigma^2$, deci $\\bar\\sigma^2 = \\omega/(1 - \\alpha)$')]),
    (T('\\textbf{Kurtosis} with Normal $z_t$ (if $3\\alpha^2 < 1$): $K = 3\\,\\dfrac{1 - \\alpha^2}{1 - 3\\alpha^2} > 3$',
       '\\textbf{Coeficientul de boltire} cu $z_t$ Normale (dacă $3\\alpha^2 < 1$): $K = 3\\,\\dfrac{1 - \\alpha^2}{1 - 3\\alpha^2} > 3$'),
     [T('Normal innovations, yet heavy-tailed shocks; if $\\alpha \\ge 1/\\sqrt{3} = @{ex.a1.lim}$, the fourth moment is infinite', 'inovații Normale, dar șocuri cu cozi groase; dacă $\\alpha \\ge 1/\\sqrt{3} = @{ex.a1.lim}$, momentul de ordinul patru este infinit')]),
    (T('\\textbf{Worked example}: $\\omega = 0.5$, $\\alpha = 0.5$', '\\textbf{Exemplu rezolvat}: $\\omega = 0{,}5$, $\\alpha = 0{,}5$'),
     [T('$\\bar\\sigma^2 = 0.5/0.5 = @{ex.a1.uv}$; $K = 3 \\times 0.75/0.25 = @{ex.a1.k}$', '$\\bar\\sigma^2 = 0{,}5/0{,}5 = @{ex.a1.uv}$; $K = 3 \\times 0{,}75/0{,}25 = @{ex.a1.k}$'),
      T('after a shock $\\varepsilon_{t-1} = 2$: $\\sigma_t^2 = 0.5 + 0.5 \\times 4 = @{ex.a1.s2}$, $\\sigma_t = @{ex.a1.s}$: the variance more than doubles',
        'după un șoc $\\varepsilon_{t-1} = 2$: $\\sigma_t^2 = 0{,}5 + 0{,}5 \\times 4 = @{ex.a1.s2}$, $\\sigma_t = @{ex.a1.s}$: varianța se dublează și mai mult')])))

chart(T('Simulated paths: i.i.d., ARCH(1) and GARCH(1,1)', 'Traiectorii simulate: i.i.d., ARCH(1) și GARCH(1,1)'), 'tsa_ch5_simulated', 'TSA_ch5_simulation_likelihood', [
    T('Three series of 1000 values with unconditional variance 1: i.i.d.\\ Normal; ARCH(1) with $\\omega = \\alpha = 0.5$; GARCH(1,1) with $\\omega = 0.02$, $\\alpha = 0.10$, $\\beta = 0.88$ (next section)',
      'Trei serii de 1000 de valori cu varianța necondiționată 1: i.i.d.\\ Normale; ARCH(1) cu $\\omega = \\alpha = 0{,}5$; GARCH(1,1) cu $\\omega = 0{,}02$, $\\alpha = 0{,}10$, $\\beta = 0{,}88$ (secțiunea următoare)'),
    T('Sample kurtosis: i.i.d.\\ $@{sim.iid.k}$, ARCH $@{sim.arch.k}$, GARCH $@{sim.garch.k}$ (theory: 3, $@{ex.a1.k}$, $@{sim.garch.kth}$): the fourth moment converges slowly',
      'Coeficientul de boltire de selecție: i.i.d.\\ $@{sim.iid.k}$, ARCH $@{sim.arch.k}$, GARCH $@{sim.garch.k}$ (teoretic: 3, $@{ex.a1.k}$, $@{sim.garch.kth}$): momentul de ordinul patru converge lent'),
    T('Interpretation: ARCH(1) gives isolated spikes; GARCH(1,1) gives long calm and long agitated periods, as in real returns',
      'Interpretare: ARCH(1) produce vîrfuri izolate; GARCH(1,1) produce perioade lungi liniștite și perioade lungi agitate, ca randamentele reale')],
    h='0.46\\textheight')

D.frame(T('ARCH($q$) and its limits', 'ARCH($q$) și limitele lui'), items(
    (T('\\textbf{ARCH($q$)}: $\\sigma_t^2 = \\omega + \\alpha_1\\varepsilon_{t-1}^2 + \\dots + \\alpha_q\\varepsilon_{t-q}^2$, $\\alpha_i \\ge 0$; stationary if $\\sum_i \\alpha_i < 1$',
       '\\textbf{ARCH($q$)}: $\\sigma_t^2 = \\omega + \\alpha_1\\varepsilon_{t-1}^2 + \\dots + \\alpha_q\\varepsilon_{t-q}^2$, $\\alpha_i \\ge 0$; staționar dacă $\\sum_i \\alpha_i < 1$'),
     [T('a shock affects the variance for exactly $q$ days; real clustering lasts for months', 'un șoc influențează varianța exact $q$ zile; clustering-ul real durează luni')])) + table(
    'lrrrrr', T('S\\&P 500, Normal $z_t$ & parameters & log-likelihood & AIC & BIC & $\\sum\\alpha_i$ (+$\\beta$)',
                'S\\&P 500, $z_t$ Normale & parametri & log-verosimilitate & AIC & BIC & $\\sum\\alpha_i$ (+$\\beta$)'),
    ['ARCH(1) & @{aqt.a1.k} & $@{aqt.a1.ll}$ & @{aqt.a1.aic} & @{aqt.a1.bic} & $@{aqt.a1.pers}$',
     'ARCH(5) & @{aqt.a5.k} & $@{aqt.a5.ll}$ & @{aqt.a5.aic} & @{aqt.a5.bic} & $@{aqt.a5.pers}$',
     'ARCH(10) & @{aqt.a10.k} & $@{aqt.a10.ll}$ & @{aqt.a10.aic} & @{aqt.a10.bic} & $@{aqt.a10.pers}$',
     'GARCH(1,1) & @{aqt.g11.k} & $@{aqt.g11.ll}$ & \\textbf{@{aqt.g11.aic}} & \\textbf{@{aqt.g11.bic}} & $@{aqt.g11.pers}$'],
    size='footnotesize') + items(
    T('AIC (Akaike) and BIC (Bayesian information criterion): $-2\\ell + $ penalty for parameters (Chapter 2); smaller is better',
      'AIC (Akaike) și BIC (Bayesian information criterion, criteriul informațional bayesian): $-2\\ell$ + o penalizare pentru parametri (Capitolul 2); valoarea mai mică este mai bună'),
    T('Interpretation: ARCH needs ten lags and still loses to GARCH(1,1), which has four parameters: the motivation for GARCH',
      'Interpretare: ARCH are nevoie de zece decalaje și tot pierde în fața GARCH(1,1), care are patru parametri: motivația pentru GARCH')) + ql('TSA_ch5_garch_estimation'), 'footnotesize')

D.recap(('the ARCH model', 'modelul ARCH'), [
    T('$\\sigma_t^2 = \\omega + \\sum_i\\alpha_i\\varepsilon_{t-i}^2$: an AR model for the squared shocks', '$\\sigma_t^2 = \\omega + \\sum_i\\alpha_i\\varepsilon_{t-i}^2$: un model AR pentru pătratele șocurilor'),
    T('Unconditional variance $\\omega/(1 - \\sum\\alpha_i)$; kurtosis above 3 even with Normal innovations', 'Varianța necondiționată $\\omega/(1 - \\sum\\alpha_i)$; boltire peste 3 chiar cu inovații Normale'),
    T('Real clustering needs many lags: GARCH replaces them with one extra parameter', 'Clustering-ul real cere multe decalaje: GARCH le înlocuiește cu un singur parametru în plus')])

# =============================================================================
# 4. GARCH(1,1)
# =============================================================================
D.section('The GARCH(1,1) model', 'Modelul GARCH(1,1)')

D.frame(T('1986: Tim Bollerslev and GARCH', '1986: Tim Bollerslev și modelul GARCH'), cols(items(
    (T('\\refBoll, a doctoral student of Engle at the University of California, San Diego: add yesterday\'s variance to the equation',
       '\\refBoll, doctorand al lui Engle la University of California, San Diego: adaugă în ecuație varianța de ieri'),
     [T('\\textbf{GARCH} (generalised ARCH): a few parameters replace a long ARCH($q$)', '\\textbf{GARCH} (generalised ARCH, ARCH generalizat): cîțiva parametri înlocuiesc un ARCH($q$) lung')]),
    T('Student-t innovations for returns: \\refBollT', 'Inovații Student-t pentru randamente: \\refBollT'),
    T('Still the benchmark: \\refHL\\ compared 330 volatility models and found GARCH(1,1) hard to beat on exchange rates',
      'Rămîne modelul de referință: \\refHL\\ au comparat 330 de modele de volatilitate și au găsit că GARCH(1,1) este greu de depășit la cursurile de schimb')),
    ph('ucsd', T('Geisel Library, University of California, San Diego', 'Biblioteca Geisel, University of California, San Diego'), h='0.36\\textheight'), wl='0.56', wr='0.40'))

D.frame(T('GARCH(1,1): definition', 'GARCH(1,1): definiție'), items(
    (T('$\\varepsilon_t = \\sigma_t z_t$, \\quad $\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2$, \\quad $\\omega > 0$, $\\alpha \\ge 0$, $\\beta \\ge 0$',
       '$\\varepsilon_t = \\sigma_t z_t$, \\quad $\\sigma_t^2 = \\omega + \\alpha\\,\\varepsilon_{t-1}^2 + \\beta\\,\\sigma_{t-1}^2$, \\quad $\\omega > 0$, $\\alpha \\ge 0$, $\\beta \\ge 0$'),
     [T('GARCH($p$,$q$) adds more lags of both kinds; in practice (1,1) is almost always enough', 'GARCH($p$,$q$) adaugă mai multe decalaje de ambele tipuri; în practică, (1,1) este aproape întotdeauna suficient')]),
    (T('Reading the parameters', 'Interpretarea parametrilor'),
     [T('$\\alpha$: the \\textbf{reaction} to yesterday\'s news; $\\beta$: the \\textbf{memory} of the variance', '$\\alpha$: \\textbf{reacția} la știrile de ieri; $\\beta$: \\textbf{memoria} varianței'),
      T('$\\alpha + \\beta$: the \\textbf{persistence}; $\\omega$ sets the long-run level', '$\\alpha + \\beta$: \\textbf{persistența}; $\\omega$ fixează nivelul de lungă durată')]),
    (T('By recursion: an ARCH($\\infty$) with geometric weights', 'Prin recurență: un ARCH($\\infty$) cu ponderi geometrice'),
     [T('$\\sigma_t^2 = \\dfrac{\\omega}{1 - \\beta} + \\alpha\\sum_{j=0}^{\\infty}\\beta^j\\,\\varepsilon_{t-1-j}^2$: all past shocks matter, with weights that decay like $\\beta^j$',
        '$\\sigma_t^2 = \\dfrac{\\omega}{1 - \\beta} + \\alpha\\sum_{j=0}^{\\infty}\\beta^j\\,\\varepsilon_{t-1-j}^2$: toate șocurile trecute contează, cu ponderi care scad ca $\\beta^j$')])))

D.frame(T('GARCH(1,1) as an ARMA(1,1) for the squared shocks', 'GARCH(1,1) ca ARMA(1,1) pentru pătratele șocurilor'), items(
    (T('As for ARCH, write $\\varepsilon_t^2 = \\sigma_t^2 + v_t$ and substitute $\\sigma_t^2 = \\varepsilon_t^2 - v_t$ in the equation', 'Ca la ARCH, scriem $\\varepsilon_t^2 = \\sigma_t^2 + v_t$ și înlocuim $\\sigma_t^2 = \\varepsilon_t^2 - v_t$ în ecuație'),
     [T('$\\varepsilon_t^2 = \\omega + (\\alpha + \\beta)\\,\\varepsilon_{t-1}^2 + v_t - \\beta\\,v_{t-1}$', '$\\varepsilon_t^2 = \\omega + (\\alpha + \\beta)\\,\\varepsilon_{t-1}^2 + v_t - \\beta\\,v_{t-1}$'),
      T('an ARMA(1,1) (Chapter 2) with AR coefficient $\\alpha + \\beta$ and MA coefficient $-\\beta$', 'un ARMA(1,1) (Capitolul 2) cu coeficientul AR $\\alpha + \\beta$ și coeficientul MA $-\\beta$')]),
    (T('Consequences', 'Consecințe'),
     [T('\\textbf{covariance stationary} if $\\alpha + \\beta < 1$ (the AR root outside the unit circle)', '\\textbf{staționar în covarianță} dacă $\\alpha + \\beta < 1$ (rădăcina AR în afara cercului unitate)'),
      T('the ACF of $\\varepsilon_t^2$ decays like $(\\alpha + \\beta)^k$: slow when the persistence is close to 1, as in the ACF of real squared returns',
        'ACF a lui $\\varepsilon_t^2$ scade ca $(\\alpha + \\beta)^k$: lent cînd persistența este aproape de 1, ca în ACF a pătratelor randamentelor reale'),
      T('identification by the ACF and the PACF (partial autocorrelation function) of $\\varepsilon_t^2$ is possible in principle, but in practice we use MLE (Section 5)', 'identificarea prin ACF și PACF (partial autocorrelation function, funcția de autocorelație parțială) a lui $\\varepsilon_t^2$ este posibilă în principiu, dar în practică folosim MLE (secțiunea 5)')])))

D.frame(T('Long-run variance, kurtosis, persistence and half-life', 'Varianța de lungă durată, boltirea, persistența și timpul de înjumătățire'), items(
    (T('\\textbf{Long-run (unconditional) variance}: $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$; annualised volatility $\\sqrt{252\\,\\bar\\sigma^2}$ for 252 trading days',
       '\\textbf{Varianța de lungă durată (necondiționată)}: $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$; volatilitatea anualizată $\\sqrt{252\\,\\bar\\sigma^2}$ pentru 252 de zile de tranzacționare'),
     [T('kurtosis with Normal $z_t$: $K = 3\\,\\dfrac{1 - (\\alpha + \\beta)^2}{1 - (\\alpha + \\beta)^2 - 2\\alpha^2} > 3$; $\\alpha = 0.10$, $\\beta = 0.88$: $K = @{ex.g.k}$',
        'boltirea cu $z_t$ Normale: $K = 3\\,\\dfrac{1 - (\\alpha + \\beta)^2}{1 - (\\alpha + \\beta)^2 - 2\\alpha^2} > 3$; $\\alpha = 0{,}10$, $\\beta = 0{,}88$: $K = @{ex.g.k}$')]),
    (T('A deviation from $\\bar\\sigma^2$ shrinks by the factor $\\alpha + \\beta$ each day (Section 9): $E_t[\\sigma_{t+h}^2] - \\bar\\sigma^2 = (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$',
       'O abatere de la $\\bar\\sigma^2$ se micșorează cu factorul $\\alpha + \\beta$ în fiecare zi (secțiunea 9): $E_t[\\sigma_{t+h}^2] - \\bar\\sigma^2 = (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$'),
     [T('\\textbf{half-life}: the number of days after which half of a variance shock is gone, $h_{1/2} = \\ln 0.5/\\ln(\\alpha + \\beta)$',
        '\\textbf{timpul de înjumătățire}: numărul de zile după care a dispărut jumătate dintr-un șoc de varianță, $h_{1/2} = \\ln 0{,}5/\\ln(\\alpha + \\beta)$'),
      T('$\\alpha + \\beta = 0.90$: @{hl.90} days; $0.98$: @{hl.98} days; $0.99$: @{hl.99} days: small changes near 1 matter a lot',
        '$\\alpha + \\beta = 0{,}90$: @{hl.90} zile; $0{,}98$: @{hl.98} zile; $0{,}99$: @{hl.99} zile: schimbările mici în apropiere de 1 contează mult')])))

D.frame(T('Worked example: one GARCH(1,1) step', 'Exemplu rezolvat: un pas GARCH(1,1)'), items(
    T('Daily returns in \\%: $\\omega = 0.02$, $\\alpha = 0.10$, $\\beta = 0.88$, $\\mu = 0$', 'Randamente zilnice în \\%: $\\omega = 0{,}02$, $\\alpha = 0{,}10$, $\\beta = 0{,}88$, $\\mu = 0$'),
    (T('Long-run quantities', 'Mărimile de lungă durată'),
     [T('persistence $\\alpha + \\beta = @{ex.g.pers}$; $\\bar\\sigma^2 = 0.02/(1 - @{ex.g.pers}) = @{ex.g.uv}$; annualised volatility $\\sqrt{252 \\times @{ex.g.uv}} = @{ex.g.vol}\\%$',
        'persistența $\\alpha + \\beta = @{ex.g.pers}$; $\\bar\\sigma^2 = 0{,}02/(1 - @{ex.g.pers}) = @{ex.g.uv}$; volatilitatea anualizată $\\sqrt{252 \\times @{ex.g.uv}} = @{ex.g.vol}\\%$'),
      T('half-life $\\ln 0.5/\\ln @{ex.g.pers} = @{ex.g.hl}$ days', 'timpul de înjumătățire $\\ln 0{,}5/\\ln @{ex.g.pers} = @{ex.g.hl}$ zile')]),
    (T('Today: $\\sigma_t^2 = 1.2$ and the return is $r_t = -3\\%$', 'Azi: $\\sigma_t^2 = 1{,}2$, iar randamentul este $r_t = -3\\%$'),
     [T('tomorrow: $\\sigma_{t+1}^2 = 0.02 + 0.10 \\times 9 + 0.88 \\times 1.2 = @{ex.g.s2}$, $\\sigma_{t+1} = @{ex.g.s}\\%$', 'mîine: $\\sigma_{t+1}^2 = 0{,}02 + 0{,}10 \\times 9 + 0{,}88 \\times 1{,}2 = @{ex.g.s2}$, $\\sigma_{t+1} = @{ex.g.s}\\%$'),
      T('the shock raised the variance from 1.2 to almost 2; it then decays towards @{ex.g.uv} with the factor @{ex.g.pers} per day',
        'șocul a crescut varianța de la 1,2 la aproape 2; apoi ea scade spre @{ex.g.uv} cu factorul @{ex.g.pers} pe zi')])))

D.frame(T('IGARCH and the link with EWMA', 'IGARCH și legătura cu EWMA'), items(
    (T('\\textbf{IGARCH} (integrated GARCH) \\refEB: $\\alpha + \\beta = 1$: a unit root in the ARMA(1,1) for $\\varepsilon_t^2$ (Chapter 3)',
       '\\textbf{IGARCH} (integrated GARCH, GARCH integrat) \\refEB: $\\alpha + \\beta = 1$: o rădăcină unitară în ARMA(1,1) pentru $\\varepsilon_t^2$ (Capitolul 3)'),
     [T('shocks to the variance never die out: no half-life, no finite unconditional variance', 'șocurile varianței nu se sting niciodată: nu există timp de înjumătățire și nici varianță necondiționată finită')]),
    (T('\\textbf{EWMA} (exponentially weighted moving average) of \\refRM: $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1 - \\lambda)r_{t-1}^2$',
       '\\textbf{EWMA} (exponentially weighted moving average, media mobilă ponderată exponențial) a \\refRM: $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1 - \\lambda)r_{t-1}^2$'),
     [T('simple exponential smoothing (Chapter 0) applied to $r_t^2$, with smoothing constant $1 - \\lambda$', 'netezirea exponențială simplă (Capitolul 0) aplicată lui $r_t^2$, cu constanta de netezire $1 - \\lambda$'),
      T('an IGARCH with $\\omega = 0$, $\\alpha = 1 - \\lambda$, $\\beta = \\lambda$; RiskMetrics uses $\\lambda = 0.94$ for daily data (half-life of the weights @{ew.hl} days)',
        'un IGARCH cu $\\omega = 0$, $\\alpha = 1 - \\lambda$, $\\beta = \\lambda$; RiskMetrics folosește $\\lambda = 0{,}94$ pentru date zilnice (timpul de înjumătățire al ponderilor: @{ew.hl} zile)')]),
    (T('\\textbf{Question for the room}: right after a crash, an analyst forecasts the volatility one year ahead with EWMA. What goes wrong?',
       '\\textbf{Întrebare pentru sală}: imediat după un crah, un analist prognozează volatilitatea peste un an cu EWMA. Unde greșește?'),
     [T('\\textbf{Answer}: EWMA keeps today\'s crisis level for the whole year; GARCH lets it decay towards the long-run level',
        '\\textbf{Răspuns}: EWMA păstrează nivelul de criză de azi pentru tot anul; GARCH îl lasă să scadă spre nivelul de lungă durată')])))

chart(T('GARCH, EWMA and a rolling window in 2020', 'GARCH, EWMA și o fereastră mobilă în 2020'), 'tsa_ch5_ewma_garch', 'TSA_ch5_volatility_markets', [
    T('S\\&P 500, annualised volatility: GARCH(1,1)-t estimated on all data, EWMA with $\\lambda = 0.94$, and the standard deviation of the last 63 days (Chapter 0: a moving average)',
      'S\\&P 500, volatilitatea anualizată: GARCH(1,1)-t estimat pe toate datele, EWMA cu $\\lambda = 0{,}94$ și abaterea standard a ultimelor 63 de zile (Capitolul 0: o medie mobilă)')], h='0.52\\textheight')

interp(('the three estimates', 'celor trei estimări'), [
    T('Peaks: GARCH @{ew.g.peak}\\% (@{ew.g.dpeak}), EWMA @{ew.e.peak}\\% (@{ew.e.dpeak}), 63-day window @{ew.w.peak}\\% (@{ew.w.dpeak})',
      'Vîrfuri: GARCH @{ew.g.peak}\\% (@{ew.g.dpeak}), EWMA @{ew.e.peak}\\% (@{ew.e.dpeak}), fereastra de 63 de zile @{ew.w.peak}\\% (@{ew.w.dpeak})'),
    (T('The rolling window reacts late and falls abruptly when the crash leaves the window (the ``ghost\'\' of the crash)', 'Fereastra mobilă reacționează tîrziu și scade brusc cînd crahul iese din fereastră („fantoma” crahului)'),
     [T('on 30 June 2020: GARCH @{ew.g.jun}\\%, EWMA @{ew.e.jun}\\%, window @{ew.w.jun}\\%', 'la 30 iunie 2020: GARCH @{ew.g.jun}\\%, EWMA @{ew.e.jun}\\%, fereastra @{ew.w.jun}\\%')]),
    T('GARCH reacts fastest ($\\alpha$ larger than $1 - \\lambda$) and decays fastest (mean reversion); EWMA lies in between', 'GARCH reacționează cel mai repede ($\\alpha$ mai mare decît $1 - \\lambda$) și scade cel mai repede (revenire la medie); EWMA este între ele')])

D.recap(('the GARCH(1,1) model', 'modelul GARCH(1,1)'), [
    T('$\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$: reaction $\\alpha$, memory $\\beta$; an ARMA(1,1) for $\\varepsilon_t^2$', '$\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$: reacția $\\alpha$, memoria $\\beta$; un ARMA(1,1) pentru $\\varepsilon_t^2$'),
    T('$\\alpha + \\beta < 1$: long-run variance $\\omega/(1 - \\alpha - \\beta)$, half-life $\\ln 0.5/\\ln(\\alpha + \\beta)$', '$\\alpha + \\beta < 1$: varianța de lungă durată $\\omega/(1 - \\alpha - \\beta)$, timpul de înjumătățire $\\ln 0{,}5/\\ln(\\alpha + \\beta)$'),
    T('$\\alpha + \\beta = 1$: IGARCH; EWMA is an IGARCH without $\\omega$', '$\\alpha + \\beta = 1$: IGARCH; EWMA este un IGARCH fără $\\omega$')])

# =============================================================================
# 5. ESTIMARE
# =============================================================================
D.section('Estimation by maximum likelihood', 'Estimarea prin verosimilitate maximă')

D.frame(T('The likelihood of a GARCH model', 'Verosimilitatea unui model GARCH'), items(
    (T('The returns are dependent, so the joint density is a product of conditional densities', 'Randamentele sînt dependente, deci densitatea comună este un produs de densități condiționate'),
     [T('$f(r_1, \\dots, r_T) = f(r_1)\\prod_{t=2}^{T} f(r_t \\mid \\mathcal{F}_{t-1})$, with $r_t \\mid \\mathcal{F}_{t-1} \\sim N(\\mu, \\sigma_t^2)$ for Normal innovations',
        '$f(r_1, \\dots, r_T) = f(r_1)\\prod_{t=2}^{T} f(r_t \\mid \\mathcal{F}_{t-1})$, cu $r_t \\mid \\mathcal{F}_{t-1} \\sim N(\\mu, \\sigma_t^2)$ pentru inovații Normale')]),
    (T('\\textbf{Conditional log-likelihood}, $\\theta = (\\mu, \\omega, \\alpha, \\beta)$:', '\\textbf{Log-verosimilitatea condiționată}, $\\theta = (\\mu, \\omega, \\alpha, \\beta)$:'),
     [T('$\\ell(\\theta) = -\\dfrac12\\sum_{t=2}^{T}\\Big[\\ln(2\\pi) + \\ln\\sigma_t^2(\\theta) + \\dfrac{(r_t - \\mu)^2}{\\sigma_t^2(\\theta)}\\Big]$',
        '$\\ell(\\theta) = -\\dfrac12\\sum_{t=2}^{T}\\Big[\\ln(2\\pi) + \\ln\\sigma_t^2(\\theta) + \\dfrac{(r_t - \\mu)^2}{\\sigma_t^2(\\theta)}\\Big]$'),
      T('$\\sigma_t^2(\\theta)$ is computed recursively from a starting value $\\sigma_1^2$ (for example the sample variance)', '$\\sigma_t^2(\\theta)$ se calculează recursiv, pornind de la o valoare inițială $\\sigma_1^2$ (de exemplu, varianța de selecție)')]),
    (T('\\textbf{MLE} (maximum likelihood estimation): $\\hat\\theta = \\arg\\max_\\theta \\ell(\\theta)$, under $\\omega > 0$, $\\alpha, \\beta \\ge 0$, $\\alpha + \\beta < 1$',
       '\\textbf{MLE} (maximum likelihood estimation, estimarea prin verosimilitate maximă): $\\hat\\theta = \\arg\\max_\\theta \\ell(\\theta)$, cu $\\omega > 0$, $\\alpha, \\beta \\ge 0$, $\\alpha + \\beta < 1$'),
     [T('no closed form: a numerical optimiser searches for the maximum; OLS is not possible, since $\\sigma_t^2$ is not observed', 'nu există o formulă explicită: un algoritm numeric caută maximul; OLS nu este posibil, deoarece $\\sigma_t^2$ nu este observat')])))

chart(T('The likelihood of ARCH(1)', 'Verosimilitatea modelului ARCH(1)'), 'tsa_ch5_lik_arch1', 'TSA_ch5_simulation_likelihood', [
    T('Simulated ARCH(1) with $\\omega = \\alpha = 0.5$; $\\ell(\\alpha)$ with $\\omega = 1 - \\alpha$; each curve minus its maximum', 'ARCH(1) simulat cu $\\omega = \\alpha = 0{,}5$; $\\ell(\\alpha)$ cu $\\omega = 1 - \\alpha$; fiecare curbă minus maximul ei'),
    T('$n = 100$: $\\hat\\alpha = @{la.100}$ (standard error $@{la.100.se}$); $n = 1000$: $\\hat\\alpha = @{la.1000}$ ($@{la.1000.se}$)', '$n = 100$: $\\hat\\alpha = @{la.100}$ (eroarea standard $@{la.100.se}$); $n = 1000$: $\\hat\\alpha = @{la.1000}$ ($@{la.1000.se}$)'),
    T('Interpretation: more data make the curve sharper; its curvature at the maximum gives the standard error', 'Interpretare: mai multe date fac curba mai ascuțită; curbura ei în punctul de maxim dă eroarea standard')],
    h='0.48\\textheight')

chart(T('The likelihood of GARCH(1,1)', 'Verosimilitatea modelului GARCH(1,1)'), 'tsa_ch5_lik_garch', 'TSA_ch5_simulation_likelihood', [
    T('Simulated GARCH(1,1) with $\\omega = 0.1$, $\\alpha = 0.1$, $\\beta = 0.8$; $\\omega$ set so that the unconditional variance equals the sample variance; contours: log-likelihood minus its maximum',
      'GARCH(1,1) simulat cu $\\omega = 0{,}1$, $\\alpha = 0{,}1$, $\\beta = 0{,}8$; $\\omega$ fixat astfel încît varianța necondiționată să fie egală cu varianța de selecție; contururi: log-verosimilitatea minus maximul ei'),
    T('$n = 500$: maximum at $(@{lg.500.a}, @{lg.500.b})$, far from the truth; $n = 2000$: $(@{lg.2000.a}, @{lg.2000.b})$', '$n = 500$: maximul în $(@{lg.500.a}; @{lg.500.b})$, departe de valorile adevărate; $n = 2000$: $(@{lg.2000.a}; @{lg.2000.b})$'),
    T('Interpretation: a long, flat ridge along which $\\alpha$ and $\\beta$ trade off: GARCH needs long samples', 'Interpretare: o creastă lungă și plată, de-a lungul căreia $\\alpha$ și $\\beta$ se compensează: GARCH are nevoie de eșantioane lungi')],
    h='0.46\\textheight')


def er(k, lab):
    return f'{lab} & $@{{step.{k}}}$ ($@{{step.{k}.se}}$) & $@{{arch.{k}}}$ & $@{{arch.{k}.sec}}$ & $@{{arch.{k}.ser}}$'


D.frame(T('Estimation step by step and with the \\texttt{arch} package', 'Estimarea pas cu pas și cu pachetul \\texttt{arch}'), table(
    'lrrrr', T('S\\&P 500, GARCH(1,1)-N & step by step (SE) & \\texttt{arch} & classic SE & robust SE',
               'S\\&P 500, GARCH(1,1)-N & pas cu pas (SE) & \\texttt{arch} & SE clasică & SE robustă'),
    [er('mu', '$\\mu$'), er('om', '$\\omega$'), er('a', '$\\alpha$'), er('b', '$\\beta$')], size='footnotesize') + items(
    (T('Step by step: $-\\ell(\\theta)$ written in Python and minimised with \\texttt{scipy.optimize.minimize} under $\\alpha + \\beta < 1$',
       'Pas cu pas: $-\\ell(\\theta)$ scrisă în Python și minimizată cu \\texttt{scipy.optimize.minimize}, cu restricția $\\alpha + \\beta < 1$'),
     [T('with the package: \\texttt{arch\\_model(r, mean=\'Constant\', vol=\'GARCH\', p=1, q=1, dist=\'normal\').fit()}', 'cu pachetul: \\texttt{arch\\_model(r, mean=\'Constant\', vol=\'GARCH\', p=1, q=1, dist=\'normal\').fit()}')]),
    T('SE (standard error), classic: from the inverse Hessian (the curvature of $\\ell$); robust: valid when $z_t$ is not Normal \\refBW',
      'SE (standard error, eroarea standard) clasică: din inversa matricei hessiene (curbura lui $\\ell$); robustă: validă și cînd $z_t$ nu este Normal \\refBW'),
    T('Interpretation: same estimates by both routes; the robust SE are about @{se.ratio} times larger: with heavy-tailed $z_t$ the classic SE overstate the precision',
      'Interpretare: aceleași estimări pe ambele căi; SE robuste sînt de circa @{se.ratio} ori mai mari: cînd $z_t$ are cozi groase, SE clasice supraestimează precizia')) + ql('TSA_ch5_garch_estimation'), 'footnotesize')

D.frame(T('Quasi-maximum likelihood and practical rules', 'Cvasi-verosimilitatea maximă și reguli practice'), items(
    (T('\\textbf{QMLE} (quasi-maximum likelihood estimation): maximise the Normal likelihood even if $z_t$ is not Normal',
       '\\textbf{QMLE} (quasi-maximum likelihood estimation, estimarea prin cvasi-verosimilitate maximă): maximizăm verosimilitatea Normală chiar dacă $z_t$ nu este Normal'),
     [T('the estimates stay consistent if $\\mu_t$ and $\\sigma_t^2$ are correctly specified, but only the robust (``sandwich\'\') SE are valid \\refBW',
        'estimările rămîn consistente dacă $\\mu_t$ și $\\sigma_t^2$ sînt corect specificate, dar doar SE robuste (de tip „sandwich”) sînt valide \\refBW')]),
    (T('Practical rules', 'Reguli practice'),
     [T('returns in \\% (not decimals): the optimiser fails when $\\omega$ is tiny; \\texttt{arch} warns about badly scaled data', 'randamente în \\% (nu în zecimale): algoritmul de optimizare eșuează cînd $\\omega$ este foarte mic; \\texttt{arch} avertizează în cazul datelor prost scalate'),
      T('EUR/RON moves about 0.3\\% a day: we estimate on $10\\,r_t$ and convert $\\omega$ back (divide by 100)', 'EUR/RON variază cu circa 0,3\\% pe zi: estimăm pe $10\\,r_t$ și convertim înapoi $\\omega$ (împărțim la 100)'),
      T('use at least 1000--2000 daily observations; check the convergence flag of the optimiser', 'folosim cel puțin 1000--2000 de observații zilnice; verificăm indicatorul de convergență al algoritmului'),
      T('an estimate on the boundary ($\\hat\\alpha + \\hat\\beta = 1$ or $\\hat\\alpha = 0$) has no usual standard error: report it as such', 'o estimare pe frontieră ($\\hat\\alpha + \\hat\\beta = 1$ sau $\\hat\\alpha = 0$) nu are o eroare standard obișnuită: o raportăm ca atare')])))

D.frame(T('Student-t and skewed-t innovations', 'Inovații Student-t și t asimetrice'), items(
    (T('GARCH with Normal $z_t$ explains only part of the kurtosis: S\\&P 500 returns $@{kurt.r}$, standardised residuals $\\hat z_t = \\hat\\varepsilon_t/\\hat\\sigma_t$ still $@{kurt.z}$',
       'GARCH cu $z_t$ Normale explică doar o parte din boltire: randamentele S\\&P 500 au $@{kurt.r}$, reziduurile standardizate $\\hat z_t = \\hat\\varepsilon_t/\\hat\\sigma_t$ încă $@{kurt.z}$'),
     [T('the remaining heavy tails must come from the distribution of $z_t$', 'cozile groase rămase trebuie să vină din distribuția lui $z_t$')]),
    (T('\\textbf{Standardised Student-t} with $\\nu > 2$ degrees of freedom \\refBollT', '\\textbf{Student-t standardizată} cu $\\nu > 2$ grade de libertate \\refBollT'),
     [T('$z_t = t_\\nu\\sqrt{(\\nu - 2)/\\nu}$ has variance 1; kurtosis $3 + 6/(\\nu - 4)$ for $\\nu > 4$; $\\nu \\to \\infty$: the Normal distribution',
        '$z_t = t_\\nu\\sqrt{(\\nu - 2)/\\nu}$ are varianța 1; coeficientul de boltire $3 + 6/(\\nu - 4)$ pentru $\\nu > 4$; $\\nu \\to \\infty$: distribuția Normală'),
      T('$\\nu$ is estimated together with $(\\mu, \\omega, \\alpha, \\beta)$: \\texttt{dist=\'t\'} in \\texttt{arch}', '$\\nu$ se estimează împreună cu $(\\mu, \\omega, \\alpha, \\beta)$: \\texttt{dist=\'t\'} în \\texttt{arch}')]),
    (T('\\textbf{Skewed t} \\refHansen: one more parameter $\\lambda \\in (-1, 1)$; $\\lambda < 0$: a longer left tail (\\texttt{dist=\'skewt\'})', '\\textbf{t asimetrică} \\refHansen: încă un parametru $\\lambda \\in (-1, 1)$; $\\lambda < 0$: o coadă stîngă mai lungă (\\texttt{dist=\'skewt\'})'),
     [T('nested models: Normal $\\subset$ t $\\subset$ skewed t; compare them with LR (likelihood ratio) tests or AIC/BIC',
        'modele incluse unul în altul: Normal $\\subset$ t $\\subset$ t asimetrică; le comparăm prin teste LR (likelihood ratio, raportul de verosimilitate) sau prin AIC/BIC')])))

D.frame(T('S\\&P 500: three innovation distributions', 'S\\&P 500: trei distribuții ale inovațiilor'), table(
    'lrrrrrrr', T('GARCH(1,1) & $\\omega$ & $\\alpha$ & $\\beta$ & $\\nu$ & $\\lambda$ & log-lik. & $\\alpha + \\beta$',
                  'GARCH(1,1) & $\\omega$ & $\\alpha$ & $\\beta$ & $\\nu$ & $\\lambda$ & log-verosim. & $\\alpha + \\beta$'),
    [T('Normal', 'Normală') + ' & $@{en.om}$ & $@{en.a}$ & $@{en.b}$ & -- & -- & $@{en.ll}$ & $@{en.pers}$',
     'Student-t & $@{et.om}$ & $@{et.a}$ & $@{et.b}$ & $@{et.nu}$ & -- & $@{et.ll}$ & $@{et.pers}$',
     T('skewed t', 't asimetrică') + ' & $@{es.om}$ & $@{es.a}$ & $@{es.b}$ & $@{es.eta}$ & $@{es.lam}$ & $@{es.ll}$ & $@{es.pers}$'],
    size='footnotesize') + items(
    T('LR statistic $2(\\ell_1 - \\ell_0)$: t against Normal $@{lr.tn}$ (1 restriction, $\\chi^2_{0.95}(1) = 3.84$); skewed t against t $@{lr.st}$: both reject the smaller model',
      'Statistica LR $2(\\ell_1 - \\ell_0)$: t față de Normal $@{lr.tn}$ (o restricție, $\\chi^2_{0.95}(1) = 3{,}84$); t asimetrică față de t $@{lr.st}$: ambele resping modelul mai mic'),
    T('$\\hat\\nu = @{et.nu}$: tails far heavier than Normal; $\\hat\\lambda = @{es.lam}$: large falls more frequent than large rises',
      '$\\hat\\nu = @{et.nu}$: cozi mult mai groase decît la distribuția Normală; $\\hat\\lambda = @{es.lam}$: scăderile mari sînt mai frecvente decît creșterile mari'),
    T('Interpretation: $\\alpha$ and $\\beta$ hardly change; the innovation distribution matters for tail quantiles (VaR), less for $\\sigma_t$',
      'Interpretare: $\\alpha$ și $\\beta$ aproape nu se schimbă; distribuția inovațiilor contează pentru cuantilele din coadă (VaR), mai puțin pentru $\\sigma_t$')) + ql('TSA_ch5_garch_estimation'), 'footnotesize')

chart(T('QQ plots of the standardised residuals', 'Graficele QQ ale reziduurilor standardizate'), 'tsa_ch5_qq', 'TSA_ch5_garch_estimation', [
    T('QQ (quantile--quantile) plot: sample quantiles of $\\hat z_t$ against the quantiles of the assumed distribution; a straight line means a good fit',
      'Graficul QQ (quantile--quantile): cuantilele de selecție ale lui $\\hat z_t$ față de cuantilele distribuției presupuse; o dreaptă înseamnă o potrivire bună'),
    T('Left, GARCH-N against the Normal: 0.1\\% quantile $@{qq.n001}$, against $@{qq.n001th}$ in theory; right, GARCH-t against the standardised t ($\\hat\\nu = @{qq.nu}$): $@{qq.t001}$ against $@{qq.t001th}$',
      'Stînga, GARCH-N față de distribuția Normală: cuantila de 0,1\\% este $@{qq.n001}$, față de $@{qq.n001th}$ teoretic; dreapta, GARCH-t față de t standardizată ($\\hat\\nu = @{qq.nu}$): $@{qq.t001}$ față de $@{qq.t001th}$'),
    T('Interpretation: the t fits much better, but the largest falls are still deeper: the left tail is the problem (asymmetry, Section 7)',
      'Interpretare: distribuția t se potrivește mult mai bine, dar cele mai mari scăderi sînt tot mai adînci: problema este coada stîngă (asimetria, secțiunea 7)')],
    h='0.46\\textheight')

D.recap(('estimation by maximum likelihood', 'estimarea prin verosimilitate maximă'), [
    T('$\\ell(\\theta)$ = sum of conditional log-densities; $\\sigma_t^2(\\theta)$ computed recursively; numerical maximisation', '$\\ell(\\theta)$ = suma log-densităților condiționate; $\\sigma_t^2(\\theta)$ calculat recursiv; maximizare numerică'),
    T('The likelihood has a ridge in $(\\alpha, \\beta)$: long samples; report robust SE', 'Verosimilitatea are o creastă în $(\\alpha, \\beta)$: eșantioane lungi; raportăm SE robuste'),
    T('Student-t or skewed-t innovations for the remaining heavy tails', 'Inovații Student-t sau t asimetrice pentru cozile groase rămase')])

# =============================================================================
# 6. PATRU PIEȚE ȘI MODELE ARMA-GARCH
# =============================================================================
D.section('GARCH in four markets and ARMA-GARCH models', 'GARCH pe patru piețe și modele ARMA-GARCH')


def mrow(k):
    return (f'{NAMES[k]} & @{{m.{k}.y0}} & $@{{m.{k}.om}}$ & $@{{m.{k}.a}}$ & $@{{m.{k}.b}}$ & $@{{m.{k}.nu}}$ & $@{{m.{k}.pers}}$ & @{{m.{k}.hl}} & '
            f'@{{m.{k}.vlr}} & @{{m.{k}.vs}}')


D.frame(T('GARCH(1,1)-t estimates', 'Estimările GARCH(1,1)-t'), table(
    'lrrrrrrrrr', T('& from & $\\omega$ & $\\alpha$ & $\\beta$ & $\\nu$ & $\\alpha + \\beta$ & half-life & long-run vol. & sample vol.',
                    '& din & $\\omega$ & $\\alpha$ & $\\beta$ & $\\nu$ & $\\alpha + \\beta$ & înjumătățire & vol. lungă durată & vol. selecție'),
    [mrow(k) for k in ASSETS], size='scriptsize') + items(
    T('Daily log returns in \\% to @{end}; half-life in days; volatilities annualised with the actual number of observations per year (Bitcoin: 365), in \\%',
      'Randamente logaritmice zilnice în \\% pînă la @{end}; timpul de înjumătățire în zile; volatilitățile anualizate cu numărul efectiv de observații pe an (Bitcoin: 365), în \\%'),
    T('S\\&P 500 and BET: $\\alpha + \\beta$ @{m.sp500.pers} and @{m.bet.pers}, half-lives of @{m.sp500.hl} and @{m.bet.hl} trading days; BET reacts more to news ($\\alpha = @{m.bet.a}$)',
      'S\\&P 500 și BET: $\\alpha + \\beta$ egal cu @{m.sp500.pers} și @{m.bet.pers}, timpi de înjumătățire de @{m.sp500.hl} și @{m.bet.hl} zile de tranzacționare; BET reacționează mai puternic la știri ($\\alpha = @{m.bet.a}$)'),
    T('EUR/RON and Bitcoin: $\\hat\\alpha + \\hat\\beta = 1$ on the boundary: IGARCH; every $\\hat\\nu$ between @{m.nu.min} and @{m.nu.max}: heavy tails everywhere',
      'EUR/RON și Bitcoin: $\\hat\\alpha + \\hat\\beta = 1$, pe frontieră: IGARCH; toate valorile $\\hat\\nu$ între @{m.nu.min} și @{m.nu.max}: cozi groase peste tot')) + ql('TSA_ch5_volatility_markets'), 'footnotesize')

D.frame(T('Interpreting IGARCH and the long-run volatility', 'Interpretarea IGARCH și a volatilității de lungă durată'), items(
    (T('EUR/RON: $\\hat\\omega \\approx 0$ and $\\hat\\beta = @{mg.lam}$: an EWMA with $\\lambda \\approx @{mg.lam}$; Bitcoin: $\\hat\\beta = @{mg.lamb}$', 'EUR/RON: $\\hat\\omega \\approx 0$ și $\\hat\\beta = @{mg.lam}$: un EWMA cu $\\lambda \\approx @{mg.lam}$; Bitcoin: $\\hat\\beta = @{mg.lamb}$'),
     [T('no half-life and no long-run volatility: the volatility level drifts like a random walk (Chapter 3)', 'nu există timp de înjumătățire și nici volatilitate de lungă durată: nivelul volatilității se deplasează ca un mers aleator (Capitolul 3)'),
      T('EUR/RON: a managed rate whose volatility fell from one regime to the next (the BNR policy changes over time)', 'EUR/RON: un curs administrat, a cărui volatilitate a scăzut de la un regim la altul (politica BNR se schimbă în timp)')]),
    (T('\\textbf{Question for the room}: S\\&P 500: long-run volatility @{m.sp500.vlr}\\%, sample volatility @{m.sp500.vs}\\%. Is the model wrong?',
       '\\textbf{Întrebare pentru sală}: S\\&P 500: volatilitatea de lungă durată @{m.sp500.vlr}\\%, volatilitatea de selecție @{m.sp500.vs}\\%. Este greșit modelul?'),
     [T('\\textbf{Answer}: not necessarily: $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$ divides by $1 - \\alpha - \\beta = @{m.sp500.1mp}$, a small and imprecise number',
        '\\textbf{Răspuns}: nu neapărat: $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$ se împarte la $1 - \\alpha - \\beta = @{m.sp500.1mp}$, un număr mic și imprecis'),
      T('the Normal GARCH gives @{en.vlr}\\%: the long-run level is the least reliable output of a persistent GARCH', 'GARCH cu inovații Normale dă @{en.vlr}\\%: nivelul de lungă durată este cel mai puțin sigur rezultat al unui GARCH persistent')])))

chart(T('Conditional volatility: S\\&P 500 and BET', 'Volatilitatea condiționată: S\\&P 500 și BET'), 'tsa_ch5_vol_sp500_bet', 'TSA_ch5_volatility_markets', [
    T('GARCH(1,1)-t volatility $\\hat\\sigma_t$, annualised; dashed: the sample volatility', 'Volatilitatea GARCH(1,1)-t $\\hat\\sigma_t$, anualizată; linia întreruptă: volatilitatea de selecție'),
    T('S\\&P 500: peaks @{v.sp500.p2008}\\% (@{v.sp500.d2008}) and @{v.sp500.p2020}\\% (@{v.sp500.d2020}); BET: @{v.bet.p2008}\\% (@{v.bet.d2008}) and @{v.bet.p2020}\\% (@{v.bet.d2020}); medians @{v.sp500.med}\\% and @{v.bet.med}\\%',
      'S\\&P 500: vîrfuri de @{v.sp500.p2008}\\% (@{v.sp500.d2008}) și @{v.sp500.p2020}\\% (@{v.sp500.d2020}); BET: @{v.bet.p2008}\\% (@{v.bet.d2008}) și @{v.bet.p2020}\\% (@{v.bet.d2020}); medianele @{v.sp500.med}\\% și @{v.bet.med}\\%')],
    h='0.54\\textheight')

chart(T('Conditional volatility: EUR/RON and Bitcoin', 'Volatilitatea condiționată: EUR/RON și Bitcoin'), 'tsa_ch5_vol_eurron_btc', 'TSA_ch5_volatility_markets', [
    T('EUR/RON: @{v.eurron.p2008}\\% at the peak of the 2008 crisis (@{v.eurron.d2008}), only @{v.eurron.p2020}\\% in 2020; on @{end}: @{v.eurron.last}\\%',
      'EUR/RON: @{v.eurron.p2008}\\% în vîrful crizei din 2008 (@{v.eurron.d2008}), doar @{v.eurron.p2020}\\% în 2020; la @{end}: @{v.eurron.last}\\%'),
    T('Bitcoin: @{v.btc.p2020}\\% on @{v.btc.d2020}; median @{v.btc.med}\\%, @{v.ratio} times the median of the S\\&P 500', 'Bitcoin: @{v.btc.p2020}\\% pe @{v.btc.d2020}; mediana @{v.btc.med}\\%, de @{v.ratio} ori mai mare decît mediana S\\&P 500'),
    T('Interpretation: the two IGARCH series have volatility levels that wander: no single ``normal\'\' level', 'Interpretare: cele două serii IGARCH au niveluri de volatilitate care se deplasează: nu există un singur nivel „normal”')],
    h='0.52\\textheight')

chart(T('How long does a volatility shock last?', 'Durata unui șoc de volatilitate'), 'tsa_ch5_persistence', 'TSA_ch5_volatility_markets', [
    T('Share of a variance shock left after $h$ days: $(\\alpha + \\beta)^h$; the half-life is where the curve crosses 1/2', 'Ponderea unui șoc de varianță rămasă după $h$ zile: $(\\alpha + \\beta)^h$; timpul de înjumătățire este punctul în care curba trece de 1/2'),
    T('Interpretation: after a crisis the S\\&P 500 needs about half a year to halve the excess variance, the BET about three months; EUR/RON and Bitcoin never',
      'Interpretare: după o criză, S\\&P 500 are nevoie de circa jumătate de an ca să înjumătățească varianța în exces, BET de circa trei luni; EUR/RON și Bitcoin, niciodată')],
    h='0.50\\textheight')

D.frame(T('ARMA-GARCH: one model for the mean and the variance', 'ARMA-GARCH: un model pentru medie și varianță'), items(
    (T('\\textbf{ARMA($p$,$q$)-GARCH(1,1)}: $r_t = c + \\sum_{i=1}^{p}\\phi_i r_{t-i} + \\varepsilon_t + \\sum_{j=1}^{q}\\theta_j\\varepsilon_{t-j}$, \\quad $\\varepsilon_t = \\sigma_t z_t$, \\quad $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$',
       '\\textbf{ARMA($p$,$q$)-GARCH(1,1)}: $r_t = c + \\sum_{i=1}^{p}\\phi_i r_{t-i} + \\varepsilon_t + \\sum_{j=1}^{q}\\theta_j\\varepsilon_{t-j}$, \\quad $\\varepsilon_t = \\sigma_t z_t$, \\quad $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$'),
     [T('the ARMA part is the model of Chapter 2; its errors are no longer i.i.d., they follow a GARCH', 'partea ARMA este modelul din Capitolul 2; erorile lui nu mai sînt i.i.d., ci urmează un GARCH')]),
    (T('Estimation: both parts at once, by maximum likelihood', 'Estimarea: ambele părți simultan, prin verosimilitate maximă'),
     [T('in \\texttt{arch}: \\texttt{arch\\_model(r, mean=\'AR\', lags=1, vol=\'GARCH\', dist=\'t\')} (AR means only; for MA terms use two steps or another package)',
        'în \\texttt{arch}: \\texttt{arch\\_model(r, mean=\'AR\', lags=1, vol=\'GARCH\', dist=\'t\')} (doar medii AR; pentru termeni MA, două etape sau alt pachet)')]),
    (T('Why not OLS for the mean and then GARCH?', 'De ce nu OLS pentru medie și apoi GARCH?'),
     [T('the OLS coefficients are still consistent, but their usual SE assume a constant variance: the $t$ tests of Chapter 2 are wrong', 'coeficienții OLS rămîn consistenți, dar SE obișnuite presupun o varianță constantă: testele $t$ din Capitolul 2 sînt greșite'),
      T('the joint maximum likelihood estimate gives less weight to stormy days, hence more precise $\\hat\\phi$ and valid intervals', 'estimarea comună prin verosimilitate maximă dă o pondere mai mică zilelor agitate, deci un $\\hat\\phi$ mai precis și intervale valide')])))


def agrow(k):
    return (f'{NAMES[k]} & @{{ag.{k}.p}} & $@{{ag.{k}.phio}}$ ($@{{ag.{k}.seo}}$) & $@{{ag.{k}.phig}}$ ($@{{ag.{k}.seg}}$) & '
            f'$@{{ag.{k}.qc}}$ (@{{ag.{k}.qcp}}) & $@{{ag.{k}.qa}}$ (@{{ag.{k}.qap}}) & ${{@{{ag.{k}.dbic}}}}$')


D.frame(T('AR(1)-GARCH(1,1)-t on four series', 'AR(1)-GARCH(1,1)-t pe patru serii'), table(
    'lrrrrrr', T('& $p$ (BIC) & $\\hat\\phi$, OLS (SE) & $\\hat\\phi$, AR-GARCH (SE) & $Q(10)$ of $\\hat z_t$, const. mean & $Q(10)$ of $\\hat z_t$, AR(1) & $\\Delta$BIC',
                 '& $p$ (BIC) & $\\hat\\phi$, OLS (SE) & $\\hat\\phi$, AR-GARCH (SE) & $Q(10)$ al $\\hat z_t$, medie const. & $Q(10)$ al $\\hat z_t$, AR(1) & $\\Delta$BIC'),
    [agrow(k) for k in ASSETS], size='scriptsize') + items(
    T('$p$: AR order chosen by BIC with a constant variance; SE of the AR-GARCH: robust; $\\Delta$BIC: AR(1)-GARCH-t minus GARCH-t with a constant mean (negative: AR(1) preferred)',
      '$p$: ordinul AR ales după BIC cu varianță constantă; SE pentru AR-GARCH: robuste; $\\Delta$BIC: AR(1)-GARCH-t minus GARCH-t cu medie constantă (negativ: se preferă AR(1))'),
    T('S\\&P 500: $\\hat\\phi$ shrinks from @{ag.sp500.phio} to @{ag.sp500.phig}: the OLS value is driven by the stormy days of 2008 and 2020', 'S\\&P 500: $\\hat\\phi$ scade de la @{ag.sp500.phio} la @{ag.sp500.phig}: valoarea OLS este determinată de zilele agitate din 2008 și 2020'),
    T('BET: a positive autocorrelation survives ($\\hat\\phi = @{ag.bet.phig}$, $t = @{ag.bet.tg}$): slow price adjustment in a less liquid market; EUR/RON: from @{ag.eurron.phio} to @{ag.eurron.phig}, no longer significant',
      'BET: autocorelația pozitivă rezistă ($\\hat\\phi = @{ag.bet.phig}$, $t = @{ag.bet.tg}$): ajustarea lentă a prețurilor pe o piață mai puțin lichidă; EUR/RON: de la @{ag.eurron.phio} la @{ag.eurron.phig}, nu mai este semnificativ'),
    T('Interpretation: the mean model and the variance model must be chosen together; the conclusions about $\\phi$ can change', 'Interpretare: modelul pentru medie și cel pentru varianță trebuie alese împreună; concluziile despre $\\phi$ se pot schimba')) + ql('TSA_ch5_arma_garch'), 'footnotesize')

chart(T('Forecast intervals that breathe', 'Intervale de prognoză care respiră'), 'tsa_ch5_bands', 'TSA_ch5_arma_garch', [
    T('BET, one-day-ahead 95\\% intervals: AR(1) with constant variance (width @{bd.wc} points every day) and AR(1)-GARCH(1,1)-t (width from @{bd.wgmin} to @{bd.wgmax} points in this window)',
      'BET, intervale de 95\\% cu o zi înainte: AR(1) cu varianță constantă (lățimea de @{bd.wc} puncte în fiecare zi) și AR(1)-GARCH(1,1)-t (lățimea de la @{bd.wgmin} la @{bd.wgmax} puncte în această fereastră)')],
    h='0.52\\textheight')

interp(('the two intervals', 'celor două intervale'), [
    (T('Over the whole sample both miss about 5\\% of the days: @{bd.out_c}\\% and @{bd.out_g}\\%', 'Pe întregul eșantion ambele ratează circa 5\\% dintre zile: @{bd.out_c}\\% și @{bd.out_g}\\%'),
     [T('the constant interval is right on average, but wrong almost every day', 'intervalul constant este corect în medie, dar greșit aproape în fiecare zi')]),
    (T('In 2017 (calm): the constant interval misses @{bd.out_c_calm2017}\\% of the days, the GARCH interval @{bd.out_g_calm2017}\\%', 'În 2017 (an liniștit): intervalul constant ratează @{bd.out_c_calm2017}\\% dintre zile, intervalul GARCH @{bd.out_g_calm2017}\\%'),
     [T('in March--April 2020: @{bd.out_c_covid}\\% against @{bd.out_g_covid}\\%: the misses of the constant interval cluster in crises', 'în martie--aprilie 2020: @{bd.out_c_covid}\\% față de @{bd.out_g_covid}\\%: ratările intervalului constant se grupează în crize')]),
    T('Chapter 2 gave intervals whose width depends only on the horizon; with GARCH it also depends on the current state', 'Capitolul 2 a dat intervale a căror lățime depinde doar de orizont; cu GARCH ea depinde și de starea curentă')])

D.recap(('four markets and ARMA-GARCH', 'patru piețe și ARMA-GARCH'), [
    T('$\\alpha + \\beta$ close to 1 everywhere; EUR/RON and Bitcoin are IGARCH', '$\\alpha + \\beta$ apropiat de 1 peste tot; EUR/RON și Bitcoin sînt IGARCH'),
    T('The long-run volatility is imprecise when $\\alpha + \\beta$ is near 1', 'Volatilitatea de lungă durată este imprecisă cînd $\\alpha + \\beta$ este aproape de 1'),
    T('ARMA-GARCH: the mean of Chapter 2 with GARCH errors; joint MLE; intervals that follow the volatility', 'ARMA-GARCH: media din Capitolul 2 cu erori GARCH; MLE comun; intervale care urmează volatilitatea')])

# =============================================================================
# 7. ASIMETRIE
# =============================================================================
D.section('Asymmetry and the leverage effect', 'Asimetria și efectul de levier')

D.frame(T('The leverage effect and the news impact curve', 'Efectul de levier și curba de impact a știrilor'), items(
    (T('\\textbf{Leverage effect}: falls in stock prices raise volatility more than rises of the same size \\refChristie', '\\textbf{Efectul de levier}: scăderile prețurilor acțiunilor cresc volatilitatea mai mult decît creșterile de aceeași mărime \\refChristie'),
     [T('a lower share price raises the debt-to-equity ratio, so the equity becomes riskier; news of higher risk also lowers prices at once', 'un preț mai mic al acțiunii crește raportul datorii/capitaluri proprii, deci acțiunea devine mai riscantă; știrile despre un risc mai mare scad și ele imediat prețurile')]),
    (T('GARCH(1,1) cannot see it: $\\sigma_t^2$ depends on $\\varepsilon_{t-1}^2$, not on the sign of $\\varepsilon_{t-1}$', 'GARCH(1,1) nu îl poate surprinde: $\\sigma_t^2$ depinde de $\\varepsilon_{t-1}^2$, nu de semnul lui $\\varepsilon_{t-1}$'),
     [T('the QQ plot and the skewed t ($\\hat\\lambda < 0$) already pointed to the left tail', 'graficul QQ și t asimetrică ($\\hat\\lambda < 0$) au arătat deja spre coada stîngă')]),
    T('\\textbf{NIC} (news impact curve) \\refEN: $\\sigma_t^2$ as a function of $\\varepsilon_{t-1}$, with $\\sigma_{t-1}^2$ fixed at the unconditional variance; for GARCH, a symmetric parabola',
      '\\textbf{NIC} (news impact curve, curba de impact a știrilor) \\refEN: $\\sigma_t^2$ ca funcție de $\\varepsilon_{t-1}$, cu $\\sigma_{t-1}^2$ fixat la varianța necondiționată; pentru GARCH, o parabolă simetrică')))

D.frame(T('GJR-GARCH and EGARCH', 'GJR-GARCH și EGARCH'), items(
    (T('\\textbf{GJR-GARCH} \\refGJR: $\\sigma_t^2 = \\omega + (\\alpha + \\gamma\\,I_{t-1})\\,\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, with $I_{t-1} = 1$ if $\\varepsilon_{t-1} < 0$, else 0',
       '\\textbf{GJR-GARCH} \\refGJR: $\\sigma_t^2 = \\omega + (\\alpha + \\gamma\\,I_{t-1})\\,\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, cu $I_{t-1} = 1$ dacă $\\varepsilon_{t-1} < 0$, altfel 0'),
     [T('good news: slope $\\alpha$; bad news: $\\alpha + \\gamma$; leverage effect: $\\gamma > 0$; in \\texttt{arch}: \\texttt{o=1}', 'știri bune: panta $\\alpha$; știri proaste: $\\alpha + \\gamma$; efect de levier: $\\gamma > 0$; în \\texttt{arch}: \\texttt{o=1}'),
      T('persistence $\\alpha + \\beta + \\gamma/2$ for symmetric $z_t$ (half of the shocks are negative)', 'persistența $\\alpha + \\beta + \\gamma/2$ pentru $z_t$ simetric (jumătate dintre șocuri sînt negative)')]),
    (T('\\textbf{EGARCH} (exponential GARCH) \\refNelson: $\\ln\\sigma_t^2 = \\omega + \\alpha\\big(|z_{t-1}| - E|z_{t-1}|\\big) + \\gamma z_{t-1} + \\beta\\ln\\sigma_{t-1}^2$',
       '\\textbf{EGARCH} (exponential GARCH, GARCH exponențial) \\refNelson: $\\ln\\sigma_t^2 = \\omega + \\alpha\\big(|z_{t-1}| - E|z_{t-1}|\\big) + \\gamma z_{t-1} + \\beta\\ln\\sigma_{t-1}^2$'),
     [T('the logarithm keeps $\\sigma_t^2 > 0$ without sign restrictions; leverage effect: $\\gamma < 0$; persistence: $\\beta$', 'logaritmul păstrează $\\sigma_t^2 > 0$ fără restricții de semn; efect de levier: $\\gamma < 0$; persistența: $\\beta$')]),
    T('Both have one parameter more than GARCH(1,1); test $\\gamma = 0$ with its $t$ statistic or with an LR test, $\\chi^2(1)$', 'Ambele au un parametru în plus față de GARCH(1,1); testăm $\\gamma = 0$ cu statistica $t$ sau cu un test LR, $\\chi^2(1)$')))

chart(T('News impact curves: S\\&P 500 and Bitcoin', 'Curbele de impact ale știrilor: S\\&P 500 și Bitcoin'), 'tsa_ch5_nic', 'TSA_ch5_asymmetry', [
    T('Model curves: $\\sigma_t^2$ against $\\varepsilon_{t-1}$, with $\\sigma_{t-1}^2$ at the sample variance; dashed: a kernel (Nadaraya--Watson) estimate of $E[r_t^2 \\mid r_{t-1}]$, without a model',
      'Curbele modelelor: $\\sigma_t^2$ în funcție de $\\varepsilon_{t-1}$, cu $\\sigma_{t-1}^2$ la varianța de selecție; linia întreruptă: o estimare nucleu (Nadaraya--Watson) a lui $E[r_t^2 \\mid r_{t-1}]$, fără model'),
    T('S\\&P 500, GJR: after a fall of 2\\%, tomorrow\'s variance is @{nic.sp500.ratio} times the variance after a rise of 2\\%; Bitcoin: ratio @{nic.btc.ratio}, a symmetric curve',
      'S\\&P 500, GJR: după o scădere de 2\\%, varianța de mîine este de @{nic.sp500.ratio} ori mai mare decît după o creștere de 2\\%; Bitcoin: raportul @{nic.btc.ratio}, o curbă simetrică')],
    h='0.54\\textheight')


def arow(k):
    return (f'{NAMES[k]} & $@{{as.{k}.a}}$ & $@{{as.{k}.g}}$ & $@{{as.{k}.gt}}$ & $@{{as.{k}.lr}}$ & @{{as.{k}.lrp}} & ${{@{{as.{k}.dbic}}}}$ & '
            f'${{@{{as.{k}.eg}}}}$ & ${{@{{as.{k}.egt}}}}$')


D.frame(T('Asymmetry in four markets', 'Asimetria pe patru piețe'), table(
    'lrrrrrrrr', T('& GJR $\\alpha$ & GJR $\\gamma$ & $t(\\gamma)$ & LR & p & $\\Delta$BIC & EGARCH $\\gamma$ & $t(\\gamma)$',
                   '& GJR $\\alpha$ & GJR $\\gamma$ & $t(\\gamma)$ & LR & p & $\\Delta$BIC & EGARCH $\\gamma$ & $t(\\gamma)$'),
    [arow(k) for k in ASSETS], size='scriptsize') + items(
    T('Student-t innovations; robust $t$ statistics; LR $= 2(\\ell_{GJR} - \\ell_{GARCH}) \\sim \\chi^2(1)$; $\\Delta$BIC: GJR minus GARCH (negative: GJR preferred)',
      'Inovații Student-t; statistici $t$ robuste; LR $= 2(\\ell_{GJR} - \\ell_{GARCH}) \\sim \\chi^2(1)$; $\\Delta$BIC: GJR minus GARCH (negativ: se preferă GJR)'),
    T('S\\&P 500: GJR $\\hat\\alpha = @{as.sp500.a}$: only bad news raises volatility; a very strong leverage effect', 'S\\&P 500: GJR $\\hat\\alpha = @{as.sp500.a}$: doar știrile proaste cresc volatilitatea; un efect de levier foarte puternic'),
    T('BET: weak, significant by LR, but BIC prefers GARCH; Bitcoin: none; EUR/RON: $r_t > 0$ means a weaker leu, so the ``bad news\'\' of GJR are days when the leu strengthens',
      'BET: slab, semnificativ după LR, dar BIC preferă GARCH; Bitcoin: niciun efect; EUR/RON: $r_t > 0$ înseamnă un leu mai slab, deci „știrile proaste” din GJR sînt zilele în care leul se întărește'),
    T('Interpretation: leverage is a property of mature equity markets, not of every financial series', 'Interpretare: efectul de levier este o proprietate a piețelor mature de acțiuni, nu a oricărei serii financiare')) + ql('TSA_ch5_asymmetry'), 'footnotesize')

D.recap(('asymmetry', 'asimetria'), [
    T('GJR: an extra slope $\\gamma$ for negative shocks; EGARCH: a sign term $\\gamma z_{t-1}$ in $\\ln\\sigma_t^2$', 'GJR: o pantă suplimentară $\\gamma$ pentru șocurile negative; EGARCH: un termen de semn $\\gamma z_{t-1}$ în $\\ln\\sigma_t^2$'),
    T('The news impact curve shows the asymmetry at a glance', 'Curba de impact a știrilor arată asimetria dintr-o privire'),
    T('Strong in the S\\&P 500, weak in the BET, absent in Bitcoin', 'Puternică la S\\&P 500, slabă la BET, absentă la Bitcoin')])

# =============================================================================
# 8. DIAGNOSTIC ȘI ALEGEREA MODELULUI
# =============================================================================
D.section('Diagnostics and model choice', 'Diagnosticare și alegerea modelului')

D.frame(T('Checking a fitted model', 'Verificarea unui model estimat'), items(
    (T('If the model is right, the \\textbf{standardised residuals} $\\hat z_t = (r_t - \\hat\\mu_t)/\\hat\\sigma_t$ are close to i.i.d.\\ with mean 0 and variance 1',
       'Dacă modelul este corect, \\textbf{reziduurile standardizate} $\\hat z_t = (r_t - \\hat\\mu_t)/\\hat\\sigma_t$ sînt aproape i.i.d., cu media 0 și varianța 1'),
     [T('Ljung--Box $Q(10)$ on $\\hat z_t$: no autocorrelation left in the mean (the check of Chapter 2)', 'Ljung--Box $Q(10)$ pe $\\hat z_t$: nu a rămas autocorelație în medie (verificarea din Capitolul 2)'),
      T('Ljung--Box $Q(10)$ on $\\hat z_t^2$ and ARCH-LM on $\\hat z_t$: no ARCH effects left', 'Ljung--Box $Q(10)$ pe $\\hat z_t^2$ și ARCH-LM pe $\\hat z_t$: nu au rămas efecte ARCH'),
      T('QQ plot of $\\hat z_t$: is the assumed distribution right?', 'graficul QQ al lui $\\hat z_t$: este corectă distribuția presupusă?')]),
    (T('\\textbf{Sign-bias test} \\refEN: regress $\\hat z_t^2$ on $I_{t-1}$, $I_{t-1}\\hat\\varepsilon_{t-1}$ and $(1 - I_{t-1})\\hat\\varepsilon_{t-1}$',
       '\\textbf{Testul de asimetrie (sign bias)} \\refEN: regresia lui $\\hat z_t^2$ pe $I_{t-1}$, $I_{t-1}\\hat\\varepsilon_{t-1}$ și $(1 - I_{t-1})\\hat\\varepsilon_{t-1}$'),
     [T('joint test $T R^2 \\sim \\chi^2(3)$: a rejection means that the sign of past shocks still predicts the variance', 'testul comun $T R^2 \\sim \\chi^2(3)$: o respingere înseamnă că semnul șocurilor trecute încă anticipează varianța')]),
    T('A passed diagnostic does not prove the model right; a failed one shows where it is wrong', 'Un diagnostic trecut nu dovedește că modelul este corect; unul picat arată unde greșește')))

chart(T('What GARCH removes', 'Autocorelațiile eliminate de GARCH'), 'tsa_ch5_acf_diag', 'TSA_ch5_diagnostics', [
    T('S\\&P 500: ACF of $r_t^2$: $@{acf.r1}$ at lag 1, still $@{acf.r50}$ at lag 50; ACF of $\\hat z_t^2$ (GARCH(1,1)-t): $@{acf.z1}$ at lag 1, at most $@{acf.zmax}$ in absolute value; band $\\pm@{acf.band}$',
      'S\\&P 500: ACF a lui $r_t^2$: $@{acf.r1}$ la decalajul 1, încă $@{acf.r50}$ la decalajul 50; ACF a lui $\\hat z_t^2$ (GARCH(1,1)-t): $@{acf.z1}$ la decalajul 1, cel mult $@{acf.zmax}$ în valoare absolută; banda $\\pm@{acf.band}$'),
    T('Interpretation: four parameters absorb almost all the volatility clustering of @{yrs} years of daily data', 'Interpretare: patru parametri absorb aproape tot volatility clustering din @{yrs} ani de date zilnice')],
    h='0.48\\textheight')

D.frame(T('Diagnostics on real data', 'Diagnosticare pe date reale'), table(
    'lrrr', T('S\\&P 500 & $Q(10)$ (p) & $Q(10)$ of squares (p) & ARCH-LM(5) (p)', 'S\\&P 500 & $Q(10)$ (p) & $Q(10)$ al pătratelor (p) & ARCH-LM(5) (p)'),
    [T('returns $r_t$', 'randamente $r_t$') + ' & $@{dg.r}$ (@{dg.r.p}) & $@{dg.r2}$ (@{dg.r2.p}) & $@{dg.lm.r}$ (@{dg.lm.r.p})',
     T('$\\hat z_t$, GARCH-N', '$\\hat z_t$, GARCH-N') + ' & $@{dg.normal.z}$ (@{dg.normal.z.p}) & $@{dg.normal.z2}$ (@{dg.normal.z2.p}) & $@{dg.normal.lm}$ (@{dg.normal.lm.p})',
     T('$\\hat z_t$, GARCH-t', '$\\hat z_t$, GARCH-t') + ' & $@{dg.t.z}$ (@{dg.t.z.p}) & $@{dg.t.z2}$ (@{dg.t.z2.p}) & $@{dg.t.lm}$ (@{dg.t.lm.p})'],
    size='scriptsize') + items(
    T('GARCH removes the ARCH effects ($Q(10)$ of squares: from $@{dg.r2}$ to $@{dg.t.z2}$); a little autocorrelation remains in the mean (constant mean; an AR(1) helps, Section 6)',
      'GARCH elimină efectele ARCH ($Q(10)$ al pătratelor: de la $@{dg.r2}$ la $@{dg.t.z2}$); rămîne puțină autocorelație în medie (medie constantă; un AR(1) ajută, secțiunea 6)'),
    T('Sign bias for the GARCH-t of the S\\&P 500: $\\chi^2(3) = @{as.sp500.sb}$ (p @{as.sp500.sbp}): the asymmetry of Section 7 is missing', 'Testul de asimetrie pentru GARCH-t pe S\\&P 500: $\\chi^2(3) = @{as.sp500.sb}$ (p @{as.sp500.sbp}): lipsește asimetria din secțiunea 7'),
    T('$Q(10)$ of $\\hat z_t^2$, GARCH-t: BET @{dm.bet.z2} (p = @{dm.bet.z2p}), Bitcoin @{dm.btc.z2} (p = @{dm.btc.z2p}), EUR/RON @{dm.eurron.z2}: suspiciously small (next slide)',
      '$Q(10)$ al lui $\\hat z_t^2$, GARCH-t: BET @{dm.bet.z2} (p = @{dm.bet.z2p}), Bitcoin @{dm.btc.z2} (p = @{dm.btc.z2p}), EUR/RON @{dm.eurron.z2}: suspect de mic (slide-ul următor)')) + ql('TSA_ch5_diagnostics'), 'footnotesize')

D.frame(T('Case study: the EUR/RON on @{mg.date}', 'Studiu de caz: EUR/RON pe @{mg.date}'), cols(items(
    (T('In April 2025 the reference rate moved by @{mg.abs}\\% a day on average: the BNR held the leu almost fixed', 'În aprilie 2025, cursul de referință a variat în medie cu @{mg.abs}\\% pe zi: BNR a menținut leul aproape fix'),
     [T('the IGARCH volatility fell to $\\hat\\sigma_t = @{mg.sig}\\%$; then the rate rose by @{mg.r}\\% (two days after the first round of the presidential election) and by @{mg.rn}\\% the next day',
        'volatilitatea IGARCH a scăzut la $\\hat\\sigma_t = @{mg.sig}\\%$; apoi cursul a crescut cu @{mg.r}\\% (la două zile după primul tur al alegerilor prezidențiale) și cu @{mg.rn}\\% a doua zi')]),
    (T('Standardised residual: $\\hat z_t = @{mg.z}$, a ``@{mg.z}-sigma\'\' day; the next largest is @{mg.z2nd}', 'Reziduul standardizat: $\\hat z_t = @{mg.z}$, o zi „de @{mg.z} sigma”; următorul ca mărime este @{mg.z2nd}'),
     [T('$Q(10)$ of $\\hat z_t^2$: @{mg.q} with this day, @{mg.qwo} (p = @{mg.qwop}) without it: one value hides the remaining ARCH effects', '$Q(10)$ al lui $\\hat z_t^2$: @{mg.q} cu această zi, @{mg.qwo} (p = @{mg.qwop}) fără ea: o singură valoare ascunde efectele ARCH rămase')]),
    T('Lesson: a managed rate is not a free market; GARCH cannot foresee a policy decision; always plot $\\hat z_t$ before trusting a test', 'Lecția: un curs administrat nu este o piață liberă; GARCH nu poate anticipa o decizie de politică; reprezentăm întotdeauna grafic $\\hat z_t$ înainte de a ne baza pe un test')),
    ph('bnr', T('The National Bank of Romania, Bucharest', 'Banca Națională a României, București'), h='0.28\\textheight'), wl='0.62', wr='0.34') + '\n\\vspace{1mm}\n' + ql('TSA_ch5_diagnostics'), 'footnotesize')


def msrow(m, lab):
    return f'{lab} & ' + ' & '.join(f'@{{ms.{m}.{d}.ll}} & @{{ms.{m}.{d}.dbic}}' for d in ['Normal', 't', 'skewedt'])


D.frame(T('Nine models for the S\\&P 500', 'Nouă modele pentru S\\&P 500'), items(
    T('AIC $= -2\\ell + 2k$ \\refAkaike; BIC $= -2\\ell + k\\ln T$ \\refSchwarz; $k$ parameters; compare models fitted on the same data', 'AIC $= -2\\ell + 2k$ \\refAkaike; BIC $= -2\\ell + k\\ln T$ \\refSchwarz; $k$ parametri; comparăm modele estimate pe aceleași date')) + table(
    'lrrrrrr', T('& \\multicolumn{2}{c}{Normal} & \\multicolumn{2}{c}{Student-t} & \\multicolumn{2}{c}{skewed t} \\\\ & log-lik. & $\\Delta$BIC & log-lik. & $\\Delta$BIC & log-lik. & $\\Delta$BIC',
                 '& \\multicolumn{2}{c}{Normală} & \\multicolumn{2}{c}{Student-t} & \\multicolumn{2}{c}{t asimetrică} \\\\ & log-verosim. & $\\Delta$BIC & log-verosim. & $\\Delta$BIC & log-verosim. & $\\Delta$BIC'),
    [msrow('GARCH', 'GARCH(1,1)'), msrow('GJR', 'GJR-GARCH(1,1)'), msrow('EGARCH', 'EGARCH(1,1)')], size='scriptsize') + items(
    T('$\\Delta$BIC: the difference from the best model, @{ms.best}', '$\\Delta$BIC: diferența față de cel mai bun model, @{ms.best}'),
    T('From the Normal GARCH: asymmetry alone (Normal EGARCH) lowers BIC by @{ms.gain.asym}, heavy tails alone (GARCH-skewed t) by @{ms.gain.dist}, both by @{ms.gain.both}',
      'Pornind de la GARCH Normal: doar asimetria (EGARCH Normal) reduce BIC cu @{ms.gain.asym}, doar cozile groase (GARCH cu t asimetrică) cu @{ms.gain.dist}, ambele cu @{ms.gain.both}'),
    T('Interpretation: with @{sty.sp500.n} observations even small gains are ``significant\'\'; the out-of-sample comparison of Section 9 is the real test',
      'Interpretare: cu @{sty.sp500.n} observații, chiar și cîștigurile mici sînt „semnificative”; comparația în afara eșantionului din secțiunea 9 este testul real')) + ql('TSA_ch5_diagnostics'), 'footnotesize')

D.recap(('diagnostics and model choice', 'diagnosticare și alegerea modelului'), [
    T('Standardised residuals: no autocorrelation in $\\hat z_t$ and $\\hat z_t^2$, the right distribution, no sign bias', 'Reziduurile standardizate: fără autocorelație în $\\hat z_t$ și $\\hat z_t^2$, distribuția potrivită, fără efect de semn'),
    T('One extreme $\\hat z_t$ can hide everything: plot before testing', 'Un singur $\\hat z_t$ extrem poate ascunde totul: graficul înaintea testului'),
    T('AIC/BIC on the same data; in-sample fit is not forecasting ability', 'AIC/BIC pe aceleași date; potrivirea în eșantion nu înseamnă capacitate de prognoză')])

# =============================================================================
# 9. PROGNOZE
# =============================================================================
D.section('Variance forecasts and their evaluation', 'Prognoza varianței și evaluarea ei')

D.frame(T('Multi-step forecasts', 'Prognoze pe mai mulți pași'), items(
    (T('One step: $\\sigma_{t+1}^2 = \\omega + \\alpha\\varepsilon_t^2 + \\beta\\sigma_t^2$ is known at the end of day $t$', 'Un pas: $\\sigma_{t+1}^2 = \\omega + \\alpha\\varepsilon_t^2 + \\beta\\sigma_t^2$ este cunoscut la sfîrșitul zilei $t$'),
     [T('two steps: $E_t[\\sigma_{t+2}^2] = \\omega + \\alpha E_t[\\varepsilon_{t+1}^2] + \\beta\\sigma_{t+1}^2 = \\omega + (\\alpha + \\beta)\\sigma_{t+1}^2$, since $E_t[\\varepsilon_{t+1}^2] = \\sigma_{t+1}^2$',
        'doi pași: $E_t[\\sigma_{t+2}^2] = \\omega + \\alpha E_t[\\varepsilon_{t+1}^2] + \\beta\\sigma_{t+1}^2 = \\omega + (\\alpha + \\beta)\\sigma_{t+1}^2$, deoarece $E_t[\\varepsilon_{t+1}^2] = \\sigma_{t+1}^2$')]),
    (T('By induction: $E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$', 'Prin inducție: $E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$'),
     [T('the forecast converges to the long-run variance, like an AR(1) forecast converges to its mean (Chapter 2)', 'prognoza converge spre varianța de lungă durată, așa cum prognoza unui AR(1) converge spre medie (Capitolul 2)')]),
    (T('Variance of the $H$-day return: the sum of the daily forecasts (the returns are uncorrelated)', 'Varianța randamentului pe $H$ zile: suma prognozelor zilnice (randamentele sînt necorelate)'),
     [T('$\\mathrm{Var}_t(r_{t+1} + \\dots + r_{t+H}) = \\sum_{h=1}^{H} E_t[\\sigma_{t+h}^2]$, not $H\\sigma_{t+1}^2$', '$\\mathrm{Var}_t(r_{t+1} + \\dots + r_{t+H}) = \\sum_{h=1}^{H} E_t[\\sigma_{t+h}^2]$, nu $H\\sigma_{t+1}^2$'),
      T('the square-root-of-time rule $\\sigma_{t+1}\\sqrt{H}$ is too high in a crisis and too low in a calm period', 'regula rădăcinii pătrate a timpului, $\\sigma_{t+1}\\sqrt{H}$, dă valori prea mari în criză și prea mici într-o perioadă liniștită')])))

D.frame(T('Worked example: the S\\&P 500 on @{end}', 'Exemplu rezolvat: S\\&P 500 la @{end}'), items(
    T('GARCH(1,1)-t: $\\alpha + \\beta = @{fc.pers}$, $\\bar\\sigma^2 = @{fc.s2bar}$; forecast for the next day $\\sigma_{t+1}^2 = @{fc.s2n}$ (below the long-run level)',
      'GARCH(1,1)-t: $\\alpha + \\beta = @{fc.pers}$, $\\bar\\sigma^2 = @{fc.s2bar}$; prognoza pentru ziua următoare $\\sigma_{t+1}^2 = @{fc.s2n}$ (sub nivelul de lungă durată)'),
    (T('Step by step', 'Pas cu pas'),
     [T('day 10: $@{fc.s2bar} + @{fc.pers}^{9}(@{fc.s2n} - @{fc.s2bar}) = @{fc.s2bar} + @{fc.pers9} \\times (@{fc.s2n} - @{fc.s2bar}) = @{fc.s210}$',
        'ziua 10: $@{fc.s2bar} + @{fc.pers}^{9}(@{fc.s2n} - @{fc.s2bar}) = @{fc.s2bar} + @{fc.pers9} \\times (@{fc.s2n} - @{fc.s2bar}) = @{fc.s210}$'),
      T('10-day variance: the sum of the ten daily forecasts $= @{fc.sum10}$, so the 10-day volatility is $@{fc.vol10}\\%$', 'varianța pe 10 zile: suma celor zece prognoze zilnice $= @{fc.sum10}$, deci volatilitatea pe 10 zile este $@{fc.vol10}\\%$'),
      T('square-root-of-time rule: $\\sqrt{10 \\times @{fc.s2n}} = @{fc.vol10sq}\\%$: too low, because volatility is expected to rise towards its long-run level',
        'regula rădăcinii pătrate a timpului: $\\sqrt{10 \\times @{fc.s2n}} = @{fc.vol10sq}\\%$: prea mică, deoarece volatilitatea este așteptată să crească spre nivelul de lungă durată')])))

chart(T('The term structure of volatility', 'Structura la termen a volatilității'), 'tsa_ch5_term_structure', 'TSA_ch5_forecasts', [
    T('S\\&P 500, GARCH(1,1)-t on all data; forecasts made on a calm day (@{ts.calm.d}), at the COVID-19 peak (@{ts.covid.d}) and on @{end}',
      'S\\&P 500, GARCH(1,1)-t pe toate datele; prognoze făcute într-o zi liniștită (@{ts.calm.d}), în vîrful crizei COVID-19 (@{ts.covid.d}) și la @{end}'),
    T('Calm day: @{ts.calm.h1}\\% tomorrow, @{ts.calm.avg250}\\% on average over a year; COVID-19 peak: @{ts.covid.h1}\\% tomorrow, still @{ts.covid.avg250}\\% on average over a year',
      'Ziua liniștită: @{ts.calm.h1}\\% mîine, în medie @{ts.calm.avg250}\\% pe un an; vîrful COVID-19: @{ts.covid.h1}\\% mîine, încă @{ts.covid.avg250}\\% în medie pe un an'),
    T('Interpretation: all curves move towards the long-run level (@{ts.lr}\\%) at the same slow speed; the slope tells whether we are in a calm period or in a storm',
      'Interpretare: toate curbele se îndreaptă spre nivelul de lungă durată (@{ts.lr}\\%) cu aceeași viteză redusă; panta arată dacă sîntem într-o perioadă liniștită sau într-o furtună')],
    h='0.46\\textheight')

D.frame(T('How to evaluate a volatility forecast', 'Evaluarea unei prognoze de volatilitate'), items(
    (T('The true $\\sigma_t^2$ is never observed: we compare the forecast $h_t$ with a noisy \\textbf{proxy}, here $r_t^2$', 'Adevăratul $\\sigma_t^2$ nu este observat niciodată: comparăm prognoza $h_t$ cu o \\textbf{aproximare} zgomotoasă, aici $r_t^2$'),
     [T('$E_{t-1}[r_t^2] = \\sigma_t^2$ (with $\\mu \\approx 0$), but $r_t^2$ is very noisy: a low $R^2$ even for a perfect forecast \\refAB',
        '$E_{t-1}[r_t^2] = \\sigma_t^2$ (cu $\\mu \\approx 0$), dar $r_t^2$ este foarte zgomotos: un $R^2$ mic chiar și pentru o prognoză perfectă \\refAB')]),
    (T('\\textbf{QLIKE} loss \\refPatton: $L(r_t^2, h_t) = \\dfrac{r_t^2}{h_t} + \\ln h_t$ (up to a constant); the smaller the mean, the better',
       'Funcția de pierdere \\textbf{QLIKE} \\refPatton: $L(r_t^2, h_t) = \\dfrac{r_t^2}{h_t} + \\ln h_t$ (pînă la o constantă); cu cît media este mai mică, cu atît mai bine'),
     [T('it ranks forecasts correctly even with a noisy proxy; it is minus the Normal log-likelihood; it punishes forecasts that are too low', 'ordonează corect prognozele chiar și cu o aproximare zgomotoasă; este minus log-verosimilitatea Normală; penalizează prognozele prea mici')]),
    (T('\\textbf{Out of sample} (Chapter 0): from 2015, each forecast uses only data up to the day before; the models are re-estimated every 250 days',
       '\\textbf{În afara eșantionului} (Capitolul 0): din 2015, fiecare prognoză folosește doar date pînă în ziua precedentă; modelele se reestimează la fiecare 250 de zile'),
     [T('\\textbf{DM} (Diebold--Mariano) test \\refDM: is the mean loss difference zero? $t$ statistic with HAC (heteroskedasticity and autocorrelation consistent) standard errors',
        'testul \\textbf{DM} (Diebold--Mariano) \\refDM: este diferența medie a pierderilor egală cu zero? statistica $t$ cu erori standard HAC (heteroskedasticity and autocorrelation consistent, robuste la heteroscedasticitate și autocorelație)')])))

chart(T('GARCH against EWMA, out of sample', 'GARCH față de EWMA, în afara eșantionului'), 'tsa_ch5_forecast_eval', 'TSA_ch5_forecasts', [
    T('One-day-ahead variance forecasts from 2015 to @{end}: GARCH(1,1)-t (re-estimated every 250 days) against EWMA with $\\lambda = 0.94$; a rising curve means that GARCH-t has the smaller QLIKE',
      'Prognoze ale varianței cu o zi înainte, din 2015 pînă la @{end}: GARCH(1,1)-t (reestimat la fiecare 250 de zile) față de EWMA cu $\\lambda = 0{,}94$; o curbă crescătoare înseamnă că GARCH-t are QLIKE mai mic')],
    h='0.50\\textheight')


def frow(k):
    return (f'{NAMES[k]} & @{{fe.{k}.n}} & $@{{fe.{k}.q.garch}}$ & $@{{fe.{k}.q.gjr}}$ & $@{{fe.{k}.q.ewma}}$ & ${{@{{fe.{k}.dm}}}}$ (@{{fe.{k}.dmp}})')


D.frame(T('Interpreting the forecast comparison', 'Interpretarea comparației prognozelor'), table(
    'lrrrrr', T('2015--2026 & days & QLIKE GARCH-t & QLIKE GJR-t & QLIKE EWMA & DM $t$, GARCH-t against EWMA (p)',
                '2015--2026 & zile & QLIKE GARCH-t & QLIKE GJR-t & QLIKE EWMA & DM $t$, GARCH-t față de EWMA (p)'),
    [frow(k) for k in ASSETS], size='scriptsize') + items(
    T('S\\&P 500 and BET: GARCH-t beats EWMA ($t < -1.96$); GJR-t has the smallest QLIKE (the leverage effect pays off out of sample)', 'S\\&P 500 și BET: GARCH-t este mai bun decît EWMA ($t < -1{,}96$); GJR-t are cel mai mic QLIKE (efectul de levier contează și în afara eșantionului)'),
    T('Bitcoin: no difference ($t = @{fe.btc.dm}$): its GARCH is an IGARCH, which is almost an EWMA', 'Bitcoin: nicio diferență ($t = @{fe.btc.dm}$): GARCH-ul lui este un IGARCH, adică aproape un EWMA'),
    T('EUR/RON: a much smaller mean QLIKE for GARCH, yet $t = @{fe.eurron.dm}$: @{ff.eurron.share}\\% of the total difference comes from one day, @{ff.eurron.day}, when EWMA forecast a near-zero variance',
      'EUR/RON: QLIKE mediu mult mai mic pentru GARCH, dar $t = @{fe.eurron.dm}$: @{ff.eurron.share}\\% din diferența totală provine dintr-o singură zi, @{ff.eurron.day}, cînd EWMA a prognozat o varianță aproape nulă'),
    T('Interpretation: QLIKE punishes forecasts that are too low; the DM test protects us from conclusions built on a single day', 'Interpretare: QLIKE penalizează prognozele prea mici; testul DM ne ferește de concluzii construite pe o singură zi')) + ql('TSA_ch5_forecasts'), 'footnotesize')

D.frame(T('Case study: Hansen and Lunde (2005)', 'Studiu de caz: Hansen și Lunde (2005)'), items(
    (T('\\textbf{Question}: does anything beat a GARCH(1,1)? \\refHL', '\\textbf{Întrebarea}: poate ceva să depășească un GARCH(1,1)? \\refHL'),
     [T('330 ARCH-type models (asymmetric, long memory, different innovations), compared out of sample on the Deutsche Mark/US dollar exchange rate and on the returns of one US stock',
        '330 de modele de tip ARCH (asimetrice, cu memorie lungă, cu inovații diferite), comparate în afara eșantionului pe cursul marcă germană/dolar american și pe randamentele unei acțiuni americane'),
      T('the proxy: realised variance from intraday returns, more precise than $r_t^2$', 'aproximarea: varianța realizată din randamente intrazilnice, mai precisă decît $r_t^2$')]),
    (T('\\textbf{Finding}: for the exchange rate, no model beat GARCH(1,1) significantly; for the stock, models with a leverage effect did better',
       '\\textbf{Rezultatul}: la cursul de schimb, niciun model nu a depășit semnificativ GARCH(1,1); la acțiune, modelele cu efect de levier au fost mai bune'),
     [T('the same pattern as our table: asymmetry matters for equities, not for currencies or Bitcoin', 'același tipar ca în tabelul nostru: asimetria contează pentru acțiuni, nu pentru cursuri de schimb sau Bitcoin')]),
    T('\\textbf{Legacy}: GARCH(1,1) is the benchmark every new volatility model, including machine learning (Chapter 9), must beat out of sample',
      '\\textbf{Moștenirea}: GARCH(1,1) este reperul pe care orice model nou de volatilitate, inclusiv de învățare automată (Capitolul 9), trebuie să-l depășească în afara eșantionului')))

D.recap(('variance forecasts', 'prognoza varianței'), [
    T('$E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$; $H$-day variance = sum of the daily forecasts',
      '$E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$; varianța pe $H$ zile = suma prognozelor zilnice'),
    T('Compare forecasts out of sample with QLIKE and the Diebold--Mariano test', 'Comparăm prognozele în afara eșantionului cu QLIKE și testul Diebold--Mariano'),
    T('GARCH beats EWMA where volatility mean-reverts', 'GARCH este mai bun decît EWMA acolo unde volatilitatea revine la medie')])

# =============================================================================
# 10. VaR
# =============================================================================
D.section('An application: VaR 1\\% from a GARCH model', 'O aplicație: VaR 1\\% dintr-un model GARCH')

D.frame(T('Conditional VaR', 'VaR condiționat'), items(
    (T('\\textbf{VaR} (value at risk) at level $\\alpha$: the loss exceeded with probability $\\alpha$, $\\mathrm{VaR}_\\alpha = -q_\\alpha(r_{t+1})$, with $q_\\alpha$ the $\\alpha$-quantile',
       '\\textbf{VaR} (value at risk, valoarea expusă la risc) la nivelul $\\alpha$: pierderea depășită cu probabilitatea $\\alpha$, $\\mathrm{VaR}_\\alpha = -q_\\alpha(r_{t+1})$, unde $q_\\alpha$ este cuantila de ordin $\\alpha$'),
     [T('here $\\alpha = 1\\%$: VaR 1\\%, a loss exceeded on one day in a hundred', 'aici $\\alpha = 1\\%$: VaR 1\\%, o pierdere depășită într-o zi din o sută')]),
    (T('With $r_{t+1} = \\mu + \\sigma_{t+1}z_{t+1}$: $\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}\\,q_\\alpha(z))$', 'Cu $r_{t+1} = \\mu + \\sigma_{t+1}z_{t+1}$: $\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}\\,q_\\alpha(z))$'),
     [T('Normal: $q_{0.01} = @{var.qn}$; standardised t: $q_{0.01} = t_\\nu^{-1}(0.01)\\sqrt{(\\nu - 2)/\\nu}$', 'Normală: $q_{0.01} = @{var.qn}$; t standardizată: $q_{0.01} = t_\\nu^{-1}(0.01)\\sqrt{(\\nu - 2)/\\nu}$'),
      T('the VaR moves every day with $\\sigma_{t+1}$: high in storms, low in calm periods', 'VaR se schimbă în fiecare zi odată cu $\\sigma_{t+1}$: mare în furtuni, mic în perioadele liniștite')]),
    (T('\\textbf{Worked example}: S\\&P 500 on @{end}, GARCH(1,1)-t: $\\hat\\mu = @{var.mu}$, $\\sigma_{t+1} = @{var.sig}$, $\\hat\\nu = @{var.nu}$',
       '\\textbf{Exemplu rezolvat}: S\\&P 500 la @{end}, GARCH(1,1)-t: $\\hat\\mu = @{var.mu}$, $\\sigma_{t+1} = @{var.sig}$, $\\hat\\nu = @{var.nu}$'),
     [T('$q = @{var.t} \\times @{var.sc} = @{var.qt}$; VaR 1\\% $= -(@{var.mu} + @{var.sig} \\times (@{var.qt})) = @{var.v1}\\%$', '$q = @{var.t} \\times @{var.sc} = @{var.qt}$; VaR 1\\% $= -(@{var.mu} + @{var.sig} \\times (@{var.qt})) = @{var.v1}\\%$'),
      T('with Normal $z_t$: $@{var.vn}\\%$; over 10 days (sum of the variances, $\\mu$ ignored): $@{var.qa} \\times \\sqrt{@{fc.sum10}} = @{var.v10}\\%$',
        'cu $z_t$ Normale: $@{var.vn}\\%$; pe 10 zile (suma varianțelor, fără $\\mu$): $@{var.qa} \\times \\sqrt{@{fc.sum10}} = @{var.v10}\\%$')])))

chart(T('VaR 1\\% through the COVID-19 crash', 'VaR 1\\% în timpul crahului COVID-19'), 'tsa_ch5_var', 'TSA_ch5_forecasts', [
    T('S\\&P 500, July 2019 -- June 2021 (@{vf.n} days): one-day VaR 1\\% from GARCH(1,1)-t (re-estimated every 250 days) and from EWMA with Normal $z_t$',
      'S\\&P 500, iulie 2019 -- iunie 2021 (@{vf.n} zile): VaR 1\\% pe o zi din GARCH(1,1)-t (reestimat la fiecare 250 de zile) și din EWMA cu $z_t$ Normale'),
    T('Exceedances (return below $-$VaR): GARCH-t @{vf.eg}, EWMA-Normal @{vf.ee}, about @{vf.exp} expected; both models react, but only after the first large falls of March 2020',
      'Depășiri (randament sub $-$VaR): GARCH-t @{vf.eg}, EWMA-Normal @{vf.ee}, circa @{vf.exp} așteptate; ambele modele reacționează, dar abia după primele scăderi mari din martie 2020')],
    h='0.48\\textheight')


def vrow(k):
    return f'{NAMES[k]} & @{{fe.{k}.n}} & @{{fe.{k}.nexp}} & @{{fe.{k}.neg}} (@{{fe.{k}.eg}}\\%) & @{{fe.{k}.nee}} (@{{fe.{k}.ee}}\\%)'


D.frame(T('Exceedances 2015--2026', 'Depășiri 2015--2026'), table(
    'lrrrr', T('VaR 1\\% & days & expected & GARCH-t & EWMA-Normal', 'VaR 1\\% & zile & așteptat & GARCH-t & EWMA-Normal'),
    [vrow(k) for k in ASSETS], size='footnotesize') + items(
    T('A correct VaR 1\\% is exceeded on 1\\% of the days, and the exceedances do not cluster', 'Un VaR 1\\% corect este depășit în 1\\% dintre zile, iar depășirile nu apar grupat'),
    T('EWMA-Normal: @{fe.ee.min}--@{fe.ee.max}\\% everywhere: the Normal quantile is too small for heavy tails', 'EWMA-Normal: @{fe.ee.min}--@{fe.ee.max}\\% peste tot: cuantila Normală este prea mică pentru cozi groase'),
    T('GARCH-t: @{fe.eg.min}--@{fe.eg.max}\\%: much closer to 1\\%; the highest for the S\\&P 500, where the leverage effect is missing',
      'GARCH-t: @{fe.eg.min}--@{fe.eg.max}\\%: mult mai aproape de 1\\%; cel mai mare pentru S\\&P 500, unde lipsește efectul de levier'),
    T('Are the differences significant? The test of \\refKupiec\\ compares the number of exceedances with a binomial distribution', 'Sînt diferențele semnificative? Testul \\refKupiec\\ compară numărul de depășiri cu o distribuție binomială')) + ql('TSA_ch5_forecasts'), 'footnotesize')

D.recap(('VaR 1\\% from a GARCH model', 'VaR 1\\% dintr-un model GARCH'), [
    T('$\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}q_\\alpha(z))$: a dynamic volatility and the right quantile', '$\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}q_\\alpha(z))$: o volatilitate dinamică și cuantila potrivită'),
    T('Normal quantiles underestimate the risk; Student-t is much closer to 1\\%', 'Cuantilele Normale subestimează riscul; Student-t este mult mai aproape de 1\\%'),
    T('Several series at once (covariances of returns): multivariate GARCH, Chapter 14 (self-study)', 'Mai multe serii deodată (covarianțele randamentelor): GARCH multivariat, Capitolul 14 (studiu individual)')])

# =============================================================================
# 11. AI
# =============================================================================
D.section('Possible contribution of AI', 'Contribuția posibilă a AI')

D.frame(T('Possible contribution of AI', 'Contribuția posibilă a AI'), items(
    T('\\textbf{Code}: a first draft of a script that runs ARCH-LM, fits GARCH, GJR and EGARCH with \\texttt{arch} on many series and tabulates the results',
      '\\textbf{Cod}: o primă versiune a unui script care aplică ARCH-LM, estimează GARCH, GJR și EGARCH cu \\texttt{arch} pe multe serii și tabelează rezultatele'),
    T('\\textbf{Explanation}: a second explanation of the ARMA(1,1) form of GARCH, or of an \\texttt{arch} output table', '\\textbf{Explicații}: o a doua explicație a formei ARMA(1,1) a modelului GARCH sau a unui tabel de rezultate \\texttt{arch}'),
    T('\\textbf{Exploration}: volatility of all BVB stocks, or of the leu against several currencies', '\\textbf{Explorare}: volatilitatea tuturor acțiunilor de la BVB sau a leului față de mai multe valute'),
    (T('Example prompt', 'Exemplu de prompt'),
     [T('\\aiprompt{Write Python code that loads daily BET closes, computes log returns in percent, runs the ARCH-LM test with 5 lags, fits AR(1)-GARCH(1,1) with Student-t innovations using the arch package, and plots the annualised conditional volatility.}',
        '\\aiprompt{Write Python code that loads daily BET closes, computes log returns in percent, runs the ARCH-LM test with 5 lags, fits AR(1)-GARCH(1,1) with Student-t innovations using the arch package, and plots the annualised conditional volatility.}')])))

D.frame(T('Checks you must run', 'Verificări necesare'), items(
    T('Units: returns in \\%, not decimals; rescale a series with a very small variance (EUR/RON) and convert $\\omega$ back', 'Unitățile: randamente în \\%, nu în zecimale; rescalăm o serie cu varianță foarte mică (EUR/RON) și convertim înapoi $\\omega$'),
    T('The persistence: $\\alpha + \\beta$ for GARCH, $\\alpha + \\beta + \\gamma/2$ for GJR, $\\beta$ for EGARCH', 'Persistența: $\\alpha + \\beta$ pentru GARCH, $\\alpha + \\beta + \\gamma/2$ pentru GJR, $\\beta$ pentru EGARCH'),
    T('Boundary estimates ($\\alpha + \\beta = 1$): no half-life and no long-run volatility; do not report infinite numbers', 'Estimările pe frontieră ($\\alpha + \\beta = 1$): fără timp de înjumătățire și fără volatilitate de lungă durată; nu raportăm valori infinite'),
    T('Calendars: Bitcoin trades 7 days a week, the BET five; annualise each series with its own frequency', 'Calendarele: Bitcoin se tranzacționează 7 zile pe săptămînă, BET cinci; anualizăm fiecare serie cu propria frecvență'),
    T('Out-of-sample forecasts use only past data; the VaR level is written as VaR 1\\%', 'Prognozele în afara eșantionului folosesc doar date trecute; nivelul VaR se scrie VaR 1\\%'),
    T('Every cited reference: it must exist; check the DOI', 'Fiecare referință citată: trebuie să existe; verificați DOI-ul')))

# =============================================================================
# REZUMAT
# =============================================================================
D.section('Summary', 'Rezumat')

D.frame(T('Key takeaways', 'Idei de reținut'), items(
    T('Returns: an almost unpredictable mean and a predictable variance, $r_t = \\mu_t + \\sigma_t z_t$; test with the ACF of $r_t^2$ and ARCH-LM',
      'Randamentele: o medie aproape imprevizibilă și o varianță previzibilă, $r_t = \\mu_t + \\sigma_t z_t$; testăm cu ACF a lui $r_t^2$ și ARCH-LM'),
    T('GARCH(1,1) is an ARMA(1,1) for $\\varepsilon_t^2$; $\\alpha + \\beta$ close to 1; IGARCH and EWMA when $\\alpha + \\beta = 1$', 'GARCH(1,1) este un ARMA(1,1) pentru $\\varepsilon_t^2$; $\\alpha + \\beta$ apropiat de 1; IGARCH și EWMA cînd $\\alpha + \\beta = 1$'),
    T('Estimate by maximum likelihood with \\texttt{arch}, report robust SE; Student-t innovations for the tails', 'Estimăm prin verosimilitate maximă cu \\texttt{arch} și raportăm SE robuste; inovații Student-t pentru cozi'),
    T('ARMA-GARCH: the mean of Chapter 2 and the variance of this chapter estimated together', 'ARMA-GARCH: media din Capitolul 2 și varianța din acest capitol, estimate împreună'),
    T('Equity indices need asymmetry (GJR, EGARCH); check $\\hat z_t$ and $\\hat z_t^2$ after every fit', 'Indicii de acțiuni au nevoie de asimetrie (GJR, EGARCH); verificăm $\\hat z_t$ și $\\hat z_t^2$ după fiecare estimare'),
    T('Forecasts revert to the long-run level; compare them out of sample with QLIKE and DM; VaR 1\\% $= -(\\mu + \\sigma_{t+1}q_{0.01}(z))$',
      'Prognozele revin la nivelul de lungă durată; le comparăm în afara eșantionului cu QLIKE și DM; VaR 1\\% $= -(\\mu + \\sigma_{t+1}q_{0.01}(z))$')))

D.frame(T('Key formulas', 'Formule de reținut'), '{\\renewcommand{\\arraystretch}{1.4}' + table(
    'll', T('\\textbf{Quantity}', '\\textbf{Mărimea}') + ' & ' + T('\\textbf{Formula}', '\\textbf{Formula}'),
    ['ARCH-LM & $\\hat\\varepsilon_t^2 = b_0 + \\sum_{i=1}^{q}b_i\\hat\\varepsilon_{t-i}^2 + u_t$, \\quad $\\mathrm{LM} = nR^2 \\sim \\chi^2(q)$',
     'ARCH($q$) & $\\sigma_t^2 = \\omega + \\sum_{i=1}^{q}\\alpha_i\\varepsilon_{t-i}^2$, \\quad $\\bar\\sigma^2 = \\omega/(1 - \\sum\\alpha_i)$',
     'GARCH(1,1) & $\\sigma_t^2 = \\omega + \\alpha\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$, \\quad $\\bar\\sigma^2 = \\omega/(1 - \\alpha - \\beta)$, \\quad $h_{1/2} = \\ln 0.5/\\ln(\\alpha + \\beta)$',
     'EWMA & $\\sigma_t^2 = \\lambda\\sigma_{t-1}^2 + (1 - \\lambda)r_{t-1}^2$',
     'GJR, EGARCH & $\\sigma_t^2 = \\omega + (\\alpha + \\gamma I_{t-1})\\varepsilon_{t-1}^2 + \\beta\\sigma_{t-1}^2$; \\quad $\\ln\\sigma_t^2 = \\omega + \\alpha(|z_{t-1}| - E|z_{t-1}|) + \\gamma z_{t-1} + \\beta\\ln\\sigma_{t-1}^2$',
     T('Log-likelihood', 'Log-verosimilitatea') + ' & $\\ell = -\\frac12\\sum_t[\\ln 2\\pi + \\ln\\sigma_t^2 + (r_t - \\mu_t)^2/\\sigma_t^2]$',
     T('Forecast', 'Prognoza') + ' & $E_t[\\sigma_{t+h}^2] = \\bar\\sigma^2 + (\\alpha + \\beta)^{h-1}(\\sigma_{t+1}^2 - \\bar\\sigma^2)$',
     'QLIKE, VaR 1\\% & $L = r_t^2/h_t + \\ln h_t$; \\quad $\\mathrm{VaR}_{t+1} = -(\\mu + \\sigma_{t+1}\\,q_{0.01}(z))$'],
    size='scriptsize') + '}')

D.frame(T('Self-assessment', 'Autoevaluare'), items(
    (T('\\textbf{Question}: $\\omega = 0.05$, $\\alpha = 0.05$, $\\beta = 0.90$. What are the long-run variance and the half-life?', '\\textbf{Întrebare}: $\\omega = 0{,}05$, $\\alpha = 0{,}05$, $\\beta = 0{,}90$. Care sînt varianța de lungă durată și timpul de înjumătățire?'),
     [T('\\textbf{Answer}: $0.05/0.05 = 1$; $\\ln 0.5/\\ln 0.95 = 13.5$ days', '\\textbf{Răspuns}: $0{,}05/0{,}05 = 1$; $\\ln 0{,}5/\\ln 0{,}95 = 13{,}5$ zile')]),
    (T('\\textbf{Question}: an ARCH-LM regression with $q = 5$ on 1000 residuals gives $R^2 = 0.004$. Are there ARCH effects at 5\\%?', '\\textbf{Întrebare}: o regresie ARCH-LM cu $q = 5$ pe 1000 de reziduuri dă $R^2 = 0{,}004$. Există efecte ARCH la 5\\%?'),
     [T('\\textbf{Answer}: no: $\\mathrm{LM} = 1000 \\times 0.004 = 4 < 11.07$', '\\textbf{Răspuns}: nu: $\\mathrm{LM} = 1000 \\times 0{,}004 = 4 < 11{,}07$')]),
    (T('\\textbf{Question}: GJR gives $\\hat\\alpha = 0$ and $\\hat\\gamma = 0.2$. What happens to the variance after a rise of 3\\%?', '\\textbf{Întrebare}: GJR dă $\\hat\\alpha = 0$ și $\\hat\\gamma = 0{,}2$. Ce se întîmplă cu varianța după o creștere de 3\\%?'),
     [T('\\textbf{Answer}: nothing beyond the decay $\\beta\\sigma_t^2$: only negative shocks raise the variance', '\\textbf{Răspuns}: nimic în afară de scăderea $\\beta\\sigma_t^2$: doar șocurile negative cresc varianța')]),
    T('Next: Chapter 6, VAR models and Granger causality (several series at once)', 'Urmează: Capitolul 6, modele VAR și cauzalitate Granger (mai multe serii deodată)')))

D.references(bib())

if __name__ == '__main__':
    finalize(D.write(V))
