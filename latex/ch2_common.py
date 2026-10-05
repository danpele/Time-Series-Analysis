r"""
ch2_common.py -- shared helpers of the Chapter 2 generators (lecture and seminar), TSA
=====================================================================================
Numbers from Quantlets/Ch_02/ch2_numbers.json (generate_all_charts.py) and sem2_results.json (seminar2.py);
the clickable citations of Chapter 2 (DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_02')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_02'
MONTHS_RO = ['ianuarie', 'februarie', 'martie', 'aprilie', 'mai', 'iunie', 'iulie', 'august', 'septembrie',
             'octombrie', 'noiembrie', 'decembrie']
MONTHS_EN = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
             'November', 'December']


def T(en, ro):
    """Bilingual text."""
    return f'⟦{en}||{ro}⟧'


def V2(en, ro):
    """A bilingual value that can sit inside ⟦..||..⟧ text: resolved in the written files by finalize()."""
    return f'⟪{en}¦{ro}⟫'


def finalize(paths):
    """Resolve the ⟪en¦ro⟫ values in the written decks (EN file first, RO file second)."""
    import re
    for p, k in zip(paths, (1, 2)):
        s = open(p, encoding='utf-8').read()
        s = re.sub(r'⟪(.*?)¦(.*?)⟫', lambda m: m.group(k), s)
        open(p, 'w', encoding='utf-8').write(s)


def date(s, day=True):
    """'2026-09-18' -> ⟦18 September 2026||18 septembrie 2026⟧ (or month and year only)."""
    y, m, d = s[:10].split('-')
    m = int(m) - 1
    if day:
        return V2(f'{int(d)} {MONTHS_EN[m]} {y}', f'{int(d)} {MONTHS_RO[m]} {y}')
    return V2(f'{MONTHS_EN[m]} {y}', f'{MONTHS_RO[m]} {y}')


def quarter(s):
    """'2026-04-01' -> 2026 Q2 / T2 2026."""
    y, m = int(s[:4]), int(s[5:7])
    q = (m - 1) // 3 + 1
    return V2(f'{y}Q{q}', f'T{q} {y}')


def pv(p, d=3):
    """A p-value in text mode: 3 decimals, or '< 0.001' (marked for the RO decimal comma)."""
    return '$<$\\,⁅0.001⁆' if p < 0.001 else '⁅' + f'{p:.{d}f}' + '⁆'


def load():
    with open(os.path.join(QL, 'ch2_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem2_results.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('FPP', 'https://otexts.com/fpp3/arima.html', 'Hyndman and Athanasopoulos (2021)', 'Hyndman și Athanasopoulos (2021)')
        + ref('BD', D_ + '10.1007/978-3-319-29854-2', 'Brockwell and Davis (2016)', 'Brockwell și Davis (2016)')
        + ref('BDtm', D_ + '10.1007/978-1-4419-0320-4', 'Brockwell and Davis (1991)', 'Brockwell și Davis (1991)')
        + ref('Hamilton', D_ + '10.2307/j.ctv14jx6sm', 'Hamilton (1994)')
        + ref('BJ', D_ + '10.1002/9781118619193', 'Box, Jenkins and Reinsel (2008)', 'Box, Jenkins și Reinsel (2008)')
        + ref('BP', D_ + '10.1080/01621459.1970.10481180', 'Box and Pierce (1970)', 'Box și Pierce (1970)')
        + ref('LB', D_ + '10.1093/biomet/65.2.297', 'Ljung and Box (1978)', 'Ljung și Box (1978)')
        + ref('Yule', D_ + '10.1098/rsta.1927.0007', 'Yule (1927)')
        + ref('Walker', D_ + '10.1098/rspa.1931.0069', 'Walker (1931)')
        + ref('Slutzky', D_ + '10.2307/1907241', 'Slutzky (1937)')
        + ref('Wold', 'https://archive.org/details/in.ernet.dli.2015.262214', 'Wold (1938)')
        + ref('Durbin', D_ + '10.2307/1401322', 'Durbin (1960)')
        + ref('Akaike', D_ + '10.1109/TAC.1974.1100705', 'Akaike (1974)')
        + ref('Schwarz', D_ + '10.1214/aos/1176344136', 'Schwarz (1978)')
        + ref('HT', D_ + '10.1093/biomet/76.2.297', 'Hurvich and Tsai (1989)', 'Hurvich și Tsai (1989)')
        + ref('JB', D_ + '10.2307/1403192', 'Jarque and Bera (1987)', 'Jarque și Bera (1987)')
        + ref('HK', D_ + '10.18637/jss.v027.i03', 'Hyndman and Khandakar (2008)', 'Hyndman și Khandakar (2008)')
        + ref('MH', D_ + '10.1016/S0169-2070(00)00057-1', 'Makridakis and Hibon (2000)', 'Makridakis și Hibon (2000)'))

BIB = {
    'Akaike': r'Akaike, H. (1974). A new look at the statistical model identification. \textit{IEEE Transactions on Automatic Control}, 19(6), 716--723. \href{https://doi.org/10.1109/TAC.1974.1100705}{doi:10.1109/TAC.1974.1100705}',
    'BJ': r'Box, G. E. P., Jenkins, G. M., \& Reinsel, G. C. (2008). \textit{Time Series Analysis: Forecasting and Control} (4th ed.). Wiley. \href{https://doi.org/10.1002/9781118619193}{doi:10.1002/9781118619193}',
    'BP': r'Box, G. E. P., \& Pierce, D. A. (1970). Distribution of residual autocorrelations in autoregressive-integrated moving average time series models. \textit{Journal of the American Statistical Association}, 65(332), 1509--1526. \href{https://doi.org/10.1080/01621459.1970.10481180}{doi:10.1080/01621459.1970.10481180}',
    'BDtm': r'Brockwell, P. J., \& Davis, R. A. (1991). \textit{Time Series: Theory and Methods} (2nd ed.). Springer. \href{https://doi.org/10.1007/978-1-4419-0320-4}{doi:10.1007/978-1-4419-0320-4}',
    'BD': r'Brockwell, P. J., \& Davis, R. A. (2016). \textit{Introduction to Time Series and Forecasting} (3rd ed.). Springer. \href{https://doi.org/10.1007/978-3-319-29854-2}{doi:10.1007/978-3-319-29854-2}',
    'Durbin': r'Durbin, J. (1960). The fitting of time-series models. \textit{Revue de l\'Institut International de Statistique}, 28(3), 233--244. \href{https://doi.org/10.2307/1401322}{doi:10.2307/1401322}',
    'Hamilton': r'Hamilton, J. D. (1994). \textit{Time Series Analysis}. Princeton University Press. \href{https://doi.org/10.2307/j.ctv14jx6sm}{doi:10.2307/j.ctv14jx6sm}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'HT': r'Hurvich, C. M., \& Tsai, C.-L. (1989). Regression and time series model selection in small samples. \textit{Biometrika}, 76(2), 297--307. \href{https://doi.org/10.1093/biomet/76.2.297}{doi:10.1093/biomet/76.2.297}',
    'FPP': r'Hyndman, R. J., \& Athanasopoulos, G. (2021). \textit{Forecasting: Principles and Practice} (3rd ed.). OTexts. \href{https://otexts.com/fpp3/}{otexts.com/fpp3}',
    'HK': r'Hyndman, R. J., \& Khandakar, Y. (2008). Automatic time series forecasting: The forecast package for R. \textit{Journal of Statistical Software}, 27(3), 1--22. \href{https://doi.org/10.18637/jss.v027.i03}{doi:10.18637/jss.v027.i03}',
    'JB': r'Jarque, C. M., \& Bera, A. K. (1987). A test for normality of observations and regression residuals. \textit{International Statistical Review}, 55(2), 163--172. \href{https://doi.org/10.2307/1403192}{doi:10.2307/1403192}',
    'LB': r'Ljung, G. M., \& Box, G. E. P. (1978). On a measure of lack of fit in time series models. \textit{Biometrika}, 65(2), 297--303. \href{https://doi.org/10.1093/biomet/65.2.297}{doi:10.1093/biomet/65.2.297}',
    'MH': r'Makridakis, S., \& Hibon, M. (2000). The M3-Competition: Results, conclusions and implications. \textit{International Journal of Forecasting}, 16(4), 451--476. \href{https://doi.org/10.1016/S0169-2070(00)00057-1}{doi:10.1016/S0169-2070(00)00057-1}',
    'Schwarz': r'Schwarz, G. (1978). Estimating the dimension of a model. \textit{The Annals of Statistics}, 6(2), 461--464. \href{https://doi.org/10.1214/aos/1176344136}{doi:10.1214/aos/1176344136}',
    'Slutzky': r'Slutzky, E. (1937). The summation of random causes as the source of cyclic processes. \textit{Econometrica}, 5(2), 105--146. \href{https://doi.org/10.2307/1907241}{doi:10.2307/1907241}',
    'Walker': r'Walker, G. (1931). On periodicity in series of related terms. \textit{Proceedings of the Royal Society of London, Series A}, 131(818), 518--532. \href{https://doi.org/10.1098/rspa.1931.0069}{doi:10.1098/rspa.1931.0069}',
    'Wold': r'Wold, H. (1938). \textit{A Study in the Analysis of Stationary Time Series}. Almqvist \& Wiksell. \href{https://archive.org/details/in.ernet.dli.2015.262214}{archive.org}',
    'Yule': r"Yule, G. U. (1927). On a method of investigating periodicities in disturbed series, with special reference to Wolfer's sunspot numbers. \textit{Philosophical Transactions of the Royal Society A}, 226, 267--298. \href{https://doi.org/10.1098/rsta.1927.0007}{doi:10.1098/rsta.1927.0007}",
}
ORDER = ['Akaike', 'BJ', 'BP', 'BDtm', 'BD', 'Durbin', 'Hamilton', 'HP', 'HT', 'FPP', 'HK', 'JB', 'LB', 'MH', 'Schwarz',
         'Slutzky', 'Walker', 'Wold', 'Yule']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
