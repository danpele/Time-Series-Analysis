r"""
ch1_common.py -- shared helpers of the Chapter 1 generators (lecture and seminar), TSA
=====================================================================================
Numbers from Quantlets/Ch_01/ch1_numbers.json (generate_all_charts.py) and sem1_results.json (seminar1.py);
the clickable citations of Chapter 1 (DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_01')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_01'
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
    with open(os.path.join(QL, 'ch1_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem1_results.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('FPP', 'https://otexts.com/fpp3/', 'Hyndman and Athanasopoulos (2021)', 'Hyndman și Athanasopoulos (2021)')
        + ref('BD', D_ + '10.1007/978-3-319-29854-2', 'Brockwell and Davis (2016)', 'Brockwell și Davis (2016)')
        + ref('BDtm', D_ + '10.1007/978-1-4419-0320-4', 'Brockwell and Davis (1991)', 'Brockwell și Davis (1991)')
        + ref('Hamilton', D_ + '10.2307/j.ctv14jx6sm', 'Hamilton (1994)')
        + ref('SS', D_ + '10.1007/978-3-319-52452-8', 'Shumway and Stoffer (2017)', 'Shumway și Stoffer (2017)')
        + ref('BP', D_ + '10.1080/01621459.1970.10481180', 'Box and Pierce (1970)', 'Box și Pierce (1970)')
        + ref('LB', D_ + '10.1093/biomet/65.2.297', 'Ljung and Box (1978)', 'Ljung și Box (1978)')
        + ref('BC', D_ + '10.1111/j.2517-6161.1964.tb00553.x', 'Box and Cox (1964)', 'Box și Cox (1964)')
        + ref('Bartlett', D_ + '10.2307/2983611', 'Bartlett (1946)')
        + ref('Yule', D_ + '10.1098/rsta.1927.0007', 'Yule (1927)')
        + ref('Slutzky', D_ + '10.2307/1907241', 'Slutzky (1937)')
        + ref('Guerrero', D_ + '10.1002/for.3980120104', 'Guerrero (1993)')
        + ref('Khinchin', D_ + '10.1007/BF01449156', 'Khintchine (1934)')
        + ref('Wold', 'https://archive.org/details/in.ernet.dli.2015.262214', 'Wold (1938)')
        + ref('Cobb', D_ + '10.1093/biomet/65.2.243', 'Cobb (1978)'))

BIB = {
    'Bartlett': r'Bartlett, M. S. (1946). On the theoretical specification and sampling properties of autocorrelated time-series. \textit{Supplement to the Journal of the Royal Statistical Society}, 8(1), 27--41. \href{https://doi.org/10.2307/2983611}{doi:10.2307/2983611}',
    'BC': r'Box, G. E. P., \& Cox, D. R. (1964). An analysis of transformations. \textit{Journal of the Royal Statistical Society: Series B}, 26(2), 211--243. \href{https://doi.org/10.1111/j.2517-6161.1964.tb00553.x}{doi:10.1111/j.2517-6161.1964.tb00553.x}',
    'BP': r'Box, G. E. P., \& Pierce, D. A. (1970). Distribution of residual autocorrelations in autoregressive-integrated moving average time series models. \textit{Journal of the American Statistical Association}, 65(332), 1509--1526. \href{https://doi.org/10.1080/01621459.1970.10481180}{doi:10.1080/01621459.1970.10481180}',
    'BDtm': r'Brockwell, P. J., \& Davis, R. A. (1991). \textit{Time Series: Theory and Methods} (2nd ed.). Springer. \href{https://doi.org/10.1007/978-1-4419-0320-4}{doi:10.1007/978-1-4419-0320-4}',
    'BD': r'Brockwell, P. J., \& Davis, R. A. (2016). \textit{Introduction to Time Series and Forecasting} (3rd ed.). Springer. \href{https://doi.org/10.1007/978-3-319-29854-2}{doi:10.1007/978-3-319-29854-2}',
    'Cobb': r'Cobb, G. W. (1978). The problem of the Nile: Conditional solution to a changepoint problem. \textit{Biometrika}, 65(2), 243--251. \href{https://doi.org/10.1093/biomet/65.2.243}{doi:10.1093/biomet/65.2.243}',
    'Guerrero': r'Guerrero, V. M. (1993). Time-series analysis supported by power transformations. \textit{Journal of Forecasting}, 12(1), 37--48. \href{https://doi.org/10.1002/for.3980120104}{doi:10.1002/for.3980120104}',
    'Hamilton': r'Hamilton, J. D. (1994). \textit{Time Series Analysis}. Princeton University Press. \href{https://doi.org/10.2307/j.ctv14jx6sm}{doi:10.2307/j.ctv14jx6sm}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'FPP': r'Hyndman, R. J., \& Athanasopoulos, G. (2021). \textit{Forecasting: Principles and Practice} (3rd ed.). OTexts. \href{https://otexts.com/fpp3/}{otexts.com/fpp3}',
    'Khinchin': r'Khintchine, A. (1934). Korrelationstheorie der stationären stochastischen Prozesse. \textit{Mathematische Annalen}, 109, 604--615. \href{https://doi.org/10.1007/BF01449156}{doi:10.1007/BF01449156}',
    'LB': r'Ljung, G. M., \& Box, G. E. P. (1978). On a measure of lack of fit in time series models. \textit{Biometrika}, 65(2), 297--303. \href{https://doi.org/10.1093/biomet/65.2.297}{doi:10.1093/biomet/65.2.297}',
    'SS': r'Shumway, R. H., \& Stoffer, D. S. (2017). \textit{Time Series Analysis and Its Applications} (4th ed.). Springer. \href{https://doi.org/10.1007/978-3-319-52452-8}{doi:10.1007/978-3-319-52452-8}',
    'Slutzky': r'Slutzky, E. (1937). The summation of random causes as the source of cyclic processes. \textit{Econometrica}, 5(2), 105--146. \href{https://doi.org/10.2307/1907241}{doi:10.2307/1907241}',
    'Wold': r'Wold, H. (1938). \textit{A Study in the Analysis of Stationary Time Series}. Almqvist \& Wiksell. \href{https://archive.org/details/in.ernet.dli.2015.262214}{archive.org}',
    'Yule': r"Yule, G. U. (1927). On a method of investigating periodicities in disturbed series, with special reference to Wolfer's sunspot numbers. \textit{Philosophical Transactions of the Royal Society A}, 226, 267--298. \href{https://doi.org/10.1098/rsta.1927.0007}{doi:10.1098/rsta.1927.0007}",
}
ORDER = ['Bartlett', 'BC', 'BP', 'BDtm', 'BD', 'Cobb', 'Guerrero', 'Hamilton', 'HP', 'FPP', 'Khinchin', 'LB', 'SS',
         'Slutzky', 'Wold', 'Yule']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
