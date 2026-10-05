r"""
ch3_common.py -- shared helpers of the Chapter 3 generators (lecture and seminar), TSA
=====================================================================================
Numbers from Quantlets/Ch_03/ch3_numbers.json (generate_all_charts.py) and sem3_results.json (seminar3.py);
the clickable citations of Chapter 3 (DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401
from ch1_common import T, V2, finalize, date, quarter, pv, MONTHS_EN, MONTHS_RO   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_03')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_03'


def month(s):
    """'2026-08' -> ⟦August 2026||august 2026⟧."""
    y, m = int(s[:4]), int(s[5:7]) - 1
    return V2(f'{MONTHS_EN[m]} {y}', f'{MONTHS_RO[m]} {y}')


def qtr(s):
    """'2026Q2' -> 2026Q2 / T2 2026."""
    y, q = s[:4], s[-1]
    return V2(f'{y}Q{q}', f'T{q} {y}')


def load():
    with open(os.path.join(QL, 'ch3_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem3_results.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('FPP', 'https://otexts.com/fpp3/', 'Hyndman and Athanasopoulos (2021)', 'Hyndman și Athanasopoulos (2021)')
        + ref('BD', D_ + '10.1007/978-3-319-29854-2', 'Brockwell and Davis (2016)', 'Brockwell și Davis (2016)')
        + ref('Hamilton', D_ + '10.2307/j.ctv14jx6sm', 'Hamilton (1994)')
        + ref('BJ', D_ + '10.1002/9781118619193', 'Box, Jenkins and Reinsel (2008)', 'Box, Jenkins și Reinsel (2008)')
        + ref('DF', D_ + '10.1080/01621459.1979.10482531', 'Dickey and Fuller (1979)', 'Dickey și Fuller (1979)')
        + ref('DFb', D_ + '10.2307/1912517', 'Dickey and Fuller (1981)', 'Dickey și Fuller (1981)')
        + ref('Fuller', D_ + '10.1002/9780470316917', 'Fuller (1996)')
        + ref('SD', D_ + '10.1093/biomet/71.3.599', 'Said and Dickey (1984)', 'Said și Dickey (1984)')
        + ref('PP', D_ + '10.1093/biomet/75.2.335', 'Phillips and Perron (1988)', 'Phillips și Perron (1988)')
        + ref('KPSS', D_ + '10.1016/0304-4076(92)90104-Y', 'Kwiatkowski et al.\\ (1992)')
        + ref('MacKinnon', D_ + '10.1080/07350015.1994.10510005', 'MacKinnon (1994)')
        + ref('MacKinnonb', 'https://ideas.repec.org/p/qed/wpaper/1227.html', 'MacKinnon (2010)')
        + ref('GN', D_ + '10.1016/0304-4076(74)90034-7', 'Granger and Newbold (1974)', 'Granger și Newbold (1974)')
        + ref('Phillips', D_ + '10.1016/0304-4076(86)90001-1', 'Phillips (1986)')
        + ref('Yule', D_ + '10.2307/2341482', 'Yule (1926)')
        + ref('NP', D_ + '10.1016/0304-3932(82)90012-5', 'Nelson and Plosser (1982)', 'Nelson și Plosser (1982)')
        + ref('NK', D_ + '10.2307/1911520', 'Nelson and Kang (1981)', 'Nelson și Kang (1981)')
        + ref('Perron', D_ + '10.2307/1913712', 'Perron (1989)')
        + ref('ZA', D_ + '10.1080/07350015.1992.10509904', 'Zivot and Andrews (1992)', 'Zivot și Andrews (1992)')
        + ref('BaiP', D_ + '10.2307/2998540', 'Bai and Perron (1998)', 'Bai și Perron (1998)')
        + ref('Schwert', D_ + '10.1080/07350015.1989.10509723', 'Schwert (1989)')
        + ref('NgP', D_ + '10.1111/1468-0262.00256', 'Ng and Perron (2001)', 'Ng și Perron (2001)')
        + ref('ERS', D_ + '10.2307/2171846', 'Elliott, Rothenberg and Stock (1996)', 'Elliott, Rothenberg și Stock (1996)')
        + ref('DP', D_ + '10.1080/07350015.1987.10509614', 'Dickey and Pantula (1987)', 'Dickey și Pantula (1987)')
        + ref('PS', D_ + '10.1016/0304-4076(77)90015-x', 'Plosser and Schwert (1977)', 'Plosser și Schwert (1977)')
        + ref('HK', D_ + '10.18637/jss.v027.i03', 'Hyndman and Khandakar (2008)', 'Hyndman și Khandakar (2008)')
        + ref('MR', D_ + '10.1016/0022-1996(83)90017-X', 'Meese and Rogoff (1983)', 'Meese și Rogoff (1983)')
        + ref('EG', D_ + '10.2307/1913236', 'Engle and Granger (1987)', 'Engle și Granger (1987)')
        + ref('LB', D_ + '10.1093/biomet/65.2.297', 'Ljung and Box (1978)', 'Ljung și Box (1978)'))

BIB = {
    'BaiP': r'Bai, J., \& Perron, P. (1998). Estimating and testing linear models with multiple structural changes. \textit{Econometrica}, 66(1), 47--78. \href{https://doi.org/10.2307/2998540}{doi:10.2307/2998540}',
    'BJ': r'Box, G. E. P., Jenkins, G. M., \& Reinsel, G. C. (2008). \textit{Time Series Analysis: Forecasting and Control} (4th ed.). Wiley. \href{https://doi.org/10.1002/9781118619193}{doi:10.1002/9781118619193}',
    'BD': r'Brockwell, P. J., \& Davis, R. A. (2016). \textit{Introduction to Time Series and Forecasting} (3rd ed.). Springer. \href{https://doi.org/10.1007/978-3-319-29854-2}{doi:10.1007/978-3-319-29854-2}',
    'DF': r'Dickey, D. A., \& Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. \textit{Journal of the American Statistical Association}, 74(366), 427--431. \href{https://doi.org/10.1080/01621459.1979.10482531}{doi:10.1080/01621459.1979.10482531}',
    'DFb': r'Dickey, D. A., \& Fuller, W. A. (1981). Likelihood ratio statistics for autoregressive time series with a unit root. \textit{Econometrica}, 49(4), 1057--1072. \href{https://doi.org/10.2307/1912517}{doi:10.2307/1912517}',
    'DP': r'Dickey, D. A., \& Pantula, S. G. (1987). Determining the order of differencing in autoregressive processes. \textit{Journal of Business \& Economic Statistics}, 5(4), 455--461. \href{https://doi.org/10.1080/07350015.1987.10509614}{doi:10.1080/07350015.1987.10509614}',
    'ERS': r'Elliott, G., Rothenberg, T. J., \& Stock, J. H. (1996). Efficient tests for an autoregressive unit root. \textit{Econometrica}, 64(4), 813--836. \href{https://doi.org/10.2307/2171846}{doi:10.2307/2171846}',
    'EG': r'Engle, R. F., \& Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. \textit{Econometrica}, 55(2), 251--276. \href{https://doi.org/10.2307/1913236}{doi:10.2307/1913236}',
    'Fuller': r'Fuller, W. A. (1996). \textit{Introduction to Statistical Time Series} (2nd ed.). Wiley. \href{https://doi.org/10.1002/9780470316917}{doi:10.1002/9780470316917}',
    'GN': r'Granger, C. W. J., \& Newbold, P. (1974). Spurious regressions in econometrics. \textit{Journal of Econometrics}, 2(2), 111--120. \href{https://doi.org/10.1016/0304-4076(74)90034-7}{doi:10.1016/0304-4076(74)90034-7}',
    'Hamilton': r'Hamilton, J. D. (1994). \textit{Time Series Analysis}. Princeton University Press. \href{https://doi.org/10.2307/j.ctv14jx6sm}{doi:10.2307/j.ctv14jx6sm}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'FPP': r'Hyndman, R. J., \& Athanasopoulos, G. (2021). \textit{Forecasting: Principles and Practice} (3rd ed.). OTexts. \href{https://otexts.com/fpp3/}{otexts.com/fpp3}',
    'HK': r'Hyndman, R. J., \& Khandakar, Y. (2008). Automatic time series forecasting: The forecast package for R. \textit{Journal of Statistical Software}, 27(3), 1--22. \href{https://doi.org/10.18637/jss.v027.i03}{doi:10.18637/jss.v027.i03}',
    'KPSS': r'Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., \& Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. \textit{Journal of Econometrics}, 54(1--3), 159--178. \href{https://doi.org/10.1016/0304-4076(92)90104-Y}{doi:10.1016/0304-4076(92)90104-Y}',
    'LB': r'Ljung, G. M., \& Box, G. E. P. (1978). On a measure of lack of fit in time series models. \textit{Biometrika}, 65(2), 297--303. \href{https://doi.org/10.1093/biomet/65.2.297}{doi:10.1093/biomet/65.2.297}',
    'MacKinnon': r'MacKinnon, J. G. (1994). Approximate asymptotic distribution functions for unit-root and cointegration tests. \textit{Journal of Business \& Economic Statistics}, 12(2), 167--176. \href{https://doi.org/10.1080/07350015.1994.10510005}{doi:10.1080/07350015.1994.10510005}',
    'MacKinnonb': r"MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's Economics Department Working Paper No.~1227, Queen's University. \href{https://ideas.repec.org/p/qed/wpaper/1227.html}{ideas.repec.org/p/qed/wpaper/1227}",
    'MR': r'Meese, R. A., \& Rogoff, K. (1983). Empirical exchange rate models of the seventies: Do they fit out of sample? \textit{Journal of International Economics}, 14(1--2), 3--24. \href{https://doi.org/10.1016/0022-1996(83)90017-X}{doi:10.1016/0022-1996(83)90017-X}',
    'NK': r'Nelson, C. R., \& Kang, H. (1981). Spurious periodicity in inappropriately detrended time series. \textit{Econometrica}, 49(3), 741--751. \href{https://doi.org/10.2307/1911520}{doi:10.2307/1911520}',
    'NP': r'Nelson, C. R., \& Plosser, C. I. (1982). Trends and random walks in macroeconomic time series: Some evidence and implications. \textit{Journal of Monetary Economics}, 10(2), 139--162. \href{https://doi.org/10.1016/0304-3932(82)90012-5}{doi:10.1016/0304-3932(82)90012-5}',
    'NgP': r'Ng, S., \& Perron, P. (2001). Lag length selection and the construction of unit root tests with good size and power. \textit{Econometrica}, 69(6), 1519--1554. \href{https://doi.org/10.1111/1468-0262.00256}{doi:10.1111/1468-0262.00256}',
    'Perron': r'Perron, P. (1989). The Great Crash, the oil price shock, and the unit root hypothesis. \textit{Econometrica}, 57(6), 1361--1401. \href{https://doi.org/10.2307/1913712}{doi:10.2307/1913712}',
    'Phillips': r'Phillips, P. C. B. (1986). Understanding spurious regressions in econometrics. \textit{Journal of Econometrics}, 33(3), 311--340. \href{https://doi.org/10.1016/0304-4076(86)90001-1}{doi:10.1016/0304-4076(86)90001-1}',
    'PP': r'Phillips, P. C. B., \& Perron, P. (1988). Testing for a unit root in time series regression. \textit{Biometrika}, 75(2), 335--346. \href{https://doi.org/10.1093/biomet/75.2.335}{doi:10.1093/biomet/75.2.335}',
    'PS': r'Plosser, C. I., \& Schwert, G. W. (1977). Estimation of a non-invertible moving average process: The case of overdifferencing. \textit{Journal of Econometrics}, 6(2), 199--224. \href{https://doi.org/10.1016/0304-4076(77)90015-x}{doi:10.1016/0304-4076(77)90015-x}',
    'SD': r'Said, S. E., \& Dickey, D. A. (1984). Testing for unit roots in autoregressive-moving average models of unknown order. \textit{Biometrika}, 71(3), 599--607. \href{https://doi.org/10.1093/biomet/71.3.599}{doi:10.1093/biomet/71.3.599}',
    'Schwert': r'Schwert, G. W. (1989). Tests for unit roots: A Monte Carlo investigation. \textit{Journal of Business \& Economic Statistics}, 7(2), 147--159. \href{https://doi.org/10.1080/07350015.1989.10509723}{doi:10.1080/07350015.1989.10509723}',
    'Yule': r'Yule, G. U. (1926). Why do we sometimes get nonsense-correlations between time-series? A study in sampling and the nature of time-series. \textit{Journal of the Royal Statistical Society}, 89(1), 1--63. \href{https://doi.org/10.2307/2341482}{doi:10.2307/2341482}',
    'ZA': r'Zivot, E., \& Andrews, D. W. K. (1992). Further evidence on the Great Crash, the oil-price shock, and the unit-root hypothesis. \textit{Journal of Business \& Economic Statistics}, 10(3), 251--270. \href{https://doi.org/10.1080/07350015.1992.10509904}{doi:10.1080/07350015.1992.10509904}',
}
ORDER = ['BaiP', 'BJ', 'BD', 'DF', 'DFb', 'DP', 'ERS', 'EG', 'Fuller', 'GN', 'Hamilton', 'HP', 'FPP', 'HK', 'KPSS', 'LB',
         'MacKinnon', 'MacKinnonb', 'MR', 'NK', 'NP', 'NgP', 'Perron', 'Phillips', 'PP', 'PS', 'SD', 'Schwert', 'Yule', 'ZA']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
