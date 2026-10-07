r"""
ch7_common.py -- shared helpers of the Chapter 7 generators (lecture and seminar), TSA
=====================================================================================
Numbers from Quantlets/Ch_07/ch7_numbers.json (generate_all_charts.py) and sem7_results.json (seminar7.py);
the clickable citations of Chapter 7 (DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401
from ch1_common import T, V2, date, quarter, pv, MONTHS_EN, MONTHS_RO   # noqa: E402,F401
from ch6_common import finalize   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_07')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_07'


def month(s):
    """'2026-08' -> ⟪August 2026¦august 2026⟫."""
    y, m = int(s[:4]), int(s[5:7]) - 1
    return V2(f'{MONTHS_EN[m]} {y}', f'{MONTHS_RO[m]} {y}')


def qtr(s):
    """'2009Q3' -> 2009Q3 / T3 2009."""
    y, q = s[:4], s[-1]
    return V2(f'{y}Q{q}', f'T{q} {y}')


def load():
    with open(os.path.join(QL, 'ch7_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem7_results.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('Hamilton', D_ + '10.2307/j.ctv14jx6sm', 'Hamilton (1994)')
        + ref('Lut', D_ + '10.1007/978-3-540-27752-1', 'Lütkepohl (2005)')
        + ref('JohBook', D_ + '10.1093/0198774508.001.0001', 'Johansen (1995)')
        + ref('Jus', D_ + '10.1093/oso/9780199285662.001.0001', 'Juselius (2006)')
        + ref('EG', D_ + '10.2307/1913236', 'Engle and Granger (1987)', 'Engle și Granger (1987)')
        + ref('Granger', D_ + '10.1016/0304-4076(81)90079-8', 'Granger (1981)')
        + ref('GrangerN', D_ + '10.1257/0002828041464669', 'Granger (2004)')
        + ref('GN', D_ + '10.1016/0304-4076(74)90034-7', 'Granger and Newbold (1974)', 'Granger și Newbold (1974)')
        + ref('Murray', D_ + '10.1080/00031305.1994.10476017', 'Murray (1994)')
        + ref('PO', D_ + '10.2307/2938339', 'Phillips and Ouliaris (1990)', 'Phillips și Ouliaris (1990)')
        + ref('MacKinnon', D_ + '10.1080/07350015.1994.10510005', 'MacKinnon (1994)')
        + ref('MacKinnonb', 'https://ideas.repec.org/p/qed/wpaper/1227.html', 'MacKinnon (2010)')
        + ref('SW', D_ + '10.1080/01621459.1988.10478707', 'Stock and Watson (1988)', 'Stock și Watson (1988)')
        + ref('SWd', D_ + '10.2307/2951763', 'Stock and Watson (1993)', 'Stock și Watson (1993)')
        + ref('DHSY', D_ + '10.2307/2231972', 'Davidson et al.\\ (1978)')
        + ref('BDM', D_ + '10.1111/1467-9892.00091', 'Banerjee, Dolado and Mestre (1998)', 'Banerjee, Dolado și Mestre (1998)')
        + ref('JohA', D_ + '10.1016/0165-1889(88)90041-3', 'Johansen (1988)')
        + ref('JohB', D_ + '10.2307/2938278', 'Johansen (1991)')
        + ref('JJ', D_ + '10.1111/j.1468-0084.1990.mp52002003.x', 'Johansen and Juselius (1990)', 'Johansen și Juselius (1990)')
        + ref('OL', D_ + '10.1111/j.1468-0084.1992.tb00013.x', 'Osterwald-Lenum (1992)')
        + ref('MHM', D_ + '10.1002/(SICI)1099-1255(199909/10)14:5<563::AID-JAE530>3.0.CO;2-R', 'MacKinnon, Haug and Michelis (1999)', 'MacKinnon, Haug și Michelis (1999)')
        + ref('RA', D_ + '10.1111/j.1467-9892.1992.tb00113.x', 'Reinsel and Ahn (1992)', 'Reinsel și Ahn (1992)')
        + ref('EY', D_ + '10.1016/0304-4076(87)90085-6', 'Engle and Yoo (1987)', 'Engle și Yoo (1987)')
        + ref('CD', D_ + '10.1080/07350015.1998.10524784', 'Christoffersen and Diebold (1998)', 'Christoffersen și Diebold (1998)')
        + ref('CS', D_ + '10.1086/261502', 'Campbell and Shiller (1987)', 'Campbell și Shiller (1987)')
        + ref('Balassa', D_ + '10.1086/258965', 'Balassa (1964)')
        + ref('TT', D_ + '10.1257/0895330042632744', 'Taylor and Taylor (2004)', 'Taylor și Taylor (2004)')
        + ref('GGR', D_ + '10.1093/rfs/hhj020', 'Gatev, Goetzmann and Rouwenhorst (2006)', 'Gatev, Goetzmann și Rouwenhorst (2006)')
        + ref('DoFaff', D_ + '10.2469/faj.v66.n4.1', 'Do and Faff (2010)', 'Do și Faff (2010)'))

BIB = {
    'Balassa': r'Balassa, B. (1964). The purchasing-power parity doctrine: A reappraisal. \textit{Journal of Political Economy}, 72(6), 584--596. \href{https://doi.org/10.1086/258965}{doi:10.1086/258965}',
    'BDM': r'Banerjee, A., Dolado, J. J., \& Mestre, R. (1998). Error-correction mechanism tests for cointegration in a single-equation framework. \textit{Journal of Time Series Analysis}, 19(3), 267--283. \href{https://doi.org/10.1111/1467-9892.00091}{doi:10.1111/1467-9892.00091}',
    'CS': r'Campbell, J. Y., \& Shiller, R. J. (1987). Cointegration and tests of present value models. \textit{Journal of Political Economy}, 95(5), 1062--1088. \href{https://doi.org/10.1086/261502}{doi:10.1086/261502}',
    'CD': r'Christoffersen, P. F., \& Diebold, F. X. (1998). Cointegration and long-horizon forecasting. \textit{Journal of Business \& Economic Statistics}, 16(4), 450--456. \href{https://doi.org/10.1080/07350015.1998.10524784}{doi:10.1080/07350015.1998.10524784}',
    'DHSY': r"Davidson, J. E. H., Hendry, D. F., Srba, F., \& Yeo, S. (1978). Econometric modelling of the aggregate time-series relationship between consumers' expenditure and income in the United Kingdom. \textit{The Economic Journal}, 88(352), 661--692. \href{https://doi.org/10.2307/2231972}{doi:10.2307/2231972}",
    'DoFaff': r'Do, B., \& Faff, R. (2010). Does simple pairs trading still work? \textit{Financial Analysts Journal}, 66(4), 83--95. \href{https://doi.org/10.2469/faj.v66.n4.1}{doi:10.2469/faj.v66.n4.1}',
    'EG': r'Engle, R. F., \& Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. \textit{Econometrica}, 55(2), 251--276. \href{https://doi.org/10.2307/1913236}{doi:10.2307/1913236}',
    'EY': r'Engle, R. F., \& Yoo, B. S. (1987). Forecasting and testing in co-integrated systems. \textit{Journal of Econometrics}, 35(1), 143--159. \href{https://doi.org/10.1016/0304-4076(87)90085-6}{doi:10.1016/0304-4076(87)90085-6}',
    'GGR': r'Gatev, E., Goetzmann, W. N., \& Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. \textit{Review of Financial Studies}, 19(3), 797--827. \href{https://doi.org/10.1093/rfs/hhj020}{doi:10.1093/rfs/hhj020}',
    'Granger': r'Granger, C. W. J. (1981). Some properties of time series data and their use in econometric model specification. \textit{Journal of Econometrics}, 16(1), 121--130. \href{https://doi.org/10.1016/0304-4076(81)90079-8}{doi:10.1016/0304-4076(81)90079-8}',
    'GrangerN': r'Granger, C. W. J. (2004). Time series analysis, cointegration, and applications. \textit{American Economic Review}, 94(3), 421--425. \href{https://doi.org/10.1257/0002828041464669}{doi:10.1257/0002828041464669}',
    'GN': r'Granger, C. W. J., \& Newbold, P. (1974). Spurious regressions in econometrics. \textit{Journal of Econometrics}, 2(2), 111--120. \href{https://doi.org/10.1016/0304-4076(74)90034-7}{doi:10.1016/0304-4076(74)90034-7}',
    'Hamilton': r'Hamilton, J. D. (1994). \textit{Time Series Analysis}. Princeton University Press. \href{https://doi.org/10.2307/j.ctv14jx6sm}{doi:10.2307/j.ctv14jx6sm}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'Joh88': r'Johansen, S. (1988). Statistical analysis of cointegration vectors. \textit{Journal of Economic Dynamics and Control}, 12(2--3), 231--254. \href{https://doi.org/10.1016/0165-1889(88)90041-3}{doi:10.1016/0165-1889(88)90041-3}',
    'Joh91': r'Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. \textit{Econometrica}, 59(6), 1551--1580. \href{https://doi.org/10.2307/2938278}{doi:10.2307/2938278}',
    'JohBook': r'Johansen, S. (1995). \textit{Likelihood-Based Inference in Cointegrated Vector Autoregressive Models}. Oxford University Press. \href{https://doi.org/10.1093/0198774508.001.0001}{doi:10.1093/0198774508.001.0001}',
    'JJ': r'Johansen, S., \& Juselius, K. (1990). Maximum likelihood estimation and inference on cointegration, with applications to the demand for money. \textit{Oxford Bulletin of Economics and Statistics}, 52(2), 169--210. \href{https://doi.org/10.1111/j.1468-0084.1990.mp52002003.x}{doi:10.1111/j.1468-0084.1990.mp52002003.x}',
    'Jus': r'Juselius, K. (2006). \textit{The Cointegrated VAR Model: Methodology and Applications}. Oxford University Press. \href{https://doi.org/10.1093/oso/9780199285662.001.0001}{doi:10.1093/oso/9780199285662.001.0001}',
    'Lut': r'Lütkepohl, H. (2005). \textit{New Introduction to Multiple Time Series Analysis}. Springer. \href{https://doi.org/10.1007/978-3-540-27752-1}{doi:10.1007/978-3-540-27752-1}',
    'MacKinnon': r'MacKinnon, J. G. (1994). Approximate asymptotic distribution functions for unit-root and cointegration tests. \textit{Journal of Business \& Economic Statistics}, 12(2), 167--176. \href{https://doi.org/10.1080/07350015.1994.10510005}{doi:10.1080/07350015.1994.10510005}',
    'MacKinnonb': r"MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's Economics Department Working Paper No.~1227, Queen's University. \href{https://ideas.repec.org/p/qed/wpaper/1227.html}{ideas.repec.org/p/qed/wpaper/1227}",
    'MHM': r'MacKinnon, J. G., Haug, A. A., \& Michelis, L. (1999). Numerical distribution functions of likelihood ratio tests for cointegration. \textit{Journal of Applied Econometrics}, 14(5), 563--577. \href{https://doi.org/10.1002/(SICI)1099-1255(199909/10)14:5<563::AID-JAE530>3.0.CO;2-R}{doi:10.1002/(SICI)1099-1255(199909/10)14:5<563::AID-JAE530>3.0.CO;2-R}',
    'Murray': r'Murray, M. P. (1994). A drunk and her dog: An illustration of cointegration and error correction. \textit{The American Statistician}, 48(1), 37--39. \href{https://doi.org/10.1080/00031305.1994.10476017}{doi:10.1080/00031305.1994.10476017}',
    'OL': r'Osterwald-Lenum, M. (1992). A note with quantiles of the asymptotic distribution of the maximum likelihood cointegration rank test statistics. \textit{Oxford Bulletin of Economics and Statistics}, 54(3), 461--472. \href{https://doi.org/10.1111/j.1468-0084.1992.tb00013.x}{doi:10.1111/j.1468-0084.1992.tb00013.x}',
    'PO': r'Phillips, P. C. B., \& Ouliaris, S. (1990). Asymptotic properties of residual based tests for cointegration. \textit{Econometrica}, 58(1), 165--193. \href{https://doi.org/10.2307/2938339}{doi:10.2307/2938339}',
    'RA': r'Reinsel, G. C., \& Ahn, S. K. (1992). Vector autoregressive models with unit roots and reduced rank structure: Estimation, likelihood ratio test, and forecasting. \textit{Journal of Time Series Analysis}, 13(4), 353--375. \href{https://doi.org/10.1111/j.1467-9892.1992.tb00113.x}{doi:10.1111/j.1467-9892.1992.tb00113.x}',
    'SW': r'Stock, J. H., \& Watson, M. W. (1988). Testing for common trends. \textit{Journal of the American Statistical Association}, 83(404), 1097--1107. \href{https://doi.org/10.1080/01621459.1988.10478707}{doi:10.1080/01621459.1988.10478707}',
    'SWd': r'Stock, J. H., \& Watson, M. W. (1993). A simple estimator of cointegrating vectors in higher order integrated systems. \textit{Econometrica}, 61(4), 783--820. \href{https://doi.org/10.2307/2951763}{doi:10.2307/2951763}',
    'TT': r'Taylor, A. M., \& Taylor, M. P. (2004). The purchasing power parity debate. \textit{Journal of Economic Perspectives}, 18(4), 135--158. \href{https://doi.org/10.1257/0895330042632744}{doi:10.1257/0895330042632744}',
}
ORDER = ['Balassa', 'BDM', 'CS', 'CD', 'DHSY', 'DoFaff', 'EG', 'EY', 'GGR', 'Granger', 'GrangerN', 'GN', 'Hamilton', 'HP',
         'Joh88', 'Joh91', 'JohBook', 'JJ', 'Jus', 'Lut', 'MacKinnon', 'MacKinnonb', 'MHM', 'Murray', 'OL', 'PO', 'RA',
         'SW', 'SWd', 'TT']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
