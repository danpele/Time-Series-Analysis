r"""
ch6_common.py -- shared helpers of the Chapter 6 generators (lecture and seminar), TSA
=====================================================================================
Numbers from Quantlets/Ch_06/ch6_numbers.json (generate_all_charts.py) and sem6_results.json (seminar6.py);
the clickable citations of Chapter 6 (DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401
from ch1_common import T, V2, finalize, date, quarter, pv, MONTHS_EN, MONTHS_RO   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_06')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_06'


def qtr(s):
    """'2026-04-01' or '2026Q2' -> 2026Q2 / T2 2026."""
    if 'Q' in s:
        y, q = s[:4], s[-1]
    else:
        y, q = s[:4], str((int(s[5:7]) - 1) // 3 + 1)
    return V2(f'{y}Q{q}', f'T{q} {y}')


def month(s):
    y, m = int(s[:4]), int(s[5:7]) - 1
    return V2(f'{MONTHS_EN[m]} {y}', f'{MONTHS_RO[m]} {y}')


def load():
    with open(os.path.join(QL, 'ch6_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem6_results.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('FPP', 'https://otexts.com/fpp3/VAR.html', 'Hyndman and Athanasopoulos (2021)', 'Hyndman și Athanasopoulos (2021)')
        + ref('Lut', D_ + '10.1007/978-3-540-27752-1', 'Lütkepohl (2005)')
        + ref('Hamilton', D_ + '10.2307/j.ctv14jx6sm', 'Hamilton (1994)')
        + ref('Sims', D_ + '10.2307/1912017', 'Sims (1980)')
        + ref('SimsPP', D_ + '10.1016/0014-2921(92)90041-T', 'Sims (1992)')
        + ref('Granger', D_ + '10.2307/1912791', 'Granger (1969)')
        + ref('LutOm', D_ + '10.1016/0304-4076(82)90011-2', 'Lütkepohl (1982)')
        + ref('TY', D_ + '10.1016/0304-4076(94)01616-8', 'Toda and Yamamoto (1995)', 'Toda și Yamamoto (1995)')
        + ref('SSW', D_ + '10.2307/2938337', 'Sims, Stock and Watson (1990)', 'Sims, Stock și Watson (1990)')
        + ref('SW', D_ + '10.1257/jep.15.4.101', 'Stock and Watson (2001)', 'Stock și Watson (2001)')
        + ref('PS', D_ + '10.1016/S0165-1765(97)00214-0', 'Pesaran and Shin (1998)', 'Pesaran și Shin (1998)')
        + ref('KPP', D_ + '10.1016/0304-4076(95)01753-4', 'Koop, Pesaran and Potter (1996)', 'Koop, Pesaran și Potter (1996)')
        + ref('DY', D_ + '10.1111/j.1468-0297.2008.02208.x', 'Diebold and Yilmaz (2009)', 'Diebold și Yilmaz (2009)')
        + ref('DYb', D_ + '10.1016/j.ijforecast.2011.02.006', 'Diebold and Yilmaz (2012)', 'Diebold și Yilmaz (2012)')
        + ref('Kilian', D_ + '10.1162/003465398557465', 'Kilian (1998)')
        + ref('Hosking', D_ + '10.1080/01621459.1980.10477520', 'Hosking (1980)')
        + ref('JB', D_ + '10.1016/0165-1765(80)90024-5', 'Jarque and Bera (1980)', 'Jarque și Bera (1980)')
        + ref('Akaike', D_ + '10.1109/TAC.1974.1100705', 'Akaike (1974)')
        + ref('Schwarz', D_ + '10.1214/aos/1176344136', 'Schwarz (1978)')
        + ref('HQ', D_ + '10.1111/j.2517-6161.1979.tb01072.x', 'Hannan and Quinn (1979)', 'Hannan și Quinn (1979)')
        + ref('DM', D_ + '10.1080/07350015.1995.10524599', 'Diebold and Mariano (1995)', 'Diebold și Mariano (1995)')
        + ref('EG', D_ + '10.2307/1913236', 'Engle and Granger (1987)', 'Engle și Granger (1987)'))

BIB = {
    'Akaike': r'Akaike, H. (1974). A new look at the statistical model identification. \textit{IEEE Transactions on Automatic Control}, 19(6), 716--723. \href{https://doi.org/10.1109/TAC.1974.1100705}{doi:10.1109/TAC.1974.1100705}',
    'DM': r'Diebold, F. X., \& Mariano, R. S. (1995). Comparing predictive accuracy. \textit{Journal of Business \& Economic Statistics}, 13(3), 253--263. \href{https://doi.org/10.1080/07350015.1995.10524599}{doi:10.1080/07350015.1995.10524599}',
    'DY': r'Diebold, F. X., \& Yilmaz, K. (2009). Measuring financial asset return and volatility spillovers, with application to global equity markets. \textit{The Economic Journal}, 119(534), 158--171. \href{https://doi.org/10.1111/j.1468-0297.2008.02208.x}{doi:10.1111/j.1468-0297.2008.02208.x}',
    'DYb': r'Diebold, F. X., \& Yilmaz, K. (2012). Better to give than to receive: Predictive directional measurement of volatility spillovers. \textit{International Journal of Forecasting}, 28(1), 57--66. \href{https://doi.org/10.1016/j.ijforecast.2011.02.006}{doi:10.1016/j.ijforecast.2011.02.006}',
    'EG': r'Engle, R. F., \& Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. \textit{Econometrica}, 55(2), 251--276. \href{https://doi.org/10.2307/1913236}{doi:10.2307/1913236}',
    'Granger': r'Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. \textit{Econometrica}, 37(3), 424--438. \href{https://doi.org/10.2307/1912791}{doi:10.2307/1912791}',
    'Hamilton': r'Hamilton, J. D. (1994). \textit{Time Series Analysis}. Princeton University Press. \href{https://doi.org/10.2307/j.ctv14jx6sm}{doi:10.2307/j.ctv14jx6sm}',
    'HQ': r'Hannan, E. J., \& Quinn, B. G. (1979). The determination of the order of an autoregression. \textit{Journal of the Royal Statistical Society: Series B}, 41(2), 190--195. \href{https://doi.org/10.1111/j.2517-6161.1979.tb01072.x}{doi:10.1111/j.2517-6161.1979.tb01072.x}',
    'Hosking': r'Hosking, J. R. M. (1980). The multivariate portmanteau statistic. \textit{Journal of the American Statistical Association}, 75(371), 602--608. \href{https://doi.org/10.1080/01621459.1980.10477520}{doi:10.1080/01621459.1980.10477520}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'FPP': r'Hyndman, R. J., \& Athanasopoulos, G. (2021). \textit{Forecasting: Principles and Practice} (3rd ed.), Section 12.3, Vector autoregressions. OTexts. \href{https://otexts.com/fpp3/VAR.html}{otexts.com/fpp3/VAR.html}',
    'JB': r'Jarque, C. M., \& Bera, A. K. (1980). Efficient tests for normality, homoscedasticity and serial independence of regression residuals. \textit{Economics Letters}, 6(3), 255--259. \href{https://doi.org/10.1016/0165-1765(80)90024-5}{doi:10.1016/0165-1765(80)90024-5}',
    'Kilian': r'Kilian, L. (1998). Small-sample confidence intervals for impulse response functions. \textit{Review of Economics and Statistics}, 80(2), 218--230. \href{https://doi.org/10.1162/003465398557465}{doi:10.1162/003465398557465}',
    'KPP': r'Koop, G., Pesaran, M. H., \& Potter, S. M. (1996). Impulse response analysis in nonlinear multivariate models. \textit{Journal of Econometrics}, 74(1), 119--147. \href{https://doi.org/10.1016/0304-4076(95)01753-4}{doi:10.1016/0304-4076(95)01753-4}',
    'LutOm': r'Lütkepohl, H. (1982). Non-causality due to omitted variables. \textit{Journal of Econometrics}, 19(2--3), 367--378. \href{https://doi.org/10.1016/0304-4076(82)90011-2}{doi:10.1016/0304-4076(82)90011-2}',
    'Lut': r'Lütkepohl, H. (2005). \textit{New Introduction to Multiple Time Series Analysis}. Springer. \href{https://doi.org/10.1007/978-3-540-27752-1}{doi:10.1007/978-3-540-27752-1}',
    'PS': r'Pesaran, M. H., \& Shin, Y. (1998). Generalized impulse response analysis in linear multivariate models. \textit{Economics Letters}, 58(1), 17--29. \href{https://doi.org/10.1016/S0165-1765(97)00214-0}{doi:10.1016/S0165-1765(97)00214-0}',
    'Schwarz': r'Schwarz, G. (1978). Estimating the dimension of a model. \textit{The Annals of Statistics}, 6(2), 461--464. \href{https://doi.org/10.1214/aos/1176344136}{doi:10.1214/aos/1176344136}',
    'Sims': r'Sims, C. A. (1980). Macroeconomics and reality. \textit{Econometrica}, 48(1), 1--48. \href{https://doi.org/10.2307/1912017}{doi:10.2307/1912017}',
    'SimsPP': r'Sims, C. A. (1992). Interpreting the macroeconomic time series facts: The effects of monetary policy. \textit{European Economic Review}, 36(5), 975--1000. \href{https://doi.org/10.1016/0014-2921(92)90041-T}{doi:10.1016/0014-2921(92)90041-T}',
    'SSW': r'Sims, C. A., Stock, J. H., \& Watson, M. W. (1990). Inference in linear time series models with some unit roots. \textit{Econometrica}, 58(1), 113--144. \href{https://doi.org/10.2307/2938337}{doi:10.2307/2938337}',
    'SW': r'Stock, J. H., \& Watson, M. W. (2001). Vector autoregressions. \textit{Journal of Economic Perspectives}, 15(4), 101--115. \href{https://doi.org/10.1257/jep.15.4.101}{doi:10.1257/jep.15.4.101}',
    'TY': r'Toda, H. Y., \& Yamamoto, T. (1995). Statistical inference in vector autoregressions with possibly integrated processes. \textit{Journal of Econometrics}, 66(1--2), 225--250. \href{https://doi.org/10.1016/0304-4076(94)01616-8}{doi:10.1016/0304-4076(94)01616-8}',
}
ORDER = ['Akaike', 'DM', 'DY', 'DYb', 'EG', 'Granger', 'Hamilton', 'HQ', 'Hosking', 'HP', 'FPP', 'JB', 'Kilian', 'KPP',
         'LutOm', 'Lut', 'PS', 'Schwarz', 'Sims', 'SimsPP', 'SSW', 'SW', 'TY']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
