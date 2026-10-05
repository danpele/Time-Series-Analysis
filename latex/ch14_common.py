r"""
ch14_common.py -- shared helpers of the Chapter 14 generator (lecture), TSA
=========================================================================
Numbers from Quantlets/Ch_14/ch14_numbers.json (generate_all_charts.py); the clickable citations of Chapter 14
(DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401
from ch1_common import T, V2, finalize, date, pv, MONTHS_EN, MONTHS_RO   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_14')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_14'


def month(s):
    y, m = int(s[:4]), int(s[5:7]) - 1
    return V2(f'{MONTHS_EN[m]} {y}', f'{MONTHS_RO[m]} {y}')


def load():
    with open(os.path.join(QL, 'ch14_numbers.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('Engle', D_ + '10.1198/073500102288618487', 'Engle (2002)')
        + ref('Boll', D_ + '10.2307/2109358', 'Bollerslev (1990)')
        + ref('BollG', D_ + '10.1016/0304-4076(86)90063-1', 'Bollerslev (1986)')
        + ref('EK', D_ + '10.1017/S0266466600009063', 'Engle and Kroner (1995)', 'Engle și Kroner (1995)')
        + ref('BEW', D_ + '10.1086/261527', 'Bollerslev, Engle and Wooldridge (1988)', 'Bollerslev, Engle și Wooldridge (1988)')
        + ref('Tse', D_ + '10.1016/S0304-4076(99)00080-9', 'Tse (2000)')
        + ref('CES', D_ + '10.1093/jjfinec/nbl005', 'Cappiello, Engle and Sheppard (2006)', 'Cappiello, Engle și Sheppard (2006)')
        + ref('Aielli', D_ + '10.1080/07350015.2013.771027', 'Aielli (2013)')
        + ref('KS', D_ + '10.2307/2331164', 'Kroner and Sultan (1993)', 'Kroner și Sultan (1993)')
        + ref('Ed', D_ + '10.1111/j.1540-6261.1979.tb02077.x', 'Ederington (1979)')
        + ref('FR', D_ + '10.1111/0022-1082.00494', 'Forbes and Rigobon (2002)', 'Forbes și Rigobon (2002)')
        + ref('LS', D_ + '10.1111/0022-1082.00340', 'Longin and Solnik (2001)', 'Longin și Solnik (2001)')
        + ref('BLR', D_ + '10.1002/jae.842', 'Bauwens, Laurent and Rombouts (2006)', 'Bauwens, Laurent și Rombouts (2006)')
        + ref('ST', D_ + '10.1007/978-3-540-71297-8_9', 'Silvennoinen and Teräsvirta (2009)', 'Silvennoinen și Teräsvirta (2009)')
        + ref('Kupiec', D_ + '10.3905/jod.1995.407942', 'Kupiec (1995)')
        + ref('Chr', D_ + '10.2307/2527341', 'Christoffersen (1998)')
        + ref('Mark', D_ + '10.1111/j.1540-6261.1952.tb01525.x', 'Markowitz (1952)')
        + ref('DY', D_ + '10.1111/j.1468-0297.2008.02208.x', 'Diebold and Yilmaz (2009)', 'Diebold și Yilmaz (2009)'))

BIB = {
    'Aielli': r'Aielli, G. P. (2013). Dynamic conditional correlation: On properties and estimation. \textit{Journal of Business \& Economic Statistics}, 31(3), 282--299. \href{https://doi.org/10.1080/07350015.2013.771027}{doi:10.1080/07350015.2013.771027}',
    'BLR': r'Bauwens, L., Laurent, S., \& Rombouts, J. V. K. (2006). Multivariate GARCH models: A survey. \textit{Journal of Applied Econometrics}, 21(1), 79--109. \href{https://doi.org/10.1002/jae.842}{doi:10.1002/jae.842}',
    'BollG': r'Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. \textit{Journal of Econometrics}, 31(3), 307--327. \href{https://doi.org/10.1016/0304-4076(86)90063-1}{doi:10.1016/0304-4076(86)90063-1}',
    'Boll': r'Bollerslev, T. (1990). Modelling the coherence in short-run nominal exchange rates: A multivariate generalized ARCH model. \textit{The Review of Economics and Statistics}, 72(3), 498--505. \href{https://doi.org/10.2307/2109358}{doi:10.2307/2109358}',
    'BEW': r'Bollerslev, T., Engle, R. F., \& Wooldridge, J. M. (1988). A capital asset pricing model with time-varying covariances. \textit{Journal of Political Economy}, 96(1), 116--131. \href{https://doi.org/10.1086/261527}{doi:10.1086/261527}',
    'CES': r'Cappiello, L., Engle, R. F., \& Sheppard, K. (2006). Asymmetric dynamics in the correlations of global equity and bond returns. \textit{Journal of Financial Econometrics}, 4(4), 537--572. \href{https://doi.org/10.1093/jjfinec/nbl005}{doi:10.1093/jjfinec/nbl005}',
    'Chr': r'Christoffersen, P. F. (1998). Evaluating interval forecasts. \textit{International Economic Review}, 39(4), 841--862. \href{https://doi.org/10.2307/2527341}{doi:10.2307/2527341}',
    'DY': r'Diebold, F. X., \& Yilmaz, K. (2009). Measuring financial asset return and volatility spillovers, with application to global equity markets. \textit{The Economic Journal}, 119(534), 158--171. \href{https://doi.org/10.1111/j.1468-0297.2008.02208.x}{doi:10.1111/j.1468-0297.2008.02208.x}',
    'Ed': r'Ederington, L. H. (1979). The hedging performance of the new futures markets. \textit{The Journal of Finance}, 34(1), 157--170. \href{https://doi.org/10.1111/j.1540-6261.1979.tb02077.x}{doi:10.1111/j.1540-6261.1979.tb02077.x}',
    'Engle': r'Engle, R. F. (2002). Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models. \textit{Journal of Business \& Economic Statistics}, 20(3), 339--350. \href{https://doi.org/10.1198/073500102288618487}{doi:10.1198/073500102288618487}',
    'EK': r'Engle, R. F., \& Kroner, K. F. (1995). Multivariate simultaneous generalized ARCH. \textit{Econometric Theory}, 11(1), 122--150. \href{https://doi.org/10.1017/S0266466600009063}{doi:10.1017/S0266466600009063}',
    'FR': r'Forbes, K. J., \& Rigobon, R. (2002). No contagion, only interdependence: Measuring stock market comovements. \textit{The Journal of Finance}, 57(5), 2223--2261. \href{https://doi.org/10.1111/0022-1082.00494}{doi:10.1111/0022-1082.00494}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'KS': r'Kroner, K. F., \& Sultan, J. (1993). Time-varying distributions and dynamic hedging with foreign currency futures. \textit{Journal of Financial and Quantitative Analysis}, 28(4), 535--551. \href{https://doi.org/10.2307/2331164}{doi:10.2307/2331164}',
    'Kupiec': r'Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. \textit{The Journal of Derivatives}, 3(2), 73--84. \href{https://doi.org/10.3905/jod.1995.407942}{doi:10.3905/jod.1995.407942}',
    'LS': r'Longin, F., \& Solnik, B. (2001). Extreme correlation of international equity markets. \textit{The Journal of Finance}, 56(2), 649--676. \href{https://doi.org/10.1111/0022-1082.00340}{doi:10.1111/0022-1082.00340}',
    'Mark': r'Markowitz, H. (1952). Portfolio selection. \textit{The Journal of Finance}, 7(1), 77--91. \href{https://doi.org/10.1111/j.1540-6261.1952.tb01525.x}{doi:10.1111/j.1540-6261.1952.tb01525.x}',
    'ST': r'Silvennoinen, A., \& Teräsvirta, T. (2009). Multivariate GARCH models. In T. G. Andersen et al.\ (Eds.), \textit{Handbook of Financial Time Series} (pp.~201--229). Springer. \href{https://doi.org/10.1007/978-3-540-71297-8_9}{doi:10.1007/978-3-540-71297-8\_9}',
    'Tse': r'Tse, Y. K. (2000). A test for constant correlations in a multivariate GARCH model. \textit{Journal of Econometrics}, 98(1), 107--127. \href{https://doi.org/10.1016/S0304-4076(99)00080-9}{doi:10.1016/S0304-4076(99)00080-9}',
}
ORDER = ['Aielli', 'BLR', 'BollG', 'Boll', 'BEW', 'CES', 'Chr', 'DY', 'Ed', 'Engle', 'EK', 'FR', 'HP', 'KS', 'Kupiec', 'LS',
         'Mark', 'ST', 'Tse']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
