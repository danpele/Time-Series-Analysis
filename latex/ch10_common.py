r"""
ch10_common.py -- shared helpers of the Chapter 10 generators (lecture and seminar), TSA
=======================================================================================
Numbers from Quantlets/Ch_10/ch10_numbers.json (generate_all_charts.py) and sem10_results.json (seminar10.py);
the clickable citations of Chapter 10 (DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401
from ch1_common import T, V2, date, quarter, pv, MONTHS_EN, MONTHS_RO   # noqa: E402,F401
from ch6_common import finalize   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_10')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_10'


def month(s):
    y, m = int(s[:4]), int(s[5:7]) - 1
    return V2(f'{MONTHS_EN[m]} {y}', f'{MONTHS_RO[m]} {y}')


def load():
    with open(os.path.join(QL, 'ch10_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem10_results.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('DK', D_ + '10.1093/acprof:oso/9780199641178.001.0001', 'Durbin and Koopman (2012)', 'Durbin și Koopman (2012)')
        + ref('Harvey', D_ + '10.1017/CBO9781107049994', 'Harvey (1989)')
        + ref('Hamilton', D_ + '10.2307/j.ctv14jx6sm', 'Hamilton (1994)')
        + ref('KN', D_ + '10.7551/mitpress/6444.001.0001', 'Kim and Nelson (1999)', 'Kim și Nelson (1999)')
        + ref('SS', D_ + '10.1007/978-3-319-52452-8', 'Shumway and Stoffer (2017)', 'Shumway și Stoffer (2017)')
        + ref('Kalman', D_ + '10.1115/1.3662552', 'Kalman (1960)')
        + ref('RTS', D_ + '10.2514/3.3166', 'Rauch, Tung and Striebel (1965)', 'Rauch, Tung și Striebel (1965)')
        + ref('Muth', D_ + '10.1080/01621459.1960.10482064', 'Muth (1960)')
        + ref('HKOS', D_ + '10.1007/978-3-540-71918-2', 'Hyndman et al.\\ (2008)')
        + ref('HJ', D_ + '10.1002/jae.3950080302', 'Harvey and Jaeger (1993)', 'Harvey și Jaeger (1993)')
        + ref('HPf', D_ + '10.2307/2953682', 'Hodrick and Prescott (1997)', 'Hodrick și Prescott (1997)')
        + ref('HamHP', D_ + '10.1162/rest\\_a\\_00706', 'Hamilton (2018)')
        + ref('SW', D_ + '10.1086/654119', 'Stock and Watson (1989)', 'Stock și Watson (1989)')
        + ref('GRS', D_ + '10.1016/j.jmoneco.2008.05.010', 'Giannone, Reichlin and Small (2008)', 'Giannone, Reichlin și Small (2008)')
        + ref('HamMS', D_ + '10.2307/1912559', 'Hamilton (1989)')
        + ref('HamEM', D_ + '10.1016/0304-4076(90)90093-9', 'Hamilton (1990)')
        + ref('Kim', D_ + '10.1016/0304-4076(94)90036-1', 'Kim (1994)')
        + ref('CP', D_ + '10.1198/073500107000000296', 'Chauvet and Piger (2008)', 'Chauvet și Piger (2008)')
        + ref('HSus', D_ + '10.1016/0304-4076(94)90067-1', 'Hamilton and Susmel (1994)', 'Hamilton și Susmel (1994)')
        + ref('LL', D_ + '10.1080/07350015.1990.10509794', 'Lamoureux and Lastrapes (1990)', 'Lamoureux și Lastrapes (1990)')
        + ref('DI', D_ + '10.1016/S0304-4076(01)00073-2', 'Diebold and Inoue (2001)', 'Diebold și Inoue (2001)'))

BIB = {
    'CP': r'Chauvet, M., \& Piger, J. (2008). A comparison of the real-time performance of business cycle dating methods. \textit{Journal of Business \& Economic Statistics}, 26(1), 42--49. \href{https://doi.org/10.1198/073500107000000296}{doi:10.1198/073500107000000296}',
    'DI': r'Diebold, F. X., \& Inoue, A. (2001). Long memory and regime switching. \textit{Journal of Econometrics}, 105(1), 131--159. \href{https://doi.org/10.1016/S0304-4076(01)00073-2}{doi:10.1016/S0304-4076(01)00073-2}',
    'DK': r'Durbin, J., \& Koopman, S. J. (2012). \textit{Time Series Analysis by State Space Methods} (2nd ed.). Oxford University Press. \href{https://doi.org/10.1093/acprof:oso/9780199641178.001.0001}{doi:10.1093/acprof:oso/9780199641178.001.0001}',
    'GRS': r'Giannone, D., Reichlin, L., \& Small, D. (2008). Nowcasting: The real-time informational content of macroeconomic data. \textit{Journal of Monetary Economics}, 55(4), 665--676. \href{https://doi.org/10.1016/j.jmoneco.2008.05.010}{doi:10.1016/j.jmoneco.2008.05.010}',
    'HamMS': r'Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. \textit{Econometrica}, 57(2), 357--384. \href{https://doi.org/10.2307/1912559}{doi:10.2307/1912559}',
    'HamEM': r'Hamilton, J. D. (1990). Analysis of time series subject to changes in regime. \textit{Journal of Econometrics}, 45(1--2), 39--70. \href{https://doi.org/10.1016/0304-4076(90)90093-9}{doi:10.1016/0304-4076(90)90093-9}',
    'Hamilton': r'Hamilton, J. D. (1994). \textit{Time Series Analysis}. Princeton University Press. \href{https://doi.org/10.2307/j.ctv14jx6sm}{doi:10.2307/j.ctv14jx6sm}',
    'HamHP': r'Hamilton, J. D. (2018). Why you should never use the Hodrick--Prescott filter. \textit{The Review of Economics and Statistics}, 100(5), 831--843. \href{https://doi.org/10.1162/rest\_a\_00706}{doi:10.1162/rest\_a\_00706}',
    'HSus': r'Hamilton, J. D., \& Susmel, R. (1994). Autoregressive conditional heteroskedasticity and changes in regime. \textit{Journal of Econometrics}, 64(1--2), 307--333. \href{https://doi.org/10.1016/0304-4076(94)90067-1}{doi:10.1016/0304-4076(94)90067-1}',
    'Harvey': r'Harvey, A. C. (1989). \textit{Forecasting, Structural Time Series Models and the Kalman Filter}. Cambridge University Press. \href{https://doi.org/10.1017/CBO9781107049994}{doi:10.1017/CBO9781107049994}',
    'HJ': r'Harvey, A. C., \& Jaeger, A. (1993). Detrending, stylized facts and the business cycle. \textit{Journal of Applied Econometrics}, 8(3), 231--247. \href{https://doi.org/10.1002/jae.3950080302}{doi:10.1002/jae.3950080302}',
    'HPf': r'Hodrick, R. J., \& Prescott, E. C. (1997). Postwar U.S. business cycles: An empirical investigation. \textit{Journal of Money, Credit and Banking}, 29(1), 1--16. \href{https://doi.org/10.2307/2953682}{doi:10.2307/2953682}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'HKOS': r'Hyndman, R. J., Koehler, A. B., Ord, J. K., \& Snyder, R. D. (2008). \textit{Forecasting with Exponential Smoothing: The State Space Approach}. Springer. \href{https://doi.org/10.1007/978-3-540-71918-2}{doi:10.1007/978-3-540-71918-2}',
    'Kalman': r'Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. \textit{Journal of Basic Engineering}, 82(1), 35--45. \href{https://doi.org/10.1115/1.3662552}{doi:10.1115/1.3662552}',
    'Kim': r'Kim, C.-J. (1994). Dynamic linear models with Markov-switching. \textit{Journal of Econometrics}, 60(1--2), 1--22. \href{https://doi.org/10.1016/0304-4076(94)90036-1}{doi:10.1016/0304-4076(94)90036-1}',
    'KN': r'Kim, C.-J., \& Nelson, C. R. (1999). \textit{State-Space Models with Regime Switching: Classical and Gibbs-Sampling Approaches with Applications}. MIT Press. \href{https://doi.org/10.7551/mitpress/6444.001.0001}{doi:10.7551/mitpress/6444.001.0001}',
    'LL': r'Lamoureux, C. G., \& Lastrapes, W. D. (1990). Persistence in variance, structural change, and the GARCH model. \textit{Journal of Business \& Economic Statistics}, 8(2), 225--234. \href{https://doi.org/10.1080/07350015.1990.10509794}{doi:10.1080/07350015.1990.10509794}',
    'Muth': r'Muth, J. F. (1960). Optimal properties of exponentially weighted forecasts. \textit{Journal of the American Statistical Association}, 55(290), 299--306. \href{https://doi.org/10.1080/01621459.1960.10482064}{doi:10.1080/01621459.1960.10482064}',
    'RTS': r'Rauch, H. E., Tung, F., \& Striebel, C. T. (1965). Maximum likelihood estimates of linear dynamic systems. \textit{AIAA Journal}, 3(8), 1445--1450. \href{https://doi.org/10.2514/3.3166}{doi:10.2514/3.3166}',
    'SS': r'Shumway, R. H., \& Stoffer, D. S. (2017). \textit{Time Series Analysis and Its Applications} (4th ed.). Springer. \href{https://doi.org/10.1007/978-3-319-52452-8}{doi:10.1007/978-3-319-52452-8}',
    'SW': r'Stock, J. H., \& Watson, M. W. (1989). New indexes of coincident and leading economic indicators. \textit{NBER Macroeconomics Annual}, 4, 351--394. \href{https://doi.org/10.1086/654119}{doi:10.1086/654119}',
}
ORDER = ['CP', 'DI', 'DK', 'GRS', 'HamMS', 'HamEM', 'Hamilton', 'HamHP', 'HSus', 'Harvey', 'HJ', 'HPf', 'HP', 'HKOS',
         'Kalman', 'Kim', 'KN', 'LL', 'Muth', 'RTS', 'SS', 'SW']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
