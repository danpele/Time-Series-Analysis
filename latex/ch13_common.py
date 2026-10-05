r"""
ch13_common.py -- shared helpers of the Chapter 13 generator (lecture), TSA
==========================================================================
Numbers from Quantlets/Ch_13/ch13_numbers.json (generate_all_charts.py); the clickable citations of Chapter 13
(DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401
from ch1_common import T, V2, finalize, date, pv, MONTHS_EN, MONTHS_RO   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_13')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_13'


def month(s):
    y, m = int(s[:4]), int(s[5:7]) - 1
    return V2(f'{MONTHS_EN[m]} {y}', f'{MONTHS_RO[m]} {y}')


def load():
    with open(os.path.join(QL, 'ch13_numbers.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('JLS', D_ + '10.1142/S0219024900000115', 'Johansen, Ledoit and Sornette (2000)', 'Johansen, Ledoit și Sornette (2000)')
        + ref('SJB', D_ + '10.1051/jp1:1996135', 'Sornette, Johansen and Bouchaud (1996)', 'Sornette, Johansen și Bouchaud (1996)')
        + ref('Sornette', D_ + '10.23943/princeton/9780691175959.001.0001', 'Sornette (2017)')
        + ref('FS', D_ + '10.1016/j.physa.2013.04.012', 'Filimonov and Sornette (2013)', 'Filimonov și Sornette (2013)')
        + ref('SZ', D_ + '10.1016/j.physa.2020.124892', 'Shu and Zhu (2020)', 'Shu și Zhu (2020)')
        + ref('Shanghai', D_ + '10.21314/JOIS.2015.063', 'Sornette et al.\\ (2015)')
        + ref('Jiang', D_ + '10.1016/j.jebo.2010.02.007', 'Jiang et al.\\ (2010)')
        + ref('Gerlach', D_ + '10.1098/rsos.180643', 'Gerlach, Demos and Sornette (2019)', 'Gerlach, Demos și Sornette (2019)')
        + ref('Feig', D_ + '10.1088/1469-7688/1/3/306', 'Feigenbaum (2001)')
        + ref('BJ', D_ + '10.1016/j.irfa.2013.05.005', 'Brée and Joseph (2013)', 'Brée și Joseph (2013)')
        + ref('Lomb', D_ + '10.1007/BF00648343', 'Lomb (1976)')
        + ref('BW', D_ + '10.3386/w0945', 'Blanchard and Watson (1982)', 'Blanchard și Watson (1982)')
        + ref('PWY', D_ + '10.1111/j.1468-2354.2010.00625.x', 'Phillips, Wu and Yu (2011)', 'Phillips, Wu și Yu (2011)')
        + ref('PSYa', D_ + '10.1111/iere.12132', 'Phillips, Shi and Yu (2015a)', 'Phillips, Shi și Yu (2015a)')
        + ref('PSYb', D_ + '10.1111/iere.12131', 'Phillips, Shi and Yu (2015b)', 'Phillips, Shi și Yu (2015b)')
        + ref('PS', D_ + '10.1016/bs.host.2018.12.002', 'Phillips and Shi (2020)', 'Phillips și Shi (2020)')
        + ref('DF', D_ + '10.1080/01621459.1979.10482531', 'Dickey and Fuller (1979)', 'Dickey și Fuller (1979)')
        + ref('Garber', D_ + '10.1257/jep.4.2.35', 'Garber (1990)')
        + ref('KA', D_ + '10.1057/9780230628045', 'Kindleberger and Aliber (2005)', 'Kindleberger și Aliber (2005)')
        + ref('Shiller', D_ + '10.1515/9781400865536', 'Shiller (2015)')
        + ref('Fama', D_ + '10.1257/aer.104.6.1467', 'Fama (2014)')
        + ref('GSY', D_ + '10.1016/j.jfineco.2018.09.002', 'Greenwood, Shleifer and You (2019)', 'Greenwood, Shleifer și You (2019)')
        + ref('PeleCrash', D_ + '10.1016/j.sbspro.2012.09.1030', 'Pele and Mazurencu-Marinescu (2012)', 'Pele și Mazurencu-Marinescu (2012)'))

BIB = {
    'BW': r'Blanchard, O. J., \& Watson, M. W. (1982). Bubbles, rational expectations and financial markets. NBER Working Paper 945. \href{https://doi.org/10.3386/w0945}{doi:10.3386/w0945}',
    'BJ': r'Brée, D. S., \& Joseph, N. L. (2013). Testing for financial crashes using the log periodic power law model. \textit{International Review of Financial Analysis}, 30, 287--297. \href{https://doi.org/10.1016/j.irfa.2013.05.005}{doi:10.1016/j.irfa.2013.05.005}',
    'DF': r'Dickey, D. A., \& Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. \textit{Journal of the American Statistical Association}, 74(366), 427--431. \href{https://doi.org/10.1080/01621459.1979.10482531}{doi:10.1080/01621459.1979.10482531}',
    'Fama': r'Fama, E. F. (2014). Two pillars of asset pricing. \textit{American Economic Review}, 104(6), 1467--1485. \href{https://doi.org/10.1257/aer.104.6.1467}{doi:10.1257/aer.104.6.1467}',
    'Feig': r'Feigenbaum, J. A. (2001). A statistical analysis of log-periodic precursors to financial crashes. \textit{Quantitative Finance}, 1(3), 346--360. \href{https://doi.org/10.1088/1469-7688/1/3/306}{doi:10.1088/1469-7688/1/3/306}',
    'FS': r'Filimonov, V., \& Sornette, D. (2013). A stable and robust calibration scheme of the log-periodic power law model. \textit{Physica A}, 392(17), 3698--3707. \href{https://doi.org/10.1016/j.physa.2013.04.012}{doi:10.1016/j.physa.2013.04.012}',
    'Garber': r'Garber, P. M. (1990). Famous first bubbles. \textit{Journal of Economic Perspectives}, 4(2), 35--54. \href{https://doi.org/10.1257/jep.4.2.35}{doi:10.1257/jep.4.2.35}',
    'Gerlach': r'Gerlach, J.-C., Demos, G., \& Sornette, D. (2019). Dissection of Bitcoin\textquoteright s multiscale bubble history from January 2012 to February 2018. \textit{Royal Society Open Science}, 6(7), 180643. \href{https://doi.org/10.1098/rsos.180643}{doi:10.1098/rsos.180643}',
    'GSY': r'Greenwood, R., Shleifer, A., \& You, Y. (2019). Bubbles for Fama. \textit{Journal of Financial Economics}, 131(1), 20--43. \href{https://doi.org/10.1016/j.jfineco.2018.09.002}{doi:10.1016/j.jfineco.2018.09.002}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'Jiang': r'Jiang, Z.-Q., Zhou, W.-X., Sornette, D., Woodard, R., Bastiaensen, K., \& Cauwels, P. (2010). Bubble diagnosis and prediction of the 2005--2007 and 2008--2009 Chinese stock market bubbles. \textit{Journal of Economic Behavior \& Organization}, 74(3), 149--162. \href{https://doi.org/10.1016/j.jebo.2010.02.007}{doi:10.1016/j.jebo.2010.02.007}',
    'JLS': r'Johansen, A., Ledoit, O., \& Sornette, D. (2000). Crashes as critical points. \textit{International Journal of Theoretical and Applied Finance}, 3(2), 219--255. \href{https://doi.org/10.1142/S0219024900000115}{doi:10.1142/S0219024900000115}',
    'KA': r'Kindleberger, C. P., \& Aliber, R. Z. (2005). \textit{Manias, Panics and Crashes: A History of Financial Crises} (5th ed.). Palgrave Macmillan. \href{https://doi.org/10.1057/9780230628045}{doi:10.1057/9780230628045}',
    'Lomb': r'Lomb, N. R. (1976). Least-squares frequency analysis of unequally spaced data. \textit{Astrophysics and Space Science}, 39(2), 447--462. \href{https://doi.org/10.1007/BF00648343}{doi:10.1007/BF00648343}',
    'PeleCrash': r'Pele, D. T., \& Mazurencu-Marinescu, M. (2012). Modelling stock market crashes: The case of Bucharest Stock Exchange. \textit{Procedia -- Social and Behavioral Sciences}, 58, 533--542. \href{https://doi.org/10.1016/j.sbspro.2012.09.1030}{doi:10.1016/j.sbspro.2012.09.1030}',
    'PS': r'Phillips, P. C. B., \& Shi, S. (2020). Real time monitoring of asset markets: Bubbles and crises. In \textit{Handbook of Statistics: Financial, Macro and Micro Econometrics Using R}, 61--80. Elsevier. \href{https://doi.org/10.1016/bs.host.2018.12.002}{doi:10.1016/bs.host.2018.12.002}',
    'PSYa': r'Phillips, P. C. B., Shi, S., \& Yu, J. (2015a). Testing for multiple bubbles: Historical episodes of exuberance and collapse in the S\&P 500. \textit{International Economic Review}, 56(4), 1043--1078. \href{https://doi.org/10.1111/iere.12132}{doi:10.1111/iere.12132}',
    'PSYb': r'Phillips, P. C. B., Shi, S., \& Yu, J. (2015b). Testing for multiple bubbles: Limit theory of real-time detectors. \textit{International Economic Review}, 56(4), 1079--1134. \href{https://doi.org/10.1111/iere.12131}{doi:10.1111/iere.12131}',
    'PWY': r'Phillips, P. C. B., Wu, Y., \& Yu, J. (2011). Explosive behavior in the 1990s Nasdaq: When did exuberance escalate asset values? \textit{International Economic Review}, 52(1), 201--226. \href{https://doi.org/10.1111/j.1468-2354.2010.00625.x}{doi:10.1111/j.1468-2354.2010.00625.x}',
    'Shiller': r'Shiller, R. J. (2015). \textit{Irrational Exuberance} (3rd ed.). Princeton University Press. \href{https://doi.org/10.1515/9781400865536}{doi:10.1515/9781400865536}',
    'SZ': r'Shu, M., \& Zhu, W. (2020). Detection of Chinese stock market bubbles with LPPLS confidence indicator. \textit{Physica A}, 557, 124892. \href{https://doi.org/10.1016/j.physa.2020.124892}{doi:10.1016/j.physa.2020.124892}',
    'Sornette': r'Sornette, D. (2017). \textit{Why Stock Markets Crash: Critical Events in Complex Financial Systems}. Princeton University Press. \href{https://doi.org/10.23943/princeton/9780691175959.001.0001}{doi:10.23943/princeton/9780691175959.001.0001}',
    'Shanghai': r'Sornette, D., Demos, G., Zhang, Q., Cauwels, P., \& Zhang, Q. (2015). Real-time prediction and post-mortem analysis of the Shanghai 2015 stock market bubble and crash. \textit{Journal of Investment Strategies}, 4(4), 77--95. \href{https://doi.org/10.21314/JOIS.2015.063}{doi:10.21314/JOIS.2015.063}',
    'SJB': r'Sornette, D., Johansen, A., \& Bouchaud, J.-P. (1996). Stock market crashes, precursors and replicas. \textit{Journal de Physique I}, 6(1), 167--175. \href{https://doi.org/10.1051/jp1:1996135}{doi:10.1051/jp1:1996135}',
}
ORDER = ['BW', 'BJ', 'DF', 'Fama', 'Feig', 'FS', 'Garber', 'Gerlach', 'GSY', 'HP', 'Jiang', 'JLS', 'KA', 'Lomb',
         'PeleCrash', 'PS', 'PSYa', 'PSYb', 'PWY', 'Shiller', 'SZ', 'Sornette', 'Shanghai', 'SJB']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
