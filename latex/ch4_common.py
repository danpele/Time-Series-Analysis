r"""
ch4_common.py -- shared helpers of the Chapter 4 generators (lecture and seminar), TSA
=====================================================================================
Numbers from Quantlets/Ch_04/ch4_numbers.json (generate_all_charts.py) and sem4_results.json (seminar4.py);
the clickable citations of Chapter 4 (DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_04')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_04'
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


def text_minus(line):
    """A negative number in text mode (outside $...$) gets a true minus sign: ' -0.81' -> ' $-$0.81'."""
    if line.lstrip().startswith('%'):
        return line
    parts = re.split(r'((?<!\\)\$)', line)
    out, math = [], False
    for part in parts:
        if part == '$':
            math = not math
            out.append(part)
        elif math:
            out.append(part)
        else:
            out.append(re.sub(r'(^|[\s(\[;:=])-(\d)', r'\1$-$\2', part))
    return ''.join(out)


def finalize(paths):
    """Resolve the ⟪en¦ro⟫ values in the written decks (EN file first, RO file second); true minus signs in text."""
    for p, k in zip(paths, (1, 2)):
        s = open(p, encoding='utf-8').read()
        s = re.sub(r'⟪(.*?)¦(.*?)⟫', lambda m: m.group(k), s)
        i = s.index('\\begin{document}')
        s = s[:i] + '\n'.join(text_minus(l) for l in s[i:].split('\n'))
        s = s.replace('p = $<$', 'p $<$')
        open(p, 'w', encoding='utf-8').write(s)


def date(s, day=True):
    """'2026-09-18' -> 18 September 2026 / 18 septembrie 2026 (or month and year only)."""
    y, m, d = s[:10].split('-')
    m = int(m) - 1
    if day:
        return V2(f'{int(d)} {MONTHS_EN[m]} {y}', f'{int(d)} {MONTHS_RO[m]} {y}')
    return V2(f'{MONTHS_EN[m]} {y}', f'{MONTHS_RO[m]} {y}')


def quarter(s):
    """'2026-04-01' -> 2026Q2 / T2 2026."""
    y, m = int(s[:4]), int(s[5:7])
    q = (m - 1) // 3 + 1
    return V2(f'{y}Q{q}', f'T{q} {y}')


def pv(p, d=3):
    """A p-value in text mode: 3 decimals, or '< 0.001' (marked for the RO decimal comma)."""
    return '$<$\\,⁅0.001⁆' if p < 0.001 else '⁅' + f'{p:.{d}f}' + '⁆'


def load():
    with open(os.path.join(QL, 'ch4_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem4_results.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
FPP = 'https://otexts.com/fpp3/'
ESS = 'https://ec.europa.eu/eurostat/web/products-manuals-and-guidelines/w/ks-gq-24-012'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('FPP', FPP, 'Hyndman and Athanasopoulos (2021)', 'Hyndman și Athanasopoulos (2021)')
        + ref('FPPsar', FPP + 'seasonal-arima.html', 'FPP3, Section 9.9', 'FPP3, secțiunea 9.9')
        + ref('FPPcomplex', FPP + 'complexseasonality.html', 'FPP3, Section 12.1', 'FPP3, secțiunea 12.1')
        + ref('FPPcv', FPP + 'tscv.html', 'FPP3, Section 5.10', 'FPP3, secțiunea 5.10')
        + ref('FPPdhr', FPP + 'dhr.html', 'FPP3, Section 10.5', 'FPP3, secțiunea 10.5')
        + ref('BD', D_ + '10.1007/978-3-319-29854-2', 'Brockwell and Davis (2016)', 'Brockwell și Davis (2016)')
        + ref('BJ', D_ + '10.1002/9781118619193', 'Box, Jenkins and Reinsel (2008)', 'Box, Jenkins și Reinsel (2008)')
        + ref('BoxCox', D_ + '10.1111/j.2517-6161.1964.tb00553.x', 'Box and Cox (1964)', 'Box și Cox (1964)')
        + ref('HEGY', D_ + '10.1016/0304-4076(90)90080-D', 'Hylleberg, Engle, Granger and Yoo (1990)', 'Hylleberg, Engle, Granger și Yoo (1990)')
        + ref('CH', D_ + '10.1080/07350015.1995.10524598', 'Canova and Hansen (1995)', 'Canova și Hansen (1995)')
        + ref('OCSB', D_ + '10.1111/j.1468-0084.1988.mp50004002.x', 'Osborn, Chui, Smith and Birchenhall (1988)', 'Osborn, Chui, Smith și Birchenhall (1988)')
        + ref('Findley', D_ + '10.1080/07350015.1998.10524743', 'Findley et al. (1998)')
        + ref('Ladiray', D_ + '10.1007/978-1-4613-0175-2', 'Ladiray and Quenneville (2001)', 'Ladiray și Quenneville (2001)')
        + ref('Dagum', D_ + '10.1007/978-3-319-31822-6', 'Dagum and Bianconcini (2016)', 'Dagum și Bianconcini (2016)')
        + ref('ESS', ESS, 'Eurostat (2024)')
        + ref('MSTL', D_ + '10.1504/ijor.2025.143957', 'Bandara, Hyndman and Bergmeir (2025)', 'Bandara, Hyndman și Bergmeir (2025)')
        + ref('TBATS', D_ + '10.1198/jasa.2011.tm09771', 'De Livera, Hyndman and Snyder (2011)', 'De Livera, Hyndman și Snyder (2011)')
        + ref('Prophet', D_ + '10.1080/00031305.2017.1380080', 'Taylor and Letham (2018)', 'Taylor și Letham (2018)')
        + ref('DM', D_ + '10.1080/07350015.1995.10524599', 'Diebold and Mariano (1995)', 'Diebold și Mariano (1995)')
        + ref('HLN', D_ + '10.1016/S0169-2070(96)00719-4', 'Harvey, Leybourne and Newbold (1997)', 'Harvey, Leybourne și Newbold (1997)')
        + ref('Tashman', D_ + '10.1016/S0169-2070(00)00065-0', 'Tashman (2000)')
        + ref('BHK', D_ + '10.1016/j.csda.2017.11.003', 'Bergmeir, Hyndman and Koo (2018)', 'Bergmeir, Hyndman și Koo (2018)')
        + ref('HKo', D_ + '10.1016/j.ijforecast.2006.03.001', 'Hyndman and Koehler (2006)', 'Hyndman și Koehler (2006)')
        + ref('HK', D_ + '10.18637/jss.v027.i03', 'Hyndman and Khandakar (2008)', 'Hyndman și Khandakar (2008)')
        + ref('BG', D_ + '10.1057/jors.1969.103', 'Bates and Granger (1969)', 'Bates și Granger (1969)')
        + ref('SW', D_ + '10.1002/for.928', 'Stock and Watson (2004)', 'Stock și Watson (2004)')
        + ref('SmithWallis', D_ + '10.1111/j.1468-0084.2008.00541.x', 'Smith and Wallis (2009)', 'Smith și Wallis (2009)')
        + ref('WangComb', D_ + '10.1016/j.ijforecast.2022.11.005', 'Wang et al. (2023)')
        + ref('MFour', D_ + '10.1016/j.ijforecast.2019.04.014', 'Makridakis, Spiliotis and Assimakopoulos (2020)', 'Makridakis, Spiliotis și Assimakopoulos (2020)')
        + ref('GEF', D_ + '10.1016/j.ijforecast.2016.02.001', 'Hong et al. (2016)')
        + ref('Petro', D_ + '10.1016/j.ijforecast.2021.11.001', 'Petropoulos et al. (2022)')
        + ref('LB', D_ + '10.1093/biomet/65.2.297', 'Ljung and Box (1978)', 'Ljung și Box (1978)'))

BIB = {
    'BG': r'Bates, J. M., \& Granger, C. W. J. (1969). The combination of forecasts. \textit{Journal of the Operational Research Society}, 20(4), 451--468. \href{https://doi.org/10.1057/jors.1969.103}{doi:10.1057/jors.1969.103}',
    'MSTL': r'Bandara, K., Hyndman, R. J., \& Bergmeir, C. (2025). MSTL: A seasonal-trend decomposition algorithm for time series with multiple seasonal patterns. \textit{International Journal of Operational Research}, 52(1), 79--98. \href{https://doi.org/10.1504/ijor.2025.143957}{doi:10.1504/ijor.2025.143957}',
    'BHK': r'Bergmeir, C., Hyndman, R. J., \& Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. \textit{Computational Statistics \& Data Analysis}, 120, 70--83. \href{https://doi.org/10.1016/j.csda.2017.11.003}{doi:10.1016/j.csda.2017.11.003}',
    'BoxCox': r'Box, G. E. P., \& Cox, D. R. (1964). An analysis of transformations. \textit{Journal of the Royal Statistical Society, Series B}, 26(2), 211--243. \href{https://doi.org/10.1111/j.2517-6161.1964.tb00553.x}{doi:10.1111/j.2517-6161.1964.tb00553.x}',
    'BJ': r'Box, G. E. P., Jenkins, G. M., \& Reinsel, G. C. (2008). \textit{Time Series Analysis: Forecasting and Control} (4th ed.). Wiley. \href{https://doi.org/10.1002/9781118619193}{doi:10.1002/9781118619193}',
    'BD': r'Brockwell, P. J., \& Davis, R. A. (2016). \textit{Introduction to Time Series and Forecasting} (3rd ed.). Springer. \href{https://doi.org/10.1007/978-3-319-29854-2}{doi:10.1007/978-3-319-29854-2}',
    'CH': r'Canova, F., \& Hansen, B. E. (1995). Are seasonal patterns constant over time? A test for seasonal stability. \textit{Journal of Business \& Economic Statistics}, 13(3), 237--252. \href{https://doi.org/10.1080/07350015.1995.10524598}{doi:10.1080/07350015.1995.10524598}',
    'Dagum': r'Dagum, E. B., \& Bianconcini, S. (2016). \textit{Seasonal Adjustment Methods and Real Time Trend-Cycle Estimation}. Springer. \href{https://doi.org/10.1007/978-3-319-31822-6}{doi:10.1007/978-3-319-31822-6}',
    'TBATS': r'De Livera, A. M., Hyndman, R. J., \& Snyder, R. D. (2011). Forecasting time series with complex seasonal patterns using exponential smoothing. \textit{Journal of the American Statistical Association}, 106(496), 1513--1527. \href{https://doi.org/10.1198/jasa.2011.tm09771}{doi:10.1198/jasa.2011.tm09771}',
    'DM': r'Diebold, F. X., \& Mariano, R. S. (1995). Comparing predictive accuracy. \textit{Journal of Business \& Economic Statistics}, 13(3), 253--263. \href{https://doi.org/10.1080/07350015.1995.10524599}{doi:10.1080/07350015.1995.10524599}',
    'ESS': r'Eurostat (2024). \textit{ESS Guidelines on Seasonal Adjustment, 2024 edition}. Publications Office of the European Union. \href{' + ESS + '}{ec.europa.eu/eurostat}',
    'Findley': r'Findley, D. F., Monsell, B. C., Bell, W. R., Otto, M. C., \& Chen, B.-C. (1998). New capabilities and methods of the X-12-ARIMA seasonal-adjustment program. \textit{Journal of Business \& Economic Statistics}, 16(2), 127--152. \href{https://doi.org/10.1080/07350015.1998.10524743}{doi:10.1080/07350015.1998.10524743}',
    'HLN': r'Harvey, D., Leybourne, S., \& Newbold, P. (1997). Testing the equality of prediction mean squared errors. \textit{International Journal of Forecasting}, 13(2), 281--291. \href{https://doi.org/10.1016/S0169-2070(96)00719-4}{doi:10.1016/S0169-2070(96)00719-4}',
    'GEF': r'Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A., \& Hyndman, R. J. (2016). Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond. \textit{International Journal of Forecasting}, 32(3), 896--913. \href{https://doi.org/10.1016/j.ijforecast.2016.02.001}{doi:10.1016/j.ijforecast.2016.02.001}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'HEGY': r'Hylleberg, S., Engle, R. F., Granger, C. W. J., \& Yoo, B. S. (1990). Seasonal integration and cointegration. \textit{Journal of Econometrics}, 44(1--2), 215--238. \href{https://doi.org/10.1016/0304-4076(90)90080-D}{doi:10.1016/0304-4076(90)90080-D}',
    'FPP': r'Hyndman, R. J., \& Athanasopoulos, G. (2021). \textit{Forecasting: Principles and Practice} (3rd ed.). OTexts. \href{https://otexts.com/fpp3/}{otexts.com/fpp3}',
    'HK': r'Hyndman, R. J., \& Khandakar, Y. (2008). Automatic time series forecasting: The forecast package for R. \textit{Journal of Statistical Software}, 27(3), 1--22. \href{https://doi.org/10.18637/jss.v027.i03}{doi:10.18637/jss.v027.i03}',
    'HKo': r'Hyndman, R. J., \& Koehler, A. B. (2006). Another look at measures of forecast accuracy. \textit{International Journal of Forecasting}, 22(4), 679--688. \href{https://doi.org/10.1016/j.ijforecast.2006.03.001}{doi:10.1016/j.ijforecast.2006.03.001}',
    'Ladiray': r'Ladiray, D., \& Quenneville, B. (2001). \textit{Seasonal Adjustment with the X-11 Method}. Lecture Notes in Statistics 158. Springer. \href{https://doi.org/10.1007/978-1-4613-0175-2}{doi:10.1007/978-1-4613-0175-2}',
    'LB': r'Ljung, G. M., \& Box, G. E. P. (1978). On a measure of lack of fit in time series models. \textit{Biometrika}, 65(2), 297--303. \href{https://doi.org/10.1093/biomet/65.2.297}{doi:10.1093/biomet/65.2.297}',
    'MFour': r'Makridakis, S., Spiliotis, E., \& Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. \textit{International Journal of Forecasting}, 36(1), 54--74. \href{https://doi.org/10.1016/j.ijforecast.2019.04.014}{doi:10.1016/j.ijforecast.2019.04.014}',
    'OCSB': r'Osborn, D. R., Chui, A. P. L., Smith, J. P., \& Birchenhall, C. R. (1988). Seasonality and the order of integration for consumption. \textit{Oxford Bulletin of Economics and Statistics}, 50(4), 361--377. \href{https://doi.org/10.1111/j.1468-0084.1988.mp50004002.x}{doi:10.1111/j.1468-0084.1988.mp50004002.x}',
    'Petro': r'Petropoulos, F., et al. (2022). Forecasting: theory and practice. \textit{International Journal of Forecasting}, 38(3), 705--871. \href{https://doi.org/10.1016/j.ijforecast.2021.11.001}{doi:10.1016/j.ijforecast.2021.11.001}',
    'SmithWallis': r'Smith, J., \& Wallis, K. F. (2009). A simple explanation of the forecast combination puzzle. \textit{Oxford Bulletin of Economics and Statistics}, 71(3), 331--355. \href{https://doi.org/10.1111/j.1468-0084.2008.00541.x}{doi:10.1111/j.1468-0084.2008.00541.x}',
    'SW': r'Stock, J. H., \& Watson, M. W. (2004). Combination forecasts of output growth in a seven-country data set. \textit{Journal of Forecasting}, 23(6), 405--430. \href{https://doi.org/10.1002/for.928}{doi:10.1002/for.928}',
    'Tashman': r'Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. \textit{International Journal of Forecasting}, 16(4), 437--450. \href{https://doi.org/10.1016/S0169-2070(00)00065-0}{doi:10.1016/S0169-2070(00)00065-0}',
    'Prophet': r'Taylor, S. J., \& Letham, B. (2018). Forecasting at scale. \textit{The American Statistician}, 72(1), 37--45. \href{https://doi.org/10.1080/00031305.2017.1380080}{doi:10.1080/00031305.2017.1380080}',
    'WangComb': r'Wang, X., Hyndman, R. J., Li, F., \& Kang, Y. (2023). Forecast combinations: An over 50-year review. \textit{International Journal of Forecasting}, 39(4), 1518--1547. \href{https://doi.org/10.1016/j.ijforecast.2022.11.005}{doi:10.1016/j.ijforecast.2022.11.005}',
}
ORDER = ['MSTL', 'BG', 'BHK', 'BoxCox', 'BJ', 'BD', 'CH', 'Dagum', 'TBATS', 'DM', 'ESS', 'Findley', 'HLN', 'GEF', 'HP', 'HEGY',
         'FPP', 'HK', 'HKo', 'Ladiray', 'LB', 'MFour', 'OCSB', 'Petro', 'SmithWallis', 'SW', 'Tashman', 'Prophet', 'WangComb']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
