r"""
ch8_common.py -- shared helpers of the Chapter 8 generators (lecture and seminar), TSA
=====================================================================================
Numbers from Quantlets/Ch_08/ch8_numbers.json (generate_all_charts.py) and sem8_results.json (seminar8.py);
the clickable citations of Chapter 8 (DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401
from ch1_common import T, V2, date, pv, MONTHS_EN, MONTHS_RO   # noqa: E402,F401
from ch6_common import finalize   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_08')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_08'


def month(s):
    y, m = int(s[:4]), int(s[5:7]) - 1
    return V2(f'{MONTHS_EN[m]} {y}', f'{MONTHS_RO[m]} {y}')


def load():
    with open(os.path.join(QL, 'ch8_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem8_results.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('BDtm', D_ + '10.1007/978-1-4419-0320-4', 'Brockwell and Davis (1991)', 'Brockwell și Davis (1991)')
        + ref('Beran', D_ + '10.1007/978-3-642-35512-7', 'Beran et al.\\ (2013)')
        + ref('Baillie', D_ + '10.1016/0304-4076(95)01732-1', 'Baillie (1996)')
        + ref('Graves', D_ + '10.3390/e19090437', 'Graves et al.\\ (2017)')
        + ref('Hurst', D_ + '10.1061/TACEAT.0006518', 'Hurst (1951)')
        + ref('MVN', D_ + '10.1137/1010093', 'Mandelbrot and Van Ness (1968)', 'Mandelbrot și Van Ness (1968)')
        + ref('Noah', D_ + '10.1029/WR004i005p00909', 'Mandelbrot and Wallis (1968)', 'Mandelbrot și Wallis (1968)')
        + ref('MW', D_ + '10.1029/WR005i005p00967', 'Mandelbrot and Wallis (1969)', 'Mandelbrot și Wallis (1969)')
        + ref('GJ', D_ + '10.1111/j.1467-9892.1980.tb00297.x', 'Granger and Joyeux (1980)', 'Granger și Joyeux (1980)')
        + ref('Hosking', D_ + '10.1093/biomet/68.1.165', 'Hosking (1981)')
        + ref('Lo', D_ + '10.2307/2938368', 'Lo (1991)')
        + ref('Peng', D_ + '10.1103/PhysRevE.49.1685', 'Peng et al.\\ (1994)')
        + ref('GPH', D_ + '10.1111/j.1467-9892.1983.tb00371.x', 'Geweke and Porter-Hudak (1983)', 'Geweke și Porter-Hudak (1983)')
        + ref('Rob', D_ + '10.1214/aos/1176324317', 'Robinson (1995)')
        + ref('Velasco', D_ + '10.1111/1467-9892.00127', 'Velasco (1999)')
        + ref('Sowell', D_ + '10.1016/0304-4076(92)90084-5', 'Sowell (1992)')
        + ref('FT', D_ + '10.1214/aos/1176349936', 'Fox and Taqqu (1986)', 'Fox și Taqqu (1986)')
        + ref('Ray', D_ + '10.1111/j.1467-9892.1993.tb00161.x', 'Ray (1993)')
        + ref('HW', D_ + '10.1080/07350015.1995.10524577', 'Hassler and Wolters (1995)', 'Hassler și Wolters (1995)')
        + ref('DGE', D_ + '10.1016/0927-5398(93)90006-D', 'Ding, Granger and Engle (1993)', 'Ding, Granger și Engle (1993)')
        + ref('BBM', D_ + '10.1016/S0304-4076(95)01749-6', 'Baillie, Bollerslev and Mikkelsen (1996)', 'Baillie, Bollerslev și Mikkelsen (1996)')
        + ref('ABDL', D_ + '10.1111/1468-0262.00418', 'Andersen et al.\\ (2003)')
        + ref('Corsi', D_ + '10.1093/jjfinec/nbp001', 'Corsi (2009)')
        + ref('DI', D_ + '10.1016/S0304-4076(01)00073-2', 'Diebold and Inoue (2001)', 'Diebold și Inoue (2001)')
        + ref('GH', D_ + '10.1016/j.jempfin.2003.03.001', 'Granger and Hyung (2004)', 'Granger și Hyung (2004)')
        + ref('Cobb', D_ + '10.1093/biomet/65.2.243', 'Cobb (1978)')
        + ref('Weron', D_ + '10.1016/S0378-4371(02)00961-5', 'Weron (2002)')
        + ref('CL', D_ + '10.1080/07350015.1993.10509936', 'Cheung and Lai (1993)', 'Cheung și Lai (1993)'))

BIB = {
    'ABDL': r'Andersen, T. G., Bollerslev, T., Diebold, F. X., \& Labys, P. (2003). Modeling and forecasting realized volatility. \textit{Econometrica}, 71(2), 579--625. \href{https://doi.org/10.1111/1468-0262.00418}{doi:10.1111/1468-0262.00418}',
    'Baillie': r'Baillie, R. T. (1996). Long memory processes and fractional integration in econometrics. \textit{Journal of Econometrics}, 73(1), 5--59. \href{https://doi.org/10.1016/0304-4076(95)01732-1}{doi:10.1016/0304-4076(95)01732-1}',
    'BBM': r'Baillie, R. T., Bollerslev, T., \& Mikkelsen, H. O. (1996). Fractionally integrated generalized autoregressive conditional heteroskedasticity. \textit{Journal of Econometrics}, 74(1), 3--30. \href{https://doi.org/10.1016/S0304-4076(95)01749-6}{doi:10.1016/S0304-4076(95)01749-6}',
    'Beran': r'Beran, J., Feng, Y., Ghosh, S., \& Kulik, R. (2013). \textit{Long-Memory Processes: Probabilistic Properties and Statistical Methods}. Springer. \href{https://doi.org/10.1007/978-3-642-35512-7}{doi:10.1007/978-3-642-35512-7}',
    'BDtm': r'Brockwell, P. J., \& Davis, R. A. (1991). \textit{Time Series: Theory and Methods} (2nd ed.). Springer. \href{https://doi.org/10.1007/978-1-4419-0320-4}{doi:10.1007/978-1-4419-0320-4}',
    'CL': r'Cheung, Y.-W., \& Lai, K. S. (1993). A fractional cointegration analysis of purchasing power parity. \textit{Journal of Business \& Economic Statistics}, 11(1), 103--112. \href{https://doi.org/10.1080/07350015.1993.10509936}{doi:10.1080/07350015.1993.10509936}',
    'Cobb': r'Cobb, G. W. (1978). The problem of the Nile: Conditional solution to a changepoint problem. \textit{Biometrika}, 65(2), 243--251. \href{https://doi.org/10.1093/biomet/65.2.243}{doi:10.1093/biomet/65.2.243}',
    'Corsi': r'Corsi, F. (2009). A simple approximate long-memory model of realized volatility. \textit{Journal of Financial Econometrics}, 7(2), 174--196. \href{https://doi.org/10.1093/jjfinec/nbp001}{doi:10.1093/jjfinec/nbp001}',
    'DI': r'Diebold, F. X., \& Inoue, A. (2001). Long memory and regime switching. \textit{Journal of Econometrics}, 105(1), 131--159. \href{https://doi.org/10.1016/S0304-4076(01)00073-2}{doi:10.1016/S0304-4076(01)00073-2}',
    'DGE': r'Ding, Z., Granger, C. W. J., \& Engle, R. F. (1993). A long memory property of stock market returns and a new model. \textit{Journal of Empirical Finance}, 1(1), 83--106. \href{https://doi.org/10.1016/0927-5398(93)90006-D}{doi:10.1016/0927-5398(93)90006-D}',
    'FT': r'Fox, R., \& Taqqu, M. S. (1986). Large-sample properties of parameter estimates for strongly dependent stationary Gaussian time series. \textit{The Annals of Statistics}, 14(2), 517--532. \href{https://doi.org/10.1214/aos/1176349936}{doi:10.1214/aos/1176349936}',
    'GPH': r'Geweke, J., \& Porter-Hudak, S. (1983). The estimation and application of long memory time series models. \textit{Journal of Time Series Analysis}, 4(4), 221--238. \href{https://doi.org/10.1111/j.1467-9892.1983.tb00371.x}{doi:10.1111/j.1467-9892.1983.tb00371.x}',
    'GH': r'Granger, C. W. J., \& Hyung, N. (2004). Occasional structural breaks and long memory with an application to the S\&P 500 absolute stock returns. \textit{Journal of Empirical Finance}, 11(3), 399--421. \href{https://doi.org/10.1016/j.jempfin.2003.03.001}{doi:10.1016/j.jempfin.2003.03.001}',
    'GJ': r'Granger, C. W. J., \& Joyeux, R. (1980). An introduction to long-memory time series models and fractional differencing. \textit{Journal of Time Series Analysis}, 1(1), 15--29. \href{https://doi.org/10.1111/j.1467-9892.1980.tb00297.x}{doi:10.1111/j.1467-9892.1980.tb00297.x}',
    'Graves': r'Graves, T., Gramacy, R., Watkins, N., \& Franzke, C. (2017). A brief history of long memory: Hurst, Mandelbrot and the road to ARFIMA, 1951--1980. \textit{Entropy}, 19(9), 437. \href{https://doi.org/10.3390/e19090437}{doi:10.3390/e19090437}',
    'HW': r'Hassler, U., \& Wolters, J. (1995). Long memory in inflation rates: International evidence. \textit{Journal of Business \& Economic Statistics}, 13(1), 37--45. \href{https://doi.org/10.1080/07350015.1995.10524577}{doi:10.1080/07350015.1995.10524577}',
    'Hosking': r'Hosking, J. R. M. (1981). Fractional differencing. \textit{Biometrika}, 68(1), 165--176. \href{https://doi.org/10.1093/biomet/68.1.165}{doi:10.1093/biomet/68.1.165}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'Hurst': r'Hurst, H. E. (1951). Long-term storage capacity of reservoirs. \textit{Transactions of the American Society of Civil Engineers}, 116(1), 770--799. \href{https://doi.org/10.1061/TACEAT.0006518}{doi:10.1061/TACEAT.0006518}',
    'Lo': r'Lo, A. W. (1991). Long-term memory in stock market prices. \textit{Econometrica}, 59(5), 1279--1313. \href{https://doi.org/10.2307/2938368}{doi:10.2307/2938368}',
    'MVN': r'Mandelbrot, B. B., \& Van Ness, J. W. (1968). Fractional Brownian motions, fractional noises and applications. \textit{SIAM Review}, 10(4), 422--437. \href{https://doi.org/10.1137/1010093}{doi:10.1137/1010093}',
    'Noah': r'Mandelbrot, B. B., \& Wallis, J. R. (1968). Noah, Joseph, and operational hydrology. \textit{Water Resources Research}, 4(5), 909--918. \href{https://doi.org/10.1029/WR004i005p00909}{doi:10.1029/WR004i005p00909}',
    'MW': r'Mandelbrot, B. B., \& Wallis, J. R. (1969). Robustness of the rescaled range R/S in the measurement of noncyclic long run statistical dependence. \textit{Water Resources Research}, 5(5), 967--988. \href{https://doi.org/10.1029/WR005i005p00967}{doi:10.1029/WR005i005p00967}',
    'Peng': r'Peng, C.-K., Buldyrev, S. V., Havlin, S., Simons, M., Stanley, H. E., \& Goldberger, A. L. (1994). Mosaic organization of DNA nucleotides. \textit{Physical Review E}, 49(2), 1685--1689. \href{https://doi.org/10.1103/PhysRevE.49.1685}{doi:10.1103/PhysRevE.49.1685}',
    'Ray': r'Ray, B. K. (1993). Modeling long-memory processes for optimal long-range prediction. \textit{Journal of Time Series Analysis}, 14(5), 511--525. \href{https://doi.org/10.1111/j.1467-9892.1993.tb00161.x}{doi:10.1111/j.1467-9892.1993.tb00161.x}',
    'Rob': r'Robinson, P. M. (1995). Gaussian semiparametric estimation of long range dependence. \textit{The Annals of Statistics}, 23(5), 1630--1661. \href{https://doi.org/10.1214/aos/1176324317}{doi:10.1214/aos/1176324317}',
    'Sowell': r'Sowell, F. (1992). Maximum likelihood estimation of stationary univariate fractionally integrated time series models. \textit{Journal of Econometrics}, 53(1--3), 165--188. \href{https://doi.org/10.1016/0304-4076(92)90084-5}{doi:10.1016/0304-4076(92)90084-5}',
    'Velasco': r'Velasco, C. (1999). Gaussian semiparametric estimation of non-stationary time series. \textit{Journal of Time Series Analysis}, 20(1), 87--127. \href{https://doi.org/10.1111/1467-9892.00127}{doi:10.1111/1467-9892.00127}',
    'Weron': r'Weron, R. (2002). Estimating long-range dependence: Finite sample properties and confidence intervals. \textit{Physica A}, 312(1--2), 285--299. \href{https://doi.org/10.1016/S0378-4371(02)00961-5}{doi:10.1016/S0378-4371(02)00961-5}',
}
ORDER = ['ABDL', 'Baillie', 'BBM', 'Beran', 'BDtm', 'CL', 'Cobb', 'Corsi', 'DI', 'DGE', 'FT', 'GPH', 'GH', 'GJ', 'Graves',
         'HW', 'Hosking', 'HP', 'Hurst', 'Lo', 'MVN', 'Noah', 'MW', 'Peng', 'Ray', 'Rob', 'Sowell', 'Velasco', 'Weron']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
