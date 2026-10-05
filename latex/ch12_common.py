r"""
ch12_common.py -- shared helpers of the Chapter 12 generator (spectral analysis), TSA
====================================================================================
Numbers from Quantlets/Ch_12/ch12_numbers.json (generate_all_charts.py); the clickable citations of Chapter 12
(DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401
from ch1_common import T, V2, finalize, date, pv, MONTHS_EN, MONTHS_RO   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_12')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_12'


def load():
    with open(os.path.join(QL, 'ch12_numbers.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('SS', D_ + '10.1007/978-3-319-52452-8', 'Shumway and Stoffer (2017)', 'Shumway și Stoffer (2017)')
        + ref('BDtm', D_ + '10.1007/978-1-4419-0320-4', 'Brockwell and Davis (1991)', 'Brockwell și Davis (1991)')
        + ref('HamTS', D_ + '10.1515/9780691218632', 'Hamilton (1994)')
        + ref('Schuster', D_ + '10.1029/TM003i001p00013', 'Schuster (1898)')
        + ref('Yule', D_ + '10.1098/rsta.1927.0007', 'Yule (1927)')
        + ref('Fisher', D_ + '10.1098/rspa.1929.0151', 'Fisher (1929)')
        + ref('Bartlett', D_ + '10.1093/biomet/37.1-2.1', 'Bartlett (1950)')
        + ref('BT', D_ + '10.1002/j.1538-7305.1958.tb03874.x', 'Blackman and Tukey (1958)', 'Blackman și Tukey (1958)')
        + ref('CT', D_ + '10.1090/S0025-5718-1965-0178586-1', 'Cooley and Tukey (1965)', 'Cooley și Tukey (1965)')
        + ref('Granger', D_ + '10.2307/1909859', 'Granger (1966)')
        + ref('Welch', D_ + '10.1109/TAU.1967.1161901', 'Welch (1967)')
        + ref('GPH', D_ + '10.1111/j.1467-9892.1983.tb00371.x', 'Geweke and Porter-Hudak (1983)', 'Geweke și Porter-Hudak (1983)')
        + ref('HPf', D_ + '10.2307/2953682', 'Hodrick and Prescott (1997)', 'Hodrick și Prescott (1997)')
        + ref('TC', D_ + '10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2', 'Torrence and Compo (1998)', 'Torrence și Compo (1998)')
        + ref('BK', D_ + '10.1162/003465399558454', 'Baxter and King (1999)', 'Baxter și King (1999)')
        + ref('HamHP', D_ + '10.1162/rest_a_00706', 'Hamilton (2018)'))

BIB = {
    'Bartlett': r'Bartlett, M. S. (1950). Periodogram analysis and continuous spectra. \textit{Biometrika}, 37(1--2), 1--16. \href{https://doi.org/10.1093/biomet/37.1-2.1}{doi:10.1093/biomet/37.1-2.1}',
    'BK': r'Baxter, M., \& King, R. G. (1999). Measuring business cycles: Approximate band-pass filters for economic time series. \textit{Review of Economics and Statistics}, 81(4), 575--593. \href{https://doi.org/10.1162/003465399558454}{doi:10.1162/003465399558454}',
    'BT': r'Blackman, R. B., \& Tukey, J. W. (1958). The measurement of power spectra from the point of view of communications engineering, Part I. \textit{Bell System Technical Journal}, 37(1), 185--282. \href{https://doi.org/10.1002/j.1538-7305.1958.tb03874.x}{doi:10.1002/j.1538-7305.1958.tb03874.x}',
    'BDtm': r'Brockwell, P. J., \& Davis, R. A. (1991). \textit{Time Series: Theory and Methods} (2nd ed.). Springer. \href{https://doi.org/10.1007/978-1-4419-0320-4}{doi:10.1007/978-1-4419-0320-4}',
    'CT': r'Cooley, J. W., \& Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. \textit{Mathematics of Computation}, 19(90), 297--301. \href{https://doi.org/10.1090/S0025-5718-1965-0178586-1}{doi:10.1090/S0025-5718-1965-0178586-1}',
    'Fisher': r'Fisher, R. A. (1929). Tests of significance in harmonic analysis. \textit{Proceedings of the Royal Society of London, Series A}, 125(796), 54--59. \href{https://doi.org/10.1098/rspa.1929.0151}{doi:10.1098/rspa.1929.0151}',
    'GPH': r'Geweke, J., \& Porter-Hudak, S. (1983). The estimation and application of long memory time series models. \textit{Journal of Time Series Analysis}, 4(4), 221--238. \href{https://doi.org/10.1111/j.1467-9892.1983.tb00371.x}{doi:10.1111/j.1467-9892.1983.tb00371.x}',
    'Granger': r'Granger, C. W. J. (1966). The typical spectral shape of an economic variable. \textit{Econometrica}, 34(1), 150--161. \href{https://doi.org/10.2307/1909859}{doi:10.2307/1909859}',
    'Ham': r'Hamilton, J. D. (1994). \textit{Time Series Analysis}. Princeton University Press. \href{https://doi.org/10.1515/9780691218632}{doi:10.1515/9780691218632}',
    'HamHP': r'Hamilton, J. D. (2018). Why you should never use the Hodrick--Prescott filter. \textit{Review of Economics and Statistics}, 100(5), 831--843. \href{https://doi.org/10.1162/rest_a_00706}{doi:10.1162/rest\_a\_00706}',
    'HPf': r'Hodrick, R. J., \& Prescott, E. C. (1997). Postwar U.S. business cycles: An empirical investigation. \textit{Journal of Money, Credit and Banking}, 29(1), 1--16. \href{https://doi.org/10.2307/2953682}{doi:10.2307/2953682}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'Schuster': r'Schuster, A. (1898). On the investigation of hidden periodicities with application to a supposed 26 day period of meteorological phenomena. \textit{Terrestrial Magnetism}, 3(1), 13--41. \href{https://doi.org/10.1029/TM003i001p00013}{doi:10.1029/TM003i001p00013}',
    'SS': r'Shumway, R. H., \& Stoffer, D. S. (2017). \textit{Time Series Analysis and Its Applications: With R Examples} (4th ed.). Springer. \href{https://doi.org/10.1007/978-3-319-52452-8}{doi:10.1007/978-3-319-52452-8}',
    'TC': r'Torrence, C., \& Compo, G. P. (1998). A practical guide to wavelet analysis. \textit{Bulletin of the American Meteorological Society}, 79(1), 61--78. \href{https://doi.org/10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2}{doi:10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2}',
    'Welch': r'Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms. \textit{IEEE Transactions on Audio and Electroacoustics}, 15(2), 70--73. \href{https://doi.org/10.1109/TAU.1967.1161901}{doi:10.1109/TAU.1967.1161901}',
    'Yule': r"Yule, G. U. (1927). On a method of investigating periodicities in disturbed series, with special reference to Wolfer's sunspot numbers. \textit{Philosophical Transactions of the Royal Society of London, Series A}, 226, 267--298. \href{https://doi.org/10.1098/rsta.1927.0007}{doi:10.1098/rsta.1927.0007}",
}
ORDER = ['Bartlett', 'BK', 'BT', 'BDtm', 'CT', 'Fisher', 'GPH', 'Granger', 'Ham', 'HamHP', 'HPf', 'HP', 'Schuster', 'SS',
         'TC', 'Welch', 'Yule']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
