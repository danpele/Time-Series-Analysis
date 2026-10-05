r"""
ch9_common.py -- shared helpers of the Chapter 9 generators (lecture and seminar), TSA
=====================================================================================
Numbers from Quantlets/Ch_09/ch9_numbers.json (generate_all_charts.py) and sem9_results.json (seminar9.py);
the clickable citations of Chapter 9 (DOIs checked against Crossref, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401
from ch1_common import T, V2, finalize, date, quarter, pv, MONTHS_EN, MONTHS_RO   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_09')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_09'


def month(s):
    y, m = int(s[:4]), int(s[5:7]) - 1
    return V2(f'{MONTHS_EN[m]} {y}', f'{MONTHS_RO[m]} {y}')


def load():
    with open(os.path.join(QL, 'ch9_numbers.json')) as f:
        return json.load(f)


def load_sem():
    with open(os.path.join(QL, 'sem9_results.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
LGBM = 'https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('FPP', 'https://otexts.com/fpp3/nnetar.html', 'Hyndman and Athanasopoulos (2021)', 'Hyndman și Athanasopoulos (2021)')
        + ref('ESL', D_ + '10.1007/978-0-387-84858-7', 'Hastie, Tibshirani and Friedman (2009)', 'Hastie, Tibshirani și Friedman (2009)')
        + ref('HK', D_ + '10.1016/j.ijforecast.2006.03.001', 'Hyndman and Koehler (2006)', 'Hyndman și Koehler (2006)')
        + ref('DM', D_ + '10.1080/07350015.1995.10524599', 'Diebold and Mariano (1995)', 'Diebold și Mariano (1995)')
        + ref('HLN', D_ + '10.1016/S0169-2070(96)00719-4', 'Harvey, Leybourne and Newbold (1997)', 'Harvey, Leybourne și Newbold (1997)')
        + ref('BHK', D_ + '10.1016/j.csda.2017.11.003', 'Bergmeir, Hyndman and Koo (2018)', 'Bergmeir, Hyndman și Koo (2018)')
        + ref('Kaufman', D_ + '10.1145/2382577.2382579', 'Kaufman et al. (2012)')
        + ref('HoerlK', D_ + '10.1080/00401706.1970.10488634', 'Hoerl and Kennard (1970)', 'Hoerl și Kennard (1970)')
        + ref('Tib', D_ + '10.1111/j.2517-6161.1996.tb02080.x', 'Tibshirani (1996)')
        + ref('Breiman', D_ + '10.1023/A:1010933404324', 'Breiman (2001)')
        + ref('Friedman', D_ + '10.1214/aos/1013203451', 'Friedman (2001)')
        + ref('XGB', D_ + '10.1145/2939672.2939785', 'Chen and Guestrin (2016)', 'Chen și Guestrin (2016)')
        + ref('LGBM', LGBM, 'Ke et al. (2017)')
        + ref('Rumelhart', D_ + '10.1038/323533a0', 'Rumelhart, Hinton and Williams (1986)', 'Rumelhart, Hinton și Williams (1986)')
        + ref('Hornik', D_ + '10.1016/0893-6080(89)90020-8', 'Hornik, Stinchcombe and White (1989)', 'Hornik, Stinchcombe și White (1989)')
        + ref('LSTM', D_ + '10.1162/neco.1997.9.8.1735', 'Hochreiter and Schmidhuber (1997)', 'Hochreiter și Schmidhuber (1997)')
        + ref('HBB', D_ + '10.1016/j.ijforecast.2020.06.008', 'Hewamalage, Bergmeir and Bandara (2021)', 'Hewamalage, Bergmeir și Bandara (2021)')
        + ref('LZ', D_ + '10.1098/rsta.2020.0209', 'Lim and Zohren (2021)', 'Lim și Zohren (2021)')
        + ref('DeepAR', D_ + '10.1016/j.ijforecast.2019.07.001', 'Salinas et al. (2020)')
        + ref('BenTaieb', D_ + '10.1016/j.eswa.2012.01.039', 'Ben Taieb et al. (2012)')
        + ref('MMH', D_ + '10.1016/j.ijforecast.2021.03.004', 'Montero-Manso and Hyndman (2021)', 'Montero-Manso și Hyndman (2021)')
        + ref('Janus', D_ + '10.1016/j.ijforecast.2019.05.008', 'Januschowski et al. (2020)')
        + ref('KB', D_ + '10.2307/1913643', 'Koenker and Bassett (1978)', 'Koenker și Bassett (1978)')
        + ref('Lei', D_ + '10.1080/01621459.2017.1307116', 'Lei et al. (2018)')
        + ref('MSAplos', D_ + '10.1371/journal.pone.0194889', 'Makridakis, Spiliotis and Assimakopoulos (2018a)', 'Makridakis, Spiliotis și Assimakopoulos (2018a)')
        + ref('MfourA', D_ + '10.1016/j.ijforecast.2018.06.001', 'Makridakis, Spiliotis and Assimakopoulos (2018b)', 'Makridakis, Spiliotis și Assimakopoulos (2018b)')
        + ref('Mfour', D_ + '10.1016/j.ijforecast.2019.04.014', 'Makridakis, Spiliotis and Assimakopoulos (2020)', 'Makridakis, Spiliotis și Assimakopoulos (2020)')
        + ref('Mfive', D_ + '10.1016/j.ijforecast.2021.11.013', 'Makridakis, Spiliotis and Assimakopoulos (2022)', 'Makridakis, Spiliotis și Assimakopoulos (2022)')
        + ref('Smyl', D_ + '10.1016/j.ijforecast.2019.03.017', 'Smyl (2020)')
        + ref('Medeiros', D_ + '10.1080/07350015.2019.1637745', 'Medeiros et al. (2021)')
        + ref('Corsi', D_ + '10.1093/jjfinec/nbp001', 'Corsi (2009)')
        + ref('Patton', D_ + '10.1016/j.jeconom.2010.03.034', 'Patton (2011)')
        + ref('GK', D_ + '10.1086/296072', 'Garman and Klass (1980)', 'Garman și Klass (1980)')
        + ref('GKX', D_ + '10.1093/rfs/hhaa009', 'Gu, Kelly and Xiu (2020)', 'Gu, Kelly și Xiu (2020)'))

BIB = {
    'BenTaieb': r'Ben Taieb, S., Bontempi, G., Atiya, A. F., \& Sorjamaa, A. (2012). A review and comparison of strategies for multi-step ahead time series forecasting based on the NN5 forecasting competition. \textit{Expert Systems with Applications}, 39(8), 7067--7083. \href{https://doi.org/10.1016/j.eswa.2012.01.039}{doi:10.1016/j.eswa.2012.01.039}',
    'BHK': r'Bergmeir, C., Hyndman, R. J., \& Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. \textit{Computational Statistics \& Data Analysis}, 120, 70--83. \href{https://doi.org/10.1016/j.csda.2017.11.003}{doi:10.1016/j.csda.2017.11.003}',
    'Breiman': r'Breiman, L. (2001). Random forests. \textit{Machine Learning}, 45(1), 5--32. \href{https://doi.org/10.1023/A:1010933404324}{doi:10.1023/A:1010933404324}',
    'XGB': r'Chen, T., \& Guestrin, C. (2016). XGBoost: A scalable tree boosting system. \textit{Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining}, 785--794. \href{https://doi.org/10.1145/2939672.2939785}{doi:10.1145/2939672.2939785}',
    'Corsi': r'Corsi, F. (2009). A simple approximate long-memory model of realized volatility. \textit{Journal of Financial Econometrics}, 7(2), 174--196. \href{https://doi.org/10.1093/jjfinec/nbp001}{doi:10.1093/jjfinec/nbp001}',
    'DM': r'Diebold, F. X., \& Mariano, R. S. (1995). Comparing predictive accuracy. \textit{Journal of Business \& Economic Statistics}, 13(3), 253--263. \href{https://doi.org/10.1080/07350015.1995.10524599}{doi:10.1080/07350015.1995.10524599}',
    'Friedman': r'Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. \textit{The Annals of Statistics}, 29(5), 1189--1232. \href{https://doi.org/10.1214/aos/1013203451}{doi:10.1214/aos/1013203451}',
    'GK': r'Garman, M. B., \& Klass, M. J. (1980). On the estimation of security price volatilities from historical data. \textit{The Journal of Business}, 53(1), 67--78. \href{https://doi.org/10.1086/296072}{doi:10.1086/296072}',
    'GKX': r'Gu, S., Kelly, B., \& Xiu, D. (2020). Empirical asset pricing via machine learning. \textit{The Review of Financial Studies}, 33(5), 2223--2273. \href{https://doi.org/10.1093/rfs/hhaa009}{doi:10.1093/rfs/hhaa009}',
    'HLN': r'Harvey, D., Leybourne, S., \& Newbold, P. (1997). Testing the equality of prediction mean squared errors. \textit{International Journal of Forecasting}, 13(2), 281--291. \href{https://doi.org/10.1016/S0169-2070(96)00719-4}{doi:10.1016/S0169-2070(96)00719-4}',
    'ESL': r'Hastie, T., Tibshirani, R., \& Friedman, J. (2009). \textit{The Elements of Statistical Learning} (2nd ed.). Springer. \href{https://doi.org/10.1007/978-0-387-84858-7}{doi:10.1007/978-0-387-84858-7}',
    'HBB': r'Hewamalage, H., Bergmeir, C., \& Bandara, K. (2021). Recurrent neural networks for time series forecasting: Current status and future directions. \textit{International Journal of Forecasting}, 37(1), 388--427. \href{https://doi.org/10.1016/j.ijforecast.2020.06.008}{doi:10.1016/j.ijforecast.2020.06.008}',
    'LSTM': r'Hochreiter, S., \& Schmidhuber, J. (1997). Long short-term memory. \textit{Neural Computation}, 9(8), 1735--1780. \href{https://doi.org/10.1162/neco.1997.9.8.1735}{doi:10.1162/neco.1997.9.8.1735}',
    'HoerlK': r'Hoerl, A. E., \& Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. \textit{Technometrics}, 12(1), 55--67. \href{https://doi.org/10.1080/00401706.1970.10488634}{doi:10.1080/00401706.1970.10488634}',
    'Hornik': r'Hornik, K., Stinchcombe, M., \& White, H. (1989). Multilayer feedforward networks are universal approximators. \textit{Neural Networks}, 2(5), 359--366. \href{https://doi.org/10.1016/0893-6080(89)90020-8}{doi:10.1016/0893-6080(89)90020-8}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'FPP': r'Hyndman, R. J., \& Athanasopoulos, G. (2021). \textit{Forecasting: Principles and Practice} (3rd ed.), Section 12.4, Neural network models. OTexts. \href{https://otexts.com/fpp3/nnetar.html}{otexts.com/fpp3/nnetar.html}',
    'HK': r'Hyndman, R. J., \& Koehler, A. B. (2006). Another look at measures of forecast accuracy. \textit{International Journal of Forecasting}, 22(4), 679--688. \href{https://doi.org/10.1016/j.ijforecast.2006.03.001}{doi:10.1016/j.ijforecast.2006.03.001}',
    'Janus': r'Januschowski, T., Gasthaus, J., Wang, Y., Salinas, D., Flunkert, V., Bohlke-Schneider, M., \& Callot, L. (2020). Criteria for classifying forecasting methods. \textit{International Journal of Forecasting}, 36(1), 167--177. \href{https://doi.org/10.1016/j.ijforecast.2019.05.008}{doi:10.1016/j.ijforecast.2019.05.008}',
    'Kaufman': r'Kaufman, S., Rosset, S., Perlich, C., \& Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. \textit{ACM Transactions on Knowledge Discovery from Data}, 6(4), 1--21. \href{https://doi.org/10.1145/2382577.2382579}{doi:10.1145/2382577.2382579}',
    'LGBM': r'Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., \& Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. \textit{Advances in Neural Information Processing Systems}, 30. \href{' + LGBM + r'}{proceedings.neurips.cc}',
    'KB': r'Koenker, R., \& Bassett, G. (1978). Regression quantiles. \textit{Econometrica}, 46(1), 33--50. \href{https://doi.org/10.2307/1913643}{doi:10.2307/1913643}',
    'Lei': r"Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., \& Wasserman, L. (2018). Distribution-free predictive inference for regression. \textit{Journal of the American Statistical Association}, 113(523), 1094--1111. \href{https://doi.org/10.1080/01621459.2017.1307116}{doi:10.1080/01621459.2017.1307116}",
    'LZ': r'Lim, B., \& Zohren, S. (2021). Time-series forecasting with deep learning: A survey. \textit{Philosophical Transactions of the Royal Society A}, 379(2194), 20200209. \href{https://doi.org/10.1098/rsta.2020.0209}{doi:10.1098/rsta.2020.0209}',
    'MSAplos': r'Makridakis, S., Spiliotis, E., \& Assimakopoulos, V. (2018a). Statistical and machine learning forecasting methods: Concerns and ways forward. \textit{PLOS ONE}, 13(3), e0194889. \href{https://doi.org/10.1371/journal.pone.0194889}{doi:10.1371/journal.pone.0194889}',
    'MfourA': r'Makridakis, S., Spiliotis, E., \& Assimakopoulos, V. (2018b). The M4 Competition: Results, findings, conclusion and way forward. \textit{International Journal of Forecasting}, 34(4), 802--808. \href{https://doi.org/10.1016/j.ijforecast.2018.06.001}{doi:10.1016/j.ijforecast.2018.06.001}',
    'Mfour': r'Makridakis, S., Spiliotis, E., \& Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. \textit{International Journal of Forecasting}, 36(1), 54--74. \href{https://doi.org/10.1016/j.ijforecast.2019.04.014}{doi:10.1016/j.ijforecast.2019.04.014}',
    'Mfive': r'Makridakis, S., Spiliotis, E., \& Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. \textit{International Journal of Forecasting}, 38(4), 1346--1364. \href{https://doi.org/10.1016/j.ijforecast.2021.11.013}{doi:10.1016/j.ijforecast.2021.11.013}',
    'Medeiros': r'Medeiros, M. C., Vasconcelos, G. F. R., Veiga, \'A., \& Zilberman, E. (2021). Forecasting inflation in a data-rich environment: The benefits of machine learning methods. \textit{Journal of Business \& Economic Statistics}, 39(1), 98--119. \href{https://doi.org/10.1080/07350015.2019.1637745}{doi:10.1080/07350015.2019.1637745}',
    'MMH': r'Montero-Manso, P., \& Hyndman, R. J. (2021). Principles and algorithms for forecasting groups of time series: Locality and globality. \textit{International Journal of Forecasting}, 37(4), 1632--1653. \href{https://doi.org/10.1016/j.ijforecast.2021.03.004}{doi:10.1016/j.ijforecast.2021.03.004}',
    'Patton': r'Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. \textit{Journal of Econometrics}, 160(1), 246--256. \href{https://doi.org/10.1016/j.jeconom.2010.03.034}{doi:10.1016/j.jeconom.2010.03.034}',
    'Rumelhart': r'Rumelhart, D. E., Hinton, G. E., \& Williams, R. J. (1986). Learning representations by back-propagating errors. \textit{Nature}, 323(6088), 533--536. \href{https://doi.org/10.1038/323533a0}{doi:10.1038/323533a0}',
    'DeepAR': r'Salinas, D., Flunkert, V., Gasthaus, J., \& Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. \textit{International Journal of Forecasting}, 36(3), 1181--1191. \href{https://doi.org/10.1016/j.ijforecast.2019.07.001}{doi:10.1016/j.ijforecast.2019.07.001}',
    'Smyl': r'Smyl, S. (2020). A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting. \textit{International Journal of Forecasting}, 36(1), 75--85. \href{https://doi.org/10.1016/j.ijforecast.2019.03.017}{doi:10.1016/j.ijforecast.2019.03.017}',
    'Tib': r'Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. \textit{Journal of the Royal Statistical Society: Series B}, 58(1), 267--288. \href{https://doi.org/10.1111/j.2517-6161.1996.tb02080.x}{doi:10.1111/j.2517-6161.1996.tb02080.x}',
}
ORDER = ['BenTaieb', 'BHK', 'Breiman', 'XGB', 'Corsi', 'DM', 'Friedman', 'GK', 'GKX', 'HLN', 'ESL', 'HBB', 'LSTM',
         'HoerlK', 'Hornik', 'HP', 'FPP', 'HK', 'Janus', 'Kaufman', 'LGBM', 'KB', 'Lei', 'LZ', 'MSAplos', 'MfourA',
         'Mfour', 'Mfive', 'Medeiros', 'MMH', 'Patton', 'Rumelhart', 'DeepAR', 'Smyl', 'Tib']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
