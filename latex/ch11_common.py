r"""
ch11_common.py -- shared helpers of the Chapter 11 generator (self-study lecture), TSA
=====================================================================================
Numbers from Quantlets/Ch_11/ch11_numbers.json (generate_all_charts.py); the clickable citations of Chapter 11
(DOIs checked against Crossref, arXiv identifiers against the arXiv API, 5 October 2026).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401
from ch1_common import T, V2, finalize, date, pv, MONTHS_EN, MONTHS_RO   # noqa: E402,F401

QL = os.path.join(ROOT, 'Quantlets', 'Ch_11')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_11'


def month(s):
    y, m = int(s[:4]), int(s[5:7]) - 1
    return V2(f'{MONTHS_EN[m]} {y}', f'{MONTHS_RO[m]} {y}')


def load():
    with open(os.path.join(QL, 'ch11_numbers.json')) as f:
        return json.load(f)


def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
A_ = 'https://arxiv.org/abs/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('FPP', 'https://otexts.com/fpp3/', 'Hyndman and Athanasopoulos (2021)', 'Hyndman și Athanasopoulos (2021)')
        + ref('Bommasani', A_ + '2108.07258', 'Bommasani et al.\\ (2021)')
        + ref('Vaswani', A_ + '1706.03762', 'Vaswani et al.\\ (2017)')
        + ref('Nie', A_ + '2211.14730', 'Nie et al.\\ (2023)')
        + ref('Ansari', A_ + '2403.07815', 'Ansari et al.\\ (2024)')
        + ref('AnsariB', A_ + '2510.15821', 'Ansari et al.\\ (2025)')
        + ref('Das', A_ + '2310.10688', 'Das et al.\\ (2024)')
        + ref('Woo', A_ + '2402.02592', 'Woo et al.\\ (2024)')
        + ref('Rasul', A_ + '2310.08278', 'Rasul et al.\\ (2023)')
        + ref('Garza', A_ + '2310.03589', 'Garza et al.\\ (2023)')
        + ref('Gruver', A_ + '2310.07820', 'Gruver et al.\\ (2023)')
        + ref('Tan', A_ + '2406.16964', 'Tan et al.\\ (2024)')
        + ref('Zeng', D_ + '10.1609/aaai.v37i9.26317', 'Zeng et al.\\ (2023)')
        + ref('Aksu', A_ + '2410.10393', 'Aksu et al.\\ (2024)')
        + ref('Monash', A_ + '2105.06643', 'Godahewa et al.\\ (2021)')
        + ref('Salinas', D_ + '10.1016/j.ijforecast.2019.07.001', 'Salinas et al.\\ (2020)')
        + ref('MH', D_ + '10.1016/S0169-2070(00)00057-1', 'Makridakis and Hibon (2000)', 'Makridakis și Hibon (2000)')
        + ref('MSA', D_ + '10.1016/j.ijforecast.2019.04.014', 'Makridakis et al.\\ (2020)')
        + ref('HK', D_ + '10.1016/j.ijforecast.2006.03.001', 'Hyndman and Koehler (2006)', 'Hyndman și Koehler (2006)')
        + ref('MW', D_ + '10.1287/mnsc.22.10.1087', 'Matheson and Winkler (1976)', 'Matheson și Winkler (1976)')
        + ref('GR', D_ + '10.1198/016214506000001437', 'Gneiting and Raftery (2007)', 'Gneiting și Raftery (2007)')
        + ref('Hew', D_ + '10.1007/s10618-022-00894-5', 'Hewamalage et al.\\ (2023)')
        + ref('DM', D_ + '10.1080/07350015.1995.10524599', 'Diebold and Mariano (1995)', 'Diebold și Mariano (1995)'))

BIB = {
    'Aksu': r'Aksu, T., Woo, G., Liu, J., Liu, X., et al.\ (2024). GIFT-Eval: A benchmark for general time series forecasting model evaluation. \href{https://arxiv.org/abs/2410.10393}{arXiv:2410.10393}',
    'Ansari': r'Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., et al.\ (2024). Chronos: Learning the language of time series. \textit{Transactions on Machine Learning Research}. \href{https://arxiv.org/abs/2403.07815}{arXiv:2403.07815}',
    'AnsariB': r'Ansari, A. F., Shchur, O., Küken, J., Auer, A., et al.\ (2025). Chronos-2: From univariate to universal forecasting. \href{https://arxiv.org/abs/2510.15821}{arXiv:2510.15821}',
    'Bommasani': r'Bommasani, R., Hudson, D. A., Adeli, E., Altman, R., et al.\ (2021). On the opportunities and risks of foundation models. \href{https://arxiv.org/abs/2108.07258}{arXiv:2108.07258}',
    'Das': r'Das, A., Kong, W., Sen, R., \& Zhou, Y. (2024). A decoder-only foundation model for time-series forecasting. \textit{Proceedings of the 41st International Conference on Machine Learning (ICML)}. \href{https://arxiv.org/abs/2310.10688}{arXiv:2310.10688}',
    'DM': r'Diebold, F. X., \& Mariano, R. S. (1995). Comparing predictive accuracy. \textit{Journal of Business \& Economic Statistics}, 13(3), 253--263. \href{https://doi.org/10.1080/07350015.1995.10524599}{doi:10.1080/07350015.1995.10524599}',
    'Garza': r'Garza, A., Challu, C., \& Mergenthaler-Canseco, M. (2023). TimeGPT-1. \href{https://arxiv.org/abs/2310.03589}{arXiv:2310.03589}',
    'GR': r'Gneiting, T., \& Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. \textit{Journal of the American Statistical Association}, 102(477), 359--378. \href{https://doi.org/10.1198/016214506000001437}{doi:10.1198/016214506000001437}',
    'Monash': r'Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., \& Montero-Manso, P. (2021). Monash time series forecasting archive. \textit{NeurIPS Track on Datasets and Benchmarks}. \href{https://arxiv.org/abs/2105.06643}{arXiv:2105.06643}',
    'Gruver': r'Gruver, N., Finzi, M., Qiu, S., \& Wilson, A. G. (2023). Large language models are zero-shot time series forecasters. \textit{Advances in Neural Information Processing Systems}, 36. \href{https://arxiv.org/abs/2310.07820}{arXiv:2310.07820}',
    'Hew': r'Hewamalage, H., Ackermann, K., \& Bergmeir, C. (2023). Forecast evaluation for data scientists: Common pitfalls and best practices. \textit{Data Mining and Knowledge Discovery}, 37(2), 788--832. \href{https://doi.org/10.1007/s10618-022-00894-5}{doi:10.1007/s10618-022-00894-5}',
    'HP': r'Huang, C., \& Petukhina, A. (2022). \textit{Applied Time Series Analysis and Forecasting with Python}. Springer. \href{https://doi.org/10.1007/978-3-031-13584-2}{doi:10.1007/978-3-031-13584-2}',
    'FPP': r'Hyndman, R. J., \& Athanasopoulos, G. (2021). \textit{Forecasting: Principles and Practice} (3rd ed.). OTexts. \href{https://otexts.com/fpp3/}{otexts.com/fpp3}',
    'HK': r'Hyndman, R. J., \& Koehler, A. B. (2006). Another look at measures of forecast accuracy. \textit{International Journal of Forecasting}, 22(4), 679--688. \href{https://doi.org/10.1016/j.ijforecast.2006.03.001}{doi:10.1016/j.ijforecast.2006.03.001}',
    'MH': r'Makridakis, S., \& Hibon, M. (2000). The M3-Competition: Results, conclusions and implications. \textit{International Journal of Forecasting}, 16(4), 451--476. \href{https://doi.org/10.1016/S0169-2070(00)00057-1}{doi:10.1016/S0169-2070(00)00057-1}',
    'MSA': r'Makridakis, S., Spiliotis, E., \& Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. \textit{International Journal of Forecasting}, 36(1), 54--74. \href{https://doi.org/10.1016/j.ijforecast.2019.04.014}{doi:10.1016/j.ijforecast.2019.04.014}',
    'MW': r'Matheson, J. E., \& Winkler, R. L. (1976). Scoring rules for continuous probability distributions. \textit{Management Science}, 22(10), 1087--1096. \href{https://doi.org/10.1287/mnsc.22.10.1087}{doi:10.1287/mnsc.22.10.1087}',
    'Nie': r'Nie, Y., Nguyen, N. H., Sinthong, P., \& Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with Transformers. \textit{International Conference on Learning Representations (ICLR)}. \href{https://arxiv.org/abs/2211.14730}{arXiv:2211.14730}',
    'Rasul': r'Rasul, K., Ashok, A., Williams, A. R., Ghonia, H., et al.\ (2023). Lag-Llama: Towards foundation models for probabilistic time series forecasting. \href{https://arxiv.org/abs/2310.08278}{arXiv:2310.08278}',
    'Salinas': r'Salinas, D., Flunkert, V., Gasthaus, J., \& Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. \textit{International Journal of Forecasting}, 36(3), 1181--1191. \href{https://doi.org/10.1016/j.ijforecast.2019.07.001}{doi:10.1016/j.ijforecast.2019.07.001}',
    'Tan': r'Tan, M., Merrill, M. A., Gupta, V., Althoff, T., \& Hartvigsen, T. (2024). Are language models actually useful for time series forecasting? \textit{Advances in Neural Information Processing Systems}, 37. \href{https://arxiv.org/abs/2406.16964}{arXiv:2406.16964}',
    'Vaswani': r'Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., et al.\ (2017). Attention is all you need. \textit{Advances in Neural Information Processing Systems}, 30. \href{https://arxiv.org/abs/1706.03762}{arXiv:1706.03762}',
    'Woo': r'Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., \& Sahoo, D. (2024). Unified training of universal time series forecasting Transformers. \textit{Proceedings of the 41st International Conference on Machine Learning (ICML)}. \href{https://arxiv.org/abs/2402.02592}{arXiv:2402.02592}',
    'Zeng': r'Zeng, A., Chen, M., Zhang, L., \& Xu, Q. (2023). Are Transformers effective for time series forecasting? \textit{Proceedings of the AAAI Conference on Artificial Intelligence}, 37(9), 11121--11128. \href{https://doi.org/10.1609/aaai.v37i9.26317}{doi:10.1609/aaai.v37i9.26317}',
}
ORDER = ['Aksu', 'Ansari', 'AnsariB', 'Bommasani', 'Das', 'DM', 'Garza', 'GR', 'Monash', 'Gruver', 'Hew', 'HP', 'FPP', 'HK',
         'MH', 'MSA', 'MW', 'Nie', 'Rasul', 'Salinas', 'Tan', 'Vaswani', 'Woo', 'Zeng']


def bib(keys=None):
    return [BIB[k] for k in ORDER if keys is None or k in keys]
