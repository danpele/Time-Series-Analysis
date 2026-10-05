r"""
ch15_common.py -- shared helpers of the Chapter 15 generators (review and exam preparation: lecture and seminar), TSA
====================================================================================================================
Numbers:
  * the key empirical facts of Chapters 0--10, read from the numbers files of each chapter
    (Quantlets/Ch_NN/chN_numbers.json; Chapter 0: ch0_values.json), so the review quotes exactly the numbers of the
    chapter slides;
  * the Box--Jenkins case and exam problem 3 from Quantlets/Ch_15/ch15_numbers.json (generate_all_charts.py);
  * the seminar numbers from Quantlets/Ch_15/sem15_results.json (seminar15.py).
Clickable citations: the DOIs already checked against Crossref for Chapters 1--10 (5 October 2026); the bibliography
entries are taken from the chapter modules, unchanged.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tsa_build import ROOT, Values, n   # noqa: E402,F401
from ch1_common import T, V2, finalize, date, quarter, pv, MONTHS_EN, MONTHS_RO   # noqa: E402,F401
import ch1_common as c1   # noqa: E402
import ch2_common as c2   # noqa: E402
import ch3_common as c3   # noqa: E402
import ch4_common as c4   # noqa: E402
import ch5_common as c5   # noqa: E402
import ch6_common as c6   # noqa: E402
import ch7_common as c7   # noqa: E402
import ch8_common as c8   # noqa: E402
import ch9_common as c9   # noqa: E402
import ch10_common as c10   # noqa: E402

QL = os.path.join(ROOT, 'Quantlets', 'Ch_15')
QLURL = 'https://github.com/danpele/Time-Series-Analysis/tree/main/Quantlets/Ch_15'


def _json(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return json.load(f)


def load():
    return _json('Quantlets/Ch_15/ch15_numbers.json')


def load_sem():
    return _json('Quantlets/Ch_15/sem15_results.json')


def chapter_numbers():
    """The numbers files of Chapters 0--10."""
    out = {0: _json('Quantlets/Ch_00/ch0_values.json')}
    for c in range(1, 11):
        out[c] = _json(f'Quantlets/Ch_{c:02d}/ch{c}_numbers.json')
    return out


def month(s):
    y, m = int(s[:4]), int(s[5:7]) - 1
    return V2(f'{MONTHS_EN[m]} {y}', f'{MONTHS_RO[m]} {y}')


def neg(V):
    """Negative numbers: a real minus sign in text and in math mode."""
    for k, v in list(V.items()):
        if isinstance(v, str) and v.startswith('⁅-'):
            V[k] = '⁅\\ensuremath{-}' + v[2:]
    return V


def facts(V):
    """The key facts of Chapters 0--10 as @{fN.name} values (numbers from the chapter files)."""
    C = chapter_numbers()
    P = V.put
    a = C[0]
    P('f0.mult', a['gdp_mult'], 1)
    P('f0.g', a['gdp_growth_ann'], 1)
    P('f0.q1', a['gdp_S1_pct'], 1)
    P('f0.q4', a['gdp_S4_pct'], 1)
    P('f0.ses', a['ses_alpha'], 2)
    P('f0.hw', a['fc_hw_mase'], 2)
    P('f0.nv', a['fc_nv_mase'], 2)
    V.raw('f0.hwwins', str(a['bm_hw_wins']))
    b = C[1]
    P('f1.r1', b['bet']['r']['r1'], 2)
    P('f1.a1', b['bet']['abs']['r1'], 2)
    P('f1.band', b['bet']['band'], 3)
    P('f1.lam', b['boxcox']['lambda'], 2)
    P('f1.od', b['overdiff']['r1_d'], 2)
    c = C[2]
    P('f2.lb', c['gdpdiag']['lb8']['lb_p'], 2)
    P('f2.sumphi', c['infl']['sum_phi'], 3)
    P('f2.period', c['sun']['period'], 1)
    P('f2.phi', c['ret']['bet']['phi'], 3)
    d = C[3]
    u1, u2 = d['urt']['Romania real GDP, log'], d['urt']['Romania real GDP, growth']
    P('f3.adf1', u1['adf']['stat'], 2)
    P('f3.adf1p', u1['adf']['p'], 2)
    P('f3.adf2', u2['adf']['stat'], 2)
    P('f3.cv', u1['adf']['crit5'], 2)
    P('f3.drift', d['gdp_diag']['params']['x1'], 2)
    P('f3.spur', 100 * d['spur_mc']['500']['levels']['reject'], 0)
    P('f3.pow', 100 * d['power']['100_c_0.95'], 1)
    e = C[4]
    P('f4.east', 100 * e['easter']['food']['params']['easter'], 1)
    P('f4.mape', e['airfc']['mape_sarima'], 1)
    P('f4.mapesn', e['airfc']['mape_snaive'], 1)
    P('f4.dhr', e['cv']['tab']['DHR']['MASE'], 2)
    P('f4.snv', e['cv']['tab']['Seasonal naive']['MASE'], 2)
    f = C[5]
    P('f5.psp', f['markets']['sp500']['pers'], 3)
    P('f5.hsp', f['markets']['sp500']['hl'], 0)
    P('f5.pbet', f['markets']['bet']['pers'], 3)
    P('f5.hbet', f['markets']['bet']['hl'], 0)
    P('f5.gj', f['asym']['sp500']['gjr_gamma'], 2)
    P('f5.gjt', f['asym']['sp500']['gjr_t'], 1)
    P('f5.kr', f['est']['kurt_r'], 1)
    P('f5.kz', f['est']['kurt_zt'], 1)
    g = C[6]
    gh = g['gr_hand']
    P('f6.F', gh['F'], 2)
    V.raw('f6.p', pv(gh['p']))
    P('f6.spt', g['spill']['total'], 1)
    P('f6.spm', g['spill']['roll_max'], 1)
    V.raw('f6.spd', date(g['spill']['roll_max_d']))
    h = C[7]
    jy = h['joh']['US yields 1y, 5y, 10y']
    P('f7.tr0', jy['trace'][0], 1)
    P('f7.cv0', jy['trace_cv5'][0], 1)
    P('f7.tr2', jy['trace'][2], 2)
    P('f7.cv2', jy['trace_cv5'][2], 2)
    P('f7.hl', h['ecm']['long']['half'], 1)
    P('f7.ppp', h['ppp']['adf_q']['p'], 2)
    P('f7.sg', h['pairs']['bvb']['gross']['sharpe'], 2)
    P('f7.sn', h['pairs']['bvb']['net']['sharpe'], 2)
    i = C[8]
    arf = next(r for r in i['fit']['table'] if r['name'] == 'ARFIMA(0,d,0)')
    P('f8.d', arf['d'], 2)
    P('f8.dse', arf['se_d'], 2)
    P('f8.hr', i['rsdfa']['rs|S&P 500 returns r'], 2)
    P('f8.ha', i['rsdfa']['rs|S&P 500 |r|'], 2)
    P('f8.nraw', i['nile']['raw']['lw'], 2)
    P('f8.nadj', i['nile']['adj']['lw'], 2)
    j = C[9]
    P('f9.kf', j['leakage']['sp500']['kfold'], 2)
    P('f9.wf', j['leakage']['sp500']['wf'], 2)
    P('f9.base', 100 * j['sign']['sp500']['base_acc'], 1)
    P('f9.gb', 100 * j['sign']['sp500']['GB']['acc'], 1)
    P('f9.ridge', j['load']['tab']['Ridge']['MASE'], 2)
    P('f9.lstm', j['load']['tab']['LSTM']['MASE'], 2)
    k = C[10]
    P('f10.K', k['nile']['ss_K'], 3)
    P('f10.q', k['nile']['q'], 3)
    P('f10.con', 100 * k['msus']['ham_conc']['concord'], 0)
    P('f10.d1', k['msro']['dur1'], 1)
    P('f10.d2', k['msro']['dur2'], 1)
    return V


# -----------------------------------------------------------------------------
# Clickable citations (the DOIs of the chapter modules, checked against Crossref)
# -----------------------------------------------------------------------------
def ref(key, url, en, ro=None):
    return f'\\newcommand{{\\ref{key}}}{{\\href{{{url}}}{{{T(en, ro or en)}}}}}\n'


D_ = 'https://doi.org/'
REFS = (ref('HP', D_ + '10.1007/978-3-031-13584-2', 'Huang and Petukhina (2022)', 'Huang și Petukhina (2022)')
        + ref('FPP', 'https://otexts.com/fpp3/', 'Hyndman and Athanasopoulos (2021)', 'Hyndman și Athanasopoulos (2021)')
        + ref('BD', D_ + '10.1007/978-3-319-29854-2', 'Brockwell and Davis (2016)', 'Brockwell și Davis (2016)')
        + ref('Hamilton', D_ + '10.2307/j.ctv14jx6sm', 'Hamilton (1994)')
        + ref('BJ', D_ + '10.1002/9781118619193', 'Box, Jenkins and Reinsel (2008)', 'Box, Jenkins și Reinsel (2008)')
        + ref('LB', D_ + '10.1093/biomet/65.2.297', 'Ljung and Box (1978)', 'Ljung și Box (1978)')
        + ref('Akaike', D_ + '10.1109/TAC.1974.1100705', 'Akaike (1974)')
        + ref('Schwarz', D_ + '10.1214/aos/1176344136', 'Schwarz (1978)')
        + ref('DF', D_ + '10.1080/01621459.1979.10482531', 'Dickey and Fuller (1979)', 'Dickey și Fuller (1979)')
        + ref('KPSS', D_ + '10.1016/0304-4076(92)90104-Y', 'Kwiatkowski et al.\\ (1992)')
        + ref('GN', D_ + '10.1016/0304-4076(74)90034-7', 'Granger and Newbold (1974)', 'Granger și Newbold (1974)')
        + ref('DM', D_ + '10.1080/07350015.1995.10524599', 'Diebold and Mariano (1995)', 'Diebold și Mariano (1995)')
        + ref('HLN', D_ + '10.1016/S0169-2070(96)00719-4', 'Harvey, Leybourne and Newbold (1997)', 'Harvey, Leybourne și Newbold (1997)')
        + ref('Engle', D_ + '10.2307/1912773', 'Engle (1982)')
        + ref('Boll', D_ + '10.1016/0304-4076(86)90063-1', 'Bollerslev (1986)')
        + ref('Sims', D_ + '10.2307/1912017', 'Sims (1980)')
        + ref('Granger', D_ + '10.2307/1912791', 'Granger (1969)')
        + ref('EG', D_ + '10.2307/1913236', 'Engle and Granger (1987)', 'Engle și Granger (1987)')
        + ref('Joh', D_ + '10.2307/2938278', 'Johansen (1991)')
        + ref('GJ', D_ + '10.1111/j.1467-9892.1980.tb00297.x', 'Granger and Joyeux (1980)', 'Granger și Joyeux (1980)')
        + ref('Hosking', D_ + '10.1093/biomet/68.1.165', 'Hosking (1981)')
        + ref('Mfour', D_ + '10.1016/j.ijforecast.2019.04.014', 'Makridakis, Spiliotis and Assimakopoulos (2020)', 'Makridakis, Spiliotis și Assimakopoulos (2020)')
        + ref('Kalman', D_ + '10.1115/1.3662552', 'Kalman (1960)')
        + ref('HamMS', D_ + '10.2307/1912559', 'Hamilton (1989)'))

_BIB = {'HP': c1.BIB['HP'], 'FPP': c1.BIB['FPP'], 'BD': c1.BIB['BD'], 'Hamilton': c1.BIB['Hamilton'], 'LB': c1.BIB['LB'],
        'BJ': c2.BIB['BJ'], 'Akaike': c2.BIB['Akaike'], 'Schwarz': c2.BIB['Schwarz'],
        'DF': c3.BIB['DF'], 'KPSS': c3.BIB['KPSS'], 'GN': c3.BIB['GN'],
        'DM': c4.BIB['DM'], 'HLN': c4.BIB['HLN'],
        'Engle': c5.BIB['Engle'], 'Boll': c5.BIB['Boll'],
        'Sims': c6.BIB['Sims'], 'Granger': c6.BIB['Granger'],
        'EG': c7.BIB['EG'], 'Joh': c7.BIB['Joh91'],
        'GJ': c8.BIB['GJ'], 'Hosking': c8.BIB['Hosking'],
        'Mfour': c9.BIB['Mfour'],
        'Kalman': c10.BIB['Kalman'], 'HamMS': c10.BIB['HamMS']}


def _sortkey(s):
    import re
    import unicodedata
    s = unicodedata.normalize('NFKD', re.sub(r'\\[a-zA-Z]+|[{}\\]', '', s)).encode('ascii', 'ignore').decode().lower()
    return s


def bib(keys=None):
    """Bibliography entries (all, or the given keys), in alphabetical order."""
    ks = keys or list(_BIB)
    return sorted((_BIB[k] for k in ks), key=_sortkey)
