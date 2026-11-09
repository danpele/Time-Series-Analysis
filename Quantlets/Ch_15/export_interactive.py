"""Export the inputs of the Chapter 15 interactive page (TSA/interactive/ch15_data.js).

The script first runs every function of the chapter (generate_all_charts.py) and checks the results against
ch15_numbers.json exactly (it stops on any drift), then exports what the page needs. Data:
  * the Romanian HICP (Eurostat prc_hicp_minr, 2015 = 100), read with the chapter's loader on 4 November 2026 and cut at
    August 2026, the last month of the slides; with this cut every number of ch15_numbers.json is reproduced exactly
    (the January 2005 - August 2026 values have not been revised since the lecture);
  * Problem 2: Romanian real GDP in the version of the slides, read back by the code of Quantlets/Ch_03/export_interactive.py
    from the vector path of charts/tsa_ch3_gdp_ident.pdf (Eurostat has revised the series), checked against ch3_numbers.json;
  * Problem 4: the quarterly GDP growth, inflation and ROBOR 3M of the Romanian VAR, read by the code of
    Quantlets/Ch_06/export_interactive.py (growth from the bars of charts/tsa_ch6_ro_macro.pdf), checked against ch6_numbers.json;
  * Problem 5: the US Treasury yields GS1, GS5, GS10 (FRED, monthly, not revised) read with the loader of Chapter 7 and cut
    at September 2026, checked against ch7_numbers.json.
Exported from the chapter's code: the 36 SARIMA fits of step 4 (estimates, standard errors, residuals; the table rows are
those of ch15_numbers.json), the one-step forecasts of the test sample, and, as extra runs of the same code on the same data,
the ADF and KPSS tests for other sample ends, the 12-month forecasts from other origins (August 2025 - July 2026) and the
ADF, KPSS and drift of Problem 2 for other sample ends. The browser computes the correlograms, Ljung-Box, Jarque-Bera, the
accuracy measures, the Diebold-Mariano test, the hand forecasts and the OLS of Problem 4 itself.
Nothing of the chapter is written: st.save_fig is replaced by plt.close and the numbers JSON is only read.
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_15/export_interactive.py   (reads Eurostat and FRED; about two minutes)
"""
import json
import math
import os
import subprocess
import sys
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX   # noqa: E402

import generate_all_charts as g         # noqa: E402

warnings.filterwarnings('ignore')
g.st.save_fig = lambda name, *a, **k: plt.close('all')     # never overwrite the charts of the slides
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch15_data.js')
N = json.load(open(os.path.join(HERE, 'ch15_numbers.json')))
NQ = lambda c: json.load(open(os.path.join(ROOT, 'Quantlets', f'Ch_{c:02d}', f'ch{c}_numbers.json')))   # noqa: E731
M_END = '2026-08-01'                     # last month of the slides
_read = g.read_eurostat
g.read_eurostat = lambda *a, **k: _read(*a, **k).loc[:M_END]
J = lambda x: json.loads(json.dumps(x, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))  # noqa: E731


def close(a, b, tol, what):
    a, b = float(a), float(b)
    if math.isnan(b) and math.isnan(a):
        return
    if not abs(a - b) <= tol * max(1.0, abs(b)):
        raise SystemExit(f'{what}: {a} differs from the lecture value {b}')


def cmp(a, b, tol, path):
    if isinstance(b, dict):
        if set(map(str, a)) != set(map(str, b)):
            raise SystemExit(f'{path}: keys {sorted(a)} != {sorted(b)}')
        a = {str(k): v for k, v in a.items()}
        for k, v in b.items():
            cmp(a[k], v, tol, f'{path}/{k}')
    elif isinstance(b, list):
        if len(a) != len(b):
            raise SystemExit(f'{path}: length {len(a)} != {len(b)}')
        for i, (x, y) in enumerate(zip(a, b)):
            cmp(x, y, tol, f'{path}[{i}]')
    elif isinstance(b, (bool, str)) or b is None:
        if a != b:
            raise SystemExit(f'{path}: {a} != {b}')
    else:
        close(a, b, tol, path)


def sg(x, k=12):
    return [None if not np.isfinite(v) else float(f'{v:.{k}g}') for v in np.ravel(np.asarray(x, float))]


def rnd(x, d):
    """The number as the slide prints it (d decimals)."""
    return round(float(x) + 0.0, d)


# ---------------------------------------------------------------- 1. every result of the chapter, exactly
def chapter_results():
    out = {}
    for name, f in [('data', lambda: g.fig_bj_data(save_it=True)), ('tests', g.bj_tests), ('acf', lambda: g.fig_bj_acf(save_it=True)),
                    ('grid', g.bj_grid), ('diag', None), ('fc', None), ('p3', g.exam_p3)]:
        print(name, flush=True)
        if name == 'diag':
            v = g.fig_bj_diag(out['grid'], save_it=True)
        elif name == 'fc':
            v = g.fig_bj_forecast(out['grid'], save_it=True)
        else:
            v = f()
        out[name] = v
        cmp(J(v), N[name], 0.0, name)
    return out


# ---------------------------------------------------------------- 2. the 36 fits of step 4 (same code, all estimates kept)
def fits(G):
    y = g.transforms(g.hicp())['y'].loc[:g.TRAIN_END]
    res = {}
    for r in G['rows']:
        m = g.fit_sarima(y, tuple(r['order']), tuple(r['sorder']))
        close(m.llf, r['loglik'], 0.0, r['model'] + ' loglik')
        close(m.bic, r['bic'], 0.0, r['model'] + ' bic')
        neff = m.nobs - m.loglikelihood_burn
        close(-2 * m.llf + r['k'] * math.log(neff), r['bic'], 1e-12, r['model'] + ' BIC = -2 lnL + k ln(T_eff)')
        e = m.resid[g.S + 1:]
        res[r['model']] = dict(names=list(m.params.index), b=sg(m.params.values), se=sg(m.bse.values), p=sg(m.pvalues.values),
                               e=sg(e.values, 10))
    ch = N['diag']['chosen']
    for k, v in ch['params'].items():
        close(res[N['diag']['model']]['b'][list(ch['params']).index(k)], v, 1e-11, 'chosen ' + k)
    return dict(neff=int(neff), e0=str(e.index[0].date()), fits=res)


# ---------------------------------------------------------------- 3. extra runs: the tests of step 2 for other sample ends
TEST_ENDS = [f'{y}-12-01' for y in range(2012, 2026)] + [M_END]


def tests_by_end():
    d = g.transforms(g.hicp())
    out = {}
    for e in TEST_ENDS:
        x = d.loc[:e]
        out[e[:7]] = {'y': g.adf_kpss(x['y'].dropna(), 'ct'), 'm': g.adf_kpss(x['m'].dropna(), 'c'),
                      'a': g.adf_kpss(x['a'].dropna(), 'c'), 'z': g.adf_kpss(x['z'].dropna(), 'c')}
    cmp(J(out[M_END[:7]]), N['tests'], 0.0, 'tests at August 2026')
    return out


# ---------------------------------------------------------------- 4. forecasts: the test sample and the 12-month forecasts
FC_ORIGINS = [str(t.date()) for t in pd.date_range('2025-08-01', M_END, freq='MS')]


def forecasts(G):
    """The lines of fig_bj_forecast: one-step forecasts of the test sample; 12-month forecasts of annual inflation from the
    lecture's origin (August 2026) and, as extra runs, from the twelve earlier origins."""
    d = g.transforms(g.hicp())
    y, w = d['y'], d['m']
    o, so = tuple(G['chosen_order']), tuple(G['chosen_sorder'])
    m = g.fit_sarima(y.loc[:g.TRAIN_END], o, so)
    mf = m.apply(y)
    test = y.index[y.index > g.TRAIN_END]
    pred = mf.get_prediction(start=test[0], end=test[-1], dynamic=False).predicted_mean
    f_sar = pred - y.shift(1).loc[test]
    f_sn, f_nv, act = w.shift(g.S).loc[test], w.shift(1).loc[test], w.loc[test]
    wt = w.loc[:g.TRAIN_END].dropna()
    scale = np.mean(np.abs(wt.values[g.S:] - wt.values[:-g.S]))
    close(scale, N['fc']['scale'], 0.0, 'MASE scale')
    for f, nm in ((f_sar, 'SARIMA'), (f_sn, 'Seasonal naive'), (f_nv, 'Naive')):
        close(np.sqrt(np.mean((act - f) ** 2)), N['fc']['acc'][nm]['rmse'], 0.0, nm + ' rmse')
    org = {}
    for t0 in FC_ORIGINS:
        yy = y.loc[:t0]
        mm = g.fit_sarima(yy, o, so)
        fc = mm.get_forecast(g.H)
        mean, se = fc.predicted_mean, fc.se_mean
        base = np.array([yy.loc[t - pd.DateOffset(months=g.S)] if t - pd.DateOffset(months=g.S) in yy.index else np.nan for t in mean.index])
        a_f = mean.values - base
        org[t0[:7]] = dict(a=sg(a_f), se=sg(se.values), first=str(mean.index[0].date()))
    L = org[M_END[:7]]
    for k, v in (('a1', L['a'][0]), ('a12', L['a'][-1]), ('se1', L['se'][0]), ('se12', L['se'][-1]), ('a6', L['a'][5])):
        close(v, N['fc'][k], 1e-11, 'forecast ' + k)
    return dict(test0=str(test[0].date()), act=sg(act.values), sar=sg(f_sar.values), sn=sg(f_sn.values), nv=sg(f_nv.values),
                scale=float(scale), origins=org)


# ---------------------------------------------------------------- 5. the data of Problems 2, 4 and 5 (code of Chapters 3, 6, 7)
def sub(chapter, code, args=()):
    """Runs `code` in Quantlets/Ch_NN (its own generate_all_charts and export_interactive), returns the JSON it prints."""
    cwd = os.path.join(ROOT, 'Quantlets', f'Ch_{chapter:02d}')
    pre = ("import os, sys, json, warnings\nos.environ['MPLBACKEND'] = 'Agg'\nsys.dont_write_bytecode = True\n"
           "sys.path.insert(0, os.getcwd())\nwarnings.filterwarnings('ignore')\n")
    r = subprocess.run([sys.executable, '-c', pre + code, *args], cwd=cwd, capture_output=True, text=True)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith('JSON')]
    if r.returncode or not line:
        raise SystemExit(f'Chapter {chapter}: {r.stderr[-2000:]}')
    return json.loads(line[-1][4:])


P2_ENDS = [f'{y}Q4' for y in range(2012, 2026)] + ['2026Q2']
P2_CODE = r'''
import export_interactive as x
g = x.g
G = x.GDP
out = {'q': [str(p) for p in G.index], 'y': [float(v) for v in G.values], 'ends': {}}
for e in sys.argv[1:]:
    y = G.loc[:e]
    d = y.diff().dropna()
    g.gdp_series = lambda start=None, y=y: y
    m = g.gdp_fit((0, 1, 0))
    out['ends'][e] = dict(adf_l=g.adf_test(y, 'ct'), kpss_l=g.kpss_test(y, 'ct'), adf_g=g.adf_test(d, 'c'), kpss_g=g.kpss_test(d, 'c'),
                          c=float(m.params['x1']), se=float(m.bse['x1']), s2=float(m.params['sigma2']), n=int(len(y)))
print('JSON' + json.dumps(out))
'''


def problem2():
    P = sub(3, P2_CODE, P2_ENDS)
    c3 = NQ(3)
    L = P['ends']['2026Q2']
    U1, U2 = c3['urt']['Romania real GDP, log'], c3['urt']['Romania real GDP, growth']
    for a, b, w in ((L['adf_l']['stat'], U1['adf']['stat'], 'ADF level'), (L['adf_g']['stat'], U2['adf']['stat'], 'ADF growth'),
                    (L['kpss_l']['stat'], U1['kpss']['stat'], 'KPSS level'), (L['kpss_g']['stat'], U2['kpss']['stat'], 'KPSS growth'),
                    (L['c'], c3['gdp_diag']['params']['x1'], 'drift'), (L['se'], c3['gdp_diag']['se']['x1'], 'drift SE'),
                    (math.sqrt(L['s2']), c3['gdp_diag']['sigma'], 'sigma')):
        close(a, b, 1e-4, 'Problem 2 ' + w)
    for a, b, d, w in ((L['adf_l']['stat'], -2.31, 2, 'ADF level'), (L['adf_g']['stat'], -10.77, 2, 'ADF growth'), (L['kpss_l']['stat'], 0.198, 3, 'KPSS'),
                       (L['kpss_g']['stat'], 0.263, 3, 'KPSS growth'), (L['c'], 0.81, 2, 'drift'), (L['se'], 0.30, 2, 'SE'), (math.sqrt(L['s2']), 2.47, 2, 'sigma')):
        if rnd(a, d) != b:
            raise SystemExit(f'Problem 2 {w}: {a} is not the slide value {b}')
    y = np.array(P['y'])
    dd = np.diff(y)
    close(dd.mean(), c3['gdp_id']['mean'], 1e-5, 'mean growth')
    return P


P4_CODE = r'''
import export_interactive as x
d = x.g.ro_var_data()
print('JSON' + json.dumps({'q0': str(d.index[0])[:10], 'q1': str(d.index[-1])[:10], 'g': d['g'].tolist(), 'pi': d['pi'].tolist(), 'i': d['i'].tolist()}))
'''


def ols_i(d, p=2, drop=None):
    """The ROBOR equation of the VAR(2) by OLS (the code of statsmodels VAR and of granger_F in Chapter 6)."""
    Y = np.column_stack([d['g'], d['pi'], d['i']])
    T = len(Y) - p
    X = np.column_stack([np.ones(T)] + [Y[p - j:len(Y) - j] for j in range(1, p + 1)])
    if drop is not None:
        X = X[:, [0] + [1 + (j - 1) * 3 + k for j in range(1, p + 1) for k in range(3) if k != drop]]
    y = Y[p:, 2]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return b, float(np.sum((y - X @ b) ** 2))


def problem4():
    P = sub(6, P4_CODE)
    c6 = NQ(6)
    b, rss_u = ols_i(P)
    names = ['const', 'L1.g', 'L1.pi', 'L1.i', 'L2.g', 'L2.pi', 'L2.i']
    for k, v in zip(names, b):
        close(v, c6['est']['params']['i'][k], 1e-4, 'Problem 4 ' + k)
    for k, v in (('L1.pi', 0.2402), ('L2.pi', -0.1718), ('L1.i', 1.0424), ('L2.i', -0.1186)):
        if rnd(b[names.index(k)], 4) != v:
            raise SystemExit(f'Problem 4 {k}: {b[names.index(k)]} is not the slide value {v}')
    _, rss_r = ols_i(P, drop=0)
    close(rss_u, c6['gr_hand']['rss_u'], 1e-4, 'RSS_U')
    close(rss_r, c6['gr_hand']['rss_r'], 1e-4, 'RSS_R')
    P['t'] = c6['est']['tvalues']['i']
    return P


P5_CODE = r'''
import generate_all_charts as g7
y = g7.us_yields().loc[:'2026-09']
print('JSON' + json.dumps({'m0': str(y.index[0]), 'm1': str(y.index[-1]), 'y1': y['GS1'].tolist(), 'y5': y['GS5'].tolist(), 'y10': y['GS10'].tolist()}))
'''


def problem5():
    P = sub(7, P5_CODE)
    V = NQ(7)['vecm']
    Y = np.column_stack([P['y1'], P['y5'], P['y10']])
    if len(Y) != V['n'] + V['k_ar_diff'] + 1 or P['m0'] != V['first'] or P['m1'] != V['last']:
        raise SystemExit(f'Problem 5: sample {P["m0"]} - {P["m1"]}, {len(Y)} months')
    sp = Y @ np.asarray(V['beta'])
    for j in range(2):
        close(sp[:, j].mean(), V['ec_mean'][j], 1e-9, f'EC{j + 1} mean')
        close(sp[:, j].std(), V['ec_sd'][j], 1e-9, f'EC{j + 1} sd')
    M = np.eye(2) + np.asarray(V['beta'])[:3, :].T @ np.asarray(V['alpha'])
    close(sorted(abs(np.linalg.eigvals(M)))[-1], V['ec_eig'][0], 1e-12, 'eigenvalue')
    P.update(beta=V['beta'], alpha=V['alpha'], alpha_t=V['alpha_t'], const=V['const_coint'], eig=V['ec_eig'], half=V['ec_half'])
    return P


def main():
    R = chapter_results()
    G = R['grid']
    h = g.hicp()
    raw = _read(*g.HICP)
    d = dict(numbers=N, m0=str(h.index[0].date()), hicp=sg(h.values), train_end=g.TRAIN_END, ident=g.label(*g.IDENT),
             downloaded='2026-11-04', last_available=str(raw.index[-1].date()))
    print('fits', flush=True)
    d['fit'] = fits(G)
    print('tests', flush=True)
    d['tests'] = tests_by_end()
    print('forecasts', flush=True)
    d['fc'] = forecasts(G)
    print('problems', flush=True)
    d['p2'] = problem2()
    d['p4'] = problem4()
    d['p5'] = problem5()
    c5 = NQ(5)['markets']['bet']
    d['p6'] = dict(params=c5['params'], se=c5['se'], n=c5['n'], vol_sample=c5['vol_sample'])
    c10, c8, c9 = NQ(10)['msro'], NQ(8), NQ(9)['leakage']['sp500']
    arf = next(r for r in c8['fit']['table'] if r['name'] == 'ARFIMA(0,d,0)')
    d['p8'] = dict(p11=c10['p11'], p22=c10['p22'], d=arf['d'], se_d=arf['se_d'])
    d['p7'] = dict(kfold=c9['kfold'], wf=c9['wf'])
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 15 interactive page, exported by Quantlets/Ch_15/export_interactive.py.\n')
        fh.write('window.CH15_DATA = ' + json.dumps(J(d), separators=(',', ':')).replace('NaN', 'null') + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    main()
