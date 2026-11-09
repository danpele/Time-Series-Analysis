"""Export the inputs of the Chapter 7 interactive page (TSA/interactive/ch7_data.js).

Uses the functions, seeds and data loaders of generate_all_charts.py, so the page starts from the same data and numbers as
the slides. The series are read with the chapter's own loaders (Quantlets/common/tsa_data.py) and cut at the end dates of
the slides: the US Treasury yields GS1, GS5, GS10 and TB3MS (FRED, monthly, January 1960 - September 2026); the BNR
reference rates (EUR, HUF, PLN, up to 18 September 2026); the Romanian and euro-area HICP, ROBOR 3M and Euribor 3M
(Eurostat, up to August 2026). With these cuts every number of ch7_numbers.json is reproduced. Romanian household
consumption and GDP (Eurostat national accounts, revised since the lecture) are read back from the line of
charts/ch7_sem_b2.pdf, a 126-point path that matplotlib writes vertex by vertex (no simplification), so the values are those
of the slides. Local data: statsmodels macrodata, the adjusted prices in data/market. Simulations use the chapter's seed.
Exported: the series, the Normal draws of the drunk-and-dog simulation, the null distributions of the Engle-Granger
statistic (the lecture's T = 200 and extra runs with other T, same seed), the Phillips-Ouliaris statistics of the
Engle-Granger table, the term-structure VECM for other sample ends and lag orders (extra runs), the out-of-sample
forecast errors and the rolling pairs-trading backtests (the lecture's entry band and extra runs with other bands).
Every value of the slides that the page recomputes is checked against ch7_numbers.json before the file is written.
No chart and no CSV of the chapter is written: save() is replaced by plt.close() and save_it=False is passed.
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_07/export_interactive.py   (reads FRED, the BNR and Eurostat; a few minutes)
"""
import json
import os
import re
import warnings

os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import pymupdf                          # noqa: E402
import matplotlib.pyplot as plt         # noqa: E402

import generate_all_charts as g         # noqa: E402

warnings.filterwarnings('ignore')
g.save = lambda name, save_it=True: plt.close('all')   # never overwrite the charts of the slides
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
OUT = os.path.join(ROOT, 'interactive', 'ch7_data.js')
N = json.load(open(os.path.join(HERE, 'ch7_numbers.json')))
FRED_END, EUROSTAT_END = '2026-09-01', '2026-08-01'      # last month of the slides (the BNR loader stops at 18 September 2026)
TOL = 1e-9


def rd(x, d=4):
    return np.round(np.asarray(x, float), d).tolist()


def close(a, b, tol, what):
    if not abs(a - b) <= tol * max(1.0, abs(b)):
        raise SystemExit(f'{what}: {a} differs from the lecture value {b}')


def cmp(a, b, tol, path, skip=()):
    """Every number of the lecture's result b equals the recomputed a (relative tolerance tol)."""
    if isinstance(b, dict):
        for k, v in b.items():
            if k in skip:
                continue
            cmp(a[k] if k in a else a[int(k)], v, tol, f'{path}/{k}', skip)
    elif isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            cmp(x, y, tol, f'{path}[{i}]', skip)
    elif isinstance(b, bool) or isinstance(b, str):
        if a != b:
            raise SystemExit(f'{path}: {a} != {b}')
    elif isinstance(b, (int, float)):
        close(float(a), float(b), tol, path)


# ---------------------------------------------------------------- the chapter's loaders, cut at the slides' end dates
_read_fred, _read_eurostat = g.read_fred, g.read_eurostat


def read_fred(ids, start=None, end=None):
    return _read_fred(ids, start).loc[:FRED_END]


def read_eurostat(dataset, key, start=None):
    k = (dataset, key)
    if k in (g.CONS_RO, g.GDP_RO):                 # 100 ln, 1995Q1 = 0: exp(./100) is the level relative to 1995Q1
        return np.exp(ROC['c' if k == g.CONS_RO else 'y'] / 100).rename('OBS_VALUE')
    s = _read_eurostat(dataset, key, start)
    return s.loc[:EUROSTAT_END] if k in (g.ROBOR, g.EURIBOR, g.HICP_RO, g.HICP_EA) else s


def slide_ro_cons():
    """Romanian household consumption and GDP (Eurostat, chain-linked volumes, SCA, 100 ln, 1995Q1 = 0), 1995Q1 - 2026Q2,
    the vintage of the slides, from the two lines of charts/ch7_sem_b2.pdf (126 vertices each, every one written)."""
    doc = pymupdf.open(os.path.join(ROOT, 'charts', 'ch7_sem_b2.pdf'))
    s = b''.join(doc.xref_stream(x) for x in doc[0].get_contents()).decode('latin1')
    ticks = re.findall(r'(-?[\d.]+) (-?[\d.]+) m\n(-?[\d.]+) (-?[\d.]+) l\n\nB\n1 w\nq\n1 0 -0 1 [\d.]+ [\d.]+ cm\nBT\n/F1 [\d.]+ Tf\n(?:-?[\d.]+ 0 Td\n)?\[ \((.*?)\) \] TJ', s)
    xt = [(float(a), lab) for a, b_, c, d_, lab in ticks if a == c][:7]
    yt = [(float(b_), lab) for a, b_, c, d_, lab in ticks if a != c][:3]
    runs = re.findall(r'(-?[\d.]+ -?[\d.]+ m\n(?:-?[\d.]+ -?[\d.]+ l\n){100,})', s)
    d = pd.date_range('1995-01-01', periods=126, freq='QS')
    days = (d - pd.Timestamp('1970-01-01')).days.values.astype(float)
    xd = np.array([(pd.Timestamp(f'{lab}-01-01') - pd.Timestamp('1970-01-01')).days for _, lab in xt], float)
    bx, ax = np.polyfit(xd, [p for p, _ in xt], 1)
    by, ay = np.polyfit([float(lab) for _, lab in yt], [p for p, _ in yt], 1)
    out = {}
    for k, nm in ((0, 'y'), (1, 'c')):
        pts = np.array([[float(v) for v in ln.split()[:2]] for ln in runs[k].strip().split('\n')])
        assert len(pts) == 126 and np.abs(pts[:, 0] - (ax + bx * days)).max() < 1e-3
        out[nm] = (pts[:, 1] - ay) / by
    return pd.DataFrame(out, index=d)[['c', 'y']]


Y = FX = PPP = ROEA = ROC = None


def load():
    global Y, FX, PPP, ROEA, ROC
    ROC = slide_ro_cons()
    g.read_fred, g.read_eurostat = read_fred, read_eurostat
    Y = read_fred(['GS1', 'GS5', 'GS10', 'TB3MS']).dropna().loc[g.YIELD_START:]
    assert len(Y) == N['joh']['US yields 1y, 5y, 10y']['n'] and str(Y.index[-1])[:7] == '2026-09'
    FX = g.cee_fx()
    assert str(FX.index[-1])[:10] == N['cee']['johansen']['last'] and len(FX) == N['cee']['johansen']['n']
    s = 100 * np.log(g.read_reference_rate('EUR').resample('MS').mean())
    PPP = pd.concat([s, 100 * np.log(read_eurostat(*g.HICP_RO)), 100 * np.log(read_eurostat(*g.HICP_EA))], axis=1, keys=['s', 'p', 'ps']).dropna()
    assert len(PPP) == N['ppp']['n']
    ROEA = pd.concat([read_eurostat(*g.ROBOR), read_eurostat(*g.EURIBOR)], axis=1, keys=['ro', 'ea']).dropna()
    assert len(ROEA) == N['roea']['full']['n']


def us_yields(cols=g.YIELDS, start=g.YIELD_START):
    return g.us_yields(cols, start)


# ---------------------------------------------------------------- blocks
def drunk_block():
    o = g.fig_drunk_dog()
    cmp(o, N['drunk'], 1e-9, 'drunk')
    rng = np.random.default_rng(g.SEED)
    u, w, v = rng.standard_normal((3, o['T']))
    return dict(u=rd(u, 6), w=rd(w, 6), v=rd(v, 6), T=o['T'])


def macro_block():
    m = g.us_macro()
    return dict(q0='1959-01-01', y=rd(m.y, 6), c=rd(m.c, 6), i=rd(m.i, 6))


def yields_block():
    ex = g.fig_examples()
    cmp(ex, {k: v for k, v in N['examples'].items() if k in ('rates', 'adf_rates_tb3', 'adf_rates_gs10', 'spread_mean')}, TOL, 'examples (yields)')
    cmp(ex, {k: v for k, v in N['examples'].items() if k not in ('rates', 'adf_rates_tb3', 'adf_rates_gs10', 'spread_mean', 'fx')}, 1e-6, 'examples')
    cmp(ex, {'fx': N['examples']['fx']}, TOL, 'examples (EUR/RON, EUR/HUF)')
    cmp(g.fig_eg_steps(), N['egsteps'], TOL, 'egsteps')
    cmp(g.fig_ecm(), N['ecm'], TOL, 'ecm')
    j = g.johansen_systems(save_it=False)      # no CSV of the chapter is rewritten
    cmp(j, {k: v for k, v in N['joh'].items() if k != 'US output, consumption, investment'}, TOL, 'joh')
    cmp(j, {'US output, consumption, investment': N['joh']['US output, consumption, investment']}, 1e-6, 'joh macro')
    cmp(g.vecm_rates(), N['vecm'], TOL, 'vecm')
    cmp(g.fig_rates(), N['rates'], TOL, 'rates')
    cmp(g.fig_vecm_irf(), N['irf'], TOL, 'irf')
    q = {k: np.round(Y[k].values * 100).astype(int).tolist() for k in ('GS1', 'GS5', 'GS10', 'TB3MS')}
    rw = np.random.default_rng(g.SEED).standard_normal(len(Y)).cumsum()        # the random walk of the ACF panel of fig_eg_steps
    return dict(m0='1960-01-01', rw=rd(rw, 5), **q)


def examples_block():
    """TLV and BRD (100 ln adjusted price) on the BET calendar since 2010 (the 2014 panel is its tail), the BNR quotes."""
    P10 = 100 * np.log(g.stock_panel(g.PAIR, '2010-01-01'))
    P14 = 100 * np.log(g.stock_panel(g.PAIR, g.PAIR_START))
    tail = P10.loc[P14.index[0]:]
    assert tail.index.equals(P14.index) and np.allclose(tail.values, P14.values)
    cmp(g.eg_test(P14.TLV, P14.BRD), N['egtab']['TLV on BRD'], 1e-9, 'banks', skip=('first',))
    d = P10.index
    gap = np.diff(d.values).astype('timedelta64[D]').astype(int)
    return dict(d0=str(d[0])[:10], gap=gap.tolist(), i14=int(len(P10) - len(P14)), tlv=rd(P10.TLV, 5), brd=rd(P10.BRD, 5),
                fx0='2005-07-31', ron=rd(FX['EUR/RON'], 6), huf=rd(FX['EUR/HUF'], 6), pln=rd(FX['EUR/PLN'], 6))


def egdist_block(Ts=(50, 100, 200, 500), R=10000):
    """Null distributions of the residual-based statistic: the lecture's T = 200 (fig_eg_dist) and the same simulation
    with other T (extra runs, same seed): Gaussian kernel densities on the lecture's grid, quantiles, MacKinnon values and
    the rejection rates with the Dickey-Fuller value at 1%, 5% and 10%."""
    from scipy.stats import gaussian_kde
    from statsmodels.tsa.adfvalues import mackinnoncrit
    out = {}
    grid = np.linspace(-6, 3, 400)
    for T in Ts:
        rng = np.random.default_rng(g.SEED)
        taus = {k: g.eg_tau_batch(k, T, R, rng) for k in (1, 2, 3)}
        o = {'kde': {}, 'q': {}, 'mk': {}, 'size': {}}
        for k in (1, 2, 3):
            o['kde'][k] = rd(gaussian_kde(taus[k])(grid), 5)
            o['q'][k] = [float(np.quantile(taus[k], a)) for a in (0.01, 0.05, 0.10)]
            o['mk'][k] = [float(v) for v in mackinnoncrit(N=k, regression='c', nobs=T)]
        for k in (2, 3):
            o['size'][k] = [float(np.mean(taus[k] < o['mk'][1][j])) for j in range(3)]
        out[str(T)] = o
        if T == 200:
            L = N['egdist']
            for k in ('1', '2', '3'):
                close(o['q'][int(k)][1], L[k]['q05'], 1e-12, f'egdist q05 {k}')
                close(o['q'][int(k)][0], L[k]['q01'], 1e-12, f'egdist q01 {k}')
                close(o['mk'][int(k)][1], L[k]['mk5'], 1e-12, f'egdist mk5 {k}')
            close(o['size'][2][1], L['2']['size_with_df'], 1e-12, 'size 2')
            close(o['size'][3][1], L['3']['size_with_df'], 1e-12, 'size 3')
    return dict(grid=[-6, 3, 400], T=list(Ts), R=R, d=out)


DROP = (0, 10, 20, 30, 40, 50)      # % of each sample dropped at its start (the lecture: 0)


def egtable_block():
    """The rows of the Engle-Granger table; Phillips-Ouliaris Z_t and p for the samples without their first k%
    (k = 0: the lecture; the page recomputes the Engle-Granger columns itself)."""
    from arch.unitroot.cointegration import phillips_ouliaris
    t = g.eg_table(False)
    for k in N['egtab']:
        cmp(t[k], N['egtab'][k], 1e-6 if 'Romania' in k else TOL, 'egtab ' + k,   # chart coordinates have 6 decimals
            skip=('const', 'first') if 'Romania' in k else ('first',))
    m = g.us_macro()
    P14 = 100 * np.log(g.stock_panel(g.PAIR, g.PAIR_START))
    P10 = 100 * np.log(g.stock_panel(g.PAIR, '2010-01-01'))
    rows = [('cons', m.c, m.y), ('rates', Y.GS10, Y.TB3MS), ('tlv', P14.TLV, P14.BRD), ('brd', P14.BRD, P14.TLV),
            ('tlv10', P10.TLV, P10.BRD), ('fx', FX['EUR/RON'], FX['EUR/HUF']), ('ro', ROC.c, ROC.y)]
    po = {}
    for key, a, b in rows:
        po[key] = []
        for k in DROP:
            i0 = int(np.floor(len(a) * k / 100))
            r = phillips_ouliaris(a.iloc[i0:], pd.DataFrame(b.iloc[i0:]), trend='c', test_type='Zt')
            po[key].append([round(float(r.stat), 5), float(r.pvalue)])
    return dict(drop=list(DROP), po=po, ro=dict(q0='1995-01-01', c=rd(ROC.c, 6), y=rd(ROC.y, 6)))


def vecm_block(ends=('2000-12', '2005-12', '2010-12', '2015-12', '2019-12', '2026-09'), ks=(1, 2, 3), H=60):
    """The term-structure VECM (rank 2, constant restricted to the relations) for other sample ends and lag orders
    (extra runs; the lecture: September 2026, k = 2): beta, the constants, alpha with t, the eigenvalues of
    I + beta'alpha and the orthogonalised responses to each shock."""
    from statsmodels.tsa.vector_ar.vecm import VECM
    y = us_yields()
    out = {}
    for e in ends:
        for k in ks:
            yy = y.loc[:e]
            res = VECM(yy, k_ar_diff=k, coint_rank=2, deterministic='ci').fit()
            B = np.asarray(res.beta)[:3, :]
            ev = sorted([float(abs(v)) for v in np.linalg.eigvals(np.eye(2) + B.T @ res.alpha)], reverse=True)
            irf = res.irf(H).orth_irfs
            out[f'{e}|{k}'] = dict(n=int(res.nobs), beta=rd(B, 6), const=rd(np.asarray(res.det_coef_coint).ravel(), 6),
                                   alpha=rd(res.alpha, 6), t=rd(res.tvalues_alpha, 4), se=rd(res.stderr_alpha, 6), ev=rd(ev, 6),
                                   irf=rd(irf, 4))
    L = out['2026-09|2']
    V = N['vecm']
    for i in range(3):
        for j in range(2):
            close(L['alpha'][i][j], V['alpha'][i][j], 1e-5, f'alpha {i}{j}')
            close(L['t'][i][j], V['alpha_t'][i][j], 1e-3, f'alpha_t {i}{j}')
    return dict(ends=list(ends), ks=list(ks), H=H, m=out)


def forecast_block(start='1990-01', step=3, H=36):
    """The errors of forecast_compare (same origins, models and re-estimation), to recompute the RMSE ratios."""
    from statsmodels.tsa.vector_ar.vecm import VECM
    from statsmodels.tsa.api import VAR
    y = us_yields()
    k = int(max(g.select_order(y, maxlags=8, deterministic='co').bic, 1))
    E = {m: [] for m in ('vecm', 'dvar', 'rw')}
    origins = list(range(y.index.get_loc(pd.Period(start, 'M')), len(y) - H, step))
    for o in origins:
        tr = y.iloc[:o + 1]
        fv = VECM(tr, k_ar_diff=k, coint_rank=2, deterministic='ci').fit().predict(steps=H)
        dy = tr.diff().dropna()
        fd = VAR(dy).fit(k, trend='n').forecast(dy.values[-k:], H).cumsum(axis=0) + tr.values[-1]
        act = y.values[o + 1:o + H + 1]
        E['vecm'].append(act - fv)
        E['dvar'].append(act - fd)
        E['rw'].append(act - tr.values[-1])
    E = {m: np.array(e) for m, e in E.items()}
    rmse = {m: np.sqrt((e ** 2).mean(axis=0)) for m, e in E.items()}
    for h in (1, 6, 12, 24, 36):
        for j in range(3):
            close(rmse['vecm'][h - 1, j] / rmse['rw'][h - 1, j], N['fc']['ratio_vecm'][str(h)][j], TOL, f'fc vecm {h} {j}')
            close(rmse['dvar'][h - 1, j] / rmse['rw'][h - 1, j], N['fc']['ratio_dvar'][str(h)][j], TOL, f'fc dvar {h} {j}')
    return dict(o=[str(y.index[o]) for o in origins], H=H, e={m: rd(e, 3) for m, e in E.items()})


def ppp_block():
    """s_t (100 ln monthly mean EUR/RON), p_t and p*_t (100 ln HICP of Romania and of the euro area), July 2005 - August 2026."""
    cmp(g.fig_ppp(), N['ppp'], TOL, 'ppp')
    return dict(m0='2005-07-01', s=rd(PPP.s, 6), p=rd(PPP.p, 6), ps=rd(PPP.ps, 6))


def roea_block(years=range(2005, 2019)):
    from arch.unitroot.cointegration import phillips_ouliaris
    cmp(g.fig_ro_ea_rates(), N['roea'], TOL, 'roea')
    po = {}
    for y0 in years:
        d = ROEA.loc[f'{y0}-01-01':]
        r = phillips_ouliaris(d.ro, d[['ea']], trend='c', test_type='Zt')
        po[str(y0)] = [round(float(r.stat), 5), float(r.pvalue)]
    r = phillips_ouliaris(ROEA.ro, ROEA[['ea']], trend='c', test_type='Zt')
    return dict(m0='1995-08-01', ro=rd(ROEA.ro, 2), ea=rd(ROEA.ea, 5), po=po, po_full=[float(r.stat), float(r.pvalue)])


def cee_block():
    cmp(g.fig_cee_fx(), N['cee'], TOL, 'cee')
    return {}


def pair_block():
    cmp(g.fig_pair(), N['pair'], 1e-9, 'pair')
    return {}


def bt(P, form=g.FORM, trade=g.TRADE, entry=g.ENTRY, pmax=0.05):
    """pairs_backtest of the chapter (same windows, tests, entry and exit rules), with the daily turnover and the gross
    return of every trade recorded, so that the page can charge any cost: net = gross - cost * turnover."""
    import itertools
    L = np.log(P)
    R = L.diff()
    names = list(P.columns)
    days, gross, turn, n_sel, trades, opens = [], [], [], [], [], 0
    for s0 in range(form, len(P) - 1, trade):
        fw = L.iloc[s0 - form:s0]
        sel = []
        for a, b in itertools.combinations(names, 2):
            if g.coint(fw[a], fw[b], trend='c', autolag='aic')[1] < pmax:
                X = np.column_stack([np.ones(form), fw[b]])
                c0, beta = np.linalg.lstsq(X, fw[a], rcond=None)[0]
                e = fw[a] - c0 - beta * fw[b]
                sel.append((a, b, c0, beta, e.mean(), e.std()))
        tw = L.iloc[s0 - 1:s0 + trade]
        idx = tw.index[1:]
        g_w, t_w = np.zeros(len(idx)), np.zeros(len(idx))
        for a, b, c0, beta, mu, sd in sel:
            z = (((tw[a] - c0 - beta * tw[b]) - mu) / sd).values
            ra, rb = R[a].reindex(tw.index).values, R[b].reindex(tw.index).values
            pos, cum = 0, 0.0
            for i in range(1, len(z)):
                r = pos * (ra[i] - beta * rb[i]) / (1 + abs(beta))
                new = pos
                if pos == 0 and z[i] > entry:
                    new = -1
                elif pos == 0 and z[i] < -entry:
                    new = 1
                elif (pos == 1 and z[i] >= 0) or (pos == -1 and z[i] <= 0) or i == len(z) - 1:
                    new = 0
                if new != 0 and pos == 0:
                    cum = 0.0
                    opens += 1
                cum += r
                if new == 0 and pos != 0:
                    trades.append(cum)
                g_w[i - 1] += r / len(sel)
                t_w[i - 1] += abs(new - pos) / len(sel)
                pos = new
        days.extend(idx)
        gross.extend(g_w)
        turn.extend(t_w)
        n_sel.append(len(sel))
    out = pd.DataFrame({'gross': gross, 'turn': turn}, index=pd.DatetimeIndex(days))
    out = out[~out.index.duplicated()]
    return out, dict(windows=len(n_sel), pairs_tested=len(names) * (len(names) - 1) // 2, selected_mean=float(np.mean(n_sel)),
                     windows_with_pairs=int(np.sum(np.array(n_sel) > 0)), trades=rd(trades, 6), opens=opens)


def pairs_block(entries=(1.5, 2.0, 2.5)):
    o = g.fig_pairs_backtest()
    cmp(o, N['pairs'], 1e-6, 'pairs')
    bvb = g.stock_panel(g.BVB, '2015-01-01')
    us = g.stock_panel(g.US_BANKS, '2005-01-01', calendar='sp500')
    res = {}
    for nm, P, cost in (('bvb', bvb, g.COST_BVB), ('us', us, g.COST_US)):
        res[nm] = {}
        for en in entries:
            r, info = bt(P, entry=en)
            if en == g.ENTRY:                      # the lecture: the same returns, gross and net of costs
                close(g.perf(r.gross)['mean'], N['pairs'][nm]['gross']['mean'], 1e-9, f'{nm} gross')
                close(g.perf(r.gross - cost * r.turn)['mean'], N['pairs'][nm]['net']['mean'], 1e-9, f'{nm} net')
                close(info['opens'], N['pairs'][nm]['trades'], 0, f'{nm} trades')
            d = r.index
            res[nm][str(en)] = dict(d0=str(d[0])[:10], gap=np.diff(d.values).astype('timedelta64[D]').astype(int).tolist(),
                                    g=rd(r.gross, 7), t=rd(r.turn, 4), **info)
    return dict(entries=list(entries), cost={'bvb': g.COST_BVB, 'us': g.COST_US}, r=res)


def mk_block():
    """MacKinnon (1994, 2010) and Osterwald-Lenum tables used by statsmodels, for the page's ports of adfuller, coint and
    coint_johansen."""
    import statsmodels.tsa.adfvalues as a
    from statsmodels.tsa.coint_tables import c_sja, c_sjt
    fin = lambda v: [float(x) if np.isfinite(x) else 1e9 for x in v]   # noqa: E731
    out = {'max': {}, 'min': {}, 'star': {}, 'small': {}, 'large': {}, 't2010': {}}
    for r in ('n', 'c', 'ct'):
        out['max'][r] = fin(a._tau_maxs[r][:4])
        out['min'][r] = fin(a._tau_mins[r][:4])
        out['star'][r] = fin(a._tau_stars[r][:4])
        out['small'][r] = [fin(x) for x in a._tau_smallps[r][:4]]
        out['large'][r] = [fin(x) for x in a._tau_largeps[r][:4]]
        out['t2010'][r] = np.asarray(a.tau_2010s[r])[:4].tolist()
    out['sjt'] = {str(d): [c_sjt(n, d).tolist() for n in (1, 2, 3)] for d in (-1, 0, 1)}
    out['sja'] = {str(d): [c_sja(n, d).tolist() for n in (1, 2, 3)] for d in (-1, 0, 1)}
    return out


def main():
    load()
    d = dict(numbers={k: N[k] for k in N})
    for k, f in [('mk', mk_block), ('drunk', drunk_block), ('macro', macro_block), ('yields', yields_block), ('mkt', examples_block),
                 ('egtab', egtable_block), ('ppp', ppp_block), ('roea', roea_block), ('cee', cee_block), ('pair', pair_block),
                 ('egdist', egdist_block), ('vecm', vecm_block), ('fc', forecast_block), ('pairs', pairs_block)]:
        print(k, flush=True)
        d[k] = f()
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 7 interactive page, exported by Quantlets/Ch_07/export_interactive.py.\n')
        fh.write('window.CH7_DATA = ' + json.dumps(d, separators=(',', ':'), default=str) + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    main()
