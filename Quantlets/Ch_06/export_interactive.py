"""Export the inputs of the Chapter 6 interactive page (TSA/interactive/ch6_data.js).

Uses the functions, seeds and data of generate_all_charts.py, so the page starts from the same data and numbers as the
slides. Eurostat has revised the Romanian GDP series since the lecture was built: the quarterly GDP growth of the slides is
read back from the bars of charts/tsa_ch6_ro_macro.pdf (scaled by the 2009Q1 value of ch6_numbers.json); inflation and
ROBOR 3M are those of the chapter's code (unchanged). Monthly series and the FRED series are cut at the end dates of the
slides (August 2026, September 2026 for the EUR/RON, 2026Q2). Exported: the Romanian quarterly and monthly series, the daily S&P 500, DAX and BET log
returns (%) of the market VAR, the US series of Stock and Watson (2001), the Normal draws of the simulated VAR(1), the
residual-bootstrap draws of the Romanian impulse responses (seed of the chapter), the rolling spillover index, the
rejection counts of the omitted-variable Monte Carlo (the lecture's design plus extra runs with other lag orders and other
effects of z on y, same seed), the pseudo out-of-sample forecast errors and the bootstrap bands of the Stock-Watson VAR.
Every value of the slides that the page recomputes is checked against ch6_numbers.json before the file is written.
No chart is written: the chapter's save() is replaced by plt.close().
Run:  MPLBACKEND=Agg python3 Quantlets/Ch_06/export_interactive.py   (Eurostat and FRED are read online; a few minutes)
"""
import json
import os
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
OUT = os.path.join(ROOT, 'interactive', 'ch6_data.js')
N = json.load(open(os.path.join(HERE, 'ch6_numbers.json')))
M_END, Q_END, FX_END = '2026-08-01', '2026-04-01', '2026-09-01'   # last month (Eurostat), last quarter, last month of EUR/RON


def rd(x, d=4):
    return np.round(np.asarray(x, float), d).tolist()


def close(a, b, tol, what):
    if not abs(a - b) <= tol * max(1.0, abs(b)):
        raise SystemExit(f'{what}: {a} differs from the lecture value {b}')


def cmp(a, b, tol, path):
    """Every number of the lecture's result b equals the recomputed a (relative tolerance tol)."""
    if isinstance(b, dict):
        for k, v in b.items():
            cmp(a[k] if k in a else a[int(k)], v, tol, f'{path}/{k}')
    elif isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            cmp(x, y, tol, f'{path}[{i}]')
    elif isinstance(b, bool) or isinstance(b, str):
        if a != b:
            raise SystemExit(f'{path}: {a} != {b}')
    elif isinstance(b, (int, float)):
        close(float(a), float(b), tol, path)


# ---------------------------------------------------------------- the slide's GDP growth, from the chart PDF
def slide_gdp():
    page = pymupdf.open(os.path.join(ROOT, 'charts', 'tsa_ch6_ro_macro.pdf'))[0]
    green = [x['rect'] for x in page.get_drawings() if x['type'] == 'f' and x.get('fill') and abs(x['fill'][0] - 0.1804) < 1e-3
             and abs(x['fill'][1] - 0.4902) < 1e-3 and x['rect'].y1 < 100]          # bars of the top-left panel (no legend patch)
    bars = sorted(green, key=lambda r: r.x0)
    base = max(set([round(r.y1, 5) for r in bars]), key=[round(r.y1, 5) for r in bars].count)   # the zero line
    h = np.array([(base - r.y0) if abs(r.y1 - base) < 1e-3 else -(r.y1 - base) for r in bars])
    idx = pd.date_range('2000-01-01', periods=len(h), freq='QS')
    assert str(idx[-1])[:10] == Q_END, idx[-1]
    k = list(idx).index(pd.Timestamp('2009-01-01'))
    s = pd.Series(h * N['ro']['g_min'] / h[k], index=idx)
    v = s.loc[g.RO_START:]
    close(v.mean(), N['ro']['mean']['g'], 1e-5, 'mean of g')
    close(v.std(), N['ro']['sd']['g'], 1e-5, 'sd of g')
    close(s.loc['2020-04-01'], N['ro']['g_min2'], 1e-5, 'g 2020Q2')
    return s


G_SLIDE = slide_gdp()
_ro_q = g.ro_quarterly


def ro_quarterly_slide(start=g.RO_START):
    q = _ro_q('2000-01-01').loc[:Q_END].copy()
    q['g'] = G_SLIDE
    return q.loc[start:].dropna(subset=['g'])


g.ro_quarterly = ro_quarterly_slide
_us = g.us_sw
g.us_sw = lambda start=g.SW_START, end=None: _us(start, end if end else Q_END)


def ro_block():
    q = g.ro_quarterly('2000-01-01')
    mm = g.ro_monthly().loc['2000-01-01':]
    m = mm.loc[:M_END]
    fx = mm['eurron'].loc[:FX_END]                      # the BNR rate runs to 18 September 2026 (the market data of the course)
    for c, key in [('pi', 'pi_last'), ('i', 'i_last'), ('u', 'u_last')]:
        close(float(m[c].dropna().iloc[-1]), N['ro'][key], 1e-6, key)
    close(float(fx.iloc[-1]), N['ro']['fx_last'], 1e-6, 'fx_last')
    assert str(m.index[-1])[:10] == M_END
    LAST = ('pi_last', 'pi_last_d', 'i_last', 'u_last', 'u_last_d', 'fx_last')      # checked above on the cut series
    cmp(g.fig_ro_macro(), {k: v for k, v in N['ro'].items() if k not in LAST}, 1e-4, 'ro')
    for name, f in [('ic', g.ic_romania), ('est', g.ro_estimates), ('gr_ro', g.granger_romania),
                    ('gr_hand', g.granger_by_hand), ('fevd', g.fig_fevd_ro), ('order', g.fig_irf_order), ('fc', g.fig_forecast_ro),
                    ('resid', g.fig_resid)]:
        cmp(f(), N[name], 1e-4, name)
    return dict(q0='2000-01-01', g=rd(q['g'], 6), pi=rd(q['pi'], 6), i=rd(q['i'], 6),
                m0='2000-01-01', pim=rd(m['pi'], 4), im=rd(m['i'], 4), um=rd(m['u'], 2), fxm=rd(fx, 5))


def boot_draws():
    """Residual-bootstrap draws of boot_irf for the Romanian VAR(2): the chapter's seed, 500 replications of 84 indices."""
    r = g.ro_fit()
    rng = np.random.default_rng(g.SEED)
    n = len(np.asarray(r.resid))
    ix = np.array([rng.integers(0, n, len(np.asarray(r.endog)) - r.k_ar) for _ in range(500)])
    o = g.fig_irf_ro()
    cmp(o, N['irf'], 1e-4, 'irf')
    return ''.join(np.base_repr(int(v), 36).rjust(2, '0') for v in ix.ravel())


def market_block():
    r = g.market_returns()
    for name, f in [('ccf', g.fig_ccf), ('gr_mkt', g.granger_markets), ('mirf', g.fig_market_irf)]:
        cmp(f(), N[name], 1e-4, name)
    rr = r.round(4)
    G = g.granger_table(rr, g.MKT_P)                    # the rounded returns of the page give the slide's tests
    for k, v in G.items():
        close(v['F'], N['gr_mkt']['tests'][k]['F'], 1e-3, f'rounded returns F {k}')
    th, _ = g.spillover_table(rr)
    close(g.spill_index(th), N['spill']['total'], 1e-3, 'rounded returns spillover')
    s = g.rolling_spillover(r)
    close(float(s.mean()), N['spill']['roll_mean'], 1e-6, 'rolling mean')
    close(float(s.max()), N['spill']['roll_max'], 1e-6, 'rolling max')
    close(float(s.iloc[-1]), N['spill']['roll_last'], 1e-6, 'rolling last')
    d = r.index
    gap = np.diff(d.values).astype('timedelta64[D]').astype(int)
    return dict(d0=str(d[0])[:10], gap=gap.tolist(), sp500=rd(rr['sp500']), dax=rd(rr['dax']), bet=rd(rr['bet']),
                roll=rd(s.values, 3), roll_i0=int(g.DY_W - 1))


def sim_block():
    """The Normal draws of fig_var_sim (seed of the chapter): y_t = c + A y_{t-1} + P z_t, t = 1..299; the first 100 are the burn-in."""
    rng = np.random.default_rng(g.SEED)
    z = np.array([rng.standard_normal(2) for _ in range(299)])
    cmp(g.fig_var_sim(), N['sim'], 1e-6, 'sim')
    cmp(g.worked_example(), N['wx'], 1e-9, 'wx')
    return dict(z=rd(z.ravel(), 6), burn=100)


def gsim_block(Ts=(50, 100, 200, 500), R=1000, ps=(1, 2, 4), cs=(0.0, 0.4, 0.8), als=(0.01, 0.05, 0.10)):
    """Rejection counts of omitted_sim: the lecture's design (effect of z on y 0.8, VAR(2)) and extra runs with the same draws."""
    out = {}
    for c in cs:
        rng = np.random.default_rng(g.SEED)
        cnt = {p: {T: {'bi': [0] * len(als), 'tri': [0] * len(als)} for T in Ts} for p in ps}
        for T in Ts:
            for _ in range(R):
                n = T + 50
                e = rng.standard_normal((n, 3))
                z = np.zeros(n)
                for t in range(1, n):
                    z[t] = 0.5 * z[t - 1] + e[t, 0]
                x = np.r_[0, 0.8 * z[:-1]] + e[:, 1]
                y = np.r_[0, 0, c * z[:-2]] + e[:, 2]
                dd = pd.DataFrame({'x': x, 'y': y, 'z': z}).iloc[50:]
                for p in ps:
                    pb, pt = g.granger_F(dd[['x', 'y']], 'x', 'y', p)['p'], g.granger_F(dd, 'x', 'y', p)['p']
                    for k, a in enumerate(als):
                        cnt[p][T]['bi'][k] += int(pb < a)
                        cnt[p][T]['tri'][k] += int(pt < a)
        out[str(c)] = {str(p): {str(T): v for T, v in cnt[p].items()} for p in ps}
    for T in Ts:
        close(out['0.8']['2'][str(T)]['bi'][1] / R, N['gr_sim'][str(T)]['bivariate'], 1e-12, f'gr_sim bi {T}')
        close(out['0.8']['2'][str(T)]['tri'][1] / R, N['gr_sim'][str(T)]['trivariate'], 1e-12, f'gr_sim tri {T}')
    return dict(R=R, T=list(Ts), p=list(ps), c=[str(c) for c in cs], al=list(als), n=out)


def oos_block():
    dro = g.ro_var_data()
    rro, ero = g.oos(dro, g.RO_P, '2015-01-01', (1, 4))
    dus = g.us_sw(g.SW_START, g.SW_END)
    rus, eus = g.oos(dus, g.SW_P, '1985-01-01', (2, 4, 8))
    cmp({'ro': rro, 'us': rus}, {'ro': N['oos']['ro'], 'us': N['oos']['us']}, 1e-6, 'oos')

    def pack(d, err, start, hs):
        i0 = d.index.get_loc(pd.Timestamp(start))
        # error of the forecast made at origin t-1 for t-1+h: target dates d.index[t-1+h]
        tg = {h: [g.qlabel(d.index[t - 1 + h]) for t in range(i0, len(d) + 1) if t - 1 + h < len(d)] for h in hs}
        return dict(target={str(h): tg[h] for h in hs},
                    e={m: {c: {str(h): rd(err[m][c][h], 5) for h in hs} for c in d.columns} for m in err})
    return dict(ro=pack(dro, ero, '2015-01-01', (1, 4)), us=pack(dus, eus, '1985-01-01', (2, 4, 8)))


def us_block():
    cmp(g.fig_us_data(), N['us'], 1e-9, 'us')
    d = g.us_sw(g.SW_START)
    r = g.fit_var(g.us_sw(g.SW_START, g.SW_END), g.SW_P)
    lo, hi = g.boot_irf(r, 24, 500, 0.10)
    cmp(g.fig_sw_irf(), N['sw'], 1e-4, 'sw')
    return dict(q0=g.qlabel(d.index[0]), pi=rd(d['pi'], 6), u=rd(d['u'], 6), R=rd(d['R'], 6), lo=rd(lo, 5), hi=rd(hi, 5))


def main():
    d = dict(numbers={k: N[k] for k in ('ro', 'ccf', 'wx', 'sim', 'ic', 'roots', 'est', 'resid', 'gr_ro', 'gr_mkt', 'gr_hand', 'gr_sim',
                                        'irf', 'order', 'fevd', 'mirf', 'spill', 'fc', 'oos', 'us', 'sw')})
    for k, f in [('ro', ro_block), ('boot', boot_draws), ('mkt', market_block), ('sim', sim_block), ('us', us_block),
                 ('oos', oos_block), ('gsim', gsim_block)]:
        print(k, flush=True)
        d[k] = f()
    with open(OUT, 'w') as fh:
        fh.write('// Inputs of the Chapter 6 interactive page, exported by Quantlets/Ch_06/export_interactive.py.\n')
        fh.write('window.CH6_DATA = ' + json.dumps(d, separators=(',', ':'), default=str) + ';\n')
    print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB')


if __name__ == '__main__':
    main()
