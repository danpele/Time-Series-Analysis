"""
generate_all_charts.py -- charts and numbers of Chapter 11 (TSA): foundation models for time series
===================================================================================================
Course data (tsa_data.py), chart style (tsa_style.py). Every number on the slides comes from here.
  * ideas         -- mean scaling and quantisation of Chronos (Ansari et al. 2024), patching (Nie et al. 2023), the
                     pinball (quantile) loss and the CRPS (Matheson and Winkler 1976; Gneiting and Raftery 2007);
  * zero-shot     -- Chronos-Bolt (tiny, small) and Chronos-2, open weights, run on a CPU without any training on
                     the target series: hourly electricity load of Romania, Romanian industrial production, EUR/RON;
  * evaluation    -- rolling-origin backtest on seven series (Romanian load, industrial production, real GDP,
                     inflation, unemployment; EUR/RON; US retail sales) against the seasonal naive forecast, ETS and
                     ARIMA: MASE (Hyndman and Koehler 2006), weighted quantile loss (a CRPS approximation), coverage
                     of the 80% interval; context length; error by horizon; origins before and after the release
                     of the weights (a contamination check); size and running time.
If the models cannot be downloaded (no internet, no chronos-forecasting package), the foundation models are skipped
and the statistical benchmarks still run.
Output: charts/tsa_ch11_*.pdf/.png, Quantlets/Ch_11/ch11_numbers.json, ch11_backtest.csv
Run:  pip install chronos-forecasting      (once)
      python3 Quantlets/Ch_11/generate_all_charts.py
Time Series Analysis - Daniel Traian PELE
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'common'))
from tsa_data import load_close, read_eurostat, read_fred   # noqa: E402
import tsa_style as st                                      # noqa: E402
from statsmodels.tsa.exponential_smoothing.ets import ETSModel   # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX           # noqa: E402

warnings.filterwarnings('ignore')
SEED = 2026
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]      # the quantile levels of Chronos-Bolt
FM = {'Chronos-Bolt tiny': 'amazon/chronos-bolt-tiny', 'Chronos-Bolt small': 'amazon/chronos-bolt-small',
      'Chronos-2': 'amazon/chronos-2'}
RELEASE = {'Chronos-Bolt tiny': '2024-11-25', 'Chronos-Bolt small': '2024-11-25', 'Chronos-2': '2025-10-30'}
MODELS = ['Seasonal naive', 'ETS', 'ARIMA'] + list(FM)
COLORS = {'Seasonal naive': st.Amber, 'ETS': st.Forest, 'ARIMA': st.Purple, 'Chronos-Bolt tiny': st.Teal,
          'Chronos-Bolt small': st.MainBlue, 'Chronos-2': st.IDAred, 'actual': st.DarkText}
LOAD_FILE = 'ch4_ro_load_hourly.csv'
LOAD_RAW = 'https://raw.githubusercontent.com/danpele/Time-Series-Analysis/main/Quantlets/Ch_04/' + LOAD_FILE
# key: (label, seasonal period m, horizon h, first forecast origin, step between origins, maximum context)
EVAL_SERIES = {
    'load': ('RO electricity load, hourly', 168, 48, '2025-07-07', 168, 2048),
    'ip': ('RO industrial production, monthly', 12, 12, '2016-01-01', 1, 2048),
    'gdp': ('RO real GDP, quarterly', 4, 4, '2012-01-01', 1, 2048),
    'infl': ('RO inflation, monthly', 12, 12, '2016-01-01', 1, 2048),
    'unemp': ('RO unemployment rate, monthly', 1, 12, '2016-01-01', 1, 2048),
    'eurron': ('EUR/RON, daily', 1, 20, '2016-01-04', 20, 2048),
    'retail': ('US retail sales, monthly', 12, 12, '2016-01-01', 1, 2048),
}
SHORT = {'load': 'RO load', 'ip': 'RO ind. prod.', 'gdp': 'RO GDP', 'infl': 'RO inflation', 'unemp': 'RO unemployment',
         'eurron': 'EUR/RON', 'retail': 'US retail'}
CONTEXTS = [96, 192, 336, 672, 1344, 2048]


def save(name, save_it=True):
    if save_it:
        st.check_no_grey(plt.gcf())
        st.save_fig(name)
    else:
        plt.show()


# =============================================================================
# DATA
# =============================================================================
def ro_load():
    """Hourly electricity load of Romania (GW), ENTSO-E extract of Chapter 4, Romanian winter time (UTC+2);
    isolated one-hour glitches (more than 30% away from both neighbours) are interpolated."""
    s = None
    local = [os.path.join(HERE, '..', 'Ch_04', LOAD_FILE)] + [os.path.join(d, 'Quantlets', 'Ch_04', LOAD_FILE)
                                                            for d in ('.', '..', '../..', '../../..')]
    for src in [p for p in local if os.path.exists(p)] + [LOAD_RAW]:
        try:
            s = pd.read_csv(src, index_col=0, parse_dates=True).iloc[:, 0]
            break
        except Exception:
            continue
    s.index = s.index + pd.Timedelta(hours=2)
    s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq='h'))
    r = s / ((s.shift(1) + s.shift(-1)) / 2)
    s[(r < 0.7) | (r > 1.3)] = np.nan
    return (s.interpolate() / 1000).rename('load')


def get_series(key):
    """The seven evaluation series (levels, as published)."""
    if key == 'load':
        return ro_load()
    if key == 'ip':      # industrial production (mining, manufacturing, energy), 2021 = 100, not seasonally adjusted
        return read_eurostat('sts_inpr_m', 'M.PRD.B-D.NSA.I21.RO').rename('ip')
    if key == 'gdp':     # real GDP, chain-linked volumes 2010, million EUR, not seasonally adjusted
        return (read_eurostat('namq_10_gdp', 'Q.CLV10_MEUR.NSA.B1GQ.RO') / 1000).rename('gdp')
    if key == 'infl':    # monthly inflation, 100 * log change of the HICP, in %
        p = read_eurostat('prc_hicp_minr', 'M.I15.TOTAL.RO')
        return (100 * np.log(p).diff()).dropna().loc['2000':].rename('infl')
    if key == 'unemp':   # unemployment rate, seasonally adjusted, % of the labour force
        return read_eurostat('une_rt_m', 'M.SA.TOTAL.PC_ACT.T.RO').rename('unemp')
    if key == 'eurron':  # BNR reference rate
        return load_close('eurron').rename('eurron')
    if key == 'retail':  # US retail sales excluding food services, not seasonally adjusted, billion USD
        return (read_fred('RSXFSN') / 1000).rename('retail')
    raise KeyError(key)


# =============================================================================
# FOUNDATION MODELS (open weights, CPU)
# =============================================================================
_PIPES = {}


def chronos_pipeline(name):
    """Load a Chronos model from Hugging Face (cached); None if the package or the download is not available."""
    if name in _PIPES:
        return _PIPES[name]
    try:
        import torch
        from chronos import BaseChronosPipeline
        _PIPES[name] = BaseChronosPipeline.from_pretrained(FM[name], device_map='cpu', torch_dtype=torch.float32)
    except Exception as e:                       # graceful fallback: the statistical models still run
        print(f'   {name} not available ({type(e).__name__}): skipped')
        _PIPES[name] = None
    return _PIPES[name]


def fm_quantiles(name, contexts, h):
    """Zero-shot quantile forecasts (n, h, 9) at LEVELS from a list of 1-D context arrays; None if unavailable."""
    pipe = chronos_pipeline(name)
    if pipe is None:
        return None
    import torch
    x = [torch.tensor(np.asarray(c, dtype=np.float32)) for c in contexts]
    out = []
    for i in range(0, len(x), 64):
        q, _ = pipe.predict_quantiles(inputs=x[i:i + 64], prediction_length=h, quantile_levels=LEVELS)
        if isinstance(q, list):                   # Chronos-2 returns one tensor per series
            q = torch.stack([t.reshape(-1, h, len(LEVELS))[0] for t in q])
        out.append(q.float().numpy())
    return np.concatenate(out)


def n_parameters(name):
    pipe = chronos_pipeline(name)
    return None if pipe is None else int(sum(p.numel() for p in pipe.model.parameters()))


# =============================================================================
# STATISTICAL BENCHMARKS (each returns quantiles (h, 9))
# =============================================================================
def normal_q(mean, sd):
    z = stats.norm.ppf(LEVELS)
    return np.asarray(mean)[:, None] + np.asarray(sd)[:, None] * z[None, :]


def snaive_q(y, h, m):
    """Seasonal naive: the value one season earlier; sd_h = sigma * sqrt(k + 1), k = number of whole seasons in h - 1
    (Hyndman and Athanasopoulos, FPP3, 5.5); sigma from the in-sample seasonal differences."""
    y = np.asarray(y, float)
    f = np.array([y[len(y) - m + (j % m)] for j in range(h)])
    e = y[m:] - y[:-m]
    sig = np.sqrt(np.mean(e ** 2))
    k = np.arange(h) // m
    return normal_q(f, sig * np.sqrt(k + 1))


def ets_q(y, h, m):
    """ETS with additive errors, damped additive trend and (if m > 1) additive seasonality; Normal quantiles."""
    y = pd.Series(np.asarray(y, float))
    seas = 'add' if 1 < m <= 52 else None
    r = ETSModel(y, error='add', trend='add', damped_trend=True, seasonal=seas,
                 seasonal_periods=m if seas else None).fit(disp=False, maxiter=200)
    p = r.get_prediction(start=len(y), end=len(y) + h - 1)
    return normal_q(np.asarray(p.predicted_mean), np.sqrt(np.asarray(p.forecast_variance)))


def select_arima(y, m):
    """Orders fixed once by AIC on the data before the first origin: (p, d, q) with p, q <= 2, d <= 1, and a seasonal
    MA(1) with one seasonal difference when 1 < m <= 52; for hourly load (m = 168) an ARMA on the weekly differences."""
    y = np.asarray(y, float)
    best = None
    for d in (0, 1):
        for p in range(3):
            for q in range(3):
                for D in ((0, 1) if 1 < m <= 52 else (0,)):
                    so = (0, D, 1, m) if D else (0, 0, 0, 0)
                    if m == 168:
                        so = (0, 1, 0, 168)
                        if d == 1:
                            continue
                    try:
                        a = SARIMAX(y, order=(p, d, q), seasonal_order=so, trend='c' if d + so[1] == 0 else 'n',
                                    simple_differencing=m == 168).fit(disp=False).aic
                    except Exception:
                        continue
                    if best is None or a < best[0]:
                        best = (a, (p, d, q), so)
    return best[1], best[2]


def arima_q(y, h, order, sorder):
    y = np.asarray(y, float)
    m = sorder[3]
    trend = 'c' if order[1] + sorder[1] == 0 else 'n'
    if m == 168:                                 # ARMA on y_t - y_{t-168}, then integrated back (h <= 168)
        r = SARIMAX(y, order=order, seasonal_order=sorder, trend=trend, simple_differencing=True).fit(disp=False)
        f = r.get_forecast(h)
        mean = y[len(y) - 168 + np.arange(h)] + np.asarray(f.predicted_mean)
        return normal_q(mean, np.sqrt(np.asarray(f.var_pred_mean)))
    r = SARIMAX(y, order=order, seasonal_order=sorder, trend=trend).fit(disp=False)
    f = r.get_forecast(h)
    return normal_q(np.asarray(f.predicted_mean), np.sqrt(np.asarray(f.var_pred_mean)))


# =============================================================================
# SCORES
# =============================================================================
def pinball(y, q, tau):
    """Quantile (pinball) loss: tau (y - q) if y >= q, (1 - tau)(q - y) otherwise."""
    u = np.asarray(y) - np.asarray(q)
    return np.maximum(tau * u, (tau - 1) * u)


def wql_parts(y, Q):
    """Numerator and denominator of the weighted quantile loss: sum over levels and steps of 2 * pinball, and
    sum |y| (times the number of levels). WQL = numerator / denominator, a discrete approximation of the CRPS."""
    num = sum(2 * pinball(y, Q[:, j], t).sum() for j, t in enumerate(LEVELS))
    return float(num), float(np.abs(y).sum() * len(LEVELS))


def mase(y, f, insample, m):
    """Mean absolute scaled error: MAE of the forecast over the in-sample MAE of the seasonal naive forecast."""
    x = np.asarray(insample, float)
    scale = np.mean(np.abs(x[m:] - x[:-m]))
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(f))) / scale)


def crps_normal(y, mu, sigma):
    """Closed-form CRPS of N(mu, sigma^2) at y (Gneiting and Raftery 2007)."""
    z = (y - mu) / sigma
    return float(sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)))


# =============================================================================
# BACKTEST
# =============================================================================
def origins_of(key, s):
    label, m, h, first, step, C = EVAL_SERIES[key]
    pos = np.arange(s.index.get_indexer([s.index[s.index >= pd.Timestamp(first)][0]])[0], len(s) - h + 1, step)
    return [int(p) for p in pos]


def backtest(key, max_origins=None, models=None, keep=False):
    """Rolling-origin evaluation of one series: at each origin every model forecasts h steps from the data up to the
    origin (at most C observations). Rows: origin, model, MASE, WQL parts, coverage of the 80% interval."""
    models = models or MODELS
    label, m, h, first, step, C = EVAL_SERIES[key]
    s = get_series(key)
    pos = origins_of(key, s)
    if max_origins:
        pos = pos[-max_origins:]
    order, sorder = select_arima(s.values[max(0, pos[0] - C):pos[0]][-1500:], m) if 'ARIMA' in models else (None, None)
    ctx = [s.values[max(0, p - C):p] for p in pos]
    Q = {}
    times = {}
    for name in [k for k in models if k in FM]:
        t0 = time.time()
        Q[name] = fm_quantiles(name, ctx, h)
        times[name] = (time.time() - t0) / len(ctx)
    for name in [k for k in models if k not in FM]:
        t0 = time.time()
        out = []
        for c in ctx:
            try:
                if name == 'Seasonal naive':
                    out.append(snaive_q(c, h, m))
                elif name == 'ETS':
                    out.append(ets_q(c[-2016:] if m == 168 else c, h, 24 if m == 168 else m))
                else:
                    out.append(arima_q(c, h, order, sorder))
            except Exception:
                out.append(snaive_q(c, h, m))
        Q[name] = np.array(out)
        times[name] = (time.time() - t0) / len(ctx)
    rows, store = [], {}
    for name, q in Q.items():
        if q is None:
            continue
        for i, p in enumerate(pos):
            y = s.values[p:p + h]
            num, den = wql_parts(y, q[i])
            rows.append({'series': key, 'origin': s.index[p], 'model': name, 'mase': mase(y, q[i][:, 4], ctx[i], m),
                         'wql_num': num, 'wql_den': den, 'cov80': float(np.mean((y >= q[i][:, 0]) & (y <= q[i][:, 8]))),
                         'abs_err': list(np.abs(y - q[i][:, 4]))})
        store[name] = q
    df = pd.DataFrame(rows)
    info = {'order': order, 'sorder': sorder, 'n_origins': len(pos), 'first': str(s.index[pos[0]]),
            'last': str(s.index[pos[-1]]), 'seconds': times}
    return (df, info, store, s, pos) if keep else (df, info)


def summarise(df):
    """Per series: mean MASE, WQL (sum of numerators over sum of denominators), coverage; relative to seasonal naive."""
    g = df.groupby('model')
    out = pd.DataFrame({'mase': g['mase'].mean(), 'wql': g['wql_num'].sum() / g['wql_den'].sum(), 'cov80': g['cov80'].mean()})
    out['rel_mase'] = out['mase'] / out.loc['Seasonal naive', 'mase']
    out['rel_wql'] = out['wql'] / out.loc['Seasonal naive', 'wql']
    return out


def run_all(max_origins=None, save_csv=True):
    """Backtest of all seven series; returns the per-row results and the per-series summaries."""
    frames, infos = [], {}
    for key in EVAL_SERIES:
        print('   backtest', key)
        df, info = backtest(key, max_origins)
        frames.append(df)
        infos[key] = info
    allr = pd.concat(frames, ignore_index=True)
    if save_csv:
        allr.drop(columns='abs_err').to_csv(os.path.join(HERE, 'ch11_backtest.csv'), index=False)
        allr[['series', 'origin', 'model', 'abs_err']].to_json(os.path.join(HERE, 'ch11_abs_err.json'), orient='records', date_format='iso')
    return allr, infos


def load_results():
    p = os.path.join(HERE, 'ch11_backtest.csv')
    if not os.path.exists(p):
        return run_all()[0]
    df = pd.read_csv(p, parse_dates=['origin'])
    pe = os.path.join(HERE, 'ch11_abs_err.json')
    if not os.path.exists(pe):                   # errors by step are only needed for fig_horizon
        return df
    e = pd.read_json(pe, orient='records')
    e['origin'] = pd.to_datetime(e['origin']).dt.tz_localize(None)
    return df.merge(e, on=['series', 'origin', 'model'], how='left')


# =============================================================================
# CHARTS: IDEAS
# =============================================================================
def worked_examples():
    """Numbers of the worked examples: mean scaling and tokens, the pinball loss, the Normal CRPS, MASE."""
    u = get_series('unemp').iloc[-6:]
    s = float(np.mean(np.abs(u.values)))
    centres = np.linspace(-15, 15, 4093)
    z = u.values / s
    tok = [int(np.argmin(np.abs(centres - v))) for v in z]
    y, q1, q5, q9 = 7.2, 6.5, 7.0, 7.6
    pl = {t: float(pinball(y, q, t)) for t, q in ((0.1, q1), (0.5, q5), (0.9, q9))}
    sc = np.array([2.0, 1.0, 0.0])                 # attention: scores q.k / sqrt(d) of one query against three keys
    w = np.exp(sc) / np.exp(sc).sum()
    vals = np.array([10.0, 20.0, 30.0])
    return {'att_w': [float(v) for v in w], 'att_out': float(w @ vals),
            'mase_mae': 0.3, 'mase_scale': 0.25, 'mase': 0.3 / 0.25,
            'cov_in': 7, 'cov_n': 10,
            'u_dates': [str(d.date()) for d in u.index], 'u': [float(v) for v in u.values], 's': s, 'z': [float(v) for v in z],
            'tok': tok, 'width': float(centres[1] - centres[0]), 'pin_y': y, 'pin_q': [q1, q5, q9], 'pin': pl,
            'crps_y': 1.0, 'crps': crps_normal(1.0, 0.0, 1.0), 'crps0': crps_normal(0.0, 0.0, 1.0),
            'crps_wide': crps_normal(1.0, 0.0, 2.0)}


def fig_series(save_it=True):
    """The seven evaluation series."""
    fig, axes = plt.subplots(4, 2, figsize=(11, 8.6))
    out = {}
    for ax, (k, col) in zip(axes.flat, zip(EVAL_SERIES, [st.MainBlue, st.IDAred, st.Forest, st.Orange, st.Purple, st.Teal, st.Amber])):
        s = get_series(k)
        x = s.loc['2026-06-01':'2026-06-30'] if k == 'load' else s.loc['2005':]
        ax.plot(x.index, x.values, color=col, lw=0.8 if k in ('load', 'eurron') else 1.1)
        ax.set_title(EVAL_SERIES[k][0] + (' (June 2026)' if k == 'load' else ''), fontsize=11)
        if k == 'load':
            import matplotlib.dates as mdates
            ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
        out[k] = {'n': int(len(s)), 'start': str(s.index[0].date()), 'end': str(s.index[-1].date())}
    axes.flat[-1].axis('off')
    fig.tight_layout()
    save('tsa_ch11_series', save_it)
    return out


def fig_tokens(save_it=True, bins=21):
    """Mean scaling and quantisation (Chronos) on Romanian industrial production; a coarse grid of `bins` bins in
    [-3, 3] instead of 4093 in [-15, 15], so that the steps are visible."""
    x = get_series('ip').iloc[-96:]
    s = float(np.mean(np.abs(x.values)))
    z = x.values / s
    grid = np.linspace(-3, 3, bins)
    tok = np.array([np.argmin(np.abs(grid - v)) for v in z])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
    a1.plot(x.index, x.values, color=st.MainBlue, label='Industrial production (2021 = 100)')
    a1.axhline(s, color=st.IDAred, ls='--', lw=1, label=f'scale s = mean |x| = {s:.1f}')
    a1.set_title('Context: last 96 months')
    st.legend_outside_bottom(a1, ncol=1, y=-0.18)
    a2.plot(x.index, z, color=st.MainBlue, lw=1, label='scaled value x / s')
    a2.step(x.index, grid[tok], where='mid', color=st.IDAred, lw=1.2, label=f'nearest bin centre (token), {bins} bins')
    for g in grid:
        if 0.5 < g < 1.5:
            a2.axhline(g, color=st.Amber, lw=0.4, ls=':')
    a2.set_ylim(min(z) - 0.08, max(z) + 0.08)
    a2.set_title('Scaled context and its tokens')
    st.legend_outside_bottom(a2, ncol=1, y=-0.18)
    fig.tight_layout()
    save('tsa_ch11_tokens', save_it)
    return {'s': s, 'zmin': float(z.min()), 'zmax': float(z.max()), 'ntok': int(len(set(tok))), 'bins': bins,
            'start': str(x.index[0].date()), 'end': str(x.index[-1].date())}


def fig_patching(save_it=True, P=16):
    """Patching: 336 hours of Romanian load cut into patches of P = 16 values; each patch becomes one input vector."""
    x = get_series('load').loc['2026-06-01':].iloc[:336]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    cols = [st.MainBlue, st.IDAred]
    for i in range(0, len(x), P):
        seg = x.iloc[i:i + P]
        ax.plot(seg.index, seg.values, color=cols[(i // P) % 2], lw=1.3,
                label=('odd patches' if (i // P) % 2 == 0 else 'even patches') if i < 2 * P else '_')
        ax.axvspan(seg.index[0], seg.index[-1], color=cols[(i // P) % 2], alpha=0.06, lw=0)
    ax.set_ylabel('GW')
    ax.set_title(f'{len(x)} hourly values = {len(x) // P} patches of {P} values (two weeks of June 2026)')
    st.legend_outside_bottom(ax, ncol=2)
    save('tsa_ch11_patching', save_it)
    return {'n': int(len(x)), 'P': P, 'patches': int(len(x) // P)}


def fig_scores(save_it=True):
    """Left: the pinball loss at levels 0.1, 0.5, 0.9 as a function of y - q. Right: the CRPS of N(0, 1) at y = 1 as
    the area of the squared distance between the predictive CDF and the step function of the outcome."""
    u = np.linspace(-3, 3, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
    for t, c in ((0.1, st.Teal), (0.5, st.MainBlue), (0.9, st.IDAred)):
        a1.plot(u, pinball(u, 0, t), color=c, label=f'level {t}')
    a1.set_xlabel('y - q (outcome minus quantile forecast)')
    a1.set_title('Pinball loss')
    st.legend_outside_bottom(a1, ncol=3, y=-0.26)
    z = np.linspace(-4, 4, 400)
    F = stats.norm.cdf(z)
    H = (z >= 1).astype(float)
    a2.plot(z, F, color=st.MainBlue, label='predictive CDF, N(0, 1)')
    a2.plot(z, H, color=st.IDAred, label='outcome y = 1 (step)')
    a2.fill_between(z, F, H, color=st.Amber, alpha=0.35, label='gap between the CDF and the step; CRPS = integral of the squared gap')
    a2.set_title(f'CRPS = {crps_normal(1.0, 0.0, 1.0):.3f}')
    st.legend_outside_bottom(a2, ncol=1, y=-0.18)
    fig.tight_layout()
    save('tsa_ch11_scores', save_it)
    return {'crps': crps_normal(1.0, 0.0, 1.0)}


# =============================================================================
# CHARTS: ZERO-SHOT FORECASTS
# =============================================================================
def fan(ax, idx, Q, color, label):
    ax.fill_between(idx, Q[:, 0], Q[:, 8], color=color, alpha=0.18, lw=0, label=f'{label}: 80% interval')
    ax.fill_between(idx, Q[:, 2], Q[:, 6], color=color, alpha=0.30, lw=0, label='_')
    ax.plot(idx, Q[:, 4], color=color, lw=1.6, label=f'{label}: median')


def fig_fan_load(save_it=True, origin='2026-06-22'):
    """48-hour zero-shot forecast of Romanian load (Chronos-Bolt small, 2048-hour context) against the seasonal naive."""
    s = get_series('load')
    p = s.index.get_loc(pd.Timestamp(origin))
    ctx = s.values[p - 2048:p]
    h = 48
    idx = s.index[p:p + h]
    Qb = fm_quantiles('Chronos-Bolt small', [ctx], h)
    Qn = snaive_q(ctx, h, 168)
    fig, ax = plt.subplots(figsize=(11, 4))
    hist = s.iloc[p - 168:p]
    ax.plot(hist.index, hist.values, color=st.DarkText, lw=1, label='observed (context, last week)')
    ax.plot(idx, s.values[p:p + h], color=st.DarkText, lw=1.6, ls='--', label='actual')
    ax.plot(idx, Qn[:, 4], color=COLORS['Seasonal naive'], lw=1.4, label='seasonal naive (one week earlier)')
    out = {'origin': origin, 'mase_naive': mase(s.values[p:p + h], Qn[:, 4], ctx, 168)}
    if Qb is not None:
        fan(ax, idx, Qb[0], COLORS['Chronos-Bolt small'], 'Chronos-Bolt small')
        y = s.values[p:p + h]
        out.update({'mase_bolt': mase(y, Qb[0][:, 4], ctx, 168), 'cov_bolt': float(np.mean((y >= Qb[0][:, 0]) & (y <= Qb[0][:, 8]))),
                    'q_24': [float(Qb[0][23, j]) for j in (0, 4, 8)], 'y_24': float(y[23]), 't_24': str(idx[23])})
        out['pin_24'] = {str(t): float(pinball(y[23], Qb[0][23, j], t)) for j, t in ((0, 0.1), (4, 0.5), (8, 0.9))}
    ax.set_ylabel('GW')
    st.legend_outside_bottom(ax, ncol=3)
    save('tsa_ch11_fan_load', save_it)
    return out


def fig_fan_monthly(save_it=True):
    """12-month zero-shot forecast of Romanian industrial production (last complete year) against ETS."""
    s = get_series('ip')
    p = len(s) - 12
    ctx = s.values[:p]
    idx = s.index[p:]
    Qb = fm_quantiles('Chronos-Bolt small', [ctx], 12)
    Qe = ets_q(ctx, 12, 12)
    fig, ax = plt.subplots(figsize=(11, 4))
    hist = s.iloc[p - 48:p]
    ax.plot(hist.index, hist.values, color=st.DarkText, lw=1, label='observed (context, last 4 years)')
    ax.plot(idx, s.values[p:], color=st.DarkText, lw=1.6, ls='--', marker='o', ms=3, label='actual')
    fan(ax, idx, Qe, COLORS['ETS'], 'ETS')
    y = s.values[p:]
    out = {'origin': str(s.index[p].date()), 'end': str(s.index[-1].date()), 'mase_ets': mase(y, Qe[:, 4], ctx, 12),
           'cov_ets': float(np.mean((y >= Qe[:, 0]) & (y <= Qe[:, 8])))}
    if Qb is not None:
        fan(ax, idx, Qb[0], COLORS['Chronos-Bolt small'], 'Chronos-Bolt small')
        out.update({'mase_bolt': mase(y, Qb[0][:, 4], ctx, 12), 'cov_bolt': float(np.mean((y >= Qb[0][:, 0]) & (y <= Qb[0][:, 8])))})
    out['mase_naive'] = mase(y, snaive_q(ctx, 12, 12)[:, 4], ctx, 12)
    ax.set_ylabel('index, 2021 = 100')
    st.legend_outside_bottom(ax, ncol=3)
    save('tsa_ch11_fan_ip', save_it)
    return out


def fig_fan_eurron(save_it=True):
    """20-day zero-shot forecast of EUR/RON (last 20 observations held out) against the random walk."""
    s = get_series('eurron')
    p = len(s) - 20
    ctx = s.values[max(0, p - 2048):p]
    idx = s.index[p:]
    Qb = fm_quantiles('Chronos-Bolt small', [ctx], 20)
    Qn = snaive_q(ctx, 20, 1)
    k = np.arange(1, 21)
    sig = np.std(np.diff(ctx))
    Qn = normal_q(np.repeat(ctx[-1], 20), sig * np.sqrt(k))
    fig, ax = plt.subplots(figsize=(11, 4))
    hist = s.iloc[p - 120:p]
    ax.plot(hist.index, hist.values, color=st.DarkText, lw=1, label='observed (context, last 120 days)')
    ax.plot(idx, s.values[p:], color=st.DarkText, lw=1.6, ls='--', label='actual')
    fan(ax, idx, Qn, COLORS['Seasonal naive'], 'random walk')
    y = s.values[p:]
    out = {'origin': str(s.index[p].date()), 'end': str(s.index[-1].date()), 'last': float(ctx[-1]),
           'width_rw': float(Qn[-1, 8] - Qn[-1, 0])}
    if Qb is not None:
        fan(ax, idx, Qb[0], COLORS['Chronos-Bolt small'], 'Chronos-Bolt small')
        out.update({'med_bolt': float(Qb[0][-1, 4]), 'width_bolt': float(Qb[0][-1, 8] - Qb[0][-1, 0]),
                    'mae_bolt': float(np.mean(np.abs(y - Qb[0][:, 4]))), 'mae_rw': float(np.mean(np.abs(y - ctx[-1])))})
    ax.set_ylabel('RON per EUR')
    st.legend_outside_bottom(ax, ncol=3)
    save('tsa_ch11_fan_eurron', save_it)
    return out


# =============================================================================
# CHARTS: EVALUATION
# =============================================================================
def table_results(df):
    """Per series and model: MASE, WQL, coverage and the ratios to the seasonal naive; overall geometric means."""
    out = {}
    for k in EVAL_SERIES:
        sm = summarise(df[df.series == k])
        out[k] = {m: {c: float(sm.loc[m, c]) for c in sm.columns} for m in sm.index}
        out[k]['_n'] = int(df[(df.series == k) & (df.model == 'Seasonal naive')].shape[0])
    models = [m for m in MODELS if all(m in out[k] for k in EVAL_SERIES)]
    out['_gm'] = {m: {c: float(np.exp(np.mean([np.log(out[k][m][c]) for k in EVAL_SERIES]))) for c in ('rel_mase', 'rel_wql')}
                  for m in models}
    for m in models:
        out['_gm'][m]['cov80'] = float(np.mean([out[k][m]['cov80'] for k in EVAL_SERIES]))
        out['_gm'][m]['wins'] = int(sum(out[k][m]['rel_wql'] == min(out[k][x]['rel_wql'] for x in models) for k in EVAL_SERIES))
    return out


def bars(ax, R, col, models, ref=1.0, log=True):
    x = np.arange(len(EVAL_SERIES))
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * w, [R[k][m][col] for k in EVAL_SERIES], w, color=COLORS[m], label=m)
    ax.axhline(ref, color=st.DarkText, lw=0.8, ls='--')
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[k] for k in EVAL_SERIES], fontsize=10.5)
    if log:
        ax.set_yscale('log')


def fig_benchmark(df=None, save_it=True):
    """Relative MASE and relative WQL (ratio to the seasonal naive forecast, log scale) by series and model."""
    df = load_results() if df is None else df
    R = table_results(df)
    models = [m for m in MODELS if m != 'Seasonal naive' and m in R['_gm']]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True)
    bars(a1, R, 'rel_mase', models)
    a1.set_ylabel('MASE / seasonal naive')
    bars(a2, R, 'rel_wql', models)
    a2.set_ylabel('WQL / seasonal naive')
    st.fig_legend_bottom(fig, ncol=5, y=0.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save('tsa_ch11_benchmark', save_it)
    return R


def fig_coverage(df=None, save_it=True):
    """Empirical coverage of the central 80% interval (10% and 90% quantiles) by series and model."""
    df = load_results() if df is None else df
    R = table_results(df)
    models = [m for m in MODELS if m in R['_gm']]
    fig, ax = plt.subplots(figsize=(11, 4))
    bars(ax, R, 'cov80', models, ref=0.8, log=False)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('share of outcomes inside')
    st.legend_outside_bottom(ax, ncol=6, y=-0.16)
    save('tsa_ch11_coverage', save_it)
    return {k: {m: R[k][m]['cov80'] for m in models} for k in EVAL_SERIES}


def fig_horizon(df=None, save_it=True, key='load'):
    """Mean absolute error by forecast step on Romanian load (48 hours ahead)."""
    df = load_results() if df is None else df
    d = df[df.series == key]
    fig, ax = plt.subplots(figsize=(11, 4))
    out = {}
    for m in [x for x in MODELS if x in set(d.model)]:
        e = np.array(d[d.model == m]['abs_err'].tolist())
        mae = e.mean(axis=0)
        ax.plot(np.arange(1, len(mae) + 1), mae, color=COLORS[m], lw=1.5, label=m)
        out[m] = {'h1': float(mae[0]), 'h24': float(mae[23]), 'h48': float(mae[-1]), 'all': float(mae.mean())}
    ax.set_xlabel('hours ahead')
    ax.set_ylabel('mean absolute error, GW')
    st.legend_outside_bottom(ax, ncol=3)
    save('tsa_ch11_horizon', save_it)
    return out


def fig_context(save_it=True, n_origins=26):
    """Zero-shot accuracy on Romanian load against the context length (last n_origins weekly origins)."""
    s = get_series('load')
    pos = origins_of('load', s)[-n_origins:]
    h = 48
    out = {}
    fig, ax = plt.subplots(figsize=(11, 4))
    for name in FM:
        if chronos_pipeline(name) is None:
            continue
        r = []
        for C in CONTEXTS:
            Q = fm_quantiles(name, [s.values[p - C:p] for p in pos], h)
            r.append(float(np.mean([mase(s.values[p:p + h], Q[i][:, 4], s.values[p - 2048:p], 168) for i, p in enumerate(pos)])))
        out[name] = r
        ax.plot(CONTEXTS, r, color=COLORS[name], marker='o', label=name)
    naive = float(np.mean([mase(s.values[p:p + h], snaive_q(s.values[p - 2048:p], h, 168)[:, 4], s.values[p - 2048:p], 168) for p in pos]))
    ax.axhline(naive, color=COLORS['Seasonal naive'], ls='--', lw=1.2, label='seasonal naive')
    ax.set_xscale('log', base=2)
    ax.set_xticks(CONTEXTS)
    ax.set_xticklabels([f'{c}\n({c / 24:.0f} days)' for c in CONTEXTS], fontsize=10)
    ax.set_xlabel('context length (hours)')
    ax.set_ylabel('MASE')
    st.legend_outside_bottom(ax, ncol=4, y=-0.3)
    save('tsa_ch11_context', save_it)
    return {'contexts': CONTEXTS, 'mase': out, 'naive': naive, 'n': len(pos)}


def fig_prepost(df=None, save_it=True):
    """Contamination check: WQL relative to the seasonal naive for origins before and after the release of the weights,
    for Chronos-Bolt small (25 November 2024) and Chronos-2 (30 October 2025), with ETS on the same origins; only the
    series whose test period spans the release date."""
    df = load_results() if df is None else df
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [3, 1.3]})
    for ax, fm in zip(axes, ('Chronos-Bolt small', 'Chronos-2')):
        cut = pd.Timestamp(RELEASE[fm])
        keys = [k for k in EVAL_SERIES if (df[df.series == k].origin < cut).any() and (df[df.series == k].origin >= cut).any()]
        x = np.arange(len(keys))
        for j, (m, part) in enumerate([(m, p) for m in ('ETS', fm) for p in ('before', 'after')]):
            vals = []
            for k in keys:
                d = df[df.series == k]
                d = d[d.origin < cut] if part == 'before' else d[d.origin >= cut]
                sm = summarise(d)
                vals.append(float(sm.loc[m, 'rel_wql']) if m in sm.index else np.nan)
                out.setdefault(fm, {}).setdefault(k, {})[f'{m}|{part}'] = vals[-1]
                out[fm][k][f'n|{part}'] = int((d.model == 'Seasonal naive').sum())
            ax.bar(x + (j - 1.5) * 0.2, vals, 0.2, facecolor='none' if part == 'before' else COLORS[m],
                   edgecolor=COLORS[m], hatch='///' if part == 'before' else None, lw=1,
                   label=f'{m}, origins {part} the release' if fm == 'Chronos-Bolt small' or m != 'ETS' else '_')
        ax.axhline(1, color=st.DarkText, lw=0.8, ls='--')
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[k] for k in keys], fontsize=10)
        ax.set_title(f'{fm}: weights released {cut.day} {cut.strftime("%b %Y")}', fontsize=11)
    axes[0].set_ylabel('WQL / seasonal naive')
    st.fig_legend_bottom(fig, ncol=3, y=0.12)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save('tsa_ch11_prepost', save_it)
    return out


def fig_size(df=None, infos=None, save_it=True):
    """Accuracy (geometric mean of the relative WQL over the seven series) against the number of parameters."""
    df = load_results() if df is None else df
    R = table_results(df)
    out = {}
    fig, ax = plt.subplots(figsize=(11, 4))
    for m in FM:
        npar = n_parameters(m)
        if npar is None or m not in R['_gm']:
            continue
        out[m] = {'params': npar, 'rel_wql': R['_gm'][m]['rel_wql']}
        ax.scatter(npar / 1e6, R['_gm'][m]['rel_wql'], s=90, color=COLORS[m], label=m, zorder=3)
    for m in ('ETS', 'ARIMA'):
        if m in R['_gm']:
            ax.axhline(R['_gm'][m]['rel_wql'], color=COLORS[m], ls='--', lw=1.2, label=f'{m} (a few parameters per series)')
    ax.axhline(1, color=COLORS['Seasonal naive'], ls=':', lw=1.2, label='seasonal naive')
    ax.set_xscale('log')
    ax.set_xlabel('parameters (millions, log scale)')
    ax.set_ylabel('geometric mean of WQL / naive')
    st.legend_outside_bottom(ax, ncol=3, y=-0.2)
    save('tsa_ch11_size', save_it)
    return out


if __name__ == '__main__':
    st.apply()
    np.random.seed(SEED)
    N = {}
    only = sys.argv[1:]
    path = os.path.join(HERE, 'ch11_numbers.json')
    if only and os.path.exists(path):
        N = json.load(open(path))
    if not only or 'backtest' in only:
        allr, infos = run_all()
        N['infos'] = infos
    for name, f in [('ex', worked_examples), ('series', fig_series), ('tokens', fig_tokens), ('patching', fig_patching),
                    ('scores', fig_scores), ('fan_load', fig_fan_load), ('fan_ip', fig_fan_monthly),
                    ('fan_eurron', fig_fan_eurron), ('bench', fig_benchmark), ('coverage', fig_coverage),
                    ('horizon', fig_horizon), ('context', fig_context), ('prepost', fig_prepost), ('size', fig_size)]:
        if only and name not in only and 'backtest' not in only:
            continue
        print(name)
        N[name] = f()
        with open(path, 'w') as fh:
            json.dump(N, fh, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    print('written ch11_numbers.json')
