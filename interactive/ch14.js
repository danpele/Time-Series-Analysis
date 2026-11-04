// Chapter 14 interactive page (multivariate GARCH models): one block per chart slide, plus the worked examples and the result
// tables of the lecture, computed in the browser from ch14_data.js (Quantlets/Ch_14/export_interactive.py): the daily log
// returns of the chapter's samples (data/market, EODHD) and the estimates of its code (GARCH(1,1) of step 1, DCC (a, b) of
// step 2). The recursions (GARCH variance filter, DCC, rolling correlations and slopes) and the backtests (Kupiec,
// Christoffersen) follow generate_all_charts.py line by line.
(function () {
'use strict';
const D = window.CH14_DATA;
const { nf, $, AX, layout, plot, Player, range, predict, exitTicket, normPpf } = window.TSAI;
const out = (id, items) => window.TSAI.out(id, items);
const T = window.CH14_TEXT;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of tsa_style.py (series colours of the chapter)
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', orange: '#E67E22', purple: '#8E44AD', teal: '#17A2B8', crimson: '#DC3545', navy: '#1F2A44' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.amber]: '181,133,63', [K.orange]: '230,126,34', [K.purple]: '142,68,173',
  [K.teal]: '23,162,184', [K.navy]: '31,42,68', [K.crimson]: '220,53,69' };
const COL = { sp500: K.blue, dax: K.teal, bet: K.red, eurron: K.green, btc: K.amber };
const NAME = D.names;
const PK = D.panel_keys;

// ---------------------------------------------------------------- small helpers
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
const has = id => !!$(id);
const val = id => +$(id).value;
const cache = {};
const memo = (k, f) => (k in cache ? cache[k] : (cache[k] = f()));
const f0 = x => nf(x, 0), f1 = x => nf(x, 1), f2 = x => nf(x, 2), f3 = x => nf(x, 3), f4 = x => nf(x, 4);
const pc = (x, d) => nf(100 * x, d === undefined ? 1 : d) + '%';
const big = x => T.big(x);
const argmin = a => a.reduce((b, v, i) => (v < a[b] ? i : b), 0);
const argmax = a => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
const amean = a => { let s = 0; for (const v of a) s += v; return s / a.length; };
const avar = (a, dd) => { const m = amean(a); let s = 0; for (const v of a) s += (v - m) ** 2; return s / (a.length - (dd === undefined ? 1 : dd)); };
const asd = a => Math.sqrt(avar(a));
const corr = (x, y) => { const mx = amean(x), my = amean(y); let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < x.length; i++) { const u = x[i] - mx, v = y[i] - my; sxy += u * v; sxx += u * u; syy += v * v; } return sxy / Math.sqrt(sxx * syy); };
const kurt = a => { const m = amean(a); let m2 = 0, m4 = 0; for (const v of a) { const d = (v - m) ** 2; m2 += d; m4 += d * d; } m2 /= a.length; m4 /= a.length; return m4 / (m2 * m2) - 3; };   // scipy: Fisher, biased
const qtl = (a, p) => { const b = Float64Array.from(a).sort(), h = (b.length - 1) * p, k = Math.floor(h); return b[k] + (h - k) * ((b[k + 1] ?? b[k]) - b[k]); };
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const trim = (x, d) => nf(+x.toFixed(d), (+x.toFixed(d)).toString().split('.')[1]?.length || 0);
const sig3 = x => (Math.abs(x) >= 100 ? f0(x) : Math.abs(x) >= 10 ? f1(x) : Math.abs(x) >= 1 ? f2(x) : f3(x));
// chi-square(1) survival function: P(chi2 > x) = erfc(sqrt(x / 2)) (erfc of Numerical Recipes, relative error < 1.2e-7)
const erfc = x => { const z = Math.abs(x), t = 1 / (1 + 0.5 * z), r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
  + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))); return x >= 0 ? r : 2 - r; };
const chi1 = x => erfc(Math.sqrt(Math.max(x, 0) / 2));
const pv = p => (p < 0.001 ? T.plt : f3(p));
const pvr = p => (p < 0.001 ? T.plt : '= ' + f3(p));
// dates
const DAY = 864e5;
const tms = s => Date.parse(s.slice(0, 10) + 'T00:00:00Z');
const isoOf = t => new Date(t).toISOString().slice(0, 10);
const expand = B => memo('dates:' + B.d0 + B.off.length, () => B.off.map(o => isoOf(tms(B.d0) + o * DAY)));
const dmy = s => T.dmy(s), my = s => T.my(s);
const between = (dates, a, b) => { const i = []; dates.forEach((d, k) => { if (d >= a && d <= b) i.push(k); }); return i; };

// ---------------------------------------------------------------- panels: one Plotly figure with several axes; stacked on narrow screens
function grid(id, cells, o) {
  const el = $(id), narrow = el.clientWidth < 560, maxc = Math.floor(el.clientWidth / 6.6);
  const wrap = t => { if (!t || t.replace(/<[^>]+>/g, '').length <= maxc) return t; const w = t.split(' '), Ls = ['']; w.forEach(x => { const c = Ls[Ls.length - 1];
    if ((c + ' ' + x).replace(/<[^>]+>/g, '').trim().length > maxc && c) Ls.push(x); else Ls[Ls.length - 1] = (c + ' ' + x).trim(); }); return Ls.join('<br>'); };
  if (narrow) cells = cells.map(c => Object.assign({}, c, { t: wrap(c.t) }));
  const extra = narrow ? Math.max(0, ...cells.map(c => (c.t || '').split('<br>').length - 1)) * 15 : 0;
  const C = narrow ? 1 : o.cols, R = narrow ? cells.length : Math.max(...cells.map(c => c.r + (c.rs || 1) - 1)) + 1;
  const per = o.rowPx || 210;
  const H = narrow ? Math.max(o.h || 360, 120 + (per + extra) * cells.length) + 26 * ((o.legendRows || 1) - 1) : (o.h || 360);
  const plotH = H - 70 - (o.legendRows || 1) * 26;
  const gx = o.gx || 0.08, gy = Math.min(0.5, (narrow ? 62 + extra : (o.gyPx || 62)) / plotH), w = (1 - gx * (C - 1)) / C, h = (1 - gy * (R - 1)) / R;
  const lay = layout({ height: H, margin: { l: o.ml || 52, r: o.mr || 10, t: 26 + extra, b: o.mb || 34 }, annotations: [], shapes: [], showlegend: o.legend !== false,
    legend: { orientation: 'h', x: 0, xanchor: 'left', y: -(narrow ? 34 : (o.legY || 40)) / plotH, yanchor: 'top', bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } } });
  const ax = cells.map((c, k) => {
    const r = narrow ? k : c.r, cc = narrow ? 0 : c.c, cs = narrow ? 1 : (c.cs || 1), rs = narrow ? 1 : (c.rs || 1);
    const x0 = cc * (w + gx), x1 = x0 + cs * w + (cs - 1) * gx, y1 = 1 - r * (h + gy), y0 = y1 - rs * h - (rs - 1) * gy, n = k ? k + 1 : '';
    lay['xaxis' + n] = AX(c.xt || '', Object.assign({ domain: [x0, Math.min(1, x1)], anchor: 'y' + n, automargin: false, title: { text: c.xt || '', font: { size: 11, color: K.navy }, standoff: 4 } }, c.x || {}));
    lay['yaxis' + n] = AX(c.yt || '', Object.assign({ domain: [Math.max(0, y0), y1], anchor: 'x' + n, automargin: c.yam || false, title: { text: c.yt || '', font: { size: 11, color: K.navy }, standoff: 2 } }, c.y || {}));
    if (c.t) lay.annotations.push({ xref: 'paper', yref: 'paper', x: x0, y: y1, xanchor: 'left', yanchor: 'bottom', yshift: 3, showarrow: false, text: c.t, font: { size: 12, color: K.navy } });
    return { xa: 'x' + n, ya: 'y' + n, xd: 'x' + n + ' domain', yd: 'y' + n + ' domain', k: k + 1 };
  });
  el.style.height = H + 'px';
  return { lay, ax, narrow, n: cells.length };
}
// a second y axis on the right of panel a (number k > number of panels)
function twin(G, a, k, title, extra) {
  G.lay['yaxis' + k] = AX('', Object.assign({ overlaying: a.ya, side: 'right', anchor: a.xa, showgrid: false, zeroline: false, automargin: false,
    title: { text: title || '', font: { size: 11, color: K.navy }, standoff: 2 } }, extra || {}));
  return { xa: a.xa, ya: 'y' + k };
}
const ln = (x, y, a, color, name, o) => { const r = Object.assign({ x, y, mode: 'lines', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.4g}<extra></extra>' }, o || {}); r.line = Object.assign({ color, width: 1.4 }, (o && o.line) || {}); return r; };
const mk = (x, y, a, color, name, o) => Object.assign({ x, y, mode: 'markers', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, marker: { color, size: 8 }, hovertemplate: '%{x}: %{y:.4g}<extra></extra>' }, o || {});
const bar = (x, y, a, color, name, o) => Object.assign({ x, y, type: 'bar', marker: { color }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.4g}<extra></extra>' }, o || {});
const hl = (a, xs, y, color, name, o) => ln(xs, xs.map(() => y), a, color, name, Object.assign({ hoverinfo: 'skip' }, o || {}));
// shaded spans drawn as traces (rectangles between y0 and y1)
function spans(a, list, y0, y1, col, al, name) {
  const xs = [], ys = [];
  list.forEach(([u, v]) => { xs.push(u, v, v, u, u, null); ys.push(y0, y0, y1, y1, y0, null); });
  return { x: xs, y: ys, fill: 'toself', fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, mode: 'lines', hoverinfo: 'skip', name: name || '', showlegend: !!name, xaxis: a.xa, yaxis: a.ya };
}
const CR = () => Object.values(D.crises);
const lw = (G, t, n) => { if (!G.narrow) return t; const w = t.split(' '), Ls = ['']; w.forEach(x => { const c = Ls[Ls.length - 1]; if ((c + ' ' + x).length > (n || 40) && c) Ls.push(x); else Ls[Ls.length - 1] = (c + ' ' + x).trim(); }); return Ls.join('<br>'); };

// ---------------------------------------------------------------- tables and worked steps (HTML)
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}>${v.v}</${tag}>`; };
  el.innerHTML = (o && o.cap ? `<p class="c14cap">${o.cap}</p>` : '') + '<table><thead><tr>' + head.map(c => cell(c, 'th')).join('') + '</tr></thead><tbody>'
    + rows.map(r => '<tr' + (r.hi ? ' class="hi"' : '') + '>' + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c14cap">${o.note}</p>` : '');
}
const R_ = v => ({ v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };

// ================================================================= the chapter's recursions
function garchSigma(r, p, s0) {               // garch11_variance + garch11_filter: sigma_t one step ahead, started at s0
  const n = r.length, s = new Float64Array(n);
  let v = s0, ep = 0;
  for (let t = 0; t < n; t++) { if (t) v = p.omega + p.alpha * ep * ep + p.beta * v; s[t] = Math.sqrt(v); ep = r[t] - p.mu; }
  return s;
}
// step 1 of one sample: sigma and z of every column; mult scales alpha with alpha + beta kept (an extra run of the filter)
function step1(B, P, s0, mult) {
  const m = mult === undefined ? 1 : mult, S = {}, Z = {};
  B.cols.forEach(c => {
    const p = Object.assign({}, P[c]); if (m !== 1) { const ab = p.alpha + p.beta; p.alpha *= m; p.beta = ab - p.alpha; }
    const r = B.r[c]; S[c] = garchSigma(r, p, s0[c]); Z[c] = Float64Array.from(r, (v, t) => (v - p.mu) / S[c][t]);
  });
  return { S, Z };
}
function corrMat(Z, keys, idx) {               // np.corrcoef of the columns of Z (rows idx, all rows if omitted)
  const sel = c => (idx ? idx.map(i => Z[c][i]) : Array.from(Z[c]));
  const X = keys.map(sel);
  return keys.map((_, i) => keys.map((_, j) => (i === j ? 1 : corr(X[i], X[j]))));
}
// DCC (Engle 2002): Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}, Q_0 = Qbar; R_t = Q_t rescaled. Returns R_t as flat arrays.
function dccPath(Z, keys, a, b, Qbar) {
  const N = keys.length, T0 = Z[keys[0]].length, Q = Qbar.map(r => r.slice()), out = new Array(T0);
  const c = 1 - a - b;
  for (let t = 0; t < T0; t++) {
    if (t) { const z = keys.map(k => Z[k][t - 1]); for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) Q[i][j] = c * Qbar[i][j] + a * z[i] * z[j] + b * Q[i][j]; }
    const d = Q.map((r, i) => Math.sqrt(r[i])), R = new Float64Array(N * N);
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) R[i * N + j] = Q[i][j] / (d[i] * d[j]);
    out[t] = R;
  }
  return out;
}
const rhoOf = (Rt, N, i, j) => Rt.map(R => R[i * N + j]);
// correlation part of the Gaussian log-likelihood for two series: -1/2 sum (log|R_t| + z' R^-1 z - z'z)
function dccLL2(z1, z2, a, b, qb) {
  let q11 = 1, q22 = 1, q12 = qb, s = 0;
  const c = 1 - a - b;
  for (let t = 0; t < z1.length; t++) {
    if (t) { const u = z1[t - 1], v = z2[t - 1]; q11 = c + a * u * u + b * q11; q22 = c + a * v * v + b * q22; q12 = c * qb + a * u * v + b * q12; }
    const r = q12 / Math.sqrt(q11 * q22), d = 1 - r * r, x = z1[t], y = z2[t];
    s += Math.log(d) + (x * x - 2 * r * x * y + y * y) / d - x * x - y * y;
  }
  return -0.5 * s;
}
function rollCorr(x, y, w) {                    // pandas x.rolling(w).corr(y), dated at the window end
  const n = x.length, o = new Array(n).fill(null);
  for (let t = w - 1; t < n; t++) {
    let mx = 0, my = 0; for (let i = t - w + 1; i <= t; i++) { mx += x[i]; my += y[i]; } mx /= w; my /= w;
    let sxy = 0, sxx = 0, syy = 0; for (let i = t - w + 1; i <= t; i++) { const u = x[i] - mx, v = y[i] - my; sxy += u * v; sxx += u * u; syy += v * v; }
    o[t] = sxy / Math.sqrt(sxx * syy);
  }
  return o;
}
function rollSlope(y, x, w) {                   // (y.rolling(w).cov(x) / x.rolling(w).var()).shift(1): past data only
  const n = x.length, o = new Array(n).fill(null);
  for (let t = w; t < n; t++) {
    let mx = 0, my = 0; for (let i = t - w; i < t; i++) { mx += x[i]; my += y[i]; } mx /= w; my /= w;
    let sxy = 0, sxx = 0; for (let i = t - w; i < t; i++) { const u = x[i] - mx; sxy += u * (y[i] - my); sxx += u * u; }
    o[t] = sxy / sxx;
  }
  return o;
}
const extremes = (v, dates) => { const ix = v.map((x, i) => (x === null ? null : i)).filter(i => i !== null), vv = ix.map(i => v[i]);
  const i0 = ix[argmin(vv)], i1 = ix[argmax(vv)]; return { min: v[i0], dmin: dates[i0], max: v[i1], dmax: dates[i1], last: v[ix[ix.length - 1]], mean: amean(vv), sd: asd(vv) }; };
const hlife = p => Math.log(0.5) / Math.log(p);

// ---------------------------------------------------------------- the chapter's samples
const PAIR = () => D.pair, PANEL = () => D.panel, HEDGE = () => D.hedge;
const pairDates = () => expand(D.pair);
const qbar2 = (S1) => corrMat(S1.Z, ['sp500', 'dax'])[0][1];
// S&P 500 and DAX, full sample, with the chapter's estimates (mult: alpha multiplier of step 1, extra run)
function pairModel(a, b, mult) {
  const k = `pair:${a}:${b}:${mult}`;
  return memo(k, () => {
    const S1 = memo('pairS1:' + mult, () => step1(D.pair, D.pair.P, D.pair.s0, mult));
    const qb = mult === 1 ? D.pair.fit.qbar[1] : qbar2(S1);
    const Rt = dccPath(S1.Z, ['sp500', 'dax'], a, b, [[1, qb], [qb, 1]]);
    return { S: S1.S, Z: S1.Z, rho: rhoOf(Rt, 2, 0, 1), qb };
  });
}
function panelModel(a, b) {
  return memo(`panel:${a}:${b}`, () => {
    const S1 = memo('panelS1', () => step1(D.panel, D.panel.P, D.panel.s0, 1));
    const Q = [0, 1, 2, 3, 4].map(i => D.panel.fit.qbar.slice(5 * i, 5 * i + 5));
    return { S: S1.S, Z: S1.Z, Rt: dccPath(S1.Z, PK, a, b, Q) };
  });
}

// ================================================================= 1. why multivariate volatility
const sigP = (w, s1, s2, r) => Math.sqrt((w * s1) ** 2 + ((1 - w) * s2) ** 2 + 2 * w * (1 - w) * r * s1 * s2);
function blkDv() {
  const w = val('dv-w'), s2 = val('dv-s'), s1 = D.ex.s1; set('dv-w', pc(w, 0)); set('dv-s', f0(s2) + '%');
  const rr = linspace(-1, 1, 201), sp = rr.map(r => sigP(w, s1, s2, r)), p = D.ex.rhos.map(r => sigP(w, s1, s2, r));
  const G = grid('dv-chart', [{ r: 0, c: 0, xt: 'Correlation rho', yt: 'Annual volatility (%)' }], { cols: 1, h: 340 });
  const a = G.ax[0];
  plot('dv-chart', [ln(rr, sp, a, K.blue, lw(G, `Portfolio volatility (${Math.round(100 * w)}/${Math.round(100 * (1 - w))}, assets 20% and ${f0(s2)}%)`), { line: { width: 2 } }),
    mk([D.ex.rhos[0]], [p[0]], a, K.green, `rho = 0.2: ${p[0].toFixed(1)}%`, { marker: { size: 10, color: K.green } }),
    mk([D.ex.rhos[1]], [p[1]], a, K.red, `rho = 0.8: ${p[1].toFixed(1)}%`, { marker: { size: 10, color: K.red } })], G.lay);
  const r0 = sigP(w, s1, s2, 0), rm = sigP(w, s1, s2, -1), r1 = sigP(w, s1, s2, 1);
  out('dv-out', [T.dv1(pc(w, 0), f0(s2), f1(p[0]), f1(p[1]), pc(p[1] / p[0] - 1, 0)), T.dv2(f1(r1), f1(r0), f1(rm))]);
  return { p02: p[0], p08: p[1], r0, rm, r1 };
}
function blkXd() {
  const r = val('xd-r'), s = val('xd-s'); set('xd-r', f2(r)); set('xd-s', f0(s) + '%');
  const t1 = 0.25 * s * s, t2 = 0.25 * s * s, t3 = 2 * 0.25 * r * s * s, v = t1 + t2 + t3, sp = Math.sqrt(Math.max(v, 0));
  const G = grid('xd-chart', [{ r: 0, c: 0, t: 'the three terms of sigma_p^2', yt: 'variance (%^2)' }, { r: 0, c: 1, t: 'sigma_p against rho', xt: 'rho', yt: 'sigma_p (%)' }], { cols: 2, h: 350, legendRows: 2 });
  const [a, b] = G.ax, rr = linspace(-1, 1, 101);
  plot('xd-chart', [bar(['w1^2 s1^2', 'w2^2 s2^2', '2 w1 w2 rho s1 s2', 'sigma_p^2'], [t1, t2, t3, v], a, [K.blue, K.teal, t3 < 0 ? K.green : K.red, K.navy], '', { showlegend: false }),
    ln(rr, rr.map(x => sigP(0.5, s, s, x)), b, K.blue, `sigma_p, each asset ${s}%`, { line: { width: 2 } }),
    mk([r], [sp], b, K.red, `rho = ${r.toFixed(2)}: ${sp.toFixed(1)}%`, { marker: { size: 10, color: K.red } })], G.lay);
  const ref = Math.sqrt(0.5 * s * s * (1 + 0.2));
  const g = x => trim(x, 2), dl = sp / ref - 1;
  steps('xd-ws', [T.xds1(g(0.25 * s * s), f2(r), g(s * s), g(t3), g(v)), T.xds2(g(v), f1(sp)), T.xds3(f1(ref), f1(sp), (dl >= 0 ? '+' : '') + pc(dl, 1))]);
  out('xd-out', [T.xd1(f2(r), f1(sp), f0(s))]);
  return { v, sp };
}
const ROLLP = () => [['sp500', 'dax', D.pair, K.blue], ['bet', 'dax', D.rollpairs.bet_dax, K.red], ['sp500', 'btc', D.rollpairs.sp500_btc, K.amber]];
function blkRo() {
  const w = val('ro-w'); set('ro-w', T.days(w));
  const G = grid('ro-chart', [{ r: 0, c: 0, yt: `Correlation (${w} trading days)`, y: { range: [-0.4, 1.0] }, x: { range: ['1999-06-01', '2026-12-31'] } }], { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0], tr = [spans(a, CR(), -0.4, 1.0, K.crimson, 0.15, 'Crisis periods (2008, 2020)')], S = {};
  ROLLP().forEach(([x, y, B, c]) => {
    const d = expand(B), rc = memo(`roll:${x}${y}:${w}`, () => rollCorr(B.r[x], B.r[y], w)), i0 = w - 1;
    tr.push(ln(d.slice(i0), rc.slice(i0), a, c, `${NAME[x]} and ${NAME[y]}`, { line: { width: 1.1 } }));
    S[`${x}_${y}`] = Object.assign(extremes(rc, d), { full: corr(B.r[x], B.r[y]), n: d.length });
  });
  tr.push(ln(['2000-01-01', '2026-12-31'], [0, 0], a, K.navy, '', { hoverinfo: 'skip', line: { width: 0.6, dash: 'dash' } }));
  plot('ro-chart', tr, G.lay);
  const L = (k, n) => T.ro1(n, f2(S[k].min), my(S[k].dmin), f2(S[k].max), my(S[k].dmax), f2(S[k].full));
  out('ro-out', [L('sp500_dax', 'S&P 500 – DAX'), L('bet_dax', 'BET – DAX'), L('sp500_btc', 'S&P 500 – Bitcoin'), T.ro2(T.days(w), f3(S.sp500_dax.sd))]);
  return S;
}
// k-day returns (sums of k consecutive common daily returns, non-overlapping blocks from the first day): an extra computation
const kSum = (r, k) => { const o = []; for (let i = 0; i + k <= r.length; i += k) { let s = 0; for (let j = i; j < i + k; j++) s += r[j]; o.push(s); } return o; };
const PAIRS10 = () => { const o = []; PK.forEach((x, i) => PK.forEach((y, j) => { if (j > i) o.push([x, y]); })); return o; };
function asyncStats() {
  return memo('async', () => {
    const P = D.panel, W = D.panel.week, o = {};
    PAIRS10().forEach(([x, y]) => { o[`${x}_${y}`] = o[`${y}_${x}`] = { daily: corr(P.r[x], P.r[y]), weekly: corr(W.r[x], W.r[y]) }; });
    return o;
  });
}
function blkTa() {
  const k = val('ta-k'), all = $('ta-p').value === 'all'; set('ta-k', T.days(k));
  const A = asyncStats(), P = D.panel, ks = range(1, 10, 1), pairs = [['sp500', 'bet', K.red], ['sp500', 'dax', K.blue], ['bet', 'dax', K.teal]];
  const G = grid('ta-chart', [{ r: 0, c: 0, t: 'correlation of k-day returns since 2015', xt: 'k (trading days)', yt: 'correlation' },
    { r: 0, c: 1, t: `sum of the last ${k} daily returns, March-May 2025`, yt: '%' }], { cols: 2, h: 380, legendRows: 3 });
  const [a, b] = G.ax, tr = [], ck = {};
  pairs.forEach(([x, y, c]) => {
    const cc = ks.map(j => memo(`kc:${x}${y}${j}`, () => corr(kSum(P.r[x], j), kSum(P.r[y], j))));
    ck[`${x}_${y}`] = cc[k - 1];
    tr.push(ln(ks, cc, a, c, `${NAME[x]} and ${NAME[y]}`, { mode: 'lines+markers', marker: { size: 5, color: c } }),
      mk([5], [A[`${x}_${y}`].weekly], a, c, '', { marker: { size: 11, color: c, symbol: 'star' } }), mk([k], [cc[k - 1]], a, K.navy, '', { marker: { size: 9, color: K.navy, symbol: 'diamond-open' } }));
  });
  tr.push(mk([null], [null], a, K.navy, 'stars: weekly (Friday) returns', { marker: { size: 11, color: K.navy, symbol: 'star' } }));
  const d = pairDates(), sp = D.pair.r.sp500, dx = D.pair.r.dax, ix = between(d, '2025-03-03', '2025-05-30');
  const roll = (r, i) => { let s = 0; for (let j = i - k + 1; j <= i; j++) s += r[j]; return s; };
  tr.push(bar(ix.map(i => d[i]), ix.map(i => roll(sp, i)), b, K.blue, 'S&P 500'), bar(ix.map(i => d[i]), ix.map(i => roll(dx, i)), b, K.teal, 'DAX'));
  G.lay.barmode = 'group';
  plot('ta-chart', tr, G.lay);
  const rows = (all ? PAIRS10() : [['sp500', 'bet'], ['sp500', 'dax'], ['bet', 'dax']]).map(([x, y]) => { const s = A[`${x}_${y}`];
    return [`${NAME[x]} – ${NAME[y]}`, R_(f2(s.daily)), R_(f2(s.weekly)), R_(nf(s.weekly - s.daily, 2))]; });
  table('ta-tb', T.taHead, rows, { note: T.taNote(big(D.panel.off.length), big(D.panel.week.off.length)) });
  const ad = ['2025-04-09', '2025-04-10'].map(s => d.indexOf(s));
  table('ta-tb2', T.taHead2, ['2025-04-02', '2025-04-03', '2025-04-04', '2025-04-07', '2025-04-08', '2025-04-09', '2025-04-10', '2025-04-11'].map(s => { const i = d.indexOf(s);
    return Object.assign([dmy(s), R_(nf(sp[i], 1) + '%'), R_(nf(dx[i], 1) + '%')], { hi: s === '2025-04-09' || s === '2025-04-10' }); }));
  out('ta-out', [T.ta1(f1(sp[ad[0]]), f1(dx[ad[0]]), f1(sp[ad[1]]), f1(dx[ad[1]])), T.ta2(T.days(k), f2(ck.sp500_bet), f2(ck.sp500_dax), f2(ck.bet_dax), k > 1)]);
  return { A, ck, apr: [sp[ad[0]], dx[ad[0]], sp[ad[1]], dx[ad[1]]] };
}

// ================================================================= 2. the framework: positive definiteness
function eig2(h11, h22, h12) { const tr = h11 + h22, dt = h11 * h22 - h12 * h12, q = Math.sqrt(tr * tr / 4 - dt); return [tr / 2 - q, tr / 2 + q]; }
function blkXp() {
  const h12 = val('xp-h'), h11 = val('xp-a'), h22 = 1; set('xp-h', f1(h12)); set('xp-a', f1(h11));
  const det = h11 * h22 - h12 * h12, rho = h12 / Math.sqrt(h11 * h22), [l1, l2] = eig2(h11, h22, h12);
  const v = Math.abs(h12) > 1e-12 ? [h12, l1 - h11] : (h11 <= h22 ? [1, 0] : [0, 1]), nv = Math.hypot(v[0], v[1]), e = [v[0] / nv, v[1] / nv];
  const ws = linspace(-1, 2, 121), pv2 = ws.map(w => w * w * h11 + (1 - w) ** 2 * h22 + 2 * w * (1 - w) * h12);
  const G = grid('xp-chart', [{ r: 0, c: 0, t: 'variance of the portfolio (w, 1 - w)', xt: 'weight w of asset 1', yt: "w'Hw" }, { r: 0, c: 1, t: 'eigenvalues of H', yt: 'eigenvalue' }], { cols: 2, h: 350, legendRows: 2 });
  const [a, b] = G.ax;
  plot('xp-chart', [ln(ws, pv2, a, K.blue, `h12 = ${h12.toFixed(1)}`, { line: { width: 2 } }), hl(a, [-1, 2], 0, K.navy, '', { line: { width: 0.8, dash: 'dash' } }),
    bar(['lambda_min', 'lambda_max'], [l1, l2], b, [l1 < 0 ? K.red : K.green, K.blue], '', { showlegend: false })], G.lay);
  const neg = ws.filter((w, i) => pv2[i] < 0);
  steps('xp-ws', [T.xps1(f1(h11), f1(h12), sig3(det)), T.xps2(f1(h12), f1(h11), f2(rho)), T.xps3(f3(l1), f3(l2), f3(e[0]), f3(e[1]), l1 < 0)]);
  out('xp-out', [T.xp1(det > 0, f2(rho), neg.length ? f2(neg[0]) : null, neg.length ? f2(neg[neg.length - 1]) : null)]);
  return { det, rho, l1, l2 };
}

// ================================================================= 3. VEC and BEKK
function nParams(N) { const k = N * (N + 1) / 2; return { VEC: k + 2 * k * k, DVEC: 3 * k, BEKK: k + 2 * N * N, DBEKK: k + 2 * N, SBEKK: k + 2, CCC: 3 * N + N * (N - 1) / 2, DCC: 3 * N + N * (N - 1) / 2 + 2 }; }
const MODELS = [['VEC', 'VEC(1,1)', K.red], ['BEKK', 'BEKK(1,1)', K.orange], ['DVEC', 'Diagonal VEC', K.purple], ['DBEKK', 'Diagonal BEKK', K.amber], ['DCC', 'DCC', K.blue], ['CCC', 'CCC', K.green]];
function blkXb() {
  const e2 = val('xb-e'), e1 = val('xb-f'), k = D.ex.bk; set('xb-e', f1(e2)); set('xb-f', f1(e1));
  const aa = k.a11 * k.a22, bb = k.b11 * k.b22, hf = (x, y) => k.c12 + aa * x * y + bb * k.h12, h = hf(e1, e2), ee = linspace(-3, 3, 61);
  const lr = k.c12 / (1 - aa - bb), hs = range(0, 60, 1), path = hs.map(j => lr + (aa + bb) ** j * (h - lr));
  const G = grid('xb-chart', [{ r: 0, c: 0, t: 'h12,t against yesterday\'s eps2', xt: 'eps2,t-1', yt: 'h12,t' }, { r: 0, c: 1, t: 'forecast of h12 without new shocks', xt: 'days ahead', yt: 'E h12' }], { cols: 2, h: 350, legendRows: 2 });
  const [a, b] = G.ax;
  plot('xb-chart', [ln(ee, ee.map(x => hf(e1, x)), a, K.blue, `eps1,t-1 = ${e1.toFixed(1)}`, { line: { width: 2 } }), hl(a, [-3, 3], k.h12, K.navy, 'h12,t-1 = 0.5', { line: { width: 1, dash: 'dash' } }),
    mk([e2], [h], a, K.red, `h12,t = ${h.toFixed(4)}`, { marker: { size: 10, color: K.red } }),
    ln(hs, path, b, K.red, 'E_t h12,t+k (extra run)', { line: { width: 2 } }), hl(b, [0, 60], lr, K.green, `long-run level ${lr.toFixed(3)}`, { line: { width: 1.2, dash: 'dash' } })], G.lay);
  const par = (x, d) => (x < 0 ? `(${nf(x, d)})` : nf(x, d));
  steps('xb-ws', [T.xbs1(f3(aa), f3(bb)), T.xbs2(f1(e1), f1(e2), f3(e1 * e2)), T.xbs3(f3(k.c12), f3(aa), par(e1 * e2, 3), f3(bb), par(aa * e1 * e2, 4), nf(bb * k.h12, 4), nf(h, 4)), T.xbs4(nf(h, 4), h > k.h12)]);
  out('xb-out', [T.xb1(nf(h, 4), f3(lr))]);
  return { aa, bb, h, lr };
}
function blkTm() {
  const n = val('tm-n'); set('tm-n', n);
  const P = nParams(n), ks = ['VEC', 'DVEC', 'BEKK', 'DBEKK', 'SBEKK', 'CCC', 'DCC'];
  const G = grid('tm-chart', [{ r: 0, c: 0, t: `parameters for N = ${n} assets (log scale)`, yt: 'parameters', y: { type: 'log' } }], { cols: 1, h: 340, legend: false });
  const a = G.ax[0];
  plot('tm-chart', [bar(ks.map(k => T.model[k]), ks.map(k => P[k]), a, ks.map(k => (k === 'DCC' || k === 'CCC' ? K.blue : K.red)), '', { text: ks.map(k => big(P[k])), textposition: 'outside', cliponaxis: false })], G.lay);
  const cols = [2, 5, 10, 50];
  table('tm-tb', T.tmHead(n), ks.map(k => [T.modelT[k], ...cols.map(c => R_(big(nParams(c)[k]))), R_(`<b>${big(P[k])}</b>`)]), { note: T.tmNote });
  out('tm-out', [T.tm1(n, big(P.VEC), big(P.BEKK), big(P.DCC), big(n * (n + 1) / 2))]);
  return P;
}
function blkPc() {
  const nm = val('pc-n'); set('pc-n', nm);
  const Ns = range(2, nm, 1);
  const G = grid('pc-chart', [{ r: 0, c: 0, xt: 'Number of assets N', yt: 'Number of parameters (log scale)', y: { type: 'log' } }], { cols: 1, h: 360, legendRows: 2 });
  const a = G.ax[0];
  plot('pc-chart', MODELS.map(([k, lab, c]) => ln(Ns, Ns.map(n => nParams(n)[k]), a, c, lab, { line: { width: 1.8, dash: k === 'CCC' ? 'dash' : 'solid' } })), G.lay);
  const P = nParams(nm);
  out('pc-out', [T.pc1(nm, big(P.VEC), big(P.BEKK), big(P.DBEKK), big(P.DCC)), T.pc2(big(P.VEC / P.DCC))]);
  return P;
}

// ================================================================= 4. CCC and DCC: worked examples
function blkXc() {
  const z2 = val('xc-z'), a = val('xc-a'), b = val('xc-b'), E = D.ex, z1 = E.z[0]; set('xc-z', f1(z2)); set('xc-a', f2(a)); set('xc-b', f2(b));
  const c = 1 - a - b, q = zz => { const q11 = c + a * z1 * z1 + b * E.q11, q22 = c + a * zz * zz + b * E.q22, q12 = c * E.qbar + a * z1 * zz + b * E.q12; return { q11, q22, q12, r: q12 / Math.sqrt(q11 * q22) }; };
  const Q = q(z2), zs = linspace(-3, 3, 61), ks = range(0, 120, 1);
  const path = ks.map(k => { const p = (a + b) ** k, q11 = 1 + p * (Q.q11 - 1), q22 = 1 + p * (Q.q22 - 1), q12 = E.qbar + p * (Q.q12 - E.qbar); return q12 / Math.sqrt(q11 * q22); });
  const G = grid('xc-chart', [{ r: 0, c: 0, t: 'rho_12,t against yesterday\'s z2', xt: 'z2,t-1', yt: 'rho_12,t' }, { r: 0, c: 1, t: 'forecast of rho_12 without new shocks', xt: 'days ahead', yt: 'rho_12' }], { cols: 2, h: 350, legendRows: 2 });
  const [A, B] = G.ax;
  plot('xc-chart', [ln(zs, zs.map(x => q(x).r), A, K.red, `z1,t-1 = ${z1}`, { line: { width: 2 } }), hl(A, [-3, 3], 0.5, K.navy, 'rho_12,t-1 = 0.5', { line: { width: 1, dash: 'dash' } }),
    mk([z2], [Q.r], A, K.blue, `rho_12,t = ${Q.r.toFixed(3)}`, { marker: { size: 10, color: K.blue } }),
    ln(ks, path, B, K.red, `a + b = ${(a + b).toFixed(2)} (extra run)`, { line: { width: 2 } }), hl(B, [0, 120], E.qbar, K.green, 'q-bar_12 = 0.5', { line: { width: 1.2, dash: 'dash' } })], G.lay);
  steps('xc-ws', [T.xcs1(f2(a), f2(b), f3(c)), T.xcs2(f3(c), f2(a), f1(z1), f1(z2), f2(b), f3(c * E.qbar), a * z1 * z2 < 0 ? `(${f3(a * z1 * z2)})` : f3(a * z1 * z2), f3(b * E.q12), f3(Q.q12)),
    T.xcs3(f3(c), f2(a), f1(z1 * z1), f2(b), f3(Q.q11), f2(z2 * z2), nf(Q.q22, 4)), T.xcs4(f3(Q.q12), f3(Q.q11), nf(Q.q22, 4), f3(Q.r))]);
  out('xc-out', [T.xc1(f3(Q.r), f3(Q.q12), a + b >= 1 ? null : f0(hlife(a + b)))]);
  return Q;
}
function blkXh() {
  const p = val('xh-p'), h = hlife(p); set('xh-p', f3(p));
  const ks = range(0, 250, 1), pp = linspace(0.9, 0.999, 100), est = [[NB.dcc.ab, 'S&P 500 and DAX', K.red], [NB.panel.ab, 'five-asset panel', K.amber]];
  const G = grid('xh-chart', [{ r: 0, c: 0, t: 'share of a correlation shock left after k days', xt: 'k (days)', yt: '(a + b)^k', y: { range: [0, 1.02] } },
    { r: 0, c: 1, t: 'half-life against a + b', xt: 'a + b', yt: 'days', y: { type: 'log', tickvals: [5, 10, 20, 50, 100, 200, 500], ticktext: ['5', '10', '20', '50', '100', '200', '500'] } }], { cols: 2, h: 360, legendRows: 3 });
  const [A, B] = G.ax;
  const tr = [ln(ks, ks.map(k => p ** k), A, K.blue, `a + b = ${p.toFixed(3)}`, { line: { width: 2 } }), mk([h], [0.5], A, K.blue, '', { marker: { size: 9, color: K.blue } }),
    ln(pp, pp.map(hlife), B, K.blue, '', { line: { width: 2 } }), mk([p], [h], B, K.blue, '', { marker: { size: 10, color: K.blue } })];
  est.forEach(([e, n, c]) => { tr.push(ln(ks, ks.map(k => e ** k), A, c, `${n}: a + b = ${e.toFixed(4)}`, { line: { width: 1.2, dash: 'dash' } }), mk([e], [hlife(e)], B, c, '', { marker: { size: 9, color: c, symbol: 'diamond' } })); });
  plot('xh-chart', tr, G.lay);
  steps('xh-ws', [T.xhs1(f3(p), nf(Math.log(p), 5), f1(h)), T.xhs2(f0(hlife(0.98)), f0(hlife(0.99))), T.xhs3(f4(NB.dcc.ab), f0(NB.dcc.hl), f4(NB.panel.ab), f0(NB.panel.hl))]);
  out('xh-out', [T.xh1(f3(p), f1(h), pc(p ** 20, 0))]);
  return { h };
}

// ================================================================= 5. DCC for the S&P 500 and the DAX
function blkT1() {
  const s = $('t1-s').value, M = pairModel(NB.dcc.a, NB.dcc.b, 1), R = D.pair.r, P = D.pair.P;
  const st = c => ({ kr: kurt(R[c]), kz: kurt(M.Z[c]), sd: asd(M.Z[c]) });
  const S = { sp500: st('sp500'), dax: st('dax') }, rc = corr(R.sp500, R.dax), zc = M.qb;
  const rr = R[s], m = amean(rr), sd = asd(rr), x = Array.from(rr, v => (v - m) / sd), z = Array.from(M.Z[s]);
  const G = grid('t1-chart', [{ r: 0, c: 0, t: `${NAME[s]}: standardised returns and GARCH residuals z_t`, xt: 'value (standard deviations)', yt: 'density (log scale)', y: { type: 'log', range: [-4.3, 0], tickvals: [1e-4, 1e-3, 0.01, 0.1, 1], ticktext: ['0.0001', '0.001', '0.01', '0.1', '1'] } }], { cols: 1, h: 360, legendRows: 2 });
  const a = G.ax[0], hist = v => { const bins = linspace(-10, 10, 81), c = new Array(80).fill(0); v.forEach(u => { const k = Math.floor((u + 10) / 0.25); if (k >= 0 && k < 80) c[k]++; });
    return { x: bins.slice(0, 80).map(b => b + 0.125), y: c.map(n => (n ? n / v.length / 0.25 : null)) }; };
  const hx = hist(x), hz = hist(z), g = linspace(-6, 6, 121);
  plot('t1-chart', [ln(hx.x, hx.y, a, K.red, `returns / sample SD (excess kurtosis ${kurt(rr).toFixed(1)})`, { mode: 'lines+markers', marker: { size: 4, color: K.red }, connectgaps: false }),
    ln(hz.x, hz.y, a, COL[s], `z_t = (r_t - mu) / sigma_t (excess kurtosis ${kurt(z).toFixed(1)})`, { mode: 'lines+markers', marker: { size: 4, color: COL[s] }, connectgaps: false }),
    ln(g, g.map(u => Math.exp(-u * u / 2) / Math.sqrt(2 * Math.PI)), a, K.navy, 'Normal density', { line: { width: 1.2, dash: 'dash' } })], G.lay);
  table('t1-tb', T.t1Head, ['sp500', 'dax'].map(c => Object.assign([NAME[c], R_(f3(P[c].mu)), R_(f3(P[c].omega)), R_(f3(P[c].alpha)), R_(f3(P[c].beta)), R_(f3(P[c].alpha + P[c].beta)), R_(f1(S[c].kr)), R_(f1(S[c].kz))], { hi: c === s })),
    { note: T.t1Note(big(R.sp500.length), dmy(pairDates()[0]), dmy(pairDates()[R.sp500.length - 1]), f3(rc), f3(zc)) });
  out('t1-out', [T.t11(NAME[s], f1(S[s].kr), f1(S[s].kz), f3(S[s].sd))]);
  return { S, rc, zc };
}
function blkS1() {
  const m = val('s1-m'); set('s1-m', f2(m));
  const M = pairModel(NB.dcc.a, NB.dcc.b, m), d = pairDates();
  const G = grid('s1-chart', [{ r: 0, c: 0, yt: 'Annualised volatility (%)' }], { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0], S = {}, tr = [];
  ['sp500', 'dax'].forEach(c => { const v = Array.from(M.S[c], x => x * Math.sqrt(252)); S[c] = extremes(v, d); tr.push(ln(d, v, a, COL[c], `${NAME[c]}: GARCH(1,1) volatility`, { line: { width: 0.9 } })); });
  const top = Math.max(S.sp500.max, S.dax.max) * 1.05;
  tr.unshift(spans(a, CR(), 0, top, K.crimson, 0.15, 'Crisis periods (2008, 2020)'));
  plot('s1-chart', tr, G.lay);
  const P = D.pair.P, al = c => P[c].alpha * m, be = c => P[c].alpha + P[c].beta - al(c);
  out('s1-out', [T.s11(f0(S.sp500.max), dmy(S.sp500.dmax), f0(S.dax.max), dmy(S.dax.dmax)),
    T.s12(m === 1, f3(al('sp500')), f3(be('sp500')), f3(al('dax')), f3(be('dax')), f3(P.sp500.alpha + P.sp500.beta), f3(P.dax.alpha + P.dax.beta))]);
  return S;
}
function blkT2() {
  const b = val('t2-b'); set('t2-b', f4(b));
  const M = pairModel(NB.dcc.a, NB.dcc.b, 1), z1 = M.Z.sp500, z2 = M.Z.dax, qb = D.pair.fit.qbar[1];
  const ll0 = memo('ll0', () => dccLL2(z1, z2, 0, 0, qb)), llE = memo('llE', () => dccLL2(z1, z2, NB.dcc.a, NB.dcc.b, qb));
  const amax = Math.min(0.06, 0.9999 - b - 1e-6), as = linspace(0, amax, 41), ll = as.map(a => memo(`ll:${a}:${b}`, () => dccLL2(z1, z2, a, b, qb)));
  const ib = argmax(ll);
  const G = grid('t2-chart', [{ r: 0, c: 0, t: `LR against CCC along a, with b = ${b.toFixed(4)}`, xt: 'a', yt: '2 (l_C(a, b) - l_C(0, 0))' }], { cols: 1, h: 350, legendRows: 2 });
  const A = G.ax[0];
  plot('t2-chart', [ln(as, ll.map(v => 2 * (v - ll0)), A, K.red, 'LR(a, b) (extra run)', { line: { width: 2 } }), hl(A, [0, amax], NB.dcc.crit, K.navy, 'chi2(2) critical value 5.99', { line: { width: 1.2, dash: 'dash' } }),
    mk([as[ib]], [2 * (ll[ib] - ll0)], A, K.blue, `best a on the grid: ${as[ib].toFixed(4)}`, { marker: { size: 10, color: K.blue } }),
    mk([NB.dcc.a], [2 * (llE - ll0)], A, K.green, 'estimate (a, b) of the lecture', { marker: { size: 10, color: K.green, symbol: 'diamond' } })], G.lay);
  const F = NB.dcc;
  table('t2-tb', T.t2Head, [['a', R_(f4(F.a)), R_(f4(F.se_a))], ['b', R_(f4(F.b)), R_(f4(F.se_b))], ['a + b', R_(f4(F.ab)), ''], [T.t2hl, R_(f0(hlife(F.ab))), ''],
    ['l<sub>C</sub><sup>DCC</sup>', R_(f1(llE)), ''], ['l<sub>C</sub><sup>CCC</sup>', R_(f1(ll0)), ''], [T.t2lr, R_(f1(2 * (llE - ll0))), ''], [T.t2crit, R_(f2(F.crit)), '']], { note: T.t2Note });
  out('t2-out', [T.t21(f4(b), f4(as[ib]), f1(2 * (ll[ib] - ll0)), f1(2 * (llE - ll0)))]);
  return { llE, ll0, lr: 2 * (llE - ll0) };
}
function blkDc() {
  const a = val('dc-a'), w = val('dc-w'), b = NB.dcc.b; set('dc-a', f4(a)); set('dc-w', T.days(w));
  const M = pairModel(a, b, 1), d = pairDates(), R = D.pair.r, roll = memo('roll:sp500dax:' + w, () => rollCorr(R.sp500, R.dax, w));
  const G = grid('dc-chart', [{ r: 0, c: 0, yt: 'Correlation' }], { cols: 1, h: 380, legendRows: 2 });
  const A = G.ax[0];
  plot('dc-chart', [spans(A, CR(), 0, 1, K.crimson, 0.15, 'Crisis periods (2008, 2020)'), ln(d, M.rho, A, K.red, 'DCC correlation', { line: { width: 0.8 } }),
    ln(d.slice(w - 1), roll.slice(w - 1), A, K.blue, `Rolling correlation (${w} days)`, { line: { width: 1.3 } }),
    hl(A, [d[0], d[d.length - 1]], M.qb, K.green, 'CCC (constant) correlation', { line: { width: 1.5, dash: 'dash' } })], G.lay);
  const E = extremes(M.rho, d), Rr = extremes(roll, d);
  const ll = memo(`lld:${a}`, () => dccLL2(M.Z.sp500, M.Z.dax, a, b, M.qb)), ll0 = memo('ll0', () => dccLL2(M.Z.sp500, M.Z.dax, 0, 0, M.qb));
  out('dc-out', [T.dc1(f4(a), f4(a + b), f2(E.min), my(E.dmin), f2(E.max), my(E.dmax), f2(E.mean)), T.dc2(f3(E.sd), T.days(w), f3(Rr.sd), f1(2 * (ll - ll0)))]);
  return { E, Rr, lr: 2 * (ll - ll0) };
}
function crisisRows(rho, S, d, kind) {
  const P = [['calm', D.calm], ['2008', D.crises['2008']], ['2020', D.crises['2020']]], R = D.pair.r;
  return P.map(([k, [x, y]]) => { const ix = between(d, x, y);
    const c = kind === 'dcc' ? amean(ix.map(i => rho[i])) : kind === 'roll' ? amean(ix.map(i => memo('roll:sp500dax:250', () => rollCorr(R.sp500, R.dax, 250))[i])) : corr(ix.map(i => R.sp500[i]), ix.map(i => R.dax[i]));
    return { k, c, v: amean(ix.map(i => S.sp500[i])) * Math.sqrt(252), n: ix.length }; });
}
function blkTc() {
  const a = val('tc-a'), kind = $('tc-r').value; set('tc-a', f4(a));
  const M = pairModel(a, NB.dcc.b, 1), d = pairDates();
  const rows = crisisRows(M.rho, M.S, d, kind), rowsD = kind === 'dcc' ? rows : crisisRows(M.rho, M.S, d, 'dcc');
  const q = {}; d.forEach((s, i) => { const k = s.slice(0, 4) + 'Q' + (Math.floor((+s.slice(5, 7) - 1) / 3) + 1); (q[k] = q[k] || []).push(i); });
  const Q = Object.entries(q).map(([k, ix]) => ({ k, v: amean(ix.map(i => M.S.sp500[i])) * Math.sqrt(252), r: amean(ix.map(i => M.rho[i])) }));
  const G = grid('tc-chart', [{ r: 0, c: 0, t: 'quarterly means: S&P 500 volatility and DCC correlation', xt: 'mean annualised S&P 500 volatility (%)', yt: 'mean DCC correlation', x: { type: 'log' } }], { cols: 1, h: 380, legendRows: 2 });
  const A = G.ax[0], cols = { calm: K.green, 2008: K.red, 2020: K.purple };
  plot('tc-chart', [mk(Q.map(x => x.v), Q.map(x => x.r), A, K.blue, 'calendar quarters 2000-2026 (extra)', { marker: { size: 6, color: `rgba(${RGB[K.blue]},0.55)` }, text: Q.map(x => x.k), hovertemplate: '%{text}: %{x:.0f}%, %{y:.3f}<extra></extra>' }),
    ...rowsD.map(r => mk([r.v], [r.c], A, cols[r.k], T.perEn[r.k], { marker: { size: 13, color: cols[r.k], symbol: 'diamond' } }))], G.lay);
  table('tc-tb', T.tcHead(kind), rows.map(r => [T.per[r.k], R_(f2(r.c)), R_(f0(r.v)), R_(r.n)]), { note: T.tcNote(kind) });
  out('tc-out', [T.tc1(f2(rowsD[0].c), f2(rowsD[1].c), f2(rowsD[2].c), f0(rowsD[0].v), f0(rowsD[1].v), f0(rowsD[2].v)), T.tc2(f2(corr(Q.map(x => Math.log(x.v)), Q.map(x => x.r))))]);
  return { rows, rowsD };
}
function blkZm() {
  const m = val('zm-m'), a = val('zm-a'); set('zm-m', f2(m)); set('zm-a', f4(a));
  const M = pairModel(a, NB.dcc.b, m), d = pairDates(), win = { 2008: ['2008-06-01', '2009-06-30'], 2020: ['2019-12-01', '2020-08-31'] };
  const G = grid('zm-chart', [{ r: 0, c: 0, t: 'Crisis of 2008', yt: 'Correlation', y: { range: [0, 1] } }, { r: 0, c: 1, t: 'Crisis of 2020', yt: 'Correlation', y: { range: [0, 1] } }],
    { cols: 2, h: 380, legendRows: 2, mr: 44, gx: 0.14 });
  const tr = [], S = {};
  ['2008', '2020'].forEach((k, j) => {
    const A = G.ax[j], B = twin(G, A, 3 + j, 'Volatility (%)'), ix = between(d, ...win[k]), x = ix.map(i => d[i]), r = ix.map(i => M.rho[i]), v = ix.map(i => M.S.sp500[i] * Math.sqrt(252));
    tr.push(ln(x, r, A, K.red, j ? '' : 'DCC correlation (left axis)', { line: { width: 1.4 } }), ln(x, v, B, K.blue, j ? '' : 'S&P 500 GARCH volatility, % (right axis)', { line: { width: 1.2 } }));
    const ir = argmax(r), iv = argmax(v); S[k] = { rmax: r[ir], rdmax: x[ir], r0: r[0], vmax: v[iv], vdmax: x[iv] };
  });
  plot('zm-chart', tr, G.lay);
  out('zm-out', [T.zm1(f0(S[2008].vmax), dmy(S[2008].vdmax), f2(S[2008].r0), f2(S[2008].rmax), dmy(S[2008].rdmax)), T.zm2(f2(S[2020].rmax), dmy(S[2020].rdmax), f0(S[2020].vmax), dmy(S[2020].vdmax), S[2020].rdmax < S[2020].vdmax),
    T.zm3(m === 1 && Math.abs(a - NB.dcc.a) < 1e-12)]);
  return S;
}

// ================================================================= 6. five markets
const panelDates = () => expand(D.panel);
function blkTp() {
  const s = $('tp-s').value, P = D.panel.P, F = NB.panel, M = panelModel(F.a, F.b), d = panelDates();
  const v = Array.from(M.S[s], x => x * Math.sqrt(252)), lr = Math.sqrt(P[s].omega / (1 - P[s].alpha - P[s].beta) * 252), E = extremes(v, d);
  const G = grid('tp-chart', [{ r: 0, c: 0, t: `${NAME[s]}: GARCH(1,1) volatility of step 1`, yt: 'Annualised volatility (%)' }], { cols: 1, h: 340, legendRows: 2 });
  const A = G.ax[0];
  plot('tp-chart', [ln(d, v, A, COL[s], `${NAME[s]} (alpha + beta = ${(P[s].alpha + P[s].beta).toFixed(3)})`, { line: { width: 1 } }),
    hl(A, [d[0], d[d.length - 1]], lr, K.navy, `long-run level ${lr.toFixed(1)}%`, { line: { width: 1.2, dash: 'dash' } })], G.lay);
  table('tp-tb', T.tpHead, PK.map(c => Object.assign([NAME[c], R_(f3(P[c].mu)), R_(f4(P[c].omega)), R_(f3(P[c].alpha)), R_(f3(P[c].beta)), R_(f3(P[c].alpha + P[c].beta))], { hi: c === s })),
    { note: T.tpNote(big(d.length), dmy(d[0]), f4(F.a), f4(F.se_a), f4(F.b), f4(F.se_b), f0(hlife(F.ab)), f1(F.lr)) });
  out('tp-out', [T.tp1(NAME[s], f3(P[s].alpha + P[s].beta), f0(E.max), dmy(E.dmax), f0(lr))]);
  return { E, lr };
}
const PAIRS4 = [['bet', 'dax', K.red], ['sp500', 'dax', K.blue], ['sp500', 'btc', K.amber], ['bet', 'eurron', K.green]];
function blkPn() {
  const a = val('pn-a'), b = NB.panel.b; set('pn-a', f4(a));
  const M = panelModel(a, b), d = panelDates(), ix = Object.fromEntries(PK.map((k, i) => [k, i]));
  const G = grid('pn-chart', [{ r: 0, c: 0, yt: 'DCC correlation (daily returns)' }], { cols: 1, h: 380, legendRows: 2 });
  const A = G.ax[0], S = {}, all = [];
  const series = PAIRS4.map(([x, y, c]) => { const r = rhoOf(M.Rt, 5, ix[x], ix[y]); S[`${x}_${y}`] = extremes(r, d); all.push(...r); return ln(d, r, A, c, `${NAME[x]} and ${NAME[y]}`, { line: { width: 1.2 } }); });
  const lo = Math.min(...all) - 0.05, hi = Math.max(...all) + 0.05;
  plot('pn-chart', [spans(A, [D.panel_crisis], lo, hi, K.crimson, 0.15, 'COVID-19 crash (2020)'), ...series, hl(A, [d[0], d[d.length - 1]], 0, K.navy, '', { line: { width: 0.6, dash: 'dash' } })], G.lay);
  const s = k => S[k];
  out('pn-out', [T.pn1(f2(s('sp500_dax').mean), f2(s('bet_dax').mean), f2(s('bet_dax').min), f2(s('bet_dax').max), my(s('bet_dax').dmax)),
    T.pn2(f2(s('sp500_btc').min), f2(s('sp500_btc').max), my(s('sp500_btc').dmax), f2(s('sp500_btc').mean), f2(s('bet_eurron').mean)), T.pn3(f4(a), f4(a + b), f0(hlife(a + b)))]);
  return S;
}
function heatMeans(a) {
  const M = panelModel(a, NB.panel.b), d = panelDates(), mean = ix => { const o = new Float64Array(25); ix.forEach(i => { for (let j = 0; j < 25; j++) o[j] += M.Rt[i][j] / ix.length; }); return o; };
  const ic = between(d, D.calm[0], D.calm[1]), ik = between(d, D.panel_crisis[0], D.panel_crisis[1]);
  const calm = mean(ic), cris = mean(ik), off = M => { let s = 0; for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) if (i !== j) s += M[i * 5 + j]; return s / 20; };
  return { calm, cris, ca: off(calm), cr: off(cris), nc: ik.length };
}
function blkHt() {
  const a = val('ht-a'); set('ht-a', f4(a));
  const H = heatMeans(a), labs = PK.map(k => NAME[k]);
  const G = grid('ht-chart', [{ r: 0, c: 0, t: 'Calm: 2017-2019', y: { autorange: 'reversed' }, x: { tickangle: -40 } }, { r: 0, c: 1, t: 'COVID-19 crash: Feb-Jun 2020', y: { autorange: 'reversed' }, x: { tickangle: -40 } }],
    { cols: 2, h: 420, legend: false, ml: 70, mb: 70, gx: 0.16, rowPx: 300 });
  const RDBU = [[0, '#053061'], [0.1, '#2166ac'], [0.2, '#4393c3'], [0.3, '#92c5de'], [0.4, '#d1e5f0'], [0.5, '#f7f7f7'], [0.6, '#fddbc7'], [0.7, '#f4a582'], [0.8, '#d6604d'], [0.9, '#b2182b'], [1, '#67001f']];
  const rows = M => [0, 1, 2, 3, 4].map(i => Array.from(M.slice(5 * i, 5 * i + 5)));
  const hm = (M, A, sc) => ({ type: 'heatmap', z: rows(M), x: labs, y: labs, xaxis: A.xa, yaxis: A.ya, zmin: -1, zmax: 1, colorscale: RDBU, showscale: sc,
    colorbar: sc ? { title: { text: 'Average DCC correlation', side: 'right' }, len: 0.9, thickness: 12 } : undefined, hovertemplate: '%{y} and %{x}: %{z:.3f}<extra></extra>' });
  const tx = (M, A) => { const x = [], y = [], t = [], c = []; rows(M).forEach((r, i) => r.forEach((v, j) => { x.push(labs[j]); y.push(labs[i]); t.push(v.toFixed(2)); c.push(Math.abs(v) > 0.6 ? '#FFFFFF' : K.navy); }));
    return { x, y, text: t, mode: 'text', textfont: { size: 11, color: c }, xaxis: A.xa, yaxis: A.ya, showlegend: false, hoverinfo: 'skip', type: 'scatter' }; };
  plot('ht-chart', [hm(H.calm, G.ax[0], false), hm(H.cris, G.ax[1], !G.narrow), tx(H.calm, G.ax[0]), tx(H.cris, G.ax[1])], G.lay);
  const ix = Object.fromEntries(PK.map((k, i) => [k, i])), g = (M, x, y) => M[ix[x] * 5 + ix[y]];
  out('ht-out', [T.ht1(f2(H.ca), f2(H.cr), H.nc), T.ht2(f2(g(H.calm, 'bet', 'dax')), f2(g(H.cris, 'bet', 'dax')), f2(g(H.calm, 'sp500', 'bet')), f2(g(H.cris, 'sp500', 'bet')), f2(g(H.calm, 'dax', 'btc')), f2(g(H.cris, 'dax', 'btc'))),
    T.ht3(f2(g(H.calm, 'bet', 'eurron')), f2(g(H.cris, 'bet', 'eurron')))]);
  return H;
}

// ================================================================= 7. portfolio VaR
function blkXv() {
  const r = val('xv-r'), s1 = val('xv-a'), s2 = val('xv-b'), V = val('xv-v'), z = NB.var.zq; set('xv-r', f2(r)); set('xv-a', f1(s1) + '%'); set('xv-b', f1(s2) + '%'); set('xv-v', T.mil(V));
  const sp = x => sigP(0.5, s1, s2, x), v = sp(r), VaR = -z * v, rr = linspace(-0.5, 1, 61);
  const G = grid('xv-chart', [{ r: 0, c: 0, t: 'VaR 1% against the correlation', xt: 'rho_t', yt: 'VaR 1% (%)' }, { r: 0, c: 1, t: 'density of the portfolio return', xt: 'return (%)', yt: 'density' }], { cols: 2, h: 350, legendRows: 2 });
  const [A, B] = G.ax, x = linspace(-5, 5, 201), dn = u => Math.exp(-0.5 * (u / v) ** 2) / (v * Math.sqrt(2 * Math.PI)), xl = x.filter(u => u <= -VaR);
  plot('xv-chart', [ln(rr, rr.map(q => -z * sp(q)), A, K.blue, 'VaR 1% = 2.326 sigma_p', { line: { width: 2 } }),
    mk(D.ex.var_rho, D.ex.var_rho.map(q => -z * sp(q)), A, K.green, 'rho = 0.3 and 0.9', { marker: { size: 9, color: K.green } }), mk([r], [VaR], A, K.red, `rho = ${r.toFixed(2)}: ${VaR.toFixed(2)}%`, { marker: { size: 11, color: K.red, symbol: 'diamond' } }),
    ln(x, x.map(dn), B, K.blue, '', { line: { width: 2 } }), ln(xl.concat([-VaR, -VaR]), xl.map(dn).concat([dn(-VaR), 0]), B, K.red, '1% left tail', { fill: 'tozeroy', fillcolor: `rgba(${RGB[K.red]},0.35)`, line: { width: 1 } })], G.lay);
  const h12 = r * s1 * s2;
  steps('xv-ws', [T.xvs1(f1(s1), f1(s2), f2(r), f2(s1 * s1), f2(s2 * s2), f3(h12)), T.xvs2(f2(s1 * s1), f2(s2 * s2), f3(h12), nf(v * v, 4), f3(v)),
    T.xvs3(f3(-z), f3(v), f2(VaR)), T.xvs4(T.mil(V), f1(VaR / 100 * V * 1000)), T.xvs5(f2(-z * sp(0.3)), f2(-z * sp(0.9)), pc(sp(0.9) / sp(0.3) - 1, 0))]);
  out('xv-out', [T.xv1(f2(r), f3(v), f2(VaR), f1(VaR / 100 * V * 1000))]);
  return { v, VaR };
}
const LEVELS = [0.005, 0.01, 0.025, 0.05];
function kupiec(viol, al) {
  const n = viol.length, x = viol.reduce((s, v) => s + v, 0), p = x / n;
  const l0 = (n - x) * Math.log(1 - al) + x * Math.log(al), l1 = (n - x) * Math.log(1 - p) + (x > 0 ? x * Math.log(p) : 0), lr = -2 * (l0 - l1);
  return { n, x, rate: p, lr, p: chi1(lr) };
}
function christoffersen(v) {
  let n00 = 0, n01 = 0, n10 = 0, n11 = 0;
  for (let t = 1; t < v.length; t++) { const a = v[t - 1], b = v[t]; if (!a && !b) n00++; else if (!a && b) n01++; else if (a && !b) n10++; else n11++; }
  const p01 = n01 / Math.max(n00 + n01, 1), p11 = n11 / Math.max(n10 + n11, 1), p = (n01 + n11) / (v.length - 1), xl = (k, q) => (k > 0 ? k * Math.log(q) : 0);
  const l0 = xl(n00 + n10, 1 - p) + xl(n01 + n11, p), l1 = xl(n00, 1 - p01) + xl(n01, p01) + xl(n10, 1 - p11) + xl(n11, p11), lr = -2 * (l0 - l1);
  return { n11, p01, p11, lr, p: chi1(lr) };
}
// the chapter's portfolio_var: parameters on 2000-2014, filters run on 2000-2026; VaR at level al
function varModel(al) {
  return memo('var:' + al, () => {
    const B = D.pair, V = D.var, d = pairDates(), w = D.weights, S1 = memo('varS1', () => step1(B, V.P, V.s0, 1));
    const qb = V.fit.qbar, Rt = dccPath(S1.Z, ['sp500', 'dax'], V.fit.a, V.fit.b, [[qb[0], qb[1]], [qb[2], qb[3]]]);
    const n = d.length, s1 = S1.S.sp500, s2 = S1.S.dax, rp = new Float64Array(n), sd = new Float64Array(n), sc = new Float64Array(n);
    const m = w[0] * V.P.sp500.mu + w[1] * V.P.dax.mu, rc = qb[1];
    const ins = d.map(x => x <= D.split), Rin = { sp500: [], dax: [] };
    for (let t = 0; t < n; t++) {
      rp[t] = w[0] * B.r.sp500[t] + w[1] * B.r.dax[t]; const r12 = Rt[t][1];
      sd[t] = Math.sqrt((w[0] * s1[t]) ** 2 + (w[1] * s2[t]) ** 2 + 2 * w[0] * w[1] * r12 * s1[t] * s2[t]);
      sc[t] = Math.sqrt((w[0] * s1[t]) ** 2 + (w[1] * s2[t]) ** 2 + 2 * w[0] * w[1] * rc * s1[t] * s2[t]);
      if (ins[t]) { Rin.sp500.push(B.r.sp500[t]); Rin.dax.push(B.r.dax[t]); }
    }
    const cv = (x, y) => { const mx = amean(x), my = amean(y); let s = 0; for (let i = 0; i < x.length; i++) s += (x[i] - mx) * (y[i] - my); return s / (x.length - 1); };
    const ss = Math.sqrt(w[0] ** 2 * cv(Rin.sp500, Rin.sp500) + w[1] ** 2 * cv(Rin.dax, Rin.dax) + 2 * w[0] * w[1] * cv(Rin.sp500, Rin.dax));
    const zq = al === 0.01 ? NB.var.zq : normPpf(al), qf = qtl(d.map((_, t) => t).filter(t => ins[t]).map(t => (rp[t] - m) / sd[t]), al);
    const VaR = { DCC: Array.from(sd, x => -(m + zq * x)), 'DCC-FHS': Array.from(sd, x => -(m + qf * x)), CCC: Array.from(sc, x => -(m + zq * x)), Static: d.map(() => -(m + zq * ss)) };
    const oi = d.map((_, t) => t).filter(t => !ins[t]), c20 = between(d, ...D.crises['2020']), res = { zq, qf, n_in: n - oi.length };
    Object.entries(VaR).forEach(([k, v]) => { const vi = oi.map(t => (rp[t] < -v[t] ? 1 : 0));
      res[k] = Object.assign(kupiec(vi, al), { ind: christoffersen(vi), mean_var: amean(oi.map(t => v[t])), max_var: Math.max(...oi.map(t => v[t])), viol2020: c20.filter(t => rp[t] < -v[t]).length, vi }); });
    return { d, rp, VaR, oi, res };
  });
}
const VMOD = ['DCC', 'DCC-FHS', 'CCC', 'Static'];
function blkVb() {
  const al = +$('vb-l').value, mo = $('vb-m').value, M = varModel(al), d = M.d, oi = M.oi, x = oi.map(t => d[t]);
  const G = grid('vb-chart', [{ r: 0, c: 0, yt: 'Daily return (%)' }], { cols: 1, h: 400, legendRows: 3 });
  const A = G.ax[0], lab = `${trim(100 * al, 1)}%`, tr = [ln(x, oi.map(t => M.rp[t]), A, K.blue, 'Portfolio return (50% S&P 500, 50% DAX), %', { line: { width: 0.5 } })];
  [[mo, K.red], ['Static', K.green]].forEach(([k, c]) => { const v = M.VaR[k], hit = oi.filter(t => M.rp[t] < -v[t]);
    tr.push(ln(x, oi.map(t => -v[t]), A, c, `minus VaR ${lab}: ${k}`, { line: { width: 1 } }), mk(hit.map(t => d[t]), hit.map(t => M.rp[t]), A, c, `Violations: ${k} (${hit.length})`, { marker: { symbol: 'triangle-down', size: 6, color: c } })); });
  plot('vb-chart', tr, G.lay);
  const R = M.res, ex = al * oi.length;
  out('vb-out', [T.vb1(lab, mo, R[mo].x, R.Static.x, f0(ex), big(oi.length)), T.vb2(mo, pc(R[mo].rate, 2), f1(R[mo].lr), pvr(R[mo].p), pc(R.Static.rate, 2), f1(R.Static.lr), pvr(R.Static.p)),
    T.vb3(lab, f2(R.Static.mean_var), f2(R[mo].mean_var), f2(R[mo].max_var), mo === 'DCC-FHS' ? f2(M.res.qf) : null)]);
  return R;
}
function blkTv() {
  const al = +$('tv-l').value, M = varModel(al), R = M.res, lab = `${trim(100 * al, 1)}%`, oi = M.oi, d = M.d;
  const G = grid('tv-chart', [{ r: 0, c: 0, t: `violations of VaR ${lab}, 2015-2026`, yt: 'violations' }, { r: 0, c: 1, t: 'violations per year', yt: 'violations' }], { cols: 2, h: 360, legendRows: 2 });
  const [A, B] = G.ax, cols = { DCC: K.red, 'DCC-FHS': K.orange, CCC: K.purple, Static: K.green };
  const yrs = [...new Set(oi.map(t => d[t].slice(0, 4)))], tr = [bar(VMOD, VMOD.map(k => R[k].x), A, VMOD.map(k => cols[k]), '', { showlegend: false }),
    hl(A, [VMOD[0], VMOD[3]], al * oi.length, K.navy, `expected: ${(al * oi.length).toFixed(0)}`, { line: { dash: 'dash', width: 1.2 } })];
  VMOD.forEach(k => { const c = yrs.map(y => oi.filter((t, j) => R[k].vi[j] && d[t].slice(0, 4) === y).length); tr.push(ln(yrs, c, B, cols[k], k, { mode: 'lines+markers', marker: { size: 5, color: cols[k] } })); });
  plot('tv-chart', tr, G.lay);
  table('tv-tb', T.tvHead, VMOD.map(k => [T.vmod[k], R_(R[k].x), R_(pc(R[k].rate, 2)), R_(f1(R[k].lr)), R_(pv(R[k].p)), R_(pv(R[k].ind.p)), R_(R[k].viol2020)]),
    { note: T.tvNote(lab, f0(al * oi.length), big(oi.length), big(R.n_in), f2(R.qf)) });
  out('tv-out', [T.tv1(lab, R.Static.ind.n11, R.Static.x, pc(R.Static.ind.p11, 1), R.Static.viol2020), T.tv2(pc(R['DCC-FHS'].rate, 2), pc(R.DCC.rate, 2), f2(R.qf))]);
  return R;
}

// ================================================================= 8. hedging
function blkXg() {
  const r = val('xg-r'), ss = val('xg-s'), sf = val('xg-f'); set('xg-r', f2(r)); set('xg-s', f1(ss) + '%'); set('xg-f', f1(sf) + '%');
  const cov = r * ss * sf, h = cov / (sf * sf), he = r * r, sh = ss * Math.sqrt(1 - r * r), hs = linspace(-0.5, 2.5, 121), vh = x => ss * ss - 2 * x * cov + x * x * sf * sf;
  const G = grid('xg-chart', [{ r: 0, c: 0, t: 'variance of r_s - h r_f', xt: 'h (units of f sold)', yt: 'variance (%^2)' }, { r: 0, c: 1, t: 'hedging effectiveness HE = rho^2', xt: 'rho', yt: 'HE', y: { range: [0, 1.02] } }], { cols: 2, h: 350, legendRows: 2 });
  const [A, B] = G.ax, rr = linspace(-1, 1, 101);
  plot('xg-chart', [ln(hs, hs.map(vh), A, K.blue, `rho = ${r.toFixed(2)}`, { line: { width: 2 } }), mk([h], [vh(h)], A, K.red, `h* = ${h.toFixed(2)}`, { marker: { size: 10, color: K.red } }),
    ln(rr, rr.map(x => x * x), B, K.navy, '', { line: { width: 1.5 } }), mk([r], [he], B, K.red, `HE = ${he.toFixed(2)}`, { marker: { size: 10, color: K.red, symbol: 'diamond' } })], G.lay);
  steps('xg-ws', [T.xgs1(f2(r), f1(ss), f1(sf), f2(cov)), T.xgs2(f2(cov), f2(sf * sf), f2(h)), T.xgs3(f2(r), f2(he), f1(ss), f2(sh)), T.xgs4(f1(ss), nf(0.5 * ss / 2, 2))]);
  out('xg-out', [T.xg1(f2(h), pc(he, 0), f2(sh), f1(ss))]);
  return { cov, h, he, sh };
}
function hedgeModel(a, w) {
  return memo(`hedge:${a}:${w}`, () => {
    const B = D.hedge, d = expand(B), S1 = memo('hedgeS1', () => step1(B, B.P, B.s0, 1)), qb = B.fit.qbar;
    const Rt = dccPath(S1.Z, ['bet', 'dax'], a, B.fit.b, [[qb[0], qb[1]], [qb[2], qb[3]]]);
    const hd = Rt.map((R, t) => R[1] * S1.S.bet[t] / S1.S.dax[t]);
    const ins = d.map(x => x <= D.split), ii = d.map((_, t) => t).filter(t => ins[t]), oi = d.map((_, t) => t).filter(t => !ins[t]);
    const xb = ii.map(t => B.r.bet[t]), xd = ii.map(t => B.r.dax[t]), mb = amean(xb), md = amean(xd);
    let sxy = 0, sxx = 0; for (let i = 0; i < xb.length; i++) { sxy += (xb[i] - mb) * (xd[i] - md); sxx += (xd[i] - md) ** 2; }
    const hs = sxy / sxx, hr = memo('rs:' + w, () => rollSlope(B.r.bet, B.r.dax, w)), un = oi.map(t => B.r.bet[t]), vu = avar(un), o = { static: hs, var_un: vu, n_oos: oi.length };
    [['static', () => hs], ['rolling', t => hr[t]], ['dcc', t => hd[t]]].forEach(([k, h]) => { const e = oi.map(t => B.r.bet[t] - h(t) * B.r.dax[t]); o['var_' + k] = avar(e); o['he_' + k] = 1 - avar(e) / vu; });
    const HD = extremes(oi.map(t => hd[t]), oi.map(t => d[t]));
    return Object.assign(o, { d, hd, hr, oi, HD });
  });
}
function blkHg() {
  const w = val('hg-w'), a = val('hg-a'); set('hg-w', T.days(w)); set('hg-a', f4(a));
  const H = hedgeModel(a, w), x = H.oi.map(t => H.d[t]);
  const G = grid('hg-chart', [{ r: 0, c: 0, yt: 'Units of DAX sold per unit of BET' }], { cols: 1, h: 380, legendRows: 2 });
  const A = G.ax[0];
  plot('hg-chart', [ln(x, H.oi.map(t => H.hd[t]), A, K.red, 'DCC hedge ratio', { line: { width: 0.7 } }), ln(x, H.oi.map(t => H.hr[t]), A, K.blue, `Rolling OLS (${w} days)`, { line: { width: 1.4 } }),
    hl(A, [x[0], x[x.length - 1]], H.static, K.green, 'Static OLS (2005-2014)', { line: { width: 1.5, dash: 'dash' } })], G.lay);
  out('hg-out', [T.hg1(f2(H.static), f2(H.HD.mean), f2(H.HD.min), f2(H.HD.max), dmy(H.HD.dmax)), T.hg2(pc(H.he_static), T.days(w), pc(H.he_rolling), pc(H.he_dcc), f4(a))]);
  return H;
}
function blkTh() {
  const w = val('th-w'); set('th-w', T.days(w));
  const H = hedgeModel(NB.hedge.a, w), d = H.d, oi = H.oi, B = D.hedge;
  const yrs = [...new Set(oi.map(t => d[t].slice(0, 4)))], ks = [['static', K.green], ['rolling', K.blue], ['dcc', K.red]];
  const hOf = { static: () => H.static, rolling: t => H.hr[t], dcc: t => H.hd[t] };
  const heY = k => yrs.map(y => { const ix = oi.filter(t => d[t].slice(0, 4) === y), u = ix.map(t => B.r.bet[t]), e = ix.map(t => B.r.bet[t] - hOf[k](t) * B.r.dax[t]); return 1 - avar(e) / avar(u); });
  const G = grid('th-chart', [{ r: 0, c: 0, t: 'hedging effectiveness out of sample', yt: 'HE (%)' }, { r: 0, c: 1, t: 'hedging effectiveness per year', yt: 'HE (%)' }], { cols: 2, h: 360, legendRows: 2 });
  const [A, Bx] = G.ax;
  plot('th-chart', [bar(ks.map(([k]) => T.hm[k]), ks.map(([k]) => 100 * H['he_' + k]), A, ks.map(([, c]) => c), '', { showlegend: false, text: ks.map(([k]) => (100 * H['he_' + k]).toFixed(1) + '%'), textposition: 'outside', cliponaxis: false }),
    ...ks.map(([k, c]) => ln(yrs, heY(k).map(v => 100 * v), Bx, c, T.hmEn[k] + (k === 'rolling' ? ` (${w} days)` : ''), { mode: 'lines+markers', marker: { size: 5, color: c } }))], G.lay);
  table('th-tb', T.thHead, ks.map(([k]) => [T.hmT[k] + (k === 'rolling' ? ` (${T.days(w)})` : ''), R_(k === 'static' ? f2(H.static) : f2(amean(oi.map(t => hOf[k](t))))), R_(f3(H['var_' + k])), R_(pc(H['he_' + k]))]),
    { note: T.thNote(big(H.n_oos), f3(H.var_un)) });
  out('th-out', [T.th1(pc(H.he_static), pc(H.he_rolling), pc(H.he_dcc), T.days(w))]);
  return H;
}

// ================================================================= web lecture: the extra blocks travel inside a chart block; move each to its own frame
function placeExtras() {
  document.querySelectorAll('.xhold > section.xblk').forEach(sec => {
    const art = document.getElementById('s' + sec.dataset.frame), head = art && art.querySelector('.fhead');
    if (head) head.after(sec); else sec.closest('section.blk').after(sec);
  });
  document.querySelectorAll('.xhold').forEach(h => { if (!h.children.length) h.remove(); });
}

// ================================================================= wiring
// [anchor, prefix, function, input moved by ▶, its values (null: the options of a select), ms]
function blkList() {
  return [['wx-div', 'xd', blkXd, 'xd-r', range(-1, 1, 0.1), 400], ['divers', 'dv', blkDv, 'dv-w', range(0.1, 0.9, 0.1), 500], ['rolling', 'ro', blkRo, 'ro-w', range(60, 500, 40), 600],
    ['t-async', 'ta', blkTa, 'ta-k', range(1, 10, 1), 600], ['wx-pd', 'xp', blkXp, 'xp-h', range(3, -3, -0.25), 350], ['wx-bekk', 'xb', blkXb, 'xb-e', range(-3, 3, 0.25), 350],
    ['t-dim', 'tm', blkTm, 'tm-n', range(2, 50, 4), 500], ['params', 'pc', blkPc, 'pc-n', range(5, 100, 5), 400], ['wx-dcc', 'xc', blkXc, 'xc-z', range(-3, 3, 0.25), 350],
    ['wx-hl', 'xh', blkXh, 'xh-p', range(0.9, 0.995, 0.005), 350], ['t-step1', 't1', blkT1, 't1-s', null, 1200], ['step1', 's1', blkS1, 's1-m', range(0.25, 2, 0.25), 600],
    ['t-dcc', 't2', blkT2, 't2-b', range(0.9, 0.99, 0.01), 500], ['dcc', 'dc', blkDc, 'dc-a', range(0, 0.022, 0.002), 500], ['t-crisis', 'tc', blkTc, 'tc-a', range(0, 0.022, 0.002), 500],
    ['zoom', 'zm', blkZm, 'zm-m', range(0.25, 2, 0.25), 600], ['t-panel', 'tp', blkTp, 'tp-s', null, 1200], ['panel', 'pn', blkPn, 'pn-a', range(0, 0.02, 0.002), 500],
    ['heat', 'ht', blkHt, 'ht-a', range(0, 0.02, 0.002), 500], ['wx-var', 'xv', blkXv, 'xv-r', range(-0.5, 1, 0.1), 400], ['var', 'vb', blkVb, 'vb-l', null, 1200],
    ['t-var', 'tv', blkTv, 'tv-l', null, 1200], ['wx-hedge', 'xg', blkXg, 'xg-r', range(-1, 1, 0.1), 400], ['hedge', 'hg', blkHg, 'hg-w', range(60, 500, 40), 600],
    ['t-hedge', 'th', blkTh, 'th-w', range(60, 500, 40), 600]];
}
function init() {
  placeExtras();
  const live = blkList().filter(b => has(b[1] + '-out')), players = {};
  live.forEach(([anchor, x, f, pid, vals, ms]) => {
    const blk = $(anchor) || $(x + '-out').closest('.blk, .wlive');
    (blk ? [...blk.querySelectorAll('input, select')] : []).forEach(el => el.addEventListener('input', f));
    if (!has(x + '-play')) return;
    const el = $(pid), v = vals || [...el.options].map(o => o.value);
    players[anchor] = new Player(x + '-play', el, v, ms);
  });
  Object.entries(T.predict).forEach(([b, sp]) => { if (has(b) && players[b]) predict(b, Object.assign({ player: players[b], veil: { out: true } }, sp)); });
  exitTicket(T.ticket);
  live.forEach(b => b[2]());
  let w = window.innerWidth, tm = null;
  window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(() => { if ((window.innerWidth < 720) !== (w < 720) || Math.abs(window.innerWidth - w) > 200) { w = window.innerWidth; live.forEach(b => b[2]()); } }, 250); });
}
window.CH14 = { garchSigma, step1, dccPath, dccLL2, rollCorr, rollSlope, pairModel, panelModel, heatMeans, varModel, hedgeModel, asyncStats, nParams, kupiec, christoffersen, blkList, init,
  blk: { blkXd, blkDv, blkRo, blkTa, blkXp, blkXb, blkTm, blkPc, blkXc, blkXh, blkT1, blkS1, blkT2, blkDc, blkTc, blkZm, blkTp, blkPn, blkHt, blkXv, blkVb, blkTv, blkXg, blkHg, blkTh } };
if (!window.CH14_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
