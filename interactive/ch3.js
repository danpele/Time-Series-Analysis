// Chapter 3 interactive page (unit roots and ARIMA models): one block per chart slide, computed in the browser from ch3_data.js.
(function () {
'use strict';
const D = window.CH3_DATA;
const { sum, solve, acf, pacf, ljungBox, normPpf, normCdf, mean, sdev, median, nf, pct, $, AX, layout, plot, out, Player, range,
  predict, exitTicket } = window.TSAI;
const T = window.CH3_TEXT;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of generate_all_charts.py / tsa_style.py
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', orange: '#E67E22', purple: '#8E44AD',
  teal: '#17A2B8', navy: '#1F2A44' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.purple]: '142,68,173', [K.orange]: '230,126,34' };

// ---------------------------------------------------------------- small helpers
const lags = n => Array.from({ length: n }, (_, i) => i + 1);
const pvar = x => { const m = mean(x); return sum(x.map(v => (v - m) ** 2)) / x.length; };
const pe = p => (p < 0.001 ? 'p < ' + nf(0.001, 3) : 'p = ' + nf(p, 3));
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
function gammaln(z) {
  const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = z, y = z, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015;
  for (const c of g) s += c / ++y;
  return -t + Math.log(2.5066282746310005 * s / x);
}
function chi2sf(x, k) {           // P(chi2_k > x), regularised upper incomplete gamma Q(k/2, x/2)
  const a = k / 2, z = x / 2;
  if (z <= 0) return 1;
  if (z < a + 1) { let ap = a, s = 1 / a, d = s; for (let n = 0; n < 500; n++) { d *= z / ++ap; s += d; if (Math.abs(d) < Math.abs(s) * 1e-14) break; }
    return 1 - s * Math.exp(-z + a * Math.log(z) - gammaln(a)); }
  let b = z + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 500; i++) { const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300; c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-14) break; }
  return Math.exp(-z + a * Math.log(z) - gammaln(a)) * h;
}
function betacf(a, b, x) {        // continued fraction of the incomplete beta function (Numerical Recipes)
  let c = 1, d = 1 - (a + b) * x / (a + 1); if (Math.abs(d) < 1e-300) d = 1e-300; d = 1 / d; let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m; let aa = m * (b - m) * x / ((a - 1 + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; h *= d * c;
    aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1 + m2));
    d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del;
    if (Math.abs(del - 1) < 1e-14) break;
  }
  return h;
}
function tsf2(t, df) {            // two-sided p-value of a Student t statistic
  const x = df / (df + t * t), a = df / 2, b = 0.5;
  const bt = Math.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
function ols(cols, y) {           // OLS of y on the columns: coefficients, standard errors, t-ratios, residuals, R^2
  const n = y.length, k = cols.length, G = cols.map(a => cols.map(b => { let s = 0; for (let t = 0; t < n; t++) s += a[t] * b[t]; return s; }));
  const gy = cols.map(a => { let s = 0; for (let t = 0; t < n; t++) s += a[t] * y[t]; return s; }), b = solve(G, gy);
  const e = y.map((v, t) => v - cols.reduce((s, c, j) => s + b[j] * c[t], 0)), sse = sum(e.map(v => v * v)), s2 = sse / (n - k);
  const inv = range(0, k - 1, 1).map(j => solve(G, range(0, k - 1, 1).map(i => (i === j ? 1 : 0)))[j]);
  const se = inv.map(v => Math.sqrt(s2 * v)), my = mean(y), sst = sum(y.map(v => (v - my) ** 2));
  return { b, se, t: b.map((v, j) => v / se[j]), e, r2: 1 - sse / sst, s2 };
}
const dw = e => { let a = 0; for (let t = 1; t < e.length; t++) a += (e[t] - e[t - 1]) ** 2; return a / sum(e.map(v => v * v)); };
const ones = n => new Array(n).fill(1);
const q2d = s => s.slice(0, 4) + '-' + String(3 * +s[5] - 2).padStart(2, '0') + '-01';      // '2020Q2' -> '2020-04-01'
const m2d = s => s + '-01';
const addM = (s, k) => { const y = +s.slice(0, 4), m = +s.slice(5, 7) - 1 + k; return `${y + Math.floor(m / 12)}-${String(m % 12 + 1).padStart(2, '0')}`; };
const addQ = (s, k) => { const i = +s.slice(0, 4) * 4 + (+s[5] - 1) + k; return `${Math.floor(i / 4)}Q${i % 4 + 1}`; };
const days = o => { const t0 = Date.parse(o.d0 + 'T00:00:00Z'), r = [o.d0]; let t = t0; o.gap.forEach(g => { t += g * 864e5; r.push(new Date(t).toISOString().slice(0, 10)); }); return r; };
const keysSorted = o => Object.keys(o).sort();

// ---------------------------------------------------------------- panels: one Plotly figure with several axes; stacked on narrow screens
// cells: [{ r, c, cs, t (title), xt, yt (axis titles), x, y (extra axis options) }]
function grid(id, cells, o) {   // eslint-disable-line no-param-reassign
  const el = $(id), narrow = el.clientWidth < 560, maxc = Math.floor(el.clientWidth / 6.6);
  const wrap = t => { if (!t || t.replace(/<[^>]+>/g, '').length <= maxc) return t; const w = t.split(' '), L = ['']; w.forEach(x => { const c = L[L.length - 1];
    if ((c + ' ' + x).replace(/<[^>]+>/g, '').trim().length > maxc && c) L.push(x); else L[L.length - 1] = (c + ' ' + x).trim(); }); return L.join('<br>'); };
  if (narrow) cells = cells.map(c => Object.assign({}, c, { t: wrap(c.t) }));
  const extra = narrow ? Math.max(0, ...cells.map(c => (c.t || '').split('<br>').length - 1)) * 15 : 0;
  const C = narrow ? 1 : o.cols, R = narrow ? cells.length : Math.max(...cells.map(c => c.r)) + 1;
  const H = narrow ? Math.max(o.h || 360, 120 + (210 + extra) * cells.length) + 26 * ((o.legendRows || 1) - 1) : (o.h || 360);
  const plotH = H - 70 - (o.legendRows || 1) * 26;
  const gx = o.gx || 0.07, gy = Math.min(0.5, (narrow ? 62 + extra : (o.gyPx || 62)) / plotH), w = (1 - gx * (C - 1)) / C, h = (1 - gy * (R - 1)) / R;
  const lay = layout({ height: H, margin: { l: 50, r: 10, t: 26 + extra, b: 34 }, annotations: [], shapes: [], showlegend: o.legend !== false,
    legend: { orientation: 'h', x: 0, xanchor: 'left', y: -(narrow ? 34 : 40) / plotH, yanchor: 'top', bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } } });
  const ax = cells.map((c, k) => {
    const r = narrow ? k : c.r, cc = narrow ? 0 : c.c, cs = narrow ? 1 : (c.cs || 1);
    const x0 = cc * (w + gx), x1 = x0 + cs * w + (cs - 1) * gx, y1 = 1 - r * (h + gy), y0 = y1 - h, n = k ? k + 1 : '';
    lay['xaxis' + n] = AX(c.xt || '', Object.assign({ domain: [x0, Math.min(1, x1)], anchor: 'y' + n, automargin: false, title: { text: c.xt || '', font: { size: 11, color: K.navy }, standoff: 4 } }, c.x || {}));
    lay['yaxis' + n] = AX(c.yt || '', Object.assign({ domain: [Math.max(0, y0), y1], anchor: 'x' + n, automargin: false, title: { text: c.yt || '', font: { size: 11, color: K.navy }, standoff: 2 } }, c.y || {}));
    if (c.t) lay.annotations.push({ xref: 'paper', yref: 'paper', x: x0, y: y1, xanchor: 'left', yanchor: 'bottom', yshift: 3, showarrow: false, text: c.t, font: { size: 12, color: K.navy } });
    return { xa: 'x' + n, ya: 'y' + n, xd: 'x' + n + ' domain', yd: 'y' + n + ' domain' };
  });
  el.style.height = H + 'px';
  return { lay, ax, narrow };
}
const hl = (a, y, color, dash, width) => ({ type: 'line', xref: a.xd, x0: 0, x1: 1, yref: a.ya, y0: y, y1: y, line: { color, width: width || 1.2, dash: dash || 'solid' } });
const vl = (a, x, color, dash, width) => ({ type: 'line', yref: a.yd, y0: 0, y1: 1, xref: a.xa, x0: x, x1: x, line: { color, width: width || 1.2, dash: dash || 'solid' } });
const leg = (name, line, marker) => Object.assign({ x: [null], y: [null], name, hoverinfo: 'skip' }, line ? { mode: 'lines', line } : { mode: 'markers', marker });
const ln = (x, y, a, color, name, o) => Object.assign({ x, y, mode: 'lines', line: { color, width: 1.3 }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya }, o || {});
function acfPanel(tr, lay, a, r, T0, color, band = true) {
  tr.push({ x: lags(r.length), y: r, type: 'bar', width: 0.55, marker: { color }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' });
  if (band) { const b = 1.96 / Math.sqrt(T0); lay.shapes.push(hl(a, b, K.red, 'dash', 1), hl(a, -b, K.red, 'dash', 1)); }
  lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.8));
}
const BAND = leg('±1.96/√T', { color: K.red, dash: 'dash', width: 1.2 });
const band = (x, lo, hi, a, col, al, name, o) => Object.assign({ x: x.concat([...x].reverse()), y: hi.concat([...lo].reverse()), fill: 'toself',
  fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, mode: 'lines', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' }, o || {});

// ================================================================= 1. four trending series
const GQ = D.gdp.q, GY = D.gdp.y;                       // 100 ln real GDP, 2000Q1-2026Q2 (slide vintage)
let FX = null;
const fxDays = () => (FX || (FX = { d: days(D.fx), v: D.fx.v }));
function blkFour() {
  const k = +$('fs-end').value, n = GY.length; set('fs-end', T.q(GQ[k]));
  const t = range(0, n - 1, 1), f = ols([ones(k + 1), t.slice(0, k + 1)], GY.slice(0, k + 1).map(v => v / 100)), line = t.map(i => f.b[0] + f.b[1] * i);
  const dev = GY.map((v, i) => v - 100 * line[i]), im = dev.reduce((b, v, i) => (Math.abs(v) > Math.abs(dev[b]) ? i : b), 0);
  const G = grid('fs-chart', [{ r: 0, c: 0, t: 'Romania: log real GDP (quarterly, SA)', x: { type: 'date' } }, { r: 0, c: 1, t: 'Romania: log HICP (monthly, 2015 = 100)', x: { type: 'date' } },
    { r: 1, c: 0, t: 'EUR/RON, BNR reference rate (daily)', x: { type: 'date' } }, { r: 1, c: 1, t: 'Log price minus its 2000 value', x: { type: 'date' } }], { cols: 2, h: 470, legendRows: 2 });
  const [a, b, c, d] = G.ax, x = GQ.map(q2d), F = { d: days(D.four.fx), v: D.four.fx.v }, B = days(D.four.bet), S = days(D.four.sp);
  const tr = [ln(x, GY.map(v => v / 100), a, K.green, 'Romania: log real GDP', { customdata: GQ, hovertemplate: '%{customdata}: %{y:.4f}<extra></extra>' }),
    ln(x, line, a, K.navy, 'fitted straight line', { line: { color: K.navy, dash: 'dash', width: 1 }, hoverinfo: 'skip' }),
    ln(D.four.hicp_m.map(m2d), D.four.hicp, b, K.red, 'Romania: log HICP', { hovertemplate: '%{x|%Y-%m}: %{y:.3f}<extra></extra>' }),
    ln(F.d, F.v, c, K.blue, 'EUR/RON', { line: { color: K.blue, width: 1 }, hovertemplate: '%{x}: %{y:.4f}<extra></extra>' }),
    ln(B, D.four.bet.v, d, K.red, 'BET', { line: { color: K.red, width: 0.9 }, hovertemplate: '%{x}: %{y:.3f}<extra>BET</extra>' }),
    ln(S, D.four.sp.v, d, K.blue, 'S&P 500', { line: { color: K.blue, width: 0.9 }, hovertemplate: '%{x}: %{y:.3f}<extra>S&P 500</extra>' })];
  if (k < n - 1) G.lay.shapes.push(vl(a, x[k], K.navy, 'dot', 1));
  plot('fs-chart', tr, G.lay);
  const N4 = NB.four;
  out('fs-out', [T.fs1(T.q(GQ[0]), T.q(GQ[k]), nf(100 * f.b[1], 2), nf(400 * f.b[1], 1)),
    T.fs2(T.q(GQ[n - 1]), nf(dev[n - 1], 1), nf(Math.abs(dev[im]), 1), T.q(GQ[im])),
    T.fs3(nf(N4.hicp_ratio, 1), nf(N4.fx_first, 2), nf(N4.fx_last, 2), nf(100 * N4.bet_logchg, 0), nf(100 * N4.sp_logchg, 0))]);
}

// ================================================================= 2. same drift, different memory
function tsdsPaths(phi, shock) {
  const e = D.tsds, n = e.length, e2 = e.slice(); e2[100] += shock;
  const P = eps => { const u = new Array(n).fill(0); for (let i = 1; i < n; i++) u[i] = phi * u[i - 1] + eps[i]; let c = 0;
    return [u.map((v, i) => 0.2 * i + v), eps.map((v, i) => 0.2 * i + (c += v))]; };
  return [P(e), P(e2)];
}
function blkTsDs() {
  const phi = +$('td-phi').value, sh = +$('td-sh').value; set('td-phi', nf(phi, 2)); set('td-sh', nf(sh, 1));
  const [[ts0, ds0], [ts1, ds1]] = tsdsPaths(phi, sh), t = range(0, 199, 1), h = range(0, 30, 1);
  const G = grid('td-chart', [{ r: 0, c: 0, t: `Trend-stationary, φ = ${phi.toFixed(2)}`, xt: 't' },
    { r: 0, c: 1, t: 'Difference-stationary (random walk)', xt: 't' },
    { r: 0, c: 2, t: 'Effect of a unit shock', xt: 'h (periods)', y: { range: [-0.05, 1.15] } }], { cols: 3, h: 360, legendRows: 2 });
  const [a, b, c] = G.ax, tl = t.map(v => 0.2 * v);
  const tr = [ln(t, ts0, a, K.blue, 'without the shock', { legendgroup: 'a' }), ln(t, ts1, a, K.red, `with a shock of +${sh} at t = 100`, { legendgroup: 'b' }),
    ln(t, tl, a, K.navy, 'trend line 0.2 t', { line: { color: K.navy, dash: 'dash', width: 0.9 }, legendgroup: 'c', hoverinfo: 'skip' }),
    ln(t, ds0, b, K.blue, '', { legendgroup: 'a' }), ln(t, ds1, b, K.red, '', { legendgroup: 'b' }),
    ln(t, tl, b, K.navy, '', { line: { color: K.navy, dash: 'dash', width: 0.9 }, hoverinfo: 'skip' }),
    ln(h, h.map(k => phi ** k), c, K.green, `trend-stationary: φ<sup>h</sup>, φ = ${phi.toFixed(2)}`, { mode: 'lines+markers', marker: { size: 4 } }),
    ln(h, h.map(() => 1), c, K.purple, 'random walk: 1', { mode: 'lines+markers', marker: { size: 4, symbol: 'square' } })];
  plot('td-chart', tr, G.lay);
  out('td-out', [T.td1(nf(phi, 2), nf(ts1[199] - ts0[199], 3), nf(ds1[199] - ds0[199], 2)), T.td2(nf(phi ** 10, 3), nf(phi ** 20, 3), phi < 1 ? nf(Math.log(0.5) / Math.log(phi), 1) : null)]);
}

// ================================================================= 3. detrending a random walk
function detrendStats(n) {
  const y = D.detrend.y.slice(0, n), t = range(0, n - 1, 1), f = ols([ones(n), t], y), r = acf(f.e, 60);
  const neg = r.findIndex(v => v < 0) + 1;
  return { y, t, f, r, neg };
}
function blkDetrend() {
  const n = +$('dt-T').value; set('dt-T', n);
  const { y, t, f, r, neg } = detrendStats(n), mc = D.detrend.mc[n];
  const G = grid('dt-chart', [{ r: 0, c: 0, t: 'Random walk and fitted line', xt: 't' }, { r: 0, c: 1, t: 'Residuals: apparent cycles', xt: 't' },
    { r: 0, c: 2, t: 'ACF of the residuals', xt: 'lag', y: { range: [-0.6, 1] } }], { cols: 3, h: 360 });
  const [a, b, c] = G.ax, tr = [ln(t, y, a, K.blue, 'random walk'), ln(t, t.map(v => f.b[0] + f.b[1] * v), a, K.red, 'fitted trend', { line: { color: K.red, dash: 'dash', width: 1.3 } }),
    ln(t, f.e, b, K.purple, 'detrended series')];
  G.lay.shapes.push(hl(b, 0, K.navy, 'solid', 0.6));
  acfPanel(tr, G.lay, c, r.slice(0, 30), n, K.purple); tr.push(BAND);
  plot('dt-chart', tr, G.lay);
  out('dt-out', [T.dt1(n, nf(f.r2, 2), nf(f.t[1], 1)), T.dt2(nf(r[0], 2), neg), T.dt3(nf(100 * mc[0], 1), nf(mc[1], 2))]);
}

// ================================================================= 4. spurious regression by simulation
const SPT = ['25', '50', '100', '200', '500', '1000'];
function histDens(x, lo, hi, w) {
  const nb = Math.round((hi - lo) / w), c = new Array(nb).fill(0); let m = 0;
  x.forEach(v => { if (v >= lo && v <= hi) { c[Math.min(nb - 1, Math.floor((v - lo) / w))]++; m++; } });
  return { x: c.map((_, i) => lo + (i + 0.5) * w), y: c.map(v => v / (m * w)) };
}
function blkSpurMC() {
  const Tn = $('sm-T').value, S = NB.spur_mc, Ts = SPT.map(Number), tl = D.spur.t[Tn].levels, td = D.spur.t[Tn].differences;
  const lim = Math.max(30, Math.ceil(Math.max(...tl.map(Math.abs)) / 10) * 10);
  const G = grid('sm-chart', [{ r: 0, c: 0, t: 'Rejection rate of a true null', xt: 'sample size T', yt: '% of samples with |t| > 1.96', x: { type: 'log', tickvals: [25, 50, 100, 200, 500, 1000] }, y: { range: [0, 100] } },
    { r: 0, c: 1, t: `t-statistic of the slope, T = ${Tn}`, xt: 't', x: { range: [-lim, lim] } }], { cols: 2, h: 370, legendRows: 2 });
  const [a, b] = G.ax, bw = lim / 60, hL = histDens(tl, -lim, lim, bw), hD = histDens(td, -lim, lim, bw), xx = range(-lim, lim, lim / 200);
  const tr = [ln(Ts, Ts.map(k => 100 * S[k].levels.reject), a, K.red, 'levels (independent random walks)', { mode: 'lines+markers', hovertemplate: 'T = %{x}: %{y:.1f}%<extra>levels</extra>' }),
    ln(Ts, Ts.map(k => 100 * S[k].differences.reject), a, K.blue, 'first differences', { mode: 'lines+markers', marker: { symbol: 'square' }, hovertemplate: 'T = %{x}: %{y:.1f}%<extra>differences</extra>' }),
    { x: hL.x, y: hL.y, type: 'bar', width: bw, marker: { color: 'rgba(205,0,0,0.55)' }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hovertemplate: 't ≈ %{x:.2f}: %{y:.3f}<extra>levels</extra>' },
    { x: hD.x, y: hD.y, type: 'bar', width: bw, marker: { color: 'rgba(26,58,110,0.65)' }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hovertemplate: 't ≈ %{x:.2f}: %{y:.3f}<extra>differences</extra>' },
    ln(xx, xx.map(v => Math.exp(-v * v / 2) / Math.sqrt(2 * Math.PI)), b, K.green, 'N(0, 1)', { line: { color: K.green, width: 1.2 }, hoverinfo: 'skip' }),
    leg('nominal size 5%', { color: K.navy, dash: 'dash', width: 1 })];
  G.lay.barmode = 'overlay';
  G.lay.shapes.push(hl(a, 5, K.navy, 'dash', 0.9), vl(a, +Tn, K.navy, 'dot', 1));
  plot('sm-chart', tr, G.lay);
  const L = S[Tn].levels, Df = S[Tn].differences;
  out('sm-out', [T.sm1(Tn, nf(100 * L.reject, 1), nf(L.abs_t_median, 1), nf(L.r2, 2), nf(L.dw, 3)), T.sm2(nf(100 * Df.reject, 1), nf(Df.abs_t_median, 2), nf(Df.r2, 3), nf(Df.dw, 2))]);
}

// ================================================================= 5. a spurious regression on real data
function spurReal(y0, y1) {
  const R = D.spur.real, ix = R.m.map((m, i) => (+m.slice(0, 4) >= y0 && +m.slice(0, 4) <= y1 ? i : -1)).filter(i => i >= 0);
  const h = ix.map(i => R.hicp[i]), s = ix.map(i => R.sp[i]), dh = h.slice(1).map((v, i) => v - h[i]), dsp = s.slice(1).map((v, i) => v - s[i]);
  const L = ols([ones(h.length), s], h), Df = ols([ones(dh.length), dsp], dh);
  return { ix, h, s, dh, dsp, L, Df, dwL: dw(L.e), dwD: dw(Df.e) };
}
function blkSpurReal() {
  let y0 = +$('sr-y0').value, y1 = +$('sr-y1').value; if (y1 < y0 + 3) { y1 = Math.min(2026, y0 + 3); $('sr-y1').value = y1; } set('sr-y0', y0); set('sr-y1', y1);
  const R = D.spur.real, S = spurReal(y0, y1), m = S.ix.map(i => R.m[i]);
  const G = grid('sr-chart', [{ r: 0, c: 0, t: `Levels: t = ${S.L.t[1].toFixed(1)}, R² = ${S.L.r2.toFixed(2)}, DW = ${S.dwL.toFixed(2)}`, xt: '100 ln S&P 500', yt: '100 ln HICP Romania' },
    { r: 0, c: 1, t: `Differences: t = ${S.Df.t[1].toFixed(2)}, R² = ${S.Df.r2.toFixed(3)}`, xt: 'monthly change of 100 ln S&P 500', yt: 'monthly inflation (%)' }], { cols: 2, h: 380 });
  const [a, b] = G.ax, ext = v => [Math.min(...v), Math.max(...v)], xa = ext(S.s), xb = ext(S.dsp);
  const tr = [{ x: S.s, y: S.h, mode: 'markers', marker: { color: K.blue, size: 5 }, name: 'monthly observations', xaxis: a.xa, yaxis: a.ya, customdata: m, hovertemplate: '%{customdata}: %{x:.1f}, %{y:.1f}<extra></extra>' },
    ln(xa, xa.map(v => S.L.b[0] + S.L.b[1] * v), a, K.red, 'OLS line', { hoverinfo: 'skip' }),
    { x: S.dsp, y: S.dh, mode: 'markers', marker: { color: K.blue, size: 5 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, customdata: m.slice(1), hovertemplate: '%{customdata}: %{x:.2f}, %{y:.2f}<extra></extra>' },
    ln(xb, xb.map(v => S.Df.b[0] + S.Df.b[1] * v), b, K.red, '', { hoverinfo: 'skip' })];
  plot('sr-chart', tr, G.lay);
  out('sr-out', [T.sr1(T.m(m[0]), T.m(m[m.length - 1]), m.length), T.sr2(nf(S.L.b[1], 2), nf(S.L.t[1], 1), nf(S.L.r2, 2), nf(S.dwL, 2), S.L.r2 > S.dwL),
    T.sr3(nf(S.Df.b[1], 4), nf(S.Df.t[1], 2), pe(tsf2(S.Df.t[1], S.dh.length - 2)), nf(S.Df.r2, 4), nf(S.dwD, 2))]);
}

// ================================================================= 6. Dickey-Fuller distributions
const DFC = { n: K.green, c: K.blue, ct: K.red }, DFN = { n: 'no constant', c: 'constant', ct: 'constant and trend' };
function dfShare(reg, c) {        // share of the 20 000 statistics below c (exact for multiples of 0.005)
  const S = D.dfdist, k = Math.round((c - S.lo) / S.w); let s = S[reg].below;
  for (let j = 0; j < Math.min(k, S[reg].cnt.length); j++) s += S[reg].cnt[j];
  return s / S.R;
}
function blkDfDist() {
  const c = +$('dd-c').value; set('dd-c', nf(c, 3));
  const S = D.dfdist, G = grid('dd-chart', [{ r: 0, c: 0, t: `Null distribution of the Dickey-Fuller statistic (random walk, T = ${S.T}); dashed: 5% critical values`, xt: 'τ = γ̂ / SE(γ̂)', x: { range: [-5.5, 3.5] } }], { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0], xx = S.x, tr = [ln(xx, xx.map(v => Math.exp(-v * v / 2) / Math.sqrt(2 * Math.PI)), a, K.navy, 'N(0, 1)', { line: { color: K.navy, dash: 'dot', width: 1.3 }, hovertemplate: '%{x:.2f}: %{y:.3f}<extra>N(0, 1)</extra>' })];
  ['n', 'c', 'ct'].forEach(reg => {
    const y = S[reg].kde, ix = xx.map((v, i) => (v <= c ? i : -1)).filter(i => i >= 0);
    if (ix.length) tr.push(band(ix.map(i => xx[i]), ix.map(() => 0), ix.map(i => y[i]), a, DFC[reg], 0.12, ''));
    tr.push(ln(xx, y, a, DFC[reg], 'DF, ' + DFN[reg], { line: { color: DFC[reg], width: 1.6 }, hovertemplate: '%{x:.2f}: %{y:.3f}<extra>' + DFN[reg] + '</extra>' }));
    G.lay.shapes.push(vl(a, NB.dfdist[reg].crit5, DFC[reg], 'dash', 0.9));
  });
  G.lay.shapes.push(vl(a, c, K.navy, 'solid', 1.6));
  tr.push(leg(`critical value used: ${c.toFixed(3)}`, { color: K.navy, width: 1.6 }));
  plot('dd-chart', tr, G.lay);
  const sh = ['n', 'c', 'ct'].map(reg => pct(dfShare(reg, c), 1));
  out('dd-out', [T.dd1(nf(c, 3), ...sh, pct(normCdf(c), 1)), T.dd2(...['n', 'c', 'ct'].map(reg => nf(NB.dfdist[reg].crit5, 2)))]);
}

// ================================================================= 7. size and power
const PHIS = [0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 1.0];
const powerAt = (Tn, reg, phi) => { const k = `${Tn}_${reg}_${phi === 1 ? '1.0' : phi}`; return NB.power[k] !== undefined ? NB.power[k] : D.power.power[k]; };
function blkPower() {
  const Tn = +$('ap-T').value, lect = [100, 250, 500].includes(Tn);
  const G = grid('ap-chart', [{ r: 0, c: 0, t: 'Constant only', xt: 'true φ', yt: '% rejections of the unit root', y: { range: [0, 105] } },
    { r: 0, c: 1, t: `T = ${Tn}: constant against constant and trend`, xt: 'true φ', y: { range: [0, 105] } }], { cols: 2, h: 370, legendRows: 2 });
  const [a, b] = G.ax, tr = [], col = { 50: K.purple, 100: K.red, 250: K.orange, 500: K.green, 1000: K.teal };
  [100, 250, 500].concat(lect ? [] : [Tn]).forEach(t => tr.push(ln(PHIS, PHIS.map(p => 100 * powerAt(t, 'c', p)), a, col[t], `T = ${t}` + ([100, 250, 500].includes(t) ? '' : ' (extra run)'),
    { mode: 'lines+markers', marker: { size: 6 }, line: { color: col[t], width: t === Tn ? 2.4 : 1.3, dash: [100, 250, 500].includes(t) ? 'solid' : 'dot' }, hovertemplate: `T = ${t}, φ = %{x}: %{y:.1f}%<extra></extra>` })));
  [['c', K.blue, 'circle', 'solid'], ['ct', K.purple, 'square', 'dash']].forEach(([reg, cl, sym, dash]) => tr.push(ln(PHIS, PHIS.map(p => 100 * powerAt(Tn, reg, p)), b, cl, `T = ${Tn}, ${DFN[reg]}`,
    { mode: 'lines+markers', marker: { size: 6, symbol: sym }, line: { color: cl, width: 1.5, dash }, hovertemplate: `${DFN[reg]}, φ = %{x}: %{y:.1f}%<extra></extra>` })));
  G.lay.shapes.push(hl(a, 5, K.navy, 'dot', 0.9), hl(b, 5, K.navy, 'dot', 0.9));
  plot('ap-chart', tr, G.lay);
  const p = (reg, phi) => pct(powerAt(Tn, reg, phi), 1), cr = D.power.crit[Tn];
  out('ap-out', [T.ap0(Tn, lect, nf(cr.c, 2), nf(cr.ct, 2)), T.ap1(p('c', 1), p('ct', 1)), T.ap2(p('c', 0.95), p('ct', 0.95), p('c', 0.9), p('ct', 0.9))]);
}

// ================================================================= 8. KPSS partial sums
function kpss(x) {                // statsmodels kpss(regression='c', nlags='auto'): statistic, lags, interpolated p-value
  const n = x.length, m = mean(x), e = x.map(v => v - m), dot = (i) => { let s = 0; for (let t = i; t < n; t++) s += e[t] * e[t - i]; return s; };
  const cov = Math.floor(Math.pow(n, 2 / 9)); let s0 = sum(e.map(v => v * v)) / n, s1 = 0;
  for (let i = 1; i <= cov; i++) { const p = dot(i) / (n / 2); s0 += p; s1 += i * p; }
  const sh = s1 / s0, L = Math.min(Math.floor(1.1447 * Math.pow(sh * sh, 1 / 3) * Math.pow(n, 1 / 3)), n - 1);
  let lr = sum(e.map(v => v * v)); for (let i = 1; i <= L; i++) lr += 2 * dot(i) * (1 - i / (L + 1)); lr /= n;
  let c = 0; const S = e.map(v => (c += v)), eta = sum(S.map(v => v * v)) / (n * n) / lr;
  const cv = [0.347, 0.463, 0.574, 0.739], pv = [0.10, 0.05, 0.025, 0.01];
  let p; if (eta <= cv[0]) p = 0.10; else if (eta >= cv[3]) p = 0.01; else { const j = cv.findIndex(v => v > eta); p = pv[j - 1] + (pv[j] - pv[j - 1]) * (eta - cv[j - 1]) / (cv[j] - cv[j - 1]); }
  return { eta, L, p, e, S };
}
const pk = p => (p >= 0.1 ? 'p ≥ ' + nf(0.1, 2) : p <= 0.01 ? 'p ≤ ' + nf(0.01, 2) : 'p = ' + nf(p, 3));
function kpssPaths(phi) {
  const e = D.kpss.e, n = e.length, ar = new Array(n).fill(0); for (let t = 1; t < n; t++) ar[t] = phi * ar[t - 1] + e[t];
  let c = 0; return [ar, D.kpss.w.map(v => (c += v))];
}
function blkKpss() {
  const phi = +$('kp-phi').value; set('kp-phi', nf(phi, 2));
  const [ar, rw] = kpssPaths(phi), A = kpss(ar), R = kpss(rw), t = range(1, ar.length, 1);
  const G = grid('kp-chart', [{ r: 0, c: 0, t: 'Demeaned series e<sub>t</sub> = y<sub>t</sub> − ȳ', xt: 't' }, { r: 0, c: 1, t: 'Partial sums S<sub>t</sub> = e<sub>1</sub> + … + e<sub>t</sub>', xt: 't' }], { cols: 2, h: 360 });
  const [a, b] = G.ax, lab = phi < 1 ? `AR(1), φ = ${phi.toFixed(2)}` : 'AR(1) with φ = 1 (a random walk)';
  const tr = [ln(t, A.e, a, K.blue, lab, { line: { color: K.blue, width: 0.9 } }), ln(t, R.e, a, K.red, 'random walk', { line: { color: K.red, width: 0.9 } }),
    ln(t, A.S, b, K.blue, '', { line: { color: K.blue, width: 1.3 } }), ln(t, R.S, b, K.red, '', { line: { color: K.red, width: 1.3 } })];
  G.lay.shapes.push(hl(b, 0, K.navy, 'solid', 0.6));
  plot('kp-chart', tr, G.lay);
  out('kp-out', [T.kp(nf(phi, 2), nf(A.eta, 3), A.L, pk(A.p), A.eta > 0.463), T.kpR(nf(R.eta, 3), R.L, pk(R.p))]);
}

// ================================================================= 9. a level shift misleads the ADF test
function zaCurve(y) {             // t-statistic of y_{t-1} for every break date b = 30..169: Δy on 1, DU, t, y_{t-1} (no lags)
  const n = y.length, dy = y.slice(1).map((v, i) => v - y[i]), yl = y.slice(0, n - 1), tt = range(1, n - 1, 1), bs = range(Math.floor(0.15 * n), Math.floor(0.85 * n) - 1, 1);
  return { bs, t: bs.map(b => ols([ones(n - 1), tt.map(v => (v >= b ? 1 : 0)), tt, yl], dy).t[3]) };
}
function blkBrkSim() {
  const dl = +$('bs-d').value; set('bs-d', nf(dl, 1));
  const B = D.brk_sim, g = B.grid[dl.toFixed(1)], step = B.u.map((_, t) => (t >= 100 ? dl : 0)), y = B.u.map((v, t) => v + step[t]), Z = zaCurve(y);
  const im = Z.t.indexOf(Math.min(...Z.t)), t = range(0, 199, 1);
  const G = grid('bs-chart', [{ r: 0, c: 0, t: `Stationary around a broken mean: ADF p = ${g.adf_p.toFixed(2)}`, xt: 't' }, { r: 0, c: 1, t: 'Zivot-Andrews: search over break dates', xt: 'candidate break date' }], { cols: 2, h: 370, legendRows: 2 });
  const [a, b] = G.ax, tr = [ln(t, y, a, K.blue, 'simulated series', { line: { color: K.blue, width: 0.9 } }),
    ln(t, step, a, K.red, `mean: 0, then ${dl} from t = 100`, { line: { color: K.red, dash: 'dash', width: 1.3 }, hoverinfo: 'skip' }),
    ln(Z.bs, Z.t, b, K.purple, 't-statistic for each break date', { hovertemplate: 'break at %{x}: t = %{y:.2f}<extra></extra>' }),
    leg(`ZA 5% critical value ${B.crit_za.toFixed(2)}`, { color: K.red, dash: 'dash', width: 1 })];
  G.lay.shapes.push(hl(b, B.crit_za, K.red, 'dash', 0.9), vl(b, Z.bs[im], K.navy, 'dot', 0.9));
  plot('bs-chart', tr, G.lay);
  out('bs-out', [T.bs1(nf(dl, 1), nf(g.adf, 2), g.adf_k, pe(g.adf_p), nf(B.crit_adf, 2), pct(g.rej, 1)),
    T.bs2(nf(g.za, 2), g.za_k, g.za_b + 1, nf(B.crit_za, 2), nf(Z.t[im], 2), Z.bs[im])]);
  return { g, min: Z.t[im], at: Z.bs[im] };
}

// ================================================================= 10. Zivot-Andrews on two real series
function blkBrkReal() {
  const reg = $('br-reg').value, BR = D.brk_real;
  const G = grid('br-chart', [{ r: 0, c: 0, t: `Nile flow: break in ${BR.nile.za[reg].brk.slice(0, 4)}, ZA = ${BR.nile.za[reg].stat.toFixed(2)}`, x: { type: 'date' } },
    { r: 0, c: 1, t: `100 ln EUR/RON (month-end): ZA = ${BR.eurron.za[reg].stat.toFixed(2)}`, x: { type: 'date' } }], { cols: 2, h: 370 });
  const tr = [];
  [['nile', K.blue], ['eurron', K.green]].forEach(([k, col], j) => {
    const S = BR[k], z = S.za[reg], a = G.ax[j], x = S.d.map(m2d), bx = m2d(z.brk);
    tr.push(ln(x, S.v, a, col, j ? '' : 'series', { hovertemplate: '%{x|%Y-%m}: %{y:.1f}<extra></extra>' }));
    tr.push(ln(x, x.map(v => (v < bx ? z.m0 : z.m1)), a, K.red, j ? '' : 'mean before and after the break', { line: { color: K.red, dash: 'dash', width: 1.3 }, hoverinfo: 'skip' }));
    G.lay.shapes.push(vl(a, bx, K.navy, 'dot', 1));
  });
  tr.push(leg('series (EUR/RON)', { color: K.green, width: 1.3 }));
  plot('br-chart', tr, G.lay);
  const N0 = BR.nile.za[reg], E = BR.eurron.za[reg];
  out('br-out', [T.br0(reg), T.brN(nf(N0.stat, 2), nf(N0.crit5, 2), pe(N0.p), N0.brk.slice(0, 4), nf(N0.m0, 0), nf(N0.m1, 0), nf(BR.nile.adf_p, 3), nf(BR.nile.kpss, 3)),
    T.brE(nf(E.stat, 2), nf(E.crit5, 2), pe(E.p), T.m(E.brk), nf(Math.exp(E.m0 / 100), 2), nf(Math.exp(E.m1 / 100), 2), nf(BR.eurron.adf_p, 2))]);
}

// ================================================================= 11. Romanian GDP: identification
function gdpId(k) {
  const y = GY.slice(0, k + 1), d = y.slice(1).map((v, i) => v - y[i]), n = d.length, r = acf(d, 12), p = pacf(r), lb = ljungBox(d, 8);
  const ord = d.map((v, i) => i).sort((u, v) => d[u] - d[v]);
  return { y, d, n, r, p, lb, m: mean(d), sd: sdev(d), w: ord.slice(0, 2), out: r.filter(v => Math.abs(v) > 1.96 / Math.sqrt(n)).length };
}
function blkGdpId() {
  const k = +$('gi-end').value; set('gi-end', T.q(GQ[k]));
  const S = gdpId(k), x = GQ.slice(0, k + 1).map(q2d);
  const G = grid('gi-chart', [{ r: 0, c: 0, t: '100 ln Y<sub>t</sub>', x: { type: 'date', range: [q2d(GQ[0]), q2d(GQ[GQ.length - 1])] }, y: { range: [Math.min(...GY) - 3, Math.max(...GY) + 3] } },
    { r: 0, c: 1, t: '100 Δln Y<sub>t</sub>: quarterly growth (%)', x: { type: 'date', range: [q2d(GQ[0]), q2d(GQ[GQ.length - 1])] }, y: { range: [-11.5, 6.5] } },
    { r: 1, c: 0, t: 'ACF of the growth rate', xt: 'lag (quarters)', y: { range: [-0.4, 0.4] } }, { r: 1, c: 1, t: 'PACF of the growth rate', xt: 'lag (quarters)', y: { range: [-0.4, 0.4] } }], { cols: 2, h: 480 });
  const [a, b, c, d] = G.ax, tr = [ln(x, S.y, a, K.green, '', { customdata: GQ, hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' }),
    ln(x.slice(1), S.d, b, K.blue, '', { customdata: GQ.slice(1), hovertemplate: '%{customdata}: %{y:.2f}%<extra></extra>' }),
    ln([x[1], x[x.length - 1]], [S.m, S.m], b, K.red, `mean ${S.m.toFixed(2)}%`, { line: { color: K.red, dash: 'dash', width: 0.9 }, hoverinfo: 'skip' })];
  acfPanel(tr, G.lay, c, S.r, S.n, K.blue); acfPanel(tr, G.lay, d, S.p, S.n, K.purple); tr.push(BAND);
  plot('gi-chart', tr, G.lay);
  out('gi-out', [T.gi1(T.q(GQ[0]), T.q(GQ[k]), S.n + 1, nf(S.m, 2), nf(4 * S.m, 1), nf(S.sd, 2)),
    T.gi2(nf(S.r[0], 2), nf(S.r[1], 2), nf(S.p[1], 2), nf(1.96 / Math.sqrt(S.n), 2), S.out), T.gi3(nf(S.lb.q, 1), pe(chi2sf(S.lb.q, 8))),
    T.gi4(nf(S.d[S.w[0]], 1), T.q(GQ[S.w[0] + 1]), nf(S.d[S.w[1]], 1), T.q(GQ[S.w[1] + 1]))]);
}

// ================================================================= 12. Romanian GDP: residual checks
function moments(e) { const n = e.length, m = mean(e), s2 = pvar(e), sk = sum(e.map(v => (v - m) ** 3)) / n / s2 ** 1.5, ku = sum(e.map(v => (v - m) ** 4)) / n / s2 ** 2;
  return { sk, ku, jb: n / 6 * (sk * sk + (ku - 3) ** 2 / 4) }; }
function blkGdpDiag() {
  const key = $('gd-mod').value, p = +key[0], q = +key[1], M = D.gdp.models[key], e = M.e, n = e.length, kk = p + q, x = GQ.slice(1).map(q2d);
  const ms = range(Math.max(kk + 1, 2), 16, 1), pv = ms.map(m => chi2sf(ljungBox(e, m).q, m - kk));
  const G = grid('gd-chart', [{ r: 0, c: 0, t: `Residuals of ARIMA(${p},1,${q}) with drift`, x: { type: 'date' } }, { r: 0, c: 1, t: 'ACF of the residuals', xt: 'lag', y: { range: [-0.4, 0.4] } },
    { r: 1, c: 0, t: 'Ljung-Box p-values', xt: 'm (lags tested)', y: { range: [0, 1] }, x: { range: [1, 16.5] } }, { r: 1, c: 1, t: 'Histogram of the residuals' }], { cols: 2, h: 480 });
  const [a, b, c, d] = G.ax, tr = [ln(x, e, a, K.blue, '', { customdata: GQ.slice(1), hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' })];
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.6));
  acfPanel(tr, G.lay, b, acf(e, 12), n, K.blue);
  tr.push({ x: ms, y: pv, mode: 'markers', marker: { color: K.purple, size: 8 }, name: 'Ljung-Box p-value', xaxis: c.xa, yaxis: c.ya, hovertemplate: 'm = %{x}: p = %{y:.3f}<extra></extra>' });
  G.lay.shapes.push(hl(c, 0.05, K.red, 'dash', 0.9));
  const lo = Math.min(...e), hi = Math.max(...e), w = (hi - lo) / 30, H = histDens(e, lo, hi, w), sd = Math.sqrt(sum(e.map(v => (v - mean(e)) ** 2)) / (n - 1)), xx = range(lo, hi, (hi - lo) / 200);
  tr.push({ x: H.x, y: H.y, type: 'bar', width: w, marker: { color: 'rgba(46,125,50,0.7)' }, name: 'residuals', xaxis: d.xa, yaxis: d.ya, hovertemplate: '%{x:.2f}: %{y:.3f}<extra></extra>' },
    ln(xx, xx.map(v => Math.exp(-(((v - mean(e)) / sd) ** 2) / 2) / (sd * Math.sqrt(2 * Math.PI))), d, K.red, 'Normal density', { hoverinfo: 'skip' }), BAND, leg('5%', { color: K.red, dash: 'dash', width: 1 }));
  plot('gd-chart', tr, G.lay);
  const mo = moments(e), q8 = ljungBox(e, 8).q, q12 = ljungBox(e, 12).q, imin = e.indexOf(lo);
  out('gd-out', [T.gd1(p, q, nf(M.aicc, 1), M.conv, nf(M.sigma, 2)), T.gd2(nf(q8, 1), 8 - kk, pe(chi2sf(q8, 8 - kk)), pe(chi2sf(q12, 12 - kk)), pv.every(v => v > 0.05)),
    T.gd3(nf(mo.ku, 1), nf(mo.jb, 0), nf(lo, 1), T.q(GQ[imin + 1]))]);
}

// ================================================================= 13. how fast the intervals grow
function psiW(phi, theta, d, n) {   // psi_j of phi(L)(1 - L)^d y = theta(L) e
  let a = [1].concat(phi.map(v => -v)); for (let k = 0; k < d; k++) a = a.concat([0]).map((v, i) => v - (i ? a[i - 1] : 0));
  const p = []; for (let j = 0; j < n; j++) { let v = j === 0 ? 1 : (j <= theta.length ? theta[j - 1] : 0); for (let i = 1; i < a.length && i <= j; i++) v -= a[i] * p[j - i]; p.push(v); }
  return p;
}
const halfW = psi => { let c = 0; return psi.map(v => 1.96 * Math.sqrt(c += v * v)); };
function blkWidth() {
  const f0 = +$('iw-f0').value, f1 = +$('iw-f1').value, th = +$('iw-th').value; set('iw-f0', nf(f0, 2)); set('iw-f1', nf(f1, 2)); set('iw-th', nf(th, 2));
  const M = [[`AR(1), φ = ${f0.toFixed(2)} (d = 0)`, [f0], [], 0, K.green], ['random walk, ARIMA(0,1,0)', [], [], 1, K.blue], [`ARIMA(1,1,0), φ = ${f1.toFixed(2)}`, [f1], [], 1, K.red],
    [`ARIMA(0,1,1), θ = ${th.toFixed(2)}`, [], [th], 1, K.orange], ['ARIMA(0,2,0)', [], [], 2, K.purple]];
  const h = range(1, 20, 1), W = M.map(m => halfW(psiW(m[1], m[2], m[3], 20)));
  const G = grid('iw-chart', [{ r: 0, c: 0, t: '95% forecast intervals: bounded for d = 0, growing for d = 1 and d = 2', xt: 'horizon h', yt: 'half-width / σ', y: { range: [0, 25] }, x: { dtick: 2 } }], { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0], tr = M.map((m, i) => ln(h, W[i], a, m[4], m[0], { mode: 'lines+markers', marker: { size: 5 }, hovertemplate: 'h = %{x}: %{y:.2f}σ<extra>' + m[0] + '</extra>' }));
  plot('iw-chart', tr, G.lay);
  const s = i => [nf(W[i][0], 2), nf(W[i][3], 2), nf(W[i][19], 2)];
  out('iw-out', [T.iw1(nf(f0, 2), ...s(0), Math.abs(f0) < 1 ? nf(1.96 / Math.sqrt(1 - f0 * f0), 2) : null), T.iw2(...s(1)), T.iw3(nf(f1, 2), ...s(2)), T.iw4(nf(th, 2), ...s(3)), T.iw5(...s(4))]);
  return W;
}

// ================================================================= 14. Romanian GDP: two forecasts
const bn = v => Math.exp(v / 100) / 1000;
function blkGdpFc() {
  const O = keysSorted(D.gdp_fc), k = +$('gf-o').value, o = O[k], F = D.gdp_fc[o], io = GQ.indexOf(o), H = F.ds.f.length; set('gf-o', T.q(o));
  const fq = range(1, H, 1).map(h => addQ(o, h)), i0 = GQ.indexOf('2012Q1'), z80 = normPpf(0.9), z95 = normPpf(0.975);
  const G = grid('gf-chart', [{ r: 0, c: 0, t: `Romania: real GDP and forecasts for ${H} quarters from ${o}, 80% and 95% intervals`, x: { type: 'date' } }], { cols: 1, h: 400, legendRows: 2 });
  const a = G.ax[0], tr = [ln(GQ.slice(i0, io + 1).map(q2d), GY.slice(i0, io + 1).map(bn), a, K.navy, 'real GDP (bn EUR, 2010 prices)', { line: { color: K.navy, width: 1.5 }, customdata: GQ.slice(i0, io + 1), hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' })];
  if (io < GQ.length - 1) tr.push(ln(GQ.slice(io, GQ.length).map(q2d), GY.slice(io).map(bn), a, K.navy, 'observed after the origin', { line: { color: K.navy, width: 1.2, dash: 'dot' }, customdata: GQ.slice(io), hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' }));
  const res = [];
  [['ds', 'ARIMA(0,1,0) with drift', K.red], ['ts', 'linear trend + AR(1) errors', K.blue]].forEach(([key, lab, col]) => {
    const m = F[key], x = fq.map(q2d);
    tr.push(band(x, m.f.map((v, h) => bn(v - z95 * m.se[h])), m.f.map((v, h) => bn(v + z95 * m.se[h])), a, col, 0.15, ''),
      band(x, m.f.map((v, h) => bn(v - z80 * m.se[h])), m.f.map((v, h) => bn(v + z80 * m.se[h])), a, col, 0.25, ''),
      ln(x, m.f.map(bn), a, col, lab, { line: { color: col, width: 1.6 }, customdata: fq, hovertemplate: '%{customdata}: %{y:.2f}<extra>' + lab + '</extra>' }));
    const obs = fq.map((q, h) => { const j = GQ.indexOf(q); return j < 0 ? null : GY[j] - m.f[h]; }).filter(v => v !== null);
    res.push(T.gf(key, nf(bn(m.f[H - 1]), 1), nf(bn(m.f[H - 1] - z95 * m.se[H - 1]), 1), nf(bn(m.f[H - 1] + z95 * m.se[H - 1]), 1),
      nf(z95 * m.se[0], 1), nf(z95 * m.se[3], 1), nf(z95 * m.se[H - 1], 1), obs.length ? nf(Math.sqrt(mean(obs.map(v => v * v))), 1) : null, obs.length));
  });
  tr.push(leg('shaded: 80% and 95% intervals', null, { color: 'rgba(31,42,68,0.25)', symbol: 'square', size: 12 }));
  plot('gf-chart', tr, G.lay);
  out('gf-out', [T.gf0(T.q(o), nf(bn(GY[io]), 1))].concat(res));
}

// ================================================================= 15. over-differencing
function dif(x, d) { let y = x; for (let k = 0; k < d; k++) y = y.slice(1).map((v, i) => v - y[i]); return y; }
const svar = x => sdev(x) ** 2;
function blkOverdiff() {
  const sr = $('od-s').value, d = +$('od-d').value, y = sr === 'ro' ? GY : D.overdiff.us, x1 = dif(y, 1), x2 = dif(y, d), th = D.overdiff.theta[sr][d];
  const lab = sr === 'ro' ? 'Romania' : 'US', r1 = acf(x1, 12), r2 = acf(x2, 12), lo = Math.min(-0.65, ...r2.map(v => v - 0.05));
  const G = grid('od-chart', [{ r: 0, c: 0, t: `${lab}: Δ 100 ln Y<sub>t</sub> (d = 1)`, xt: 'lag (quarters)', y: { range: [lo, 0.35] } },
    { r: 0, c: 1, t: `${lab}: Δ<sup>${d}</sup> 100 ln Y<sub>t</sub> (d = ${d}, over-differenced)`, xt: 'lag (quarters)', y: { range: [lo, 0.35] } }], { cols: 2, h: 340 });
  const tr = []; acfPanel(tr, G.lay, G.ax[0], r1, x1.length, K.blue); acfPanel(tr, G.lay, G.ax[1], r2, x2.length, K.red);
  tr.push(leg('sample ACF', null, { color: K.blue, symbol: 'square', size: 10 }), BAND);
  plot('od-chart', tr, G.lay);
  out('od-out', [T.od1(sr, nf(r1[0], 2), nf(svar(x1), 2), d, nf(r2[0], 2), nf(svar(x2), 2)), T.od2(d, nf(th[0], 3), nf(th[1], 3))]);
}

// ================================================================= 16. inflation: d = 0 or d = 1
function blkInfl() {
  const I = D.infl, k0 = '0' + $('in-m0').value, k1 = '1' + $('in-m1').value, M0 = I.fits[k0], M1 = I.fits[k1], H = M0.f.length, last = I.m[I.m.length - 1];
  const fm = range(1, H, 1).map(h => addM(last, h)), z = normPpf(0.975), i0 = I.m.indexOf('2018-01'), x = fm.map(m2d);
  const G = grid('in-chart', [{ r: 0, c: 0, t: `Romania: inflation forecasts for ${H} months, 95% intervals`, x: { type: 'date' } }], { cols: 1, h: 400, legendRows: 2 });
  const a = G.ax[0], tr = [ln(I.m.slice(i0).map(m2d), I.y.slice(i0), a, K.navy, '12-month HICP inflation (%)', { line: { color: K.navy, width: 1.4 }, hovertemplate: '%{x|%Y-%m}: %{y:.2f}%<extra></extra>' })];
  const name = kk => `ARIMA(${kk[1]},${kk[0]},${kk[2]})` + (kk[0] === '0' ? ' with mean' : ''), nameT = kk => `ARIMA(${kk[1]},${kk[0]},${kk[2]})` + (kk[0] === '0' ? T.wm : '');
  [[k0, M0, K.blue], [k1, M1, K.red]].forEach(([kk, M, col]) => tr.push(band(x, M.f.map((v, h) => v - z * M.se[h]), M.f.map((v, h) => v + z * M.se[h]), a, col, 0.15, ''),
    ln(x, M.f, a, col, name(kk), { line: { color: col, width: 1.6 }, hovertemplate: '%{x|%Y-%m}: %{y:.2f}%<extra>' + name(kk) + '</extra>' })));
  G.lay.shapes.push(hl(a, M0.mu, K.blue, 'dot', 0.9));
  tr.push(leg('estimated mean (d = 0)', { color: K.blue, dash: 'dot', width: 1 }));
  plot('in-chart', tr, G.lay);
  const best = d => { const ks = keysSorted(I.fits).filter(kk => kk[0] === d && I.fits[kk].conv); return ks.reduce((b, kk) => (I.fits[kk].aicc < I.fits[b].aicc ? kk : b), ks[0]); };
  const line = (kk, M) => T.inm(nameT(kk), nf(M.aicc, 1), M.conv, nf(M.f[11], 1), nf(M.f[H - 1], 1), nf(M.f[H - 1] - z * M.se[H - 1], 1), nf(M.f[H - 1] + z * M.se[H - 1], 1), T.m(fm[H - 1]));
  out('in-out', [line(k0, M0) + (M0.mu !== null ? T.inmu(nf(M0.mu, 2)) : ''), line(k1, M1), T.inb(nameT(best('0')), nameT(best('1')))]);
}

// ================================================================= 17. EUR/RON: ARIMA against the random walk
function fxCompare(split) {
  const F = fxDays(), y = F.v, n = y.length, d = y.slice(1).map((v, i) => v - y[i]), dd_ = F.d.slice(1);
  const s = dd_.findIndex(v => v >= split), tr = d.slice(0, s), mu = mean(tr), f = ols([ones(s - 1), tr.slice(0, s - 1)], tr.slice(1)), c = f.b[0], phi = f.b[1];
  const test = d.slice(s), prev = d.slice(s - 1, d.length - 1), eR = test, eD = test.map(v => v - mu), eA = test.map((v, i) => v - (c + phi * prev[i]));
  const rm = e => Math.sqrt(mean(e.map(v => v * v)));
  const p0 = F.d.findIndex(v => v >= split), e20 = { rw: [], drift: [], ar: [] };
  for (let i = 0; i < n - p0 - 20; i += 20) {
    const pos = p0 + i, base = y[pos], tg = y[pos + 20]; let dd2 = d[pos - 1], path = 0;
    for (let k = 0; k < 20; k++) { dd2 = c + phi * dd2; path += dd2; }
    e20.rw.push(tg - base); e20.drift.push(tg - base - 20 * mu); e20.ar.push(tg - base - path);
  }
  let a1 = 0, a2 = 0; const cum1 = [], cum2 = []; test.forEach((_, i) => { a1 += eD[i] ** 2 - eR[i] ** 2; a2 += eA[i] ** 2 - eR[i] ** 2; cum1.push(a1); cum2.push(a2); });
  return { F, s, p0, mu, c, phi, nTr: tr.length, nTe: test.length, r1: { rw: rm(eR), drift: rm(eD), ar: rm(eA) }, r20: { rw: rm(e20.rw), drift: rm(e20.drift), ar: rm(e20.ar) }, n20: e20.rw.length,
    dates: dd_.slice(s), cum1, cum2 };
}
function blkFx() {
  const yr = +$('fx-y').value; set('fx-y', yr);
  const S = fxCompare(`${yr}-01-01`), F = S.F, v0 = F.d.findIndex(v => v >= `${yr - 2}-01-01`);
  const G = grid('fx-chart', [{ r: 0, c: 0, t: 'EUR/RON, BNR reference rate', x: { type: 'date' } }, { r: 0, c: 1, t: 'Cumulative difference of squared 1-day errors', x: { type: 'date' } }], { cols: 2, h: 370, legendRows: 2 });
  const [a, b] = G.ax, tr = [ln(F.d.slice(v0), F.v.slice(v0).map(v => Math.exp(v / 100)), a, K.green, 'EUR/RON', { line: { color: K.green, width: 1 }, hovertemplate: '%{x}: %{y:.4f}<extra></extra>' }),
    leg('start of the test period', { color: K.navy, dash: 'dot', width: 1 }),
    ln(S.dates, S.cum1, b, K.purple, 'random walk with drift minus random walk', { hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }),
    ln(S.dates, S.cum2, b, K.red, 'ARIMA(1,1,0) minus random walk', { hovertemplate: '%{x}: %{y:.3f}<extra></extra>' })];
  G.lay.shapes.push(vl(a, F.d[S.p0], K.navy, 'dot', 1), hl(b, 0, K.navy, 'solid', 0.6));
  plot('fx-chart', tr, G.lay);
  const g1 = 100 * (1 - S.r1.ar / S.r1.rw), g20 = 100 * (1 - S.r20.ar / S.r20.rw);
  out('fx-out', [T.fx0(S.nTr, T.dmy(F.d[S.p0 - 1]), S.nTe, nf(S.phi, 3)), T.fx1(nf(S.r1.rw, 4), nf(S.r1.drift, 4), nf(S.r1.ar, 4), nf(g1, 1)),
    T.fx2(S.n20, nf(S.r20.rw, 3), nf(S.r20.drift, 3), nf(S.r20.ar, 3), nf(g20, 1))]);
}

// ================================================================= 18. US real GDP after 2007
function blkUs() {
  const yr = +$('us-y').value, o = `${yr}Q4`; set('us-y', o.replace(/(\d{4})Q4/, (m, y) => T.q(`${y}Q4`)));
  const U = D.us, F = U.origins[o], io = U.q.indexOf(o), H = F.ts.f.length, fq = U.q.slice(io + 1, io + 1 + H), x = fq.map(q2d), v0 = U.q.indexOf(`${Math.min(1995, yr - 12)}Q1`), z = normPpf(0.975);
  const G = grid('us-chart', [{ r: 0, c: 0, t: `US real GDP: two forecasts made in ${o}`, x: { type: 'date' } }, { r: 0, c: 1, t: 'Outcome minus forecast (log points)', x: { type: 'date' } }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax, tr = [ln(U.q.slice(v0).map(q2d), U.y.slice(v0), a, K.navy, 'US real GDP, 100 ln', { line: { color: K.navy, width: 1.4 }, customdata: U.q.slice(v0), hovertemplate: '%{customdata}: %{y:.1f}<extra></extra>' })];
  const res = [];
  [['ts', K.blue, `trend-stationary forecast from ${o}`], ['ds', K.red, `ARIMA(1,1,0) with drift from ${o}`]].forEach(([k, col, lab]) => {
    const M = F[k], gap = fq.map((q, h) => U.y[io + 1 + h] - M.f[h]);
    tr.push(band(x, M.f.map((v, h) => v - z * M.se[h]), M.f.map((v, h) => v + z * M.se[h]), a, col, 0.12, ''), ln(x, M.f, a, col, lab, { line: { color: col, width: 1.3 }, customdata: fq, hovertemplate: '%{customdata}: %{y:.1f}<extra></extra>' }),
      ln(x, gap, b, col, '', { customdata: fq, hovertemplate: '%{customdata}: %{y:.1f}<extra></extra>' }));
    const he = z * M.se[H - 1];
    res.push(T.us(k, nf(gap[5], 1), nf(gap[H - 1], 1), nf(he, 1), Math.abs(gap[H - 1]) <= he));
  });
  G.lay.shapes.push(hl(b, 0, K.navy, 'solid', 0.6));
  plot('us-chart', tr, G.lay);
  out('us-out', [T.us0(T.q(o), T.q(fq[5]), T.q(fq[H - 1]))].concat(res));
}

// ================================================================= wiring
const ALL = [blkFour, blkTsDs, blkDetrend, blkSpurMC, blkSpurReal, blkDfDist, blkPower, blkKpss, blkBrkSim, blkBrkReal, blkGdpId, blkGdpDiag, blkWidth, blkGdpFc,
  blkOverdiff, blkInfl, blkFx, blkUs];
function init() {
  const has = id => !!$(id), on = (ids, f) => ids.forEach(id => $(id).addEventListener('input', f));
  const players = {}, P = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  const nq = GY.length - 1;
  $('fs-end').max = nq; $('fs-end').value = nq; on(['fs-end'], blkFour); P('fs-play', 'fs-end', range(19, nq, 2), 250);
  on(['td-phi', 'td-sh'], blkTsDs); P('td-play', 'td-phi', range(0, 1, 0.05), 350);
  on(['dt-T'], blkDetrend); P('dt-play', 'dt-T', range(50, 200, 10), 450);
  on(['sm-T'], blkSpurMC); P('sm-play', 'sm-T', SPT, 1000);
  on(['sr-y0', 'sr-y1'], blkSpurReal); P('sr-play', 'sr-y0', range(2005, 2022, 1), 500);
  on(['dd-c'], blkDfDist); P('dd-play', 'dd-c', range(-3.5, -1.5, 0.05), 200);
  on(['ap-T'], blkPower); P('ap-play', 'ap-T', ['50', '100', '250', '500', '1000'], 1000);
  on(['kp-phi'], blkKpss); P('kp-play', 'kp-phi', range(0, 1, 0.05), 400);
  on(['bs-d'], blkBrkSim); P('bs-play', 'bs-d', range(0, 10, 0.5), 450);
  on(['br-reg'], blkBrkReal); P('br-play', 'br-reg', ['c', 't', 'ct'], 1300);
  $('gi-end').max = nq; $('gi-end').value = nq; on(['gi-end'], blkGdpId); P('gi-play', 'gi-end', range(39, nq, 1), 180);
  on(['gd-mod'], blkGdpDiag); P('gd-play', 'gd-mod', keysSorted(D.gdp.models), 1000);
  on(['iw-f0', 'iw-f1', 'iw-th'], blkWidth); P('iw-play', 'iw-th', range(-0.95, 0.95, 0.05), 250);
  const nO = Object.keys(D.gdp_fc).length; $('gf-o').max = nO - 1; $('gf-o').value = nO - 1; on(['gf-o'], blkGdpFc); P('gf-play', 'gf-o', range(0, nO - 1, 1), 350);
  on(['od-s', 'od-d'], blkOverdiff); P('od-play', 'od-d', ['2', '3'], 1300);
  on(['in-m0', 'in-m1'], blkInfl); P('in-play', 'in-m1', ['00', '01', '02', '10', '11', '12', '20', '21', '22', '30', '31', '32'], 900);
  on(['fx-y'], blkFx); P('fx-play', 'fx-y', range(2010, 2025, 1), 700);
  on(['us-y'], blkUs); P('us-play', 'us-y', range(1990, 2019, 1), 450);
  const PFX = { 'four-series': 'fs', 'ts-ds': 'td', detrend: 'dt', 'spurious-mc': 'sm', 'spurious-real': 'sr', 'df-dist': 'dd', 'adf-power': 'ap', 'kpss-sums': 'kp',
    'breaks-sim': 'bs', 'breaks-real': 'br', 'gdp-ident': 'gi', 'gdp-diag': 'gd', 'interval-width': 'iw', 'gdp-forecast': 'gf', overdiff: 'od', inflation: 'in', eurron: 'fx', 'us-gdp': 'us' };
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if (has(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: { out: true } }, sp)); });
  if (T.ticket && exitTicket && document.querySelector('main.imain')) exitTicket(T.ticket);
  ALL.forEach(f => f());
  let w = window.innerWidth, tm = null;
  window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(() => { if ((window.innerWidth < 720) !== (w < 720) || Math.abs(window.innerWidth - w) > 200) { w = window.innerWidth; ALL.forEach(f => f()); } }, 250); });
}
window.CH3 = { chi2sf, tsf2, ols, dw, tsdsPaths, detrendStats, spurReal, dfShare, powerAt, kpss, kpssPaths, zaCurve, gdpId, moments, psiW, halfW, fxCompare, dif, svar, bn };
if (!window.CH3_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
