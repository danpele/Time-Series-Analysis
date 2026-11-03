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
function dateLine(d, v, cut) {   // OLS line of v on calendar time, fitted on the dates before cut; two end points (a line on a date axis)
  const tm = d.map(s => Date.parse(s.length > 7 ? s : s + '-01')), ix = tm.map((t, i) => (t < cut ? i : -1)).filter(i => i >= 0);
  if (ix.length < 8) return { x: [], y: [] };
  const tx = ix.map(i => tm[i]), mx = mean(tx), my = mean(ix.map(i => v[i])), b = sum(ix.map((i, j) => (tx[j] - mx) * (v[i] - my))) / sum(tx.map(t => (t - mx) ** 2));
  return { x: [d[0], d[d.length - 1]], y: [my + b * (tm[0] - mx), my + b * (tm[tm.length - 1] - mx)] };
}
function blkFour() {
  const k = +$('fs-end').value, n = GY.length; set('fs-end', T.q(GQ[k]));
  const t = range(0, n - 1, 1), f = ols([ones(k + 1), t.slice(0, k + 1)], GY.slice(0, k + 1).map(v => v / 100)), line = t.map(i => f.b[0] + f.b[1] * i);
  const dev = GY.map((v, i) => v - 100 * line[i]), im = dev.reduce((b, v, i) => (Math.abs(v) > Math.abs(dev[b]) ? i : b), 0);
  const G = grid('fs-chart', [{ r: 0, c: 0, t: 'Romania: log real GDP (quarterly, SA)', x: { type: 'date' } }, { r: 0, c: 1, t: 'Romania: log HICP (monthly, 2015 = 100)', x: { type: 'date' } },
    { r: 1, c: 0, t: 'EUR/RON, BNR reference rate (daily)', x: { type: 'date' } }, { r: 1, c: 1, t: 'Log price minus its 2000 value', x: { type: 'date' } }], { cols: 2, h: 470, legendRows: 2 });
  const [a, b, c, d] = G.ax, x = GQ.map(q2d), F = { d: days(D.four.fx), v: D.four.fx.v }, B = days(D.four.bet), S = days(D.four.sp);
  const tr = [ln(x, GY.map(v => v / 100), a, K.green, 'Romania: log real GDP', { customdata: GQ, hovertemplate: '%{customdata}: %{y:.4f}<extra></extra>' }),
    ln(x, line, a, K.navy, 'straight line fitted up to the dotted date', { line: { color: K.navy, dash: 'dash', width: 1 }, hoverinfo: 'skip' }),
    ln(D.four.hicp_m.map(m2d), D.four.hicp, b, K.red, 'Romania: log HICP', { hovertemplate: '%{x|%Y-%m}: %{y:.3f}<extra></extra>' }),
    ln(F.d, F.v, c, K.blue, 'EUR/RON', { line: { color: K.blue, width: 1 }, hovertemplate: '%{x}: %{y:.4f}<extra></extra>' }),
    ln(B, D.four.bet.v, d, K.red, 'BET', { line: { color: K.red, width: 0.9 }, hovertemplate: '%{x}: %{y:.3f}<extra>BET</extra>' }),
    ln(S, D.four.sp.v, d, K.blue, 'S&P 500', { line: { color: K.blue, width: 0.9 }, hovertemplate: '%{x}: %{y:.3f}<extra>S&P 500</extra>' })];
  const cut = Date.parse(q2d(addQ(GQ[k], 1))), dash = { color: K.navy, dash: 'dash', width: 1 };
  [[D.four.hicp_m.map(m2d), D.four.hicp, b], [F.d, F.v, c], [B, D.four.bet.v, d], [S, D.four.sp.v, d]].forEach(([dd, v, ax]) => {
    const L = dateLine(dd, v, cut); tr.push(ln(L.x, L.y, ax, K.navy, '', { line: dash, hoverinfo: 'skip' })); });
  if (k < n - 1) G.ax.forEach(ax => G.lay.shapes.push(vl(ax, x[k], K.navy, 'dot', 1)));
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
    ln(t, ts1, b, '#9AA0A6', `middle panel: trend-stationary path with the shock, φ = ${phi.toFixed(2)}`, { line: { color: '#9AA0A6', width: 2.2 }, hovertemplate: 't = %{x}: %{y:.2f}<extra>trend-stationary</extra>' }),
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
    leg('nominal size 5%', { color: K.navy, dash: 'dash', width: 1 }),
    { x: [+Tn, +Tn], y: [100 * S[Tn].levels.reject, 100 * S[Tn].differences.reject], mode: 'markers', marker: { color: [K.red, K.blue], size: 13, symbol: 'circle-open', line: { width: 2.5 } },
      name: `T = ${Tn} (histogram on the right)`, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'T = %{x}: %{y:.1f}%<extra></extra>' }];
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
    const lo = Math.min(...S.v), hi = Math.max(...S.v);
    tr.push(ln([bx, bx], [lo, hi], a, K.navy, j ? '' : 'estimated break date', { line: { color: K.navy, dash: 'dot', width: 1.2 }, hovertemplate: `break: ${z.brk}<extra></extra>` }));
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
const ODK = [['ro', '2'], ['ro', '3'], ['us', '2'], ['us', '3']];
function blkOverdiff(ev) {
  if (ev && ev.target && ev.target.id === 'od-k') { const [s0, d0] = ODK[+ev.target.value]; $('od-s').value = s0; $('od-d').value = d0; }
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

// ================================================================= tables and worked examples (blocks of their own frames)
const X = () => T.x;
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}${v.cs ? ` colspan="${v.cs}"` : ''}>${v.v}</${tag}>`; };
  const hd = (Array.isArray(head[0]) ? head : [head]).map(r => '<tr>' + r.map(c => cell(c, 'th')).join('') + '</tr>').join('');
  el.innerHTML = (o && o.cap ? `<p class="c3cap">${o.cap}</p>` : '') + '<table><thead>' + hd + '</thead><tbody>'
    + rows.map(r => `<tr${r.hi ? ' class="hi"' : ''}>` + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c3cap">${o.note}</p>` : '');
}
const nz = (v, d = 2) => nf(Math.abs(v) < 0.5 * 10 ** -d ? 0 : v, d);                // no '−0.00'
const num = (x, d, bold) => ({ v: x === null || x === undefined || !isFinite(x) ? '—' : bold ? `<b>${nz(x, d)}</b>` : nz(x, d), cls: 'r' });
const txt = (v, bold) => ({ v: bold ? `<b>${v}</b>` : v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const hiRow = (r, on) => { if (on) r.hi = true; return r; };
const pn = (x, d) => (x < 0 ? `(${nz(x, d)})` : nz(x, d));                       // a factor: negative numbers in parentheses
const fp = p => (p < 0.001 ? '< ' + nf(0.001, 3) : nf(p, 3));

// MacKinnon (1994) p-values and MacKinnon (2010) critical values of the (A)DF statistic, one regressor (statsmodels adfvalues)
const MK = { n: { max: 1e9, min: -19.04, star: -1.04, sp: [0.6344, 1.2378, 0.032496], lp: [0.4797, 0.93557, -0.06999, 0.033066],
  cv: [[-2.56574, -2.2358, -3.627, 0], [-1.941, -0.2686, -3.365, 31.223], [-1.61682, 0.2656, -2.714, 25.364]] },
c: { max: 2.74, min: -18.83, star: -1.61, sp: [2.1659, 1.4412, 0.038269], lp: [1.7339, 0.93202, -0.12745, -0.010368],
  cv: [[-3.43035, -6.5393, -16.786, -79.433], [-2.86154, -2.8903, -4.234, -40.04], [-2.56677, -1.5384, -2.809, 0]] },
ct: { max: 0.7, min: -16.18, star: -2.89, sp: [3.2512, 1.6047, 0.049588], lp: [2.5261, 0.61654, -0.37956, -0.060285],
  cv: [[-3.95877, -9.0531, -28.428, -134.155], [-3.41049, -4.3904, -9.036, -45.374], [-3.12705, -2.5856, -3.925, -22.38]] } };
function mkp(t, reg) { const m = MK[reg]; if (t > m.max) return 1; if (t < m.min) return 0; const c = t <= m.star ? m.sp : m.lp; return normCdf(c.reduce((s, v, i) => s + v * t ** i, 0)); }
const mkc = (reg, n) => MK[reg].cv.map(r => (n === Infinity ? r[0] : r[0] + r[1] / n + r[2] / n ** 2 + r[3] / n ** 3));   // [1%, 5%, 10%]
const LV = { '0.01': 0, '0.05': 1, '0.1': 2 };
const KC = { c: { '0.1': 0.347, '0.05': 0.463, '0.01': 0.739 }, ct: { '0.1': 0.119, '0.05': 0.146, '0.01': 0.216 } };   // KPSS critical values

// ADF regression with k lagged differences on the last N observations (statsmodels adfuller: same sample for every k
// when choosing k, then the chosen k on all the observations it allows)
function adfFit(x, reg, k, Nfix) {
  const dx = dif(x, 1), N = Nfix || dx.length - k, off = dx.length - N, rows = range(off, dx.length - 1, 1), cols = [];
  if (reg !== 'n') cols.push(ones(N)); if (reg === 'ct') cols.push(range(1, N, 1));
  cols.push(rows.map(i => x[i]));
  for (let j = 1; j <= k; j++) cols.push(rows.map(i => dx[i - j]));
  const o = ols(cols, dx.slice(off)), ig = reg === 'n' ? 0 : reg === 'c' ? 1 : 2, m = cols.length, ssr = o.s2 * (N - m);
  const llf = -N / 2 * (Math.log(2 * Math.PI) + Math.log(ssr / N) + 1);
  return { o, ig, k, tau: o.t[ig], gam: o.b[ig], se: o.se[ig], N, m, llf, aic: -2 * llf + 2 * m, bic: -2 * llf + Math.log(N) * m };
}
const kmaxOf = (n, reg) => Math.min(Math.floor(n / 2) - (reg === 'n' ? 0 : reg.length) - 1, Math.ceil(12 * (n / 100) ** 0.25));
function adfAuto(x, reg, ic) {
  const kmax = kmaxOf(x.length, reg), N = x.length - 1 - kmax, sel = range(0, kmax, 1).map(k => adfFit(x, reg, k, N));
  let best = 0; sel.forEach((s, k) => { if (s[ic] < sel[best][ic]) best = k; });
  const fit = adfFit(x, reg, best); return { kmax, Nsel: N, sel, k: best, fit, p: mkp(fit.tau, reg), cv: mkc(reg, fit.N) };
}
// KPSS around a level (reg 'c') or a trend ('ct'), bandwidth chosen as in statsmodels (nlags='auto') unless L is given
function kpssG(x, reg, Lfix) {
  const n = x.length, e = reg === 'c' ? x.map(v => v - mean(x)) : ols([ones(n), range(1, n, 1)], x).e;
  const dot = i => { let s = 0; for (let t = i; t < n; t++) s += e[t] * e[t - i]; return s; };
  let L = Lfix;
  if (L === null || L === undefined) {
    const cov = Math.floor(Math.pow(n, 2 / 9)); let s0 = dot(0) / n, s1 = 0;
    for (let i = 1; i <= cov; i++) { const p = dot(i) / (n / 2); s0 += p; s1 += i * p; }
    L = Math.min(Math.floor(1.1447 * Math.pow((s1 / s0) ** 2, 1 / 3) * Math.pow(n, 1 / 3)), n - 1);
  }
  const g = range(0, L, 1).map(j => dot(j) / n), lr = g[0] + 2 * sum(g.slice(1).map((v, i) => v * (1 - (i + 1) / (L + 1))));
  let c = 0; const S = e.map(v => (c += v)), ss = sum(S.map(v => v * v)), eta = ss / (n * n) / lr;
  const cv = reg === 'c' ? [0.347, 0.463, 0.574, 0.739] : [0.119, 0.146, 0.176, 0.216], pv = [0.10, 0.05, 0.025, 0.01];
  let p; if (eta <= cv[0]) p = 0.10; else if (eta >= cv[3]) p = 0.01; else { const j = cv.findIndex(v => v > eta); p = pv[j - 1] + (pv[j] - pv[j - 1]) * (eta - cv[j - 1]) / (cv[j] - cv[j - 1]); }
  return { eta, L, p, e, S, ss, g, lr, n, crit5: cv[1] };
}
// series of the slides that the browser tests itself (quarterly, monthly, yearly)
let SERC = null;
function SER() {
  if (SERC) return SERC;
  const hm = D.four.hicp_m, i0 = hm.indexOf('2005-01'), h = D.four.hicp.slice(i0).map(v => 100 * v), pi = D.infl.y, us = D.overdiff.us, uq = D.overdiff.us_q, N0 = D.brk_real.nile;
  SERC = { gdp: { y: GY, reg: 'ct', d: GQ, f: 'q' }, dgdp: { y: dif(GY, 1), reg: 'c', d: GQ.slice(1), f: 'q' }, hicp: { y: h, reg: 'ct', d: hm.slice(i0), f: 'm' },
    infl: { y: pi, reg: 'c', d: D.infl.m, f: 'm' }, dinfl: { y: dif(pi, 1), reg: 'c', d: D.infl.m.slice(1), f: 'm' }, us: { y: us, reg: 'ct', d: uq, f: 'q' },
    dus: { y: dif(us, 1), reg: 'c', d: uq.slice(1), f: 'q' }, nile: { y: N0.v, reg: 'c', d: N0.d, f: 'y' } };
  return SERC;
}
const SEN = { gdp: '100 ln Romanian real GDP', dgdp: 'Romanian GDP growth', hicp: '100 ln Romanian HICP', infl: 'Romanian 12-month inflation', dinfl: 'change of inflation',
  us: '100 ln US real GDP', dus: 'US GDP growth', nile: 'Nile flow' };
const dlab = (s, i) => (s.f === 'q' ? T.q(s.d[i]) : s.f === 'm' ? T.m(s.d[i]) : s.d[i].slice(0, 4));
const xdate = s => s.d.map(v => (s.f === 'q' ? q2d(v) : m2d(v.slice(0, 7))));

// ---------------------------------------------------------------- Critical values of the Dickey-Fuller test
function blkXCv() {
  const Tn = +$('cv-T').value; set('cv-T', Tn);
  const regs = ['n', 'c', 'ct'], Ts = [50, 100, 250], rows = Ts.concat(Ts.includes(Tn) ? [] : [Tn]).sort((a, b) => a - b).concat([Infinity]);
  table('cv-tb', [[{ v: '' }, { v: X().case.n, cs: 3 }, { v: X().case.c, cs: 3 }, { v: X().case.ct, cs: 3 }], ['T'].concat(...regs.map(() => ['1%', '5%', '10%']))],
    rows.map(t => hiRow([t === Infinity ? '∞' : String(t)].concat(...regs.map(r => mkc(r, t).map(v => num(v, 2)))), t === Tn)), { cap: X().cvCap });
  const tt = range(20, 1000, 5), G = grid('cv-chart', [{ r: 0, c: 0, t: '5% critical value of τ by sample size', xt: 'T (log scale)', yt: '5% critical value', x: { type: 'log' }, y: { range: [-3.8, -1.4] } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0], tr = regs.map(r => ln(tt, tt.map(t => mkc(r, t)[1]), a, DFC[r], DFN[r], { hovertemplate: `${DFN[r]}, T = %{x}: %{y:.3f}<extra></extra>` }));
  tr.push({ x: [250, 250, 250], y: regs.map(r => NB.dfdist[r].q05_sim), mode: 'markers', marker: { color: regs.map(r => DFC[r]), size: 9, symbol: 'diamond' }, name: 'simulated 5% quantile, T = 250', xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{y:.3f}<extra></extra>' },
    ln([20, 1000], [-1.645, -1.645], a, K.navy, 'Normal 5% value −1.645', { line: { color: K.navy, dash: 'dot', width: 1.2 } }));
  tr.push({ x: regs.map(() => Tn), y: regs.map(r => mkc(r, Tn)[1]), mode: 'markers', marker: { color: regs.map(r => DFC[r]), size: 10, symbol: 'circle-open', line: { width: 2.5 } },
    name: `5% critical values at T = ${Tn}`, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'T = %{x}: %{y:.3f}<extra></extra>' });
  G.lay.shapes.push(vl(a, Tn, K.navy, 'dot', 0.8));
  plot('cv-chart', tr, G.lay);
  const c = regs.map(r => nz(mkc(r, Tn)[1], 2));
  out('cv-out', [X().cv1(Tn, ...c), X().cv2(...regs.map(r => nz(NB.dfdist[r].q05_sim, 2)))]);
}

// ---------------------------------------------------------------- Worked example: Romanian real GDP (Dickey-Fuller by hand)
function blkXDf() {
  const end = +$('xd-end').value, reg = $('xd-reg').value, kk = $('xd-k').value, y = GY.slice(0, end + 1); set('xd-end', T.q(GQ[end]));
  const kmax = kmaxOf(y.length, reg), A = /IC$/.test(kk) ? adfAuto(y, reg, kk.toLowerCase()) : null, k = A ? A.k : Math.min(+kk, kmax), F = A ? A.fit : adfFit(y, reg, k);
  const cv = mkc(reg, F.N), p = mkp(F.tau, reg), phi = 1 + F.gam, names = (reg === 'n' ? [] : reg === 'c' ? ['c'] : ['c', 'b (t)']).concat(['γ (y<sub>t−1</sub>)']).concat(range(1, k, 1).map(j => `δ<sub>${j}</sub> (Δy<sub>t−${j}</sub>)`));
  const S = [X().xd1(T.q(GQ[0]), T.q(GQ[end]), F.N, k, A ? kk : null, kmax), X().xd2(reg, nz(F.gam, 4), nz(F.se, 4), nz(phi, 3)), X().xd3(nz(F.gam, 4), nz(F.se, 4), nz(F.tau, 2)),
    X().xd4(reg, F.N, nz(cv[0], 2), nz(cv[1], 2), nz(cv[2], 2), fp(p)), X().xd5(nz(F.tau, 2), nz(cv[1], 2), F.tau < cv[1], F.tau < -1.645)];
  if (phi > 0 && phi < 1) S.push(X().xd6(nz(phi, 3), nz(Math.log(0.5) / Math.log(phi), 1)));
  steps('xd-ws', S);
  table('xd-tb', [X().coef, X().est, 'SE', 't'], F.o.b.map((b, j) => hiRow([names[j], num(b, 4), num(F.o.se[j], 4), num(F.o.t[j], 2)], j === F.ig)), { cap: X().xdCap(F.N, nz(Math.sqrt(F.o.s2), 3)) });
  const ks = range(0, kmax, 1), taus = ks.map(j => adfFit(y, reg, j)), G = grid('xd-chart', [{ r: 0, c: 0, t: `ADF τ by number of lags k (${DFN[reg]})`, xt: 'k', yt: 'τ', x: { dtick: 1 } }], { cols: 1, h: 320, legendRows: 2 });
  const a = G.ax[0];
  plot('xd-chart', [ln(ks, taus.map(f => f.tau), a, K.blue, 'τ', { mode: 'lines+markers', marker: { size: 6 }, hovertemplate: 'k = %{x}: τ = %{y:.2f}<extra></extra>' }),
    ln(ks, taus.map(f => mkc(reg, f.N)[1]), a, K.red, '5% critical value (MacKinnon)', { line: { color: K.red, dash: 'dash', width: 1.2 } }),
    ln(ks, ks.map(() => -1.645), a, K.navy, 'Normal 5% value −1.645', { line: { color: K.navy, dash: 'dot', width: 1.1 } }),
    { x: [k], y: [F.tau], mode: 'markers', marker: { color: K.red, size: 12, symbol: 'circle-open', line: { width: 2 } }, name: 'k used', xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' }], G.lay);
  out('xd-out', [X().xd(nz(F.tau, 2), nz(cv[1], 2), fp(p), F.tau < cv[1])]);
  return { F, cv, p, k };
}

// ---------------------------------------------------------------- Choosing the number of lags k
function blkXLag() {
  const s = SER()[$('lg-s').value], reg = $('lg-reg').value, ic = $('lg-ic').value, A = adfAuto(s.y, reg, ic);
  table('lg-tb', ['k', 'AIC', 'BIC', 'τ'], A.sel.map((f, j) => hiRow([String(j), num(f.aic, 2, j === A.k && ic === 'aic'), num(f.bic, 2, j === A.k && ic === 'bic'), num(f.tau, 2)], j === A.k)),
    { cap: X().lgCap(A.Nsel, A.kmax) });
  steps('lg-ws', [X().lg1(s.y.length, nz(12 * (s.y.length / 100) ** 0.25, 2), A.kmax), X().lg2(A.kmax, A.Nsel, ic.toUpperCase(), A.k), X().lg3(A.k, A.fit.N, nz(A.fit.tau, 2), fp(A.p), nz(A.cv[1], 2))]);
  out('lg-out', [X().lg(X().ser[$('lg-s').value], X().case[reg], ic.toUpperCase(), A.k, nz(A.fit.tau, 2), fp(A.p))]);
  return A;
}

// ---------------------------------------------------------------- Interpreting the power curves (table of size and power)
function blkXPow() {
  const reg = $('pw-reg').value, phi = +$('pw-phi').value, Ts = [50, 100, 250, 500, 1000];
  table('pw-tb', [['φ'].concat(Ts.map(t => `T = ${t}` + ([100, 250, 500].includes(t) ? '' : ' *')))],
    PHIS.map(f => hiRow([nz(f, f === 0.925 || f === 0.975 ? 3 : 2)].concat(Ts.map(t => num(100 * powerAt(t, reg, f), 1, f === 1))), f === phi)), { cap: X().pwCap(X().case[reg]), note: X().pwNote });
  const G = grid('pw-chart', [{ r: 0, c: 0, t: `Rejections of the unit root at φ = ${phi} by sample size`, xt: 'T (log scale)', yt: '% rejections', x: { type: 'log' }, y: { range: [0, 105] } }], { cols: 1, h: 330, legendRows: 2 });
  const a = G.ax[0], tr = [['c', K.blue, 'circle', 'solid'], ['ct', K.purple, 'square', 'dash']].map(([r, c, sym, dash]) => ln(Ts, Ts.map(t => 100 * powerAt(t, r, phi)), a, c, DFN[r],
    { mode: 'lines+markers', marker: { size: 7, symbol: sym }, line: { color: c, width: r === reg ? 2.2 : 1.3, dash }, hovertemplate: `${DFN[r]}, T = %{x}: %{y:.1f}%<extra></extra>` }));
  G.lay.shapes.push(hl(a, 5, K.navy, 'dot', 0.9));
  plot('pw-chart', tr, G.lay);
  const v = t => nz(100 * powerAt(t, reg, phi), 1);
  out('pw-out', [X().pw1(X().case[reg], nz(100 * powerAt(100, reg, 1), 2), nz(100 * powerAt(250, reg, 1), 2), nz(100 * powerAt(500, reg, 1), 2)), X().pw2(nf(phi, phi === 0.925 || phi === 0.975 ? 3 : 2), v(50), v(100), v(250), v(500), v(1000))]);
}

// ---------------------------------------------------------------- The KPSS test: worked example from the partial sums
function blkXKpss() {
  const key = $('xk-s').value, s = SER()[key], reg = $('xk-reg').value, Ls = $('xk-L').value, K0 = kpssG(s.y, reg, Ls === 'auto' ? null : +Ls);
  const n = K0.n, iS = K0.S.reduce((b, v, i) => (Math.abs(v) > Math.abs(K0.S[b]) ? i : b), 0);
  steps('xk-ws', [X().xk1(reg, n, X().ser[key]), X().xk2(nz(K0.S[iS], 1), dlab(s, iS), nz(K0.ss, 0)), X().xk3(nz(K0.g[0], 3), K0.L, Ls === 'auto', nz(K0.lr, 3)),
    X().xk4(nz(K0.ss, 0), n, nz(K0.lr, 3), nz(K0.eta, 3)), X().xk5(nz(K0.eta, 3), nf(K0.crit5, 3), K0.eta > K0.crit5, pk(K0.p))]);
  table('xk-tb', ['j', 'γ̂<sub>j</sub>', X().wgt, X().contrib], K0.g.map((g, j) => [String(j), num(g, 3), num(j ? 1 - j / (K0.L + 1) : 1, 3), num(j ? 2 * g * (1 - j / (K0.L + 1)) : g, 3)])
    .concat([[{ v: `<b>λ̂²</b>` }, '', '', num(K0.lr, 3, true)]]), { cap: X().xkCap(K0.L) });
  const x = xdate(s), G = grid('xk-chart', [{ r: 0, c: 0, t: `Residuals e<sub>t</sub> (${reg === 'c' ? 'deviations from the mean' : 'deviations from a linear trend'})`, x: { type: 'date' } },
    { r: 0, c: 1, t: 'Partial sums S<sub>t</sub> = e<sub>1</sub> + … + e<sub>t</sub>', x: { type: 'date' } }], { cols: 2, h: 340 });
  const [a, b] = G.ax, ends = [x[0], x[x.length - 1]], lam = Math.sqrt(Math.max(K0.lr, 0)), sd = Math.sqrt(K0.g[0]), thr = Math.sqrt(K0.crit5 * n * Math.max(K0.lr, 0)), rms = Math.sqrt(K0.ss / n);
  const two = (y, ax, col, name, dash, g) => [ln(ends, [y, y], ax, col, name, { line: { color: col, dash, width: 1.3 }, legendgroup: g, hovertemplate: `${nf(y, 2)}<extra></extra>` }),
    ln(ends, [-y, -y], ax, col, '', { line: { color: col, dash, width: 1.3 }, legendgroup: g, hovertemplate: `${nf(-y, 2)}<extra></extra>` })];
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.6), hl(b, 0, K.navy, 'solid', 0.6));
  plot('xk-chart', [ln(x, K0.e, a, K.blue, SEN[key], { line: { color: K.blue, width: 1 }, customdata: s.d, hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' }),
    ...two(sd, a, '#9AA0A6', '± σ̂ = √γ̂<sub>0</sub>', 'dot', 's'), ...two(lam, a, K.green, `± λ̂ (long-run, L = ${K0.L})`, 'dash', 'l'),
    ln(x, K0.S, b, K.red, 'partial sums', { line: { color: K.red, width: 1.4 }, customdata: s.d, hovertemplate: '%{customdata}: %{y:.1f}<extra></extra>' }),
    ...two(rms, b, K.red, 'root mean square of S<sub>t</sub>', 'dot', 'r'), ...two(thr, b, K.navy, `5% threshold λ̂√(${nf(K0.crit5, 3)}·T): reject if the RMS is outside`, 'dash', 't')], G.lay);
  out('xk-out', [X().xk(X().ser[key], reg, nz(K0.eta, 3), K0.L, nf(K0.crit5, 3), K0.eta > K0.crit5)]);
  return K0;
}

// ---------------------------------------------------------------- the unit-root table of the slides (D.tab.urt)
function urtRow(e, det, ic, lv) {
  const reg = det === 'slide' ? e.reg : det, a = e.adf[`${reg}_${ic}`], kp = e.kpss[reg] || null, er = e.ers[`${reg}_${ic}`] || null, pp = e.pp[reg];
  const ca = mkc(reg, a[3])[LV[lv]], kc = kp ? KC[reg][lv] : null, ra = a[1] < +lv, rk = kp ? kp[0] > kc : null;
  const v = kp ? (ra ? (rk ? 'conflict' : 'I0') : (rk ? 'I1' : 'inc')) : null;
  return { reg, a, kp, er, pp, ca, kc, ra, rk, v, cpp: mkc(reg, e.n - 1)[LV[lv]] };
}
const fpa = p => (p < 0.001 ? '<&nbsp;' + nf(0.001, 3) : nf(p, 3));
const URS = ['RO GDP', 'RO GDP growth', 'RO HICP', 'RO inflation', 'RO Δ inflation', 'EUR/RON', 'EUR/RON returns', 'BET', 'BET returns', 'S&P 500', 'S&P 500 returns', 'US GDP', 'US GDP growth', 'Nile'];
function blkXUrt() {
  const det = $('ut-det').value, ic = $('ut-ic').value, lv = $('ut-lv').value, U = D.tab.urt, R = U.map(e => urtRow(e, det, ic, lv));
  const big = n => (n >= 1000 ? nf(n, 0).replace(/(\d)(?=(\d{3})+$)/, '$1&nbsp;') : String(n));
  table('ut-tb', [X().utH], U.map((e, i) => { const r = R[i];
    return [X().urtName[i], r.reg === 'ct' ? 'c, t' : r.reg === 'c' ? 'c' : '—', txt(big(e.n)), txt(`${nz(r.a[0], 2)} (${fpa(r.a[1])})`, r.ra), txt(String(r.a[2])),
      txt(`${nz(r.pp[0], 2)}`, r.pp[0] < r.cpp), r.kp ? txt(nz(r.kp[0], 3), r.rk) : txt('—'), r.er ? txt(`${nz(r.er[0], 2)} (${r.er[2]})`, r.er[0] < r.er[3 + LV[lv]]) : txt('—'),
      r.v ? X().verd[r.v] : '—']; }), { cap: X().utCap(Math.round(100 * lv) + '%'), note: X().utNote });
  const G = grid('ut-chart', [{ r: 0, c: 0, t: `ADF τ and the ${Math.round(100 * lv)}% critical value`, xt: 'τ', x: { range: [-12.8, 0.8] }, y: { autorange: 'reversed', tickfont: { size: 10 } } }], { cols: 1, h: 520, legendRows: 1 });
  const a = G.ax[0]; G.lay.margin.l = 108;
  const ys = URS, cl = v => Math.max(-12, v);
  plot('ut-chart', [{ x: R.map(r => cl(r.a[0])), y: ys, mode: 'markers', marker: { color: R.map(r => (r.ra ? K.green : K.red)), size: 9, symbol: R.map(r => (r.a[0] < -12 ? 'triangle-left' : 'circle')) },
    name: 'ADF τ (green: rejected; ◀: below −12)', xaxis: a.xa, yaxis: a.ya, customdata: R.map(r => r.a[0]), hovertemplate: '%{y}: τ = %{customdata:.2f}<extra></extra>' },
  { x: R.map(r => r.ca), y: ys, mode: 'markers', marker: { color: K.navy, size: 14, symbol: 'line-ns', line: { width: 2, color: K.navy } }, name: `${Math.round(100 * lv)}% critical value`, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{y}: %{x:.2f}<extra></extra>' }], G.lay);
  const cnt = k => R.filter(r => r.v === k).length;
  out('ut-out', [X().ut1(det, ic, Math.round(100 * lv) + '%', cnt('I1'), cnt('I0'), cnt('inc'), cnt('conflict'), R.filter(r => !r.v).length), X().ut2(nz(R[0].a[0], 2), fpa(R[0].a[1]), R[0].kp ? nz(R[0].kp[0], 3) : '—', R[0].v)]);
  return R;
}
// ---------------------------------------------------------------- ADF and KPSS together: the 2 x 2 table, filled with the series
function blkXAk() {
  const det = $('ak-det').value, ic = $('ak-ic').value, lv = $('ak-lv').value, U = D.tab.urt, R = U.map(e => urtRow(e, det, ic, lv));
  const cell = (ra, rk) => R.map((r, i) => (r.ra === ra && r.rk === rk ? X().urtShort[i] : null)).filter(Boolean).join('; ') || '—';
  table('ak-tb', ['', X().kNo, X().kYes], [[`<b>${X().aYes}</b>`, `<b>I(0)</b>: ${cell(true, false)}`, `<b>${X().conf}</b>: ${cell(true, true)}`],
    [`<b>${X().aNo}</b>`, `<b>${X().inc}</b>: ${cell(false, false)}`, `<b>I(1)</b>: ${cell(false, true)}`]], { cap: X().akCap(Math.round(100 * lv) + '%') });
  const c = (ra, rk) => R.filter(r => r.ra === ra && r.rk === rk).length;
  out('ak-out', [X().ak(Math.round(100 * lv) + '%', c(true, false), c(false, true), c(false, false), c(true, true))]);
}

// ---------------------------------------------------------------- Interpreting the simulated break: ZA statistic against the no-lag curve
const zsCache = {};
function zsRow(dl) {
  const k = dl.toFixed(1); if (zsCache[k]) return zsCache[k];
  const g = D.brk_sim.grid[k], y = D.brk_sim.u.map((v, t) => v + (t >= 100 ? dl : 0)), Z = zaCurve(y), im = Z.t.indexOf(Math.min(...Z.t));
  return (zsCache[k] = { g, min: Z.t[im], at: Z.bs[im] });
}
function blkXZaSim() {
  const dl = +$('zs-d').value; set('zs-d', nf(dl, 1));
  const ds = range(0, 10, 1).concat(Number.isInteger(dl) ? [] : [dl]).sort((a, b) => a - b), B = D.brk_sim;
  table('zs-tb', [X().zsH], ds.map(d => { const r = zsRow(d), g = r.g; return hiRow([nz(d, 1), num(100 * g.rej, 1), txt(`${nz(g.adf, 2)} (${g.adf_k})`), num(g.adf_p, 3),
    txt(`${nz(g.za, 2)} (${g.za_k})`, g.za < B.crit_za), txt(String(g.za_b + 1)), txt(nz(r.min, 2), r.min < B.crit_za), txt(String(r.at))], d === dl); }), { cap: X().zsCap(nz(B.crit_adf, 2), nz(B.crit_za, 2)) });
  const r = zsRow(dl), g = r.g;
  out('zs-out', [X().zs1(nf(dl, 1), pct(g.rej, 1), nz(g.adf_p, 2)), X().zs2(nz(g.za, 2), g.za_k, g.za_b + 1, nz(r.min, 2), r.at, nz(B.crit_za, 2))]);
  return r;
}

// ---------------------------------------------------------------- Interpreting the two breaks: Zivot-Andrews, three versions
function blkXZa() {
  const reg = $('zt-reg').value, BR = D.brk_real, regs = ['c', 't', 'ct'], rows = [], br = [];
  [['nile', 'Nile'], ['eurron', 'EUR/RON']].forEach(([k, lab]) => regs.forEach(r => {
    const z = BR[k].za[r], m = v => (k === 'eurron' ? nz(Math.exp(v / 100), 2) : nz(v, 0));
    rows.push(hiRow([X().zaSer[k], X().zaReg[r], txt(nz(z.stat, 2), z.stat < z.crit5), num(z.crit5, 2), txt(fp(z.p)), txt(k === 'nile' ? z.brk.slice(0, 4) : T.m(z.brk)), txt(String(z.lags)), txt(m(z.m0)), txt(m(z.m1))], r === reg));
    br.push({ k, lab, r, z });
  }));
  table('zt-tb', [X().ztH], rows, { cap: X().ztCap(nz(BR.nile.adf_p, 3), nz(BR.nile.kpss, 3), nz(BR.eurron.adf_p, 2), nz(BR.eurron.kpss, 3)) });
  const G = grid('zt-chart', [{ r: 0, c: 0, t: 'Zivot-Andrews statistic and its 5% critical value', yt: 'ZA statistic', y: { range: [-8, 0] } }], { cols: 1, h: 320, legendRows: 2 });
  const a = G.ax[0], cat = br.map(b => `${b.lab}<br>${{ c: 'level', t: 'trend', ct: 'both' }[b.r]}`);
  plot('zt-chart', [{ x: cat, y: br.map(b => b.z.stat), type: 'bar', marker: { color: br.map(b => (b.r === reg ? K.red : b.k === 'nile' ? K.blue : K.green)) }, name: 'ZA statistic', xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
    { x: cat, y: br.map(b => b.z.crit5), mode: 'markers', marker: { color: K.navy, size: 18, symbol: 'line-ew', line: { width: 2.5, color: K.navy } }, name: '5% critical value', xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
    (() => { const sel = br.map((b, i) => (b.r === reg ? i : -1)).filter(i => i >= 0);
      return { x: sel.map(i => cat[i]), y: sel.map(i => br[i].z.stat), text: sel.map(i => nz(br[i].z.stat, 2) + (br[i].z.stat < br[i].z.crit5 ? ' ✓' : ' ✗')), mode: 'markers+text', textposition: 'bottom center',
        marker: { color: K.red, size: 11, symbol: 'diamond' }, textfont: { color: K.red, size: 12 }, name: 'selected variant (✓ rejects at 5%)', xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }; })()], G.lay);
  const N0 = BR.nile.za[reg], E = BR.eurron.za[reg];
  out('zt-out', [X().zt1(reg, nz(N0.stat, 2), nz(N0.crit5, 2), N0.brk.slice(0, 4), nz(N0.m0, 0), nz(N0.m1, 0)), X().zt2(nz(E.stat, 2), nz(E.crit5, 2), T.m(E.brk), nz(Math.exp(E.m0 / 100), 2), nz(Math.exp(E.m1 / 100), 2))]);
}

// ---------------------------------------------------------------- Model choice by AICc
const ACI = { aicc: 0, aic: 1, bic: 2 };
function blkXAicc() {
  const O = keysSorted(D.tab.aicc), o = O[+$('ac-end').value], c = $('ac-c').value, G0 = D.tab.aicc[o], ci = ACI[c]; set('ac-end', T.q(o));
  const ks = Object.keys(G0).sort(), ok = ks.filter(k => G0[k][3]), best = ok.reduce((b, k) => (G0[k][ci] < G0[b][ci] ? k : b)), srt = ok.slice().sort((a, b) => G0[a][ci] - G0[b][ci]);
  table('ac-tb', ['', 'q = 0', 'q = 1', 'q = 2'], [0, 1, 2].map(p => [`p = ${p}`].concat([0, 1, 2].map(q => { const v = G0[`${p}${q}`]; return txt(nz(v[ci], 1) + (v[3] ? '' : '<sup>†</sup>'), `${p}${q}` === best); }))),
    { cap: X().acCap(c.toUpperCase().replace('AICC', 'AICc'), T.q(GQ[0]), T.q(o), GQ.indexOf(o)), note: X().acNote });
  const B = G0[best], n = GQ.indexOf(o);
  out('ac-out', [X().ac1(best[0], best[1], c.toUpperCase().replace('AICC', 'AICc'), nz(G0[srt[1]][ci] - B[ci], 1), srt[1][0], srt[1][1]), X().ac2(best[0], best[1], nz(B[4], 2), nz(B[5], 2), nz(B[4] / B[5], 2), nz(B[6], 2), n)]);
  return { best, B, gap: G0[srt[1]][ci] - B[ci] };
}

// ---------------------------------------------------------------- Interpreting the residual checks: Ljung-Box and Jarque-Bera
function blkXDiag() {
  const key = $('dg-mod').value, p = +key[0], q = +key[1], e = D.gdp.models[key].e, n = e.length, kk = p + q, m = +$('dg-m').value; set('dg-m', m);
  table('dg-tb', ['m', 'Q*(m)', X().df, 'p', X().at5], range(1, 16, 1).map(mm => { const Q = ljungBox(e, mm).q, df = mm - kk, pv = df > 0 ? chi2sf(Q, df) : null;
    return hiRow([String(mm), num(Q, 2), txt(df > 0 ? String(df) : '—'), pv === null ? txt('—') : num(pv, 3), txt(pv === null ? '—' : pv < 0.05 ? X().rej : X().norej)], mm === m); }),
  { cap: X().dgCap(p, q, n) });
  const mo = moments(e), lo = Math.min(...e), hi = Math.max(...e), sk = mo.sk;
  table('dg-tb2', [X().mom, X().val], [[X().skew, num(sk, 2)], [X().kurt, num(mo.ku, 2)], ['Jarque–Bera', num(mo.jb, 1)], [X().jbp, txt(fp(chi2sf(mo.jb, 2)))],
    [X().minr, txt(`${nz(lo, 2)} (${T.q(GQ[e.indexOf(lo) + 1])})`)], [X().maxr, txt(`${nz(hi, 2)} (${T.q(GQ[e.indexOf(hi) + 1])})`)]]);
  const Q = ljungBox(e, m).q, df = m - kk;
  out('dg-out', [df > 0 ? X().dg1(m, nz(Q, 2), df, fp(chi2sf(Q, df))) : X().dg0(m, kk), X().dg2(nz(mo.ku, 1), nz(mo.jb, 0))]);
}

// ---------------------------------------------------------------- Forecasting with ARIMA: the variance of a random walk and of a trend-stationary model
function blkXFvar() {
  const h = +$('fv-h').value, phi = +$('fv-phi').value; set('fv-h', h); set('fv-phi', nf(phi, 3));
  const P = NB.gdp_fc, s2d = P.ds.params.sigma2, s2t = P.ts.params.sigma2, z = 1.96;
  const vD = j => j * s2d, vT = j => s2t * (1 - phi ** (2 * j)) / (1 - phi * phi), lim = z * Math.sqrt(s2t / (1 - phi * phi));
  steps('fv-ws', [X().fv1(nz(s2d, 3), h, nz(vD(h), 2), nz(z * Math.sqrt(vD(h)), 2)), X().fv2(nz(s2t, 3), nz(phi, 3), h, nz(vT(h), 2), nz(z * Math.sqrt(vT(h)), 2)),
    X().fv3(nz(lim, 2)), X().fv4(h, nz(Math.sqrt(vD(h) / vT(h)), 2))]);
  const hs = [1, 4, 8, 12, 20, 40].concat([1, 4, 8, 12, 20, 40].includes(h) ? [] : [h]).sort((a, b) => a - b);
  table('fv-tb', ['h', X().fvD, X().fvT, X().ratio], hs.map(j => hiRow([String(j), num(z * Math.sqrt(vD(j)), 2), num(z * Math.sqrt(vT(j)), 2), num(Math.sqrt(vD(j) / vT(j)), 2)], j === h)), { cap: X().fvCap });
  const hh = range(1, 40, 1), G = grid('fv-chart', [{ r: 0, c: 0, t: '95% half-width of the GDP forecast interval (% of GDP)', xt: 'horizon h (quarters)', yt: 'half-width', x: { dtick: 4 } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0];
  plot('fv-chart', [ln(hh, hh.map(j => z * Math.sqrt(vD(j))), a, K.red, 'I(1): ARIMA(0,1,0) with drift, 1.96σ√h', { hovertemplate: 'h = %{x}: %{y:.2f}<extra></extra>' }),
    ln(hh, hh.map(j => z * Math.sqrt(vT(j))), a, K.blue, `I(0) around a trend: AR(1), φ = ${phi.toFixed(3)}`, { hovertemplate: 'h = %{x}: %{y:.2f}<extra></extra>' }),
    ln([1, 40], [lim, lim], a, K.blue, 'limit 1.96σ/√(1 − φ²)', { line: { color: K.blue, dash: 'dot', width: 1.1 } }),
    { x: [h, h], y: [z * Math.sqrt(vD(h)), z * Math.sqrt(vT(h))], mode: 'markers', marker: { color: [K.red, K.blue], size: 10 }, showlegend: false, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' }], G.lay);
  out('fv-out', [X().fv(h, nz(z * Math.sqrt(vD(h)), 1), nz(z * Math.sqrt(vT(h)), 1))]);
  return { d: z * Math.sqrt(vD(h)), t: z * Math.sqrt(vT(h)), lim };
}

// ---------------------------------------------------------------- Worked example: ARIMA(1,1,0) forecasts
function blkX110() {
  const phi = +$('x1-phi').value, sg = +$('x1-s').value, yT = +$('x1-y').value, dT = +$('x1-dy').value, H = +$('x1-H').value;
  set('x1-phi', nf(phi, 2)); set('x1-s', nf(sg, 1)); set('x1-y', nf(yT, 0)); set('x1-dy', nf(dT, 1)); set('x1-H', H);
  const hs = range(1, Math.max(H, 3), 1), dF = hs.map(h => phi ** h * dT), yF = []; dF.reduce((y, d) => { yF.push(y + d); return y + d; }, yT);
  const psi = psiW([phi], [], 1, Math.max(H, 3)), V = []; psi.reduce((c, v) => { V.push(sg * sg * (c + v * v)); return c + v * v; }, 0);
  const half = V.map(v => 1.96 * Math.sqrt(v));
  steps('x1-ws', [X().x11(nz(phi, 2), nz(dT, 1), nz(dF[0], 2), nz(yT, 0), nz(yF[0], 2)), X().x12(nz(phi, 2), nz(dF[0], 2), nz(dF[1], 2), nz(yF[0], 2), nz(yF[1], 2), nz(dF[2], 3), nz(yF[2], 2)),
    X().x13(nz(phi, 2), nz(psi[1], 2), nz(psi[2], 2)), X().x14(nz(sg * sg, 2), nz(V[0], 2), nz(V[1], 2), nz(V[2], 2)), X().x15(nz(half[0], 2), nz(half[1], 2), nz(half[2], 2), nz(half[2] / half[0], 2))]);
  table('x1-tb', ['h', 'Δŷ<sub>T+h</sub>', 'ŷ<sub>T+h</sub>', 'ψ<sub>h−1</sub>', X().var, X().half, X().rwHalf], hs.slice(0, H).map((h, i) => [String(h), num(dF[i], 3), num(yF[i], 2), num(psi[i], 3), num(V[i], 2), num(half[i], 2), num(1.96 * sg * Math.sqrt(h), 2)]), { cap: X().x1Cap });
  const xs = [0].concat(hs.slice(0, H)), yy = [yT].concat(yF.slice(0, H)), G = grid('x1-chart', [{ r: 0, c: 0, t: 'Forecasts from y<sub>T</sub> with 95% intervals', xt: 'horizon h', yt: 'y', x: { dtick: 1 } }], { cols: 1, h: 320, legendRows: 2 });
  const a = G.ax[0], up = [yT].concat(yF.slice(0, H).map((v, i) => v + half[i])), lo = [yT].concat(yF.slice(0, H).map((v, i) => v - half[i]));
  const rw = xs.map(h => 1.96 * sg * Math.sqrt(h));
  plot('x1-chart', [band(xs, lo, up, a, K.red, 0.15, ''), ln(xs, yy, a, K.red, `ARIMA(1,1,0), φ = ${phi.toFixed(2)}`, { mode: 'lines+markers', marker: { size: 6 }, hovertemplate: 'h = %{x}: %{y:.2f}<extra></extra>' }),
    ln(xs, xs.map(() => yT), a, K.blue, 'random walk: flat at y_T', { line: { color: K.blue, dash: 'dash', width: 1.3 } }),
    ln(xs, rw.map(v => yT + v), a, K.blue, 'random walk ±1.96σ√h', { line: { color: K.blue, dash: 'dot', width: 1 } }), ln(xs, rw.map(v => yT - v), a, K.blue, '', { line: { color: K.blue, dash: 'dot', width: 1 } })], G.lay);
  out('x1-out', [X().x1(nz(yF[0], 2), nz(yF[1], 2), nz(yF[2], 2), nz(half[0], 2), nz(half[1], 2), nz(half[2], 2))]);
  return { yF, psi, V, half };
}

// ---------------------------------------------------------------- Interpreting the interval widths: φ = 0.5 (chart) and φ = 0.6 (worked example)
function blkXWid() {
  const h = +$('xw-h').value, f0 = +$('xw-f0').value, f1 = +$('xw-f1').value, th = +$('xw-th').value; set('xw-h', h); set('xw-f0', nf(f0, 2)); set('xw-f1', nf(f1, 2)); set('xw-th', nf(th, 2));
  const n = Math.max(h, 20), M = [[`AR(1), φ = ${nz(f0, 2)} (d = 0)`, [f0], [], 0, Math.abs(f0) < 1 ? 1.96 / Math.sqrt(1 - f0 * f0) : null], [X().rw, [], [], 1, null],
    [`ARIMA(1,1,0), φ = ${nz(f1, 2)}`, [f1], [], 1, null]].concat(Math.abs(f1 - 0.6) > 1e-9 ? [[`ARIMA(1,1,0), φ = ${nz(0.6, 2)} (${X().wex})`, [0.6], [], 1, null]] : [])
    .concat([[`ARIMA(0,1,1), θ = ${nz(th, 2)}`, [], [th], 1, null], ['ARIMA(0,2,0)', [], [], 2, null]]);
  const W = M.map(m => { const p = psiW(m[1], m[2], m[3], n); return { p, w: halfW(p) }; }), rwH = W[1].w[h - 1];
  table('xw-tb', [X().model, 'ψ<sub>1</sub>', 'ψ<sub>2</sub>', 'ψ<sub>3</sub>', 'h = 1', 'h = 4', `h = ${h}`, X().vsRw, X().limit],
    M.map((m, i) => [m[0], num(W[i].p[1], 3), num(W[i].p[2], 3), num(W[i].p[3], 3), num(W[i].w[0], 2), num(W[i].w[3], 2), num(W[i].w[h - 1], 2, true), num(W[i].w[h - 1] / rwH, 2), m[4] ? num(m[4], 2) : txt('∞')]),
    { cap: X().xwCap });
  const i6 = M.findIndex(m => m[3] === 1 && m[1].length && Math.abs(m[1][0] - 0.6) < 1e-9), i5 = 2, w6 = psiW([0.6], [], 1, n), w5 = psiW([0.5], [], 1, n);
  steps('xw-ws', [X().xw1(nz(f0, 2), M[0][4] ? nz(M[0][4], 2) : null, h, nz(W[0].w[h - 1], 2)), X().xw2(h, nz(Math.sqrt(h), 2), nz(W[1].w[h - 1], 2)),
    X().xw3(nz(f1, 2), nz(f1 < 1 ? 1 / (1 - f1) : Infinity, 2), h, nz(W[i5].w[h - 1], 2)), X().xw4(h, nz(halfW(w5)[h - 1], 2), nz(halfW(w6)[h - 1], 2)),
    X().xw5(nz(th, 2), nz(1 + th, 2), h, nz(W[M.length - 2].w[h - 1], 2)), X().xw6(h, nz(W[M.length - 1].w[h - 1], 2))]);
  out('xw-out', [X().xw(h, nz(W[0].w[h - 1], 2), nz(W[1].w[h - 1], 2), nz(W[i5].w[h - 1], 2), nz(W[M.length - 2].w[h - 1], 2), nz(W[M.length - 1].w[h - 1], 2))]);
  return { W, M, i6 };
}

// ---------------------------------------------------------------- Interpreting the GDP forecasts: the table of the two forecasts
function blkXFct() {
  const O = keysSorted(D.gdp_fc), o = O[+$('ft-o').value], F = D.gdp_fc[o], lv = +$('ft-lv').value, z = normPpf(0.5 + lv / 2), H = F.ds.f.length, io = GQ.indexOf(o); set('ft-o', T.q(o));
  const fq = range(1, H, 1).map(h => addQ(o, h));
  table('ft-tb', [[{ v: '' }, { v: '' }, { v: X().ftDS, cs: 3 }, { v: X().ftTS, cs: 3 }, { v: '' }], ['h', X().quarter, X().fc, X().intv(Math.round(100 * lv)), X().halfp, X().fc, X().intv(Math.round(100 * lv)), X().halfp, X().obs]],
    fq.map((q, i) => { const j = GQ.indexOf(q), c = k => [num(bn(F[k].f[i]), 1), txt(`[${nz(bn(F[k].f[i] - z * F[k].se[i]), 1)}; ${nz(bn(F[k].f[i] + z * F[k].se[i]), 1)}]`), num(z * F[k].se[i], 1)];
      return hiRow([String(i + 1), T.q(q)].concat(c('ds'), c('ts'), [j < 0 ? txt('—') : num(bn(GY[j]), 1)]), [1, 4, 12].includes(i + 1)); }), { cap: X().ftCap(T.q(o), nz(bn(GY[io]), 1)) });
  const h12 = H - 1;
  out('ft-out', [X().ft1(Math.round(100 * lv), nz(bn(F.ds.f[h12]), 1), nz(z * F.ds.se[0], 1), nz(z * F.ds.se[3], 1), nz(z * F.ds.se[h12], 1), nz(F.ds.se[h12] / F.ds.se[0], 2)),
    X().ft2(nz(bn(F.ts.f[h12]), 1), nz(z * F.ts.se[h12], 1))]);
}

// ---------------------------------------------------------------- Interpreting the HICP regression: slope, t, R² and DW by hand
function blkXSpur() {
  let y0 = +$('xs-y0').value, y1 = +$('xs-y1').value; if (y1 < y0 + 3) { y1 = Math.min(2026, y0 + 3); $('xs-y1').value = y1; } set('xs-y0', y0); set('xs-y1', y1);
  const kind = $('xs-k').value, S = spurReal(y0, y1), lev = kind === 'lev', x = lev ? S.s : S.dsp, y = lev ? S.h : S.dh, O = lev ? S.L : S.Df, n = y.length;
  const mx = mean(x), my = mean(y), sxx = sum(x.map(v => (v - mx) ** 2)), sxy = sum(x.map((v, i) => (v - mx) * (y[i] - my))), b = sxy / sxx, s2 = O.s2, se = Math.sqrt(s2 / sxx), t = b / se;
  steps('xs-ws', [X().xs1(lev, n, nz(mx, 2), nz(my, 2)), X().xs2(nz(sxy, 1), nz(sxx, 1), nz(b, 4)), X().xs3(nz(s2, 4), nz(se, 4), nz(t, 2)),
    X().xs4(nz(t, 2), n - 2, nz(t * t / (t * t + n - 2), 4)), X().xs5(nz(lev ? S.dwL : S.dwD, 2), lev ? S.L.r2 > S.dwL : S.Df.r2 > S.dwD)]);
  const col = (o, d, nn) => [num(o.b[1], 4), num(o.se[1], 4), num(o.t[1], 2), txt(fp(tsf2(o.t[1], nn - 2))), num(o.r2, 4), num(d, 2)];
  const L = col(S.L, S.dwL, S.h.length), Dd = col(S.Df, S.dwD, S.dh.length), lab = [X().slope, 'SE', 't', 'p', 'R²', 'DW'];
  table('xs-tb', ['', X().levels, X().diffs], [['n', txt(String(S.h.length)), txt(String(S.dh.length))]].concat(lab.map((l, i) => hiRow([l, L[i], Dd[i]], i === 2 || i === 4))), { cap: X().xsCap });
  out('xs-out', [X().xs(nz(S.L.t[1], 1), nz(S.L.r2, 2), nz(S.dwL, 2), nz(S.Df.t[1], 2), nz(S.Df.r2, 4))]);
}

// blocks of the worked examples and result tables (own frames): [anchor, prefix, function, controls, player parameter, values, ms]
const XB = [['wx-spur', 'xs', blkXSpur, ['xs-y0', 'xs-y1', 'xs-k'], 'xs-y0', range(2005, 2022, 1), 600],
  ['cv-table', 'cv', blkXCv, ['cv-T'], 'cv-T', range(20, 1000, 20), 250],
  ['wx-df', 'xd', blkXDf, ['xd-end', 'xd-reg', 'xd-k'], 'xd-k', range(0, 13, 1).map(String), 700],
  ['lag-table', 'lg', blkXLag, ['lg-s', 'lg-reg', 'lg-ic'], 'lg-s', ['gdp', 'dgdp', 'hicp', 'infl', 'dinfl', 'us', 'dus', 'nile'], 1300],
  ['power-table', 'pw', blkXPow, ['pw-reg', 'pw-phi'], 'pw-phi', PHIS.map(String), 900],
  ['wx-kpss', 'xk', blkXKpss, ['xk-s', 'xk-reg', 'xk-L'], 'xk-L', range(0, 20, 1).map(String), 500],
  ['adf-kpss', 'ak', blkXAk, ['ak-det', 'ak-ic', 'ak-lv'], 'ak-lv', ['0.01', '0.05', '0.1'], 1400],
  ['urt-table', 'ut', blkXUrt, ['ut-det', 'ut-ic', 'ut-lv'], 'ut-det', ['slide', 'n', 'c', 'ct'], 1500],
  ['za-sim', 'zs', blkXZaSim, ['zs-d'], 'zs-d', range(0, 10, 0.5), 500],
  ['za-table', 'zt', blkXZa, ['zt-reg'], 'zt-reg', ['c', 't', 'ct'], 1300],
  ['aicc-table', 'ac', blkXAicc, ['ac-end', 'ac-c'], 'ac-end', range(0, 42, 1), 350],
  ['diag-table', 'dg', blkXDiag, ['dg-mod', 'dg-m'], 'dg-m', range(1, 16, 1), 500],
  ['wx-fvar', 'fv', blkXFvar, ['fv-h', 'fv-phi'], 'fv-h', range(1, 40, 1), 200],
  ['wx-110', 'x1', blkX110, ['x1-phi', 'x1-s', 'x1-y', 'x1-dy', 'x1-H'], 'x1-phi', range(-0.9, 0.95, 0.05), 350],
  ['wx-width', 'xw', blkXWid, ['xw-h', 'xw-f0', 'xw-f1', 'xw-th'], 'xw-f1', range(0.05, 0.95, 0.05), 350],
  ['fc-table', 'ft', blkXFct, ['ft-o', 'ft-lv'], 'ft-o', range(0, 42, 1), 350]];
function placeExtras() {                // web lecture: the extra blocks travel inside a chart block; move each to its own frame
  document.querySelectorAll('.xhold > section.xblk').forEach(sec => {
    const art = document.getElementById('s' + sec.dataset.frame), head = art && art.querySelector('.fhead');
    if (head) head.after(sec); else sec.closest('section.blk').after(sec);
  });
  document.querySelectorAll('.xhold').forEach(h => { if (!h.children.length) h.remove(); });
}

// ================================================================= wiring
const ALL = [blkFour, blkTsDs, blkDetrend, blkSpurMC, blkSpurReal, blkDfDist, blkPower, blkKpss, blkBrkSim, blkBrkReal, blkGdpId, blkGdpDiag, blkWidth, blkGdpFc,
  blkOverdiff, blkInfl, blkFx, blkUs];
function init() {
  placeExtras();
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
  const odk = Object.assign(document.createElement('input'), { type: 'range', id: 'od-k', min: 0, max: 3, step: 1, value: 3, hidden: true }); $('od-play').after(odk);
  on(['od-s', 'od-d', 'od-k'], blkOverdiff); P('od-play', 'od-k', range(0, 3, 1), 1300);     // ▶: Romania d = 2, 3, then the US d = 2, 3
  on(['in-m0', 'in-m1'], blkInfl); P('in-play', 'in-m1', ['00', '01', '02', '10', '11', '12', '20', '21', '22', '30', '31', '32'], 900);
  on(['fx-y'], blkFx); P('fx-play', 'fx-y', range(2010, 2025, 1), 700);
  on(['us-y'], blkUs); P('us-play', 'us-y', range(1990, 2019, 1), 450);
  const PFX = { 'four-series': 'fs', 'ts-ds': 'td', detrend: 'dt', 'spurious-mc': 'sm', 'spurious-real': 'sr', 'df-dist': 'dd', 'adf-power': 'ap', 'kpss-sums': 'kp',
    'breaks-sim': 'bs', 'breaks-real': 'br', 'gdp-ident': 'gi', 'gdp-diag': 'gd', 'interval-width': 'iw', 'gdp-forecast': 'gf', overdiff: 'od', inflation: 'in', eurron: 'fx', 'us-gdp': 'us' };
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if (has(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: { out: true } }, sp)); });
  const live = XB.filter(([, x]) => has(x + '-out'));
  if (has('xd-end')) { $('xd-end').max = nq; $('xd-end').value = nq; }
  const nA = Object.keys(D.tab.aicc).length, nF = Object.keys(D.gdp_fc).length;
  if (has('ac-end')) { $('ac-end').max = nA - 1; $('ac-end').value = nA - 1; }
  if (has('ft-o')) { $('ft-o').max = nF - 1; $('ft-o').value = nF - 1; }
  if (has('fv-phi')) $('fv-phi').value = NB.gdp_fc.ts.params['ar.L1'];
  if (has('lg-s')) $('lg-s').addEventListener('input', () => { $('lg-reg').value = SER()[$('lg-s').value].reg; });
  if (has('xk-s')) $('xk-s').addEventListener('input', () => { $('xk-reg').value = SER()[$('xk-s').value].reg; });
  live.forEach(([, x, f, ids, pid, vals, ms]) => { on(ids, f); P(x + '-play', pid, vals, ms); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  ALL.forEach(f => f());
  live.forEach(b => b[2]());
  let w = window.innerWidth, tm = null;
  window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(() => { if ((window.innerWidth < 720) !== (w < 720) || Math.abs(window.innerWidth - w) > 200) { w = window.innerWidth; ALL.forEach(f => f()); live.forEach(b => b[2]()); } }, 250); });
}
window.CH3 = { mkp, mkc, adfFit, adfAuto, kpssG, SER, urtRow, zsRow, XB, ALL, chi2sf, tsf2, ols, dw, tsdsPaths, detrendStats, spurReal, dfShare, powerAt, kpss, kpssPaths, zaCurve, gdpId, moments, psiW, halfW, fxCompare, dif, svar, bn };
if (!window.CH3_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
