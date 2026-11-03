// Chapter 4 interactive page (seasonality and forecasting): one block per chart slide, computed in the browser from ch4_data.js.
(function () {
'use strict';
const D = window.CH4_DATA;
const { sum, solve, acf, pacf, ljungBox, normPpf, normCdf, mean, sdev, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket } = window.TSAI;
const T = window.CH4_TEXT;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of generate_all_charts.py / tsa_style.py
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', orange: '#E67E22', purple: '#8E44AD',
  teal: '#17A2B8', navy: '#1F2A44' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.purple]: '142,68,173', [K.orange]: '230,126,34',
  [K.amber]: '181,133,63', [K.teal]: '23,162,184' };
const MC = { 'Seasonal naive': K.amber, ETS: K.teal, SARIMA: K.purple, DHR: K.blue, TBATS: K.green, Prophet: K.orange, Combination: K.red };

// ---------------------------------------------------------------- small helpers
const lags = n => Array.from({ length: n }, (_, i) => i + 1);
const pvar = x => { const m = mean(x); return sum(x.map(v => (v - m) ** 2)) / x.length; };
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
const pe = p => (p < 0.001 ? 'p < ' + nf(0.001, 3) : 'p = ' + nf(p, 3));
const sgn = (x, d) => (x >= 0 ? '+' : '') + nf(x, d);
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
function lsq(X, y) {              // least squares coefficients and fitted values (columns in X)
  const G = X.map(a => X.map(b => sum(a.map((v, i) => v * b[i])))), b = solve(G, X.map(a => sum(a.map((v, i) => v * y[i]))));
  return { b, fit: y.map((_, i) => X.reduce((s, c, j) => s + b[j] * c[i], 0)) };
}
const polyfit1 = (x, y) => { const mx = mean(x), my = mean(y); const b = sum(x.map((v, i) => (v - mx) * (y[i] - my))) / sum(x.map(v => (v - mx) ** 2)); return [b, my - b * mx]; };
const corr = (a, b) => { const ma = mean(a), mb = mean(b); return sum(a.map((v, i) => (v - ma) * (b[i] - mb))) / Math.sqrt(sum(a.map(v => (v - ma) ** 2)) * sum(b.map(v => (v - mb) ** 2))); };
const ym = (m0, k) => { const y = +m0.slice(0, 4), m = +m0.slice(5, 7) - 1 + k; return `${y + Math.floor(m / 12)}-${String(((m % 12) + 12) % 12 + 1).padStart(2, '0')}`; };
const mdates = o => o.v.map((_, i) => ym(o.m0, i) + '-01');
const qlab = i => `${2000 + Math.floor(i / 4)}Q${i % 4 + 1}`;
const qdate = i => `${2000 + Math.floor(i / 4)}-${String(3 * (i % 4) + 1).padStart(2, '0')}-01`;
const DAY = 864e5, HOUR = 36e5;
const d2t = s => Date.parse(s.slice(0, 10) + 'T00:00:00Z');
const t2d = t => new Date(t).toISOString().slice(0, 10);
const addD = (s, k) => t2d(d2t(s) + k * DAY);
const dow = s => (new Date(d2t(s)).getUTCDay() + 6) % 7;           // Monday = 0
const keysSorted = o => Object.keys(o).sort();
const viridis = u => { const A = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]], x = Math.min(0.9999, Math.max(0, u)) * 4, i = Math.floor(x), f = x - i;
  return 'rgb(' + A[i].map((v, j) => Math.round(v + f * (A[i + 1][j] - v))).join(',') + ')'; };

// ---------------------------------------------------------------- panels: one Plotly figure with several axes; stacked on narrow screens
// cells: [{ r, c, cs, t (title), xt, yt (axis titles), x, y (extra axis options) }]
function grid(id, cells, o) {
  const el = $(id), narrow = el.clientWidth < 560, maxc = Math.floor(el.clientWidth / 6.6);
  const wrap = t => { if (!t || t.replace(/<[^>]+>/g, '').length <= maxc) return t; const w = t.split(' '), L = ['']; w.forEach(x => { const c = L[L.length - 1];
    if ((c + ' ' + x).replace(/<[^>]+>/g, '').trim().length > maxc && c) L.push(x); else L[L.length - 1] = (c + ' ' + x).trim(); }); return L.join('<br>'); };
  if (narrow) cells = cells.map(c => Object.assign({}, c, { t: wrap(c.t) }));
  const extra = narrow ? Math.max(0, ...cells.map(c => (c.t || '').split('<br>').length - 1)) * 15 : 0;
  const C = narrow ? 1 : o.cols, R = narrow ? cells.length : Math.max(...cells.map(c => c.r)) + 1;
  const per = o.rowPx || 210;
  const H = narrow ? Math.max(o.h || 360, 120 + (per + extra) * cells.length) + 26 * ((o.legendRows || 1) - 1) : (o.h || 360);
  const plotH = H - 70 - (o.legendRows || 1) * 26;
  const gx = o.gx || 0.07, gy = Math.min(0.5, (narrow ? 62 + extra : (o.gyPx || 62)) / plotH), w = (1 - gx * (C - 1)) / C, h = (1 - gy * (R - 1)) / R;
  const lay = layout({ height: H, margin: { l: o.ml || 50, r: o.mr || 10, t: 26 + extra, b: 34 }, annotations: [], shapes: [], showlegend: o.legend !== false,
    legend: { orientation: 'h', x: 0, xanchor: 'left', y: -(narrow ? 34 : 40) / plotH, yanchor: 'top', bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } } });
  const ax = cells.map((c, k) => {
    const r = narrow ? k : c.r, cc = narrow ? 0 : c.c, cs = narrow ? 1 : (c.cs || 1);
    const x0 = cc * (w + gx), x1 = x0 + cs * w + (cs - 1) * gx, y1 = 1 - r * (h + gy), y0 = y1 - h, n = k ? k + 1 : '';
    lay['xaxis' + n] = AX(c.xt || '', Object.assign({ domain: [x0, Math.min(1, x1)], anchor: 'y' + n, automargin: false, title: { text: c.xt || '', font: { size: 11, color: K.navy }, standoff: 4 } }, c.x || {}));
    lay['yaxis' + n] = AX(c.yt || '', Object.assign({ domain: [Math.max(0, y0), y1], anchor: 'x' + n, automargin: false, title: { text: c.yt || '', font: { size: 11, color: K.navy }, standoff: 2 } }, c.y || {}));
    if (c.t) lay.annotations.push({ xref: 'paper', yref: 'paper', x: x0, y: y1, xanchor: 'left', yanchor: 'bottom', yshift: 3, showarrow: false, text: c.t, font: { size: 12, color: K.navy } });
    return { xa: 'x' + n, ya: 'y' + n, xd: 'x' + n + ' domain', yd: 'y' + n + ' domain', k: k + 1 };
  });
  el.style.height = H + 'px';
  return { lay, ax, narrow };
}
const hl = (a, y, color, dash, width) => ({ type: 'line', xref: a.xd, x0: 0, x1: 1, yref: a.ya, y0: y, y1: y, line: { color, width: width || 1.2, dash: dash || 'solid' } });
const vl = (a, x, color, dash, width) => ({ type: 'line', yref: a.yd, y0: 0, y1: 1, xref: a.xa, x0: x, x1: x, line: { color, width: width || 1.2, dash: dash || 'solid' } });
const rect = (a, x0, x1, col, al) => ({ type: 'rect', xref: a.xa, yref: a.yd, x0, x1, y0: 0, y1: 1, fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, layer: 'below' });
const leg = (name, line, marker) => Object.assign({ x: [null], y: [null], name, hoverinfo: 'skip' }, line ? { mode: 'lines', line } : { mode: 'markers', marker });
const ln = (x, y, a, color, name, o) => Object.assign({ x, y, mode: 'lines', line: { color, width: 1.3 }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya }, o || {});
const band = (x, lo, hi, a, col, al, name, o) => Object.assign({ x: x.concat([...x].reverse()), y: hi.concat([...lo].reverse()), fill: 'toself',
  fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, mode: 'lines', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' }, o || {});
function acfBars(tr, lay, a, r, T0, color, o = {}) {
  (o.mark || []).forEach(m => lay.shapes.push(vl(a, m, K.orange, 'dot', 0.9)));
  tr.push({ x: lags(r.length), y: r, type: 'bar', width: 0.6, marker: { color }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' });
  if (T0) { const b = 1.96 / Math.sqrt(T0); lay.shapes.push(hl(a, b, K.red, 'dash', 1), hl(a, -b, K.red, 'dash', 1)); }
  lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.7));
}
const BAND = leg('±1.96/√T', { color: K.red, dash: 'dash', width: 1.2 });
const yrange = (v, lo, hi) => [Math.min(lo, Math.min(...v) - 0.05), Math.max(hi, Math.max(...v) + 0.05)];
const outside = (r, b) => r.map((v, i) => (Math.abs(v) > b ? i + 1 : 0)).filter(Boolean);

// ---------------------------------------------------------------- data of the chapter
const GDP = D.sa.gdp.map(v => v / 1000);                          // bn EUR (2010 prices), not adjusted, 2000Q1-2026Q2 (slide version)
const LY = D.sa.gdp.map(v => 100 * Math.log(v));                  // 100 ln Y
const SCA = D.sa.sca.map(v => v / 1000);                          // SCA, 2010Q1-2026Q2
const MONTHS_EN = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
const yearOf = (m0, i) => +ym(m0, i).slice(0, 4), monOf = (m0, i) => +ym(m0, i).slice(5, 7);
let HD = null;                                                     // hourly load, GW
const hourly = () => HD || (HD = { t0: Date.parse(D.load.h0.replace(' ', 'T') + 'Z'), v: D.load.hv });
const hIdx = s => Math.round((Date.parse(s + 'Z') - hourly().t0) / HOUR);
const DL = { d0: D.load.d0, v: D.load.dv };                       // daily mean load, GW, 2022-01-01 - 2026-06-30
const dIdx = s => Math.round((d2t(s) - d2t(DL.d0)) / DAY);
const HOL = D.load.hol;

// ================================================================= 1. six series
const SE_DEF = { gdp: [2000, 2026, 2000, 2026], tour: [2015, 2019, 2010, 2026], infl: [2010, 2019, 2010, 2026] };
function seMeans(k, y0, y1) {
  if (k === 'gdp') { const s = [[], [], [], []]; GDP.forEach((v, i) => { const y = 2000 + Math.floor(i / 4); if (y >= y0 && y <= y1) s[i % 4].push(v); }); return s.map(mean); }
  const o = D.series[k], s = Array.from({ length: 12 }, () => []);
  o.v.forEach((v, i) => { const y = yearOf(o.m0, i); if (y >= y0 && y <= y1) s[monOf(o.m0, i) - 1].push(v); });
  return s.map(a => (a.length ? mean(a) : NaN));
}
function blkSeries() {
  const k = $('se-s').value; let y0 = +$('se-y0').value, y1 = +$('se-y1').value;
  if (y1 < y0) { y1 = y0; $('se-y1').value = y0; }
  set('se-y0', y0); set('se-y1', y1);
  const G = grid('se-chart', [{ r: 0, c: 0, t: 'GDP, not adjusted (bn EUR, 2010 prices)', x: { type: 'date' } }, { r: 0, c: 1, t: 'Retail trade volume (2021 = 100)', x: { type: 'date' } },
    { r: 1, c: 0, t: 'Industrial production (2021 = 100)', x: { type: 'date' } }, { r: 1, c: 1, t: 'Nights in tourist accommodation (millions)', x: { type: 'date' } },
    { r: 2, c: 0, t: 'HICP inflation, month on month (%)', x: { type: 'date' } }, { r: 2, c: 1, t: 'Electricity load, daily mean (GW)', x: { type: 'date' } }], { cols: 2, h: 600, legend: false, gyPx: 48 });
  const [a, b, c, d, e, f] = G.ax, S = D.series, tr = [];
  // every panel: the whole series in grey, the years y0-y1 of the window in colour (▶ widens the window)
  const two = (x, y, ax, col, w, o) => { const inW = x.map(t => +t.slice(0, 4) >= y0 && +t.slice(0, 4) <= y1);
    tr.push(ln(x, y, ax, 'rgba(31,42,68,0.28)', '', { line: { color: 'rgba(31,42,68,0.28)', width: w }, hoverinfo: 'skip' }),
      ln(x, y.map((v, i) => (inW[i] ? v : null)), ax, col, '', Object.assign({ line: { color: col, width: w } }, o))); };
  two(GDP.map((_, i) => qdate(i)), GDP, a, K.blue, 1.1, { customdata: GDP.map((_, i) => qlab(i)), hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' });
  two(mdates(S.ret), S.ret.v, b, K.red, 1.3, { hovertemplate: '%{x|%Y-%m}: %{y:.1f}<extra></extra>' });
  two(mdates(S.ipi), S.ipi.v, c, K.green, 1.3, { hovertemplate: '%{x|%Y-%m}: %{y:.1f}<extra></extra>' });
  two(mdates(S.tour), S.tour.v, d, K.orange, 1.3, { hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' });
  two(mdates(S.infl), S.infl.v, e, K.purple, 1.3, { hovertemplate: '%{x|%Y-%m}: %{y:.2f}%<extra></extra>' });
  two(DL.v.map((_, i) => addD(DL.d0, i)), DL.v, f, K.teal, 0.6, { hovertemplate: '%{x}: %{y:.2f}<extra></extra>' });
  G.lay.shapes.push(hl(e, 0, K.navy, 'solid', 0.5), rect({ gdp: a, tour: d, infl: e }[k], `${y0}-01-01`, `${y1}-12-31`, K.amber, 0.18));
  plot('se-chart', tr, G.lay);
  const m = seMeans(k, y0, y1);
  if (k === 'gdp') out('se-out', [T.se_g(y0, y1, m.map(v => nf(v, 1)), nf(100 * (m[3] / m[0] - 1), 0))]);
  else {
    const ord = m.map((v, i) => [v, i]).sort((p, q) => q[0] - p[0]);
    if (k === 'tour') out('se-out', [T.se_t(y0, y1, nf(m[7] / m[0], 2), T.mon[ord[0][1]], T.mon[ord[11][1]], nf(m[7], 2), nf(m[0], 2))]);
    else out('se-out', [T.se_i(y0, y1, T.mon[ord[0][1]], nf(ord[0][0], 3), T.mon[ord[1][1]], nf(ord[1][0], 3), T.mon[ord[11][1]], nf(ord[11][0], 3))]);
  }
}
function seDefaults() { const [a, b, lo, hi] = SE_DEF[$('se-s').value]; ['se-y0', 'se-y1'].forEach((id, j) => { $(id).min = lo; $(id).max = hi; $(id).value = j ? b : a; }); }

// ================================================================= 2. seasonal plot and subseries
function blkSplot() {
  const yr = +$('sp-y').value; set('sp-y', yr);
  const G = grid('sp-chart', [{ r: 0, c: 0, t: 'Tourism nights by month, 2012-2025 (millions)', x: { tickvals: range(1, 12, 1), ticktext: 'JFMAMJJASOND'.split('') } },
    { r: 0, c: 1, t: 'GDP subseries by quarter, 2005-2026 (bn EUR)', x: { tickvals: [0.5, 1.5, 2.5, 3.5], ticktext: ['Q1', 'Q2', 'Q3', 'Q4'], range: [0, 4] } }], { cols: 2, h: 400, legendRows: 2 });
  const [a, b] = G.ax, o = D.series.tour, tr = [];
  const byY = {}; o.v.forEach((v, i) => { const y = yearOf(o.m0, i); if (y >= 2012 && y <= 2025) (byY[y] = byY[y] || [])[monOf(o.m0, i) - 1] = v; });
  const ys = Object.keys(byY).map(Number).sort();
  ys.forEach((y, i) => { const sp = y === 2020 || y === 2021;
    tr.push(ln(range(1, 12, 1), byY[y], a, y === 2020 ? K.red : viridis(i / (ys.length - 1)), [2012, 2020, 2021, 2025].includes(y) ? String(y) : '',
      { line: { color: y === 2020 ? K.red : viridis(i / (ys.length - 1)), width: sp ? 1.8 : 1, dash: sp ? 'dash' : 'solid' }, legendgroup: 'y', hovertemplate: `${y}, %{x}: %{y:.2f}<extra></extra>` })); });
  tr.push(ln(range(1, 12, 1), byY[yr], a, K.navy, T.selYear(yr), { line: { color: K.navy, width: 3 }, hovertemplate: `${yr}, %{x}: %{y:.2f}<extra></extra>` }));
  const cols = [K.blue, K.green, K.orange, K.red], q0 = 20, qm = [], sx = [], sy = [], sl = [];
  for (let q = 0; q < 4; q++) {
    const idx = range(q0 + q, GDP.length - 1, 4), v = idx.map(i => GDP[i]), x = v.map((_, j) => q + 0.1 + 0.8 * j / (v.length - 1)); qm.push(mean(v));
    tr.push(ln(x, v, b, cols[q], `Q${q + 1}`, { line: { color: cols[q], width: 1.1 }, customdata: idx.map(qlab), hovertemplate: '%{customdata}: %{y:.1f}<extra></extra>' }));
    G.lay.shapes.push({ type: 'line', xref: b.xa, yref: b.ya, x0: q + 0.1, x1: q + 0.9, y0: qm[q], y1: qm[q], line: { color: K.navy, width: 2 } });
    const j = idx.findIndex(i => 2000 + Math.floor(i / 4) === yr); if (j >= 0) { sx.push(x[j]); sy.push(v[j]); sl.push(qlab(idx[j])); }
  }
  tr.push({ x: sx, y: sy, mode: 'markers', marker: { color: K.navy, size: 10, symbol: 'diamond', line: { color: '#fff', width: 1 } }, xaxis: b.xa, yaxis: b.ya, showlegend: false,
    customdata: sl, hovertemplate: '%{customdata}: %{y:.1f}<extra></extra>' });
  tr.push(leg('quarter mean', { color: K.navy, width: 2 }));
  plot('sp-chart', tr, G.lay);
  let ok = 0, n = 0; for (let y = 2005; y <= 2025; y++) { const i = 4 * (y - 2000), v = GDP.slice(i, i + 4); n++; if (v[0] < v[1] && v[1] < v[2] && v[2] < v[3]) ok++; }
  const s = byY[yr], mx = s.indexOf(Math.max(...s)), mn = s.indexOf(Math.min(...s));
  out('sp-out', [T.sp1(yr, nf(s[7], 2), nf(s[0], 2), nf(s[7] / s[0], 1), nf(s[3], 3), T.mon[mx], T.mon[mn]), T.sp2(qm.map(v => nf(v, 1)), ok, n)]);
}

// ================================================================= 3. roots of 1 - z^s
function blkCircle() {
  const s = +$('uc-s').value;
  const G = grid('uc-chart', [{ r: 0, c: 0, t: `s = 4: roots of 1 − z<sup>4</sup>; rings: the ${s} roots of 1 − z<sup>${s}</sup>`, x: { range: [-1.45, 1.45], zeroline: false }, y: { range: [-1.45, 1.45], scaleanchor: 'x', zeroline: false } },
    { r: 0, c: 1, t: `s = ${s}: ${s} roots of 1 − z<sup>${s}</sup>`, x: { range: [-1.45, 1.45], zeroline: false }, y: { range: [-1.45, 1.45], scaleanchor: 'x2', zeroline: false } }], { cols: 2, h: 420, legendRows: 2 });
  const tr = [], th = range(0, 2 * Math.PI + 0.02, 0.02);
  [[G.ax[0], 4, K.blue], [G.ax[1], s, K.red]].forEach(([a, m, col], j) => {
    tr.push(ln(th.map(Math.cos), th.map(Math.sin), a, K.navy, '', { line: { color: K.navy, width: 0.8 }, hoverinfo: 'skip' }));
    G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.4), vl(a, 0, K.navy, 'solid', 0.4));
    const k = range(0, m - 1, 1);
    tr.push({ x: k.map(i => Math.cos(2 * Math.PI * i / m)), y: k.map(i => Math.sin(2 * Math.PI * i / m)), mode: 'markers', marker: { color: col, size: 9 }, xaxis: a.xa, yaxis: a.ya,
      name: j ? `roots of 1 − z<sup>${m}</sup>` : 'roots of 1 − z<sup>4</sup>', showlegend: true, customdata: k.map(i => `k = ${i}: frequency ${i}/${m}`), hovertemplate: '%{customdata}<extra></extra>' });
    tr.push({ x: [1], y: [0], mode: 'markers', marker: { size: 15, color: 'rgba(0,0,0,0)', line: { color: K.green, width: 2 } }, xaxis: a.xa, yaxis: a.ya, name: 'z = 1: zero frequency (trend)', showlegend: j === 0, legendgroup: 'z1', hoverinfo: 'skip' });
    if (m % 2 === 0) tr.push({ x: [-1], y: [0], mode: 'markers', marker: { symbol: 'square', size: 15, color: 'rgba(0,0,0,0)', line: { color: K.orange, width: 2 } }, xaxis: a.xa, yaxis: a.ya, name: 'z = −1: frequency 1/2 (period 2)', showlegend: j === 0, legendgroup: 'z2', hoverinfo: 'skip' });
  });
  const kk = range(0, s - 1, 1);
  tr.push({ x: kk.map(i => Math.cos(2 * Math.PI * i / s)), y: kk.map(i => Math.sin(2 * Math.PI * i / s)), mode: 'markers', marker: { size: 13, color: 'rgba(0,0,0,0)', line: { color: K.red, width: 1.6 } },
    xaxis: G.ax[0].xa, yaxis: G.ax[0].ya, showlegend: false, customdata: kk.map(i => `1 − z<sup>${s}</sup>, k = ${i}: frequency ${i}/${s}`), hovertemplate: '%{customdata}<extra></extra>' });
  plot('uc-chart', tr, G.lay);
  const ks = range(1, Math.floor(s / 2), 1);
  out('uc-out', [T.uc1(s, s % 2 ? 1 : 2, Math.floor((s - 1) / 2)), T.uc2(s, ks.map(k => [k, s / k]))]);
}

// ================================================================= 4. GDP: regular and seasonal differences
function diffs(y) {
  const d1 = y.slice(1).map((v, i) => v - y[i]), d4 = y.slice(4).map((v, i) => v - y[i]), dd = d4.slice(1).map((v, i) => v - d4[i]);
  return { d1, d4, dd };
}
function blkGdpDiff() {
  const e = +$('df-end').value; set('df-end', T.q(qlab(e)));
  const y = LY.slice(0, e + 1), S = diffs(y);
  const G = grid('df-chart', [{ r: 0, c: 0, t: '100 Δ ln Y<sub>t</sub> (quarter on quarter)', x: { type: 'date' } }, { r: 0, c: 1, t: '100 Δ<sub>4</sub> ln Y<sub>t</sub> (year on year)', x: { type: 'date' } },
    { r: 0, c: 2, t: '100 ΔΔ<sub>4</sub> ln Y<sub>t</sub>', x: { type: 'date' } },
    { r: 1, c: 0, xt: 'lag (quarters)', y: { range: [-1, 1] } }, { r: 1, c: 1, xt: 'lag (quarters)', y: { range: [-1, 1] } }, { r: 1, c: 2, xt: 'lag (quarters)', y: { range: [-1, 1] } }],
  { cols: 3, h: 470, legendRows: 1, gx: 0.06 });
  const tr = [], res = [];
  [['d1', 1, K.blue], ['d4', 4, K.green], ['dd', 5, K.purple]].forEach(([k, off, col], j) => {
    const s = S[k], x = s.map((_, i) => qdate(i + off)), r = acf(s, 16);
    tr.push(ln(x, s, G.ax[j], col, '', { line: { color: col, width: 1 }, customdata: x.map((_, i) => qlab(i + off)), hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' }));
    G.lay.shapes.push(hl(G.ax[j], 0, K.navy, 'solid', 0.5));
    acfBars(tr, G.lay, G.ax[j + 3], r, s.length, col, { mark: [4, 8, 12, 16] });
    res.push([s.length, sdev(s), r]);
  });
  tr.push(BAND, leg('seasonal lags 4, 8, 12, 16', { color: K.orange, dash: 'dot', width: 1 }));
  plot('df-chart', tr, G.lay);
  out('df-out', [T.df0(T.q(qlab(e)), y.length)].concat(res.map(([n, sd, r], j) => T.df1(j, n, nf(sd, 1), nf(r[0], 2), nf(r[3], 2), nf(r[7], 2), nf(1.96 / Math.sqrt(n), 2)))));
}

// ================================================================= 5. theoretical ACF and PACF
function maAcf(c, n) {            // autocorrelations 1..n of an MA with coefficients c (c[0] = 1)
  const g = k => { let s = 0; for (let j = 0; j + k < c.length; j++) s += c[j] * c[j + k]; return s; };
  const g0 = g(0); return lags(n).map(k => g(k) / g0);
}
function theory(Phi, Th, th, n = 40) {
  const sar = lags(n).map(k => (k % 12 ? 0 : Phi ** (k / 12)));
  const csma = new Array(13).fill(0); csma[0] = 1; csma[12] = Th;
  const cair = new Array(14).fill(0); cair[0] = 1; cair[1] = th; cair[12] = Th; cair[13] = th * Th;
  const sma = maAcf(csma, n), air = maAcf(cair, n);
  return { sar: [sar, pacf(sar)], sma: [sma, pacf(sma)], air: [air, pacf(air)] };
}
function blkTheory() {
  const Phi = +$('th-Phi').value, Th = +$('th-Th').value, th = +$('th-th').value; set('th-Phi', nf(Phi, 2)); set('th-Th', nf(Th, 2)); set('th-th', nf(th, 2));
  const R = theory(Phi, Th, th), all = [].concat(...Object.values(R).map(p => p[0].concat(p[1]))), yr = yrange(all, -0.75, 0.9);
  const tit = [`SAR(1)<sub>12</sub>, Φ = ${nf(Phi, 2)}`, `SMA(1)<sub>12</sub>, Θ = ${nf(Th, 2)}`, `Airline: θ = ${nf(th, 2)}, Θ = ${nf(Th, 2)}`];
  const cells = []; ['ACF', 'PACF'].forEach((lab, i) => tit.forEach((t, j) => cells.push({ r: i, c: j, t: `${lab}: ${t}`, xt: i ? 'lag (months)' : '', y: { range: yr } })));
  const G = grid('th-chart', cells, { cols: 3, h: 480, gx: 0.06 }), tr = [];
  [[R.sar, K.blue], [R.sma, K.green], [R.air, K.red]].forEach(([p, col], j) => [0, 1].forEach(i => acfBars(tr, G.lay, G.ax[3 * i + j], p[i], 0, col, { mark: [12, 24, 36] })));
  tr.push(leg('seasonal lags 12, 24, 36', { color: K.orange, dash: 'dot', width: 1 }));
  plot('th-chart', tr, G.lay);
  out('th-out', [T.th1(nf(Phi, 2), nf(R.sar[0][11], 3), nf(R.sar[0][23], 3), nf(R.sar[0][35], 3), nf(R.sar[1][11], 3)),
    T.th2(nf(Th, 2), nf(R.sma[0][11], 3), nf(R.sma[1][11], 3), nf(R.sma[1][23], 3), nf(R.sma[1][35], 3)),
    T.th3(nf(th, 2), nf(Th, 2), nf(R.air[0][0], 3), nf(R.air[0][11], 3), nf(R.air[0][10], 3), nf(R.air[0][12], 3))]);
}

// ================================================================= 6. airline: identification
const AIR = D.air.y, AIRL = AIR.map(Math.log);
const AI_TR = { ddl: ['ΔΔ<sub>12</sub> ln y<sub>t</sub>', 13], d12l: ['Δ<sub>12</sub> ln y<sub>t</sub>', 12], d1l: ['Δ ln y<sub>t</sub>', 1], ddy: ['ΔΔ<sub>12</sub> y<sub>t</sub>', 13] };
function airW(k) {
  const L = k === 'ddy' ? AIR : AIRL, d12 = L.slice(12).map((v, i) => v - L[i]);
  if (k === 'd12l') return d12;
  if (k === 'd1l') return L.slice(1).map((v, i) => v - L[i]);
  return d12.slice(1).map((v, i) => v - d12[i]);
}
function blkAirline() {
  const k = $('ai-tr').value, w = airW(k), r = acf(w, 36), p = pacf(r), b = 1.96 / Math.sqrt(w.length), nm = AI_TR[k][0];
  const yr = yrange(r.concat(p), -0.5, 0.5);
  const lv = k === 'ddy', G = grid('ai-chart', [{ r: 0, c: 0, t: `Airline passengers: ${lv ? 'y' : 'ln y'} (left), ${nm} (right)`, x: { type: 'date' } }, { r: 0, c: 1, t: `ACF of ${nm}`, xt: 'lag (months)', y: { range: yr } },
    { r: 0, c: 2, t: `PACF of ${nm}`, xt: 'lag (months)', y: { range: yr } }], { cols: 3, h: 380, legendRows: 2, gx: 0.075, mr: 30 });
  const [a, b1, c] = G.ax, x = AIR.map((_, i) => ym('1949-01', i) + '-01'), tr = [];
  tr.push(ln(x, lv ? AIR : AIRL, a, K.blue, lv ? 'passengers y (thousands)' : 'ln y, log of passengers', { line: { color: K.blue, width: 1.1 }, hovertemplate: lv ? '%{x|%Y-%m}: %{y:.0f}<extra></extra>' : '%{x|%Y-%m}: %{y:.3f}<extra></extra>' }));
  G.lay.yaxis4 = AX('', { overlaying: 'y', side: 'right', anchor: 'x', automargin: false, showgrid: false, zeroline: false, tickfont: { size: 10, color: K.orange } });
  tr.push({ x: x.slice(AIR.length - w.length), y: w, mode: 'lines', line: { color: K.orange, width: 1 }, name: `${nm.replace(/<[^>]+>/g, '')} (right axis)`, xaxis: 'x', yaxis: 'y4', hovertemplate: '%{x|%Y-%m}: %{y:.3f}<extra></extra>' });
  acfBars(tr, G.lay, b1, r, w.length, K.purple, { mark: [12, 24, 36] });
  acfBars(tr, G.lay, c, p, w.length, K.purple, { mark: [12, 24, 36] });
  tr.push(BAND);
  plot('ai-chart', tr, G.lay);
  const f = v => nf(v, 3), list = l => (l.length ? l.join(', ') : '—');
  out('ai-out', [T.ai1(nm, w.length, nf(b, 3)), T.ai2(f(r[0]), f(r[2]), f(r[11]), f(r[23]), list(outside(r, b))),
    T.ai3(f(p[0]), f(p[1]), f(p[2]), f(p[11]), f(p[23]), list(outside(p, b)))]);
}

// ================================================================= 7. airline: forecasts out of sample
function blkAirFc() {
  const yr = +$('af-o').value, o = D.air.org[yr]; set('af-o', T.dec(yr));
  const i0 = (yr - 1949) * 12 + 12, H = AIR.length - i0, xs = range(0, H - 1, 1).map(h => ym('1949-01', i0 + h) + '-01');
  const sn = range(0, H - 1, 1).map(h => AIR[i0 - 12 + (h % 12)]), act = AIR.slice(i0);
  const G = grid('af-chart', [{ r: 0, c: 0, t: `Airline model: ${H} forecasts out of sample`, x: { type: 'date' } }], { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0], s0 = Math.max(0, i0 - 48), tr = [
    ln(AIR.slice(s0).map((_, i) => ym('1949-01', s0 + i) + '-01'), AIR.slice(s0), a, K.blue, 'passengers (thousands)', { line: { color: K.blue, width: 1.3 }, hovertemplate: '%{x|%Y-%m}: %{y:.0f}<extra></extra>' }),
    band(xs, o.lo, o.hi, a, K.red, 0.15, '95% interval'),
    ln(xs, o.f, a, K.red, `airline model, forecast from Dec ${yr}`, { line: { color: K.red, width: 1.4 }, hovertemplate: '%{x|%Y-%m}: %{y:.0f}<extra></extra>' }),
    ln(xs, sn, a, K.amber, 'seasonal naive', { line: { color: K.amber, width: 1.1, dash: 'dash' }, hovertemplate: '%{x|%Y-%m}: %{y:.0f}<extra></extra>' })];
  G.lay.shapes.push(vl(a, `${yr}-12-15`, K.navy, 'dot', 0.6));
  plot('af-chart', tr, G.lay);
  const mape = f => 100 * mean(act.map((v, h) => Math.abs(v - f[h]) / v));
  const L = H - 1, inside = act[L] >= o.lo[L] && act[L] <= o.hi[L];
  out('af-out', [T.af0(yr, yr === 1958, H), T.af1(nf(o.th, 3), nf(o.th_se, 3), nf(o.Th, 3), nf(o.Th_se, 3), nf(100 * o.s, 1)),
    T.af2(nf(mape(o.f), 1), nf(mape(sn), 1)), T.af3(nf(o.f[L], 0), nf(o.lo[L], 0), nf(o.hi[L], 0), nf(act[L], 0), inside), T.af4(nf(o.hi[0] - o.lo[0], 0), nf(o.hi[L] - o.lo[L], 0), H)]);
}

// ================================================================= 8. GDP model diagnostics
function jb(e) {
  const n = e.length, m = mean(e), m2 = sum(e.map(v => (v - m) ** 2)) / n, m3 = sum(e.map(v => (v - m) ** 3)) / n, m4 = sum(e.map(v => (v - m) ** 4)) / n;
  const S = m3 / m2 ** 1.5, Ku = m4 / m2 ** 2, j = n / 6 * (S * S + (Ku - 3) ** 2 / 4);
  return { jb: j, p: Math.exp(-j / 2), S, Ku };
}
function diagStats(m) {
  const M = D.gdp.models[m], e = M.e, lb8 = ljungBox(e, 8, M.k), lb12 = ljungBox(e, 12, M.k);
  const q8 = lb8.q !== undefined ? lb8.q : lb8.Q, q12 = lb12.q !== undefined ? lb12.q : lb12.Q;
  const i = e.reduce((b, v, j) => (Math.abs(v) > Math.abs(e[b]) ? j : b), 0);
  return { M, e, q8, p8: chi2sf(q8, 8 - M.k), p12: chi2sf(q12, 12 - M.k), J: jb(e), io: i, z: e[i] / sdev(e) };
}
const MODEL_ORDER = () => Object.keys(D.gdp.models);
function blkGdpDiag() {
  const m = $('dg-m').value, S = diagStats(m), e = S.e, x = e.map((_, i) => qdate(i + 5));
  const G = grid('dg-chart', [{ r: 0, c: 0, t: 'Residuals (%)', x: { type: 'date' } }, { r: 0, c: 1, t: 'ACF of the residuals', xt: 'lag (quarters)', y: { range: yrange(acf(e, 16), -0.6, 0.6) } },
    { r: 0, c: 2, t: 'Distribution of the residuals' }], { cols: 3, h: 370, legendRows: 2, gx: 0.06 });
  const [a, b, c] = G.ax, tr = [ln(x, e, a, K.blue, '', { line: { color: K.blue, width: 1 }, customdata: e.map((_, i) => qlab(i + 5)), hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' })];
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.5));
  acfBars(tr, G.lay, b, acf(e, 16), e.length, K.blue, { mark: [4, 8, 12, 16] });
  const lo = Math.min(...e), hi = Math.max(...e), nb = 22, w = (hi - lo) / nb, cnt = new Array(nb).fill(0);
  e.forEach(v => { cnt[Math.min(nb - 1, Math.floor((v - lo) / w))]++; });
  tr.push({ x: cnt.map((_, i) => lo + (i + 0.5) * w), y: cnt.map(v => v / (e.length * w)), type: 'bar', width: w * 0.98, marker: { color: 'rgba(23,162,184,0.8)' }, xaxis: c.xa, yaxis: c.ya, name: 'residuals', hovertemplate: '%{x:.1f}: %{y:.3f}<extra></extra>' });
  const mu = mean(e), sd = sdev(e), gx = range(lo - 1, hi + 1, (hi - lo + 2) / 199);
  tr.push(ln(gx, gx.map(v => Math.exp(-0.5 * ((v - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI))), c, K.red, 'Normal density', { line: { color: K.red, width: 1.4 }, hoverinfo: 'skip' }), BAND);
  plot('dg-chart', tr, G.lay);
  const M = S.M, par = Object.entries(M.par).map(([k, v]) => `${({ 'ar.L1': 'φ₁', 'ar.L2': 'φ₂', 'ma.L1': 'θ₁', 'ma.L2': 'θ₂', 'ar.S.L4': 'Φ', 'ma.S.L4': 'Θ' })[k] || k} = ${nf(v[0], 3)} (${nf(v[1], 3)})`).join(', ');
  out('dg-out', [T.dg1(m, M.k, nf(M.aicc, 1), nf(M.bic, 1), nf(M.sigma, 2), m === NB.gdpm.best_bic, m === NB.gdpm.best_aicc), T.dg2(par),
    T.dg3(nf(S.q8, 2), 8 - M.k, pe(S.p8), pe(S.p12)), T.dg4(nf(S.J.jb, 1), pe(S.J.p), T.q(qlab(S.io + 5)), nf(e[S.io], 1), nf(S.z, 1))]);
}

// ================================================================= 9. GDP forecasts
function gdpFc(Th, H = 8) {
  const y = LY, n = y.length, w = []; for (let t = 5; t < n; t++) w.push(y[t] - y[t - 1] - y[t - 4] + y[t - 5]);
  const eps = (th) => { const e = []; w.forEach((v, i) => e.push(v - th * (i >= 4 ? e[i - 4] : 0))); return e; };
  const e = eps(Th), e0 = eps(D.gdp.fc.Th), rms = a => Math.sqrt(mean(a.map(v => v * v)));
  const sig = D.gdp.fc.s * rms(e) / rms(e0);
  const yy = y.slice(), E = e.slice();                        // E[i] is the residual of y[i + 5]
  for (let h = 1; h <= H; h++) { const t = n - 1 + h, j = t - 4 - 5; yy.push(yy[t - 1] + yy[t - 4] - yy[t - 5] + (j < E.length ? Th * E[j] : 0)); }
  const psi = [1]; for (let j = 1; j < H; j++) { const ma = j === 4 ? Th : 0; psi.push(ma + (psi[j - 1] || 0) + (j >= 4 ? psi[j - 4] : 0) - (j >= 5 ? psi[j - 5] : 0)); }
  const se = range(1, H, 1).map(h => sig * Math.sqrt(sum(psi.slice(0, h).map(v => v * v))));
  const f = yy.slice(n);
  return { f, se, sig, yy };
}
function blkGdpFc() {
  const Th = +$('gf-Th').value, F = gdpFc(Th), z95 = normPpf(0.975), z80 = normPpf(0.9), lev = v => Math.exp(v / 100) / 1000;
  set('gf-Th', nf(Th, 3));
  const n = LY.length, xf = range(n, n + 7, 1).map(qdate), s0 = 64;
  const G = grid('gf-chart', [{ r: 0, c: 0, cs: 3, t: 'Romanian GDP: SARIMA forecasts, 8 quarters', x: { type: 'date' } }, { r: 0, c: 3, cs: 2, t: 'Year-on-year growth, 100 Δ<sub>4</sub> ln Y (%)', x: { type: 'date' } }],
    { cols: 5, h: 400, legendRows: 2, gx: 0.04 });
  const [a, b] = G.ax, f = F.f.map(lev), lo95 = F.f.map((v, h) => lev(v - z95 * F.se[h])), hi95 = F.f.map((v, h) => lev(v + z95 * F.se[h]));
  const lo80 = F.f.map((v, h) => lev(v - z80 * F.se[h])), hi80 = F.f.map((v, h) => lev(v + z80 * F.se[h]));
  const fq = range(n, n + 7, 1).map(qlab), g = range(4, F.yy.length - 1, 1).map(t => F.yy[t] - F.yy[t - 4]), gi = range(4, F.yy.length - 1, 1);
  const tr = [ln(GDP.slice(s0).map((_, i) => qdate(s0 + i)), GDP.slice(s0), a, K.blue, 'GDP, not adjusted (bn EUR)', { line: { color: K.blue, width: 1.3 }, customdata: GDP.slice(s0).map((_, i) => qlab(s0 + i)), hovertemplate: '%{customdata}: %{y:.1f}<extra></extra>' }),
    band(xf, lo95, hi95, a, K.red, 0.12, '95% interval'), band(xf, lo80, hi80, a, K.red, 0.22, '80% interval'),
    ln(xf, f, a, K.red, 'SARIMA forecast', { mode: 'lines+markers', marker: { size: 5 }, customdata: fq, hovertemplate: '%{customdata}: %{y:.1f}<extra></extra>' }),
    { x: gi.slice(-16, -8).map(qdate), y: g.slice(-16, -8), type: 'bar', marker: { color: K.blue }, name: 'observed', xaxis: b.xa, yaxis: b.ya, customdata: gi.slice(-16, -8).map(qlab), hovertemplate: '%{customdata}: %{y:.2f}%<extra></extra>' },
    { x: gi.slice(-8).map(qdate), y: g.slice(-8), type: 'bar', marker: { color: K.red }, name: 'implied by the forecast', xaxis: b.xa, yaxis: b.ya, customdata: gi.slice(-8).map(qlab), hovertemplate: '%{customdata}: %{y:.2f}%<extra></extra>' }];
  G.lay.shapes.push(hl(b, 0, K.navy, 'solid', 0.5));
  plot('gf-chart', tr, G.lay);
  const lect = Math.abs(Th - D.gdp.fc.Th) < 5e-5;
  out('gf-out', [T.gf0(nf(Th, 3), nf(1 + Th, 2), lect, nf(F.sig, 2)), T.gf1(T.q(fq[0]), f.slice(0, 4).map(v => nf(v, 1))),
    T.gf2(nf(lo95[0], 1), nf(hi95[0], 1), nf(lo95[7], 1), nf(hi95[7], 1)), T.gf3(g.slice(-8).map(v => nf(v, 2)))]);
}

// ================================================================= 10. not adjusted and adjusted
function blkSa() {
  const q = +$('sa-q').value; set('sa-q', T.q(qlab(q)));
  const G = grid('sa-chart', [{ r: 0, c: 0, t: 'GDP, bn EUR (2010 prices)', x: { type: 'date' } }, { r: 0, c: 1, t: 'Retail trade volume, 2021 = 100', x: { type: 'date' } },
    { r: 1, c: 0, t: 'GDP: factor NSA / SCA by quarter', x: { type: 'date' } }, { r: 1, c: 1, t: 'Retail trade: factor NSA / SCA', x: { type: 'date' } }], { cols: 2, h: 560, legendRows: 2, gyPx: 70 });
  const [a, b, c, d] = G.ax, N0 = 40, gx = SCA.map((_, i) => qdate(N0 + i)), gn = GDP.slice(N0), R = D.sa.ret, RS = D.sa.ret_sca;
  const tr = [ln(gx, gn, a, K.blue, 'not adjusted (NSA)', { line: { color: K.blue, width: 1 }, legendgroup: 'n', customdata: gx.map((_, i) => qlab(N0 + i)), hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' }),
    ln(gx, SCA, a, K.red, 'adjusted (SCA)', { line: { color: K.red, width: 1.6 }, legendgroup: 's', customdata: gx.map((_, i) => qlab(N0 + i)), hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' }),
    ln(mdates(R), R.v, b, K.blue, '', { line: { color: K.blue, width: 0.9 }, legendgroup: 'n', hovertemplate: '%{x|%Y-%m}: %{y:.1f}<extra></extra>' }),
    ln(mdates(RS), RS.v, b, K.red, '', { line: { color: K.red, width: 1.6 }, legendgroup: 's', hovertemplate: '%{x|%Y-%m}: %{y:.1f}<extra></extra>' })];
  const cols = [K.blue, K.green, K.orange, K.red];
  for (let k = 0; k < 4; k++) { const ii = range(k, SCA.length - 1, 4);
    tr.push(ln(ii.map(i => gx[i]), ii.map(i => gn[i] / SCA[i]), c, cols[k], `Q${k + 1}`, { mode: 'lines+markers', marker: { size: 4 }, line: { color: cols[k], width: 1.3 }, customdata: ii.map(i => qlab(N0 + i)), hovertemplate: '%{customdata}: %{y:.3f}<extra></extra>' })); }
  const nR = Math.min(R.v.length, RS.v.length);
  tr.push(ln(mdates(R).slice(0, nR), R.v.slice(0, nR).map((v, i) => v / RS.v[i]), d, K.purple, 'retail: factor NSA / SCA', { hovertemplate: '%{x|%Y-%m}: %{y:.3f}<extra></extra>' }));
  G.lay.shapes.push(hl(c, 1, K.navy, 'solid', 0.5), hl(d, 1, K.navy, 'solid', 0.5), vl(a, qdate(q), K.amber, 'dash', 1.2), vl(c, qdate(q), K.amber, 'dash', 1.2));
  const iq = q - N0, rm = range(0, 2, 1).map(j => 12 * (2000 + Math.floor(q / 4) - +R.m0.slice(0, 4)) + 3 * (q % 4) + j).filter(j => j >= 0 && j < nR), rx = rm.map(j => ym(R.m0, j) + '-01');
  const pt = (x, y, ax, nm) => ({ x, y, mode: 'markers', marker: { size: 9, color: K.amber, line: { color: K.navy, width: 1 } }, xaxis: ax.xa, yaxis: ax.ya, name: nm, showlegend: !!nm, legendgroup: 'q', hovertemplate: '%{y:.3f}<extra></extra>' });
  tr.push(pt([qdate(q), qdate(q)], [gn[iq], SCA[iq]], a, `${qlab(q)} (and its months)`), pt(rx.concat(rx), rm.map(j => R.v[j]).concat(rm.map(j => RS.v[j])), b, ''),
    pt([qdate(q)], [gn[iq] / SCA[iq]], c, ''), pt(rx, rm.map(j => R.v[j] / RS.v[j]), d, ''));
  plot('sa-chart', tr, G.lay);
  const i = q - N0, L = v => 100 * Math.log(v), yr = 2000 + Math.floor(q / 4), y0 = 4 * (yr - 2000) - N0;
  const res = [T.sa1(T.q(qlab(q)), nf(L(gn[i]) - L(gn[i - 1]), 2), nf(L(SCA[i]) - L(SCA[i - 1]), 2)),
    T.sa3(nf(gn[i] / SCA[i], 3))];
  if (i >= 4) res.splice(1, 0, T.sa2(nf(L(gn[i]) - L(gn[i - 4]), 2), nf(L(SCA[i]) - L(SCA[i - 4]), 2)));
  if (y0 >= 0 && y0 + 3 < SCA.length) res.push(T.sa4(yr, nf(sum(gn.slice(y0, y0 + 4)), 1), nf(sum(SCA.slice(y0, y0 + 4)), 1)));
  out('sa-out', res);
}

// ================================================================= 11. Fourier approximation
// K = 6 as in the chart of the lecture: sin and cos for k = 1..5 plus sin(2π6t/12), which is zero at every month (11 regressors, 10 effective);
// full = true uses cos(2π6t/12) instead, and the 11 coefficients reproduce the 12 monthly points.
function fourierCols(t, K0, full) {
  const X = []; for (let k = 1; k <= Math.min(K0, 5); k++) { X.push(t.map(u => Math.sin(2 * Math.PI * k * u / 12))); X.push(t.map(u => Math.cos(2 * Math.PI * k * u / 12))); }
  if (K0 === 6) X.push(t.map(u => (full ? Math.cos(Math.PI * u) : Math.sin(Math.PI * u))));
  return X;
}
function fourierFit(K0, full) {
  const t = range(1, 12, 1), pat = D.fourier.pat, X = fourierCols(t, K0, full), F = lsq(K0 === 6 && !full ? X.slice(0, -1) : X, pat);
  const b = K0 === 6 && !full ? F.b.concat([0]) : F.b;     // the coefficient of the zero column is 0 (minimum-norm least squares)
  const r2 = 1 - sum(pat.map((v, i) => (v - F.fit[i]) ** 2)) / sum(pat.map(v => v * v));
  return { b, r2, nc: X.length };
}
function blkFourier() {
  const K0 = +$('fo-K').value; set('fo-K', K0);
  const G = grid('fo-chart', [{ r: 0, c: 0, cs: 3, t: 'Tourism nights: seasonal pattern and Fourier fits', x: { tickvals: range(1, 12, 1), ticktext: 'JFMAMJJASOND'.split('') } },
    { r: 0, c: 3, cs: 2, t: 'The first three harmonics', x: { tickvals: range(1, 12, 1) } }], { cols: 5, h: 430, legendRows: 3, gx: 0.05 });
  const [a, b] = G.ax, tt = range(0.6, 12.4, 11.8 / 299), t = range(1, 12, 1);
  const tr = [{ x: t, y: D.fourier.pat, mode: 'markers', marker: { color: K.navy, size: 8 }, name: 'monthly seasonal effect (log)', xaxis: a.xa, yaxis: a.ya, hovertemplate: 'month %{x}: %{y:.3f}<extra></extra>' }];
  const show = [[1, K.blue], [2, K.green], [3, K.orange], [6, K.red]];
  if (![1, 2, 3, 6].includes(K0)) show.push([K0, K.purple]);
  show.sort((p, q) => p[0] - q[0]).forEach(([k, col]) => { const F = fourierFit(k), X = fourierCols(tt, k), y = tt.map((_, i) => X.reduce((s, c, j) => s + F.b[j] * c[i], 0));
    tr.push(ln(tt, y, a, col, `K = ${k}: ${F.nc} coefficients, R² = ${F.r2.toFixed(3)}`, { line: { color: col, width: k === K0 ? 3.2 : 1.4 }, hovertemplate: `K = ${k}, t = %{x:.1f}: %{y:.3f}<extra></extra>` })); });
  [[1, K.blue], [2, K.green], [3, K.orange]].forEach(([k, col]) => {
    tr.push(ln(tt, tt.map(u => Math.cos(2 * Math.PI * k * u / 12)), b, col, `cos(2π${k}t/12)`, { hoverinfo: 'skip' }),
      ln(tt, tt.map(u => Math.sin(2 * Math.PI * k * u / 12)), b, col, `sin(2π${k}t/12)`, { line: { color: col, width: 1, dash: 'dash' }, hoverinfo: 'skip' })); });
  { const FK = fourierFit(K0), j = Math.min(K0, 6), Xk = fourierCols(tt, K0), c0 = 2 * j - 2;
    tr.push(ln(tt, tt.map((_, i) => (j === 6 ? FK.b[10] * Xk[10][i] : FK.b[c0] * Xk[c0][i] + FK.b[c0 + 1] * Xk[c0 + 1][i])), b, K.navy, `harmonic k = ${K0} in the fit (α sin + β cos)`,
      { line: { color: K.navy, width: 3 }, hovertemplate: `k = ${K0}, t = %{x:.1f}: %{y:.3f}<extra></extra>` })); }
  plot('fo-chart', tr, G.lay);
  const F = fourierFit(K0), amp = range(1, Math.min(K0, 5), 1).map(k => nf(Math.hypot(F.b[2 * k - 2], F.b[2 * k - 1]), 3));
  const res = [T.fo1(K0, F.nc, nf(F.r2, 3)), T.fo2(amp)];
  if (K0 === 6) res.push(T.fo3(nf(fourierFit(5).r2, 3), nf(fourierFit(6, true).r2, 3)));
  out('fo-out', res);
}

// ================================================================= 12. Easter and retail trade
const EAS = D.easter.eas;
function easterShare(m0, n, w) {
  const s = new Array(n).fill(0), y0 = +m0.slice(0, 4), base = y0 * 12 + +m0.slice(5, 7) - 1;
  Object.values(EAS).forEach(e => { for (let j = 1; j <= w; j++) { const d = addD(e, -j), k = +d.slice(0, 4) * 12 + +d.slice(5, 7) - 1 - base; if (k >= 0 && k < n) s[k] += 1 / w; } });
  return s;
}
function easterPts(w) {
  const o = D.easter.food, n = o.v.length, sh = easterShare(o.m0, n, w), P = [];
  for (let i = 12; i < n; i++) { const y = yearOf(o.m0, i), m = monOf(o.m0, i);
    if ([3, 4, 5].includes(m) && y !== 2020 && y !== 2021) P.push({ m, y, g: 100 * (o.v[i] - o.v[i - 12]), ds: sh[i] - sh[i - 12] }); }
  return P;
}
function blkEaster() {
  const w = +$('ea-w').value; set('ea-w', w);
  const G = grid('ea-chart', [{ r: 0, c: 0, cs: 3, t: 'Orthodox Easter Sunday, 2000-2030', yt: 'day of April' }, { r: 0, c: 3, cs: 2, t: 'Food retail: March-May growth and Easter', xt: 'change in the Easter share of the month (y/y)', yt: 'y/y growth (%)' }],
    { cols: 5, h: 420, legendRows: 2, gx: 0.075 });
  const [a, b] = G.ax, yrs = Object.keys(EAS).map(Number), tr = [];
  const doy = yrs.map(y => (d2t(EAS[y]) - d2t(`${y}-03-31`)) / DAY), col = yrs.map(y => (EAS[y].slice(5, 7) === '05' ? K.red : (EAS[y].slice(5, 7) === '03' ? K.green : K.blue)));
  tr.push({ x: yrs, y: doy, mode: 'markers', marker: { color: col, size: 7 }, xaxis: a.xa, yaxis: a.ya, showlegend: false, customdata: yrs.map(y => EAS[y]), hovertemplate: '%{customdata}<extra></extra>' },
    leg('Easter in April', null, { color: K.blue, size: 7 }), leg('Easter in May', null, { color: K.red, size: 7 }));
  tr.unshift({ x: [].concat(...yrs.map(y => [y, y, null])), y: [].concat(...doy.map(v => [v - w, v - 1, null])), mode: 'lines', line: { color: 'rgba(181,133,63,0.55)', width: 3 }, xaxis: a.xa, yaxis: a.ya,
    name: `the w = ${w} days before Easter`, hoverinfo: 'skip' });
  G.lay.shapes.push(hl(a, 0.5, K.navy, 'solid', 0.5), hl(a, 30.5, K.navy, 'solid', 0.5));
  const P = easterPts(w), [sl, ic] = polyfit1(P.map(p => p.ds), P.map(p => p.g));
  [[3, K.green, 'March'], [4, K.blue, 'April'], [5, K.red, 'May']].forEach(([m, c, nm]) => { const q = P.filter(p => p.m === m);
    tr.push({ x: q.map(p => p.ds), y: q.map(p => p.g), mode: 'markers', marker: { color: c, size: 7 }, name: nm, xaxis: b.xa, yaxis: b.ya, customdata: q.map(p => `${nm} ${p.y}`), hovertemplate: '%{customdata}: %{x:.2f}, %{y:.1f}%<extra></extra>' }); });
  tr.push(ln([-1, 1], [ic - sl, ic + sl], b, K.navy, `OLS line, slope ${sl.toFixed(1)} pp`, { line: { color: K.navy, width: 1, dash: 'dash' }, hoverinfo: 'skip' }));
  G.lay.shapes.push(hl(b, 0, K.navy, 'solid', 0.4), vl(b, 0, K.navy, 'solid', 0.4));
  plot('ea-chart', tr, G.lay);
  const R = D.easter.reg[w], f = R.food, t = R.total, may = yrs.filter(y => y <= 2026 && EAS[y].slice(5, 7) === '05');
  out('ea-out', [T.ea1(w, w === 10, nf(sl, 1), P.length), T.ea2(nf(100 * f[0], 1), nf(100 * f[1], 1), nf(f[0] / f[1], 1), nf(100 * f[2], 2), nf(100 * f[3], 2)),
    T.ea3(nf(100 * t[0], 1), nf(100 * t[1], 1), nf(t[0] / t[1], 1), nf(100 * t[2], 2), nf(100 * t[3], 2)), T.ea4(may.join(', '))]);
}

// ================================================================= 13. hourly load
const LH_W0 = '2022-01-03', LH_N = Math.floor((d2t('2026-06-30') - d2t(LH_W0)) / DAY / 7) - 2;
const ACF_Y = {};
function hourAcf(yr) {
  if (ACF_Y[yr]) return ACF_Y[yr];
  const i0 = Math.max(0, hIdx(`${yr}-01-01T00:00:00`)), i1 = hIdx(`${yr}-12-31T23:00:00`), x = hourly().v.slice(i0, i1 + 1), r = acf(x, 400);
  const prof = new Array(24).fill(0), cnt = new Array(24).fill(0); x.forEach((v, i) => { prof[i % 24] += v; cnt[i % 24]++; });
  return (ACF_Y[yr] = { r, prof: prof.map((v, i) => v / cnt[i]), n: x.length });
}
function blkHourly() {
  const k = +$('lh-wk').value, d0 = addD(LH_W0, 7 * k), d1 = addD(d0, 20), yr = $('lh-yr').value; set('lh-wk', T.dmy(d0));
  const i0 = hIdx(`${d0}T00:00:00`), x = hourly().v.slice(i0, i0 + 21 * 24), ts = x.map((_, i) => new Date(d2t(d0) + i * HOUR).toISOString().slice(0, 16).replace('T', ' '));
  const A = hourAcf(yr);
  const G = grid('lh-chart', [{ r: 0, c: 0, cs: 3, t: `Hourly load, ${T.dmyEn(d0)} - ${T.dmyEn(d1)} (GW)`, x: { type: 'date', tickformat: '%d %b' } }, { r: 0, c: 3, cs: 2, t: 'ACF of hourly load', xt: 'lag (hours)' }],
    { cols: 5, h: 380, legendRows: 2, gx: 0.05 });
  const [a, b] = G.ax, tr = [ln(ts, x, a, K.teal, 'hourly load (GW)', { line: { color: K.teal, width: 0.9 }, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }),
    leg('weekends', null, { color: 'rgba(181,133,63,0.35)', size: 12, symbol: 'square' })];
  for (let j = 0; j < 21; j++) { const d = addD(d0, j); if (dow(d) === 5) G.lay.shapes.push(rect(a, d, addD(d, 2), K.amber, 0.15)); }
  tr.push(ln(lags(400), A.r, b, K.teal, `sample ACF, ${yr}`, { hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' }), leg('lags 24, 168, 336', { color: K.orange, dash: 'dot', width: 1 }));
  [24, 168, 336].forEach(m => G.lay.shapes.push(vl(b, m, K.orange, 'dot', 0.9)));
  G.lay.shapes.push(hl(b, 0, K.navy, 'solid', 0.5));
  plot('lh-chart', tr, G.lay);
  const pk = A.prof.indexOf(Math.max(...A.prof)), tg = A.prof.indexOf(Math.min(...A.prof));
  out('lh-out', [T.lh1(T.dmy(d0), T.dmy(d1), nf(mean(x), 2), nf(Math.min(...x), 2), nf(Math.max(...x), 2)),
    T.lh2(yr, A.n, nf(A.r[11], 2), nf(A.r[23], 3), nf(A.r[83], 2), nf(A.r[167], 3)), T.lh3(yr, pk, nf(A.prof[pk], 1), tg, nf(A.prof[tg], 1))]);
}

// ================================================================= 14. MSTL
function mstlStats(d0) {
  const W = D.load.mstl[d0], i0 = hIdx(`${d0}T00:00:00`), y = hourly().v.slice(i0, i0 + 672), rem = y.map((v, i) => v - W.tr[i] - W.s24[i] - W.s168[i]);
  const tot = pvar(y.map((v, i) => v - W.tr[i])), ptp = a => Math.max(...a) - Math.min(...a);
  return { W, y, rem, sh24: pvar(W.s24) / tot, sh168: pvar(W.s168) / tot, shr: pvar(rem) / tot, a24: ptp(W.s24.slice(0, 168)), a168: ptp(W.s168.slice(0, 168)) };
}
function blkMstl() {
  const d0 = $('ms-w').value, S = mstlStats(d0), d1 = addD(d0, 27), ts = S.y.map((_, i) => new Date(d2t(d0) + i * HOUR).toISOString().slice(0, 16).replace('T', ' '));
  const comp = [['load (GW)', S.y, K.teal], ['trend', S.W.tr, K.blue], ['daily seasonal (period 24)', S.W.s24, K.green], ['weekly seasonal (period 168)', S.W.s168, K.orange], ['remainder', S.rem, K.purple]];
  const G = grid('ms-chart', comp.map((c, i) => ({ r: i, c: 0, t: i ? c[0] : `MSTL decomposition of hourly load, ${T.dmyEn(d0)} - ${T.dmyEn(d1)}: ${c[0]}`, x: { type: 'date', tickformat: '%d %b', showticklabels: i === 4 } })),
    { cols: 1, h: 620, legend: false, gyPx: 30, rowPx: 120 });
  plot('ms-chart', comp.map((c, i) => ln(ts, c[1], G.ax[i], c[2], '', { line: { color: c[2], width: 0.8 }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' })), G.lay);
  out('ms-out', [T.ms1(T.dmy(d0), T.dmy(d1), d0 === '2025-02-03'), T.ms2(nf(100 * S.sh24, 0), nf(100 * S.sh168, 0), nf(100 * S.shr, 0)), T.ms3(nf(S.a24, 1), nf(S.a168, 1))]);
}

// ================================================================= 15. daily load and holidays
function dailyStats(p) {
  const v = DL.v, days = v.map((_, i) => addD(DL.d0, i)), inP = d => p === 'all' || d.slice(0, 4) === p;
  const wd = Array.from({ length: 7 }, () => []), mo = Array.from({ length: 12 }, () => []), sel = [];
  days.forEach((d, i) => { if (!inP(d)) return; sel.push(i); mo[+d.slice(5, 7) - 1].push(v[i]); if (!HOL[d]) wd[dow(d)].push(v[i]); });
  const W = wd.map(mean), low = [...sel].sort((x, y) => v[x] - v[y]).slice(0, 3), eas = sel.filter(i => HOL[days[i]] === 'Easter');
  return { W, days, sel, low, eas: eas.length ? mean(eas.map(i => v[i])) : null, m: mean(sel.map(i => v[i])), jm: mo[0].length && mo[4].length ? mean(mo[0]) / mean(mo[4]) - 1 : null };
}
function blkDaily() {
  const p = $('ld-yr').value, S = dailyStats(p), v = DL.v, x = S.days;
  const G = grid('ld-chart', [{ r: 0, c: 0, cs: 3, t: 'Daily electricity load in Romania, 2022-2026', x: { type: 'date' } },
    { r: 0, c: 3, cs: 2, t: `Mean load by day (no holidays), GW${p === 'all' ? '' : ', ' + p}`, x: { tickvals: range(0, 6, 1), ticktext: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] },
      y: { range: [Math.min(...S.W) * 0.85, Math.max(...S.W) * 1.04] } }], { cols: 5, h: 380, legendRows: 2, gx: 0.06 });
  const [a, b] = G.ax, hi = x.map((d, i) => (HOL[d] ? i : -1)).filter(i => i >= 0), ei = hi.filter(i => HOL[x[i]] === 'Easter');
  const tr = [ln(x, v, a, K.teal, 'daily mean load (GW)', { line: { color: K.teal, width: 0.7 }, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }),
    { x: hi.map(i => x[i]), y: hi.map(i => v[i]), mode: 'markers', marker: { color: K.orange, size: 5 }, name: 'public holidays', xaxis: a.xa, yaxis: a.ya, customdata: hi.map(i => HOL[x[i]]), hovertemplate: '%{x}, %{customdata}: %{y:.2f}<extra></extra>' },
    { x: ei.map(i => x[i]), y: ei.map(i => v[i]), mode: 'markers', marker: { color: K.red, size: 6.5 }, name: 'Orthodox Easter (Friday to Monday)', xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
    { x: range(0, 6, 1), y: S.W, type: 'bar', marker: { color: [K.blue, K.blue, K.blue, K.blue, K.blue, K.orange, K.red] }, xaxis: b.xa, yaxis: b.ya, showlegend: false, hovertemplate: '%{y:.3f} GW<extra></extra>' }];
  if (p !== 'all') G.lay.shapes.push(rect(a, `${p}-01-01`, `${p}-12-31`, K.amber, 0.15));
  const lw = p === 'all' ? [] : S.low;
  tr.push({ x: lw.map(i => x[i]), y: lw.map(i => v[i]), mode: 'markers', marker: { size: 13, color: 'rgba(0,0,0,0)', line: { color: K.navy, width: 2 } }, xaxis: a.xa, yaxis: a.ya, name: 'three lowest days of the year',
    hovertemplate: '%{x}: %{y:.2f}<extra></extra>' });
  plot('ld-chart', tr, G.lay);
  const res = [T.ld1(p, S.sel.length), T.ld2(nf(100 * (S.W[5] / S.W[2] - 1), 1), nf(100 * (S.W[6] / S.W[2] - 1), 1))];
  if (S.jm !== null) res.push(T.ld3(nf(100 * S.jm, 0)));
  res.push(T.ld4(S.low.map(i => [T.dmy(x[i]), HOL[x[i]] || null, nf(v[i], 2)]), nf(S.m, 2), S.eas === null ? null : nf(S.eas, 2)));
  out('ld-out', res);
}

// ================================================================= 16. Prophet
function blkProphet() {
  const Kp = $('pr-K').value, h = $('pr-h').value, C = D.prophet.cfg[Kp + h], d0 = D.prophet.d0, n = C.tr.length, x = range(0, n - 1, 1).map(i => addD(d0, i));
  const G = grid('pr-chart', [{ r: 0, c: 0, t: 'Trend (GW)', x: { type: 'date' } }, { r: 0, c: 1, t: 'Weekly component (GW)', x: { tickvals: range(0, 6, 1), ticktext: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] } },
    { r: 1, c: 0, t: 'Yearly component, shown for 2024 (GW)', x: { type: 'date', tickformat: '%b' } }, { r: 1, c: 1, t: 'Holiday effects (GW)', y: { automargin: true } }], { cols: 2, h: 600, legendRows: 2, gyPx: 72, gx: 0.12 });
  const [a, b, c, d] = G.ax, i1 = dIdx(addD(d0, n - 1)), i0 = dIdx(d0);
  const tr = [ln(x, DL.v.slice(i0, i1 + 1), a, K.teal, 'daily load', { line: { color: K.teal, width: 0.5 }, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' }),
    ln(x, C.tr, a, K.blue, 'trend g(t)', { line: { color: K.blue, width: 1.8 }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }), leg('changepoints used', { color: K.orange, dash: 'dot', width: 1 }),
    { x: range(0, 6, 1), y: C.wk, type: 'bar', marker: { color: [K.blue, K.blue, K.blue, K.blue, K.blue, K.orange, K.red] }, xaxis: b.xa, yaxis: b.ya, showlegend: false, hovertemplate: '%{y:.3f} GW<extra></extra>' },
    ln(C.yr.map((_, i) => addD('2024-01-01', i)), C.yr, c, K.green, '', { line: { color: K.green, width: 1.5 }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' })];
  C.cp.forEach(cp => G.lay.shapes.push(vl(a, cp, K.orange, 'dot', 0.6)));
  G.lay.shapes.push(hl(b, 0, K.navy, 'solid', 0.5), hl(c, 0, K.navy, 'solid', 0.5));
  const E = Object.entries(C.hol).sort((p, q) => p[1] - q[1]);
  if (E.length) { tr.push({ y: E.map(e => e[0]), x: E.map(e => e[1]), type: 'bar', orientation: 'h', marker: { color: E.map(e => (e[1] < -0.6 ? K.red : K.orange)) }, xaxis: d.xa, yaxis: d.ya, showlegend: false, hovertemplate: '%{y}: %{x:.3f}<extra></extra>' });
    G.lay.shapes.push(vl(d, 0, K.navy, 'solid', 0.5)); G.lay['yaxis' + d.k] = Object.assign(G.lay['yaxis' + d.k], { type: 'category', tickfont: { size: 9, color: K.navy } }); } else {
    tr.push({ x: [0], y: [0], xaxis: d.xa, yaxis: d.ya, mode: 'text', text: ['no holidays in this fit'], textfont: { size: 12, color: K.navy }, showlegend: false, hoverinfo: 'skip' });
    Object.assign(G.lay['xaxis' + d.k], { range: [-1, 1], showticklabels: false, showgrid: false, zeroline: false });
    Object.assign(G.lay['yaxis' + d.k], { range: [-1, 1], showticklabels: false, showgrid: false, zeroline: false }); }
  G.lay['xaxis' + d.k].automargin = true;
  plot('pr-chart', tr, G.lay);
  const ymx = C.yr.indexOf(Math.max(...C.yr)), ymn = C.yr.indexOf(Math.min(...C.yr));
  const res = [T.pr1(Kp, h === '1', Kp === '10' && h === '1'), T.pr2(nf(C.tr[0], 2), nf(C.tr[n - 1], 2), C.cp.length), T.pr3(nf(C.wk[2], 2), nf(C.wk[5], 2), nf(C.wk[6], 2)),
    T.pr4(sgn(C.yr[ymx], 2), T.dmy(addD('2024-01-01', ymx)), nf(C.yr[ymn], 2), T.dmy(addD('2024-01-01', ymn)))];
  if (E.length) res.push(T.pr5(E.slice(0, 3).map(e => [e[0], nf(e[1], 2)])));
  res.push(T.pr6(nf(1000 * C.mae, 0)));
  out('pr-out', res);
}

// ================================================================= 17. cross-validation scheme
function blkScheme() {
  const h = +$('cs-h').value, st = +$('cs-st').value, n = 6; set('cs-h', h); set('cs-st', st);
  const Lx = Math.max(26, 10 + st * (n - 1) + h + 1);
  const G = grid('cs-chart', [{ r: 0, c: 0, t: 'Time-series cross-validation: the origin moves forward, the model is refitted', xt: 'time', x: { range: [-0.5, Lx + 0.5] },
    y: { tickvals: range(0, n - 1, 1), ticktext: range(1, n, 1).map(i => `origin ${i}`), autorange: 'reversed' } }], { cols: 1, h: 330, legendRows: 1, ml: 70 });
  const a = G.ax[0], y = range(0, n - 1, 1), te = y.map(i => 10 + st * i), tr = [
    { y, x: te, base: 0, type: 'bar', orientation: 'h', width: 0.55, marker: { color: K.blue }, name: 'training sample', xaxis: a.xa, yaxis: a.ya, hovertemplate: 'training: 1-%{x}<extra></extra>' },
    { y, x: y.map(() => h), base: te, type: 'bar', orientation: 'h', width: 0.55, marker: { color: K.red }, name: `forecast window (h = ${h})`, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' },
    { y, x: te.map(v => Math.max(0, Lx - v - h)), base: te.map(v => v + h), type: 'bar', orientation: 'h', width: 0.55, marker: { color: 'rgba(181,133,63,0.25)' }, name: 'not used at this origin', xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' }];
  G.lay.barmode = 'overlay';
  plot('cs-chart', tr, G.lay);
  out('cs-out', [T.cs1(h, st, st < h, Math.ceil(h / st)), T.cs2()]);
}

// ================================================================= 18-20. cross-validation results
const CV = D.cv, NO = CV.origins.length, HZ = 14;
const actual = (o, j) => DL.v[dIdx(o) + 1 + j];
const err = (m, k, j) => actual(CV.origins[k], j) - CV.fc[m][k][j];
const holWin = k => range(1, HZ, 1).some(j => { const d = addD(CV.origins[k], j); return HOL[d] || (d.slice(5, 7) === '12' && +d.slice(8) >= 24) || (d.slice(5, 7) === '01' && +d.slice(8) <= 2); });
function blkLoadFc() {
  const k = CV.origins.indexOf($('lf-o').value), o = CV.origins[k];
  const G = grid('lf-chart', [{ r: 0, c: 0, t: `Daily load (GW): forecasts made on ${T.dmyEn(o)}, 14 days`, x: { type: 'date', tickformat: '%d %b' } }], { cols: 1, h: 420, legendRows: 2 });
  const a = G.ax[0], i0 = dIdx(o), hx = range(i0 - 27, Math.min(DL.v.length - 1, i0 + HZ), 1), fx = range(1, HZ, 1).map(j => addD(o, j));
  const tr = [ln(hx.map(i => addD(DL.d0, i)), hx.map(i => DL.v[i]), a, K.navy, 'actual load', { mode: 'lines+markers', marker: { size: 4 }, line: { color: K.navy, width: 1.6 }, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' })];
  CV.models.forEach(m => tr.push(ln(fx, CV.fc[m][k], a, MC[m], m, { line: { color: MC[m], width: 1.3, dash: m === 'Seasonal naive' ? 'dash' : 'solid' }, hovertemplate: `${m}, %{x}: %{y:.2f}<extra></extra>` })));
  hx.forEach(i => { const d = addD(DL.d0, i); if (HOL[d]) G.lay.shapes.push(rect(a, addD(d, 0) + ' 00:00', d + ' 23:59', K.orange, 0.12)); });
  G.lay.shapes.push(vl(a, o + ' 12:00', K.navy, 'dot', 0.7));
  plot('lf-chart', tr, G.lay);
  const mae = CV.models.map(m => [m, 1000 * mean(range(0, HZ - 1, 1).map(j => Math.abs(err(m, k, j))))]).sort((p, q) => p[1] - q[1]);
  const hol = [...new Set(fx.filter(d => HOL[d]).map(d => HOL[d]))];
  out('lf-out', [T.lf1(T.dmy(o), T.dmy(fx[0]), T.dmy(fx[HZ - 1]), hol, o === NB.loadfc.origin), T.lf2(mae.map(([m, v]) => [m, nf(v, 0)]))]);
}
function dmRaw(l1, l2) {
  const d = l1.map((v, i) => v - l2[i]), n = d.length, db = mean(d), v = pvar(d) / n, dm = db / Math.sqrt(v), hln = Math.sqrt((n - 1) / n) * dm;
  return { dm, hln, p: tsf2(hln, n - 1), n };
}
function cvTable(sub) {
  const sc = NB.cv.scale, R = {};
  CV.models.forEach(m => {
    const e = [].concat(...sub.map(k => range(0, HZ - 1, 1).map(j => err(m, k, j))));
    const win = sub.map(k => mean(range(0, HZ - 1, 1).map(j => err(m, k, j) ** 2)));
    const byh = range(0, HZ - 1, 1).map(j => sub.map(k => err(m, k, j)));
    R[m] = { MAE: mean(e.map(Math.abs)), RMSE: Math.sqrt(mean(e.map(v => v * v))), win, hM: byh.map(a => mean(a.map(Math.abs))), hR: byh.map(a => Math.sqrt(mean(a.map(v => v * v)))) };
    R[m].MASE = R[m].MAE / sc;
  });
  return R;
}
function blkCvRes() {
  const met = $('cr-m').value, wsel = $('cr-w').value, all = range(0, NO - 1, 1);
  const sub = wsel === 'all' ? all : all.filter(k => holWin(k) === (wsel === 'hol')), R = cvTable(sub);
  const G = grid('cr-chart', [{ r: 0, c: 0, cs: 4, t: `${met === 'MAE' ? 'Mean absolute error' : 'Root mean squared error'}, ${sub.length} origins, all horizons`, xt: `${met} (MW)`, y: { autorange: 'reversed', automargin: true } },
    { r: 0, c: 4, cs: 5, t: `${met} by forecast horizon`, xt: 'horizon h (days)', yt: `${met} (MW)` }], { cols: 9, h: 420, legendRows: 2, gx: 0.07, ml: 100 });
  const [a, b] = G.ax, M = CV.models, v = M.map(m => 1000 * R[m][met]);
  const tr = [{ y: M, x: v, type: 'bar', orientation: 'h', marker: { color: M.map(m => MC[m]) }, text: v.map(z => z.toFixed(0)), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: K.navy }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: '%{y}: %{x:.0f} MW<extra></extra>' }];
  G.lay['xaxis'].range = [0, Math.max(...v) * 1.18];
  M.forEach(m => tr.push(ln(range(1, HZ, 1), (met === 'MAE' ? R[m].hM : R[m].hR).map(z => 1000 * z), b, MC[m], m, { mode: 'lines+markers', marker: { size: 4 }, line: { color: MC[m], width: m === 'Combination' ? 1.8 : 1.2, dash: m === 'Seasonal naive' ? 'dash' : 'solid' }, hovertemplate: `${m}, h = %{x}: %{y:.0f}<extra></extra>` })));
  plot('cr-chart', tr, G.lay);
  const rank = [...M].sort((p, q) => R[p][met] - R[q][met]);
  const dm = M.filter(m => m !== 'Seasonal naive').map(m => { const t = dmRaw(R[m].win, R['Seasonal naive'].win); return [m, nf(t.hln, 2), pe(t.p)]; });
  const dd = dmRaw(R.Combination.win, R.DHR.win);
  out('cr-out', [T.cr1(wsel, sub.length), T.cr2(met, rank.map(m => [m, nf(1000 * R[m][met], 0)])), T.cr3(rank.filter(m => R[m].MASE < 1).map(m => [m, nf(R[m].MASE, 2)])), T.cr4(dm), T.cr5(nf(dd.hln, 2), pe(dd.p))]);
}
function blkComb() {
  const m1 = $('cb-1').value, m2 = $('cb-2').value, ks = range(0, NO - 1, 1), js = range(0, HZ - 1, 1);
  const act = [].concat(...ks.map(k => js.map(j => actual(CV.origins[k], j)))), f1 = [].concat(...ks.map(k => CV.fc[m1][k])), f2 = [].concat(...ks.map(k => CV.fc[m2][k]));
  const w = range(0, 1, 0.01), mae = w.map(x => 1000 * mean(act.map((y, i) => Math.abs(y - x * f1[i] - (1 - x) * f2[i])))), rm = w.map(x => 1000 * Math.sqrt(mean(act.map((y, i) => (y - x * f1[i] - (1 - x) * f2[i]) ** 2))));
  const G = grid('cb-chart', [{ r: 0, c: 0, t: `Combining ${m1} and ${m2}: error as a function of the weight`, xt: `weight w of ${m1} (1 - w on ${m2})` }], { cols: 1, h: 360, legendRows: 1 });
  const a = G.ax[0], ib = mae.indexOf(Math.min(...mae));
  plot('cb-chart', [ln(w, mae, a, K.blue, 'MAE (MW)', { line: { color: K.blue, width: 1.6 }, hovertemplate: 'w = %{x:.2f}: %{y:.1f}<extra></extra>' }), ln(w, rm, a, K.red, 'RMSE (MW)', { line: { color: K.red, width: 1.6 }, hovertemplate: 'w = %{x:.2f}: %{y:.1f}<extra></extra>' }),
    leg('equal weights', { color: K.green, dash: 'dash', width: 1 }),
    { x: [w[ib]], y: [mae[ib]], mode: 'markers', marker: { color: K.blue, size: 8, symbol: 'diamond' }, xaxis: a.xa, yaxis: a.ya, name: 'lowest MAE', hovertemplate: 'w = %{x:.2f}: %{y:.1f}<extra></extra>' }],
  Object.assign(G.lay, { shapes: [vl(a, 0.5, K.green, 'dash', 1)] }));
  const e1 = act.map((y, i) => y - f1[i]), e2 = act.map((y, i) => y - f2[i]), rho = corr(e1, e2), s1 = 1000 * Math.sqrt(pvar(e1)), s2 = 1000 * Math.sqrt(pvar(e2));
  const ws = (s2 * s2 - rho * s1 * s2) / (s1 * s1 + s2 * s2 - 2 * rho * s1 * s2);
  out('cb-out', [T.cb1(m1, m2, nf(mae[100], 0), nf(mae[0], 0), nf(mae[50], 0), nf(rm[100], 0), nf(rm[0], 0), nf(rm[50], 0)), T.cb2(nf(w[ib], 2), nf(mae[ib], 0)),
    T.cb3(nf(rho, 2), nf(s1, 0), nf(s2, 0), nf(ws, 2))]);
}

// ================================================================= tables and worked examples (blocks of their own frames)
const X = () => T.x;
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}${v.cs ? ` colspan="${v.cs}"` : ''}>${v.v}</${tag}>`; };
  const hd = (Array.isArray(head[0]) ? head : [head]).map(r => '<tr>' + r.map(c => cell(c, 'th')).join('') + '</tr>').join('');
  el.innerHTML = (o && o.cap ? `<p class="c4cap">${o.cap}</p>` : '') + '<table><thead>' + hd + '</thead><tbody>'
    + rows.map(r => `<tr${r.hi ? ' class="hi"' : ''}>` + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c4cap">${o.note}</p>` : '');
}
const nz = (v, d = 2) => nf(Math.abs(v) < 0.5 * 10 ** -d ? 0 : v, d);                // no '−0.00'
const num = (x, d, bold) => ({ v: x === null || x === undefined || !isFinite(x) ? '—' : bold ? `<b>${nz(x, d)}</b>` : nz(x, d), cls: 'r' });
const txt = (v, bold) => ({ v: bold ? `<b>${v}</b>` : v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const hiRow = (r, on) => { if (on) r.hi = true; return r; };
const pn = (x, d) => (x < 0 ? `(${nz(x, d)})` : nz(x, d));                       // a factor: negative numbers in parentheses
const fp = p => (p < 0.001 ? '< ' + nf(0.001, 3) : nf(p, 3));
const sel = id => $(id).value;
function tq(df, a = 0.975) { let lo = 0, hi = 50; for (let i = 0; i < 80; i++) { const m = (lo + hi) / 2; if (tsf2(m, df) > 2 * (1 - a)) lo = m; else hi = m; } return (lo + hi) / 2; }
function ols(cols, y) {           // OLS of y on the columns: coefficients, standard errors, t-ratios, residuals, the inverse of X'X (columns asked)
  const n = y.length, k = cols.length, G = cols.map(a => cols.map(b => { let s = 0; for (let t = 0; t < n; t++) s += a[t] * b[t]; return s; }));
  const gy = cols.map(a => { let s = 0; for (let t = 0; t < n; t++) s += a[t] * y[t]; return s; }), b = solve(G, gy);
  const e = y.map((v, t) => v - cols.reduce((s, c, j) => s + b[j] * c[t], 0)), sse = sum(e.map(v => v * v)), s2 = sse / (n - k);
  const inv = range(0, k - 1, 1).map(j => solve(G, range(0, k - 1, 1).map(i => (i === j ? 1 : 0))));
  const se = inv.map((c, j) => Math.sqrt(s2 * c[j]));
  return { b, se, t: b.map((v, j) => v / se[j]), e, s2, sse, n, k, inv };
}
const MONEN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// ---------------------------------------------------------------- The seasonal difference: (1 - L^4) on 100 ln GDP
function blkXSdiff() {
  const t = +$('xs-q').value, y = LY; set('xs-q', T.q(qlab(t)));
  const d4 = y[t] - y[t - 4], d1 = y[t] - y[t - 1], dd = y[t] - y[t - 1] - y[t - 4] + y[t - 5], d4p = y[t - 1] - y[t - 5], d1p = y[t - 4] - y[t - 5];
  const q = i => T.q(qlab(i));
  steps('xs-ws', [X().s1(q(t), nf(GDP[t], 3), nz(y[t], 3), q(t - 4), nf(GDP[t - 4], 3), nz(y[t - 4], 3)),
    X().s2(nz(y[t], 3), nz(y[t - 4], 3), nz(d4, 3), q(t), q(t - 4)),
    X().s3(nz(y[t], 3), nz(y[t - 1], 3), nz(d1, 3)),
    X().s4(nz(y[t], 3), nz(y[t - 1], 3), nz(y[t - 4], 3), nz(y[t - 5], 3), nz(dd, 3)),
    X().s5(nz(d4, 3), nz(d4p, 3), nz(d4 - d4p, 3), nz(d1, 3), nz(d1p, 3), nz(d1 - d1p, 3))]);
  const rows = range(Math.max(5, t - 7), t, 1).map(i => hiRow([q(i), num(GDP[i], 2), num(y[i], 2), num(y[i] - y[i - 1], 2), num(y[i] - y[i - 4], 2), num(y[i] - y[i - 1] - y[i - 4] + y[i - 5], 2)], i === t));
  table('xs-tb', [X().quarter, X().yBn, 'y<sub>t</sub> = 100 ln Y<sub>t</sub>', 'Δy<sub>t</sub>', 'Δ<sub>4</sub>y<sub>t</sub>', 'ΔΔ<sub>4</sub>y<sub>t</sub>'], rows);
  out('xs-out', [X().s0(q(t), nf(d1, 1), nf(d4, 2), nf(dd, 2))]);
  return { d4, d1, dd };
}

// ---------------------------------------------------------------- Interpreting the seasonal roots: z_k = exp(2 pi i k / s)
function blkXRoots() {
  const s = +$('xr-s').value, kEl = $('xr-k'); kEl.max = s - 1; if (+kEl.value > s - 1) kEl.value = 1; const k = +kEl.value; set('xr-k', k);
  const a = Math.cos(2 * Math.PI * k / s), b = Math.sin(2 * Math.PI * k / s), unit = X().unit[s] || X().unit0;
  const fac = kk => { if (kk === 0) return '1 − z'; if (2 * kk === s) return '1 + z'; const c = 2 * Math.cos(2 * Math.PI * kk / s), ac = Math.abs(c);
    return ac < 1e-9 ? '1 + z²' : `1 ${c >= 0 ? '−' : '+'} ${Math.abs(ac - 1) < 1e-9 ? '' : nz(ac, 3)}z + z²`; };
  const zc = (x, y) => `${nz(x, 3)} ${y < 0 ? '−' : '+'} ${nz(Math.abs(y), 3)}i`;
  steps('xr-ws', [X().r1(k, s, nz(2 * Math.PI * k / s, 4), zc(a, b)),
    X().r2(s, k, nz(Math.hypot(a, b), 3)),
    k === 0 ? X().r3a(s) : X().r3(k, s, nz(k / s, 4), nz(s / k, 2), unit),
    k === 0 || 2 * k === s ? X().r4a(fac(k), k === 0 ? 1 : -1) : X().r4(s - k, zc(a, -b), fac(k)),
    X().r5(s, range(0, Math.floor(s / 2), 1).map(fac).map(f => `(${f})`).join(''))]);
  const rows = range(0, s - 1, 1).map(kk => { const x = Math.cos(2 * Math.PI * kk / s), y = Math.sin(2 * Math.PI * kk / s);
    return hiRow([String(kk), txt(zc(x, y)), num(kk / s, 4), kk === 0 ? txt('∞') : num(s / kk, 2), kk <= s / 2 ? txt(fac(kk)) : txt(X().same(s - kk))], kk === k); });
  table('xr-tb', ['k', 'z<sub>k</sub>', X().freq, X().period(unit), X().factor], rows);
  out('xr-out', [X().r0(s, s % 2 ? 1 : 2, Math.floor((s - 1) / 2))]);
}

// ---------------------------------------------------------------- HEGY: regression, filters, critical values by simulation
const HG_END = range(2008, 2025, 1).map(y => 4 * (y - 2000) + 3).concat([LY.length - 1]);
const lnG = D.sa.gdp.map(v => Math.log(v));
function hegyFit(T0, k, trend) {
  const y = lnG.slice(0, T0), n = y.length, L = (x, j) => x.map((_, t) => (t - j >= 0 ? x[t - j] : NaN));
  const y1 = y.map((v, t) => (t >= 3 ? v + y[t - 1] + y[t - 2] + y[t - 3] : NaN)), y2 = y.map((v, t) => (t >= 3 ? -(v - y[t - 1] + y[t - 2] - y[t - 3]) : NaN));
  const y3 = y.map((v, t) => (t >= 2 ? -(v - y[t - 2]) : NaN)), d4 = y.map((v, t) => (t >= 4 ? v - y[t - 4] : NaN));
  const C = [L(y1, 1), L(y2, 1), L(y3, 2), L(y3, 1), y.map(() => 1)].concat([1, 2, 3].map(j => y.map((_, t) => (t % 4 === j ? 1 : 0))));
  if (trend) C.push(y.map((_, t) => t));
  for (let j = 1; j <= k; j++) C.push(L(d4, j));
  const ok = range(0, n - 1, 1).filter(t => !isNaN(d4[t]) && C.every(c => !isNaN(c[t])));
  const R = ols(C.map(c => ok.map(t => c[t])), ok.map(t => d4[t])), m = ok.length;
  const V = [[R.inv[2][2], R.inv[2][3]], [R.inv[3][2], R.inv[3][3]]].map(r => r.map(v => v * R.s2)), bb = [R.b[2], R.b[3]];
  const Vi = solve(V, bb), F = (bb[0] * Vi[0] + bb[1] * Vi[1]) / 2;
  const aic = m * Math.log(2 * Math.PI) + m * Math.log(R.sse / m) + m + 2 * C.length;
  return { R, F, n: m, aic, y, y1, y2, y3, d4, T0 };
}
function hegy(T0, kSel, trend) {
  let k = +kSel; if (kSel === 'AIC') { const a = range(0, 4, 1).map(j => hegyFit(T0, j, trend).aic); k = a.indexOf(Math.min(...a)); }
  const h = hegyFit(T0, k, trend), cv = D.hegy_cv[`${h.n}${trend ? 't' : 'c'}`];
  return Object.assign(h, { k, cv });
}
function blkXHegy() {
  const e = HG_END[+$('hg-end').value], kSel = sel('hg-k'), tr = sel('hg-det') === 't'; set('hg-end', T.q(qlab(e)));
  const H = hegy(e + 1, kSel, tr), t = e, yv = H.y, R = H.R;
  steps('hg-ws', [X().h1(T.q(qlab(t)), nz(yv[t], 4), nz(yv[t - 1], 4), nz(yv[t - 2], 4), nz(yv[t - 3], 4), nz(H.y1[t], 4)),
    X().h2(nz(yv[t], 4), pn(yv[t - 1], 4), pn(yv[t - 2], 4), pn(yv[t - 3], 4), nz(H.y2[t], 4)),
    X().h3(nz(yv[t], 4), nz(yv[t - 2], 4), nz(H.y3[t], 4), nz(H.d4[t], 4)),
    X().h4(H.n, H.k, kSel === 'AIC', tr)]);
  const lab = ['π<sub>1</sub> (y<sub>1,t−1</sub>)', 'π<sub>2</sub> (y<sub>2,t−1</sub>)', 'π<sub>3</sub> (y<sub>3,t−2</sub>)', 'π<sub>4</sub> (y<sub>3,t−1</sub>)'];
  table('hg-tb', [X().coef, X().est, 'SE', 't'], lab.map((l, j) => [l, num(R.b[j], 4), num(R.se[j], 4), num(R.t[j], 2)]), { cap: X().hCap1 });
  const dec = (rej) => (rej ? X().rej : X().norej);
  const st = [[`t(π<sub>1</sub>), ${X().root} 1`, R.t[0], H.cv[0], R.t[0] < H.cv[0]], [`t(π<sub>2</sub>), ${X().root} −1`, R.t[1], H.cv[1], R.t[1] < H.cv[1]],
    [`F(π<sub>3</sub>, π<sub>4</sub>), ${X().roots} ±i`, H.F, H.cv[2], H.F > H.cv[2]]];
  table('hg-tb2', [X().test, X().stat, X().cv5, X().decision], st.map(r => [r[0], num(r[1], 2), num(r[2], 2), txt(dec(r[3]))]), { cap: X().hCap2(T.q(qlab(0)), T.q(qlab(e)), H.n) });
  const G = grid('hg-chart', [{ r: 0, c: 0, t: 't(π1) and t(π2) by sample end (start 2000Q1)', xt: 'sample end', x: { type: 'category', tickvals: ['2008Q4', '2011Q4', '2014Q4', '2017Q4', '2020Q4', '2023Q4', '2026Q2'], ticktext: ['2008', '2011', '2014', '2017', '2020', '2023', '2026Q2'] } }, { r: 0, c: 1, t: 'F(π3, π4) by sample end', xt: 'sample end', x: { type: 'category', tickvals: ['2008Q4', '2011Q4', '2014Q4', '2017Q4', '2020Q4', '2023Q4', '2026Q2'], ticktext: ['2008', '2011', '2014', '2017', '2020', '2023', '2026Q2'] } }],
    { cols: 2, h: 360, legendRows: 3, gx: 0.08 });
  const [a, b] = G.ax, xl = HG_END.map(qlab), xs = xl, all = HG_END.map(i => hegy(i + 1, kSel, tr));
  const trs = [
    ln(xs, all.map(h => h.R.t[0]), a, K.blue, 't(π1), root 1', { mode: 'lines+markers', marker: { size: 5 }, customdata: xl, hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' }),
    ln(xs, all.map(h => h.cv[0]), a, K.blue, '5% critical value, t(π1)', { line: { color: K.blue, dash: 'dash', width: 1 }, hoverinfo: 'skip' }),
    ln(xs, all.map(h => h.R.t[1]), a, K.orange, 't(π2), root −1', { mode: 'lines+markers', marker: { size: 5 }, line: { color: K.orange, width: 1.3 }, customdata: xl, hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' }),
    ln(xs, all.map(h => h.cv[1]), a, K.orange, '5% critical value, t(π2)', { line: { color: K.orange, dash: 'dash', width: 1 }, hoverinfo: 'skip' }),
    ln(xs, all.map(h => h.F), b, K.green, 'F(π3, π4), roots ±i', { mode: 'lines+markers', marker: { size: 5 }, line: { color: K.green, width: 1.3 }, customdata: xl, hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' }),
    ln(xs, all.map(h => h.cv[2]), b, K.green, '5% critical value, F', { line: { color: K.green, dash: 'dash', width: 1 }, hoverinfo: 'skip' })];
  const xe = qlab(e); G.lay.shapes.push(vl(a, xe, K.navy, 'dot', 0.8), vl(b, xe, K.navy, 'dot', 0.8));
  const mk = { size: 12, color: 'rgba(0,0,0,0)', line: { color: K.navy, width: 2 } };
  trs.push({ x: [xe, xe], y: [R.t[0], R.t[1]], mode: 'markers', marker: mk, xaxis: a.xa, yaxis: a.ya, name: `sample end ${xe}`, hovertemplate: `${xe}: %{y:.2f}<extra></extra>` },
    { x: [xe], y: [H.F], mode: 'markers', marker: mk, xaxis: b.xa, yaxis: b.ya, showlegend: false, hovertemplate: `${xe}: %{y:.2f}<extra></extra>` });
  plot('hg-chart', trs, G.lay);
  const nrej = st.filter(r => r[3]).length;
  out('hg-out', [X().h0(T.q(qlab(e)), H.n, H.k, nf(R.t[0], 2), nf(R.t[1], 2), nf(H.F, 2), nrej), X().hNote()]);
  return H;
}

// ---------------------------------------------------------------- Seasonal unit-root tests on Romanian data: OCSB, Canova-Hansen, KPSS
const ST_SER = ['GDP', 'Retail trade', 'Industrial production', 'Tourism nights', 'HICP'];
function blkXStests() {
  const yr = $('su-end').value; set('su-end', yr);
  const rows = ST_SER.map(s => { const r = D.stests[s][yr];
    return hiRow([X().ser[s], String(s === 'GDP' ? 4 : 12), String(r.n), num(r.ocsb[0], 2), num(r.ocsb[1], 2), txt(String(r.ocsb[2]), true), num(r.ch[0], 3), num(r.ch[1], 3), txt(String(r.ch[2]), true), txt(String(r.d), true)], s === 'GDP'); });
  table('su-tb', [[X().ser0, 'm', 'n', { v: 'OCSB', cs: 3 }, { v: 'Canova–Hansen', cs: 3 }, 'KPSS'],
    ['', '', '', X().stat, X().cv5, 'D', X().stat, X().cv5, 'D', 'd']], rows, { cap: X().suCap(yr), note: X().suNote });
  const g = D.stests.GDP[yr];
  out('su-out', [X().su0(yr, g.ocsb[2], g.ch[2], g.d, ST_SER.filter(s => s !== 'GDP' && D.stests[s][yr].ocsb[2] === 0 && D.stests[s][yr].ch[2] === 0).length)]);
}

// ---------------------------------------------------------------- Worked example: the airline model
function blkXAir() {
  const th = +$('xa-th').value, Th = +$('xa-Th').value; set('xa-th', nf(th, 2)); set('xa-Th', nf(Th, 2));
  const g0 = (1 + th * th) * (1 + Th * Th), r1 = th / (1 + th * th), r12 = Th / (1 + Th * Th), r11 = r1 * r12;
  steps('xa-ws', [X().a1(pn(th, 2), pn(Th, 2), nz(th * Th, 4)),
    X().a2(nz(th, 2), nz(Th, 2), nz(g0, 4)),
    X().a3(pn(th, 2), nz(1 + th * th, 4), nz(r1, 3), pn(Th, 2), nz(1 + Th * Th, 4), nz(r12, 3), pn(r1, 3), pn(r12, 3), nz(r11, 3))]);
  const fa = NB.airfc, a1 = fa.full_theta / (1 + fa.full_theta ** 2), a12 = fa.full_Theta / (1 + fa.full_Theta ** 2), A = NB.air;
  table('xa-tb', [X().lag, X().maw, X().rhoTh, X().rhoFit, X().rhoSample],
    [['1', num(th, 2), num(r1, 3), num(a1, 3), num(A.r1, 3)], ['11', num(0, 2), num(r11, 3), num(a1 * a12, 3), num(A.r11, 3)], ['12', num(Th, 2), num(r12, 3), num(a12, 3), num(A.r12, 3)],
      ['13', num(th * Th, 4), num(r11, 3), num(a1 * a12, 3), num(A.r13, 3)]], { cap: X().aCap(nf(fa.full_theta, 3), nf(fa.full_Theta, 3), A.Tw, nf(A.band, 2)) });
  out('xa-out', [X().a0(nz(g0, 4), nf(r1, 3), nf(r12, 3), nf(r11, 3))]);
  return { g0, r1, r12, r11 };
}

// ---------------------------------------------------------------- Forecasting with SARIMA: the airline recursion

function airRec(yr) {
  const a = D.air2[yr], T0 = (yr - 1949) * 12 + 11, E = a.e, yy = AIRL.slice(0, T0 + 1), rs = t => { const j = t - (T0 - 12); return j >= 0 && j < 13 ? E[j] : 0; }, P = [];
  for (let h = 1; h <= 24; h++) { const t = T0 + h, m = a.th * rs(t - 1) + a.Th * rs(t - 12) + a.th * a.Th * rs(t - 13);
    yy[t] = yy[t - 1] + yy[t - 12] - yy[t - 13] + m; P.push({ h, t, a: yy[t - 1], b: yy[t - 12], c: yy[t - 13], m, e1: rs(t - 1), e12: rs(t - 12), e13: rs(t - 13), f: yy[t] }); }
  return { a, T0, P };
}
function blkXFc() {
  const yr = +$('xf-o').value, h = +$('xf-h').value; set('xf-o', T.dec(yr)); set('xf-h', h);
  const { a, T0, P } = airRec(yr), p = P[h - 1], mo = t => `${MONEN[t % 12]} ${1949 + Math.floor(t / 12)}`, obs = t => t <= T0;
  const tag = t => (obs(t) ? X().obsT : X().fcT);
  const lst = [X().f1(yr, nf(a.th, 3), nf(a.Th, 3), nf(a.th * a.Th, 3)),
    X().f2(h, mo(p.t), mo(p.t - 1), tag(p.t - 1), nz(p.a, 4), mo(p.t - 12), tag(p.t - 12), nz(p.b, 4), mo(p.t - 13), tag(p.t - 13), nz(p.c, 4)),
    h <= 13 ? X().f3(nz(a.th, 3), pn(p.e1, 4), nz(a.Th, 3), pn(p.e12, 4), nz(a.th * a.Th, 3), pn(p.e13, 4), nz(p.m, 4)) : X().f3b(h),
    X().f4(nz(p.a, 4), pn(p.b, 4), pn(p.c, 4), pn(p.m, 4), nz(p.f, 4), nf(Math.exp(p.f), 1), nf(D.air.org[yr].f[h - 1], 1), nf(D.air.y[p.t], 0))];
  steps('xf-ws', lst);
  const hs = range(1, 13, 1).concat(h > 13 ? [h] : []);
  table('xf-tb', [X().hh, X().month, 'ŷ<sub>T+h−1</sub>', 'ŷ<sub>T+h−12</sub>', 'ŷ<sub>T+h−13</sub>', X().maPart, 'ŷ<sub>T+h</sub> (ln)', X().fcLev, X().actual],
    hs.map(k => { const q = P[k - 1]; return hiRow([String(k), mo(q.t), num(q.a, 4), num(q.b, 4), num(q.c, 4), num(q.m, 4), num(q.f, 4), num(Math.exp(q.f), 1), num(D.air.y[q.t], 0)], k === h); }));
  out('xf-out', [X().f0(mo(p.t), nf(Math.exp(p.f), 1), nf(D.air.y[p.t], 0), nf(100 * (D.air.y[p.t] - Math.exp(p.f)) / D.air.y[p.t], 1))]);
  return P;
}

// ---------------------------------------------------------------- Interpreting the airline forecasts: estimates by origin
const AIR_O = range(1952, 1958, 1);
function airRow(yr) {
  const a = D.air2[yr], o = D.air.org[yr];
  if (yr === 'full') return { a, n: a.n };
  const i0 = (yr - 1949) * 12 + 12, act = D.air.y.slice(i0), sn = act.map((_, h) => D.air.y[i0 - 12 + (h % 12)]);
  const mape = f => 100 * mean(act.map((v, h) => Math.abs(v - f[h]) / v)), L = act.length - 1;
  return { a, n: a.n, mA: mape(o.f), mS: mape(sn), f: o.f[L], lo: o.lo[L], hi: o.hi[L], act: act[L] };
}
function blkXAirTab() {
  const yr = +$('at-o').value; set('at-o', T.dec(yr));
  const R = AIR_O.map(airRow), F = airRow('full'), rw = (lab, r, hi) => hiRow([lab, String(r.n), num(r.a.th, 3), num(r.a.th_se, 3), num(r.a.Th, 3), num(r.a.Th_se, 3), num(100 * r.a.s, 2),
    num(r.a.lb[0], 1), num(r.a.lb[1], 3), num(r.mA, 1), num(r.mS, 1), r.f ? txt(`${nf(r.f, 0)} [${nf(r.lo, 0)}; ${nf(r.hi, 0)}]`) : txt('—'), r.act ? num(r.act, 0) : txt('—')], hi);
  table('at-tb', [X().sample, 'n', 'θ̂', 'SE', 'Θ̂', 'SE', 'σ̂ (%)', 'Q*(24)', 'p', X().mapeA, X().mapeS, X().dec60, X().actual],
    R.map((r, i) => rw(`1949–${AIR_O[i]}`, r, AIR_O[i] === yr)).concat([rw('1949–1960', F, false)]), { cap: X().atCap });
  const G = grid('at-chart', [{ r: 0, c: 0, t: 'Estimates by end of the estimation sample, ±1.96 SE', xt: 'last year of the estimation sample', y: { range: [-1.7, 0.5] } }, { r: 0, c: 1, t: 'MAPE of the forecasts to December 1960 (%)', xt: 'forecast origin: December of' }],
    { cols: 2, h: 360, legendRows: 2, gx: 0.08 });
  const [a, b] = G.ax, xs = AIR_O, er = (v, s) => ({ type: 'data', array: s.map(x => 1.96 * x), color: undefined, thickness: 1.2, width: 3 });
  const th = R.map(r => r.a.th), Th = R.map(r => r.a.Th);
  const mA = R.map(r => r.mA), mS = R.map(r => r.mS), lab = (v, x) => (x === yr ? v.toFixed(1) : '');
  plot('at-chart', [ln(xs, th, a, K.blue, 'θ̂ (regular MA)', { mode: 'markers+lines', error_y: Object.assign(er(th, R.map(r => r.a.th_se)), { color: K.blue }), hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }),
    ln(xs.map(x => x + 0.12), Th, a, K.red, 'Θ̂ (seasonal MA)', { mode: 'markers+lines', line: { color: K.red, width: 1.3 }, error_y: Object.assign(er(Th, R.map(r => r.a.Th_se)), { color: K.red }), hovertemplate: '%{x:.0f}: %{y:.3f}<extra></extra>' }),
    { x: [yr, yr + 0.12], y: [th[yr - 1952], Th[yr - 1952]], mode: 'markers', marker: { size: 13, color: 'rgba(0,0,0,0)', line: { color: K.navy, width: 2 } }, xaxis: a.xa, yaxis: a.ya, name: `sample 1949–${yr}`, hovertemplate: '%{y:.3f}<extra></extra>' },
    { x: xs, y: mA, type: 'bar', name: 'airline model', marker: { color: K.red }, text: mA.map((v, i) => lab(v, xs[i])), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: K.navy }, xaxis: b.xa, yaxis: b.ya, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' },
    { x: xs, y: mS, type: 'bar', name: 'seasonal naive', marker: { color: K.amber }, text: mS.map((v, i) => lab(v, xs[i])), textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: K.navy }, xaxis: b.xa, yaxis: b.ya, hovertemplate: '%{x}: %{y:.1f}%<extra></extra>' }],
  Object.assign(G.lay, { barmode: 'group', shapes: [hl(a, 0, K.navy, 'solid', 0.5), vl(a, yr, K.navy, 'dot', 0.8)] }));
  const r = R[yr - 1952];
  out('at-out', [X().at0(yr, nf(r.a.th, 3), nf(r.a.Th, 3), nf(r.a.lb[1], 2), nf(r.mA, 1), nf(r.mS, 1)), X().at1(nf(F.a.th, 3), nf(F.a.Th, 3), nf(F.a.lb[1], 2))]);
}

// ---------------------------------------------------------------- SARIMA models for Romanian GDP: AICc, BIC, Ljung-Box, estimates
const GM_ORDER = ['(0,1,1)(0,1,1)4', '(0,1,0)(0,1,1)4', '(1,1,0)(0,1,1)4', '(1,1,1)(0,1,1)4', '(0,1,2)(0,1,1)4', '(0,1,1)(1,1,0)4', '(0,1,1)(1,1,1)4', '(0,1,1)(0,1,0)4',
  '(2,1,0)(0,1,1)4', '(1,1,0)(1,1,0)4'];
const sub4 = m => m.replace(/\)4$/, ')₄');
const PARN = { 'ar.L1': 'φ₁', 'ar.L2': 'φ₂', 'ma.L1': 'θ₁', 'ma.L2': 'θ₂', 'ar.S.L4': 'Φ', 'ma.S.L4': 'Θ' };
function blkXGdpm() {
  const m = sel('gm-m'), lm = +$('gm-L').value; set('gm-L', lm);
  const rowN = mm => NB.gdpm.rows.find(r => r.model === mm), nE = 101;
  const S = GM_ORDER.map(mm => { const M = D.gdp.models[mm], q = ljungBox(M.e, lm, M.k), Q = q.q !== undefined ? q.q : q.Q; return { mm, M, ll: rowN(mm).loglik, Q, p: lm > M.k ? chi2sf(Q, lm - M.k) : NaN }; });
  const amin = Math.min(...S.map(s => s.M.aicc)), bmin = Math.min(...S.map(s => s.M.bic));
  table('gm-tb', [X().model, X().kPar, 'ln L', 'AICc', 'BIC', `LB(${lm}), p`, 'σ̂ (%)'],
    S.map((s, i) => hiRow([sub4(s.mm) + (i >= 8 ? ' *' : ''), String(s.M.k), num(s.ll, 2), num(s.M.aicc, 1, s.M.aicc === amin), num(s.M.bic, 1, s.M.bic === bmin), num(s.p, 3), num(s.M.sigma, 2)], s.mm === m)),
    { cap: X().gmCap, note: X().gmNote(lm) });
  const s = S.find(z => z.mm === m), k = s.M.k + 1, aic = -2 * s.ll + 2 * k, corr = 2 * k * (k + 1) / (nE - k - 1), bic = -2 * s.ll + k * Math.log(nE);
  steps('gm-ws', [X().g1(sub4(m), s.M.k, k, nE), X().g2(pn(s.ll, 3), k, nz(aic, 3)), X().g3(nz(aic, 3), k, k + 1, nE - k - 1, nz(corr, 3), nz(aic + corr, 3)),
    X().g4(pn(s.ll, 3), k, nz(Math.log(nE), 4), nz(bic, 3))]);
  table('gm-tb2', [X().coef, X().est, 'SE', 't'], Object.entries(s.M.par).map(([nm, v]) => [PARN[nm] || nm, num(v[0], 3), num(v[1], 3), num(v[0] / v[1], 2)]), { cap: X().gmCap2(sub4(m)) });
  const G = grid('gm-chart', [{ r: 0, c: 0, t: `AICc and BIC (lower is better); ✗: Ljung–Box(${lm}) p < 0.05`, xt: 'criterion', y: { autorange: 'reversed', automargin: true } }], { cols: 1, h: 430, legendRows: 1, ml: 130 });
  const a = G.ax[0], lab = S.map(z => sub4(z.mm) + (z.p < 0.05 ? ' ✗' : ''));   // ✗: Ljung-Box with m lags rejects white-noise residuals at 5%
  G.lay.xaxis.range = [510, Math.max(...S.map(z => z.M.bic)) + 9];
  const tl = k => S.map(z => (z.mm === m ? z.M[k].toFixed(1) : '')), tp = { textposition: 'outside', cliponaxis: false, textfont: { size: 10, color: K.red } };
  plot('gm-chart', [{ y: lab, x: S.map(z => z.M.aicc), type: 'bar', orientation: 'h', name: 'AICc', marker: { color: S.map(z => (z.mm === m ? K.red : K.blue)) }, xaxis: a.xa, yaxis: a.ya, text: tl('aicc'), ...tp, hovertemplate: '%{y}: AICc %{x:.1f}<extra></extra>' },
    { y: lab, x: S.map(z => z.M.bic), type: 'bar', orientation: 'h', name: 'BIC', marker: { color: S.map(z => (z.mm === m ? 'rgba(205,0,0,0.45)' : 'rgba(23,162,184,0.8)')) }, xaxis: a.xa, yaxis: a.ya, text: tl('bic'), ...tp, hovertemplate: '%{y}: BIC %{x:.1f}<extra></extra>' }],
  Object.assign(G.lay, { barmode: 'group' }));
  out('gm-out', [X().gm0(sub4(m), nf(s.M.aicc, 1), nf(s.M.bic, 1), nf(aic + corr - amin, 1), nf(s.M.bic - bmin, 1), lm, isFinite(s.p) ? pe(s.p) : '—')]);
  return { S, aicc: aic + corr, bic };
}

// ---------------------------------------------------------------- Interpreting the GDP forecasts: the forecast table
function blkXGdpfcTab() {
  const Th = +$('gt-Th').value, lv = +sel('gt-lv'), z = normPpf(0.5 + lv / 2); set('gt-Th', nf(Th, 3));
  const F = gdpFc(Th), n = LY.length, lev = v => Math.exp(v / 100) / 1000;
  const rows = F.f.map((v, h) => { const t = n + h, prev = t - 4 < n ? LY[t - 4] : F.f[h - 4];
    return [T.q(qlab(t)), num(lev(v), 1), num(lev(v - z * F.se[h]), 1), num(lev(v + z * F.se[h]), 1), num(100 * (Math.exp(z * F.se[h] / 100) - 1), 1), num(v - prev, 2), num(lev(prev), 1)]; });
  table('gt-tb', [X().quarter, X().fcBn, X().lower(Math.round(lv * 100)), X().upper(Math.round(lv * 100)), X().halfPct, X().yoy, X().lastYear], rows, { cap: X().gtCap(nf(Th, 3), nf(F.sig, 2)) });
  const f = F.f.map(lev);
  out('gt-out', [X().gt0(nf(GDP[n - 1], 1), f.slice(0, 4).map(v => nf(v, 1)), Math.round(lv * 100), nf(lev(F.f[0] - z * F.se[0]), 1), nf(lev(F.f[0] + z * F.se[0]), 1), nf(lev(F.f[7] - z * F.se[7]), 1), nf(lev(F.f[7] + z * F.se[7]), 1)),
    X().gt1(nf(1 + Th, 2), nf(Th, 3))]);
}

// ---------------------------------------------------------------- Interpreting the adjusted series: factors, annual totals, quarter-on-quarter growth
function blkXSa() {
  let y0 = +$('st-y0').value, y1 = +$('st-y1').value; if (y1 < y0) { y1 = y0; $('st-y1').value = y0; }
  const q = +$('st-q').value; set('st-y0', y0); set('st-y1', y1); set('st-q', T.q(qlab(q)));
  const N0 = 40, gn = GDP.slice(N0), fac = range(0, 3, 1).map(k => { const v = SCA.map((s, i) => [2010 + Math.floor(i / 4), i % 4, gn[i] / s]).filter(r => r[1] === k && r[0] >= y0 && r[0] <= y1).map(r => r[2]); return { m: mean(v), lo: Math.min(...v), hi: Math.max(...v), n: v.length }; });
  table('st-tb', [X().quarter0, X().facMean(y0, y1), X().facMin, X().facMax], fac.map((f, k) => [`Q${k + 1}`, num(f.m, 3), num(f.lo, 3), num(f.hi, 3)]), { cap: X().stCap });
  const yrs = range(y0, Math.min(y1, 2025), 1), tot = yrs.map(y => { const i = 4 * (y - 2010); return [sum(gn.slice(i, i + 4)), sum(SCA.slice(i, i + 4))]; });
  table('st-tb2', [X().year, 'NSA', 'SCA', X().diffPct], yrs.map((y, i) => [String(y), num(tot[i][0], 2), num(tot[i][1], 2), num(100 * (tot[i][1] / tot[i][0] - 1), 2)]), { cap: X().stCap2 });
  const R = D.sa.ret, RS = D.sa.ret_sca, rf = mm => { const v = R.v.map((x, i) => [yearOf(R.m0, i), monOf(R.m0, i), x / RS.v[i]]).filter(r => r[1] === mm && r[0] >= Math.max(y0, 2015) && r[0] <= y1).map(r => r[2]); return v.length ? mean(v) : NaN; };
  const i = q - N0, L = v => 100 * Math.log(v), dn = L(gn[i]) - L(gn[i - 1]), ds = L(SCA[i]) - L(SCA[i - 1]);
  steps('st-ws', [X().q1(T.q(qlab(q)), T.q(qlab(q - 1)), nf(gn[i], 3), nf(gn[i - 1], 3), nz(dn, 2)), X().q2(nf(SCA[i], 3), nf(SCA[i - 1], 3), nz(ds, 2)),
    X().q3(nz(dn - ds, 2), nf(gn[i] / SCA[i], 3), nf(gn[i - 1] / SCA[i - 1], 3), nz(L(gn[i] / SCA[i]) - L(gn[i - 1] / SCA[i - 1]), 2))]);
  out('st-out', [X().st0(y0, y1, fac.map(f => nf(f.m, 2)))].concat(y1 >= 2015 ? [X().st1(Math.max(y0, 2015), y1, nf(rf(12), 2), nf(rf(1), 2))] : []));
  return { fac, dn, ds };
}

// ---------------------------------------------------------------- Seasonal dummies and Fourier terms: one sine-cosine pair
const DHRK = range(1, 4, 1).map(k => [NB.dhr.coef[`sin${k}_365.25`], NB.dhr.coef[`cos${k}_365.25`]]);
function blkXFour() {
  const cs = sel('xo-c'), tr = cs === 'tour', m = tr ? 12 : 365.25, Kmax = tr ? 3 : 4, kEl = $('xo-k'), tEl = $('xo-t');
  kEl.max = Kmax; if (+kEl.value > Kmax) kEl.value = 1; tEl.max = tr ? 12 : 365; if (+tEl.value > +tEl.max) tEl.value = tr ? 8 : 1;
  const k = +kEl.value, tt = +tEl.value, F = tr ? fourierFit(3) : null, co = tr ? range(1, 3, 1).map(j => [F.b[2 * j - 2], F.b[2 * j - 1]]) : DHRK;
  const date = tr ? null : addD('2025-01-01', tt - 1), t = tr ? tt : (d2t(date) - d2t('2000-01-01')) / DAY;
  set('xo-k', k); set('xo-t', tr ? X().monName[tt - 1] : T.dmy(date));
  const term = j => { const ang = 2 * Math.PI * j * t / m, s = Math.sin(ang), c = Math.cos(ang); return { ang, s, c, v: co[j - 1][0] * s + co[j - 1][1] * c, amp: Math.hypot(...co[j - 1]) }; };
  const P = range(1, Kmax, 1).map(term), p = P[k - 1], tot = sum(P.map(z => z.v));
  steps('xo-ws', [tr ? X().o1t(tt, X().monName[tt - 1]) : X().o1d(T.dmy(date), t),
    X().o2(k, nf(t, 0), tr ? '12' : nf(365.25, 2), nz(p.ang, 4), nz((p.ang / (2 * Math.PI)) % 1, 4)),
    X().o3(nz(p.s, 4), nz(p.c, 4)),
    X().o4(k, nz(co[k - 1][0], 4), pn(p.s, 4), nz(co[k - 1][1], 4), pn(p.c, 4), nz(p.v, 4), tr ? '' : ' GW', nz(p.amp, 4)),
    tr ? X().o5t(Kmax, nz(tot, 4), nz(D.fourier.pat[tt - 1], 4), nf(Math.exp(tot), 3)) : X().o5d(Kmax, nz(tot, 3), nf(1000 * tot, 0)),
    X().o6(Kmax, 2 * Kmax, tr)]);
  table('xo-tb', ['k', 'α<sub>k</sub> (sin)', 'β<sub>k</sub> (cos)', `2πk·t/${tr ? 12 : '365.25'}`, 'sin', 'cos', X().contrib, X().amp],
    P.map((z, j) => hiRow([String(j + 1), num(co[j][0], 4), num(co[j][1], 4), num(z.ang, 3), num(z.s, 4), num(z.c, 4), num(z.v, 4), num(z.amp, 4)], j + 1 === k)), { cap: tr ? X().oCapT : X().oCapD });
  out('xo-out', [tr ? X().o0t(X().monName[tt - 1], nf(tot, 3), nf(D.fourier.pat[tt - 1], 3)) : X().o0d(T.dmy(date), nf(1000 * tot, 0))]);
}

// ---------------------------------------------------------------- Interpreting the Fourier fits: R2 and AICc by K
function fourierRow(K0) {
  const pat = D.fourier.pat, n = 12, F = K0 === 6 ? fourierFit(6, true) : fourierFit(K0), t = range(1, 12, 1), Xc = fourierCols(t, K0, K0 === 6);
  const fit = t.map((_, i) => Xc.reduce((s, c, j) => s + F.b[j] * c[i], 0)), ssr = sum(pat.map((v, i) => (v - fit[i]) ** 2)), kc = K0 === 6 ? 11 : 2 * K0, kk = kc + 1;
  const aicc = ssr > 1e-12 && n - kk - 1 > 0 ? n * Math.log(ssr / n) + 2 * kk + 2 * kk * (kk + 1) / (n - kk - 1) : NaN;
  return { K0, kc, r2: F.r2, ssr, sd: n - kc > 0 ? Math.sqrt(ssr / (n - kc)) : NaN, aicc, aug: fit[7], jan: fit[0] };
}
function blkXFourTab() {
  const K0 = +$('fk-K').value; set('fk-K', K0);
  const R = range(1, 6, 1).map(fourierRow), am = Math.min(...R.filter(r => isFinite(r.aicc)).map(r => r.aicc)), pat = D.fourier.pat;
  table('fk-tb', ['K', X().ncoef, 'R²', 'SSR', X().resSd, 'AICc', X().aug, X().jan],
    R.map(r => hiRow([String(r.K0), String(r.kc), num(r.r2, 4), num(r.ssr, 5), num(r.sd, 4), num(r.aicc, 1, r.aicc === am), num(r.aug, 3), num(r.jan, 3)], r.K0 === K0)),
    { cap: X().fkCap(nf(pat[7], 3), nf(pat[0], 3)), note: X().fkNote(nf(fourierFit(6).r2, 4), nf(R[4].r2, 4)) });
  const G = grid('fk-chart', [{ r: 0, c: 0, t: 'R² by number of harmonics K', xt: 'K', x: { dtick: 1 }, y: { range: [0.9, 1.005] } }, { r: 0, c: 1, t: 'AICc by number of harmonics K', xt: 'K', x: { dtick: 1, range: [0.5, 6.5] } }],
    { cols: 2, h: 340, legendRows: 1, gx: 0.09 });
  const [a, b] = G.ax, ks = R.map(r => r.K0), ok = R.filter(r => isFinite(r.aicc));
  plot('fk-chart', [ln(ks, R.map(r => r.r2), a, K.blue, 'R²', { mode: 'lines+markers', marker: { size: 7, color: ks.map(k => (k === K0 ? K.red : K.blue)) }, hovertemplate: 'K = %{x}: %{y:.4f}<extra></extra>' }),
    ln(ok.map(r => r.K0), ok.map(r => r.aicc), b, K.purple, 'AICc (K = 5 and K = 6 leave too few degrees of freedom)', { mode: 'lines+markers', line: { color: K.purple, width: 1.3 }, marker: { size: 7, color: ok.map(r => (r.K0 === K0 ? K.red : K.purple)) }, hovertemplate: 'K = %{x}: %{y:.1f}<extra></extra>' }),
    { x: [K0], y: [R[K0 - 1].r2], mode: 'markers', marker: { size: 15, color: 'rgba(0,0,0,0)', line: { color: K.red, width: 2 } }, xaxis: a.xa, yaxis: a.ya, name: `K = ${K0}`, hovertemplate: 'K = %{x}: %{y:.4f}<extra></extra>' },
    { x: isFinite(R[K0 - 1].aicc) ? [K0] : [], y: isFinite(R[K0 - 1].aicc) ? [R[K0 - 1].aicc] : [], mode: 'markers', marker: { size: 15, color: 'rgba(0,0,0,0)', line: { color: K.red, width: 2 } }, xaxis: b.xa, yaxis: b.ya, showlegend: false, hovertemplate: 'K = %{x}: %{y:.1f}<extra></extra>' }], G.lay);
  const r = R[K0 - 1], best = ok.find(z => z.aicc === am);
  out('fk-out', [X().fk0(K0, r.kc, nf(r.r2, 3), nf(r.aug, 3), nf(pat[7], 3)), X().fk1(best.K0, nf(best.aicc, 1))]);
  return R;
}

// ---------------------------------------------------------------- Calendar effects and Orthodox Easter: the Easter share
function blkXEaster() {
  const yr = +$('xe-y').value, w = +$('xe-w').value; set('xe-y', yr); set('xe-w', w);
  const e = EAS[yr], days = range(1, w, 1).map(j => addD(e, -j)).reverse(), byM = {};
  days.forEach(d => { const k = d.slice(0, 7); byM[k] = (byM[k] || []).concat([d]); });
  const ms = Object.keys(byM).sort(), mn = k => X().monName[+k.slice(5, 7) - 1], bE = D.easter.reg[w].food[0];
  steps('xe-ws', [X().e1(yr, T.dmy(e), w, T.dmy(days[0]), T.dmy(days[w - 1])),
    X().e2(ms.map(k => [mn(k), byM[k].length, w, nz(byM[k].length / w, 2)])),
    X().e3(w, nf(100 * bE, 2), ms.map(k => [mn(k), nz(100 * bE * byM[k].length / w, 2)]))]);
  table('xe-tb', [X().month0, X().daysIn, 'E<sub>t</sub>', X().effFood], ms.map(k => [mn(k) + ' ' + yr, String(byM[k].length), num(byM[k].length / w, 2), num(100 * bE * byM[k].length / w, 2)]), { cap: X().xeCap(w) });
  out('xe-out', [X().e0(yr, T.dmy(e), ms.map(k => [mn(k), nz(byM[k].length / w, 2)]))]);
}

// ---------------------------------------------------------------- Interpreting the Easter effect: coefficients by window w
function blkXEaTab() {
  const w = +$('et-w').value; set('et-w', w);
  const R = D.easter.reg[w], row = (lab, r) => [lab, num(100 * r[0], 2), num(100 * r[1], 2), num(r[0] / r[1], 2), txt(fp(2 * (1 - normCdf(Math.abs(r[0] / r[1]))))), num(100 * r[2], 2), num(100 * r[3], 2), num(r[2] / r[3], 1)];
  table('et-tb', [[X().series0, { v: X().easterCol(w), cs: 4 }, { v: X().wdCol, cs: 3 }], ['', 'β̂<sub>E</sub> (%)', 'SE', 't', 'p', 'β̂<sub>W</sub> (%)', 'SE', 't']],
    [row(X().food, R.food), row(X().total, R.total)], { cap: X().etCap, note: w === 10 ? X().etNote(nf(100 * NB.easter.food.params.ao_2020_04, 1), nf(100 * NB.easter.total.params.ao_2020_04, 1)) : '' });
  const ws = range(1, 20, 1), G = grid('et-chart', [{ r: 0, c: 0, t: 'Easter coefficient β_E by window w, ±1.96 SE (%)', xt: 'window w (days before Easter)', x: { dtick: 2 } },
    { r: 0, c: 1, t: 'Raw scatter slope by w (pp)', xt: 'window w', x: { dtick: 2 } }], { cols: 2, h: 360, legendRows: 2, gx: 0.09 });
  const [a, b] = G.ax, f = ws.map(x => D.easter.reg[x].food), tt = ws.map(x => D.easter.reg[x].total);
  const slope = ws.map(x => { const P = easterPts(x); return polyfit1(P.map(p => p.ds), P.map(p => p.g))[0]; });
  plot('et-chart', [ln(ws, f.map(r => 100 * r[0]), a, K.red, 'food retail', { mode: 'lines+markers', marker: { size: 5 }, error_y: { type: 'data', array: f.map(r => 196 * r[1]), color: K.red, thickness: 1, width: 2 }, hovertemplate: 'w = %{x}: %{y:.2f}%<extra></extra>' }),
    ln(ws.map(x => x + 0.25), tt.map(r => 100 * r[0]), a, K.blue, 'all retail', { mode: 'lines+markers', marker: { size: 5 }, error_y: { type: 'data', array: tt.map(r => 196 * r[1]), color: K.blue, thickness: 1, width: 2 }, hovertemplate: 'w = %{x:.0f}: %{y:.2f}%<extra></extra>' }),
    ln(ws, slope, b, K.navy, 'slope of y/y growth on the change in the Easter share', { mode: 'lines+markers', marker: { size: 5 }, line: { color: K.navy, width: 1.2 }, hovertemplate: 'w = %{x}: %{y:.2f}<extra></extra>' }),
    { x: [w, w + 0.25], y: [100 * R.food[0], 100 * R.total[0]], mode: 'markers', marker: { size: 13, color: 'rgba(0,0,0,0)', line: { color: K.amber, width: 2.2 } }, xaxis: a.xa, yaxis: a.ya, name: `w = ${w}`, hovertemplate: '%{y:.2f}%<extra></extra>' },
    { x: [w], y: [slope[w - 1]], mode: 'markers', marker: { size: 13, color: 'rgba(0,0,0,0)', line: { color: K.amber, width: 2.2 } }, xaxis: b.xa, yaxis: b.ya, showlegend: false, hovertemplate: 'w = %{x}: %{y:.2f}<extra></extra>' }],
  Object.assign(G.lay, { shapes: [hl(a, 0, K.navy, 'solid', 0.5), vl(a, w, K.amber, 'dash', 1), vl(b, w, K.amber, 'dash', 1)] }));
  out('et-out', [X().et0(w, w === 10, nf(100 * R.food[0], 1), nf(100 * R.food[1], 1), nf(R.food[0] / R.food[1], 1), nf(100 * R.total[0], 1), nf(100 * R.total[1], 1), nf(100 * R.total[2], 2), nf(100 * R.food[2], 2))]);
}

// ---------------------------------------------------------------- A dynamic harmonic regression for daily load: coefficients and the annual component
const DHR_ROWS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'holiday', 'easter', 'xmas'];
function dhrDet(d, K0) {
  const t = (d2t(d) - d2t('2000-01-01')) / DAY, fo = range(1, K0, 1).reduce((s, k) => s + DHRK[k - 1][0] * Math.sin(2 * Math.PI * k * t / 365.25) + DHRK[k - 1][1] * Math.cos(2 * Math.PI * k * t / 365.25), 0);
  const wd = dow(d), wk = wd === 6 ? 0 : NB.dhr.coef[DHR_ROWS[wd]], h = HOL[d], mo = d.slice(5, 7), dd = +d.slice(8, 10);
  const xm = (mo === '12' && dd >= 24) || (mo === '01' && dd <= 2), ea = h === 'Easter', ho = !!h && !ea && !xm;
  return { fo, wk, sp: (ea ? NB.dhr.coef.easter : 0) + (xm ? NB.dhr.coef.xmas : 0) + (ho ? NB.dhr.coef.holiday : 0), ea, xm, ho, h };
}
function blkXDhr() {
  const di = +$('dr-d').value, K0 = +$('dr-K').value, d = addD('2025-01-01', di); set('dr-d', T.dmy(d)); set('dr-K', K0);
  const C = NB.dhr.coef, S = NB.dhr.se, lab = X().dhrLab;
  table('dr-tb', [X().coef, X().est + ' (GW)', 'SE', 't'], DHR_ROWS.map(k => [lab[k], num(C[k], 3), num(S[k], 3), num(C[k] / S[k], 1)])
    .concat(range(1, 4, 1).map(k => [`α<sub>${k}</sub>, β<sub>${k}</sub> (${X().annual})`, txt(`${nz(C[`sin${k}_365.25`], 3)}; ${nz(C[`cos${k}_365.25`], 3)}`), txt(`${nz(S[`sin${k}_365.25`], 3)}; ${nz(S[`cos${k}_365.25`], 3)}`), txt(`${nz(C[`sin${k}_365.25`] / S[`sin${k}_365.25`], 1)}; ${nz(C[`cos${k}_365.25`] / S[`cos${k}_365.25`], 1)}`)]))
    .concat([['AR(1)', num(NB.dhr.ar[0], 3), txt('—'), txt('—')], ['MA(1)', num(NB.dhr.ma[0], 3), txt('—'), txt('—')], ['σ̂ (MW)', num(1000 * NB.dhr.sigma, 0), txt('—'), txt('—')]]),
  { cap: X().drCap(nf(NB.dhr.lb14.lb, 1), NB.dhr.lb14.df, pe(NB.dhr.lb14.lb_p)) });
  const z = dhrDet(d, K0), z4 = dhrDet(d, 4);
  steps('dr-ws', [X().d1(T.dmy(d), X().wdName[dow(d)], nz(z.wk, 3)), X().d2(K0, nz(z.fo, 3)), X().d3(z.h ? X().hol(z.h) : null, z.ea, z.xm, z.ho, nz(z.sp, 3)),
    X().d4(nz(z.wk, 3), pn(z.fo, 3), pn(z.sp, 3), nz(z.wk + z.fo + z.sp, 3))]);
  const days = range(0, 364, 1).map(i => addD('2025-01-01', i)), G = grid('dr-chart', [{ r: 0, c: 0, t: 'Annual Fourier component of the DHR, 2025 (GW)', x: { type: 'date', tickformat: '%b' } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0], cols = [K.blue, K.green, K.orange, K.red];
  plot('dr-chart', range(1, 4, 1).map(k => ln(days, days.map(x => dhrDet(x, k).fo), a, cols[k - 1], `K = ${k}`, { line: { color: cols[k - 1], width: k === K0 ? 2.8 : 1.1 }, opacity: k === K0 ? 1 : 0.45, hovertemplate: `K = ${k}, %{x|%d %b}: %{y:.3f}<extra></extra>` }))
    .concat([{ x: [d, d, d, d], y: range(1, 4, 1).map(k => dhrDet(d, k).fo), mode: 'markers', marker: { size: range(1, 4, 1).map(k => (k === K0 ? 12 : 7)), color: cols.map((c, i) => (i + 1 === K0 ? c : c + '66')), line: { color: K.navy, width: 1 } }, xaxis: a.xa, yaxis: a.ya, name: T.dmyEn(d), hovertemplate: '%{y:.3f}<extra></extra>' }]),
    Object.assign(G.lay, { shapes: [hl(a, 0, K.navy, 'solid', 0.5), vl(a, d, K.amber, 'dash', 1.2)] }));
  const f4 = days.map(x => dhrDet(x, 4).fo), imx = f4.indexOf(Math.max(...f4)), imn = f4.indexOf(Math.min(...f4));
  out('dr-out', [X().dr0(nf(C.wed, 2), nf(C.mon, 2), nf(C.sat, 2), nf(C.holiday, 2), nf(C.easter, 2), nf(C.xmas, 2)), X().dr1(T.dmy(days[imx]), nf(f4[imx], 2), T.dmy(days[imn]), nf(f4[imn], 2), nf(z4.wk + z4.fo + z4.sp, 2), T.dmy(d))]);
}

// ---------------------------------------------------------------- Diebold-Mariano by horizon
const loss = (e, l) => (l === 'se' ? e * e : Math.abs(e));
function dmByH(m1, m2, l) {
  return range(0, HZ - 1, 1).map(j => { const d = range(0, NO - 1, 1).map(k => loss(err(m1, k, j), l) - loss(err(m2, k, j), l)); return Object.assign(dmRaw(d, d.map(() => 0)), { db: mean(d) }); });
}
function winLoss(m, l) { return range(0, NO - 1, 1).map(k => mean(range(0, HZ - 1, 1).map(j => loss(err(m, k, j), l)))); }
function blkXDmH() {
  const m1 = sel('dz-1'), m2 = sel('dz-2'), l = sel('dz-l');
  const R = dmByH(m1, m2, l), W = dmRaw(winLoss(m1, l), winLoss(m2, l)), c = tq(NO - 1), u = l === 'se' ? 1e3 : 1e3;
  const G = grid('dz-chart', [{ r: 0, c: 0, t: `HLN statistic by horizon, ${m1} minus ${m2} (${l === 'se' ? 'squared' : 'absolute'} errors)`, xt: 'horizon h (days)', yt: 'HLN', x: { dtick: 1 } }], { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0], hs = range(1, HZ, 1);
  plot('dz-chart', [{ x: hs, y: R.map(r => r.hln), type: 'bar', marker: { color: R.map(r => (Math.abs(r.hln) > c ? K.red : K.blue)) }, name: 'HLN at horizon h (26 origins)', xaxis: a.xa, yaxis: a.ya, hovertemplate: 'h = %{x}: %{y:.2f}<extra></extra>' },
    leg(`±t0.975(25) = ±${c.toFixed(2)}`, { color: K.red, dash: 'dash', width: 1.2 }), leg(`window average, all horizons: ${W.hln.toFixed(2)}`, { color: K.green, dash: 'dot', width: 1.6 })],
  Object.assign(G.lay, { shapes: [hl(a, c, K.red, 'dash', 1.1), hl(a, -c, K.red, 'dash', 1.1), hl(a, 0, K.navy, 'solid', 0.5), hl(a, W.hln, K.green, 'dot', 1.6)] }));
  table('dz-tb', ['h', `d̄ (${l === 'se' ? 'MW²·10³' : 'MW'})`, 'DM', 'HLN', 'p'], R.map((r, j) => hiRow([String(j + 1), num(r.db * (l === 'se' ? 1e3 : 1e3), 1), num(r.dm, 2), num(r.hln, 2), txt(fp(r.p))], Math.abs(r.hln) > c)), { cap: X().dzCap });
  const sig = R.map((r, j) => [j + 1, r]).filter(([, r]) => Math.abs(r.hln) > c).map(([h]) => h);
  out('dz-out', [X().dz0(m1, m2, l, nf(W.hln, 2), pe(W.p)), X().dz1(sig, HZ, nf(c, 2))]);
  return { R, W, u };
}

// ---------------------------------------------------------------- Worked example: a Diebold-Mariano test
const DM5 = [-0.30, 0.10, -0.50, -0.20, -0.40];
function blkXDm() {
  const src = sel('xd-src'), m1 = sel('xd-1'), m2 = sel('xd-2'), l = sel('xd-l'), real = src === 'real';
  ['xd-1', 'xd-2', 'xd-l'].forEach(i => { $(i).disabled = !real; });
  const l1 = real ? winLoss(m1, l) : null, l2 = real ? winLoss(m2, l) : null, d = real ? l1.map((v, i) => v - l2[i]) : DM5, n = d.length;
  const db = mean(d), g0 = pvar(d), dm = db / Math.sqrt(g0 / n), c = Math.sqrt((n - 1) / n), hln = c * dm, p = tsf2(hln, n - 1), tc = tq(n - 1);
  const dd = real ? 3 : 2, show = d.map(v => pn(v, dd)).join(', ');
  steps('xd-ws', [real ? X().w1r(m1, m2, l, n, show) : X().w1s(show), X().w2(n, nz(sum(d), dd + 1), nz(db, 4)),
    X().w3(n, nz(g0, 5)), X().w4(nz(db, 4), nz(g0, 5), n, nz(dm, 2)), X().w5(n, nz(c, 4), nz(dm, 2), nz(hln, 2)),
    X().w6(nz(Math.abs(hln), 2), Math.abs(hln) > tc, n - 1, nz(tc, 2), pe(p), Math.abs(hln) > 1.96 !== Math.abs(hln) > tc)]);
  table('xd-tb', real ? ['j', X().origin, `L(e<sub>1</sub>), ${X().mn(m1)}`, `L(e<sub>2</sub>), ${X().mn(m2)}`, 'd<sub>j</sub>'] : ['j', 'd<sub>j</sub>', 'd<sub>j</sub> − d̄', '(d<sub>j</sub> − d̄)²'],
    real ? d.map((v, j) => [String(j + 1), T.dmy(CV.origins[j]), num(l1[j], 4), num(l2[j], 4), num(v, 4)]) : d.map((v, j) => [String(j + 1), num(v, 2), num(v - db, 2), num((v - db) ** 2, 4)]),
    { cap: real ? X().xdCapR(l) : X().xdCapS });
  out('xd-out', [X().xd0(real ? `${X().mn(m1)} − ${X().mn(m2)}` : X().slide5, nf(dm, 2), nf(hln, 2), pe(p))]);
  return { dm, hln, p };
}

// ---------------------------------------------------------------- Interpreting the cross-validation results: the table, with MASE worked out
function cvSub(wsel, hsel) {
  const all = range(0, NO - 1, 1), ks = wsel === 'all' ? all : all.filter(k => holWin(k) === (wsel === 'hol'));
  const js = hsel === 'all' ? range(0, HZ - 1, 1) : hsel === '1-7' ? range(0, 6, 1) : range(7, 13, 1);
  return { ks, js };
}
function blkXCv() {
  const wsel = sel('cv-w'), hsel = sel('cv-h'), l = sel('cv-l'), ref = sel('cv-r'), mm = sel('cv-m'), { ks, js } = cvSub(wsel, hsel);
  const sc = NB.cv.scale, R = {}, M = CV.models;
  M.forEach(m => { const e = [].concat(...ks.map(k => js.map(j => err(m, k, j)))); R[m] = { MAE: mean(e.map(Math.abs)), RMSE: Math.sqrt(mean(e.map(v => v * v))), win: ks.map(k => mean(js.map(j => loss(err(m, k, j), l)))) }; R[m].MASE = R[m].MAE / sc; });
  const dmc = (a, b) => { if (a === b) return txt('—'); const t = dmRaw(R[a].win, R[b].win); return txt(`${nz(t.hln, 2)} (${fp(t.p)})`); };
  const best = k => Math.min(...M.map(m => R[m][k]));
  table('cv-tb', [X().method, 'MAE', 'RMSE', 'MASE', X().dmVs('Seasonal naive'), X().dmVs(ref)],
    M.map(m => hiRow([X().mn(m), num(1000 * R[m].MAE, 0, R[m].MAE === best('MAE')), num(1000 * R[m].RMSE, 0, R[m].RMSE === best('RMSE')), num(R[m].MASE, 2, R[m].MASE === best('MASE')), dmc(m, 'Seasonal naive'), dmc(m, ref)], m === mm)),
    { cap: X().cvCap(ks.length, js.length, l) });
  const y = DL.v.slice(0, dIdx(CV.origins[0]) + 1), ab = y.slice(7).map((v, i) => Math.abs(v - y[i])), dd = i => addD(DL.d0, i);
  steps('cv-ws', [X().m1(T.dmy(DL.d0), T.dmy(CV.origins[0]), y.length),
    X().m2(T.dmy(dd(7)), nf(y[7], 3), T.dmy(dd(0)), nf(y[0], 3), nz(ab[0], 3), T.dmy(dd(8)), nf(y[8], 3), T.dmy(dd(1)), nf(y[1], 3), nz(ab[1], 3)),
    X().m3(ab.length, nz(sum(ab), 2), nf(1000 * sc, 1)),
    X().m4(mm, ks.length * js.length, nf(1000 * R[mm].MAE, 1), nf(1000 * R[mm].MAE, 1), nf(1000 * sc, 1), nz(R[mm].MASE, 3), R[mm].MASE < 1)]);
  const below = M.filter(m => R[m].MASE < 1);
  out('cv-out', [X().cv0(ks.length, js.length, [...M].sort((p, q) => R[p].MAE - R[q].MAE)[0], below)]);
  return R;
}

// ---------------------------------------------------------------- A second test: Romanian GDP
const GC = D.gdpcv, GCM = ['SARIMA', 'ETS', 'Seasonal naive', 'Combination'];
function dmH(d, h) {
  const n = d.length, db = mean(d), g = range(0, h - 1, 1).map(k => sum(d.slice(k).map((v, i) => (v - db) * (d[i] - db))) / (n - k));
  const v = (g[0] + 2 * sum(g.slice(1))) / n, dm = db / Math.sqrt(v), hln = Math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) * dm;
  return { db, dm, hln, p: tsf2(hln, n - 1), n };
}
function blkXGdpCv() {
  const ex = sel('gc-s') === 'ex', met = sel('gc-m'), l = sel('gc-l');
  const ks = range(0, GC.origins.length - 1, 1).filter(k => !ex || !(GC.origins[k] >= '2019-01-01' && GC.origins[k] <= '2020-12-31'));
  const E = (m, h) => ks.map(k => GC.err[m][k][h - 1]), st = (m, h) => { const e = E(m, h); return met === 'MAE' ? mean(e.map(Math.abs)) : Math.sqrt(mean(e.map(v => v * v))); };
  const V = GCM.map(m => range(1, 4, 1).map(h => st(m, h)));
  table('gc-tb', [X().method, `${met}, h = 1`, 'h = 2', 'h = 3', 'h = 4'], GCM.map((m, i) => [X().gcName[m], ...V[i].map((v, j) => num(v, 2, v === Math.min(...V.map(r => r[j]))))]), { cap: X().gcCap(ks.length, ex, met) });
  const D2 = ['ETS', 'Seasonal naive'].map(r => range(1, 4, 1).map(h => dmH(E('SARIMA', h).map((v, i) => loss(v, l) - loss(E(r, h)[i], l)), h)));
  table('gc-tb2', [X().gcDm, 'h = 1', 'h = 2', 'h = 3', 'h = 4'], ['ETS', 'Seasonal naive'].map((r, i) => [X().gcVs(r), ...D2[i].map(t => txt(`${nz(t.hln, 2)} (${fp(t.p)})`))]), { cap: X().gcCap2(l) });
  const G = grid('gc-chart', [{ r: 0, c: 0, t: `${met} by horizon, ${ks.length} quarterly origins (%)`, xt: 'horizon h (quarters)', x: { dtick: 1 } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0], col = { SARIMA: K.purple, ETS: K.teal, 'Seasonal naive': K.amber, Combination: K.red };
  plot('gc-chart', GCM.map((m, i) => ln([1, 2, 3, 4], V[i], a, col[m], m === 'SARIMA' ? 'SARIMA (airline)' : m === 'Combination' ? 'Combination (SARIMA, ETS)' : m,
    { mode: 'lines+markers', marker: { size: 6 }, line: { color: col[m], width: 1.5, dash: m === 'Seasonal naive' ? 'dash' : 'solid' }, hovertemplate: `${m}, h = %{x}: %{y:.2f}<extra></extra>` })), G.lay);
  out('gc-out', [X().gc0(ks.length, ex, met, nf(V[0][0], 2), nf(V[1][0], 2), nf(V[0][3], 2), nf(V[1][3], 2), nf(V[2][3], 2)), X().gc1(nf(D2[0][0].hln, 2), pe(D2[0][0].p), nf(D2[0][3].hln, 2), pe(D2[0][3].p))]);
  return { V, D2 };
}

// ---------------------------------------------------------------- Why combining forecasts works: Bates-Granger with the slide's and the real numbers
function allErr(m) { return [].concat(...range(0, NO - 1, 1).map(k => range(0, HZ - 1, 1).map(j => 1000 * err(m, k, j)))); }
function blkXComb() {
  const real = sel('xc-src') === 'real', m1 = sel('xc-1'), m2 = sel('xc-2'), w = +$('xc-w').value; set('xc-w', nf(w, 2));
  ['xc-1', 'xc-2'].forEach(i => { $(i).disabled = !real; });
  let s1 = 1, s2 = 1.2, rho = 0.75, e1 = null, e2 = null;
  if (real) { e1 = allErr(m1); e2 = allErr(m2); s1 = Math.sqrt(pvar(e1)); s2 = Math.sqrt(pvar(e2)); rho = corr(e1, e2); }
  const mse = x => x * x * s1 * s1 + (1 - x) ** 2 * s2 * s2 + 2 * x * (1 - x) * rho * s1 * s2, ws = (s2 * s2 - rho * s1 * s2) / (s1 * s1 + s2 * s2 - 2 * rho * s1 * s2), d = real ? 0 : 3;
  const L = [X().c1(real ? `${m1}, ${m2}` : null, nz(s1, d), nz(s2, d), nz(rho, 3), real),
    X().c2(nz(w, 2), nz(s1, d), nz(1 - w, 2), nz(s2, d), nz(rho, 3), nz(mse(w), real ? 0 : 3)),
    X().c3(nz(s2 * s2, real ? 0 : 2), nz(rho * s1 * s2, real ? 0 : 2), nz(s1 * s1 + s2 * s2 - 2 * rho * s1 * s2, real ? 0 : 2), nz(ws, 2), nz(mse(ws), real ? 0 : 3)),
    X().c4(nz(mse(0.5), real ? 0 : 3), nz(Math.min(s1, s2) ** 2, real ? 0 : 3), mse(0.5) < Math.min(s1, s2) ** 2)];
  if (real) {
    const k = CV.origins.indexOf('2026-04-06'), j = 5, y = actual(CV.origins[k], j), f1 = CV.fc[m1][k][j], f2 = CV.fc[m2][k][j], fc = w * f1 + (1 - w) * f2;
    L.push(X().c5(T.dmy(addD(CV.origins[k], j + 1)), nf(1000 * y, 0), nf(1000 * f1, 0), nf(1000 * f2, 0), nz(w, 2), nz(1 - w, 2), nf(1000 * fc, 0), nz(1000 * (y - fc), 0), nz(1000 * (y - f1), 0), nz(1000 * (y - f2), 0)));
    const ec = e1.map((v, i) => w * v + (1 - w) * e2[i]), bias = mean(ec);
    L.push(X().c6(nz(mse(w), 0), nz(bias, 1), nz(mse(w) + bias * bias, 0), nf(Math.sqrt(mean(ec.map(v => v * v))), 1), nf(mean(ec.map(Math.abs)), 1)));
  }
  steps('xc-ws', L);
  out('xc-out', [X().c0(real, nz(w, 2), nf(mse(w), real ? 0 : 3), nf(ws, 2), nf(mse(ws), real ? 0 : 3))]);
  return { s1, s2, rho, ws, mse05: mse(0.5), msew: mse(ws) };
}

// ---------------------------------------------------------------- Interpreting the combination: weights and errors of the combination schemes
const CB_SETS = { five: ['ETS', 'SARIMA', 'DHR', 'TBATS', 'Prophet'], dp: ['DHR', 'Prophet'], es: ['ETS', 'SARIMA'], et: ['ETS', 'TBATS'] };
function blkXCombTab() {
  const S = CB_SETS[sel('cw-s')], js = range(0, HZ - 1, 1), ks = range(0, NO - 1, 1);
  const act = [].concat(...ks.map(k => js.map(j => actual(CV.origins[k], j)))), F = S.map(m => [].concat(...CV.fc[m]));
  const evalW = Wk => { const e = act.map((y, i) => { const k = Math.floor(i / HZ), w = Wk(k); return y - S.reduce((s, m, q) => s + w[q] * F[q][i], 0); }); return [1000 * mean(e.map(Math.abs)), 1000 * Math.sqrt(mean(e.map(v => v * v)))]; };
  const mseW = (m, kk) => mean([].concat(...kk.map(k => js.map(j => err(m, k, j) ** 2))));
  const inv = kk => { const v = S.map(m => 1 / mseW(m, kk)), s = sum(v); return v.map(x => x / s); };
  const eq = S.map(() => 1 / S.length), hind = inv(ks), past = ks.map(k => (k === 0 ? eq : inv(range(0, k - 1, 1))));
  const avgPast = S.map((_, q) => mean(past.map(w => w[q])));
  const rows = [[X().eqW, eq, evalW(() => eq)], [X().invPast, avgPast, evalW(k => past[k])], [X().invHind, hind, evalW(() => hind)]];
  if (S.length === 2) { const g = range(0, 1, 0.01).map(w => [w, evalW(() => [w, 1 - w])]), b = g.reduce((p, q) => (q[1][0] < p[1][0] ? q : p)); rows.push([X().bestHind, [b[0], 1 - b[0]], b[1]]); }
  S.forEach((m, q) => rows.push([X().alone(m), S.map((_, r) => (r === q ? 1 : 0)), evalW(() => S.map((_, r) => (r === q ? 1 : 0)))]));
  const mn = Math.min(...rows.map(r => r[2][0]));
  table('cw-tb', [X().scheme, ...S.map(m => `w<sub>${X().mn(m)}</sub>`), 'MAE (MW)', 'RMSE (MW)'], rows.map(r => [r[0], ...r[1].map(v => num(v, 2)), num(r[2][0], 0, r[2][0] === mn), num(r[2][1], 0)]), { cap: X().cwCap(S.length) });
  out('cw-out', [X().cw0(S.length, nf(rows[0][2][0], 0), nf(rows[1][2][0], 0), nf(rows[2][2][0], 0), nf(Math.min(...rows.slice(-S.length).map(r => r[2][0])), 0))]);
  return rows;
}

function placeExtras() {                // web lecture: the extra blocks travel inside a chart block; move each to its own frame
  document.querySelectorAll('.xhold > section.xblk').forEach(sec => {
    const art = document.getElementById('s' + sec.dataset.frame), head = art && art.querySelector('.fhead');
    if (head) head.after(sec); else sec.closest('section.blk').after(sec);
  });
  document.querySelectorAll('.xhold').forEach(h => { if (!h.children.length) h.remove(); });
}
// [anchor, prefix, function, inputs, input moved by ▶, its values, ms]
const XB = [['wx-sdiff', 'xs', blkXSdiff, ['xs-q'], 'xs-q', range(5, LY.length - 1, 1), 250],
  ['wx-roots', 'xr', blkXRoots, ['xr-s', 'xr-k'], 'xr-k', range(0, 11, 1), 700],
  ['hegy-table', 'hg', blkXHegy, ['hg-end', 'hg-k', 'hg-det'], 'hg-end', range(0, HG_END.length - 1, 1), 600],
  ['sur-table', 'su', blkXStests, ['su-end'], 'su-end', range(2012, 2025, 1), 700],
  ['wx-air', 'xa', blkXAir, ['xa-th', 'xa-Th'], 'xa-Th', range(-0.95, 0.95, 0.05), 300],
  ['wx-fc', 'xf', blkXFc, ['xf-o', 'xf-h'], 'xf-h', range(1, 24, 1), 600],
  ['air-table', 'at', blkXAirTab, ['at-o'], 'at-o', range(1952, 1958, 1), 900],
  ['gdpm-table', 'gm', blkXGdpm, ['gm-m', 'gm-L'], 'gm-m', GM_ORDER, 700],
  ['gdpfc-table', 'gt', blkXGdpfcTab, ['gt-Th', 'gt-lv'], 'gt-Th', range(-0.95, 0, 0.05), 400],
  ['sa-table', 'st', blkXSa, ['st-y0', 'st-y1', 'st-q'], 'st-q', range(41, LY.length - 1, 1), 300],
  ['wx-fourier', 'xo', blkXFour, ['xo-c', 'xo-k', 'xo-t'], 'xo-t', range(1, 365, 7), 200],
  ['fourier-table', 'fk', blkXFourTab, ['fk-K'], 'fk-K', range(1, 6, 1), 900],
  ['wx-easter', 'xe', blkXEaster, ['xe-y', 'xe-w'], 'xe-y', range(2000, 2030, 1), 500],
  ['easter-table', 'et', blkXEaTab, ['et-w'], 'et-w', range(1, 20, 1), 500],
  ['dhr-table', 'dr', blkXDhr, ['dr-d', 'dr-K'], 'dr-d', range(0, 364, 7), 200],
  ['dm-h', 'dz', blkXDmH, ['dz-1', 'dz-2', 'dz-l'], 'dz-2', ['Seasonal naive', 'ETS', 'SARIMA', 'TBATS', 'Prophet', 'Combination'], 1100],
  ['wx-dm', 'xd', blkXDm, ['xd-src', 'xd-1', 'xd-2', 'xd-l'], 'xd-src', ['slide', 'real'], 1500],
  ['cv-table', 'cv', blkXCv, ['cv-w', 'cv-h', 'cv-l', 'cv-r', 'cv-m'], 'cv-m', ['Seasonal naive', 'ETS', 'SARIMA', 'DHR', 'TBATS', 'Prophet', 'Combination'], 1000],
  ['gdpcv-table', 'gc', blkXGdpCv, ['gc-s', 'gc-m', 'gc-l'], 'gc-s', ['all', 'ex'], 1400],
  ['wx-comb', 'xc', blkXComb, ['xc-src', 'xc-1', 'xc-2', 'xc-w'], 'xc-w', range(0, 1, 0.05), 300],
  ['comb-table', 'cw', blkXCombTab, ['cw-s'], 'cw-s', ['five', 'dp', 'es', 'et'], 1400]];

// ================================================================= wiring
const ALL = [blkSeries, blkSplot, blkCircle, blkGdpDiff, blkTheory, blkAirline, blkAirFc, blkGdpDiag, blkGdpFc, blkSa, blkFourier, blkEaster, blkHourly, blkMstl,
  blkDaily, blkProphet, blkScheme, blkLoadFc, blkCvRes, blkComb];
function fillSelect(id, vals, sel, lab) { const s = $(id); s.innerHTML = vals.map(v => `<option value="${v}"${v === sel ? ' selected' : ''}>${lab ? lab(v) : v}</option>`).join(''); }
function init() {
  placeExtras();
  const has = id => !!$(id), on = (ids, f) => ids.forEach(id => $(id).addEventListener('input', f));
  const players = {}, P = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  const from0 = (btn, id, v0, v) => $(btn).addEventListener('click', () => { if (!players[btn].timer && $(id).value === String(v0)) $(id).value = v; }, true);   // ▶ from the slide state: start the sweep at its first value
  // ▶ that moves several controls: a detached input steps through the states, apply(state) sets the controls and redraws
  const drive = (btn, states, apply, ms) => { const el = document.createElement('input'); el.value = states.length - 1;
    el.addEventListener('input', () => apply(states[+el.value])); return (players[btn] = new Player(btn, el, range(0, states.length - 1, 1), ms)); };
  seDefaults(); $('se-s').addEventListener('input', () => { seDefaults(); blkSeries(); }); on(['se-y0', 'se-y1'], blkSeries); P('se-play', 'se-y1', range(2000, 2026, 1), 450);
  on(['sp-y'], blkSplot); P('sp-play', 'sp-y', range(2012, 2025, 1), 600);
  on(['uc-s'], blkCircle); P('uc-play', 'uc-s', ['2', '3', '4', '6', '7', '12', '24'], 1100); from0('uc-play', 'uc-s', '12', '24');
  on(['df-end'], blkGdpDiff); P('df-play', 'df-end', range(19, LY.length - 1, 2), 250);
  on(['th-Phi', 'th-Th', 'th-th'], blkTheory);
  // ▶: first θ from −0.9 to 0.9 (Φ = 0.8, Θ = −0.6, the airline panels), then Φ from 0.8 to −0.4 and Θ from −0.6 to 0.6 together (θ = −0.4, all panels)
  drive('th-play', range(-0.9, 0.9, 0.1).map(t => [0.8, -0.6, t]).concat(range(0, 12, 1).map(k => [+(0.8 - 0.1 * k).toFixed(2), +(-0.6 + 0.1 * k).toFixed(2), -0.4])),
    ([f, T0, t]) => { $('th-Phi').value = f; $('th-Th').value = T0; $('th-th').value = t; blkTheory(); }, 450);
  on(['ai-tr'], blkAirline); P('ai-play', 'ai-tr', keysSorted(AI_TR), 1300);
  on(['af-o'], blkAirFc); P('af-play', 'af-o', range(1952, 1958, 1), 900);
  fillSelect('dg-m', MODEL_ORDER(), NB.gdpm.best_bic, m => m.replace(/\)4$/, ')₄') + (m === NB.gdpm.best_bic ? T.bestBic : '') + (m === NB.gdpm.best_aicc ? T.bestAicc : ''));
  on(['dg-m'], blkGdpDiag); P('dg-play', 'dg-m', [...MODEL_ORDER()].sort(), 1100);
  $('gf-Th').value = D.gdp.fc.Th; on(['gf-Th'], blkGdpFc); P('gf-play', 'gf-Th', range(-0.95, 0, 0.05), 450);
  on(['sa-q'], blkSa); P('sa-play', 'sa-q', range(41, LY.length - 1, 1), 300); from0('sa-play', 'sa-q', 104, LY.length - 1);
  on(['fo-K'], blkFourier); P('fo-play', 'fo-K', range(1, 6, 1), 900);
  on(['ea-w'], blkEaster); P('ea-play', 'ea-w', range(1, 20, 1), 600);
  $('lh-wk').max = LH_N; $('lh-wk').value = Math.round((d2t('2025-01-13') - d2t(LH_W0)) / DAY / 7); on(['lh-wk', 'lh-yr'], blkHourly);
  // ▶: the window moves from January 2022 and the ACF follows the year of the window (2026 has half a year: its ACF stays at 2025)
  drive('lh-play', range(0, LH_N, 4), k => { $('lh-wk').value = k; $('lh-yr').value = String(Math.min(2025, +addD(LH_W0, 7 * k).slice(0, 4))); blkHourly(); }, 350);
  fillSelect('ms-w', keysSorted(D.load.mstl), '2025-02-03', d => T.dmy(d) + (d === '2025-02-03' ? T.lect : '')); on(['ms-w'], blkMstl); P('ms-play', 'ms-w', keysSorted(D.load.mstl), 1100);
  on(['ld-yr'], blkDaily); P('ld-play', 'ld-yr', ['2022', '2023', '2024', '2025', '2026'], 1200);
  on(['pr-K', 'pr-h'], blkProphet); P('pr-play', 'pr-K', ['10', '20', '3', '5'], 1300);
  on(['cs-h', 'cs-st'], blkScheme); P('cs-play', 'cs-st', range(1, 4, 1), 1000);
  fillSelect('lf-o', CV.origins, NB.loadfc.origin, d => T.dmy(d) + (d === NB.loadfc.origin ? T.lect : '')); on(['lf-o'], blkLoadFc); P('lf-play', 'lf-o', CV.origins, 1000);
  on(['cr-m', 'cr-w'], blkCvRes); P('cr-play', 'cr-w', ['all', 'hol', 'nohol'], 1400);
  const ms = CV.models.filter(m => m !== 'Combination');
  fillSelect('cb-1', ms, 'DHR'); fillSelect('cb-2', ms, 'Prophet'); on(['cb-1', 'cb-2'], blkComb); P('cb-play', 'cb-2', [...ms].sort(), 1100);
  const PFX = { series: 'se', 'seasonal-plot': 'sp', 'unit-circle': 'uc', 'gdp-diff': 'df', 'theory-acf': 'th', airline: 'ai', 'airline-fc': 'af', 'gdp-diag': 'dg',
    'gdp-fc': 'gf', 'sa-nsa': 'sa', fourier: 'fo', easter: 'ea', 'load-hourly': 'lh', mstl: 'ms', 'load-daily': 'ld', prophet: 'pr', 'cv-scheme': 'cs',
    'load-fc': 'lf', 'cv-results': 'cr', combination: 'cb' };
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if (has(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: { out: true } }, sp)); });
  const live = XB.filter(([, x]) => has(x + '-out'));
  if (has('gt-Th')) $('gt-Th').value = D.gdp.fc.Th;
  live.forEach(([, x, f, ids, pid, vals, ms]) => { on(ids, f); P(x + '-play', pid, vals, ms); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  ALL.forEach(f => f());
  live.forEach(b => b[2]());
  let w = window.innerWidth, tm = null;
  window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(() => { if ((window.innerWidth < 720) !== (w < 720) || Math.abs(window.innerWidth - w) > 200) { w = window.innerWidth; ALL.forEach(f => f()); live.forEach(b => b[2]()); } }, 250); });
}
window.CH4 = { XB, hegy, hegyFit, airRec, airRow, fourierRow, dmByH, winLoss, dmH, dhrDet, cvSub, allErr, tq, chi2sf, tsf2, seMeans, diffs, theory, airW, diagStats, gdpFc, fourierFit, easterPts, hourAcf, mstlStats, dailyStats, cvTable, dmRaw, holWin, err, jb };
if (!window.CH4_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
