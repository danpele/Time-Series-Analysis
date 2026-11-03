// Chapter 4 interactive page (seasonality and forecasting): one block per chart slide, computed in the browser from ch4_data.js.
(function () {
'use strict';
const D = window.CH4_DATA;
const { sum, solve, acf, pacf, ljungBox, normPpf, mean, sdev, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket } = window.TSAI;
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
  const [a, b, c, d, e, f] = G.ax, S = D.series;
  const tr = [ln(GDP.map((_, i) => qdate(i)), GDP, a, K.blue, '', { line: { color: K.blue, width: 1.1 }, customdata: GDP.map((_, i) => qlab(i)), hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' }),
    ln(mdates(S.ret), S.ret.v, b, K.red, '', { hovertemplate: '%{x|%Y-%m}: %{y:.1f}<extra></extra>' }),
    ln(mdates(S.ipi), S.ipi.v, c, K.green, '', { hovertemplate: '%{x|%Y-%m}: %{y:.1f}<extra></extra>' }),
    ln(mdates(S.tour), S.tour.v, d, K.orange, '', { hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' }),
    ln(mdates(S.infl), S.infl.v, e, K.purple, '', { hovertemplate: '%{x|%Y-%m}: %{y:.2f}%<extra></extra>' }),
    ln(DL.v.map((_, i) => addD(DL.d0, i)), DL.v, f, K.teal, '', { line: { color: K.teal, width: 0.6 }, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' })];
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
  const cols = [K.blue, K.green, K.orange, K.red], q0 = 20, qm = [];
  for (let q = 0; q < 4; q++) {
    const idx = range(q0 + q, GDP.length - 1, 4), v = idx.map(i => GDP[i]), x = v.map((_, j) => q + 0.1 + 0.8 * j / (v.length - 1)); qm.push(mean(v));
    tr.push(ln(x, v, b, cols[q], `Q${q + 1}`, { line: { color: cols[q], width: 1.1 }, customdata: idx.map(qlab), hovertemplate: '%{customdata}: %{y:.1f}<extra></extra>' }));
    G.lay.shapes.push({ type: 'line', xref: b.xa, yref: b.ya, x0: q + 0.1, x1: q + 0.9, y0: qm[q], y1: qm[q], line: { color: K.navy, width: 2 } });
  }
  tr.push(leg('quarter mean', { color: K.navy, width: 2 }));
  plot('sp-chart', tr, G.lay);
  let ok = 0, n = 0; for (let y = 2005; y <= 2025; y++) { const i = 4 * (y - 2000), v = GDP.slice(i, i + 4); n++; if (v[0] < v[1] && v[1] < v[2] && v[2] < v[3]) ok++; }
  const s = byY[yr], mx = s.indexOf(Math.max(...s)), mn = s.indexOf(Math.min(...s));
  out('sp-out', [T.sp1(yr, nf(s[7], 2), nf(s[0], 2), nf(s[7] / s[0], 1), nf(s[3], 3), T.mon[mx], T.mon[mn]), T.sp2(qm.map(v => nf(v, 1)), ok, n)]);
}

// ================================================================= 3. roots of 1 - z^s
function blkCircle() {
  const s = +$('uc-s').value;
  const G = grid('uc-chart', [{ r: 0, c: 0, t: 's = 4: 4 roots of 1 − z<sup>4</sup>', x: { range: [-1.45, 1.45], zeroline: false }, y: { range: [-1.45, 1.45], scaleanchor: 'x', zeroline: false } },
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
  const G = grid('ai-chart', [{ r: 0, c: 0, t: 'Airline passengers, 1949-1960', x: { type: 'date' } }, { r: 0, c: 1, t: `ACF of ${nm}`, xt: 'lag (months)', y: { range: yr } },
    { r: 0, c: 2, t: `PACF of ${nm}`, xt: 'lag (months)', y: { range: yr } }], { cols: 3, h: 380, legendRows: 2, gx: 0.075, mr: 30 });
  const [a, b1, c] = G.ax, x = AIR.map((_, i) => ym('1949-01', i) + '-01'), tr = [];
  tr.push(ln(x, AIR, a, K.blue, 'passengers (thousands)', { line: { color: K.blue, width: 1.1 }, hovertemplate: '%{x|%Y-%m}: %{y:.0f}<extra></extra>' }));
  G.lay.yaxis4 = AX('', { overlaying: 'y', side: 'right', anchor: 'x', automargin: false, showgrid: false, tickfont: { size: 10, color: K.orange } });
  tr.push({ x, y: AIRL, mode: 'lines', line: { color: K.orange, width: 1 }, name: 'log scale (right axis)', xaxis: 'x', yaxis: 'y4', hovertemplate: '%{x|%Y-%m}: %{y:.3f}<extra></extra>' });
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

// ================================================================= wiring
const ALL = [blkSeries, blkSplot, blkCircle, blkGdpDiff, blkTheory, blkAirline, blkAirFc, blkGdpDiag, blkGdpFc, blkSa, blkFourier, blkEaster, blkHourly, blkMstl,
  blkDaily, blkProphet, blkScheme, blkLoadFc, blkCvRes, blkComb];
function fillSelect(id, vals, sel, lab) { const s = $(id); s.innerHTML = vals.map(v => `<option value="${v}"${v === sel ? ' selected' : ''}>${lab ? lab(v) : v}</option>`).join(''); }
function init() {
  const has = id => !!$(id), on = (ids, f) => ids.forEach(id => $(id).addEventListener('input', f));
  const players = {}, P = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  seDefaults(); $('se-s').addEventListener('input', () => { seDefaults(); blkSeries(); }); on(['se-y0', 'se-y1'], blkSeries); P('se-play', 'se-y1', range(2000, 2026, 1), 450);
  on(['sp-y'], blkSplot); P('sp-play', 'sp-y', range(2012, 2025, 1), 600);
  on(['uc-s'], blkCircle); P('uc-play', 'uc-s', ['2', '3', '4', '6', '7', '12', '24'], 1100);
  on(['df-end'], blkGdpDiff); P('df-play', 'df-end', range(19, LY.length - 1, 2), 250);
  on(['th-Phi', 'th-Th', 'th-th'], blkTheory); P('th-play', 'th-th', range(-0.9, 0.9, 0.1), 450);
  on(['ai-tr'], blkAirline); P('ai-play', 'ai-tr', keysSorted(AI_TR), 1300);
  on(['af-o'], blkAirFc); P('af-play', 'af-o', range(1952, 1958, 1), 900);
  fillSelect('dg-m', MODEL_ORDER(), NB.gdpm.best_bic, m => m.replace(/\)4$/, ')₄') + (m === NB.gdpm.best_bic ? T.bestBic : '') + (m === NB.gdpm.best_aicc ? T.bestAicc : ''));
  on(['dg-m'], blkGdpDiag); P('dg-play', 'dg-m', [...MODEL_ORDER()].sort(), 1100);
  $('gf-Th').value = D.gdp.fc.Th; on(['gf-Th'], blkGdpFc); P('gf-play', 'gf-Th', range(-0.95, 0, 0.05), 450);
  on(['sa-q'], blkSa); P('sa-play', 'sa-q', range(41, LY.length - 1, 1), 300);
  on(['fo-K'], blkFourier); P('fo-play', 'fo-K', range(1, 6, 1), 900);
  on(['ea-w'], blkEaster); P('ea-play', 'ea-w', range(1, 20, 1), 600);
  $('lh-wk').max = LH_N; $('lh-wk').value = Math.round((d2t('2025-01-13') - d2t(LH_W0)) / DAY / 7); on(['lh-wk', 'lh-yr'], blkHourly); P('lh-play', 'lh-wk', range(0, LH_N, 4), 500);
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
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  ALL.forEach(f => f());
  let w = window.innerWidth, tm = null;
  window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(() => { if ((window.innerWidth < 720) !== (w < 720) || Math.abs(window.innerWidth - w) > 200) { w = window.innerWidth; ALL.forEach(f => f()); } }, 250); });
}
window.CH4 = { chi2sf, tsf2, seMeans, diffs, theory, airW, diagStats, gdpFc, fourierFit, easterPts, hourAcf, mstlStats, dailyStats, cvTable, dmRaw, holWin, err, jb };
if (!window.CH4_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
