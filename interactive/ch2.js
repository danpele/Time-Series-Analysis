// Chapter 2 interactive page (ARMA models): one block per chart slide, computed in the browser from ch2_data.js.
(function () {
'use strict';
const D = window.CH2_DATA;
const { sum, solve, acf, pacf, ljungBox, armaSim, normPpf, quantile, rng, mean, nf, pct, $, AX, layout, plot, out, Player, range,
  predict, exitTicket } = window.TSAI;
const T = window.CH2_TEXT;

// ---------------------------------------------------------------- colours of generate_all_charts.py / tsa_style.py
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', orange: '#E67E22', purple: '#8E44AD',
  teal: '#17A2B8', navy: '#1F2A44' };

// ---------------------------------------------------------------- small helpers
const lags = n => Array.from({ length: n }, (_, i) => i + 1);
const pvar = x => { const m = mean(x); return sum(x.map(v => (v - m) ** 2)) / x.length; };
const fp = p => (p < 0.001 ? '< ' + nf(0.001, 3) : nf(p, 3));
const pe = p => (p < 0.001 ? 'p < ' + nf(0.001, 3) : 'p = ' + nf(p, 3));     // 'p = 0,375' or 'p < 0,001'
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
function psiW(phi, theta, n) {    // psi_j = theta_j + sum_i phi_i psi_{j-i}, psi_0 = 1
  const p = [1];
  for (let j = 1; j < n; j++) { let v = j <= theta.length ? theta[j - 1] : 0; for (let i = 0; i < Math.min(j, phi.length); i++) v += phi[i] * p[j - 1 - i]; p.push(v); }
  return p;
}
function acfTheo(phi, theta, k) { // rho(h) = sum psi_j psi_{j+h} / sum psi_j^2 (causal representation, 3000 terms)
  const ps = psiW(phi, theta, 3000 + k), g = h => { let s = 0; for (let j = 0; j < 3000; j++) s += ps[j] * ps[j + h]; return s; }, g0 = g(0);
  return lags(k).map(h => g(h) / g0);
}
const ar2Stat = (a, b) => b > -1 && a + b < 1 && b - a < 1;
function ar2Roots(a, b) {         // roots of 1 - a z - b z^2 and inverse roots (lambda^2 - a lambda - b = 0)
  const d = a * a + 4 * b;
  if (d >= 0) { const l = [(a + Math.sqrt(d)) / 2, (a - Math.sqrt(d)) / 2]; return { complex: false, inv: l.map(v => [v, 0]), mod: l.map(Math.abs) }; }
  const re = a / 2, im = Math.sqrt(-d) / 2, m = Math.hypot(re, im);
  return { complex: true, inv: [[re, im], [re, -im]], mod: [m, m] };
}
const golden = (f, lo, hi, tol = 1e-7) => {   // maximiser of a unimodal f on [lo, hi]
  const r = (Math.sqrt(5) - 1) / 2; let a = lo, b = hi, c = b - r * (b - a), d = a + r * (b - a), fc = f(c), fd = f(d);
  while (b - a > tol) { if (fc > fd) { b = d; d = c; fd = fc; c = b - r * (b - a); fc = f(c); } else { a = c; c = d; fc = fd; d = a + r * (b - a); fd = f(d); } }
  return (a + b) / 2;
};
const q2d = s => s.slice(0, 4) + '-' + String(3 * +s[5] - 2).padStart(2, '0') + '-01';      // '2020Q2' -> '2020-04-01'
const m2d = s => s + '-01';
const addM = (s, k) => { const y = +s.slice(0, 4), m = +s.slice(5, 7) - 1 + k; return `${y + Math.floor(m / 12)}-${String(m % 12 + 1).padStart(2, '0')}-01`; };
const addQ = (s, k) => { const i = +s.slice(0, 4) * 4 + (+s[5] - 1) + k; return `${Math.floor(i / 4)}Q${i % 4 + 1}`; };

// ---------------------------------------------------------------- panels: one Plotly figure with several axes; stacked on narrow screens
// cells: [{ r, c, cs, t (title), xt, yt (axis titles), x, y (extra axis options) }]
function grid(id, cells, o) {   // eslint-disable-line no-param-reassign
  const el = $(id), narrow = el.clientWidth < 560, maxc = Math.floor(el.clientWidth / 6.6);
  const wrap = t => { if (!t || t.replace(/<[^>]+>/g, '').length <= maxc) return t; const w = t.split(' '), L = ['']; w.forEach(x => { const c = L[L.length - 1];
    if ((c + ' ' + x).replace(/<[^>]+>/g, '').trim().length > maxc && c) L.push(x); else L[L.length - 1] = (c + ' ' + x).trim(); }); return L.join('<br>'); };
  if (narrow) cells = cells.map(c => Object.assign({}, c, { t: wrap(c.t) }));
  const extra = narrow ? Math.max(0, ...cells.map(c => (c.t || '').split('<br>').length - 1)) * 15 : 0;
  const C = narrow ? 1 : o.cols, R = narrow ? cells.length : Math.max(...cells.map(c => c.r)) + 1;
  const H = narrow ? Math.max(o.h || 360, 120 + (210 + extra) * cells.length) : (o.h || 360);
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
function acfPanel(tr, lay, a, r, T0, color, theo, band = true) {
  tr.push({ x: lags(r.length), y: r, type: 'bar', width: 0.55, marker: { color }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' });
  if (theo) tr.push({ x: lags(theo.length), y: theo, mode: 'markers', marker: { color: K.orange, size: 6 }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: 'theory, lag %{x}: %{y:.3f}<extra></extra>' });
  if (band) { const b = 1.96 / Math.sqrt(T0); lay.shapes.push(hl(a, b, K.red, 'dash', 1), hl(a, -b, K.red, 'dash', 1)); }
  lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.8));
}
const BAND = leg('±1.96/√T', { color: K.red, dash: 'dash', width: 1.2 });
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };

// ================================================================= AR(1) for three values of phi
function ar1Path(phi) { const e = D.ar1_e, x = new Array(e.length).fill(0); for (let t = 1; t < e.length; t++) x[t] = phi * x[t - 1] + e[t]; return x.slice(300); }
const A1S = [0.9, 0.3, -0.7];       // slide values; during ▶ the middle and right phi move in proportion to the left one
function blkAR1(ev) {
  if (ev && !ev.isTrusted && ev.target && ev.target.id === 'a1-p1') { const v = +ev.target.value;
    $('a1-p2').value = +(v * A1S[1] / A1S[0]).toFixed(2); $('a1-p3').value = +(v * A1S[2] / A1S[0]).toFixed(2); }
  const ph = ['a1-p1', 'a1-p2', 'a1-p3'].map(id => +$(id).value), cols = [K.blue, K.green, K.red];
  ph.forEach((p, i) => set(['a1-p1', 'a1-p2', 'a1-p3'][i], nf(p, 2)));
  const X = ph.map(ar1Path), big = Math.max(...X.flat().map(Math.abs)) > 7;
  const cells = ph.map((p, j) => ({ r: 0, c: j, t: `AR(1), φ = ${p}`, y: big ? {} : { range: [-7, 7] } }))
    .concat(ph.map((p, j) => ({ r: 1, c: j, t: `ACF, φ = ${p}`, xt: 'lag h', y: { range: [-1, 1] } })));
  const G = grid('a1-chart', cells, { cols: 3, h: 440 }), tr = [], res = [];
  X.forEach((x, j) => {
    const a = G.ax[j], b = G.ax[3 + j], r = acf(x, 15), th = lags(15).map(h => ph[j] ** h);
    tr.push({ x: lags(x.length), y: x, mode: 'lines', line: { color: cols[j], width: 1 }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: 't = %{x}: %{y:.2f}<extra></extra>' });
    G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.8));
    acfPanel(tr, G.lay, b, r, x.length, cols[j], th);
    res.push(T.a1(nf(ph[j], 2), nf(r[0], 2), nf(r[1], 2), nf(pvar(x), 2), Math.abs(ph[j]) < 1 ? nf(1 / (1 - ph[j] ** 2), 2) : '∞'));
  });
  tr.push(leg('sample ACF', null, { color: K.blue, symbol: 'square', size: 10 }), leg('theory φ^h', null, { color: K.orange, size: 7 }), BAND);
  plot('a1-chart', tr, G.lay);
  out('a1-out', res);
}

// ================================================================= AR(2): triangle, inverse roots, ACF
const AR2 = () => D.numbers.ar2;
const AR2COL = { A: K.blue, B: K.red, C: K.purple, Sunspots: K.amber };
function blkAR2() {
  const a = +$('a2-f1').value, b = +$('a2-f2').value; set('a2-f1', nf(a, 2)); set('a2-f2', nf(b, 2));
  const ex = AR2(), names = ['A', 'B', 'C', 'Sunspots'];
  const same = names.find(k => Math.abs(ex[k].phi[0] - a) < 1e-9 && Math.abs(ex[k].phi[1] - b) < 1e-9) ||
    names.find(k => Math.abs(ex[k].phi[0] - a) < 0.005 && Math.abs(ex[k].phi[1] - b) < 0.005);
  const G = grid('a2-chart', [{ r: 0, c: 0, t: 'Stationarity triangle', xt: 'φ<sub>1</sub>', yt: 'φ<sub>2</sub>', x: { range: [-2.2, 2.2] }, y: { range: [-1.2, 1.2] } },
    { r: 0, c: 1, t: 'Inverse roots and the unit circle', x: { range: [-1.35, 1.35], zeroline: true }, y: { range: [-1.35, 1.35] } },
    { r: 0, c: 2, t: 'Theoretical ACF of A and ' + (same && same !== 'A' ? same : 'your model'), xt: 'lag h' }], { cols: 3, h: 380, legendRows: 2 });
  const [t1, t2, t3] = G.ax, tr = [];
  G.lay['yaxis2'].scaleanchor = 'x2';
  const f1 = range(-2, 2, 0.02);
  tr.push({ x: [-2, 0, 2, -2], y: [-1, 1, -1, -1], fill: 'toself', fillcolor: 'rgba(23,162,184,0.12)', mode: 'lines', line: { color: K.teal, width: 1.5 }, name: 'stationary region', xaxis: t1.xa, yaxis: t1.ya, hoverinfo: 'skip' });
  tr.push({ x: f1, y: f1.map(v => -v * v / 4), mode: 'lines', line: { color: K.green, dash: 'dash', width: 1.3 }, name: 'φ<sub>1</sub>² + 4φ<sub>2</sub> = 0', xaxis: t1.xa, yaxis: t1.ya, hoverinfo: 'skip' });
  G.lay.annotations.push({ xref: t1.xa, yref: t1.ya, x: -0.75, y: 0.12, text: 'real roots', showarrow: false, font: { size: 11, color: K.navy } },
    { xref: t1.xa, yref: t1.ya, x: 0, y: -0.75, text: 'complex roots', showarrow: false, font: { size: 11, color: K.navy } });
  const th = range(0, 2 * Math.PI, 0.02);
  tr.push({ x: th.map(Math.cos), y: th.map(Math.sin), mode: 'lines', line: { color: K.navy, width: 1 }, xaxis: t2.xa, yaxis: t2.ya, showlegend: false, hoverinfo: 'skip' });
  names.forEach(k => {
    const [p1, p2] = ex[k].phi, R = ar2Roots(p1, p2);
    tr.push({ x: [p1], y: [p2], mode: 'markers+text', text: [k], textposition: 'top right', textfont: { color: AR2COL[k], size: 12 }, marker: { color: AR2COL[k], size: 10 }, name: k, legendgroup: k,
      xaxis: t1.xa, yaxis: t1.ya, hovertemplate: `${k}: φ = (${p1.toFixed(3)}, ${p2.toFixed(3)})<extra></extra>` });
    tr.push({ x: R.inv.map(z => z[0]), y: R.inv.map(z => z[1]), mode: 'markers', marker: { color: AR2COL[k], size: 10 }, legendgroup: k, showlegend: false, xaxis: t2.xa, yaxis: t2.ya,
      hovertemplate: `${k}: λ = %{x:.3f} %{y:+.3f}i<extra></extra>` });
  });
  const R = ar2Roots(a, b), st = ar2Stat(a, b);
  tr.push({ x: [a], y: [b], mode: 'markers', marker: { color: K.navy, size: 13, symbol: 'x-thin', line: { color: K.navy, width: 3 } }, name: 'your model', legendgroup: 'you', xaxis: t1.xa, yaxis: t1.ya, hovertemplate: 'φ = (%{x:.2f}, %{y:.2f})<extra></extra>' });
  tr.push({ x: R.inv.map(z => z[0]), y: R.inv.map(z => z[1]), mode: 'markers', marker: { color: K.navy, size: 13, symbol: 'x-thin', line: { color: K.navy, width: 3 } }, legendgroup: 'you', showlegend: false, xaxis: t2.xa, yaxis: t2.ya, hovertemplate: 'λ = %{x:.3f} %{y:+.3f}i<extra></extra>' });
  const rA = acfTheo(ex.A.phi, [], 30);
  tr.push({ x: lags(30), y: rA, mode: 'lines+markers', marker: { size: 5, color: K.blue }, line: { color: K.blue, width: 1.2 }, xaxis: t3.xa, yaxis: t3.ya, showlegend: false, hovertemplate: 'A, lag %{x}: %{y:.3f}<extra></extra>' });
  if (st) { const c = same ? AR2COL[same] : K.navy; tr.push({ x: lags(30), y: acfTheo([a, b], [], 30), mode: 'lines+markers', marker: { size: 5, color: c }, line: { color: c, width: 1.2 }, xaxis: t3.xa, yaxis: t3.ya, showlegend: false, hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' }); }
  else G.lay.annotations.push({ xref: t3.xd, yref: t3.yd, x: 0.98, y: 0.98, xanchor: 'right', yanchor: 'top', text: 'not stationary: no ACF', showarrow: false, bgcolor: 'rgba(255,255,255,0.85)', font: { color: K.red, size: 12 } });
  G.lay.shapes.push(hl(t2, 0, K.navy, 'solid', 0.5), vl(t2, 0, K.navy, 'solid', 0.5), hl(t3, 0, K.navy, 'solid', 0.8));
  plot('a2-chart', tr, G.lay);
  const zr = R.complex ? `${nf(a / (2 * -b), 3)} ± ${nf(Math.sqrt(-(a * a + 4 * b)) / (2 * -b), 3)} i` : R.inv.map(z => (Math.abs(z[0]) < 1e-12 ? '∞' : nf(1 / z[0], 2))).join(T.and);
  const lines = [T.a2a(nf(a, 2), nf(b, 2), zr, R.mod.map(m => nf(m, 3)).join(T.and), st)];
  if (R.complex && st) lines.push(T.a2c(nf(2 * Math.PI / Math.acos(a / (2 * Math.sqrt(-b))), 1), nf(Math.sqrt(-b), 3)));
  if (st) lines.push(T.a2r(nf(a / (1 - b), 3)));
  out('a2-out', lines);
}

// ================================================================= MA(1): theta and -theta
function blkMA1() {
  const th0 = +$('m1-th').value; set('m1-th', nf(th0, 2));
  const e = D.ma1_e, rows = [[th0, K.blue], [-th0, K.red]], cells = [], tr = [], res = [];
  const X = rows.map(([th]) => e.slice(1).map((v, i) => v + th * e[i])), big = Math.max(...X.flat().map(Math.abs)) > 5;
  rows.forEach(([th], i) => cells.push({ r: i, c: 0, t: `MA(1), θ = ${+th.toFixed(2)}`, y: big ? {} : { range: [-5, 5] } }, { r: i, c: 1, t: 'ACF', y: { range: [-0.65, 0.65] }, xt: i ? 'lag h' : '' },
    { r: i, c: 2, t: 'PACF', y: { range: [-0.65, 0.65] }, xt: i ? 'lag h' : '' }));
  const G = grid('m1-chart', cells, { cols: 3, h: 440 });
  rows.forEach(([th, c], i) => {
    const x = X[i], r = acf(x, 12), pc = pacf(r), rt = [th / (1 + th * th)].concat(new Array(11).fill(0)), pt = pacf(rt);
    const [a, b, d] = G.ax.slice(3 * i, 3 * i + 3);
    tr.push({ x: lags(x.length), y: x, mode: 'lines', line: { color: c, width: 0.9 }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: 't = %{x}: %{y:.2f}<extra></extra>' });
    acfPanel(tr, G.lay, b, r, x.length, c, rt); acfPanel(tr, G.lay, d, pc, x.length, c, pt);
    res.push(T.m1(nf(th, 2), nf(rt[0], 3), nf(r[0], 2), nf(pc[1], 2), Math.abs(th) <= 1));
  });
  tr.push(leg('sample', null, { color: K.blue, symbol: 'square', size: 10 }), leg('theoretical', null, { color: K.orange, size: 7 }), BAND);
  plot('m1-chart', tr, G.lay);
  out('m1-out', res);
}

// ================================================================= rho(1) of an MA(1)
function blkMArho() {
  const t0 = +$('mr-th').value; set('mr-th', nf(t0, 2));
  const th = range(-4, 4, 0.01), f = t => t / (1 + t * t);
  const G = grid('mr-chart', [{ r: 0, c: 0, t: 'MA(1): θ and 1/θ give the same autocorrelation', xt: 'θ', x: { range: [-4.1, 4.1] }, y: { range: [-0.6, 0.6] } }], { cols: 1, h: 360 });
  const a = G.ax[0];
  G.lay.shapes.push({ type: 'rect', xref: a.xa, yref: a.yd, x0: -1, x1: 1, y0: 0, y1: 1, fillcolor: 'rgba(46,125,50,0.10)', line: { width: 0 }, layer: 'below' },
    hl(a, 0.5, K.red, 'dash', 1), hl(a, -0.5, K.red, 'dash', 1), hl(a, 0, K.navy, 'solid', 0.8));
  const tr = [{ x: th, y: th.map(f), mode: 'lines', line: { color: K.blue, width: 2 }, name: 'ρ(1) = θ/(1 + θ²)', xaxis: a.xa, yaxis: a.ya, hovertemplate: 'θ = %{x:.2f}: ρ(1) = %{y:.3f}<extra></extra>' },
    leg('±0.5: the largest possible |ρ(1)|', { color: K.red, dash: 'dash', width: 1.2 }), leg('invertible: |θ| < 1', null, { color: 'rgba(46,125,50,0.25)', symbol: 'square', size: 14 })];
  const pts = [[t0, K.orange]]; if (Math.abs(t0) > 1e-9 && Math.abs(Math.abs(t0) - 1) > 1e-9) pts.push([1 / t0, K.purple]);
  pts.forEach(([t, c]) => { if (Math.abs(t) <= 4.1) tr.push({ x: [t], y: [f(t)], mode: 'markers', marker: { color: c, size: 12 }, name: `θ = ${+t.toFixed(2)}: ρ(1) = ${+f(t).toFixed(3)}`, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'θ = %{x:.3f}: ρ(1) = %{y:.3f}<extra></extra>' }); });
  plot('mr-chart', tr, G.lay);
  const inv = Math.abs(t0) < 1 ? t0 : Math.abs(t0) > 1 ? 1 / t0 : null;
  out('mr-out', [T.mr(nf(t0, 2), nf(f(t0), 3), Math.abs(t0) > 1e-9 ? nf(1 / t0, 2) : null, inv === null ? null : nf(inv, 2), Math.abs(Math.abs(t0) - 1) < 1e-9)]);
}

// ================================================================= psi weights
const PSI = { 'AR(1)': { phi: [0.8, 0], th: [0, 0], use: [1, 0, 0, 0], col: K.blue }, 'AR(2)': { phi: [1.0, -0.6], th: [0, 0], use: [1, 1, 0, 0], col: K.red },
  'MA(2)': { phi: [0, 0], th: [0.6, 0.3], use: [0, 0, 1, 1], col: K.green }, 'ARMA(1,1)': { phi: [0.7, 0], th: [0.4, 0], use: [1, 0, 1, 0], col: K.purple } };
const psiPar = k => { const m = PSI[k]; return { phi: m.phi.filter((_, i) => m.use[i]), th: m.th.filter((_, i) => m.use[2 + i]) }; };
const psiLab = k => { const { phi, th } = psiPar(k), f = v => +v.toFixed(2); return k === 'AR(1)' ? `AR(1), φ = ${f(phi[0])}` : k === 'AR(2)' ? `AR(2), φ = (${f(phi[0])}, ${f(phi[1])})`
  : k === 'MA(2)' ? `MA(2), θ = (${f(th[0])}, ${f(th[1])})` : `ARMA(1,1), φ = ${f(phi[0])}, θ = ${f(th[0])}`; };
function psiSync() {             // the sliders show the parameters of the selected model
  const m = PSI[$('ps-m').value], ids = ['ps-a', 'ps-b', 'ps-c', 'ps-d'], v = m.phi.concat(m.th);
  ids.forEach((id, i) => { $(id).disabled = !m.use[i]; $(id).value = v[i]; set(id, m.use[i] ? nf(v[i], 2) : '—'); });
}
function blkPsi(ev) {
  const k = $('ps-m').value, m = PSI[k], ids = ['ps-a', 'ps-b', 'ps-c', 'ps-d'];
  if (ev && ev.target && ids.includes(ev.target.id)) { const i = ids.indexOf(ev.target.id), v = +ev.target.value; (i < 2 ? m.phi : m.th)[i % 2] = v; set(ev.target.id, nf(v, 2));
    if (!ev.isTrusted && i === 0) { PSI['AR(1)'].phi[0] = v; PSI['AR(2)'].phi[0] = v; PSI['MA(2)'].th[0] = v; PSI['ARMA(1,1)'].phi[0] = v; } }
  const names = Object.keys(PSI), n = 16;
  const G = grid('ps-chart', names.map((nm, j) => ({ r: 0, c: j, t: psiLab(nm), xt: 'j', yt: j ? '' : 'ψ<sub>j</sub>', y: { range: [-1.2, 2.2] } })), { cols: 4, h: 340, legend: false, gx: 0.05 });
  const tr = [], ys = [];
  names.forEach((nm, j) => { const { phi, th } = psiPar(nm), p = psiW(phi, th, n); ys.push(...p);
    tr.push({ x: range(0, n - 1, 1), y: p, type: 'bar', width: 0.6, marker: { color: PSI[nm].col }, xaxis: G.ax[j].xa, yaxis: G.ax[j].ya, showlegend: false, hovertemplate: 'ψ<sub>%{x}</sub> = %{y:.3f}<extra></extra>' });
    G.lay.shapes.push(hl(G.ax[j], 0, K.navy, 'solid', 0.8)); });
  const lo = Math.min(-1.2, ...ys), hi = Math.max(2.2, ...ys);
  G.ax.forEach((a, j) => { G.lay['yaxis' + (j ? j + 1 : '')].range = [Math.max(lo, -6), Math.min(hi, 6)]; });
  plot('ps-chart', tr, G.lay);
  const { phi, th } = psiPar(k), p = psiW(phi, th, 400), st = phi.length < 2 ? Math.abs(phi[0] || 0) < 1 : ar2Stat(phi[0], phi[1]);
  const lab = k + ', ' + (phi.length ? 'φ = ' + phi.map(v => nf(v, 2)).join('; ') : '') + (phi.length && th.length ? ', ' : '') + (th.length ? 'θ = ' + th.map(v => nf(v, 2)).join('; ') : '');
  out('ps-out', [T.ps(lab, p.slice(1, 6).map(v => nf(v, 2)).join('; '), st ? nf(sum(p.map(v => v * v)), 2) : null)]);
}

// ================================================================= theoretical and sample ACF / PACF
let ptSeed = 0;
function ptPath(name, Tn) {
  const { phi, theta } = D.models[name];
  if (!ptSeed) { const e = D.pat_e[name], x = new Array(e.length).fill(0);
    for (let t = 0; t < e.length; t++) { let v = e[t]; phi.forEach((f, i) => { if (t - 1 - i >= 0) v += f * x[t - 1 - i]; }); theta.forEach((f, j) => { if (t - 1 - j >= 0) v += f * e[t - 1 - j]; }); x[t] = v; }
    return x.slice(500, 500 + Tn); }
  const z = rng(1000 * ptSeed + Object.keys(D.models).indexOf(name)); return armaSim(phi, theta, 500, z, 500).slice(0, Tn);
}
function blkPatterns() {
  const Tn = +$('pt-T').value; set('pt-T', Tn);
  const names = Object.keys(D.models), cols = [K.blue, K.red, K.green];
  const cells = names.map((n, j) => ({ r: 0, c: j, t: n + ': ACF', y: { range: [-0.8, 1] } })).concat(names.map((n, j) => ({ r: 1, c: j, t: n + ': PACF', xt: 'lag h', y: { range: [-0.8, 1] } })));
  const G = grid('pt-chart', cells, { cols: 3, h: 440 }), tr = [], res = [];
  names.forEach((n, j) => {
    const { phi, theta } = D.models[n], x = ptPath(n, Tn), r = acf(x, 15), pc = pacf(r), rt = acfTheo(phi, theta, 15), pt = pacf(rt);
    acfPanel(tr, G.lay, G.ax[j], r, Tn, cols[j], rt); acfPanel(tr, G.lay, G.ax[3 + j], pc, Tn, cols[j], pt);
    const b = 1.96 / Math.sqrt(Tn), outA = r.map((v, i) => (Math.abs(v) > b && Math.abs(rt[i]) < 0.02 ? i + 1 : 0)).filter(Boolean);
    res.push(T.pt(n, nf(r[0], 2), nf(rt[0], 2), nf(r[1], 2), nf(rt[1], 2), nf(pc[1], 2), nf(pt[1], 2), nf(b, 2)));
  });
  tr.push(leg('sample', null, { color: K.blue, symbol: 'square', size: 10 }), leg('theoretical', null, { color: K.orange, size: 7 }), BAND);
  plot('pt-chart', tr, G.lay);
  out('pt-out', res);
}

// ================================================================= sampling distributions of the estimators
function ar1Est(x) {
  const n = x.length, yw = acf(x, 1)[0];
  let mz = 0, my = 0; for (let t = 1; t < n; t++) { mz += x[t - 1]; my += x[t]; } mz /= n - 1; my /= n - 1;
  let sxy = 0, sxx = 0; for (let t = 1; t < n; t++) { sxy += (x[t - 1] - mz) * (x[t] - my); sxx += (x[t - 1] - mz) ** 2; }
  const L = phi => { const a = 1 - phi * phi; let s1 = 0; for (let t = 1; t < n; t++) s1 += x[t] - phi * x[t - 1];
    const mu = (a * x[0] + (1 - phi) * s1) / (a + (n - 1) * (1 - phi) ** 2); let S = a * (x[0] - mu) ** 2;
    for (let t = 1; t < n; t++) S += (x[t] - mu - phi * (x[t - 1] - mu)) ** 2; return -n / 2 * Math.log(S / n) + 0.5 * Math.log(a); };
  return [yw, sxy / sxx, golden(L, -0.999, 0.999, 1e-6)];
}
function ma1Est(x) {
  const n = x.length, m = mean(x), xc = x.map(v => v - m);
  const ssr = th => { let e = 0, s = 0; for (const v of xc) { e = v - th * e; s += e * e; } return -s; };
  const L = th => {               // exact Gaussian likelihood (innovations algorithm), mean profiled out by GLS
    let v = 1 + th * th, ux = x[0], u1 = 1, sv = Math.log(v), a = ux * u1 / v, b = u1 * u1 / v, c = ux * ux / v;
    for (let t = 1; t < n; t++) { const k = th / v; v = 1 + th * th - th * k; ux = x[t] - k * ux; u1 = 1 - k * u1; sv += Math.log(v); a += ux * u1 / v; b += u1 * u1 / v; c += ux * ux / v; }
    const S = c - a * a / b; return -n / 2 * Math.log(S / n) - sv / 2; };
  return [golden(ssr, -0.99, 0.99, 1e-6), golden(L, -0.999, 0.999, 1e-6)];
}
const esCache = {};
function esSim(phi, th, Tn) {
  const key = [phi, th, Tn].join('|');
  if (!esCache[key]) {
    const E = D.est;
    if (phi === E.phi && th === E.theta && Tn === E.T) esCache[key] = { ar: [E.ar.yw, E.ar.cls, E.ar.ml], ma: [E.ma.cls, E.ma.ml], lecture: true };
    else {
      const z = rng(2026 + Math.round(1000 * phi) + 7 * Math.round(100 * th) + Tn), A = [[], [], []], M = [[], []];
      for (let s = 0; s < 400; s++) { const e = ar1Est(armaSim([phi], [], Tn, z, 500)); e.forEach((v, i) => A[i].push(v)); }
      for (let s = 0; s < 400; s++) { const e = ma1Est(armaSim([], [th], Tn, z, 500)); e.forEach((v, i) => M[i].push(v)); }
      esCache[key] = { ar: A, ma: M, lecture: false };
    }
  }
  return esCache[key];
}
function stepHist(v, lo, hi, nb) {
  const w = (hi - lo) / nb, c = new Array(nb).fill(0);
  v.forEach(x => { const k = Math.floor((x - lo) / w); if (k >= 0 && k < nb) c[k]++; else if (x === hi) c[nb - 1]++; });
  const xs = [lo], ys = [0]; c.forEach((n, k) => { xs.push(lo + k * w, lo + (k + 1) * w); ys.push(n, n); }); xs.push(hi); ys.push(0);
  return { x: xs, y: ys };
}
function blkEst() {
  const phi = +$('es-phi').value, th = +$('es-th').value, Tn = +$('es-T').value; set('es-phi', nf(phi, 2)); set('es-th', nf(th, 1));
  const S = esSim(phi, th, Tn), sd = a => Math.sqrt(pvar(a));
  const lecture = S.lecture, aLo = lecture ? 0.55 : Math.min(...S.ar.flat()) - 0.01, aHi = lecture ? 1.05 : Math.max(...S.ar.flat()) + 0.01;
  const mLo = lecture ? 0.05 : Math.min(...S.ma.flat()) - 0.01, mHi = lecture ? 0.95 : Math.max(...S.ma.flat()) + 0.01;
  const G = grid('es-chart', [{ r: 0, c: 0, t: `AR(1), φ = ${phi}, T = ${Tn}: 400 samples`, xt: 'estimate of φ', yt: 'samples' },
    { r: 0, c: 1, t: `MA(1), θ = ${th}, T = ${Tn}: conditional LS and ML`, xt: 'estimate of θ' }], { cols: 2, h: 370 });
  const tr = [], names = ['Yule-Walker', 'conditional LS', 'Gaussian ML'], cols = [K.orange, K.blue, K.red];
  S.ar.forEach((v, i) => { const h = stepHist(v, aLo, aHi, 39); tr.push({ x: h.x, y: h.y, mode: 'lines', line: { color: cols[i], width: 1.8, shape: 'hv' }, name: names[i], legendgroup: names[i], xaxis: G.ax[0].xa, yaxis: G.ax[0].ya, hovertemplate: names[i] + ': %{y} samples from %{x:.3f}<extra></extra>' }); });
  S.ma.forEach((v, i) => { const h = stepHist(v, mLo, mHi, 39); tr.push({ x: h.x, y: h.y, mode: 'lines', line: { color: cols[i + 1], width: 1.8, shape: 'hv' }, legendgroup: names[i + 1], showlegend: false, xaxis: G.ax[1].xa, yaxis: G.ax[1].ya, hovertemplate: names[i + 1] + ': %{y} samples from %{x:.3f}<extra></extra>' }); });
  tr.push(leg('true value', { color: K.navy, dash: 'dash', width: 1.2 }));
  G.lay.shapes.push(vl(G.ax[0], phi, K.navy, 'dash'), vl(G.ax[1], th, K.navy, 'dash'));
  plot('es-chart', tr, G.lay);
  const ms = a => `${nf(mean(a), 3)} (${nf(sd(a), 3)})`;
  out('es-out', [T.es0(lecture), T.esA(...S.ar.map(ms), nf(Math.sqrt(Math.max(0, 1 - phi * phi) / Tn), 3), nf(-(1 + 3 * phi) / Tn, 3)),
    T.esM(...S.ma.map(ms), nf(Math.sqrt(Math.max(0, 1 - th * th) / Tn), 3))]);
}

// ================================================================= AIC and BIC: how often the true order
function arOlsIC(x, pmax) {
  const Tn = x.length, n = Tn - pmax, k = pmax + 1, cols = [new Array(n).fill(1)];
  for (let i = 1; i <= pmax; i++) cols.push(x.slice(pmax - i, Tn - i));
  const y = x.slice(pmax), Gm = cols.map(a => cols.map(b => { let s = 0; for (let t = 0; t < n; t++) s += a[t] * b[t]; return s; }));
  const g = cols.map(a => { let s = 0; for (let t = 0; t < n; t++) s += a[t] * y[t]; return s; }), yy = sum(y.map(v => v * v));
  let ba = [Infinity, 0], bb = [Infinity, 0];
  for (let p = 0; p < k; p++) {
    const idx = range(0, p, 1), b = solve(idx.map(i => idx.map(j => Gm[i][j])), idx.map(i => g[i])), s2 = (yy - sum(b.map((v, i) => v * g[i]))) / n;
    const ll = -0.5 * n * (Math.log(2 * Math.PI * s2) + 1), q = p + 2, aic = -2 * ll + 2 * q, bic = -2 * ll + q * Math.log(n);
    if (aic < ba[0]) ba = [aic, p]; if (bic < bb[0]) bb = [bic, p];
  }
  return [ba[1], bb[1]];
}
const icCache = {};
function icFreq(f2, Tn) {
  const key = f2 + '|' + Tn, I = D.ic;
  if (!icCache[key]) {
    if (Math.abs(f2 - I.phi[1]) < 1e-9 && I.freq[Tn]) icCache[key] = Object.assign({ lecture: true }, I.freq[Tn]);
    else { const z = rng(4000 + Tn + Math.round(100 * f2)), A = new Array(7).fill(0), B = new Array(7).fill(0);
      for (let s = 0; s < I.nrep; s++) { const [a, b] = arOlsIC(armaSim([I.phi[0], f2], [], Tn, z, 500), I.pmax); A[a] += 1 / I.nrep; B[b] += 1 / I.nrep; }
      icCache[key] = { aic: A, bic: B, lecture: false }; }
  }
  return icCache[key];
}
function blkIC() {
  const f2 = +$('ic-f2').value, T1 = +$('ic-T').value; set('ic-f2', nf(f2, 2));
  const Ts = [T1, 1000], F = Ts.map(t => icFreq(f2, t));
  const G = grid('ic-chart', Ts.map((t, j) => ({ r: 0, c: j, t: `True AR(2), φ = (0.5, ${f2}), T = ${t}: ${D.ic.nrep} samples`, xt: 'chosen order p', yt: j ? '' : '% of samples', y: { range: [0, 100] }, x: { dtick: 1 } })), { cols: 2, h: 360 });
  const tr = [], p = range(0, 6, 1);
  F.forEach((f, j) => { ['aic', 'bic'].forEach((c, i) => tr.push({ x: p.map(v => v + (i ? 0.2 : -0.2)), y: f[c].map(v => 100 * v), type: 'bar', width: 0.4, marker: { color: i ? K.red : K.blue }, name: c.toUpperCase(), legendgroup: c, showlegend: j === 0,
    xaxis: G.ax[j].xa, yaxis: G.ax[j].ya, hovertemplate: c.toUpperCase() + ', p = %{customdata}: %{y:.1f}%<extra></extra>', customdata: p })); G.lay.shapes.push(vl(G.ax[j], 2, K.green, 'dash')); });
  tr.push(leg('true order p = 2', { color: K.green, dash: 'dash', width: 1.2 }));
  plot('ic-chart', tr, G.lay);
  const s = (a, i, j) => pct(sum(a.slice(i, j)), 1);
  out('ic-out', Ts.map((t, j) => T.ic(t, s(F[j].aic, 2, 3), s(F[j].bic, 2, 3), s(F[j].aic, 0, 2), s(F[j].bic, 0, 2), s(F[j].aic, 3, 7), s(F[j].bic, 3, 7), F[j].lecture)));
}

// ================================================================= Ljung-Box degrees of freedom
function lbRate(set_, m, df, alpha) {
  const n = D.lb.n; let k = 0;
  set_.forEach(r => { let q = 0; for (let h = 1; h <= m; h++) q += r[h - 1] ** 2 / (n - h); if (chi2sf(n * (n + 2) * q, df) < alpha) k++; });
  return 100 * k / set_.length;
}
function blkLB() {
  const m = +$('lb-m').value, al = +$('lb-a').value; set('lb-m', m);
  const S = [['AR(1) residuals', D.lb.ar, 1], ['ARMA(1,1) residuals', D.lb.arma, 2]];
  const R = S.map(([, s, pq]) => [lbRate(s, m, m, al), lbRate(s, m, m - pq, al)]);
  const ms = range(3, m, 1), curves = S.map(([, s, pq]) => [ms.map(v => lbRate(s, v, v, al)), ms.map(v => lbRate(s, v, v - pq, al))]);
  const A = 100 * al, G = grid('lb-chart', [{ r: 0, c: 0, t: `Ljung-Box on residuals of the true model, T = ${D.lb.T}, m = ${m}`, yt: 'rejection rate (%)', y: { range: [0, Math.max(12, 2.2 * A)] } },
    { r: 0, c: 1, t: `Rejection rate for m = 3, …, ${m}`, xt: 'm', x: { range: [2.5, 20.5] }, y: { range: [0, Math.max(12, 2.2 * A)] } }], { cols: 2, h: 370, legendRows: 2 });
  const [a, b] = G.ax, tr = [];
  [['df = m', K.orange], ['df = m − p − q', K.blue]].forEach(([nm, c], i) => tr.push({ x: S.map(s => s[0]), y: R.map(r => r[i]), type: 'bar', name: nm, marker: { color: c }, offsetgroup: i, xaxis: a.xa, yaxis: a.ya, hovertemplate: nm + ': %{y:.1f}%<extra></extra>' }));
  curves.forEach((cv, j) => cv.forEach((y, i) => tr.push({ x: ms, y, mode: 'lines+markers', marker: { size: ms.map(v => (v === m ? 9 : 5)), symbol: j ? 'square' : 'circle' }, line: { color: i ? K.blue : K.orange, width: 1.5, dash: j ? 'dot' : 'solid' },
    name: (j ? 'ARMA(1,1)' : 'AR(1)') + (i ? ', df = m − p − q' : ', df = m'), showlegend: false, xaxis: b.xa, yaxis: b.ya, hovertemplate: (j ? 'ARMA(1,1)' : 'AR(1)') + (i ? ', df = m − p − q' : ', df = m') + ', m = %{x}: %{y:.1f}%<extra></extra>' })));
  G.lay.barmode = 'group';
  tr.push(leg(`nominal level ${A}%`, { color: K.red, dash: 'dash', width: 1.2 }), leg('AR(1): solid, ARMA(1,1): dotted', { color: K.navy, width: 1.2 }));
  G.lay.shapes.push(hl(a, A, K.red, 'dash'), hl(b, A, K.red, 'dash'));
  plot('lb-chart', tr, G.lay);
  out('lb-out', [T.lb(m, nf(A, 0), nf(R[0][0], 1), nf(R[0][1], 1), nf(R[1][0], 1), nf(R[1][1], 1))]);
}

// ================================================================= mean reversion in forecasts
function blkFct() {
  const xT = +$('fc-x').value, p1 = +$('fc-p1').value, p2 = +$('fc-p2').value, th = +$('fc-th').value, eT = 2, H = 20;
  set('fc-x', nf(xT, 1)); set('fc-p1', nf(p1, 2)); set('fc-p2', nf(p2, 2)); set('fc-th', nf(th, 2));
  const cases = [[`AR(1), φ = ${p1}`, [p1], [], K.blue], [`AR(1), φ = ${p2}`, [p2], [], K.green], [`MA(1), θ = ${th}, ε<sub>T</sub> = ${eT}`, [], [th], K.purple]];
  const h = range(0, H, 1), res = [], tr = [], all = [];
  const P = cases.map(([lab, ph, tt]) => { const ps = psiW(ph, tt, H + 1), f = ph.length ? h.map(k => xT * ph[0] ** k) : h.map(k => (k === 0 ? xT : k === 1 ? tt[0] * eT : 0));
    const se = [0]; let c = 0; for (let k = 0; k < H; k++) { c += ps[k] ** 2; se.push(Math.sqrt(c)); }
    const su = Math.sqrt(sum(psiW(ph, tt, 2000).map(v => v * v))); all.push(...f.map((v, k) => v + 1.96 * se[k]), ...f.map((v, k) => v - 1.96 * se[k]), 1.96 * su, -1.96 * su); return { f, se, su }; });
  const lim = Math.min(12, Math.max(4.5, ...all.map(Math.abs)) * 1.05);
  const G = grid('fc-chart', cases.map((c, j) => ({ r: 0, c: j, t: c[0], xt: 'horizon h', y: { range: [-lim, lim] } })), { cols: 3, h: 360 });
  cases.forEach(([lab, , , col], j) => { const { f, se, su } = P[j], a = G.ax[j], rgb = col === K.blue ? '26,58,110' : col === K.green ? '46,125,50' : '142,68,173';
    tr.push({ x: h.concat([...h].reverse()), y: f.map((v, k) => v + 1.96 * se[k]).concat(f.map((v, k) => v - 1.96 * se[k]).reverse()), fill: 'toself', fillcolor: `rgba(${rgb},0.15)`, line: { width: 0 }, mode: 'lines', name: '95% interval', legendgroup: 'band', showlegend: j === 0, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' });
    tr.push({ x: h, y: f, mode: 'lines+markers', marker: { size: 5, color: col }, line: { color: col, width: 1.5 }, name: 'point forecast', legendgroup: 'f', showlegend: j === 0, xaxis: a.xa, yaxis: a.ya, customdata: se, hovertemplate: 'h = %{x}: %{y:.2f}, se %{customdata:.2f}<extra></extra>' });
    if (isFinite(su) && su < 1e6) G.lay.shapes.push(hl(a, 1.96 * su, K.red, 'dash', 1), hl(a, -1.96 * su, K.red, 'dash', 1));
    G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.8));
    res.push(T.fc(j < 2 ? 'AR(1), φ = ' + nf(j ? p2 : p1, 2) : 'MA(1), θ = ' + nf(th, 2), nf(f[1], 2), nf(f[5], 2), nf(se[5], 2), isFinite(su) && su < 1e6 ? nf(su, 2) : '∞'));
  });
  tr.push(leg('μ ± 1.96 σ<sub>X</sub>', { color: K.red, dash: 'dash', width: 1.2 }));
  plot('fc-chart', tr, G.lay);
  out('fc-out', res);
}

// ================================================================= GDP growth: identification
function blkGdpId() {
  const g = D.gdp, k = +$('gi-end').value, y = g.y.slice(0, k + 1), d = g.dates.slice(0, k + 1), n = y.length;
  set('gi-end', T.q(d[n - 1]));
  const r = acf(y, 16), pc = pacf(r), m = mean(y), lb = ljungBox(y, 8);
  const G = grid('gi-chart', [{ r: 0, c: 0, cs: 2, t: 'Romania: real GDP growth, 100 (ln Y<sub>t</sub> − ln Y<sub>t−4</sub>), %', x: { type: 'date', range: [q2d(g.dates[0]), q2d(g.dates[g.dates.length - 1])] }, y: { range: [-11, 13.5] } },
    { r: 1, c: 0, t: 'ACF', xt: 'lag (quarters)', y: { range: [-0.5, 0.8] } }, { r: 1, c: 1, t: 'PACF', xt: 'lag (quarters)', y: { range: [-0.5, 0.8] } }], { cols: 2, h: 460 });
  const [a, b, c] = G.ax, tr = [];
  tr.push({ x: d.map(q2d), y, mode: 'lines', line: { color: K.green, width: 1.5 }, name: 'GDP growth', showlegend: false, xaxis: a.xa, yaxis: a.ya, customdata: d, hovertemplate: '%{customdata}: %{y:.2f}%<extra></extra>' });
  tr.push({ x: [q2d(d[0]), q2d(d[n - 1])], y: [m, m], mode: 'lines', line: { color: K.orange, dash: 'dash', width: 1.3 }, name: `mean ${m.toFixed(2)}%`, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' });
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.8));
  acfPanel(tr, G.lay, b, r, n, K.green); acfPanel(tr, G.lay, c, pc, n, K.green);
  tr.push(BAND);
  plot('gi-chart', tr, G.lay);
  out('gi-out', [T.gi1(T.q(d[0]), T.q(d[n - 1]), n, nf(m, 2), nf(1.96 / Math.sqrt(n), 2)),
    T.gi2(r.slice(0, 4).map(v => nf(v, 2)).join('; '), nf(pc[0], 2), nf(pc[4], 2)), T.gi3(nf(lb.q, 1), pe(chi2sf(lb.q, 8)))]);
}

// ================================================================= GDP growth: diagnostics
function probplot(e) {            // scipy.stats.probplot: Filliben plotting positions, least-squares line
  const n = e.length, s = [...e].sort((u, v) => u - v), mN = Math.pow(0.5, 1 / n);
  const q = range(1, n, 1).map(i => normPpf(i === 1 ? 1 - mN : i === n ? mN : (i - 0.3175) / (n + 0.365)));
  const mx = mean(q), my = mean(s); let sxy = 0, sxx = 0; q.forEach((v, i) => { sxy += (v - mx) * (s[i] - my); sxx += (v - mx) ** 2; });
  const sl = sxy / sxx; return { q, s, sl, ic: my - sl * mx };
}
function blkGdpDiag() {
  const key = $('gd-mod').value, dfm = $('gd-df').value === 'right', p = +key[0], q = +key[1], M = D.gdp.models[key], e = M.resid, n = e.length;
  const dates = D.gdp.dates.slice(Math.max(p, q)), sd = Math.sqrt(sum(e.map(v => (v - mean(e)) ** 2)) / (n - 1)), fit = p + q;
  const ms = range(fit + 1, 16, 1), pv = ms.map(m => chi2sf(ljungBox(e, m).q, dfm ? m - fit : m));
  const G = grid('gd-chart', [{ r: 0, c: 0, t: `Residuals of ARMA(${p},${q})`, x: { type: 'date' } }, { r: 0, c: 1, t: 'ACF of the residuals', y: { range: [-0.4, 0.4] } },
    { r: 1, c: 0, t: `Ljung-Box p-values, df = m${dfm ? (fit ? ' - ' + fit : '') : ''}`, xt: 'm', y: { range: [0, 1] }, x: { range: [0.5, 16.5] } }, { r: 1, c: 1, t: 'Normal QQ plot of the standardised residuals', xt: 'theoretical quantiles' }], { cols: 2, h: 470 });
  const [a, b, c, d] = G.ax, tr = [];
  tr.push({ x: dates.map(q2d), y: e, mode: 'lines', line: { color: K.blue, width: 1.2 }, showlegend: false, xaxis: a.xa, yaxis: a.ya, customdata: dates, hovertemplate: '%{customdata}: %{y:.2f}<extra></extra>' });
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.8));
  acfPanel(tr, G.lay, b, acf(e, 16), n, K.blue);
  tr.push({ x: ms, y: pv, mode: 'markers', marker: { color: K.blue, size: 9 }, showlegend: false, xaxis: c.xa, yaxis: c.ya, hovertemplate: 'm = %{x}: p = %{y:.3f}<extra></extra>' });
  G.lay.shapes.push(hl(c, 0.05, K.orange, 'dash'));
  const z = probplot(e.map(v => v / sd)), qq = [Math.min(...z.q), Math.max(...z.q)];
  tr.push({ x: z.q, y: z.s, mode: 'markers', marker: { color: K.blue, size: 6 }, showlegend: false, xaxis: d.xa, yaxis: d.ya, hovertemplate: '%{x:.2f}, %{y:.2f}<extra></extra>' });
  tr.push({ x: qq, y: qq.map(v => z.ic + z.sl * v), mode: 'lines', line: { color: K.red, width: 1.3 }, name: 'Normal reference line', xaxis: d.xa, yaxis: d.ya, hoverinfo: 'skip' });
  tr.push(BAND, leg('5% level', { color: K.orange, dash: 'dash', width: 1.2 }));
  plot('gd-chart', tr, G.lay);
  const m = mean(e), s2 = pvar(e), sk = sum(e.map(v => (v - m) ** 3)) / n / s2 ** 1.5, ku = sum(e.map(v => (v - m) ** 4)) / n / s2 ** 2, jb = n / 6 * (sk * sk + (ku - 3) ** 2 / 4);
  const lb8 = ljungBox(e, 8).q, i = e.reduce((bi, v, k) => (Math.abs(v) > Math.abs(e[bi]) ? k : bi), 0), lbs = ljungBox(e.map(v => v * v), 8).q;
  out('gd-out', [T.gd1(p, q, nf(M.aic, 1), nf(M.bic, 1)), T.gd2(nf(lb8, 2), 8 - fit, pe(chi2sf(lb8, 8 - fit)), pe(chi2sf(lb8, 8)), fp(Math.min(...pv))),
    T.gd3(nf(jb, 1), pe(Math.exp(-jb / 2)), nf(sk, 2), nf(ku, 2), nf(e[i], 1), T.q(dates[i]), nf(e[i] / sd, 2), pe(chi2sf(lbs, 8)))]);
}

// ================================================================= GDP growth: forecasts
function blkGdpFc() {
  const g = D.gdp, O = Object.keys(g.origins), k = +$('gf-o').value, o = O[k], lv = +$('gf-lv').value, z = normPpf(1 - (1 - lv) / 2);
  set('gf-o', T.q(o));
  const io = g.dates.indexOf(o), fq = range(1, 8, 1).map(h => addQ(o, h)), real = fq.map(q => { const j = g.dates.indexOf(q); return j < 0 ? null : g.y[j]; });
  const i0 = Math.max(0, io - 39), i1 = Math.min(g.y.length - 1, io + 8);
  const G = grid('gf-chart', [{ r: 0, c: 0, t: `Romania: annual real GDP growth (%), forecasts from ${o} for the next 8 quarters`, x: { type: 'date' } }], { cols: 1, h: 400, legendRows: 2 });
  const a = G.ax[0], tr = [];
  tr.push({ x: g.dates.slice(i0, io + 1).map(q2d), y: g.y.slice(i0, io + 1), mode: 'lines', line: { color: K.navy, width: 1.5 }, name: 'observed', xaxis: a.xa, yaxis: a.ya, customdata: g.dates.slice(i0, io + 1), hovertemplate: '%{customdata}: %{y:.2f}%<extra></extra>' });
  if (i1 > io) tr.push({ x: g.dates.slice(io, i1 + 1).map(q2d), y: g.y.slice(io, i1 + 1), mode: 'lines+markers', marker: { size: 5 }, line: { color: K.navy, width: 1.3, dash: 'dot' }, name: 'observed after the origin', xaxis: a.xa, yaxis: a.ya, customdata: g.dates.slice(io, i1 + 1), hovertemplate: '%{customdata}: %{y:.2f}%<extra></extra>' });
  const res = [];
  [['03', 'ARMA(0,3)', K.red, '205,0,0'], ['10', 'ARMA(1,0)', K.blue, '26,58,110']].forEach(([key, lab, col, rgb]) => {
    const F = g.origins[o][key], lo = F.f.map((v, h) => v - z * F.se[h]), hi = F.f.map((v, h) => v + z * F.se[h]);
    tr.push({ x: fq.map(q2d).concat([...fq].reverse().map(q2d)), y: hi.concat([...lo].reverse()), fill: 'toself', fillcolor: `rgba(${rgb},0.13)`, line: { width: 0 }, mode: 'lines', name: `${lab}: ${Math.round(100 * lv)}% interval`, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' });
    tr.push({ x: fq.map(q2d), y: F.f, mode: 'lines+markers', marker: { size: 6, color: col }, line: { color: col, width: 1.6 }, name: `${lab} forecast`, xaxis: a.xa, yaxis: a.ya, customdata: fq.map((q, h) => [q, lo[h], hi[h]]),
      hovertemplate: '%{customdata[0]}: %{y:.2f}% [%{customdata[1]:.1f}, %{customdata[2]:.1f}]<extra>' + lab + '</extra>' });
    const er = real.map((v, h) => (v === null ? null : v - F.f[h])).filter(v => v !== null);
    res.push(T.gf(lab, nf(F.f[0], 2), nf(F.f[3], 2), nf(F.f[7], 2), `[${nf(lo[0], 1)}; ${nf(hi[0], 1)}]`, `[${nf(lo[7], 1)}; ${nf(hi[7], 1)}]`,
      er.length ? nf(Math.sqrt(mean(er.map(v => v * v))), 2) : null, er.length));
  });
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.8));
  plot('gf-chart', tr, G.lay);
  out('gf-out', [T.gf0(T.q(o), Math.round(100 * lv))].concat(res));
}

// ================================================================= inflation: AR(p)
function blkInfl() {
  const I = D.infl, p = $('in-p').value, F = I.fit[p], H = F.f.length, last = I.dates[I.dates.length - 1];
  const idx = range(1, H, 1).map(h => addM(last, h)), z95 = normPpf(0.975), z80 = normPpf(0.9);
  const G = grid('in-chart', [{ r: 0, c: 0, t: `Romania: HICP inflation, 12-month rate (%), and an AR(${p}) forecast`, x: { type: 'date' } }], { cols: 1, h: 400, legendRows: 2 });
  const a = G.ax[0];
  G.lay.shapes.push({ type: 'rect', xref: a.xd, x0: 0, x1: 1, yref: a.ya, y0: 1.5, y1: 3.5, fillcolor: 'rgba(46,125,50,0.10)', line: { width: 0 }, layer: 'below' }, hl(a, F.mu, K.orange, 'dash'));
  const band = (zz, al, nm) => ({ x: idx.concat([...idx].reverse()), y: F.f.map((v, h) => v + zz * F.se[h]).concat(F.f.map((v, h) => v - zz * F.se[h]).reverse()), fill: 'toself', fillcolor: `rgba(26,58,110,${al})`, line: { width: 0 }, mode: 'lines', name: nm, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' });
  const tr = [{ x: I.dates.map(m2d), y: I.y, mode: 'lines', line: { color: K.red, width: 1.4 }, name: '12-month inflation', xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x|%Y-%m}: %{y:.2f}%<extra></extra>' },
    { x: idx, y: F.f, mode: 'lines', line: { color: K.blue, width: 2 }, name: `AR(${p}) forecast`, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x|%Y-%m}: %{y:.2f}%<extra></extra>' },
    band(z95, 0.12, '95% interval'), band(z80, 0.22, '80% interval'),
    leg(`estimated mean ${F.mu.toFixed(2)}%`, { color: K.orange, dash: 'dash', width: 1.2 }), leg('BNR target 2.5% ± 1 p.p.', null, { color: 'rgba(46,125,50,0.3)', symbol: 'square', size: 14 })];
  plot('in-chart', tr, G.lay);
  const bics = Object.values(I.fit).map(f => f.bic), best = bics.indexOf(Math.min(...bics)) + 1;
  out('in-out', [T.in1(p, nf(F.bic, 1), best, nf(Math.min(...bics), 1)), T.in2(F.phi.map((v, i) => `φ̂<sub>${i + 1}</sub> = ${nf(v, 3)} (${nf(F.se_phi[i], 3)})`).join(', '), nf(sum(F.phi), 3), nf(F.inv_max, 3), nf(F.mu, 2)),
    T.in3(nf(I.y[I.y.length - 1], 2), nf(F.f[0], 2), nf(F.f[H - 1], 2), T.m(idx[H - 1]), `[${nf(F.f[H - 1] - z95 * F.se[H - 1], 1)}; ${nf(F.f[H - 1] + z95 * F.se[H - 1], 1)}]`),
    T.in4(nf(F.lb12, 1), 12 - +p, pe(F.lb12_p))]);
}

// ================================================================= daily returns: BET and EUR/RON
function ar1Ols(r) {
  const n = r.length; let mz = 0, my = 0; for (let t = 1; t < n; t++) { mz += r[t - 1]; my += r[t]; } mz /= n - 1; my /= n - 1;
  let sxy = 0, sxx = 0; for (let t = 1; t < n; t++) { sxy += (r[t - 1] - mz) * (r[t] - my); sxx += (r[t - 1] - mz) ** 2; }
  const phi = sxy / sxx, c = my - phi * mz, e = []; for (let t = 1; t < n; t++) e.push(r[t] - c - phi * r[t - 1]);
  const s2 = sum(e.map(v => v * v)) / (n - 3); return { phi, se: Math.sqrt(s2 / sxx), e };
}
function blkRet() {
  let y0 = +$('rt-y0').value, y1 = +$('rt-y1').value; if (y1 < y0 + 2) { y1 = y0 + 2; $('rt-y1').value = y1; } set('rt-y0', y0); set('rt-y1', y1);
  const S = [['bet', 'BET', K.red], ['eurron', 'EUR/RON', K.green]], cells = [], tr = [], res = [];
  const C2 = S.map(([k]) => {
    const R = D.ret[k], ix = R.dates.map((d, j) => (+d.slice(0, 4) >= y0 && +d.slice(0, 4) <= y1 ? j : -1)).filter(j => j >= 0), r = ix.map(j => R.r[j]), n = r.length;
    const full = n === R.r.length, f = ar1Ols(r), e = full ? r.slice(1).map((v, t) => v - R.mu - R.phi * (r[t] - R.mu)) : f.e;
    return { R, ix, r, n, full, f, e, ra: acf(r, 20), rs: acf(e.map(v => v * v), 20) };
  });
  const top = (a, lo, hi) => [Math.min(lo, ...a.map(c => Math.min(...c) - 0.03)), Math.max(hi, ...a.map(c => Math.max(...c) + 0.03))];
  const yA = top(C2.map(c => c.ra), -0.1, 0.25), yB = top(C2.map(c => c.rs), -0.1, 0.45);
  S.forEach(([, lab], i) => cells.push({ r: i, c: 0, t: `${lab}: ACF of daily returns`, xt: i ? 'lag (days)' : '', y: { range: yA } }, { r: i, c: 1, t: `${lab}: ACF of squared AR(1) residuals`, xt: i ? 'lag (days)' : '', y: { range: yB } }));
  const G = grid('rt-chart', cells, { cols: 2, h: 440 });
  S.forEach(([k, lab, col], i) => {
    const { R, ix, r, n, full, f, e, ra, rs } = C2[i];
    acfPanel(tr, G.lay, G.ax[2 * i], ra, n, col); acfPanel(tr, G.lay, G.ax[2 * i + 1], rs, n, col);
    const lb = ljungBox(e, 10).q, lbs = ljungBox(e.map(v => v * v), 10).q, r2 = 1 - pvar(e) / pvar(r.slice(1));
    res.push(T.rt(lab, R.dates[ix[0]], R.dates[ix[n - 1]], n, nf(ra[0], 3), nf(full ? R.phi : f.phi, 3), nf(100 * r2, 1), nf(lb, 1), pe(chi2sf(lb, 9)), nf(lbs, 0), full));
  });
  tr.push(BAND);
  plot('rt-chart', tr, G.lay);
  out('rt-out', res);
}

// ================================================================= sunspots
function blkSun() {
  const S = D.sun, p = +$('ss-p').value, F = S.fit[p], yrs = S.years.map(String);
  const G = grid('ss-chart', [{ r: 0, c: 0, cs: 2, t: `Yearly sunspot numbers and AR(${p}) one-step predictions` }, { r: 0, c: 2, t: 'Information criteria, full sample', xt: 'AR order p', x: { dtick: 1 } }], { cols: 3, h: 380, legendRows: 2 });
  const [a, b] = G.ax, ps = range(1, 10, 1), aic = ps.map(k => S.fit[k].aic), bic = ps.map(k => S.fit[k].bic);
  G.lay.shapes.push({ type: 'rect', xref: a.xa, x0: 1749, x1: 1924, yref: a.yd, y0: 0, y1: 1, fillcolor: 'rgba(23,162,184,0.08)', line: { width: 0 }, layer: 'below' }, vl(b, p, K.navy, 'dot', 1));
  const tr = [{ x: S.years, y: S.y, mode: 'lines', line: { color: K.amber, width: 1.4 }, name: 'sunspot number', xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.1f}<extra></extra>' },
    { x: S.years.slice(p), y: F.pred.slice(p), mode: 'lines', line: { color: K.blue, width: 1.1 }, name: `AR(${p}): one-step prediction`, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.1f}<extra></extra>' },
    leg("Yule's sample 1749-1924", null, { color: 'rgba(23,162,184,0.3)', symbol: 'square', size: 14 }),
    { x: ps, y: aic, mode: 'lines+markers', marker: { color: K.blue, size: 6 }, line: { color: K.blue }, name: 'AIC', xaxis: b.xa, yaxis: b.ya, hovertemplate: 'p = %{x}: AIC %{y:.1f}<extra></extra>' },
    { x: ps, y: bic, mode: 'lines+markers', marker: { color: K.red, size: 6, symbol: 'square' }, line: { color: K.red }, name: 'BIC', xaxis: b.xa, yaxis: b.ya, hovertemplate: 'p = %{x}: BIC %{y:.1f}<extra></extra>' },
    { x: [p, p], y: [F.aic, F.bic], mode: 'markers+text', marker: { color: 'rgba(0,0,0,0)', size: 15, line: { color: K.navy, width: 2 } }, text: [`AIC ${F.aic.toFixed(1)}`, `BIC ${F.bic.toFixed(1)}`],
      textposition: p > 6 ? ['bottom left', 'top left'] : ['bottom right', 'top right'], textfont: { size: 11, color: K.navy }, cliponaxis: false, name: `AR(${p})`, xaxis: b.xa, yaxis: b.ya, hoverinfo: 'skip' }];
  plot('ss-chart', tr, G.lay);
  const ba = ps[aic.indexOf(Math.min(...aic))], bb = ps[bic.indexOf(Math.min(...bic))], lines = [T.ss1(p, F.phi.map(v => nf(v, 3)).join('; '), nf(100 * F.r2, 1), nf(F.aic, 1), nf(F.bic, 1), ba, bb)];
  if (p === 2) { const [f1, f2] = F.phi; if (f1 * f1 + 4 * f2 < 0) lines.push(T.ss2(nf(2 * Math.PI / Math.acos(f1 / (2 * Math.sqrt(-f2))), 1), nf(Math.sqrt(-f2), 3))); }
  lines.push(T.ss3(nf(D.numbers.sun.phi1, 3), nf(D.numbers.sun.phi2, 3), nf(D.numbers.sun.period, 1)));
  out('ss-out', lines);
}

// ================================================================= tables and worked examples (blocks of their own frames)
const X = () => T.x;
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}>${v.v}</${tag}>`; };
  el.innerHTML = (o && o.cap ? `<p class="c2cap">${o.cap}</p>` : '') + '<table><thead><tr>' + head.map(c => cell(c, 'th')).join('') + '</tr></thead><tbody>'
    + rows.map(r => `<tr${r.hi ? ' class="hi"' : ''}>` + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c2cap">${o.note}</p>` : '');
}
const nz = (v, d = 2) => nf(Math.abs(v) < 0.5 * 10 ** -d ? 0 : v, d);                // no '−0.00'
const num = (x, d, bold) => ({ v: x === null || !isFinite(x) ? '—' : bold ? `<b>${nz(x, d)}</b>` : nz(x, d), cls: 'r' });
const txt = (v, bold) => ({ v: bold ? `<b>${v}</b>` : v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const pn = (x, d) => (x < 0 ? `(${nz(x, d)})` : nz(x, d));                       // a factor in a product: negative numbers in parentheses
const sg = (x, d) => (x < 0 ? ' − ' + nz(-x, d) : ' + ' + nz(x, d));            // a term of a sum
const hiRow = (r, on) => { if (on) r.hi = true; return r; };
const ns = v => nz(v, 4).replace(/([.,]\d*?)0+$/, '$1').replace(/[.,]$/, '');          // up to 4 decimals, no trailing zeros
const yn = b => (b ? X().yes : X().no);
const cpx = (re, im, d) => (Math.abs(im) < 1e-12 ? nz(re, d) : `${nz(re, d)} ${im < 0 ? '−' : '+'} ${nz(Math.abs(im), d)}i`);
function chi2ppf(p, k) { let lo = 0, hi = 10 * k + 100; for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; if (1 - chi2sf(m, k) < p) lo = m; else hi = m; } return (lo + hi) / 2; }
const modLab = k => `ARMA(${k[0]},${k[1]})`;
const GKEYS = () => Object.keys(D.gdpfit).sort();
const gfit = k => { const F = D.gdpfit[k], n = F.nobs, aic = -2 * F.llf + 2 * F.k;
  return Object.assign({ p: +k[0], q: +k[1], aic, bic: -2 * F.llf + F.k * Math.log(n), aicc: aic + 2 * F.k * (F.k + 1) / (n - F.k - 1), sig2: F.b[F.b.length - 1] }, F); };

// ---------------------------------------------------------------- Worked example: an AR(1)
function blkXAR1() {
  const c = +$('xa-c').value, phi = +$('xa-phi').value, s2 = +$('xa-s2').value, h = +$('xa-h').value;
  set('xa-c', nf(c, 1)); set('xa-phi', nf(phi, 2)); set('xa-s2', nf(s2, 2)); set('xa-h', h);
  const mu = c / (1 - phi), g0 = s2 / (1 - phi * phi), a = Math.abs(phi), hlf = a > 0 ? Math.log(0.5) / Math.log(a) : 0, ps = range(0, 10, 1).map(j => phi ** j);
  steps('xa-ws', [X().xa1(nf(a, 2)), X().xa2(nf(c, 1), pn(phi, 2), nz(mu, 2)), X().xa3(ns(s2), ns(phi * phi), nz(g0, 2)),
    X().xa4(h, pn(phi, 2), nz(phi ** h, 3)), a > 0 ? X().xa5(nf(a, 2), nf(hlf, 2), phi < 0) : X().xa5z(),
    X().xa6(ps.slice(0, 4).map(v => nz(v, 3)).join(X().sep), nz(ps[2], 2))]);
  table('xa-tb', ['j', 'ψ<sub>j</sub> = φ<sup>j</sup>', 'ρ(j) = φ<sup>j</sup>', 'γ(j) = φ<sup>j</sup>γ(0)'],
    ps.map((v, j) => hiRow([txt(j), num(v, 3), num(v, 3), num(v * g0, 3)], j === h)), { cap: X().xaCap(nz(mu, 2), nz(g0, 2)) });
  const G = grid('xa-chart', [{ r: 0, c: 0, t: `AR(1), φ = ${phi}: ρ(h) = φ<sup>h</sup> and the half-life`, xt: 'lag h', y: { range: [-1, 1] }, x: { range: [-0.5, 20.5] } }], { cols: 1, h: 300 });
  const ax = G.ax[0], hh = range(0, 20, 1);
  G.lay.shapes.push(hl(ax, 0, K.navy, 'solid', 0.8), hl(ax, 0.5, K.orange, 'dot', 1), hl(ax, -0.5, K.orange, 'dot', 1));
  if (a > 0) G.lay.shapes.push(vl(ax, hlf, K.green, 'dash', 1.3));
  plot('xa-chart', [{ x: hh, y: hh.map(k => phi ** k), type: 'bar', width: 0.55, marker: { color: hh.map(k => (k === h ? K.red : K.blue)) }, xaxis: ax.xa, yaxis: ax.ya, showlegend: false, hovertemplate: 'ρ(%{x}) = %{y:.3f}<extra></extra>' },
    leg(`half-life ${a > 0 ? hlf.toFixed(2) : 0}`, { color: K.green, dash: 'dash', width: 1.3 }), leg('±0.5', { color: K.orange, dash: 'dot', width: 1 })], G.lay);
  out('xa-out', [X().xa(nf(phi, 2), nz(mu, 2), nz(g0, 2), h, nz(phi ** h, 3), a > 0 ? nf(hlf, 2) : '0')]);
}

// ---------------------------------------------------------------- The AR(p) model and its characteristic polynomial: roots of 1 - phi1 z - phi2 z^2
function rootsZ(a, b) {           // roots of 1 - a z - b z^2: [[re, im], ...]
  if (Math.abs(b) < 1e-12) return Math.abs(a) < 1e-12 ? [] : [[1 / a, 0]];
  const d = a * a + 4 * b;
  if (d >= 0) return [(-a + Math.sqrt(d)) / (2 * b), (-a - Math.sqrt(d)) / (2 * b)].map(v => [v, 0]);
  return [[-a / (2 * b), Math.sqrt(-d) / (2 * b)], [-a / (2 * b), -Math.sqrt(-d) / (2 * b)]];
}
function blkXRoots() {
  const a = +$('xr-f1').value, b = +$('xr-f2').value; set('xr-f1', nf(a, 3)); set('xr-f2', nf(b, 3));
  const z = rootsZ(a, b), md = z.map(r => Math.hypot(r[0], r[1])), d = a * a + 4 * b, st = z.every((r, i) => md[i] > 1), ex = AR2();
  const zs = z.length === 2 && Math.abs(z[0][1]) > 1e-12 ? `${nz(z[0][0], 3)} ± ${nz(Math.abs(z[0][1]), 3)}i` : z.map(r => nz(r[0], 3)).join(T.and);
  const S = [];
  S.push(X().xr1(`1${sg(-a, 3)}z${sg(-b, 3)}z²`));
  if (Math.abs(b) > 1e-12) S.push(X().xr2(pn(a, 3), pn(b, 3), nz(d, 3), d < 0), X().xr3(zs, md.map(m => nz(m, 3)).join(T.and)));
  else S.push(X().xr3b(nf(a, 3), z.length ? nz(z[0][0], 3) : '∞'));
  S.push(X().xr4(md.map(m => nz(1 / m, 3)).join(T.and)));
  S.push(X().xr5(nz(a + b, 3), yn(a + b < 1), nz(b - a, 3), yn(b - a < 1), nz(Math.abs(b), 3), yn(Math.abs(b) < 1)), X().xr6(st));
  steps('xr-ws', S);
  const rows = ['A', 'B', 'C', 'Sunspots'].map(k => [k, ex[k].phi]).concat([[X().yours, [a, b]]]);
  table('xr-tb', [X().model, 'φ<sub>1</sub>', 'φ<sub>2</sub>', 'φ<sub>1</sub>² + 4φ<sub>2</sub>', X().roots, 'min |z|', 'max |λ|', 'φ<sub>1</sub> + φ<sub>2</sub>', X().stat],
    rows.map(([k, [p1, p2]], i) => { const zz = rootsZ(p1, p2), mm = zz.map(r => Math.hypot(r[0], r[1])), s = mm.every(m => m > 1);
      const zl = zz.length === 2 && Math.abs(zz[0][1]) > 1e-12 ? `${nz(zz[0][0], 2)} ± ${nz(Math.abs(zz[0][1]), 2)}i` : zz.map(r => nz(r[0], 2)).join('; ');
      return hiRow([k, num(p1, 3), num(p2, 3), num(p1 * p1 + 4 * p2, 3), txt(zl), num(Math.min(...mm), 3), num(1 / Math.min(...mm), 3), num(p1 + p2, 3), txt(yn(s), !s)], i === 4); }));
  const G = grid('xr-chart', [{ r: 0, c: 0, t: 'Roots z of φ(z) = 1 − φ<sub>1</sub>z − φ<sub>2</sub>z² and the unit circle', xt: 'real part', yt: 'imaginary part', x: { range: [-3, 3] }, y: { range: [-2.2, 2.2] } }], { cols: 1, h: 380, legendRows: 2 });
  const ax = G.ax[0], th = range(0, 2 * Math.PI, 0.02); G.lay['yaxis'].scaleanchor = 'x';
  const tr = [{ x: th.map(Math.cos), y: th.map(Math.sin), fill: 'toself', fillcolor: 'rgba(205,0,0,0.07)', mode: 'lines', line: { color: K.navy, width: 1.2 }, name: 'unit circle: roots inside → not stationary', xaxis: ax.xa, yaxis: ax.ya, hoverinfo: 'skip' }];
  ['A', 'B', 'C', 'Sunspots'].forEach(k => { const zz = rootsZ(...ex[k].phi);
    tr.push({ x: zz.map(r => r[0]), y: zz.map(r => r[1]), mode: 'markers', marker: { color: AR2COL[k], size: 10 }, name: k, xaxis: ax.xa, yaxis: ax.ya, hovertemplate: `${k}: z = %{x:.3f} %{y:+.3f}i<extra></extra>` }); });
  tr.push({ x: z.map(r => r[0]), y: z.map(r => r[1]), mode: 'markers', marker: { color: K.navy, size: 13, symbol: 'x-thin', line: { color: K.navy, width: 3 } }, name: 'your model', xaxis: ax.xa, yaxis: ax.ya, hovertemplate: 'z = %{x:.3f} %{y:+.3f}i<extra></extra>' });
  G.lay.shapes.push(hl(ax, 0, K.navy, 'solid', 0.5), vl(ax, 0, K.navy, 'solid', 0.5));
  plot('xr-chart', tr, G.lay);
  out('xr-out', [X().xr(nf(a, 3), nf(b, 3), zs, nz(Math.min(...md), 3), st)]);
}

// ---------------------------------------------------------------- Worked example: theta = 2 or theta = 0.5?
function blkXTheta() {
  const th = +$('xt-th').value, s2 = +$('xt-s2').value; set('xt-th', nf(th, 2)); set('xt-s2', nf(s2, 2));
  const g0 = s2 * (1 + th * th), g1 = th * s2, S = [X().xt1(nf(th, 2), nf(s2, 2), nz(g0, 2), nz(g1, 2))];
  if (Math.abs(th) < 1e-9) { steps('xt-ws', S.concat([X().xt0()])); }
  else {
    const t2 = 1 / th, v2 = th * th * s2;
    S.push(X().xt2(nz(t2, 3), nz(v2, 3), nz(v2 * (1 + t2 * t2), 2), nz(t2 * v2, 2)), X().xt3(nz(th / (1 + th * th), 3)));
    const inv = Math.abs(th) < 1 ? th : t2;
    S.push(Math.abs(Math.abs(th) - 1) < 1e-9 ? X().xt4u() : X().xt4(nz(inv, 3), range(0, 2, 1).map(j => nz((-inv) ** j, 3)).join('; '), nz(Math.abs(th) < 1 ? t2 : th, 3), range(0, 2, 1).map(j => nz((-(Math.abs(th) < 1 ? t2 : th)) ** j, 2)).join('; ')));
    if (Math.abs(Math.abs(th) - 1) > 1e-9) S.push(X().xt5(nz(inv, 3), nz(Math.abs(th) < 1 ? s2 : v2, 3)));
    steps('xt-ws', S);
  }
  const jj = range(0, 8, 1), t2 = Math.abs(th) > 1e-9 ? 1 / th : null;
  table('xt-tb', ['j', `(−θ)<sup>j</sup>, θ = ${nz(th, 2)}`, t2 === null ? '—' : `(−θ)<sup>j</sup>, θ = ${nz(t2, 3)}`],
    jj.map(j => [txt(j), num((-th) ** j, 3), t2 === null ? txt('—') : num((-t2) ** j, 3)]), { cap: X().xtCap });
  const G = grid('xt-chart', [{ r: 0, c: 0, t: 'AR(∞) weights |(−θ)<sup>j</sup>| of the two representations, log scale', xt: 'j', y: { type: 'log' } }], { cols: 1, h: 300 });
  const ax = G.ax[0], tr = [];
  if (Math.abs(th) > 1e-9) [[th, K.red], [t2, K.blue]].forEach(([t, c]) => tr.push({ x: jj, y: jj.map(j => Math.abs(t) ** j), mode: 'lines+markers', marker: { size: 6, color: c }, line: { color: c, width: 1.6 }, name: `θ = ${+t.toFixed(3)}${Math.abs(t) < 1 ? ' (invertible)' : ''}`, xaxis: ax.xa, yaxis: ax.ya, hovertemplate: 'j = %{x}: %{y:.3g}<extra></extra>' }));
  G.lay.shapes.push(hl(ax, 1, K.navy, 'dot', 1));
  plot('xt-chart', tr, G.lay);
  out('xt-out', [Math.abs(th) < 1e-9 ? X().xt0() : X().xt(nf(th, 2), nz(1 / th, 3), nz(th / (1 + th * th), 3), nz(g0, 2), nz(g1, 2))]);
}

// ---------------------------------------------------------------- The psi weights of an ARMA(1,1)
function blkXPsi() {
  const phi = +$('xp-phi').value, th = +$('xp-th').value; set('xp-phi', nf(phi, 2)); set('xp-th', nf(th, 2));
  const ps = psiW([phi], [th], 13), tot = 1 + (phi + th) ** 2 / (1 - phi * phi);
  steps('xp-ws', [X().xp1(nf(phi, 2), pn(th, 2), nz(phi + th, 2)), X().xp2(nf(phi, 2), pn(ps[1], 2), nz(ps[2], 2), nz(ps[3], 2)),
    X().xp3(pn(phi + th, 2), nf(phi * phi, 2), nz(tot, 2)), Math.abs(phi + th) < 1e-9 ? X().xpCF() : X().xp4(nz(ps[1] ** 2 / tot * 100, 1))]);
  let cs = 0;
  table('xp-tb', ['j', X().rec, 'ψ<sub>j</sub>', 'ψ<sub>j</sub>²', 'Σ<sub>i≤j</sub> ψ<sub>i</sub>²', X().share], ps.map((v, j) => { cs += v * v;
    const r = j === 0 ? '1' : j === 1 ? `φ + θ = ${nf(phi, 2)}${sg(th, 2)}` : `φψ<sub>${j - 1}</sub> = ${nf(phi, 2)} · ${pn(ps[j - 1], 3)}`;
    return [txt(j), txt(r), num(v, 3), num(v * v, 3), num(cs, 3), txt(nz(100 * cs / tot, 1) + '%')]; }), { cap: X().xpCap(nz(tot, 3)) });
  const jj = range(0, 30, 1), P = psiW([phi], [th], 31); let c2 = 0; const cum = P.map(v => (c2 += v * v));
  const G = grid('xp-chart', [{ r: 0, c: 0, t: `ARMA(1,1), φ = ${phi}, θ = ${th}: cumulative sum of ψ<sub>j</sub>²`, xt: 'j', y: { rangemode: 'tozero' } }], { cols: 1, h: 300 });
  const ax = G.ax[0]; G.lay.shapes.push(hl(ax, tot, K.red, 'dash', 1.2));
  plot('xp-chart', [{ x: jj, y: cum, mode: 'lines+markers', marker: { size: 5, color: K.purple }, line: { color: K.purple, width: 1.6 }, name: 'Σ ψ<sub>i</sub>², i ≤ j', xaxis: ax.xa, yaxis: ax.ya, hovertemplate: 'j = %{x}: %{y:.3f}<extra></extra>' },
    leg(`γ(0)/σ² = ${tot.toFixed(2)}`, { color: K.red, dash: 'dash', width: 1.2 })], G.lay);
  out('xp-out', [X().xp(nf(phi, 2), nf(th, 2), ps.slice(1, 5).map(v => nz(v, 2)).join('; '), nz(tot, 2))]);
}

// ---------------------------------------------------------------- Moments of the ARMA(1,1)
function blkXArma11() {
  const phi = +$('xm-phi').value, th = +$('xm-th').value; set('xm-phi', nf(phi, 2)); set('xm-th', nf(th, 2));
  const den = 1 + 2 * phi * th + th * th, g0 = den / (1 - phi * phi), r1 = Math.abs(phi + th) < 1e-12 ? 0 : (1 + phi * th) * (phi + th) / den, hh = range(1, 10, 1), r = hh.map(h => r1 * phi ** (h - 1));
  steps('xm-ws', [X().xm1(nf(phi, 2), pn(th, 2), nz(den, 2), nz(1 - phi * phi, 2), nz(g0, 2)), X().xm2(nz(1 + phi * th, 2), pn(phi + th, 2), nz(den, 2), nz(r1, 3)),
    X().xm3(nf(phi, 2), nz(r[1], 3), nz(r[2], 3)), Math.abs(phi + th) < 1e-9 ? X().xmCF(nf(phi, 2), nf(th, 2)) : X().xm4(nz(r1, 3), nf(phi, 2), th > 1e-9 ? 1 : th < -1e-9 ? -1 : 0, nz(th, 2))]);
  table('xm-tb', ['h', 'ρ(h), ARMA(1,1)', 'φ<sup>h</sup>, AR(1)', 'ρ(h)/φ<sup>h</sup>'], hh.slice(0, 8).map((h, i) => [txt(h), num(r[i], 3), num(phi ** h, 3), Math.abs(phi) > 1e-9 ? num(r[i] / phi ** h, 3) : txt('—')]), { cap: X().xmCap });
  const G = grid('xm-chart', [{ r: 0, c: 0, t: `ACF of ARMA(1,1), φ = ${phi}, θ = ${th}, and of AR(1) with the same φ`, xt: 'lag h', y: { range: [-1, 1] } }], { cols: 1, h: 300 });
  const ax = G.ax[0]; G.lay.shapes.push(hl(ax, 0, K.navy, 'solid', 0.8));
  plot('xm-chart', [{ x: hh, y: r, type: 'bar', width: 0.55, marker: { color: K.purple }, name: 'ARMA(1,1)', xaxis: ax.xa, yaxis: ax.ya, hovertemplate: 'ρ(%{x}) = %{y:.3f}<extra></extra>' },
    { x: hh, y: hh.map(h => phi ** h), mode: 'markers', marker: { color: K.orange, size: 8 }, name: 'AR(1): φ<sup>h</sup>', xaxis: ax.xa, yaxis: ax.ya, hovertemplate: 'φ^%{x} = %{y:.3f}<extra></extra>' }], G.lay);
  out('xm-out', [X().xm(nf(phi, 2), nf(th, 2), nz(g0, 2), nz(r1, 3), nz(r[1], 3))]);
}

// ---------------------------------------------------------------- Identification: the ACF-PACF table
function blkXIdent() {
  const name = $('it-m').value, Tn = +$('it-T').value; set('it-T', Tn);
  const { phi, theta } = D.models[name], rt = acfTheo(phi, theta, 10), pt = pacf(rt), x = ptPath(name, Tn), r = acf(x, 10), pc = pacf(r), b = 1.96 / Math.sqrt(Tn);
  const fam = phi.length && theta.length ? 3 : phi.length ? 1 : 2;
  table('it-tb1', [X().model, 'ACF', 'PACF'], X().idRows.map((rw, i) => hiRow(rw.map(v => ({ v })), i === fam)), { cap: X().itCap1(name) });
  const big = (v, k) => (Math.abs(v) > b ? `<b>${nz(v, 2)}</b>` : nz(v, 2));
  table('it-tb2', ['h', 'ρ(h)', 'φ<sub>hh</sub>', 'ρ̂(h)', 'φ̂<sub>hh</sub>'], range(1, 10, 1).map(h => [txt(h), num(rt[h - 1], 3), num(pt[h - 1], 3), txt(big(r[h - 1])), txt(big(pc[h - 1]))]),
    { cap: X().itCap2(name, Tn, nz(b, 3)) });
  const outA = range(1, 10, 1).filter(h => Math.abs(r[h - 1]) > b), outP = range(1, 10, 1).filter(h => Math.abs(pc[h - 1]) > b);
  out('it-out', [X().it(name, X().idRows[fam][0], outA.length ? outA.join(', ') : '—', outP.length ? outP.join(', ') : '—', nz(b, 3))]);
}

// ---------------------------------------------------------------- Worked example: Yule-Walker on Romanian GDP growth
function olsAR(y, p) {             // conditional least squares: OLS of y_t on 1, y_{t-1}, ..., y_{t-p}
  const n = y.length - p, Xm = range(p, y.length - 1, 1).map(t => [1].concat(range(1, p, 1).map(i => y[t - i]))), yy = y.slice(p);
  const XX = Xm[0].map((_, i) => Xm[0].map((__, j) => sum(Xm.map(r => r[i] * r[j])))), Xy = Xm[0].map((_, i) => sum(Xm.map((r, t) => r[i] * yy[t])));
  const bh = solve(XX, Xy), e = yy.map((v, t) => v - sum(Xm[t].map((x, i) => x * bh[i]))), s2 = sum(e.map(v => v * v)) / (n - p - 1);
  const se = bh.map((_, i) => Math.sqrt(s2 * solve(XX, XX.map((__, j) => (j === i ? 1 : 0)))[i]));
  return { b: bh.slice(1), se: se.slice(1), s2: sum(e.map(v => v * v)) / n };
}
function ywAR(y, p) {
  const n = y.length, r = acf(y, p), m = mean(y), g0 = sum(y.map(v => (v - m) ** 2)) / (n - 1);
  const R = range(0, p - 1, 1).map(i => range(0, p - 1, 1).map(j => (i === j ? 1 : r[Math.abs(i - j) - 1]))), ph = solve(R, r.slice(0, p));
  const s2 = g0 * (1 - sum(ph.map((v, i) => v * r[i]))), se = ph.map((_, i) => Math.sqrt(s2 / (g0 * n) * solve(R, R.map((__, j) => (j === i ? 1 : 0)))[i]));
  return { r, g0, b: ph, se, s2 };
}
function blkXYW() {
  const k = +$('xy-end').value, p = +$('xy-p').value, g = D.gdp, y = g.y.slice(0, k + 1), n = y.length, full = n === g.y.length; set('xy-end', T.q(g.dates[k]));
  const W = ywAR(y, 2), [r1, r2] = W.r, det = 1 - r1 * r1, W1 = ywAR(y, 1);
  const S = [X().xy1(T.q(g.dates[0]), T.q(g.dates[k]), n, nf(r1, 4), nf(r2, 4), nf(W.g0, 2)), X().xy2(nf(r1, 4), nf(det, 3), nf(1 - Math.round(r1 * 100) ** 2 / 1e4, 3))];
  if (p === 2) S.push(X().xy3(nf(r1, 4), nf(1 - r2, 4), nf(det, 3), nz(W.b[0], 3)), X().xy4(nf(r2, 4), nf(r1 * r1, 4), nf(det, 3), nz(W.b[1], 3)),
    X().xy5(nf(W.g0, 2), nz(1 - W.b[0] * r1 - W.b[1] * r2, 4), nz(W.s2, 2)));
  else S.push(X().xy3a(nf(r1, 4)), X().xy5a(nf(W.g0, 2), nf(det, 4), nz(W1.s2, 2)));
  steps('xy-ws', S);
  const o1 = olsAR(y, 1), o2 = olsAR(y, 2), M = k2 => { const F = D.gdpfit[k2]; return { b: F.b.slice(1, 1 + +k2[0]), se: F.se.slice(1, 1 + +k2[0]), s2: F.b[F.b.length - 1] }; };
  const est = (lab, e, pp) => hiRow([lab, txt(`AR(${pp})`), num(e.b[0], 3), num(e.se[0], 3), pp > 1 ? num(e.b[1], 3) : txt('—'), pp > 1 ? num(e.se[1], 3) : txt('—'), num(e.s2, 2)], pp === p);
  const rows = [est(X().yw, W1, 1), est(X().yw, W, 2), est(X().cls, o1, 1), est(X().cls, o2, 2)];
  if (full) rows.push(est(X().ml, M('10'), 1), est(X().ml, M('20'), 2));
  table('xy-tb', [X().estim, X().model, 'φ̂<sub>1</sub>', 'se', 'φ̂<sub>2</sub>', 'se', 'σ̂²'], rows, { cap: X().xyCap(n), note: full ? X().xyNote : X().xyNote2 });
  out('xy-out', [X().xy(p, T.q(g.dates[k]), p === 2 ? `φ̂₁ = ${nz(W.b[0], 3)}, φ̂₂ = ${nz(W.b[1], 3)}` : `φ̂ = ${nz(W1.b[0], 3)}`, nz(p === 2 ? W.s2 : W1.s2, 2))]);
}

// ---------------------------------------------------------------- Maximum likelihood: AR(1) on GDP growth by hand
function ar1Lik(x) {               // exact profile log-likelihood of phi (mu by GLS, sigma^2 concentrated) and the conditional one (OLS)
  const n = x.length;
  const ex = phi => { const a = 1 - phi * phi; let s1 = 0; for (let t = 1; t < n; t++) s1 += x[t] - phi * x[t - 1];
    const mu = (a * x[0] + (1 - phi) * s1) / (a + (n - 1) * (1 - phi) ** 2); let S = a * (x[0] - mu) ** 2;
    for (let t = 1; t < n; t++) S += (x[t] - mu - phi * (x[t - 1] - mu)) ** 2;
    return { ll: -n / 2 * (Math.log(2 * Math.PI * S / n) + 1) + 0.5 * Math.log(a), mu, s2: S / n }; };
  const co = phi => { let my = 0, mz = 0; for (let t = 1; t < n; t++) { my += x[t]; mz += x[t - 1]; } my /= n - 1; mz /= n - 1; const c = my - phi * mz; let S = 0;
    for (let t = 1; t < n; t++) S += (x[t] - c - phi * x[t - 1]) ** 2; return { ll: -(n - 1) / 2 * (Math.log(2 * Math.PI * S / (n - 1)) + 1), c, s2: S / (n - 1) }; };
  return { ex, co };
}
function blkXML() {
  const k = +$('xl-end').value, g = D.gdp, y = g.y.slice(0, k + 1), n = y.length, full = n === g.y.length; set('xl-end', T.q(g.dates[k]));
  const L = ar1Lik(y), ph = golden(f => L.ex(f).ll, -0.99, 0.99, 1e-9), pc = golden(f => L.co(f).ll, -0.99, 0.99, 1e-9), E = L.ex(ph), C = L.co(pc), h = 1e-4;
  const d2 = (L.ex(ph + h).ll - 2 * E.ll + L.ex(ph - h).ll) / (h * h), se = 1 / Math.sqrt(-d2), sa = Math.sqrt((1 - ph * ph) / n);
  const S = [X().xl1(), X().xl2(n, nz(ph, 3), nz(E.mu, 2), nz(E.s2, 2), nz(E.ll, 2)), X().xl3(n - 1, nz(pc, 3), nz(C.s2, 2), nz(C.ll, 2)),
    X().xl4(nz(-d2, 1), nz(se, 3), nz(ph, 3), nz(sa, 3)), X().xl5(nz(ph, 3), nz(se, 3), nz(ph / se, 1), nz(ph - 1.96 * se, 3), nz(ph + 1.96 * se, 3))];
  if (full) S.push(X().xl6(nz(D.gdpfit['10'].se[1], 3)));
  steps('xl-ws', S);
  const grd = range(Math.max(-0.5, ph - 0.45), Math.min(0.99, ph + 0.35), 0.005);
  const G = grid('xl-chart', [{ r: 0, c: 0, t: `AR(1) on GDP growth up to ${g.dates[k]}: log-likelihood of φ, relative to its maximum`, xt: 'φ', yt: 'ln L(φ) − ln L(φ̂)', y: { range: [-8, 0.5] } }], { cols: 1, h: 340, legendRows: 2 });
  const ax = G.ax[0];
  G.lay.shapes.push(hl(ax, -1.92, K.orange, 'dot', 1.2), vl(ax, ph, K.blue, 'dash', 1));
  plot('xl-chart', [{ x: grd, y: grd.map(f => L.ex(f).ll - E.ll), mode: 'lines', line: { color: K.blue, width: 2 }, name: 'exact (Gaussian MLE)', xaxis: ax.xa, yaxis: ax.ya, hovertemplate: 'φ = %{x:.3f}: %{y:.2f}<extra></extra>' },
    { x: grd, y: grd.map(f => L.co(f).ll - C.ll), mode: 'lines', line: { color: K.green, width: 1.5, dash: 'dash' }, name: 'conditional (first factor dropped)', xaxis: ax.xa, yaxis: ax.ya, hovertemplate: 'φ = %{x:.3f}: %{y:.2f}<extra></extra>' },
    { x: grd, y: grd.map(f => -((f - ph) ** 2) / (2 * se * se)), mode: 'lines', line: { color: K.red, width: 1.2, dash: 'dot' }, name: 'quadratic: −(φ − φ̂)²/(2 se²)', xaxis: ax.xa, yaxis: ax.ya, hoverinfo: 'skip' },
    leg('−1.92 = −χ²<sub>0.95</sub>(1)/2', { color: K.orange, dash: 'dot', width: 1.2 })], G.lay);
  out('xl-out', [X().xl(T.q(g.dates[k]), nz(ph, 3), nz(se, 3), nz(sa, 3), nz(pc, 3))]);
}

// ---------------------------------------------------------------- Information criteria: AIC and BIC from the log-likelihood
function blkXIC() {
  const key = $('xi-m').value, F = gfit(key), n = F.nobs, ks = GKEYS().map(gfit);
  const ba = ks.reduce((a, b) => (b.aic < a.aic ? b : a)), bb = ks.reduce((a, b) => (b.bic < a.bic ? b : a)), lnT = Math.log(n);
  const nm = f => modLab(f.p + '' + f.q);
  const S = [X().xi1(nm(F), F.p, F.q, F.k, nz(F.llf, 2)), X().xi2(nz(-2 * F.llf, 2), F.k, nz(F.aic, 2)), X().xi3(nz(-2 * F.llf, 2), F.k, nz(lnT, 3), nz(F.bic, 2)),
    X().xi4(nz(F.aic, 2), F.k, n, nz(F.aicc, 2))];
  const other = [ba, bb].find(f => f !== F && !(f.p === F.p && f.q === F.q));
  if (other) S.push(X().xi5(nm(other), nz(other.llf - F.llf, 2), other.k - F.k, nz(other.aic - F.aic, 2), nz(other.bic - F.bic, 2)));
  steps('xi-ws', S);
  const cols = [F].concat([ba, bb].filter((f, i, a) => !(f.p === F.p && f.q === F.q) && a.findIndex(g2 => g2.p === f.p && g2.q === f.q) === i));
  const row = (lab, fn, d) => [lab].concat(cols.map(f => num(fn(f), d)));
  table('xi-tb', [''].concat(cols.map(f => nm(f) + (f === ba || (f.p === ba.p && f.q === ba.q) ? ' · AIC' : '') + (f.p === bb.p && f.q === bb.q ? ' · BIC' : ''))),
    [row('k', f => f.k, 0), row('ln L', f => f.llf, 2), row('−2 ln L', f => -2 * f.llf, 2), row(X().penA, f => 2 * f.k, 2), row(X().penB, f => f.k * lnT, 2),
      hiRow(row('AIC', f => f.aic, 2), true), hiRow(row('BIC', f => f.bic, 2), true), row('AICc', f => f.aicc, 2)], { cap: X().xiCap(n) });
  out('xi-out', [X().xi(nm(F), nz(F.aic, 2), nz(F.bic, 2), nm(ba), nz(ba.aic, 2), nm(bb), nz(bb.bic, 2))]);
}

// ---------------------------------------------------------------- Residual diagnostics: Ljung-Box with m - p - q df and Jarque-Bera
function blkXDiag() {
  const key = $('dt-mod').value, p = +key[0], q = +key[1], e = D.gdp.models[key].resid, n = e.length, fit = p + q;
  const mEl = $('dt-m'); mEl.min = fit + 1; if (+mEl.value <= fit) mEl.value = fit + 1; const m = +mEl.value; set('dt-m', m);
  const rows = range(fit + 1, 16, 1).map(mm => { const Q = ljungBox(e, mm).q, df = mm - fit;
    return hiRow([txt(mm), num(Q, 2), txt(df), txt(fp(chi2sf(Q, df))), num(chi2ppf(0.95, df), 2), txt(fp(chi2sf(Q, mm))), txt(chi2sf(Q, df) < 0.05 ? X().rej : X().norej)], mm === m); });
  table('dt-tb', ['m', 'Q*(m)', 'df = m − p − q', 'p', 'χ²<sub>0.95</sub>(df)', X().pWrong, X().at5], rows, { cap: X().dtCap(modLab(key), n) });
  const mu = mean(e), s2 = pvar(e), sk = sum(e.map(v => (v - mu) ** 3)) / n / s2 ** 1.5, ku = sum(e.map(v => (v - mu) ** 4)) / n / s2 ** 2, jb = n / 6 * (sk * sk + (ku - 3) ** 2 / 4);
  const Q = ljungBox(e, m).q, Qs = ljungBox(e.map(v => v * v), 8).q;
  steps('dt-ws', [X().dt1(m, nz(Q, 2), m, fit, m - fit, fp(chi2sf(Q, m - fit)), fp(chi2sf(Q, m))), X().dt2(nz(sk, 2), nz(ku, 2)),
    X().dt3(n, nz(sk * sk, 3), nz((ku - 3) ** 2 / 4, 3), nz(jb, 1), jb > 5.991, pe(Math.exp(-jb / 2))), X().dt4(nz(Qs, 2), pe(chi2sf(Qs, 8)))]);
  out('dt-out', [X().dt(modLab(key), m, nz(Q, 2), m - fit, pe(chi2sf(Q, m - fit)), nz(jb, 1), pe(Math.exp(-jb / 2)))]);
}

// ---------------------------------------------------------------- Worked example: forecasting an AR(1)
function blkXFc() {
  const c = +$('xf-c').value, phi = +$('xf-phi').value, s2 = +$('xf-s2').value, xT = +$('xf-x').value, H = 10;
  set('xf-c', nf(c, 1)); set('xf-phi', nf(phi, 2)); set('xf-s2', nf(s2, 2)); set('xf-x', nf(xT, 1));
  const mu = c / (1 - phi), hs = range(1, H, 1), f = hs.map(h => mu + phi ** h * (xT - mu)), v = hs.map(h => s2 * (1 - phi ** (2 * h)) / (1 - phi * phi)), g0 = s2 / (1 - phi * phi), z = 1.96;
  const ci = h => `[${nz(f[h - 1] - z * Math.sqrt(v[h - 1]), 2)}; ${nz(f[h - 1] + z * Math.sqrt(v[h - 1]), 2)}]`;
  steps('xf-ws', [X().xf1(nf(c, 1), pn(phi, 2), nz(mu, 2), nf(xT, 1), pn(xT - mu, 2)), X().xf2(nz(mu, 2), pn(phi, 2), pn(xT - mu, 2), nz(f[0], 2), ns(phi * phi), nz(f[1], 2), nz(f[9], 2)),
    X().xf3(nf(s2, 2), nz(v[0], 2), nf(phi, 2), nz(v[1], 2)), X().xf4(ci(1), ci(2)), X().xf5(nz(mu, 2), nz(Math.sqrt(g0), 2))]);
  table('xf-tb', ['h', 'X̂<sub>T+h|T</sub>', 'σ<sub>h</sub>²', 'σ<sub>h</sub>', X().lo95, X().hi95, X().width], hs.map((h, i) => [txt(h), num(f[i], 2), num(v[i], 3), num(Math.sqrt(v[i]), 3),
    num(f[i] - z * Math.sqrt(v[i]), 2), num(f[i] + z * Math.sqrt(v[i]), 2), num(2 * z * Math.sqrt(v[i]), 2)]).concat([[txt('∞'), num(mu, 2), num(g0, 3), num(Math.sqrt(g0), 3), num(mu - z * Math.sqrt(g0), 2), num(mu + z * Math.sqrt(g0), 2), num(2 * z * Math.sqrt(g0), 2)]]));
  const hh = range(1, 30, 1), w = hh.map(h => 2 * z * Math.sqrt(s2 * (1 - phi ** (2 * h)) / (1 - phi * phi)));
  const G = grid('xf-chart', [{ r: 0, c: 0, t: `AR(1), φ = ${phi}, σ² = ${s2}: width of the 95% forecast interval`, xt: 'horizon h', yt: '2 × 1.96 σ<sub>h</sub>', y: { rangemode: 'tozero' } }], { cols: 1, h: 300 });
  const ax = G.ax[0]; G.lay.shapes.push(hl(ax, 2 * z * Math.sqrt(g0), K.red, 'dash', 1.2));
  plot('xf-chart', [{ x: hh, y: w, mode: 'lines+markers', marker: { size: 5, color: K.blue }, line: { color: K.blue, width: 1.6 }, name: 'interval width', xaxis: ax.xa, yaxis: ax.ya, hovertemplate: 'h = %{x}: %{y:.2f}<extra></extra>' },
    leg(`long run: 2 × 1.96 √γ(0) = ${(2 * z * Math.sqrt(g0)).toFixed(2)}`, { color: K.red, dash: 'dash', width: 1.2 })], G.lay);
  out('xf-out', [X().xf(nz(f[0], 2), ci(1), nz(f[1], 2), ci(2), nz(f[9], 2), nz(mu, 2))]);
}

// ---------------------------------------------------------------- Step 2: estimation and information criteria (full grid p, q <= 3)
function blkXGrid() {
  const cr = $('gg-c').value, srt = $('gg-s').value, ks = GKEYS().map(k => Object.assign(gfit(k), { key: k }));
  const ba = ks.reduce((a, b) => (b.aic < a.aic ? b : a)), bb = ks.reduce((a, b) => (b.bic < a.bic ? b : a)), mn = Math.min(...ks.map(f => f[cr]));
  const L = srt === 'pq' ? ks : [...ks].sort((a, b) => a[cr] - b[cr]);
  table('gg-tb', [X().model, 'k', 'ln L', 'AIC', 'BIC', 'AICc', 'σ̂', 'LB Q*(8)', 'df', 'p', 'Δ' + cr.toUpperCase()], L.map(f => { const e = D.gdp.models[f.key].resid, Q = ljungBox(e, 8).q, df = 8 - f.p - f.q;
    return hiRow([modLab(f.key) + (f === ba ? ' · AIC' : '') + (f === bb ? ' · BIC' : ''), txt(f.k), num(f.llf, 2), num(f.aic, 2, f === ba), num(f.bic, 2, f === bb), num(f.aicc, 2), num(Math.sqrt(f.sig2), 2), num(Q, 2), txt(df), txt(fp(chi2sf(Q, df))), num(f[cr] - mn, 2)], f[cr] === mn); }),
    { cap: X().ggCap(ks[0].nobs) });
  const z = range(0, 3, 1).map(p => range(0, 3, 1).map(q => ks.find(f => f.p === p && f.q === q)[cr] - mn));
  const G = grid('gg-chart', [{ r: 0, c: 0, t: `${cr.toUpperCase()} − min ${cr.toUpperCase()} over the (p, q) grid`, xt: 'MA order q', yt: 'AR order p', x: { dtick: 1 }, y: { dtick: 1 } }], { cols: 1, h: 340, legend: false });
  const ax = G.ax[0];
  G.lay.annotations.push(...z.flatMap((r, p) => r.map((v, q) => ({ xref: ax.xa, yref: ax.ya, x: q, y: p, text: v.toFixed(1), showarrow: false, font: { size: 12, color: v < 12 ? '#ffffff' : K.navy } }))));
  plot('gg-chart', [{ z, x: [0, 1, 2, 3], y: [0, 1, 2, 3], type: 'heatmap', colorscale: [[0, '#1A3A6E'], [0.15, '#17A2B8'], [1, '#F2F5FA']], zmin: 0, zmax: 70, xaxis: ax.xa, yaxis: ax.ya,
    colorbar: { thickness: 10, len: 0.8, tickfont: { size: 10, color: K.navy } }, hovertemplate: 'ARMA(%{y},%{x}): Δ = %{z:.2f}<extra></extra>' }], G.lay);
  out('gg-out', [X().gg(modLab(ba.key), nz(ba.aic, 2), modLab(bb.key), nz(bb.bic, 2), nz(gfit('03').aic - ba.aic, 2), cr.toUpperCase())]);
}

// ---------------------------------------------------------------- Interpreting the estimated MA(3): coefficients, standard errors, roots
function blkXMA3() {
  const key = $('mq-m').value, F = gfit(key), nm = X().pname, z = normPpf(0.975);
  const rows = F.names.map((p, i) => { const t = F.b[i] / F.se[i];
    return [nm(p), num(F.b[i], 3), num(F.se[i], 3), num(t, 2), txt(p === 'sigma2' ? '—' : `[${nz(F.b[i] - z * F.se[i], 2)}; ${nz(F.b[i] + z * F.se[i], 2)}]`), txt(p === 'sigma2' ? '—' : yn(Math.abs(t) > z))]; });
  table('mq-tb', [X().param, X().est, 'se (OPG)', 't', X().ci95, '|t| > 1.96'], rows, { cap: X().mqCap(modLab(key)),
    note: X().mqNote(F.ma_mod.length ? F.ma_mod.map(v => nz(v, 3)).join('; ') : '—', F.ma_mod.length ? yn(Math.min(...F.ma_mod) > 1) : '—', F.ar_mod.length ? F.ar_mod.map(v => nz(v, 3)).join('; ') : '—', F.ar_mod.length ? yn(Math.min(...F.ar_mod) > 1) : '—') });
  const ths = F.names.map((p, i) => (p.startsWith('ma.') ? F.b[i] : null)).filter(v => v !== null);
  const S = [X().mq1(modLab(key), nz(F.b[0], 2), nz(F.se[0], 2), nz(F.b[0] - z * F.se[0], 2), nz(F.b[0] + z * F.se[0], 2)), X().mq2(nz(Math.sqrt(F.sig2), 2), nz(F.sig2, 3))];
  if (ths.length) S.push(X().mq3(ths.map(v => nz(v, 3)).join('; '), nz(Math.min(...F.ma_mod), 3), Math.min(...F.ma_mod) > 1, ths.some(v => Math.abs(v) > 1)));
  steps('mq-ws', S);
  out('mq-out', [X().mq(modLab(key), F.names.filter(p => p !== 'const' && p !== 'sigma2').map((p, i) => `${nm(p)} ${nz(F.b[F.names.indexOf(p)], 3)} (${nz(F.se[F.names.indexOf(p)], 3)})`).join(', ') || '—')]);
}

// ---------------------------------------------------------------- Interpreting the GDP forecasts: table and interval width
function blkXGdpFc() {
  const g = D.gdp, O = Object.keys(g.origins), k = +$('ft-o').value, o = O[k], lv = +$('ft-lv').value, z = normPpf(1 - (1 - lv) / 2), last = k === O.length - 1;
  set('ft-o', T.q(o));
  const fq = range(1, 8, 1).map(h => addQ(o, h)), real = fq.map(q => { const j = g.dates.indexOf(q); return j < 0 ? null : g.y[j]; });
  const A = g.origins[o]['03'], B = g.origins[o]['10'];
  table('ft-tb', ['h', X().quarter, 'MA(3)', 'se', `${Math.round(100 * lv)}%`, 'AR(1)', 'se', `${Math.round(100 * lv)}%`, X().obs],
    fq.map((q, h) => [txt(h + 1), txt(T.q(q)), num(A.f[h], 2), num(A.se[h], 2), txt(`[${nz(A.f[h] - z * A.se[h], 1)}; ${nz(A.f[h] + z * A.se[h], 1)}]`),
      num(B.f[h], 2), num(B.se[h], 2), txt(`[${nz(B.f[h] - z * B.se[h], 1)}; ${nz(B.f[h] + z * B.se[h], 1)}]`), real[h] === null ? txt('—') : num(real[h], 2)]), { cap: X().ftCap(T.q(o)) });
  const S = [X().ft1(nz(A.se[0], 2), nz(z, 2), `[${nz(A.f[0] - z * A.se[0], 1)}; ${nz(A.f[0] + z * A.se[0], 1)}]`), X().ft2(nz(A.f[3], 2), nz(A.se[3], 2), nz(A.se[7], 2))];
  if (last) { const F = gfit('03'), th = F.b.slice(1, 4), s = 1 + sum(th.map(v => v * v));
    S.push(X().ft3(nz(F.sig2, 3), th.map(v => nz(v * v, 3)).join(' + '), nz(s, 3), nz(Math.sqrt(F.sig2 * s), 2), nz(F.b[0], 2), nz(z, 2), `[${nz(F.b[0] - z * Math.sqrt(F.sig2 * s), 1)}; ${nz(F.b[0] + z * Math.sqrt(F.sig2 * s), 1)}]`)); }
  S.push(X().ft4(nz(B.se[0], 2), nz(B.se[7], 2)));
  steps('ft-ws', S);
  const hs = range(1, 8, 1), G = grid('ft-chart', [{ r: 0, c: 0, t: `Width of the ${Math.round(100 * lv)}% interval, forecasts from ${o}`, xt: 'horizon h (quarters)', yt: 'percentage points', y: { rangemode: 'tozero' } }], { cols: 1, h: 300 });
  const ax = G.ax[0];
  plot('ft-chart', [['MA(3)', A, K.red], ['AR(1)', B, K.blue]].map(([lab, F, c]) => ({ x: hs, y: F.se.map(v => 2 * z * v), mode: 'lines+markers', marker: { size: 6, color: c }, line: { color: c, width: 1.6 }, name: lab, xaxis: ax.xa, yaxis: ax.ya, hovertemplate: lab + ', h = %{x}: %{y:.2f}<extra></extra>' })), G.lay);
  const er = (F) => { const e = real.map((v, h) => (v === null ? null : v - F.f[h])).filter(v => v !== null); return e.length ? nz(Math.sqrt(mean(e.map(v => v * v))), 2) : null; };
  out('ft-out', [X().ft(T.q(o), Math.round(100 * lv), nz(2 * z * A.se[0], 1), nz(2 * z * A.se[7], 1), nz(2 * z * B.se[0], 1), nz(2 * z * B.se[7], 1), er(A), er(B), real.filter(v => v !== null).length)]);
}

// ---------------------------------------------------------------- Interpreting the AR(1) for daily returns: table
function ar1Fit(r) {               // conditional LS with the classical and the OPG (outer product of gradients) standard errors
  const n = r.length - 1, x = r.slice(0, -1), y = r.slice(1), mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0; for (let t = 0; t < n; t++) { sxy += (x[t] - mx) * (y[t] - my); sxx += (x[t] - mx) ** 2; }
  const phi = sxy / sxx, c = my - phi * mx, e = y.map((v, t) => v - c - phi * x[t]), s2 = sum(e.map(v => v * v)) / n;
  const O = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  e.forEach((v, t) => { const s = [v / s2, v * x[t] / s2, -0.5 / s2 + v * v / (2 * s2 * s2)]; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) O[i][j] += s[i] * s[j]; });
  return { phi, e, n, seC: Math.sqrt(sum(e.map(v => v * v)) / (n - 2) / sxx), seO: Math.sqrt(solve(O, [0, 1, 0])[1]), r2: 1 - pvar(e) / pvar(y) };
}
function blkXRet() {
  let y0 = +$('rx-y0').value, y1 = +$('rx-y1').value; if (y1 < y0 + 2) { y1 = y0 + 2; $('rx-y1').value = y1; } set('rx-y0', y0); set('rx-y1', y1);
  const C = [['bet', 'BET'], ['eurron', 'EUR/RON']].map(([k, lab]) => {
    const R = D.ret[k], r = R.r.filter((_, j) => +R.dates[j].slice(0, 4) >= y0 && +R.dates[j].slice(0, 4) <= y1), full = r.length === R.r.length, F = ar1Fit(r);
    const e = full ? r.slice(1).map((v, t) => v - R.mu - R.phi * (r[t] - R.mu)) : F.e, phi = full ? R.phi : F.phi, seO = full ? D.retse[k].opg : F.seO, seI = full ? D.retse[k].oim : F.seC;
    const m = mean(e), s2 = pvar(e), ku = sum(e.map(v => (v - m) ** 4)) / e.length / (s2 * s2), lb = ljungBox(e, 10).q, lbs = ljungBox(e.map(v => v * v), 10).q;
    return { k, lab, n: r.length, phi, seO, seI, r2: 1 - pvar(e) / pvar(r.slice(1)), lb, lbs, ku, full, d0: R.dates.find(d => +d.slice(0, 4) >= y0) };
  });
  const row = (lab, f) => [lab].concat(C.map(f));
  table('rx-tb', ['', 'BET', 'EUR/RON'], [row(X().nobs, c => txt(c.n)), row('φ̂', c => num(c.phi, 3)), hiRow(row(X().seOPG, c => num(c.seO, 4)), true), hiRow(row(X().tOPG, c => num(c.phi / c.seO, 1)), true),
    row(X().seCl, c => num(c.seI, 4)), row(X().tCl, c => num(c.phi / c.seI, 1)), row('√((1 − φ̂²)/T)', c => num(Math.sqrt((1 - c.phi ** 2) / c.n), 4)), row('R² (%)', c => num(100 * c.r2, 1)),
    row(X().lbRes, c => txt(`${nz(c.lb, 1)} (${fp(chi2sf(c.lb, 9))})`)), row(X().lbSq, c => num(c.lbs, 0)), row(X().kurt, c => num(c.ku, 1))], { cap: X().rxCap(y0, y1), note: X().rxNote(C[0].full) });
  steps('rx-ws', C.map(c => X().rx1(c.lab, nz(c.phi, 3), nz(c.seO, 4), nz(c.phi / c.seO, 1), nz(c.seI, 4), nz(c.phi / c.seI, 1), c.n, nz(Math.sqrt((1 - c.phi ** 2) / c.n), 4))));
  out('rx-out', C.map(c => X().rx(c.lab, c.n, nz(c.phi, 3), nz(c.phi / c.seI, 1), nz(100 * c.r2, 1))));
}

// ================================================================= wiring
const ALL = [blkAR1, blkAR2, blkMA1, blkMArho, blkPsi, blkPatterns, blkEst, blkIC, blkLB, blkFct, blkGdpId, blkGdpDiag, blkGdpFc, blkInfl, blkRet, blkSun];
// blocks of the worked examples and result tables (own frames): [anchor, prefix, function, controls, player parameter, values, ms]
const XB = [['wx-ar1', 'xa', blkXAR1, ['xa-c', 'xa-phi', 'xa-s2', 'xa-h'], 'xa-phi', range(-0.9, 0.99, 0.03), 300],
  ['ar-roots', 'xr', blkXRoots, ['xr-f1', 'xr-f2'], 'xr-f2', range(-0.95, 0.95, 0.05), 380],
  ['wx-theta', 'xt', blkXTheta, ['xt-th', 'xt-s2'], 'xt-th', range(-3, 3, 0.1), 260],
  ['wx-psi', 'xp', blkXPsi, ['xp-phi', 'xp-th'], 'xp-th', range(-0.95, 0.95, 0.05), 380],
  ['wx-arma11', 'xm', blkXArma11, ['xm-phi', 'xm-th'], 'xm-th', range(-0.95, 0.95, 0.05), 380],
  ['ident-table', 'it', blkXIdent, ['it-m', 'it-T'], 'it-T', range(50, 500, 25), 450],
  ['wx-yw', 'xy', blkXYW, ['xy-end', 'xy-p'], 'xy-end', range(39, 105, 2), 300],
  ['ml-lik', 'xl', blkXML, ['xl-end'], 'xl-end', range(39, 105, 2), 300],
  ['wx-ic', 'xi', blkXIC, ['xi-m'], 'xi-m', ['00', '10', '20', '30', '01', '11', '02', '03', '13', '23', '33'], 1000],
  ['diag-table', 'dt', blkXDiag, ['dt-mod', 'dt-m'], 'dt-m', range(1, 16, 1), 500],
  ['wx-fc', 'xf', blkXFc, ['xf-c', 'xf-phi', 'xf-s2', 'xf-x'], 'xf-phi', range(0, 0.99, 0.03), 280],
  ['gdp-grid', 'gg', blkXGrid, ['gg-c', 'gg-s'], 'gg-c', ['aic', 'bic', 'aicc'], 1300],
  ['ma3-table', 'mq', blkXMA3, ['mq-m'], 'mq-m', ['03', '13', '23', '33', '10', '20', '11'], 1100],
  ['gdp-fc-table', 'ft', blkXGdpFc, ['ft-o', 'ft-lv'], 'ft-o', range(0, 48, 1), 350],
  ['ret-table', 'rx', blkXRet, ['rx-y0', 'rx-y1'], 'rx-y0', range(2000, 2023, 1), 600]];
function placeExtras() {                // web lecture: the extra blocks travel inside a chart block; move each to its own frame
  document.querySelectorAll('.xhold > section.xblk').forEach(sec => {
    const art = document.getElementById('s' + sec.dataset.frame), head = art && art.querySelector('.fhead');
    if (head) head.after(sec); else sec.closest('section.blk').after(sec);
  });
  document.querySelectorAll('.xhold').forEach(h => { if (!h.children.length) h.remove(); });
}
function init() {
  placeExtras();
  const has = id => !!$(id), on = (ids, f) => ids.forEach(id => $(id).addEventListener('input', f));
  const players = {}, P = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  on(['a1-p1', 'a1-p2', 'a1-p3'], blkAR1); P('a1-play', 'a1-p1', range(0, 0.99, 0.03), 260);
  const from0 = (btn, id, v0, v) => $(btn).addEventListener('click', () => { if (!players[btn].timer && +$(id).value === v0) $(id).value = v; }, true);   // ▶ from the slide state: start the sweep at its first value
  from0('a1-play', 'a1-p1', 0.9, 0.99);
  $('a2-m').addEventListener('input', () => { const m = AR2()[$('a2-m').value]; $('a2-f1').value = m.phi[0]; $('a2-f2').value = m.phi[1]; blkAR2(); });
  on(['a2-f1', 'a2-f2'], blkAR2); P('a2-play', 'a2-f2', range(-0.95, 0.3, 0.05), 380);
  on(['m1-th'], blkMA1); P('m1-play', 'm1-th', range(0, 2, 0.1), 380);
  on(['mr-th'], blkMArho); P('mr-play', 'mr-th', range(-4, 4, 0.1), 110);
  $('ps-m').addEventListener('input', () => { psiSync(); blkPsi(); }); on(['ps-a', 'ps-b', 'ps-c', 'ps-d'], blkPsi); psiSync();
  P('ps-play', 'ps-a', range(0, 0.95, 0.05).concat(range(0.9, 0.7, -0.05)), 380);     // ▶: the first coefficient of all four models, 0 -> 0.95 -> 0.7
  $('ps-play').addEventListener('click', () => { if ($('ps-m').value === 'MA(2)') { $('ps-m').value = 'ARMA(1,1)'; psiSync(); } }, true); from0('ps-play', 'ps-a', 0.7, 2);
  on(['pt-T'], blkPatterns); P('pt-play', 'pt-T', range(50, 500, 25), 450);
  $('pt-new').addEventListener('click', () => { ptSeed++; blkPatterns(); });
  on(['es-phi', 'es-th', 'es-T'], blkEst); P('es-play', 'es-phi', range(0.5, 0.98, 0.04), 900);
  on(['ic-f2', 'ic-T'], blkIC); P('ic-play', 'ic-f2', range(0.05, 0.45, 0.05), 1100);
  on(['lb-m', 'lb-a'], blkLB); P('lb-play', 'lb-m', range(3, 20, 1), 450); from0('lb-play', 'lb-m', 10, 20);
  on(['fc-x', 'fc-p1', 'fc-p2', 'fc-th'], blkFct); P('fc-play', 'fc-x', range(3, -4, -0.5).concat(range(-3.5, 3, 0.5)), 300);
  $('gi-end').max = D.gdp.y.length - 1; $('gi-end').value = D.gdp.y.length - 1; on(['gi-end'], blkGdpId); P('gi-play', 'gi-end', range(39, D.gdp.y.length - 1, 1), 180);
  on(['gd-mod', 'gd-df'], blkGdpDiag); P('gd-play', 'gd-mod', Object.keys(D.gdp.models).sort(), 1000);
  const nO = Object.keys(D.gdp.origins).length; $('gf-o').max = nO - 1; $('gf-o').value = nO - 1; on(['gf-o', 'gf-lv'], blkGdpFc); P('gf-play', 'gf-o', range(0, nO - 1, 1), 350);
  on(['in-p'], blkInfl); P('in-play', 'in-p', ['1', '2', '3', '4', '5', '6'], 1100);
  on(['rt-y0', 'rt-y1'], blkRet); P('rt-play', 'rt-y0', range(2000, 2023, 1), 500);
  on(['ss-p'], blkSun); P('ss-play', 'ss-p', range(1, 10, 1), 900);
  const PFX = { 'ar1': 'a1', 'ar2': 'a2', 'ma1': 'm1', 'ma1-rho': 'mr', 'psi': 'ps', 'patterns': 'pt', 'estimators': 'es', 'ic': 'ic', 'lb-df': 'lb',
    'forecast-mean': 'fc', 'gdp-ident': 'gi', 'gdp-diag': 'gd', 'gdp-forecast': 'gf', 'inflation': 'in', 'returns': 'rt', 'sunspots': 'ss' };
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if (has(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: { out: true } }, sp)); });
  const live = XB.filter(([, x]) => has(x + '-out'));
  live.forEach(([, x, f, ids, pid, vals, ms]) => { on(ids, f); P(x + '-play', pid, vals, ms); });
  if (has('xr-m')) $('xr-m').addEventListener('input', () => { const m = AR2()[$('xr-m').value]; $('xr-f1').value = m.phi[0]; $('xr-f2').value = m.phi[1]; blkXRoots(); });
  if (has('xy-end')) { $('xy-end').max = D.gdp.y.length - 1; $('xy-end').value = D.gdp.y.length - 1; }
  if (has('xl-end')) { $('xl-end').max = D.gdp.y.length - 1; $('xl-end').value = D.gdp.y.length - 1; }
  if (has('ft-o')) { const nO = Object.keys(D.gdp.origins).length; $('ft-o').max = nO - 1; $('ft-o').value = nO - 1; }
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  ALL.forEach(f => f());
  live.forEach(b => b[2]());
  let w = window.innerWidth, tm = null;
  window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(() => { if ((window.innerWidth < 720) !== (w < 720) || Math.abs(window.innerWidth - w) > 200) { w = window.innerWidth; ALL.forEach(f => f()); live.forEach(b => b[2]()); } }, 250); });
}
window.CH2 = { chi2sf, chi2ppf, psiW, acfTheo, ar2Roots, rootsZ, ar1Est, ma1Est, arOlsIC, lbRate, probplot, ar1Path, ptPath, esSim, icFreq, ar1Ols, olsAR, ywAR, ar1Lik, ar1Fit, gfit, ALL, XB };
if (!window.CH2_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
