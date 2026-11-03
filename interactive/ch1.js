// Chapter 1 interactive page (stochastic processes and stationarity): one block per chart slide, computed in the browser
// from ch1_data.js.
(function () {
'use strict';
const D = window.CH1_DATA;
const { sum, acf, pacf, rng, mean, sdev, nf: NF, pct, $, AX, layout, plot, out, Player, range, predict, exitTicket } = window.TSAI;
const T = window.CH1_TEXT;

// ---------------------------------------------------------------- colours of generate_all_charts.py / tsa_style.py
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', orange: '#E67E22', purple: '#8E44AD',
  teal: '#17A2B8', crimson: '#DC3545', navy: '#1F2A44' };
const PAL = [K.blue, K.red, K.green, K.amber, K.purple, K.orange, K.teal, K.crimson];
const rgba = (hex, a) => `rgba(${parseInt(hex.slice(1, 3), 16)},${parseInt(hex.slice(3, 5), 16)},${parseInt(hex.slice(5, 7), 16)},${a})`;

// ---------------------------------------------------------------- small helpers
const nf = (v, d = 1) => NF(Math.abs(v) < 0.5 * 10 ** -d ? 0 : v, d);                    // no '−0.00'
const sig = v => nf(v, Math.max(0, 2 - Math.floor(Math.log10(Math.abs(v) || 1))));       // three significant digits
const lags = n => Array.from({ length: n }, (_, i) => i + 1);
const pvar = x => { const m = mean(x); return sum(x.map(v => (v - m) ** 2)) / x.length; };
const skew = x => { const m = mean(x), n = x.length; let m2 = 0, m3 = 0; x.forEach(v => { const d = v - m; m2 += d * d; m3 += d * d * d; }); m2 /= n; m3 /= n; return m3 / m2 ** 1.5; };
const kurt = x => { const m = mean(x), n = x.length; let m2 = 0, m4 = 0; x.forEach(v => { const d = (v - m) ** 2; m2 += d; m4 += d * d; }); m2 /= n; m4 /= n; return m4 / (m2 * m2); };
const cumsum = x => { let s = 0; return x.map(v => (s += v)); };
const d100 = p => p.slice(1).map((v, i) => 100 * (v - p[i]));
const pe = p => (p < 0.001 ? 'p < ' + nf(0.001, 3) : 'p = ' + nf(p, 3));
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
function chi2ppf(p, k) { let lo = 0, hi = 10 * k + 100; for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; if (1 - chi2sf(m, k) < p) lo = m; else hi = m; } return (lo + hi) / 2; }
function lbq(x, m) { const n = x.length, r = acf(x, m); let q = 0; for (let k = 1; k <= m; k++) q += r[k - 1] ** 2 / (n - k); q *= n * (n + 2); return { q, p: chi2sf(q, m) }; }
const binom = (m, k, p) => { let c = 1; for (let i = 0; i < k; i++) c = c * (m - i) / (i + 1); return c * p ** k * (1 - p) ** (m - k); };
function days(o) { const out_ = [o.d0]; let t = Date.parse(o.d0 + 'T00:00:00Z'); o.gap.forEach(g => { t += g * 864e5; out_.push(new Date(t).toISOString().slice(0, 10)); }); return out_; }
const m2d = s => s + '-01';
const ar1 = (e, phi, x0) => { const x = new Array(e.length); x[0] = x0 === undefined ? e[0] : x0; for (let t = 1; t < e.length; t++) x[t] = phi * x[t - 1] + e[t]; return x; };
const joinPaths = (P, xs) => { const X = [], Y = []; P.forEach(p => { X.push(...xs, null); Y.push(...p, null); }); return { x: X, y: Y }; };

// ---------------------------------------------------------------- panels: one Plotly figure with several axes; stacked on narrow screens
// cells: [{ r, c, cs, t (title), xt, yt (axis titles), x, y (extra axis options) }]
function grid(id, cells, o) {
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

// ---------------------------------------------------------------- real series, decoded once
const R = (() => {
  const r = D.real, gd = r.gdp.d.map(m2d), hd = r.hicp.d.map(m2d), lh = r.hicp.y.map(Math.log);
  const infl = lh.slice(12).map((v, i) => 100 * (v - lh[i])), fxd = days(r.fx), bd = days(r.bet), sd = days(r.sp);
  return { gdp: { d: gd, y: r.gdp.y }, nom: { d: r.nom.d.map(m2d), y: r.nom.y }, infl: { d: hd.slice(12), y: infl },
    fx: { d: fxd, y: r.fx.y }, bet: { d: bd, p: r.bet.p }, sp: { d: sd, p: r.sp.p },
    nile: { yr: r.nile.y.map((_, i) => r.nile.y0 + i), y: r.nile.y }, sun: { yr: r.sun.y.map((_, i) => r.sun.y0 + i), y: r.sun.y } };
})();
const RET = {
  bet: { name: 'BET', d: R.bet.d.slice(1), r: d100(R.bet.p) },
  sp: { name: 'S&P 500', d: R.sp.d.slice(1), r: d100(R.sp.p) },
  fx: { name: 'EUR/RON', d: R.fx.d.slice(1), r: d100(R.fx.y.map(Math.log)) }
};

// ================================================================= four series
function blkFour() {
  const y0 = +$('fs-y0').value, len = +$('fs-len').value, y1 = Math.min(y0 + len, 2027), full = y0 <= 1995 && y1 >= 2027;
  set('fs-y0', y0);
  const G = grid('fs-chart', [{ r: 0, c: 0, t: 'Romania: real GDP, quarterly (bn EUR, 2010 prices)', x: { type: 'date' } }, { r: 0, c: 1, t: 'Romania: HICP inflation, 12-month (%)', x: { type: 'date' } },
    { r: 1, c: 0, t: 'EUR/RON, BNR reference rate (daily)', x: { type: 'date' } }, { r: 1, c: 1, t: 'BET: daily log returns (%)', x: { type: 'date' } }], { cols: 2, h: 460, legend: false });
  const S = [[R.gdp.d, R.gdp.y.map(v => v / 1000), K.green, 1.5, 'bn EUR'], [R.infl.d, R.infl.y, K.red, 1.5, '%'], [R.fx.d, R.fx.y, K.blue, 1.5, 'RON'], [RET.bet.d, RET.bet.r, K.amber, 0.5, '%']];
  const tr = [], res = [T.fs0(y0, y1 - 1, full)], rows = [];
  S.forEach(([d, y, c, w], j) => {
    const a = G.ax[j];
    tr.push({ x: d, y, mode: 'lines', line: { color: c, width: w }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: '%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>' });
    if (j === 1) G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.6));
    if (!full) G.lay.shapes.push({ type: 'rect', xref: a.xa, yref: a.yd, x0: `${y0}-01-01`, x1: `${y1}-01-01`, y0: 0, y1: 1, fillcolor: rgba(K.orange, 0.15), line: { width: 0 }, layer: 'below' });
    const ix = d.map((_, i) => i).filter(i => d[i] >= `${y0}` && d[i] < `${y1}`), v = ix.map(i => y[i]);
    res.push(v.length > 2 ? T.fs(j, nf(mean(v), 2), nf(sdev(v), 2), v.length) : T.fsNone(j));
    if (!full && v.length > 2) {          // the window's mean and mean ± one standard deviation, drawn over the window
      const xw = [d[ix[0]], d[ix[ix.length - 1]]], m = mean(v), sv = sdev(v);
      tr.push({ x: xw, y: [m, m], mode: 'lines', line: { color: K.navy, width: 2.2 }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: `window mean: ${m.toFixed(2)}<extra></extra>` });
      [m - sv, m + sv].forEach(u => tr.push({ x: xw, y: [u, u], mode: 'lines', line: { color: K.navy, width: 1.2, dash: 'dot' }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: `window mean ± sd: ${u.toFixed(2)}<extra></extra>` }));
    }
    if (v.length > 2) { const dg = j === 2 ? 4 : 2, dl = j < 2 ? 7 : 10, lo = ix[v.indexOf(Math.min(...v))], hi = ix[v.indexOf(Math.max(...v))];
      rows.push([T.x.fsN[j], txt(v.length), num(v[0], dg), num(v[v.length - 1], dg), num(mean(v), dg), num(sdev(v), dg), txt(`${nf(y[lo], dg)} (${d[lo].slice(0, dl)})`), txt(`${nf(y[hi], dg)} (${d[hi].slice(0, dl)})`)]); }
  });
  plot('fs-chart', tr, G.lay);
  out('fs-out', res);
  table('fs-tb', T.x.fsH, rows, { cap: T.fs0(y0, y1 - 1, full) });
}

// ================================================================= ensemble of paths
function blkEns() {
  const phi = +$('en-phi').value, t0 = +$('en-t').value; set('en-phi', nf(phi, 2)); set('en-t', t0);
  const E = D.ens, n = 100, sd = 1 / Math.sqrt(1 - phi * phi), t = lags(n);
  const A = E.e.map((e, i) => ar1(e, phi, sd * E.z0[i])), W = E.w.map(cumsum);
  const G = grid('en-chart', [{ r: 0, c: 0, t: `Stationary AR(1), φ = ${phi}: same band at every t`, xt: 't' }, { r: 0, c: 1, t: 'Random walk: the band widens like √t', xt: 't' },
    { r: 1, c: 0, t: `AR(1): the 40 values at t = ${t0}`, xt: `X<sub>${t0}</sub>` }, { r: 1, c: 1, t: `Random walk: the 40 values at t = ${t0}`, xt: `X<sub>${t0}</sub>` }], { cols: 2, h: 640 });
  const [a, b, ca, cb] = G.ax, tr = [], pa = joinPaths(A, t), pw = joinPaths(W, t);
  // cross-sections at t0: histogram of the 40 paths and the theoretical N(0, Var) density
  const xsA = A.map(x => x[t0 - 1]), xsW = W.map(x => x[t0 - 1]), vA = sd * sd, vW = t0;
  const dens = (v, L) => { const xs = range(-L, L, L / 60); return { x: xs, y: xs.map(z => Math.exp(-z * z / (2 * v)) / Math.sqrt(2 * Math.PI * v)) }; };
  const LA = Math.max(4, 3.5 * sd), LW = 25, dA = dens(vA, LA), dW = dens(vW, LW);
  tr.push({ x: xsA, type: 'histogram', histnorm: 'probability density', xbins: { start: -LA, end: LA, size: LA / 10 }, marker: { color: rgba(K.blue, 0.5) }, showlegend: false, xaxis: ca.xa, yaxis: ca.ya, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' });
  tr.push({ x: dA.x, y: dA.y, mode: 'lines', line: { color: K.navy, width: 1.6 }, name: 'N(0, Var X<sub>t</sub>)', xaxis: ca.xa, yaxis: ca.ya, hoverinfo: 'skip' });
  tr.push({ x: xsW, type: 'histogram', histnorm: 'probability density', xbins: { start: -LW, end: LW, size: 2.5 }, marker: { color: rgba(K.green, 0.5) }, showlegend: false, xaxis: cb.xa, yaxis: cb.ya, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' });
  tr.push({ x: dW.x, y: dW.y, mode: 'lines', line: { color: K.navy, width: 1.6 }, showlegend: false, xaxis: cb.xa, yaxis: cb.ya, hoverinfo: 'skip' });
  const axKey = s => s.replace(/^x/, 'xaxis').replace(/^y/, 'yaxis');
  G.lay[axKey(ca.xa)].range = [-LA, LA]; G.lay[axKey(cb.xa)].range = [-LW, LW];
  tr.push({ x: t.concat([...t].reverse()), y: t.map(() => 1.96 * sd).concat(t.map(() => -1.96 * sd)), fill: 'toself', fillcolor: rgba(K.orange, 0.18), line: { width: 0 }, mode: 'lines', name: '±1.96 sd(X<sub>t</sub>)', xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' });
  tr.push({ x: t.concat([...t].reverse()), y: t.map(v => 1.96 * Math.sqrt(v)).concat([...t].reverse().map(v => -1.96 * Math.sqrt(v))), fill: 'toself', fillcolor: rgba(K.orange, 0.18), line: { width: 0 }, mode: 'lines', showlegend: false, xaxis: b.xa, yaxis: b.ya, hoverinfo: 'skip' });
  tr.push({ x: pa.x, y: pa.y, mode: 'lines', line: { color: rgba(K.blue, 0.45), width: 0.7 }, name: 'one path', xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' });
  tr.push({ x: pw.x, y: pw.y, mode: 'lines', line: { color: rgba(K.green, 0.45), width: 0.7 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hoverinfo: 'skip' });
  // AR(1) band: ±1.96 sd, at most ±13.9 at φ = 0.99; the random walk keeps ±25
  // the AR(1) panel keeps one scale for every φ, so the band visibly widens as ▶ raises φ
  G.lay[axKey(a.ya)].range = [-15, 15]; G.lay[axKey(b.ya)].range = [-LW, LW];
  G.lay.shapes.push(vl(a, t0, K.navy, 'dot', 1), vl(b, t0, K.navy, 'dot', 1));
  // the 40 values at t0 on the paths: the points the histograms below are made of
  tr.push({ x: xsA.map(() => t0), y: xsA, mode: 'markers', marker: { color: K.navy, size: 5 }, name: `the 40 values at t = ${t0}`, xaxis: a.xa, yaxis: a.ya, hovertemplate: `t = ${t0}: %{y:.2f}<extra></extra>` },
    { x: xsW.map(() => t0), y: xsW, mode: 'markers', marker: { color: K.navy, size: 5 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hovertemplate: `t = ${t0}: %{y:.2f}<extra></extra>` });
  plot('en-chart', tr, G.lay);
  const va = pvar(A.map(x => x[t0 - 1])), vw = pvar(W.map(x => x[t0 - 1]));
  const res = [T.en1(nf(phi, 2), nf(sd * sd, 2), nf(1.96 * sd, 2), t0, nf(va, 2)), T.en2(t0, nf(1.96 * Math.sqrt(t0), 2), nf(vw, 1))];
  if (t0 === 100) res.push(T.en3(nf(D.numbers.ensemble.rw_var_sim, 1)));
  out('en-out', res);
}

// ================================================================= weakly but not strictly stationary
const ceCache = {};
function ceOdd(k) {
  if (!ceCache[k]) {
    if (k === 5) ceCache[k] = D.ce.chi.map(v => (v - 5) / Math.sqrt(10));
    else { const z = rng(7000 + k), s = Math.sqrt(2 * k); ceCache[k] = D.ce.ev.map(() => { let c = 0; for (let j = 0; j < k; j++) c += z() ** 2; return (c - k) / s; }); }
  }
  return ceCache[k];
}
function blkCounter() {
  const k = +$('ce-k').value; set('ce-k', k);
  const ev = D.ce.ev, od = ceOdd(k), x = []; ev.forEach((v, i) => x.push(v, od[i]));
  const G = grid('ce-chart', [{ r: 0, c: 0, t: 'First 100 observations', xt: 't' }, { r: 0, c: 1, t: 'Distribution of X<sub>t</sub> for even and odd t', x: { range: [-4, 6] } }], { cols: 2, h: 360 });
  const [a, b] = G.ax;
  const tr = [{ x: range(0, 99, 1), y: x.slice(0, 100), mode: 'lines', line: { color: K.blue, width: 1 }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: 't = %{x}: %{y:.2f}<extra></extra>' },
    { x: ev, type: 'histogram', histnorm: 'probability density', xbins: { start: -4, end: 6, size: 10 / 49 }, marker: { color: rgba(K.blue, 0.55) }, name: 'even t: N(0, 1)', xaxis: b.xa, yaxis: b.ya },
    { x: od, type: 'histogram', histnorm: 'probability density', xbins: { start: -4, end: 6, size: 10 / 49 }, marker: { color: rgba(K.red, 0.55) }, name: `odd t: (χ²<sub>${k}</sub> − ${k})/√${2 * k}`, xaxis: b.xa, yaxis: b.ya }];
  G.lay.barmode = 'overlay';
  plot('ce-chart', tr, G.lay);
  const p = (v, c) => v.filter(u => u < c).length / v.length;
  out('ce-out', [T.ce1(k, nf(mean(ev), 3), nf(mean(od), 3), nf(pvar(ev), 2), nf(pvar(od), 2)),
    T.ce2(nf(skew(ev), 2), nf(skew(od), 2), nf(Math.sqrt(8 / k), 2)),
    T.ce3(pct(p(ev, -1.6), 1), pct(p(od, -1.6), 1), nf(-Math.sqrt(k / 2), 2))]);
}

// ================================================================= which series are stationary?
function blkNonstat() {
  const b = +$('ns-b').value, s1 = +$('ns-s').value, sh = +$('ns-d').value, te = +$('ns-t').value; set('ns-b', nf(b, 3)); set('ns-s', nf(s1, 1)); set('ns-d', nf(sh, 1)); set('ns-t', te);
  const n = 300, t = range(0, n - 1, 1), N0 = D.ns;
  const X = [N0.a, N0.b.map((v, i) => b * i + v), N0.c.map((v, i) => v * (0.4 + (s1 - 0.4) * i / (n - 1))), N0.d.map((v, i) => v + (i >= n / 2 ? sh : 0))];
  const names = ['Series A', 'Series B', 'Series C', 'Series D'], cols = [K.blue, K.red, K.green, K.purple];
  const G = grid('ns-chart', names.map((nm, j) => ({ r: j >> 1, c: j & 1, t: nm })), { cols: 2, h: 440, legend: false });
  const tr = X.map((x, j) => ({ x: t, y: x, mode: 'lines', line: { color: cols[j], width: 1 }, xaxis: G.ax[j].xa, yaxis: G.ax[j].ya, showlegend: false, hovertemplate: 't = %{x}: %{y:.2f}<extra></extra>' }));
  // rolling window of W observations, drawn up to the window that ends at te: mean (solid) and mean ± 2 sd (dotted)
  const W = 50, tw = range(W - 1, te - 1, 1), last = [];
  X.forEach((x, j) => {
    const mu = [], lo = [], hi = [];
    tw.forEach(e => { const w = x.slice(e - W + 1, e + 1), m = mean(w), sv = sdev(w); mu.push(m); lo.push(m - 2 * sv); hi.push(m + 2 * sv); });
    last.push([mu[mu.length - 1], (hi[hi.length - 1] - lo[lo.length - 1]) / 4]);
    tr.push({ x: tw, y: mu, mode: 'lines', line: { color: K.navy, width: 2 }, xaxis: G.ax[j].xa, yaxis: G.ax[j].ya, showlegend: false, hovertemplate: `t = %{x}: rolling mean (${W} obs.) %{y:.2f}<extra></extra>` });
    [lo, hi].forEach(y => tr.push({ x: tw, y, mode: 'lines', line: { color: K.navy, width: 1, dash: 'dot' }, xaxis: G.ax[j].xa, yaxis: G.ax[j].ya, showlegend: false, hovertemplate: `t = %{x}: rolling mean ± 2 sd %{y:.2f}<extra></extra>` }));
  });
  plot('ns-chart', tr, G.lay);
  out('ns-out', X.map((x, j) => { const h1 = x.slice(0, n / 2), h2 = x.slice(n / 2); return T.ns(names[j].slice(-1), nf(mean(h1), 2), nf(mean(h2), 2), nf(sdev(h1), 2), nf(sdev(h2), 2)); })
    .concat([T.ns2(te - W, te - 1, last.map(([m]) => nf(m, 2)), last.map(([, sv]) => nf(sv, 2)))]));
  const cond = [null, b > 0 ? T.x.nsB(nf(b, 3)) : null, s1 > 0.4 + 1e-9 ? T.x.nsC(nf(s1 / 0.4, 1)) : null, sh ? T.x.nsD(nf(sh, 1)) : null];
  table('ns-tb', T.x.nsH, X.map((x, j) => { const h1 = x.slice(0, n / 2), h2 = x.slice(n / 2);
    return [names[j].slice(-1), num(mean(h1), 2), num(mean(h2), 2), num(sdev(h1), 2), num(sdev(h2), 2), num(sdev(h2) / sdev(h1), 2), cond[j] || T.x.nsOK]; }), { cap: T.x.nsCap });
}

// ================================================================= three kinds of white noise
function garch1(z, alpha, beta, n) {   // generate_all_charts.simulate_garch with omega = 1 - alpha - beta (variance 1)
  const om = 1 - alpha - beta, e = new Array(z.length).fill(0); let s2 = om / (1 - alpha - beta);
  for (let t = 1; t < z.length; t++) { s2 = om + alpha * e[t - 1] ** 2 + beta * s2; e[t] = Math.sqrt(s2) * z[t]; }
  return e.slice(z.length - n);
}
function blkWN() {
  const al = +$('wn-a').value, be = 0.85, n = +$('wn-n').value; set('wn-a', nf(al, 2)); set('wn-n', n);
  // the first n of the 1000 observations of the lecture
  const S = [['Gaussian i.i.d.', D.wn.ga.slice(0, n), K.blue], ['Laplace i.i.d.', D.wn.la.slice(0, n), K.green],
    [Math.abs(al - 0.1) < 1e-9 ? 'GARCH(1,1): weak WN' : `GARCH(1,1), α = ${al}`, garch1(D.wn.z, al, be, 1000).slice(0, n), K.red]];
  const A2 = S.map(([, x]) => acf(x.map(v => v * v), 20)), top = Math.max(0.35, ...A2.flat()) + 0.03;
  const cells = S.map(([nm], j) => ({ r: 0, c: j, t: nm + (n < 1000 ? `, first ${n} obs.` : ''), y: { range: [-7, 7] }, x: { range: [0, 1000] } })).concat(S.map((_, j) => ({ r: 1, c: j, t: 'ACF of X<sub>t</sub><sup>2</sup>', y: { range: [-0.12, top] } })));
  const G = grid('wn-chart', cells, { cols: 3, h: 440 }), tr = [], res = [], rows = [];
  S.forEach(([nm, x, c], j) => {
    tr.push({ x: range(0, n - 1, 1), y: x, mode: 'lines', line: { color: c, width: 0.6 }, xaxis: G.ax[j].xa, yaxis: G.ax[j].ya, showlegend: false, hovertemplate: 't = %{x}: %{y:.2f}<extra></extra>' });
    const x2 = x.map(v => v * v);
    acfPanel(tr, G.lay, G.ax[3 + j], A2[j], n, c);
    const l1 = lbq(x, 10), l2 = lbq(x2, 10);
    res.push(T.wn(j, nf(al, 2), pe(l1.p), nf(l2.q, 1), pe(l2.p), nf(kurt(x), 2)));
    rows.push([nm, num(kurt(x), 2), num(l1.q, 2), txt(pv(l1.p)), l1.p < 0.05 ? T.x.rej : T.x.norej, num(l2.q, 1), txt(pv(l2.p)), l2.p < 0.05 ? T.x.rej : T.x.norej]);
  });
  table('wn-tb', T.x.wnH, rows, { cap: T.x.wnCap(n) });
  res.unshift(T.wn0(n, nf(1.96 / Math.sqrt(n), 3)));
  tr.push(leg('ACF of squares', null, { color: K.blue, symbol: 'square', size: 10 }), BAND);
  plot('wn-chart', tr, G.lay);
  out('wn-out', res);
}

// ================================================================= random walks and their variance
function blkRW() {
  const t0 = +$('rw-t').value, c = +$('rw-c').value; set('rw-t', t0); set('rw-c', nf(c, 2));
  const n = 250, t = lags(n), P = D.rw.paths.map(p => p.map((v, i) => v + c * (i + 1)));
  const G = grid('rw-chart', [{ r: 0, c: 0, t: `${P.length} random walks, σ = 1` + (c ? `, drift c = ${c}` : ''), xt: 't' }, { r: 0, c: 1, t: 'Var(X<sub>t</sub>) grows linearly in t', xt: 't' }], { cols: 2, h: 380 });
  const [a, b] = G.ax, tr = [];
  [0, 1, 2].forEach(k => { const pp = joinPaths(P.filter((_, i) => i % 3 === k), t); tr.push({ x: pp.x, y: pp.y, mode: 'lines', line: { color: rgba(PAL[k], 0.5), width: 0.6 }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hoverinfo: 'skip' }); });
  tr.push({ x: t, y: t.map(v => c * v + 1.96 * Math.sqrt(v)), mode: 'lines', line: { color: K.red, dash: 'dash', width: 1.4 }, name: (c ? 'ct ' : '') + '±1.96√t', xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' },
    { x: t, y: t.map(v => c * v - 1.96 * Math.sqrt(v)), mode: 'lines', line: { color: K.red, dash: 'dash', width: 1.4 }, showlegend: false, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' });
  tr.push({ x: P.map(() => t0), y: P.map(p => p[t0 - 1]), mode: 'markers', marker: { color: K.navy, size: 5 }, name: `the ${P.length} values at t = ${t0}`, xaxis: a.xa, yaxis: a.ya, hovertemplate: `t = ${t0}: %{y:.2f}<extra></extra>` });
  tr.push({ x: t, y: D.rw.var, mode: 'lines', line: { color: K.blue, width: 1.8 }, name: `variance across ${D.rw.nsim} paths`, xaxis: b.xa, yaxis: b.ya, hovertemplate: 't = %{x}: %{y:.1f}<extra></extra>' },
    { x: t, y: t, mode: 'lines', line: { color: K.orange, dash: 'dash', width: 1.6 }, name: 'theory: tσ²', xaxis: b.xa, yaxis: b.ya, hoverinfo: 'skip' },
    { x: [t0], y: [D.rw.var[t0 - 1]], mode: 'markers', marker: { color: K.navy, size: 9 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hoverinfo: 'skip' });
  G.lay.shapes.push(vl(a, t0, K.navy, 'dot', 1), vl(b, t0, K.navy, 'dot', 1));
  plot('rw-chart', tr, G.lay);
  const res = [T.rw1(t0, nf(D.rw.var[t0 - 1], 1), t0, pct(D.rw.out[t0 - 1], 1))];
  if (t0 <= n - 10) res.push(T.rw2(t0, t0 + 10, nf(D.rw.c10[t0 - 1], 3), nf(Math.sqrt(t0 / (t0 + 10)), 3)));
  if (c) res.push(T.rw3(nf(c, 2), t0, nf(c * t0, 1)));
  out('rw-out', res);
  const s0 = Math.min(t0 + 10, n);
  steps('rw-ws', [T.rws1(t0, nf(D.rw.var[t0 - 1], 1)), t0 <= n - 10 ? T.rws2(t0, s0, nf(Math.sqrt(t0 / s0), 3), nf(D.rw.c10[t0 - 1], 3)) : T.rws2n(t0),
    T.rws3(t0, nf(1.96 * Math.sqrt(t0), 2), pct(D.rw.out[t0 - 1], 1)), c ? T.rws4(nf(c, 2), t0, nf(c * t0, 1)) : T.rws4n()]);
  const ts = [...new Set([1, 10, 50, 100, 150, 200, 240, 250, t0])].sort((x, y) => x - y);
  table('rw-tb', T.x.rwH, ts.map(u => hiRow([u, num(u, 0), num(D.rw.var[u - 1], 1), num(1.96 * Math.sqrt(u), 2), txt(pct(D.rw.out[u - 1], 1)),
    u <= n - 10 ? num(Math.sqrt(u / (u + 10)), 3) : '', u <= n - 10 ? num(D.rw.c10[u - 1], 3) : ''], u === t0)), { cap: T.x.rwCap(D.rw.nsim) });
}

// ================================================================= which one is real?
function blkSpot() {
  const n = +$('sr-n').value, view = $('sr-v').value, S = D.spot; set('sr-n', n);
  const Y = S.z.map((z, i) => { if (z === null) { const p = S.p.slice(S.p.length - n - 1); return p.map(v => v - p[0]); }
    return [0].concat(cumsum(z.slice(0, n).map(v => S.mu + S.sd * v))); });
  const L = 'ABCDEF', acfv = view === 'acf';
  const G = grid('sr-chart', L.split('').map((l, j) => ({ r: j / 3 | 0, c: j % 3, t: acfv ? `Series ${l}: ACF of daily changes` : `Series ${l}`, xt: acfv && j > 2 ? 'lag h' : '',
    x: acfv ? {} : { showticklabels: false }, y: acfv ? { range: [-0.25, 0.25] } : {} })), { cols: 3, h: 440, legend: acfv });
  const tr = [], res = [];
  Y.forEach((y, j) => {
    const dy = y.slice(1).map((v, i) => v - y[i]), a = G.ax[j];
    if (acfv) acfPanel(tr, G.lay, a, acf(dy, 10), n, K.blue);
    else tr.push({ x: range(0, n, 1), y: y.map(v => 100 * v), mode: 'lines', line: { color: K.blue, width: 0.9 }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: 'day %{x}: %{y:.1f}%<extra></extra>' });
    const l = lbq(dy, 10); res.push(T.sr(L[j], nf(acf(dy, 1)[0], 3), pe(l.p)));
  });
  if (acfv) tr.push(BAND);
  if (!acfv && !G.narrow) G.lay.annotations.push({ xref: 'paper', yref: 'paper', x: -0.055, y: 0.5, textangle: -90, showarrow: false, text: 'change in log price since day 0 (%)', font: { size: 11, color: K.navy } });
  plot('sr-chart', tr, G.lay);
  const d = S.d.slice(S.d.length - n - 1);
  res.push(T.sr0(L[S.pos], d[0], d[d.length - 1], n));
  out('sr-out', res);
}

// ================================================================= S&P 500: the level and its difference
function blkSP() {
  const y0 = +$('sp-y0').value, dd = +$('sp-d').value; set('sp-y0', y0);
  const i0 = R.sp.d.findIndex(d => d >= `${y0}`), d = R.sp.d.slice(i0), p = R.sp.p.slice(i0), r = d100(p), r2 = r.slice(1).map((v, i) => v - r[i]);
  const G = grid('sp-chart', [{ r: 0, c: 0, t: 'S&P 500: ln P<sub>t</sub>', x: { type: 'date' } },
    { r: 1, c: 0, t: dd === 1 ? 'S&P 500: r<sub>t</sub> = 100 Δ ln P<sub>t</sub> = 100 (1 − L) ln P<sub>t</sub> (%)' : 'S&P 500: 100 Δ² ln P<sub>t</sub> = 100 (1 − L)² ln P<sub>t</sub>', x: { type: 'date' } }], { cols: 1, h: 440, legend: false });
  const [a, b] = G.ax;
  plot('sp-chart', [{ x: d, y: p, mode: 'lines', line: { color: K.blue, width: 1.3 }, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>' },
    { x: d.slice(dd), y: dd === 1 ? r : r2, mode: 'lines', line: { color: K.red, width: 0.5 }, xaxis: b.xa, yaxis: b.ya, hovertemplate: '%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>' }], G.lay);
  const ap = acf(p, 50), res = [T.sp1(d[0], p.length, nf(ap[0], 3), nf(ap[49], 3)), T.sp2(r.length, nf(mean(r), 3), nf(sdev(r), 2), nf(acf(r, 1)[0], 3))];
  if (dd === 2) res.push(T.sp3(nf(acf(r2, 1)[0], 3), nf(sdev(r2), 2)));
  out('sp-out', res);
}

// ================================================================= Wold weights
function blkWold() {
  const a = +$('wd-a').value, b = +$('wd-b').value, th = +$('wd-th').value; set('wd-a', nf(a, 2)); set('wd-b', nf(b, 2)); set('wd-th', nf(th, 1));
  const J = 15, j = range(0, J, 1), f = v => +v.toFixed(2);
  const cases = [[`AR(1), φ = ${f(a)}: ψ<sub>j</sub> = ${a < 0 ? `(${f(a)})` : f(a)}<sup>j</sup>`, j.map(k => a ** k), K.blue, 'ar', a],
    [`AR(1), φ = ${f(b)}: ψ<sub>j</sub> = ${b < 0 ? `(${f(b)})` : f(b)}<sup>j</sup>`, j.map(k => b ** k), K.green, 'ar', b],
    [`MA(1), θ = ${f(th)}: ψ<sub>1</sub> = ${f(th)}, then 0`, j.map(k => (k === 0 ? 1 : k === 1 ? th : 0)), K.purple, 'ma', th],
    ['Random walk: ψ<sub>j</sub> = 1 (•: φ<sup>j</sup>)', j.map(() => 1), K.red, 'rw', 1]];
  const all = cases.flatMap(c => c[1]), lo = Math.min(-0.7, ...all) * 1.08, hi = Math.max(1.05, ...all) * 1.05;
  const G = grid('wd-chart', cases.map((c, k) => ({ r: 0, c: k, t: c[0], xt: 'j', y: { range: [lo, hi] } })), { cols: 4, h: 320, legend: false, gx: 0.05 });
  const tr = cases.map((c, k) => ({ x: j, y: c[1], type: 'bar', width: 0.6, marker: { color: c[2] }, xaxis: G.ax[k].xa, yaxis: G.ax[k].ya, showlegend: false, hovertemplate: 'ψ<sub>%{x}</sub> = %{y:.3f}<extra></extra>' }));
  // the random walk is the AR(1) at φ = 1: the weights of the first panel, for comparison
  tr.push({ x: j, y: cases[0][1], mode: 'markers', marker: { color: K.navy, size: 6 }, xaxis: G.ax[3].xa, yaxis: G.ax[3].ya, showlegend: false, hovertemplate: `φ = ${f(a)}: φ<sup>%{x}</sup> = %{y:.3f}<extra></extra>` });
  G.ax.forEach(x => G.lay.shapes.push(hl(x, 0, K.navy, 'solid', 0.6)));
  plot('wd-chart', tr, G.lay);
  out('wd-out', cases.slice(0, 3).map(([, ps, , kind, v]) => {
    const s2 = kind === 'ma' ? 1 + v * v : Math.abs(v) < 1 ? 1 / (1 - v * v) : null;
    return T.wd(kind, nf(v, 2), nf(ps[5], 3), nf(ps[10], 3), s2 === null ? null : nf(s2, 2));
  }).concat([T.wdRW()]));
}

// ================================================================= time averages: ergodic and non-ergodic
function blkErgo() {
  const phi = +$('eg-phi').value, sz = +$('eg-z').value, Tn = +$('eg-T').value; set('eg-phi', nf(phi, 2)); set('eg-z', nf(sz, 1)); set('eg-T', Tn);
  const E = D.ergo, n = 2000, t = lags(Tn);
  const A = E.e.map(e => ar1(e, phi).slice(E.burn)), B = E.u.map((u, i) => u.map(v => sz * E.z[i] + v));
  const run = x => { let s = 0; return x.map((v, i) => (s += v) / (i + 1)); };
  const G = grid('eg-chart', [{ r: 0, c: 0, t: `AR(1), φ = ${phi}: X̄<sub>T</sub> → μ = 0 on every path`, xt: 'T (log scale)', yt: 'X̄<sub>T</sub>', x: { type: 'log' } },
    { r: 0, c: 1, t: `X<sub>t</sub> = Z + ε<sub>t</sub>` + (sz === 1 ? '' : `, sd(Z) = ${sz}`) + ': X̄<sub>T</sub> → Z, not μ = 0', xt: 'T (log scale)', x: { type: 'log' } }], { cols: 2, h: 360, legend: false });
  const [a, b] = G.ax, tr = [], ea = [], eb = [];
  // the running means are drawn up to T = Tn on a fixed scale (that of the full 2000 observations)
  let lim = 0;
  A.forEach((x, i) => { const m = run(x); lim = Math.max(lim, ...m.map(Math.abs)); ea.push(m[Tn - 1]); tr.push({ x: t, y: m.slice(0, Tn), mode: 'lines', line: { color: PAL[i % 8], width: 1 }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: 'T = %{x}: %{y:.3f}<extra></extra>' }); });
  B.forEach((x, i) => { const m = run(x); lim = Math.max(lim, ...m.map(Math.abs)); eb.push(m[Tn - 1]); tr.push({ x: t, y: m.slice(0, Tn), mode: 'lines', line: { color: PAL[i % 8], width: 1 }, xaxis: b.xa, yaxis: b.ya, showlegend: false, hovertemplate: 'T = %{x}: %{y:.3f}<extra></extra>' }); });
  [a, b].forEach(u => tr.push({ x: ea.map(() => Tn), y: u === a ? ea : eb, mode: 'markers', marker: { color: K.navy, size: 6 }, xaxis: u.xa, yaxis: u.ya, showlegend: false, hovertemplate: `X̄<sub>${Tn}</sub> = %{y:.3f}<extra></extra>` }));
  lim *= 1.05;
  G.lay.yaxis.range = [-lim, lim]; G.lay.yaxis2.range = [-lim, lim]; G.lay.xaxis.range = G.lay.xaxis2.range = [0, Math.log10(n)];
  G.lay.shapes.push(hl(a, 0, K.navy, 'dash', 0.8), hl(b, 0, K.navy, 'dash', 0.8));
  plot('eg-chart', tr, G.lay);
  const zs = E.z.map(z => sz * z);
  out('eg-out', [T.eg1(nf(phi, 2), Tn, nf(Math.min(...ea), 2), nf(Math.max(...ea), 2), nf(1 / ((1 - phi) * Math.sqrt(Tn)), 3)),
    T.eg2(nf(sz, 1), Tn, nf(Math.min(...eb), 2), nf(Math.max(...eb), 2), nf(Math.min(...zs), 2), nf(Math.max(...zs), 2))]);
}

// ================================================================= sampling distribution of the sample ACF
const btCache = {};
function btSim(Tn) {
  if (!btCache[Tn]) {
    if (Tn === D.bart.T) btCache[Tn] = { r1: D.bart.r1, mask: D.bart.mask, lecture: true };
    else { const z = rng(800 + Tn), b = 1.96 / Math.sqrt(Tn), r1 = [], mask = [];
      for (let s = 0; s < 2000; s++) { const x = Array.from({ length: Tn }, z), r = acf(x, 20); r1.push(r[0]); let m = 0; r.forEach((v, k) => { if (Math.abs(v) > b) m |= 1 << k; }); mask.push(m); }
      btCache[Tn] = { r1, mask, lecture: false }; }
  }
  return btCache[Tn];
}
function blkBart() {
  const m = +$('bt-m').value, Tn = +$('bt-T').value; set('bt-m', m);
  const S = btSim(Tn), b = 1.96 / Math.sqrt(Tn), mm = (1 << m) - 1;
  const cnt = S.mask.map(v => { let c = 0, w = v & mm; while (w) { c += w & 1; w >>= 1; } return c; });
  const G = grid('bt-chart', [{ r: 0, c: 0, t: 'White noise: ρ̂(1) ≈ N(0, 1/T)' }, { r: 0, c: 1, t: `Number of lags (of ${m}) outside the bands`, xt: 'lags outside', x: { dtick: 1 } }], { cols: 2, h: 370, legendRows: 2 });
  const [a, c] = G.ax, gx = range(-0.45, 0.45, 0.005), k = range(0, 7, 1), nsim = S.r1.length;
  const emp = k.map(v => cnt.filter(u => u === v).length / nsim);
  // histogram of ρ̂(1) (density), split into the samples with no lag of the first m outside the band and those with at least one
  const lim = Math.max(0.45, 4.5 / Math.sqrt(Tn)), bw = 2 * lim / 60, mid = range(0, 59, 1).map(i => -lim + (i + 0.5) * bw), h0 = mid.map(() => 0), h1 = mid.map(() => 0);
  S.r1.forEach((v, i) => { const q = Math.floor((v + lim) / bw); if (q >= 0 && q < 60) (cnt[i] ? h1 : h0)[q] += 1 / (nsim * bw); });
  const tr = [{ x: mid, y: h0, type: 'bar', width: bw, marker: { color: rgba(K.blue, 0.6) }, name: `ρ̂(1), ${nsim} samples of T = ${Tn}: no lag of ${m} outside`, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x:.3f}: %{y:.2f}<extra></extra>' },
    { x: mid, y: h1, type: 'bar', width: bw, marker: { color: rgba(K.red, 0.55) }, name: `at least one lag of ${m} outside`, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x:.3f}: %{y:.2f}<extra></extra>' },
    { x: gx, y: gx.map(v => Math.sqrt(Tn / (2 * Math.PI)) * Math.exp(-Tn * v * v / 2)), mode: 'lines', line: { color: K.orange, width: 2 }, name: 'N(0, 1/T)', xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' },
    leg('±1.96/√T', { color: K.red, dash: 'dash', width: 1.2 }),
    { x: k.map(v => v - 0.18), y: emp, type: 'bar', width: 0.36, marker: { color: K.blue }, name: 'simulated share', xaxis: c.xa, yaxis: c.ya, customdata: k, hovertemplate: '%{customdata} outside: %{y:.3f}<extra></extra>' },
    { x: k.map(v => v + 0.18), y: k.map(v => binom(m, v, 0.05)), type: 'bar', width: 0.36, marker: { color: K.orange }, name: `Binomial(${m}, 0.05)`, xaxis: c.xa, yaxis: c.ya, customdata: k, hovertemplate: '%{customdata} outside: %{y:.3f}<extra></extra>' }];
  G.lay.shapes.push(vl(a, b, K.red, 'dash', 1), vl(a, -b, K.red, 'dash', 1));
  G.lay.xaxis.range = [-lim, lim]; G.lay.barmode = 'stack';
  plot('bt-chart', tr, G.lay);
  const sd = Math.sqrt(pvar(S.r1));
  out('bt-out', [T.bt1(Tn, nsim, S.lecture, nf(mean(S.r1), 3), nf(-1 / Tn, 3), nf(sd, 3), nf(1 / Math.sqrt(Tn), 3), nf(b, 3)),
    T.bt2(m, pct(1 - emp[0], 1), pct(1 - 0.95 ** m, 1), nf(mean(cnt), 2), nf(0.05 * m, 2))]);
}

// ================================================================= theoretical and sample ACF of four processes
function blkACFm() {
  const phi = +$('am-phi').value, th = +$('am-th').value, Tn = +$('am-T').value; set('am-phi', nf(phi, 2)); set('am-th', nf(th, 2)); set('am-T', Tn);
  const A = D.acfm, ma = A.ma.map((v, i) => v + (i ? th * A.ma[i - 1] : 0));
  const X = [A.wn.slice(0, Tn), ar1(A.ar, phi).slice(500, 500 + Tn), ma.slice(500, 500 + Tn), cumsum(A.rw.slice(0, Tn))];
  const names = ['White noise', `AR(1), φ = ${phi}`, `MA(1), θ = ${th}`, 'Random walk'];
  const theo = [new Array(20).fill(0), lags(20).map(h => phi ** h), lags(20).map(h => (h === 1 ? th / (1 + th * th) : 0)), null];
  const G = grid('am-chart', names.map((nm, j) => ({ r: 0, c: j, t: nm, xt: 'lag h', y: { range: [-0.4, 1.05] } })), { cols: 4, h: 330, gx: 0.05 });
  const tr = [], res = [];
  X.forEach((x, j) => { const r = acf(x, 20); acfPanel(tr, G.lay, G.ax[j], r, Tn, K.blue, theo[j]);
    res.push(T.am(j, [null, nf(phi, 2), nf(th, 2), null][j], nf(r[0], 2), nf(r[1], 2), nf(r[9], 2), theo[j] ? [nf(theo[j][0], 2), nf(theo[j][1], 2), nf(theo[j][9], 2)] : null)); });
  tr.push(leg('sample ACF', null, { color: K.blue, symbol: 'square', size: 10 }), leg('theoretical ACF', null, { color: K.orange, size: 7 }), BAND);
  plot('am-chart', tr, G.lay);
  res.push(T.amB(Tn, nf(1.96 / Math.sqrt(Tn), 3)));
  out('am-out', res);
}

// ================================================================= ACF and PACF of three processes
function blkACFP() {
  const a = +$('ap-a').value, f2 = +$('ap-b').value, th = +$('ap-c').value; set('ap-a', nf(a, 2)); set('ap-b', nf(f2, 2)); set('ap-c', nf(th, 2));
  const [e1, e2, e3] = D.ap, n = 500;
  const x2 = new Array(e2.length); for (let t = 0; t < e2.length; t++) x2[t] = e2[t] + (t >= 1 ? 0.5 * x2[t - 1] : 0) + (t >= 2 ? f2 * x2[t - 2] : 0);
  const X = [ar1(e1, a).slice(500), x2.slice(500), e3.map((v, i) => v + (i ? th * e3[i - 1] : 0)).slice(500)];
  const names = [`AR(1), φ = ${a}`, `AR(2), φ<sub>1</sub> = 0.5, φ<sub>2</sub> = ${f2}`, `MA(1), θ = ${th}`];
  const cells = names.map((nm, j) => ({ r: 0, c: j, t: nm + ': ACF', y: { range: [-0.45, 1] } })).concat(names.map((_, j) => ({ r: 1, c: j, t: 'PACF', xt: 'lag h', y: { range: [-0.45, 1] } })));
  const G = grid('ap-chart', cells, { cols: 3, h: 440 }), tr = [], res = [];
  X.forEach((x, j) => { const r = acf(x, 20), p = pacf(r); acfPanel(tr, G.lay, G.ax[j], r, n, K.blue); acfPanel(tr, G.lay, G.ax[3 + j], p, n, K.green);
    res.push(T.ap(j, [nf(a, 2), nf(f2, 2), nf(th, 2)][j], nf(r[0], 2), nf(r[1], 2), nf(p[0], 2), nf(p[1], 2), nf(p[2], 2))); });
  tr.push(leg('sample ACF', null, { color: K.blue, symbol: 'square', size: 10 }), leg('sample PACF', null, { color: K.green, symbol: 'square', size: 10 }), BAND);
  plot('ap-chart', tr, G.lay);
  const r1 = 0.5 / (1 - f2);
  res.push(T.apT(nf(f2, 2), nf(r1, 3), nf(0.5 * r1 + f2, 3)));
  out('ap-out', res);
}

// ================================================================= ACF of returns, absolute and squared returns
function blkBetACF() {
  const s = RET[$('ba-s').value], y0 = +$('ba-y0').value; set('ba-y0', y0);
  const i0 = s.d.findIndex(d => d >= `${y0}`), r = s.r.slice(Math.max(0, i0)), n = r.length;
  const X = [['r<sub>t</sub>', r, K.blue], ['|r<sub>t</sub>|', r.map(Math.abs), K.green], ['r<sub>t</sub><sup>2</sup>', r.map(v => v * v), K.red]];
  const G = grid('ba-chart', X.map(([l], j) => ({ r: 0, c: j, t: `${s.name}: ACF of ${l}`, xt: 'lag h', y: { range: [-0.1, 0.45] } })), { cols: 3, h: 340 });
  const tr = [], A = [];
  X.forEach(([, x, c], j) => { const a = acf(x, 50); A.push(a); acfPanel(tr, G.lay, G.ax[j], a, n, c); });
  tr.push(BAND);
  plot('ba-chart', tr, G.lay);
  const l1 = lbq(r, 10), l2 = lbq(X[2][1], 10);
  table('ba-tb', [T.x.series, ...[1, 2, 5, 10, 20, 50].map(h => `ρ̂(${h})`), 'Q*(10)', 'p'], X.map(([lb, x], j) => { const q = j === 0 ? l1 : j === 2 ? l2 : lbq(x, 10);
    return [`${s.name}: ${lb}`, ...[1, 2, 5, 10, 20, 50].map(h => num(A[j][h - 1], 3, Math.abs(A[j][h - 1]) > 1.96 / Math.sqrt(n))), num(q.q, 1), txt(pv(q.p))]; }), { cap: T.x.baCap(n, nf(1.96 / Math.sqrt(n), 3)) });
  out('ba-out', [T.ba1(s.name, s.d[Math.max(0, i0)], n, nf(1.96 / Math.sqrt(n), 3)), T.ba2(nf(A[0][0], 3), nf(l1.q, 1), pe(l1.p)),
    T.ba3(nf(A[1][0], 2), nf(A[1][49], 2), nf(A[2][0], 2), nf(l2.q, 0))]);
}

// ================================================================= size of Box-Pierce and Ljung-Box
function blkLBsize() {
  const m = +$('ls-m').value, Tn = $('ls-T').value, al = $('ls-a').value; set('ls-m', m);
  // the (T, m) pairs of the slide, plus the current m for every T
  const Rr = D.lbs.rates, A = 100 * +al, cases = [50, 100, 500].flatMap(t => [...new Set([10, 20, m])].sort((x, y) => x - y).map(mm => [t, mm]));
  const G = grid('ls-chart', [{ r: 0, c: 0, t: `Gaussian white noise, ${D.lbs.nsim} samples each`, yt: 'rejections of a true H<sub>0</sub> (%)', xt: 'T, m', x: { type: 'category' } },
    { r: 0, c: 1, t: `T = ${Tn}: rejections for every m`, xt: 'number of lags m', x: { dtick: 2 } }], { cols: 2, h: 380 });
  const [a, b] = G.ax, labs = cases.map(([t, mm]) => `${t}, ${mm}`), ms = lags(20);
  const tr = [{ x: labs, y: cases.map(([t, mm]) => 100 * Rr[t][al].bp[mm - 1]), type: 'bar', name: 'Box-Pierce Q', marker: { color: K.blue }, offsetgroup: 0, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'T, m = %{x}: %{y:.1f}%<extra></extra>' },
    { x: labs, y: cases.map(([t, mm]) => 100 * Rr[t][al].lb[mm - 1]), type: 'bar', name: 'Ljung-Box Q*', marker: { color: K.orange }, offsetgroup: 1, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'T, m = %{x}: %{y:.1f}%<extra></extra>' },
    // right: the rejection rates for m = 1, …, current m drawn solid, the larger m faded
    ...[['bp', K.blue, 'circle', 'Box-Pierce'], ['lb', K.orange, 'square', 'Ljung-Box']].flatMap(([k, c, sy, nm]) => [
      { x: ms.slice(0, m), y: Rr[Tn][al][k].slice(0, m).map(v => 100 * v), mode: 'lines+markers', marker: { size: 6, color: c, symbol: sy }, line: { color: c, width: 1.8 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hovertemplate: nm + ', m = %{x}: %{y:.1f}%<extra></extra>' },
      { x: ms.slice(m - 1), y: Rr[Tn][al][k].slice(m - 1).map(v => 100 * v), mode: 'lines+markers', opacity: 0.3, marker: { size: 4, color: c, symbol: sy }, line: { color: c, width: 1.2 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hovertemplate: nm + ', m = %{x}: %{y:.1f}%<extra></extra>' }]),
    leg(`nominal ${A}%`, { color: K.red, dash: 'dash', width: 1.2 })];
  G.lay.barmode = 'group';
  const top = Math.max(2.2 * A, ...[...Object.values(Rr).flatMap(o => o[al].lb), ...Rr[Tn][al].bp].map(v => 100 * v)) * 1.1;
  G.lay.yaxis.range = [0, top]; G.lay.yaxis2.range = [0, top]; G.lay.xaxis.tickangle = G.narrow ? -30 : 0; G.lay.xaxis.tickfont = { size: 10, color: K.navy };
  G.lay.shapes.push(hl(a, A, K.red, 'dash'), hl(b, A, K.red, 'dash'), vl(b, m, K.navy, 'dot', 1));
  plot('ls-chart', tr, G.lay);
  out('ls-out', [T.ls(Tn, m, nf(A, 0), nf(100 * Rr[Tn][al].bp[m - 1], 1), nf(100 * Rr[Tn][al].lb[m - 1], 1), D.lbs.nsim)]);
}

// ================================================================= Ljung-Box statistics of real series
const LQ = (() => {
  const gq = R.gdp.y.map(Math.log), g1 = gq.slice(1).map((v, i) => 100 * (v - gq[i]));
  return [['S&P 500 returns', RET.sp.d, RET.sp.r, K.blue], ['BET returns', RET.bet.d, RET.bet.r, K.red], ['EUR/RON returns', RET.fx.d, RET.fx.r, K.green],
    ['S&P 500 squared returns', RET.sp.d, RET.sp.r.map(v => v * v), K.teal], ['BET squared returns', RET.bet.d, RET.bet.r.map(v => v * v), K.orange],
    ['GDP growth, q/q', R.gdp.d.slice(1), g1, K.purple]];
})();
function blkLBstats() {
  const m = +$('lq-m').value, y0 = +$('lq-y0').value; set('lq-m', m); set('lq-y0', y0);
  const ms = lags(30), cr = ms.map(k => chi2ppf(0.95, k));
  const G = grid('lq-chart', [{ r: 0, c: 0, xt: 'number of lags m', yt: 'Ljung-Box Q*(m) (log scale)', y: { type: 'log' } }], { cols: 1, h: 420, legendRows: 3 });
  const a = G.ax[0], tr = [], res = [];
  LQ.forEach(([nm, d, x, c], j) => {
    const i0 = d.findIndex(v => v >= `${y0}`), y = x.slice(Math.max(0, i0)), n = y.length;
    if (n < 40) { res.push(T.lqFew(j, n)); return; }
    const r = acf(y, 30);
    let q = 0; const Q = ms.map(k => (q += r[k - 1] ** 2 / (n - k)) * n * (n + 2));
    tr.push({ x: ms, y: Q, mode: 'lines', line: { color: c, width: 1.8 }, name: nm, xaxis: a.xa, yaxis: a.ya, hovertemplate: nm + ', m = %{x}: %{y:.1f}<extra></extra>' });
    res.push(T.lq(j, n, m, nf(Q[m - 1], 1), pe(chi2sf(Q[m - 1], m))));
  });
  tr.push({ x: ms, y: cr, mode: 'lines', line: { color: K.navy, dash: 'dash', width: 1.4 }, name: '5% critical value χ²<sub>0.95</sub>(m)', xaxis: a.xa, yaxis: a.ya, hovertemplate: 'm = %{x}: %{y:.2f}<extra></extra>' });
  G.lay.shapes.push(vl(a, m, K.navy, 'dot', 1));
  plot('lq-chart', tr, G.lay);
  out('lq-out', [T.lq0(y0, m, nf(cr[m - 1], 2))].concat(res));
}

// ================================================================= Romanian real GDP: log, quarterly and annual growth
function blkGDP() {
  const k = +$('gt-k').value; set('gt-k', k);
  const d = R.gdp.d, l = R.gdp.y.map(Math.log), d1 = l.slice(1).map((v, i) => 100 * (v - l[i])), dk = l.slice(k).map((v, i) => 100 * (v - l[i]));
  const S = [['ln Y<sub>t</sub>', d, l, K.blue], ['100 Δ ln Y<sub>t</sub> (q/q, %)', d.slice(1), d1, K.green],
    [k === 4 ? '100 Δ<sub>4</sub> ln Y<sub>t</sub> (y/y, %)' : `100 Δ<sub>${k}</sub> ln Y<sub>t</sub> (%)`, d.slice(k), dk, K.red]];
  // three panels: ln Y with ln Y lagged by k (Δ_k ln Y is the gap); Δ_k ln Y with the quarterly growth as reference; the ACF of the three series
  const dx = { type: 'date', dtick: 'M120', tickformat: '%Y' }, kl = k === 4 ? '100 Δ<sub>4</sub> ln Y<sub>t</sub> (y/y, %)' : `100 Δ<sub>${k}</sub> ln Y<sub>t</sub> (%)`;
  const G = grid('gt-chart', [{ r: 0, c: 0, t: `ln Y<sub>t</sub> and ln Y<sub>t−${k}</sub>`, x: dx }, { r: 0, c: 1, t: `${kl} and 100 Δ ln Y<sub>t</sub> (q/q, %)`, x: dx },
    { r: 1, c: 0, cs: 2, t: `ACF: bars 100 Δ<sub>${k}</sub> ln Y<sub>t</sub>; dots ln Y<sub>t</sub> and 100 Δ ln Y<sub>t</sub>`, xt: 'lag (quarters)', y: { range: [-0.6, 1] }, x: { dtick: 1 } }], { cols: 2, h: 470, legendRows: 2 });
  const [pa, pb, pc] = G.ax, tr = [], A = S.map(([, , x]) => acf(x, 16)), L16 = lags(16);
  tr.push({ x: d, y: l, mode: 'lines', line: { color: K.blue, width: 1.4 }, name: 'ln Y<sub>t</sub>', xaxis: pa.xa, yaxis: pa.ya, hovertemplate: '%{x|%Y-%m}: %{y:.3f}<extra></extra>' },
    { x: d.slice(k), y: l.slice(0, l.length - k), mode: 'lines', line: { color: K.navy, width: 1.1, dash: 'dash' }, name: `ln Y<sub>t−${k}</sub>`, xaxis: pa.xa, yaxis: pa.ya, hovertemplate: '%{x|%Y-%m}: %{y:.3f}<extra></extra>' },
    { x: d.slice(1), y: d1, mode: 'lines', line: { color: K.green, width: 0.9 }, name: '100 Δ ln Y<sub>t</sub> (q/q)', xaxis: pb.xa, yaxis: pb.ya, hovertemplate: 'q/q %{x|%Y-%m}: %{y:.2f}<extra></extra>' },
    { x: d.slice(k), y: dk, mode: 'lines', line: { color: K.red, width: 1.6 }, name: `100 Δ<sub>${k}</sub> ln Y<sub>t</sub>`, xaxis: pb.xa, yaxis: pb.ya, hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' });
  acfPanel(tr, G.lay, pc, A[2], dk.length, K.red);
  tr.push({ x: L16, y: A[0], mode: 'markers', marker: { color: K.blue, size: 7 }, name: 'ACF of ln Y<sub>t</sub>', xaxis: pc.xa, yaxis: pc.ya, hovertemplate: 'ln Y, lag %{x}: %{y:.3f}<extra></extra>' },
    { x: L16, y: A[1], mode: 'markers', marker: { color: K.green, size: 7, symbol: 'diamond' }, name: 'ACF of 100 Δ ln Y<sub>t</sub>', xaxis: pc.xa, yaxis: pc.ya, hovertemplate: 'q/q, lag %{x}: %{y:.3f}<extra></extra>' }, BAND);
  G.lay.shapes.push(hl(pb, 0, K.navy, 'solid', 0.6));
  plot('gt-chart', tr, G.lay);
  const res = [T.gt1(nf(A[0][0], 2), nf(A[0][7], 2)), T.gt2(nf(A[1][3], 2), nf(A[1][1], 2), nf(sdev(d1), 1)),
    T.gt3(k, nf(mean(dk), 2), nf(sdev(dk), 2), nf(A[2][0], 2), nf(A[2][3], 2), nf(dk[dk.length - 1], 1))];
  if (k === 4) res.push(T.gt4(nf(dk[d.slice(k).indexOf('2020-04-01')], 1)));
  table('gt-tb', [T.x.series, 'T', T.x.mean, T.x.sd, 'ρ̂(1)', 'ρ̂(2)', 'ρ̂(4)', 'ρ̂(8)', 'Q*(8)', 'p'], S.map(([lb, , x], j) => { const q = lbq(x, 8);
    return [lb.replace(/ \(.*\)$/, ''), txt(x.length), num(mean(x), j ? 2 : 3), num(sdev(x), j ? 2 : 3), num(A[j][0], 2), num(A[j][1], 2), num(A[j][3], 2), num(A[j][7], 2), num(q.q, 1), txt(pv(q.p))]; }), { cap: T.x.gtCap });
  out('gt-out', res);
}

// ================================================================= Box-Cox for Romanian nominal GDP
const boxcox = (y, l) => (Math.abs(l) < 1e-8 ? y.map(Math.log) : y.map(v => (v ** l - 1) / l));
function guerrero(y, l) {
  const nb = Math.floor(y.length / 4), z = y.slice(y.length - nb * 4), ratio = [];
  for (let i = 0; i < nb; i++) { const b = z.slice(4 * i, 4 * i + 4); ratio.push(sdev(b) / mean(b) ** (1 - l)); }
  return sdev(ratio) / mean(ratio);
}
let bcGrid = null;
function blkBoxCox() {
  const l = +$('bc-l').value, y = R.nom.y, d = R.nom.d; set('bc-l', nf(l, 2));
  if (!bcGrid) { const g = range(-1, 2, 0.01); bcGrid = { g, cv: g.map(v => guerrero(y, v)) }; }
  const i = bcGrid.cv.indexOf(Math.min(...bcGrid.cv)), lh = bcGrid.g[i], w = boxcox(y, l);
  // seasonal amplitude of each complete year (maximum minus minimum of its four quarters), relative to the mean of the first five years
  const ampY = z => { const yrs = {}; d.forEach((s, j) => { const k = s.slice(0, 4); (yrs[k] = yrs[k] || []).push(z[j]); });
    return Object.entries(yrs).filter(([, v]) => v.length === 4).map(([k, v]) => [+k, Math.max(...v) - Math.min(...v)]); };
  const Aw = ampY(w), Ay = ampY(y), amp = Aw.map(([, v]) => v), a0 = mean(amp.slice(0, 5)), a1 = mean(amp.slice(-5)), ay0 = mean(Ay.slice(0, 5).map(([, v]) => v));
  const G = grid('bc-chart', [{ r: 0, c: 0, t: 'Seasonal amplitude / first 5 years', xt: 'year' }, { r: 0, c: 1, t: `Box-Cox, λ = ${l.toFixed(2)}`, x: { type: 'date' } },
    { r: 0, c: 2, t: "Guerrero's criterion against λ", xt: 'λ' }], { cols: 3, h: 360 });
  const [a, b, c] = G.ax;
  plot('bc-chart', [{ x: Ay.map(([k]) => k), y: Ay.map(([, v]) => v / ay0), mode: 'lines+markers', marker: { size: 4 }, line: { color: K.navy, width: 1.2, dash: 'dash' }, name: 'nominal GDP Y<sub>t</sub> (λ = 1)', xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
    { x: Aw.map(([k]) => k), y: amp.map(v => v / a0), mode: 'lines+markers', marker: { size: 4 }, line: { color: K.green, width: 1.8 }, name: `Box-Cox, λ = ${l.toFixed(2)}`, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
    { x: d, y: w, mode: 'lines', line: { color: K.green, width: 1.3 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hovertemplate: '%{x|%Y-%m}: %{y:.4g}<extra></extra>' },
    { x: bcGrid.g, y: bcGrid.cv, mode: 'lines', line: { color: K.red, width: 1.6 }, showlegend: false, xaxis: c.xa, yaxis: c.ya, hovertemplate: 'λ = %{x:.2f}: %{y:.3f}<extra></extra>' },
    { x: [l], y: [guerrero(y, l)], mode: 'markers', marker: { color: K.orange, size: 10 }, showlegend: false, xaxis: c.xa, yaxis: c.ya, hoverinfo: 'skip' }], Object.assign(G.lay, { shapes: [vl(c, lh, K.navy, 'dash', 1), hl(a, 1, K.navy, 'solid', 0.6)] }));
  out('bc-out', [T.bc1(nf(l, 2), nf(guerrero(y, l), 3), nf(lh, 2), nf(bcGrid.cv[i], 3), nf(guerrero(y, 0), 3), nf(guerrero(y, 1), 2)), T.bc2(sig(a0), sig(a1), nf(a1 / a0, 2))]);
}

// ================================================================= over-differencing
function blkOverdiff() {
  const phi = +$('od-phi').value; set('od-phi', nf(phi, 2));
  const x = ar1(D.od, phi), dx = x.slice(1).map((v, i) => v - x[i]), n = 500;
  const rx = acf(x, 10), rd = acf(dx, 10), th = lags(10).map(h => (phi < 1 ? -(1 - phi) * phi ** (h - 1) / 2 : 0));
  const lt = phi === 0 ? 'White noise ε<sub>t</sub>' : phi >= 1 ? 'Random walk X<sub>t</sub>' : `AR(1), φ = ${phi}: X<sub>t</sub>`;
  const rt = phi === 0 ? 'Over-differenced: Δε<sub>t</sub> = ε<sub>t</sub> − ε<sub>t−1</sub>' : 'ΔX<sub>t</sub> = X<sub>t</sub> − X<sub>t−1</sub>';
  const hi = Math.max(0.3, ...rx) * 1.08;
  const G = grid('od-chart', [{ r: 0, c: 0, t: lt, xt: 'lag h', y: { range: [-0.65, hi] } }, { r: 0, c: 1, t: rt, xt: 'lag h', y: { range: [-0.65, hi] } }], { cols: 2, h: 330 });
  const tr = [];
  acfPanel(tr, G.lay, G.ax[0], rx, n, K.blue); acfPanel(tr, G.lay, G.ax[1], rd, n, K.red, th);
  tr.push(leg('sample ACF', null, { color: K.blue, symbol: 'square', size: 10 }), leg('theoretical ACF', null, { color: K.orange, size: 7 }), BAND);
  plot('od-chart', tr, G.lay);
  out('od-out', [T.od1(nf(phi, 2), nf(rx[0], 2), nf(pvar(x), 2)), T.od2(nf(rd[0], 2), nf(th[0], 2), nf(pvar(dx), 2))]);
}

// ================================================================= the Nile and the sunspots
function blkTextbook() {
  const B = +$('tb-y').value; set('tb-y', B);
  const N_ = R.nile, pre = N_.y.filter((_, i) => N_.yr[i] <= B), post = N_.y.filter((_, i) => N_.yr[i] > B), m0 = mean(pre), m1 = mean(post);
  const mu = N_.yr.map(v => (v <= B ? m0 : m1)), res = N_.y.map((v, i) => v - mu[i]);
  const rn = acf(N_.y, 20), rr = acf(res, 20), rs = acf(R.sun.y, 40), yd = v => `${v}-01-01`;
  // the same two means, before and after B, removed from the sunspots: a series without a break
  const S_ = R.sun, sm0 = mean(S_.y.filter((_, i) => S_.yr[i] <= B)), sm1 = mean(S_.y.filter((_, i) => S_.yr[i] > B)), smu = S_.yr.map(v => (v <= B ? sm0 : sm1)), rsr = acf(S_.y.map((v, i) => v - smu[i]), 40);
  const G = grid('tb-chart', [{ r: 0, c: 0, t: 'Nile at Aswan: yearly flow (10<sup>8</sup> m<sup>3</sup>)', x: { type: 'date' } }, { r: 0, c: 1, t: 'Yearly sunspot numbers', x: { type: 'date' } },
    { r: 1, c: 0, xt: 'lag (years)' }, { r: 1, c: 1, xt: 'lag (years)' }], { cols: 2, h: 470, legendRows: 2 });
  const [a, b, c, d] = G.ax, l20 = lags(20);
  const tr = [{ x: N_.yr.map(yd), y: N_.y, mode: 'lines', line: { color: K.blue, width: 1.4 }, showlegend: false, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x|%Y}: %{y}<extra></extra>' },
    { x: N_.yr.map(yd), y: mu, mode: 'lines', line: { color: K.red, dash: 'dash', width: 1.4, shape: 'hv' }, name: B === 1898 ? 'mean before and after 1898' : `mean up to ${B} and after`, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' },
    { x: l20.map(v => v - 0.18), y: rn, type: 'bar', width: 0.36, marker: { color: K.blue }, name: 'ACF of the flow', xaxis: c.xa, yaxis: c.ya, customdata: l20, hovertemplate: 'lag %{customdata}: %{y:.3f}<extra></extra>' },
    { x: l20.map(v => v + 0.18), y: rr, type: 'bar', width: 0.36, marker: { color: K.orange }, name: 'ACF after removing the two means', xaxis: c.xa, yaxis: c.ya, customdata: l20, hovertemplate: 'lag %{customdata}: %{y:.3f}<extra></extra>' },
    { x: R.sun.yr.map(yd), y: R.sun.y, mode: 'lines', line: { color: K.amber, width: 1.2 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hovertemplate: '%{x|%Y}: %{y}<extra></extra>' }];
  const bn = 1.96 / Math.sqrt(N_.y.length);
  G.lay.shapes.push(hl(c, bn, K.red, 'dash', 1), hl(c, -bn, K.red, 'dash', 1), hl(c, 0, K.navy, 'solid', 0.8));
  acfPanel(tr, G.lay, d, rs, R.sun.y.length, K.amber);
  tr.push({ x: S_.yr.map(yd), y: smu, mode: 'lines', line: { color: K.red, dash: 'dash', width: 1.4, shape: 'hv' }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hoverinfo: 'skip' },
    { x: lags(40), y: rsr, mode: 'markers', marker: { color: K.orange, size: 6 }, showlegend: false, xaxis: d.xa, yaxis: d.ya, hovertemplate: 'after removing the two means, lag %{x}: %{y:.3f}<extra></extra>' });
  plot('tb-chart', tr, G.lay);
  const q0 = lbq(N_.y, 10), q1 = lbq(res, 10);
  let pk = 6; for (let h = 6; h <= 15; h++) if (rs[h - 1] > rs[pk - 1]) pk = h;
  let mn = 1; for (let h = 1; h <= 10; h++) if (rs[h - 1] < rs[mn - 1]) mn = h;
  table('tb-tb', [T.x.series, 'T', 'ρ̂(1)', 'ρ̂(2)', 'ρ̂(5)', 'ρ̂(10)', 'Q*(10)', 'p'], [[T.x.tbN[0], R.nile.y], [T.x.tbN[1](B), res], [T.x.tbN[2], R.sun.y]].map(([lb, x], j) => {
    const a = j === 0 ? rn : j === 1 ? rr : rs, q = j === 0 ? q0 : j === 1 ? q1 : lbq(x, 10);
    return [lb, txt(x.length), num(a[0], 2), num(a[1], 2), num(a[4], 2), num(a[9], 2), num(q.q, 1), txt(pv(q.p))]; }));
  out('tb-out', [T.tb1(B, nf(m0, 0), nf(m1, 0)), T.tb2(nf(rn[0], 2), nf(q0.q, 1), pe(q0.p)), T.tb3(nf(rr[0], 2), nf(q1.q, 1), pe(q1.p)),
    T.tb4(nf(rs[mn - 1], 2), mn, nf(rs[pk - 1], 2), pk), T.tb5(B, nf(sm0, 1), nf(sm1, 1), nf(rs[0], 2), nf(rsr[0], 2), nf(rs[pk - 1], 2), nf(rsr[pk - 1], 2), pk)]);
}

// ================================================================= tables and worked examples
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}>${v.v}</${tag}>`; };
  el.innerHTML = (o && o.cap ? `<p class="c1cap">${o.cap}</p>` : '') + '<table><thead><tr>' + head.map(c => cell(c, 'th')).join('') + '</tr></thead><tbody>'
    + rows.map(r => `<tr${r.hi ? ' class="hi"' : ''}>` + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c1cap">${o.note}</p>` : '');
}
const num = (x, d, bold) => ({ v: bold ? `<b>${nf(x, d)}</b>` : nf(x, d), cls: 'r' });
const txt = (v, bold) => ({ v: bold ? `<b>${v}</b>` : v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const pn = (x, d) => (x < 0 ? `(${nf(x, d)})` : nf(x, d));                      // a factor in a product: negative numbers in parentheses
const sg = (x, d) => (x < 0 ? ' − ' + nf(-x, d) : ' + ' + nf(x, d));           // a term of a sum
const pv = p => (p < 0.001 ? '&lt; ' + nf(0.001, 3) : nf(p, 3));
const hiRow = (r, on) => { if (on) r.hi = true; return r; };
const qlab = d => T.q(+d.slice(0, 4), Math.floor((+d.slice(5, 7) - 1) / 3) + 1);

const LBK = ['S&P 500 returns', 'S&P 500 squared returns', 'BET returns', 'BET squared returns', 'EUR/RON returns', 'GDP growth, q/q', 'GDP growth, y/y', 'HICP inflation, m/m', 'Nile flow', 'Sunspots'];
// the ten series of the Ljung-Box table (slide „Interpreting the Ljung-Box statistics”), in the order of the slide
const S10 = (() => {
  const lg = R.gdp.y.map(Math.log), lh = D.real.hicp.y.map(Math.log), hd = D.real.hicp.d.map(m2d);
  const hm = [], hmd = []; for (let i = 1; i < lh.length; i++) if (hd[i] >= '2005') { hm.push(100 * (lh[i] - lh[i - 1])); hmd.push(hd[i]); }
  return [['sp', RET.sp.r, RET.sp.d], ['sp2', RET.sp.r.map(v => v * v), RET.sp.d], ['bet', RET.bet.r, RET.bet.d], ['bet2', RET.bet.r.map(v => v * v), RET.bet.d],
    ['fx', RET.fx.r, RET.fx.d], ['gq', lg.slice(1).map((v, i) => 100 * (v - lg[i])), R.gdp.d.slice(1)], ['gy', lg.slice(4).map((v, i) => 100 * (v - lg[i])), R.gdp.d.slice(4)],
    ['hm', hm, hmd], ['nile', R.nile.y, R.nile.yr.map(String)], ['sun', R.sun.y, R.sun.yr.map(String)]].map(([k, x, d], j) => {
    const lr = D.lbr[LBK[j]];                          // the sample ACF of the chapter code (lags 1..30), from the unrounded data
    return { k, x, d, name: T.x.s10[k], r: lr && lr.n === x.length ? lr.r : acf(x, 30) }; });
})();
const s10 = k => S10.find(s => s.k === k);
function lbPath(x, m, r0) {        // Box-Pierce Q(h) and Ljung-Box Q*(h), h = 1..m, with p-values
  const n = x.length, r = r0 ? r0.slice(0, m) : acf(x, m); let a = 0, b = 0;
  return r.map((v, i) => { const h = i + 1; a += v * v; b += v * v / (n - h); const q = n * a, qs = n * (n + 2) * b; return { h, r: v, q, qs, pq: chi2sf(q, h), pqs: chi2sf(qs, h), cr: chi2ppf(0.95, h) }; });
}

// ---------------------------------------------------------------- properties of the autocovariance function
function blkAcov() {
  const r1 = +$('cp-r1').value, r2 = +$('cp-r2').value, a2 = +$('cp-a2').value, a3 = +$('cp-a3').value;
  set('cp-r1', nf(r1, 2)); set('cp-r2', nf(r2, 2)); set('cp-a2', nf(a2, a2 % 1 ? 1 : 0)); set('cp-a3', nf(a3, a3 % 1 ? 1 : 0));
  const t0 = 1 + a2 * a2 + a3 * a3, t1 = 2 * (a2 + a2 * a3) * r1, t2 = 2 * a3 * r2, v = t0 + t1 + t2;
  const lmin = Math.min(1 - r2, (2 + r2 - Math.sqrt(r2 * r2 + 8 * r1 * r1)) / 2), bnd = 2 * r1 * r1 - 1, ok = r2 >= bnd - 1e-12;
  const fa = v => nf(v, v % 1 ? 1 : 0);
  steps('cp-ws', [T.cps1(fa(a2), fa(a3), nf(r1, 2), nf(r2, 2), a2 < 0 ? `(${fa(a2)})` : fa(a2), a3 < 0 ? `(${fa(a3)})` : fa(a3), pn(r2, 2)),
    T.cps2(nf(t0, 2) + sg(t1, 2) + sg(t2, 2), nf(v, 2)),
    v < -1e-12 ? T.cps3n(nf(v, 2)) : T.cps3p(nf(v, 2)),
    T.cps4(nf(lmin, 2), nf(bnd, 2), ok)]);
  const G = grid('cp-chart', [{ r: 0, c: 0, t: 'Admissible pairs (ρ(1), ρ(2)) of a stationary process', xt: 'ρ(1)', yt: 'ρ(2)', x: { range: [-1.05, 1.05] }, y: { range: [-1.05, 1.05] } }], { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0], xs = range(-1, 1, 0.02), c1 = 2 * (a2 + a2 * a3);
  // the weights a = (1, a₂, a₃) rule out the half-plane where Var(Σ a_i X_i) < 0: its border is t0 + c1 ρ(1) + 2a₃ ρ(2) = 0
  const zl = Math.abs(a3) > 1e-12 ? { x: [-1.05, 1.05], y: [-1.05, 1.05].map(r => -(t0 + c1 * r) / (2 * a3)) } : Math.abs(c1) > 1e-12 ? { x: [-t0 / c1, -t0 / c1], y: [-1.05, 1.05] } : { x: [null], y: [null] };
  plot('cp-chart', [{ x: [...xs, ...[...xs].reverse()], y: [...xs.map(x => 2 * x * x - 1), ...xs.map(() => 1)], fill: 'toself', fillcolor: rgba(K.green, 0.15), line: { color: K.green, width: 1.2 }, mode: 'lines', name: 'ρ(2) ≥ 2ρ(1)² − 1', xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' },
    { x: xs, y: xs.map(x => x * x), mode: 'lines', line: { color: K.blue, dash: 'dot', width: 1.4 }, name: 'AR(1): ρ(2) = ρ(1)²', xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' },
    { x: zl.x, y: zl.y, mode: 'lines', line: { color: K.red, dash: 'dash', width: 1.4 }, name: `Var(X<sub>1</sub>${sg(a2, a2 % 1 ? 1 : 0)}X<sub>2</sub>${sg(a3, a3 % 1 ? 1 : 0)}X<sub>3</sub>) = 0`, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' },
    { x: [r1], y: [r2], mode: 'markers', marker: { color: ok ? K.green : K.red, size: 12, symbol: ok ? 'circle' : 'x' }, name: ok ? 'admissible' : 'not an ACF', xaxis: a.xa, yaxis: a.ya, hovertemplate: 'ρ(1) = %{x:.2f}, ρ(2) = %{y:.2f}<extra></extra>' }], G.lay);
  out('cp-out', [T.cp(nf(r1, 2), nf(r2, 2), nf(v, 2), ok)]);
}

// ---------------------------------------------------------------- worked example: MA(1)
function blkMA1() {
  const th = +$('xm-th').value, s2 = +$('xm-s2').value; set('xm-th', nf(th, 2)); set('xm-s2', nf(s2, 2));
  const g0 = s2 * (1 + th * th), g1 = th * s2, r1 = g1 / g0;
  steps('xm-ws', [T.xms1(nf(s2, 2), pn(th, 2), nf(g0, 3)), T.xms2(pn(th, 2), nf(s2, 2), nf(g1, 3)), T.xms3(nf(g1, 3), nf(g0, 3), nf(r1, 3)), T.xms4(nf(Math.abs(r1), 3))]);
  table('xm-tb', ['h', 'γ(h)', 'ρ(h)'], [0, 1, 2, 3].map(h => [h, num(h === 0 ? g0 : h === 1 ? g1 : 0, 4), num(h === 0 ? 1 : h === 1 ? r1 : 0, 3)]));
  const G = grid('xm-chart', [{ r: 0, c: 0, t: 'ρ(1) = θ/(1 + θ²) against θ', xt: 'θ', y: { range: [-0.6, 0.6] } }, { r: 0, c: 1, t: `ACF of the MA(1), θ = ${th.toFixed(2)}`, xt: 'lag h', y: { range: [-1, 1.05] } }], { cols: 2, h: 330, legend: false });
  const [a, b] = G.ax, xs = range(-2, 2, 0.02);
  G.lay.shapes.push(hl(a, 0.5, K.navy, 'dot', 1), hl(a, -0.5, K.navy, 'dot', 1), hl(b, 0, K.navy, 'solid', 0.8));
  plot('xm-chart', [{ x: xs, y: xs.map(x => x / (1 + x * x)), mode: 'lines', line: { color: K.blue, width: 1.8 }, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'θ = %{x:.2f}: %{y:.3f}<extra></extra>' },
    { x: [th], y: [r1], mode: 'markers', marker: { color: K.orange, size: 11 }, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' },
    { x: [0, 1, 2, 3, 4, 5], y: [1, r1, 0, 0, 0, 0], type: 'bar', width: 0.5, marker: { color: K.green }, xaxis: b.xa, yaxis: b.ya, hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' }], G.lay);
  out('xm-out', [T.xm(nf(th, 2), nf(g0, 2), nf(r1, 3))]);
}

// ---------------------------------------------------------------- worked example: AR(1) and its Wold weights
function blkAR1() {
  const ph = +$('xa-phi').value, s2 = +$('xa-s2').value, h = +$('xa-h').value; set('xa-phi', nf(ph, 2)); set('xa-s2', nf(s2, 2)); set('xa-h', h);
  const g0 = s2 / (1 - ph * ph), S = 1 / (1 - ph * ph), J = 10, cum = [];
  let c = 0; for (let j = 0; j <= 20; j++) { c += ph ** (2 * j); cum.push(c); }
  steps('xa-ws', [T.xas1(nf(s2, 2), pn(ph, 2), nf(1 - ph * ph, 4), nf(g0, 2)),
    T.xas2(pn(ph, 2), h, nf(ph, 3), nf(ph ** h, 3), nf(ph ** 10, 3), nf(g0 * ph ** h, 3)),
    T.xas3(pn(ph, 2), nf(S, 2), J, nf(cum[J], 2), nf(100 * cum[J] / S, 1)),
    ph > 0 ? T.xas4(nf(Math.log(0.5) / Math.log(ph), 1)) : T.xas4n()]);
  table('xa-tb', ['j', 'ψ<sub>j</sub> = ρ(j)', 'ψ<sub>j</sub><sup>2</sup>', 'Σ<sub>i≤j</sub> ψ<sub>i</sub><sup>2</sup>', T.x.share, 'γ(j)'],
    range(0, J, 1).map(j => hiRow([j, num(ph ** j, 4), num(ph ** (2 * j), 4), num(cum[j], 4), txt(nf(100 * cum[j] / S, 1) + '%'), num(g0 * ph ** j, 4)], j === h)));
  const G = grid('xa-chart', [{ r: 0, c: 0, t: `Wold weights ψ<sub>j</sub> = φ<sup>j</sup>, φ = ${ph.toFixed(2)}`, xt: 'j', y: { range: [-1, 1.05] } },
    { r: 0, c: 1, t: 'Cumulative σ²Σψ<sub>j</sub><sup>2</sup> and its limit γ(0) = σ²/(1 − φ²)', xt: 'j' }], { cols: 2, h: 340 });
  const [a, b] = G.ax, js = range(0, 20, 1);
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.8), hl(b, g0, K.red, 'dash', 1.2), vl(a, h, K.navy, 'dot', 1));
  plot('xa-chart', [{ x: js, y: js.map(j => ph ** j), type: 'bar', width: 0.55, marker: { color: K.blue }, name: 'ψ<sub>j</sub>', xaxis: a.xa, yaxis: a.ya, hovertemplate: 'j = %{x}: %{y:.3f}<extra></extra>' },
    { x: js, y: cum.map(v => s2 * v), mode: 'lines+markers', line: { color: K.green, width: 1.6 }, marker: { size: 5 }, name: 'σ²Σ<sub>i≤j</sub> ψ<sub>i</sub><sup>2</sup>', xaxis: b.xa, yaxis: b.ya, hovertemplate: 'j = %{x}: %{y:.3f}<extra></extra>' },
    leg('γ(0) = σ²/(1 − φ²)', { color: K.red, dash: 'dash', width: 1.2 })], G.lay);
  out('xa-out', [T.xa(nf(ph, 2), nf(g0, 2), h, nf(ph ** h, 3))]);
}

// ---------------------------------------------------------------- worked example: differencing
const XD = [10, 12, 15, 14, 18];
function blkDiff() {
  const src = $('xd-s').value, gd = src === 'gdp'; $('xd-e').disabled = !gd; $('xd-e').closest('.ctl').style.display = gd ? '' : 'none';
  const e = +$('xd-e').value, Y = R.gdp.y, d = R.gdp.d; set('xd-e', gd ? qlab(d[e]) : '—');
  if (!gd) {
    const d1 = XD.slice(1).map((v, i) => v - XD[i]), d2 = d1.slice(1).map((v, i) => v - d1[i]);
    steps('xd-ws', [T.xds1(XD.slice(1).map((v, i) => `${v} − ${XD[i]} = ${nf(d1[i], 0)}`).join(', ')),
      T.xds2(d1.slice(1).map((v, i) => `${nf(v, 0)} − ${pn(d1[i], 0)} = ${nf(d2[i], 0)}`).join(', '), `${XD[2]} − 2 × ${XD[1]} + ${XD[0]} = ${nf(d2[0], 0)}`),
      T.xds3(XD.length, d1.length, d2.length)]);
    table('xd-tb', ['t', 'x<sub>t</sub>', 'Δx<sub>t</sub>', 'Δ<sup>2</sup>x<sub>t</sub>'], XD.map((v, t) => [t + 1, num(v, 0), t >= 1 ? num(d1[t - 1], 0) : '', t >= 2 ? num(d2[t - 2], 0) : '']));
    const G = grid('xd-chart', [{ r: 0, c: 0, t: 'x<sub>t</sub>, Δx<sub>t</sub> and Δ<sup>2</sup>x<sub>t</sub>', xt: 't', x: { dtick: 1 } }], { cols: 1, h: 300 }), a = G.ax[0], ts = [1, 2, 3, 4, 5];
    G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.8));
    plot('xd-chart', [{ x: ts, y: XD, mode: 'lines+markers', line: { color: K.blue, width: 1.8 }, name: 'x<sub>t</sub>', xaxis: a.xa, yaxis: a.ya },
      { x: ts.slice(1), y: d1, mode: 'lines+markers', line: { color: K.green, width: 1.8 }, name: 'Δx<sub>t</sub>', xaxis: a.xa, yaxis: a.ya },
      { x: ts.slice(2), y: d2, mode: 'lines+markers', line: { color: K.red, width: 1.8 }, name: 'Δ<sup>2</sup>x<sub>t</sub>', xaxis: a.xa, yaxis: a.ya }], G.lay);
    out('xd-out', [T.xd0(d1.map(v => nf(v, 0)).join('; '), d2.map(v => nf(v, 0)).join('; '))]);
    return;
  }
  const l = Y.map(Math.log), g1 = i => 100 * (l[i] - l[i - 1]), g4 = i => 100 * (l[i] - l[i - 4]);
  steps('xd-ws', [T.xdg1(qlab(d[e]), nf(Y[e], 1), nf(Y[e - 1], 1), nf(Y[e] - Y[e - 1], 1)),
    T.xdg2(nf(l[e], 4), nf(l[e - 1], 4), nf(g1(e), 2), nf(100 * (Y[e] / Y[e - 1] - 1), 2)),
    T.xdg3(qlab(d[e - 4]), nf(Y[e - 4], 1), nf(l[e - 4], 4), nf(g4(e), 2)),
    T.xdg4(nf(g1(e), 2), pn(g1(e - 1), 2), nf(g1(e) - g1(e - 1), 2))]);
  table('xd-tb', [T.x.quarter, 'Y<sub>t</sub>', 'ln Y<sub>t</sub>', 'ΔY<sub>t</sub>', '100 Δ ln Y<sub>t</sub>', '100 (Y<sub>t</sub>/Y<sub>t−1</sub> − 1)', '100 Δ<sub>4</sub> ln Y<sub>t</sub>'],
    range(e - 7, e, 1).map(i => hiRow([qlab(d[i]), num(Y[i], 1), num(l[i], 4), num(Y[i] - Y[i - 1], 1), num(g1(i), 2), num(100 * (Y[i] / Y[i - 1] - 1), 2), num(g4(i), 2)], i === e)), { cap: T.x.xdCap });
  const G = grid('xd-chart', [{ r: 0, c: 0, t: 'Romania: real GDP (bn EUR, 2010 prices)', x: { type: 'date' } }, { r: 0, c: 1, t: 'Growth rates (%)', x: { type: 'date' } }], { cols: 2, h: 330 });
  const [a, b] = G.ax, i1 = range(1, Y.length - 1, 1), i4 = range(4, Y.length - 1, 1);
  G.lay.shapes.push({ type: 'rect', xref: a.xa, yref: a.yd, x0: d[e - 7], x1: d[e], y0: 0, y1: 1, fillcolor: rgba(K.orange, 0.2), line: { width: 0 }, layer: 'below' }, vl(b, d[e], K.navy, 'dot', 1), hl(b, 0, K.navy, 'solid', 0.6));
  plot('xd-chart', [{ x: d, y: Y.map(v => v / 1000), mode: 'lines', line: { color: K.blue, width: 1.4 }, showlegend: false, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' },
    { x: i1.map(i => d[i]), y: i1.map(g1), mode: 'lines', line: { color: K.green, width: 1 }, name: '100 Δ ln Y<sub>t</sub> (q/q)', xaxis: b.xa, yaxis: b.ya, hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' },
    { x: i4.map(i => d[i]), y: i4.map(g4), mode: 'lines', line: { color: K.red, width: 1.6 }, name: '100 Δ<sub>4</sub> ln Y<sub>t</sub> (y/y)', xaxis: b.xa, yaxis: b.ya, hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' }], G.lay);
  out('xd-out', [T.xd1(qlab(d[e]), nf(g1(e), 2), nf(100 * (Y[e] / Y[e - 1] - 1), 2), nf(g4(e), 2))]);
}

// ---------------------------------------------------------------- worked example: sample autocovariance and autocorrelation
function xsData(src, n) {
  if (src === 'nile') return { x: R.nile.y.slice(0, n), lab: R.nile.yr.slice(0, n).map(String) };
  if (src === 'gdp') { const s = s10('gy'); return { x: s.x.slice(-n), lab: s.d.slice(-n).map(qlab) }; }
  return { x: [3, 5, 4, 6, 7], lab: ['1', '2', '3', '4', '5'] };
}
function blkSACF() {
  const src = $('xs-s').value, sl = src === 'slide'; $('xs-T').disabled = sl; $('xs-T').closest('.ctl').style.display = sl ? 'none' : '';
  const { x, lab } = xsData(src, +$('xs-T').value), n = x.length; set('xs-T', n);
  $('xs-k').max = n - 1; const k = Math.min(+$('xs-k').value, n - 1); set('xs-k', k);
  const dx = src === 'gdp' ? 2 : 0, m = mean(x), dv = x.map(v => v - m), pr = range(0, n - 1 - k, 1).map(t => dv[t] * dv[t + k]);
  const g0 = sum(dv.map(v => v * v)) / n, gk = sum(pr) / n, rk = gk / g0, b = 1.96 / Math.sqrt(n);
  steps('xs-ws', [T.xss1(x.map(v => nf(v, dx)).join(' + '), n, nf(sum(x), dx), nf(m, sl ? 0 : 2), dv.map(v => nf(v, sl ? 0 : 2)).join('; ')),
    T.xss2(nf(sum(dv.map(v => v * v)), sl ? 0 : 1), n, nf(g0, sl ? 1 : 1)),
    T.xss3(k, pr.map((v, t) => `${pn(dv[t], sl ? 0 : 1)} × ${pn(dv[t + k], sl ? 0 : 1)}`).join(' + '), nf(sum(pr), sl ? 0 : 1), n, nf(gk, sl ? 1 : 1)),
    T.xss4(k, nf(gk, 1), nf(g0, 1), nf(rk, sl ? 2 : 3), n, nf(b, 2), Math.abs(rk) > b)]);
  table('xs-tb', ['t', T.x.obs, 'x<sub>t</sub>', 'x<sub>t</sub> − x̄', `x<sub>t+${k}</sub> − x̄`, `(x<sub>t</sub> − x̄)(x<sub>t+${k}</sub> − x̄)`],
    x.map((v, t) => [t + 1, lab[t], num(v, dx), num(dv[t], sl ? 0 : 2), t + k < n ? num(dv[t + k], sl ? 0 : 2) : '', t + k < n ? num(pr[t], sl ? 0 : 1) : '']));
  const r = range(1, n - 1, 1).map(h => sum(range(0, n - 1 - h, 1).map(t => dv[t] * dv[t + h])) / n / g0);
  const G = grid('xs-chart', [{ r: 0, c: 0, t: `Sample ACF, T = ${n}`, xt: 'lag h', y: { range: [-1.05, 1.05] }, x: { dtick: 1 } },
    { r: 0, c: 1, t: `Products (x<sub>t</sub> − x̄)(x<sub>t+${k}</sub> − x̄), t = 1, …, ${n - k}: their sum / T = γ̂(${k})`, xt: 't', x: { dtick: n > 8 ? 2 : 1, range: [0.4, n - 0.4] } }], { cols: 2, h: 320 }), [a, a2] = G.ax;
  G.lay.shapes.push(hl(a, b, K.red, 'dash', 1), hl(a, -b, K.red, 'dash', 1), hl(a, 0, K.navy, 'solid', 0.8), hl(a2, 0, K.navy, 'solid', 0.8));
  // lags 1, …, k (computed so far, lag k in orange) solid; the larger lags faded
  plot('xs-chart', [{ x: range(1, k, 1), y: r.slice(0, k), type: 'bar', width: 0.5, marker: { color: range(1, k, 1).map(h => (h === k ? K.orange : K.blue)) }, showlegend: false, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' },
    { x: range(k + 1, n - 1, 1), y: r.slice(k), type: 'bar', width: 0.5, opacity: 0.3, marker: { color: K.blue }, showlegend: false, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' }, BAND,
    { x: range(1, n - k, 1), y: pr, type: 'bar', width: 0.5, marker: { color: pr.map(v => (v < 0 ? K.red : K.green)) }, showlegend: false, xaxis: a2.xa, yaxis: a2.ya, hovertemplate: `t = %{x}: (x<sub>t</sub> − x̄)(x<sub>t+${k}</sub> − x̄) = %{y:.2f}<extra></extra>` }], G.lay);
  out('xs-out', [T.xs(T.x.xsSrc[src], n, k, nf(gk, sl ? 1 : 1), nf(rk, sl ? 2 : 3), nf(b, 2))]);
}

// ---------------------------------------------------------------- confidence bands for the sample ACF
function blkBands() {
  const s = s10($('xb-s').value), m = +$('xb-m').value; $('xb-h').max = m; const h = Math.min(+$('xb-h').value, m); set('xb-m', m); set('xb-h', h);
  const x = s.x, n = x.length, r = s.r.slice(0, m), p = pacf(r), b = 1.96 / Math.sqrt(n);
  const bb = r.map((_, i) => 1.96 * Math.sqrt((1 + 2 * sum(r.slice(0, i).map(v => v * v))) / n));
  const nOut = r.filter(v => Math.abs(v) > b).length, nOutB = r.filter((v, i) => Math.abs(v) > bb[i]).length, pOut = p.filter(v => Math.abs(v) > b).length;
  const S2 = sum(r.slice(0, h - 1).map(v => v * v));
  steps('xb-ws', [T.xbs1(n, nf(Math.sqrt(n), 2), nf(b, 4)), T.xbs2(h, nf(r[h - 1], 4), Math.abs(r[h - 1]) > b),
    T.xbs3(h, h - 1, nf(S2, 4), n, nf(bb[h - 1], 4), Math.abs(r[h - 1]) > bb[h - 1]), T.xbs4(m, nOut, nf(0.05 * m, 1), nOutB, pOut)]);
  const st = on => txt(on ? '<b>*</b>' : '');
  table('xb-tb', ['h', 'ρ̂(h)', `±${nf(1.96, 2)}/√T`, T.x.bartlett, 'φ̂<sub>hh</sub>', T.x.outA, T.x.outB, T.x.outP],
    r.map((v, i) => hiRow([i + 1, num(v, 4, Math.abs(v) > b), num(b, 4), num(bb[i], 4), num(p[i], 4, Math.abs(p[i]) > b), st(Math.abs(v) > b), st(Math.abs(v) > bb[i]), st(Math.abs(p[i]) > b)], i + 1 === h)),
    { cap: T.x.xbCap(s.name, n) });
  const G = grid('xb-chart', [{ r: 0, c: 0, t: `${T.x.s10en[s.k]}: sample ACF`, xt: 'lag h' }, { r: 0, c: 1, t: `${T.x.s10en[s.k]}: sample PACF`, xt: 'lag h' }], { cols: 2, h: 340 });
  const [a, c] = G.ax, L = lags(m);
  G.lay.shapes.push(hl(a, b, K.red, 'dash', 1), hl(a, -b, K.red, 'dash', 1), hl(a, 0, K.navy, 'solid', 0.8), hl(c, b, K.red, 'dash', 1), hl(c, -b, K.red, 'dash', 1), hl(c, 0, K.navy, 'solid', 0.8), vl(a, h, K.navy, 'dot', 1));
  plot('xb-chart', [{ x: [...L, ...[...L].reverse()], y: [...bb, ...[...bb].reverse().map(v => -v)], fill: 'toself', fillcolor: rgba(K.teal, 0.18), line: { width: 0 }, mode: 'lines', name: 'Bartlett band, MA(h − 1)', xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' },
    { x: L, y: r, type: 'bar', width: 0.55, marker: { color: K.blue }, name: 'ACF', xaxis: a.xa, yaxis: a.ya, hovertemplate: 'lag %{x}: %{y:.4f}<extra></extra>' },
    { x: L, y: p, type: 'bar', width: 0.55, marker: { color: K.green }, name: 'PACF', xaxis: c.xa, yaxis: c.ya, hovertemplate: 'lag %{x}: %{y:.4f}<extra></extra>' }, BAND], G.lay);
  out('xb-out', [T.xb(s.name, n, nf(b, 3), m, nOut, nf(0.05 * m, 1), nOutB)]);
}

// ---------------------------------------------------------------- worked example: the Ljung-Box statistic
const XL = { n: 100, r: [0.25, 0.12, -0.08] };
function blkWxLB() {
  const src = $('xl-s').value, sl = src === 'slide', mmax = sl ? 3 : 30; $('xl-m').max = mmax;
  const m = Math.min(+$('xl-m').value, mmax); set('xl-m', m);
  let n, P;
  if (sl) { n = XL.n; let a = 0, b = 0; P = XL.r.map((v, i) => { const h = i + 1; a += v * v; b += v * v / (n - h); const q = n * a, qs = n * (n + 2) * b; return { h, r: v, q, qs, pq: chi2sf(q, h), pqs: chi2sf(qs, h), cr: chi2ppf(0.95, h) }; }); }
  else { const s = s10(src); n = s.x.length; P = lbPath(s.x, 30, s.r); }
  const Pm = P.slice(0, m), last = Pm[m - 1], dr = sl ? 4 : 5, more = m > 4 ? ' + …' : '';
  const sq = Pm.slice(0, 4).map(o => nf(o.r * o.r, dr)), fr = Pm.slice(0, 4).map(o => `${nf(o.r * o.r, dr)}/${n - o.h}`);
  steps('xl-ws', [T.xls1(n, m, Pm.slice(0, 4).map(o => nf(o.r, sl ? 2 : 4)).join('; ') + (m > 4 ? '; …' : ''), nf(1.96 / Math.sqrt(n), 3), Pm.filter(o => Math.abs(o.r) > 1.96 / Math.sqrt(n)).length),
    T.xls2(m, n, sq.join(' + ') + more, nf(last.q / n, dr), nf(last.q, 2)),
    T.xls3(m, n, n + 2, fr.join(' + ') + more, nf(last.qs / (n * (n + 2)), 6), nf(last.qs, 2)),
    T.xls4(m, nf(last.cr, 2), nf(last.qs, 2), last.qs > last.cr, pv(last.pqs), pv(last.pq))]);
  table('xl-tb', ['h', 'ρ̂(h)', 'ρ̂(h)<sup>2</sup>', 'ρ̂(h)<sup>2</sup>/(T − h)', 'Q(h)', 'Q*(h)', `χ²<sub>${nf(0.95, 2)}</sub>(h)`, 'p (Q*)'],
    Pm.map(o => [o.h, num(o.r, 4), num(o.r * o.r, 5), num(o.r * o.r / (n - o.h), 7), num(o.q, 2), num(o.qs, 2, o.qs > o.cr), num(o.cr, 2), txt(pv(o.pqs))]));
  const G = grid('xl-chart', [{ r: 0, c: 0, t: 'Q(m), Q*(m) and the 5% critical value', xt: 'number of lags m', x: { dtick: sl ? 1 : 5 } }, { r: 0, c: 1, t: 'p-values (log scale; below 10<sup>−16</sup> drawn at 10<sup>−16</sup>)', xt: 'number of lags m', y: { type: 'log', exponentformat: 'power' }, x: { dtick: sl ? 1 : 5 } }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax, H = P.map(o => o.h), fl = v => Math.max(v, 1e-16);
  G.lay.shapes.push(vl(a, m, K.navy, 'dot', 1), vl(b, m, K.navy, 'dot', 1), hl(b, 0.05, K.red, 'dash', 1.2));
  plot('xl-chart', [{ x: H, y: P.map(o => o.q), mode: 'lines+markers', line: { color: K.amber, width: 1.6 }, marker: { size: 5 }, name: 'Box-Pierce Q(m)', xaxis: a.xa, yaxis: a.ya, hovertemplate: 'm = %{x}: %{y:.2f}<extra></extra>' },
    { x: H, y: P.map(o => o.qs), mode: 'lines+markers', line: { color: K.blue, width: 1.8 }, marker: { size: 5 }, name: 'Ljung-Box Q*(m)', xaxis: a.xa, yaxis: a.ya, hovertemplate: 'm = %{x}: %{y:.2f}<extra></extra>' },
    { x: H, y: P.map(o => o.cr), mode: 'lines', line: { color: K.navy, dash: 'dash', width: 1.4 }, name: 'χ²<sub>0.95</sub>(m)', xaxis: a.xa, yaxis: a.ya, hovertemplate: 'm = %{x}: %{y:.2f}<extra></extra>' },
    { x: H, y: P.map(o => fl(o.pq)), mode: 'lines+markers', line: { color: K.amber, width: 1.4 }, marker: { size: 5 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hovertemplate: 'Q, m = %{x}: %{y:.3g}<extra></extra>' },
    { x: H, y: P.map(o => fl(o.pqs)), mode: 'lines+markers', line: { color: K.blue, width: 1.6 }, marker: { size: 5 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hovertemplate: 'Q*, m = %{x}: %{y:.3g}<extra></extra>' },
    leg('5% level', { color: K.red, dash: 'dash', width: 1.2 })], G.lay);
  out('xl-out', [T.xl(sl ? T.x.slideEx : s10(src).name, m, nf(last.q, 2), nf(last.qs, 2), pe(last.pqs), last.qs > last.cr)]);
}

// ---------------------------------------------------------------- Ljung-Box statistics of real series: the table of the slide
function blkLBtab() {
  const m = +$('lt-m').value, bp = $('lt-q').value === 'bp'; set('lt-m', m);
  const cr = chi2ppf(0.95, m), rows = [], rat = [];
  S10.forEach(s => {
    const P = lbPath(s.x, Math.max(m, 4), s.r), o = P[m - 1], q = bp ? o.q : o.qs, p = bp ? o.pq : o.pqs;
    rat.push(q / cr);
    rows.push([s.name, txt(nf(s.x.length, 0)), num(P[0].r, 2), num(P[3].r, 2), num(q, 1), num(cr, 2), txt(p < 0.001 ? '&lt; ' + nf(0.001, 3) : p < 0.05 ? nf(p, 3) : `<b>${nf(p, 3)}</b>`), num(q / cr, 1)]);
  });
  const st = bp ? 'Q(m)' : 'Q*(m)';
  table('lt-tb', [T.x.series, 'T', 'ρ̂(1)', 'ρ̂(4)', st.replace('m', m), `χ²<sub>${nf(0.95, 2)}</sub>(${m})`, 'p', `${st.replace('m', m)}/χ²<sub>${nf(0.95, 2)}</sub>`], rows);
  const G = grid('lt-chart', [{ r: 0, c: 0, t: `${bp ? 'Q(m)' : 'Q*(m)'} / 5% critical value, m = ${m}`, xt: 'log scale', x: { type: 'log' }, y: { autorange: 'reversed' } }], { cols: 1, h: 380, legend: false });
  const a = G.ax[0], SH = ['S&P 500 returns', 'S&P 500 squared', 'BET returns', 'BET squared', 'EUR/RON returns', 'GDP q/q', 'GDP y/y', 'HICP m/m', 'Nile', 'Sunspots'];
  G.lay.margin.l = 118; G.lay.shapes.push(vl(a, 1, K.red, 'dash', 1.2));
  plot('lt-chart', [{ y: SH, x: rat, type: 'bar', orientation: 'h', marker: { color: S10.map((_, j) => PAL[j % PAL.length]) }, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{y}: %{x:.1f}<extra></extra>' }], G.lay);
  const nrej = rat.filter(v => v > 1).length, jm = rat.indexOf(Math.min(...rat));
  out('lt-out', [T.lt(bp ? 'Box–Pierce' : 'Ljung–Box', m, nf(cr, 2), nrej, S10[jm].name, nf(rat[jm], 1))]);
}

// ---------------------------------------------------------------- worked example: the Box-Cox transformation
function guerreroP(y, l, p) {
  const nb = Math.floor(y.length / p), z = y.slice(y.length - nb * p), ratio = [], bl = [];
  for (let i = 0; i < nb; i++) { const b = z.slice(p * i, p * i + p), mu = mean(b), s = sdev(b); ratio.push(s / mu ** (1 - l)); bl.push({ i0: y.length - nb * p + p * i, mu, s, q: s / mu ** (1 - l) }); }
  return { cv: sdev(ratio) / mean(ratio), bl };
}
const XC = (() => [['nom', R.nom.y, R.nom.d, 4], ['gdp', R.gdp.y, R.gdp.d, 4], ['hicp', D.real.hicp.y, D.real.hicp.d.map(m2d), 12]].map(([k, y, d, p]) => ({ k, y, d, p })))();
const xcGrid = {};
function blkXBC() {
  const l = +$('xc-l').value, y2 = +$('xc-y').value, s = XC.find(o => o.k === $('xc-s').value), y1 = 100; set('xc-l', nf(l, 2)); set('xc-y', y2);
  const bc = v => (Math.abs(l) < 1e-8 ? Math.log(v) : (v ** l - 1) / l), w1 = bc(y1), w2 = bc(y2), f3 = v => nf(v, Math.abs(v - Math.round(v)) < 5e-4 ? 0 : 3);
  XC.forEach(o => { if (!xcGrid[o.k]) { const g = range(-1, 2, 0.01), cv = g.map(v => guerreroP(o.y, v, o.p).cv), i = cv.indexOf(Math.min(...cv)); xcGrid[o.k] = { g, cv, lh: g[i], cmin: cv[i] }; } });
  const Gs = xcGrid[s.k], gq = guerreroP(s.y, l, s.p), B = gq.bl;
  const blab = b => (s.p === 4 ? `${qlab(s.d[b.i0])}–${qlab(s.d[b.i0 + 3])}` : `${s.d[b.i0].slice(0, 7)} – ${s.d[b.i0 + 11].slice(0, 7)}`);
  steps('xc-ws', [Math.abs(l) < 1e-8 ? T.xcs1L(y1, y2, f3(w1), f3(w2)) : T.xcs1(nf(l, 2), y1, y2, f3(y1 ** l), f3(y2 ** l), f3(w1), f3(w2)),
    T.xcs2(y2 - y1, f3(w2 - w1), nf(y2 / y1, 2), nf(w2 / w1, 2)),
    Math.abs(l) < 1e-8 ? T.xcs3L(f3(w2), nf(Math.exp(w2), 1)) : T.xcs3(nf(l, 2), (w2 < 0 ? `(${f3(w2)})` : f3(w2)), nf((l * w2 + 1) ** (1 / l), 1)),
    T.xcs4(T.x.xcName[s.k], B.length, s.p, blab(B[0]), nf(B[0].s, 1), nf(B[0].mu, 1), nf(1 - l, 2), nf(B[0].q, 4), blab(B[B.length - 1]), nf(B[B.length - 1].q, 4), nf(gq.cv, 3), nf(Gs.lh, 2), nf(Gs.cmin, 3))]);
  table('xc-tb', [T.x.series, T.x.blocks, 'λ̂', T.x.critAt('λ̂'), T.x.critAt('λ = 0'), T.x.critAt('λ = 1'), T.x.critAt(`λ = ${nf(l, 2)}`)],
    XC.map(o => hiRow([T.x.xcName[o.k], txt(`${Math.floor(o.y.length / o.p)} × ${o.p}`), num(xcGrid[o.k].lh, 2, true), num(xcGrid[o.k].cmin, 3), num(guerreroP(o.y, 0, o.p).cv, 3), num(guerreroP(o.y, 1, o.p).cv, 3), num(guerreroP(o.y, l, o.p).cv, 3)], o.k === s.k)), { cap: T.x.xcCap });
  const sel = [0, 1, 2, 3, B.length - 4, B.length - 3, B.length - 2, B.length - 1];
  table('xc-tb2', [T.x.block, T.x.meanI, T.x.sdI, `sd<sub>i</sub>/mean<sub>i</sub><sup>${nf(1 - l, 2)}</sup>`],
    sel.map((i, j) => (j === 4 ? [[{ v: '…' }, '', '', ''], [blab(B[i]), num(B[i].mu, 1), num(B[i].s, 1), num(B[i].q, 4)]] : [[blab(B[i]), num(B[i].mu, 1), num(B[i].s, 1), num(B[i].q, 4)]])).flat(),
    { cap: T.x.xcCap2(T.x.xcName[s.k], B.length), note: T.x.xcNote(nf(mean(B.map(b => b.q)), 4), nf(sdev(B.map(b => b.q)), 4), nf(gq.cv, 3)) });
  const G = grid('xc-chart', [{ r: 0, c: 0, t: "Guerrero's criterion against λ", xt: 'λ', yt: 'coefficient of variation' }, { r: 0, c: 1, t: `${{ nom: 'Nominal GDP', gdp: 'Real GDP', hicp: 'HICP index' }[s.k]}: Box-Cox, λ = ${l.toFixed(2)}`, x: { type: 'date' } }], { cols: 2, h: 360, legendRows: 2 });
  const [a, a2] = G.ax, cols = { nom: K.blue, gdp: K.green, hicp: K.red }, nm = { nom: 'Nominal GDP (quarterly)', gdp: 'Real GDP (quarterly)', hicp: 'HICP index (monthly)' };
  G.lay.shapes.push(vl(a, l, K.navy, 'dot', 1.2));
  plot('xc-chart', XC.map(o => ({ x: xcGrid[o.k].g, y: xcGrid[o.k].cv, mode: 'lines', line: { color: cols[o.k], width: o.k === s.k ? 2.4 : 1.3 }, name: nm[o.k], xaxis: a.xa, yaxis: a.ya, hovertemplate: nm[o.k] + ', λ = %{x:.2f}: %{y:.3f}<extra></extra>' }))
    .concat(XC.map(o => ({ x: [xcGrid[o.k].lh], y: [xcGrid[o.k].cmin], mode: 'markers', marker: { color: cols[o.k], size: 9 }, showlegend: false, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'λ̂ = %{x:.2f}<extra></extra>' })))
    .concat(XC.map(o => ({ x: [l], y: [guerreroP(o.y, l, o.p).cv], mode: 'markers', marker: { color: '#FFFFFF', size: 8, line: { color: cols[o.k], width: 2 } }, showlegend: false, xaxis: a.xa, yaxis: a.ya, hovertemplate: `λ = ${nf(l, 2)}: %{y:.3f}<extra></extra>` })))
    .concat([{ x: s.d, y: s.y.map(bc), mode: 'lines', line: { color: cols[s.k], width: 1.3 }, showlegend: false, xaxis: a2.xa, yaxis: a2.ya, hovertemplate: '%{x|%Y-%m}: %{y:.4g}<extra></extra>' }]), G.lay);
  out('xc-out', [T.xc(nf(l, 2), y2, f3(w1), f3(w2), T.x.xcName[s.k], nf(gq.cv, 3), nf(Gs.lh, 2))]);
}

// ================================================================= wiring
const ALL = [blkFour, blkEns, blkCounter, blkNonstat, blkWN, blkRW, blkSpot, blkSP, blkWold, blkErgo, blkBart, blkACFm, blkACFP, blkBetACF,
  blkLBsize, blkLBstats, blkGDP, blkBoxCox, blkOverdiff, blkTextbook];
// blocks of the worked examples and result tables (own frames): [anchor, prefix, function, controls, player parameter, values, ms]
const XB = [['acov-props', 'cp', blkAcov, ['cp-r1', 'cp-r2', 'cp-a2', 'cp-a3'], 'cp-r2', range(-1, 1, 0.05), 350],
  ['wx-ma1', 'xm', blkMA1, ['xm-th', 'xm-s2'], 'xm-th', range(-2, 2, 0.1), 300], ['wx-ar1', 'xa', blkAR1, ['xa-phi', 'xa-s2', 'xa-h'], 'xa-phi', range(0, 0.99, 0.03), 300],
  ['wx-diff', 'xd', blkDiff, ['xd-s', 'xd-e'], 'xd-e', range(8, 125, 3), 300], ['wx-sacf', 'xs', blkSACF, ['xs-s', 'xs-T', 'xs-k'], 'xs-k', range(1, 4, 1), 900],
  ['bands', 'xb', blkBands, ['xb-s', 'xb-m', 'xb-h'], 'xb-m', range(5, 30, 1), 350], ['wx-lb', 'xl', blkWxLB, ['xl-s', 'xl-m'], 'xl-m', range(1, 30, 1), 350],
  ['lb-table', 'lt', blkLBtab, ['lt-m', 'lt-q'], 'lt-m', range(1, 30, 1), 450], ['wx-boxcox', 'xc', blkXBC, ['xc-l', 'xc-y', 'xc-s'], 'xc-l', range(-1, 2, 0.1), 350]];
function veilTables(b, x) {          // the live tables and worked examples of a prediction block stay blurred as long as its readout
  const blk = $(b), o = $(x + '-out'); if (!blk || !o) return;
  const xs = [...blk.querySelectorAll('.c1t, .c1x')], f = () => xs.forEach(e => e.classList.toggle('c1veil', o.classList.contains('veiled-out')));
  f(); new MutationObserver(f).observe(o, { attributes: true, attributeFilter: ['class'] });
}
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
  on(['fs-y0', 'fs-len'], blkFour); P('fs-play', 'fs-y0', range(1995, 2021, 1), 450);
  $('fs-play').addEventListener('click', () => { if (+$('fs-len').value >= 32) $('fs-len').value = '5'; }, true);
  on(['en-phi', 'en-t'], blkEns); P('en-play', 'en-t', range(2, 100, 2), 220);
  on(['ce-k'], blkCounter); P('ce-play', 'ce-k', range(1, 60, 1), 200);
  on(['ns-b', 'ns-s', 'ns-d', 'ns-t'], blkNonstat); P('ns-play', 'ns-t', range(50, 300, 10), 300);
  on(['wn-a', 'wn-n'], blkWN); P('wn-play', 'wn-n', range(100, 1000, 50), 500);
  on(['rw-t', 'rw-c'], blkRW); P('rw-play', 'rw-t', range(10, 250, 10), 250);
  on(['sr-n', 'sr-v'], blkSpot); P('sr-play', 'sr-n', range(100, 500, 50), 600);
  on(['sp-y0', 'sp-d'], blkSP); P('sp-play', 'sp-y0', range(2000, 2024, 1), 500);
  // ▶ of a block that compares several processes moves one coefficient and gives the same value to the other panels
  const link = (src, btn, f) => $(src).addEventListener('input', () => { const p = players[btn]; if (p && p.timer) f(+$(src).value); });
  link('wd-a', 'wd-play', v => { $('wd-b').value = -v; $('wd-th').value = v; });
  on(['wd-a', 'wd-b', 'wd-th'], blkWold); P('wd-play', 'wd-a', range(0, 1, 0.05), 350);
  on(['eg-phi', 'eg-z', 'eg-T'], blkErgo); P('eg-play', 'eg-T', [10, 20, 30, 50, 80, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000], 450);
  on(['bt-m', 'bt-T'], blkBart); P('bt-play', 'bt-m', range(1, 20, 1), 450);
  on(['am-phi', 'am-th', 'am-T'], blkACFm); P('am-play', 'am-T', range(50, 500, 25), 350);
  link('ap-b', 'ap-play', v => { $('ap-a').value = v; $('ap-c').value = v; });
  on(['ap-a', 'ap-b', 'ap-c'], blkACFP); P('ap-play', 'ap-b', range(-0.4, 0.45, 0.05), 450);
  on(['ba-s', 'ba-y0'], blkBetACF); P('ba-play', 'ba-y0', range(2000, 2024, 1), 500);
  on(['ls-m', 'ls-T', 'ls-a'], blkLBsize); P('ls-play', 'ls-m', range(1, 20, 1), 400);
  on(['lq-m', 'lq-y0'], blkLBstats); P('lq-play', 'lq-y0', range(1995, 2025, 1), 450);
  on(['gt-k'], blkGDP); P('gt-play', 'gt-k', range(1, 8, 1), 900);
  on(['bc-l'], blkBoxCox); P('bc-play', 'bc-l', range(-1, 2, 0.1), 300);
  on(['od-phi'], blkOverdiff); P('od-play', 'od-phi', range(0, 1, 0.05), 400);
  on(['tb-y'], blkTextbook); P('tb-play', 'tb-y', range(1875, 1965, 1), 150);
  const live = XB.filter(([, x]) => has(x + '-out'));
  live.forEach(([, x, f, ids, pid, vals, ms]) => { on(ids, f); P(x + '-play', pid, vals, ms); });
  if (has('xd-play')) $('xd-play').addEventListener('click', () => { if ($('xd-s').value !== 'gdp') { $('xd-s').value = 'gdp'; $('xd-e').disabled = false; $('xd-e').closest('.ctl').style.display = ''; } }, true);
  if (has('xl-play')) $('xl-play').addEventListener('click', () => { if ($('xl-s').value === 'slide') { $('xl-s').value = 'bet'; $('xl-m').max = 30; } }, true);
  if (has('xs-play')) $('xs-play').addEventListener('click', () => { if (+$('xs-k').max < 4) $('xs-k').max = 4; }, true);
  const PFX = { 'four-series': 'fs', 'ensemble': 'en', 'counterexample': 'ce', 'nonstationary': 'ns', 'white-noise': 'wn', 'random-walk': 'rw',
    'spot-real': 'sr', 'sp500-diff': 'sp', 'wold': 'wd', 'ergodicity': 'eg', 'bartlett': 'bt', 'acf-models': 'am', 'acf-pacf': 'ap',
    'bet-acf': 'ba', 'lb-size': 'ls', 'lb-stats': 'lq', 'gdp-transform': 'gt', 'boxcox': 'bc', 'overdiff': 'od', 'textbook': 'tb' };
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if (has(b)) { predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: { out: true } }, sp)); veilTables(b, PFX[b]); } });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  ALL.forEach(f => f());
  live.forEach(b => b[2]());
  let w = window.innerWidth, tm = null;
  window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(() => { if ((window.innerWidth < 720) !== (w < 720) || Math.abs(window.innerWidth - w) > 200) { w = window.innerWidth; ALL.forEach(f => f()); live.forEach(b => b[2]()); } }, 250); });
}
window.CH1 = { chi2sf, chi2ppf, lbq, lbPath, garch1, ar1, guerrero, guerreroP, boxcox, ceOdd, btSim, R, RET, LQ, S10, XC, days, ALL, XB };
if (!window.CH1_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
