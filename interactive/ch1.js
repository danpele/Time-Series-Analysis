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
  const tr = [], res = [T.fs0(y0, y1 - 1, full)];
  S.forEach(([d, y, c, w], j) => {
    const a = G.ax[j];
    tr.push({ x: d, y, mode: 'lines', line: { color: c, width: w }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: '%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>' });
    if (j === 1) G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.6));
    if (!full) G.lay.shapes.push({ type: 'rect', xref: a.xa, yref: a.yd, x0: `${y0}-01-01`, x1: `${y1}-01-01`, y0: 0, y1: 1, fillcolor: rgba(K.orange, 0.15), line: { width: 0 }, layer: 'below' });
    const v = y.filter((_, i) => d[i] >= `${y0}` && d[i] < `${y1}`);
    res.push(v.length > 2 ? T.fs(j, nf(mean(v), 2), nf(sdev(v), 2), v.length) : T.fsNone(j));
  });
  plot('fs-chart', tr, G.lay);
  out('fs-out', res);
}

// ================================================================= ensemble of paths
function blkEns() {
  const phi = +$('en-phi').value, t0 = +$('en-t').value; set('en-phi', nf(phi, 2)); set('en-t', t0);
  const E = D.ens, n = 100, sd = 1 / Math.sqrt(1 - phi * phi), t = lags(n);
  const A = E.e.map((e, i) => ar1(e, phi, sd * E.z0[i])), W = E.w.map(cumsum);
  const G = grid('en-chart', [{ r: 0, c: 0, t: `Stationary AR(1), φ = ${phi}: same band at every t`, xt: 't' }, { r: 0, c: 1, t: 'Random walk: the band widens like √t', xt: 't' }], { cols: 2, h: 380 });
  const [a, b] = G.ax, tr = [], pa = joinPaths(A, t), pw = joinPaths(W, t);
  tr.push({ x: t.concat([...t].reverse()), y: t.map(() => 1.96 * sd).concat(t.map(() => -1.96 * sd)), fill: 'toself', fillcolor: rgba(K.orange, 0.18), line: { width: 0 }, mode: 'lines', name: '±1.96 sd(X<sub>t</sub>)', xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' });
  tr.push({ x: t.concat([...t].reverse()), y: t.map(v => 1.96 * Math.sqrt(v)).concat([...t].reverse().map(v => -1.96 * Math.sqrt(v))), fill: 'toself', fillcolor: rgba(K.orange, 0.18), line: { width: 0 }, mode: 'lines', showlegend: false, xaxis: b.xa, yaxis: b.ya, hoverinfo: 'skip' });
  tr.push({ x: pa.x, y: pa.y, mode: 'lines', line: { color: rgba(K.blue, 0.45), width: 0.7 }, name: 'one path', xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' });
  tr.push({ x: pw.x, y: pw.y, mode: 'lines', line: { color: rgba(K.green, 0.45), width: 0.7 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hoverinfo: 'skip' });
  const lim = Math.max(25, 1.96 * sd * 1.15);
  G.lay.yaxis.range = [-lim, lim]; G.lay.yaxis2.range = [-lim, lim];
  G.lay.shapes.push(vl(a, t0, K.navy, 'dot', 1), vl(b, t0, K.navy, 'dot', 1));
  tr.push(leg(`cross-section at t = ${t0}`, { color: K.navy, dash: 'dot', width: 1.2 }));
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
  const b = +$('ns-b').value, s1 = +$('ns-s').value, sh = +$('ns-d').value; set('ns-b', nf(b, 3)); set('ns-s', nf(s1, 1)); set('ns-d', nf(sh, 1));
  const n = 300, t = range(0, n - 1, 1), N0 = D.ns;
  const X = [N0.a, N0.b.map((v, i) => b * i + v), N0.c.map((v, i) => v * (0.4 + (s1 - 0.4) * i / (n - 1))), N0.d.map((v, i) => v + (i >= n / 2 ? sh : 0))];
  const names = ['Series A', 'Series B', 'Series C', 'Series D'], cols = [K.blue, K.red, K.green, K.purple];
  const G = grid('ns-chart', names.map((nm, j) => ({ r: j >> 1, c: j & 1, t: nm })), { cols: 2, h: 440, legend: false });
  const tr = X.map((x, j) => ({ x: t, y: x, mode: 'lines', line: { color: cols[j], width: 1 }, xaxis: G.ax[j].xa, yaxis: G.ax[j].ya, showlegend: false, hovertemplate: 't = %{x}: %{y:.2f}<extra></extra>' }));
  plot('ns-chart', tr, G.lay);
  out('ns-out', X.map((x, j) => { const h1 = x.slice(0, n / 2), h2 = x.slice(n / 2); return T.ns(names[j].slice(-1), nf(mean(h1), 2), nf(mean(h2), 2), nf(sdev(h1), 2), nf(sdev(h2), 2)); }));
}

// ================================================================= three kinds of white noise
function garch1(z, alpha, beta, n) {   // generate_all_charts.simulate_garch with omega = 1 - alpha - beta (variance 1)
  const om = 1 - alpha - beta, e = new Array(z.length).fill(0); let s2 = om / (1 - alpha - beta);
  for (let t = 1; t < z.length; t++) { s2 = om + alpha * e[t - 1] ** 2 + beta * s2; e[t] = Math.sqrt(s2) * z[t]; }
  return e.slice(z.length - n);
}
function blkWN() {
  const al = +$('wn-a').value, be = 0.85; set('wn-a', nf(al, 2));
  const n = 1000, S = [['Gaussian i.i.d.', D.wn.ga, K.blue], ['Laplace i.i.d.', D.wn.la, K.green],
    [Math.abs(al - 0.1) < 1e-9 ? 'GARCH(1,1): weak WN' : `GARCH(1,1), α = ${al}`, garch1(D.wn.z, al, be, n), K.red]];
  const A2 = S.map(([, x]) => acf(x.map(v => v * v), 20)), top = Math.max(0.35, ...A2.flat()) + 0.03;
  const cells = S.map(([nm], j) => ({ r: 0, c: j, t: nm, y: { range: [-7, 7] } })).concat(S.map((_, j) => ({ r: 1, c: j, t: 'ACF of X<sub>t</sub><sup>2</sup>', y: { range: [-0.12, top] } })));
  const G = grid('wn-chart', cells, { cols: 3, h: 440 }), tr = [], res = [];
  S.forEach(([nm, x, c], j) => {
    tr.push({ x: range(0, 499, 1), y: x.slice(0, 500), mode: 'lines', line: { color: c, width: 0.6 }, xaxis: G.ax[j].xa, yaxis: G.ax[j].ya, showlegend: false, hovertemplate: 't = %{x}: %{y:.2f}<extra></extra>' });
    const x2 = x.map(v => v * v);
    acfPanel(tr, G.lay, G.ax[3 + j], A2[j], n, c);
    const l1 = lbq(x, 10), l2 = lbq(x2, 10);
    res.push(T.wn(j, nf(al, 2), pe(l1.p), nf(l2.q, 1), pe(l2.p), nf(kurt(x), 2)));
  });
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
  tr.push({ x: t, y: D.rw.var, mode: 'lines', line: { color: K.blue, width: 1.8 }, name: `variance across ${D.rw.nsim} paths`, xaxis: b.xa, yaxis: b.ya, hovertemplate: 't = %{x}: %{y:.1f}<extra></extra>' },
    { x: t, y: t, mode: 'lines', line: { color: K.orange, dash: 'dash', width: 1.6 }, name: 'theory: tσ²', xaxis: b.xa, yaxis: b.ya, hoverinfo: 'skip' },
    { x: [t0], y: [D.rw.var[t0 - 1]], mode: 'markers', marker: { color: K.navy, size: 9 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hoverinfo: 'skip' });
  G.lay.shapes.push(vl(a, t0, K.navy, 'dot', 1), vl(b, t0, K.navy, 'dot', 1));
  plot('rw-chart', tr, G.lay);
  const res = [T.rw1(t0, nf(D.rw.var[t0 - 1], 1), t0, pct(D.rw.out[t0 - 1], 1))];
  if (t0 <= n - 10) res.push(T.rw2(t0, t0 + 10, nf(D.rw.c10[t0 - 1], 3), nf(Math.sqrt(t0 / (t0 + 10)), 3)));
  if (c) res.push(T.rw3(nf(c, 2), t0, nf(c * t0, 1)));
  out('rw-out', res);
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
    ['Random walk: ψ<sub>j</sub> = 1', j.map(() => 1), K.red, 'rw', 1]];
  const all = cases.flatMap(c => c[1]), lo = Math.min(-0.7, ...all) * 1.08, hi = Math.max(1.05, ...all) * 1.05;
  const G = grid('wd-chart', cases.map((c, k) => ({ r: 0, c: k, t: c[0], xt: 'j', y: { range: [lo, hi] } })), { cols: 4, h: 320, legend: false, gx: 0.05 });
  const tr = cases.map((c, k) => ({ x: j, y: c[1], type: 'bar', width: 0.6, marker: { color: c[2] }, xaxis: G.ax[k].xa, yaxis: G.ax[k].ya, showlegend: false, hovertemplate: 'ψ<sub>%{x}</sub> = %{y:.3f}<extra></extra>' }));
  G.ax.forEach(x => G.lay.shapes.push(hl(x, 0, K.navy, 'solid', 0.6)));
  plot('wd-chart', tr, G.lay);
  out('wd-out', cases.slice(0, 3).map(([, ps, , kind, v]) => {
    const s2 = kind === 'ma' ? 1 + v * v : Math.abs(v) < 1 ? 1 / (1 - v * v) : null;
    return T.wd(kind, nf(v, 2), nf(ps[5], 3), nf(ps[10], 3), s2 === null ? null : nf(s2, 2));
  }).concat([T.wdRW()]));
}

// ================================================================= time averages: ergodic and non-ergodic
function blkErgo() {
  const phi = +$('eg-phi').value, sz = +$('eg-z').value; set('eg-phi', nf(phi, 2)); set('eg-z', nf(sz, 1));
  const E = D.ergo, n = 2000, t = lags(n);
  const A = E.e.map(e => ar1(e, phi).slice(E.burn)), B = E.u.map((u, i) => u.map(v => sz * E.z[i] + v));
  const run = x => { let s = 0; return x.map((v, i) => (s += v) / (i + 1)); };
  const G = grid('eg-chart', [{ r: 0, c: 0, t: `AR(1), φ = ${phi}: X̄<sub>T</sub> → μ = 0 on every path`, xt: 'T (log scale)', yt: 'X̄<sub>T</sub>', x: { type: 'log' } },
    { r: 0, c: 1, t: `X<sub>t</sub> = Z + ε<sub>t</sub>` + (sz === 1 ? '' : `, sd(Z) = ${sz}`) + ': X̄<sub>T</sub> → Z, not μ = 0', xt: 'T (log scale)', x: { type: 'log' } }], { cols: 2, h: 360, legend: false });
  const [a, b] = G.ax, tr = [], ea = [], eb = [];
  A.forEach((x, i) => { const m = run(x); ea.push(m[n - 1]); tr.push({ x: t, y: m, mode: 'lines', line: { color: PAL[i % 8], width: 1 }, xaxis: a.xa, yaxis: a.ya, showlegend: false, hovertemplate: 'T = %{x}: %{y:.3f}<extra></extra>' }); });
  B.forEach((x, i) => { const m = run(x); eb.push(m[n - 1]); tr.push({ x: t, y: m, mode: 'lines', line: { color: PAL[i % 8], width: 1 }, xaxis: b.xa, yaxis: b.ya, showlegend: false, hovertemplate: 'T = %{x}: %{y:.3f}<extra></extra>' }); });
  const all = tr.flatMap(s => s.y), lim = Math.max(...all.map(Math.abs)) * 1.05;
  G.lay.yaxis.range = [-lim, lim]; G.lay.yaxis2.range = [-lim, lim];
  G.lay.shapes.push(hl(a, 0, K.navy, 'dash', 0.8), hl(b, 0, K.navy, 'dash', 0.8));
  plot('eg-chart', tr, G.lay);
  const zs = E.z.map(z => sz * z);
  out('eg-out', [T.eg1(nf(phi, 2), nf(Math.min(...ea), 2), nf(Math.max(...ea), 2), nf(1 / ((1 - phi) * Math.sqrt(n)), 3)),
    T.eg2(nf(sz, 1), nf(Math.min(...eb), 2), nf(Math.max(...eb), 2), nf(Math.min(...zs), 2), nf(Math.max(...zs), 2))]);
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
  const tr = [{ x: S.r1, type: 'histogram', histnorm: 'probability density', nbinsx: 60, marker: { color: rgba(K.blue, 0.6) }, name: `ρ̂(1), ${nsim} samples of T = ${Tn}`, xaxis: a.xa, yaxis: a.ya },
    { x: gx, y: gx.map(v => Math.sqrt(Tn / (2 * Math.PI)) * Math.exp(-Tn * v * v / 2)), mode: 'lines', line: { color: K.orange, width: 2 }, name: 'N(0, 1/T)', xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' },
    leg('±1.96/√T', { color: K.red, dash: 'dash', width: 1.2 }),
    { x: k.map(v => v - 0.18), y: emp, type: 'bar', width: 0.36, marker: { color: K.blue }, name: 'simulated share', xaxis: c.xa, yaxis: c.ya, customdata: k, hovertemplate: '%{customdata} outside: %{y:.3f}<extra></extra>' },
    { x: k.map(v => v + 0.18), y: k.map(v => binom(m, v, 0.05)), type: 'bar', width: 0.36, marker: { color: K.orange }, name: `Binomial(${m}, 0.05)`, xaxis: c.xa, yaxis: c.ya, customdata: k, hovertemplate: '%{customdata} outside: %{y:.3f}<extra></extra>' }];
  G.lay.shapes.push(vl(a, b, K.red, 'dash', 1), vl(a, -b, K.red, 'dash', 1));
  G.lay.xaxis.range = [-Math.max(0.45, 4.5 / Math.sqrt(Tn)), Math.max(0.45, 4.5 / Math.sqrt(Tn))];
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
  out('ba-out', [T.ba1(s.name, s.d[Math.max(0, i0)], n, nf(1.96 / Math.sqrt(n), 3)), T.ba2(nf(A[0][0], 3), nf(l1.q, 1), pe(l1.p)),
    T.ba3(nf(A[1][0], 2), nf(A[1][49], 2), nf(A[2][0], 2), nf(l2.q, 0))]);
}

// ================================================================= size of Box-Pierce and Ljung-Box
function blkLBsize() {
  const m = +$('ls-m').value, Tn = $('ls-T').value, al = $('ls-a').value; set('ls-m', m);
  const Rr = D.lbs.rates, cases = [[50, 10], [50, 20], [100, 10], [100, 20], [500, 10], [500, 20]], A = 100 * +al;
  const G = grid('ls-chart', [{ r: 0, c: 0, t: `Gaussian white noise, ${D.lbs.nsim} samples each`, yt: 'rejections of a true H<sub>0</sub> (%)', xt: 'T, m', x: { type: 'category' } },
    { r: 0, c: 1, t: `T = ${Tn}: rejections for every m`, xt: 'number of lags m', x: { dtick: 2 } }], { cols: 2, h: 380 });
  const [a, b] = G.ax, labs = cases.map(([t, mm]) => `${t}, ${mm}`), ms = lags(20);
  const tr = [{ x: labs, y: cases.map(([t, mm]) => 100 * Rr[t][al].bp[mm - 1]), type: 'bar', name: 'Box-Pierce Q', marker: { color: K.blue }, offsetgroup: 0, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'T, m = %{x}: %{y:.1f}%<extra></extra>' },
    { x: labs, y: cases.map(([t, mm]) => 100 * Rr[t][al].lb[mm - 1]), type: 'bar', name: 'Ljung-Box Q*', marker: { color: K.orange }, offsetgroup: 1, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'T, m = %{x}: %{y:.1f}%<extra></extra>' },
    { x: ms, y: Rr[Tn][al].bp.map(v => 100 * v), mode: 'lines+markers', marker: { size: 5, color: K.blue }, line: { color: K.blue, width: 1.5 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hovertemplate: 'Box-Pierce, m = %{x}: %{y:.1f}%<extra></extra>' },
    { x: ms, y: Rr[Tn][al].lb.map(v => 100 * v), mode: 'lines+markers', marker: { size: 5, color: K.orange, symbol: 'square' }, line: { color: K.orange, width: 1.5 }, showlegend: false, xaxis: b.xa, yaxis: b.ya, hovertemplate: 'Ljung-Box, m = %{x}: %{y:.1f}%<extra></extra>' },
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
  const cells = S.map(([t], j) => ({ r: 0, c: j, t, x: { type: 'date', dtick: 'M120', tickformat: '%Y' } })).concat(S.map((_, j) => ({ r: 1, c: j, t: 'ACF', xt: 'lag (quarters)', y: { range: [-0.6, 1] } })));
  const G = grid('gt-chart', cells, { cols: 3, h: 440, legend: false }), tr = [], A = [];
  S.forEach(([, dd, x, c], j) => { tr.push({ x: dd, y: x, mode: 'lines', line: { color: c, width: 1.2 }, xaxis: G.ax[j].xa, yaxis: G.ax[j].ya, showlegend: false, hovertemplate: '%{x|%Y-%m}: %{y:.3f}<extra></extra>' });
    const a = acf(x, 16); A.push(a); acfPanel(tr, G.lay, G.ax[3 + j], a, x.length, c); });
  plot('gt-chart', tr, G.lay);
  const res = [T.gt1(nf(A[0][0], 2), nf(A[0][7], 2)), T.gt2(nf(A[1][3], 2), nf(A[1][1], 2), nf(sdev(d1), 1)),
    T.gt3(k, nf(mean(dk), 2), nf(sdev(dk), 2), nf(A[2][0], 2), nf(A[2][3], 2), nf(dk[dk.length - 1], 1))];
  if (k === 4) res.push(T.gt4(nf(dk[d.slice(k).indexOf('2020-04-01')], 1)));
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
  const G = grid('bc-chart', [{ r: 0, c: 0, t: 'Nominal GDP (bn EUR): Y<sub>t</sub>', x: { type: 'date' } }, { r: 0, c: 1, t: `Box-Cox, λ = ${l.toFixed(2)}`, x: { type: 'date' } },
    { r: 0, c: 2, t: "Guerrero's criterion against λ", xt: 'λ' }], { cols: 3, h: 340, legend: false });
  const [a, b, c] = G.ax;
  plot('bc-chart', [{ x: d, y: y.map(v => v / 1000), mode: 'lines', line: { color: K.blue, width: 1.3 }, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x|%Y-%m}: %{y:.2f}<extra></extra>' },
    { x: d, y: w, mode: 'lines', line: { color: K.green, width: 1.3 }, xaxis: b.xa, yaxis: b.ya, hovertemplate: '%{x|%Y-%m}: %{y:.4g}<extra></extra>' },
    { x: bcGrid.g, y: bcGrid.cv, mode: 'lines', line: { color: K.red, width: 1.6 }, xaxis: c.xa, yaxis: c.ya, hovertemplate: 'λ = %{x:.2f}: %{y:.3f}<extra></extra>' },
    { x: [l], y: [guerrero(y, l)], mode: 'markers', marker: { color: K.orange, size: 10 }, xaxis: c.xa, yaxis: c.ya, hoverinfo: 'skip' }], Object.assign(G.lay, { shapes: [vl(c, lh, K.navy, 'dash', 1)] }));
  const yrs = {}; d.forEach((s, j) => { const k = s.slice(0, 4); (yrs[k] = yrs[k] || []).push(w[j]); });
  const amp = Object.values(yrs).filter(v => v.length === 4).map(v => Math.max(...v) - Math.min(...v)), a0 = mean(amp.slice(0, 5)), a1 = mean(amp.slice(-5));
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
  plot('tb-chart', tr, G.lay);
  const q0 = lbq(N_.y, 10), q1 = lbq(res, 10);
  let pk = 6; for (let h = 6; h <= 15; h++) if (rs[h - 1] > rs[pk - 1]) pk = h;
  let mn = 1; for (let h = 1; h <= 10; h++) if (rs[h - 1] < rs[mn - 1]) mn = h;
  out('tb-out', [T.tb1(B, nf(m0, 0), nf(m1, 0)), T.tb2(nf(rn[0], 2), nf(q0.q, 1), pe(q0.p)), T.tb3(nf(rr[0], 2), nf(q1.q, 1), pe(q1.p)),
    T.tb4(nf(rs[mn - 1], 2), mn, nf(rs[pk - 1], 2), pk)]);
}

// ================================================================= wiring
const ALL = [blkFour, blkEns, blkCounter, blkNonstat, blkWN, blkRW, blkSpot, blkSP, blkWold, blkErgo, blkBart, blkACFm, blkACFP, blkBetACF,
  blkLBsize, blkLBstats, blkGDP, blkBoxCox, blkOverdiff, blkTextbook];
function init() {
  const has = id => !!$(id), on = (ids, f) => ids.forEach(id => $(id).addEventListener('input', f));
  const players = {}, P = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  on(['fs-y0', 'fs-len'], blkFour); P('fs-play', 'fs-y0', range(1995, 2021, 1), 450);
  $('fs-play').addEventListener('click', () => { if (+$('fs-len').value >= 32) $('fs-len').value = '5'; }, true);
  on(['en-phi', 'en-t'], blkEns); P('en-play', 'en-phi', range(0, 0.99, 0.03), 300);
  on(['ce-k'], blkCounter); P('ce-play', 'ce-k', range(1, 60, 1), 200);
  on(['ns-b', 'ns-s', 'ns-d'], blkNonstat); P('ns-play', 'ns-b', range(0, 0.1, 0.005), 300);
  on(['wn-a'], blkWN); P('wn-play', 'wn-a', range(0, 0.14, 0.01), 600);
  on(['rw-t', 'rw-c'], blkRW); P('rw-play', 'rw-t', range(10, 250, 10), 250);
  on(['sr-n', 'sr-v'], blkSpot); P('sr-play', 'sr-n', range(100, 500, 50), 600);
  on(['sp-y0', 'sp-d'], blkSP); P('sp-play', 'sp-y0', range(2000, 2024, 1), 500);
  on(['wd-a', 'wd-b', 'wd-th'], blkWold); P('wd-play', 'wd-a', range(0, 1, 0.05), 350);
  on(['eg-phi', 'eg-z'], blkErgo); P('eg-play', 'eg-phi', range(0, 0.99, 0.03), 350);
  on(['bt-m', 'bt-T'], blkBart); P('bt-play', 'bt-m', range(1, 20, 1), 450);
  on(['am-phi', 'am-th', 'am-T'], blkACFm); P('am-play', 'am-T', range(50, 500, 25), 350);
  on(['ap-a', 'ap-b', 'ap-c'], blkACFP); P('ap-play', 'ap-b', range(-0.4, 0.45, 0.05), 450);
  on(['ba-s', 'ba-y0'], blkBetACF); P('ba-play', 'ba-y0', range(2000, 2024, 1), 500);
  on(['ls-m', 'ls-T', 'ls-a'], blkLBsize); P('ls-play', 'ls-m', range(1, 20, 1), 400);
  on(['lq-m', 'lq-y0'], blkLBstats); P('lq-play', 'lq-y0', range(1995, 2025, 1), 450);
  on(['gt-k'], blkGDP); P('gt-play', 'gt-k', range(1, 8, 1), 900);
  on(['bc-l'], blkBoxCox); P('bc-play', 'bc-l', range(-1, 2, 0.1), 300);
  on(['od-phi'], blkOverdiff); P('od-play', 'od-phi', range(0, 1, 0.05), 400);
  on(['tb-y'], blkTextbook); P('tb-play', 'tb-y', range(1875, 1965, 1), 150);
  const PFX = { 'four-series': 'fs', 'ensemble': 'en', 'counterexample': 'ce', 'nonstationary': 'ns', 'white-noise': 'wn', 'random-walk': 'rw',
    'spot-real': 'sr', 'sp500-diff': 'sp', 'wold': 'wd', 'ergodicity': 'eg', 'bartlett': 'bt', 'acf-models': 'am', 'acf-pacf': 'ap',
    'bet-acf': 'ba', 'lb-size': 'ls', 'lb-stats': 'lq', 'gdp-transform': 'gt', 'boxcox': 'bc', 'overdiff': 'od', 'textbook': 'tb' };
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if (has(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: { out: true } }, sp)); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  ALL.forEach(f => f());
  let w = window.innerWidth, tm = null;
  window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(() => { if ((window.innerWidth < 720) !== (w < 720) || Math.abs(window.innerWidth - w) > 200) { w = window.innerWidth; ALL.forEach(f => f()); } }, 250); });
}
window.CH1 = { chi2sf, chi2ppf, lbq, garch1, ar1, guerrero, boxcox, ceOdd, btSim, R, RET, LQ, days };
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
