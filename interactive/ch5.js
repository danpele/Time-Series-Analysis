// Chapter 5 interactive page (conditional volatility: ARCH and GARCH): one block per chart slide, computed in the browser from ch5_data.js.
(function () {
'use strict';
const D = window.CH5_DATA;
const { sum, solve, acf, ljungBox, normPpf, quantile, mean, nf, $, AX, layout, plot, out, Player, range, predict, exitTicket } = window.TSAI;
const T = window.CH5_TEXT;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of generate_all_charts.py / tsa_style.py
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', orange: '#E67E22', purple: '#8E44AD', teal: '#17A2B8', navy: '#1F2A44' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.purple]: '142,68,173', [K.orange]: '230,126,34',
  [K.amber]: '181,133,63', [K.teal]: '23,162,184', [K.navy]: '31,42,68' };
const AS = ['sp500', 'bet', 'eurron', 'btc'];
const NAME = { sp500: 'S&P 500', bet: 'BET', eurron: 'EUR/RON', btc: 'Bitcoin' };
const COL = { sp500: K.blue, bet: K.red, eurron: K.green, btc: K.amber };
const EPIS = { 2008: ['2008-09-01', '2009-03-31'], 2020: ['2020-02-15', '2020-05-31'] };

// ---------------------------------------------------------------- small helpers
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
const pe = p => (p < 0.001 ? 'p < ' + nf(0.001, 3) : 'p = ' + nf(p, 3));
const lags = n => Array.from({ length: n }, (_, i) => i + 1);
const pvar = x => { const m = mean(x); return sum(x.map(v => (v - m) ** 2)) / x.length; };
const sd1 = x => { const m = mean(x); return Math.sqrt(sum(x.map(v => (v - m) ** 2)) / (x.length - 1)); };
const kurt = x => { const m = mean(x); let m2 = 0, m4 = 0; x.forEach(v => { const d = (v - m) ** 2; m2 += d; m4 += d * d; }); m2 /= x.length; m4 /= x.length; return m4 / (m2 * m2); };
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
const tCdf = (t, df) => (t < 0 ? tsf2(t, df) / 2 : 1 - tsf2(t, df) / 2);
function tPpf(p, df) {            // quantile of the Student t distribution (bisection on the distribution function)
  if (p === 0.5) return 0;
  let lo = -1e4, hi = 1e4;
  for (let i = 0; i < 200 && hi - lo > 1e-12 * Math.max(1, Math.abs(lo)); i++) { const m = (lo + hi) / 2; if (tCdf(m, df) < p) lo = m; else hi = m; }
  return (lo + hi) / 2;
}
const stdT = (p, nu) => tPpf(p, nu) * Math.sqrt((nu - 2) / nu);       // quantile of the standardised t (variance 1)
function lsq(X, y) {              // least squares coefficients and residuals (columns in X)
  const G = X.map(a => X.map(b => sum(a.map((v, i) => v * b[i])))), b = solve(G, X.map(a => sum(a.map((v, i) => v * y[i]))));
  return { b, res: y.map((v, i) => v - X.reduce((s, c, j) => s + b[j] * c[i], 0)) };
}
function archLm(x, q = 5) {       // Engle's ARCH-LM: e_t^2 on a constant and q lags of e^2 (e = x - mean); LM = n R^2
  const m = mean(x), e2 = x.map(v => (v - m) ** 2), y = e2.slice(q), n = y.length;
  const X = [y.map(() => 1)].concat(lags(q).map(j => e2.slice(q - j, e2.length - j)));
  const { res } = lsq(X, y), my = mean(y), r2 = 1 - sum(res.map(u => u * u)) / sum(y.map(v => (v - my) ** 2));
  return { lm: n * r2, r2, n, p: chi2sf(n * r2, q) };
}
const lb = (x, h = 10, dof = 0) => { const q = ljungBox(x, h).q; return { q, p: chi2sf(q, h - dof) }; };
function dm(la, lb_, L = 5) {     // Diebold-Mariano: mean of d = la - lb, Newey-West (Bartlett, L lags) standard error with the n/(n-1) correction
  const d = la.map((v, i) => v - lb_[i]), n = d.length, m = mean(d), u = d.map(v => v - m);
  let s = sum(u.map(v => v * v));
  for (let l = 1; l <= L; l++) { let c = 0; for (let t = l; t < n; t++) c += u[t] * u[t - l]; s += 2 * (1 - l / (L + 1)) * c; }
  const t = m / Math.sqrt(s / n / n * n / (n - 1));
  return { mean: m, t, p: 2 * (1 - normCdf(Math.abs(t))) };
}
const normCdf = window.TSAI.normCdf;
const DAY = 864e5;
const d2t = s => Date.parse(s.slice(0, 10) + 'T00:00:00Z');
const t2d = t => new Date(t).toISOString().slice(0, 10);

// ---------------------------------------------------------------- panels: one Plotly figure with several axes; stacked on narrow screens
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
const hband = (a, y0, y1, col, al) => ({ type: 'rect', xref: a.xd, yref: a.ya, x0: 0, x1: 1, y0, y1, fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, layer: 'below' });
const leg = (name, line, marker) => Object.assign({ x: [null], y: [null], name, hoverinfo: 'skip' }, line ? { mode: 'lines', line } : marker.fill ? { mode: 'markers', marker: { color: marker.fill, symbol: 'square', size: 11 } } : { mode: 'markers', marker });
const ln = (x, y, a, color, name, o) => Object.assign({ x, y, mode: 'lines', line: { color, width: 1.3 }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya }, o || {});
const EPI_LEG = () => [leg('Sep 2008 - Mar 2009 (global financial crisis)', null, { fill: `rgba(${RGB[K.purple]},0.3)` }), leg('Feb - May 2020 (COVID-19)', null, { fill: `rgba(${RGB[K.orange]},0.3)` })];
const episodes = (a, d0, al) => Object.entries(EPIS).filter(([, [s]]) => s > d0).map(([e, [s, f]]) => rect(a, s, f, e === '2008' ? K.purple : K.orange, al));

// ---------------------------------------------------------------- data of the chapter
const S = {};
AS.forEach(k => {
  const s = D.s[k], d = [s.d0]; let t = d2t(s.d0);
  s.gap.forEach(gp => { t += gp * DAY; d.push(t2d(t)); });
  S[k] = { d, r: s.r, g: s.g, ppy: s.ppy, ann: Math.sqrt(s.ppy) };
});
const idx = (k, day) => { const d = S[k].d; let lo = 0, hi = d.length; while (lo < hi) { const m = (lo + hi) >> 1; if (d[m] < day) lo = m + 1; else hi = m; } return lo; };   // first index with date >= day
function s2rec(eps, om, a, b, s0) { const s = new Array(eps.length); s[0] = s0; for (let t = 1; t < eps.length; t++) s[t] = om + a * eps[t - 1] ** 2 + b * s[t - 1]; return s; }
const GS = {};                    // GARCH(1,1)-t conditional variances of the lecture (%^2)
const garch = k => GS[k] || (GS[k] = s2rec(S[k].r.map(v => v - S[k].g.mu), S[k].g.omega, S[k].g.alpha, S[k].g.beta, S[k].g.s0));
const zt = k => { const s = garch(k), g = S[k].g; return S[k].r.map((v, i) => (v - g.mu) / Math.sqrt(s[i])); };
function ewma(r, lam = 0.94, init = 250) { const h = new Array(r.length); h[0] = mean(r.slice(0, init).map(v => v * v)); for (let t = 1; t < r.length; t++) h[t] = lam * h[t - 1] + (1 - lam) * r[t - 1] ** 2; return h; }
const EW = {};
const ewma94 = k => EW[k] || (EW[k] = ewma(S[k].r));
function slice(k, a, b) { const i0 = idx(k, a), i1 = idx(k, b + 'z'); return [i0, i1]; }   // [first index >= a, first index > b)
const peakIn = (d, v, a, b) => { let m = -Infinity, dm = ''; d.forEach((x, i) => { if (x >= a && x <= b && v[i] > m) { m = v[i]; dm = x; } }); return [m, dm]; };

// ================================================================= 1. four markets, one pattern
function sdYear(k, y) { const [i0, i1] = slice(k, `${y}-01-01`, `${y}-12-31`); return i1 - i0 > 20 ? sd1(S[k].r.slice(i0, i1)) : NaN; }
function sdEpi(k, e) { const [a, b] = EPIS[e]; if (a < S[k].d[0]) return NaN; const [i0, i1] = slice(k, a, b); return sd1(S[k].r.slice(i0, i1)); }
function blkReturns() {
  const y = +$('re-y').value; set('re-y', y);
  const G = grid('re-chart', AS.map((k, i) => ({ r: i >> 1, c: i & 1, t: `${NAME[k]}: daily log return (%)`, yt: '%', x: { type: 'date' } })),
    { cols: 2, h: 470, gyPx: 52, legendRows: 2 });
  const tr = [];
  AS.forEach((k, i) => {
    const a = G.ax[i];
    tr.push(ln(S[k].d, S[k].r, a, COL[k], null, { line: { color: COL[k], width: 0.6 }, hovertemplate: '%{x}: %{y:.2f}%<extra>' + NAME[k] + '</extra>' }));
    G.lay.shapes.push(...episodes(a, S[k].d[0], 0.18));
    if (`${y}-12-31` >= S[k].d[0]) G.lay.shapes.push(rect(a, `${y}-01-01` < S[k].d[0] ? S[k].d[0] : `${y}-01-01`, `${y}-12-31`, K.teal, 0.28));
    // +/- 2 daily standard deviations of the selected year, drawn over the whole sample: how far the other years leave that band
    const s = sdYear(k, y), x2 = [S[k].d[0], S[k].d[S[k].d.length - 1]], b2 = isNaN(s) ? [null, null] : [2 * s, 2 * s];
    [1, -1].forEach(sg => tr.push(ln(x2, b2.map(v => (v === null ? null : sg * v)), a, K.navy, null,
      { line: { color: K.navy, width: 1.2, dash: 'dash' }, hovertemplate: (sg > 0 ? '+' : '-') + '2 sd of ' + y + ': %{y:.2f}%<extra>' + NAME[k] + '</extra>' })));
  });
  tr.push(...EPI_LEG(), leg(`selected year: ${y}`, null, { fill: `rgba(${RGB[K.teal]},0.4)` }), leg(`+/- 2 daily standard deviations of ${y}`, { color: K.navy, dash: 'dash', width: 1.2 }));
  plot('re-chart', tr, G.lay);
  const rows = AS.map(k => {
    const s = sdYear(k, y), e8 = sdEpi(k, '2008'), e0 = sdEpi(k, '2020');
    return T.re1(NAME[k], y, isNaN(s) ? null : nf(s, 2), isNaN(e8) ? null : nf(e8, 2), nf(e0, 2), isNaN(s) || isNaN(e8) ? null : nf(e8 / s, 1), isNaN(s) ? null : nf(e0 / s, 1));
  });
  out('re-out', rows);
}

// ================================================================= 2. the sign is unpredictable, the size is not
function acfSq(k, y0, f) {
  const i0 = idx(k, `${y0}-01-01`), x = S[k].r.slice(i0), x2 = x.map(v => (f === 'abs' ? Math.abs(v) : v * v));
  const a1 = acf(x, 50), a2 = acf(x2, 50), b = 1.96 / Math.sqrt(x.length);
  return { a1, a2, b, n: x.length, d0: S[k].d[i0], o1: a1.filter(v => Math.abs(v) > b).length, o2: a2.filter(v => Math.abs(v) > b).length };
}
function blkAcfSq() {
  const f = $('aq-f').value, y0 = +$('aq-y').value; set('aq-y', y0);
  const G = grid('aq-chart', AS.map((k, i) => ({ r: i >> 1, c: i & 1, t: NAME[k], xt: i > 1 ? 'lag k (days)' : '', yt: (i & 1) ? '' : 'ACF' })), { cols: 2, h: 470, gyPx: 56 });
  const tr = [], rows = [];
  AS.forEach((k, i) => {
    const a = G.ax[i], s = acfSq(k, y0, f), j = lags(50);
    tr.push({ x: j.map(v => v - 0.2), y: s.a1, type: 'bar', width: 0.4, marker: { color: K.blue }, name: 'returns r(t)', showlegend: i === 0, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'lag %{text}: %{y:.3f}<extra>r(t)</extra>', text: j, textposition: 'none' },
      { x: j.map(v => v + 0.2), y: s.a2, type: 'bar', width: 0.4, marker: { color: K.red }, name: f === 'abs' ? 'absolute returns |r(t)|' : 'squared returns r(t)^2', showlegend: i === 0, xaxis: a.xa, yaxis: a.ya,
        hovertemplate: 'lag %{text}: %{y:.3f}<extra>' + (f === 'abs' ? '|r(t)|' : 'r(t)^2') + '</extra>', text: j, textposition: 'none' });
    G.lay.shapes.push(hband(a, -s.b, s.b, K.teal, 0.25), hl(a, 0, K.navy, 'solid', 0.6));
    rows.push(T.aq1(NAME[k], s.d0, s.n, f, nf(s.a1[0], 2), nf(s.a2[0], 2), nf(s.a2[49], 3), s.o2, s.o1, nf(s.b, 3)));
  });
  tr.push(leg('95% band, +/- 1.96/sqrt(T)', null, { fill: `rgba(${RGB[K.teal]},0.35)` }));
  G.lay.barmode = 'overlay';
  plot('aq-chart', tr, G.lay);
  out('aq-out', rows);
}

// ================================================================= 3. the mean model leaves ARCH effects (BET)
function arOls(r, p) {            // AR(p) for the mean by OLS on r_p, ..., r_{n-1}; residuals
  if (p === 0) { const m = mean(r); return { res: r.map(v => v - m), b: [m] }; }
  const y = r.slice(p), X = [y.map(() => 1)].concat(lags(p).map(j => r.slice(p - j, r.length - j)));
  return lsq(X, y);
}
let ZAG = null;
function zArGarch() {             // standardised residuals of AR(1)-GARCH(1,1)-t (BET) and their conditional volatility, t = 1..n-1
  if (ZAG) return ZAG;
  const r = S.bet.r, g = D.ag, e = r.slice(1).map((v, i) => v - g.c - g.phi * r[i]), s = s2rec(e, g.omega, g.alpha, g.beta, g.s1);
  return (ZAG = { e, s, z: e.map((v, i) => v / Math.sqrt(s[i])) });
}
const ZP = {};
function zArP(p) {                // AR(p) mean residuals standardised by the GARCH(1,1) variance with the lecture's parameters (re-estimated only for p = 1)
  if (p === 1) return zArGarch();
  if (ZP[p]) return ZP[p];
  const g = D.ag, e = arOls(S.bet.r, p).res, s = s2rec(e, g.omega, g.alpha, g.beta, g.s1);
  return (ZP[p] = { e, s, z: e.map((v, i) => v / Math.sqrt(s[i])) });
}
function arStats(p) {
  const { res } = arOls(S.bet.r, p), e2 = res.map(v => v * v);
  return { n: res.length, lbe: lb(res, 10, p), lbe2: lb(e2), r1: acf(e2, 1)[0], lm: archLm(res), a: acf(res, 30), a2: acf(e2, 30) };
}
function blkArma() {
  const p = +$('ar-p').value; set('ar-p', p);
  const L = arStats(p), z = zArP(p).z, z2 = z.map(v => v * v), b = 1.96 / Math.sqrt(L.n), j = lags(30);
  const G = grid('ar-chart', [{ r: 0, c: 0, t: p ? `AR(${p}) with constant variance: residuals e(t)` : 'Constant mean with constant variance: residuals e(t)', xt: 'lag k (days)', yt: 'autocorrelation' },
    { r: 0, c: 1, t: `${p ? `AR(${p})` : 'Constant mean'}-GARCH(1,1)-t: standardised residuals z(t)`, xt: 'lag k (days)', y: { matches: 'y' } }], { cols: 2, h: 360 });
  const tr = [];
  [[L.a, L.a2], [acf(z, 30), acf(z2, 30)]].forEach(([u, v], i) => {
    const a = G.ax[i];
    tr.push({ x: j.map(x => x - 0.2), y: u, type: 'bar', width: 0.4, marker: { color: K.blue }, name: 'residuals', showlegend: i === 0, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'lag %{text}: %{y:.3f}<extra></extra>', text: j, textposition: 'none' },
      { x: j.map(x => x + 0.2), y: v, type: 'bar', width: 0.4, marker: { color: K.red }, name: 'squared residuals', showlegend: i === 0, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'lag %{text}: %{y:.3f}<extra></extra>', text: j, textposition: 'none' });
    G.lay.shapes.push(hband(a, -b, b, K.teal, 0.25), hl(a, 0, K.navy, 'solid', 0.6));
  });
  tr.push(leg('95% band, +/- 1.96/sqrt(T)', null, { fill: `rgba(${RGB[K.teal]},0.35)` }));
  G.lay.barmode = 'overlay';
  plot('ar-chart', tr, G.lay);
  const lz = lb(z, 10, p), lz2 = lb(z2), mz = archLm(z);
  out('ar-out', [T.ar1(p, nf(L.lbe.q, 1), 10 - p, pe(L.lbe.p), nf(L.r1, 2), nf(L.lbe2.q, 0), pe(L.lbe2.p), nf(L.lm.r2, 4), L.lm.n, nf(L.lm.lm, 1), pe(L.lm.p)),
    T.ar2(p, nf(acf(z2, 1)[0], 3), nf(lz2.q, 1), pe(lz2.p), nf(mz.lm, 1), pe(mz.p), nf(lz.q, 1), pe(lz.p))]);
}

// ================================================================= 4. simulated paths
function simPath(z, om, a, b, burn) {
  const m = z.length, e = new Array(m); let s2 = om / (1 - a - b);
  e[0] = Math.sqrt(s2) * z[0];
  for (let t = 1; t < m; t++) { s2 = om + a * e[t - 1] ** 2 + b * s2; e[t] = Math.sqrt(s2) * z[t]; }
  return e.slice(burn);
}
const kArch = a => (3 * a * a < 1 ? 3 * (1 - a * a) / (1 - 3 * a * a) : Infinity);
const kGarch = (a, b) => { const p = a + b, d = 1 - p * p - 2 * a * a; return d > 0 ? 3 * (1 - p * p) / d : Infinity; };
function simStats(x) { return { sd: Math.sqrt(pvar(x)), k: kurt(x), mx: Math.max(...x.map(Math.abs)), r1: acf(x.map(v => v * v), 1)[0] }; }
function blkSim() {
  const a1 = +$('si-a1').value, a = +$('si-a').value; let b = +$('si-b').value;
  if (a + b > 0.99) { b = Math.round((0.99 - a) * 100) / 100; $('si-b').value = b; }
  set('si-a1', nf(a1, 2)); set('si-a', nf(a, 2)); set('si-b', nf(b, 2));
  const S0 = D.sim, P = [[`ARCH(1): omega = ${+(1 - a1).toFixed(2)}, alpha = ${a1}`, simPath(S0.z1, 1 - a1, a1, 0, S0.burn), K.red],
    [`GARCH(1,1): omega = ${+(1 - a - b).toFixed(2)}, alpha = ${a.toFixed(2)}, beta = ${b.toFixed(2)}`, simPath(S0.z2, 1 - a - b, a, b, S0.burn), K.green]];
  const lim = Math.max(...S0.iid.map(Math.abs), ...P.map(p => Math.max(...p[1].map(Math.abs)))) * 1.05, t = lags(1000).map(v => v - 1);
  const G = grid('si-chart', P.map((p, i) => ({ r: i, c: 0, xt: i === 1 ? 'time t' : '', y: { range: [-lim, lim] }, x: { showticklabels: i === 1 } })), { cols: 1, h: 470, gyPx: 22, rowPx: 170, legendRows: 3 });
  const tr = [];
  P.forEach((p, i) => tr.push(ln(t, S0.iid, G.ax[i], '#9AA0A6', i ? null : 'i.i.d. Normal(0, 1), same variance (grey, both panels)', { line: { color: '#9AA0A6', width: 0.8 }, hovertemplate: 't = %{x}: %{y:.2f}<extra>i.i.d.</extra>' }),
    ln(t, p[1], G.ax[i], p[2], p[0], { line: { color: p[2], width: 0.8 }, hovertemplate: 't = %{x}: %{y:.2f}<extra></extra>' })));
  plot('si-chart', tr, G.lay);
  const st = [S0.iid].concat(P.map(p => p[1])).map(simStats);
  const kt = v => (isFinite(v) ? nf(v, 2) : '∞');
  out('si-out', [T.si1(nf(st[0].k, 2), nf(st[0].r1, 3)), T.si2(nf(a1, 2), nf(st[1].k, 2), kt(kArch(a1)), nf(st[1].r1, 3), nf(st[1].mx, 2)),
    T.si3(nf(a, 2), nf(b, 2), nf(st[2].k, 2), kt(kGarch(a, b)), nf(st[2].r1, 3), b > 0 ? nf(Math.log(0.5) / Math.log(a + b), 1) : null)]);
}

// ================================================================= 5. GARCH, EWMA and a rolling window in 2020 (S&P 500)
function ewStats(lam, w) {
  const k = 'sp500', r = S[k].r, d = S[k].d, an = S[k].ann, g = garch(k).map(v => Math.sqrt(v) * an), e = ewma(r, lam).map(v => Math.sqrt(v) * an);
  const roll = r.map((_, i) => (i >= w - 1 ? sd1(r.slice(i - w + 1, i + 1)) * an : NaN));
  const [i0, i1] = slice(k, '2019-10-01', '2021-03-31'), jun = idx(k, '2020-07-01') - 1;
  const sl = v => v.slice(i0, i1), dd = d.slice(i0, i1);
  const pk = v => peakIn(dd, sl(v), '2019-10-01', '2021-03-31');
  return { dd, g: sl(g), e: sl(e), w: sl(roll), pg: pk(g), pe: pk(e), pw: pk(roll), jg: g[jun], je: e[jun], jw: roll[jun] };
}
function blkEwma() {
  const lam = +$('ew-l').value, w = +$('ew-w').value; set('ew-l', nf(lam, 2)); set('ew-w', w);
  const s = ewStats(lam, w);
  const G = grid('ew-chart', [{ r: 0, c: 0, yt: '% per year', x: { type: 'date' } }], { cols: 1, h: 380 });
  const a = G.ax[0];
  plot('ew-chart', [ln(s.dd, s.g, a, K.blue, 'GARCH(1,1)-t', { line: { color: K.blue, width: 1.8 }, hovertemplate: '%{x}: %{y:.1f}%<extra>GARCH</extra>' }),
    ln(s.dd, s.e, a, K.red, `EWMA, lambda = ${lam.toFixed(2)}`, { line: { color: K.red, width: 1.5, dash: 'dash' }, hovertemplate: '%{x}: %{y:.1f}%<extra>EWMA</extra>' }),
    ln(s.dd, s.w, a, K.green, `${w}-day rolling standard deviation`, { line: { color: K.green, width: 1.5 }, hovertemplate: '%{x}: %{y:.1f}%<extra>window</extra>' })], G.lay);
  out('ew-out', [T.ew1(nf(s.pg[0], 0), T.dmy(s.pg[1]), nf(s.pe[0], 0), T.dmy(s.pe[1]), w, nf(s.pw[0], 0), T.dmy(s.pw[1])),
    T.ew2(nf(s.jg, 0), nf(s.je, 0), nf(s.jw, 0)), T.ew3(nf(lam, 2), nf(1 - lam, 2), nf(Math.log(0.5) / Math.log(lam), 1), nf(S.sp500.g.alpha, 3))]);
}

// ================================================================= 6. the likelihood of ARCH(1)
const arch1ll = (a, x) => { let l = 0; for (let t = 1; t < x.length; t++) { const s2 = (1 - a) + a * x[t - 1] ** 2; l += -0.5 * Math.log(s2) - 0.5 * x[t] ** 2 / s2; } return l; };
function argmax1(f, lo, hi) {     // golden-section search for the maximum of a unimodal function on [lo, hi]
  const g = (Math.sqrt(5) - 1) / 2; let a = lo, b = hi, c = b - g * (b - a), d = a + g * (b - a), fc = f(c), fd = f(d);
  for (let i = 0; i < 80; i++) { if (fc > fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); } else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); } }
  return (a + b) / 2;
}
function lik1(n, a0) {
  const x = simPath(D.sim.lik1[n], 1 - a0, a0, 0, D.sim.burn), ah = argmax1(a => arch1ll(a, x), 0.001, 0.999), h = 1e-4;
  const d2 = (arch1ll(ah + h, x) - 2 * arch1ll(ah, x) + arch1ll(ah - h, x)) / h / h;
  const gr = range(0.05, 0.95, 0.005), ll = gr.map(a => arch1ll(a, x)), m = Math.max(...ll);
  return { ah, se: 1 / Math.sqrt(-d2), gr, ll: ll.map(v => v - m) };
}
function blkLik1() {
  const a0 = +$('l1-a').value, n2 = $('l1-n').value; set('l1-a', nf(a0, 2));
  const L = [['100', lik1('100', a0), K.blue], [n2, lik1(n2, a0), K.red]];
  const G = grid('l1-chart', [{ r: 0, c: 0, xt: 'alpha', yt: 'log-likelihood minus maximum', y: { range: [-40, 2] }, x: { range: [0.05, 0.95] } }], { cols: 1, h: 380, legendRows: 3 });
  const a = G.ax[0], tr = [];
  L.forEach(([n, s, c]) => {
    tr.push(ln(s.gr, s.ll, a, c, `n = ${n}: log-likelihood minus its maximum`, { line: { color: c, width: 2 }, hovertemplate: 'alpha = %{x:.3f}: %{y:.1f}<extra>n = ' + n + '</extra>' }),
      leg(`n = ${n}: maximum at alpha = ${s.ah.toFixed(3)}`, { color: c, dash: 'dash', width: 1.2 }));
    G.lay.shapes.push(vl(a, s.ah, c, 'dash', 1.2));
  });
  tr.push(leg(`true alpha = ${a0}`, { color: K.navy, dash: 'dot', width: 1.4 }));
  G.lay.shapes.push(vl(a, a0, K.navy, 'dot', 1.4));
  plot('l1-chart', tr, G.lay);
  out('l1-out', L.map(([n, s]) => T.l11(n, nf(s.ah, 3), nf(s.se, 3), nf(s.ah - 1.96 * s.se, 2), nf(s.ah + 1.96 * s.se, 2))).concat([T.l12(nf(a0, 2), nf(L[0][1].se / L[1][1].se, 1), nf(Math.sqrt(+n2 / 100), 1))]));
}

// ================================================================= 7. the likelihood of GARCH(1,1)
const LG_A = range(0.01, 0.30, 0.29 / 58), LG_B = range(0.50, 0.97, 0.47 / 47), LG_LEV = [40, 20, 10, 5, 3, 2, 1];
const LG_COL = [K.purple, K.blue, K.teal, K.green, K.amber, K.orange, K.red];
function likG(n) {
  const x = simPath(D.sim.z7, 0.1, 0.1, 0.8, D.sim.burn).slice(0, n), s2bar = pvar(x), x2 = x.map(v => v * v);
  let best = -Infinity, bi = 0, bj = 0;
  const L = LG_B.map((b, i) => LG_A.map((a, j) => {
    if (a + b >= 0.995) return null;
    const om = (1 - a - b) * s2bar; let s2 = s2bar, l = 0;
    for (let t = 1; t < n; t++) { s2 = om + a * x2[t - 1] + b * s2; l += -0.5 * Math.log(s2) - 0.5 * x2[t] / s2; }
    if (l > best) { best = l; bi = i; bj = j; }
    return l;
  }));
  return { Z: L.map(r => r.map(v => (v === null ? null : v - best))), a: LG_A[bj], b: LG_B[bi] };
}
const LGC = {};
const likGc = n => LGC[n] || (LGC[n] = likG(n));
function blkLikG() {
  const n = +$('lg-n').value; set('lg-n', n);
  const P = [[500, likGc(500)], [n, likGc(n)]], path = range(250, n, 250).map(m => [m, likGc(m)]);
  const G = grid('lg-chart', P.map(([m, s], i) => ({ r: 0, c: i, t: `n = ${m}: maximum at alpha = ${s.a.toFixed(3)}, beta = ${s.b.toFixed(2)}`, xt: 'alpha', yt: i ? '' : 'beta',
    x: { range: [0.01, 0.30] }, y: { range: [0.50, 0.97] } })), { cols: 2, h: 420, legendRows: 2 });
  const tr = [];
  P.forEach(([, s], i) => {
    const a = G.ax[i];
    LG_LEV.forEach((lv, k) => tr.push({ type: 'contour', x: LG_A, y: LG_B, z: s.Z, xaxis: a.xa, yaxis: a.ya, showscale: false, connectgaps: false, hoverinfo: 'skip', showlegend: false,
      contours: { start: -lv, end: -lv, size: 1, coloring: 'lines', showlabels: true, labelfont: { size: 10, color: LG_COL[k] } }, line: { color: LG_COL[k], width: 1.3 }, colorscale: [[0, LG_COL[k]], [1, LG_COL[k]]] }));
    tr.push({ x: [s.a], y: [s.b], mode: 'markers', marker: { color: K.red, size: 9 }, name: 'maximum on the grid', showlegend: i === 0, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'alpha = %{x:.3f}, beta = %{y:.2f}<extra></extra>' },
      { x: [0.1], y: [0.8], mode: 'markers', marker: { color: K.navy, size: 10, symbol: 'x' }, name: 'true values: alpha = 0.1, beta = 0.8', showlegend: i === 0, xaxis: a.xa, yaxis: a.ya, hoverinfo: 'skip' },
      ln(LG_A, LG_A.map(v => 1 - v), a, K.purple, 'alpha + beta = 1 (IGARCH boundary)', { line: { color: K.purple, dash: 'dash', width: 1.3 }, showlegend: i === 0, hoverinfo: 'skip' }));
  });
  // left panel: the path of the grid maximum as n grows from 250 to the n of the right panel, over the n = 500 contours
  tr.push({ x: path.map(([, s]) => s.a), y: path.map(([, s]) => s.b), text: path.map(([m]) => m), mode: 'lines+markers', line: { color: K.navy, width: 1.2 },
    marker: { color: K.navy, size: 6, symbol: 'diamond' }, name: `maximum on the grid for n = 250, 500, …, ${n} (left)`, xaxis: G.ax[0].xa, yaxis: G.ax[0].ya,
    hovertemplate: 'n = %{text}: alpha = %{x:.3f}, beta = %{y:.2f}<extra></extra>' });
  plot('lg-chart', tr, G.lay);
  out('lg-out', P.map(([m, s]) => T.lg1(m, nf(s.a, 3), nf(s.b, 2), nf(s.a + s.b, 3))));
}

// ================================================================= 8. QQ plots of the standardised residuals (S&P 500)
let QQZ = null;
function qqZ() {
  if (QQZ) return QQZ;
  const zn = D.gn, zz = zt('sp500').sort((a, b) => a - b);                  // GARCH-N residuals as exported (sorted); GARCH-t recomputed
  const N = zn.length, keep = [];
  for (let i = 0; i < N; i++) if (i < 150 || i >= N - 150 || i % 8 === 0) keep.push(i);
  return (QQZ = { zn, zt: zz, keep, u: keep.map(i => (i + 0.5) / N) });
}
function blkQQ() {
  const nu = +$('qq-nu').value; set('qq-nu', nf(nu, 2));
  const Q = qqZ(), thN = Q.u.map(normPpf), thT = Q.u.map(u => stdT(u, nu));
  const G = grid('qq-chart', [{ r: 0, c: 0, t: 'GARCH(1,1)-N residuals', xt: 'theoretical quantile', yt: 'sample quantile' }, { r: 0, c: 1, t: 'GARCH(1,1)-t residuals', xt: 'theoretical quantile' }], { cols: 2, h: 420, legendRows: 3 });
  const tr = [];
  // each panel: the residuals against the Normal (grey, fixed) and against the standardised t(nu) of the slider (colour, moves with nu)
  [[Q.zn, K.blue, 'GARCH(1,1)-N'], [Q.zt, K.red, 'GARCH(1,1)-t']].forEach(([z, c, nm], i) => {
    const a = G.ax[i], zs = Q.keep.map(k => z[k]), lo = Math.min(thN[0], thT[0], zs[0]), hi = Math.max(thN[thN.length - 1], thT[thT.length - 1], zs[zs.length - 1]);
    tr.push({ x: thN, y: zs, mode: 'markers', marker: { color: '#9AA0A6', size: 4, opacity: 0.5 }, name: 'residuals against the Normal (grey, both panels)', showlegend: i === 0, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'Normal %{x:.2f}, sample %{y:.2f}<extra>' + nm + '</extra>' },
      { x: thT, y: zs, mode: 'markers', marker: { color: c, size: 4, opacity: 0.65 }, name: `${nm} residuals against the standardised t(${nu.toFixed(1)})`, xaxis: a.xa, yaxis: a.ya, hovertemplate: 't %{x:.2f}, sample %{y:.2f}<extra>' + nm + '</extra>' },
      ln([lo, hi], [lo, hi], a, K.navy, i ? null : '45-degree line', { line: { color: K.navy, dash: 'dash', width: 1 }, hoverinfo: 'skip', showlegend: i === 0 }));
  });
  plot('qq-chart', tr, G.lay);
  const qn = quantile(Q.zn, 0.001), qt = quantile(Q.zt, 0.001), q9n = quantile(Q.zn, 0.999), q9t = quantile(Q.zt, 0.999);
  out('qq-out', [T.qq1(nf(qn, 2), nf(normPpf(0.001), 2), nf(q9n, 2), nf(normPpf(0.999), 2)), T.qq2(nf(nu, 2), nf(qt, 2), nf(stdT(0.001, nu), 2), nf(q9t, 2), nf(stdT(0.999, nu), 2),
    nu > 4 ? nf(3 + 6 / (nu - 4), 1) : null)]);
}

// ================================================================= 9-10. conditional volatility
function volPath(k, c) {          // annualised GARCH(1,1)-t volatility with alpha = c alpha_hat, beta = (alpha_hat + beta_hat) - alpha, omega_hat
  const g = S[k].g, a = c * g.alpha, b = g.alpha + g.beta - a, an = S[k].ann;
  const s = c === 1 ? garch(k) : s2rec(S[k].r.map(v => v - g.mu), g.omega, a, b, g.s0);
  return { a, b, v: s.map(x => Math.sqrt(x) * an) };
}
function volStats(k, c) {
  const p = volPath(k, c), d = S[k].d, v = p.v, srt = [...v].sort((x, y) => x - y), res = { a: p.a, b: p.b, v, med: (srt[(v.length - 1) >> 1] + srt[v.length >> 1]) / 2, last: v[v.length - 1] };
  Object.entries(EPIS).forEach(([e, [x, y]]) => { res['p' + e] = x > d[0] ? peakIn(d, v, x, y) : null; });
  res.max = peakIn(d, v, d[0], d[d.length - 1]);
  res.samp = sd1(S[k].r) * S[k].ann;
  return res;
}
function blkVol(pf, ks) {
  const c = +$(pf + '-c').value; set(pf + '-c', nf(c, 2));
  const G = grid(pf + '-chart', ks.map((k, i) => ({ r: i, c: 0, yt: '% per year', x: { type: 'date' } })), { cols: 1, h: 470, gyPx: 40, legendRows: 3 });
  const tr = [], rows = [], st = {};
  ks.forEach((k, i) => {
    const a = G.ax[i], s = volStats(k, c); st[k] = s;
    tr.push(ln(S[k].d, s.v, a, COL[k], `${NAME[k]}: GARCH(1,1)-t volatility, annualised (%)`, { line: { color: COL[k], width: 1 }, hovertemplate: '%{x}: %{y:.1f}%<extra>' + NAME[k] + '</extra>' }));
    G.lay.shapes.push(...episodes(a, S[k].d[0], 0.15), hl(a, s.samp, K.navy, 'dash', 1));
    const pk = e => (s['p' + e] ? [nf(s['p' + e][0], 0), T.dmy(s['p' + e][1])] : null);
    rows.push(T.vo1(NAME[k], nf(s.a, 3), nf(s.b, 3), pk('2008'), pk('2020'), nf(s.med, 1), nf(s.last, 1), nf(s.samp, 1)));
  });
  tr.push(leg('sample volatility, annualised (%)', { color: K.navy, dash: 'dash', width: 1 }), ...EPI_LEG());
  plot(pf + '-chart', tr, G.lay);
  if (ks.includes('btc')) rows.push(T.vo2(nf(st.btc.med / volStats('sp500', c).med, 1)));
  out(pf + '-out', rows);
}
const blkVol1 = () => blkVol('v1', ['sp500', 'bet']);
const blkVol2 = () => blkVol('v2', ['eurron', 'btc']);

// ================================================================= 11. how long does a volatility shock last?
const IG = 0.9995;
function persRows(dec) {
  return AS.map(k => { const p0 = S[k].g.pers, p = dec === 'x' ? p0 : +p0.toFixed(+dec); return { k, p, ig: p0 >= IG || p >= 1, hl: Math.log(0.5) / Math.log(p) }; });
}
function blkPers() {
  const dec = $('pe-d').value, ph = +$('pe-p').value; set('pe-p', nf(ph, 3));
  const P = persRows(dec), h = range(0, 250, 1);
  const G = grid('pe-chart', [{ r: 0, c: 0, xt: 'days after the shock, h', yt: 'share of the shock left', y: { range: [0, 1.02] } }], { cols: 1, h: 400, legendRows: 3 });
  const a = G.ax[0], fp = p => (dec === 'x' ? p.toFixed(3) : p.toFixed(+dec));
  const tr = P.map(o => ln(h, h.map(x => Math.min(o.p, 1) ** x), a, COL[o.k], o.ig ? `${NAME[o.k]}: alpha + beta = 1 (IGARCH), no half-life` : `${NAME[o.k]}: alpha + beta = ${fp(o.p)}, half-life ${o.hl.toFixed(0)} days`,
    { line: { color: COL[o.k], width: o.k === 'eurron' ? 2.6 : 1.8, dash: o.k === 'btc' ? 'dot' : 'solid' }, hovertemplate: 'h = %{x}: %{y:.3f}<extra>' + NAME[o.k] + '</extra>' }));
  tr.push(ln(h, h.map(x => ph ** x), a, K.purple, `alpha + beta = ${ph.toFixed(3)}, half-life ${(Math.log(0.5) / Math.log(ph)).toFixed(1)} days`, { line: { color: K.purple, width: 1.4, dash: 'dash' }, hovertemplate: 'h = %{x}: %{y:.3f}<extra></extra>' }),
    leg('half of the shock', { color: K.navy, dash: 'dot', width: 1 }));
  G.lay.shapes.push(hl(a, 0.5, K.navy, 'dot', 1));
  plot('pe-chart', tr, G.lay);
  const rows = P.filter(o => !o.ig).map(o => T.pe1(NAME[o.k], dec === 'x' ? nf(o.p, 4) : nf(o.p, +dec), nf(o.hl, 1)));
  out('pe-out', rows.concat([T.pe2(nf(ph, 3), nf(Math.log(0.5) / Math.log(ph), 1))]));
}

// ================================================================= 12. forecast intervals that breathe (BET)
function bandStats(cov) {
  const r = S.bet.r, d = S.bet.d, { b, res } = arOls(r, 1), s = Math.sqrt(sum(res.map(v => v * v)) / (res.length - 2));
  const zc = cov === 95 ? 1.96 : normPpf((1 + cov / 100) / 2), g = D.ag, Z = zArGarch(), q = stdT((1 + cov / 100) / 2, g.nu);
  const lc = [], hc = [], lg = [], hg = [], oc = [], og = [];
  for (let t = 1; t < r.length; t++) {
    const mo = b[0] + b[1] * r[t - 1], mg = g.c + g.phi * r[t - 1], sg = Math.sqrt(Z.s[t - 1]);
    lc.push(mo - zc * s); hc.push(mo + zc * s); lg.push(mg - q * sg); hg.push(mg + q * sg);
    oc.push(r[t] < mo - zc * s || r[t] > mo + zc * s); og.push(r[t] < mg - q * sg || r[t] > mg + q * sg);
  }
  const dd = d.slice(1), share = (o, a, z) => { let n = 0, k = 0; dd.forEach((x, i) => { if (x >= a && x <= z) { n++; k += o[i]; } }); return k / n; };
  const w = (lo, hi, a, z) => { const v = dd.map((x, i) => (x >= a && x <= z ? hi[i] - lo[i] : NaN)).filter(x => !isNaN(x)); return [Math.min(...v), Math.max(...v)]; };
  return { dd, lc, hc, lg, hg, s, zc, q, all: [share(oc, '0', '9'), share(og, '0', '9')], y17: [share(oc, '2017-01-01', '2017-12-31'), share(og, '2017-01-01', '2017-12-31')],
    cov: [share(oc, '2020-03-01', '2020-04-30'), share(og, '2020-03-01', '2020-04-30')], wg: w(lg, hg, '2019-07-01', '2021-06-30') };
}
function blkBands() {
  const cov = +$('bd-c').value; set('bd-c', cov + '%');
  const B = bandStats(cov), [i0, i1] = slice('bet', '2019-07-01', '2021-06-30'), sl = v => v.slice(i0 - 1, i1 - 1);
  const G = grid('bd-chart', [{ r: 0, c: 0, yt: '%', x: { type: 'date' } }], { cols: 1, h: 400, legendRows: 2 });
  const a = G.ax[0], x = sl(B.dd);
  plot('bd-chart', [ln(x, S.bet.r.slice(i0, i1), a, K.red, 'BET: daily log return (%)', { line: { color: K.red, width: 0.7 }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }),
    ln(x, sl(B.lc), a, K.green, `${cov}% interval, AR(1) with constant variance`, { line: { color: K.green, width: 1.3, dash: 'dash' }, hovertemplate: '%{y:.2f}<extra></extra>' }),
    ln(x, sl(B.hc), a, K.green, null, { line: { color: K.green, width: 1.3, dash: 'dash' }, hovertemplate: '%{y:.2f}<extra></extra>' }),
    ln(x, sl(B.lg), a, K.blue, `${cov}% interval, AR(1)-GARCH(1,1)-t`, { line: { color: K.blue, width: 1.3 }, hovertemplate: '%{y:.2f}<extra></extra>' }),
    ln(x, sl(B.hg), a, K.blue, null, { line: { color: K.blue, width: 1.3 }, hovertemplate: '%{y:.2f}<extra></extra>' })], G.lay);
  const p = v => nf(100 * v, 1);
  out('bd-out', [T.bd1(cov, nf(2 * B.zc * B.s, 1), nf(B.wg[0], 1), nf(B.wg[1], 1)), T.bd2(cov, p(B.all[0]), p(B.all[1]), nf(100 - cov, 0)),
    T.bd3(p(B.y17[0]), p(B.y17[1]), p(B.cov[0]), p(B.cov[1]))]);
}

// ================================================================= 13. news impact curves
function nic(k, m, eps) {         // sigma_t^2 against eps_{t-1}, sigma_{t-1}^2 at the sample variance
  const P = D.asym[k], s2 = P.s2bar, p = P[m];
  if (m === 'egarch') return eps.map(e => { const z = e / Math.sqrt(s2); return Math.exp(p.omega + p['alpha[1]'] * (Math.abs(z) - Math.sqrt(2 / Math.PI)) + p['gamma[1]'] * z + p['beta[1]'] * Math.log(s2)); });
  return eps.map(e => p.omega + (p['alpha[1]'] + (m === 'gjr' && e < 0 ? p['gamma[1]'] : 0)) * e * e + p['beta[1]'] * s2);
}
function kernelNic(k, gr, c) {
  const r = S[k].r, x = r.slice(0, -1), y = r.slice(1).map(v => v * v), h = c * Math.sqrt(pvar(x));
  return { h, v: gr.map(g => { let sw = 0, sy = 0; for (let i = 0; i < x.length; i++) { const u = (g - x[i]) / h, w = Math.exp(-0.5 * u * u); sw += w; sy += w * y[i]; } return sy / sw; }) };
}
function blkNic() {
  const k2 = $('ni-k').value, xs = +$('ni-x').value, c = +$('ni-h').value; set('ni-x', nf(xs, 1) + '%'); set('ni-h', nf(c, 2));
  const ks = ['sp500', k2];
  const G = grid('ni-chart', ks.map((k, i) => ({ r: 0, c: i, t: NAME[k], xt: 'shock yesterday, eps(t-1) (%)', yt: i ? '' : 'variance today (%^2)' })), { cols: 2, h: 440, legendRows: 3 });
  const tr = [], rows = [];
  ks.forEach((k, i) => {
    const a = G.ax[i], r = S[k].r, gr = range(quantile(r, 0.01), quantile(r, 0.99), (quantile(r, 0.99) - quantile(r, 0.01)) / 200);
    [['garch', 'GARCH(1,1)-t', K.blue], ['gjr', 'GJR-GARCH(1,1)-t', K.red], ['egarch', 'EGARCH(1,1)-t', K.green]].forEach(([m, nm, col]) =>
      tr.push(ln(gr, nic(k, m, gr), a, col, nm, { line: { color: col, width: 1.8 }, showlegend: i === 0, hovertemplate: 'eps = %{x:.2f}: %{y:.2f}<extra>' + nm + '</extra>' })));
    const kn = kernelNic(k, gr, c);
    tr.push(ln(gr, kn.v, a, K.amber, 'kernel estimate of E[r(t)^2 | r(t-1)]', { line: { color: K.amber, width: 1.5, dash: 'dash' }, showlegend: i === 0, hovertemplate: 'r(t-1) = %{x:.2f}: %{y:.2f}<extra>kernel</extra>' }));
    G.lay.shapes.push(vl(a, -xs, K.navy, 'dot', 1), vl(a, xs, K.navy, 'dot', 1));
    const rg = nic(k, 'gjr', [-xs, xs]), re = nic(k, 'egarch', [-xs, xs]), kk = kernelNic(k, [-xs, xs], c);
    // the variance after a fall and after a rise of the selected size: one pair of points per curve
    [[nic(k, 'garch', [-xs, xs]), K.blue, 'GARCH'], [rg, K.red, 'GJR'], [re, K.green, 'EGARCH'], [kk.v, K.amber, 'kernel']].forEach(([v, col, nm], j) =>
      tr.push({ x: [-xs, xs], y: v, mode: 'markers', marker: { color: col, size: 8, line: { color: '#fff', width: 1 } },
        showlegend: false, xaxis: a.xa, yaxis: a.ya,
        hovertemplate: 'eps = %{x:.1f}: %{y:.2f}<extra>' + nm + '</extra>' }));
    rows.push(T.ni1(NAME[k], nf(xs, 1), nf(rg[0] / rg[1], 2), nf(re[0] / re[1], 2), nf(kk.v[0] / kk.v[1], 2), nf(D.asym[k].gjr['alpha[1]'], 3), nf(D.asym[k].gjr['gamma[1]'], 3)));
  });
  tr.push(leg(`points: variance after a shock of -${nf(xs, 1)}% and +${nf(xs, 1)}%`, null, { color: K.navy, size: 8 }));
  rows.push(T.ni2(nf(c, 2), nf(kernelNic('sp500', [0], c).h, 2), NAME[k2], nf(kernelNic(k2, [0], c).h, 2)));
  plot('ni-chart', tr, G.lay);
  out('ni-out', rows);
}

// ================================================================= 14. what GARCH removes
function diagStats(k, drop) {
  const r = S[k].r, z = zt(k), r2 = r.map(v => v * v);
  let im = 0; z.forEach((v, i) => { if (Math.abs(v) > Math.abs(z[im])) im = i; });
  const zz = drop ? z.filter((_, i) => i !== im) : z, z2 = zz.map(v => v * v), b = 1.96 / Math.sqrt(r.length);
  const a1 = acf(r2, 50), a2 = acf(z2, 50);
  return { a1, a2, b, max2: Math.max(...a2.map(Math.abs)), lr2: lb(r2), lz2: lb(z2), lz: lb(zz), im, zm: z[im], dm: S[k].d[im] };
}
function blkDiag() {
  const k = $('dg-k').value, drop = $('dg-o').value === 'drop', s = diagStats(k, drop), j = lags(50);
  const G = grid('dg-chart', [{ r: 0, c: 0, xt: 'lag k (days)', yt: 'autocorrelation' }], { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0];
  plot('dg-chart', [{ x: j.map(v => v - 0.2), y: s.a1, type: 'bar', width: 0.4, marker: { color: COL[k] }, name: `${NAME[k]}: squared returns`, xaxis: a.xa, yaxis: a.ya, hovertemplate: 'lag %{text}: %{y:.3f}<extra></extra>', text: j, textposition: 'none' },
    { x: j.map(v => v + 0.2), y: s.a2, type: 'bar', width: 0.4, marker: { color: k === 'bet' ? K.purple : K.red }, name: `${NAME[k]}: squared standardised residuals, GARCH(1,1)-t` + (drop ? ', largest |z| day removed' : ''), xaxis: a.xa, yaxis: a.ya, hovertemplate: 'lag %{text}: %{y:.3f}<extra></extra>', text: j, textposition: 'none' },
    leg('95% band, +/- 1.96/sqrt(T)', { color: K.navy, dash: 'dash', width: 1 })], Object.assign(G.lay, { barmode: 'overlay', shapes: [hl(a, s.b, K.navy, 'dash', 0.9), hl(a, -s.b, K.navy, 'dash', 0.9), hl(a, 0, K.navy, 'solid', 0.6)] }));
  out('dg-out', [T.dg1(NAME[k], nf(s.a1[0], 2), nf(s.a1[49], 3), nf(s.a2[0], 3), nf(s.max2, 3), nf(s.b, 3)),
    T.dg2(nf(s.lr2.q, 0), nf(s.lz2.q, s.lz2.q < 0.1 ? 3 : 1), pe(s.lz2.p), nf(s.lz.q, 1), pe(s.lz.p)), T.dg3(T.dmy(s.dm), nf(s.zm, 1), drop)]);
}

// ================================================================= 15. the term structure of volatility (S&P 500)
const TS_D = ['2017-11-03', '2020-03-16'];
function termFc(i, H = 250) {
  const g = S.sp500.g, pers = g.alpha + g.beta, s2bar = g.omega / (1 - pers), st = garch('sp500')[i], e = S.sp500.r[i] - g.mu;
  const s2n = g.omega + g.alpha * e * e + g.beta * st, f = range(1, H, 1).map(h => s2bar + pers ** (h - 1) * (s2n - s2bar));
  const an = S.sp500.ppy; let c = 0; const avg = f.map((v, j) => { c += v; return Math.sqrt(c / (j + 1) * an); });
  return { f, s2n, s2bar, day: f.map(v => Math.sqrt(v * an)), avg, sum10: sum(f.slice(0, 10)) };
}
function blkTerm() {
  const i = +$('ts-o').value, d = S.sp500.d[i]; set('ts-o', T.dmy(d));
  const P = [[TS_D[0], termFc(idx('sp500', TS_D[0])), K.green], [TS_D[1], termFc(idx('sp500', TS_D[1])), K.red], [d, termFc(i), K.blue]];
  const G = grid('ts-chart', [{ r: 0, c: 0, xt: 'horizon h (trading days)', yt: '% per year' }], { cols: 1, h: 400, legendRows: 3 });
  const a = G.ax[0], h = range(1, 250, 1), tr = [];
  P.forEach(([dd, f, c]) => tr.push(ln(h, f.day, a, c, `forecast made on ${dd}`, { line: { color: c, width: 1.8 }, hovertemplate: 'h = %{x}: %{y:.1f}%<extra>day t+h</extra>' }),
    ln(h, f.avg, a, c, null, { line: { color: c, width: 1.3, dash: 'dash' }, hovertemplate: 'h = %{x}: %{y:.1f}%<extra>average to t+h</extra>' })));
  const lr = Math.sqrt(P[0][1].s2bar * S.sp500.ppy);
  tr.push(leg('long-run volatility', { color: K.navy, dash: 'dot', width: 1 }), leg('solid: volatility on day t+h', { color: K.navy, width: 1.6 }), leg('dashed: average volatility up to day t+h', { color: K.navy, width: 1.2, dash: 'dash' }));
  G.lay.shapes.push(hl(a, lr, K.navy, 'dot', 1));
  plot('ts-chart', tr, G.lay);
  const f = P[2][1];
  out('ts-out', P.map(([dd, s]) => T.ts1(T.dmy(dd), nf(s.day[0], 1), nf(s.avg[9], 1), nf(s.avg[249], 1))).concat([T.ts2(nf(lr, 1)),
    T.ts3(T.dmy(d), nf(f.s2n, 3), nf(f.sum10, 2), nf(Math.sqrt(f.sum10), 2), nf(Math.sqrt(10 * f.s2n), 2))]));
}

// ================================================================= 16. GARCH against EWMA, out of sample
const qlike = (px, h) => px / h + Math.log(h);
const FE_M = { ge: ['ewma', 'garch', 'QLIKE(EWMA) - QLIKE(GARCH-t)', 'EWMA', 'GARCH-t'], je: ['ewma', 'gjr', 'QLIKE(EWMA) - QLIKE(GJR-t)', 'EWMA', 'GJR-t'], gj: ['garch', 'gjr', 'QLIKE(GARCH-t) - QLIKE(GJR-t)', 'GARCH-t', 'GJR-t'] };
function feStats(k, m, drop) {
  const O = D.oos[k], i0 = O.i0, r = S[k].r.slice(i0), px = r.map(v => v * v), H = { garch: O.garch, gjr: O.gjr, ewma: ewma94(k).slice(i0) };
  const [a, b] = FE_M[m], La = px.map((v, i) => qlike(v, H[a][i])), Lb = px.map((v, i) => qlike(v, H[b][i]));
  const dd = La.map((v, i) => v - Lb[i]); let im = 0; dd.forEach((v, i) => { if (v > dd[im]) im = i; });
  const keep = i => !drop || i !== im, d = S[k].d.slice(i0).filter((_, i) => keep(i)), ddk = dd.filter((_, i) => keep(i));
  let c = 0; const cum = ddk.map(v => (c += v));
  const t = dm(Lb.filter((_, i) => keep(i)), La.filter((_, i) => keep(i)));
  return { d, cum, total: c, im, dmax: S[k].d[i0 + im], vmax: dd[im], share: dd[im] / sum(dd), qa: mean(La.filter((_, i) => keep(i))), qb: mean(Lb.filter((_, i) => keep(i))), t, n: ddk.length };
}
function blkFe() {
  const m = $('fe-m').value, drop = $('fe-x').value === 'drop', M = FE_M[m];
  const G = grid('fe-chart', [{ r: 0, c: 0, t: M[2] + ', cumulated', yt: 'cumulative loss difference', x: { type: 'date' } }, { r: 0, c: 1, t: 'EUR/RON', x: { type: 'date' } }], { cols: 2, h: 400, gx: 0.09 });
  const tr = [], rows = [];
  AS.forEach(k => {
    const s = feStats(k, m, drop), a = G.ax[k === 'eurron' ? 1 : 0];
    tr.push(ln(s.d, s.cum, a, COL[k], NAME[k], { line: { color: COL[k], width: 1.5 }, hovertemplate: '%{x}: %{y:.1f}<extra>' + NAME[k] + '</extra>' }));
    rows.push(T.fe1(NAME[k], s.n, M[4], nf(s.qb, 3), M[3], nf(s.qa, 3), nf(s.t.t, 2), pe(s.t.p), T.dmy(s.dmax), nf(s.vmax, 1), drop || s.total <= 0 ? null : nf(100 * s.share, 0)));
  });
  G.ax.forEach(a => G.lay.shapes.push(hl(a, 0, K.navy, 'dash', 0.8)));
  plot('fe-chart', tr, G.lay);
  out('fe-out', rows);
}

// ================================================================= 17. VaR 1% through the COVID-19 crash
function varStats(k, al, a, b) {
  const O = D.oos[k], i0 = O.i0, r = S[k].r.slice(i0), d = S[k].d.slice(i0), ew = ewma94(k).slice(i0), zN = normPpf(al);
  const vg = new Array(r.length);
  O.win.forEach(([s, mu, nu], j) => { const e = j + 1 < O.win.length ? O.win[j + 1][0] : r.length, q = stdT(al, nu); for (let t = s; t < e; t++) vg[t] = -(mu + Math.sqrt(O.garch[t]) * q); });
  const ve = ew.map(h => -Math.sqrt(h) * zN), sel = d.map(x => x >= a && x <= b);
  const n = sel.filter(Boolean).length, eg = r.map((v, t) => sel[t] && v < -vg[t]), ee = r.map((v, t) => sel[t] && v < -ve[t]);
  return { d, r, vg, ve, sel, n, eg, ee, ng: eg.filter(Boolean).length, ne: ee.filter(Boolean).length };
}
function blkVar() {
  const k = $('va-k').value, al = +$('va-a').value / 100, w = $('va-w').value, [a, b] = w === 'covid' ? ['2019-07-01', '2021-06-30'] : ['2015-01-01', '2026-09-18'];
  const s = varStats(k, al, a, b);
  const ix = s.d.map((_, i) => i).filter(i => s.sel[i]), pick = v => ix.map(i => v[i]), x = pick(s.d);
  const G = grid('va-chart', [{ r: 0, c: 0, yt: '%', x: { type: 'date' } }], { cols: 1, h: 400, legendRows: 3 });
  const A = G.ax[0], labEn = `VaR ${+(100 * al).toFixed(1)}%`;
  plot('va-chart', [ln(x, pick(s.r), A, COL[k], `${NAME[k]}: daily log return (%)`, { line: { color: COL[k], width: 0.7 }, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }),
    ln(x, pick(s.vg).map(v => -v), A, K.red, `minus ${labEn}, GARCH(1,1)-t`, { line: { color: K.red, width: 1.4 }, hovertemplate: '%{x}: %{y:.2f}<extra>GARCH-t</extra>' }),
    ln(x, pick(s.ve).map(v => -v), A, K.green, `minus ${labEn}, EWMA-Normal`, { line: { color: K.green, width: 1.2, dash: 'dash' }, hovertemplate: '%{x}: %{y:.2f}<extra>EWMA</extra>' }),
    { x: ix.filter(i => s.eg[i]).map(i => s.d[i]), y: ix.filter(i => s.eg[i]).map(i => s.r[i]), mode: 'markers', marker: { color: K.red, size: 8 }, name: `exceedance of the GARCH-t VaR`, xaxis: A.xa, yaxis: A.ya, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' },
    { x: ix.filter(i => s.ee[i]).map(i => s.d[i]), y: ix.filter(i => s.ee[i]).map(i => s.r[i]), mode: 'markers', marker: { color: K.green, size: 10, symbol: 'x' }, name: `exceedance of the EWMA-Normal VaR`, xaxis: A.xa, yaxis: A.ya, hovertemplate: '%{x}: %{y:.2f}%<extra></extra>' }], G.lay);
  out('va-out', [T.va1(NAME[k], T.dmy(x[0]), T.dmy(x[x.length - 1]), s.n, nf(100 * al, al === 0.025 || al === 0.005 ? 1 : 0), nf(s.n * al, 1), s.ng, nf(100 * s.ng / s.n, 2), s.ne, nf(100 * s.ne / s.n, 2))]);
}

// ================================================================= wiring
const ALL = [blkReturns, blkAcfSq, blkArma, blkSim, blkEwma, blkLik1, blkLikG, blkQQ, blkVol1, blkVol2, blkPers, blkBands, blkNic, blkDiag, blkTerm, blkFe, blkVar];
function init() {
  const has = id => !!$(id), on = (ids, f) => ids.forEach(id => $(id).addEventListener('input', f));
  const players = {}, P = (btn, id, vals, ms) => (players[btn] = new Player(btn, $(id), vals, ms));
  on(['re-y'], blkReturns); P('re-play', 're-y', range(2000, 2026, 1), 600);
  on(['aq-f', 'aq-y'], blkAcfSq); P('aq-play', 'aq-y', range(2000, 2020, 2), 700);
  on(['ar-p'], blkArma); P('ar-play', 'ar-p', range(0, 5, 1), 900);
  // ▶ raises beta of GARCH(1,1) and gives ARCH(1) the same persistence, alpha = alpha + beta of GARCH (on the slider's grid, at most 0.9)
  $('si-b').addEventListener('input', () => { const p = players['si-play']; if (p && p.timer) $('si-a1').value = Math.min(0.9, Math.max(0.05, Math.round((+$('si-a').value + +$('si-b').value) * 20) / 20)); });
  on(['si-a1', 'si-a', 'si-b'], blkSim); P('si-play', 'si-b', range(0, 0.88, 0.04), 450);
  on(['ew-l', 'ew-w'], blkEwma); P('ew-play', 'ew-l', range(0.80, 0.99, 0.01), 450);
  on(['l1-a', 'l1-n'], blkLik1); P('l1-play', 'l1-n', ['250', '500', '1000', '2000', '5000'], 1100);
  on(['lg-n'], blkLikG); P('lg-play', 'lg-n', range(250, 5000, 250), 600);
  on(['qq-nu'], blkQQ); P('qq-play', 'qq-nu', [3, 4, 5, 6.26, 8, 10, 15, 20, 30], 800);
  on(['v1-c'], blkVol1); P('v1-play', 'v1-c', range(0.25, 2, 0.25), 800);
  on(['v2-c'], blkVol2); P('v2-play', 'v2-c', range(0.25, 2, 0.25), 800);
  on(['pe-d', 'pe-p'], blkPers); P('pe-play', 'pe-p', range(0.9, 0.999, 0.003), 400);
  on(['bd-c'], blkBands); P('bd-play', 'bd-c', range(80, 99, 1), 500);
  on(['ni-k', 'ni-x', 'ni-h'], blkNic); P('ni-play', 'ni-x', range(1, 4, 0.5), 800);
  on(['dg-k', 'dg-o'], blkDiag); P('dg-play', 'dg-k', ['bet', 'btc', 'eurron', 'sp500'], 1200);
  const ts = $('ts-o'); ts.min = idx('sp500', '2007-01-01'); ts.max = S.sp500.d.length - 1; ts.value = ts.max;
  on(['ts-o'], blkTerm); P('ts-play', 'ts-o', range(idx('sp500', '2020-02-03'), idx('sp500', '2020-06-30'), 5), 350);
  on(['fe-m', 'fe-x'], blkFe); P('fe-play', 'fe-m', ['ge', 'gj', 'je'], 1300);
  on(['va-k', 'va-a', 'va-w'], blkVar); P('va-play', 'va-a', ['0.5', '1', '2.5', '5'], 1200);
  const PFX = { returns: 're', 'acf-squares': 'aq', 'arma-resid': 'ar', simulated: 'si', 'ewma-garch': 'ew', 'lik-arch1': 'l1', 'lik-garch': 'lg', qq: 'qq',
    'vol-sp500-bet': 'v1', 'vol-eurron-btc': 'v2', persistence: 'pe', bands: 'bd', nic: 'ni', 'acf-diag': 'dg', 'term-structure': 'ts', 'forecast-eval': 'fe', var: 'va' };
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if (has(b)) predict(b, Object.assign({ player: players[PFX[b] + '-play'], veil: { out: true } }, sp)); });
  if (T.ticket && exitTicket) exitTicket(T.ticket);
  ALL.forEach(f => f());
  let w = window.innerWidth, tm = null;
  window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(() => { if ((window.innerWidth < 720) !== (w < 720) || Math.abs(window.innerWidth - w) > 200) { w = window.innerWidth; ALL.forEach(f => f()); } }, 250); });
}
window.CH5 = { S, garch, zt, ewma, sdYear, sdEpi, acfSq, arStats, zArGarch, zArP, simPath, simStats, kArch, kGarch, ewStats, lik1, likG, qqZ, stdT, volStats, persRows,
  bandStats, nic, kernelNic, diagStats, termFc, feStats, varStats, archLm, lb, dm, idx, chi2sf };
if (!window.CH5_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
