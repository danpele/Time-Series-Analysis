// Chapter 7 interactive page (cointegration and VECM): one block per chart slide, plus the result tables and the worked
// examples of the lecture, computed in the browser from ch7_data.js.
(function () {
'use strict';
const D = window.CH7_DATA;
const { sum, chol, eigSym, mean, nf, $, AX, layout, plot, Player, range, predict, exitTicket } = window.TSAI;
const fixP = x => x.replace(/= &lt;/g, '&lt;').replace(/=&lt;/g, '&lt;');
const out = (id, items) => window.TSAI.out(id, items.map(fixP));
const normCdf = window.TSAI.normCdf;
const T = window.CH7_TEXT;
const L = T.L;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of generate_all_charts.py / tsa_style.py
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', purple: '#8E44AD', teal: '#17A2B8', orange: '#E67E22', navy: '#1F2A44' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.purple]: '142,68,173', [K.amber]: '181,133,63',
  [K.teal]: '23,162,184', [K.orange]: '230,126,34', [K.navy]: '31,42,68' };

// ---------------------------------------------------------------- small helpers
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
const pe = p => (p < 0.001 ? 'p &lt; ' + nf(0.001, 3) : 'p = ' + nf(p, 3));
const pv = p => (p < 0.001 ? '&lt; ' + nf(0.001, 3) : nf(p, 3));
const has = id => !!$(id);
const sdp = x => { const m = mean(x); return Math.sqrt(sum(x.map(v => (v - m) ** 2)) / x.length); };       // numpy std (ddof 0)
const sd1 = x => { const m = mean(x); return Math.sqrt(sum(x.map(v => (v - m) ** 2)) / (x.length - 1)); };
const cumsum = x => { let s = 0; return x.map(v => (s += v)); };
const first = (x, v0) => x.map(v => v - (v0 === undefined ? x[0] : v0));
const cache = {};
const memo = (k, f) => (k in cache ? cache[k] : (cache[k] = f()));

// ---------------------------------------------------------------- statistics (ports of statsmodels: OLS, adfuller, mackinnonp/crit, coint, coint_johansen)
const MK = D.mk;
function cholSolve(A, b) {
  const n = A.length, Lc = A.map(r => r.slice());
  for (let j = 0; j < n; j++) { let s = Lc[j][j]; for (let k = 0; k < j; k++) s -= Lc[j][k] * Lc[j][k]; Lc[j][j] = Math.sqrt(s);
    for (let i = j + 1; i < n; i++) { let t = Lc[i][j]; for (let k = 0; k < j; k++) t -= Lc[i][k] * Lc[j][k]; Lc[i][j] = t / Lc[j][j]; } }
  const fw = v => { const y = v.slice(); for (let i = 0; i < n; i++) { for (let k = 0; k < i; k++) y[i] -= Lc[i][k] * y[k]; y[i] /= Lc[i][i]; } return y; };
  const bw = y => { const x = y.slice(); for (let i = n - 1; i >= 0; i--) { for (let k = i + 1; k < n; k++) x[i] -= Lc[k][i] * x[k]; x[i] /= Lc[i][i]; } return x; };
  return { x: b ? bw(fw(b)) : null, inv: () => { const I = []; for (let j = 0; j < n; j++) { const e = new Array(n).fill(0); e[j] = 1; I.push(bw(fw(e))); } return I; } };
}
function olsFit(y, row, k) {               // row(t) -> regressors; b, ssr, n, k, inverse of X'X
  const n = y.length, G = Array.from({ length: k }, () => new Array(k).fill(0)), h = new Array(k).fill(0);
  for (let t = 0; t < n; t++) { const x = row(t); for (let a = 0; a < k; a++) { h[a] += x[a] * y[t]; const Ga = G[a], xa = x[a]; for (let c = a; c < k; c++) Ga[c] += xa * x[c]; } }
  for (let a = 0; a < k; a++) for (let c = 0; c < a; c++) G[a][c] = G[c][a];
  const S = cholSolve(G, h), b = S.x;
  let ssr = 0; for (let t = 0; t < n; t++) { const x = row(t); let f = 0; for (let a = 0; a < k; a++) f += x[a] * b[a]; ssr += (y[t] - f) ** 2; }
  return { b, ssr, n, k, inv: S.inv };
}
function olsT(y, row, k) {                 // coefficients, standard errors and t
  const f = olsFit(y, row, k), s2 = f.ssr / (f.n - k), iv = f.inv(), se = f.b.map((_, a) => Math.sqrt(s2 * iv[a][a]));
  let my = mean(y), sst = 0; y.forEach(v => { sst += (v - my) ** 2; });
  return { b: f.b, se, t: f.b.map((v, a) => v / se[a]), n: f.n, r2: 1 - f.ssr / sst };
}
const polyv = (c, x) => c.reduce((s, v, i) => s + v * x ** i, 0);
function mackinnonp(stat, reg, n) {
  const i = n - 1;
  if (stat > MK.max[reg][i]) return 1; if (stat < MK.min[reg][i]) return 0;
  return normCdf(polyv(stat <= MK.star[reg][i] ? MK.small[reg][i] : MK.large[reg][i], stat));
}
const mackinnoncrit = (n, reg, nobs) => MK.t2010[reg][n - 1].map(c => (isFinite(nobs) ? polyv(c, 1 / nobs) : c[0]));
function adfReg(x, reg, lag, maxlag) {      // ADF regression with `lag` lagged differences on the last len - 1 - maxlag changes
  const dx = []; for (let t = 1; t < x.length; t++) dx.push(x[t] - x[t - 1]);
  const nobs = dx.length - maxlag, nt = reg === 'n' ? 0 : reg.length, y = dx.slice(dx.length - nobs);
  const row = t => { const j = dx.length - nobs + t, r = []; if (nt) r.push(1); if (nt === 2) r.push(t + 1); r.push(x[j]);
    for (let l = 1; l <= lag; l++) r.push(dx[j - l]); return r; };
  return olsFit(y, row, nt + 1 + lag);
}
function adfStat(f, reg) { const nt = reg === 'n' ? 0 : reg.length, s2 = f.ssr / (f.n - f.k), iv = f.inv(); return { gamma: f.b[nt], se: Math.sqrt(s2 * iv[nt][nt]) }; }
function adfuller(x, reg, fixed) {          // autolag AIC (statsmodels.adfuller), or a fixed number of lags
  const n0 = x.length - 1, nt = reg === 'n' ? 0 : reg.length;
  let bl = fixed;
  if (fixed === undefined) {
    const maxlag = Math.min(Math.floor(n0 / 2) - nt - 1, Math.ceil(12 * Math.pow(n0 / 100, 0.25)));
    let best = Infinity; bl = 0;
    for (let l = 0; l <= maxlag; l++) { const f = adfReg(x, reg, l, maxlag), aic = f.n * Math.log(f.ssr / f.n) + 2 * f.k; if (aic < best - 1e-9) { best = aic; bl = l; } }
  }
  const f = adfReg(x, reg, bl, bl), g = adfStat(f, reg), stat = g.gamma / g.se;
  return { stat, p: mackinnonp(stat, reg, 1), lags: bl, crit5: mackinnoncrit(1, reg, f.n)[1], nobs: f.n, gamma: g.gamma, se: g.se };
}
function ols1(y, x) {
  const n = y.length, mx = mean(x), my = mean(y); let sxy = 0, sxx = 0, syy = 0;
  for (let t = 0; t < n; t++) { sxy += (x[t] - mx) * (y[t] - my); sxx += (x[t] - mx) ** 2; syy += (y[t] - my) ** 2; }
  const b = sxy / sxx, a = my - b * mx, e = y.map((v, t) => v - a - b * x[t]);
  let ee = 0, dd = 0; e.forEach((v, t) => { ee += v * v; if (t) dd += (v - e[t - 1]) ** 2; });
  return { a, b, e, r2: 1 - ee / syy, dw: dd / ee, n };
}
function egTest(y, x, fixed) {             // Engle-Granger (statsmodels coint, trend c) with its step 1
  const s1 = ols1(y, x), adf = adfuller(s1.e, 'n', fixed), crit = mackinnoncrit(2, 'c', y.length - 1);
  return Object.assign({}, s1, { tau: adf.stat, p: mackinnonp(adf.stat, 'c', 2), lags: adf.lags, gamma: adf.gamma, se: adf.se, crit1: crit[0], crit5: crit[1], crit10: crit[2], dfcrit5: mackinnoncrit(1, 'c', y.length - 1)[1] });
}
function acfv(x, Kmax) { const m = mean(x), d = x.map(v => v - m), c0 = sum(d.map(v => v * v)); const r = [];
  for (let k = 1; k <= Kmax; k++) { let s = 0; for (let t = k; t < d.length; t++) s += d[t] * d[t - k]; r.push(s / c0); } return r; }
const halfLife = r => (r > 0 && r < 1 ? Math.log(0.5) / Math.log(r) : Infinity);
const mmul = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const tr = A => A[0].map((_, j) => A.map(r => r[j]));
function triInv(Lm) { const n = Lm.length, I = Array.from({ length: n }, (_, i) => { const r = new Array(n).fill(0); r[i] = 1; return r; });
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { let s = I[i][j]; for (let k = 0; k < i; k++) s -= Lm[i][k] * I[k][j]; I[i][j] = s / Lm[i][i]; } return I; }
function johansen(Y, det, k) {             // statsmodels coint_johansen: rows = observations
  const n = Y.length, m = Y[0].length;
  const detrend = (Z, ord) => {
    if (ord < 0) return Z.map(r => r.slice());
    const N_ = Z.length, X = Z.map((_, t) => { const v = N_ === 1 ? -1 : -1 + 2 * t / (N_ - 1), r = []; for (let p = ord; p >= 0; p--) r.push(v ** p); return r; });
    const o = Z.map(r => r.slice());
    for (let j = 0; j < Z[0].length; j++) { const f = olsFit(Z.map(r => r[j]), t => X[t], ord + 1); for (let t = 0; t < N_; t++) o[t][j] -= X[t].reduce((s, v, a) => s + v * f.b[a], 0); }
    return o;
  };
  const f = det > -1 ? 0 : det, E = detrend(Y, det);
  const dx = []; for (let t = 1; t < n; t++) dx.push(E[t].map((v, j) => v - E[t - 1][j]));
  let z = []; for (let t = k; t < dx.length; t++) { const r = []; for (let l = 1; l <= k; l++) r.push(...dx[t - l]); z.push(r); }
  if (k) z = detrend(z, f);
  const resid = A => { if (!k) return A; const q = z[0].length, o = A.map(r => r.slice());
    for (let j = 0; j < A[0].length; j++) { const fit = olsFit(A.map(r => r[j]), t => z[t], q); for (let t = 0; t < A.length; t++) o[t][j] -= z[t].reduce((s, v, a) => s + v * fit.b[a], 0); }
    return o; };
  const r0 = resid(detrend(dx.slice(k), f)), rk = resid(detrend(E.slice(0, n - k).slice(1), f)), Tn = rk.length;
  const cov = (A, B) => A[0].map((_, i) => B[0].map((_, j) => { let s = 0; for (let t = 0; t < Tn; t++) s += A[t][i] * B[t][j]; return s / Tn; }));
  const skk = cov(rk, rk), sk0 = cov(rk, r0), s00i = cholSolve(cov(r0, r0)).inv();
  const sig = mmul(mmul(sk0, s00i), tr(sk0)), Li = triInv(chol(skk)), Ms = mmul(mmul(Li, sig), tr(Li));
  const ev = eigSym(Ms.map((r, i) => r.map((v, j) => (v + Ms[j][i]) / 2))).slice().sort((a, b) => b - a);
  const trace = ev.map((_, i) => -Tn * sum(ev.slice(i).map(l => Math.log(1 - l)))), maxeig = ev.map(l => -Tn * Math.log(1 - l));
  const cvt = ev.map((_, i) => MK.sjt[String(det)][m - i - 1]), cvm = ev.map((_, i) => MK.sja[String(det)][m - i - 1]);
  const rank = (s, c) => { for (let i = 0; i < m; i++) if (s[i] < c[i][1]) return i; return m; };
  return { eig: ev, trace, maxeig, cvt, cvm, T: Tn, n, k, rt: rank(trace, cvt), rm: rank(maxeig, cvm) };
}
function eig2(A) { const tr_ = A[0][0] + A[1][1], dt = A[0][0] * A[1][1] - A[0][1] * A[1][0], disc = tr_ * tr_ / 4 - dt;
  if (disc >= 0) { const s = Math.sqrt(disc); return [tr_ / 2 + s, tr_ / 2 - s].map(Math.abs).sort((a, b) => b - a); }
  const md = Math.sqrt(dt); return [md, md]; }

// ---------------------------------------------------------------- panels: one Plotly figure with several axes; stacked on narrow screens
function grid(id, cells, o) {
  const el = $(id), narrow = el.clientWidth < 560, maxc = Math.floor(el.clientWidth / 6.6);
  const wrap = t => { if (!t || t.replace(/<[^>]+>/g, '').length <= maxc) return t; const w = t.split(' '), Ls = ['']; w.forEach(x => { const c = Ls[Ls.length - 1];
    if ((c + ' ' + x).replace(/<[^>]+>/g, '').trim().length > maxc && c) Ls.push(x); else Ls[Ls.length - 1] = (c + ' ' + x).trim(); }); return Ls.join('<br>'); };
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
const ln = (x, y, a, color, name, o) => { const r = Object.assign({ x, y, mode: 'lines', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }, o || {}); r.line = Object.assign({ color, width: 1.4 }, (o && o.line) || {}); return r; };
const bar = (x, y, a, color, name, o) => Object.assign({ x, y, type: 'bar', marker: { color }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya }, o || {});
const band = (x, lo, hi, a, col, al, name) => ({ x: [...x, ...[...x].reverse()], y: [...hi, ...[...lo].reverse()], fill: 'toself', fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, mode: 'lines', hoverinfo: 'skip', name: name || '', showlegend: !!name, xaxis: a.xa, yaxis: a.ya });
const DATE = { type: 'date' };

// ---------------------------------------------------------------- tables and worked steps (HTML)
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}${v.span ? ` colspan="${v.span}"` : ''}>${v.v}</${tag}>`; };
  el.innerHTML = (o && o.cap ? `<p class="c7cap">${fixP(o.cap)}</p>` : '') + '<table><thead>' + (o && o.head2 ? '<tr>' + o.head2.map(c => cell(c, 'th')).join('') + '</tr>' : '')
    + '<tr>' + head.map(c => cell(c, 'th')).join('') + '</tr></thead><tbody>'
    + rows.map(r => '<tr' + (r.hi ? ' class="hi"' : '') + '>' + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c7cap">${o.note}</p>` : '');
}
const num = (x, d, bold) => ({ v: bold ? `<b>${nf(x, d)}</b>` : nf(x, d), cls: 'r' });
const txt = (v, bold) => ({ v: bold ? `<b>${v}</b>` : v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${fixP(s)}</li>`).join(''); };
const pn = (x, d) => (x < 0 ? `(${nf(x, d)})` : nf(x, d));

// ---------------------------------------------------------------- data of the chapter
const isoM = (y0, m0, k) => { const mm = m0 - 1 + k, y = y0 + Math.floor(mm / 12), m = mm % 12 + 1; return `${y}-${String(m).padStart(2, '0')}-01`; };
const isoQ = (y0, k) => isoM(y0, 1, 3 * k);
const monthEnd = iso => { const y = +iso.slice(0, 4), m = +iso.slice(5, 7); return new Date(Date.UTC(y, m, 0)).toISOString().slice(0, 10); };
const days = (d0, gap) => { const d = [d0]; let t = Date.parse(d0 + 'T00:00:00Z'); gap.forEach(g0 => { t += g0 * 864e5; d.push(new Date(t).toISOString().slice(0, 10)); }); return d; };
const YD = { d: D.yields.GS1.map((_, k) => isoM(1960, 1, k)), gs1: D.yields.GS1.map(v => v / 100), gs5: D.yields.GS5.map(v => v / 100),
  gs10: D.yields.GS10.map(v => v / 100), tb3: D.yields.TB3MS.map(v => v / 100), rw: D.yields.rw };
const MAC = { d: D.macro.y.map((_, k) => isoQ(1959, k)), y: D.macro.y, c: D.macro.c, i: D.macro.i };
const BK = { d: days(D.mkt.d0, D.mkt.gap), tlv: D.mkt.tlv, brd: D.mkt.brd, i14: D.mkt.i14 };
const FXD = { d: D.mkt.ron.map((_, k) => monthEnd(isoM(2005, 7, k))), ron: D.mkt.ron, huf: D.mkt.huf, pln: D.mkt.pln };
const PPD = { d: D.ppp.s.map((_, k) => isoM(2005, 7, k)), s: D.ppp.s, p: D.ppp.p, ps: D.ppp.ps, rel: D.ppp.p.map((v, k) => v - D.ppp.ps[k]) };
const RE = { d: D.roea.ro.map((_, k) => isoM(1995, 8, k)), ro: D.roea.ro, ea: D.roea.ea };
const ROC = { d: D.egtab.ro.c.map((_, k) => isoQ(1995, k)), c: D.egtab.ro.c, y: D.egtab.ro.y };
const DR = D.drunk;
const dropK = (n, k) => Math.floor(n * k / 100);
const YRS = d => +d.slice(0, 4);
const sl = (o, i0, i1) => { const r = {}; Object.keys(o).forEach(k => { r[k] = Array.isArray(o[k]) ? o[k].slice(i0, i1) : o[k]; }); return r; };
const idxFrom = (d, iso) => { const i = d.findIndex(x => x >= iso); return i < 0 ? d.length : i; };
const idxTo = (d, iso) => { let i = d.length - 1; while (i > 0 && d[i] > iso) i--; return i; };

// ================================================================= 1. the drunk and her dog
function drunkSim(c, d) {
  const n = DR.T, x = new Array(n).fill(0), y = new Array(n).fill(0), z = new Array(n).fill(0);
  y[0] = 2;
  for (let t = 1; t < n; t++) { x[t] = x[t - 1] + DR.u[t] + c * (y[t - 1] - x[t - 1]); y[t] = y[t - 1] + DR.w[t] + d * (x[t - 1] - y[t - 1]); z[t] = z[t - 1] + DR.v[t]; }
  return { x, y, z };
}
function drunkStats(c, d) {
  const s = drunkSim(c, d), dist = s.y.map((v, t) => v - s.x[t]), stray = s.z.map((v, t) => v - s.x[t]);
  return { s, dist, stray, ax: adfuller(s.x, 'c'), ad: adfuller(dist, 'c'), as: adfuller(stray, 'c'), sdd: sdp(dist), sds: sdp(stray), eg: egTest(s.y, s.x), rho: 1 - c - d };
}
function blkDrunk() {
  const c = +$('dd-c').value, d = +$('dd-d').value; set('dd-c', nf(c, 2)); set('dd-d', nf(d, 2));
  const S = drunkStats(c, d), t = range(0, DR.T - 1, 1);
  const G = grid('dd-chart', [{ r: 0, c: 0, t: 'Three random walks', xt: 't' }, { r: 0, c: 1, t: 'The distances', xt: 't' }], { cols: 2, h: 340, legendRows: 2 });
  const [a, b] = G.ax;
  G.lay.shapes.push(hl(b, 0, K.navy, 'solid', 0.6));
  plot('dd-chart', [ln(t, S.s.x, a, K.blue, 'the drunk x_t'), ln(t, S.s.y, a, K.red, 'her dog y_t'), ln(t, S.s.z, a, K.amber, 'a stray dog z_t'),
    ln(t, S.dist, b, K.red, 'dog minus drunk y_t - x_t', { line: { color: K.red, width: 1 } }), ln(t, S.stray, b, K.orange, 'stray dog minus drunk z_t - x_t', { line: { color: K.orange, width: 1 } })], G.lay);
  const hlf = halfLife(S.rho);
  out('dd-out', [T.dd1(nf(c, 2), nf(d, 2), nf(S.rho, 2), isFinite(hlf) ? nf(hlf, 2) : '∞'), T.dd2(pv(S.ax.p), pv(S.ad.p), nf(S.sdd, 2), pv(S.as.p), nf(S.sds, 1))]);
  table('dd-tb', [L.series, { v: 'τ (ADF)', cls: 'r' }, { v: L.lags, cls: 'r' }, { v: 'p', cls: 'r' }, { v: L.sd, cls: 'r' }],
    [['x<sub>t</sub>', num(S.ax.stat, 2), txt(S.ax.lags), txt(pv(S.ax.p), S.ax.p < 0.05), num(sdp(S.s.x), 2)],
     ['y<sub>t</sub> − x<sub>t</sub>', num(S.ad.stat, 2), txt(S.ad.lags), txt(pv(S.ad.p), S.ad.p < 0.05), num(S.sdd, 2)],
     ['z<sub>t</sub> − x<sub>t</sub>', num(S.as.stat, 2), txt(S.as.lags), txt(pv(S.as.p), S.as.p < 0.05), num(S.sds, 2)]],
    { cap: T.ddCap(nf(S.eg.tau, 2), pv(S.eg.p), nf(S.eg.b, 3)) });
}

// ================================================================= 2. four pairs of trending series
function pairsData(k) {
  const cut = (o, n) => sl(o, dropK(n, k));
  const R = cut({ d: YD.d, y: YD.gs10, x: YD.tb3 }, YD.d.length), M = cut({ d: MAC.d, y: MAC.c, x: MAC.y }, MAC.d.length);
  const B = cut({ d: BK.d.slice(BK.i14), y: BK.tlv.slice(BK.i14), x: BK.brd.slice(BK.i14) }, BK.d.length - BK.i14);
  const F = cut({ d: FXD.d, y: FXD.ron, x: FXD.huf }, FXD.d.length);
  return { R, M, B, F };
}
function blkExamples() {
  const k = +$('ex-k').value; set('ex-k', k + '%');
  const P = pairsData(k);
  const G = grid('ex-chart', [{ r: 0, c: 0, t: 'US Treasury yields (% per year)', x: DATE }, { r: 0, c: 1, t: 'US output and consumption (100 ln, start = 0)', x: DATE },
    { r: 1, c: 0, t: 'Two Romanian banks (100 ln adjusted price, start = 0)', x: DATE }, { r: 1, c: 1, t: 'Two exchange rates (100 ln, start = 0)', x: DATE }], { cols: 2, h: 520, gyPx: 60, legendRows: 2 });
  const A = G.ax, o = { line: { width: 1 } };
  plot('ex-chart', [ln(P.R.d, P.R.x, A[0], K.blue, '3-month bill', o), ln(P.R.d, P.R.y, A[0], K.red, '10-year yield', o),
    ln(P.M.d, first(P.M.x), A[1], K.green, 'real GDP'), ln(P.M.d, first(P.M.y), A[1], K.purple, 'real consumption'),
    ln(P.B.d, first(P.B.y), A[2], K.red, 'Banca Transilvania', o), ln(P.B.d, first(P.B.x), A[2], K.blue, 'BRD', o),
    ln(P.F.d, first(P.F.y), A[3], K.green, 'EUR/RON'), ln(P.F.d, first(P.F.x), A[3], K.orange, 'EUR/HUF')], G.lay);
  const rows = [['rates', P.R], ['cons', P.M], ['banks', P.B], ['fx', P.F]].map(([key, p]) => [key, p, egTest(p.y, p.x)]);
  table('ex-tb', [L.pair, { v: 'T', cls: 'r' }, { v: 'b̂', cls: 'r' }, { v: 'R²', cls: 'r' }, { v: 'τ (k)', cls: 'r' }, { v: 'p', cls: 'r' }, { v: L.crit5, cls: 'r' }],
    rows.map(([key, p, e]) => [T.pairName[key] + ` (${({ rates: T.my, cons: T.q, banks: T.dmy, fx: T.my })[key](p.d[0])})`, txt(String(e.n)), num(e.b, 3), num(e.r2, 2), txt(`${nf(e.tau, 2)} (${e.lags})`), txt(pv(e.p), e.p < 0.05), num(e.crit5, 2)]),
    { cap: T.exCap(k) });
  out('ex-out', [T.ex1(rows.map(([key, , e]) => [T.pairName[key], nf(e.tau, 2), pv(e.p), e.p < 0.05]))]);
}

// ================================================================= 3. the null distribution of the Engle-Granger statistic
const EGD = D.egdist, EGX = range(EGD.grid[0], EGD.grid[1], (EGD.grid[1] - EGD.grid[0]) / (EGD.grid[2] - 1)).slice(0, EGD.grid[2]);
const npdf = x => Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
function blkEgDist() {
  const Tn = $('ed-T').value, al = +$('ed-a').value, j = { 1: 0, 5: 1, 10: 2 }[al], o = EGD.d[Tn];
  const G = grid('ed-chart', [{ r: 0, c: 0, t: `Null distributions of the t-statistic (T = ${Tn}, ${EGD.R.toLocaleString("en-US")} simulations); dotted: ${al}% quantiles`, xt: 't-statistic', x: { range: [-6, 3] } }], { cols: 1, h: 360, legendRows: 2 });
  const a = G.ax[0], cols = { 1: K.blue, 2: K.red, 3: K.green }, labs = { 1: 'Dickey-Fuller, one series (constant)', 2: 'Engle-Granger, 2 variables', 3: 'Engle-Granger, 3 variables' };
  const tr_ = [ln(EGX, EGX.map(npdf), a, K.navy, 'N(0, 1)', { line: { color: K.navy, width: 1, dash: 'dash' } })];
  [1, 2, 3].forEach(k => { tr_.push(ln(EGX, o.kde[k], a, cols[k], labs[k])); G.lay.shapes.push(vl(a, o.q[k][j], cols[k], 'dot', 1.2)); });
  plot('ed-chart', tr_, G.lay);
  table('ed-tb', [L.vars, { v: T.edQ(al), cls: 'r' }, { v: T.edMk(al), cls: 'r' }, { v: T.edSize(al), cls: 'r' }],
    [1, 2, 3].map(k => [k === 1 ? T.edOne : String(k), num(o.q[k][j], 2), num(o.mk[k][j], 2), k === 1 ? txt('–') : txt(nf(100 * o.size[k][j], 1) + '%', true)]),
    { cap: T.edCap(Tn) });
  out('ed-out', [T.ed1(Tn, al, nf(o.q[1][j], 2), nf(o.q[2][j], 2), nf(o.q[3][j], 2), nf(o.mk[1][j], 2), nf(o.mk[2][j], 2), nf(o.mk[3][j], 2)),
    T.ed2(al, nf(o.mk[1][j], 2), nf(100 * o.size[2][j], 1), nf(100 * o.size[3][j], 1))]);
}

// ================================================================= 4. the step-1 residuals
function blkEgSteps() {
  const k = +$('es-k').value, Lg = +$('es-L').value; set('es-k', k + '%'); set('es-L', Lg);
  const P = pairsData(k), e1 = ols1(P.M.y, P.M.x), e2 = ols1(P.R.y, P.R.x), rw = YD.rw.slice(0, P.R.y.length);
  const a1 = acfv(e1.e, Lg), a2 = acfv(e2.e, Lg), a3 = acfv(rw, Lg), lags = range(1, Lg, 1);
  const G = grid('es-chart', [{ r: 0, c: 0, t: 'Consumption on income: residuals', x: DATE }, { r: 0, c: 1, t: '10-year on 3-month yield: residuals', x: DATE },
    { r: 0, c: 2, t: 'Sample ACF', xt: 'lag' }], { cols: 3, h: 330, gx: 0.06, legendRows: 2 });
  const [A1, A2, A3] = G.ax;
  [A1, A2, A3].forEach(ax => G.lay.shapes.push(hl(ax, 0, K.navy, 'solid', 0.6)));
  plot('es-chart', [ln(P.M.d, e1.e, A1, K.purple, null, { line: { color: K.purple, width: 1 } }), ln(P.R.d, e2.e, A2, K.red, null, { line: { color: K.red, width: 0.9 } }),
    ln(lags, a1, A3, K.purple, 'consumption residuals (quarters)', { mode: 'lines+markers', marker: { size: 4 } }),
    ln(lags, a2, A3, K.red, 'yield residuals (months)', { mode: 'lines+markers', marker: { size: 4, symbol: 'square' } }),
    ln(lags, a3, A3, K.blue, 'a random walk', { line: { color: K.blue, width: 1.2, dash: 'dash' } })], G.lay);
  out('es-out', [T.es1(T.q(P.M.d[0]), nf(a1[0], 2), nf(a1[7], 2), nf(sd1(e1.e), 2)), T.es2(T.my(P.R.d[0]), nf(a2[0], 2), nf(a2[11], 2), nf(sd1(e2.e), 2), nf(a3[11], 2))]);
}

// ================================================================= 5. error correction models
function ecmFit(y, x, p) {                // statsmodels OLS of Delta y on const, e_{t-1}, Delta x_t and p lagged differences
  const s1 = ols1(y, x), n = y.length, dy = [], dx = [];
  for (let t = 1; t < n; t++) { dy.push(y[t] - y[t - 1]); dx.push(x[t] - x[t - 1]); }
  const t0 = p, yy = dy.slice(t0), row = t => { const j = t + t0, r = [1, s1.e[j], dx[j]]; for (let l = 1; l <= p; l++) r.push(dy[j - l], dx[j - l]); return r; };
  const f = olsT(yy, row, 3 + 2 * p);
  return { s1, g: f.b[1], gt: f.t[1], gse: f.se[1], d0: f.b[2], c: f.b[0], b: f.b, t: f.t, n: f.n, half: halfLife(1 + f.b[1]) };
}
function shortFit(y, x) {                 // the 3-month rate on the same equilibrium error (fig_ecm)
  const s1 = ols1(y, x), n = y.length, dy = [], dx = [];
  for (let t = 1; t < n; t++) { dy.push(y[t] - y[t - 1]); dx.push(x[t] - x[t - 1]); }
  const f = olsT(dx.slice(1), t => [1, s1.e[t + 1], dy[t], dx[t]], 4);
  return { g: f.b[1], gt: f.t[1] };
}
function ecmState(endY, p) {
  const mi = idxTo(MAC.d, `${Math.min(endY, 2009)}-12-31`) + 1, yi = idxTo(YD.d, `${endY}-12-31`) + 1;
  return { cons: ecmFit(MAC.c.slice(0, mi), MAC.y.slice(0, mi), p), long: ecmFit(YD.gs10.slice(0, yi), YD.tb3.slice(0, yi), p),
    short: shortFit(YD.gs10.slice(0, yi), YD.tb3.slice(0, yi)), ce: MAC.d[mi - 1], ye: YD.d[yi - 1] };
}
function blkEcm() {
  const e = +$('em-e').value, p = +$('em-p').value; set('em-e', e); set('em-p', p);
  const S = ecmState(e, p), h = range(0, 40, 1), dec = g0 => h.map(x => (1 + g0) ** x);
  const G = grid('em-chart', [{ r: 0, c: 0, t: 'A unit deviation after h periods: (1+γ)^h', xt: 'h' }, { r: 0, c: 1, t: 'Estimated adjustment of the equilibrium error', xt: 'h' }], { cols: 2, h: 360, legendRows: 3 });
  const [a, b] = G.ax, tr_ = [];
  [[-0.05, K.blue], [-0.2, K.red], [-0.5, K.green]].forEach(([g0, col]) => tr_.push(ln(h, dec(g0), a, col, `γ = ${nf(g0, 2)}: half-life ${nf(halfLife(1 + g0), 1)}`)));
  tr_.push(ln(h, dec(S.cons.g), a, K.purple, null, { line: { color: K.purple, width: 1.2, dash: 'dot' } }), ln(h, dec(S.long.g), a, K.orange, null, { line: { color: K.orange, width: 1.2, dash: 'dot' } }));
  [[S.cons, K.purple, `US consumption, quarters (γ̂ = ${nf(S.cons.g, 3)})`], [S.long, K.orange, `10-year yield, months (γ̂ = ${nf(S.long.g, 3)})`]].forEach(([f, col, name]) => {
    tr_.push(band(h, dec(Math.min(-1e-9, f.g + 2 * f.gse)), dec(Math.max(-1.99, f.g - 2 * f.gse)), b, col, 0.15, null), ln(h, dec(f.g), b, col, name)); });
  tr_.push(leg('±2 SE of γ̂ (band)', null, { color: `rgba(${RGB[K.purple]},0.3)`, symbol: 'square', size: 11 }));
  [a, b].forEach(ax => G.lay.shapes.push(hl(ax, 0.5, K.navy, 'dot', 0.8)));
  plot('em-chart', tr_, G.lay);
  out('em-out', [T.em1(T.q(S.ce), nf(S.cons.g, 3), nf(S.cons.gt, 2), nf(S.cons.half, 1), nf(S.cons.d0, 2)),
    T.em2(T.my(S.ye), nf(S.long.g, 3), nf(S.long.gt, 2), nf(S.long.half, 1), nf(S.long.d0, 2), nf(S.short.g, 3), nf(S.short.gt, 2))]);
}

// ================================================================= 6-7. the term-structure VECM (precomputed fits)
const VM = D.vecm;
const vfit = (e, k) => VM.m[`${e}|${k}`];
const ymLabel = e => T.my(e + '-01');
function blkRates() {
  const e = $('ra-e').value, k = +$('ra-k').value, f = vfit(e, k), n = idxTo(YD.d, e + '-01') + 1;
  const d = YD.d.slice(0, n), Yv = d.map((_, t) => [YD.gs1[t], YD.gs5[t], YD.gs10[t]]);
  const z = [0, 1].map(j => Yv.map(r => r[0] * f.beta[0][j] + r[1] * f.beta[1][j] + r[2] * f.beta[2][j] + f.const[j]));
  const G = grid('ra-chart', [{ r: 0, c: 0, t: 'US Treasury yields (% per year)', x: DATE }, { r: 0, c: 1, t: 'Equilibrium errors β̂ᵀy_t of the VECM', x: DATE }], { cols: 2, h: 340, legendRows: 2 });
  const [a, b] = G.ax, sg = v => (v < 0 ? '−' : '+') + nf(Math.abs(v), 2).replace('−', '');
  G.lay.shapes.push(hl(b, 0, K.navy, 'solid', 0.6));
  plot('ra-chart', [ln(d, Yv.map(r => r[0]), a, K.blue, '1-year', { line: { width: 0.9 } }), ln(d, Yv.map(r => r[1]), a, K.green, '5-year', { line: { width: 0.9 } }),
    ln(d, Yv.map(r => r[2]), a, K.red, '10-year', { line: { width: 0.9 } }),
    ln(d, Yv.map(r => -(r[0] + f.const[0]) / f.beta[2][0]), a, K.purple, '10-year implied by the 1-year and the first relation', { line: { color: K.purple, width: 0.8, dash: 'dot' } }),
    ln(d, z[0], b, K.purple, `1y ${sg(f.beta[2][0])} 10y + const.`, { line: { width: 0.8 } }), ln(d, z[1], b, K.orange, `5y ${sg(f.beta[2][1])} 10y + const.`, { line: { width: 0.8 } })], G.lay);
  const ad = z.map(s => adfuller(s, 'c'));
  out('ra-out', [T.ra1(ymLabel(e), k, f.n, nf(f.beta[2][0], 3), nf(f.const[0], 2), nf(f.beta[2][1], 3), nf(f.const[1], 2)),
    T.ra2(pv(ad[0].p), pv(ad[1].p), nf(f.ev[0], 3), nf(f.ev[1], 3), nf(halfLife(f.ev[0]), 1), nf(halfLife(f.ev[1]), 1))]);
}
function blkIrf() {
  const e = $('ir-e').value, s = +$('ir-s').value, k = +$('ir-k').value, f = vfit(e, k), h = range(0, VM.H, 1), nm = ['1-year', '5-year', '10-year'];
  const G = grid('ir-chart', [{ r: 0, c: 0, t: `Response (percentage points) to a one-standard-deviation shock to the ${nm[s]} yield`, xt: 'months' }], { cols: 1, h: 340, legendRows: 1 });
  const a = G.ax[0];
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.6));
  plot('ir-chart', [0, 1, 2].map(i => ln(h, f.irf.map(M => M[i][s]), a, [K.blue, K.green, K.red][i], nm[i] + ' yield', { line: { color: [K.blue, K.green, K.red][i], width: 1.6 } })), G.lay);
  const r = hh => [0, 1, 2].map(i => nf(f.irf[hh][i][s], 2));
  out('ir-out', [T.ir1(ymLabel(e), k, T.shock[s], r(0), r(12), r(VM.H))]);
}

// ================================================================= 8. VECM against a VAR in differences, out of sample
const FC = D.fc;
function fcStats(y0) {
  const ok = FC.o.map((o, i) => (+o.slice(0, 4) >= y0 ? i : -1)).filter(i => i >= 0), H = FC.H;
  const rm = (m, f) => range(0, H - 1, 1).map(h => Math.sqrt(mean(ok.map(i => f(FC.e[m][i][h]) ** 2))));
  const R = {}; ['vecm', 'dvar', 'rw'].forEach(m => { R[m] = [0, 1, 2].map(j => rm(m, v => v[j])); R[m].push(rm(m, v => v[2] - v[0])); });
  return { R, n: ok.length, first: FC.o[ok[0]] };
}
function blkFc() {
  const y0 = +$('fc-y').value; set('fc-y', y0);
  const S = fcStats(y0), hs = range(1, FC.H, 1), ttl_ = ['1-year: RMSE / RW', '5-year: RMSE / RW', '10-year: RMSE / RW', '10y - 1y spread: RMSE / RW'];
  const G = grid('fc-chart', ttl_.map((t, j) => ({ r: 0, c: j, t, xt: 'horizon (months)' })), { cols: 4, h: 320, gx: 0.05, legendRows: 1 });
  const tr_ = [];
  G.ax.forEach((a, j) => { tr_.push(ln(hs, S.R.vecm[j].map((v, i) => v / S.R.rw[j][i]), a, K.red, j ? null : 'VECM (rank 2)', { line: { color: K.red, width: 1.5 } }),
    ln(hs, S.R.dvar[j].map((v, i) => v / S.R.rw[j][i]), a, K.blue, j ? null : 'VAR in differences', { line: { color: K.blue, width: 1.5 } })); G.lay.shapes.push(hl(a, 1, K.navy, 'dash', 0.9)); });
  tr_.push(leg('random walk = 1', { color: K.navy, dash: 'dash', width: 1 }));
  plot('fc-chart', tr_, G.lay);
  const r = (m, j, h) => S.R[m][j][h - 1] / S.R.rw[j][h - 1];
  out('fc-out', [T.fc1(S.n, T.my(S.first + '-01'), nf(r('vecm', 0, 6), 2), nf(r('vecm', 1, 6), 2), nf(r('dvar', 0, 12), 2)), T.fc2(nf(r('vecm', 3, 24), 2), nf(r('vecm', 3, 36), 2), nf(r('dvar', 3, 24), 2), nf(r('dvar', 3, 36), 2))]);
  const h = +$('fc-h').value; set('fc-h', h);
  table('fc-tb', [L.series, { v: 'VECM / RW', cls: 'r' }, { v: 'VAR Δ / RW', cls: 'r' }, { v: 'RMSE RW', cls: 'r' }],
    [0, 1, 2, 3].map(j => [T.fcName[j], num(r('vecm', j, h), 2, r('vecm', j, h) < r('dvar', j, h)), num(r('dvar', j, h), 2, r('dvar', j, h) < r('vecm', j, h)), num(S.R.rw[j][h - 1], 2)]),
    { cap: T.fcCap(h, S.n) });
}

// ================================================================= 9. PPP for the leu
function pppState(y0) {
  const i0 = idxFrom(PPD.d, `${y0}-01-01`), s = PPD.s.slice(i0), rel = PPD.rel.slice(i0), q = s.map((v, t) => v - rel[t]);
  const J = johansen(s.map((v, t) => [v, PPD.p[i0 + t], PPD.ps[i0 + t]]), 0, NB.ppp.johansen.k_ar_diff);
  return { d: PPD.d.slice(i0), s, rel, q, aq: adfuller(q, 'c'), aqt: adfuller(q, 'ct'), eg: egTest(s, rel), J };
}
function blkPpp() {
  const y0 = +$('pp-y').value; set('pp-y', y0);
  const S = pppState(y0), G = grid('pp-chart', [{ r: 0, c: 0, t: `100 ln, ${T.my(S.d[0])} = 0`, x: DATE }, { r: 0, c: 1, t: `Real exchange rate (100 ln, ${T.my(S.d[0])} = 0)`, x: DATE }], { cols: 2, h: 340, legendRows: 2 });
  const [a, b] = G.ax, q0 = first(S.q);
  G.lay.shapes.push(hl(b, 0, K.navy, 'solid', 0.6));
  plot('pp-chart', [ln(S.d, first(S.s), a, K.green, 'EUR/RON, s_t'), ln(S.d, first(S.rel), a, K.red, 'relative prices, p_t - p*_t'), ln(S.d, q0, b, K.blue, 'real exchange rate, q_t = s_t - p_t + p*_t')], G.lay);
  const ds = S.s[S.s.length - 1] - S.s[0], dr = S.rel[S.rel.length - 1] - S.rel[0];
  out('pp-out', [T.pp1(T.my(S.d[0]), nf(ds, 1), nf(dr, 1), nf(ds - dr, 1)), T.pp2(pv(S.aq.p), pv(S.aqt.p), nf(S.eg.b, 2), pv(S.eg.p))]);
  const J = S.J;
  table('pp-tb', [L.test, { v: L.stat, cls: 'r' }, { v: 'p', cls: 'r' }, L.verdict],
    [[T.ppT.adf, num(S.aq.stat, 2), txt(pv(S.aq.p)), S.aq.p < 0.05 ? T.rej : T.norej], [T.ppT.adft, num(S.aqt.stat, 2), txt(pv(S.aqt.p)), S.aqt.p < 0.05 ? T.rej : T.norej],
     [T.ppT.eg(nf(S.eg.b, 2)), num(S.eg.tau, 2), txt(pv(S.eg.p)), S.eg.p < 0.05 ? T.rej : T.norej],
     [T.ppT.joh(J.k), num(J.trace[0], 1), txt('–'), T.ppJoh(nf(J.cvt[0][1], 2), J.rt)]],
    { cap: T.ppCap(T.my(S.d[0]), S.q.length) });
}

// ================================================================= 10. ROBOR and Euribor
function roeaState(y0) {
  const i0 = idxFrom(RE.d, `${y0}-01-01`), ro = RE.ro.slice(i0), ea = RE.ea.slice(i0);
  return { i0, d: RE.d.slice(i0), ro, ea, full: egTest(RE.ro, RE.ea), win: egTest(ro, ea), ecm: ecmFit(ro, ea, 1), sp: mean(ro.map((v, t) => v - ea[t])) };
}
function blkRoea() {
  const y0 = +$('re-y').value; set('re-y', y0);
  const S = roeaState(y0);
  const G = grid('re-chart', [{ r: 0, c: 0, t: 'Since 1995 (log scale; Euribor below 0.05 shown at 0.05)', x: DATE, y: { type: 'log' } }, { r: 0, c: 1, t: `Since ${y0} (% per year)`, x: DATE }], { cols: 2, h: 350, legendRows: 2 });
  const [a, b] = G.ax, cl = v => Math.max(v, 0.05), fade = col => ({ line: { color: `rgba(${RGB[col]},0.35)`, width: 0.9 } });
  G.lay.shapes.push(rect(a, S.d[0], RE.d[RE.d.length - 1], K.teal, 0.12), hl(b, 0, K.navy, 'solid', 0.6));
  plot('re-chart', [ln(RE.d, RE.ro, a, K.red, null, fade(K.red)), ln(RE.d, RE.ea.map(cl), a, K.blue, null, fade(K.blue)),
    ln(S.d, S.ro, a, K.red, 'ROBOR 3M (Romania)'), ln(S.d, S.ea.map(cl), a, K.blue, 'Euribor 3M (euro area)'),
    ln(S.d, S.ro, b, K.red, null), ln(S.d, S.ea, b, K.blue, null), ln(S.d, S.ro.map((v, t) => v - S.ea[t]), b, K.green, 'spread ROBOR - Euribor', { line: { color: K.green, width: 1, dash: 'dash' } })], G.lay);
  const po = D.roea.po[String(y0)];
  out('re-out', [T.re1(pv(S.full.p), pv(D.roea.po_full[1]), y0, pv(S.win.p), pv(po[1]), nf(S.win.b, 2), nf(S.sp, 2)), T.re2(nf(S.ecm.g, 3), nf(S.ecm.gt, 2), nf(S.ecm.half, 0))]);
  table('re-tb', [L.sample, { v: 'T', cls: 'r' }, { v: 'b̂', cls: 'r' }, { v: 'τ (k)', cls: 'r' }, { v: 'p', cls: 'r' }, { v: 'Z<sub>t</sub>', cls: 'r' }, { v: 'p', cls: 'r' }],
    [[T.reFull, txt(String(S.full.n)), num(S.full.b, 2), txt(`${nf(S.full.tau, 2)} (${S.full.lags})`), txt(pv(S.full.p), S.full.p < 0.05), num(D.roea.po_full[0], 2), txt(pv(D.roea.po_full[1]), D.roea.po_full[1] < 0.05)],
     [T.reWin(y0), txt(String(S.win.n)), num(S.win.b, 2), txt(`${nf(S.win.tau, 2)} (${S.win.lags})`), txt(pv(S.win.p), S.win.p < 0.05), num(po[0], 2), txt(pv(po[1]), po[1] < 0.05)]],
    { cap: T.reCap });
}

// ================================================================= 11. three Central European currencies
function ceeState(y0, k) {
  const i0 = idxFrom(FXD.d, `${y0}-01-01`), d = FXD.d.slice(i0), Yv = d.map((_, t) => [FXD.ron[t + i0], FXD.huf[t + i0], FXD.pln[t + i0]]);
  return { d, Yv, j: johansen(Yv, 0, k), adf: [0, 1, 2].map(c => adfuller(Yv.map(r => r[c]), 'c')) };
}
function johBars(G, a, j, showLeg) {
  const xs = ['r = 0', 'r ≤ 1', 'r ≤ 2'], tr_ = [];
  tr_.push(bar(xs, j.trace, a, K.blue, showLeg ? 'trace statistic' : null, { offsetgroup: 'a' }), bar(xs, j.maxeig, a, K.amber, showLeg ? 'max-eigenvalue statistic' : null, { offsetgroup: 'b' }),
    { x: xs, y: j.cvt.map(c => c[1]), type: 'scatter', mode: 'markers', marker: { symbol: 'line-ew-open', size: 26, color: K.red, line: { width: 3, color: K.red } }, name: showLeg ? '5% critical value' : '', showlegend: !!showLeg, xaxis: a.xa, yaxis: a.ya, offsetgroup: 'a', hovertemplate: '%{y:.2f}<extra></extra>' },
    { x: xs, y: j.cvm.map(c => c[1]), type: 'scatter', mode: 'markers', marker: { symbol: 'line-ew-open', size: 26, color: K.red, line: { width: 3, color: K.red } }, showlegend: false, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{y:.2f}<extra></extra>' });
  return tr_;
}
function blkCee() {
  const y0 = +$('cx-y').value, k = +$('cx-k').value; set('cx-y', y0); set('cx-k', k);
  const S = ceeState(y0, k);
  const G = grid('cx-chart', [{ r: 0, c: 0, cs: 2, t: `Three Central European currencies against the euro (100 ln, ${T.my(S.d[0])} = 0)`, x: DATE }, { r: 0, c: 2, t: 'Johansen tests' }], { cols: 3, h: 340, gx: 0.08, legendRows: 2 });
  const [a, b] = G.ax, nm = ['EUR/RON', 'EUR/HUF', 'EUR/PLN'], cols = [K.green, K.orange, K.purple];
  G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.6)); G.lay.barmode = 'group';
  plot('cx-chart', [...[0, 1, 2].map(c => ln(S.d, first(S.Yv.map(r => r[c])), a, cols[c], nm[c])), ...johBars(G, b, S.j, true)], G.lay);
  const ch = [0, 1, 2].map(c => S.Yv[S.Yv.length - 1][c] - S.Yv[0][c]);
  out('cx-out', [T.cx1(T.my(S.d[0]), ch.map(v => nf(v, 1)), S.adf.map(r => pv(r.p))), T.cx2(k, S.j.T, nf(S.j.trace[0], 1), nf(S.j.cvt[0][1], 2), nf(S.j.trace[1], 1), S.j.rt)]);
}

// ================================================================= 12. Banca Transilvania and BRD
function pairState(y0, band_) {
  const i0 = idxFrom(BK.d, `${y0}-01-01`), d = BK.d.slice(i0), A = BK.tlv.slice(i0), B = BK.brd.slice(i0), r = ols1(A, B);
  const m = mean(r.e), s = sd1(r.e), z = r.e.map(v => (v - m) / s);
  const fit = olsFit(r.e.slice(1), t => [1, r.e[t]], 2), rho = fit.b[1];
  let cr = 0; for (let t = 1; t < z.length; t++) if (Math.sign(z[t]) !== Math.sign(z[t - 1])) cr++;
  // the in-sample rule of fig_pairs_backtest: full-sample beta, mean and sd, entry at |z| > band, exit at 0, cost 0.20%
  const LA = A.map(v => v / 100), LB = B.map(v => v / 100), r0 = ols1(LA, LB), e0 = r0.e, m0 = mean(e0), s0 = sd1(e0), z0 = e0.map(v => (v - m0) / s0);
  let pos = 0; const rets = [];
  for (let i = 1; i < z0.length; i++) {
    rets.push(pos * ((LA[i] - LA[i - 1]) - r0.b * (LB[i] - LB[i - 1])) / (1 + Math.abs(r0.b)));
    let nw = pos;
    if (pos === 0 && z0[i] > band_) nw = -1; else if (pos === 0 && z0[i] < -band_) nw = 1; else if ((pos === 1 && z0[i] >= 0) || (pos === -1 && z0[i] <= 0)) nw = 0;
    if (nw !== pos) rets[rets.length - 1] -= 0.002 * Math.abs(nw - pos);
    pos = nw;
  }
  const mu = mean(rets) * 252, vol = sd1(rets) * Math.sqrt(252);
  return { d, A, B, r, z, rho, half: halfLife(rho), out: mean(z.map(v => (Math.abs(v) > band_ ? 1 : 0))), cr, ins: { mean: 100 * mu, sharpe: mu / vol } };
}
function blkPair() {
  const y0 = +$('pr-y').value, bnd = +$('pr-z').value; set('pr-y', y0); set('pr-z', nf(bnd, 2));
  const S = pairState(y0, bnd);
  const G = grid('pr-chart', [{ r: 0, c: 0, t: '100 ln adjusted price, start = 0', x: DATE }, { r: 0, c: 1, t: `Spread TLV - ${nf(S.r.b, 2)} BRD, standardised`, x: DATE }], { cols: 2, h: 340, legendRows: 2 });
  const [a, b] = G.ax;
  G.lay.shapes.push(hl(b, bnd, K.red, 'dash', 0.9), hl(b, -bnd, K.red, 'dash', 0.9), hl(b, 0, K.navy, 'solid', 0.6));
  plot('pr-chart', [ln(S.d, first(S.A), a, K.red, 'Banca Transilvania (TLV)', { line: { width: 1 } }), ln(S.d, first(S.B), a, K.blue, 'BRD', { line: { width: 1 } }),
    ln(S.d, S.z, b, K.purple, 'spread z-score (full sample)', { line: { width: 0.8 } }), leg(`entry bands ±${nf(bnd, 2)}`, { color: K.red, dash: 'dash', width: 1 })], G.lay);
  out('pr-out', [T.pr1(T.dmy(S.d[0]), S.d.length, nf(S.r.b, 2), nf(S.rho, 3), nf(S.half, 0), nf(100 * S.out, 1), nf(bnd, 2), S.cr), T.pr2(nf(bnd, 2), nf(S.ins.mean, 2), nf(S.ins.sharpe, 2))]);
}

// ================================================================= 13. pairs trading out of sample
const PB = D.pairs;
function btState(entry, mult) {
  const res = {};
  ['bvb', 'us'].forEach(mk => {
    const r = PB.r[mk][entry], c = PB.cost[mk] * mult, d = days(r.d0, r.gap), net = r.g.map((v, i) => v - c * r.t[i]);
    const perf = x => { const m = mean(x) * 252, s = sd1(x) * Math.sqrt(252); return { mean: 100 * m, sharpe: m / s, cum: cumsum(x.map(v => 100 * Math.log1p(v))) }; };
    const tn = r.trades.map(v => v - 2 * c), drag = (mean(r.g) - mean(net)) / Math.max(c, 1e-12);
    res[mk] = { d, g: perf(r.g), n: perf(net), trades: r.opens, win: mean(tn.map(v => (v > 0 ? 1 : 0))), mt: mean(tn), be: c > 0 ? mean(r.g) / drag : mean(r.g) / mean(r.t), c, info: r };
  });
  return res;
}
function blkBt() {
  const en = $('pb-e').value, mult = +$('pb-c').value; set('pb-c', nf(mult, 2) + '×');
  const S = btState(en, mult);
  const G = grid('pb-chart', [{ r: 0, c: 0, t: `Bucharest: 8 shares, 28 pairs (cost ${nf(100 * S.bvb.c, 2)}%)`, x: DATE, yt: 'cumulative log growth (%)' }, { r: 0, c: 1, t: `US banks: 6 shares, 15 pairs (cost ${nf(100 * S.us.c, 2)}%)`, x: DATE }], { cols: 2, h: 340, ml: 56, legendRows: 1 });
  const [a, b] = G.ax, tr_ = [];
  [[a, S.bvb], [b, S.us]].forEach(([ax, s], k) => { tr_.push(ln(s.d, s.g.cum, ax, K.blue, k ? null : 'gross of costs'), ln(s.d, s.n.cum, ax, K.red, k ? null : 'net of costs')); G.lay.shapes.push(hl(ax, 0, K.navy, 'solid', 0.6)); });
  plot('pb-chart', tr_, G.lay);
  const o = s => [nf(s.g.mean, 2), nf(s.g.sharpe, 2), nf(s.n.mean, 2), nf(s.n.sharpe, 2), s.trades, nf(100 * s.win, 1), nf(100 * s.mt, 2), nf(100 * s.be, 2)];
  out('pb-out', [T.pb1(nf(+en, 1), nf(100 * S.bvb.c, 2), S.bvb.info.windows, S.bvb.info.windows_with_pairs, ...o(S.bvb)), T.pb2(nf(100 * S.us.c, 2), ...o(S.us))]);
}

// ================================================================= EXTRA. worked example: common stochastic trends
function blkTrends() {
  const a = +$('ct-a').value, c = +$('ct-c').value; set('ct-a', nf(a, 1)); set('ct-c', nf(c, 1));
  const n = DR.T, x = cumsum(DR.u), u = DR.w, v = DR.v, y = x.map((s, t) => a * s + u[t]), z = x.map((s, t) => c * s + v[t]), t = range(0, n - 1, 1);
  const b2 = -a / c, z1 = y.map((s, i) => s - a * x[i]), z2 = y.map((s, i) => s + b2 * z[i]), z3 = y.map((s, i) => s - x[i]);
  const G = grid('ct-chart', [{ r: 0, c: 0, t: 'Three I(1) series with one common trend', xt: 't' }, { r: 0, c: 1, t: 'Combinations β′y_t', xt: 't' }], { cols: 2, h: 340, legendRows: 3 });
  const [A, B] = G.ax;
  G.lay.shapes.push(hl(B, 0, K.navy, 'solid', 0.6));
  plot('ct-chart', [ln(t, x, A, K.blue, 'x_t (random walk)'), ln(t, y, A, K.red, `y_t = ${nf(a, 1)} x_t + u_t`), ln(t, z, A, K.green, `z_t = ${nf(c, 1)} x_t + v_t`),
    ln(t, z1, B, K.purple, `β1 = (${nf(-a, 1)}, 1, 0)`, { line: { width: 1 } }), ln(t, z2, B, K.orange, `β2 = (0, 1, ${nf(b2, 2)})`, { line: { width: 1 } }),
    ln(t, z3, B, K.amber, 'y_t - x_t (not a cointegrating vector)', { line: { width: 1, dash: 'dot' } })], G.lay);
  const ad = [z1, z2, z3].map(s => adfuller(s, 'c')), last = n - 1;
  steps('ct-ws', [T.cts1(nf(a, 1), nf(c, 1), nf(-a, 1), nf(b2, 2)), T.cts2(nf(y[last], 2), nf(x[last], 2), nf(z[last], 2), nf(-a, 1), nf(z1[last], 2), nf(b2, 2), nf(z2[last], 2)),
    T.cts3(nf(sdp(z1), 2), nf(sdp(z2), 2), nf(sdp(z3), 1), pv(ad[0].p), pv(ad[1].p), pv(ad[2].p), a === 1)]);
  out('ct-out', [T.ct1(nf(-a, 1), nf(b2, 2), pv(ad[0].p), pv(ad[1].p), pv(ad[2].p))]);
}

// ================================================================= EXTRA. critical values of the Engle-Granger test
function blkCrit() {
  const Tv = $('ec-T').value, nobs = Tv === 'inf' ? Infinity : +Tv, ns = [1, 2, 3, 4];
  const cv = reg => ns.map(n => mackinnoncrit(n, reg, nobs));
  const C = cv('c'), CT = cv('ct');
  const G = grid('ec-chart', [{ r: 0, c: 0, t: `5% critical values, T = ${Tv === 'inf' ? '∞' : Tv}`, xt: 'variables n (1: Dickey-Fuller)' }], { cols: 1, h: 320, legendRows: 1 });
  const a = G.ax[0];
  G.lay.barmode = 'group';
  plot('ec-chart', [bar(ns.map(String), C.map(r => r[1]), a, K.blue, 'constant', { hovertemplate: '%{y:.3f}<extra></extra>' }), bar(ns.map(String), CT.map(r => r[1]), a, K.red, 'constant and trend', { hovertemplate: '%{y:.3f}<extra></extra>' })], G.lay);
  table('ec-tb', [L.vars, '1%', '5%', '10%', '1%', '5%', '10%'].map((h, i) => (i ? { v: h, cls: 'r' } : h)),
    ns.map((n, i) => [n === 1 ? T.edOne : String(n), ...C[i].map(v => num(v, 2)), ...CT[i].map(v => num(v, 2))]),
    { head2: ['', { v: T.cstOnly, span: 3, cls: 'r' }, { v: T.cstTrend, span: 3, cls: 'r' }], cap: T.ecCap(Tv === 'inf' ? '∞' : Tv) });
  out('ec-out', [T.ec1(Tv === 'inf' ? '∞' : Tv, nf(C[0][1], 2), nf(C[1][1], 2), nf(C[2][1], 2), nf(C[3][1], 2))]);
}

// ================================================================= EXTRA. worked example: US consumption and income
function blkWxEg() {
  const e = +$('we-e').value, kk = $('we-k').value; set('we-e', e);
  const n = idxTo(MAC.d, `${e}-12-31`) + 1, d = MAC.d.slice(0, n), c = MAC.c.slice(0, n), y = MAC.y.slice(0, n);
  const r = egTest(c, y, kk === 'aic' ? undefined : +kk), dfc = mackinnoncrit(1, 'c', n - 1)[1];
  const G = grid('we-chart', [{ r: 0, c: 0, t: 'Step 1 residuals û_t', x: DATE }, { r: 0, c: 1, t: 'Step 2: τ against the 5% critical values' }], { cols: 2, h: 320, legendRows: 1 });
  const [A, B] = G.ax;
  G.lay.shapes.push(hl(A, 0, K.navy, 'solid', 0.6));
  plot('we-chart', [ln(d, r.e, A, K.purple, null, { line: { width: 1 } }),
    bar(['τ', 'Engle-Granger, n = 2', 'Dickey-Fuller, n = 1'], [r.tau, r.crit5, dfc], B, [K.purple, K.red, K.blue], null, { hovertemplate: '%{y:.3f}<extra></extra>' })], G.lay);
  const lec = n === MAC.d.length && kk === 'aic';
  steps('we-ws', [T.wes1(T.q(d[0]), T.q(d[n - 1]), n, nf(r.a, 1), nf(r.b, 3), nf(r.r2, 4), nf(r.dw, 2)),
    T.wes2(r.lags, nf(r.gamma, 4), nf(r.se, 4), nf(r.tau, 2)),
    T.wes3(nf(r.tau, 2), nf(r.crit5, 2), nf(dfc, 2), pv(r.p), r.tau < r.crit5, r.tau < dfc, lec ? [nf(NB.egtab['US consumption on income'].po_zt, 2), pv(NB.egtab['US consumption on income'].po_p)] : null)]);
  out('we-out', [T.we1(T.q(d[n - 1]), nf(r.b, 3), nf(r.tau, 2), r.lags, nf(r.crit5, 2), pv(r.p))]);
}

// ================================================================= EXTRA. Engle-Granger and Phillips-Ouliaris on real data
const ETR = ['cons', 'rates', 'tlv', 'brd', 'tlv10', 'fx', 'ro'];
function etData(key) {
  const P14 = { d: BK.d.slice(BK.i14), a: BK.tlv.slice(BK.i14), b: BK.brd.slice(BK.i14) };
  return { cons: [MAC.c, MAC.y, MAC.d], rates: [YD.gs10, YD.tb3, YD.d], tlv: [P14.a, P14.b, P14.d], brd: [P14.b, P14.a, P14.d], tlv10: [BK.tlv, BK.brd, BK.d],
    fx: [FXD.ron, FXD.huf, FXD.d], ro: [ROC.c, ROC.y, ROC.d] }[key];
}
function etRow(key, k) {
  const [y, x, d] = etData(key), i0 = dropK(y.length, k);
  return Object.assign(egTest(y.slice(i0), x.slice(i0)), { d0: d[i0], po: D.egtab.po[key][D.egtab.drop.indexOf(k)] });
}
function blkEt() {
  const k = +$('et-k').value; set('et-k', k + '%');
  const R = ETR.map(key => [key, memo(`et${key}${k}`, () => etRow(key, k))]);
  const G = grid('et-chart', [{ r: 0, c: 0, t: 'Engle-Granger τ and its 5% critical value' }], { cols: 1, h: 330, ml: 60, legendRows: 1 });
  const a = G.ax[0], xs = R.map(([key]) => T.etShort[key]);
  plot('et-chart', [bar(xs, R.map(([, r]) => r.tau), a, R.map(([, r]) => (r.p < 0.05 ? K.red : K.blue)), 'τ (red: p < 0.05)', { hovertemplate: '%{y:.2f}<extra></extra>' }),
    { x: xs, y: R.map(([, r]) => r.crit5), type: 'scatter', mode: 'markers', marker: { symbol: 'line-ew-open', size: 30, color: K.navy, line: { width: 3, color: K.navy } }, name: '5% critical value (2 variables)', xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{y:.2f}<extra></extra>' }], G.lay);
  table('et-tb', [L.regression, L.freq, { v: 'T', cls: 'r' }, { v: 'b̂', cls: 'r' }, { v: 'τ (k)', cls: 'r' }, { v: 'p', cls: 'r' }, { v: 'Z<sub>t</sub>', cls: 'r' }, { v: 'p', cls: 'r' }],
    R.map(([key, r]) => [T.etName[key], T.etFreq[key], txt(String(r.n)), num(r.b, 2), txt(`${nf(r.tau, 2)} (${r.lags})`), txt(pv(r.p), r.p < 0.05), num(r.po[0], 2), txt(pv(r.po[1]), r.po[1] < 0.05)]),
    { cap: T.etCap(k) });
  const fx = R[5][1], ro = R[6][1], tl = R[4][1];
  out('et-out', [T.et1(k, R.filter(([, r]) => r.p < 0.05).length, nf(tl.tau, 2), pv(tl.p), nf(ro.tau, 2), pv(ro.p), pv(ro.po[1]))]);
  // worked example: the residual ADF of the selected row
  const key = $('et-r').value, r = R.find(x => x[0] === key)[1], dfc = mackinnoncrit(1, 'c', r.n - 1)[1];
  steps('et-ws', [T.ets1(T.etName[key], r.n, nf(r.b, 3), r.lags), T.ets2(nf(r.gamma, 5), nf(r.se, 5), nf(r.tau, 2)),
    T.ets3(nf(r.tau, 2), nf(r.crit5, 2), nf(dfc, 2), pv(r.p), r.tau < r.crit5, r.tau < dfc)]);
}

// ================================================================= EXTRA. speed of adjustment and half-life
function blkHalf() {
  const g0 = +$('wh-g').value, unit = $('wh-u').value; set('wh-g', nf(g0, 2));
  const per = unit === 'm' ? 12 : 4, H = Math.min(120, Math.max(24, Math.ceil(4 * halfLife(1 + g0)))), h = range(0, H, 1), hh = halfLife(1 + g0);
  const xs = h.map(v => v / per), sim = []; let u = 3;
  for (let t = 0; t <= H; t++) { sim.push(u); u = (1 + g0) * u + 0.3 * DR.w[t + 1]; }
  const G = grid('wh-chart', [{ r: 0, c: 0, t: 'A unit deviation after h periods: (1+γ)^h', xt: 'years' }, { r: 0, c: 1, t: 'An equilibrium error u_t = (1+γ)u_(t-1) + shock, from u_0 = 3', xt: 'years' }], { cols: 2, h: 330, legendRows: 1 });
  const [A, B] = G.ax;
  G.lay.shapes.push(hl(A, 0.5, K.navy, 'dot', 0.9), vl(A, hh / per, K.red, 'dash', 1), hl(B, 0, K.navy, 'solid', 0.6));
  plot('wh-chart', [ln(xs, h.map(v => (1 + g0) ** v), A, K.blue, `(1 + γ)^h, γ = ${nf(g0, 2)}`), ln(xs, sim, B, K.purple, 'simulated equilibrium error (extra run, the chapter draws)', { line: { width: 1 } }),
    ln(xs, h.map(v => 3 * (1 + g0) ** v), B, K.blue, null, { line: { color: K.blue, width: 1, dash: 'dash' } })], G.lay);
  steps('wh-ws', [T.whs1(nf(g0, 2), nf(1 + g0, 2), nf(Math.log(1 + g0), 4), nf(hh, 2)), T.whs2(nf(hh, 2), T.unit[unit], nf(hh / per, 2)), T.whs3(nf((1 + g0) ** 1, 3), nf((1 + g0) ** 4, 3), nf((1 + g0) ** 12, 3))]);
  out('wh-out', [T.wh1(nf(g0, 2), nf(hh, 1), T.unit[unit], nf(hh / per, 2))]);
}

// ================================================================= EXTRA. worked example: one quarter of consumption
function blkWxEcm() {
  const u0 = +$('wc-u').value, dy = +$('wc-g').value; set('wc-u', nf(u0, 1)); set('wc-g', nf(dy, 1));
  const f = memo('ecmcons', () => ecmFit(MAC.c, MAC.y, 1)), sr = f.d0 * dy, ec = f.g * u0, tot = f.c + sr + ec;
  const G = grid('wc-chart', [{ r: 0, c: 0, t: 'Predicted growth of consumption (%): the parts' }, { r: 0, c: 1, t: 'The deviation from equilibrium in the next quarters', xt: 'quarters' }], { cols: 2, h: 320, legendRows: 1 });
  const [A, B] = G.ax, h = range(0, 24, 1);
  G.lay.shapes.push(hl(A, 0, K.navy, 'solid', 0.6), hl(B, 0, K.navy, 'solid', 0.6));
  plot('wc-chart', [bar(['constant', 'δ0 Δy_t', 'γ û_(t-1)', 'Δc_t'], [f.c, sr, ec, tot], A, [K.amber, K.green, K.red, K.blue], null, { hovertemplate: '%{y:.3f}<extra></extra>' }),
    ln(h, h.map(v => u0 * (1 + f.g) ** v), B, K.purple, `û_h = (1 + γ̂)^h û_0, half-life ${nf(f.half, 1)} quarters`)], G.lay);
  steps('wc-ws', [T.wcs1(nf(f.c, 3), nf(f.d0, 3), nf(f.g, 3)), T.wcs2(nf(f.d0, 3), pn(dy, 1), nf(sr, 3), nf(f.g, 3), pn(u0, 1), nf(ec, 3)),
    T.wcs3(nf(f.c, 3), nf(sr, 3), nf(ec, 3), nf(tot, 2), nf(f.c + sr, 2))]);
  out('wc-out', [T.wc1(nf(u0, 1), nf(dy, 1), nf(tot, 2), nf(f.c + sr, 2))]);
}

// ================================================================= EXTRA. worked example: a bivariate VECM
function blkWxVecm() {
  const a1 = +$('wv-a1').value, a2 = +$('wv-a2').value; set('wv-a1', nf(a1, 2)); set('wv-a2', nf(a2, 2));
  const n = 200, y1 = [0], y2 = [3];
  for (let t = 1; t < n; t++) { const z = y1[t - 1] - y2[t - 1]; y1.push(y1[t - 1] + a1 * z + DR.u[t]); y2.push(y2[t - 1] + a2 * z + DR.w[t]); }
  const t = range(0, n - 1, 1), z = y1.map((v, i) => v - y2[i]), av = y1.map((v, i) => (v + y2[i]) / 2), rho = 1 + a1 - a2;
  const G = grid('wv-chart', [{ r: 0, c: 0, t: 'y1, y2 and their average', xt: 't' }, { r: 0, c: 1, t: 'The gap z_t = y1_t - y2_t', xt: 't' }], { cols: 2, h: 320, legendRows: 2 });
  const [A, B] = G.ax;
  G.lay.shapes.push(hl(B, 0, K.navy, 'solid', 0.6));
  const cap = v => Math.max(-1e3, Math.min(1e3, v));
  plot('wv-chart', [ln(t, y1.map(cap), A, K.blue, 'y1'), ln(t, y2.map(cap), A, K.red, 'y2'), ln(t, av.map(cap), A, K.green, 'average (y1 + y2)/2', { line: { width: 1, dash: 'dot' } }),
    ln(t, z.map(cap), B, K.purple, 'gap (extra run, the chapter draws)', { line: { width: 1 } })], G.lay);
  const hh = halfLife(rho), st = rho > -1 && rho < 1;
  steps('wv-ws', [T.wvs1(nf(a1, 2), nf(a2, 2), nf(a1, 2), nf(-a1, 2), nf(a2, 2), nf(-a2, 2)), T.wvs2(nf(a1, 2), nf(a2, 2), nf(a1 - a2, 2), nf(rho, 2), st),
    T.wvs3(st, isFinite(hh) ? nf(hh, 1) : '∞', a1 * a2 <= 0 && a1 <= 0 && a2 >= 0)]);
  out('wv-out', [T.wv1(nf(rho, 2), st, isFinite(hh) && st ? nf(hh, 1) : '∞', nf(sdp(z), 2))]);
}

// ================================================================= EXTRA. worked example: from eigenvalues to statistics
const yieldsY = (i0, i1) => { const o = []; for (let t = i0; t < i1; t++) o.push([YD.gs1[t], YD.gs5[t], YD.gs10[t]]); return o; };
function blkWxJoh() {
  const k = +$('wj-k').value, e = +$('wj-e').value; set('wj-k', k); set('wj-e', e);
  const n = idxTo(YD.d, `${e}-12-31`) + 1, j = memo(`joh0|${n}|${k}|0`, () => johansen(yieldsY(0, n), 0, k));
  const G = grid('wj-chart', [{ r: 0, c: 0, t: `US yields 1y, 5y, 10y, 1960 - ${e}: trace and max-eigenvalue statistics`, xt: 'null hypothesis' }], { cols: 1, h: 330, legendRows: 2 });
  G.lay.barmode = 'group';
  plot('wj-chart', johBars(G, G.ax[0], j, true), G.lay);
  const ml = j.eig.map(l => -Math.log(1 - l));
  steps('wj-ws', [T.wjs1(n, k, j.T, j.eig.map(l => nf(l, 4)), ml.map(v => nf(v, 4))),
    T.wjs2(j.T, ml.map(v => nf(v, 4)), nf(j.trace[0], 1), nf(j.cvt[0][1], 2), nf(j.maxeig[0], 1), nf(j.cvm[0][1], 2)),
    T.wjs3(nf(j.trace[1], 1), nf(j.cvt[1][1], 2), nf(j.trace[2], 1), nf(j.cvt[2][1], 2), j.rt)]);
  out('wj-out', [T.wj1(T.my(YD.d[n - 1]), k, j.T, nf(j.trace[0], 1), nf(j.trace[1], 1), nf(j.trace[2], 1), j.rt, j.rm)]);
}

// ================================================================= EXTRA. Johansen tests on three systems
const JLEC = { yields: 2, macro: 1, fx: 1 };
function jtState(kk) {
  const kOf = s => (kk === 'bic' ? JLEC[s] : +kk);
  return {
    yields: memo(`joh0|${YD.d.length}|${kOf('yields')}|0`, () => johansen(yieldsY(0, YD.d.length), 0, kOf('yields'))),
    macro: memo(`jmac|${kOf('macro')}`, () => johansen(MAC.y.map((v, t) => [v, MAC.c[t], MAC.i[t]]), 0, kOf('macro'))),
    fx: memo(`jfx|${kOf('fx')}`, () => johansen(FXD.ron.map((v, t) => [v, FXD.huf[t], FXD.pln[t]]), 0, kOf('fx'))), kOf };
}
function blkJt() {
  const kk = $('jt-k').value, S = jtState(kk), sys = ['yields', 'macro', 'fx'];
  const G = grid('jt-chart', sys.map((s, i) => ({ r: 0, c: i, t: { yields: 'US yields', macro: 'US y, c, i', fx: 'RON, HUF, PLN' }[s] + ` (k = ${S.kOf(s)})` })), { cols: 3, h: 320, gx: 0.06, legendRows: 2 });
  G.lay.barmode = 'group';
  plot('jt-chart', sys.flatMap((s, i) => johBars(G, G.ax[i], S[s], i === 0)), G.lay);
  table('jt-tb', [L.system, { v: 'T', cls: 'r' }, { v: 'k', cls: 'r' }, ...['λ<sub>trace</sub>(0)', 'λ<sub>trace</sub>(1)', 'λ<sub>trace</sub>(2)', 'λ<sub>max</sub>(0)', 'λ<sub>max</sub>(1)'].map(v => ({ v, cls: 'r' })), { v: L.rank, cls: 'r' }],
    sys.map(s => { const j = S[s]; return [T.jtTitle[s], txt(String(j.n)), txt(String(j.k)), ...j.trace.map((v, i) => num(v, 1, v > j.cvt[i][1])), ...j.maxeig.slice(0, 2).map((v, i) => num(v, 1, v > j.cvm[i][1])), txt(String(j.rt), true)]; })
      .concat([[L.cv5, '', '', ...S.yields.cvt.map(c => num(c[1], 2)), ...S.yields.cvm.slice(0, 2).map(c => num(c[1], 2)), '']]),
    { cap: T.jtCap });
  out('jt-out', [T.jt1(sys.map(s => [T.jtTitle[s], S[s].k, nf(S[s].trace[0], 1), S[s].rt]))]);
}

// ================================================================= EXTRA. the deterministic case changes the answer
function blkJd() {
  const y0 = +$('jd-y').value, k = +$('jd-k').value; set('jd-y', y0); set('jd-k', k);
  const i0 = idxFrom(YD.d, `${y0}-01-01`), Yv = yieldsY(i0, YD.d.length), cs = [-1, 0, 1];
  const J = cs.map(c => memo(`joh${c}|${i0}|${k}`, () => johansen(Yv, c, k)));
  const G = grid('jd-chart', cs.map((c, i) => ({ r: 0, c: i, t: T.jdTitle[i] })), { cols: 3, h: 320, gx: 0.06, legendRows: 2 });
  G.lay.barmode = 'group';
  plot('jd-chart', J.flatMap((j, i) => johBars(G, G.ax[i], j, i === 0)), G.lay);
  table('jd-tb', [L.det, ...['λ<sub>trace</sub>(0)', '5%', 'λ<sub>trace</sub>(1)', '5%', 'λ<sub>trace</sub>(2)', '5%'].map(v => ({ v, cls: 'r' })), { v: L.rank, cls: 'r' }],
    J.map((j, i) => [T.jdName[i], ...j.trace.flatMap((v, r) => [num(v, 1, v > j.cvt[r][1]), num(j.cvt[r][1], 2)]), txt(String(j.rt), true)]), { cap: T.jdCap(T.my(YD.d[i0]), k, J[0].T) });
  out('jd-out', [T.jd1(T.my(YD.d[i0]), k, J.map(j => j.rt), nf(J[2].trace[2], 1), nf(J[2].cvt[2][1], 2))]);
}

// ================================================================= EXTRA. the adjustment coefficients
function blkAlpha() {
  const e = $('at-e').value, k = +$('at-k').value, f = vfit(e, k);
  const G = grid('at-chart', [{ r: 0, c: 0, t: 'α̂ on the first error (1y - b 10y), ±1.96 SE' }, { r: 0, c: 1, t: 'α̂ on the second error (5y - b 10y), ±1.96 SE' }], { cols: 2, h: 320, legendRows: 1 });
  const xs = ['Δ 1-year', 'Δ 5-year', 'Δ 10-year'];
  const tr_ = [0, 1].map(jj => ({ x: xs, y: [0, 1, 2].map(i => f.alpha[i][jj]), type: 'scatter', mode: 'markers', marker: { size: 9, color: [K.blue, K.green, K.red] }, xaxis: G.ax[jj].xa, yaxis: G.ax[jj].ya, showlegend: false,
    error_y: { type: 'data', array: [0, 1, 2].map(i => 1.96 * f.se[i][jj]), color: K.navy, thickness: 1.2, width: 6 }, hovertemplate: '%{y:.3f}<extra></extra>' }));
  G.ax.forEach(a => G.lay.shapes.push(hl(a, 0, K.navy, 'solid', 0.6)));
  plot('at-chart', tr_, G.lay);
  table('at-tb', [L.eq, { v: 'α̂<sub>·1</sub>', cls: 'r' }, { v: 't', cls: 'r' }, { v: 'α̂<sub>·2</sub>', cls: 'r' }, { v: 't', cls: 'r' }],
    [0, 1, 2].map(i => [T.atEq[i], num(f.alpha[i][0], 3), num(f.t[i][0], 2, Math.abs(f.t[i][0]) >= 1.96), num(f.alpha[i][1], 3), num(f.t[i][1], 2, Math.abs(f.t[i][1]) >= 1.96)]),
    { cap: T.atCap(ymLabel(e), k, f.n, nf(f.beta[2][0], 3), nf(f.beta[2][1], 3)) });
  const B = [[f.beta[0][0], f.beta[1][0], f.beta[2][0]], [f.beta[0][1], f.beta[1][1], f.beta[2][1]]];
  const M = [0, 1].map(i => [0, 1].map(jj => (i === jj ? 1 : 0) + B[i].reduce((s, v, q) => s + v * f.alpha[q][jj], 0))), ev = eig2(M);
  steps('at-ws', [T.ats1(M.map(r => r.map(v => nf(v, 3)))), T.ats2(nf(ev[0], 3), nf(ev[1], 3)), T.ats3(nf(ev[0], 3), nf(halfLife(ev[0]), 1), nf(ev[1], 3), nf(halfLife(ev[1]), 1))]);
  const ta = [0, 1, 2].map(i => Math.max(Math.abs(f.t[i][0]), Math.abs(f.t[i][1])));
  out('at-out', [T.at1(ymLabel(e), k, nf(ta[0], 2), nf(Math.min(Math.abs(f.t[1][0]), Math.abs(f.t[1][1]), Math.abs(f.t[2][0]), Math.abs(f.t[2][1])), 2), nf(Math.max(Math.abs(f.t[1][0]), Math.abs(f.t[1][1]), Math.abs(f.t[2][0]), Math.abs(f.t[2][1])), 2))]);
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
const BLK = [['drunk', 'dd', blkDrunk, 'dd-c', range(0, 0.5, 0.05), 600], ['examples', 'ex', blkExamples, 'ex-k', range(0, 60, 5), 700],
  ['eg-dist', 'ed', blkEgDist, 'ed-T', null, 1300], ['eg-steps', 'es', blkEgSteps, 'es-k', range(0, 60, 5), 700],
  ['ecm', 'em', blkEcm, 'em-e', range(1980, 2026, 2), 600], ['rates', 'ra', blkRates, 'ra-e', null, 1300], ['vecm-irf', 'ir', blkIrf, 'ir-e', null, 1300],
  ['forecast', 'fc', blkFc, 'fc-y', range(1990, 2015, 1), 600], ['ppp', 'pp', blkPpp, 'pp-y', range(2005, 2016, 1), 800], ['ro-ea', 're', blkRoea, 're-y', range(2005, 2018, 1), 800],
  ['cee-fx', 'cx', blkCee, 'cx-y', range(2005, 2016, 1), 800], ['pair', 'pr', blkPair, 'pr-y', range(2010, 2020, 1), 800], ['pairs-bt', 'pb', blkBt, 'pb-c', range(0, 3, 0.25), 600],
  ['wx-trends', 'ct', blkTrends, 'ct-a', range(0.5, 3, 0.5), 900], ['eg-crit', 'ec', blkCrit, 'ec-T', null, 1100], ['wx-eg', 'we', blkWxEg, 'we-e', range(1975, 2009, 2), 700],
  ['eg-table', 'et', blkEt, 'et-k', null, 1400], ['wx-half', 'wh', blkHalf, 'wh-g', range(-0.9, -0.05, 0.05), 500], ['wx-ecm', 'wc', blkWxEcm, 'wc-u', range(-4, 4, 0.5), 500],
  ['wx-vecm', 'wv', blkWxVecm, 'wv-a2', range(-0.3, 0.5, 0.05), 600], ['wx-joh', 'wj', blkWxJoh, 'wj-e', range(1990, 2026, 2), 700],
  ['joh-table', 'jt', blkJt, 'jt-k', null, 1400], ['joh-det', 'jd', blkJd, 'jd-y', range(1960, 2000, 4), 900], ['alpha-table', 'at', blkAlpha, 'at-e', null, 1300]];
function init() {
  placeExtras();
  const live = BLK.filter(b => has(b[1] + '-out')), players = {};
  live.forEach(([anchor, x, f, pid, vals, ms]) => {
    const blk = $(anchor) || $(x + '-out').closest('.blk, .wlive');
    (blk ? [...blk.querySelectorAll('input, select')] : []).forEach(el => el.addEventListener('input', f));
    if (!has(x + '-play')) return;
    const el = $(pid), v = vals || [...el.options].map(o => o.value);
    players[anchor] = new Player(x + '-play', el, v, ms);
  });
  if (T.predict && predict) Object.entries(T.predict).forEach(([b, sp]) => { if (has(b) && players[b]) predict(b, Object.assign({ player: players[b], veil: { out: true } }, sp)); });
  exitTicket(T.ticket);
  live.forEach(b => b[2]());
  let w = window.innerWidth, tm = null;
  window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(() => { if ((window.innerWidth < 720) !== (w < 720) || Math.abs(window.innerWidth - w) > 200) { w = window.innerWidth; live.forEach(b => b[2]()); } }, 250); });
}
window.CH7 = { YD, MAC, BK, FXD, PPD, RE, ROC, adfuller, egTest, johansen, mackinnoncrit, mackinnonp, ols1, ecmFit, shortFit, ecmState, acfv, halfLife, drunkStats, pairsData,
  fcStats, pppState, PPD, roeaState, ceeState, pairState, btState, etRow, jtState, vfit, eig2, yieldsY, BLK };
if (!window.CH7_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
