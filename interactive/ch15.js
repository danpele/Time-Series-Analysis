// Chapter 15 interactive page (course review: the Box-Jenkins workflow on the Romanian HICP and the eight review problems):
// one block per chart slide, plus the result tables and the worked problems of the lecture, computed in the browser from
// ch15_data.js (Quantlets/Ch_15/export_interactive.py): the HICP of the slides (Eurostat, January 2005 - August 2026), the 36
// SARIMA fits of step 4 (estimates and residuals of the chapter's code), the forecasts of step 6, and the data of Problems 2, 4
// and 5 (code of Chapters 3, 6 and 7). Correlograms, Ljung-Box, Jarque-Bera, accuracy measures, Diebold-Mariano, the OLS of the
// ROBOR equation and the hand computations of the problems follow generate_all_charts.py and build_chapter15.py line by line.
(function () {
'use strict';
const D = window.CH15_DATA;
const { nf, $, AX, layout, plot, Player, range, predict, exitTicket, normPpf, acf, pacf, solve } = window.TSAI;
const out = (id, items) => window.TSAI.out(id, items);
const T = window.CH15_TEXT;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of tsa_style.py (MainBlue, IDAred, Amber, Forest, DarkText)
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', orange: '#E67E22', purple: '#8E44AD', teal: '#17A2B8', navy: '#1F2A44' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.amber]: '181,133,63', [K.navy]: '31,42,68', [K.teal]: '23,162,184' };

// ---------------------------------------------------------------- small helpers
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
const has = id => !!$(id);
const val = id => +$(id).value;
const sel = id => $(id).value;
const cache = {};
const memo = (k, f) => (k in cache ? cache[k] : (cache[k] = f()));
const f0 = x => nf(x, 0), f1 = x => nf(x, 1), f2 = x => nf(x, 2), f3 = x => nf(x, 3), f4 = x => nf(x, 4);
const amean = a => { let s = 0; for (const v of a) s += v; return s / a.length; };
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const par = (x, d) => (x < 0 ? `(${nf(x, d)})` : nf(x, d));
const tr = (x, d) => { const v = +x.toFixed(d); return nf(v, (String(v).split('.')[1] || '').length); };   // no trailing zeros
const sgn = (x, d) => `${x < 0 ? '−' : '+'} ${nf(Math.abs(x), d)}`;
const pv = p => (p < 0.001 ? T.plt : f3(p));
const pvr = p => (p < 0.001 ? T.plt : '= ' + f3(p));
const my = s => T.my(s);
const argmax = a => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
const argmin = a => a.reduce((b, v, i) => (v < a[b] ? i : b), 0);

// ---------------------------------------------------------------- distributions (Numerical Recipes: gammln, gammp/gammq, betai)
function lgam(x) {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015; for (const v of c) s += v / ++y;
  return -t + Math.log(2.5066282746310005 * s / x);
}
function gser(a, x) { let ap = a, s = 1 / a, d = s; for (let n = 0; n < 1000; n++) { ap++; d *= x / ap; s += d; if (Math.abs(d) < Math.abs(s) * 1e-16) break; } return s * Math.exp(-x + a * Math.log(x) - lgam(a)); }
function gcf(a, x) {
  const F = 1e-300; let b = x + 1 - a, c = 1 / F, d = 1 / b, h = d;
  for (let i = 1; i < 1000; i++) { const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < F) d = F; c = b + an / c; if (Math.abs(c) < F) c = F; d = 1 / d; const e = d * c; h *= e; if (Math.abs(e - 1) < 1e-16) break; }
  return Math.exp(-x + a * Math.log(x) - lgam(a)) * h;
}
const chi2sf = (x, k) => (x <= 0 ? 1 : x / 2 < k / 2 + 1 ? 1 - gser(k / 2, x / 2) : gcf(k / 2, x / 2));
function betacf(a, b, x) {
  const F = 1e-300, qab = a + b, qap = a + 1, qam = a - 1; let c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < F) d = F; d = 1 / d; let h = d;
  for (let m = 1; m <= 1000; m++) {
    const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < F) d = F; c = 1 + aa / c; if (Math.abs(c) < F) c = F; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < F) d = F; c = 1 + aa / c; if (Math.abs(c) < F) c = F; d = 1 / d; const e = d * c; h *= e; if (Math.abs(e - 1) < 1e-16) break;
  }
  return h;
}
function ibeta(a, b, x) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const bt = Math.exp(lgam(a + b) - lgam(a) - lgam(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
}
const t2 = (t, df) => ibeta(df / 2, 0.5, df / (df + t * t));                 // two-sided p-value of a t statistic
const tcdf = (t, df) => { const p = 0.5 * t2(t, df); return t > 0 ? 1 - p : p; };
const fsf = (F, d1, d2) => (F <= 0 ? 1 : ibeta(d2 / 2, d1 / 2, d2 / (d2 + d1 * F)));
const bisect = (f, lo, hi) => { for (let i = 0; i < 200; i++) { const m = (lo + hi) / 2; if (f(m)) lo = m; else hi = m; } return (lo + hi) / 2; };
const tppf = (p, df) => bisect(x => tcdf(x, df) < p, -60, 60);
const fppf = (p, d1, d2) => bisect(x => 1 - fsf(x, d1, d2) < p, 0, 200);
const chi2ppf = (p, k) => bisect(x => 1 - chi2sf(x, k) < p, 0, 500);

// ---------------------------------------------------------------- time-series tools of the chapter
function acfAdj(x, K0) {                       // autocorrelations with the divisor n - k (statsmodels pacf, method 'ywadjusted')
  const n = x.length, m = amean(x); let c0 = 0; for (const v of x) c0 += (v - m) ** 2; c0 /= n;
  const r = []; for (let k = 1; k <= K0; k++) { let c = 0; for (let t = k; t < n; t++) c += (x[t] - m) * (x[t - k] - m); r.push(c / (n - k) / c0); }
  return r;
}
const pacfYW = (x, K0) => pacf(acfAdj(x, K0));
function lbox(e, m, k) { const n = e.length, r = acf(e, m); let q = 0; for (let j = 1; j <= m; j++) q += r[j - 1] ** 2 / (n - j); q *= n * (n + 2); return { q, df: m - k, p: chi2sf(q, m - k) }; }
function moments(e) {                          // scipy.stats.skew, kurtosis (fisher=False), jarque_bera
  const n = e.length, m = amean(e); let m2 = 0, m3 = 0, m4 = 0; for (const v of e) { const d = v - m; m2 += d * d; m3 += d * d * d; m4 += d * d * d * d; }
  m2 /= n; m3 /= n; m4 /= n; const s = m3 / m2 ** 1.5, k = m4 / (m2 * m2); const jb = n / 6 * (s * s + (k - 3) ** 2 / 4);
  return { s, k, jb, p: chi2sf(jb, 2) };
}
// psi weights of an ARMA written with lag polynomials as coefficient arrays (index = lag): phi(L) x = theta(L) e
const pmul = (a, b) => { const c = new Array(a.length + b.length - 1).fill(0); a.forEach((x, i) => b.forEach((y, j) => { c[i + j] += x * y; })); return c; };
function psi(ar, ma, n) {                      // ar = [1, -phi1, ...], ma = [1, theta1, ...]
  const p = []; for (let j = 0; j < n; j++) { let v = j < ma.length ? ma[j] : 0; for (let i = 1; i < ar.length && i <= j; i++) v -= ar[i] * p[j - i]; p.push(v); } return p;
}
function armaAcf(ar, ma, K0) {
  const w = psi(ar, ma, 1200), g = k => { let s = 0; for (let j = 0; j + k < w.length; j++) s += w[j] * w[j + k]; return s; }, g0 = g(0);
  return range(1, K0, 1).map(k => g(k) / g0);
}

// ---------------------------------------------------------------- the HICP of the slides and its transformations
const MON = i => { const t = new Date(Date.UTC(+D.m0.slice(0, 4), +D.m0.slice(5, 7) - 1 + i, 1)); return t.toISOString().slice(0, 10); };
const HD = D.hicp.map((_, i) => MON(i));
const Y = D.hicp.map(v => 100 * Math.log(v));
const W = Y.map((v, i) => (i ? v - Y[i - 1] : null));                // monthly inflation 100 Δ ln P
const A = Y.map((v, i) => (i >= 12 ? v - Y[i - 12] : null));         // annual inflation 100 Δ12 ln P
const Z = A.map((v, i) => (i >= 13 ? v - A[i - 1] : null));          // z = ΔΔ12 ln P
const idxOf = d => HD.indexOf(d);
const SER = { m: W, a: A, z: Z };
const nn = (arr, i0, i1) => { const o = [], d = []; for (let i = i0; i <= i1; i++) if (arr[i] !== null) { o.push(arr[i]); d.push(HD[i]); } return { v: o, d }; };

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
    return { xa: 'x' + n, ya: 'y' + n, k: k + 1 };
  });
  el.style.height = H + 'px';
  return { lay, ax, narrow, n: cells.length };
}
const ln = (x, y, a, color, name, o) => { const r = Object.assign({ x, y, mode: 'lines', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.4g}<extra></extra>' }, o || {}); r.line = Object.assign({ color, width: 1.4 }, (o && o.line) || {}); return r; };
const mk = (x, y, a, color, name, o) => Object.assign({ x, y, mode: 'markers', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, marker: { color, size: 8 }, hovertemplate: '%{x}: %{y:.4g}<extra></extra>' }, o || {});
const bar = (x, y, a, color, name, o) => Object.assign({ x, y, type: 'bar', marker: { color }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.4g}<extra></extra>' }, o || {});
const hl = (a, xs, y, color, name, o) => ln(xs, xs.map(() => y), a, color, name, Object.assign({ hoverinfo: 'skip' }, o || {}));
const band = (x, lo, hi, a, col, al, name) => ({ x: x.concat([...x].reverse()), y: lo.concat([...hi].reverse()), fill: 'toself', fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 },
  mode: 'lines', hoverinfo: 'skip', name: name || '', showlegend: !!name, xaxis: a.xa, yaxis: a.ya });
const lw = (G, t, n) => { if (!G.narrow) return t; const w = t.split(' '), Ls = ['']; w.forEach(x => { const c = Ls[Ls.length - 1]; if ((c + ' ' + x).length > (n || 40) && c) Ls.push(x); else Ls[Ls.length - 1] = (c + ' ' + x).trim(); }); return Ls.join('<br>'); };

// ---------------------------------------------------------------- tables and worked steps (HTML)
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}>${v.v}</${tag}>`; };
  el.innerHTML = (o && o.cap ? `<p class="c15cap">${o.cap}</p>` : '') + '<table><thead><tr>' + head.map(c => cell(c, 'th')).join('') + '</tr></thead><tbody>'
    + rows.map(r => '<tr' + (r.hi ? ' class="hi"' : '') + '>' + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c15cap">${o.note}</p>` : '');
}
const R_ = v => ({ v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };

// ---------------------------------------------------------------- model labels and the fits of step 4
const ROWS = NB.grid.rows, FIT = D.fit.fits, NEFF = D.fit.neff;
const IDENT = NB.grid.ident.model, CHOSEN = NB.grid.chosen;
const TAB5 = () => [IDENT].concat(ROWS.filter(r => r.model !== IDENT).slice(0, 4).map(r => r.model));   // the rows of the slide's table
const row = m => ROWS.find(r => r.model === m);
const sub12 = m => m + '12';                                                  // (2,1,1)(0,1,1)12 in the charts
const nARMA = m => { const r = row(m); return r.order[0] + r.order[2] + r.sorder[0] + r.sorder[2]; };
const E0 = D.fit.e0, EI0 = idxOf(E0);
const resid = m => FIT[m].e;
const sig2 = m => { const f = FIT[m]; return f.b[f.names.indexOf('sigma2')]; };
const pget = (m, k) => { const f = FIT[m], j = f.names.indexOf(k); return j < 0 ? 0 : f.b[j]; };
const stdres = m => memo('zs:' + m, () => { const s = Math.sqrt(sig2(m)); return resid(m).map(v => v / s); });

// ================================================================= 1. Step 1: the data
function blkDa() {
  const yr = val('da-e'), end = `${yr}-08-01`, i1 = idxOf(end); set('da-e', T.my(end));
  const G = grid('da-chart', [{ r: 0, c: 0, yt: 'index', x: { range: ['2004-10-01', '2026-12-31'], showticklabels: false } }, { r: 1, c: 0, yt: '%', x: { range: ['2004-10-01', '2026-12-31'] }, y: { range: [-4, 15] } }],
    { cols: 1, h: 420, gyPx: 30, legendRows: 2 });
  const [a, b] = G.ax, d = HD.slice(0, i1 + 1), w = nn(W, 0, i1), an = nn(A, 0, i1);
  plot('da-chart', [ln(d, D.hicp.slice(0, i1 + 1), a, K.blue, 'HICP of Romania, 2015 = 100', { line: { width: 1.8 } }),
    bar(w.d, w.v, b, K.amber, 'monthly inflation, 100 Δln P (%)'), ln(an.d, an.v, b, K.red, 'annual inflation, 100 Δ12 ln P (%)', { line: { width: 1.6 } }),
    hl(b, ['2004-10-01', '2026-12-31'], 0, K.navy, '', { line: { width: 0.6 } })], G.lay);
  const ia = argmax(an.v), ib = argmin(an.v);
  out('da-out', [T.da1(d.length, T.my(end), f2(D.hicp[i1]), f2(amean(w.v))), T.da2(f1(an.v[ib]), T.my(an.d[ib]), f1(an.v[ia]), T.my(an.d[ia]), f1(an.v[an.v.length - 1])), ...(yr < 2026 ? [T.extra] : [])]);
  return { n: d.length, mean: amean(w.v), min: an.v[ib], dmin: an.d[ib], max: an.v[ia], dmax: an.d[ia], last: D.hicp[i1] };
}
function monthMeans(i0) {
  const s = new Array(12).fill(0), c = new Array(12).fill(0);
  for (let i = Math.max(1, i0); i < W.length; i++) { const k = +HD[i].slice(5, 7) - 1; s[k] += W[i]; c[k]++; }
  return s.map((v, k) => v / c[k]);
}
const MN = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
function blkTd() {
  const y0 = val('td-s'), i0 = idxOf(`${y0}-01-01`); set('td-s', String(y0));
  const mm = monthMeans(i0), full = monthMeans(0);
  const G = grid('td-chart', [{ r: 0, c: 0, t: `mean monthly inflation by calendar month, ${y0}-2026`, yt: '% per month' }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0];
  plot('td-chart', [bar(MN, mm, a, mm.map(v => (v < 0 ? K.green : K.amber)), `mean of 100 Δln P, ${y0}-2026`),
    mk(MN, full, a, K.navy, 'January 2005 - August 2026 (lecture)', { marker: { size: 9, color: K.navy, symbol: 'diamond-open' } })], G.lay);
  const hi = argmax(mm), lo = argmin(mm);
  out('td-out', [T.td1(y0, f2(mm[0]), f2(mm[5]), T.mon[hi], f2(mm[hi]), T.mon[lo], f2(mm[lo])), T.td2(f1(NB.data.a_2025_07), f1(NB.data.a_2025_08), f2(NB.data.m_2025_08)),
    ...(y0 > 2005 ? [T.extra] : [])]);
  return { mm };
}

// ================================================================= 2. Step 2: the tests
const TSER = [['y', '100 ln P'], ['m', 'Δ ln P'], ['a', 'Δ12 ln P'], ['z', 'ΔΔ12 ln P']];
function verdict(r, k) {
  const ar = r.adf_p < 0.05, kr = r.kpss > r.kpss_cv5;
  return !ar && kr ? 'ur' : ar && !kr ? (k === 'm' ? 'sts' : 'st') : !ar && !kr ? 'inc' : 'conf';
}
function blkTt() {
  const e = sel('tt-e'), R = D.tests[e];
  const G = grid('tt-chart', [{ r: 0, c: 0, t: 'ADF tau and the 5% critical value', yt: 'tau' }, { r: 0, c: 1, t: 'KPSS eta and the 5% critical value', yt: 'eta' }], { cols: 2, h: 350, legendRows: 2 });
  const [a, b] = G.ax, lab = TSER.map(([, l]) => l);
  plot('tt-chart', [bar(lab, TSER.map(([k]) => R[k].adf), a, TSER.map(([k]) => (R[k].adf_p < 0.05 ? K.green : K.red)), '', { showlegend: false }),
    mk(lab, TSER.map(([k]) => R[k].adf_cv5), a, K.navy, '5% critical value', { marker: { size: 14, color: K.navy, symbol: 'line-ew-open', line: { width: 3, color: K.navy } } }),
    bar(lab, TSER.map(([k]) => R[k].kpss), b, TSER.map(([k]) => (R[k].kpss > R[k].kpss_cv5 ? K.red : K.green)), '', { showlegend: false }),
    mk(lab, TSER.map(([k]) => R[k].kpss_cv5), b, K.navy, '', { marker: { size: 14, color: K.navy, symbol: 'line-ew-open', line: { width: 3, color: K.navy } } }),
    mk([null], [null], a, K.green, 'green: stationarity supported (ADF rejects / KPSS does not)', { marker: { size: 9, color: K.green, symbol: 'square' } }),
    mk([null], [null], a, K.red, 'red: unit root supported', { marker: { size: 9, color: K.red, symbol: 'square' } })], G.lay);
  const rows = TSER.map(([k, l]) => { const r = R[k]; return [T.serT[k], r.reg === 'ct' ? T.ct : T.c, R_(f2(r.adf)), R_(pv(r.adf_p)), R_(`${f3(r.kpss)} (${f3(r.kpss_cv5)})`), T.verd[verdict(r, k)]]; });
  table('tt-tb', T.ttHead, rows, { note: T.ttNote(R.z.n, f2(R.y.adf_cv5), f2(R.m.adf_cv5)) });
  out('tt-out', [T.tt1(T.my(e + '-01'), T.verd[verdict(R.a, 'a')], f2(R.a.adf), f3(R.a.kpss), f2(R.z.adf), f3(R.z.kpss)), ...(e !== '2026-08' ? [T.extra] : [])]);
  return { R, v: TSER.map(([k]) => verdict(R[k], k)) };
}

// ================================================================= 3. Step 3: the correlogram
function corr(k, K0) { const x = nn(SER[k], 0, Y.length - 1).v; return memo(`acf:${k}:${K0}`, () => ({ n: x.length, r: acf(x, K0), p: pacfYW(x, K0) })); }
function blkAc() {
  const s = sel('ac-s'), K0 = val('ac-k'); set('ac-k', String(K0));
  const C = corr(s, K0), bd = 1.96 / Math.sqrt(C.n), lags = range(1, K0, 1), col = lags.map(k => (k % 12 === 0 ? K.red : K.blue));
  const nm = TSER.find(([k]) => k === s)[1];
  const G = grid('ac-chart', [{ r: 0, c: 0, t: `ACF of ${nm}`, xt: 'lag (months); seasonal lags in red' }, { r: 0, c: 1, t: `PACF of ${nm}`, xt: 'lag (months); seasonal lags in red' }],
    { cols: 2, h: 340, legendRows: 1 });
  const [a, b] = G.ax, tr = [];
  [[a, C.r], [b, C.p]].forEach(([x, v], j) => {
    tr.push(bar(lags, v, x, col, '', { showlegend: false, width: 0.55 }), hl(x, [0.5, K0 + 0.5], bd, K.green, j ? '' : '±1.96/√T', { line: { width: 1, dash: 'dash' } }),
      hl(x, [0.5, K0 + 0.5], -bd, K.green, '', { line: { width: 1, dash: 'dash' } }));
  });
  G.lay.yaxis2.matches = 'y';
  plot('ac-chart', tr, G.lay);
  const outR = lags.filter((k, i) => Math.abs(C.r[i]) > bd), at = (v, k) => (k <= K0 ? f2(v[k - 1]) : '–');
  out('ac-out', [T.ac1(nm, C.n, f3(bd), f2(C.r[0]), f2(C.r[1]), at(C.r, 12), at(C.r, 24)), T.ac2(f2(C.p[0]), at(C.p, 12), at(C.p, 24), at(C.p, 36)), T.ac3(outR.join(', '))]);
  return C;
}
function blkTa() {
  const ph = val('ta-p'), th = val('ta-q'); set('ta-p', f3(ph)); set('ta-q', f3(th));
  const C = corr('z', 36), lags = range(1, 36, 1);
  const ar = [1, -ph], ma = new Array(13).fill(0); ma[0] = 1; ma[12] = th;
  const r = armaAcf(ar, ma, 36), bd = 1.96 / Math.sqrt(C.n);
  const G = grid('ta-chart', [{ r: 0, c: 0, t: 'ACF of z: sample and the AR(1) x seasonal MA(1) model', xt: 'lag (months)' }], { cols: 1, h: 360, legendRows: 2 });
  const a = G.ax[0];
  plot('ta-chart', [bar(lags, C.r, a, lags.map(k => (k % 12 === 0 ? K.red : K.blue)), 'sample ACF of z', { width: 0.55, opacity: 0.85 }),
    ln(lags, r, a, K.amber, `model: phi = ${ph.toFixed(3)}, Theta = ${th.toFixed(3)}`, { mode: 'lines+markers', marker: { size: 5, color: K.amber }, line: { width: 2 } }),
    hl(a, [0.5, 36.5], bd, K.green, '±1.96/√T', { line: { width: 1, dash: 'dash' } }), hl(a, [0.5, 36.5], -bd, K.green, '', { line: { width: 1, dash: 'dash' } })], G.lay);
  out('ta-out', [T.ta1(f2(C.r[0]), f2(C.r[11]), f2(C.r[0] * C.r[11]), f2(C.r[10]), f2(C.r[12])), T.ta2(sgn(-ph, 3), sgn(th, 3), f2(r[0]), f2(r[11]), f2(r[10]), f2(r[12])),
    ...(Math.abs(ph - NB.diag.ident.params['ar.L1']) > 1e-9 || Math.abs(th - NB.diag.ident.params['ma.S.L12']) > 1e-9 ? [T.extra] : [])]);
  return { r, C };
}

// ================================================================= 4. Step 4: the candidates and the chosen model
const IC = (r, c) => -2 * r.loglik + c * r.k;
function gridPick(c, al) {
  const ok = r => r.converged && r.lb12_p > al && r.lb24_p > al;
  const srt = ROWS.slice().sort((x, y) => IC(x, c) - IC(y, c)), best = srt[0], ch = srt.find(ok);
  return { srt, best, ch, ok };
}
function blkTg() {
  const c = val('tg-c'), al = +sel('tg-a'); set('tg-c', f2(c));
  const P = gridPick(c, al), xs = ROWS.map((r, i) => r.k + ((i % 7) - 3) * 0.06);
  const G = grid('tg-chart', [{ r: 0, c: 0, t: `-2 ln L + c k for the 36 models, c = ${c.toFixed(2)}`, xt: 'k: estimated parameters (variance included)', yt: '-2 ln L + c k' }],
    { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0], v = ROWS.map(r => IC(r, c)), pass = ROWS.map(P.ok), lab = ROWS.map(r => sub12(r.model));
  const pick = (f, col, nm, sym, sz) => mk(xs.filter((_, i) => f(i)), v.filter((_, i) => f(i)), a, col, nm, { text: lab.filter((_, i) => f(i)), hovertemplate: '%{text}: %{y:.1f}<extra></extra>', marker: { size: sz || 8, color: col, symbol: sym || 'circle' } });
  plot('tg-chart', [pick(i => pass[i], K.green, `passes Ljung-Box at lags 12 and 24 (level ${al})`), pick(i => !pass[i], K.red, 'fails Ljung-Box'),
    pick(i => ROWS[i].model === IDENT, K.amber, 'identified (1,1,0)(0,1,1)12', 'diamond-open', 14), pick(i => ROWS[i].model === (P.ch || {}).model, K.blue, 'chosen: lowest criterion among the models that pass', 'star', 16)], G.lay);
  const show = [row(IDENT)].concat(P.srt.filter(r => r.model !== IDENT).slice(0, 4));
  table('tg-tb', T.tgHead(f2(c)), show.map((r, j) => Object.assign([T.mod(r.model) + (j ? '' : T.identTag), R_(r.k), R_(f1(r.aicc)), R_(f1(r.bic)), R_(f3(r.lb12_p)), R_(f3(r.lb24_p)), R_(f1(IC(r, c)))],
    { hi: P.ch && r.model === P.ch.model })), { note: T.tgNote(NB.grid.n_train, NEFF, f2(Math.log(NEFF)), f2(al)) });
  out('tg-out', [T.tg1(f2(c), T.mod(P.best.model), T.mod(P.ch ? P.ch.model : '–'), f2(al)), T.tg2(f2(Math.log(NEFF)), NEFF)]);
  return { best: P.best.model, ch: P.ch && P.ch.model, show: show.map(r => r.model) };
}
const PNAME = { 'ar.L1': ['φ₁', 'ar.L1'], 'ar.L2': ['φ₂', 'ar.L2'], 'ma.L1': ['θ₁', 'ma.L1'], 'ma.L2': ['θ₂', 'ma.L2'], 'ar.S.L12': ['Φ₁', 'ar.S.L12'], 'ma.S.L12': ['Θ₁', 'ma.S.L12'], sigma2: ['σ²', ''] };
function polys(m) {
  const f = k => pget(m, k);
  const ar = [1, -f('ar.L1'), -f('ar.L2')], ma = [1, f('ma.L1'), f('ma.L2')], sar = new Array(13).fill(0), sma = new Array(13).fill(0);
  sar[0] = sma[0] = 1; sar[12] = -f('ar.S.L12'); sma[12] = f('ma.S.L12');
  return { ar, ma, sar, sma };
}
// roots of each factor: a quadratic c0 + c1 L + c2 L^2 and a seasonal 1 + s L^12 (|L| = |s|^(-1/12), 12 angles)
function qroots(c) { const [c0, c1, c2] = c; if (Math.abs(c2) < 1e-14) return Math.abs(c1) < 1e-14 ? [] : [[-c0 / c1, 0]]; const d = c1 * c1 - 4 * c2 * c0;
  return d >= 0 ? [[(-c1 + Math.sqrt(d)) / (2 * c2), 0], [(-c1 - Math.sqrt(d)) / (2 * c2), 0]] : [[-c1 / (2 * c2), Math.sqrt(-d) / (2 * c2)], [-c1 / (2 * c2), -Math.sqrt(-d) / (2 * c2)]]; }
function sroots(s) { if (Math.abs(s) < 1e-14) return []; const r = Math.abs(s) ** (-1 / 12), ph = s > 0 ? Math.PI / 12 : 0;   // L^12 = -1/s
  return range(0, 11, 1).map(k => [r * Math.cos(ph + 2 * Math.PI * k / 12), r * Math.sin(ph + 2 * Math.PI * k / 12)]); }
const inv = ([x, y]) => { const q = x * x + y * y; return [x / q, -y / q]; };
const modl = ([x, y]) => Math.hypot(x, y);
function blkTc() {
  const m = sel('tc-m'), P = polys(m), arR = qroots(P.ar).concat(sroots(P.sar[12])), maR = qroots(P.ma).concat(sroots(P.sma[12]));
  const zp = psi(pmul(P.ar, P.sar), pmul(P.ma, P.sma), 49);
  const ap = []; zp.reduce((s, v) => { ap.push(s + v); return s + v; }, 0);                // annual inflation: (1 - L)^(-1) applied to z
  const G = grid('tc-chart', [{ r: 0, c: 0, t: 'inverse roots and the unit circle', xt: 'real', yt: 'imaginary', x: { range: [-1.25, 1.25] }, y: { range: [-1.25, 1.25] } },
    { r: 0, c: 1, t: 'annual inflation after a shock of 1 pp to monthly inflation', xt: 'months after the shock', yt: 'pp' }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax, th = linspace(0, 2 * Math.PI, 181), iA = arR.map(inv), iM = maR.map(inv);
  plot('tc-chart', [ln(th.map(Math.cos), th.map(Math.sin), a, K.navy, '', { hoverinfo: 'skip', line: { width: 1 } }),
    mk(iA.map(p => p[0]), iA.map(p => p[1]), a, K.blue, 'AR inverse roots', { marker: { size: 9, color: K.blue, symbol: 'x' } }),
    mk(iM.map(p => p[0]), iM.map(p => p[1]), a, K.red, 'MA inverse roots', { marker: { size: 8, color: K.red, symbol: 'circle-open', line: { width: 1.5, color: K.red } } }),
    bar(range(0, 48, 1), ap, b, K.amber, `SARIMA${sub12(m)}`), hl(b, [-0.5, 48.5], 0, K.navy, '', { line: { width: 0.6 } })], G.lay);
  const f = FIT[m], rows = f.names.map((k, j) => (k === 'sigma2' ? [PNAME[k][0], R_(f3(f.b[j])), '', '', ''] : [`${PNAME[k][0]} (${PNAME[k][1]})`, R_(f3(f.b[j])), R_(f3(f.se[j])), R_(f2(f.b[j] / f.se[j])), R_(pv(f.p[j]))]));
  table('tc-tb', T.tcHead, rows, { note: T.tcNote(T.mod(m)) });
  const mA = arR.length ? Math.min(...arR.map(modl)) : null, mM = maR.length ? Math.min(...maR.map(modl)) : null;
  out('tc-out', [T.tc1(T.mod(m), mA === null ? '–' : f3(mA), mM === null ? '–' : f3(mM)), T.tc2(f2(ap[11]), f2(ap[12]), f2(ap[24]), ap[11] > 1)]);
  return { mA, mM, ap };
}

// ================================================================= 5. Step 5: the residual checks
const qq = x => { const s = Float64Array.from(x).sort(), n = s.length; return { t: Array.from(s, (_, i) => normPpf((i + 1) / (n + 1))), s: Array.from(s) }; };
function diagOf(m) {
  return memo('diag:' + m, () => {
    const e = resid(m), zs = stdres(m), k = nARMA(m), mo = moments(e);
    const big = zs.map((v, i) => [i, v]).sort((x, y) => Math.abs(y[1]) - Math.abs(x[1])).slice(0, 3).map(([i, v]) => [HD[EI0 + i], v]);
    return { lb12: lbox(e, 12, k), lb24: lbox(e, 24, k), mo, big, k };
  });
}
function blkDg() {
  const m = sel('dg-m'), K0 = val('dg-k'); set('dg-k', String(K0));
  const zs = stdres(m), e = resid(m), ei = resid(IDENT), dts = zs.map((_, i) => HD[EI0 + i]), lags = range(1, K0, 1), bd = 1.96 / Math.sqrt(e.length);
  const Q = qq(zs), lim = Math.ceil(Math.max(Math.abs(Q.s[0]), Math.abs(Q.s[Q.s.length - 1]), 3));
  const G = grid('dg-chart', [{ r: 0, c: 0, t: 'residuals over time' }, { r: 0, c: 1, t: 'ACF of the residuals', xt: 'lag' }, { r: 0, c: 2, t: 'QQ plot against N(0, 1)', xt: 'N(0, 1) quantiles', yt: 'residual quantiles' }],
    { cols: 3, h: 360, legendRows: 2, gx: 0.07 });
  const [a, b, c] = G.ax;
  plot('dg-chart', [ln(dts, zs, a, K.blue, `standardised residuals, ${sub12(m)}`, { line: { width: 0.9 } }), hl(a, [dts[0], dts[dts.length - 1]], 1.96, K.green, '', { line: { width: 0.8, dash: 'dash' } }),
    hl(a, [dts[0], dts[dts.length - 1]], -1.96, K.green, '', { line: { width: 0.8, dash: 'dash' } }),
    bar(lags.map(k => k - 0.18), acf(ei, K0), b, K.amber, `${sub12(IDENT)} (identified)`, { width: 0.36 }), bar(lags.map(k => k + 0.18), acf(e, K0), b, K.blue, `${sub12(m)} (selected)`, { width: 0.36 }),
    hl(b, [0.5, K0 + 0.5], bd, K.green, '', { line: { width: 1, dash: 'dash' } }), hl(b, [0.5, K0 + 0.5], -bd, K.green, '', { line: { width: 1, dash: 'dash' } }),
    mk(Q.t, Q.s, c, K.blue, '', { marker: { size: 4, color: K.blue }, showlegend: false }), ln([-lim, lim], [-lim, lim], c, K.red, '45-degree line', { hoverinfo: 'skip' })], G.lay);
  const Dg = diagOf(m), Di = diagOf(IDENT);
  out('dg-out', [T.dg1(T.mod(m), f2(Dg.lb12.q), Dg.lb12.df, f3(Dg.lb12.p), f2(Dg.lb24.q), Dg.lb24.df, f3(Dg.lb24.p)),
    T.dg2(f3(Di.lb12.p), f3(Di.lb24.p)), T.dg3(f0(Dg.mo.jb), f1(Dg.mo.k), T.my(Dg.big[0][0]), f1(Dg.big[0][1]), T.my(Dg.big[1][0]), f1(Dg.big[1][1]))]);
  return Dg;
}
function blkTx() {
  const mm = val('tx-m'); set('tx-m', String(mm));
  const ms = TAB5(), cols = [K.amber, K.purple, K.blue, K.teal, K.green], xs = range(6, 36, 1);
  const G = grid('tx-chart', [{ r: 0, c: 0, t: 'Ljung-Box p-value of the residuals, df = m - k', xt: 'm (lags tested)', yt: 'p-value', y: { range: [0, 1] } }], { cols: 1, h: 380, legendRows: 3 });
  const a = G.ax[0], tr = [];
  ms.forEach((m, j) => {
    const k = nARMA(m), e = resid(m), p = xs.map(x => (x > k ? lbox(e, x, k).p : null));
    tr.push(ln(xs, p, a, cols[j], sub12(m) + (m === IDENT ? ' (identified)' : ''), { mode: 'lines+markers', marker: { size: 4, color: cols[j] }, line: { width: m === CHOSEN ? 2.4 : 1.2 } }));
    tr.push(mk([mm], [mm > k ? lbox(e, mm, k).p : null], a, cols[j], '', { marker: { size: 11, color: cols[j], symbol: 'diamond' } }));
  });
  tr.push(hl(a, [6, 36], 0.05, K.red, '5%', { line: { width: 1, dash: 'dash' } }));
  plot('tx-chart', tr, G.lay);
  table('tx-tb', T.txHead(mm), ms.map(m => { const d = diagOf(m), q = lbox(resid(m), mm, d.k);
    return Object.assign([T.mod(m), R_(`${f2(q.q)} (${q.df})`), R_(f3(q.p)), R_(f3(d.lb12.p)), R_(f3(d.lb24.p)), R_(f0(d.mo.jb)), R_(f1(d.mo.k)), `${T.my(d.big[0][0])} (${f1(d.big[0][1])})`], { hi: m === CHOSEN }); }),
  { note: T.txNote });
  const dc = diagOf(CHOSEN), qc = lbox(resid(CHOSEN), mm, dc.k), qi = lbox(resid(IDENT), mm, nARMA(IDENT));
  out('tx-out', [T.tx1(mm, T.mod(CHOSEN), f2(qc.q), qc.df, f3(qc.p), f3(qi.p)), ...(mm !== 12 && mm !== 24 ? [T.extra] : [])]);
  return { qc, qi };
}

// ================================================================= 6. Step 6: forecasts
const FC = D.fc, TI0 = idxOf(FC.test0), ORG = Object.keys(FC.origins);
const LEV = { 80: 1.2815515655446004, 95: 1.959963984540054, 99: 2.5758293035489004 };
function blkFc() {
  const o = ORG[val('fc-o')], bm = sel('fc-b'), lv = +sel('fc-l'), zq = LEV[lv]; set('fc-o', T.my(o + '-01'));
  const io = idxOf(o + '-01'), nt = io - TI0 + 1, td = HD.slice(TI0, io + 1), F = FC.origins[o];
  const fd = range(1, 12, 1).map(h => MON(io + h)), a = F.a, lo = a.map((v, h) => v - zq * F.se[h]), hi = a.map((v, h) => v + zq * F.se[h]);
  const i15 = idxOf('2015-01-01'), hist = nn(A, i15, io);
  const G = grid('fc-chart', [{ r: 0, c: 0, t: 'one-step forecasts of monthly inflation (%)', x: { range: [MON(TI0 - 1), MON(TI0 + 24)] } },
    { r: 0, c: 1, t: `annual inflation: forecast to ${T.myEn(fd[11])}`, x: { range: ['2015-01-01', '2027-09-30'] }, y: { range: [-4, 15] } }], { cols: 2, h: 380, legendRows: 3 });
  const [A1, B1] = G.ax, bmv = bm === 'sn' ? FC.sn : FC.nv;
  plot('fc-chart', [ln(td, W.slice(TI0, io + 1), A1, K.navy, 'actual', { mode: 'lines+markers', marker: { size: 6, color: K.navy } }),
    ln(td, FC.sar.slice(0, nt), A1, K.blue, `SARIMA${sub12(CHOSEN)}`, { mode: 'lines+markers', marker: { size: 6, color: K.blue, symbol: 'square' }, line: { width: 1.6 } }),
    ln(td, bmv.slice(0, nt), A1, K.amber, bm === 'sn' ? 'seasonal naive' : 'naive', { mode: 'lines+markers', marker: { size: 6, color: K.amber, symbol: 'triangle-up' }, line: { dash: 'dash' } }),
    hl(A1, [MON(TI0 - 1), MON(TI0 + 24)], 0, K.navy, '', { line: { width: 0.5 } }),
    ln(hist.d, hist.v, B1, K.red, 'annual inflation (%)', { line: { width: 1.6 } }), band(fd, lo, hi, B1, K.blue, 0.2, `${lv}% interval`),
    ln(fd, a, B1, K.blue, 'forecast', { line: { width: 2.2 } }), hl(B1, ['2015-01-01', '2027-09-30'], 2.5, K.green, 'BNR target 2.5%', { line: { width: 1, dash: 'dash' } })], G.lay);
  const act = W.slice(TI0, io + 1), rm = f => Math.sqrt(amean(act.map((v, i) => (v - f[i]) ** 2)));
  out('fc-out', [T.fc1(T.my(o + '-01'), f1(A[io]), T.my(fd[0]), f1(a[0]), f1(lo[0]), f1(hi[0]), T.my(fd[11]), f1(a[11]), f1(lo[11]), f1(hi[11]), lv),
    T.fc2(nt, f2(rm(FC.sar.slice(0, nt))), f2(rm(FC.sn.slice(0, nt))), f2(rm(FC.nv.slice(0, nt)))), ...(o !== ORG[ORG.length - 1] || lv !== 95 ? [T.extra] : [])]);
  return { a1: a[0], lo1: lo[0], hi1: hi[0], a12: a[11], lo12: lo[11], hi12: hi[11] };
}
function dmTest(e1, e2, abs) {                 // dm_test of the chapter, h = 1, HLN correction, t(n - 1) p-value
  const d = e1.map((v, i) => (abs ? Math.abs(v) - Math.abs(e2[i]) : v * v - e2[i] * e2[i])), n = d.length, db = amean(d);
  const g0 = amean(d.map(x => (x - db) ** 2)), dm = db / Math.sqrt(g0 / n), hln = Math.sqrt((n - 1) / n) * dm;
  return { db, dm, hln, p: t2(hln, n - 1), n, d };
}
function accOf(n, abs) {
  const act = W.slice(TI0, TI0 + n), er = f => act.map((v, i) => v - f[i]);
  const E = { sar: er(FC.sar), sn: er(FC.sn), nv: er(FC.nv) }, acc = {};
  Object.entries(E).forEach(([k, e]) => { const mae = amean(e.map(Math.abs)); acc[k] = { rmse: Math.sqrt(amean(e.map(x => x * x))), mae, mase: mae / FC.scale }; });
  return { E, acc, dsn: dmTest(E.sar, E.sn, abs), dnv: dmTest(E.sar, E.nv, abs) };
}
function blkTf() {
  const n = val('tf-n'), abs = sel('tf-l') === 'abs'; set('tf-n', T.months(n));
  const R = accOf(n, abs), td = HD.slice(TI0, TI0 + n);
  const cum = d => { let s = 0; return d.map(v => (s += v)); };
  const G = grid('tf-chart', [{ r: 0, c: 0, t: `cumulative loss differential (${abs ? 'absolute' : 'squared'} errors)`, yt: 'sum of d_t, SARIMA minus benchmark' },
    { r: 0, c: 1, t: 'absolute forecast errors (pp)', yt: '|e_t|' }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax;
  plot('tf-chart', [ln(td, cum(R.dsn.d), a, K.amber, 'against the seasonal naive', { mode: 'lines+markers', marker: { size: 5, color: K.amber }, line: { width: 2 } }),
    ln(td, cum(R.dnv.d), a, K.purple, 'against the naive method', { mode: 'lines+markers', marker: { size: 5, color: K.purple }, line: { width: 2 } }),
    hl(a, [td[0], td[td.length - 1]], 0, K.navy, '', { line: { width: 0.6 } }),
    bar(td, R.E.sar.map(Math.abs), b, K.blue, `SARIMA${sub12(CHOSEN)}`), bar(td, R.E.sn.map(Math.abs), b, K.amber, 'seasonal naive')], G.lay);
  G.lay.barmode = 'group';
  const A0 = R.acc;
  table('tf-tb', T.tfHead, [['sar', `SARIMA${T.sub(sub12(CHOSEN))}`], ['sn', T.sn], ['nv', T.nv]].map(([k, l]) => Object.assign([l, R_(f2(A0[k].rmse)), R_(f2(A0[k].mae)), R_(f2(A0[k].mase))], { hi: k === 'sar' })),
    { note: T.tfNote(n, f3(FC.scale)) });
  const w = range(0, n - 1, 1).sort((i, j) => Math.abs(R.E.sar[j]) - Math.abs(R.E.sar[i])).slice(0, 2);
  out('tf-out', [T.tf1(abs, f2(R.dsn.hln), f3(R.dsn.p), f2(R.dnv.hln), f3(R.dnv.p)), T.tf2(n, T.my(td[w[0]]), f2(W[TI0 + w[0]]), f2(FC.sar[w[0]]), T.my(td[w[1]]), f2(W[TI0 + w[1]]), f2(FC.sar[w[1]])),
    ...(n !== 24 || abs ? [T.extra] : [])]);
  return R;
}

// ================================================================= 7. the exam problems
// ---- Problem 1: a trend with AR(1) errors
function blkQ1() {
  const ph = val('q1-f'), yT = val('q1-y'); set('q1-f', f2(ph)); set('q1-y', f1(yT));
  const a0 = 10, b0 = 0.5, Tn = 100, uT = yT - a0 - b0 * Tn, hs = range(1, 20, 1);
  const fc = hs.map(h => a0 + b0 * (Tn + h) + ph ** h * uT), vh = hs.map(h => (1 - ph ** (2 * h)) / (1 - ph * ph));
  const r1 = -(1 - ph) / 2, ks = range(1, 10, 1), rk = ks.map(k => ph ** (k - 1) * r1);
  const G = grid('q1-chart', [{ r: 0, c: 0, t: 'forecasts of Y from T = 100, 95% band', xt: 't', yt: 'Y' }, { r: 0, c: 1, t: 'ACF of the differenced series ΔY', xt: 'lag', y: { range: [-0.6, 0.2] } }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax, tt = range(90, 120, 1);
  plot('q1-chart', [ln(tt, tt.map(t => a0 + b0 * t), a, K.navy, 'trend 10 + 0.5 t', { line: { width: 1, dash: 'dash' } }), band(hs.map(h => Tn + h), fc.map((v, i) => v - 1.96 * Math.sqrt(vh[i])), fc.map((v, i) => v + 1.96 * Math.sqrt(vh[i])), a, K.blue, 0.18, '95% band'),
    ln(hs.map(h => Tn + h), fc, a, K.blue, `forecast, phi = ${ph.toFixed(2)}`, { line: { width: 2 } }), mk([Tn], [yT], a, K.red, `Y_T = ${yT.toFixed(1)}`, { marker: { size: 10, color: K.red } }),
    bar(ks, rk, b, K.purple, 'ARMA(1,1) with theta = -1', { width: 0.5 })], G.lay);
  const v0 = 1 / (1 - ph * ph), g0 = 2 / (1 + ph), g1 = -(1 - ph) / (1 + ph);
  steps('q1-ws', [T.q1s1(tr(ph, 2), tr(ph * ph, 4), f4(v0)), T.q1s2(tr(ph, 2), f2(g0), f2(g1), f2(r1)), T.q1s3(tr(yT, 1), f1(uT), tr(ph, 2), f2(fc[0]), f2(fc[9]), f3(vh[9])), T.q1s4]);
  out('q1-out', [T.q1o(tr(ph, 2), f3(ph ** 10 * uT), f3(1 / (1 - ph * ph))), ...(ph !== 0.6 || yT !== 62 ? [T.extraP] : [])]);
  return { v0, g0, g1, r1, f1: fc[0], f10: fc[9], v10: vh[9] };
}
// ---- Problem 2: Romanian GDP
const P2 = D.p2, P2E = Object.keys(P2.ends);
function blkQ2() {
  const e = sel('q2-e'), R = P2.ends[e], n = P2.q.indexOf(e), y = P2.y.slice(0, n + 1), g = y.slice(1).map((v, i) => v - y[i]), qd = P2.q.slice(1, n + 1);
  const qx = q => `${q.slice(0, 4)}-${String(3 * (+q.slice(5) - 1) + 1).padStart(2, '0')}-01`;
  const r = acf(g, 12), p = pacf(r), bd = 1.96 / Math.sqrt(g.length), lb = lbox(g, 8, 0), c8 = chi2ppf(0.95, 8), mg = amean(g);
  const G = grid('q2-chart', [{ r: 0, c: 0, t: 'quarterly growth g_t = 100 Δln Y_t (%)', x: { range: ['1999-10-01', '2026-09-30'] } }, { r: 0, c: 1, t: 'ACF of the growth', xt: 'lag (quarters)', y: { range: [-0.5, 0.5] } }],
    { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax, xd = qd.map(qx), ks = range(1, 12, 1);
  plot('q2-chart', [ln(xd, g, a, K.blue, `growth, 2000Q2-${e}`), hl(a, [xd[0], xd[xd.length - 1]], mg, K.red, `mean ${mg.toFixed(2)}%`, { line: { width: 1, dash: 'dash' } }),
    bar(ks, r, b, K.blue, '', { width: 0.5, showlegend: false }), hl(b, [0.5, 12.5], bd, K.green, '±1.96/√T', { line: { width: 1, dash: 'dash' } }), hl(b, [0.5, 12.5], -bd, K.green, '', { line: { width: 1, dash: 'dash' } })], G.lay);
  const al = R.adf_l, ag = R.adf_g, kl = R.kpss_l, kg = R.kpss_g, s = Math.sqrt(R.s2);
  table('q2-tb', T.q2Head, [['100 ln Y<sub>t</sub>', 'c, t', R_(f2(al.stat)), R_(pv(al.p)), R_(`${f2(al.crit1)}; ${f2(al.crit5)}; ${f2(al.crit10)}`), R_(f3(kl.stat)), R_(f3(kl.crit5))],
    ['100 Δln Y<sub>t</sub>', 'c', R_(f2(ag.stat)), R_(pv(ag.p)), R_(`${f2(ag.crit1)}; ${f2(ag.crit5)}; ${f2(ag.crit10)}`), R_(f3(kg.stat)), R_(f3(kg.crit5))]]);
  steps('q2-ws', [T.q2s1(f2(al.stat), f2(al.crit5), al.stat < al.crit5, f2(ag.stat), f2(ag.crit5), ag.stat < ag.crit5),
    T.q2s2(f3(kl.stat), f3(kl.crit5), kl.stat > kl.crit5, f3(kg.stat), f3(kg.crit5), kg.stat > kg.crit5),
    T.q2s3(f2(r[0]), f2(r[1]), f2(p[0]), f2(p[1]), f2(bd), f2(lb.q), f2(c8), f3(lb.p), lb.q < c8),
    T.q2s4(f2(R.c), f2(R.se), f2(s), f2(R.c / R.se), f1(4 * R.c))]);
  out('q2-out', [T.q2o(e, g.length, f2(mg), f2(s)), ...(e !== '2026Q2' ? [T.extraP] : [])]);
  return { r, p, bd, lb, c8, R };
}
// ---- Problem 3: SARIMA output and a forecast by hand
const P3 = NB.p3, I3 = idxOf(P3.dT), O3 = range(I3 - 23, I3, 1);
function hand(i, ph, Ph) { return ph * Z[i] + Ph * Z[i - 11] - ph * Ph * Z[i - 12]; }
function blkQ3() {
  const io = O3[val('q3-o')], ph = val('q3-p'), Ph = val('q3-q'); set('q3-o', T.my(HD[io])); set('q3-p', f3(ph)); set('q3-q', f3(Ph));
  const zn = hand(io, ph, Ph), an = A[io] + zn, w0 = I3 - 35, xs = HD.slice(w0, I3 + 2).length ? range(w0, I3, 1) : [];
  const fx = range(I3 - 23, I3, 1).map(i => MON(i + 1)), fv = range(I3 - 23, I3, 1).map(i => hand(i, ph, Ph));
  const G = grid('q3-chart', [{ r: 0, c: 0, t: 'z_t = ΔΔ12 100 ln HICP and the forecasts by hand', yt: 'pp' }, { r: 0, c: 1, t: 'annual inflation and the forecast for the next month', yt: '%' }],
    { cols: 2, h: 370, legendRows: 2 });
  const [a, b] = G.ax;
  plot('q3-chart', [ln(xs.map(i => HD[i]), xs.map(i => Z[i]), a, K.navy, 'z_t', { mode: 'lines+markers', marker: { size: 4, color: K.navy } }),
    ln(fx, fv, a, K.blue, 'one-step forecast by hand', { line: { width: 1.4, dash: 'dot' } }), mk([MON(io + 1)], [zn], a, K.red, `forecast for ${T.myEn(MON(io + 1))}`, { marker: { size: 11, color: K.red, symbol: 'star' } }),
    ln(xs.map(i => HD[i]), xs.map(i => A[i]), b, K.red, 'annual inflation (%)', { line: { width: 1.6 } }), mk([MON(io + 1)], [an], b, K.blue, 'forecast', { marker: { size: 11, color: K.blue, symbol: 'star' } }),
    hl(b, [HD[w0], MON(I3 + 1)], 2.5, K.green, 'BNR target 2.5%', { line: { width: 1, dash: 'dash' } })], G.lay);
  const pP = ph * Ph, t1 = ph * Z[io], t2_ = Ph * Z[io - 11], t3 = -pP * Z[io - 12];
  steps('q3-ws', [T.q3s1, T.q3s2(f3(ph), f3(Ph), f2(Math.abs(ph)), f2(Math.abs(Ph) ** (1 / 12)), Math.abs(ph) < 1 && Math.abs(Ph) < 1), T.q3s3(f2(P3.lb['12'].q), P3.lb['12'].df, f3(P3.lb['12'].p)),
    T.q3s4(f3(ph), par(Z[io], 3), par(Ph, 3), par(Z[io - 11], 3), par(pP, 3), par(Z[io - 12], 3), par(t1, 3), par(t2_, 3), f3(t3), f3(zn), T.my(MON(io + 1)), T.my(HD[io]), T.my(HD[io - 11]), T.my(HD[io - 12])),
    T.q3s5(f3(A[io]), par(zn, 3), f2(an), T.my(MON(io + 1)))]);
  const known = io + 1 < Z.length;
  out('q3-out', [T.q3o(T.my(MON(io + 1)), f2(an), known ? f2(A[io + 1]) : null, known ? f3(Z[io + 1]) : null),
    ...(io !== I3 || Math.abs(ph - P3.phi) > 1e-9 || Math.abs(Ph - P3.Phi) > 1e-9 ? [T.extraP] : [])]);
  return { zn, an, t1, t2: t2_, t3, pP };
}
// ---- Problem 4: VAR output and Granger causality
const P4 = D.p4, Q4 = range(0, P4.g.length - 1, 1).map(i => { const t = new Date(Date.UTC(+P4.q0.slice(0, 4), +P4.q0.slice(5, 7) - 1 + 3 * i, 1)); return `${t.getUTCFullYear()}Q${t.getUTCMonth() / 3 + 1}`; });
const P4E = Q4.filter(q => (q.endsWith('Q4') && +q.slice(0, 4) >= 2015) || q === Q4[Q4.length - 1]);
function olsEq(n, eq, drop) {                  // one equation of the VAR(2) on the first n quarters (const, L1 g pi i, L2 g pi i)
  const V = [P4.g, P4.pi, P4.i], p = 2, Tn = n - p, keep = [0, 1, 2, 3, 4, 5, 6].filter(j => drop === undefined || (j - 1) % 3 !== drop || j === 0);
  const X = [], y = [];
  for (let t = p; t < n; t++) { const r = [1]; for (let j = 1; j <= p; j++) for (let k = 0; k < 3; k++) r.push(V[k][t - j]); X.push(keep.map(j => r[j])); y.push(V[eq][t]); }
  const q = keep.length, XtX = range(0, q - 1, 1).map(i => range(0, q - 1, 1).map(j => X.reduce((s, r) => s + r[i] * r[j], 0))), Xty = range(0, q - 1, 1).map(i => X.reduce((s, r, t) => s + r[i] * y[t], 0));
  const b = solve(XtX, Xty), e = y.map((v, t) => v - X[t].reduce((s, x, j) => s + x * b[j], 0)), rss = e.reduce((s, v) => s + v * v, 0), df = Tn - 7;
  const s2 = rss / df, se = range(0, q - 1, 1).map(i => { const u = new Array(q).fill(0); u[i] = 1; return Math.sqrt(s2 * solve(XtX, u)[i]); });
  const my_ = amean(y), r2 = 1 - amean(e.map(v => v * v)) / amean(y.map(v => (v - my_) ** 2));
  return { b, se, rss, Tn, df, r2 };
}
function granger(n, cause, eq) { const U = olsEq(n, eq), Rr = olsEq(n, eq, cause), F = ((Rr.rss - U.rss) / 2) / (U.rss / U.df); return { F, p: fsf(F, 2, U.df), rssU: U.rss, rssR: Rr.rss, df2: U.df }; }
function lrOf(n) { const b = olsEq(n, 2).b; return (b[2] + b[5]) / (1 - b[3] - b[6]); }
function blkQ4() {
  const e = sel('q4-e'), n = Q4.indexOf(e) + 1, U = olsEq(n, 2), b = U.b, t = b.map((v, j) => v / U.se[j]);
  const lrs = P4E.map(q => lrOf(Q4.indexOf(q) + 1)), lr = (b[2] + b[5]) / (1 - b[3] - b[6]);
  const hs = range(0, 40, 1), path = []; hs.forEach(h => { path.push(h === 0 ? 0 : b[2] + (h >= 2 ? b[5] : 0) + b[3] * path[h - 1] + b[6] * (h >= 2 ? path[h - 2] : 0)); });
  const G = grid('q4-chart', [{ r: 0, c: 0, t: 'long-run effect of +1 pp inflation on ROBOR, by sample end', yt: 'pp', y: { range: [-1, 4] } },
    { r: 0, c: 1, t: `ROBOR after a permanent +1 pp in inflation`, xt: 'quarters', yt: 'pp' }], { cols: 2, h: 370, legendRows: 2 });
  const [a, bb] = G.ax;
  plot('q4-chart', [ln(P4E, lrs, a, K.blue, 'long-run effect', { mode: 'lines+markers', marker: { size: 5, color: K.blue } }), mk([e], [lr], a, K.red, `sample to ${e}: ${lr.toFixed(2)}`, { marker: { size: 12, color: K.red, symbol: 'star' } }),
    ln(hs, path, bb, K.red, 'response of i (pi held 1 pp higher, g unchanged)', { line: { width: 2 } }), hl(bb, [0, 40], lr, K.navy, 'long-run effect', { line: { width: 1, dash: 'dash' } })], G.lay);
  const cf = j => `${f3(b[j])} [${f2(t[j])}]`;
  table('q4-tb', T.q4Head, [['g<sub>t−1</sub>', R_(cf(1)), 'g<sub>t−2</sub>', R_(cf(4))], ['π<sub>t−1</sub>', R_(cf(2)), 'π<sub>t−2</sub>', R_(cf(5))], ['i<sub>t−1</sub>', R_(cf(3)), 'i<sub>t−2</sub>', R_(cf(6))],
    [T.const, R_(cf(0)), 'R<sup>2</sup>', R_(f2(U.r2))]]);
  const Gg = granger(n, 0, 2), Gp = granger(n, 1, 2), Gi = granger(n, 2, 1);
  table('q4-tb2', T.q4Head2, [[T.q4h('g', 'i'), R_(f2(Gg.F)), R_(pv(Gg.p))], [T.q4h('π', 'i'), R_(f2(Gp.F)), R_(pv(Gp.p))], [T.q4h('i', 'π'), R_(f2(Gi.F)), R_(pv(Gi.p))]]);
  const r4 = x => Math.round(x * 1e4) / 1e4, num = r4(b[2]) + r4(b[5]), den = 1 - r4(b[3]) - r4(b[6]), crit = fppf(0.95, 2, Gg.df2);
  const Fr = ((Math.round(Gg.rssR * 100) - Math.round(Gg.rssU * 100)) / 200) / (Math.round(Gg.rssU * 100) / 100 / Gg.df2);
  steps('q4-ws', [T.q4s1(pvr(Gg.p), pvr(Gp.p), pvr(Gi.p), Gg.p < 0.05, Gp.p < 0.05, Gi.p < 0.05), T.q4s2(f3(b[0]), f3(b[1]), f3(b[4]), f3(b[2]), par(b[5], 3), f3(b[3]), par(b[6], 3)),
    T.q4s3(f4(b[2]), par(r4(b[5]), 4), f4(b[3]), par(r4(b[6]), 4), f4(num), f4(den), f2(num / den)),
    T.q4s4(f2(Gg.rssR), f2(Gg.rssU), Gg.df2, f2(Gg.F), f2(crit), Gg.F > crit, f2(Fr), f2(Fr) !== f2(Gg.F))]);
  out('q4-out', [T.q4o(e, U.Tn, f2(lr), f4(den)), ...(e !== Q4[Q4.length - 1] ? [T.extraP] : [])]);
  return { b, t, lr, num, den, F: Gg.F, Fr, crit, Gg, Gp, Gi, r2: U.r2 };
}
// ---- Problem 5: cointegration and VECM output
const P5 = D.p5;
function ecSeries() {
  return memo('ec', () => { const B = P5.beta, c = P5.const; return [0, 1].map(j => P5.y1.map((_, t) => B[0][j] * P5.y1[t] + B[1][j] * P5.y5[t] + B[2][j] * P5.y10[t] + c[j])); });
}
function ecM(k) {                              // I + beta' (k alpha), 2 x 2
  const B = P5.beta, Al = P5.alpha, M = [[1, 0], [0, 1]];
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) for (let r = 0; r < 3; r++) M[i][j] += B[r][i] * k * Al[r][j];
  return M;
}
function eig2(M) { const tr = M[0][0] + M[1][1], dt = M[0][0] * M[1][1] - M[0][1] * M[1][0], d = tr * tr / 4 - dt;
  return d >= 0 ? [tr / 2 + Math.sqrt(d), tr / 2 - Math.sqrt(d)].map(Math.abs).sort((x, y) => y - x) : [Math.sqrt(dt), Math.sqrt(dt)]; }
const hlife = v => Math.log(0.5) / Math.log(v);
function blkQ5() {
  const k = val('q5-k'), eq = +sel('q5-y'); set('q5-k', f2(k));
  const E = ecSeries(), M = ecM(k), ev = eig2(M), hv = ev.map(hlife), nT = P5.y1.length;
  const md = t => { const y = 1960 + Math.floor(t / 12), m = t % 12 + 1; return `${y}-${String(m).padStart(2, '0')}-01`; };
  const t0 = nT - 180, xs = range(t0, nT - 1, 1), mu = E.map(e => amean(e)), dev = [E[0][nT - 1] - mu[0], E[1][nT - 1] - mu[1]], pr = [[], []], hs = range(0, 60, 1);
  let z = dev.slice(); hs.forEach(() => { pr[0].push(mu[0] + z[0]); pr[1].push(mu[1] + z[1]); z = [M[0][0] * z[0] + M[0][1] * z[1], M[1][0] * z[0] + M[1][1] * z[1]]; });
  const ks = linspace(0.25, 2, 36), hk = ks.map(x => eig2(ecM(x)).map(hlife));
  const G = grid('q5-chart', [{ r: 0, c: 0, t: 'equilibrium errors EC1, EC2 and their path without new shocks', yt: 'pp' },
    { r: 0, c: 1, t: 'half-lives of the equilibrium errors', xt: 'multiplier of alpha', yt: 'months', y: { type: 'log', tickvals: [2, 5, 10, 20, 50, 100], ticktext: ['2', '5', '10', '20', '50', '100'] } }], { cols: 2, h: 380, legendRows: 3 });
  const [a, b] = G.ax, fx = hs.map(h => md(nT - 1 + h));
  plot('q5-chart', [ln(xs.map(md), xs.map(t => E[0][t]), a, K.blue, 'EC1 = y1 - 1.045 y10 + 1.223'), ln(xs.map(md), xs.map(t => E[1][t]), a, K.red, 'EC2 = y5 - 1.044 y10 + 0.587'),
    ln(fx, pr[0], a, K.blue, '', { line: { width: 2, dash: 'dot' } }), ln(fx, pr[1], a, K.red, '', { line: { width: 2, dash: 'dot' } }),
    ln(ks, hk.map(h => h[0]), b, K.purple, 'slow component'), ln(ks, hk.map(h => h[1]), b, K.teal, 'fast component'),
    mk([k, k], hv, b, K.navy, `multiplier ${k.toFixed(2)}: ${hv[0].toFixed(1)} and ${hv[1].toFixed(1)} months`, { marker: { size: 10, color: K.navy, symbol: 'diamond' } })], G.lay);
  const Al = P5.alpha, At = P5.alpha_t, nm = ['y<sup>(1)</sup>', 'y<sup>(5)</sup>', 'y<sup>(10)</sup>'];
  steps('q5-ws', [T.q5s1, T.q5s2(nm[eq], f3(k * Al[eq][0]), par(k * Al[eq][1], 3), f2(At[eq][0]), f2(At[eq][1]), f2(At[0][0]), f2(At[0][1])),
    T.q5s3(f3(-P5.beta[2][0]), f3(-P5.beta[2][1]), f3(ev[0]), f3(ev[1]), f1(hv[0]), f1(hv[1]))]);
  out('q5-out', [T.q5o(tr(k, 2), f1(hv[0]), f1(hv[1]), f2(E[0][nT - 1]), f2(E[1][nT - 1])), ...(k !== 1 ? [T.extraP] : [])]);
  return { ev, hv };
}
// ---- Problem 6: GARCH output and VaR 1%
const G6 = D.p6.params;
function blkQ6() {
  const eT = val('q6-e'), s2T = val('q6-s'), nu = val('q6-n'); set('q6-e', f1(eT)); set('q6-s', f1(s2T)); set('q6-n', String(nu));
  const om = G6.omega, al = G6['alpha[1]'], be = G6['beta[1]'], mu = G6.mu, ps = al + be, lv = om / (1 - ps), hlf = Math.log(0.5) / Math.log(ps);
  const s2n = om + al * eT * eT + be * s2T, sn = Math.sqrt(s2n), tq = tppf(0.01, nu), sc = Math.sqrt((nu - 2) / nu), q = tq * sc, zN = normPpf(0.01);
  const ee = linspace(-6, 6, 121), hs = range(1, 250, 1), path = hs.map(h => lv + ps ** (h - 1) * (s2n - lv));
  const G = grid('q6-chart', [{ r: 0, c: 0, t: 'news impact: sigma^2_t+1 against today\'s shock', xt: 'eps_t (%)', yt: 'sigma^2_t+1' },
    { r: 0, c: 1, t: 'forecast of the variance without new shocks', xt: 'trading days ahead', yt: 'E sigma^2_t+h' }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax;
  plot('q6-chart', [ln(ee, ee.map(x => om + al * x * x + be * s2T), a, K.blue, `sigma^2_t = ${s2T.toFixed(1)}`, { line: { width: 2 } }), mk([eT], [s2n], a, K.red, `eps_t = ${eT.toFixed(1)}: ${s2n.toFixed(4)}`, { marker: { size: 11, color: K.red } }),
    ln(hs, path, b, K.red, 'E sigma^2_t+h', { line: { width: 2 } }), hl(b, [1, 250], lv, K.green, `long-run variance ${lv.toFixed(2)}`, { line: { width: 1.2, dash: 'dash' } }),
    mk([1 + hlf], [lv + 0.5 * (s2n - lv)], b, K.navy, `half of the gap closed: ${hlf.toFixed(0)} days`, { marker: { size: 10, color: K.navy, symbol: 'diamond' } })], G.lay);
  const VaR = -(mu + sn * q), VaRn = -(mu + sn * zN);
  steps('q6-ws', [T.q6s1(f4(ps), f0(hlf), f4(om), f2(lv), f1(Math.sqrt(250 * lv))),
    T.q6s2(f4(om), nf(al, 5), tr(eT * eT, 2), f4(be), f1(s2T), f4(al * eT * eT), f4(be * s2T), f4(s2n), f3(sn)),
    T.q6s3(nu, f4(tq), `${nu - 2}/${nu}`, f4(sc), f3(q), f3(mu), f3(sn), f2(VaR), f2(VaRn))]);
  out('q6-out', [T.q6o(f2(VaR), f2(VaRn), f1(D.p6.vol_sample)), ...(eT !== -3 || s2T !== 1 || nu !== 5 ? [T.extraP] : [])]);
  return { ps, hlf, lv, s2n, sn, tq, q, VaR, VaRn, ann: Math.sqrt(250 * lv) };
}
// ---- Problem 7: evaluating forecasts
const EA = [0.3, -0.5, 0.2, 0.6, -0.4, 0.1], EB = [0.5, -0.7, 0.6, 0.4, -0.9, 0.3];
function blkQ7() {
  const lam = val('q7-l'), sc = val('q7-s'); set('q7-l', f2(lam)); set('q7-s', f2(sc));
  const eb = EB.map(v => lam * v), rm = e => Math.sqrt(amean(e.map(x => x * x))), ma = e => amean(e.map(Math.abs));
  const d = EA.map((v, i) => v * v - eb[i] * eb[i]), n = d.length, db = amean(d), sd = Math.sqrt(d.reduce((s, x) => s + (x - db) ** 2, 0) / (n - 1)), dm = db / (sd / Math.sqrt(n)), cr = tppf(0.975, n - 1);
  const ms = range(1, 6, 1);
  const G = grid('q7-chart', [{ r: 0, c: 0, t: 'forecast errors on the six test months', xt: 'test month', yt: 'pp' }, { r: 0, c: 1, t: 'loss differential d_t = e_A^2 - e_B^2', xt: 'test month' }], { cols: 2, h: 350, legendRows: 2 });
  const [a, b] = G.ax;
  plot('q7-chart', [bar(ms, EA, a, K.blue, 'SARIMA'), bar(ms, eb, a, K.amber, `seasonal naive x ${lam.toFixed(2)}`), bar(ms, d, b, d.map(v => (v < 0 ? K.green : K.red)), '', { showlegend: false }),
    hl(b, [0.5, 6.5], db, K.navy, `mean ${db.toFixed(3)}`, { line: { width: 1.5, dash: 'dash' } })], G.lay);
  G.lay.barmode = 'group';
  steps('q7-ws', [T.q7s1(f3(rm(EA)), f3(ma(EA)), f2(ma(EA) / sc), f3(rm(eb)), f3(ma(eb)), f2(ma(eb) / sc), f2(sc)),
    T.q7s2(d.map(x => f2(x)).join('; '), f3(db), f3(sd), f2(dm), f3(cr), Math.abs(dm) > cr), T.q7s3(f2(D.p7.kfold), f2(D.p7.wf))]);
  const lamStar = bisect(L => { const e2 = EB.map(v => L * v), dd = EA.map((v, i) => v * v - e2[i] * e2[i]), m = amean(dd), s = Math.sqrt(dd.reduce((q, x) => q + (x - m) ** 2, 0) / 5); return Math.abs(m / (s / Math.sqrt(6))) < cr; }, 1, 5);
  out('q7-out', [T.q7o(tr(lam, 2), f2(dm), f2(lamStar)), ...(lam !== 1 || sc !== 0.6 ? [T.extraP] : [])]);
  return { rA: rm(EA), mA: ma(EA), rB: rm(eb), mB: ma(eb), db, sd, dm, cr, lamStar };
}
// ---- Problem 8: state space, regimes and long memory
function blkQ8() {
  const s2n = val('q8-q'), y = val('q8-y'), p11 = val('q8-a'), p22 = val('q8-b'), d = val('q8-d'); set('q8-q', f2(s2n)); set('q8-y', f1(y)); set('q8-a', f4(p11)); set('q8-b', f4(p22)); set('q8-d', f2(d));
  const s2e = 4, a0 = 10, P0 = 2, F = P0 + s2e, Kt = P0 / F, af = a0 + Kt * (y - a0), Pf = P0 * (1 - Kt), Pn = Pf + s2n;
  const q = s2n / s2e, Pb = (q + Math.sqrt(q * q + 4 * q)) / 2, Kb = Pb / (Pb + 1);
  const ts = range(1, 15, 1), Ks = []; let P = P0; ts.forEach(() => { const k = P / (P + s2e); Ks.push(k); P = P * (1 - k) + s2n; });
  const G = grid('q8-chart', [{ r: 0, c: 0, t: 'Kalman gain K_t from P_t = 2 (sigma^2_eps = 4)', xt: 't', yt: 'K_t', y: { range: [0, 1] } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0];
  plot('q8-chart', [ln(ts, Ks, a, K.blue, `K_t, sigma^2_eta = ${s2n.toFixed(2)}`, { mode: 'lines+markers', marker: { size: 6, color: K.blue }, line: { width: 2 } }),
    hl(a, [1, 15], Kb, K.red, `steady-state gain = SES weight ${Kb.toFixed(3)}`, { line: { width: 1.4, dash: 'dash' } })], G.lay);
  const D1 = 1 / (1 - p11), D2 = 1 / (1 - p22), pi1 = (1 - p22) / (2 - p11 - p22);
  steps('q8-ws', [T.q8s1(f3(Kt), tr(y, 1), tr(y - a0, 1), f2(af), f3(Pf), f2(s2n), f3(Pn)), T.q8s2(f2(q), f4(Pb), f3(Kb)),
    T.q8s3(f4(p11), f4(p22), f1(D1), f1(D2), f4(1 - p22), f4(2 - p11 - p22), f3(pi1)), T.q8s4(f2(d), f2(d + 0.5), f2(d / (1 - d)), d > 0 && d < 0.5 ? 'st' : d >= 0.5 && d < 1 ? 'nst' : d <= 0 && d > -0.5 ? 'anti' : 'no')]);
  out('q8-out', [T.q8o(tr(s2n, 2), f3(Kb), f0(100 * pi1)), ...(s2n !== 1 || y !== 13 || Math.abs(p11 - D.p8.p11) > 1e-9 || Math.abs(p22 - D.p8.p22) > 1e-9 || Math.abs(d - D.p8.d) > 1e-9 ? [T.extraP] : [])]);
  return { F, Kt, af, Pf, Pn, Pb, Kb, D1, D2, pi1, H: d + 0.5, r1: d / (1 - d) };
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
  return [['data', 'da', blkDa, 'da-e', range(2010, 2026, 1), 450], ['t-data', 'td', blkTd, 'td-s', range(2005, 2020, 1), 450],
    ['t-tests', 'tt', blkTt, 'tt-e', null, 700], ['acf', 'ac', blkAc, 'ac-s', null, 1300], ['t-acf', 'ta', blkTa, 'ta-p', range(-0.6, 0.9, 0.1), 400],
    ['t-grid', 'tg', blkTg, 'tg-c', range(2, 9, 0.5), 450], ['t-chosen', 'tc', blkTc, 'tc-m', null, 1200], ['diag', 'dg', blkDg, 'dg-m', null, 1200],
    ['t-diag', 'tx', blkTx, 'tx-m', range(6, 36, 2), 400], ['fc', 'fc', blkFc, 'fc-o', range(0, ORG.length - 1, 1), 600], ['t-fc', 'tf', blkTf, 'tf-n', range(6, 24, 1), 450],
    ['p1', 'q1', blkQ1, 'q1-f', range(0, 0.95, 0.05), 350], ['p2', 'q2', blkQ2, 'q2-e', null, 700], ['p3', 'q3', blkQ3, 'q3-o', range(0, 23, 1), 450],
    ['p4', 'q4', blkQ4, 'q4-e', null, 700], ['p5', 'q5', blkQ5, 'q5-k', range(0.25, 2, 0.25), 450], ['p6', 'q6', blkQ6, 'q6-e', range(-6, 6, 0.5), 350],
    ['p7', 'q7', blkQ7, 'q7-l', range(0.5, 2, 0.1), 400], ['p8', 'q8', blkQ8, 'q8-q', range(0.25, 8, 0.25), 350]];
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
window.CH15 = { acfAdj, pacfYW, lbox, moments, chi2sf, t2, fsf, tppf, fppf, chi2ppf, psi, armaAcf, gridPick, diagOf, accOf, dmTest, olsEq, granger, lrOf, ecM, eig2, hand, monthMeans,
  verdict, polys, qroots, sroots, TAB5, blkList, init, W, A, Z, HD,
  blk: { blkDa, blkTd, blkTt, blkAc, blkTa, blkTg, blkTc, blkDg, blkTx, blkFc, blkTf, blkQ1, blkQ2, blkQ3, blkQ4, blkQ5, blkQ6, blkQ7, blkQ8 } };
if (!window.CH15_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
