// Chapter 9 interactive page (machine learning for time series): one block per chart slide, plus the result tables and the
// worked examples of the lecture, computed in the browser from ch9_data.js (Quantlets/Ch_09/export_interactive.py).
(function () {
'use strict';
const D = window.CH9_DATA;
const { sum, mean, nf, $, AX, layout, plot, Player, range, predict, exitTicket, normCdf } = window.TSAI;
const out = (id, items) => window.TSAI.out(id, items);
const T = window.CH9_TEXT;
const L = T.L;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of tsa_style.py / generate_all_charts.py
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', orange: '#E67E22', purple: '#8E44AD', teal: '#17A2B8', crimson: '#DC3545',
  olive: '#6B8E23', navy: '#1F2A44', black: '#000000' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.amber]: '181,133,63', [K.orange]: '230,126,34', [K.purple]: '142,68,173',
  [K.teal]: '23,162,184', [K.crimson]: '220,53,69', [K.navy]: '31,42,68', [K.olive]: '107,142,35' };
const MCOL = { 'Seasonal naive': K.amber, ETS: K.teal, SARIMA: K.purple, DHR: K.blue, Combination: K.crimson, Ridge: K.teal, RF: K.green, 'GB direct': K.red,
  'GB recursive': K.orange, MLP: K.olive, LSTM: K.purple, RW: K.amber, AR: K.blue, Lasso: K.teal, 'GB local': K.red, 'GB global': K.orange, HAR: K.blue, GB: K.red,
  Mean: K.amber, Logit: K.blue };
const STAT = ['Seasonal naive', 'ETS', 'SARIMA', 'DHR', 'Combination'];

// ---------------------------------------------------------------- small helpers
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
const has = id => !!$(id);
const cache = {};
const memo = (k, f) => (k in cache ? cache[k] : (cache[k] = f()));
const f1 = x => nf(x, 1), f2 = x => nf(x, 2), f3 = x => nf(x, 3), f4 = x => nf(x, 4);
const mw = x => nf(1000 * x, 0);
const pc = (x, d) => nf(100 * x, d === undefined ? 1 : d) + '%';
const sgn = (x, d) => (x >= 0 ? '+' : '') + nf(x, d);
const pv = p => (p < 0.001 ? (T.lt || '< ') + nf(0.001, 3) : nf(p, p < 0.01 ? 4 : 3));
const sdp = x => { const m = mean(x); return Math.sqrt(sum(x.map(v => (v - m) ** 2)) / x.length); };
const corr = (a, b) => { const ma = mean(a), mb = mean(b); let s = 0, sa = 0, sb = 0; for (let i = 0; i < a.length; i++) { s += (a[i] - ma) * (b[i] - mb); sa += (a[i] - ma) ** 2; sb += (b[i] - mb) ** 2; } return s / Math.sqrt(sa * sb); };
const addDays = (iso, k) => new Date(Date.parse(iso + 'T00:00:00Z') + k * 864e5).toISOString().slice(0, 10);
const dayIdx = (d0, iso) => Math.round((Date.parse(iso + 'T00:00:00Z') - Date.parse(d0 + 'T00:00:00Z')) / 864e5);
const dow = iso => (new Date(iso + 'T00:00:00Z').getUTCDay() + 6) % 7;      // Monday = 0, ..., Sunday = 6 (pandas dayofweek)
const isoM = (m0, k) => { const y0 = +m0.slice(0, 4), mm = +m0.slice(5, 7) - 1 + k; return `${y0 + Math.floor(mm / 12)}-${String(mm % 12 + 1).padStart(2, '0')}-01`; };
const months = (m0, n) => Array.from({ length: n }, (_, k) => isoM(m0, k));
const ols1 = (x, y) => { const mx = mean(x), my = mean(y); let a = 0, b = 0; for (let i = 0; i < x.length; i++) { a += (x[i] - mx) * (y[i] - my); b += (x[i] - mx) ** 2; } const s = a / b; return [my - s * mx, s]; };

// ---------------------------------------------------------------- distributions: Student t (regularised incomplete beta)
function lgam(x) { const c = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, t = x + 5.5; t -= (x + 0.5) * Math.log(t); let s = 1.000000000190015; for (const v of c) s += v / ++y; return -t + Math.log(2.5066282746310005 * s / x); }
function betacf(a, b, x) {
  let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap; if (Math.abs(d) < 1e-300) d = 1e-300; d = 1 / d; let h = d;
  for (let m = 1; m <= 300; m++) { const m2 = 2 * m; let aa = m * (b - m) * x / ((qam + m2) * (a + m2)); d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2)); d = 1 + aa * d; if (Math.abs(d) < 1e-300) d = 1e-300; c = 1 + aa / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const del = d * c; h *= del; if (Math.abs(del - 1) < 3e-16) break; }
  return h;
}
function ibeta(a, b, x) { if (x <= 0) return 0; if (x >= 1) return 1; const bt = Math.exp(lgam(a + b) - lgam(a) - lgam(b) + a * Math.log(x) + b * Math.log(1 - x));
  return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b; }
const tP2 = (t, df) => ibeta(df / 2, 0.5, df / (df + t * t));          // two-sided p-value of t with df degrees of freedom
const nP2 = z => 2 * (1 - normCdf(Math.abs(z)));

// ---------------------------------------------------------------- the chapter's tests
function dmRaw(l1, l2, h) {                  // dm_raw: d = l1 - l2, h - 1 autocovariances (rectangular), HLN correction, t(n - 1)
  h = h || 1; const d = l1.map((v, i) => v - l2[i]), n = d.length, db = mean(d), g = [];
  for (let k = 0; k < h; k++) { let s = 0; for (let t = k; t < n; t++) s += (d[t] - db) * (d[t - k] - db); g.push(s / (n - k)); }
  const v = (g[0] + 2 * sum(g.slice(1))) / n, dm = db / Math.sqrt(v), hln = Math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n) * dm;
  return { d, dbar: db, g, v, dm, hln, p: tP2(Math.abs(hln), n - 1), n };
}
function dmHac(d, lag) {                     // dm_hac: Newey-West (Bartlett) long-run variance, N(0, 1) p-value
  const n = d.length, m = mean(d), u = d.map(v => v - m); let lrv = sum(u.map(v => v * v)) / n;
  for (let j = 1; j <= lag; j++) { let s = 0; for (let t = j; t < n; t++) s += u[t] * u[t - j]; lrv += 2 * (1 - j / (lag + 1)) * s / n; }
  const t = m / Math.sqrt(lrv / n); return { t, p: nP2(t) };
}
function auc(y, p) {                         // roc_auc_score: ranks with ties averaged
  const n = y.length, ix = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b]), r = new Float64Array(n);
  for (let i = 0; i < n;) { let j = i; while (j + 1 < n && p[ix[j + 1]] === p[ix[i]]) j++; const rk = (i + j) / 2 + 1; for (let k = i; k <= j; k++) r[ix[k]] = rk; i = j + 1; }
  let n1 = 0, s = 0; for (let i = 0; i < n; i++) if (y[i]) { n1++; s += r[i]; }
  return (s - n1 * (n1 + 1) / 2) / (n1 * (n - n1));
}

// ---------------------------------------------------------------- regression trees (scikit-learn's exact greedy CART: float32 features,
// midpoint thresholds, squared error, min_samples_leaf)
function cart(X, y, idx, maxDepth, minLeaf) {
  const Xf = X.map(c => Float32Array.from(c));
  function node(ix, depth) {
    const n = ix.length; let s = 0, ss = 0; for (const i of ix) { s += y[i]; ss += y[i] * y[i]; }
    const nd = { n, value: s / n, sse: ss - s * s / n, ix };
    if (depth >= maxDepth || n < 2 * minLeaf || n < 2 || nd.sse <= 1e-12 * n) return nd;
    let best = null, bi = -Infinity;
    Xf.forEach((col, f) => {
      const o = [...ix].sort((a, b) => col[a] - col[b]); let sl = 0;
      for (let p = 1; p < n; p++) {
        sl += y[o[p - 1]];
        if (p < minLeaf || n - p < minLeaf) continue;
        if (!(col[o[p]] > col[o[p - 1]] + 1e-7)) continue;
        const sr = s - sl, imp = sl * sl / p + sr * sr / (n - p);
        if (imp > bi) { bi = imp; let thr = (col[o[p - 1]] + col[o[p]]) / 2; if (thr === col[o[p]]) thr = col[o[p - 1]]; best = { f, thr }; }
      }
    });
    if (!best || bi - s * s / n <= 0) return nd;
    const lft = ix.filter(i => Xf[best.f][i] <= best.thr), rgt = ix.filter(i => Xf[best.f][i] > best.thr);
    return Object.assign(nd, { feat: best.f, thr: best.thr, left: node(lft, depth + 1), right: node(rgt, depth + 1) });
  }
  const root = node(idx, 0);
  root.predict = row => { let nd = root; while (nd.left) nd = Math.fround(row[nd.feat]) <= nd.thr ? nd.left : nd.right; return nd.value; };
  return root;
}
const leaves = nd => (nd.left ? [...leaves(nd.left), ...leaves(nd.right)] : [nd]);

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
    legend: { orientation: 'h', x: 0, xanchor: 'left', y: -(narrow ? 34 : 40) / plotH, yanchor: 'top', bgcolor: 'rgba(0,0,0,0)', font: { size: 11 } } });
  const ax = cells.map((c, k) => {
    const r = narrow ? k : c.r, cc = narrow ? 0 : c.c, cs = narrow ? 1 : (c.cs || 1), rs = narrow ? 1 : (c.rs || 1);
    const x0 = cc * (w + gx), x1 = x0 + cs * w + (cs - 1) * gx, y1 = 1 - r * (h + gy), y0 = y1 - rs * h - (rs - 1) * gy, n = k ? k + 1 : '';
    lay['xaxis' + n] = AX(c.xt || '', Object.assign({ domain: [x0, Math.min(1, x1)], anchor: 'y' + n, automargin: false, title: { text: c.xt || '', font: { size: 11, color: K.navy }, standoff: 4 } }, c.x || {}));
    lay['yaxis' + n] = AX(c.yt || '', Object.assign({ domain: [Math.max(0, y0), y1], anchor: 'x' + n, automargin: c.yam || false, title: { text: c.yt || '', font: { size: 11, color: K.navy }, standoff: 2 } }, c.y || {}));
    if (c.t) lay.annotations.push({ xref: 'paper', yref: 'paper', x: x0, y: y1, xanchor: 'left', yanchor: 'bottom', yshift: 3, showarrow: false, text: c.t, font: { size: 12, color: K.navy } });
    return { xa: 'x' + n, ya: 'y' + n, xd: 'x' + n + ' domain', yd: 'y' + n + ' domain', k: k + 1 };
  });
  el.style.height = H + 'px';
  return { lay, ax, narrow };
}
const hl = (a, y, color, dash, width) => ({ type: 'line', xref: a.xd, x0: 0, x1: 1, yref: a.ya, y0: y, y1: y, line: { color, width: width || 1.2, dash: dash || 'solid' } });
const vl = (a, x, color, dash, width) => ({ type: 'line', yref: a.yd, y0: 0, y1: 1, xref: a.xa, x0: x, x1: x, line: { color, width: width || 1.2, dash: dash || 'solid' } });
const leg = (name, line, marker) => Object.assign({ x: [null], y: [null], name, hoverinfo: 'skip' }, line ? { mode: 'lines', line } : { mode: 'markers', marker });
const ln = (x, y, a, color, name, o) => { const r = Object.assign({ x, y, mode: 'lines', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }, o || {}); r.line = Object.assign({ color, width: 1.4 }, (o && o.line) || {}); return r; };
const mk = (x, y, a, color, name, o) => Object.assign({ x, y, mode: 'markers', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, marker: { color, size: 6 }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }, o || {});
const bar = (x, y, a, color, name, o) => Object.assign({ x, y, type: 'bar', marker: { color }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }, o || {});
const band = (x, lo, hi, a, col, al, name) => ({ x: [...x, ...[...x].reverse()], y: [...hi, ...[...lo].reverse()], fill: 'toself', fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, mode: 'lines', hoverinfo: 'skip', name: name || '', showlegend: !!name, xaxis: a.xa, yaxis: a.ya });
const hline = (a, xs, y, color, name, o) => ln(xs, xs.map(() => y), a, color, name, Object.assign({ hoverinfo: 'skip' }, o || {}));      // a reference line as a trace
const vline = (a, x, y0, y1, color, name, o) => ln([x, x], [y0, y1], a, color, name, Object.assign({ hoverinfo: 'skip' }, o || {}));
const DATE = { type: 'date' }, LOG = { type: 'log', exponentformat: 'power' };

// ---------------------------------------------------------------- tables and worked steps (HTML)
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}>${v.v}</${tag}>`; };
  el.innerHTML = (o && o.cap ? `<p class="c9cap">${o.cap}</p>` : '') + '<table><thead><tr>' + head.map(c => cell(c, 'th')).join('') + '</tr></thead><tbody>'
    + rows.map(r => '<tr' + (r.hi ? ' class="hi"' : '') + '>' + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c9cap">${o.note}</p>` : '');
}
const num = (x, d, bold) => ({ v: bold ? `<b>${nf(x, d)}</b>` : nf(x, d), cls: 'r' });
const txt = (v, bold) => ({ v: bold ? `<b>${v}</b>` : v, cls: 'r' });
const H_ = v => ({ v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const hiRow = (r, on) => { r.hi = on; return r; };
const MN = m => (T.model && T.model[m]) || m;       // model name in the page language (tables and readouts only)

// ---------------------------------------------------------------- data of the chapter
const LD = D.load, LOADV = LD.v, NLD = LOADV.length, LDATES = Array.from({ length: NLD + 14 }, (_, i) => addDays(LD.d0, i));
const ORIG = D.err.origins, NOR = ORIG.length, MODELS = D.err.models;
const actualAt = (o, h) => LOADV[dayIdx(LD.d0, addDays(o, h))];
const ACT = ORIG.flatMap(o => range(1, 14, 1).map(h => actualAt(o, h)));          // origin-major, horizon-minor, as D.err.fc
const ERR = m => memo('err|' + m, () => D.err.fc[m].map((f, i) => ACT[i] - f));
const SCALE = memo('scale', () => { const n = dayIdx(LD.d0, ORIG[0]) + 1, tr = LOADV.slice(0, n); let s = 0; for (let t = 7; t < n; t++) s += Math.abs(tr[t] - tr[t - 7]); return s / (n - 7); });
const HOL = i => LD.easter[i] + LD.xmas[i] + LD.holiday[i] > 0;

// ================================================================= 1. the load: the series and three features
function featState(h) {
  const k = Math.ceil(h / 7), r = { t: [], lag0: [], same: [], m7: [], y: [], sun: [] };
  for (let t = 27; t + h < NLD; t++) {
    let m = 0; for (let j = 0; j < 7; j++) m += LOADV[t - j];
    r.t.push(t); r.lag0.push(LOADV[t]); r.same.push(LOADV[t + h - 7 * k]); r.m7.push(m / 7); r.y.push(LOADV[t + h]); r.sun.push(dow(LDATES[t + h]) === 6);
  }
  return Object.assign(r, { k, c0: corr(r.lag0, r.y), c6: corr(r.same, r.y), cm: corr(r.m7, r.y), n: r.t.length });
}
function blkFe() {
  const h = +$('fe-h').value; set('fe-h', h);
  const S = memo('fe|' + h, () => featState(h)), dd = LDATES.slice(0, NLD);
  const G = grid('fe-chart', [{ r: 0, c: 0, cs: 3, yt: 'GW', x: DATE }, { r: 1, c: 0, xt: `y_t (yesterday, for the target y_t+${h})`, yt: `y_t+${h} (GW)` },
    { r: 1, c: 1, xt: `y_t${h - 7 * S.k ? (h - 7 * S.k > 0 ? '+' : '') + (h - 7 * S.k) : ''} (same weekday, ${S.k === 1 ? 'last week' : S.k + ' weeks back'})` }, { r: 1, c: 2, xt: '7-day mean at t' }], { cols: 3, h: 540, legendRows: 2, gyPx: 70 });
  const [a, b, c, e] = G.ax, hi = dd.map((_, i) => i).filter(HOL), col = S.sun.map(s => (s ? K.orange : K.teal));
  const fx = dd.map((_, j) => (j >= 7 * S.k ? LOADV[j - 7 * S.k] : null));
  const sc = (x, ax) => mk(x, S.y, ax, col, '', { marker: { size: 3, color: col, opacity: 0.5 }, hovertemplate: '%{x:.2f} → %{y:.2f}<extra></extra>' });
  plot('fe-chart', [ln(dd, LOADV, a, K.blue, 'daily mean load (GW)', { line: { width: 0.9 } }),
    ln(dd, fx, a, K.teal, `same weekday ${S.k} week${S.k > 1 ? 's' : ''} before (the feature for h = ${h})`, { line: { width: 0.7 }, opacity: 0.6 }),
    mk(hi.map(i => dd[i]), hi.map(i => LOADV[i]), a, K.red, 'public holidays', { marker: { size: 4, color: K.red } }),
    sc(S.lag0, b), sc(S.same, c), sc(S.m7, e), leg('target on Monday to Saturday', null, { color: K.teal, size: 7 }), leg('target on Sunday', null, { color: K.orange, size: 7 })], G.lay);
  const F = NB.features;
  out('fe-out', [T.fe1(f2(F.mean), f2(F.min), T.dmy(F.min_d), f2(F.max), T.dmy(F.max_d), F.n), T.fe2(h, S.k, S.n, f2(S.c0), f2(S.c6), f2(S.cm))]);
}

// ================================================================= EXTRA. a lag table for the electricity load
const WL_END = '2025-06-29';
function blkWl() {
  const off = +$('wl-e').value, p = +$('wl-p').value, h = +$('wl-h').value, end = addDays(WL_END, off), ie = dayIdx(LD.d0, end), i0 = ie - 8;
  set('wl-e', T.dmy(end)); set('wl-p', p); set('wl-h', h);
  const dd = LDATES.slice(i0, ie + 1), v = LOADV.slice(i0, ie + 1), dw = dd.map(dow);
  const G = grid('wl-chart', [{ r: 0, c: 0, yt: 'GW', x: { tickvals: dd, ticktext: dd.map((d, j) => `${['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][dw[j]]}<br>${d.slice(5)}`) } }], { cols: 1, h: 300, legendRows: 1 });
  const ax = G.ax[0], rows = [];
  for (let j = p - 1; j + h <= 8; j++) rows.push(j);
  plot('wl-chart', [bar(dd, v, ax, dw.map(w => (w === 6 ? K.orange : K.blue)), 'daily mean load (GW)', { opacity: 0.85 }),
    mk(rows.map(j => dd[j + h]), rows.map(j => v[j + h]), ax, K.red, `targets y_t+${h} of the table`, { marker: { size: 9, symbol: 'diamond', color: K.red } }),
    leg('Sunday', null, { color: K.orange, symbol: 'square', size: 10 })], Object.assign(G.lay, { yaxis: Object.assign(G.lay.yaxis, { range: [3.5, 7.2] }) }));
  table('wl-tb', [T.L.origin, ...range(0, p - 1, 1).map(j => H_(j ? `y<sub>t−${j}</sub>` : 'y<sub>t</sub>')), H_(`${T.L.target} y<sub>t+${h}</sub>`)],
    rows.map(j => [T.wd[dw[j]], ...range(0, p - 1, 1).map(l => num(v[j - l], 2)), num(v[j + h], 2, true)]), { cap: T.wlCap(T.dmy(dd[0]), T.dmy(dd[8])) });
  steps('wl-ws', [T.wls1(9, p, h, 9 - p - h + 1), T.wls2(NB.features.n, h, NB.features.n - 28 - h + 1), T.wls3(T.wd[dw[rows[0] + h]])]);
  out('wl-out', [T.wl1(rows.length, p, h, T.dmy(dd[rows[0]]), f2(v[rows[0]]), f2(v[rows[0] + h]))]);
}

// ================================================================= 2. recursive and direct forecasts (extra run: linear models on the load)
function stState(o) {
  const io = dayIdx(LD.d0, o), fit = hh => { const x = [], y = []; for (let t = 0; t + hh <= io; t++) { x.push(LOADV[t]); y.push(LOADV[t + hh]); } return ols1(x, y); };
  const [a, b] = fit(1), dir = range(1, 14, 1).map(fit), yT = LOADV[io], rec = [];
  let f = yT; for (let hh = 1; hh <= 14; hh++) { f = a + b * f; rec.push(f); }
  const act = range(1, 14, 1).map(hh => LOADV[io + hh]), dfc = dir.map(([ah, bh]) => ah + bh * yT);
  return { io, a, b, dir, rec, dfc, act, yT, maeR: mean(act.map((v, i) => Math.abs(v - rec[i]))), maeD: mean(act.map((v, i) => Math.abs(v - dfc[i]))) };
}
function blkSt() {
  const o = $('st-o').value, S = memo('st|' + o, () => stState(o)), hh = range(1, 14, 1);
  const G = grid('st-chart', [{ r: 0, c: 0, t: `Load: last 28 days and 14-day forecasts from ${o}`, yt: 'GW', x: DATE }, { r: 0, c: 1, t: 'Coefficient on y_T by horizon', xt: 'horizon h (days)' }], { cols: 2, h: 380, legendRows: 3 });
  const [a, b] = G.ax, past = range(S.io - 27, S.io, 1), fd = hh.map(k => LDATES[S.io + k]);
  plot('st-chart', [ln(past.map(i => LDATES[i]), past.map(i => LOADV[i]), a, K.black, 'actual daily load', { mode: 'lines+markers', marker: { size: 3 } }),
    ln(fd, S.act, a, K.black, '', { mode: 'lines+markers', marker: { size: 3 }, line: { dash: 'dot' } }),
    ln(fd, S.rec, a, K.orange, 'recursive: one AR(1), iterated', { mode: 'lines+markers', marker: { size: 4 } }),
    ln(fd, S.dfc, a, K.red, 'direct: one AR(1) per horizon', { mode: 'lines+markers', marker: { size: 4, symbol: 'diamond' } }),
    vline(a, LDATES[S.io], 3.5, 8, K.blue, '', { line: { dash: 'dot', width: 1 } }),
    ln(hh, hh.map(k => S.b ** k), b, K.orange, 'recursive: implied b^h', { mode: 'lines+markers', marker: { size: 4 } }),
    ln(hh, S.dir.map(d => d[1]), b, K.red, 'direct: estimated b_h', { mode: 'lines+markers', marker: { size: 4, symbol: 'diamond' } })], G.lay);
  out('st-out', [T.st1(T.dmy(o), f3(S.a), f3(S.b), f3(S.b * S.b), f3(S.dir[1][1]), f3(S.b ** 7), f3(S.dir[6][1])), T.st2(mw(S.maeR), mw(S.maeD))]);
}

// ================================================================= EXTRA. recursive and direct two-step forecasts
function blkWs() {
  const yT = +$('ws-y').value, b = +$('ws-b').value, a = 1.2; set('ws-y', f2(yT)); set('ws-b', f2(b));
  const f1_ = a + b * yT, f2r = a + b * f1_, ia = a * (1 + b), ib = b * b, fd = 2.3 + 0.6 * yT, xs = range(3, 8, 0.1);
  const G = grid('ws-chart', [{ r: 0, c: 0, t: 'Two-step forecast against the last value y_T', xt: 'y_T' }, { r: 0, c: 1, t: `Recursive path from y_T = ${yT.toFixed(1)}`, xt: 'horizon h' }], { cols: 2, h: 340, legendRows: 2 });
  const [p, q] = G.ax, hh = range(0, 8, 1), path = [yT]; for (let k = 1; k <= 8; k++) path.push(a + b * path[k - 1]);
  plot('ws-chart', [ln(xs, xs.map(x => ia + ib * x), p, K.orange, `recursive: ${ia.toFixed(2)} + ${ib.toFixed(2)} y_T`), ln(xs, xs.map(x => 2.3 + 0.6 * x), p, K.red, 'direct: 2.3 + 0.6 y_T'),
    mk([yT, yT], [f2r, fd], p, [K.orange, K.red], '', { marker: { size: 10, color: [K.orange, K.red] } }),
    ln(hh, path, q, K.orange, 'recursive forecasts', { mode: 'lines+markers', marker: { size: 6 } }), mk([2], [fd], q, K.red, 'direct two-step forecast', { marker: { size: 11, symbol: 'diamond', color: K.red } }),
    hline(q, [0, 8], a / (1 - b), K.blue, `mean of the AR(1), a/(1 - b) = ${(a / (1 - b)).toFixed(2)}`, { line: { dash: 'dot', width: 1 } })], G.lay);
  steps('ws-ws', [T.wss1(f2(yT), f2(b), f2(f1_), f2(f2r)), T.wss2(f2(b), f2(ia), f2(ib)), T.wss3(f2(yT), f2(fd), f2(f2r - fd))]);
  out('ws-out', [T.ws1(f2(yT), f2(f2r), f2(fd))]);
}

// ================================================================= 3. four validation schemes
function cvState(k, h, n) {
  n = n || 48; const perm = D.perm48, folds = [], q = Math.floor(n / k), r = n % k; let s = 0;
  for (let i = 0; i < k; i++) { const m = q + (i < r ? 1 : 0); folds.push(perm.slice(s, s + m)); s += m; }
  const rows = [], z = [], size = Math.floor(n / (k + 2));
  for (let i = 0; i < k; i++) { const lab = new Array(n).fill(1); folds[i].forEach(j => { lab[j] = 2; }); rows.push(['random K-fold', i + 1]); z.push(lab); }
  ['expanding', 'rolling', 'gap'].forEach(sc => { for (let i = 0; i < k; i++) {
    const lab = new Array(n).fill(0), end = size * (i + 2), st = sc === 'rolling' ? end - 2 * size : 0;
    for (let t = st; t < end; t++) lab[t] = 1; if (sc === 'gap') for (let t = end - h; t < end; t++) lab[t] = 3; for (let t = end; t < Math.min(n, end + size); t++) lab[t] = 2;
    rows.push([{ expanding: 'walk-forward, expanding', rolling: 'walk-forward, rolling', gap: 'walk-forward with a gap' }[sc], i + 1]); z.push(lab); } });
  const ov = lab => { let o = 0, nt = 0; lab.forEach((v, t) => { if (v !== 2) return; nt++; for (let j = Math.max(0, t - h + 1); j <= Math.min(n - 1, t + h - 1); j++) if (j !== t && lab[j] === 1) { o++; break; } }); return [o, nt]; };
  const share = g => { let o = 0, nt = 0; z.slice(g * k, g * k + k).forEach(lab => { const [a, b] = ov(lab); o += a; nt += b; }); return o / nt; };
  return { rows, z, size, kf: share(0), ex: share(1), gp: share(3) };
}
function blkCv() {
  const k = +$('cv-k').value, h = +$('cv-g').value; set('cv-k', k); set('cv-g', h);
  const S = cvState(k, h), G = grid('cv-chart', [{ r: 0, c: 0, xt: 'time', y: { autorange: 'reversed', tickfont: { size: 10 } } }], { cols: 1, h: 120 + 16 * S.rows.length, rowPx: 60 + 16 * S.rows.length, legendRows: 2, ml: 170 });
  const ax = G.ax[0], CS = ['#EAF0F8', K.blue, K.red, K.amber], cs = [[0, CS[0]], [0.25, CS[0]], [0.25, CS[1]], [0.5, CS[1]], [0.5, CS[2]], [0.75, CS[2]], [0.75, CS[3]], [1, CS[3]]];
  const ylab = S.rows.map(([g, i]) => `${g}, ${g === 'random K-fold' ? 'fold' : 'step'} ${i}`);
  plot('cv-chart', [{ type: 'heatmap', z: S.z, x: range(1, 48, 1), y: ylab, zmin: -0.5, zmax: 3.5, colorscale: cs, showscale: false, xgap: 1, ygap: 2, xaxis: ax.xa, yaxis: ax.ya,
    hovertemplate: '%{y}, t = %{x}<extra></extra>' },
  ...['training', 'test', 'gap (target overlaps the test block)', 'not used'].map((nm, i) => leg(nm, null, { color: CS[[1, 2, 3, 0][i]], symbol: 'square', size: 12, line: { color: K.blue, width: i === 3 ? 1 : 0 } }))], G.lay);
  out('cv-out', [T.cv1(k, S.size, h), T.cv2(h, pc(S.kf, 0), pc(S.ex, 0), pc(S.gp, 0))]);
}

// ================================================================= 4. leakage: random folds on overlapping targets
const LKH = D.leak.h, LKS = { kfold: 'random 5-fold (shuffled)', wf: 'walk-forward', gap: 'walk-forward with a gap of h days' }, LKC = { kfold: K.red, wf: K.blue, gap: K.green };
const r2p = F => 1 - sum(F.map(f => f[1])) / sum(F.map(f => f[2]));
function blkLk() {
  const h = $('lk-h').value, R = D.leak.res[h];
  const G = grid('lk-chart', [{ r: 0, c: 0, t: `Target: sum of the next ${h} daily returns`, yt: 'out-of-sample R^2 (%)' }, { r: 0, c: 1, t: 'R^2 against the horizon of the summed target', xt: 'h (days, log scale)', x: { type: 'log', tickvals: LKH } }],
    { cols: 2, h: 380, legendRows: 3 });
  const [a, b] = G.ax, xs = ['simulated random walk<br>(nothing to predict)', 'S&P 500, 2000-2026'], tr = [];
  Object.keys(LKS).forEach(s => { const v = ['sim', 'sp500'].map(q => 100 * r2p(R[q][s]));
    tr.push(bar(xs, v, a, LKC[s], LKS[s].replace('h days', h + ' days'), { text: v.map(x => x.toFixed(1)), textposition: 'outside', cliponaxis: false }));
    ['sim', 'sp500'].forEach(q => tr.push(ln(LKH, LKH.map(hh => 100 * r2p(D.leak.res[hh][q][s])), b, LKC[s], '', { mode: 'lines+markers', marker: { size: 5 }, line: { dash: q === 'sim' ? 'dash' : 'solid', width: 1.4 } }))); });
  tr.push(mk(Array(6).fill(+h), Object.keys(LKS).flatMap(s => ['sim', 'sp500'].map(q => 100 * r2p(R[q][s]))), b, K.navy, `h = ${h}`, { marker: { size: 11, symbol: 'circle-open', color: K.navy, line: { width: 2 } } }),
    leg('dashed: simulated random walk; solid: S&P 500', { color: K.navy, dash: 'dash', width: 1 }));
  G.lay.barmode = 'group'; G.lay.shapes.push(hl(a, 0, K.black, 'solid', 0.8), hl(b, 0, K.black, 'solid', 0.8));
  plot('lk-chart', tr, G.lay);
  const v = q => Object.keys(LKS).map(s => pc(r2p(R[q][s])));
  out('lk-out', [T.lk1(h, ...v('sim'), ...v('sp500')), T.lk2(h, h - 1, f3((h - 1) / h), R.sp500.n)]);
}

// ================================================================= EXTRA. the leakage experiment, step by step
function targets(r, h) { const n = r.length, Y = []; for (let t = 62; t + h < n; t++) { let s = 0; for (let j = 1; j <= h; j++) s += r[t + j]; Y.push(s); } return Y; }
function blkWk() {
  const q = $('wk-s').value, h = $('wk-h').value, sc = $('wk-c').value, R = D.leak.res[h][q];
  const Y = memo(`wk|${q}|${h}`, () => targets(q === 'sim' ? D.leak.sim : D.leak.sp.r, +h)), lags = range(1, 30, 1);
  const m = mean(Y), g0 = sum(Y.map(v => (v - m) ** 2)), ac = lags.map(k => { let s = 0; for (let t = k; t < Y.length; t++) s += (Y[t] - m) * (Y[t - k] - m); return s / g0; });
  const G = grid('wk-chart', [{ r: 0, c: 0, t: 'R^2 of each test fold (%)', xt: 'fold' }, { r: 0, c: 1, t: `ACF of the target (sum of the next ${h} returns)`, xt: 'lag k (days)' }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax, tr = Object.keys(LKS).map(s => bar([1, 2, 3, 4, 5], R[s].map(f => 100 * (1 - f[1] / f[2])), a, LKC[s], LKS[s].replace('h days', h + ' days')));
  tr.push(bar(lags, ac, b, `rgba(${RGB[K.blue]},0.7)`, 'sample ACF'), ln(lags, lags.map(k => Math.max(0, 1 - k / h)), b, K.red, 'overlap: max(0, 1 - k/h)', { line: { dash: 'dash' } }));
  G.lay.barmode = 'group'; G.lay.shapes.push(hl(a, 0, K.black, 'solid', 0.8));
  plot('wk-chart', tr, G.lay);
  const F = R[sc], ss = F.map(f => f[1]), st_ = F.map(f => f[2]);
  steps('wk-ws', [T.wks1(h, h - 1, f3((h - 1) / h), f3(ac[0]), R.n), T.wks2(T.scheme[sc], F.map(f => `${f[0]}: ${nf(f[1], 0)} / ${nf(f[2], 0)}`).join(T.sep)),
    T.wks3(nf(sum(ss), 0), nf(sum(st_), 0), pc(r2p(F)))]);
  out('wk-out', [T.wk1(T.series[q], h, pc(r2p(R.kfold)), pc(r2p(R.wf)), pc(r2p(R.gap)))]);
}

// ================================================================= 5. bias and variance of regression trees
const BVX = D.bv.x, BVZ = D.bv.z, GRID400 = range(0, 1, 1 / 399).map(v => Math.min(1, v));
const sin2 = x => Math.sin(2 * Math.PI * x);
function blkBv() {
  const s = +$('bv-s').value, dep = +$('bv-d').value; set('bv-s', f2(s)); set('bv-d', dep);
  const tab = D.bv.tab[s.toFixed(2)], y = BVX.map((x, i) => sin2(x) + s * BVZ[i]), idx = range(0, 79, 1), dd = range(1, 10, 1);
  const G = grid('bv-chart', [{ r: 0, c: 0, xt: 'x', yt: 'y' }, { r: 0, c: 1, xt: 'tree depth (complexity)', yt: 'mean squared error' }], { cols: 2, h: 380, legendRows: 3 });
  const [a, b] = G.ax, tr = [mk(BVX, y, a, K.blue, 'training data', { marker: { size: 5, color: K.blue, opacity: 0.6 } }), ln(GRID400, GRID400.map(sin2), a, K.black, 'true function', { line: { width: 1.6 } })];
  [[1, K.amber], [3, K.green], [dep, K.red]].forEach(([d, c]) => { const t = cart([BVX], y, idx, d, 1); tr.push(ln(GRID400, GRID400.map(x => t.predict([x])), a, c, `tree of depth ${d}`, { line: { width: 1.6, shape: 'hv' } })); });
  [[0, K.amber, 'squared bias'], [1, K.purple, 'variance'], [2, K.red, 'expected test error'], [3, K.green, 'training error']].forEach(([j, c, nm]) =>
    tr.push(ln(dd, tab.map(r => r[j]), b, c, nm, { mode: 'lines+markers', marker: { size: 5 }, line: { width: j === 2 ? 2 : 1.4, dash: j === 3 ? 'dash' : 'solid' } })));
  tr.push(hline(b, [1, 10], s * s, K.blue, 'noise variance', { line: { dash: 'dot', width: 1.2 } }));
  plot('bv-chart', tr, G.lay);
  const best = tab.reduce((bi, r, i) => (r[2] < tab[bi][2] ? i : bi), 0);
  out('bv-out', [T.bv1(f2(s), best + 1, f3(tab[best][2]), f3(s * s)), T.bv2(f3(tab[0][0]), f3(tab[0][1]), f3(tab[9][0]), f3(tab[9][1]), f3(tab[9][3]))]);
}

// ================================================================= EXTRA. ridge and lasso with one standardised feature
const SXX = 100;
const ridge1 = (s, l) => s / (SXX + l), lasso1 = (s, l) => Math.sign(s) * Math.max(Math.abs(s) - l / 2, 0) / SXX;
function blkWr() {
  const ll = +$('wr-l').value, lr = +$('wr-r').value, s = +$('wr-s').value; set('wr-l', ll); set('wr-r', lr); set('wr-s', s);
  const xs = range(0, 260, 2), G = grid('wr-chart', [{ r: 0, c: 0, xt: 'penalty lambda', yt: 'coefficient beta' }], { cols: 1, h: 330, legendRows: 2 });
  const ax = G.ax[0];
  plot('wr-chart', [ln(xs, xs.map(l => ridge1(s, l)), ax, K.blue, 'ridge: S_xy / (S_xx + lambda)'), ln(xs, xs.map(l => lasso1(s, l)), ax, K.red, 'lasso: soft thresholding'),
    hline(ax, [0, 260], s / SXX, K.navy, 'OLS: S_xy / S_xx', { line: { dash: 'dot', width: 1 } }),
    mk([lr], [ridge1(s, lr)], ax, K.blue, `ridge, lambda = ${lr}`, { marker: { size: 11, color: K.blue } }), mk([ll], [lasso1(s, ll)], ax, K.red, `lasso, lambda = ${ll}`, { marker: { size: 11, symbol: 'diamond', color: K.red } })], G.lay);
  steps('wr-ws1', [T.wrs1(s, SXX, f2(s / SXX)), T.wrs2(lr, s, SXX + lr, f3(ridge1(s, lr)), f3(SXX / (SXX + lr)))]);
  steps('wr-ws2', [T.wrs3(ll, nf(ll / 2, 1), Math.abs(s), nf(Math.max(Math.abs(s) - ll / 2, 0), 1), f3(lasso1(s, ll))), T.wrs4(2 * Math.abs(s))]);
  out('wr-out', [T.wr1(f2(s / SXX), lr, f3(ridge1(s, lr)), ll, f3(lasso1(s, ll)), 2 * Math.abs(s))]);
}

// ================================================================= 6. ridge and lasso paths on the load features
const SH = D.shrink, SHF = SH.feats;
const HIGH = { lag0: K.blue, lag6: K.crimson, same_dow: K.red, mean7: K.green, sat: K.orange, holiday: K.purple, easter: K.amber, xmas: K.teal };
const entryOrder = M => range(0, SHF.length - 1, 1).sort((i, j) => M.entry[j] - M.entry[i]);
function blkSh() {
  const e = $('sh-e').value, M = SH.m[e];
  const G = grid('sh-chart', [{ r: 0, c: 0, t: 'ridge: all coefficients shrink smoothly', xt: 'ridge penalty lambda', yt: 'coefficient (GW)', x: { type: 'log', autorange: 'reversed', exponentformat: 'power' } },
    { r: 0, c: 1, t: 'lasso: coefficients drop to exactly zero', xt: 'lasso penalty lambda', x: { type: 'log', autorange: 'reversed', exponentformat: 'power' } }], { cols: 2, h: 400, legendRows: 3 });
  const [a, b] = G.ax, tr = [];
  SHF.forEach((f, j) => { const c = HIGH[f], o = c ? { line: { color: c, width: 1.8, dash: f === 'lag6' ? 'dash' : 'solid' } } : { line: { color: K.blue, width: 0.8, dash: 'dot' } };
    tr.push(ln(M.lams, M.ridge[j], a, c || K.blue, c ? f : '', Object.assign({ hovertemplate: f + ': %{y:.3f}<extra></extra>' }, o)), ln(M.alphas, M.lasso[j], b, c || K.blue, '', Object.assign({ hovertemplate: f + ': %{y:.3f}<extra></extra>' }, o))); });
  const all = M.lasso.flat(), lo = Math.min(...all), hi = Math.max(...all);
  tr.push(vline(b, M.alpha_cv, lo, hi, K.red, 'lasso penalty chosen by walk-forward CV', { line: { dash: 'dash', width: 1 } }), leg('other features', { color: K.blue, dash: 'dot', width: 0.8 }));
  plot('sh-chart', tr, G.lay);
  const o = entryOrder(M), kept = M.coef_cv.filter(v => Math.abs(v) > 1e-8).length, c = f => M.coef_cv[SHF.indexOf(f)];
  out('sh-out', [T.sh1(T.dmy(e), M.n, o.slice(0, 3).map(j => SHF[j]).join(', '), kept, f2(c('lag0')), f2(c('mon'))), T.sh2(nf(M.alpha_cv, 5))]);
}

// ================================================================= EXTRA. the order in which the lasso keeps the features
function blkLt() {
  const e = $('lt-e').value, off = +$('lt-k').value, nr = +$('lt-n').value, M = SH.m[e]; set('lt-n', nr);
  const icv = M.alphas.reduce((bi, a, i) => (Math.abs(Math.log(a / M.alpha_cv)) < Math.abs(Math.log(M.alphas[bi] / M.alpha_cv)) ? i : bi), 0);
  const ia = Math.max(0, Math.min(M.alphas.length - 1, icv + off)), lam = off ? M.alphas[ia] : M.alpha_cv, co = SHF.map((_, j) => (off ? M.lasso[j][ia] : M.coef_cv[j]));
  set('lt-k', (off > 0 ? '+' : '') + off + ` (λ = ${nf(lam, 5)})`);
  const o = entryOrder(M), kept = co.filter(v => Math.abs(v) > 1e-8).length;
  const G = grid('lt-chart', [{ r: 0, c: 0, t: `Lasso coefficients at lambda = ${lam.toPrecision(3)} (${kept} of 30 non-zero)`, yt: 'coefficient (GW per s.d.)', x: { tickangle: -60, tickfont: { size: 9 } } }], { cols: 1, h: 360, legendRows: 1, mb: 70 });
  const ax = G.ax[0];
  plot('lt-chart', [bar(SHF, co, ax, co.map(v => (v >= 0 ? K.blue : K.red)), 'coefficient at the selected penalty'), hline(ax, [SHF[0], SHF[29]], 0, K.black, '', { line: { width: 0.6 } })], G.lay);
  table('lt-tb', [T.L.rank, T.L.feature, H_(T.L.entry), H_(T.L.coef)], o.slice(0, nr).map((j, r) => hiRow([r + 1, SHF[j], num(M.entry[j], 4), num(co[j], 3, Math.abs(co[j]) > 1e-8)], r < 3)), { cap: T.ltCap(T.dmy(e), M.n) });
  out('lt-out', [T.lt1(o.slice(0, 3).map(j => SHF[j]).join(', '), nf(lam, 5), kept)]);
}

// ================================================================= 7. a depth-2 tree for tomorrow's load
const TRD = D.tree, TRN = TRD.y.length, TRDATES = Array.from({ length: TRN }, (_, i) => addDays(TRD.d0, i));
function treeState(e, dep, ml) {
  const idx = range(0, TRN - 1, 1).filter(i => TRDATES[i] < e), t = cart(TRD.X, TRD.y, idx, dep, ml), lv = leaves(t);
  const yy = idx.map(i => TRD.y[i]), my = mean(yy), fit = idx.map(i => t.predict(TRD.X.map(c => c[i])));
  return { t, lv, idx, fit, r2: 1 - sum(yy.map((v, j) => (v - fit[j]) ** 2)) / sum(yy.map(v => (v - my) ** 2)) };
}
function blkTr() {
  const e = $('tr-e').value, dep = +$('tr-d').value, ml = +$('tr-m').value; set('tr-d', dep); set('tr-m', ml);
  const S = memo(`tr|${e}|${dep}|${ml}`, () => treeState(e, dep, ml)), cols = TRD.cols;
  const G = grid('tr-chart', [{ r: 0, c: 0, cs: 2, x: { visible: false, range: [-0.05, 1.05] }, y: { visible: false, range: [-dep - 0.6, 0.5] } }, { r: 0, c: 2, t: 'Last 60 training days', yt: 'GW', x: DATE }],
    { cols: 3, h: 420, legendRows: 2, gx: 0.06 });
  const [a, b] = G.ax, nx = [], ny = [], nt = [], ex = [], ey = [];
  (function place(nd, d, x0, x1, px, py) {
    const x = (x0 + x1) / 2; nx.push(x); ny.push(-d);
    nt.push(nd.left ? `${cols[nd.feat]} ≤ ${nd.thr.toFixed(2)}<br>n = ${nd.n}` : `<b>${nd.value.toFixed(2)} GW</b><br>n = ${nd.n}`);
    if (px !== null) { ex.push(px, x, null); ey.push(py, -d, null); }
    if (nd.left) { place(nd.left, d + 1, x0, x, x, -d); place(nd.right, d + 1, x, x1, x, -d); }
  })(S.t, 0, 0, 1, null, null);
  const last = S.idx.slice(-60);
  plot('tr-chart', [ln(ex, ey, a, K.blue, '', { hoverinfo: 'skip', line: { width: 1 } }),
    { x: nx, y: ny, text: nt, mode: 'markers+text', textposition: 'middle center', textfont: { size: dep > 3 ? 8 : 10, color: K.navy }, marker: { size: dep > 3 ? 36 : dep === 3 ? 50 : 70, color: '#FFFFFF', symbol: 'square', line: { color: K.blue, width: 1 } },
      xaxis: a.xa, yaxis: a.ya, showlegend: false, hoverinfo: 'skip' },
    ln(last.map(i => addDays(TRDATES[i], 1)), last.map(i => TRD.y[i]), b, K.black, 'actual load y_t+1', { mode: 'lines+markers', marker: { size: 3 } }),
    ln(last.map(i => addDays(TRDATES[i], 1)), S.fit.slice(-60), b, K.red, 'tree forecast (a leaf mean)', { line: { shape: 'hv', width: 1.6 } })], G.lay);
  out('tr-out', [S.t.left ? T.tr1(cols[S.t.feat], f2(S.t.thr), S.lv.map(l => f2(l.value)).join(T.sep), S.lv.map(l => l.n).join(T.sep), S.idx.length) : T.tr0(S.idx.length), T.tr2(f2(S.r2), S.lv.length)]);
}

// ================================================================= EXTRA. gradient boosting, step by step
const WBY = [5, 6, 8];
function boost(nu, M) {
  let F = WBY.map(() => mean(WBY)); const path = [F.slice()], mse = [mean(WBY.map((y, i) => (y - F[i]) ** 2))], stumps = [];
  for (let m = 1; m <= M; m++) {
    const r = WBY.map((y, i) => y - F[i]);
    const cand = [1, 2].map(s => { const L = mean(r.slice(0, s)), R = mean(r.slice(s)); return { s, L, R, sse: sum(r.map((v, i) => (v - (i < s ? L : R)) ** 2)) }; });
    const g = cand[0].sse <= cand[1].sse ? cand[0] : cand[1];
    F = F.map((f, i) => f + nu * (i < g.s ? g.L : g.R)); path.push(F.slice()); mse.push(mean(WBY.map((y, i) => (y - F[i]) ** 2))); stumps.push(Object.assign({ r }, g));
  }
  return { path, mse, stumps };
}
function blkWb() {
  const nu = +$('wb-n').value, m = +$('wb-m').value; set('wb-n', f2(nu)); set('wb-m', m);
  const B = boost(nu, 60), mm = range(0, 60, 1);
  const G = grid('wb-chart', [{ r: 0, c: 0, t: 'F_m at the three training points', xt: 'number of trees m' }, { r: 0, c: 1, t: 'Training MSE', xt: 'number of trees m', y: LOG }], { cols: 2, h: 340, legendRows: 2 });
  const [a, b] = G.ax, cl = [K.blue, K.green, K.red], tr = [];
  WBY.forEach((y, i) => { tr.push(ln(mm, B.path.map(p => p[i]), a, cl[i], `point ${i + 1} (target ${y})`), hline(a, [0, 60], y, cl[i], '', { line: { dash: 'dot', width: 1 } })); });
  tr.push(ln(mm, B.mse.map(v => Math.max(v, 1e-12)), b, K.purple, `learning rate nu = ${nu.toFixed(2)}`), mk([m], [Math.max(B.mse[m], 1e-12)], b, K.purple, '', { marker: { size: 9, color: K.purple } }));
  plot('wb-chart', tr, G.lay);
  const s1 = B.stumps[0], F0 = mean(WBY), pm = B.path[m];
  steps('wb-ws', [T.wbs1(f2(F0), s1.r.map(v => sgn(v, 2)).join(T.sep)), T.wbs2(f2(s1.L), f2(s1.R), f2(nu), f2(F0), f2(s1.R), f3(B.path[1][2]), f3(B.path[1][0])),
    T.wbs3(m, pm.map(v => f3(v)).join(T.sep), f4(B.mse[m]))]);
  out('wb-out', [T.wb1(f2(nu), [5, 20, 60].map(k => f3(B.path[k][2])).join(T.sep))]);
}

// ================================================================= 8. one tree, a forest and boosting
function blkEn() {
  const s = $('en-s').value, E = D.ens.m[s], dd = range(1, 14, 1), it = range(1, 600, 1);
  const G = grid('en-chart', [{ r: 0, c: 0, xt: 'maximum depth', yt: 'validation MSE (GW^2)' }, { r: 0, c: 1, xt: 'number of trees', x: { type: 'log' } }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax, ym = Math.max(E.naive, E.tree[0]) * 1.1;
  G.lay.yaxis.range = [0, ym]; G.lay.yaxis2.range = [0, ym];
  plot('en-chart', [ln(dd, E.tree, a, K.amber, 'single tree', { mode: 'lines+markers', marker: { size: 5 } }), ln(dd, E.rf, a, K.green, 'random forest (200 trees)', { mode: 'lines+markers', marker: { size: 5 } }),
    hline(a, [1, 14], E.naive, K.purple, 'seasonal naive (same weekday last week)', { line: { dash: 'dot' } }),
    ln(it, E.gb03, b, K.red, 'boosting, learning rate 0.3'), ln(it, E.gb005, b, K.blue, 'boosting, learning rate 0.05'), hline(b, [1, 600], E.naive, K.purple, '', { line: { dash: 'dot' } })], G.lay);
  const am = v => v.reduce((bi, x, i) => (x < v[bi] ? i : bi), 0), jt = am(E.tree), jr = am(E.rf), j3 = am(E.gb03), j5 = am(E.gb005);
  out('en-out', [T.en1(T.dmy(s), E.n_tr, E.n_va, f3(E.tree[jt]), jt + 1, f3(E.rf[jr]), jr + 1, f3(E.naive)), T.en2(f3(E.gb03[j3]), j3 + 1, f3(E.gb005[j5]), j5 + 1)]);
}

// ================================================================= 9. trees cannot extrapolate
const EXD = D.extra, EXM = months(EXD.m0, EXD.p.length);
function blkEx() {
  const s = $('ex-s').value, M = EXD.m[s], i0 = EXM.indexOf(M.t0), te = EXM.slice(i0), pt = EXD.p.slice(i0), j17 = EXM.indexOf('2017-01-01');
  const G = grid('ex-chart', [{ r: 0, c: 0, yt: 'index', x: DATE }], { cols: 1, h: 360, legendRows: 2 });
  const ax = G.ax[0], all = EXD.p.slice(j17);
  plot('ex-chart', [ln(EXM.slice(j17), all, ax, K.black, 'HICP index, Romania (2015 = 100)', { line: { width: 1.6 } }),
    ln(te, M.fl, ax, K.red, 'random forest on lagged levels', { line: { dash: 'dash', width: 1.6 } }), ln(te, M.fc, ax, K.green, 'random forest on monthly changes', { line: { dash: 'dashdot', width: 1.6 } }),
    hline(ax, [EXM[j17], EXM[EXM.length - 1]], M.max, K.amber, 'largest level seen in training', { line: { dash: 'dot', width: 1.2 } }),
    vline(ax, EXM[i0 - 1], Math.min(...all) - 3, Math.max(...all) + 3, K.blue, '', { line: { dash: 'dot', width: 1 } })], G.lay);
  const rm = f => Math.sqrt(mean(pt.map((v, i) => (v - f[i]) ** 2)));
  out('ex-out', [T.ex1(T.my(EXM[i0 - 1]), f1(M.max), f1(EXD.p[EXD.p.length - 1]), T.my(EXM[EXM.length - 1]), f1(M.fl[M.fl.length - 1]), T.dmy(EXD.downloaded)),
    T.ex2(f1(rm(M.fl)), f2(rm(M.fc)), te.length)]);
}

// ================================================================= 10. permutation importance
function blkIm() {
  const h = +$('im-h').value, h2 = 15 - h; set('im-h', `${h} / ${h2}`);
  const top = hh => { const v = D.imp.m[hh], o = range(0, 29, 1).sort((i, j) => v[j] - v[i]).slice(0, 8).reverse(); return { f: o.map(j => D.imp.feats[j]), v: o.map(j => 1000 * v[j]) }; };
  const A = top(h), B = top(h2);
  const G = grid('im-chart', [{ r: 0, c: 0, t: `horizon ${h} day${h > 1 ? 's' : ''}`, xt: 'increase in MAE (MW)', yam: true }, { r: 0, c: 1, t: `horizon ${h2} days`, xt: 'increase in MAE (MW)', yam: true }],
    { cols: 2, h: 360, legendRows: 1, ml: 80, gx: 0.16, legend: false });
  const [a, b] = G.ax;
  plot('im-chart', [bar(A.v, A.f, a, K.blue, '', { orientation: 'h', hovertemplate: '%{y}: %{x:.0f} MW<extra></extra>' }), bar(B.v, B.f, b, K.red, '', { orientation: 'h', hovertemplate: '%{y}: %{x:.0f} MW<extra></extra>' })], G.lay);
  out('im-out', [T.im1(h, A.f[7], nf(A.v[7], 0), A.f[6], nf(A.v[6], 0)), T.im1(h2, B.f[7], nf(B.v[7], 0), B.f[6], nf(B.v[6], 0))]);
}

// ================================================================= 11. a small network and its activation functions
const relu = z => Math.max(0, z), logi = z => 1 / (1 + Math.exp(-z));
function blkMl() {
  const q = +$('ml-q').value; set('ml-q', q);
  const G = grid('ml-chart', [{ r: 0, c: 0, x: { visible: false, range: [-0.3, 1.25] }, y: { visible: false, range: [0, 1.05] } }, { r: 0, c: 1, xt: 'z', yt: 'g(z)', y: { range: [-1.2, 2.5] } }], { cols: 2, h: 380, legendRows: 3 });
  const [a, b] = G.ax, lay = [[0.85, 0.5, 0.15], range(0, q - 1, 1).map(j => (q === 1 ? 0.5 : 0.85 - 0.7 * j / (q - 1))), [0.5]], xs = [0.1, 0.5, 0.9], ex = [], ey = [];
  for (let l = 0; l < 2; l++) lay[l].forEach(y0 => lay[l + 1].forEach(y1 => { ex.push(xs[l], xs[l + 1], null); ey.push(y0, y1, null); }));
  const zs = range(-4, 4, 0.02), kn = range(-4, 4, 8 / q), pl = z => { if (z <= -4) return Math.tanh(-4); if (z >= 4) return Math.tanh(4); const i = Math.min(q - 1, Math.floor((z + 4) / (8 / q))), z0 = kn[i], z1 = kn[i + 1];
    return Math.tanh(z0) + (Math.tanh(z1) - Math.tanh(z0)) * (z - z0) / (z1 - z0); };
  const err = Math.max(...zs.map(z => Math.abs(pl(z) - Math.tanh(z))));
  plot('ml-chart', [ln(ex, ey, a, K.teal, '', { hoverinfo: 'skip', line: { width: 0.8 } }),
    mk(lay[0].map(() => 0.1), lay[0], a, K.blue, 'inputs', { marker: { size: 18, color: K.blue, line: { color: K.black, width: 1 } }, text: ['y_t', 'y_t-6', 'holiday'], textposition: 'middle left', mode: 'markers+text', hoverinfo: 'skip' }),
    mk(lay[1].map(() => 0.5), lay[1], a, K.green, `${q} hidden neuron${q > 1 ? 's' : ''}: h_j = g(b_j + w_j'x)`, { marker: { size: 18, color: K.green, line: { color: K.black, width: 1 } }, hoverinfo: 'skip' }),
    mk([0.9], [0.5], a, K.red, "output: y_hat = c + v'h", { marker: { size: 18, color: K.red, line: { color: K.black, width: 1 } }, hoverinfo: 'skip' }),
    ln(zs, zs.map(logi), b, K.blue, 'logistic 1/(1+e^-z)', { line: { width: 1.8 } }), ln(zs, zs.map(Math.tanh), b, K.red, 'tanh z', { line: { width: 1.8 } }), ln(zs, zs.map(relu), b, K.green, 'ReLU max(0, z)', { line: { width: 1.8 } }),
    ln(zs, zs.map(pl), b, K.orange, `tanh built from ${q} ReLU neuron${q > 1 ? 's' : ''}`, { line: { width: 1.4, dash: 'dash' } })], G.lay);
  G.lay.shapes.push(hl(b, 0, K.black, 'solid', 0.6));
  out('ml-out', [T.ml1(q, 3 * q, q, q, 5 * q + 1), T.ml2(q, f3(err))]);
}

// ================================================================= 12. the LSTM of the chapter: forecasts and training loss at every origin
function blkRn() {
  const o = $('rn-o').value, k = ORIG.indexOf(o), io = dayIdx(LD.d0, o), hh = range(1, 14, 1), loss = D.lstm.loss[k];
  const fd = hh.map(h => LDATES[io + h]), past = range(io - 27, io, 1), f = m => D.err.fc[m].slice(14 * k, 14 * k + 14), act = ACT.slice(14 * k, 14 * k + 14);
  const G = grid('rn-chart', [{ r: 0, c: 0, cs: 2, t: `14-day forecasts from ${o}`, yt: 'GW', x: DATE }, { r: 0, c: 2, t: 'LSTM training loss (standardised MSE)', xt: 'epoch', y: LOG }], { cols: 3, h: 380, legendRows: 2 });
  const [a, b] = G.ax;
  plot('rn-chart', [ln([...past.map(i => LDATES[i]), ...fd], [...past.map(i => LOADV[i]), ...act], a, K.black, 'actual daily load', { mode: 'lines+markers', marker: { size: 3 } }),
    ln(fd, f('LSTM'), a, K.purple, 'LSTM (56 days in, 14 out)', { mode: 'lines+markers', marker: { size: 4 }, line: { dash: 'dot', width: 1.8 } }),
    ln(fd, f('DHR'), a, K.blue, 'DHR', { mode: 'lines+markers', marker: { size: 4 } }), vline(a, o, 3.5, 8, K.blue, '', { line: { dash: 'dot', width: 1 } }),
    ln(range(1, loss.length, 1), loss, b, K.purple, '', { line: { width: 1.6 } })], G.lay);
  const mae = v => mean(v.map((x, i) => Math.abs(act[i] - x)));
  out('rn-out', [T.rn1(T.dmy(o), f3(loss[0]), f3(loss[loss.length - 1]), loss.length), T.rn2(mw(mae(f('LSTM'))), mw(mae(f('DHR'))))]);
}

// ================================================================= 13. the load: 26 origins, 14 horizons
function loadStats(ok, hs) {          // ok: origins used (indices), hs: horizons used
  const R = {};
  MODELS.forEach(m => { const e = ERR(m), sel = [], W = [];
    ok.forEach(k => { let s = 0; hs.forEach(h => { const v = e[14 * k + h - 1]; sel.push(v); s += Math.abs(v); }); W.push(s / hs.length); });
    const mae = mean(sel.map(Math.abs)); R[m] = { mae, rmse: Math.sqrt(mean(sel.map(v => v * v))), mase: mae / SCALE, W }; });
  return R;
}
const HS = { '1-14': range(1, 14, 1), '1-7': range(1, 7, 1), '8-14': range(8, 14, 1) };
function blkLd() {
  const k0 = +$('ld-k').value, ref = $('ld-r').value, hs = $('ld-h').value; set('ld-k', k0);
  const ok = range(k0, NOR - 1, 1), R = loadStats(ok, HS[hs]), dm = MODELS.map(m => (m === ref ? null : dmRaw(R[m].W, R[ref].W, 1)));
  const G = grid('ld-chart', [{ r: 0, c: 0, t: 'MASE (scale: 309 MW)', x: { tickangle: -45, tickfont: { size: 9 } } }, { r: 0, c: 1, t: `DM p-value against ${ref} (log scale)`, y: { type: 'log', range: [-5, 0.05] }, x: { tickangle: -45, tickfont: { size: 9 } } }],
    { cols: 2, h: 400, legendRows: 1, mb: 90 });
  const [a, b] = G.ax, cl = MODELS.map(m => (STAT.includes(m) ? K.blue : K.red)), oth = MODELS.filter(m => m !== ref);
  plot('ld-chart', [bar(MODELS, MODELS.map(m => R[m].mase), a, cl, '', { text: MODELS.map(m => R[m].mase.toFixed(2)), textposition: 'outside', cliponaxis: false }),
    hline(a, [MODELS[0], MODELS[10]], 1, K.navy, 'MASE = 1; p = 0.05', { line: { dash: 'dot', width: 1 } }),
    bar(oth, oth.map(m => dm[MODELS.indexOf(m)].p), b, oth.map(m => (STAT.includes(m) ? K.blue : K.red)), ''), hline(b, [oth[0], oth[oth.length - 1]], 0.05, K.navy, '', { line: { dash: 'dot', width: 1 } }),
    leg('Chapter 4 (statistical)', null, { color: K.blue, symbol: 'square', size: 10 }), leg('machine learning', null, { color: K.red, symbol: 'square', size: 10 })], G.lay);
  table('ld-tb', [T.L.model, H_('MAE (GW)'), H_('RMSE (GW)'), H_('MASE'), H_(T.L.dmp(MN(ref)))],
    MODELS.map((m, i) => hiRow([MN(m), num(R[m].mae, 3), num(R[m].rmse, 3), num(R[m].mase, 2), m === ref ? txt('–') : txt(pv(dm[i].p), dm[i].p < 0.05)], m === 'Ridge' || m === ref)),
    { cap: T.ldCap(ok.length, T.dmy(ORIG[k0]), hs) });
  const ri = MODELS.indexOf('Ridge');
  out('ld-out', [T.ld1(ok.length, f2(R.DHR.mase), f2(R.Ridge.mase), ref === 'Ridge' ? '–' : pv(dm[ri].p), MN(ref))]);
}
function blkLr() {
  const n = +$('lr-n').value; set('lr-n', n);
  const ok = range(0, n - 1, 1), R = loadStats(ok, HS['1-14']), hh = range(1, 14, 1);
  const nar = $('lr-chart').clientWidth < 560;
  const G = grid('lr-chart', [{ r: 0, c: 0, xt: nar ? 'MASE' : 'MASE (below 1: better than the in-sample seasonal naive)', y: { autorange: 'reversed' }, yam: true }, { r: 0, c: 1, xt: 'horizon (days)', yt: 'MAE (GW)' }],
    { cols: 2, h: 420, legendRows: nar ? 4 : 2, ml: 110, gx: 0.1 });
  const [a, b] = G.ax, sh = ['Seasonal naive', 'DHR', 'Combination', 'GB direct', 'GB recursive', 'LSTM'];
  const byh = m => hh.map(h => mean(ok.map(k => Math.abs(ERR(m)[14 * k + h - 1]))));
  plot('lr-chart', [bar(MODELS.map(m => R[m].mase), MODELS, a, MODELS.map(m => (STAT.includes(m) ? K.blue : K.red)), '', { orientation: 'h', text: MODELS.map(m => R[m].mase.toFixed(2)), textposition: 'outside', cliponaxis: false, hovertemplate: '%{y}: %{x:.3f}<extra></extra>' }),
    vline(a, 1, MODELS[0], MODELS[10], K.navy, '', { line: { dash: 'dot', width: 1 } }),
    ...sh.map(m => ln(hh, byh(m), b, MCOL[m], m, { mode: 'lines+markers', marker: { size: 4 }, line: { dash: m === 'GB recursive' || m === 'LSTM' ? 'dash' : 'solid' } }))], G.lay);
  const o = [...MODELS].sort((p, q) => R[p].mase - R[q].mase);
  out('lr-out', [T.lr1(n, T.dmy(ORIG[n - 1]), MN(o[0]), f2(R[o[0]].mase), MN(o[1]), f2(R[o[1]].mase), f2(R.LSTM.mase)), T.lr2(mw(byh('DHR')[0]), mw(byh('DHR')[13]), mw(byh('GB direct')[0]), mw(byh('GB direct')[13]))]);
}

// ================================================================= EXTRA. MASE with the scale of the seasonal naive method
function scaleM(m) { const n = dayIdx(LD.d0, ORIG[0]) + 1, d = []; for (let t = m; t < n; t++) d.push(Math.abs(LOADV[t] - LOADV[t - m])); return { d, s: mean(d), n }; }
function blkWm() {
  const m = +$('wm-m').value, md = $('wm-x').value; set('wm-m', m);
  const S = memo('wm|' + m, () => scaleM(m)), all = loadStats(range(0, NOR - 1, 1), HS['1-14']), dd = LDATES.slice(m, S.n);
  const G = grid('wm-chart', [{ r: 0, c: 0, cs: 2, t: `|y_t - y_t-${m}| in the first training window (GW)`, x: DATE }, { r: 0, c: 2, t: `MASE with m = ${m}`, x: { tickangle: -45, tickfont: { size: 9 } } }],
    { cols: 3, h: 380, legendRows: 2, mb: 80 });
  const [a, b] = G.ax;
  plot('wm-chart', [ln(dd, S.d, a, K.blue, `|y_t - y_t-${m}|`, { line: { width: 0.6 } }), hline(a, [dd[0], dd[dd.length - 1]], S.s, K.red, `scale = mean = ${(1000 * S.s).toFixed(0)} MW`, { line: { width: 1.6 } }),
    bar(MODELS, MODELS.map(x => all[x].mae / S.s), b, MODELS.map(x => (STAT.includes(x) ? K.blue : K.red)), ''), hline(b, [MODELS[0], MODELS[10]], 1, K.navy, '', { line: { dash: 'dot', width: 1 } })], G.lay);
  steps('wm-ws', [T.wms1(T.dmy(LD.d0), T.dmy(ORIG[0]), S.n, m, S.n - m, mw(S.s)), T.wms2(MN(md), mw(all[md].mae), mw(S.s), f3(all[md].mae / S.s)), T.wms3(m, mw(SCALE), f3(all[md].mae / SCALE))]);
  out('wm-out', [T.wm1(m, mw(S.s), MN(md), f2(all[md].mae / S.s))]);
}

// ================================================================= EXTRA. a Diebold-Mariano statistic, step by step
function blkWd() {
  const m = $('wd-a').value, Lg = +$('wd-L').value, ref = $('wd-r').value; set('wd-L', Lg);
  const R = loadStats(range(0, NOR - 1, 1), HS['1-14']), D0 = dmRaw(R[m].W, R[ref].W, Lg + 1), Ls = range(0, 5, 1), ps = Ls.map(l => dmRaw(R[m].W, R[ref].W, l + 1).p);
  const G = grid('wd-chart', [{ r: 0, c: 0, cs: 2, t: `d_t = mean |e| of ${m} minus mean |e| of ${ref}, by origin (MW)`, x: DATE }, { r: 0, c: 2, t: 'DM p-value by number of lags', xt: 'h - 1 (lags)', y: { range: [0, 1.02] } }],
    { cols: 3, h: 360, legendRows: 2 });
  const [a, b] = G.ax, dm = D0.d.map(v => 1000 * v);
  plot('wd-chart', [bar(ORIG, dm, a, dm.map(v => (v > 0 ? K.red : K.blue)), 'd_t > 0: the reference is better at that origin'),
    hline(a, [ORIG[0], ORIG[NOR - 1]], 1000 * D0.dbar, K.navy, `mean d = ${(1000 * D0.dbar).toFixed(1)} MW`, { line: { dash: 'dash', width: 1.4 } }),
    ln(Ls, ps, b, K.purple, '', { mode: 'lines+markers', marker: { size: 6 } }), mk([Lg], [D0.p], b, K.red, '', { marker: { size: 12, color: K.red, symbol: 'circle-open', line: { width: 2 } } }),
    hline(b, [0, 5], 0.05, K.navy, '', { line: { dash: 'dot', width: 1 } })], G.lay);
  steps('wd-ws', [T.wds1(MN(m), MN(ref), D0.n, nf(1000 * D0.dbar, 1)), T.wds2(Lg, D0.g.map((g, k) => `γ̂<sub>${k}</sub> = ${nf(1e6 * g, 1)}`).join(T.sep), nf(1e6 * D0.v, 2)),
    T.wds3(nf(1000 * D0.dbar, 2), nf(1000 * Math.sqrt(D0.v), 2), f3(D0.dm), f3(Math.sqrt((D0.n + 1 - 2 * (Lg + 1) + (Lg + 1) * Lg / D0.n) / D0.n)), f3(D0.hln)), T.wds4(D0.n - 1, f3(D0.p))]);
  out('wd-out', [T.wd1(MN(m), MN(ref), Lg, f3(D0.hln), f3(D0.p))]);
}

// ================================================================= 14. forecasts around Orthodox Easter 2026
function blkLf() {
  const o = $('lf-o').value, k = ORIG.indexOf(o), io = dayIdx(LD.d0, o), hh = range(1, 14, 1), fd = hh.map(h => LDATES[io + h]), act = ACT.slice(14 * k, 14 * k + 14);
  const G = grid('lf-chart', [{ r: 0, c: 0, yt: 'GW', x: DATE }], { cols: 1, h: 340, legendRows: 2 });
  const ax = G.ax[0], past = range(io - 28, io, 1), ms = [['Seasonal naive', 'dot'], ['DHR', 'solid'], ['GB direct', 'dash'], ['GB recursive', 'dashdot'], ['LSTM', 'dot']];
  const hol = fd.map((d, j) => (HOL(io + 1 + j) ? d : null)).filter(Boolean);
  plot('lf-chart', [ln([...past.map(i => LDATES[i]), ...fd], [...past.map(i => LOADV[i]), ...act], ax, K.black, 'actual daily load', { line: { width: 1.5 } }),
    ...ms.map(([m, ds]) => ln(fd, D.err.fc[m].slice(14 * k, 14 * k + 14), ax, MCOL[m], m, { mode: 'lines+markers', marker: { size: 4 }, line: { dash: ds, width: 1.6 } })),
    mk(hol, hol.map(d => LOADV[dayIdx(LD.d0, d)]), ax, K.red, 'public holiday in the window', { marker: { size: 11, symbol: 'star', color: K.red } }),
    vline(ax, o, 3.5, 8, K.blue, '', { line: { dash: 'dot', width: 1 } })], G.lay);
  const mae = m => mean(D.err.fc[m].slice(14 * k, 14 * k + 14).map((f, i) => Math.abs(act[i] - f)));
  out('lf-out', [T.lf1(T.dmy(o), ms.map(([m]) => `${MN(m)} ${mw(mae(m))} MW`).join(T.sep)), T.lf2(hol.length, MN([...MODELS].sort((p, q) => mae(p) - mae(q))[0]), mw(Math.min(...MODELS.map(mae))))]);
}

// ================================================================= EXTRA. the pinball loss
const pinball = (y, q, t) => (y >= q ? t * (y - q) : (1 - t) * (q - y));
function blkWp() {
  const t = +$('wp-t').value, q = +$('wp-q').value; set('wp-t', f2(t)); set('wp-q', f1(q));
  const ys = range(5.5, 8.5, 0.02), qs = range(4, 8.5, 0.02), A = ACT, ml = qs.map(v => mean(A.map(y => pinball(y, v, t)))), jm = ml.reduce((bi, v, i) => (v < ml[bi] ? i : bi), 0);
  const sorted = [...A].sort((x, y) => x - y), eq = sorted[Math.ceil(t * A.length) - 1];
  const G = grid('wp-chart', [{ r: 0, c: 0, t: `Pinball loss for q = ${q.toFixed(1)} GW, tau = ${t.toFixed(2)}`, xt: 'outcome y (GW)' }, { r: 0, c: 1, t: 'Mean pinball loss over the 364 outcomes of Section 6', xt: 'forecast q (GW)' }],
    { cols: 2, h: 350, legendRows: 2 });
  const [a, b] = G.ax;
  plot('wp-chart', [ln(ys, ys.map(y => pinball(y, q, t)), a, K.blue, 'L_tau(y, q)', { line: { width: 1.8 } }), mk([7.4, 6.5], [pinball(7.4, q, t), pinball(6.5, q, t)], a, K.red, 'y = 7.4 and y = 6.5 GW', { marker: { size: 10, color: K.red } }),
    ln(qs, ml, b, K.purple, 'mean loss', { line: { width: 1.8 } }), mk([qs[jm]], [ml[jm]], b, K.red, `minimum at q = ${qs[jm].toFixed(2)} GW`, { marker: { size: 11, symbol: 'diamond', color: K.red } })], G.lay);
  steps('wp-ws', [T.wps1(f2(t), f1(q), f3(pinball(7.4, q, t)), f3(pinball(6.5, q, t))), T.wps2(f2(t), f2(qs[jm]), f2(eq)), T.wps3()]);
  out('wp-out', [T.wp1(f2(t), f1(q), f3(pinball(7.4, q, t)), f3(pinball(6.5, q, t)))]);
}

// ================================================================= EXTRA. split conformal prediction
const IV = D.int, LEV = IV.levels;
const lvIdx = a => LEV.findIndex(v => Math.abs(v - a) < 1e-9);
function confCover(a) { const j = lvIdx(a); let c = 0, w = 0, lo = 0, hi = 0; ACT.forEach((y, i) => { const q = IV.q[i][j], l = IV.cpoint[i] - q, u = IV.cpoint[i] + q; w += 2 * q; if (y < l) lo++; else if (y > u) hi++; else c++; });
  const n = ACT.length; return { cover: c / n, width: w / n, below: lo / n, above: hi / n }; }
function blkWc() {
  const a = +$('wc-a').value, h = +$('wc-h').value; set('wc-a', pc(a, 0));
  const res = IV.first[h - 1], n = IV.n_cal, k = Math.ceil((n + 1) * a), qh = res[Math.min(k, n) - 1], C = LEV.map(confCover);
  const G = grid('wc-chart', [{ r: 0, c: 0, t: `Sorted |errors| of the calibration set: origin ${ORIG[0]}, h = ${h}`, xt: 'rank', yt: 'MW' }, { r: 0, c: 1, t: 'Coverage of the 364 forecasts', xt: 'nominal level', yt: 'coverage' }],
    { cols: 2, h: 360, legendRows: 2 });
  const [p, q] = G.ax;
  plot('wc-chart', [ln(range(1, n, 1), res.map(v => 1000 * v), p, K.blue, '|e|, sorted', { mode: 'lines+markers', marker: { size: 3 } }), mk([k], [1000 * qh], p, K.red, `q_hat: rank ${k} of ${n}`, { marker: { size: 11, color: K.red } }),
    ln(LEV, C.map(c => c.cover), q, K.purple, 'split conformal', { mode: 'lines+markers', marker: { size: 5 } }), ln([0.5, 0.95], [0.5, 0.95], q, K.navy, 'coverage = level', { line: { dash: 'dot', width: 1 } }),
    mk([a], [C[lvIdx(a)].cover], q, K.red, '', { marker: { size: 12, symbol: 'circle-open', color: K.red, line: { width: 2 } } })], G.lay);
  const i0 = h - 1, y0 = ACT[i0], c0 = IV.cpoint[i0];
  steps('wc-ws', [T.wcs1(n, pc(a, 0), f2(1 - a), n + 1, k), T.wcs2(k, mw(qh)), T.wcs3(T.dmy(ORIG[0]), h, f2(c0), f2(c0 - qh), f2(c0 + qh), f2(y0), y0 >= c0 - qh && y0 <= c0 + qh)]);
  out('wc-out', [T.wc1(pc(a, 0), pc(C[lvIdx(a)].cover), mw(C[lvIdx(a)].width))]);
}

// ================================================================= 15. 90% intervals for the load
function blkIv() {
  const a = +$('iv-l').value; set('iv-l', pc(a, 0));
  const j = lvIdx(a), n = ACT.length, dates = ORIG.flatMap(o => range(1, 14, 1).map(h => addDays(o, h))), ord = range(0, n - 1, 1).sort((p, q) => (dates[p] < dates[q] ? -1 : dates[p] > dates[q] ? 1 : p - q));
  const G = grid('iv-chart', [{ r: 0, c: 0, cs: 2, yt: 'GW', x: DATE }, { r: 0, c: 2, xt: 'horizon (days)', yt: 'coverage (%)', y: { range: [40, 105] } }], { cols: 3, h: 380, legendRows: 2 });
  const [p, q] = G.ax, X = ord.map(i => dates[i]), lo = ord.map(i => IV.cpoint[i] - IV.q[i][j]), hi = ord.map(i => IV.cpoint[i] + IV.q[i][j]);
  const cov = (inside) => range(1, 14, 1).map(h => 100 * mean(ORIG.map((_, k) => inside(14 * k + h - 1))));
  const qi = i => ACT[i] >= IV.qlo[i] && ACT[i] <= IV.qhi[i], ci = i => ACT[i] >= IV.cpoint[i] - IV.q[i][j] && ACT[i] <= IV.cpoint[i] + IV.q[i][j];
  plot('iv-chart', [band(X, ord.map(i => IV.qlo[i]), ord.map(i => IV.qhi[i]), p, K.teal, 0.3, 'quantile boosting, 90%'),
    ln(X, lo, p, K.purple, `split conformal, ${Math.round(100 * a)}%`, { line: { dash: 'dash', width: 0.9 } }), ln(X, hi, p, K.purple, '', { line: { dash: 'dash', width: 0.9 } }),
    mk(X, ord.map(i => ACT[i]), p, K.black, 'actual', { marker: { size: 3, color: K.black } }),
    ln(range(1, 14, 1), cov(qi), q, K.teal, 'quantile boosting', { mode: 'lines+markers', marker: { size: 5 } }), ln(range(1, 14, 1), cov(ci), q, K.purple, 'split conformal', { mode: 'lines+markers', marker: { size: 5, symbol: 'square' } }),
    hline(q, [1, 14], 100 * a, K.red, `nominal ${Math.round(100 * a)}%`, { line: { dash: 'dot' } })], G.lay);
  const C = confCover(a), Q = { cover: mean(range(0, n - 1, 1).map(i => +qi(i))), width: mean(range(0, n - 1, 1).map(i => IV.qhi[i] - IV.qlo[i])),
    below: mean(range(0, n - 1, 1).map(i => +(ACT[i] < IV.qlo[i]))), above: mean(range(0, n - 1, 1).map(i => +(ACT[i] > IV.qhi[i]))) };
  out('iv-out', [T.iv1(pc(Q.cover), mw(Q.width), pc(Q.below), pc(Q.above)), T.iv2(pc(a, 0), pc(C.cover), mw(C.width), pc(C.below), pc(C.above))]);
}

// ================================================================= 16. Romanian inflation: forecasts and relative RMSE
const INF = D.inf, INM = INF.models, PIM = months(INF.pi0, INF.pi.length);
function infErr(h, from, to) {             // errors by model at horizon h for origins in [from, to]
  const M = INF.m[h], od = months(M.o0, M.act.length), ix = od.map((d, i) => i).filter(i => od[i] >= from && od[i] <= to), E = {};
  INM.forEach(m => { E[m] = ix.map(i => M.act[i] - M.fc[m][i]); }); return { E, od: ix.map(i => od[i]), ix };
}
const rmse = e => Math.sqrt(mean(e.map(v => v * v)));
function blkIn() {
  const y0 = +$('in-y').value, h = $('in-h').value; set('in-y', y0);
  const from = `${y0}-01-01`, to = '2099-12-01', hs = ['1', '3', '6', '12'], M = INF.m[h], od = months(M.o0, M.act.length), sel = od.map((d, i) => i).filter(i => od[i] >= from);
  const G = grid('in-chart', [{ r: 0, c: 0, cs: 2, yt: '%', x: DATE }, { r: 0, c: 2, xt: 'horizon (months)', yt: 'relative RMSE', x: { tickvals: [1, 3, 6, 12] } }], { cols: 3, h: 380, legendRows: 3 });
  const [a, b] = G.ax, j15 = PIM.indexOf('2015-01-01'), tr = [ln(PIM.slice(j15), INF.pi.slice(j15), a, K.black, 'inflation, Romania (HICP, 12 months)', { line: { width: 1.6 } })];
  [['RW', 'dot'], ['AR', 'solid'], ['GB local', 'dash'], ['GB global', 'dashdot']].forEach(([m, ds]) => tr.push(ln(sel.map(i => isoM(od[i], +h)), sel.map(i => M.fc[m][i]), a, MCOL[m], `${m}, ${h} month${h > 1 ? 's' : ''} ahead`, { line: { dash: ds, width: 1.4 } })));
  const rel = hs.map(hh => { const { E } = infErr(hh, from, to); const r = rmse(E.RW); return Object.fromEntries(INM.map(m => [m, rmse(E[m]) / r])); });
  [['AR', 'circle'], ['Lasso', 'square'], ['RF', 'triangle-up'], ['GB local', 'diamond'], ['GB global', 'triangle-down']].forEach(([m, s]) =>
    tr.push(ln([1, 3, 6, 12], rel.map(r => r[m]), b, MCOL[m], m, { mode: 'lines+markers', marker: { size: 6, symbol: s } })));
  tr.push(hline(b, [1, 12], 1, MCOL.RW, 'random walk = 1', { line: { dash: 'dot' } }));
  plot('in-chart', tr, G.lay);
  const F = NB.inf.fig;
  out('in-out', [T.in1(f1(F.max), T.my(F.max_d), f1(F.min), T.my(F.min_d), f1(F.last), T.my(F.last_d), T.dmy(INF.downloaded)), T.in2(y0, hs.map((hh, i) => `h = ${hh}: ${f2(rel[i].AR)}`).join(T.sep), f2(rel[3]['GB global']), f2(rel[3]['GB local']))]);
}

// ================================================================= EXTRA. the inflation results in numbers
const PER = { all: ['2016-01-01', '2099-12-01'], p1620: ['2016-01-01', '2020-12-01'], p2123: ['2021-01-01', '2023-12-01'], p2426: ['2024-01-01', '2099-12-01'] };
function blkIt() {
  const per = $('it-p').value, ref = $('it-r').value, [from, to] = PER[per], hs = ['1', '3', '6', '12'];
  const R = hs.map(h => { const { E } = infErr(h, from, to), r = rmse(E.RW); return { E, n: E.RW.length, rm: Object.fromEntries(INM.map(m => [m, rmse(E[m])])), rw: r,
    dm: Object.fromEntries(INM.filter(m => m !== ref).map(m => [m, dmRaw(E[m].map(v => v * v), E[ref].map(v => v * v), +h)])) }; });
  const G = grid('it-chart', [{ r: 0, c: 0, xt: 'horizon (months)', yt: 'RMSE (pp)', x: { type: 'category' } }], { cols: 1, h: 340, legendRows: 2 });
  const ax = G.ax[0];
  plot('it-chart', INM.map(m => bar(hs.map(h => 'h = ' + h), R.map(r => r.rm[m]), ax, MCOL[m], m)), Object.assign(G.lay, { barmode: 'group' }));
  table('it-tb', [T.L.rmseRel, ...hs.map(h => H_('h = ' + h))], INM.map(m => hiRow([MN(m), ...R.map(r => txt(`${f2(r.rm[m])} (${f2(r.rm[m] / r.rw)})`, r.rm[m] < r.rw))], m === 'AR')), { cap: T.itCap(T.per[per], R.map(r => r.n).join(T.sep)) });
  table('it-dm', [T.L.dmp(MN(ref)), ...hs.map(h => H_('h = ' + h))], INM.filter(m => m !== ref).map(m => [MN(m), ...R.map(r => txt(pv(r.dm[m].p), r.dm[m].p < 0.05))]), { note: T.itNote });
  out('it-out', [T.it1(T.per[per], R.map(r => f2(r.rm.AR / r.rw)).join(T.sep), R.map(r => f2(r.rm['GB global'] / r.rw)).join(T.sep))]);
}

// ================================================================= EXTRA. realised volatility: HAR against ML in numbers
const RVM = ['HAR', 'Lasso', 'RF', 'GB', 'MLP'];
function rvState(k, y0, y1) {
  const V = D.rv[k], yr = V.d.map(d => +d.slice(0, 4)), ix = range(0, V.d.length - 1, 1).filter(i => yr[i] >= y0 && (!y1 || yr[i] <= y1));
  const yi = y => V.years.indexOf(y), varOf = m => ix.map(i => Math.exp(V.pred[m][i] + V.s2[m][yi(yr[i])] / 2));
  const R = {}; RVM.forEach(m => { const vv = varOf(m); R[m] = { v: vv, L: ix.map((i, j) => V.rv[i] / vv[j] + Math.log(vv[j])) }; });
  const sH = sum(ix.map(i => (V.y[i] - V.pred.HAR[i]) ** 2));
  RVM.forEach(m => { R[m].qlike = mean(R[m].L); R[m].r2 = 1 - sum(ix.map(i => (V.y[i] - V.pred[m][i]) ** 2)) / sH; R[m].dm = m === 'HAR' ? null : dmHac(R[m].L.map((v, j) => v - R.HAR.L[j]), 5); });
  return { R, ix, n: ix.length };
}
function blkRt() {
  const y0 = +$('rt-y').value; set('rt-y', y0);
  const S = { sp500: memo('rv|sp500|' + y0, () => rvState('sp500', y0)), dax: memo('rv|dax|' + y0, () => rvState('dax', y0)) };
  const G = grid('rt-chart', [{ r: 0, c: 0, t: 'R^2 against HAR (%)' }, { r: 0, c: 1, t: 'DM t of the QLIKE difference against HAR' }], { cols: 2, h: 340, legendRows: 1 });
  const [a, b] = G.ax, ml = RVM.slice(1);
  plot('rt-chart', [bar(ml, ml.map(m => 100 * S.sp500.R[m].r2), a, K.blue, 'S&P 500'), bar(ml, ml.map(m => 100 * S.dax.R[m].r2), a, K.red, 'DAX'),
    bar(ml, ml.map(m => S.sp500.R[m].dm.t), b, K.blue, ''), bar(ml, ml.map(m => S.dax.R[m].dm.t), b, K.red, ''),
    hline(b, [ml[0], ml[3]], -1.96, K.navy, 't = -1.96', { line: { dash: 'dot', width: 1 } })], Object.assign(G.lay, { barmode: 'group' }));
  const row = m => [MN(m), ...['sp500', 'dax'].flatMap(k => { const r = S[k].R[m]; return [txt(sgn(100 * r.r2, 1)), num(r.qlike, 3), m === 'HAR' ? txt('–') : txt(f2(r.dm.t), r.dm.t < -1.96)]; })];
  table('rt-tb', [T.L.model, H_('S&P: R² vs HAR (%)'), H_('QLIKE'), H_('DM t'), H_('DAX: R² vs HAR (%)'), H_('QLIKE'), H_('DM t')], RVM.map(row), { cap: T.rtCap(y0, S.sp500.n, S.dax.n) });
  out('rt-out', [T.rt1(y0, S.sp500.n, S.dax.n, sgn(100 * S.sp500.R.Lasso.r2, 1), sgn(100 * S.sp500.R.MLP.r2, 1), f2(S.sp500.R.Lasso.dm.t), f2(S.sp500.R.MLP.dm.t))]);
}

// ================================================================= 17. volatility forecasts in two storms
const RVSETS = { 'RF,GB': ['RF', 'GB'], 'Lasso,MLP': ['Lasso', 'MLP'], 'RF,Lasso': ['RF', 'Lasso'], 'GB,MLP': ['GB', 'MLP'] };
function blkRv() {
  const k = $('rv-k').value, ms = RVSETS[$('rv-m').value], ya = +$('rv-a').value, yb = +$('rv-b').value, V = D.rv[k], nm = { sp500: 'S&P 500', dax: 'DAX' }[k];
  const win = y => { const e = y === 2025 ? '2025-09-10' : `${y}-12-31`; return range(0, V.d.length - 1, 1).filter(i => V.d[i] >= `${y}-01-01` && V.d[i] <= e); };
  const G = grid('rv-chart', [{ r: 0, c: 0, t: `${nm}, ${ya}`, yt: 'volatility (% per year)', x: DATE }, { r: 0, c: 1, t: `${nm}, ${yb}`, x: DATE }], { cols: 2, h: 380, legendRows: 2 });
  const tr = [], ql = [];
  [[ya, G.ax[0], 0], [yb, G.ax[1], 1]].forEach(([y, ax, s]) => {
    const ix = win(y), yi = V.years.indexOf(y), dd = ix.map(i => V.d[i]);
    tr.push(ln(dd, ix.map(i => Math.sqrt(252 * V.rv[i])), ax, K.black, s ? '' : 'realised, next 5 days', { line: { width: 1.4 } }));
    const qq = {};
    ['HAR', ...ms].forEach((m, j) => { const v = ix.map(i => Math.exp(V.pred[m][i] + V.s2[m][yi] / 2));
      tr.push(ln(dd, v.map(x => Math.sqrt(252 * x)), ax, MCOL[m], s ? '' : `${m} forecast`, { line: { width: 1.4, dash: ['solid', 'dash', 'dot'][j] } }));
      qq[m] = mean(ix.map((i, q) => V.rv[i] / v[q] + Math.log(v[q]))); });
    ql.push([y, qq]);
  });
  plot('rv-chart', tr, G.lay);
  out('rv-out', ql.map(([y, qq]) => T.rv1(nm, y, Object.entries(qq).map(([m, v]) => `${m} ${f3(v)}`).join(T.sep))));
}

// ================================================================= EXTRA. the sign of tomorrow's return: accuracy, the baseline and AUC
const SGM = ['Logit', 'RF', 'GB'];
function signState(k, y0, thr) {
  const S = D.sign[k], yr = S.d.map(d => +d.slice(0, 4)), ix = range(0, S.d.length - 1, 1).filter(i => yr[i] >= y0), n = ix.length, y = ix.map(i => S.y[i]);
  const base = ix.map(i => S.base[S.years.indexOf(yr[i])]), ab = mean(base.map((b, j) => +(b === y[j])));
  const R = { n, up: mean(y), base: ab };
  SGM.forEach(m => { const p = ix.map(i => S.p[m][i]), acc = mean(p.map((v, j) => +((v >= thr ? 1 : 0) === y[j]))), z = (acc - ab) / Math.sqrt(ab * (1 - ab) / n);
    R[m] = { acc, diff: acc - ab, z, p: nP2(z), auc: memo(`auc|${k}|${y0}|${m}`, () => auc(y, p)), up: mean(p.map(v => +(v >= thr))) }; });
  return R;
}
function blkSb() {
  const y0 = +$('sb-y').value; set('sb-y', y0);
  const S = { sp500: signState('sp500', y0, 0.5), bet: signState('bet', y0, 0.5) }, nm = { sp500: 'S&P 500', bet: 'BET' };
  const G = grid('sb-chart', [{ r: 0, c: 0, t: 'AUC (0.5: no ranking skill)', y: { range: [0.45, 0.6] } }], { cols: 1, h: 320, legendRows: 1 });
  const ax = G.ax[0];
  plot('sb-chart', [bar(SGM, SGM.map(m => S.sp500[m].auc), ax, K.blue, 'S&P 500'), bar(SGM, SGM.map(m => S.bet[m].auc), ax, K.red, 'BET'), hline(ax, [SGM[0], SGM[2]], 0.5, K.navy, '', { line: { dash: 'dot', width: 1 } })],
    Object.assign(G.lay, { barmode: 'group' }));
  const rows = []; ['sp500', 'bet'].forEach(k => { const R = S[k]; rows.push(hiRow([`<b>${nm[k]}</b>`, txt(pc(R.base)), txt('–'), txt('–'), txt('–'), txt('–'), txt(T.baseRow(pc(R.up)))], true));
    SGM.forEach(m => rows.push([MN(m), txt(pc(R[m].acc)), txt(sgn(100 * R[m].diff, 1)), txt(f2(R[m].z)), txt(pv(R[m].p), R[m].p < 0.05), txt(f3(R[m].auc)), txt(pc(R[m].up))])); });
  table('sb-tb', [T.L.model, H_(T.L.acc), H_(T.L.vsBase), H_('z'), H_('p'), H_('AUC'), H_(T.L.upShare)], rows, { cap: T.sbCap(y0, S.sp500.n, S.bet.n) });
  const all = ['sp500', 'bet'].flatMap(k => SGM.map(m => S[k][m].auc));
  out('sb-out', [T.sb1(y0, pc(S.sp500.up), pc(S.bet.up), f3(Math.min(...all)), f3(Math.max(...all)))]);
}
function blkSi() {
  const thr = +$('si-t').value; set('si-t', f2(thr));
  const S = { sp500: memo('si|sp|' + thr, () => signState('sp500', 2008, thr)), bet: memo('si|bet|' + thr, () => signState('bet', 2008, thr)) };
  const G = grid('si-chart', [{ r: 0, c: 0, yt: 'accuracy (%)', y: { range: [45, 60] } }], { cols: 1, h: 340, legendRows: 2 });
  const ax = G.ax[0], xs = ['sp500', 'bet'].map(k => `${{ sp500: 'S&P 500', bet: 'BET' }[k]}, ${D.sign[k].d[0].slice(0, 4)}-2026`);
  const ser = [['base', K.amber, 'baseline: always the majority class'], ['Logit', K.blue, 'Logit'], ['RF', K.green, 'RF'], ['GB', K.red, 'GB']];
  plot('si-chart', ser.map(([m, c, lab]) => { const v = ['sp500', 'bet'].map(k => 100 * (m === 'base' ? S[k].base : S[k][m].acc));
    return bar(xs, v, ax, c, lab, { text: v.map(x => x.toFixed(1)), textposition: 'outside', cliponaxis: false }); }), Object.assign(G.lay, { barmode: 'group' }));
  out('si-out', [T.si1(f2(thr), pc(S.sp500.base), SGM.map(m => `${MN(m)} ${pc(S.sp500[m].acc)}`).join(T.sep), pc(S.bet.base), SGM.map(m => `${MN(m)} ${pc(S.bet[m].acc)}`).join(T.sep)),
    T.si2(pc(S.sp500.GB.up), pv(S.sp500.GB.p))]);
}

// ================================================================= 18. M4: the accuracy of all methods
const M4 = D.m4, M4T = { Statistical: K.blue, 'Combination (S)': K.teal, 'Combination (S & ML)': K.purple, 'Combination (ML)': K.orange, 'Machine Learning': K.red, Hybrid: K.green, Other: K.amber };
function blkM4() {
  const w = +$('m4-w').value; set('m4-w', f1(w));
  const i2 = M4.method.indexOf('Naive2'), ic = M4.method.indexOf('Comb'), ow = M4.smape.map((s, i) => w * s / M4.smape[i2] + (1 - w) * M4.mase[i] / M4.mase[i2]);
  const ord = range(0, ow.length - 1, 1).sort((a, b) => ow[a] - ow[b]), rk = []; ord.forEach((i, r) => { rk[i] = r + 1; });
  const G = grid('m4-chart', [{ r: 0, c: 0, xt: 'rank (by OWA)', yt: 'OWA (Naive2 = 1)', y: { range: [0.78, 1.7] } }], { cols: 1, h: 330, legendRows: 2 });
  const ax = G.ax[0], n = ow.length;
  plot('m4-chart', [...Object.entries(M4T).map(([t, c]) => { const ix = range(0, n - 1, 1).filter(i => M4.type[i] === t);
    return mk(ix.map(i => rk[i]), ix.map(i => ow[i]), ax, c, t, { marker: { size: 8, color: c }, text: ix.map(i => M4.method[i]), hovertemplate: '%{text}: %{y:.3f}<extra></extra>' }); }),
    hline(ax, [1, n], 1, K.amber, '', { line: { dash: 'dot', width: 1.2 } }), hline(ax, [1, n], ow[ic], K.teal, '', { line: { dash: 'dash', width: 1 } })],
    Object.assign(G.lay, { annotations: G.lay.annotations.concat([{ x: n, y: 1, xref: ax.xa, yref: ax.ya, text: 'Naive2', showarrow: false, xanchor: 'right', yanchor: 'bottom', font: { size: 11, color: K.navy } },
      { x: n, y: ow[ic], xref: ax.xa, yref: ax.ya, text: 'Comb (SES, Holt, damped)', showarrow: false, xanchor: 'right', yanchor: 'top', font: { size: 11, color: K.navy } }]) }));
  const ml = range(0, n - 1, 1).filter(i => M4.type[i] === 'Machine Learning'), mb = ml.reduce((b, i) => (ow[i] < ow[b] ? i : b), ml[0]);
  out('m4-out', [T.m41(f1(w), M4.method[ord[0]], M4.type[ord[0]], f3(ow[ord[0]]), f3(ow[ic]), rk[ic]), T.m42(ml.length, rk[mb], f3(ow[mb]), ml.filter(i => ow[i] < ow[ic]).length, ml.filter(i => ow[i] < 1).length, ow.filter(v => v > 1.7).length)]);
}

// ================================================================= 19. Gu, Kelly and Xiu (2020): published numbers
function blkGk() {
  const m = $('gk-m').value, G0 = D.gkx, j = G0.models.indexOf(m), cl = G0.models.map((_, i) => (i < 5 ? K.blue : i < 7 ? K.green : K.red));
  const G = grid('gk-chart', [{ r: 0, c: 0, t: 'monthly out-of-sample R^2 (%)', x: { tickangle: -60, tickfont: { size: 10 } } }, { r: 0, c: 1, t: 'Sharpe ratio, long-short decile portfolio', x: { tickangle: -60, tickfont: { size: 10 } } }],
    { cols: 2, h: 360, legendRows: 1, mb: 80 });
  const [a, b] = G.ax;
  plot('gk-chart', [bar(G0.models, G0.r2, a, cl, ''), bar(G0.models, G0.sr, b, cl, ''),
    mk([m], [G0.r2[j]], a, K.navy, `selected: ${m}`, { marker: { size: 12, symbol: 'triangle-down', color: K.navy } }), mk([m], [G0.sr[j]], b, K.navy, '', { marker: { size: 12, symbol: 'triangle-down', color: K.navy } }),
    leg('linear models', null, { color: K.blue, symbol: 'square', size: 10 }), leg('trees', null, { color: K.green, symbol: 'square', size: 10 }), leg('neural networks', null, { color: K.red, symbol: 'square', size: 10 })], G.lay);
  out('gk-out', [T.gk1(m, f2(G0.r2[j]), f2(G0.sr[j]), f2(G0.sr[j] / G0.sr[0])), T.gk2(m, f2(G0.sr[j] * Math.sqrt(30)))]);
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
const BLK = [['features', 'fe', blkFe, 'fe-h', range(1, 14, 1), 700], ['wx-lag', 'wl', blkWl, 'wl-e', range(-14, 14, 1), 500],
  ['strategies', 'st', blkSt, 'st-o', null, 800], ['wx-strat', 'ws', blkWs, 'ws-y', range(3, 8, 0.5), 600],
  ['cv-schemes', 'cv', blkCv, 'cv-g', range(0, 6, 1), 700], ['leakage', 'lk', blkLk, 'lk-h', null, 900], ['wx-leak', 'wk', blkWk, 'wk-h', null, 900],
  ['bias-variance', 'bv', blkBv, 'bv-s', range(0.1, 0.6, 0.05), 600], ['wx-shrink', 'wr', blkWr, 'wr-l', range(0, 200, 10), 400],
  ['shrinkage', 'sh', blkSh, 'sh-e', null, 900], ['lasso-table', 'lt', blkLt, 'lt-k', range(-12, 12, 2), 600],
  ['tree', 'tr', blkTr, 'tr-e', null, 1000], ['wx-boost', 'wb', blkWb, 'wb-n', range(0.05, 1, 0.05), 400], ['ensembles', 'en', blkEn, 'en-s', null, 1000],
  ['extrapolation', 'ex', blkEx, 'ex-s', null, 1000], ['importance', 'im', blkIm, 'im-h', range(1, 7, 1), 800],
  ['mlp', 'ml', blkMl, 'ml-q', range(1, 10, 1), 600], ['rnn', 'rn', blkRn, 'rn-o', null, 700],
  ['load-table', 'ld', blkLd, 'ld-k', range(0, 20, 2), 700], ['load-results', 'lr', blkLr, 'lr-n', range(6, 26, 2), 600], ['wx-mase', 'wm', blkWm, 'wm-m', range(1, 14, 1), 600],
  ['wx-dm', 'wd', blkWd, 'wd-a', null, 900], ['load-forecasts', 'lf', blkLf, 'lf-o', null, 800],
  ['wx-pinball', 'wp', blkWp, 'wp-t', range(0.05, 0.95, 0.05), 500], ['wx-conformal', 'wc', blkWc, 'wc-a', range(0.5, 0.95, 0.05), 600], ['intervals', 'iv', blkIv, 'iv-l', range(0.5, 0.95, 0.05), 700],
  ['inflation', 'in', blkIn, 'in-y', range(2016, 2022, 1), 800], ['inf-table', 'it', blkIt, 'it-p', null, 1000],
  ['rv-table', 'rt', blkRt, 'rt-y', range(2013, 2024, 1), 800], ['rv', 'rv', blkRv, 'rv-m', null, 1100],
  ['sign-table', 'sb', blkSb, 'sb-y', range(2008, 2020, 2), 900], ['sign', 'si', blkSi, 'si-t', range(0.4, 0.6, 0.02), 600],
  ['m4', 'm4', blkM4, 'm4-w', range(0, 1, 0.1), 600], ['gkx', 'gk', blkGk, 'gk-m', null, 600]];
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
  Object.entries(T.predict).forEach(([b, sp]) => { if (has(b) && players[b]) predict(b, Object.assign({ player: players[b], veil: { out: true } }, sp)); });
  exitTicket(T.ticket);
  live.forEach(b => b[2]());
  let w = window.innerWidth, tm = null;
  window.addEventListener('resize', () => { clearTimeout(tm); tm = setTimeout(() => { if ((window.innerWidth < 720) !== (w < 720) || Math.abs(window.innerWidth - w) > 200) { w = window.innerWidth; live.forEach(b => b[2]()); } }, 250); });
}
window.CH9 = { featState, stState, cvState, r2p, targets, cart, leaves, treeState, boost, loadStats, dmRaw, dmHac, auc, tP2, scaleM, confCover, infErr, rvState, signState, SCALE, ACT, ERR,
  ridge1, lasso1, pinball, entryOrder, BLK, init };
if (!window.CH9_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
