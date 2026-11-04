// Chapter 11 interactive page (foundation models for time series): one block per chart slide, plus the result tables and
// the worked examples of the lecture, computed in the browser from ch11_data.js (Quantlets/Ch_11/export_interactive.py):
// the saved backtest of the chapter, the zero-shot quantiles drawn in the charts of the slides and the series of the chapter.
// No foundation model runs in the browser.
(function () {
'use strict';
const D = window.CH11_DATA;
const { sum, mean, nf, $, AX, layout, plot, Player, range, predict, exitTicket, normCdf, normPpf } = window.TSAI;
const out = (id, items) => window.TSAI.out(id, items);
const T = window.CH11_TEXT;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of tsa_style.py / generate_all_charts.py
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', orange: '#E67E22', purple: '#8E44AD', teal: '#17A2B8', crimson: '#DC3545',
  navy: '#1F2A44' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.amber]: '181,133,63', [K.orange]: '230,126,34', [K.purple]: '142,68,173',
  [K.teal]: '23,162,184', [K.navy]: '31,42,68' };
const MC = { 'Seasonal naive': K.amber, ETS: K.green, ARIMA: K.purple, 'Chronos-Bolt tiny': K.teal, 'Chronos-Bolt small': K.blue, 'Chronos-2': K.red };

// ---------------------------------------------------------------- small helpers
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
const has = id => !!$(id);
const cache = {};
const memo = (k, f) => (k in cache ? cache[k] : (cache[k] = f()));
const f0 = x => nf(x, 0), f1 = x => nf(x, 1), f2 = x => nf(x, 2), f3 = x => nf(x, 3), f4 = x => nf(x, 4);
const pc = (x, d) => nf(100 * x, d === undefined ? 0 : d) + '%';
const sdp = x => { const m = mean(x); return Math.sqrt(sum(x.map(v => (v - m) ** 2)) / x.length); };
const cumsum = a => { let s = 0; return a.map(v => (s += v)); };
const argmin = a => a.reduce((b, v, i) => (v < a[b] ? i : b), 0);
const argmax = a => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
const isoQ = (q0, k) => { const y0 = +q0.slice(0, 4), m = +q0.slice(5, 7) - 1 + 3 * k; return `${y0 + Math.floor(m / 12)}-${String(m % 12 + 1).padStart(2, '0')}-01`; };
const isoM = (m0, k) => { const y0 = +m0.slice(0, 4), m = +m0.slice(5, 7) - 1 + k; return `${y0 + Math.floor(m / 12)}-${String(m % 12 + 1).padStart(2, '0')}-01`; };
const phi = z => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);

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
    return { xa: 'x' + n, ya: 'y' + n, xd: 'x' + n + ' domain', yd: 'y' + n + ' domain', k: k + 1, dom: [Math.max(0, y0), y1] };
  });
  el.style.height = H + 'px';
  return { lay, ax, narrow };
}
// a second y axis on the right of panel a (overlaying it); returns the axis handle of the overlay
function overlay(G, a, name, title, extra) {
  const n = name, yname = 'yaxis' + n.slice(1);
  G.lay[yname] = AX(title || '', Object.assign({ overlaying: a.ya, side: 'right', anchor: a.xa, showgrid: false, automargin: false, title: { text: title || '', font: { size: 11, color: K.navy }, standoff: 2 } }, extra || {}));
  G.lay.margin.r = Math.max(G.lay.margin.r, 46);
  return { xa: a.xa, ya: n, xd: a.xd, yd: n + ' domain' };
}
const leg = (name, line, marker) => Object.assign({ x: [null], y: [null], name, hoverinfo: 'skip' }, line ? { mode: 'lines', line } : { mode: 'markers', marker });
const ln = (x, y, a, color, name, o) => { const r = Object.assign({ x, y, mode: 'lines', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }, o || {}); r.line = Object.assign({ color, width: 1.4 }, (o && o.line) || {}); return r; };
const mk = (x, y, a, color, name, o) => Object.assign({ x, y, mode: 'markers', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, marker: { color, size: 6 }, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }, o || {});
const bar = (x, y, a, color, name, o) => Object.assign({ x, y, type: 'bar', marker: { color }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }, o || {});
const band = (x, lo, hi, a, col, al, name) => ({ x: [...x, ...[...x].reverse()], y: [...hi, ...[...lo].reverse()], fill: 'toself', fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, mode: 'lines', hoverinfo: 'skip', name: name || '', showlegend: !!name, xaxis: a.xa, yaxis: a.ya });
const hline = (a, xs, y, color, name, o) => ln(xs, xs.map(() => y), a, color, name, Object.assign({ hoverinfo: 'skip' }, o || {}));
const vline = (a, x, y0, y1, color, name, o) => ln([x, x], [y0, y1], a, color, name, Object.assign({ hoverinfo: 'skip' }, o || {}));
const DATE = { type: 'date' }, LOG = { type: 'log', exponentformat: 'power' };

// ---------------------------------------------------------------- tables and worked steps (HTML)
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}>${v.v}</${tag}>`; };
  el.innerHTML = (o && o.cap ? `<p class="c11cap">${o.cap}</p>` : '') + '<table><thead><tr>' + head.map(c => cell(c, 'th')).join('') + '</tr></thead><tbody>'
    + rows.map(r => '<tr' + (r.hi ? ' class="hi"' : '') + '>' + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c11cap">${o.note}</p>` : '');
}
const txt = (v, bold) => ({ v: bold ? `<b>${v}</b>` : v, cls: 'r' });
const H_ = v => ({ v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const hiRow = (r, on) => { r.hi = on; return r; };


// ================================================================= shared: dates, scores, the saved backtest
const hourIso = (t0, k) => new Date(Date.parse(t0.replace(' ', 'T').slice(0, 16) + ':00Z') + k * 3600e3).toISOString().slice(0, 16).replace('T', ' ');
const shiftM = (d, k) => { const t = 12 * +d.slice(0, 4) + +d.slice(5, 7) - 1 + k, y = Math.floor(t / 12); return `${y}-${String(t - 12 * y + 1).padStart(2, '0')}-${d.slice(8, 10)}`; };
const pinball = (y, q, tau) => { const u = y - q; return Math.max(tau * u, (tau - 1) * u); };
const crpsN = (y, mu, s) => { const z = (y - mu) / s; return s * (z * (2 * normCdf(z) - 1) + 2 * phi(z) - 1 / Math.sqrt(Math.PI)); };
const LEV9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
const BS = D.bench.series, SER = Object.keys(BS), MODELS = D.bench.models, FMS = MODELS.slice(3), RIVALS = MODELS.slice(1);
const SH = D.bench.short, SH2 = k => SH[k].replace(' ', '<br>');
const lab = m => m.replace('Chronos-Bolt', 'Bolt');
const nm = k => T.sname[k];
const mname = m => T.mn[m] || lab(m), mfull = m => T.mn[m] || m;
function summ(k, mask) {                    // generate_all_charts.summarise on the origins where mask is true
  const S = BS[k], I = S.origins.map((_, i) => i).filter(i => mask[i]);
  if (!I.length) return null;
  const r = { _n: I.length };
  MODELS.forEach(m => { const M = S.models[m]; r[m] = { mase: mean(I.map(i => M.mase[i])), wql: sum(I.map(i => M.num[i])) / sum(I.map(i => M.den[i])), cov: mean(I.map(i => M.cov[i])) }; });
  MODELS.forEach(m => { r[m].rel_mase = r[m].mase / r['Seasonal naive'].mase; r[m].rel_wql = r[m].wql / r['Seasonal naive'].wql; });
  return r;
}
const fromMask = (k, from) => BS[k].origins.map(o => !from || o >= from);
function results(from) {                    // generate_all_charts.table_results, on the origins from a date (all: from = '')
  return memo('res|' + from, () => {
    const R = {}; SER.forEach(k => { const s = summ(k, fromMask(k, from)); if (s) R[k] = s; });
    const ks = Object.keys(R), gm = {};
    MODELS.forEach(m => { gm[m] = { rel_mase: Math.exp(mean(ks.map(k => Math.log(R[k][m].rel_mase)))), rel_wql: Math.exp(mean(ks.map(k => Math.log(R[k][m].rel_wql)))),
      cov: mean(ks.map(k => R[k][m].cov)), wins: ks.filter(k => R[k][m].rel_wql === Math.min(...MODELS.map(x => R[k][x].rel_wql))).length }; });
    return { R, gm, ks, n: sum(ks.map(k => R[k]._n)) };
  });
}
const FROM = () => ['', '2018-01-01', '2020-01-01', '2022-01-01', '2024-01-01'];
const fromLab = f => (f ? f.slice(0, 4) : '');
const best = (o, ks, f) => ks.reduce((b, k) => (f(o[k]) < f(o[b]) ? k : b), ks[0]);
const fmtList = (pairs, d) => pairs.map(([n, v]) => `${n} ${nf(v, d === undefined ? 2 : d)}`).join(T.sep);
function groupBars(a, R, ks, models, col) {
  return models.map(m => bar(ks.map(k => SH[k]), ks.map(k => R[k][m][col]), a, MC[m], a.k === 1 ? m : '', { legendgroup: m, hovertemplate: `${m}, %{x}: %{y:.3f}<extra></extra>` }));
}

// ================================================================= 1. worked example: attention weights
function attn(c) {
  const sc = [2 * c, c, 0], e = sc.map(Math.exp), S = sum(e), w = e.map(v => v / S), v = [10, 20, 30];
  return { sc, e, S, w, v, out: sum(w.map((x, j) => x * v[j])) };
}
function blkWa() {
  const c = +$('wa-c').value; set('wa-c', f2(c));
  const A = attn(c), keys = ['key 1 (v = 10)', 'key 2 (v = 20)', 'key 3 (v = 30)'];
  const G = grid('wa-chart', [{ r: 0, c: 0, t: 'weights w_j (softmax of the scores)', y: { range: [0, 1.08] } }, { r: 0, c: 1, t: 'w_j v_j and the output', y: { range: [0, 31] } }],
    { cols: 2, h: 340 });
  const [a, b] = G.ax;
  plot('wa-chart', [bar(keys, A.w, a, K.blue, 'weight w_j', { text: A.w.map(x => x.toFixed(2)), textposition: 'outside' }),
    bar(keys, A.w.map((x, j) => x * A.v[j]), b, K.amber, 'contribution w_j v_j'), hline(b, keys, A.out, K.red, `output = sum of w_j v_j = ${A.out.toFixed(1)}`, { line: { width: 2, dash: 'dash' } })], G.lay);
  steps('wa-ws', [T.was1(f2(A.sc[0]), f2(A.sc[1]), f2(A.sc[2])), T.was2(f3(A.e[0]), f3(A.e[1]), f3(A.e[2]), f3(A.S)), T.was3(f2(A.w[0]), f2(A.w[1]), f2(A.w[2])),
    T.was4(f2(A.w[0]), f2(A.w[1]), f2(A.w[2]), f1(A.out))]);
  out('wa-out', [T.wa1(f2(c), pc(A.w[0]), f1(A.out))]);
  return A;
}

// ================================================================= 2. mean scaling and quantisation of industrial production
const TOKB = { 21: [-3, 3, 21], 41: [-3, 3, 41], 101: [-3, 3, 101], 4093: [-15, 15, 4093] };
function quantise(x, lo, hi, n) {
  const s = mean(x.map(Math.abs)), w = (hi - lo) / (n - 1), z = x.map(v => v / s), tok = z.map(v => Math.round((v - lo) / w));
  return { s, w, z, tok, cen: tok.map(t => lo + t * w), ntok: new Set(tok).size };
}
function blkTk() {
  const n = +$('tk-n').value, B = $('tk-b').value; set('tk-n', n);
  const all = D.tokens.y, x = all.slice(all.length - n), d = x.map((_, i) => isoM(D.tokens.m0, all.length - n + i));
  const [lo, hi, nb] = TOKB[B], Q = quantise(x, lo, hi, nb);
  const G = grid('tk-chart', [{ r: 0, c: 0, t: `Context: last ${n} months`, x: DATE }, { r: 0, c: 1, t: 'Scaled context and its tokens', x: DATE,
    y: { range: [Math.min(...Q.z) - 0.08, Math.max(...Q.z) + 0.08] } }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax, gl = [];
  if (nb <= 101) for (let t = 0; t < nb; t++) { const g = lo + t * Q.w; if (g > 0.5 && g < 1.5) gl.push(hline(b, [d[0], d[d.length - 1]], g, K.amber, '', { line: { width: 0.6, dash: 'dot' } })); }
  plot('tk-chart', [ln(d, x, a, K.blue, 'Industrial production (2021 = 100)'), hline(a, [d[0], d[d.length - 1]], Q.s, K.red, `scale s = mean |x| = ${Q.s.toFixed(1)}`, { line: { width: 1, dash: 'dash' } }),
    ...gl, ln(d, Q.z, b, K.blue, 'scaled value x / s', { line: { width: 1 } }),
    ln(d, Q.cen, b, K.red, `nearest bin centre (token), ${nb} bins`, { line: { width: 1.4, shape: 'hvh' } })], G.lay);
  const ia = d.indexOf('2020-04-01'), ta = ia >= 0 ? Q.tok[ia] : null;
  out('tk-out', [T.tk1(n, T.my(d[0]), T.my(d[d.length - 1]), f1(Q.s), f2(Math.min(...Q.z)), f2(Math.max(...Q.z))),
    T.tk2(nb, nf(Q.w, nb > 1000 ? 4 : 2), nf(lo, 0), nf(hi, 0), Q.ntok),
    ia >= 0 ? T.tk3(ta, Q.tok.filter(t => t === ta).length) : T.tk3n, T.tk4(T.my(D.tokens.rev.month), f1(D.tokens.rev.slide), f1(D.tokens.rev.today), T.dmy(D.downloaded))]);
  return { s: Q.s, zmin: Math.min(...Q.z), zmax: Math.max(...Q.z), ntok: Q.ntok, apr: ia >= 0 ? Q.tok.filter(t => t === ta).length : null };
}

// ================================================================= 3. worked example: scaling and tokens (unemployment)
function blkWt() {
  const nb = +$('wt-b').value, u = NB.ex.u, d = NB.ex.u_dates, Q = quantise(u, -15, 15, nb);
  const G = grid('wt-chart', [{ r: 0, c: 0, t: 'scaled values x_t / s and their bin centres', x: DATE }, { r: 0, c: 1, t: 'tokens (bin numbers)', x: DATE }], { cols: 2, h: 340 });
  const [a, b] = G.ax, gl = [];
  const zl = Math.min(...Q.z, ...Q.cen) - 0.02, zh = Math.max(...Q.z, ...Q.cen) + 0.02;
  for (let t = Math.ceil((zl + 15) / Q.w); -15 + t * Q.w <= zh && gl.length < 40; t++) gl.push(hline(a, [d[0], d[5]], -15 + t * Q.w, K.amber, '', { line: { width: 0.6, dash: 'dot' } }));
  G.lay[a.ya.replace('y', 'yaxis')].range = [zl, zh];
  const tmin = Math.min(...Q.tok), tmax = Math.max(...Q.tok);
  G.lay[b.ya.replace('y', 'yaxis')].range = [tmin - Math.max(2, 0.6 * (tmax - tmin)), tmax + Math.max(2, 0.6 * (tmax - tmin))];
  plot('wt-chart', [...gl, mk(d, Q.z, a, K.blue, 'x_t / s', { marker: { size: 9, color: K.blue } }), mk(d, Q.cen, a, K.red, 'nearest bin centre', { marker: { size: 9, symbol: 'x', color: K.red } }),
    mk(d, Q.tok, b, K.red, 'token', { mode: 'markers+text', text: Q.tok.map(String), textposition: 'top center', marker: { size: 8, color: K.red } })], G.lay);
  const j = +$('wt-m').value, r = (Q.z[j] + 15) / Q.w;
  steps('wt-ws', [T.wts1(u.map(f1).join(T.semi), f2(Q.s)), T.wts2(T.my(d[j]), f1(u[j]), f2(Q.s), f3(Q.z[j])), T.wts3(nb, nf(Q.w, nb > 1000 ? 5 : 3), f3(Q.z[j]), f2(r), Q.tok[j], f4(Q.cen[j])),
    T.wts4(Q.tok.join(', '))]);
  out('wt-out', [T.wt1(nb, new Set(Q.tok).size, nf(0.1 / Q.s / Q.w, 2))]);
  return Q;
}

// ================================================================= 4. patching the Romanian load
function blkPa() {
  const P = +$('pa-P').value, n = +$('pa-n').value; set('pa-P', P); set('pa-n', n);
  const y = D.series.load.y.slice(0, n), t = y.map((_, i) => hourIso(D.series.load.t0, i)), np = Math.floor(n / P);
  const odd = y.map((v, i) => (Math.floor(i / P) % 2 === 0 && i < np * P ? v : null)), even = y.map((v, i) => (Math.floor(i / P) % 2 === 1 && i < np * P ? v : null));
  const rest = y.map((v, i) => (i >= np * P ? v : null));
  const G = grid('pa-chart', [{ r: 0, c: 0, t: `${n} hourly values = ${np} patches of ${P} values${n % P ? ` (+ ${n % P} left over)` : ''}`, yt: 'GW', x: DATE }], { cols: 1, h: 330 });
  const a = G.ax[0];
  G.lay.shapes = range(0, np - 1, 1).map(k => ({ type: 'rect', xref: a.xa, yref: a.yd, x0: t[k * P], x1: t[Math.min(n - 1, k * P + P - 1)], y0: 0, y1: 1,
    fillcolor: `rgba(${RGB[k % 2 ? K.red : K.blue]},0.07)`, line: { width: 0 }, layer: 'below' }));
  plot('pa-chart', [ln(t, odd, a, K.blue, 'odd patches', { connectgaps: false, line: { width: 1.3 } }), ln(t, even, a, K.red, 'even patches', { connectgaps: false, line: { width: 1.3 } }),
    ...(n % P ? [ln(t, rest, a, K.navy, 'values left over', { line: { width: 1, dash: 'dot' } })] : [])], G.lay);
  const c2048 = Math.ceil(2048 / P);
  out('pa-out', [T.pa1(n, np, P, n % P), T.pa2(T.big(n * n), T.big(np * np), f0(n * n / (np * np))), T.pa3(P, nf(P / 24, 2), Math.ceil(168 / P), c2048, T.big(c2048 * c2048))]);
  return { np, wk: Math.ceil(168 / P), c2048 };
}

// ================================================================= 5. the families side by side (extra chart: inputs per context)
const FAMH = { none: [], apache: [0, 1, 2, 3, 5], patch: [1, 2, 3, 4], dec: [3, 5] };
function blkFt() {
  const C = +$('ft-c').value; set('ft-c', C);
  const cs = range(128, 4096, 128), fam = [['Chronos: one token per value', 1, K.teal], ['Chronos-Bolt, Chronos-2: patches of 16', 16, K.blue], ['TimesFM: input patches of 32', 32, K.red]];
  const G = grid('ft-chart', [{ r: 0, c: 0, t: 'inputs to the Transformer', xt: 'context length C (values)', x: LOG, y: LOG },
    { r: 0, c: 1, t: 'attention comparisons per layer (inputs squared)', xt: 'context length C (values)', x: LOG, y: LOG }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax, tr = [];
  fam.forEach(([nm, P, col]) => { const n = cs.map(c => Math.ceil(c / P)), nC = Math.ceil(C / P);
    tr.push(ln(cs, n, a, col, nm), ln(cs, n.map(v => v * v), b, col, ''), mk([C], [nC], a, col, '', { marker: { size: 9, color: col } }), mk([C], [nC * nC], b, col, '', { marker: { size: 9, color: col } })); });
  plot('ft-chart', tr, G.lay);
  const h = $('ft-h').value, H = FAMH[h], rows = T.fam.map((r, i) => hiRow(r.map(v => ({ v })), H.includes(i)));
  table('ft-tb', T.famHead, rows, { note: h === 'none' ? T.famNote : T.famSel(h, H.length) });
  out('ft-out', [T.ft1(C, C, T.big(C * C), Math.ceil(C / 16), T.big(Math.ceil(C / 16) ** 2), Math.ceil(C / 32), T.big(Math.ceil(C / 32) ** 2))]);
  return { C, n16: Math.ceil(C / 16), c1: C * C };
}

// ================================================================= 6. the pinball loss and the CRPS
function blkSc() {
  const y = +$('sc-y').value, s = +$('sc-s').value; set('sc-y', f2(y)); set('sc-s', f2(s));
  const u = range(-3, 3, 0.02), z = range(-6, 6, 0.02), taus = [[0.1, K.teal], [0.5, K.blue], [0.9, K.red]];
  const crps = crpsN(y, 0, s);
  const G = grid('sc-chart', [{ r: 0, c: 0, t: 'Pinball loss', xt: 'y - q (outcome minus quantile forecast)', y: { range: [0, 2.75] } },
    { r: 0, c: 1, t: `CRPS = ${crps.toFixed(3)}`, y: { range: [-0.05, 1.05] } }], { cols: 2, h: 380, legendRows: 3 });
  const [a, b] = G.ax, tr = [];
  taus.forEach(([t, c]) => { const ut = Math.max(-3, Math.min(3, y - s * normPpf(t)));
    tr.push(ln(u, u.map(v => pinball(v, 0, t)), a, c, `level ${t}`), mk([ut], [pinball(ut, 0, t)], a, c, '', { marker: { size: 9, color: c } })); });
  const F = z.map(v => normCdf(v / s)), Hs = z.map(v => +(v >= y));
  tr.push(ln(z, Hs, b, K.red, `outcome y = ${y.toFixed(2)} (step)`, { line: { width: 1.6, shape: 'hv' } }),
    ln(z, F, b, K.blue, `predictive CDF, N(0, ${(s * s).toFixed(2)})`, { fill: 'tonexty', fillcolor: `rgba(${RGB[K.amber]},0.35)` }),
    leg(`gap between the CDF and the step;${G.narrow ? '<br>' : ' '}CRPS = integral of the squared gap`, null, { color: `rgba(${RGB[K.amber]},0.5)`, symbol: 'square', size: 12 }));
  plot('sc-chart', tr, G.lay);
  const L = [0.1, 0.5, 0.9].map(t => pinball(y, s * normPpf(t), t)), L9 = LEV9.map(t => pinball(y, s * normPpf(t), t));
  out('sc-out', [T.sc1(f2(y), f2(s * s), f3(crps)), T.sc2(f3(s * normPpf(0.9)), f3(L[0]), f3(L[1]), f3(L[2]), f3(2 * mean(L9))),
    T.sc3(f3(crpsN(0, 0, s)), f2(s * s), f3(crpsN(y, 0, 2 * s)), f2(4 * s * s), f2(y))]);
  return { crps, c0: crpsN(0, 0, s), cw: crpsN(y, 0, 2 * s), q9: 2 * mean(L9) };
}

// ================================================================= 7. the weighted quantile loss on the backtest
function blkWw() {
  const k = $('ww-s').value, m = $('ww-m').value, S = BS[k], o = S.origins, M = S.models[m], N0 = S.models['Seasonal naive'];
  const io = $('ww-o'); io.max = o.length; if (+io.value > o.length) io.value = o.length; const i = +io.value - 1; set('ww-o', `${i + 1} (${o[i].slice(0, 10)})`);
  const per = M.num.map((v, j) => v / M.den[j]), perN = N0.num.map((v, j) => v / N0.den[j]);
  const W = MODELS.map(x => sum(S.models[x].num) / sum(S.models[x].den)), Wm = W[MODELS.indexOf(m)], Wn = W[0];
  const dx = k === 'load' ? o.map(x => x.slice(0, 10)) : o;
  const G = grid('ww-chart', [{ r: 0, c: 0, t: `WQL of each origin, ${SH[k]}`, x: DATE, y: { rangemode: 'tozero' } }, { r: 0, c: 1, t: 'pooled WQL over all origins', y: { rangemode: 'tozero' } }],
    { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax;
  plot('ww-chart', [ln(dx, perN, a, K.amber, 'Seasonal naive', { line: { width: 1 } }), ln(dx, per, a, MC[m], m, { line: { width: 1.3 } }), mk([dx[i]], [per[i]], a, MC[m], '', { marker: { size: 10, color: MC[m], symbol: 'circle-open', line: { width: 2 } } }),
    bar(MODELS.map(lab), W, b, MODELS.map(x => (x === m ? MC[x] : `rgba(${RGB[MC[x]]},0.35)`)), '', { hovertemplate: '%{x}: %{y:.4f}<extra></extra>' })], G.lay);
  steps('ww-ws', [T.wws1(i + 1, o[i].slice(0, 10), f3(M.num[i]), f3(M.den[i]), f4(per[i])), T.wws2(o.length, f2(sum(M.num)), f2(sum(M.den)), f4(Wm)),
    T.wws3(f4(Wn), f4(Wm), f2(Wm / Wn))]);
  out('ww-out', [T.ww1(m, nm(k), f4(Wm), f2(Wm / Wn), o.length, pc(mean(per.map((v, j) => +(v < perN[j])))))]);
  return { rel: Wm / Wn, W: Wm };
}

// ================================================================= 8. worked example: pinball losses
const PINQ = [[0.1, 6.5, K.teal], [0.5, 7.0, K.blue], [0.9, 7.6, K.red]];
function blkWp() {
  const y = +$('wp-y').value; set('wp-y', f1(y));
  const ys = range(6.0, 8.4, 0.02);
  const G = grid('wp-chart', [{ r: 0, c: 0, t: 'loss of each quantile forecast against the outcome y', xt: 'outcome y (%)', y: { rangemode: 'tozero' } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0], tr = [];
  PINQ.forEach(([t, q, c]) => tr.push(ln(ys, ys.map(v => pinball(v, q, t)), a, c, `level ${t}, q = ${q.toFixed(1)}`), mk([y], [pinball(y, q, t)], a, c, '', { marker: { size: 9, color: c } })));
  const cr = ys.map(v => (2 / 3) * sum(PINQ.map(([t, q]) => pinball(v, q, t)))), L = PINQ.map(([t, q]) => pinball(y, q, t)), C = (2 / 3) * sum(L);
  tr.push(ln(ys, cr, a, K.navy, 'approximate CRPS: (2/3) x sum of the three losses', { line: { width: 1.6, dash: 'dash' } }), mk([y], [C], a, K.navy, '', { marker: { size: 9, color: K.navy } }));
  plot('wp-chart', tr, G.lay);
  steps('wp-ws', PINQ.map(([t, q], j) => T.wps1(f1(t), f1(y), f1(q), y >= q, f2(L[j]))).concat([T.wps2(f2(L[0]), f2(L[1]), f2(L[2]), f3(C)), T.wps3(y >= 6.5 && y <= 7.6)]));
  out('wp-out', [T.wp1(f1(y), f3(C), f1(PINQ[argmax(L)][0]))]);
  return { L, C };
}

// ================================================================= 9. MASE: the scale of the seasonal naive
function blkWm() {
  const m = $('wm-m').value, F = D.fan_load, ctx = F.ctx, sc = F.scale[m], y = F.y, med = F.bolt[2], nv = F.naive[1];
  const dd = ctx.slice(+m).map((v, i) => Math.abs(v - ctx[i])), td = dd.map((_, i) => hourIso(F.t0, i + +m)), th = y.map((_, j) => hourIso(NB.fan_load.origin + ' 00:00', j));
  const eb = y.map((v, j) => Math.abs(v - med[j]) / sc), en = y.map((v, j) => Math.abs(v - nv[j]) / sc);
  const G = grid('wm-chart', [{ r: 0, c: 0, t: `denominator: |y_t - y_(t-${m})| over the 2048-hour context`, yt: 'GW', x: DATE }, { r: 0, c: 1, t: 'scaled errors |y - forecast| / scale, 48 hours', x: DATE }],
    { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax;
  plot('wm-chart', [ln(td, dd, a, K.navy, '', { line: { width: 0.6 } }), hline(a, [td[0], td[td.length - 1]], sc, K.red, `scale = mean = ${sc.toFixed(3)} GW`, { line: { width: 2, dash: 'dash' } }),
    ln(th, eb, b, K.blue, 'Chronos-Bolt small'), ln(th, en, b, K.amber, 'seasonal naive (one week earlier)'),
    hline(b, [th[0], th[47]], mean(eb), K.blue, '', { line: { width: 1, dash: 'dash' } }), hline(b, [th[0], th[47]], mean(en), K.amber, '', { line: { width: 1, dash: 'dash' } })], G.lay);
  const A = +$('wm-a').value, Bv = +$('wm-b').value; set('wm-a', f2(A)); set('wm-b', f2(Bv));
  const mb = mean(y.map((v, j) => Math.abs(v - med[j]))), mn = mean(y.map((v, j) => Math.abs(v - nv[j])));
  steps('wm-ws', [T.wms1(f2(A), f2(Bv), f2(A / Bv)), T.wms2(m, f3(sc)), T.wms3(f3(mb), f3(sc), f2(mb / sc), f3(mn), f2(mn / sc))]);
  out('wm-out', [T.wm1(m, f2(mean(eb)), f2(mean(en)))]);
  return { mb: mean(eb), mn: mean(en), sc };
}

// ================================================================= 10. coverage and relative scores
function blkWc() {
  const k = $('wc-s').value, m = $('wc-m').value, S = BS[k], o = S.origins, cv = S.models[m].cov, run = cumsum(cv).map((v, i) => v / (i + 1));
  const R = results('').R, rr = SER.map(x => R[x][m].rel_wql), gm = Math.exp(mean(rr.map(Math.log))), am = mean(rr);
  const dx = k === 'load' ? o.map(x => x.slice(0, 10)) : o;
  const G = grid('wc-chart', [{ r: 0, c: 0, t: `${m}, ${SH[k]}: share of the steps inside the 80% interval`, x: DATE, y: { range: [-0.03, 1.05] } },
    { r: 0, c: 1, t: `${m}: WQL ratio to the seasonal naive by series`, y: { rangemode: 'tozero' }, x: { tickangle: 0 } }], { cols: 2, h: 400, legendRows: 2, mb: 46 });
  const [a, b] = G.ax;
  plot('wc-chart', [mk(dx, cv, a, MC[m], 'each origin', { marker: { size: 5, color: MC[m] } }), ln(dx, run, a, K.navy, 'running mean', { line: { width: 1.6 } }),
    hline(a, [dx[0], dx[dx.length - 1]], 0.8, K.red, 'target 0.8', { line: { width: 1, dash: 'dash' } }),
    bar(SER.map(SH2), rr, b, MC[m], '', { hovertemplate: '%{x}: %{y:.3f}<extra></extra>' }), hline(b, SER.map(SH2), gm, K.navy, `geometric mean ${gm.toFixed(3)}`, { line: { width: 2, dash: 'dash' } }),
    hline(b, SER.map(SH2), am, K.amber, `arithmetic mean ${am.toFixed(3)}`, { line: { width: 1.4, dash: 'dot' } })], G.lay);
  const kk = +$('wc-k').value, nn = +$('wc-n').value, r1 = +$('wc-r').value; set('wc-k', kk); set('wc-n', nn); set('wc-r', f2(r1));
  const ki = Math.min(kk, nn);
  steps('wc-ws', [T.wcs1(ki, nn, pc(ki / nn), ki / nn < 0.8 - 1e-9 ? 'narrow' : ki / nn > 0.8 + 1e-9 ? 'wide' : 'ok'), T.wcs2(f2(r1), f2(Math.sqrt(r1 * 0.5)), f3((r1 + 0.5) / 2))]);
  out('wc-out', [T.wc1(mfull(m), nm(k), pc(mean(cv), 1), pc(cv.filter(v => v < 0.8).length / cv.length), o.length), T.wc2(f2(gm), f2(am))]);
  return { cov: mean(cv), gm, am };
}

// ================================================================= 11. the seven series
const SROW = [['load', 0, 0, K.blue], ['ip', 0, 1, K.red], ['gdp', 1, 0, K.green], ['infl', 1, 1, K.orange], ['unemp', 2, 0, K.purple], ['eurron', 2, 1, K.teal], ['retail', 3, 0, K.amber]];
function seriesXY(k) {
  const S = D.series[k];
  if (k === 'load') return { x: S.y.map((_, i) => hourIso(S.t0, i)), y: S.y };
  if (k === 'eurron') return { x: S.d, y: S.y };
  return { x: S.y.map((_, i) => (k === 'gdp' ? isoQ(S.p0, i) : isoM(S.p0, i))), y: S.y };
}
function blkSr() {
  const v = $('sr-v').value, cells = [];
  SROW.forEach(([k, r, c]) => cells.push({ r, c, t: D.bench.label[k] + (k === 'load' ? ' (June 2026)' : '') + (v === 'sd' ? `: y_t - y_(t-${D.series.m[k]})` : v === 'd1' ? ': y_t - y_(t-1)' : ''), x: DATE }));
  const G = grid('sr-chart', cells, { cols: 2, h: 820, rowPx: 170, legend: false, gyPx: 52 });
  const tr = [], mabs = {};
  SROW.forEach(([k, , , col], j) => {
    const { x, y } = seriesXY(k), L = v === 'sd' ? D.series.m[k] : v === 'd1' ? 1 : 0;
    const yy = L ? y.slice(L).map((u, i) => u - y[i]) : y, xx = x.slice(L);
    mabs[k] = mean(yy.map(Math.abs));
    tr.push(ln(xx, yy, G.ax[j], col, '', { line: { width: k === 'load' || k === 'eurron' ? 0.8 : 1.1 } }));
  });
  plot('sr-chart', tr, G.lay);
  const N_ = NB.series, rv = D.series.retail;
  out('sr-out', [T.sr1(T.big(N_.gdp.n), T.big(N_.load.n), T.big(N_.eurron.n)),
    v === 'lev' ? T.sr2l : T.sr2(v === 'sd', SROW.map(([k]) => [nm(k), mabs[k], k]).map(([n, x, k]) => `${n} ${nf(x, k === 'eurron' ? 4 : 2)}`).join(T.sep)),
    T.sr3(f1(rv.slide_last[0]), f1(rv.today_last[0]), f1(rv.slide_last[1]), f1(rv.today_last[1]), f1(D.tokens.rev.slide), f1(D.tokens.rev.today), T.dmy(D.downloaded))]);
  return mabs;
}

// ================================================================= 12. load, 48 hours ahead
function fanTraces(x, q, a, col, name, lv) {   // q: rows of quantiles; lv: indices of [lo80, lo50, med, hi50, hi80]
  return [band(x, q[lv[0]], q[lv[4]], a, col, 0.18, `${name}: 80% interval`), band(x, q[lv[1]], q[lv[3]], a, col, 0.3, ''), ln(x, q[lv[2]], a, col, `${name}: median`, { line: { width: 1.6 } })];
}
function blkFl() {
  const H = +$('fl-H').value; set('fl-H', H);
  const F = D.fan_load, q = F.bolt, y = F.y, org = NB.fan_load.origin + ' 00:00';
  const th = y.map((_, j) => hourIso(org, j)), hx = range(-168, -1, 1).map(j => hourIso(org, j)), hist = F.ctx.slice(2048 - 168);
  const G = grid('fl-chart', [{ r: 0, c: 0, yt: 'GW', x: DATE }], { cols: 1, h: 400, legendRows: 3 });
  const a = G.ax[0], ins = range(0, H - 1, 1), outI = ins.filter(j => y[j] < q[0][j] || y[j] > q[4][j]);
  plot('fl-chart', [ln(hx, hist, a, K.navy, 'observed (context, last week)', { line: { width: 1 } }), ln(th, y, a, K.navy, 'actual', { line: { width: 1.6, dash: 'dash' } }),
    ln(th, F.naive[1], a, K.amber, 'seasonal naive (one week earlier)', { line: { width: 1.4 } }), ...fanTraces(th, q, a, K.blue, 'Chronos-Bolt small', [0, 1, 2, 3, 4]),
    mk(outI.map(j => th[j]), outI.map(j => y[j]), a, K.red, `hours 1-${H}: outcome outside the 80% band`, { marker: { size: 6, color: K.red } }),
    vline(a, th[H - 1], 3.4, 7.2, K.red, '', { line: { width: 1, dash: 'dot' } })], G.lay);
  const sc = F.scale['168'], mb = mean(ins.map(j => Math.abs(y[j] - q[2][j]))) / sc, mn = mean(ins.map(j => Math.abs(y[j] - F.naive[1][j]))) / sc;
  const k80 = ins.filter(j => y[j] >= q[0][j] && y[j] <= q[4][j]).length, k50 = ins.filter(j => y[j] >= q[1][j] && y[j] <= q[3][j]).length, ab = ins.filter(j => y[j] > q[2][j]).length;
  out('fl-out', [T.fl1(H, f2(mb), f2(mn), f3(sc)), T.fl2(k80, H, pc(k80 / H), k50, ab)]);
  return { mb, mn, cov: k80 / H, k80, ab };
}

// ================================================================= 13. worked example: scoring one hour
function blkWh() {
  const h = +$('wh-h').value; set('wh-h', h);
  const F = D.fan_load, q = F.bolt, y = F.y, org = NB.fan_load.origin + ' 00:00', th = y.map((_, j) => hourIso(org, j)), j = h - 1;
  const G = grid('wh-chart', [{ r: 0, c: 0, t: 'Chronos-Bolt small, 48 hours from 22 June 2026, 00:00', yt: 'GW', x: DATE }, { r: 0, c: 1, t: `pinball losses at hour ${h}`, yt: 'GW', xt: 'level', x: { type: 'category' } }],
    { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax, Ls = [0.1, 0.3, 0.5, 0.7, 0.9].map((t, i) => pinball(y[j], q[i][j], t));
  G.lay[b.ya.replace('y', 'yaxis')].range = [0, 1.25 * Math.max(...Ls, 0.01)];
  plot('wh-chart', [...fanTraces(th, q, a, K.blue, 'Chronos-Bolt small', [0, 1, 2, 3, 4]), ln(th, y, a, K.navy, 'actual', { line: { width: 1.4, dash: 'dash' } }),
    mk([th[j], th[j], th[j]], [q[0][j], q[2][j], q[4][j]], a, K.blue, 'quantiles 0.1, 0.5, 0.9 at the hour', { marker: { size: 8, symbol: 'line-ew-open', line: { width: 3, color: K.blue } } }),
    mk([th[j]], [y[j]], a, K.red, 'outcome at the hour', { marker: { size: 9, color: K.red } }),
    bar(['0.1', '0.3', '0.5', '0.7', '0.9'], Ls, b, [K.teal, `rgba(${RGB[K.teal]},0.4)`, K.blue, `rgba(${RGB[K.red]},0.4)`, K.red], '', { text: Ls.map(v => v.toFixed(3)), textposition: 'outside', hovertemplate: 'level %{x}: %{y:.4f}<extra></extra>' })], G.lay);
  const L3 = [0, 2, 4].map(i => Ls[i]), inb = y[j] >= q[0][j] && y[j] <= q[4][j];
  steps('wh-ws', [T.whs1(T.hm(th[j]), f4(q[0][j]), f4(q[2][j]), f4(q[4][j]), f4(y[j])), ...[0.1, 0.5, 0.9].map((t, i) => T.whs2(f1(t), y[j] >= q[2 * i][j], f4(y[j]), f4(q[2 * i][j]), f3(L3[i]))),
    T.whs3(inb)]);
  out('wh-out', [T.wh1(h, f3(L3[0]), f3(L3[1]), f3(L3[2]), inb)]);
  return { L3, inb };
}

// ================================================================= 14. industrial production, 12 months ahead
function blkFi() {
  const H = +$('fi-H').value; set('fi-H', H);
  const F = D.fan_ip, hx = F.hist.map((_, i) => isoM(F.m0, i)), fx = F.y.map((_, j) => isoM(NB.fan_ip.origin, j)), y = F.y;
  const G = grid('fi-chart', [{ r: 0, c: 0, yt: 'index, 2021 = 100', x: DATE }], { cols: 1, h: 400, legendRows: 3 });
  const a = G.ax[0], ins = range(0, H - 1, 1);
  plot('fi-chart', [ln(hx, F.hist, a, K.navy, 'observed (context, last 4 years)', { line: { width: 1 } }), ln(fx, y, a, K.navy, 'actual', { mode: 'lines+markers', marker: { size: 5 }, line: { width: 1.6, dash: 'dash' } }),
    ...fanTraces(fx, F.ets, a, K.green, 'ETS', [0, 2, 4, 6, 8]), ...fanTraces(fx, F.bolt, a, K.blue, 'Chronos-Bolt small', [0, 1, 2, 3, 4]),
    vline(a, fx[H - 1], 75, 118, K.red, '', { line: { width: 1, dash: 'dot' } })], G.lay);
  const ma = f => mean(ins.map(j => Math.abs(y[j] - f[j]))) / F.scale, cov = (lo, hi) => ins.filter(j => y[j] >= lo[j] && y[j] <= hi[j]).length / H;
  const r = { bolt: ma(F.bolt[2]), ets: ma(F.ets[4]), naive: ma(F.naive), cb: cov(F.bolt[0], F.bolt[4]), ce: cov(F.ets[0], F.ets[8]),
    wb: F.bolt[4][H - 1] - F.bolt[0][H - 1], we: F.ets[8][H - 1] - F.ets[0][H - 1] };
  out('fi-out', [T.fi1(H, T.my(fx[H - 1]), f2(r.bolt), f2(r.ets), f2(r.naive)), T.fi2(pc(r.cb), pc(r.ce), f1(r.wb), f1(r.we)), T.fi3(f1(y[11]), f1(F.today[11]), T.dmy(D.downloaded))]);
  return r;
}

// ================================================================= 15. EUR/RON, 20 days ahead
function blkFe() {
  const H = +$('fe-H').value; set('fe-H', H);
  const F = D.fan_eurron, d = F.d, hx = d.slice(0, 120), fx = d.slice(120), y = F.y;
  const rw = [0.1, 0.3, 0.5, 0.7, 0.9].map(t => fx.map((_, j) => F.last + F.sd * Math.sqrt(j + 1) * normPpf(t)));
  const G = grid('fe-chart', [{ r: 0, c: 0, yt: 'RON per EUR', x: DATE }], { cols: 1, h: 400, legendRows: 3 });
  const a = G.ax[0], ins = range(0, H - 1, 1);
  plot('fe-chart', [ln(hx, F.hist, a, K.navy, 'observed (context, last 120 days)', { line: { width: 1 } }), ln(fx, y, a, K.navy, 'actual', { line: { width: 1.6, dash: 'dash' } }),
    ...fanTraces(fx, rw, a, K.amber, 'random walk', [0, 1, 2, 3, 4]), ...fanTraces(fx, F.bolt, a, K.blue, 'Chronos-Bolt small', [0, 1, 2, 3, 4]),
    vline(a, fx[H - 1], 5.09, 5.31, K.red, '', { line: { width: 1, dash: 'dot' } })], G.lay);
  const j = H - 1, r = { med: F.bolt[2][j], wb: F.bolt[4][j] - F.bolt[0][j], wr: rw[4][j] - rw[0][j], mb: mean(ins.map(i => Math.abs(y[i] - F.bolt[2][i]))), mr: mean(ins.map(i => Math.abs(y[i] - F.last))) };
  out('fe-out', [T.fe1(H, T.dmy(fx[j]), f4(r.med), f4(F.last)), T.fe2(f3(r.wb), f3(r.wr)), T.fe3(H, f4(r.mb), f4(r.mr))]);
  return r;
}

// ================================================================= 16. results table (extra chart: the scores as a heat map)
const SCORE = { rel_wql: 'WQL / seasonal naive', rel_mase: 'MASE / seasonal naive', cov: 'coverage of the 80% interval' };
function blkRt() {
  const s = $('rt-k').value, { R, gm } = results(''), ms = s === 'cov' ? MODELS : RIVALS;
  const z = SER.map(k => ms.map(m => R[k][m][s]));
  const G = grid('rt-chart', [{ r: 0, c: 0, t: SCORE[s], y: { autorange: 'reversed' } }], { cols: 1, h: 380, legend: false, ml: 110 });
  const a = G.ax[0], mid = s === 'cov' ? 0.8 : 1, dev = Math.max(...z.flat().map(v => (s === 'cov' ? Math.abs(v - mid) : Math.abs(Math.log(v)))));
  plot('rt-chart', [{ type: 'heatmap', x: ms.map(lab), y: SER.map(k => SH[k]), z: s === 'cov' ? z : z.map(r => r.map(Math.log)), text: z.map(r => r.map(v => v.toFixed(2))), texttemplate: '%{text}',
    colorscale: [[0, K.blue], [0.5, '#FFFFFF'], [1, K.red]], zmin: -dev, zmax: dev, ...(s === 'cov' ? { zmin: mid - dev, zmax: mid + dev, colorscale: [[0, K.red], [0.5, '#FFFFFF'], [1, K.blue]] } : {}),
    showscale: false, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{y}, %{x}: %{text}<extra></extra>' }], G.lay);
  const v = $('rt-v').value, col = v === 'rel' ? 'rel_wql' : 'wql', dg = v === 'rel' ? 2 : 4;
  const rows = SER.map(k => { const bm = best(R[k], RIVALS, x => x[col]); return [T.sname[k], txt(R[k]._n), ...RIVALS.map(m => txt(nf(R[k][m][col], dg), m === bm))]; });
  if (v === 'rel') { const bg = best(gm, RIVALS, x => x.rel_wql); rows.push(hiRow([T.L.gm, txt(''), ...RIVALS.map(m => txt(f2(gm[m].rel_wql), m === bg))], true)); }
  table('rt-tb', [T.L.series, H_(T.L.origins), ...RIVALS.map(m => H_(lab(m)))], rows,
    { note: v === 'rel' ? T.rtNote : T.rtNoteAbs(SER.map(k => `${SH[k]} ${nf(R[k]['Seasonal naive'].wql, 4)}`).join(T.sep)) });
  const srt = RIVALS.map(m => [lab(m), gm[m][s === 'cov' ? 'cov' : s]]).sort((p, q_) => p[1] - q_[1]);
  out('rt-out', [T.rt1(s, fmtList(srt)), T.rt2(f2(R.eurron['Chronos-2'].rel_wql), f2(R.eurron['Chronos-2'].rel_mase))]);
  return { R, gm };
}

// ================================================================= 17. MASE and WQL by series and model
function blkBm() {
  const from = $('bm-f').value, { R, gm, ks, n } = results(from);
  const G = grid('bm-chart', [{ r: 0, c: 0, yt: 'MASE ratio', y: LOG }, { r: 1, c: 0, yt: 'WQL ratio', y: LOG }], { cols: 1, h: 520, legendRows: 2, rowPx: 230 });
  const [a, b] = G.ax;
  plot('bm-chart', [...groupBars(a, R, ks, RIVALS, 'rel_mase'), ...groupBars(b, R, ks, RIVALS, 'rel_wql'), hline(a, ks.map(k => SH[k]), 1, K.navy, '', { line: { width: 1, dash: 'dash' } }),
    hline(b, ks.map(k => SH[k]), 1, K.navy, '', { line: { width: 1, dash: 'dash' } })], Object.assign(G.lay, { barmode: 'group', bargap: 0.18 }));
  const srt = RIVALS.map(m => [lab(m), gm[m].rel_wql]).sort((p, q_) => p[1] - q_[1]), bw = best(gm, RIVALS, x => x.rel_wql);
  out('bm-out', [T.bm1(fromLab(from), n, ks.length), T.bm2(fmtList(srt)), T.bm3(lab(bw), gm[bw].wins, ks.length),
    R.eurron ? T.bm4(f2(R.eurron['Chronos-2'].rel_wql), f2(R.eurron['Chronos-2'].rel_mase)) : '']);
  return { gm, wins: gm[bw].wins, bw };
}

// ================================================================= 18. coverage of the 80% intervals
function blkCv() {
  const from = $('cv-f').value, { R, gm, ks } = results(from);
  const G = grid('cv-chart', [{ r: 0, c: 0, yt: 'share of outcomes inside', y: { range: [0, 1.05] } }], { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0];
  plot('cv-chart', [...groupBars(a, R, ks, MODELS, 'cov'), hline(a, ks.map(k => SH[k]), 0.8, K.navy, '', { line: { width: 1, dash: 'dash' } })], Object.assign(G.lay, { barmode: 'group', bargap: 0.18 }));
  const rr = R.retail ? MODELS.map(m => [mname(m), R.retail[m].cov]).sort((p, q_) => p[1] - q_[1]) : null;
  out('cv-out', [T.cv1(fromLab(from), MODELS.map(m => `${mname(m)} ${pc(gm[m].cov)}`).join(T.sep)),
    rr ? T.cv2(rr[0][0], pc(rr[0][1], 1), rr[1][0], pc(rr[1][1], 1), rr[5][0], pc(rr[5][1], 1)) : '', T.cv3(pc(gm['Chronos-2'].cov), f2(gm['Chronos-2'].rel_wql))]);
  return { gm, rr };
}

// ================================================================= 19. error by horizon
function blkHz() {
  const k = $('hz-s').value, w = $('hz-w').value, S = BS[k], n = S.origins.length, h = D.bench.h[k];
  const I = range(0, n - 1, 1).filter(i => (w === 'all' ? true : w === 'a' ? i < Math.floor(n / 2) : i >= Math.floor(n / 2)));
  const steps_ = range(1, h, 1), mae = {};
  MODELS.forEach(m => { mae[m] = steps_.map((_, j) => mean(I.map(i => S.models[m].err[i][j]))); });
  const G = grid('hz-chart', [{ r: 0, c: 0, xt: k === 'load' ? 'hours ahead' : k === 'eurron' ? 'days ahead' : k === 'gdp' ? 'quarters ahead' : 'months ahead', yt: 'mean absolute error' }],
    { cols: 1, h: 360, legendRows: 2 });
  const a = G.ax[0];
  plot('hz-chart', MODELS.map(m => ln(steps_, mae[m], a, MC[m], m, { line: { width: 1.5 } })), G.lay);
  const j2 = Math.ceil(h / 2) - 1, P = (m, j) => nf(mae[m][j], k === 'eurron' ? 4 : 3);
  const worst = j => MODELS.every(m => mae.ETS[j] >= mae[m][j]), etsWorst = steps_.find((_, j) => steps_.slice(j).every((__, i) => worst(j + i)));
  out('hz-out', [T.hz1(nm(k), I.length, w, S.origins[I[0]].slice(0, 10), S.origins[I[I.length - 1]].slice(0, 10)),
    T.hz2(1, P('ARIMA', 0), P('Chronos-2', 0), P('Seasonal naive', 0)), T.hz2(j2 + 1, P('ARIMA', j2), P('Chronos-2', j2), P('Seasonal naive', j2)),
    T.hz2(h, P('ARIMA', h - 1), P('Chronos-2', h - 1), P('Seasonal naive', h - 1)), etsWorst ? T.hz3(etsWorst) : T.hz3n]);
  return { mae, etsWorst };
}

// ================================================================= 20. how much context is needed
function blkCx() {
  const i = +$('cx-c').value, rel = $('cx-r').value === 'rel', C = NB.context.contexts, nv = NB.context.naive, M = NB.context.mase; set('cx-c', `${C[i]} (${(C[i] / 24).toFixed(0)} ${T.days})`);
  const f = v => (rel ? v / nv : v), xt = C.map(c => `${c}<br>(${(c / 24).toFixed(0)} days)`);
  const G = grid('cx-chart', [{ r: 0, c: 0, xt: 'context length (hours)', yt: rel ? 'MASE / seasonal naive' : 'MASE', x: { type: 'log', tickvals: C, ticktext: xt } }], { cols: 1, h: 380, legendRows: 2, mb: 50 });
  const a = G.ax[0], tr = [];
  FMS.forEach(m => tr.push(ln(C, M[m].map(f), a, MC[m], m, { mode: 'lines+markers', marker: { size: 7 } }), mk([C[i]], [f(M[m][i])], a, MC[m], '', { marker: { size: 14, color: MC[m], symbol: 'circle-open', line: { width: 2.5 } } })));
  tr.push(hline(a, [C[0], C[5]], f(nv), K.amber, 'seasonal naive', { line: { width: 1.2, dash: 'dash' } }), vline(a, C[i], rel ? 0.4 : 0.5, rel ? 1.85 : 2.1, K.navy, '', { line: { width: 1, dash: 'dot' } }));
  plot('cx-chart', tr, G.lay);
  const bm = FMS.reduce((b, m) => (M[m][i] < M[b][i] ? m : b), FMS[0]);
  out('cx-out', [T.cx1(C[i], f1(C[i] / 24), C[i] >= 168), T.cx2(FMS.map(m => `${lab(m)} ${f2(f(M[m][i]))}`).join(T.sep), f2(f(nv)), rel), T.cx3(lab(bm), FMS.filter(m => M[m][i] < nv).length)]);
  return { i, best: bm, beat: FMS.filter(m => M[m][i] < nv).length };
}

// ================================================================= 21. size and accuracy
function blkSz() {
  const from = $('sz-f').value, met = $('sz-m').value, { gm } = results(from), Sz = NB.size;
  const G = grid('sz-chart', [{ r: 0, c: 0, xt: 'parameters (millions, log scale)', yt: `geometric mean of ${met === 'rel_wql' ? 'WQL' : 'MASE'} / naive`, x: { type: 'log', range: [Math.log10(5), Math.log10(200)] } }],
    { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0], xs = [5, 200];
  plot('sz-chart', [...FMS.map(m => mk([Sz[m].params / 1e6], [gm[m][met]], a, MC[m], m, { marker: { size: 13, color: MC[m] } })),
    hline(a, xs, gm.ETS[met], K.green, 'ETS (a few parameters per series)', { line: { width: 1.2, dash: 'dash' } }), hline(a, xs, gm.ARIMA[met], K.purple, 'ARIMA (a few parameters per series)', { line: { width: 1.2, dash: 'dash' } }),
    hline(a, xs, 1, K.amber, 'seasonal naive', { line: { width: 1.2, dash: 'dot' } })], G.lay);
  const sec = NB.infos.load.seconds, ms_ = m => f0(1000 * sec[m]);
  out('sz-out', [T.sz1(fromLab(from), met === 'rel_wql' ? 'WQL' : 'MASE', FMS.map(m => `${lab(m)} (${T.mil(f0(Sz[m].params / 1e6))}) ${f2(gm[m][met])}`).join(T.sep), f2(gm.ARIMA[met]), f2(gm.ETS[met])),
    T.sz2(ms_('Chronos-Bolt tiny'), ms_('Chronos-Bolt small'), ms_('Chronos-2'), ms_('ARIMA'), ms_('ETS'))]);
  return { gm };
}

// ================================================================= 22. contamination check: before and after the release
function prepost(fm, cut) {
  const res = [];
  SER.forEach(k => { const o = BS[k].origins, bef = o.map(x => x < cut), aft = o.map(x => x >= cut);
    if (!bef.some(Boolean) || !aft.some(Boolean)) return;
    const b = summ(k, bef), a = summ(k, aft); res.push({ k, ets: [b.ETS.rel_wql, a.ETS.rel_wql], fm: [b[fm].rel_wql, a[fm].rel_wql], n: [b._n, a._n] }); });
  return res;
}
function blkPp() {
  const sh = +$('pp-c').value, F = ['Chronos-Bolt small', 'Chronos-2'], cuts = F.map(m => shiftM(D.bench.release[m], sh));
  const P = F.map((m, i) => prepost(m, cuts[i])), n0 = Math.max(1, P[0].length), n1 = Math.max(1, P[1].length);
  const G = grid('pp-chart', [{ r: 0, c: 0, cs: n0, t: `Chronos-Bolt small: cut at ${cuts[0]}${sh ? ' (placebo)' : ' (release of the weights)'}`, yt: 'WQL ratio', x: { tickangle: 0 } },
    { r: 0, c: n0, cs: n1, t: `Chronos-2: ${cuts[1]}`, x: { tickangle: 0 } }], { cols: n0 + n1, h: 420, legendRows: 3, mb: 46, gx: 0.05 });
  const tr = [];
  P.forEach((rows, i) => { const a = G.ax[i], xs = rows.map(r => SH2(r.k));
    [['ETS', 'ets', K.green], [F[i], 'fm', MC[F[i]]]].forEach(([m, key, col]) => [0, 1].forEach(p => tr.push(bar(xs, rows.map(r => r[key][p]), a, p ? col : 'rgba(0,0,0,0)',
      i === 0 || m !== 'ETS' ? `${m}, origins ${p ? 'after' : 'before'} the cut` : '', { legendgroup: m + p, marker: p ? { color: col } : { color: 'rgba(0,0,0,0)', line: { color: col, width: 1 }, pattern: { shape: '/', fgcolor: col, size: 6 } },
        hovertemplate: `${m}, ${p ? 'after' : 'before'}: %{y:.3f}<extra></extra>` }))));
    tr.push(hline(a, xs, 1, K.navy, '', { line: { width: 1, dash: 'dash' } })); });
  plot('pp-chart', tr, Object.assign(G.lay, { barmode: 'group', bargap: 0.2 }));
  const g = (i, k) => P[i].find(r => r.k === k), c2e = g(1, 'eurron'), c2l = g(1, 'load'), bs = g(0, 'retail');
  out('pp-out', [T.pp0(sh, cuts[0], cuts[1]), c2e ? T.pp1(f2(c2e.fm[0]), c2e.n[0], f2(c2e.fm[1]), c2e.n[1]) : T.ppn(nm('eurron')), c2l ? T.pp2(f2(c2l.fm[0]), c2l.n[0], f2(c2l.fm[1]), c2l.n[1]) : T.ppn(nm('load')),
    bs ? T.pp3(f2(bs.fm[0]), f2(bs.fm[1]), f2(bs.ets[0]), f2(bs.ets[1]), bs.n[0], bs.n[1]) : T.ppn(nm('retail'))]);
  return { P };
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
const BLK = [['wx-att', 'wa', blkWa, 'wa-c', range(0, 3, 0.25), 500], ['wx-tok', 'wt', blkWt, 'wt-b', null, 1100], ['tokens', 'tk', blkTk, 'tk-n', range(12, 96, 12), 700],
  ['patching', 'pa', blkPa, 'pa-P', range(4, 48, 4), 600], ['fam-table', 'ft', blkFt, 'ft-c', range(256, 4096, 256), 400], ['wx-wql', 'ww', blkWw, 'ww-s', null, 1000],
  ['scores', 'sc', blkSc, 'sc-y', range(-2, 2, 0.25), 400], ['wx-pin', 'wp', blkWp, 'wp-y', range(6, 8.4, 0.1), 300], ['wx-mase', 'wm', blkWm, 'wm-m', null, 1200],
  ['wx-cov', 'wc', blkWc, 'wc-m', null, 1000], ['series', 'sr', blkSr, 'sr-v', null, 1600], ['fan-load', 'fl', blkFl, 'fl-H', range(6, 48, 6), 500],
  ['wx-hour', 'wh', blkWh, 'wh-h', range(1, 48, 1), 250], ['fan-ip', 'fi', blkFi, 'fi-H', range(1, 12, 1), 500], ['fan-eurron', 'fe', blkFe, 'fe-H', range(1, 20, 1), 350],
  ['res-table', 'rt', blkRt, 'rt-k', null, 1300], ['benchmark', 'bm', blkBm, 'bm-f', null, 1300], ['coverage', 'cv', blkCv, 'cv-f', null, 1300],
  ['horizon', 'hz', blkHz, 'hz-w', null, 1200], ['context', 'cx', blkCx, 'cx-c', range(0, 5, 1), 800], ['size', 'sz', blkSz, 'sz-f', null, 1200], ['prepost', 'pp', blkPp, 'pp-c', null, 1300]];
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
window.CH11 = { pinball, crpsN, summ, results, attn, quantise, prepost, hourIso, shiftM, BLK, init };
if (!window.CH11_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
