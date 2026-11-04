// Chapter 13 interactive page (bubbles and LPPL models): one block per chart slide, plus the worked examples, the result tables
// and the extra charts of the lecture, computed in the browser from ch13_data.js (Quantlets/Ch_13/export_interactive.py): the
// daily closes of the six episodes, the draws of the simulations of the chapter, the weekly log prices of the PSY samples with
// the Monte Carlo critical values of the chapter, the LPPL fits of the chapter's code, its cost landscape, its indicator series
// and the daily closes of the evaluation. The functions below follow generate_all_charts.py line by line.
(function () {
'use strict';
const D = window.CH13_DATA;
const { sum, mean, nf, $, AX, layout, plot, Player, range, predict, exitTicket } = window.TSAI;
const out = (id, items) => window.TSAI.out(id, items);
const T = window.CH13_TEXT;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of tsa_style.py (series colours of the chapter)
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', orange: '#E67E22', purple: '#8E44AD', teal: '#17A2B8', crimson: '#DC3545', navy: '#1F2A44' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.amber]: '181,133,63', [K.orange]: '230,126,34', [K.purple]: '142,68,173',
  [K.teal]: '23,162,184', [K.navy]: '31,42,68', [K.crimson]: '220,53,69' };
const EPK = ['sp500', 'ndx', 'bet', 'ssec', 'btc17', 'btc21'];
const ECOL = { sp500: K.blue, ndx: K.teal, bet: K.orange, ssec: K.purple, btc17: K.amber, btc21: K.amber };
const ELAB = { sp500: 'S&P 500, 2000', ndx: 'Nasdaq 100, 2000', bet: 'BET, 2007', ssec: 'Shanghai Composite, 2015', btc17: 'Bitcoin, 2017', btc21: 'Bitcoin, 2021' };
const PCOL = { ndx: K.teal, btc: K.amber, bet: K.red, ssec: K.purple };

// ---------------------------------------------------------------- small helpers
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
const has = id => !!$(id);
const cache = {};
const memo = (k, f) => (k in cache ? cache[k] : (cache[k] = f()));
const f0 = x => nf(x, 0), f1 = x => nf(x, 1), f2 = x => nf(x, 2), f3 = x => nf(x, 3), f4 = x => nf(x, 4);
const pc = (x, d) => nf(100 * x, d === undefined ? 0 : d) + '%';
const argmin = a => a.reduce((b, v, i) => (v < a[b] ? i : b), 0);
const argmax = a => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const median = a => { const b = [...a].sort((x, y) => x - y), n = b.length; return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2; };
const qtl = (a, p) => { const b = [...a].sort((x, y) => x - y), h = (b.length - 1) * p, k = Math.floor(h); return b[k] + (h - k) * ((b[k + 1] ?? b[k]) - b[k]); };
const sig = (x, d) => (Math.abs(x) >= 1e-3 || x === 0 ? nf(x, d) : nf(x, d + 2));
const big = x => T.big(x);
const sgn = n => (n > 0 ? '+' + n : n < 0 ? '−' + (-n) : '0');
const trim = (x, d) => nf(+x.toFixed(d), (+x.toFixed(d)).toString().split('.')[1]?.length || 0);
// dates: ISO strings 'YYYY-MM-DD'; decimal years as in yrs()/todate() of the chapter
const DAY = 864e5;
const tms = s => Date.parse(s.slice(0, 10) + 'T00:00:00Z');
const isoOf = t => new Date(t).toISOString().slice(0, 10);
const addDays = (s, k) => isoOf(tms(s) + k * DAY);
const dDays = (a, b) => Math.round((tms(b) - tms(a)) / DAY);
const yrs = s => { const y = +s.slice(0, 4), doy = Math.round((tms(s) - Date.UTC(y, 0, 1)) / DAY) + 1; return y + (doy - 1) / 365.25; };
const tdMs = x => { const y = Math.floor(x); return Date.UTC(y, 0, 1) + (x - y) * 365.25 * DAY; };
const tdIso = x => isoOf(Math.floor(tdMs(x) / DAY) * DAY);
const tdPl = x => new Date(tdMs(x)).toISOString().slice(0, 19).replace('T', ' ');
const expand = S => S.off.map(o => addDays(S.d0, o));
const dmy = s => T.dmy(s), my = s => T.my(s);

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
    return { xa: 'x' + n, ya: 'y' + n, xd: 'x' + n + ' domain', yd: 'y' + n + ' domain', k: k + 1, dom: [Math.max(0, y0), y1], x1: Math.min(1, x1) };
  });
  el.style.height = H + 'px';
  return { lay, ax, narrow, n: cells.length };
}
// a second y axis on the right of panel a (number k > number of panels)
function twin(G, a, k, title, extra) {
  G.lay['yaxis' + k] = AX('', Object.assign({ overlaying: a.ya, side: 'right', anchor: a.xa, showgrid: false, zeroline: false, automargin: false,
    title: { text: title || '', font: { size: 11, color: K.navy }, standoff: 2 } }, extra || {}));
  return { xa: a.xa, ya: 'y' + k, xd: a.xd, yd: 'y' + k + ' domain' };
}
const ln = (x, y, a, color, name, o) => { const r = Object.assign({ x, y, mode: 'lines', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.4g}<extra></extra>' }, o || {}); r.line = Object.assign({ color, width: 1.4 }, (o && o.line) || {}); return r; };
const mk = (x, y, a, color, name, o) => Object.assign({ x, y, mode: 'markers', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, marker: { color, size: 7 }, hovertemplate: '%{x}: %{y:.4g}<extra></extra>' }, o || {});
const bar = (x, y, a, color, name, o) => Object.assign({ x, y, type: 'bar', marker: { color }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.4g}<extra></extra>' }, o || {});
const hl = (a, xs, y, color, name, o) => ln(xs, xs.map(() => y), a, color, name, Object.assign({ hoverinfo: 'skip' }, o || {}));
const vl = (a, x, y0, y1, color, name, o) => ln([x, x], [y0, y1], a, color, name, Object.assign({ hoverinfo: 'skip' }, o || {}));
// shaded spans drawn as traces (rectangles between y0 and y1), so that a change of the spans changes the data of the panel
function spans(a, list, y0, y1, col, al, name) {
  const xs = [], ys = [];
  list.forEach(([u, v]) => { xs.push(u, v, v, u, u, null); ys.push(y0, y0, y1, y1, y0, null); });
  if (!list.length) { xs.push(null); ys.push(null); }
  return { x: xs, y: ys, fill: 'toself', fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, mode: 'lines', hoverinfo: 'skip', name: name || '', showlegend: !!name, xaxis: a.xa, yaxis: a.ya };
}
const vspan = (a, x0, x1, col, al) => ({ type: 'rect', xref: a.xa, yref: a.yd, x0, x1, y0: 0, y1: 1, fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, layer: 'below' });
const vline = (a, x, col, dash, w) => ({ type: 'line', xref: a.xa, yref: a.yd, x0: x, x1: x, y0: 0, y1: 1, line: { color: col, width: w || 1.2, dash: dash || 'dash' } });
const legBox = (name, col, al) => ({ x: [null], y: [null], name, mode: 'markers', marker: { symbol: 'square', size: 12, color: `rgba(${RGB[col]},${al})` }, hoverinfo: 'skip' });
const legLine = (name, col, dash) => ({ x: [null], y: [null], name, mode: 'lines', line: { color: col, dash: dash || 'solid', width: 1.4 }, hoverinfo: 'skip' });
const LOG = { type: 'log' };
// log axis with ticks at 1, 2, 5 x 10^k (denser when the range is narrow), labelled 1k, 2k, ...
function logRange(lo, hi) {
  const e0 = Math.floor(Math.log10(lo)), e1 = Math.ceil(Math.log10(hi)), pick = ms => { const v = []; for (let e = e0; e <= e1; e++) ms.forEach(m => { const x = +(m * 10 ** e).toPrecision(6); if (x >= lo && x <= hi) v.push(x); }); return v; };
  let v = pick([1, 2, 5]); if (v.length < 3) v = pick([1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]); if (v.length > 7) v = v.filter((_, i) => i % 2 === 0);
  const fmt = x => (x >= 1000 ? +(x / 1000).toPrecision(4) + 'k' : String(x));
  return { type: 'log', range: [Math.log10(lo), Math.log10(hi)], tickvals: v, ticktext: v.map(fmt) };
}
const yaxOf = (G, a) => G.lay[a.ya.replace('y', 'yaxis')];
// legend text wrapped on narrow screens
const lw = (G, t, n) => { if (!G.narrow) return t; const w = t.split(' '), Ls = ['']; w.forEach(x => { const c = Ls[Ls.length - 1]; if ((c + ' ' + x).length > (n || 40) && c) Ls.push(x); else Ls[Ls.length - 1] = (c + ' ' + x).trim(); }); return Ls.join('<br>'); };

// ---------------------------------------------------------------- tables and worked steps (HTML)
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}>${v.v}</${tag}>`; };
  el.innerHTML = (o && o.cap ? `<p class="c13cap">${o.cap}</p>` : '') + '<table><thead><tr>' + head.map(c => cell(c, 'th')).join('') + '</tr></thead><tbody>'
    + rows.map(r => '<tr' + (r.hi ? ' class="hi"' : '') + '>' + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c13cap">${o.note}</p>` : '');
}
const R_ = v => ({ v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };

// ================================================================= 1. episodes (fig_episodes)
const HMON = { 3: 91, 6: 182, 9: 273, 12: 365, 18: 540 };          // months after the peak -> calendar days (12 months: 365 days, the lecture)
function epData(k) {
  return memo('ep' + k, () => {
    const E = D.ep[k], d = expand(E), p = E.p;
    let lo = -1, pk = -1;
    d.forEach((s, i) => {
      if (s >= E.low[0] && s <= E.low[1] && (lo < 0 || p[i] < p[lo])) lo = i;
      if (s >= E.peak[0] && s <= E.peak[1] && (pk < 0 || p[i] > p[pk])) pk = i;
    });
    const lim = addDays(d[pk], -30); let i2 = 0;
    d.forEach((s, i) => { if (s <= lim) i2 = i; });
    return { d, p, lo, pk, i2 };
  });
}
function epStats(k, days, atEnd) {
  const E = epData(k), { d, p, lo, pk } = E, end = addDays(d[pk], days);
  let tr = pk, last = pk;
  for (let i = pk; i < d.length && d[i] <= end; i++) { if (p[i] < p[tr]) tr = i; last = i; }
  const yrs_ = dDays(d[lo], d[pk]) / 365.25, ix = atEnd ? last : tr;
  return { low: d[lo], peak: d[pk], pLow: p[lo], pPeak: p[pk], runup: p[pk] / p[lo] - 1, years: yrs_, growth: Math.log(p[pk] / p[lo]) / yrs_,
    fall: p[ix] / p[pk] - 1, trough: d[ix], end, i2: E.i2 };
}
function blkEp() {
  const mo = +$('ep-h').value, days = HMON[mo];
  const cells = EPK.map((k, i) => ({ r: Math.floor(i / 3), c: i % 3, t: ELAB[k] }));
  const G = grid('ep-chart', cells, { cols: 3, h: 470, gx: 0.06, gyPx: 74, legendRows: 2, rowPx: 190, ml: 46 });
  const tr = [], S = {};
  EPK.forEach((k, i) => {
    const a = G.ax[i], E = epData(k), s = epStats(k, days), lo = E.lo, x0 = addDays(E.d[lo], -180), x1 = addDays(E.d[E.pk], 540);
    const idx = E.d.map((v, j) => j).filter(j => E.d[j] >= x0 && E.d[j] <= x1);
    S[k] = s;
    const ps = idx.map(j => E.p[j]);
    Object.assign(yaxOf(G, a), logRange(Math.min(...ps) * 0.93, Math.max(...ps) * 1.07));
    tr.push(ln(idx.map(j => E.d[j]), ps, a, ECOL[k], '', { line: { width: 1.1 } }));
    tr.push(ln([s.peak, s.end], [s.pPeak * (1 + s.fall), s.pPeak * (1 + s.fall)], a, K.navy, i ? '' : `lowest close within ${mo} months after the peak`, { line: { width: 1.2, dash: 'dot' }, hoverinfo: 'skip' }));
    tr.push(mk([s.trough], [s.pPeak * (1 + s.fall)], a, K.navy, '', { marker: { size: 7, color: K.navy, symbol: 'triangle-down' }, hovertemplate: `${k}: fall ${(100 * s.fall).toFixed(0)}%<extra></extra>` }));
    G.lay.shapes.push(vspan(a, s.low, s.peak, ECOL[k], 0.12), vline(a, s.peak, K.red, 'dash', 1));
  });
  tr.push(legLine('peak', K.red, 'dash'), legBox('run-up: from the low to the peak', K.blue, 0.2));
  plot('ep-chart', tr, G.lay);
  const fl = EPK.map(k => S[k].fall);
  out('ep-out', [T.ep1(pc(Math.min(...EPK.map(k => S[k].runup))), pc(Math.max(...EPK.map(k => S[k].runup)))),
    T.ep2(mo, pc(-Math.max(...fl)), pc(-Math.min(...fl)), EPK.map(k => `${ELAB[k]} ${pc(-S[k].fall)}`).join(T.semi))]);
  return S;
}

// ================================================================= EXTRA. table: interpreting the six episodes
function blkTe() {
  const mo = +$('te-h').value, days = HMON[mo], atEnd = $('te-f').value === 'end';
  const S = Object.fromEntries(EPK.map(k => [k, epStats(k, days, atEnd)]));
  const G = grid('te-chart', [{ r: 0, c: 0, xt: 'growth per year: ln(P_peak / P_low) / years', yt: `fall ${mo} months after the peak`, y: { tickformat: '.0%', range: [-1, 0.1] }, x: { tickformat: '.0%' } }], { cols: 1, h: 360, legendRows: 2 });
  const a = G.ax[0];
  plot('te-chart', EPK.map(k => mk([S[k].growth], [S[k].fall], a, ECOL[k], ELAB[k], { mode: 'markers+text', text: [ELAB[k]], textposition: k === 'btc21' ? 'bottom left' : k === 'bet' ? 'bottom center' : 'top center', textfont: { size: 10, color: ECOL[k] }, marker: { size: 10, color: ECOL[k], symbol: k === 'btc21' ? 'diamond' : 'circle' }, showlegend: false }))
    .concat([hl(a, [0, 3.3], 0, K.navy, '', { line: { width: 1, dash: 'dot' } })]), G.lay);
  table('te-tb', T.teHead(mo, atEnd), EPK.map(k => [T.epName(k), dmy(S[k].low), dmy(S[k].peak), R_(pc(S[k].runup)), R_(pc(S[k].growth)), R_(pc(-S[k].fall)), dmy(S[k].trough)]), { note: T.teNote(atEnd) });
  const b = S.btc17, sp = S.sp500;
  out('te-out', [T.te1(f1(b.pPeak / b.pLow), f2(b.years), pc(sp.runup)), T.te2(mo, atEnd, pc(-Math.max(...EPK.map(k => S[k].fall))), pc(-Math.min(...EPK.map(k => S[k].fall))))]);
  return S;
}

// ================================================================= 2. rational bubble and AR(1) roots (fig_rational)
function bwPath(r, pi) {
  const e = D.sim.bwe, u = D.sim.bwu, b = [1.0];
  for (let t = 1; t < 400; t++) { const eps = 0.3 * e[t - 1]; b.push(u[t - 1] < pi ? (1 + r) / pi * b[t - 1] + eps : eps); }
  return b;
}
const fundPath = () => memo('fund', () => { const f = []; let s = 0; D.sim.fz.forEach(z => { s += 0.3 * z; f.push(20 + s); }); return f; });
function ar1(rho) { const e = D.sim.ar, y = [0]; for (let t = 1; t < 200; t++) y.push(rho * y[t - 1] + e[t]); return y; }
function ratStats(r, pi) {
  const b = bwPath(r, pi), F = fundPath(), P = F.map((v, i) => v + b[i]);
  let bursts = 0; for (let t = 1; t < 400; t++) if (Math.abs(b[t]) < 1.5 && b[t - 1] > 4) bursts++;
  const rho = 1 + r, A = { 0.9: ar1(0.9), 1: ar1(1.0), x: ar1(rho) };
  const neg = P.findIndex(v => v < 0);
  return { b, F, P, bursts, max: Math.max(...b), A, rho, end: { 0.9: A[0.9][199], 1: A[1][199], x: A.x[199] }, g100: rho ** 100, dbl: Math.log(2) / Math.log(rho), neg, minP: Math.min(...P), minB: Math.min(...b.slice(0, 120)) };
}
function blkRa() {
  const r = +$('ra-r').value, pi = +$('ra-p').value; set('ra-r', pc(r, 1)); set('ra-p', f2(pi));
  const S = ratStats(r, pi), t = range(0, 399, 1), t2 = range(0, 199, 1);
  const G = grid('ra-chart', [{ r: 0, c: 0, t: 'price = fundamental value + bubble', xt: 'period', yt: 'price' }, { r: 0, c: 1, t: 'y_t = rho y_(t-1) + e_t, same shocks', xt: 'period', yt: 'y_t' }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax;
  plot('ra-chart', [ln(t, S.P, a, K.red, 'price = fundamental + bubble', { line: { width: 1.2 } }), ln(t, S.F, a, K.blue, 'fundamental value', { line: { width: 1.3 } }),
    hl(a, [0, 399], 0, K.navy, '', { line: { width: 0.8, dash: 'dot' } }),
    ln(t2, S.A[0.9], b, K.green, 'rho = 0.9 (stationary)'), ln(t2, S.A[1], b, K.blue, 'rho = 1 (unit root)'), ln(t2, S.A.x, b, K.red, `rho = 1 + r = ${S.rho.toFixed(3)} (explosive)`)], G.lay);
  out('ra-out', [T.ra1(f0(S.max), S.bursts, pc(r, 1), f2(pi), f2((1 + r) / pi)), T.ra2(S.minP < 0, f1(S.minP)),
    T.ra3(trim(S.rho, 4), f2(S.g100), S.dbl > 1e6 ? '∞' : f0(S.dbl), f2(S.end[0.9]), f1(S.end[1]), big(S.end.x))]);
  return S;
}

// ================================================================= EXTRA. worked example: a bubble that can burst
function blkWb() {
  const r = +$('wb-r').value, pi = +$('wb-p').value, k = +$('wb-k').value; set('wb-r', pc(r, 1)); set('wb-p', f2(pi)); set('wb-k', k);
  const g = (1 + r) / pi - 1, dur = 1 / (1 - pi), half = Math.log(0.5) / Math.log(pi), ks = range(0, 100, 1);
  const G = grid('wb-chart', [{ r: 0, c: 0, t: 'probability that the bubble survives k periods', xt: 'k (periods)', y: { range: [0, 1.02] } },
    { r: 0, c: 1, t: 'bubble size B_k / B_0 (log scale)', xt: 'k (periods)', y: logRange(0.9, Math.max(1.5, ((1 + r) / pi) ** 100 * 1.1)) }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax;
  plot('wb-chart', [ln(ks, ks.map(v => pi ** v), a, K.blue, `pi^k, pi = ${pi}`, { line: { width: 1.8 } }),
    mk([half], [0.5], a, K.red, `half-life ${half.toFixed(1)} periods`, { marker: { size: 9, color: K.red } }), mk([k], [pi ** k], a, K.navy, '', { marker: { size: 8, color: K.navy, symbol: 'diamond' } }),
    ln(ks, ks.map(v => ((1 + r) / pi) ** v), b, K.red, `while it survives: ((1 + r) / pi)^k`, { line: { width: 1.8 } }),
    ln(ks, ks.map(v => (1 + r) ** v), b, K.green, 'expected value: (1 + r)^k', { line: { width: 1.8, dash: 'dash' } }),
    mk([k], [((1 + r) / pi) ** k], b, K.navy, `k = ${k}`, { marker: { size: 8, color: K.navy, symbol: 'diamond' } })], G.lay);
  steps('wb-ws', [T.wbs1(nf(1 + r, 3), f2(pi), nf(1 + r, 3)), T.wbs2(nf(1 + r, 3), f2(pi), pc(g, 2), pc(r, 1)), T.wbs3(f2(pi), f1(dur), f1(half)),
    T.wbs4(k, f2(pi), f3(pi ** k), f2(((1 + r) / pi) ** k), f2((1 + r) ** k))]);
  out('wb-out', [T.wb1(pc(g, 2), f1(dur), f1(half))]);
  return { g, dur, half };
}

// ================================================================= 3. explosive roots: right-tailed ADF, SADF, GSADF, BSADF (psy)
function cums(y) {
  const n = y.length - 1, x = new Float64Array(n + 1), xx = new Float64Array(n + 1), z = new Float64Array(n + 1), zz = new Float64Array(n + 1), xz = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) { const a = y[i], d = y[i + 1] - y[i]; x[i + 1] = x[i] + a; xx[i + 1] = xx[i] + a * a; z[i + 1] = z[i] + d; zz[i + 1] = zz[i] + d * d; xz[i + 1] = xz[i] + a * d; }
  return { x, xx, z, zz, xz };
}
function adfAt(C, e, s) {                 // OLS of Delta y on (1, y_{t-1}) over the rows s..e: delta, a, SE and t
  const Sx = C.x[e + 1] - C.x[s], Sxx = C.xx[e + 1] - C.xx[s], Sz = C.z[e + 1] - C.z[s], Szz = C.zz[e + 1] - C.zz[s], Sxz = C.xz[e + 1] - C.xz[s];
  const n = e + 1 - s, den = n * Sxx - Sx * Sx, b = (n * Sxz - Sx * Sz) / den, a = (Sz - b * Sx) / n, ssr = Math.max(Szz - a * Sz - b * Sxz, 1e-300);
  const se = Math.sqrt(ssr / (n - 2) * n / den);
  return { delta: b, a, se, t: b / se, n };
}
function psyRun(k) {
  return memo('psy' + k, () => {
    const P = D.psy[k], y = P.y, n = y.length - 1, r0 = 0.01 + 1.8 / Math.sqrt(n), w0 = Math.floor(r0 * n), C = cums(y);
    const bsadf = new Array(n).fill(NaN), fwd = new Array(n).fill(NaN), arg = new Array(n).fill(-1);
    for (let e = w0 - 1; e < n; e++) {
      let best = -Infinity, bs = -1;
      for (let s = 0; s <= e - w0 + 1; s++) { const t = adfAt(C, e, s).t; if (s === 0) fwd[e] = t; if (t > best) { best = t; bs = s; } }
      bsadf[e] = best; arg[e] = bs;
    }
    const fin = a => a.filter(v => !isNaN(v));
    return { y, C, n, r0, w0, bsadf, fwd, arg, adf: fwd[n - 1], sadf: Math.max(...fin(fwd)), gsadf: Math.max(...fin(bsadf)), dates: P.dates, end: P.dates.slice(1), cv: P.cv, bq: P.bq };
  });
}
function epsAbove(stat, cv, dates, minLen, last) {        // date-stamping: runs of BSADF above its critical value of at least minLen observations
  const n = last === undefined ? stat.length : last + 1, out_ = [];
  const ab = i => cv[i] !== null && !isNaN(stat[i]) && stat[i] > cv[i];
  let i = 0;
  while (i < n) {
    if (ab(i)) { let j = i; while (j + 1 < n && ab(j + 1)) j++; if (j - i + 1 >= minLen) out_.push({ a: dates[i], b: dates[j], len: j - i + 1, i, j }); i = j + 1; } else i++;
  }
  return out_;
}
const minLenLec = n => Math.round(Math.log(n));
const valid = a => a.map(v => (v === null || isNaN(v) ? null : v));

// ---------------------------------------------------------------- 3a. Nasdaq 100: price, BSADF and dated episodes
function blkPn() {
  const L = +$('pn-l').value, lev = $('pn-c').value; set('pn-l', L);
  const R = psyRun('ndx'), cv = R.bq[lev], ep = epsAbove(R.bsadf, cv, R.end, L), pr = R.y.map(Math.exp);
  const lo = Math.min(...pr) * 0.85, hi = Math.max(...pr) * 1.15, blo = Math.min(...R.bsadf.filter(v => !isNaN(v))) - 0.3, bhi = Math.max(...R.bsadf.filter(v => !isNaN(v))) + 0.3;
  const G = grid('pn-chart', [{ r: 0, c: 0, t: 'Nasdaq 100 (weekly close, log scale)', y: logRange(lo, hi) }, { r: 1, c: 0, yt: 'BSADF', y: { range: [blo, bhi] } }], { cols: 1, h: 470, legendRows: 2, gyPx: 40 });
  const [a, b] = G.ax, sp = ep.map(e => [e.a, e.b]);
  plot('pn-chart', [spans(a, sp, lo, hi, K.red, 0.15, `dated explosive episode (at least ${L} weeks)`), ln(R.dates, pr, a, K.teal, 'Nasdaq 100 (weekly close, log scale)', { line: { width: 1.2 } }),
    spans(b, sp, blo, bhi, K.red, 0.15), ln(R.end, valid(R.bsadf), b, K.blue, 'BSADF statistic', { line: { width: 1.2 } }), ln(R.end, cv, b, K.red, `${lev}% critical value`, { line: { width: 1.2, dash: 'dash' } })], G.lay);
  const lg = ep.reduce((m, e) => (e.len > m.len ? e : m), { len: 0 });
  out('pn-out', [T.pn1(f2(R.adf), f2(R.cv.adf[lev]), f2(R.sadf), f2(R.cv.sadf[lev]), f2(R.gsadf), f2(R.cv.gsadf[lev]), lev),
    T.pn2(ep.length, L, lg.len ? dmy(lg.a) : '', lg.len ? dmy(lg.b) : '', lg.len, ep.length ? my(ep[0].a) : '')]);
  return { R, ep };
}

// ---------------------------------------------------------------- EXTRA. SADF and GSADF: the sup over windows
function blkWs() {
  const R = psyRun('ndx'), e = Math.min(R.n - 1, Math.max(R.w0 - 1, +$('ws-e').value)), gs = $('ws-m').value === 'g'; set('ws-e', dmy(R.end[e]));
  const seq = gs ? R.bsadf : R.fwd, s0 = gs ? R.arg[e] : 0, A = adfAt(R.C, e, s0), pr = R.y.map(Math.exp);
  const run = seq.map((v, i) => (i > e || isNaN(v) ? null : v)), runMax = []; let m = -Infinity;
  run.forEach(v => { if (v !== null && v > m) m = v; runMax.push(v === null ? null : m); });
  const supSoFar = Math.max(...run.filter(v => v !== null)), cvv = gs ? R.cv.gsadf['95'] : R.cv.sadf['95'], fin = gs ? R.gsadf : R.sadf;
  const lo = Math.min(...pr) * 0.85, hi = Math.max(...pr) * 1.15;
  const G = grid('ws-chart', [{ r: 0, c: 0, t: gs ? 'window [r1, r2]: the start r1 with the largest ADF' : 'window [0, r2]: the start fixed at the first week', y: logRange(lo, hi) },
    { r: 1, c: 0, yt: gs ? 'BSADF(r2)' : 'ADF over [0, r2]' }], { cols: 1, h: 470, legendRows: 2, gyPx: 40 });
  const [a, b] = G.ax;
  plot('ws-chart', [spans(a, [[R.dates[s0], R.dates[e + 1]]], lo, hi, K.amber, 0.25, `window ending ${R.end[e]}`), ln(R.dates, pr, a, K.teal, 'Nasdaq 100 (weekly close)', { line: { width: 1.1 } }),
    ln(R.end, valid(seq), b, K.blue, gs ? 'BSADF(r2) = sup over r1 of ADF over [r1, r2]' : 'ADF over [0, r2], r2 moving forward', { line: { width: 1.1 } }),
    ln(R.end, runMax, b, K.green, gs ? 'running sup: GSADF of the data so far' : 'running sup: SADF of the data so far', { line: { width: 2 } }),
    hl(b, [R.end[0], R.end[R.n - 1]], cvv, K.red, `95% critical value of ${gs ? 'GSADF' : 'SADF'} (T = ${R.n})`, { line: { width: 1.2, dash: 'dash' } }),
    mk([R.end[e]], [seq[e]], b, K.navy, '', { marker: { size: 9, color: K.navy } })], G.lay);
  steps('ws-ws', [T.wss1(gs, dmy(R.dates[s0]), dmy(R.end[e]), A.n), T.wss2(sig(A.delta, 4), sig(A.se, 4), f2(A.t)), T.wss3(gs, f2(supSoFar)), T.wss4(gs, f2(fin), f2(cvv), R.n, R.w0)]);
  out('ws-out', [T.ws1(f2(R.sadf), f2(R.cv.sadf['95']), f2(R.gsadf), f2(R.cv.gsadf['95']), f2(R.adf), f2(R.cv.adf['95']))]);
  return { A, supSoFar };
}

// ---------------------------------------------------------------- EXTRA. BSADF: a date-stamp crossing in real time
function blkWd() {
  const R = psyRun('ndx'), lev = $('wd-c').value, L = +$('wd-l').value, cv = R.bq[lev]; set('wd-l', L);
  const i0 = R.end.indexOf('1998-06-05'), i1 = R.end.indexOf('2001-06-29'), e = Math.min(i1, Math.max(i0, +$('wd-e').value)); set('wd-e', dmy(R.end[e]));
  const ab = i => cv[i] !== null && R.bsadf[i] > cv[i];
  let run = 0; for (let i = e; i >= 0 && ab(i); i--) run++;
  const first = run ? R.end[e - run + 1] : null, conf = run >= L ? R.end[e - run + L] : null;
  const ep = epsAbove(R.bsadf, cv, R.end, L, e);
  const idx = range(i0, i1, 1), xs = idx.map(i => R.end[i]);
  const G = grid('wd-chart', [{ r: 0, c: 0, yt: 'BSADF', xt: '' }], { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0], ys = idx.map(i => R.bsadf[i]), ylo = Math.min(...ys, ...idx.map(i => cv[i])) - 0.3, yhi = Math.max(...ys) + 0.3;
  G.lay.xaxis.range = [xs[0], xs[xs.length - 1]];
  plot('wd-chart', [spans(a, ep.map(x => [x.a, x.b]), ylo, yhi, K.red, 0.15, `episodes dated with data up to ${R.end[e]}`),
    ln(xs, idx.map(i => (i <= e ? R.bsadf[i] : null)), a, K.blue, 'BSADF, data up to the cursor', { line: { width: 1.6 } }),
    ln(xs, idx.map(i => (i > e ? R.bsadf[i] : null)), a, K.blue, 'BSADF, later weeks', { line: { width: 1, dash: 'dot' }, opacity: 0.6 }),
    ln(xs, idx.map(i => cv[i]), a, K.red, `${lev}% critical value`, { line: { width: 1.2, dash: 'dash' } }),
    mk([R.end[e]], [R.bsadf[e]], a, K.navy, '', { marker: { size: 10, color: K.navy } })], G.lay);
  steps('wd-ws', [T.wds1(R.n, f2(Math.log(R.n)), minLenLec(R.n), L), T.wds2(dmy(R.end[e]), f2(R.bsadf[e]), f2(cv[e]), lev, ab(e)),
    T.wds3(run, first ? dmy(first) : '', L, conf ? dmy(conf) : '')]);
  out('wd-out', [T.wd1(ep.length, dmy(R.end[e]), L, lev)]);
  return { run, first, conf, ep };
}

// ---------------------------------------------------------------- EXTRA. worked example: one window of the Nasdaq 100
function blkWa() {
  const R = psyRun('ndx'), e = Math.min(R.n - 1, Math.max(R.w0 - 1, +$('wa-e').value)), mode = $('wa-s').value; set('wa-e', dmy(R.end[e]));
  const smax = e - R.w0 + 1, stats = range(0, smax, 1).map(s => adfAt(R.C, e, s).t);
  const s0 = mode === 'best' ? R.arg[e] : mode === 'first' ? 0 : smax, A = adfAt(R.C, e, s0), cv = R.bq['95'][e], pr = R.y.map(Math.exp);
  const lo = Math.min(...pr) * 0.85, hi = Math.max(...pr) * 1.15;
  const G = grid('wa-chart', [{ r: 0, c: 0, t: 'Nasdaq 100, weekly close (log scale), and the window', y: logRange(lo, hi) },
    { r: 1, c: 0, t: `t-statistic of delta for every start r1, windows ending ${R.end[e]}`, yt: 'ADF t' }], { cols: 1, h: 480, legendRows: 2, gyPx: 52 });
  const [a, b] = G.ax, xs = range(0, smax, 1).map(s => R.dates[s]);
  plot('wa-chart', [spans(a, [[R.dates[s0], R.dates[e + 1]]], lo, hi, K.amber, 0.25, `window of the calculation (${A.n} weekly changes)`), ln(R.dates, pr, a, K.teal, 'Nasdaq 100', { line: { width: 1.1 } }),
    ln(xs, stats, b, K.blue, 'ADF t over [r1, r2], by start date r1', { line: { width: 1.2 } }), hl(b, [xs[0], xs[xs.length - 1]], cv, K.red, '95% critical value of BSADF at r2', { line: { dash: 'dash', width: 1.2 } }),
    mk([R.dates[R.arg[e]]], [R.bsadf[e]], b, K.green, 'largest: BSADF(r2)', { marker: { size: 10, color: K.green, symbol: 'star' } }), mk([R.dates[s0]], [A.t], b, K.navy, '', { marker: { size: 8, color: K.navy } })], G.lay);
  steps('wa-ws', [T.was1(dmy(R.dates[s0]), dmy(R.end[e]), A.n, mode), T.was2(nf(A.a, 4), nf(A.delta, 4)), T.was3(nf(A.delta, 6), nf(A.se, 6), f2(A.t)),
    T.was4(nf(1 + A.delta, 4), f2(A.t), f2(cv), A.t > cv)]);
  out('wa-out', [T.wa1(dmy(R.end[e]), f2(R.bsadf[e]), dmy(R.dates[R.arg[e]]), f2(cv), R.bsadf[e] > cv)]);
  return A;
}

// ---------------------------------------------------------------- EXTRA. table: interpreting the Nasdaq 100 tests
function psyTimeline(a, keys, L, lev, yLab) {
  const tr = [];
  keys.forEach((k, j) => {
    const R = psyRun(k), ep = epsAbove(R.bsadf, R.bq[lev], R.end, L);
    tr.push(ln([R.dates[0], R.dates[R.n]], [j, j], a, K.navy, '', { line: { width: 1, dash: 'dot' }, hoverinfo: 'skip' }));
    tr.push(spans(a, ep.map(x => [x.a, x.b]), j - 0.3, j + 0.3, PCOL[k], 0.75, ''));
  });
  return tr;
}
function blkTn() {
  const lev = $('tn-c').value, L = +$('tn-l').value; set('tn-l', L);
  const R = psyRun('ndx'), ep = epsAbove(R.bsadf, R.bq[lev], R.end, L);
  const st = [['ADF', R.adf, R.cv.adf], ['SADF', R.sadf, R.cv.sadf], ['GSADF', R.gsadf, R.cv.gsadf]];
  const G = grid('tn-chart', [{ r: 0, c: 0, t: 'statistic and critical value' }, { r: 0, c: 1, t: `episodes of at least ${L} weeks`, y: { range: [-0.6, 0.6], showticklabels: false } }], { cols: 2, h: 340, legendRows: 2, gx: 0.06 });
  const [a, b] = G.ax;
  plot('tn-chart', [bar(st.map(s => s[0]), st.map(s => s[1]), a, K.blue, 'statistic'), mk(st.map(s => s[0]), st.map(s => s[2][lev]), a, K.red, `${lev}% critical value`, { marker: { size: 14, color: K.red, symbol: 'line-ew-open', line: { width: 3, color: K.red } } }),
    ...psyTimeline(b, ['ndx'], L, lev)], G.lay);
  table('tn-tb', T.tnHead, st.map(s => [s[0], R_(f2(s[1])), R_(f2(s[2]['90'])), R_(f2(s[2]['95'])), R_(f2(s[2]['99'])), s[1] > s[2][lev] ? T.yes : T.no]), { note: T.tnNote(R.n, R.w0, nf(R.r0, 3), lev) });
  table('tn-tb2', T.tnHead2, ep.map((x, i) => [R_(i + 1), dmy(x.a), dmy(x.b), R_(x.len)]), { note: T.tnNote2(L, minLenLec(R.n)) });
  out('tn-out', [T.tn1(ep.length, L, lev)]);
  return { ep };
}

// ---------------------------------------------------------------- 3b. Bitcoin, BET and Shanghai: real-time date-stamping
const PK = ['btc', 'bet', 'ssec'];
function blkPp() {
  const sh = +$('pp-s').value, L = +$('pp-l').value, lev = $('pp-c').value; set('pp-s', pc(sh)); set('pp-l', L);
  const cells = []; PK.forEach((k, j) => { cells.push({ r: 0, c: j, t: D.psy[k].label }); cells.push({ r: 1, c: j, yt: j ? '' : 'BSADF' }); });
  const G = grid('pp-chart', cells, { cols: 3, h: 470, gx: 0.06, gyPx: 40, legendRows: 2, rowPx: 170, ml: 46 });
  const tr = [], S = {};
  PK.forEach((k, j) => {
    const R = psyRun(k), last = Math.max(R.w0 - 1, Math.round(sh * R.n) - 1), cv = R.bq[lev], ep = epsAbove(R.bsadf, cv, R.end, L, last);
    const pr = R.y.map(Math.exp), lo = Math.min(...pr) * 0.85, hi = Math.max(...pr) * 1.15, b = R.bsadf.filter(v => !isNaN(v)), blo = Math.min(...b) - 0.3, bhi = Math.max(...b) + 0.3;
    const a1 = G.ax[2 * j], a2 = G.ax[2 * j + 1], sp = ep.map(x => [x.a, x.b]);
    Object.assign(yaxOf(G, a1), logRange(lo, hi));
    G.lay[a2.ya.replace('y', 'yaxis')].range = [blo, bhi];
    G.lay[a1.xa.replace('x', 'xaxis')].range = [R.dates[0], R.dates[R.n]];
    G.lay[a2.xa.replace('x', 'xaxis')].range = [R.dates[0], R.dates[R.n]];
    tr.push(spans(a1, sp, lo, hi, K.red, 0.15, j ? '' : 'dated explosive episode'), ln(R.dates.slice(0, last + 2), pr.slice(0, last + 2), a1, PCOL[k], '', { line: { width: 1.1 } }),
      spans(a2, sp, blo, bhi, K.red, 0.15), ln(R.end.slice(0, last + 1), valid(R.bsadf.slice(0, last + 1)), a2, K.blue, j ? '' : 'BSADF', { line: { width: 1.1 } }),
      ln(R.end, cv, a2, K.red, j ? '' : `${lev}% critical value`, { line: { width: 1.1, dash: 'dash' } }));
    S[k] = { ep, last, end: R.end[last], gsadf: Math.max(...R.bsadf.slice(0, last + 1).filter(v => !isNaN(v))) };
  });
  plot('pp-chart', tr, G.lay);
  out('pp-out', [T.pp1(pc(sh), PK.map(k => T.pp1m(D.psy[k].label, dmy(S[k].end), S[k].ep.length)).join(T.semi)),
    T.pp2(PK.map(k => T.pp2m(D.psy[k].label, f2(psyRun(k).gsadf), f2(psyRun(k).cv.gsadf[lev]))).join(T.semi), lev)]);
  return S;
}

// ---------------------------------------------------------------- EXTRA. table: interpreting the three markets
function blkTs() {
  const lev = $('ts-c').value, L = +$('ts-l').value; set('ts-l', L);
  const G = grid('ts-chart', [{ r: 0, c: 0, t: 'GSADF and SADF against their critical values' }, { r: 0, c: 1, t: `episodes of at least ${L} weeks`, y: { tickvals: [0, 1, 2], ticktext: PK.map(k => D.psy[k].label), range: [-0.6, 2.6] } }],
    { cols: 2, h: 360, legendRows: 2, gx: 0.12 });
  const [a, b] = G.ax, lab = PK.map(k => D.psy[k].label);
  const RR = PK.map(psyRun);
  const xg = [-0.2, 0.8, 1.8], xs_ = [0.2, 1.2, 2.2], cvm = (col, nm) => ({ marker: { size: 18, color: col, symbol: 'line-ew-open', line: { width: 3, color: col } }, name: nm });
  Object.assign(G.lay.xaxis, { tickvals: [0, 1, 2], ticktext: lab, range: [-0.6, 2.6] });
  plot('ts-chart', [bar(xg, RR.map(R => R.gsadf), a, K.blue, 'GSADF', { width: 0.38 }), bar(xs_, RR.map(R => R.sadf), a, K.teal, 'SADF', { width: 0.38 }),
    mk(xg, RR.map(R => R.cv.gsadf[lev]), a, K.red, '', cvm(K.red, `${lev}% critical value of GSADF`)), mk(xs_, RR.map(R => R.cv.sadf[lev]), a, K.navy, '', cvm(K.navy, `${lev}% critical value of SADF`)),
    ...psyTimeline(b, PK, L, lev)], G.lay);
  const rows = PK.map((k, j) => { const R = RR[j], ep = epsAbove(R.bsadf, R.bq[lev], R.end, L);
    return [D.psy[k].label, `${dmy(R.dates[0])} – ${dmy(R.dates[R.n])}`, R_(R.n), R_(f2(R.sadf)), R_(f2(R.cv.sadf[lev])), R_(f2(R.gsadf)), R_(f2(R.cv.gsadf[lev])), ep.map(x => `${my(x.a)} – ${my(x.b)}`).join('; ') || '–']; });
  table('ts-tb', T.tsHead(lev), rows, { note: T.tsNote(L, lev) });
  out('ts-out', [T.ts1(lev, RR.map((R, j) => `${lab[j]} ${f2(R.cv.gsadf[lev])}`).join(T.semi))]);
  return RR.map(R => R.cv.gsadf[lev]);
}

// ================================================================= 4. LPPL: growth paths and components (fig_growth, fig_lppl_components)
function blkGr() {
  const m = +$('gr-m').value, w = +$('gr-w').value; set('gr-m', f2(m)); set('gr-w', w);
  const t = linspace(0, 0.985, 600), expo = t.map(v => 0.8 * v), sing = t.map(v => 1 - (1 - v) ** m), lp = t.map((v, i) => sing[i] + 0.06 * (1 - v) ** m * Math.cos(w * Math.log(1 - v)));
  const d = y => y.slice(1).map((v, i) => (v - y[i]) / (t[i + 1] - t[i])), g = d(sing);
  const G = grid('gr-chart', [{ r: 0, c: 0, xt: 'time t (critical time tc = 1)', yt: 'log price ln p(t)' }, { r: 0, c: 1, xt: 'time t (critical time tc = 1)', yt: 'growth rate d ln p / dt', y: { range: [0, 6] } }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax, t1 = t.slice(1);
  plot('gr-chart', [ln(t, expo.map(v => v - expo[0]), a, K.blue, 'exponential: ln p = g t'), ln(t, sing.map(v => v - sing[0]), a, K.red, `power-law singularity: A + B (tc - t)^m, m = ${m}`),
    ln(t, lp.map(v => v - lp[0]), a, K.green, `LPPL: singularity with log-periodic oscillations, omega = ${w}`),
    ln(t1, d(expo), b, K.blue, ''), ln(t1, g, b, K.red, ''), ln(t1, d(lp), b, K.green, ''), vl(a, 1, -0.1, 1.1, K.navy, '', { line: { dash: 'dot', width: 0.8 } }), vl(b, 1, 0, 6, K.navy, '', { line: { dash: 'dot', width: 0.8 } })], G.lay);
  const r = { g_start: g[0], g_mid: g[Math.floor(g.length / 2)], g_end: g[g.length - 1] };
  out('gr-out', [T.gr1(f2(m), f2(r.g_start), f2(r.g_mid), f2(r.g_end)), T.gr2(f2(m), f2(1 - m))]);
  return r;
}
function blkCm() {
  const m = +$('cm-m').value, w = +$('cm-w').value, cb = +$('cm-c').value; set('cm-m', f2(m)); set('cm-w', w); set('cm-c', f2(cb));
  const t = linspace(0, 0.995, 800), dt = t.map(v => 1 - v);
  const G = grid('cm-chart', [{ r: 0, c: 0, t: 'A + B (tc - t)^m,  A = 1, B = -1', xt: 'time t (tc = 1)' }, { r: 0, c: 1, t: `(tc - t)^m cos(omega ln(tc - t)), m = ${m}`, xt: 'time t (tc = 1)' },
    { r: 0, c: 2, t: 'ln p(t): the sum', xt: 'time t (tc = 1)' }], { cols: 3, h: 360, legendRows: 3, gx: 0.05 });
  const [a, b, c] = G.ax, tr = [];
  [[0.3, K.blue], [0.5, K.red], [0.8, K.green]].forEach(([mm, col]) => tr.push(ln(t, dt.map(v => 1 - v ** mm), a, col, `m = ${mm}`)));
  tr.push(ln(t, dt.map(v => 1 - v ** m), a, K.navy, `m = ${m} (chosen)`, { line: { dash: 'dash', width: 2 } }));
  [[6, K.purple], [10, K.orange]].forEach(([ww, col]) => tr.push(ln(t, dt.map(v => v ** m * Math.cos(ww * Math.log(v))), b, col, `omega = ${ww}`)));
  tr.push(ln(t, dt.map(v => v ** m * Math.cos(w * Math.log(v))), b, K.teal, `omega = ${w} (chosen)`, { line: { dash: 'dash', width: 1.6 } }));
  tr.push(ln(t, dt.map(v => 1 - v ** m + cb * v ** m * Math.cos(w * Math.log(v))), c, K.blue, `LPPL: m = ${m}, omega = ${w}, C/|B| = ${cb}`, { line: { width: 1.6 } }));
  [a, b, c].forEach(x => G.lay.shapes.push(vline(x, 1, K.navy, 'dot', 0.8)));
  plot('cm-chart', tr, G.lay);
  const lam = Math.exp(2 * Math.PI / w);
  out('cm-out', [T.cm1(w, f2(lam), f0(200 / lam), f0(200 / lam ** 2), f0(200 / lam ** 3)), T.cm2(f2(m), f2(cb), f2(m / (w * cb)))]);
  return { lam };
}

// ---------------------------------------------------------------- EXTRA. worked example: log-periodic oscillations, lambda
function blkWl() {
  const w = +$('wl-w').value, d0 = +$('wl-d').value; set('wl-w', f2(w)); set('wl-d', d0);
  const lam = Math.exp(2 * Math.PI / w), days = linspace(Math.log(1), Math.log(400), 900).map(Math.exp), phi = -w * Math.log(d0);   // phase chosen so that a maximum falls at d0 days
  const pk = range(0, 6, 1).map(n => d0 / lam ** n).filter(v => v >= 1);
  const G = grid('wl-chart', [{ r: 0, c: 0, xt: 'days before the critical time tc - t', yt: 'cos(omega ln(tc - t) - phi)', x: { autorange: 'reversed' }, y: { range: [-1.15, 1.25] } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0];
  plot('wl-chart', [ln(days, days.map(v => Math.cos(w * Math.log(v) + phi)), a, K.blue, `cos(omega ln(tc - t) - phi), omega = ${w}`, { line: { width: 1.4 } }),
    mk(pk, pk.map(() => 1), a, K.red, lw(G, `maxima: ${pk.map(v => v.toFixed(0)).join(', ')} days before tc (ratio lambda = ${lam.toFixed(2)})`), { marker: { size: 9, color: K.red } })], G.lay);
  steps('wl-ws', [T.wls1(f2(w), f2(2 * Math.PI / w), f2(lam)), T.wls2(d0, f2(lam), f0(d0 / lam), f0(d0 / lam ** 2), f0(d0 / lam ** 3)), T.wls3(f2(Math.exp(2 * Math.PI / 6.28)), f2(Math.exp(2 * Math.PI / 10)))]);
  out('wl-out', [T.wl1(f2(w), f2(lam), pk.length)]);
  return { lam, pk };
}

// ---------------------------------------------------------------- EXTRA. worked example: an LPPL value by hand
const LP = { A: 8, B: -1.2, C: 0.08, tc: 2, phi: 0 };
const lpVal = (t, m, w) => { const dt = LP.tc - t, fm = dt ** m, lg = Math.log(dt), cs = Math.cos(w * lg - LP.phi); return { dt, fm, lg, cs, lnp: LP.A + LP.B * fm + LP.C * fm * cs }; };
function blkWp() {
  const m = +$('wp-m').value, w = +$('wp-w').value, t = +$('wp-t').value; set('wp-m', f2(m)); set('wp-w', w);
  const ts = linspace(0, 1.999, 900), V = lpVal(t, m, w), pts = [0, 1, 1.75, 1.99].map(x => lpVal(x, m, w));
  const G = grid('wp-chart', [{ r: 0, c: 0, xt: 't (years, tc = 2)', yt: 'ln p(t)' }, { r: 0, c: 1, xt: 't (years, tc = 2)', yt: 'p(t)' }], { cols: 2, h: 340, legendRows: 2 });
  const [a, b] = G.ax, Ls = ts.map(x => lpVal(x, m, w).lnp);
  plot('wp-chart', [ln(ts, Ls, a, K.blue, lw(G, `ln p = 8 - 1.2 (2 - t)^m + 0.08 (2 - t)^m cos(omega ln(2 - t)), m = ${m}, omega = ${w}`), { line: { width: 1.5 } }),
    mk([0, 1, 1.75, 1.99], pts.map(p => p.lnp), a, K.navy, 'the values of the lecture: t = 0, 1, 1.75, 1.99', { marker: { size: 7, color: K.navy } }), mk([t], [V.lnp], a, K.red, `t = ${t}`, { marker: { size: 11, color: K.red } }),
    ln(ts, Ls.map(Math.exp), b, K.blue, '', { line: { width: 1.5 } }), hl(b, [0, 2], Math.exp(LP.A), K.navy, 'e^A = 2981: the price at tc', { line: { dash: 'dot', width: 1 } }), mk([t], [Math.exp(V.lnp)], b, K.red, '', { marker: { size: 11, color: K.red } })], G.lay);
  steps('wp-ws', [T.wps1(nf(t, 2), nf(V.dt, 2), f2(m), f3(V.fm)), T.wps2(nf(V.dt, 2), f3(V.lg), f2(w), f3(V.cs)), T.wps3(f3(V.fm), f3(V.cs), f3(V.lnp), f0(Math.exp(V.lnp)))]);
  out('wp-out', [T.wp1(f0(Math.exp(pts[1].lnp)), f0(Math.exp(pts[3].lnp)), f0(Math.exp(LP.A)), pc((Math.exp(pts[3].lnp) - Math.exp(pts[2].lnp)) / (Math.exp(pts[3].lnp) - Math.exp(pts[0].lnp))))]);
  return { V, pts };
}

// ---------------------------------------------------------------- LPPL fits of the chapter: path and filter conditions
const FL = D.filter, SR = D.search;
function lpplPath(f, t) { const dt = Math.max(f.tc - t, 1e-9), fm = dt ** f.m, lg = Math.log(dt); return f.A + f.B * fm + f.C1 * fm * Math.cos(f.w * lg) + f.C2 * fm * Math.sin(f.w * lg); }
function conds(f, b) {                    // lppl_conditions with the bounds b (the lecture: Shu and Zhu 2020)
  b = Object.assign({ m0: FL.m[0], m1: FL.m[1], w0: FL.w[0], w1: FL.w[1], tc0: FL.tc_frac[0], tc1: FL.tc_frac[1], osc: FL.osc_min, rel: FL.rel_err_max }, b || {});
  const D_ = f.t2 - f.t1;
  const c = { B: f.B < 0, m: b.m0 <= f.m && f.m <= b.m1, w: b.w0 <= f.w && f.w <= b.w1, tc: f.t2 + b.tc0 * D_ <= f.tc && f.tc <= f.t2 + b.tc1 * D_, osc: f.osc >= b.osc,
    damping: f.damping >= SR.damping_min, rel_err: f.rel_err <= b.rel };
  if (Object.values(c).every(Boolean)) { c.lomb = f.lomb_p <= FL.lomb_alpha; c.ar1 = c.lomb ? Math.max(f.adf_p, f.pp_p) < FL.ar1_alpha : false; } else { c.lomb = null; c.ar1 = null; }
  c.q_param = ['B', 'm', 'w', 'tc', 'osc', 'damping'].every(k => c[k] === true);
  c.q_full = ['B', 'm', 'w', 'tc', 'osc', 'damping', 'rel_err', 'lomb', 'ar1'].every(k => c[k] === true);
  return c;
}
// Householder least squares (as numpy lstsq): beta and SSR of y on the columns of X
function lsq(X, y) {
  const n = X.length, p = X[0].length, A = X.map(r => r.slice()), b = y.slice();
  for (let k = 0; k < p; k++) {
    let nr = 0; for (let i = k; i < n; i++) nr += A[i][k] * A[i][k]; nr = Math.sqrt(nr);
    const al = A[k][k] > 0 ? -nr : nr, v = new Float64Array(n); for (let i = k; i < n; i++) v[i] = A[i][k]; v[k] -= al;
    let vv = 0; for (let i = k; i < n; i++) vv += v[i] * v[i]; if (vv === 0) continue;
    for (let j = k; j < p; j++) { let s = 0; for (let i = k; i < n; i++) s += v[i] * A[i][j]; s = 2 * s / vv; for (let i = k; i < n; i++) A[i][j] -= s * v[i]; }
    let s = 0; for (let i = k; i < n; i++) s += v[i] * b[i]; s = 2 * s / vv; for (let i = k; i < n; i++) b[i] -= s * v[i];
  }
  const be = new Array(p).fill(0);
  for (let k = p - 1; k >= 0; k--) { let s = b[k]; for (let j = k + 1; j < p; j++) s -= A[k][j] * be[j]; be[k] = s / A[k][k]; }
  let ssr = 0; for (let i = p; i < n; i++) ssr += b[i] * b[i];
  return { beta: be, ssr };
}
function lpplLinear(t, y, tc, m, w) {
  const X = t.map(v => { const dt = Math.max(tc - v, 1e-9), f = dt ** m, lg = Math.log(dt); return [1, f, f * Math.cos(w * lg), f * Math.sin(w * lg)]; });
  return lsq(X, y);
}

// ---------------------------------------------------------------- EXTRA. worked example: the conditions of the filter
function blkWf() {
  const k = $('wf-e').value, F = EPK.map(x => D.fits[x].rows['30']), f = D.fits[k].rows['30'], c = conds(f);
  const pos = g => (g.tc - g.t2) / (g.t2 - g.t1);
  const G = grid('wf-chart', [{ r: 0, c: 0, t: 'exponent m and frequency omega', xt: 'm', yt: 'omega', x: { range: [-0.05, 1.08] }, y: { range: [0, 27] } },
    { r: 0, c: 1, t: 'critical time and oscillations', xt: '(tc - t2) / (t2 - t1)', yt: 'oscillations in the window', x: { range: [-0.03, 0.37] }, y: { range: [0, 9] } }], { cols: 2, h: 380, legendRows: 3 });
  const [a, b] = G.ax;
  G.lay.shapes.push({ type: 'rect', xref: a.xa, yref: a.ya, x0: FL.m[0], x1: FL.m[1], y0: FL.w[0], y1: FL.w[1], fillcolor: 'rgba(46,125,50,0.10)', line: { color: K.green, width: 1.2 } },
    { type: 'rect', xref: b.xa, yref: b.ya, x0: FL.tc_frac[0], x1: FL.tc_frac[1], y0: FL.osc_min, y1: 9, fillcolor: 'rgba(46,125,50,0.10)', line: { color: K.green, width: 1.2 } },
    { type: 'line', xref: b.xa, yref: b.yd, x0: SR.tc_frac[1], x1: SR.tc_frac[1], y0: 0, y1: 1, line: { color: K.red, width: 1.2, dash: 'dash' } },
    { type: 'line', xref: a.xa, yref: a.yd, x0: 1, x1: 1, y0: 0, y1: 1, line: { color: K.red, width: 1.2, dash: 'dash' } });
  plot('wf-chart', [legBox(lw(G, 'filter: 0.01 <= m <= 0.99, 2 <= omega <= 25; tc within 0.2 (t2 - t1) of t2, at least 2.5 oscillations'), K.green, 0.3), legLine(lw(G, 'edge of the search space (m = 1, tc = t2 + (t2 - t1)/3)'), K.red, 'dash'),
    ...EPK.map(x => mk([D.fits[x].rows['30'].m], [D.fits[x].rows['30'].w], a, ECOL[x], '', { mode: 'markers+text', text: [ELAB[x].replace(/, \d+/, '').replace(' Composite', '') + ELAB[x].slice(-5)], textposition: x === 'sp500' ? 'top left' : x === 'bet' ? 'bottom center' : D.fits[x].rows['30'].m > 0.9 ? 'middle left' : 'middle right', textfont: { size: 9, color: ECOL[x] }, marker: { size: 8, color: ECOL[x] } })),
    mk([f.m], [f.w], a, K.navy, `${ELAB[k]}`, { marker: { size: 16, color: 'rgba(0,0,0,0)', line: { width: 2.5, color: K.navy } } }),
    ...EPK.map(x => mk([pos(D.fits[x].rows['30'])], [D.fits[x].rows['30'].osc], b, ECOL[x], '', { marker: { size: 8, color: ECOL[x] } })),
    mk([pos(f)], [f.osc], b, K.navy, '', { marker: { size: 16, color: 'rgba(0,0,0,0)', line: { width: 2.5, color: K.navy } } })], G.lay);
  const ok = v => (v === null ? T.na : v ? T.pass : T.fail);
  steps('wf-ws', [T.wfs1(ELAB[k], dmy(D.fits[k].low), dmy(f.end), f.n, tdIso(f.tc), dmy(tdIso(f.tc))), T.wfs2(f3(f.B), ok(c.B), f3(f.m), ok(c.m), f2(f.w), ok(c.w)),
    T.wfs3(f3(pos(f)), ok(c.tc), f2(f.osc), ok(c.osc), f2(f.damping), ok(c.damping)), T.wfs4(pc(f.rel_err, 1), ok(c.rel_err), c.lomb === null ? null : f.lomb_p < 0.001 ? '< ' + nf(0.001, 3) : f3(f.lomb_p), ok(c.lomb), ok(c.ar1)),
    T.wfs5(c.q_param, c.q_full)]);
  out('wf-out', [T.wf1(EPK.filter(x => conds(D.fits[x].rows['30']).q_param).map(x => ELAB[x]).join(T.semi), EPK.filter(x => conds(D.fits[x].rows['30']).q_full).length)]);
  return c;
}

// ================================================================= 5. estimation: the cost landscape (fig_cost)
function costAt(wmax, mmax) {
  const C = D.cost, S = C.S[String(wmax)], nt = C.tcs.length, nm = C.ms.length, jm = C.ms.filter(v => v <= mmax + 1e-12).length;
  const M = range(0, nt - 1, 1).map(i => S.slice(i * nm, (i + 1) * nm));
  const prof = M.map(r => Math.min(...r.slice(0, jm)));
  let bi = 0, bj = 0; M.forEach((r, i) => r.slice(0, jm).forEach((v, j) => { if (v < M[bi][bj]) { bi = i; bj = j; } }));
  let loc = 0; for (let i = 1; i < nt - 1; i++) if (prof[i] < prof[i - 1] && prof[i] < prof[i + 1]) loc++;
  return { M, prof, bi, bj, loc, ratio: Math.max(...prof) / Math.min(...prof), tc: tdIso(C.tcs[bi]), m: C.ms[bj], jm };
}
function blkCo() {
  const wmax = +$('co-w').value, mmax = +$('co-m').value; set('co-w', wmax); set('co-m', f2(mmax));
  const C = D.cost, R = costAt(wmax, mmax), tx = C.tcs.map(tdPl), ms = C.ms;
  const G = grid('co-chart', [{ r: 0, c: 0, xt: 'critical time tc', yt: 'exponent m' }, { r: 0, c: 1, xt: 'critical time tc', yt: 'SSR' }], { cols: 2, h: 380, legendRows: 2, gx: 0.2 });
  const [a, b] = G.ax;
  const z = ms.map((_, j) => R.M.map(r => (j < R.jm ? Math.log(r[j]) : null)));
  plot('co-chart', [{ type: 'heatmap', x: tx, y: ms, z, colorscale: 'Viridis', xaxis: a.xa, yaxis: a.ya, showscale: !G.narrow, colorbar: { title: { text: 'log SSR', side: 'top', font: { size: 10 } }, x: a.x1 + 0.01, xanchor: 'left', len: 0.75, thickness: 8, tickfont: { size: 10 } }, hovertemplate: 'tc %{x}<br>m %{y:.2f}<br>log SSR %{z:.4f}<extra></extra>', showlegend: false },
    mk([tx[R.bi]], [ms[R.bj]], a, K.red, 'global minimum', { marker: { size: 16, color: K.red, symbol: 'star' } }), vl(a, C.peak, ms[0], ms[ms.length - 1], K.orange, 'actual peak', { line: { dash: 'dash', width: 1.2 } }),
    ln(tx, R.prof, b, K.blue, `min SSR over m <= ${mmax} and omega in [2, ${wmax}]`, { line: { width: 1.6 } }), vl(b, C.peak, Math.min(...R.prof), Math.max(...R.prof), K.orange, '', { line: { dash: 'dash', width: 1.2 } }),
    mk([tx[R.bi]], [R.prof[R.bi]], b, K.red, '', { marker: { size: 10, color: K.red, symbol: 'star' } })], G.lay);
  out('co-out', [T.co1(wmax, f2(mmax), R.loc, pc(R.ratio - 1, 1)), T.co2(dmy(R.tc), f2(R.m), dmy(C.peak), big(C.tcs.length * C.ms.length * 47))]);
  return R;
}

// ---------------------------------------------------------------- EXTRA. worked example: the Shanghai fit at the end of the window
function ssecWin() {
  return memo('ssw', () => { const E = epData('ssec'), idx = range(E.lo, E.i2, 1); return { d: idx.map(i => E.d[i]), t: idx.map(i => yrs(E.d[i])), y: idx.map(i => Math.log(E.p[i])), p: idx.map(i => E.p[i]) }; });
}
function blkWx() {
  const f0_ = D.fits.ssec.rows['30'], sh = +$('wx-s').value, m = +$('wx-m').value, w = +$('wx-w').value, at = $('wx-t').value; set('wx-s', sh); set('wx-m', f3(m)); set('wx-w', f2(w));
  const W = ssecWin(), tc = f0_.tc + sh / 365.25, L = lpplLinear(W.t, W.y, tc, m, w), [A, B, C1, C2] = L.beta, f = { tc, m, w, A, B, C1, C2 };
  const shs = range(-30, 60, 2), prof = shs.map(s => lpplLinear(W.t, W.y, f0_.tc + s / 365.25, m, w).ssr);
  const iT = at === 't2' ? W.t.length - 1 : at === 't1' ? 0 : Math.floor((W.t.length - 1) / 2), tt = W.t[iT], dt = tc - tt, fm = dt ** m, lg = Math.log(dt), cs = Math.cos(w * lg), sn = Math.sin(w * lg);
  const lnh = A + B * fm + C1 * fm * cs + C2 * fm * sn;
  const tp = linspace(W.t[0], tc - 1e-4, 500);
  const G = grid('wx-chart', [{ r: 0, c: 0, t: 'Shanghai Composite: window 20 March 2014 - 13 May 2015', yt: 'log price' }, { r: 0, c: 1, t: 'SSR of step 1, m and omega fixed', xt: 'tc minus the estimate (days)', yt: 'SSR' }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax;
  plot('wx-chart', [ln(W.d, W.y, a, K.purple, 'ln p, Shanghai Composite', { line: { width: 1 } }), ln(tp.map(tdPl), tp.map(v => lpplPath(f, v)), a, K.red, `LPPL, step 1 for tc = ${tdIso(tc)}`, { line: { width: 1.6 } }),
    mk([W.d[iT]], [lnh], a, K.navy, '', { marker: { size: 9, color: K.navy } }),
    ln(shs, prof, b, K.blue, 'SSR(tc), OLS of step 1 at each tc', { line: { width: 1.4 } }), mk([sh], [L.ssr], b, K.red, '', { marker: { size: 10, color: K.red } })], G.lay);
  steps('wx-ws', [T.wxs1(dmy(tdIso(tc)), nf(tc, 3), f3(m), f2(w), f3(A), f3(B), f3(C1), f3(C2), f3(Math.hypot(C1, C2))),
    T.wxs2(dmy(W.d[iT]), nf(tt, 3), f3(dt), f3(fm), f3(lg)), T.wxs3(f3(cs), f3(sn), f3(A), f3(B * fm), f3(fm * (C1 * cs + C2 * sn)), f3(lnh), f3(W.y[iT])),
    T.wxs4(f0(Math.exp(lnh)), f0(W.p[iT]), pc(Math.exp(lnh) / W.p[iT] - 1, 1))]);
  out('wx-out', [T.wx1(sh, f4(L.ssr), f4(f0_.ssr), pc(L.ssr / f0_.ssr - 1, 2))]);
  return { A, B, C1, C2, lnh, ssr: L.ssr, p: W.p[iT] };
}

// ---------------------------------------------------------------- 5b. LPPL fits before the peak (fig_fits)
function blkFi() {
  const lag = $('fi-l').value;
  const cells = EPK.map((k, i) => ({ r: Math.floor(i / 3), c: i % 3, t: ELAB[k] }));
  const G = grid('fi-chart', cells, { cols: 3, h: 470, gx: 0.06, gyPx: 74, legendRows: 2, rowPx: 190, ml: 46 });
  const tr = [];
  EPK.forEach((k, i) => {
    const a = G.ax[i], E = epData(k), f = D.fits[k].rows[lag], pk = E.d[E.pk], x1 = addDays(pk, 120), idx = range(E.lo, E.d.length - 1, 1).filter(j => E.d[j] <= x1);
    const ys = idx.map(j => Math.log(E.p[j])), tt = linspace(f.t1, Math.min(f.tc, f.t2 + 0.6) - 1e-4, 400), yp = tt.map(v => lpplPath(f, v));
    const ylo = Math.min(...ys, ...yp) - 0.05, yhi = Math.max(...ys, ...yp) + 0.05;
    G.lay[a.ya.replace('y', 'yaxis')].range = [ylo, yhi];
    tr.push(ln(idx.map(j => E.d[j]), ys, a, ECOL[k], '', { line: { width: 1 } }), ln(tt.map(tdPl), yp, a, K.red, i ? '' : 'LPPL fit', { line: { width: 1.5 } }),
      vl(a, f.end, ylo, yhi, K.navy, i ? '' : 'end of the fit window t2', { line: { dash: 'dot', width: 1 } }), vl(a, tdPl(f.tc), ylo, yhi, K.green, i ? '' : 'estimated critical time tc', { line: { dash: 'dashdot', width: 1.2 } }));
    G.lay.shapes.push(vline(a, pk, K.navy, 'dash', 0.9));
  });
  tr.push(legLine('actual peak', K.navy, 'dash'));
  G.lay[G.ax[0].ya.replace('y', 'yaxis')].title = { text: 'log price', font: { size: 11, color: K.navy } };
  plot('fi-chart', tr, G.lay);
  const err = EPK.map(k => dDays(D.fits[k].peak, tdIso(D.fits[k].rows[lag].tc)));
  out('fi-out', [T.fi1(lag, EPK.map((k, i) => `${ELAB[k]} ${sgn(err[i])}`).join(T.semi)),
    T.fi2(EPK.filter(k => D.fits[k].rows[lag].q_param).map(k => ELAB[k]).join(T.semi), EPK.filter(k => D.fits[k].rows[lag].q_full).map(k => ELAB[k]).join(T.semi))]);
  return err;
}

// ---------------------------------------------------------------- EXTRA. table: interpreting the six fits
function blkTf() {
  const lag = $('tf-l').value, show = $('tf-v').value;
  const rows = EPK.map(k => ({ k, f: D.fits[k].rows[lag], pk: D.fits[k].peak }));
  const G = grid('tf-chart', [{ r: 0, c: 0, yt: 'days relative to the actual peak' }], { cols: 1, h: 360, legendRows: 2 });
  const a = G.ax[0], lab = rows.map(r => ELAB[r.k]);
  plot('tf-chart', [bar(lab, rows.map(r => dDays(r.pk, tdIso(r.f.tc))), a, rows.map(r => ECOL[r.k]), 'estimated tc minus the peak'),
    mk(lab, rows.map(r => dDays(r.pk, r.f.end)), a, K.navy, 'end of the window t2', { marker: { size: 12, symbol: 'line-ew-open', color: K.navy, line: { width: 3, color: K.navy } } }),
    mk(lab, rows.map(r => dDays(r.pk, tdIso(r.f.t2 + SR.tc_frac[1] * (r.f.t2 - r.f.t1)))), a, K.red, 'upper bound of the search: t2 + (t2 - t1)/3', { marker: { size: 12, symbol: 'line-ew-open', color: K.red, line: { width: 3, color: K.red } } }),
    mk(lab, rows.map(r => dDays(r.pk, tdIso(r.f.t2 + FL.tc_frac[1] * (r.f.t2 - r.f.t1)))), a, K.green, 'upper bound of the filter: t2 + 0.2 (t2 - t1)', { marker: { size: 10, symbol: 'diamond-open', color: K.green, line: { width: 2, color: K.green } } })], G.lay);
  const atB = r => Math.abs(r.f.tc - (r.f.t2 + SR.tc_frac[1] * (r.f.t2 - r.f.t1))) < 2 / 365.25;
  const yn = v => (v ? T.yes : T.no);
  if (show === 'p') table('tf-tb', T.tfHeadP, rows.map(r => [ELAB[r.k], dmy(r.f.end), dmy(tdIso(r.f.tc)), R_(sgn(dDays(r.pk, tdIso(r.f.tc)))), R_(f2(r.f.m)), R_(f1(r.f.w)), dmy(tdIso(r.f.t2 + SR.tc_frac[1] * (r.f.t2 - r.f.t1))) + (atB(r) ? T.atBound : ''), yn(r.f.q_param), yn(r.f.q_full)]),
    { note: T.tfNoteP(lag) });
  else table('tf-tb', T.tfHeadC, rows.map(r => { const c = r.f.cond; const s = v => (v === null ? '–' : v ? '✓' : '✗'); return [ELAB[r.k], ...['B', 'm', 'w', 'tc', 'osc', 'damping', 'rel_err', 'lomb', 'ar1'].map(x => ({ v: s(c[x]), cls: 'c' }))]; }), { note: T.tfNoteC });
  out('tf-out', [T.tf1(lag, rows.filter(atB).map(r => ELAB[r.k]).join(T.semi), rows.filter(r => r.f.m > 0.99).map(r => ELAB[r.k]).join(T.semi))]);
  return rows;
}

// ================================================================= 6. confidence from many windows
function blkWi() {
  const fr = +$('wi-f').value, m1 = +$('wi-m').value; set('wi-f', f2(fr)); set('wi-m', f2(m1));
  const G = grid('wi-chart', [{ r: 0, c: 0, t: 'Nasdaq 100, 2000', xt: 'window length t2 - t1 (observations)', yt: 'estimated critical time tc' }, { r: 0, c: 1, t: 'Shanghai Composite, 2015', xt: 'window length t2 - t1 (observations)' }], { cols: 2, h: 400, legendRows: 3, ml: 82, gx: 0.12 });
  const tr = [], S = {};
  ['ndx', 'ssec'].forEach((k, j) => {
    const a = G.ax[j], W = D.win[k], rows = W.rows.map(r => Object.assign({ c: conds(r, { tc1: fr, m1 }) }, r)), q = rows.filter(r => r.c.q_param), nq = rows.filter(r => !r.c.q_param);
    tr.push(mk(q.map(r => r.L), q.map(r => tdPl(r.tc)), a, K.green, j ? '' : 'passes the parameter conditions', { marker: { size: 8, color: K.green } }),
      mk(nq.map(r => r.L), nq.map(r => tdPl(r.tc)), a, K.red, j ? '' : 'fails them', { marker: { size: 8, color: K.red, symbol: 'x' } }),
      ln(rows.map(r => r.L), rows.map(r => tdPl(r.t2 + fr * (r.t2 - r.t1))), a, K.green, j ? '' : `upper bound of the tc condition: t2 + ${fr} (t2 - t1)`, { line: { dash: 'dot', width: 1.2 } }),
      hl(a, [50, 750], W.peak, K.red, j ? '' : 'actual peak', { line: { dash: 'dash', width: 1 } }), hl(a, [50, 750], W.t2, K.navy, j ? '' : 'end of the windows t2', { line: { dash: 'dot', width: 1 } }));
    const tq = q.map(r => r.tc);
    S[k] = { n: rows.length, np: q.length, nf: rows.filter(r => r.c.q_full).length, q10: tq.length ? tdIso(qtl(tq, 0.1)) : null, q50: tq.length ? tdIso(qtl(tq, 0.5)) : null, q90: tq.length ? tdIso(qtl(tq, 0.9)) : null };
  });
  plot('wi-chart', tr, G.lay);
  out('wi-out', [T.wi1(f2(fr), f2(m1), S.ndx.np, S.ndx.nf, S.ssec.np, S.ssec.nf), T.wi2(S.ndx.q50 ? dmy(S.ndx.q50) : '', S.ndx.q10 ? dmy(S.ndx.q10) : '', S.ndx.q90 ? dmy(S.ndx.q90) : '', S.ssec.q50 ? dmy(S.ssec.q50) : '')]);
  return S;
}

// ---------------------------------------------------------------- EXTRA. table: interpreting the windows
const CK = ['B', 'm', 'w', 'tc', 'osc', 'damping', 'rel_err'];
function blkTw() {
  const L0 = +$('tw-L').value, all = $('tw-q').value === 'all'; set('tw-L', L0);
  const S = {};
  ['ndx', 'ssec'].forEach(k => {
    const rows = D.win[k].rows.filter(r => r.L >= L0), c = rows.map(r => conds(r)), tq = rows.filter((r, i) => all || c[i].q_param).map(r => r.tc);
    S[k] = { n: rows.length, sh: Object.fromEntries(CK.map(x => [x, c.filter(v => v[x]).length / rows.length])), np: c.filter(v => v.q_param).length, nf: c.filter(v => v.q_full).length,
      q: tq.length ? [0.1, 0.5, 0.9].map(p => tdIso(qtl(tq, p))) : null };
  });
  const G = grid('tw-chart', [{ r: 0, c: 0, yt: 'share of the windows that pass', y: { range: [0, 1.08], tickformat: '.0%' } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0], lab = ['B < 0', '0.01 <= m <= 0.99', '2 <= omega <= 25', 'tc condition', 'oscillations >= 2.5', 'damping >= 1', 'error <= 15%'];
  plot('tw-chart', [bar(lab, CK.map(x => S.ndx.sh[x]), a, K.teal, `Nasdaq 100 (${S.ndx.n} windows)`), bar(lab, CK.map(x => S.ssec.sh[x]), a, K.purple, `Shanghai Composite (${S.ssec.n} windows)`)], Object.assign(G.lay, { barmode: 'group' }));
  table('tw-tb', T.twHead(all), ['ndx', 'ssec'].map(k => [ELAB[k], dmy(D.win[k].t2), R_(S[k].n), R_(S[k].np), R_(S[k].nf), R_(pc(S[k].np / S[k].n)), ...(S[k].q ? S[k].q.map(dmy) : ['–', '–', '–']), dmy(D.win[k].peak)]), { note: T.twNote(L0) });
  out('tw-out', [T.tw1(L0, pc(S.ssec.sh.m), pc(S.ndx.sh.m), CK.reduce((b, x) => (S.ssec.sh[x] < S.ssec.sh[b] ? x : b), 'B'))]);
  return S;
}

// ---------------------------------------------------------------- 6b. the indicator around four peaks (fig_ci)
function ciBlock(p, keys) {
  const c = +$(p + '-c').value; set(p + '-c', f2(c));
  const G = grid(p + '-chart', keys.map((k, j) => ({ r: 0, c: j, t: ELAB[k], yt: j ? '' : 'log price' })), { cols: 2, h: 400, legendRows: 3, gx: 0.1, mr: 40 });
  const tr = [], S = {};
  keys.forEach((k, j) => {
    const a = G.ax[j], b = twin(G, a, 3 + j, 'indicator', { range: [0, 1] }), I = D.ci[k], P = I.price, d = expand(P), lp = P.p.map(Math.log);
    const pmap = Object.fromEntries(d.map((x, i) => [x, lp[i]])), al = I.d.filter((x, i) => I.param[i] !== null && I.param[i] >= c);
    tr.push({ x: I.d, y: I.param, type: 'scatter', mode: 'lines', line: { shape: 'hvh', width: 0 }, fill: 'tozeroy', fillcolor: 'rgba(46,125,50,0.35)', xaxis: b.xa, yaxis: b.ya, name: j ? '' : 'indicator, parameter conditions (right axis)', showlegend: !j, hovertemplate: '%{x}: %{y:.2f}<extra></extra>' },
      ln(I.d, I.full, b, K.purple, j ? '' : 'indicator, all conditions (right axis)', { line: { width: 1.3, shape: 'hvh' } }), hl(b, [I.d[0], I.d[I.d.length - 1]], c, K.green, j ? '' : `alarm threshold c = ${c}`, { line: { dash: 'dot', width: 1.2 } }),
      ln(d, lp, a, ECOL[k], j ? '' : 'log price, in the colour of the series (left axis)', { line: { width: 1 } }),
      mk(al, al.map(x => pmap[x]), a, K.green, j ? '' : 'date with indicator >= c', { marker: { size: 6, color: K.green, symbol: 'triangle-up' } }));
    G.lay.shapes.push(vline(a, I.peak, K.red, 'dash', 0.9));
    const pre = I.d.map((x, i) => i).filter(i => I.d[i] <= I.peak);
    const fa = pre.find(i => I.param[i] !== null && I.param[i] >= c);
    const ip = argmax(I.param.map(v => (v === null ? -1 : v))), iff = argmax(I.full.map(v => (v === null ? -1 : v)));
    S[k] = { maxP: I.param[ip], maxPd: I.d[ip], maxF: I.full[iff], maxFd: I.d[iff], first: fa === undefined ? null : I.d[fa], nAl: al.length, n: I.d.length };
  });
  tr.push(legLine('actual peak', K.red, 'dash'));
  plot(p + '-chart', tr, G.lay);
  out(p + '-out', keys.map(k => T.ci1(ELAB[k], f2(S[k].maxP), dmy(S[k].maxPd), f2(S[k].maxF), dmy(S[k].maxFd))).concat([T.ci2(f2(c), keys.map(k => `${ELAB[k]}: ${S[k].nAl} / ${S[k].n}${S[k].first ? T.ciFirst(dmy(S[k].first)) : ''}`).join(T.semi))]));
  return S;
}
const blkC1 = () => ciBlock('c1', ['ndx', 'ssec']);
const blkC2 = () => ciBlock('c2', ['bet', 'btc17']);

// ---------------------------------------------------------------- 6c. the critical time moves with the data (fig_tc_path)
function blkTp() {
  const kk = +$('tp-k').value; set('tp-k', kk);
  const G = grid('tp-chart', ['ndx', 'btc17'].map((k, j) => ({ r: 0, c: j, t: ELAB[k], xt: 'end of the window t2', yt: j ? '' : 'estimated critical time tc' })), { cols: 2, h: 400, legendRows: 3, ml: 82, gx: 0.12 });
  const tr = [], S = {};
  ['ndx', 'btc17'].forEach((k, j) => {
    const a = G.ax[j], P = D.tcp[k], rows = P.rows, i = Math.min(rows.length - 1, kk - 1), r = rows[i], q = rows.filter(x => x.q_param), nq = rows.filter(x => !x.q_param);
    const ub = rows.map(x => tdPl(x.t2 + SR.tc_frac[1] * (x.t2 - x.t1)));
    tr.push(ln(rows.map(x => x.end), rows.map(x => tdPl(x.tc)), a, ECOL[k], '', { line: { width: 0.8 } }),
      mk(q.map(x => x.end), q.map(x => tdPl(x.tc)), a, K.green, j ? '' : 'tc of a fit that passes the parameter conditions', { marker: { size: 7, color: K.green } }),
      mk(nq.map(x => x.end), nq.map(x => tdPl(x.tc)), a, K.red, j ? '' : 'tc of a fit that fails them', { marker: { size: 7, color: K.red, symbol: 'x' } }),
      ln(rows.map(x => x.end), rows.map(x => x.end), a, K.navy, j ? '' : 'tc = t2 (crash "today")', { line: { dash: 'dot', width: 0.8 }, hoverinfo: 'skip' }),
      ln(rows.map(x => x.end), ub, a, K.orange, j ? '' : 'upper bound of the search: t2 + (t2 - t1)/3', { line: { dash: 'dashdot', width: 1 }, hoverinfo: 'skip' }),
      hl(a, [rows[0].end, rows[rows.length - 1].end], P.peak, K.red, j ? '' : 'actual peak', { line: { dash: 'dash', width: 0.9 } }),
      mk([r.end], [tdPl(r.tc)], a, K.navy, j ? '' : 'the fit at the chosen end date', { marker: { size: 15, color: 'rgba(0,0,0,0)', line: { width: 2.5, color: K.navy } } }));
    const tcs = rows.map(x => x.tc), err = rows.map(x => Math.abs(dDays(P.peak, tdIso(x.tc))));
    const atB = rows.filter(x => Math.abs(x.tc - (x.t2 + SR.tc_frac[1] * (x.t2 - x.t1))) < 2 / 365.25).length;
    S[k] = { r, n: rows.length, share: q.length / rows.length, min: tdIso(Math.min(...tcs)), max: tdIso(Math.max(...tcs)), med: median(err), atB };
  });
  plot('tp-chart', tr, G.lay);
  out('tp-out', ['ndx', 'btc17'].map(k => T.tp1(ELAB[k], dmy(S[k].r.end), dmy(tdIso(S[k].r.tc)), dDays(D.tcp[k].peak, tdIso(S[k].r.tc)), S[k].r.q_param)).concat(
    [T.tp2(S.ndx.n, dmy(S.ndx.min), dmy(S.ndx.max), pc(S.ndx.share), S.ndx.atB, S.btc17.n, dmy(S.btc17.min), dmy(S.btc17.max), pc(S.btc17.share), f0(S.btc17.med))]));
  return S;
}

// ================================================================= 7. evaluation over the whole sample (fig_eval)
function futureFall(d, p, horizon) {
  const n = d.length, out_ = new Array(n).fill(NaN), t = d.map(tms), last = t[n - 1]; let j = 0;
  for (let i = 0; i < n; i++) {
    const lim = t[i] + horizon * DAY; if (lim > last) break;
    j = Math.max(j, i + 1); while (j < n && t[j] <= lim) j++;
    if (j > i + 1) { let m = Infinity; for (let u = i + 1; u < j; u++) if (p[u] < m) m = p[u]; out_[i] = m / p[i] - 1; }
  }
  return out_;
}
function evalData(k, horizon) {
  return memo(`ev${k}${horizon}`, () => {
    const E = D.ev[k], d = expand(E.price), ff = futureFall(d, E.price.p, horizon), map = Object.fromEntries(d.map((x, i) => [x, ff[i]]));
    return { d, p: E.price.p, ci: E, fall: E.d.map(x => map[x]) };
  });
}
const LEVELS = [0.05, 0.1, 0.2, 0.3, 0.4];
function hitRates(V, crash, col, levels) {
  const C = V.ci[col], idx = V.fall.map((f, i) => i).filter(i => !isNaN(V.fall[i]));
  const cr = i => V.fall[i] <= -crash, base = idx.filter(cr).length / idx.length, o = { base, n: idx.length };
  (levels || LEVELS).forEach(c => { const a = idx.filter(i => C[i] !== null && C[i] >= c); o[c] = { share: a.length / idx.length, n: a.length, hit: a.length ? a.filter(cr).length / a.length : NaN }; });
  return o;
}
function alarmEps(V, level, crash, col, gap) {
  const C = V.ci[col], o = [];
  V.ci.d.forEach((x, i) => { if (C[i] === null || C[i] < level) return; if (!o.length || dDays(o[o.length - 1].last, x) > (gap || 90)) o.push({ first: x, last: x, i }); else o[o.length - 1].last = x; });
  o.forEach(e => { const f = V.fall[e.i]; e.fall = isNaN(f) ? null : f; e.crash = isNaN(f) ? null : f <= -crash; });
  return o;
}
function crashList(d, p, level) {
  const o = []; let pkT = d[0], pk = p[0], trT = pkT, tr = pk;
  for (let i = 0; i < d.length; i++) {
    if (p[i] >= pk) { if (tr / pk - 1 <= -level) o.push({ peak: pkT, trough: trT, fall: tr / pk - 1 }); pkT = d[i]; pk = p[i]; trT = d[i]; tr = p[i]; } else if (p[i] < tr) { trT = d[i]; tr = p[i]; }
  }
  if (tr / pk - 1 <= -level) o.push({ peak: pkT, trough: trT, fall: tr / pk - 1 });
  return o;
}
function evalStats(k, c, crash, horizon) {
  const V = evalData(k, horizon), hr = hitRates(V, crash, 'param', LEVELS.concat([c])), hf = hitRates(V, crash, 'full', LEVELS.concat([c])), ep = alarmEps(V, c, crash, 'param');
  const i0 = V.d.indexOf(V.ci.d[0]), falls = crashList(V.d.slice(i0), V.p.slice(i0), crash), al = V.ci.d.filter((x, i) => V.ci.param[i] !== null && V.ci.param[i] >= c);
  falls.forEach(f => { f.before = al.some(x => x <= f.peak && dDays(x, f.peak) < horizon); });
  return { V, hr, hf, ep, falls, al };
}
function evBlock(p, k, lab, col) {
  const c = +$(p + '-c').value, crash = +$(p + '-x').value / 100, hz = +$(p + '-h').value; set(p + '-c', f2(c)); set(p + '-x', pc(crash));
  const S = evalStats(k, c, crash, hz), V = S.V;
  const G = grid(p + '-chart', [{ r: 0, c: 0, cs: 1, t: `${lab}: alarms (parameter conditions >= ${c})` }, { r: 0, c: 1, t: `${lab}: hit rate by threshold`, xt: 'alarm threshold c', yt: 'hit rate', y: { range: [0, 1.02] } }],
    { cols: 2, h: 400, legendRows: 3, gx: 0.1 });
  const [a, b] = G.ax, pm = Object.fromEntries(V.d.map((x, i) => [x, V.p[i]])), i0 = V.d.indexOf(V.ci.d[0]), pv = V.p.slice(i0);
  Object.assign(yaxOf(G, a), logRange(Math.min(...pv) * 0.9, Math.max(...pv) * 1.1));
  const hit = [], fa = [], unk = [];
  V.ci.d.forEach((x, i) => { const v = V.ci.param[i]; if (v === null || v < c) return; const f = V.fall[i]; (isNaN(f) ? unk : f <= -crash ? hit : fa).push(x); });
  const lv = LEVELS.slice();
  plot(p + '-chart', [ln(V.d.slice(i0), V.p.slice(i0), a, col, lab, { line: { width: 0.9 } }),
    mk(hit, hit.map(x => pm[x]), a, K.green, `alarm followed by a fall of ${(100 * crash).toFixed(0)}% or more`, { marker: { size: 6, color: K.green } }),
    mk(fa, fa.map(x => pm[x]), a, K.red, 'false alarm', { marker: { size: 6, color: K.red, symbol: 'x' } }),
    ln(lv, lv.map(v => S.hr[v].hit), b, K.green, 'hit rate, parameter conditions', { mode: 'lines+markers', marker: { size: 7, color: K.green } }),
    ln(lv, lv.map(v => S.hf[v].hit), b, K.purple, 'hit rate, all conditions', { mode: 'lines+markers', marker: { size: 7, color: K.purple, symbol: 'square' } }),
    hl(b, [0, 0.45], S.hr.base, K.red, 'unconditional frequency', { line: { dash: 'dash', width: 1.2 } }),
    mk([c], [isNaN(S.hr[c].hit) ? null : S.hr[c].hit], b, K.navy, '', { marker: { size: 13, color: 'rgba(0,0,0,0)', line: { width: 2.5, color: K.navy } } })], G.lay);
  const nb = S.falls.filter(f => f.before).length;
  out(p + '-out', [T.ev1(lab, pc(S.hr.base), pc(crash), hz, f2(c), pc(S.hr[c].share), S.hr[c].n, isNaN(S.hr[c].hit) ? null : pc(S.hr[c].hit)),
    T.ev2(S.ep.length, S.ep.filter(e => e.crash).length, S.ep.filter(e => e.crash !== null).length), T.ev3(S.falls.length, pc(crash), nb, S.falls.filter(f => f.before).map(f => f.peak.slice(0, 4)).filter((x, i, A) => A.indexOf(x) === i).join(', '))]);
  return S;
}
const blkE1 = () => evBlock('e1', 'sp500', 'S&P 500', K.blue);
const blkE2 = () => evBlock('e2', 'btc', 'Bitcoin', K.amber);

// ---------------------------------------------------------------- EXTRA. table: interpreting the evaluation
function blkTv() {
  const crash = +$('tv-x').value / 100, col = $('tv-v').value; set('tv-x', pc(crash));
  const G = grid('tv-chart', [{ r: 0, c: 0, t: 'S&P 500', xt: 'alarm threshold c', yt: 'hit rate', y: { range: [0, 1.02] } }, { r: 0, c: 1, t: 'Bitcoin', xt: 'alarm threshold c', y: { range: [0, 1.02] } }], { cols: 2, h: 360, legendRows: 2 });
  const tr = [], S = {};
  [['sp500', K.blue], ['btc', K.amber]].forEach(([k, cl], j) => {
    const V = evalData(k, D.horizon), hr = hitRates(V, crash, col), a = G.ax[j];
    tr.push(ln(LEVELS, LEVELS.map(v => hr[v].hit), a, cl, j ? '' : `hit rate (${col === 'param' ? 'parameter conditions' : 'all conditions'})`, { mode: 'lines+markers', marker: { size: 7, color: cl } }),
      hl(a, [0, 0.45], hr.base, K.red, j ? '' : 'unconditional frequency', { line: { dash: 'dash', width: 1.2 } }));
    S[k] = { hr, ep: alarmEps(V, D.level, crash, col), falls: evalStats(k, D.level, crash, D.horizon).falls };
  });
  plot('tv-chart', tr, G.lay);
  const rows = [];
  [['sp500', 'S&P 500'], ['btc', 'Bitcoin']].forEach(([k, lab]) => {
    rows.push(Object.assign([lab, T.base, R_('100%'), R_(S[k].hr.n), R_(pc(S[k].hr.base, 1))], { hi: true }));
    LEVELS.forEach(c => rows.push(['', `c = ${nf(c, 2)}`, R_(pc(S[k].hr[c].share, 1)), R_(S[k].hr[c].n), R_(isNaN(S[k].hr[c].hit) ? '–' : pc(S[k].hr[c].hit, 1))]));
  });
  table('tv-tb', T.tvHead, rows, { note: T.tvNote(pc(crash), col) });
  const e = k => S[k].ep;
  out('tv-out', [T.tv1(pc(crash), pc(S.sp500.hr.base), pc(S.sp500.hr[0.2].hit), pc(S.btc.hr.base), pc(S.btc.hr[0.2].hit)),
    T.tv2(e('sp500').length, e('sp500').filter(x => x.crash).length, e('btc').length, e('btc').filter(x => x.crash).length, S.sp500.falls.length, S.sp500.falls.filter(f => f.before).length, S.btc.falls.length, S.btc.falls.filter(f => f.before).length)]);
  return S;
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
const NDXI = () => D.psy.ndx.dates.slice(1);
const ix = s => NDXI().indexOf(s);
// [anchor, prefix, function, input moved by ▶, its values (null: the options of a select), ms]
function blkList() {
  const n = D.psy.ndx.y.length - 1, w0 = Math.floor((0.01 + 1.8 / Math.sqrt(n)) * n);
  return [['episodes', 'ep', blkEp, 'ep-h', null, 900], ['t-ep', 'te', blkTe, 'te-h', null, 900],
    ['wx-bw', 'wb', blkWb, 'wb-p', range(0.9, 0.99, 0.01), 500], ['rational', 'ra', blkRa, 'ra-r', range(0, 0.04, 0.0025), 500],
    ['wx-sadf', 'ws', blkWs, 'ws-e', range(w0 - 1, n - 1, 12).concat([n - 1]), 350], ['wx-bsadf', 'wd', blkWd, 'wd-e', range(ix('1998-06-05'), ix('2001-06-29'), 3), 300],
    ['wx-adf', 'wa', blkWa, 'wa-e', range(ix('1999-01-01'), ix('2000-12-29'), 4), 350], ['psy-ndx', 'pn', blkPn, 'pn-l', range(1, 20, 1), 500], ['t-ndx', 'tn', blkTn, 'tn-c', null, 900],
    ['psy-panel', 'pp', blkPp, 'pp-s', range(0.3, 1, 0.05), 500], ['t-psy', 'ts', blkTs, 'ts-c', null, 900],
    ['growth', 'gr', blkGr, 'gr-m', range(0.1, 0.9, 0.1), 500], ['wx-lam', 'wl', blkWl, 'wl-w', range(2, 25, 1), 400], ['components', 'cm', blkCm, 'cm-m', range(0.1, 0.9, 0.1), 500],
    ['wx-lppl', 'wp', blkWp, 'wp-m', range(0.1, 0.9, 0.1), 500], ['wx-filter', 'wf', blkWf, 'wf-e', null, 900],
    ['cost', 'co', blkCo, 'co-w', D.cost.wmax, 500], ['wx-ssec', 'wx', blkWx, 'wx-s', range(-30, 60, 6), 400], ['fits', 'fi', blkFi, 'fi-l', null, 900], ['t-fits', 'tf', blkTf, 'tf-l', null, 900],
    ['windows', 'wi', blkWi, 'wi-f', range(0.05, 0.33, 0.02), 400], ['t-win', 'tw', blkTw, 'tw-L', range(50, 500, 50), 500],
    ['ci', 'c1', blkC1, 'c1-c', range(0.05, 0.6, 0.05), 400], ['ci-b', 'c2', blkC2, 'c2-c', range(0.05, 0.6, 0.05), 400], ['tc-path', 'tp', blkTp, 'tp-k', range(1, 39, 1), 300],
    ['eval', 'e1', blkE1, 'e1-c', range(0.05, 0.5, 0.05), 600], ['eval-btc', 'e2', blkE2, 'e2-c', range(0.05, 0.5, 0.05), 600], ['t-eval', 'tv', blkTv, 'tv-x', range(10, 40, 5), 600]];
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
window.CH13 = { epData, epStats, bwPath, ratStats, adfAt, psyRun, epsAbove, conds, lsq, lpplLinear, lpplPath, costAt, futureFall, evalData, hitRates, alarmEps, crashList, evalStats,
  lpVal, ssecWin, yrs, tdIso, blkList, init, blk: { blkEp, blkTe, blkWb, blkRa, blkWs, blkWd, blkWa, blkPn, blkTn, blkPp, blkTs, blkGr, blkWl, blkCm, blkWp, blkWf, blkCo, blkWx, blkFi, blkTf, blkWi, blkTw, blkC1, blkC2, blkTp, blkE1, blkE2, blkTv } };
if (!window.CH13_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
