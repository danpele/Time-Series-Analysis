// Chapter 10 interactive page (state space models, the Kalman filter and Markov switching): one block per chart slide, plus
// the result tables and the worked examples of the lecture, computed in the browser from ch10_data.js
// (Quantlets/Ch_10/export_interactive.py).
(function () {
'use strict';
const D = window.CH10_DATA;
const { sum, mean, nf, $, AX, layout, plot, Player, range, predict, exitTicket, normCdf, solve } = window.TSAI;
const out = (id, items) => window.TSAI.out(id, items);
const T = window.CH10_TEXT;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of tsa_style.py / generate_all_charts.py
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', orange: '#E67E22', purple: '#8E44AD', teal: '#17A2B8', crimson: '#DC3545',
  navy: '#1F2A44' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.amber]: '181,133,63', [K.orange]: '230,126,34', [K.purple]: '142,68,173',
  [K.teal]: '23,162,184', [K.navy]: '31,42,68' };

// ---------------------------------------------------------------- small helpers
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
const has = id => !!$(id);
const cache = {};
const memo = (k, f) => (k in cache ? cache[k] : (cache[k] = f()));
const f0 = x => nf(x, 0), f1 = x => nf(x, 1), f2 = x => nf(x, 2), f3 = x => nf(x, 3), f4 = x => nf(x, 4);
const pc = (x, d) => nf(100 * x, d === undefined ? 0 : d) + '%';
const sdp = x => { const m = mean(x); return Math.sqrt(sum(x.map(v => (v - m) ** 2)) / x.length); };
const corr = (a, b) => { const ma = mean(a), mb = mean(b); let s = 0, sa = 0, sb = 0; for (let i = 0; i < a.length; i++) { s += (a[i] - ma) * (b[i] - mb); sa += (a[i] - ma) ** 2; sb += (b[i] - mb) ** 2; } return s / Math.sqrt(sa * sb); };
const cumsum = a => { let s = 0; return a.map(v => (s += v)); };
const argmin = a => a.reduce((b, v, i) => (v < a[b] ? i : b), 0);
const argmax = a => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
const isoQ = (q0, k) => { const y0 = +q0.slice(0, 4), m = +q0.slice(5, 7) - 1 + 3 * k; return `${y0 + Math.floor(m / 12)}-${String(m % 12 + 1).padStart(2, '0')}-01`; };
const isoM = (m0, k) => { const y0 = +m0.slice(0, 4), m = +m0.slice(5, 7) - 1 + k; return `${y0 + Math.floor(m / 12)}-${String(m % 12 + 1).padStart(2, '0')}-01`; };
const qlab = d => `${d.slice(0, 4)}Q${(+d.slice(5, 7) + 2) / 3}`;          // chart labels (English)
const Z90 = 1.6448536269514722;
const phi = z => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
const chi2sf = (x, df) => { let s = 0, t = 1; const h = x / 2; for (let i = 0; i < df / 2; i++) { if (i) t *= h / i; s += t; } return Math.exp(-h) * s; };   // even df

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
// shaded periods (rec = 0/1 by date) as shapes of panel a
function shade(a, dts, rec, step) {
  const sh = []; let on = null;
  dts.forEach((d, i) => { if (rec[i] && on === null) on = d; if (!rec[i] && on !== null) { sh.push([on, d]); on = null; } });
  if (on !== null) sh.push([on, step(dts[dts.length - 1])]);
  return sh.map(([x0, x1]) => ({ type: 'rect', xref: a.xa, yref: a.yd, x0, x1, y0: 0, y1: 1, fillcolor: `rgba(${RGB[K.amber]},0.28)`, line: { width: 0 }, layer: 'below' }));
}
const legShade = name => leg(name, null, { color: `rgba(${RGB[K.amber]},0.5)`, symbol: 'square', size: 12 });

// ---------------------------------------------------------------- tables and worked steps (HTML)
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}>${v.v}</${tag}>`; };
  el.innerHTML = (o && o.cap ? `<p class="c10cap">${o.cap}</p>` : '') + '<table><thead><tr>' + head.map(c => cell(c, 'th')).join('') + '</tr></thead><tbody>'
    + rows.map(r => '<tr' + (r.hi ? ' class="hi"' : '') + '>' + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c10cap">${o.note}</p>` : '');
}
const txt = (v, bold) => ({ v: bold ? `<b>${v}</b>` : v, cls: 'r' });
const H_ = v => ({ v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };
const hiRow = (r, on) => { r.hi = on; return r; };

// ================================================================= state space machinery
// local level filter: y_t = mu_t + eps_t, mu_t+1 = mu_t + eta_t; diffuse start a1 = 0, P1 = 1e7; first d terms left out of log L
function llf(y, s2e, s2n, a1, P1, d) {
  a1 = a1 || 0; P1 = P1 === undefined ? 1e7 : P1; d = d === undefined ? 1 : d;
  const n = y.length, ap = new Float64Array(n), Pp = new Float64Array(n), af = new Float64Array(n), Pf = new Float64Array(n), v = new Float64Array(n).fill(NaN),
    F = new Float64Array(n).fill(NaN), Kg = new Float64Array(n);
  let a = a1, P = P1, ll = 0, nobs = 0;
  for (let t = 0; t < n; t++) {
    ap[t] = a; Pp[t] = P;
    if (y[t] === null || Number.isNaN(y[t])) { af[t] = a; Pf[t] = P; }
    else {
      v[t] = y[t] - a; F[t] = P + s2e; Kg[t] = P / F[t]; af[t] = a + Kg[t] * v[t]; Pf[t] = P - Kg[t] * P;
      if (t >= d) { ll += -0.5 * (Math.log(2 * Math.PI) + Math.log(F[t]) + v[t] * v[t] / F[t]); nobs++; }
    }
    a = af[t]; P = Pf[t] + s2n;
  }
  return { ap, Pp, af, Pf, v, F, K: Kg, ll, nobs, aNext: a, PNext: P, s2e, s2n };
}
function rts1(kf) {                           // Rauch-Tung-Striebel smoother of the local level model (T = 1)
  const n = kf.af.length, as = Float64Array.from(kf.af), Ps = Float64Array.from(kf.Pf), J = new Float64Array(n).fill(NaN);
  for (let t = n - 2; t >= 0; t--) { J[t] = kf.Pf[t] / kf.Pp[t + 1]; as[t] = kf.af[t] + J[t] * (as[t + 1] - kf.ap[t + 1]); Ps[t] = kf.Pf[t] + J[t] * J[t] * (Ps[t + 1] - kf.Pp[t + 1]); }
  return { as, Ps, J };
}
function distSmooth1(kf) {                    // standardised auxiliary residuals of the local level model (disturbance smoother)
  const n = kf.af.length, obs = new Float64Array(n).fill(NaN), lev = new Float64Array(n).fill(NaN);
  let r = 0, N = 0;
  for (let t = n - 1; t >= 0; t--) {
    lev[t] = r / Math.sqrt(N);
    if (Number.isNaN(kf.v[t])) continue;
    const Kt = kf.K[t], L = 1 - Kt, F = kf.F[t], v = kf.v[t];
    obs[t] = (v / F - Kt * r) / Math.sqrt(1 / F + Kt * Kt * N);
    r = v / F + L * r; N = 1 / F + L * L * N;
  }
  return { obs, lev };
}
const steady = q => { const p = (q + Math.sqrt(q * q + 4 * q)) / 2; return { P: p, K: p / (p + 1) }; };
function profile(y, q) {                      // log L with sigma2_eps concentrated out (filter with sigma2_eps = 1, sigma2_eta = q)
  const kf = llf(y, 1, q); let s = 0, lf = 0, n = 0;
  for (let t = 1; t < y.length; t++) if (!Number.isNaN(kf.v[t])) { s += kf.v[t] ** 2 / kf.F[t]; lf += Math.log(kf.F[t]); n++; }
  const s2 = s / n; return { ll: -0.5 * n * (Math.log(2 * Math.PI) + 1 + Math.log(s2)) - 0.5 * lf, s2, n };
}
function golden(f, lo, hi, tol) {             // maximum of a unimodal f on [lo, hi]
  const g = (Math.sqrt(5) - 1) / 2; let a = lo, b = hi, c = b - g * (b - a), d = a + g * (b - a), fc = f(c), fd = f(d);
  while (b - a > (tol || 1e-7)) { if (fc > fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); } else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); } }
  return (a + b) / 2;
}
const sesPath = (y, al, l0) => { let l = l0; return y.map(x => { const o = l; l = al * x + (1 - al) * l; return o; }); };

// 2 x 2 helpers
const m2 = { mul: (A, B) => [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]], [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]],
  tr: A => [[A[0][0], A[1][0]], [A[0][1], A[1][1]]], add: (A, B) => [[A[0][0] + B[0][0], A[0][1] + B[0][1]], [A[1][0] + B[1][0], A[1][1] + B[1][1]]],
  sub: (A, B) => [[A[0][0] - B[0][0], A[0][1] - B[0][1]], [A[1][0] - B[1][0], A[1][1] - B[1][1]]],
  inv: A => { const d = A[0][0] * A[1][1] - A[0][1] * A[1][0]; return [[A[1][1] / d, -A[0][1] / d], [-A[1][0] / d, A[0][0] / d]]; },
  mv: (A, x) => [A[0][0] * x[0] + A[0][1] * x[1], A[1][0] * x[0] + A[1][1] * x[1]] };
// Kalman filter with a 2-dimensional state and a scalar observation: y_t = z_t a_t + eps_t (z_t a row), a_t+1 = Tm a_t + eta_t
function kf2(y, Zs, Tm, H, Q, a1, P1, d) {
  const n = y.length, ap = [], Pp = [], af = [], Pf = [], v = new Float64Array(n), F = new Float64Array(n), Kg = [];
  let a = a1.slice(), P = P1.map(r => r.slice()), ll = 0;
  for (let t = 0; t < n; t++) {
    ap.push(a); Pp.push(P);
    const z = Zs(t), Pz = m2.mv(P, z);
    v[t] = y[t] - (z[0] * a[0] + z[1] * a[1]); F[t] = z[0] * Pz[0] + z[1] * Pz[1] + H;
    const k = [Pz[0] / F[t], Pz[1] / F[t]]; Kg.push(k);
    const a_ = [a[0] + k[0] * v[t], a[1] + k[1] * v[t]], P_ = [[P[0][0] - k[0] * Pz[0], P[0][1] - k[0] * Pz[1]], [P[1][0] - k[1] * Pz[0], P[1][1] - k[1] * Pz[1]]];
    af.push(a_); Pf.push(P_);
    if (t >= d) ll += -0.5 * (Math.log(2 * Math.PI) + Math.log(F[t]) + v[t] * v[t] / F[t]);
    a = m2.mv(Tm, a_); P = m2.add(m2.mul(m2.mul(Tm, P_), m2.tr(Tm)), Q);
  }
  return { ap, Pp, af, Pf, v, F, K: Kg, ll, Tm };
}
function rts2(kf) {
  const n = kf.af.length, as = kf.af.map(a => a.slice()), Ps = kf.Pf.map(P => P.map(r => r.slice())), Tt = m2.tr(kf.Tm);
  for (let t = n - 2; t >= 0; t--) {
    const J = m2.mul(m2.mul(kf.Pf[t], Tt), m2.inv(kf.Pp[t + 1])), da = [as[t + 1][0] - kf.ap[t + 1][0], as[t + 1][1] - kf.ap[t + 1][1]], dj = m2.mv(J, da);
    as[t] = [kf.af[t][0] + dj[0], kf.af[t][1] + dj[1]];
    Ps[t] = m2.add(kf.Pf[t], m2.mul(m2.mul(J, m2.sub(Ps[t + 1], kf.Pp[t + 1])), m2.tr(J)));
  }
  return { as, Ps };
}
// Hodrick-Prescott trend: (I + lambda D'D) tau = y, D the second difference (pentadiagonal system, banded elimination)
function hpTrend(y, lam) {
  const n = y.length, A = Array.from({ length: n }, () => new Float64Array(5)), b = Float64Array.from(y);   // A[i][2 + j - i] for |j - i| <= 2
  for (let k = 0; k + 2 < n; k++) { const c = [1, -2, 1]; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) A[k + i][2 + (k + j) - (k + i)] += lam * c[i] * c[j]; }
  for (let i = 0; i < n; i++) A[i][2] += 1;
  for (let i = 0; i < n; i++) for (let r = i + 1; r <= Math.min(n - 1, i + 2); r++) {
    const f = A[r][2 + i - r] / A[i][2];
    for (let j = i; j <= Math.min(n - 1, i + 2); j++) A[r][2 + j - r] -= f * A[i][2 + j - i];
    b[r] -= f * b[i];
  }
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let j = i + 1; j <= Math.min(n - 1, i + 2); j++) s -= A[i][2 + j - i] * x[j]; x[i] = s / A[i][2]; }
  return x;
}
// Hamilton (2018): OLS of y_t+h on 1, y_t, ..., y_t-p+1; the residual is the cycle
function hamFilter(y, h, p) {
  const n = y.length, rows = [], ys = [], idx = [];
  for (let t = h + p - 1; t < n; t++) { rows.push([1, ...range(0, p - 1, 1).map(j => y[t - h - j])]); ys.push(y[t]); idx.push(t); }
  const k = p + 1, XtX = Array.from({ length: k }, () => new Array(k).fill(0)), Xty = new Array(k).fill(0);
  rows.forEach((r, i) => { for (let a = 0; a < k; a++) { Xty[a] += r[a] * ys[i]; for (let c = 0; c < k; c++) XtX[a][c] += r[a] * r[c]; } });
  const b = solve(XtX, Xty), cyc = new Array(n).fill(null);
  rows.forEach((r, i) => { cyc[idx[i]] = ys[i] - r.reduce((s, x, j) => s + x * b[j], 0); });
  return { cyc, b };
}
// Hamilton filter and Kim smoother, two regimes: y_t = mu_S + eps, eps ~ N(0, s2_S); p11 = Pr(1 -> 1), p22 = Pr(2 -> 2); ergodic start
function hamilton(y, mu, s2, p11, p22) {
  const n = y.length, pi1 = (1 - p22) / (2 - p11 - p22), pred = [], filt = [], dens = [];
  let ll = 0, prev = null;
  for (let t = 0; t < n; t++) {
    const pr = prev ? [prev[0] * p11 + prev[1] * (1 - p22), prev[0] * (1 - p11) + prev[1] * p22] : [pi1, 1 - pi1];
    const f = [0, 1].map(j => phi((y[t] - mu[j]) / Math.sqrt(s2[j])) / Math.sqrt(s2[j])), j0 = pr[0] * f[0], j1 = pr[1] * f[1], s = j0 + j1;
    pred.push(pr); dens.push(f); prev = [j0 / s, j1 / s]; filt.push(prev); ll += Math.log(s);
  }
  const sm = new Array(n); sm[n - 1] = filt[n - 1];
  const P = [[p11, 1 - p11], [1 - p22, p22]];
  for (let t = n - 2; t >= 0; t--) { const r = [sm[t + 1][0] / pred[t + 1][0], sm[t + 1][1] / pred[t + 1][1]]; sm[t] = [0, 1].map(i => filt[t][i] * (P[i][0] * r[0] + P[i][1] * r[1])); }
  return { pred, filt, sm, dens, ll, pi1, p1: { pred: pred.map(p => p[0]), filt: filt.map(p => p[0]), sm: sm.map(p => p[0]) } };
}
function concord(prob, rec) {                 // 'probability > 0.5' against the NBER indicator
  const cls = prob.map(p => +(p > 0.5)); let ok = 0, h = 0, nr = 0, fa = 0, ne = 0;
  cls.forEach((c, i) => { ok += +(c === rec[i]); if (rec[i]) { nr++; h += c; } else { ne++; fa += c; } });
  return { concord: ok / cls.length, hit: h / nr, false: fa / ne };
}

// ---------------------------------------------------------------- data of the chapter
const NY = D.nile.y, YRS = NY.map((_, i) => D.nile.y0 + i), NYR = YRS.length;
const S2E = NB.nile.s2_eps, QML = NB.nile.s2_eta / NB.nile.s2_eps;
const QS = [0.001, 0.003, 0.01, 0.03, QML, 0.3, 1, 3, 10], QI0 = 4;
const qOf = id => QS[+$(id).value];
const qTxt = q => (q === QML ? f3(q) + T.ml : (q < 0.01 ? nf(q, 3) : q < 1 ? nf(q, 2) : nf(q, 0)));
const nile = q => memo('nile|' + q, () => { const kf = llf(NY, S2E, q * S2E), sm = rts1(kf); return { kf, sm, ss: steady(q), ds: distSmooth1(kf) }; });
const yi = yr => yr - D.nile.y0;

const RO = D.ro, RY = RO.y, RQ = RY.map((_, i) => isoQ(RO.q0, i)), NRO = RY.length;
const LAMS = [1, 10, 100, 400, 1600, 6400, 25600, 100000], LI0 = 4;
const hpOf = (y, lam) => memo(`hp|${lam}|${y.length}`, () => { const tr = hpTrend(y, lam); return { tr, gap: y.map((v, i) => v - tr[i]) }; });
const RG = RY.slice(1).map((v, i) => v - RY[i]), RGQ = RQ.slice(1);           // Romanian growth q/q (100 x log difference)

const US = D.us, UG = US.g, UQ = UG.map((_, i) => isoQ(US.q0, i)), UREC = US.rec;
const uIdx = d => UQ.indexOf(d);
const EXT = NB.msus.ext;
const usX = () => { const i0 = uIdx(US.x0), i1 = uIdx('2019-10-01'); return { y: UG.slice(i0, i1 + 1), q: UQ.slice(i0, i1 + 1), rec: UREC.slice(i0, i1 + 1) }; };
const usXA = () => { const i0 = uIdx(US.x0); return { y: UG.slice(i0), q: UQ.slice(i0), rec: UREC.slice(i0) }; };
const nextQ = d => isoQ(d, 1);

// ================================================================= 1. local level and local linear trend paths
function blkSe() {
  const qr = +$('se-q').value, sz = +$('se-z').value, n = +$('se-T').value; set('se-q', f1(qr)); set('se-z', f3(sz)); set('se-T', n);
  const E = D.ss.eps.slice(0, n), Et = D.ss.eta.slice(0, n), Zt = D.ss.zeta.slice(0, n), t = range(0, n - 1, 1), res = [];
  const G = grid('se-chart', [{ r: 0, c: 0, t: 'local level: thin = observed y_t, thick = level μ_t', xt: 't' }, { r: 0, c: 1, t: 'local linear trend', xt: 't' }], { cols: 2, h: 400, legendRows: 3, gx: 0.12 });
  const [a, b] = G.ax, tr = [];
  [[0.01, K.blue], [0.1, K.green], [qr, K.red]].forEach(([q, c]) => {
    const mu = cumsum(Et.map(e => Math.sqrt(q) * e)).map(v => 10 + v), y = mu.map((m, i) => m + E[i]);
    tr.push(ln(t, y, a, c, '', { line: { width: 0.9 } }), ln(t, mu, a, c, `q = ${q === qr ? q.toFixed(1) : q}: y_t thin, μ_t thick`, { line: { width: 2.4 } }));
    const dy = y.slice(1).map((v, i) => v - y[i]);
    res.push({ q, acf: corr(dy.slice(1), dy.slice(0, -1)), sd: sdp(dy), th: -1 / (q + 2) });
  });
  const slope = cumsum(Zt.map(z => sz * z)).map(v => 0.05 + v), mu = cumsum(slope).map(v => 10 + v), y = mu.map((m, i) => m + 2 * E[i]);
  const s2 = overlay(G, b, 'y3', 'slope', {});
  tr.push(ln(t, y, b, K.purple, 'y_t, local linear trend', { line: { width: 0.9 } }), ln(t, mu, b, K.orange, 'level μ_t', { line: { width: 2 } }),
    ln(t, slope, s2, K.teal, 'slope β_t (right axis)', { line: { width: 1.4, dash: 'dash' } }));
  plot('se-chart', tr, G.lay);
  out('se-out', [T.se1(n, res.map(r => T.seq(r.q === qr ? f1(r.q) : nf(r.q, r.q < 0.1 ? 2 : 1), f2(r.acf), f3(r.th), f2(r.sd))).join(T.sep)), T.se2(f3(sz), f3(slope[n - 1]))]);
  return { res, slope: slope[n - 1] };
}

// ================================================================= EXTRA. worked example: AR(2) for US GDP growth
function arma() {
  return memo('arma', () => {
    const A = NB.arma, x = D.arma.x, mu = A.mu, Tm = [[A.phi1, A.phi2], [1, 0]];
    const M = [[1 - Tm[0][0] * Tm[0][0], -Tm[0][0] * Tm[0][1], -Tm[0][1] * Tm[0][0], -Tm[0][1] * Tm[0][1]], [-Tm[0][0] * Tm[1][0], 1 - Tm[0][0] * Tm[1][1], -Tm[0][1] * Tm[1][0], -Tm[0][1] * Tm[1][1]],
      [-Tm[1][0] * Tm[0][0], -Tm[1][0] * Tm[0][1], 1 - Tm[1][1] * Tm[0][0], -Tm[1][1] * Tm[0][1]], [-Tm[1][0] * Tm[1][0], -Tm[1][0] * Tm[1][1], -Tm[1][1] * Tm[1][0], 1 - Tm[1][1] * Tm[1][1]]];
    const p = solve(M, [A.s2, 0, 0, 0]), P1 = [[p[0], p[1]], [p[2], p[3]]];
    const kf = kf2(x.map(v => v - mu), () => [1, 0], Tm, 0, [[A.s2, 0], [0, 0]], [0, 0], P1, 0);
    return { x, mu, P1, kf, d: D.arma.d };
  });
}
function blkWa() {
  const t = +$('wa-t').value, S = arma(), kf = S.kf, i = t - 1; set('wa-t', `${t} (${T.q(S.d[i])})`);
  const G = grid('wa-chart', [{ r: 0, c: 0, cs: 2, t: 'US real GDP growth (% q/q) and the one-step forecast of the filter', x: DATE }, { r: 1, c: 0, t: 'prediction-error variance F_t', xt: 't (quarter)' },
    { r: 1, c: 1, t: 'log-likelihood contribution of quarter t', xt: 't (quarter)' }], { cols: 2, h: 520, legendRows: 2, gyPx: 70 });
  const [a, b, c] = G.ax, fc = kf.ap.map(v => S.mu + v[0]), sd = Array.from(kf.F, F => Math.sqrt(F)), tt = range(1, 12, 1);
  const lt = Array.from(kf.v, (v, k) => -0.5 * (Math.log(2 * Math.PI) + Math.log(kf.F[k]) + v * v / kf.F[k]));
  plot('wa-chart', [band(S.d, fc.map((f, k) => f - Z90 * sd[k]), fc.map((f, k) => f + Z90 * sd[k]), a, K.red, 0.12, '90% band of the forecast'),
    ln(S.d, S.x, a, K.blue, 'GDP growth', { line: { width: 0.9 } }), ln(S.d, fc, a, K.red, 'one-step forecast μ + Z a_t', { line: { width: 1.4 } }),
    mk([S.d[i]], [S.x[i]], a, K.navy, `quarter t = ${t}`, { marker: { size: 10, symbol: 'diamond', color: K.navy } }),
    bar(tt, tt.map(k => kf.F[k - 1]), b, tt.map(k => (k === t ? K.red : K.blue)), ''), hline(b, [0.5, 12.5], NB.arma.s2, K.navy, 'σ² of the AR(2)', { line: { dash: 'dot', width: 1 } }),
    bar(tt, tt.map(k => lt[k - 1]), c, tt.map(k => (k === t ? K.red : K.teal)), ''),
    mk([t], [kf.F[i]], b, K.navy, '', { marker: { size: 9, symbol: 'diamond', color: K.navy } }), mk([t], [lt[i]], c, K.navy, '', { marker: { size: 9, symbol: 'diamond', color: K.navy } })], G.lay);
  const P = S.P1, v = kf.v[i], F = kf.F[i], Kk = kf.K[i], ll = kf.ll;
  steps('wa-ws', [T.was1(f3(NB.arma.phi1), f3(NB.arma.phi2), f3(NB.arma.s2), f3(P[0][0]), f3(P[0][1])),
    T.was2(t, T.q(S.d[i]), f3(S.x[i]), f3(fc[i]), f3(v), f3(F), f3(Kk[0]), f3(Kk[1])), T.was3(t, f3(lt[i]), nf(ll, 4), nf(NB.arma.ll_sarimax, 4))]);
  out('wa-out', [T.wa1(t, T.q(S.d[i]), f2(S.x[i]), f2(fc[i]), f3(F)), T.wa2(nf(ll, 4), nf(NB.arma.ll_sarimax, 4), S.x.length)]);
  return { ll, F1: kf.F[0], F3: kf.F[2], P1: P };
}

// ================================================================= EXTRA. the local level filter by hand (three periods)
function byHand(y, a1, P1, s2e, s2n, nRows) {
  const rows = []; let a = a1, P = P1;
  for (let t = 1; t <= nRows; t++) {
    const yt = t <= y.length ? y[t - 1] : null, F = P + s2e, Kt = yt === null ? 0 : P / F, v = yt === null ? null : yt - a;
    const af = yt === null ? a : a + Kt * v, Pf = yt === null ? P : P * (1 - Kt);
    rows.push({ t, y: yt, a, P, F, K: Kt, Kg: P / F, v, af, Pf }); a = af; P = Pf + s2n;
  }
  return { rows, aNext: a, PNext: P };
}
function blkWk() {
  const s2e = +$('wk-e').value, s2n = +$('wk-n').value, P1 = +$('wk-p').value, st = +$('wk-s').value; set('wk-e', f1(s2e)); set('wk-n', f2(s2n)); set('wk-p', f1(P1));
  const R = byHand([12, 11, 14], 10, P1, s2e, s2n, 3), L = byHand(new Array(12).fill(0), 10, P1, s2e, s2n, 12), q = s2n / s2e, ss = steady(q), tt = range(1, 12, 1);
  const G = grid('wk-chart', [{ r: 0, c: 0, t: 'data, prediction a_t and filtered level a_t|t', xt: 't', x: { dtick: 1 } }, { r: 0, c: 1, t: 'Kalman gain K_t', xt: 't' }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax;
  plot('wk-chart', [mk([1, 2, 3], [12, 11, 14], a, K.blue, 'observation y_t', { marker: { size: 10, color: K.blue } }),
    ln([1, 2, 3, 4], [...R.rows.map(r => r.a), R.aNext], a, K.orange, 'prediction a_t', { mode: 'lines+markers', line: { dash: 'dash' }, marker: { size: 7 } }),
    ln([1, 2, 3], R.rows.map(r => r.af), a, K.red, 'filtered a_t|t', { mode: 'lines+markers', marker: { size: 8 } }),
    ln(tt, L.rows.map(r => r.Kg), b, K.blue, 'K_t = P_t/(P_t + σ²_ε)', { mode: 'lines+markers', marker: { size: 5 } }),
    hline(b, [1, 12], ss.K, K.red, `steady state K̄ = ${ss.K.toFixed(3)}`, { line: { dash: 'dash', width: 1.2 } })], G.lay);
  const r = R.rows[st - 1];
  steps('wk-ws', [T.wks1(st, f2(r.P), f1(s2e), f2(r.F), f3(r.K)), T.wks2(st, f0(r.y), f2(r.a), f2(r.v)),
    T.wks3(st, f2(r.a), f3(r.K), f2(r.v), f2(r.af), f2(r.P), f2(r.Pf)), T.wks4(st + 1, f2(r.af), f2(r.Pf), f2(s2n), f2(r.Pf + s2n))]);
  out('wk-out', [T.wk1(f3(R.rows[0].K), f3(R.rows[1].K), f3(R.rows[2].K), f2(R.rows[2].af)), T.wk2(f2(q), f3(ss.K), f2(ss.P * s2e))]);
  return R;
}

// ================================================================= EXTRA. the table of the three periods
function blkKt() {
  const y3 = +$('kt-y').value, a1 = +$('kt-a').value, nr = +$('kt-r').value; set('kt-y', f1(y3)); set('kt-a', f1(a1));
  const R = byHand([12, 11, y3], a1, 4, 4, 1, nr), r3 = R.rows[2], tt = R.rows.map(r => r.t);
  const G = grid('kt-chart', [{ r: 0, c: 0, t: 'observations and filtered level', xt: 't', x: { dtick: 1 } }, { r: 0, c: 1, t: 'surprise v_t and the move of the level K_t v_t', xt: 't', x: { dtick: 1 } }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax, ob = R.rows.filter(r => r.y !== null);
  plot('kt-chart', [mk(ob.map(r => r.t), ob.map(r => r.y), a, K.blue, 'observation y_t', { marker: { size: 10, color: K.blue } }),
    ln(tt, R.rows.map(r => r.af), a, K.red, 'filtered level a_t|t', { mode: 'lines+markers', marker: { size: 7 } }),
    ln(tt, R.rows.map(r => r.a), a, K.orange, 'prediction a_t', { mode: 'lines+markers', line: { dash: 'dash' }, marker: { size: 6 } }),
    bar(ob.map(r => r.t), ob.map(r => r.v), b, K.blue, 'surprise v_t'), bar(ob.map(r => r.t), ob.map(r => r.K * r.v), b, K.red, 'move of the level K_t v_t')],
    Object.assign(G.lay, { barmode: 'group' }));
  const dash = '–', rows = R.rows.map(r => [String(r.t), txt(r.y === null ? dash : nf(r.y, Number.isInteger(r.y) ? 0 : 1)), txt(f2(r.a)), txt(f2(r.P)), txt(f2(r.F)), txt(r.y === null ? dash : f3(r.K)),
    txt(r.v === null ? dash : f2(r.v)), txt(f2(r.af), true), txt(f2(r.Pf))]);
  table('kt-tb', ['t', H_('y<sub>t</sub>'), H_('a<sub>t</sub>'), H_('P<sub>t</sub>'), H_('F<sub>t</sub>'), H_('K<sub>t</sub>'), H_('v<sub>t</sub>'), H_('a<sub>t|t</sub>'), H_('P<sub>t|t</sub>')], rows,
    { cap: T.ktCap(f1(a1), nr) });
  out('kt-out', [T.kt1(f1(y3), f2(r3.v), f3(r3.K), f2(r3.K * r3.v), f2(r3.af)), T.kt2(f2(R.rows.length > 3 ? R.rows[3].a : R.aNext), f2(R.rows.length > 3 ? R.rows[3].P : R.PNext), f2((R.rows.length > 3 ? R.rows[3].P : R.PNext) + 4))]);
  return R;
}

// ================================================================= EXTRA. steady state and exponential smoothing
const QSS = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10];
function blkWs() {
  const q = QSS[+$('ws-q').value], al = +$('ws-a').value; set('ws-q', nf(q, 2)); set('ws-a', f2(al));
  const ss = steady(q), L = byHand(new Array(15).fill(0), 10, 4, 4, 4 * q, 15), tt = range(1, 15, 1), qq = range(-3, 1, 0.05).map(e => 10 ** e);
  const G = grid('ws-chart', [{ r: 0, c: 0, t: 'steady-state gain K̄ against q', xt: 'q = σ²_η/σ²_ε (log scale)', x: LOG, y: { range: [0, 1] } },
    { r: 0, c: 1, t: 'K_t from P_1 = 4, σ²_ε = 4', xt: 't', y: { range: [0, 1] } }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax;
  plot('ws-chart', [ln(qq, qq.map(x => steady(x).K), a, K.blue, 'K̄(q) = SES weight α'), mk([q], [ss.K], a, K.red, `q = ${q}: K̄ = ${ss.K.toFixed(3)}`, { marker: { size: 11, color: K.red } }),
    ln(tt, L.rows.map(r => r.Kg), b, K.blue, 'K_t', { mode: 'lines+markers', marker: { size: 5 } }), hline(b, [1, 15], ss.K, K.red, '', { line: { dash: 'dash', width: 1.2 } })], G.lay);
  steps('ws-ws', [T.wss1(f2(q), f4(q * q + 4 * q), f4(ss.P), f2(4 * ss.P)), T.wss2(f2(4 * ss.P), f3(ss.K)), T.wss3(f2(al), f3(al * al / (1 - al)))]);
  out('ws-out', [T.ws1(f2(q), f2(4 * ss.P), f3(ss.K), f3(L.rows[2].Kg))]);
  return { ss };
}

// ================================================================= 2. the Nile: the Kalman filter of the local level model
function blkNf() {
  const q = qOf('nf-q'); set('nf-q', qTxt(q));
  const S = nile(q), kf = S.kf, G = grid('nf-chart', [{ r: 0, c: 0, xt: 'year', yt: '10^8 m^3' }], { cols: 1, h: 360, legendRows: 2 }), a = G.ax[0];
  const sd = Array.from(kf.Pf, Math.sqrt), af = Array.from(kf.af), x1 = YRS.slice(1);
  plot('nf-chart', [band(x1, af.slice(1).map((v, i) => v - Z90 * sd[i + 1]), af.slice(1).map((v, i) => v + Z90 * sd[i + 1]), a, K.red, 0.15, '90% band of the filtered level'),
    mk(YRS, NY, a, K.blue, 'annual flow y_t', { marker: { size: 5, color: K.blue } }), ln(x1, Array.from(kf.ap).slice(1), a, K.orange, 'prediction a_t', { line: { width: 1.2, dash: 'dash' } }),
    ln(YRS, af, a, K.red, 'filtered level a_t|t', { line: { width: 2 } })], G.lay);
  const i = yi(1899);
  out('nf-out', [T.nf1(qTxt(q), f0(q * S2E), f3(S.ss.K)), T.nf2(f0(NY[i]), f0(kf.ap[i]), f0(kf.v[i]), f3(kf.K[i]), f0(kf.af[i]), f0(kf.af[yi(1905)])),
    T.nf3(f0(kf.af[NYR - 1]), f0(sd[NYR - 1]), f0(mean(NY)))]);
  return { kf, ss: S.ss };
}

// ================================================================= EXTRA. maximum likelihood: the table of the estimates
function nileML(end) {
  return memo('nm|' + end, () => {
    const y = NY.slice(0, yi(end) + 1), f = lq => profile(y, 10 ** lq).ll, lq = golden(f, -5, 1, 1e-9), q = 10 ** lq, P = profile(y, q);
    const s2e = P.s2, k = 2, ll = P.ll;
    const ge = range(3.6, 4.6, 1 / 30), gn = range(2, 4, 1 / 15);
    const z = gn.map(b => ge.map(c => llf(y, 10 ** c, 10 ** b).ll));
    const qs = range(-3, 0.5, 0.05).map(e => 10 ** e);
    return { n: y.length, q, s2e, s2n: q * s2e, ll, aic: -2 * ll + 2 * k, bic: -2 * ll + Math.log(P.n) * k, Kb: steady(q).K, ge, gn, z, qs, pl: qs.map(x => profile(y, x).ll) };
  });
}
function blkNm() {
  const end = +$('nm-e').value, u = $('nm-u').value; set('nm-e', end);
  const M = nileML(end), G = grid('nm-chart', [{ r: 0, c: 0, t: `log L over (σ²_ε, σ²_η), 1871-${end}`, xt: 'log10 σ²_ε', yt: 'log10 σ²_η' },
    { r: 0, c: 1, t: 'profile log-likelihood', xt: 'q (log scale)', x: LOG }], { cols: 2, h: 400, legendRows: 2, gx: 0.1 });
  const [a, b] = G.ax, zmax = Math.max(...M.z.flat());
  plot('nm-chart', [{ type: 'contour', x: M.ge, y: M.gn, z: M.z.map(r => r.map(v => Math.max(v, zmax - 30))), xaxis: a.xa, yaxis: a.ya, showscale: false, ncontours: 16,
    colorscale: [[0, '#F7FAFD'], [0.5, '#7FA7D9'], [1, K.blue]], contours: { coloring: 'heatmap' }, hovertemplate: 'log L %{z:.1f}<extra></extra>', name: '', showlegend: false },
    mk([Math.log10(M.s2e)], [Math.log10(M.s2n)], a, K.red, 'maximum (concentrated profile)', { marker: { size: 11, symbol: 'x', color: K.red } }),
    ln(M.qs, M.pl, b, K.blue, 'profile log L'), mk([M.q], [M.ll], b, K.red, '', { marker: { size: 10, color: K.red } })], G.lay);
  const N1 = NB.nile, S = NB.sm, sd = u === 'sd', V = x => (sd ? f1(Math.sqrt(x)) : f0(x)), dash = '–';
  const rows = [[sd ? 'σ<sub>ε</sub>' : 'σ²<sub>ε</sub>', txt(V(N1.s2_eps)), txt(V(S.s2_eps)), txt(V(M.s2e), true)],
    [sd ? 'σ<sub>η</sub>' : 'σ²<sub>η</sub>', txt(V(N1.s2_eta)), txt(V(S.s2_eta)), txt(V(M.s2n), true)],
    ['q = σ²<sub>η</sub>/σ²<sub>ε</sub>', txt(f4(N1.q)), txt(f4(S.s2_eta / S.s2_eps)), txt(f4(M.q), true)], ['log L', txt(f2(N1.loglik)), txt(f2(S.loglik)), txt(f2(M.ll), true)],
    ['AIC', txt(f2(N1.aic)), txt(dash), txt(f2(M.aic))], ['BIC', txt(f2(N1.bic)), txt(dash), txt(f2(M.bic))], ['K̄', txt(f3(N1.ss_K)), txt(f3(steady(S.s2_eta / S.s2_eps).K)), txt(f3(M.Kb))],
    [T.L.nobs, txt(N1.n), txt(N1.n), txt(M.n)]];
  table('nm-tb', [T.L.par, H_(T.L.own), H_(T.L.sm), H_(T.L.conc(end))], rows, { cap: T.nmCap, note: T.nmNote });
  out('nm-out', [T.nm1(end, f3(M.q), f0(M.s2e), f0(M.s2n), f2(M.ll)), T.nm2(f3(M.Kb))]);
  return M;
}

// ================================================================= 3. the gain converges; SES reproduces the filter
function blkGs() {
  const q = qOf('gs-q'), al = $('gs-a').value; set('gs-q', qTxt(q));
  const S = nile(q), kf = S.kf, Kb = S.ss.K, alpha = al === 'K' ? Kb : NB.gain.alpha_ses;
  const fk = Array.from(kf.ap).slice(1), fs = sesPath(NY, alpha, NY[0]).slice(1), gap = fk.map((v, i) => Math.abs(v - fs[i]));
  const G = grid('gs-chart', [{ r: 0, c: 0, t: 'gain of the update step', xt: 'year' }, { r: 0, c: 1, t: 'Kalman filter and exponential smoothing', xt: 'year' }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax, yy = YRS.slice(1, 21);
  plot('gs-chart', [ln(yy, Array.from(kf.K).slice(1, 21), a, K.blue, 'Kalman gain K_t', { mode: 'lines+markers', marker: { size: 6 } }),
    hline(a, [yy[0], yy[19]], Kb, K.red, `steady state K̄ = ${Kb.toFixed(3)}`, { line: { dash: 'dash', width: 1.2 } }),
    mk(YRS, NY, b, K.blue, 'annual flow', { marker: { size: 4, color: K.blue } }), ln(YRS.slice(1), fk, b, K.red, 'Kalman one-step forecast', { line: { width: 2.2 } }),
    ln(YRS.slice(1), fs, b, K.green, `SES with α = ${alpha.toFixed(3)}`, { line: { width: 1.2, dash: 'dash' } })], G.lay);
  const Kk = Array.from(kf.K), nc = Kk.findIndex((k, t) => t >= 1 && Kk.slice(t).every(x => Math.abs(x - Kb) < 0.001));
  out('gs-out', [T.gs1(f3(Kk[1]), f3(Kk[4]), nc + 1, f3(Kb)), T.gs2(f3(alpha), f2(gap[4]), f2(Math.max(...gap.slice(19)))), T.gs3(f3(NB.gain.alpha_ses), f3(NB.gain.q_from_alpha))]);
  return { K2: Kk[1], K5: Kk[4], nc: nc + 1, gap5: gap[4], gap20: Math.max(...gap.slice(19)) };
}

// ================================================================= EXTRA. the smoother step
function blkWr() {
  const yr = +$('wr-y').value; set('wr-y', yr);
  const S = nile(QML), kf = S.kf, sm = S.sm, i = yi(yr), lo = Math.max(1871, yr - 12), hi = Math.min(1970, yr + 12), w = range(lo, hi, 1);
  const G = grid('wr-chart', [{ r: 0, c: 0, t: `filtered and smoothed level, ${lo}-${hi}`, xt: 'year', yt: '10^8 m^3' }, { r: 0, c: 1, t: 'smoother gain J_t', xt: 'year' }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax, W = arr => w.map(y => arr[yi(y)]);
  plot('wr-chart', [mk(w, W(NY), a, K.blue, 'annual flow', { marker: { size: 5, color: K.blue } }), ln(w, W(kf.af), a, K.orange, 'filtered a_t|t'), ln(w, W(sm.as), a, K.red, 'smoothed α̂_t', { line: { width: 2 } }),
    mk([yr, yr], [kf.af[i], sm.as[i]], a, K.navy, `year ${yr}`, { marker: { size: 10, symbol: 'diamond', color: K.navy } }),
    ln(YRS.slice(1, -1), Array.from(sm.J).slice(1, -1), b, K.blue, 'J_t = P_t|t / P_t+1'), mk([yr], [sm.J[i]], b, K.navy, '', { marker: { size: 10, symbol: 'diamond', color: K.navy } })], G.lay);
  steps('wr-ws', [T.wrs1(yr, f0(kf.Pf[i]), yr + 1, f0(kf.Pp[i + 1]), f3(sm.J[i])), T.wrs2(yr, f1(kf.af[i]), f3(sm.J[i]), f1(sm.as[i + 1]), f1(kf.ap[i + 1]), f1(sm.as[i])),
    T.wrs3(f0(kf.Pf[i]), f3(sm.J[i]), f0(sm.Ps[i + 1]), f0(kf.Pp[i + 1]), f0(sm.Ps[i]), f1(Math.sqrt(sm.Ps[i])), f1(Math.sqrt(kf.Pf[i])))]);
  out('wr-out', [T.wr1(yr, f0(kf.af[i]), f0(sm.as[i]), f3(sm.J[i]))]);
  return { af: kf.af[i], as: sm.as[i] };
}

// ================================================================= 4. the Nile: filtered against smoothed level
function blkNs() {
  const q = qOf('ns-q'); set('ns-q', qTxt(q));
  const S = nile(q), kf = S.kf, sm = S.sm, G = grid('ns-chart', [{ r: 0, c: 0, xt: 'year', yt: '10^8 m^3' }], { cols: 1, h: 360, legendRows: 2 }), a = G.ax[0];
  const sds = Array.from(sm.Ps, Math.sqrt), as = Array.from(sm.as);
  plot('ns-chart', [band(YRS, as.map((v, i) => v - Z90 * sds[i]), as.map((v, i) => v + Z90 * sds[i]), a, K.red, 0.15, '90% band of the smoothed level'),
    mk(YRS, NY, a, K.blue, 'annual flow', { marker: { size: 4, color: K.blue } }), ln(YRS.slice(1), Array.from(kf.af).slice(1), a, K.orange, 'filtered level a_t|t', { line: { width: 1.3 } }),
    ln(YRS, as, a, K.red, 'smoothed level α̂_t', { line: { width: 2.2 } })], G.lay);
  const i97 = yi(1897), i00 = yi(1900), i20 = yi(1920);
  out('ns-out', [T.ns1(qTxt(q), f0(as[i97]), f0(kf.af[i97]), f0(as[i00]), f0(kf.af[i00])), T.ns2(f0(sds[i20]), f0(Math.sqrt(kf.Pf[i20])), f0(sds[NYR - 1]), f0(Math.sqrt(kf.Pf[NYR - 1])))]);
  return { s1897: as[i97], f1897: kf.af[i97], s1900: as[i00], f1900: kf.af[i00], sd_s: sds[i20], sd_f: Math.sqrt(kf.Pf[i20]) };
}

// ================================================================= EXTRA. the prediction-error decomposition
function blkWl() {
  const q = qOf('wl-q'), yr = +$('wl-y').value; set('wl-q', qTxt(q)); set('wl-y', yr);
  const kf = nile(q).kf, x = YRS.slice(1), lt = x.map(y => { const i = yi(y); return -0.5 * (Math.log(2 * Math.PI) + Math.log(kf.F[i]) + kf.v[i] ** 2 / kf.F[i]); }), cl = cumsum(lt);
  const G = grid('wl-chart', [{ r: 0, c: 0, t: 'contribution of each year to log L', xt: 'year' }, { r: 0, c: 1, t: 'cumulative log L', xt: 'year' }], { cols: 2, h: 360, legendRows: 1 });
  const [a, b] = G.ax, j = argmin(lt);
  plot('wl-chart', [bar(x, lt, a, x.map(y => (y === yr ? K.red : K.blue)), ''), ln(x, cl, b, K.blue, `q = ${q.toFixed(3)}: log L = ${cl[cl.length - 1].toFixed(2)}`),
    mk([yr], [cl[yr - 1872]], b, K.red, `year ${yr}`, { marker: { size: 9, color: K.red } })], G.lay);
  const i = yi(yr);
  steps('wl-ws', [T.wls1(yr, f1(NY[i]), f1(kf.ap[i]), f1(kf.v[i]), f0(kf.F[i])), T.wls2(yr, f3(Math.log(2 * Math.PI)), f3(Math.log(kf.F[i])), f3(kf.v[i] ** 2 / kf.F[i]), f3(lt[yr - 1872])),
    T.wls3(x.length, f2(kf.ll), f0(kf.F[0]))]);
  out('wl-out', [T.wl1(qTxt(q), f2(kf.ll), kf.nobs), T.wl2(x[j], f2(lt[j]))]);
  return { ll: kf.ll, worst: x[j] };
}

// ================================================================= 5. profile likelihood and the choice of q
function blkLk() {
  const q = qOf('lk-q'); set('lk-q', qTxt(q));
  const qs = memo('lkq', () => range(-4, 1, 5 / 119).map(e => 10 ** e)), pl = memo('lkp', () => qs.map(x => profile(NY, x).ll)), mx = Math.max(...pl);
  const G = grid('lk-chart', [{ r: 0, c: 0, xt: 'signal-to-noise ratio q = σ²_η/σ²_ε', x: LOG, y: { range: [mx - 25, mx + 3] } }, { r: 0, c: 1, xt: 'year' }], { cols: 2, h: 380, legendRows: 3 });
  const [a, b] = G.ax, P = profile(NY, q), Pm = profile(NY, QML);
  const lv = x => Array.from(llf(NY, 1, x).af).slice(1);
  plot('lk-chart', [ln(qs, pl, a, K.blue, 'profile log-likelihood', { line: { width: 1.8 } }), vline(a, QML, mx - 25, mx + 3, K.red, `ML: q = ${QML.toFixed(3)}`, { line: { dash: 'dash', width: 1.2 } }),
    mk([q], [P.ll], a, K.purple, `chosen q = ${q < 0.01 ? q : q.toFixed(3)}`, { marker: { size: 10, color: K.purple } }),
    mk(YRS, NY, b, K.blue, 'annual flow', { marker: { size: 4, color: K.blue } }), ln(YRS.slice(1), lv(0.001), b, K.green, 'filtered level, q = 0.001', { line: { width: 1.6 } }),
    ln(YRS.slice(1), lv(q), b, K.purple, `filtered level, q = ${q < 0.01 ? q : q.toFixed(3)}`, { line: { width: 1.6 } }), ln(YRS.slice(1), lv(1), b, K.amber, 'filtered level, q = 1', { line: { width: 1.6 } })], G.lay);
  out('lk-out', [T.lk1(f3(QML), f0(Pm.s2), f2(Pm.ll)), T.lk2(qTxt(q), f2(P.ll), f1(2 * (Pm.ll - P.ll)), f0(P.s2))]);
  return { ll: P.ll, lr: 2 * (Pm.ll - P.ll), s2: Pm.s2, llm: Pm.ll };
}

// ================================================================= 6. diagnostics of the Nile model
function blkDg() {
  const q = qOf('dg-q'); set('dg-q', qTxt(q));
  const S = nile(q), kf = S.kf, e = YRS.slice(1).map(y => kf.v[yi(y)] / Math.sqrt(kf.F[yi(y)])), lev = Array.from(S.ds.lev).slice(0, -1), obs = Array.from(S.ds.obs);
  const G = grid('dg-chart', [{ r: 0, c: 0, xt: 'year' }, { r: 0, c: 1, xt: 'year (shock between year t and t + 1)' }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax;
  plot('dg-chart', [bar(YRS.slice(1), e, a, K.blue, 'standardised prediction error e_t'), hline(a, [1872, 1970], 1.96, K.red, '+/- 1.96', { line: { dash: 'dash', width: 1 } }), hline(a, [1872, 1970], -1.96, K.red, '', { line: { dash: 'dash', width: 1 } }),
    bar(YRS.slice(0, -1), lev, b, K.green, 'auxiliary residual of the level'), hline(b, [1871, 1969], 1.96, K.red, '', { line: { dash: 'dash', width: 1 } }), hline(b, [1871, 1969], -1.96, K.red, '', { line: { dash: 'dash', width: 1 } })], G.lay);
  const lb = window.TSAI.ljungBox(e, 10), m = mean(e), c = e.map(v => v - m), m2_ = mean(c.map(v => v * v)), sk = mean(c.map(v => v ** 3)) / m2_ ** 1.5, ku = mean(c.map(v => v ** 4)) / m2_ ** 2;
  const jb = e.length / 6 * (sk * sk + (ku - 3) ** 2 / 4), kl = argmin(lev), ko = argmin(obs.map(v => (Number.isNaN(v) ? Infinity : v)));
  out('dg-out', [T.dg1(qTxt(q), f1(lb.q), f2(chi2sf(lb.q, 10)), f2(jb), f2(Math.exp(-jb / 2)), e.filter(v => Math.abs(v) > 1.96).length, e.length),
    T.dg2(YRS[kl], f2(lev[kl]), YRS[ko], f2(obs[ko]))]);
  return { lb: lb.q, lbp: chi2sf(lb.q, 10), jb, jbp: Math.exp(-jb / 2), nout: e.filter(v => Math.abs(v) > 1.96).length, lev: [YRS[kl], lev[kl]], obs: [YRS[ko], obs[ko]] };
}

// ================================================================= 7. missing years and forecasts of the Nile
function blkMi() {
  const q = qOf('mi-q'), L = +$('mi-g').value, H = +$('mi-h').value; set('mi-q', qTxt(q)); set('mi-g', L); set('mi-h', H);
  const ym = NY.map((v, i) => { const y = YRS[i]; return L && ((y >= 1891 && y < 1891 + L) || (y >= 1931 && y < 1931 + L)) ? NaN : v; });
  const kf = llf(ym, S2E, q * S2E), sm = rts1(kf), sd = Array.from(sm.Ps, Math.sqrt), as = Array.from(sm.as);
  const ext = [...NY, ...new Array(H).fill(NaN)], fy = range(1971, 1970 + H, 1), k2 = llf(ext, S2E, q * S2E);
  const fp = fy.map(y => k2.ap[y - 1871]), fsd = fy.map(y => Math.sqrt(k2.Pp[y - 1871] + S2E));
  const G = grid('mi-chart', [{ r: 0, c: 0, t: L ? `${L} + ${L} years removed` : 'no year removed', xt: 'year' }, { r: 0, c: 1, t: `forecasts 1971-${1970 + H} as missing values`, xt: 'year' }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax, obsI = range(0, NYR - 1, 1).filter(i => !Number.isNaN(ym[i]));
  G.lay.shapes = L ? [[1891, 1890 + L], [1931, 1930 + L]].map(([x0, x1]) => ({ type: 'rect', xref: a.xa, yref: a.yd, x0: x0 - 0.5, x1: x1 + 0.5, y0: 0, y1: 1, fillcolor: `rgba(${RGB[K.amber]},0.12)`, line: { width: 0 }, layer: 'below' })) : [];
  plot('mi-chart', [band(YRS, as.map((v, i) => v - Z90 * sd[i]), as.map((v, i) => v + Z90 * sd[i]), a, K.red, 0.15, '90% band'), mk(obsI.map(i => YRS[i]), obsI.map(i => NY[i]), a, K.blue, 'observed flow', { marker: { size: 4, color: K.blue } }),
    ln(YRS, as, a, K.red, 'smoothed level', { line: { width: 2 } }), ...(L ? [leg('missing years', null, { color: `rgba(${RGB[K.amber]},0.4)`, symbol: 'square', size: 12 })] : []),
    mk(YRS, NY, b, K.blue, '', { marker: { size: 4, color: K.blue } }), band(fy, fp.map((v, i) => v - Z90 * fsd[i]), fp.map((v, i) => v + Z90 * fsd[i]), b, K.green, 0.15, '90% forecast interval for y'),
    ln(fy, fp, b, K.green, 'forecast', { line: { width: 2 } })], G.lay);
  const full = nile(q).sm, ym_ = L ? 1891 + Math.floor((L - 1) / 2) : 1900;
  out('mi-out', [L ? T.mi1(L, ym_, f0(sd[yi(ym_)]), f0(Math.sqrt(full.Ps[yi(ym_)])), ym.filter(Number.isNaN).length) : T.mi0(f0(Math.sqrt(full.Ps[yi(1900)]))),
    T.mi2(qTxt(q), f0(fp[0]), f0(fsd[0]), H, f0(fsd[H - 1]))]);
  return { sdGap: sd[yi(ym_)], f1: fp[0], sd1: fsd[0], sdH: fsd[H - 1] };
}

// ================================================================= EXTRA. the HP filter is a state space smoother
function blkWh() {
  const lam = LAMS[+$('wh-l').value]; set('wh-l', lam);
  const y0 = RY[0], yc = RY.map(v => v - y0), hp = hpOf(RY, lam);
  const ks = memo('whk|' + lam, () => { const kf = kf2(yc, () => [1, 0], [[1, 1], [0, 1]], 1, [[0, 0], [0, 1 / lam]], [0, 0], [[1e7, 0], [0, 1e7]], 0); return rts2(kf).as.map(a => a[0] + y0); });
  const dmax = Math.max(...ks.map((v, i) => Math.abs(v - hp.tr[i])));
  const G = grid('wh-chart', [{ r: 0, c: 0, t: `trend, λ = ${lam}`, yt: '100 × log GDP', x: DATE }, { r: 0, c: 1, t: 'gap y_t − trend (%)', x: DATE }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax;
  plot('wh-chart', [ln(RQ, RY, a, K.blue, '100 × log real GDP', { line: { width: 1.2 } }), ln(RQ, hp.tr, a, K.green, 'HP trend (penalised least squares)', { line: { width: 2.4 } }),
    ln(RQ, ks, a, K.red, 'Kalman smoother, smooth trend, σ²_ε = 1, σ²_ζ = 1/λ', { line: { width: 1.3, dash: 'dot' } }), ln(RQ, hp.gap, b, K.green, 'HP gap', { line: { width: 1.6 } }),
    hline(b, [RQ[0], RQ[NRO - 1]], 0, K.navy, '', { line: { width: 0.6 } })], G.lay);
  steps('wh-ws', [T.whs1(lam, nf(1 / lam, lam >= 1000 ? 6 : 4)), T.whs2(), T.whs3(dmax < 5e-5 ? T.lt + nf(0.0001, 4) : nf(dmax, 4))]);
  out('wh-out', [T.wh1(lam, f1(sdp(hp.gap)), f1(Math.max(...hp.gap)), T.q(RQ[argmax(hp.gap)]))]);
  return { dmax, hp };
}

// ================================================================= 8. Romanian GDP: trend
function blkRt() {
  const lam = LAMS[+$('rt-l').value]; set('rt-l', lam);
  const hp = hpOf(RY, lam), G = grid('rt-chart', [{ r: 0, c: 0, yt: '100 × log, million EUR (2010 prices)', x: DATE }], { cols: 1, h: 380, legendRows: 1, ml: 60 }), a = G.ax[0];
  plot('rt-chart', [ln(RQ, RY, a, K.blue, '100 × log real GDP', { line: { width: 1.4 } }), ln(RQ, RO.trend, a, K.red, 'UC smooth trend (smoothed)', { line: { width: 2 } }),
    ln(RQ, hp.tr, a, K.green, `HP trend, λ = ${lam}`, { line: { width: 1.4, dash: 'dash' } })], G.lay);
  const Tn = NB.trend, P = RO.par, gt = RO.trend.slice(1).map((v, i) => 4 * (v - RO.trend[i]));
  out('rt-out', [T.rt1(f1(4 * (RY[NRO - 1] - RY[0]) / (NRO - 1)), f1(Math.max(...gt)), T.q(RQ[1 + argmax(gt)]), f1(Math.min(...gt)), T.q(RQ[1 + argmin(gt)])),
    T.rt2(f3(P.sigma2_trend), f2(P.ar_L1), f2(P.ar_L2), f2(P.sigma2_ar), f0(P.sigma2_ar / P.sigma2_trend)),
    T.rt3(lam, f2(Math.max(...RY.map((v, i) => Math.abs(hp.tr[i] - RO.trend[i])))))]);
  return { g: 4 * (RY[NRO - 1] - RY[0]) / (NRO - 1), ratio: P.sigma2_ar / P.sigma2_trend, Tn };
}

// ================================================================= 9. the Romanian output gap: three methods
const inW = (d, a, b) => d >= a && d <= b;
function blkOg() {
  const h = +$('og-h').value, p = +$('og-p').value, lam = LAMS[+$('og-l').value]; set('og-h', h); set('og-p', p); set('og-l', lam);
  const hp = hpOf(RY, lam), hm = memo(`og|${h}|${p}`, () => hamFilter(RY, h, p)), uc = RO.cyc;
  const G = grid('og-chart', [{ r: 0, c: 0, yt: '% of trend output', x: DATE }], { cols: 1, h: 380, legendRows: 2 }), a = G.ax[0];
  plot('og-chart', [hline(a, [RQ[0], RQ[NRO - 1]], 0, K.navy, '', { line: { width: 0.6 } }), ln(RQ, uc, a, K.red, 'UC cycle (smooth trend + AR(2))', { line: { width: 2 } }),
    ln(RQ, hp.gap, a, K.green, `HP gap, λ = ${lam}`, { line: { width: 1.4, dash: 'dash' } }), ln(RQ, hm.cyc, a, K.purple, `Hamilton filter (h = ${h}, p = ${p})`, { line: { width: 1.4 } })], G.lay);
  const ok = range(0, NRO - 1, 1).filter(i => hm.cyc[i] !== null), pk = arr => { const w = range(0, NRO - 1, 1).filter(i => inW(RQ[i], '2007-01-01', '2009-10-01') && arr[i] !== null); const j = w.reduce((b, i) => (arr[i] > arr[b] ? i : b), w[0]); return [arr[j], RQ[j]]; };
  const S = { sdU: sdp(uc), sdH: sdp(hp.gap), sdM: sdp(ok.map(i => hm.cyc[i])), cUH: corr(uc, hp.gap), cUM: corr(ok.map(i => uc[i]), ok.map(i => hm.cyc[i])), cHM: corr(ok.map(i => hp.gap[i]), ok.map(i => hm.cyc[i])),
    pU: pk(uc), pH: pk(hp.gap), pM: pk(hm.cyc), bs: sum(hm.b.slice(1)), lost: ok[0] };
  const i20 = RQ.indexOf('2020-04-01');
  out('og-out', [T.og1(f1(S.pU[0]), T.q(S.pU[1]), f1(S.pH[0]), f1(S.pM[0]), f1(uc[i20]), f1(hp.gap[i20]), f1(hm.cyc[i20])), T.og2(f2(S.cUH), f2(S.cUM), f2(S.cHM)),
    T.og3(f1(S.sdU), f1(S.sdH), f1(S.sdM), p, f2(S.bs), S.lost), T.og4(T.q(RQ[NRO - 1]), f1(uc[NRO - 1]), f1(hp.gap[NRO - 1]), f1(hm.cyc[NRO - 1]))]);
  return S;
}

// ================================================================= 10. real time against hindsight
function hpRealtime(lam) {
  return memo('rtm|' + lam, () => { const o = new Array(NRO).fill(null); for (let t = 40; t <= NRO; t++) { const g = hpTrend(RY.slice(0, t), lam); o[t - 1] = RY[t - 1] - g[t - 1]; } return o; });
}
function blkRl() {
  const e = $('rl-e').value, lam = LAMS[+$('rl-l').value]; set('rl-l', lam);
  const V = RO.vint[e], n = V.n, q = RQ.slice(0, n), fin = hpOf(RY.slice(0, n), lam).gap, rt = hpRealtime(lam).slice(0, n);
  const G = grid('rl-chart', [{ r: 0, c: 0, t: 'Hodrick-Prescott filter', x: DATE }, { r: 0, c: 1, t: 'unobserved-components model', x: DATE }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax;
  plot('rl-chart', [hline(a, [q[0], q[n - 1]], 0, K.navy, '', { line: { width: 0.6 } }), ln(q, fin, a, K.green, 'HP gap, final (two-sided)', { line: { width: 2 } }),
    ln(q, rt, a, K.orange, 'HP gap, real time (one-sided)', { line: { width: 1.4 } }), hline(b, [q[0], q[n - 1]], 0, K.navy, '', { line: { width: 0.6 } }),
    ln(q, V.cs, b, K.red, 'UC cycle, smoothed (two-sided)', { line: { width: 2 } }), ln(q, V.cf, b, K.blue, 'UC cycle, filtered (one-sided)', { line: { width: 1.4 } })],
    Object.assign(G.lay, { yaxis2: Object.assign(G.lay.yaxis2, { matches: 'y' }) }));
  const ok = range(39, n - 1, 1), i8 = RQ.indexOf('2008-07-01'), rms = (A, B) => Math.sqrt(mean(ok.map(i => (A[i] - B[i]) ** 2)));
  const S = { hpRt: rt[i8], hpFin: fin[i8], ucF: V.cf[i8], ucS: V.cs[i8], revH: rms(rt, fin), revU: rms(V.cf, V.cs), cH: corr(ok.map(i => rt[i]), ok.map(i => fin[i])), cU: corr(ok.map(i => V.cf[i]), ok.map(i => V.cs[i])) };
  out('rl-out', [T.rl1(T.q(q[n - 1]), f1(S.hpRt), f1(S.hpFin), f1(S.ucF), f1(S.ucS)), T.rl2(T.q(q[n - 1]), f1(S.revH), f1(S.revU), f2(S.cH), f2(S.cU))]);
  return S;
}

// ================================================================= 11. a time-varying beta
const TVS = [0, 0.1, 0.3, 1, 3, 10], TV = NB.tvp;
function tvpRun(m) {
  return memo('tv|' + m, () => {
    const x = D.tvp.x, y = D.tvp.y, kf = kf2(y, t => [1, x[t]], [[1, 0], [0, 1]], TV.s2_eps, [[0, 0], [0, m * TV.s2_beta]], [0, 0], [[1e4, 0], [0, 1e4]], 2), sm = rts2(kf);
    return { ll: kf.ll, b: sm.as.map(a => a[1]), sd: sm.Ps.map(P => Math.sqrt(P[1][1])) };
  });
}
function blkTv() {
  const m = TVS[+$('tv-s').value], w = +$('tv-w').value; set('tv-s', T.tvS(nf(m, m < 1 && m > 0 ? 1 : 0), m === 1)); set('tv-w', w);
  const R = tvpRun(m), x = D.tvp.x, y = D.tvp.y, d = D.tvp.d, n = d.length;
  const roll = memo('tvr|' + w, () => d.map((_, t) => { if (t < w - 1) return null; const xs = x.slice(t - w + 1, t + 1), ys = y.slice(t - w + 1, t + 1), mx = mean(xs), my = mean(ys); let s = 0, s2 = 0; for (let i = 0; i < w; i++) { s += (xs[i] - mx) * (ys[i] - my); s2 += (xs[i] - mx) ** 2; } return s / s2; }));
  const ols = memo('tvols', () => { const mx = mean(x), my = mean(y); let s = 0, s2 = 0; for (let i = 0; i < n; i++) { s += (x[i] - mx) * (y[i] - my); s2 += (x[i] - mx) ** 2; } return s / s2; });
  const G = grid('tv-chart', [{ r: 0, c: 0, yt: 'beta of the BET on the Euro Stoxx 50', x: DATE }], { cols: 1, h: 380, legendRows: 2 }), a = G.ax[0];
  plot('tv-chart', [band(d, R.b.map((b, i) => b - Z90 * R.sd[i]), R.b.map((b, i) => b + Z90 * R.sd[i]), a, K.red, 0.15, '90% band'), ln(d, roll, a, K.blue, `rolling OLS, ${w} weeks`, { line: { width: 1 } }),
    ln(d, R.b, a, K.red, 'Kalman smoothed β_t', { line: { width: 2 } }), hline(a, [d[0], d[n - 1]], ols, K.green, `constant OLS β = ${ols.toFixed(2)}`, { line: { dash: 'dash', width: 1.3 } })], G.lay);
  const imx = argmax(R.b), imn = argmin(R.b), rr = roll.filter(v => v !== null), dr = rr.slice(1).map((v, i) => v - rr[i]), db = R.b.slice(1).map((v, i) => v - R.b[i]);
  out('tv-out', [T.tv1(f2(ols), f2(R.b[imx]), T.my(d[imx]), f2(R.b[imn]), T.my(d[imn])), T.tv2(f1(2 * (R.ll - TV.loglik_const)), f3(Math.sqrt(m * TV.s2_beta)), m === 1),
    T.tv3(f2(R.b[n - 1]), f2(R.sd[n - 1]), w, f3(sdp(dr)), f3(sdp(db)))]);
  return { ols, bmax: R.b[imx], dmax: d[imx], bmin: R.b[imn], dmin: d[imn], lr: 2 * (R.ll - TV.loglik_const), last: R.b[n - 1], sdl: R.sd[n - 1] };
}

// ================================================================= 12. a common factor of US activity
function blkDf() {
  const k = +$('df-k').value, F = D.dfm.fac[String(k)], n = F.s.length, md = range(0, n - 1, 1).map(i => isoM(D.dfm.m0, i)), rec = D.dfm.rec;
  const G = grid('df-chart', [{ r: 0, c: 0, cs: 2, yt: 'standardised units (clipped)', x: DATE, y: { range: [-6, 4] } }, { r: 1, c: 0, cs: 2, t: 'last 24 months', x: DATE }], { cols: 2, h: 520, legendRows: 2, gyPx: 70 });
  const [a, b] = G.ax, l24 = range(n - 24, n - 1, 1), miss = range(n - k, n - 1, 1);
  G.lay.shapes = shade(a, md, rec, d => isoM(d, 1));
  plot('df-chart', [ln(md, F.s, a, K.blue, 'common factor (smoothed)', { line: { width: 1 } }), hline(a, [md[0], md[n - 1]], 0, K.navy, '', { line: { width: 0.6 } }),
    mk(miss.map(i => md[i]), miss.map(i => F.s[i]), a, K.red, k ? 'months without income and sales' : '', { marker: { size: 7, color: K.red } }), legShade('NBER recessions'),
    ln(l24.map(i => md[i]), l24.map(i => F.s[i]), b, K.blue, '', { mode: 'lines+markers', marker: { size: 5 } }), ln(l24.map(i => md[i]), l24.map(i => F.f[i]), b, K.teal, 'filtered (one-sided)', { line: { dash: 'dot', width: 1.3 } }),
    mk(miss.map(i => md[i]), miss.map(i => F.s[i]), b, K.red, '', { marker: { size: 9, color: K.red } }), hline(b, [md[n - 24], md[n - 1]], 0, K.navy, '', { line: { width: 0.6 } })], G.lay);
  const Ld = NB.dfm.loadings, r1 = F.s.filter((_, i) => rec[i] === 1), r0 = F.s.filter((_, i) => rec[i] === 0), Dm = NB.dfm, tod = D.dfm.today;
  out('df-out', [T.df1(f2(Ld.PAYEMS), f2(Ld.INDPRO), f2(Ld.CMRMTSPL), f2(Ld.W875RX1), f2(Dm.phi1), f2(Dm.phi2)),
    T.df2(f1(mean(r1)), f2(mean(r0)), pc(mean(r1.map(v => +(v < 0)))), f0(Math.min(...F.s)), T.my(md[argmin(F.s)])),
    T.df3(k, T.my(md[n - 1]), f2(F.s[n - 1]), f1(Dm.last), f2(tod.last), T.dmy(D.dfm.downloaded))]);
  return { last: F.s[n - 1], mrec: mean(r1), mexp: mean(r0) };
}

// ================================================================= EXTRA. transition matrix, durations, ergodic probabilities
function blkWd() {
  const p11 = +$('wd-p').value, p22 = +$('wd-q').value, h = +$('wd-h').value; set('wd-p', f2(p11)); set('wd-q', f2(p22)); set('wd-h', h);
  const dd = range(1, 30, 1), hh = range(0, 30, 1), pi1 = (1 - p22) / (2 - p11 - p22), lamb = p11 + p22 - 1;
  const G = grid('wd-chart', [{ r: 0, c: 0, t: 'duration of a spell: Pr(D = d)', xt: 'd (periods)' }, { r: 0, c: 1, t: 'Pr(S_t+h = 1 | S_t)', xt: 'h', y: { range: [0, 1] } }], { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax, ph = (i, k) => (i === 1 ? pi1 + (1 - pi1) * lamb ** k : pi1 - pi1 * lamb ** k);
  plot('wd-chart', [bar(dd, dd.map(d => p11 ** (d - 1) * (1 - p11)), a, K.red, 'regime 1'), bar(dd, dd.map(d => p22 ** (d - 1) * (1 - p22)), a, K.blue, 'regime 2'),
    ln(hh, hh.map(k => ph(1, k)), b, K.red, 'from S_t = 1', { mode: 'lines+markers', marker: { size: 4 } }), ln(hh, hh.map(k => ph(2, k)), b, K.blue, 'from S_t = 2', { mode: 'lines+markers', marker: { size: 4 } }),
    hline(b, [0, 30], pi1, K.navy, `ergodic π_1 = ${pi1.toFixed(3)}`, { line: { dash: 'dot', width: 1.2 } })], Object.assign(G.lay, { barmode: 'group' }));
  steps('wd-ws', [T.wds1(f2(p11), f2(1 - p11), f2(1 / (1 - p11)), f2(p22), f2(1 / (1 - p22))), T.wds2(f2(1 - p22), f2(2 - p11 - p22), f3(pi1)),
    T.wds3(h, f3(ph(1, h)), f3(1 - ph(1, h)), f3(ph(2, h)), f3(1 - ph(2, h)), f3(lamb))]);
  out('wd-out', [T.wd1(f2(p11), f1(1 / (1 - p11)), f2(p22), f1(1 / (1 - p22)), f3(pi1))]);
  return { d1: 1 / (1 - p11), d2: 1 / (1 - p22), pi1 };
}

// ================================================================= EXTRA. the Hamilton filter, one quarter
const usModel = () => memo('usm', () => { const X = usX(); return Object.assign(X, hamilton(X.y, [EXT.const_1, EXT.const_2], [EXT.sigma2, EXT.sigma2], EXT.p11, EXT.p22)); });
const WFQ = range(0, 15, 1).map(k => isoQ('2007-01-01', k));
function blkWf() {
  const q = WFQ[+$('wf-t').value]; set('wf-t', T.q(q));
  const M = usModel(), i = M.q.indexOf(q), y = M.y[i], pr = M.pred[i], f = M.dens[i], fl = M.filt[i], s = M.sm[i], prev = M.filt[i - 1], ys = range(-3, 3.5, 0.05);
  const sg = Math.sqrt(EXT.sigma2), G = grid('wf-chart', [{ r: 0, c: 0, t: `regime densities and y_t, ${qlab(q)}`, xt: 'growth, % q/q' }, { r: 0, c: 1, t: 'Pr(recession regime)', y: { range: [0, 1] } }],
    { cols: 2, h: 380, legendRows: 2 });
  const [a, b] = G.ax;
  plot('wf-chart', [ln(ys, ys.map(v => phi((v - EXT.const_1) / sg) / sg), a, K.red, 'f_1: recession regime'), ln(ys, ys.map(v => phi((v - EXT.const_2) / sg) / sg), a, K.blue, 'f_2: expansion regime'),
    vline(a, y, 0, 0.55, K.navy, `y_t = ${y.toFixed(2)}`, { line: { width: 1.6, dash: 'dash' } }),
    bar(['predicted', 'filtered', 'smoothed'], [pr[0], fl[0], s[0]], b, [K.teal, K.blue, K.red], '', { text: [pr[0], fl[0], s[0]].map(v => v.toFixed(3)), textposition: 'outside', cliponaxis: false })], G.lay);
  steps('wf-ws', [T.wfs1(T.q(M.q[i - 1]), f3(prev[0]), f3(prev[1]), f3(EXT.p11), f3(1 - EXT.p22), f3(pr[0])),
    T.wfs2(f2(y), f2(EXT.const_1), f2(EXT.const_2), f2(sg), f3(f[0]), f3(f[1])), T.wfs3(f3(pr[0] * f[0]), f3(pr[1] * f[1]), f3(fl[0]))]);
  out('wf-out', [T.wf1(T.q(q), f2(y), f3(pr[0]), f3(fl[0]), f3(s[0]))]);
  return { pr: pr[0], fl: fl[0], sm: s[0] };
}

// ================================================================= EXTRA. Hamilton's specification: the two maxima
function blkMt() {
  const m = $('mt-m').value, c = +$('mt-c').value; set('mt-c', f2(c));
  const p = US.maxima[m], i0 = uIdx(US.ham0), q = p.map((_, k) => UQ[i0 + k]), rec = q.map((_, k) => UREC[i0 + k]);
  const G = grid('mt-chart', [{ r: 0, c: 0, t: `smoothed Pr(regime 1), maximum ${m}`, x: DATE, y: { range: [-0.02, 1.02] } }], { cols: 1, h: 330, legendRows: 1 }), a = G.ax[0];
  G.lay.shapes = shade(a, q, rec, nextQ);
  plot('mt-chart', [ln(q, p, a, m === 'A' ? K.red : K.purple, `maximum ${m}: smoothed probability of regime 1`, { line: { width: 1.6 } }), legShade('NBER recessions')], G.lay);
  const conc = P => { const cls = P.map(v => +(v > c)), nr = sum(rec), ne = rec.length - nr; return { hit: sum(cls.map((v, k) => v * rec[k])) / nr, fa: sum(cls.map((v, k) => v * (1 - rec[k]))) / ne }; };
  const A = NB.msus.ham, B = NB.msus.ham_b, cA = conc(US.maxima.A), cB = conc(US.maxima.B);
  const rows = [[T.L.means, txt(`${f2(A.const_1)} / ${f2(A.const_2)}`), txt(`${f2(B.const_1)} / ${f2(B.const_2)}`)], ['p<sub>11</sub>, p<sub>22</sub>', txt(`${f3(A.p11)}${T.sep}${f3(A.p22)}`), txt(`${f3(B.p11)}${T.sep}${f3(B.p22)}`)],
    [T.L.dur, txt(`${f1(1 / (1 - A.p11))} / ${f1(1 / (1 - A.p22))}`), txt(`${f1(1 / (1 - B.p11))} / ${f1(1 / (1 - B.p22))}`)], ['log L', txt(f2(A.loglik)), txt(f2(B.loglik), true)],
    [T.L.found(f2(c)), txt(pc(cA.hit), true), txt(pc(cB.hit))], [T.L.falseA(f2(c)), txt(pc(cA.fa)), txt(pc(cB.fa))]];
  table('mt-tb', [T.L.msar, H_(T.L.maxA), H_(T.L.maxB)], rows, { cap: T.mtCap });
  const cc = conc(p);
  out('mt-out', [T.mt1(m, sum(p.map(v => +(v > 0.5))), p.length, pc(cc.hit), sum(rec))]);
  return { hitA: cA.hit, hitB: cB.hit, n1: sum(p.map(v => +(v > 0.5))) };
}

// ================================================================= 13. US recessions from GDP growth
const MUP = [0.3, 0.4, 0.5, 0.6, EXT.p11, 0.75, 0.8, 0.85, 0.9, 0.95];
function blkMu() {
  const p11 = MUP[+$('mu-p').value]; set('mu-p', f3(p11));
  const X = usXA(), H = memo('mu|' + p11, () => hamilton(X.y, [EXT.const_1, EXT.const_2], [EXT.sigma2, EXT.sigma2], p11, EXT.p22)), sp = H.p1.sm;
  const G = grid('mu-chart', [{ r: 0, c: 0, x: DATE, y: { range: [-4, 4] } }, { r: 1, c: 0, x: DATE, y: { range: [-0.02, 1.02] } }], { cols: 1, h: 460, legendRows: 3, gyPx: 40 });
  const [a, b] = G.ax, sh = q => shade(q, X.q, X.rec, nextQ), mfit = sp.map(p => p * EXT.const_1 + (1 - p) * EXT.const_2);
  G.lay.shapes = [...sh(a), ...sh(b)];
  plot('mu-chart', [ln(X.q, X.y, a, K.blue, 'real GDP growth, % q/q', { line: { width: 0.9 } }), ln(X.q, mfit, a, K.orange, 'smoothed regime mean', { line: { width: 1.4 } }),
    hline(a, [X.q[0], X.q[X.q.length - 1]], 0, K.navy, '', { line: { width: 0.6 } }), legShade('NBER recessions'),
    ln(X.q, sp, b, K.red, 'smoothed Pr(recession regime)', { line: { width: 1.6 } }), vline(b, '2020-01-01', -0.02, 1.02, K.purple, 'end of the estimation sample', { line: { dash: 'dash', width: 1.1 } })], G.lay);
  const iE = X.q.indexOf('2019-10-01'), C = concord(sp.slice(0, iE + 1), X.rec.slice(0, iE + 1)), i20 = X.q.indexOf('2020-04-01'), i21 = X.q.indexOf('2021-01-01');
  out('mu-out', [T.mu1(f2(EXT.const_1), f2(EXT.const_2), f2(Math.sqrt(EXT.sigma2)), f3(p11), f3(EXT.p22)), T.mu2(f1(1 / (1 - p11)), f1(1 / (1 - EXT.p22)), pc(H.pi1)),
    T.mu3(pc(C.concord, 1), pc(C.hit, 1), pc(C.false, 1)), T.mu4(f2(sp[i20]), f2(Math.max(...sp.slice(i21))))]);
  return { C, p20: sp[i20], mx: Math.max(...sp.slice(i21)), pi1: H.pi1 };
}

// ================================================================= 14. filtered against smoothed probabilities
function blkMf() {
  const [w0, w1] = $('mf-w').value.split('|'), late = w1 > '2019-10-01';
  const X = late ? usXA() : usX(), H = late ? memo('mfx', () => hamilton(X.y, [EXT.const_1, EXT.const_2], [EXT.sigma2, EXT.sigma2], EXT.p11, EXT.p22)) : usModel();
  const ix = range(0, X.q.length - 1, 1).filter(i => inW(X.q[i], w0, w1)), q = ix.map(i => X.q[i]), rec = ix.map(i => X.rec[i]);
  const G = grid('mf-chart', [{ r: 0, c: 0, x: DATE, y: { range: [-0.02, 1.02] } }], { cols: 1, h: 360, legendRows: 2 }), a = G.ax[0];
  G.lay.shapes = shade(a, q, rec, nextQ);
  const P = k => ix.map(i => H.p1[k][i]);
  plot('mf-chart', [ln(q, P('pred'), a, K.teal, 'predicted Pr(S_t = 1 | y_1, ..., y_t-1)', { mode: 'lines+markers', line: { dash: 'dot', width: 1.2 }, marker: { size: 4 } }),
    ln(q, P('filt'), a, K.blue, 'filtered Pr(S_t = 1 | y_1, ..., y_t)', { mode: 'lines+markers', marker: { size: 6 } }), ln(q, P('sm'), a, K.red, 'smoothed Pr(S_t = 1 | y_1, ..., y_n)', { line: { width: 2 } }),
    legShade('NBER recessions')], G.lay);
  const first = arr => { const j = arr.findIndex(v => v > 0.5); return j < 0 ? null : q[j]; }, r0 = rec.indexOf(1);
  const fq = first(P('filt')), sq = first(P('sm'));
  const iE = usX().q.length - 1, M0 = usModel(), Cf = concord(M0.p1.filt.slice(0, iE + 1), usX().rec);
  out('mf-out', [T.mf1(qlab(w0).slice(0, 4), qlab(w1).slice(0, 4), r0 < 0 ? null : T.q(q[r0]), fq ? T.q(fq) : null, sq ? T.q(sq) : null),
    r0 < 0 ? T.mf0 : T.mf2(T.q(q[r0]), f2(P('filt')[r0]), f2(P('sm')[r0])), T.mf3(pc(Cf.concord), pc(Cf.hit), late)]);
  return { fq, sq, Cf };
}

// ================================================================= EXTRA. checks: which regimes did the model find?
function blkMc() {
  const e = $('mc-e').value, F = US.fits[e], i0 = uIdx(US.x0), q = F.sv_p.map((_, k) => UQ[i0 + k]), rec = q.map((_, k) => UREC[i0 + k]), y = q.map((_, k) => UG[i0 + k]);
  const G = grid('mc-chart', [{ r: 0, c: 0, t: 'switching variance: Pr(high-variance regime)', x: DATE, y: { range: [-0.02, 1.02] } }, { r: 1, c: 0, t: 'switching mean: Pr(low-mean regime)', x: DATE, y: { range: [-0.02, 1.02] } }],
    { cols: 1, h: 470, legendRows: 2, gyPx: 50 });
  const [a, b] = G.ax;
  G.lay.shapes = [...shade(a, q, rec, nextQ), ...shade(b, q, rec, nextQ)];
  plot('mc-chart', [ln(q, F.sv_p, a, K.purple, 'high-variance regime', { line: { width: 1.5 } }), ln(q, F.mean_p, b, K.red, 'low-mean regime', { line: { width: 1.5 } }), legShade('NBER recessions')], G.lay);
  const sv = F.sv, mn = F.mean, calm = q.findIndex((_, k) => F.sv_p[k] < 0.5), C = concord(F.sv_p, rec), n1 = sum(F.mean_p.map(v => +(v > 0.5)));
  const lowQ = q.filter((_, k) => F.mean_p[k] > 0.5);
  out('mc-out', [T.mc1(T.q(e), f2(Math.sqrt(sv.sigma2_2)), f2(Math.sqrt(sv.sigma2_1)), f0(sv.dur2), f0(sv.dur1), calm < 0 ? '–' : T.q(q[calm]), pc(C.concord), pc(C.false)),
    T.mc2(f2(mn.const_1), f2(mn.const_2), f1(mn.dur1), f1(mn.dur2), n1, n1 === 1 ? T.q(lowQ[0]) : '')]);
  return { sd2: Math.sqrt(sv.sigma2_2), sd1: Math.sqrt(sv.sigma2_1), switch: calm < 0 ? null : q[calm], C, n1, dur2: mn.dur2 };
}

// ================================================================= 15. Romanian GDP growth regimes
const MR = NB.msro, MRP = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, MR.p11, 0.95, 0.98];
function blkMr() {
  const p11 = MRP[+$('mr-p').value]; set('mr-p', f3(p11));
  const H = memo('mr|' + p11, () => hamilton(RG, [MR.const_1, MR.const_2], [MR.sigma2_1, MR.sigma2_2], p11, MR.p22)), sp = H.p1.sm;
  const G = grid('mr-chart', [{ r: 0, c: 0, x: DATE }, { r: 1, c: 0, x: DATE, y: { range: [-0.02, 1.02] } }], { cols: 1, h: 460, legendRows: 2, gyPx: 40 });
  const [a, b] = G.ax, mfit = sp.map(p => p * MR.const_1 + (1 - p) * MR.const_2);
  plot('mr-chart', [bar(RGQ, RG, a, K.blue, 'real GDP growth, % q/q'), ln(RGQ, mfit, a, K.orange, 'smoothed regime mean', { line: { width: 1.6 } }),
    ln(RGQ, sp, b, K.red, 'smoothed Pr(volatile regime)', { line: { width: 1.5, shape: 'hvh' }, fill: 'tozeroy', fillcolor: `rgba(${RGB[K.red]},0.25)` })], G.lay);
  const vol = sp.map(p => p > 0.5), sp_ = []; let c = null;
  vol.forEach((v, i) => { if (v && c === null) c = i; if (!v && c !== null) { sp_.push([c, i - 1]); c = null; } }); if (c !== null) sp_.push([c, vol.length - 1]);
  out('mr-out', [T.mr1(f2(MR.const_2), f1(Math.sqrt(MR.sigma2_2)), f2(MR.const_1), f1(Math.sqrt(MR.sigma2_1))),
    T.mr2(f3(p11), f1(1 / (1 - MR.p22)), f1(1 / (1 - p11)), pc(mean(vol.map(Number))), pc(H.pi1)),
    T.mr3(sp_.map(([i, j]) => T.qq(RGQ[i], RGQ[j])).join(T.sep)), T.mr4(f2(sp[sp.length - 1]))]);
  return { spells: sp_.map(([i, j]) => [RGQ[i], RGQ[j]]), share: mean(vol.map(Number)), last: sp[sp.length - 1] };
}

// ================================================================= 16. calm and turbulent stock markets
const VR = NB.vol, VRP = [0.5, 0.7, 0.8, 0.9, VR.p11, 0.95, 0.98];
function blkVr() {
  const p11 = VRP[+$('vr-p').value]; set('vr-p', f3(p11));
  const r = D.vol.r, d = D.vol.d, H = memo('vr|' + p11, () => hamilton(r, [VR.const_1, VR.const_2], [VR.sigma2_1, VR.sigma2_2], p11, VR.p22)), sp = H.p1.sm;
  const s2t = sp.map(p => p * VR.sigma2_1 + (1 - p) * VR.sigma2_2), vms = s2t.map(v => Math.sqrt(52 * v));
  const G = grid('vr-chart', [{ r: 0, c: 0, x: DATE }, { r: 1, c: 0, yt: '% per year', x: DATE }], { cols: 1, h: 480, legendRows: 3, gyPx: 40 });
  const [a, b] = G.ax, a2 = overlay(G, a, 'y3', '', { range: [0, 1.05] });
  plot('vr-chart', [ln(d, r, a, K.blue, 'weekly log return, %', { line: { width: 0.6 } }),
    ln(d, s2t.map(v => Z90 * Math.sqrt(v)), a, K.orange, '±1.645 × regime volatility', { line: { width: 0.9 } }), ln(d, s2t.map(v => -Z90 * Math.sqrt(v)), a, K.orange, '', { line: { width: 0.9 } }),
    ln(d, sp, a2, K.red, 'smoothed Pr(turbulent regime), right axis', { line: { width: 0 }, fill: 'tozeroy', fillcolor: `rgba(${RGB[K.red]},0.25)` }),
    ln(d, vms, b, K.red, 'Markov switching: √(Σ_k Pr(k) σ²_k), annualised', { line: { width: 1.4 } }), ln(d, D.vol.garch, b, K.green, 'GARCH(1,1) conditional volatility, annualised', { line: { width: 1 } })], G.lay);
  const turb = sp.map(p => +(p > 0.5)), yrs = {}; d.forEach((x, i) => { const y = x.slice(0, 4); (yrs[y] = yrs[y] || []).push(turb[i]); });
  const top = Object.entries(yrs).map(([y, v]) => [y, mean(v)]).sort((u, v) => v[1] - u[1]).slice(0, 3);
  out('vr-out', [T.vr1(f0(Math.sqrt(52 * VR.sigma2_2)), f2(VR.const_2), f0(Math.sqrt(52 * VR.sigma2_1)), f2(VR.const_1)),
    T.vr2(f3(p11), f0(1 / (1 - VR.p22)), f0(1 / (1 - p11)), pc(mean(turb)), pc(H.pi1)), T.vr3(top.map(([y, v]) => `${y} (${pc(v)})`).join(T.sep)),
    T.vr4(f2(corr(vms, D.vol.garch)), f2(VR.garch_a), f2(VR.garch_b))]);
  return { share: mean(turb), top, c: corr(vms, D.vol.garch), pi1: H.pi1 };
}

// ================================================================= 17. regimes imitate memory
function blkRm() {
  const M = +$('rm-m').value, L = +$('rm-l').value; set('rm-m', M); set('rm-l', L);
  const S = D.mem, lags = range(1, L, 1), sim = lags.map((_, k) => mean(S.sims.slice(0, M).map(s => s[k])));
  const G = grid('rm-chart', [{ r: 0, c: 0, xt: 'lag (weeks)', yt: 'ACF of |r|' }], { cols: 1, h: 360, legendRows: 2 }), a = G.ax[0];
  plot('rm-chart', [mk(lags, S.acf.slice(0, L), a, K.blue, 'S&P 500 weekly |r|', { marker: { size: 5, color: K.blue } }), ln(lags, sim, a, K.red, `simulated two-regime model, mean of ${M} path${M === 1 ? '' : 's'}`, { line: { width: 2 } }),
    ln(lags, S.iid.slice(0, L), a, K.green, 'same returns in random order', { line: { width: 1 } }), hline(a, [1, L], 0, K.navy, '', { line: { width: 0.6 } })], G.lay);
  const ds = S.d.slice(0, M), ab = S.ab.slice(0, Math.min(M, 15));
  out('rm-out', [T.rm1(f2(S.acf[0]), f2(S.acf[25]), M, f2(sim[0]), L >= 26 ? f2(sim[25]) : null), T.rm2(f2(NB.mem.d_data), M, f2(mean(ds)), f2(sdp(ds)), f2(NB.mem.d_iid)),
    T.rm3(ab.length, f2(mean(ab)), f2(Math.min(...ab)))]);
  return { s1: sim[0], s26: sim[25], d: mean(ds), dsd: sdp(ds), ab: mean(ab) };
}

// ================================================================= EXTRA. the Markov-switching models of the chapter: parameters and durations
function msModels() {
  const sv = NB.pit.sv;
  return [{ k: 'us', p: EXT, per: 4, ll: EXT.loglik, n: EXT.n, mu: [EXT.const_1, EXT.const_2], sd: [Math.sqrt(EXT.sigma2), Math.sqrt(EXT.sigma2)], c: K.red },
    { k: 'sv', p: sv, per: 4, ll: sv.loglik, n: sv.n, mu: [sv.const_1, sv.const_2], sd: [Math.sqrt(sv.sigma2_1), Math.sqrt(sv.sigma2_2)], c: K.purple },
    { k: 'ro', p: MR, per: 4, ll: MR.loglik, n: MR.n, mu: [MR.const_1, MR.const_2], sd: [Math.sqrt(MR.sigma2_1), Math.sqrt(MR.sigma2_2)], c: K.green },
    { k: 'sp', p: VR, per: 52, ll: VR.loglik, n: VR.n, mu: [VR.const_1, VR.const_2], sd: [Math.sqrt(VR.sigma2_1), Math.sqrt(VR.sigma2_2)], c: K.blue }];
}
function blkMp() {
  const h = +$('mp-h').value, u = $('mp-u').value; set('mp-h', h);
  const Ms = msModels(), dd = range(1, 60, 1);
  const G = grid('mp-chart', [{ r: 0, c: 0, t: 'regime 1: Pr(spell lasts more than d periods) = p11^d', xt: 'd (periods)', y: { range: [0, 1] } },
    { r: 0, c: 1, t: 'regime 2: p22^d', xt: 'd (periods)', y: { range: [0, 1] } }], { cols: 2, h: 400, legendRows: 2 });
  const [a, b] = G.ax, tr = [];
  Ms.forEach(m => { tr.push(ln(dd, dd.map(d => m.p.p11 ** d), a, m.c, T.mname[m.k]), ln(dd, dd.map(d => m.p.p22 ** d), b, m.c, ''),
    mk([h], [m.p.p11 ** h], a, m.c, '', { marker: { size: 9, color: m.c } }), mk([h], [m.p.p22 ** h], b, m.c, '', { marker: { size: 9, color: m.c } })); });
  plot('mp-chart', tr, G.lay);
  const D_ = (p, per) => (u === 'y' ? f1(1 / (1 - p) / per) : f1(1 / (1 - p)));
  const rows = [[T.L.reg1, ...Ms.map(m => txt(T.reg1[m.k]))], [T.L.mu12, ...Ms.map(m => txt(`${f2(m.mu[0])} / ${f2(m.mu[1])}`))], [T.L.sd12, ...Ms.map(m => txt(`${f2(m.sd[0])} / ${f2(m.sd[1])}`))],
    ['p<sub>11</sub> / p<sub>22</sub>', ...Ms.map(m => txt(`${f3(m.p.p11)} / ${f3(m.p.p22)}`))], [T.L.dur12(u), ...Ms.map(m => txt(`${D_(m.p.p11, m.per)} / ${D_(m.p.p22, m.per)}`, true))],
    ['π<sub>1</sub>', ...Ms.map(m => txt(f2(m.p.ergodic1)))], ['log L', ...Ms.map(m => txt(f2(m.ll)))], [T.L.nobs, ...Ms.map(m => txt(m.n))]];
  table('mp-tb', [T.L.model, ...Ms.map(m => H_(T.mname[m.k]))], rows, { cap: T.mpCap });
  out('mp-out', [T.mp1(h, Ms.map(m => `${T.mname[m.k]} ${f2(m.p.p11 ** h)}`).join(T.sep))]);
  return Ms;
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
const BLK = [['ss-examples', 'se', blkSe, 'se-T', range(20, 200, 20), 600], ['wx-arma', 'wa', blkWa, 'wa-t', range(1, 12, 1), 700],
  ['wx-kf', 'wk', blkWk, 'wk-e', range(0.5, 10, 0.5), 400], ['kf-table', 'kt', blkKt, 'kt-y', range(8, 20, 0.5), 400], ['wx-ss', 'ws', blkWs, 'ws-q', range(0, 9, 1), 600],
  ['nile-filter', 'nf', blkNf, 'nf-q', range(0, 8, 1), 700], ['nile-ml', 'nm', blkNm, 'nm-e', range(1920, 1970, 5), 900], ['gain-ses', 'gs', blkGs, 'gs-q', range(0, 8, 1), 700],
  ['wx-rts', 'wr', blkWr, 'wr-y', range(1885, 1915, 1), 400], ['nile-smooth', 'ns', blkNs, 'ns-q', range(0, 8, 1), 700], ['wx-lik', 'wl', blkWl, 'wl-q', range(0, 8, 1), 700],
  ['likelihood', 'lk', blkLk, 'lk-q', range(0, 8, 1), 700], ['diagnostics', 'dg', blkDg, 'dg-q', range(0, 8, 1), 700], ['missing', 'mi', blkMi, 'mi-q', range(0, 8, 1), 700],
  ['wx-hp', 'wh', blkWh, 'wh-l', range(0, 7, 1), 700], ['ro-trend', 'rt', blkRt, 'rt-l', range(0, 7, 1), 700], ['output-gap', 'og', blkOg, 'og-h', range(4, 12, 1), 700],
  ['realtime', 'rl', blkRl, 'rl-e', null, 600], ['tvp-beta', 'tv', blkTv, 'tv-s', range(0, 5, 1), 900], ['dfm', 'df', blkDf, 'df-k', range(0, 6, 1), 800],
  ['wx-dur', 'wd', blkWd, 'wd-p', range(0.05, 0.95, 0.05), 400], ['wx-ham', 'wf', blkWf, 'wf-t', range(0, 15, 1), 600], ['ms-table', 'mt', blkMt, 'mt-m', null, 1200],
  ['ms-us', 'mu', blkMu, 'mu-p', range(0, 9, 1), 700], ['ms-filtered', 'mf', blkMf, 'mf-w', null, 1100], ['ms-checks', 'mc', blkMc, 'mc-e', null, 1000],
  ['ms-ro', 'mr', blkMr, 'mr-p', range(0, 8, 1), 700], ['vol-regimes', 'vr', blkVr, 'vr-p', range(0, 6, 1), 900], ['regimes-memory', 'rm', blkRm, 'rm-m', range(1, 40, 3), 400],
  ['ms-params', 'mp', blkMp, 'mp-h', range(1, 40, 3), 500]];
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
window.CH10 = { llf, rts1, distSmooth1, steady, profile, golden, kf2, rts2, hpTrend, hamFilter, hamilton, concord, byHand, arma, nile, nileML, tvpRun, hpRealtime, usModel, msModels,
  QS, QML, LAMS, BLK, init };
if (!window.CH10_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
