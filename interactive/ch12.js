// Chapter 12 interactive page (spectral analysis): one block per chart slide, plus the worked examples, the result tables and
// the extra charts of the lecture, computed in the browser from ch12_data.js (Quantlets/Ch_12/export_interactive.py): the
// simulated draws of the chapter (seed 2026), the sunspots, the hourly load of Romania, US and Romanian GDP, the S&P 500
// returns and US industrial production and unemployment. The functions below follow generate_all_charts.py line by line.
(function () {
'use strict';
const D = window.CH12_DATA;
const { sum, mean, nf, $, AX, layout, plot, Player, range, predict, exitTicket, quantile } = window.TSAI;
const out = (id, items) => window.TSAI.out(id, items);
const T = window.CH12_TEXT;
const NB = D.numbers;

// ---------------------------------------------------------------- colours of tsa_style.py
const K = { blue: '#1A3A6E', red: '#CD0000', green: '#2E7D32', amber: '#B5853F', orange: '#E67E22', purple: '#8E44AD', teal: '#17A2B8', crimson: '#DC3545', navy: '#1F2A44' };
const RGB = { [K.blue]: '26,58,110', [K.red]: '205,0,0', [K.green]: '46,125,50', [K.amber]: '181,133,63', [K.orange]: '230,126,34', [K.purple]: '142,68,173',
  [K.teal]: '23,162,184', [K.navy]: '31,42,68', [K.crimson]: '220,53,69' };
const PAL = [K.blue, K.red, K.green, K.amber, K.purple, K.orange, K.teal, K.crimson];

// ---------------------------------------------------------------- small helpers
const set = (id, v) => { const e = $(id + '-v'); if (e) e.textContent = v; };
const has = id => !!$(id);
const cache = {};
const memo = (k, f) => (k in cache ? cache[k] : (cache[k] = f()));
const f0 = x => nf(x, 0), f1 = x => nf(x, 1), f2 = x => nf(x, 2), f3 = x => nf(x, 3), f4 = x => nf(x, 4);
const pc = (x, d) => nf(100 * x, d === undefined ? 0 : d) + '%';
const argmin = a => a.reduce((b, v, i) => (v < a[b] ? i : b), 0);
const argmax = a => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);
const sd0 = x => { const m = mean(x); return Math.sqrt(mean(x.map(v => (v - m) ** 2))); };
const sd1 = x => { const m = mean(x); return Math.sqrt(sum(x.map(v => (v - m) ** 2)) / (x.length - 1)); };
const median = a => { const b = [...a].sort((x, y) => x - y), n = b.length; return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2; };
const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
const geomspace = (a, b, n) => Array.from({ length: n }, (_, i) => Math.exp(Math.log(a) + (Math.log(b) - Math.log(a)) * i / (n - 1)));
const trapz = (y, x) => { let s = 0; for (let i = 1; i < y.length; i++) s += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2; return s; };
const big = x => T.big(x);
const sig3 = x => { if (!isFinite(x)) return '∞'; const a = Math.abs(x); return a >= 1e4 ? big(x) : nf(x, a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : a >= 0.1 ? 3 : 4); };

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
const yax = (G, a) => G.lay[a.ya.replace('y', 'yaxis')];
const xax = (G, a) => G.lay[a.xa.replace('x', 'xaxis')];
const ln = (x, y, a, color, name, o) => { const r = Object.assign({ x, y, mode: 'lines', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.4g}<extra></extra>' }, o || {}); r.line = Object.assign({ color, width: 1.4 }, (o && o.line) || {}); return r; };
const mk = (x, y, a, color, name, o) => Object.assign({ x, y, mode: 'markers', name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, marker: { color, size: 7 }, hovertemplate: '%{x}: %{y:.4g}<extra></extra>' }, o || {});
const bar = (x, y, a, color, name, o) => Object.assign({ x, y, type: 'bar', marker: { color }, name, showlegend: !!name, xaxis: a.xa, yaxis: a.ya, hovertemplate: '%{x}: %{y:.4g}<extra></extra>' }, o || {});
const area = (x, y, a, col, al, name) => ({ x: [x[0], ...x, x[x.length - 1]], y: [0, ...y, 0], fill: 'toself', fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, mode: 'lines', hoverinfo: 'skip', name: name || '', showlegend: !!name, xaxis: a.xa, yaxis: a.ya });
const band = (x, lo, hi, a, col, al, name) => ({ x: [...x, ...[...x].reverse()], y: [...hi, ...[...lo].reverse()], fill: 'toself', fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, mode: 'lines', hoverinfo: 'skip', name: name || '', showlegend: !!name, xaxis: a.xa, yaxis: a.ya });
const hl = (a, xs, y, color, name, o) => ln(xs, xs.map(() => y), a, color, name, Object.assign({ hoverinfo: 'skip' }, o || {}));
const vl = (a, x, y0, y1, color, name, o) => ln([x, x], [y0, y1], a, color, name, Object.assign({ hoverinfo: 'skip' }, o || {}));
const stems = (x, y, a, color, name, o) => { const xs = [], ys = []; x.forEach((v, i) => { xs.push(v, v, null); ys.push(0, y[i], null); }); return ln(xs, ys, a, color, name, Object.assign({ hoverinfo: 'skip' }, o || {})); };
const vspan = (a, x0, x1, col, al) => ({ type: 'rect', xref: a.xa, yref: a.yd, x0, x1, y0: 0, y1: 1, fillcolor: `rgba(${RGB[col]},${al})`, line: { width: 0 }, layer: 'below' });
const legBox = (name, col, al) => ({ x: [null], y: [null], name, mode: 'markers', marker: { symbol: 'square', size: 12, color: `rgba(${RGB[col]},${al})` }, hoverinfo: 'skip' });
const LOG = { type: 'log', exponentformat: 'power' };

// ---------------------------------------------------------------- tables and worked steps (HTML)
function table(id, head, rows, o) {
  const el = $(id); if (!el) return;
  const cell = (c, tag) => { const v = typeof c === 'object' && c !== null ? c : { v: c }; return `<${tag}${v.cls ? ` class="${v.cls}"` : ''}>${v.v}</${tag}>`; };
  el.innerHTML = (o && o.cap ? `<p class="c12cap">${o.cap}</p>` : '') + '<table><thead><tr>' + head.map(c => cell(c, 'th')).join('') + '</tr></thead><tbody>'
    + rows.map(r => '<tr' + (r.hi ? ' class="hi"' : '') + '>' + r.map(c => cell(c, 'td')).join('') + '</tr>').join('') + '</tbody></table>' + (o && o.note ? `<p class="c12cap">${o.note}</p>` : '');
}
const R_ = v => ({ v, cls: 'r' });
const steps = (id, items) => { const el = $(id); if (el) el.innerHTML = items.map(s => `<li>${s}</li>`).join(''); };

// ================================================================= numerical core (generate_all_charts.py)
// FFT: radix-2 for powers of two, Bluestein (chirp z) for other lengths
function fft2(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}
function fft(x, xi) {                     // returns [re, im] of sum_t x_t e^{-2 pi i k t / n}, t = 0..n-1
  const n = x.length;
  if ((n & (n - 1)) === 0) { const re = Float64Array.from(x), im = xi ? Float64Array.from(xi) : new Float64Array(n); fft2(re, im); return [re, im]; }
  let M = 1; while (M < 2 * n - 1) M <<= 1;
  const wr = new Float64Array(n), wi = new Float64Array(n);
  for (let k = 0; k < n; k++) { const a = Math.PI * ((k * k) % (2 * n)) / n; wr[k] = Math.cos(a); wi[k] = -Math.sin(a); }
  const ar = new Float64Array(M), ai = new Float64Array(M), br = new Float64Array(M), bi = new Float64Array(M);
  for (let k = 0; k < n; k++) { const xr = x[k], xim = xi ? xi[k] : 0; ar[k] = xr * wr[k] - xim * wi[k]; ai[k] = xr * wi[k] + xim * wr[k]; }
  br[0] = wr[0]; bi[0] = -wi[0];
  for (let k = 1; k < n; k++) { br[k] = br[M - k] = wr[k]; bi[k] = bi[M - k] = -wi[k]; }
  fft2(ar, ai); fft2(br, bi);
  for (let k = 0; k < M; k++) { const r = ar[k] * br[k] - ai[k] * bi[k], i = ar[k] * bi[k] + ai[k] * br[k]; ar[k] = r; ai[k] = -i; }
  fft2(ar, ai);                            // inverse through conjugation
  const re = new Float64Array(n), im = new Float64Array(n);
  for (let k = 0; k < n; k++) { const cr = ar[k] / M, ci = -ai[k] / M; re[k] = cr * wr[k] - ci * wi[k]; im[k] = cr * wi[k] + ci * wr[k]; }
  return [re, im];
}
// periodogram I(nu_j) = |d(nu_j)|^2 at nu_j = j/T, j = 1..[T/2], mean removed; Hann taper rescaled to keep the power
function periodogram(x0, taper) {
  const T_ = x0.length, m = mean(x0); let x = x0.map(v => v - m);
  if (taper) { const w = x.map((_, t) => 0.5 * (1 - Math.cos(2 * Math.PI * (t + 0.5) / T_))), s = Math.sqrt(mean(w.map(v => v * v))); x = x.map((v, t) => v * w[t] / s); }
  const [re, im] = fft(x), J = Math.floor(T_ / 2), nu = [], I = [];
  for (let j = 1; j <= J; j++) { nu.push(j / T_); I.push((re[j] * re[j] + im[j] * im[j]) / T_); }
  return { nu, I };
}
function daniell(I, m) {                  // average of 2m + 1 ordinates, reflected at the ends (numpy pad 'reflect')
  if (!m) return I.slice();
  const n = I.length, p = [];
  for (let k = m; k >= 1; k--) p.push(I[k]);
  p.push(...I);
  for (let k = 1; k <= m; k++) p.push(I[n - 1 - k]);
  const L = 2 * m + 1, o = []; let s = 0;
  for (let i = 0; i < L; i++) s += p[i];
  o.push(s / L);
  for (let i = L; i < p.length; i++) { s += p[i] - p[i - L]; o.push(s / L); }
  return o;
}
// Welch (scipy.signal.welch: periodic Hann window, constant detrend per segment, density, one-sided), divided by 2
function welchRaw(x, y, n, nover) {
  const step = n - nover, K_ = Math.floor((x.length - nover) / step), w = range(0, n - 1, 1).map(k => 0.5 - 0.5 * Math.cos(2 * Math.PI * k / n));
  const sc = 1 / sum(w.map(v => v * v)), H = Math.floor(n / 2) + 1;
  const Pxx = new Float64Array(H), Pyy = new Float64Array(H), Rr = new Float64Array(H), Ri = new Float64Array(H);
  for (let s = 0; s < K_; s++) {
    const seg = x.slice(s * step, s * step + n), ms = mean(seg), [xr, xi] = fft(seg.map((v, k) => (v - ms) * w[k]));
    let yr = xr, yi = xi;
    if (y) { const sg = y.slice(s * step, s * step + n), my = mean(sg); [yr, yi] = fft(sg.map((v, k) => (v - my) * w[k])); }
    for (let k = 0; k < H; k++) {
      Pxx[k] += (xr[k] * xr[k] + xi[k] * xi[k]) * sc / K_; Pyy[k] += (yr[k] * yr[k] + yi[k] * yi[k]) * sc / K_;
      Rr[k] += (xr[k] * yr[k] + xi[k] * yi[k]) * sc / K_; Ri[k] += (xr[k] * yi[k] - xi[k] * yr[k]) * sc / K_;      // conj(X) Y
    }
  }
  const one = k => (k > 0 && (n % 2 ? true : k < H - 1) ? 2 : 1);
  return { nu: range(0, H - 1, 1).map(k => k / n), Pxx: [...Pxx].map((v, k) => v * one(k)), Pyy: [...Pyy].map((v, k) => v * one(k)),
    Pr: [...Rr].map((v, k) => v * one(k)), Pi: [...Ri].map((v, k) => v * one(k)), K: K_ };
}
function welch(x, n) {
  const m = mean(x), W = welchRaw(x.map(v => v - m), null, n, Math.floor(n / 2));
  return { nu: W.nu.slice(1), f: W.Pxx.slice(1).map(v => v / 2), K: W.K };
}
function armaSpec(nu, phi, theta, s2) {
  const one = v => { const th = -2 * Math.PI * v; let nr = 1, ni = 0, dr = 1, di = 0;
    (theta || []).forEach((t, k) => { nr += t * Math.cos(th * (k + 1)); ni += t * Math.sin(th * (k + 1)); });
    (phi || []).forEach((p, k) => { dr -= p * Math.cos(th * (k + 1)); di -= p * Math.sin(th * (k + 1)); });
    return (s2 === undefined ? 1 : s2) * (nr * nr + ni * ni) / (dr * dr + di * di); };
  return Array.isArray(nu) ? nu.map(one) : one(nu);
}
// log Gamma (Lanczos), regularized lower incomplete gamma, chi-square distribution and quantile
function lgam(z) {
  const g = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * z))) - lgam(1 - z);
  z -= 1; let a = 0.99999999999980993; const t = z + 7.5;
  g.forEach((c, i) => { a += c / (z + i + 1); });
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}
function gammaP(a, x) {
  if (x <= 0) return 0;
  if (x < a + 1) { let s = 1 / a, d = s, ap = a; for (let n = 0; n < 500; n++) { ap += 1; d *= x / ap; s += d; if (Math.abs(d) < Math.abs(s) * 1e-16) break; } return s * Math.exp(-x + a * Math.log(x) - lgam(a)); }
  let b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
  for (let i = 1; i < 500; i++) { const an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) d = 1e-300; c = b + an / c; if (Math.abs(c) < 1e-300) c = 1e-300; d = 1 / d; const de = d * c; h *= de; if (Math.abs(de - 1) < 1e-16) break; }
  return 1 - Math.exp(-x + a * Math.log(x) - lgam(a)) * h;
}
const chi2cdf = (x, k) => gammaP(k / 2, x / 2);
function chi2ppf(p, k) { let lo = 0, hi = Math.max(10, 10 * k); while (chi2cdf(hi, k) < p) hi *= 2; for (let i = 0; i < 200; i++) { const mid = (lo + hi) / 2; if (chi2cdf(mid, k) < p) lo = mid; else hi = mid; } return (lo + hi) / 2; }
const chi2pdf = (x, k) => (x <= 0 ? 0 : Math.exp((k / 2 - 1) * Math.log(x) - x / 2 - (k / 2) * Math.log(2) - lgam(k / 2)));
function chi2band(fh, df, level) { const a = 1 - level, q1 = chi2ppf(1 - a / 2, df), q0 = chi2ppf(a / 2, df); return { lo: fh.map(v => df * v / q1), hi: fh.map(v => df * v / q0), q0, q1 }; }
function fisherG(I) {                     // Fisher's g and its exact p-value
  const m = I.length, g = Math.max(...I) / sum(I); let p = 0;
  for (let k = 1; k <= Math.floor(1 / g); k++) { const b = 1 - k * g; if (b <= 0) break; p += (k % 2 ? 1 : -1) * Math.exp(lgam(m + 1) - lgam(k + 1) - lgam(m - k + 1) + (m - 1) * Math.log(b)); }
  return { g, m, p: Math.min(Math.max(p, 0), 1), j: argmax(I) + 1 };
}
function hpTrend(y, lam) {                // (I + lam K'K) tau = y, K the second difference; banded LDL' solve
  const n = y.length, a = new Float64Array(n), b = new Float64Array(n), c = new Float64Array(n);   // A[i][i], A[i][i-1], A[i][i-2]
  for (let i = 0; i < n; i++) a[i] = 1 + lam * (i === 0 || i === n - 1 ? 1 : i === 1 || i === n - 2 ? 5 : 6);
  for (let i = 1; i < n; i++) b[i] = lam * (i === 1 || i === n - 1 ? -2 : -4);
  for (let i = 2; i < n; i++) c[i] = lam;
  const Dg = new Float64Array(n), l1 = new Float64Array(n), l2 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    l2[i] = i >= 2 ? c[i] / Dg[i - 2] : 0;
    l1[i] = i >= 1 ? (b[i] - (i >= 2 ? l2[i] * Dg[i - 2] * l1[i - 1] : 0)) / Dg[i - 1] : 0;
    Dg[i] = a[i] - (i >= 1 ? l1[i] * l1[i] * Dg[i - 1] : 0) - (i >= 2 ? l2[i] * l2[i] * Dg[i - 2] : 0);
  }
  const z = new Float64Array(n), t = new Float64Array(n);
  for (let i = 0; i < n; i++) z[i] = y[i] - (i >= 1 ? l1[i] * z[i - 1] : 0) - (i >= 2 ? l2[i] * z[i - 2] : 0);
  for (let i = n - 1; i >= 0; i--) t[i] = z[i] / Dg[i] - (i + 1 < n ? l1[i + 1] * t[i + 1] : 0) - (i + 2 < n ? l2[i + 2] * t[i + 2] : 0);
  return [...t];
}
const hpCycle = (y, lam) => { const t = hpTrend(y, lam); return y.map((v, i) => v - t[i]); };
const hpGain = (nu, lam) => { const a = 4 * lam * (1 - Math.cos(2 * Math.PI * nu)) ** 2; return a / (1 + a); };
function gph(I, nu, m) {
  const X = nu.slice(0, m).map(v => -Math.log(4 * Math.sin(Math.PI * v) ** 2)), Y = I.slice(0, m).map(Math.log), mx = mean(X), my = mean(Y);
  const d = sum(X.map((v, i) => (v - mx) * (Y[i] - my))) / sum(X.map(v => (v - mx) ** 2));
  return { d, c: my - d * mx, se: Math.PI / Math.sqrt(24 * m), m, nu_m: nu[m - 1] };
}
function morlet(x0, periods, w0) {        // Morlet wavelet power / variance (Torrence and Compo), by FFT
  const T_ = x0.length, m = mean(x0), s0 = sd0(x0), x = x0.map(v => (v - m) / s0);
  let n = 1; while (n < 2 * T_) n <<= 1;
  const xp = new Array(n).fill(0); x.forEach((v, i) => { xp[i] = v; });
  const [Xr, Xi] = fft(xp), om = range(0, n - 1, 1).map(k => 2 * Math.PI * (k < Math.ceil(n / 2) ? k : k - n) / n);
  const ff = 4 * Math.PI / (w0 + Math.sqrt(2 + w0 * w0));
  return periods.map(per => {
    const s = per / ff, re = new Float64Array(n), im = new Float64Array(n);
    for (let k = 0; k < n; k++) { if (om[k] <= 0) continue; const p = Math.PI ** -0.25 * Math.sqrt(2 * Math.PI * s) * Math.exp(-0.5 * (s * om[k] - w0) ** 2); re[k] = Xr[k] * p; im[k] = -Xi[k] * p; }
    fft2(re, im);                          // inverse via conjugation
    return range(0, T_ - 1, 1).map(t => (re[t] * re[t] + im[t] * im[t]) / (n * n));
  });
}

// ================================================================= data
const SUN = D.sun.y, SUNY = SUN.map((_, i) => D.sun.y0 + i);
const isoQ = (q0, k) => { const y0 = +q0.slice(0, 4), m = +q0.slice(5, 7) - 1 + 3 * k; return `${y0 + Math.floor(m / 12)}-${String(m % 12 + 1).padStart(2, '0')}-01`; };
const isoM = (m0, k) => { const y0 = +m0.slice(0, 4), m = +m0.slice(5, 7) - 1 + k; return `${y0 + Math.floor(m / 12)}-${String(m % 12 + 1).padStart(2, '0')}-01`; };
const hourIso = (t0, k) => new Date(Date.parse(t0.replace(' ', 'T').slice(0, 16) + ':00Z') + k * 3600e3).toISOString().slice(0, 16).replace('T', ' ');
const LOAD = D.load.y, LT = LOAD.map((_, i) => hourIso(D.load.t0, i));
const sunPg = () => memo('sunpg', () => periodogram(SUN));
const BCQ = [6, 32];

// ================================================================= 1. two cycles plus noise
function fourierSeries(A2, s) { const t = range(1, 120, 1); return { t, c1: t.map(v => 2 * Math.cos(2 * Math.PI * v / 12)), c2: t.map(v => A2 * Math.cos(2 * Math.PI * v / 4 + 1)), e: D.sim.e120.map(v => s * v) }; }
function fourierStats(A2, s) {
  const S = fourierSeries(A2, s), x = S.t.map((_, i) => S.c1[i] + S.c2[i] + S.e[i]), P = periodogram(x), nu = P.nu;
  const j1 = argmin(nu.map(v => Math.abs(v - 1 / 12))), j2 = argmin(nu.map(v => Math.abs(v - 1 / 4))), m = mean(x);
  return { S, x, P, I12: P.I[j1], I4: P.I[j2], share: (P.I[j1] + P.I[j2]) / sum(P.I), var: mean(x.map(v => (v - m) ** 2)), th12: 4 * 120 / 4, th4: A2 * A2 * 120 / 4 };
}
function blkFo() {
  const A2 = +$('fo-a').value, s = +$('fo-s').value; set('fo-a', f2(A2)); set('fo-s', f2(s));
  const F = fourierStats(A2, s), S = F.S;
  const G = grid('fo-chart', [{ r: 0, c: 0, t: 'Two cycles plus noise', xt: 't' }, { r: 0, c: 1, t: 'Periodogram', xt: 'frequency nu (cycles per observation)' }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax;
  plot('fo-chart', [ln(S.t, F.x, a, K.blue, 'series x_t', { line: { width: 1.2 } }), ln(S.t, S.c1, a, K.red, '2 cos(2 pi t / 12): period 12', { line: { width: 1.1, dash: 'dash' } }),
    ln(S.t, S.c2, a, K.green, `${A2.toFixed(2)} cos(2 pi t / 4 + 1): period 4`, { line: { width: 1, dash: 'dot' } }),
    stems(F.P.nu, F.P.I, b, K.blue, 'periodogram I(nu_j)', { line: { width: 1.6 } }),
    mk([1 / 12, 1 / 4], [F.I12, F.I4], b, K.red, '', { mode: 'markers+text', text: ['1/12', '1/4'], textposition: 'middle right', textfont: { color: K.red }, marker: { size: 5, color: K.red } })], G.lay);
  out('fo-out', [T.fo1(f1(F.I12), f1(F.I4), f0(F.th12), f1(F.th4)), T.fo2(pc(F.share), f2(F.var), f2(2 + A2 * A2 / 2 + s * s))]);
  return F;
}

// ================================================================= 2. aliasing
function aliasOf(P, dt) { const v = dt / P, a = Math.abs(v - Math.round(v)); return { v, a, pObs: a > 1e-12 ? 1 / a : Infinity, pM: a > 1e-12 ? dt / a : Infinity }; }
function blkAl() {
  const dt = +$('al-d').value, P = +$('al-p').value; set('al-d', dt); set('al-p', f1(P));
  const A = aliasOf(P, dt), tm = linspace(0, 36, 1000), tq = range(0, 36, dt);
  const G = grid('al-chart', [{ r: 0, c: 0, xt: 'months', x: { tickvals: range(0, 36, 3) }, y: { range: [-1.15, 1.15] } }], { cols: 1, h: 330 });
  const a = G.ax[0];
  plot('al-chart', [ln(tm, tm.map(t => Math.cos(2 * Math.PI * t / P)), a, K.blue, `true cycle: period ${+P.toFixed(2)} months`, { line: { width: 1.1 } }),
    ln(tm, tm.map(t => (isFinite(A.pM) ? Math.cos(2 * Math.PI * t / A.pM) : 1)), a, K.red, isFinite(A.pM) ? `alias: period ${+A.pM.toFixed(2)} months` : 'alias: a constant (period infinite)', { line: { width: 1.6, dash: 'dash' } }),
    mk(tq, tq.map(t => Math.cos(2 * Math.PI * t / P)), a, K.amber, `observations every ${dt} ${dt === 1 ? 'month' : 'months'}`, { marker: { size: 9, color: K.amber } })], G.lay);
  out('al-out', [T.al1(dt, f1(P), f3(A.v), A.v > 0.5), T.al2(f3(A.a), isFinite(A.pObs) ? f2(A.pObs) : '∞', isFinite(A.pM) ? f1(A.pM) : '∞', Math.abs(A.a - 1 / P * dt) < 1e-9)]);
  return A;
}

// ================================================================= 3. gallery of spectral densities
const SPECTRA = [['white noise', [], []], ['AR(1), phi = 0.6', [0.6], []], ['AR(1), phi = -0.6', [-0.6], []], ['MA(1), theta = 0.5', [], [0.5]],
  ['AR(2), phi = (1.5, -0.75)', [1.5, -0.75], []], ['ARMA(1,1), phi = 0.6, theta = 0.4', [0.6], [0.4]]];
function spectraStats(s2, nuMax) {
  const nu = linspace(0, 0.5, 501);
  return SPECTRA.map(([lab, p, q]) => { const f = armaSpec(nu, p, q, s2), k = Math.round(nuMax / 0.001), v = 2 * trapz(f, nu);
    return { lab, f, nu, var: v, f0: f[0], f05: f[500], argmax: nu[argmax(f)], share: 2 * trapz(f.slice(0, k + 1), nu.slice(0, k + 1)) / v }; });
}
function blkSp() {
  const nm = +$('sp-n').value, s2 = +$('sp-s').value; set('sp-n', f2(nm)); set('sp-s', f2(s2));
  const S = spectraStats(s2, nm);
  const G = grid('sp-chart', S.map((r, i) => ({ r: Math.floor(i / 3), c: i % 3, t: r.lab, xt: i >= 3 ? 'frequency nu' : '', y: { range: [0, Math.max(...r.f) * 1.12] } })), { cols: 3, h: 470, legend: false, gx: 0.06, rowPx: 190 });
  const tr = [];
  S.forEach((r, i) => { const a = G.ax[i], k = Math.round(nm / 0.001);
    tr.push(area(r.nu.slice(0, k + 1), r.f.slice(0, k + 1), a, PAL[i], 0.15), ln(r.nu, r.f, a, PAL[i], '', { line: { width: 1.8 } }), mk([r.nu[k]], [r.f[k]], a, PAL[i], '', { marker: { size: 8, color: PAL[i] } })); });
  plot('sp-chart', tr, G.lay);
  out('sp-out', [T.sp1(f2(s2), S.map(r => f4(r.var))), T.sp2(f2(nm), S.map(r => pc(r.share)))]);
  return S;
}

// ================================================================= 4. sunspots and their periodogram
function sunStats(y0, y1) {
  const i0 = y0 - D.sun.y0, i1 = y1 - D.sun.y0, x = SUN.slice(i0, i1 + 1), P = periodogram(x), nu = P.nu, I = P.I, j = argmax(I);
  const per = nu.map(v => 1 / v), bandS = sum(I.filter((_, k) => per[k] >= 9 && per[k] <= 13)) / sum(I);
  const order = I.map((v, k) => [v, k]).sort((p, q) => q[0] - p[0]).slice(0, 3).map(p => 1 / nu[p[1]]);
  return { x, yrs: SUNY.slice(i0, i1 + 1), P, T: x.length, j: j + 1, nu: nu[j], period: 1 / nu[j], share_band: bandS, top3: order, fisher: fisherG(I.filter((_, k) => nu[k] < 0.5)) };
}
function blkSs() {
  const y0 = +$('ss-a').value, y1 = +$('ss-b').value; set('ss-a', y0); set('ss-b', y1);
  const S = sunStats(y0, y1);
  const G = grid('ss-chart', [{ r: 0, c: 0, t: `Sunspots, ${y0}-${y1}`, xt: 'year' }, { r: 0, c: 1, t: 'Raw periodogram', xt: 'frequency nu (cycles per year)' }], { cols: 2, h: 360 });
  const [a, b] = G.ax;
  plot('ss-chart', [ln(S.yrs, S.x, a, K.amber, 'yearly mean sunspot number', { line: { width: 1.1 } }), stems(S.P.nu, S.P.I, b, K.blue, 'periodogram I(nu_j)', { line: { width: 1.2 } }),
    mk([S.nu], [S.P.I[S.j - 1]], b, K.red, '', { mode: 'markers+text', text: [`peak: nu = ${S.nu.toFixed(4)}, period ${S.period.toFixed(1)} years`], textposition: 'middle right', textfont: { color: K.red }, marker: { size: 6, color: K.red } })], G.lay);
  const F = S.fisher;
  out('ss-out', [T.ss1(S.T, S.j, f4(S.nu), f1(S.period), S.top3.slice(1).map(f1)), T.ss2(pc(S.share_band)), T.ss3(F.m, f3(F.g), f4(1 / F.m), F.p < 0.001, f3(F.p))]);
  return S;
}

// ================================================================= 5. the periodogram of white noise
function noiseStats(x, m) {
  const P = periodogram(x), I = daniell(P.I, m), I0 = I.filter((_, k) => P.nu[k] < 0.5);
  return { nu: P.nu, I, mean: mean(I0), sd: sd0(I0), max: Math.max(...I0), q95: quantile(I0, 0.95) };
}
function blkWn() {
  const m = +$('wn-m').value, n2 = +$('wn-T').value; set('wn-m', m);
  const A = noiseStats(D.sim.w128, m), B = noiseStats(D.sim.w2048.slice(0, n2), m), top = Math.max(A.max, B.max) * 1.05;
  const G = grid('wn-chart', [{ r: 0, c: 0, t: 'White noise, T = 128', xt: 'frequency nu', y: { range: [0, top] } }, { r: 0, c: 1, t: `White noise, T = ${n2}`, xt: 'frequency nu', y: { range: [0, top] } }], { cols: 2, h: 340 });
  const [a, b] = G.ax;
  plot('wn-chart', [ln(A.nu, A.I, a, K.blue, m ? `Daniell m = ${m}, T = 128` : 'periodogram, T = 128', { line: { width: 0.8 } }), ln(B.nu, B.I, b, K.purple, m ? `Daniell m = ${m}, T = ${n2}` : `periodogram, T = ${n2}`, { line: { width: 0.8 } }),
    hl(a, [0, 0.5], 1, K.red, 'true spectrum f(nu) = 1', { line: { width: 1.8, dash: 'dash' } }), hl(b, [0, 0.5], 1, K.red, '', { line: { width: 1.8, dash: 'dash' } })], G.lay);
  out('wn-out', [T.wn1(128, f2(A.mean), f2(A.sd), f2(A.max)), T.wn1(n2, f2(B.mean), f2(B.sd), f2(B.max)), T.wn2(m, 2 * m + 1, f2(1 / Math.sqrt(2 * m + 1)), f2(B.q95), f2(m ? chi2ppf(0.95, 2 * (2 * m + 1)) / (2 * (2 * m + 1)) : NB.incons.chi2_q95))]);
  return { A, B };
}

// ================================================================= 6. leakage and tapering
function leakStats(j1, a2) {
  const T_ = 256, nu1 = j1 / T_, nu2 = 0.3, x = range(0, T_ - 1, 1).map(t => Math.cos(2 * Math.PI * nu1 * t) + a2 * Math.cos(2 * Math.PI * nu2 * t) + 0.002 * D.sim.e256[t]);
  const P = periodogram(x), Pt = periodogram(x, true), nu = P.nu, k = argmin(nu.map(v => Math.abs(v - nu2)));
  const near = nu.map((v, i) => [Math.abs(v - nu2), i]).filter(([d]) => d > 3 / T_ && d < 10 / T_).map(([, i]) => i);
  const rb = median(near.map(i => P.I[i])), tb = median(near.map(i => Pt.I[i]));
  return { nu, I: P.I, It: Pt.I, nu1, raw_at2: P.I[k], tap_at2: Pt.I[k], raw_bg: rb, tap_bg: tb, raw_ratio: P.I[k] / rb, tap_ratio: Pt.I[k] / tb };
}
function blkLk() {
  const j1 = +$('lk-j').value, a2 = +$('lk-a').value; set('lk-j', f1(j1)); set('lk-a', f3(a2));
  const L = leakStats(j1, a2);
  const G = grid('lk-chart', [{ r: 0, c: 0, xt: 'frequency nu', yt: 'power (log scale)', y: Object.assign({ range: [-8, 2.5] }, LOG) }], { cols: 1, h: 360, legendRows: 2 });
  const a = G.ax[0];
  plot('lk-chart', [ln(L.nu, L.I, a, K.red, 'raw periodogram (rectangular window)', { line: { width: 1.2 } }), ln(L.nu, L.It, a, K.blue, 'Hann-tapered periodogram', { line: { width: 1.4 } }),
    vl(a, L.nu1, 1e-8, 300, K.amber, `strong cycle, nu = ${L.nu1.toFixed(4)}`, { line: { width: 1, dash: 'dash' } }), vl(a, 0.3, 1e-8, 300, K.green, `weak cycle, nu = 0.30 (amplitude ${+a2.toFixed(3)})`, { line: { width: 1, dash: 'dash' } })], G.lay);
  out('lk-out', [T.lk1(f4(L.nu1), f1(j1), Math.abs(j1 - Math.round(j1)) < 1e-9), T.lk2(sig3(L.raw_at2 / L.raw_bg), sig3(L.tap_at2 / L.tap_bg))]);
  return L;
}

// ================================================================= 7. smoothed estimates of the AR(2) spectrum
const AR2 = [1.5, -0.75];
function smoothStats(m2, nw) {
  const x = D.sim.ar2, P = periodogram(x), nu = P.nu, I = P.I, f = armaSpec(nu, AR2), lf = f.map(Math.log), lmse = v => mean(v.map((u, i) => (Math.log(u) - lf[i]) ** 2));
  const d2 = daniell(I, 2), dm = daniell(I, m2), W = welch(x, nw), fw = armaSpec(W.nu, AR2);
  const est = v => ({ peak: nu[argmax(v)], mse: lmse(v), fpeak: Math.max(...v) });
  return { nu, I, f, d2: est(d2), dm: est(dm), v2: d2, vm: dm, W, welch: { peak: W.nu[argmax(W.f)], mse: mean(W.f.map((u, i) => (Math.log(u) - Math.log(fw[i])) ** 2)), K: W.K, fpeak: Math.max(...W.f) },
    true_peak: nu[argmax(f)], raw_peak: nu[argmax(I)], mse_raw: lmse(I), ftrue_peak: Math.max(...f) };
}
function blkSm() {
  const m2 = +$('sm-m').value, nw = +$('sm-w').value; set('sm-m', m2);
  const S = smoothStats(m2, nw), top = Math.max(...S.I) * 1.05, Tn = S.nu.length * 2;
  const G = grid('sm-chart', [{ r: 0, c: 0, t: 'Raw periodogram', xt: 'frequency nu', x: { range: [0, 0.5] }, y: { range: [0, top] } },
    { r: 0, c: 1, t: 'Smoothed estimates', xt: 'frequency nu', x: { range: [0, 0.5] }, y: { range: [0, top] } }], { cols: 2, h: 360, legendRows: 2 });
  const [a, b] = G.ax;
  const w0 = S.true_peak - m2 / Tn, w1 = S.true_peak + m2 / Tn;
  plot('sm-chart', [band([w0, w1], [0, 0], [top, top], a, K.purple, 0.15, `Daniell window around the true peak: ${2 * m2 + 1} ordinates`),
    ln(S.nu, S.I, a, K.teal, 'raw periodogram', { line: { width: 0.8 } }), ln(S.nu, S.f, a, K.red, 'true AR(2) spectrum', { line: { width: 2, dash: 'dash' } }),
    ln(S.nu, S.v2, b, K.blue, 'Daniell, m = 2 (L = 5)', { line: { width: 1.3 } }), ln(S.nu, S.vm, b, K.purple, `Daniell, m = ${m2} (L = ${2 * m2 + 1})`, { line: { width: 1.6 } }),
    ln(S.W.nu, S.W.f, b, K.orange, `Welch, segments of ${nw}`, { line: { width: 1.4, dash: 'dashdot' } }), ln(S.nu, S.f, b, K.red, '', { line: { width: 2, dash: 'dash' } })], G.lay);
  out('sm-out', [T.sm1(f3(S.true_peak), f3(S.raw_peak), f2(S.mse_raw)), T.sm2(m2, f3(S.dm.peak), f2(S.dm.mse), f0(S.dm.fpeak), f0(S.ftrue_peak)), T.sm3(nw, S.welch.K, f3(S.welch.peak), f2(S.welch.mse))]);
  return S;
}

// ================================================================= 8. sunspots: smoothed spectrum with a confidence band
function sunCi(m, level) {
  const P = sunPg(), nu = P.nu, fh = daniell(P.I, m), df = 2 * (2 * m + 1), B = chi2band(fh, df, level), j = argmax(fh), j2 = argmin(nu.map(v => Math.abs(v - 0.18)));
  return { nu, I: P.I, fh, lo: B.lo, hi: B.hi, df, L: 2 * m + 1, Bw: (2 * m + 1) / SUN.length, j, period: 1 / nu[j], f: fh[j], flo: B.lo[j], fhi: B.hi[j], q0: B.q0, q1: B.q1, f018: fh[j2], hi018: B.hi[j2], lo018: B.lo[j2] };
}
function blkSc() {
  const m = +$('sc-m').value, lv = +$('sc-c').value / 100; set('sc-m', m);
  const S = sunCi(m, lv);
  const G = grid('sc-chart', [{ r: 0, c: 0, xt: 'frequency nu (cycles per year)', yt: 'spectrum (log scale)', y: Object.assign({ range: [-1.5, 5.6] }, LOG) }], { cols: 1, h: 360, legendRows: 2 });
  const a = G.ax[0];
  plot('sc-chart', [band(S.nu, S.lo, S.hi, a, K.teal, 0.2, `${Math.round(lv * 100)}% confidence band (chi-square, df = ${S.df})`), ln(S.nu, S.I, a, K.amber, 'raw periodogram', { line: { width: 0.7 } }),
    ln(S.nu, S.fh, a, K.blue, `Daniell smoother, m = ${m}`, { line: { width: 1.8 } }),
    mk([S.nu[S.j]], [S.f], a, K.red, '', { mode: 'markers+text', text: [`period ${S.period.toFixed(1)} years`], textposition: 'top right', textfont: { color: K.red }, marker: { size: 6, color: K.red } })], G.lay);
  out('sc-out', [T.sc1(m, S.L, S.df, f3(S.Bw)), T.sc2(f1(S.period), big(S.f), Math.round(lv * 100), big(S.flo), big(S.fhi), f2(S.flo / S.f), f2(S.fhi / S.f)), T.sc3(big(S.f018), big(S.hi018), S.flo > S.hi018)]);
  return S;
}

// ================================================================= 9-10. GDP growth, HP cycles and their spectra
const GDP = {
  US: { g: D.us.g, lev: D.us.lev, q0: D.us.q0, l0: isoQ(D.us.q0, -1) },
  RO: { g: D.ro.g, lev: D.ro.lev, q0: '1995-04-01', l0: '1995-01-01' },
};
const cyc = (k, lam) => memo(`hp|${k}|${lam}`, () => hpCycle(GDP[k].lev, lam));
function gdpStats(lam) {
  const o = {};
  ['US', 'RO'].forEach(k => { const g = GDP[k].g, c = cyc(k, lam); o[k] = { T: g.length, mean: mean(g), sd: sd1(g), cyc_sd: sd1(c) }; });
  return o;
}
function blkGs() {
  const lam = +$('gs-l').value, S = gdpStats(lam);
  const G = grid('gs-chart', [{ r: 0, c: 0, t: 'United States', x: { type: 'date' } }, { r: 0, c: 1, t: 'Romania', x: { type: 'date' } }], { cols: 2, h: 350 });
  const tr = [];
  ['US', 'RO'].forEach((k, i) => { const a = G.ax[i], g = GDP[k].g, c = cyc(k, lam), dg = g.map((_, j) => isoQ(GDP[k].q0, j)), dc = c.map((_, j) => isoQ(GDP[k].l0, j));
    tr.push(ln(dg, g, a, i ? K.red : K.blue, i ? '' : 'quarterly growth, %', { line: { width: 0.9 } }), ln(dc, c, a, K.green, i ? '' : `HP cycle, % of trend (lambda = ${lam})`, { line: { width: 1.5 } }),
      hl(a, [dc[0], dc[dc.length - 1]], 0, K.navy, '', { line: { width: 0.5 } })); });
  plot('gs-chart', tr, G.lay);
  const r = D.ro.today;
  out('gs-out', [T.gs1('US', f2(S.US.mean), f2(S.US.sd), f2(S.US.cyc_sd), lam), T.gs1('RO', f2(S.RO.mean), f2(S.RO.sd), f2(S.RO.cyc_sd), lam), T.gs2(f3(r.mean), f3(r.sd), f3(r.cyc_sd), f3(NB.gdp.RO.mean), f3(NB.gdp.RO.sd), f3(NB.gdp.RO.cyc_sd), T.dmy(D.downloaded))]);
  return S;
}
function gdpSpec(m, lam, bc) {
  const o = {}, lo = 1 / bc[1], hi = 1 / bc[0];
  ['US', 'RO'].forEach(k => ['growth', 'HP cycle'].forEach(kind => {
    const x = kind === 'growth' ? GDP[k].g : cyc(k, lam), P = periodogram(x), fh = daniell(P.I, m), mx = mean(x), v = mean(x.map(u => (u - mx) ** 2));
    const S = sum(P.I), j = argmax(fh);
    o[`${k}.${kind}`] = { nu: P.nu, sp: fh.map(u => u / v), share_bc: sum(P.I.filter((_, i) => P.nu[i] >= lo && P.nu[i] <= hi)) / S, share_low: sum(P.I.filter((_, i) => P.nu[i] < lo)) / S, peak_period: 1 / P.nu[j], T: x.length };
  }));
  return o;
}
function blkGp() {
  const m = +$('gp-m').value, lam = +$('gp-l').value; set('gp-m', m);
  const S = gdpSpec(m, lam, BCQ);
  const G = grid('gp-chart', [{ r: 0, c: 0, t: 'Spectrum of the growth / variance', xt: 'frequency nu (cycles per quarter)', x: { range: [0, 0.5] } },
    { r: 0, c: 1, t: `Spectrum of the HP cycle (lambda = ${lam}) / variance`, xt: 'frequency nu (cycles per quarter)', x: { range: [0, 0.5] } }], { cols: 2, h: 350 });
  G.lay.shapes = G.ax.map(a => vspan(a, 1 / 32, 1 / 6, K.amber, 0.15));
  const tr = [legBox('business-cycle band, 6-32 quarters', K.amber, 0.3)];
  ['growth', 'HP cycle'].forEach((kind, i) => ['US', 'RO'].forEach(k => { const s = S[`${k}.${kind}`];
    tr.push(ln(s.nu, s.sp, G.ax[i], k === 'US' ? K.blue : K.red, i ? '' : (k === 'US' ? 'United States' : 'Romania'), { line: { width: 1.6 } })); }));
  plot('gp-chart', tr, G.lay);
  const r = (k, kind) => S[`${k}.${kind}`];
  out('gp-out', [T.gp1(m, pc(r('US', 'growth').share_bc), f1(r('US', 'growth').peak_period), pc(r('RO', 'growth').share_bc), f2(r('RO', 'growth').peak_period)),
    T.gp2(lam, pc(r('US', 'HP cycle').share_bc), f1(r('US', 'HP cycle').peak_period), f1(r('US', 'HP cycle').peak_period / 4), pc(r('RO', 'HP cycle').share_bc), pc(r('RO', 'HP cycle').share_low), f1(r('RO', 'HP cycle').peak_period))]);
  return S;
}

// ================================================================= 11. spectrum of the Romanian hourly electricity load
const loadPg = () => memo('loadpg', () => periodogram(LOAD));
function loadShares(width, nh) {
  const P = loadPg(), I = P.I, nu = P.nu, T_ = LOAD.length, tot = sum(I);
  const share = periods => { const idx = new Set(); periods.forEach(p => { const j = Math.round(T_ / p); for (let k = Math.max(j - width, 1) - 1; k < Math.min(j + width, I.length - 1); k++) idx.add(k); });
    let s = 0; idx.forEach(k => { s += I[k]; }); return { s: s / tot, idx }; };
  const daily = share(range(1, nh, 1).map(k => 24 / k)), weekly = share([...range(1, nh, 1).map(k => 168 / k), 336]);
  const long = sum(I.filter((_, k) => 1 / nu[k] > 720)) / tot;
  return { daily, weekly, long };
}
const loadTop = () => memo('loadtop', () => { const P = loadPg(), order = P.I.map((v, k) => [v, k]).sort((a, b) => b[0] - a[0]), top = [];
  for (const [v, k] of order) { const p = 1 / P.nu[k]; if (top.every(q => Math.abs(p - q.p) / q.p > 0.05)) top.push({ p, v, k }); if (top.length === 6) break; } return top; });
const LYEARS = { all: ['2022-01-01 00:00', '2026-06-30 23:00'] };
[2022, 2023, 2024, 2025].forEach(y => { LYEARS[y] = [`${y}-01-01 00:00`, `${y}-12-31 23:00`]; });
LYEARS[2026] = ['2026-01-01 00:00', '2026-06-30 23:00'];
function loadWelch(sample, weeks) {
  return memo(`lw|${sample}|${weeks}`, () => { const [a, b] = LYEARS[sample], i0 = LT.indexOf(a), i1 = LT.indexOf(b), x = LOAD.slice(i0, i1 + 1), W = welch(x, 24 * 7 * weeks);
    const top = [], order = W.f.map((v, k) => [v, k]).sort((p, q) => q[0] - p[0]);
    for (const [, k] of order) { const p = 1 / W.nu[k]; if (p <= 336 && top.every(q => Math.abs(p - q) / q > 0.05)) top.push(p); if (top.length === 4) break; }
    return { x, W, top, T: x.length, mean: mean(x) }; });
}
function blkLd() {
  const sample = $('ld-y').value, weeks = +$('ld-w').value, L = loadWelch(sample, weeks), yr = sample === 'all' ? 2025 : +sample;
  const i0 = LT.indexOf(`${yr}-03-03 00:00`), i1 = LT.indexOf(`${yr}-03-16 23:00`), wt = LT.slice(i0, i1 + 1), wy = LOAD.slice(i0, i1 + 1).map(v => v / 1000);
  const G = grid('ld-chart', [{ r: 0, c: 0, t: `Two weeks of March ${yr}`, x: { type: 'date', tickangle: -30 } },
    { r: 0, c: 1, t: `Spectrum of the hourly load, ${sample === 'all' ? '2022-2026' : sample}`, xt: 'period (hours, log scale)', yt: 'spectrum (log scale)', x: Object.assign({ autorange: 'reversed' }, LOG), y: LOG }], { cols: 2, h: 360, mb: 46 });
  const [a, b] = G.ax, per = L.W.nu.map(v => 1 / v), fmax = Math.max(...L.W.f);
  const tr = [ln(wt, wy, a, K.blue, 'load, GW (UTC hours)', { line: { width: 1.1 } }), ln(per, L.W.f, b, K.red, `Welch spectrum (${weeks}-week segments)`, { line: { width: 1.2 } })];
  [[168, '168 h'], [24, '24 h'], [12, '12 h'], [8, '8 h']].forEach(([p, t]) => tr.push(vl(b, p, Math.min(...L.W.f), fmax * 2, K.green, '', { line: { width: 0.8, dash: 'dot' } }),
    { x: [p], y: [fmax * 2], mode: 'text', text: [t], textposition: 'bottom left', textfont: { color: K.green, size: 10 }, xaxis: b.xa, yaxis: b.ya, showlegend: false, hoverinfo: 'skip' }));
  plot('ld-chart', tr, G.lay);
  out('ld-out', [T.ld1(sample, L.T, big(L.mean), L.W.K, weeks), T.ld2(L.top.map(p => f1(p)))]);
  return L;
}

// ================================================================= 12. the pole at zero in stock-market data
const spPg = () => memo('sppg', () => { const r = D.sp.r, a = r.map(Math.abs); return { r: periodogram(r), abs: periodogram(a) }; });
function lmStats(bw) { const P = spPg(), Tn = D.sp.r.length, m = Math.floor(Tn ** bw); return { r: gph(P.r.I, P.r.nu, m), abs: gph(P.abs.I, P.abs.nu, m), m, T: Tn }; }
function binLog(P, nb) {
  const nu = P.nu, edges = geomspace(nu[0], 0.5, nb + 1), byBin = new Map();
  nu.forEach((v, i) => { let k = 0; while (k < edges.length && v >= edges[k]) k++; if (!byBin.has(k)) byBin.set(k, []); byBin.get(k).push(i); });
  const ks = [...byBin.keys()].sort((a, b) => a - b);
  return { x: ks.map(k => Math.exp(mean(byBin.get(k).map(i => Math.log(nu[i]))))), y: ks.map(k => mean(byBin.get(k).map(i => P.I[i]))) };
}
function blkLm() {
  const bw = +$('lm-b').value, nb = +$('lm-k').value; set('lm-b', f2(bw));
  const S = lmStats(bw), P = spPg();
  const G = grid('lm-chart', [{ r: 0, c: 0, xt: 'frequency nu (cycles per day, log scale)', yt: 'power (log scale)', x: LOG, y: LOG }], { cols: 1, h: 380, legendRows: 2 });
  const a = G.ax[0], tr = [];
  [['returns r_t', P.r, S.r, K.blue], ['absolute returns |r_t|', P.abs, S.abs, K.red]].forEach(([lab, Pg, g, c]) => { const B = binLog(Pg, nb), xx = Pg.nu.slice(0, S.m);
    const br = G.narrow ? '<br>' : ' ';
    tr.push(ln(B.x, B.y, a, c, `${lab}: periodogram,${br}averaged in ${nb} log-frequency bins`, { mode: 'lines+markers', marker: { size: 4, color: c }, line: { width: 1 } }),
      ln(xx, xx.map(v => Math.exp(g.c - g.d * Math.log(4 * Math.sin(Math.PI * v) ** 2))), a, c, `${lab}: GPH line, d = ${g.d.toFixed(2)}`, { line: { width: 2.4, dash: 'dash' } })); });
  plot('lm-chart', tr, G.lay);
  out('lm-out', [T.lm1(f2(bw), big(S.T), S.m, f1(1 / S.r.nu_m)), T.lm2(f2(S.r.d), f3(S.r.se), f2(S.abs.d))]);
  return S;
}

// ================================================================= 13. squared gains of common filters
function filtAt(P, lam) { const nu = 1 / P, G_ = hpGain(nu, lam); return { nu, d1: 4 * Math.sin(Math.PI * nu) ** 2, d4: 4 * Math.sin(4 * Math.PI * nu) ** 2, G: G_, g2: G_ * G_, t2: (1 - G_) ** 2 }; }
function hpHalf(lam) { const a = 1 / (Math.SQRT2 - 1), c = Math.sqrt(a / (4 * lam)), nu = Math.acos(1 - c) / (2 * Math.PI); return { nu, P: 1 / nu }; }
function blkFl() {
  const P = +$('fl-p').value, lam = +$('fl-l').value; set('fl-p', P);
  const nu = linspace(0, 0.5, 1001), F = filtAt(P, lam);
  const G = grid('fl-chart', [{ r: 0, c: 0, t: 'Differencing filters', xt: 'frequency nu (cycles per quarter)', yt: 'squared gain |H(nu)|^2' },
    { r: 0, c: 1, t: `HP filters, lambda = ${lam}`, xt: 'frequency nu (cycles per quarter)', yt: 'squared gain |H(nu)|^2' }], { cols: 2, h: 350, legendRows: 2, ml: 58 });
  const [a, b] = G.ax;
  G.lay.shapes = [a, b].map(x => vspan(x, 1 / 32, 1 / 6, K.amber, 0.15));
  plot('fl-chart', [ln(nu, nu.map(v => 4 * Math.sin(Math.PI * v) ** 2), a, K.blue, 'first difference 1 - L', { line: { width: 1.8 } }),
    ln(nu, nu.map(v => 4 * Math.sin(4 * Math.PI * v) ** 2), a, K.orange, 'seasonal difference 1 - L^4', { line: { width: 1.6, dash: 'dash' } }),
    ln(nu, nu.map(v => hpGain(v, lam) ** 2), b, K.green, `HP cycle filter, lambda = ${lam}`, { line: { width: 1.8 } }), ln(nu, nu.map(v => (1 - hpGain(v, lam)) ** 2), b, K.purple, 'HP trend filter', { line: { width: 1.6, dash: 'dash' } }),
    legBox('business-cycle band', K.amber, 0.3),
    mk([F.nu, F.nu], [F.d1, F.d4], a, K.red, `period ${P} quarters (nu = ${F.nu.toFixed(4)})`, { marker: { size: 9, color: K.red, symbol: 'circle-open', line: { width: 2 } } }),
    mk([F.nu, F.nu], [F.g2, F.t2], b, K.red, '', { marker: { size: 9, color: K.red, symbol: 'circle-open', line: { width: 2 } } })], G.lay);
  const h = hpHalf(lam);
  out('fl-out', [T.fl1(P, f3(F.d1), f3(F.d4)), T.fl2(P, lam, f3(F.g2), f3(F.t2)), T.fl3(lam, f1(h.P), f1(h.P / 4))]);
  return F;
}

// ================================================================= 14. coherence and phase
function cohStats(n) {
  const x = D.coh.ip, y = D.coh.du, W = welchRaw(x, y, n, 0), nu = W.nu, C = nu.map((_, k) => (W.Pr[k] ** 2 + W.Pi[k] ** 2) / (W.Pxx[k] * W.Pyy[k])), ph = nu.map((_, k) => Math.atan2(W.Pi[k], W.Pr[k]));
  const K_ = Math.floor(x.length / n), thr = 1 - 0.05 ** (1 / (K_ - 1)), bnd = nu.map(v => v >= 1 / 96 && v <= 1 / 18), j = argmin(nu.map(v => Math.abs(v - 1 / 48)));
  const cb = C.filter((_, k) => bnd[k]), mx = mean(x), my = mean(y);
  const corr = sum(x.map((v, i) => (v - mx) * (y[i] - my))) / Math.sqrt(sum(x.map(v => (v - mx) ** 2)) * sum(y.map(v => (v - my) ** 2)));
  return { nu, C, ph, K: K_, thr, coh_bc: mean(cb), coh_bc_max: Math.max(...cb), coh_hi: mean(C.filter((_, k) => nu[k] > 1 / 6)), phase48: ph[j], nu48: nu[j], lag48: -ph[j] / (2 * Math.PI * nu[j]), corr0: corr };
}
function blkCo() {
  const n = +$('co-n').value, S = cohStats(n);
  const G = grid('co-chart', [{ r: 0, c: 0, t: 'Squared coherence', xt: 'frequency nu (cycles per month)', y: { range: [0, 1] } }, { r: 0, c: 1, t: 'Phase', xt: 'frequency nu (cycles per month)' }], { cols: 2, h: 340, legendRows: 2 });
  const [a, b] = G.ax;
  G.lay.shapes = [a, b].map(x => vspan(x, 1 / 96, 1 / 18, K.amber, 0.15));
  plot('co-chart', [legBox('business-cycle band, 18-96 months', K.amber, 0.3), ln(S.nu, S.C, a, K.blue, `squared coherence (segments of ${n} months)`, { mode: 'lines+markers', marker: { size: 4, color: K.blue }, line: { width: 1.6 } }),
    hl(a, [0, 0.5], S.thr, K.red, `5% significance threshold (${S.thr.toFixed(2)})`, { line: { width: 1.2, dash: 'dash' } }),
    ln(S.nu, S.ph, b, K.purple, 'phase (radians)', { mode: 'lines+markers', marker: { size: 4, color: K.purple }, line: { width: 1.4 } }), hl(b, [0, 0.5], 0, K.navy, '', { line: { width: 0.5 } })], G.lay);
  out('co-out', [T.co1(n, S.K, f2(S.thr)), T.co2(f2(S.coh_bc), f2(S.coh_bc_max), f2(S.coh_hi), f2(S.corr0)), T.co3(f1(1 / S.nu48), f2(S.phase48), f1(S.lag48))]);
  return S;
}

// ================================================================= 15. wavelet power of the sunspots
const WPER = geomspace(2, 64, 60);
function wavStats(w0, a0) {
  const P = memo('wav|' + w0, () => morlet(SUN, WPER, w0)), k = argmin(WPER.map(v => Math.abs(v - 11))), bnd = WPER.map(v => v >= 8 && v <= 14);
  const pw = SUNY.map((_, t) => mean(P.filter((_, i) => bnd[i]).map(r => r[t]))), win = SUNY.map(y => y >= a0 && y <= a0 + 35);
  return { P, p11_mean: mean(P[k]), weak: mean(pw.filter((_, t) => win[t])), all: mean(pw), max_year: SUNY[argmax(pw)] };
}
function blkWv() {
  const w0 = +$('wv-w').value, a0 = +$('wv-a').value; set('wv-a', `${a0}-${a0 + 35}`);
  const S = wavStats(w0, a0);
  const G = grid('wv-chart', [{ r: 0, c: 0, xt: 'year', yt: 'period (years)', y: { type: 'log', autorange: 'reversed', tickvals: [2, 4, 8, 11, 16, 32, 64], ticktext: ['2', '4', '8', '11', '16', '32', '64'] } }], { cols: 1, h: 360, legend: false, mr: 90 });
  const a = G.ax[0];
  G.lay.shapes = [{ type: 'rect', xref: a.xa, yref: a.ya, x0: a0, x1: a0 + 35, y0: 8, y1: 14, line: { color: K.red, width: 2, dash: 'dash' }, fillcolor: 'rgba(0,0,0,0)' }];
  plot('wv-chart', [{ type: 'contour', x: SUNY, y: WPER, z: S.P, colorscale: 'Viridis', ncontours: 12, line: { width: 0 }, xaxis: a.xa, yaxis: a.ya,
    colorbar: { title: { text: 'wavelet power / variance', side: 'right', font: { size: 11, color: K.navy } }, thickness: 12, tickfont: { size: 10, color: K.navy } }, hovertemplate: '%{x}, period %{y:.1f}: %{z:.2f}<extra></extra>' }], G.lay);
  out('wv-out', [T.wv1(w0, a0, a0 + 35, f1(S.weak), f1(S.all), S.max_year)]);
  return S;
}

// ================================================================= EXTRA 1. a cycle: amplitude, frequency, phase
function blkWy() {
  const P = +$('wy-p').value, ph = +$('wy-f').value, A = +$('wy-a').value; set('wy-p', P); set('wy-f', f2(ph)); set('wy-a', f1(A));
  const nu = 1 / P, t = range(1, 48, 1), tf = linspace(1, 48, 600), a_ = A * Math.cos(ph), b_ = -A * Math.sin(ph);
  const G = grid('wy-chart', [{ r: 0, c: 0, xt: 't (observations)', y: { range: [-3.3, 3.3] } }], { cols: 1, h: 330, legendRows: 2 });
  const a = G.ax[0];
  plot('wy-chart', [ln(tf, tf.map(v => A * Math.cos(2 * Math.PI * nu * v + ph)), a, K.blue, `A cos(2 pi nu t + phi), A = ${A}, nu = 1/${P}, phi = ${ph}`, { line: { width: 1.6 } }),
    mk(t, t.map(v => A * Math.cos(2 * Math.PI * nu * v + ph)), a, K.blue, 'observations x_t', { marker: { size: 6, color: K.blue } }),
    ln(tf, tf.map(v => a_ * Math.cos(2 * Math.PI * nu * v)), a, K.red, `a cos(2 pi nu t), a = ${a_.toFixed(2)}`, { line: { width: 1.1, dash: 'dash' } }),
    ln(tf, tf.map(v => b_ * Math.sin(2 * Math.PI * nu * v)), a, K.green, `b sin(2 pi nu t), b = ${b_.toFixed(2)}`, { line: { width: 1.1, dash: 'dot' } })], G.lay);
  const u = $('wy-u').value;
  steps('wy-ws', [T.wys1(P, f4(nu), f4(2 * Math.PI * nu)), T.wys2(u, P), T.wys3(f1(A), f2(ph), f3(a_), f3(b_), f2(a_ * a_ + b_ * b_), f2(A * A)), T.wys4(P, P > 2)]);
  out('wy-out', [T.wy1(P, f4(nu), 48 / P)]);
  return { nu, a: a_, b: b_ };
}

// ================================================================= EXTRA 2. worked example: the DFT of four numbers
const cfmt = (re, im) => { const r = Math.abs(re) < 1e-12 ? 0 : re, i = Math.abs(im) < 1e-12 ? 0 : im; if (!i) return f2(r); if (!r) return `${i < 0 ? '−' : ''}${Math.abs(i) === 1 ? '' : f2(Math.abs(i))}i`; return `${f2(r)} ${i < 0 ? '−' : '+'} ${f2(Math.abs(i))}i`; };
function dft4(x) {
  const T_ = x.length, m = mean(x), y = x.map(v => v - m), d = range(0, T_ - 1, 1).map(j => { let re = 0, im = 0, terms = [];
    y.forEach((v, t) => { const ang = -2 * Math.PI * j * (t + 1) / T_, cr = Math.cos(ang), ci = Math.sin(ang); terms.push([v * cr, v * ci]); re += v * cr; im += v * ci; });
    return { re, im, terms, dr: re / Math.sqrt(T_), di: im / Math.sqrt(T_), I: (re * re + im * im) / T_ }; });
  return { x, m, y, d, ss: sum(y.map(v => v * v)) };
}
function blkWd() {
  const x1 = +$('wd-a').value, x3 = +$('wd-b').value; set('wd-a', x1); set('wd-b', x3);
  const X = dft4([x1, 2, x3, 2]), j = +$('wd-j').value, D_ = X.d[j];
  const G = grid('wd-chart', [{ r: 0, c: 0, t: 'data x_t and the mean', xt: 't', x: { tickvals: [1, 2, 3, 4] }, y: { range: [Math.min(-1, ...X.x) - 0.5, Math.max(...X.x) + 1] } },
    { r: 0, c: 1, t: '|d(nu_j)|^2 (deviations from the mean)', xt: 'nu_j = j/4', x: { tickvals: [0, 0.25, 0.5, 0.75], ticktext: ['0', '1/4', '1/2', '3/4'] }, y: { range: [0, Math.max(4.5, ...X.d.map(v => v.I)) * 1.15] } }], { cols: 2, h: 330 });
  const [a, b] = G.ax;
  plot('wd-chart', [bar([1, 2, 3, 4], X.x, a, K.blue, 'x_t', { text: X.x.map(String), textposition: 'outside' }), hl(a, [0.6, 4.4], X.m, K.red, `mean = ${X.m.toFixed(2)}`, { line: { width: 1.5, dash: 'dash' } }),
    bar([0, 0.25, 0.5, 0.75], X.d.map(v => v.I), b, K.amber, '|d(nu_j)|^2', { width: 0.1, text: X.d.map(v => v.I.toFixed(2)), textposition: 'outside' })], G.lay);
  const nuL = ['0', '1/4', '1/2', '3/4'][j];
  steps('wd-ws', [T.wds1(X.x.map(f0).join(T.semi), f2(X.m), X.y.map(f2).join(T.semi)), T.wds2(nuL, D_.terms.map(([r, i]) => cfmt(r, i)).reduce((a, t) => (a === null ? t : t.startsWith('−') ? `${a} − ${t.slice(1)}` : `${a} + ${t}`), null), cfmt(D_.re, D_.im)),
    T.wds3(nuL, cfmt(D_.dr, D_.di), f2(D_.I)), T.wds4(X.d.slice(1).map(v => f2(v.I)).join(' + '), f2(sum(X.d.slice(1).map(v => v.I))), f2(X.ss), f2(X.ss / 4))]);
  out('wd-out', [T.wd1(X.x.map(f0).join(T.semi), X.d.map(v => f2(v.I)).join(T.semi))]);
  return X;
}

// ================================================================= EXTRA 3. interpreting aliasing: the folding of frequencies
function blkWa() {
  const P = +$('wa-p').value, dt = +$('wa-d').value; set('wa-p', f1(P));
  const A = aliasOf(P, dt), fm = linspace(0, 1, 801), app = fm.map(v => { const u = v * dt; return Math.abs(u - Math.round(u)) / dt; });
  const G = grid('wa-chart', [{ r: 0, c: 0, xt: 'true frequency (cycles per month)', yt: 'apparent frequency (cycles per month)', y: { range: [0, 0.55] } }], { cols: 1, h: 330, legendRows: 2 });
  const a = G.ax[0];
  plot('wa-chart', [ln(fm, app, a, K.blue, `apparent frequency, one observation every ${dt} ${dt === 1 ? 'month' : 'months'}`, { line: { width: 1.6 } }),
    hl(a, [0, 1], 1 / (2 * dt), K.red, `Nyquist frequency 1/(2 x ${dt}) = ${(1 / (2 * dt)).toFixed(3)} cycles per month`, { line: { width: 1.2, dash: 'dash' } }),
    mk([1 / P], [A.a / dt], a, K.amber, `true cycle of ${+P.toFixed(2)} months`, { marker: { size: 11, color: K.amber } }), vl(a, 1 / P, 0, A.a / dt, K.amber, '', { line: { width: 1, dash: 'dot' } })], G.lay);
  steps('wa-ws', [T.was1(f1(P), f4(1 / P), dt, f4(A.v)), T.was2(f4(A.v), A.v > 0.5, f4(A.a)), T.was3(isFinite(A.pObs) ? f2(A.pObs) : '∞', isFinite(A.pM) ? f1(A.pM) : '∞', dt)]);
  out('wa-out', [T.wa1(f1(P), dt, isFinite(A.pM) ? f1(A.pM) : '∞', Math.abs(A.pM - P) < 1e-9)]);
  return A;
}

// ================================================================= EXTRA 4. worked example: AR(1) and MA(1) spectra
function blkWr() {
  const phi = +$('wr-p').value, th = +$('wr-t').value; set('wr-p', f1(phi)); set('wr-t', f1(th));
  const nu = linspace(0, 0.5, 501), fa = armaSpec(nu, [phi], []), fm = armaSpec(nu, [], [th]);
  const G = grid('wr-chart', [{ r: 0, c: 0, xt: 'frequency nu', yt: 'f(nu), log scale', y: LOG }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0];
  plot('wr-chart', [ln(nu, fa, a, K.blue, `AR(1), phi = ${phi.toFixed(1)}`, { line: { width: 1.8 } }), ln(nu, fm, a, K.red, `MA(1), theta = ${th.toFixed(1)}`, { line: { width: 1.8, dash: 'dash' } }),
    mk([0, 0.5], [fa[0], fa[500]], a, K.blue, '', { mode: 'markers+text', text: [fa[0].toFixed(2), fa[500].toFixed(2)], textposition: ['middle right', 'middle left'], textfont: { color: K.blue }, marker: { size: 8, color: K.blue } }),
    mk([0, 0.5], [fm[0], fm[500]], a, K.red, '', { mode: 'markers+text', text: [fm[0].toFixed(2), fm[500].toFixed(2)], textposition: ['bottom right', 'bottom left'], textfont: { color: K.red }, marker: { size: 8, color: K.red } })], G.lay);
  const v = 1 / (1 - phi * phi), vi = 2 * trapz(fa, nu);
  steps('wr-ws', [T.wrs1(f1(phi), f2(1 - phi), f4((1 - phi) ** 2), f2(fa[0])), T.wrs2(f1(phi), f2(1 + phi), f4((1 + phi) ** 2), f2(fa[500]), f2(fa[0] / fa[500])),
    T.wrs3(f1(th), f2(fm[0]), f2(fm[500])), T.wrs4(f1(phi), f4(1 - phi * phi), f4(v), f4(vi))]);
  out('wr-out', [T.wr1(phi, f2(fa[0] / fa[500]))]);
  return { f0: fa[0], f5: fa[500], m0: fm[0], m5: fm[500], v, vi };
}

// ================================================================= EXTRA 5. worked example: the peak of an AR(2) spectrum
function ar2Peak(p1, p2) {
  const disc = p1 * p1 + 4 * p2, c = p1 * (p2 - 1) / (4 * p2), inside = Math.abs(c) <= 1, nu = inside ? Math.acos(c) / (2 * Math.PI) : 0;
  return { disc, c, inside, nu, period: inside && nu > 0 ? 1 / nu : Infinity, fpeak: armaSpec(nu, [p1, p2]), f0: armaSpec(0, [p1, p2]) };
}
function blkW2() {
  const p1 = +$('w2-a').value, p2 = +$('w2-b').value; set('w2-a', f2(p1)); set('w2-b', f2(p2));
  const A = ar2Peak(p1, p2), nu = linspace(0, 0.5, 501), f = armaSpec(nu, [p1, p2]), rho = [1, p1 / (1 - p2)];
  for (let h = 2; h <= 40; h++) rho.push(p1 * rho[h - 1] + p2 * rho[h - 2]);
  const G = grid('w2-chart', [{ r: 0, c: 0, t: 'spectral density f(nu)', xt: 'frequency nu' }, { r: 0, c: 1, t: 'autocorrelation rho(h)', xt: 'lag h', y: { range: [-1.05, 1.05] } }], { cols: 2, h: 340 });
  const [a, b] = G.ax;
  plot('w2-chart', [ln(nu, f, a, K.purple, `AR(2), phi = (${p1.toFixed(2)}, ${p2.toFixed(2)})`, { line: { width: 1.8 } }),
    mk([A.nu], [A.fpeak], a, K.red, A.inside ? `peak: nu* = ${A.nu.toFixed(4)}, period ${A.period.toFixed(1)}` : 'maximum at nu = 0', { marker: { size: 9, color: K.red } }),
    bar(range(0, 40, 1), rho, b, K.blue, 'rho(h)', { width: 0.35 })], G.lay);
  steps('w2-ws', [T.w2s1(f2(p1), f2(p2), f2(A.disc), A.disc < 0), T.w2s2(f2(p1), f2(p2), f3(A.c), A.inside, f4(A.nu), f1(A.period)), T.w2s3(f1(A.fpeak), f2(A.f0), f2(1 - p1 - p2))]);
  out('w2-out', [T.w21(A.inside, f1(A.period), f1(A.fpeak / A.f0))]);
  return A;
}

// ================================================================= EXTRA 6. table: interpreting the sunspot periodogram
function blkTs() {
  const lo = +$('ts-a').value, hi = +$('ts-b').value; set('ts-a', lo); set('ts-b', hi);
  const P = sunPg(), per = P.nu.map(v => 1 / v), tot = sum(P.I), inB = per.map(p => p >= lo && p <= hi), share = sum(P.I.filter((_, k) => inB[k])) / tot;
  const sel = per.map((p, k) => k).filter(k => per[k] >= 2 && per[k] <= 40);
  const G = grid('ts-chart', [{ r: 0, c: 0, xt: 'period (years, log scale)', yt: 'I(nu_j)', x: { type: 'log', tickvals: [2, 3, 5, 8, 11, 15, 20, 30, 40] } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0], ord = P.I.map((v, k) => [v, k]).sort((p, q) => q[0] - p[0]), k_ = +$('ts-k').value, top = ord.slice(0, k_);
  G.lay.shapes = [vspan(a, lo, hi, K.amber, 0.18)];
  plot('ts-chart', [legBox(`periods ${lo}-${hi} years: ${(100 * share).toFixed(0)}% of the variance`, K.amber, 0.35), stems(sel.map(k => per[k]), sel.map(k => P.I[k]), a, K.blue, 'periodogram I(nu_j)', { line: { width: 1.4 } }),
    stems(sel.filter(k => inB[k]).map(k => per[k]), sel.filter(k => inB[k]).map(k => P.I[k]), a, K.amber, '', { line: { width: 2.2 } }),
    mk(top.map(([, k]) => per[k]), top.map(([v]) => v), a, K.red, `the ${k_} largest ordinates`, { marker: { size: 8, color: K.red } })], G.lay);
  table('ts-tb', T.tsHead, top.map(([v, k], i) => [R_(i + 1), R_(k + 1), R_(f4(P.nu[k])), R_(f2(per[k])), R_(big(v)), R_(pc(v / tot, 1))]), { note: T.tsNote(lo, hi, pc(share, 1), pc(sum(top.map(([v]) => v)) / tot, 1)) });
  out('ts-out', [T.ts1(lo, hi, pc(share), sel.length)]);
  return { share, top: top.map(([, k]) => per[k]) };
}

// ================================================================= EXTRA 7. statistical properties of the periodogram (extra chart)
function blkXp() {
  const n = +$('xp-T').value, c = +$('xp-c').value; set('xp-c', f1(c));
  const x = n === 128 ? D.sim.w128 : D.sim.w2048, P = periodogram(x), I0 = P.I.filter((_, k) => P.nu[k] < 0.5), above = I0.filter(v => v > c).length;
  const xs = linspace(0, 8, 161), nb = 32;
  const G = grid('xp-chart', [{ r: 0, c: 0, xt: 'I(nu_j) / f(nu_j)', yt: 'density', x: { range: [0, 8] } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0];
  const bw = 8 / nb, cnt = range(0, nb - 1, 1).map(k => I0.filter(v => v >= k * bw && v < (k + 1) * bw).length / (I0.length * bw));
  plot('xp-chart', [bar(range(0, nb - 1, 1).map(k => (k + 0.5) * bw), cnt, a, `rgba(${RGB[K.blue]},0.45)`, `ordinates of white noise, T = ${n} (${I0.length}), density`, { width: bw * 0.95, hovertemplate: '%{x:.2f}: %{y:.3f}<extra></extra>' }),
    ln(xs, xs.map(v => Math.exp(-v)), a, K.red, 'chi-square(2) / 2 = exponential, mean 1', { line: { width: 2 } }),
    vl(a, c, 0, 1, K.amber, `threshold c = ${c}`, { line: { width: 2, dash: 'dash' } })], G.lay);
  out('xp-out', [T.xp1(n, I0.length, f2(mean(I0)), f2(sd0(I0) ** 2)), T.xp2(f1(c), above, I0.length, pc(above / I0.length, 1), pc(Math.exp(-c), 1)), T.xp3(f2(quantile(I0, 0.95)), f2(NB.incons.chi2_q95))]);
  return { above, share: above / I0.length, q95: quantile(I0, 0.95) };
}

// ================================================================= EXTRA 8. the Daniell estimator, ordinate by ordinate
function blkWl() {
  const m = +$('wl-m').value, j = +$('wl-j').value; set('wl-m', m); set('wl-j', j);
  const P = sunPg(), fh = daniell(P.I, m), L = 2 * m + 1, idx = range(j - m, j + m, 1), vals = idx.map(i => P.I[i - 1]), avg = mean(vals);
  const sel = range(1, 70, 1);
  const G = grid('wl-chart', [{ r: 0, c: 0, xt: 'frequency nu (cycles per year)', yt: 'I(nu_j)', x: { range: [0, 70.5 / 309] } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0];
  plot('wl-chart', [stems(sel.map(i => P.nu[i - 1]), sel.map(i => P.I[i - 1]), a, K.amber, 'raw periodogram', { line: { width: 1.4 } }),
    stems(idx.map(i => P.nu[i - 1]), vals, a, K.red, `the L = ${L} ordinates averaged at j = ${j}`, { line: { width: 2.6 } }),
    ln(sel.map(i => P.nu[i - 1]), sel.map(i => fh[i - 1]), a, K.blue, `Daniell estimate, m = ${m}`, { line: { width: 1.8 } }),
    mk([P.nu[j - 1]], [avg], a, K.navy, '', { marker: { size: 10, color: K.navy, symbol: 'diamond' } })], G.lay);
  steps('wl-ws', [T.wls1(j, f4(P.nu[j - 1]), f1(1 / P.nu[j - 1]), m, L), T.wls2(idx.map((i, k) => `I(ν<sub>${i}</sub>) = ${big(vals[k])}`).join(T.semi), big(sum(vals)), L, big(avg)),
    T.wls3(L, f4(L / 309), 2 * L, f2(1 / L))]);
  out('wl-out', [T.wl1(m, L, f1(1 / P.nu[argmax(fh)]), big(Math.max(...fh)))]);
  return { avg, L, peak: 1 / P.nu[argmax(fh)] };
}

// ================================================================= EXTRA 9. table: interpreting the smoothed estimates
function blkTm() {
  const m = +$('tm-m').value, nw = +$('tm-w').value; set('tm-m', m);
  const ms = range(0, 30, 1), rows = memo('tmcurve', () => ms.map(k => smoothStats(k, 128).dm)), S = smoothStats(m, nw);
  const G = grid('tm-chart', [{ r: 0, c: 0, t: 'MSE of log f-hat (variance + bias)', xt: 'Daniell m' }, { r: 0, c: 1, t: 'height of the highest smoothed ordinate', xt: 'Daniell m' }], { cols: 2, h: 330, legendRows: 2 });
  const [a, b] = G.ax;
  plot('tm-chart', [ln(ms, rows.map(r => r.mse), a, K.purple, 'Daniell estimate', { mode: 'lines+markers', marker: { size: 4, color: K.purple } }), mk([m], [S.dm.mse], a, K.red, `m = ${m}`, { marker: { size: 11, color: K.red } }),
    ln(ms, rows.map(r => r.fpeak), b, K.purple, '', { mode: 'lines+markers', marker: { size: 4, color: K.purple } }), mk([m], [S.dm.fpeak], b, K.red, '', { marker: { size: 11, color: K.red } }),
    hl(b, [0, 30], S.ftrue_peak, K.red, `true peak height ${S.ftrue_peak.toFixed(0)}`, { line: { width: 1.5, dash: 'dash' } })], G.lay);
  const Tn = 512, row = (lab, L, v, hi) => Object.assign([lab, R_(L), R_(L === '–' ? '–' : f4(L / Tn)), R_(L === '–' ? '–' : 2 * L), R_(f3(v.peak)), R_(f1(1 / v.peak)), R_(f1(v.fpeak)), R_(f2(v.mse))], { hi });
  const raw = { peak: S.raw_peak, fpeak: Math.max(...S.I), mse: S.mse_raw }, rs = [row(T.tmRaw, 1, raw), row('Daniell, m = 2', 5, S.d2), row('Daniell, m = 10', 21, smoothStats(10, 128).dm)];
  if (m !== 2 && m !== 10) rs.push(row(`Daniell, m = ${m}`, 2 * m + 1, S.dm, true));
  rs.push(Object.assign([`Welch, ${S.welch.K} × ${nw}`, R_('–'), R_('–'), R_('–'), R_(f3(S.welch.peak)), R_(f1(1 / S.welch.peak)), R_(f1(S.welch.fpeak)), R_(f2(S.welch.mse))], { hi: nw !== 128 }));
  table('tm-tb', T.tmHead, rs, { note: T.tmNote(f3(S.true_peak), f1(1 / S.true_peak), f1(S.ftrue_peak)) });
  out('tm-out', [T.tm1(m, 2 * m + 1, f4((2 * m + 1) / Tn), f2(S.dm.mse), f1(S.dm.fpeak), ms[argmin(rows.map(r => r.mse))])]);
  return { S, best: ms[argmin(rows.map(r => r.mse))] };
}

// ================================================================= EXTRA 10. worked example: a confidence band with L ordinates
function blkWb() {
  const L = +$('wb-L').value, lv = +$('wb-c').value / 100; set('wb-L', L);
  const df = 2 * L, a2 = (1 - lv) / 2, q0 = chi2ppf(a2, df), q1 = chi2ppf(1 - a2, df), xs = linspace(0.01, Math.max(40, q1 * 1.4), 400), Ls = range(1, 21, 2);
  const G = grid('wb-chart', [{ r: 0, c: 0, t: `chi-square density, df = ${df}`, xt: 'x' }, { r: 0, c: 1, t: 'band multipliers of f-hat', xt: 'L (ordinates averaged)', y: Object.assign({}, LOG) }], { cols: 2, h: 340, legendRows: 2 });
  const [a, b] = G.ax, pdf = xs.map(v => chi2pdf(v, df));
  const lt = xs.filter(v => v <= q0), rt = xs.filter(v => v >= q1);
  plot('wb-chart', [ln(xs, pdf, a, K.blue, 'density', { line: { width: 1.8 } }), area(lt, lt.map(v => chi2pdf(v, df)), a, K.red, 0.35, `tails of ${(a2 * 100).toFixed(1)}% each`),
    area(rt, rt.map(v => chi2pdf(v, df)), a, K.red, 0.35),
    ln(Ls, Ls.map(l => 2 * l / chi2ppf(a2, 2 * l)), b, K.teal, 'upper multiplier df / q(alpha/2)', { mode: 'lines+markers', marker: { size: 4, color: K.teal } }),
    ln(Ls, Ls.map(l => 2 * l / chi2ppf(1 - a2, 2 * l)), b, K.purple, 'lower multiplier df / q(1 - alpha/2)', { mode: 'lines+markers', marker: { size: 4, color: K.purple } }),
    mk([L, L], [df / q0, df / q1], b, K.red, '', { marker: { size: 10, color: K.red } })], G.lay);
  const S = sunCi((L - 1) / 2, lv), pct_ = +(lv * 100).toFixed(0);
  steps('wb-ws', [T.wbs1(L, (L - 1) / 2, df, f3(a2), f2(q0), f3(1 - a2), f2(q1)), T.wbs2(df, f2(q1), f2(df / q1), f2(q0), f2(df / q0)),
    T.wbs3((L - 1) / 2, f1(S.period), big(S.f), pct_, big(S.flo), big(S.fhi))]);
  out('wb-out', [T.wb1(L, pct_, f2(df / q1), f2(df / q0), f1((df / q0) / (df / q1)))]);
  return { q0, q1, lo: df / q1, hi: df / q0, S };
}

// ================================================================= EXTRA 11. table: interpreting the GDP spectra
function blkTg() {
  const lo = +$('tg-a').value, hi = +$('tg-b').value, m = +$('tg-m').value; set('tg-a', lo); set('tg-b', hi);
  const S = gdpSpec(m, 1600, [lo, hi]), keys = ['US.growth', 'RO.growth', 'US.HP cycle', 'RO.HP cycle'], labs = ['US growth', 'RO growth', 'US HP cycle', 'RO HP cycle'];
  const G = grid('tg-chart', [{ r: 0, c: 0, yt: 'share of the variance', y: { range: [0, 1], tickformat: '.0%' } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0], low = keys.map(k => S[k].share_low), bc = keys.map(k => S[k].share_bc);
  plot('tg-chart', [bar(labs, low, a, K.purple, `periods longer than ${hi} quarters`), bar(labs, bc, a, K.amber, `business-cycle band, ${lo}-${hi} quarters`),
    bar(labs, keys.map((_, i) => 1 - low[i] - bc[i]), a, K.teal, `periods shorter than ${lo} quarters`)], Object.assign(G.lay, { barmode: 'stack' }));
  table('tg-tb', T.tgHead, keys.map((k, i) => [T.tgLab[i], R_(S[k].T), R_(pc(S[k].share_bc)), R_(pc(S[k].share_low)), R_(f2(S[k].peak_period)), R_(f1(S[k].peak_period / 4))]), { note: T.tgNote(m) });
  out('tg-out', [T.tg1(lo, hi, f3(1 / hi), f3(1 / lo), pc(S['US.growth'].share_bc), pc(S['RO.growth'].share_bc), pc(S['US.HP cycle'].share_bc), pc(S['RO.HP cycle'].share_bc))]);
  return S;
}

// ================================================================= EXTRA 12. table: interpreting the load spectrum
function blkTl() {
  const w = +$('tl-w').value, nh = +$('tl-k').value; set('tl-w', w);
  const S = loadShares(w, nh), P = loadPg(), top = loadTop(), j24 = Math.round(LOAD.length / 24), ks = range(j24 - 14, j24 + 14, 1);
  const G = grid('tl-chart', [{ r: 0, c: 0, t: 'share of the variance', y: { range: [0, 0.6], tickformat: '.0%' } }, { r: 0, c: 1, t: 'ordinates around the 24-hour period', xt: 'period (hours)', yt: 'I(nu_j), log scale', y: LOG }], { cols: 2, h: 340, legendRows: 2 });
  const [a, b] = G.ax, inD = ks.filter(k => S.daily.idx.has(k - 1));
  plot('tl-chart', [bar(['daily cycle<br>and harmonics', 'weekly cycle<br>and harmonics', 'periods longer<br>than a month'], [S.daily.s, S.weekly.s, S.long], a, K.blue, '', { text: [S.daily.s, S.weekly.s, S.long].map(v => (100 * v).toFixed(1) + '%'), textposition: 'outside' }),
    mk(ks.map(k => 1 / P.nu[k - 1]), ks.map(k => P.I[k - 1]), b, K.navy, 'periodogram ordinates', { marker: { size: 5, color: K.navy } }),
    mk(inD.map(k => 1 / P.nu[k - 1]), inD.map(k => P.I[k - 1]), b, K.red, `counted in the daily share (half-width ${w})`, { marker: { size: 10, color: K.red, symbol: 'circle-open', line: { width: 2 } } })], G.lay);
  table('tl-tb', T.tlHead, [[T.tlRow[0](nh), R_(pc(S.daily.s, 1))], [T.tlRow[1](nh), R_(pc(S.weekly.s, 1))], [T.tlRow[2], R_(pc(S.long, 1))]],
    { note: T.tlNote(top.map(t => f1(t.p)).join(T.semi), LOAD.length) });
  out('tl-out', [T.tl1(w, 2 * w, pc(S.daily.s), pc(S.weekly.s), pc(S.long))]);
  return S;
}

// ================================================================= EXTRA 13. worked example: the HP gain at a chosen period
function blkWh() {
  const lam = +$('wh-l').value, P = +$('wh-p').value; set('wh-p', P);
  const per = geomspace(2, 200, 300), F = filtAt(P, lam), h = hpHalf(lam);
  const G = grid('wh-chart', [{ r: 0, c: 0, xt: 'period (quarters, log scale)', x: { type: 'log', tickvals: [2, 4, 8, 16, 32, 64, 128] }, y: { range: [0, 1.05] } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0];
  plot('wh-chart', [ln(per, per.map(p => hpGain(1 / p, lam)), a, K.blue, `gain G, lambda = ${lam}`, { line: { width: 1.6, dash: 'dot' } }),
    ln(per, per.map(p => hpGain(1 / p, lam) ** 2), a, K.green, 'squared gain of the cycle filter G^2', { line: { width: 1.8 } }),
    ln(per, per.map(p => (1 - hpGain(1 / p, lam)) ** 2), a, K.purple, 'squared gain of the trend filter (1 - G)^2', { line: { width: 1.6, dash: 'dash' } }),
    mk([P, P, P], [F.G, F.g2, F.t2], a, K.red, `period ${P} quarters`, { marker: { size: 9, color: K.red } }), vl(a, h.P, 0, 1.05, K.navy, `half power: ${h.P.toFixed(1)} quarters`, { line: { width: 1, dash: 'dash' } })], G.lay);
  const c = 1 - Math.cos(2 * Math.PI / P), A_ = 4 * lam * c * c;
  steps('wh-ws', [T.whs1(P, f4(1 / P), f4(2 * Math.PI / P), f5(c)), T.whs2(lam, f5(c), f4(A_)), T.whs3(f4(A_), f4(F.G), f3(F.g2), f3(F.t2)), T.whs4(P, f3(F.d1))]);
  out('wh-out', [T.wh1(lam, f1(h.P), f1(h.P / 4))]);
  return F;
}
function f5(x) { return nf(x, 5); }

// ================================================================= EXTRA 14. table: interpreting the filter gains
const FPER = [2, 4, 6, 8, 16, 32, 40, 80];
function blkTf() {
  const lam = +$('tf-l').value, sq = $('tf-q').value === 'sq';
  const per = geomspace(2, 200, 300);
  const G = grid('tf-chart', [{ r: 0, c: 0, xt: 'period (quarters, log scale)', yt: 'squared gain', x: { type: 'log', tickvals: FPER }, y: { range: [0, 4.2] } }], { cols: 1, h: 340, legendRows: 2 });
  const a = G.ax[0], tr = [];
  [['first difference 1 - L', p => filtAt(p, lam).d1, K.blue], ['seasonal difference 1 - L^4', p => filtAt(p, lam).d4, K.orange], [`HP cycle filter, lambda = ${lam}`, p => filtAt(p, lam).g2, K.green], ['HP trend filter', p => filtAt(p, lam).t2, K.purple]]
    .forEach(([n, f, c]) => tr.push(ln(per, per.map(f), a, c, n, { line: { width: 1.6 } }), mk(FPER, FPER.map(f), a, c, '', { marker: { size: 6, color: c } })));
  plot('tf-chart', tr, G.lay);
  const r = p => { const F = filtAt(p, lam); return sq ? [F.d1, F.d4, F.g2, F.t2] : [Math.sqrt(F.d1), Math.sqrt(F.d4), F.G, 1 - F.G]; };
  table('tf-tb', T.tfHead(sq), FPER.map(p => Object.assign([R_(p), R_(f3(1 / p)), ...r(p).map(v => R_(f3(v)))], { hi: [8, 32, 40].includes(p) })), { note: T.tfNote(sq, lam) });
  const F32 = filtAt(32, lam);
  out('tf-out', [T.tf1(lam, f3(filtAt(8, lam).g2), f3(F32.g2), f3(filtAt(40, lam).g2), f1(hpHalf(lam).P))]);
  return { g8: filtAt(8, lam).g2, g32: F32.g2, g40: filtAt(40, lam).g2 };
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
const LAMS = null;
const BLK = [['wx-cycle', 'wy', blkWy, 'wy-p', range(2, 24, 1), 400], ['fourier', 'fo', blkFo, 'fo-a', range(0, 2, 0.25), 600], ['wx-dft', 'wd', blkWd, 'wd-b', range(-4, 4, 1), 600],
  ['aliasing', 'al', blkAl, 'al-d', range(1, 6, 1), 900], ['wx-alias', 'wa', blkWa, 'wa-p', range(2, 12, 0.5), 400],
  ['wx-arma', 'wr', blkWr, 'wr-p', range(-0.9, 0.9, 0.1), 400], ['wx-ar2', 'w2', blkW2, 'w2-b', range(-0.95, -0.65, 0.05), 700], ['spectra', 'sp', blkSp, 'sp-n', range(0.05, 0.5, 0.05), 500],
  ['sunspots', 'ss', blkSs, 'ss-b', range(1808, 2008, 25), 700], ['t-sun', 'ts', blkTs, 'ts-b', range(11, 20, 1), 600], ['x-prop', 'xp', blkXp, 'xp-c', range(1, 5, 0.5), 600],
  ['noise', 'wn', blkWn, 'wn-m', range(0, 10, 1), 600], ['leakage', 'lk', blkLk, 'lk-j', range(25, 26, 0.1), 500], ['wx-daniell', 'wl', blkWl, 'wl-m', range(0, 8, 1), 700],
  ['smoothing', 'sm', blkSm, 'sm-m', range(1, 20, 1), 500], ['t-smooth', 'tm', blkTm, 'tm-m', range(0, 30, 2), 500], ['wx-band', 'wb', blkWb, 'wb-L', range(1, 21, 2), 600],
  ['sun-ci', 'sc', blkSc, 'sc-m', range(0, 8, 1), 700], ['gdp-series', 'gs', blkGs, 'gs-l', LAMS, 900], ['gdp-spectra', 'gp', blkGp, 'gp-m', range(0, 6, 1), 700],
  ['t-gdp', 'tg', blkTg, 'tg-b', range(16, 48, 4), 600], ['load', 'ld', blkLd, 'ld-y', null, 1200], ['t-load', 'tl', blkTl, 'tl-w', range(1, 10, 1), 600],
  ['long-memory', 'lm', blkLm, 'lm-b', range(0.45, 0.85, 0.05), 700], ['wx-hp', 'wh', blkWh, 'wh-l', null, 900], ['filters', 'fl', blkFl, 'fl-p', [4, 6, 8, 12, 16, 24, 32, 40, 60, 80], 600],
  ['t-filt', 'tf', blkTf, 'tf-l', null, 900], ['coherence', 'co', blkCo, 'co-n', null, 900], ['wavelet', 'wv', blkWv, 'wv-w', null, 1000]];
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
window.CH12 = { fft, periodogram, daniell, welch, welchRaw, armaSpec, chi2ppf, fisherG, hpCycle, hpGain, gph, morlet, fourierStats, aliasOf, spectraStats, sunStats, noiseStats,
  leakStats, smoothStats, sunCi, gdpStats, gdpSpec, loadShares, loadTop, loadWelch, lmStats, filtAt, hpHalf, cohStats, wavStats, dft4, ar2Peak, BLK, init };
if (!window.CH12_NO_INIT) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init(); }
})();
